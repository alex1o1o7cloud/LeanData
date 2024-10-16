import Mathlib

namespace NUMINAMATH_CALUDE_total_gift_cost_l2394_239469

def engagement_ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def diamond_bracelet_cost : ℕ := 2 * engagement_ring_cost

theorem total_gift_cost : engagement_ring_cost + car_cost + diamond_bracelet_cost = 14000 := by
  sorry

end NUMINAMATH_CALUDE_total_gift_cost_l2394_239469


namespace NUMINAMATH_CALUDE_females_wearing_glasses_l2394_239451

/-- In a town with a given population, number of males, and percentage of females wearing glasses,
    calculate the number of females wearing glasses. -/
theorem females_wearing_glasses
  (total_population : ℕ)
  (males : ℕ)
  (female_glasses_percentage : ℚ)
  (h1 : total_population = 5000)
  (h2 : males = 2000)
  (h3 : female_glasses_percentage = 30 / 100) :
  (total_population - males) * female_glasses_percentage = 900 := by
sorry

end NUMINAMATH_CALUDE_females_wearing_glasses_l2394_239451


namespace NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l2394_239443

/-- The volume of a cube inscribed in a sphere of radius R -/
theorem volume_cube_inscribed_sphere (R : ℝ) (R_pos : 0 < R) :
  ∃ (V : ℝ), V = (8 / 9) * Real.sqrt 3 * R^3 :=
sorry

end NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l2394_239443


namespace NUMINAMATH_CALUDE_fraction_calculation_l2394_239419

theorem fraction_calculation (N : ℝ) (F : ℝ) (h1 : N = 90) 
  (h2 : 3 + (1/2) * (1/3) * (1/5) * N = F * N) : F = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2394_239419


namespace NUMINAMATH_CALUDE_freds_allowance_l2394_239453

def weekly_allowance (A x y : ℝ) : Prop :=
  -- Fred spent half of his allowance on movie tickets
  let movie_cost := A / 2
  -- Lunch cost y dollars less than the cost of the tickets
  let lunch_cost := x - y
  -- He earned 6 dollars from washing the car and 5 dollars from mowing the lawn
  let earned := 6 + 5
  -- At the end of the day, he had 20 dollars
  movie_cost + lunch_cost + earned + (A - movie_cost - lunch_cost) = 20

theorem freds_allowance :
  ∃ A x y : ℝ, weekly_allowance A x y ∧ A = 9 := by sorry

end NUMINAMATH_CALUDE_freds_allowance_l2394_239453


namespace NUMINAMATH_CALUDE_all_divisible_by_nine_l2394_239449

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem all_divisible_by_nine (n : ℕ) 
  (h1 : is_five_digit n) 
  (h2 : digit_sum n = 30) : 
  n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_divisible_by_nine_l2394_239449


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2394_239409

/-- Given two regular polygons with the same perimeter, where the first has 38 sides
    and a side length twice that of the second, prove the second has 76 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) :
  s > 0 →
  38 * (2 * s) = n * s →
  n = 76 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2394_239409


namespace NUMINAMATH_CALUDE_cubic_tangent_line_problem_l2394_239485

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line
    at x = 1 passes through the point (2, 7), then a = 1. -/
theorem cubic_tangent_line_problem (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 1
  let tangent_line : ℝ → ℝ := λ x ↦ f 1 + f' 1 * (x - 1)
  tangent_line 2 = 7 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_tangent_line_problem_l2394_239485


namespace NUMINAMATH_CALUDE_sally_initial_cards_l2394_239472

def initial_cards : ℕ := 27
def cards_from_dan : ℕ := 41
def cards_bought : ℕ := 20
def total_cards : ℕ := 88

theorem sally_initial_cards : 
  initial_cards + cards_from_dan + cards_bought = total_cards :=
by sorry

end NUMINAMATH_CALUDE_sally_initial_cards_l2394_239472


namespace NUMINAMATH_CALUDE_democrat_ratio_l2394_239479

/-- Proves that the ratio of democrats to total participants is 1:3 given the specified conditions -/
theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 990 →
  female_democrats = 165 →
  (∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    2 * female_democrats = female_participants ∧
    4 * female_democrats = male_participants) →
  (3 : ℚ) * (female_democrats + female_democrats) = total_participants := by
  sorry


end NUMINAMATH_CALUDE_democrat_ratio_l2394_239479


namespace NUMINAMATH_CALUDE_pet_shop_hamsters_l2394_239470

theorem pet_shop_hamsters (total : ℕ) (kittens : ℕ) (birds : ℕ) 
  (h1 : total = 77)
  (h2 : kittens = 32)
  (h3 : birds = 30)
  : total - kittens - birds = 15 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_hamsters_l2394_239470


namespace NUMINAMATH_CALUDE_polygon_area_is_400_l2394_239444

/-- The area of a right triangle -/
def rightTriangleArea (base height : ℝ) : ℝ := 0.5 * base * height

/-- The area of a trapezoid -/
def trapezoidArea (shortBase longBase height : ℝ) : ℝ := 0.5 * (shortBase + longBase) * height

/-- The total area of the polygon -/
def polygonArea (triangleBase triangleHeight trapezoidShortBase trapezoidLongBase trapezoidHeight : ℝ) : ℝ :=
  2 * rightTriangleArea triangleBase triangleHeight + 
  2 * trapezoidArea trapezoidShortBase trapezoidLongBase trapezoidHeight

theorem polygon_area_is_400 :
  polygonArea 10 10 10 20 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_400_l2394_239444


namespace NUMINAMATH_CALUDE_combined_pencil_length_l2394_239452

-- Define the length of a pencil in cubes
def pencil_length : ℕ := 12

-- Define the number of pencils
def num_pencils : ℕ := 2

-- Theorem: The combined length of two pencils is 24 cubes
theorem combined_pencil_length :
  num_pencils * pencil_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_combined_pencil_length_l2394_239452


namespace NUMINAMATH_CALUDE_factorization_proof_l2394_239421

theorem factorization_proof (x y m n : ℝ) : 
  x^2 * (m - n) + y^2 * (n - m) = (m - n) * (x + y) * (x - y) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l2394_239421


namespace NUMINAMATH_CALUDE_count_factors_l2394_239410

/-- The number of distinct, whole-number factors of 3^5 * 5^3 * 7^2 -/
def num_factors : ℕ := 72

/-- The prime factorization of the number -/
def prime_factorization : List (ℕ × ℕ) := [(3, 5), (5, 3), (7, 2)]

/-- Theorem stating that the number of distinct, whole-number factors of 3^5 * 5^3 * 7^2 is 72 -/
theorem count_factors : 
  (List.prod (prime_factorization.map (fun (p, e) => e + 1))) = num_factors := by
  sorry

end NUMINAMATH_CALUDE_count_factors_l2394_239410


namespace NUMINAMATH_CALUDE_total_potatoes_to_cook_l2394_239478

/-- Given a cooking scenario where:
  * 6 potatoes are already cooked
  * Each potato takes 8 minutes to cook
  * It takes 72 minutes to cook the remaining potatoes
  Prove that the total number of potatoes to be cooked is 15. -/
theorem total_potatoes_to_cook (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) :
  already_cooked = 6 →
  cooking_time_per_potato = 8 →
  remaining_cooking_time = 72 →
  already_cooked + (remaining_cooking_time / cooking_time_per_potato) = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_potatoes_to_cook_l2394_239478


namespace NUMINAMATH_CALUDE_max_xy_value_l2394_239467

theorem max_xy_value (x y : ℕ+) (h : 5 * x + 3 * y = 100) : x * y ≤ 165 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2394_239467


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l2394_239441

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_values (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l2394_239441


namespace NUMINAMATH_CALUDE_fraction_identification_l2394_239411

-- Define what a fraction is
def is_fraction (x : ℚ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = n / d

-- Define the given expressions
def expr1 (a : ℚ) : ℚ := 2 / a
def expr2 (a : ℚ) : ℚ := 2 * a / 3
def expr3 (b : ℚ) : ℚ := -b / 2
def expr4 (a : ℚ) : ℚ := (3 * a + 1) / 2

-- State the theorem
theorem fraction_identification (a b : ℚ) (ha : a ≠ 0) : 
  is_fraction (expr1 a) ∧ 
  ¬is_fraction (expr2 a) ∧ 
  ¬is_fraction (expr3 b) ∧ 
  ¬is_fraction (expr4 a) :=
sorry

end NUMINAMATH_CALUDE_fraction_identification_l2394_239411


namespace NUMINAMATH_CALUDE_cannot_determine_best_method_l2394_239487

/-- Represents an investment method --/
inductive InvestmentMethod
  | OneYear
  | ThreeYearThenOneYear
  | FiveOneYearThenFiveYear

/-- Calculates the final amount for a given investment method --/
def calculateFinalAmount (method : InvestmentMethod) (initialAmount : ℝ) : ℝ :=
  match method with
  | .OneYear => initialAmount * (1 + 0.0156) ^ 10
  | .ThreeYearThenOneYear => initialAmount * (1 + 0.0206 * 3) ^ 3 * (1 + 0.0156)
  | .FiveOneYearThenFiveYear => initialAmount * (1 + 0.0156) ^ 5 * (1 + 0.0282 * 5)

/-- Theorem stating that the best investment method cannot be determined without calculation --/
theorem cannot_determine_best_method (initialAmount : ℝ) :
  ∀ (m1 m2 : InvestmentMethod), m1 ≠ m2 →
  ∃ (result1 result2 : ℝ),
    calculateFinalAmount m1 initialAmount = result1 ∧
    calculateFinalAmount m2 initialAmount = result2 ∧
    (result1 > result2 ∨ result1 < result2) :=
by
  sorry

#check cannot_determine_best_method

end NUMINAMATH_CALUDE_cannot_determine_best_method_l2394_239487


namespace NUMINAMATH_CALUDE_constant_value_proof_l2394_239415

theorem constant_value_proof :
  ∀ (t : ℝ) (constant : ℝ),
    let x := 1 - 2 * t
    let y := constant * t - 2
    (t = 0.75 → x = y) →
    constant = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l2394_239415


namespace NUMINAMATH_CALUDE_binary_addition_and_predecessor_l2394_239437

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  sorry

theorem binary_addition_and_predecessor :
  let M : List Bool := [false, true, true, true, false, true]
  let M_plus_5 : List Bool := [true, true, false, false, true, true]
  let M_plus_5_pred : List Bool := [false, true, false, false, true, true]
  (binary_to_decimal M) + 5 = binary_to_decimal M_plus_5 ∧
  binary_to_decimal M_plus_5 - 1 = binary_to_decimal M_plus_5_pred :=
by
  sorry

#check binary_addition_and_predecessor

end NUMINAMATH_CALUDE_binary_addition_and_predecessor_l2394_239437


namespace NUMINAMATH_CALUDE_marble_distribution_l2394_239414

theorem marble_distribution (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) :
  angela = a ∧ 
  brian = 2 * a ∧ 
  caden = 6 * a ∧ 
  daryl = 42 * a ∧
  angela + brian + caden + daryl = 126 →
  a = 42 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l2394_239414


namespace NUMINAMATH_CALUDE_at_least_four_same_probability_l2394_239499

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling a specific value on a single die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability that all five dice show the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def fourSameProbability : ℚ :=
  (numDice : ℚ) * (singleDieProbability ^ (numDice - 2)) * (1 - singleDieProbability)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_probability :
  allSameProbability + fourSameProbability = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_at_least_four_same_probability_l2394_239499


namespace NUMINAMATH_CALUDE_buddy_card_count_l2394_239424

def card_count (initial : ℕ) : ℕ := 
  let tuesday := initial - (initial * 30 / 100)
  let wednesday := tuesday + (tuesday * 20 / 100)
  let thursday := wednesday - (wednesday * 25 / 100)
  let friday := thursday + (thursday / 3)
  let saturday := friday + (friday * 2)
  let sunday := saturday + (saturday * 40 / 100) - 15
  let next_monday := sunday + ((saturday * 40 / 100) * 3)
  next_monday

theorem buddy_card_count : card_count 200 = 1297 := by
  sorry

end NUMINAMATH_CALUDE_buddy_card_count_l2394_239424


namespace NUMINAMATH_CALUDE_mark_kate_difference_l2394_239440

/-- The number of hours Kate charged to the project -/
def kate_hours : ℕ := 28

/-- The total number of hours charged to the project -/
def total_hours : ℕ := 180

/-- Pat's hours are twice Kate's -/
def pat_hours : ℕ := 2 * kate_hours

/-- Mark's hours are three times Kate's -/
def mark_hours : ℕ := 3 * kate_hours

/-- Linda's hours are half of Kate's -/
def linda_hours : ℕ := kate_hours / 2

theorem mark_kate_difference :
  mark_hours - kate_hours = 56 ∧
  pat_hours + kate_hours + mark_hours + linda_hours = total_hours :=
by sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l2394_239440


namespace NUMINAMATH_CALUDE_mean_problem_l2394_239413

theorem mean_problem (x y : ℝ) : 
  (28 + x + 42 + y + 78 + 104) / 6 = 62 → 
  x + y = 120 ∧ (x + y) / 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l2394_239413


namespace NUMINAMATH_CALUDE_three_valid_configurations_l2394_239408

/-- Represents a square in the configuration --/
structure Square :=
  (label : Char)

/-- Represents the F-shaped configuration --/
def FConfiguration : Finset Square := sorry

/-- The set of additional lettered squares --/
def AdditionalSquares : Finset Square := sorry

/-- Predicate to check if a configuration is valid (foldable into a cube with one open non-bottom side) --/
def IsValidConfiguration (config : Finset Square) : Prop := sorry

/-- The number of valid configurations --/
def ValidConfigurationsCount : ℕ := sorry

/-- Theorem stating that there are exactly 3 valid configurations --/
theorem three_valid_configurations :
  ValidConfigurationsCount = 3 := by sorry

end NUMINAMATH_CALUDE_three_valid_configurations_l2394_239408


namespace NUMINAMATH_CALUDE_port_vessels_l2394_239422

theorem port_vessels (cruise_ships cargo_ships sailboats fishing_boats : ℕ) :
  cruise_ships = 4 →
  cargo_ships = 2 * cruise_ships →
  ∃ (x : ℕ), sailboats = cargo_ships + x →
  sailboats = 7 * fishing_boats →
  cruise_ships + cargo_ships + sailboats + fishing_boats = 28 →
  sailboats - cargo_ships = 6 := by
  sorry

end NUMINAMATH_CALUDE_port_vessels_l2394_239422


namespace NUMINAMATH_CALUDE_track_width_l2394_239426

theorem track_width (r : ℝ) 
  (h1 : 2 * π * (2 * r) - 2 * π * r = 16 * π) 
  (h2 : 2 * r - r = r) : r = 8 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l2394_239426


namespace NUMINAMATH_CALUDE_chinese_dinner_cost_l2394_239406

theorem chinese_dinner_cost (num_people : ℕ) (num_appetizers : ℕ) (appetizer_cost : ℚ)
  (tip_percentage : ℚ) (rush_fee : ℚ) (total_spent : ℚ) :
  num_people = 4 →
  num_appetizers = 2 →
  appetizer_cost = 6 →
  tip_percentage = 0.2 →
  rush_fee = 5 →
  total_spent = 77 →
  ∃ (main_meal_cost : ℚ),
    main_meal_cost * num_people +
    num_appetizers * appetizer_cost +
    (main_meal_cost * num_people + num_appetizers * appetizer_cost) * tip_percentage +
    rush_fee = total_spent ∧
    main_meal_cost = 12 :=
by sorry

end NUMINAMATH_CALUDE_chinese_dinner_cost_l2394_239406


namespace NUMINAMATH_CALUDE_least_common_solution_l2394_239483

theorem least_common_solution : ∃ b : ℕ, b > 0 ∧ 
  b % 7 = 6 ∧ 
  b % 11 = 10 ∧ 
  b % 13 = 12 ∧
  (∀ c : ℕ, c > 0 ∧ c % 7 = 6 ∧ c % 11 = 10 ∧ c % 13 = 12 → b ≤ c) ∧
  b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_least_common_solution_l2394_239483


namespace NUMINAMATH_CALUDE_terms_before_ten_l2394_239450

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_ten (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 105 ∧ d = -5 →
  arithmetic_sequence a₁ d 20 = 10 ∧
  ∀ k : ℕ, k < 20 → arithmetic_sequence a₁ d k > 10 :=
by sorry

end NUMINAMATH_CALUDE_terms_before_ten_l2394_239450


namespace NUMINAMATH_CALUDE_simon_fraction_of_alvin_age_l2394_239466

/-- Given that Alvin is 30 years old and Simon is 10 years old, prove that Simon will be 3/7 of Alvin's age in 5 years. -/
theorem simon_fraction_of_alvin_age (alvin_age : ℕ) (simon_age : ℕ) : 
  alvin_age = 30 → simon_age = 10 → (simon_age + 5 : ℚ) / (alvin_age + 5 : ℚ) = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_simon_fraction_of_alvin_age_l2394_239466


namespace NUMINAMATH_CALUDE_ant_beetle_distance_difference_l2394_239464

/-- Calculates the percentage difference in distance traveled between an ant and a beetle -/
theorem ant_beetle_distance_difference :
  let ant_distance : ℝ := 600  -- meters
  let ant_time : ℝ := 12       -- minutes
  let beetle_speed : ℝ := 2.55 -- km/h
  
  let ant_speed : ℝ := (ant_distance / 1000) / (ant_time / 60)
  let beetle_distance : ℝ := (beetle_speed * ant_time) / 60 * 1000
  
  let difference : ℝ := ant_distance - beetle_distance
  let percentage_difference : ℝ := (difference / ant_distance) * 100
  
  percentage_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_ant_beetle_distance_difference_l2394_239464


namespace NUMINAMATH_CALUDE_ship_speeds_l2394_239494

theorem ship_speeds (t : ℝ) (d : ℝ) (speed_diff : ℝ) :
  t = 2 →
  d = 174 →
  speed_diff = 3 →
  ∃ (x : ℝ),
    x > 0 ∧
    (x + speed_diff) > 0 ∧
    (t * x)^2 + (t * (x + speed_diff))^2 = d^2 →
    (x = 60 ∧ x + speed_diff = 63) :=
by sorry

end NUMINAMATH_CALUDE_ship_speeds_l2394_239494


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l2394_239475

/-- Represents the typing service rates and manuscript details --/
structure ManuscriptTyping where
  total_pages : Nat
  initial_cost : Nat
  first_revision_cost : Nat
  second_revision_cost : Nat
  subsequent_revision_cost : Nat
  pages_revised_once : Nat
  pages_revised_twice : Nat
  pages_revised_thrice : Nat
  pages_revised_four_times : Nat
  pages_revised_five_times : Nat

/-- Calculates the total cost of typing and revising a manuscript --/
def total_typing_cost (m : ManuscriptTyping) : Nat :=
  m.total_pages * m.initial_cost +
  m.pages_revised_once * m.first_revision_cost +
  m.pages_revised_twice * (m.first_revision_cost + m.second_revision_cost) +
  m.pages_revised_thrice * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost) +
  m.pages_revised_four_times * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost * 2) +
  m.pages_revised_five_times * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost * 3)

/-- Theorem stating that the total cost for the given manuscript is $5750 --/
theorem manuscript_typing_cost :
  let m := ManuscriptTyping.mk 400 10 8 6 4 60 40 20 10 5
  total_typing_cost m = 5750 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l2394_239475


namespace NUMINAMATH_CALUDE_h_bounds_l2394_239427

/-- The probability that the distance between two randomly chosen points on (0,1) is less than h -/
def probability (h : ℝ) : ℝ := h * (2 - h)

/-- Theorem stating the bounds of h given the probability constraints -/
theorem h_bounds (h : ℝ) (h_pos : 0 < h) (h_lt_one : h < 1) 
  (prob_lower : 1/4 < probability h) (prob_upper : probability h < 3/4) : 
  1/2 - Real.sqrt 3 / 2 < h ∧ h < 1/2 + Real.sqrt 3 / 2 := by
  sorry

#check h_bounds

end NUMINAMATH_CALUDE_h_bounds_l2394_239427


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2394_239439

theorem smallest_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2394_239439


namespace NUMINAMATH_CALUDE_animal_sightings_proof_l2394_239430

/-- The number of times families see animals in January -/
def january_sightings : ℕ := 26

/-- The number of times families see animals in February -/
def february_sightings : ℕ := 3 * january_sightings

/-- The number of times families see animals in March -/
def march_sightings : ℕ := february_sightings / 2

/-- The total number of times families see animals in the first three months -/
def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem animal_sightings_proof : total_sightings = 143 := by
  sorry

end NUMINAMATH_CALUDE_animal_sightings_proof_l2394_239430


namespace NUMINAMATH_CALUDE_kennel_long_furred_dogs_l2394_239497

/-- Represents the number of dogs with a certain property in a kennel -/
structure DogCount where
  total : ℕ
  brown : ℕ
  neither_long_nor_brown : ℕ

/-- Calculates the number of long-furred dogs in the kennel -/
def long_furred_dogs (d : DogCount) : ℕ :=
  d.total - d.neither_long_nor_brown - d.brown

/-- Theorem stating that in a kennel with the given properties, there are 10 long-furred dogs -/
theorem kennel_long_furred_dogs :
  let d : DogCount := ⟨45, 27, 8⟩
  long_furred_dogs d = 10 := by
  sorry

#eval long_furred_dogs ⟨45, 27, 8⟩

end NUMINAMATH_CALUDE_kennel_long_furred_dogs_l2394_239497


namespace NUMINAMATH_CALUDE_exists_max_a_l2394_239482

def is_valid_number (a d e : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 8 ∧
  (500000 + 100000 * a + 1000 * d + 500 + 20 + 4 + e) % 24 = 0

theorem exists_max_a : ∃ (d e : ℕ), is_valid_number 9 d e :=
sorry

end NUMINAMATH_CALUDE_exists_max_a_l2394_239482


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l2394_239474

theorem quadratic_inequality_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) →
  (0 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l2394_239474


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2394_239416

/-- For an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 64 times the sum of the series with the first four terms removed,
    then r = 1/2 -/
theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r)) = 64 * (a * r^4 / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2394_239416


namespace NUMINAMATH_CALUDE_scientific_notation_42000_l2394_239495

theorem scientific_notation_42000 :
  (42000 : ℝ) = 4.2 * (10 : ℝ) ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_42000_l2394_239495


namespace NUMINAMATH_CALUDE_pollution_data_median_mode_l2394_239455

def pollution_data : List ℕ := [31, 35, 31, 34, 30, 32, 31]

def median (l : List ℕ) : ℕ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem pollution_data_median_mode : 
  median pollution_data = 31 ∧ mode pollution_data = 31 := by sorry

end NUMINAMATH_CALUDE_pollution_data_median_mode_l2394_239455


namespace NUMINAMATH_CALUDE_average_problem_l2394_239465

theorem average_problem (x : ℝ) : (15 + 25 + 35 + x) / 4 = 30 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2394_239465


namespace NUMINAMATH_CALUDE_correct_calculation_l2394_239480

theorem correct_calculation (x : ℤ) (h : x + 44 - 39 = 63) : (x + 39) - 44 = 53 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2394_239480


namespace NUMINAMATH_CALUDE_hall_volume_l2394_239433

/-- A rectangular hall with specific dimensions and area properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  area_equality : 2 * (length * width) = 2 * (length * height) + 2 * (width * height)

/-- The volume of a rectangular hall with the given properties is 972 cubic meters -/
theorem hall_volume (hall : RectangularHall) 
  (h_length : hall.length = 18)
  (h_width : hall.width = 9) : 
  hall.length * hall.width * hall.height = 972 := by
  sorry

end NUMINAMATH_CALUDE_hall_volume_l2394_239433


namespace NUMINAMATH_CALUDE_pebble_ratio_l2394_239428

/-- Prove that the ratio of pebbles Lance threw to pebbles Candy threw is 3:1 -/
theorem pebble_ratio : 
  let candy_pebbles : ℕ := 4
  let lance_pebbles : ℕ := candy_pebbles + 8
  (lance_pebbles : ℚ) / (candy_pebbles : ℚ) = 3 := by sorry

end NUMINAMATH_CALUDE_pebble_ratio_l2394_239428


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2394_239432

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, sum n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- The common difference of an arithmetic sequence is 2 if 2S₃ = 3S₂ + 6 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : 2 * seq.sum 3 = 3 * seq.sum 2 + 6) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2394_239432


namespace NUMINAMATH_CALUDE_box_volume_calculation_l2394_239434

/-- The conversion factor from feet to meters -/
def feet_to_meters : ℝ := 0.3048

/-- The edge length of each box in feet -/
def edge_length_feet : ℝ := 5

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The total volume of the boxes in cubic meters -/
def total_volume : ℝ := 14.144

theorem box_volume_calculation :
  (num_boxes : ℝ) * (edge_length_feet * feet_to_meters)^3 = total_volume := by
  sorry

end NUMINAMATH_CALUDE_box_volume_calculation_l2394_239434


namespace NUMINAMATH_CALUDE_point_P_coordinates_and_PQ_length_l2394_239459

def point_P (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3*n)

def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

def point_Q (n : ℝ) : ℝ × ℝ := (n, -4)

def parallel_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

theorem point_P_coordinates_and_PQ_length :
  ∃ n : ℝ,
    let p := point_P n
    let q := point_Q n
    fourth_quadrant p ∧
    distance_to_x_axis p = distance_to_y_axis p + 1 ∧
    parallel_to_x_axis p q ∧
    p = (6, -7) ∧
    |p.1 - q.1| = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_and_PQ_length_l2394_239459


namespace NUMINAMATH_CALUDE_concave_arithmetic_sequence_condition_l2394_239404

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

/-- A sequence is concave if a_{n-1} + a_{n+1} ≥ 2a_n for n ≥ 2 -/
def is_concave (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n - 1) + a (n + 1) ≥ 2 * a n

theorem concave_arithmetic_sequence_condition (d : ℝ) :
  let b := arithmetic_sequence 4 d
  is_concave (λ n => b n / n) → d ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_concave_arithmetic_sequence_condition_l2394_239404


namespace NUMINAMATH_CALUDE_trees_needed_for_road_l2394_239493

/-- The number of trees needed to plant on one side of a road -/
def num_trees (road_length : ℕ) (interval : ℕ) : ℕ :=
  road_length / interval + 1

/-- Theorem: The number of trees needed for a 1500m road with 25m intervals is 61 -/
theorem trees_needed_for_road : num_trees 1500 25 = 61 := by
  sorry

end NUMINAMATH_CALUDE_trees_needed_for_road_l2394_239493


namespace NUMINAMATH_CALUDE_shoe_discount_percentage_l2394_239484

def shoe_price : ℝ := 200
def shirts_price : ℝ := 160
def final_discount : ℝ := 0.05
def final_amount : ℝ := 285

theorem shoe_discount_percentage :
  ∃ (x : ℝ), 
    (shoe_price * (1 - x / 100) + shirts_price) * (1 - final_discount) = final_amount ∧
    x = 30 :=
  sorry

end NUMINAMATH_CALUDE_shoe_discount_percentage_l2394_239484


namespace NUMINAMATH_CALUDE_fish_per_black_duck_is_ten_l2394_239431

/-- Represents the number of fish per duck for each duck color -/
structure FishPerDuck where
  white : ℕ
  multicolor : ℕ

/-- Represents the number of ducks for each color -/
structure DuckCounts where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- Calculates the number of fish per black duck -/
def fishPerBlackDuck (fpd : FishPerDuck) (dc : DuckCounts) (totalFish : ℕ) : ℚ :=
  let fishForWhite := fpd.white * dc.white
  let fishForMulticolor := fpd.multicolor * dc.multicolor
  let fishForBlack := totalFish - fishForWhite - fishForMulticolor
  (fishForBlack : ℚ) / dc.black

theorem fish_per_black_duck_is_ten :
  let fpd : FishPerDuck := { white := 5, multicolor := 12 }
  let dc : DuckCounts := { white := 3, black := 7, multicolor := 6 }
  let totalFish : ℕ := 157
  fishPerBlackDuck fpd dc totalFish = 10 := by
  sorry

end NUMINAMATH_CALUDE_fish_per_black_duck_is_ten_l2394_239431


namespace NUMINAMATH_CALUDE_solution_set_l2394_239490

theorem solution_set (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | a^(2*x - 7) > a^(4*x - 1)} = {x : ℝ | x > -3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l2394_239490


namespace NUMINAMATH_CALUDE_wilsons_theorem_l2394_239438

theorem wilsons_theorem (N : ℕ) (h : N > 1) :
  (Nat.factorial (N - 1) % N = N - 1) ↔ Nat.Prime N := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l2394_239438


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l2394_239457

/-- The number of arrangements of 5 people where 3 specific people maintain their relative order but are not adjacent -/
def photo_arrangements : ℕ := 20

/-- The number of ways to choose 2 positions from 5 available positions -/
def choose_two_from_five : ℕ := 20

theorem photo_arrangement_count :
  photo_arrangements = choose_two_from_five :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l2394_239457


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2394_239460

/-- Given real numbers a, b, and c, and polynomials g and f as defined,
    prove that f(-1) = -29041 -/
theorem polynomial_evaluation (a b c : ℝ) : 
  let g := fun (x : ℝ) => x^3 + a*x^2 + x + 20
  let f := fun (x : ℝ) => x^4 + x^3 + b*x^2 + 200*x + c
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  f (-1) = -29041 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2394_239460


namespace NUMINAMATH_CALUDE_max_value_interval_l2394_239417

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

theorem max_value_interval (a : ℝ) (h1 : a ≤ 4) :
  (∃ (x : ℝ), x ∈ Set.Ioo (3 - a^2) a ∧
   ∀ (y : ℝ), y ∈ Set.Ioo (3 - a^2) a → f y ≤ f x) →
  Real.sqrt 2 < a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_interval_l2394_239417


namespace NUMINAMATH_CALUDE_average_and_difference_l2394_239420

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 32 → |45 - (y + 5)| = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l2394_239420


namespace NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l2394_239498

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (h : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, h x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The main theorem -/
theorem monic_quartic_value_at_zero 
  (h : ℝ → ℝ) 
  (monic_quartic : MonicQuarticPolynomial h)
  (h_neg_two : h (-2) = -4)
  (h_one : h 1 = -1)
  (h_three : h 3 = -9)
  (h_five : h 5 = -25) : 
  h 0 = -30 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l2394_239498


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2394_239471

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2394_239471


namespace NUMINAMATH_CALUDE_football_players_l2394_239458

theorem football_players (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 35)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 6) :
  total - tennis + both - neither = 26 := by
  sorry

end NUMINAMATH_CALUDE_football_players_l2394_239458


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l2394_239488

/-- Given an integer n and a digit d, returns the number composed of n repetitions of d -/
def repeat_digit (n : ℕ) (d : ℕ) : ℕ := 
  d * (10^n - 1) / 9

/-- Returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_9ab (a b : ℕ) : 
  a = repeat_digit 1985 8 → 
  b = repeat_digit 1985 5 → 
  sum_of_digits (9 * a * b) = 17865 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l2394_239488


namespace NUMINAMATH_CALUDE_right_triangle_circle_properties_l2394_239418

/-- Properties of right triangles relating inscribed and circumscribed circles -/
theorem right_triangle_circle_properties (a b c r R p : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → p > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  p = a + b + c →    -- Perimeter
  r = (a + b - c) / 2 →  -- Inradius formula
  R = c / 2 →        -- Circumradius formula
  (p / c - r / R = 2) ∧
  (r / R ≤ 1 / (Real.sqrt 2 + 1)) ∧
  (r / R = 1 / (Real.sqrt 2 + 1) ↔ a = b) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_circle_properties_l2394_239418


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l2394_239491

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of people to be seated -/
def total_people : ℕ := 10

/-- The number of people with seating restrictions -/
def restricted_people : ℕ := 4

/-- The number of ways to arrange 10 people in a row, where 4 specific people cannot sit in 4 consecutive seats -/
def seating_arrangements : ℕ := 
  factorial total_people - factorial (total_people - restricted_people + 1) * factorial restricted_people

theorem correct_seating_arrangements : seating_arrangements = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l2394_239491


namespace NUMINAMATH_CALUDE_alcohol_concentration_correct_l2394_239468

/-- The concentration of alcohol in the container after n operations --/
def alcohol_concentration (n : ℕ) : ℚ :=
  (12 - 9 * (3/4)^(n-1)) / (32 - 9 * (3/4)^(n-1))

/-- The amount of water in the container after n operations --/
def water_amount (n : ℕ) : ℚ :=
  20/3 * (2/3)^(n-1)

/-- The amount of alcohol in the container after n operations --/
def alcohol_amount (n : ℕ) : ℚ :=
  4 * (2/3)^(n-1) - 6 * (1/2)^n

/-- The theorem stating that the alcohol_concentration function correctly calculates
    the concentration of alcohol in the container after n operations --/
theorem alcohol_concentration_correct (n : ℕ) :
  alcohol_concentration n = alcohol_amount n / (water_amount n + alcohol_amount n) :=
by sorry

/-- The initial amount of water in the container --/
def initial_water : ℚ := 10

/-- The amount of alcohol added in the first step --/
def first_alcohol_addition : ℚ := 1

/-- The amount of alcohol added in the second step --/
def second_alcohol_addition : ℚ := 1/2

/-- The fraction of liquid removed in each step --/
def removal_fraction : ℚ := 1/3

/-- The ratio of alcohol added in each step compared to the previous step --/
def alcohol_addition_ratio : ℚ := 1/2

end NUMINAMATH_CALUDE_alcohol_concentration_correct_l2394_239468


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_and_cubes_l2394_239496

theorem consecutive_integers_sum_of_squares_and_cubes :
  ∀ n : ℤ,
  (n - 1)^2 + n^2 + (n + 1)^2 = 8450 →
  n = 53 ∧ (n - 1)^3 + n^3 + (n + 1)^3 = 446949 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_and_cubes_l2394_239496


namespace NUMINAMATH_CALUDE_special_trapezoid_area_l2394_239454

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The measure of the adjacent angles in degrees -/
  adjacent_angle : ℝ
  /-- The measure of the angle between diagonals facing the base in degrees -/
  diag_angle : ℝ

/-- The area of a special trapezoid -/
noncomputable def area (t : SpecialTrapezoid) : ℝ := sorry

/-- Theorem stating the area of a specific trapezoid is 2 -/
theorem special_trapezoid_area :
  ∀ t : SpecialTrapezoid,
    t.shorter_base = 2 ∧
    t.adjacent_angle = 135 ∧
    t.diag_angle = 150 →
    area t = 2 :=
by sorry

end NUMINAMATH_CALUDE_special_trapezoid_area_l2394_239454


namespace NUMINAMATH_CALUDE_closest_to_70_l2394_239401

def A : ℚ := 254 / 5
def B : ℚ := 400 / 6
def C : ℚ := 492 / 7

def target : ℚ := 70

theorem closest_to_70 :
  |C - target| ≤ |A - target| ∧ |C - target| ≤ |B - target| :=
sorry

end NUMINAMATH_CALUDE_closest_to_70_l2394_239401


namespace NUMINAMATH_CALUDE_similar_triangles_with_two_equal_sides_l2394_239436

theorem similar_triangles_with_two_equal_sides (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  (a = 80 ∧ b = 100) →
  (d = 80 ∧ e = 100) →
  a / d = b / e →
  a / d = c / f →
  b / e = c / f →
  ((c = 64 ∧ f = 125) ∨ (c = 125 ∧ f = 64)) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_with_two_equal_sides_l2394_239436


namespace NUMINAMATH_CALUDE_division_of_fractions_l2394_239412

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l2394_239412


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l2394_239489

/-- Proves that given a mixture with an initial ratio of milk to water of 3:1,
    if adding 5 litres of milk changes the ratio to 4:1,
    then the initial volume of the mixture was 20 litres. -/
theorem mixture_volume_proof (V : ℝ) : 
  (3 / 4 * V) / (1 / 4 * V) = 3 / 1 →  -- Initial ratio of milk to water is 3:1
  ((3 / 4 * V + 5) / (1 / 4 * V) = 4 / 1) →  -- New ratio after adding 5 litres of milk is 4:1
  V = 20 := by  -- Initial volume is 20 litres
sorry

end NUMINAMATH_CALUDE_mixture_volume_proof_l2394_239489


namespace NUMINAMATH_CALUDE_fly_can_always_escape_l2394_239446

/-- Represents a bug (fly or spider) in the octahedron -/
structure Bug where
  position : ℝ × ℝ × ℝ
  speed : ℝ

/-- Represents the octahedron -/
structure Octahedron where
  vertices : List (ℝ × ℝ × ℝ)
  edges : List ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))

/-- Represents the state of the chase -/
structure ChaseState where
  octahedron : Octahedron
  fly : Bug
  spiders : List Bug

/-- Function to determine if the fly can escape -/
def canFlyEscape (state : ChaseState) : Prop :=
  ∃ (nextPosition : ℝ × ℝ × ℝ), nextPosition ∈ state.octahedron.vertices ∧
    ∀ (spider : Bug), spider ∈ state.spiders →
      ‖spider.position - nextPosition‖ > ‖state.fly.position - nextPosition‖ * (spider.speed / state.fly.speed)

/-- The main theorem -/
theorem fly_can_always_escape (r : ℝ) (h : r < 25) :
  ∀ (state : ChaseState),
    state.fly.speed = 50 ∧
    (∀ spider ∈ state.spiders, spider.speed = r) ∧
    state.fly.position ∈ state.octahedron.vertices ∧
    state.spiders.length = 3 →
    canFlyEscape state :=
  sorry

end NUMINAMATH_CALUDE_fly_can_always_escape_l2394_239446


namespace NUMINAMATH_CALUDE_borrowed_amount_l2394_239400

theorem borrowed_amount (P : ℝ) 
  (h1 : (P * 12 / 100 * 3) + (P * 9 / 100 * 5) + (P * 13 / 100 * 3) = 8160) : 
  P = 6800 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l2394_239400


namespace NUMINAMATH_CALUDE_door_crank_time_l2394_239486

/-- Represents the time taken for various parts of the game show challenge -/
structure GameShowTimes where
  firstRun : Nat  -- Time for first run in seconds
  secondRun : Nat -- Time for second run in seconds
  totalTime : Nat -- Total time for the entire event in seconds

/-- Calculates the time taken to crank open the door -/
def timeToCrankDoor (times : GameShowTimes) : Nat :=
  times.totalTime - (times.firstRun + times.secondRun)

/-- Theorem stating that the time to crank open the door is 73 seconds -/
theorem door_crank_time (times : GameShowTimes) 
  (h1 : times.firstRun = 7 * 60 + 23)
  (h2 : times.secondRun = 5 * 60 + 58)
  (h3 : times.totalTime = 874) :
  timeToCrankDoor times = 73 := by
  sorry

#eval timeToCrankDoor { firstRun := 7 * 60 + 23, secondRun := 5 * 60 + 58, totalTime := 874 }

end NUMINAMATH_CALUDE_door_crank_time_l2394_239486


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2394_239429

def second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, α ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 2) (2 * k * Real.pi + Real.pi)

theorem half_angle_quadrant (α : Real) (h : second_quadrant α) :
  ∃ k : ℤ, α / 2 ∈ Set.Ioo (k * Real.pi) (k * Real.pi + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2394_239429


namespace NUMINAMATH_CALUDE_tan_three_expression_value_l2394_239423

theorem tan_three_expression_value (θ : Real) (h : Real.tan θ = 3) :
  2 * (Real.sin θ)^2 - 3 * (Real.sin θ) * (Real.cos θ) - 4 * (Real.cos θ)^2 = -4/10 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_value_l2394_239423


namespace NUMINAMATH_CALUDE_new_mean_after_adding_specific_problem_l2394_239448

theorem new_mean_after_adding (n : ℕ) (original_mean add_value : ℝ) :
  n > 0 →
  let new_mean := (n * original_mean + n * add_value) / n
  new_mean = original_mean + add_value :=
by sorry

theorem specific_problem :
  let n : ℕ := 15
  let original_mean : ℝ := 40
  let add_value : ℝ := 13
  (n * original_mean + n * add_value) / n = 53 :=
by sorry

end NUMINAMATH_CALUDE_new_mean_after_adding_specific_problem_l2394_239448


namespace NUMINAMATH_CALUDE_Z_in_first_quadrant_l2394_239447

def Z : ℂ := Complex.I * (1 - 2 * Complex.I)

theorem Z_in_first_quadrant : 
  Complex.re Z > 0 ∧ Complex.im Z > 0 := by
  sorry

end NUMINAMATH_CALUDE_Z_in_first_quadrant_l2394_239447


namespace NUMINAMATH_CALUDE_asterisk_value_l2394_239473

theorem asterisk_value : ∃ x : ℚ, (63 / 21) * (x / 189) = 1 ∧ x = 63 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_value_l2394_239473


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l2394_239492

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digits_consecutive (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ, n = d₁ * 100 + d₂ * 10 + d₃ ∧
  ((d₁ + 1 = d₂ ∧ d₂ + 1 = d₃) ∨
   (d₁ + 1 = d₃ ∧ d₃ + 1 = d₂) ∨
   (d₂ + 1 = d₁ ∧ d₁ + 1 = d₃) ∨
   (d₂ + 1 = d₃ ∧ d₃ + 1 = d₁) ∨
   (d₃ + 1 = d₁ ∧ d₁ + 1 = d₂) ∨
   (d₃ + 1 = d₂ ∧ d₂ + 1 = d₁))

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_sum_of_digits_of_sum :
  ∀ a b : ℕ, is_three_digit a → is_three_digit b →
  digits_consecutive a → digits_consecutive b →
  ∃ S : ℕ, S = a + b ∧ sum_of_digits S ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l2394_239492


namespace NUMINAMATH_CALUDE_team_supporters_equal_positive_responses_l2394_239463

-- Define the four teams
inductive Team
| Spartak
| Dynamo
| Zenit
| Lokomotiv

-- Define the result of a match
inductive MatchResult
| Win
| Lose

-- Define a function to represent fan behavior
def fanResponse (team : Team) (result : MatchResult) : Bool :=
  match result with
  | MatchResult.Win => true
  | MatchResult.Lose => false

-- Define the theorem
theorem team_supporters_equal_positive_responses 
  (match1 : Team → MatchResult) 
  (match2 : Team → MatchResult)
  (positiveResponses : Team → Nat)
  (h1 : ∀ t, (match1 t = MatchResult.Win) ≠ (match2 t = MatchResult.Win))
  (h2 : positiveResponses Team.Spartak = 200)
  (h3 : positiveResponses Team.Dynamo = 300)
  (h4 : positiveResponses Team.Zenit = 500)
  (h5 : positiveResponses Team.Lokomotiv = 600)
  : ∀ t, positiveResponses t = 
    (if fanResponse t (match1 t) then 1 else 0) + 
    (if fanResponse t (match2 t) then 1 else 0) := by
  sorry


end NUMINAMATH_CALUDE_team_supporters_equal_positive_responses_l2394_239463


namespace NUMINAMATH_CALUDE_smallest_product_l2394_239445

def digits : List ℕ := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, is_valid_arrangement a b c d →
  product a b c d ≥ 3876 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l2394_239445


namespace NUMINAMATH_CALUDE_total_cost_calculation_total_cost_is_832_l2394_239425

/-- Calculate the total cost of sandwiches and sodas with discount and tax -/
theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) 
  (sandwich_discount tax_rate : ℚ) : ℚ :=
  let sandwich_cost := sandwich_price * sandwich_quantity
  let soda_cost := soda_price * soda_quantity
  let discounted_sandwich_cost := sandwich_cost * (1 - sandwich_discount)
  let subtotal := discounted_sandwich_cost + soda_cost
  let total_with_tax := subtotal * (1 + tax_rate)
  total_with_tax

/-- Prove that the total cost is $8.32 given the specific conditions -/
theorem total_cost_is_832 :
  total_cost_calculation 2.44 0.87 2 4 0.15 0.09 = 8.32 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_total_cost_is_832_l2394_239425


namespace NUMINAMATH_CALUDE_tourist_money_theorem_l2394_239461

/-- Represents the amount of money a tourist has at the end of each day -/
def money_after_day (initial_money : ℚ) (day : ℕ) : ℚ :=
  match day with
  | 0 => initial_money
  | n + 1 => (money_after_day initial_money n) / 2 - 100

/-- Theorem stating that if a tourist spends half their money plus 100 Ft each day for 5 days
    and ends up with no money, they must have started with 6200 Ft -/
theorem tourist_money_theorem :
  ∃ (initial_money : ℚ), 
    (money_after_day initial_money 5 = 0) ∧ 
    (initial_money = 6200) :=
by sorry

end NUMINAMATH_CALUDE_tourist_money_theorem_l2394_239461


namespace NUMINAMATH_CALUDE_complement_of_A_l2394_239481

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x - 2 > 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2394_239481


namespace NUMINAMATH_CALUDE_parcel_weight_sum_l2394_239403

theorem parcel_weight_sum (x y z : ℝ) 
  (h1 : x + y = 168) 
  (h2 : y + z = 174) 
  (h3 : x + z = 180) : 
  x + y + z = 261 := by
  sorry

end NUMINAMATH_CALUDE_parcel_weight_sum_l2394_239403


namespace NUMINAMATH_CALUDE_jack_and_beanstalk_height_l2394_239407

/-- The height of the sky island in Jack and the Beanstalk --/
def sky_island_height (day_climb : ℕ) (night_slide : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - 1) * (day_climb - night_slide) + day_climb

theorem jack_and_beanstalk_height :
  sky_island_height 25 3 64 = 1411 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_beanstalk_height_l2394_239407


namespace NUMINAMATH_CALUDE_find_m_value_l2394_239435

theorem find_m_value (a : ℝ) (m : ℝ) : 
  (∀ x, 2*x^2 - 3*x + a < 0 ↔ m < x ∧ x < 1) →
  (2*m^2 - 3*m + a = 0 ∧ 2*1^2 - 3*1 + a = 0) →
  m = 1/2 := by sorry

end NUMINAMATH_CALUDE_find_m_value_l2394_239435


namespace NUMINAMATH_CALUDE_square_area_error_l2394_239477

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.06 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 12.36 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2394_239477


namespace NUMINAMATH_CALUDE_lemonade_syrup_parts_l2394_239442

/-- Given a solution with 8 parts water for every L parts lemonade syrup,
    prove that if removing 2.1428571428571423 parts and replacing with water
    results in 25% lemonade syrup, then L = 2.6666666666666665 -/
theorem lemonade_syrup_parts (L : ℝ) : 
  (L = 0.25 * (8 + L)) → L = 2.6666666666666665 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_syrup_parts_l2394_239442


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l2394_239402

theorem two_digit_numbers_problem :
  ∃ (x y : ℕ), 10 ≤ x ∧ x < y ∧ y < 100 ∧
  (1000 * y + x) % (100 * x + y) = 590 ∧
  (1000 * y + x) / (100 * x + y) = 2 ∧
  2 * y + 3 * x = 72 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l2394_239402


namespace NUMINAMATH_CALUDE_dianes_honey_harvest_l2394_239476

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest (last_year_harvest : ℕ) (increase : ℕ) : 
  last_year_harvest = 2479 → increase = 6085 → last_year_harvest + increase = 8564 := by
  sorry

end NUMINAMATH_CALUDE_dianes_honey_harvest_l2394_239476


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l2394_239456

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l2394_239456


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_neg_one_rational_two_rational_three_rational_l2394_239462

theorem sqrt_three_irrational :
  ∀ (x : ℝ), x ^ 2 = 3 → ¬ (∃ (a b : ℤ), b ≠ 0 ∧ x = a / b) :=
by sorry

theorem neg_one_rational : ∃ (a b : ℤ), b ≠ 0 ∧ -1 = a / b :=
by sorry

theorem two_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 2 = a / b :=
by sorry

theorem three_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 3 = a / b :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_neg_one_rational_two_rational_three_rational_l2394_239462


namespace NUMINAMATH_CALUDE_one_integer_solution_implies_a_range_l2394_239405

theorem one_integer_solution_implies_a_range (a : ℝ) :
  (∃! x : ℤ, (x : ℝ) - a ≥ 0 ∧ 2 * (x : ℝ) - 10 < 0) →
  3 < a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_one_integer_solution_implies_a_range_l2394_239405
