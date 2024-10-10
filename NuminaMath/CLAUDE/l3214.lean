import Mathlib

namespace bus_full_after_twelve_stops_l3214_321405

/-- The number of seats in the bus -/
def bus_seats : ℕ := 78

/-- The function representing the total number of passengers after n stops -/
def total_passengers (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that 12 is the smallest positive integer that fills the bus -/
theorem bus_full_after_twelve_stops :
  (∀ k : ℕ, k > 0 → k < 12 → total_passengers k < bus_seats) ∧
  total_passengers 12 = bus_seats :=
sorry

end bus_full_after_twelve_stops_l3214_321405


namespace price_calculation_l3214_321493

/-- Calculates the total price for jewelry and paintings after a price increase -/
def total_price (
  original_jewelry_price : ℕ
  ) (original_painting_price : ℕ
  ) (jewelry_price_increase : ℕ
  ) (painting_price_increase_percent : ℕ
  ) (jewelry_quantity : ℕ
  ) (painting_quantity : ℕ
  ) : ℕ :=
  let new_jewelry_price := original_jewelry_price + jewelry_price_increase
  let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percent) / 100
  (new_jewelry_price * jewelry_quantity) + (new_painting_price * painting_quantity)

theorem price_calculation :
  total_price 30 100 10 20 2 5 = 680 := by
  sorry

end price_calculation_l3214_321493


namespace banana_pear_difference_l3214_321451

/-- Represents a bowl of fruit with apples, pears, and bananas. -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Properties of the fruit bowl -/
def is_valid_fruit_bowl (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas > bowl.pears ∧
  bowl.apples + bowl.pears + bowl.bananas = 19 ∧
  bowl.bananas = 9

theorem banana_pear_difference (bowl : FruitBowl) 
  (h : is_valid_fruit_bowl bowl) : 
  bowl.bananas - bowl.pears = 3 := by
  sorry

end banana_pear_difference_l3214_321451


namespace angle_measure_l3214_321483

/-- An angle in degrees satisfies the given condition if its supplement is four times its complement -/
theorem angle_measure (x : ℝ) : (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_l3214_321483


namespace power_of_two_equation_l3214_321455

theorem power_of_two_equation (k : ℤ) : 
  2^1998 - 2^1997 - 2^1996 + 2^1995 = k * 2^1995 → k = 3 := by
  sorry

end power_of_two_equation_l3214_321455


namespace isosceles_right_triangle_line_equation_l3214_321494

/-- A line that forms an isosceles right-angled triangle with the coordinate axes -/
structure IsoscelesRightTriangleLine where
  a : ℝ
  eq : (x y : ℝ) → Prop
  passes_through : eq 2 3
  isosceles_right : ∀ (x y : ℝ), eq x y → (x / a + y / a = 1) ∨ (x / a + y / (-a) = 1)

/-- The equation of the line is either x + y - 5 = 0 or x - y + 1 = 0 -/
theorem isosceles_right_triangle_line_equation (l : IsoscelesRightTriangleLine) :
  (∀ x y, l.eq x y ↔ x + y - 5 = 0) ∨ (∀ x y, l.eq x y ↔ x - y + 1 = 0) := by
  sorry

end isosceles_right_triangle_line_equation_l3214_321494


namespace tan_x_equals_negative_seven_l3214_321428

theorem tan_x_equals_negative_seven (x : ℝ) 
  (h1 : Real.sin (x + π/4) = 3/5)
  (h2 : Real.sin (x - π/4) = 4/5) : 
  Real.tan x = -7 := by
  sorry

end tan_x_equals_negative_seven_l3214_321428


namespace five_letter_words_same_ends_l3214_321467

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- The number of freely chosen letters in each word --/
def free_letters : ℕ := word_length - 2

/-- The number of five-letter words with the same first and last letter --/
def count_words : ℕ := alphabet_size ^ (free_letters + 1)

theorem five_letter_words_same_ends :
  count_words = 456976 := by sorry

end five_letter_words_same_ends_l3214_321467


namespace complete_square_quadratic_l3214_321440

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (r s : ℝ), 16 * x^2 + 32 * x - 2048 = 0 ↔ (x + r)^2 = s ∧ s = 129 :=
by sorry

end complete_square_quadratic_l3214_321440


namespace range_of_a_l3214_321449

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x^2 - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, y ≥ -(1 / Real.exp 1) → ∃ x : ℝ, f a x = y) →
  (∀ x : ℝ, f a x ≥ -(1 / Real.exp 1)) →
  a ≥ Real.exp 1 :=
by sorry

end range_of_a_l3214_321449


namespace trigonometric_problem_l3214_321446

theorem trigonometric_problem (α β : Real) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h_distance : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 10 / 5)
  (h_tan : Real.tan (α/2) = 1/2) :
  Real.cos (α - β) = 4/5 ∧ Real.cos α = 3/5 ∧ Real.cos β = 24/25 := by
sorry

end trigonometric_problem_l3214_321446


namespace nested_circles_radius_l3214_321460

theorem nested_circles_radius (A₁ A₂ : ℝ) : 
  A₁ > 0 → 
  A₂ > 0 → 
  (∃ d : ℝ, A₂ = A₁ + d ∧ A₁ + 2*A₂ = A₂ + d) → 
  A₁ + 2*A₂ = π * 5^2 → 
  ∃ r : ℝ, r > 0 ∧ A₁ = π * r^2 ∧ r = Real.sqrt 5 :=
sorry

end nested_circles_radius_l3214_321460


namespace exponential_decreasing_zero_two_l3214_321406

theorem exponential_decreasing_zero_two (m n : ℝ) : m > n → (0.2 : ℝ) ^ m < (0.2 : ℝ) ^ n := by
  sorry

end exponential_decreasing_zero_two_l3214_321406


namespace not_multiple_of_121_l3214_321471

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := by
  sorry

end not_multiple_of_121_l3214_321471


namespace no_real_roots_for_nonzero_k_l3214_321476

theorem no_real_roots_for_nonzero_k (k : ℝ) (hk : k ≠ 0) :
  ∀ x : ℝ, x^2 + 2*k*x + 3*k^2 ≠ 0 := by
  sorry

end no_real_roots_for_nonzero_k_l3214_321476


namespace family_weight_problem_l3214_321473

/-- Given a family with a grandmother, daughter, and child, prove their weights satisfy certain conditions and the combined weight of the daughter and child is 60 kg. -/
theorem family_weight_problem (grandmother daughter child : ℝ) : 
  grandmother + daughter + child = 110 →
  child = (1 / 5) * grandmother →
  daughter = 50 →
  daughter + child = 60 := by
sorry

end family_weight_problem_l3214_321473


namespace truck_speed_l3214_321490

/-- Calculates the speed of a truck in kilometers per hour -/
theorem truck_speed (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 10) :
  (distance / time) * 3.6 = 216 := by
  sorry

#check truck_speed

end truck_speed_l3214_321490


namespace x_one_minus_f_equals_one_l3214_321413

/-- Proves that for x = (3 + √5)^20, n = ⌊x⌋, and f = x - n, x(1 - f) = 1 -/
theorem x_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 5) ^ 20
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end x_one_minus_f_equals_one_l3214_321413


namespace cafeteria_pies_l3214_321477

/-- Given a cafeteria with total apples, apples handed out, and apples needed per pie,
    calculate the number of pies that can be made. -/
def pies_made (total_apples handed_out apples_per_pie : ℕ) : ℕ :=
  (total_apples - handed_out) / apples_per_pie

/-- Theorem: The cafeteria can make 9 pies with the given conditions. -/
theorem cafeteria_pies :
  pies_made 525 415 12 = 9 := by
  sorry

end cafeteria_pies_l3214_321477


namespace halfway_fraction_l3214_321436

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 :=
sorry

end halfway_fraction_l3214_321436


namespace sarahs_pastry_flour_l3214_321404

def rye_flour : ℕ := 5
def wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def total_flour : ℕ := 20

def pastry_flour : ℕ := total_flour - (rye_flour + wheat_bread_flour + chickpea_flour)

theorem sarahs_pastry_flour : pastry_flour = 2 := by
  sorry

end sarahs_pastry_flour_l3214_321404


namespace midpoint_value_l3214_321426

/-- Given two distinct points (m, n) and (p, q) on the curve x^2 - 5xy + 2y^2 + 7x - 6y + 3 = 0,
    where (m + 2, n + k) is the midpoint of the line segment connecting (m, n) and (p, q),
    and the line passing through (m, n) and (p, q) has the equation x - 5y + 1 = 0,
    prove that k = 2/5. -/
theorem midpoint_value (m n p q k : ℝ) : 
  (m ≠ p ∨ n ≠ q) →
  m^2 - 5*m*n + 2*n^2 + 7*m - 6*n + 3 = 0 →
  p^2 - 5*p*q + 2*q^2 + 7*p - 6*q + 3 = 0 →
  m + 2 = (m + p) / 2 →
  n + k = (n + q) / 2 →
  m - 5*n + 1 = 0 →
  p - 5*q + 1 = 0 →
  k = 2/5 := by
  sorry

end midpoint_value_l3214_321426


namespace league_games_l3214_321488

theorem league_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end league_games_l3214_321488


namespace prob_more_than_4_draws_is_31_35_l3214_321411

-- Define the number of new and old coins
def new_coins : ℕ := 3
def old_coins : ℕ := 4
def total_coins : ℕ := new_coins + old_coins

-- Define the probability function
noncomputable def prob_more_than_4_draws : ℚ :=
  1 - (
    -- Probability of drawing all new coins in first 3 draws
    (new_coins / total_coins) * ((new_coins - 1) / (total_coins - 1)) * ((new_coins - 2) / (total_coins - 2)) +
    -- Probability of drawing all new coins in first 4 draws (but not in first 3)
    3 * ((old_coins / total_coins) * (new_coins / (total_coins - 1)) * ((new_coins - 1) / (total_coins - 2)) * ((new_coins - 2) / (total_coins - 3)))
  )

-- Theorem statement
theorem prob_more_than_4_draws_is_31_35 : prob_more_than_4_draws = 31 / 35 :=
  sorry

end prob_more_than_4_draws_is_31_35_l3214_321411


namespace calculation_proof_l3214_321427

theorem calculation_proof : ((-1/3)⁻¹ : ℝ) - (Real.sqrt 3 - 2)^0 + 4 * Real.cos (π/4) = -4 + 2 * Real.sqrt 2 := by
  sorry

end calculation_proof_l3214_321427


namespace grasshopper_theorem_l3214_321442

/-- Represents the order of grasshoppers -/
inductive GrasshopperOrder
  | Even
  | Odd

/-- Represents a single jump of a grasshopper -/
def jump (order : GrasshopperOrder) : GrasshopperOrder :=
  match order with
  | GrasshopperOrder.Even => GrasshopperOrder.Odd
  | GrasshopperOrder.Odd => GrasshopperOrder.Even

/-- Represents multiple jumps of grasshoppers -/
def multipleJumps (initialOrder : GrasshopperOrder) (n : Nat) : GrasshopperOrder :=
  match n with
  | 0 => initialOrder
  | Nat.succ m => jump (multipleJumps initialOrder m)

theorem grasshopper_theorem :
  multipleJumps GrasshopperOrder.Even 1999 = GrasshopperOrder.Odd :=
by sorry

end grasshopper_theorem_l3214_321442


namespace pizza_area_difference_l3214_321458

theorem pizza_area_difference : ∃ (N : ℝ), 
  (abs (N - 96) < 1) ∧ 
  (π * 7^2 = π * 5^2 * (1 + N / 100)) := by
  sorry

end pizza_area_difference_l3214_321458


namespace probability_sum_5_l3214_321447

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (fun p => p.1 < p.2)

def sum_5_pairs : Finset (ℕ × ℕ) := valid_pairs.filter (fun p => p.1 + p.2 = 5)

theorem probability_sum_5 : 
  (sum_5_pairs.card : ℚ) / valid_pairs.card = 1 / 5 := by sorry

end probability_sum_5_l3214_321447


namespace solve_potatoes_problem_l3214_321431

def potatoes_problem (initial : ℕ) (gina : ℕ) : Prop :=
  let tom := 2 * gina
  let anne := tom / 3
  let remaining := initial - (gina + tom + anne)
  remaining = 47

theorem solve_potatoes_problem :
  potatoes_problem 300 69 := by
  sorry

end solve_potatoes_problem_l3214_321431


namespace compute_expression_l3214_321421

theorem compute_expression : 20 * (180 / 3 + 40 / 5 + 16 / 32 + 2) = 1410 := by
  sorry

end compute_expression_l3214_321421


namespace set_operations_l3214_321480

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {3, 4, 5}

theorem set_operations :
  (A ∪ B = {1, 3, 4, 5, 7, 9}) ∧
  (A ∩ B = {3, 5}) ∧
  ({x | x ∈ A ∧ x ∉ B} = {1, 7, 9}) := by
  sorry

end set_operations_l3214_321480


namespace cubic_sum_theorem_l3214_321474

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 + 10) / p = (q^3 + 10) / q ∧ (q^3 + 10) / q = (r^3 + 10) / r) : 
  p^3 + q^3 + r^3 = -30 := by
  sorry

end cubic_sum_theorem_l3214_321474


namespace line_through_point_and_intersection_l3214_321401

/-- The line passing through P(2, -3) and the intersection of two given lines -/
theorem line_through_point_and_intersection :
  let P : ℝ × ℝ := (2, -3)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x + 2 * y - 4
  let line2 : ℝ → ℝ → ℝ := λ x y => x - y + 5
  let result_line : ℝ → ℝ → ℝ := λ x y => 3.4 * x + 1.6 * y - 2
  -- The result line passes through P
  (result_line P.1 P.2 = 0) ∧
  -- The result line passes through the intersection point of line1 and line2
  (∃ x y : ℝ, line1 x y = 0 ∧ line2 x y = 0 ∧ result_line x y = 0) :=
by
  sorry

end line_through_point_and_intersection_l3214_321401


namespace min_attempts_eq_n_l3214_321470

/-- Represents a binary code of length n -/
def BinaryCode (n : ℕ) := Fin n → Bool

/-- Feedback from an attempt -/
inductive Feedback
| NoClick
| Click
| Open

/-- Function representing an attempt to open the safe -/
def attempt (n : ℕ) (secretCode : BinaryCode n) (tryCode : BinaryCode n) : Feedback :=
  sorry

/-- Minimum number of attempts required to open the safe -/
def minAttempts (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of attempts is n -/
theorem min_attempts_eq_n (n : ℕ) : minAttempts n = n :=
  sorry

end min_attempts_eq_n_l3214_321470


namespace fixed_point_exponential_function_l3214_321418

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 := by sorry

end fixed_point_exponential_function_l3214_321418


namespace f_values_l3214_321409

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

theorem f_values : f 2 = 14 ∧ f (-2) = 2 := by sorry

end f_values_l3214_321409


namespace x_seventh_minus_27x_squared_l3214_321478

theorem x_seventh_minus_27x_squared (x : ℝ) (h : x^3 - 3*x = 6) :
  x^7 - 27*x^2 = 9*(x + 1)*(x + 6) := by
  sorry

end x_seventh_minus_27x_squared_l3214_321478


namespace division_problem_l3214_321452

theorem division_problem (A : ℕ) : A = 8 ↔ 41 = 5 * A + 1 := by sorry

end division_problem_l3214_321452


namespace mollys_current_age_l3214_321414

/-- Represents the ages of Sandy and Molly -/
structure Ages where
  sandy : ℕ
  molly : ℕ

/-- The ratio of Sandy's age to Molly's age is 4:3 -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.molly = 3 * ages.sandy

/-- Sandy will be 42 years old after 6 years -/
def sandy_future_age (ages : Ages) : Prop :=
  ages.sandy + 6 = 42

theorem mollys_current_age (ages : Ages) 
  (h1 : age_ratio ages) 
  (h2 : sandy_future_age ages) : 
  ages.molly = 27 := by
  sorry

end mollys_current_age_l3214_321414


namespace factorization_problem_1_factorization_problem_2_l3214_321432

-- Problem 1
theorem factorization_problem_1 (a b x : ℝ) :
  x^2 * (a - b) + 4 * (b - a) = (a - b) * (x + 2) * (x - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  -a^3 + 6 * a^2 * b - 9 * a * b^2 = -a * (a - 3 * b)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l3214_321432


namespace sequence_equality_l3214_321469

/-- Given two sequences of real numbers (xₙ) and (yₙ) defined as follows:
    x₁ = y₁ = 1
    xₙ₊₁ = (xₙ + 2) / (xₙ + 1)
    yₙ₊₁ = (yₙ² + 2) / (2yₙ)
    Prove that yₙ₊₁ = x₂ⁿ holds for n = 0, 1, 2, ... -/
theorem sequence_equality (x y : ℕ → ℝ) 
    (hx1 : x 1 = 1)
    (hy1 : y 1 = 1)
    (hx : ∀ n : ℕ, x (n + 1) = (x n + 2) / (x n + 1))
    (hy : ∀ n : ℕ, y (n + 1) = (y n ^ 2 + 2) / (2 * y n)) :
  ∀ n : ℕ, y (n + 1) = x (2 ^ n) := by
  sorry


end sequence_equality_l3214_321469


namespace solution_set_l3214_321481

theorem solution_set (x y : ℝ) : 
  x - 2*y = 1 → x^3 - 8*y^3 - 6*x*y = 1 → y = (x-1)/2 := by
  sorry

end solution_set_l3214_321481


namespace building_height_percentage_l3214_321499

theorem building_height_percentage (L M R : ℝ) : 
  M = 100 → 
  R = L + M - 20 → 
  L + M + R = 340 → 
  L / M * 100 = 80 := by
sorry

end building_height_percentage_l3214_321499


namespace work_duration_l3214_321434

/-- Given two workers p and q, where p can complete a job in 15 days and q in 20 days,
    this theorem proves that if 0.5333333333333333 of the job remains after they work
    together for d days, then d must equal 4. -/
theorem work_duration (p q d : ℝ) : 
  p = 1 / 15 →
  q = 1 / 20 →
  1 - (p + q) * d = 0.5333333333333333 →
  d = 4 := by
  sorry


end work_duration_l3214_321434


namespace vehicle_license_count_l3214_321417

/-- The number of possible letters for a license -/
def num_letters : ℕ := 3

/-- The number of possible digits for each position in a license -/
def num_digits : ℕ := 10

/-- The number of digit positions in a license -/
def num_digit_positions : ℕ := 6

/-- The total number of possible vehicle licenses -/
def total_licenses : ℕ := num_letters * (num_digits ^ num_digit_positions)

theorem vehicle_license_count :
  total_licenses = 3000000 := by sorry

end vehicle_license_count_l3214_321417


namespace distance_blown_westward_is_200km_l3214_321403

/-- Represents the journey of a ship -/
structure ShipJourney where
  speed : ℝ
  travelTime : ℝ
  totalDistance : ℝ
  finalPosition : ℝ

/-- Calculates the distance blown westward by the storm -/
def distanceBlownWestward (journey : ShipJourney) : ℝ :=
  journey.speed * journey.travelTime - journey.finalPosition

/-- Theorem stating the distance blown westward is 200 km -/
theorem distance_blown_westward_is_200km (journey : ShipJourney) 
  (h1 : journey.speed = 30)
  (h2 : journey.travelTime = 20)
  (h3 : journey.speed * journey.travelTime = journey.totalDistance / 2)
  (h4 : journey.finalPosition = journey.totalDistance / 3) :
  distanceBlownWestward journey = 200 := by
  sorry

#check distance_blown_westward_is_200km

end distance_blown_westward_is_200km_l3214_321403


namespace silk_order_total_l3214_321430

/-- The number of yards of green silk dyed by the factory -/
def green_silk : ℕ := 61921

/-- The number of yards of pink silk dyed by the factory -/
def pink_silk : ℕ := 49500

/-- The total number of yards of silk dyed by the factory -/
def total_silk : ℕ := green_silk + pink_silk

theorem silk_order_total :
  total_silk = 111421 :=
by sorry

end silk_order_total_l3214_321430


namespace cricket_game_initial_overs_l3214_321489

/-- Prove that the number of overs played initially is 10, given the conditions of the cricket game. -/
theorem cricket_game_initial_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 6.25 →
  remaining_overs = 40 →
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
    target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry

end cricket_game_initial_overs_l3214_321489


namespace expected_balls_in_original_position_l3214_321407

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 8

/-- The number of independent transpositions -/
def num_transpositions : ℕ := 3

/-- The probability that a specific ball is chosen in any swap -/
def prob_chosen : ℚ := 1 / 4

/-- The probability that a ball is not chosen in a single swap -/
def prob_not_chosen : ℚ := 1 - prob_chosen

/-- The probability that a ball is in its original position after all transpositions -/
def prob_original_position : ℚ := prob_not_chosen ^ num_transpositions + 
  num_transpositions * prob_chosen ^ 2 * prob_not_chosen

/-- The expected number of balls in their original positions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

theorem expected_balls_in_original_position :
  expected_original_positions = 9 / 2 := by sorry

end expected_balls_in_original_position_l3214_321407


namespace max_shoe_pairs_l3214_321450

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 20 → lost_shoes = 9 → max_pairs = 11 →
  max_pairs = initial_pairs - lost_shoes ∧ 
  max_pairs * 2 + lost_shoes ≤ initial_pairs * 2 :=
by sorry

end max_shoe_pairs_l3214_321450


namespace negation_of_universal_proposition_l3214_321472

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 4*x + 4 ≥ 0) ↔ (∃ x : ℝ, x^2 - 4*x + 4 < 0) :=
by sorry

end negation_of_universal_proposition_l3214_321472


namespace sqrt_equation_root_l3214_321443

theorem sqrt_equation_root :
  ∃ x : ℝ, (Real.sqrt x + Real.sqrt (x + 2) = 12) ∧ (x = 5041 / 144) := by
  sorry

end sqrt_equation_root_l3214_321443


namespace expression_simplification_l3214_321435

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -3) :
  (a^2 - 9) / (a^2 + 6*a + 9) / ((a - 3) / (a^2 + 3*a)) - (a - a^2) / (a - 1) = 2*a :=
by sorry

end expression_simplification_l3214_321435


namespace triangle_inequality_l3214_321444

theorem triangle_inequality (a b c m : ℝ) (γ : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < m) (h5 : 0 < γ) (h6 : γ < π) : 
  a + b + m ≤ ((2 + Real.cos (γ / 2)) / (2 * Real.sin (γ / 2))) * c := by
  sorry

end triangle_inequality_l3214_321444


namespace fruit_distribution_l3214_321491

/-- The number of ways to distribute n identical items among k distinct recipients --/
def distribute_identical (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute m distinct items among k distinct recipients --/
def distribute_distinct (m k : ℕ) : ℕ := k^m

theorem fruit_distribution :
  let apples : ℕ := 6
  let distinct_fruits : ℕ := 3  -- orange, plum, tangerine
  let people : ℕ := 3
  distribute_identical apples people * distribute_distinct distinct_fruits people = 756 := by
sorry

end fruit_distribution_l3214_321491


namespace inequality_abc_l3214_321439

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) ≤ 1 ∧
  ((a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_abc_l3214_321439


namespace derivative_positive_solution_set_l3214_321497

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

def solution_set : Set ℝ := Set.Ioi 2

theorem derivative_positive_solution_set :
  ∀ x > 0, x ∈ solution_set ↔ deriv f x > 0 :=
sorry

end derivative_positive_solution_set_l3214_321497


namespace relative_complement_M_N_l3214_321410

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem relative_complement_M_N : (M \ N) = {-1, 0, 3} := by
  sorry

end relative_complement_M_N_l3214_321410


namespace cube_root_of_negative_twenty_seven_l3214_321465

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cube_root_of_negative_twenty_seven :
  cubeRoot (-27) = -3 := by sorry

end cube_root_of_negative_twenty_seven_l3214_321465


namespace binomial_probability_equals_eight_twentyseven_l3214_321424

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (dist : BinomialDistribution) (k : ℕ) : ℝ :=
  (dist.n.choose k) * (dist.p ^ k) * ((1 - dist.p) ^ (dist.n - k))

theorem binomial_probability_equals_eight_twentyseven :
  let ξ : BinomialDistribution := ⟨4, 1/3, by norm_num, by norm_num⟩
  binomialPMF ξ 2 = 8/27 := by
  sorry

end binomial_probability_equals_eight_twentyseven_l3214_321424


namespace max_tasty_compote_weight_l3214_321438

theorem max_tasty_compote_weight 
  (fresh_apples : ℝ) 
  (dried_apples : ℝ) 
  (fresh_water_content : ℝ) 
  (dried_water_content : ℝ) 
  (max_water_content : ℝ) :
  fresh_apples = 4 →
  dried_apples = 1 →
  fresh_water_content = 0.9 →
  dried_water_content = 0.12 →
  max_water_content = 0.95 →
  ∃ (max_compote : ℝ),
    max_compote = 25.6 ∧
    ∀ (added_water : ℝ),
      (fresh_apples * fresh_water_content + 
       dried_apples * dried_water_content + 
       added_water) / 
      (fresh_apples + dried_apples + added_water) ≤ max_water_content →
      fresh_apples + dried_apples + added_water ≤ max_compote :=
by sorry

end max_tasty_compote_weight_l3214_321438


namespace third_derivative_y_l3214_321422

noncomputable def y (x : ℝ) : ℝ := (Real.log (3 + x)) / (3 + x)

theorem third_derivative_y (x : ℝ) (h : x ≠ -3) : 
  (deriv^[3] y) x = (11 - 6 * Real.log (3 + x)) / (3 + x)^4 :=
by sorry

end third_derivative_y_l3214_321422


namespace constant_term_value_l3214_321454

theorem constant_term_value (x y z : ℤ) (k : ℤ) 
  (eq1 : 4 * x + y + z = 80)
  (eq2 : 2 * x - y - z = 40)
  (eq3 : 3 * x + y - z = k)
  (h_x : x = 20) : k = 60 := by
  sorry

end constant_term_value_l3214_321454


namespace solution_for_x_and_y_l3214_321487

theorem solution_for_x_and_y (a x y : Real) (k : Int) (h1 : x + y = a) (h2 : Real.sin x ^ 2 + Real.sin y ^ 2 = 1 - Real.cos a) (h3 : Real.cos a ≠ 0) :
  x = a / 2 + k * Real.pi ∧ y = a / 2 - k * Real.pi :=
by sorry

end solution_for_x_and_y_l3214_321487


namespace parabola_directrix_equation_l3214_321402

/-- Represents a parabola in the form y^2 = 4px --/
structure Parabola where
  p : ℝ

/-- The directrix of a parabola --/
def directrix (para : Parabola) : ℝ := -para.p

theorem parabola_directrix_equation :
  let para : Parabola := ⟨1⟩
  directrix para = -1 := by
  sorry

end parabola_directrix_equation_l3214_321402


namespace largest_root_range_l3214_321475

theorem largest_root_range (b₀ b₁ b₂ : ℝ) 
  (h₀ : |b₀| < 3) (h₁ : |b₁| < 3) (h₂ : |b₂| < 3) :
  ∃ r : ℝ, 3.5 < r ∧ r < 5 ∧
  (∀ x : ℝ, x > 0 → x^4 + x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ r) ∧
  (∃ x : ℝ, x > 0 ∧ x^4 + x^3 + b₂*x^2 + b₁*x + b₀ = 0 ∧ x = r) :=
by sorry

end largest_root_range_l3214_321475


namespace stream_speed_l3214_321466

theorem stream_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (downstream_wind : ℝ) 
  (upstream_wind : ℝ) 
  (h1 : downstream_distance = 110) 
  (h2 : upstream_distance = 85) 
  (h3 : downstream_time = 5) 
  (h4 : upstream_time = 6) 
  (h5 : downstream_wind = 3) 
  (h6 : upstream_wind = 2) : 
  ∃ (boat_speed stream_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed + downstream_wind) * downstream_time ∧ 
    upstream_distance = (boat_speed - stream_speed + upstream_wind) * upstream_time ∧ 
    stream_speed = 3.4 := by
  sorry

end stream_speed_l3214_321466


namespace face_mask_profit_l3214_321445

/-- Calculates the profit from selling face masks given the following conditions:
  * 12 boxes of face masks were bought at $9 per box
  * Each box contains 50 masks
  * 6 boxes were repacked and sold at $5 per 25 pieces
  * The remaining 300 masks were sold in baggies at 10 pieces for $3
-/
def calculate_profit : ℤ :=
  let boxes_bought := 12
  let cost_per_box := 9
  let masks_per_box := 50
  let repacked_boxes := 6
  let price_per_25_pieces := 5
  let remaining_masks := 300
  let price_per_10_pieces := 3

  let total_cost := boxes_bought * cost_per_box
  let revenue_repacked := repacked_boxes * (masks_per_box / 25) * price_per_25_pieces
  let revenue_baggies := (remaining_masks / 10) * price_per_10_pieces
  let total_revenue := revenue_repacked + revenue_baggies
  
  total_revenue - total_cost

/-- Theorem stating that the profit from selling face masks under the given conditions is $42 -/
theorem face_mask_profit : calculate_profit = 42 := by
  sorry

end face_mask_profit_l3214_321445


namespace parrots_per_cage_l3214_321495

theorem parrots_per_cage (num_cages : ℝ) (parakeets_per_cage : ℝ) (total_birds : ℕ) :
  num_cages = 6 →
  parakeets_per_cage = 2 →
  total_birds = 48 →
  ∃ parrots_per_cage : ℕ, 
    (parrots_per_cage : ℝ) * num_cages + parakeets_per_cage * num_cages = total_birds ∧
    parrots_per_cage = 6 := by
  sorry

end parrots_per_cage_l3214_321495


namespace fraction_simplification_l3214_321492

theorem fraction_simplification :
  1 / (1 / (1/2)^1 + 1 / (1/2)^2 + 1 / (1/2)^3) = 1 / 14 := by sorry

end fraction_simplification_l3214_321492


namespace unique_solution_equation_l3214_321412

theorem unique_solution_equation (x : ℝ) (h : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end unique_solution_equation_l3214_321412


namespace expression_values_l3214_321457

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 := by
sorry

end expression_values_l3214_321457


namespace puzzle_arrangement_count_l3214_321419

/-- The number of letters in the word "puzzle" -/
def n : ℕ := 6

/-- The number of times the letter "z" appears in "puzzle" -/
def z_count : ℕ := 2

/-- The number of distinct arrangements of the letters in "puzzle" -/
def puzzle_arrangements : ℕ := n.factorial / z_count.factorial

theorem puzzle_arrangement_count : puzzle_arrangements = 360 := by
  sorry

end puzzle_arrangement_count_l3214_321419


namespace journalism_club_arrangement_l3214_321437

/-- The number of students in the arrangement -/
def num_students : ℕ := 5

/-- The number of teachers in the arrangement -/
def num_teachers : ℕ := 2

/-- The number of possible positions for the teacher pair -/
def teacher_pair_positions : ℕ := num_students - 1

/-- The total number of arrangements -/
def total_arrangements : ℕ := num_students.factorial * (teacher_pair_positions * num_teachers.factorial)

/-- Theorem stating that the total number of arrangements is 960 -/
theorem journalism_club_arrangement :
  total_arrangements = 960 := by sorry

end journalism_club_arrangement_l3214_321437


namespace scientific_notation_of_9560000_l3214_321482

theorem scientific_notation_of_9560000 :
  9560000 = 9.56 * (10 : ℝ) ^ 6 :=
by sorry

end scientific_notation_of_9560000_l3214_321482


namespace equation_solution_l3214_321468

theorem equation_solution :
  ∃! x : ℝ, (9 - 3*x) * (3^x) - (x - 2) * (x^2 - 5*x + 6) = 0 ∧ x = 3 :=
by sorry

end equation_solution_l3214_321468


namespace expression_value_l3214_321400

theorem expression_value : (3^2 - 5*3 + 6) / (3 - 2) = 0 := by
  sorry

end expression_value_l3214_321400


namespace polar_eq_of_cartesian_line_l3214_321415

/-- The polar coordinate equation ρ cos θ = 1 represents the line x = 1 in Cartesian coordinates -/
theorem polar_eq_of_cartesian_line (ρ θ : ℝ) :
  (ρ * Real.cos θ = 1) ↔ (ρ * Real.cos θ = 1) :=
by sorry

end polar_eq_of_cartesian_line_l3214_321415


namespace dylan_ice_cube_trays_l3214_321453

/-- The number of ice cube trays Dylan needs to fill -/
def num_trays_to_fill (glass_cubes : ℕ) (pitcher_multiplier : ℕ) (tray_capacity : ℕ) : ℕ :=
  ((glass_cubes + glass_cubes * pitcher_multiplier) + tray_capacity - 1) / tray_capacity

/-- Theorem stating that Dylan needs to fill 2 ice cube trays -/
theorem dylan_ice_cube_trays : 
  num_trays_to_fill 8 2 12 = 2 := by
  sorry

#eval num_trays_to_fill 8 2 12

end dylan_ice_cube_trays_l3214_321453


namespace min_value_expression_l3214_321496

theorem min_value_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end min_value_expression_l3214_321496


namespace intersection_of_M_and_N_l3214_321433

-- Define the sets M and N
def M : Set ℝ := {x | ∃ t : ℝ, x = 2^t}
def N : Set ℝ := {x | ∃ t : ℝ, x = Real.sin t}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 0 1 := by sorry

end intersection_of_M_and_N_l3214_321433


namespace wilson_children_ages_l3214_321484

theorem wilson_children_ages (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_youngest : a = 4) (h_middle : b = 7) (h_average : (a + b + c) / 3 = 7) :
  c = 10 := by
  sorry

end wilson_children_ages_l3214_321484


namespace stating_number_of_regions_correct_l3214_321461

/-- 
Given n lines in a plane where no two lines are parallel and no three lines are concurrent,
this function returns the number of regions the plane is divided into.
-/
def number_of_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- 
Theorem stating that n lines in a plane, with no two lines parallel and no three lines concurrent,
divide the plane into (n(n+1)/2) + 1 regions.
-/
theorem number_of_regions_correct (n : ℕ) : 
  number_of_regions n = n * (n + 1) / 2 + 1 := by
  sorry

end stating_number_of_regions_correct_l3214_321461


namespace black_pens_count_l3214_321448

theorem black_pens_count (total_pens blue_pens : ℕ) 
  (h1 : total_pens = 8)
  (h2 : blue_pens = 4) :
  total_pens - blue_pens = 4 := by
  sorry

end black_pens_count_l3214_321448


namespace coconut_grove_solution_l3214_321462

-- Define the problem parameters
def coconut_grove (x : ℝ) : Prop :=
  -- (x + 3) trees yield 60 nuts per year
  ∃ (yield1 : ℝ), yield1 = 60 * (x + 3) ∧
  -- x trees yield 120 nuts per year
  ∃ (yield2 : ℝ), yield2 = 120 * x ∧
  -- (x - 3) trees yield 180 nuts per year
  ∃ (yield3 : ℝ), yield3 = 180 * (x - 3) ∧
  -- The average yield per year per tree is 100
  (yield1 + yield2 + yield3) / (3 * x) = 100

-- Theorem stating that x = 6 is the unique solution
theorem coconut_grove_solution :
  ∃! x : ℝ, coconut_grove x ∧ x = 6 :=
sorry

end coconut_grove_solution_l3214_321462


namespace chocolate_comparison_l3214_321420

theorem chocolate_comparison 
  (robert_chocolates : ℕ)
  (robert_price : ℚ)
  (nickel_chocolates : ℕ)
  (nickel_discount : ℚ)
  (h1 : robert_chocolates = 7)
  (h2 : robert_price = 2)
  (h3 : nickel_chocolates = 5)
  (h4 : nickel_discount = 1.5)
  (h5 : robert_chocolates * robert_price = nickel_chocolates * (robert_price - nickel_discount)) :
  ∃ (n : ℕ), (robert_price * robert_chocolates) / (robert_price - nickel_discount) - robert_chocolates = n ∧ n = 21 := by
  sorry

end chocolate_comparison_l3214_321420


namespace water_added_to_container_l3214_321456

/-- The amount of water added to fill a container from 30% to 3/4 full -/
theorem water_added_to_container (capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) 
  (h1 : capacity = 100)
  (h2 : initial_fraction = 0.3)
  (h3 : final_fraction = 3/4) :
  final_fraction * capacity - initial_fraction * capacity = 45 :=
by sorry

end water_added_to_container_l3214_321456


namespace one_belt_one_road_values_road_line_equation_l3214_321485

/-- Definition of "one belt, one road" relationship -/
def one_belt_one_road (a b c m n : ℝ) : Prop :=
  ∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ y = m * x + 1 ∧
  (∃ (x₀ : ℝ), a * x₀^2 + b * x₀ + c = m * x₀ + 1 ∧
   ∀ (x : ℝ), a * x^2 + b * x + c ≥ m * x + 1)

/-- Theorem for part 1 -/
theorem one_belt_one_road_values :
  one_belt_one_road 1 (-2) n (-1) 1 :=
sorry

/-- Theorem for part 2 -/
theorem road_line_equation (m n : ℝ) :
  (∃ (x : ℝ), m * (x + 1)^2 - 6 = 6 / x) ∧
  (∀ (x : ℝ), m * (x + 1)^2 - 6 ≥ 2 * x - 4) →
  m = 2 ∨ m = -2/3 :=
sorry

end one_belt_one_road_values_road_line_equation_l3214_321485


namespace eighth_term_of_geometric_sequence_l3214_321416

/-- Given a geometric sequence with first term 3 and second term 9/2,
    prove that the eighth term is 6561/128 -/
theorem eighth_term_of_geometric_sequence (a : ℕ → ℚ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 9/2)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (a 2 / a 1)) :
  a 8 = 6561/128 := by
sorry

end eighth_term_of_geometric_sequence_l3214_321416


namespace zero_subset_integers_negation_squared_positive_l3214_321429

-- Define the set containing only 0
def zero_set : Set ℤ := {0}

-- Statement 1: {0} is a subset of ℤ
theorem zero_subset_integers : zero_set ⊆ Set.univ := by sorry

-- Statement 2: Negation of "for all x in ℤ, x² > 0" is "there exists x in ℤ such that x² ≤ 0"
theorem negation_squared_positive :
  (¬ ∀ x : ℤ, x^2 > 0) ↔ (∃ x : ℤ, x^2 ≤ 0) := by sorry

end zero_subset_integers_negation_squared_positive_l3214_321429


namespace arithmetic_sequence_formula_l3214_321463

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 2) 
  (h_a7 : a 7 = 10) : 
  ∀ n : ℕ, a n = 2 * n - 4 := by
sorry

end arithmetic_sequence_formula_l3214_321463


namespace min_value_3x_plus_9y_l3214_321423

theorem min_value_3x_plus_9y (x y : ℝ) (h : x + 2 * y = 2) :
  3 * x + 9 * y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, x₀ + 2 * y₀ = 2 ∧ 3 * x₀ + 9 * y₀ = 6 := by
  sorry

end min_value_3x_plus_9y_l3214_321423


namespace golden_ratio_greater_than_three_fifths_l3214_321498

theorem golden_ratio_greater_than_three_fifths : (Real.sqrt 5 - 1) / 2 > 3 / 5 := by
  sorry

end golden_ratio_greater_than_three_fifths_l3214_321498


namespace flower_percentages_l3214_321464

def total_flowers : ℕ := 30
def red_flowers : ℕ := 7
def white_flowers : ℕ := 6
def blue_flowers : ℕ := 5
def yellow_flowers : ℕ := 4

def purple_flowers : ℕ := total_flowers - (red_flowers + white_flowers + blue_flowers + yellow_flowers)

def percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

theorem flower_percentages :
  (percentage (red_flowers + white_flowers + blue_flowers) total_flowers = 60) ∧
  (percentage purple_flowers total_flowers = 26.67) ∧
  (percentage yellow_flowers total_flowers = 13.33) :=
by sorry

end flower_percentages_l3214_321464


namespace contrapositive_odd_sum_even_l3214_321459

def is_odd (n : ℤ) : Prop := ∃ k, n = 2*k + 1

def is_even (n : ℤ) : Prop := ∃ k, n = 2*k

theorem contrapositive_odd_sum_even :
  (∀ a b : ℤ, (is_odd a ∧ is_odd b) → is_even (a + b)) ↔
  (∀ a b : ℤ, ¬is_even (a + b) → ¬(is_odd a ∧ is_odd b)) :=
sorry

end contrapositive_odd_sum_even_l3214_321459


namespace largest_n_for_factorization_l3214_321408

theorem largest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ a b : ℤ, 5 * x^2 + n * x + 48 = (5 * x + a) * (x + b)) → 
  n ≤ 241 :=
by sorry

end largest_n_for_factorization_l3214_321408


namespace negative_sqrt_eleven_squared_l3214_321486

theorem negative_sqrt_eleven_squared : (-Real.sqrt 11)^2 = 11 := by
  sorry

end negative_sqrt_eleven_squared_l3214_321486


namespace field_width_l3214_321479

/-- A rectangular field with length 7/5 of its width and perimeter 336 meters has a width of 70 meters -/
theorem field_width (w : ℝ) (h1 : w > 0) : 
  2 * (7/5 * w + w) = 336 → w = 70 := by
  sorry

end field_width_l3214_321479


namespace skew_lines_definition_l3214_321425

-- Define a type for lines in 3D space
def Line3D : Type := ℝ × ℝ × ℝ → Prop

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define the property of two lines intersecting
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line3D) : Prop :=
  ¬(parallel l1 l2) ∧ ¬(intersect l1 l2)

-- Theorem stating the definition of skew lines
theorem skew_lines_definition (l1 l2 : Line3D) :
  skew l1 l2 ↔ (¬(parallel l1 l2) ∧ ¬(intersect l1 l2)) := by sorry

end skew_lines_definition_l3214_321425


namespace gloria_purchase_l3214_321441

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions from the problem -/
def store_conditions (p : StorePrices) : Prop :=
  p.pencil + p.notebook = 80 ∧
  p.pencil + p.eraser = 45 ∧
  3 * p.pencil + 3 * p.notebook + 3 * p.eraser = 315

/-- The theorem to prove -/
theorem gloria_purchase (p : StorePrices) : 
  store_conditions p → p.notebook + p.eraser = 85 := by
  sorry

end gloria_purchase_l3214_321441
