import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3584_358428

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 8) :
  (x + 1/y)^2 + (y + 1/x)^2 ≥ 289/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3584_358428


namespace NUMINAMATH_CALUDE_probability_sum_23_l3584_358424

/-- Represents a 20-faced die with specific numbered faces and one blank face -/
structure SpecialDie :=
  (numbered_faces : List Nat)
  (blank_face : Unit)

/-- Defines Die A with faces 1-18, 20, and one blank -/
def dieA : SpecialDie :=
  { numbered_faces := List.range 18 ++ [20],
    blank_face := () }

/-- Defines Die B with faces 1-7, 9-20, and one blank -/
def dieB : SpecialDie :=
  { numbered_faces := List.range 7 ++ List.range' 9 20,
    blank_face := () }

/-- Calculates the probability of rolling a sum of 23 with two specific dice -/
def probabilitySum23 (d1 d2 : SpecialDie) : Rat :=
  sorry

theorem probability_sum_23 :
  probabilitySum23 dieA dieB = 7 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_23_l3584_358424


namespace NUMINAMATH_CALUDE_alice_minimum_speed_l3584_358410

-- Define the problem parameters
def distance : ℝ := 180
def bob_speed : ℝ := 40
def alice_delay : ℝ := 0.5

-- Define the theorem
theorem alice_minimum_speed :
  ∀ (alice_speed : ℝ),
  alice_speed > distance / (distance / bob_speed - alice_delay) →
  alice_speed * (distance / bob_speed - alice_delay) > distance :=
by sorry

end NUMINAMATH_CALUDE_alice_minimum_speed_l3584_358410


namespace NUMINAMATH_CALUDE_right_triangle_area_l3584_358421

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 2 →
  angle = 45 * (π / 180) →
  let area := (hypotenuse^2 / 4)
  area = 32 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3584_358421


namespace NUMINAMATH_CALUDE_physics_marks_l3584_358495

def marks_english : ℕ := 81
def marks_mathematics : ℕ := 65
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 76
def total_subjects : ℕ := 5

theorem physics_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology
  total_marks - known_marks = 82 := by sorry

end NUMINAMATH_CALUDE_physics_marks_l3584_358495


namespace NUMINAMATH_CALUDE_seed_placement_count_l3584_358413

/-- The number of ways to select k items from n distinct items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n distinct items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of seed placement methods -/
def totalPlacements : ℕ := sorry

theorem seed_placement_count :
  let totalSeeds : ℕ := 10
  let bottleCount : ℕ := 6
  let seedsNotForBottleOne : ℕ := 2
  totalPlacements = choose (totalSeeds - seedsNotForBottleOne) 1 * arrange (totalSeeds - 1) (bottleCount - 1) := by
  sorry

end NUMINAMATH_CALUDE_seed_placement_count_l3584_358413


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3584_358439

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x + Real.arcsin (x^2 * Real.sin (6 / x))
  else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3584_358439


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3584_358475

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = -1 - Complex.I → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3584_358475


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3584_358463

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a < b → a < b + 1) ∧
  (∃ a b : ℝ, a < b + 1 ∧ ¬(a < b)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3584_358463


namespace NUMINAMATH_CALUDE_circumscribing_circle_diameter_l3584_358407

/-- The diameter of a circle circumscribing 8 tangent circles -/
theorem circumscribing_circle_diameter (r : ℝ) (h : r = 5) : 
  let n : ℕ := 8
  let small_circle_radius := r
  let large_circle_diameter := 2 * r * (3 + Real.sqrt 3)
  large_circle_diameter = 10 * (3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circumscribing_circle_diameter_l3584_358407


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3584_358457

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = m + 3 * x) ↔ m = 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3584_358457


namespace NUMINAMATH_CALUDE_unique_consecutive_odd_primes_l3584_358412

theorem unique_consecutive_odd_primes :
  ∀ p q r : ℕ,
  Prime p ∧ Prime q ∧ Prime r →
  p < q ∧ q < r →
  Odd p ∧ Odd q ∧ Odd r →
  q = p + 2 ∧ r = q + 2 →
  p = 3 ∧ q = 5 ∧ r = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_consecutive_odd_primes_l3584_358412


namespace NUMINAMATH_CALUDE_molly_gift_cost_l3584_358458

/-- Represents the cost structure and family composition for Molly's gift-sending scenario -/
structure GiftSendingScenario where
  cost_per_package : ℕ
  num_parents : ℕ
  num_brothers : ℕ
  num_children_per_brother : ℕ

/-- Calculates the total number of relatives Molly needs to send gifts to -/
def total_relatives (scenario : GiftSendingScenario) : ℕ :=
  scenario.num_parents + 
  scenario.num_brothers + 
  scenario.num_brothers + -- for sisters-in-law
  scenario.num_brothers * scenario.num_children_per_brother

/-- Calculates the total cost of sending gifts to all relatives -/
def total_cost (scenario : GiftSendingScenario) : ℕ :=
  scenario.cost_per_package * total_relatives scenario

/-- Theorem stating that Molly's total cost for sending gifts is $70 -/
theorem molly_gift_cost : 
  ∀ (scenario : GiftSendingScenario), 
  scenario.cost_per_package = 5 ∧ 
  scenario.num_parents = 2 ∧ 
  scenario.num_brothers = 3 ∧ 
  scenario.num_children_per_brother = 2 → 
  total_cost scenario = 70 := by
  sorry

end NUMINAMATH_CALUDE_molly_gift_cost_l3584_358458


namespace NUMINAMATH_CALUDE_sum_power_inequality_l3584_358461

theorem sum_power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^a * b^b + a^b * b^a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_power_inequality_l3584_358461


namespace NUMINAMATH_CALUDE_second_company_base_rate_l3584_358452

/-- Represents the base rate and per-minute charge for a telephone company -/
structure TelephoneRate where
  baseRate : ℝ
  perMinuteCharge : ℝ

/-- Calculates the total charge for a given number of minutes -/
def totalCharge (rate : TelephoneRate) (minutes : ℝ) : ℝ :=
  rate.baseRate + rate.perMinuteCharge * minutes

theorem second_company_base_rate :
  let unitedRate : TelephoneRate := { baseRate := 11, perMinuteCharge := 0.25 }
  let otherRate : TelephoneRate := { baseRate := x, perMinuteCharge := 0.20 }
  let minutes : ℝ := 20
  totalCharge unitedRate minutes = totalCharge otherRate minutes →
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l3584_358452


namespace NUMINAMATH_CALUDE_red_marbles_in_bag_l3584_358417

theorem red_marbles_in_bag (total : ℕ) (prob : ℚ) (h_total : total = 84) (h_prob : prob = 36/49) :
  ∃ red : ℕ, red = 12 ∧ (1 - (red : ℚ) / total) * (1 - (red : ℚ) / total) = prob :=
sorry

end NUMINAMATH_CALUDE_red_marbles_in_bag_l3584_358417


namespace NUMINAMATH_CALUDE_lcm_fraction_evenness_l3584_358468

theorem lcm_fraction_evenness (x y z : ℕ+) :
  ∃ (k : ℕ), k > 0 ∧ k % 2 = 0 ∧
  (Nat.lcm x.val y.val + Nat.lcm y.val z.val) / Nat.lcm x.val z.val = k ∧
  ∀ (n : ℕ), n > 0 → n % 2 = 0 →
    ∃ (a b c : ℕ+), (Nat.lcm a.val b.val + Nat.lcm b.val c.val) / Nat.lcm a.val c.val = n :=
by sorry

end NUMINAMATH_CALUDE_lcm_fraction_evenness_l3584_358468


namespace NUMINAMATH_CALUDE_game_points_total_l3584_358476

/-- Game points calculation -/
theorem game_points_total (eric mark samanta daisy jake : ℕ) : 
  eric = 6 ∧ 
  mark = eric + eric / 2 ∧ 
  samanta = mark + 8 ∧ 
  daisy = (samanta + mark + eric) - (samanta + mark + eric) / 4 ∧
  jake = max samanta (max mark (max eric daisy)) - min samanta (min mark (min eric daisy)) →
  samanta + mark + eric + daisy + jake = 67 := by
  sorry


end NUMINAMATH_CALUDE_game_points_total_l3584_358476


namespace NUMINAMATH_CALUDE_eighteen_wheeler_axles_l3584_358403

/-- Represents the toll calculation for a truck on a bridge -/
def toll_formula (num_axles : ℕ) : ℚ :=
  2.5 + 0.5 * (num_axles - 2)

theorem eighteen_wheeler_axles :
  ∃ (num_axles : ℕ),
    (18 = 2 + 4 * (num_axles - 1)) ∧
    (toll_formula num_axles = 4) ∧
    (num_axles = 5) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_wheeler_axles_l3584_358403


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3584_358447

theorem sqrt_sum_inequality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
  Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > 2 ∧
  ∀ m : ℝ, (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
    Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
    Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > m) →
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3584_358447


namespace NUMINAMATH_CALUDE_greater_number_problem_l3584_358497

theorem greater_number_problem (x y : ℝ) : 
  x + y = 40 → x - y = 10 → x > y → x = 25 := by
sorry

end NUMINAMATH_CALUDE_greater_number_problem_l3584_358497


namespace NUMINAMATH_CALUDE_inequality_proof_l3584_358459

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3584_358459


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l3584_358480

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → ∃ (a b : ℕ), a^2 - b^2 = 221 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 229 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l3584_358480


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3584_358479

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt (kevin_eggs bonnie_eggs george_eggs cheryl_eggs : ℕ) 
  (hk : kevin_eggs = 5)
  (hb : bonnie_eggs = 13)
  (hg : george_eggs = 9)
  (hc : cheryl_eggs = 56) :
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3584_358479


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l3584_358443

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6

def score_red : ℕ := 2
def score_white : ℕ := 1

def ways_to_draw (n r w : ℕ) : ℕ := Nat.choose num_red_balls r * Nat.choose num_white_balls w

theorem ball_drawing_theorem :
  (ways_to_draw 4 4 0 + ways_to_draw 4 3 1 + ways_to_draw 4 2 2 = 115) ∧
  (ways_to_draw 5 2 3 + ways_to_draw 5 3 2 + ways_to_draw 5 4 1 = 186) := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l3584_358443


namespace NUMINAMATH_CALUDE_wilsons_theorem_l3584_358444

theorem wilsons_theorem (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ p ∣ (Nat.factorial (p - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l3584_358444


namespace NUMINAMATH_CALUDE_reggies_money_l3584_358404

/-- The amount of money Reggie's father gave him -/
def money_given : ℕ := sorry

/-- The number of books Reggie bought -/
def books_bought : ℕ := 5

/-- The cost of each book in dollars -/
def book_cost : ℕ := 2

/-- The amount of money Reggie has left after buying the books -/
def money_left : ℕ := 38

/-- Theorem stating that the money given by Reggie's father is $48 -/
theorem reggies_money : money_given = books_bought * book_cost + money_left := by sorry

end NUMINAMATH_CALUDE_reggies_money_l3584_358404


namespace NUMINAMATH_CALUDE_max_distance_to_origin_l3584_358409

open Complex

theorem max_distance_to_origin (z : ℂ) (h_norm : abs z = 1) : 
  let w := 2*z - Complex.I*z
  ∀ ε > 0, abs w ≤ 3 + ε :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_origin_l3584_358409


namespace NUMINAMATH_CALUDE_total_weekly_sleep_time_l3584_358469

/-- Represents the sleep patterns of animals -/
structure SleepPattern where
  evenDaySleep : ℕ
  oddDaySleep : ℕ

/-- Calculates the total weekly sleep for an animal given its sleep pattern -/
def weeklyTotalSleep (pattern : SleepPattern) : ℕ :=
  3 * pattern.evenDaySleep + 4 * pattern.oddDaySleep

/-- The sleep pattern of a cougar -/
def cougarSleep : SleepPattern :=
  { evenDaySleep := 4, oddDaySleep := 6 }

/-- The sleep pattern of a zebra -/
def zebraSleep : SleepPattern :=
  { evenDaySleep := cougarSleep.evenDaySleep + 2,
    oddDaySleep := cougarSleep.oddDaySleep + 2 }

/-- Theorem stating the total weekly sleep time for both animals -/
theorem total_weekly_sleep_time :
  weeklyTotalSleep cougarSleep + weeklyTotalSleep zebraSleep = 86 := by
  sorry


end NUMINAMATH_CALUDE_total_weekly_sleep_time_l3584_358469


namespace NUMINAMATH_CALUDE_square_area_is_26_l3584_358493

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square of the distance between two points -/
def squaredDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The area of a square given its four vertices -/
def squareArea (p q r s : Point) : ℝ :=
  squaredDistance p q

theorem square_area_is_26 : 
  let p : Point := ⟨1, 2⟩
  let q : Point := ⟨-4, 3⟩
  let r : Point := ⟨-3, -2⟩
  let s : Point := ⟨2, -3⟩
  squareArea p q r s = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_26_l3584_358493


namespace NUMINAMATH_CALUDE_no_odd_tens_digit_squares_l3584_358442

/-- The set of numbers from 1 to 50 -/
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 50}

/-- A number ends in 3 or 7 -/
def ends_in_3_or_7 (n : Nat) : Prop := n % 10 = 3 ∨ n % 10 = 7

/-- The tens digit of a number -/
def tens_digit (n : Nat) : Nat := (n / 10) % 10

/-- A number is even -/
def is_even (n : Nat) : Prop := n % 2 = 0

theorem no_odd_tens_digit_squares :
  ∀ n ∈ S, ends_in_3_or_7 n → is_even (tens_digit (n^2)) := by sorry

end NUMINAMATH_CALUDE_no_odd_tens_digit_squares_l3584_358442


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l3584_358466

theorem perfect_square_binomial : ∃ a : ℝ, ∀ x : ℝ, x^2 - 20*x + 100 = (x - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l3584_358466


namespace NUMINAMATH_CALUDE_sibling_product_specific_household_l3584_358440

/-- In a household with girls and boys, one boy counts all other children as siblings. -/
structure Household where
  girls : ℕ
  boys : ℕ
  counter : ℕ
  counter_is_boy : counter < boys

/-- The number of sisters the counter sees -/
def sisters (h : Household) : ℕ := h.girls

/-- The number of brothers the counter sees -/
def brothers (h : Household) : ℕ := h.boys - 1

/-- The product of sisters and brothers the counter sees -/
def sibling_product (h : Household) : ℕ := sisters h * brothers h

theorem sibling_product_specific_household :
  ∀ h : Household, h.girls = 5 → h.boys = 7 → sibling_product h = 24 := by
  sorry

end NUMINAMATH_CALUDE_sibling_product_specific_household_l3584_358440


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_in_expansion_fourth_term_has_largest_coefficient_l3584_358464

theorem largest_binomial_coefficient_in_expansion :
  ∀ k : ℕ, k ≤ 6 → Nat.choose 6 3 ≥ Nat.choose 6 k :=
by sorry

theorem fourth_term_has_largest_coefficient :
  ∃ k : ℕ, k = 4 ∧
  ∀ j : ℕ, j ≤ 6 → Nat.choose 6 (k - 1) ≥ Nat.choose 6 j :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_in_expansion_fourth_term_has_largest_coefficient_l3584_358464


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3584_358445

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3584_358445


namespace NUMINAMATH_CALUDE_future_ratio_years_l3584_358460

/-- Represents the ages and time in the problem -/
structure AgeData where
  vimal_initial : ℕ  -- Vimal's age 6 years ago
  saroj_initial : ℕ  -- Saroj's age 6 years ago
  years_passed : ℕ   -- Years passed since the initial ratio

/-- The conditions of the problem -/
def problem_conditions (data : AgeData) : Prop :=
  data.vimal_initial * 5 = data.saroj_initial * 6 ∧  -- Initial ratio 6:5
  data.saroj_initial + 6 = 16 ∧                      -- Saroj's current age is 16
  (data.vimal_initial + 6 + 4) * 10 = (data.saroj_initial + 6 + 4) * 11  -- Future ratio 11:10

/-- The theorem to be proved -/
theorem future_ratio_years (data : AgeData) :
  problem_conditions data → data.years_passed = 4 := by
  sorry


end NUMINAMATH_CALUDE_future_ratio_years_l3584_358460


namespace NUMINAMATH_CALUDE_simplify_expression_l3584_358432

theorem simplify_expression (a b : ℝ) : 
  3*b*(3*b^2 + 2*b) - b^2 + 2*a*(2*a^2 - 3*a) - 4*a*b = 
  9*b^3 + 5*b^2 + 4*a^3 - 6*a^2 - 4*a*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3584_358432


namespace NUMINAMATH_CALUDE_product_equals_64_l3584_358462

theorem product_equals_64 : 
  (1/2 : ℚ) * 4 * (1/8) * 16 * (1/32) * 64 * (1/128) * 256 * (1/512) * 1024 * (1/2048) * 4096 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_64_l3584_358462


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3584_358400

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 4 * x + 1 / x ≥ 4 ∧ 
  (4 * x + 1 / x = 4 ↔ x = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3584_358400


namespace NUMINAMATH_CALUDE_sweets_per_person_l3584_358401

/-- Represents the number of sweets Jennifer has of each color -/
structure Sweets :=
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of sweets -/
def total_sweets (s : Sweets) : ℕ := s.green + s.blue + s.yellow

/-- Represents the number of people sharing the sweets -/
def num_people : ℕ := 4

/-- Jennifer's sweets -/
def jennifer_sweets : Sweets := ⟨212, 310, 502⟩

/-- Theorem: Each person gets 256 sweets when Jennifer's sweets are shared equally -/
theorem sweets_per_person :
  (total_sweets jennifer_sweets) / num_people = 256 := by sorry

end NUMINAMATH_CALUDE_sweets_per_person_l3584_358401


namespace NUMINAMATH_CALUDE_employee_salary_calculation_l3584_358427

/-- Proves that given two employees m and n with a total weekly pay of $572, 
    where m's salary is 120% of n's salary, n's weekly pay is $260. -/
theorem employee_salary_calculation (total_pay m_salary n_salary : ℝ) : 
  total_pay = 572 →
  m_salary = 1.2 * n_salary →
  total_pay = m_salary + n_salary →
  n_salary = 260 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_calculation_l3584_358427


namespace NUMINAMATH_CALUDE_max_at_2_implies_c_6_l3584_358455

/-- The function f(x) = x(x-c)² has a maximum value at x=2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  ∀ x, f x ≤ f 2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x=2, then c = 6 -/
theorem max_at_2_implies_c_6 : 
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_at_2_implies_c_6_l3584_358455


namespace NUMINAMATH_CALUDE_mixture_carbonated_water_percentage_l3584_358418

/-- Calculates the percentage of carbonated water in a mixture of two solutions -/
def carbonated_water_percentage (solution1_percent : ℝ) (solution1_carbonated : ℝ) 
  (solution2_carbonated : ℝ) : ℝ :=
  solution1_percent * solution1_carbonated + (1 - solution1_percent) * solution2_carbonated

theorem mixture_carbonated_water_percentage :
  carbonated_water_percentage 0.1999999999999997 0.80 0.55 = 0.5999999999999999 := by
  sorry

#eval carbonated_water_percentage 0.1999999999999997 0.80 0.55

end NUMINAMATH_CALUDE_mixture_carbonated_water_percentage_l3584_358418


namespace NUMINAMATH_CALUDE_stationery_cost_l3584_358436

def total_spent : ℝ := 32
def backpack_cost : ℝ := 15
def notebook_cost : ℝ := 3
def notebook_count : ℕ := 5

theorem stationery_cost :
  total_spent - (backpack_cost + notebook_cost * notebook_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l3584_358436


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l3584_358481

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- three different side lengths
  a + b + c = 30 →        -- perimeter is 30
  a < 15 ∧ b < 15 ∧ c < 15 →  -- each side is less than 15
  a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 →  -- each side is at most 14
  ∃ (x y z : ℕ), x + y + z = 30 ∧ x = 14 ∧ y < x ∧ z < x ∧ y ≠ z →  -- there exists a triangle with max side 14
  (∀ (s : ℕ), s ≤ a ∨ s ≤ b ∨ s ≤ c) →  -- s is not greater than all sides
  14 = max a (max b c)  -- 14 is the maximum side length
  := by sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l3584_358481


namespace NUMINAMATH_CALUDE_parabola_properties_l3584_358477

/-- Given a parabola y = ax² - 5x - 3 passing through (-1, 4), prove its properties -/
theorem parabola_properties (a : ℝ) : 
  (a * (-1)^2 - 5 * (-1) - 3 = 4) → -- The parabola passes through (-1, 4)
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 - 5 * x₁ - 3 = 0 ∧ a * x₂^2 - 5 * x₂ - 3 = 0) ∧ -- Intersects x-axis at two points
  (- (-5) / (2 * a) = 5/4) -- Axis of symmetry is x = 5/4
  := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3584_358477


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3584_358472

theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3584_358472


namespace NUMINAMATH_CALUDE_volume_ratio_octahedron_cube_l3584_358489

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  volume : ℝ

/-- A cube whose vertices are the centers of the faces of a regular octahedron -/
structure RelatedCube where
  diagonal : ℝ
  volume : ℝ

/-- The relationship between a regular octahedron and its related cube -/
def octahedron_cube_relation (o : RegularOctahedron) (c : RelatedCube) : Prop :=
  c.diagonal = 2 * o.edge_length

theorem volume_ratio_octahedron_cube (o : RegularOctahedron) (c : RelatedCube) 
  (h : octahedron_cube_relation o c) : 
  o.volume / c.volume = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_octahedron_cube_l3584_358489


namespace NUMINAMATH_CALUDE_susan_money_left_l3584_358420

/-- The amount of money Susan has left after spending at the fair -/
def money_left (initial_amount food_cost ride_cost game_cost : ℕ) : ℕ :=
  initial_amount - (food_cost + ride_cost + game_cost)

/-- Theorem stating that Susan has 10 dollars left to spend -/
theorem susan_money_left :
  let initial_amount := 80
  let food_cost := 15
  let ride_cost := 3 * food_cost
  let game_cost := 10
  money_left initial_amount food_cost ride_cost game_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_money_left_l3584_358420


namespace NUMINAMATH_CALUDE_complex_power_48_l3584_358499

theorem complex_power_48 :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^48 = Complex.ofReal (-1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_48_l3584_358499


namespace NUMINAMATH_CALUDE_businessman_travel_l3584_358446

theorem businessman_travel (morning_bike : ℕ) (evening_bike : ℕ) (car_trips : ℕ) :
  morning_bike = 10 →
  evening_bike = 12 →
  car_trips = 8 →
  morning_bike + evening_bike + car_trips - 15 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_businessman_travel_l3584_358446


namespace NUMINAMATH_CALUDE_sum_of_photo_areas_l3584_358488

-- Define the side lengths of the three square photos
def photo1_side : ℝ := 2
def photo2_side : ℝ := 3
def photo3_side : ℝ := 1

-- Define the function to calculate the area of a square
def square_area (side : ℝ) : ℝ := side * side

-- Theorem: The sum of the areas of the three square photos is 14 square inches
theorem sum_of_photo_areas :
  square_area photo1_side + square_area photo2_side + square_area photo3_side = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_photo_areas_l3584_358488


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3584_358483

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
  x = 8 ∧ y = -8 * Real.sqrt 3 ∧
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt (x^2 + y^2) ∧
  θ = 2 * Real.pi - Real.pi / 3 →
  r = 16 ∧ θ = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3584_358483


namespace NUMINAMATH_CALUDE_first_question_percentage_l3584_358426

theorem first_question_percentage (second_correct : Real) 
                                  (neither_correct : Real)
                                  (both_correct : Real) :
  second_correct = 25 →
  neither_correct = 20 →
  both_correct = 20 →
  ∃ (first_correct : Real),
    first_correct = 75 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l3584_358426


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3584_358402

theorem smallest_sum_of_squares (x y z : ℝ) : 
  (x + 4) * (y - 4) = 0 → 
  3 * z - 2 * y = 5 → 
  x^2 + y^2 + z^2 ≥ 457/9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3584_358402


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l3584_358491

theorem stratified_sampling_size (total_employees : ℕ) (male_employees : ℕ) (female_sample : ℕ) (sample_size : ℕ) : 
  total_employees = 120 →
  male_employees = 90 →
  female_sample = 9 →
  (total_employees - male_employees) / total_employees = female_sample / sample_size →
  sample_size = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l3584_358491


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l3584_358437

def sequence_term (n : ℕ) : ℚ := (-1)^n * (n^2 : ℚ) / (2*n - 1)

theorem sequence_formula_correct : 
  (sequence_term 1 = -1) ∧ 
  (sequence_term 2 = 4/3) ∧ 
  (sequence_term 3 = -9/5) ∧ 
  (sequence_term 4 = 16/7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l3584_358437


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l3584_358416

theorem divisible_by_thirteen (n : ℕ) : ∃ k : ℤ, (7^(2*n) + 10^(n+1) + 2 * 10^n) = 13 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l3584_358416


namespace NUMINAMATH_CALUDE_marvins_substitution_l3584_358478

theorem marvins_substitution (a b c d f : ℤ) : 
  a = 3 → b = 4 → c = 7 → d = 5 →
  (a + b - c + d - f = a + (b - (c + (d - f)))) →
  f = 5 := by sorry

end NUMINAMATH_CALUDE_marvins_substitution_l3584_358478


namespace NUMINAMATH_CALUDE_correct_cobs_per_row_l3584_358486

/-- Represents the number of corn cobs in each row -/
def cobs_per_row : ℕ := 4

/-- Represents the number of rows in the first field -/
def rows_field1 : ℕ := 13

/-- Represents the number of rows in the second field -/
def rows_field2 : ℕ := 16

/-- Represents the total number of corn cobs -/
def total_cobs : ℕ := 116

/-- Theorem stating that the number of corn cobs per row is correct -/
theorem correct_cobs_per_row : 
  cobs_per_row * rows_field1 + cobs_per_row * rows_field2 = total_cobs := by
  sorry

end NUMINAMATH_CALUDE_correct_cobs_per_row_l3584_358486


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3584_358419

theorem stratified_sampling_sample_size (total_population : ℕ) (elderly_population : ℕ) (elderly_sample : ℕ) (sample_size : ℕ) :
  total_population = 162 →
  elderly_population = 27 →
  elderly_sample = 6 →
  (elderly_sample : ℚ) / sample_size = (elderly_population : ℚ) / total_population →
  sample_size = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3584_358419


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3584_358438

/-- Represents the rate of interest per annum as a percentage -/
def rate : ℝ := 9

/-- The amount lent to B -/
def principal_B : ℝ := 5000

/-- The amount lent to C -/
def principal_C : ℝ := 3000

/-- The time period for B's loan in years -/
def time_B : ℝ := 2

/-- The time period for C's loan in years -/
def time_C : ℝ := 4

/-- The total interest received from both B and C -/
def total_interest : ℝ := 1980

/-- Theorem stating that the given rate satisfies the problem conditions -/
theorem interest_rate_proof :
  (principal_B * rate * time_B / 100 + principal_C * rate * time_C / 100) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3584_358438


namespace NUMINAMATH_CALUDE_mangoes_purchased_is_nine_l3584_358449

/-- The amount of mangoes purchased, given the conditions of the problem -/
def mangoes_purchased (apple_kg : ℕ) (apple_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_kg * apple_rate) / mango_rate

/-- Theorem stating that the amount of mangoes purchased is 9 kg -/
theorem mangoes_purchased_is_nine :
  mangoes_purchased 8 70 45 965 = 9 := by sorry

end NUMINAMATH_CALUDE_mangoes_purchased_is_nine_l3584_358449


namespace NUMINAMATH_CALUDE_complex_product_equals_2401_l3584_358453

theorem complex_product_equals_2401 :
  let x : ℂ := Complex.exp (2 * Real.pi * I / 9)
  (3 * x + x^2) * (3 * x^2 + x^4) * (3 * x^3 + x^6) * (3 * x^4 + x^8) *
  (3 * x^5 + x^10) * (3 * x^6 + x^12) * (3 * x^7 + x^14) = 2401 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_2401_l3584_358453


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3584_358471

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 4
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 2
  let ways_to_choose_republicans : ℕ := (total_republicans.choose subcommittee_republicans)
  let ways_to_choose_democrats : ℕ := (total_democrats.choose subcommittee_democrats)
  ways_to_choose_republicans * ways_to_choose_democrats = 1260 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3584_358471


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3584_358448

/-- The coefficient of x in the simplified form of 2(x - 5) + 5(8 - 3x^2 + 6x) - 9(3x - 2) is 5 -/
theorem coefficient_of_x (x : ℝ) : 
  let expression := 2*(x - 5) + 5*(8 - 3*x^2 + 6*x) - 9*(3*x - 2)
  ∃ a b c : ℝ, expression = a*x^2 + 5*x + c := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3584_358448


namespace NUMINAMATH_CALUDE_characterize_f_l3584_358405

def is_valid_f (f : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, f (n + 1) ≥ f n) ∧
  (∀ m n : ℕ+, Nat.gcd m.val n.val = 1 → f (m * n) = f m * f n)

theorem characterize_f (f : ℕ+ → ℝ) (hf : is_valid_f f) :
  (∃ a : ℝ, a ≥ 0 ∧ ∀ n : ℕ+, f n = (n : ℝ) ^ a) ∨ (∀ n : ℕ+, f n = 0) :=
sorry

end NUMINAMATH_CALUDE_characterize_f_l3584_358405


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3584_358467

theorem tan_alpha_value (α : Real) (h : Real.tan (α - Real.pi/4) = 1/6) : 
  Real.tan α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3584_358467


namespace NUMINAMATH_CALUDE_joan_balloons_l3584_358494

theorem joan_balloons (initial : ℕ) : initial + 2 = 10 → initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3584_358494


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3584_358496

theorem arithmetic_calculation : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3584_358496


namespace NUMINAMATH_CALUDE_value_of_expression_l3584_358492

-- Define the polynomial
def p (x h k : ℝ) : ℝ := 5 * x^4 - h * x^2 + k

-- State the theorem
theorem value_of_expression (h k : ℝ) :
  (p 3 h k = 0) → (p (-1) h k = 0) → (p 2 h k = 0) → |5 * h - 4 * k| = 70 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3584_358492


namespace NUMINAMATH_CALUDE_pomelo_sales_theorem_l3584_358451

/-- Represents the sales data for a week -/
structure WeeklySales where
  planned_daily : ℕ
  deviations : List ℤ
  selling_price : ℕ
  shipping_cost : ℕ

/-- Calculates the difference between highest and lowest sales days -/
def sales_difference (sales : WeeklySales) : ℕ :=
  let max_dev := sales.deviations.maximum?
  let min_dev := sales.deviations.minimum?
  match max_dev, min_dev with
  | some max, some min => (max - min).natAbs
  | _, _ => 0

/-- Calculates the total sales for the week -/
def total_sales (sales : WeeklySales) : ℕ :=
  sales.planned_daily * 7 + sales.deviations.sum.natAbs

/-- Calculates the total profit for the week -/
def total_profit (sales : WeeklySales) : ℕ :=
  (sales.selling_price - sales.shipping_cost) * (total_sales sales)

/-- Main theorem to prove -/
theorem pomelo_sales_theorem (sales : WeeklySales)
  (h1 : sales.planned_daily = 100)
  (h2 : sales.deviations = [3, -5, -2, 11, -7, 13, 5])
  (h3 : sales.selling_price = 8)
  (h4 : sales.shipping_cost = 3) :
  sales_difference sales = 20 ∧
  total_sales sales = 718 ∧
  total_profit sales = 3590 := by
  sorry


end NUMINAMATH_CALUDE_pomelo_sales_theorem_l3584_358451


namespace NUMINAMATH_CALUDE_square_sum_formula_l3584_358408

theorem square_sum_formula (x y c a : ℝ) 
  (h1 : x * y = 2 * c) 
  (h2 : 1 / x^2 + 1 / y^2 = 3 * a) : 
  (x + y)^2 = 12 * a * c^2 + 4 * c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_formula_l3584_358408


namespace NUMINAMATH_CALUDE_probability_three_girls_l3584_358422

theorem probability_three_girls (total : ℕ) (girls : ℕ) (chosen : ℕ) : 
  total = 15 → girls = 9 → chosen = 3 →
  (Nat.choose girls chosen : ℚ) / (Nat.choose total chosen : ℚ) = 12 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_l3584_358422


namespace NUMINAMATH_CALUDE_probability_sum_six_two_dice_l3584_358454

/-- A fair die has 6 sides -/
def fairDieSides : ℕ := 6

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes -/
def probability (favorableOutcomes totalOutcomes : ℕ) : ℚ :=
  favorableOutcomes / totalOutcomes

/-- The total number of possible outcomes when throwing a die twice is the square of the number of sides -/
def totalOutcomes (sides : ℕ) : ℕ :=
  sides * sides

/-- The favorable outcomes are the pairs of numbers that sum to 6 -/
def favorableOutcomes : ℕ := 5

theorem probability_sum_six_two_dice :
  probability favorableOutcomes (totalOutcomes fairDieSides) = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_two_dice_l3584_358454


namespace NUMINAMATH_CALUDE_only_zero_solution_for_diophantine_equation_l3584_358450

theorem only_zero_solution_for_diophantine_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_solution_for_diophantine_equation_l3584_358450


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l3584_358474

/-- Calculates the time taken for a monkey to climb a tree given the tree height and climbing behavior -/
def monkey_climb_time (tree_height : ℕ) (hop_up : ℕ) (slip_back : ℕ) : ℕ := sorry

/-- Theorem stating that for the given conditions, the monkey takes 15 hours to reach the top -/
theorem monkey_climb_theorem :
  monkey_climb_time 51 7 4 = 15 := by sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l3584_358474


namespace NUMINAMATH_CALUDE_no_primes_for_d_10_l3584_358425

theorem no_primes_for_d_10 : ¬∃ (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (q * r) ∣ (p^2 + 10) ∧
  (r * p) ∣ (q^2 + 10) ∧
  (p * q) ∣ (r^2 + 10) :=
by
  sorry

-- Note: The case for d = 11 is not included as the solution was inconclusive

end NUMINAMATH_CALUDE_no_primes_for_d_10_l3584_358425


namespace NUMINAMATH_CALUDE_sum_a_equals_1649_l3584_358484

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 20 = 0 then 15
  else if n % 20 = 0 ∧ n % 18 = 0 then 20
  else if n % 18 = 0 ∧ n % 15 = 0 then 18
  else 0

theorem sum_a_equals_1649 :
  (Finset.range 2999).sum a = 1649 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_equals_1649_l3584_358484


namespace NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l3584_358411

theorem only_negative_three_less_than_negative_two :
  ((-3 : ℝ) < -2) ∧
  ((-1 : ℝ) > -2) ∧
  ((-Real.sqrt 2 : ℝ) > -2) ∧
  ((-Real.pi / 2 : ℝ) > -2) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l3584_358411


namespace NUMINAMATH_CALUDE_remaining_numbers_l3584_358435

theorem remaining_numbers (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 9 →
  total_avg = 18 →
  subset = 4 →
  subset_avg = 8 →
  remaining_avg = 26 →
  total - subset = (total * total_avg - subset * subset_avg) / remaining_avg :=
by
  sorry

#eval (9 : ℕ) - 4  -- Expected output: 5

end NUMINAMATH_CALUDE_remaining_numbers_l3584_358435


namespace NUMINAMATH_CALUDE_staff_distribution_ways_l3584_358456

/-- The number of ways to distribute n indistinguishable objects among k distinct containers,
    with each container receiving at least min_per_container and at most max_per_container objects. -/
def distribute_objects (n : ℕ) (k : ℕ) (min_per_container : ℕ) (max_per_container : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 90 ways to distribute 5 staff members among 3 schools
    with each school receiving at least 1 and at most 2 staff members. -/
theorem staff_distribution_ways : distribute_objects 5 3 1 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_staff_distribution_ways_l3584_358456


namespace NUMINAMATH_CALUDE_union_of_M_and_Q_l3584_358482

def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

theorem union_of_M_and_Q : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_Q_l3584_358482


namespace NUMINAMATH_CALUDE_intersection_M_N_l3584_358498

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {y | y ≥ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3584_358498


namespace NUMINAMATH_CALUDE_square_difference_equals_720_l3584_358431

theorem square_difference_equals_720 : (30 + 12)^2 - (12^2 + 30^2) = 720 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_720_l3584_358431


namespace NUMINAMATH_CALUDE_smallest_abs_value_rational_l3584_358415

theorem smallest_abs_value_rational (q : ℚ) : |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_value_rational_l3584_358415


namespace NUMINAMATH_CALUDE_percentage_of_total_l3584_358406

theorem percentage_of_total (N F M : ℝ) 
  (h1 : N = 0.05 * F) 
  (h2 : N = 0.20 * M) : 
  N / (F + M) = 0.04 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_total_l3584_358406


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l3584_358485

theorem simplify_sqrt_difference : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l3584_358485


namespace NUMINAMATH_CALUDE_stuffed_dogs_count_l3584_358490

/-- The number of boxes of stuffed toy dogs -/
def num_boxes : ℕ := 7

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 4

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem stuffed_dogs_count : total_dogs = 28 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_dogs_count_l3584_358490


namespace NUMINAMATH_CALUDE_second_reader_average_pages_per_day_l3584_358434

/-- Calculates the average pages read per day by the second-place reader -/
def average_pages_per_day (break_days : ℕ) (deshaun_books : ℕ) (avg_pages_per_book : ℕ) (second_reader_percentage : ℚ) : ℚ :=
  let deshaun_total_pages := deshaun_books * avg_pages_per_book
  let second_reader_pages := second_reader_percentage * deshaun_total_pages
  second_reader_pages / break_days

/-- Theorem stating that the second-place reader averaged 180 pages per day -/
theorem second_reader_average_pages_per_day :
  average_pages_per_day 80 60 320 (3/4) = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_reader_average_pages_per_day_l3584_358434


namespace NUMINAMATH_CALUDE_inequality_proof_l3584_358470

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : (a + b) * (b + c) * (c + a) = 1) : 
  a^2 / (1 + Real.sqrt (b * c)) + 
  b^2 / (1 + Real.sqrt (c * a)) + 
  c^2 / (1 + Real.sqrt (a * b)) ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3584_358470


namespace NUMINAMATH_CALUDE_triangle_problem_l3584_358465

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (2 * c = Real.sqrt 3 * a + 2 * b * Real.cos A) →
  (c = 7) →
  (b * Real.sin A = Real.sqrt 3) →
  (B = π / 6 ∧ b = Real.sqrt 19) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3584_358465


namespace NUMINAMATH_CALUDE_total_homework_pages_l3584_358487

def math_homework_pages : ℕ := 10
def reading_homework_difference : ℕ := 3

def total_pages : ℕ := math_homework_pages + (math_homework_pages + reading_homework_difference)

theorem total_homework_pages : total_pages = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_homework_pages_l3584_358487


namespace NUMINAMATH_CALUDE_polyhedron_distance_equation_l3584_358423

/-- A convex polyhedron with 12 regular triangular faces -/
structure Polyhedron :=
  (e : ℝ)  -- Common edge length
  (t : ℝ)  -- Additional length in the distance between non-adjacent five-edged vertices

/-- The distance between two non-adjacent five-edged vertices is (e+t) -/
def distance (p : Polyhedron) : ℝ := p.e + p.t

/-- Theorem: For the given polyhedron, t³ - 7et² + 2e³ = 0 -/
theorem polyhedron_distance_equation (p : Polyhedron) : 
  p.t^3 - 7 * p.e * p.t^2 + 2 * p.e^3 = 0 :=
sorry

end NUMINAMATH_CALUDE_polyhedron_distance_equation_l3584_358423


namespace NUMINAMATH_CALUDE_projectile_motion_time_l3584_358473

/-- The equation of motion for a projectile launched from the ground -/
def equation_of_motion (v : ℝ) (t : ℝ) : ℝ := -16 * t^2 + v * t

/-- The initial velocity of the projectile in feet per second -/
def initial_velocity : ℝ := 80

/-- The height reached by the projectile in feet -/
def height_reached : ℝ := 100

/-- The time taken to reach the specified height -/
def time_to_reach_height : ℝ := 2.5

theorem projectile_motion_time :
  equation_of_motion initial_velocity time_to_reach_height = height_reached :=
by sorry

end NUMINAMATH_CALUDE_projectile_motion_time_l3584_358473


namespace NUMINAMATH_CALUDE_set_intersection_example_l3584_358433

theorem set_intersection_example : 
  ({3, 5, 6, 8} : Set ℕ) ∩ ({4, 5, 8} : Set ℕ) = {5, 8} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3584_358433


namespace NUMINAMATH_CALUDE_coat_price_proof_l3584_358430

/-- Proves that the original price of a coat is $500 given the specified conditions -/
theorem coat_price_proof (P : ℝ) 
  (h1 : 0.70 * P = 350) : P = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_proof_l3584_358430


namespace NUMINAMATH_CALUDE_teachers_made_28_materials_l3584_358441

/-- Given the number of recycled materials made by a group and the total number of recycled products to be sold, 
    calculate the number of recycled materials made by teachers. -/
def teachers_recycled_materials (group_materials : ℕ) (total_products : ℕ) : ℕ :=
  total_products - group_materials

/-- Theorem: Given that the group made 65 recycled materials and the total number of recycled products
    to be sold is 93, prove that the teachers made 28 recycled materials. -/
theorem teachers_made_28_materials : teachers_recycled_materials 65 93 = 28 := by
  sorry

end NUMINAMATH_CALUDE_teachers_made_28_materials_l3584_358441


namespace NUMINAMATH_CALUDE_expression_factorization_l3584_358414

theorem expression_factorization (x : ℝ) : 
  (12 * x^3 + 95 * x - 6) - (-3 * x^3 + 5 * x - 6) = 15 * x * (x^2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3584_358414


namespace NUMINAMATH_CALUDE_probability_white_then_red_l3584_358429

/-- The probability of drawing a white marble first and then a red marble second, without replacement, from a bag containing 4 red marbles and 6 white marbles. -/
theorem probability_white_then_red (red_marbles white_marbles : ℕ) 
  (h_red : red_marbles = 4) 
  (h_white : white_marbles = 6) : 
  (white_marbles : ℚ) / (red_marbles + white_marbles) * 
  (red_marbles : ℚ) / (red_marbles + white_marbles - 1) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_then_red_l3584_358429
