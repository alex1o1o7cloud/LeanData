import Mathlib

namespace NUMINAMATH_CALUDE_factor_cubic_expression_l2173_217314

theorem factor_cubic_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_factor_cubic_expression_l2173_217314


namespace NUMINAMATH_CALUDE_area_outside_smaller_squares_l2173_217320

theorem area_outside_smaller_squares (larger_side : ℝ) (smaller_side : ℝ) : 
  larger_side = 10 → 
  smaller_side = 4 → 
  larger_side^2 - 2 * smaller_side^2 = 68 := by
sorry

end NUMINAMATH_CALUDE_area_outside_smaller_squares_l2173_217320


namespace NUMINAMATH_CALUDE_days_passed_before_realization_l2173_217398

/-- Represents the contractor's job scenario -/
structure JobScenario where
  totalDays : ℕ
  initialWorkers : ℕ
  workCompletedFraction : ℚ
  workersFired : ℕ
  remainingDays : ℕ

/-- Calculates the number of days passed before the contractor realized a fraction of work was done -/
def daysPassedBeforeRealization (scenario : JobScenario) : ℕ :=
  sorry

/-- The theorem stating that for the given scenario, 20 days passed before realization -/
theorem days_passed_before_realization :
  let scenario : JobScenario := {
    totalDays := 100,
    initialWorkers := 10,
    workCompletedFraction := 1/4,
    workersFired := 2,
    remainingDays := 75
  }
  daysPassedBeforeRealization scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_days_passed_before_realization_l2173_217398


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2173_217307

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = π / 5 ∧  -- Given condition
  a * Real.cos B - b * Real.cos A = c →  -- Given equation
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2173_217307


namespace NUMINAMATH_CALUDE_part_one_part_two_l2173_217312

-- Define sets A and B
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1: Prove that (C_R B) ∩ A = {x | 3 ≤ x ≤ 5} when m = 3
theorem part_one : (Set.compl (B 3) ∩ A) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2: Prove that if A ∩ B = {x | -1 < x < 4}, then m = 8
theorem part_two : (∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4}) → (∃ m : ℝ, m = 8) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2173_217312


namespace NUMINAMATH_CALUDE_triangle_problem_l2173_217302

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where c = √3, b = 1, and B = 30°, prove that C is either 60° or 120°,
    and the corresponding area S is either √3/2 or √3/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  c = Real.sqrt 3 →
  b = 1 →
  B = 30 * π / 180 →
  ((C = 60 * π / 180 ∧ S = Real.sqrt 3 / 2) ∨
   (C = 120 * π / 180 ∧ S = Real.sqrt 3 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2173_217302


namespace NUMINAMATH_CALUDE_exists_a_plus_ω_l2173_217364

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem exists_a_plus_ω : ∃ (a ω : ℝ), 
  ω > 0 ∧ 
  (∀ x, f ω a x = f ω a (2 * Real.pi / 3 - x)) ∧ 
  (∀ x, f ω a (Real.pi / 6) ≤ f ω a x) ∧ 
  0 ≤ a + ω ∧ a + ω ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_exists_a_plus_ω_l2173_217364


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l2173_217324

theorem square_reciprocal_sum (n : ℝ) (h : n + 1/n = 10) :
  n^2 + 1/n^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l2173_217324


namespace NUMINAMATH_CALUDE_full_price_revenue_l2173_217337

/-- Represents a concert ticket sale scenario -/
structure ConcertSale where
  fullPrice : ℕ  -- Number of full-price tickets
  discountPrice : ℕ  -- Number of discount-price tickets
  price : ℕ  -- Price of a full-price ticket in dollars

/-- Conditions for a valid concert sale -/
def isValidSale (sale : ConcertSale) : Prop :=
  sale.fullPrice + sale.discountPrice = 200 ∧
  sale.fullPrice * sale.price + sale.discountPrice * (sale.price / 3) = 3000

/-- Theorem stating the revenue from full-price tickets -/
theorem full_price_revenue (sale : ConcertSale) 
  (h : isValidSale sale) : sale.fullPrice * sale.price = 1500 := by
  sorry


end NUMINAMATH_CALUDE_full_price_revenue_l2173_217337


namespace NUMINAMATH_CALUDE_tenth_square_area_l2173_217377

theorem tenth_square_area : 
  let initial_side : ℝ := 2
  let side_sequence : ℕ → ℝ := λ n => initial_side * (Real.sqrt 2) ^ (n - 1)
  let area : ℕ → ℝ := λ n => (side_sequence n) ^ 2
  area 10 = 2048 := by sorry

end NUMINAMATH_CALUDE_tenth_square_area_l2173_217377


namespace NUMINAMATH_CALUDE_sock_inventory_theorem_l2173_217383

/-- Represents the number of socks of a particular color --/
structure SockCount where
  pairs : ℕ
  singles : ℕ

/-- Represents the total sock inventory --/
structure SockInventory where
  blue : SockCount
  green : SockCount
  red : SockCount

def initial_inventory : SockInventory := {
  blue := { pairs := 20, singles := 0 },
  green := { pairs := 15, singles := 0 },
  red := { pairs := 15, singles := 0 }
}

def lost_socks : SockInventory := {
  blue := { pairs := 0, singles := 3 },
  green := { pairs := 0, singles := 2 },
  red := { pairs := 0, singles := 2 }
}

def donated_socks : SockInventory := {
  blue := { pairs := 0, singles := 10 },
  green := { pairs := 0, singles := 15 },
  red := { pairs := 0, singles := 10 }
}

def purchased_socks : SockInventory := {
  blue := { pairs := 5, singles := 0 },
  green := { pairs := 3, singles := 0 },
  red := { pairs := 2, singles := 0 }
}

def gifted_socks : SockInventory := {
  blue := { pairs := 2, singles := 0 },
  green := { pairs := 1, singles := 0 },
  red := { pairs := 0, singles := 0 }
}

def update_inventory (inv : SockInventory) (change : SockInventory) : SockInventory :=
  { blue := { pairs := inv.blue.pairs + change.blue.pairs - (inv.blue.singles + change.blue.singles) / 2,
              singles := (inv.blue.singles + change.blue.singles) % 2 },
    green := { pairs := inv.green.pairs + change.green.pairs - (inv.green.singles + change.green.singles) / 2,
               singles := (inv.green.singles + change.green.singles) % 2 },
    red := { pairs := inv.red.pairs + change.red.pairs - (inv.red.singles + change.red.singles) / 2,
             singles := (inv.red.singles + change.red.singles) % 2 } }

def total_pairs (inv : SockInventory) : ℕ :=
  inv.blue.pairs + inv.green.pairs + inv.red.pairs

theorem sock_inventory_theorem :
  let final_inventory := update_inventory 
                          (update_inventory 
                            (update_inventory 
                              (update_inventory initial_inventory lost_socks) 
                            donated_socks) 
                          purchased_socks) 
                        gifted_socks
  total_pairs final_inventory = 43 := by
  sorry

end NUMINAMATH_CALUDE_sock_inventory_theorem_l2173_217383


namespace NUMINAMATH_CALUDE_evaluate_expression_l2173_217392

theorem evaluate_expression : (49^2 - 35^2) + (15^2 - 9^2) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2173_217392


namespace NUMINAMATH_CALUDE_anne_tom_age_sum_l2173_217332

theorem anne_tom_age_sum : 
  ∀ (A T : ℝ),
  A = T + 9 →
  A + 7 = 5 * (T - 3) →
  A + T = 24.5 :=
by
  sorry

end NUMINAMATH_CALUDE_anne_tom_age_sum_l2173_217332


namespace NUMINAMATH_CALUDE_compound_propositions_truth_l2173_217335

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x < y → x^2 > y^2

-- Theorem statement
theorem compound_propositions_truth (hp : p) (hq : ¬q) : 
  (p ∧ q = False) ∧ 
  (p ∨ q = True) ∧ 
  (p ∧ (¬q) = True) ∧ 
  ((¬p) ∨ q = False) := by
  sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_l2173_217335


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2173_217350

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2173_217350


namespace NUMINAMATH_CALUDE_third_to_second_night_ratio_l2173_217306

/-- Represents Billy's sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- Calculates the total sleep over four nights -/
def total_sleep (sp : SleepPattern) : ℝ :=
  sp.first_night + sp.second_night + sp.third_night + sp.fourth_night

/-- Theorem stating the ratio of third to second night's sleep -/
theorem third_to_second_night_ratio 
  (sp : SleepPattern)
  (h1 : sp.first_night = 6)
  (h2 : sp.second_night = sp.first_night + 2)
  (h3 : sp.fourth_night = 3 * sp.third_night)
  (h4 : total_sleep sp = 30) :
  sp.third_night / sp.second_night = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_third_to_second_night_ratio_l2173_217306


namespace NUMINAMATH_CALUDE_only_x0_is_perfect_square_l2173_217375

-- Define the sequence (x_n)
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * x (n + 1) - x n

-- Define a perfect square
def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

-- Theorem statement
theorem only_x0_is_perfect_square :
  ∀ n : ℕ, isPerfectSquare (x n) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_x0_is_perfect_square_l2173_217375


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2173_217325

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 1)
  (h2 : seq.a 1 + seq.a 3 + seq.a 5 = 21) :
  seq.a 2 + seq.a 4 + seq.a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2173_217325


namespace NUMINAMATH_CALUDE_twenty_cent_coins_count_l2173_217376

/-- Represents the coin collection of Alex -/
structure CoinCollection where
  total_coins : ℕ
  ten_cent_coins : ℕ
  twenty_cent_coins : ℕ
  total_is_sum : total_coins = ten_cent_coins + twenty_cent_coins
  all_coins_accounted : total_coins = 14

/-- Calculates the number of different values obtainable from a given coin collection -/
def different_values (c : CoinCollection) : ℕ :=
  27 - c.ten_cent_coins

/-- The main theorem stating that if there are 22 different obtainable values, 
    then there must be 9 20-cent coins -/
theorem twenty_cent_coins_count 
  (c : CoinCollection) 
  (h : different_values c = 22) : 
  c.twenty_cent_coins = 9 := by
  sorry

end NUMINAMATH_CALUDE_twenty_cent_coins_count_l2173_217376


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2173_217333

theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (∃ (a b c d : ℚ), a + b + c + d = T ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →  -- Four children's ages sum to T
  (T - N = 3 * (T - 4 * N)) →  -- Condition from N years ago
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2173_217333


namespace NUMINAMATH_CALUDE_figure_with_perimeter_91_has_11_tiles_l2173_217367

/-- Represents a figure in the sequence --/
structure Figure where
  tiles : ℕ
  perimeter : ℕ

/-- The side length of each equilateral triangle tile in cm --/
def tileSideLength : ℕ := 7

/-- The first figure in the sequence --/
def firstFigure : Figure :=
  { tiles := 1
  , perimeter := 3 * tileSideLength }

/-- Generates the next figure in the sequence --/
def nextFigure (f : Figure) : Figure :=
  { tiles := f.tiles + 1
  , perimeter := f.perimeter + tileSideLength }

/-- Theorem: The figure with perimeter 91 cm consists of 11 tiles --/
theorem figure_with_perimeter_91_has_11_tiles :
  ∃ (n : ℕ), (n.iterate nextFigure firstFigure).perimeter = 91 ∧
             (n.iterate nextFigure firstFigure).tiles = 11 := by
  sorry

end NUMINAMATH_CALUDE_figure_with_perimeter_91_has_11_tiles_l2173_217367


namespace NUMINAMATH_CALUDE_soft_drink_storage_l2173_217331

theorem soft_drink_storage (small_initial big_initial : ℕ) 
  (big_sold_percent : ℚ) (total_remaining : ℕ) :
  small_initial = 6000 →
  big_initial = 14000 →
  big_sold_percent = 23 / 100 →
  total_remaining = 15580 →
  ∃ (small_sold_percent : ℚ),
    small_sold_percent = 37 / 100 ∧
    (small_initial : ℚ) * (1 - small_sold_percent) + 
    (big_initial : ℚ) * (1 - big_sold_percent) = total_remaining := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_storage_l2173_217331


namespace NUMINAMATH_CALUDE_composite_divisor_theorem_l2173_217352

def proper_divisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ 1 < d ∧ d < n}

def increased_divisors (n : ℕ) : Set ℕ :=
  {d + 1 | d ∈ proper_divisors n}

theorem composite_divisor_theorem (n : ℕ) :
  (∃ m : ℕ, increased_divisors n = proper_divisors m) ↔ n = 4 ∨ n = 8 :=
sorry

end NUMINAMATH_CALUDE_composite_divisor_theorem_l2173_217352


namespace NUMINAMATH_CALUDE_peaches_before_picking_l2173_217389

-- Define the variables
def peaches_picked : ℕ := 52
def total_peaches_now : ℕ := 86

-- Define the theorem
theorem peaches_before_picking (peaches_picked total_peaches_now : ℕ) :
  peaches_picked = 52 →
  total_peaches_now = 86 →
  total_peaches_now - peaches_picked = 34 := by
sorry

end NUMINAMATH_CALUDE_peaches_before_picking_l2173_217389


namespace NUMINAMATH_CALUDE_function_values_l2173_217370

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else x^2

theorem function_values (a : ℝ) : f (-1) = 2 * f a → a = Real.sqrt 3 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l2173_217370


namespace NUMINAMATH_CALUDE_conference_handshakes_l2173_217327

/-- The number of companies at the conference -/
def num_companies : ℕ := 5

/-- The number of representatives per company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the conference -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - reps_per_company

/-- The total number of handshakes at the conference -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem conference_handshakes : total_handshakes = 250 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2173_217327


namespace NUMINAMATH_CALUDE_hypotenuse_ratio_from_area_ratio_l2173_217305

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  area : ℝ

-- Theorem statement
theorem hypotenuse_ratio_from_area_ratio
  (t1 t2 : IsoscelesRightTriangle)
  (h_area : t2.area = 2 * t1.area) :
  t2.hypotenuse = Real.sqrt 2 * t1.hypotenuse :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_ratio_from_area_ratio_l2173_217305


namespace NUMINAMATH_CALUDE_mean_temperature_is_88_75_l2173_217372

def temperatures : List ℚ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 :
  (temperatures.sum / temperatures.length : ℚ) = 355/4 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_is_88_75_l2173_217372


namespace NUMINAMATH_CALUDE_find_divisor_l2173_217339

theorem find_divisor (N : ℕ) (D : ℕ) (h1 : N = 269 * D) (h2 : N % 67 = 1) : D = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2173_217339


namespace NUMINAMATH_CALUDE_sandy_obtained_45_marks_l2173_217336

/-- Calculates the total marks obtained by Sandy given the number of correct and incorrect sums. -/
def sandy_marks (total_sums : ℕ) (correct_sums : ℕ) : ℤ :=
  let incorrect_sums := total_sums - correct_sums
  3 * correct_sums - 2 * incorrect_sums

/-- Proves that Sandy obtained 45 marks given the problem conditions. -/
theorem sandy_obtained_45_marks :
  sandy_marks 30 21 = 45 := by
  sorry

#eval sandy_marks 30 21

end NUMINAMATH_CALUDE_sandy_obtained_45_marks_l2173_217336


namespace NUMINAMATH_CALUDE_total_selling_price_calculation_craig_appliance_sales_l2173_217366

/-- Calculates the total selling price of appliances given commission details --/
theorem total_selling_price_calculation 
  (fixed_commission : ℝ) 
  (variable_commission_rate : ℝ) 
  (num_appliances : ℕ) 
  (total_commission : ℝ) : ℝ :=
  let total_fixed_commission := fixed_commission * num_appliances
  let variable_commission := total_commission - total_fixed_commission
  variable_commission / variable_commission_rate

/-- Proves that the total selling price is $3620 given the problem conditions --/
theorem craig_appliance_sales : 
  total_selling_price_calculation 50 0.1 6 662 = 3620 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_calculation_craig_appliance_sales_l2173_217366


namespace NUMINAMATH_CALUDE_f_36_equals_2pq_l2173_217308

/-- A function satisfying f(xy) = f(x) + f(y) for all x and y -/
def LogLikeFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * y) = f x + f y

/-- Main theorem: f(36) = 2(p + q) given the conditions -/
theorem f_36_equals_2pq (f : ℝ → ℝ) (p q : ℝ) 
  (h1 : LogLikeFunction f) 
  (h2 : f 2 = p) 
  (h3 : f 3 = q) : 
  f 36 = 2 * (p + q) := by
  sorry


end NUMINAMATH_CALUDE_f_36_equals_2pq_l2173_217308


namespace NUMINAMATH_CALUDE_magazine_cost_l2173_217301

theorem magazine_cost (book magazine : ℚ)
  (h1 : 2 * book + 2 * magazine = 26)
  (h2 : book + 3 * magazine = 27) :
  magazine = 7 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l2173_217301


namespace NUMINAMATH_CALUDE_lydia_planting_age_l2173_217379

/-- Represents the age of Lydia when she planted the tree -/
def planting_age : ℕ := sorry

/-- The time it takes for an apple tree to bear fruit -/
def fruit_bearing_time : ℕ := 7

/-- Lydia's current age -/
def current_age : ℕ := 9

/-- Lydia's age when she first eats an apple from her tree -/
def first_apple_age : ℕ := 11

theorem lydia_planting_age : 
  planting_age = first_apple_age - fruit_bearing_time := by sorry

end NUMINAMATH_CALUDE_lydia_planting_age_l2173_217379


namespace NUMINAMATH_CALUDE_probability_sum_10_l2173_217304

def die_faces : Nat := 6

def total_outcomes : Nat := die_faces * die_faces

def favorable_outcomes : Nat := 3 * 2 - 1

theorem probability_sum_10 : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_10_l2173_217304


namespace NUMINAMATH_CALUDE_sum_of_digits_1024_base5_l2173_217342

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_1024_base5 :
  sumList (toBase5 1024) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1024_base5_l2173_217342


namespace NUMINAMATH_CALUDE_meet_at_starting_line_l2173_217362

theorem meet_at_starting_line (henry_time margo_time cameron_time : ℕ) 
  (henry_eq : henry_time = 7)
  (margo_eq : margo_time = 12)
  (cameron_eq : cameron_time = 9) :
  Nat.lcm (Nat.lcm henry_time margo_time) cameron_time = 252 := by
  sorry

end NUMINAMATH_CALUDE_meet_at_starting_line_l2173_217362


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2173_217373

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x - a)^4 / ((a - b) * (a - c)) + (x - b)^4 / ((b - a) * (b - c)) + (x - c)^4 / ((c - a) * (c - b)) =
  x^4 - 2*(a+b+c)*x^3 + (a^2+b^2+c^2+2*a*b+2*b*c+2*c*a)*x^2 - 2*a*b*c*x := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2173_217373


namespace NUMINAMATH_CALUDE_max_length_sum_l2173_217368

def length (k : ℕ) : ℕ := sorry

def has_even_power_prime_factor (n : ℕ) : Prop := sorry

def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem max_length_sum (x y : ℕ) 
  (hx : x > 1) 
  (hy : y > 1) 
  (hsum : x + 3 * y < 1000) 
  (hx_even : has_even_power_prime_factor x) 
  (hy_even : has_even_power_prime_factor y) 
  (hp : smallest_prime_factor x + smallest_prime_factor y ≡ 0 [MOD 3]) :
  ∀ (a b : ℕ), a > 1 → b > 1 → a + 3 * b < 1000 → 
    has_even_power_prime_factor a → has_even_power_prime_factor b → 
    smallest_prime_factor a + smallest_prime_factor b ≡ 0 [MOD 3] →
    length x + length y ≥ length a + length b :=
sorry

end NUMINAMATH_CALUDE_max_length_sum_l2173_217368


namespace NUMINAMATH_CALUDE_even_function_symmetry_l2173_217395

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_symmetry (f : ℝ → ℝ) (h : EvenFunction (fun x ↦ f (x + 1))) :
  ∀ x, f (1 + x) = f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_even_function_symmetry_l2173_217395


namespace NUMINAMATH_CALUDE_complex_sum_argument_l2173_217361

theorem complex_sum_argument : 
  let z : ℂ := Complex.exp (7 * π * I / 60) + Complex.exp (17 * π * I / 60) + 
                Complex.exp (27 * π * I / 60) + Complex.exp (37 * π * I / 60) + 
                Complex.exp (47 * π * I / 60)
  Complex.arg z = 9 * π / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l2173_217361


namespace NUMINAMATH_CALUDE_puppies_per_dog_l2173_217319

theorem puppies_per_dog (num_dogs : ℕ) (total_puppies : ℕ) : 
  num_dogs = 15 → total_puppies = 75 → total_puppies / num_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_dog_l2173_217319


namespace NUMINAMATH_CALUDE_compound_proposition_falsehood_l2173_217388

theorem compound_proposition_falsehood (p q : Prop) 
  (hp : p) (hq : q) : 
  (p ∨ q) ∧ (p ∧ q) ∧ ¬(¬p ∧ q) ∧ (¬p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_falsehood_l2173_217388


namespace NUMINAMATH_CALUDE_angle_BAD_measure_l2173_217396

-- Define the geometric configuration
structure GeometricConfiguration where
  -- We don't need to explicitly define points, just the angles
  angleABC : ℝ
  angleBDE : ℝ
  angleDBE : ℝ
  -- We'll define angleABD in terms of angleABC

-- Define the theorem
theorem angle_BAD_measure (config : GeometricConfiguration) 
  (h1 : config.angleABC = 132)
  (h2 : config.angleBDE = 31)
  (h3 : config.angleDBE = 30)
  : 180 - (180 - config.angleABC) - config.angleBDE - config.angleDBE = 119 := by
  sorry


end NUMINAMATH_CALUDE_angle_BAD_measure_l2173_217396


namespace NUMINAMATH_CALUDE_remainder_of_2743_base12_div_9_l2173_217393

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (n : ℕ) : ℕ :=
  let d0 := n % 12
  let d1 := (n / 12) % 12
  let d2 := (n / 144) % 12
  let d3 := n / 1728
  d3 * 1728 + d2 * 144 + d1 * 12 + d0

/-- The base-12 number 2743 --/
def n : ℕ := 2743

theorem remainder_of_2743_base12_div_9 :
  (base12ToBase10 n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2743_base12_div_9_l2173_217393


namespace NUMINAMATH_CALUDE_history_class_grade_distribution_l2173_217399

theorem history_class_grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℝ) (B_count : ℕ) : 
  total_students = 52 →
  prob_A = 0.5 * prob_B →
  prob_C = 2 * prob_B →
  prob_D = 0.5 * prob_B →
  prob_A + prob_B + prob_C + prob_D = 1 →
  B_count = 13 →
  (0.5 * B_count : ℝ) + B_count + (2 * B_count) + (0.5 * B_count) = total_students := by
  sorry

end NUMINAMATH_CALUDE_history_class_grade_distribution_l2173_217399


namespace NUMINAMATH_CALUDE_line_translation_l2173_217365

-- Define the original line
def original_line (x : ℝ) : ℝ := -2 * x + 3

-- Define the translation
def translation : ℝ := 2

-- Define the translated line
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Theorem statement
theorem line_translation :
  ∀ x : ℝ, translated_line x = original_line x - translation := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l2173_217365


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2173_217316

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * 0.4 = 300 → initial_apples = 750 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2173_217316


namespace NUMINAMATH_CALUDE_one_true_related_proposition_l2173_217322

theorem one_true_related_proposition :
  let P : ℝ → Prop := λ b => b = 3
  let Q : ℝ → Prop := λ b => b^2 = 9
  let converse := ∀ b, Q b → P b
  let negation := ∀ b, ¬(P b) → ¬(Q b)
  let inverse := ∀ b, ¬(Q b) → ¬(P b)
  (converse ∨ negation ∨ inverse) ∧ ¬(converse ∧ negation) ∧ ¬(converse ∧ inverse) ∧ ¬(negation ∧ inverse) :=
by
  sorry

#check one_true_related_proposition

end NUMINAMATH_CALUDE_one_true_related_proposition_l2173_217322


namespace NUMINAMATH_CALUDE_ad_transmission_cost_l2173_217323

/-- The cost of transmitting advertisements during a race -/
theorem ad_transmission_cost
  (num_ads : ℕ)
  (ad_duration : ℕ)
  (cost_per_minute : ℕ)
  (h1 : num_ads = 5)
  (h2 : ad_duration = 3)
  (h3 : cost_per_minute = 4000) :
  num_ads * ad_duration * cost_per_minute = 60000 :=
by sorry

end NUMINAMATH_CALUDE_ad_transmission_cost_l2173_217323


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2173_217344

/-- Given a quadratic inequality ax^2 - 6x + a^2 < 0 with solution set (1,m),
    prove that m = 2 -/
theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) : 
  (∀ x, x ∈ Set.Ioo 1 m ↔ a * x^2 - 6 * x + a^2 < 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2173_217344


namespace NUMINAMATH_CALUDE_group_size_l2173_217378

theorem group_size (over_30 : ℕ) (prob_under_20 : ℚ) :
  over_30 = 90 →
  prob_under_20 = 7/16 →
  ∃ (total : ℕ),
    total = over_30 + (total - over_30) ∧
    (total - over_30) / total = prob_under_20 ∧
    total = 160 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2173_217378


namespace NUMINAMATH_CALUDE_village_apple_trees_l2173_217353

/-- Represents the number of trees in a village -/
structure VillageTrees where
  peach : ℕ
  apple : ℕ

/-- Calculates the total number of apple trees in a village given a 10% sample -/
def totalAppleTrees (sample : ℕ) : ℕ := sample * 10

theorem village_apple_trees (v : VillageTrees) (h : totalAppleTrees 80 = v.apple) :
  v.apple = 800 := by
  sorry

end NUMINAMATH_CALUDE_village_apple_trees_l2173_217353


namespace NUMINAMATH_CALUDE_orange_balloons_count_l2173_217317

/-- Given the initial number of orange balloons and the number of additional orange balloons found,
    prove that the total number of orange balloons is equal to their sum. -/
theorem orange_balloons_count 
  (initial_orange : ℝ) 
  (found_orange : ℝ) : 
  initial_orange + found_orange = 11 :=
by
  sorry

#check orange_balloons_count 9 2

end NUMINAMATH_CALUDE_orange_balloons_count_l2173_217317


namespace NUMINAMATH_CALUDE_solve_for_q_l2173_217391

theorem solve_for_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 16 / 3) : 
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2173_217391


namespace NUMINAMATH_CALUDE_solution_of_quadratic_equation_l2173_217360

theorem solution_of_quadratic_equation :
  {x : ℝ | 2 * (x + 1) = x * (x + 1)} = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_solution_of_quadratic_equation_l2173_217360


namespace NUMINAMATH_CALUDE_increasing_function_parameter_range_l2173_217390

/-- Given that f(x) = x^3 + ax + 1/x is an increasing function on (1/2, +∞),
    prove that a ∈ [13/4, +∞) -/
theorem increasing_function_parameter_range
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x > 1/2, f x = x^3 + a*x + 1/x)
  (h2 : StrictMono f) :
  a ∈ Set.Ici (13/4) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_parameter_range_l2173_217390


namespace NUMINAMATH_CALUDE_marias_carrots_l2173_217381

theorem marias_carrots : 
  ∃ (initial_carrots : ℕ), 
    initial_carrots - 11 + 15 = 52 ∧ 
    initial_carrots = 48 := by
  sorry

end NUMINAMATH_CALUDE_marias_carrots_l2173_217381


namespace NUMINAMATH_CALUDE_A_intersect_B_l2173_217356

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2173_217356


namespace NUMINAMATH_CALUDE_negative_product_of_negatives_l2173_217329

theorem negative_product_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : -a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_of_negatives_l2173_217329


namespace NUMINAMATH_CALUDE_bricks_decrease_by_one_l2173_217351

/-- Represents a brick wall with a given number of rows, total bricks, and bricks in the bottom row. -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the number of bricks in a given row of the wall. -/
def bricksInRow (wall : BrickWall) (row : ℕ) : ℕ :=
  wall.bottomRowBricks - (row - 1)

/-- Theorem stating that for a specific brick wall, the number of bricks decreases by 1 in each row going up. -/
theorem bricks_decrease_by_one (wall : BrickWall)
    (h1 : wall.rows = 5)
    (h2 : wall.totalBricks = 200)
    (h3 : wall.bottomRowBricks = 38) :
    ∀ row : ℕ, row > 1 → row ≤ wall.rows →
      bricksInRow wall row = bricksInRow wall (row - 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_bricks_decrease_by_one_l2173_217351


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2173_217371

theorem sin_330_degrees : 
  Real.sin (330 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2173_217371


namespace NUMINAMATH_CALUDE_gcf_of_180_and_126_l2173_217303

theorem gcf_of_180_and_126 : Nat.gcd 180 126 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_and_126_l2173_217303


namespace NUMINAMATH_CALUDE_quadratic_integer_criterion_l2173_217387

/-- A quadratic trinomial ax^2 + bx + c where a, b, and c are real numbers -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.eval (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Proposition: 2a, a+b, and c are all integers if and only if 
    ax^2 + bx + c takes integer values for all integer x -/
theorem quadratic_integer_criterion (q : QuadraticTrinomial) :
  (∀ x : ℤ, ∃ n : ℤ, q.eval x = n) ↔ 
  (∃ m n p : ℤ, 2 * q.a = m ∧ q.a + q.b = n ∧ q.c = p) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_criterion_l2173_217387


namespace NUMINAMATH_CALUDE_simplify_expression_l2173_217309

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 + 1/2) :
  (a - Real.sqrt 3) * (a + Real.sqrt 3) - a * (a - 6) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2173_217309


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2173_217313

/-- A circle with equation x^2 + y^2 = 4m is tangent to a line with equation x - y = 2√m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = 4*m ∧ x - y = 2*Real.sqrt m) →
  (∀ (x y : ℝ), x^2 + y^2 = 4*m → (x - y ≠ 2*Real.sqrt m ∨ (x - y = 2*Real.sqrt m ∧ 
    ∀ ε > 0, ∃ x' y', (x' - x)^2 + (y' - y)^2 < ε^2 ∧ x'^2 + y'^2 > 4*m))) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2173_217313


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l2173_217318

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → f x > f y) →  -- f is decreasing on ℝ
  f (3 * a) < f (-2 * a + 10) →
  a > 2 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l2173_217318


namespace NUMINAMATH_CALUDE_log_101600_value_l2173_217321

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_101600_value (h : log 102 = 0.3010) : log 101600 = 2.3010 := by
  sorry

end NUMINAMATH_CALUDE_log_101600_value_l2173_217321


namespace NUMINAMATH_CALUDE_least_possible_difference_l2173_217363

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  (∀ (x' y' z' : ℤ), x' < y' → y' < z' → y' - x' > 5 → Even x' → Odd y' → Odd z' → z' - x' ≥ z - x) →
  z - x = 9 := by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l2173_217363


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2173_217397

theorem largest_solution_of_equation :
  let f (x : ℚ) := 7 * (9 * x^2 + 11 * x + 12) - x * (9 * x - 46)
  ∃ (x : ℚ), f x = 0 ∧ (∀ (y : ℚ), f y = 0 → y ≤ x) ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2173_217397


namespace NUMINAMATH_CALUDE_park_area_ratio_l2173_217357

theorem park_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / ((3*s)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_park_area_ratio_l2173_217357


namespace NUMINAMATH_CALUDE_folded_paper_distance_l2173_217358

theorem folded_paper_distance (area : ℝ) (h_area : area = 12) : ℝ :=
  let side_length := Real.sqrt area
  let folded_side_length := Real.sqrt (area / 2)
  let distance := Real.sqrt (2 * folded_side_length ^ 2)
  
  have h_distance : distance = 2 * Real.sqrt 6 := by sorry
  
  distance

end NUMINAMATH_CALUDE_folded_paper_distance_l2173_217358


namespace NUMINAMATH_CALUDE_painter_scenario_proof_l2173_217346

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem: For the given painting scenario, the time to paint the remaining rooms is 49 hours. -/
theorem painter_scenario_proof :
  time_to_paint_remaining 12 7 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_painter_scenario_proof_l2173_217346


namespace NUMINAMATH_CALUDE_money_left_calculation_l2173_217343

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_spent := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem stating that the money left is equal to 50 - 15p -/
theorem money_left_calculation (p : ℝ) : money_left p = 50 - 15 * p := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l2173_217343


namespace NUMINAMATH_CALUDE_triangles_in_decagon_l2173_217315

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 sides -/
def decagonSides : ℕ := 10

/-- Theorem: The number of triangles that can be formed from the vertices of a regular decagon is equal to the number of ways to choose 3 vertices out of 10 -/
theorem triangles_in_decagon :
  trianglesInDecagon = Nat.choose decagonSides 3 := by
  sorry

#eval trianglesInDecagon -- Should output 120

end NUMINAMATH_CALUDE_triangles_in_decagon_l2173_217315


namespace NUMINAMATH_CALUDE_xy_equals_three_l2173_217340

theorem xy_equals_three (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x ≠ y) 
  (h4 : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l2173_217340


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2173_217374

theorem cube_equation_solution (x y z : ℕ) (h : x^3 = 3*y^3 + 9*z^3) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2173_217374


namespace NUMINAMATH_CALUDE_sqrt_inequality_abc_inequality_l2173_217385

-- Problem 1
theorem sqrt_inequality : Real.sqrt 7 + Real.sqrt 13 < 3 + Real.sqrt 11 := by sorry

-- Problem 2
theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * (b^2 + c^2) + b * (c^2 + a^2) + c * (a^2 + b^2) ≥ 6 * a * b * c := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_abc_inequality_l2173_217385


namespace NUMINAMATH_CALUDE_line_segment_translation_l2173_217311

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation vector -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def translatePoint (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem line_segment_translation (A B : Point) (A_new : Point) :
  A = { x := 1, y := 2 } →
  B = { x := 7, y := 5 } →
  A_new = { x := -6, y := -3 } →
  let t : Translation := { dx := A_new.x - A.x, dy := A_new.y - A.y }
  translatePoint B t = { x := 0, y := 0 } := by sorry

end NUMINAMATH_CALUDE_line_segment_translation_l2173_217311


namespace NUMINAMATH_CALUDE_sum_of_digits_product_76_eights_76_fives_l2173_217341

/-- Represents a number consisting of n repetitions of a single digit -/
def repeatedDigitNumber (digit : Nat) (n : Nat) : Nat :=
  digit * (10^n - 1) / 9

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem sum_of_digits_product_76_eights_76_fives : 
  sumOfDigits (repeatedDigitNumber 8 76 * repeatedDigitNumber 5 76) = 304 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_product_76_eights_76_fives_l2173_217341


namespace NUMINAMATH_CALUDE_defeat_points_zero_l2173_217300

/-- Represents the point system and match results for a football competition. -/
structure FootballCompetition where
  victoryPoints : ℕ := 3
  drawPoints : ℕ := 1
  defeatPoints : ℕ
  totalMatches : ℕ := 20
  pointsAfter5Games : ℕ := 14
  minVictoriesRemaining : ℕ := 6
  finalPointTarget : ℕ := 40

/-- Theorem stating that the points for a defeat must be zero under the given conditions. -/
theorem defeat_points_zero (fc : FootballCompetition) : fc.defeatPoints = 0 := by
  sorry

#check defeat_points_zero

end NUMINAMATH_CALUDE_defeat_points_zero_l2173_217300


namespace NUMINAMATH_CALUDE_l_companion_properties_l2173_217338

/-- Definition of an l-companion function -/
def is_l_companion (f : ℝ → ℝ) (l : ℝ) : Prop :=
  l ≠ 0 ∧ Continuous f ∧ ∀ x : ℝ, f (x + l) + l * f x = 0

theorem l_companion_properties (f : ℝ → ℝ) (l : ℝ) (h : is_l_companion f l) :
  (∀ c : ℝ, is_l_companion (λ _ => c) l → c = 0) ∧
  ¬ is_l_companion (λ x => x) l ∧
  ¬ is_l_companion (λ x => x^2) l ∧
  ∃ x : ℝ, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_l_companion_properties_l2173_217338


namespace NUMINAMATH_CALUDE_count_numbers_with_at_most_three_digits_is_900_l2173_217386

/-- Count of positive integers less than 1000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  -- Single-digit numbers
  9 +
  -- Two-digit numbers
  (9 + 144 + 9) +
  -- Three-digit numbers
  (9 + 216 + 504)

/-- Theorem stating that the count of positive integers less than 1000
    with at most three different digits is 900 -/
theorem count_numbers_with_at_most_three_digits_is_900 :
  count_numbers_with_at_most_three_digits = 900 := by
  sorry

#eval count_numbers_with_at_most_three_digits

end NUMINAMATH_CALUDE_count_numbers_with_at_most_three_digits_is_900_l2173_217386


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l2173_217354

theorem sqrt_difference_equals_negative_six_sqrt_two :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = -6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l2173_217354


namespace NUMINAMATH_CALUDE_square_field_area_l2173_217334

/-- The area of a square field with side length 14 meters is 196 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 14 → 
  area = side_length ^ 2 → 
  area = 196 :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2173_217334


namespace NUMINAMATH_CALUDE_quadratic_radicals_same_type_l2173_217328

theorem quadratic_radicals_same_type (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_same_type_l2173_217328


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2173_217310

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 48, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2173_217310


namespace NUMINAMATH_CALUDE_connie_markers_count_l2173_217384

/-- The number of red markers Connie has -/
def red_markers : ℕ := 2315

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

theorem connie_markers_count : total_markers = 3343 := by
  sorry

end NUMINAMATH_CALUDE_connie_markers_count_l2173_217384


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l2173_217349

theorem five_digit_divisibility (a b c d e : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) 
  (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (h6 : e ≤ 9) :
  let n := 10000 * a + 1000 * b + 100 * c + 10 * d + e
  let m := 1000 * a + 100 * b + 10 * d + e
  (∃ k : ℕ, n = k * m) →
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 100 * c = (k - 1) * m :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l2173_217349


namespace NUMINAMATH_CALUDE_table_satisfies_function_l2173_217348

def f (x : ℝ) : ℝ := 100 - 5*x - 5*x^2

theorem table_satisfies_function : 
  (f 0 = 100) ∧ 
  (f 1 = 90) ∧ 
  (f 2 = 70) ∧ 
  (f 3 = 40) ∧ 
  (f 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_table_satisfies_function_l2173_217348


namespace NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_eight_l2173_217359

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points as vectors
variable (O A B C D E : V)

-- Define the scalar k
variable (k : ℝ)

-- Define the condition from the problem
def vector_equation (O A B C D E : V) (k : ℝ) : Prop :=
  4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) + (E - O) = 0

-- Define coplanarity
def coplanar (A B C D E : V) : Prop :=
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) + d • (E - A) = 0

-- State the theorem
theorem coplanar_iff_k_eq_neg_eight
  (O A B C D E : V) (k : ℝ) :
  vector_equation O A B C D E k →
  (coplanar A B C D E ↔ k = -8) :=
sorry

end NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_eight_l2173_217359


namespace NUMINAMATH_CALUDE_container_fullness_l2173_217382

def container_capacity : ℝ := 120
def initial_fullness : ℝ := 0.35
def added_water : ℝ := 48

theorem container_fullness :
  let initial_water := initial_fullness * container_capacity
  let total_water := initial_water + added_water
  let final_fullness := total_water / container_capacity
  final_fullness = 0.75 := by sorry

end NUMINAMATH_CALUDE_container_fullness_l2173_217382


namespace NUMINAMATH_CALUDE_vectors_collinear_l2173_217355

/-- The problem setup -/
structure GeometrySetup where
  -- The coordinate system
  P : ℝ × ℝ
  Q : ℝ × ℝ
  S : ℝ × ℝ
  T : ℝ × ℝ
  N : ℝ × ℝ
  M : ℝ × ℝ
  -- Conditions
  hl : S.1 = -1
  hT : T = (3, 0)
  hPl : S.2 = P.2
  hOP_ST : P.1 * 4 - P.2 * S.2 = 0
  hC : Q.2^2 = 4 * Q.1
  hPQ : ∃ (t : ℝ), (1 - P.1) * t + P.1 = Q.1 ∧ (0 - P.2) * t + P.2 = Q.2
  hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  hN : N = (-1, 0)

/-- The theorem to be proved -/
theorem vectors_collinear (g : GeometrySetup) : 
  ∃ (k : ℝ), (g.M.1 - g.S.1, g.M.2 - g.S.2) = k • (g.Q.1 - g.N.1, g.Q.2 - g.N.2) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l2173_217355


namespace NUMINAMATH_CALUDE_horner_method_example_l2173_217330

def f (x : ℝ) : ℝ := 6*x^5 + 5*x^4 - 4*x^3 + 3*x^2 - 2*x + 1

theorem horner_method_example : f 2 = 249 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l2173_217330


namespace NUMINAMATH_CALUDE_circle_radii_formula_l2173_217394

/-- Given two circles with centers separated by distance c, external common tangents
    intersecting at angle α, and internal common tangents intersecting at angle β,
    this theorem proves the formulas for the radii R and r of the circles. -/
theorem circle_radii_formula (c α β : ℝ) (hc : c > 0) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) :
  ∃ (R r : ℝ),
    R = c * Real.sin ((β + α) / 4) * Real.cos ((β - α) / 4) ∧
    r = c * Real.cos ((β + α) / 4) * Real.sin ((β - α) / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_radii_formula_l2173_217394


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2173_217380

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (4 - m) + y^2 / (m - 2) = 1 → 
    (y = (1/3) * x ∨ y = -(1/3) * x)) → 
  m = 7/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2173_217380


namespace NUMINAMATH_CALUDE_fibonacci_sum_identity_l2173_217369

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_sum_identity (n m : ℕ) (h1 : n ≥ 1) (h2 : m ≥ 0) :
  fib (n + m) = fib (n - 1) * fib m + fib n * fib (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_identity_l2173_217369


namespace NUMINAMATH_CALUDE_smallest_y_for_81_power_gt_7_power_42_l2173_217345

theorem smallest_y_for_81_power_gt_7_power_42 :
  ∃ y : ℕ, (∀ z : ℕ, 81^z ≤ 7^42 → z < y) ∧ 81^y > 7^42 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_81_power_gt_7_power_42_l2173_217345


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l2173_217347

theorem fraction_product_theorem (fractions : Finset (ℕ × ℕ)) : 
  (fractions.card = 48) →
  (∀ (n : ℕ), n ∈ fractions.image Prod.fst → 2 ≤ n ∧ n ≤ 49) →
  (∀ (d : ℕ), d ∈ fractions.image Prod.snd → 2 ≤ d ∧ d ≤ 49) →
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 49 → (fractions.filter (λ f => f.fst = k)).card = 1) →
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 49 → (fractions.filter (λ f => f.snd = k)).card = 1) →
  (∃ (f : ℕ × ℕ), f ∈ fractions ∧ f.fst % f.snd = 0) ∨
  (∃ (subset : Finset (ℕ × ℕ)), subset ⊆ fractions ∧ subset.card ≤ 25 ∧ 
    (subset.prod (λ f => f.fst) % subset.prod (λ f => f.snd) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l2173_217347


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2173_217326

/-- Proves the ratio of square feet painted on Tuesday to Monday is 2:1 -/
theorem tuesday_to_monday_ratio (monday : ℝ) (wednesday : ℝ) (total : ℝ) : 
  monday = 30 →
  wednesday = monday / 2 →
  total = monday + wednesday + (total - monday - wednesday) →
  (total - monday - wednesday) / monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2173_217326
