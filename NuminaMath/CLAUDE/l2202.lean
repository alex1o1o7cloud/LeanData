import Mathlib

namespace NUMINAMATH_CALUDE_equation_I_consecutive_odd_equation_I_not_prime_equation_II_not_consecutive_odd_equation_II_multiple_of_5_equation_II_consecutive_int_l2202_220211

-- Define the necessary types and functions
def ConsecutiveOdd (x y z : ℕ) : Prop := y = x + 2 ∧ z = y + 2
def ConsecutiveInt (x y z w : ℕ) : Prop := y = x + 1 ∧ z = y + 1 ∧ w = z + 1
def MultipleOf5 (n : ℕ) : Prop := ∃ k, n = 5 * k

-- Theorem statements
theorem equation_I_consecutive_odd :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ ConsecutiveOdd x y z ∧ x + y + z = 45 := by sorry

theorem equation_I_not_prime :
  ¬ ∃ x y z : ℕ, x.Prime ∧ y.Prime ∧ z.Prime ∧ x + y + z = 45 := by sorry

theorem equation_II_not_consecutive_odd :
  ¬ ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    ConsecutiveOdd x y z ∧ w = z + 2 ∧ x + y + z + w = 50 := by sorry

theorem equation_II_multiple_of_5 :
  ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    MultipleOf5 x ∧ MultipleOf5 y ∧ MultipleOf5 z ∧ MultipleOf5 w ∧
    x + y + z + w = 50 := by sorry

theorem equation_II_consecutive_int :
  ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    ConsecutiveInt x y z w ∧ x + y + z + w = 50 := by sorry

end NUMINAMATH_CALUDE_equation_I_consecutive_odd_equation_I_not_prime_equation_II_not_consecutive_odd_equation_II_multiple_of_5_equation_II_consecutive_int_l2202_220211


namespace NUMINAMATH_CALUDE_train_speed_l2202_220225

/-- Calculates the speed of a train given its length, time to pass a person moving in the opposite direction, and the person's speed. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) : 
  train_length = 275 →
  passing_time = 15 →
  person_speed = 6 →
  (train_length / 1000) / (passing_time / 3600) - person_speed = 60 :=
by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2202_220225


namespace NUMINAMATH_CALUDE_polynomial_equation_odd_degree_l2202_220293

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- The statement of the theorem -/
theorem polynomial_equation_odd_degree (d : ℕ) :
  (d > 0 ∧ ∃ (P Q : RealPolynomial), 
    (Polynomial.degree P = d) ∧ 
    (∀ x : ℝ, P.eval x ^ 2 + 1 = (x^2 + 1) * Q.eval x ^ 2)) ↔ 
  Odd d :=
sorry

end NUMINAMATH_CALUDE_polynomial_equation_odd_degree_l2202_220293


namespace NUMINAMATH_CALUDE_family_savings_calculation_l2202_220254

def tax_rate : Float := 0.13

def ivan_salary : Float := 55000
def vasilisa_salary : Float := 45000
def mother_salary : Float := 18000
def father_salary : Float := 20000
def son_scholarship : Float := 3000
def mother_pension : Float := 10000
def son_extra_scholarship : Float := 15000

def monthly_expenses : Float := 74000

def net_income (gross_income : Float) : Float :=
  gross_income * (1 - tax_rate)

def total_income_before_may2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income mother_salary + net_income father_salary + son_scholarship

def total_income_may_to_aug2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + son_scholarship

def total_income_from_sept2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + 
  son_scholarship + net_income son_extra_scholarship

theorem family_savings_calculation :
  (total_income_before_may2018 - monthly_expenses = 49060) ∧
  (total_income_may_to_aug2018 - monthly_expenses = 43400) ∧
  (total_income_from_sept2018 - monthly_expenses = 56450) := by
  sorry

end NUMINAMATH_CALUDE_family_savings_calculation_l2202_220254


namespace NUMINAMATH_CALUDE_no_integer_cubes_l2202_220262

theorem no_integer_cubes (a b : ℤ) : 
  a ≥ 1 → b ≥ 1 → 
  (∃ x : ℤ, a^5 * b + 3 = x^3) → 
  (∃ y : ℤ, a * b^5 + 3 = y^3) → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_integer_cubes_l2202_220262


namespace NUMINAMATH_CALUDE_shopping_spree_remaining_amount_l2202_220235

def initial_amount : ℝ := 78

def kite_price_euro : ℝ := 6
def euro_to_usd : ℝ := 1.2

def frisbee_price_pound : ℝ := 7
def pound_to_usd : ℝ := 1.4

def roller_skates_price : ℝ := 15
def roller_skates_discount : ℝ := 0.125

def lego_set_price : ℝ := 25
def lego_set_discount : ℝ := 0.15

def puzzle_price : ℝ := 12
def puzzle_tax : ℝ := 0.075

def remaining_amount : ℝ := initial_amount - 
  (kite_price_euro * euro_to_usd +
   frisbee_price_pound * pound_to_usd +
   roller_skates_price * (1 - roller_skates_discount) +
   lego_set_price * (1 - lego_set_discount) +
   puzzle_price * (1 + puzzle_tax))

theorem shopping_spree_remaining_amount : 
  remaining_amount = 13.725 := by sorry

end NUMINAMATH_CALUDE_shopping_spree_remaining_amount_l2202_220235


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2202_220207

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2202_220207


namespace NUMINAMATH_CALUDE_like_terms_imply_x_power_y_equals_nine_l2202_220298

theorem like_terms_imply_x_power_y_equals_nine (a b x y : ℝ) 
  (h : ∃ (k : ℝ), 3 * a^(x+7) * b^4 = k * (-a^4 * b^(2*y))) : 
  x^y = 9 := by sorry

end NUMINAMATH_CALUDE_like_terms_imply_x_power_y_equals_nine_l2202_220298


namespace NUMINAMATH_CALUDE_pink_roses_count_is_300_l2202_220297

/-- Calculates the number of pink roses in Mrs. Dawson's garden -/
def pink_roses_count : ℕ :=
  let total_rows : ℕ := 30
  let roses_per_row : ℕ := 50
  let red_roses : ℕ := (2 * roses_per_row) / 5
  let blue_roses : ℕ := 1
  let remaining_after_blue : ℕ := roses_per_row - red_roses - blue_roses
  let white_roses : ℕ := remaining_after_blue / 4
  let yellow_roses : ℕ := 2
  let remaining_after_yellow : ℕ := remaining_after_blue - white_roses - yellow_roses
  let purple_roses : ℕ := (3 * remaining_after_yellow) / 8
  let orange_roses : ℕ := 3
  let pink_roses_per_row : ℕ := remaining_after_yellow - purple_roses - orange_roses
  total_rows * pink_roses_per_row

theorem pink_roses_count_is_300 : pink_roses_count = 300 := by
  sorry

end NUMINAMATH_CALUDE_pink_roses_count_is_300_l2202_220297


namespace NUMINAMATH_CALUDE_oil_production_fraction_l2202_220294

/-- Represents the fraction of oil sent for production -/
def x : ℝ := sorry

/-- Initial sulfur concentration -/
def initial_conc : ℝ := 0.015

/-- Sulfur concentration of first replacement oil -/
def first_repl_conc : ℝ := 0.005

/-- Sulfur concentration of second replacement oil -/
def second_repl_conc : ℝ := 0.02

/-- Theorem stating that the fraction of oil sent for production is 1/2 -/
theorem oil_production_fraction :
  (initial_conc - initial_conc * x + first_repl_conc * x - 
   (initial_conc - initial_conc * x + first_repl_conc * x) * x + 
   second_repl_conc * x = initial_conc) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_oil_production_fraction_l2202_220294


namespace NUMINAMATH_CALUDE_all_star_arrangement_l2202_220251

def number_of_arrangements (n_cubs : ℕ) (n_red_sox : ℕ) (n_yankees : ℕ) : ℕ :=
  let n_cubs_with_coach := n_cubs + 1
  let n_teams := 3
  n_teams.factorial * n_cubs_with_coach.factorial * n_red_sox.factorial * n_yankees.factorial

theorem all_star_arrangement :
  number_of_arrangements 4 3 2 = 8640 := by
  sorry

end NUMINAMATH_CALUDE_all_star_arrangement_l2202_220251


namespace NUMINAMATH_CALUDE_broken_seashells_l2202_220295

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (broken : ℕ) : 
  total = 7 → unbroken = 3 → broken = total - unbroken → broken = 4 := by
sorry

end NUMINAMATH_CALUDE_broken_seashells_l2202_220295


namespace NUMINAMATH_CALUDE_max_prob_with_highest_prob_player_second_l2202_220245

/-- Represents a chess player with a winning probability -/
structure Player where
  winProb : ℝ

/-- Represents the order of games played -/
inductive GameOrder
| ABC
| ACB
| BAC
| BCA
| CAB
| CBA

/-- Calculates the probability of winning two consecutive games given a game order -/
def probTwoConsecutiveWins (p₁ p₂ p₃ : ℝ) (order : GameOrder) : ℝ :=
  match order with
  | GameOrder.ABC => 2 * (p₁ * p₂)
  | GameOrder.ACB => 2 * (p₁ * p₃)
  | GameOrder.BAC => 2 * (p₂ * p₁)
  | GameOrder.BCA => 2 * (p₂ * p₃)
  | GameOrder.CAB => 2 * (p₃ * p₁)
  | GameOrder.CBA => 2 * (p₃ * p₂)

theorem max_prob_with_highest_prob_player_second 
  (A B C : Player) 
  (h₁ : 0 < A.winProb) 
  (h₂ : A.winProb < B.winProb) 
  (h₃ : B.winProb < C.winProb) :
  ∀ (order : GameOrder), 
    probTwoConsecutiveWins A.winProb B.winProb C.winProb order ≤ 
    max (probTwoConsecutiveWins A.winProb B.winProb C.winProb GameOrder.CAB)
        (probTwoConsecutiveWins A.winProb B.winProb C.winProb GameOrder.CBA) :=
by sorry

end NUMINAMATH_CALUDE_max_prob_with_highest_prob_player_second_l2202_220245


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2202_220283

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and the downstream travel details. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 196)
  (h3 : downstream_time = 7)
  : ∃ (boat_speed : ℝ), boat_speed = 24 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2202_220283


namespace NUMINAMATH_CALUDE_subtraction_properties_l2202_220200

theorem subtraction_properties (a b : ℝ) : 
  ((a - b)^2 = (b - a)^2) ∧ 
  (|a - b| = |b - a|) ∧ 
  (a - b = -b + a) ∧ 
  (a - b = b - a ↔ a = b) := by
sorry

end NUMINAMATH_CALUDE_subtraction_properties_l2202_220200


namespace NUMINAMATH_CALUDE_minimum_detectors_l2202_220281

/-- Represents a position on the board -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a detector on the board -/
structure Detector :=
  (pos : Position)

/-- Represents a ship on the board -/
structure Ship :=
  (pos : Position)
  (size : Nat)

def boardSize : Nat := 2015
def shipSize : Nat := 1500

/-- Checks if a detector can detect a ship at a given position -/
def canDetect (d : Detector) (s : Ship) : Prop :=
  d.pos.x ≥ s.pos.x ∧ d.pos.x < s.pos.x + s.size ∧
  d.pos.y ≥ s.pos.y ∧ d.pos.y < s.pos.y + s.size

/-- Checks if a set of detectors can determine the position of any ship -/
def canDetermineShipPosition (detectors : List Detector) : Prop :=
  ∀ (s1 s2 : Ship),
    (∀ (d : Detector), d ∈ detectors → (canDetect d s1 ↔ canDetect d s2)) →
    s1 = s2

theorem minimum_detectors :
  ∃ (detectors : List Detector),
    detectors.length = 1030 ∧
    canDetermineShipPosition detectors ∧
    ∀ (d : List Detector),
      d.length < 1030 →
      ¬ canDetermineShipPosition d :=
sorry

end NUMINAMATH_CALUDE_minimum_detectors_l2202_220281


namespace NUMINAMATH_CALUDE_correct_banana_distribution_l2202_220219

def banana_distribution (total dawn lydia donna emily : ℚ) : Prop :=
  total = 550.5 ∧
  dawn = lydia + 93 ∧
  lydia = 80.25 ∧
  donna = emily / 2 ∧
  dawn + lydia + donna + emily = total

theorem correct_banana_distribution :
  ∃ (dawn lydia donna emily : ℚ),
    banana_distribution total dawn lydia donna emily ∧
    dawn = 173.25 ∧
    lydia = 80.25 ∧
    donna = 99 ∧
    emily = 198 := by
  sorry

end NUMINAMATH_CALUDE_correct_banana_distribution_l2202_220219


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2202_220202

theorem sin_585_degrees :
  Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2202_220202


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l2202_220231

theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 = 10*x - 4*y + 6 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 6 - 10*h + 4*k) ∧ h + k = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l2202_220231


namespace NUMINAMATH_CALUDE_pigeonhole_divisibility_l2202_220274

theorem pigeonhole_divisibility (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_divisibility_l2202_220274


namespace NUMINAMATH_CALUDE_f_is_even_l2202_220280

-- Define a real-valued function g
variable (g : ℝ → ℝ)

-- Define the property of g being an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define function f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^4)|

-- Define the property of f being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : IsOdd g) : IsEven (f g) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l2202_220280


namespace NUMINAMATH_CALUDE_clock_hands_right_angle_period_l2202_220217

/-- The number of times clock hands are at right angles in 12 hours -/
def right_angles_per_12_hours : ℕ := 22

/-- The number of times clock hands are at right angles in the given period -/
def given_right_angles : ℕ := 88

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

theorem clock_hands_right_angle_period :
  (given_right_angles / right_angles_per_12_hours) * 12 = hours_per_day :=
sorry

end NUMINAMATH_CALUDE_clock_hands_right_angle_period_l2202_220217


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_10080_l2202_220289

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem least_n_factorial_divisible_by_10080 :
  ∀ n : ℕ, n > 0 → (is_divisible (factorial n) 10080 → n ≥ 7) ∧
  (is_divisible (factorial 7) 10080) :=
sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_10080_l2202_220289


namespace NUMINAMATH_CALUDE_sticker_count_l2202_220237

/-- The number of stickers Karl has -/
def karl_stickers : ℕ := 25

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := karl_stickers + 20

/-- The number of stickers Ben has -/
def ben_stickers : ℕ := ryan_stickers - 10

/-- The total number of stickers placed in the book -/
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem sticker_count : total_stickers = 105 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l2202_220237


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2202_220282

/-- Theorem about a specific triangle ABC -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  S = 5 * Real.sqrt 3 →
  a = 5 →
  (1/2) * a * c * Real.sin B = S →
  B = π / 3 ∧ b = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2202_220282


namespace NUMINAMATH_CALUDE_forty_fifth_turn_turning_position_1978_to_2010_l2202_220286

-- Define the sequence of turning positions
def turningPosition (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    (1 + n / 2) * (n / 2) + 1
  else
    ((n + 1) / 2)^2 + 1

-- Theorem for the 45th turning position
theorem forty_fifth_turn : turningPosition 45 = 530 := by
  sorry

-- Theorem for the turning position between 1978 and 2010
theorem turning_position_1978_to_2010 :
  ∃ n : ℕ, turningPosition n = 1981 ∧
    1978 < turningPosition n ∧ turningPosition n < 2010 ∧
    ∀ m : ℕ, m ≠ n →
      (1978 < turningPosition m → turningPosition m ≥ 2010) ∨
      (turningPosition m ≤ 1978) := by
  sorry

end NUMINAMATH_CALUDE_forty_fifth_turn_turning_position_1978_to_2010_l2202_220286


namespace NUMINAMATH_CALUDE_set_A_equality_l2202_220246

def A : Set ℕ := {x | x < 3}

theorem set_A_equality : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_A_equality_l2202_220246


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2202_220241

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, x^5 + x^4 + 1 = 3^y * 7^z ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2202_220241


namespace NUMINAMATH_CALUDE_pickle_discount_l2202_220243

/-- Calculates the discount on a jar of pickles based on grocery purchases and change received --/
theorem pickle_discount (meat_price meat_weight buns_price lettuce_price tomato_price tomato_weight pickle_price bill change : ℝ) :
  meat_price = 3.5 ∧
  meat_weight = 2 ∧
  buns_price = 1.5 ∧
  lettuce_price = 1 ∧
  tomato_price = 2 ∧
  tomato_weight = 1.5 ∧
  pickle_price = 2.5 ∧
  bill = 20 ∧
  change = 6 →
  pickle_price - ((meat_price * meat_weight + buns_price + lettuce_price + tomato_price * tomato_weight + pickle_price) - (bill - change)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pickle_discount_l2202_220243


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2202_220258

theorem algebraic_expression_value (a b : ℝ) : 
  (a * 2^3 + b * 2 - 7 = -19) → 
  (a * (-2)^3 + b * (-2) - 7 = 5) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2202_220258


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2202_220223

/-- The sum of the infinite series ∑(n=1 to ∞) (4n^2 - 2n + 1) / 3^n is equal to 5 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n^2 - 2 * n + 1) / 3^n) = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2202_220223


namespace NUMINAMATH_CALUDE_increasing_cubic_function_l2202_220253

/-- A function f(x) = x³ + ax - 2 is increasing on [1, +∞) if and only if a ≥ -3 -/
theorem increasing_cubic_function (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => x^3 + a*x - 2)) ↔ a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_l2202_220253


namespace NUMINAMATH_CALUDE_min_red_pieces_l2202_220266

theorem min_red_pieces (w b r : ℕ) : 
  b ≥ w / 2 →
  b ≤ r / 3 →
  w + b ≥ 55 →
  r ≥ 57 ∧ ∀ r', (∃ w' b', b' ≥ w' / 2 ∧ b' ≤ r' / 3 ∧ w' + b' ≥ 55) → r' ≥ r :=
by sorry

end NUMINAMATH_CALUDE_min_red_pieces_l2202_220266


namespace NUMINAMATH_CALUDE_eggs_leftover_l2202_220263

theorem eggs_leftover (daniel : Nat) (eliza : Nat) (fiona : Nat) (george : Nat)
  (h1 : daniel = 53)
  (h2 : eliza = 68)
  (h3 : fiona = 26)
  (h4 : george = 47) :
  (daniel + eliza + fiona + george) % 15 = 14 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l2202_220263


namespace NUMINAMATH_CALUDE_mady_balls_equals_ternary_sum_l2202_220259

/-- Represents the state of a box in Mady's game -/
inductive BoxState
| Empty : BoxState
| OneBall : BoxState
| TwoBalls : BoxState

/-- Converts a natural number to its ternary (base 3) representation -/
def toTernary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then []
      else (m % 3) :: aux (m / 3)
    aux n |>.reverse

/-- Simulates Mady's ball-placing process for a given number of steps -/
def madyProcess (steps : ℕ) : List BoxState :=
  sorry

/-- Counts the number of balls in the final state -/
def countBalls (state : List BoxState) : ℕ :=
  sorry

/-- The main theorem: The number of balls after 2023 steps equals the sum of ternary digits of 2023 -/
theorem mady_balls_equals_ternary_sum :
  countBalls (madyProcess 2023) = (toTernary 2023).sum := by
  sorry

end NUMINAMATH_CALUDE_mady_balls_equals_ternary_sum_l2202_220259


namespace NUMINAMATH_CALUDE_petya_strategy_exists_l2202_220215

theorem petya_strategy_exists (opponent_choice : ℚ → ℚ) : 
  ∃ (a b c : ℚ), 
    ∃ (x y : ℂ), 
      (x^3 + a*x^2 + b*x + c = 0) ∧ 
      (y^3 + a*y^2 + b*y + c = 0) ∧ 
      (y - x = 2014) ∧
      ((a = opponent_choice b ∧ c = opponent_choice 0) ∨ 
       (b = opponent_choice a ∧ c = opponent_choice 0) ∨ 
       (a = opponent_choice c ∧ b = opponent_choice 0)) :=
by sorry

end NUMINAMATH_CALUDE_petya_strategy_exists_l2202_220215


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2202_220205

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = 0 ∧ k = 0) →  -- center at (0,0)
  c = 8 →            -- focus at (0,8)
  a = 4 →            -- vertex at (0,-4)
  c^2 = a^2 + b^2 →  -- relationship between a, b, and c
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2202_220205


namespace NUMINAMATH_CALUDE_square_difference_equality_l2202_220265

theorem square_difference_equality : 1012^2 - 1008^2 - 1006^2 + 1002^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2202_220265


namespace NUMINAMATH_CALUDE_well_diameter_is_six_l2202_220244

def well_depth : ℝ := 24
def well_volume : ℝ := 678.5840131753953

theorem well_diameter_is_six :
  ∃ (d : ℝ), d = 6 ∧ well_volume = π * (d / 2)^2 * well_depth := by sorry

end NUMINAMATH_CALUDE_well_diameter_is_six_l2202_220244


namespace NUMINAMATH_CALUDE_probability_red_or_white_l2202_220236

-- Define the total number of marbles
def total_marbles : ℕ := 60

-- Define the number of blue marbles
def blue_marbles : ℕ := 5

-- Define the number of red marbles
def red_marbles : ℕ := 9

-- Define the number of white marbles
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

-- Theorem statement
theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l2202_220236


namespace NUMINAMATH_CALUDE_trigonometric_sum_simplification_l2202_220227

open Real BigOperators

theorem trigonometric_sum_simplification (n : ℕ) (α : ℝ) :
  (cos α + ∑ k in Finset.range (n - 1), (n.choose k) * cos ((k + 1) * α) + cos ((n + 1) * α) = 
   2^n * (cos (α / 2))^n * cos ((n + 2) * α / 2)) ∧
  (sin α + ∑ k in Finset.range (n - 1), (n.choose k) * sin ((k + 1) * α) + sin ((n + 1) * α) = 
   2^n * (cos (α / 2))^n * sin ((n + 2) * α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_simplification_l2202_220227


namespace NUMINAMATH_CALUDE_matrix_N_property_l2202_220214

theorem matrix_N_property :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (3 : ℝ) • u) ∧
    N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l2202_220214


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2202_220233

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 2 → x + y^3 + z^4 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2202_220233


namespace NUMINAMATH_CALUDE_triangle_height_ratio_l2202_220234

theorem triangle_height_ratio (a b c h₁ h₂ h₃ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧
  (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 ∧
  a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ →
  (h₁ : ℝ) / 20 = (h₂ : ℝ) / 15 ∧ (h₂ : ℝ) / 15 = (h₃ : ℝ) / 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_ratio_l2202_220234


namespace NUMINAMATH_CALUDE_min_unsuccessful_placements_l2202_220216

/-- A board is represented as a function from (Fin 8 × Fin 8) to Int -/
def Board := Fin 8 → Fin 8 → Int

/-- A tetromino is represented as a list of four pairs of coordinates -/
def Tetromino := List (Fin 8 × Fin 8)

/-- A valid board has only 1 and -1 as values -/
def validBoard (b : Board) : Prop :=
  ∀ i j, b i j = 1 ∨ b i j = -1

/-- A valid tetromino has four distinct cells within the board -/
def validTetromino (t : Tetromino) : Prop :=
  t.length = 4 ∧ t.Nodup

/-- The sum of a tetromino's cells on a board -/
def tetrominoSum (b : Board) (t : Tetromino) : Int :=
  t.foldl (fun sum (i, j) => sum + b i j) 0

/-- An unsuccessful placement has a non-zero sum -/
def unsuccessfulPlacement (b : Board) (t : Tetromino) : Prop :=
  tetrominoSum b t ≠ 0

/-- The main theorem -/
theorem min_unsuccessful_placements (b : Board) (h : validBoard b) :
  ∃ (unsuccessfulPlacements : List Tetromino),
    unsuccessfulPlacements.length ≥ 36 ∧
    ∀ t ∈ unsuccessfulPlacements, validTetromino t ∧ unsuccessfulPlacement b t :=
  sorry

end NUMINAMATH_CALUDE_min_unsuccessful_placements_l2202_220216


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2202_220222

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 75 → b = 100 → c^2 = a^2 + b^2 → c = 125 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2202_220222


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l2202_220221

/-- Calculate the interest rate given principal, simple interest, and time -/
theorem calculate_interest_rate (P SI T : ℝ) (h_positive : P > 0 ∧ SI > 0 ∧ T > 0) :
  ∃ R : ℝ, SI = P * R * T / 100 := by
  sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l2202_220221


namespace NUMINAMATH_CALUDE_total_silver_dollars_l2202_220242

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := phung_dollars + 5

/-- The total number of silver dollars owned by all three -/
def total_dollars : ℕ := chiu_dollars + phung_dollars + ha_dollars

theorem total_silver_dollars : total_dollars = 205 := by
  sorry

end NUMINAMATH_CALUDE_total_silver_dollars_l2202_220242


namespace NUMINAMATH_CALUDE_age_difference_l2202_220213

theorem age_difference (D M : ℕ) : 
  (M = 11 * D) →
  (M + 13 = 2 * (D + 13)) →
  (M - D = 40) := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2202_220213


namespace NUMINAMATH_CALUDE_unique_solution_l2202_220220

/-- Represents a three-digit number with distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- The statements made by the students -/
def statements (n : ThreeDigitNumber) : Prop :=
  (n.tens > n.hundreds ∧ n.tens > n.ones) ∧  -- Petya's statement
  (n.ones = 8) ∧                             -- Vasya's statement
  (n.ones > n.hundreds ∧ n.ones > n.tens) ∧  -- Tolya's statement
  (n.ones = (n.hundreds + n.tens) / 2)       -- Dima's statement

/-- The theorem to prove -/
theorem unique_solution :
  ∃! n : ThreeDigitNumber, (∃ (i : Fin 4), ¬statements n) ∧
    (∀ (j : Fin 4), j ≠ i → statements n) ∧
    n.hundreds = 7 ∧ n.tens = 9 ∧ n.ones = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2202_220220


namespace NUMINAMATH_CALUDE_multiples_of_seven_between_15_and_200_l2202_220248

theorem multiples_of_seven_between_15_and_200 : 
  (Finset.filter (fun n => n % 7 = 0 ∧ n > 15 ∧ n < 200) (Finset.range 200)).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_seven_between_15_and_200_l2202_220248


namespace NUMINAMATH_CALUDE_basic_computer_price_l2202_220247

/-- Given the price of a basic computer and a printer, prove that the basic computer costs $2000 -/
theorem basic_computer_price
  (basic_computer printer : ℝ)
  (total_price : basic_computer + printer = 2500)
  (enhanced_total : ∃ (enhanced_total : ℝ), enhanced_total = basic_computer + 500 + printer)
  (printer_ratio : printer = (1/6) * (basic_computer + 500 + printer)) :
  basic_computer = 2000 := by
  sorry

end NUMINAMATH_CALUDE_basic_computer_price_l2202_220247


namespace NUMINAMATH_CALUDE_max_min_sum_theorem_l2202_220291

/-- A function satisfying the given property -/
def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2014

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2014 * x^2013

theorem max_min_sum_theorem (f : ℝ → ℝ) (hf : FunctionProperty f)
  (hM : ∃ M : ℝ, ∀ x : ℝ, g f x ≤ M)
  (hm : ∃ m : ℝ, ∀ x : ℝ, m ≤ g f x)
  (hMm : ∃ M m : ℝ, (∀ x : ℝ, g f x ≤ M ∧ m ≤ g f x) ∧ 
    (∃ x1 x2 : ℝ, g f x1 = M ∧ g f x2 = m)) :
  ∃ M m : ℝ, (∀ x : ℝ, g f x ≤ M ∧ m ≤ g f x) ∧ 
    (∃ x1 x2 : ℝ, g f x1 = M ∧ g f x2 = m) ∧ M + m = -4028 :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_theorem_l2202_220291


namespace NUMINAMATH_CALUDE_max_sum_abc_l2202_220290

theorem max_sum_abc (a b c : ℤ) 
  (h1 : a + b = 2006) 
  (h2 : c - a = 2005) 
  (h3 : a < b) : 
  (∀ x y z : ℤ, x + y = 2006 → z - x = 2005 → x < y → x + y + z ≤ a + b + c) ∧ 
  a + b + c = 5013 :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l2202_220290


namespace NUMINAMATH_CALUDE_pen_drawing_probabilities_l2202_220296

/-- Represents a box of pens with different classes -/
structure PenBox where
  total : Nat
  firstClass : Nat
  secondClass : Nat
  thirdClass : Nat

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem about probabilities when drawing pens from a box -/
theorem pen_drawing_probabilities (box : PenBox)
  (h1 : box.total = 6)
  (h2 : box.firstClass = 3)
  (h3 : box.secondClass = 2)
  (h4 : box.thirdClass = 1)
  (h5 : box.total = box.firstClass + box.secondClass + box.thirdClass) :
  let totalCombinations := choose box.total 2
  let exactlyOneFirstClass := box.firstClass * (box.secondClass + box.thirdClass)
  let noThirdClass := choose (box.firstClass + box.secondClass) 2
  (exactlyOneFirstClass : Rat) / totalCombinations = 3 / 5 ∧
  (noThirdClass : Rat) / totalCombinations = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pen_drawing_probabilities_l2202_220296


namespace NUMINAMATH_CALUDE_trig_identity_l2202_220260

theorem trig_identity (α : Real) (h : Real.sin α + Real.cos α = Real.sqrt 2) :
  Real.tan α + Real.cos α / Real.sin α = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2202_220260


namespace NUMINAMATH_CALUDE_sum_of_combinations_l2202_220224

theorem sum_of_combinations : Finset.sum (Finset.range 5) (fun k => Nat.choose 6 (k + 1)) = 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l2202_220224


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l2202_220285

theorem max_value_inequality (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hsum : a + b + c + d = 100) : 
  (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3) ≤ 2 * 25^(1/3) :=
by sorry

theorem max_value_achievable : 
  ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 100 ∧
  (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3) = 2 * 25^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l2202_220285


namespace NUMINAMATH_CALUDE_elise_comic_book_expense_l2202_220279

theorem elise_comic_book_expense (initial_amount : ℤ) (saved_amount : ℤ) 
  (puzzle_cost : ℤ) (amount_left : ℤ) :
  initial_amount = 8 →
  saved_amount = 13 →
  puzzle_cost = 18 →
  amount_left = 1 →
  initial_amount + saved_amount - puzzle_cost - amount_left = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_elise_comic_book_expense_l2202_220279


namespace NUMINAMATH_CALUDE_feasible_measures_correct_l2202_220203

-- Define the set of all proposed measures
def AllMeasures : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set of infeasible measures
def InfeasibleMeasures : Set ℕ := {4, 5, 6, 8}

-- Define a predicate for feasible measures
def IsFeasibleMeasure (m : ℕ) : Prop :=
  m ∈ AllMeasures ∧ m ∉ InfeasibleMeasures

-- Define the set of feasible measures
def FeasibleMeasures : Set ℕ := {m ∈ AllMeasures | IsFeasibleMeasure m}

-- Theorem statement
theorem feasible_measures_correct :
  FeasibleMeasures = AllMeasures \ InfeasibleMeasures :=
sorry

end NUMINAMATH_CALUDE_feasible_measures_correct_l2202_220203


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2202_220299

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 15) = 12 → x = 129 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2202_220299


namespace NUMINAMATH_CALUDE_average_remaining_is_70_l2202_220271

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : ℕ
  hundred : ℕ

/-- The problem setup for travelers checks -/
def travelersChecksProblem (tc : TravelersChecks) : Prop :=
  tc.fifty + tc.hundred = 30 ∧
  50 * tc.fifty + 100 * tc.hundred = 1800

/-- The average amount of remaining checks after spending 15 $50 checks -/
def averageRemainingAmount (tc : TravelersChecks) : ℚ :=
  (50 * (tc.fifty - 15) + 100 * tc.hundred) / (tc.fifty + tc.hundred - 15)

/-- Theorem stating that the average amount of remaining checks is $70 -/
theorem average_remaining_is_70 (tc : TravelersChecks) :
  travelersChecksProblem tc → averageRemainingAmount tc = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_remaining_is_70_l2202_220271


namespace NUMINAMATH_CALUDE_total_loss_proof_l2202_220288

/-- Represents the capital and loss of an investor -/
structure Investor where
  capital : ℝ
  loss : ℝ

/-- Calculates the total loss given two investors -/
def totalLoss (investor1 investor2 : Investor) : ℝ :=
  investor1.loss + investor2.loss

/-- Theorem: Given two investors with capitals in ratio 1:9 and losses proportional to their investments,
    if one investor loses Rs 900, the total loss is Rs 1000 -/
theorem total_loss_proof (investor1 investor2 : Investor) 
    (h1 : investor1.capital = (1/9) * investor2.capital)
    (h2 : investor1.loss / investor2.loss = investor1.capital / investor2.capital)
    (h3 : investor2.loss = 900) :
    totalLoss investor1 investor2 = 1000 := by
  sorry

#eval totalLoss { capital := 1, loss := 100 } { capital := 9, loss := 900 }

end NUMINAMATH_CALUDE_total_loss_proof_l2202_220288


namespace NUMINAMATH_CALUDE_extreme_value_probability_l2202_220273

-- Define the die outcomes
def DieOutcome := Fin 6

-- Define the probability space
def Ω := DieOutcome × DieOutcome

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the condition for extreme value
def hasExtremeValue (a b : ℕ) : Prop := a^2 > 4*b

-- State the theorem
theorem extreme_value_probability : 
  P {ω : Ω | hasExtremeValue ω.1.val.succ ω.2.val.succ} = 17/36 := by sorry

end NUMINAMATH_CALUDE_extreme_value_probability_l2202_220273


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2202_220210

/-- Prove that if the cost price of 75 articles equals the selling price of 56.25 articles,
    then the gain percent is 33.33%. -/
theorem gain_percent_calculation (C S : ℝ) (h : 75 * C = 56.25 * S) :
  (S - C) / C * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2202_220210


namespace NUMINAMATH_CALUDE_coplanar_points_scalar_l2202_220226

theorem coplanar_points_scalar (O E F G H : EuclideanSpace ℝ (Fin 3)) (m : ℝ) :
  (O = 0) →
  (4 • (E - O) - 3 • (F - O) + 2 • (G - O) + m • (H - O) = 0) →
  (∃ (a b c d : ℝ), a • (E - O) + b • (F - O) + c • (G - O) + d • (H - O) = 0 ∧ 
    (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_coplanar_points_scalar_l2202_220226


namespace NUMINAMATH_CALUDE_ball_probability_l2202_220208

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 17)
  (h5 : red = 3)
  (h6 : purple = 1)
  (h7 : total = white + green + yellow + red + purple) :
  (total - (red + purple)) / total = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2202_220208


namespace NUMINAMATH_CALUDE_cube_cylinder_radius_l2202_220270

/-- Represents a cube with edge length a -/
structure Cube (a : ℝ) where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a cylinder -/
structure Cylinder where
  axis : ℝ × ℝ × ℝ
  radius : ℝ

/-- Given a cube and a cylinder, checks if the cube's vertices A, B, and D₁ lie on the cylinder's surface -/
def verticesOnCylinderSurface (cube : Cube a) (cyl : Cylinder) : Prop :=
  sorry

/-- Checks if the cylinder's axis is parallel to the line DC₁ of the cube -/
def cylinderAxisParallelToDC₁ (cube : Cube a) (cyl : Cylinder) : Prop :=
  sorry

theorem cube_cylinder_radius (a : ℝ) (cube : Cube a) (cyl : Cylinder) 
  (h1 : verticesOnCylinderSurface cube cyl)
  (h2 : cylinderAxisParallelToDC₁ cube cyl) :
  cyl.radius = (3 * a * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_cube_cylinder_radius_l2202_220270


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2202_220209

theorem fraction_sum_equality (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2202_220209


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2202_220239

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 4*x - 6*y

theorem circle_passes_through_points :
  (circle_equation 0 0 = 0) ∧
  (circle_equation 4 0 = 0) ∧
  (circle_equation (-1) 1 = 0) :=
by sorry

#check circle_passes_through_points

end NUMINAMATH_CALUDE_circle_passes_through_points_l2202_220239


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2202_220229

theorem arithmetic_calculations :
  ((1 : ℚ) - 1^2 + 2 * 5 / (1/5) = 49) ∧
  (24 * (1/6 : ℚ) + 24 * (-1/8) - (-24) * (1/2) = 13) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2202_220229


namespace NUMINAMATH_CALUDE_y_value_proof_l2202_220256

/-- Given that 150% of x is equal to 75% of y and x = 24, prove that y = 48 -/
theorem y_value_proof (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2202_220256


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2202_220249

theorem complex_fraction_calculation : 
  (7 + 4/25 + 8.6) / ((4 + 5/7 - 0.005 * 900) / (6/7)) = 63.04 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2202_220249


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l2202_220257

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A number is divisible by both 3 and 5 if it's divisible by 15. -/
def divisible_by_3_and_5 (n : ℕ) : Prop := n % 15 = 0

theorem smallest_perfect_square_divisible_by_3_and_5 :
  (∀ n : ℕ, n > 0 → is_perfect_square n → divisible_by_3_and_5 n → n ≥ 225) ∧
  (is_perfect_square 225 ∧ divisible_by_3_and_5 225) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l2202_220257


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2202_220278

theorem no_positive_integer_solutions
  (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2*n) * y * (y + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2202_220278


namespace NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l2202_220206

def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_angle_and_perimeter 
  (a b c : ℝ) (A B C : ℝ) :
  triangle a b c →
  a > 2 →
  b - c = 1 →
  Real.sqrt 3 * a * Real.cos C = c * Real.sin A →
  (C = Real.pi / 3 ∧
   ∃ (p : ℝ), p = a + b + c ∧ p ≥ 9 + 6 * Real.sqrt 2 ∧
   ∀ (q : ℝ), q = a + b + c → q ≥ p) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l2202_220206


namespace NUMINAMATH_CALUDE_tangent_line_condition_l2202_220276

-- Define the condition for a line being tangent to a circle
def is_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + 2 ∧ x^2 + y^2 = 1 ∧
  ∀ x' y' : ℝ, y' = k * x' + 2 → x'^2 + y'^2 ≥ 1

-- State the theorem
theorem tangent_line_condition :
  (∀ k : ℝ, ¬(k = Real.sqrt 3) → ¬(is_tangent k)) ∧
  ¬(∀ k : ℝ, ¬(is_tangent k) → ¬(k = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l2202_220276


namespace NUMINAMATH_CALUDE_stone_slab_length_l2202_220201

theorem stone_slab_length (total_area : ℝ) (num_slabs : ℕ) (slab_length : ℝ) :
  total_area = 72 →
  num_slabs = 50 →
  (slab_length ^ 2) * num_slabs = total_area →
  slab_length = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stone_slab_length_l2202_220201


namespace NUMINAMATH_CALUDE_picnic_basket_theorem_l2202_220287

/-- Calculate the total cost of a picnic basket given the number of people and item prices -/
def picnic_basket_cost (num_people : ℕ) (sandwich_price fruit_salad_price soda_price snack_price : ℚ) : ℚ :=
  let sandwich_cost := num_people * sandwich_price
  let fruit_salad_cost := num_people * fruit_salad_price
  let soda_cost := num_people * 2 * soda_price
  let snack_cost := 3 * snack_price
  sandwich_cost + fruit_salad_cost + soda_cost + snack_cost

/-- The total cost of the picnic basket is $60 -/
theorem picnic_basket_theorem :
  picnic_basket_cost 4 5 3 2 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_picnic_basket_theorem_l2202_220287


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l2202_220228

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l2202_220228


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2202_220218

/-- Represents a color of a tile -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents a position in the grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 19)

/-- Represents the coloring of the grid -/
def Coloring := Position → Color

/-- Represents a rectangle in the grid -/
structure Rectangle :=
  (topLeft : Position)
  (bottomRight : Position)

/-- Checks if all vertices of a rectangle have the same color -/
def sameColorVertices (r : Rectangle) (c : Coloring) : Prop :=
  let tl := c r.topLeft
  let tr := c ⟨r.topLeft.row, r.bottomRight.col⟩
  let bl := c ⟨r.bottomRight.row, r.topLeft.col⟩
  let br := c r.bottomRight
  tl = tr ∧ tl = bl ∧ tl = br

theorem monochromatic_rectangle_exists (c : Coloring) : 
  ∃ (r : Rectangle), sameColorVertices r c := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2202_220218


namespace NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l2202_220232

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solve_inequality (x : ℝ) :
  f x (-1) ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2 :=
sorry

-- Part 2
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l2202_220232


namespace NUMINAMATH_CALUDE_min_moves_to_exit_l2202_220250

/-- Represents the direction of car movement -/
inductive Direction
| Left
| Right
| Up
| Down

/-- Represents a car in the parking lot -/
structure Car where
  id : Nat
  position : Nat × Nat

/-- Represents the parking lot -/
structure ParkingLot where
  cars : List Car
  width : Nat
  height : Nat

/-- Represents a move in the solution -/
structure Move where
  car : Car
  direction : Direction

/-- Checks if a car can exit the parking lot -/
def canExit (pl : ParkingLot) (car : Car) : Prop :=
  sorry

/-- Checks if a sequence of moves is valid -/
def isValidMoveSequence (pl : ParkingLot) (moves : List Move) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_to_exit (pl : ParkingLot) (car : Car) :
  (∃ (moves : List Move), isValidMoveSequence pl moves ∧ canExit pl car) →
  (∃ (minMoves : List Move), isValidMoveSequence pl minMoves ∧ canExit pl car ∧ minMoves.length = 6) :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_exit_l2202_220250


namespace NUMINAMATH_CALUDE_test_ways_count_l2202_220284

/-- Represents the number of genuine items in the test. -/
def genuine_items : ℕ := 5

/-- Represents the number of defective items in the test. -/
def defective_items : ℕ := 4

/-- Represents the total number of tests conducted. -/
def total_tests : ℕ := 5

/-- Calculates the number of ways to conduct the test under the given conditions. -/
def test_ways : ℕ := sorry

/-- Theorem stating that the number of ways to conduct the test is 480. -/
theorem test_ways_count : test_ways = 480 := by sorry

end NUMINAMATH_CALUDE_test_ways_count_l2202_220284


namespace NUMINAMATH_CALUDE_smaller_number_of_product_323_and_difference_2_l2202_220212

theorem smaller_number_of_product_323_and_difference_2 :
  ∀ x y : ℕ+,
  (x : ℕ) * y = 323 →
  (x : ℕ) - y = 2 →
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_of_product_323_and_difference_2_l2202_220212


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l2202_220238

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (green_green_pairs : ℕ) : 
  green_students = 65 →
  red_students = 85 →
  total_students = 150 →
  total_pairs = 75 →
  green_green_pairs = 30 →
  (green_students + red_students = total_students) →
  (2 * total_pairs = total_students) →
  (∃ (red_red_pairs : ℕ), red_red_pairs = 40 ∧ 
    green_green_pairs + red_red_pairs + (total_pairs - green_green_pairs - red_red_pairs) = total_pairs) :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l2202_220238


namespace NUMINAMATH_CALUDE_range_of_a_l2202_220261

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2202_220261


namespace NUMINAMATH_CALUDE_complex_cube_root_l2202_220264

theorem complex_cube_root (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  ↑a + ↑b * Complex.I = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l2202_220264


namespace NUMINAMATH_CALUDE_min_omega_value_l2202_220292

theorem min_omega_value (f : ℝ → ℝ) (ω φ T : ℝ) : 
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < π →
  T > 0 →
  (∀ t > 0, (∀ x, f (x + t) = f x) → T ≤ t) →
  f T = Real.sqrt 3 / 2 →
  f (π / 9) = 0 →
  3 ≤ ω ∧ ∀ ω' ≥ 0, (
    (∀ x, Real.cos (ω' * x + φ) = Real.cos (ω * x + φ)) →
    (Real.cos (ω' * T + φ) = Real.sqrt 3 / 2) →
    (Real.cos (ω' * π / 9 + φ) = 0) →
    ω' ≥ 3
  ) := by sorry


end NUMINAMATH_CALUDE_min_omega_value_l2202_220292


namespace NUMINAMATH_CALUDE_karlson_candy_theorem_l2202_220275

/-- The number of initial ones on the board -/
def initial_ones : ℕ := 39

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 39

/-- The maximum number of candies Karlson could have eaten -/
def max_candies : ℕ := initial_ones.choose 2

theorem karlson_candy_theorem : 
  max_candies = (initial_ones * (initial_ones - 1)) / 2 := by
  sorry

#eval max_candies

end NUMINAMATH_CALUDE_karlson_candy_theorem_l2202_220275


namespace NUMINAMATH_CALUDE_system_solution_l2202_220230

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^2 - 5*x*y + 6*y^2 = 0
def equation2 (x y : ℝ) : Prop := x^2 + y^2 + x - 11*y - 2 = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(-2/5, -1/5), (4, 2), (-3/5, -1/5), (3, 1)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2202_220230


namespace NUMINAMATH_CALUDE_employee_discount_price_l2202_220255

theorem employee_discount_price (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  wholesale_cost = 200 →
  markup_percentage = 0.20 →
  discount_percentage = 0.25 →
  let retail_price := wholesale_cost * (1 + markup_percentage)
  let discounted_price := retail_price * (1 - discount_percentage)
  discounted_price = 180 := by
sorry


end NUMINAMATH_CALUDE_employee_discount_price_l2202_220255


namespace NUMINAMATH_CALUDE_keychain_arrangements_l2202_220240

/-- The number of keys on the keychain -/
def total_keys : ℕ := 6

/-- The number of keys that must be adjacent -/
def adjacent_keys : ℕ := 3

/-- The number of distinct arrangements of the adjacent keys -/
def adjacent_arrangements : ℕ := Nat.factorial adjacent_keys

/-- The number of distinct arrangements of the remaining groups (adjacent group + other keys) -/
def group_arrangements : ℕ := Nat.factorial (total_keys - adjacent_keys + 1 - 1)

/-- The total number of distinct arrangements -/
def total_arrangements : ℕ := adjacent_arrangements * group_arrangements

theorem keychain_arrangements :
  total_arrangements = 36 :=
sorry

end NUMINAMATH_CALUDE_keychain_arrangements_l2202_220240


namespace NUMINAMATH_CALUDE_largest_possible_a_l2202_220267

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 50) : 
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_a_l2202_220267


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l2202_220268

theorem no_function_satisfies_equation :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (2 * f x) = x + 1998 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l2202_220268


namespace NUMINAMATH_CALUDE_min_distance_complex_l2202_220272

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 2) :
  ∃ (min_val : ℝ), (∀ (w : ℂ), Complex.abs (w - (1 + 2*I)) = 2 → Complex.abs (w + 1) ≥ min_val) ∧
                   (∃ (z₀ : ℂ), Complex.abs (z₀ - (1 + 2*I)) = 2 ∧ Complex.abs (z₀ + 1) = min_val) ∧
                   min_val = 2 * Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l2202_220272


namespace NUMINAMATH_CALUDE_area_is_40_l2202_220277

/-- Two perpendicular lines intersecting at point B -/
structure PerpendicularLines where
  B : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  perpendicular : Bool
  intersect_at_B : Bool
  y_intercept_product : ℝ

/-- The area of triangle BRS given two perpendicular lines -/
def triangle_area (lines : PerpendicularLines) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle BRS is 40 -/
theorem area_is_40 (lines : PerpendicularLines) 
  (h1 : lines.B = (8, 6))
  (h2 : lines.perpendicular = true)
  (h3 : lines.intersect_at_B = true)
  (h4 : lines.y_intercept_product = -24)
  : triangle_area lines = 40 := by
  sorry

end NUMINAMATH_CALUDE_area_is_40_l2202_220277


namespace NUMINAMATH_CALUDE_race_time_difference_per_hurdle_l2202_220204

/-- Given a race with the following parameters:
  * Total distance: 120 meters
  * Hurdles placed every 20 meters
  * Runner A's total time: 36 seconds
  * Runner B's total time: 45 seconds
Prove that the time difference between the runners at each hurdle is 1.5 seconds. -/
theorem race_time_difference_per_hurdle 
  (total_distance : ℝ) 
  (hurdle_interval : ℝ)
  (runner_a_time : ℝ)
  (runner_b_time : ℝ)
  (h1 : total_distance = 120)
  (h2 : hurdle_interval = 20)
  (h3 : runner_a_time = 36)
  (h4 : runner_b_time = 45) :
  (runner_b_time - runner_a_time) / (total_distance / hurdle_interval) = 1.5 := by
sorry

end NUMINAMATH_CALUDE_race_time_difference_per_hurdle_l2202_220204


namespace NUMINAMATH_CALUDE_garland_arrangements_correct_l2202_220269

/-- The number of ways to arrange 6 blue, 7 red, and 9 white light bulbs in a garland,
    such that no two white light bulbs are consecutive -/
def garland_arrangements : ℕ :=
  Nat.choose 13 6 * Nat.choose 14 9

/-- Theorem stating that the number of garland arrangements is correct -/
theorem garland_arrangements_correct :
  garland_arrangements = 3435432 := by sorry

end NUMINAMATH_CALUDE_garland_arrangements_correct_l2202_220269


namespace NUMINAMATH_CALUDE_range_of_f_l2202_220252

open Real

noncomputable def f (x : ℝ) : ℝ := (1 + cos x)^2023 + (1 - cos x)^2023

theorem range_of_f : 
  ∀ y ∈ Set.range (f ∘ (fun x => x * π / 3) ∘ fun t => t * 2 - 1), 2 ≤ y ∧ y ≤ 2^2023 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2202_220252
