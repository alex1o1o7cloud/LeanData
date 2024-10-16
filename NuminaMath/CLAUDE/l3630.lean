import Mathlib

namespace NUMINAMATH_CALUDE_roller_coaster_probability_l3630_363064

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 4

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 3

/-- The probability of choosing a different car on the second ride -/
def prob_second_ride : ℚ := 3 / 4

/-- The probability of choosing a different car on the third ride -/
def prob_third_ride : ℚ := 1 / 2

/-- The probability of riding in 3 different cars over 3 rides -/
def prob_three_different_cars : ℚ := 3 / 8

theorem roller_coaster_probability :
  prob_three_different_cars = 1 * prob_second_ride * prob_third_ride :=
sorry

end NUMINAMATH_CALUDE_roller_coaster_probability_l3630_363064


namespace NUMINAMATH_CALUDE_work_completion_proof_l3630_363074

/-- Represents the time taken to complete the work -/
def total_days : ℕ := 11

/-- Represents the rate at which person a completes the work -/
def rate_a : ℚ := 1 / 24

/-- Represents the rate at which person b completes the work -/
def rate_b : ℚ := 1 / 30

/-- Represents the rate at which person c completes the work -/
def rate_c : ℚ := 1 / 40

/-- Represents the days c left before completion of work -/
def days_c_left : ℕ := 4

theorem work_completion_proof :
  ∃ (x : ℕ), x = days_c_left ∧
  (rate_a + rate_b + rate_c) * (total_days - x : ℚ) + (rate_a + rate_b) * x = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3630_363074


namespace NUMINAMATH_CALUDE_initial_sum_calculation_l3630_363038

theorem initial_sum_calculation (final_amount : ℚ) (interest_rate : ℚ) (years : ℕ) :
  final_amount = 1192 →
  interest_rate = 48.00000000000001 →
  years = 4 →
  final_amount = (1000 : ℚ) + years * interest_rate :=
by
  sorry

#eval (1000 : ℚ) + 4 * 48.00000000000001 -- This should evaluate to 1192

end NUMINAMATH_CALUDE_initial_sum_calculation_l3630_363038


namespace NUMINAMATH_CALUDE_claire_earnings_l3630_363089

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def red_rose_price : ℚ := 3/4

def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses
def red_roses_to_sell : ℕ := red_roses / 2

theorem claire_earnings : 
  (red_roses_to_sell : ℚ) * red_rose_price = 75 := by sorry

end NUMINAMATH_CALUDE_claire_earnings_l3630_363089


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l3630_363018

def total_selling_price : ℕ := 18000
def total_length : ℕ := 300
def loss_per_metre : ℕ := 5

theorem cost_price_per_metre : 
  (total_selling_price / total_length) + loss_per_metre = 65 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l3630_363018


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_equals_one_l3630_363054

theorem triangle_side_ratio_sum_equals_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_A : ℝ := 60 * π / 180
  (a^2 = b^2 + c^2 - 2*b*c*(angle_A.cos)) →
  (c / (a + b) + b / (a + c) = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_equals_one_l3630_363054


namespace NUMINAMATH_CALUDE_unique_a_value_l3630_363043

theorem unique_a_value (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3630_363043


namespace NUMINAMATH_CALUDE_compute_expression_l3630_363044

theorem compute_expression : 10 + 4 * (5 - 10)^3 = -490 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3630_363044


namespace NUMINAMATH_CALUDE_trip_cost_proof_l3630_363026

/-- Calculates the total cost of a trip for two people with a discount -/
def total_cost (original_price discount : ℕ) : ℕ :=
  2 * (original_price - discount)

/-- Proves that the total cost of the trip for two people is $266 -/
theorem trip_cost_proof (original_price discount : ℕ) 
  (h1 : original_price = 147) 
  (h2 : discount = 14) : 
  total_cost original_price discount = 266 := by
  sorry

#eval total_cost 147 14

end NUMINAMATH_CALUDE_trip_cost_proof_l3630_363026


namespace NUMINAMATH_CALUDE_complex_product_equals_sqrt_216_l3630_363034

-- Define complex numbers p and q
variable (p q : ℂ)

-- Define the real number x
variable (x : ℝ)

-- State the theorem
theorem complex_product_equals_sqrt_216 
  (h1 : Complex.abs p = 3)
  (h2 : Complex.abs q = 5)
  (h3 : p * q = x - 3 * Complex.I) :
  x = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_sqrt_216_l3630_363034


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3630_363014

/-- Given that 40 smaller pie crusts each use 1/8 cup of flour,
    prove that 25 larger pie crusts using the same total amount of flour
    will each require 1/5 cup of flour. -/
theorem pie_crust_flour_calculation (small_crusts : ℕ) (large_crusts : ℕ)
  (small_flour : ℚ) (large_flour : ℚ) :
  small_crusts = 40 →
  large_crusts = 25 →
  small_flour = 1/8 →
  small_crusts * small_flour = large_crusts * large_flour →
  large_flour = 1/5 := by
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3630_363014


namespace NUMINAMATH_CALUDE_total_pet_food_is_624_ounces_l3630_363060

/-- Calculates the total weight of pet food in ounces based on given conditions --/
def total_pet_food_ounces : ℕ :=
  let cat_food_bags : ℕ := 3
  let cat_food_weight : ℕ := 3
  let dog_food_bags : ℕ := 4
  let dog_food_weight : ℕ := cat_food_weight + 2
  let bird_food_bags : ℕ := 5
  let bird_food_weight : ℕ := cat_food_weight - 1
  let ounces_per_pound : ℕ := 16
  
  let total_weight_pounds : ℕ := 
    cat_food_bags * cat_food_weight +
    dog_food_bags * dog_food_weight +
    bird_food_bags * bird_food_weight
  
  total_weight_pounds * ounces_per_pound

/-- Theorem stating that the total weight of pet food is 624 ounces --/
theorem total_pet_food_is_624_ounces : 
  total_pet_food_ounces = 624 := by
  sorry

end NUMINAMATH_CALUDE_total_pet_food_is_624_ounces_l3630_363060


namespace NUMINAMATH_CALUDE_kate_lives_on_15_dollars_per_month_kate_has_frugal_lifestyle_l3630_363070

/-- Represents a person living in New York --/
structure NYResident where
  name : String
  monthly_expenses : ℕ
  uses_dumpster_diving : Bool
  has_frugal_habits : Bool

/-- Represents Kate Hashimoto --/
def kate : NYResident :=
  { name := "Kate Hashimoto"
  , monthly_expenses := 15
  , uses_dumpster_diving := true
  , has_frugal_habits := true }

/-- Theorem stating that Kate can live on $15 a month in New York --/
theorem kate_lives_on_15_dollars_per_month :
  kate.monthly_expenses = 15 ∧ kate.uses_dumpster_diving ∧ kate.has_frugal_habits :=
by sorry

/-- Definition of a frugal lifestyle in New York --/
def is_frugal_lifestyle (r : NYResident) : Prop :=
  r.monthly_expenses ≤ 15 ∧ r.uses_dumpster_diving ∧ r.has_frugal_habits

/-- Theorem stating that Kate has a frugal lifestyle --/
theorem kate_has_frugal_lifestyle : is_frugal_lifestyle kate :=
by sorry

end NUMINAMATH_CALUDE_kate_lives_on_15_dollars_per_month_kate_has_frugal_lifestyle_l3630_363070


namespace NUMINAMATH_CALUDE_prob_at_least_one_one_correct_l3630_363041

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one die showing a 1 when two fair 8-sided dice are rolled -/
def prob_at_least_one_one : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing a 1 
    when two fair 8-sided dice are rolled is 15/64 -/
theorem prob_at_least_one_one_correct : 
  prob_at_least_one_one = 1 - (num_sides - 1)^2 / num_sides^2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_one_correct_l3630_363041


namespace NUMINAMATH_CALUDE_fourth_root_of_81_l3630_363090

theorem fourth_root_of_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_81_l3630_363090


namespace NUMINAMATH_CALUDE_ashley_tablet_battery_life_l3630_363099

/-- Represents the battery life of Ashley's tablet -/
structure TabletBattery where
  fullLifeIdle : ℝ  -- Battery life in hours when idle
  fullLifeActive : ℝ  -- Battery life in hours when active
  usedTime : ℝ  -- Total time used since last charge
  activeTime : ℝ  -- Time spent actively using the tablet

/-- Calculates the remaining battery life of Ashley's tablet -/
def remainingBatteryLife (tb : TabletBattery) : ℝ :=
  sorry

/-- Theorem stating that Ashley's tablet will last 8 more hours -/
theorem ashley_tablet_battery_life :
  ∀ (tb : TabletBattery),
    tb.fullLifeIdle = 36 ∧
    tb.fullLifeActive = 4 ∧
    tb.usedTime = 12 ∧
    tb.activeTime = 2 →
    remainingBatteryLife tb = 8 :=
  sorry

end NUMINAMATH_CALUDE_ashley_tablet_battery_life_l3630_363099


namespace NUMINAMATH_CALUDE_triangle_properties_l3630_363024

/-- Triangle ABC with vertices A(5,1), B(1,3), and C(4,4) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from AB in triangle ABC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - p.2 - 4 = 0

/-- The circumcircle of triangle ABC -/
def circumcircle (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => (p.1 - 3)^2 + (p.2 - 2)^2 = 5

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = (5, 1)) 
  (h2 : t.B = (1, 3)) 
  (h3 : t.C = (4, 4)) : 
  (∀ p, altitude t p ↔ 2 * p.1 - p.2 - 4 = 0) ∧ 
  (∀ p, circumcircle t p ↔ (p.1 - 3)^2 + (p.2 - 2)^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3630_363024


namespace NUMINAMATH_CALUDE_discount_difference_l3630_363046

theorem discount_difference : 
  let initial_amount : ℝ := 12000
  let single_discount : ℝ := 0.3
  let first_successive_discount : ℝ := 0.2
  let second_successive_discount : ℝ := 0.1
  let single_discounted_amount : ℝ := initial_amount * (1 - single_discount)
  let successive_discounted_amount : ℝ := initial_amount * (1 - first_successive_discount) * (1 - second_successive_discount)
  successive_discounted_amount - single_discounted_amount = 240 :=
by sorry

end NUMINAMATH_CALUDE_discount_difference_l3630_363046


namespace NUMINAMATH_CALUDE_hundred_with_six_digits_l3630_363051

theorem hundred_with_six_digits (x : ℕ) (h : x ≠ 0) (h2 : x < 10) :
  (100 * x + 10 * x + x) - (10 * x + x) = 100 * x :=
by sorry

end NUMINAMATH_CALUDE_hundred_with_six_digits_l3630_363051


namespace NUMINAMATH_CALUDE_movie_ticket_price_decrease_l3630_363092

theorem movie_ticket_price_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100)
  (h2 : new_price = 80) : 
  (old_price - new_price) / old_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_price_decrease_l3630_363092


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_inequality_l3630_363002

theorem sum_and_reciprocal_inequality (x : ℝ) (hx : x > 0) : 
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_inequality_l3630_363002


namespace NUMINAMATH_CALUDE_car_travel_time_l3630_363080

/-- Given a car and a train traveling between two stations, this theorem proves
    the time taken by the car to reach the destination. -/
theorem car_travel_time (car_time train_time : ℝ) : 
  train_time = car_time + 2 →  -- The train takes 2 hours longer than the car
  car_time + train_time = 11 → -- The combined time is 11 hours
  car_time = 4.5 := by
  sorry

#check car_travel_time

end NUMINAMATH_CALUDE_car_travel_time_l3630_363080


namespace NUMINAMATH_CALUDE_correct_propositions_l3630_363084

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_not_in_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)

-- Axioms for the properties of these operations
axiom perpendicular_sym {α β : Plane} : perpendicular α β → perpendicular β α
axiom parallel_sym {α β : Plane} : parallel α β → parallel β α
axiom line_parallel_plane_sym {l : Line} {α : Plane} : line_parallel_plane l α → line_parallel_plane l α

-- The theorem to be proved
theorem correct_propositions 
  (m n : Line) (α β γ : Plane) : 
  (perpendicular α β ∧ line_perpendicular_plane m β ∧ line_not_in_plane m α → line_parallel_plane m α) ∧
  (parallel α β ∧ line_in_plane m α → line_parallel_plane m β) ∧
  ¬(perpendicular α β ∧ line_parallel_line n m → line_parallel_plane n α ∧ line_parallel_plane n β) ∧
  ¬(perpendicular α β ∧ perpendicular α γ → parallel β γ) :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_l3630_363084


namespace NUMINAMATH_CALUDE_evaluate_expression_l3630_363052

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/2) (hy : y = 1/3) (hz : z = -3) : 
  x^2 * y^3 * z^2 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3630_363052


namespace NUMINAMATH_CALUDE_M_equals_N_l3630_363056

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l3630_363056


namespace NUMINAMATH_CALUDE_min_cuts_for_quadrilaterals_l3630_363057

/-- Represents the number of cuts made on the paper -/
def num_cuts : ℕ := 1699

/-- Represents the number of quadrilaterals to be obtained -/
def target_quadrilaterals : ℕ := 100

/-- Represents the initial number of vertices in a square -/
def initial_vertices : ℕ := 4

/-- Represents the maximum number of new vertices added per cut -/
def max_new_vertices_per_cut : ℕ := 4

/-- Represents the number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

theorem min_cuts_for_quadrilaterals :
  (num_cuts + 1 = target_quadrilaterals) ∧
  (initial_vertices + num_cuts * max_new_vertices_per_cut ≥ target_quadrilaterals * vertices_per_quadrilateral) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_quadrilaterals_l3630_363057


namespace NUMINAMATH_CALUDE_expression_evaluation_l3630_363048

theorem expression_evaluation : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3630_363048


namespace NUMINAMATH_CALUDE_fraction_inequality_l3630_363095

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 → (4 * x + 3 ≤ 9 - 3 * x ↔ -1 ≤ x ∧ x ≤ 6/7) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3630_363095


namespace NUMINAMATH_CALUDE_pencil_weight_l3630_363073

theorem pencil_weight (total_weight : ℝ) (case_weight : ℝ) (num_pencils : ℕ) 
  (h1 : total_weight = 11.14)
  (h2 : case_weight = 0.5)
  (h3 : num_pencils = 14) :
  (total_weight - case_weight) / num_pencils = 0.76 := by
sorry

end NUMINAMATH_CALUDE_pencil_weight_l3630_363073


namespace NUMINAMATH_CALUDE_not_perfect_power_of_ten_sixes_and_zeros_l3630_363096

def is_composed_of_ten_sixes_and_zeros (n : ℕ) : Prop :=
  ∃ k, n = 6666666666 * 10^k

theorem not_perfect_power_of_ten_sixes_and_zeros (n : ℕ) 
  (h : is_composed_of_ten_sixes_and_zeros n) : 
  ¬ ∃ (a b : ℕ), b > 1 ∧ n = a^b :=
sorry

end NUMINAMATH_CALUDE_not_perfect_power_of_ten_sixes_and_zeros_l3630_363096


namespace NUMINAMATH_CALUDE_smallest_winning_k_l3630_363065

/-- Represents a square on the game board --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the game state --/
structure GameState where
  board : Square → Option Char
  mike_moves : Nat
  harry_moves : Nat

/-- Checks if a sequence forms a winning pattern --/
def is_winning_sequence (s : List Char) : Bool :=
  s = ['H', 'M', 'M'] || s = ['M', 'M', 'H']

/-- Checks if there's a winning sequence on the board --/
def has_winning_sequence (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for Mike --/
def MikeStrategy := Nat → List Square

/-- Represents a strategy for Harry --/
def HarryStrategy := GameState → List Square

/-- Simulates a game with given strategies --/
def play_game (k : Nat) (mike_strat : MikeStrategy) (harry_strat : HarryStrategy) : Bool :=
  sorry

/-- Defines what it means for Mike to have a winning strategy --/
def mike_has_winning_strategy (k : Nat) : Prop :=
  ∃ (mike_strat : MikeStrategy), ∀ (harry_strat : HarryStrategy), 
    play_game k mike_strat harry_strat = true

/-- The main theorem stating that 16 is the smallest k for which Mike has a winning strategy --/
theorem smallest_winning_k : 
  (mike_has_winning_strategy 16) ∧ 
  (∀ k < 16, ¬(mike_has_winning_strategy k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_k_l3630_363065


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_eq_77_l3630_363059

-- Define the function f
def f (a : ℝ) : ℝ := 2 * a - 3

-- Define the function F
def F (a b : ℝ) : ℝ := b * (a + b)

-- Theorem statement
theorem F_of_4_f_of_5_eq_77 : F 4 (f 5) = 77 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_eq_77_l3630_363059


namespace NUMINAMATH_CALUDE_value_of_expression_l3630_363069

theorem value_of_expression (a b : ℝ) (h : a - 2*b = -2) : 4 - 2*a + 4*b = 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3630_363069


namespace NUMINAMATH_CALUDE_payment_divisible_by_25_l3630_363068

theorem payment_divisible_by_25 (B : ℕ) (h : B ≤ 9) : 
  ∃ k : ℕ, 2000 + 100 * B + 5 = 25 * k := by
  sorry

end NUMINAMATH_CALUDE_payment_divisible_by_25_l3630_363068


namespace NUMINAMATH_CALUDE_black_length_is_two_l3630_363071

def pencil_length : ℝ := 6
def purple_length : ℝ := 3
def blue_length : ℝ := 1

theorem black_length_is_two :
  pencil_length - purple_length - blue_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_black_length_is_two_l3630_363071


namespace NUMINAMATH_CALUDE_circle_sum_is_twenty_l3630_363006

def CircleSum (digits : Finset ℕ) (sum : ℕ) : Prop :=
  ∃ (a b c d e f x : ℕ),
    digits = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    5 ∈ digits ∧
    2 ∈ digits ∧
    x + a + b + 5 = sum ∧
    x + e + f + 2 = sum ∧
    5 + c + d + 2 = sum

theorem circle_sum_is_twenty :
  ∃ (digits : Finset ℕ) (sum : ℕ), CircleSum digits sum ∧ sum = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_is_twenty_l3630_363006


namespace NUMINAMATH_CALUDE_merchant_markup_problem_l3630_363094

theorem merchant_markup_problem (markup_percentage : ℝ) : 
  (∀ cost_price : ℝ, cost_price > 0 →
    let marked_price := cost_price * (1 + markup_percentage / 100)
    let discounted_price := marked_price * (1 - 25 / 100)
    let profit_percentage := (discounted_price - cost_price) / cost_price * 100
    profit_percentage = 20) →
  markup_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_merchant_markup_problem_l3630_363094


namespace NUMINAMATH_CALUDE_sashas_work_portion_l3630_363081

theorem sashas_work_portion (car1 car2 car3 : ℚ) 
  (h1 : car1 = 1 / 3)
  (h2 : car2 = 1 / 5)
  (h3 : car3 = 1 / 15) :
  (car1 + car2 + car3) / 3 = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_sashas_work_portion_l3630_363081


namespace NUMINAMATH_CALUDE_unfactorable_quartic_l3630_363045

theorem unfactorable_quartic :
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_unfactorable_quartic_l3630_363045


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3630_363023

theorem min_value_sum_fractions (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a ≥ 9 ∧
  ((a + b + k) / c + (a + c + k) / b + (b + c + k) / a = 9 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3630_363023


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3630_363000

/-- A function representing inverse proportionality --/
def inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x * x = k

/-- The main theorem --/
theorem inverse_proportion_problem (f : ℝ → ℝ) 
  (h1 : inversely_proportional f) 
  (h2 : f (-10) = 5) : 
  f (-4) = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3630_363000


namespace NUMINAMATH_CALUDE_square_side_length_l3630_363032

theorem square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + 17 > 0 ∧ 
  x + 11 > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 → 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3630_363032


namespace NUMINAMATH_CALUDE_alice_burger_spending_l3630_363072

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The number of burgers Alice purchases each day -/
def burgers_per_day : ℕ := 4

/-- The cost of each burger in dollars -/
def burger_cost : ℕ := 13

/-- The total amount Alice spent on burgers in June -/
def total_spent : ℕ := june_days * burgers_per_day * burger_cost

theorem alice_burger_spending :
  total_spent = 1560 := by
  sorry

end NUMINAMATH_CALUDE_alice_burger_spending_l3630_363072


namespace NUMINAMATH_CALUDE_max_value_on_curve_l3630_363040

noncomputable def max_value (b : ℝ) : ℝ :=
  if 0 < b ∧ b ≤ 4 then b^2 / 4 + 4 else 2 * b

theorem max_value_on_curve (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 → x^2 + 2*y ≤ max_value b) ∧
  (∃ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 ∧ x^2 + 2*y = max_value b) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l3630_363040


namespace NUMINAMATH_CALUDE_probability_not_adjacent_correct_l3630_363027

/-- The number of chairs in a row -/
def total_chairs : ℕ := 10

/-- The number of available chairs (excluding the last one) -/
def available_chairs : ℕ := total_chairs - 1

/-- The probability that two people don't sit next to each other 
    when randomly selecting from the first 9 chairs of 10 -/
def probability_not_adjacent : ℚ := 7 / 9

/-- Theorem stating the probability of two people not sitting adjacent 
    when randomly selecting from 9 out of 10 chairs -/
theorem probability_not_adjacent_correct : 
  probability_not_adjacent = 1 - (2 * available_chairs - 2) / (available_chairs * (available_chairs - 1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_correct_l3630_363027


namespace NUMINAMATH_CALUDE_division_equation_problem_l3630_363062

theorem division_equation_problem (A B C : ℕ) : 
  (∃ (q : ℕ), A = B * q + 8) → -- A ÷ B = C with remainder 8
  (A + B + C = 2994) →         -- Sum condition
  (A = 8 ∨ A = 2864) :=        -- Conclusion
by
  sorry

end NUMINAMATH_CALUDE_division_equation_problem_l3630_363062


namespace NUMINAMATH_CALUDE_triangle_vector_sum_zero_l3630_363033

-- Define a triangle as a structure with three points
structure Triangle (V : Type*) [AddCommGroup V] :=
  (A B C : V)

-- Theorem statement
theorem triangle_vector_sum_zero {V : Type*} [AddCommGroup V] (t : Triangle V) :
  t.B - t.A + t.C - t.B + t.A - t.C = (0 : V) := by sorry

end NUMINAMATH_CALUDE_triangle_vector_sum_zero_l3630_363033


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l3630_363015

theorem rightmost_three_digits_of_7_to_1987 :
  7^1987 ≡ 643 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l3630_363015


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3630_363013

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 3/2) : x^2 + 1/x^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3630_363013


namespace NUMINAMATH_CALUDE_equidistant_function_property_l3630_363028

open Complex

theorem equidistant_function_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ z : ℂ, abs ((a + b * I) * z - z) = abs ((a + b * I) * z)) →
  abs (a + b * I) = 5 →
  b^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l3630_363028


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_5_6_l3630_363076

theorem greatest_four_digit_divisible_by_3_5_6 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 9990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_5_6_l3630_363076


namespace NUMINAMATH_CALUDE_zhuzhuxia_defeats_l3630_363007

/-- Represents the game state after a certain number of rounds -/
structure GameState where
  rounds : ℕ
  monsters_defeated : ℕ

/-- Theorem stating that after 8 rounds with 20 monsters defeated, Zhuzhuxia has been defeated 8 times -/
theorem zhuzhuxia_defeats (game : GameState) 
  (h1 : game.rounds = 8) 
  (h2 : game.monsters_defeated = 20) : 
  (game.rounds : ℕ) = 8 := by sorry

end NUMINAMATH_CALUDE_zhuzhuxia_defeats_l3630_363007


namespace NUMINAMATH_CALUDE_function_roots_bound_l3630_363011

/-- The function f(x) defined with given parameters has no more than 14 positive roots -/
theorem function_roots_bound 
  (a b c d : ℝ) 
  (k l m p q r : ℕ) 
  (h1 : k ≥ l ∧ l ≥ m) 
  (h2 : p ≥ q ∧ q ≥ r) :
  let f : ℝ → ℝ := λ x => a*(x+1)^k * (x+2)^p + b*(x+1)^l * (x+2)^q + c*(x+1)^m * (x+2)^r - d
  ∃ (S : Finset ℝ), (∀ x ∈ S, x > 0 ∧ f x = 0) ∧ Finset.card S ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_function_roots_bound_l3630_363011


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l3630_363088

/-- Given a parabola y = ax² + bx + c with 0 < 2a < b, and points A(1/2, y₁), B(0, y₂), C(-1, y₃) on the parabola,
    prove that y₁ > y₂ > y₃ -/
theorem parabola_point_ordering (a b c y₁ y₂ y₃ : ℝ) :
  0 < 2 * a → 2 * a < b →
  y₁ = a * (1/2)^2 + b * (1/2) + c →
  y₂ = c →
  y₃ = a * (-1)^2 + b * (-1) + c →
  y₁ > y₂ ∧ y₂ > y₃ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l3630_363088


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_350_l3630_363083

theorem closest_perfect_square_to_350 :
  ∀ n : ℕ, n ≠ 19 → (n ^ 2 : ℤ) ≠ 361 → |350 - (19 ^ 2 : ℤ)| ≤ |350 - (n ^ 2 : ℤ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_350_l3630_363083


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3630_363030

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- The point P -/
def P : ℝ × ℝ := (4, 0)

/-- The circle ⊙P -/
def circleP : Circle := { center := P, radius := 5 }

/-- The line y = kx + 2 -/
def line (k : ℝ) : Line := { k := k, b := 2 }

/-- Theorem: The line y = kx + 2 (k ≠ 0) always intersects the circle ⊙P -/
theorem line_intersects_circle (k : ℝ) (h : k ≠ 0) : 
  ∃ (x y : ℝ), (y = k * x + 2) ∧ 
  ((x - circleP.center.1)^2 + (y - circleP.center.2)^2 = circleP.radius^2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3630_363030


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l3630_363021

theorem quadratic_function_max_value (m : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + 2*x + m
  f (1/2) > f (-1) ∧ f (1/2) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l3630_363021


namespace NUMINAMATH_CALUDE_smallest_a1_l3630_363063

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 13 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ -/
theorem smallest_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  ∀ a₁ : ℝ, a 1 = a₁ → a₁ ≥ 13 / 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a1_l3630_363063


namespace NUMINAMATH_CALUDE_R_equals_eleven_l3630_363010

def F : ℝ := 2^121 - 1

def Q : ℕ := 120

theorem R_equals_eleven :
  Real.sqrt (Real.log (1 + F) / Real.log 2) = 11 := by sorry

end NUMINAMATH_CALUDE_R_equals_eleven_l3630_363010


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3630_363055

/-- The equation of the line passing through the intersection points of two circles -/
theorem intersection_line_equation (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) :
  c1_center = (-8, -6) →
  c2_center = (4, 5) →
  r1 = 10 →
  r2 = Real.sqrt 41 →
  ∃ (x y : ℝ), ((x - c1_center.1)^2 + (y - c1_center.2)^2 = r1^2) ∧
                ((x - c2_center.1)^2 + (y - c2_center.2)^2 = r2^2) ∧
                (x + y = -59/11) :=
by sorry


end NUMINAMATH_CALUDE_intersection_line_equation_l3630_363055


namespace NUMINAMATH_CALUDE_opposite_of_negative_eleven_l3630_363049

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_negative_eleven : opposite (-11) = 11 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eleven_l3630_363049


namespace NUMINAMATH_CALUDE_stone_slab_length_l3630_363042

theorem stone_slab_length (n : ℕ) (total_area : ℝ) (h1 : n = 30) (h2 : total_area = 120) :
  ∃ (slab_length : ℝ), slab_length > 0 ∧ n * slab_length^2 = total_area ∧ slab_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_length_l3630_363042


namespace NUMINAMATH_CALUDE_subtracted_number_l3630_363012

theorem subtracted_number (x N : ℤ) (h1 : 3 * x = (N - x) + 16) (h2 : x = 13) : N = 36 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3630_363012


namespace NUMINAMATH_CALUDE_height_difference_is_4b_minus_8_l3630_363037

/-- A circle inside a parabola y = 4x^2, tangent at two points -/
structure TangentCircle where
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- x-coordinate of one tangent point (the other is -a) -/
  a : ℝ
  /-- The point (a, 4a^2) lies on the parabola -/
  tangent_on_parabola : 4 * a^2 = 4 * a^2
  /-- The point (a, 4a^2) lies on the circle -/
  tangent_on_circle : a^2 + (4 * a^2 - b)^2 = (b - 4 * a^2)^2 + a^2
  /-- Relation between a and b derived from tangency condition -/
  tangency_relation : 4 * b - a^2 = 8

/-- The difference in height between the circle's center and tangent points -/
def height_difference (c : TangentCircle) : ℝ := c.b - 4 * c.a^2

/-- Theorem: The height difference is always 4b - 8 -/
theorem height_difference_is_4b_minus_8 (c : TangentCircle) :
  height_difference c = 4 * c.b - 8 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_is_4b_minus_8_l3630_363037


namespace NUMINAMATH_CALUDE_inequality_proof_l3630_363067

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (3 * x^2 - x) / (1 + x^2) + (3 * y^2 - y) / (1 + y^2) + (3 * z^2 - z) / (1 + z^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3630_363067


namespace NUMINAMATH_CALUDE_divide_3x8_rectangle_into_trominoes_l3630_363004

/-- Represents an L-shaped tromino -/
structure LTromino :=
  (cells : Nat)

/-- Represents a rectangle -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Number of ways to divide a rectangle into L-shaped trominoes -/
def divideRectangle (r : Rectangle) (t : LTromino) : Nat :=
  sorry

/-- Theorem: The number of ways to divide a 3 × 8 rectangle into L-shaped trominoes is 16 -/
theorem divide_3x8_rectangle_into_trominoes :
  let r := Rectangle.mk 8 3
  let t := LTromino.mk 3
  divideRectangle r t = 16 := by
  sorry

end NUMINAMATH_CALUDE_divide_3x8_rectangle_into_trominoes_l3630_363004


namespace NUMINAMATH_CALUDE_miranda_monthly_savings_l3630_363017

def heels_price : ℕ := 210
def shipping_cost : ℕ := 20
def sister_contribution : ℕ := 50
def saving_months : ℕ := 3
def total_paid : ℕ := 230

theorem miranda_monthly_savings :
  (total_paid - sister_contribution) / saving_months = 60 := by
  sorry

end NUMINAMATH_CALUDE_miranda_monthly_savings_l3630_363017


namespace NUMINAMATH_CALUDE_annas_money_l3630_363077

theorem annas_money (initial_amount : ℚ) : 
  (initial_amount * (1 - 3/8) * (1 - 1/5) = 36) → initial_amount = 72 := by
  sorry

end NUMINAMATH_CALUDE_annas_money_l3630_363077


namespace NUMINAMATH_CALUDE_equation_solution_l3630_363001

theorem equation_solution : 
  ∃ x : ℝ, x + (x + 1) + (x + 2) + (x + 3) = 34 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3630_363001


namespace NUMINAMATH_CALUDE_arccos_of_negative_one_equals_pi_l3630_363058

theorem arccos_of_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_negative_one_equals_pi_l3630_363058


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3630_363085

theorem perfect_square_condition (m : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = y^2) → 
  m = 196 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3630_363085


namespace NUMINAMATH_CALUDE_smallest_square_division_smallest_square_division_is_two_total_squares_l3630_363050

theorem smallest_square_division (n : ℕ) : n > 0 ∧ 4*n - 4 = 2*n → n ≥ 2 :=
by sorry

theorem smallest_square_division_is_two : 
  ∃ (n : ℕ), n > 0 ∧ 4*n - 4 = 2*n ∧ ∀ (m : ℕ), (m > 0 ∧ 4*m - 4 = 2*m) → n ≤ m :=
by sorry

theorem total_squares (n : ℕ) : n > 0 ∧ 4*n - 4 = 2*n → n^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_division_smallest_square_division_is_two_total_squares_l3630_363050


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l3630_363098

theorem last_digit_of_sum (n : ℕ) : 
  (54^2020 + 28^2022) % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l3630_363098


namespace NUMINAMATH_CALUDE_unique_base_for_256_l3630_363053

theorem unique_base_for_256 : ∃! (b : ℕ), b > 0 ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_for_256_l3630_363053


namespace NUMINAMATH_CALUDE_monotonic_decreasing_cubic_function_l3630_363047

theorem monotonic_decreasing_cubic_function (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-1 : ℝ) 1, 
    x < y → (a * x^3 - 3*x) > (a * y^3 - 3*y)) →
  0 < a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_cubic_function_l3630_363047


namespace NUMINAMATH_CALUDE_total_texts_is_forty_l3630_363087

/-- The number of texts Sydney sent to Allison and Brittney on both days -/
def total_texts (monday_texts_per_person tuesday_texts_per_person : ℕ) : ℕ :=
  2 * (monday_texts_per_person + tuesday_texts_per_person)

/-- Theorem stating that the total number of texts is 40 -/
theorem total_texts_is_forty :
  total_texts 5 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_texts_is_forty_l3630_363087


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3630_363079

/-- A point in the second quadrant with given distances to the axes has specific coordinates -/
theorem point_in_second_quadrant (P : ℝ × ℝ) : 
  P.1 < 0 ∧ P.2 > 0 ∧  -- P is in the second quadrant
  |P.2| = 5 ∧          -- distance to x-axis is 5
  |P.1| = 3            -- distance to y-axis is 3
  → P = (-3, 5) := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3630_363079


namespace NUMINAMATH_CALUDE_find_x_l3630_363022

theorem find_x : ∃ x : ℝ, 0.65 * x = 0.20 * 682.50 ∧ x = 210 := by sorry

end NUMINAMATH_CALUDE_find_x_l3630_363022


namespace NUMINAMATH_CALUDE_horner_v2_at_2_l3630_363016

def horner_polynomial (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

def horner_v2 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := 2*x + 1
  v1 * x

theorem horner_v2_at_2 : horner_v2 2 = 10 := by
  sorry

#eval horner_v2 2

end NUMINAMATH_CALUDE_horner_v2_at_2_l3630_363016


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3630_363036

theorem imaginary_part_of_z : 
  let z : ℂ := Complex.I * ((-1 : ℂ) + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3630_363036


namespace NUMINAMATH_CALUDE_sine_addition_formula_l3630_363005

theorem sine_addition_formula (x y z : ℝ) :
  Real.sin (x + y) * Real.cos z + Real.cos (x + y) * Real.sin z = Real.sin (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_sine_addition_formula_l3630_363005


namespace NUMINAMATH_CALUDE_labrador_starting_weight_l3630_363082

/-- The starting weight of the labrador puppy -/
def L : ℝ := 40

/-- The starting weight of the dachshund puppy -/
def dachshund_weight : ℝ := 12

/-- The weight gain percentage for both dogs -/
def weight_gain_percentage : ℝ := 0.25

/-- The weight difference between the dogs at the end of the year -/
def weight_difference : ℝ := 35

/-- Theorem stating that the labrador puppy's starting weight satisfies the given conditions -/
theorem labrador_starting_weight :
  L * (1 + weight_gain_percentage) - dachshund_weight * (1 + weight_gain_percentage) = weight_difference := by
  sorry

end NUMINAMATH_CALUDE_labrador_starting_weight_l3630_363082


namespace NUMINAMATH_CALUDE_count_non_negative_l3630_363075

theorem count_non_negative : 
  let numbers := [-(-4), |-1|, -|0|, (-2)^3]
  (numbers.filter (λ x => x ≥ 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_non_negative_l3630_363075


namespace NUMINAMATH_CALUDE_cubic_root_property_l3630_363019

-- Define the cubic polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the roots and their properties
theorem cubic_root_property (x₁ x₂ x₃ : ℝ) 
  (h1 : f x₁ = 0) (h2 : f x₂ = 0) (h3 : f x₃ = 0)
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  x₃^2 - x₂^2 = x₃ - x₁ := by
sorry

end NUMINAMATH_CALUDE_cubic_root_property_l3630_363019


namespace NUMINAMATH_CALUDE_one_fourth_of_5_6_l3630_363031

theorem one_fourth_of_5_6 : (5.6 : ℚ) / 4 = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_5_6_l3630_363031


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3630_363020

/-- 
Given a rectangular prism with width w, length 2w, and height w/2,
where the sum of all edge lengths is 88 cm,
prove that the volume of the prism is 85184/343 cm³.
-/
theorem rectangular_prism_volume 
  (w : ℝ) 
  (h_edge_sum : 4 * w + 8 * w + 2 * w = 88) :
  (2 * w) * w * (w / 2) = 85184 / 343 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3630_363020


namespace NUMINAMATH_CALUDE_savings_ratio_l3630_363061

/-- Proves that the ratio of Megan's daily savings to Leah's daily savings is 2:1 -/
theorem savings_ratio :
  -- Josiah's savings
  let josiah_daily : ℚ := 1/4
  let josiah_days : ℕ := 24
  -- Leah's savings
  let leah_daily : ℚ := 1/2
  let leah_days : ℕ := 20
  -- Megan's savings
  let megan_days : ℕ := 12
  -- Total savings
  let total_savings : ℚ := 28
  -- Calculations
  let josiah_total : ℚ := josiah_daily * josiah_days
  let leah_total : ℚ := leah_daily * leah_days
  let megan_total : ℚ := total_savings - josiah_total - leah_total
  let megan_daily : ℚ := megan_total / megan_days
  -- Theorem
  megan_daily / leah_daily = 2 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l3630_363061


namespace NUMINAMATH_CALUDE_brownie_division_l3630_363091

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of brownies -/
def pan : Dimensions := ⟨15, 25⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 5⟩

/-- Theorem stating that the pan can be divided into exactly 25 pieces -/
theorem brownie_division :
  (area pan) / (area piece) = 25 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l3630_363091


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3630_363025

theorem quadratic_real_roots (k : ℝ) : 
  k > 0 → ∃ x : ℝ, x^2 - x - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3630_363025


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l3630_363039

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 360 is 4 -/
theorem distinct_prime_factors_of_divisor_sum_360 : 
  num_distinct_prime_factors (sum_of_divisors 360) = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l3630_363039


namespace NUMINAMATH_CALUDE_faster_train_speed_l3630_363093

/-- Proves that the speed of the faster train is 180 km/h given the problem conditions --/
theorem faster_train_speed
  (train1_length : ℝ)
  (train2_length : ℝ)
  (initial_distance : ℝ)
  (slower_train_speed : ℝ)
  (time_to_meet : ℝ)
  (h1 : train1_length = 100)
  (h2 : train2_length = 200)
  (h3 : initial_distance = 450)
  (h4 : slower_train_speed = 90)
  (h5 : time_to_meet = 9.99920006399488)
  : ∃ (faster_train_speed : ℝ), faster_train_speed = 180 := by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l3630_363093


namespace NUMINAMATH_CALUDE_max_dominoes_on_problem_board_l3630_363035

/-- Represents a cell on the grid board -/
inductive Cell
| White
| Black

/-- Represents the grid board -/
def Board := List (List Cell)

/-- Represents a domino placement on the board -/
structure Domino where
  row : Nat
  col : Nat
  horizontal : Bool

/-- Checks if a domino placement is valid on the board -/
def isValidDomino (board : Board) (domino : Domino) : Bool :=
  sorry

/-- Counts the number of valid domino placements on the board -/
def countValidDominoes (board : Board) (dominoes : List Domino) : Nat :=
  sorry

/-- The specific board layout from the problem -/
def problemBoard : Board :=
  sorry

/-- Theorem stating that the maximum number of dominoes on the problem board is 16 -/
theorem max_dominoes_on_problem_board :
  ∀ (dominoes : List Domino),
    countValidDominoes problemBoard dominoes ≤ 16 :=
  sorry

end NUMINAMATH_CALUDE_max_dominoes_on_problem_board_l3630_363035


namespace NUMINAMATH_CALUDE_orange_count_l3630_363097

/-- The number of oranges in a bin after some changes -/
def final_oranges (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Proof that given 40 initial oranges, removing 25 and adding 21 results in 36 oranges -/
theorem orange_count : final_oranges 40 25 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l3630_363097


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3630_363078

/-- The quadratic function f(x) = -x^2 + bx - 7 is negative only for x < 2 or x > 6 -/
def quadratic_inequality (b : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + b*x - 7 < 0) ↔ (x < 2 ∨ x > 6)

/-- Given the quadratic inequality condition, prove that b = 8 -/
theorem quadratic_inequality_solution :
  ∃ b : ℝ, quadratic_inequality b ∧ b = 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3630_363078


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3630_363008

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ) 
  (base : ℝ) :
  area = 128 ∧ 
  altitude_base_relation = (λ x => 2 * x) ∧ 
  area = base * (altitude_base_relation base) →
  base = 8 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3630_363008


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l3630_363009

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0 ∧ ∀ φ, φ > 0 ∧ Real.cos (15 * π / 180) = Real.sin (35 * π / 180) + Real.sin φ → θ ≤ φ) →
  Real.cos (15 * π / 180) = Real.sin (35 * π / 180) + Real.sin θ →
  θ = 55 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l3630_363009


namespace NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l3630_363086

/-- The ratio of the volume of a cylinder inscribed in a sphere to the volume of the sphere,
    where the cylinder's height is 4/3 of the sphere's radius. -/
theorem cylinder_sphere_volume_ratio (R : ℝ) (h : R > 0) :
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_height := (4 / 3) * R
  let cylinder_radius := Real.sqrt ((5 / 9) * R^2)
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  cylinder_volume / sphere_volume = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l3630_363086


namespace NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l3630_363003

theorem no_prime_roots_for_specific_quadratic :
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 71 ∧
    (p : ℤ) * q = k ∧
    p ≠ q :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l3630_363003


namespace NUMINAMATH_CALUDE_prime_equation_solution_l3630_363029

theorem prime_equation_solution (p q r : ℕ) (A : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  (2 * p * q * r + 50 * p * q = A) ∧
  (7 * p * q * r + 55 * p * r = A) ∧
  (8 * p * q * r + 12 * q * r = A) →
  A = 1980 := by
sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l3630_363029


namespace NUMINAMATH_CALUDE_hamster_lifespan_difference_l3630_363066

/-- Represents the lifespans of a hamster, bat, and frog. -/
structure AnimalLifespans where
  hamster : ℕ
  bat : ℕ
  frog : ℕ

/-- The conditions of the problem. -/
def problemConditions (a : AnimalLifespans) : Prop :=
  a.bat = 10 ∧
  a.frog = 4 * a.hamster ∧
  a.hamster + a.bat + a.frog = 30

/-- The theorem to be proved. -/
theorem hamster_lifespan_difference (a : AnimalLifespans) 
  (h : problemConditions a) : a.bat - a.hamster = 6 := by
  sorry

#check hamster_lifespan_difference

end NUMINAMATH_CALUDE_hamster_lifespan_difference_l3630_363066
