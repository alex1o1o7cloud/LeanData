import Mathlib

namespace NUMINAMATH_CALUDE_emma_age_theorem_l4030_403051

def emma_age_problem (emma_current_age : ℕ) (age_difference : ℕ) (sister_future_age : ℕ) : Prop :=
  let sister_current_age := emma_current_age + age_difference
  let years_passed := sister_future_age - sister_current_age
  emma_current_age + years_passed = 47

theorem emma_age_theorem :
  emma_age_problem 7 9 56 :=
by
  sorry

end NUMINAMATH_CALUDE_emma_age_theorem_l4030_403051


namespace NUMINAMATH_CALUDE_star_3_5_l4030_403030

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2 + 3*(a+b)

-- State the theorem
theorem star_3_5 : star 3 5 = 88 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l4030_403030


namespace NUMINAMATH_CALUDE_max_fraction_value_101_l4030_403005

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def max_fraction_value (n : ℕ) : ℚ :=
  (factorial n) / 4

theorem max_fraction_value_101 :
  ∀ (f : ℚ), f = max_fraction_value 101 ∨ f < max_fraction_value 101 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_value_101_l4030_403005


namespace NUMINAMATH_CALUDE_box_volume_and_area_l4030_403090

/-- A rectangular box with given dimensions -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular box -/
def volume (box : RectangularBox) : ℝ :=
  box.length * box.width * box.height

/-- Calculate the maximum ground area of a rectangular box -/
def maxGroundArea (box : RectangularBox) : ℝ :=
  box.length * box.width

/-- Theorem about the volume and maximum ground area of a specific rectangular box -/
theorem box_volume_and_area (box : RectangularBox)
    (h1 : box.length = 20)
    (h2 : box.width = 15)
    (h3 : box.height = 5) :
    volume box = 1500 ∧ maxGroundArea box = 300 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_and_area_l4030_403090


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l4030_403010

/-- Triangle Inequality Theorem: A triangle can be formed if the sum of the lengths 
    of any two sides is greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proof that the set of line segments (5, 8, 2) cannot form a triangle -/
theorem cannot_form_triangle : ¬ triangle_inequality 5 8 2 := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l4030_403010


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4030_403089

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + 
    a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4030_403089


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l4030_403018

theorem fraction_equals_zero (x : ℝ) :
  (x + 2) / (3 - x) = 0 ∧ 3 - x ≠ 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l4030_403018


namespace NUMINAMATH_CALUDE_equation_solutions_l4030_403036

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos (2 * x)

-- State the theorem
theorem equation_solutions :
  ∃! (solutions : Finset ℝ), solutions.card = 60 ∧ ∀ x, x ∈ solutions ↔ equation x :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l4030_403036


namespace NUMINAMATH_CALUDE_cube_volume_problem_l4030_403003

theorem cube_volume_problem (a : ℝ) : 
  (a + 3) * (a - 2) * a - a^3 = 6 → a = 3 + Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l4030_403003


namespace NUMINAMATH_CALUDE_max_cans_consumed_correct_verify_100_cans_l4030_403054

def exchange_rate : ℕ := 3

def max_cans_consumed (n : ℕ) : ℕ :=
  n + (n - 1) / 2

theorem max_cans_consumed_correct (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), k * exchange_rate ≤ max_cans_consumed n ∧
             max_cans_consumed n < (k + 1) * exchange_rate :=
by sorry

-- Verify the specific case for 100 cans
theorem verify_100_cans :
  max_cans_consumed 67 ≥ 100 ∧ max_cans_consumed 66 < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_cans_consumed_correct_verify_100_cans_l4030_403054


namespace NUMINAMATH_CALUDE_combined_weight_l4030_403021

/-- The weight of a peach in grams -/
def peach_weight : ℝ := sorry

/-- The weight of a bun in grams -/
def bun_weight : ℝ := sorry

/-- Condition 1: One peach weighs the same as 2 buns plus 40 grams -/
axiom condition1 : peach_weight = 2 * bun_weight + 40

/-- Condition 2: One peach plus 80 grams weighs the same as one bun plus 200 grams -/
axiom condition2 : peach_weight + 80 = bun_weight + 200

/-- Theorem: The combined weight of 1 peach and 1 bun is 280 grams -/
theorem combined_weight : peach_weight + bun_weight = 280 := by sorry

end NUMINAMATH_CALUDE_combined_weight_l4030_403021


namespace NUMINAMATH_CALUDE_vacant_seats_l4030_403062

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (vacant_seats : ℕ) : 
  total_seats = 700 →
  filled_percentage = 75 / 100 →
  vacant_seats = total_seats - (filled_percentage * total_seats).floor →
  vacant_seats = 175 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l4030_403062


namespace NUMINAMATH_CALUDE_chessboard_coloring_l4030_403016

/-- A move on the chessboard changes the color of all squares in a 2x2 area. -/
def ChessboardMove (n : ℕ) := Fin n → Fin n → Bool

/-- The initial chessboard coloring. -/
def InitialChessboard (n : ℕ) : Fin n → Fin n → Bool :=
  λ i j => (i.val + j.val) % 2 = 0

/-- A sequence of moves. -/
def MoveSequence (n : ℕ) := List (ChessboardMove n)

/-- Apply a move to the chessboard. -/
def ApplyMove (board : Fin n → Fin n → Bool) (move : ChessboardMove n) : Fin n → Fin n → Bool :=
  λ i j => board i j ≠ move i j

/-- Apply a sequence of moves to the chessboard. -/
def ApplyMoveSequence (n : ℕ) (board : Fin n → Fin n → Bool) (moves : MoveSequence n) : Fin n → Fin n → Bool :=
  moves.foldl ApplyMove board

/-- Check if all squares on the board have the same color. -/
def AllSameColor (board : Fin n → Fin n → Bool) : Prop :=
  ∀ i j k l, board i j = board k l

/-- Main theorem: There exists a finite sequence of moves that turns all squares
    the same color if and only if n is divisible by 4. -/
theorem chessboard_coloring (n : ℕ) (h : n ≥ 3) :
  (∃ (moves : MoveSequence n), AllSameColor (ApplyMoveSequence n (InitialChessboard n) moves)) ↔
  4 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_chessboard_coloring_l4030_403016


namespace NUMINAMATH_CALUDE_train_crossing_time_l4030_403020

/-- Proves that a train crossing a signal pole takes 18 seconds given specific conditions -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_length = 300 →
  platform_length = 600.0000000000001 →
  platform_crossing_time = 54 →
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4030_403020


namespace NUMINAMATH_CALUDE_tangerines_per_box_l4030_403012

theorem tangerines_per_box
  (total : ℕ)
  (boxes : ℕ)
  (remaining : ℕ)
  (h1 : total = 29)
  (h2 : boxes = 8)
  (h3 : remaining = 5)
  : (total - remaining) / boxes = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tangerines_per_box_l4030_403012


namespace NUMINAMATH_CALUDE_perfect_square_proof_l4030_403077

theorem perfect_square_proof (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  (2 * l - n - k) * (2 * l - n + k) / 2 = (l - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_proof_l4030_403077


namespace NUMINAMATH_CALUDE_function_property_implies_range_l4030_403061

theorem function_property_implies_range (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x : ℝ, f x - f (-x) = 2 * x) →
  (∀ x : ℝ, x > 0 → deriv f x > 1) →
  (∀ a : ℝ, f a - f (1 - a) ≥ 2 * a - 1) →
  (∀ a : ℝ, f a - f (1 - a) ≥ 2 * a - 1 → a ∈ Set.Ici (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_property_implies_range_l4030_403061


namespace NUMINAMATH_CALUDE_school_band_seats_l4030_403004

/-- Calculates the total number of seats needed for a school band given the number of players for each instrument. -/
def total_seats (flute trumpet trombone drummer clarinet french_horn : ℕ) : ℕ :=
  flute + trumpet + trombone + drummer + clarinet + french_horn

/-- Proves that the total number of seats needed for the school band is 65. -/
theorem school_band_seats : ∃ (flute trumpet trombone drummer clarinet french_horn : ℕ),
  flute = 5 ∧
  trumpet = 3 * flute ∧
  trombone = trumpet - 8 ∧
  drummer = trombone + 11 ∧
  clarinet = 2 * flute ∧
  french_horn = trombone + 3 ∧
  total_seats flute trumpet trombone drummer clarinet french_horn = 65 := by
  sorry

#eval total_seats 5 15 7 18 10 10

end NUMINAMATH_CALUDE_school_band_seats_l4030_403004


namespace NUMINAMATH_CALUDE_jennas_eel_length_l4030_403079

theorem jennas_eel_length (j b : ℝ) (h1 : j = b / 3) (h2 : j + b = 64) :
  j = 16 := by
  sorry

end NUMINAMATH_CALUDE_jennas_eel_length_l4030_403079


namespace NUMINAMATH_CALUDE_cake_mix_buyers_l4030_403037

theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) 
  (h1 : total = 100)
  (h2 : muffin = 40)
  (h3 : both = 19)
  (h4 : neither_prob = 29/100) :
  ∃ cake : ℕ, cake = 50 ∧ 
    cake + muffin - both = total - (neither_prob * total).num := by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_l4030_403037


namespace NUMINAMATH_CALUDE_seven_count_l4030_403028

-- Define the range of integers
def IntRange := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Function to count occurrences of a digit in a number
def countDigit (d : ℕ) (n : ℕ) : ℕ := sorry

-- Function to count total occurrences of a digit in a range
def totalOccurrences (d : ℕ) (range : Set ℕ) : ℕ := sorry

-- Theorem statement
theorem seven_count :
  totalOccurrences 7 IntRange = 19 := by sorry

end NUMINAMATH_CALUDE_seven_count_l4030_403028


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l4030_403094

theorem rectangle_area_increase (L W : ℝ) (h1 : L * W = 500) : 
  (1.2 * L) * (1.2 * W) = 720 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l4030_403094


namespace NUMINAMATH_CALUDE_max_x5y_given_constraint_l4030_403026

theorem max_x5y_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * (x + 2 * y) = 9) :
  x^5 * y ≤ 54 ∧ ∃ x0 y0 : ℝ, x0 > 0 ∧ y0 > 0 ∧ x0 * (x0 + 2 * y0) = 9 ∧ x0^5 * y0 = 54 :=
sorry

end NUMINAMATH_CALUDE_max_x5y_given_constraint_l4030_403026


namespace NUMINAMATH_CALUDE_people_needed_for_two_hours_l4030_403019

/-- Represents the rate at which water enters the boat (units per hour) -/
def water_entry_rate : ℝ := 2

/-- Represents the amount of water one person can bail out per hour -/
def bailing_rate : ℝ := 1

/-- Represents the total amount of water to be bailed out -/
def total_water : ℝ := 30

/-- Given the conditions from the problem, proves that 14 people are needed to bail out the water in 2 hours -/
theorem people_needed_for_two_hours : 
  (∀ (p : ℕ), p = 10 → p * bailing_rate * 3 = total_water + water_entry_rate * 3) →
  (∀ (p : ℕ), p = 5 → p * bailing_rate * 8 = total_water + water_entry_rate * 8) →
  ∃ (p : ℕ), p = 14 ∧ p * bailing_rate * 2 = total_water + water_entry_rate * 2 :=
by sorry

end NUMINAMATH_CALUDE_people_needed_for_two_hours_l4030_403019


namespace NUMINAMATH_CALUDE_range_of_m_l4030_403065

-- Define the propositions p and q
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x)

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (sufficient_not_necessary (p) (q m)) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4030_403065


namespace NUMINAMATH_CALUDE_markup_percentage_is_40_percent_l4030_403006

/-- Proves that the markup percentage on the selling price of a desk is 40% given the specified conditions. -/
theorem markup_percentage_is_40_percent
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (markup : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 150)
  (h2 : selling_price = purchase_price + markup)
  (h3 : gross_profit = 100)
  (h4 : gross_profit = selling_price - purchase_price) :
  (markup / selling_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_is_40_percent_l4030_403006


namespace NUMINAMATH_CALUDE_solve_system_l4030_403097

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 8) (eq2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l4030_403097


namespace NUMINAMATH_CALUDE_no_fixed_points_l4030_403008

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

/-- The specific function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ :=
  x^2 + 1

/-- Theorem: f(x) = x^2 + 1 has no fixed points -/
theorem no_fixed_points : ¬∃ x : ℝ, is_fixed_point f x := by
  sorry

end NUMINAMATH_CALUDE_no_fixed_points_l4030_403008


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l4030_403064

theorem jason_pokemon_cards (initial_cards : ℕ) (bought_cards : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 676 → bought_cards = 224 → remaining_cards = initial_cards - bought_cards → 
  remaining_cards = 452 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l4030_403064


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l4030_403085

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x * y - x - y = 3) :
  (∃ (m : ℝ), m = 9 ∧ ∀ z w, z > 0 → w > 0 → z * w - z - w = 3 → x * y ≤ z * w) ∧
  (∃ (n : ℝ), n = 6 ∧ ∀ z w, z > 0 → w > 0 → z * w - z - w = 3 → x + y ≤ z + w) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l4030_403085


namespace NUMINAMATH_CALUDE_fourth_power_difference_l4030_403001

theorem fourth_power_difference (a b : ℝ) : 
  (a - b)^4 = a^4 - 4*a^3*b + 6*a^2*b^2 - 4*a*b^3 + b^4 :=
by
  sorry

-- The given condition is not directly used in the theorem statement,
-- but it can be used in the proof (which is omitted here).
-- If needed, it can be stated as a separate lemma:

lemma fourth_power_sum (a b : ℝ) :
  (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_l4030_403001


namespace NUMINAMATH_CALUDE_equation_solution_l4030_403013

theorem equation_solution (y : ℝ) : 
  (y / 5) / 3 = 5 / (y / 3) → y = 15 ∨ y = -15 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4030_403013


namespace NUMINAMATH_CALUDE_ContrapositiveDual_l4030_403023

-- Define what it means for a number to be even
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k

-- The original proposition
def OriginalProposition : Prop :=
  ∀ a b : Int, IsEven a ∧ IsEven b → IsEven (a + b)

-- The contrapositive we want to prove
def Contrapositive : Prop :=
  ∀ a b : Int, ¬IsEven (a + b) → ¬(IsEven a ∧ IsEven b)

-- The theorem stating that the contrapositive is correct
theorem ContrapositiveDual : OriginalProposition ↔ Contrapositive := by
  sorry

end NUMINAMATH_CALUDE_ContrapositiveDual_l4030_403023


namespace NUMINAMATH_CALUDE_is_15th_term_l4030_403059

/-- Definition of the sequence -/
def a (n : ℕ) : ℕ := 3^(7*(n-1))

/-- Theorem stating that 3^98 is the 15th term of the sequence -/
theorem is_15th_term : a 15 = 3^98 := by
  sorry

end NUMINAMATH_CALUDE_is_15th_term_l4030_403059


namespace NUMINAMATH_CALUDE_fruit_sales_calculation_l4030_403048

/-- Calculate the total money collected from selling fruits with price increases -/
theorem fruit_sales_calculation (lemon_price grape_price orange_price apple_price : ℚ)
  (lemon_count grape_count orange_count apple_count : ℕ)
  (lemon_increase grape_increase orange_increase apple_increase : ℚ) :
  let new_lemon_price := lemon_price * (1 + lemon_increase)
  let new_grape_price := grape_price * (1 + grape_increase)
  let new_orange_price := orange_price * (1 + orange_increase)
  let new_apple_price := apple_price * (1 + apple_increase)
  lemon_count * new_lemon_price + grape_count * new_grape_price +
  orange_count * new_orange_price + apple_count * new_apple_price = 2995 :=
by
  sorry

#check fruit_sales_calculation 8 7 5 4 80 140 60 100 (1/2) (1/4) (1/10) (1/5)

end NUMINAMATH_CALUDE_fruit_sales_calculation_l4030_403048


namespace NUMINAMATH_CALUDE_empty_set_proof_l4030_403050

theorem empty_set_proof : {x : ℝ | x > 9 ∧ x < 3} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l4030_403050


namespace NUMINAMATH_CALUDE_dvd_discount_amount_l4030_403052

/-- The discount on a pack of DVDs -/
def discount (original_price discounted_price : ℝ) : ℝ :=
  original_price - discounted_price

/-- Theorem: The discount on each pack of DVDs is 25 dollars -/
theorem dvd_discount_amount : discount 76 51 = 25 := by
  sorry

end NUMINAMATH_CALUDE_dvd_discount_amount_l4030_403052


namespace NUMINAMATH_CALUDE_admission_charge_problem_l4030_403000

/-- The admission charge problem -/
theorem admission_charge_problem (child_charge : ℚ) (total_charge : ℚ) (num_children : ℕ) 
  (h1 : child_charge = 3/4)
  (h2 : total_charge = 13/4)
  (h3 : num_children = 3) :
  total_charge - (↑num_children * child_charge) = 1 := by
  sorry

end NUMINAMATH_CALUDE_admission_charge_problem_l4030_403000


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l4030_403031

theorem consecutive_sum_product (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (a + b + c = 48) → (a * c = 255) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l4030_403031


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l4030_403039

theorem irrationality_of_sqrt_two_and_rationality_of_others : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = Real.sqrt 2) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = 1 / 3) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = 0) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = -0.7) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l4030_403039


namespace NUMINAMATH_CALUDE_student_pairing_fraction_l4030_403025

theorem student_pairing_fraction (t s : ℕ) (ht : t > 0) (hs : s > 0) :
  (t / 4 : ℚ) = (3 * s / 7 : ℚ) →
  (3 * s / 7 : ℚ) / ((t : ℚ) + (s : ℚ)) = 3 / 19 := by
sorry

end NUMINAMATH_CALUDE_student_pairing_fraction_l4030_403025


namespace NUMINAMATH_CALUDE_right_angled_triangle_l4030_403056

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)

/-- The theorem stating that if certain conditions are met, the triangle is right-angled. -/
theorem right_angled_triangle (t : Triangle) 
  (h1 : (Real.sqrt 3 * t.c) / (t.a * Real.cos t.B) = Real.tan t.A + Real.tan t.B)
  (h2 : t.b - t.c = (Real.sqrt 3 * t.a) / 3) : 
  t.B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l4030_403056


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l4030_403091

/-- Given that x = -1 is a solution of the quadratic equation ax^2 + bx + 23 = 0,
    prove that -a + b + 2000 = 2023 -/
theorem quadratic_solution_implies_sum (a b : ℝ) 
  (h : a * (-1)^2 + b * (-1) + 23 = 0) : 
  -a + b + 2000 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l4030_403091


namespace NUMINAMATH_CALUDE_wire_cutting_l4030_403068

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (longer_piece : ℝ) : 
  total_length = 30 →
  difference = 2 →
  longer_piece = total_length / 2 + difference / 2 →
  longer_piece = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l4030_403068


namespace NUMINAMATH_CALUDE_tan_alpha_plus_beta_l4030_403060

theorem tan_alpha_plus_beta (α β : Real) 
  (h1 : Real.tan α = 1) 
  (h2 : 3 * Real.sin β = Real.sin (2 * α + β)) : 
  Real.tan (α + β) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_beta_l4030_403060


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4030_403092

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^2 + b*x + c
  f_sin_nonneg : ∀ α, f (Real.sin α) ≥ 0
  f_cos_nonpos : ∀ β, f (2 + Real.cos β) ≤ 0

theorem quadratic_function_properties (qf : QuadraticFunction) :
  qf.f 1 = 0 ∧ 
  qf.c ≥ 3 ∧
  (∀ α, qf.f (Real.sin α) ≤ 8 → qf.f = fun x ↦ x^2 - 4*x + 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4030_403092


namespace NUMINAMATH_CALUDE_tax_deduction_percentage_l4030_403011

theorem tax_deduction_percentage (weekly_income : ℝ) (water_bill : ℝ) (tithe_percentage : ℝ) (remaining_amount : ℝ)
  (h1 : weekly_income = 500)
  (h2 : water_bill = 55)
  (h3 : tithe_percentage = 10)
  (h4 : remaining_amount = 345)
  (h5 : remaining_amount = weekly_income - (weekly_income * (tithe_percentage / 100)) - water_bill - (weekly_income * (tax_percentage / 100))) :
  tax_percentage = 10 := by
  sorry


end NUMINAMATH_CALUDE_tax_deduction_percentage_l4030_403011


namespace NUMINAMATH_CALUDE_pizza_toppings_l4030_403080

/-- Represents a pizza with a given number of slices and topping distribution -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  both_toppings : ℕ
  pepperoni_only : ℕ
  mushroom_only : ℕ
  h_total : total_slices = pepperoni_only + mushroom_only + both_toppings
  h_pepperoni : pepperoni_slices = pepperoni_only + both_toppings
  h_mushroom : mushroom_slices = mushroom_only + both_toppings

/-- Theorem stating that a pizza with the given conditions has 2 slices with both toppings -/
theorem pizza_toppings (p : Pizza) 
  (h_total : p.total_slices = 18)
  (h_pep : p.pepperoni_slices = 10)
  (h_mush : p.mushroom_slices = 10) :
  p.both_toppings = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l4030_403080


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_15_l4030_403096

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression (α : Type*) [Add α] [SMul ℕ α] where
  a₁ : α
  d : α

variable {α : Type*} [LinearOrderedField α]

def term (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a₁ + (n - 1) • ap.d

def sum_n_terms (ap : ArithmeticProgression α) (n : ℕ) : α :=
  (n : α) * (ap.a₁ + term ap n) / 2

theorem arithmetic_progression_sum_15
  (ap : ArithmeticProgression α)
  (h_sum : term ap 3 + term ap 9 = 6)
  (h_prod : term ap 3 * term ap 9 = 135 / 16) :
  sum_n_terms ap 15 = 37.5 ∨ sum_n_terms ap 15 = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_15_l4030_403096


namespace NUMINAMATH_CALUDE_min_deliveries_to_breakeven_l4030_403086

def van_cost : ℕ := 8000
def earning_per_delivery : ℕ := 15
def gas_cost_per_delivery : ℕ := 5

theorem min_deliveries_to_breakeven :
  ∃ (d : ℕ), d * (earning_per_delivery - gas_cost_per_delivery) ≥ van_cost ∧
  ∀ (k : ℕ), k * (earning_per_delivery - gas_cost_per_delivery) ≥ van_cost → k ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_deliveries_to_breakeven_l4030_403086


namespace NUMINAMATH_CALUDE_min_value_of_f_l4030_403074

def f (x : ℝ) := 27 * x - x^3

theorem min_value_of_f :
  ∃ (min : ℝ), min = -54 ∧
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4030_403074


namespace NUMINAMATH_CALUDE_inequality_proof_l4030_403014

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_equality : c^2 + a*b = a^2 + b^2) : 
  c^2 + a*b ≤ a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4030_403014


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l4030_403032

theorem binomial_coefficient_congruence (p n : ℕ) (hp : Prime p) :
  (Nat.choose (n * p) n) ≡ n [MOD p^2] := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l4030_403032


namespace NUMINAMATH_CALUDE_great_eighteen_games_l4030_403024

/-- Great Eighteen Soccer League -/
structure SoccerLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in the league -/
def total_games (league : SoccerLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let intra_games := (league.divisions * league.teams_per_division * (league.teams_per_division - 1) * league.intra_division_games) / 2
  let inter_games := (total_teams * (total_teams - league.teams_per_division) * league.inter_division_games) / 2
  intra_games + inter_games

/-- Theorem: The Great Eighteen Soccer League has 351 scheduled games -/
theorem great_eighteen_games :
  let league := SoccerLeague.mk 3 6 3 2
  total_games league = 351 := by
  sorry

end NUMINAMATH_CALUDE_great_eighteen_games_l4030_403024


namespace NUMINAMATH_CALUDE_movie_length_after_cuts_l4030_403073

def original_length : ℝ := 97
def cut_scene1 : ℝ := 4.5
def cut_scene2 : ℝ := 2.75
def cut_scene3 : ℝ := 6.25

theorem movie_length_after_cuts :
  original_length - (cut_scene1 + cut_scene2 + cut_scene3) = 83.5 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cuts_l4030_403073


namespace NUMINAMATH_CALUDE_two_digit_R_equals_R_plus_two_l4030_403029

def R (n : ℕ) : ℕ := 
  (n % 2) + (n % 3) + (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8) + (n % 9)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_R_equals_R_plus_two :
  ∃! (s : Finset ℕ), s.card = 2 ∧ 
    (∀ n ∈ s, is_two_digit n ∧ R n = R (n + 2)) ∧
    (∀ n, is_two_digit n → R n = R (n + 2) → n ∈ s) :=
sorry

end NUMINAMATH_CALUDE_two_digit_R_equals_R_plus_two_l4030_403029


namespace NUMINAMATH_CALUDE_cupboard_capacity_l4030_403058

/-- Calculates the total number of tea cups that can be stored in multiple cupboards. -/
def total_tea_cups (num_cupboards : ℕ) (compartments_per_cupboard : ℕ) (cups_per_compartment : ℕ) : ℕ :=
  num_cupboards * compartments_per_cupboard * cups_per_compartment

/-- Theorem stating that 8 cupboards with 5 compartments each, holding 85 cups per compartment,
    can store a total of 3400 tea cups. -/
theorem cupboard_capacity :
  total_tea_cups 8 5 85 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_cupboard_capacity_l4030_403058


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l4030_403044

theorem unique_positive_integer_solution :
  ∃! (x : ℕ+), (4 * (x - 1) : ℝ) < 3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l4030_403044


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l4030_403047

def satisfies_inequality (x : ℤ) : Prop :=
  |7 * x - 3| - 2 * x < 5 - 3 * x

theorem greatest_integer_satisfying_inequality :
  satisfies_inequality 0 ∧
  ∀ y : ℤ, y > 0 → ¬(satisfies_inequality y) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l4030_403047


namespace NUMINAMATH_CALUDE_binomial_inequality_l4030_403027

theorem binomial_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : x ≠ 0) (h3 : n ≥ 2) :
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l4030_403027


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l4030_403071

def total_missiles : ℕ := 50
def selected_missiles : ℕ := 5

def systematic_sampling (total : ℕ) (selected : ℕ) : ℕ := total / selected

def generate_sequence (start : ℕ) (interval : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (fun i => start + i * interval)

theorem correct_systematic_sampling :
  let interval := systematic_sampling total_missiles selected_missiles
  let sequence := generate_sequence 3 interval selected_missiles
  interval = 10 ∧ sequence = [3, 13, 23, 33, 43] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sampling_l4030_403071


namespace NUMINAMATH_CALUDE_black_queen_thought_l4030_403035

-- Define the possible states for each character
inductive State
  | Asleep
  | Awake

-- Define the characters
structure Character where
  name : String
  state : State
  thought : State

-- Define the perverse judgment property
def perverseJudgment (c : Character) : Prop :=
  (c.state = State.Asleep ∧ c.thought = State.Awake) ∨
  (c.state = State.Awake ∧ c.thought = State.Asleep)

-- Define the rational judgment property
def rationalJudgment (c : Character) : Prop :=
  (c.state = State.Asleep ∧ c.thought = State.Asleep) ∨
  (c.state = State.Awake ∧ c.thought = State.Awake)

-- Theorem statement
theorem black_queen_thought (blackKing blackQueen : Character) :
  blackKing.name = "Black King" →
  blackQueen.name = "Black Queen" →
  blackKing.thought = State.Asleep →
  (perverseJudgment blackKing ∨ rationalJudgment blackKing) →
  (perverseJudgment blackQueen ∨ rationalJudgment blackQueen) →
  blackQueen.thought = State.Asleep :=
by
  sorry

end NUMINAMATH_CALUDE_black_queen_thought_l4030_403035


namespace NUMINAMATH_CALUDE_gcd_47_power_plus_one_l4030_403084

theorem gcd_47_power_plus_one : Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_47_power_plus_one_l4030_403084


namespace NUMINAMATH_CALUDE_certain_number_proof_l4030_403009

theorem certain_number_proof : ∃ n : ℕ, n - 999 = 9001 ∧ n = 10000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4030_403009


namespace NUMINAMATH_CALUDE_circle_symmetry_minimum_l4030_403088

theorem circle_symmetry_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 1 = 0 ↔ x^2 + y^2 + 4*x - 2*y + 1 = 0 ∧ a*x - b*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (a' + 2*b') / (a' * b') ≥ (a + 2*b) / (a * b)) →
  (a + 2*b) / (a * b) = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_minimum_l4030_403088


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l4030_403069

theorem unique_congruence_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -1872 [ZMOD 9] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l4030_403069


namespace NUMINAMATH_CALUDE_a_closed_form_l4030_403041

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 5 * a (n + 1) - 6 * a n + 4^(n + 1)

theorem a_closed_form (n : ℕ) :
  a n = 2^(n + 1) - 3^(n + 1) + 2 * 4^n :=
by sorry

end NUMINAMATH_CALUDE_a_closed_form_l4030_403041


namespace NUMINAMATH_CALUDE_intersection_sum_l4030_403055

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 5*x < 0}
def N (p : ℝ) : Set ℝ := {x | p < x ∧ x < 6}

-- Define the intersection of M and N
def M_intersect_N (p q : ℝ) : Set ℝ := {x | 2 < x ∧ x < q}

-- Theorem statement
theorem intersection_sum (p q : ℝ) : 
  M ∩ N p = M_intersect_N p q → p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l4030_403055


namespace NUMINAMATH_CALUDE_quadratic_root_implies_d_value_l4030_403063

theorem quadratic_root_implies_d_value 
  (d : ℝ) 
  (h : ∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = (-14 + Real.sqrt 20) / 4 ∨ x = (-14 - Real.sqrt 20) / 4) :
  d = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_d_value_l4030_403063


namespace NUMINAMATH_CALUDE_max_y_over_x_on_circle_l4030_403053

theorem max_y_over_x_on_circle (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : 
  ∃ (max : ℝ), (∀ (a b : ℝ), (a - 2)^2 + b^2 = 3 → b / a ≤ max) ∧ max = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_y_over_x_on_circle_l4030_403053


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_l4030_403045

theorem no_real_solutions_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) ↔ k < -9/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_l4030_403045


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4030_403046

theorem solve_exponential_equation :
  ∃ y : ℝ, (5 : ℝ)^9 = 25^y ∧ y = (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4030_403046


namespace NUMINAMATH_CALUDE_larry_basketball_shots_l4030_403095

theorem larry_basketball_shots 
  (initial_shots : ℕ) 
  (initial_success_rate : ℚ) 
  (additional_shots : ℕ) 
  (new_success_rate : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_success_rate = 3/5)
  (h3 : additional_shots = 10)
  (h4 : new_success_rate = 13/20) :
  (new_success_rate * (initial_shots + additional_shots) - initial_success_rate * initial_shots : ℚ) = 8 := by
sorry

end NUMINAMATH_CALUDE_larry_basketball_shots_l4030_403095


namespace NUMINAMATH_CALUDE_max_value_of_e_l4030_403072

theorem max_value_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (product_condition : a*b + a*c + a*d + a*e + b*c + b*d + b*e + c*d + c*e + d*e = 20) :
  e ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_e_l4030_403072


namespace NUMINAMATH_CALUDE_widget_count_l4030_403093

theorem widget_count : ∃ (a b c d e f : ℕ),
  3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255 ∧
  3^a * 11^b * 5^c * 7^d * 13^e * 17^f = 351125648000 ∧
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_widget_count_l4030_403093


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l4030_403015

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x - 20
def line2 (x y : ℝ) : Prop := 3 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate : 
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l4030_403015


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l4030_403038

theorem paper_folding_thickness (initial_thickness : ℝ) (num_folds : ℕ) (floor_height : ℝ) :
  initial_thickness = 0.1 →
  num_folds = 20 →
  floor_height = 3 →
  ⌊(2^num_folds * initial_thickness / 1000) / floor_height⌋ = 35 :=
by sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l4030_403038


namespace NUMINAMATH_CALUDE_arrangement_count_l4030_403007

def number_of_arrangements (black red blue : ℕ) : ℕ :=
  Nat.factorial (black + red + blue) / (Nat.factorial black * Nat.factorial red * Nat.factorial blue)

theorem arrangement_count :
  number_of_arrangements 2 3 4 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l4030_403007


namespace NUMINAMATH_CALUDE_circle_radius_from_perimeter_l4030_403081

theorem circle_radius_from_perimeter (perimeter : ℝ) (radius : ℝ) :
  perimeter = 8 ∧ perimeter = 2 * Real.pi * radius → radius = 4 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_perimeter_l4030_403081


namespace NUMINAMATH_CALUDE_second_investment_interest_rate_l4030_403066

theorem second_investment_interest_rate
  (total_income : ℝ)
  (investment1_principal : ℝ)
  (investment1_rate : ℝ)
  (investment2_principal : ℝ)
  (total_investment : ℝ)
  (h1 : total_income = 575)
  (h2 : investment1_principal = 3000)
  (h3 : investment1_rate = 0.085)
  (h4 : investment2_principal = 5000)
  (h5 : total_investment = 8000)
  (h6 : total_investment = investment1_principal + investment2_principal) :
  let investment1_income := investment1_principal * investment1_rate
  let investment2_income := total_income - investment1_income
  let investment2_rate := investment2_income / investment2_principal
  investment2_rate = 0.064 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_interest_rate_l4030_403066


namespace NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l4030_403070

theorem sqrt_pi_squared_minus_6pi_plus_9 : 
  Real.sqrt (π^2 - 6*π + 9) = π - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l4030_403070


namespace NUMINAMATH_CALUDE_nancys_payment_is_384_l4030_403034

/-- Nancy's annual payment for her daughter's car insurance -/
def nancys_annual_payment (monthly_cost : ℝ) (nancys_percentage : ℝ) : ℝ :=
  monthly_cost * nancys_percentage * 12

/-- Theorem: Nancy's annual payment for her daughter's car insurance is $384 -/
theorem nancys_payment_is_384 :
  nancys_annual_payment 80 0.4 = 384 := by
  sorry

end NUMINAMATH_CALUDE_nancys_payment_is_384_l4030_403034


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l4030_403042

/-- Given an ellipse with equation x²/25 + y²/9 = 1, if a point A on the ellipse has a symmetric 
point B with respect to the origin, and F₂ is its right focus, and AF₂ is perpendicular to BF₂, 
then the area of triangle AF₂B is 9. -/
theorem ellipse_triangle_area (x y : ℝ) (F₂ : ℝ × ℝ) (A B : ℝ × ℝ) : 
  x^2/25 + y^2/9 = 1 →  -- Ellipse equation
  A.1^2/25 + A.2^2/9 = 1 →  -- A is on the ellipse
  B = (-A.1, -A.2) →  -- B is symmetric to A with respect to origin
  F₂.1 = 4 ∧ F₂.2 = 0 →  -- F₂ is the right focus
  (A.1 - F₂.1) * (B.1 - F₂.1) + (A.2 - F₂.2) * (B.2 - F₂.2) = 0 →  -- AF₂ ⊥ BF₂
  (abs (A.1 - F₂.1) * abs (A.2 - B.2)) / 2 = 9 :=  -- Area of triangle AF₂B
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l4030_403042


namespace NUMINAMATH_CALUDE_julia_internet_speed_l4030_403022

-- Define the given conditions
def songs_downloaded : ℕ := 7200
def download_time_minutes : ℕ := 30
def song_size_mb : ℕ := 5

-- Define the internet speed calculation function
def calculate_internet_speed (songs : ℕ) (time_minutes : ℕ) (size_mb : ℕ) : ℚ :=
  (songs * size_mb : ℚ) / (time_minutes * 60 : ℚ)

-- Theorem statement
theorem julia_internet_speed :
  calculate_internet_speed songs_downloaded download_time_minutes song_size_mb = 20 := by
  sorry


end NUMINAMATH_CALUDE_julia_internet_speed_l4030_403022


namespace NUMINAMATH_CALUDE_rush_delivery_percentage_l4030_403017

theorem rush_delivery_percentage (original_cost : ℝ) (rush_cost_per_type : ℝ) (num_types : ℕ) :
  original_cost = 40 →
  rush_cost_per_type = 13 →
  num_types = 4 →
  (rush_cost_per_type * num_types - original_cost) / original_cost * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rush_delivery_percentage_l4030_403017


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l4030_403043

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l4030_403043


namespace NUMINAMATH_CALUDE_andrew_payment_l4030_403057

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: Andrew paid 1055 to the shopkeeper -/
theorem andrew_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l4030_403057


namespace NUMINAMATH_CALUDE_jenna_earnings_l4030_403083

def calculate_earnings (distance : ℕ) : ℚ :=
  let first_100 := min distance 100
  let next_200 := min (distance - 100) 200
  let beyond_300 := max (distance - 300) 0
  0.4 * first_100 + 0.5 * next_200 + 0.6 * beyond_300

def round_trip_distance : ℕ := 800

def base_earnings : ℚ := 2 * calculate_earnings (round_trip_distance / 2)

def bonus : ℚ := 100 * (round_trip_distance / 500)

def weather_reduction : ℚ := 0.1

def rest_stop_reduction : ℚ := 0.05

def performance_incentive : ℚ := 0.05

def maintenance_cost : ℚ := 50 * (round_trip_distance / 500)

def fuel_cost_rate : ℚ := 0.15

theorem jenna_earnings :
  let reduced_bonus := bonus * (1 - weather_reduction)
  let earnings_with_bonus := base_earnings + reduced_bonus
  let earnings_with_incentive := earnings_with_bonus * (1 + performance_incentive)
  let earnings_after_rest_stop := earnings_with_incentive * (1 - rest_stop_reduction)
  let fuel_cost := earnings_after_rest_stop * fuel_cost_rate
  let net_earnings := earnings_after_rest_stop - maintenance_cost - fuel_cost
  net_earnings = 380 := by sorry

end NUMINAMATH_CALUDE_jenna_earnings_l4030_403083


namespace NUMINAMATH_CALUDE_P_value_at_seven_l4030_403033

-- Define the polynomial P(x)
def P (a b c d e f : ℝ) (x : ℂ) : ℂ :=
  (3 * x^4 - 39 * x^3 + a * x^2 + b * x + c) *
  (4 * x^4 - 64 * x^3 + d * x^2 + e * x + f)

-- State the theorem
theorem P_value_at_seven 
  (a b c d e f : ℝ) 
  (h : Set.range (fun (x : ℂ) => P a b c d e f x) = {1, 2, 3, 4, 6}) : 
  P a b c d e f 7 = 69120 := by
sorry

end NUMINAMATH_CALUDE_P_value_at_seven_l4030_403033


namespace NUMINAMATH_CALUDE_max_x_plus_z_l4030_403098

theorem max_x_plus_z (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x*t + y*z = 6) :
  x + z ≤ Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_x_plus_z_l4030_403098


namespace NUMINAMATH_CALUDE_sum_of_seven_odd_integers_remainder_l4030_403049

def consecutive_odd_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

theorem sum_of_seven_odd_integers_remainder (start : ℕ) (h : start = 12095) :
  (consecutive_odd_integers start 7).sum % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_odd_integers_remainder_l4030_403049


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_l4030_403078

theorem envelope_addressing_equation : 
  ∀ (x : ℝ), 
  (∃ (machine1_time machine2_time combined_time : ℝ),
    machine1_time = 12 ∧ 
    combined_time = 4 ∧
    machine2_time = x ∧
    (1 / machine1_time + 1 / machine2_time = 1 / combined_time)) ↔
  (1 / 12 + 1 / x = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_l4030_403078


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l4030_403082

def n : ℕ := 240345

theorem sum_of_prime_factors (p : ℕ → Prop) 
  (h_prime : ∀ x, p x ↔ Nat.Prime x) : 
  ∃ (a b c : ℕ), 
    p a ∧ p b ∧ p c ∧ 
    n = a * b * c ∧ 
    a + b + c = 16011 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l4030_403082


namespace NUMINAMATH_CALUDE_product_coefficients_sum_l4030_403067

theorem product_coefficients_sum (m n : ℚ) : 
  (∀ k : ℚ, (5*k^2 - 4*k + m) * (2*k^2 + n*k - 5) = 10*k^4 - 28*k^3 + 23*k^2 - 18*k + 15) →
  m + n = 35/3 := by
sorry

end NUMINAMATH_CALUDE_product_coefficients_sum_l4030_403067


namespace NUMINAMATH_CALUDE_root_transformation_l4030_403076

/-- Given that a, b, and c are the roots of x^3 - 4x + 6 = 0,
    prove that a - 3, b - 3, and c - 3 are the roots of x^3 + 9x^2 + 23x + 21 = 0 -/
theorem root_transformation (a b c : ℂ) : 
  (a^3 - 4*a + 6 = 0) ∧ (b^3 - 4*b + 6 = 0) ∧ (c^3 - 4*c + 6 = 0) →
  ((a - 3)^3 + 9*(a - 3)^2 + 23*(a - 3) + 21 = 0) ∧
  ((b - 3)^3 + 9*(b - 3)^2 + 23*(b - 3) + 21 = 0) ∧
  ((c - 3)^3 + 9*(c - 3)^2 + 23*(c - 3) + 21 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l4030_403076


namespace NUMINAMATH_CALUDE_lotto_winnings_theorem_l4030_403087

/-- The amount of money won by each boy in the "Russian Lotto" draw. -/
structure LottoWinnings where
  kolya : ℕ
  misha : ℕ
  vitya : ℕ

/-- The conditions of the "Russian Lotto" draw. -/
def lotto_conditions (w : LottoWinnings) : Prop :=
  w.misha = w.kolya + 943 ∧
  w.vitya = w.misha + 127 ∧
  w.misha + w.kolya = w.vitya + 479

/-- The theorem stating the correct winnings for each boy. -/
theorem lotto_winnings_theorem :
  ∃ (w : LottoWinnings), lotto_conditions w ∧ w.kolya = 606 ∧ w.misha = 1549 ∧ w.vitya = 1676 :=
by
  sorry

end NUMINAMATH_CALUDE_lotto_winnings_theorem_l4030_403087


namespace NUMINAMATH_CALUDE_solve_for_n_l4030_403002

def first_seven_multiples_of_four : List ℕ := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem solve_for_n (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 8 := by sorry

end NUMINAMATH_CALUDE_solve_for_n_l4030_403002


namespace NUMINAMATH_CALUDE_jingyuetan_park_probability_l4030_403075

theorem jingyuetan_park_probability (total_envelopes : ℕ) (jingyuetan_tickets : ℕ) 
  (changying_tickets : ℕ) (h1 : total_envelopes = 5) (h2 : jingyuetan_tickets = 3) 
  (h3 : changying_tickets = 2) (h4 : total_envelopes = jingyuetan_tickets + changying_tickets) :
  (jingyuetan_tickets : ℚ) / total_envelopes = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_jingyuetan_park_probability_l4030_403075


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l4030_403099

/-- The point of intersection for two lines defined by linear equations -/
def intersection_point : ℚ × ℚ := (24/25, 34/25)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = 7x - 4 -/
def line2 (x y : ℚ) : Prop := 2 * y = 7 * x - 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l4030_403099


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4030_403040

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) ∧
  C = π/6 ∧
  a = 1 ∧
  b = Real.sqrt 3 →
  B = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4030_403040
