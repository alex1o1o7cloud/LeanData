import Mathlib

namespace NUMINAMATH_CALUDE_part1_part2_l3523_352361

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Define the solution set for part 1
def solution_set_part1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ -1/2}

-- Define the solution set for part 2
def solution_set_part2 (a : ℝ) : Set ℝ :=
  if a = -1 then {-1}
  else if a < -1 then {x | -1 ≤ x ∧ x ≤ 1/a}
  else {x | 1/a ≤ x ∧ x ≤ -1}

-- Theorem for part 1
theorem part1 :
  ∀ x ∈ solution_set_part1, quadratic_inequality (-2) x ∧
  ∀ a ≠ -2, ∃ x ∈ solution_set_part1, ¬(quadratic_inequality a x) :=
sorry

-- Theorem for part 2
theorem part2 (a : ℝ) (h : a < 0) :
  ∀ x, quadratic_inequality a x ↔ x ∈ solution_set_part2 a :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3523_352361


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_l3523_352366

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implication 
  (a b c : Line) (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : perpendicular a α) 
  (h4 : perpendicular b β) 
  (h5 : parallel_lines a b) : 
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_l3523_352366


namespace NUMINAMATH_CALUDE_real_part_of_complex_square_l3523_352344

theorem real_part_of_complex_square : Complex.re ((5 : ℂ) + 2 * Complex.I) ^ 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_square_l3523_352344


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3523_352304

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  left_focus_x : ℝ
  left_focus_on_directrix : left_focus_x = -5  -- directrix of y^2 = 20x is x = -5
  asymptote_slope : b / a = 4 / 3

/-- The standard equation of the hyperbola is x^2/9 - y^2/16 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) : 
  h.a = 3 ∧ h.b = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3523_352304


namespace NUMINAMATH_CALUDE_whitney_money_left_l3523_352393

/-- The amount of money Whitney has left over after her purchase at the school book fair -/
def money_left_over : ℕ :=
  let initial_money : ℕ := 2 * 20
  let poster_cost : ℕ := 5
  let notebook_cost : ℕ := 4
  let bookmark_cost : ℕ := 2
  let num_posters : ℕ := 2
  let num_notebooks : ℕ := 3
  let num_bookmarks : ℕ := 2
  let total_cost : ℕ := poster_cost * num_posters + notebook_cost * num_notebooks + bookmark_cost * num_bookmarks
  initial_money - total_cost

theorem whitney_money_left : money_left_over = 14 := by
  sorry

end NUMINAMATH_CALUDE_whitney_money_left_l3523_352393


namespace NUMINAMATH_CALUDE_two_digit_addition_proof_l3523_352309

theorem two_digit_addition_proof (A B C : ℕ) : 
  A ≠ B → B ≠ C → A ≠ C →
  A ≤ 9 → B ≤ 9 → C ≤ 9 →
  A ≠ 0 → C ≠ 0 →
  (10 * A + B) + (10 * C + B) = 100 * C + C * 10 + 6 →
  B = 8 := by
sorry

end NUMINAMATH_CALUDE_two_digit_addition_proof_l3523_352309


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3523_352303

theorem sufficient_not_necessary_condition :
  (∀ x y : ℝ, x > 3 ∧ y > 3 → x + y > 6) ∧
  (∃ x y : ℝ, x + y > 6 ∧ ¬(x > 3 ∧ y > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3523_352303


namespace NUMINAMATH_CALUDE_fraction_difference_l3523_352331

theorem fraction_difference (A B C : ℚ) (k m : ℕ) : 
  A = 3 * k / (2 * m) →
  B = 2 * k / (3 * m) →
  C = k / (4 * m) →
  A + B + C = 29 / 60 →
  A - B - C = 7 / 60 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l3523_352331


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3523_352375

def set_A : Set ℝ := {x | -4 < x ∧ x < 2}
def set_B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3523_352375


namespace NUMINAMATH_CALUDE_valid_license_plates_count_l3523_352325

/-- Represents a license plate with 4 characters -/
structure LicensePlate :=
  (first : Char) (second : Char) (third : Nat) (fourth : Nat)

/-- Checks if a character is a letter (A-Z) -/
def isLetter (c : Char) : Bool :=
  'A' ≤ c ∧ c ≤ 'Z'

/-- Checks if a number is a single digit (0-9) -/
def isDigit (n : Nat) : Bool :=
  n < 10

/-- Checks if a license plate satisfies all conditions -/
def isValidLicensePlate (plate : LicensePlate) : Prop :=
  isLetter plate.first ∧
  isLetter plate.second ∧
  isDigit plate.third ∧
  isDigit plate.fourth ∧
  plate.third = plate.fourth ∧
  (plate.first.toNat = plate.third ∨ plate.second.toNat = plate.third)

/-- The number of valid license plates -/
def numValidLicensePlates : Nat :=
  (26 * 26) * 10

theorem valid_license_plates_count :
  numValidLicensePlates = 6760 :=
by sorry

end NUMINAMATH_CALUDE_valid_license_plates_count_l3523_352325


namespace NUMINAMATH_CALUDE_count_divisors_not_mult_14_l3523_352395

def n : ℕ := sorry

-- n is the smallest positive integer satisfying the conditions
axiom n_minimal : ∀ m : ℕ, m > 0 → m < n →
  ¬(∃ k : ℕ, m / 2 = k ^ 2) ∨
  ¬(∃ k : ℕ, m / 3 = k ^ 3) ∨
  ¬(∃ k : ℕ, m / 5 = k ^ 5) ∨
  ¬(∃ k : ℕ, m / 7 = k ^ 7)

-- n satisfies the conditions
axiom n_div_2_square : ∃ k : ℕ, n / 2 = k ^ 2
axiom n_div_3_cube : ∃ k : ℕ, n / 3 = k ^ 3
axiom n_div_5_fifth : ∃ k : ℕ, n / 5 = k ^ 5
axiom n_div_7_seventh : ∃ k : ℕ, n / 7 = k ^ 7

def divisors_not_mult_14 (n : ℕ) : ℕ := sorry

theorem count_divisors_not_mult_14 : divisors_not_mult_14 n = 19005 := by sorry

end NUMINAMATH_CALUDE_count_divisors_not_mult_14_l3523_352395


namespace NUMINAMATH_CALUDE_river_rowing_time_l3523_352345

/-- Conversion factor from yards to meters -/
def yards_to_meters : ℝ := 0.9144

/-- Initial width of the river in yards -/
def initial_width_yards : ℝ := 50

/-- Final width of the river in yards -/
def final_width_yards : ℝ := 80

/-- Rate of river width increase in yards per 10 meters -/
def width_increase_rate : ℝ := 2

/-- Rowing speed in meters per second -/
def rowing_speed : ℝ := 5

/-- Time taken to row from initial width to final width -/
def time_taken : ℝ := 30

theorem river_rowing_time :
  let initial_width_meters := initial_width_yards * yards_to_meters
  let final_width_meters := final_width_yards * yards_to_meters
  let width_difference := final_width_meters - initial_width_meters
  let width_increase_per_10m := width_increase_rate * yards_to_meters
  let distance := (width_difference / width_increase_per_10m) * 10
  distance / rowing_speed = time_taken :=
by sorry

end NUMINAMATH_CALUDE_river_rowing_time_l3523_352345


namespace NUMINAMATH_CALUDE_center_coordinates_sum_l3523_352382

/-- Given two points as the endpoints of a diameter of a circle, 
    prove that the sum of the coordinates of the center is 0. -/
theorem center_coordinates_sum (x₁ y₁ x₂ y₂ : ℝ) 
  (h : x₁ = 9 ∧ y₁ = -5 ∧ x₂ = -3 ∧ y₂ = -1) : 
  ((x₁ + x₂) / 2) + ((y₁ + y₂) / 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_center_coordinates_sum_l3523_352382


namespace NUMINAMATH_CALUDE_total_visitors_three_days_l3523_352374

def visitors_rachels_day : ℕ := 92
def visitors_previous_day : ℕ := 419
def visitors_day_before_previous : ℕ := 103

theorem total_visitors_three_days :
  visitors_rachels_day + visitors_previous_day + visitors_day_before_previous = 614 := by
  sorry

end NUMINAMATH_CALUDE_total_visitors_three_days_l3523_352374


namespace NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l3523_352322

open Set
open Function

-- Define the concept of a monotonic function
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x

-- Define the concept of having a maximum and minimum value on an interval
def HasMaxMin (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ max min : ℝ, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ max) ∧
                 (∀ x, a ≤ x ∧ x ≤ b → min ≤ f x)

theorem monotonic_sufficient_not_necessary (a b : ℝ) (h : a ≤ b) :
  (∀ f : ℝ → ℝ, Monotonic f a b → HasMaxMin f a b) ∧
  (∃ f : ℝ → ℝ, HasMaxMin f a b ∧ ¬Monotonic f a b) :=
sorry

end NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l3523_352322


namespace NUMINAMATH_CALUDE_max_at_2_implies_c_6_l3523_352349

/-- The function f(x) = x(x-c)² has a maximum value at x = 2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  ∀ x, f x ≤ f 2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x = 2, then c = 6 -/
theorem max_at_2_implies_c_6 : 
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_at_2_implies_c_6_l3523_352349


namespace NUMINAMATH_CALUDE_shaded_area_is_63_l3523_352319

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of two intersecting rectangles -/
structure IntersectingRectangles where
  rect1 : Rectangle
  rect2 : Rectangle
  overlap : Rectangle

/-- Calculates the shaded area formed by two intersecting rectangles -/
def IntersectingRectangles.shadedArea (ir : IntersectingRectangles) : ℝ :=
  ir.rect1.area + ir.rect2.area - ir.overlap.area

/-- The main theorem stating that the shaded area is 63 square units -/
theorem shaded_area_is_63 (ir : IntersectingRectangles)
  (h1 : ir.rect1 = { width := 4, height := 12 })
  (h2 : ir.rect2 = { width := 5, height := 7 })
  (h3 : ir.overlap = { width := 4, height := 5 }) :
  ir.shadedArea = 63 := by
  sorry

#check shaded_area_is_63

end NUMINAMATH_CALUDE_shaded_area_is_63_l3523_352319


namespace NUMINAMATH_CALUDE_gregs_gold_is_20_l3523_352323

/-- Represents the amount of gold Greg has -/
def gregs_gold : ℝ := sorry

/-- Represents the amount of gold Katie has -/
def katies_gold : ℝ := sorry

/-- The total amount of gold is 100 -/
axiom total_gold : gregs_gold + katies_gold = 100

/-- Greg has four times less gold than Katie -/
axiom gold_ratio : gregs_gold = katies_gold / 4

/-- Theorem stating that Greg's gold amount is 20 -/
theorem gregs_gold_is_20 : gregs_gold = 20 := by sorry

end NUMINAMATH_CALUDE_gregs_gold_is_20_l3523_352323


namespace NUMINAMATH_CALUDE_sine_graph_shift_l3523_352300

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (1/2 * (x - 4*π/5) + π/5) = 3 * Real.sin (1/2 * x - π/5) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l3523_352300


namespace NUMINAMATH_CALUDE_division_remainder_sum_l3523_352364

theorem division_remainder_sum (n : ℕ) : 
  (n / 7 = 13 ∧ n % 7 = 1) → ((n + 9) / 8 + (n + 9) % 8 = 17) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_sum_l3523_352364


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l3523_352362

theorem prime_square_mod_180 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  p^2 % 180 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l3523_352362


namespace NUMINAMATH_CALUDE_seating_arrangement_l3523_352385

theorem seating_arrangement (n m : ℕ) (h1 : n = 6) (h2 : m = 4) : 
  (n.factorial / (n - m).factorial) = 360 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3523_352385


namespace NUMINAMATH_CALUDE_fraction_simplification_l3523_352347

theorem fraction_simplification : 
  (3/7 + 2/3) / (5/11 + 3/8) = 119/90 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3523_352347


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3523_352391

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2012^2 + 2^2012 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3523_352391


namespace NUMINAMATH_CALUDE_alexander_buckwheat_investment_l3523_352317

theorem alexander_buckwheat_investment (initial_price : ℝ) (final_price : ℝ)
  (one_year_rate_2020 : ℝ) (two_year_rate : ℝ) (one_year_rate_2021 : ℝ)
  (h1 : initial_price = 70)
  (h2 : final_price = 100)
  (h3 : one_year_rate_2020 = 0.1)
  (h4 : two_year_rate = 0.08)
  (h5 : one_year_rate_2021 = 0.05) :
  (initial_price * (1 + one_year_rate_2020) * (1 + one_year_rate_2021) < final_price) ∧
  (initial_price * (1 + two_year_rate)^2 < final_price) :=
by sorry

end NUMINAMATH_CALUDE_alexander_buckwheat_investment_l3523_352317


namespace NUMINAMATH_CALUDE_trick_decks_total_spent_l3523_352394

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (price_per_deck : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  price_per_deck * (victor_decks + friend_decks)

/-- Theorem stating the total amount spent by Victor and his friend -/
theorem trick_decks_total_spent :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spent_l3523_352394


namespace NUMINAMATH_CALUDE_macey_savings_weeks_l3523_352376

/-- Calculates the number of weeks needed to save for a shirt -/
def weeks_to_save (total_cost savings_rate amount_saved : ℚ) : ℚ :=
  (total_cost - amount_saved) / savings_rate

/-- Proves that Macey needs 3 weeks to save for the shirt -/
theorem macey_savings_weeks : 
  let total_cost : ℚ := 3
  let amount_saved : ℚ := 3/2
  let savings_rate : ℚ := 1/2
  weeks_to_save total_cost savings_rate amount_saved = 3 := by
  sorry

end NUMINAMATH_CALUDE_macey_savings_weeks_l3523_352376


namespace NUMINAMATH_CALUDE_bert_ernie_stamp_ratio_l3523_352328

/-- The number of stamps Peggy has -/
def peggy_stamps : ℕ := 75

/-- The number of stamps Peggy needs to add to match Bert's collection -/
def stamps_to_add : ℕ := 825

/-- The number of stamps Ernie has -/
def ernie_stamps : ℕ := 3 * peggy_stamps

/-- The number of stamps Bert has -/
def bert_stamps : ℕ := peggy_stamps + stamps_to_add

/-- The ratio of Bert's stamps to Ernie's stamps -/
def stamp_ratio : ℚ := bert_stamps / ernie_stamps

theorem bert_ernie_stamp_ratio :
  stamp_ratio = 4 / 1 := by sorry

end NUMINAMATH_CALUDE_bert_ernie_stamp_ratio_l3523_352328


namespace NUMINAMATH_CALUDE_area_of_region_l3523_352329

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ 2 ∧ abs p.2 ≤ 2 ∧ abs (abs p.1 - abs p.2) ≤ 1}

-- State the theorem
theorem area_of_region : MeasureTheory.volume R = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l3523_352329


namespace NUMINAMATH_CALUDE_window_area_calc_l3523_352307

/-- The area of a rectangular window -/
def window_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular window with length 6 feet and width 10 feet is 60 square feet -/
theorem window_area_calc :
  window_area 6 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_window_area_calc_l3523_352307


namespace NUMINAMATH_CALUDE_arithmetic_sequences_prime_term_l3523_352313

/-- Two arithmetic sequences with their sums -/
def ArithmeticSequences (a b : ℕ → ℕ) (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 6 : ℚ) / (n + 1 : ℚ)

/-- The m-th term of the second sequence is prime -/
def SecondSequencePrimeTerm (b : ℕ → ℕ) (m : ℕ) : Prop :=
  m > 0 ∧ Nat.Prime (b m)

theorem arithmetic_sequences_prime_term 
  (a b : ℕ → ℕ) (S T : ℕ → ℚ) (m : ℕ) :
  ArithmeticSequences a b S T →
  SecondSequencePrimeTerm b m →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_prime_term_l3523_352313


namespace NUMINAMATH_CALUDE_matrix_determinant_l3523_352308

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, 5; 1, -3, 2; 3, 6, -1]
  Matrix.det A = 57 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3523_352308


namespace NUMINAMATH_CALUDE_reflection_of_D_l3523_352341

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let p'' := (p'.2, p'.1)   -- Reflect over y = x
  (p''.1, p''.2 + 1)        -- Translate up by 1

def D : ℝ × ℝ := (4, 1)

theorem reflection_of_D : 
  reflect_y_eq_x_minus_1 (reflect_x D) = (-2, 5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_D_l3523_352341


namespace NUMINAMATH_CALUDE_decimal_digit_13_14_l3523_352305

def decimal_cycle (n d : ℕ) (cycle : List ℕ) : Prop :=
  ∀ k : ℕ, (n * 10^k) % d = (cycle.take ((k - 1) % cycle.length + 1)).foldl (λ acc x => (10 * acc + x) % d) 0

theorem decimal_digit_13_14 :
  decimal_cycle 13 14 [9, 2, 8, 5, 7, 1] →
  (13 * 10^150) / 14 % 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_decimal_digit_13_14_l3523_352305


namespace NUMINAMATH_CALUDE_probability_theorem_l3523_352360

/-- The probability of opening all safes given the number of keys and safes -/
def probability_open_all_safes (k n : ℕ) : ℚ :=
  if n > k then k / n else 1

/-- The theorem stating the probability of opening all safes -/
theorem probability_theorem (k n : ℕ) (h : n > k) :
  probability_open_all_safes k n = k / n := by
  sorry

#eval probability_open_all_safes 2 94

end NUMINAMATH_CALUDE_probability_theorem_l3523_352360


namespace NUMINAMATH_CALUDE_u_2008_eq_4008_l3523_352337

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ :=
  sorry

/-- Theorem stating that the 2008th term of the sequence is 4008 -/
theorem u_2008_eq_4008 : u 2008 = 4008 := by
  sorry

end NUMINAMATH_CALUDE_u_2008_eq_4008_l3523_352337


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3523_352335

-- Define set A
def A : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B (domain of lg(x-1))
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3523_352335


namespace NUMINAMATH_CALUDE_wheel_probability_l3523_352356

theorem wheel_probability (p_E p_F p_G p_H p_I : ℝ) : 
  p_E = 1/5 →
  p_F = 3/10 →
  p_G = p_H →
  p_I = 2 * p_G →
  p_E + p_F + p_G + p_H + p_I = 1 →
  p_G = 1/8 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l3523_352356


namespace NUMINAMATH_CALUDE_min_dot_product_on_W_l3523_352388

/-- The trajectory W of point P -/
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 2 ∧ p.1 ≥ Real.sqrt 2}

/-- The dot product of two vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

/-- The origin point -/
def O : ℝ × ℝ := (0, 0)

theorem min_dot_product_on_W :
  ∀ A B : ℝ × ℝ, A ∈ W → B ∈ W → A ≠ B →
  ∀ C D : ℝ × ℝ, C ∈ W → D ∈ W →
  dot_product (C.1 - O.1, C.2 - O.2) (D.1 - O.1, D.2 - O.2) ≥
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) →
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_W_l3523_352388


namespace NUMINAMATH_CALUDE_expression_evaluation_l3523_352342

theorem expression_evaluation : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3523_352342


namespace NUMINAMATH_CALUDE_infinite_primes_with_special_sequences_l3523_352397

/-- d_p(n) is the remainder of the Euclidean division of n by p -/
def d_p (p n : ℕ) : ℕ := n % p

/-- A p-sequence is a sequence (a_n) where a_{n+1} = a_n + d_p(a_n) for all n ≥ 0 -/
def is_p_sequence (p : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d_p p (a n)

/-- There exist infinitely many odd primes p such that two conditions hold -/
theorem infinite_primes_with_special_sequences : 
  ∃ S : Set ℕ, (∀ p ∈ S, Nat.Prime p ∧ Odd p) ∧ Set.Infinite S ∧
    (∀ p ∈ S, 
      (∃ a b : ℕ → ℕ, is_p_sequence p a ∧ is_p_sequence p b ∧
        (∃ T₁ T₂ : Set ℕ, Set.Infinite T₁ ∧ Set.Infinite T₂ ∧
          (∀ n ∈ T₁, a n > b n) ∧ (∀ n ∈ T₂, a n < b n))) ∧
      (∃ a b : ℕ → ℕ, is_p_sequence p a ∧ is_p_sequence p b ∧
        a 0 < b 0 ∧ ∀ n ≥ 1, a n > b n)) :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_with_special_sequences_l3523_352397


namespace NUMINAMATH_CALUDE_average_of_sixty_results_l3523_352327

theorem average_of_sixty_results (A : ℝ) : 
  (60 * A + 40 * 60) / 100 = 48 → A = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_of_sixty_results_l3523_352327


namespace NUMINAMATH_CALUDE_clearance_savings_l3523_352321

def coat_price : ℝ := 100
def pants_price : ℝ := 50
def coat_discount : ℝ := 0.30
def pants_discount : ℝ := 0.60

theorem clearance_savings : 
  let total_original := coat_price + pants_price
  let total_savings := coat_price * coat_discount + pants_price * pants_discount
  total_savings / total_original = 0.40 := by sorry

end NUMINAMATH_CALUDE_clearance_savings_l3523_352321


namespace NUMINAMATH_CALUDE_factorial_sum_l3523_352348

theorem factorial_sum : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 6 * Nat.factorial 6 = 40200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_l3523_352348


namespace NUMINAMATH_CALUDE_arthur_spent_fraction_l3523_352368

theorem arthur_spent_fraction (initial_amount : ℚ) (remaining_amount : ℚ) 
  (h1 : initial_amount = 200)
  (h2 : remaining_amount = 40) :
  (initial_amount - remaining_amount) / initial_amount = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_spent_fraction_l3523_352368


namespace NUMINAMATH_CALUDE_principal_calculation_l3523_352306

def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem principal_calculation (interest rate time : ℚ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 1)
  (h3 : time = 5) :
  ∃ (principal : ℚ), simple_interest principal rate time = interest ∧ principal = 80325 :=
by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3523_352306


namespace NUMINAMATH_CALUDE_torch_relay_probability_l3523_352383

/-- The number of torchbearers --/
def n : ℕ := 18

/-- The common difference of the arithmetic sequence --/
def d : ℕ := 3

/-- The probability of selecting three numbers from 1 to n that form an arithmetic 
    sequence with common difference d --/
def probability (n d : ℕ) : ℚ :=
  (3 * (n - 2 * d)) / (n * (n - 1) * (n - 2))

/-- The main theorem: the probability for the given problem is 1/68 --/
theorem torch_relay_probability : probability n d = 1 / 68 := by
  sorry


end NUMINAMATH_CALUDE_torch_relay_probability_l3523_352383


namespace NUMINAMATH_CALUDE_projection_implies_y_value_l3523_352302

/-- Given vectors v and w', if the projection of v on w' is proj_v_w, then y = -11/3 -/
theorem projection_implies_y_value (v w' proj_v_w : ℝ × ℝ) (y : ℝ) :
  v = (1, y) →
  w' = (-3, 1) →
  proj_v_w = (2, -2/3) →
  proj_v_w = (((v.1 * w'.1 + v.2 * w'.2) / (w'.1 ^ 2 + w'.2 ^ 2)) * w'.1,
              ((v.1 * w'.1 + v.2 * w'.2) / (w'.1 ^ 2 + w'.2 ^ 2)) * w'.2) →
  y = -11/3 := by
sorry

end NUMINAMATH_CALUDE_projection_implies_y_value_l3523_352302


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_right_l3523_352381

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  h : R = (a * Real.sqrt (b * c)) / (b + c)

/-- The angles of a triangle -/
structure Angles where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: If a triangle's circumradius satisfies the given equation, 
    then it is an isosceles right triangle -/
theorem triangle_is_isosceles_right (t : Triangle) : 
  ∃ (angles : Angles), 
    angles.α = 90 ∧ 
    angles.β = 45 ∧ 
    angles.γ = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_right_l3523_352381


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l3523_352310

theorem player_a_not_losing_probability 
  (p_win : ℝ) 
  (p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l3523_352310


namespace NUMINAMATH_CALUDE_basketball_weight_l3523_352332

theorem basketball_weight (skateboard_weight : ℝ) (num_skateboards num_basketballs : ℕ) :
  skateboard_weight = 20 →
  num_skateboards = 4 →
  num_basketballs = 5 →
  num_basketballs * (skateboard_weight * num_skateboards / num_basketballs) = num_skateboards * skateboard_weight →
  skateboard_weight * num_skateboards / num_basketballs = 16 :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_l3523_352332


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3523_352390

theorem greatest_integer_satisfying_inequality :
  ∀ n : ℤ, (∀ x : ℤ, 3 * |2 * x + 1| + 10 > 28 → x ≤ n) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3523_352390


namespace NUMINAMATH_CALUDE_circle_properties_l3523_352399

/-- Given a circle with circumference 36 cm, prove its radius, diameter, and area -/
theorem circle_properties (C : ℝ) (h : C = 36) :
  ∃ (r d A : ℝ),
    r = 18 / Real.pi ∧
    d = 36 / Real.pi ∧
    A = 324 / Real.pi ∧
    C = 2 * Real.pi * r ∧
    d = 2 * r ∧
    A = Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3523_352399


namespace NUMINAMATH_CALUDE_find_y_l3523_352363

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(2*y) = 4) (h2 : x = 4) : y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3523_352363


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3523_352312

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, x^2 - a*x + 3 < 0) → (a > 3 ∧ ∃ b > 3, ¬(∀ x ∈ Set.Icc 1 3, x^2 - b*x + 3 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3523_352312


namespace NUMINAMATH_CALUDE_inequality_proof_l3523_352339

theorem inequality_proof (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  1/(4-a^2) + 1/(4-b^2) + 1/(4-c^2) ≤ 9/((a+b+c)^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3523_352339


namespace NUMINAMATH_CALUDE_tens_digit_of_nine_power_2010_l3523_352318

def last_two_digits (n : ℕ) : ℕ := n % 100

def cycle_of_nine : List ℕ := [09, 81, 29, 61, 49, 41, 69, 21, 89, 01]

theorem tens_digit_of_nine_power_2010 :
  (last_two_digits (9^2010)) / 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_power_2010_l3523_352318


namespace NUMINAMATH_CALUDE_basis_vectors_classification_l3523_352301

def is_basis (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 ≠ 0 ∧ v₁ ≠ (0, 0) ∧ v₂ ≠ (0, 0)

theorem basis_vectors_classification :
  let a₁ : ℝ × ℝ := (0, 0)
  let a₂ : ℝ × ℝ := (1, 2)
  let b₁ : ℝ × ℝ := (2, -1)
  let b₂ : ℝ × ℝ := (1, 2)
  let c₁ : ℝ × ℝ := (-1, -2)
  let c₂ : ℝ × ℝ := (1, 2)
  let d₁ : ℝ × ℝ := (1, 1)
  let d₂ : ℝ × ℝ := (1, 2)
  ¬(is_basis a₁ a₂) ∧
  ¬(is_basis c₁ c₂) ∧
  (is_basis b₁ b₂) ∧
  (is_basis d₁ d₂) :=
by sorry

end NUMINAMATH_CALUDE_basis_vectors_classification_l3523_352301


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3523_352352

theorem smallest_number_divisible (a b c d e f : ℕ) (h1 : a = 35 ∧ b = 66)
  (h2 : c = 28 ∧ d = 165) (h3 : e = 25 ∧ f = 231) :
  ∃ (n : ℚ), n = 700 / 33 ∧
  (∃ (k1 k2 k3 : ℕ), n / (a / b) = k1 ∧ n / (c / d) = k2 ∧ n / (e / f) = k3) ∧
  ∀ (m : ℚ), m < n →
  ¬(∃ (l1 l2 l3 : ℕ), m / (a / b) = l1 ∧ m / (c / d) = l2 ∧ m / (e / f) = l3) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l3523_352352


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3523_352369

theorem imaginary_part_of_z (z : ℂ) : 
  z = Complex.I * (3 - 2 * Complex.I) * Complex.I ∧ z.re = 0 → z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3523_352369


namespace NUMINAMATH_CALUDE_independence_of_beta_l3523_352371

theorem independence_of_beta (α β : ℝ) : 
  ∃ (f : ℝ → ℝ), ∀ β, 
    (Real.sin (α + β))^2 + (Real.sin (β - α))^2 - 
    2 * Real.sin (α + β) * Real.sin (β - α) * Real.cos (2 * α) = f α :=
by sorry

end NUMINAMATH_CALUDE_independence_of_beta_l3523_352371


namespace NUMINAMATH_CALUDE_ab_not_necessary_nor_sufficient_for_a_plus_b_l3523_352380

theorem ab_not_necessary_nor_sufficient_for_a_plus_b :
  ∃ (a b : ℝ), (a * b > 0 ∧ a + b ≤ 0) ∧
  ∃ (c d : ℝ), (c * d ≤ 0 ∧ c + d > 0) := by
  sorry

end NUMINAMATH_CALUDE_ab_not_necessary_nor_sufficient_for_a_plus_b_l3523_352380


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l3523_352336

theorem number_puzzle_solution :
  ∃ (A B C D E : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A > 5 ∧
    A % B = 0 ∧
    C + A = D ∧
    B + C + E = A ∧
    B + C < E ∧
    C + E < B + 5 ∧
    A = 8 ∧ B = 2 ∧ C = 1 ∧ D = 9 ∧ E = 5 :=
by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l3523_352336


namespace NUMINAMATH_CALUDE_movie_ticket_price_l3523_352387

/-- The price of a 3D movie ticket --/
def price_3d : ℕ := sorry

/-- The price of a matinee ticket --/
def price_matinee : ℕ := 5

/-- The price of an evening ticket --/
def price_evening : ℕ := 12

/-- The number of matinee tickets sold --/
def num_matinee : ℕ := 200

/-- The number of evening tickets sold --/
def num_evening : ℕ := 300

/-- The number of 3D tickets sold --/
def num_3d : ℕ := 100

/-- The total revenue from all ticket sales --/
def total_revenue : ℕ := 6600

theorem movie_ticket_price :
  price_3d = 20 ∧
  price_matinee * num_matinee +
  price_evening * num_evening +
  price_3d * num_3d = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_price_l3523_352387


namespace NUMINAMATH_CALUDE_yellow_mms_added_l3523_352398

/-- Represents the number of M&Ms of each color in the jar -/
structure MandMs where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- The initial state of the jar -/
def initial_jar : MandMs :=
  { green := 20, red := 20, yellow := 0 }

/-- The state of the jar after Carter eats 12 green M&Ms -/
def after_carter_eats (jar : MandMs) : MandMs :=
  { jar with green := jar.green - 12 }

/-- The state of the jar after Carter's sister eats half the red M&Ms -/
def after_sister_eats (jar : MandMs) : MandMs :=
  { jar with red := jar.red / 2 }

/-- The final state of the jar after yellow M&Ms are added -/
def final_jar (jar : MandMs) (yellow_added : ℕ) : MandMs :=
  { jar with yellow := jar.yellow + yellow_added }

/-- The probability of picking a green M&M from the jar -/
def prob_green (jar : MandMs) : ℚ :=
  jar.green / (jar.green + jar.red + jar.yellow)

/-- The theorem stating the number of yellow M&Ms added -/
theorem yellow_mms_added : 
  ∃ yellow_added : ℕ,
    let jar1 := after_carter_eats initial_jar
    let jar2 := after_sister_eats jar1
    let jar3 := final_jar jar2 yellow_added
    prob_green jar3 = 1/4 ∧ yellow_added = 14 := by
  sorry

end NUMINAMATH_CALUDE_yellow_mms_added_l3523_352398


namespace NUMINAMATH_CALUDE_cori_age_relation_l3523_352320

theorem cori_age_relation (cori_age aunt_age : ℕ) (years : ℕ) : 
  cori_age = 3 → aunt_age = 19 → 
  (cori_age + years : ℚ) = (1 / 3) * (aunt_age + years : ℚ) → 
  years = 5 := by sorry

end NUMINAMATH_CALUDE_cori_age_relation_l3523_352320


namespace NUMINAMATH_CALUDE_cone_slant_height_l3523_352355

/-- The slant height of a cone with surface area 5π and net sector angle 90° is 4. -/
theorem cone_slant_height (r l : ℝ) (h1 : π * r^2 + π * r * l = 5 * π) 
  (h2 : 2 * π * r = 1/4 * (2 * π * l)) : l = 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3523_352355


namespace NUMINAMATH_CALUDE_opposite_of_two_opposite_definition_l3523_352389

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 2 is -2
theorem opposite_of_two : opposite 2 = -2 := by
  -- The proof goes here
  sorry

-- Theorem proving the definition of opposite
theorem opposite_definition (x : ℝ) : x + opposite x = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_opposite_definition_l3523_352389


namespace NUMINAMATH_CALUDE_train_speed_proof_l3523_352316

/-- The speed of the second train in km/hr -/
def second_train_speed : ℝ := 40

/-- The additional distance traveled by the first train in km -/
def additional_distance : ℝ := 100

/-- The total distance between P and Q in km -/
def total_distance : ℝ := 900

/-- The speed of the first train in km/hr -/
def first_train_speed : ℝ := 50

theorem train_speed_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    first_train_speed * t = second_train_speed * t + additional_distance ∧
    first_train_speed * t + second_train_speed * t = total_distance :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l3523_352316


namespace NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l3523_352378

/-- Given a unit square AEFD and rectangles ABCD and BCFE, where the ratio of length to width
    of ABCD equals the ratio of length to width of BCFE, and AB has length W,
    prove that W = (1 + √5) / 2. -/
theorem rectangle_ratio_golden_ratio (W : ℝ) : 
  (W > 0) →  -- W is positive
  (W / 1 = 1 / (W - 1)) →  -- ratio equality condition
  W = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l3523_352378


namespace NUMINAMATH_CALUDE_crescent_area_implies_square_area_l3523_352338

/-- Given a square with side length s, the area of 8 "crescent" shapes formed by
    semicircles on its sides and the sides of its inscribed square (formed by
    connecting midpoints) is equal to πs². If this area is 5 square centimeters,
    then the area of the original square is 10 square centimeters. -/
theorem crescent_area_implies_square_area :
  ∀ s : ℝ,
  s > 0 →
  π * s^2 = 5 →
  s^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_crescent_area_implies_square_area_l3523_352338


namespace NUMINAMATH_CALUDE_high_school_sample_senior_count_is_160_l3523_352392

theorem high_school_sample (total : ℕ) (junior_percent : ℚ) (not_sophomore_percent : ℚ) 
  (freshman_sophomore_diff : ℕ) : ℕ :=
  let junior_count : ℕ := (junior_percent * total).num.toNat
  let sophomore_count : ℕ := ((1 - not_sophomore_percent) * total).num.toNat
  let freshman_count : ℕ := sophomore_count + freshman_sophomore_diff
  total - (junior_count + sophomore_count + freshman_count)

theorem senior_count_is_160 :
  high_school_sample 800 (27/100) (75/100) 24 = 160 := by
  sorry

end NUMINAMATH_CALUDE_high_school_sample_senior_count_is_160_l3523_352392


namespace NUMINAMATH_CALUDE_continued_fraction_theorem_l3523_352346

-- Define the continued fraction for part 1
def continued_fraction_1 : ℚ :=
  1 + 1 / (2 + 1 / (3 + 1 / 4))

-- Define the continued fraction for part 2
def continued_fraction_2 (a b c : ℕ) : ℚ :=
  a + 1 / (b + 1 / c)

-- Define the equation for part 3
def continued_fraction_equation (y : ℝ) : Prop :=
  y = 8 + 1 / y

theorem continued_fraction_theorem :
  (continued_fraction_1 = 43 / 30) ∧
  (355 / 113 = continued_fraction_2 3 7 16) ∧
  (∃ y : ℝ, continued_fraction_equation y ∧ y = 4 + Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_continued_fraction_theorem_l3523_352346


namespace NUMINAMATH_CALUDE_root_value_theorem_l3523_352350

theorem root_value_theorem (a : ℝ) : 
  (a^2 - 4*a - 6 = 0) → (a^2 - 4*a + 3 = 9) := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3523_352350


namespace NUMINAMATH_CALUDE_trig_identity_l3523_352370

open Real

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (sin θ ^ 6 / a ^ 2 + cos θ ^ 6 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) →
  (sin θ ^ 12 / a ^ 5 + cos θ ^ 12 / b ^ 5 = 1 / a ^ 5) :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l3523_352370


namespace NUMINAMATH_CALUDE_parabola_intersection_implies_nonzero_c_l3523_352377

/-- Two points on a parabola -/
structure ParabolaPoints (a b c : ℝ) :=
  (x₁ : ℝ)
  (x₂ : ℝ)
  (y₁ : ℝ)
  (y₂ : ℝ)
  (on_parabola₁ : y₁ = x₁^2)
  (on_parabola₂ : y₂ = x₂^2)
  (on_quadratic₁ : y₁ = a * x₁^2 + b * x₁ + c)
  (on_quadratic₂ : y₂ = a * x₂^2 + b * x₂ + c)
  (opposite_sides : x₁ * x₂ < 0)
  (right_angle : (x₁ - x₂)^2 + (y₁ - y₂)^2 = x₁^2 + y₁^2 + x₂^2 + y₂^2)

theorem parabola_intersection_implies_nonzero_c (a b c : ℝ) :
  (∃ p : ParabolaPoints a b c, True) → c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_implies_nonzero_c_l3523_352377


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3523_352373

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) : 
  ((2 * Real.sin α ^ 2 + Real.sin (2 * α)) / Real.cos (2 * α) = 24 / 7) ∧ 
  (Real.tan (α + 5 * Real.pi / 4) = 7) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3523_352373


namespace NUMINAMATH_CALUDE_m_cubed_plus_two_m_squared_minus_2001_l3523_352384

theorem m_cubed_plus_two_m_squared_minus_2001 (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 - 2001 = -2000 := by
  sorry

end NUMINAMATH_CALUDE_m_cubed_plus_two_m_squared_minus_2001_l3523_352384


namespace NUMINAMATH_CALUDE_system_solution_unique_l3523_352330

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x - 2 * y = 5 ∧ x + 4 * y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3523_352330


namespace NUMINAMATH_CALUDE_vector_equation_sum_l3523_352334

/-- Given vectors a, b, c in R², if a = xb + yc for some real x and y, then x + y = 0 -/
theorem vector_equation_sum (a b c : Fin 2 → ℝ)
    (ha : a = ![3, -1])
    (hb : b = ![-1, 2])
    (hc : c = ![2, 1])
    (x y : ℝ)
    (h : a = x • b + y • c) :
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_sum_l3523_352334


namespace NUMINAMATH_CALUDE_pentagon_side_length_l3523_352324

/-- The side length of a regular pentagon with perimeter equal to that of an equilateral triangle with side length 20/9 cm is 4/3 cm. -/
theorem pentagon_side_length (s : ℝ) : 
  (5 * s = 3 * (20 / 9)) → s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_side_length_l3523_352324


namespace NUMINAMATH_CALUDE_shirt_price_l3523_352314

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The total cost of 3 pairs of jeans and 2 shirts is $69 -/
axiom first_purchase : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The total cost of 2 pairs of jeans and 3 shirts is $86 -/
axiom second_purchase : 2 * jeans_cost + 3 * shirt_cost = 86

/-- The cost of one shirt is $24 -/
theorem shirt_price : shirt_cost = 24 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l3523_352314


namespace NUMINAMATH_CALUDE_janet_total_miles_l3523_352359

/-- Represents Janet's running schedule for a week -/
structure WeekSchedule where
  days : ℕ
  milesPerDay : ℕ

/-- Calculates the total miles run in a week -/
def weeklyMiles (schedule : WeekSchedule) : ℕ :=
  schedule.days * schedule.milesPerDay

/-- Janet's running schedule for three weeks -/
def janetSchedule : List WeekSchedule :=
  [{ days := 5, milesPerDay := 8 },
   { days := 4, milesPerDay := 10 },
   { days := 3, milesPerDay := 6 }]

/-- Theorem: Janet ran a total of 98 miles over the three weeks -/
theorem janet_total_miles :
  (janetSchedule.map weeklyMiles).sum = 98 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_miles_l3523_352359


namespace NUMINAMATH_CALUDE_speed_conversion_l3523_352358

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph_factor : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed : ℝ := 20

/-- Theorem: Converting 20 mps to kmph results in 72 kmph -/
theorem speed_conversion :
  given_speed * mps_to_kmph_factor = 72 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3523_352358


namespace NUMINAMATH_CALUDE_peaches_in_knapsack_l3523_352340

/-- Given a total of 60 peaches distributed among two identical bags and a knapsack,
    where the knapsack contains half as many peaches as each bag,
    prove that the number of peaches in the knapsack is 12. -/
theorem peaches_in_knapsack :
  let total_peaches : ℕ := 60
  let knapsack_peaches : ℕ := x
  let bag_peaches : ℕ := 2 * x
  x + bag_peaches + bag_peaches = total_peaches →
  x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_peaches_in_knapsack_l3523_352340


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l3523_352333

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop :=
  x^2 - 16*y^2 - 8*x + 16 = 0

/-- The first line represented by the equation -/
def line1 (x y : ℝ) : Prop :=
  x = 4 + 4*y

/-- The second line represented by the equation -/
def line2 (x y : ℝ) : Prop :=
  x = 4 - 4*y

/-- Theorem stating that the equation represents two lines -/
theorem equation_represents_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l3523_352333


namespace NUMINAMATH_CALUDE_short_bar_length_l3523_352396

/-- Given a stick of length 950 cm divided into two parts, where one part is 150 cm longer than the other, 
    the length of the shorter part is 400 cm. -/
theorem short_bar_length (total_length : ℝ) (difference : ℝ) (short_length : ℝ) : 
  total_length = 950 ∧ difference = 150 →
  short_length + (short_length + difference) = total_length →
  short_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_short_bar_length_l3523_352396


namespace NUMINAMATH_CALUDE_scaled_variance_l3523_352367

-- Define a dataset type
def Dataset := List Real

-- Define the standard deviation function
noncomputable def standardDeviation (data : Dataset) : Real :=
  sorry

-- Define the variance function
noncomputable def variance (data : Dataset) : Real :=
  sorry

-- Define a function to scale a dataset
def scaleDataset (data : Dataset) (scale : Real) : Dataset :=
  data.map (· * scale)

-- Theorem statement
theorem scaled_variance 
  (data : Dataset) 
  (h : standardDeviation data = 2) : 
  variance (scaleDataset data 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_scaled_variance_l3523_352367


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l3523_352386

theorem larger_number_of_pair (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 156) (h3 : x > y) :
  x = (5 + Real.sqrt 649) / 2 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l3523_352386


namespace NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l3523_352311

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given natural number. -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range (n/2 + 1))).card

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50. -/
theorem four_prime_pairs_sum_50 : count_prime_pairs 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l3523_352311


namespace NUMINAMATH_CALUDE_tire_circumference_l3523_352372

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (rotation_speed : ℝ) (car_velocity : ℝ) : 
  rotation_speed = 400 ∧ car_velocity = 96 → 
  (car_velocity * 1000 / 60) / rotation_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l3523_352372


namespace NUMINAMATH_CALUDE_problem_solution_l3523_352351

theorem problem_solution (x y : ℝ) (hx : x = 1/2) (hy : y = 2) :
  (1/3) * x^8 * y^9 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3523_352351


namespace NUMINAMATH_CALUDE_correct_reasoning_directions_l3523_352365

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define a function that describes the direction of each reasoning type
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating the correct reasoning directions
theorem correct_reasoning_directions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_directions_l3523_352365


namespace NUMINAMATH_CALUDE_remaining_payment_l3523_352357

def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 105

theorem remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) :
  deposit_percentage = 10 / 100 →
  deposit_amount = 105 →
  (deposit_amount / deposit_percentage) - deposit_amount = 945 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l3523_352357


namespace NUMINAMATH_CALUDE_range_of_a_l3523_352343

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 8*x - 20 ≤ 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x, x^2 - 8*x - 20 ≤ 0 ∧ x^2 - 2*x + 1 - a^2 > 0) ∧
  a > 0 → 
  a ≥ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3523_352343


namespace NUMINAMATH_CALUDE_mike_tv_and_games_time_l3523_352326

/-- Given Mike's TV and video game habits, prove the total time spent on both activities in a week. -/
theorem mike_tv_and_games_time (tv_hours_per_day : ℕ) (video_game_days_per_week : ℕ) : 
  tv_hours_per_day = 4 →
  video_game_days_per_week = 3 →
  (tv_hours_per_day * 7 + video_game_days_per_week * (tv_hours_per_day / 2)) = 34 := by
sorry


end NUMINAMATH_CALUDE_mike_tv_and_games_time_l3523_352326


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l3523_352354

theorem weekend_rain_probability (prob_saturday prob_sunday : ℝ) 
  (h1 : prob_saturday = 0.3)
  (h2 : prob_sunday = 0.6)
  (h3 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1)
  (h4 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  1 - (1 - prob_saturday) * (1 - prob_sunday) = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l3523_352354


namespace NUMINAMATH_CALUDE_girls_in_class_l3523_352353

theorem girls_in_class (total : ℕ) (difference : ℕ) : 
  total = 63 → difference = 7 → (total + difference) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l3523_352353


namespace NUMINAMATH_CALUDE_find_m_max_sum_squares_max_sum_squares_achievable_l3523_352379

-- Define the condition for the unique integer solution
def uniqueIntegerSolution (m : ℤ) : Prop :=
  ∃! (x : ℤ), |2 * x - m| ≤ 1

-- Define the condition for a, b, c
def abcCondition (a b c : ℝ) : Prop :=
  4 * a^4 + 4 * b^4 + 4 * c^4 = 6

-- Theorem 1: Prove m = 6
theorem find_m (m : ℤ) (h : uniqueIntegerSolution m) : m = 6 := by
  sorry

-- Theorem 2: Prove the maximum value of a^2 + b^2 + c^2
theorem max_sum_squares (a b c : ℝ) (h : abcCondition a b c) :
  a^2 + b^2 + c^2 ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

-- Theorem 3: Prove the maximum value is achievable
theorem max_sum_squares_achievable :
  ∃ a b c : ℝ, abcCondition a b c ∧ a^2 + b^2 + c^2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_m_max_sum_squares_max_sum_squares_achievable_l3523_352379


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l3523_352315

/-- Given a hyperbola with equation x^2 - my^2 = 3m (where m > 0),
    prove that the value of b in its standard form is √3. -/
theorem hyperbola_b_value (m : ℝ) (h : m > 0) :
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 - m*y^2 = 3*m
  ∃ (a b : ℝ), (∀ (x y : ℝ), C (x, y) ↔ (x^2 / (a^2) - y^2 / (b^2) = 1)) ∧ b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l3523_352315
