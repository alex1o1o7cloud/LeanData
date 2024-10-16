import Mathlib

namespace NUMINAMATH_CALUDE_logarithm_equations_l2061_206199

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm with arbitrary base
noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_equations :
  (lg 4 + lg 500 - lg 2 = 3) ∧
  ((27 : ℝ)^(1/3) + (log 3 2) * (log 2 3) = 4) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equations_l2061_206199


namespace NUMINAMATH_CALUDE_acid_mixture_concentration_l2061_206187

/-- Proves that mixing 1.2 L of 10% acid solution with 0.8 L of 5% acid solution 
    results in a 2 L solution with 8% acid concentration -/
theorem acid_mixture_concentration : 
  let total_volume : Real := 2
  let volume_10_percent : Real := 1.2
  let volume_5_percent : Real := total_volume - volume_10_percent
  let concentration_10_percent : Real := 10 / 100
  let concentration_5_percent : Real := 5 / 100
  let total_acid : Real := 
    volume_10_percent * concentration_10_percent + 
    volume_5_percent * concentration_5_percent
  let final_concentration : Real := (total_acid / total_volume) * 100
  final_concentration = 8 := by sorry

end NUMINAMATH_CALUDE_acid_mixture_concentration_l2061_206187


namespace NUMINAMATH_CALUDE_ducks_at_lake_michigan_l2061_206140

theorem ducks_at_lake_michigan (ducks_north_pond : ℕ) (ducks_lake_michigan : ℕ) : 
  ducks_north_pond = 2 * ducks_lake_michigan + 6 →
  ducks_north_pond = 206 →
  ducks_lake_michigan = 100 := by
sorry

end NUMINAMATH_CALUDE_ducks_at_lake_michigan_l2061_206140


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2061_206141

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 6) % 12 = 0 ∧
  (n - 6) % 16 = 0 ∧
  (n - 6) % 18 = 0 ∧
  (n - 6) % 21 = 0 ∧
  (n - 6) % 28 = 0 ∧
  (n - 6) % 35 = 0 ∧
  (n - 6) % 39 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 65526 ∧
  ∀ m : ℕ, m < 65526 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2061_206141


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2061_206145

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 2 ∧ b = 3 ∧ c^2 = a^2 + b^2 → c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2061_206145


namespace NUMINAMATH_CALUDE_trapezoid_area_is_42_5_l2061_206133

/-- A trapezoid bounded by the lines y = x + 2, y = 12, y = 7, and the y-axis -/
structure Trapezoid where
  -- Line equations
  line1 : ℝ → ℝ := λ x => x + 2
  line2 : ℝ → ℝ := λ _ => 12
  line3 : ℝ → ℝ := λ _ => 7
  y_axis : ℝ → ℝ := λ _ => 0

/-- The area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the area of the trapezoid is 42.5 square units -/
theorem trapezoid_area_is_42_5 (t : Trapezoid) : trapezoid_area t = 42.5 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_42_5_l2061_206133


namespace NUMINAMATH_CALUDE_bracelet_sale_earnings_l2061_206122

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  total_bracelets : ℕ
  single_price : ℕ
  pair_price : ℕ
  single_sales : ℕ

/-- Calculates the total earnings from selling bracelets -/
def total_earnings (sale : BraceletSale) : ℕ :=
  let remaining_bracelets := sale.total_bracelets - (sale.single_sales / sale.single_price)
  let pair_sales := remaining_bracelets / 2
  (sale.single_sales / sale.single_price) * sale.single_price + pair_sales * sale.pair_price

/-- Theorem stating that the total earnings from the given scenario is $132 -/
theorem bracelet_sale_earnings :
  let sale : BraceletSale := {
    total_bracelets := 30,
    single_price := 5,
    pair_price := 8,
    single_sales := 60
  }
  total_earnings sale = 132 := by sorry

end NUMINAMATH_CALUDE_bracelet_sale_earnings_l2061_206122


namespace NUMINAMATH_CALUDE_extreme_points_condition_l2061_206126

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^3 + 2*a*x^2 + x + 1

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 4*a*x + 1

-- Theorem statement
theorem extreme_points_condition (a x₁ x₂ : ℝ) : 
  (f_derivative a x₁ = 0) →  -- x₁ is an extreme point
  (f_derivative a x₂ = 0) →  -- x₂ is an extreme point
  (x₂ - x₁ = 2) →            -- Given condition
  (a^2 = 3) :=               -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_extreme_points_condition_l2061_206126


namespace NUMINAMATH_CALUDE_unique_divisible_by_29_l2061_206142

/-- Converts a base 7 number of the form 34x1 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

/-- Checks if a number is divisible by 29 --/
def isDivisibleBy29 (n : ℕ) : Prop := n % 29 = 0

theorem unique_divisible_by_29 :
  ∃! x : ℕ, x < 7 ∧ isDivisibleBy29 (base7ToDecimal x) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_29_l2061_206142


namespace NUMINAMATH_CALUDE_tara_had_fifteen_l2061_206100

/-- The amount of money Megan has -/
def megan_money : ℕ := sorry

/-- The amount of money Tara has -/
def tara_money : ℕ := megan_money + 4

/-- The cost of the scooter -/
def scooter_cost : ℕ := 26

/-- Theorem stating that Tara had $15 -/
theorem tara_had_fifteen :
  (megan_money + tara_money = scooter_cost) →
  tara_money = 15 := by
  sorry

end NUMINAMATH_CALUDE_tara_had_fifteen_l2061_206100


namespace NUMINAMATH_CALUDE_half_power_inequality_l2061_206163

theorem half_power_inequality (m n : ℝ) (h : m > n) : (1/2 : ℝ)^m < (1/2 : ℝ)^n := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l2061_206163


namespace NUMINAMATH_CALUDE_shape_division_count_l2061_206167

/-- A shape with 17 cells -/
def Shape : Type := Unit

/-- A rectangle of size 1 × 2 -/
def Rectangle : Type := Unit

/-- A square of size 1 × 1 -/
def Square : Type := Unit

/-- A division of the shape into rectangles and a square -/
def Division : Type := List Rectangle × Square

/-- The number of ways to divide the shape -/
def numDivisions (s : Shape) : ℕ := 10

/-- Theorem: There are 10 ways to divide the shape into 8 rectangles and 1 square -/
theorem shape_division_count (s : Shape) :
  (numDivisions s = 10) ∧
  (∀ d : Division, List.length (d.1) = 8) :=
sorry

end NUMINAMATH_CALUDE_shape_division_count_l2061_206167


namespace NUMINAMATH_CALUDE_prob_two_cards_two_suits_l2061_206178

/-- The probability of drawing a card of a specific suit from a standard deck -/
def prob_specific_suit : ℚ := 1 / 4

/-- The number of cards drawn -/
def num_draws : ℕ := 6

/-- The number of suits we're interested in -/
def num_suits : ℕ := 2

/-- The number of cards needed from each suit -/
def cards_per_suit : ℕ := 2

/-- The probability of getting the desired outcome when drawing six cards with replacement -/
def prob_desired_outcome : ℚ := (prob_specific_suit ^ (num_draws : ℕ))

theorem prob_two_cards_two_suits :
  prob_desired_outcome = 1 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_cards_two_suits_l2061_206178


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2061_206153

theorem decimal_to_fraction (d : ℚ) (h : d = 0.34) : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d.gcd n = 1 ∧ (n : ℚ) / d = 0.34 ∧ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2061_206153


namespace NUMINAMATH_CALUDE_sum_of_triangle_perimeters_sum_of_specific_triangle_perimeters_l2061_206198

/-- The sum of perimeters of an infinite series of equilateral triangles -/
theorem sum_of_triangle_perimeters (initial_perimeter : ℝ) : 
  initial_perimeter > 0 →
  (∑' n, initial_perimeter * (1/2)^n) = 2 * initial_perimeter :=
by
  sorry

/-- The specific case where the initial triangle has a perimeter of 180 cm -/
theorem sum_of_specific_triangle_perimeters : 
  (∑' n, 180 * (1/2)^n) = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangle_perimeters_sum_of_specific_triangle_perimeters_l2061_206198


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2061_206106

theorem fifth_term_of_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2061_206106


namespace NUMINAMATH_CALUDE_coupon_probability_l2061_206194

def total_coupons : ℕ := 17
def semyon_coupons : ℕ := 9
def temyon_missing : ℕ := 6

theorem coupon_probability : 
  (Nat.choose temyon_missing temyon_missing * Nat.choose (total_coupons - temyon_missing) (semyon_coupons - temyon_missing)) / 
  Nat.choose total_coupons semyon_coupons = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_probability_l2061_206194


namespace NUMINAMATH_CALUDE_garden_roller_diameter_l2061_206176

/-- The diameter of a garden roller given its length, area covered, and number of revolutions -/
theorem garden_roller_diameter 
  (length : ℝ) 
  (area_covered : ℝ) 
  (revolutions : ℝ) 
  (h1 : length = 3) 
  (h2 : area_covered = 66) 
  (h3 : revolutions = 5) : 
  ∃ (diameter : ℝ), diameter = 1.4 ∧ 
    area_covered = revolutions * 2 * (22/7) * (diameter/2) * length := by
sorry

end NUMINAMATH_CALUDE_garden_roller_diameter_l2061_206176


namespace NUMINAMATH_CALUDE_complement_of_union_l2061_206130

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- Theorem statement
theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2061_206130


namespace NUMINAMATH_CALUDE_two_digit_sum_theorem_l2061_206110

def is_valid_set (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  (10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b) = 484

def valid_sets : List (Fin 10 × Fin 10 × Fin 10) :=
  [(9, 4, 9), (9, 5, 8), (9, 6, 7), (8, 6, 8), (8, 7, 7)]

theorem two_digit_sum_theorem (a b c : ℕ) :
  is_valid_set a b c →
  (a, b, c) ∈ valid_sets.map (fun (x, y, z) => (x.val, y.val, z.val)) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_theorem_l2061_206110


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2061_206115

theorem consecutive_integers_average (x : ℤ) : 
  (((x - 9) + (x - 7) + (x - 5) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 : ℚ) = 31/2 →
  ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 : ℚ) = 49/2 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2061_206115


namespace NUMINAMATH_CALUDE_whistle_solution_l2061_206161

/-- The number of whistles Sean, Charles, and Alex have. -/
def whistle_problem (W_Sean W_Charles W_Alex : ℕ) : Prop :=
  W_Sean = 2483 ∧ 
  W_Charles = W_Sean - 463 ∧
  W_Alex = W_Charles - 131

theorem whistle_solution :
  ∀ W_Sean W_Charles W_Alex : ℕ,
  whistle_problem W_Sean W_Charles W_Alex →
  W_Charles = 2020 ∧ 
  W_Alex = 1889 ∧
  W_Sean + W_Charles + W_Alex = 6392 :=
by
  sorry

#check whistle_solution

end NUMINAMATH_CALUDE_whistle_solution_l2061_206161


namespace NUMINAMATH_CALUDE_remainder_of_b_mod_29_l2061_206193

theorem remainder_of_b_mod_29 :
  let b := (((13⁻¹ : ZMod 29) + (17⁻¹ : ZMod 29) + (19⁻¹ : ZMod 29))⁻¹ : ZMod 29)
  b = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_b_mod_29_l2061_206193


namespace NUMINAMATH_CALUDE_triangle_angle_properties_l2061_206132

theorem triangle_angle_properties (α : Real) (h1 : 0 < α) (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  (Real.tan α = -4/3) ∧ 
  ((Real.sin (3*π/2 + α) * Real.sin (π/2 - α) * Real.tan (π - α)^3) / 
   (Real.cos (π/2 + α) * Real.cos (3*π/2 - α)) = -4/3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_properties_l2061_206132


namespace NUMINAMATH_CALUDE_spinner_probability_l2061_206177

/-- Represents a game board based on an equilateral triangle -/
structure GameBoard where
  /-- The number of regions formed by altitudes and one median -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Ensure the number of shaded regions is less than or equal to the total regions -/
  h_valid : shaded_regions ≤ total_regions

/-- Calculate the probability of landing in a shaded region -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- The main theorem to be proved -/
theorem spinner_probability (board : GameBoard) 
  (h_total : board.total_regions = 12)
  (h_shaded : board.shaded_regions = 3) : 
  probability board = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2061_206177


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l2061_206191

/-- Calculates the final length of Isabella's hair after growth -/
def final_hair_length (initial_length growth : ℕ) : ℕ := initial_length + growth

/-- Theorem: Isabella's hair length after growth -/
theorem isabellas_hair_growth (initial_length growth : ℕ) 
  (h1 : initial_length = 18) 
  (h2 : growth = 4) : 
  final_hair_length initial_length growth = 22 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l2061_206191


namespace NUMINAMATH_CALUDE_function_shift_l2061_206129

/-- Given a function f with the specified properties, prove that g can be obtained
    by shifting f to the left by π/8 units. -/
theorem function_shift (ω : ℝ) (h1 : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 4)
  let g : ℝ → ℝ := λ x ↦ Real.cos (ω * x)
  (∀ x, f (x + π / ω) = f x) →  -- minimum positive period is π
  ∀ x, g x = f (x + π / 8) := by
sorry

end NUMINAMATH_CALUDE_function_shift_l2061_206129


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2061_206112

theorem largest_reciprocal (a b c d e : ℝ) 
  (ha : a = 1/4) 
  (hb : b = 3/7) 
  (hc : c = 0.25) 
  (hd : d = 7) 
  (he : e = 5000) : 
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l2061_206112


namespace NUMINAMATH_CALUDE_man_walking_problem_l2061_206120

theorem man_walking_problem (x : ℝ) :
  let final_x := x - 6 * Real.sin (2 * Real.pi / 3)
  let final_y := 6 * Real.cos (2 * Real.pi / 3)
  final_x ^ 2 + final_y ^ 2 = 12 →
  x = 3 * Real.sqrt 3 + Real.sqrt 3 ∨ x = 3 * Real.sqrt 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_man_walking_problem_l2061_206120


namespace NUMINAMATH_CALUDE_chairperson_and_committee_count_l2061_206104

/-- The number of ways to choose a chairperson and a 3-person committee from a group of 10 people,
    where the chairperson is not a member of the committee. -/
def chairperson_and_committee (total_people : ℕ) (committee_size : ℕ) : ℕ :=
  total_people * (Nat.choose (total_people - 1) committee_size)

/-- Theorem stating that the number of ways to choose a chairperson and a 3-person committee
    from a group of 10 people, where the chairperson is not a member of the committee, is 840. -/
theorem chairperson_and_committee_count :
  chairperson_and_committee 10 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_chairperson_and_committee_count_l2061_206104


namespace NUMINAMATH_CALUDE_perimeter_of_three_quarter_circles_l2061_206171

/-- The perimeter of a figure formed by three quarters of a circle with area 36π -/
theorem perimeter_of_three_quarter_circles (r : ℝ) : 
  r^2 * π = 36 * π → 
  3 * (π * r / 2) + 2 * r = 9 * π + 12 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_three_quarter_circles_l2061_206171


namespace NUMINAMATH_CALUDE_jewelry_thief_l2061_206197

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the properties
def is_telling_truth (p : Person) : Prop := sorry
def is_thief (p : Person) : Prop := sorry

-- State the theorem
theorem jewelry_thief :
  -- Only one person is telling the truth
  (∃! p : Person, is_telling_truth p) →
  -- Only one person stole the jewelry
  (∃! p : Person, is_thief p) →
  -- A's statement
  (is_telling_truth Person.A ↔ ¬is_thief Person.A) →
  -- B's statement
  (is_telling_truth Person.B ↔ is_thief Person.C) →
  -- C's statement
  (is_telling_truth Person.C ↔ is_thief Person.D) →
  -- D's statement
  (is_telling_truth Person.D ↔ ¬is_thief Person.D) →
  -- Conclusion: A stole the jewelry
  is_thief Person.A :=
by
  sorry


end NUMINAMATH_CALUDE_jewelry_thief_l2061_206197


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2061_206134

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_distance (P : ℝ × ℝ) (h1 : is_on_ellipse P.1 P.2) 
  (h2 : distance P F1 = 2) : distance P F2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2061_206134


namespace NUMINAMATH_CALUDE_rope_length_comparison_l2061_206107

theorem rope_length_comparison (L : ℝ) (h : L > 0) : 
  ¬ (∀ L, L - (1/3) = L - (L/3)) :=
sorry

end NUMINAMATH_CALUDE_rope_length_comparison_l2061_206107


namespace NUMINAMATH_CALUDE_calculate_premium_percentage_l2061_206125

/-- Given an investment scenario, calculate the premium percentage on shares. -/
theorem calculate_premium_percentage
  (total_investment : ℝ)
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : total_investment = 14400)
  (h2 : face_value = 100)
  (h3 : dividend_rate = 0.05)
  (h4 : total_dividend = 600) :
  (total_investment / (total_dividend / (dividend_rate * face_value)) - face_value) / face_value * 100 = 20 := by
sorry


end NUMINAMATH_CALUDE_calculate_premium_percentage_l2061_206125


namespace NUMINAMATH_CALUDE_max_value_inequality_l2061_206105

theorem max_value_inequality (m n : ℝ) (hm : m ≠ -3) :
  (∀ x : ℝ, x - 3 * Real.log x + 1 ≥ m * Real.log x + n) →
  (∃ k : ℝ, k = (n - 3) / (m + 3) ∧
    k ≤ -Real.log 2 ∧
    ∀ l : ℝ, l = (n - 3) / (m + 3) → l ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2061_206105


namespace NUMINAMATH_CALUDE_quadratic_solution_base_n_l2061_206136

/-- Given an integer n > 8, if n is a solution of x^2 - ax + b = 0 where a in base-n is 21,
    then b in base-n is 101. -/
theorem quadratic_solution_base_n (n : ℕ) (a b : ℕ) (h1 : n > 8) 
  (h2 : n^2 - a*n + b = 0) (h3 : a = 2*n + 1) : 
  b = n^2 + n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_base_n_l2061_206136


namespace NUMINAMATH_CALUDE_partner_a_profit_share_l2061_206127

/-- Calculates the share of profit for partner A in a business partnership --/
theorem partner_a_profit_share
  (initial_a initial_b : ℕ)
  (withdrawal_a addition_b : ℕ)
  (total_months : ℕ)
  (change_month : ℕ)
  (total_profit : ℕ)
  (h1 : initial_a = 2000)
  (h2 : initial_b = 4000)
  (h3 : withdrawal_a = 1000)
  (h4 : addition_b = 1000)
  (h5 : total_months = 12)
  (h6 : change_month = 8)
  (h7 : total_profit = 630) :
  let investment_months_a := initial_a * change_month + (initial_a - withdrawal_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + addition_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  let a_share := (investment_months_a * total_profit) / total_investment_months
  a_share = 175 := by sorry

end NUMINAMATH_CALUDE_partner_a_profit_share_l2061_206127


namespace NUMINAMATH_CALUDE_max_residents_is_475_l2061_206174

/-- Represents the configuration of a section in the block of flats -/
structure SectionConfig where
  floors : Nat
  apartments_per_floor : Nat
  apartment_capacities : List Nat

/-- Calculates the maximum capacity of a single section -/
def section_capacity (config : SectionConfig) : Nat :=
  config.floors * (config.apartment_capacities.sum)

/-- Represents the configuration of the entire block of flats -/
structure BlockConfig where
  lower : SectionConfig
  middle : SectionConfig
  upper : SectionConfig

/-- Calculates the maximum capacity of the entire block -/
def block_capacity (config : BlockConfig) : Nat :=
  section_capacity config.lower + section_capacity config.middle + section_capacity config.upper

/-- The actual configuration of the block as described in the problem -/
def actual_block_config : BlockConfig :=
  { lower := { floors := 4, apartments_per_floor := 5, apartment_capacities := [4, 4, 5, 5, 6] },
    middle := { floors := 5, apartments_per_floor := 6, apartment_capacities := [3, 3, 3, 4, 4, 6] },
    upper := { floors := 6, apartments_per_floor := 5, apartment_capacities := [8, 8, 8, 10, 10] } }

/-- Theorem stating that the maximum capacity of the block is 475 -/
theorem max_residents_is_475 : block_capacity actual_block_config = 475 := by
  sorry

end NUMINAMATH_CALUDE_max_residents_is_475_l2061_206174


namespace NUMINAMATH_CALUDE_max_value_on_circle_l2061_206131

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 10 →
  4*x + 3*y ≤ 32 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l2061_206131


namespace NUMINAMATH_CALUDE_triangle_side_length_l2061_206128

/-- In a triangle ABC, if a = 1, c = 2, and B = 60°, then b = √3 -/
theorem triangle_side_length (a c b : ℝ) (B : ℝ) : 
  a = 1 → c = 2 → B = π / 3 → b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) → b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2061_206128


namespace NUMINAMATH_CALUDE_shopkeeper_bananas_l2061_206147

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) : 
  oranges = 600 →
  (oranges * 85 / 100 + bananas * 97 / 100 : ℚ) = (oranges + bananas) * 898 / 1000 →
  bananas = 400 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_bananas_l2061_206147


namespace NUMINAMATH_CALUDE_five_seventeenths_repetend_l2061_206101

def decimal_repetend (n d : ℕ) (repetend : List ℕ) : Prop :=
  ∃ (k : ℕ), (n : ℚ) / d = (k : ℚ) / 10^(repetend.length) + 
    (List.sum (List.zipWith (λ (digit place) => (digit : ℚ) / 10^place) repetend 
    (List.range repetend.length))) / (10^(repetend.length) - 1)

theorem five_seventeenths_repetend :
  decimal_repetend 5 17 [2, 9, 4, 1, 1, 7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5] :=
sorry

end NUMINAMATH_CALUDE_five_seventeenths_repetend_l2061_206101


namespace NUMINAMATH_CALUDE_area_of_four_squares_l2061_206109

/-- The area of a shape composed of four identical squares with side length 3 cm is 36 cm² -/
theorem area_of_four_squares (side_length : ℝ) (h1 : side_length = 3) : 
  4 * (side_length ^ 2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_four_squares_l2061_206109


namespace NUMINAMATH_CALUDE_triangle_and_vector_theorem_l2061_206181

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * cos t.B = t.b * cos t.C

/-- The vector m -/
def m (A : ℝ) : ℝ × ℝ := (sin A, cos (2 * A))

/-- The vector n -/
def n : ℝ × ℝ := (6, 1)

/-- The dot product of m and n -/
def dotProduct (A : ℝ) : ℝ := 6 * sin A + cos (2 * A)

/-- The main theorem -/
theorem triangle_and_vector_theorem (t : Triangle) 
  (h : triangleCondition t) : 
  t.B = π / 3 ∧ 
  (∀ A, dotProduct A ≤ 5) ∧ 
  (∃ A, dotProduct A = 5) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_and_vector_theorem_l2061_206181


namespace NUMINAMATH_CALUDE_jerome_money_left_l2061_206158

def jerome_problem (initial_half : ℕ) (meg_amount : ℕ) : Prop :=
  let initial_total := 2 * initial_half
  let after_meg := initial_total - meg_amount
  let bianca_amount := 3 * meg_amount
  let final_amount := after_meg - bianca_amount
  final_amount = 54

theorem jerome_money_left : jerome_problem 43 8 := by
  sorry

end NUMINAMATH_CALUDE_jerome_money_left_l2061_206158


namespace NUMINAMATH_CALUDE_constant_function_proof_l2061_206111

theorem constant_function_proof (f : ℝ → ℝ) 
  (h_continuous : Continuous f) 
  (h_condition : ∀ (x : ℝ) (t : ℝ), t ≥ 0 → f x = f (Real.exp t * x)) : 
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l2061_206111


namespace NUMINAMATH_CALUDE_set_operations_l2061_206155

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x < 4}

theorem set_operations :
  (A ∪ B = {x | -1 < x ∧ x < 4}) ∧
  (A ∩ B = {x | 1 ≤ x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2061_206155


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2061_206118

def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2061_206118


namespace NUMINAMATH_CALUDE_line_intercepts_l2061_206144

/-- A line in the 2D plane defined by the equation y = x + 3 -/
structure Line where
  slope : ℝ := 1
  y_intercept : ℝ := 3

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ × ℝ :=
  (-l.y_intercept, 0)

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.y_intercept)

theorem line_intercepts (l : Line) :
  x_intercept l = (-3, 0) ∧ y_intercept l = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l2061_206144


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2061_206150

theorem tan_alpha_value (α : ℝ) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) : 
  Real.tan α = -1/3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2061_206150


namespace NUMINAMATH_CALUDE_function_properties_l2061_206154

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- f'(x) is symmetric about x = -1/2
  f' a b 1 = 0 →                           -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                        -- values of a and b
  f a b (-2) = 21 ∧ f a b 1 = -6           -- extreme values
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l2061_206154


namespace NUMINAMATH_CALUDE_can_collection_increase_l2061_206116

/-- Proves that the daily increase in can collection is 5 cans --/
theorem can_collection_increase (initial_cans : ℕ) (days : ℕ) (total_cans : ℕ) 
  (h1 : initial_cans = 20)
  (h2 : days = 5)
  (h3 : total_cans = 150)
  (h4 : ∃ x : ℕ, total_cans = initial_cans * days + (days * (days - 1) / 2) * x) :
  ∃ x : ℕ, x = 5 ∧ total_cans = initial_cans * days + (days * (days - 1) / 2) * x := by
  sorry

end NUMINAMATH_CALUDE_can_collection_increase_l2061_206116


namespace NUMINAMATH_CALUDE_range_of_a_l2061_206117

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2061_206117


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2061_206186

theorem sin_300_degrees : Real.sin (300 * Real.pi / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2061_206186


namespace NUMINAMATH_CALUDE_remainder_proof_l2061_206172

theorem remainder_proof : (2058167 + 934) % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2061_206172


namespace NUMINAMATH_CALUDE_bookstore_problem_l2061_206146

/-- Represents the number of magazine types at each price point -/
structure MagazineTypes :=
  (twoYuan : ℕ)
  (oneYuan : ℕ)

/-- Represents the total budget and purchasing constraints -/
structure PurchaseConstraints :=
  (budget : ℕ)
  (maxPerType : ℕ)

/-- Calculates the number of different purchasing methods -/
def purchasingMethods (types : MagazineTypes) (constraints : PurchaseConstraints) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem bookstore_problem :
  let types := MagazineTypes.mk 8 3
  let constraints := PurchaseConstraints.mk 10 1
  purchasingMethods types constraints = 266 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_problem_l2061_206146


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l2061_206139

theorem smallest_prime_12_less_than_square : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), Prime (k^2 - 12) ∧ k^2 - 12 > 0)) ∧ 
  Prime (n^2 - 12) ∧ 
  n^2 - 12 > 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l2061_206139


namespace NUMINAMATH_CALUDE_pizza_slice_ratio_l2061_206169

theorem pizza_slice_ratio : 
  ∀ (total_slices lunch_slices : ℕ),
    total_slices = 12 →
    lunch_slices ≤ total_slices →
    (total_slices - lunch_slices) / 3 + 4 = total_slices - lunch_slices →
    lunch_slices = total_slices / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_ratio_l2061_206169


namespace NUMINAMATH_CALUDE_apollo_chariot_wheels_l2061_206159

theorem apollo_chariot_wheels (months_in_year : ℕ) (total_cost : ℕ) 
  (initial_rate : ℕ) (h1 : months_in_year = 12) (h2 : total_cost = 54) 
  (h3 : initial_rate = 3) : 
  ∃ (x : ℕ), x * initial_rate + (months_in_year - x) * (2 * initial_rate) = total_cost ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_apollo_chariot_wheels_l2061_206159


namespace NUMINAMATH_CALUDE_guppies_theorem_l2061_206138

def guppies_problem (haylee jose charliz nicolai : ℕ) : Prop :=
  haylee = 3 * 12 ∧
  jose = haylee / 2 ∧
  charliz = jose / 3 ∧
  nicolai = 4 * charliz ∧
  haylee + jose + charliz + nicolai = 84

theorem guppies_theorem : ∃ haylee jose charliz nicolai : ℕ, guppies_problem haylee jose charliz nicolai :=
sorry

end NUMINAMATH_CALUDE_guppies_theorem_l2061_206138


namespace NUMINAMATH_CALUDE_solve_tangerines_l2061_206143

def tangerines_problem (initial_eaten : ℕ) (later_eaten : ℕ) : Prop :=
  ∃ (total : ℕ), 
    (initial_eaten + later_eaten = total) ∧ 
    (total - initial_eaten - later_eaten = 0)

theorem solve_tangerines : tangerines_problem 10 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_tangerines_l2061_206143


namespace NUMINAMATH_CALUDE_sum_difference_squares_l2061_206113

theorem sum_difference_squares (x y : ℝ) 
  (h1 : x > y) 
  (h2 : x + y = 10) 
  (h3 : x - y = 19) : 
  (x + y)^2 - (x - y)^2 = -261 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_squares_l2061_206113


namespace NUMINAMATH_CALUDE_find_a_l2061_206196

def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 4 * (2 * x + a)) ∧ (deriv (f a)) 2 = 20 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2061_206196


namespace NUMINAMATH_CALUDE_xyz_inequality_l2061_206114

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x*y + y*z + z*x = 1) : x*y*z*(x + y + z) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2061_206114


namespace NUMINAMATH_CALUDE_sum_of_ages_l2061_206156

/-- The sum of Eunji and Yuna's ages given their age difference -/
theorem sum_of_ages (eunji_age : ℕ) (age_difference : ℕ) : 
  eunji_age = 7 → age_difference = 5 → eunji_age + (eunji_age + age_difference) = 19 := by
  sorry

#check sum_of_ages

end NUMINAMATH_CALUDE_sum_of_ages_l2061_206156


namespace NUMINAMATH_CALUDE_ten_year_old_dog_human_years_l2061_206170

/-- Calculates the equivalent human years for a dog's age. -/
def dogYearsToHumanYears (dogAge : ℕ) : ℕ :=
  if dogAge = 0 then 0
  else if dogAge = 1 then 15
  else if dogAge = 2 then 24
  else 24 + 5 * (dogAge - 2)

/-- Theorem: A 10-year-old dog has lived 64 human years. -/
theorem ten_year_old_dog_human_years :
  dogYearsToHumanYears 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ten_year_old_dog_human_years_l2061_206170


namespace NUMINAMATH_CALUDE_calculation_proof_l2061_206149

theorem calculation_proof : (3.64 - 2.1) * 1.5 = 2.31 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2061_206149


namespace NUMINAMATH_CALUDE_min_box_height_l2061_206175

def box_height (x : ℝ) : ℝ := x + 5

def surface_area (x : ℝ) : ℝ := 6 * x^2 + 20 * x

def base_perimeter_plus_height (x : ℝ) : ℝ := 5 * x + 5

theorem min_box_height :
  ∀ x : ℝ,
  x > 0 →
  surface_area x ≥ 150 →
  base_perimeter_plus_height x ≥ 25 →
  box_height x ≥ 9 ∧
  (∃ y : ℝ, y > 0 ∧ surface_area y ≥ 150 ∧ base_perimeter_plus_height y ≥ 25 ∧ box_height y = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_box_height_l2061_206175


namespace NUMINAMATH_CALUDE_not_divisible_by_three_l2061_206151

theorem not_divisible_by_three (n : ℕ+) 
  (h : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n.val = k) :
  ¬(3 ∣ n.val) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_l2061_206151


namespace NUMINAMATH_CALUDE_food_shelf_life_l2061_206173

/-- The shelf life function for a food product -/
noncomputable def shelf_life (k b : ℝ) (x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the shelf life at 30°C and the maximum temperature for 80 hours shelf life -/
theorem food_shelf_life (k b : ℝ) :
  (shelf_life k b 0 = 160) →
  (shelf_life k b 20 = 40) →
  (shelf_life k b 30 = 20) ∧
  (∀ x : ℝ, shelf_life k b x ≥ 80 ↔ x ≤ 10) := by
  sorry


end NUMINAMATH_CALUDE_food_shelf_life_l2061_206173


namespace NUMINAMATH_CALUDE_cookie_boxes_theorem_l2061_206179

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (mark_sold ann_sold : ℕ),
    mark_sold = n - 8 ∧ 
    ann_sold = n - 2 ∧ 
    mark_sold ≥ 1 ∧ 
    ann_sold ≥ 1 ∧ 
    mark_sold + ann_sold < n) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_theorem_l2061_206179


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l2061_206184

theorem circle_equation_through_points :
  let general_circle_eq (x y D E F : ℝ) := x^2 + y^2 + D*x + E*y + F = 0
  let specific_circle_eq (x y : ℝ) := x^2 + y^2 - 4*x - 6*y = 0
  (∀ x y, general_circle_eq x y (-4) (-6) 0 ↔ specific_circle_eq x y) ∧
  specific_circle_eq 0 0 ∧
  specific_circle_eq 4 0 ∧
  specific_circle_eq (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l2061_206184


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2061_206182

theorem simplify_sqrt_expression :
  (2 * Real.sqrt 10) / (Real.sqrt 4 + Real.sqrt 3 + Real.sqrt 5) =
  (4 * Real.sqrt 10 - 15 * Real.sqrt 2) / 11 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2061_206182


namespace NUMINAMATH_CALUDE_annulus_area_l2061_206164

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  b : ℝ  -- radius of the larger circle
  c : ℝ  -- radius of the smaller circle
  h : b > c

/-- The configuration of the annulus with a tangent line. -/
structure AnnulusConfig extends Annulus where
  a : ℝ  -- length of the tangent line XZ
  d : ℝ  -- length of YZ
  e : ℝ  -- length of XY

/-- The area of an annulus is πa², where a is the length of a tangent line
    from a point on the smaller circle to the larger circle. -/
theorem annulus_area (config : AnnulusConfig) : 
  (config.b ^ 2 - config.c ^ 2) * π = config.a ^ 2 * π := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l2061_206164


namespace NUMINAMATH_CALUDE_distance_first_to_last_l2061_206148

-- Define the number of trees
def num_trees : ℕ := 8

-- Define the distance between first and fifth tree
def distance_1_to_5 : ℝ := 80

-- Theorem to prove
theorem distance_first_to_last :
  let distance_between_trees := distance_1_to_5 / 4
  let num_spaces := num_trees - 1
  distance_between_trees * num_spaces = 140 := by
sorry

end NUMINAMATH_CALUDE_distance_first_to_last_l2061_206148


namespace NUMINAMATH_CALUDE_train_length_calculation_l2061_206195

-- Define the given parameters
def train_speed : ℝ := 60  -- km/h
def man_speed : ℝ := 6     -- km/h
def passing_time : ℝ := 12 -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := train_speed + man_speed
  let relative_speed_mps : ℝ := relative_speed * (5 / 18)
  let train_length : ℝ := relative_speed_mps * passing_time
  train_length = 220 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2061_206195


namespace NUMINAMATH_CALUDE_divisible_by_35_l2061_206180

theorem divisible_by_35 (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_35_l2061_206180


namespace NUMINAMATH_CALUDE_triangle_side_length_l2061_206188

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S_ABC : ℝ) :
  a = 4 →
  B = π / 3 →
  S_ABC = 6 * Real.sqrt 3 →
  b = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2061_206188


namespace NUMINAMATH_CALUDE_fraction_power_five_l2061_206103

theorem fraction_power_five : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_five_l2061_206103


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l2061_206183

/-- The least 4-digit number divisible by 15, 25, 40, and 75 is 1200 -/
theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  15 ∣ n ∧ 25 ∣ n ∧ 40 ∣ n ∧ 75 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) ∧ 15 ∣ m ∧ 25 ∣ m ∧ 40 ∣ m ∧ 75 ∣ m → m ≥ n) ∧
  n = 1200 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l2061_206183


namespace NUMINAMATH_CALUDE_sum_square_free_coefficients_eight_l2061_206192

/-- A function that calculates the sum of coefficients of square-free terms -/
def sumSquareFreeCoefficients (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => sumSquareFreeCoefficients (k + 1) + (k + 1) * sumSquareFreeCoefficients k

/-- The main theorem stating that the sum of coefficients of square-free terms
    in the product of (1 + xᵢxⱼ) for 1 ≤ i < j ≤ 8 is 764 -/
theorem sum_square_free_coefficients_eight :
  sumSquareFreeCoefficients 8 = 764 := by
  sorry

/-- Helper lemma: The recurrence relation for sumSquareFreeCoefficients -/
lemma sum_square_free_coefficients_recurrence (n : ℕ) :
  n ≥ 2 →
  sumSquareFreeCoefficients n = 
    sumSquareFreeCoefficients (n-1) + (n-1) * sumSquareFreeCoefficients (n-2) := by
  sorry

end NUMINAMATH_CALUDE_sum_square_free_coefficients_eight_l2061_206192


namespace NUMINAMATH_CALUDE_smaller_factor_of_5610_l2061_206137

theorem smaller_factor_of_5610 (a b : Nat) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 5610 → 
  min a b = 34 := by
sorry

end NUMINAMATH_CALUDE_smaller_factor_of_5610_l2061_206137


namespace NUMINAMATH_CALUDE_range_of_f_l2061_206123

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 5

-- Define the domain
def domain : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 3 < y ∧ y < 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2061_206123


namespace NUMINAMATH_CALUDE_arithmetic_number_difference_l2061_206152

/-- A 4-digit number is arithmetic if its digits are distinct and form an arithmetic sequence. -/
def is_arithmetic (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a d : ℤ), 
    let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
    digits.map Int.ofNat = [a, a + d, a + 2*d, a + 3*d] ∧
    digits.toFinset.card = 4

/-- The largest arithmetic 4-digit number -/
def largest_arithmetic : ℕ := 9876

/-- The smallest arithmetic 4-digit number -/
def smallest_arithmetic : ℕ := 1234

theorem arithmetic_number_difference :
  is_arithmetic largest_arithmetic ∧
  is_arithmetic smallest_arithmetic ∧
  largest_arithmetic - smallest_arithmetic = 8642 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_number_difference_l2061_206152


namespace NUMINAMATH_CALUDE_farmer_seeds_total_l2061_206135

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := 20

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := 2

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_wednesday + seeds_thursday

theorem farmer_seeds_total :
  total_seeds = 22 :=
by sorry

end NUMINAMATH_CALUDE_farmer_seeds_total_l2061_206135


namespace NUMINAMATH_CALUDE_congruence_solution_l2061_206124

theorem congruence_solution (n : ℤ) : 13 * 26 ≡ 8 [ZMOD 47] := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2061_206124


namespace NUMINAMATH_CALUDE_gasoline_distribution_impossible_l2061_206168

theorem gasoline_distribution_impossible : 
  ¬ ∃ (A B C : ℝ), 
    A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧
    A + B + C = 50 ∧ 
    A = B + 10 ∧ 
    C + 26 = B :=
by sorry

end NUMINAMATH_CALUDE_gasoline_distribution_impossible_l2061_206168


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l2061_206165

/-- Given two vectors a and b in R², where a = (2,1) and b = (k,3),
    if a + 2b is parallel to 2a - b, then k = 6 -/
theorem vector_parallel_condition (k : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![k, 3]
  (∃ (t : ℝ), t ≠ 0 ∧ (a + 2 • b) = t • (2 • a - b)) →
  k = 6 :=
by sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l2061_206165


namespace NUMINAMATH_CALUDE_ab_value_l2061_206121

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧
    Real.sqrt (2 * log a) = m ∧
    Real.sqrt (2 * log b) = n ∧
    log (Real.sqrt a) = (m^2 : ℝ) / 4 ∧
    log (Real.sqrt b) = (n^2 : ℝ) / 4 ∧
    m + n + (m^2 : ℝ) / 4 + (n^2 : ℝ) / 4 = 104) →
  a * b = 10^260 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l2061_206121


namespace NUMINAMATH_CALUDE_compare_expressions_l2061_206162

theorem compare_expressions (x : ℝ) (h : x ≥ 0) :
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧
  (0 ≤ x ∧ x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_compare_expressions_l2061_206162


namespace NUMINAMATH_CALUDE_distance_to_tangent_line_l2061_206166

def curve (x : ℝ) : ℝ := 2*x - x^3

def tangent_point : ℝ × ℝ := (-1, curve (-1))

def tangent_slope : ℝ := 2 - 3*(-1)^2

def point_P : ℝ × ℝ := (3, 2)

theorem distance_to_tangent_line :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := 2
  (a * point_P.1 + b * point_P.2 + c) / Real.sqrt (a^2 + b^2) = 7 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_tangent_line_l2061_206166


namespace NUMINAMATH_CALUDE_flu_spreads_indefinitely_flu_stops_spreading_l2061_206190

-- Define the population as a finite type
variable {Population : Type} [Finite Population]

-- Define the state of a person
inductive State
  | Healthy
  | Infected
  | Immune

-- Define the friendship relation
variable (friends : Population → Population → Prop)

-- Define the state of the population on a given day
variable (state : ℕ → Population → State)

-- Define the condition that each person visits their friends daily
axiom daily_visits : ∀ (d : ℕ) (p q : Population), friends p q → True

-- Define the condition that healthy people become ill after visiting sick friends
axiom infection_spread : ∀ (d : ℕ) (p : Population), 
  state d p = State.Healthy → 
  (∃ (q : Population), friends p q ∧ state d q = State.Infected) → 
  state (d + 1) p = State.Infected

-- Define the condition that illness lasts one day, followed by immunity
axiom illness_duration : ∀ (d : ℕ) (p : Population),
  state d p = State.Infected → state (d + 1) p = State.Immune

-- Define the condition that immunity lasts at least one day
axiom immunity_duration : ∀ (d : ℕ) (p : Population),
  state d p = State.Immune → state (d + 1) p ≠ State.Infected

-- Theorem 1: If some people have immunity initially, the flu can spread indefinitely
theorem flu_spreads_indefinitely (h : ∃ (p : Population), state 0 p = State.Immune) :
  ∀ (n : ℕ), ∃ (d : ℕ) (p : Population), d ≥ n ∧ state d p = State.Infected :=
sorry

-- Theorem 2: If no one has immunity initially, the flu will eventually stop spreading
theorem flu_stops_spreading (h : ∀ (p : Population), state 0 p ≠ State.Immune) :
  ∃ (n : ℕ), ∀ (d : ℕ) (p : Population), d ≥ n → state d p ≠ State.Infected :=
sorry

end NUMINAMATH_CALUDE_flu_spreads_indefinitely_flu_stops_spreading_l2061_206190


namespace NUMINAMATH_CALUDE_triangle_area_tangent_circles_l2061_206160

/-- Given two non-overlapping circles with radii r₁ and r₂, where one common internal tangent
    is perpendicular to one common external tangent, the area S of the triangle formed by
    these tangents and the third common tangent satisfies one of two formulas. -/
theorem triangle_area_tangent_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₁ ≠ r₂) :
  ∃ S : ℝ, (S = (r₁ * r₂ * (r₁ + r₂)) / |r₁ - r₂|) ∨ (S = (r₁ * r₂ * |r₁ - r₂|) / (r₁ + r₂)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_tangent_circles_l2061_206160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l2061_206157

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_2017 :
  arithmetic_sequence 4 3 672 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l2061_206157


namespace NUMINAMATH_CALUDE_p_and_q_iff_a_in_range_l2061_206108

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x, x^2 + 2*a*x + a + 2 = 0

def q (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- State the theorem
theorem p_and_q_iff_a_in_range (a : ℝ) : 
  (p a ∧ q a) ↔ a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_p_and_q_iff_a_in_range_l2061_206108


namespace NUMINAMATH_CALUDE_inscribed_circles_area_limit_l2061_206119

/-- Represents the sum of areas of the first n inscribed circles -/
def S (n : ℕ) (a : ℝ) : ℝ := sorry

/-- The limit of S_n as n approaches infinity -/
def S_limit (a : ℝ) : ℝ := sorry

theorem inscribed_circles_area_limit (a b : ℝ) (h : 0 < a ∧ a ≤ b) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n a - S_limit a| < ε ∧ S_limit a = (π * a^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_limit_l2061_206119


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2061_206102

theorem quadratic_root_value (c : ℚ) : 
  (∀ x : ℚ, (3/2 * x^2 + 13*x + c = 0) ↔ (x = (-13 + Real.sqrt 23)/3 ∨ x = (-13 - Real.sqrt 23)/3)) →
  c = 146/6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2061_206102


namespace NUMINAMATH_CALUDE_bananas_per_box_l2061_206189

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 10) :
  total_bananas / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l2061_206189


namespace NUMINAMATH_CALUDE_log_inequality_l2061_206185

theorem log_inequality (h1 : 4^5 < 7^4) (h2 : 11^4 < 7^5) : 
  Real.log 11 / Real.log 7 < Real.log 243 / Real.log 81 ∧ 
  Real.log 243 / Real.log 81 < Real.log 7 / Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2061_206185
