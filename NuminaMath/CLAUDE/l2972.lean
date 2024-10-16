import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2972_297251

theorem complex_fraction_simplification :
  let z : ℂ := (3 - 2*I) / (5 - 2*I)
  z = 19/29 - (4/29)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2972_297251


namespace NUMINAMATH_CALUDE_tan_product_30_15_l2972_297276

theorem tan_product_30_15 :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_30_15_l2972_297276


namespace NUMINAMATH_CALUDE_expression_factorization_l2972_297223

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l2972_297223


namespace NUMINAMATH_CALUDE_quadratic_coefficients_divisible_by_three_l2972_297233

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ := λ x ↦ a * x^2 + b * x + c

/-- The property that a polynomial is divisible by 3 for all integer inputs -/
def DivisibleByThreeForAllIntegers (P : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, P x = 3 * k

theorem quadratic_coefficients_divisible_by_three
  (a b c : ℤ)
  (h : DivisibleByThreeForAllIntegers (QuadraticPolynomial a b c)) :
  (∃ k₁ k₂ k₃ : ℤ, a = 3 * k₁ ∧ b = 3 * k₂ ∧ c = 3 * k₃) :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_divisible_by_three_l2972_297233


namespace NUMINAMATH_CALUDE_min_value_sum_l2972_297254

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ (m : ℝ), m = 10 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x * y = 1 → 
    (x + 1/x) + (y + 1/y) ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l2972_297254


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_18_l2972_297266

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_45_and_60_not_18 : 
  (∀ n : ℕ, n < 810 → (is_multiple n 45 ∧ is_multiple n 60) → is_multiple n 18) ∧ 
  is_multiple 810 45 ∧ 
  is_multiple 810 60 ∧ 
  ¬is_multiple 810 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_18_l2972_297266


namespace NUMINAMATH_CALUDE_lowest_price_scheme_l2972_297241

-- Define the pricing schemes
def schemeA (price : ℝ) : ℝ := price * 1.1 * 0.9
def schemeB (price : ℝ) : ℝ := price * 0.9 * 1.1
def schemeC (price : ℝ) : ℝ := price * 1.15 * 0.85
def schemeD (price : ℝ) : ℝ := price * 1.2 * 0.8

-- Theorem statement
theorem lowest_price_scheme (price : ℝ) (h : price > 0) :
  schemeD price = min (schemeA price) (min (schemeB price) (schemeC price)) :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_scheme_l2972_297241


namespace NUMINAMATH_CALUDE_only_paintable_integer_l2972_297281

/-- Represents a painting pattern for the fence. -/
structure PaintingPattern where
  start : ℕ
  interval : ℕ

/-- Checks if a given triple (h, t, u) results in a valid painting pattern. -/
def isValidPainting (h t u : ℕ) : Prop :=
  let harold := PaintingPattern.mk 4 h
  let tanya := PaintingPattern.mk 5 (2 * t)
  let ulysses := PaintingPattern.mk 6 (3 * u)
  ∀ n : ℕ, n ≥ 1 →
    (∃! painter, painter ∈ [harold, tanya, ulysses] ∧
      ∃ k, n = painter.start + painter.interval * k)

/-- Calculates the paintable integer for a given triple (h, t, u). -/
def paintableInteger (h t u : ℕ) : ℕ :=
  100 * h + 20 * t + 2 * u

/-- The main theorem stating that 390 is the only paintable integer. -/
theorem only_paintable_integer :
  ∀ h t u : ℕ, h > 0 ∧ t > 0 ∧ u > 0 →
    isValidPainting h t u ↔ paintableInteger h t u = 390 :=
sorry

end NUMINAMATH_CALUDE_only_paintable_integer_l2972_297281


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2972_297232

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_sum_of_squares 
  (a b : ℝ) 
  (h : (a - 2 * i) * i = b - i) : 
  a^2 + b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2972_297232


namespace NUMINAMATH_CALUDE_combined_area_square_triangle_l2972_297294

/-- The combined area of a square with diagonal 30 m and an equilateral triangle sharing that diagonal as its side is 450 m² + 225√3 m². -/
theorem combined_area_square_triangle (diagonal : ℝ) (h_diagonal : diagonal = 30) :
  let square_side := diagonal / Real.sqrt 2
  let square_area := square_side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * diagonal ^ 2
  square_area + triangle_area = 450 + 225 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_combined_area_square_triangle_l2972_297294


namespace NUMINAMATH_CALUDE_angle_between_polar_lines_eq_arctan_half_l2972_297202

/-- The angle between two lines in polar coordinates -/
def angle_between_polar_lines (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- First line equation in polar coordinates -/
def line1 (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + 2 * Real.sin θ) = 1

/-- Second line equation in polar coordinates -/
def line2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = 1

/-- Theorem stating the angle between the two given lines -/
theorem angle_between_polar_lines_eq_arctan_half :
  angle_between_polar_lines line1 line2 = Real.arctan (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_angle_between_polar_lines_eq_arctan_half_l2972_297202


namespace NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l2972_297230

def count_quadruplets (n : ℕ) : ℕ :=
  sorry

theorem smallest_n_for_quadruplets : 
  (∃ (n : ℕ), 
    n > 0 ∧ 
    count_quadruplets n = 50000 ∧
    (∀ (a b c d : ℕ), 
      (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = n) → 
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) ∧
    (∀ (m : ℕ), m < n → 
      (count_quadruplets m ≠ 50000 ∨
       ∃ (a b c d : ℕ), 
         (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
          Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m) ∧
         (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0)))) ∧
  (∀ (n : ℕ), 
    n > 0 ∧ 
    count_quadruplets n = 50000 ∧
    (∀ (a b c d : ℕ), 
      (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = n) → 
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) →
    n ≥ 4459000) ∧
  count_quadruplets 4459000 = 50000 ∧
  (∀ (a b c d : ℕ), 
    (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
     Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 4459000) → 
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l2972_297230


namespace NUMINAMATH_CALUDE_tan_sum_pi_fourth_l2972_297209

theorem tan_sum_pi_fourth (θ : Real) (h : Real.tan θ = 1/3) : 
  Real.tan (θ + π/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_fourth_l2972_297209


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2972_297286

theorem complex_fraction_sum : 
  let S := 1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
           1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)
  S = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2972_297286


namespace NUMINAMATH_CALUDE_marilyn_final_bottle_caps_l2972_297257

/-- Calculates the final number of bottle caps Marilyn has after a series of exchanges --/
def final_bottle_caps (initial : ℕ) (shared : ℕ) (received : ℕ) : ℕ :=
  let remaining := initial - shared + received
  remaining - remaining / 2

/-- Theorem stating that Marilyn ends up with 55 bottle caps --/
theorem marilyn_final_bottle_caps : 
  final_bottle_caps 165 78 23 = 55 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_final_bottle_caps_l2972_297257


namespace NUMINAMATH_CALUDE_remaining_distance_to_hotel_l2972_297262

/-- Calculates the remaining distance to the hotel given Samuel's journey conditions --/
theorem remaining_distance_to_hotel : 
  let total_distance : ℕ := 600
  let speed1 : ℕ := 50
  let time1 : ℕ := 3
  let speed2 : ℕ := 80
  let time2 : ℕ := 4
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let traveled_distance := distance1 + distance2
  total_distance - traveled_distance = 130 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_hotel_l2972_297262


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2972_297280

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 0.56 ∧ x = 56 / 99 :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2972_297280


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2972_297287

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x => a^(x - 3) + 3
  f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2972_297287


namespace NUMINAMATH_CALUDE_geography_letter_collections_l2972_297295

/-- Represents the number of ways to choose vowels from GEOGRAPHY -/
def vowel_choices : ℕ := 4

/-- Represents the number of ways to choose consonants from GEOGRAPHY -/
def consonant_choices : ℕ := 17

/-- Represents the total number of distinct letter collections -/
def total_distinct_collections : ℕ := vowel_choices * consonant_choices

/-- Theorem stating that the number of distinct possible collections of letters from GEOGRAPHY,
    where two vowels and three consonants are randomly selected, and G's, A's, and H's are
    indistinguishable, is equal to 68 -/
theorem geography_letter_collections :
  total_distinct_collections = 68 := by
  sorry

end NUMINAMATH_CALUDE_geography_letter_collections_l2972_297295


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l2972_297258

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l2972_297258


namespace NUMINAMATH_CALUDE_kaleb_chocolate_pieces_l2972_297249

theorem kaleb_chocolate_pieces (initial_boxes : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → boxes_given = 5 → pieces_per_box = 6 →
  (initial_boxes - boxes_given) * pieces_per_box = 54 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_chocolate_pieces_l2972_297249


namespace NUMINAMATH_CALUDE_partnership_investment_l2972_297285

/-- A partnership business problem -/
theorem partnership_investment (b_investment c_investment c_profit total_profit : ℕ) 
  (hb : b_investment = 72000)
  (hc : c_investment = 81000)
  (hcp : c_profit = 36000)
  (htp : total_profit = 80000) :
  ∃ a_investment : ℕ, 
    (c_profit : ℚ) / (total_profit : ℚ) = (c_investment : ℚ) / ((a_investment : ℚ) + (b_investment : ℚ) + (c_investment : ℚ)) ∧ 
    a_investment = 27000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l2972_297285


namespace NUMINAMATH_CALUDE_product_of_max_min_elements_l2972_297229

def S : Set ℝ := {z | ∃ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 5 ∧ z = 5/x + y}

theorem product_of_max_min_elements (M N : ℝ) : 
  (∀ z ∈ S, z ≤ M) ∧ (M ∈ S) ∧ (∀ z ∈ S, N ≤ z) ∧ (N ∈ S) →
  M * N = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_max_min_elements_l2972_297229


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l2972_297282

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - x + 4

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x ≥ -1/2 → y > x → f y < f x :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l2972_297282


namespace NUMINAMATH_CALUDE_difference_of_squares_l2972_297201

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2972_297201


namespace NUMINAMATH_CALUDE_two_number_problem_l2972_297219

-- Define the types for logicians and their knowledge states
inductive Logician | A | B
inductive Knowledge | Known | Unknown

-- Define a function to represent the state of knowledge after each exchange
def knowledge_state (exchange : ℕ) (l : Logician) : Knowledge := sorry

-- Define the conditions for the sum and sum of squares
def sum_condition (u v : ℕ) : Prop := u + v = 17

def sum_of_squares_condition (u v : ℕ) : Prop := u^2 + v^2 = 145

-- Define the main theorem
theorem two_number_problem (u v : ℕ) :
  u > 0 ∧ v > 0 ∧
  sum_condition u v ∧
  sum_of_squares_condition u v ∧
  (∀ e, e < 6 → knowledge_state e Logician.A = Knowledge.Unknown) ∧
  (∀ e, e < 6 → knowledge_state e Logician.B = Knowledge.Unknown) ∧
  knowledge_state 6 Logician.B = Knowledge.Known
  → (u = 8 ∧ v = 9) ∨ (u = 9 ∧ v = 8) := by sorry

end NUMINAMATH_CALUDE_two_number_problem_l2972_297219


namespace NUMINAMATH_CALUDE_broken_eggs_count_l2972_297270

/-- Given a total of 24 eggs, where some are broken, some are cracked, and some are perfect,
    prove that the number of broken eggs is 3 under the following conditions:
    1. The number of cracked eggs is twice the number of broken eggs
    2. The difference between perfect and cracked eggs is 9 -/
theorem broken_eggs_count (broken : ℕ) (cracked : ℕ) (perfect : ℕ) : 
  perfect + cracked + broken = 24 →
  cracked = 2 * broken →
  perfect - cracked = 9 →
  broken = 3 := by sorry

end NUMINAMATH_CALUDE_broken_eggs_count_l2972_297270


namespace NUMINAMATH_CALUDE_circle_area_with_radius_four_l2972_297234

theorem circle_area_with_radius_four (π : ℝ) : 
  let r : ℝ := 4
  let area := π * r^2
  area = 16 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_radius_four_l2972_297234


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2972_297264

/-- Given a circle with area π/4 square units, its diameter is 1 unit. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), π * r^2 = π / 4 → 2 * r = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2972_297264


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2972_297248

/-- An arithmetic sequence with given second and eighth terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_proof
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = -6)
  (h_a8 : a 8 = -18) :
  (∃ d : ℤ, d = -2 ∧ ∀ n : ℕ, a n = -2 * n - 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2972_297248


namespace NUMINAMATH_CALUDE_closest_ratio_is_one_l2972_297227

/-- Represents the admission fee structure and total collection -/
structure AdmissionData where
  adult_fee : ℕ
  child_fee : ℕ
  total_collected : ℕ

/-- Finds the ratio of adults to children closest to 1 given admission data -/
def closest_ratio_to_one (data : AdmissionData) : ℚ :=
  sorry

/-- The main theorem stating that the closest ratio to 1 is exactly 1 for the given data -/
theorem closest_ratio_is_one :
  let data : AdmissionData := {
    adult_fee := 30,
    child_fee := 15,
    total_collected := 2700
  }
  closest_ratio_to_one data = 1 := by sorry

end NUMINAMATH_CALUDE_closest_ratio_is_one_l2972_297227


namespace NUMINAMATH_CALUDE_university_applications_l2972_297216

theorem university_applications (n m k : ℕ) (h1 : n = 7) (h2 : m = 2) (h3 : k = 4) : 
  (Nat.choose (n - m + 1) k) + (Nat.choose m 1 * Nat.choose (n - m) (k - 1)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_university_applications_l2972_297216


namespace NUMINAMATH_CALUDE_loaves_sold_is_one_l2972_297215

/-- Represents the baker's sales and prices --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  pastry_price : ℚ
  bread_price : ℚ
  sales_difference : ℚ

/-- Calculates the number of loaves of bread sold today --/
def loaves_sold_today (s : BakerSales) : ℚ :=
  ((s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price) -
   (s.today_pastries * s.pastry_price + s.sales_difference)) / s.bread_price

/-- Theorem stating that the number of loaves sold today is 1 --/
theorem loaves_sold_is_one (s : BakerSales)
  (h1 : s.usual_pastries = 20)
  (h2 : s.usual_bread = 10)
  (h3 : s.today_pastries = 14)
  (h4 : s.pastry_price = 2)
  (h5 : s.bread_price = 4)
  (h6 : s.sales_difference = 48) :
  loaves_sold_today s = 1 := by
  sorry

#eval loaves_sold_today {
  usual_pastries := 20,
  usual_bread := 10,
  today_pastries := 14,
  pastry_price := 2,
  bread_price := 4,
  sales_difference := 48
}

end NUMINAMATH_CALUDE_loaves_sold_is_one_l2972_297215


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2972_297218

/-- The time taken for a train to completely cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed : ℝ) 
  (h1 : train_length = 400) 
  (h2 : bridge_length = 300) 
  (h3 : train_speed = 55.99999999999999) : 
  (train_length + bridge_length) / train_speed = (400 + 300) / 55.99999999999999 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2972_297218


namespace NUMINAMATH_CALUDE_perfect_square_solutions_l2972_297274

theorem perfect_square_solutions : 
  {n : ℤ | ∃ m : ℤ, n^2 + 6*n + 24 = m^2} = {4, -2, -4, -10} := by sorry

end NUMINAMATH_CALUDE_perfect_square_solutions_l2972_297274


namespace NUMINAMATH_CALUDE_average_apples_per_guest_l2972_297239

/-- Represents the number of apples per serving -/
def apples_per_serving : ℚ := 3/2

/-- Represents the ratio of Red Delicious to Granny Smith apples per serving -/
def apple_ratio : ℚ := 2

/-- Represents the number of guests -/
def num_guests : ℕ := 12

/-- Represents the number of pies -/
def num_pies : ℕ := 3

/-- Represents the number of servings per pie -/
def servings_per_pie : ℕ := 8

/-- Represents the number of cups of apple pieces per Red Delicious apple -/
def red_delicious_cups : ℚ := 1

/-- Represents the number of cups of apple pieces per Granny Smith apple -/
def granny_smith_cups : ℚ := 5/4

/-- Theorem stating that the average number of apples each guest eats is 3 -/
theorem average_apples_per_guest : 
  (num_pies * servings_per_pie * apples_per_serving) / num_guests = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_per_guest_l2972_297239


namespace NUMINAMATH_CALUDE_compute_expression_l2972_297226

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2972_297226


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l2972_297245

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 4 / 3)
  (h3 : s / q = 1 / 8) :
  p / r = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l2972_297245


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2972_297252

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * c - b) / Real.cos B = a / Real.cos A ∧
  a = Real.sqrt 7 ∧
  2 * b = 3 * c

theorem triangle_ABC_properties {a b c A B C : ℝ} 
  (h : triangle_ABC a b c A B C) :
  A = π / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2972_297252


namespace NUMINAMATH_CALUDE_oil_leak_before_fixing_l2972_297255

theorem oil_leak_before_fixing (total_leak : ℕ) (leak_during_work : ℕ) 
  (h1 : total_leak = 11687)
  (h2 : leak_during_work = 5165) :
  total_leak - leak_during_work = 6522 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_before_fixing_l2972_297255


namespace NUMINAMATH_CALUDE_largest_subarray_sum_l2972_297273

/-- A type representing a 5x5 array of natural numbers -/
def Array5x5 := Fin 5 → Fin 5 → ℕ

/-- Predicate to check if an array contains distinct numbers from 1 to 25 -/
def isValidArray (a : Array5x5) : Prop :=
  ∀ i j, 1 ≤ a i j ∧ a i j ≤ 25 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → a i j ≠ a i' j'

/-- Sum of a 2x2 subarray starting at position (i, j) -/
def subarraySum (a : Array5x5) (i j : Fin 4) : ℕ :=
  a i j + a i (j + 1) + a (i + 1) j + a (i + 1) (j + 1)

/-- Theorem stating that 45 is the largest N satisfying the given property -/
theorem largest_subarray_sum : 
  (∀ a : Array5x5, isValidArray a → ∀ i j : Fin 4, subarraySum a i j ≥ 45) ∧
  ¬(∀ a : Array5x5, isValidArray a → ∀ i j : Fin 4, subarraySum a i j ≥ 46) :=
sorry

end NUMINAMATH_CALUDE_largest_subarray_sum_l2972_297273


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l2972_297205

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l2972_297205


namespace NUMINAMATH_CALUDE_fraction_equality_l2972_297208

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 1) :
  (a + a*b - b) / (a - 2*a*b - b) = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2972_297208


namespace NUMINAMATH_CALUDE_infinitely_many_composite_sums_l2972_297247

theorem infinitely_many_composite_sums : 
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ 
  ∀ (k n : ℕ), ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ n^4 + (f k)^4 = x * y :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_sums_l2972_297247


namespace NUMINAMATH_CALUDE_curve_equation_and_min_distance_l2972_297235

/-- The curve C defined by points P(x,y) satisfying |PA| = 2|PB| where A(-3,0) and B(3,0) -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 3)^2 + p.2^2 = 4 * ((p.1 - 3)^2 + p.2^2)}

/-- The line l1 defined by x + y + 3 = 0 -/
def l1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 3 = 0}

theorem curve_equation_and_min_distance :
  /- The equation of curve C is (x-5)^2 + y^2 = 16 -/
  (C = {p : ℝ × ℝ | (p.1 - 5)^2 + p.2^2 = 16}) ∧
  /- The minimum distance from any point on l1 to C is 4 -/
  (∀ Q ∈ l1, ∃ M ∈ C, ∀ P ∈ C, dist Q P ≥ dist Q M ∧ dist Q M ≥ 4) ∧
  (∃ Q ∈ l1, ∃ M ∈ C, dist Q M = 4) :=
sorry

end NUMINAMATH_CALUDE_curve_equation_and_min_distance_l2972_297235


namespace NUMINAMATH_CALUDE_contractor_fine_proof_l2972_297278

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine_per_day (total_days : ℕ) (pay_per_day : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let total_earned := pay_per_day * worked_days
  (total_earned - total_payment) / absent_days

/-- Proves that the fine per day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_proof :
  calculate_fine_per_day 30 25 425 10 = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_contractor_fine_proof_l2972_297278


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l2972_297283

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ x y : ℝ, y = a * x - 2) ∧ 
  (∃ x y : ℝ, y = 2 * x + 1) ∧ 
  (a * 2 = -1) → 
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l2972_297283


namespace NUMINAMATH_CALUDE_fraction_scaling_l2972_297213

theorem fraction_scaling (a b : ℝ) :
  (2*a + 2*b) / ((2*a)^2 + (2*b)^2) = (1/2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_scaling_l2972_297213


namespace NUMINAMATH_CALUDE_correct_plate_set_is_valid_l2972_297220

/-- Represents a plate with a count of bacteria -/
structure Plate where
  count : ℕ
  deriving Repr

/-- Represents a set of plates used in the dilution spread plate method -/
structure PlateSet where
  plates : List Plate
  dilutionFactor : ℕ
  deriving Repr

/-- Checks if a plate count is valid (between 30 and 300) -/
def isValidCount (count : ℕ) : Bool :=
  30 ≤ count ∧ count ≤ 300

/-- Checks if a plate set is valid for the dilution spread plate method -/
def isValidPlateSet (ps : PlateSet) : Bool :=
  ps.plates.length ≥ 3 ∧ 
  ps.plates.all (fun p => isValidCount p.count) ∧
  ps.dilutionFactor = 10^6

/-- Calculates the average count of a plate set -/
def averageCount (ps : PlateSet) : ℚ :=
  let total : ℚ := ps.plates.foldl (fun acc p => acc + p.count) 0
  total / ps.plates.length

/-- The correct plate set for the problem -/
def correctPlateSet : PlateSet :=
  { plates := [⟨210⟩, ⟨240⟩, ⟨250⟩],
    dilutionFactor := 10^6 }

theorem correct_plate_set_is_valid :
  isValidPlateSet correctPlateSet ∧ 
  averageCount correctPlateSet = 233 :=
sorry

end NUMINAMATH_CALUDE_correct_plate_set_is_valid_l2972_297220


namespace NUMINAMATH_CALUDE_max_integer_difference_l2972_297212

theorem max_integer_difference (x y : ℝ) (hx : 6 < x ∧ x < 10) (hy : 10 < y ∧ y < 17) :
  (⌊y⌋ : ℤ) - (⌈x⌉ : ℤ) ≤ 9 ∧ ∃ (x₀ y₀ : ℝ), 6 < x₀ ∧ x₀ < 10 ∧ 10 < y₀ ∧ y₀ < 17 ∧ (⌊y₀⌋ : ℤ) - (⌈x₀⌉ : ℤ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l2972_297212


namespace NUMINAMATH_CALUDE_unique_solution_system_l2972_297296

theorem unique_solution_system (x y z : ℝ) : 
  x < y → y < z → z < 6 →
  1 / (y - x) + 1 / (z - y) ≤ 2 →
  1 / (6 - z) + 2 ≤ x →
  x = 3 ∧ y = 4 ∧ z = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2972_297296


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2972_297237

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ  -- Length of equal sides
  base : ℕ  -- Length of the base
  isValid : 2 * side > base  -- Triangle inequality

/-- Check if two triangles have the same perimeter -/
def samePerimeter (t1 t2 : IsoscelesTriangle) : Prop :=
  2 * t1.side + t1.base = 2 * t2.side + t2.base

/-- Check if two triangles have the same area -/
def sameArea (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base * (t1.side ^ 2 - (t1.base / 2) ^ 2).sqrt = 
  t2.base * (t2.side ^ 2 - (t2.base / 2) ^ 2).sqrt

/-- Check if the base ratio of two triangles is 5:4 -/
def baseRatio54 (t1 t2 : IsoscelesTriangle) : Prop :=
  5 * t2.base = 4 * t1.base

/-- The main theorem -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    samePerimeter t1 t2 ∧
    sameArea t1 t2 ∧
    baseRatio54 t1 t2 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      samePerimeter s1 s2 →
      sameArea s1 s2 →
      baseRatio54 s1 s2 →
      2 * t1.side + t1.base ≤ 2 * s1.side + s1.base) ∧
    2 * t1.side + t1.base = 138 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2972_297237


namespace NUMINAMATH_CALUDE_additional_bottles_needed_l2972_297217

def medium_bottle_capacity : ℕ := 50
def giant_bottle_capacity : ℕ := 750
def bottles_already_owned : ℕ := 3

theorem additional_bottles_needed : 
  (giant_bottle_capacity / medium_bottle_capacity) - bottles_already_owned = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_bottles_needed_l2972_297217


namespace NUMINAMATH_CALUDE_comb_cost_is_one_l2972_297260

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℝ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℝ := 1

/-- Kristine's total purchase cost in dollars -/
def kristine_cost : ℝ := barrette_cost + comb_cost

/-- Crystal's total purchase cost in dollars -/
def crystal_cost : ℝ := 3 * barrette_cost + comb_cost

/-- The total amount spent by both girls in dollars -/
def total_spent : ℝ := 14

theorem comb_cost_is_one :
  kristine_cost + crystal_cost = total_spent → comb_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_comb_cost_is_one_l2972_297260


namespace NUMINAMATH_CALUDE_fourteen_in_base_two_l2972_297225

theorem fourteen_in_base_two : 
  (14 : ℕ).digits 2 = [0, 1, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_fourteen_in_base_two_l2972_297225


namespace NUMINAMATH_CALUDE_geometric_sum_eight_terms_l2972_297279

theorem geometric_sum_eight_terms :
  let a₀ : ℚ := 2/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 6560/6561 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_eight_terms_l2972_297279


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2972_297207

theorem quadratic_real_roots (n : ℕ+) :
  (∃ x : ℝ, x^2 - 4*x + n.val = 0) ↔ n.val ∈ ({1, 2, 3, 4} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2972_297207


namespace NUMINAMATH_CALUDE_find_n_l2972_297200

theorem find_n : ∃ n : ℤ, (11 : ℝ) ^ (4 * n) = (1 / 11 : ℝ) ^ (n - 30) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2972_297200


namespace NUMINAMATH_CALUDE_point_C_values_l2972_297204

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the number line with three points -/
structure NumberLine where
  A : Point
  B : Point
  C : Point

/-- Checks if folding at one point makes the other two coincide -/
def foldingCondition (line : NumberLine) : Prop :=
  (abs (line.A.value - line.B.value) = 2 * abs (line.A.value - line.C.value)) ∨
  (abs (line.A.value - line.B.value) = 2 * abs (line.B.value - line.C.value)) ∨
  (abs (line.A.value - line.C.value) = abs (line.B.value - line.C.value))

/-- The main theorem to prove -/
theorem point_C_values (line : NumberLine) :
  ((line.A.value + 3)^2 + abs (line.B.value - 1) = 0) →
  foldingCondition line →
  (line.C.value = -7 ∨ line.C.value = -1 ∨ line.C.value = 5) := by
  sorry

end NUMINAMATH_CALUDE_point_C_values_l2972_297204


namespace NUMINAMATH_CALUDE_lists_count_l2972_297265

/-- The number of distinct items to choose from -/
def n : ℕ := 15

/-- The number of times we draw an item -/
def k : ℕ := 4

/-- The number of possible lists when drawing with replacement -/
def num_lists : ℕ := n^k

theorem lists_count : num_lists = 50625 := by
  sorry

end NUMINAMATH_CALUDE_lists_count_l2972_297265


namespace NUMINAMATH_CALUDE_calculate_expression_l2972_297297

theorem calculate_expression : 
  2 / (-1/4) - |(-Real.sqrt 18)| + (1/5)⁻¹ = -3 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2972_297297


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2972_297206

/-- The constant term in the expansion of (√5/5 * x^2 + 1/x)^6 is 3 -/
theorem constant_term_binomial_expansion :
  let a := Real.sqrt 5 / 5
  let b := 1
  let n := 6
  let k := 4  -- The value of k where x^(2n-3k) = x^0
  (Nat.choose n k) * a^(n-k) * b^k = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2972_297206


namespace NUMINAMATH_CALUDE_sequence_sum_equals_exp_l2972_297250

/-- Given a positive integer m, y_k is a sequence defined by:
    y_0 = 1
    y_1 = m
    y_{k+2} = ((m+1)y_{k+1} - (m-k)y_k) / (k+1) for k ≥ 0
    This theorem states that the sum of all terms in the sequence equals e^(m+1) -/
theorem sequence_sum_equals_exp (m : ℕ+) : ∃ (y : ℕ → ℝ), 
  y 0 = 1 ∧ 
  y 1 = m ∧ 
  (∀ k : ℕ, y (k + 2) = ((m + 1 : ℝ) * y (k + 1) - (m - k) * y k) / (k + 1)) ∧
  (∑' k, y k) = Real.exp (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_exp_l2972_297250


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2972_297243

-- Define S(k) as the sum of digits of k in decimal representation
def S (k : ℕ) : ℕ := sorry

-- Define the property that P(x) must satisfy
def satisfies_property (P : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2016 → (S (P n) = P (S n) ∧ P n > 0)

-- Define the set of valid polynomials
def valid_polynomial (P : ℕ → ℕ) : Prop :=
  (∃ c : ℕ, c ≥ 1 ∧ c ≤ 9 ∧ (∀ x : ℕ, P x = c)) ∨
  (∀ x : ℕ, P x = x)

-- Theorem statement
theorem polynomial_characterization :
  ∀ P : ℕ → ℕ, satisfies_property P → valid_polynomial P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2972_297243


namespace NUMINAMATH_CALUDE_cannot_form_square_l2972_297244

/-- The number of sticks of length 1 cm -/
def sticks_1cm : ℕ := 6

/-- The number of sticks of length 2 cm -/
def sticks_2cm : ℕ := 3

/-- The number of sticks of length 3 cm -/
def sticks_3cm : ℕ := 6

/-- The number of sticks of length 4 cm -/
def sticks_4cm : ℕ := 5

/-- The total length of all sticks in cm -/
def total_length : ℕ := sticks_1cm * 1 + sticks_2cm * 2 + sticks_3cm * 3 + sticks_4cm * 4

/-- Theorem stating that it's impossible to form a square with the given sticks -/
theorem cannot_form_square : ¬(total_length % 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_square_l2972_297244


namespace NUMINAMATH_CALUDE_length_of_GH_l2972_297224

/-- Given four squares arranged in a specific formation, prove that the length of segment GH is 29 -/
theorem length_of_GH (square_A square_C square_E square_smallest : ℝ) 
  (h1 : square_A - square_C = 11)
  (h2 : square_C - square_E = 5)
  (h3 : square_E - square_smallest = 13) :
  square_A - square_smallest = 29 := by
  sorry

end NUMINAMATH_CALUDE_length_of_GH_l2972_297224


namespace NUMINAMATH_CALUDE_four_black_faces_symmetry_l2972_297238

/-- Represents the symmetry types of a cube. -/
inductive CubeSymmetryType
  | A
  | B1
  | B2
  | C

/-- Represents a cube with some faces painted black. -/
structure PaintedCube where
  blackFaces : Finset (Fin 6)
  blackFaceCount : blackFaces.card = 4

/-- Returns the symmetry type of a painted cube. -/
def symmetryType (cube : PaintedCube) : CubeSymmetryType :=
  sorry

/-- Theorem stating that a cube with four black faces has a symmetry type equivalent to B1 or B2. -/
theorem four_black_faces_symmetry (cube : PaintedCube) :
  symmetryType cube = CubeSymmetryType.B1 ∨ symmetryType cube = CubeSymmetryType.B2 :=
sorry

end NUMINAMATH_CALUDE_four_black_faces_symmetry_l2972_297238


namespace NUMINAMATH_CALUDE_rice_distribution_l2972_297221

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 33 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 15 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l2972_297221


namespace NUMINAMATH_CALUDE_corn_planting_bags_used_l2972_297271

/-- Represents the corn planting scenario with given conditions -/
structure CornPlanting where
  kids : ℕ
  earsPerRow : ℕ
  seedsPerEar : ℕ
  seedsPerBag : ℕ
  payPerRow : ℚ
  dinnerCost : ℚ

/-- Calculates the number of bags of corn seeds used by each kid -/
def bagsUsedPerKid (cp : CornPlanting) : ℚ :=
  let totalEarned := 2 * cp.dinnerCost
  let rowsPlanted := totalEarned / cp.payPerRow
  let seedsPerRow := cp.earsPerRow * cp.seedsPerEar
  let totalSeeds := rowsPlanted * seedsPerRow
  totalSeeds / cp.seedsPerBag

/-- Theorem stating that each kid used 140 bags of corn seeds -/
theorem corn_planting_bags_used
  (cp : CornPlanting)
  (h1 : cp.kids = 4)
  (h2 : cp.earsPerRow = 70)
  (h3 : cp.seedsPerEar = 2)
  (h4 : cp.seedsPerBag = 48)
  (h5 : cp.payPerRow = 3/2)
  (h6 : cp.dinnerCost = 36) :
  bagsUsedPerKid cp = 140 := by
  sorry

end NUMINAMATH_CALUDE_corn_planting_bags_used_l2972_297271


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l2972_297236

/-- The quadratic equation x^2 - x + 1 = 0 has roots α and β -/
def has_roots (α β : ℂ) : Prop :=
  α^2 - α + 1 = 0 ∧ β^2 - β + 1 = 0

/-- The quadratic function f(x) = x^2 - 2x + 2 -/
def f (x : ℂ) : ℂ := x^2 - 2*x + 2

/-- Theorem stating that f(x) satisfies the required conditions -/
theorem quadratic_function_satisfies_conditions (α β : ℂ) 
  (h : has_roots α β) : f α = β ∧ f β = α ∧ f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l2972_297236


namespace NUMINAMATH_CALUDE_triangle_area_is_four_l2972_297293

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The area of a triangle given two sides and the sine of the included angle. -/
def triangleArea (s1 s2 sinAngle : ℝ) : ℝ :=
  0.5 * s1 * s2 * sinAngle

/-- The theorem stating that the area of the given triangle is 4. -/
theorem triangle_area_is_four (t : Triangle) 
    (ha : t.a = 2)
    (hc : t.c = 5)
    (hcosB : Real.cos t.angleB = 3/5) : 
    triangleArea t.a t.c (Real.sin t.angleB) = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_four_l2972_297293


namespace NUMINAMATH_CALUDE_power_division_23_l2972_297269

theorem power_division_23 : (23 : ℕ)^11 / (23 : ℕ)^8 = 12167 := by
  sorry

end NUMINAMATH_CALUDE_power_division_23_l2972_297269


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2972_297242

-- Define the colors
inductive Color
  | Red
  | Blue
  | Green

-- Define the board as a function from coordinates to colors
def Board := ℤ × ℤ → Color

-- Define a rectangle on the board
def IsRectangle (board : Board) (x1 y1 x2 y2 : ℤ) : Prop :=
  x1 ≠ x2 ∧ y1 ≠ y2 ∧
  board (x1, y1) = board (x2, y1) ∧
  board (x1, y1) = board (x1, y2) ∧
  board (x1, y1) = board (x2, y2)

-- The main theorem
theorem monochromatic_rectangle_exists (board : Board) :
  ∃ x1 y1 x2 y2 : ℤ, IsRectangle board x1 y1 x2 y2 := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2972_297242


namespace NUMINAMATH_CALUDE_smallest_shift_for_scaled_periodic_function_l2972_297292

-- Define a periodic function with period 20
def isPeriodic20 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 20) = f x

-- Define the property we want to prove
def smallestShift (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f ((x - a) / 5) = f (x / 5)) ∧
  (∀ b, 0 < b → b < a → ∃ x, f ((x - b) / 5) ≠ f (x / 5))

-- Theorem statement
theorem smallest_shift_for_scaled_periodic_function (f : ℝ → ℝ) (h : isPeriodic20 f) :
  smallestShift f 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_shift_for_scaled_periodic_function_l2972_297292


namespace NUMINAMATH_CALUDE_gym_floor_area_per_person_l2972_297211

theorem gym_floor_area_per_person :
  ∀ (base height : ℝ) (num_students : ℕ),
    base = 9 →
    height = 8 →
    num_students = 24 →
    (base * height) / num_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_gym_floor_area_per_person_l2972_297211


namespace NUMINAMATH_CALUDE_four_card_selection_three_suits_l2972_297267

theorem four_card_selection_three_suits (deck_size : Nat) (suits : Nat) (cards_per_suit : Nat) 
  (selection_size : Nat) (suits_represented : Nat) (cards_from_main_suit : Nat) :
  deck_size = suits * cards_per_suit →
  selection_size = 4 →
  suits = 4 →
  cards_per_suit = 13 →
  suits_represented = 3 →
  cards_from_main_suit = 2 →
  (suits.choose 1) * (suits - 1).choose (suits_represented - 1) * 
  (cards_per_suit.choose cards_from_main_suit) * 
  (cards_per_suit.choose 1) * (cards_per_suit.choose 1) = 158184 := by
sorry

end NUMINAMATH_CALUDE_four_card_selection_three_suits_l2972_297267


namespace NUMINAMATH_CALUDE_oil_production_scientific_notation_l2972_297231

theorem oil_production_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 45000000 = a * (10 : ℝ) ^ n ∧ a = 4.5 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_oil_production_scientific_notation_l2972_297231


namespace NUMINAMATH_CALUDE_dollar_operation_result_l2972_297256

/-- Custom dollar operation -/
def dollar (a b c : ℝ) : ℝ := (a - b + c)^2

/-- Theorem statement -/
theorem dollar_operation_result (x z : ℝ) :
  dollar ((x + z)^2) ((z - x)^2) ((x - z)^2) = (x + z)^4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_operation_result_l2972_297256


namespace NUMINAMATH_CALUDE_average_after_addition_l2972_297203

theorem average_after_addition (numbers : List ℝ) (initial_average : ℝ) (addition : ℝ) : 
  numbers.length = 15 →
  initial_average = 40 →
  addition = 12 →
  (numbers.map (· + addition)).sum / numbers.length = 52 := by
  sorry

end NUMINAMATH_CALUDE_average_after_addition_l2972_297203


namespace NUMINAMATH_CALUDE_equation_rearrangement_l2972_297261

theorem equation_rearrangement (s P k c n : ℝ) 
  (h : P = s / ((1 + k)^n + c)) 
  (h_pos : s > 0) 
  (h_k_pos : k > -1) 
  (h_P_pos : P > 0) 
  (h_denom_pos : (s/P) - c > 0) : 
  n = (Real.log ((s/P) - c)) / (Real.log (1 + k)) := by
sorry

end NUMINAMATH_CALUDE_equation_rearrangement_l2972_297261


namespace NUMINAMATH_CALUDE_four_point_partition_l2972_297299

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A straight line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- A set of four points in a plane --/
def FourPoints := Fin 4 → Point

/-- A partition of four points into two non-empty subsets --/
structure Partition (pts : FourPoints) where
  set1 : Set (Fin 4)
  set2 : Set (Fin 4)
  partition : set1 ∪ set2 = Set.univ
  nonempty1 : set1.Nonempty
  nonempty2 : set2.Nonempty

/-- Check if a line separates two sets of points --/
def separates (l : Line) (pts : FourPoints) (p : Partition pts) : Prop :=
  (∀ i ∈ p.set1, (pts i).onLine l) ∧ (∀ i ∈ p.set2, ¬(pts i).onLine l) ∨
  (∀ i ∈ p.set1, ¬(pts i).onLine l) ∧ (∀ i ∈ p.set2, (pts i).onLine l)

/-- The main theorem --/
theorem four_point_partition (pts : FourPoints) :
  ∃ p : Partition pts, ∀ l : Line, ¬separates l pts p := by
  sorry

end NUMINAMATH_CALUDE_four_point_partition_l2972_297299


namespace NUMINAMATH_CALUDE_max_playground_area_l2972_297240

/-- Represents the dimensions of a rectangular playground. -/
structure Playground where
  width : ℝ
  length : ℝ

/-- The total fencing available for the playground. -/
def totalFencing : ℝ := 480

/-- Calculates the area of a playground. -/
def area (p : Playground) : ℝ := p.width * p.length

/-- Checks if a playground satisfies the fencing constraint. -/
def satisfiesFencingConstraint (p : Playground) : Prop :=
  p.length + 2 * p.width = totalFencing

/-- Theorem stating the maximum area of the playground. -/
theorem max_playground_area :
  ∃ (p : Playground), satisfiesFencingConstraint p ∧
    area p = 28800 ∧
    ∀ (q : Playground), satisfiesFencingConstraint q → area q ≤ area p :=
sorry

end NUMINAMATH_CALUDE_max_playground_area_l2972_297240


namespace NUMINAMATH_CALUDE_kyle_corn_purchase_l2972_297288

-- Define the problem parameters
def total_pounds : ℝ := 30
def total_cost : ℝ := 22.50
def corn_price : ℝ := 1.05
def beans_price : ℝ := 0.55

-- Define the theorem
theorem kyle_corn_purchase :
  ∃ (corn beans : ℝ),
    corn + beans = total_pounds ∧
    corn_price * corn + beans_price * beans = total_cost ∧
    corn = 12 := by
  sorry

end NUMINAMATH_CALUDE_kyle_corn_purchase_l2972_297288


namespace NUMINAMATH_CALUDE_distance_between_points_l2972_297290

/-- The distance between two points given their net movements -/
theorem distance_between_points (south west : ℝ) (h : south = 30 ∧ west = 40) :
  Real.sqrt (south^2 + west^2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2972_297290


namespace NUMINAMATH_CALUDE_sum_quotient_reciprocal_l2972_297263

theorem sum_quotient_reciprocal (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 45) (h4 : x * y = 500) : 
  (x / y) + (1 / x) + (1 / y) = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_sum_quotient_reciprocal_l2972_297263


namespace NUMINAMATH_CALUDE_parabola_line_distance_l2972_297214

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  k : ℝ

/-- Problem statement -/
theorem parabola_line_distance (parab : Parabola) (l : Line) : 
  (parab.p / 2 = 4) →
  (∃ x y : ℝ, x^2 = 2 * parab.p * y ∧ y = l.k * (x + 1)) →
  (∀ x y : ℝ, x^2 = 2 * parab.p * y ∧ y = l.k * (x + 1) → x = -1) →
  let focus_distance := dist (0, parab.p / 2) (x, l.k * (x + 1))
  (∃ x : ℝ, focus_distance = 1 ∨ focus_distance = 4 ∨ focus_distance = Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_distance_l2972_297214


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l2972_297259

/-- The ellipse in the first quadrant -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1 ∧ x > 0 ∧ y > 0

/-- The dot product of OP and PF -/
def dot_product (x y : ℝ) : ℝ := x*(3-x) - y^2

theorem ellipse_dot_product_range :
  ∀ x y : ℝ, ellipse x y → -16 < dot_product x y ∧ dot_product x y ≤ -39/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l2972_297259


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l2972_297298

theorem set_membership_implies_value (m : ℚ) : 
  let A : Set ℚ := {m + 2, 2 * m^2 + m}
  3 ∈ A → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l2972_297298


namespace NUMINAMATH_CALUDE_union_not_all_reals_l2972_297246

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {y | 0 < y}

theorem union_not_all_reals : M ∪ N ≠ Set.univ := by sorry

end NUMINAMATH_CALUDE_union_not_all_reals_l2972_297246


namespace NUMINAMATH_CALUDE_max_equal_ending_digits_of_squares_max_equal_ending_digits_is_tight_l2972_297272

/-- The maximum number of equal non-zero digits that can appear at the end of a perfect square in base 10 -/
def max_equal_ending_digits : ℕ := 3

/-- A function that returns the number of equal non-zero digits at the end of a number in base 10 -/
def count_equal_ending_digits (n : ℕ) : ℕ := sorry

theorem max_equal_ending_digits_of_squares :
  ∀ n : ℕ, count_equal_ending_digits (n^2) ≤ max_equal_ending_digits :=
by sorry

theorem max_equal_ending_digits_is_tight :
  ∃ n : ℕ, count_equal_ending_digits (n^2) = max_equal_ending_digits :=
by sorry

end NUMINAMATH_CALUDE_max_equal_ending_digits_of_squares_max_equal_ending_digits_is_tight_l2972_297272


namespace NUMINAMATH_CALUDE_daisy_shop_total_sales_l2972_297275

def daisy_shop_sales (day1 : ℕ) (day2_increase : ℕ) (day3_decrease : ℕ) (day4 : ℕ) : ℕ :=
  let day2 := day1 + day2_increase
  let day3 := 2 * day2 - day3_decrease
  day1 + day2 + day3 + day4

theorem daisy_shop_total_sales :
  daisy_shop_sales 45 20 10 120 = 350 := by
  sorry

end NUMINAMATH_CALUDE_daisy_shop_total_sales_l2972_297275


namespace NUMINAMATH_CALUDE_furniture_store_optimal_profit_l2972_297289

/-- Represents the furniture store's purchase and sales plan -/
structure FurnitureStore where
  a : ℝ  -- Original purchase price of dining table
  tableRetailPrice : ℝ := 270
  chairRetailPrice : ℝ := 70
  setPrice : ℝ := 500
  numTables : ℕ
  numChairs : ℕ

/-- Calculates the profit for the furniture store -/
def profit (store : FurnitureStore) : ℝ :=
  let numSets := store.numTables / 2
  let remainingTables := store.numTables - numSets
  let chairsInSets := numSets * 4
  let remainingChairs := store.numChairs - chairsInSets
  (store.setPrice - store.a - 4 * (store.a - 110)) * numSets +
  (store.tableRetailPrice - store.a) * remainingTables +
  (store.chairRetailPrice - (store.a - 110)) * remainingChairs

/-- The main theorem to be proved -/
theorem furniture_store_optimal_profit (store : FurnitureStore) :
  (600 / store.a = 160 / (store.a - 110)) →
  (store.numChairs = 5 * store.numTables + 20) →
  (store.numTables + store.numChairs ≤ 200) →
  (∃ (maxProfit : ℝ), 
    maxProfit = 7950 ∧ 
    store.a = 150 ∧ 
    store.numTables = 30 ∧ 
    store.numChairs = 170 ∧
    profit store = maxProfit ∧
    ∀ (otherStore : FurnitureStore), 
      (600 / otherStore.a = 160 / (otherStore.a - 110)) →
      (otherStore.numChairs = 5 * otherStore.numTables + 20) →
      (otherStore.numTables + otherStore.numChairs ≤ 200) →
      profit otherStore ≤ maxProfit) := by
  sorry

end NUMINAMATH_CALUDE_furniture_store_optimal_profit_l2972_297289


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l2972_297222

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 5*a^2 + 7*a - 2 = 0) ∧ 
  (b^3 - 5*b^2 + 7*b - 2 = 0) ∧ 
  (c^3 - 5*c^2 + 7*c - 2 = 0) → 
  ((a - 3)^3 + 4*(a - 3)^2 + 4*(a - 3) + 1 = 0) ∧
  ((b - 3)^3 + 4*(b - 3)^2 + 4*(b - 3) + 1 = 0) ∧
  ((c - 3)^3 + 4*(c - 3)^2 + 4*(c - 3) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l2972_297222


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l2972_297253

theorem largest_triangle_perimeter :
  ∀ y : ℕ,
  y > 0 →
  y < 16 →
  7 + y > 9 →
  9 + y > 7 →
  (∀ z : ℕ, z > 0 → z < 16 → 7 + z > 9 → 9 + z > 7 → 7 + 9 + y ≥ 7 + 9 + z) →
  7 + 9 + y = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l2972_297253


namespace NUMINAMATH_CALUDE_f_symmetry_l2972_297291

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2972_297291


namespace NUMINAMATH_CALUDE_cooking_probability_l2972_297284

-- Define a finite set of courses
def Courses : Type := Fin 4

-- Define a probability measure on the set of courses
def prob : Courses → ℚ := λ _ => 1 / 4

-- Theorem statement
theorem cooking_probability :
  ∀ (c : Courses), prob c = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_cooking_probability_l2972_297284


namespace NUMINAMATH_CALUDE_zebras_permutations_l2972_297228

theorem zebras_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_zebras_permutations_l2972_297228


namespace NUMINAMATH_CALUDE_exists_sum_all_odd_digits_l2972_297268

/-- A function that returns true if all digits of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- A function that reverses the digits of a three-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 100 + (n / 10 % 10) * 10 + (n / 100)

/-- Theorem stating that there exists a three-digit number A such that
    A + reverseDigits(A) has all odd digits -/
theorem exists_sum_all_odd_digits :
  ∃ A : ℕ, 100 ≤ A ∧ A < 1000 ∧ allDigitsOdd (A + reverseDigits A) :=
sorry

end NUMINAMATH_CALUDE_exists_sum_all_odd_digits_l2972_297268


namespace NUMINAMATH_CALUDE_binary_equals_base_4_l2972_297277

-- Define the binary number
def binary_num : List Bool := [true, false, true, false, true, true, true, false, true]

-- Define the base 4 number
def base_4_num : List Nat := [1, 1, 3, 1]

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : Nat :=
  bin.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert base 4 to decimal
def base_4_to_decimal (b4 : List Nat) : Nat :=
  b4.reverse.enum.foldl (fun acc (i, d) => acc + d * 4^i) 0

-- Theorem statement
theorem binary_equals_base_4 :
  binary_to_decimal binary_num = base_4_to_decimal base_4_num := by
  sorry

end NUMINAMATH_CALUDE_binary_equals_base_4_l2972_297277


namespace NUMINAMATH_CALUDE_subset_intersection_condition_l2972_297210

theorem subset_intersection_condition (n : ℕ) (h : n ≥ 4) :
  (∀ (S : Finset (Finset (Fin n))) (h_card : S.card = n) 
    (h_subsets : ∀ s ∈ S, s.card = 3),
    ∃ (s1 s2 : Finset (Fin n)), s1 ∈ S ∧ s2 ∈ S ∧ s1 ≠ s2 ∧ (s1 ∩ s2).card = 1) ↔
  n % 4 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_l2972_297210
