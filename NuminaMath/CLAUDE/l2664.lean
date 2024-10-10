import Mathlib

namespace target_hit_probability_l2664_266455

theorem target_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/2) 
  (h2 : prob_B = 1/3) : 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
  sorry

end target_hit_probability_l2664_266455


namespace mono_decreasing_inequality_l2664_266433

/-- A function f is monotonically decreasing on ℝ -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- Given a monotonically decreasing function f on ℝ,
    if f(2m) > f(1+m), then m < 1 -/
theorem mono_decreasing_inequality (f : ℝ → ℝ) (m : ℝ)
    (h_mono : MonotonicallyDecreasing f) (h_ineq : f (2 * m) > f (1 + m)) :
    m < 1 :=
  sorry

end mono_decreasing_inequality_l2664_266433


namespace solve_for_y_l2664_266412

theorem solve_for_y (x y : ℝ) (h1 : x^2 = 2*y - 6) (h2 : x = 3) : y = 7.5 := by
  sorry

end solve_for_y_l2664_266412


namespace water_depth_in_specific_tank_l2664_266400

/-- Represents a horizontal cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the possible depths of water in a cylindrical tank given the surface area -/
def water_depth (tank : CylindricalTank) (surface_area : ℝ) : Set ℝ :=
  { h : ℝ | ∃ (c : ℝ), 
    c = surface_area / tank.length ∧ 
    c = 2 * Real.sqrt (tank.diameter * h - h^2) ∧ 
    0 < h ∧ 
    h < tank.diameter }

/-- Theorem stating the possible depths of water in a specific cylindrical tank -/
theorem water_depth_in_specific_tank : 
  let tank : CylindricalTank := { length := 12, diameter := 8 }
  let surface_area := 48
  water_depth tank surface_area = {4 - 2 * Real.sqrt 3, 4 + 2 * Real.sqrt 3} :=
by
  sorry

end water_depth_in_specific_tank_l2664_266400


namespace sufficient_condition_for_x_squared_minus_a_nonnegative_l2664_266454

theorem sufficient_condition_for_x_squared_minus_a_nonnegative 
  (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≥ 0) ↔ 
  (a ≤ -1 ∧ ∃ b : ℝ, b > -1 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - b ≥ 0) :=
sorry

end sufficient_condition_for_x_squared_minus_a_nonnegative_l2664_266454


namespace quadratic_max_value_l2664_266411

def f (a : ℝ) (x : ℝ) := a * x^2 + 2 * a * x + 1

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = -3 ∨ a = 3/8 := by sorry

end quadratic_max_value_l2664_266411


namespace simple_interest_problem_l2664_266474

/-- Proves that given the conditions of the problem, the principal amount must be 700 --/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_rate : R > 0) : 
  (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56 → P = 700 := by
  sorry

#check simple_interest_problem

end simple_interest_problem_l2664_266474


namespace remainder_theorem_l2664_266403

theorem remainder_theorem (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := by
  sorry

end remainder_theorem_l2664_266403


namespace probability_between_X_and_Z_l2664_266466

/-- Given a line segment XW where XW = 4XZ = 8YW, the probability of selecting a point between X and Z is 1/4 -/
theorem probability_between_X_and_Z (XW XZ YW : ℝ) 
  (h1 : XW = 4 * XZ) 
  (h2 : XW = 8 * YW) 
  (h3 : XW > 0) : 
  XZ / XW = 1 / 4 := by
  sorry

end probability_between_X_and_Z_l2664_266466


namespace min_value_reciprocal_sum_l2664_266483

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / (b + 1) ≥ 2) ∧
  (1 / a + 1 / (b + 1) = 2 ↔ a = 1 / 2 ∧ b = 1 / 2) := by
sorry

end min_value_reciprocal_sum_l2664_266483


namespace extra_minutes_per_A_is_correct_l2664_266453

/-- The number of extra minutes earned for each A grade -/
def extra_minutes_per_A : ℕ := 2

/-- The normal recess time in minutes -/
def normal_recess : ℕ := 20

/-- The number of A grades -/
def num_A : ℕ := 10

/-- The number of B grades -/
def num_B : ℕ := 12

/-- The number of C grades -/
def num_C : ℕ := 14

/-- The number of D grades -/
def num_D : ℕ := 5

/-- The total recess time in minutes -/
def total_recess : ℕ := 47

theorem extra_minutes_per_A_is_correct :
  extra_minutes_per_A * num_A + num_B - num_D = total_recess - normal_recess :=
by sorry

end extra_minutes_per_A_is_correct_l2664_266453


namespace difference_multiple_of_nine_l2664_266496

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem difference_multiple_of_nine (q r : ℕ) :
  is_two_digit q ∧ 
  is_two_digit r ∧ 
  r = reverse_digits q ∧
  (∀ x y : ℕ, is_two_digit x ∧ is_two_digit y ∧ y = reverse_digits x → x - y ≤ 27) →
  ∃ k : ℕ, q - r = 9 * k ∨ r - q = 9 * k :=
sorry

end difference_multiple_of_nine_l2664_266496


namespace square_side_length_average_l2664_266409

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end square_side_length_average_l2664_266409


namespace jamie_calculation_l2664_266424

theorem jamie_calculation (x : ℝ) : (x / 8 + 20 = 28) → (x * 8 - 20 = 492) := by
  sorry

end jamie_calculation_l2664_266424


namespace arithmetic_sequence_sum_l2664_266457

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 48) →
  a 6 + a 7 = 24 := by
sorry

end arithmetic_sequence_sum_l2664_266457


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l2664_266461

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 - 1)

-- Theorem for real number condition
theorem real_number_condition (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = -1 := by sorry

-- Theorem for imaginary number condition
theorem imaginary_number_condition (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1 := by sorry

-- Theorem for pure imaginary number condition
theorem pure_imaginary_number_condition (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l2664_266461


namespace quadratic_equation_m_value_l2664_266431

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 3) * x^(m^2 - 7) + m*x - 2 = a*x^2 + b*x + c) ∧ 
  (m + 3 ≠ 0) → 
  m = 3 := by
sorry

end quadratic_equation_m_value_l2664_266431


namespace unique_pair_divisibility_l2664_266458

theorem unique_pair_divisibility (a b : ℕ) : 
  (7^a - 3^b) ∣ (a^4 + b^2) → a = 2 ∧ b = 4 := by
  sorry

end unique_pair_divisibility_l2664_266458


namespace sufficient_but_not_necessary_condition_l2664_266467

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 ≤ a) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 ≤ a) → (a ≥ 5)) :=
by sorry

end sufficient_but_not_necessary_condition_l2664_266467


namespace line_not_in_second_quadrant_l2664_266484

-- Define the line l with equation x - y - a² = 0
def line_equation (x y a : ℝ) : Prop := x - y - a^2 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, line_equation x y a → ¬ second_quadrant x y :=
sorry

end line_not_in_second_quadrant_l2664_266484


namespace calculate_interest_rate_l2664_266489

/-- Given a sum of money invested at simple interest, this theorem proves
    that if the interest earned is a certain amount more than what would
    be earned at a reference rate, then the actual interest rate can be
    calculated. -/
theorem calculate_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (reference_rate : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 4200)
  (h2 : time = 2)
  (h3 : reference_rate = 0.12)
  (h4 : interest_difference = 504)
  : (principal * time * reference_rate + interest_difference) / (principal * time) = 0.18 := by
  sorry

end calculate_interest_rate_l2664_266489


namespace least_number_for_divisibility_l2664_266463

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1055 ∧ m = 23) :
  ∃ (x : ℕ), (n + x) % m = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % m ≠ 0 ∧ x = 3 :=
sorry

end least_number_for_divisibility_l2664_266463


namespace fraction_reduction_l2664_266426

theorem fraction_reduction (n : ℤ) :
  (∃ k : ℤ, n = 7 * k + 1) ↔
  (∃ m : ℤ, 4 * n + 3 = 7 * m) ∧ (∃ l : ℤ, 5 * n + 2 = 7 * l) :=
by sorry

end fraction_reduction_l2664_266426


namespace balanced_132_l2664_266401

def is_balanced (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧  -- All digits are different
  n = (10 * (n / 100) + (n / 10) % 10) +
      (10 * (n / 100) + n % 10) +
      (10 * ((n / 10) % 10) + n / 100) +
      (10 * ((n / 10) % 10) + n % 10) +
      (10 * (n % 10) + n / 100) +
      (10 * (n % 10) + (n / 10) % 10)  -- Sum of all possible two-digit numbers

theorem balanced_132 : is_balanced 132 := by
  sorry

end balanced_132_l2664_266401


namespace triangle_properties_l2664_266435

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  c = a * (Real.cos B + Real.sqrt 3 * Real.sin B) →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 →
  a = 1 →
  A = π / 6 ∧ a + b + c = Real.sqrt 3 + 2 := by
sorry

end triangle_properties_l2664_266435


namespace leopard_arrangement_l2664_266402

theorem leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 : ℕ) * Nat.factorial 2 * Nat.factorial (n - 3) = 2880 := by
  sorry

end leopard_arrangement_l2664_266402


namespace first_month_sale_l2664_266473

def average_sale : ℝ := 6000
def num_months : ℕ := 5
def sale_2 : ℝ := 5660
def sale_3 : ℝ := 6200
def sale_4 : ℝ := 6350
def sale_5 : ℝ := 6500
def sale_6 : ℝ := 5870

theorem first_month_sale (sale_1 : ℝ) :
  (sale_1 + sale_2 + sale_3 + sale_4 + sale_5) / num_months = average_sale →
  sale_1 = 5290 := by
sorry

end first_month_sale_l2664_266473


namespace calligraphy_class_equation_l2664_266446

theorem calligraphy_class_equation (x : ℕ+) 
  (h1 : 6 * x = planned + 7)
  (h2 : 5 * x = planned - 13)
  : 6 * x - 7 = 5 * x + 13 := by
  sorry

end calligraphy_class_equation_l2664_266446


namespace cats_remaining_l2664_266423

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) 
  (h1 : siamese = 13) 
  (h2 : house = 5) 
  (h3 : sold = 10) : 
  siamese + house - sold = 8 := by
  sorry

end cats_remaining_l2664_266423


namespace square_of_arithmetic_mean_geq_product_l2664_266478

theorem square_of_arithmetic_mean_geq_product (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : c = (a + b) / 2) : c^2 ≥ a * b := by
  sorry

end square_of_arithmetic_mean_geq_product_l2664_266478


namespace binomial_coefficient_sum_l2664_266487

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 := by
sorry

end binomial_coefficient_sum_l2664_266487


namespace triangle_area_theorem_l2664_266465

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end triangle_area_theorem_l2664_266465


namespace knife_sharpening_problem_l2664_266471

/-- Represents the pricing structure for knife sharpening -/
structure KnifeSharpeningPrices where
  first_knife : ℕ → ℕ
  next_three_knives : ℕ → ℕ
  remaining_knives : ℕ → ℕ

/-- Calculates the total cost of sharpening knives -/
def total_cost (prices : KnifeSharpeningPrices) (num_knives : ℕ) : ℕ :=
  prices.first_knife 1 +
  prices.next_three_knives (min 3 (num_knives - 1)) +
  prices.remaining_knives (max 0 (num_knives - 4))

/-- Theorem stating that given the pricing structure and total cost, the number of knives is 9 -/
theorem knife_sharpening_problem (prices : KnifeSharpeningPrices) 
  (h1 : prices.first_knife 1 = 5)
  (h2 : ∀ n : ℕ, n ≤ 3 → prices.next_three_knives n = 4 * n)
  (h3 : ∀ n : ℕ, prices.remaining_knives n = 3 * n)
  (h4 : total_cost prices 9 = 32) :
  ∃ (n : ℕ), total_cost prices n = 32 ∧ n = 9 := by
  sorry


end knife_sharpening_problem_l2664_266471


namespace kite_coefficient_sum_l2664_266407

/-- Represents a parabola in the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Represents a kite formed by the intersection of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola
  area : ℝ

/-- The sum of coefficients a and b for two parabolas forming a kite with area 12 -/
def coefficient_sum (k : Kite) : ℝ := k.p1.a + (-k.p2.a)

/-- Theorem stating that the sum of coefficients a and b is 1.5 for the given conditions -/
theorem kite_coefficient_sum :
  ∀ (k : Kite),
    k.p1.c = -2 ∧ 
    k.p2.c = 4 ∧ 
    k.area = 12 →
    coefficient_sum k = 1.5 := by
  sorry


end kite_coefficient_sum_l2664_266407


namespace ones_digit_of_large_power_l2664_266488

theorem ones_digit_of_large_power : ∃ n : ℕ, n > 0 ∧ 17^(17*(5^5)) ≡ 7 [ZMOD 10] := by
  sorry

end ones_digit_of_large_power_l2664_266488


namespace coin_combination_theorem_l2664_266462

/-- Represents the number of different coin values obtainable -/
def different_values (num_five_cent : ℕ) (num_ten_cent : ℕ) : ℕ :=
  29 - num_five_cent

/-- Theorem stating that given 15 coins with 22 different obtainable values, there must be 8 10-cent coins -/
theorem coin_combination_theorem :
  ∀ (num_five_cent num_ten_cent : ℕ),
    num_five_cent + num_ten_cent = 15 →
    different_values num_five_cent num_ten_cent = 22 →
    num_ten_cent = 8 := by
  sorry

#check coin_combination_theorem

end coin_combination_theorem_l2664_266462


namespace constant_term_expansion_l2664_266428

/-- The constant term in the expansion of (√x + 3/x)^12 -/
def constantTerm : ℕ := 40095

/-- The binomial coefficient (12 choose 8) -/
def binomialCoeff : ℕ := 495

theorem constant_term_expansion :
  constantTerm = binomialCoeff * 3^4 := by sorry

end constant_term_expansion_l2664_266428


namespace division_problem_l2664_266438

theorem division_problem (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 64 → quotient = 8 → dividend = divisor * quotient → divisor = 8 := by
  sorry

end division_problem_l2664_266438


namespace stratified_sampling_second_year_l2664_266416

/-- The number of students in the second year of high school -/
def second_year_students : ℕ := 750

/-- The probability of a student being selected in the stratified sampling -/
def selection_probability : ℚ := 2 / 100

/-- The number of students to be drawn from the second year -/
def students_drawn : ℕ := 15

/-- Theorem stating that the number of students drawn from the second year
    is equal to the product of the total number of second-year students
    and the selection probability -/
theorem stratified_sampling_second_year :
  (second_year_students : ℚ) * selection_probability = students_drawn := by
  sorry

end stratified_sampling_second_year_l2664_266416


namespace jos_number_l2664_266494

theorem jos_number (n k l : ℕ) : 
  0 < n ∧ n < 150 ∧ n = 9 * k - 2 ∧ n = 8 * l - 4 →
  n ≤ 132 ∧ (∃ (k' l' : ℕ), 132 = 9 * k' - 2 ∧ 132 = 8 * l' - 4) :=
by sorry

end jos_number_l2664_266494


namespace cans_per_bag_l2664_266440

theorem cans_per_bag (total_bags : ℕ) (total_cans : ℕ) (h1 : total_bags = 9) (h2 : total_cans = 72) :
  total_cans / total_bags = 8 :=
by sorry

end cans_per_bag_l2664_266440


namespace power_ranger_stickers_l2664_266439

theorem power_ranger_stickers (total : ℕ) (difference : ℕ) (first_box : ℕ) : 
  total = 58 → difference = 12 → first_box + (first_box + difference) = total → first_box = 23 := by
  sorry

end power_ranger_stickers_l2664_266439


namespace train_length_l2664_266499

/-- Given a train that crosses a platform of length 350 meters in 39 seconds
    and crosses a signal pole in 18 seconds, the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
    (h1 : platform_length = 350)
    (h2 : platform_time = 39)
    (h3 : pole_time = 18) :
    (platform_length * pole_time) / (platform_time - pole_time) = 300 :=
by sorry

end train_length_l2664_266499


namespace average_headcount_spring_terms_l2664_266475

def spring_02_03 : ℕ := 10900
def spring_03_04 : ℕ := 10500
def spring_04_05 : ℕ := 10700
def spring_05_06 : ℕ := 11300

def total_headcount : ℕ := spring_02_03 + spring_03_04 + spring_04_05 + spring_05_06
def num_terms : ℕ := 4

theorem average_headcount_spring_terms :
  (total_headcount : ℚ) / num_terms = 10850 := by sorry

end average_headcount_spring_terms_l2664_266475


namespace sequence_problem_l2664_266497

theorem sequence_problem (a : ℕ → ℤ) (h1 : a 5 = 14) (h2 : ∀ n : ℕ, a (n + 1) - a n = n + 1) : a 1 = 0 := by
  sorry

end sequence_problem_l2664_266497


namespace intersection_point_properties_l2664_266415

/-- Point P where the given lines intersect -/
def P : ℝ × ℝ := (1, 1)

/-- Line perpendicular to 3x + 4y - 15 = 0 passing through P -/
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

/-- Line with equal intercepts passing through P -/
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0

theorem intersection_point_properties :
  (∃ (x y : ℝ), 2 * x - y - 1 = 0 ∧ x - 2 * y + 1 = 0 ∧ (x, y) = P) ∧
  (∀ (x y : ℝ), l₁ x y ↔ (4 * x - 3 * y - 1 = 0 ∧ (x, y) = P ∨ (x, y) ≠ P)) ∧
  (∀ (x y : ℝ), l₂ x y ↔ (x + y - 2 = 0 ∧ (x, y) = P ∨ (x, y) ≠ P)) := by
  sorry

end intersection_point_properties_l2664_266415


namespace increase_in_average_age_increase_in_average_age_is_two_l2664_266492

/-- The increase in average age when replacing two men with two women in a group -/
theorem increase_in_average_age : ℝ :=
  let initial_group_size : ℕ := 8
  let replaced_men_ages : List ℝ := [20, 24]
  let women_average_age : ℝ := 30
  let total_age_increase : ℝ := 2 * women_average_age - replaced_men_ages.sum
  total_age_increase / initial_group_size

/-- Proof that the increase in average age is 2 years -/
theorem increase_in_average_age_is_two :
  increase_in_average_age = 2 := by
  sorry

end increase_in_average_age_increase_in_average_age_is_two_l2664_266492


namespace cosine_sine_sum_equals_sqrt_two_over_two_l2664_266405

theorem cosine_sine_sum_equals_sqrt_two_over_two : 
  Real.cos (70 * π / 180) * Real.cos (335 * π / 180) + 
  Real.sin (110 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end cosine_sine_sum_equals_sqrt_two_over_two_l2664_266405


namespace circle_ellipse_tangent_l2664_266414

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - 3 = 0

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

-- Define the focus F
def focus_F (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x = -c

theorem circle_ellipse_tangent (m a c : ℝ) :
  m < 0 →
  (∀ x y, circle_M m x y → (x - 1)^2 + y^2 = 4) →
  (∃ x y, circle_M m x y ∧ line_l c x y) →
  (∀ x y, circle_M m x y ∧ line_l c x y → (x - 1)^2 + y^2 = 4) →
  a = 2 :=
sorry

end circle_ellipse_tangent_l2664_266414


namespace equation_three_solutions_l2664_266498

theorem equation_three_solutions :
  ∃ (s : Finset ℝ), (s.card = 3) ∧ 
  (∀ x ∈ s, (x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0) ∧
  (∀ y : ℝ, (y^2 - 6*y + 9) / (y - 1) - (3 - y) / (y^2 - 1) = 0 → y ∈ s) := by
  sorry

end equation_three_solutions_l2664_266498


namespace quadratic_function_a_value_l2664_266430

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexX (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexY (f : QuadraticFunction) : ℚ :=
  f.eval (f.vertexX)

theorem quadratic_function_a_value (f : QuadraticFunction) :
  f.vertexX = 2 ∧ f.vertexY = 5 ∧ f.eval 1 = 2 ∧ f.eval 3 = 2 → f.a = -3 := by
  sorry

end quadratic_function_a_value_l2664_266430


namespace smallest_number_of_eggs_l2664_266410

/-- The number of eggs in a full container -/
def full_container : ℕ := 15

/-- The number of eggs in an underfilled container -/
def underfilled_container : ℕ := 14

/-- The number of underfilled containers -/
def num_underfilled : ℕ := 3

/-- The minimum number of eggs initially bought -/
def min_initial_eggs : ℕ := 151

theorem smallest_number_of_eggs (n : ℕ) (h : n > min_initial_eggs) : 
  (∃ (c : ℕ), n = c * full_container - num_underfilled * (full_container - underfilled_container)) →
  162 ≤ n :=
by sorry

end smallest_number_of_eggs_l2664_266410


namespace min_a_for_inequality_l2664_266481

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end min_a_for_inequality_l2664_266481


namespace lottery_cheating_suspicion_l2664_266476

/-- The number of balls in the lottery -/
def total_balls : ℕ := 45

/-- The number of winning balls in each draw -/
def winning_balls : ℕ := 6

/-- The probability of no repetition in a single draw -/
def p : ℚ := (total_balls - winning_balls).choose winning_balls / total_balls.choose winning_balls

/-- The suspicion threshold -/
def threshold : ℚ := 1 / 100

theorem lottery_cheating_suspicion :
  p ^ 6 < threshold ∧ p ^ 5 ≥ threshold :=
sorry

end lottery_cheating_suspicion_l2664_266476


namespace unique_square_divisible_by_six_in_range_l2664_266429

theorem unique_square_divisible_by_six_in_range : ∃! x : ℕ, 
  (∃ n : ℕ, x = n^2) ∧ 
  (x % 6 = 0) ∧ 
  (50 ≤ x) ∧ 
  (x ≤ 150) ∧ 
  x = 144 := by
sorry

end unique_square_divisible_by_six_in_range_l2664_266429


namespace cone_height_l2664_266469

/-- Given a cone with slant height 13 cm and lateral area 65π cm², prove its height is 12 cm -/
theorem cone_height (s : ℝ) (l : ℝ) (h : ℝ) : 
  s = 13 → l = 65 * Real.pi → l = Real.pi * s * (l / (Real.pi * s)) → h^2 + (l / (Real.pi * s))^2 = s^2 → h = 12 := by
  sorry

end cone_height_l2664_266469


namespace helga_wrote_250_articles_l2664_266486

/-- Represents Helga's work schedule and article writing capacity --/
structure HelgaWorkSchedule where
  articles_per_half_hour : ℕ
  regular_hours_per_day : ℕ
  regular_days_per_week : ℕ
  extra_hours_thursday : ℕ
  extra_hours_friday : ℕ

/-- Calculates the total number of articles Helga wrote in a week --/
def total_articles_in_week (schedule : HelgaWorkSchedule) : ℕ :=
  let articles_per_hour := schedule.articles_per_half_hour * 2
  let regular_articles := articles_per_hour * schedule.regular_hours_per_day * schedule.regular_days_per_week
  let extra_articles := articles_per_hour * (schedule.extra_hours_thursday + schedule.extra_hours_friday)
  regular_articles + extra_articles

/-- Theorem stating that Helga wrote 250 articles in the given week --/
theorem helga_wrote_250_articles : 
  ∀ (schedule : HelgaWorkSchedule), 
    schedule.articles_per_half_hour = 5 ∧ 
    schedule.regular_hours_per_day = 4 ∧ 
    schedule.regular_days_per_week = 5 ∧ 
    schedule.extra_hours_thursday = 2 ∧ 
    schedule.extra_hours_friday = 3 →
    total_articles_in_week schedule = 250 := by
  sorry

end helga_wrote_250_articles_l2664_266486


namespace election_votes_theorem_l2664_266485

/-- Proves that in an election with two candidates, if the first candidate got 60% of the votes
    and the second candidate got 240 votes, then the total number of votes was 600. -/
theorem election_votes_theorem (total_votes : ℕ) (first_candidate_percentage : ℚ) 
    (second_candidate_votes : ℕ) : 
    first_candidate_percentage = 60 / 100 →
    second_candidate_votes = 240 →
    (1 - first_candidate_percentage) * total_votes = second_candidate_votes →
    total_votes = 600 := by
  sorry

#check election_votes_theorem

end election_votes_theorem_l2664_266485


namespace factorial_ratio_plus_two_l2664_266482

theorem factorial_ratio_plus_two : Nat.factorial 50 / Nat.factorial 48 + 2 = 2452 := by
  sorry

end factorial_ratio_plus_two_l2664_266482


namespace field_trip_students_l2664_266404

/-- Proves that the number of students on a field trip is equal to the product of seats per bus and number of buses -/
theorem field_trip_students (seats_per_bus : ℕ) (num_buses : ℕ) (h1 : seats_per_bus = 9) (h2 : num_buses = 5) :
  seats_per_bus * num_buses = 45 := by
  sorry

#check field_trip_students

end field_trip_students_l2664_266404


namespace intersection_of_M_and_N_l2664_266464

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x ≥ -2}

-- The theorem to prove
theorem intersection_of_M_and_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l2664_266464


namespace unique_solution_mn_l2664_266421

theorem unique_solution_mn (m n : ℕ+) 
  (h1 : (n^4 : ℕ) ∣ 2*m^5 - 1)
  (h2 : (m^4 : ℕ) ∣ 2*n^5 + 1) :
  m = 1 ∧ n = 1 := by
  sorry

end unique_solution_mn_l2664_266421


namespace bianca_drawing_time_l2664_266470

theorem bianca_drawing_time (school_time home_time total_time : ℕ) : 
  school_time = 22 → total_time = 41 → home_time = total_time - school_time → home_time = 19 :=
by sorry

end bianca_drawing_time_l2664_266470


namespace days_to_clear_land_l2664_266491

/-- Represents the number of feet in a yard -/
def feet_per_yard : ℝ := 3

/-- Represents the length of the land in feet -/
def land_length_feet : ℝ := 900

/-- Represents the width of the land in feet -/
def land_width_feet : ℝ := 200

/-- Represents the number of rabbits -/
def num_rabbits : ℕ := 100

/-- Represents the area one rabbit can clear per day in square yards -/
def area_per_rabbit_per_day : ℝ := 10

/-- Theorem stating the number of days needed to clear the land -/
theorem days_to_clear_land : 
  ⌈(land_length_feet / feet_per_yard) * (land_width_feet / feet_per_yard) / 
   (num_rabbits : ℝ) / area_per_rabbit_per_day⌉ = 21 := by sorry

end days_to_clear_land_l2664_266491


namespace negation_of_proposition_l2664_266495

theorem negation_of_proposition (a : ℝ) : 
  ¬(a ≠ 0 → a^2 > 0) ↔ (a = 0 → a^2 ≤ 0) := by sorry

end negation_of_proposition_l2664_266495


namespace equal_roots_quadratic_l2664_266434

/-- 
If a quadratic equation x² - 2x + m = 0 has two equal real roots,
then m = 1.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   (∀ y : ℝ, y^2 - 2*y + m = 0 → y = x)) → 
  m = 1 := by
sorry

end equal_roots_quadratic_l2664_266434


namespace prob_odd_after_removal_on_die_l2664_266480

/-- Represents a standard die face with its number of dots -/
inductive DieFace
| one
| two
| three
| four
| five
| six

/-- Calculates the number of ways to choose 2 dots from n dots -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the probability of choosing 2 dots from a face with n dots -/
def prob_choose_from_face (n : ℕ) : ℚ := (choose_two n : ℚ) / (choose_two 21 : ℚ)

/-- Determines if a face will have an odd number of dots after removing 2 dots -/
def odd_after_removal (face : DieFace) : Bool :=
  match face with
  | DieFace.one => false
  | DieFace.two => false
  | DieFace.three => false
  | DieFace.four => true
  | DieFace.five => false
  | DieFace.six => true

/-- Calculates the probability of getting an odd number of dots on a specific face after removal -/
def prob_odd_after_removal (face : DieFace) : ℚ :=
  if odd_after_removal face then
    match face with
    | DieFace.four => prob_choose_from_face 4
    | DieFace.six => prob_choose_from_face 6
    | _ => 0
  else 0

/-- The main theorem to prove -/
theorem prob_odd_after_removal_on_die : 
  (1 / 6 : ℚ) * (prob_odd_after_removal DieFace.one + 
                 prob_odd_after_removal DieFace.two + 
                 prob_odd_after_removal DieFace.three + 
                 prob_odd_after_removal DieFace.four + 
                 prob_odd_after_removal DieFace.five + 
                 prob_odd_after_removal DieFace.six) = 1 / 60 := by
  sorry

end prob_odd_after_removal_on_die_l2664_266480


namespace identical_sequences_l2664_266459

/-- Given two sequences of n real numbers where the first is strictly increasing,
    and their element-wise sum is strictly increasing,
    prove that the sequences are identical. -/
theorem identical_sequences
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h_a_increasing : ∀ i j : Fin n, i < j → a i < a j)
  (h_sum_increasing : ∀ i j : Fin n, i < j → a i + b i < a j + b j) :
  a = b :=
sorry

end identical_sequences_l2664_266459


namespace cos_2alpha_value_l2664_266443

theorem cos_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.cos (2 * α) = -7 / 25 := by
  sorry

end cos_2alpha_value_l2664_266443


namespace complex_number_location_l2664_266477

theorem complex_number_location (a : ℝ) (h : 0 < a ∧ a < 1) :
  let z : ℂ := Complex.mk a (a - 1)
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end complex_number_location_l2664_266477


namespace river_flow_speed_l2664_266460

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey -/
theorem river_flow_speed (distance : ℝ) (boat_speed : ℝ) (total_time : ℝ) :
  distance = 48 →
  boat_speed = 6 →
  total_time = 18 →
  ∃ (river_speed : ℝ),
    river_speed > 0 ∧
    (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed) = total_time) ∧
    river_speed = 2 := by
  sorry


end river_flow_speed_l2664_266460


namespace complex_square_expansion_l2664_266441

theorem complex_square_expansion (x y c : ℝ) : 
  (x + Complex.I * y + c)^2 = x^2 + c^2 - y^2 + 2*c*x + Complex.I * (2*x*y + 2*c*y) := by
  sorry

end complex_square_expansion_l2664_266441


namespace cosine_sum_inequality_l2664_266408

theorem cosine_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  |Real.cos x| + |Real.cos y| + |Real.cos z| ≥ 1 := by
  sorry

end cosine_sum_inequality_l2664_266408


namespace efficient_coefficient_computation_l2664_266490

/-- Represents a method to compute polynomial coefficients -/
structure ComputationMethod where
  (compute : (ℝ → ℝ) → List ℝ)
  (addition_count : ℕ)
  (multiplication_count : ℕ)

/-- A 6th degree polynomial -/
def Polynomial6 (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ → ℝ :=
  fun x ↦ x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆

/-- Theorem: There exists a method to compute coefficients of a 6th degree polynomial
    using its roots with no more than 15 additions and 15 multiplications -/
theorem efficient_coefficient_computation :
  ∃ (method : ComputationMethod),
    (∀ (p : ℝ → ℝ) (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
      (∀ x, p x = (x + r₁) * (x + r₂) * (x + r₃) * (x + r₄) * (x + r₅) * (x + r₆)) →
      ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
        p = Polynomial6 a₁ a₂ a₃ a₄ a₅ a₆ ∧
        method.compute p = [a₁, a₂, a₃, a₄, a₅, a₆]) ∧
    method.addition_count ≤ 15 ∧
    method.multiplication_count ≤ 15 :=
by sorry


end efficient_coefficient_computation_l2664_266490


namespace cubic_root_equation_solution_l2664_266418

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (Real.rpow (17 * x - 1) (1/3) + Real.rpow (11 * x + 1) (1/3) = 2 * Real.rpow x (1/3)) ↔ x = 0 := by
  sorry

end cubic_root_equation_solution_l2664_266418


namespace find_A_l2664_266493

def round_down_hundreds (n : ℕ) : ℕ := n / 100 * 100

def is_valid_number (n : ℕ) : Prop := 
  ∃ (A : ℕ), A < 10 ∧ n = 1000 + A * 100 + 77

theorem find_A : 
  ∀ (n : ℕ), is_valid_number n → round_down_hundreds n = 1700 → n = 1777 :=
sorry

end find_A_l2664_266493


namespace power_of_three_squared_to_fourth_l2664_266479

theorem power_of_three_squared_to_fourth : (3^2)^4 = 6561 := by
  sorry

end power_of_three_squared_to_fourth_l2664_266479


namespace fertilizer_weight_calculation_l2664_266451

/-- Calculates the total weight of fertilizers applied to a given area -/
theorem fertilizer_weight_calculation 
  (field_area : ℝ) 
  (fertilizer_a_rate : ℝ) 
  (fertilizer_a_area : ℝ) 
  (fertilizer_b_rate : ℝ) 
  (fertilizer_b_area : ℝ) 
  (area_to_fertilize : ℝ) : 
  field_area = 10800 ∧ 
  fertilizer_a_rate = 150 ∧ 
  fertilizer_a_area = 3000 ∧ 
  fertilizer_b_rate = 180 ∧ 
  fertilizer_b_area = 4000 ∧ 
  area_to_fertilize = 3600 → 
  (fertilizer_a_rate * area_to_fertilize / fertilizer_a_area) + 
  (fertilizer_b_rate * area_to_fertilize / fertilizer_b_area) = 342 := by
  sorry

#check fertilizer_weight_calculation

end fertilizer_weight_calculation_l2664_266451


namespace sufficient_not_necessary_l2664_266444

open Real

theorem sufficient_not_necessary (α : ℝ) :
  (∀ α, α = π/4 → sin α = cos α) ∧
  (∃ α, α ≠ π/4 ∧ sin α = cos α) :=
by sorry

end sufficient_not_necessary_l2664_266444


namespace solve_for_q_l2664_266437

theorem solve_for_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end solve_for_q_l2664_266437


namespace crayon_count_l2664_266425

theorem crayon_count (initial_crayons added_crayons : ℕ) 
  (h1 : initial_crayons = 9)
  (h2 : added_crayons = 3) : 
  initial_crayons + added_crayons = 12 := by
  sorry

end crayon_count_l2664_266425


namespace rhombus_constructible_l2664_266447

/-- Represents a rhombus in 2D space -/
structure Rhombus where
  /-- Side length of the rhombus -/
  side : ℝ
  /-- Difference between the two diagonals -/
  diag_diff : ℝ
  /-- Assumption that side length is positive -/
  side_pos : side > 0
  /-- Assumption that diagonal difference is non-negative and less than twice the side length -/
  diag_diff_valid : 0 ≤ diag_diff ∧ diag_diff < 2 * side

/-- Theorem stating that a rhombus can be constructed given a side length and diagonal difference -/
theorem rhombus_constructible (a : ℝ) (d : ℝ) (h1 : a > 0) (h2 : 0 ≤ d ∧ d < 2 * a) :
  ∃ (r : Rhombus), r.side = a ∧ r.diag_diff = d :=
sorry

end rhombus_constructible_l2664_266447


namespace wife_account_percentage_l2664_266442

def income : ℝ := 1000000

def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def orphan_house_percentage : ℝ := 0.05
def final_amount : ℝ := 50000

theorem wife_account_percentage : 
  let children_total := children_percentage * num_children * income
  let remaining_after_children := income - children_total
  let orphan_house_donation := orphan_house_percentage * remaining_after_children
  let remaining_after_donation := remaining_after_children - orphan_house_donation
  let wife_account := remaining_after_donation - final_amount
  (wife_account / income) * 100 = 33 := by sorry

end wife_account_percentage_l2664_266442


namespace sequence_property_l2664_266422

theorem sequence_property (a : ℕ → ℝ) 
  (h_pos : ∀ n : ℕ, n ≥ 1 → a n > 0)
  (h_ineq : ∀ n : ℕ, n ≥ 1 → (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)) :
  a 21022 ≤ 1 :=
sorry

end sequence_property_l2664_266422


namespace parabola_directrix_parameter_l2664_266452

/-- Given a parabola with equation x² = ay and directrix y = 1, prove that a = -4 -/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- Parabola equation
  (1 = -a/4) →              -- Relation between 'a' and directrix
  a = -4 := by
sorry

end parabola_directrix_parameter_l2664_266452


namespace seven_eighths_of_48_l2664_266417

theorem seven_eighths_of_48 : (7 : ℚ) / 8 * 48 = 42 := by
  sorry

end seven_eighths_of_48_l2664_266417


namespace ethan_reading_pages_l2664_266472

theorem ethan_reading_pages : 
  ∀ (total_pages saturday_morning sunday_pages pages_left saturday_night : ℕ),
  total_pages = 360 →
  saturday_morning = 40 →
  sunday_pages = 2 * (saturday_morning + saturday_night) →
  pages_left = 210 →
  total_pages = saturday_morning + saturday_night + sunday_pages + pages_left →
  saturday_night = 10 := by
sorry

end ethan_reading_pages_l2664_266472


namespace age_ratio_sachin_rahul_l2664_266406

/-- Proves that the ratio of Sachin's age to Rahul's age is 7:9 given their age difference --/
theorem age_ratio_sachin_rahul :
  ∀ (sachin_age rahul_age : ℚ),
    sachin_age = 31.5 →
    rahul_age = sachin_age + 9 →
    ∃ (a b : ℕ), a = 7 ∧ b = 9 ∧ sachin_age / rahul_age = a / b := by
  sorry

end age_ratio_sachin_rahul_l2664_266406


namespace shirt_cost_equation_l2664_266419

theorem shirt_cost_equation (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive (number of shirts can't be negative or zero)
  (h2 : 1.5 * x > 0) -- Ensure 1.5x is positive
  (h3 : 7800 = (1.5 * x) * ((6400 / x) - 30)) -- Total cost of type A shirts
  (h4 : 6400 = x * (6400 / x)) -- Total cost of type B shirts
  : 7800 / (1.5 * x) + 30 = 6400 / x := by
  sorry

#check shirt_cost_equation

end shirt_cost_equation_l2664_266419


namespace overlap_squares_area_l2664_266445

/-- Given two identical squares with side length 12 that overlap to form a 12 by 20 rectangle,
    the area of the non-overlapping region of one square is 48. -/
theorem overlap_squares_area (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 12 →
  rect_length = 20 →
  rect_width = 12 →
  (2 * square_side^2) - (rect_length * rect_width) = 48 := by
  sorry

end overlap_squares_area_l2664_266445


namespace muffin_count_l2664_266449

/-- Given a ratio of doughnuts to cookies to muffins and the number of doughnuts,
    calculate the number of muffins. -/
def calculate_muffins (doughnut_ratio : ℕ) (cookie_ratio : ℕ) (muffin_ratio : ℕ) (num_doughnuts : ℕ) : ℕ :=
  (num_doughnuts / doughnut_ratio) * muffin_ratio

/-- Theorem stating that given the ratio 5:3:1 for doughnuts:cookies:muffins
    and 50 doughnuts, there are 10 muffins. -/
theorem muffin_count : calculate_muffins 5 3 1 50 = 10 := by
  sorry

#eval calculate_muffins 5 3 1 50

end muffin_count_l2664_266449


namespace road_trip_duration_l2664_266432

/-- Road trip duration calculation -/
theorem road_trip_duration (jenna_distance : ℝ) (friend_distance : ℝ) 
  (jenna_speed : ℝ) (friend_speed : ℝ) (num_breaks : ℕ) (break_duration : ℝ) :
  jenna_distance = 200 →
  friend_distance = 100 →
  jenna_speed = 50 →
  friend_speed = 20 →
  num_breaks = 2 →
  break_duration = 0.5 →
  (jenna_distance / jenna_speed) + (friend_distance / friend_speed) + 
    (num_breaks : ℝ) * break_duration = 10 := by
  sorry

#check road_trip_duration

end road_trip_duration_l2664_266432


namespace bananas_profit_theorem_l2664_266456

/-- The number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 84

/-- The purchase price in dollars for 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- The selling price in dollars for 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- The total profit in dollars -/
def total_profit : ℝ := 7.00

/-- Theorem stating that the number of pounds of bananas purchased is correct -/
theorem bananas_profit_theorem :
  bananas_purchased * (selling_price / 4 - purchase_price / 3) = total_profit :=
by sorry

end bananas_profit_theorem_l2664_266456


namespace min_honey_amount_l2664_266448

theorem min_honey_amount (o h : ℝ) : 
  (o ≥ 8 + h / 3 ∧ o ≤ 3 * h) → h ≥ 3 := by
  sorry

end min_honey_amount_l2664_266448


namespace largest_consecutive_sum_28_l2664_266468

/-- The sum of n consecutive positive integers starting from a -/
def sumConsecutive (n : ℕ) (a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate to check if a sequence of n consecutive integers starting from a sums to 28 -/
def isValidSequence (n : ℕ) (a : ℕ) : Prop :=
  a > 0 ∧ sumConsecutive n a = 28

theorem largest_consecutive_sum_28 :
  (∃ a : ℕ, isValidSequence 7 a) ∧
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ, isValidSequence n a) :=
sorry

end largest_consecutive_sum_28_l2664_266468


namespace cosine_inequality_equivalence_l2664_266420

theorem cosine_inequality_equivalence (x : Real) : 
  (∃ k : Int, (-π/6 + 2*π*k < x ∧ x < π/6 + 2*π*k) ∨ 
              (2*π/3 + 2*π*k < x ∧ x < 4*π/3 + 2*π*k)) ↔ 
  (Real.cos (2*x) - 4*Real.cos (π/4)*Real.cos (5*π/12)*Real.cos x + 
   Real.cos (5*π/6) + 1 > 0) := by
  sorry

end cosine_inequality_equivalence_l2664_266420


namespace exam_correct_answers_l2664_266413

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : Nat
  correct_score : Int
  wrong_score : Int

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  total_score : Int

/-- Calculates the number of correctly answered questions. -/
def correct_answers (result : ExamResult) : Nat :=
  sorry

/-- Theorem stating that for the given exam conditions, 
    the number of correct answers is 42. -/
theorem exam_correct_answers 
  (e : Exam) 
  (r : ExamResult) 
  (h1 : e.total_questions = 80) 
  (h2 : e.correct_score = 4) 
  (h3 : e.wrong_score = -1) 
  (h4 : r.exam = e) 
  (h5 : r.total_score = 130) : 
  correct_answers r = 42 := by
  sorry

end exam_correct_answers_l2664_266413


namespace cake_surface_area_change_l2664_266450

/-- The change in surface area of a cylindrical cake after removing a cube from its top center -/
theorem cake_surface_area_change 
  (h : ℝ) -- height of the cylinder
  (r : ℝ) -- radius of the cylinder
  (s : ℝ) -- side length of the cube
  (h_pos : h > 0)
  (r_pos : r > 0)
  (s_pos : s > 0)
  (h_val : h = 5)
  (r_val : r = 2)
  (s_val : s = 1)
  (h_ge_s : h ≥ s) -- ensure the cube fits in the cylinder
  : (2 * π * r * s + s^2) - s^2 = 5 :=
sorry

end cake_surface_area_change_l2664_266450


namespace decomposition_fifth_power_fourth_l2664_266436

-- Define the function that gives the starting odd number for m^n
def startOdd (m n : ℕ) : ℕ := 
  2 * (m - 1) * (n - 1) + 1

-- Define the function that gives the k-th odd number in the sequence
def kthOdd (start k : ℕ) : ℕ := 
  start + 2 * (k - 1)

-- Theorem statement
theorem decomposition_fifth_power_fourth (m : ℕ) (h : m = 5) : 
  kthOdd (startOdd m 4) 3 = 125 := by
sorry

end decomposition_fifth_power_fourth_l2664_266436


namespace abs_value_complex_l2664_266427

/-- The absolute value of ((1+i)³)/2 is equal to √2 -/
theorem abs_value_complex : Complex.abs ((1 + Complex.I) ^ 3 / 2) = Real.sqrt 2 := by
  sorry

end abs_value_complex_l2664_266427
