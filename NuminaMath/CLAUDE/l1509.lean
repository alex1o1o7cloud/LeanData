import Mathlib

namespace NUMINAMATH_CALUDE_wayne_shrimp_cocktail_l1509_150930

/-- Calculates the number of shrimp served per guest given the total spent, cost per pound, shrimp per pound, and number of guests. -/
def shrimp_per_guest (total_spent : ℚ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) (num_guests : ℕ) : ℚ :=
  (total_spent / cost_per_pound * shrimp_per_pound) / num_guests

/-- Proves that Wayne plans to serve 5 shrimp per guest given the problem conditions. -/
theorem wayne_shrimp_cocktail :
  let total_spent : ℚ := 170
  let cost_per_pound : ℚ := 17
  let shrimp_per_pound : ℕ := 20
  let num_guests : ℕ := 40
  shrimp_per_guest total_spent cost_per_pound shrimp_per_pound num_guests = 5 := by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_cocktail_l1509_150930


namespace NUMINAMATH_CALUDE_f_properties_l1509_150936

/-- The function f(x) = x³ - 3mx + n --/
def f (m n x : ℝ) : ℝ := x^3 - 3*m*x + n

/-- Theorem stating the values of m and n, and the extrema in [0,3] --/
theorem f_properties (m n : ℝ) (hm : m > 0) 
  (hmax : ∃ x, ∀ y, f m n y ≤ f m n x)
  (hmin : ∃ x, ∀ y, f m n x ≤ f m n y)
  (hmax_val : ∃ x, f m n x = 6)
  (hmin_val : ∃ x, f m n x = 2) :
  m = 1 ∧ n = 4 ∧ 
  (∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, f 1 4 y ≤ f 1 4 x) ∧
  (∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, f 1 4 x ≤ f 1 4 y) ∧
  (∃ x ∈ Set.Icc 0 3, f 1 4 x = 2) ∧
  (∃ x ∈ Set.Icc 0 3, f 1 4 x = 22) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1509_150936


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1509_150932

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 18 * b^4 + 72 * c^4 + 1 / (27 * a * b * c) ≥ 4 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 18 * b^4 + 72 * c^4 + 1 / (27 * a * b * c) = 4 ↔
  a = ((9/4)^(1/4) * (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3)) ∧
  b = (2^(1/4) * (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3)) ∧
  c = (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1509_150932


namespace NUMINAMATH_CALUDE_total_sales_correct_l1509_150968

/-- Calculates the total amount of money made from selling bracelets and necklaces. -/
def calculate_total_sales (
  bracelet_price : ℕ)
  (bracelet_discount_price : ℕ)
  (necklace_price : ℕ)
  (necklace_discount_price : ℕ)
  (regular_bracelets_sold : ℕ)
  (discounted_bracelets_sold : ℕ)
  (regular_necklaces_sold : ℕ)
  (discounted_necklace_sets_sold : ℕ) : ℕ :=
  (regular_bracelets_sold * bracelet_price) +
  (discounted_bracelets_sold / 2 * bracelet_discount_price) +
  (regular_necklaces_sold * necklace_price) +
  (discounted_necklace_sets_sold * necklace_discount_price)

theorem total_sales_correct :
  calculate_total_sales 5 8 10 25 12 12 8 2 = 238 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_correct_l1509_150968


namespace NUMINAMATH_CALUDE_fred_balloons_l1509_150938

theorem fred_balloons (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 709 → given_away = 221 → remaining = initial - given_away → remaining = 488 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloons_l1509_150938


namespace NUMINAMATH_CALUDE_least_four_digit_9_heavy_l1509_150996

def is_9_heavy (n : ℕ) : Prop := n % 9 = 6

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_four_digit_9_heavy : 
  (∀ m : ℕ, is_four_digit m → is_9_heavy m → 1005 ≤ m) ∧ 
  is_four_digit 1005 ∧ 
  is_9_heavy 1005 := by sorry

end NUMINAMATH_CALUDE_least_four_digit_9_heavy_l1509_150996


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1509_150937

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a - 2 * Complex.I) / (1 + 2 * Complex.I)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1509_150937


namespace NUMINAMATH_CALUDE_sum_of_two_integers_l1509_150972

theorem sum_of_two_integers (x y : ℤ) : x = 32 → y = 2 * x → x + y = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_integers_l1509_150972


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1509_150988

theorem arithmetic_sqrt_of_nine : ∃ x : ℝ, x ≥ 0 ∧ x ^ 2 = 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1509_150988


namespace NUMINAMATH_CALUDE_sequence_general_term_l1509_150977

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℝ := 4 * n^2 + 2 * n

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 8 * n - 2

/-- Theorem stating that the given general term formula is correct -/
theorem sequence_general_term (n : ℕ) : 
  S n - S (n - 1) = a n := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1509_150977


namespace NUMINAMATH_CALUDE_smallest_integer_linear_combination_l1509_150941

theorem smallest_integer_linear_combination (m n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 5013 * a + 111111 * b) ∧
  ∀ (l : ℕ), l > 0 → (∃ (c d : ℤ), l = 5013 * c + 111111 * d) → k ≤ l :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_linear_combination_l1509_150941


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1509_150904

theorem arithmetic_equality : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1509_150904


namespace NUMINAMATH_CALUDE_intersection_M_N_l1509_150943

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = Set.Icc 2 5 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1509_150943


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_five_l1509_150931

theorem polynomial_value_at_negative_five (a b c : ℝ) : 
  (5^5 * a + 5^3 * b + 5 * c + 2 = 8) → 
  ((-5)^5 * a + (-5)^3 * b + (-5) * c - 3 = -9) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_five_l1509_150931


namespace NUMINAMATH_CALUDE_segment_area_approx_l1509_150900

/-- Represents a circular segment -/
structure CircularSegment where
  arcLength : ℝ
  chordLength : ℝ

/-- Calculates the area of a circular segment -/
noncomputable def segmentArea (segment : CircularSegment) : ℝ :=
  sorry

/-- Theorem stating that the area of the given circular segment is approximately 14.6 -/
theorem segment_area_approx :
  let segment : CircularSegment := { arcLength := 10, chordLength := 8 }
  abs (segmentArea segment - 14.6) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_segment_area_approx_l1509_150900


namespace NUMINAMATH_CALUDE_f_composition_minus_one_l1509_150956

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_composition_minus_one : f (f (-1)) = 5 := by sorry

end NUMINAMATH_CALUDE_f_composition_minus_one_l1509_150956


namespace NUMINAMATH_CALUDE_hexagon_circumradius_theorem_l1509_150926

-- Define a hexagon as a set of 6 points in 2D space
def Hexagon : Type := Fin 6 → ℝ × ℝ

-- Define the property of being convex for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define the property that all sides of the hexagon have length 1
def all_sides_unit_length (h : Hexagon) : Prop := sorry

-- Define the circumradius of a triangle given by three points
def circumradius (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hexagon_circumradius_theorem (h : Hexagon) 
  (convex : is_convex h) 
  (unit_sides : all_sides_unit_length h) : 
  max (circumradius (h 0) (h 2) (h 4)) (circumradius (h 1) (h 3) (h 5)) ≥ 1 := by sorry

end NUMINAMATH_CALUDE_hexagon_circumradius_theorem_l1509_150926


namespace NUMINAMATH_CALUDE_V_min_at_2_minus_sqrt2_l1509_150925

open Real

/-- The volume function V(a) -/
noncomputable def V (a : ℝ) : ℝ := 
  π * ((3-a) * (log (3-a))^2 + 2*a * log (3-a) - (1-a) * (log (1-a))^2 - 2*a * log (1-a))

/-- The theorem stating that V(a) has a minimum at a = 2 - √2 -/
theorem V_min_at_2_minus_sqrt2 :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ 
  (∀ (x : ℝ), 0 < x → x < 1 → V x ≥ V a) ∧ 
  a = 2 - sqrt 2 := by
  sorry

/-- Verify that 2 - √2 is indeed between 0 and 1 -/
lemma two_minus_sqrt_two_in_range : 
  0 < 2 - sqrt 2 ∧ 2 - sqrt 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_V_min_at_2_minus_sqrt2_l1509_150925


namespace NUMINAMATH_CALUDE_remaining_income_percentage_l1509_150978

theorem remaining_income_percentage (total_income : ℝ) (food_percentage : ℝ) (education_percentage : ℝ) (rent_percentage : ℝ) :
  food_percentage = 35 →
  education_percentage = 25 →
  rent_percentage = 80 →
  total_income > 0 →
  let remaining_after_food_education := total_income * (1 - (food_percentage + education_percentage) / 100)
  let remaining_after_rent := remaining_after_food_education * (1 - rent_percentage / 100)
  remaining_after_rent / total_income = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_remaining_income_percentage_l1509_150978


namespace NUMINAMATH_CALUDE_professor_chair_selections_eq_24_l1509_150947

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 11

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the minimum number of chairs between professors -/
def min_separation : ℕ := 2

/-- Calculates the number of ways to select chairs for professors -/
def professor_chair_selections : ℕ := sorry

/-- Theorem stating that the number of ways to select chairs for professors is 24 -/
theorem professor_chair_selections_eq_24 :
  professor_chair_selections = 24 := by sorry

end NUMINAMATH_CALUDE_professor_chair_selections_eq_24_l1509_150947


namespace NUMINAMATH_CALUDE_min_value_of_function_l1509_150933

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ y_min : ℝ, y_min = 7 ∧ ∀ y : ℝ, y = 4*x + 1/(4*x - 5) → y ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1509_150933


namespace NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l1509_150917

theorem a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ ¬(∀ a b : ℝ, a^2 > b^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l1509_150917


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1509_150986

theorem min_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x + 8*y + 1 = 0) :
  ∃ (m : ℝ), m = -2*Real.sqrt 2 - 1 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 - 2*a + 8*b + 1 = 0 → m ≤ a + 2*b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1509_150986


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1509_150959

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1509_150959


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1509_150928

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 5 * x - 12 = 0 ↔ x = 3 ∨ x = -4/3) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1509_150928


namespace NUMINAMATH_CALUDE_matrix_power_2019_l1509_150998

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2019 :
  A ^ 2019 = !![1, 0; 4038, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2019_l1509_150998


namespace NUMINAMATH_CALUDE_horner_method_correct_l1509_150914

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 -/
def f : ℝ → ℝ := fun x => x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

theorem horner_method_correct :
  let coeffs := [1, -5, 6, 0, 1, 0.3, 2]
  horner_eval coeffs (-2) = f (-2) ∧ f (-2) = 325.4 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_correct_l1509_150914


namespace NUMINAMATH_CALUDE_chessboard_squares_l1509_150994

/-- The number of squares of a given size in an 8x8 chessboard -/
def squares_of_size (n : Nat) : Nat :=
  (9 - n) ^ 2

/-- The total number of squares in an 8x8 chessboard -/
def total_squares : Nat :=
  (Finset.range 8).sum squares_of_size

theorem chessboard_squares :
  total_squares = 204 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_squares_l1509_150994


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1509_150945

/-- A hyperbola with center at the origin, one focus at (-√5, 0), and a point P such that
    the midpoint of PF₁ is at (0, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (P : ℝ × ℝ) : 
  (∃ (x y : ℝ), P = (x, y) ∧ x^2 - y^2/4 = 1) ↔ 
  (∃ (x y : ℝ), P = (x, y) ∧ 
    -- P is on the hyperbola
    (x - (-Real.sqrt 5))^2 + y^2 = (x - Real.sqrt 5)^2 + y^2 ∧ 
    -- Midpoint of PF₁ is (0, 2)
    ((x + (-Real.sqrt 5))/2 = 0 ∧ (y + 0)/2 = 2)) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l1509_150945


namespace NUMINAMATH_CALUDE_min_value_theorem_l1509_150913

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 ∧ ∃ (a' b' c' : ℝ), a'^2 + 2*a'*b' + b'^2 + 3*c'^2 = 324 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1509_150913


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1509_150961

/-- For an arithmetic sequence {a_n} with sum S_n = 2n - 1, prove the common ratio is 2 -/
theorem arithmetic_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = 2 * n - 1) 
  (h2 : ∀ n, S n = n * a 1) : 
  (a 2) / (a 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1509_150961


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1509_150918

theorem smallest_integer_with_remainder (n : ℕ) : n = 170 ↔ 
  (n > 1 ∧ 
   n % 3 = 2 ∧ 
   n % 7 = 2 ∧ 
   n % 8 = 2 ∧ 
   ∀ m : ℕ, m > 1 → m % 3 = 2 → m % 7 = 2 → m % 8 = 2 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1509_150918


namespace NUMINAMATH_CALUDE_seating_arrangements_l1509_150973

-- Define the number of seats, adults, and children
def numSeats : ℕ := 6
def numAdults : ℕ := 3
def numChildren : ℕ := 3

-- Define a function to calculate permutations
def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

-- Theorem statement
theorem seating_arrangements :
  2 * (permutations numAdults numAdults) * (permutations numChildren numChildren) = 72 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1509_150973


namespace NUMINAMATH_CALUDE_james_sticker_payment_ratio_l1509_150944

/-- Proves that the ratio of James's payment to the total cost of stickers is 1/2 -/
theorem james_sticker_payment_ratio :
  let num_packs : ℕ := 4
  let stickers_per_pack : ℕ := 30
  let cost_per_sticker : ℚ := 1/10
  let james_payment : ℚ := 6
  let total_stickers : ℕ := num_packs * stickers_per_pack
  let total_cost : ℚ := (total_stickers : ℚ) * cost_per_sticker
  james_payment / total_cost = 1/2 := by
sorry


end NUMINAMATH_CALUDE_james_sticker_payment_ratio_l1509_150944


namespace NUMINAMATH_CALUDE_fair_rides_calculation_fair_rides_proof_l1509_150919

/-- Calculates the number of rides taken by each person at a fair given specific conditions. -/
theorem fair_rides_calculation (entrance_fee_under_18 : ℚ) (ride_cost : ℚ) 
  (total_spent : ℚ) (num_people : ℕ) : ℚ :=
  let entrance_fee_18_plus := entrance_fee_under_18 * (1 + 1/5)
  let total_entrance_fee := entrance_fee_18_plus + 2 * entrance_fee_under_18
  let rides_cost := total_spent - total_entrance_fee
  let total_rides := rides_cost / ride_cost
  total_rides / num_people

/-- Proves that under the given conditions, each person took 3 rides. -/
theorem fair_rides_proof :
  fair_rides_calculation 5 (1/2) (41/2) 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fair_rides_calculation_fair_rides_proof_l1509_150919


namespace NUMINAMATH_CALUDE_simultaneous_arrivals_l1509_150922

/-- The distance between points A and B in meters -/
def distance : ℕ := 2010

/-- The speed of the m-th messenger in meters per minute -/
def speed (m : ℕ) : ℕ := m

/-- The time taken by the m-th messenger to reach point B -/
def time (m : ℕ) : ℚ := distance / m

/-- The total number of messengers -/
def total_messengers : ℕ := distance

/-- Predicate for whether two messengers arrive simultaneously -/
def arrive_simultaneously (m n : ℕ) : Prop :=
  1 ≤ m ∧ m < n ∧ n ≤ total_messengers ∧ time m = time n

theorem simultaneous_arrivals :
  ∀ m n : ℕ, arrive_simultaneously m n ↔ m * n = distance ∧ 1 ≤ m ∧ m < n ∧ n ≤ total_messengers :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_arrivals_l1509_150922


namespace NUMINAMATH_CALUDE_percentage_problem_l1509_150958

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 640 = 0.20 * 650 + 190 → P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1509_150958


namespace NUMINAMATH_CALUDE_range_of_x_in_triangle_l1509_150999

/-- Given a triangle ABC with vectors AB and AC, prove the range of x -/
theorem range_of_x_in_triangle (x : ℝ) : 
  let AB : ℝ × ℝ := (x, 2*x)
  let AC : ℝ × ℝ := (3*x, 2)
  -- Dot product is negative for obtuse angle
  (x * (3*x) + (2*x) * 2 < 0) →
  -- x is in the open interval (-4/3, 0)
  -4/3 < x ∧ x < 0 :=
by sorry


end NUMINAMATH_CALUDE_range_of_x_in_triangle_l1509_150999


namespace NUMINAMATH_CALUDE_even_product_probability_spinners_l1509_150993

/-- Represents a spinner with sections labeled by natural numbers -/
structure Spinner :=
  (sections : List ℕ)

/-- The probability of getting an even product when spinning two spinners -/
def evenProductProbability (spinnerA spinnerB : Spinner) : ℚ :=
  sorry

/-- Spinner A with 6 equal sections: 1, 1, 2, 2, 3, 4 -/
def spinnerA : Spinner :=
  ⟨[1, 1, 2, 2, 3, 4]⟩

/-- Spinner B with 4 equal sections: 1, 3, 5, 6 -/
def spinnerB : Spinner :=
  ⟨[1, 3, 5, 6]⟩

/-- Theorem stating that the probability of getting an even product
    when spinning spinnerA and spinnerB is 5/8 -/
theorem even_product_probability_spinners :
  evenProductProbability spinnerA spinnerB = 5/8 :=
sorry

end NUMINAMATH_CALUDE_even_product_probability_spinners_l1509_150993


namespace NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l1509_150964

theorem opposite_signs_abs_sum_less_abs_diff (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l1509_150964


namespace NUMINAMATH_CALUDE_two_tangent_circles_through_point_l1509_150962

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an angle formed by three points -/
structure Angle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Checks if a point is inside an angle -/
def isInsideAngle (M : Point) (angle : Angle) : Prop :=
  sorry

/-- Checks if a circle is tangent to both sides of an angle -/
def isTangentToAngle (circle : Circle) (angle : Angle) : Prop :=
  sorry

/-- Checks if a circle passes through a point -/
def passesThrough (circle : Circle) (point : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem two_tangent_circles_through_point 
  (angle : Angle) (M : Point) (h : isInsideAngle M angle) :
  ∃ (c1 c2 : Circle), 
    c1 ≠ c2 ∧ 
    isTangentToAngle c1 angle ∧ 
    isTangentToAngle c2 angle ∧ 
    passesThrough c1 M ∧ 
    passesThrough c2 M ∧
    ∀ (c : Circle), 
      isTangentToAngle c angle → 
      passesThrough c M → 
      (c = c1 ∨ c = c2) :=
  sorry

end NUMINAMATH_CALUDE_two_tangent_circles_through_point_l1509_150962


namespace NUMINAMATH_CALUDE_favorite_numbers_parity_l1509_150911

/-- Represents a person's favorite number -/
structure FavoriteNumber where
  value : ℤ

/-- Represents whether a number is even or odd -/
inductive Parity
  | Even
  | Odd

/-- Returns the parity of an integer -/
def parity (n : ℤ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- The problem setup -/
structure FavoriteNumbers where
  jan : FavoriteNumber
  dan : FavoriteNumber
  anna : FavoriteNumber
  hana : FavoriteNumber
  h1 : parity (dan.value + 3 * jan.value) = Parity.Odd
  h2 : parity ((anna.value - hana.value) * 5) = Parity.Odd
  h3 : parity (dan.value * hana.value + 17) = Parity.Even

/-- The main theorem to prove -/
theorem favorite_numbers_parity (nums : FavoriteNumbers) :
  parity nums.dan.value = Parity.Odd ∧
  parity nums.hana.value = Parity.Odd ∧
  parity nums.anna.value = Parity.Even ∧
  parity nums.jan.value = Parity.Even :=
sorry

end NUMINAMATH_CALUDE_favorite_numbers_parity_l1509_150911


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l1509_150974

/-- Given that real numbers 4, m, and 9 form a geometric sequence, 
    prove that the eccentricity of the conic section represented by 
    the equation x²/m + y² = 1 is either √(30)/6 or √7. -/
theorem conic_section_eccentricity (m : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ m = 4 * r ∧ 9 = m * r) →
  let e := if m > 0 
    then Real.sqrt (30) / 6 
    else Real.sqrt 7
  (∀ x y : ℝ, x^2 / m + y^2 = 1) →
  ∃ (a b c : ℝ), 
    (m > 0 → a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 - b^2 ∧ e = c / a) ∧
    (m < 0 → a^2 = 1 ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ e = c / a) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l1509_150974


namespace NUMINAMATH_CALUDE_complex_multiplication_l1509_150980

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1509_150980


namespace NUMINAMATH_CALUDE_primes_arithmetic_sequence_ones_digit_l1509_150906

/-- A function that returns the ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem primes_arithmetic_sequence_ones_digit 
  (p q r s : ℕ) 
  (hp : isPrime p) 
  (hq : isPrime q) 
  (hr : isPrime r) 
  (hs : isPrime s)
  (hseq : q = p + 8 ∧ r = q + 8 ∧ s = r + 8)
  (hp_gt_5 : p > 5) :
  onesDigit p = 3 := by
sorry

end NUMINAMATH_CALUDE_primes_arithmetic_sequence_ones_digit_l1509_150906


namespace NUMINAMATH_CALUDE_collection_for_44_members_l1509_150949

/-- Calculates the total collection amount in rupees for a group of students -/
def total_collection_rupees (num_members : ℕ) (paise_per_rupee : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / paise_per_rupee

/-- Proves that the total collection amount for 44 members is 19.36 rupees -/
theorem collection_for_44_members :
  total_collection_rupees 44 100 = 19.36 := by
  sorry

#eval total_collection_rupees 44 100

end NUMINAMATH_CALUDE_collection_for_44_members_l1509_150949


namespace NUMINAMATH_CALUDE_monkey_hop_distance_l1509_150946

/-- Represents the climbing problem of a monkey on a tree. -/
def monkey_climb (tree_height : ℝ) (total_hours : ℕ) (slip_distance : ℝ) (hop_distance : ℝ) : Prop :=
  let net_climb_per_hour := hop_distance - slip_distance
  (total_hours - 1 : ℝ) * net_climb_per_hour + hop_distance = tree_height

/-- Theorem stating that for the given conditions, the monkey must hop 3 feet each hour. -/
theorem monkey_hop_distance :
  monkey_climb 20 18 2 3 :=
sorry

end NUMINAMATH_CALUDE_monkey_hop_distance_l1509_150946


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_9_l1509_150934

theorem smallest_common_multiple_of_8_and_9 : ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 9 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 8 ∣ m ∧ 9 ∣ m) → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_9_l1509_150934


namespace NUMINAMATH_CALUDE_initial_observations_count_l1509_150923

theorem initial_observations_count (n : ℕ) 
  (h1 : (n : ℝ) > 0)
  (h2 : ∃ S : ℝ, S / n = 11)
  (h3 : ∃ new_obs : ℝ, (S + new_obs) / (n + 1) = 10)
  (h4 : new_obs = 4) :
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_initial_observations_count_l1509_150923


namespace NUMINAMATH_CALUDE_opposite_of_neg_nine_l1509_150983

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of -9 is 9
theorem opposite_of_neg_nine : opposite (-9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_nine_l1509_150983


namespace NUMINAMATH_CALUDE_cube_root_of_4x_plus_3y_is_3_l1509_150916

theorem cube_root_of_4x_plus_3y_is_3 (x y : ℝ) : 
  y = Real.sqrt (3 - x) + Real.sqrt (x - 3) + 5 → 
  (4 * x + 3 * y) ^ (1/3 : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_4x_plus_3y_is_3_l1509_150916


namespace NUMINAMATH_CALUDE_f_max_at_two_l1509_150975

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

-- State the theorem
theorem f_max_at_two :
  ∃ (max : ℝ), f 2 = max ∧ ∀ x, f x ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_f_max_at_two_l1509_150975


namespace NUMINAMATH_CALUDE_unique_digit_satisfying_conditions_l1509_150981

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

/-- Constructs a number in the form 282,1A4 given a digit A -/
def constructNumber (A : Digit) : ℕ := 282100 + 10 * A.val + 4

/-- The main theorem: there exists exactly one digit A satisfying both conditions -/
theorem unique_digit_satisfying_conditions : 
  ∃! (A : Digit), isDivisibleBy 75 A.val ∧ isDivisibleBy (constructNumber A) 4 :=
sorry

end NUMINAMATH_CALUDE_unique_digit_satisfying_conditions_l1509_150981


namespace NUMINAMATH_CALUDE_oatmeal_raisin_percentage_l1509_150955

/-- Given a class of students and cookie distribution, calculate the percentage of students who want oatmeal raisin cookies. -/
theorem oatmeal_raisin_percentage 
  (total_students : ℕ) 
  (cookies_per_student : ℕ) 
  (oatmeal_raisin_cookies : ℕ) 
  (h1 : total_students = 40)
  (h2 : cookies_per_student = 2)
  (h3 : oatmeal_raisin_cookies = 8) : 
  (oatmeal_raisin_cookies / cookies_per_student) / total_students * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_raisin_percentage_l1509_150955


namespace NUMINAMATH_CALUDE_vector_magnitude_l1509_150960

def problem (m n : ℝ × ℝ) : Prop :=
  let ⟨mx, my⟩ := m
  let ⟨nx, ny⟩ := n
  (mx * nx + my * ny = 0) ∧  -- m perpendicular to n
  (m.1 - 2 * n.1 = 11) ∧     -- x-component of m - 2n = 11
  (m.2 - 2 * n.2 = -2) ∧     -- y-component of m - 2n = -2
  (mx^2 + my^2 = 25)         -- |m| = 5

theorem vector_magnitude (m n : ℝ × ℝ) :
  problem m n → n.1^2 + n.2^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1509_150960


namespace NUMINAMATH_CALUDE_derivative_special_function_l1509_150997

open Real

/-- The derivative of (1 + 8 cosh² x * ln(cosh x)) / (2 cosh² x) -/
theorem derivative_special_function (x : ℝ) :
  deriv (λ x => (1 + 8 * (cosh x)^2 * log (cosh x)) / (2 * (cosh x)^2)) x
  = (sinh x * (4 * (cosh x)^2 - 1)) / (cosh x)^3 :=
by sorry

end NUMINAMATH_CALUDE_derivative_special_function_l1509_150997


namespace NUMINAMATH_CALUDE_consecutive_values_exist_l1509_150905

/-- A polynomial that takes on three consecutive integer values at three consecutive integer points -/
def polynomial (a : ℤ) (x : ℤ) : ℤ := x^3 - 18*x^2 + a*x + 1784

theorem consecutive_values_exist :
  ∃ (k n : ℤ),
    polynomial a (k-1) = n-1 ∧
    polynomial a k = n ∧
    polynomial a (k+1) = n+1 :=
sorry

end NUMINAMATH_CALUDE_consecutive_values_exist_l1509_150905


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l1509_150935

theorem no_real_solutions_for_equation : 
  ¬∃ y : ℝ, (10 - y)^2 = 4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l1509_150935


namespace NUMINAMATH_CALUDE_original_price_calculation_l1509_150979

/-- 
Given an article sold at a 40% profit, where the profit amount is 700 (in some currency unit),
prove that the original price of the article is 1750 (in the same currency unit).
-/
theorem original_price_calculation (profit_percentage : ℝ) (profit_amount : ℝ) 
  (h1 : profit_percentage = 40) 
  (h2 : profit_amount = 700) : 
  ∃ (original_price : ℝ), 
    original_price * (1 + profit_percentage / 100) - original_price = profit_amount ∧ 
    original_price = 1750 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1509_150979


namespace NUMINAMATH_CALUDE_flying_scotsman_norwich_difference_l1509_150927

/-- Proves that Flying Scotsman had 20 more carriages than Norwich -/
theorem flying_scotsman_norwich_difference :
  let euston : ℕ := 130
  let norwich : ℕ := 100
  let total : ℕ := 460
  let norfolk : ℕ := euston - 20
  let flying_scotsman : ℕ := total - (euston + norfolk + norwich)
  flying_scotsman - norwich = 20 := by
  sorry

end NUMINAMATH_CALUDE_flying_scotsman_norwich_difference_l1509_150927


namespace NUMINAMATH_CALUDE_third_candidate_votes_correct_l1509_150942

/-- The number of votes received by the third candidate in an election with three candidates,
    where two candidates received 7636 and 11628 votes respectively,
    and the winning candidate got 54.336448598130836% of the total votes. -/
def third_candidate_votes : ℕ :=
  let total_votes : ℕ := 7636 + 11628 + 2136
  let winning_votes : ℕ := 11628
  let winning_percentage : ℚ := 54336448598130836 / 100000000000000000
  2136

theorem third_candidate_votes_correct :
  let total_votes : ℕ := 7636 + 11628 + third_candidate_votes
  let winning_votes : ℕ := 11628
  let winning_percentage : ℚ := 54336448598130836 / 100000000000000000
  (winning_votes : ℚ) / (total_votes : ℚ) = winning_percentage :=
by sorry

#eval third_candidate_votes

end NUMINAMATH_CALUDE_third_candidate_votes_correct_l1509_150942


namespace NUMINAMATH_CALUDE_height_of_congruent_triangles_l1509_150967

/-- Triangle congruence relation -/
def CongruentTriangles (t1 t2 : Type) : Prop := sorry

/-- Area of a triangle -/
def TriangleArea (t : Type) : ℝ := sorry

/-- Height of a triangle on a given side -/
def TriangleHeight (t : Type) (side : ℝ) : ℝ := sorry

/-- Side length of a triangle -/
def TriangleSide (t : Type) (side : String) : ℝ := sorry

theorem height_of_congruent_triangles 
  (ABC DEF : Type) 
  (h_cong : CongruentTriangles ABC DEF) 
  (h_side : TriangleSide ABC "AB" = TriangleSide DEF "DE" ∧ TriangleSide ABC "AB" = 4) 
  (h_area : TriangleArea DEF = 10) :
  TriangleHeight ABC (TriangleSide ABC "AB") = 5 := by
  sorry

end NUMINAMATH_CALUDE_height_of_congruent_triangles_l1509_150967


namespace NUMINAMATH_CALUDE_max_happy_monkeys_theorem_l1509_150929

/-- Represents the number of each fruit type available --/
structure FruitCounts where
  pears : ℕ
  bananas : ℕ
  peaches : ℕ
  tangerines : ℕ

/-- Represents the criteria for a monkey to be happy --/
def happy_monkey (fruits : FruitCounts) : Prop :=
  ∃ (a b c : ℕ), a + b + c = 3 ∧ a + b + c ≤ fruits.pears + fruits.bananas + fruits.peaches + fruits.tangerines

/-- The maximum number of happy monkeys given the fruit counts --/
def max_happy_monkeys (fruits : FruitCounts) : ℕ :=
  Nat.min ((fruits.pears + fruits.bananas + fruits.peaches) / 2) fruits.tangerines

/-- Theorem stating the maximum number of happy monkeys for the given fruit counts --/
theorem max_happy_monkeys_theorem (fruits : FruitCounts) :
  fruits.pears = 20 →
  fruits.bananas = 30 →
  fruits.peaches = 40 →
  fruits.tangerines = 50 →
  max_happy_monkeys fruits = 45 :=
by
  sorry

#eval max_happy_monkeys ⟨20, 30, 40, 50⟩

end NUMINAMATH_CALUDE_max_happy_monkeys_theorem_l1509_150929


namespace NUMINAMATH_CALUDE_first_car_speed_l1509_150920

/-- 
Given two cars starting from opposite ends of a highway, this theorem proves
that the speed of the first car is 25 mph under the given conditions.
-/
theorem first_car_speed 
  (highway_length : ℝ) 
  (second_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 175) 
  (h2 : second_car_speed = 45) 
  (h3 : meeting_time = 2.5) :
  ∃ (first_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    first_car_speed = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l1509_150920


namespace NUMINAMATH_CALUDE_ab_greater_than_ac_l1509_150907

theorem ab_greater_than_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_ac_l1509_150907


namespace NUMINAMATH_CALUDE_gabby_makeup_set_savings_l1509_150951

/-- Proves that Gabby needs $10 more to buy the makeup set -/
theorem gabby_makeup_set_savings (makeup_cost initial_savings mom_contribution : ℕ) 
  (h1 : makeup_cost = 65)
  (h2 : initial_savings = 35)
  (h3 : mom_contribution = 20) :
  makeup_cost - initial_savings - mom_contribution = 10 := by
  sorry

end NUMINAMATH_CALUDE_gabby_makeup_set_savings_l1509_150951


namespace NUMINAMATH_CALUDE_chessboard_tiling_l1509_150970

/-- Represents a chessboard configuration -/
inductive ChessboardConfig
  | OneCornerRemoved
  | TwoOppositeCorners
  | TwoNonOppositeCorners

/-- Represents whether a configuration is tileable or not -/
inductive Tileable
  | Yes
  | No

/-- Function to determine if a chessboard configuration is tileable with 2x1 dominoes -/
def isTileable (config : ChessboardConfig) : Tileable :=
  match config with
  | ChessboardConfig.OneCornerRemoved => Tileable.No
  | ChessboardConfig.TwoOppositeCorners => Tileable.No
  | ChessboardConfig.TwoNonOppositeCorners => Tileable.Yes

theorem chessboard_tiling :
  (isTileable ChessboardConfig.OneCornerRemoved = Tileable.No) ∧
  (isTileable ChessboardConfig.TwoOppositeCorners = Tileable.No) ∧
  (isTileable ChessboardConfig.TwoNonOppositeCorners = Tileable.Yes) := by
  sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l1509_150970


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1509_150924

/-- Given a geometric sequence {a_n} where a₇ = 1/4 and a₃a₅ = 4(a₄ - 1), prove that a₂ = 8 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m)
  (h_a7 : a 7 = 1 / 4)
  (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) :
  a 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1509_150924


namespace NUMINAMATH_CALUDE_die_roll_average_l1509_150992

def die_rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]
def next_roll : Nat := 2
def total_rolls : Nat := die_rolls.length + 1

theorem die_roll_average :
  (die_rolls.sum + next_roll) / total_rolls = 3 := by
sorry

end NUMINAMATH_CALUDE_die_roll_average_l1509_150992


namespace NUMINAMATH_CALUDE_factors_of_504_l1509_150971

def number_of_positive_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_504 : number_of_positive_factors 504 = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_504_l1509_150971


namespace NUMINAMATH_CALUDE_min_questions_to_determine_order_l1509_150921

/-- Represents a question that reveals the relative order of 50 numbers -/
def Question := Fin 100 → Prop

/-- The set of all possible permutations of numbers from 1 to 100 -/
def Permutations := Fin 100 → Fin 100

/-- A function that determines if a given permutation is consistent with the answers to all questions -/
def IsConsistent (p : Permutations) (qs : List Question) : Prop := sorry

/-- The minimum number of questions needed to determine the order of 100 integers -/
def MinQuestions : ℕ := 5

theorem min_questions_to_determine_order :
  ∀ (qs : List Question),
    (∀ (p₁ p₂ : Permutations), IsConsistent p₁ qs ∧ IsConsistent p₂ qs → p₁ = p₂) →
    qs.length ≥ MinQuestions :=
sorry

end NUMINAMATH_CALUDE_min_questions_to_determine_order_l1509_150921


namespace NUMINAMATH_CALUDE_base_7_23456_equals_6068_l1509_150957

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_23456_equals_6068 :
  base_7_to_10 [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_7_23456_equals_6068_l1509_150957


namespace NUMINAMATH_CALUDE_C₁_is_unit_circle_intersection_point_C₁_k4_equation_l1509_150953

-- Define the curves C₁ and C₂
def C₁ (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.cos t ^ k ∧ p.2 = Real.sin t ^ k}

def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 16 * p.2 + 3 = 0}

-- Part 1: Prove that C₁ when k = 1 is a unit circle
theorem C₁_is_unit_circle :
  C₁ 1 = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} := by sorry

-- Part 2: Prove that (1/4, 1/4) is an intersection point of C₁ and C₂ when k = 4
theorem intersection_point :
  (1/4, 1/4) ∈ C₁ 4 ∧ (1/4, 1/4) ∈ C₂ := by sorry

-- Helper theorem: The equation of C₁ when k = 4 can be written as √x + √y = 1
theorem C₁_k4_equation (p : ℝ × ℝ) :
  p ∈ C₁ 4 ↔ Real.sqrt p.1 + Real.sqrt p.2 = 1 := by sorry

end NUMINAMATH_CALUDE_C₁_is_unit_circle_intersection_point_C₁_k4_equation_l1509_150953


namespace NUMINAMATH_CALUDE_kira_morning_downloads_l1509_150989

/-- The number of songs Kira downloaded in the morning -/
def morning_songs : ℕ := sorry

/-- The number of songs Kira downloaded later in the day -/
def afternoon_songs : ℕ := 15

/-- The number of songs Kira downloaded at night -/
def night_songs : ℕ := 3

/-- The size of each song in MB -/
def song_size : ℕ := 5

/-- The total memory space occupied by all songs in MB -/
def total_memory : ℕ := 140

theorem kira_morning_downloads : 
  morning_songs = 10 ∧ 
  song_size * (morning_songs + afternoon_songs + night_songs) = total_memory := by
  sorry

end NUMINAMATH_CALUDE_kira_morning_downloads_l1509_150989


namespace NUMINAMATH_CALUDE_cylinder_surface_area_from_hemisphere_l1509_150908

/-- Given a hemisphere with total surface area Q and a cylinder with the same base and volume,
    prove that the total surface area of the cylinder is (10/9)Q. -/
theorem cylinder_surface_area_from_hemisphere (Q : ℝ) (R : ℝ) (h : ℝ) :
  Q > 0 →  -- Ensure Q is positive
  Q = 3 * Real.pi * R^2 →  -- Total surface area of hemisphere
  h = (2/3) * R →  -- Height of cylinder with same volume
  (2 * Real.pi * R^2 + 2 * Real.pi * R * h) = (10/9) * Q := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_from_hemisphere_l1509_150908


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l1509_150948

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l1509_150948


namespace NUMINAMATH_CALUDE_dot_product_zero_l1509_150990

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![4, 3]

theorem dot_product_zero : 
  (Finset.sum Finset.univ (λ i => a i * (2 * a i - b i))) = 0 := by sorry

end NUMINAMATH_CALUDE_dot_product_zero_l1509_150990


namespace NUMINAMATH_CALUDE_same_color_probability_l1509_150940

/-- The probability of drawing two balls of the same color from an urn -/
theorem same_color_probability (w b r : ℕ) (hw : w = 4) (hb : b = 6) (hr : r = 5) :
  let total := w + b + r
  let p_white := (w / total) * ((w - 1) / (total - 1))
  let p_black := (b / total) * ((b - 1) / (total - 1))
  let p_red := (r / total) * ((r - 1) / (total - 1))
  p_white + p_black + p_red = 31 / 105 := by
  sorry

#check same_color_probability

end NUMINAMATH_CALUDE_same_color_probability_l1509_150940


namespace NUMINAMATH_CALUDE_living_room_set_cost_l1509_150903

/-- The total cost of a living room set -/
def total_cost (sofa_cost armchair_cost coffee_table_cost : ℕ) (num_armchairs : ℕ) : ℕ :=
  sofa_cost + num_armchairs * armchair_cost + coffee_table_cost

/-- Theorem: The total cost of the specified living room set is $2,430 -/
theorem living_room_set_cost : total_cost 1250 425 330 2 = 2430 := by
  sorry

end NUMINAMATH_CALUDE_living_room_set_cost_l1509_150903


namespace NUMINAMATH_CALUDE_equation_solution_l1509_150902

theorem equation_solution : 
  ∃! y : ℝ, (y^3 + 3*y^2) / (y^2 + 5*y + 6) + y = -8 ∧ y^2 + 5*y + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1509_150902


namespace NUMINAMATH_CALUDE_range_of_c_l1509_150995

/-- The statement of the theorem --/
theorem range_of_c (c : ℝ) : c > 0 ∧ c ≠ 1 →
  (((∀ x y : ℝ, x < y → c^x > c^y) ∨ (∀ x : ℝ, x + |x - 2*c| > 1)) ∧
   ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧ (∀ x : ℝ, x + |x - 2*c| > 1))) ↔
  (c ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l1509_150995


namespace NUMINAMATH_CALUDE_percentage_calculation_l1509_150965

theorem percentage_calculation : (200 / 50) * 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1509_150965


namespace NUMINAMATH_CALUDE_probability_properties_l1509_150939

theorem probability_properties (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 0.5) (hB : P B = 0.3) :
  (∀ h : A ∩ B = ∅, P (A ∪ B) = 0.8) ∧ 
  (∀ h : P (A ∩ B) = P A * P B, P (A ∪ B) = 0.65) ∧
  (∀ h : P (B ∩ A) / P A = 0.5, P (B ∩ Aᶜ) / P Aᶜ = 0.1) := by
  sorry

end NUMINAMATH_CALUDE_probability_properties_l1509_150939


namespace NUMINAMATH_CALUDE_rachel_money_left_l1509_150912

theorem rachel_money_left (initial_amount : ℚ) : 
  initial_amount = 200 →
  initial_amount - (initial_amount / 4 + initial_amount / 5 + initial_amount / 10 + initial_amount / 8) = 65 := by
  sorry

end NUMINAMATH_CALUDE_rachel_money_left_l1509_150912


namespace NUMINAMATH_CALUDE_complex_multiplication_l1509_150909

theorem complex_multiplication : (1 + Complex.I) ^ 6 * (1 - Complex.I) = -8 - 8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1509_150909


namespace NUMINAMATH_CALUDE_tim_blue_marbles_l1509_150982

theorem tim_blue_marbles (fred_marbles : ℕ) (fred_tim_ratio : ℕ) (h1 : fred_marbles = 385) (h2 : fred_tim_ratio = 35) :
  fred_marbles / fred_tim_ratio = 11 := by
  sorry

end NUMINAMATH_CALUDE_tim_blue_marbles_l1509_150982


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1509_150952

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 4 * n = k ^ 2) ∧ 
    (∃ (m : ℕ), 3 * n = m ^ 3)) ∧
  (∀ (n : ℕ), n > 0 ∧ n < 144 → 
    ¬(∃ (k : ℕ), 4 * n = k ^ 2) ∨ 
    ¬(∃ (m : ℕ), 3 * n = m ^ 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1509_150952


namespace NUMINAMATH_CALUDE_books_difference_l1509_150991

theorem books_difference (bobby_books kristi_books : ℕ) 
  (h1 : bobby_books = 142) 
  (h2 : kristi_books = 78) : 
  bobby_books - kristi_books = 64 := by
sorry

end NUMINAMATH_CALUDE_books_difference_l1509_150991


namespace NUMINAMATH_CALUDE_sin_510_degrees_l1509_150954

theorem sin_510_degrees : Real.sin (510 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_510_degrees_l1509_150954


namespace NUMINAMATH_CALUDE_pair_probability_after_removal_l1509_150915

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset ℕ)
  (count : ℕ → ℕ)
  (total : ℕ)

/-- Initial deck configuration -/
def initial_deck : Deck :=
  { cards := Finset.range 12,
    count := λ n => if n ∈ Finset.range 12 then 4 else 0,
    total := 48 }

/-- Deck after removing two pairs -/
def deck_after_removal (d : Deck) : Deck :=
  { cards := d.cards,
    count := λ n => if n ∈ d.cards then d.count n - 2 else 0,
    total := d.total - 4 }

/-- Number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Number of ways to form pairs from remaining cards -/
def pair_ways (d : Deck) : ℕ :=
  (d.cards.filter (λ n => d.count n = 4)).card * 6 +
  (d.cards.filter (λ n => d.count n = 2)).card * 1

/-- Probability of selecting a pair -/
def pair_probability (d : Deck) : ℚ :=
  pair_ways d / choose_two d.total

theorem pair_probability_after_removal :
  pair_probability (deck_after_removal initial_deck) = 31 / 473 := by
  sorry

#eval pair_probability (deck_after_removal initial_deck)

end NUMINAMATH_CALUDE_pair_probability_after_removal_l1509_150915


namespace NUMINAMATH_CALUDE_equation_solution_l1509_150987

theorem equation_solution : ∃ (z₁ z₂ : ℂ), 
  z₁ = -1 + Complex.I ∧ 
  z₂ = -1 - Complex.I ∧ 
  (∀ x : ℂ, x ≠ -2 → -x^3 = (4*x + 2)/(x + 2) ↔ (x = z₁ ∨ x = z₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1509_150987


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1509_150950

/-- An arithmetic sequence with first term 0 and non-zero common difference -/
structure ArithmeticSequence where
  d : ℝ
  hd : d ≠ 0
  a : ℕ → ℝ
  h_init : a 1 = 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  ∃ m : ℕ, seq.a m = seq.a 1 + seq.a 2 + seq.a 3 + seq.a 4 + seq.a 5 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1509_150950


namespace NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l1509_150985

theorem x_squared_plus_7x_plus_12_bounds 
  (x : ℝ) (h : x^2 - 7*x + 12 < 0) : 
  48 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l1509_150985


namespace NUMINAMATH_CALUDE_hockey_league_games_l1509_150976

theorem hockey_league_games (n : ℕ) (m : ℕ) (total_games : ℕ) : 
  n = 25 → -- number of teams
  m = 15 → -- number of times each team faces every other team
  total_games = (n * (n - 1) / 2) * m →
  total_games = 4500 :=
by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1509_150976


namespace NUMINAMATH_CALUDE_soccer_balls_theorem_l1509_150966

/-- The number of soccer balls originally purchased by the store -/
def original_balls : ℕ := 130

/-- The wholesale price of each soccer ball -/
def wholesale_price : ℕ := 30

/-- The retail price of each soccer ball -/
def retail_price : ℕ := 45

/-- The number of soccer balls remaining when the profit is calculated -/
def remaining_balls : ℕ := 30

/-- The profit made when there are 30 balls remaining -/
def profit : ℕ := 1500

/-- Theorem stating that the number of originally purchased soccer balls is 130 -/
theorem soccer_balls_theorem :
  (retail_price - wholesale_price) * (original_balls - remaining_balls) = profit :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_theorem_l1509_150966


namespace NUMINAMATH_CALUDE_lathe_probabilities_l1509_150984

-- Define the yield rates and processing percentages
def yield_rate_1 : ℝ := 0.15
def yield_rate_2 : ℝ := 0.10
def process_percent_1 : ℝ := 0.60
def process_percent_2 : ℝ := 0.40

-- Define the theorem
theorem lathe_probabilities :
  -- Probability of both lathes producing excellent parts simultaneously
  yield_rate_1 * yield_rate_2 = 0.015 ∧
  -- Probability of randomly selecting an excellent part from mixed parts
  process_percent_1 * yield_rate_1 + process_percent_2 * yield_rate_2 = 0.13 :=
by sorry

end NUMINAMATH_CALUDE_lathe_probabilities_l1509_150984


namespace NUMINAMATH_CALUDE_geometric_sequence_parabola_vertex_l1509_150901

-- Define a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) : Prop :=
  parabola x = y ∧ ∀ t : ℝ, parabola t ≥ y

-- Theorem statement
theorem geometric_sequence_parabola_vertex (a b c d : ℝ) :
  is_geometric_sequence a b c d →
  is_vertex b c →
  a * d = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_parabola_vertex_l1509_150901


namespace NUMINAMATH_CALUDE_given_program_has_syntax_error_l1509_150969

/-- Represents the structure of a DO-UNTIL loop -/
inductive DOUntilLoop
| correct : (body : String) → (condition : String) → DOUntilLoop
| incorrect : (body : String) → (untilKeyword : String) → (condition : String) → DOUntilLoop

/-- The given program structure -/
def givenProgram : DOUntilLoop :=
  DOUntilLoop.incorrect "x=x*x" "UNTIL" "x>10"

/-- Checks if a DO-UNTIL loop has correct syntax -/
def hasCorrectSyntax (loop : DOUntilLoop) : Prop :=
  match loop with
  | DOUntilLoop.correct _ _ => True
  | DOUntilLoop.incorrect _ _ _ => False

/-- Theorem stating that the given program has a syntax error -/
theorem given_program_has_syntax_error :
  ¬(hasCorrectSyntax givenProgram) := by
  sorry


end NUMINAMATH_CALUDE_given_program_has_syntax_error_l1509_150969


namespace NUMINAMATH_CALUDE_second_question_percentage_l1509_150910

/-- Represents the percentage of boys in a test scenario -/
structure TestPercentages where
  first : ℝ  -- Percentage who answered the first question correctly
  neither : ℝ  -- Percentage who answered neither question correctly
  both : ℝ  -- Percentage who answered both questions correctly

/-- 
Given the percentages of boys who answered the first question correctly, 
neither question correctly, and both questions correctly, 
proves that the percentage who answered the second question correctly is 55%.
-/
theorem second_question_percentage (p : TestPercentages) 
  (h1 : p.first = 75)
  (h2 : p.neither = 20)
  (h3 : p.both = 50) : 
  ∃ second : ℝ, second = 55 := by
  sorry

#check second_question_percentage

end NUMINAMATH_CALUDE_second_question_percentage_l1509_150910


namespace NUMINAMATH_CALUDE_sum_and_opposites_l1509_150963

theorem sum_and_opposites : 
  let a := -5
  let b := -2
  let c := abs b
  let d := 0
  (a + b + c + d = -5) ∧ 
  (- a = 5) ∧ 
  (- b = 2) ∧ 
  (- c = -2) ∧ 
  (- d = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_opposites_l1509_150963
