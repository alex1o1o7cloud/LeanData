import Mathlib

namespace NUMINAMATH_CALUDE_expression_factorization_l549_54982

theorem expression_factorization (x : ℝ) : 
  (21 * x^4 + 90 * x^3 + 40 * x - 10) - (7 * x^4 + 6 * x^3 + 8 * x - 6) = 
  2 * x * (7 * x^3 + 42 * x^2 + 16) - 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l549_54982


namespace NUMINAMATH_CALUDE_gcd_of_product_of_differences_l549_54929

theorem gcd_of_product_of_differences (a b c d : ℤ) : 
  ∃ (k : ℤ), (12 : ℤ) ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) ∧
  ∀ (m : ℤ), (∀ (x y z w : ℤ), m ∣ (x - y) * (x - z) * (x - w) * (y - z) * (y - w) * (z - w)) → m ∣ 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_product_of_differences_l549_54929


namespace NUMINAMATH_CALUDE_range_of_two_alpha_l549_54926

theorem range_of_two_alpha (α β : ℝ) 
  (h1 : π < α + β ∧ α + β < 4 / 3 * π)
  (h2 : -π < α - β ∧ α - β < -π / 3) :
  0 < 2 * α ∧ 2 * α < π :=
by sorry

end NUMINAMATH_CALUDE_range_of_two_alpha_l549_54926


namespace NUMINAMATH_CALUDE_max_dot_product_on_ellipses_l549_54974

def ellipse_C1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

def ellipse_C2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem max_dot_product_on_ellipses :
  ∀ x1 y1 x2 y2 : ℝ,
  ellipse_C1 x1 y1 → ellipse_C2 x2 y2 →
  dot_product x1 y1 x2 y2 ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_on_ellipses_l549_54974


namespace NUMINAMATH_CALUDE_p_is_8x_squared_minus_8_l549_54906

-- Define the numerator polynomial
def num (x : ℝ) : ℝ := x^4 - 2*x^3 - 7*x + 6

-- Define the properties of p(x)
def has_vertical_asymptotes (p : ℝ → ℝ) : Prop :=
  p 1 = 0 ∧ p (-1) = 0

def no_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, ∀ x : ℝ, ∃ c : ℝ, |p x| ≤ c * |x|^n

-- Main theorem
theorem p_is_8x_squared_minus_8 (p : ℝ → ℝ) :
  has_vertical_asymptotes p →
  no_horizontal_asymptote p →
  p 2 = 24 →
  ∀ x : ℝ, p x = 8*x^2 - 8 :=
by sorry

end NUMINAMATH_CALUDE_p_is_8x_squared_minus_8_l549_54906


namespace NUMINAMATH_CALUDE_inequality_solution_set_l549_54956

def f (x : ℝ) := abs x + abs (x - 4)

theorem inequality_solution_set :
  {x : ℝ | f (x^2 + 2) > f x} = {x | x < -2 ∨ x > Real.sqrt 2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l549_54956


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l549_54981

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 3)
  (h_a5 : a 5 = 9) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l549_54981


namespace NUMINAMATH_CALUDE_number_multiplied_by_five_thirds_l549_54976

theorem number_multiplied_by_five_thirds : ∃ x : ℚ, (5 : ℚ) / 3 * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_five_thirds_l549_54976


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l549_54907

theorem identity_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f (x - y)) + x = f (x + y)) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l549_54907


namespace NUMINAMATH_CALUDE_election_votes_total_l549_54959

theorem election_votes_total (V A B C : ℝ) 
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V) :
  V = 60000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_total_l549_54959


namespace NUMINAMATH_CALUDE_correct_calculation_l549_54983

theorem correct_calculation (x : ℝ) (h : 15 * x = 45) : 5 * x = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l549_54983


namespace NUMINAMATH_CALUDE_road_trip_distance_l549_54930

/-- Road trip problem -/
theorem road_trip_distance (total_distance michelle_distance : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : michelle_distance = 294)
  (h3 : ∃ (tracy_distance : ℕ), tracy_distance > 2 * michelle_distance)
  (h4 : ∃ (katie_distance : ℕ), michelle_distance = 3 * katie_distance) :
  ∃ (tracy_distance : ℕ), tracy_distance = total_distance - michelle_distance - (michelle_distance / 3) ∧ 
    tracy_distance - 2 * michelle_distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l549_54930


namespace NUMINAMATH_CALUDE_school_pupils_count_l549_54955

theorem school_pupils_count (girls : ℕ) (boys : ℕ) (teachers : ℕ) : girls = 308 → boys = 318 → teachers = 36 → girls + boys = 626 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_count_l549_54955


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l549_54925

def P (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_polynomial_property (a b : ℝ) :
  P a b 10 + P a b 30 = 40 → P a b 20 = -80 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l549_54925


namespace NUMINAMATH_CALUDE_valid_paintings_count_l549_54946

/-- Represents a color in the painting. -/
inductive Color
  | Green
  | Red
  | Blue

/-- Represents a position in the 3x3 grid. -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Represents a painting of the 3x3 grid. -/
def Painting := Position → Color

/-- Checks if a painting satisfies the color placement rules. -/
def validPainting (p : Painting) : Prop :=
  ∀ (pos : Position),
    (p pos = Color.Green →
      ∀ (above : Position), above.row = pos.row - 1 → above.col = pos.col → p above ≠ Color.Red) ∧
    (p pos = Color.Green →
      ∀ (right : Position), right.row = pos.row → right.col = pos.col + 1 → p right ≠ Color.Red) ∧
    (p pos = Color.Blue →
      ∀ (left : Position), left.row = pos.row → left.col = pos.col - 1 → p left ≠ Color.Red)

/-- The number of valid paintings. -/
def numValidPaintings : ℕ := sorry

theorem valid_paintings_count :
  numValidPaintings = 78 :=
sorry

end NUMINAMATH_CALUDE_valid_paintings_count_l549_54946


namespace NUMINAMATH_CALUDE_max_divisor_with_equal_remainders_l549_54953

theorem max_divisor_with_equal_remainders : 
  ∃ (k : ℕ), 
    (81849 % 243 = k) ∧ 
    (106392 % 243 = k) ∧ 
    (124374 % 243 = k) ∧ 
    (∀ m : ℕ, m > 243 → 
      ¬(∃ r : ℕ, (81849 % m = r) ∧ (106392 % m = r) ∧ (124374 % m = r))) := by
  sorry

end NUMINAMATH_CALUDE_max_divisor_with_equal_remainders_l549_54953


namespace NUMINAMATH_CALUDE_cricket_run_rate_proof_l549_54999

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let first_runs := (first_run_rate * first_overs : ℚ).floor
  let remaining_runs := target - first_runs
  (remaining_runs : ℚ) / remaining_overs

/-- Proves that the required run rate for the remaining 40 overs is 6.5 -/
theorem cricket_run_rate_proof :
  required_run_rate 50 10 (32/10) 292 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_proof_l549_54999


namespace NUMINAMATH_CALUDE_range_of_f_range_of_a_l549_54993

-- Define the function f
def f (x : ℝ) : ℝ := 2 * |x - 1| - |x - 4|

-- Theorem for the range of f
theorem range_of_f : Set.range f = Set.Ici (-3) := by sorry

-- Define the inequality function g
def g (x a : ℝ) : ℝ := 2 * |x - 1| - |x - a|

-- Theorem for the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, g x a ≥ -1) ↔ a ∈ Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_a_l549_54993


namespace NUMINAMATH_CALUDE_janet_final_lives_l549_54961

/-- Calculates the final number of lives for Janet in the video game --/
def final_lives (initial_lives : ℕ) (lives_lost : ℕ) (points_earned : ℕ) : ℕ :=
  let remaining_lives := initial_lives - lives_lost
  let lives_earned := (points_earned / 100) * 2
  let lives_lost_penalty := points_earned / 200
  remaining_lives + lives_earned - lives_lost_penalty

theorem janet_final_lives : 
  final_lives 47 23 1840 = 51 := by
  sorry

end NUMINAMATH_CALUDE_janet_final_lives_l549_54961


namespace NUMINAMATH_CALUDE_simplify_expression_l549_54943

theorem simplify_expression (x y : ℝ) :
  (25 * x + 70 * y) + (15 * x + 34 * y) - (13 * x + 55 * y) = 27 * x + 49 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l549_54943


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l549_54975

/-- Given a polynomial f(x) = ax^4 + bx^3 + cx^2 + dx + e where f(-3) = -5,
    prove that 8a - 4b + 2c - d + e = -5 -/
theorem polynomial_value_theorem (a b c d e : ℝ) :
  (fun x : ℝ ↦ a * x^4 + b * x^3 + c * x^2 + d * x + e) (-3) = -5 →
  8 * a - 4 * b + 2 * c - d + e = -5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l549_54975


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l549_54986

/-- Represents the number of cone types -/
def num_cone_types : ℕ := 2

/-- Represents the maximum number of scoops -/
def max_scoops : ℕ := 3

/-- Represents the number of ice cream flavors -/
def num_flavors : ℕ := 4

/-- Represents the number of topping choices -/
def num_toppings : ℕ := 4

/-- Represents the maximum number of toppings allowed -/
def max_toppings : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the total number of ice cream combinations -/
def total_combinations : ℕ := 
  let one_scoop := num_flavors
  let two_scoops := num_flavors + choose num_flavors 2
  let three_scoops := num_flavors + num_flavors * (num_flavors - 1) + choose num_flavors 3
  let scoop_combinations := one_scoop + two_scoops + three_scoops
  let topping_combinations := 1 + num_toppings + choose num_toppings 2
  num_cone_types * scoop_combinations * topping_combinations

/-- Theorem stating that the total number of ice cream combinations is 748 -/
theorem ice_cream_combinations : total_combinations = 748 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l549_54986


namespace NUMINAMATH_CALUDE_max_chords_for_ten_points_l549_54973

/-- Given n points on a circle, max_chords_no_triangle calculates the maximum number of chords
    that can be drawn between these points without forming any triangles. -/
def max_chords_no_triangle (n : ℕ) : ℕ :=
  (n^2) / 4

/-- Theorem stating that for 10 points on a circle, the maximum number of chords
    that can be drawn without forming triangles is 25. -/
theorem max_chords_for_ten_points :
  max_chords_no_triangle 10 = 25 :=
by sorry

end NUMINAMATH_CALUDE_max_chords_for_ten_points_l549_54973


namespace NUMINAMATH_CALUDE_symmetric_functions_property_l549_54902

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem symmetric_functions_property (h1 : ∀ x, f (x - 1) = g⁻¹ x) (h2 : g 2 = 0) : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_functions_property_l549_54902


namespace NUMINAMATH_CALUDE_cookies_eaten_yesterday_l549_54920

/-- Given the number of cookies eaten today and the difference between today and yesterday,
    calculate the number of cookies eaten yesterday. -/
def cookies_yesterday (today : ℕ) (difference : ℕ) : ℕ :=
  today - difference

/-- Theorem stating that given 140 cookies eaten today and 30 fewer yesterday,
    the number of cookies eaten yesterday was 110. -/
theorem cookies_eaten_yesterday :
  cookies_yesterday 140 30 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_yesterday_l549_54920


namespace NUMINAMATH_CALUDE_marcus_pretzels_l549_54971

theorem marcus_pretzels (total : ℕ) (john : ℕ) (alan : ℕ) (marcus : ℕ) 
  (h1 : total = 95)
  (h2 : john = 28)
  (h3 : alan = john - 9)
  (h4 : marcus = john + 12) :
  marcus = 40 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pretzels_l549_54971


namespace NUMINAMATH_CALUDE_toys_per_box_l549_54916

/-- Given that Paul filled up four boxes and packed a total of 32 toys,
    prove that the number of toys in each box is 8. -/
theorem toys_per_box (total_toys : ℕ) (num_boxes : ℕ) (h1 : total_toys = 32) (h2 : num_boxes = 4) :
  total_toys / num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_box_l549_54916


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l549_54932

theorem complex_fraction_sum (a b : ℝ) :
  (a + b * Complex.I : ℂ) = (3 + Complex.I) / (1 - Complex.I) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l549_54932


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l549_54995

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I)^2 = 3 - 4 * Complex.I) :
  Complex.abs z = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l549_54995


namespace NUMINAMATH_CALUDE_work_completion_proof_l549_54957

/-- The number of days B takes to finish the work alone -/
def B : ℝ := 10

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def B_remaining : ℝ := 3.0000000000000004

/-- The number of days A takes to finish the work alone -/
def A : ℝ := 4

theorem work_completion_proof :
  2 * (1 / A + 1 / B) + B_remaining * (1 / B) = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l549_54957


namespace NUMINAMATH_CALUDE_soda_distribution_impossibility_l549_54984

theorem soda_distribution_impossibility (total_sodas : ℕ) (sisters : ℕ) : 
  total_sodas = 9 →
  sisters = 2 →
  ¬∃ (sodas_per_sibling : ℕ), 
    sodas_per_sibling > 0 ∧ 
    total_sodas = sodas_per_sibling * (sisters + 2 * sisters) :=
by
  sorry

end NUMINAMATH_CALUDE_soda_distribution_impossibility_l549_54984


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l549_54987

theorem price_decrease_percentage (original_price new_price : ℚ) :
  original_price = 1750 →
  new_price = 1050 →
  (original_price - new_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l549_54987


namespace NUMINAMATH_CALUDE_cats_count_pet_store_cats_l549_54938

/-- Given a ratio of cats to dogs and the number of dogs, calculate the number of cats -/
theorem cats_count (cat_ratio : ℕ) (dog_ratio : ℕ) (dog_count : ℕ) : ℕ :=
  (cat_ratio * dog_count) / dog_ratio

/-- Prove that with a cat to dog ratio of 3:4 and 20 dogs, there are 15 cats -/
theorem pet_store_cats : cats_count 3 4 20 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cats_count_pet_store_cats_l549_54938


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_6m_l549_54962

theorem factorization_3m_squared_minus_6m (m : ℝ) : 3 * m^2 - 6 * m = 3 * m * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_6m_l549_54962


namespace NUMINAMATH_CALUDE_solve_letter_problem_l549_54965

def letter_problem (brother_letters : ℕ) (greta_extra : ℕ) : Prop :=
  let greta_letters := brother_letters + greta_extra
  let total_greta_brother := brother_letters + greta_letters
  let mother_letters := 2 * total_greta_brother
  let total_letters := brother_letters + greta_letters + mother_letters
  (brother_letters = 40) ∧ (greta_extra = 10) → (total_letters = 270)

theorem solve_letter_problem : letter_problem 40 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_letter_problem_l549_54965


namespace NUMINAMATH_CALUDE_l_shaped_area_l549_54970

/-- The area of an L-shaped region formed by subtracting two smaller squares from a larger square --/
theorem l_shaped_area (square_side : ℝ) (small_square1_side : ℝ) (small_square2_side : ℝ)
  (h1 : square_side = 6)
  (h2 : small_square1_side = 2)
  (h3 : small_square2_side = 3)
  (h4 : small_square1_side < square_side)
  (h5 : small_square2_side < square_side) :
  square_side^2 - small_square1_side^2 - small_square2_side^2 = 23 := by
  sorry

#check l_shaped_area

end NUMINAMATH_CALUDE_l_shaped_area_l549_54970


namespace NUMINAMATH_CALUDE_loyalty_program_benefits_l549_54904

-- Define the structure for a bank
structure Bank where
  cardUsage : ℝ
  customerLoyalty : ℝ
  transactionVolume : ℝ

-- Define the structure for the Central Bank
structure CentralBank where
  nationalPaymentSystemUsage : ℝ
  consumerSpending : ℝ

-- Define the effect of the loyalty program
def loyaltyProgramEffect (bank : Bank) (centralBank : CentralBank) : Bank × CentralBank :=
  let newBank : Bank := {
    cardUsage := bank.cardUsage * 1.2,
    customerLoyalty := bank.customerLoyalty * 1.15,
    transactionVolume := bank.transactionVolume * 1.25
  }
  let newCentralBank : CentralBank := {
    nationalPaymentSystemUsage := centralBank.nationalPaymentSystemUsage * 1.3,
    consumerSpending := centralBank.consumerSpending * 1.1
  }
  (newBank, newCentralBank)

-- Theorem stating the benefits of the loyalty program
theorem loyalty_program_benefits 
  (bank : Bank) 
  (centralBank : CentralBank) :
  let (newBank, newCentralBank) := loyaltyProgramEffect bank centralBank
  newBank.cardUsage > bank.cardUsage ∧
  newBank.customerLoyalty > bank.customerLoyalty ∧
  newBank.transactionVolume > bank.transactionVolume ∧
  newCentralBank.nationalPaymentSystemUsage > centralBank.nationalPaymentSystemUsage ∧
  newCentralBank.consumerSpending > centralBank.consumerSpending :=
by
  sorry


end NUMINAMATH_CALUDE_loyalty_program_benefits_l549_54904


namespace NUMINAMATH_CALUDE_product_63_57_l549_54980

theorem product_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_product_63_57_l549_54980


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l549_54994

/-- The number of books Robert can read given his reading speed, book length, and available time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem stating that Robert can read 2 books in 8 hours -/
theorem robert_reading_capacity :
  books_read 100 400 8 = 2 := by
  sorry

#eval books_read 100 400 8

end NUMINAMATH_CALUDE_robert_reading_capacity_l549_54994


namespace NUMINAMATH_CALUDE_angle_D_value_l549_54908

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the theorem
theorem angle_D_value :
  A + B = 180 →  -- Condition 1
  C = D →        -- Condition 2
  C + 50 + 60 = 180 →  -- Condition 3
  D = 70 :=  -- Conclusion
by
  sorry  -- Proof is omitted as per instructions


end NUMINAMATH_CALUDE_angle_D_value_l549_54908


namespace NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l549_54915

theorem tan_value_from_sin_cos_equation (α : ℝ) 
  (h : 3 * Real.sin ((33 * π) / 14 + α) = -5 * Real.cos ((5 * π) / 14 + α)) : 
  Real.tan ((5 * π) / 14 + α) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l549_54915


namespace NUMINAMATH_CALUDE_gcd_a_squared_plus_9a_plus_24_and_a_plus_4_l549_54931

theorem gcd_a_squared_plus_9a_plus_24_and_a_plus_4 (a : ℤ) (h : ∃ k : ℤ, a = 1428 * k) :
  Nat.gcd (Int.natAbs (a^2 + 9*a + 24)) (Int.natAbs (a + 4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_a_squared_plus_9a_plus_24_and_a_plus_4_l549_54931


namespace NUMINAMATH_CALUDE_man_work_days_l549_54917

/-- Given a woman can complete a piece of work in 40 days and a man is 25% more efficient than a woman,
    prove that the man can complete the same piece of work in 32 days. -/
theorem man_work_days (woman_days : ℕ) (man_efficiency : ℚ) :
  woman_days = 40 →
  man_efficiency = 5 / 4 →
  ∃ (man_days : ℕ), man_days = 32 ∧ (man_days : ℚ) * man_efficiency = woman_days := by
  sorry

end NUMINAMATH_CALUDE_man_work_days_l549_54917


namespace NUMINAMATH_CALUDE_basketball_distribution_l549_54940

theorem basketball_distribution (total_basketballs : ℕ) (basketballs_per_class : ℕ) (num_classes : ℕ) : 
  total_basketballs = 54 → 
  basketballs_per_class = 7 → 
  total_basketballs = num_classes * basketballs_per_class →
  num_classes = 7 := by
sorry

end NUMINAMATH_CALUDE_basketball_distribution_l549_54940


namespace NUMINAMATH_CALUDE_prime_sequence_l549_54989

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_l549_54989


namespace NUMINAMATH_CALUDE_tape_shortage_l549_54968

/-- Proves that 180 feet of tape is insufficient to wrap around a 35x80 foot field and three 5-foot circumference trees, requiring an additional 65 feet. -/
theorem tape_shortage (field_width : ℝ) (field_length : ℝ) (tree_circumference : ℝ) (num_trees : ℕ) (available_tape : ℝ) : 
  field_width = 35 → 
  field_length = 80 → 
  tree_circumference = 5 → 
  num_trees = 3 → 
  available_tape = 180 → 
  (2 * (field_width + field_length) + num_trees * tree_circumference) - available_tape = 65 := by
  sorry

end NUMINAMATH_CALUDE_tape_shortage_l549_54968


namespace NUMINAMATH_CALUDE_intersection_M_N_l549_54913

/-- Set M is defined as the set of all real numbers x where 0 < x < 4 -/
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

/-- Set N is defined as the set of all real numbers x where 1/3 ≤ x ≤ 5 -/
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

/-- The intersection of sets M and N -/
theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l549_54913


namespace NUMINAMATH_CALUDE_calculation_proof_l549_54921

theorem calculation_proof : 3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l549_54921


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l549_54949

/-- A quadratic function with vertex (3, 2) passing through (-2, -18) has a = -4/5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (2 = a * 3^2 + b * 3 + c) →             -- Condition 2 (vertex)
  (3 = -b / (2 * a)) →                    -- Condition 2 (vertex x-coordinate)
  (-18 = a * (-2)^2 + b * (-2) + c) →     -- Condition 3
  a = -4/5 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l549_54949


namespace NUMINAMATH_CALUDE_line_equation_theorem_l549_54933

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the area of the triangle formed by a line and the coordinate axes -/
def triangleArea (l : Line) : ℝ := sorry

/-- Check if a line passes through a given point -/
def passesThrough (l : Line) (x y : ℝ) : Prop := 
  l.a * x + l.b * y + l.c = 0

/-- The main theorem -/
theorem line_equation_theorem (l : Line) :
  triangleArea l = 3 ∧ passesThrough l (-3) 4 →
  (l.a = 2 ∧ l.b = 3 ∧ l.c = -6) ∨ (l.a = 8 ∧ l.b = 3 ∧ l.c = 12) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l549_54933


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l549_54954

theorem square_plus_inverse_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l549_54954


namespace NUMINAMATH_CALUDE_train_length_proof_l549_54992

/-- Proves that a train with given speed crossing a bridge of known length in a specific time has a particular length -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) (train_length : ℝ) : 
  bridge_length = 320 →
  crossing_time = 40 →
  train_speed_kmh = 42.3 →
  train_length = 150 →
  (train_length + bridge_length) = (train_speed_kmh * 1000 / 3600) * crossing_time := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l549_54992


namespace NUMINAMATH_CALUDE_share_ratio_proof_l549_54905

theorem share_ratio_proof (total amount : ℕ) (a_share b_share c_share : ℕ) 
  (h1 : amount = 595)
  (h2 : a_share + b_share + c_share = amount)
  (h3 : a_share = 420)
  (h4 : b_share = 105)
  (h5 : c_share = 70)
  (h6 : 3 * a_share = 2 * b_share) : 
  b_share * 2 = c_share * 3 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_proof_l549_54905


namespace NUMINAMATH_CALUDE_uniform_price_calculation_l549_54978

/-- Represents the price of the uniform in Rupees -/
def uniform_price : ℕ := 25

/-- Represents the full year salary in Rupees -/
def full_year_salary : ℕ := 900

/-- Represents the number of months served -/
def months_served : ℕ := 9

/-- Represents the actual payment received for the partial service in Rupees -/
def partial_payment : ℕ := 650

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

theorem uniform_price_calculation :
  uniform_price = (full_year_salary * months_served / months_in_year) - partial_payment := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_calculation_l549_54978


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_9_l549_54947

theorem circle_area_with_diameter_9 (π : Real) (h : π = Real.pi) :
  let d := 9
  let r := d / 2
  let area := π * r^2
  area = π * (9/2)^2 := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_9_l549_54947


namespace NUMINAMATH_CALUDE_triangle_angle_sixty_degrees_l549_54977

theorem triangle_angle_sixty_degrees (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  C = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sixty_degrees_l549_54977


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l549_54979

/-- Proves that if reducing a bucket's capacity to 4/5 of its original requires 250 buckets to fill a tank, 
    then the number of buckets needed with the original capacity is 200. -/
theorem bucket_capacity_problem (tank_volume : ℝ) (original_capacity : ℝ) 
  (h1 : tank_volume > 0) (h2 : original_capacity > 0) :
  (tank_volume = 250 * (4/5 * original_capacity)) → 
  (tank_volume = 200 * original_capacity) :=
by
  sorry

#check bucket_capacity_problem

end NUMINAMATH_CALUDE_bucket_capacity_problem_l549_54979


namespace NUMINAMATH_CALUDE_keith_books_l549_54941

theorem keith_books (jason_books : ℕ) (total_books : ℕ) (h1 : jason_books = 21) (h2 : total_books = 41) :
  total_books - jason_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_keith_books_l549_54941


namespace NUMINAMATH_CALUDE_complex_subtraction_l549_54952

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 3*I) :
  a - 3*b = -1 - 12*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l549_54952


namespace NUMINAMATH_CALUDE_inscribed_rectangle_theorem_l549_54967

-- Define the triangle
def triangle_sides : (ℝ × ℝ × ℝ) := (10, 17, 21)

-- Define the perimeter of the inscribed rectangle
def rectangle_perimeter : ℝ := 24

-- Define the function to calculate the sides of the inscribed rectangle
def inscribed_rectangle_sides (triangle : ℝ × ℝ × ℝ) (perimeter : ℝ) : (ℝ × ℝ) :=
  sorry

-- Theorem statement
theorem inscribed_rectangle_theorem :
  inscribed_rectangle_sides triangle_sides rectangle_perimeter = (5 + 7/13, 6 + 6/13) :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_theorem_l549_54967


namespace NUMINAMATH_CALUDE_only_negative_three_less_than_reciprocal_l549_54948

def is_less_than_reciprocal (x : ℝ) : Prop :=
  x ≠ 0 ∧ x < 1 / x

theorem only_negative_three_less_than_reciprocal :
  (is_less_than_reciprocal (-3)) ∧
  (¬ is_less_than_reciprocal (-1/2)) ∧
  (¬ is_less_than_reciprocal 0) ∧
  (¬ is_less_than_reciprocal 1) ∧
  (¬ is_less_than_reciprocal (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_less_than_reciprocal_l549_54948


namespace NUMINAMATH_CALUDE_train_length_proof_l549_54912

theorem train_length_proof (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ) :
  platform_crossing_time = 39 →
  pole_crossing_time = 18 →
  platform_length = 1050 →
  ∃ train_length : ℝ, train_length = 900 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l549_54912


namespace NUMINAMATH_CALUDE_medication_dosage_range_l549_54991

theorem medication_dosage_range 
  (daily_min : ℝ) 
  (daily_max : ℝ) 
  (num_doses : ℕ) 
  (h1 : daily_min = 60) 
  (h2 : daily_max = 120) 
  (h3 : num_doses = 4) :
  ∃ x_min x_max : ℝ, 
    x_min = daily_min / num_doses ∧ 
    x_max = daily_max / num_doses ∧ 
    x_min = 15 ∧ 
    x_max = 30 ∧ 
    ∀ x : ℝ, (x_min ≤ x ∧ x ≤ x_max) ↔ (15 ≤ x ∧ x ≤ 30) :=
by sorry

end NUMINAMATH_CALUDE_medication_dosage_range_l549_54991


namespace NUMINAMATH_CALUDE_smallest_sum_20_consecutive_triangular_l549_54900

/-- The sum of 20 consecutive integers starting from n -/
def sum_20_consecutive (n : ℤ) : ℤ := 10 * (2 * n + 19)

/-- A triangular number -/
def triangular_number (m : ℕ) : ℕ := m * (m + 1) / 2

/-- Proposition: 190 is the smallest sum of 20 consecutive integers that is also a triangular number -/
theorem smallest_sum_20_consecutive_triangular :
  ∃ (m : ℕ), 
    (∀ (n : ℤ), sum_20_consecutive n ≥ 190) ∧ 
    (sum_20_consecutive 0 = 190) ∧ 
    (triangular_number m = 190) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_20_consecutive_triangular_l549_54900


namespace NUMINAMATH_CALUDE_victoria_shopping_theorem_l549_54901

def shopping_and_dinner_problem (initial_amount : ℝ) 
  (jacket_price : ℝ) (jacket_quantity : ℕ)
  (trouser_price : ℝ) (trouser_quantity : ℕ)
  (purse_price : ℝ) (purse_quantity : ℕ)
  (discount_rate : ℝ) (dinner_bill : ℝ) : Prop :=
  let jacket_cost := jacket_price * jacket_quantity
  let trouser_cost := trouser_price * trouser_quantity
  let purse_cost := purse_price * purse_quantity
  let discountable_cost := jacket_cost + trouser_cost
  let discount_amount := discountable_cost * discount_rate
  let shopping_cost := discountable_cost - discount_amount + purse_cost
  let dinner_cost := dinner_bill / 1.15
  let total_spent := shopping_cost + dinner_cost
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 3725

theorem victoria_shopping_theorem : 
  shopping_and_dinner_problem 10000 250 8 180 15 450 4 0.15 552.50 :=
by sorry

end NUMINAMATH_CALUDE_victoria_shopping_theorem_l549_54901


namespace NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_5_l549_54937

theorem x_gt_2_necessary_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 2) ∧ (∃ x : ℝ, x > 2 ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_5_l549_54937


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l549_54945

/-- The number of ways to distribute n different balls into k different boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n different balls into k different boxes with exactly m empty boxes -/
def distributeWithEmpty (n k m : ℕ) : ℕ := sorry

theorem ball_distribution_theorem (n k : ℕ) (hn : n = 4) (hk : k = 4) :
  distribute n k = 256 ∧
  distributeWithEmpty n k 1 = 144 ∧
  distributeWithEmpty n k 2 = 84 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l549_54945


namespace NUMINAMATH_CALUDE_walters_age_l549_54988

theorem walters_age (walter_age_2005 : ℕ) (grandmother_age_2005 : ℕ) : 
  walter_age_2005 = grandmother_age_2005 / 3 →
  (2005 - walter_age_2005) + (2005 - grandmother_age_2005) = 3858 →
  walter_age_2005 + 5 = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_walters_age_l549_54988


namespace NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l549_54960

/-- An angle is in the first quadrant if it's between 0 and π/2 radians -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

/-- The main theorem stating the equivalence between an angle being in the first quadrant
    and the sum of its sine and cosine being greater than 1 -/
theorem first_quadrant_iff_sin_cos_sum_gt_one (α : ℝ) :
  is_first_quadrant α ↔ Real.sin α + Real.cos α > 1 :=
sorry

end NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l549_54960


namespace NUMINAMATH_CALUDE_min_sum_same_last_three_digits_l549_54966

/-- Given two positive integers m and n where n > m ≥ 1, this theorem states that
    if 1978^n and 1978^m have the same last three digits, then m + n ≥ 106. -/
theorem min_sum_same_last_three_digits (m n : ℕ) (hm : m ≥ 1) (hn : n > m) :
  (1978^n : ℕ) % 1000 = (1978^m : ℕ) % 1000 → m + n ≥ 106 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_same_last_three_digits_l549_54966


namespace NUMINAMATH_CALUDE_correct_operation_l549_54944

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l549_54944


namespace NUMINAMATH_CALUDE_orange_boxes_total_l549_54958

theorem orange_boxes_total (box1_capacity box2_capacity box3_capacity : ℕ)
  (box1_fill box2_fill box3_fill : ℚ) :
  box1_capacity = 80 →
  box2_capacity = 50 →
  box3_capacity = 60 →
  box1_fill = 3/4 →
  box2_fill = 3/5 →
  box3_fill = 2/3 →
  (↑box1_capacity * box1_fill + ↑box2_capacity * box2_fill + ↑box3_capacity * box3_fill : ℚ) = 130 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_total_l549_54958


namespace NUMINAMATH_CALUDE_calculation_one_l549_54998

theorem calculation_one : (1) - 2 + (-3) - (-5) + 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_one_l549_54998


namespace NUMINAMATH_CALUDE_sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45_l549_54928

theorem sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45 :
  Real.sqrt 8 - abs (-2) + (1/3)⁻¹ - 2 * Real.cos (45 * π / 180) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45_l549_54928


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_gcd_fermat_numbers_l549_54934

-- Part (a)
theorem gcd_power_minus_one (a m n : ℕ) (ha : a > 1) (hm : m ≠ n) :
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 := by
sorry

-- Part (b)
def fermat (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (fermat n) (fermat m) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_gcd_fermat_numbers_l549_54934


namespace NUMINAMATH_CALUDE_fish_count_l549_54903

theorem fish_count (total_tables : ℕ) (special_table_fish : ℕ) (regular_table_fish : ℕ)
  (h1 : total_tables = 32)
  (h2 : special_table_fish = 3)
  (h3 : regular_table_fish = 2) :
  (total_tables - 1) * regular_table_fish + special_table_fish = 65 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l549_54903


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l549_54964

theorem first_number_in_ratio (a b : ℕ) : 
  a ≠ 0 → b ≠ 0 → 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 84 → 
  a = 63 :=
by sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l549_54964


namespace NUMINAMATH_CALUDE_peters_change_l549_54923

/-- Calculates the change left after Peter buys glasses -/
theorem peters_change (small_price large_price total_money small_count large_count : ℕ) : 
  small_price = 3 →
  large_price = 5 →
  total_money = 50 →
  small_count = 8 →
  large_count = 5 →
  total_money - (small_price * small_count + large_price * large_count) = 1 := by
sorry

end NUMINAMATH_CALUDE_peters_change_l549_54923


namespace NUMINAMATH_CALUDE_repeating_decimal_027_product_l549_54963

/-- Represents a repeating decimal with a 3-digit repeating sequence -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_027_product : 
  let x := RepeatingDecimal 0 2 7
  let (n, d) := (x.num, x.den)
  (n.gcd d = 1) → n * d = 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_027_product_l549_54963


namespace NUMINAMATH_CALUDE_fraction_inequality_l549_54969

theorem fraction_inequality (a b c d e : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0)
  (h5 : e < 0) :
  e / (a - c) > e / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l549_54969


namespace NUMINAMATH_CALUDE_cards_selection_count_l549_54951

/-- The number of ways to select 3 cards from 12 cards (3 each of red, yellow, green, and blue) 
    such that they are not all the same color and there is at most 1 blue card. -/
def select_cards : ℕ := sorry

/-- The total number of cards -/
def total_cards : ℕ := 12

/-- The number of cards of each color -/
def cards_per_color : ℕ := 3

/-- The number of colors -/
def num_colors : ℕ := 4

/-- The number of cards to be selected -/
def cards_to_select : ℕ := 3

theorem cards_selection_count : 
  select_cards = Nat.choose total_cards cards_to_select - 
                 num_colors - 
                 (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1) := by
  sorry

end NUMINAMATH_CALUDE_cards_selection_count_l549_54951


namespace NUMINAMATH_CALUDE_managing_team_selection_l549_54985

def society_size : ℕ := 20
def team_size : ℕ := 3

theorem managing_team_selection :
  Nat.choose society_size team_size = 1140 := by
  sorry

end NUMINAMATH_CALUDE_managing_team_selection_l549_54985


namespace NUMINAMATH_CALUDE_hedge_cost_proof_l549_54919

/-- Calculates the total cost of concrete blocks for a hedge --/
def total_cost (sections : ℕ) (blocks_per_section : ℕ) (cost_per_block : ℕ) : ℕ :=
  sections * blocks_per_section * cost_per_block

/-- Proves that the total cost of concrete blocks for the hedge is $480 --/
theorem hedge_cost_proof :
  total_cost 8 30 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_hedge_cost_proof_l549_54919


namespace NUMINAMATH_CALUDE_divisibility_and_modulo_l549_54909

theorem divisibility_and_modulo (n : ℤ) (h : 11 ∣ (4 * n + 3)) : 
  n % 11 = 2 ∧ n^4 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_modulo_l549_54909


namespace NUMINAMATH_CALUDE_cubeTowerSurfaceAreaIs1221_l549_54996

/-- Calculates the surface area of a cube tower given a list of cube side lengths -/
def cubeTowerSurfaceArea (sideLengths : List ℕ) : ℕ :=
  match sideLengths with
  | [] => 0
  | [x] => 6 * x^2
  | x :: xs => 4 * x^2 + cubeTowerSurfaceArea xs

/-- The list of cube side lengths in the tower -/
def towerSideLengths : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Theorem stating that the surface area of the cube tower is 1221 square units -/
theorem cubeTowerSurfaceAreaIs1221 :
  cubeTowerSurfaceArea towerSideLengths = 1221 := by
  sorry


end NUMINAMATH_CALUDE_cubeTowerSurfaceAreaIs1221_l549_54996


namespace NUMINAMATH_CALUDE_orange_juice_mixture_l549_54914

theorem orange_juice_mixture (pitcher_capacity : ℚ) 
  (first_pitcher_fraction : ℚ) (second_pitcher_fraction : ℚ) : 
  pitcher_capacity > 0 →
  first_pitcher_fraction = 1/4 →
  second_pitcher_fraction = 3/7 →
  (first_pitcher_fraction * pitcher_capacity + 
   second_pitcher_fraction * pitcher_capacity) / 
  (2 * pitcher_capacity) = 95/280 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_l549_54914


namespace NUMINAMATH_CALUDE_N2O3_molecular_weight_l549_54935

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in atomic mass units (amu) -/
def N2O3_weight : ℝ := nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

theorem N2O3_molecular_weight : N2O3_weight = 76.02 := by sorry

end NUMINAMATH_CALUDE_N2O3_molecular_weight_l549_54935


namespace NUMINAMATH_CALUDE_distance_between_parallel_points_l549_54942

/-- Given two points A(4, a) and B(5, b) on a line parallel to y = x + m,
    prove that the distance between A and B is √2. -/
theorem distance_between_parallel_points :
  ∀ (a b m : ℝ),
  (b - a) / (5 - 4) = 1 →  -- Parallel condition
  Real.sqrt ((5 - 4)^2 + (b - a)^2) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_parallel_points_l549_54942


namespace NUMINAMATH_CALUDE_expected_terms_is_ten_l549_54910

/-- A fair tetrahedral die with faces numbered 1 to 4 -/
structure TetrahedralDie :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4})

/-- The state of the sequence -/
inductive SequenceState
| Zero  : SequenceState  -- No distinct numbers seen
| One   : SequenceState  -- One distinct number seen
| Two   : SequenceState  -- Two distinct numbers seen
| Three : SequenceState  -- Three distinct numbers seen
| Four  : SequenceState  -- All four numbers seen

/-- Expected number of terms to complete the sequence from a given state -/
noncomputable def expectedTerms (s : SequenceState) : ℝ :=
  match s with
  | SequenceState.Zero  => sorry
  | SequenceState.One   => sorry
  | SequenceState.Two   => sorry
  | SequenceState.Three => sorry
  | SequenceState.Four  => 0

/-- Main theorem: The expected number of terms in the sequence is 10 -/
theorem expected_terms_is_ten (d : TetrahedralDie) : 
  expectedTerms SequenceState.Zero = 10 := by sorry

end NUMINAMATH_CALUDE_expected_terms_is_ten_l549_54910


namespace NUMINAMATH_CALUDE_sequence_formula_l549_54924

-- Define the sequence and its sum
def S (n : ℕ) : ℕ := 2 * n^2 + n

-- Define the nth term of the sequence
def a (n : ℕ) : ℕ := 4 * n - 1

-- Theorem statement
theorem sequence_formula (n : ℕ) : S n - S (n-1) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l549_54924


namespace NUMINAMATH_CALUDE_couple_driving_exam_probability_l549_54911

/-- Represents the probability of passing an exam for each attempt -/
structure ExamProbability where
  male : ℚ
  female : ℚ

/-- Represents the exam attempt limits and fee structure -/
structure ExamRules where
  free_attempts : ℕ
  max_attempts : ℕ
  fee : ℚ

/-- Calculates the probability of a couple passing the exam under given conditions -/
def couple_exam_probability (prob : ExamProbability) (rules : ExamRules) : ℚ × ℚ :=
  sorry

theorem couple_driving_exam_probability :
  let prob := ExamProbability.mk (3/4) (2/3)
  let rules := ExamRules.mk 2 5 200
  let result := couple_exam_probability prob rules
  result.1 = 5/6 ∧ result.2 = 1/9 :=
sorry

end NUMINAMATH_CALUDE_couple_driving_exam_probability_l549_54911


namespace NUMINAMATH_CALUDE_equal_product_grouping_l549_54939

theorem equal_product_grouping (numbers : Finset ℕ) 
  (h_numbers : numbers = {12, 30, 42, 44, 57, 91, 95, 143}) :
  (12 * 42 * 95 * 143 : ℕ) = (30 * 44 * 57 * 91 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_equal_product_grouping_l549_54939


namespace NUMINAMATH_CALUDE_problem_solution_l549_54927

open Real

/-- The function f(x) = e^x + sin(x) + b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := exp x + sin x + b

/-- The function g(x) = xe^x -/
noncomputable def g (x : ℝ) : ℝ := x * exp x

theorem problem_solution :
  (∀ b : ℝ, (∀ x : ℝ, x ≥ 0 → f b x ≥ 0) → b ≥ -1) ∧
  (∀ m : ℝ, (∃ b : ℝ, (∀ x : ℝ, exp x + b = x - 1) ∧
                     (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ exp x₁ - 2 = (m - 2*x₁)/x₁ ∧
                                   exp x₂ - 2 = (m - 2*x₂)/x₂) ∧
                     (∀ x : ℝ, exp x - 2 = (m - 2*x)/x → x = x₁ ∨ x = x₂)) →
   -1/exp 1 < m ∧ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l549_54927


namespace NUMINAMATH_CALUDE_chromium_percentage_in_combined_alloy_l549_54918

/-- Calculates the percentage of chromium in a new alloy formed by combining two other alloys -/
theorem chromium_percentage_in_combined_alloy 
  (chromium_percent1 : ℝ) 
  (weight1 : ℝ) 
  (chromium_percent2 : ℝ) 
  (weight2 : ℝ) 
  (h1 : chromium_percent1 = 12)
  (h2 : weight1 = 15)
  (h3 : chromium_percent2 = 8)
  (h4 : weight2 = 40) :
  let total_chromium := (chromium_percent1 / 100) * weight1 + (chromium_percent2 / 100) * weight2
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 9.09 := by
  sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_combined_alloy_l549_54918


namespace NUMINAMATH_CALUDE_trajectory_and_line_theorem_l549_54972

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 49/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1/4

-- Define the trajectory of P
def trajectory_P (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = -2

theorem trajectory_and_line_theorem :
  ∃ k : ℝ, k^2 = 2 ∧
  (∀ x y : ℝ, trajectory_P x y →
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
      trajectory_P x₁ y₁ ∧ trajectory_P x₂ y₂ ∧
      dot_product_condition x₁ y₁ x₂ y₂)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_line_theorem_l549_54972


namespace NUMINAMATH_CALUDE_square_sum_from_means_l549_54997

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 24)
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 156) :
  a^2 + b^2 = 1992 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l549_54997


namespace NUMINAMATH_CALUDE_rotated_line_equation_l549_54922

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * x + b * y + c = 0

/-- Rotates a line counterclockwise by π/2 around its y-axis intersection --/
def rotate_line_pi_over_2 (l : Line) : Line :=
  sorry

theorem rotated_line_equation :
  let original_line : Line := ⟨2, -1, -2, by sorry⟩
  let rotated_line := rotate_line_pi_over_2 original_line
  rotated_line.a = 1 ∧ rotated_line.b = 2 ∧ rotated_line.c = 4 :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l549_54922


namespace NUMINAMATH_CALUDE_sum_of_squares_l549_54950

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0) 
  (h7 : a ≠ 0) 
  (h8 : b ≠ 0) 
  (h9 : c ≠ 0) : 
  x^2 + y^2 + z^2 = ((a*b)^2 + (a*c)^2 + (b*c)^2) / (a*b*c) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l549_54950


namespace NUMINAMATH_CALUDE_dice_probability_l549_54990

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def num_favorable : ℕ := 4

theorem dice_probability :
  let p_first_die : ℚ := (num_favorable : ℚ) / num_sides
  let p_remaining : ℚ := 1 / 2
  let combinations : ℕ := Nat.choose (num_dice - 1) (num_favorable - 1)
  p_first_die * combinations * p_remaining ^ (num_dice - 1) = 35 / 256 := by
    sorry

end NUMINAMATH_CALUDE_dice_probability_l549_54990


namespace NUMINAMATH_CALUDE_ordering_proof_l549_54936

theorem ordering_proof (a b c : ℝ) 
  (ha : a = Real.log 2.6)
  (hb : b = 0.5 * 1.8^2)
  (hc : c = 1.1^5) : 
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ordering_proof_l549_54936
