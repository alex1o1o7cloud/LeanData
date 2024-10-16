import Mathlib

namespace NUMINAMATH_CALUDE_abcd_sum_l1952_195253

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -2)
  (eq3 : a + c + d = 5)
  (eq4 : b + c + d = 4) :
  a * b + c * d = 26 / 9 := by
sorry

end NUMINAMATH_CALUDE_abcd_sum_l1952_195253


namespace NUMINAMATH_CALUDE_problem_statement_l1952_195265

theorem problem_statement (x y : ℝ) (θ : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_θ_range : π/4 < θ ∧ θ < π/2)
  (h_eq1 : Real.cos θ / x = Real.sin θ / y)
  (h_eq2 : Real.sin θ^2 / x^2 + Real.cos θ^2 / y^2 = 10 / (3 * (x^2 + y^2))) :
  (x + y)^2 / (x^2 + y^2) = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1952_195265


namespace NUMINAMATH_CALUDE_solution_water_amount_l1952_195234

/-- Given a solution with an original ratio of bleach : detergent : water as 2 : 40 : 100,
    when the ratio of bleach to detergent is tripled and the ratio of detergent to water is halved,
    and the new solution contains 60 liters of detergent, prove that the amount of water
    in the new solution is 75 liters. -/
theorem solution_water_amount
  (original_ratio : Fin 3 → ℚ)
  (h_original : original_ratio = ![2, 40, 100])
  (new_detergent : ℚ)
  (h_new_detergent : new_detergent = 60)
  : ∃ (new_ratio : Fin 3 → ℚ) (water : ℚ),
    (new_ratio 0 / new_ratio 1 = 3 * (original_ratio 0 / original_ratio 1)) ∧
    (new_ratio 1 / new_ratio 2 = (original_ratio 1 / original_ratio 2) / 2) ∧
    (new_ratio 1 = new_detergent) ∧
    (water = 75) := by
  sorry

end NUMINAMATH_CALUDE_solution_water_amount_l1952_195234


namespace NUMINAMATH_CALUDE_remainder_sum_product_l1952_195254

theorem remainder_sum_product (X Y Z E S T U s t q : ℕ) 
  (hX : X > Y) (hY : Y > Z)
  (hS : X % E = S) (hT : Y % E = T) (hU : Z % E = U)
  (hs : (X * Y * Z) % E = s) (ht : (S * T * U) % E = t)
  (hq : (X * Y * Z + S * T * U) % E = q) :
  q = (2 * s) % E :=
sorry

end NUMINAMATH_CALUDE_remainder_sum_product_l1952_195254


namespace NUMINAMATH_CALUDE_water_in_tank_after_rain_l1952_195216

/-- Calculates the final amount of water in a tank after evaporation, draining, and rain. -/
def final_water_amount (initial_water evaporated_water drained_water rain_duration rain_rate : ℕ) : ℕ :=
  let remaining_after_evaporation := initial_water - evaporated_water
  let remaining_after_draining := remaining_after_evaporation - drained_water
  let rain_amount := (rain_duration / 10) * rain_rate
  remaining_after_draining + rain_amount

/-- Theorem stating that the final amount of water in the tank is 1550 liters. -/
theorem water_in_tank_after_rain :
  final_water_amount 6000 2000 3500 30 350 = 1550 := by
  sorry

end NUMINAMATH_CALUDE_water_in_tank_after_rain_l1952_195216


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1952_195219

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 15.999

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ :=
  hydrogen_count * hydrogen_weight +
  chlorine_count * chlorine_weight +
  oxygen_count * oxygen_weight

theorem compound_molecular_weight :
  molecular_weight = 68.456 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1952_195219


namespace NUMINAMATH_CALUDE_simplify_expression_l1952_195270

theorem simplify_expression (z : ℝ) : (7 - Real.sqrt (z^2 - 49))^2 = z^2 - 14 * Real.sqrt (z^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1952_195270


namespace NUMINAMATH_CALUDE_base7UnitsDigitIs6_l1952_195218

/-- The units digit of the base-7 representation of the product of 328 and 57 -/
def base7UnitsDigit : ℕ :=
  (328 * 57) % 7

/-- Theorem stating that the units digit of the base-7 representation of the product of 328 and 57 is 6 -/
theorem base7UnitsDigitIs6 : base7UnitsDigit = 6 := by
  sorry

end NUMINAMATH_CALUDE_base7UnitsDigitIs6_l1952_195218


namespace NUMINAMATH_CALUDE_christmas_ball_colors_l1952_195298

/-- Given a total number of balls and the number of balls per color, 
    calculate the number of colors used. -/
def number_of_colors (total_balls : ℕ) (balls_per_color : ℕ) : ℕ :=
  total_balls / balls_per_color

/-- Prove that the number of colors used is 10 given the problem conditions. -/
theorem christmas_ball_colors :
  let total_balls : ℕ := 350
  let balls_per_color : ℕ := 35
  number_of_colors total_balls balls_per_color = 10 := by
  sorry

end NUMINAMATH_CALUDE_christmas_ball_colors_l1952_195298


namespace NUMINAMATH_CALUDE_circle_area_with_chord_l1952_195223

theorem circle_area_with_chord (chord_length : ℝ) (center_to_chord : ℝ) (area : ℝ) : 
  chord_length = 10 →
  center_to_chord = 5 →
  area = π * (center_to_chord^2 + (chord_length / 2)^2) →
  area = 50 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_with_chord_l1952_195223


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1952_195250

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 7) / (x - 4) = (x - 1) / (x + 6) ∧ x = -19/9 ∧ x ≠ -6 ∧ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1952_195250


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1952_195204

theorem arithmetic_sequence_common_difference :
  ∀ (a d : ℚ) (n : ℕ),
    a = 2 →
    a + (n - 1) * d = 20 →
    n * (a + (a + (n - 1) * d)) / 2 = 132 →
    d = 18 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1952_195204


namespace NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l1952_195206

/-- Given a > 0 and f(x) = x³ - ax is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotone_cubic (a : ℝ) : 
  a > 0 → 
  (∀ x y, 1 ≤ x → x ≤ y → x^3 - a*x ≤ y^3 - a*y) → 
  a ≤ 3 ∧ ∀ b, (b > 0 ∧ (∀ x y, 1 ≤ x → x ≤ y → x^3 - b*x ≤ y^3 - b*y)) → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l1952_195206


namespace NUMINAMATH_CALUDE_equation_solution_l1952_195297

theorem equation_solution : {x : ℝ | x^2 = 2*x} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1952_195297


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1952_195207

theorem sum_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 220) (h2 : x * y = 52) : x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1952_195207


namespace NUMINAMATH_CALUDE_solve_for_k_l1952_195221

theorem solve_for_k (x k : ℝ) : x + k - 4 = 0 → x = 2 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l1952_195221


namespace NUMINAMATH_CALUDE_initial_books_count_l1952_195213

/-- The number of people who borrowed books on the first day -/
def borrowers : ℕ := 5

/-- The number of books each person borrowed on the first day -/
def books_per_borrower : ℕ := 2

/-- The number of books borrowed on the second day -/
def second_day_borrowed : ℕ := 20

/-- The number of books remaining on the shelf after the second day -/
def remaining_books : ℕ := 70

/-- The initial number of books on the shelf -/
def initial_books : ℕ := borrowers * books_per_borrower + second_day_borrowed + remaining_books

theorem initial_books_count : initial_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l1952_195213


namespace NUMINAMATH_CALUDE_limit_exists_and_equals_20_21_l1952_195283

/-- Sum of exponents of 71 and 97 in the prime factorization of n -/
def s (n : ℕ+) : ℕ :=
  sorry

/-- Function f(n) = (-1)^(s(n)) -/
def f (n : ℕ+) : ℤ :=
  (-1) ^ (s n)

/-- Sum of f(x) from x = 1 to n -/
def S (n : ℕ+) : ℤ :=
  (Finset.range n).sum (fun x => f ⟨x + 1, Nat.succ_pos x⟩)

/-- The main theorem: limit of S(n)/n exists and equals 20/21 -/
theorem limit_exists_and_equals_20_21 :
    ∃ (L : ℚ), L = 20 / 21 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N,
      |((S n : ℚ) / n) - L| < ε :=
  sorry

end NUMINAMATH_CALUDE_limit_exists_and_equals_20_21_l1952_195283


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l1952_195269

theorem angle_inequality_equivalence (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → x^2 * Real.cos θ - x * (2 - x) + (2 - x)^2 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l1952_195269


namespace NUMINAMATH_CALUDE_exists_quadratic_function_l1952_195247

/-- A quadratic function that fits the given points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem exists_quadratic_function : 
  ∃ (a b c : ℝ), 
    quadratic_function a b c 1 = 1 ∧
    quadratic_function a b c 2 = 4 ∧
    quadratic_function a b c 4 = 16 ∧
    quadratic_function a b c 5 = 25 ∧
    quadratic_function a b c 7 = 49 ∧
    quadratic_function a b c 8 = 64 ∧
    quadratic_function a b c 10 = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_quadratic_function_l1952_195247


namespace NUMINAMATH_CALUDE_mango_buying_rate_l1952_195225

/-- Represents the rate at which mangoes are bought and sold -/
structure MangoRate where
  buy : ℚ  -- Buying rate (rupees per x mangoes)
  sell : ℚ  -- Selling rate (mangoes per rupee)

/-- Calculates the profit percentage given buying and selling rates -/
def profit_percentage (rate : MangoRate) : ℚ :=
  (rate.sell⁻¹ / rate.buy - 1) * 100

/-- Proves that the buying rate is 2 rupees for x mangoes given the conditions -/
theorem mango_buying_rate (rate : MangoRate) 
  (h_sell : rate.sell = 3)
  (h_profit : profit_percentage rate = 50) :
  rate.buy = 2 := by
  sorry

end NUMINAMATH_CALUDE_mango_buying_rate_l1952_195225


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identity_l1952_195228

-- Define the triangle PQR
def Triangle (P Q R : ℝ) : Prop := 
  ∃ (pq pr qr : ℝ), pq = 7 ∧ pr = 8 ∧ qr = 5 ∧ 
  pq + pr > qr ∧ pq + qr > pr ∧ pr + qr > pq

-- State the theorem
theorem triangle_trigonometric_identity (P Q R : ℝ) 
  (h : Triangle P Q R) : 
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identity_l1952_195228


namespace NUMINAMATH_CALUDE_james_marbles_left_james_final_marbles_l1952_195251

/-- Represents the number of marbles in each bag -/
structure Bags where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  g : Nat

/-- Calculates the total number of marbles in all bags -/
def totalMarbles (bags : Bags) : Nat :=
  bags.a + bags.b + bags.c + bags.d + bags.e + bags.f + bags.g

/-- Represents James' marble collection -/
structure MarbleCollection where
  initialTotal : Nat
  bags : Bags
  forgottenBag : Nat

/-- Theorem stating that James will have 20 marbles left -/
theorem james_marbles_left (collection : MarbleCollection) : Nat :=
  if collection.initialTotal = 28 ∧
     collection.bags.a = 4 ∧
     collection.bags.b = 3 ∧
     collection.bags.c = 5 ∧
     collection.bags.d = 2 * collection.bags.c - 1 ∧
     collection.bags.e = collection.bags.a / 2 ∧
     collection.bags.f = 3 ∧
     collection.bags.g = collection.bags.e ∧
     collection.forgottenBag = 4 ∧
     totalMarbles collection.bags = collection.initialTotal
  then
    collection.initialTotal - (collection.bags.d + collection.bags.f) + collection.forgottenBag
  else
    0

/-- Main theorem to prove -/
theorem james_final_marbles (collection : MarbleCollection) :
  james_marbles_left collection = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_marbles_left_james_final_marbles_l1952_195251


namespace NUMINAMATH_CALUDE_estimate_pi_l1952_195292

theorem estimate_pi (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let estimate := 4 * (m / n + 1 / 2)
  estimate = 78 / 25 := by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_l1952_195292


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1952_195274

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 2^p) 
  (hy : y = 1 + 2^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1952_195274


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1952_195242

theorem gcd_from_lcm_and_ratio (A B : ℕ) (h1 : lcm A B = 180) (h2 : A * 5 = B * 2) : 
  gcd A B = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1952_195242


namespace NUMINAMATH_CALUDE_multiple_choice_probabilities_l1952_195246

/-- Represents the scoring rules for multiple-choice questions -/
structure ScoringRules where
  all_correct : Nat
  some_correct : Nat
  incorrect_or_none : Nat

/-- Represents the probabilities of selecting different numbers of options -/
structure SelectionProbabilities where
  one_option : Real
  two_options : Real
  three_options : Real

/-- Represents a multiple-choice question with its correct answer -/
structure MultipleChoiceQuestion where
  correct_options : Nat

/-- Theorem stating the probabilities for the given scenario -/
theorem multiple_choice_probabilities 
  (rules : ScoringRules)
  (probs : SelectionProbabilities)
  (q11 q12 : MultipleChoiceQuestion)
  (h1 : rules.all_correct = 5 ∧ rules.some_correct = 2 ∧ rules.incorrect_or_none = 0)
  (h2 : probs.one_option = 1/3 ∧ probs.two_options = 1/3 ∧ probs.three_options = 1/3)
  (h3 : q11.correct_options = 2 ∧ q12.correct_options = 2) :
  (∃ (p1 p2 : Real),
    -- Probability of getting 2 points for question 11
    p1 = 1/6 ∧
    -- Probability of scoring a total of 7 points for questions 11 and 12
    p2 = 1/54) :=
  sorry

end NUMINAMATH_CALUDE_multiple_choice_probabilities_l1952_195246


namespace NUMINAMATH_CALUDE_larger_number_proof_l1952_195256

theorem larger_number_proof (a b : ℝ) : 
  a > 0 → b > 0 → a > b → a + b = 9 * (a - b) → a + b = 36 → a = 20 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1952_195256


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l1952_195282

theorem floor_sqrt_eight_count :
  (Finset.filter (fun x : ℕ => ⌊Real.sqrt x⌋ = 8) (Finset.range 81)).card = 17 :=
sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l1952_195282


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1952_195293

theorem reciprocal_sum_pairs : 
  (∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6) ∧
    s.card = 9) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1952_195293


namespace NUMINAMATH_CALUDE_kite_area_theorem_l1952_195224

/-- A symmetrical quadrilateral kite -/
structure Kite where
  base : ℝ
  height : ℝ

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := k.base * k.height

/-- Theorem: The area of a kite with base 35 and height 15 is 525 -/
theorem kite_area_theorem (k : Kite) (h1 : k.base = 35) (h2 : k.height = 15) :
  kite_area k = 525 := by
  sorry

#check kite_area_theorem

end NUMINAMATH_CALUDE_kite_area_theorem_l1952_195224


namespace NUMINAMATH_CALUDE_inequality_proof_l1952_195215

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  4 * a^3 * (a - b) ≥ a^4 - b^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1952_195215


namespace NUMINAMATH_CALUDE_real_part_of_complex_square_l1952_195261

theorem real_part_of_complex_square : Complex.re ((1 + 2 * Complex.I) ^ 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_square_l1952_195261


namespace NUMINAMATH_CALUDE_exponential_function_point_l1952_195277

/-- Given a function y = a^(x-m) + n - 3 where a > 0 and a ≠ 1, 
    if the graph always passes through the point (3, 2), then m + n = 7 -/
theorem exponential_function_point (a m n : ℝ) : 
  a > 0 → a ≠ 1 → (∀ x : ℝ, a^(x - m) + n - 3 = 2 ↔ x = 3) → m + n = 7 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_point_l1952_195277


namespace NUMINAMATH_CALUDE_function_behavior_l1952_195248

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : is_symmetric_about_one f)
  (h3 : is_decreasing_on f 1 2) :
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l1952_195248


namespace NUMINAMATH_CALUDE_percentage_seniors_in_statistics_l1952_195245

theorem percentage_seniors_in_statistics :
  ∀ (total_students : ℕ) (seniors_in_statistics : ℕ),
    total_students = 120 →
    seniors_in_statistics = 54 →
    (seniors_in_statistics : ℚ) / ((total_students : ℚ) / 2) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_seniors_in_statistics_l1952_195245


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l1952_195202

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2156 → n + (n + 1) = 93 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l1952_195202


namespace NUMINAMATH_CALUDE_min_sum_squares_l1952_195243

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1952_195243


namespace NUMINAMATH_CALUDE_complex_multiplication_division_l1952_195296

theorem complex_multiplication_division (z₁ z₂ : ℂ) : 
  z₁ = 1 - I → z₂ = 1 + I → (z₁ * z₂) / I = -2 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_division_l1952_195296


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1952_195232

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  let S := (a^2 - a*b + b^2) * (b^2 - b*c + c^2) * (c^2 - c*a + a^2)
  ∃ (max_value : ℝ), max_value = 12 ∧ S ≤ max_value :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1952_195232


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l1952_195255

def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l1952_195255


namespace NUMINAMATH_CALUDE_min_pool_cost_l1952_195258

/-- Represents the cost of constructing a rectangular pool -/
def pool_cost (l w h : ℝ) : ℝ :=
  120 * l * w + 80 * 2 * h * (l + w)

/-- The minimum cost of constructing a pool with given specifications -/
theorem min_pool_cost :
  ∀ l w : ℝ,
  l > 0 ∧ w > 0 →
  l * w * 2 = 8 →
  pool_cost l w 2 ≥ 1760 :=
sorry

end NUMINAMATH_CALUDE_min_pool_cost_l1952_195258


namespace NUMINAMATH_CALUDE_intersection_point_product_l1952_195284

noncomputable section

-- Define the curves in polar coordinates
def C₁ (θ : Real) : Real := 2 * Real.cos θ
def C₂ (θ : Real) : Real := 3 / (Real.cos θ + Real.sin θ)

-- Define the condition for α
def valid_α (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- State the theorem
theorem intersection_point_product (α : Real) 
  (h₁ : valid_α α) 
  (h₂ : C₁ α * C₂ α = 3) : 
  α = Real.pi / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_point_product_l1952_195284


namespace NUMINAMATH_CALUDE_inequality_proof_l1952_195271

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : (x^2 / (1 + x^2)) + (y^2 / (1 + y^2)) + (z^2 / (1 + z^2)) = 2) : 
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1952_195271


namespace NUMINAMATH_CALUDE_mean_temperature_l1952_195209

def temperatures : List ℝ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = 85.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1952_195209


namespace NUMINAMATH_CALUDE_triangle_area_l1952_195273

/-- The area of a triangle with sides a = 4, b = 5, and angle C = 60° is 5√3 -/
theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 4) (h2 : b = 5) (h3 : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1952_195273


namespace NUMINAMATH_CALUDE_linear_systems_solution_l1952_195214

/-- Given two systems of linear equations with the same solution, 
    prove the solution, the values of a and b, and a related expression. -/
theorem linear_systems_solution :
  ∃ (x y a b : ℝ),
    -- First system of equations
    (2 * x + 5 * y = -26 ∧ a * x - b * y = -4) ∧
    -- Second system of equations
    (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
    -- The solution
    (x = 2 ∧ y = -6) ∧
    -- The values of a and b
    (a = 1 ∧ b = -1) ∧
    -- The value of the expression
    ((2 * a + b) ^ 2020 = 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_systems_solution_l1952_195214


namespace NUMINAMATH_CALUDE_baseball_card_ratio_l1952_195212

theorem baseball_card_ratio (rob_total : ℕ) (jess_doubles : ℕ) : 
  rob_total = 24 →
  jess_doubles = 40 →
  (jess_doubles : ℚ) / ((rob_total : ℚ) / 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_ratio_l1952_195212


namespace NUMINAMATH_CALUDE_factorial_sum_equation_l1952_195286

theorem factorial_sum_equation : ∃ n m : ℕ, n * n.factorial + m * m.factorial = 4032 ∧ n = 7 ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equation_l1952_195286


namespace NUMINAMATH_CALUDE_gcd_problem_l1952_195229

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1193 * k ∧ Odd k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1952_195229


namespace NUMINAMATH_CALUDE_remainder_sum_mod_60_l1952_195241

theorem remainder_sum_mod_60 (c d : ℤ) 
  (h1 : c % 120 = 114)
  (h2 : d % 180 = 174) : 
  (c + d) % 60 = 48 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_60_l1952_195241


namespace NUMINAMATH_CALUDE_expression_factorization_l1952_195235

theorem expression_factorization (a b c : ℝ) :
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l1952_195235


namespace NUMINAMATH_CALUDE_sum_236_83_base4_l1952_195238

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 4 representation -/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem: The sum of 236 and 83 in base 4 is [1, 3, 3, 2, 3] -/
theorem sum_236_83_base4 :
  addBase4 (toBase4 236) (toBase4 83) = [1, 3, 3, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_sum_236_83_base4_l1952_195238


namespace NUMINAMATH_CALUDE_largest_multiple_of_18_with_9_and_0_digits_l1952_195205

def is_multiple_of_18 (n : ℕ) : Prop := ∃ k : ℕ, n = 18 * k

def digits_are_9_or_0 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 9 ∨ d = 0

theorem largest_multiple_of_18_with_9_and_0_digits :
  ∃ m : ℕ,
    is_multiple_of_18 m ∧
    digits_are_9_or_0 m ∧
    (∀ n : ℕ, is_multiple_of_18 n → digits_are_9_or_0 n → n ≤ m) ∧
    m = 900 ∧
    m / 18 = 50 := by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_18_with_9_and_0_digits_l1952_195205


namespace NUMINAMATH_CALUDE_bryden_received_is_ten_l1952_195249

/-- The amount a collector pays for a state quarter, as a multiple of its face value -/
def collector_rate : ℚ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/2

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 4

/-- The amount Bryden will receive from the collector in dollars -/
def bryden_received : ℚ := collector_rate * quarter_value * bryden_quarters

theorem bryden_received_is_ten : bryden_received = 10 := by
  sorry

end NUMINAMATH_CALUDE_bryden_received_is_ten_l1952_195249


namespace NUMINAMATH_CALUDE_equation_solution_l1952_195203

theorem equation_solution : 
  ∃! x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (4 : ℝ)^(3*x) = (64 : ℝ)^(4*x) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1952_195203


namespace NUMINAMATH_CALUDE_elective_course_arrangements_l1952_195259

def slots : ℕ := 6
def courses : ℕ := 3

theorem elective_course_arrangements : 
  (slots.factorial) / ((slots - courses).factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_arrangements_l1952_195259


namespace NUMINAMATH_CALUDE_rook_placement_count_l1952_195289

def chessboard_size : ℕ := 8
def num_rooks : ℕ := 6

theorem rook_placement_count :
  (Nat.choose chessboard_size num_rooks) * (Nat.factorial chessboard_size / Nat.factorial (chessboard_size - num_rooks)) = 564480 :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_count_l1952_195289


namespace NUMINAMATH_CALUDE_one_fourths_in_seven_halves_l1952_195244

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_seven_halves_l1952_195244


namespace NUMINAMATH_CALUDE_f_max_on_interval_f_greater_than_3x_solution_set_l1952_195264

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x + 2) * abs (x - 2)

-- Theorem for the maximum value of f(x) on [-3, 1]
theorem f_max_on_interval :
  ∃ (M : ℝ), M = 4 ∧ ∀ x ∈ Set.Icc (-3) 1, f x ≤ M :=
sorry

-- Theorem for the solution set of f(x) > 3x
theorem f_greater_than_3x_solution_set :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ (-4 < x ∧ x < 1)} :=
sorry

end NUMINAMATH_CALUDE_f_max_on_interval_f_greater_than_3x_solution_set_l1952_195264


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l1952_195279

-- Define the motion equation
def s (t : ℝ) : ℝ := t^3 + t^2 - 1

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3 : v 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l1952_195279


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1952_195201

def p (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 6*x - 18

theorem polynomial_divisibility :
  (∃ q : ℝ → ℝ, p = fun x ↦ (x - 3) * q x) ∧
  (∃ r : ℝ → ℝ, p = fun x ↦ (2*x^2 + 6) * r x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1952_195201


namespace NUMINAMATH_CALUDE_E_parity_l1952_195299

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 1) + E n

def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem E_parity : (isEven (E 2021) ∧ ¬isEven (E 2022) ∧ ¬isEven (E 2023)) := by sorry

end NUMINAMATH_CALUDE_E_parity_l1952_195299


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l1952_195240

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def line_through_point (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem ellipse_constant_product :
  ∀ k : ℝ,
  ∃ x1 y1 x2 y2 : ℝ,
  ellipse x1 y1 ∧ ellipse x2 y2 ∧
  line_through_point k 1 x1 y1 ∧
  line_through_point k 1 x2 y2 ∧
  x1 ≠ x2 →
  dot_product (17/8 - x1) (-y1) (17/8 - x2) (-y2) = 33/64 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l1952_195240


namespace NUMINAMATH_CALUDE_bottles_remaining_l1952_195257

/-- Calculates the number of bottles remaining in storage given the initial quantities and percentages sold. -/
theorem bottles_remaining (small_initial : ℕ) (big_initial : ℕ) (small_percent_sold : ℚ) (big_percent_sold : ℚ) :
  small_initial = 6000 →
  big_initial = 15000 →
  small_percent_sold = 11 / 100 →
  big_percent_sold = 12 / 100 →
  (small_initial - small_initial * small_percent_sold) + (big_initial - big_initial * big_percent_sold) = 18540 := by
sorry

end NUMINAMATH_CALUDE_bottles_remaining_l1952_195257


namespace NUMINAMATH_CALUDE_fraction_absolute_value_less_than_one_l1952_195226

theorem fraction_absolute_value_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |((x - y) / (1 - x * y))| < 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_absolute_value_less_than_one_l1952_195226


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_one_l1952_195208

theorem factorization_a_squared_minus_one (a : ℝ) : a^2 - 1 = (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_one_l1952_195208


namespace NUMINAMATH_CALUDE_time_for_one_smoothie_l1952_195210

/-- The time it takes to make a certain number of smoothies -/
def time_to_make_smoothies (n : ℕ) : ℕ := 55

/-- The number of smoothies made in the given time -/
def number_of_smoothies : ℕ := 5

/-- Proves that the time to make one smoothie is 11 minutes -/
theorem time_for_one_smoothie :
  time_to_make_smoothies number_of_smoothies / number_of_smoothies = 11 :=
sorry

end NUMINAMATH_CALUDE_time_for_one_smoothie_l1952_195210


namespace NUMINAMATH_CALUDE_power_of_power_equals_power_of_product_three_squared_to_fourth_power_l1952_195211

theorem power_of_power_equals_power_of_product (a m n : ℕ) :
  (a^m)^n = a^(m*n) :=
sorry

theorem three_squared_to_fourth_power :
  (3^2)^4 = 3^8 ∧ 3^8 = 6561 :=
sorry

end NUMINAMATH_CALUDE_power_of_power_equals_power_of_product_three_squared_to_fourth_power_l1952_195211


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l1952_195295

theorem concentric_circles_radius (r R : ℝ) (h1 : r = 4) 
  (h2 : (1.5 * R)^2 - (0.75 * r)^2 = 3.6 * (R^2 - r^2)) : R = 6 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l1952_195295


namespace NUMINAMATH_CALUDE_log_equation_solution_l1952_195230

theorem log_equation_solution :
  ∀ x : ℝ, (Real.log x - 3 * Real.log 4 = -3) → x = 0.064 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1952_195230


namespace NUMINAMATH_CALUDE_expression_simplification_l1952_195217

theorem expression_simplification (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 + 6 = 45*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1952_195217


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1952_195275

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

/-- The opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1952_195275


namespace NUMINAMATH_CALUDE_max_total_profit_max_avg_annual_profit_l1952_195231

/-- The total profit function for a coach operation -/
def total_profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

/-- The average annual profit function for a coach operation -/
def avg_annual_profit (x : ℕ+) : ℚ := (total_profit x) / x

/-- Theorem stating the year of maximum total profit -/
theorem max_total_profit :
  ∃ (x : ℕ+), ∀ (y : ℕ+), total_profit x ≥ total_profit y ∧ x = 9 :=
sorry

/-- Theorem stating the year of maximum average annual profit -/
theorem max_avg_annual_profit :
  ∃ (x : ℕ+), ∀ (y : ℕ+), avg_annual_profit x ≥ avg_annual_profit y ∧ x = 6 :=
sorry

end NUMINAMATH_CALUDE_max_total_profit_max_avg_annual_profit_l1952_195231


namespace NUMINAMATH_CALUDE_tangent_line_cubic_function_l1952_195276

/-- Given a cubic function f(x) = ax³ - 2x passing through the point (-1, 4),
    this theorem states that the equation of the tangent line to y = f(x) at x = -1
    is 8x + y + 4 = 0. -/
theorem tangent_line_cubic_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 2*x
  f (-1) = 4 →
  let m : ℝ := (6 * (-1)^2 + 2)  -- Derivative of f at x = -1
  let tangent_line : ℝ → ℝ := λ x ↦ m * (x - (-1)) + f (-1)
  ∀ x y, y = tangent_line x ↔ 8*x + y + 4 = 0 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_cubic_function_l1952_195276


namespace NUMINAMATH_CALUDE_eggs_per_group_l1952_195291

theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) : 
  total_eggs = 8 → num_groups = 4 → eggs_per_group = total_eggs / num_groups → eggs_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l1952_195291


namespace NUMINAMATH_CALUDE_line_translation_proof_l1952_195252

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

/-- The original line y = 4x -/
def originalLine : Line :=
  { slope := 4, intercept := 0 }

/-- The amount of downward translation -/
def translationAmount : ℝ := 5

theorem line_translation_proof :
  translateLine originalLine translationAmount = { slope := 4, intercept := -5 } := by
  sorry

end NUMINAMATH_CALUDE_line_translation_proof_l1952_195252


namespace NUMINAMATH_CALUDE_shelbys_driving_time_l1952_195222

/-- Shelby's driving problem -/
theorem shelbys_driving_time (speed_sun speed_rain : ℝ) (total_time total_distance : ℝ) 
  (h1 : speed_sun = 30)
  (h2 : speed_rain = 20)
  (h3 : total_time = 40)
  (h4 : total_distance = 16)
  (h5 : speed_sun > 0 ∧ speed_rain > 0) :
  ∃ (time_rain : ℝ), 
    time_rain = 24 ∧ 
    time_rain > 0 ∧ 
    time_rain < total_time ∧
    (speed_sun * (total_time - time_rain) / 60 + speed_rain * time_rain / 60 = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_shelbys_driving_time_l1952_195222


namespace NUMINAMATH_CALUDE_zongzi_purchase_theorem_l1952_195294

/-- Represents the properties of zongzi purchases in a supermarket. -/
structure ZongziPurchase where
  price_a : ℝ  -- Unit price of type A zongzi
  price_b : ℝ  -- Unit price of type B zongzi
  quantity_a : ℝ  -- Quantity of type A zongzi
  quantity_b : ℝ  -- Quantity of type B zongzi

/-- Theorem stating the properties of the zongzi purchase and the maximum purchase of type A zongzi. -/
theorem zongzi_purchase_theorem (z : ZongziPurchase) : 
  z.price_a * z.quantity_a = 1200 ∧ 
  z.price_b * z.quantity_b = 800 ∧ 
  z.quantity_b = z.quantity_a + 50 ∧ 
  z.price_a = 2 * z.price_b → 
  z.price_a = 8 ∧ z.price_b = 4 ∧ 
  (∀ m : ℕ, m ≤ 87 ↔ (m : ℝ) * 8 + (200 - m) * 4 ≤ 1150) :=
by sorry

#check zongzi_purchase_theorem

end NUMINAMATH_CALUDE_zongzi_purchase_theorem_l1952_195294


namespace NUMINAMATH_CALUDE_three_distinct_roots_l1952_195263

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem three_distinct_roots 
  (a b c x₁ x₂ : ℝ) 
  (h1 : ∃ x₁ x₂, x₁ ≠ x₂ ∧ (3*x₁^2 + 2*a*x₁ + b = 0) ∧ (3*x₂^2 + 2*a*x₂ + b = 0)) 
  (h2 : f a b c x₁ = x₁) 
  (h3 : x₁ < x₂) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 3*(f a b c x)^2 + 2*a*(f a b c x) + b = 0 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l1952_195263


namespace NUMINAMATH_CALUDE_equation_solution_l1952_195281

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ = -1) ∧ 
              (3 * x₂^2 - 6 * x₂ = -1) ∧ 
              (x₁ = 1 + Real.sqrt 6 / 3) ∧ 
              (x₂ = 1 - Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1952_195281


namespace NUMINAMATH_CALUDE_polynomial_root_magnitude_implies_a_range_l1952_195267

/-- A polynomial of degree 4 with real coefficients -/
structure Polynomial4 (a : ℝ) where
  coeff : Fin 5 → ℝ
  coeff_0 : coeff 0 = 2
  coeff_1 : coeff 1 = a
  coeff_2 : coeff 2 = 9
  coeff_3 : coeff 3 = a
  coeff_4 : coeff 4 = 2

/-- The roots of a polynomial -/
def roots (p : Polynomial4 a) : Finset ℂ := sorry

/-- Predicate to check if all roots are complex -/
def allRootsComplex (p : Polynomial4 a) : Prop :=
  ∀ r ∈ roots p, r.im ≠ 0

/-- Predicate to check if all root magnitudes are not equal to 1 -/
def allRootMagnitudesNotOne (p : Polynomial4 a) : Prop :=
  ∀ r ∈ roots p, Complex.abs r ≠ 1

/-- The main theorem -/
theorem polynomial_root_magnitude_implies_a_range (a : ℝ) (p : Polynomial4 a) 
    (h1 : allRootsComplex p) (h2 : allRootMagnitudesNotOne p) : 
    a ∈ Set.Ioo (-2 * Real.sqrt 10) (2 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_magnitude_implies_a_range_l1952_195267


namespace NUMINAMATH_CALUDE_unique_magnitude_for_quadratic_roots_l1952_195266

theorem unique_magnitude_for_quadratic_roots (z : ℂ) : 
  z^2 - 8*z + 37 = 0 → ∃! m : ℝ, ∃ w : ℂ, w^2 - 8*w + 37 = 0 ∧ Complex.abs w = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_for_quadratic_roots_l1952_195266


namespace NUMINAMATH_CALUDE_tuesday_temperature_l1952_195280

theorem tuesday_temperature
  (temp_tues wed thur fri : ℝ)
  (avg_tues_wed_thur : (temp_tues + wed + thur) / 3 = 52)
  (avg_wed_thur_fri : (wed + thur + fri) / 3 = 54)
  (fri_temp : fri = 53) :
  temp_tues = 47 := by
sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l1952_195280


namespace NUMINAMATH_CALUDE_min_sum_of_mn_l1952_195288

theorem min_sum_of_mn (m n : ℕ+) (h : m.val * n.val - 2 * m.val - 3 * n.val - 20 = 0) :
  ∃ (m' n' : ℕ+), m'.val * n'.val - 2 * m'.val - 3 * n'.val - 20 = 0 ∧ 
  m'.val + n'.val = 20 ∧ 
  ∀ (a b : ℕ+), a.val * b.val - 2 * a.val - 3 * b.val - 20 = 0 → a.val + b.val ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_mn_l1952_195288


namespace NUMINAMATH_CALUDE_tan_equality_implies_x_120_l1952_195220

theorem tan_equality_implies_x_120 (x : Real) :
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 120 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_implies_x_120_l1952_195220


namespace NUMINAMATH_CALUDE_function_properties_l1952_195272

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := Real.log (a^x - b^x) / Real.log 10

-- State the theorem
theorem function_properties (a b : ℝ) (ha : a > 1) (hb : 0 < b) (hab : b < 1) :
  -- 1. Domain of f is (0, +∞)
  (∀ x : ℝ, x > 0 → (a^x - b^x > 0)) ∧
  -- 2. No two distinct points with same y-value
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a b x₁ ≠ f a b x₂) ∧
  -- 3. Condition for f to be positive on (1, +∞)
  (a ≥ b + 1 → ∀ x : ℝ, x > 1 → f a b x > 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1952_195272


namespace NUMINAMATH_CALUDE_set_equality_proof_l1952_195200

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}

def B (a : ℝ) : Set ℝ := {x | x < a}

def C : Set ℝ := {2, 3}

theorem set_equality_proof : 
  ∀ a : ℝ, (A ∪ B a = A) ↔ (a ∈ C) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_proof_l1952_195200


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l1952_195290

/-- Given a line segment from (-3, 9) to (4, 10) parameterized by x = at + b and y = ct + d,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (-3, 9), prove that a^2 + b^2 + c^2 + d^2 = 140 -/
theorem line_segment_param_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  b = -3 →
  d = 9 →
  a + b = 4 →
  c + d = 10 →
  a^2 + b^2 + c^2 + d^2 = 140 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l1952_195290


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l1952_195233

theorem multiply_and_simplify (x : ℝ) : (x^4 + 16*x^2 + 256) * (x^2 - 16) = x^4 + 32*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l1952_195233


namespace NUMINAMATH_CALUDE_no_equal_shards_l1952_195287

theorem no_equal_shards : ¬∃ (x y : ℕ), 17 * x + 18 * (35 - y) = 17 * (25 - x) + 18 * y := by
  sorry

end NUMINAMATH_CALUDE_no_equal_shards_l1952_195287


namespace NUMINAMATH_CALUDE_inequality_proof_l1952_195236

theorem inequality_proof (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) : 
  (Real.sqrt x + Real.sqrt y) * (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y)) > 1 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1952_195236


namespace NUMINAMATH_CALUDE_no_integer_solution_l1952_195227

theorem no_integer_solution : 
  ¬ ∃ (x y z : ℤ), (x - y)^3 + (y - z)^3 + (z - x)^3 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1952_195227


namespace NUMINAMATH_CALUDE_count_divisible_by_3_or_5_is_28_l1952_195260

/-- The count of numbers from 1 to 60 that are divisible by either 3 or 5 or both -/
def count_divisible_by_3_or_5 : ℕ :=
  let n := 60
  let divisible_by_3 := n / 3
  let divisible_by_5 := n / 5
  let divisible_by_15 := n / 15
  divisible_by_3 + divisible_by_5 - divisible_by_15

theorem count_divisible_by_3_or_5_is_28 : count_divisible_by_3_or_5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_3_or_5_is_28_l1952_195260


namespace NUMINAMATH_CALUDE_equation_solution_l1952_195262

theorem equation_solution :
  ∃ x : ℚ, 
    (((x / 128 + (1 + 2 / 7)) / (5 - 4 * (2 / 21) * 0.75)) / 
    ((1 / 3 + 5 / 7 * 1.4) / ((4 - 2 * (2 / 3)) * 3)) = 4.5) ∧ 
    x = 1440 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1952_195262


namespace NUMINAMATH_CALUDE_teal_opposite_yellow_l1952_195285

-- Define the colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Violet
  | Teal
  | Lime

-- Define the positions on the cube
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

-- Define a cube as a function from Position to Color
def Cube := Position → Color

-- Define the property of having distinct colors for each face
def has_distinct_colors (cube : Cube) : Prop :=
  ∀ p1 p2 : Position, p1 ≠ p2 → cube p1 ≠ cube p2

-- Define the views of the cube
def first_view (cube : Cube) : Prop :=
  cube Position.Top = Color.Blue ∧
  cube Position.Front = Color.Violet ∧
  cube Position.Right = Color.Yellow

def second_view (cube : Cube) : Prop :=
  cube Position.Top = Color.Blue ∧
  cube Position.Front = Color.Orange ∧
  cube Position.Right = Color.Yellow

def third_view (cube : Cube) : Prop :=
  cube Position.Top = Color.Blue ∧
  cube Position.Front = Color.Lime ∧
  cube Position.Right = Color.Yellow

-- Define the theorem
theorem teal_opposite_yellow (cube : Cube) :
  has_distinct_colors cube →
  first_view cube →
  second_view cube →
  third_view cube →
  cube Position.Left = Color.Teal →
  cube Position.Right = Color.Yellow :=
by sorry

end NUMINAMATH_CALUDE_teal_opposite_yellow_l1952_195285


namespace NUMINAMATH_CALUDE_pizza_theorem_l1952_195278

/-- The number of pizzas ordered for a class celebration --/
def pizza_problem (num_boys : ℕ) (num_girls : ℕ) (boys_pizzas : ℕ) : Prop :=
  num_girls = 11 ∧
  num_boys > num_girls ∧
  boys_pizzas = 10 ∧
  ∃ (total_pizzas : ℚ),
    total_pizzas = boys_pizzas + (num_girls : ℚ) * (boys_pizzas : ℚ) / (2 * num_boys : ℚ) ∧
    total_pizzas = 11

theorem pizza_theorem :
  ∃ (num_boys : ℕ), pizza_problem num_boys 11 10 :=
sorry

end NUMINAMATH_CALUDE_pizza_theorem_l1952_195278


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1952_195239

/-- Proves that a car's initial fuel efficiency is 24 miles per gallon given specific conditions -/
theorem car_fuel_efficiency 
  (initial_efficiency : ℝ) 
  (improvement_factor : ℝ) 
  (tank_capacity : ℝ) 
  (additional_miles : ℝ) 
  (h1 : improvement_factor = 4/3) 
  (h2 : tank_capacity = 12) 
  (h3 : additional_miles = 96) 
  (h4 : tank_capacity * initial_efficiency * improvement_factor - 
        tank_capacity * initial_efficiency = additional_miles) : 
  initial_efficiency = 24 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1952_195239


namespace NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l1952_195268

theorem tangent_and_trigonometric_identity (α β : Real) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (Real.pi - β) = 3/2) :
  (Real.tan α = -7/4) ∧ 
  ((Real.sin (Real.pi/2 + α) - Real.sin (Real.pi + α)) / (Real.cos α + 2 * Real.sin α) = 3/10) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l1952_195268


namespace NUMINAMATH_CALUDE_friends_games_total_l1952_195237

/-- The total number of games Katie's friends have -/
def total_friends_games (new_friends_games old_friends_games : ℕ) : ℕ :=
  new_friends_games + old_friends_games

/-- Theorem: Katie's friends have 141 games in total -/
theorem friends_games_total :
  total_friends_games 88 53 = 141 := by
  sorry

end NUMINAMATH_CALUDE_friends_games_total_l1952_195237
