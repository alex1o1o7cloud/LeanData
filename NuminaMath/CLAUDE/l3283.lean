import Mathlib

namespace NUMINAMATH_CALUDE_inverse_value_equivalence_l3283_328351

-- Define the function f
def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

-- Theorem stating that finding f⁻¹(-3.5) is equivalent to solving 7x³ - 2x² + 5x - 5.5 = 0
theorem inverse_value_equivalence :
  ∀ x : ℝ, f x = -3.5 ↔ 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
by
  sorry

-- Note: The actual inverse function is not defined as it's not expressible in elementary functions

end NUMINAMATH_CALUDE_inverse_value_equivalence_l3283_328351


namespace NUMINAMATH_CALUDE_candy_distribution_l3283_328354

theorem candy_distribution (S M L : ℕ) 
  (total : S + M + L = 110)
  (without_jelly : S + L = 100)
  (relation : M + L = S + M + 20) :
  S = 40 ∧ L = 60 ∧ M = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3283_328354


namespace NUMINAMATH_CALUDE_percentage_relation_l3283_328324

theorem percentage_relation (x y : ℕ+) (h1 : y * x = 100 * 100) (h2 : y = 125) :
  (y : ℝ) / ((25 : ℝ) / 100 * x) * 100 = 625 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3283_328324


namespace NUMINAMATH_CALUDE_percentage_of_seats_filled_l3283_328346

/-- Given a public show with 600 seats in total and 330 vacant seats,
    prove that 45% of the seats were filled. -/
theorem percentage_of_seats_filled (total_seats : ℕ) (vacant_seats : ℕ) : 
  total_seats = 600 →
  vacant_seats = 330 →
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_seats_filled_l3283_328346


namespace NUMINAMATH_CALUDE_triple_value_equation_l3283_328345

theorem triple_value_equation (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → x = 5 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_triple_value_equation_l3283_328345


namespace NUMINAMATH_CALUDE_sequence_sum_l3283_328338

theorem sequence_sum (n x y : ℝ) : 
  (3 + 16 + 33 + (n + 1) + x + y) / 6 = 25 → n + x + y = 97 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3283_328338


namespace NUMINAMATH_CALUDE_line_relationship_l3283_328300

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- Define the intersecting relationship between lines
variable (intersecting : Line → Line → Prop)

-- Define the skew relationship between lines
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem line_relationship (a b c : Line) 
  (h1 : parallel a c) 
  (h2 : ¬ parallel b c) : 
  intersecting a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3283_328300


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3283_328329

theorem simplify_fraction_product : (240 / 12) * (5 / 150) * (12 / 3) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3283_328329


namespace NUMINAMATH_CALUDE_binomial_coefficient_formula_l3283_328377

theorem binomial_coefficient_formula (n k : ℕ) (h1 : k < n) (h2 : 0 < k) :
  Nat.choose n k = (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_formula_l3283_328377


namespace NUMINAMATH_CALUDE_a_squared_plus_2a_plus_1_is_perfect_square_l3283_328393

/-- Definition of a perfect square trinomial -/
def is_perfect_square_trinomial (p : ℝ → ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, p x = (q x)^2 ∧ (∃ a b : ℝ, q x = a*x + b)

/-- The expression a^2 + 2a + 1 is a perfect square trinomial -/
theorem a_squared_plus_2a_plus_1_is_perfect_square :
  is_perfect_square_trinomial (λ a : ℝ ↦ a^2 + 2*a + 1) :=
sorry

end NUMINAMATH_CALUDE_a_squared_plus_2a_plus_1_is_perfect_square_l3283_328393


namespace NUMINAMATH_CALUDE_picture_books_count_l3283_328359

theorem picture_books_count (total : ℕ) (fiction : ℕ) : 
  total = 35 →
  fiction = 5 →
  let nonfiction := fiction + 4
  let autobiographies := 2 * fiction
  let other_books := fiction + nonfiction + autobiographies
  total - other_books = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_picture_books_count_l3283_328359


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3283_328337

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3283_328337


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3283_328398

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of the given cistern -/
theorem cistern_wet_surface_area :
  let length : ℝ := 6
  let width : ℝ := 4
  let depth : ℝ := 1.25
  wetSurfaceArea length width depth = 49 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3283_328398


namespace NUMINAMATH_CALUDE_longest_all_green_interval_is_20_seconds_l3283_328380

/-- Represents a traffic light with its timing properties -/
structure TrafficLight where
  greenDuration : ℝ
  yellowDuration : ℝ
  redDuration : ℝ
  cycleStart : ℝ

/-- Calculates the longest interval during which all lights are green -/
def longestAllGreenInterval (lights : List TrafficLight) : ℝ :=
  sorry

/-- The main theorem stating the longest interval of all green lights -/
theorem longest_all_green_interval_is_20_seconds :
  let lights : List TrafficLight := List.range 8 |>.map (fun i =>
    { greenDuration := 90  -- 1.5 minutes in seconds
      yellowDuration := 3
      redDuration := 90    -- 1.5 minutes in seconds
      cycleStart := i * 10 -- Each light starts 10 seconds after the previous
    })
  longestAllGreenInterval lights = 20 := by
  sorry

end NUMINAMATH_CALUDE_longest_all_green_interval_is_20_seconds_l3283_328380


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3283_328326

theorem pure_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := (x^2 - 1) + (x - 1) * I
  (∃ (y : ℝ), z = y * I) → x = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3283_328326


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3283_328311

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_lengths (x : ℕ) :
  (x > 0) →
  (is_valid_triangle (3 * x) 10 (x^2)) →
  (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3283_328311


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3283_328358

theorem polynomial_factorization (a b c : ℚ) : 
  b^2 - c^2 + a*(a + 2*b) = (a + b + c)*(a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3283_328358


namespace NUMINAMATH_CALUDE_largest_divisor_of_prime_square_difference_l3283_328308

theorem largest_divisor_of_prime_square_difference (p q : ℕ) 
  (hp : Prime p) (hq : Prime q) (h_order : q < p) : 
  (∀ (d : ℕ), d > 2 → ∃ (p' q' : ℕ), Prime p' ∧ Prime q' ∧ q' < p' ∧ ¬(d ∣ (p'^2 - q'^2))) ∧ 
  (∀ (p' q' : ℕ), Prime p' → Prime q' → q' < p' → 2 ∣ (p'^2 - q'^2)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_prime_square_difference_l3283_328308


namespace NUMINAMATH_CALUDE_simplify_expression_l3283_328336

theorem simplify_expression : 
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3283_328336


namespace NUMINAMATH_CALUDE_sequence_general_term_l3283_328392

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℝ := 2 * n^2 + n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℝ := 4 * n - 1

/-- Theorem stating that a_n = 4n - 1 given S_n = 2n^2 + n -/
theorem sequence_general_term (n : ℕ) : a n = S (n + 1) - S n := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3283_328392


namespace NUMINAMATH_CALUDE_inequality_proof_l3283_328343

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2)/c + (c^2 - b^2)/a + (a^2 - c^2)/b ≥ 3*a - 4*b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3283_328343


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l3283_328376

theorem sin_cos_sum_equals_half : 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l3283_328376


namespace NUMINAMATH_CALUDE_inequality_proof_l3283_328319

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 3*b^3) / (5*a + b) + (b^3 + 3*c^3) / (5*b + c) + (c^3 + 3*a^3) / (5*c + a) ≥ 
  2/3 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3283_328319


namespace NUMINAMATH_CALUDE_minimum_dice_value_l3283_328386

theorem minimum_dice_value (X : ℕ) : (1 + 5 + X > 2 + 4 + 5) ↔ X ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_minimum_dice_value_l3283_328386


namespace NUMINAMATH_CALUDE_intersection_m_value_l3283_328353

theorem intersection_m_value (x y : ℝ) (m : ℝ) : 
  (3 * x + y = m) →
  (-0.75 * x + y = -22) →
  (x = 6) →
  (m = 0.5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_m_value_l3283_328353


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3283_328321

theorem min_perimeter_triangle (a b x : ℕ) (h1 : a = 40) (h2 : b = 50) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, y ≠ x → a + b + y > a + b + x) → 
  a + b + x = 101 := by
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3283_328321


namespace NUMINAMATH_CALUDE_kronecker_irreducibility_criterion_l3283_328382

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The degree of a polynomial -/
noncomputable def degree (f : IntPolynomial) : ℕ := sorry

/-- Checks if a polynomial g is a proper divisor of f -/
def is_proper_divisor (f g : IntPolynomial) : Prop := sorry

/-- Kronecker's irreducibility criterion -/
theorem kronecker_irreducibility_criterion (f : IntPolynomial) :
  (∀ g : IntPolynomial, is_proper_divisor f g → degree g > (degree f) / 2) ↔
  (∀ g : IntPolynomial, ¬(is_proper_divisor f g)) :=
sorry

end NUMINAMATH_CALUDE_kronecker_irreducibility_criterion_l3283_328382


namespace NUMINAMATH_CALUDE_horizontal_shift_shift_left_3_units_l3283_328328

-- Define the original function
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define the transformed function
def g (x : ℝ) : ℝ := f (2 * x + 3)

-- Theorem stating the horizontal shift
theorem horizontal_shift :
  ∀ x : ℝ, g x = f (x + 3) :=
by
  sorry

-- Theorem stating that the shift is 3 units to the left
theorem shift_left_3_units :
  ∀ x : ℝ, g x = f (x + 3) ∧ (x + 3) - x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_horizontal_shift_shift_left_3_units_l3283_328328


namespace NUMINAMATH_CALUDE_conference_handshakes_result_l3283_328327

/-- The number of handshakes at a conference with specified conditions -/
def conference_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let total_possible_handshakes := total_people * (total_people - 1)
  let handshakes_not_occurring := num_companies * reps_per_company * (reps_per_company - 1)
  (total_possible_handshakes - handshakes_not_occurring) / 2

/-- Theorem stating the number of handshakes for the given conference conditions -/
theorem conference_handshakes_result :
  conference_handshakes 3 5 = 75 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_result_l3283_328327


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3283_328373

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 5 * k ∧ n % 3 = 1

theorem smallest_valid_number : (∀ m : ℕ, m > 0 ∧ m < 10 → ¬(is_valid m)) ∧ is_valid 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3283_328373


namespace NUMINAMATH_CALUDE_paint_calculation_l3283_328331

theorem paint_calculation (total_paint : ℚ) : 
  (2 / 3 : ℚ) * total_paint + (1 / 5 : ℚ) * ((1 / 3 : ℚ) * total_paint) = 264 → 
  total_paint = 360 := by
sorry

end NUMINAMATH_CALUDE_paint_calculation_l3283_328331


namespace NUMINAMATH_CALUDE_original_profit_percentage_l3283_328381

/-- Calculates the profit percentage given the cost price and selling price -/
def profitPercentage (costPrice sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem original_profit_percentage
  (costPrice : ℚ)
  (sellingPrice : ℚ)
  (h1 : costPrice = 80)
  (h2 : profitPercentage (costPrice * (1 - 0.2)) (sellingPrice - 16.8) = 30) :
  profitPercentage costPrice sellingPrice = 25 := by
  sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l3283_328381


namespace NUMINAMATH_CALUDE_sal_and_phil_combined_money_l3283_328379

/-- Given that Kim has 40% more money than Sal, Sal has 20% less money than Phil,
    and Kim has $1.12, prove that Sal and Phil have a combined total of $1.80. -/
theorem sal_and_phil_combined_money :
  ∀ (kim sal phil : ℝ),
  kim = 1.4 * sal →
  sal = 0.8 * phil →
  kim = 1.12 →
  sal + phil = 1.80 :=
by
  sorry

end NUMINAMATH_CALUDE_sal_and_phil_combined_money_l3283_328379


namespace NUMINAMATH_CALUDE_bank_interest_rate_determination_l3283_328340

/-- Proves that given two equal deposits with the same interest rate but different time periods, 
    if the difference in interest is known, then the interest rate can be determined. -/
theorem bank_interest_rate_determination 
  (principal : ℝ) 
  (time1 time2 : ℝ) 
  (interest_difference : ℝ) : 
  principal = 640 →
  time1 = 3.5 →
  time2 = 5 →
  interest_difference = 144 →
  ∃ (rate : ℝ), 
    (principal * time2 * rate / 100 - principal * time1 * rate / 100 = interest_difference) ∧
    rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_bank_interest_rate_determination_l3283_328340


namespace NUMINAMATH_CALUDE_percentage_passed_all_subjects_l3283_328304

theorem percentage_passed_all_subjects 
  (fail_hindi : Real) 
  (fail_english : Real) 
  (fail_both : Real) 
  (fail_math : Real) 
  (h1 : fail_hindi = 0.2) 
  (h2 : fail_english = 0.7) 
  (h3 : fail_both = 0.1) 
  (h4 : fail_math = 0.5) : 
  (1 - (fail_hindi + fail_english - fail_both)) * (1 - fail_math) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_subjects_l3283_328304


namespace NUMINAMATH_CALUDE_percentage_comparison_l3283_328333

theorem percentage_comparison (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : 0.02 * A > 0.03 * B) :
  0.05 * A > 0.07 * B := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l3283_328333


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3283_328390

theorem slope_angle_of_line (x y : ℝ) :
  let line_eq := x * Real.tan (π / 6) + y - 7 = 0
  let slope := -Real.tan (π / 6)
  let slope_angle := Real.arctan (-slope)
  slope_angle = 5 * π / 6 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3283_328390


namespace NUMINAMATH_CALUDE_colored_square_theorem_l3283_328317

/-- Represents a coloring of a square grid -/
def Coloring (n : ℕ) := Fin (n^2 + 1) → Fin (n^2 + 1)

/-- Counts the number of distinct colors in a given row or column -/
def distinctColors (n : ℕ) (c : Coloring n) (isRow : Bool) (index : Fin (n^2 + 1)) : ℕ :=
  sorry

theorem colored_square_theorem (n : ℕ) (c : Coloring n) :
  ∃ (isRow : Bool) (index : Fin (n^2 + 1)), distinctColors n c isRow index ≥ n + 1 :=
sorry

end NUMINAMATH_CALUDE_colored_square_theorem_l3283_328317


namespace NUMINAMATH_CALUDE_prob_two_pairs_eq_nine_twentytwo_l3283_328349

/-- Represents the number of socks of each color --/
def socks_per_color : ℕ := 3

/-- Represents the number of colors --/
def num_colors : ℕ := 4

/-- Represents the total number of socks --/
def total_socks : ℕ := socks_per_color * num_colors

/-- Represents the number of socks drawn --/
def socks_drawn : ℕ := 5

/-- Calculates the probability of drawing exactly two pairs of socks with different colors --/
def prob_two_pairs : ℚ :=
  (num_colors.choose 2 * (num_colors - 2).choose 1 * socks_per_color.choose 2 * socks_per_color.choose 2 * socks_per_color.choose 1) /
  (total_socks.choose socks_drawn)

theorem prob_two_pairs_eq_nine_twentytwo : prob_two_pairs = 9 / 22 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_pairs_eq_nine_twentytwo_l3283_328349


namespace NUMINAMATH_CALUDE_light_glow_time_l3283_328347

/-- The number of seconds between 1:57:58 am and 3:20:47 am -/
def total_seconds : ℕ := 4969

/-- The maximum number of times the light glowed -/
def max_glows : ℚ := 155.28125

/-- The time it takes for one glow in seconds -/
def time_per_glow : ℕ := 32

theorem light_glow_time :
  (total_seconds : ℚ) / max_glows = time_per_glow := by sorry

end NUMINAMATH_CALUDE_light_glow_time_l3283_328347


namespace NUMINAMATH_CALUDE_smallest_expressible_proof_l3283_328363

/-- Represents the number of marbles in each box type -/
def box_sizes : Finset ℕ := {13, 11, 7}

/-- Checks if a number can be expressed as a non-negative integer combination of box sizes -/
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 13 * a + 11 * b + 7 * c

/-- The smallest number such that all larger numbers are expressible -/
def smallest_expressible : ℕ := 30

theorem smallest_expressible_proof :
  (∀ m : ℕ, m > smallest_expressible → is_expressible m) ∧
  (∀ k : ℕ, k < smallest_expressible → ∃ n : ℕ, n > k ∧ ¬is_expressible n) :=
sorry

end NUMINAMATH_CALUDE_smallest_expressible_proof_l3283_328363


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3283_328352

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2) + f (x * y) = f x * f y + y * f x + x * f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    is either constantly zero or the negation function. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = -x) := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3283_328352


namespace NUMINAMATH_CALUDE_min_value_expression_l3283_328339

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = y) :
  ∃ (min : ℝ), min = 0 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a = b →
    (a + 1/b) * (a + 1/b - 2) + (b + 1/a) * (b + 1/a - 2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3283_328339


namespace NUMINAMATH_CALUDE_horner_third_intermediate_value_l3283_328330

def horner_polynomial (a : List ℚ) (x : ℚ) : ℚ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def horner_intermediate (a : List ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  (a.take (n + 1)).foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_third_intermediate_value :
  let f (x : ℚ) := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64
  let coeffs := [1, -12, 60, -160, 240, -192, 64]
  let x := 2
  horner_intermediate coeffs x 3 = -80 := by sorry

end NUMINAMATH_CALUDE_horner_third_intermediate_value_l3283_328330


namespace NUMINAMATH_CALUDE_factorial_sum_eq_power_of_three_l3283_328312

theorem factorial_sum_eq_power_of_three (a b c d : ℕ) : 
  a ≤ b → b ≤ c → a.factorial + b.factorial + c.factorial = 3^d →
  ((a, b, c, d) = (1, 1, 1, 1) ∨ (a, b, c, d) = (1, 2, 3, 2) ∨ (a, b, c, d) = (1, 2, 4, 3)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_eq_power_of_three_l3283_328312


namespace NUMINAMATH_CALUDE_photograph_perimeter_l3283_328362

/-- 
Given a rectangular photograph with a border, this theorem proves that 
if the total area with a 1-inch border is m square inches, and 
the total area with a 3-inch border is (m + 52) square inches, 
then the perimeter of the photograph is 10 inches.
-/
theorem photograph_perimeter 
  (w l m : ℝ) 
  (h1 : (w + 2) * (l + 2) = m) 
  (h2 : (w + 6) * (l + 6) = m + 52) : 
  2 * (w + l) = 10 :=
sorry

end NUMINAMATH_CALUDE_photograph_perimeter_l3283_328362


namespace NUMINAMATH_CALUDE_education_funds_calculation_l3283_328334

/-- The GDP of China in 2012 in trillion yuan -/
def gdp_2012 : ℝ := 43.5

/-- The proportion of national financial education funds in GDP -/
def education_funds_proportion : ℝ := 0.04

/-- The national financial education funds expenditure for 2012 in billion yuan -/
def education_funds_2012 : ℝ := gdp_2012 * 1000 * education_funds_proportion

/-- Proof that the national financial education funds expenditure for 2012 
    is equal to 1.74 × 10^4 billion yuan -/
theorem education_funds_calculation : 
  education_funds_2012 = 1.74 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_education_funds_calculation_l3283_328334


namespace NUMINAMATH_CALUDE_three_halves_equals_one_point_five_l3283_328322

theorem three_halves_equals_one_point_five : (3 : ℚ) / 2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_three_halves_equals_one_point_five_l3283_328322


namespace NUMINAMATH_CALUDE_bag_contents_theorem_l3283_328341

/-- Represents the contents of a bag of colored balls. -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of selecting two red balls. -/
def probTwoRed (bag : BagContents) : ℚ :=
  (bag.red.choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- Calculates the probability of selecting one red and one yellow ball. -/
def probRedYellow (bag : BagContents) : ℚ :=
  ((bag.red.choose 1 * bag.yellow.choose 1) : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- Calculates the expected value of the number of red balls selected. -/
def expectedRedBalls (bag : BagContents) : ℚ :=
  0 * ((bag.yellow + bag.green).choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ) +
  1 * ((bag.red.choose 1 * (bag.yellow + bag.green).choose 1) : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ) +
  2 * (bag.red.choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- The main theorem stating the properties of the bag contents and expected value. -/
theorem bag_contents_theorem (bag : BagContents) :
  bag.red = 4 ∧ 
  probTwoRed bag = 1/6 ∧ 
  probRedYellow bag = 1/3 → 
  bag.yellow - bag.green = 1 ∧ 
  expectedRedBalls bag = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_bag_contents_theorem_l3283_328341


namespace NUMINAMATH_CALUDE_exists_quadratic_through_point_l3283_328396

-- Define a quadratic function
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

-- State the theorem
theorem exists_quadratic_through_point :
  ∃ (a b c : ℝ), a > 0 ∧ quadratic_function a b c 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_quadratic_through_point_l3283_328396


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3283_328360

/-- Given a school with 460 students, where 325 play football, 175 play cricket, 
    and 50 play neither, prove that 90 students play both sports. -/
theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) 
  (h1 : total = 460)
  (h2 : football = 325)
  (h3 : cricket = 175)
  (h4 : neither = 50)
  : total = football + cricket - 90 + neither := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3283_328360


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l3283_328394

def is_prime (n : ℕ) : Prop := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def has_no_prime_factor_less_than (n m : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square_with_large_factors : 
  ∃ (n : ℕ), n > 0 ∧ 
  ¬ is_prime n ∧ 
  ¬ is_perfect_square n ∧ 
  has_no_prime_factor_less_than n 50 ∧ 
  ∀ (m : ℕ), m > 0 ∧ 
    m < n ∧ 
    ¬ is_prime m ∧ 
    ¬ is_perfect_square m → 
    ¬ has_no_prime_factor_less_than m 50 :=
by
  use 3127
  sorry

#eval 3127

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l3283_328394


namespace NUMINAMATH_CALUDE_double_elimination_tournament_players_l3283_328384

/-- Represents a double elimination tournament -/
structure DoubleEliminationTournament where
  num_players : ℕ
  num_matches : ℕ

/-- Theorem: In a double elimination tournament with 63 matches, there are 32 players -/
theorem double_elimination_tournament_players (t : DoubleEliminationTournament) 
  (h : t.num_matches = 63) : t.num_players = 32 := by
  sorry

end NUMINAMATH_CALUDE_double_elimination_tournament_players_l3283_328384


namespace NUMINAMATH_CALUDE_f_extrema_l3283_328316

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 1

theorem f_extrema : 
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l3283_328316


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l3283_328399

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 47)
  (h3 : max_ac_no_stripes = 45) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = 8 ∧ 
    ∀ (cars_with_stripes : ℕ), 
      cars_with_stripes ≥ min_cars_with_stripes →
      cars_with_stripes + max_ac_no_stripes ≥ total_cars - cars_without_ac :=
by
  sorry

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l3283_328399


namespace NUMINAMATH_CALUDE_intersection_point_l3283_328389

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 - 1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

theorem intersection_point (m : ℝ × ℝ) (hA : m ∈ A) (hB : m ∈ B) : m = (4, 7) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3283_328389


namespace NUMINAMATH_CALUDE_fraction_ordering_l3283_328355

def t₁ : ℚ := (100^100 + 1) / (100^90 + 1)
def t₂ : ℚ := (100^99 + 1) / (100^89 + 1)
def t₃ : ℚ := (100^101 + 1) / (100^91 + 1)
def t₄ : ℚ := (101^101 + 1) / (101^91 + 1)
def t₅ : ℚ := (101^100 + 1) / (101^90 + 1)
def t₆ : ℚ := (99^99 + 1) / (99^89 + 1)
def t₇ : ℚ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering :
  t₆ < t₇ ∧ t₇ < t₂ ∧ t₂ < t₁ ∧ t₁ < t₃ ∧ t₃ < t₅ ∧ t₅ < t₄ :=
by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3283_328355


namespace NUMINAMATH_CALUDE_driveway_snow_volume_l3283_328301

/-- Calculates the total volume of snow on a driveway with given dimensions and snow depths -/
theorem driveway_snow_volume 
  (driveway_length : ℝ) 
  (driveway_width : ℝ) 
  (section1_length : ℝ) 
  (section1_depth : ℝ) 
  (section2_length : ℝ) 
  (section2_depth : ℝ) 
  (h1 : driveway_length = 30) 
  (h2 : driveway_width = 3) 
  (h3 : section1_length = 10) 
  (h4 : section1_depth = 1) 
  (h5 : section2_length = 20) 
  (h6 : section2_depth = 0.5) 
  (h7 : section1_length + section2_length = driveway_length) : 
  section1_length * driveway_width * section1_depth + 
  section2_length * driveway_width * section2_depth = 60 :=
by
  sorry

#check driveway_snow_volume

end NUMINAMATH_CALUDE_driveway_snow_volume_l3283_328301


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_four_l3283_328378

theorem fraction_equality_implies_x_equals_four (x : ℝ) :
  (x ≠ 0) → (x ≠ -2) → (6 / (x + 2) = 4 / x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_four_l3283_328378


namespace NUMINAMATH_CALUDE_equation_solution_l3283_328342

theorem equation_solution :
  ∃ x : ℝ, (3 / x - 2 / (x - 2) = 0) ∧ (x = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3283_328342


namespace NUMINAMATH_CALUDE_f_increasing_f_two_zeros_l3283_328302

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x + 1) + a * x

-- Statement 1: f(x) is increasing when a > 2
theorem f_increasing (a : ℝ) (h : a > 2) : 
  StrictMono (f a) := by sorry

-- Statement 2: f(x) has two zeros iff a ∈ (0, 2)
theorem f_two_zeros (a : ℝ) : 
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 0 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_two_zeros_l3283_328302


namespace NUMINAMATH_CALUDE_some_number_proof_l3283_328370

theorem some_number_proof (x y : ℝ) (h1 : 5 * x + 3 = 10 * x - y) (h2 : x = 4) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_some_number_proof_l3283_328370


namespace NUMINAMATH_CALUDE_alice_minimum_score_l3283_328314

def minimum_score (scores : List Float) (target_average : Float) (total_terms : Nat) : Float :=
  let sum_scores := scores.sum
  let remaining_terms := total_terms - scores.length
  (target_average * total_terms.toFloat - sum_scores) / remaining_terms.toFloat

theorem alice_minimum_score :
  let alice_scores := [84, 88, 82, 79]
  let target_average := 85
  let total_terms := 5
  minimum_score alice_scores target_average total_terms = 92 := by
  sorry

end NUMINAMATH_CALUDE_alice_minimum_score_l3283_328314


namespace NUMINAMATH_CALUDE_unique_solution_m_l3283_328383

theorem unique_solution_m (m : ℚ) : 
  (∃! x, (x - 3) / (m * x + 4) = 2 * x) ↔ m = 49 / 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_m_l3283_328383


namespace NUMINAMATH_CALUDE_solve_equation_l3283_328385

theorem solve_equation : ∃ x : ℝ, 90 + (x * 12) / (180 / 3) = 91 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3283_328385


namespace NUMINAMATH_CALUDE_fish_catch_total_l3283_328305

/-- The total number of fish caught by Leo, Agrey, and Sierra -/
def total_fish (leo agrey sierra : ℕ) : ℕ := leo + agrey + sierra

/-- Theorem stating the total number of fish caught given the conditions -/
theorem fish_catch_total :
  ∀ (leo agrey sierra : ℕ),
    leo = 40 →
    agrey = leo + 20 →
    sierra = agrey + 15 →
    total_fish leo agrey sierra = 175 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_total_l3283_328305


namespace NUMINAMATH_CALUDE_brendan_afternoon_catch_brendan_fishing_proof_l3283_328368

theorem brendan_afternoon_catch (morning_catch : ℕ) (thrown_back : ℕ) (dad_catch : ℕ) (total_catch : ℕ) : ℕ :=
  let kept_morning := morning_catch - thrown_back
  let afternoon_catch := total_catch - kept_morning - dad_catch
  afternoon_catch

theorem brendan_fishing_proof :
  let morning_catch := 8
  let thrown_back := 3
  let dad_catch := 13
  let total_catch := 23
  brendan_afternoon_catch morning_catch thrown_back dad_catch total_catch = 5 := by
  sorry

end NUMINAMATH_CALUDE_brendan_afternoon_catch_brendan_fishing_proof_l3283_328368


namespace NUMINAMATH_CALUDE_f_3_range_l3283_328361

/-- Given a quadratic function f(x) = ax^2 - c with specific constraints on f(1) and f(2),
    we prove that f(3) lies within a certain range. -/
theorem f_3_range (a c : ℝ) (h1 : -4 ≤ a - c ∧ a - c ≤ -1) (h2 : -1 ≤ 4*a - c ∧ 4*a - c ≤ 5) :
  -1 ≤ 9*a - c ∧ 9*a - c ≤ 20 := by
  sorry

#check f_3_range

end NUMINAMATH_CALUDE_f_3_range_l3283_328361


namespace NUMINAMATH_CALUDE_triangle_area_equality_l3283_328318

theorem triangle_area_equality (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x^2 + y^2 = 49)
  (h5 : y^2 + y*z + z^2 = 36)
  (h6 : x^2 + Real.sqrt 3 * x * z + z^2 = 25) :
  2*x*y + Real.sqrt 3 * y*z + z*x = 24 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_equality_l3283_328318


namespace NUMINAMATH_CALUDE_sum_parity_from_cube_sum_parity_l3283_328306

theorem sum_parity_from_cube_sum_parity (n m : ℤ) (h : Even (n^3 + m^3)) : Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_sum_parity_from_cube_sum_parity_l3283_328306


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3283_328313

theorem sum_of_three_numbers : 2.12 + 0.004 + 0.345 = 2.469 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3283_328313


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l3283_328387

theorem complex_multiplication_sum (a b : ℝ) (i : ℂ) : 
  i ^ 2 = -1 → 
  a + b * i = (1 + i) * (2 - i) → 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l3283_328387


namespace NUMINAMATH_CALUDE_vertical_stripe_percentage_is_ten_percent_l3283_328310

/-- Represents the distribution of shirt types in a college cafeteria. -/
structure ShirtDistribution where
  total : Nat
  checkered : Nat
  polkaDotted : Nat
  plain : Nat
  horizontalMultiplier : Nat

/-- Calculates the percentage of people wearing vertical stripes. -/
def verticalStripePercentage (d : ShirtDistribution) : Rat :=
  let stripes := d.total - (d.checkered + d.polkaDotted + d.plain)
  let horizontal := d.checkered * d.horizontalMultiplier
  let vertical := stripes - horizontal
  (vertical : Rat) / d.total * 100

/-- Theorem stating that the percentage of people wearing vertical stripes is 10%. -/
theorem vertical_stripe_percentage_is_ten_percent : 
  let d : ShirtDistribution := {
    total := 100,
    checkered := 12,
    polkaDotted := 15,
    plain := 3,
    horizontalMultiplier := 5
  }
  verticalStripePercentage d = 10 := by sorry

end NUMINAMATH_CALUDE_vertical_stripe_percentage_is_ten_percent_l3283_328310


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l3283_328371

theorem triangle_abc_theorem (a b c : ℝ) (h : a^2 = b^2 + c^2 + b*c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A = 2*π/3 ∧ (a = 2*Real.sqrt 3 ∧ b = 2 → c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l3283_328371


namespace NUMINAMATH_CALUDE_students_not_taking_languages_l3283_328367

theorem students_not_taking_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (both : ℕ) 
  (h1 : total = 28) 
  (h2 : french = 5) 
  (h3 : spanish = 10) 
  (h4 : both = 4) : 
  total - (french + spanish + both) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_not_taking_languages_l3283_328367


namespace NUMINAMATH_CALUDE_abc_product_l3283_328325

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 29) (h5 : (1 : ℚ) / a + 1 / b + 1 / c + 399 / (a * b * c) = 1) :
  a * b * c = 992 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3283_328325


namespace NUMINAMATH_CALUDE_age_difference_l3283_328320

/-- The ages of three people satisfying certain ratios and sum --/
structure Ages where
  patrick : ℕ
  michael : ℕ
  monica : ℕ
  patrick_michael_ratio : patrick * 5 = michael * 3
  michael_monica_ratio : michael * 4 = monica * 3
  sum_of_ages : patrick + michael + monica = 88

/-- The difference between Monica's and Patrick's ages is 22 years --/
theorem age_difference (ages : Ages) : ages.monica - ages.patrick = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3283_328320


namespace NUMINAMATH_CALUDE_range_of_m_is_zero_to_one_l3283_328335

open Real

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := log x / log (1/2)

-- State the theorem
theorem range_of_m_is_zero_to_one :
  ∀ x m : ℝ, 
  0 < x → x < 1 → 
  log_half x = m / (1 - m) → 
  0 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_is_zero_to_one_l3283_328335


namespace NUMINAMATH_CALUDE_container_capacity_l3283_328397

theorem container_capacity (C : ℝ) 
  (h1 : 0.3 * C + 36 = 0.75 * C) : C = 80 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3283_328397


namespace NUMINAMATH_CALUDE_unique_digit_solution_l3283_328375

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def to_two_digit_number (a b : ℕ) : ℕ := 10 * a + b

def to_eight_digit_number (c d e f g h i j : ℕ) : ℕ :=
  10000000 * c + 1000000 * d + 100000 * e + 10000 * f + 1000 * g + 100 * h + 10 * i + j

theorem unique_digit_solution :
  ∃! (A B C D : ℕ),
    is_single_digit A ∧
    is_single_digit B ∧
    is_single_digit C ∧
    is_single_digit D ∧
    A ^ (to_two_digit_number A B) = to_eight_digit_number C C B B D D C A :=
by
  sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l3283_328375


namespace NUMINAMATH_CALUDE_sean_apples_count_l3283_328364

/-- Proves that the number of apples Sean has after receiving apples from Susan
    is equal to the total number of apples mentioned. -/
theorem sean_apples_count (initial_apples : ℕ) (apples_from_susan : ℕ) (total_apples : ℕ)
    (h1 : initial_apples = 9)
    (h2 : apples_from_susan = 8)
    (h3 : total_apples = 17) :
    initial_apples + apples_from_susan = total_apples := by
  sorry

end NUMINAMATH_CALUDE_sean_apples_count_l3283_328364


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l3283_328374

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 73 extra apples -/
theorem cafeteria_extra_apples :
  ∀ (red_apples green_apples students_wanting_fruit : ℕ),
    red_apples = 43 →
    green_apples = 32 →
    students_wanting_fruit = 2 →
    extra_apples red_apples green_apples students_wanting_fruit = 73 :=
by
  sorry


end NUMINAMATH_CALUDE_cafeteria_extra_apples_l3283_328374


namespace NUMINAMATH_CALUDE_range_of_m_main_theorem_l3283_328348

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + m*p.1 + 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (A m ∩ B).Nonempty → m ∈ Set.Iic (-1) := by
  sorry

-- Define the range of m
def range_m : Set ℝ := {m : ℝ | ∃ x : ℝ, (A m ∩ B).Nonempty}

-- State the main theorem
theorem main_theorem : range_m = Set.Iic (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_main_theorem_l3283_328348


namespace NUMINAMATH_CALUDE_platform_length_l3283_328369

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, the length of the platform is 350 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 39 →
  pole_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length / pole_time) * platform_time = train_length + platform_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3283_328369


namespace NUMINAMATH_CALUDE_second_item_is_14_l3283_328315

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  initial_selection : ℕ

/-- Calculates the second item in a systematic sample -/
def second_item (s : SystematicSampling) : ℕ :=
  s.initial_selection + (s.population_size / s.sample_size)

/-- Theorem: In the given systematic sampling scenario, the second item is 14 -/
theorem second_item_is_14 :
  let s : SystematicSampling := {
    population_size := 60,
    sample_size := 6,
    initial_selection := 4
  }
  second_item s = 14 := by
  sorry


end NUMINAMATH_CALUDE_second_item_is_14_l3283_328315


namespace NUMINAMATH_CALUDE_valid_k_values_l3283_328395

def A (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

def ends_with_k_zeros (m k : ℕ) : Prop :=
  ∃ r : ℕ, m = r * 10^k ∧ r % 10 ≠ 0

theorem valid_k_values :
  {k : ℕ | ∃ n : ℕ, ends_with_k_zeros (A n) k} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_valid_k_values_l3283_328395


namespace NUMINAMATH_CALUDE_distance_XY_is_80_l3283_328303

/-- The distance from X to Y in miles. -/
def distance_XY : ℝ := 80

/-- Yolanda's walking rate in miles per hour. -/
def yolanda_rate : ℝ := 8

/-- Bob's walking rate in miles per hour. -/
def bob_rate : ℝ := 9

/-- The distance Bob walked when they met, in miles. -/
def bob_distance : ℝ := 38.11764705882353

/-- The time difference between Yolanda and Bob's start times, in hours. -/
def time_difference : ℝ := 1

theorem distance_XY_is_80 :
  distance_XY = yolanda_rate * (time_difference + bob_distance / bob_rate) + bob_distance :=
sorry

end NUMINAMATH_CALUDE_distance_XY_is_80_l3283_328303


namespace NUMINAMATH_CALUDE_f_diff_at_pi_l3283_328307

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 3 * x^2 + 7 * Real.sin x

theorem f_diff_at_pi : f Real.pi - f (-Real.pi) = -2 * Real.pi^3 := by
  sorry

end NUMINAMATH_CALUDE_f_diff_at_pi_l3283_328307


namespace NUMINAMATH_CALUDE_problem_statement_l3283_328309

theorem problem_statement (x y : ℝ) (hx : x = 12) (hy : y = 18) :
  (x - y) * ((x + y)^2) = -5400 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3283_328309


namespace NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l3283_328366

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset Nat)
  (size : cards.card = 52)
  (suits : Finset Nat)
  (suit_size : suits.card = 4)

/-- The number of hearts in a standard deck. -/
def hearts_count : Nat := 13

/-- The probability of drawing two hearts from a well-shuffled standard deck. -/
def prob_two_hearts (d : Deck) : ℚ :=
  (hearts_count * (hearts_count - 1)) / (d.cards.card * (d.cards.card - 1))

/-- Theorem: The probability of drawing two hearts from a well-shuffled standard deck is 1/17. -/
theorem prob_two_hearts_is_one_seventeenth (d : Deck) :
  prob_two_hearts d = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l3283_328366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3283_328356

/-- An arithmetic sequence with given first two terms -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ a 2 = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  a 4 + a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3283_328356


namespace NUMINAMATH_CALUDE_fraction_difference_l3283_328372

theorem fraction_difference (a b : ℝ) : 
  a / (a + 1) - b / (b + 1) = (a - b) / ((a + 1) * (b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l3283_328372


namespace NUMINAMATH_CALUDE_max_volume_container_frame_l3283_328323

/-- Represents a rectangular container frame constructed from a steel bar -/
structure ContainerFrame where
  total_length : ℝ
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of the container frame -/
def volume (c : ContainerFrame) : ℝ :=
  c.length * c.width * c.height

/-- Checks if the container frame satisfies the given conditions -/
def is_valid_frame (c : ContainerFrame) : Prop :=
  c.total_length = 14.8 ∧
  c.length = c.width + 0.5 ∧
  2 * (c.length + c.width) + 4 * c.height = c.total_length

/-- Theorem stating the maximum volume and corresponding height -/
theorem max_volume_container_frame :
  ∃ (c : ContainerFrame),
    is_valid_frame c ∧
    c.height = 1.8 ∧
    volume c = 1.512 ∧
    ∀ (c' : ContainerFrame), is_valid_frame c' → volume c' ≤ volume c :=
sorry

end NUMINAMATH_CALUDE_max_volume_container_frame_l3283_328323


namespace NUMINAMATH_CALUDE_customer_buys_score_of_eggs_l3283_328344

/-- Definition of a score in terms of units -/
def score : ℕ := 20

/-- Definition of a dozen in terms of units -/
def dozen : ℕ := 12

/-- The number of eggs a customer receives when buying a score of eggs -/
def eggs_in_score : ℕ := score

theorem customer_buys_score_of_eggs : eggs_in_score = 20 := by sorry

end NUMINAMATH_CALUDE_customer_buys_score_of_eggs_l3283_328344


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3283_328388

theorem complex_magnitude_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + (1 - i)^2
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3283_328388


namespace NUMINAMATH_CALUDE_max_square_plots_l3283_328350

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available internal fencing -/
def availableFence : ℕ := 1500

/-- Calculates the number of square plots given the side length of a square -/
def numPlots (d : FieldDimensions) (side : ℕ) : ℕ :=
  (d.length / side) * (d.width / side)

/-- Calculates the amount of internal fencing needed for a given number of squares per side -/
def fencingNeeded (d : FieldDimensions) (squaresPerSide : ℕ) : ℕ :=
  (d.length * (squaresPerSide - 1)) + (d.width * (squaresPerSide - 1))

/-- Theorem stating that 576 is the maximum number of square plots -/
theorem max_square_plots (d : FieldDimensions) (h1 : d.length = 30) (h2 : d.width = 45) :
  ∀ n : ℕ, numPlots d n ≤ 576 ∧ fencingNeeded d (d.width / (d.width / 24)) ≤ availableFence :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l3283_328350


namespace NUMINAMATH_CALUDE_original_number_proof_l3283_328332

theorem original_number_proof : 
  ∃ x : ℝ, (x * 1.4 = 700) ∧ (x = 500) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3283_328332


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3283_328357

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n, a (n + 1) - a n = 2)
  (h3 : a 3 = 4) :
  a 12 = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3283_328357


namespace NUMINAMATH_CALUDE_middle_number_is_four_l3283_328365

theorem middle_number_is_four (a b c : ℕ) : 
  a < b ∧ b < c  -- numbers are in increasing order
  → a + b + c = 15  -- numbers sum to 15
  → a ≠ b ∧ b ≠ c ∧ a ≠ c  -- numbers are all different
  → a > 0 ∧ b > 0 ∧ c > 0  -- numbers are positive
  → (∀ x y, x < y ∧ x + y < 15 → ∃ z, x < z ∧ z < y ∧ x + z + y = 15)  -- leftmost doesn't uniquely determine
  → (∀ x y, x < y ∧ x + y > 0 → ∃ z, z < x ∧ x < y ∧ z + x + y = 15)  -- rightmost doesn't uniquely determine
  → b = 4  -- middle number is 4
  := by sorry

end NUMINAMATH_CALUDE_middle_number_is_four_l3283_328365


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3283_328391

theorem sum_remainder_mod_seven :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3283_328391
