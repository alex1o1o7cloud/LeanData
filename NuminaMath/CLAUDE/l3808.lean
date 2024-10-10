import Mathlib

namespace height_pillar_E_l3808_380886

/-- Regular octagon with pillars -/
structure OctagonWithPillars where
  /-- Side length of the octagon -/
  side_length : ℝ
  /-- Height of pillar at vertex A -/
  height_A : ℝ
  /-- Height of pillar at vertex B -/
  height_B : ℝ
  /-- Height of pillar at vertex C -/
  height_C : ℝ

/-- Theorem: Height of pillar at E in a regular octagon with given pillar heights -/
theorem height_pillar_E (octagon : OctagonWithPillars) 
  (h_A : octagon.height_A = 15)
  (h_B : octagon.height_B = 12)
  (h_C : octagon.height_C = 13) :
  ∃ (height_E : ℝ), height_E = 5 := by
  sorry

end height_pillar_E_l3808_380886


namespace accounting_majors_l3808_380819

theorem accounting_majors (p q r s t u : ℕ) : 
  p * q * r * s * t * u = 51030 →
  1 < p → p < q → q < r → r < s → s < t → t < u →
  p = 2 := by sorry

end accounting_majors_l3808_380819


namespace train_passenger_count_l3808_380812

def train_problem (initial_passengers : ℕ) (first_station_pickup : ℕ) (final_passengers : ℕ) : ℕ :=
  let after_first_drop := initial_passengers - (initial_passengers / 3)
  let after_first_pickup := after_first_drop + first_station_pickup
  let after_second_drop := after_first_pickup / 2
  final_passengers - after_second_drop

theorem train_passenger_count :
  train_problem 288 280 248 = 12 := by
  sorry

end train_passenger_count_l3808_380812


namespace manuscript_typing_cost_l3808_380806

/-- The cost per page for the first typing of a manuscript -/
def first_typing_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (revision_cost : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - revision_cost * (pages_revised_once + 2 * pages_revised_twice)) / total_pages

theorem manuscript_typing_cost :
  first_typing_cost 200 80 20 3 1360 = 5 := by
  sorry

end manuscript_typing_cost_l3808_380806


namespace factors_of_34848_l3808_380866

/-- The number of positive factors of 34848 -/
def num_factors : ℕ := 54

/-- 34848 as a natural number -/
def n : ℕ := 34848

theorem factors_of_34848 : Nat.card (Nat.divisors n) = num_factors := by
  sorry

end factors_of_34848_l3808_380866


namespace square_side_length_difference_l3808_380840

/-- Given four squares with known side length differences, prove that the total difference
    in side length from the largest to the smallest square is the sum of these differences. -/
theorem square_side_length_difference (AB CD FE : ℝ) (hAB : AB = 11) (hCD : CD = 5) (hFE : FE = 13) :
  ∃ (GH : ℝ), GH = AB + CD + FE :=
by sorry

end square_side_length_difference_l3808_380840


namespace food_for_horses_l3808_380846

/-- Calculates the total amount of food needed for horses over a number of days. -/
def total_food_needed (num_horses : ℕ) (num_days : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) (grain_per_day : ℕ) : ℕ :=
  let total_oats := num_horses * num_days * oats_per_meal * oats_meals_per_day
  let total_grain := num_horses * num_days * grain_per_day
  total_oats + total_grain

/-- Theorem stating that the total food needed for 4 horses over 3 days is 132 pounds. -/
theorem food_for_horses :
  total_food_needed 4 3 4 2 3 = 132 := by
  sorry

end food_for_horses_l3808_380846


namespace initial_books_count_l3808_380820

theorem initial_books_count (initial_books sold_books given_books remaining_books : ℕ) :
  sold_books = 11 →
  given_books = 35 →
  remaining_books = 62 →
  initial_books = sold_books + given_books + remaining_books →
  initial_books = 108 := by
  sorry

end initial_books_count_l3808_380820


namespace acai_juice_cost_per_litre_l3808_380875

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of the mixed fruit juice -/
def mixed_juice_cost : ℝ := 262.85

/-- The volume of mixed fruit juice in litres -/
def mixed_juice_volume : ℝ := 32

/-- The volume of açaí berry juice in litres -/
def acai_juice_volume : ℝ := 21.333333333333332

/-- The total volume of the cocktail in litres -/
def total_volume : ℝ := mixed_juice_volume + acai_juice_volume

theorem acai_juice_cost_per_litre : 
  ∃ (acai_cost : ℝ),
    acai_cost = 3105.00 ∧
    mixed_juice_cost * mixed_juice_volume + acai_cost * acai_juice_volume = 
    cocktail_cost * total_volume :=
by sorry

end acai_juice_cost_per_litre_l3808_380875


namespace triangle_abc_properties_l3808_380834

theorem triangle_abc_properties (A B C : Real) (h : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  A + B = 3 * C →
  2 * Real.sin (A - C) = Real.sin B →
  -- AB = 5 (implicitly used in the height calculation)
  -- Prove sin A and height h
  Real.sin A = 3 * (10 : Real).sqrt / 10 ∧ h = 6 := by
  sorry

end triangle_abc_properties_l3808_380834


namespace quadratic_two_distinct_roots_l3808_380835

/-- The quadratic equation (m-1)x^2 - 4x + 1 = 0 has two distinct real roots if and only if m < 5 and m ≠ 1 -/
theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 4 * x + 1 = 0 ∧ (m - 1) * y^2 - 4 * y + 1 = 0) ↔ 
  (m < 5 ∧ m ≠ 1) :=
sorry

end quadratic_two_distinct_roots_l3808_380835


namespace product_digit_sum_l3808_380821

theorem product_digit_sum : 
  let a := 2^20
  let b := 5^17
  let product := a * b
  (List.sum (product.digits 10)) = 8 := by sorry

end product_digit_sum_l3808_380821


namespace equation_solution_l3808_380884

theorem equation_solution (x y : ℝ) :
  (x^4 + 1) * (y^4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by sorry

end equation_solution_l3808_380884


namespace digit_sequence_bound_l3808_380830

/-- Given a positive integer N with n digits, if all its digits are distinct
    and the sum of any three consecutive digits is divisible by 5, then n ≤ 6. -/
theorem digit_sequence_bound (N : ℕ) (n : ℕ) : 
  (N ≥ 10^(n-1) ∧ N < 10^n) →  -- N is an n-digit number
  (∀ i j, i ≠ j → (N / 10^i) % 10 ≠ (N / 10^j) % 10) →  -- All digits are distinct
  (∀ i, i + 2 < n → ((N / 10^i) % 10 + (N / 10^(i+1)) % 10 + (N / 10^(i+2)) % 10) % 5 = 0) →  -- Sum of any three consecutive digits is divisible by 5
  n ≤ 6 :=
by sorry

end digit_sequence_bound_l3808_380830


namespace largest_n_divisibility_l3808_380817

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 1221 → ¬(∃ k : ℕ, n^3 + 99 = k * (n + 11)) ∧ 
  ∃ k : ℕ, 1221^3 + 99 = k * (1221 + 11) :=
sorry

end largest_n_divisibility_l3808_380817


namespace toy_spending_ratio_l3808_380816

theorem toy_spending_ratio (initial_amount : ℕ) (toy_cost : ℕ) (final_amount : ℕ) :
  initial_amount = 204 →
  final_amount = 51 →
  toy_cost + (initial_amount - toy_cost) / 2 + final_amount = initial_amount →
  toy_cost * 2 = initial_amount :=
by sorry

end toy_spending_ratio_l3808_380816


namespace factorization_2x_cubed_minus_8x_l3808_380802

theorem factorization_2x_cubed_minus_8x (x : ℝ) : 
  2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) := by
sorry


end factorization_2x_cubed_minus_8x_l3808_380802


namespace arithmetic_squares_sequence_l3808_380852

theorem arithmetic_squares_sequence (k : ℤ) : 
  (∃! k : ℤ, 
    (∃ a : ℤ, 
      (49 + k = a^2) ∧ 
      (361 + k = (a + 2)^2) ∧ 
      (784 + k = (a + 4)^2))) := by
  sorry

end arithmetic_squares_sequence_l3808_380852


namespace fraction_equality_l3808_380871

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) := by
sorry

end fraction_equality_l3808_380871


namespace find_a_over_b_l3808_380828

-- Define the region
def region (x y : ℝ) : Prop :=
  x ≥ 1 ∧ x + y ≤ 4 ∧ ∃ a b : ℝ, a * x + b * y + 2 ≥ 0

-- Define the objective function
def z (x y : ℝ) : ℝ := 2 * x + y

-- State the theorem
theorem find_a_over_b :
  ∃ a b : ℝ,
    (∀ x y : ℝ, region x y → z x y ≤ 7) ∧
    (∃ x y : ℝ, region x y ∧ z x y = 7) ∧
    (∀ x y : ℝ, region x y → z x y ≥ 1) ∧
    (∃ x y : ℝ, region x y ∧ z x y = 1) ∧
    a / b = -1 :=
sorry

end find_a_over_b_l3808_380828


namespace discount_difference_is_399_l3808_380831

def initial_price : ℝ := 8000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.05]
def option2_discounts : List ℝ := [0.35, 0.10, 0.05]

def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem discount_difference_is_399 :
  apply_discounts initial_price option1_discounts -
  apply_discounts initial_price option2_discounts = 399 := by
  sorry

end discount_difference_is_399_l3808_380831


namespace equality_in_different_bases_l3808_380885

theorem equality_in_different_bases : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 
  (3 * a^2 + 4 * a + 2 : ℕ) = (9 * b + 7 : ℕ) := by
  sorry

end equality_in_different_bases_l3808_380885


namespace sum_of_integers_and_squares_l3808_380888

-- Define the sum of integers from a to b, inclusive
def sumIntegers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

-- Define the sum of squares from a to b, inclusive
def sumSquares (a b : Int) : Int :=
  (b * (b + 1) * (2 * b + 1) - (a - 1) * a * (2 * a - 1)) / 6

theorem sum_of_integers_and_squares : 
  sumIntegers (-50) 40 + sumSquares 10 40 = 21220 :=
by
  sorry

end sum_of_integers_and_squares_l3808_380888


namespace simplify_and_evaluate_l3808_380855

theorem simplify_and_evaluate (a : ℝ) (h : a = -Real.sqrt 2) :
  (a - 3) / a * 6 / (a^2 - 6*a + 9) - (2*a + 6) / (a^2 - 9) = Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l3808_380855


namespace hamans_dropped_trays_l3808_380824

theorem hamans_dropped_trays 
  (initial_trays : ℕ) 
  (additional_trays : ℕ) 
  (total_eggs_sold : ℕ) 
  (eggs_per_tray : ℕ) 
  (h1 : initial_trays = 10)
  (h2 : additional_trays = 7)
  (h3 : total_eggs_sold = 540)
  (h4 : eggs_per_tray = 30) :
  initial_trays + additional_trays + 1 - (total_eggs_sold / eggs_per_tray) = 8 :=
by sorry

end hamans_dropped_trays_l3808_380824


namespace N_is_perfect_square_l3808_380873

/-- Constructs the number N with n ones and n+1 twos, ending with 5 -/
def constructN (n : ℕ) : ℕ :=
  (10^(2*n+2) + 10^(n+2) + 25) / 9

/-- Theorem stating that N is a perfect square for any natural number n -/
theorem N_is_perfect_square (n : ℕ) : ∃ m : ℕ, (constructN n) = m^2 := by
  sorry

end N_is_perfect_square_l3808_380873


namespace equation_solution_l3808_380864

theorem equation_solution (x : ℚ) : 
  x ≠ 2/3 →
  ((7*x + 3) / (3*x^2 + 7*x - 6) = 3*x / (3*x - 2)) ↔ (x = 1/3 ∨ x = -3) := by
sorry

end equation_solution_l3808_380864


namespace white_balls_count_l3808_380854

/-- Calculates the number of white balls in a bag given specific conditions -/
theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) : 
  total = 60 ∧ 
  green = 18 ∧ 
  yellow = 5 ∧ 
  red = 6 ∧ 
  purple = 9 ∧ 
  prob_not_red_purple = 3/4 → 
  total - (green + yellow + red + purple) = 22 := by
sorry

end white_balls_count_l3808_380854


namespace quadratic_solution_product_l3808_380837

theorem quadratic_solution_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (2*a - 5) * (3*b - 4) = 47 := by
sorry

end quadratic_solution_product_l3808_380837


namespace x_sum_less_than_2m_l3808_380858

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 5 - a / Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * f a x

theorem x_sum_less_than_2m (a : ℝ) (m x₁ x₂ : ℝ) 
  (h1 : m ≥ 1) 
  (h2 : x₁ < m) 
  (h3 : x₂ > m) 
  (h4 : g a x₁ + g a x₂ = 2 * g a m) : 
  x₁ + x₂ < 2 * m := by
  sorry

end x_sum_less_than_2m_l3808_380858


namespace recommended_intake_proof_l3808_380844

/-- Recommended intake of added sugar for men per day (in calories) -/
def recommended_intake : ℝ := 150

/-- Calories in the soft drink -/
def soft_drink_calories : ℝ := 2500

/-- Percentage of calories from added sugar in the soft drink -/
def soft_drink_sugar_percentage : ℝ := 0.05

/-- Calories of added sugar in each candy bar -/
def candy_bar_sugar_calories : ℝ := 25

/-- Number of candy bars consumed -/
def candy_bars_consumed : ℕ := 7

/-- Percentage by which Mark exceeded the recommended intake -/
def excess_percentage : ℝ := 1

theorem recommended_intake_proof :
  let soft_drink_sugar := soft_drink_calories * soft_drink_sugar_percentage
  let candy_sugar := candy_bar_sugar_calories * candy_bars_consumed
  let total_sugar := soft_drink_sugar + candy_sugar
  total_sugar = recommended_intake * (1 + excess_percentage) :=
by sorry

end recommended_intake_proof_l3808_380844


namespace tenth_term_of_arithmetic_sequence_l3808_380848

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence 
  (a₁ : ℚ) (a₂₀ : ℚ) (h₁ : a₁ = 5/11) (h₂₀ : a₂₀ = 9/11)
  (h_seq : ∀ n, arithmetic_sequence a₁ ((a₂₀ - a₁) / 19) n = 
    arithmetic_sequence (5/11) ((9/11 - 5/11) / 19) n) :
  arithmetic_sequence a₁ ((a₂₀ - a₁) / 19) 10 = 1233/2309 :=
by sorry

end tenth_term_of_arithmetic_sequence_l3808_380848


namespace addition_and_multiplication_of_integers_l3808_380878

theorem addition_and_multiplication_of_integers : 
  (-3 + 2 = -1) ∧ ((-3) * 2 = -6) := by sorry

end addition_and_multiplication_of_integers_l3808_380878


namespace unique_divisibility_l3808_380865

def is_divisible_by_only_one_small_prime (n : ℕ) : Prop :=
  ∃! p, p < 10 ∧ Nat.Prime p ∧ n % p = 0

def number_form (B : ℕ) : ℕ := 404300 + B

theorem unique_divisibility :
  ∃! B, B < 10 ∧ is_divisible_by_only_one_small_prime (number_form B) ∧ number_form B = 404304 := by
  sorry

end unique_divisibility_l3808_380865


namespace lilac_paint_mixture_l3808_380805

/-- Given a paint mixture where 70% is blue, 20% is red, and the rest is white,
    if 140 ounces of blue paint is added, then 20 ounces of white paint is added. -/
theorem lilac_paint_mixture (blue_percent : ℝ) (red_percent : ℝ) (blue_amount : ℝ) : 
  blue_percent = 0.7 →
  red_percent = 0.2 →
  blue_amount = 140 →
  let total_amount := blue_amount / blue_percent
  let white_percent := 1 - blue_percent - red_percent
  let white_amount := total_amount * white_percent
  white_amount = 20 := by
  sorry

end lilac_paint_mixture_l3808_380805


namespace z_purely_imaginary_iff_z_in_second_quadrant_iff_l3808_380822

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

theorem z_purely_imaginary_iff (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) ↔ m = -1/2 := by sorry

theorem z_in_second_quadrant_iff (m : ℝ) :
  (Complex.re (z m) < 0 ∧ Complex.im (z m) > 0) ↔ -1/2 < m ∧ m < 1 := by sorry

end z_purely_imaginary_iff_z_in_second_quadrant_iff_l3808_380822


namespace smallest_p_in_prime_sum_l3808_380832

theorem smallest_p_in_prime_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p + q = r →
  1 < p →
  p < q →
  r > 10 →
  ∃ (p' : ℕ), p' = 2 ∧ (∀ (p'' : ℕ), 
    Nat.Prime p'' → 
    p'' + q = r → 
    1 < p'' → 
    p'' < q → 
    r > 10 → 
    p' ≤ p'') :=
by sorry

end smallest_p_in_prime_sum_l3808_380832


namespace line_segment_param_sum_l3808_380870

/-- Given a line segment connecting points (-3,10) and (4,16) represented by
    parametric equations x = at + b and y = ct + d where 0 ≤ t ≤ 1,
    and t = 0 corresponds to (-3,10), prove that a² + b² + c² + d² = 194 -/
theorem line_segment_param_sum (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = -3 ∧ d = 10) →
  (a + b = 4 ∧ c + d = 16) →
  a^2 + b^2 + c^2 + d^2 = 194 := by
sorry

end line_segment_param_sum_l3808_380870


namespace least_three_digit_with_digit_product_12_l3808_380859

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
sorry

end least_three_digit_with_digit_product_12_l3808_380859


namespace artist_cube_structure_surface_area_l3808_380856

/-- Represents the cube structure described in the problem -/
structure CubeStructure where
  totalCubes : ℕ
  cubeEdgeLength : ℝ
  bottomLayerSize : ℕ
  topLayerSize : ℕ

/-- Calculates the exposed surface area of the cube structure -/
def exposedSurfaceArea (cs : CubeStructure) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem artist_cube_structure_surface_area :
  ∃ (cs : CubeStructure),
    cs.totalCubes = 16 ∧
    cs.cubeEdgeLength = 1 ∧
    cs.bottomLayerSize = 3 ∧
    cs.topLayerSize = 2 ∧
    exposedSurfaceArea cs = 49 :=
  sorry

end artist_cube_structure_surface_area_l3808_380856


namespace pairing_count_l3808_380882

/-- The number of bowls -/
def num_bowls : ℕ := 6

/-- The number of glasses -/
def num_glasses : ℕ := 4

/-- The number of fixed pairings -/
def num_fixed_pairings : ℕ := 1

/-- The number of remaining bowls after fixed pairing -/
def num_remaining_bowls : ℕ := num_bowls - num_fixed_pairings

/-- The number of remaining glasses after fixed pairing -/
def num_remaining_glasses : ℕ := num_glasses - num_fixed_pairings

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_remaining_bowls * num_remaining_glasses + num_fixed_pairings

theorem pairing_count : total_pairings = 16 := by
  sorry

end pairing_count_l3808_380882


namespace triple_solution_l3808_380833

theorem triple_solution (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 1 ∧ a * (2*b - 2*a - c) ≥ 1/2 →
  ((a = 1/Real.sqrt 6 ∧ b = 2/Real.sqrt 6 ∧ c = -1/Real.sqrt 6) ∨
   (a = -1/Real.sqrt 6 ∧ b = -2/Real.sqrt 6 ∧ c = 1/Real.sqrt 6)) :=
by sorry

end triple_solution_l3808_380833


namespace stone_151_is_9_l3808_380800

/-- Represents the number of stones in the arrangement. -/
def num_stones : ℕ := 12

/-- Represents the modulus for the counting pattern. -/
def counting_modulus : ℕ := 22

/-- The number we want to find the original stone for. -/
def target_count : ℕ := 151

/-- Function to determine the original stone number given a count. -/
def original_stone (count : ℕ) : ℕ :=
  (count - 1) % counting_modulus + 1

theorem stone_151_is_9 : original_stone target_count = 9 := by
  sorry

end stone_151_is_9_l3808_380800


namespace smallest_prime_divisor_of_sum_l3808_380825

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (3^15 + 11^13)) →
  2 ≤ Nat.minFac (3^15 + 11^13) ∧ Nat.minFac (3^15 + 11^13) = 2 :=
by sorry

end smallest_prime_divisor_of_sum_l3808_380825


namespace chord_length_difference_l3808_380838

theorem chord_length_difference (r₁ r₂ : ℝ) (hr₁ : r₁ = 26) (hr₂ : r₂ = 5) :
  let longest_chord := 2 * r₁
  let shortest_chord := 2 * Real.sqrt (r₁^2 - (r₁ - r₂)^2)
  longest_chord - shortest_chord = 52 - 2 * Real.sqrt 235 :=
by sorry

end chord_length_difference_l3808_380838


namespace canada_population_1998_l3808_380883

/-- The population of Canada in 1998 in millions -/
def canada_population_millions : ℝ := 30.3

/-- One million in standard form -/
def million : ℕ := 1000000

/-- Theorem: The population of Canada in 1998 was 30,300,000 -/
theorem canada_population_1998 : 
  (canada_population_millions * million : ℝ) = 30300000 := by sorry

end canada_population_1998_l3808_380883


namespace cubic_three_zeros_l3808_380894

/-- A function f(x) = x^3 - 3x + a has three distinct zeros if and only if -2 < a < 2 -/
theorem cubic_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x + a = 0 ∧ y^3 - 3*y + a = 0 ∧ z^3 - 3*z + a = 0) ↔
  -2 < a ∧ a < 2 :=
by sorry

end cubic_three_zeros_l3808_380894


namespace apartment_ages_puzzle_l3808_380861

def is_valid_triplet (a b c : ℕ) : Prop :=
  a * b * c = 1296 ∧ a < 100 ∧ b < 100 ∧ c < 100

def has_duplicate_sum (triplets : List (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (t1 t2 : ℕ × ℕ × ℕ), t1 ∈ triplets ∧ t2 ∈ triplets ∧ t1 ≠ t2 ∧ 
    t1.1 + t1.2.1 + t1.2.2 = t2.1 + t2.2.1 + t2.2.2

theorem apartment_ages_puzzle :
  ∃! (a b c : ℕ), 
    is_valid_triplet a b c ∧
    (∀ triplets : List (ℕ × ℕ × ℕ), (∀ (x y z : ℕ), (x, y, z) ∈ triplets → is_valid_triplet x y z) →
      has_duplicate_sum triplets → (a, b, c) ∈ triplets) ∧
    a < b ∧ b < c ∧ c < 100 ∧
    a + b + c = 91 :=
by sorry

end apartment_ages_puzzle_l3808_380861


namespace reciprocal_inequality_l3808_380845

theorem reciprocal_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 1 / a < 1 / b := by
  sorry

end reciprocal_inequality_l3808_380845


namespace find_constant_b_l3808_380847

theorem find_constant_b (a b c : ℚ) : 
  (∀ x : ℚ, (4 * x^3 - 3 * x + 7/2) * (a * x^2 + b * x + c) = 
    12 * x^5 - 14 * x^4 + 18 * x^3 - (23/3) * x^2 + (14/2) * x - 3) →
  b = -7/2 := by
sorry

end find_constant_b_l3808_380847


namespace extra_coverage_area_l3808_380869

/-- Represents the area covered by one bag of grass seed in square feet. -/
def bag_coverage : ℕ := 250

/-- Represents the length of the lawn from house to curb in feet. -/
def lawn_length : ℕ := 22

/-- Represents the width of the lawn from side to side in feet. -/
def lawn_width : ℕ := 36

/-- Represents the number of bags of grass seed bought. -/
def bags_bought : ℕ := 4

/-- Calculates the extra area that can be covered by leftover grass seed after reseeding the lawn. -/
theorem extra_coverage_area : 
  bags_bought * bag_coverage - lawn_length * lawn_width = 208 := by
  sorry

end extra_coverage_area_l3808_380869


namespace calculate_expression_l3808_380850

theorem calculate_expression : 
  Real.sqrt 5 * 5^(1/3) + 15 / 5 * 3 - 9^(5/2) = 5^(5/6) - 234 := by
  sorry

end calculate_expression_l3808_380850


namespace circle_area_l3808_380807

theorem circle_area (r : ℝ) (h : 2 * π * r = 18 * π) : π * r^2 = 81 * π := by
  sorry

end circle_area_l3808_380807


namespace package_weight_l3808_380895

theorem package_weight (total_weight : ℝ) (first_butcher_packages : ℕ) (second_butcher_packages : ℕ) (third_butcher_packages : ℕ) 
  (h1 : total_weight = 100)
  (h2 : first_butcher_packages = 10)
  (h3 : second_butcher_packages = 7)
  (h4 : third_butcher_packages = 8) :
  ∃ (package_weight : ℝ), 
    package_weight * (first_butcher_packages + second_butcher_packages + third_butcher_packages) = total_weight ∧ 
    package_weight = 4 := by
  sorry

end package_weight_l3808_380895


namespace exists_minimum_top_number_l3808_380862

/-- Represents a square pyramid of blocks -/
structure SquarePyramid where
  base : Matrix (Fin 4) (Fin 4) ℕ
  layer2 : Matrix (Fin 3) (Fin 3) ℕ
  layer3 : Matrix (Fin 2) (Fin 2) ℕ
  top : ℕ

/-- Checks if the pyramid is valid according to the given conditions -/
def isValidPyramid (p : SquarePyramid) : Prop :=
  (∀ i j, p.base i j ∈ Finset.range 17) ∧
  (∀ i j, p.layer2 i j = p.base (i+1) (j+1) + p.base (i+1) j + p.base i (j+1)) ∧
  (∀ i j, p.layer3 i j = p.layer2 (i+1) (j+1) + p.layer2 (i+1) j + p.layer2 i (j+1)) ∧
  (p.top = p.layer3 1 1 + p.layer3 1 0 + p.layer3 0 1)

/-- The main theorem statement -/
theorem exists_minimum_top_number :
  ∃ (min : ℕ), ∀ (p : SquarePyramid), isValidPyramid p → p.top ≥ min :=
sorry


end exists_minimum_top_number_l3808_380862


namespace max_area_CDFE_l3808_380843

/-- A square with side length 1 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 1 ∧ B.2 = 0 ∧ C.1 = 1 ∧ C.2 = 1 ∧ D.1 = 0 ∧ D.2 = 1)

/-- Points E and F on sides AB and AD respectively -/
def PointsEF (s : Square) (x : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((x, 0), (0, x))

/-- Area of quadrilateral CDFE -/
def AreaCDFE (s : Square) (x : ℝ) : ℝ :=
  x * (1 - x)

/-- The maximum area of quadrilateral CDFE is 1/4 -/
theorem max_area_CDFE (s : Square) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → AreaCDFE s y ≤ AreaCDFE s x ∧
  AreaCDFE s x = 1/4 :=
sorry

end max_area_CDFE_l3808_380843


namespace average_difference_with_input_error_l3808_380810

theorem average_difference_with_input_error (n : ℕ) (correct_value wrong_value : ℝ) : 
  n = 30 → correct_value = 75 → wrong_value = 15 → 
  (correct_value - wrong_value) / n = -2 := by
sorry

end average_difference_with_input_error_l3808_380810


namespace complex_number_problem_l3808_380804

theorem complex_number_problem (z : ℂ) 
  (h1 : z.re > 0) 
  (h2 : Complex.abs z = 2 * Real.sqrt 5) 
  (h3 : (Complex.I + 2) * z = Complex.I * (Complex.I * z).im) 
  (h4 : ∃ (m n : ℝ), z^2 + m*z + n = 0) : 
  z = 4 + 2*Complex.I ∧ 
  ∃ (m n : ℝ), z^2 + m*z + n = 0 ∧ m = -8 ∧ n = 20 := by
  sorry

end complex_number_problem_l3808_380804


namespace quadratic_inequality_range_l3808_380823

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + a - 3 < 0) ↔ a < 4 :=
sorry

end quadratic_inequality_range_l3808_380823


namespace probability_three_out_of_ten_l3808_380839

/-- The probability of selecting at least one defective item from a set of products -/
def probability_at_least_one_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ)

/-- Theorem stating the probability of selecting at least one defective item
    when 3 out of 10 items are defective and 3 items are randomly selected -/
theorem probability_three_out_of_ten :
  probability_at_least_one_defective 10 3 3 = 17 / 24 := by
  sorry

end probability_three_out_of_ten_l3808_380839


namespace sum_x_y_equals_two_l3808_380851

-- Define the function f(t) = t^3 + 2003t
def f (t : ℝ) : ℝ := t^3 + 2003*t

-- State the theorem
theorem sum_x_y_equals_two (x y : ℝ) 
  (hx : f (x - 1) = -1) 
  (hy : f (y - 1) = 1) : 
  x + y = 2 := by
  sorry


end sum_x_y_equals_two_l3808_380851


namespace lg_expression_equals_two_l3808_380853

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_two :
  (lg 5)^2 + lg 2 * lg 50 = 2 := by
  sorry

end lg_expression_equals_two_l3808_380853


namespace largest_among_rationals_l3808_380836

theorem largest_among_rationals : 
  let numbers : List ℚ := [-2/3, -2, -1, -5]
  (∀ x ∈ numbers, x ≤ -2/3) ∧ (-2/3 ∈ numbers) := by
  sorry

end largest_among_rationals_l3808_380836


namespace max_min_product_l3808_380880

theorem max_min_product (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (sum_eq : p + q + r = 13) (sum_prod_eq : p * q + q * r + r * p = 30) :
  ∃ (n : ℝ), n = min (p * q) (min (q * r) (r * p)) ∧ n ≤ 10 ∧
  ∀ (m : ℝ), m = min (p * q) (min (q * r) (r * p)) → m ≤ n :=
sorry

end max_min_product_l3808_380880


namespace tree_distance_l3808_380863

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let distance_between (i j : ℕ) := d * (j - i : ℝ) / 4
  distance_between 1 n = 175 := by
  sorry

end tree_distance_l3808_380863


namespace regular_polygon_sides_l3808_380860

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end regular_polygon_sides_l3808_380860


namespace fraction_between_l3808_380893

theorem fraction_between (p q : ℕ+) (h1 : (6 : ℚ) / 11 < p / q) (h2 : p / q < (5 : ℚ) / 9) 
  (h3 : ∀ (r s : ℕ+), (6 : ℚ) / 11 < r / s → r / s < (5 : ℚ) / 9 → s ≥ q) : 
  p + q = 31 := by
  sorry

end fraction_between_l3808_380893


namespace soccer_substitutions_remainder_l3808_380881

/-- The number of ways to make substitutions in a soccer game -/
def substitution_ways (total_players start_players max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem about the remainder of substitution ways when divided by 1000 -/
theorem soccer_substitutions_remainder :
  let total_players := 22
  let start_players := 11
  let max_substitutions := 4
  (substitution_ways total_players start_players max_substitutions) % 1000 = 122 := by
  sorry

end soccer_substitutions_remainder_l3808_380881


namespace count_sevens_1_to_100_l3808_380898

/-- Count of digit 7 in numbers from 1 to 100 -/
def countSevens : ℕ → ℕ
| 0 => 0
| (n + 1) => (if n + 1 < 101 then (if (n + 1) % 10 = 7 || (n + 1) / 10 = 7 then 1 else 0) else 0) + countSevens n

theorem count_sevens_1_to_100 : countSevens 100 = 20 := by
  sorry

end count_sevens_1_to_100_l3808_380898


namespace line_y_coordinate_at_x_10_l3808_380811

/-- Given a line passing through points (4, 0) and (-2, -3),
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 :
  let m : ℚ := (0 - (-3)) / (4 - (-2))  -- Slope of the line
  let b : ℚ := 0 - m * 4                -- y-intercept of the line
  m * 10 + b = 3 := by sorry

end line_y_coordinate_at_x_10_l3808_380811


namespace three_digit_multiples_of_6_and_8_l3808_380876

theorem three_digit_multiples_of_6_and_8 : 
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 8 = 0) (Finset.range 900 ∪ {999})).card = 37 := by
  sorry

end three_digit_multiples_of_6_and_8_l3808_380876


namespace quadratic_roots_difference_l3808_380857

theorem quadratic_roots_difference (a b : ℝ) : 
  (2 : ℝ) ∈ {x : ℝ | x^2 - (a+1)*x + a = 0} ∧ 
  b ∈ {x : ℝ | x^2 - (a+1)*x + a = 0} → 
  a - b = 1 := by
sorry

end quadratic_roots_difference_l3808_380857


namespace keith_total_cost_l3808_380829

def rabbit_toy_cost : ℚ := 6.51
def pet_food_cost : ℚ := 5.79
def cage_cost : ℚ := 12.51
def found_money : ℚ := 1.00

theorem keith_total_cost :
  rabbit_toy_cost + pet_food_cost + cage_cost - found_money = 23.81 := by
  sorry

end keith_total_cost_l3808_380829


namespace intersection_and_subset_l3808_380897

def set_A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def set_B (a : ℝ) : Set ℝ := {x | 1 - a < x ∧ x ≤ 3*a + 1}

theorem intersection_and_subset :
  (∀ x : ℝ, x ∈ (set_A ∩ set_B 1) ↔ (0 < x ∧ x ≤ 3)) ∧
  (∀ a : ℝ, set_B a ⊆ set_A ↔ a ≤ 2/3) := by sorry

end intersection_and_subset_l3808_380897


namespace isosceles_triangle_base_length_l3808_380842

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 20 cm has a base of 6 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 20 →
  base = 6 := by
sorry

end isosceles_triangle_base_length_l3808_380842


namespace profit_percentage_specific_l3808_380849

/-- The profit percentage after markup and discount -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem: Given a 60% markup and a 25% discount, the profit percentage is 20% -/
theorem profit_percentage_specific : profit_percentage 0.6 0.25 = 20 := by
  sorry

end profit_percentage_specific_l3808_380849


namespace bakers_sales_l3808_380899

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (h1 : cakes_sold = 158) 
  (h2 : cakes_sold = pastries_sold + 11) : 
  pastries_sold = 147 := by
  sorry

end bakers_sales_l3808_380899


namespace range_of_f_on_I_l3808_380827

-- Define the function
def f (x : ℝ) : ℝ := x^2 + x

-- Define the interval
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f_on_I :
  {y | ∃ x ∈ I, f x = y} = {y | -1/4 ≤ y ∧ y ≤ 12} := by
  sorry

end range_of_f_on_I_l3808_380827


namespace chebyshev_roots_l3808_380813

def T : ℕ → (Real → Real)
  | 0 => λ x => 1
  | 1 => λ x => x
  | (n + 2) => λ x => 2 * x * T (n + 1) x + T n x

theorem chebyshev_roots (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  T n (Real.cos ((2 * k - 1 : ℝ) * Real.pi / (2 * n : ℝ))) = 0 := by
  sorry

end chebyshev_roots_l3808_380813


namespace modulus_sum_complex_numbers_l3808_380889

theorem modulus_sum_complex_numbers : 
  Complex.abs ((3 : ℂ) - 8*I + (4 : ℂ) + 6*I) = Real.sqrt 53 := by sorry

end modulus_sum_complex_numbers_l3808_380889


namespace spring_length_at_9kg_spring_length_conditions_l3808_380815

/-- A linear function representing the relationship between mass and spring length. -/
def spring_length (x : ℝ) : ℝ := 0.5 * x + 10

/-- Theorem stating that the spring length is 14.5 cm when the mass is 9 kg. -/
theorem spring_length_at_9kg :
  spring_length 0 = 10 →
  spring_length 1 = 10.5 →
  spring_length 9 = 14.5 := by
  sorry

/-- Proof that the spring_length function satisfies the given conditions. -/
theorem spring_length_conditions :
  spring_length 0 = 10 ∧ spring_length 1 = 10.5 := by
  sorry

end spring_length_at_9kg_spring_length_conditions_l3808_380815


namespace general_equation_l3808_380887

theorem general_equation (n : ℕ+) :
  (n + 1 : ℚ) / ((n + 1)^2 - 1) - 1 / (n * (n + 1) * (n + 2)) = 1 / (n + 1) := by
  sorry

end general_equation_l3808_380887


namespace one_hundred_twenty_fifth_number_with_digit_sum_5_l3808_380872

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth natural number whose digits sum to 5 -/
def nth_number_with_digit_sum_5 (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the 125th number with digit sum 5 is 41000 -/
theorem one_hundred_twenty_fifth_number_with_digit_sum_5 :
  nth_number_with_digit_sum_5 125 = 41000 := by sorry

end one_hundred_twenty_fifth_number_with_digit_sum_5_l3808_380872


namespace ratio_bounds_l3808_380879

theorem ratio_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  2 / 3 ≤ b / a ∧ b / a ≤ 3 / 2 := by
sorry

end ratio_bounds_l3808_380879


namespace handshake_theorem_l3808_380896

def corporate_event (n : ℕ) (completed_handshakes : ℕ) : Prop :=
  let total_handshakes := n * (n - 1) / 2
  total_handshakes - completed_handshakes = 42

theorem handshake_theorem :
  corporate_event 10 3 := by
  sorry

end handshake_theorem_l3808_380896


namespace ball_attendance_l3808_380877

theorem ball_attendance :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end ball_attendance_l3808_380877


namespace triple_transmission_more_accurate_main_theorem_l3808_380801

/-- Probability of correctly decoding 0 in single transmission -/
def single_transmission_prob (α : ℝ) : ℝ := 1 - α

/-- Probability of correctly decoding 0 in triple transmission -/
def triple_transmission_prob (α : ℝ) : ℝ := 3 * α * (1 - α)^2 + (1 - α)^3

/-- Theorem stating that triple transmission is more accurate than single for sending 0 when 0 < α < 0.5 -/
theorem triple_transmission_more_accurate (α : ℝ) 
  (h1 : 0 < α) (h2 : α < 0.5) : 
  triple_transmission_prob α > single_transmission_prob α := by
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < 0.5) (h3 : 0 < β) (h4 : β < 1) :
  triple_transmission_prob α > single_transmission_prob α ∧
  single_transmission_prob α = 1 - α ∧
  triple_transmission_prob α = 3 * α * (1 - α)^2 + (1 - α)^3 := by
  sorry

end triple_transmission_more_accurate_main_theorem_l3808_380801


namespace quadratic_roots_expression_l3808_380818

theorem quadratic_roots_expression (a b : ℝ) : 
  a^2 + a - 3 = 0 → b^2 + b - 3 = 0 → 4 * b^2 - a^3 = (53 + 8 * Real.sqrt 13) / 4 := by
  sorry

end quadratic_roots_expression_l3808_380818


namespace folded_rectangle_area_l3808_380808

/-- Given a rectangle with dimensions 5 by 8, when folded to form a trapezoid
    where corners touch, the area of the resulting trapezoid is 55/2. -/
theorem folded_rectangle_area (rect_width : ℝ) (rect_length : ℝ) 
    (h_width : rect_width = 5)
    (h_length : rect_length = 8)
    (trapezoid_short_base : ℝ)
    (h_short_base : trapezoid_short_base = 3)
    (trapezoid_long_base : ℝ)
    (h_long_base : trapezoid_long_base = rect_length)
    (trapezoid_height : ℝ)
    (h_height : trapezoid_height = rect_width) : 
  (trapezoid_short_base + trapezoid_long_base) * trapezoid_height / 2 = 55 / 2 := by
  sorry

end folded_rectangle_area_l3808_380808


namespace borrowing_interest_rate_l3808_380803

/-- Proves that the borrowing interest rate is 4% given the conditions of the problem -/
theorem borrowing_interest_rate
  (principal : ℝ)
  (time : ℝ)
  (lending_rate : ℝ)
  (gain_per_year : ℝ)
  (h1 : principal = 5000)
  (h2 : time = 2)
  (h3 : lending_rate = 0.06)
  (h4 : gain_per_year = 100)
  : (principal * lending_rate - gain_per_year) / principal = 0.04 := by
  sorry

#eval (5000 * 0.06 - 100) / 5000  -- Should output 0.04

end borrowing_interest_rate_l3808_380803


namespace difference_of_squares_l3808_380892

theorem difference_of_squares (x y : ℝ) : (x + 2*y) * (x - 2*y) = x^2 - 4*y^2 := by
  sorry

end difference_of_squares_l3808_380892


namespace expected_value_is_eight_l3808_380867

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The set of marble numbers -/
def marbles : Finset ℕ := Finset.range n

/-- The sum of all possible pairs of marbles -/
def sum_of_pairs : ℕ := (marbles.powerset.filter (fun s => s.card = 2)).sum (fun s => s.sum id)

/-- The number of ways to choose 2 marbles out of n -/
def num_combinations : ℕ := n.choose 2

/-- The expected value of the sum of two randomly drawn marbles -/
def expected_value : ℚ := (sum_of_pairs : ℚ) / num_combinations

theorem expected_value_is_eight : expected_value = 8 := by sorry

end expected_value_is_eight_l3808_380867


namespace third_circle_radius_value_l3808_380891

/-- A sequence of six circles tangent to each other and to two parallel lines -/
structure TangentCircles where
  radii : Fin 6 → ℝ
  smallest_radius : radii 0 = 10
  largest_radius : radii 5 = 20
  tangent : ∀ i : Fin 5, radii i < radii (i + 1)

/-- The radius of the third circle from the smallest in the sequence -/
def third_circle_radius (tc : TangentCircles) : ℝ := tc.radii 2

/-- The theorem stating that the radius of the third circle is 10 · ⁵√4 -/
theorem third_circle_radius_value (tc : TangentCircles) :
  third_circle_radius tc = 10 * (4 : ℝ) ^ (1/5) :=
sorry

end third_circle_radius_value_l3808_380891


namespace items_left_in_store_l3808_380841

/-- Given the number of items ordered, sold, and in the storeroom, 
    calculate the total number of items left in the whole store. -/
theorem items_left_in_store 
  (items_ordered : ℕ) 
  (items_sold : ℕ) 
  (items_in_storeroom : ℕ) 
  (h1 : items_ordered = 4458)
  (h2 : items_sold = 1561)
  (h3 : items_in_storeroom = 575) :
  items_ordered - items_sold + items_in_storeroom = 3472 :=
by sorry

end items_left_in_store_l3808_380841


namespace parallelogram_roots_l3808_380826

def polynomial (a : ℝ) (z : ℂ) : ℂ :=
  z^4 - 6*z^3 + 11*a*z^2 - 3*(2*a^2 + 3*a - 3)*z + 1

def forms_parallelogram (roots : List ℂ) : Prop :=
  ∃ (w₁ w₂ : ℂ), roots = [w₁, -w₁, w₂, -w₂]

theorem parallelogram_roots (a : ℝ) :
  (∃ (roots : List ℂ), (∀ z ∈ roots, polynomial a z = 0) ∧
                       roots.length = 4 ∧
                       forms_parallelogram roots) ↔ a = 3 :=
sorry

end parallelogram_roots_l3808_380826


namespace marcus_baseball_cards_l3808_380814

theorem marcus_baseball_cards 
  (initial_cards : ℝ) 
  (additional_cards : ℝ) 
  (h1 : initial_cards = 210.0) 
  (h2 : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 := by
  sorry

end marcus_baseball_cards_l3808_380814


namespace factorization_left_to_right_l3808_380868

/-- Factorization from left to right for x^2 - 1 -/
theorem factorization_left_to_right :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_left_to_right_l3808_380868


namespace arrangements_theorem_l3808_380874

def number_of_arrangements (n : ℕ) (a_not_first : Bool) (b_not_last : Bool) : ℕ :=
  if n = 5 ∧ a_not_first ∧ b_not_last then
    78
  else
    0

theorem arrangements_theorem :
  ∀ (n : ℕ) (a_not_first b_not_last : Bool),
    n = 5 → a_not_first → b_not_last →
    number_of_arrangements n a_not_first b_not_last = 78 :=
by
  sorry

end arrangements_theorem_l3808_380874


namespace largest_angle_of_triangle_l3808_380890

theorem largest_angle_of_triangle (a b c : ℝ) : 
  a = 70 → b = 80 → c = 180 - a - b → a + b + c = 180 → max a (max b c) = 80 := by
sorry

end largest_angle_of_triangle_l3808_380890


namespace binary_10101_is_21_l3808_380809

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_is_21_l3808_380809
