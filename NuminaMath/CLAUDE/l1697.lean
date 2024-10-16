import Mathlib

namespace NUMINAMATH_CALUDE_yamaimo_moving_problem_l1697_169787

/-- The Yamaimo family's moving problem -/
theorem yamaimo_moving_problem (initial_weight : ℝ) (initial_book_percentage : ℝ) 
  (final_book_percentage : ℝ) (new_weight : ℝ) : 
  initial_weight = 100 →
  initial_book_percentage = 99 / 100 →
  final_book_percentage = 95 / 100 →
  initial_weight * initial_book_percentage = 
    new_weight * final_book_percentage →
  initial_weight * (1 - initial_book_percentage) = 
    new_weight * (1 - final_book_percentage) →
  new_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_yamaimo_moving_problem_l1697_169787


namespace NUMINAMATH_CALUDE_initial_brownies_count_l1697_169749

/-- The number of brownies initially made by Mother -/
def initial_brownies : ℕ := sorry

/-- The number of brownies eaten by Father -/
def father_eaten : ℕ := 8

/-- The number of brownies eaten by Mooney -/
def mooney_eaten : ℕ := 4

/-- The number of new brownies added the next morning -/
def new_brownies : ℕ := 24

/-- The total number of brownies after adding the new ones -/
def total_brownies : ℕ := 36

theorem initial_brownies_count : initial_brownies = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_brownies_count_l1697_169749


namespace NUMINAMATH_CALUDE_expression_evaluation_l1697_169732

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1697_169732


namespace NUMINAMATH_CALUDE_problem_statement_l1697_169794

theorem problem_statement (a b : ℝ) 
  (h1 : a + 1 / (a + 1) = b + 1 / (b - 1) - 2)
  (h2 : a ≠ -1)
  (h3 : b ≠ 1)
  (h4 : a - b + 2 ≠ 0) :
  a * b - a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1697_169794


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l1697_169799

theorem parallelepiped_surface_area (a b c : ℝ) (h_sphere : a^2 + b^2 + c^2 = 12) 
  (h_volume : a * b * c = 8) : 2 * (a * b + b * c + c * a) = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_surface_area_l1697_169799


namespace NUMINAMATH_CALUDE_jill_study_time_l1697_169710

/-- Calculates the total minutes studied over 3 days given a specific study pattern -/
def totalMinutesStudied (day1Hours : ℕ) : ℕ :=
  let day2Hours := 2 * day1Hours
  let day3Hours := day2Hours - 1
  (day1Hours + day2Hours + day3Hours) * 60

/-- Proves that given Jill's study pattern, she studies for 540 minutes over 3 days -/
theorem jill_study_time : totalMinutesStudied 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_jill_study_time_l1697_169710


namespace NUMINAMATH_CALUDE_soda_bottle_count_l1697_169764

theorem soda_bottle_count (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 49) (h2 : diet_soda = 40) : 
  regular_soda + diet_soda = 89 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_count_l1697_169764


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1697_169785

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![2, m]

-- Define the sum of vectors
def vector_sum (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i + w i

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Theorem statement
theorem perpendicular_vectors (m : ℝ) : 
  dot_product (vector_sum a (b m)) a = 0 ↔ m = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1697_169785


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1697_169725

/-- A system of equations has exactly one solution if and only if a = 2 and b = -1 -/
theorem unique_solution_condition (a b : ℝ) : 
  (∃! x y, y = x^2 ∧ y = 2*x + b) ↔ (a = 2 ∧ b = -1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1697_169725


namespace NUMINAMATH_CALUDE_g_range_l1697_169762

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + (Real.pi / 2) * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9*x + 27)

theorem g_range :
  ∀ y ∈ Set.range g, π^2 / 6 ≤ y ∧ y ≤ 4*π^2 / 3 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = π^2 / 6 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = 4*π^2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_g_range_l1697_169762


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l1697_169781

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l1697_169781


namespace NUMINAMATH_CALUDE_smallest_c_for_quadratic_inequality_l1697_169783

theorem smallest_c_for_quadratic_inequality : 
  ∃ c : ℝ, c = 2 ∧ (∀ x : ℝ, -x^2 + 9*x - 14 ≥ 0 → x ≥ c) := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_quadratic_inequality_l1697_169783


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1697_169773

theorem bowling_ball_weight (kayak_weight : ℝ) (ball_weight : ℝ) :
  kayak_weight = 36 →
  9 * ball_weight = 2 * kayak_weight →
  ball_weight = 8 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1697_169773


namespace NUMINAMATH_CALUDE_walking_time_difference_l1697_169711

/-- Proof of the walking time difference between Cara and Don --/
theorem walking_time_difference 
  (total_distance : ℝ) 
  (cara_speed : ℝ) 
  (don_speed : ℝ) 
  (cara_distance : ℝ) 
  (h1 : total_distance = 45) 
  (h2 : cara_speed = 6) 
  (h3 : don_speed = 5) 
  (h4 : cara_distance = 30) : 
  (cara_distance / cara_speed) - ((total_distance - cara_distance) / don_speed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_difference_l1697_169711


namespace NUMINAMATH_CALUDE_roll_five_probability_l1697_169752

/-- A cube with six faces -/
structure Cube where
  faces : Fin 6 → ℕ

/-- The specific cube described in the problem -/
def problemCube : Cube :=
  { faces := λ i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => 2
    | ⟨3, _⟩ => 4
    | ⟨4, _⟩ => 5
    | ⟨5, _⟩ => 5
    | _ => 0 }

/-- The probability of rolling a specific number on the cube -/
def rollProbability (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i = n) Finset.univ).card / 6

/-- Theorem stating that the probability of rolling a 5 on the problem cube is 1/3 -/
theorem roll_five_probability :
  rollProbability problemCube 5 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_roll_five_probability_l1697_169752


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l1697_169771

/-- Given that α^2005 + β^2005 can be expressed as a polynomial in α+β and αβ,
    this function represents that polynomial. -/
def polynomial_expression (x y : ℝ) : ℝ := sorry

/-- The sum of the coefficients of the polynomial expression -/
def sum_of_coefficients : ℝ := sorry

/-- Theorem stating that the sum of the coefficients is 1 -/
theorem sum_of_coefficients_is_one : sum_of_coefficients = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l1697_169771


namespace NUMINAMATH_CALUDE_complex_absolute_value_equation_l1697_169761

theorem complex_absolute_value_equation :
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 10 ∧ t = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_equation_l1697_169761


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1697_169720

theorem polynomial_factorization (x : ℝ) :
  x^4 - 6*x^3 + 11*x^2 - 6*x = x*(x - 1)*(x - 2)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1697_169720


namespace NUMINAMATH_CALUDE_min_colors_17gon_l1697_169747

/-- A coloring of the vertices of a regular 17-gon -/
def Coloring := Fin 17 → ℕ

/-- The distance between two vertices in a 17-gon -/
def distance (i j : Fin 17) : Fin 17 := 
  Fin.ofNat ((i.val - j.val + 17) % 17)

/-- Whether two vertices should have different colors -/
def should_differ (i j : Fin 17) : Prop :=
  let d := distance i j
  d = 2 ∨ d = 4 ∨ d = 8 ∨ d = 15 ∨ d = 13 ∨ d = 9

/-- A valid coloring of the 17-gon -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ i j : Fin 17, should_differ i j → c i ≠ c j

/-- The main theorem -/
theorem min_colors_17gon : 
  (∃ c : Coloring, is_valid_coloring c ∧ Finset.card (Finset.image c Finset.univ) = 4) ∧
  (∀ c : Coloring, is_valid_coloring c → Finset.card (Finset.image c Finset.univ) ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_colors_17gon_l1697_169747


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1697_169759

theorem x_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 100
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1697_169759


namespace NUMINAMATH_CALUDE_min_weights_to_balance_three_grams_l1697_169721

/-- Represents a combination of weights -/
structure WeightCombination :=
  (nine_gram : ℤ)
  (thirteen_gram : ℤ)

/-- Calculates the total weight of a combination -/
def total_weight (w : WeightCombination) : ℤ :=
  9 * w.nine_gram + 13 * w.thirteen_gram

/-- Calculates the total number of weights used -/
def num_weights (w : WeightCombination) : ℕ :=
  w.nine_gram.natAbs + w.thirteen_gram.natAbs

/-- Checks if a combination balances 3 grams -/
def balances_three_grams (w : WeightCombination) : Prop :=
  total_weight w = 3

/-- The set of all weight combinations that balance 3 grams -/
def balancing_combinations : Set WeightCombination :=
  {w | balances_three_grams w}

theorem min_weights_to_balance_three_grams :
  ∃ (w : WeightCombination),
    w ∈ balancing_combinations ∧
    num_weights w = 7 ∧
    ∀ (w' : WeightCombination),
      w' ∈ balancing_combinations →
      num_weights w' ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_min_weights_to_balance_three_grams_l1697_169721


namespace NUMINAMATH_CALUDE_grape_jelly_beans_problem_l1697_169766

theorem grape_jelly_beans_problem (g c : ℕ) : 
  g = 3 * c →                   -- Initial ratio
  g - 15 = 5 * (c - 5) →        -- Final ratio after eating
  g = 15                        -- Conclusion: original number of grape jelly beans
  := by sorry

end NUMINAMATH_CALUDE_grape_jelly_beans_problem_l1697_169766


namespace NUMINAMATH_CALUDE_joan_has_sixteen_seashells_l1697_169719

/-- The number of seashells Joan has after giving some to Mike -/
def joans_remaining_seashells (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem: Joan has 16 seashells after giving Mike 63 of her initial 79 seashells -/
theorem joan_has_sixteen_seashells :
  joans_remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_sixteen_seashells_l1697_169719


namespace NUMINAMATH_CALUDE_binary_11011000_equals_quaternary_3120_l1697_169731

/-- Converts a binary (base 2) number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 11011000₂ -/
def binary_11011000 : List Bool := [true, true, false, true, true, false, false, false]

theorem binary_11011000_equals_quaternary_3120 :
  decimal_to_quaternary (binary_to_decimal binary_11011000) = [3, 1, 2, 0] := by
  sorry

#eval decimal_to_quaternary (binary_to_decimal binary_11011000)

end NUMINAMATH_CALUDE_binary_11011000_equals_quaternary_3120_l1697_169731


namespace NUMINAMATH_CALUDE_binary_digit_difference_l1697_169770

/-- The number of digits in the binary representation of a positive integer -/
def binaryDigits (n : ℕ+) : ℕ := Nat.log2 n + 1

/-- The difference in the number of binary digits between 950 and 150 -/
theorem binary_digit_difference : binaryDigits 950 - binaryDigits 150 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l1697_169770


namespace NUMINAMATH_CALUDE_range_of_f_when_k_4_range_of_k_for_monotone_f_l1697_169708

/-- The function f(x) = (k-2)x^2 + 2kx - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + 2 * k * x - 3

/-- The range of f(x) when k = 4 in the interval (-4, 1) -/
theorem range_of_f_when_k_4 :
  Set.Icc (-11 : ℝ) 7 = Set.image (f 4) (Set.Ioo (-4 : ℝ) 1) := by sorry

/-- The range of k for which f(x) is monotonically increasing in [1, 2] -/
theorem range_of_k_for_monotone_f :
  ∀ k : ℝ, (∀ x y : ℝ, x ∈ Set.Icc (1 : ℝ) 2 → y ∈ Set.Icc (1 : ℝ) 2 → x ≤ y → f k x ≤ f k y) ↔
  k ∈ Set.Ici (4/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_f_when_k_4_range_of_k_for_monotone_f_l1697_169708


namespace NUMINAMATH_CALUDE_sum_base8_equals_467_l1697_169740

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Adds two base-8 numbers --/
def addBase8 (a b : ℕ) : ℕ := base10ToBase8 (base8ToBase10 a + base8ToBase10 b)

theorem sum_base8_equals_467 :
  addBase8 (addBase8 236 157) 52 = 467 := by sorry

end NUMINAMATH_CALUDE_sum_base8_equals_467_l1697_169740


namespace NUMINAMATH_CALUDE_people_behind_yuna_l1697_169765

theorem people_behind_yuna (total : Nat) (in_front : Nat) (behind : Nat) : 
  total = 7 → in_front = 2 → behind = total - in_front - 1 → behind = 4 := by
  sorry

end NUMINAMATH_CALUDE_people_behind_yuna_l1697_169765


namespace NUMINAMATH_CALUDE_prove_first_divisor_l1697_169768

def least_number : ℕ := 1394

def first_divisor : ℕ := 6

theorem prove_first_divisor :
  (least_number % first_divisor = 14) ∧
  (2535 % first_divisor = 1929) ∧
  (40 % first_divisor = 34) :=
by sorry

end NUMINAMATH_CALUDE_prove_first_divisor_l1697_169768


namespace NUMINAMATH_CALUDE_d_sufficient_not_necessary_for_a_l1697_169701

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B → C) ∧ (C → B))
variable (h3 : C → D ∧ ¬(D → C))

-- Theorem statement
theorem d_sufficient_not_necessary_for_a :
  D → A ∧ ¬(A → D) :=
sorry

end NUMINAMATH_CALUDE_d_sufficient_not_necessary_for_a_l1697_169701


namespace NUMINAMATH_CALUDE_min_production_quantity_to_break_even_l1697_169788

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 250 * x

-- Define the break-even condition
def breaks_even (x : ℝ) : Prop := sales_revenue x ≥ total_cost x

-- Theorem statement
theorem min_production_quantity_to_break_even :
  ∃ (x : ℝ), x = 150 ∧ x ∈ Set.Ioo 0 240 ∧ breaks_even x ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 240 → breaks_even y → y ≥ x :=
sorry

end NUMINAMATH_CALUDE_min_production_quantity_to_break_even_l1697_169788


namespace NUMINAMATH_CALUDE_inequality_solution_l1697_169703

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 1) ≥ x^2 / (x - 1) + 7/6) ↔ 
  (x < (-1 - Real.sqrt 5) / 2 ∨ 
   (-1 < x ∧ x < 1) ∨ 
   x > (-1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1697_169703


namespace NUMINAMATH_CALUDE_system_solution_l1697_169780

theorem system_solution (u v w : ℚ) 
  (eq1 : 3 * u - 4 * v + w = 26)
  (eq2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1697_169780


namespace NUMINAMATH_CALUDE_no_solution_condition_l1697_169727

theorem no_solution_condition (r : ℝ) :
  (∀ x y : ℝ, x^2 = y^2 ∧ (x - r)^2 + y^2 = 1 → False) ↔ r < -Real.sqrt 2 ∨ r > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1697_169727


namespace NUMINAMATH_CALUDE_geometric_progression_terms_l1697_169753

theorem geometric_progression_terms (a q : ℝ) : 
  a + a * q = 20 → a * q^2 + a * q^3 = 20/9 →
  ((a = 15 ∧ q = 1/3) ∨ (a = 30 ∧ q = -1/3)) ∧
  (∃ (terms : Fin 4 → ℝ), 
    (terms 0 = a ∧ terms 1 = a * q ∧ terms 2 = a * q^2 ∧ terms 3 = a * q^3) ∧
    ((terms 0 = 15 ∧ terms 1 = 5 ∧ terms 2 = 5/3 ∧ terms 3 = 5/9) ∨
     (terms 0 = 30 ∧ terms 1 = -10 ∧ terms 2 = 10/3 ∧ terms 3 = -10/9))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_terms_l1697_169753


namespace NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l1697_169723

theorem polar_to_rectangular_coordinates :
  let r : ℝ := 4
  let θ : ℝ := 5 * π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = -2 * Real.sqrt 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l1697_169723


namespace NUMINAMATH_CALUDE_f_at_four_equals_zero_l1697_169729

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_at_four_equals_zero : f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_at_four_equals_zero_l1697_169729


namespace NUMINAMATH_CALUDE_megaTek_circle_graph_error_l1697_169772

theorem megaTek_circle_graph_error :
  let total_degrees : ℕ := 360
  let manufacturing_degrees : ℕ := 252
  let administration_degrees : ℕ := 68
  let research_degrees : ℕ := 40
  manufacturing_degrees + administration_degrees + research_degrees = total_degrees :=
by
  sorry

end NUMINAMATH_CALUDE_megaTek_circle_graph_error_l1697_169772


namespace NUMINAMATH_CALUDE_final_price_fraction_l1697_169784

/-- The final price of a dress for a staff member after discounts and tax -/
def final_price (d : ℝ) : ℝ :=
  let discount_price := d * (1 - 0.45)
  let staff_price := discount_price * (1 - 0.40)
  staff_price * (1 + 0.08)

/-- Theorem stating the final price as a fraction of the initial price -/
theorem final_price_fraction (d : ℝ) :
  final_price d = 0.3564 * d := by
  sorry

end NUMINAMATH_CALUDE_final_price_fraction_l1697_169784


namespace NUMINAMATH_CALUDE_candy_ratio_l1697_169715

/-- Proves that the ratio of grape candies to cherry candies is 3:1 given the problem conditions --/
theorem candy_ratio (cherry : ℕ) (grape : ℕ) (apple : ℕ) (total_cost : ℚ) :
  grape = 3 * cherry →
  apple = 2 * grape →
  total_cost = 200 →
  (2.5 : ℚ) * (cherry + grape + apple) = total_cost →
  (grape : ℚ) / cherry = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l1697_169715


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l1697_169757

/-- The probability of selecting two red balls from a bag containing 6 red, 5 blue, and 2 green balls, when 2 balls are picked at random. -/
theorem probability_two_red_balls (red blue green : ℕ) (h1 : red = 6) (h2 : blue = 5) (h3 : green = 2) :
  let total := red + blue + green
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l1697_169757


namespace NUMINAMATH_CALUDE_base_conversion_problem_l1697_169755

theorem base_conversion_problem (a b c : ℕ) (h1 : a ≤ 6) (h2 : b ≤ 6) (h3 : c ≤ 6) 
  (h4 : a ≤ 8) (h5 : b ≤ 8) (h6 : c ≤ 8) :
  (49 * a + 7 * b + c = 81 * c + 9 * b + a) → (49 * a + 7 * b + c = 248) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l1697_169755


namespace NUMINAMATH_CALUDE_dave_phone_difference_l1697_169743

theorem dave_phone_difference (initial_apps initial_files final_apps final_files : ℕ) : 
  initial_apps = 11 → 
  initial_files = 3 → 
  final_apps = 2 → 
  final_files = 24 → 
  final_files - final_apps = 22 := by
  sorry

end NUMINAMATH_CALUDE_dave_phone_difference_l1697_169743


namespace NUMINAMATH_CALUDE_students_playing_basketball_l1697_169779

/-- The number of students who play basketball in a college, given the total number of students,
    the number of students who play cricket, and the number of students who play both sports. -/
theorem students_playing_basketball
  (total : ℕ)
  (cricket : ℕ)
  (both : ℕ)
  (h1 : total = 880)
  (h2 : cricket = 500)
  (h3 : both = 220) :
  total = cricket + (cricket + both - total) - both :=
by sorry

end NUMINAMATH_CALUDE_students_playing_basketball_l1697_169779


namespace NUMINAMATH_CALUDE_factorization_count_l1697_169730

theorem factorization_count : 
  ∃! (S : Finset ℤ), 
    (∀ m : ℤ, m ∈ S ↔ 
      ∃ a b : ℤ, ∀ x : ℝ, x^2 + m*x - 16 = (x + a)*(x + b)) ∧ 
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_factorization_count_l1697_169730


namespace NUMINAMATH_CALUDE_symmetry_point_of_sine_function_l1697_169763

/-- Given a function f(x) = sin(ωx + π/6) with ω > 0, if the distance between adjacent
    symmetry axes is π/2 and the graph is symmetrical about (x₀, 0) where x₀ ∈ [0, π/2],
    then x₀ = 5π/12 -/
theorem symmetry_point_of_sine_function (ω : ℝ) (x₀ : ℝ) :
  ω > 0 →
  (2 * π) / ω = π →
  x₀ ∈ Set.Icc 0 (π / 2) →
  (∀ x, Real.sin (ω * x + π / 6) = Real.sin (ω * (2 * x₀ - x) + π / 6)) →
  x₀ = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_of_sine_function_l1697_169763


namespace NUMINAMATH_CALUDE_only_point_distance_no_conditional_l1697_169735

-- Define the four types of mathematical problems
inductive MathProblem
  | QuadraticEquation
  | LineCircleRelationship
  | StudentRanking
  | PointDistance

-- Define a function that determines if a problem requires conditional statements
def requiresConditionalStatements (problem : MathProblem) : Prop :=
  match problem with
  | MathProblem.QuadraticEquation => true
  | MathProblem.LineCircleRelationship => true
  | MathProblem.StudentRanking => true
  | MathProblem.PointDistance => false

-- Theorem stating that only PointDistance does not require conditional statements
theorem only_point_distance_no_conditional :
  ∀ (problem : MathProblem),
    ¬(requiresConditionalStatements problem) ↔ problem = MathProblem.PointDistance := by
  sorry

#check only_point_distance_no_conditional

end NUMINAMATH_CALUDE_only_point_distance_no_conditional_l1697_169735


namespace NUMINAMATH_CALUDE_problem_equivalence_l1697_169744

theorem problem_equivalence :
  (3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2) ∧
  (|Real.sqrt 3 - Real.sqrt 2| + |Real.sqrt 3 - 2| + Real.sqrt ((-2)^2) = 4 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_equivalence_l1697_169744


namespace NUMINAMATH_CALUDE_sum_of_integers_l1697_169700

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1697_169700


namespace NUMINAMATH_CALUDE_johns_base_salary_l1697_169709

/-- John's monthly savings rate as a decimal -/
def savings_rate : ℝ := 0.10

/-- John's monthly savings amount in dollars -/
def savings_amount : ℝ := 400

/-- Theorem stating John's monthly base salary -/
theorem johns_base_salary :
  ∀ (base_salary : ℝ),
  base_salary * savings_rate = savings_amount →
  base_salary = 4000 := by
  sorry

end NUMINAMATH_CALUDE_johns_base_salary_l1697_169709


namespace NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l1697_169795

theorem minimum_questionnaires_to_mail 
  (response_rate : ℝ) 
  (required_responses : ℕ) 
  (h1 : response_rate = 0.7) 
  (h2 : required_responses = 300) : 
  ℕ := by
  
  sorry

#check minimum_questionnaires_to_mail

end NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l1697_169795


namespace NUMINAMATH_CALUDE_contrapositive_theorem_l1697_169736

theorem contrapositive_theorem (a b c : ℝ) :
  (abc = 0 → a = 0 ∨ b = 0 ∨ c = 0) ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → abc ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_l1697_169736


namespace NUMINAMATH_CALUDE_intersection_point_l1697_169741

theorem intersection_point (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = 2*a ∧ p.2 = |p.1 - a| - 1) → a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_l1697_169741


namespace NUMINAMATH_CALUDE_triangle_properties_l1697_169739

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 - t.c^2 = t.b^2 - (8 * t.b * t.c) / 5)
  (h2 : t.a = 6)
  (h3 : Real.sin t.B = 4/5) :
  (Real.sin t.A = 3/5) ∧ 
  ((1/2 * t.b * t.c * Real.sin t.A = 24) ∨ 
   (1/2 * t.b * t.c * Real.sin t.A = 168/25)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1697_169739


namespace NUMINAMATH_CALUDE_postal_stamp_problem_l1697_169750

theorem postal_stamp_problem :
  ∀ (x : ℕ),
  (75 : ℕ) = 40 + (75 - 40) →
  (480 : ℕ) = 40 * 5 + (75 - 40) * x →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_postal_stamp_problem_l1697_169750


namespace NUMINAMATH_CALUDE_largest_solution_bound_l1697_169791

theorem largest_solution_bound (x : ℝ) : 
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60) →
  x ≤ -0.642 ∧ x > -0.643 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_bound_l1697_169791


namespace NUMINAMATH_CALUDE_club_contribution_proof_l1697_169713

/-- Proves that the initial contribution per member is $300 --/
theorem club_contribution_proof (n : ℕ) (x : ℝ) : 
  n = 10 → -- Initial number of members
  (n + 5) * (x - 100) = n * x → -- Total amount remains constant with 5 more members
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_club_contribution_proof_l1697_169713


namespace NUMINAMATH_CALUDE_largest_common_term_l1697_169789

def is_in_arithmetic_sequence (a : ℕ) (first d : ℕ) : Prop :=
  ∃ k : ℕ, a = first + k * d

theorem largest_common_term (a₁ d₁ a₂ d₂ : ℕ) (h₁ : a₁ = 3) (h₂ : d₁ = 8) (h₃ : a₂ = 5) (h₄ : d₂ = 9) :
  (∀ n : ℕ, n > 131 ∧ n ≤ 150 → ¬(is_in_arithmetic_sequence n a₁ d₁ ∧ is_in_arithmetic_sequence n a₂ d₂)) ∧
  (is_in_arithmetic_sequence 131 a₁ d₁ ∧ is_in_arithmetic_sequence 131 a₂ d₂) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l1697_169789


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l1697_169705

theorem quadratic_roots_average (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 - 4 * a * x + b
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 →
  (x + y) / 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l1697_169705


namespace NUMINAMATH_CALUDE_kira_away_time_l1697_169712

/-- Represents the eating rate of the cat in hours per pound of kibble -/
def eating_rate : ℝ := 4

/-- Represents the initial amount of kibble in pounds -/
def initial_kibble : ℝ := 3

/-- Represents the remaining amount of kibble in pounds -/
def remaining_kibble : ℝ := 1

/-- Calculates the time Kira was away based on the given conditions -/
def time_away : ℝ := (initial_kibble - remaining_kibble) * eating_rate

/-- Proves that the time Kira was away from home is 8 hours -/
theorem kira_away_time : time_away = 8 := by
  sorry

end NUMINAMATH_CALUDE_kira_away_time_l1697_169712


namespace NUMINAMATH_CALUDE_student_arrangement_count_l1697_169786

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of positions available for female students (not at the ends) -/
def female_positions : ℕ := total_students - 2

/-- The number of ways to arrange female students in available positions -/
def female_arrangements : ℕ := Nat.choose female_positions num_female

/-- The number of ways to arrange the remaining male students -/
def male_arrangements : ℕ := Nat.factorial num_male

/-- The total number of arrangements -/
def total_arrangements : ℕ := female_arrangements * male_arrangements

theorem student_arrangement_count :
  total_arrangements = 36 := by sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l1697_169786


namespace NUMINAMATH_CALUDE_sin_value_for_given_condition_l1697_169767

theorem sin_value_for_given_condition (θ : Real) 
  (h1 : 5 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < π) : 
  Real.sin θ = (Real.sqrt 41 - 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_for_given_condition_l1697_169767


namespace NUMINAMATH_CALUDE_square_root_of_increased_number_l1697_169790

theorem square_root_of_increased_number (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 + 2) = Real.sqrt ((Real.sqrt x^2) + 2) :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_increased_number_l1697_169790


namespace NUMINAMATH_CALUDE_solution_properties_l1697_169716

theorem solution_properties (a b : ℝ) (h : a^2 - 5*b^2 = 1) :
  (0 < a + b * Real.sqrt 5 → a ≥ 0) ∧
  (1 < a + b * Real.sqrt 5 → a ≥ 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_properties_l1697_169716


namespace NUMINAMATH_CALUDE_empty_boxes_count_l1697_169742

/-- The number of empty boxes after n operations, where n is the number of non-empty boxes. -/
def empty_boxes (n : ℕ) : ℤ :=
  -1 + 6 * n

/-- The theorem stating that when there are 34 non-empty boxes, there are 203 empty boxes. -/
theorem empty_boxes_count : empty_boxes 34 = 203 := by
  sorry

end NUMINAMATH_CALUDE_empty_boxes_count_l1697_169742


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1697_169758

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 0, a}
  let B : Set ℝ := {0, Real.sqrt a}
  B ⊆ A → a = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1697_169758


namespace NUMINAMATH_CALUDE_calculate_number_of_children_l1697_169746

/-- Calculates the number of children in a family based on their savings distribution --/
theorem calculate_number_of_children 
  (husband_contribution : ℝ) 
  (wife_contribution : ℝ) 
  (saving_period_months : ℕ) 
  (weeks_per_month : ℕ) 
  (amount_per_child : ℝ) 
  (h1 : husband_contribution = 335)
  (h2 : wife_contribution = 225)
  (h3 : saving_period_months = 6)
  (h4 : weeks_per_month = 4)
  (h5 : amount_per_child = 1680) :
  ⌊(((husband_contribution + wife_contribution) * (saving_period_months * weeks_per_month)) / 2) / amount_per_child⌋ = 4 := by
  sorry


end NUMINAMATH_CALUDE_calculate_number_of_children_l1697_169746


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1697_169775

def A : Set ℝ := {x | (1 : ℝ) / (x - 1) ≤ 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1697_169775


namespace NUMINAMATH_CALUDE_tan_three_properties_l1697_169760

theorem tan_three_properties (θ : Real) (h : Real.tan θ = 3) :
  (Real.cos θ / (Real.sin θ + 2 * Real.cos θ) = 1/5) ∧
  (Real.tan (θ - 5 * Real.pi / 4) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_three_properties_l1697_169760


namespace NUMINAMATH_CALUDE_initial_wage_solution_l1697_169734

def initial_wage_problem (x : ℝ) : Prop :=
  let after_raise := x * 1.20
  let after_cut := after_raise * 0.75
  after_cut = 9

theorem initial_wage_solution :
  ∃ x : ℝ, initial_wage_problem x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_wage_solution_l1697_169734


namespace NUMINAMATH_CALUDE_reflection_problem_l1697_169738

/-- Reflection of a point across a line --/
def reflect (x y m b : ℝ) : ℝ × ℝ := sorry

/-- The problem statement --/
theorem reflection_problem (m b : ℝ) :
  reflect (-4) 2 m b = (6, 0) → m + b = 1 := by sorry

end NUMINAMATH_CALUDE_reflection_problem_l1697_169738


namespace NUMINAMATH_CALUDE_max_quadratic_solution_power_l1697_169737

/-- Given positive integers a, b, c that are powers of k, and r is the unique real solution
    to ax^2 - bx + c = 0 where r < 100, prove that the maximum possible value of r is 64 -/
theorem max_quadratic_solution_power (k a b c : ℕ+) (r : ℝ) :
  (∃ m n p : ℕ, a = k ^ m ∧ b = k ^ n ∧ c = k ^ p) →
  (∀ x : ℝ, a * x^2 - b * x + c = 0 ↔ x = r) →
  r < 100 →
  r ≤ 64 :=
sorry

end NUMINAMATH_CALUDE_max_quadratic_solution_power_l1697_169737


namespace NUMINAMATH_CALUDE_coins_missing_l1697_169756

theorem coins_missing (total : ℚ) : 
  let lost := (1 : ℚ) / 3 * total
  let found := (3 : ℚ) / 4 * lost
  let remaining := total - lost + found
  total - remaining = (1 : ℚ) / 12 * total := by
sorry

end NUMINAMATH_CALUDE_coins_missing_l1697_169756


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l1697_169745

theorem tan_half_product_squared (a b : ℝ) :
  6 * (Real.cos a + Real.cos b) + 3 * (Real.sin a + Real.sin b) + 5 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2)) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l1697_169745


namespace NUMINAMATH_CALUDE_star_three_four_l1697_169714

/-- The ⋆ operation defined on real numbers -/
def star (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

/-- Theorem stating that 3 ⋆ 4 = 0 -/
theorem star_three_four : star 3 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l1697_169714


namespace NUMINAMATH_CALUDE_unique_zero_addition_l1697_169769

theorem unique_zero_addition (x : ℤ) :
  (∀ n : ℤ, n + x = n) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_addition_l1697_169769


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1697_169722

def f (x : ℝ) := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1697_169722


namespace NUMINAMATH_CALUDE_box_side_length_l1697_169751

/-- Proves that the length of one side of a cubic box can be calculated given the total volume,
    total cost, and cost per box. -/
theorem box_side_length 
  (cost_per_box : ℝ) 
  (total_volume : ℝ) 
  (total_cost : ℝ) 
  (cost_per_box_positive : cost_per_box > 0)
  (total_volume_positive : total_volume > 0)
  (total_cost_positive : total_cost > 0) :
  ∃ (side_length : ℝ), 
    side_length = (total_volume / (total_cost / cost_per_box)) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_box_side_length_l1697_169751


namespace NUMINAMATH_CALUDE_diagonal_path_shorter_than_sides_l1697_169702

theorem diagonal_path_shorter_than_sides (ε : ℝ) (h : ε > 0) : ∃ δ : ℝ, 
  0 < δ ∧ δ < ε ∧ 
  |(2 - Real.sqrt 2) / 2 - 0.293| < δ :=
sorry

end NUMINAMATH_CALUDE_diagonal_path_shorter_than_sides_l1697_169702


namespace NUMINAMATH_CALUDE_sum_of_angles_with_tangent_roots_l1697_169798

theorem sum_of_angles_with_tangent_roots (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (∃ x y : Real, x^2 - 5*x + 6 = 0 ∧ y^2 - 5*y + 6 = 0 ∧ Real.tan α = x ∧ Real.tan β = y) →
  α + β = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_with_tangent_roots_l1697_169798


namespace NUMINAMATH_CALUDE_sum_of_squares_l1697_169728

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 4)
  (eq2 : y^2 - 5*z = 5)
  (eq3 : z^2 - 7*x = -8) :
  x^2 + y^2 + z^2 = 83/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1697_169728


namespace NUMINAMATH_CALUDE_square_side_length_l1697_169718

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 289 ∧ area = side * side → side = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1697_169718


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l1697_169704

theorem cricket_bat_cost_price 
  (profit_a_to_b : Real) 
  (profit_b_to_c : Real) 
  (price_c_pays : Real) : 
  profit_a_to_b = 0.20 →
  profit_b_to_c = 0.25 →
  price_c_pays = 234 →
  ∃ (cost_price_a : Real), 
    cost_price_a = 156 ∧ 
    price_c_pays = (1 + profit_b_to_c) * ((1 + profit_a_to_b) * cost_price_a) :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l1697_169704


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l1697_169724

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) 
  (total_progressives : ℚ) :
  num_provinces = 6 →
  num_traditionalists_per_province = total_progressives / 9 →
  (num_provinces : ℚ) * num_traditionalists_per_province / 
    (total_progressives + (num_provinces : ℚ) * num_traditionalists_per_province) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l1697_169724


namespace NUMINAMATH_CALUDE_fixed_term_deposit_result_l1697_169717

/-- Calculates the total amount after a fixed term deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that the total amount after the fixed term deposit is 21998 yuan -/
theorem fixed_term_deposit_result : 
  let principal : ℝ := 20000
  let rate : ℝ := 0.0333
  let time : ℝ := 3
  totalAmount principal rate time = 21998 := by
sorry


end NUMINAMATH_CALUDE_fixed_term_deposit_result_l1697_169717


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l1697_169782

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop := s1.side = s2.side

-- Theorem: Not all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ s1 s2 : Square, congruent s1 s2 := by
  sorry

-- All other properties of squares that are true
theorem all_squares_equiangular (s : Square) : True := by
  sorry

theorem all_squares_rectangles (s : Square) : True := by
  sorry

theorem all_squares_regular_polygons (s : Square) : True := by
  sorry

theorem all_squares_similar (s1 s2 : Square) : True := by
  sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l1697_169782


namespace NUMINAMATH_CALUDE_sum_of_root_pairs_is_124_l1697_169776

def root_pairs : List (Nat × Nat) := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]

def sum_of_pairs (pairs : List (Nat × Nat)) : Nat :=
  pairs.map (fun (a, b) => a + b) |> List.sum

theorem sum_of_root_pairs_is_124 : sum_of_pairs root_pairs = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_root_pairs_is_124_l1697_169776


namespace NUMINAMATH_CALUDE_amy_candy_problem_l1697_169748

theorem amy_candy_problem (initial_candy : ℕ) : ∃ (given : ℕ), 
  given + 5 ≤ initial_candy ∧ given - 5 = 1 → given = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_candy_problem_l1697_169748


namespace NUMINAMATH_CALUDE_infinite_sum_of_squares_with_neighbors_l1697_169754

theorem infinite_sum_of_squares_with_neighbors (k : ℕ) :
  ∃ n : ℕ,
    (∃ a b : ℕ, n = a^2 + b^2) ∧
    (∀ x y : ℕ, (n - 1) ≠ x^2 + y^2) ∧
    (∀ x y : ℕ, (n + 1) ≠ x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_infinite_sum_of_squares_with_neighbors_l1697_169754


namespace NUMINAMATH_CALUDE_cosine_angle_OAB_l1697_169792

/-- Given points A and B in a 2D Cartesian coordinate system with O as the origin,
    prove that the cosine of angle OAB is equal to -√2/10. -/
theorem cosine_angle_OAB (A B : ℝ × ℝ) (h_A : A = (-3, -4)) (h_B : B = (5, -12)) :
  let O : ℝ × ℝ := (0, 0)
  let AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let dot_product := AO.1 * AB.1 + AO.2 * AB.2
  let magnitude_AO := Real.sqrt (AO.1^2 + AO.2^2)
  let magnitude_AB := Real.sqrt (AB.1^2 + AB.2^2)
  dot_product / (magnitude_AO * magnitude_AB) = -Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_angle_OAB_l1697_169792


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_l1697_169797

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_l1697_169797


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1697_169778

/-- Parabola defined by y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line with slope √3 passing through a point -/
def line_with_slope_sqrt3 (x y x0 y0 : ℝ) : Prop :=
  y - y0 = Real.sqrt 3 * (x - x0)

/-- Focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Definition of the length of a line segment on the parabola -/
def segment_length (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

theorem parabola_intersection_length :
  ∀ A B : ℝ × ℝ,
  (∃ x1 x2 y1 y2 : ℝ,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    line_with_slope_sqrt3 x1 y1 focus.1 focus.2 ∧
    line_with_slope_sqrt3 x2 y2 focus.1 focus.2) →
  segment_length A.1 B.1 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1697_169778


namespace NUMINAMATH_CALUDE_matching_probability_five_pairs_l1697_169726

/-- A box containing shoes -/
structure ShoeBox where
  pairs : ℕ
  total : ℕ
  total_eq : total = 2 * pairs

/-- The probability of selecting a matching pair of shoes -/
def matchingProbability (box : ShoeBox) : ℚ :=
  box.pairs / (box.total * (box.total - 1) / 2)

/-- Theorem: The probability of selecting a matching pair from a box with 5 pairs is 1/9 -/
theorem matching_probability_five_pairs :
  let box : ShoeBox := ⟨5, 10, rfl⟩
  matchingProbability box = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_matching_probability_five_pairs_l1697_169726


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1697_169793

theorem linear_equation_solution (m : ℝ) :
  (∃ k : ℝ, ∀ x, 3 * x^(m-1) + 2 = k * x + (-3)) →
  (∀ y, 3 * m * y + 2 * y = 3 + m ↔ y = 5/8) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1697_169793


namespace NUMINAMATH_CALUDE_min_distance_sum_l1697_169706

/-- A rectangle with sides 20 cm and 10 cm -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AB_length : dist A B = 20)
  (BC_length : dist B C = 10)

/-- The sum of distances BM + MN -/
def distance_sum (rect : Rectangle) (M : ℝ × ℝ) (N : ℝ × ℝ) : ℝ :=
  dist rect.B M + dist M N

/-- M is on AC -/
def M_on_AC (rect : Rectangle) (M : ℝ × ℝ) : Prop :=
  sorry

/-- N is on AB -/
def N_on_AB (rect : Rectangle) (N : ℝ × ℝ) : Prop :=
  sorry

theorem min_distance_sum (rect : Rectangle) :
  ∃ (M N : ℝ × ℝ), M_on_AC rect M ∧ N_on_AB rect N ∧
    (∀ (M' N' : ℝ × ℝ), M_on_AC rect M' → N_on_AB rect N' →
      distance_sum rect M N ≤ distance_sum rect M' N') ∧
    distance_sum rect M N = 16 :=
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1697_169706


namespace NUMINAMATH_CALUDE_edwards_initial_money_l1697_169796

/-- Given that Edward spent $16 and has $2 left, his initial amount of money was $18. -/
theorem edwards_initial_money :
  ∀ (initial spent left : ℕ),
    spent = 16 →
    left = 2 →
    initial = spent + left →
    initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l1697_169796


namespace NUMINAMATH_CALUDE_inequality_proof_l1697_169733

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 3) :
  Real.sqrt (3 - ((x + y) / 2)^2) + Real.sqrt (3 - ((y + z) / 2)^2) + Real.sqrt (3 - ((z + x) / 2)^2) ≥ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1697_169733


namespace NUMINAMATH_CALUDE_circles_covering_path_implies_odd_l1697_169774

/-- A configuration of n circles on a plane. -/
structure CircleConfiguration (n : ℕ) where
  /-- The set of circles. -/
  circles : Fin n → Set (ℝ × ℝ)
  /-- Any two circles intersect at exactly two points. -/
  two_intersections : ∀ (i j : Fin n), i ≠ j → ∃! (p q : ℝ × ℝ), p ≠ q ∧ p ∈ circles i ∧ p ∈ circles j ∧ q ∈ circles i ∧ q ∈ circles j
  /-- No three circles have a common point. -/
  no_triple_intersection : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬∃ (p : ℝ × ℝ), p ∈ circles i ∧ p ∈ circles j ∧ p ∈ circles k

/-- A path that covers all circles in the configuration. -/
def CoveringPath (n : ℕ) (config : CircleConfiguration n) :=
  ∃ (path : ℕ → Fin n), ∀ (i : Fin n), ∃ (k : ℕ), path k = i

/-- The main theorem: if there exists a covering path for n circles satisfying the given conditions,
    then n must be odd. -/
theorem circles_covering_path_implies_odd (n : ℕ) (config : CircleConfiguration n) :
  CoveringPath n config → Odd n :=
sorry

end NUMINAMATH_CALUDE_circles_covering_path_implies_odd_l1697_169774


namespace NUMINAMATH_CALUDE_hedgehog_strawberries_l1697_169777

theorem hedgehog_strawberries (num_hedgehogs num_baskets strawberries_per_basket : ℕ) 
  (remaining_fraction : ℚ) : 
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  remaining_fraction = 2 / 9 →
  (num_baskets * strawberries_per_basket * (1 - remaining_fraction)) / num_hedgehogs = 1050 := by
  sorry

end NUMINAMATH_CALUDE_hedgehog_strawberries_l1697_169777


namespace NUMINAMATH_CALUDE_boat_trip_distance_l1697_169707

/-- Proves that given a boat with speed 9 kmph in standing water, a stream with speed 1.5 kmph,
    and a round trip time of 48 hours, the distance to the destination is 210 km. -/
theorem boat_trip_distance (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 1.5 →
  total_time = 48 →
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time →
  distance = 210 := by
  sorry

end NUMINAMATH_CALUDE_boat_trip_distance_l1697_169707
