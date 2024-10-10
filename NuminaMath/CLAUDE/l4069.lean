import Mathlib

namespace complex_cube_sum_magnitude_l4069_406922

theorem complex_cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 3)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) = 81/2 := by
  sorry

end complex_cube_sum_magnitude_l4069_406922


namespace negation_of_proposition_l4069_406945

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ m : ℝ, m ≥ 0 → 4^m ≥ 4*m)) ↔ (∃ m : ℝ, m ≥ 0 ∧ 4^m < 4*m) :=
by sorry

end negation_of_proposition_l4069_406945


namespace existence_of_c_l4069_406900

theorem existence_of_c (n : ℕ) (a b : Fin n → ℝ) 
  (h_n : n ≥ 2)
  (h_pos : ∀ i, a i > 0 ∧ b i > 0)
  (h_less : ∀ i, a i < b i)
  (h_sum : (Finset.sum Finset.univ b) < 1 + (Finset.sum Finset.univ a)) :
  ∃ c : ℝ, ∀ (i : Fin n) (k : ℤ), (a i + c + k) * (b i + c + k) > 0 := by
  sorry

end existence_of_c_l4069_406900


namespace unwashed_shirts_l4069_406930

theorem unwashed_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 21 → washed = 29 → 
  short_sleeve + long_sleeve - washed = 1 := by
sorry

end unwashed_shirts_l4069_406930


namespace power_of_two_plus_three_l4069_406932

/-- Definition of the sequences a_i and b_i -/
def sequence_step (a b : ℤ) : ℤ × ℤ :=
  if a < b then (2*a + 1, b - a - 1)
  else if a > b then (a - b - 1, 2*b + 1)
  else (a, b)

/-- Theorem statement -/
theorem power_of_two_plus_three (n : ℕ) :
  (∃ k : ℕ, ∃ a b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = n ∧
    (∀ i : ℕ, i > 0 → (a i, b i) = sequence_step (a (i-1)) (b (i-1))) ∧
    a k = b k) →
  ∃ m : ℕ, n + 3 = 2^m := by
  sorry

end power_of_two_plus_three_l4069_406932


namespace profit_percentage_l4069_406936

theorem profit_percentage (cost_price selling_price : ℝ) 
  (h : cost_price = 0.96 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.96 - 1) * 100 := by
  sorry

end profit_percentage_l4069_406936


namespace robert_ate_more_than_nickel_l4069_406921

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 7

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_than_nickel : chocolate_difference = 2 := by
  sorry

end robert_ate_more_than_nickel_l4069_406921


namespace inequality_proof_l4069_406998

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (a * b^n)^(1/(n+1 : ℝ)) < (a + n * b) / (n + 1 : ℝ) := by
  sorry

end inequality_proof_l4069_406998


namespace factorization_ab_minus_a_l4069_406974

theorem factorization_ab_minus_a (a b : ℝ) : a * b - a = a * (b - 1) := by
  sorry

end factorization_ab_minus_a_l4069_406974


namespace no_double_application_function_l4069_406980

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 := by
  sorry

end no_double_application_function_l4069_406980


namespace complex_number_imaginary_part_l4069_406960

theorem complex_number_imaginary_part (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (z.im = 2) → a = 3 := by
  sorry

end complex_number_imaginary_part_l4069_406960


namespace work_rate_proof_l4069_406991

/-- The work rate of person A per day -/
def work_rate_A : ℚ := 1 / 4

/-- The work rate of person B per day -/
def work_rate_B : ℚ := 1 / 2

/-- The work rate of person C per day -/
def work_rate_C : ℚ := 1 / 8

/-- The combined work rate of A, B, and C per day -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

theorem work_rate_proof : combined_work_rate = 7 / 8 := by
  sorry

end work_rate_proof_l4069_406991


namespace money_bounds_l4069_406979

theorem money_bounds (a b : ℝ) 
  (h1 : 4 * a + b < 60) 
  (h2 : 6 * a - b = 30) : 
  a < 9 ∧ b < 24 := by
  sorry

end money_bounds_l4069_406979


namespace triangle_angle_value_l4069_406975

/-- Theorem: In a triangle with angles 40°, 3x, and x, the value of x is 35°. -/
theorem triangle_angle_value (x : ℝ) : 
  40 + 3 * x + x = 180 → x = 35 := by
  sorry

end triangle_angle_value_l4069_406975


namespace A_intersect_B_l4069_406950

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | x < 3}

theorem A_intersect_B : A ∩ B = {0, 1, 2} := by
  sorry

end A_intersect_B_l4069_406950


namespace dictionary_chunk_pages_l4069_406902

def is_permutation (a b : ℕ) : Prop := sorry

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem dictionary_chunk_pages (first_page last_page : ℕ) :
  first_page = 213 →
  is_permutation first_page last_page →
  is_even last_page →
  ∀ p, is_permutation first_page p ∧ is_even p → p ≤ last_page →
  last_page - first_page + 1 = 100 :=
by sorry

end dictionary_chunk_pages_l4069_406902


namespace min_value_quadratic_form_l4069_406937

theorem min_value_quadratic_form (x y : ℝ) :
  x^2 - x*y + 4*y^2 ≥ 0 ∧ (x^2 - x*y + 4*y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end min_value_quadratic_form_l4069_406937


namespace gargamel_tire_savings_l4069_406964

/-- The total amount saved when buying tires on sale -/
def total_savings (num_tires : ℕ) (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) * num_tires

/-- Proof that Gargamel saved $36 on his tire purchase -/
theorem gargamel_tire_savings :
  let num_tires : ℕ := 4
  let original_price : ℚ := 84
  let sale_price : ℚ := 75
  total_savings num_tires original_price sale_price = 36 := by
  sorry

end gargamel_tire_savings_l4069_406964


namespace simplify_expression_l4069_406990

theorem simplify_expression (x : ℝ) :
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = -9*x + 2 := by
  sorry

end simplify_expression_l4069_406990


namespace part1_part2_l4069_406949

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the specific conditions of our triangle
def ourTriangle (t : Triangle) : Prop :=
  isValidTriangle t ∧
  t.B = Real.pi / 3 ∧  -- 60 degrees
  t.c = 8

-- Define the midpoint condition
def isMidpoint (M : ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  M.1 = (B.1 + C.1) / 2 ∧ M.2 = (B.2 + C.2) / 2

-- Theorem for part 1
theorem part1 (t : Triangle) (M : ℝ × ℝ) (B C : ℝ × ℝ) :
  ourTriangle t →
  isMidpoint M B C →
  (Real.sqrt 3) * (Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2)) = 
    Real.sqrt ((t.A - M.1)^2 + (t.A - M.2)^2) →
  t.b = 8 :=
sorry

-- Theorem for part 2
theorem part2 (t : Triangle) :
  ourTriangle t →
  t.b = 12 →
  (1/2) * t.b * t.c * Real.sin t.A = 24 * Real.sqrt 2 + 8 * Real.sqrt 3 :=
sorry

end part1_part2_l4069_406949


namespace sum_of_seven_place_values_l4069_406923

/-- Given the number 87953.0727, this theorem states that the sum of the place values
    of the three 7's in this number is equal to 7,000.0707. -/
theorem sum_of_seven_place_values (n : ℝ) (h : n = 87953.0727) :
  7000 + 0.07 + 0.0007 = 7000.0707 := by
  sorry

end sum_of_seven_place_values_l4069_406923


namespace geometric_sequence_monotonicity_l4069_406917

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The first three terms of a sequence are strictly increasing -/
def FirstThreeIncreasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_monotonicity
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  FirstThreeIncreasing a ↔ MonotonicallyIncreasing a :=
by sorry

end geometric_sequence_monotonicity_l4069_406917


namespace three_A_plus_six_B_m_value_when_independent_l4069_406915

-- Define A and B as functions of x and m
def A (x m : ℝ) : ℝ := 2*x^2 + 3*m*x - 2*x - 1
def B (x m : ℝ) : ℝ := -x^2 + m*x - 1

-- Theorem 1: 3A + 6B = (15m-6)x - 9
theorem three_A_plus_six_B (x m : ℝ) : 
  3 * A x m + 6 * B x m = (15*m - 6)*x - 9 := by sorry

-- Theorem 2: When 3A + 6B is independent of x, m = 2/5
theorem m_value_when_independent (m : ℝ) :
  (∀ x : ℝ, 3 * A x m + 6 * B x m = (15*m - 6)*x - 9) →
  (∀ x y : ℝ, 3 * A x m + 6 * B x m = 3 * A y m + 6 * B y m) →
  m = 2/5 := by sorry

end three_A_plus_six_B_m_value_when_independent_l4069_406915


namespace negative_sqrt_two_less_than_negative_one_l4069_406967

theorem negative_sqrt_two_less_than_negative_one : -Real.sqrt 2 < -1 := by
  sorry

end negative_sqrt_two_less_than_negative_one_l4069_406967


namespace negation_equivalence_l4069_406972

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := x + a * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + y + 2 = 0

-- Define when two lines are parallel
def parallel (a : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a x y

-- State the theorem
theorem negation_equivalence :
  ¬(((a = 1) ∨ (a = -1)) → parallel a) ↔ 
  ((a ≠ 1) ∧ (a ≠ -1)) → ¬(parallel a) :=
sorry

end negation_equivalence_l4069_406972


namespace graph_connectivity_probability_l4069_406941

def num_vertices : Nat := 20
def num_edges_removed : Nat := 35

theorem graph_connectivity_probability :
  let total_edges := num_vertices * (num_vertices - 1) / 2
  let remaining_edges := total_edges - num_edges_removed
  let prob_connected := 1 - (num_vertices * Nat.choose remaining_edges (remaining_edges - num_vertices + 1)) / Nat.choose total_edges num_edges_removed
  prob_connected = 1 - (20 * Nat.choose 171 16) / Nat.choose 190 35 := by
  sorry

end graph_connectivity_probability_l4069_406941


namespace expand_expression_l4069_406989

theorem expand_expression (x y : ℝ) : 25 * (3 * x + 6 - 4 * y) = 75 * x + 150 - 100 * y := by
  sorry

end expand_expression_l4069_406989


namespace pens_given_to_sharon_proof_l4069_406910

def initial_pens : ℕ := 7
def mikes_pens : ℕ := 22
def final_pens : ℕ := 39

def pens_given_to_sharon : ℕ := 19

theorem pens_given_to_sharon_proof :
  ((initial_pens + mikes_pens) * 2) - final_pens = pens_given_to_sharon := by
  sorry

end pens_given_to_sharon_proof_l4069_406910


namespace negation_of_existence_negation_of_proposition_l4069_406929

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 2*x) ↔ (∀ x : ℝ, x^2 + 1 ≥ 2*x) := by sorry

end negation_of_existence_negation_of_proposition_l4069_406929


namespace tan_equation_solution_exists_l4069_406908

open Real

theorem tan_equation_solution_exists :
  ∃! θ : ℝ, 0 < θ ∧ θ < π/6 ∧
  tan θ + tan (θ + π/6) + tan (3*θ) = 0 ∧
  0 < tan θ ∧ tan θ < 1 := by
sorry

end tan_equation_solution_exists_l4069_406908


namespace percentage_problem_l4069_406985

theorem percentage_problem (x : ℝ) : 
  (20 / 100 * 40) + (x / 100 * 60) = 23 ↔ x = 25 := by
  sorry

end percentage_problem_l4069_406985


namespace two_numbers_divisible_by_three_l4069_406970

def numbers : List Nat := [222, 2222, 22222, 222222]

theorem two_numbers_divisible_by_three : 
  (numbers.filter (fun n => n % 3 = 0)).length = 2 := by sorry

end two_numbers_divisible_by_three_l4069_406970


namespace min_value_fraction_l4069_406977

theorem min_value_fraction (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≥ 4/3 := by
  sorry

end min_value_fraction_l4069_406977


namespace birthday_cookies_l4069_406912

theorem birthday_cookies (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) :
  friends = 4 →
  packages = 3 →
  cookies_per_package = 25 →
  (packages * cookies_per_package) / (friends + 1) = 15 :=
by sorry

end birthday_cookies_l4069_406912


namespace two_digit_number_sum_l4069_406993

theorem two_digit_number_sum (n : ℕ) : 
  (10 ≤ n ∧ n < 100) →  -- n is a two-digit number
  (n / 2 : ℚ) = (n / 4 : ℚ) + 3 →  -- one half of n exceeds its one fourth by 3
  (n / 10 + n % 10 = 3) :=  -- sum of digits is 3
by
  sorry

end two_digit_number_sum_l4069_406993


namespace parking_cost_theorem_l4069_406994

/-- The number of hours for the initial parking cost -/
def initial_hours : ℝ := 2

/-- The initial parking cost -/
def initial_cost : ℝ := 9

/-- The cost per hour for excess hours -/
def excess_cost_per_hour : ℝ := 1.75

/-- The total number of hours parked -/
def total_hours : ℝ := 9

/-- The average cost per hour for the total parking time -/
def average_cost_per_hour : ℝ := 2.361111111111111

theorem parking_cost_theorem :
  initial_hours = 2 ∧
  initial_cost + excess_cost_per_hour * (total_hours - initial_hours) =
    average_cost_per_hour * total_hours :=
by sorry

end parking_cost_theorem_l4069_406994


namespace new_ratio_is_one_to_two_l4069_406901

/-- Represents the coin collection --/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- The initial state of the coin collection --/
def initial_collection : CoinCollection :=
  { gold := 0, silver := 0 }

/-- The condition that initially there is one gold coin for every 3 silver coins --/
axiom initial_ratio (c : CoinCollection) : c.gold * 3 = c.silver

/-- The operation of adding 15 gold coins to the collection --/
def add_gold_coins (c : CoinCollection) : CoinCollection :=
  { gold := c.gold + 15, silver := c.silver }

/-- The total number of coins after adding 15 gold coins is 135 --/
axiom total_coins_after_addition (c : CoinCollection) :
  (add_gold_coins c).gold + (add_gold_coins c).silver = 135

/-- Theorem stating that the new ratio of gold to silver coins is 1:2 --/
theorem new_ratio_is_one_to_two (c : CoinCollection) :
  2 * (add_gold_coins c).gold = (add_gold_coins c).silver :=
sorry

end new_ratio_is_one_to_two_l4069_406901


namespace not_in_E_iff_perfect_square_l4069_406966

/-- The set E of floor values of n + √n + 1/2 for natural numbers n -/
def E : Set ℕ := {m | ∃ n : ℕ, m = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋}

/-- A positive integer m is not in set E if and only if it's a perfect square -/
theorem not_in_E_iff_perfect_square (m : ℕ) (hm : m > 0) : 
  m ∉ E ↔ ∃ k : ℕ, m = k^2 := by sorry

end not_in_E_iff_perfect_square_l4069_406966


namespace stamp_arrangement_exists_l4069_406951

/-- Represents the quantity of each stamp denomination -/
def stamp_quantities : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- Represents the value of each stamp denomination -/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- A function to calculate the number of unique stamp arrangements -/
def count_stamp_arrangements (quantities : List Nat) (values : List Nat) (target : Nat) : Nat :=
  sorry

/-- Theorem stating that there exists a positive number of unique arrangements -/
theorem stamp_arrangement_exists :
  ∃ n : Nat, n > 0 ∧ count_stamp_arrangements stamp_quantities stamp_values 15 = n :=
sorry

end stamp_arrangement_exists_l4069_406951


namespace charging_piles_growth_equation_l4069_406957

/-- Given the number of charging piles built in the first and third months, 
    and the monthly average growth rate, this theorem states the equation 
    that relates these quantities. -/
theorem charging_piles_growth_equation 
  (initial_piles : ℕ) 
  (final_piles : ℕ) 
  (x : ℝ) 
  (h1 : initial_piles = 301)
  (h2 : final_piles = 500)
  (h3 : x ≥ 0) -- Assuming non-negative growth rate
  (h4 : x ≤ 1) -- Assuming growth rate is at most 100%
  : initial_piles * (1 + x)^2 = final_piles := by
  sorry

end charging_piles_growth_equation_l4069_406957


namespace solution_of_equation_l4069_406928

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -3^x else 1 - x^2

-- State the theorem
theorem solution_of_equation (x : ℝ) :
  f x = -3 ↔ x = 1 ∨ x = -2 :=
sorry

end solution_of_equation_l4069_406928


namespace expression_simplification_l4069_406958

theorem expression_simplification (a b : ℚ) (ha : a = -1) (hb : b = 1/2) :
  2 * a^2 * b - (3 * a * b^2 - (4 * a * b^2 - 2 * a^2 * b)) = 1/2 := by
  sorry

end expression_simplification_l4069_406958


namespace exists_k_all_multiples_contain_all_digits_l4069_406982

/-- For a given positive integer, check if its decimal representation contains all digits from 0 to 9 -/
def containsAllDigits (n : ℕ+) : Prop := sorry

/-- For a given positive integer k and a set of positive integers, check if k*i contains all digits for all i in the set -/
def allMultiplesContainAllDigits (k : ℕ+) (s : Set ℕ+) : Prop :=
  ∀ i ∈ s, containsAllDigits (i * k)

/-- Main theorem: For all positive integers n, there exists a positive integer k such that
    k, 2k, ..., nk all contain all digits from 0 to 9 in their decimal representations -/
theorem exists_k_all_multiples_contain_all_digits (n : ℕ+) :
  ∃ k : ℕ+, allMultiplesContainAllDigits k (Set.Icc 1 n) := by sorry

end exists_k_all_multiples_contain_all_digits_l4069_406982


namespace inequality_solution_l4069_406968

theorem inequality_solution : 
  {x : ℕ | x > 0 ∧ 3 * x - 4 < 2 * x} = {1, 2, 3} := by sorry

end inequality_solution_l4069_406968


namespace expression_value_l4069_406953

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)    -- absolute value of m is 3
  : (a + b) / m + m^2 - c * d = 8 := by
  sorry

end expression_value_l4069_406953


namespace lipstick_cost_l4069_406984

/-- Calculates the cost of each lipstick given the order details -/
theorem lipstick_cost (total_items : ℕ) (num_slippers : ℕ) (slipper_price : ℚ)
  (num_lipsticks : ℕ) (num_hair_colors : ℕ) (hair_color_price : ℚ) (total_paid : ℚ) :
  total_items = num_slippers + num_lipsticks + num_hair_colors →
  total_items = 18 →
  num_slippers = 6 →
  slipper_price = 5/2 →
  num_lipsticks = 4 →
  num_hair_colors = 8 →
  hair_color_price = 3 →
  total_paid = 44 →
  (total_paid - (num_slippers * slipper_price + num_hair_colors * hair_color_price)) / num_lipsticks = 5/4 :=
by sorry

end lipstick_cost_l4069_406984


namespace tenth_term_is_three_point_five_l4069_406942

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- The first term of our sequence -/
def a₁ : ℚ := 1/2

/-- The second term of our sequence -/
def a₂ : ℚ := 5/6

/-- The common difference of our sequence -/
def d : ℚ := a₂ - a₁

theorem tenth_term_is_three_point_five :
  arithmetic_sequence a₁ d 10 = 7/2 := by sorry

end tenth_term_is_three_point_five_l4069_406942


namespace rectangle_area_diagonal_l4069_406973

/-- Given a rectangle with length-to-width ratio of 5:2 and diagonal d, 
    prove that its area A can be expressed as A = kd^2, where k = 10/29 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end rectangle_area_diagonal_l4069_406973


namespace translate_f_to_g_l4069_406931

def f (x : ℝ) : ℝ := 2 * x^2

def g (x : ℝ) : ℝ := 2 * (x + 1)^2 + 3

theorem translate_f_to_g : 
  ∀ x : ℝ, g x = f (x + 1) + 3 := by sorry

end translate_f_to_g_l4069_406931


namespace quadratic_roots_l4069_406920

def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*a*x + c

theorem quadratic_roots (a c : ℝ) (h : a ≠ 0) :
  (quadratic_function a c (-1) = 0) →
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧
    ∀ x : ℝ, quadratic_function a c x = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by
  sorry

end quadratic_roots_l4069_406920


namespace swallowing_not_complete_disappearance_l4069_406948

/-- Represents a snake in the swallowing process --/
structure Snake where
  length : ℝ
  swallowed : ℝ

/-- Represents the state of two snakes swallowing each other --/
structure SwallowingState where
  snake1 : Snake
  snake2 : Snake
  ring_size : ℝ

/-- The swallowing process between two snakes --/
def swallowing_process (initial_state : SwallowingState) : Prop :=
  ∀ t : ℝ, t ≥ 0 →
    ∃ state : SwallowingState,
      state.ring_size < initial_state.ring_size ∧
      state.snake1.swallowed > initial_state.snake1.swallowed ∧
      state.snake2.swallowed > initial_state.snake2.swallowed ∧
      state.snake1.length + state.snake2.length > 0

/-- Theorem stating that the swallowing process does not result in complete disappearance --/
theorem swallowing_not_complete_disappearance (initial_state : SwallowingState) :
  swallowing_process initial_state →
  ∃ final_state : SwallowingState, final_state.snake1.length + final_state.snake2.length > 0 :=
by sorry

end swallowing_not_complete_disappearance_l4069_406948


namespace eleven_power_2023_mod_5_l4069_406978

theorem eleven_power_2023_mod_5 : 11^2023 % 5 = 1 := by sorry

end eleven_power_2023_mod_5_l4069_406978


namespace unique_x_with_rational_sums_l4069_406905

theorem unique_x_with_rational_sums (x : ℝ) :
  (∃ a : ℚ, x + Real.sqrt 3 = a) ∧ 
  (∃ b : ℚ, x^2 + Real.sqrt 3 = b) →
  x = 1/2 - Real.sqrt 3 := by
sorry

end unique_x_with_rational_sums_l4069_406905


namespace line_perp_parallel_implies_planes_perp_l4069_406925

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (a : Line) (M N : Plane)
  (h1 : perpendicular a M)
  (h2 : parallel a N) :
  perp_planes M N :=
sorry

end line_perp_parallel_implies_planes_perp_l4069_406925


namespace imaginary_part_of_complex_fraction_l4069_406959

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.im ((1 - i) / ((1 + i)^2)) = 1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l4069_406959


namespace consecutive_integers_sum_l4069_406938

theorem consecutive_integers_sum (n : ℚ) : 
  (n - 1) + (n + 1) + (n + 2) = 175 → n = 57 + 2/3 := by
  sorry

end consecutive_integers_sum_l4069_406938


namespace line_l_equation_circle_M_equations_l4069_406988

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define perpendicularity of lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define the equation of line l
def l (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the equations of circle M
def M₁ (x y : ℝ) : Prop := (x + 5/7)^2 + (y + 10/7)^2 = 25/49
def M₂ (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∀ x y : ℝ, l x y ↔ (∃ m : ℝ, perpendicular m 2 ∧ y - P.2 = m * (x - P.1)) :=
sorry

-- Theorem for the equations of circle M
theorem circle_M_equations :
  ∀ x y : ℝ, 
    (∃ a b r : ℝ, 
      l₁ a b ∧ 
      (∀ t : ℝ, (t - a)^2 + b^2 = r^2 → t = 0) ∧ 
      ((a + b + 2)^2 / 2 + 1/2 = r^2) ∧
      ((x - a)^2 + (y - b)^2 = r^2)) 
    ↔ (M₁ x y ∨ M₂ x y) :=
sorry

end line_l_equation_circle_M_equations_l4069_406988


namespace fraction_simplification_l4069_406961

theorem fraction_simplification : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end fraction_simplification_l4069_406961


namespace largest_inscribed_circle_and_dot_product_range_l4069_406927

-- Define the plane region
def plane_region (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 4 ≥ 0 ∧ 
  x + Real.sqrt 3 * y + 4 ≥ 0 ∧ 
  x ≤ 2

-- Define the largest inscribed circle
def largest_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the geometric sequence condition
def geometric_sequence (pa pm pb : ℝ) : Prop :=
  ∃ r : ℝ, (pa = pm * r ∧ pm = pb * r) ∨ (pm = pa * r ∧ pb = pm * r)

-- Define the dot product of PA and PB
def dot_product_pa_pb (px py : ℝ) : ℝ :=
  (px + 2) * (px - 2) + py * (-py)

-- The main theorem
theorem largest_inscribed_circle_and_dot_product_range :
  (∀ x y : ℝ, plane_region x y → largest_circle x y) ∧
  (∀ px py : ℝ, largest_circle px py →
    (∀ pa pm pb : ℝ, geometric_sequence pa pm pb →
      -2 ≤ dot_product_pa_pb px py ∧ dot_product_pa_pb px py < 0)) := by
  sorry

end largest_inscribed_circle_and_dot_product_range_l4069_406927


namespace salary_ratio_degree_to_diploma_l4069_406934

/-- Represents the monthly salary of a diploma holder in dollars. -/
def diploma_monthly_salary : ℕ := 4000

/-- Represents the annual salary of a degree holder in dollars. -/
def degree_annual_salary : ℕ := 144000

/-- Represents the number of months in a year. -/
def months_per_year : ℕ := 12

/-- Theorem stating that the ratio of annual salaries between degree and diploma holders is 3:1. -/
theorem salary_ratio_degree_to_diploma :
  (degree_annual_salary : ℚ) / (diploma_monthly_salary * months_per_year) = 3 := by
  sorry

#check salary_ratio_degree_to_diploma

end salary_ratio_degree_to_diploma_l4069_406934


namespace jar_to_pot_ratio_l4069_406992

/-- Proves that the ratio of jars to clay pots is 2:1 given the problem conditions --/
theorem jar_to_pot_ratio :
  ∀ (num_pots : ℕ),
  (∃ (k : ℕ), 16 = k * num_pots) →
  16 * 5 + num_pots * (5 * 3) = 200 →
  (16 : ℚ) / num_pots = 2 := by
  sorry

end jar_to_pot_ratio_l4069_406992


namespace third_smallest_four_digit_in_pascal_l4069_406935

/-- Pascal's triangle coefficient -/
def pascal (n k : ℕ) : ℕ := sorry

/-- The n-th row of Pascal's triangle -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Predicate for a number being four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The third smallest four-digit number in Pascal's triangle -/
theorem third_smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), pascal n k = 1002 ∧ 
  (∀ (m l : ℕ), pascal m l < 1002 → ¬(is_four_digit (pascal m l))) ∧
  (∃! (p q r s : ℕ), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    is_four_digit (pascal p s) ∧ 
    is_four_digit (pascal q s) ∧
    pascal p s < pascal q s ∧
    pascal q s < 1002) := by sorry

end third_smallest_four_digit_in_pascal_l4069_406935


namespace inequality_solution_l4069_406946

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a * x - 20) * Real.log (2 * a / x) ≤ 0) ↔ a = Real.sqrt 10 := by
  sorry

end inequality_solution_l4069_406946


namespace divisor_of_a_l4069_406995

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 18)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 60)
  (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) :
  5 ∣ a := by
  sorry

end divisor_of_a_l4069_406995


namespace pepperoni_pizza_coverage_l4069_406926

theorem pepperoni_pizza_coverage (pizza_diameter : ℝ) (pepperoni_count : ℕ) 
  (pepperoni_across : ℕ) : 
  pizza_diameter = 12 →
  pepperoni_across = 8 →
  pepperoni_count = 32 →
  (pepperoni_count * (pizza_diameter / pepperoni_across / 2)^2) / 
  (pizza_diameter / 2)^2 = 1 / 2 := by
  sorry

#check pepperoni_pizza_coverage

end pepperoni_pizza_coverage_l4069_406926


namespace problem_solution_l4069_406933

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x/y + y/x = 8) :
  (x + y)/(x - y) = Real.sqrt 15 / 3 := by
sorry

end problem_solution_l4069_406933


namespace probability_red_black_heart_value_l4069_406952

/-- The probability of drawing a red card first, then a black card, and then a red heart
    from a deck of 104 cards with 52 red cards (of which 26 are hearts) and 52 black cards. -/
def probability_red_black_heart (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ) (heart_cards : ℕ) : ℚ :=
  (red_cards : ℚ) / total_cards *
  (black_cards : ℚ) / (total_cards - 1) *
  (heart_cards - 1 : ℚ) / (total_cards - 2)

/-- The probability of drawing a red card first, then a black card, and then a red heart
    from a deck of 104 cards with 52 red cards (of which 26 are hearts) and 52 black cards
    is equal to 25/3978. -/
theorem probability_red_black_heart_value :
  probability_red_black_heart 104 52 52 26 = 25 / 3978 :=
by
  sorry

#eval probability_red_black_heart 104 52 52 26

end probability_red_black_heart_value_l4069_406952


namespace triangle_properties_l4069_406940

/-- Triangle ABC with side lengths a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.sin t.B - Real.cos t.B = 2 * Real.sin (t.B - π / 6))
  (h2 : t.b = 1)
  (h3 : t.A = 5 * π / 12) :
  (t.c = Real.sqrt 6 / 3) ∧ 
  (∀ h : ℝ, h ≤ Real.sqrt 3 / 2 → 
    ∃ (a c : ℝ), 
      a > 0 ∧ c > 0 ∧ 
      a * c ≤ 1 ∧ 
      h = Real.sqrt 3 / 2 * a * c) := by
  sorry

end triangle_properties_l4069_406940


namespace production_quantities_max_type_A_for_school_l4069_406987

-- Define the parameters
def total_production : ℕ := 400000
def cost_A : ℚ := 1.2
def cost_B : ℚ := 0.4
def price_A : ℚ := 1.6
def price_B : ℚ := 0.6
def profit : ℚ := 110000
def school_budget : ℚ := 7680
def discount_A : ℚ := 0.1
def school_purchase : ℕ := 10000

-- Part 1: Production quantities
theorem production_quantities :
  ∃ (x y : ℕ),
    x + y = total_production ∧
    (price_A - cost_A) * x + (price_B - cost_B) * y = profit ∧
    x = 15000 ∧
    y = 25000 :=
sorry

-- Part 2: Maximum type A books for school
theorem max_type_A_for_school :
  ∃ (m : ℕ),
    m ≤ school_purchase ∧
    price_A * (1 - discount_A) * m + price_B * (school_purchase - m) ≤ school_budget ∧
    m = 2000 ∧
    ∀ n, n > m → 
      price_A * (1 - discount_A) * n + price_B * (school_purchase - n) > school_budget :=
sorry

end production_quantities_max_type_A_for_school_l4069_406987


namespace sparcs_characterization_l4069_406955

-- Define "grows to"
def grows_to (s r : ℝ) : Prop :=
  ∃ n : ℕ+, s ^ (n : ℝ) = r

-- Define "sparcs"
def sparcs (r : ℝ) : Prop :=
  {s : ℝ | grows_to s r}.Finite

-- Theorem statement
theorem sparcs_characterization (r : ℝ) :
  sparcs r ↔ r = -1 ∨ r = 0 ∨ r = 1 := by
  sorry

end sparcs_characterization_l4069_406955


namespace composite_product_ratio_l4069_406939

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]

def product (l : List Nat) : Nat := l.foldl (· * ·) 1

theorem composite_product_ratio :
  (product first_six_composites : ℚ) / (product next_six_composites) = 1 / 49 := by
  sorry

end composite_product_ratio_l4069_406939


namespace least_positive_angle_theorem_l4069_406914

/-- The least positive angle θ (in degrees) satisfying sin 15° = cos 40° + cos θ is 115° -/
theorem least_positive_angle_theorem : 
  let θ : ℝ := 115
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → 
    Real.sin (15 * π / 180) ≠ Real.cos (40 * π / 180) + Real.cos (φ * π / 180) ∧
    Real.sin (15 * π / 180) = Real.cos (40 * π / 180) + Real.cos (θ * π / 180) :=
by sorry

end least_positive_angle_theorem_l4069_406914


namespace six_digit_difference_l4069_406963

/-- Function f for 6-digit numbers -/
def f (n : ℕ) : ℕ :=
  let u := n / 100000 % 10
  let v := n / 10000 % 10
  let w := n / 1000 % 10
  let x := n / 100 % 10
  let y := n / 10 % 10
  let z := n % 10
  2^u * 3^v * 5^w * 7^x * 11^y * 13^z

/-- Theorem: If f(abcdef) = 13 * f(ghijkl), then abcdef - ghijkl = 1 -/
theorem six_digit_difference (abcdef ghijkl : ℕ) 
  (h1 : 100000 ≤ abcdef ∧ abcdef < 1000000)
  (h2 : 100000 ≤ ghijkl ∧ ghijkl < 1000000)
  (h3 : f abcdef = 13 * f ghijkl) : 
  abcdef - ghijkl = 1 := by
  sorry

end six_digit_difference_l4069_406963


namespace yard_sale_problem_l4069_406918

theorem yard_sale_problem (total_items video_games dvds books working_video_games working_dvds : ℕ) 
  (h1 : total_items = 56)
  (h2 : video_games = 30)
  (h3 : dvds = 15)
  (h4 : books = total_items - video_games - dvds)
  (h5 : working_video_games = 20)
  (h6 : working_dvds = 10) :
  (video_games - working_video_games) + (dvds - working_dvds) = 15 := by
  sorry

end yard_sale_problem_l4069_406918


namespace probability_not_hearing_favorite_in_6_minutes_l4069_406965

/-- Represents a playlist of songs with increasing durations -/
structure Playlist where
  num_songs : ℕ
  duration_increment : ℕ
  shortest_duration : ℕ
  favorite_duration : ℕ

/-- Calculates the probability of not hearing the entire favorite song 
    within a given time limit -/
def probability_not_hearing_favorite (p : Playlist) (time_limit : ℕ) : ℚ :=
  sorry

/-- The specific playlist described in the problem -/
def marcel_playlist : Playlist :=
  { num_songs := 12
  , duration_increment := 30
  , shortest_duration := 60
  , favorite_duration := 300 }

/-- The main theorem to prove -/
theorem probability_not_hearing_favorite_in_6_minutes :
  probability_not_hearing_favorite marcel_playlist 360 = 1813 / 1980 := by
  sorry

end probability_not_hearing_favorite_in_6_minutes_l4069_406965


namespace battery_life_comparison_l4069_406913

-- Define the battery characteristics
def tablet_standby : ℚ := 18
def tablet_continuous : ℚ := 6
def smartphone_standby : ℚ := 30
def smartphone_continuous : ℚ := 4

-- Define the usage
def tablet_total_time : ℚ := 14
def tablet_usage_time : ℚ := 2
def smartphone_total_time : ℚ := 20
def smartphone_usage_time : ℚ := 3

-- Define the battery consumption rates
def tablet_standby_rate : ℚ := 1 / tablet_standby
def tablet_usage_rate : ℚ := 1 / tablet_continuous
def smartphone_standby_rate : ℚ := 1 / smartphone_standby
def smartphone_usage_rate : ℚ := 1 / smartphone_continuous

-- Define the theorem
theorem battery_life_comparison : 
  let tablet_battery_used := (tablet_total_time - tablet_usage_time) * tablet_standby_rate + tablet_usage_time * tablet_usage_rate
  let smartphone_battery_used := (smartphone_total_time - smartphone_usage_time) * smartphone_standby_rate + smartphone_usage_time * smartphone_usage_rate
  let smartphone_battery_remaining := 1 - smartphone_battery_used
  tablet_battery_used ≥ 1 ∧ 
  smartphone_battery_remaining / smartphone_standby_rate = 9 :=
by sorry

end battery_life_comparison_l4069_406913


namespace subtracted_number_proof_l4069_406962

theorem subtracted_number_proof (initial_number : ℝ) (subtracted_number : ℝ) : 
  initial_number = 22.142857142857142 →
  ((initial_number + 5) * 7) / 5 - subtracted_number = 33 →
  subtracted_number = 5 := by
sorry

end subtracted_number_proof_l4069_406962


namespace divide_fractions_l4069_406976

theorem divide_fractions (a b c : ℚ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 4 / 7) : 
  c / a = 21 / 20 := by
sorry

end divide_fractions_l4069_406976


namespace swimmers_second_meeting_time_l4069_406983

theorem swimmers_second_meeting_time
  (pool_length : ℝ)
  (henry_speed : ℝ)
  (george_speed : ℝ)
  (first_meeting_time : ℝ)
  (h1 : pool_length = 100)
  (h2 : george_speed = 2 * henry_speed)
  (h3 : first_meeting_time = 1)
  (h4 : henry_speed * first_meeting_time + george_speed * first_meeting_time = pool_length) :
  let second_meeting_time := 2 * first_meeting_time
  ∃ (distance_henry distance_george : ℝ),
    distance_henry + distance_george = pool_length ∧
    distance_henry = henry_speed * second_meeting_time ∧
    distance_george = george_speed * second_meeting_time :=
by sorry


end swimmers_second_meeting_time_l4069_406983


namespace sin_780_degrees_l4069_406997

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_780_degrees_l4069_406997


namespace problem_solution_l4069_406956

theorem problem_solution (θ : Real) (x : Real) :
  let A := (5 * Real.sin θ + 4 * Real.cos θ) / (3 * Real.sin θ + Real.cos θ)
  let B := x^3 + 1/x^3
  Real.tan θ = 2 →
  x + 1/x = 2 * A →
  A = 2 ∧ B = 52 := by
sorry

end problem_solution_l4069_406956


namespace cake_ratio_theorem_l4069_406954

/-- Proves that the ratio of cakes sold to total cakes baked is 1:2 --/
theorem cake_ratio_theorem (cakes_per_day : ℕ) (days_baked : ℕ) (cakes_left : ℕ) :
  cakes_per_day = 20 →
  days_baked = 9 →
  cakes_left = 90 →
  let total_cakes := cakes_per_day * days_baked
  let cakes_sold := total_cakes - cakes_left
  (cakes_sold : ℚ) / total_cakes = 1 / 2 :=
by
  sorry

#check cake_ratio_theorem

end cake_ratio_theorem_l4069_406954


namespace sum_of_squares_l4069_406971

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end sum_of_squares_l4069_406971


namespace triangle_side_function_is_identity_l4069_406969

/-- A function satisfying the triangle side and perimeter conditions -/
def TriangleSideFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → f x > 0) ∧ 
  (∀ x y z, x > 0 → y > 0 → z > 0 →
    (x + f y > f (f z) + f x ∧ 
     f (f y) + z > x + f y ∧
     f (f z) + f x > f (f y) + z)) ∧
  (∀ p, p > 0 → ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + f y + f (f y) + z + f (f z) + f x = p)

/-- The main theorem stating that the identity function is the only function
    satisfying the triangle side and perimeter conditions -/
theorem triangle_side_function_is_identity 
  (f : ℝ → ℝ) (hf : TriangleSideFunction f) : 
  ∀ x, x > 0 → f x = x :=
sorry

end triangle_side_function_is_identity_l4069_406969


namespace exists_n_where_B_less_than_A_largest_n_where_B_leq_A_l4069_406907

-- Define A(n) for Alphonse's jumps
def A (n : ℕ) : ℕ :=
  n / 8 + n % 8

-- Define B(n) for Beryl's jumps
def B (n : ℕ) : ℕ :=
  n / 7 + n % 7

-- Part (a)
theorem exists_n_where_B_less_than_A :
  ∃ n : ℕ, n > 200 ∧ B n < A n :=
sorry

-- Part (b)
theorem largest_n_where_B_leq_A :
  ∀ n : ℕ, B n ≤ A n → n ≤ 343 :=
sorry

end exists_n_where_B_less_than_A_largest_n_where_B_leq_A_l4069_406907


namespace half_sum_of_consecutive_odd_primes_is_composite_l4069_406903

/-- Two natural numbers are consecutive primes if they are both prime and there are no primes between them. -/
def ConsecutivePrimes (p p' : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime p' ∧ p < p' ∧ ∀ q, p < q → q < p' → ¬Nat.Prime q

/-- A natural number is composite if it's greater than 1 and not prime. -/
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬Nat.Prime n

theorem half_sum_of_consecutive_odd_primes_is_composite
  (p p' : ℕ) (h : ConsecutivePrimes p p') (hp_odd : Odd p) (hp'_odd : Odd p') (hp_ge_3 : p ≥ 3) :
  Composite ((p + p') / 2) :=
sorry

end half_sum_of_consecutive_odd_primes_is_composite_l4069_406903


namespace sqrt_calculations_l4069_406906

theorem sqrt_calculations : 
  (2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2) ∧ 
  ((Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2) := by
  sorry

end sqrt_calculations_l4069_406906


namespace one_percent_of_x_l4069_406996

theorem one_percent_of_x (x : ℝ) (h : (89 / 100) * 19 = (19 / 100) * x) : 
  (1 / 100) * x = 89 / 100 := by
sorry

end one_percent_of_x_l4069_406996


namespace sum_of_powers_l4069_406919

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end sum_of_powers_l4069_406919


namespace max_value_sqrt_sum_l4069_406924

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_cond : x + y + z = 2)
  (x_cond : x ≥ -1)
  (y_cond : y ≥ -3/2)
  (z_cond : z ≥ -2) :
  ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ + y₀ + z₀ = 2 ∧ 
    x₀ ≥ -1 ∧ 
    y₀ ≥ -3/2 ∧ 
    z₀ ≥ -2 ∧
    Real.sqrt (4 * x₀ + 2) + Real.sqrt (4 * y₀ + 6) + Real.sqrt (4 * z₀ + 8) = 2 * Real.sqrt 30 ∧
    ∀ (x y z : ℝ), 
      x + y + z = 2 → 
      x ≥ -1 → 
      y ≥ -3/2 → 
      z ≥ -2 → 
      Real.sqrt (4 * x + 2) + Real.sqrt (4 * y + 6) + Real.sqrt (4 * z + 8) ≤ 2 * Real.sqrt 30 :=
by
  sorry

end max_value_sqrt_sum_l4069_406924


namespace evaluate_expression_l4069_406904

theorem evaluate_expression (a x : ℝ) (h : x = a + 5) : x - a + 4 = 9 := by
  sorry

end evaluate_expression_l4069_406904


namespace farmer_land_ownership_l4069_406986

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 90) →
  total_land = 1000 := by
  sorry

end farmer_land_ownership_l4069_406986


namespace harpers_rubber_bands_l4069_406981

/-- Harper's rubber band problem -/
theorem harpers_rubber_bands :
  ∀ (h : ℕ),                        -- h represents Harper's number of rubber bands
  (h + (h - 6) = 24) →              -- Total rubber bands condition
  h = 15 := by                      -- Prove that Harper has 15 rubber bands
sorry


end harpers_rubber_bands_l4069_406981


namespace max_sum_of_digits_l4069_406944

theorem max_sum_of_digits (x z : ℕ) : 
  x ≤ 9 → z ≤ 9 → x > z → 99 * (x - z) = 693 → 
  ∃ d : ℕ, d = 11 ∧ ∀ x' z' : ℕ, x' ≤ 9 → z' ≤ 9 → x' > z' → 99 * (x' - z') = 693 → x' + z' ≤ d :=
by sorry

end max_sum_of_digits_l4069_406944


namespace plane_perpendicular_sufficient_not_necessary_l4069_406916

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (in_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_sufficient_not_necessary
  (α β : Plane) (m b c : Line) :
  intersect α β m →
  in_plane b α →
  in_plane c β →
  perpendicular c m →
  (plane_perpendicular α β → perpendicular c b) ∧
  ¬(perpendicular c b → plane_perpendicular α β) :=
sorry

end plane_perpendicular_sufficient_not_necessary_l4069_406916


namespace variance_of_successes_l4069_406943

/-- The number of experiments -/
def n : ℕ := 30

/-- The probability of success in a single experiment -/
def p : ℚ := 5/9

/-- The variance of the number of successes in n independent experiments 
    with probability of success p -/
def variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_successes : variance n p = 200/27 := by
  sorry

end variance_of_successes_l4069_406943


namespace square_difference_l4069_406999

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 9/17) 
  (h2 : x - y = 1/119) : 
  x^2 - y^2 = 9/2003 := by
  sorry

end square_difference_l4069_406999


namespace solution_set_part1_range_of_a_part2_l4069_406911

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Part 1: Solution set for f(x) > 1 when a = -2
theorem solution_set_part1 :
  {x : ℝ | f (-2) x > 1} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

-- Part 2: Range of a when f(x) > 0 for all x ∈ [1, +∞)
theorem range_of_a_part2 :
  (∀ x : ℝ, x ≥ 1 → f a x > 0) ↔ a > -3 := by sorry

end solution_set_part1_range_of_a_part2_l4069_406911


namespace subtraction_example_l4069_406909

theorem subtraction_example : (3.75 : ℝ) - 1.46 = 2.29 := by sorry

end subtraction_example_l4069_406909


namespace min_value_expression_l4069_406947

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + a/(b^2) + b ≥ 2 * Real.sqrt 2 ∧
  (1/a + a/(b^2) + b = 2 * Real.sqrt 2 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2) :=
by sorry

end min_value_expression_l4069_406947
