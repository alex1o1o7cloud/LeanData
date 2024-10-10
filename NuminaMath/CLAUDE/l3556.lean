import Mathlib

namespace identity_condition_l3556_355692

theorem identity_condition (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) → 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) := by
  sorry

end identity_condition_l3556_355692


namespace power_of_two_greater_than_square_l3556_355679

theorem power_of_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end power_of_two_greater_than_square_l3556_355679


namespace product_from_hcf_lcm_l3556_355613

theorem product_from_hcf_lcm (a b : ℕ+) (h_hcf : Nat.gcd a b = 14) (h_lcm : Nat.lcm a b = 183) :
  a * b = 2562 := by
  sorry

end product_from_hcf_lcm_l3556_355613


namespace quaternary_1010_equals_68_l3556_355698

/-- Converts a quaternary (base 4) digit to its decimal value --/
def quaternaryToDecimal (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Calculates the decimal value of a quaternary number represented as a list of digits --/
def quaternaryListToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + quaternaryToDecimal d * (4 ^ (digits.length - 1 - i))) 0

/-- The quaternary representation of the number to be converted --/
def quaternaryNumber : List Nat := [1, 0, 1, 0]

/-- Statement: The quaternary number 1010₍₄₎ is equal to the decimal number 68 --/
theorem quaternary_1010_equals_68 : 
  quaternaryListToDecimal quaternaryNumber = 68 := by
  sorry

end quaternary_1010_equals_68_l3556_355698


namespace polynomial_remainder_theorem_l3556_355651

theorem polynomial_remainder_theorem (x : ℝ) : 
  (4 * x^3 - 8 * x^2 + 11 * x - 5) % (2 * x - 4) = 17 := by
  sorry

end polynomial_remainder_theorem_l3556_355651


namespace min_value_geometric_sequence_l3556_355605

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 7b₃ is -9/14. -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → 
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = 2 * r' ∧ b₃' = b₂' * r') → 
    3 * b₂ + 7 * b₃ ≤ 3 * b₂' + 7 * b₃') → 
  3 * b₂ + 7 * b₃ = -9/14 :=
by sorry

end min_value_geometric_sequence_l3556_355605


namespace fraction_problem_l3556_355682

theorem fraction_problem (x : ℚ) : x = 4/5 ↔ 0.55 * 40 = x * 25 + 2 := by
  sorry

end fraction_problem_l3556_355682


namespace tangent_product_simplification_l3556_355643

theorem tangent_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 :=
by
  sorry

end tangent_product_simplification_l3556_355643


namespace large_box_length_l3556_355635

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: If a large box with dimensions L × 14 × 16 can fit exactly 64 small boxes
    with dimensions 3 × 7 × 2, then L must be 12. -/
theorem large_box_length (L : ℝ) : 
  let largeBox : BoxDimensions := ⟨L, 14, 16⟩
  let smallBox : BoxDimensions := ⟨3, 7, 2⟩
  (boxVolume largeBox) / (boxVolume smallBox) = 64 → L = 12 := by
sorry

end large_box_length_l3556_355635


namespace mabel_handled_90_l3556_355640

/-- The number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (t : Transactions) : Prop :=
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.cal = (2 * t.anthony) / 3 ∧
  t.jade = t.cal + 18 ∧
  t.jade = 84

/-- The theorem stating that Mabel handled 90 transactions -/
theorem mabel_handled_90 :
  ∃ (t : Transactions), satisfiesConditions t ∧ t.mabel = 90 := by
  sorry


end mabel_handled_90_l3556_355640


namespace total_production_all_companies_l3556_355625

/-- Represents the production of cars by a company in different continents -/
structure CarProduction where
  northAmerica : ℕ
  europe : ℕ
  asia : ℕ

/-- Calculates the total production for a company -/
def totalProduction (p : CarProduction) : ℕ :=
  p.northAmerica + p.europe + p.asia

/-- The production data for Car Company A -/
def companyA : CarProduction :=
  { northAmerica := 3884
    europe := 2871
    asia := 1529 }

/-- The production data for Car Company B -/
def companyB : CarProduction :=
  { northAmerica := 4357
    europe := 3690
    asia := 1835 }

/-- The production data for Car Company C -/
def companyC : CarProduction :=
  { northAmerica := 2937
    europe := 4210
    asia := 977 }

/-- Theorem stating that the total production of all companies is 26,290 -/
theorem total_production_all_companies :
  totalProduction companyA + totalProduction companyB + totalProduction companyC = 26290 := by
  sorry

end total_production_all_companies_l3556_355625


namespace sum_of_coefficients_l3556_355623

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x + y)^4 = a₁*x^4 + a₂*x^3*y + a₃*x^2*y^2 + a₄*x*y^3 + a₅*y^4) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 16 := by
sorry

end sum_of_coefficients_l3556_355623


namespace battle_station_staffing_l3556_355650

def n : ℕ := 20
def k : ℕ := 5

theorem battle_station_staffing :
  (n.factorial) / ((n - k).factorial) = 930240 := by
  sorry

end battle_station_staffing_l3556_355650


namespace sphere_in_cube_l3556_355677

theorem sphere_in_cube (edge : ℝ) (radius : ℝ) : 
  edge = 8 →
  (4 / 3) * Real.pi * radius^3 = (1 / 2) * edge^3 →
  radius = (192 / Real.pi)^(1/3) := by
sorry

end sphere_in_cube_l3556_355677


namespace quadratic_function_value_l3556_355688

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  f a b c 1 = 7 → f a b c 3 = 19 → f a b c 2 = 12 := by
  sorry

end quadratic_function_value_l3556_355688


namespace sqrt_3_irrational_l3556_355612

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l3556_355612


namespace value_of_y_l3556_355695

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.25 * y) (h2 : x = 24) : y = 144 := by
  sorry

end value_of_y_l3556_355695


namespace product_trailing_zeros_l3556_355691

-- Define the product
def product : ℕ := 45 * 320 * 60

-- Define a function to count trailing zeros
def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + count_trailing_zeros (n / 10)
  else 0

-- Theorem statement
theorem product_trailing_zeros :
  count_trailing_zeros product = 3 := by
  sorry

end product_trailing_zeros_l3556_355691


namespace candy_distribution_l3556_355685

theorem candy_distribution (tabitha julie carlos stan : ℕ) : 
  tabitha = 22 →
  julie = tabitha / 2 →
  carlos = 2 * stan →
  tabitha + julie + carlos + stan = 72 →
  stan = 13 := by
sorry

end candy_distribution_l3556_355685


namespace shot_put_surface_area_l3556_355611

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem shot_put_surface_area :
  ∀ (d : ℝ), d = 9 → 4 * Real.pi * (d / 2)^2 = 81 * Real.pi := by
  sorry

end shot_put_surface_area_l3556_355611


namespace square_product_sequence_max_l3556_355649

/-- A sequence of natural numbers where each pair of consecutive numbers has a perfect square product -/
def SquareProductSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ k, (a n) * (a (n + 1)) = k^2

theorem square_product_sequence_max (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are different
  (SquareProductSequence a) →   -- Product of consecutive pairs is a perfect square
  (a 0 = 42) →                  -- First number is 42
  (∃ n, n < 20 ∧ a n ≥ 16800) :=  -- At least one of the first 20 numbers is ≥ 16800
by sorry

end square_product_sequence_max_l3556_355649


namespace inequalities_for_M_l3556_355680

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequalities_for_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end inequalities_for_M_l3556_355680


namespace power_of_negative_cube_l3556_355671

theorem power_of_negative_cube (a : ℝ) : (-a^3)^4 = a^12 := by sorry

end power_of_negative_cube_l3556_355671


namespace ring_area_between_circles_l3556_355694

theorem ring_area_between_circles (π : ℝ) (h : π > 0) :
  let r₁ : ℝ := 12
  let r₂ : ℝ := 7
  let area_larger := π * r₁^2
  let area_smaller := π * r₂^2
  area_larger - area_smaller = 95 * π :=
by sorry

end ring_area_between_circles_l3556_355694


namespace joint_business_profit_l3556_355657

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  a_investment : ℝ
  b_investment : ℝ
  a_period : ℝ
  b_period : ℝ
  b_profit : ℝ

/-- Calculates the total profit given the conditions of the joint business -/
def total_profit (jb : JointBusiness) : ℝ :=
  7 * jb.b_profit

/-- Theorem stating that under the given conditions, the total profit is 28000 -/
theorem joint_business_profit (jb : JointBusiness) 
  (h1 : jb.a_investment = 3 * jb.b_investment)
  (h2 : jb.a_period = 2 * jb.b_period)
  (h3 : jb.b_profit = 4000) :
  total_profit jb = 28000 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 4000 }

end joint_business_profit_l3556_355657


namespace prove_a_equals_six_l3556_355674

/-- Given a function f' and a real number a, proves that a = 6 -/
theorem prove_a_equals_six (f' : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end prove_a_equals_six_l3556_355674


namespace complex_equation_solution_l3556_355684

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I →
  Complex.I * b + a = 1 + 2 * Complex.I :=
by sorry

end complex_equation_solution_l3556_355684


namespace exists_same_color_right_triangle_l3556_355631

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

-- Define a coloring function
def Coloring := Point → Color

-- Define a property for a right-angled triangle
def isRightAngledTriangle (p q r : Point) : Prop := sorry

-- Theorem statement
theorem exists_same_color_right_triangle 
  (triangle : EquilateralTriangle) 
  (coloring : Coloring) : 
  ∃ (p q r : Point), 
    (coloring p = coloring q) ∧ 
    (coloring q = coloring r) ∧ 
    isRightAngledTriangle p q r :=
sorry

end exists_same_color_right_triangle_l3556_355631


namespace find_divisor_l3556_355686

theorem find_divisor (n : ℕ) (k : ℕ) (h1 : n + k = 8261966) (h2 : k = 11) :
  11 ∣ n + k :=
sorry

end find_divisor_l3556_355686


namespace factor_divisor_proof_l3556_355675

theorem factor_divisor_proof :
  (∃ n : ℕ, 18 = 3 * n) ∧ 
  (∃ m : ℕ, 187 = 17 * m) ∧ 
  (¬ ∃ k : ℕ, 52 = 17 * k) ∧
  (∃ p : ℕ, 160 = 8 * p) := by
  sorry

end factor_divisor_proof_l3556_355675


namespace fruit_prices_l3556_355614

/-- Fruit prices problem -/
theorem fruit_prices (total_cost apple_cost orange_cost banana_cost : ℚ) : 
  total_cost = 7.84 ∧ 
  orange_cost = apple_cost + 0.28 ∧ 
  banana_cost = apple_cost - 0.15 ∧ 
  3 * apple_cost + 7 * orange_cost + 5 * banana_cost = total_cost →
  apple_cost = 0.442 ∧ orange_cost = 0.722 ∧ banana_cost = 0.292 := by
sorry

#eval (0.442 : ℚ) + 0.28 -- Should output 0.722
#eval (0.442 : ℚ) - 0.15 -- Should output 0.292
#eval 3 * (0.442 : ℚ) + 7 * 0.722 + 5 * 0.292 -- Should output 7.84

end fruit_prices_l3556_355614


namespace x_equals_y_cubed_plus_2y_squared_minus_1_l3556_355676

theorem x_equals_y_cubed_plus_2y_squared_minus_1 (x y : ℝ) :
  x / (x - 1) = (y^3 + 2*y^2 - 1) / (y^3 + 2*y^2 - 2) → x = y^3 + 2*y^2 - 1 := by
  sorry

end x_equals_y_cubed_plus_2y_squared_minus_1_l3556_355676


namespace equation_solution_l3556_355627

theorem equation_solution :
  ∃ x : ℚ, (5 * x + 6 * x = 360 - 10 * (x - 4)) ∧ x = 400 / 21 := by
  sorry

end equation_solution_l3556_355627


namespace lily_typing_session_duration_l3556_355637

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  typing_speed : ℕ  -- words per minute
  break_duration : ℕ  -- minutes
  total_time : ℕ  -- minutes
  total_words : ℕ  -- words typed

/-- Calculates the duration of each typing session before a break -/
def typing_session_duration (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that Lily's typing session duration is 8 minutes -/
theorem lily_typing_session_duration :
  let scenario : TypingScenario := {
    typing_speed := 15,
    break_duration := 2,
    total_time := 19,
    total_words := 255
  }
  typing_session_duration scenario = 8 := by
  sorry

end lily_typing_session_duration_l3556_355637


namespace multinomial_binomial_equality_l3556_355634

theorem multinomial_binomial_equality (n : ℕ) : 
  Nat.choose n 2 * Nat.choose (n - 2) 2 = 3 * Nat.choose n 4 + 3 * Nat.choose n 3 := by
  sorry

end multinomial_binomial_equality_l3556_355634


namespace sum_of_digits_of_10_pow_100_minus_100_l3556_355636

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digits of 10^100 - 100 is 882 -/
theorem sum_of_digits_of_10_pow_100_minus_100 : 
  sum_of_digits (10^100 - 100) = 882 := by sorry

end sum_of_digits_of_10_pow_100_minus_100_l3556_355636


namespace vector_properties_l3556_355667

def a : ℝ × ℝ := (3, 0)
def b : ℝ × ℝ := (-5, 5)
def c (k : ℝ) : ℝ × ℝ := (2, k)

theorem vector_properties :
  (∃ θ : ℝ, θ = Real.pi * 3 / 4 ∧ 
    Real.cos θ = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) ∧
  (∃ k : ℝ, b.1 / (c k).1 = b.2 / (c k).2 → k = -2) ∧
  (∃ k : ℝ, b.1 * (a.1 + (c k).1) + b.2 * (a.2 + (c k).2) = 0 → k = 5) :=
by sorry

end vector_properties_l3556_355667


namespace exists_n_divisors_n_factorial_divisible_by_2019_l3556_355662

theorem exists_n_divisors_n_factorial_divisible_by_2019 :
  ∃ n : ℕ+, (2019 : ℕ) ∣ (Nat.card (Nat.divisors (Nat.factorial n))) := by
  sorry

end exists_n_divisors_n_factorial_divisible_by_2019_l3556_355662


namespace union_M_complement_N_equals_U_l3556_355600

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x < 1}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem union_M_complement_N_equals_U : M ∪ (U \ N) = U := by sorry

end union_M_complement_N_equals_U_l3556_355600


namespace jinsu_kicks_to_exceed_hoseok_l3556_355652

theorem jinsu_kicks_to_exceed_hoseok (hoseok_kicks : ℕ) (jinsu_first : ℕ) (jinsu_second : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first = 15 →
  jinsu_second = 15 →
  ∃ (jinsu_third : ℕ), 
    jinsu_third = 19 ∧ 
    jinsu_first + jinsu_second + jinsu_third > hoseok_kicks ∧
    ∀ (x : ℕ), x < 19 → jinsu_first + jinsu_second + x ≤ hoseok_kicks :=
by sorry

end jinsu_kicks_to_exceed_hoseok_l3556_355652


namespace range_of_f_l3556_355629

noncomputable def odot (a b : ℝ) : ℝ := if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ := odot (2^x) (2^(-x))

theorem range_of_f :
  (∀ y, y ∈ Set.range f → 0 < y ∧ y ≤ 1) ∧
  (∀ y, 0 < y ∧ y ≤ 1 → ∃ x, f x = y) :=
sorry

end range_of_f_l3556_355629


namespace exactly_three_solutions_exactly_two_solutions_l3556_355646

/-- The number of solutions for the system of equations:
    5|x| - 12|y| = 5
    x^2 + y^2 - 28x + 196 - a^2 = 0
-/
def numSolutions (a : ℝ) : ℕ :=
  sorry

/-- The system has exactly 3 solutions if and only if |a| = 13 or |a| = 15 -/
theorem exactly_three_solutions (a : ℝ) :
  numSolutions a = 3 ↔ (abs a = 13 ∨ abs a = 15) :=
sorry

/-- The system has exactly 2 solutions if and only if |a| = 5 or 13 < |a| < 15 -/
theorem exactly_two_solutions (a : ℝ) :
  numSolutions a = 2 ↔ (abs a = 5 ∨ (13 < abs a ∧ abs a < 15)) :=
sorry

end exactly_three_solutions_exactly_two_solutions_l3556_355646


namespace bus_rental_problem_l3556_355633

/-- Represents the capacity of buses --/
structure BusCapacity where
  typeA : ℕ
  typeB : ℕ

/-- Represents the rental plan --/
structure RentalPlan where
  bus65 : ℕ
  bus45 : ℕ
  bus30 : ℕ

/-- The main theorem to prove --/
theorem bus_rental_problem 
  (capacity : BusCapacity) 
  (plan : RentalPlan) : 
  (3 * capacity.typeA + 2 * capacity.typeB = 195) →
  (2 * capacity.typeA + 4 * capacity.typeB = 210) →
  (capacity.typeA = 45) →
  (capacity.typeB = 30) →
  (plan.bus65 = 2) →
  (plan.bus45 = 2) →
  (plan.bus30 = 3) →
  (65 * plan.bus65 + 45 * plan.bus45 + 30 * plan.bus30 = 303 + 7) →
  (plan.bus65 + plan.bus45 + plan.bus30 = 7) →
  True := by
  sorry

end bus_rental_problem_l3556_355633


namespace solve_for_y_l3556_355639

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 6 = y + 2) (h2 : x = 5) : y = 14 := by
  sorry

end solve_for_y_l3556_355639


namespace junior_score_l3556_355659

theorem junior_score (n : ℝ) (junior_score : ℝ) :
  n > 0 →
  0.1 * n * junior_score + 0.9 * n * 83 = n * 84 →
  junior_score = 93 := by
sorry

end junior_score_l3556_355659


namespace sine_cosine_inequality_l3556_355621

theorem sine_cosine_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < π / 2) (h3 : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
  sorry

end sine_cosine_inequality_l3556_355621


namespace pyramid_display_sum_l3556_355609

/-- Proves that the sum of an arithmetic sequence with given parameters is 255 -/
theorem pyramid_display_sum : 
  ∀ (a₁ aₙ d n : ℕ),
  a₁ = 12 →
  aₙ = 39 →
  d = 3 →
  aₙ = a₁ + (n - 1) * d →
  (n : ℕ) * (a₁ + aₙ) / 2 = 255 :=
by
  sorry

end pyramid_display_sum_l3556_355609


namespace cubic_expansion_coefficient_l3556_355648

theorem cubic_expansion_coefficient (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = (a * x + Real.sqrt x)^3) ∧ 
   (∃ c : ℝ, ∀ x, f x = c * x^3 + x^(5/2) * Real.sqrt x + x^2 + Real.sqrt x * x + 1 ∧ c = 20)) →
  a = Real.rpow 20 (1/3) :=
sorry

end cubic_expansion_coefficient_l3556_355648


namespace t_grid_sum_l3556_355647

/-- Represents a T-shaped grid with 6 distinct digits --/
structure TGrid where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
               d ≠ e ∧ d ≠ f ∧
               e ≠ f
  h_range : a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            f ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)
  h_vertical_sum : a + b + c = 20
  h_horizontal_sum : d + f = 7

theorem t_grid_sum (g : TGrid) : a + b + c + d + e + f = 33 := by
  sorry


end t_grid_sum_l3556_355647


namespace odd_function_value_l3556_355670

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_value :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x < 0, f x = x^3 + x + 1) →  -- f(x) = x^3 + x + 1 for x < 0
  f 2 = 9 := by sorry

end odd_function_value_l3556_355670


namespace tangent_product_identity_l3556_355607

theorem tangent_product_identity : 
  (1 + Real.tan (17 * π / 180)) * 
  (1 + Real.tan (18 * π / 180)) * 
  (1 + Real.tan (27 * π / 180)) * 
  (1 + Real.tan (28 * π / 180)) = 4 := by
  sorry

end tangent_product_identity_l3556_355607


namespace no_eight_consecutive_almost_squares_l3556_355617

/-- Definition of an almost square -/
def is_almost_square (n : ℕ) : Prop :=
  ∃ k p : ℕ, (n = k^2 ∨ n = k^2 * p) ∧ (p = 1 ∨ Nat.Prime p)

/-- Theorem stating that 8 consecutive almost squares are impossible -/
theorem no_eight_consecutive_almost_squares :
  ¬ ∃ n : ℕ, ∀ i : Fin 8, is_almost_square (n + i) :=
sorry

end no_eight_consecutive_almost_squares_l3556_355617


namespace effective_price_change_l3556_355689

theorem effective_price_change (P : ℝ) : 
  let price_after_first_discount := P * (1 - 0.3)
  let price_after_second_discount := price_after_first_discount * (1 - 0.2)
  let final_price := price_after_second_discount * (1 + 0.1)
  final_price = P * (1 - 0.384) :=
by sorry

end effective_price_change_l3556_355689


namespace squared_gt_implies_abs_gt_but_not_conversely_l3556_355683

theorem squared_gt_implies_abs_gt_but_not_conversely :
  (∀ a b : ℝ, a^2 > b^2 → |a| > b) ∧
  (∃ a b : ℝ, |a| > b ∧ a^2 ≤ b^2) :=
by sorry

end squared_gt_implies_abs_gt_but_not_conversely_l3556_355683


namespace expression_evaluation_l3556_355641

theorem expression_evaluation : 
  68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by sorry

end expression_evaluation_l3556_355641


namespace value_of_x_l3556_355672

theorem value_of_x : ∃ x : ℝ, 
  x * 0.48 * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001 ∧ 
  abs (x - 3.6) < 0.0000000000001 :=
by sorry

end value_of_x_l3556_355672


namespace balls_in_boxes_l3556_355624

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (λ i => (Nat.choose n i) * (k ^ (n - i)))

theorem balls_in_boxes : distribute_balls 6 2 = 665 := by
  sorry

end balls_in_boxes_l3556_355624


namespace danys_farm_bushels_l3556_355653

/-- Calculates the total number of bushels needed for animals on a farm for one day. -/
def total_bushels_needed (num_cows num_sheep num_chickens : ℕ) 
  (cow_sheep_consumption chicken_consumption : ℕ) : ℕ :=
  (num_cows + num_sheep) * cow_sheep_consumption + num_chickens * chicken_consumption

/-- Theorem stating the total number of bushels needed for Dany's farm animals for one day. -/
theorem danys_farm_bushels : 
  total_bushels_needed 4 3 7 2 3 = 35 := by
  sorry

end danys_farm_bushels_l3556_355653


namespace min_c_value_l3556_355615

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : b = c - 1)
  (h4 : ∃! p : ℝ × ℝ, p.1^2 + p.2 = 2003 ∧ 
    p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|) :
  c.val ≥ 1006 ∧ ∃ (a' b' : ℕ+), a' < b' ∧ b' < 1006 ∧ b' = 1005 ∧
    ∃! p : ℝ × ℝ, p.1^2 + p.2 = 2003 ∧ 
      p.2 = |p.1 - a'.val| + |p.1 - b'.val| + |p.1 - 1006| := by
  sorry

end min_c_value_l3556_355615


namespace percentage_problem_l3556_355630

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 680 = p * 1000 - 30) : p = 0.2 := by
  sorry

end percentage_problem_l3556_355630


namespace unique_f_one_l3556_355666

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2 - x * y

/-- The theorem stating that f(1) = 2 is the only solution -/
theorem unique_f_one (f : ℝ → ℝ) (h : SatisfiesEquation f) : f 1 = 2 := by
  sorry

#check unique_f_one

end unique_f_one_l3556_355666


namespace choir_group_calculation_l3556_355656

theorem choir_group_calculation (total_members : ℕ) (group1 group2 group3 : ℕ) (absent : ℕ) :
  total_members = 162 →
  group1 = 22 →
  group2 = 33 →
  group3 = 36 →
  absent = 7 →
  ∃ (group4 group5 : ℕ),
    group4 = group2 - 3 ∧
    group5 = total_members - absent - (group1 + group2 + group3 + group4) ∧
    group5 = 34 :=
by sorry

end choir_group_calculation_l3556_355656


namespace fraction_simplification_l3556_355696

theorem fraction_simplification (x : ℝ) (h : x ≠ 0) : (4 * x) / (x + 2 * x) = 4 / 3 := by
  sorry

end fraction_simplification_l3556_355696


namespace investment_problem_l3556_355604

/-- The investment problem -/
theorem investment_problem (a b total_profit a_profit : ℕ)
  (h1 : a = 6300)
  (h2 : b = 4200)
  (h3 : total_profit = 12600)
  (h4 : a_profit = 3780)
  (h5 : ∀ x : ℕ, a / (a + b + x) = a_profit / total_profit) :
  ∃ c : ℕ, c = 10500 ∧ a / (a + b + c) = a_profit / total_profit :=
sorry

end investment_problem_l3556_355604


namespace work_completion_time_l3556_355681

/-- Given that A can do a work in 6 days and B can do the same work in 12 days,
    prove that A and B working together can finish the work in 4 days. -/
theorem work_completion_time (work : ℝ) (days_A : ℝ) (days_B : ℝ)
    (h_work : work > 0)
    (h_days_A : days_A = 6)
    (h_days_B : days_B = 12) :
    work / (work / days_A + work / days_B) = 4 := by
  sorry

end work_completion_time_l3556_355681


namespace square_roots_theorem_l3556_355660

theorem square_roots_theorem (x : ℝ) (m : ℝ) : 
  x > 0 → (2*m - 1)^2 = x → (2 - m)^2 = x → x = 9 := by
  sorry

end square_roots_theorem_l3556_355660


namespace club_size_after_four_years_l3556_355645

/-- Represents the number of people in the club after k years -/
def club_size (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 3 * club_size n - 14

/-- The theorem stating the club size after 4 years -/
theorem club_size_after_four_years :
  club_size 4 = 1060 := by
  sorry

end club_size_after_four_years_l3556_355645


namespace fraction_unchanged_l3556_355655

theorem fraction_unchanged (x y : ℝ) : 
  x / (3 * x + y) = (3 * x) / (3 * (3 * x + y)) :=
by sorry

end fraction_unchanged_l3556_355655


namespace gohul_independent_time_l3556_355610

/-- Ram's work rate in job completion per day -/
def ram_rate : ℚ := 1 / 10

/-- Time taken when Ram and Gohul work together -/
def combined_time : ℚ := 5.999999999999999

/-- Gohul's independent work time -/
def gohul_time : ℚ := 15

/-- Combined work rate of Ram and Gohul -/
def combined_rate : ℚ := 1 / combined_time

theorem gohul_independent_time :
  ram_rate + (1 / gohul_time) = combined_rate :=
sorry

end gohul_independent_time_l3556_355610


namespace license_plate_count_l3556_355644

/-- The number of possible letters in each position of the license plate. -/
def num_letters : ℕ := 26

/-- The number of positions for digits in the license plate. -/
def num_digit_positions : ℕ := 3

/-- The number of ways to choose positions for odd digits. -/
def num_odd_digit_arrangements : ℕ := 3

/-- The number of possible odd digits. -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits. -/
def num_even_digits : ℕ := 5

/-- The total number of license plates with 3 letters followed by 3 digits,
    where exactly two digits are odd and one is even. -/
theorem license_plate_count : 
  (num_letters ^ 3) * num_odd_digit_arrangements * (num_odd_digits ^ 2 * num_even_digits) = 6591000 :=
by sorry

end license_plate_count_l3556_355644


namespace gold_bars_weighing_l3556_355601

theorem gold_bars_weighing (C₁ C₂ C₃ C₄ C₅ C₆ C₇ C₈ C₉ C₁₀ C₁₁ C₁₂ C₁₃ : ℝ) 
  (h₁ : C₁ ≥ 0) (h₂ : C₂ ≥ 0) (h₃ : C₃ ≥ 0) (h₄ : C₄ ≥ 0) (h₅ : C₅ ≥ 0)
  (h₆ : C₆ ≥ 0) (h₇ : C₇ ≥ 0) (h₈ : C₈ ≥ 0) (h₉ : C₉ ≥ 0) (h₁₀ : C₁₀ ≥ 0)
  (h₁₁ : C₁₁ ≥ 0) (h₁₂ : C₁₂ ≥ 0) (h₁₃ : C₁₃ ≥ 0)
  (W₁ : ℝ) (hW₁ : W₁ = C₁ + C₂)
  (W₂ : ℝ) (hW₂ : W₂ = C₁ + C₃)
  (W₃ : ℝ) (hW₃ : W₃ = C₂ + C₃)
  (W₄ : ℝ) (hW₄ : W₄ = C₄ + C₅)
  (W₅ : ℝ) (hW₅ : W₅ = C₆ + C₇)
  (W₆ : ℝ) (hW₆ : W₆ = C₈ + C₉)
  (W₇ : ℝ) (hW₇ : W₇ = C₁₀ + C₁₁)
  (W₈ : ℝ) (hW₈ : W₈ = C₁₂ + C₁₃) :
  C₁ + C₂ + C₃ + C₄ + C₅ + C₆ + C₇ + C₈ + C₉ + C₁₀ + C₁₁ + C₁₂ + C₁₃ = 
  (W₁ + W₂ + W₃) / 2 + (W₄ + W₅ + W₆ + W₇ + W₈) :=
by sorry

end gold_bars_weighing_l3556_355601


namespace odd_function_inequality_l3556_355626

/-- An odd, differentiable function satisfying certain conditions -/
structure OddFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (-x) = -f x
  diff : Differentiable ℝ f
  cond : ∀ x < 0, 2 * f x + x * deriv f x < 0

/-- Theorem stating the relationship between f(1), 2016f(√2016), and 2017f(√2017) -/
theorem odd_function_inequality (f : OddFunction) :
  f.f 1 < 2016 * f.f (Real.sqrt 2016) ∧
  2016 * f.f (Real.sqrt 2016) < 2017 * f.f (Real.sqrt 2017) := by
  sorry

end odd_function_inequality_l3556_355626


namespace function_property_l3556_355673

/-- Given a function f(x, y) = kx + 1/y, if f(a, b) = f(b, a) for a ≠ b, then f(ab, 1) = 0 -/
theorem function_property (k : ℝ) (a b : ℝ) (h : a ≠ b) :
  (k * a + 1 / b = k * b + 1 / a) → (k * (a * b) + 1 = 0) :=
by sorry

end function_property_l3556_355673


namespace algebraic_expression_value_l3556_355661

theorem algebraic_expression_value :
  let a : ℝ := 1 + Real.sqrt 2
  let b : ℝ := Real.sqrt 3
  a^2 + b^2 - 2*a + 1 = 5 := by
sorry

end algebraic_expression_value_l3556_355661


namespace division_of_fractions_l3556_355619

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end division_of_fractions_l3556_355619


namespace total_potatoes_to_cook_l3556_355602

/-- Given a cooking scenario where:
  * 6 potatoes are already cooked
  * Each potato takes 8 minutes to cook
  * It takes 72 minutes to cook the remaining potatoes
  Prove that the total number of potatoes to be cooked is 15. -/
theorem total_potatoes_to_cook (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) :
  already_cooked = 6 →
  cooking_time_per_potato = 8 →
  remaining_cooking_time = 72 →
  already_cooked + (remaining_cooking_time / cooking_time_per_potato) = 15 :=
by sorry

end total_potatoes_to_cook_l3556_355602


namespace pond_width_calculation_l3556_355620

/-- Represents a rectangular pond -/
structure RectangularPond where
  length : ℝ
  width : ℝ
  depth : ℝ
  volume : ℝ

/-- Theorem: Given a rectangular pond with length 20 meters, depth 5 meters, 
    and volume 1200 cubic meters, its width is 12 meters -/
theorem pond_width_calculation (pond : RectangularPond) 
  (h1 : pond.length = 20)
  (h2 : pond.depth = 5)
  (h3 : pond.volume = 1200)
  (h4 : pond.volume = pond.length * pond.width * pond.depth) :
  pond.width = 12 := by
  sorry

end pond_width_calculation_l3556_355620


namespace product_of_binary_and_ternary_l3556_355632

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldl (fun acc (i, digit) => acc + digit * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary := [true, true, false, true]  -- 1011 in binary (least significant bit first)
  let ternary := [2, 0, 1]  -- 102 in ternary (least significant digit first)
  (binary_to_decimal binary) * (ternary_to_decimal ternary) = 121 := by
  sorry

end product_of_binary_and_ternary_l3556_355632


namespace count_valid_markings_l3556_355699

/-- Represents a valid marking of an 8x8 chessboard -/
def ValidMarking : Type := 
  { marking : Fin 8 → Fin 8 // 
    (∀ i j, i ≠ j → marking i ≠ marking j) ∧ 
    (∀ i, marking i ≠ 0 ∧ marking i ≠ 7) ∧
    (marking 0 ≠ 0 ∧ marking 0 ≠ 7) ∧
    (marking 7 ≠ 0 ∧ marking 7 ≠ 7) }

/-- The number of valid markings on an 8x8 chessboard -/
def numValidMarkings : ℕ := sorry

/-- The theorem stating the number of valid markings -/
theorem count_valid_markings : numValidMarkings = 21600 := by sorry

end count_valid_markings_l3556_355699


namespace complex_square_root_l3556_355654

theorem complex_square_root (z : ℂ) : z^2 = -5 - 12*I → z = 2 - 3*I ∨ z = -2 + 3*I := by
  sorry

end complex_square_root_l3556_355654


namespace fraction_identification_l3556_355618

-- Define what a fraction is
def is_fraction (x : ℚ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = n / d

-- Define the given expressions
def expr1 (a : ℚ) : ℚ := 2 / a
def expr2 (a : ℚ) : ℚ := 2 * a / 3
def expr3 (b : ℚ) : ℚ := -b / 2
def expr4 (a : ℚ) : ℚ := (3 * a + 1) / 2

-- State the theorem
theorem fraction_identification (a b : ℚ) (ha : a ≠ 0) : 
  is_fraction (expr1 a) ∧ 
  ¬is_fraction (expr2 a) ∧ 
  ¬is_fraction (expr3 b) ∧ 
  ¬is_fraction (expr4 a) :=
sorry

end fraction_identification_l3556_355618


namespace unit_vectors_collinear_with_AB_l3556_355663

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def unitVectorAB : Set (ℝ × ℝ) := {v | ∃ k : ℝ, k ≠ 0 ∧ v = (k * AB.1, k * AB.2) ∧ v.1^2 + v.2^2 = 1}

theorem unit_vectors_collinear_with_AB :
  unitVectorAB = {(3/5, -4/5), (-3/5, 4/5)} :=
by sorry

end unit_vectors_collinear_with_AB_l3556_355663


namespace twenty_fifth_decimal_of_n_over_11_l3556_355668

theorem twenty_fifth_decimal_of_n_over_11 (n : ℕ) (h : n / 11 = 9) :
  (n : ℚ) / 11 - (n / 11 : ℕ) = 0 :=
sorry

end twenty_fifth_decimal_of_n_over_11_l3556_355668


namespace store_earnings_proof_l3556_355606

def store_earnings : ℕ := by
  let graphics_cards := 10 * 600
  let hard_drives := 14 * 80
  let cpus := 8 * 200
  let ram := 4 * 60
  let power_supply_units := 12 * 90
  let monitors := 6 * 250
  let keyboards := 18 * 40
  let mice := 24 * 20
  exact graphics_cards + hard_drives + cpus + ram + power_supply_units + monitors + keyboards + mice

theorem store_earnings_proof : store_earnings = 12740 := by
  sorry

end store_earnings_proof_l3556_355606


namespace geometric_sequence_sum_l3556_355687

theorem geometric_sequence_sum (n : ℕ) (a : ℝ) (r : ℝ) (S : ℝ) : 
  a = 1 → r = (1/2 : ℝ) → S = (31/16 : ℝ) → S = a * (1 - r^n) / (1 - r) → n = 5 := by
  sorry

end geometric_sequence_sum_l3556_355687


namespace union_sets_implies_m_equals_three_l3556_355638

theorem union_sets_implies_m_equals_three (m : ℝ) :
  let A : Set ℝ := {2, m}
  let B : Set ℝ := {1, m^2}
  A ∪ B = {1, 2, 3, 9} →
  m = 3 := by
sorry

end union_sets_implies_m_equals_three_l3556_355638


namespace sarahs_deleted_folder_size_l3556_355603

theorem sarahs_deleted_folder_size 
  (initial_free : ℝ) 
  (initial_used : ℝ) 
  (new_files_size : ℝ) 
  (new_drive_size : ℝ) 
  (new_drive_free : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : new_files_size = 2)
  (h4 : new_drive_size = 20)
  (h5 : new_drive_free = 10) : 
  ∃ (deleted_folder_size : ℝ), 
    deleted_folder_size = 4.6 ∧ 
    initial_used - deleted_folder_size + new_files_size = new_drive_size - new_drive_free :=
by sorry

end sarahs_deleted_folder_size_l3556_355603


namespace bee_count_correct_l3556_355658

def bee_count (day : Nat) : ℕ :=
  match day with
  | 0 => 144  -- Monday
  | 1 => 432  -- Tuesday
  | 2 => 216  -- Wednesday
  | 3 => 432  -- Thursday
  | 4 => 648  -- Friday
  | 5 => 486  -- Saturday
  | 6 => 1944 -- Sunday
  | _ => 0    -- Invalid day

def daily_multiplier (day : Nat) : ℚ :=
  match day with
  | 0 => 1    -- Monday (base)
  | 1 => 3    -- Tuesday
  | 2 => 1/2  -- Wednesday
  | 3 => 2    -- Thursday
  | 4 => 3/2  -- Friday
  | 5 => 3/4  -- Saturday
  | 6 => 4    -- Sunday
  | _ => 0    -- Invalid day

theorem bee_count_correct (day : Nat) :
  day < 7 →
  (day = 0 ∨ (bee_count day : ℚ) = (bee_count (day - 1) : ℚ) * daily_multiplier day) :=
by sorry

end bee_count_correct_l3556_355658


namespace scientific_notation_equivalence_l3556_355628

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.000136 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.36 ∧ n = -4 := by
  sorry

end scientific_notation_equivalence_l3556_355628


namespace restaurant_menu_fraction_l3556_355669

/-- Given a restaurant menu with vegan dishes and dietary restrictions, 
    calculate the fraction of suitable dishes. -/
theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (vegan_dishes : ℕ) (restricted_vegan_dishes : ℕ) : 
  vegan_dishes = (3 : ℕ) * total_dishes / 10 →
  vegan_dishes = 9 →
  restricted_vegan_dishes = 7 →
  (vegan_dishes - restricted_vegan_dishes : ℚ) / total_dishes = 1 / 15 := by
  sorry

end restaurant_menu_fraction_l3556_355669


namespace sin_double_angle_special_case_l3556_355693

theorem sin_double_angle_special_case (θ : ℝ) (h : Real.tan θ + (Real.tan θ)⁻¹ = Real.sqrt 5) : 
  Real.sin (2 * θ) = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_double_angle_special_case_l3556_355693


namespace chloe_final_score_l3556_355608

/-- Chloe's points at the end of a trivia game -/
theorem chloe_final_score (first_round second_round last_round : Int) 
  (h1 : first_round = 40)
  (h2 : second_round = 50)
  (h3 : last_round = -4) :
  first_round + second_round + last_round = 86 := by
  sorry

end chloe_final_score_l3556_355608


namespace sum_of_f_powers_equals_510_l3556_355665

def f (x : ℚ) : ℚ := (1 + 10 * x) / (10 - 100 * x)

def f_power (n : ℕ) : ℚ → ℚ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_power n)

theorem sum_of_f_powers_equals_510 :
  (Finset.range 6000).sum (λ n => f_power (n + 1) (1/2)) = 510 := by
  sorry

end sum_of_f_powers_equals_510_l3556_355665


namespace original_bananas_total_l3556_355664

theorem original_bananas_total (willie_bananas charles_bananas : ℝ) 
  (h1 : willie_bananas = 48.0) 
  (h2 : charles_bananas = 35.0) : 
  willie_bananas + charles_bananas = 83.0 := by
  sorry

end original_bananas_total_l3556_355664


namespace complex_magnitude_problem_l3556_355616

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 20)
  (h2 : Complex.abs (z + 2 * w) = 10)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = 0 := by
  sorry

end complex_magnitude_problem_l3556_355616


namespace factoring_equation_l3556_355622

theorem factoring_equation (m : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + m * x + 1 = (2 * x - 1)^2) → 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, 4 * x^2 + m * x + 1 = f x * f x :=
by sorry

end factoring_equation_l3556_355622


namespace count_magic_numbers_l3556_355697

def is_magic_number (N : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k > 0 ∧ (m * 10^k + N) % N = 0

theorem count_magic_numbers :
  (∃! (L : List ℕ), 
    (∀ N ∈ L, N < 600 ∧ is_magic_number N) ∧
    (∀ N < 600, is_magic_number N → N ∈ L) ∧
    L.length = 13) :=
sorry

end count_magic_numbers_l3556_355697


namespace min_sum_squares_of_roots_l3556_355642

theorem min_sum_squares_of_roots (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + a*x₁ + a + 3 = 0 → 
  x₂^2 + a*x₂ + a + 3 = 0 → 
  x₁ ≠ x₂ →
  ∃ (m : ℝ), ∀ (b : ℝ) (y₁ y₂ : ℝ), 
    y₁^2 + b*y₁ + b + 3 = 0 → 
    y₂^2 + b*y₂ + b + 3 = 0 → 
    y₁ ≠ y₂ →
    y₁^2 + y₂^2 ≥ m ∧ 
    x₁^2 + x₂^2 = m ∧
    m = 2 :=
by sorry

end min_sum_squares_of_roots_l3556_355642


namespace marching_band_members_l3556_355678

theorem marching_band_members : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 2 ∧ 
  n = 163 := by
sorry

end marching_band_members_l3556_355678


namespace point_not_in_fourth_quadrant_l3556_355690

theorem point_not_in_fourth_quadrant :
  ¬ ∃ a : ℝ, (a - 3 > 0 ∧ a + 3 < 0) := by
  sorry

end point_not_in_fourth_quadrant_l3556_355690
