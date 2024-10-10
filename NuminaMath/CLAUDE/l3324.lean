import Mathlib

namespace polynomial_not_equal_77_l3324_332437

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
sorry

end polynomial_not_equal_77_l3324_332437


namespace monomial_properties_l3324_332442

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coeff : ℚ
  a_exp : ℕ
  b_exp : ℕ

/-- The given monomial 3a²b/2 -/
def given_monomial : Monomial := { coeff := 3/2, a_exp := 2, b_exp := 1 }

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℚ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.a_exp + m.b_exp

theorem monomial_properties :
  coefficient given_monomial = 3/2 ∧ degree given_monomial = 3 := by
  sorry

end monomial_properties_l3324_332442


namespace hcf_problem_l3324_332464

theorem hcf_problem (a b h : ℕ) (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b) :
  (Nat.gcd a b = h) →
  (∃ k : ℕ, Nat.lcm a b = 10 * 15 * k) →
  (max a b = 450) →
  h = 30 := by
sorry

end hcf_problem_l3324_332464


namespace probability_select_from_both_sets_l3324_332463

/-- The probability of selecting one card from each of two sets when drawing two cards at random without replacement, given a total of 13 cards where one set has 6 cards and the other has 7 cards. -/
theorem probability_select_from_both_sets : 
  ∀ (total : ℕ) (set1 : ℕ) (set2 : ℕ),
  total = 13 → set1 = 6 → set2 = 7 →
  (set1 / total * set2 / (total - 1) + set2 / total * set1 / (total - 1) : ℚ) = 7 / 13 := by
sorry

end probability_select_from_both_sets_l3324_332463


namespace problem_1_problem_2_problem_3_l3324_332459

-- Problem 1
theorem problem_1 : 3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27 = 6 * Real.sqrt 3 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12) = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by sorry

-- Problem 3
theorem problem_3 : (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6)^2 = 3 + 2 * Real.sqrt 30 := by sorry

end problem_1_problem_2_problem_3_l3324_332459


namespace division_remainder_l3324_332445

theorem division_remainder : ∃ r : ℕ, 
  12401 = 163 * 76 + r ∧ r < 163 :=
by
  -- The proof goes here
  sorry

end division_remainder_l3324_332445


namespace sale_price_lower_than_original_l3324_332475

theorem sale_price_lower_than_original (x : ℝ) (h : x > 0) :
  0.75 * (1.30 * x) < x := by
  sorry

end sale_price_lower_than_original_l3324_332475


namespace polygon_diagonals_equal_sides_l3324_332460

theorem polygon_diagonals_equal_sides : ∃ (n : ℕ), n > 0 ∧ n * (n - 3) / 2 = n := by
  sorry

end polygon_diagonals_equal_sides_l3324_332460


namespace max_value_expression_l3324_332455

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + b^2)) ≤ a^2 + b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + b^2)) = a^2 + b^2) := by
  sorry

end max_value_expression_l3324_332455


namespace binary_11001_is_25_l3324_332435

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11001_is_25 :
  binary_to_decimal [true, false, false, true, true] = 25 := by
  sorry

end binary_11001_is_25_l3324_332435


namespace sum_of_fractions_equals_negative_three_l3324_332447

theorem sum_of_fractions_equals_negative_three 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h_sum : a + b + c = 3) :
  1 / (b^2 + c^2 - 3*a^2) + 1 / (a^2 + c^2 - 3*b^2) + 1 / (a^2 + b^2 - 3*c^2) = -3 := by
  sorry

end sum_of_fractions_equals_negative_three_l3324_332447


namespace cubic_function_derivative_l3324_332439

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
  sorry

end cubic_function_derivative_l3324_332439


namespace area_of_triangle_ABC_l3324_332402

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Triangle ABC on a grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  (1/2 : ℚ) * ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) : ℤ)

theorem area_of_triangle_ABC (t : GridTriangle) :
  t.A = ⟨0, 0⟩ →
  t.B = ⟨0, 2⟩ →
  t.C = ⟨3, 0⟩ →
  triangleArea t.A t.B t.C = 1/2 := by
  sorry

#check area_of_triangle_ABC

end area_of_triangle_ABC_l3324_332402


namespace equal_ratios_imply_k_value_l3324_332410

theorem equal_ratios_imply_k_value (x y z k : ℝ) 
  (h1 : 12 / (x + z) = k / (z - y))
  (h2 : k / (z - y) = 5 / (y - x))
  (h3 : y = 0) : k = 17 := by
  sorry

end equal_ratios_imply_k_value_l3324_332410


namespace hawks_score_l3324_332462

/-- 
Given:
- The total points scored by both teams is 50
- The Eagles won by a margin of 18 points

Prove that the Hawks scored 16 points
-/
theorem hawks_score (total_points eagles_points hawks_points : ℕ) : 
  total_points = 50 →
  eagles_points = hawks_points + 18 →
  eagles_points + hawks_points = total_points →
  hawks_points = 16 := by
sorry

end hawks_score_l3324_332462


namespace valid_XY_for_divisibility_by_72_l3324_332457

/- Define a function to represent the number 42X4Y -/
def number (X Y : ℕ) : ℕ := 42000 + X * 100 + 40 + Y

/- Define the property of being a single digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/- Define the main theorem -/
theorem valid_XY_for_divisibility_by_72 :
  ∀ X Y : ℕ, 
  is_single_digit X → is_single_digit Y →
  (number X Y % 72 = 0 ↔ ((X = 8 ∧ Y = 0) ∨ (X = 0 ∧ Y = 8))) :=
by sorry

end valid_XY_for_divisibility_by_72_l3324_332457


namespace four_correct_statements_l3324_332412

theorem four_correct_statements (a b m : ℝ) : 
  -- Statement 1
  (∀ m, a * m^2 > b * m^2 → a > b) ∧
  -- Statement 2
  (a > b → a * |a| > b * |b|) ∧
  -- Statement 3
  (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b) ∧
  -- Statement 4
  (a > b ∧ b > 0 ∧ |Real.log a| = |Real.log b| → 2 * a + b > 3) :=
by sorry

end four_correct_statements_l3324_332412


namespace difference_of_squares_l3324_332426

theorem difference_of_squares (a b : ℝ) : (2*a + b) * (b - 2*a) = b^2 - 4*a^2 := by
  sorry

end difference_of_squares_l3324_332426


namespace intersection_is_open_interval_l3324_332428

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

-- Define the complement of M relative to ℝ
def M_complement : Set ℝ := {y | y ∉ M}

-- Define the intersection of M_complement and N
def intersection : Set ℝ := M_complement ∩ N

-- Theorem statement
theorem intersection_is_open_interval :
  intersection = {x | -Real.sqrt 3 < x ∧ x < -1} :=
by sorry

end intersection_is_open_interval_l3324_332428


namespace tshirt_cost_l3324_332474

-- Define the problem parameters
def initial_amount : ℕ := 26
def jumper_cost : ℕ := 9
def heels_cost : ℕ := 5
def remaining_amount : ℕ := 8

-- Define the theorem to prove
theorem tshirt_cost : 
  initial_amount - jumper_cost - heels_cost - remaining_amount = 4 := by
  sorry

end tshirt_cost_l3324_332474


namespace sum_of_first_three_terms_l3324_332415

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define S_3 as the sum of the first three terms
def S3 : ℚ := a 1 + a 2 + a 3

-- Theorem statement
theorem sum_of_first_three_terms : S3 = 10 := by
  sorry

end sum_of_first_three_terms_l3324_332415


namespace zoo_ratio_l3324_332480

theorem zoo_ratio (sea_lions penguins : ℕ) : 
  sea_lions = 48 →
  penguins = sea_lions + 84 →
  (sea_lions : ℚ) / penguins = 4 / 11 := by
  sorry

end zoo_ratio_l3324_332480


namespace cookie_theorem_l3324_332472

/-- The number of combinations when selecting 8 cookies from 4 types, with at least one of each type -/
def cookieCombinations : ℕ := 46

/-- The function that calculates the number of combinations -/
def calculateCombinations (totalCookies : ℕ) (cookieTypes : ℕ) : ℕ :=
  sorry

theorem cookie_theorem :
  calculateCombinations 8 4 = cookieCombinations :=
by sorry

end cookie_theorem_l3324_332472


namespace cube_edge_length_equality_l3324_332433

theorem cube_edge_length_equality (a : ℝ) : 
  let parallelepiped_volume : ℝ := 2 * 3 * 6
  let parallelepiped_surface_area : ℝ := 2 * (2 * 3 + 3 * 6 + 2 * 6)
  let cube_volume : ℝ := a^3
  let cube_surface_area : ℝ := 6 * a^2
  (parallelepiped_volume / cube_volume = parallelepiped_surface_area / cube_surface_area) → a = 3 :=
by sorry

end cube_edge_length_equality_l3324_332433


namespace green_ball_probability_l3324_332477

-- Define the number of balls of each color
def green_balls : ℕ := 2
def black_balls : ℕ := 3
def red_balls : ℕ := 6

-- Define the total number of balls
def total_balls : ℕ := green_balls + black_balls + red_balls

-- Define the probability of drawing a green ball
def prob_green_ball : ℚ := green_balls / total_balls

-- Theorem stating the probability of drawing a green ball
theorem green_ball_probability : prob_green_ball = 2 / 11 := by
  sorry

end green_ball_probability_l3324_332477


namespace direct_inverse_variation_l3324_332438

/-- Given that R varies directly as S and inversely as T, 
    prove that S = 20/3 when R = 5 and T = 1/3, 
    given that R = 2, T = 1/2, and S = 4 in another case. -/
theorem direct_inverse_variation (R S T : ℚ) : 
  (∃ k : ℚ, ∀ R S T, R = k * S / T) →  -- R varies directly as S and inversely as T
  (2 : ℚ) = (4 : ℚ) / (1/2 : ℚ) →      -- When R = 2, S = 4, and T = 1/2
  (5 : ℚ) = S / (1/3 : ℚ) →            -- When R = 5 and T = 1/3
  S = 20/3 := by
sorry

end direct_inverse_variation_l3324_332438


namespace union_of_A_and_B_l3324_332484

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

theorem union_of_A_and_B : A ∪ B = Ioc (-2) 4 := by
  sorry

end union_of_A_and_B_l3324_332484


namespace negative_fraction_comparison_l3324_332481

theorem negative_fraction_comparison : -3/4 > -4/5 := by
  sorry

end negative_fraction_comparison_l3324_332481


namespace function_inequality_constraint_l3324_332409

theorem function_inequality_constraint (x a : ℝ) : 
  x > 0 → (2 * x + 1 > a * x) → a ≤ 2 := by
  sorry

end function_inequality_constraint_l3324_332409


namespace no_five_consecutive_divisible_by_2025_l3324_332470

def x (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 :
  ∀ k : ℕ, ∃ i : Fin 5, ¬(2025 ∣ x (k + i.val)) :=
by sorry

end no_five_consecutive_divisible_by_2025_l3324_332470


namespace a_less_than_one_sufficient_not_necessary_l3324_332446

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 + 2 * a * x + 6 * y + 5 * a = 0

-- Define what it means for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

-- Theorem stating that a < 1 is sufficient but not necessary
theorem a_less_than_one_sufficient_not_necessary :
  (∀ a : ℝ, a < 1 → is_circle a) ∧
  ¬(∀ a : ℝ, is_circle a → a < 1) :=
sorry

end a_less_than_one_sufficient_not_necessary_l3324_332446


namespace not_all_perfect_squares_l3324_332432

theorem not_all_perfect_squares (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k ^ 2 := by
  sorry

end not_all_perfect_squares_l3324_332432


namespace circle_center_l3324_332423

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center (x y : ℝ) : 
  (3 * x + 4 * y = 24) →  -- First tangent line
  (3 * x + 4 * y = -16) →  -- Second tangent line
  (x - 2 * y = 0) →  -- Line containing the center
  (x = 4/5 ∧ y = 2/5)  -- Center coordinates
  :=
by sorry

end circle_center_l3324_332423


namespace perfect_square_k_l3324_332493

theorem perfect_square_k (K : ℕ) (h1 : K > 1) (h2 : 1000 < K^4) (h3 : K^4 < 5000) :
  ∃ (n : ℕ), K^4 = n^2 ↔ K = 6 ∨ K = 7 ∨ K = 8 := by
  sorry

end perfect_square_k_l3324_332493


namespace octal_563_equals_base12_261_l3324_332488

-- Define a function to convert from octal to decimal
def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from decimal to base 12
def decimal_to_base12 (n : ℕ) : ℕ :=
  (n / 144) * 100 + ((n / 12) % 12) * 10 + (n % 12)

-- Theorem statement
theorem octal_563_equals_base12_261 :
  decimal_to_base12 (octal_to_decimal 563) = 261 :=
sorry

end octal_563_equals_base12_261_l3324_332488


namespace total_money_end_is_3933_33_l3324_332498

/-- Calculates the total money after splitting and increasing the remainder -/
def totalMoneyAtEnd (cecilMoney : ℚ) : ℚ :=
  let catherineMoney := 2 * cecilMoney - 250
  let carmelaMoney := 2 * cecilMoney + 50
  let averageMoney := (cecilMoney + catherineMoney + carmelaMoney) / 3
  let carlosMoney := averageMoney + 200
  let totalMoney := cecilMoney + catherineMoney + carmelaMoney + carlosMoney
  let splitAmount := totalMoney / 7
  let remainingAmount := totalMoney - (splitAmount * 7)
  let increase := remainingAmount * (5 / 100)
  totalMoney + increase

/-- Theorem stating that the total money at the end is $3933.33 -/
theorem total_money_end_is_3933_33 :
  totalMoneyAtEnd 600 = 3933.33 := by sorry

end total_money_end_is_3933_33_l3324_332498


namespace complement_of_union_l3324_332491

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (M ∪ N)) = {4} := by
  sorry

end complement_of_union_l3324_332491


namespace inequality_proof_l3324_332425

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 1) :
  (x^(n-1) - 1) / (n-1 : ℝ) ≤ (x^n - 1) / n :=
by sorry

end inequality_proof_l3324_332425


namespace simplify_expression_l3324_332420

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end simplify_expression_l3324_332420


namespace product_equals_one_l3324_332403

theorem product_equals_one :
  16 * 0.5 * 4 * 0.0625 / 2 = 1 := by
  sorry

end product_equals_one_l3324_332403


namespace product_of_tangents_plus_one_l3324_332421

theorem product_of_tangents_plus_one (α : ℝ) :
  (1 + Real.tan (α * π / 12)) * (1 + Real.tan (α * π / 6)) = 2 := by
  sorry

end product_of_tangents_plus_one_l3324_332421


namespace stratified_sampling_correct_l3324_332427

/-- Represents the job titles in the school --/
inductive JobTitle
| Senior
| Intermediate
| Clerk

/-- Represents the school staff distribution --/
structure StaffDistribution where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  clerk : ℕ
  sum_eq_total : senior + intermediate + clerk = total

/-- Represents the sample distribution --/
structure SampleDistribution where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  clerk : ℕ
  sum_eq_total : senior + intermediate + clerk = total

/-- Checks if a sample distribution is correctly stratified --/
def isCorrectlySampled (staff : StaffDistribution) (sample : SampleDistribution) : Prop :=
  sample.senior * staff.total = staff.senior * sample.total ∧
  sample.intermediate * staff.total = staff.intermediate * sample.total ∧
  sample.clerk * staff.total = staff.clerk * sample.total

/-- The main theorem to prove --/
theorem stratified_sampling_correct 
  (staff : StaffDistribution)
  (sample : SampleDistribution)
  (h_staff : staff = { 
    total := 150, 
    senior := 45, 
    intermediate := 90, 
    clerk := 15, 
    sum_eq_total := by norm_num
  })
  (h_sample : sample = {
    total := 10,
    senior := 3,
    intermediate := 6,
    clerk := 1,
    sum_eq_total := by norm_num
  }) : 
  isCorrectlySampled staff sample := by
  sorry

end stratified_sampling_correct_l3324_332427


namespace quadratic_inequality_condition_l3324_332495

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_inequality_condition_l3324_332495


namespace analysis_method_seeks_sufficient_condition_l3324_332422

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  trace_effect_to_cause : Bool
  start_from_inequality : Bool

/-- A condition in the context of inequality proofs -/
inductive Condition
  | necessary
  | sufficient
  | necessary_and_sufficient
  | necessary_or_sufficient

/-- The condition sought by the analysis method -/
def condition_sought (method : AnalysisMethod) : Condition :=
  Condition.sufficient

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition (method : AnalysisMethod) 
  (h1 : method.trace_effect_to_cause = true) 
  (h2 : method.start_from_inequality = true) : 
  condition_sought method = Condition.sufficient := by sorry

end analysis_method_seeks_sufficient_condition_l3324_332422


namespace problem_solution_l3324_332479

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -5) :
  x + x^3 / y^2 + y^3 / x^2 + y = 285 := by
  sorry

end problem_solution_l3324_332479


namespace cookie_ratio_l3324_332492

/-- Proves that the ratio of cookies baked by Jake to Clementine is 2:1 given the problem conditions -/
theorem cookie_ratio (clementine jake tory : ℕ) (total_revenue : ℕ) : 
  clementine = 72 →
  tory = (jake + clementine) / 2 →
  total_revenue = 648 →
  2 * (clementine + jake + tory) = total_revenue →
  jake = 2 * clementine :=
by sorry

end cookie_ratio_l3324_332492


namespace propositions_truth_l3324_332471

-- Define the necessary geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- Define the propositions
def proposition_1 (p1 p2 p3 : Plane) (l1 l2 : Line) : Prop :=
  line_in_plane l1 p1 → line_in_plane l2 p1 →
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2

def proposition_2 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2

def proposition_3 (l1 l2 l3 : Line) : Prop :=
  perpendicular_line_plane l1 l3 → perpendicular_line_plane l2 l3 → parallel l1 l2

def proposition_4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular p1 p2 →
  line_in_plane l p1 →
  ¬perpendicular_line_plane l (line_of_intersection p1 p2) →
  ¬perpendicular_line_plane l p2

-- Theorem stating which propositions are true and which are false
theorem propositions_truth : 
  (∃ p1 p2 p3 : Plane, ∃ l1 l2 : Line, ¬proposition_1 p1 p2 p3 l1 l2) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_2 p1 p2 l) ∧
  (∃ l1 l2 l3 : Line, ¬proposition_3 l1 l2 l3) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_4 p1 p2 l) :=
sorry

end propositions_truth_l3324_332471


namespace mod_equivalence_unique_solution_l3324_332465

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2753 [ZMOD 8] ∧ n = 7 := by
  sorry

end mod_equivalence_unique_solution_l3324_332465


namespace max_marks_proof_l3324_332453

/-- Given a student needs 60% to pass, got 220 marks, and failed by 50 marks, prove the maximum marks are 450. -/
theorem max_marks_proof (passing_percentage : Real) (student_marks : ℕ) (failing_margin : ℕ) 
  (h1 : passing_percentage = 0.60)
  (h2 : student_marks = 220)
  (h3 : failing_margin = 50) :
  (student_marks + failing_margin) / passing_percentage = 450 := by
  sorry

end max_marks_proof_l3324_332453


namespace complex_number_location_l3324_332452

theorem complex_number_location : ∃ (z : ℂ), z = Complex.I * (Complex.I - 1) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_location_l3324_332452


namespace inequality_proofs_l3324_332478

theorem inequality_proofs 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) 
  (x : ℝ) (hx : x ≥ 0) 
  (a b p q : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) (hq : q > 0)
  (hpq : 1 / p + 1 / q = 1) : 
  (x^α - α*x ≤ 1 - α) ∧ (a * b ≤ (1/p) * a^p + (1/q) * b^q) := by
  sorry

end inequality_proofs_l3324_332478


namespace youngest_child_age_l3324_332466

def arithmetic_progression (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem youngest_child_age 
  (children : ℕ) 
  (ages : List ℕ) 
  (h_children : children = 8)
  (h_ages : ages = arithmetic_progression 2 3 7)
  : ages.head? = some 2 := by
  sorry

#eval arithmetic_progression 2 3 7

end youngest_child_age_l3324_332466


namespace vince_monthly_savings_l3324_332451

/-- Calculate Vince's monthly savings given the salon conditions --/
theorem vince_monthly_savings :
  let haircut_price : ℝ := 18
  let coloring_price : ℝ := 30
  let treatment_price : ℝ := 40
  let fixed_expenses : ℝ := 280
  let product_cost_per_customer : ℝ := 2
  let commission_rate : ℝ := 0.05
  let recreation_rate : ℝ := 0.20
  let haircut_customers : ℕ := 45
  let coloring_customers : ℕ := 25
  let treatment_customers : ℕ := 10

  let total_earnings : ℝ := 
    haircut_price * haircut_customers + 
    coloring_price * coloring_customers + 
    treatment_price * treatment_customers

  let total_customers : ℕ := 
    haircut_customers + coloring_customers + treatment_customers

  let variable_expenses : ℝ := 
    product_cost_per_customer * total_customers + 
    commission_rate * total_earnings

  let total_expenses : ℝ := fixed_expenses + variable_expenses

  let net_earnings : ℝ := total_earnings - total_expenses

  let recreation_amount : ℝ := recreation_rate * total_earnings

  let monthly_savings : ℝ := net_earnings - recreation_amount

  monthly_savings = 1030 := by
    sorry

end vince_monthly_savings_l3324_332451


namespace no_special_quadrilateral_l3324_332489

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def side_lengths_different (q : Quadrilateral) : Prop := sorry

def angles_different (q : Quadrilateral) : Prop := sorry

-- Define functions to get side lengths and angles
def side_length (q : Quadrilateral) (side : Fin 4) : ℝ := sorry

def angle (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

-- Define predicates for greatest and smallest
def is_greatest_side (q : Quadrilateral) (side : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ side → side_length q side > side_length q other

def is_smallest_side (q : Quadrilateral) (side : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ side → side_length q side < side_length q other

def is_greatest_angle (q : Quadrilateral) (vertex : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ vertex → angle q vertex > angle q other

def is_smallest_angle (q : Quadrilateral) (vertex : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ vertex → angle q vertex < angle q other

-- Define the main theorem
theorem no_special_quadrilateral :
  ¬ ∃ (q : Quadrilateral) (s g a : Fin 4),
    is_convex q ∧
    side_lengths_different q ∧
    angles_different q ∧
    is_smallest_side q s ∧
    is_greatest_side q g ∧
    is_greatest_angle q a ∧
    is_smallest_angle q ((a + 2) % 4) ∧
    (a + 1) % 4 ≠ s ∧
    (a + 3) % 4 ≠ s ∧
    ((a + 2) % 4 + 1) % 4 ≠ g ∧
    ((a + 2) % 4 + 3) % 4 ≠ g :=
sorry

end no_special_quadrilateral_l3324_332489


namespace natural_number_solution_system_l3324_332401

theorem natural_number_solution_system (x y z t a b : ℕ) : 
  x^2 + y^2 = a ∧ 
  z^2 + t^2 = b ∧ 
  (x^2 + t^2) * (z^2 + y^2) = 50 →
  ((x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
   (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
   (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1)) :=
by sorry

#check natural_number_solution_system

end natural_number_solution_system_l3324_332401


namespace m_range_characterization_l3324_332486

theorem m_range_characterization (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (3 - m) * x + 1 > 0 ∨ m * x > 0) ↔ 1/9 < m ∧ m < 1 := by
sorry

end m_range_characterization_l3324_332486


namespace decagon_triangles_l3324_332417

/-- The number of vertices in a decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem decagon_triangles : choose n k = 120 := by
  sorry

end decagon_triangles_l3324_332417


namespace three_A_minus_two_B_three_A_minus_two_B_special_case_l3324_332414

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2*x^2 + 3*x*y - 2*x - 1
def B (x y : ℝ) : ℝ := -x^2 + x*y - 1

-- Theorem for the general case
theorem three_A_minus_two_B (x y : ℝ) :
  3 * A x y - 2 * B x y = 8*x^2 + 7*x*y - 6*x - 1 :=
by sorry

-- Theorem for the specific case when |x+2| + (y-1)^2 = 0
theorem three_A_minus_two_B_special_case (x y : ℝ) 
  (h : |x + 2| + (y - 1)^2 = 0) :
  3 * A x y - 2 * B x y = 29 :=
by sorry

end three_A_minus_two_B_three_A_minus_two_B_special_case_l3324_332414


namespace f_value_at_log_half_24_l3324_332497

/-- An odd function with period 2 and specific definition on (0,1) -/
def f (x : ℝ) : ℝ :=
  sorry

theorem f_value_at_log_half_24 :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 1) = f (x - 1)) →  -- f has period 2
  (∀ x ∈ Set.Ioo 0 1, f x = 2^x - 2) →  -- definition on (0,1)
  f (Real.log 24 / Real.log (1/2)) = 1/2 := by
  sorry

end f_value_at_log_half_24_l3324_332497


namespace relationship_between_a_and_b_l3324_332461

theorem relationship_between_a_and_b (a b : ℝ) 
  (ha : a > 0) (hb : b < 0) (hab : a + b < 0) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end relationship_between_a_and_b_l3324_332461


namespace susan_remaining_spaces_l3324_332454

/-- Represents a board game with a given number of spaces and a player's movements. -/
structure BoardGame where
  total_spaces : ℕ
  movements : List ℤ

/-- Calculates the remaining spaces to the end of the game. -/
def remaining_spaces (game : BoardGame) : ℕ :=
  game.total_spaces - (game.movements.sum.toNat)

/-- Susan's board game scenario -/
def susan_game : BoardGame :=
  { total_spaces := 48,
    movements := [8, -3, 6] }

/-- Theorem stating that Susan needs to move 37 spaces to reach the end -/
theorem susan_remaining_spaces :
  remaining_spaces susan_game = 37 := by
  sorry

end susan_remaining_spaces_l3324_332454


namespace population_growth_l3324_332496

theorem population_growth (initial_population : ℝ) (final_population : ℝ) (second_year_increase : ℝ) :
  initial_population = 1000 →
  final_population = 1320 →
  second_year_increase = 0.20 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 0.10 ∧
    final_population = initial_population * (1 + first_year_increase) * (1 + second_year_increase) :=
by sorry

end population_growth_l3324_332496


namespace servant_served_nine_months_l3324_332449

/-- Represents the compensation and service time of a servant --/
structure ServantCompensation where
  fullYearSalary : ℕ  -- Salary for a full year in Rupees
  uniformPrice : ℕ    -- Price of the uniform in Rupees
  receivedSalary : ℕ  -- Salary actually received in Rupees
  monthsServed : ℕ    -- Number of months served

/-- Calculates the total compensation for a full year --/
def fullYearCompensation (s : ServantCompensation) : ℕ :=
  s.fullYearSalary + s.uniformPrice

/-- Calculates the total compensation received --/
def totalReceived (s : ServantCompensation) : ℕ :=
  s.receivedSalary + s.uniformPrice

/-- Theorem stating that under given conditions, the servant served for 9 months --/
theorem servant_served_nine_months (s : ServantCompensation)
  (h1 : s.fullYearSalary = 600)
  (h2 : s.uniformPrice = 200)
  (h3 : s.receivedSalary = 400)
  : s.monthsServed = 9 := by
  sorry


end servant_served_nine_months_l3324_332449


namespace product_of_roots_equals_32_l3324_332424

theorem product_of_roots_equals_32 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by sorry

end product_of_roots_equals_32_l3324_332424


namespace right_triangle_sets_l3324_332441

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 2 ∧ b = 3 ∧ c = 4) ∧
  ¬(a^2 + b^2 = c^2) ∧
  (3^2 + 4^2 = 5^2) ∧
  (6^2 + 8^2 = 10^2) ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) :=
by sorry

end right_triangle_sets_l3324_332441


namespace movie_attendance_l3324_332490

theorem movie_attendance (total_cost concession_cost child_ticket adult_ticket : ℕ)
  (num_children : ℕ) (h1 : total_cost = 76) (h2 : concession_cost = 12)
  (h3 : child_ticket = 7) (h4 : adult_ticket = 10) (h5 : num_children = 2) :
  (total_cost - concession_cost - num_children * child_ticket) / adult_ticket = 5 := by
  sorry

end movie_attendance_l3324_332490


namespace initial_apples_count_l3324_332482

/-- Represents the number of apple trees Rachel has -/
def total_trees : ℕ := 52

/-- Represents the number of apples picked from one tree -/
def apples_picked : ℕ := 2

/-- Represents the number of apples remaining on the tree after picking -/
def apples_remaining : ℕ := 7

/-- Theorem stating that the initial number of apples on the tree is equal to
    the sum of apples remaining and apples picked -/
theorem initial_apples_count : 
  ∃ (initial_apples : ℕ), initial_apples = apples_remaining + apples_picked :=
by sorry

end initial_apples_count_l3324_332482


namespace juice_remaining_l3324_332405

theorem juice_remaining (initial : ℚ) (given : ℚ) (remaining : ℚ) : 
  initial = 5 → given = 18/7 → remaining = initial - given → remaining = 17/7 := by
  sorry

end juice_remaining_l3324_332405


namespace prime_divides_mn_minus_one_l3324_332467

theorem prime_divides_mn_minus_one (m n p : ℕ) 
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) 
  (h_p_prime : Nat.Prime p) 
  (h_m_lt_n : m < n) 
  (h_n_lt_p : n < p) 
  (h_p_div_m_sq : p ∣ (m^2 + 1)) 
  (h_p_div_n_sq : p ∣ (n^2 + 1)) : 
  p ∣ (m * n - 1) := by
  sorry

end prime_divides_mn_minus_one_l3324_332467


namespace irrational_numbers_in_set_l3324_332444

theorem irrational_numbers_in_set : 
  let S : Set ℝ := {1/3, Real.pi, 0, Real.sqrt 5}
  ∀ x ∈ S, Irrational x ↔ (x = Real.pi ∨ x = Real.sqrt 5) :=
by sorry

end irrational_numbers_in_set_l3324_332444


namespace point_on_hyperbola_l3324_332487

theorem point_on_hyperbola : 
  let x : ℝ := 2
  let y : ℝ := 3
  y = 6 / x := by sorry

end point_on_hyperbola_l3324_332487


namespace isabel_total_songs_l3324_332473

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := 2

/-- The number of jazz albums Isabel bought -/
def jazz_albums : ℕ := 4

/-- The number of rock albums Isabel bought -/
def rock_albums : ℕ := 3

/-- The number of songs in each country album -/
def songs_per_country_album : ℕ := 9

/-- The number of songs in each pop album -/
def songs_per_pop_album : ℕ := 9

/-- The number of songs in each jazz album -/
def songs_per_jazz_album : ℕ := 12

/-- The number of songs in each rock album -/
def songs_per_rock_album : ℕ := 14

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := 
  country_albums * songs_per_country_album +
  pop_albums * songs_per_pop_album +
  jazz_albums * songs_per_jazz_album +
  rock_albums * songs_per_rock_album

theorem isabel_total_songs : total_songs = 162 := by
  sorry

end isabel_total_songs_l3324_332473


namespace partnership_profit_calculation_l3324_332494

/-- Represents the profit distribution in a partnership --/
structure ProfitDistribution where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  profit_share_C : ℕ

/-- Calculates the total profit given a ProfitDistribution --/
def calculate_total_profit (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.investment_A + pd.investment_B + pd.investment_C
  let profit_per_unit := pd.profit_share_C * total_investment / pd.investment_C
  profit_per_unit

/-- Theorem stating that given the specific investments and C's profit share, the total profit is 86400 --/
theorem partnership_profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.investment_A = 12000)
  (h2 : pd.investment_B = 16000)
  (h3 : pd.investment_C = 20000)
  (h4 : pd.profit_share_C = 36000) :
  calculate_total_profit pd = 86400 := by
  sorry


end partnership_profit_calculation_l3324_332494


namespace ratio_sum_to_base_l3324_332400

theorem ratio_sum_to_base (x y : ℝ) (h : y / x = 3 / 7) : (x + y) / x = 10 / 7 := by
  sorry

end ratio_sum_to_base_l3324_332400


namespace parking_lot_cars_l3324_332431

theorem parking_lot_cars (initial_cars : ℕ) (cars_left : ℕ) (extra_cars_entered : ℕ) :
  initial_cars = 80 →
  cars_left = 13 →
  extra_cars_entered = 5 →
  initial_cars - cars_left + (cars_left + extra_cars_entered) = 85 :=
by
  sorry

end parking_lot_cars_l3324_332431


namespace beanie_tickets_l3324_332429

def arcade_tickets (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

theorem beanie_tickets : arcade_tickets 11 10 16 = 5 := by
  sorry

end beanie_tickets_l3324_332429


namespace zero_point_location_l3324_332413

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log (1/2)

-- Define the theorem
theorem zero_point_location 
  (a b c x₀ : ℝ) 
  (h1 : f a * f b * f c < 0)
  (h2 : 0 < a) (h3 : a < b) (h4 : b < c)
  (h5 : f x₀ = 0) : 
  x₀ > a := by
  sorry

end zero_point_location_l3324_332413


namespace arithmetic_sequences_bound_l3324_332485

theorem arithmetic_sequences_bound (n k b : ℕ) (d₁ d₂ : ℤ) :
  0 < b → b < n →
  (∀ i j, i ≠ j → i ≤ n → j ≤ n → ∃ (x y : ℤ), x ≠ y ∧ 
    (∃ (a r : ℤ), x = a + r * (if i ≤ b then d₁ else d₂) ∧
                  y = a + r * (if i ≤ b then d₁ else d₂)) ∧
    (∃ (a r : ℤ), x = a + r * (if j ≤ b then d₁ else d₂) ∧
                  y = a + r * (if j ≤ b then d₁ else d₂))) →
  b ≤ 2 * (k - d₂ / Int.gcd d₁ d₂) - 1 := by
sorry

end arithmetic_sequences_bound_l3324_332485


namespace least_number_with_remainder_four_l3324_332408

theorem least_number_with_remainder_four (n : ℕ) : n = 184 ↔ 
  (n > 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ m < n → 
    (m % 5 ≠ 4 ∨ m % 6 ≠ 4 ∨ m % 9 ≠ 4 ∨ m % 12 ≠ 4)) ∧
  (n % 5 = 4) ∧ (n % 6 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) :=
by sorry

end least_number_with_remainder_four_l3324_332408


namespace magnitude_of_2_plus_3i_l3324_332407

theorem magnitude_of_2_plus_3i :
  Complex.abs (2 + 3 * Complex.I) = Real.sqrt 13 := by
  sorry

end magnitude_of_2_plus_3i_l3324_332407


namespace cube_corners_equivalence_l3324_332416

/-- A corner piece consists of three 1x1x1 cubes -/
def corner_piece : ℕ := 3

/-- The dimensions of the cube -/
def cube_dimension : ℕ := 3

/-- The number of corner pieces -/
def num_corners : ℕ := 9

/-- Theorem: The total number of 1x1x1 cubes in a 3x3x3 cube 
    is equal to the total number of 1x1x1 cubes in 9 corner pieces -/
theorem cube_corners_equivalence : 
  cube_dimension ^ 3 = num_corners * corner_piece := by
  sorry

end cube_corners_equivalence_l3324_332416


namespace stratified_sampling_company_a_l3324_332404

def total_representatives : ℕ := 100
def company_a_representatives : ℕ := 40
def company_b_representatives : ℕ := 60
def total_sample_size : ℕ := 10

theorem stratified_sampling_company_a :
  (company_a_representatives * total_sample_size) / total_representatives = 4 :=
by sorry

end stratified_sampling_company_a_l3324_332404


namespace acme_cheaper_at_min_shirts_l3324_332450

/-- Acme's cost function -/
def acme_cost (x : ℕ) : ℕ := 75 + 10 * x

/-- Beta's cost function -/
def beta_cost (x : ℕ) : ℕ :=
  if x < 30 then 15 * x else 14 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme : ℕ := 20

theorem acme_cheaper_at_min_shirts :
  (∀ x < min_shirts_for_acme, beta_cost x ≤ acme_cost x) ∧
  (beta_cost min_shirts_for_acme > acme_cost min_shirts_for_acme) := by
  sorry

#check acme_cheaper_at_min_shirts

end acme_cheaper_at_min_shirts_l3324_332450


namespace alpha_squared_gt_beta_squared_l3324_332456

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
sorry

end alpha_squared_gt_beta_squared_l3324_332456


namespace range_of_2a_minus_b_l3324_332411

theorem range_of_2a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 2) (hb : 2 < b ∧ b < 3) :
  ∀ x, (∃ a b, (-2 < a ∧ a < 2) ∧ (2 < b ∧ b < 3) ∧ x = 2*a - b) ↔ -7 < x ∧ x < 2 :=
sorry

end range_of_2a_minus_b_l3324_332411


namespace average_difference_l3324_332468

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 60) : 
  c - a = 30 := by
  sorry

end average_difference_l3324_332468


namespace no_solution_for_a_l3324_332440

theorem no_solution_for_a (x : ℝ) (h : x = 4) :
  ¬∃ a : ℝ, a / (x + 4) + a / (x - 4) = a / (x - 4) :=
sorry

end no_solution_for_a_l3324_332440


namespace escalator_speed_l3324_332434

/-- The speed of an escalator given its length, a person's walking speed, and the time taken to cover the entire length. -/
theorem escalator_speed (escalator_length : ℝ) (walking_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 126 ∧ walking_speed = 3 ∧ time_taken = 9 →
  ∃ (escalator_speed : ℝ), 
    escalator_speed = 11 ∧ 
    (escalator_speed + walking_speed) * time_taken = escalator_length :=
by sorry

end escalator_speed_l3324_332434


namespace calculate_expression_l3324_332443

theorem calculate_expression : 12 - (-18) + (-7) = 23 := by
  sorry

end calculate_expression_l3324_332443


namespace mari_made_64_buttons_l3324_332499

/-- Given the number of buttons Sue made -/
def sue_buttons : ℕ := 6

/-- Kendra's buttons in terms of Sue's -/
def kendra_buttons : ℕ := 2 * sue_buttons

/-- Mari's buttons in terms of Kendra's -/
def mari_buttons : ℕ := 4 + 5 * kendra_buttons

/-- Theorem stating that Mari made 64 buttons -/
theorem mari_made_64_buttons : mari_buttons = 64 := by
  sorry

end mari_made_64_buttons_l3324_332499


namespace sum_two_smallest_prime_factors_250_l3324_332436

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → p ≤ q}

theorem sum_two_smallest_prime_factors_250 :
  ∃ (a b : ℕ), a ∈ smallest_prime_factors 250 ∧ 
               b ∈ smallest_prime_factors 250 ∧ 
               a ≠ b ∧
               a + b = 7 :=
sorry

end sum_two_smallest_prime_factors_250_l3324_332436


namespace no_solution_exists_l3324_332418

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (heq : x^y + 1 = z^2) : False :=
sorry

end no_solution_exists_l3324_332418


namespace area_triangle_parallel_lines_circle_l3324_332483

/-- Given two parallel lines with distance x between them, where one line is tangent 
    to a circle of radius R at point A and the other line intersects the circle at 
    points B and C, the area S of triangle ABC is equal to x √(2Rx - x²). -/
theorem area_triangle_parallel_lines_circle (R x : ℝ) (h : 0 < R ∧ 0 < x ∧ x < 2*R) :
  ∃ (S : ℝ), S = x * Real.sqrt (2 * R * x - x^2) := by
  sorry

end area_triangle_parallel_lines_circle_l3324_332483


namespace total_wheels_count_l3324_332430

/-- Represents the total number of wheels in Jordan's driveway -/
def total_wheels : ℕ :=
  let cars := 2
  let car_wheels := 4
  let bikes := 3
  let bike_wheels := 2
  let trash_can_wheels := 2
  let tricycle_wheels := 3
  let roller_skate_wheels := 4
  let wheelchair_wheels := 6
  let wagon_wheels := 4
  
  cars * car_wheels +
  (bikes - 1) * bike_wheels + 1 +
  trash_can_wheels +
  tricycle_wheels +
  (roller_skate_wheels - 1) +
  wheelchair_wheels +
  wagon_wheels

theorem total_wheels_count : total_wheels = 31 := by
  sorry

end total_wheels_count_l3324_332430


namespace even_increasing_inequality_l3324_332458

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_left (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y ∧ y ≤ -1 → f x ≤ f y

-- State the theorem
theorem even_increasing_inequality 
  (h_even : is_even f) 
  (h_incr : increasing_on_left f) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) := by
sorry

end even_increasing_inequality_l3324_332458


namespace fiona_peeled_22_l3324_332448

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  martin_rate : ℕ
  fiona_rate : ℕ
  fiona_join_time : ℕ

/-- Calculates the number of potatoes Fiona peeled -/
def fiona_peeled (scenario : PotatoPeeling) : ℕ :=
  let martin_peeled := scenario.martin_rate * scenario.fiona_join_time
  let remaining := scenario.total_potatoes - martin_peeled
  let combined_rate := scenario.martin_rate + scenario.fiona_rate
  let combined_time := (remaining + combined_rate - 1) / combined_rate -- Ceiling division
  scenario.fiona_rate * combined_time

/-- Theorem stating that Fiona peeled 22 potatoes -/
theorem fiona_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.martin_rate = 4)
  (h3 : scenario.fiona_rate = 6)
  (h4 : scenario.fiona_join_time = 6) :
  fiona_peeled scenario = 22 := by
  sorry

#eval fiona_peeled { total_potatoes := 60, martin_rate := 4, fiona_rate := 6, fiona_join_time := 6 }

end fiona_peeled_22_l3324_332448


namespace parrots_left_on_branch_l3324_332406

/-- Represents the number of birds on a tree branch -/
structure BirdCount where
  parrots : ℕ
  crows : ℕ

/-- The initial state of birds on the branch -/
def initialState : BirdCount where
  parrots := 7
  crows := 13 - 7

/-- The number of birds that flew away -/
def flownAway : ℕ :=
  initialState.crows - 1

/-- The final state of birds on the branch -/
def finalState : BirdCount where
  parrots := initialState.parrots - flownAway
  crows := 1

theorem parrots_left_on_branch :
  finalState.parrots = 2 :=
sorry

end parrots_left_on_branch_l3324_332406


namespace factorial_500_trailing_zeroes_l3324_332469

def trailing_zeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeroes :
  trailing_zeroes 500 = 124 := by
  sorry

end factorial_500_trailing_zeroes_l3324_332469


namespace ten_thousandths_place_of_5_over_32_l3324_332476

theorem ten_thousandths_place_of_5_over_32 : 
  ∃ (a b c d : ℕ), (5 : ℚ) / 32 = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (6 : ℚ) / 10000 + (d : ℚ) / 100000 :=
by sorry

end ten_thousandths_place_of_5_over_32_l3324_332476


namespace exists_node_not_on_line_l3324_332419

/-- Represents a node on the grid --/
structure Node :=
  (x : Nat) (y : Nat)

/-- Represents a polygonal line on the grid --/
structure Line :=
  (nodes : List Node)

/-- The grid size --/
def gridSize : Nat := 100

/-- Checks if a node is on the boundary of the grid --/
def isOnBoundary (n : Node) : Bool :=
  n.x = 0 || n.x = gridSize || n.y = 0 || n.y = gridSize

/-- Checks if a node is a corner of the grid --/
def isCorner (n : Node) : Bool :=
  (n.x = 0 && n.y = 0) || (n.x = 0 && n.y = gridSize) ||
  (n.x = gridSize && n.y = 0) || (n.x = gridSize && n.y = gridSize)

/-- Theorem: There exists a non-corner node not on any line --/
theorem exists_node_not_on_line (lines : List Line) : 
  ∃ (n : Node), !isCorner n ∧ ∀ (l : Line), l ∈ lines → n ∉ l.nodes :=
sorry


end exists_node_not_on_line_l3324_332419
