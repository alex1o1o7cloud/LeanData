import Mathlib

namespace NUMINAMATH_CALUDE_tire_price_proof_l2518_251832

/-- The regular price of a tire -/
def regular_price : ℝ := 115.71

/-- The total amount paid for four tires -/
def total_paid : ℝ := 405

/-- The promotion deal: 3 tires at regular price, 1 at half price -/
theorem tire_price_proof :
  3 * regular_price + (1/2) * regular_price = total_paid :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l2518_251832


namespace NUMINAMATH_CALUDE_willies_bananas_unchanged_l2518_251805

/-- Willie's banana count remains unchanged regardless of Charles' banana count changes -/
theorem willies_bananas_unchanged (willie_initial : ℕ) (charles_initial charles_lost : ℕ) :
  willie_initial = 48 → willie_initial = willie_initial :=
by
  sorry

end NUMINAMATH_CALUDE_willies_bananas_unchanged_l2518_251805


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2518_251830

theorem quadratic_equation_roots : ∃ x : ℝ, (∀ y : ℝ, -y^2 + 2*y - 1 = 0 ↔ y = x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2518_251830


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l2518_251868

theorem sum_x_y_equals_negative_one (x y : ℝ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : 5 * x + 3 * y = 1) : 
  x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l2518_251868


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2518_251882

def U : Set Int := {-2, -1, 1, 3, 5}
def A : Set Int := {-1, 3}

theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {-2, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2518_251882


namespace NUMINAMATH_CALUDE_hockey_league_games_l2518_251813

theorem hockey_league_games (n : ℕ) (m : ℕ) (total_games : ℕ) : 
  n = 25 → -- number of teams
  m = 15 → -- number of times each team faces every other team
  total_games = (n * (n - 1) / 2) * m →
  total_games = 4500 :=
by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2518_251813


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2518_251888

/-- Calculates the amount after two years of compound interest with different rates for each year. -/
def amountAfterTwoYears (initialAmount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amountAfterFirstYear := initialAmount * (1 + rate1)
  amountAfterFirstYear * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated. -/
theorem compound_interest_calculation (initialAmount : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : initialAmount = 9828) 
  (h2 : rate1 = 0.04) 
  (h3 : rate2 = 0.05) :
  amountAfterTwoYears initialAmount rate1 rate2 = 10732.176 := by
  sorry

#eval amountAfterTwoYears 9828 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l2518_251888


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l2518_251899

/-- Represents the number of valid sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n+2 => validSequences (n+1) + validSequences n

/-- The length of the sequence -/
def sequenceLength : ℕ := 12

/-- The total number of possible sequences -/
def totalSequences : ℕ := 2^sequenceLength

theorem probability_no_consecutive_ones :
  (validSequences sequenceLength : ℚ) / totalSequences = 377 / 4096 := by
  sorry

#eval validSequences sequenceLength + totalSequences

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l2518_251899


namespace NUMINAMATH_CALUDE_tea_cheese_ratio_l2518_251878

/-- Represents the prices of items in Ursula's purchase -/
structure PurchasePrices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ

/-- The conditions of Ursula's purchase -/
def purchase_conditions (p : PurchasePrices) : Prop :=
  p.butter + p.bread + p.cheese + p.tea = 21 ∧
  p.bread = p.butter / 2 ∧
  p.butter = 0.8 * p.cheese ∧
  p.tea = 10

/-- The theorem stating the ratio of tea price to cheese price -/
theorem tea_cheese_ratio (p : PurchasePrices) :
  purchase_conditions p → p.tea / p.cheese = 2 := by
  sorry

end NUMINAMATH_CALUDE_tea_cheese_ratio_l2518_251878


namespace NUMINAMATH_CALUDE_expression_evaluation_l2518_251812

theorem expression_evaluation :
  Real.sqrt 8 + (1/2)⁻¹ - 2 * Real.sin (45 * π / 180) - abs (1 - Real.sqrt 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2518_251812


namespace NUMINAMATH_CALUDE_student_speaking_probability_l2518_251852

/-- The probability of a student speaking truth -/
def prob_truth : ℝ := 0.30

/-- The probability of a student speaking lie -/
def prob_lie : ℝ := 0.20

/-- The probability of a student speaking both truth and lie -/
def prob_both : ℝ := 0.10

/-- The probability of a student speaking either truth or lie -/
def prob_truth_or_lie : ℝ := prob_truth + prob_lie - prob_both

theorem student_speaking_probability :
  prob_truth_or_lie = 0.40 := by sorry

end NUMINAMATH_CALUDE_student_speaking_probability_l2518_251852


namespace NUMINAMATH_CALUDE_milk_for_six_cookies_l2518_251844

/-- Calculates the amount of milk needed for a given number of cookies. -/
def milk_needed (cookies : ℕ) : ℚ :=
  (5000 : ℚ) * cookies / 24

theorem milk_for_six_cookies :
  milk_needed 6 = 1250 := by sorry


end NUMINAMATH_CALUDE_milk_for_six_cookies_l2518_251844


namespace NUMINAMATH_CALUDE_remove_32_toothpicks_eliminates_triangles_l2518_251887

/-- A triangular figure constructed with toothpicks -/
structure TriangularFigure where
  toothpicks : ℕ
  triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (f : TriangularFigure) : ℕ :=
  f.horizontal_toothpicks

/-- Theorem stating that removing 32 toothpicks is sufficient to eliminate all triangles 
    in a specific triangular figure -/
theorem remove_32_toothpicks_eliminates_triangles (f : TriangularFigure) 
  (h1 : f.toothpicks = 42)
  (h2 : f.triangles > 35)
  (h3 : f.horizontal_toothpicks = 32) :
  min_toothpicks_to_remove f = 32 := by
  sorry

end NUMINAMATH_CALUDE_remove_32_toothpicks_eliminates_triangles_l2518_251887


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l2518_251880

theorem shoe_price_calculation (discount_rate : ℝ) (savings : ℝ) (original_price : ℝ) : 
  discount_rate = 0.30 →
  savings = 46 →
  original_price = savings / discount_rate →
  original_price = 153.33 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l2518_251880


namespace NUMINAMATH_CALUDE_number_divided_by_16_equals_4_l2518_251862

theorem number_divided_by_16_equals_4 (x : ℤ) : x / 16 = 4 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_16_equals_4_l2518_251862


namespace NUMINAMATH_CALUDE_polynomial_not_factorizable_l2518_251853

theorem polynomial_not_factorizable :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ),
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = x^200 * y^200 + 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_not_factorizable_l2518_251853


namespace NUMINAMATH_CALUDE_same_color_probability_l2518_251820

theorem same_color_probability (total : ℕ) (black white : ℕ) 
  (h1 : (black * (black - 1)) / (total * (total - 1)) = 1 / 7)
  (h2 : (white * (white - 1)) / (total * (total - 1)) = 12 / 35) :
  ((black * (black - 1)) + (white * (white - 1))) / (total * (total - 1)) = 17 / 35 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l2518_251820


namespace NUMINAMATH_CALUDE_knight_returns_to_start_l2518_251871

/-- A castle in Mara -/
structure Castle where
  id : ℕ

/-- The graph of castles and roads in Mara -/
structure MaraGraph where
  castles : Set Castle
  roads : Castle → Set Castle
  finite_castles : Set.Finite castles
  three_roads : ∀ c, (roads c).ncard = 3

/-- A turn direction -/
inductive Turn
| Left
| Right

/-- A path through the castles -/
structure KnightPath (G : MaraGraph) where
  path : ℕ → Castle
  turns : ℕ → Turn
  valid_path : ∀ n, G.roads (path n) (path (n + 1))
  alternating_turns : ∀ n, turns n ≠ turns (n + 1)

/-- The theorem stating that the knight will return to the original castle -/
theorem knight_returns_to_start (G : MaraGraph) (p : KnightPath G) :
  ∃ n m, n < m ∧ p.path n = p.path m := by sorry

end NUMINAMATH_CALUDE_knight_returns_to_start_l2518_251871


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2518_251856

theorem quadratic_equation_coefficients :
  ∀ (x : ℝ), 3 * x^2 + 1 = 5 * x ↔ 3 * x^2 + (-5) * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2518_251856


namespace NUMINAMATH_CALUDE_greatest_divisor_of_consecutive_multiples_of_four_l2518_251855

theorem greatest_divisor_of_consecutive_multiples_of_four : ∃ (k : ℕ), 
  k > 0 ∧ 
  (∀ (n : ℕ), 
    (4*n * 4*(n+1) * 4*(n+2)) % k = 0) ∧
  (∀ (m : ℕ), 
    m > k → 
    ∃ (n : ℕ), (4*n * 4*(n+1) * 4*(n+2)) % m ≠ 0) ∧
  k = 768 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_consecutive_multiples_of_four_l2518_251855


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_squares_multiple_of_360_l2518_251822

theorem smallest_k_for_sum_squares_multiple_of_360 :
  ∃ k : ℕ+, (k.val * (k.val + 1) * (2 * k.val + 1)) % 2160 = 0 ∧
  ∀ m : ℕ+, m < k → (m.val * (m.val + 1) * (2 * m.val + 1)) % 2160 ≠ 0 ∧
  k = 175 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_squares_multiple_of_360_l2518_251822


namespace NUMINAMATH_CALUDE_quadruple_batch_cans_l2518_251865

/-- Represents the number of cans of each ingredient in a normal batch of chili -/
structure ChiliBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Calculates the total number of cans in a batch of chili -/
def totalCans (batch : ChiliBatch) : ℕ :=
  batch.chilis + batch.beans + batch.tomatoes

/-- Defines a normal batch of chili according to Carla's recipe -/
def normalBatch : ChiliBatch :=
  { chilis := 1
  , beans := 2
  , tomatoes := 3 }  -- 50% more than beans

/-- Theorem: A quadruple batch of Carla's chili requires 24 cans -/
theorem quadruple_batch_cans :
  4 * (totalCans normalBatch) = 24 := by
  sorry


end NUMINAMATH_CALUDE_quadruple_batch_cans_l2518_251865


namespace NUMINAMATH_CALUDE_group_leader_selection_l2518_251858

theorem group_leader_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_group_leader_selection_l2518_251858


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2518_251800

theorem polynomial_factorization (a x : ℝ) : 
  a * x^3 + x + a + 1 = (x + 1) * (a * x^2 - a * x + a + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2518_251800


namespace NUMINAMATH_CALUDE_atomic_weight_Br_l2518_251875

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := 137.33

/-- The molecular weight of the compound -/
def molecular_weight : ℝ := 297

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def num_Br : ℕ := 2

/-- Theorem: The atomic weight of Bromine (Br) is 79.835 -/
theorem atomic_weight_Br :
  let x := (molecular_weight - num_Ba * atomic_weight_Ba) / num_Br
  x = 79.835 := by sorry

end NUMINAMATH_CALUDE_atomic_weight_Br_l2518_251875


namespace NUMINAMATH_CALUDE_complement_of_union_is_empty_l2518_251828

def U : Finset Nat := {1, 2, 3, 4}
def A : Finset Nat := {1, 3, 4}
def B : Finset Nat := {2, 3, 4}

theorem complement_of_union_is_empty :
  (U \ (A ∪ B) : Finset Nat) = ∅ := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_empty_l2518_251828


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l2518_251804

theorem hippopotamus_crayons (initial_crayons final_crayons : ℕ) 
  (h1 : initial_crayons = 87) 
  (h2 : final_crayons = 80) : 
  initial_crayons - final_crayons = 7 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_crayons_l2518_251804


namespace NUMINAMATH_CALUDE_prob_one_defective_out_of_two_l2518_251807

/-- The probability of selecting exactly one defective product when randomly choosing 2 out of 5 products, where 2 are defective and 3 are qualified. -/
theorem prob_one_defective_out_of_two (total : Nat) (defective : Nat) (selected : Nat) : 
  total = 5 → defective = 2 → selected = 2 → 
  (Nat.choose defective 1 * Nat.choose (total - defective) (selected - 1)) / Nat.choose total selected = 3/5 := by
sorry

end NUMINAMATH_CALUDE_prob_one_defective_out_of_two_l2518_251807


namespace NUMINAMATH_CALUDE_f_min_at_three_l2518_251814

/-- The quadratic function f(x) = x^2 - 6x + 8 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l2518_251814


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2518_251819

-- Define the geometric sequence and its sum
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 2

-- Define the general formula for the sequence
def general_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 * (3 ^ (n - 1))

-- Theorem statement
theorem geometric_sequence_formula 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : geometric_sequence a S) : 
  general_formula a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2518_251819


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l2518_251872

theorem quadratic_perfect_square (x : ℝ) (d : ℝ) :
  (∃ b : ℝ, ∀ x, x^2 + 60*x + d = (x + b)^2) ↔ d = 900 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l2518_251872


namespace NUMINAMATH_CALUDE_ratio_problem_l2518_251886

/-- Given ratios for x, y, and z, prove their values -/
theorem ratio_problem (x y z : ℚ) : 
  (x / 12 = 5 / 1) → 
  (y / 21 = 7 / 3) → 
  (z / 16 = 4 / 2) → 
  (x = 60 ∧ y = 49 ∧ z = 32) :=
by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2518_251886


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2518_251866

theorem marble_selection_ways (total_marbles : ℕ) (selected_marbles : ℕ) (blue_marble : ℕ) :
  total_marbles = 10 →
  selected_marbles = 4 →
  blue_marble = 1 →
  (total_marbles.choose (selected_marbles - blue_marble)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2518_251866


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_two_zeros_l2518_251864

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with fewer than two zeros -/
def fewer_than_two_zeros : ℕ := 826686

/-- The number of 6-digit numbers with at least two zeros -/
def at_least_two_zeros : ℕ := total_six_digit_numbers - fewer_than_two_zeros

theorem six_digit_numbers_with_two_zeros :
  at_least_two_zeros = 73314 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_two_zeros_l2518_251864


namespace NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_equals_zero_l2518_251869

/-- The minimum value of max_{0 ≤ x ≤ 2} |x^2 - 2xy| for y in ℝ is 0 -/
theorem min_max_abs_x_squared_minus_2xy_equals_zero :
  ∃ y : ℝ, ∀ y' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x^2 - 2*x*y| ≤ |x^2 - 2*x*y'|) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_equals_zero_l2518_251869


namespace NUMINAMATH_CALUDE_count_solution_pairs_l2518_251846

/-- The number of ordered pairs (a, b) of complex numbers satisfying the given equations -/
def solution_count : ℕ := 24

/-- The predicate defining the condition for a pair of complex numbers -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^4 * b^6 = 1 ∧ a^8 * b^3 = 1

theorem count_solution_pairs :
  (∃! (s : Finset (ℂ × ℂ)), s.card = solution_count ∧ 
   ∀ p ∈ s, satisfies_equations p.1 p.2 ∧
   ∀ a b, satisfies_equations a b → (a, b) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l2518_251846


namespace NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l2518_251849

-- Define the criteria for a set
structure SetCriteria where
  definiteness : Bool
  distinctness : Bool
  unorderedness : Bool

-- Define a predicate for whether a collection can form a set
def canFormSet (c : SetCriteria) : Bool :=
  c.definiteness ∧ c.distinctness ∧ c.unorderedness

-- Define the property of being "close to 0"
def closeToZero (ε : ℝ) (x : ℝ) : Prop := abs x < ε

-- Theorem stating that "Numbers close to 0" cannot form a set
theorem numbers_close_to_zero_not_set : 
  ∃ ε > 0, ¬∃ (S : Set ℝ), (∀ x ∈ S, closeToZero ε x) ∧ 
  (canFormSet ⟨true, true, true⟩) :=
sorry

end NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l2518_251849


namespace NUMINAMATH_CALUDE_circle_center_distance_l2518_251857

theorem circle_center_distance (x y : ℝ) : 
  (x^2 + y^2 = 6*x + 8*y + 9) → 
  Real.sqrt ((11 - x)^2 + (5 - y)^2) = Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_circle_center_distance_l2518_251857


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2518_251829

/-- Proves that the ratio of Tom's current age to the number of years ago when his age was three times the sum of his children's ages is 5.5 -/
theorem toms_age_ratio :
  ∀ (T N : ℝ),
  (∃ (a b c d : ℝ), T = a + b + c + d) →  -- T is the sum of four children's ages
  (T - N = 3 * (T - 4 * N)) →              -- N years ago condition
  T / N = 5.5 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2518_251829


namespace NUMINAMATH_CALUDE_point_three_units_away_l2518_251859

theorem point_three_units_away (A : ℝ) (h : A = 2) :
  ∀ B : ℝ, abs (B - A) = 3 → (B = -1 ∨ B = 5) :=
by sorry

end NUMINAMATH_CALUDE_point_three_units_away_l2518_251859


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2518_251825

/-- Given a two-digit number, constructs a six-digit number by repeating it three times -/
def repeat_twice (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + n

/-- Theorem: For any two-digit number, the six-digit number formed by repeating it three times is divisible by 10101 -/
theorem six_digit_divisibility (n : ℕ) (h : n ≥ 10 ∧ n ≤ 99) : 
  (repeat_twice n) % 10101 = 0 := by
  sorry


end NUMINAMATH_CALUDE_six_digit_divisibility_l2518_251825


namespace NUMINAMATH_CALUDE_area_of_triangle_with_given_conditions_l2518_251802

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C : ℝ × ℝ)

structure TriangleWithPoint extends Triangle :=
  (P : ℝ × ℝ)

-- Define the conditions
def isScaleneRightTriangle (t : Triangle) : Prop := sorry

def isPointOnHypotenuse (t : TriangleWithPoint) : Prop := sorry

def angleABP30 (t : TriangleWithPoint) : Prop := sorry

def lengthAP3 (t : TriangleWithPoint) : Prop := sorry

def lengthCP1 (t : TriangleWithPoint) : Prop := sorry

-- Define the area function
def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_with_given_conditions (t : TriangleWithPoint) 
  (h1 : isScaleneRightTriangle t.toTriangle)
  (h2 : isPointOnHypotenuse t)
  (h3 : angleABP30 t)
  (h4 : lengthAP3 t)
  (h5 : lengthCP1 t) :
  triangleArea t.toTriangle = 12/5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_with_given_conditions_l2518_251802


namespace NUMINAMATH_CALUDE_dislike_tv_and_books_l2518_251861

/-- Given a population where some dislike TV and some of those also dislike books,
    calculate the number of people who dislike both TV and books. -/
theorem dislike_tv_and_books
  (total_population : ℕ)
  (tv_dislike_percent : ℚ)
  (book_dislike_percent : ℚ)
  (h_total : total_population = 1500)
  (h_tv : tv_dislike_percent = 25 / 100)
  (h_book : book_dislike_percent = 15 / 100) :
  ⌊(tv_dislike_percent * book_dislike_percent * total_population : ℚ)⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_dislike_tv_and_books_l2518_251861


namespace NUMINAMATH_CALUDE_zeroes_elimination_theorem_l2518_251879

/-- A step in the digit replacement process. -/
structure Step where
  digits_removed : ℕ := 2

/-- The initial state of the blackboard. -/
structure Blackboard where
  zeroes : ℕ
  ones : ℕ

/-- The final state after all steps are completed. -/
structure FinalState where
  steps : ℕ
  remaining_ones : ℕ

/-- The theorem to be proved. -/
theorem zeroes_elimination_theorem (initial : Blackboard) (final : FinalState) :
  initial.zeroes = 150 ∧
  final.steps = 76 ∧
  final.remaining_ones = initial.ones - 2 →
  initial.ones = 78 :=
by sorry

end NUMINAMATH_CALUDE_zeroes_elimination_theorem_l2518_251879


namespace NUMINAMATH_CALUDE_james_injury_timeline_l2518_251893

/-- The number of days it took for James's pain to subside -/
def pain_subsided_days : ℕ := 3

/-- The total number of days until James can lift heavy again -/
def total_days : ℕ := 39

/-- The number of additional days James waits after the injury is fully healed -/
def additional_waiting_days : ℕ := 3

/-- The number of days (3 weeks) James waits before lifting heavy -/
def heavy_lifting_wait_days : ℕ := 21

theorem james_injury_timeline : 
  pain_subsided_days * 5 + pain_subsided_days + additional_waiting_days + heavy_lifting_wait_days = total_days :=
by sorry

end NUMINAMATH_CALUDE_james_injury_timeline_l2518_251893


namespace NUMINAMATH_CALUDE_equation_solution_l2518_251840

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem
theorem equation_solution (x y : ℝ) :
  2 * (f x) - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2518_251840


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2518_251848

theorem polynomial_divisibility (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  (x - y)^5 + (y - z)^5 + (z - x)^5 = 
  -5 * (x - y) * (y - z) * (z - x) * ((x - y)^2 + (x - y) * (y - z) + (y - z)^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2518_251848


namespace NUMINAMATH_CALUDE_embankment_construction_time_l2518_251837

/-- Given that 60 workers take 3 days to build half of an embankment,
    prove that 45 workers would take 8 days to build the entire embankment,
    assuming all workers work at the same rate. -/
theorem embankment_construction_time
  (workers_60 : ℕ) (days_60 : ℕ) (half_embankment : ℚ)
  (workers_45 : ℕ) (days_45 : ℕ) (full_embankment : ℚ)
  (h1 : workers_60 = 60)
  (h2 : days_60 = 3)
  (h3 : half_embankment = 1/2)
  (h4 : workers_45 = 45)
  (h5 : days_45 = 8)
  (h6 : full_embankment = 1)
  (h7 : ∀ w d, w * d * half_embankment = workers_60 * days_60 * half_embankment →
               w * d * full_embankment = workers_45 * days_45 * full_embankment) :
  workers_45 * days_45 * full_embankment = workers_60 * days_60 * full_embankment :=
by sorry

end NUMINAMATH_CALUDE_embankment_construction_time_l2518_251837


namespace NUMINAMATH_CALUDE_group_meal_cost_l2518_251811

/-- The cost of a meal for a group at a restaurant -/
def mealCost (totalPeople : ℕ) (kids : ℕ) (adultMealPrice : ℕ) : ℕ :=
  (totalPeople - kids) * adultMealPrice

/-- Theorem: The meal cost for a group of 13 people with 9 kids is $28 -/
theorem group_meal_cost : mealCost 13 9 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_l2518_251811


namespace NUMINAMATH_CALUDE_largest_power_dividing_product_l2518_251806

-- Define pow function
def pow (n : ℕ) : ℕ := sorry

-- Define the product of pow(n) from 2 to 5300
def product : ℕ := sorry

-- State the theorem
theorem largest_power_dividing_product : 
  (∃ m : ℕ, (2010 ^ m : ℕ) ∣ product ∧ 
   ∀ k : ℕ, k > m → ¬((2010 ^ k : ℕ) ∣ product)) ∧ 
  (∃ m : ℕ, m = 77 ∧ (2010 ^ m : ℕ) ∣ product ∧ 
   ∀ k : ℕ, k > m → ¬((2010 ^ k : ℕ) ∣ product)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_dividing_product_l2518_251806


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l2518_251847

/-- The largest power of 3 that divides n! -/
def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27)

/-- The ones digit of 3^n -/
def onesDigitOf3ToPower (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is impossible, but Lean requires it

theorem ones_digit_of_largest_power_of_3_dividing_27_factorial :
  onesDigitOf3ToPower (largestPowerOf3DividingFactorial 27) = 3 := by
  sorry

#eval onesDigitOf3ToPower (largestPowerOf3DividingFactorial 27)

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l2518_251847


namespace NUMINAMATH_CALUDE_trigonometric_sum_zero_l2518_251821

theorem trigonometric_sum_zero (x y z : ℝ) 
  (h1 : Real.cos (2 * x) + 2 * Real.cos (2 * y) + 3 * Real.cos (2 * z) = 0)
  (h2 : Real.sin (2 * x) + 2 * Real.sin (2 * y) + 3 * Real.sin (2 * z) = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_zero_l2518_251821


namespace NUMINAMATH_CALUDE_ellipse_and_chord_l2518_251850

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about an ellipse and a chord -/
theorem ellipse_and_chord 
  (C : Ellipse) 
  (h_ecc : C.a * C.a - C.b * C.b = (C.a * C.a) / 4) 
  (h_point : (2 : ℝ) * (2 : ℝ) / (C.a * C.a) + (-3 : ℝ) * (-3 : ℝ) / (C.b * C.b) = 1)
  (M : Point) 
  (h_M : M.x = -1 ∧ M.y = 2) :
  (∃ (D : Ellipse), D.a * D.a = 16 ∧ D.b * D.b = 12) ∧
  (∃ (l : Line), l.a = 3 ∧ l.b = -8 ∧ l.c = 19) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_chord_l2518_251850


namespace NUMINAMATH_CALUDE_derivative_log_base_3_l2518_251854

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem derivative_log_base_3 (x : ℝ) (h : x > 0) :
  deriv f x = 1 / (x * Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_log_base_3_l2518_251854


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2518_251841

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2518_251841


namespace NUMINAMATH_CALUDE_death_rate_per_two_seconds_prove_death_rate_l2518_251870

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℕ := 4

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 86400

/-- Theorem stating the death rate in people per two seconds -/
theorem death_rate_per_two_seconds : ℕ :=
  2

/-- Proof of the death rate given the birth rate and net population increase -/
theorem prove_death_rate : death_rate_per_two_seconds = 2 := by
  sorry


end NUMINAMATH_CALUDE_death_rate_per_two_seconds_prove_death_rate_l2518_251870


namespace NUMINAMATH_CALUDE_max_value_theorem_l2518_251839

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 ≤ 4) :
  ∃ (M : ℝ), M = 12 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 ≤ 4 → |a+2*b-4| + |3-a-b| ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2518_251839


namespace NUMINAMATH_CALUDE_triangle_area_is_integer_l2518_251810

-- Define a point in the plane
structure Point where
  x : Int
  y : Int

-- Define a function to check if a number is odd
def isOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Define a triangle with three points
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Define the area of a triangle
def triangleArea (t : Triangle) : Rat :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  Rat.ofInt (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

-- Theorem statement
theorem triangle_area_is_integer (t : Triangle) :
  t.p1 = Point.mk 1 1 →
  (isOdd t.p2.x ∧ isOdd t.p2.y) →
  (isOdd t.p3.x ∧ isOdd t.p3.y) →
  t.p2 ≠ t.p3 →
  ∃ n : Int, triangleArea t = n := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_integer_l2518_251810


namespace NUMINAMATH_CALUDE_johns_cows_value_increase_l2518_251892

/-- Calculates the increase in value of cows after weight gain -/
def cow_value_increase (initial_weights : Fin 3 → ℝ) (increase_factors : Fin 3 → ℝ) (price_per_pound : ℝ) : ℝ :=
  let new_weights := fun i => initial_weights i * increase_factors i
  let initial_values := fun i => initial_weights i * price_per_pound
  let new_values := fun i => new_weights i * price_per_pound
  (Finset.sum Finset.univ new_values) - (Finset.sum Finset.univ initial_values)

/-- The increase in value of John's cows after weight gain -/
theorem johns_cows_value_increase :
  let initial_weights : Fin 3 → ℝ := ![732, 845, 912]
  let increase_factors : Fin 3 → ℝ := ![1.35, 1.28, 1.4]
  let price_per_pound : ℝ := 2.75
  cow_value_increase initial_weights increase_factors price_per_pound = 2358.40 := by
  sorry

end NUMINAMATH_CALUDE_johns_cows_value_increase_l2518_251892


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l2518_251808

theorem equilateral_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
fun equilateral_side isosceles_perimeter isosceles_base =>
  let isosceles_side := equilateral_side
  let equilateral_perimeter := 3 * equilateral_side
  isosceles_perimeter = 2 * isosceles_side + isosceles_base ∧
  isosceles_perimeter = 40 ∧
  isosceles_base = 10 →
  equilateral_perimeter = 45

-- The proof would go here, but we'll skip it as requested
theorem equilateral_triangle_perimeter_proof :
  equilateral_triangle_perimeter 15 40 10 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l2518_251808


namespace NUMINAMATH_CALUDE_fruit_store_profit_l2518_251823

-- Define the cost price
def cost_price : ℝ := 40

-- Define the linear function for weekly sales quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_quantity x

-- Define the new profit function with increased cost
def new_profit (x m : ℝ) : ℝ := (x - cost_price - m) * sales_quantity x

theorem fruit_store_profit :
  -- 1. The selling price that maximizes profit is 70 yuan/kg
  (∀ x : ℝ, profit x ≤ profit 70) ∧
  -- 2. The maximum profit is 1800 yuan
  (profit 70 = 1800) ∧
  -- 3. When the cost price increases by m yuan/kg (m > 0), and the profit decreases
  --    for selling prices > 76 yuan/kg, then 0 < m ≤ 12
  (∀ m : ℝ, m > 0 →
    (∀ x : ℝ, x > 76 → (∀ y : ℝ, y > x → new_profit y m < new_profit x m)) →
    m ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_fruit_store_profit_l2518_251823


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2518_251890

/-- Given a right circular cone with diameter 14 and altitude 16, and an inscribed right circular
cylinder whose height is twice its radius, the radius of the cylinder is 56/15. -/
theorem inscribed_cylinder_radius (cone_diameter : ℝ) (cone_altitude : ℝ) (cylinder_radius : ℝ) :
  cone_diameter = 14 →
  cone_altitude = 16 →
  (∃ (cylinder_height : ℝ), cylinder_height = 2 * cylinder_radius) →
  cylinder_radius = 56 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2518_251890


namespace NUMINAMATH_CALUDE_square_side_length_l2518_251835

theorem square_side_length (x : ℝ) (h : x > 0) : 4 * x = 2 * (x ^ 2) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2518_251835


namespace NUMINAMATH_CALUDE_product_divisibility_l2518_251891

theorem product_divisibility : ∃ k : ℕ, 86 * 87 * 88 * 89 * 90 * 91 * 92 = 7 * k := by
  sorry

#check product_divisibility

end NUMINAMATH_CALUDE_product_divisibility_l2518_251891


namespace NUMINAMATH_CALUDE_gcd_11121_12012_l2518_251809

theorem gcd_11121_12012 : Nat.gcd 11121 12012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_11121_12012_l2518_251809


namespace NUMINAMATH_CALUDE_car_speed_l2518_251895

theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 624 ∧ time = 2 + 2/5 → speed = distance / time → speed = 260 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l2518_251895


namespace NUMINAMATH_CALUDE_employee_work_hours_l2518_251801

/-- The number of hours worked per week by both employees -/
def hours_per_week : ℕ := 40

/-- The hourly rate of the first employee -/
def rate1 : ℚ := 20

/-- The hourly rate of the second employee -/
def rate2 : ℚ := 22

/-- The hourly government subsidy for the second employee -/
def subsidy : ℚ := 6

/-- The weekly savings by hiring the cheaper employee -/
def weekly_savings : ℚ := 160

theorem employee_work_hours :
  rate1 * hours_per_week - (rate2 * hours_per_week - subsidy * hours_per_week) = weekly_savings :=
by sorry

end NUMINAMATH_CALUDE_employee_work_hours_l2518_251801


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l2518_251827

theorem valentines_day_theorem (x y : ℕ) : 
  x * y = x + y + 28 → x * y = 60 :=
by sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l2518_251827


namespace NUMINAMATH_CALUDE_complex_on_line_l2518_251815

/-- Given a complex number z = (m-1) + (m+2)i that corresponds to a point on the line 2x-y=0,
    prove that m = 4. -/
theorem complex_on_line (m : ℝ) : 
  let z : ℂ := Complex.mk (m - 1) (m + 2)
  2 * z.re - z.im = 0 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_line_l2518_251815


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2518_251851

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- Atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- Atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- Number of Carbon atoms in the compound -/
def num_C : ℕ := 3

/-- Number of Nitrogen atoms in the compound -/
def num_N : ℕ := 1

/-- Number of Chlorine atoms in the compound -/
def num_Cl : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ :=
  (num_H : ℝ) * atomic_weight_H +
  (num_C : ℝ) * atomic_weight_C +
  (num_N : ℝ) * atomic_weight_N +
  (num_Cl : ℝ) * atomic_weight_Cl +
  (num_O : ℝ) * atomic_weight_O

/-- Theorem stating that the molecular weight of the compound is 135.51 g/mol -/
theorem compound_molecular_weight :
  molecular_weight = 135.51 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2518_251851


namespace NUMINAMATH_CALUDE_corner_cut_rectangle_l2518_251838

/-- Given a rectangle ABCD with dimensions AB = 18 m and AD = 12 m,
    and identical right-angled isosceles triangles cut off from the corners,
    leaving a smaller rectangle PQRS. The total area cut off is 180 m². -/
theorem corner_cut_rectangle (AB AD : ℝ) (area_cut : ℝ) (PR : ℝ) : AB = 18 → AD = 12 → area_cut = 180 → PR = 18 - 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_rectangle_l2518_251838


namespace NUMINAMATH_CALUDE_more_stable_performance_l2518_251826

/-- Represents a student's performance in throwing solid balls -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if a student's performance is more stable than another's -/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students with equal average scores, the one with smaller variance has more stable performance -/
theorem more_stable_performance (student_A student_B : StudentPerformance)
  (h_equal_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 0.1)
  (h_B_variance : student_B.variance = 0.02) :
  more_stable student_B student_A :=
by sorry

end NUMINAMATH_CALUDE_more_stable_performance_l2518_251826


namespace NUMINAMATH_CALUDE_dima_wins_l2518_251836

-- Define the game board as a set of integers from 1 to 100
def GameBoard : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100}

-- Define a type for player strategies
def Strategy := GameBoard → ℕ

-- Define the winning condition for Mitya
def MityaWins (a b : ℕ) : Prop := (a + b) % 7 = 0

-- Define the game result
inductive GameResult
| MityaVictory
| DimaVictory

-- Define the game play function
def playGame (mityaStrategy dimaStrategy : Strategy) : GameResult :=
  sorry -- Actual game logic would go here

-- Theorem statement
theorem dima_wins :
  ∃ (dimaStrategy : Strategy),
    ∀ (mityaStrategy : Strategy),
      playGame mityaStrategy dimaStrategy = GameResult.DimaVictory :=
sorry

end NUMINAMATH_CALUDE_dima_wins_l2518_251836


namespace NUMINAMATH_CALUDE_missing_number_proof_l2518_251896

theorem missing_number_proof (some_number : ℤ) : 
  some_number = 3 → |9 - 8 * (some_number - 12)| - |5 - 11| = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2518_251896


namespace NUMINAMATH_CALUDE_cube_order_l2518_251843

theorem cube_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_l2518_251843


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2518_251883

/-- Line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Circle with center at origin -/
structure Circle where
  radius : ℝ

/-- The perpendicular line to a given line passing through a point -/
def perpendicularLine (l : Line) (p : ℝ × ℝ) : Line :=
  sorry

/-- The length of the chord formed by the intersection of a line and a circle -/
def chordLength (l : Line) (c : Circle) : ℝ :=
  sorry

/-- The area of a circle -/
def circleArea (c : Circle) : ℝ :=
  sorry

theorem circle_area_theorem (l : Line) (c : Circle) :
  l.point1 = (2, 1) →
  l.point2 = (1, -1) →
  let m := perpendicularLine l (2, 1)
  chordLength m c = 6 * Real.sqrt 5 / 5 →
  circleArea c = 5 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2518_251883


namespace NUMINAMATH_CALUDE_sum_of_ages_l2518_251897

theorem sum_of_ages (a b c : ℕ) : 
  a = 11 → 
  (a - 3) + (b - 3) + (c - 3) = 6 * (a - 3) → 
  a + b + c = 57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2518_251897


namespace NUMINAMATH_CALUDE_eulers_formula_l2518_251881

/-- A planar graph structure -/
structure PlanarGraph where
  V : Type*  -- Set of vertices
  E : Type*  -- Set of edges
  F : Type*  -- Set of faces
  n : ℕ      -- Number of vertices
  m : ℕ      -- Number of edges
  ℓ : ℕ      -- Number of faces
  is_connected : Prop  -- Property that the graph is connected

/-- Euler's formula for planar graphs -/
theorem eulers_formula (G : PlanarGraph) : G.n - G.m + G.ℓ = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2518_251881


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2518_251831

/-- Given a line 2ax + by - 2 = 0 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 1/a + 1/b is 4 -/
theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b = 2 → (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 2 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2518_251831


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l2518_251834

/-- The number of red roses at Greene Nursery -/
def red_roses : ℕ := 1491

/-- The number of yellow carnations at Greene Nursery -/
def yellow_carnations : ℕ := 3025

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 1768

/-- The total number of flowers at Greene Nursery -/
def total_flowers : ℕ := red_roses + yellow_carnations + white_roses

theorem greene_nursery_flower_count : total_flowers = 6284 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l2518_251834


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_197_l2518_251874

theorem first_nonzero_digit_after_decimal_1_197 : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (1000 : ℚ) / 197 = (5 : ℚ) + (d : ℚ) / (10 : ℚ) ^ (n + 1) + (1 : ℚ) / (10 : ℚ) ^ (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_197_l2518_251874


namespace NUMINAMATH_CALUDE_equal_probability_for_all_l2518_251803

/-- Represents the sampling method used in the TV show -/
structure SamplingMethod where
  total_population : ℕ
  sample_size : ℕ
  removed_first : ℕ
  
/-- The probability of being selected for each individual in the population -/
def selection_probability (sm : SamplingMethod) : ℚ :=
  sm.sample_size / sm.total_population

/-- The specific sampling method used in the TV show -/
def tv_show_sampling : SamplingMethod := {
  total_population := 2014
  sample_size := 50
  removed_first := 14
}

theorem equal_probability_for_all (sm : SamplingMethod) :
  selection_probability sm = 25 / 1007 :=
sorry

#check equal_probability_for_all tv_show_sampling

end NUMINAMATH_CALUDE_equal_probability_for_all_l2518_251803


namespace NUMINAMATH_CALUDE_number_rewriting_l2518_251889

theorem number_rewriting :
  (29800000 = 2980 * 10000) ∧ (14000000000 = 140 * 100000000) := by
  sorry

end NUMINAMATH_CALUDE_number_rewriting_l2518_251889


namespace NUMINAMATH_CALUDE_fiftieth_central_ring_number_l2518_251824

/-- Returns the number of digits in a positive integer -/
def numDigits (n : ℕ+) : ℕ :=
  (Nat.log 10 n.val) + 1

/-- Defines a Central Ring Number -/
def isCentralRingNumber (x : ℕ+) : Prop :=
  numDigits (3 * x) > numDigits x

/-- Returns the nth Central Ring Number -/
def nthCentralRingNumber (n : ℕ+) : ℕ+ :=
  sorry

theorem fiftieth_central_ring_number :
  nthCentralRingNumber 50 = 81 :=
sorry

end NUMINAMATH_CALUDE_fiftieth_central_ring_number_l2518_251824


namespace NUMINAMATH_CALUDE_soldier_height_arrangement_l2518_251877

theorem soldier_height_arrangement (n : ℕ) (a b : Fin n → ℝ) :
  (∀ i : Fin n, a i ≤ b i) →
  (∀ i j : Fin n, i < j → a i ≥ a j) →
  (∀ i j : Fin n, i < j → b i ≥ b j) →
  ∀ i : Fin n, a i ≤ b i :=
by sorry

end NUMINAMATH_CALUDE_soldier_height_arrangement_l2518_251877


namespace NUMINAMATH_CALUDE_binomial_simplification_l2518_251884

/-- Given two binomials M and N in terms of x, prove that if 2(M) - 3(N) = 4x - 6 - 9x - 15,
    then N = 3x + 5 and the simplified expression P = -5x - 21 -/
theorem binomial_simplification (x : ℝ) (M N : ℝ → ℝ) :
  (∀ x, 2 * M x - 3 * N x = 4 * x - 6 - 9 * x - 15) →
  (∀ x, N x = 3 * x + 5) ∧
  (∀ x, 2 * M x - 3 * N x = -5 * x - 21) :=
by sorry

end NUMINAMATH_CALUDE_binomial_simplification_l2518_251884


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2518_251867

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 48) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2518_251867


namespace NUMINAMATH_CALUDE_cubic_rational_roots_l2518_251842

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The roots of a cubic polynomial -/
def roots (p : CubicPolynomial) : Set ℚ :=
  {x : ℚ | x^3 + p.a * x^2 + p.b * x + p.c = 0}

/-- The theorem stating the only possible sets of rational roots for a cubic polynomial -/
theorem cubic_rational_roots (p : CubicPolynomial) :
  roots p = {0, 1, -2} ∨ roots p = {1, -1, -1} := by
  sorry


end NUMINAMATH_CALUDE_cubic_rational_roots_l2518_251842


namespace NUMINAMATH_CALUDE_whale_ratio_theorem_l2518_251885

/-- The ratio of male whales on the third trip to the first trip -/
def whale_ratio : ℚ := 1 / 2

/-- The number of male whales on the first trip -/
def first_trip_males : ℕ := 28

/-- The number of female whales on the first trip -/
def first_trip_females : ℕ := 2 * first_trip_males

/-- The number of baby whales on the second trip -/
def second_trip_babies : ℕ := 8

/-- The total number of whales observed -/
def total_whales : ℕ := 178

/-- The number of male whales on the third trip -/
def third_trip_males : ℕ := total_whales - (first_trip_males + first_trip_females + second_trip_babies + 2 * second_trip_babies + first_trip_females)

theorem whale_ratio_theorem : 
  (third_trip_males : ℚ) / first_trip_males = whale_ratio := by
  sorry

end NUMINAMATH_CALUDE_whale_ratio_theorem_l2518_251885


namespace NUMINAMATH_CALUDE_unique_root_implies_specific_angles_l2518_251863

/-- Given α ∈ (0, π), if the equation |2x - 1/2| + |(\sqrt{6} - \sqrt{2})x| = sin α
    has exactly one real root, then α = π/12 or α = 11π/12 -/
theorem unique_root_implies_specific_angles (α : Real) 
    (h1 : α ∈ Set.Ioo 0 Real.pi)
    (h2 : ∃! x : Real, |2*x - 1/2| + |((Real.sqrt 6) - (Real.sqrt 2))*x| = Real.sin α) :
    α = Real.pi/12 ∨ α = 11*Real.pi/12 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_implies_specific_angles_l2518_251863


namespace NUMINAMATH_CALUDE_abs_neg_three_times_two_l2518_251818

theorem abs_neg_three_times_two : |(-3)| * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_times_two_l2518_251818


namespace NUMINAMATH_CALUDE_probability_red_or_green_is_13_22_l2518_251894

/-- Represents the count of jelly beans for each color -/
structure JellyBeanCounts where
  orange : ℕ
  purple : ℕ
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting either a red or green jelly bean -/
def probability_red_or_green (counts : JellyBeanCounts) : ℚ :=
  (counts.red + counts.green : ℚ) / (counts.orange + counts.purple + counts.red + counts.green)

/-- Theorem stating the probability of selecting a red or green jelly bean -/
theorem probability_red_or_green_is_13_22 :
  let counts : JellyBeanCounts := ⟨4, 5, 6, 7⟩
  probability_red_or_green counts = 13 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_green_is_13_22_l2518_251894


namespace NUMINAMATH_CALUDE_prob_two_empty_given_at_least_one_empty_l2518_251898

/-- The number of balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The number of ways to place balls into boxes with exactly one empty box -/
def ways_one_empty : ℕ := 144

/-- The number of ways to place balls into boxes with exactly two empty boxes -/
def ways_two_empty : ℕ := 84

/-- The number of ways to place balls into boxes with exactly three empty boxes -/
def ways_three_empty : ℕ := 4

/-- The probability of exactly two boxes being empty given at least one box is empty -/
theorem prob_two_empty_given_at_least_one_empty :
  (ways_two_empty : ℚ) / (ways_one_empty + ways_two_empty + ways_three_empty) = 21 / 58 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_empty_given_at_least_one_empty_l2518_251898


namespace NUMINAMATH_CALUDE_seedling_packaging_l2518_251816

/-- The number of seedlings to be placed in packets -/
def total_seedlings : ℕ := 420

/-- The number of seeds required in each packet -/
def seeds_per_packet : ℕ := 7

/-- The number of packets needed to place all seedlings -/
def packets_needed : ℕ := total_seedlings / seeds_per_packet

theorem seedling_packaging : packets_needed = 60 := by
  sorry

end NUMINAMATH_CALUDE_seedling_packaging_l2518_251816


namespace NUMINAMATH_CALUDE_product_equals_fraction_fraction_is_simplified_l2518_251845

/-- The repeating decimal 0.256̄ as a rational number -/
def repeating_decimal : ℚ := 256 / 999

/-- The product of 0.256̄ and 12 -/
def product : ℚ := repeating_decimal * 12

/-- Theorem stating that the product of 0.256̄ and 12 is equal to 1024/333 -/
theorem product_equals_fraction : product = 1024 / 333 := by
  sorry

/-- Theorem stating that 1024/333 is in its simplest form -/
theorem fraction_is_simplified : Int.gcd 1024 333 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_fraction_is_simplified_l2518_251845


namespace NUMINAMATH_CALUDE_class_size_multiple_of_eight_l2518_251873

theorem class_size_multiple_of_eight (boys girls total : ℕ) : 
  girls = 7 * boys → total = boys + girls → ∃ k : ℕ, total = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_class_size_multiple_of_eight_l2518_251873


namespace NUMINAMATH_CALUDE_power_of_product_l2518_251833

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l2518_251833


namespace NUMINAMATH_CALUDE_train_turn_radians_l2518_251817

/-- Given a circular railway arc with radius 2 km and a train moving at 30 km/h,
    the number of radians the train turns through in 10 seconds is 1/24. -/
theorem train_turn_radians (r : ℝ) (v : ℝ) (t : ℝ) :
  r = 2 →  -- radius in km
  v = 30 → -- speed in km/h
  t = 10 / 3600 → -- time in hours (10 seconds converted to hours)
  (v * t) / r = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_train_turn_radians_l2518_251817


namespace NUMINAMATH_CALUDE_peanuts_in_box_l2518_251860

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 4

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 8

/-- The total number of peanuts in the box after Mary adds more -/
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_in_box : total_peanuts = 12 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l2518_251860


namespace NUMINAMATH_CALUDE_intersection_M_N_l2518_251876

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | Real.log x / Real.log (1/2) > -1}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2518_251876
