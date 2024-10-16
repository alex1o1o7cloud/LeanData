import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_characterization_l332_33202

theorem linear_function_characterization (f : ℕ → ℕ) :
  (∀ x y : ℕ, f (x + y) = f x + f y) →
  ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l332_33202


namespace NUMINAMATH_CALUDE_total_marbles_is_240_l332_33275

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * dozen

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 4 * jessica_marbles

/-- The number of red marbles Alex has -/
def alex_marbles : ℕ := jessica_marbles + 2 * dozen

/-- The total number of red marbles Jessica, Sandy, and Alex have -/
def total_marbles : ℕ := jessica_marbles + sandy_marbles + alex_marbles

theorem total_marbles_is_240 : total_marbles = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_240_l332_33275


namespace NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l332_33220

theorem min_value_product (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

theorem min_value_product_achieved (x : ℝ) : 
  ∃ y : ℝ, (15 - y) * (13 - y) * (15 + y) * (13 + y) = -784 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l332_33220


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l332_33233

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l332_33233


namespace NUMINAMATH_CALUDE_can_guess_majority_winners_l332_33290

/-- Represents a tennis tournament with n players -/
structure TennisTournament (n : ℕ) where
  /-- Final scores of each player -/
  scores : Fin n → ℕ
  /-- Total number of matches in the tournament -/
  total_matches : ℕ := n * (n - 1) / 2

/-- Theorem stating that it's possible to guess more than half of the match winners -/
theorem can_guess_majority_winners (n : ℕ) (tournament : TennisTournament n) :
  ∃ (guessed_matches : ℕ), guessed_matches > tournament.total_matches / 2 :=
sorry

end NUMINAMATH_CALUDE_can_guess_majority_winners_l332_33290


namespace NUMINAMATH_CALUDE_num_2d_faces_6cube_l332_33249

/-- The number of 2-D square faces in a 6-dimensional cube of side length 6 -/
def num_2d_faces (n : ℕ) (side_length : ℕ) : ℕ :=
  (Nat.choose n 4) * (side_length + 1)^4 * side_length^2

/-- Theorem stating the number of 2-D square faces in a 6-cube of side length 6 -/
theorem num_2d_faces_6cube :
  num_2d_faces 6 6 = 1296150 := by
  sorry

end NUMINAMATH_CALUDE_num_2d_faces_6cube_l332_33249


namespace NUMINAMATH_CALUDE_determine_sanity_with_one_question_l332_33236

-- Define the types
inductive Species : Type
| Human
| Vampire

inductive MentalState : Type
| Sane
| Insane

-- Define the Transylvanian type
structure Transylvanian :=
  (species : Species)
  (mental_state : MentalState)

-- Define the question type
inductive Question : Type
| AreYouAPerson

-- Define the answer type
inductive Answer : Type
| Yes
| No

-- Define the response function
def respond (t : Transylvanian) (q : Question) : Answer :=
  match t.mental_state, q with
  | MentalState.Sane, Question.AreYouAPerson => Answer.Yes
  | MentalState.Insane, Question.AreYouAPerson => Answer.No

-- Theorem statement
theorem determine_sanity_with_one_question :
  ∃ (q : Question), ∀ (t : Transylvanian),
    (respond t q = Answer.Yes ↔ t.mental_state = MentalState.Sane) ∧
    (respond t q = Answer.No ↔ t.mental_state = MentalState.Insane) :=
by sorry

end NUMINAMATH_CALUDE_determine_sanity_with_one_question_l332_33236


namespace NUMINAMATH_CALUDE_toy_purchase_problem_l332_33230

theorem toy_purchase_problem (toy_cost : ℝ) (discount_rate : ℝ) (total_paid : ℝ) :
  toy_cost = 3 →
  discount_rate = 0.2 →
  total_paid = 12 →
  (1 - discount_rate) * (toy_cost * (total_paid / ((1 - discount_rate) * toy_cost))) = total_paid →
  ∃ n : ℕ, n = 5 ∧ n * toy_cost = total_paid / (1 - discount_rate) := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_problem_l332_33230


namespace NUMINAMATH_CALUDE_cloth_sale_solution_l332_33242

/-- Represents the problem of calculating the total sale amount for cloth -/
def ClothSaleProblem (totalMeters : ℕ) (lossPerMeter : ℕ) (costPricePerMeter : ℕ) : Prop :=
  let sellingPricePerMeter := costPricePerMeter - lossPerMeter
  let totalAmount := sellingPricePerMeter * totalMeters
  totalAmount = 36000

/-- Theorem stating the solution to the cloth sale problem -/
theorem cloth_sale_solution :
  ClothSaleProblem 600 10 70 := by
  sorry

#check cloth_sale_solution

end NUMINAMATH_CALUDE_cloth_sale_solution_l332_33242


namespace NUMINAMATH_CALUDE_part_one_part_two_l332_33247

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : 
  (∀ x, f a x ≥ 3 ↔ x ≤ 1 ∨ x ≥ 5) → a = 2 :=
sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f 2 x + f 2 (x + 4) ≥ m) → m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l332_33247


namespace NUMINAMATH_CALUDE_tamtam_orange_shells_l332_33295

/-- The number of orange shells in Tamtam's collection --/
def orange_shells (total purple pink yellow blue : ℕ) : ℕ :=
  total - (purple + pink + yellow + blue)

/-- Theorem stating the number of orange shells in Tamtam's collection --/
theorem tamtam_orange_shells :
  orange_shells 65 13 8 18 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tamtam_orange_shells_l332_33295


namespace NUMINAMATH_CALUDE_derivative_not_always_constant_l332_33214

-- Define a real-valued function
def f : ℝ → ℝ := sorry

-- Define the derivative of f at a point x
def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ := sorry

-- Theorem stating that the derivative is not always a constant
theorem derivative_not_always_constant :
  ∃ (f : ℝ → ℝ) (x y : ℝ), x ≠ y → derivative_at f x ≠ derivative_at f y :=
sorry

end NUMINAMATH_CALUDE_derivative_not_always_constant_l332_33214


namespace NUMINAMATH_CALUDE_eq1_solution_eq2_solution_eq3_solution_eq4_solution_eq5_solution_l332_33292

-- Equation 1: 3x^2 - 15 = 0
theorem eq1_solution (x : ℝ) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ↔ 3 * x^2 - 15 = 0 := by sorry

-- Equation 2: x^2 - 8x + 15 = 0
theorem eq2_solution (x : ℝ) : x = 3 ∨ x = 5 ↔ x^2 - 8*x + 15 = 0 := by sorry

-- Equation 3: x^2 - 6x + 7 = 0
theorem eq3_solution (x : ℝ) : x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2 ↔ x^2 - 6*x + 7 = 0 := by sorry

-- Equation 4: 2x^2 - 6x + 1 = 0
theorem eq4_solution (x : ℝ) : x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2 ↔ 2*x^2 - 6*x + 1 = 0 := by sorry

-- Equation 5: (2x^2 + 3x)^2 - 4(2x^2 + 3x) - 5 = 0
theorem eq5_solution (x : ℝ) : x = -5/2 ∨ x = 1 ∨ x = -1/2 ∨ x = -1 ↔ (2*x^2 + 3*x)^2 - 4*(2*x^2 + 3*x) - 5 = 0 := by sorry

end NUMINAMATH_CALUDE_eq1_solution_eq2_solution_eq3_solution_eq4_solution_eq5_solution_l332_33292


namespace NUMINAMATH_CALUDE_prob_at_most_one_success_in_three_trials_l332_33299

/-- The probability of at most one success in three independent trials -/
theorem prob_at_most_one_success_in_three_trials (p : ℝ) (h : p = 1/3) :
  p^0 * (1-p)^3 + 3 * p^1 * (1-p)^2 = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_success_in_three_trials_l332_33299


namespace NUMINAMATH_CALUDE_quadratic_equation_always_real_roots_l332_33281

theorem quadratic_equation_always_real_roots (m : ℝ) :
  ∃ x : ℝ, m * x^2 - (5*m - 1) * x + (4*m - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_always_real_roots_l332_33281


namespace NUMINAMATH_CALUDE_rhombus_area_l332_33250

/-- The area of a rhombus with specific side length and diagonal difference -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 →
  diag_diff = 8 →
  area = 6 * Real.sqrt 210 - 12 →
  ∃ (d1 d2 : ℝ), 
    d1 > 0 ∧ d2 > 0 ∧
    d2 - d1 = diag_diff ∧
    d1 * d2 / 2 = area ∧
    d1^2 / 4 + side^2 = (d2 / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l332_33250


namespace NUMINAMATH_CALUDE_matrix_vector_multiplication_l332_33251

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; -3, 4]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![3; -1]

theorem matrix_vector_multiplication :
  A * v = !![7; -13] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_multiplication_l332_33251


namespace NUMINAMATH_CALUDE_equal_distribution_l332_33276

/-- Proves that when Rs 42,900 is distributed equally among 22 persons, each person receives Rs 1,950. -/
theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) : 
  total_amount = 42900 → 
  num_persons = 22 → 
  amount_per_person = total_amount / num_persons → 
  amount_per_person = 1950 := by
sorry

end NUMINAMATH_CALUDE_equal_distribution_l332_33276


namespace NUMINAMATH_CALUDE_perimeter_is_18_l332_33282

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define points A and B on the left branch
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line passing through F1, A, and B
def line_through_F1AB (p : ℝ × ℝ) : Prop := sorry

-- State that A and B are on the hyperbola
axiom A_on_hyperbola : hyperbola A.1 A.2
axiom B_on_hyperbola : hyperbola B.1 B.2

-- State that A and B are on the line passing through F1
axiom A_on_line : line_through_F1AB A
axiom B_on_line : line_through_F1AB B

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State that the distance between A and B is 5
axiom AB_distance : distance A B = 5

-- Define the perimeter of triangle AF2B
def perimeter_AF2B : ℝ := distance A F2 + distance B F2 + distance A B

-- Theorem to prove
theorem perimeter_is_18 : perimeter_AF2B = 18 := by sorry

end NUMINAMATH_CALUDE_perimeter_is_18_l332_33282


namespace NUMINAMATH_CALUDE_correlation_relationships_l332_33271

-- Define the types of relationships
inductive Relationship
  | PointCoordinate
  | AppleYieldClimate
  | TreeDiameterHeight
  | StudentID

-- Define a function to determine if a relationship involves correlation
def involvesCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleYieldClimate => True
  | Relationship.TreeDiameterHeight => True
  | _ => False

-- Theorem statement
theorem correlation_relationships :
  (involvesCorrelation Relationship.PointCoordinate = False) ∧
  (involvesCorrelation Relationship.AppleYieldClimate = True) ∧
  (involvesCorrelation Relationship.TreeDiameterHeight = True) ∧
  (involvesCorrelation Relationship.StudentID = False) :=
sorry

end NUMINAMATH_CALUDE_correlation_relationships_l332_33271


namespace NUMINAMATH_CALUDE_third_subtraction_difference_1230_411_l332_33241

/-- The difference obtained from the third subtraction when using the method of successive subtraction to find the GCD of 1230 and 411 -/
def third_subtraction_difference (a b : ℕ) : ℕ :=
  let d₁ := a - b
  let d₂ := d₁ - b
  d₂ - b

theorem third_subtraction_difference_1230_411 :
  third_subtraction_difference 1230 411 = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_subtraction_difference_1230_411_l332_33241


namespace NUMINAMATH_CALUDE_binary_110_eq_6_l332_33215

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110_eq_6 :
  binary_to_decimal [true, true, false] = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_eq_6_l332_33215


namespace NUMINAMATH_CALUDE_school_teachers_count_l332_33239

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sampled_students : ℕ) : 
  total = 2400 →
  sample_size = 160 →
  sampled_students = 150 →
  (total : ℚ) / sample_size = 15 →
  total - (sampled_students * ((total : ℚ) / sample_size).floor) = 150 :=
by sorry

end NUMINAMATH_CALUDE_school_teachers_count_l332_33239


namespace NUMINAMATH_CALUDE_count_valid_numbers_l332_33270

/-- The set of available digits --/
def available_digits : Finset Nat := {2, 4, 6, 7, 8}

/-- A function that generates all valid three-digit numbers --/
def valid_numbers : Finset Nat :=
  Finset.filter (λ n : Nat => 
    n ≥ 100 ∧ n < 1000 ∧
    (n / 100) ∈ available_digits ∧
    ((n / 10) % 10) ∈ available_digits ∧
    (n % 10) ∈ available_digits ∧
    (n / 100) ≠ ((n / 10) % 10) ∧
    (n / 100) ≠ (n % 10) ∧
    ((n / 10) % 10) ≠ (n % 10)
  ) (Finset.range 1000)

/-- The theorem stating the number of valid three-digit numbers --/
theorem count_valid_numbers : Finset.card valid_numbers = 60 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l332_33270


namespace NUMINAMATH_CALUDE_binomial_n_equals_eight_l332_33223

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with expectation 4 and variance 2, n = 8 -/
theorem binomial_n_equals_eight (X : BinomialDistribution) 
  (h_exp : expectation X = 4)
  (h_var : variance X = 2) : 
  X.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_equals_eight_l332_33223


namespace NUMINAMATH_CALUDE_m_range_theorem_l332_33283

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

-- State the theorem
theorem m_range_theorem : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l332_33283


namespace NUMINAMATH_CALUDE_katherine_website_time_l332_33265

/-- Given that Naomi takes 1/4 times more time than Katherine to develop a website,
    and Naomi developed 30 websites in 750 hours, prove that Katherine takes 20 hours
    to develop a website. -/
theorem katherine_website_time (naomi_time_ratio : ℚ) (naomi_websites : ℕ) (naomi_total_time : ℕ) 
  (h1 : naomi_time_ratio = 5/4)
  (h2 : naomi_websites = 30)
  (h3 : naomi_total_time = 750) :
  ∃ (katherine_time : ℚ), 
    katherine_time * naomi_time_ratio = (naomi_total_time : ℚ) / (naomi_websites : ℚ) ∧ 
    katherine_time = 20 := by sorry

end NUMINAMATH_CALUDE_katherine_website_time_l332_33265


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l332_33243

/-- A parallelogram representing a crosswalk on a street -/
structure Crosswalk where
  curb_length : ℝ  -- Length along the curb
  stripe_length : ℝ  -- Length of each stripe
  angle : ℝ  -- Angle between curb and stripe
  street_width : ℝ  -- Width of the street (perpendicular to curb)

/-- Theorem stating the relation between crosswalk dimensions -/
theorem crosswalk_stripe_distance (c : Crosswalk)
  (h_curb : c.curb_length = 20)
  (h_stripe : c.stripe_length = 60)
  (h_angle : c.angle = π / 6)  -- 30 degrees in radians
  (h_width : c.street_width = 50) :
  c.curb_length * c.street_width = c.stripe_length * (1000 * Real.sqrt 3 / 90) * Real.cos c.angle :=
sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l332_33243


namespace NUMINAMATH_CALUDE_amy_candy_difference_l332_33228

/-- Amy's candy distribution problem -/
theorem amy_candy_difference (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 6 → left = 5 → given_away - left = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_candy_difference_l332_33228


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l332_33246

theorem binomial_expansion_example : 12^4 + 4*(12^3) + 6*(12^2) + 4*12 + 1 = 28561 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l332_33246


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l332_33267

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / 9 - y^2 / a = 1

-- Define the right focus
def right_focus (a : ℝ) : Prop := hyperbola (Real.sqrt 13) 0 a

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Theorem statement
theorem hyperbola_asymptotes (a : ℝ) :
  right_focus a → ∀ x y, hyperbola x y a → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l332_33267


namespace NUMINAMATH_CALUDE_existence_equivalence_l332_33289

theorem existence_equivalence : 
  (∃ (x : ℝ), x^2 + 1 < 0) ↔ (∃ (x : ℝ), x^2 + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_equivalence_l332_33289


namespace NUMINAMATH_CALUDE_triangle_qca_area_l332_33212

/-- Given points Q, A, C in a coordinate plane and that triangle QCA is right-angled at C,
    prove that the area of triangle QCA is (36 - 3p) / 2 -/
theorem triangle_qca_area (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 12)
  let A : ℝ × ℝ := (3, 12)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := (1 / 2) * (A.1 - Q.1) * (Q.2 - C.2)
  triangle_area = (36 - 3*p) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_qca_area_l332_33212


namespace NUMINAMATH_CALUDE_greatest_number_of_sets_l332_33254

theorem greatest_number_of_sets (t_shirts : ℕ) (buttons : ℕ) : 
  t_shirts = 4 → buttons = 20 → 
  (∃ (sets : ℕ), sets > 0 ∧ 
    t_shirts % sets = 0 ∧ 
    buttons % sets = 0 ∧
    ∀ (k : ℕ), k > 0 ∧ t_shirts % k = 0 ∧ buttons % k = 0 → k ≤ sets) →
  Nat.gcd t_shirts buttons = 4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_of_sets_l332_33254


namespace NUMINAMATH_CALUDE_quadratic_factorization_l332_33201

theorem quadratic_factorization (b c : ℝ) :
  (∀ x, x^2 + b*x + c = 0 ↔ x = -2 ∨ x = 3) →
  ∀ x, x^2 + b*x + c = (x + 2) * (x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l332_33201


namespace NUMINAMATH_CALUDE_b_subset_a_iff_a_eq_two_or_three_c_subset_a_iff_m_condition_l332_33200

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + (a-1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- Theorem 1: B is a subset of A if and only if a = 2 or a = 3
theorem b_subset_a_iff_a_eq_two_or_three :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ↔ (a = 2 ∨ a = 3) :=
sorry

-- Theorem 2: C is a subset of A if and only if m = 3 or -2√2 < m < 2√2
theorem c_subset_a_iff_m_condition (m : ℝ) :
  C m ⊆ A ↔ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_b_subset_a_iff_a_eq_two_or_three_c_subset_a_iff_m_condition_l332_33200


namespace NUMINAMATH_CALUDE_division_problem_l332_33277

theorem division_problem (dividend : ℕ) (divisor : ℝ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 17698 →
  divisor = 198.69662921348313 →
  remainder = 14 →
  quotient = 89 →
  (dividend : ℝ) = divisor * (quotient : ℝ) + (remainder : ℝ) :=
by
  sorry

#eval (17698 : ℝ) - 198.69662921348313 * 89 - 14

end NUMINAMATH_CALUDE_division_problem_l332_33277


namespace NUMINAMATH_CALUDE_aisha_head_fraction_l332_33210

/-- Miss Aisha's height measurements -/
structure AishaHeight where
  total : ℝ
  legs : ℝ
  rest : ℝ
  head : ℝ

/-- Properties of Miss Aisha's height -/
def aisha_properties (h : AishaHeight) : Prop :=
  h.total = 60 ∧
  h.legs = (1/3) * h.total ∧
  h.rest = 25 ∧
  h.head = h.total - (h.legs + h.rest)

/-- Theorem: Miss Aisha's head is 1/4 of her total height -/
theorem aisha_head_fraction (h : AishaHeight) 
  (hprops : aisha_properties h) : h.head / h.total = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_aisha_head_fraction_l332_33210


namespace NUMINAMATH_CALUDE_function_g_theorem_l332_33203

theorem function_g_theorem (g : ℝ → ℝ) 
  (h1 : g 0 = 2)
  (h2 : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2) :
  ∀ x : ℝ, g x = 2 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_function_g_theorem_l332_33203


namespace NUMINAMATH_CALUDE_fathers_remaining_chocolates_fathers_remaining_chocolates_eq_five_l332_33227

theorem fathers_remaining_chocolates 
  (initial_chocolates : ℕ) 
  (num_sisters : ℕ) 
  (chocolates_to_mother : ℕ) 
  (chocolates_eaten : ℕ) : ℕ :=
  let total_people := num_sisters + 1
  let chocolates_per_person := initial_chocolates / total_people
  let chocolates_given_to_father := total_people * (chocolates_per_person / 2)
  let remaining_chocolates := chocolates_given_to_father - chocolates_to_mother - chocolates_eaten
  remaining_chocolates

theorem fathers_remaining_chocolates_eq_five :
  fathers_remaining_chocolates 20 4 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fathers_remaining_chocolates_fathers_remaining_chocolates_eq_five_l332_33227


namespace NUMINAMATH_CALUDE_interval_of_decrease_l332_33284

/-- Given a function f with derivative f'(x) = 2x - 4, 
    prove that the interval of decrease for f(x-1) is (-∞, 3) -/
theorem interval_of_decrease (f : ℝ → ℝ) (h : ∀ x, deriv f x = 2 * x - 4) :
  ∀ x, x < 3 ↔ deriv (fun y ↦ f (y - 1)) x < 0 := by
  sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l332_33284


namespace NUMINAMATH_CALUDE_sum_of_integers_l332_33244

theorem sum_of_integers (a b c : ℕ+) 
  (h : (a:ℝ)^2 + (b:ℝ)^2 + (c:ℝ)^2 + 43 ≤ (a:ℝ)*(b:ℝ) + 9*(b:ℝ) + 8*(c:ℝ)) : 
  (a:ℝ) + (b:ℝ) + (c:ℝ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l332_33244


namespace NUMINAMATH_CALUDE_statue_cost_l332_33298

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) : 
  selling_price = 750 ∧ 
  profit_percentage = 35 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 555.56 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l332_33298


namespace NUMINAMATH_CALUDE_intersection_M_N_l332_33266

def M : Set ℝ := {1, 2}
def N : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l332_33266


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l332_33222

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 - 4*I) = 1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l332_33222


namespace NUMINAMATH_CALUDE_distance_from_focus_to_line_l332_33296

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus
def right_focus : ℝ × ℝ := (3, 0)

-- State the theorem
theorem distance_from_focus_to_line :
  let (x₀, y₀) := right_focus
  ∃ d : ℝ, d = |x₀ + 2*y₀ - 8| / Real.sqrt (1^2 + 2^2) ∧ d = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_distance_from_focus_to_line_l332_33296


namespace NUMINAMATH_CALUDE_salary_increase_proof_l332_33211

theorem salary_increase_proof (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) :
  num_employees = 20 ∧ initial_avg = 1500 ∧ manager_salary = 4650 →
  (((num_employees : ℚ) * initial_avg + manager_salary) / (num_employees + 1 : ℚ)) - initial_avg = 150 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l332_33211


namespace NUMINAMATH_CALUDE_different_orders_count_l332_33278

def memo_count : ℕ := 11
def processed_memos : Finset ℕ := {9, 10}

def possible_remaining_memos : Finset ℕ := Finset.range 9 ∪ {11}

def insert_positions (n : ℕ) : ℕ := n + 2

/-- The number of different orders for processing the remaining memos -/
def different_orders : ℕ :=
  (Finset.range 9).sum fun j =>
    (Nat.choose 8 j) * (insert_positions j)

theorem different_orders_count :
  different_orders = 1536 := by
  sorry

end NUMINAMATH_CALUDE_different_orders_count_l332_33278


namespace NUMINAMATH_CALUDE_equation_one_solution_l332_33207

theorem equation_one_solution :
  ∀ x : ℝ, x^4 - x^2 - 6 = 0 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l332_33207


namespace NUMINAMATH_CALUDE_max_product_sum_300_l332_33258

theorem max_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l332_33258


namespace NUMINAMATH_CALUDE_candy_bars_per_box_l332_33213

/-- Proves that the number of candy bars in each box is 10 given the specified conditions --/
theorem candy_bars_per_box 
  (num_boxes : ℕ) 
  (selling_price buying_price : ℚ)
  (total_profit : ℚ)
  (h1 : num_boxes = 5)
  (h2 : selling_price = 3/2)
  (h3 : buying_price = 1)
  (h4 : total_profit = 25) :
  (total_profit / (num_boxes * (selling_price - buying_price))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_candy_bars_per_box_l332_33213


namespace NUMINAMATH_CALUDE_deepak_age_l332_33268

theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (future_years : ℕ) (rahul_future_age : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 3 →
  future_years = 6 →
  rahul_future_age = 18 →
  ∃ (x : ℚ), rahul_ratio * x + future_years = rahul_future_age ∧ deepak_ratio * x = 9 :=
by sorry

end NUMINAMATH_CALUDE_deepak_age_l332_33268


namespace NUMINAMATH_CALUDE_robin_gum_pieces_l332_33259

/-- Calculates the total number of gum pieces Robin has. -/
def total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : ℕ :=
  packages * pieces_per_package + extra_pieces

/-- Proves that Robin has 997 pieces of gum in total. -/
theorem robin_gum_pieces :
  total_gum_pieces 43 23 8 = 997 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_pieces_l332_33259


namespace NUMINAMATH_CALUDE_rachel_books_total_l332_33232

theorem rachel_books_total (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 9)
  (h2 : mystery_shelves = 6)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 72 :=
by sorry

end NUMINAMATH_CALUDE_rachel_books_total_l332_33232


namespace NUMINAMATH_CALUDE_abc_inequalities_l332_33217

theorem abc_inequalities (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a > b) (h5 : b > c) (h6 : a + b + c = 0) :
  (c / a + a / c ≤ -2) ∧ (-2 < c / a ∧ c / a < -1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l332_33217


namespace NUMINAMATH_CALUDE_solution_set_implies_m_equals_one_l332_33255

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*m*x - 4

-- State the theorem
theorem solution_set_implies_m_equals_one :
  (∀ x : ℝ, f m x < 0 ↔ -4 < x ∧ x < 1) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_equals_one_l332_33255


namespace NUMINAMATH_CALUDE_reflect_F_final_coords_l332_33297

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def F : ℝ × ℝ := (1, 3)

theorem reflect_F_final_coords :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) F = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflect_F_final_coords_l332_33297


namespace NUMINAMATH_CALUDE_triangle_area_is_eight_l332_33245

/-- A line in 2D space represented by its equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The triangle formed by the intersection of three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Calculate the area of a triangle given its three lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { l1 := { m := 0, b := 7 },
    l2 := { m := 2, b := 3 },
    l3 := { m := -2, b := 3 } }

theorem triangle_area_is_eight :
  triangleArea problemTriangle = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_eight_l332_33245


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l332_33288

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l332_33288


namespace NUMINAMATH_CALUDE_base7_to_base4_conversion_l332_33274

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- The given number in base 7 -/
def given_number : ℕ := 563

theorem base7_to_base4_conversion :
  base10ToBase4 (base7ToBase10 given_number) = 10202 := by sorry

end NUMINAMATH_CALUDE_base7_to_base4_conversion_l332_33274


namespace NUMINAMATH_CALUDE_marbles_redistribution_l332_33294

/-- The number of marbles Tyrone gives to Eric -/
def marblesGiven : ℕ := 19

/-- The initial number of marbles Tyrone had -/
def tyronesInitial : ℕ := 120

/-- The initial number of marbles Eric had -/
def ericsInitial : ℕ := 15

/-- Proposition: The number of marbles Tyrone gave to Eric satisfies the conditions -/
theorem marbles_redistribution :
  (tyronesInitial - marblesGiven) = 3 * (ericsInitial + marblesGiven) ∧
  marblesGiven > 0 ∧
  marblesGiven < tyronesInitial :=
by
  sorry

#check marbles_redistribution

end NUMINAMATH_CALUDE_marbles_redistribution_l332_33294


namespace NUMINAMATH_CALUDE_light_wash_water_usage_l332_33280

/-- Represents the water usage of a washing machine -/
structure WashingMachine where
  heavyWashWater : ℕ
  regularWashWater : ℕ
  lightWashWater : ℕ
  heavyWashCount : ℕ
  regularWashCount : ℕ
  lightWashCount : ℕ
  bleachedLoadsCount : ℕ
  totalWaterUsage : ℕ

/-- Theorem stating that the light wash water usage is 2 gallons -/
theorem light_wash_water_usage 
  (wm : WashingMachine) 
  (heavy_wash : wm.heavyWashWater = 20)
  (regular_wash : wm.regularWashWater = 10)
  (wash_counts : wm.heavyWashCount = 2 ∧ wm.regularWashCount = 3 ∧ wm.lightWashCount = 1)
  (bleached_loads : wm.bleachedLoadsCount = 2)
  (total_water : wm.totalWaterUsage = 76)
  (water_balance : wm.totalWaterUsage = 
    wm.heavyWashWater * wm.heavyWashCount + 
    wm.regularWashWater * wm.regularWashCount + 
    wm.lightWashWater * (wm.lightWashCount + wm.bleachedLoadsCount)) :
  wm.lightWashWater = 2 := by
  sorry

end NUMINAMATH_CALUDE_light_wash_water_usage_l332_33280


namespace NUMINAMATH_CALUDE_geometric_sequence_min_S3_l332_33279

/-- Given a geometric sequence with positive terms, prove that the minimum value of S_3 is 6 -/
theorem geometric_sequence_min_S3 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- positive terms
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  a 4 * a 8 = 2 * a 10 →  -- given condition
  (∃ S : ℕ → ℝ, ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum of first n terms
  (∃ min_S3 : ℝ, ∀ S3, S3 = a 1 + a 2 + a 3 → S3 ≥ min_S3 ∧ min_S3 = 6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_S3_l332_33279


namespace NUMINAMATH_CALUDE_incorrect_operation_l332_33240

theorem incorrect_operation : 
  (∀ a : ℝ, (-a)^4 = a^4) ∧ 
  (∀ a : ℝ, -a + 3*a = 2*a) ∧ 
  (¬ ∀ a : ℝ, (2*a^2)^3 = 6*a^5) ∧ 
  (∀ a : ℝ, a^6 / a^2 = a^4) := by sorry

end NUMINAMATH_CALUDE_incorrect_operation_l332_33240


namespace NUMINAMATH_CALUDE_white_balls_count_l332_33225

theorem white_balls_count (total green blue yellow white : ℕ) : 
  total = green + blue + yellow + white →
  4 * green = total →
  8 * blue = total →
  12 * yellow = total →
  blue = 6 →
  white = 26 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l332_33225


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l332_33264

-- Define the total number of team members
def total_members : ℕ := 18

-- Define the number of players in the starting lineup
def lineup_size : ℕ := 8

-- Define the number of interchangeable positions
def interchangeable_positions : ℕ := 6

-- Theorem statement
theorem volleyball_lineup_combinations :
  (total_members) *
  (total_members - 1) *
  (Nat.choose (total_members - 2) interchangeable_positions) =
  2448272 := by sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l332_33264


namespace NUMINAMATH_CALUDE_largest_number_value_l332_33248

theorem largest_number_value (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 5) :
  c = 41.67 := by
sorry

end NUMINAMATH_CALUDE_largest_number_value_l332_33248


namespace NUMINAMATH_CALUDE_specific_rectangle_perimeter_l332_33286

/-- Represents a rectangle with two internal segments --/
structure CutRectangle where
  AD : ℝ  -- Length of side AD
  AB : ℝ  -- Length of side AB
  EF : ℝ  -- Length of internal segment EF
  GH : ℝ  -- Length of internal segment GH

/-- Calculates the total perimeter of the two shapes formed by cutting the rectangle --/
def totalPerimeter (r : CutRectangle) : ℝ :=
  2 * (r.AD + r.AB + r.EF + r.GH)

/-- Theorem stating that for a specific rectangle, the total perimeter is 40 --/
theorem specific_rectangle_perimeter :
  ∃ (r : CutRectangle), r.AD = 10 ∧ r.AB = 6 ∧ r.EF = 2 ∧ r.GH = 2 ∧ totalPerimeter r = 40 := by
  sorry

end NUMINAMATH_CALUDE_specific_rectangle_perimeter_l332_33286


namespace NUMINAMATH_CALUDE_fifth_term_value_l332_33293

def a (n : ℕ+) : ℚ := n / (n^2 + 25)

theorem fifth_term_value : a 5 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l332_33293


namespace NUMINAMATH_CALUDE_isosceles_triangle_equation_l332_33260

def isIsosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def isRoot (x n : ℝ) : Prop := x^2 - 8*x + n = 0

theorem isosceles_triangle_equation (n : ℝ) : 
  (∃ (a b : ℝ), isIsosceles 3 a b ∧ isRoot a n ∧ isRoot b n) → (n = 15 ∨ n = 16) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_equation_l332_33260


namespace NUMINAMATH_CALUDE_total_food_items_donated_l332_33237

/-- The total number of food items donated by five companies given specific donation rules -/
theorem total_food_items_donated (foster_chickens : ℕ) : foster_chickens = 45 →
  ∃ (american_water hormel_chickens boudin_chickens delmonte_water : ℕ),
    american_water = 2 * foster_chickens ∧
    hormel_chickens = 3 * foster_chickens ∧
    boudin_chickens = hormel_chickens / 3 ∧
    delmonte_water = american_water - 30 ∧
    (boudin_chickens + delmonte_water) % 7 = 0 ∧
    foster_chickens = (hormel_chickens + boudin_chickens) / 2 ∧
    foster_chickens + american_water + hormel_chickens + boudin_chickens + delmonte_water = 375 :=
by sorry

end NUMINAMATH_CALUDE_total_food_items_donated_l332_33237


namespace NUMINAMATH_CALUDE_percentage_increase_l332_33208

theorem percentage_increase (x y z : ℝ) : 
  y = 0.5 * z →  -- y is 50% less than z
  x = 0.6 * z →  -- x is 60% of z
  x = 1.2 * y    -- x is 20% more than y (equivalent to 120% of y)
  := by sorry

end NUMINAMATH_CALUDE_percentage_increase_l332_33208


namespace NUMINAMATH_CALUDE_yoongi_calculation_l332_33253

theorem yoongi_calculation : (30 + 5) / 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_calculation_l332_33253


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l332_33257

theorem polynomial_root_problem (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ t : ℝ, t^3 + a*t^2 + b*t + 20 = 0 ↔ t = x ∨ t = y ∨ t = z)) →
  (∀ t : ℝ, t^3 + a*t^2 + b*t + 20 = 0 → t^4 + t^3 + b*t^2 + c*t + 200 = 0) →
  (1 : ℝ)^4 + (1 : ℝ)^3 + b*(1 : ℝ)^2 + c*(1 : ℝ) + 200 = 132 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l332_33257


namespace NUMINAMATH_CALUDE_rectangle_area_l332_33231

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 96,
    prove that its area is 432. -/
theorem rectangle_area (breadth : ℝ) (length : ℝ) : 
  length = 3 * breadth → 
  2 * (length + breadth) = 96 → 
  length * breadth = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l332_33231


namespace NUMINAMATH_CALUDE_smallest_n_cookies_l332_33235

theorem smallest_n_cookies (n : ℕ) : (∃ k : ℕ, 15 * n - 1 = 11 * k) ↔ n ≥ 3 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_cookies_l332_33235


namespace NUMINAMATH_CALUDE_album_jumps_l332_33221

/-- Calculates the total number of jumps a person can make while listening to an album. -/
theorem album_jumps (jumps_per_second : ℕ) (song_length : ℚ) (num_songs : ℕ) :
  jumps_per_second = 1 →
  song_length = 3.5 →
  num_songs = 10 →
  (jumps_per_second * 60 : ℚ) * (song_length * num_songs) = 2100 := by
  sorry

end NUMINAMATH_CALUDE_album_jumps_l332_33221


namespace NUMINAMATH_CALUDE_julie_school_year_earnings_l332_33238

/-- Julie's summer work details and school year work conditions -/
structure WorkDetails where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_weeks : ℕ
  school_hours_per_week : ℕ
  rate_increase : ℚ

/-- Calculate Julie's school year earnings based on her work details -/
def calculate_school_year_earnings (w : WorkDetails) : ℚ :=
  let summer_hourly_rate := w.summer_earnings / (w.summer_weeks * w.summer_hours_per_week)
  let school_hourly_rate := summer_hourly_rate * (1 + w.rate_increase)
  school_hourly_rate * w.school_weeks * w.school_hours_per_week

/-- Theorem stating that Julie's school year earnings are $3750 -/
theorem julie_school_year_earnings :
  let w : WorkDetails := {
    summer_weeks := 10,
    summer_hours_per_week := 40,
    summer_earnings := 4000,
    school_weeks := 30,
    school_hours_per_week := 10,
    rate_increase := 1/4
  }
  calculate_school_year_earnings w = 3750 := by sorry

end NUMINAMATH_CALUDE_julie_school_year_earnings_l332_33238


namespace NUMINAMATH_CALUDE_larger_number_problem_l332_33256

theorem larger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 147)
  (relation : x = 0.375 * y + 4)
  (x_larger : x > y) : 
  x = 43 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l332_33256


namespace NUMINAMATH_CALUDE_min_transactions_to_identify_coins_l332_33218

/-- Represents the set of coin values available -/
def CoinValues : Finset Nat := {1, 2, 5, 10, 20}

/-- The cost of one candy in florins -/
def CandyCost : Nat := 1

/-- Represents a vending machine transaction -/
structure Transaction where
  coin_inserted : Nat
  change_returned : Nat

/-- Function to determine if all coin values can be identified -/
def can_identify_all_coins (transactions : List Transaction) : Prop :=
  ∀ c ∈ CoinValues, ∃ t ∈ transactions, t.coin_inserted = c ∨ t.change_returned = c - CandyCost

/-- The main theorem stating that 4 is the minimum number of transactions required -/
theorem min_transactions_to_identify_coins :
  (∃ transactions : List Transaction, transactions.length = 4 ∧ can_identify_all_coins transactions) ∧
  (∀ transactions : List Transaction, transactions.length < 4 → ¬ can_identify_all_coins transactions) :=
sorry

end NUMINAMATH_CALUDE_min_transactions_to_identify_coins_l332_33218


namespace NUMINAMATH_CALUDE_aisha_has_largest_answer_l332_33262

def starting_number : ℕ := 15

def maria_calculation (n : ℕ) : ℕ := ((n - 2) * 3) + 5

def liam_calculation (n : ℕ) : ℕ := (n * 3 - 2) + 5

def aisha_calculation (n : ℕ) : ℕ := ((n - 2) + 5) * 3

theorem aisha_has_largest_answer :
  aisha_calculation starting_number > maria_calculation starting_number ∧
  aisha_calculation starting_number > liam_calculation starting_number :=
sorry

end NUMINAMATH_CALUDE_aisha_has_largest_answer_l332_33262


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l332_33224

/-- Represents the cost price of cloth per meter -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (loss_per_meter : ℕ) : ℚ :=
  (selling_price + total_meters * loss_per_meter) / total_meters

/-- Proves that the cost price per meter is 45 given the problem conditions -/
theorem shopkeeper_cloth_cost_price :
  cost_price_per_meter 450 18000 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l332_33224


namespace NUMINAMATH_CALUDE_item_price_ratio_l332_33229

theorem item_price_ratio (c p q : ℝ) (h1 : p = 0.8 * c) (h2 : q = 1.2 * c) : q / p = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_item_price_ratio_l332_33229


namespace NUMINAMATH_CALUDE_kim_morning_routine_time_l332_33205

/-- Represents the time in minutes for Kim's morning routine -/
def morning_routine_time (coffee_time : ℕ) (status_update_time : ℕ) (payroll_update_time : ℕ) (num_employees : ℕ) : ℕ :=
  coffee_time + (status_update_time + payroll_update_time) * num_employees

/-- Theorem stating that Kim's morning routine takes 50 minutes -/
theorem kim_morning_routine_time :
  morning_routine_time 5 2 3 9 = 50 := by
  sorry

#eval morning_routine_time 5 2 3 9

end NUMINAMATH_CALUDE_kim_morning_routine_time_l332_33205


namespace NUMINAMATH_CALUDE_equation_and_inequality_solution_l332_33263

theorem equation_and_inequality_solution :
  (∃ x : ℝ, (x - 3) * (x - 2) + 18 = (x + 9) * (x + 1) ∧ x = 1) ∧
  (∀ x : ℝ, (3 * x + 4) * (3 * x - 4) < 9 * (x - 2) * (x + 3) ↔ x > 38 / 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_and_inequality_solution_l332_33263


namespace NUMINAMATH_CALUDE_cut_difference_l332_33204

/-- The amount cut off the skirt in inches -/
def skirt_cut : ℝ := 0.75

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The difference between the amount cut off the skirt and the amount cut off the pants -/
theorem cut_difference : skirt_cut - pants_cut = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_cut_difference_l332_33204


namespace NUMINAMATH_CALUDE_number_exists_l332_33209

theorem number_exists : ∃ N : ℝ, (N / 10 - N / 1000) = 700 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l332_33209


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l332_33252

/-- The slope angle of the line x + √3 * y - 5 = 0 is 150 degrees. -/
theorem slope_angle_of_line (x y : ℝ) : 
  x + Real.sqrt 3 * y - 5 = 0 → 
  ∃ α : ℝ, α = 150 * π / 180 ∧ (Real.tan α = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l332_33252


namespace NUMINAMATH_CALUDE_soccer_league_games_l332_33291

/-- The total number of games played in a soccer league. -/
def totalGames (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264. -/
theorem soccer_league_games :
  totalGames 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l332_33291


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l332_33219

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l332_33219


namespace NUMINAMATH_CALUDE_original_number_l332_33261

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 117 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l332_33261


namespace NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_16x_l332_33269

theorem factorization_of_4x_cubed_minus_16x (x : ℝ) : 4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_16x_l332_33269


namespace NUMINAMATH_CALUDE_average_salary_before_manager_l332_33285

/-- Proves that the average salary of employees is 1500 given the conditions -/
theorem average_salary_before_manager (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) :
  num_employees = 20 →
  manager_salary = 12000 →
  avg_increase = 500 →
  (∃ (avg_salary : ℕ),
    (num_employees + 1) * (avg_salary + avg_increase) = num_employees * avg_salary + manager_salary ∧
    avg_salary = 1500) :=
by sorry

end NUMINAMATH_CALUDE_average_salary_before_manager_l332_33285


namespace NUMINAMATH_CALUDE_joan_balloons_l332_33216

/-- The number of blue balloons Joan has after gaining more -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

/-- Theorem stating that Joan has 95 blue balloons after gaining more -/
theorem joan_balloons : total_balloons 72 23 = 95 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l332_33216


namespace NUMINAMATH_CALUDE_curve_is_line_l332_33234

/-- The curve represented by the equation (x+2y-1)√(x²+y²-2x+2)=0 is a line. -/
theorem curve_is_line : 
  ∀ (x y : ℝ), (x + 2*y - 1) * Real.sqrt (x^2 + y^2 - 2*x + 2) = 0 ↔ x + 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_curve_is_line_l332_33234


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l332_33226

theorem sum_of_square_areas (side1 side2 : ℝ) (h1 : side1 = 11) (h2 : side2 = 5) :
  side1 * side1 + side2 * side2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l332_33226


namespace NUMINAMATH_CALUDE_circle_center_in_third_quadrant_l332_33272

/-- A line passes through the first, second, and third quadrants -/
structure LineInQuadrants (a b : ℝ) : Prop :=
  (passes_through_123 : a > 0 ∧ b > 0)

/-- A circle with center (-a, -b) and radius r -/
structure Circle (a b r : ℝ) : Prop :=
  (positive_radius : r > 0)

/-- The third quadrant -/
def ThirdQuadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem circle_center_in_third_quadrant
  (a b r : ℝ) (line : LineInQuadrants a b) (circle : Circle a b r) :
  ThirdQuadrant (-a) (-b) :=
sorry

end NUMINAMATH_CALUDE_circle_center_in_third_quadrant_l332_33272


namespace NUMINAMATH_CALUDE_area_under_arcsin_cos_l332_33273

noncomputable def f (x : ℝ) := Real.arcsin (Real.cos x)

theorem area_under_arcsin_cos : ∫ x in (0)..(3 * Real.pi), |f x| = (3 * Real.pi^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_under_arcsin_cos_l332_33273


namespace NUMINAMATH_CALUDE_exp_log_properties_l332_33287

-- Define the exponential and logarithmic functions
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for the properties of exponential and logarithmic functions
theorem exp_log_properties :
  -- Domain and range of exponential function
  (∀ x : ℝ, ∃ y : ℝ, exp 2 x = y) ∧
  (∀ y : ℝ, y > 0 → ∃ x : ℝ, exp 2 x = y) ∧
  (exp 2 0 = 1) ∧
  -- Domain and range of logarithmic function
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, log 2 x = y) ∧
  (∀ y : ℝ, ∃ x : ℝ, x > 0 ∧ log 2 x = y) ∧
  (log 2 1 = 0) ∧
  -- Logarithm properties
  (∀ a M N : ℝ, a > 0 ∧ a ≠ 1 ∧ M > 0 ∧ N > 0 →
    log a (M * N) = log a M + log a N) ∧
  (∀ a N : ℝ, a > 0 ∧ a ≠ 1 ∧ N > 0 →
    exp a (log a N) = N) ∧
  (∀ a b m n : ℝ, a > 0 ∧ a ≠ 1 ∧ b > 0 ∧ m ≠ 0 →
    log (exp a m) (exp b n) = (n / m) * log a b) :=
by sorry

#check exp_log_properties

end NUMINAMATH_CALUDE_exp_log_properties_l332_33287


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l332_33206

theorem tan_sum_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 :=
by
  have h : Real.tan (45 * π / 180) = (Real.tan (10 * π / 180) + Real.tan (35 * π / 180)) /
    (1 - Real.tan (10 * π / 180) * Real.tan (35 * π / 180)) := by sorry
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l332_33206
