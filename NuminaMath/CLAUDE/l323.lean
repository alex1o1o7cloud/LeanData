import Mathlib

namespace handshake_count_l323_32355

def number_of_couples : ℕ := 15

def number_of_people : ℕ := 2 * number_of_couples

def handshakes_among_men : ℕ := (number_of_couples - 1) * (number_of_couples - 2) / 2

def handshakes_between_men_and_women : ℕ := number_of_couples * (number_of_couples - 1)

def total_handshakes : ℕ := handshakes_among_men + handshakes_between_men_and_women

theorem handshake_count : total_handshakes = 301 := by
  sorry

end handshake_count_l323_32355


namespace profit_calculation_l323_32352

theorem profit_calculation (cost_price : ℝ) (x : ℝ) : 
  (40 * cost_price = x * (cost_price * 1.25)) → x = 32 :=
by
  sorry

end profit_calculation_l323_32352


namespace inequality_equivalence_l323_32346

theorem inequality_equivalence (x : ℝ) (h : x > 0) : 
  3/8 + |x - 14/24| < 8/12 ↔ 7/24 < x ∧ x < 7/8 := by
sorry

end inequality_equivalence_l323_32346


namespace number_of_subsets_l323_32391

/-- For a finite set with n elements, the number of subsets is 2^n -/
theorem number_of_subsets (S : Type*) [Fintype S] : 
  Finset.card (Finset.powerset (Finset.univ : Finset S)) = 2 ^ Fintype.card S := by
  sorry

end number_of_subsets_l323_32391


namespace prime_squared_product_l323_32322

theorem prime_squared_product (p q : ℕ) : 
  Prime p → Prime q → Nat.totient (p^2 * q^2) = 11424 → p^2 * q^2 = 7^2 * 17^2 := by
  sorry

end prime_squared_product_l323_32322


namespace min_lamps_l323_32363

theorem min_lamps (n p : ℕ) (h1 : p > 0) : 
  (∃ (p : ℕ), p > 0 ∧ 
    (p + 10*n - 30) - p = 100 ∧ 
    (∀ m : ℕ, m < n → ¬(∃ (q : ℕ), q > 0 ∧ (q + 10*m - 30) - q = 100))) → 
  n = 13 := by
sorry

end min_lamps_l323_32363


namespace alice_unanswered_questions_l323_32368

/-- Represents a scoring system for a math competition -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int
  initial : Int

/-- Represents the results of a math competition -/
structure CompetitionResult where
  correct : Nat
  wrong : Nat
  unanswered : Nat
  total_questions : Nat
  new_score : Int
  old_score : Int

def new_system : ScoringSystem := ⟨6, 0, 3, 0⟩
def old_system : ScoringSystem := ⟨5, -2, 0, 20⟩

/-- Calculates the score based on a given scoring system and competition result -/
def calculate_score (system : ScoringSystem) (result : CompetitionResult) : Int :=
  system.initial + 
  system.correct * result.correct + 
  system.wrong * result.wrong + 
  system.unanswered * result.unanswered

theorem alice_unanswered_questions 
  (result : CompetitionResult)
  (h1 : result.new_score = 105)
  (h2 : result.old_score = 75)
  (h3 : result.total_questions = 30)
  (h4 : calculate_score new_system result = result.new_score)
  (h5 : calculate_score old_system result = result.old_score)
  (h6 : result.correct + result.wrong + result.unanswered = result.total_questions) :
  result.unanswered = 5 := by
  sorry

#check alice_unanswered_questions

end alice_unanswered_questions_l323_32368


namespace cube_construction_problem_l323_32349

theorem cube_construction_problem :
  ∃! (a b c : ℕ+), a^3 + b^3 + c^3 + 648 = (a + b + c)^3 :=
sorry

end cube_construction_problem_l323_32349


namespace a_over_two_plus_a_is_fraction_l323_32305

/-- Definition of a fraction -/
def is_fraction (x y : ℝ) : Prop := ∃ (a b : ℝ), x = a ∧ y = b ∧ b ≠ 0

/-- The expression a / (2 + a) is a fraction -/
theorem a_over_two_plus_a_is_fraction (a : ℝ) : is_fraction a (2 + a) := by
  sorry

end a_over_two_plus_a_is_fraction_l323_32305


namespace combined_shape_perimeter_l323_32333

/-- The perimeter of a combined shape of a right triangle and rectangle -/
theorem combined_shape_perimeter (a b c d : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 10) 
  (h4 : d^2 = a^2 + b^2) : a + b + c + d = 22 := by
  sorry

end combined_shape_perimeter_l323_32333


namespace product_in_base_10_l323_32329

-- Define the binary number 11001₂
def binary_num : ℕ := 25

-- Define the ternary number 112₃
def ternary_num : ℕ := 14

-- Theorem to prove
theorem product_in_base_10 : binary_num * ternary_num = 350 := by
  sorry

end product_in_base_10_l323_32329


namespace line_proof_l323_32323

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y + 1 = 0
def line2 (x y : ℝ) : Prop := x - 3*y + 4 = 0
def line3 (x y : ℝ) : Prop := 3*x + 4*y - 7 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := y = (4/3)*x + (1/9)

-- Theorem statement
theorem line_proof :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧
    (result_line x₀ y₀) ∧
    (∀ (x y : ℝ), line3 x y → (y - y₀) = -(3/4) * (x - x₀)) :=
by sorry

end line_proof_l323_32323


namespace factorization_of_2x2_minus_2y2_l323_32376

theorem factorization_of_2x2_minus_2y2 (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end factorization_of_2x2_minus_2y2_l323_32376


namespace right_triangle_trig_inequality_l323_32334

theorem right_triangle_trig_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : A < π/4) 
  (h3 : A + B + C = π) (h4 : C = π/2) : Real.cos B < Real.sin B := by
  sorry

end right_triangle_trig_inequality_l323_32334


namespace ratio_fraction_value_l323_32303

-- Define the ratio condition
def ratio_condition (X Y Z : ℚ) : Prop :=
  X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- State the theorem
theorem ratio_fraction_value (X Y Z : ℚ) (h : ratio_condition X Y Z) :
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end ratio_fraction_value_l323_32303


namespace system_solvable_iff_l323_32370

/-- The system of equations -/
def system (b a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*b*(b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)

/-- The theorem stating the condition for the existence of a solution -/
theorem system_solvable_iff (b : ℝ) :
  (∃ a : ℝ, ∃ x y : ℝ, system b a x y) ↔ -14 < b ∧ b < 9 :=
sorry

end system_solvable_iff_l323_32370


namespace company_managers_count_l323_32372

theorem company_managers_count 
  (num_associates : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ) 
  (h1 : num_associates = 75)
  (h2 : avg_salary_managers = 90000)
  (h3 : avg_salary_associates = 30000)
  (h4 : avg_salary_company = 40000) :
  ∃ (num_managers : ℕ), 
    (num_managers : ℝ) * avg_salary_managers + (num_associates : ℝ) * avg_salary_associates = 
    ((num_managers : ℝ) + (num_associates : ℝ)) * avg_salary_company ∧ 
    num_managers = 15 := by
  sorry

end company_managers_count_l323_32372


namespace absolute_value_quadratic_equation_solution_l323_32374

theorem absolute_value_quadratic_equation_solution :
  let y₁ : ℝ := (-1 + Real.sqrt 241) / 6
  let y₂ : ℝ := (1 - Real.sqrt 145) / 6
  (|y₁ - 4| + 3 * y₁^2 = 16) ∧ (|y₂ - 4| + 3 * y₂^2 = 16) := by
  sorry

end absolute_value_quadratic_equation_solution_l323_32374


namespace existence_of_prime_and_integer_l323_32339

theorem existence_of_prime_and_integer (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (n : ℕ+), q.Prime ∧ q < p ∧ p ∣ n.val^2 - q := by
  sorry

end existence_of_prime_and_integer_l323_32339


namespace four_roots_implies_a_in_interval_l323_32380

-- Define the polynomial
def P (x a : ℝ) : ℝ := x^4 + 8*x^3 + 18*x^2 + 8*x + a

-- Define the property of having four distinct real roots
def has_four_distinct_real_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (P x₁ a = 0 ∧ P x₂ a = 0 ∧ P x₃ a = 0 ∧ P x₄ a = 0)

-- Theorem statement
theorem four_roots_implies_a_in_interval :
  ∀ a : ℝ, has_four_distinct_real_roots a → a ∈ Set.Ioo (-8 : ℝ) 1 :=
by sorry

end four_roots_implies_a_in_interval_l323_32380


namespace square_of_95_l323_32342

theorem square_of_95 : 95^2 = 9025 := by
  sorry

end square_of_95_l323_32342


namespace base_five_product_l323_32398

/-- Represents a number in base 5 --/
def BaseFive : Type := ℕ

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : BaseFive := sorry

/-- Multiplies two numbers in base 5 --/
def multBaseFive (a b : BaseFive) : BaseFive := sorry

/-- Adds two numbers in base 5 --/
def addBaseFive (a b : BaseFive) : BaseFive := sorry

theorem base_five_product :
  let a : BaseFive := toBaseFive 121
  let b : BaseFive := toBaseFive 11
  let c : BaseFive := toBaseFive 1331
  multBaseFive a b = c := by sorry

end base_five_product_l323_32398


namespace total_sum_calculation_l323_32335

/-- Given a sum to be divided among four parts in the ratio 5 : 9 : 6 : 5,
    if the sum of the first and third parts is $7022.222222222222,
    then the total sum is $15959.59595959596. -/
theorem total_sum_calculation (a b c d : ℝ) : 
  a / 5 = b / 9 ∧ a / 5 = c / 6 ∧ a / 5 = d / 5 →
  a + c = 7022.222222222222 →
  a + b + c + d = 15959.59595959596 := by
  sorry

end total_sum_calculation_l323_32335


namespace actual_height_is_236_l323_32326

/-- The actual height of a boy in a class, given the following conditions:
  * There are 35 boys in the class
  * The initial average height was calculated as 185 cm
  * One boy's height was wrongly written as 166 cm
  * The actual average height is 183 cm
-/
def actual_height : ℕ :=
  let num_boys : ℕ := 35
  let initial_avg : ℕ := 185
  let wrong_height : ℕ := 166
  let actual_avg : ℕ := 183
  let initial_total : ℕ := num_boys * initial_avg
  let actual_total : ℕ := num_boys * actual_avg
  let height_difference : ℕ := initial_total - actual_total
  wrong_height + height_difference

theorem actual_height_is_236 : actual_height = 236 := by
  sorry

end actual_height_is_236_l323_32326


namespace polynomial_root_mean_l323_32318

theorem polynomial_root_mean (a b c d k : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (k - a) * (k - b) * (k - c) * (k - d) = 4 →
  k = (a + b + c + d) / 4 := by
sorry

end polynomial_root_mean_l323_32318


namespace parallel_vectors_condition_l323_32361

/-- Given vectors a and b, where a is parallel to their sum, prove that the y-component of b is -3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) : 
  a = (-1, 1) → 
  b.1 = 3 → 
  ∃ (k : ℝ), k • a = (a + b) → 
  b.2 = -3 := by
  sorry

end parallel_vectors_condition_l323_32361


namespace lowry_big_bonsai_sold_l323_32300

/-- Represents the sale of bonsai trees -/
structure BonsaiSale where
  small_price : ℕ  -- Price of a small bonsai
  big_price : ℕ    -- Price of a big bonsai
  small_sold : ℕ   -- Number of small bonsai sold
  total_earnings : ℕ -- Total earnings from the sale

/-- Calculates the number of big bonsai sold -/
def big_bonsai_sold (sale : BonsaiSale) : ℕ :=
  (sale.total_earnings - sale.small_price * sale.small_sold) / sale.big_price

/-- Theorem stating the number of big bonsai sold in Lowry's sale -/
theorem lowry_big_bonsai_sold :
  let sale := BonsaiSale.mk 30 20 3 190
  big_bonsai_sold sale = 5 := by
  sorry

end lowry_big_bonsai_sold_l323_32300


namespace complex_magnitude_equation_l323_32306

theorem complex_magnitude_equation (m : ℝ) (h : m > 0) : 
  Complex.abs (5 + m * Complex.I) = 5 * Real.sqrt 26 → m = 25 := by
  sorry

end complex_magnitude_equation_l323_32306


namespace book_sale_amount_l323_32383

/-- Calculates the total amount received from selling books given the following conditions:
  * A fraction of the books were sold
  * A certain number of books remained unsold
  * Each sold book was sold at a fixed price
-/
def totalAmountReceived (fractionSold : Rat) (remainingBooks : Nat) (pricePerBook : Rat) : Rat :=
  let totalBooks := remainingBooks / (1 - fractionSold)
  let soldBooks := totalBooks * fractionSold
  soldBooks * pricePerBook

/-- Proves that given the specific conditions of the book sale, 
    the total amount received is $255 -/
theorem book_sale_amount : 
  totalAmountReceived (2/3) 30 (21/5) = 255 := by
  sorry

end book_sale_amount_l323_32383


namespace mall_promotion_max_purchase_l323_32366

/-- Calculates the maximum value of goods that can be purchased given a cashback rule and initial amount --/
def max_purchase_value (cashback_amount : ℕ) (cashback_threshold : ℕ) (initial_amount : ℕ) : ℕ :=
  sorry

/-- The maximum value of goods that can be purchased given the specific conditions --/
theorem mall_promotion_max_purchase :
  max_purchase_value 40 200 650 = 770 :=
sorry

end mall_promotion_max_purchase_l323_32366


namespace unique_solution_for_exponential_equation_l323_32315

theorem unique_solution_for_exponential_equation :
  ∀ a b p : ℕ+,
    p.val.Prime →
    2^(a.val) + p^(b.val) = 19^(a.val) →
    a = 1 ∧ b = 1 ∧ p = 17 :=
by sorry

end unique_solution_for_exponential_equation_l323_32315


namespace quadratic_equation_k_value_l323_32304

theorem quadratic_equation_k_value (x₁ x₂ k : ℝ) : 
  x₁^2 - 3*x₁ + k = 0 →
  x₂^2 - 3*x₂ + k = 0 →
  x₁ * x₂ + 2*x₁ + 2*x₂ = 1 →
  k = -5 := by
sorry

end quadratic_equation_k_value_l323_32304


namespace shaded_area_fraction_l323_32393

/-- Represents a square with two internal unshaded squares -/
structure SquareWithInternalSquares where
  /-- Side length of the large square -/
  side : ℝ
  /-- Side length of the bottom-left unshaded square -/
  bottomLeftSide : ℝ
  /-- Side length of the top-right unshaded square -/
  topRightSide : ℝ
  /-- The bottom-left square's side is half of the large square's side -/
  bottomLeftHalf : bottomLeftSide = side / 2
  /-- The top-right square's diagonal is one-third of the large square's diagonal -/
  topRightThird : topRightSide * Real.sqrt 2 = side * Real.sqrt 2 / 3

/-- The fraction of the shaded area in a square with two internal unshaded squares is 19/36 -/
theorem shaded_area_fraction (s : SquareWithInternalSquares) :
  (s.side^2 - s.bottomLeftSide^2 - s.topRightSide^2) / s.side^2 = 19 / 36 := by
  sorry

end shaded_area_fraction_l323_32393


namespace algebraic_equality_l323_32302

theorem algebraic_equality (a b c k m n : ℝ) 
  (h1 : b^2 - n^2 = a^2 - k^2) 
  (h2 : a^2 - k^2 = c^2 - m^2) : 
  (b*m - c*n)/(a - k) + (c*k - a*m)/(b - n) + (a*n - b*k)/(c - m) = 0 := by
  sorry

end algebraic_equality_l323_32302


namespace largest_n_for_exponential_inequality_l323_32343

theorem largest_n_for_exponential_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), Real.exp (n * x) + Real.exp (-n * x) ≥ n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), Real.exp (m * y) + Real.exp (-m * y) < m) :=
by sorry

end largest_n_for_exponential_inequality_l323_32343


namespace absolute_value_equality_l323_32362

theorem absolute_value_equality (x : ℝ) : 
  |(-x)| = |(-8)| → x = 8 ∨ x = -8 := by
  sorry

end absolute_value_equality_l323_32362


namespace work_break_difference_l323_32311

/-- Calculates the difference between water breaks and sitting breaks
    given work duration and break intervals. -/
def break_difference (work_duration : ℕ) (water_interval : ℕ) (sitting_interval : ℕ) : ℕ :=
  (work_duration / water_interval) - (work_duration / sitting_interval)

/-- Proves that for 240 minutes of work, with water breaks every 20 minutes
    and sitting breaks every 120 minutes, there are 10 more water breaks than sitting breaks. -/
theorem work_break_difference :
  break_difference 240 20 120 = 10 := by
  sorry

end work_break_difference_l323_32311


namespace min_value_trig_fraction_l323_32325

theorem min_value_trig_fraction (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 3 ≥ (2/3) * ((Real.sin x)^6 + (Real.cos x)^6 + 3) := by
  sorry

end min_value_trig_fraction_l323_32325


namespace gcd_2952_1386_l323_32371

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end gcd_2952_1386_l323_32371


namespace hyperbola_eccentricity_l323_32365

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (C : Hyperbola) (P F₁ F₂ Q O : Point) : 
  (P.x^2 / C.a^2 - P.y^2 / C.b^2 = 1) →  -- P is on the hyperbola
  (F₁.x < 0 ∧ F₁.y = 0 ∧ F₂.x > 0 ∧ F₂.y = 0) →  -- F₁ and F₂ are left and right foci
  ((P.x - F₂.x) * (F₁.x - F₂.x) + (P.y - F₂.y) * (F₁.y - F₂.y) = 0) →  -- PF₂ ⟂ F₁F₂
  (∃ t : ℝ, Q.x = 0 ∧ Q.y = t * P.y + (1 - t) * F₁.y) →  -- PF₁ intersects y-axis at Q
  (O.x = 0 ∧ O.y = 0) →  -- O is the origin
  (∃ M : Point, ∃ r : ℝ, 
    (M.x - O.x)^2 + (M.y - O.y)^2 = r^2 ∧
    (M.x - F₂.x)^2 + (M.y - F₂.y)^2 = r^2 ∧
    (M.x - P.x)^2 + (M.y - P.y)^2 = r^2 ∧
    (M.x - Q.x)^2 + (M.y - Q.y)^2 = r^2) →  -- OF₂PQ has an inscribed circle
  (F₂.x^2 - F₁.x^2) / C.a^2 = 4  -- Eccentricity is 2
:= by sorry

end hyperbola_eccentricity_l323_32365


namespace common_ratio_is_four_l323_32381

/-- Geometric sequence with sum S_n of first n terms -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

theorem common_ratio_is_four 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : geometric_sequence a S)
  (h1 : 3 * S 3 = a 4 - 2)
  (h2 : 3 * S 2 = a 3 - 2) :
  a 2 / a 1 = 4 := by sorry

end common_ratio_is_four_l323_32381


namespace restaurant_bill_calculation_l323_32324

theorem restaurant_bill_calculation (num_adults num_teenagers num_children : ℕ)
  (adult_meal_cost teenager_meal_cost child_meal_cost : ℚ)
  (soda_cost dessert_cost appetizer_cost : ℚ)
  (num_desserts num_appetizers : ℕ)
  (h1 : num_adults = 6)
  (h2 : num_teenagers = 3)
  (h3 : num_children = 1)
  (h4 : adult_meal_cost = 9)
  (h5 : teenager_meal_cost = 7)
  (h6 : child_meal_cost = 5)
  (h7 : soda_cost = 2.5)
  (h8 : dessert_cost = 4)
  (h9 : appetizer_cost = 6)
  (h10 : num_desserts = 3)
  (h11 : num_appetizers = 2) :
  (num_adults * adult_meal_cost +
   num_teenagers * teenager_meal_cost +
   num_children * child_meal_cost +
   (num_adults + num_teenagers + num_children) * soda_cost +
   num_desserts * dessert_cost +
   num_appetizers * appetizer_cost) = 129 :=
by sorry

end restaurant_bill_calculation_l323_32324


namespace cos_120_degrees_l323_32350

theorem cos_120_degrees : Real.cos (2 * π / 3) = -1 / 2 := by
  sorry

end cos_120_degrees_l323_32350


namespace simplify_power_expression_l323_32395

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by
  sorry

end simplify_power_expression_l323_32395


namespace hooper_bay_lobster_ratio_l323_32369

/-- The ratio of lobster in Hooper Bay to other harbors -/
theorem hooper_bay_lobster_ratio :
  let total_lobster : ℕ := 480
  let other_harbors_lobster : ℕ := 80 + 80
  let hooper_bay_lobster : ℕ := total_lobster - other_harbors_lobster
  (hooper_bay_lobster : ℚ) / other_harbors_lobster = 2 := by
  sorry

end hooper_bay_lobster_ratio_l323_32369


namespace population_increase_rate_example_l323_32384

/-- Given an initial population and a final population after one year,
    calculate the population increase rate as a percentage. -/
def population_increase_rate (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem stating that for an initial population of 220 and
    a final population of 242, the increase rate is 10%. -/
theorem population_increase_rate_example :
  population_increase_rate 220 242 = 10 := by
  sorry

end population_increase_rate_example_l323_32384


namespace min_b_value_l323_32313

/-- Given a parabola and a circle with specific intersection properties, 
    the minimum value of b is 2 -/
theorem min_b_value (k a b r : ℝ) : 
  k > 0 → 
  (∀ x y, y = k * x^2 → (x - a)^2 + (y - b)^2 = r^2 → 
    (x = 0 ∧ y = 0) ∨ y = k * x + b) →
  a^2 + b^2 = r^2 →
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    k * x₁^2 = (x₁ - a)^2 + (k * x₁^2 - b)^2 - r^2 ∧
    k * x₂^2 = (x₂ - a)^2 + (k * x₂^2 - b)^2 - r^2 ∧
    k * x₃^2 = (x₃ - a)^2 + (k * x₃^2 - b)^2 - r^2) →
  b ≥ 2 :=
by sorry

end min_b_value_l323_32313


namespace sum_of_squares_of_roots_l323_32396

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 10*x + 9 = 0 → ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 82 ∧ (x = s₁ ∨ x = s₂) := by
  sorry

end sum_of_squares_of_roots_l323_32396


namespace y_axis_inclination_l323_32316

-- Define the concept of an axis
def Axis : Type := ℝ → ℝ

-- Define the x-axis and y-axis
def x_axis : Axis := λ x => 0
def y_axis : Axis := λ y => y

-- Define the concept of perpendicular axes
def perpendicular (a b : Axis) : Prop := sorry

-- Define the concept of inclination angle
def inclination_angle (a : Axis) : ℝ := sorry

-- Theorem statement
theorem y_axis_inclination :
  perpendicular x_axis y_axis →
  inclination_angle y_axis = 90 :=
sorry

end y_axis_inclination_l323_32316


namespace emails_remaining_proof_l323_32308

/-- Given an initial number of emails, calculates the number of emails remaining in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
def remaining_emails (initial : ℕ) : ℕ :=
  let after_trash := initial / 2
  let to_work_folder := (40 * after_trash) / 100
  after_trash - to_work_folder

/-- Proves that given 400 initial emails, 120 emails remain in the inbox after the operations. -/
theorem emails_remaining_proof :
  remaining_emails 400 = 120 := by
  sorry

#eval remaining_emails 400  -- Should output 120

end emails_remaining_proof_l323_32308


namespace slope_range_of_intersecting_line_l323_32344

/-- Given points A, B, and P, and a line l passing through P and intersecting line segment AB,
    prove that the range of the slope of line l is [0, π/4] ∪ [3π/4, π). -/
theorem slope_range_of_intersecting_line (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) 
    (hA : A = (1, -2))
    (hB : B = (2, 1))
    (hP : P = (0, -1))
    (hl : P ∈ l)
    (hintersect : ∃ Q ∈ l, Q ∈ Set.Icc A B) :
  ∃ s : Set ℝ, s = Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π ∧
    ∀ θ : ℝ, (∃ Q ∈ l, Q ≠ P ∧ Real.tan θ = (Q.2 - P.2) / (Q.1 - P.1)) → θ ∈ s :=
sorry

end slope_range_of_intersecting_line_l323_32344


namespace simplify_2A_minus_3B_specific_value_2A_minus_3B_l323_32379

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := 4 * a * b + 2 * b^2 - a^2

/-- Theorem stating that 2A - 3B simplifies to -a² - 2ab for all real a and b -/
theorem simplify_2A_minus_3B (a b : ℝ) : 2 * A a b - 3 * B a b = -a^2 - 2*a*b := by
  sorry

/-- Theorem stating that when a = -1 and b = 4, 2A - 3B equals 7 -/
theorem specific_value_2A_minus_3B : 2 * A (-1) 4 - 3 * B (-1) 4 = 7 := by
  sorry

end simplify_2A_minus_3B_specific_value_2A_minus_3B_l323_32379


namespace total_strawberries_l323_32312

/-- The number of strawberries picked by Jonathan and Matthew together -/
def jonathan_matthew_total : ℕ := 350

/-- The number of strawberries picked by Matthew and Zac together -/
def matthew_zac_total : ℕ := 250

/-- The number of strawberries picked by Zac alone -/
def zac_alone : ℕ := 200

/-- Theorem stating that the total number of strawberries picked is 550 -/
theorem total_strawberries : 
  ∃ (j m z : ℕ), 
    j + m = jonathan_matthew_total ∧ 
    m + z = matthew_zac_total ∧ 
    z = zac_alone ∧ 
    j + m + z = 550 := by
  sorry

end total_strawberries_l323_32312


namespace union_equals_reals_l323_32331

-- Define sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Theorem statement
theorem union_equals_reals : A ∪ B = Set.univ := by sorry

end union_equals_reals_l323_32331


namespace perpendicular_vectors_difference_magnitude_l323_32348

/-- Given plane vectors a = (-2, k) and b = (2, 4), if a is perpendicular to b, 
    then |a - b| = 5 -/
theorem perpendicular_vectors_difference_magnitude 
  (k : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (-2, k)) 
  (hb : b = (2, 4)) 
  (hperp : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖a - b‖ = 5 := by
  sorry

end perpendicular_vectors_difference_magnitude_l323_32348


namespace edge_sum_is_112_l323_32359

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  x : ℝ
  d : ℝ
  volume_eq : x^3 * (d + 1)^3 = 512
  surface_area_eq : 2 * (x^2 * (d + 1) + x^2 * (d + 1)^2 + x^2 * (d + 1)^3) = 448

/-- The sum of the lengths of all edges of the rectangular solid -/
def edge_sum (solid : RectangularSolid) : ℝ :=
  4 * (solid.x + solid.x * (solid.d + 1) + solid.x * (solid.d + 1)^2)

/-- Theorem stating that the sum of the lengths of all edges is 112 -/
theorem edge_sum_is_112 (solid : RectangularSolid) : edge_sum solid = 112 := by
  sorry

#check edge_sum_is_112

end edge_sum_is_112_l323_32359


namespace exactly_one_truck_congestion_at_least_two_trucks_congestion_l323_32307

-- Define the probabilities for highways I and II
def prob_congestion_I : ℚ := 1/10
def prob_no_congestion_I : ℚ := 9/10
def prob_congestion_II : ℚ := 3/5
def prob_no_congestion_II : ℚ := 2/5

-- Define the events
def event_A : ℚ := prob_congestion_I
def event_B : ℚ := prob_congestion_I
def event_C : ℚ := prob_congestion_II

-- Theorem for the first question
theorem exactly_one_truck_congestion :
  prob_congestion_I * prob_no_congestion_I + prob_no_congestion_I * prob_congestion_I = 9/50 := by sorry

-- Theorem for the second question
theorem at_least_two_trucks_congestion :
  event_A * event_B * (1 - event_C) + 
  event_A * (1 - event_B) * event_C + 
  (1 - event_A) * event_B * event_C + 
  event_A * event_B * event_C = 59/500 := by sorry

end exactly_one_truck_congestion_at_least_two_trucks_congestion_l323_32307


namespace gcd_of_three_numbers_l323_32345

theorem gcd_of_three_numbers : Nat.gcd 10234 (Nat.gcd 14322 24570) = 18 := by
  sorry

end gcd_of_three_numbers_l323_32345


namespace hyperbola_equation_l323_32341

/-- The set of complex numbers z satisfying the equation (1/5)^|z-3| = (1/5)^(|z+3|-1) 
    forms a hyperbola with foci on the x-axis, a real semi-axis length of 1/2, 
    and specifically represents the right branch. -/
theorem hyperbola_equation (z : ℂ) : 
  (1/5 : ℝ) ^ Complex.abs (z - 3) = (1/5 : ℝ) ^ (Complex.abs (z + 3) - 1) →
  ∃ (a : ℝ), a = 1/2 ∧ 
    Complex.abs (z + 3) - Complex.abs (z - 3) = 2 * a ∧
    z.re > 0 := by
  sorry

end hyperbola_equation_l323_32341


namespace triangle_operation_result_l323_32317

-- Define the triangle operation
def triangle (a b : ℝ) : ℝ := a^2 - 2*b

-- Theorem statement
theorem triangle_operation_result : triangle (-2) (triangle 3 4) = 2 := by
  sorry

end triangle_operation_result_l323_32317


namespace min_volume_pyramid_l323_32387

/-- A pyramid with a regular triangular base -/
structure Pyramid where
  base_side_length : ℝ
  apex_angle : ℝ

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := sorry

/-- The constraint on the apex angle -/
def apex_angle_constraint (p : Pyramid) : Prop :=
  p.apex_angle ≤ 2 * Real.arcsin (1/3)

theorem min_volume_pyramid :
  ∃ (p : Pyramid),
    p.base_side_length = 6 ∧
    apex_angle_constraint p ∧
    (∀ (q : Pyramid),
      q.base_side_length = 6 →
      apex_angle_constraint q →
      volume p ≤ volume q) ∧
    volume p = 5 * Real.sqrt 23 :=
sorry

end min_volume_pyramid_l323_32387


namespace cloth_selling_price_l323_32388

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Theorem stating that the total selling price for 85 meters of cloth with a profit of 10 Rs per meter and a cost price of 95 Rs per meter is 8925 Rs. -/
theorem cloth_selling_price :
  totalSellingPrice 85 10 95 = 8925 := by
  sorry

end cloth_selling_price_l323_32388


namespace height_difference_ruby_xavier_l323_32392

-- Define heights as natural numbers (in centimeters)
def janet_height : ℕ := 62
def charlene_height : ℕ := 2 * janet_height
def pablo_height : ℕ := charlene_height + 70
def ruby_height : ℕ := pablo_height - 2
def xavier_height : ℕ := charlene_height + 84
def paul_height : ℕ := ruby_height + 45

-- Theorem statement
theorem height_difference_ruby_xavier : 
  xavier_height - ruby_height = 7 := by sorry

end height_difference_ruby_xavier_l323_32392


namespace susan_age_in_five_years_l323_32367

/-- Represents the ages and time relationships in the problem -/
structure AgeRelationship where
  j : ℕ  -- James' current age
  n : ℕ  -- Janet's current age
  s : ℕ  -- Susan's current age
  x : ℕ  -- Years until James turns 37

/-- The conditions given in the problem -/
def problem_conditions (ar : AgeRelationship) : Prop :=
  (ar.j - 8 = 2 * (ar.n - 8)) ∧  -- 8 years ago, James was twice Janet's age
  (ar.j + ar.x = 37) ∧           -- In x years, James will turn 37
  (ar.s = ar.n - 3)              -- Susan was born when Janet turned 3

/-- The theorem to be proved -/
theorem susan_age_in_five_years (ar : AgeRelationship) 
  (h : problem_conditions ar) : 
  ar.s + 5 = ar.n + 2 := by
  sorry


end susan_age_in_five_years_l323_32367


namespace unique_quadratic_solution_l323_32340

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 24 * x + c = 0) →  -- exactly one solution
  (a + c = 31) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 9 ∧ c = 22) :=                 -- conclusion
by
  sorry

end unique_quadratic_solution_l323_32340


namespace expected_abs_difference_10_days_l323_32377

/-- Represents the wealth difference between two entities -/
def WealthDifference := ℤ

/-- Probability of each outcome -/
def p_cat_wins : ℝ := 0.25
def p_fox_wins : ℝ := 0.25
def p_both_lose : ℝ := 0.5

/-- Number of days -/
def num_days : ℕ := 10

/-- Expected value of absolute wealth difference after n days -/
def expected_abs_difference (n : ℕ) : ℝ := sorry

/-- Theorem: Expected absolute wealth difference after 10 days is 1 -/
theorem expected_abs_difference_10_days :
  expected_abs_difference num_days = 1 := by sorry

end expected_abs_difference_10_days_l323_32377


namespace small_square_area_l323_32373

theorem small_square_area (n : ℕ) : n > 0 → (
  let outer_square_area : ℝ := 1
  let small_square_area : ℝ := 1 / 1985
  let side_length : ℝ := 1 / n
  let diagonal_segment : ℝ := (n - 1) / n
  let small_square_side : ℝ := diagonal_segment / Real.sqrt 1985
  small_square_side * small_square_side = small_square_area
) ↔ n = 32 := by sorry

end small_square_area_l323_32373


namespace prob_different_topics_is_five_sixths_l323_32360

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from 6 available topics is 5/6 -/
theorem prob_different_topics_is_five_sixths :
  prob_different_topics = 5/6 := by sorry

end prob_different_topics_is_five_sixths_l323_32360


namespace final_i_is_16_l323_32309

def update_i (i : ℕ) : ℕ :=
  let new_i := 2 * i
  if new_i > 20 then new_i - 20 else new_i

def final_i : ℕ :=
  (List.range 5).foldl (fun acc _ => update_i acc) 2

theorem final_i_is_16 : final_i = 16 := by
  sorry

end final_i_is_16_l323_32309


namespace problem_solution_l323_32389

theorem problem_solution (f : ℝ → ℝ) (m a b c : ℝ) 
  (h1 : ∀ x, f x = |x - m|)
  (h2 : Set.Icc (-1) 5 = {x | f x ≤ 3})
  (h3 : a - 2*b + 2*c = m) : 
  m = 2 ∧ (∃ (min : ℝ), min = 4/9 ∧ a^2 + b^2 + c^2 ≥ min) := by
  sorry

end problem_solution_l323_32389


namespace base_number_proof_l323_32332

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^28) 
  (h2 : n = 27) : 
  x = 4 := by
  sorry

end base_number_proof_l323_32332


namespace johns_average_speed_l323_32310

/-- Calculates the overall average speed given two activities with their respective durations and speeds. -/
def overall_average_speed (duration1 duration2 : ℚ) (speed1 speed2 : ℚ) : ℚ :=
  (duration1 * speed1 + duration2 * speed2) / (duration1 + duration2)

/-- Proves that John's overall average speed is 11.6 mph given his scooter ride and jog. -/
theorem johns_average_speed :
  let scooter_duration : ℚ := 40 / 60  -- 40 minutes in hours
  let scooter_speed : ℚ := 20  -- 20 mph
  let jog_duration : ℚ := 60 / 60  -- 60 minutes in hours
  let jog_speed : ℚ := 6  -- 6 mph
  overall_average_speed scooter_duration jog_duration scooter_speed jog_speed = 58 / 5 := by
  sorry

#eval (58 : ℚ) / 5  -- Should evaluate to 11.6

end johns_average_speed_l323_32310


namespace smallest_base_perfect_square_l323_32351

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Represents 1111 in any base -/
def digits1111 : List Nat := [1, 1, 1, 1]

/-- The main theorem -/
theorem smallest_base_perfect_square :
  (∀ b : Nat, b > 0 → b < 7 → ¬isPerfectSquare (toBase10 digits1111 b)) ∧
  isPerfectSquare (toBase10 digits1111 7) := by
  sorry

#check smallest_base_perfect_square

end smallest_base_perfect_square_l323_32351


namespace rhombus_perimeter_l323_32336

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  4 * side = 40 := by sorry

end rhombus_perimeter_l323_32336


namespace octal_243_equals_decimal_163_l323_32390

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal % 100) / 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal number 243 is equal to 163 in decimal --/
theorem octal_243_equals_decimal_163 : octal_to_decimal 243 = 163 := by
  sorry

end octal_243_equals_decimal_163_l323_32390


namespace sum_of_coefficients_l323_32330

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
    (a * Real.sqrt 6 + b * Real.sqrt 8) / c) →
  (∀ (a' b' c' : ℕ+), 
    (∃ (k' : ℚ), k' * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (a' * Real.sqrt 6 + b' * Real.sqrt 8) / c') →
    c ≤ c') →
  a.val + b.val + c.val = 106 := by
sorry

end sum_of_coefficients_l323_32330


namespace subset_implies_lower_bound_l323_32394

theorem subset_implies_lower_bound (a : ℝ) : 
  (∀ x : ℝ, x < 5 → x < a) → a ≥ 5 := by sorry

end subset_implies_lower_bound_l323_32394


namespace a_worked_six_days_l323_32386

/-- Represents the number of days worked by person a -/
def days_a : ℕ := sorry

/-- Represents the daily wage of person a -/
def wage_a : ℕ := sorry

/-- Represents the daily wage of person b -/
def wage_b : ℕ := sorry

/-- Represents the daily wage of person c -/
def wage_c : ℕ := sorry

/-- The theorem stating that person a worked for 6 days given the conditions -/
theorem a_worked_six_days :
  wage_c = 105 ∧
  wage_a / wage_b = 3 / 4 ∧
  wage_b / wage_c = 4 / 5 ∧
  days_a * wage_a + 9 * wage_b + 4 * wage_c = 1554 →
  days_a = 6 := by sorry

end a_worked_six_days_l323_32386


namespace contrapositive_isosceles_angles_l323_32337

-- Define a triangle ABC
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the property of being an isosceles triangle
def isIsosceles (t : Triangle) : Prop := sorry

-- Define the property of having two equal interior angles
def hasTwoEqualAngles (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_angles (t : Triangle) :
  (¬ isIsosceles t → ¬ hasTwoEqualAngles t) ↔
  (hasTwoEqualAngles t → isIsosceles t) := by
  sorry

end contrapositive_isosceles_angles_l323_32337


namespace vectors_form_basis_l323_32358

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ (![e₁, e₂] : Fin 2 → ℝ × ℝ) :=
sorry

end vectors_form_basis_l323_32358


namespace max_value_of_expression_l323_32327

theorem max_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ 15 ∧ 
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 :=
by sorry

end max_value_of_expression_l323_32327


namespace only_rational_root_l323_32353

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

theorem only_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = -1 := by sorry

end only_rational_root_l323_32353


namespace convex_polyhedron_inequalities_l323_32399

/-- Represents a convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  faces_at_least_three_edges : 2 * edges ≥ 3 * faces

/-- The inequalities for convex polyhedrons. -/
theorem convex_polyhedron_inequalities (p : ConvexPolyhedron) :
  (3 * p.vertices ≥ 6 + p.faces) ∧ (3 * p.edges ≥ 6 + p.faces) := by
  sorry

end convex_polyhedron_inequalities_l323_32399


namespace louis_fabric_purchase_l323_32378

/-- The cost of velvet fabric per yard -/
def fabric_cost_per_yard : ℚ := 24

/-- The cost of the pattern -/
def pattern_cost : ℚ := 15

/-- The total cost of silver thread -/
def thread_cost : ℚ := 6

/-- The total amount spent -/
def total_spent : ℚ := 141

/-- The number of yards of fabric bought -/
def yards_bought : ℚ := (total_spent - pattern_cost - thread_cost) / fabric_cost_per_yard

theorem louis_fabric_purchase : yards_bought = 5 := by
  sorry

end louis_fabric_purchase_l323_32378


namespace opposite_of_three_l323_32397

theorem opposite_of_three (x : ℝ) : -x = 3 → x = -3 := by
  sorry

end opposite_of_three_l323_32397


namespace coordinates_product_l323_32338

/-- Given points A and M, where M is one-third of the way from A to B, 
    prove that the product of B's coordinates is -85 -/
theorem coordinates_product (A M : ℝ × ℝ) (h1 : A = (4, 2)) (h2 : M = (1, 7)) : 
  let B := (3 * M.1 - 2 * A.1, 3 * M.2 - 2 * A.2)
  B.1 * B.2 = -85 := by sorry

end coordinates_product_l323_32338


namespace price_restoration_l323_32364

theorem price_restoration (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let price_after_increases := original_price * (1 + 0.1) * (1 + 0.1) * (1 + 0.05)
  let price_after_decrease := price_after_increases * (1 - 0.22)
  price_after_decrease = original_price := by sorry

end price_restoration_l323_32364


namespace salt_mixture_proof_l323_32347

/-- Proves that adding 50 ounces of 60% salt solution to 50 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_mixture_proof :
  let initial_volume : ℝ := 50
  let initial_concentration : ℝ := 0.20
  let added_volume : ℝ := 50
  let added_concentration : ℝ := 0.60
  let final_concentration : ℝ := 0.40
  let final_volume : ℝ := initial_volume + added_volume
  let initial_salt : ℝ := initial_volume * initial_concentration
  let added_salt : ℝ := added_volume * added_concentration
  let final_salt : ℝ := initial_salt + added_salt
  (final_salt / final_volume) = final_concentration :=
by sorry


end salt_mixture_proof_l323_32347


namespace M_remainder_81_l323_32356

/-- The largest integer multiple of 9 with no repeated digits and all non-zero digits -/
def M : ℕ :=
  sorry

/-- M is a multiple of 9 -/
axiom M_multiple_of_9 : M % 9 = 0

/-- All digits of M are different -/
axiom M_distinct_digits : ∀ i j, i ≠ j → (M / 10^i % 10) ≠ (M / 10^j % 10)

/-- All digits of M are non-zero -/
axiom M_nonzero_digits : ∀ i, (M / 10^i % 10) ≠ 0

/-- M is the largest such number -/
axiom M_largest : ∀ n, n % 9 = 0 → (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)) → 
                  (∀ i, (n / 10^i % 10) ≠ 0) → n ≤ M

theorem M_remainder_81 : M % 100 = 81 :=
  sorry

end M_remainder_81_l323_32356


namespace exists_sequence_mod_23_l323_32382

/-- Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence with the desired property -/
theorem exists_sequence_mod_23 : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ 
  F 1 = 1 ∧ 
  (∀ n : ℕ, F (n + 2) = 3 * F (n + 1) - F n) ∧
  F 12 ≡ 0 [ZMOD 23] := by
  sorry


end exists_sequence_mod_23_l323_32382


namespace inverse_value_l323_32357

noncomputable section

variables (f g : ℝ → ℝ)

-- f⁻¹(g(x)) = x^4 - 1
axiom inverse_composition (x : ℝ) : f⁻¹ (g x) = x^4 - 1

-- g has an inverse
axiom g_has_inverse : Function.Bijective g

theorem inverse_value : g⁻¹ (f 10) = (11 : ℝ)^(1/4) := by sorry

end inverse_value_l323_32357


namespace percentage_of_green_caps_l323_32320

/-- Calculates the percentage of green bottle caps -/
theorem percentage_of_green_caps 
  (total_caps : ℕ) 
  (red_caps : ℕ) 
  (h1 : total_caps = 125) 
  (h2 : red_caps = 50) 
  (h3 : red_caps ≤ total_caps) : 
  (((total_caps - red_caps) : ℚ) / total_caps) * 100 = 60 := by
  sorry

#check percentage_of_green_caps

end percentage_of_green_caps_l323_32320


namespace cubic_polynomials_common_roots_l323_32301

theorem cubic_polynomials_common_roots (c d : ℝ) :
  c = -5 ∧ d = -6 →
  ∃ (r s : ℝ), r ≠ s ∧
    (r^3 + c*r^2 + 12*r + 7 = 0) ∧ 
    (r^3 + d*r^2 + 15*r + 9 = 0) ∧
    (s^3 + c*s^2 + 12*s + 7 = 0) ∧ 
    (s^3 + d*s^2 + 15*s + 9 = 0) :=
by sorry

end cubic_polynomials_common_roots_l323_32301


namespace min_value_quadratic_min_value_quadratic_achieved_l323_32375

theorem min_value_quadratic (x : ℝ) : 3 * x^2 - 18 * x + 2048 ≥ 2021 := by sorry

theorem min_value_quadratic_achieved : ∃ x : ℝ, 3 * x^2 - 18 * x + 2048 = 2021 := by sorry

end min_value_quadratic_min_value_quadratic_achieved_l323_32375


namespace paving_cost_l323_32314

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 6.5) (h2 : width = 2.75) (h3 : rate = 600) :
  length * width * rate = 10725 := by
  sorry

end paving_cost_l323_32314


namespace average_shirts_per_person_l323_32321

/-- Represents the average number of shirts made by each person per day -/
def S : ℕ := sorry

/-- The number of employees -/
def employees : ℕ := 20

/-- The number of hours in a shift -/
def shift_hours : ℕ := 8

/-- The hourly wage in dollars -/
def hourly_wage : ℕ := 12

/-- The bonus per shirt made in dollars -/
def bonus_per_shirt : ℕ := 5

/-- The selling price of a shirt in dollars -/
def shirt_price : ℕ := 35

/-- The daily nonemployee expenses in dollars -/
def nonemployee_expenses : ℕ := 1000

/-- The daily profit in dollars -/
def daily_profit : ℕ := 9080

theorem average_shirts_per_person (S : ℕ) :
  S * (shirt_price * employees - bonus_per_shirt * employees) = 
  daily_profit + nonemployee_expenses + employees * shift_hours * hourly_wage →
  S = 20 := by sorry

end average_shirts_per_person_l323_32321


namespace unique_power_of_two_plus_one_l323_32328

theorem unique_power_of_two_plus_one : 
  ∃! (n : ℕ), ∃ (A p : ℕ), p > 1 ∧ 2^n + 1 = A^p :=
by
  sorry

end unique_power_of_two_plus_one_l323_32328


namespace linear_equation_solve_l323_32354

theorem linear_equation_solve (x y : ℝ) :
  2 * x + y = 5 → y = -2 * x + 5 := by
  sorry

end linear_equation_solve_l323_32354


namespace ratio_of_sum_to_difference_l323_32385

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 2 := by
  sorry

end ratio_of_sum_to_difference_l323_32385


namespace population_increase_rate_l323_32319

/-- If a population increases by 220 persons in 55 minutes at a constant rate,
    then the rate of population increase is 15 seconds per person. -/
theorem population_increase_rate 
  (total_increase : ℕ) 
  (time_minutes : ℕ) 
  (h1 : total_increase = 220)
  (h2 : time_minutes = 55) :
  (time_minutes * 60) / total_increase = 15 := by
  sorry

end population_increase_rate_l323_32319
