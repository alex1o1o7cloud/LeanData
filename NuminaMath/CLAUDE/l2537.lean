import Mathlib

namespace vectors_not_collinear_l2537_253770

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 4, -2]
  let b : Fin 3 → ℝ := ![1, 1, -1]
  let c₁ : Fin 3 → ℝ := a + b
  let c₂ : Fin 3 → ℝ := 4 • a + 2 • b
  ¬ (∃ (k : ℝ), c₁ = k • c₂) :=
by sorry

end vectors_not_collinear_l2537_253770


namespace sin_cos_sixth_power_sum_l2537_253747

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.cos (2 * θ) = 1 / 5) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 7 / 25 := by
  sorry

end sin_cos_sixth_power_sum_l2537_253747


namespace f_2017_equals_one_l2537_253739

theorem f_2017_equals_one (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ θ, f (Real.cos θ) = Real.cos (2 * θ)) :
  f 2017 = 1 := by
  sorry

end f_2017_equals_one_l2537_253739


namespace unique_solution_linear_system_l2537_253749

theorem unique_solution_linear_system (x y z : ℝ) :
  (3*x + 2*y + 2*z = 13) ∧
  (2*x + 3*y + 2*z = 14) ∧
  (2*x + 2*y + 3*z = 15) ↔
  (x = 1 ∧ y = 2 ∧ z = 3) :=
by sorry

end unique_solution_linear_system_l2537_253749


namespace least_n_for_g_prime_product_l2537_253758

def g (n : ℕ) : ℕ := n.choose 3

def isArithmeticProgression (p₁ p₂ p₃ : ℕ) (d : ℕ) : Prop :=
  p₂ = p₁ + d ∧ p₃ = p₂ + d

theorem least_n_for_g_prime_product : 
  ∃ (p₁ p₂ p₃ : ℕ),
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧
    isArithmeticProgression p₁ p₂ p₃ 336 ∧
    g 2019 = p₁ * p₂ * p₃ ∧
    (∀ n < 2019, ¬∃ (q₁ q₂ q₃ : ℕ),
      q₁.Prime ∧ q₂.Prime ∧ q₃.Prime ∧
      q₁ < q₂ ∧ q₂ < q₃ ∧
      isArithmeticProgression q₁ q₂ q₃ 336 ∧
      g n = q₁ * q₂ * q₃) :=
by sorry

end least_n_for_g_prime_product_l2537_253758


namespace simplify_expression_l2537_253785

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end simplify_expression_l2537_253785


namespace newspaper_ad_cost_newspaper_ad_cost_proof_l2537_253713

/-- The total cost for three companies purchasing ads in a newspaper -/
theorem newspaper_ad_cost (num_companies : ℕ) (num_ad_spaces : ℕ) 
  (ad_length : ℝ) (ad_width : ℝ) (cost_per_sqft : ℝ) : ℝ :=
  let ad_area := ad_length * ad_width
  let cost_per_ad := ad_area * cost_per_sqft
  let cost_per_company := cost_per_ad * num_ad_spaces
  num_companies * cost_per_company

/-- Proof that the total cost for three companies purchasing 10 ad spaces each, 
    where each ad space is a 12-foot by 5-foot rectangle and costs $60 per square foot, 
    is $108,000 -/
theorem newspaper_ad_cost_proof :
  newspaper_ad_cost 3 10 12 5 60 = 108000 := by
  sorry

end newspaper_ad_cost_newspaper_ad_cost_proof_l2537_253713


namespace bees_after_six_days_l2537_253708

/-- Calculates the number of bees in the beehive after n days -/
def bees_in_hive (n : ℕ) : ℕ :=
  let a₁ : ℕ := 4  -- Initial term (1 original bee + 3 companions)
  let q : ℕ := 3   -- Common ratio (each bee brings 3 companions)
  a₁ * (q^n - 1) / (q - 1)

/-- Theorem stating that the number of bees after 6 days is 1456 -/
theorem bees_after_six_days :
  bees_in_hive 6 = 1456 := by
  sorry

end bees_after_six_days_l2537_253708


namespace syllogism_structure_l2537_253753

-- Define syllogism as a structure in deductive reasoning
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define deductive reasoning
def DeductiveReasoning : Type := Prop → Prop

-- Theorem stating that syllogism in deductive reasoning consists of major premise, minor premise, and conclusion
theorem syllogism_structure (dr : DeductiveReasoning) :
  ∃ (s : Syllogism), dr s.major_premise ∧ dr s.minor_premise ∧ dr s.conclusion :=
sorry

end syllogism_structure_l2537_253753


namespace units_digit_of_product_l2537_253789

theorem units_digit_of_product (a b c : ℕ) : 
  (2^1501 * 5^1602 * 11^1703) % 10 = 0 := by sorry

end units_digit_of_product_l2537_253789


namespace line_plane_perpendicular_l2537_253730

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpendicular m α →
  parallel m n →
  subset n β →
  plane_perpendicular α β :=
sorry

end line_plane_perpendicular_l2537_253730


namespace jerry_book_pages_l2537_253792

/-- Calculates the total number of pages in a book given the number of pages read on Saturday, Sunday, and the number of pages remaining. -/
def total_pages (pages_saturday : ℕ) (pages_sunday : ℕ) (pages_remaining : ℕ) : ℕ :=
  pages_saturday + pages_sunday + pages_remaining

/-- Theorem stating that the total number of pages in Jerry's book is 93. -/
theorem jerry_book_pages : total_pages 30 20 43 = 93 := by
  sorry

end jerry_book_pages_l2537_253792


namespace special_triangle_sides_l2537_253744

/-- A triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Vertex B of the triangle -/
  B : ℝ × ℝ
  /-- Equation of the altitude on side AB: ax + by + c = 0 -/
  altitude : ℝ × ℝ × ℝ
  /-- Equation of the angle bisector of angle A: dx + ey + f = 0 -/
  angle_bisector : ℝ × ℝ × ℝ

/-- Theorem about the equations of sides in a special triangle -/
theorem special_triangle_sides 
  (t : SpecialTriangle) 
  (h1 : t.B = (-2, 0))
  (h2 : t.altitude = (1, 3, -26))
  (h3 : t.angle_bisector = (1, 1, -2)) :
  ∃ (AB AC : ℝ × ℝ × ℝ),
    AB = (3, -1, 6) ∧ 
    AC = (1, -3, 10) := by
  sorry

end special_triangle_sides_l2537_253744


namespace hyperbola_eccentricity_l2537_253727

/-- The eccentricity of the hyperbola 3x^2 - y^2 = 3 is 2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 2 ∧ 
  ∀ (x y : ℝ), 3 * x^2 - y^2 = 3 → 
  e = (Real.sqrt ((3 * x^2 + y^2) / 3)) / (Real.sqrt (3 * x^2 / 3)) := by
  sorry

end hyperbola_eccentricity_l2537_253727


namespace marble_probability_difference_l2537_253720

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1001

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1001

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_same : ℚ := (red_marbles.choose 2 + black_marbles.choose 2 : ℚ) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def P_diff : ℚ := (red_marbles * black_marbles : ℚ) / total_marbles.choose 2

/-- The theorem stating that the absolute difference between P_same and P_diff is 1/2001 -/
theorem marble_probability_difference : |P_same - P_diff| = 1 / 2001 := by
  sorry

end marble_probability_difference_l2537_253720


namespace grouping_theorem_l2537_253799

/- Define the number of men and women -/
def num_men : ℕ := 4
def num_women : ℕ := 5

/- Define the size of each group -/
def group_size : ℕ := 3

/- Define the total number of groups -/
def num_groups : ℕ := 3

/- Define the function to calculate the number of ways to group people -/
def group_ways : ℕ :=
  let first_group_men := 1
  let first_group_women := 2
  let second_group_men := 2
  let second_group_women := 1
  (num_men.choose first_group_men * num_women.choose first_group_women) *
  ((num_men - first_group_men).choose second_group_men * (num_women - first_group_women).choose second_group_women)

/- Theorem statement -/
theorem grouping_theorem :
  group_ways = 360 :=
sorry

end grouping_theorem_l2537_253799


namespace expression_evaluation_l2537_253745

theorem expression_evaluation (x y : ℝ) 
  (h : (x + 2)^2 + |y - 2/3| = 0) : 
  1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2) = 6 + 4/9 := by
  sorry

end expression_evaluation_l2537_253745


namespace complex_equation_solution_l2537_253786

theorem complex_equation_solution (i : ℂ) (a : ℝ) :
  i * i = -1 →
  (1 + i) * (a - i) = 3 + i →
  a = 2 := by
sorry

end complex_equation_solution_l2537_253786


namespace circle_sum_formula_l2537_253701

/-- The sum of numbers on a circle after n divisions -/
def circle_sum (n : ℕ) : ℝ :=
  2 * 3^n

/-- Theorem stating the sum of numbers on the circle after n divisions -/
theorem circle_sum_formula (n : ℕ) : circle_sum n = 2 * 3^n := by
  sorry

end circle_sum_formula_l2537_253701


namespace product_of_squared_terms_l2537_253737

theorem product_of_squared_terms (x : ℝ) : 3 * x^2 * (2 * x^2) = 6 * x^4 := by
  sorry

end product_of_squared_terms_l2537_253737


namespace largest_r_is_two_l2537_253788

/-- A sequence of positive integers satisfying the given inequality -/
def ValidSequence (a : ℕ → ℕ) (r : ℝ) : Prop :=
  ∀ n : ℕ, (a n ≤ a (n + 2)) ∧ ((a (n + 2))^2 ≤ (a n)^2 + r * a (n + 1))

/-- The sequence eventually stabilizes -/
def EventuallyStable (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n

/-- The main theorem stating that 2 is the largest real number satisfying the condition -/
theorem largest_r_is_two :
  (∀ a : ℕ → ℕ, ValidSequence a 2 → EventuallyStable a) ∧
  (∀ r > 2, ∃ a : ℕ → ℕ, ValidSequence a r ∧ ¬EventuallyStable a) := by
  sorry

end largest_r_is_two_l2537_253788


namespace ninth_term_is_nine_l2537_253777

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first six terms is 21 -/
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  /-- The seventh term is 7 -/
  seventh_term : a + 6*d = 7

/-- The ninth term of the arithmetic sequence is 9 -/
theorem ninth_term_is_nine (seq : ArithmeticSequence) : seq.a + 8*seq.d = 9 := by
  sorry

end ninth_term_is_nine_l2537_253777


namespace trig_expression_simplification_l2537_253721

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π - α) * Real.sin (3 * π - α) + Real.sin (-α - π) * Real.sin (α - 2 * π)) /
  (Real.sin (4 * π - α) * Real.sin (5 * π + α)) = -2 := by
  sorry

end trig_expression_simplification_l2537_253721


namespace no_common_terms_except_one_l2537_253791

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one :
  ∀ m n : ℕ, m > 0 ∧ n > 0 → x m ≠ y n :=
by sorry

end no_common_terms_except_one_l2537_253791


namespace expression_evaluation_l2537_253796

theorem expression_evaluation :
  (4^4 - 4*(4-2)^4)^(4+1) = 14889702426 := by sorry

end expression_evaluation_l2537_253796


namespace binomial_coefficient_ratio_l2537_253722

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 9 := by
sorry

end binomial_coefficient_ratio_l2537_253722


namespace catches_ratio_l2537_253704

theorem catches_ratio (joe_catches tammy_catches derek_catches : ℕ) : 
  joe_catches = 23 →
  tammy_catches = 30 →
  tammy_catches = derek_catches / 3 + 16 →
  derek_catches / joe_catches = 42 / 23 := by
  sorry

end catches_ratio_l2537_253704


namespace factorization_equality_l2537_253729

theorem factorization_equality (a b : ℝ) : 3*a - 9*a*b = 3*a*(1 - 3*b) := by
  sorry

end factorization_equality_l2537_253729


namespace negation_equivalence_l2537_253793

theorem negation_equivalence :
  (¬ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) := by
  sorry

end negation_equivalence_l2537_253793


namespace fifth_term_value_l2537_253707

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_value
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a 2)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end fifth_term_value_l2537_253707


namespace remaining_capacity_theorem_l2537_253742

/-- Represents the meal capacity and consumption for a trekking group --/
structure MealCapacity where
  adult_capacity : ℕ
  child_capacity : ℕ
  adults_eaten : ℕ

/-- Calculates the number of children that can be catered with the remaining food --/
def remaining_child_capacity (m : MealCapacity) : ℕ :=
  sorry

/-- Theorem stating that given the specific meal capacity and consumption, 
    the remaining food can cater to 45 children --/
theorem remaining_capacity_theorem (m : MealCapacity) 
  (h1 : m.adult_capacity = 70)
  (h2 : m.child_capacity = 90)
  (h3 : m.adults_eaten = 35) :
  remaining_child_capacity m = 45 := by
  sorry

end remaining_capacity_theorem_l2537_253742


namespace necessary_condition_for_existence_l2537_253736

theorem necessary_condition_for_existence (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, x^2 - a > 0) → a ≤ 4 := by
  sorry

end necessary_condition_for_existence_l2537_253736


namespace alpine_ridge_length_l2537_253764

/-- Represents the Alpine Ridge Trail hike --/
structure AlpineRidgeTrail where
  /-- Distance hiked on each of the five days --/
  day : Fin 5 → ℝ
  /-- First three days total 30 miles --/
  first_three_days : day 0 + day 1 + day 2 = 30
  /-- Second and fourth days average 15 miles --/
  second_fourth_avg : (day 1 + day 3) / 2 = 15
  /-- Last two days total 28 miles --/
  last_two_days : day 3 + day 4 = 28
  /-- First and fourth days total 34 miles --/
  first_fourth_days : day 0 + day 3 = 34

/-- The total length of the Alpine Ridge Trail is 58 miles --/
theorem alpine_ridge_length (trail : AlpineRidgeTrail) : 
  trail.day 0 + trail.day 1 + trail.day 2 + trail.day 3 + trail.day 4 = 58 := by
  sorry


end alpine_ridge_length_l2537_253764


namespace largest_divisor_of_n_squared_divisible_by_18_l2537_253728

theorem largest_divisor_of_n_squared_divisible_by_18 (n : ℕ+) (h : 18 ∣ n^2) :
  6 = Nat.gcd 6 n ∧ ∀ m : ℕ, m ∣ n → m ≤ 6 :=
sorry

end largest_divisor_of_n_squared_divisible_by_18_l2537_253728


namespace log_inequalities_l2537_253709

-- Define the logarithm functions
noncomputable def log₃ (x : ℝ) := Real.log x / Real.log 3
noncomputable def log₁₃ (x : ℝ) := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_inequalities :
  (∀ x y, x < y → log₃ x < log₃ y) →  -- log₃ is increasing
  (∀ x y, x < y → log₁₃ x > log₁₃ y) →  -- log₁₃ is decreasing
  (1/5)^0 = 1 →
  log₃ 4 > (1/5)^0 ∧ (1/5)^0 > log₁₃ 10 :=
by sorry

end log_inequalities_l2537_253709


namespace min_value_of_f_l2537_253769

/-- A cubic function with a constant term. -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + m

/-- The theorem stating the minimum value of f on [0, 2] given its maximum value. -/
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f m y ≤ f m x) ∧
  (∀ x ∈ Set.Icc 0 2, f m x ≤ 3) →
  ∃ x ∈ Set.Icc 0 2, f m x = -1 ∧ ∀ y ∈ Set.Icc 0 2, -1 ≤ f m y :=
by sorry

end min_value_of_f_l2537_253769


namespace fraction_to_decimal_equivalence_l2537_253756

theorem fraction_to_decimal_equivalence : (1 : ℚ) / 4 = (25 : ℚ) / 100 := by sorry

end fraction_to_decimal_equivalence_l2537_253756


namespace trigonometric_product_equals_one_l2537_253797

theorem trigonometric_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end trigonometric_product_equals_one_l2537_253797


namespace average_income_of_M_and_N_l2537_253772

/-- Given the monthly incomes of three individuals M, N, and O, prove that the average income of M and N is 5050. -/
theorem average_income_of_M_and_N (M N O : ℕ) : 
  (N + O) / 2 = 6250 →
  (M + O) / 2 = 5200 →
  M = 4000 →
  (M + N) / 2 = 5050 := by
sorry

end average_income_of_M_and_N_l2537_253772


namespace product_pure_imaginary_implies_a_equals_six_l2537_253768

theorem product_pure_imaginary_implies_a_equals_six :
  ∀ (a : ℝ), 
  (∃ (b : ℝ), (a + 2*I) * (1 + 3*I) = b*I ∧ b ≠ 0) →
  a = 6 := by
sorry

end product_pure_imaginary_implies_a_equals_six_l2537_253768


namespace negation_equivalence_l2537_253762

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x ∈ Set.Icc 1 2 → 2 * x^2 - 3 ≥ 0) ↔
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - 3 < 0) :=
by sorry

end negation_equivalence_l2537_253762


namespace inequality_solution_set_l2537_253748

/-- The solution set of the inequality -x^2 - 2x + 3 > 0 is the open interval (-3, 1) -/
theorem inequality_solution_set : 
  {x : ℝ | -x^2 - 2*x + 3 > 0} = Set.Ioo (-3) 1 := by sorry

end inequality_solution_set_l2537_253748


namespace factorization_3m_squared_minus_12_l2537_253754

theorem factorization_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := by
  sorry

end factorization_3m_squared_minus_12_l2537_253754


namespace modular_inverse_13_mod_1728_l2537_253715

theorem modular_inverse_13_mod_1728 : ∃ x : ℕ, x < 1728 ∧ (13 * x) % 1728 = 1 :=
by
  use 133
  sorry

end modular_inverse_13_mod_1728_l2537_253715


namespace triangle_exists_l2537_253724

/-- Triangle inequality theorem for a triangle with sides a, b, and c -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A theorem stating that a triangle can be formed with side lengths 6, 8, and 13 -/
theorem triangle_exists : triangle_inequality 6 8 13 := by
  sorry

end triangle_exists_l2537_253724


namespace intersection_of_sets_l2537_253716

theorem intersection_of_sets : ∀ (A B : Set ℕ),
  A = {1, 2, 3, 4, 5} →
  B = {3, 5} →
  A ∩ B = {3, 5} := by
  sorry

end intersection_of_sets_l2537_253716


namespace sue_buttons_l2537_253711

theorem sue_buttons (mari kendra sue : ℕ) : 
  mari = 8 →
  kendra = 5 * mari + 4 →
  sue = kendra / 2 →
  sue = 22 := by
sorry

end sue_buttons_l2537_253711


namespace expression_value_l2537_253732

theorem expression_value (α : Real) (h : Real.tan α = -3/4) :
  (3 * (Real.sin (α/2))^2 + 2 * Real.sin (α/2) * Real.cos (α/2) + (Real.cos (α/2))^2 - 2) /
  (Real.sin (π/2 + α) * Real.tan (-3*π + α) + Real.cos (6*π - α)) = -7 :=
by sorry

end expression_value_l2537_253732


namespace root_equation_solution_l2537_253731

theorem root_equation_solution (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (∀ N : ℝ, N ≠ 1 → N^(1/a + 1/(a*b) + 1/(a*b*c)) = N^(25/36)) → b = 3 := by
  sorry

end root_equation_solution_l2537_253731


namespace percentage_difference_l2537_253717

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) :
  (x - y) / x * 100 = 200 / 3 := by
sorry

end percentage_difference_l2537_253717


namespace square_difference_divided_by_nine_l2537_253743

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by sorry

end square_difference_divided_by_nine_l2537_253743


namespace expression_equality_l2537_253705

theorem expression_equality (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a^2 - b^2) / (a * b) - (a * b - b * c) / (a * b - a * c) = (c * a - (c - 1) * b) / b :=
by sorry

end expression_equality_l2537_253705


namespace bookstore_shipment_problem_l2537_253733

theorem bookstore_shipment_problem :
  ∀ (B : ℕ), 
    (70 : ℚ) / 100 * B = 45 →
    B = 64 :=
by
  sorry

end bookstore_shipment_problem_l2537_253733


namespace f_min_value_a_range_characterization_l2537_253755

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x - 2) - x + 5

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 := by
  sorry

-- Define the set of valid values for a
def valid_a_set : Set ℝ := {a | a ≤ -5 ∨ a ≥ 1}

-- Theorem for the range of a
theorem a_range_characterization (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (x + 2) ≥ 3) ↔ a ∈ valid_a_set := by
  sorry

end f_min_value_a_range_characterization_l2537_253755


namespace jelly_bean_problem_l2537_253750

theorem jelly_bean_problem (b c : ℕ) : 
  b = 3 * c →                  -- Initial ratio
  b - 5 = 5 * (c - 15) →       -- Ratio after eating jelly beans
  b = 105 :=                   -- Conclusion
by sorry

end jelly_bean_problem_l2537_253750


namespace complex_expansion_l2537_253752

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_expansion : (1 - i) * (1 + 2*i)^2 = 1 + 7*i := by
  sorry

end complex_expansion_l2537_253752


namespace fraction_to_decimal_l2537_253723

theorem fraction_to_decimal : (11 : ℚ) / 125 = (88 : ℚ) / 1000 := by sorry

end fraction_to_decimal_l2537_253723


namespace roots_square_sum_l2537_253759

theorem roots_square_sum (a b : ℝ) : 
  (∀ x, x^2 - 2*x - 1 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 6 := by
sorry

end roots_square_sum_l2537_253759


namespace composition_of_functions_l2537_253700

theorem composition_of_functions (f g : ℝ → ℝ) :
  (∀ x, f x = 5 - 2 * x) →
  (∀ x, g x = x^2 + x + 1) →
  f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := by
  sorry

end composition_of_functions_l2537_253700


namespace odd_function_max_value_l2537_253767

-- Define the function f on (-∞, 0)
def f (x : ℝ) : ℝ := x * (1 + x)

-- State the theorem
theorem odd_function_max_value :
  (∀ x < 0, f x = x * (1 + x)) →  -- f is defined as x(1+x) on (-∞, 0)
  (∀ x : ℝ, f (-x) = -f x) →      -- f is an odd function
  (∃ M : ℝ, M = 1/4 ∧ ∀ x > 0, f x ≤ M) -- Maximum value on (0, +∞) is 1/4
  := by sorry

end odd_function_max_value_l2537_253767


namespace investment_scientific_notation_l2537_253766

-- Define the value in billion yuan
def investment : ℝ := 845

-- Define the scientific notation representation
def scientific_notation : ℝ := 8.45 * (10 ^ 3)

-- Theorem statement
theorem investment_scientific_notation : investment = scientific_notation := by
  sorry

end investment_scientific_notation_l2537_253766


namespace inequality_and_equality_condition_l2537_253740

theorem inequality_and_equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2*b + b^2*c + c^2*d + d^2*a ∧
  (a^3 + b^3 + c^3 + d^3 = a^2*b + b^2*c + c^2*d + d^2*a ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end inequality_and_equality_condition_l2537_253740


namespace sin_n_equals_cos_682_l2537_253760

theorem sin_n_equals_cos_682 (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (682 * π / 180) → n = 128 := by
  sorry

end sin_n_equals_cos_682_l2537_253760


namespace sqrt_difference_sum_abs_l2537_253773

theorem sqrt_difference_sum_abs : 1 + (Real.sqrt 2 - Real.sqrt 3) + |Real.sqrt 2 - Real.sqrt 3| = 1 := by
  sorry

end sqrt_difference_sum_abs_l2537_253773


namespace circle_coverage_l2537_253794

-- Define a circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if one circle can cover another
def canCover (c1 c2 : Circle) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 ≤ c2.radius^2 →
    (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 ≤ c1.radius^2

-- Theorem statement
theorem circle_coverage (M1 M2 : Circle) (h : M2.radius > M1.radius) :
  canCover M2 M1 ∧ ¬(canCover M1 M2) := by
  sorry

end circle_coverage_l2537_253794


namespace no_natural_solution_l2537_253782

theorem no_natural_solution : ¬∃ (m n : ℕ), m * n * (m + n) = 2020 := by
  sorry

end no_natural_solution_l2537_253782


namespace min_value_of_f_l2537_253779

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 2*a

theorem min_value_of_f (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hf : f a 2 = 1/3) :
  ∃ (m : ℝ), IsMinOn (f a) (Set.Icc 0 3) m ∧ m = -1/3 := by sorry

end min_value_of_f_l2537_253779


namespace fraction_equality_l2537_253775

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x - 2*y) / (2*x + 3*y) = 1) : 
  (2*x - 5*y) / (5*x + 2*y) = -5/31 := by
  sorry

end fraction_equality_l2537_253775


namespace sequence_properties_l2537_253774

def is_root (a : ℝ) : Prop := a^2 - 3*a - 5 = 0

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m

theorem sequence_properties (a : ℕ → ℝ) :
  (is_root (a 3) ∧ is_root (a 10) ∧ arithmetic_sequence a → a 5 + a 8 = 3) ∧
  (is_root (a 3) ∧ is_root (a 10) ∧ geometric_sequence a → a 6 * a 7 = -5) :=
sorry

end sequence_properties_l2537_253774


namespace imaginary_unit_multiplication_l2537_253765

theorem imaginary_unit_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end imaginary_unit_multiplication_l2537_253765


namespace inscribed_sphere_radius_l2537_253790

/-- Given a regular tetrahedron formed by the centers of four spheres in a
    triangular pyramid of ten equal spheres, prove that the radius of the
    sphere inscribed at the center of the tetrahedron is √6 - 1, given that
    the radius of the circumscribed sphere of the tetrahedron is 5√2 + 5. -/
theorem inscribed_sphere_radius (R : ℝ) (r : ℝ) :
  R = 5 * Real.sqrt 2 + 5 →
  r = Real.sqrt 6 - 1 :=
by sorry

end inscribed_sphere_radius_l2537_253790


namespace arithmetic_sequence_value_l2537_253702

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem arithmetic_sequence_value :
  ∀ a : ℝ, is_arithmetic_sequence 2 a 10 → a = 6 := by
  sorry

end arithmetic_sequence_value_l2537_253702


namespace sum_of_squares_first_15_sum_of_squares_second_15_l2537_253735

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_first_15 :
  sum_of_squares 15 = 1200 :=
by sorry

-- The condition about the second 15 integers is not directly used in the proof,
-- but we include it as a hypothesis to match the original problem
theorem sum_of_squares_second_15 :
  sum_of_squares 30 - sum_of_squares 15 = 8175 :=
by sorry

end sum_of_squares_first_15_sum_of_squares_second_15_l2537_253735


namespace min_xy_point_l2537_253710

theorem min_xy_point (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1/x + 1/(2*y) + 3/(2*x*y) = 1) :
  x * y ≥ 9/2 ∧ (x * y = 9/2 ↔ x = 3 ∧ y = 3/2) := by
  sorry

end min_xy_point_l2537_253710


namespace yolandas_walking_rate_l2537_253712

/-- Proves that Yolanda's walking rate is 5 miles per hour given the problem conditions -/
theorem yolandas_walking_rate
  (total_distance : ℝ)
  (bobs_rate : ℝ)
  (time_difference : ℝ)
  (bobs_distance : ℝ)
  (h1 : total_distance = 60)
  (h2 : bobs_rate = 6)
  (h3 : time_difference = 1)
  (h4 : bobs_distance = 30) :
  (total_distance - bobs_distance) / (bobs_distance / bobs_rate + time_difference) = 5 :=
by sorry

end yolandas_walking_rate_l2537_253712


namespace average_weight_increase_l2537_253718

/-- Proves that the increase in average weight when including a teacher is 400 grams -/
theorem average_weight_increase (num_students : Nat) (avg_weight_students : ℝ) (teacher_weight : ℝ) :
  num_students = 24 →
  avg_weight_students = 35 →
  teacher_weight = 45 →
  ((num_students + 1) * ((num_students * avg_weight_students + teacher_weight) / (num_students + 1)) -
   (num_students * avg_weight_students)) * 1000 = 400 := by
  sorry

end average_weight_increase_l2537_253718


namespace cindy_marbles_problem_l2537_253781

theorem cindy_marbles_problem (initial_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 1000 →
  num_friends = 6 →
  marbles_per_friend = 120 →
  7 * (initial_marbles - num_friends * marbles_per_friend) = 1960 :=
by
  sorry

end cindy_marbles_problem_l2537_253781


namespace bus_left_seats_count_l2537_253776

/-- Represents the seating arrangement in a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  seat_capacity : ℕ
  total_capacity : ℕ

/-- The bus seating arrangement satisfies the given conditions -/
def valid_bus_seating (bus : BusSeating) : Prop :=
  bus.right_seats = bus.left_seats - 3 ∧
  bus.back_seat_capacity = 7 ∧
  bus.seat_capacity = 3 ∧
  bus.total_capacity = 88 ∧
  bus.total_capacity = bus.seat_capacity * (bus.left_seats + bus.right_seats) + bus.back_seat_capacity

/-- The number of seats on the left side of the bus is 15 -/
theorem bus_left_seats_count (bus : BusSeating) (h : valid_bus_seating bus) : bus.left_seats = 15 := by
  sorry


end bus_left_seats_count_l2537_253776


namespace abc_bad_theorem_l2537_253761

def is_valid_quadruple (A B C D : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ D ≠ 0 ∧ C ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (100 * A + 10 * B + C) * D = (100 * B + 10 * A + D) * C

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2,1,7,4), (1,2,4,7), (8,1,9,2), (1,8,2,9), (7,2,8,3), (2,7,3,8), (6,3,7,4), (3,6,4,7)}

theorem abc_bad_theorem :
  {q : ℕ × ℕ × ℕ × ℕ | is_valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2} = solution_set :=
sorry

end abc_bad_theorem_l2537_253761


namespace f_eight_minus_f_four_l2537_253795

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_eight_minus_f_four (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 5)
  (h1 : f 1 = 1)
  (h2 : f 2 = 3) :
  f 8 - f 4 = -2 := by
  sorry

end f_eight_minus_f_four_l2537_253795


namespace x_equals_one_sufficient_not_necessary_l2537_253703

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, x^2 + x - 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x^2 + x - 2 = 0) :=
by sorry

end x_equals_one_sufficient_not_necessary_l2537_253703


namespace triangle_area_l2537_253706

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a^2 + b^2 - c^2 = 6√3 - 2ab and C = 60°, then the area of triangle ABC is 3/2. -/
theorem triangle_area (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 6 * Real.sqrt 3 - 2*a*b) 
  (h2 : Real.cos (Real.pi / 3) = 1/2) : 
  (1/2) * a * b * Real.sin (Real.pi / 3) = 3/2 := by
  sorry

end triangle_area_l2537_253706


namespace max_value_product_l2537_253714

theorem max_value_product (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) ≤ 27 :=
by sorry

end max_value_product_l2537_253714


namespace quadratic_equation_solution_l2537_253741

theorem quadratic_equation_solution (x : ℝ) : 
  -x^2 - (-18 + 12) * x - 8 = -(x - 2) * (x - 4) := by
  sorry

end quadratic_equation_solution_l2537_253741


namespace no_solution_range_l2537_253719

theorem no_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) → a ∈ Set.Iic 8 := by
  sorry

end no_solution_range_l2537_253719


namespace right_triangle_345_ratio_l2537_253787

theorem right_triangle_345_ratio (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) : a^2 + b^2 = c^2 := by
  sorry

end right_triangle_345_ratio_l2537_253787


namespace limit_f_at_infinity_l2537_253726

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((a^x - 1) / (x * (a - 1)))^(1/x)

theorem limit_f_at_infinity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ ε > 0, ∃ N, ∀ x ≥ N, |f a x - (if a > 1 then a else 1)| < ε) :=
sorry

end limit_f_at_infinity_l2537_253726


namespace bullet_speed_difference_wild_bill_scenario_l2537_253780

/-- The speed difference of a bullet fired from a moving horse -/
theorem bullet_speed_difference (v_horse : ℝ) (v_bullet : ℝ) :
  v_horse > 0 → v_bullet > v_horse →
  (v_bullet + v_horse) - (v_bullet - v_horse) = 2 * v_horse := by
  sorry

/-- Wild Bill's scenario -/
theorem wild_bill_scenario :
  let v_horse : ℝ := 20
  let v_bullet : ℝ := 400
  (v_bullet + v_horse) - (v_bullet - v_horse) = 40 := by
  sorry

end bullet_speed_difference_wild_bill_scenario_l2537_253780


namespace right_triangle_third_side_l2537_253734

theorem right_triangle_third_side (x y : ℝ) :
  (x > 0 ∧ y > 0) →
  (|x^2 - 4| + Real.sqrt (y^2 - 5*y + 6) = 0) →
  ∃ z : ℝ, (z = 2 * Real.sqrt 2 ∨ z = Real.sqrt 13 ∨ z = Real.sqrt 5) ∧
           (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) :=
by sorry

end right_triangle_third_side_l2537_253734


namespace factor_expression_l2537_253784

theorem factor_expression (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end factor_expression_l2537_253784


namespace simple_interest_rate_l2537_253778

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℚ) (h1 : principal = 23) 
  (h2 : time = 3) (h3 : interest = 3.45) : 
  interest / (principal * time) = 0.05 := by
  sorry

end simple_interest_rate_l2537_253778


namespace complex_number_in_first_quadrant_l2537_253771

/-- The complex number z = (3+i)/(1-i) corresponds to a point in the first quadrant -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (0 < z.re) ∧ (0 < z.im) := by sorry

end complex_number_in_first_quadrant_l2537_253771


namespace function_equation_solution_l2537_253751

open Real

theorem function_equation_solution (f : ℝ → ℝ) (h : ∀ x ∈ Set.Ioo (-1) 1, 2 * f x - f (-x) = log (x + 1)) :
  ∀ x ∈ Set.Ioo (-1) 1, f x = (2/3) * log (x + 1) + (1/3) * log (1 - x) := by
  sorry

end function_equation_solution_l2537_253751


namespace rectangle_area_l2537_253757

theorem rectangle_area (side : ℝ) (h1 : side > 0) :
  let perimeter := 8 * side
  let area := 4 * side^2
  perimeter = 160 → area = 1600 := by
sorry

end rectangle_area_l2537_253757


namespace tan_ratio_from_sin_sum_diff_l2537_253798

theorem tan_ratio_from_sin_sum_diff (x y : ℝ) 
  (h1 : Real.sin (x + y) = 5/8) 
  (h2 : Real.sin (x - y) = 1/4) : 
  Real.tan x / Real.tan y = 2 := by
  sorry

end tan_ratio_from_sin_sum_diff_l2537_253798


namespace water_price_problem_l2537_253738

/-- The residential water price problem -/
theorem water_price_problem (last_year_price : ℝ) 
  (h1 : last_year_price > 0)
  (h2 : 30 / (1.2 * last_year_price) - 15 / last_year_price = 5) : 
  1.2 * last_year_price = 6 := by
  sorry

#check water_price_problem

end water_price_problem_l2537_253738


namespace certain_number_value_value_is_232_l2537_253725

theorem certain_number_value : ℤ → ℤ → Prop :=
  fun n value => 5 * n - 28 = value

theorem value_is_232 (n : ℤ) (value : ℤ) 
  (h1 : n = 52) 
  (h2 : certain_number_value n value) : 
  value = 232 := by
sorry

end certain_number_value_value_is_232_l2537_253725


namespace consecutive_squareful_numbers_l2537_253783

/-- A natural number is squareful if it has a square divisor greater than 1 -/
def IsSquareful (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

/-- For any natural number k, there exist k consecutive squareful numbers -/
theorem consecutive_squareful_numbers :
  ∀ k : ℕ, ∃ n : ℕ, ∀ i : ℕ, i < k → IsSquareful (n + i) :=
sorry

end consecutive_squareful_numbers_l2537_253783


namespace bat_wings_area_is_3_25_l2537_253763

/-- Rectangle DEFA with given dimensions and points -/
structure Rectangle where
  width : ℝ
  height : ℝ
  dc : ℝ
  cb : ℝ
  ba : ℝ

/-- Calculate the area of the "bat wings" in the given rectangle -/
def batWingsArea (rect : Rectangle) : ℝ := sorry

/-- Theorem stating that the area of the "bat wings" is 3.25 -/
theorem bat_wings_area_is_3_25 (rect : Rectangle) 
  (h1 : rect.width = 5)
  (h2 : rect.height = 3)
  (h3 : rect.dc = 2)
  (h4 : rect.cb = 1.5)
  (h5 : rect.ba = 1.5) :
  batWingsArea rect = 3.25 := by sorry

end bat_wings_area_is_3_25_l2537_253763


namespace equation_conditions_l2537_253746

theorem equation_conditions (a b c d : ℝ) :
  (2*a + 3*b) / (b + 2*c) = (3*c + 2*d) / (d + 2*a) →
  (2*a = 3*c) ∨ (2*a + 3*b + d + 2*c = 0) :=
by sorry

end equation_conditions_l2537_253746
