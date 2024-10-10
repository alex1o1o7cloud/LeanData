import Mathlib

namespace expansion_properties_l207_20700

-- Define the binomial expansion function
def binomial_expansion (x : ℝ) (n : ℕ) : ℝ → ℝ := sorry

-- Define the coefficient function for the expansion
def coefficient (x : ℝ) (n r : ℕ) : ℝ := sorry

-- Define the general term of the expansion
def general_term (x : ℝ) (n r : ℕ) : ℝ := sorry

theorem expansion_properties :
  let f := binomial_expansion x 8
  -- The first three coefficients are in arithmetic sequence
  ∃ (a d : ℝ), coefficient x 8 0 = a ∧ 
               coefficient x 8 1 = a + d ∧ 
               coefficient x 8 2 = a + 2*d →
  -- 1. The term containing x to the first power
  (∃ (r : ℕ), general_term x 8 r = (35/8) * x) ∧
  -- 2. The rational terms involving x
  (∀ (r : ℕ), r ≤ 8 → 
    (∃ (k : ℤ), general_term x 8 r = x^k) ↔ 
    (general_term x 8 r = x^4 ∨ 
     general_term x 8 r = (35/8) * x ∨ 
     general_term x 8 r = 1/(256 * x^2))) ∧
  -- 3. The terms with the largest coefficient
  (∀ (r : ℕ), r ≤ 8 → 
    coefficient x 8 r ≤ 7 ∧
    (coefficient x 8 r = 7 ↔ (r = 2 ∨ r = 3))) :=
sorry

end expansion_properties_l207_20700


namespace geometric_progression_quadratic_vertex_l207_20723

/-- Given a geometric progression a, b, c, d and a quadratic function,
    prove that ad = 3 --/
theorem geometric_progression_quadratic_vertex (a b c d : ℝ) :
  (∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric progression condition
  (2 * b^2 - 4 * b + 5 = c) →                      -- vertex condition
  a * d = 3 := by
sorry

end geometric_progression_quadratic_vertex_l207_20723


namespace irwin_score_product_l207_20792

/-- Represents the types of baskets in Jamshid and Irwin's basketball game -/
inductive BasketType
  | Two
  | Five
  | Eleven
  | Thirteen

/-- Returns the point value of a given basket type -/
def basketValue (b : BasketType) : ℕ :=
  match b with
  | BasketType.Two => 2
  | BasketType.Five => 5
  | BasketType.Eleven => 11
  | BasketType.Thirteen => 13

/-- Irwin's score at halftime -/
def irwinScore : ℕ := 2 * basketValue BasketType.Eleven

theorem irwin_score_product : irwinScore = 22 := by
  sorry

end irwin_score_product_l207_20792


namespace ratio_equality_implies_fraction_value_l207_20714

theorem ratio_equality_implies_fraction_value 
  (x y z : ℝ) 
  (h : x / 3 = y / 5 ∧ y / 5 = z / 7) : 
  (y + z) / (3 * x - y) = 3 := by
sorry

end ratio_equality_implies_fraction_value_l207_20714


namespace jane_ate_12_swirls_l207_20731

/-- Given a number of cinnamon swirls and people, calculate how many swirls each person ate. -/
def swirls_per_person (total_swirls : ℕ) (num_people : ℕ) : ℕ :=
  total_swirls / num_people

/-- Theorem stating that Jane ate 12 cinnamon swirls. -/
theorem jane_ate_12_swirls (total_swirls : ℕ) (num_people : ℕ) 
  (h1 : total_swirls = 120) 
  (h2 : num_people = 10) :
  swirls_per_person total_swirls num_people = 12 := by
  sorry

#eval swirls_per_person 120 10

end jane_ate_12_swirls_l207_20731


namespace power_negative_product_l207_20753

theorem power_negative_product (a : ℝ) : (-a)^3 * (-a)^5 = a^8 := by
  sorry

end power_negative_product_l207_20753


namespace complementary_angles_theorem_l207_20777

theorem complementary_angles_theorem (α β : Real) : 
  (α + β = 180) →  -- complementary angles
  (α - β / 2 = 30) →  -- half of β is 30° less than α
  (α = 80) :=  -- measure of α is 80°
by sorry

end complementary_angles_theorem_l207_20777


namespace property_necessary_not_sufficient_l207_20715

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Definition of an arithmetic sequence -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property a₃ + a₇ = 2a₅ -/
def property (a : Sequence) : Prop :=
  a 3 + a 7 = 2 * a 5

/-- The main theorem stating that the property is necessary but not sufficient -/
theorem property_necessary_not_sufficient :
  (∀ a : Sequence, is_arithmetic a → property a) ∧
  (∃ a : Sequence, ¬is_arithmetic a ∧ property a) :=
sorry

end property_necessary_not_sufficient_l207_20715


namespace data_set_range_l207_20725

/-- The range of a data set with maximum value 78 and minimum value 21 is 57. -/
theorem data_set_range : ℝ → ℝ → ℝ → Prop :=
  fun (max min range : ℝ) =>
    max = 78 ∧ min = 21 → range = max - min → range = 57

/-- Proof of the theorem -/
lemma prove_data_set_range : data_set_range 78 21 57 := by
  sorry

end data_set_range_l207_20725


namespace percentage_difference_l207_20754

theorem percentage_difference : (150 * 62 / 100) - (250 * 20 / 100) = 43 := by
  sorry

end percentage_difference_l207_20754


namespace partnership_profit_share_l207_20781

/-- A partnership problem with four partners A, B, C, and D --/
theorem partnership_profit_share
  (capital_A : ℚ) (capital_B : ℚ) (capital_C : ℚ) (capital_D : ℚ) (total_profit : ℕ)
  (h1 : capital_A = 1 / 3)
  (h2 : capital_B = 1 / 4)
  (h3 : capital_C = 1 / 5)
  (h4 : capital_A + capital_B + capital_C + capital_D = 1)
  (h5 : total_profit = 2490) :
  ∃ (share_A : ℕ), share_A = 830 ∧ share_A = (capital_A * total_profit).num :=
sorry

end partnership_profit_share_l207_20781


namespace complex_fraction_real_l207_20799

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * I) / ((2 : ℂ) + I)).im = 0 → a = (1/2 : ℝ) := by
  sorry

end complex_fraction_real_l207_20799


namespace new_eurasian_bridge_length_scientific_notation_l207_20720

theorem new_eurasian_bridge_length_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 10900 = a * (10 : ℝ) ^ n ∧ a = 1.09 ∧ n = 4 := by
  sorry

end new_eurasian_bridge_length_scientific_notation_l207_20720


namespace defective_product_arrangements_l207_20760

theorem defective_product_arrangements :
  let total_products : ℕ := 7
  let defective_products : ℕ := 4
  let non_defective_products : ℕ := 3
  let third_defective_position : ℕ := 4

  (Nat.choose non_defective_products 1) *
  (Nat.choose defective_products 1) *
  (Nat.choose (defective_products - 1) 1) *
  1 *
  (Nat.choose 2 1) *
  ((total_products - third_defective_position) - (defective_products - 3)) = 288 :=
by sorry

end defective_product_arrangements_l207_20760


namespace unique_four_digit_square_with_repeated_digits_l207_20778

theorem unique_four_digit_square_with_repeated_digits : 
  ∃! n : ℕ, 
    1000 ≤ n ∧ n ≤ 9999 ∧ 
    (∃ m : ℕ, n = m^2) ∧
    (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1100 * a + 11 * b) ∧
    n = 7744 :=
by sorry

end unique_four_digit_square_with_repeated_digits_l207_20778


namespace johns_age_multiple_l207_20755

/-- Given the ages and relationships described in the problem, prove that John's age 3 years ago
    was twice James' age 6 years from now. -/
theorem johns_age_multiple (john_current_age james_brother_age james_brother_age_diff : ℕ)
  (h1 : john_current_age = 39)
  (h2 : james_brother_age = 16)
  (h3 : james_brother_age_diff = 4) : 
  (john_current_age - 3) = 2 * (james_brother_age - james_brother_age_diff + 6) := by
  sorry

end johns_age_multiple_l207_20755


namespace martha_cards_l207_20736

/-- The number of cards Martha initially had -/
def initial_cards : ℕ := 3

/-- The number of cards Martha received from Emily -/
def cards_from_emily : ℕ := 76

/-- The total number of cards Martha ended up with -/
def total_cards : ℕ := 79

/-- Theorem stating that the initial number of cards plus the cards received equals the total cards -/
theorem martha_cards : initial_cards + cards_from_emily = total_cards := by
  sorry

end martha_cards_l207_20736


namespace batsman_average_increase_l207_20735

theorem batsman_average_increase (total_runs : ℕ → ℕ) (innings : ℕ) :
  innings = 17 →
  total_runs innings = total_runs (innings - 1) + 74 →
  (total_runs innings : ℚ) / innings = 26 →
  (total_runs innings : ℚ) / innings - (total_runs (innings - 1) : ℚ) / (innings - 1) = 3 := by
  sorry

#check batsman_average_increase

end batsman_average_increase_l207_20735


namespace intersection_points_count_l207_20759

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem intersection_points_count 
  (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_def : ∀ x ∈ Set.Icc 0 2, f x = x^3 - x) :
  (Set.Icc 0 6 ∩ {x | f x = 0}).ncard = 7 := by
  sorry

end intersection_points_count_l207_20759


namespace evaluate_expression_l207_20775

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := by
  sorry

end evaluate_expression_l207_20775


namespace no_primes_of_form_l207_20763

theorem no_primes_of_form (m : ℕ) (hm : m > 0) : 
  ¬ Prime (2^(5*m) + 2^m + 1) := by
sorry

end no_primes_of_form_l207_20763


namespace cafe_menu_combinations_l207_20750

theorem cafe_menu_combinations (n : ℕ) (h : n = 12) : 
  n * (n - 1) = 132 := by
  sorry

end cafe_menu_combinations_l207_20750


namespace simplify_sqrt_expression_l207_20712

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (2 * x^3))^2) = x^3 / 2 + 1 / (2 * x^3) := by
  sorry

end simplify_sqrt_expression_l207_20712


namespace wizard_elixir_combinations_l207_20708

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted stones available. -/
def num_stones : ℕ := 6

/-- The number of stones incompatible with a specific herb. -/
def incompatible_stones : ℕ := 3

/-- The number of herbs that have incompatible stones. -/
def herbs_with_incompatibility : ℕ := 1

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_stones * herbs_with_incompatibility

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
sorry

end wizard_elixir_combinations_l207_20708


namespace salary_calculation_correct_l207_20732

/-- Calculates the salary after three months of increases -/
def salary_after_three_months (initial_salary : ℝ) (first_month_increase : ℝ) : ℝ :=
  let month1 := initial_salary * (1 + first_month_increase)
  let month2 := month1 * (1 + 2 * first_month_increase)
  let month3 := month2 * (1 + 4 * first_month_increase)
  month3

/-- Theorem stating that the salary after three months matches the expected value -/
theorem salary_calculation_correct : 
  salary_after_three_months 2000 0.05 = 2772 := by
  sorry


end salary_calculation_correct_l207_20732


namespace geometric_sequence_fourth_term_l207_20745

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_second : a 2 = 4)
  (h_sixth : a 6 = 64) :
  a 4 = 16 := by
sorry

end geometric_sequence_fourth_term_l207_20745


namespace complex_multiplication_l207_20726

theorem complex_multiplication (i : ℂ) : i * i = -1 → (-1 + i) * (2 - i) = -1 + 3 * i := by
  sorry

end complex_multiplication_l207_20726


namespace f_42_17_l207_20730

def is_valid_f (f : ℚ → Int) : Prop :=
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → f x * f y = -1) ∧
  (∀ x : ℚ, f x = 1 ∨ f x = -1) ∧
  f 0 = 1

theorem f_42_17 (f : ℚ → Int) (h : is_valid_f f) : f (42/17) = -1 := by
  sorry

end f_42_17_l207_20730


namespace function_ordering_l207_20776

/-- A function f is even with respect to x = -1 -/
def IsEvenShifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 1) = f (-x - 1)

/-- The function f is strictly decreasing in terms of its values when x > -1 -/
def IsStrictlyDecreasingShifted (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → (f x₂ - f x₁) * (x₂ - x₁) < 0

theorem function_ordering (f : ℝ → ℝ) 
    (h1 : IsEvenShifted f) 
    (h2 : IsStrictlyDecreasingShifted f) : 
    f (-2) < f 1 ∧ f 1 < f 2 := by
  sorry

end function_ordering_l207_20776


namespace arithmetic_sequence_sum_l207_20713

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_3 + a_7 = 37, the sum a_2 + a_4 + a_6 + a_8 = 74. -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end arithmetic_sequence_sum_l207_20713


namespace problem_statement_l207_20771

theorem problem_statement :
  ∀ (a b x y z : ℝ),
    (a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
    (let c := z^2 + 2*x + Real.pi/6;
     a = x^2 + 2*y + Real.pi/2 ∧
     b = y^2 + 2*z + Real.pi/3 →
     max a (max b c) > 0) :=
by sorry

end problem_statement_l207_20771


namespace triangle_max_area_l207_20744

/-- Given a triangle ABC with side lengths a, b, c opposite angles A, B, C respectively,
    prove that the maximum area is √3 under the given conditions. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (Real.sin A - Real.sin B) / Real.sin C = (c - b) / (2 + b) →
  (∀ a' b' c' A' B' C',
    a' = 2 →
    (Real.sin A' - Real.sin B') / Real.sin C' = (c' - b') / (2 + b') →
    (1/2) * a' * b' * Real.sin C' ≤ Real.sqrt 3) ∧
  (∃ b' c',
    (Real.sin A - Real.sin B) / Real.sin C = (c' - b') / (2 + b') →
    (1/2) * a * b' * Real.sin C = Real.sqrt 3) :=
by sorry

end triangle_max_area_l207_20744


namespace square_difference_area_l207_20791

theorem square_difference_area (a b : ℝ) : 
  (a + b)^2 - a^2 = 2*a*b + b^2 := by sorry

end square_difference_area_l207_20791


namespace x_value_when_one_in_set_l207_20769

theorem x_value_when_one_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x ≠ x^2 → x = -1 := by
  sorry

end x_value_when_one_in_set_l207_20769


namespace smallest_four_divisors_sum_of_squares_l207_20706

theorem smallest_four_divisors_sum_of_squares (n : ℕ+) 
  (d1 d2 d3 d4 : ℕ+) 
  (h_div : ∀ m : ℕ+, m ∣ n → m ≥ d1 ∧ m ≥ d2 ∧ m ≥ d3 ∧ m ≥ d4)
  (h_order : d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  (h_sum : n = d1^2 + d2^2 + d3^2 + d4^2) : 
  n = 130 := by
sorry

end smallest_four_divisors_sum_of_squares_l207_20706


namespace fraction_simplification_l207_20779

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  (3 * x / (x - 2) - x / (x + 2)) * ((x^2 - 4) / x) = 2 * x + 8 := by
  sorry

end fraction_simplification_l207_20779


namespace unique_abcabc_cube_minus_square_l207_20703

/-- A number is of the form abcabc if it equals 1001 * (100a + 10b + c) for some digits a, b, c -/
def is_abcabc (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 1001 * (100 * a + 10 * b + c)

/-- The main theorem stating that 78 is the unique positive integer x 
    such that x^3 - x^2 is a six-digit number of the form abcabc -/
theorem unique_abcabc_cube_minus_square :
  ∃! (x : ℕ), x > 0 ∧ 100000 ≤ x^3 - x^2 ∧ x^3 - x^2 < 1000000 ∧ is_abcabc (x^3 - x^2) ∧ x = 78 :=
sorry

end unique_abcabc_cube_minus_square_l207_20703


namespace triangle_area_l207_20786

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R satisfying certain conditions,
    prove that its area is (7√3201)/3 -/
theorem triangle_area (P Q R : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25) 
    (h3 : 2 * Real.cos Q = Real.cos P + Real.cos R) : 
    ∃ (area : ℝ), area = (7 * Real.sqrt 3201) / 3 ∧ 
    area = r * (P + Q + R) / 2 := by
  sorry


end triangle_area_l207_20786


namespace triangle_is_right_angled_l207_20743

theorem triangle_is_right_angled (a b c : ℝ) : 
  a = 3 ∧ b = 5 ∧ (3 * c^2 - 10 * c = 8) ∧ c > 0 → 
  a^2 + c^2 = b^2 := by
  sorry

end triangle_is_right_angled_l207_20743


namespace num_large_beds_is_two_l207_20701

/-- The number of seeds that can be planted in a large bed -/
def large_bed_capacity : ℕ := 100

/-- The number of seeds that can be planted in a medium bed -/
def medium_bed_capacity : ℕ := 60

/-- The number of medium beds -/
def num_medium_beds : ℕ := 2

/-- The total number of seeds that can be planted -/
def total_seeds : ℕ := 320

/-- Theorem stating that the number of large beds is 2 -/
theorem num_large_beds_is_two :
  ∃ (n : ℕ), n * large_bed_capacity + num_medium_beds * medium_bed_capacity = total_seeds ∧ n = 2 :=
sorry

end num_large_beds_is_two_l207_20701


namespace complement_characterization_l207_20783

-- Define the universe of quadrilaterals
def Quadrilateral : Type := sorry

-- Define properties of quadrilaterals
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def has_right_angle (q : Quadrilateral) : Prop := sorry

-- Define sets A and B
def A : Set Quadrilateral := {q | is_rhombus q ∨ is_rectangle q}
def B : Set Quadrilateral := {q | is_rectangle q}

-- Define the complement of B with respect to A
def C_AB : Set Quadrilateral := A \ B

-- Theorem to prove
theorem complement_characterization :
  C_AB = {q : Quadrilateral | is_rhombus q ∧ ¬has_right_angle q} :=
sorry

end complement_characterization_l207_20783


namespace combination_equality_l207_20793

theorem combination_equality (x : ℕ) : 
  (Nat.choose 14 x = Nat.choose 14 (2*x - 4)) → (x = 4 ∨ x = 6) := by
  sorry

end combination_equality_l207_20793


namespace omega_sum_l207_20766

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = -ω^2 := by
  sorry

end omega_sum_l207_20766


namespace prop_1_false_prop_2_true_prop_3_false_l207_20727

-- Proposition 1
theorem prop_1_false : ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  a * d = b * c ∧ ¬(∃ r : ℝ, (b = a * r ∧ c = b * r ∧ d = c * r) ∨ 
                             (a = b * r ∧ b = c * r ∧ c = d * r)) := by
  sorry

-- Proposition 2
theorem prop_2_true : ∀ (a : ℤ), 2 ∣ a → Even a := by
  sorry

-- Proposition 3
theorem prop_3_false : ∃ (A : ℝ), 
  30 * π / 180 < A ∧ A < π ∧ Real.sin A ≤ 1 / 2 := by
  sorry

end prop_1_false_prop_2_true_prop_3_false_l207_20727


namespace spinner_probability_F_l207_20782

/-- Represents a spinner with three sections -/
structure Spinner :=
  (D : ℚ) (E : ℚ) (F : ℚ)

/-- The probability of landing on each section of the spinner -/
def probability (s : Spinner) : ℚ := s.D + s.E + s.F

theorem spinner_probability_F (s : Spinner) 
  (hD : s.D = 2/5) 
  (hE : s.E = 1/5) 
  (hP : probability s = 1) : 
  s.F = 2/5 := by
  sorry


end spinner_probability_F_l207_20782


namespace jerome_theorem_l207_20704

def jerome_problem (initial_money : ℝ) : Prop :=
  let half_money : ℝ := 43
  let meg_amount : ℝ := 8
  let bianca_amount : ℝ := 3 * meg_amount
  let after_meg_bianca : ℝ := initial_money - meg_amount - bianca_amount
  let nathan_amount : ℝ := after_meg_bianca / 2
  let after_nathan : ℝ := after_meg_bianca - nathan_amount
  let charity_percentage : ℝ := 0.2
  let charity_amount : ℝ := charity_percentage * after_nathan
  let final_amount : ℝ := after_nathan - charity_amount

  (initial_money / 2 = half_money) ∧
  (final_amount = 21.60)

theorem jerome_theorem : 
  ∃ (initial_money : ℝ), jerome_problem initial_money :=
by
  sorry

end jerome_theorem_l207_20704


namespace max_value_implies_sum_l207_20733

/-- The function f(x) = x^3 + ax^2 + bx - a^2 - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem max_value_implies_sum (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧ 
  (f a b 1 = 10) ∧ 
  (f' a b 1 = 0) →
  a + b = 3 := by sorry

end max_value_implies_sum_l207_20733


namespace triangle_properties_l207_20707

/-- Represents a triangle with sides x, y, and z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if the triangle satisfies the given conditions -/
def satisfiesConditions (t : Triangle) (a : ℝ) : Prop :=
  t.x + t.y = 3 * t.z ∧
  t.z + t.y = t.x + a ∧
  t.x + t.z = 60

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties :
  ∀ (t : Triangle) (a : ℝ),
    satisfiesConditions t a →
    (0 < a ∧ a < 60) ∧
    (a = 30 → t.x = 35 ∧ t.y = 40 ∧ t.z = 25) :=
by sorry

end triangle_properties_l207_20707


namespace georges_walk_speed_l207_20788

/-- Proves that given the conditions, George must walk at 6 mph for the last segment to arrive on time -/
theorem georges_walk_speed (total_distance : Real) (normal_speed : Real) (first_half_distance : Real) (first_half_speed : Real) :
  total_distance = 1.5 →
  normal_speed = 3 →
  first_half_distance = 0.75 →
  first_half_speed = 2 →
  (total_distance / normal_speed - first_half_distance / first_half_speed) / (total_distance - first_half_distance) = 6 := by
  sorry


end georges_walk_speed_l207_20788


namespace place_value_ratio_l207_20780

def number : ℚ := 86572.4908

theorem place_value_ratio : 
  ∃ (tens hundredths : ℚ), 
    (tens = 10) ∧ 
    (hundredths = 0.01) ∧ 
    (tens / hundredths = 1000) :=
by sorry

end place_value_ratio_l207_20780


namespace mushroom_picking_profit_l207_20787

/-- Calculates the money made on the first day of a three-day mushroom picking trip -/
theorem mushroom_picking_profit (total_mushrooms day2_mushrooms price_per_mushroom : ℕ) : 
  total_mushrooms = 65 →
  day2_mushrooms = 12 →
  price_per_mushroom = 2 →
  (total_mushrooms - day2_mushrooms - 2 * day2_mushrooms) * price_per_mushroom = 58 := by
  sorry

end mushroom_picking_profit_l207_20787


namespace cube_minus_reciprocal_cube_l207_20747

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : 
  x^3 - 1/x^3 = 125 := by sorry

end cube_minus_reciprocal_cube_l207_20747


namespace polynomial_division_l207_20761

theorem polynomial_division (x : ℝ) (h : x ≠ 0) : 2 * x^3 / x^2 = 2 * x := by
  sorry

end polynomial_division_l207_20761


namespace smallest_solution_congruence_l207_20770

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (3 * x) % 17 = 14 % 17 ∧ ∀ (y : ℕ), y > 0 → (3 * y) % 17 = 14 % 17 → x ≤ y :=
by sorry

end smallest_solution_congruence_l207_20770


namespace engineer_designer_ratio_l207_20772

theorem engineer_designer_ratio (e d : ℕ) (h_total : (40 * e + 55 * d) / (e + d) = 45) :
  e = 2 * d := by
  sorry

end engineer_designer_ratio_l207_20772


namespace min_value_sum_reciprocals_l207_20717

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

end min_value_sum_reciprocals_l207_20717


namespace sphere_surface_area_rectangular_solid_l207_20765

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * Real.pi * radius^2 = 50 * Real.pi := by
  sorry

end sphere_surface_area_rectangular_solid_l207_20765


namespace circle_area_theorem_l207_20728

theorem circle_area_theorem (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
  (h1 : r = 42)
  (h2 : chord_length = 78)
  (h3 : intersection_distance = 18) :
  ∃ (m n d : ℕ), 
    (m * π - n * Real.sqrt d : ℝ) = 294 * π - 81 * Real.sqrt 3 ∧
    m + n + d = 378 :=
by sorry

end circle_area_theorem_l207_20728


namespace min_sum_xyz_l207_20719

theorem min_sum_xyz (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) :
  ∀ a b c : ℤ, (a - 10) * (b - 5) * (c - 2) = 1000 → x + y + z ≤ a + b + c ∧ x + y + z = 92 :=
sorry

end min_sum_xyz_l207_20719


namespace min_value_expression_l207_20710

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 4 * x^2 + y^2 + 1 / (x * y) ≥ 17 / 2 :=
by sorry

end min_value_expression_l207_20710


namespace base_seven_addition_problem_l207_20716

/-- Given a base 7 addition problem 3XY₇ + 52₇ = 42X₇, prove that X + Y = 6 in base 10 -/
theorem base_seven_addition_problem (X Y : Fin 7) :
  (3 * 7 * 7 + X * 7 + Y) + (5 * 7 + 2) = 4 * 7 * 7 + 2 * 7 + X →
  (X : ℕ) + (Y : ℕ) = 6 := by
  sorry

end base_seven_addition_problem_l207_20716


namespace fraction_equality_l207_20757

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 6)
  (h3 : p / q = 1 / 15) :
  m / q = 1 / 5 := by sorry

end fraction_equality_l207_20757


namespace stratified_sample_correct_l207_20729

/-- Represents the number of people to be selected from each group in a stratified sampling --/
structure StratifiedSample where
  regular : ℕ
  middle : ℕ
  senior : ℕ

/-- Calculates the stratified sample given total employees and managers --/
def calculateStratifiedSample (total : ℕ) (middle : ℕ) (senior : ℕ) (toSelect : ℕ) : StratifiedSample :=
  sorry

/-- Theorem stating that the calculated stratified sample is correct --/
theorem stratified_sample_correct :
  let total := 160
  let middle := 30
  let senior := 10
  let toSelect := 20
  let result := calculateStratifiedSample total middle senior toSelect
  result.regular = 16 ∧ result.middle = 3 ∧ result.senior = 1 := by
  sorry

end stratified_sample_correct_l207_20729


namespace car_hire_problem_l207_20789

theorem car_hire_problem (total_cost : ℝ) (a_hours c_hours : ℝ) (b_cost : ℝ) :
  total_cost = 520 →
  a_hours = 7 →
  c_hours = 11 →
  b_cost = 160 →
  ∃ b_hours : ℝ,
    b_cost = (total_cost / (a_hours + b_hours + c_hours)) * b_hours ∧
    b_hours = 8 := by
  sorry

end car_hire_problem_l207_20789


namespace rope_length_l207_20709

/-- Given a rope cut into two parts with a ratio of 2:3, where the shorter part is 16 meters long,
    the total length of the rope is 40 meters. -/
theorem rope_length (shorter_part : ℝ) (ratio_short : ℝ) (ratio_long : ℝ) :
  shorter_part = 16 →
  ratio_short = 2 →
  ratio_long = 3 →
  (shorter_part / ratio_short) * (ratio_short + ratio_long) = 40 :=
by sorry

end rope_length_l207_20709


namespace area_smaller_circle_l207_20740

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  h_positive : 0 < r
  h_tangent : R = 2 * r
  h_common_tangent : ∃ (P A B : ℝ × ℝ), 
    let d := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
    d = 5 ∧ d = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

/-- The area of the smaller circle in a TangentCircles configuration is 25π/8 -/
theorem area_smaller_circle (tc : TangentCircles) : 
  π * tc.r^2 = 25 * π / 8 := by
  sorry

end area_smaller_circle_l207_20740


namespace sum_of_xy_l207_20718

theorem sum_of_xy (x y : ℕ+) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 18 ∨ x + y = 10 := by
  sorry

end sum_of_xy_l207_20718


namespace triangle_properties_l207_20758

/-- Given a triangle ABC with the following properties:
  - m = (sin C, sin B cos A)
  - n = (b, 2c)
  - m · n = 0
  - a = 2√3
  - sin B + sin C = 1
  Prove that:
  1. The measure of angle A is 2π/3
  2. The area of triangle ABC is √3
-/
theorem triangle_properties (a b c A B C : ℝ) 
  (m : ℝ × ℝ) (n : ℝ × ℝ) 
  (hm : m = (Real.sin C, Real.sin B * Real.cos A))
  (hn : n = (b, 2 * c))
  (hdot : m.1 * n.1 + m.2 * n.2 = 0)
  (ha : a = 2 * Real.sqrt 3)
  (hsin : Real.sin B + Real.sin C = 1) :
  A = 2 * Real.pi / 3 ∧ 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry

end triangle_properties_l207_20758


namespace two_inequalities_l207_20797

theorem two_inequalities :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 ≥ a*b + a*c + b*c) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end two_inequalities_l207_20797


namespace upstream_speed_calculation_l207_20724

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the man's speed in still water and downstream,
    his upstream speed is 20 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed)
  (h1 : s.stillWater = 24)
  (h2 : s.downstream = 28) :
  upstreamSpeed s = 20 := by
sorry

#eval upstreamSpeed { stillWater := 24, downstream := 28 }

end upstream_speed_calculation_l207_20724


namespace geometric_sequence_cubic_root_count_l207_20721

/-- Given a, b, c form a geometric sequence, the equation ax³ + bx² + cx = 0 has exactly one real root -/
theorem geometric_sequence_cubic_root_count 
  (a b c : ℝ) 
  (h_geom : ∃ (r : ℝ), b = a * r ∧ c = b * r ∧ r ≠ 0) :
  (∃! x : ℝ, a * x^3 + b * x^2 + c * x = 0) :=
sorry

end geometric_sequence_cubic_root_count_l207_20721


namespace larger_segment_is_50_l207_20762

/-- Represents a triangle with sides a, b, c and an altitude h dropped on side c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  x : ℝ  -- shorter segment of side c
  valid_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  altitude_property : a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2

/-- The larger segment of the side c in a triangle with sides 40, 50, 90 is 50 --/
theorem larger_segment_is_50 :
  ∀ t : Triangle, t.a = 40 ∧ t.b = 50 ∧ t.c = 90 → (t.c - t.x = 50) :=
by sorry

end larger_segment_is_50_l207_20762


namespace line_parallel_to_plane_l207_20796

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the "not contained in" relation between a line and a plane
variable (notContainedIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane (l : Line) (α : Plane) :
  notContainedIn l α → parallel l α := by sorry

end line_parallel_to_plane_l207_20796


namespace consecutive_squares_not_equal_consecutive_fourth_powers_l207_20702

theorem consecutive_squares_not_equal_consecutive_fourth_powers :
  ∀ x y : ℕ+, x^2 + (x + 1)^2 ≠ y^4 + (y + 1)^4 := by
sorry

end consecutive_squares_not_equal_consecutive_fourth_powers_l207_20702


namespace inequality_proof_l207_20705

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^3 / (a^3 + 2*b^2)) + (b^3 / (b^3 + 2*c^2)) + (c^3 / (c^3 + 2*a^2)) ≥ 1 := by
  sorry

end inequality_proof_l207_20705


namespace unique_four_digit_number_l207_20738

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℕ := (n / 1000) - (n % 10)

theorem unique_four_digit_number : 
  ∃! n : ℕ, 
    is_four_digit n ∧ 
    digit_sum n = 16 ∧ 
    middle_digits_sum n = 10 ∧ 
    thousands_minus_units n = 2 ∧ 
    n % 11 = 0 ∧
    n = 4642 := by sorry

end unique_four_digit_number_l207_20738


namespace largest_power_of_two_dividing_2007_to_1024_minus_1_l207_20794

theorem largest_power_of_two_dividing_2007_to_1024_minus_1 :
  (∃ (n : ℕ), 2^n ∣ (2007^1024 - 1)) ∧
  (∀ (m : ℕ), m > 14 → ¬(2^m ∣ (2007^1024 - 1))) :=
sorry

end largest_power_of_two_dividing_2007_to_1024_minus_1_l207_20794


namespace f_less_than_g_for_x_greater_than_one_l207_20734

/-- Given functions f and g with specified properties, f(x) < g(x) for x > 1 -/
theorem f_less_than_g_for_x_greater_than_one 
  (f g : ℝ → ℝ)
  (h_f : ∀ x, f x = Real.log x)
  (h_g : ∃ a b : ℝ, ∀ x, g x = a * x + b / x)
  (h_common_tangent : ∃ x₀, x₀ > 0 ∧ f x₀ = g x₀ ∧ (deriv f) x₀ = (deriv g) x₀)
  (x : ℝ)
  (h_x : x > 1) :
  f x < g x := by
sorry

end f_less_than_g_for_x_greater_than_one_l207_20734


namespace circle_from_equation_l207_20768

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in general form -/
def CircleEquation (x y : ℝ) (A B C D E : ℝ) : Prop :=
  A * x^2 + B * x + C * y^2 + D * y + E = 0

theorem circle_from_equation :
  ∃ (c : Circle), 
    (∀ (x y : ℝ), CircleEquation x y 1 (-6) 1 2 (-12) ↔ 
      (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
    c.center = (3, -1) ∧
    c.radius = Real.sqrt 22 := by
  sorry

end circle_from_equation_l207_20768


namespace scientific_notation_of_31400000_l207_20774

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_31400000 :
  toScientificNotation 31400000 = ScientificNotation.mk 3.14 7 (by norm_num) :=
sorry

end scientific_notation_of_31400000_l207_20774


namespace different_answers_for_fedya_question_l207_20741

-- Define the types of people
inductive Person : Type
| truthTeller : Person
| liar : Person

-- Define the possible answers
inductive Answer : Type
| yes : Answer
| no : Answer

-- Define the function that determines how a person answers
def answerQuestion (p : Person) (isNameFedya : Bool) : Answer :=
  match p with
  | Person.truthTeller => if isNameFedya then Answer.yes else Answer.no
  | Person.liar => if isNameFedya then Answer.no else Answer.yes

-- State the theorem
theorem different_answers_for_fedya_question 
  (fedya : Person) 
  (vadim : Person) 
  (h1 : fedya = Person.truthTeller) 
  (h2 : vadim = Person.liar) :
  answerQuestion fedya true ≠ answerQuestion vadim false :=
sorry

end different_answers_for_fedya_question_l207_20741


namespace largest_five_digit_multiple_largest_five_digit_multiple_exists_l207_20739

theorem largest_five_digit_multiple (n : Nat) : n ≤ 99999 ∧ n ≥ 10000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ 99936 :=
by
  sorry

theorem largest_five_digit_multiple_exists : ∃ n : Nat, n = 99936 ∧ n ≤ 99999 ∧ n ≥ 10000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n :=
by
  sorry

end largest_five_digit_multiple_largest_five_digit_multiple_exists_l207_20739


namespace sunshine_car_rentals_rate_l207_20722

theorem sunshine_car_rentals_rate (sunshine_daily_rate city_daily_rate city_mile_rate : ℚ)
  (equal_cost_miles : ℕ) :
  sunshine_daily_rate = 17.99 ∧
  city_daily_rate = 18.95 ∧
  city_mile_rate = 0.16 ∧
  equal_cost_miles = 48 →
  ∃ sunshine_mile_rate : ℚ,
    sunshine_mile_rate = 0.18 ∧
    sunshine_daily_rate + sunshine_mile_rate * equal_cost_miles =
    city_daily_rate + city_mile_rate * equal_cost_miles :=
by sorry

end sunshine_car_rentals_rate_l207_20722


namespace smallest_a_with_50_squares_l207_20767

theorem smallest_a_with_50_squares : ∃ (a : ℕ), 
  (a = 4486) ∧ 
  (∀ k : ℕ, k < a → (∃ (n : ℕ), n * n > k ∧ n * n < 3 * k) → 
    (∃ (m : ℕ), m < 50)) ∧
  (∃ (l : ℕ), l = 50 ∧ 
    (∀ i : ℕ, i ≤ l → ∃ (s : ℕ), s * s > a ∧ s * s < 3 * a)) :=
sorry

end smallest_a_with_50_squares_l207_20767


namespace fraction_evaluation_l207_20795

theorem fraction_evaluation :
  ⌈(23 / 11 : ℚ) - ⌈(37 / 19 : ℚ)⌉⌉ / ⌈(35 / 11 : ℚ) + ⌈(11 * 19 / 37 : ℚ)⌉⌉ = (1 / 10 : ℚ) := by
  sorry

end fraction_evaluation_l207_20795


namespace ten_team_round_robin_l207_20748

def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

theorem ten_team_round_robin :
  roundRobinGames 10 = 45 := by
  sorry

end ten_team_round_robin_l207_20748


namespace quadratic_sum_bound_l207_20749

/-- Represents a quadratic function of the form y = x^2 - (a+2)x + 2a + 1 -/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - (a + 2) * x + 2 * a + 1

/-- Theorem: For a quadratic function passing through (-1, y₀) where y₀ is the minimum,
    any two different points A(m, n) and B(2-m, p) on the parabola satisfy n + p > -8 -/
theorem quadratic_sum_bound
  (a : ℝ)
  (y₀ : ℝ)
  (h1 : QuadraticFunction a (-1) = y₀)
  (h2 : ∀ x y, y = QuadraticFunction a x → y ≥ y₀)
  (m n p : ℝ)
  (h3 : n = QuadraticFunction a m)
  (h4 : p = QuadraticFunction a (2 - m))
  (h5 : m ≠ 2 - m) :
  n + p > -8 := by
  sorry

end quadratic_sum_bound_l207_20749


namespace merchant_salt_problem_l207_20737

theorem merchant_salt_problem (x : ℝ) : 
  (x > 0) →
  (x + 100 > x) →
  (x + 220 > x + 100) →
  (x / (x + 100) = (x + 100) / (x + 220)) →
  (x = 500) :=
by
  sorry

end merchant_salt_problem_l207_20737


namespace flowerbed_perimeter_sum_l207_20751

/-- Calculates the perimeter of a rectangle given its width and length -/
def rectanglePerimeter (width length : ℝ) : ℝ := 2 * (width + length)

/-- Proves that the total perimeter of three flowerbeds with given dimensions is 69 meters -/
theorem flowerbed_perimeter_sum : 
  let flowerbed1_width : ℝ := 4
  let flowerbed1_length : ℝ := 2 * flowerbed1_width - 1
  let flowerbed2_length : ℝ := flowerbed1_length + 3
  let flowerbed2_width : ℝ := flowerbed1_width - 2
  let flowerbed3_width : ℝ := (flowerbed1_width + flowerbed2_width) / 2
  let flowerbed3_length : ℝ := (flowerbed1_length + flowerbed2_length) / 2
  rectanglePerimeter flowerbed1_width flowerbed1_length +
  rectanglePerimeter flowerbed2_width flowerbed2_length +
  rectanglePerimeter flowerbed3_width flowerbed3_length = 69 := by
  sorry


end flowerbed_perimeter_sum_l207_20751


namespace vector_calculation_l207_20764

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_calculation : 
  (-2 • a + 4 • b) = ![-6, -8] := by sorry

end vector_calculation_l207_20764


namespace f_sum_negative_l207_20742

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property_1 : ∀ x : ℝ, f (-x) = -f (x + 4)
axiom f_property_2 : ∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 2 → f x₁ > f x₂

-- Define the theorem
theorem f_sum_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 := by
  sorry

end f_sum_negative_l207_20742


namespace sqrt_neg_four_squared_l207_20773

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end sqrt_neg_four_squared_l207_20773


namespace brick_length_proof_l207_20790

theorem brick_length_proof (w h A : ℝ) (hw : w = 4) (hh : h = 3) (hA : A = 164) :
  let l := (A - 2 * w * h) / (2 * (w + h))
  l = 10 := by sorry

end brick_length_proof_l207_20790


namespace certain_value_problem_l207_20752

theorem certain_value_problem (n v : ℝ) : n = 10 → (1/2) * n + v = 11 → v = 6 := by
  sorry

end certain_value_problem_l207_20752


namespace modulus_of_complex_quotient_l207_20798

theorem modulus_of_complex_quotient :
  Complex.abs (Complex.I / (1 + 2 * Complex.I)) = Real.sqrt 5 / 5 := by
  sorry

end modulus_of_complex_quotient_l207_20798


namespace solution_set_eq_neg_one_one_l207_20746

-- Define the solution set of x^2 - 1 = 0
def solution_set : Set ℝ := {x : ℝ | x^2 - 1 = 0}

-- Theorem stating that the solution set is exactly {-1, 1}
theorem solution_set_eq_neg_one_one : solution_set = {-1, 1} := by
  sorry

end solution_set_eq_neg_one_one_l207_20746


namespace problem_1_problem_2_problem_3_problem_4_l207_20785

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by sorry

-- Problem 4
theorem problem_4 : (-19 - 15/16) * 8 = -159 - 1/2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l207_20785


namespace jungkook_total_sheets_l207_20756

/-- The number of sheets in a bundle of colored paper -/
def sheets_per_bundle : ℕ := 10

/-- The number of bundles Jungkook has -/
def bundles : ℕ := 3

/-- The number of additional individual sheets Jungkook has -/
def individual_sheets : ℕ := 8

/-- Theorem stating the total number of sheets Jungkook has -/
theorem jungkook_total_sheets :
  bundles * sheets_per_bundle + individual_sheets = 38 := by
  sorry

end jungkook_total_sheets_l207_20756


namespace square_perimeter_l207_20784

theorem square_perimeter (s : ℝ) (h : s = 13) : 4 * s = 52 := by
  sorry

end square_perimeter_l207_20784


namespace total_worth_of_toys_l207_20711

theorem total_worth_of_toys (total_toys : Nat) (special_toy_value : Nat) (regular_toy_value : Nat) :
  total_toys = 9 →
  special_toy_value = 12 →
  regular_toy_value = 5 →
  (total_toys - 1) * regular_toy_value + special_toy_value = 52 := by
  sorry

end total_worth_of_toys_l207_20711
