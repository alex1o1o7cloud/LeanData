import Mathlib

namespace x_equals_nine_l3306_330605

/-- The star operation defined as a ⭐ b = 5a - 3b -/
def star (a b : ℝ) : ℝ := 5 * a - 3 * b

/-- Theorem stating that X = 9 given the condition X ⭐ (3 ⭐ 2) = 18 -/
theorem x_equals_nine : ∃ X : ℝ, star X (star 3 2) = 18 ∧ X = 9 := by
  sorry

end x_equals_nine_l3306_330605


namespace greatest_4digit_base9_divisible_by_7_l3306_330641

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 9 number --/
def is4DigitBase9 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 8888

theorem greatest_4digit_base9_divisible_by_7 :
  ∃ n : ℕ, is4DigitBase9 n ∧
           base9ToDecimal n % 7 = 0 ∧
           ∀ m : ℕ, is4DigitBase9 m ∧ base9ToDecimal m % 7 = 0 → m ≤ n ∧
           n = 9000 :=
sorry

end greatest_4digit_base9_divisible_by_7_l3306_330641


namespace solve_banana_cost_l3306_330680

def banana_cost_problem (initial_amount remaining_amount pears_cost asparagus_cost chicken_cost : ℕ) 
  (num_banana_packs : ℕ) : Prop :=
  let total_spent := initial_amount - remaining_amount
  let other_items_cost := pears_cost + asparagus_cost + chicken_cost
  let banana_total_cost := total_spent - other_items_cost
  banana_total_cost / num_banana_packs = 4

theorem solve_banana_cost :
  banana_cost_problem 55 28 2 6 11 2 := by
  sorry

end solve_banana_cost_l3306_330680


namespace number_subtraction_problem_l3306_330614

theorem number_subtraction_problem : ∃! x : ℝ, 0.4 * x - 11 = 23 := by
  sorry

end number_subtraction_problem_l3306_330614


namespace no_distributive_laws_hold_l3306_330646

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2 * b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
sorry

end no_distributive_laws_hold_l3306_330646


namespace total_trophies_is_430_l3306_330600

/-- Calculates the total number of trophies Jack and Michael will have after three years -/
def totalTrophiesAfterThreeYears (michaelCurrentTrophies : ℕ) (michaelTrophyIncrease : ℕ) (jackMultiplier : ℕ) : ℕ :=
  let michaelFutureTrophies := michaelCurrentTrophies + michaelTrophyIncrease
  let jackFutureTrophies := jackMultiplier * michaelCurrentTrophies
  michaelFutureTrophies + jackFutureTrophies

/-- Theorem stating that the total number of trophies after three years is 430 -/
theorem total_trophies_is_430 : 
  totalTrophiesAfterThreeYears 30 100 10 = 430 := by
  sorry

end total_trophies_is_430_l3306_330600


namespace min_value_of_f_l3306_330690

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end min_value_of_f_l3306_330690


namespace circle_radii_sum_l3306_330696

theorem circle_radii_sum : ∀ r : ℝ, 
  r > 0 →
  (r - 4)^2 + r^2 = (r + 2)^2 →
  ∃ r' : ℝ, r' > 0 ∧ (r' - 4)^2 + r'^2 = (r' + 2)^2 ∧ r + r' = 12 :=
by sorry

end circle_radii_sum_l3306_330696


namespace range_of_m_when_g_has_three_zeros_l3306_330693

/-- The quadratic function f(x) -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- The function g(x) -/
def g (m : ℝ) (x : ℝ) : ℝ := |f x| - f x - 2*m*x - 2*m^2

/-- The theorem stating the range of m when g(x) has three distinct zeros -/
theorem range_of_m_when_g_has_three_zeros :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  m ∈ Set.Ioo ((1 - 2*Real.sqrt 7)/3) (-1) ∪ Set.Ioo 2 ((1 + 2*Real.sqrt 7)/3) :=
sorry

end range_of_m_when_g_has_three_zeros_l3306_330693


namespace decagon_ratio_l3306_330676

/-- Represents a decagon with specific properties -/
structure Decagon where
  unit_squares : ℕ
  triangles : ℕ
  triangle_base : ℝ
  bottom_square : ℕ
  bottom_area : ℝ

/-- Theorem statement for the decagon problem -/
theorem decagon_ratio 
  (d : Decagon)
  (h1 : d.unit_squares = 12)
  (h2 : d.triangles = 2)
  (h3 : d.triangle_base = 3)
  (h4 : d.bottom_square = 1)
  (h5 : d.bottom_area = 6)
  : ∃ (xq yq : ℝ), xq / yq = 1 ∧ xq + yq = 3 := by
  sorry

end decagon_ratio_l3306_330676


namespace remainder_eight_pow_six_plus_one_mod_seven_l3306_330630

theorem remainder_eight_pow_six_plus_one_mod_seven :
  (8^6 + 1) % 7 = 2 := by
  sorry

end remainder_eight_pow_six_plus_one_mod_seven_l3306_330630


namespace g_expression_l3306_330650

-- Define polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom poly_sum : ∀ x, f x + g x = 2 * x^2 + 3 * x + 4
axiom f_def : ∀ x, f x = 2 * x^3 - x^2 - 4 * x + 5

-- State the theorem
theorem g_expression : ∀ x, g x = -2 * x^3 + 3 * x^2 + 7 * x - 1 := by
  sorry

end g_expression_l3306_330650


namespace root_sum_square_problem_l3306_330662

theorem root_sum_square_problem (α β : ℝ) : 
  (α^2 + 2*α - 2025 = 0) → 
  (β^2 + 2*β - 2025 = 0) → 
  α^2 + 3*α + β = 2023 := by
sorry

end root_sum_square_problem_l3306_330662


namespace power_calculation_l3306_330610

theorem power_calculation : ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 := by
  sorry

end power_calculation_l3306_330610


namespace original_profit_margin_exists_l3306_330647

/-- Given a reduction in purchase price and an increase in profit margin,
    there exists a unique original profit margin. -/
theorem original_profit_margin_exists :
  ∃! x : ℝ, 
    0 ≤ x ∧ x ≤ 100 ∧
    (1 + (x + 8) / 100) * (1 - 0.064) = 1 + x / 100 :=
by sorry

end original_profit_margin_exists_l3306_330647


namespace least_student_number_l3306_330629

theorem least_student_number (p : ℕ) (q : ℕ) : 
  q % 7 = 0 ∧ 
  q ≥ 1000 ∧ 
  q % (p + 1) = 1 ∧ 
  q % (p + 2) = 1 ∧ 
  q % (p + 3) = 1 ∧ 
  (∀ r : ℕ, r % 7 = 0 ∧ 
            r ≥ 1000 ∧ 
            r % (p + 1) = 1 ∧ 
            r % (p + 2) = 1 ∧ 
            r % (p + 3) = 1 → 
            q ≤ r) → 
  q = 1141 :=
by sorry

end least_student_number_l3306_330629


namespace rectangular_strip_dimensions_l3306_330648

theorem rectangular_strip_dimensions (a b c : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43 →
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) :=
by sorry

end rectangular_strip_dimensions_l3306_330648


namespace leftover_value_l3306_330623

/-- The number of quarters in a roll -/
def quarters_per_roll : ℕ := 50

/-- The number of dimes in a roll -/
def dimes_per_roll : ℕ := 40

/-- The number of quarters Kim has -/
def kim_quarters : ℕ := 95

/-- The number of dimes Kim has -/
def kim_dimes : ℕ := 183

/-- The number of quarters Mark has -/
def mark_quarters : ℕ := 157

/-- The number of dimes Mark has -/
def mark_dimes : ℕ := 328

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1 / 4

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The total value of leftover coins after making complete rolls -/
theorem leftover_value : 
  let total_quarters := kim_quarters + mark_quarters
  let total_dimes := kim_dimes + mark_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value = 18/5 := by
  sorry

end leftover_value_l3306_330623


namespace remainder_equality_l3306_330625

theorem remainder_equality (a b : ℕ) (h1 : a ≠ b) (h2 : a > b) :
  ∃ (q1 q2 r : ℕ), a = (a - b) * q1 + r ∧ b = (a - b) * q2 + r ∧ r < a - b :=
sorry

end remainder_equality_l3306_330625


namespace upper_pyramid_volume_l3306_330643

/-- The volume of the upper smaller pyramid formed by cutting a right square pyramid -/
theorem upper_pyramid_volume 
  (base_edge : ℝ) 
  (slant_edge : ℝ) 
  (cut_height : ℝ) 
  (h : base_edge = 12 * Real.sqrt 2) 
  (s : slant_edge = 15) 
  (c : cut_height = 5) : 
  ∃ (volume : ℝ), 
    volume = (1/6) * ((12 * Real.sqrt 2 * (Real.sqrt 153 - 5)) / Real.sqrt 153)^2 * (Real.sqrt 153 - 5) :=
by sorry

end upper_pyramid_volume_l3306_330643


namespace midpoint_minus_eighth_l3306_330652

theorem midpoint_minus_eighth (a b c : ℚ) : 
  a = 1/4 → b = 1/2 → c = 1/8 → 
  ((a + b) / 2) - c = 1/4 := by sorry

end midpoint_minus_eighth_l3306_330652


namespace least_odd_number_satisfying_conditions_l3306_330626

theorem least_odd_number_satisfying_conditions : ∃ (m₁ m₂ n₁ n₂ : ℕ+), 
  let a : ℕ := 261
  (a = m₁.val ^ 2 + n₁.val ^ 2) ∧
  (a ^ 2 = m₂.val ^ 2 + n₂.val ^ 2) ∧
  (m₁.val - n₁.val = m₂.val - n₂.val) ∧
  (∀ (b : ℕ) (k₁ k₂ l₁ l₂ : ℕ+), b < a → b % 2 = 1 → b > 5 →
    (b = k₁.val ^ 2 + l₁.val ^ 2 ∧
     b ^ 2 = k₂.val ^ 2 + l₂.val ^ 2 ∧
     k₁.val - l₁.val = k₂.val - l₂.val) → False) :=
by sorry

end least_odd_number_satisfying_conditions_l3306_330626


namespace sum_of_factors_60_l3306_330631

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_60 : sum_of_factors 60 = 168 := by
  sorry

end sum_of_factors_60_l3306_330631


namespace odd_function_extension_l3306_330609

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension
  (f : ℝ → ℝ)
  (odd : is_odd f)
  (pos_def : ∀ x > 0, f x = x - 1) :
  ∀ x < 0, f x = x + 1 := by
sorry

end odd_function_extension_l3306_330609


namespace square_difference_equality_l3306_330642

theorem square_difference_equality : 30^2 - 2*(30*5) + 5^2 = 625 := by
  sorry

end square_difference_equality_l3306_330642


namespace binomial_16_13_l3306_330640

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by
  sorry

end binomial_16_13_l3306_330640


namespace perpendicular_bisector_of_chord_l3306_330684

-- Define the given line
def givenLine (x y : ℝ) : Prop := 2 * x + 3 * y + 1 = 0

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Define the perpendicular bisector
def perpendicularBisector (x y : ℝ) : Prop := 3 * x - 2 * y - 7 = 0

-- Theorem statement
theorem perpendicular_bisector_of_chord (A B : ℝ × ℝ) :
  givenLine A.1 A.2 ∧ givenLine B.1 B.2 ∧
  givenCircle A.1 A.2 ∧ givenCircle B.1 B.2 →
  ∃ (M : ℝ × ℝ), perpendicularBisector M.1 M.2 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 :=
by
  sorry

end perpendicular_bisector_of_chord_l3306_330684


namespace complex_conjugate_sum_l3306_330695

theorem complex_conjugate_sum (α β : ℝ) :
  2 * Complex.exp (Complex.I * α) + 2 * Complex.exp (Complex.I * β) = -1/2 + 4/5 * Complex.I →
  2 * Complex.exp (-Complex.I * α) + 2 * Complex.exp (-Complex.I * β) = -1/2 - 4/5 * Complex.I :=
by sorry

end complex_conjugate_sum_l3306_330695


namespace triangle_side_length_l3306_330651

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  c = Real.sqrt 3 →
  b = 2 * Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = 3 :=
by sorry

end triangle_side_length_l3306_330651


namespace distance_between_points_l3306_330699

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -2)
  let p2 : ℝ × ℝ := (8, 8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by sorry

end distance_between_points_l3306_330699


namespace water_formed_equals_three_l3306_330627

-- Define the chemical species
inductive ChemicalSpecies
| NH4Cl
| NaOH
| NaCl
| NH3
| H2O

-- Define the reaction equation
def reactionEquation : List (ChemicalSpecies × Int) :=
  [(ChemicalSpecies.NH4Cl, -1), (ChemicalSpecies.NaOH, -1),
   (ChemicalSpecies.NaCl, 1), (ChemicalSpecies.NH3, 1), (ChemicalSpecies.H2O, 1)]

-- Define the initial amounts of reactants
def initialNH4Cl : ℕ := 3
def initialNaOH : ℕ := 3

-- Function to calculate the moles of water formed
def molesOfWaterFormed (nh4cl : ℕ) (naoh : ℕ) : ℕ :=
  min nh4cl naoh

-- Theorem statement
theorem water_formed_equals_three :
  molesOfWaterFormed initialNH4Cl initialNaOH = 3 := by
  sorry


end water_formed_equals_three_l3306_330627


namespace prob_at_least_one_female_l3306_330674

/-- The probability of selecting at least one female student when choosing 3 students from a group of 3 male and 2 female students. -/
theorem prob_at_least_one_female (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) : 
  n_male = 3 → n_female = 2 → n_select = 3 →
  (1 : ℚ) - (Nat.choose n_male n_select : ℚ) / (Nat.choose (n_male + n_female) n_select : ℚ) = 9/10 :=
by sorry

end prob_at_least_one_female_l3306_330674


namespace satisfying_polynomial_form_l3306_330672

/-- A polynomial that satisfies the given equation for all real numbers a, b, c 
    such that ab + bc + ca = 0 -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a * b + b * c + c * a = 0 → 
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- Theorem stating the form of polynomials satisfying the equation -/
theorem satisfying_polynomial_form (P : ℝ → ℝ) :
  SatisfyingPolynomial P →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^4 + b * x^2 := by
  sorry

end satisfying_polynomial_form_l3306_330672


namespace calculation_proof_l3306_330688

theorem calculation_proof : 20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := by
  sorry

end calculation_proof_l3306_330688


namespace fraction_simplification_l3306_330656

theorem fraction_simplification : (2 + 4) / (1 + 2) = 2 := by
  sorry

end fraction_simplification_l3306_330656


namespace find_r_l3306_330653

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end find_r_l3306_330653


namespace ross_breaths_per_minute_l3306_330660

/-- Calculates the number of breaths per minute given the air inhaled per breath and total air inhaled in 24 hours. -/
def breaths_per_minute (air_per_breath : ℚ) (total_air_24h : ℚ) : ℚ :=
  (total_air_24h / air_per_breath) / (24 * 60)

/-- Theorem stating that Ross takes 17 breaths per minute. -/
theorem ross_breaths_per_minute :
  breaths_per_minute (5/9) 13600 = 17 := by
  sorry

#eval breaths_per_minute (5/9) 13600

end ross_breaths_per_minute_l3306_330660


namespace oblomov_weight_change_l3306_330635

theorem oblomov_weight_change : 
  let spring_factor : ℝ := 0.75
  let summer_factor : ℝ := 1.20
  let autumn_factor : ℝ := 0.90
  let winter_factor : ℝ := 1.20
  spring_factor * summer_factor * autumn_factor * winter_factor < 1 := by
sorry

end oblomov_weight_change_l3306_330635


namespace larger_number_l3306_330649

theorem larger_number (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 := by
  sorry

end larger_number_l3306_330649


namespace square_sum_theorem_l3306_330683

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) :
  x^2 + y^2 = 33 := by
sorry

end square_sum_theorem_l3306_330683


namespace intersection_property_l3306_330677

-- Define the circle
def Circle (a : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a}

-- Define the line
def Line := {p : ℝ × ℝ | p.1 + p.2 = 1}

-- Define the origin
def O : ℝ × ℝ := (0, 0)

theorem intersection_property (a : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Circle a ∩ Line) 
  (hB : B ∈ Circle a ∩ Line) 
  (C : ℝ × ℝ) 
  (hC : C ∈ Circle a) 
  (h_vec : (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) = (C.1 - O.1, C.2 - O.2)) :
  a = 2 :=
sorry

end intersection_property_l3306_330677


namespace least_value_cubic_equation_l3306_330634

theorem least_value_cubic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^3 + 3 * y^2 + 5 * y + 1
  ∃ y_min : ℝ,
    f y_min = 5 ∧
    ∀ y : ℝ, f y = 5 → y ≥ y_min ∧
    y_min = 1 :=
by sorry

end least_value_cubic_equation_l3306_330634


namespace coordinate_sum_with_slope_l3306_330616

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 4,
    if the slope of segment AB is 3/4, then the sum of B's coordinates is 28/3. -/
theorem coordinate_sum_with_slope (x : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 4)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  slope = 3/4 → x + 4 = 28/3 := by
  sorry

end coordinate_sum_with_slope_l3306_330616


namespace test_number_satisfies_conditions_l3306_330682

/-- The test number that satisfies the given conditions -/
def test_number : ℕ := 5

/-- The average score before the current test -/
def previous_average : ℚ := 85

/-- The desired new average score -/
def new_average : ℚ := 88

/-- The score needed on the current test -/
def current_test_score : ℕ := 100

theorem test_number_satisfies_conditions :
  (new_average * test_number : ℚ) - (previous_average * (test_number - 1) : ℚ) = current_test_score := by
  sorry

end test_number_satisfies_conditions_l3306_330682


namespace sum_of_two_numbers_l3306_330698

theorem sum_of_two_numbers (x y : ℝ) : 
  y = 2 * x + 3 ∧ y = 19 → x + y = 27 := by
  sorry

end sum_of_two_numbers_l3306_330698


namespace sufficient_not_necessary_condition_l3306_330678

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, b ≥ 0 → (a + 1)^2 + b ≥ 0) ∧ 
  (∃ a b : ℝ, (a + 1)^2 + b ≥ 0 ∧ b < 0) := by
  sorry

end sufficient_not_necessary_condition_l3306_330678


namespace sheep_wool_production_l3306_330622

/-- Calculates the amount of wool produced per sheep given the total number of sheep,
    payment to the shearer, price per pound of wool, and total profit. -/
def wool_per_sheep (num_sheep : ℕ) (shearer_payment : ℕ) (price_per_pound : ℕ) (profit : ℕ) : ℕ :=
  ((profit + shearer_payment) / price_per_pound) / num_sheep

/-- Proves that given 200 sheep, $2000 paid to shearer, $20 per pound of wool,
    and $38000 profit, each sheep produces 10 pounds of wool. -/
theorem sheep_wool_production :
  wool_per_sheep 200 2000 20 38000 = 10 := by
  sorry

end sheep_wool_production_l3306_330622


namespace quadrilateral_circumscription_l3306_330613

def can_be_circumscribed (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a + c = 180 ∧ b + d = 180

theorem quadrilateral_circumscription :
  (∃ (x : ℝ), can_be_circumscribed (2*x) (4*x) (5*x) (3*x)) ∧
  (∀ (x : ℝ), ¬can_be_circumscribed (5*x) (7*x) (8*x) (9*x)) := by
  sorry

end quadrilateral_circumscription_l3306_330613


namespace min_value_inequality_l3306_330602

theorem min_value_inequality (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 :=
by sorry

end min_value_inequality_l3306_330602


namespace car_cost_difference_l3306_330666

/-- The cost difference between buying and renting a car for a year -/
theorem car_cost_difference (rental_cost : ℕ) (purchase_cost : ℕ) : 
  rental_cost = 20 → purchase_cost = 30 → purchase_cost * 12 - rental_cost * 12 = 120 := by
  sorry

end car_cost_difference_l3306_330666


namespace mistaken_multiplication_l3306_330694

theorem mistaken_multiplication (x : ℤ) : 139 * 43 - 139 * x = 1251 → x = 34 := by
  sorry

end mistaken_multiplication_l3306_330694


namespace square_root_equality_l3306_330639

theorem square_root_equality (x : ℝ) : 
  (Real.sqrt (x + 3) = 3) → ((x + 3)^2 = 81) := by
  sorry

end square_root_equality_l3306_330639


namespace correct_seating_arrangements_l3306_330611

/-- The number of ways to seat 2 students in a row of 5 desks with at least one empty desk between them -/
def seatingArrangements : ℕ := 6

/-- The number of desks in the row -/
def numDesks : ℕ := 5

/-- The number of students to be seated -/
def numStudents : ℕ := 2

/-- The minimum number of empty desks required between the students -/
def minEmptyDesks : ℕ := 1

theorem correct_seating_arrangements :
  seatingArrangements = 
    (numDesks - numStudents - minEmptyDesks + 1) * (numStudents) :=
by sorry

end correct_seating_arrangements_l3306_330611


namespace power_sum_reciprocal_integer_l3306_330606

/-- For a non-zero real number x where x + 1/x is an integer, x^n + 1/x^n is an integer for all natural numbers n. -/
theorem power_sum_reciprocal_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m := by
  sorry

end power_sum_reciprocal_integer_l3306_330606


namespace jenny_peanut_butter_cookies_l3306_330657

theorem jenny_peanut_butter_cookies :
  ∀ (jenny_pb : ℕ) (jenny_cc marcus_pb marcus_lemon : ℕ),
    jenny_cc = 50 →
    marcus_pb = 30 →
    marcus_lemon = 20 →
    jenny_pb + marcus_pb = jenny_cc + marcus_lemon →
    jenny_pb = 40 := by
  sorry

end jenny_peanut_butter_cookies_l3306_330657


namespace rectangle_shorter_side_l3306_330681

theorem rectangle_shorter_side
  (area : ℝ) (perimeter : ℝ)
  (h_area : area = 117)
  (h_perimeter : perimeter = 44)
  : ∃ (length width : ℝ),
    length * width = area ∧
    2 * (length + width) = perimeter ∧
    min length width = 9 :=
by sorry

end rectangle_shorter_side_l3306_330681


namespace mean_twice_mode_iff_x_21_l3306_330663

def is_valid_list (x : ℕ) : Prop :=
  x > 0 ∧ x ≤ 100

def mean_of_list (x : ℕ) : ℚ :=
  (31 + 58 + 98 + 3 * x) / 6

def mode_of_list (x : ℕ) : ℕ := x

theorem mean_twice_mode_iff_x_21 :
  ∀ x : ℕ, is_valid_list x →
    (mean_of_list x = 2 * mode_of_list x) ↔ x = 21 := by
  sorry

end mean_twice_mode_iff_x_21_l3306_330663


namespace lcm_18_24_l3306_330620

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l3306_330620


namespace paco_ate_18_cookies_l3306_330665

/-- The number of cookies Paco ate -/
def cookies_eaten (initial : ℕ) (given : ℕ) : ℕ := given + given

/-- Proof that Paco ate 18 cookies -/
theorem paco_ate_18_cookies (initial : ℕ) (given : ℕ) 
  (h1 : initial = 41)
  (h2 : given = 9) :
  cookies_eaten initial given = 18 := by
  sorry

end paco_ate_18_cookies_l3306_330665


namespace circle_center_l3306_330697

/-- The center of a circle given by the equation 3x^2 - 6x + 3y^2 + 12y - 75 = 0 is (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0) → 
  (∃ r : ℝ, (x - 1)^2 + (y - (-2))^2 = r^2) := by
sorry

end circle_center_l3306_330697


namespace cosine_axis_of_symmetry_l3306_330673

/-- The axis of symmetry for a cosine function translated to the left by π/6 units -/
theorem cosine_axis_of_symmetry (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * (x + π / 6))
  ∃ (x : ℝ), x = -π / 6 + k * π / 2 ∧ 
    (∀ (y : ℝ), f (x - y) = f (x + y)) :=
sorry

end cosine_axis_of_symmetry_l3306_330673


namespace sin_sum_pi_minus_plus_alpha_l3306_330621

theorem sin_sum_pi_minus_plus_alpha (α : ℝ) : 
  Real.sin (π - α) + Real.sin (π + α) = 0 := by
  sorry

end sin_sum_pi_minus_plus_alpha_l3306_330621


namespace expenditure_ratio_l3306_330659

theorem expenditure_ratio (income : ℝ) (h : income > 0) :
  let savings_rate := 0.35
  let income_increase := 0.35
  let savings_increase := 1.0

  let savings_year1 := savings_rate * income
  let expenditure_year1 := income - savings_year1

  let income_year2 := income * (1 + income_increase)
  let savings_year2 := savings_year1 * (1 + savings_increase)
  let expenditure_year2 := income_year2 - savings_year2

  let total_expenditure := expenditure_year1 + expenditure_year2

  (total_expenditure / expenditure_year1) = 2
  := by sorry

end expenditure_ratio_l3306_330659


namespace geometric_progression_ratio_l3306_330615

theorem geometric_progression_ratio (a b c d x y z r : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  a * x * (y - z) ≠ 0 →
  b * y * (z - x) ≠ 0 →
  c * z * (x - y) ≠ 0 →
  d * x * (y - z) ≠ 0 →
  a * x * (y - z) ≠ b * y * (z - x) →
  b * y * (z - x) ≠ c * z * (x - y) →
  c * z * (x - y) ≠ d * x * (y - z) →
  (∃ k : ℝ, k ≠ 0 ∧ 
    b * y * (z - x) = k * (a * x * (y - z)) ∧
    c * z * (x - y) = k * (b * y * (z - x)) ∧
    d * x * (y - z) = k * (c * z * (x - y))) →
  r^3 + r^2 + r + 1 = 0 :=
by sorry

end geometric_progression_ratio_l3306_330615


namespace bottle_cap_distribution_l3306_330686

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 35 → num_groups = 7 → caps_per_group = total_caps / num_groups → caps_per_group = 5 := by
  sorry

end bottle_cap_distribution_l3306_330686


namespace extreme_values_and_increasing_condition_l3306_330604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem extreme_values_and_increasing_condition :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → f (-1/2) x ≤ f (-1/2) x₀) ∧
  f (-1/2) x₀ = 0 ∧
  (∀ (y : ℝ), y > 0 → ∃ (z : ℝ), z > y ∧ f (-1/2) z > f (-1/2) y) ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), 0 < x ∧ x < y → f a x < f a y) ↔ a ≥ 1 / (2 * Real.exp 2)) :=
by sorry

end extreme_values_and_increasing_condition_l3306_330604


namespace trajectory_of_A_l3306_330644

-- Define the points B and C
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 0)

-- Define the perimeter of triangle ABC
def perimeter : ℝ := 16

-- Define the trajectory of point A
def trajectory (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1 ∧ y ≠ 0

-- Theorem statement
theorem trajectory_of_A (A : ℝ × ℝ) :
  (dist A B + dist A C + dist B C = perimeter) →
  trajectory A.1 A.2 :=
by sorry


end trajectory_of_A_l3306_330644


namespace mans_rate_l3306_330645

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 12) :
  (speed_with_stream + speed_against_stream) / 2 = 19 := by
  sorry

end mans_rate_l3306_330645


namespace smallest_c_value_l3306_330658

/-- The smallest possible value of c in a sequence satisfying specific conditions -/
theorem smallest_c_value : ∃ (a b c : ℤ),
  (a < b ∧ b < c) ∧                    -- a < b < c are integers
  (2 * b = a + c) ∧                    -- arithmetic progression
  (a * a = c * b) ∧                    -- geometric progression
  (∃ (m n p : ℤ), a = 5 * m ∧ b = 5 * n ∧ c = 5 * p) ∧  -- multiples of 5
  (0 < a ∧ 0 < b ∧ 0 < c) ∧            -- all numbers are positive
  (c = 20) ∧                           -- c equals 20
  (∀ (a' b' c' : ℤ),                   -- for any other triple satisfying the conditions
    (a' < b' ∧ b' < c') →
    (2 * b' = a' + c') →
    (a' * a' = c' * b') →
    (∃ (m' n' p' : ℤ), a' = 5 * m' ∧ b' = 5 * n' ∧ c' = 5 * p') →
    (0 < a' ∧ 0 < b' ∧ 0 < c') →
    (c ≤ c')) :=                       -- c is the smallest possible value
by sorry


end smallest_c_value_l3306_330658


namespace rectangle_ordering_l3306_330633

-- Define a rectangle in a Cartesian plane
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

-- Define the "preferable" relation
def preferable (a b : Rectangle) : Prop :=
  (a.x_max ≤ b.x_min) ∨ (a.y_max ≤ b.y_min)

-- Main theorem
theorem rectangle_ordering {n : ℕ} (rectangles : Fin n → Rectangle) 
  (h_nonoverlap : ∀ i j, i ≠ j → 
    (rectangles i).x_max ≤ (rectangles j).x_min ∨
    (rectangles j).x_max ≤ (rectangles i).x_min ∨
    (rectangles i).y_max ≤ (rectangles j).y_min ∨
    (rectangles j).y_max ≤ (rectangles i).y_min) :
  ∃ (σ : Equiv.Perm (Fin n)), ∀ i j, i < j → 
    preferable (rectangles (σ i)) (rectangles (σ j)) := by
  sorry

end rectangle_ordering_l3306_330633


namespace functional_equation_solution_l3306_330664

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the only function satisfying the functional equation is f(z) = 1 - z²/2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ z : ℝ, f z = 1 - z^2 / 2 := by
  sorry

end functional_equation_solution_l3306_330664


namespace geometric_sequence_sum_l3306_330612

theorem geometric_sequence_sum (a b c q : ℝ) (h_seq : (a + b + c) * q = b + c - a ∧
                                                    (b + c - a) * q = c + a - b ∧
                                                    (c + a - b) * q = a + b - c) :
  q^3 + q^2 + q = 1 := by
sorry

end geometric_sequence_sum_l3306_330612


namespace polynomial_simplification_l3306_330692

theorem polynomial_simplification (x : ℝ) :
  (5 * x^10 + 8 * x^9 + 3 * x^8) + (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9) =
  2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9 := by
  sorry

end polynomial_simplification_l3306_330692


namespace complex_sum_theorem_l3306_330601

theorem complex_sum_theorem (p r s u v x y : ℝ) : 
  let q : ℝ := 4
  let sum_real : ℝ := p + r + u + x
  let sum_imag : ℝ := q + s + v + y
  u = -p - r - x →
  sum_real = 0 →
  sum_imag = 7 →
  s + v + y = 3 := by sorry

end complex_sum_theorem_l3306_330601


namespace non_coincident_terminal_sides_l3306_330638

def has_coincident_terminal_sides (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

theorem non_coincident_terminal_sides :
  ¬ has_coincident_terminal_sides 1050 (-300) :=
by sorry

end non_coincident_terminal_sides_l3306_330638


namespace stormi_bicycle_savings_l3306_330636

/-- The amount of additional money Stormi needs to afford a bicycle -/
def additional_money_needed (num_cars : ℕ) (price_per_car : ℕ) (num_lawns : ℕ) (price_per_lawn : ℕ) (bicycle_cost : ℕ) : ℕ :=
  bicycle_cost - (num_cars * price_per_car + num_lawns * price_per_lawn)

/-- Theorem stating that Stormi needs $24 more to afford the bicycle -/
theorem stormi_bicycle_savings : additional_money_needed 3 10 2 13 80 = 24 := by
  sorry

end stormi_bicycle_savings_l3306_330636


namespace tina_book_expense_l3306_330617

def savings_june : ℤ := 27
def savings_july : ℤ := 14
def savings_august : ℤ := 21
def spent_on_shoes : ℤ := 17
def money_left : ℤ := 40

theorem tina_book_expense :
  ∃ (book_expense : ℤ),
    savings_june + savings_july + savings_august - book_expense - spent_on_shoes = money_left ∧
    book_expense = 5 := by
  sorry

end tina_book_expense_l3306_330617


namespace baseball_team_groups_l3306_330632

/-- The number of groups formed from new and returning players -/
def number_of_groups (new_players returning_players players_per_group : ℕ) : ℕ :=
  (new_players + returning_players) / players_per_group

/-- Theorem stating that the number of groups is 9 given the specific conditions -/
theorem baseball_team_groups : number_of_groups 48 6 6 = 9 := by
  sorry

end baseball_team_groups_l3306_330632


namespace difference_of_difference_eq_intersection_l3306_330691

-- Define the difference of two sets
def set_difference (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem difference_of_difference_eq_intersection
  {α : Type*} (A B : Set α) (hA : A.Nonempty) (hB : B.Nonempty) :
  A \ (A \ B) = A ∩ B :=
sorry

end difference_of_difference_eq_intersection_l3306_330691


namespace unique_m_for_direct_proportion_l3306_330603

/-- A function f(x) is a direct proportion function if it can be written as f(x) = kx for some non-zero constant k. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m+1)x + m^2 - 1 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 1) * x + m^2 - 1

/-- Theorem: The only value of m that makes f(m) a direct proportion function is 1 -/
theorem unique_m_for_direct_proportion :
  ∃! m : ℝ, IsDirectProportion (f m) ∧ m = 1 := by
  sorry

end unique_m_for_direct_proportion_l3306_330603


namespace reciprocal_sum_l3306_330637

theorem reciprocal_sum (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 56) :
  1 / x + 1 / y = 15 / 56 := by
  sorry

end reciprocal_sum_l3306_330637


namespace floor_ceiling_sum_l3306_330667

theorem floor_ceiling_sum : ⌊(-2.54 : ℝ)⌋ + ⌈(25.4 : ℝ)⌉ = 23 := by sorry

end floor_ceiling_sum_l3306_330667


namespace absolute_value_sum_difference_l3306_330608

theorem absolute_value_sum_difference (x y : ℚ) 
  (hx : |x| = 9) (hy : |y| = 5) : 
  ((x < 0 ∧ y > 0) → x + y = -4) ∧
  (|x + y| = x + y → (x - y = 4 ∨ x - y = 14)) := by
  sorry

end absolute_value_sum_difference_l3306_330608


namespace new_average_rent_l3306_330654

/-- Calculates the new average rent per person after one person's rent is increased -/
theorem new_average_rent (num_friends : ℕ) (initial_average : ℚ) (increased_rent : ℚ) (increase_percentage : ℚ) : 
  num_friends = 4 →
  initial_average = 800 →
  increased_rent = 1250 →
  increase_percentage = 16 / 100 →
  (num_friends * initial_average - increased_rent + increased_rent * (1 + increase_percentage)) / num_friends = 850 :=
by sorry

end new_average_rent_l3306_330654


namespace bacon_tomato_difference_l3306_330628

theorem bacon_tomato_difference (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 228)
  (h2 : bacon = 337)
  (h3 : tomatoes = 23) :
  bacon - tomatoes = 314 := by
  sorry

end bacon_tomato_difference_l3306_330628


namespace triangle_ratio_equality_l3306_330689

/-- Given a triangle with side length a, height h_a corresponding to side a,
    inradius r, and semiperimeter p, prove that (2p / a) = (h_a / r) -/
theorem triangle_ratio_equality (a h_a r p : ℝ) (h_positive : a > 0 ∧ h_a > 0 ∧ r > 0 ∧ p > 0)
  (h_area_inradius : p * r = (1/2) * a * h_a) : 
  (2 * p / a) = (h_a / r) := by
  sorry

end triangle_ratio_equality_l3306_330689


namespace inequality_implies_range_l3306_330607

theorem inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → 4^x - 2^(x+1) - a ≤ 0) →
  a ≥ 8 := by
sorry

end inequality_implies_range_l3306_330607


namespace not_ellipse_for_certain_m_l3306_330669

/-- The equation of the curve -/
def curve_equation (m x y : ℝ) : Prop :=
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m)

/-- Definition of an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  1 < m ∧ m < 3 ∧ m ≠ 2

/-- The theorem to be proved -/
theorem not_ellipse_for_certain_m :
  ∀ m : ℝ, (m ≤ 1 ∨ m = 2 ∨ m ≥ 3) →
    ¬(is_ellipse m) :=
sorry

end not_ellipse_for_certain_m_l3306_330669


namespace dvd_rental_cost_l3306_330670

/-- The cost per DVD given the total cost and number of DVDs rented --/
def cost_per_dvd (total_cost : ℚ) (num_dvds : ℕ) : ℚ :=
  total_cost / num_dvds

/-- Theorem stating that the cost per DVD is $1.20 given the problem conditions --/
theorem dvd_rental_cost : 
  let total_cost : ℚ := 48/10
  let num_dvds : ℕ := 4
  cost_per_dvd total_cost num_dvds = 12/10 := by
  sorry

end dvd_rental_cost_l3306_330670


namespace least_n_factorial_divisible_by_32_l3306_330685

theorem least_n_factorial_divisible_by_32 :
  ∀ n : ℕ, n > 0 → (n.factorial % 32 = 0) → n ≥ 8 :=
by
  sorry

end least_n_factorial_divisible_by_32_l3306_330685


namespace cube_root_of_four_condition_l3306_330661

theorem cube_root_of_four_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) := by
  sorry

end cube_root_of_four_condition_l3306_330661


namespace sum_of_solutions_quadratic_l3306_330671

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -24
  let b : ℝ := 72
  let c : ℝ := -120
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = 3 :=
by sorry

end sum_of_solutions_quadratic_l3306_330671


namespace second_subject_grade_l3306_330675

/-- Represents the grade of a student in a subject as a percentage -/
def Grade := Fin 101

/-- Calculates the average of three grades -/
def average (g1 g2 g3 : Grade) : ℚ :=
  (g1.val + g2.val + g3.val) / 3

theorem second_subject_grade 
  (g1 g3 : Grade) 
  (h1 : g1.val = 60) 
  (h3 : g3.val = 80) :
  ∃ (g2 : Grade), average g1 g2 g3 = 70 ∧ g2.val = 70 := by
  sorry

end second_subject_grade_l3306_330675


namespace zoe_bought_eight_roses_l3306_330618

/-- Calculates the number of roses bought given the total spent, cost per flower, and number of daisies. -/
def roses_bought (total_spent : ℕ) (cost_per_flower : ℕ) (num_daisies : ℕ) : ℕ :=
  (total_spent - cost_per_flower * num_daisies) / cost_per_flower

/-- Proves that Zoe bought 8 roses given the problem conditions. -/
theorem zoe_bought_eight_roses (total_spent : ℕ) (cost_per_flower : ℕ) (num_daisies : ℕ) 
    (h1 : total_spent = 30)
    (h2 : cost_per_flower = 3)
    (h3 : num_daisies = 2) : 
  roses_bought total_spent cost_per_flower num_daisies = 8 := by
  sorry

#eval roses_bought 30 3 2  -- Should output 8

end zoe_bought_eight_roses_l3306_330618


namespace xiaoSiScore_l3306_330668

/-- Represents the correctness of an answer -/
inductive Correctness
| Correct
| Incorrect

/-- Represents a single question in the test -/
structure Question where
  number : Nat
  points : Nat
  correctness : Correctness

/-- Calculates the score for a single question -/
def scoreQuestion (q : Question) : Nat :=
  match q.correctness with
  | Correctness.Correct => q.points
  | Correctness.Incorrect => 0

/-- Xiao Si's test answers -/
def xiaoSiAnswers : List Question :=
  [
    { number := 1, points := 20, correctness := Correctness.Correct },
    { number := 2, points := 20, correctness := Correctness.Incorrect },
    { number := 3, points := 20, correctness := Correctness.Incorrect },
    { number := 4, points := 20, correctness := Correctness.Incorrect },
    { number := 5, points := 20, correctness := Correctness.Incorrect }
  ]

/-- Calculates the total score for the test -/
def calculateTotalScore (answers : List Question) : Nat :=
  answers.foldl (fun acc q => acc + scoreQuestion q) 0

/-- Theorem stating that Xiao Si's total score is 20 points -/
theorem xiaoSiScore : calculateTotalScore xiaoSiAnswers = 20 := by
  sorry


end xiaoSiScore_l3306_330668


namespace other_root_of_complex_equation_l3306_330679

theorem other_root_of_complex_equation (z : ℂ) :
  z ^ 2 = -72 + 27 * I →
  (-6 + 3 * I) ^ 2 = -72 + 27 * I →
  ∃ w : ℂ, w ^ 2 = -72 + 27 * I ∧ w ≠ -6 + 3 * I ∧ w = 6 - 3 * I :=
by sorry

end other_root_of_complex_equation_l3306_330679


namespace elevator_weight_problem_l3306_330655

/-- Given 6 people in an elevator with an average weight of 156 lbs,
    if a 7th person enters and the new average weight becomes 151 lbs,
    then the weight of the 7th person is 121 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight new_avg_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_avg_weight = 151 →
  (initial_people * initial_avg_weight + (initial_people + 1) * new_avg_weight - initial_people * new_avg_weight) = 121 :=
by sorry

end elevator_weight_problem_l3306_330655


namespace man_rowing_speed_l3306_330687

/-- The speed of a man rowing in still water, given his speeds with wind influence -/
theorem man_rowing_speed 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (wind_speed : ℝ) 
  (h1 : upstream_speed = 25) 
  (h2 : downstream_speed = 65) 
  (h3 : wind_speed = 5) : 
  (upstream_speed + downstream_speed) / 2 = 45 := by
sorry


end man_rowing_speed_l3306_330687


namespace balls_in_boxes_l3306_330624

/-- The number of ways to place 3 different balls into 4 boxes. -/
def total_ways : ℕ := 4^3

/-- The number of ways to place 3 different balls into the first 3 boxes. -/
def ways_without_fourth : ℕ := 3^3

/-- The number of ways to place 3 different balls into 4 boxes,
    such that the 4th box contains at least one ball. -/
def ways_with_fourth : ℕ := total_ways - ways_without_fourth

theorem balls_in_boxes : ways_with_fourth = 37 := by
  sorry

end balls_in_boxes_l3306_330624


namespace cube_plus_n_minus_two_power_of_two_l3306_330619

theorem cube_plus_n_minus_two_power_of_two (n : ℕ+) :
  (∃ k : ℕ, (n : ℕ)^3 + n - 2 = 2^k) ↔ n = 2 ∨ n = 5 := by
  sorry

end cube_plus_n_minus_two_power_of_two_l3306_330619
