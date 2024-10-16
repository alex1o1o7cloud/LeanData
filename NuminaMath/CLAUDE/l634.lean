import Mathlib

namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l634_63409

theorem right_angled_triangle_set :
  ∀ (a b c : ℝ),
  (a = 3 ∧ b = 4 ∧ c = 5) →
  a^2 + b^2 = c^2 ∧
  ¬(1^2 + 2^2 = 3^2) ∧
  ¬(5^2 + 12^2 = 14^2) ∧
  ¬((Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2) :=
by
  sorry

#check right_angled_triangle_set

end NUMINAMATH_CALUDE_right_angled_triangle_set_l634_63409


namespace NUMINAMATH_CALUDE_chess_club_team_probability_l634_63457

def total_members : ℕ := 20
def num_boys : ℕ := 12
def num_girls : ℕ := 8
def team_size : ℕ := 4

theorem chess_club_team_probability :
  let total_combinations := Nat.choose total_members team_size
  let valid_combinations := 
    Nat.choose num_boys 2 * Nat.choose num_girls 2 + 
    Nat.choose num_boys 3 * Nat.choose num_girls 1 + 
    Nat.choose num_boys 4 * Nat.choose num_girls 0
  (valid_combinations : ℚ) / total_combinations = 4103 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_team_probability_l634_63457


namespace NUMINAMATH_CALUDE_angle_D_value_l634_63471

-- Define the angles as real numbers
variable (A B C D E : ℝ)

-- State the given conditions
axiom angle_sum : A + B = 180
axiom angle_C_eq_D : C = D
axiom angle_A_value : A = 50
axiom angle_E_value : E = 60
axiom triangle1_sum : A + B + E = 180
axiom triangle2_sum : B + C + D = 180

-- State the theorem to be proved
theorem angle_D_value : D = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l634_63471


namespace NUMINAMATH_CALUDE_three_valid_k_values_l634_63498

/-- The sum of k consecutive natural numbers starting from n -/
def consecutiveSum (n k : ℕ) : ℕ := k * (2 * n + k - 1) / 2

/-- Predicate to check if k is a valid solution -/
def isValidK (k : ℕ) : Prop :=
  k > 1 ∧ ∃ n : ℕ, consecutiveSum n k = 2000

theorem three_valid_k_values :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ k, k ∈ s ↔ isValidK k :=
sorry

end NUMINAMATH_CALUDE_three_valid_k_values_l634_63498


namespace NUMINAMATH_CALUDE_ceiling_times_x_204_l634_63479

theorem ceiling_times_x_204 (x : ℝ) : ⌈x⌉ * x = 204 → x = 13.6 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_times_x_204_l634_63479


namespace NUMINAMATH_CALUDE_orange_division_l634_63472

theorem orange_division (oranges : ℕ) (friends : ℕ) (pieces_per_friend : ℕ) 
  (h1 : oranges = 80) 
  (h2 : friends = 200) 
  (h3 : pieces_per_friend = 4) : 
  (friends * pieces_per_friend) / oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_division_l634_63472


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_calculate_expression_3_calculate_expression_4_l634_63407

-- Part 1
theorem simplify_expression_1 (x : ℝ) : 2 * x^2 + 3 * x - 3 * x^2 + 4 * x = -x^2 + 7 * x := by sorry

-- Part 2
theorem simplify_expression_2 (a : ℝ) : 3 * a - 5 * (a + 1) + 4 * (2 + a) = 2 * a + 3 := by sorry

-- Part 3
theorem calculate_expression_3 : (-2/3 + 5/8 - 1/6) * (-24) = 5 := by sorry

-- Part 4
theorem calculate_expression_4 : -(1^4) + 16 / ((-2)^3) * |(-3) - 1| = -9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_calculate_expression_3_calculate_expression_4_l634_63407


namespace NUMINAMATH_CALUDE_area_enclosed_is_nine_halves_l634_63442

-- Define the constant term a
def a : ℝ := 3

-- Define the functions for the line and curve
def f (x : ℝ) : ℝ := a * x
def g (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem area_enclosed_is_nine_halves :
  ∫ x in (0)..(3), (f x - g x) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_is_nine_halves_l634_63442


namespace NUMINAMATH_CALUDE_distributive_property_l634_63495

theorem distributive_property (a : ℝ) : 2 * (a - 1) = 2 * a - 2 := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l634_63495


namespace NUMINAMATH_CALUDE_paper_stack_height_l634_63490

theorem paper_stack_height (sheets : ℕ) (height : ℝ) : 
  (400 : ℝ) / 4 = sheets / height → sheets = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_stack_height_l634_63490


namespace NUMINAMATH_CALUDE_scissors_count_l634_63493

/-- The number of scissors initially in the drawer -/
def initial_scissors : ℕ := 39

/-- The number of scissors Dan added to the drawer -/
def added_scissors : ℕ := 13

/-- The total number of scissors after Dan's addition -/
def total_scissors : ℕ := initial_scissors + added_scissors

/-- Theorem stating that the total number of scissors is 52 -/
theorem scissors_count : total_scissors = 52 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l634_63493


namespace NUMINAMATH_CALUDE_pie_crust_flour_usage_l634_63414

/-- Given that 40 pie crusts each use 1/8 cup of flour, 
    prove that 25 larger pie crusts using the same total amount of flour 
    will each use 1/5 cup of flour. -/
theorem pie_crust_flour_usage 
  (initial_crusts : ℕ) 
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ) 
  (total_flour : ℚ) :
  initial_crusts = 40 →
  initial_flour_per_crust = 1/8 →
  new_crusts = 25 →
  total_flour = initial_crusts * initial_flour_per_crust →
  total_flour = new_crusts * (1/5 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_usage_l634_63414


namespace NUMINAMATH_CALUDE_ratio_expression_value_l634_63456

theorem ratio_expression_value (P Q R : ℚ) (h : P / Q = 3 / 2 ∧ Q / R = 2 / 6) :
  (4 * P + 3 * Q) / (5 * R - 2 * P) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l634_63456


namespace NUMINAMATH_CALUDE_square_numbers_existence_l634_63461

theorem square_numbers_existence (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ (q1 q2 : ℕ), q1.Prime ∧ q2.Prime ∧ q1 ≠ q2 ∧
    ¬(p^2 ∣ (q1^(p-1) - 1)) ∧ ¬(p^2 ∣ (q2^(p-1) - 1)) := by
  sorry

end NUMINAMATH_CALUDE_square_numbers_existence_l634_63461


namespace NUMINAMATH_CALUDE_absolute_value_equation_l634_63468

theorem absolute_value_equation : ∃! x : ℝ, |x - 30| + |x - 24| = |3*x - 72| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l634_63468


namespace NUMINAMATH_CALUDE_arithmetic_sum_example_l634_63452

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℤ) (d : ℤ) (l : ℤ) : ℤ :=
  let n : ℤ := (l - a) / d + 1
  n * (a + l) / 2

/-- Theorem: The sum of the arithmetic sequence with first term -41,
    common difference 3, and last term 7 is -289 -/
theorem arithmetic_sum_example : arithmetic_sum (-41) 3 7 = -289 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_example_l634_63452


namespace NUMINAMATH_CALUDE_angle_problem_l634_63404

theorem angle_problem (A B : ℝ) (h1 : A = 4 * B) (h2 : 90 - B = 4 * (90 - A)) : B = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l634_63404


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l634_63423

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a - b < a^2 - b^2) ∧
  (∃ a b : ℝ, a - b < a^2 - b^2 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l634_63423


namespace NUMINAMATH_CALUDE_g_of_6_l634_63445

def g (x : ℝ) : ℝ := 2*x^4 - 13*x^3 + 28*x^2 - 32*x - 48

theorem g_of_6 : g 6 = 552 := by
  sorry

end NUMINAMATH_CALUDE_g_of_6_l634_63445


namespace NUMINAMATH_CALUDE_tailor_buttons_l634_63473

theorem tailor_buttons (green : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = green + 10)
  (h2 : blue = green - 5)
  (h3 : green + yellow + blue = 275) :
  green = 90 := by
sorry

end NUMINAMATH_CALUDE_tailor_buttons_l634_63473


namespace NUMINAMATH_CALUDE_dumbbell_weight_l634_63491

/-- Given information about exercise bands and total weight, calculate the weight of the dumbbell. -/
theorem dumbbell_weight 
  (num_bands : ℕ) 
  (resistance_per_band : ℕ) 
  (total_weight : ℕ) 
  (h1 : num_bands = 2)
  (h2 : resistance_per_band = 5)
  (h3 : total_weight = 30) :
  total_weight - (num_bands * resistance_per_band) = 20 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_weight_l634_63491


namespace NUMINAMATH_CALUDE_median_triangle_inequalities_l634_63432

-- Define a structure for a triangle with angles
structure Triangle where
  α : Real
  β : Real
  γ : Real

-- Define a structure for a triangle formed from medians
structure MedianTriangle where
  α_m : Real
  β_m : Real
  γ_m : Real

-- Main theorem
theorem median_triangle_inequalities (T : Triangle) (M : MedianTriangle)
  (h1 : T.α > T.β)
  (h2 : T.β > T.γ)
  : T.α > M.α_m ∧
    T.α > M.β_m ∧
    M.γ_m > T.β ∧
    T.β > M.α_m ∧
    M.β_m > T.γ ∧
    M.γ_m > T.γ := by
  sorry

end NUMINAMATH_CALUDE_median_triangle_inequalities_l634_63432


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l634_63486

-- Define the quadratic equation
def has_real_roots (m : ℕ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

-- Define the original proposition
def original_prop (m : ℕ) : Prop := m > 0 → has_real_roots m

-- Define the contrapositive
def contrapositive (m : ℕ) : Prop := ¬(has_real_roots m) → m ≤ 0

-- Theorem statement
theorem contrapositive_equivalence :
  ∀ m : ℕ, m > 0 → (original_prop m ↔ contrapositive m) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l634_63486


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_implies_n_binomial_remainder_l634_63465

/-- The binomial expansion (x + 3x^2)^n -/
def binomial (x : ℝ) (n : ℕ) := (x + 3 * x^2)^n

/-- The sum of binomial coefficients of (x + 3x^2)^n -/
def sumBinomialCoefficients (n : ℕ) := 2^n

theorem binomial_coefficient_sum_implies_n (n : ℕ) :
  sumBinomialCoefficients n = 128 → n = 7 := by sorry

theorem binomial_remainder (n : ℕ) :
  (binomial 3 2016) % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_implies_n_binomial_remainder_l634_63465


namespace NUMINAMATH_CALUDE_sum_product_difference_l634_63438

theorem sum_product_difference (x y : ℝ) : 
  x + y = 24 → x * y = 23 → |x - y| = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_product_difference_l634_63438


namespace NUMINAMATH_CALUDE_amendment_effects_l634_63464

-- Define the administrative actions included in the amendment
def administrative_actions : Set String := 
  {"abuse of administrative power", "illegal fundraising", "apportionment of expenses", "failure to pay benefits"}

-- Define the amendment to the Administrative Litigation Law
def administrative_litigation_amendment (actions : Set String) : Prop :=
  ∀ action ∈ actions, action ∈ administrative_actions

-- Define the concept of standardizing government power exercise
def standardizes_government_power (amendment : Set String → Prop) : Prop :=
  amendment administrative_actions → 
    ∃ standard : String, standard = "improved government power exercise"

-- Define the concept of protecting citizens' rights
def protects_citizens_rights (amendment : Set String → Prop) : Prop :=
  amendment administrative_actions → 
    ∃ protection : String, protection = "better protection of citizens' rights"

-- Theorem statement
theorem amendment_effects 
  (h : administrative_litigation_amendment administrative_actions) :
  standardizes_government_power administrative_litigation_amendment ∧ 
  protects_citizens_rights administrative_litigation_amendment :=
by sorry

end NUMINAMATH_CALUDE_amendment_effects_l634_63464


namespace NUMINAMATH_CALUDE_mary_eggs_problem_l634_63492

theorem mary_eggs_problem (initial_eggs found_eggs final_eggs : ℕ) 
  (h1 : found_eggs = 4)
  (h2 : final_eggs = 31)
  (h3 : final_eggs = initial_eggs + found_eggs) :
  initial_eggs = 27 := by
  sorry

end NUMINAMATH_CALUDE_mary_eggs_problem_l634_63492


namespace NUMINAMATH_CALUDE_greatest_two_digit_product_12_proof_l634_63476

/-- The greatest two-digit whole number whose digits have a product of 12 -/
def greatest_two_digit_product_12 : ℕ := 62

/-- Predicate to check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Function to get the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Function to get the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

theorem greatest_two_digit_product_12_proof :
  (is_two_digit greatest_two_digit_product_12) ∧
  (tens_digit greatest_two_digit_product_12 * ones_digit greatest_two_digit_product_12 = 12) ∧
  (∀ m : ℕ, is_two_digit m → 
    tens_digit m * ones_digit m = 12 → 
    m ≤ greatest_two_digit_product_12) :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_product_12_proof_l634_63476


namespace NUMINAMATH_CALUDE_no_integer_solution_for_rectangle_l634_63497

theorem no_integer_solution_for_rectangle : 
  ¬ ∃ (w l : ℕ), w > 0 ∧ l > 0 ∧ w * l = 24 ∧ (w = l ∨ w = 2 * l) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_rectangle_l634_63497


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l634_63413

theorem multiple_with_binary_digits (n : ℕ+) : 
  ∃ m : ℕ, 
    (n : ℕ) ∣ m ∧ 
    (Nat.digits 2 m).length ≤ n ∧ 
    ∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l634_63413


namespace NUMINAMATH_CALUDE_fourth_coaster_speed_l634_63430

/-- Given 5 rollercoasters with known speeds for 4 of them and a known average speed for all 5,
    prove that the speed of the unknown coaster is equal to the total speed (based on the average)
    minus the sum of the known speeds. -/
theorem fourth_coaster_speed
  (speed1 speed2 speed3 speed5 : ℝ)
  (average_speed : ℝ)
  (h1 : speed1 = 50)
  (h2 : speed2 = 62)
  (h3 : speed3 = 73)
  (h5 : speed5 = 40)
  (h_avg : average_speed = 59)
  : ∃ speed4 : ℝ,
    speed4 = 5 * average_speed - (speed1 + speed2 + speed3 + speed5) :=
by sorry

end NUMINAMATH_CALUDE_fourth_coaster_speed_l634_63430


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l634_63422

def f (x : ℝ) := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y →
    (3 * x^2 + 1 = 4) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l634_63422


namespace NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l634_63484

-- Define a harmonic progression
def IsHarmonicProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℚ, ∀ k : ℕ, k > 0 → (1 : ℚ) / a (k + 1) - (1 : ℚ) / a k = d

-- Part (a)
theorem exists_finite_harmonic_progression (N : ℕ) :
  ∃ a : ℕ → ℕ, (∀ k : ℕ, k < N → a k < a (k + 1)) ∧ IsHarmonicProgression a :=
sorry

-- Part (b)
theorem no_infinite_harmonic_progression :
  ¬ ∃ a : ℕ → ℕ, (∀ k : ℕ, a k < a (k + 1)) ∧ IsHarmonicProgression a :=
sorry

end NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l634_63484


namespace NUMINAMATH_CALUDE_sum_of_different_geometric_not_geometric_l634_63459

/-- Given two geometric sequences with different common ratios, their sum sequence is not a geometric sequence -/
theorem sum_of_different_geometric_not_geometric
  {α : Type*} [Field α]
  (a b : ℕ → α)
  (p q : α)
  (hp : p ≠ q)
  (ha : ∀ n, a (n + 1) = p * a n)
  (hb : ∀ n, b (n + 1) = q * b n)
  : ¬ (∃ r : α, ∀ n, (a (n + 1) + b (n + 1)) = r * (a n + b n)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_different_geometric_not_geometric_l634_63459


namespace NUMINAMATH_CALUDE_min_value_expression_l634_63449

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  4 * a^2 + 4 * b^2 + 1 / (a + b)^2 ≥ 2 ∧
  (4 * a^2 + 4 * b^2 + 1 / (a + b)^2 = 2 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l634_63449


namespace NUMINAMATH_CALUDE_wire_length_from_sphere_l634_63405

/-- The length of a wire drawn from a metallic sphere -/
theorem wire_length_from_sphere (r_sphere r_wire : ℝ) (h : r_sphere = 24 ∧ r_wire = 0.16) :
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let l_wire := v_sphere / (Real.pi * r_wire ^ 2)
  l_wire = 675000 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_from_sphere_l634_63405


namespace NUMINAMATH_CALUDE_integer_double_root_theorem_l634_63469

/-- A polynomial with integer coefficients of the form x^4 + b_3x^3 + b_2x^2 + b_1x + 48 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 48

/-- The set of possible integer double roots -/
def PossibleRoots : Set ℤ := {-4, -2, -1, 1, 2, 4}

theorem integer_double_root_theorem (b₃ b₂ b₁ s : ℤ) :
  (∃ k : ℤ, IntPolynomial b₃ b₂ b₁ x = (x - s)^2 * (x^2 + kx + m)) →
  s ∈ PossibleRoots := by
  sorry

end NUMINAMATH_CALUDE_integer_double_root_theorem_l634_63469


namespace NUMINAMATH_CALUDE_jerry_tips_problem_l634_63419

/-- The amount Jerry needs to earn on the fifth night to achieve an average of $50 per night -/
theorem jerry_tips_problem (
  days_per_week : ℕ)
  (target_average : ℝ)
  (past_earnings : List ℝ)
  (h1 : days_per_week = 5)
  (h2 : target_average = 50)
  (h3 : past_earnings = [20, 60, 15, 40]) :
  target_average * days_per_week - past_earnings.sum = 115 := by
  sorry

end NUMINAMATH_CALUDE_jerry_tips_problem_l634_63419


namespace NUMINAMATH_CALUDE_min_throws_correct_l634_63403

/-- The probability of hitting the target on a single throw -/
def p : ℝ := 0.6

/-- The desired minimum probability of hitting the target at least once -/
def min_prob : ℝ := 0.9

/-- The function that calculates the probability of hitting the target at least once in n throws -/
def prob_hit_at_least_once (n : ℕ) : ℝ := 1 - (1 - p)^n

/-- The minimum number of throws needed to exceed the desired probability -/
def min_throws : ℕ := 3

theorem min_throws_correct :
  (∀ k < min_throws, prob_hit_at_least_once k ≤ min_prob) ∧
  prob_hit_at_least_once min_throws > min_prob :=
sorry

end NUMINAMATH_CALUDE_min_throws_correct_l634_63403


namespace NUMINAMATH_CALUDE_algebraic_expressions_simplification_l634_63478

theorem algebraic_expressions_simplification (x y m a b c : ℝ) :
  (4 * y * (-2 * x * y^2) = -8 * x * y^3) ∧
  ((-5/2 * x^2) * (-4 * x) = 10 * x^3) ∧
  ((3 * m^2) * (-2 * m^3)^2 = 12 * m^8) ∧
  ((-a * b^2 * c^3)^2 * (-a^2 * b)^3 = -a^8 * b^7 * c^6) := by
sorry


end NUMINAMATH_CALUDE_algebraic_expressions_simplification_l634_63478


namespace NUMINAMATH_CALUDE_complex_equation_solution_l634_63480

theorem complex_equation_solution (z : ℂ) : 
  z / (1 - Complex.I) = Complex.I^2016 + Complex.I^2017 → z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l634_63480


namespace NUMINAMATH_CALUDE_min_value_fraction_l634_63408

theorem min_value_fraction (x : ℝ) (h : x > 7) :
  (x^2 + 49) / (x - 7) ≥ 7 + 14 * Real.sqrt 2 ∧
  ∃ y > 7, (y^2 + 49) / (y - 7) = 7 + 14 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l634_63408


namespace NUMINAMATH_CALUDE_prob_even_sum_is_seven_sixteenths_l634_63463

/-- Represents the dartboard with inner and outer circles and point values -/
structure Dartboard where
  inner_radius : ℝ
  outer_radius : ℝ
  inner_values : Fin 3 → ℕ
  outer_values : Fin 3 → ℕ

/-- Calculates the probability of getting an even sum with two darts -/
def prob_even_sum (d : Dartboard) : ℚ :=
  sorry

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard where
  inner_radius := 4
  outer_radius := 8
  inner_values := ![3, 5, 5]
  outer_values := ![4, 3, 3]

theorem prob_even_sum_is_seven_sixteenths :
  prob_even_sum problem_dartboard = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_seven_sixteenths_l634_63463


namespace NUMINAMATH_CALUDE_line_equation_through_points_l634_63406

/-- The equation of a line passing through two points. -/
def line_equation (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  λ x y => y - y1 = m * (x - x1)

/-- Theorem: The equation of the line passing through P(-2, 5) and Q(4, 1/2) is 3x + 4y - 14 = 0. -/
theorem line_equation_through_points :
  let p1 : ℝ × ℝ := (-2, 5)
  let p2 : ℝ × ℝ := (4, 1/2)
  ∀ x y : ℝ, line_equation p1 p2 x y ↔ 3 * x + 4 * y - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l634_63406


namespace NUMINAMATH_CALUDE_simplify_fraction_l634_63485

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) :
  (a - 1) / a / (a - 1 / a) = 1 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l634_63485


namespace NUMINAMATH_CALUDE_sine_ratio_equals_one_l634_63439

theorem sine_ratio_equals_one (c : ℝ) (h : c = 2 * π / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin (2 * c) * Real.sin (4 * c) * Real.sin (6 * c) * Real.sin (8 * c) * Real.sin (10 * c)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sine_ratio_equals_one_l634_63439


namespace NUMINAMATH_CALUDE_factorization_sum_l634_63483

theorem factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 17*x + 72 = (x + a) * (x + b)) →
  (∀ x, x^2 + 9*x - 90 = (x + b) * (x - c)) →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l634_63483


namespace NUMINAMATH_CALUDE_mary_money_left_l634_63426

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 2 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem stating that Mary has 50 - 8q dollars left after her purchases -/
theorem mary_money_left (q : ℝ) : money_left q = 50 - 8 * q := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l634_63426


namespace NUMINAMATH_CALUDE_sweater_markup_percentage_l634_63450

/-- Given a sweater with wholesale cost and retail price, proves the markup percentage. -/
theorem sweater_markup_percentage 
  (W : ℝ) -- Wholesale cost
  (R : ℝ) -- Normal retail price
  (h1 : W > 0) -- Wholesale cost is positive
  (h2 : R > 0) -- Retail price is positive
  (h3 : 0.4 * R = 1.2 * W) -- Condition for 60% discount and 20% profit
  : (R - W) / W * 100 = 200 :=
by sorry

end NUMINAMATH_CALUDE_sweater_markup_percentage_l634_63450


namespace NUMINAMATH_CALUDE_sin_alpha_value_l634_63436

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/6) = 1/3) :
  Real.sin α = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l634_63436


namespace NUMINAMATH_CALUDE_b3f_hex_to_decimal_l634_63410

/-- Converts a single hexadecimal digit to its decimal value -/
def hexToDecimal (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hexStringToDecimal (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hexToDecimal c) 0

theorem b3f_hex_to_decimal :
  hexStringToDecimal "B3F" = 2879 := by
  sorry

end NUMINAMATH_CALUDE_b3f_hex_to_decimal_l634_63410


namespace NUMINAMATH_CALUDE_remainder_nineteen_power_plus_nineteen_mod_twenty_l634_63427

theorem remainder_nineteen_power_plus_nineteen_mod_twenty : (19^19 + 19) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nineteen_power_plus_nineteen_mod_twenty_l634_63427


namespace NUMINAMATH_CALUDE_log_equation_solution_l634_63416

theorem log_equation_solution :
  ∀ y : ℝ, (Real.log y + 3 * Real.log 5 = 1) ↔ (y = 2/25) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l634_63416


namespace NUMINAMATH_CALUDE_toy_football_sales_performance_toy_football_sales_performance_equality_l634_63494

/-- Represents the sales performance of two students selling toy footballs --/
theorem toy_football_sales_performance
  (x y z : ℝ)  -- Prices of toy footballs in three sessions
  (hx : x > 0) (hy : y > 0) (hz : z > 0)  -- Prices are positive
  : (x + y + z) / 3 ≥ 3 / (1/x + 1/y + 1/z) := by
  sorry

/-- Equality condition for the sales performance --/
theorem toy_football_sales_performance_equality
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  : (x + y + z) / 3 = 3 / (1/x + 1/y + 1/z) ↔ x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_toy_football_sales_performance_toy_football_sales_performance_equality_l634_63494


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l634_63415

theorem yellow_marbles_count (blue : ℕ) (red : ℕ) (yellow : ℕ) :
  blue = 7 →
  red = 11 →
  (yellow : ℚ) / (blue + red + yellow : ℚ) = 1/4 →
  yellow = 6 :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l634_63415


namespace NUMINAMATH_CALUDE_bingley_has_six_bracelets_l634_63482

/-- The number of bracelets Bingley has remaining after the exchanges -/
def bingleys_remaining_bracelets (bingley_initial : ℕ) (kelly_initial : ℕ) : ℕ :=
  let bingley_after_kelly := bingley_initial + kelly_initial / 4
  bingley_after_kelly - bingley_after_kelly / 3

/-- Theorem stating that Bingley will have 6 bracelets remaining -/
theorem bingley_has_six_bracelets : 
  bingleys_remaining_bracelets 5 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bingley_has_six_bracelets_l634_63482


namespace NUMINAMATH_CALUDE_union_when_a_is_neg_two_intersection_equals_B_iff_l634_63433

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 6}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem 1: When a = -2, A ∪ B = {x | -5 ≤ x ≤ 6}
theorem union_when_a_is_neg_two :
  A ∪ B (-2) = {x : ℝ | -5 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem 2: A ∩ B = B if and only if a ≥ -1
theorem intersection_equals_B_iff (a : ℝ) :
  A ∩ B a = B a ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_neg_two_intersection_equals_B_iff_l634_63433


namespace NUMINAMATH_CALUDE_max_alpha_is_half_l634_63467

/-- The set of functions satisfying the given condition -/
def F : Set (ℝ → ℝ) :=
  {f | ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x}

/-- The theorem stating that 1/2 is the maximum α -/
theorem max_alpha_is_half :
    (∃ α : ℝ, ∀ f ∈ F, ∀ x > 0, f x ≥ α * x) ∧
    (∀ β : ℝ, (∀ f ∈ F, ∀ x > 0, f x ≥ β * x) → β ≤ 1/2) :=
  sorry


end NUMINAMATH_CALUDE_max_alpha_is_half_l634_63467


namespace NUMINAMATH_CALUDE_solid_with_isosceles_triangle_views_is_tetrahedron_l634_63446

/-- A solid object in 3D space -/
structure Solid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a view (projection) of a solid -/
inductive View
  | Front
  | Top
  | Side

/-- Represents the shape of a view -/
inductive Shape
  | IsoscelesTriangle
  | Other

/-- Function to get the shape of a view for a given solid -/
def viewShape (s : Solid) (v : View) : Shape :=
  sorry -- Implementation details

/-- Predicate to check if a solid is a tetrahedron -/
def isTetrahedron (s : Solid) : Prop :=
  sorry -- Definition of a tetrahedron

/-- Theorem: If all three views of a solid are isosceles triangles, then it's a tetrahedron -/
theorem solid_with_isosceles_triangle_views_is_tetrahedron (s : Solid) :
  (∀ v : View, viewShape s v = Shape.IsoscelesTriangle) →
  isTetrahedron s :=
sorry

end NUMINAMATH_CALUDE_solid_with_isosceles_triangle_views_is_tetrahedron_l634_63446


namespace NUMINAMATH_CALUDE_math_score_calculation_l634_63496

theorem math_score_calculation (initial_average : ℝ) (num_initial_subjects : ℕ) (average_drop : ℝ) :
  initial_average = 95 →
  num_initial_subjects = 3 →
  average_drop = 3 →
  let total_initial_score := initial_average * num_initial_subjects
  let new_average := initial_average - average_drop
  let new_total_score := new_average * (num_initial_subjects + 1)
  new_total_score - total_initial_score = 83 := by
  sorry

end NUMINAMATH_CALUDE_math_score_calculation_l634_63496


namespace NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l634_63489

/-- The curve defined by the polar equation θ = π/4 is a straight line -/
theorem polar_equation_pi_over_four_is_line : 
  ∀ (r : ℝ), ∃ (x y : ℝ), x = r * Real.cos (π/4) ∧ y = r * Real.sin (π/4) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l634_63489


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l634_63444

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l634_63444


namespace NUMINAMATH_CALUDE_books_remaining_l634_63424

theorem books_remaining (initial_books : ℕ) (given_away : ℕ) (sold : ℕ) : 
  initial_books = 108 → given_away = 35 → sold = 11 → 
  initial_books - given_away - sold = 62 := by
sorry

end NUMINAMATH_CALUDE_books_remaining_l634_63424


namespace NUMINAMATH_CALUDE_circle_equation_l634_63401

theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ (x - 2)^2 + y^2 = r^2) ∧ 
  ((-2)^2 + 0^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l634_63401


namespace NUMINAMATH_CALUDE_centroid_of_S_l634_63418

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ p.2 ∧ p.2 ≤ abs p.1 + 3 ∧ p.2 ≤ 4}

-- Define the centroid of a set
def centroid (T : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Theorem statement
theorem centroid_of_S :
  centroid S = (0, 13/5) := by sorry

end NUMINAMATH_CALUDE_centroid_of_S_l634_63418


namespace NUMINAMATH_CALUDE_parallel_transitive_l634_63434

-- Define a type for vectors
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define a predicate for parallel vectors
def parallel (u v : V) : Prop := ∃ (k : ℝ), v = k • u

-- State the theorem
theorem parallel_transitive {a b c : V} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : parallel a b) (hac : parallel a c) : parallel b c := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l634_63434


namespace NUMINAMATH_CALUDE_total_feathers_is_11638_l634_63475

/-- The total number of feathers needed for all animals in the circus performance --/
def total_feathers : ℕ :=
  let group1_animals : ℕ := 934
  let group1_feathers_per_crown : ℕ := 7
  let group2_animals : ℕ := 425
  let group2_feathers_per_crown : ℕ := 12
  (group1_animals * group1_feathers_per_crown) + (group2_animals * group2_feathers_per_crown)

/-- Theorem stating that the total number of feathers needed is 11638 --/
theorem total_feathers_is_11638 : total_feathers = 11638 := by
  sorry

end NUMINAMATH_CALUDE_total_feathers_is_11638_l634_63475


namespace NUMINAMATH_CALUDE_original_worker_count_l634_63488

/-- Given a work that can be completed by some workers in 65 days,
    and adding 10 workers reduces the time to 55 days,
    prove that the original number of workers is 55. -/
theorem original_worker_count (work : ℕ) : ∃ (workers : ℕ), 
  (workers * 65 = (workers + 10) * 55) ∧ 
  (workers = 55) := by
  sorry

end NUMINAMATH_CALUDE_original_worker_count_l634_63488


namespace NUMINAMATH_CALUDE_dog_weight_is_ten_l634_63458

/-- Represents the weights of a kitten, rabbit, and dog satisfying certain conditions -/
structure AnimalWeights where
  kitten : ℝ
  rabbit : ℝ
  dog : ℝ
  total_weight : kitten + rabbit + dog = 30
  kitten_rabbit_twice_dog : kitten + rabbit = 2 * dog
  kitten_dog_equals_rabbit : kitten + dog = rabbit

/-- The weight of the dog in the AnimalWeights structure is 10 pounds -/
theorem dog_weight_is_ten (w : AnimalWeights) : w.dog = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_is_ten_l634_63458


namespace NUMINAMATH_CALUDE_calculation_proof_l634_63481

theorem calculation_proof : 72 / (6 / 3) * 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l634_63481


namespace NUMINAMATH_CALUDE_cos_2x_eq_cos_2y_l634_63402

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_eq_cos_2y_l634_63402


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l634_63429

theorem pure_imaginary_fraction (a : ℝ) :
  (∃ b : ℝ, (a^2 + Complex.I) / (1 - Complex.I) = Complex.I * b) →
  a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l634_63429


namespace NUMINAMATH_CALUDE_min_quotient_four_digit_number_l634_63499

def is_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

theorem min_quotient_four_digit_number :
  ∃ (a b c d : ℕ),
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∀ (w x y z : ℕ),
      is_digit w → is_digit x → is_digit y → is_digit z →
      w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
      (1000 * a + 100 * b + 10 * c + d : ℚ) / (a + b + c + d) ≤
      (1000 * w + 100 * x + 10 * y + z : ℚ) / (w + x + y + z) :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_four_digit_number_l634_63499


namespace NUMINAMATH_CALUDE_snake_length_difference_l634_63448

/-- Given two snakes with a combined length of 70 inches, where one snake is 41 inches long,
    prove that the difference in length between the longer and shorter snake is 12 inches. -/
theorem snake_length_difference (combined_length jake_length : ℕ)
  (h1 : combined_length = 70)
  (h2 : jake_length = 41)
  (h3 : jake_length > combined_length - jake_length) :
  jake_length - (combined_length - jake_length) = 12 := by
  sorry

end NUMINAMATH_CALUDE_snake_length_difference_l634_63448


namespace NUMINAMATH_CALUDE_tetrahedron_edge_assignment_exists_l634_63470

/-- Represents a tetrahedron with face areas -/
structure Tetrahedron where
  s : ℝ  -- smallest face area
  S : ℝ  -- largest face area
  a : ℝ  -- another face area
  b : ℝ  -- another face area
  h_s_smallest : s ≤ S ∧ s ≤ a ∧ s ≤ b
  h_S_largest : S ≥ s ∧ S ≥ a ∧ S ≥ b
  h_positive : s > 0 ∧ S > 0 ∧ a > 0 ∧ b > 0

/-- Represents the assignment of numbers to the edges of a tetrahedron -/
structure EdgeAssignment (t : Tetrahedron) where
  e1 : ℝ  -- edge common to smallest and largest face
  e2 : ℝ  -- edge of smallest face
  e3 : ℝ  -- edge of smallest face
  e4 : ℝ  -- edge of largest face
  e5 : ℝ  -- edge of largest face
  e6 : ℝ  -- remaining edge
  h_non_negative : e1 ≥ 0 ∧ e2 ≥ 0 ∧ e3 ≥ 0 ∧ e4 ≥ 0 ∧ e5 ≥ 0 ∧ e6 ≥ 0

/-- The theorem stating that a valid edge assignment exists for any tetrahedron -/
theorem tetrahedron_edge_assignment_exists (t : Tetrahedron) :
  ∃ (ea : EdgeAssignment t),
    ea.e1 + ea.e2 + ea.e3 = t.s ∧
    ea.e1 + ea.e4 + ea.e5 = t.S ∧
    ea.e2 + ea.e5 + ea.e6 = t.a ∧
    ea.e3 + ea.e4 + ea.e6 = t.b :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_assignment_exists_l634_63470


namespace NUMINAMATH_CALUDE_right_triangle_arc_segment_l634_63431

theorem right_triangle_arc_segment (AC CB : ℝ) (h_AC : AC = 15) (h_CB : CB = 8) :
  let AB := Real.sqrt (AC^2 + CB^2)
  let CP := (AC * CB) / AB
  let PB := Real.sqrt (CB^2 - CP^2)
  let BD := 2 * PB
  BD = 128 / 17 := by sorry

end NUMINAMATH_CALUDE_right_triangle_arc_segment_l634_63431


namespace NUMINAMATH_CALUDE_kozlov_inequality_l634_63440

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
  2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_kozlov_inequality_l634_63440


namespace NUMINAMATH_CALUDE_custom_operation_result_l634_63460

/-- Custom operation ⊕ -/
def oplus (x y : ℝ) : ℝ := x + 2*y + 3

theorem custom_operation_result :
  ∀ (a b : ℝ), 
  (oplus (oplus (a^3) (a^2)) a = oplus (a^3) (oplus (a^2) a)) ∧
  (oplus (oplus (a^3) (a^2)) a = b) →
  a + b = 21/8 := by
sorry

end NUMINAMATH_CALUDE_custom_operation_result_l634_63460


namespace NUMINAMATH_CALUDE_ln_f_greater_than_one_max_a_value_l634_63474

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Theorem for part (I)
theorem ln_f_greater_than_one :
  ∀ x : ℝ, Real.log (f (-1) x) > 1 := by sorry

-- Theorem for part (II)
theorem max_a_value :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≥ a) ∧
  (∀ b : ℝ, (∀ x : ℝ, f b x ≥ b) → b ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ln_f_greater_than_one_max_a_value_l634_63474


namespace NUMINAMATH_CALUDE_min_teams_for_players_l634_63466

theorem min_teams_for_players (total_players : ℕ) (max_per_team : ℕ) (min_teams : ℕ) : 
  total_players = 30 → 
  max_per_team = 7 → 
  min_teams = 5 → 
  (∀ t : ℕ, t < min_teams → t * max_per_team < total_players) ∧ 
  (min_teams * (total_players / min_teams) = total_players) := by
  sorry

end NUMINAMATH_CALUDE_min_teams_for_players_l634_63466


namespace NUMINAMATH_CALUDE_max_notebooks_is_11_l634_63425

def single_notebook_cost : ℕ := 2
def pack_4_cost : ℕ := 6
def pack_7_cost : ℕ := 9
def total_money : ℕ := 15
def max_pack_7 : ℕ := 1

def notebooks_count (singles pack_4 pack_7 : ℕ) : ℕ :=
  singles + 4 * pack_4 + 7 * pack_7

def total_cost (singles pack_4 pack_7 : ℕ) : ℕ :=
  single_notebook_cost * singles + pack_4_cost * pack_4 + pack_7_cost * pack_7

theorem max_notebooks_is_11 :
  ∃ (singles pack_4 pack_7 : ℕ),
    notebooks_count singles pack_4 pack_7 = 11 ∧
    total_cost singles pack_4 pack_7 ≤ total_money ∧
    pack_7 ≤ max_pack_7 ∧
    ∀ (s p4 p7 : ℕ),
      total_cost s p4 p7 ≤ total_money →
      p7 ≤ max_pack_7 →
      notebooks_count s p4 p7 ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_is_11_l634_63425


namespace NUMINAMATH_CALUDE_platform_length_theorem_l634_63453

def train_length : ℝ := 250

theorem platform_length_theorem (X Y : ℝ) (platform_time signal_time : ℝ) 
  (h1 : platform_time = 40)
  (h2 : signal_time = 20)
  (h3 : Y * signal_time = train_length) :
  Y = 12.5 ∧ ∃ L, L = X * platform_time - train_length := by
  sorry

#check platform_length_theorem

end NUMINAMATH_CALUDE_platform_length_theorem_l634_63453


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l634_63411

theorem initial_mean_calculation (n : ℕ) (correct_value wrong_value : ℝ) (correct_mean : ℝ) :
  n = 20 ∧ 
  correct_value = 160 ∧ 
  wrong_value = 135 ∧ 
  correct_mean = 151.25 →
  (n * correct_mean - correct_value + wrong_value) / n = 152.5 := by
sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l634_63411


namespace NUMINAMATH_CALUDE_distance_to_line_l634_63462

/-- Represents a square with side length 2 inches -/
structure Square where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The configuration of three squares, where the middle one is rotated -/
structure SquareConfiguration where
  left : Square
  middle : Square
  right : Square
  middle_rotated : middle.side_length = left.side_length ∧ middle.side_length = right.side_length

/-- The theorem stating the distance of point B from the original line -/
theorem distance_to_line (config : SquareConfiguration) :
  let diagonal := config.middle.side_length * Real.sqrt 2
  let height_increase := diagonal / 2
  let original_height := config.middle.side_length / 2
  height_increase + original_height = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_line_l634_63462


namespace NUMINAMATH_CALUDE_expand_and_simplify_l634_63441

theorem expand_and_simplify (x : ℝ) : (x + 2) * (x - 2) - x * (x + 1) = -x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l634_63441


namespace NUMINAMATH_CALUDE_pell_solution_valid_pell_recurrence_relation_l634_63412

/-- Pell's equation solution type -/
structure PellSolution (D : ℕ) where
  x : ℤ
  y : ℤ
  eq : x^2 - D * y^2 = 1

/-- Generate the kth Pell solution -/
def genPellSolution (D : ℕ) (x₀ y₀ : ℤ) (k : ℕ) : PellSolution D :=
  sorry

theorem pell_solution_valid (D : ℕ) (x₀ y₀ : ℤ) (h : ¬ ∃ n : ℕ, n^2 = D) 
    (h₀ : x₀^2 - D * y₀^2 = 1) (k : ℕ) :
  let sol := genPellSolution D x₀ y₀ k
  sol.x^2 - D * sol.y^2 = 1 :=
sorry

theorem pell_recurrence_relation (D : ℕ) (x₀ y₀ : ℤ) (h : ¬ ∃ n : ℕ, n^2 = D) 
    (h₀ : x₀^2 - D * y₀^2 = 1) (k : ℕ) :
  let x₁ := (genPellSolution D x₀ y₀ (k+1)).x
  let x₂ := (genPellSolution D x₀ y₀ (k+2)).x
  let x := (genPellSolution D x₀ y₀ k).x
  x₂ = 2 * x₀ * x₁ - x :=
sorry

end NUMINAMATH_CALUDE_pell_solution_valid_pell_recurrence_relation_l634_63412


namespace NUMINAMATH_CALUDE_distance_between_trees_l634_63451

/-- Given a yard of length 375 meters with 26 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
    (h1 : yard_length = 375)
    (h2 : num_trees = 26) :
    yard_length / (num_trees - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l634_63451


namespace NUMINAMATH_CALUDE_license_plate_count_l634_63421

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of odd digits --/
def num_odd_digits : ℕ := 5

/-- The number of even digits --/
def num_even_digits : ℕ := 5

/-- The total number of possible license plates --/
def total_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_even_digits

theorem license_plate_count :
  total_plates = 439400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l634_63421


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l634_63443

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 740 →
  absent_children = 370 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    initial_bananas * total_children = (initial_bananas + extra_bananas) * (total_children - absent_children) ∧
    initial_bananas = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l634_63443


namespace NUMINAMATH_CALUDE_function_composition_theorem_l634_63435

theorem function_composition_theorem (a b : ℝ) :
  (∀ x : ℝ, (3 * ((a * x + b) : ℝ) - 6 : ℝ) = 4 * x + 3) →
  a + b = 13 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_theorem_l634_63435


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l634_63420

theorem floor_ceil_sum : ⌊(3.998 : ℝ)⌋ + ⌈(7.002 : ℝ)⌉ = 11 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l634_63420


namespace NUMINAMATH_CALUDE_remainder_sum_modulo_l634_63477

theorem remainder_sum_modulo (x y : ℤ) 
  (hx : x % 126 = 37) 
  (hy : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_modulo_l634_63477


namespace NUMINAMATH_CALUDE_star_property_l634_63454

def star (a b : ℝ) : ℝ := (a + b)^2

theorem star_property (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_star_property_l634_63454


namespace NUMINAMATH_CALUDE_inverse_congruence_solution_l634_63487

theorem inverse_congruence_solution (p : ℕ) (a : ℤ) (hp : Nat.Prime p) :
  (∃ x : ℤ, (a * x) % p = 1 ∧ x % p = a % p) ↔ (a % p = 1 ∨ a % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_congruence_solution_l634_63487


namespace NUMINAMATH_CALUDE_art_exhibition_tickets_l634_63417

theorem art_exhibition_tickets (advanced_price door_price total_tickets total_revenue : ℕ) 
  (h1 : advanced_price = 8)
  (h2 : door_price = 14)
  (h3 : total_tickets = 140)
  (h4 : total_revenue = 1720) :
  ∃ (advanced_tickets : ℕ),
    advanced_tickets * advanced_price + (total_tickets - advanced_tickets) * door_price = total_revenue ∧
    advanced_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_art_exhibition_tickets_l634_63417


namespace NUMINAMATH_CALUDE_growth_rate_correct_optimal_price_correct_l634_63400

-- Define the visitor numbers
def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000

-- Define the milk tea shop parameters
def cost_price : ℕ := 6
def base_price : ℕ := 25
def base_sales : ℕ := 300
def price_elasticity : ℕ := 30
def target_profit : ℕ := 6300

-- Part 1: Growth rate
def average_growth_rate : ℚ := 1/5

theorem growth_rate_correct :
  (visitors_2022 : ℚ) * (1 + average_growth_rate)^2 = visitors_2024 := by sorry

-- Part 2: Optimal price
def optimal_price : ℕ := 20

theorem optimal_price_correct :
  (optimal_price - cost_price) * (base_sales + price_elasticity * (base_price - optimal_price)) = target_profit ∧
  ∀ p : ℕ, p < base_price → p > optimal_price →
    (p - cost_price) * (base_sales + price_elasticity * (base_price - p)) < target_profit := by sorry

end NUMINAMATH_CALUDE_growth_rate_correct_optimal_price_correct_l634_63400


namespace NUMINAMATH_CALUDE_paint_coverage_l634_63455

/-- Proves that given a cube with 10-foot edges, if it costs $16 to paint the entire surface
    of the cube and paint costs $3.20 per quart, then one quart of paint covers 120 square feet. -/
theorem paint_coverage (cube_edge : Real) (total_cost : Real) (cost_per_quart : Real) :
  cube_edge = 10 →
  total_cost = 16 →
  cost_per_quart = 3.20 →
  (6 * cube_edge^2) / (total_cost / cost_per_quart) = 120 := by
  sorry

end NUMINAMATH_CALUDE_paint_coverage_l634_63455


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l634_63428

/-- The cost price of a computer table given its selling price and markup percentage. -/
def cost_price (selling_price : ℚ) (markup_percent : ℚ) : ℚ :=
  selling_price / (1 + markup_percent / 100)

/-- Theorem stating that the cost price of a computer table is 6525 
    given a selling price of 8091 and a markup of 24%. -/
theorem computer_table_cost_price :
  cost_price 8091 24 = 6525 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l634_63428


namespace NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l634_63447

theorem linear_diophantine_equation_solutions
  (a b c : ℤ) (x₀ y₀ : ℤ) (h : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c ↔ ∃ k : ℤ, x = x₀ + k * b ∧ y = y₀ - k * a :=
by sorry

end NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l634_63447


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l634_63437

/-- Given a marsh with geese and ducks, calculate the number of ducks -/
theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) :
  total_birds - geese = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l634_63437
