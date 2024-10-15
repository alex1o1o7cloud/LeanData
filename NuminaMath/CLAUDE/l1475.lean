import Mathlib

namespace NUMINAMATH_CALUDE_min_value_one_iff_k_eq_two_ninths_l1475_147540

/-- The expression as a function of x, y, and k -/
def f (x y k : ℝ) : ℝ := 9*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9

/-- The theorem stating the minimum value of f is 1 iff k = 2/9 -/
theorem min_value_one_iff_k_eq_two_ninths :
  (∀ x y : ℝ, f x y (2/9 : ℝ) ≥ 1) ∧ (∃ x y : ℝ, f x y (2/9 : ℝ) = 1) ↔
  ∀ k : ℝ, (∀ x y : ℝ, f x y k ≥ 1) ∧ (∃ x y : ℝ, f x y k = 1) → k = 2/9 :=
sorry

end NUMINAMATH_CALUDE_min_value_one_iff_k_eq_two_ninths_l1475_147540


namespace NUMINAMATH_CALUDE_angle_inclination_range_l1475_147591

-- Define the slope k
def k : ℝ := sorry

-- Define the angle of inclination α in radians
def α : ℝ := sorry

-- Define the relationship between k and α
axiom slope_angle_relation : k = Real.tan α

-- Define the range of k
axiom k_range : -1 ≤ k ∧ k < 1

-- Define the range of α (0 to π)
axiom α_range : 0 ≤ α ∧ α < Real.pi

-- Theorem to prove
theorem angle_inclination_range :
  (0 ≤ α ∧ α < Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_inclination_range_l1475_147591


namespace NUMINAMATH_CALUDE_interest_problem_l1475_147564

/-- Given a sum of money put at simple interest for 3 years, if increasing the
    interest rate by 2% results in Rs. 360 more interest, then the sum is Rs. 6000. -/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 2) * 3 / 100 - P * R * 3 / 100 = 360) → P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l1475_147564


namespace NUMINAMATH_CALUDE_rectangle_side_multiple_of_6_l1475_147536

/-- A rectangle constructed from 1 x 6 rectangles -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)
  (area : ℕ)
  (area_eq : area = length * width)
  (divisible_by_6 : 6 ∣ area)

/-- Theorem: One side of a rectangle constructed from 1 x 6 rectangles is a multiple of 6 -/
theorem rectangle_side_multiple_of_6 (r : Rectangle) : 
  6 ∣ r.length ∨ 6 ∣ r.width :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_multiple_of_6_l1475_147536


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l1475_147501

/-- A parabola with equation y = x² - 4x - m -/
structure Parabola where
  m : ℝ

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = p.x^2 - 4*p.x - para.m

theorem parabola_point_relationship (para : Parabola) (A B C : Point)
    (hA : A.x = 2) (hB : B.x = -3) (hC : C.x = -1)
    (onA : lies_on A para) (onB : lies_on B para) (onC : lies_on C para) :
    A.y < C.y ∧ C.y < B.y := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l1475_147501


namespace NUMINAMATH_CALUDE_competition_sequences_count_l1475_147528

/-- The number of possible competition sequences for two teams with 7 members each -/
def competition_sequences : ℕ :=
  Nat.choose 14 7

/-- Theorem stating that the number of competition sequences is 3432 -/
theorem competition_sequences_count : competition_sequences = 3432 := by
  sorry

end NUMINAMATH_CALUDE_competition_sequences_count_l1475_147528


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1475_147542

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ (x - 2)^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1475_147542


namespace NUMINAMATH_CALUDE_science_fair_sophomores_fraction_l1475_147522

theorem science_fair_sophomores_fraction (s j n : ℕ) : 
  s > 0 → -- Ensure s is positive to avoid division by zero
  s = j → -- Equal number of sophomores and juniors
  j = n → -- Number of juniors equals number of seniors
  (4 * s / 5 : ℚ) / ((4 * s / 5 : ℚ) + (3 * j / 4 : ℚ) + (n / 3 : ℚ)) = 240 / 565 := by
  sorry

#check science_fair_sophomores_fraction

end NUMINAMATH_CALUDE_science_fair_sophomores_fraction_l1475_147522


namespace NUMINAMATH_CALUDE_mushroom_price_per_unit_l1475_147546

theorem mushroom_price_per_unit (total_mushrooms day2_mushrooms day1_revenue : ℕ) : 
  total_mushrooms = 65 →
  day2_mushrooms = 12 →
  day1_revenue = 58 →
  (total_mushrooms - day2_mushrooms - 2 * day2_mushrooms) * 2 = day1_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_mushroom_price_per_unit_l1475_147546


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1475_147583

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within that sphere,
    this theorem relates the surface area of the outer cube to the surface area of the inner cube. -/
theorem inscribed_cube_surface_area (outer_surface_area : ℝ) :
  outer_surface_area = 54 →
  ∃ (inner_surface_area : ℝ),
    inner_surface_area = 18 ∧
    (∃ (outer_side_length inner_side_length : ℝ),
      outer_surface_area = 6 * outer_side_length^2 ∧
      inner_surface_area = 6 * inner_side_length^2 ∧
      inner_side_length = outer_side_length / Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1475_147583


namespace NUMINAMATH_CALUDE_even_operations_l1475_147571

theorem even_operations (a b : ℤ) (ha : Even a) (hb : Odd b) : 
  Even (a * b) ∧ Even (a * a) := by
  sorry

end NUMINAMATH_CALUDE_even_operations_l1475_147571


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1475_147519

/-- The standard equation of a hyperbola passing through specific points and sharing asymptotes with another hyperbola -/
theorem hyperbola_standard_equation :
  ∀ (x y : ℝ → ℝ),
  (∃ (t : ℝ), x t = -3 ∧ y t = 2 * Real.sqrt 7) →
  (∃ (t : ℝ), x t = 6 * Real.sqrt 2 ∧ y t = -7) →
  (∃ (t : ℝ), x t = 2 ∧ y t = 2 * Real.sqrt 3) →
  (∀ (t : ℝ), (x t)^2 / 4 - (y t)^2 / 3 = 1 ↔ ∃ (k : ℝ), k * ((x t)^2 / 4 - (y t)^2 / 3) = k) →
  ∀ (t : ℝ), (y t)^2 / 9 - (x t)^2 / 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1475_147519


namespace NUMINAMATH_CALUDE_max_value_of_f_l1475_147560

def S : Finset ℕ := {0, 1, 2, 3, 4}

def f (a b c d e : ℕ) : ℕ := e * c^a + b - d

theorem max_value_of_f :
  ∃ (a b c d e : ℕ),
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    f a b c d e = 39 ∧
    ∀ (a' b' c' d' e' : ℕ),
      a' ∈ S → b' ∈ S → c' ∈ S → d' ∈ S → e' ∈ S →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' →
      b' ≠ c' → b' ≠ d' → b' ≠ e' →
      c' ≠ d' → c' ≠ e' →
      d' ≠ e' →
      f a' b' c' d' e' ≤ 39 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1475_147560


namespace NUMINAMATH_CALUDE_combined_age_proof_l1475_147500

/-- Given that Hezekiah is 4 years old and Ryanne is 7 years older than Hezekiah,
    prove that their combined age is 15 years. -/
theorem combined_age_proof (hezekiah_age : ℕ) (ryanne_age : ℕ) : 
  hezekiah_age = 4 → 
  ryanne_age = hezekiah_age + 7 → 
  hezekiah_age + ryanne_age = 15 := by
sorry

end NUMINAMATH_CALUDE_combined_age_proof_l1475_147500


namespace NUMINAMATH_CALUDE_abc_inequality_l1475_147511

theorem abc_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1475_147511


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l1475_147597

/-- An arithmetic progression with a_3 = 10 -/
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d ∧ a 3 = 10

/-- The sum of a_1, a_2, and a_6 in the arithmetic progression -/
def sum_terms (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 6

theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  arithmetic_progression a → sum_terms a = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l1475_147597


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1475_147559

theorem toy_store_revenue_ratio : 
  ∀ (N D J : ℝ),
  J = (1 / 2) * N →
  D = (10 / 3) * ((N + J) / 2) →
  N / D = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1475_147559


namespace NUMINAMATH_CALUDE_additional_investment_rate_l1475_147509

/-- Proves that the interest rate of an additional investment is 10% given specific conditions --/
theorem additional_investment_rate (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (total_rate : ℝ) : 
  initial_investment = 2400 →
  initial_rate = 0.05 →
  additional_investment = 600 →
  total_rate = 0.06 →
  (initial_investment * initial_rate + additional_investment * 0.1) / 
    (initial_investment + additional_investment) = total_rate := by
  sorry

#check additional_investment_rate

end NUMINAMATH_CALUDE_additional_investment_rate_l1475_147509


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1475_147526

/-- Given a hyperbola with the standard equation (x²/a² - y²/b² = 1),
    one focus at (-2, 0), and the angle between asymptotes is 60°,
    prove that its equation is either x² - y²/3 = 1 or x²/3 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 4) →
  (b / a = Real.sqrt 3 ∨ b / a = Real.sqrt 3 / 3) →
  ((∀ x y : ℝ, x^2 - y^2 / 3 = 1) ∨ (∀ x y : ℝ, x^2 / 3 - y^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1475_147526


namespace NUMINAMATH_CALUDE_ratio_shoes_to_total_earned_l1475_147508

def rate_per_hour : ℕ := 14
def hours_per_day : ℕ := 2
def days_worked : ℕ := 7
def money_left : ℕ := 49

def total_hours : ℕ := hours_per_day * days_worked
def total_earned : ℕ := total_hours * rate_per_hour
def money_before_mom : ℕ := money_left * 2
def money_spent_shoes : ℕ := total_earned - money_before_mom

theorem ratio_shoes_to_total_earned :
  (money_spent_shoes : ℚ) / total_earned = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_shoes_to_total_earned_l1475_147508


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l1475_147576

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 5 < x)
  (h2 : 7 < x ∧ x < 18)
  (h3 : 2 < x ∧ x < 13)
  (h4 : 9 < x ∧ x < 12)
  (h5 : x + 1 < 13) :
  ∃ (y : ℤ), x < y ∧ (∀ (z : ℤ), x < z → y ≤ z) ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l1475_147576


namespace NUMINAMATH_CALUDE_fraction_simplification_l1475_147594

theorem fraction_simplification :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1475_147594


namespace NUMINAMATH_CALUDE_harvester_problem_l1475_147573

/-- Represents the number of harvesters of each type -/
structure HarvesterCount where
  typeA : ℕ
  typeB : ℕ

/-- Represents a plan for introducing additional harvesters -/
structure IntroductionPlan where
  additionalTypeA : ℕ
  additionalTypeB : ℕ

/-- The problem statement -/
theorem harvester_problem 
  (total_harvesters : ℕ)
  (typeA_capacity : ℕ)
  (typeB_capacity : ℕ)
  (total_daily_harvest : ℕ)
  (new_target : ℕ)
  (additional_harvesters : ℕ)
  (h1 : total_harvesters = 20)
  (h2 : typeA_capacity = 80)
  (h3 : typeB_capacity = 120)
  (h4 : total_daily_harvest = 2080)
  (h5 : new_target > 2900)
  (h6 : additional_harvesters = 8) :
  ∃ (initial : HarvesterCount) (plans : List IntroductionPlan),
    initial.typeA + initial.typeB = total_harvesters ∧
    initial.typeA * typeA_capacity + initial.typeB * typeB_capacity = total_daily_harvest ∧
    initial.typeA = 8 ∧
    initial.typeB = 12 ∧
    plans.length = 3 ∧
    ∀ plan ∈ plans, 
      plan.additionalTypeA + plan.additionalTypeB = additional_harvesters ∧
      (initial.typeA + plan.additionalTypeA) * typeA_capacity + 
      (initial.typeB + plan.additionalTypeB) * typeB_capacity > new_target :=
by sorry

end NUMINAMATH_CALUDE_harvester_problem_l1475_147573


namespace NUMINAMATH_CALUDE_cos_sin_pi_eighth_difference_l1475_147565

theorem cos_sin_pi_eighth_difference (π : Real) : 
  (Real.cos (π / 8))^4 - (Real.sin (π / 8))^4 = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_pi_eighth_difference_l1475_147565


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l1475_147582

theorem unique_six_digit_number : ∃! n : ℕ,
  (100000 ≤ n ∧ n < 1000000) ∧  -- 6-digit number
  (n % 10 = 2 ∧ n / 100000 = 2) ∧  -- begins and ends with 2
  (∃ k : ℕ, n = (2*k - 2) * (2*k) * (2*k + 2)) ∧  -- product of three consecutive even integers
  n = 287232 :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l1475_147582


namespace NUMINAMATH_CALUDE_platform_length_l1475_147563

/-- Given a train of length 1200 m that crosses a tree in 120 sec and passes a platform in 230 sec,
    the length of the platform is 1100 m. -/
theorem platform_length (train_length : ℝ) (tree_crossing_time : ℝ) (platform_passing_time : ℝ) :
  train_length = 1200 →
  tree_crossing_time = 120 →
  platform_passing_time = 230 →
  let train_speed := train_length / tree_crossing_time
  let platform_length := train_speed * platform_passing_time - train_length
  platform_length = 1100 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1475_147563


namespace NUMINAMATH_CALUDE_max_gcd_of_product_7200_l1475_147592

theorem max_gcd_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧
  ∀ (x y : ℕ), x * y = 7200 → Nat.gcd x y ≤ Nat.gcd a b ∧
  Nat.gcd a b = 60 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_product_7200_l1475_147592


namespace NUMINAMATH_CALUDE_monomial_degree_5_l1475_147533

/-- The degree of a monomial of the form 3a^2b^n -/
def monomialDegree (n : ℕ) : ℕ := 2 + n

theorem monomial_degree_5 (n : ℕ) : monomialDegree n = 5 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_degree_5_l1475_147533


namespace NUMINAMATH_CALUDE_function_positive_range_l1475_147551

-- Define the function f(x) = -x^2 + 2x + 3
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem function_positive_range :
  ∀ x : ℝ, f x > 0 ↔ -1 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_function_positive_range_l1475_147551


namespace NUMINAMATH_CALUDE_mikails_age_correct_l1475_147527

/-- Mikail's age on his birthday -/
def mikails_age : ℕ := 9

/-- Amount of money Mikail receives per year of age -/
def money_per_year : ℕ := 5

/-- Total amount of money Mikail receives on his birthday -/
def total_money : ℕ := 45

/-- Theorem: Mikail's age is correct given the money he receives -/
theorem mikails_age_correct : mikails_age = total_money / money_per_year := by
  sorry

end NUMINAMATH_CALUDE_mikails_age_correct_l1475_147527


namespace NUMINAMATH_CALUDE_number_equation_l1475_147513

theorem number_equation (x : ℝ) : (0.5 * x = (3/5) * x - 10) ↔ (x = 100) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1475_147513


namespace NUMINAMATH_CALUDE_book_problem_solution_l1475_147568

def book_problem (cost_loss : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : Prop :=
  let selling_price := cost_loss * (1 - loss_percent)
  let cost_gain := selling_price / (1 + gain_percent)
  cost_loss + cost_gain = 360

theorem book_problem_solution :
  book_problem 210 0.15 0.19 :=
sorry

end NUMINAMATH_CALUDE_book_problem_solution_l1475_147568


namespace NUMINAMATH_CALUDE_common_chord_equation_l1475_147530

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

/-- The equation of the line on which the common chord lies -/
def common_chord (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the common chord of the two circles lies on the line x - y + 1 = 0 -/
theorem common_chord_equation :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1475_147530


namespace NUMINAMATH_CALUDE_young_in_sample_is_seven_l1475_147549

/-- Represents the number of employees in each age group and the sample size --/
structure EmployeeData where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ
  sampleSize : ℕ

/-- Calculates the number of young employees in a stratified sample --/
def youngInSample (data : EmployeeData) : ℕ :=
  (data.young * data.sampleSize) / data.total

/-- Theorem stating that for the given employee data, the number of young employees in the sample is 7 --/
theorem young_in_sample_is_seven (data : EmployeeData)
  (h1 : data.total = 750)
  (h2 : data.young = 350)
  (h3 : data.middleAged = 250)
  (h4 : data.elderly = 150)
  (h5 : data.sampleSize = 15) :
  youngInSample data = 7 := by
  sorry


end NUMINAMATH_CALUDE_young_in_sample_is_seven_l1475_147549


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l1475_147599

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  sideLength : ℕ
  smallCubeSize : ℕ
  largeCubeSize : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (c : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 4 with specific corner removals has 48 edges -/
theorem modified_cube_edge_count :
  let c : ModifiedCube := ⟨4, 1, 2⟩
  edgeCount c = 48 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l1475_147599


namespace NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_l1475_147593

-- Define the number with 8 repeated ones
def X : ℕ := 11111111

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_X_squared : sum_of_digits (X^2) = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_l1475_147593


namespace NUMINAMATH_CALUDE_girls_count_l1475_147506

/-- The number of boys in the school -/
def num_boys : ℕ := 841

/-- The difference between the number of boys and girls -/
def boy_girl_diff : ℕ := 807

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys - boy_girl_diff

theorem girls_count : num_girls = 34 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l1475_147506


namespace NUMINAMATH_CALUDE_light_ray_reflection_l1475_147596

/-- A light ray reflection problem -/
theorem light_ray_reflection 
  (M : ℝ × ℝ) 
  (N : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) : 
  M = (2, 6) → 
  N = (-3, 4) → 
  (∀ x y, l x y ↔ x - y + 3 = 0) → 
  ∃ A B C : ℝ, 
    (∀ x y, A * x + B * y + C = 0 ↔ 
      (∃ K : ℝ × ℝ, 
        -- K is symmetric to M with respect to l
        (K.1 - M.1) / (K.2 - M.2) = -1 ∧ 
        l ((K.1 + M.1) / 2) ((K.2 + M.2) / 2) ∧
        -- N lies on the line through K
        (N.2 - K.2) / (N.1 - K.1) = (y - K.2) / (x - K.1))) ∧
    A = 1 ∧ B = -6 ∧ C = 27 :=
by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_l1475_147596


namespace NUMINAMATH_CALUDE_fraction_equality_l1475_147516

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1475_147516


namespace NUMINAMATH_CALUDE_factorial_sum_mod_30_l1475_147523

theorem factorial_sum_mod_30 : (1 + 2 + 6 + 24 + 120) % 30 = 3 := by sorry

end NUMINAMATH_CALUDE_factorial_sum_mod_30_l1475_147523


namespace NUMINAMATH_CALUDE_days_without_calls_is_244_l1475_147567

/-- The number of days in the year --/
def year_days : ℕ := 365

/-- The intervals at which the nephews call --/
def call_intervals : List ℕ := [4, 6, 8]

/-- Calculate the number of days without calls --/
def days_without_calls (total_days : ℕ) (intervals : List ℕ) : ℕ :=
  total_days - (total_days / intervals.head! + total_days / intervals.tail.head! + total_days / intervals.tail.tail.head! -
    total_days / (intervals.head!.lcm intervals.tail.head!) - 
    total_days / (intervals.head!.lcm intervals.tail.tail.head!) - 
    total_days / (intervals.tail.head!.lcm intervals.tail.tail.head!) +
    total_days / (intervals.head!.lcm intervals.tail.head!).lcm intervals.tail.tail.head!)

theorem days_without_calls_is_244 :
  days_without_calls year_days call_intervals = 244 := by
  sorry

end NUMINAMATH_CALUDE_days_without_calls_is_244_l1475_147567


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1475_147544

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, -1, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1475_147544


namespace NUMINAMATH_CALUDE_floor_equation_equivalence_l1475_147525

def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, (k + 1/5 ≤ x ∧ x < k + 1/3) ∨
           (k + 2/5 ≤ x ∧ x < k + 3/5) ∨
           (k + 2/3 ≤ x ∧ x < k + 4/5)

theorem floor_equation_equivalence (x : ℝ) :
  ⌊(5 : ℝ) * x⌋ = ⌊(3 : ℝ) * x⌋ + 2 * ⌊x⌋ + 1 ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_equivalence_l1475_147525


namespace NUMINAMATH_CALUDE_earthquake_relief_donation_scientific_notation_l1475_147557

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem earthquake_relief_donation_scientific_notation :
  toScientificNotation 3990000000 = ScientificNotation.mk 3.99 9 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_earthquake_relief_donation_scientific_notation_l1475_147557


namespace NUMINAMATH_CALUDE_two_valid_selections_l1475_147554

def numbers : List ℕ := [1, 2, 3, 4, 5]

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

def validSelection (a b : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧
  average (numbers.filter (λ x => x ≠ a ∧ x ≠ b)) = average numbers

theorem two_valid_selections :
  (∃! (pair : ℕ × ℕ), validSelection pair.1 pair.2) ∨
  (∃! (pair1 pair2 : ℕ × ℕ), 
    validSelection pair1.1 pair1.2 ∧ 
    validSelection pair2.1 pair2.2 ∧ 
    pair1 ≠ pair2) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_selections_l1475_147554


namespace NUMINAMATH_CALUDE_birds_in_marsh_l1475_147545

theorem birds_in_marsh (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_marsh_l1475_147545


namespace NUMINAMATH_CALUDE_train_speed_l1475_147547

/-- A train passes a pole in 5 seconds and crosses a 360-meter long stationary train in 25 seconds. -/
theorem train_speed (pole_passing_time : ℝ) (stationary_train_length : ℝ) (crossing_time : ℝ)
  (h1 : pole_passing_time = 5)
  (h2 : stationary_train_length = 360)
  (h3 : crossing_time = 25) :
  ∃ (speed : ℝ), speed = 18 ∧ 
    speed * pole_passing_time = speed * crossing_time - stationary_train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l1475_147547


namespace NUMINAMATH_CALUDE_no_equality_under_condition_l1475_147510

theorem no_equality_under_condition :
  ¬∃ (a b c : ℝ), (a^2 + b*c = (a + b)*(a + c)) ∧ (a + b + c = 2) :=
sorry

end NUMINAMATH_CALUDE_no_equality_under_condition_l1475_147510


namespace NUMINAMATH_CALUDE_watch_sale_loss_percentage_l1475_147524

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_sale_loss_percentage (cost_price : ℝ) (additional_amount : ℝ) (gain_percentage : ℝ) :
  cost_price = 3000 →
  additional_amount = 540 →
  gain_percentage = 8 →
  ∃ (loss_percentage : ℝ),
    loss_percentage = 10 ∧
    cost_price * (1 + gain_percentage / 100) = 
    cost_price * (1 - loss_percentage / 100) + additional_amount :=
by
  sorry

end NUMINAMATH_CALUDE_watch_sale_loss_percentage_l1475_147524


namespace NUMINAMATH_CALUDE_wills_initial_amount_l1475_147555

/-- The amount of money Will's mom gave him initially -/
def initial_amount : ℕ := 74

/-- The cost of the sweater Will bought -/
def sweater_cost : ℕ := 9

/-- The cost of the T-shirt Will bought -/
def tshirt_cost : ℕ := 11

/-- The cost of the shoes Will bought -/
def shoes_cost : ℕ := 30

/-- The refund percentage for the returned shoes -/
def refund_percentage : ℚ := 90 / 100

/-- The amount of money Will has left after all transactions -/
def money_left : ℕ := 51

theorem wills_initial_amount :
  initial_amount = 
    money_left + 
    sweater_cost + 
    tshirt_cost + 
    shoes_cost - 
    (↑shoes_cost * refund_percentage).floor :=
by sorry

end NUMINAMATH_CALUDE_wills_initial_amount_l1475_147555


namespace NUMINAMATH_CALUDE_parabola_intersection_l1475_147512

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection points -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Main theorem -/
theorem parabola_intersection (C : Parabola) (l : Line) (I : IntersectionPoints) :
  l.slope = 2 ∧
  l.point = (C.p / 2, 0) ∧
  (I.A.1 - C.p / 2) * (I.A.1 - C.p / 2) + I.A.2 * I.A.2 = 20 ∧
  (I.B.1 - C.p / 2) * (I.B.1 - C.p / 2) + I.B.2 * I.B.2 = 20 ∧
  I.A.2 * I.A.2 = 2 * C.p * I.A.1 ∧
  I.B.2 * I.B.2 = 2 * C.p * I.B.1 →
  C.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1475_147512


namespace NUMINAMATH_CALUDE_waiter_customers_l1475_147581

theorem waiter_customers (initial : ℕ) (left : ℕ) (new : ℕ) : 
  initial = 14 → left = 3 → new = 39 → initial - left + new = 50 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l1475_147581


namespace NUMINAMATH_CALUDE_smallest_four_digit_2_mod_5_l1475_147577

theorem smallest_four_digit_2_mod_5 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n % 5 = 2 → n ≥ 1002 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_2_mod_5_l1475_147577


namespace NUMINAMATH_CALUDE_second_car_distance_l1475_147561

/-- Calculates the distance traveled by the second car given the initial separation,
    the distance traveled by the first car, and the final distance between the cars. -/
def distance_traveled_by_second_car (initial_separation : ℝ) (distance_first_car : ℝ) (final_distance : ℝ) : ℝ :=
  initial_separation - (distance_first_car + final_distance)

/-- Theorem stating that given the conditions of the problem, 
    the second car must have traveled 87 km. -/
theorem second_car_distance : 
  let initial_separation : ℝ := 150
  let distance_first_car : ℝ := 25
  let final_distance : ℝ := 38
  distance_traveled_by_second_car initial_separation distance_first_car final_distance = 87 := by
  sorry

#eval distance_traveled_by_second_car 150 25 38

end NUMINAMATH_CALUDE_second_car_distance_l1475_147561


namespace NUMINAMATH_CALUDE_range_of_c_over_a_l1475_147589

theorem range_of_c_over_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a > b) (h3 : b > c) :
  -2 < c / a ∧ c / a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_over_a_l1475_147589


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l1475_147562

theorem max_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l1475_147562


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l1475_147569

theorem sum_of_fourth_powers_squared (x y z : ℤ) (h : x + y + z = 0) :
  ∃ (n : ℤ), 2 * (x^4 + y^4 + z^4) = n^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l1475_147569


namespace NUMINAMATH_CALUDE_simplify_expression_l1475_147520

theorem simplify_expression (y : ℝ) : 
  3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2) = 0 * y^2 + 0 * y - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1475_147520


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1475_147585

/-- Given a geometric sequence {a_n} where a_6 + a_8 = 4, 
    prove that a_8(a_4 + 2a_6 + a_8) = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1475_147585


namespace NUMINAMATH_CALUDE_linear_function_equation_l1475_147514

/-- A linear function passing through (3, 2) and intersecting positive x and y axes --/
structure LinearFunctionWithConstraints where
  f : ℝ → ℝ
  is_linear : ∀ x y c : ℝ, f (x + y) = f x + f y ∧ f (c * x) = c * f x
  passes_through : f 3 = 2
  intersects_x_axis : ∃ a : ℝ, a > 0 ∧ f a = 0
  intersects_y_axis : ∃ b : ℝ, b > 0 ∧ f 0 = b
  sum_of_intersects : let a := Classical.choose (intersects_x_axis)
                      let b := Classical.choose (intersects_y_axis)
                      a + b = 12

/-- The equation of the linear function satisfies the given constraints --/
theorem linear_function_equation (l : LinearFunctionWithConstraints) :
  (∀ x, l.f x = -2 * x + 8) ∨ (∀ x, l.f x = -1/3 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_equation_l1475_147514


namespace NUMINAMATH_CALUDE_bridge_length_l1475_147503

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 148 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  let bridge_length := total_distance - train_length
  bridge_length = 227 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1475_147503


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1475_147578

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 (3 * x) = Nat.choose 28 (x + 8)) ↔ (x = 4 ∨ x = 5) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1475_147578


namespace NUMINAMATH_CALUDE_cloud_height_above_lake_l1475_147579

/-- The height of a cloud above a lake surface, given observation conditions --/
theorem cloud_height_above_lake (h : ℝ) (elevation_angle depression_angle : ℝ) : 
  h = 10 → 
  elevation_angle = 30 * π / 180 →
  depression_angle = 45 * π / 180 →
  ∃ (cloud_height : ℝ), abs (cloud_height - 37.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_cloud_height_above_lake_l1475_147579


namespace NUMINAMATH_CALUDE_lollipop_sequence_l1475_147558

theorem lollipop_sequence (a b c d e : ℕ) : 
  a + b + c + d + e = 100 →
  b = a + 6 →
  c = b + 6 →
  d = c + 6 →
  e = d + 6 →
  c = 20 := by sorry

end NUMINAMATH_CALUDE_lollipop_sequence_l1475_147558


namespace NUMINAMATH_CALUDE_unused_types_count_l1475_147507

/-- The number of natural resources --/
def num_resources : ℕ := 6

/-- The number of types of nature use developed --/
def types_developed : ℕ := 23

/-- The total number of possible combinations of resource usage --/
def total_combinations : ℕ := 2^num_resources

/-- The number of valid combinations (excluding the all-zero combination) --/
def valid_combinations : ℕ := total_combinations - 1

/-- The number of unused types of nature use --/
def unused_types : ℕ := valid_combinations - types_developed

theorem unused_types_count : unused_types = 40 := by
  sorry

end NUMINAMATH_CALUDE_unused_types_count_l1475_147507


namespace NUMINAMATH_CALUDE_bricks_per_square_meter_l1475_147552

-- Define the parameters
def num_rooms : ℕ := 5
def room_length : ℝ := 4
def room_width : ℝ := 5
def room_height : ℝ := 2
def bricks_per_room : ℕ := 340

-- Define the theorem
theorem bricks_per_square_meter :
  let room_area : ℝ := room_length * room_width
  let bricks_per_sq_meter : ℝ := bricks_per_room / room_area
  bricks_per_sq_meter = 17 := by sorry

end NUMINAMATH_CALUDE_bricks_per_square_meter_l1475_147552


namespace NUMINAMATH_CALUDE_product_of_roots_l1475_147590

theorem product_of_roots (x : ℝ) : 
  (25 * x^2 + 60 * x - 350 = 0) → 
  ∃ r₁ r₂ : ℝ, (r₁ * r₂ = -14 ∧ 25 * r₁^2 + 60 * r₁ - 350 = 0 ∧ 25 * r₂^2 + 60 * r₂ - 350 = 0) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1475_147590


namespace NUMINAMATH_CALUDE_multiples_of_three_imply_F_equals_six_l1475_147553

def first_number (D E : ℕ) : ℕ := 8000000 + D * 100000 + 70000 + 3000 + E * 10 + 2

def second_number (D E F : ℕ) : ℕ := 4000000 + 100000 + 70000 + D * 1000 + E * 100 + 60 + F

theorem multiples_of_three_imply_F_equals_six (D E : ℕ) 
  (h1 : D < 10) (h2 : E < 10) 
  (h3 : ∃ k : ℕ, first_number D E = 3 * k) 
  (h4 : ∃ m : ℕ, second_number D E 6 = 3 * m) : 
  ∃ F : ℕ, F = 6 ∧ F < 10 ∧ ∃ n : ℕ, second_number D E F = 3 * n :=
sorry

end NUMINAMATH_CALUDE_multiples_of_three_imply_F_equals_six_l1475_147553


namespace NUMINAMATH_CALUDE_olympiad_solution_l1475_147532

def olympiad_problem (N_a N_b N_c N_ab N_ac N_bc N_abc : ℕ) : Prop :=
  let total := N_a + N_b + N_c + N_ab + N_ac + N_bc + N_abc
  let B_not_A := N_b + N_bc
  let C_not_A := N_c + N_bc
  let A_and_others := N_ab + N_ac + N_abc
  let only_one := N_a + N_b + N_c
  total = 25 ∧
  B_not_A = 2 * C_not_A ∧
  N_a = A_and_others + 1 ∧
  2 * N_a = only_one

theorem olympiad_solution :
  ∀ N_a N_b N_c N_ab N_ac N_bc N_abc,
  olympiad_problem N_a N_b N_c N_ab N_ac N_bc N_abc →
  N_b = 6 := by
sorry

end NUMINAMATH_CALUDE_olympiad_solution_l1475_147532


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1475_147570

theorem repeating_decimal_division : 
  let a := (36 : ℚ) / 99
  let b := (12 : ℚ) / 99
  a / b = 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1475_147570


namespace NUMINAMATH_CALUDE_baseball_team_score_l1475_147504

theorem baseball_team_score :
  let total_players : ℕ := 9
  let high_scorers : ℕ := 5
  let high_scorer_average : ℕ := 50
  let low_scorer_average : ℕ := 5
  let low_scorers : ℕ := total_players - high_scorers
  let total_score : ℕ := high_scorers * high_scorer_average + low_scorers * low_scorer_average
  total_score = 270 := by sorry

end NUMINAMATH_CALUDE_baseball_team_score_l1475_147504


namespace NUMINAMATH_CALUDE_percent_relation_l1475_147521

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 0.5 * a) :
  c = 0.5 * b := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1475_147521


namespace NUMINAMATH_CALUDE_diamond_four_three_l1475_147566

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_four_three_l1475_147566


namespace NUMINAMATH_CALUDE_trig_problem_l1475_147529

theorem trig_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (4 * Real.sin x * Real.cos x - Real.cos x^2 = -64/25) := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l1475_147529


namespace NUMINAMATH_CALUDE_eggs_remaining_l1475_147572

theorem eggs_remaining (original : ℝ) (removed : ℝ) (remaining : ℝ) : 
  original = 35.3 → removed = 4.5 → remaining = original - removed → remaining = 30.8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_remaining_l1475_147572


namespace NUMINAMATH_CALUDE_last_locker_theorem_l1475_147543

/-- The number of lockers in the hall -/
def num_lockers : ℕ := 2048

/-- The pattern of opening lockers -/
def open_pattern (n : ℕ) : Bool :=
  if n % 3 = 1 then true  -- opened in first pass
  else if n % 3 = 2 then true  -- opened in second pass
  else false  -- opened in third pass

/-- The last locker opened is the largest multiple of 3 not exceeding the number of lockers -/
def last_locker_opened (total : ℕ) : ℕ :=
  total - (total % 3)

theorem last_locker_theorem :
  last_locker_opened num_lockers = 2046 ∧
  ∀ n, n > last_locker_opened num_lockers → n ≤ num_lockers → open_pattern n = false :=
by sorry

end NUMINAMATH_CALUDE_last_locker_theorem_l1475_147543


namespace NUMINAMATH_CALUDE_xyz_max_value_l1475_147580

theorem xyz_max_value (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≤ 3/125 := by
sorry

end NUMINAMATH_CALUDE_xyz_max_value_l1475_147580


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l1475_147535

def income : ℕ := 21000
def savings : ℕ := 7000
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio : 
  (income : ℚ) / (expenditure : ℚ) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l1475_147535


namespace NUMINAMATH_CALUDE_line_and_circle_equations_l1475_147584

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space represented by the equation (x-h)² + (y-k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Given two points, determine if a line passes through the first point and is perpendicular to the line connecting the two points -/
def isPerpendicular (p1 p2 : Point) (l : Line) : Prop :=
  -- Line passes through p1
  l.a * p1.x + l.b * p1.y + l.c = 0 ∧
  -- Line is perpendicular to the line connecting p1 and p2
  l.a * (p2.x - p1.x) + l.b * (p2.y - p1.y) = 0

/-- Given two points, determine if a circle has these points as the endpoints of its diameter -/
def isDiameter (p1 p2 : Point) (c : Circle) : Prop :=
  -- Center of the circle is the midpoint of p1 and p2
  c.h = (p1.x + p2.x) / 2 ∧
  c.k = (p1.y + p2.y) / 2 ∧
  -- Radius of the circle is half the distance between p1 and p2
  c.r^2 = ((p2.x - p1.x)^2 + (p2.y - p1.y)^2) / 4

theorem line_and_circle_equations (A B : Point) (l : Line) (C : Circle)
    (hA : A.x = -3 ∧ A.y = -1)
    (hB : B.x = 5 ∧ B.y = 5)
    (hl : l.a = 4 ∧ l.b = 3 ∧ l.c = 15)
    (hC : C.h = 1 ∧ C.k = 2 ∧ C.r = 5) :
    isPerpendicular A B l ∧ isDiameter A B C := by
  sorry

end NUMINAMATH_CALUDE_line_and_circle_equations_l1475_147584


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1475_147587

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → 
  Even heart → 
  heart + club ≤ 26 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1475_147587


namespace NUMINAMATH_CALUDE_pyramid_height_is_two_main_theorem_l1475_147575

/-- A right square pyramid with given properties -/
structure RightSquarePyramid where
  top_side : ℝ
  bottom_side : ℝ
  lateral_area : ℝ
  height : ℝ

/-- The theorem stating the height of the pyramid is 2 -/
theorem pyramid_height_is_two (p : RightSquarePyramid) : p.height = 2 :=
  by
  have h1 : p.top_side = 3 := by sorry
  have h2 : p.bottom_side = 6 := by sorry
  have h3 : p.lateral_area = p.top_side ^ 2 + p.bottom_side ^ 2 := by sorry
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem : ∃ (p : RightSquarePyramid), 
  p.top_side = 3 ∧ 
  p.bottom_side = 6 ∧ 
  p.lateral_area = p.top_side ^ 2 + p.bottom_side ^ 2 ∧
  p.height = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_is_two_main_theorem_l1475_147575


namespace NUMINAMATH_CALUDE_two_times_first_exceeds_three_times_second_l1475_147595

theorem two_times_first_exceeds_three_times_second (x y : ℝ) : 
  x + y = 10 → x = 7 → y = 3 → 2 * x - 3 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_times_first_exceeds_three_times_second_l1475_147595


namespace NUMINAMATH_CALUDE_afternoon_and_evening_emails_l1475_147502

def morning_emails : ℕ := 4
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

theorem afternoon_and_evening_emails :
  afternoon_emails + evening_emails = 13 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_and_evening_emails_l1475_147502


namespace NUMINAMATH_CALUDE_instantaneous_rate_of_change_l1475_147550

/-- Given a curve y = x^2 + 2x, prove that if the instantaneous rate of change
    at a point M is 6, then the coordinates of point M are (2, 8). -/
theorem instantaneous_rate_of_change (x y : ℝ) : 
  y = x^2 + 2*x →                             -- Curve equation
  (2*x + 2 : ℝ) = 6 →                         -- Instantaneous rate of change is 6
  (x, y) = (2, 8) :=                          -- Coordinates of point M
by sorry

end NUMINAMATH_CALUDE_instantaneous_rate_of_change_l1475_147550


namespace NUMINAMATH_CALUDE_cody_spent_25_tickets_on_beanie_l1475_147539

/-- The number of tickets Cody spent on the beanie -/
def tickets_spent_on_beanie (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

/-- Proof that Cody spent 25 tickets on the beanie -/
theorem cody_spent_25_tickets_on_beanie :
  tickets_spent_on_beanie 49 6 30 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cody_spent_25_tickets_on_beanie_l1475_147539


namespace NUMINAMATH_CALUDE_sequence_properties_l1475_147518

def S (n : ℕ) : ℝ := -n^2 + 7*n + 1

def a (n : ℕ) : ℝ :=
  if n = 1 then 7
  else -2*n + 8

theorem sequence_properties :
  (∀ n > 4, a n < 0) ∧
  (∀ n : ℕ, n ≠ 0 → S n ≤ S 3 ∧ S n ≤ S 4) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1475_147518


namespace NUMINAMATH_CALUDE_hardware_store_lcm_l1475_147531

theorem hardware_store_lcm : Nat.lcm 13 (Nat.lcm 19 (Nat.lcm 8 (Nat.lcm 11 (Nat.lcm 17 23)))) = 772616 := by
  sorry

end NUMINAMATH_CALUDE_hardware_store_lcm_l1475_147531


namespace NUMINAMATH_CALUDE_probability_for_given_scenario_l1475_147588

/-- The probability that at least 4 people stay for the entire basketball game -/
def probability_at_least_4_stay (total_people : ℕ) (certain_stay : ℕ) (uncertain_stay : ℕ) 
  (prob_uncertain_stay : ℚ) : ℚ :=
  sorry

/-- Theorem stating the probability for the specific scenario -/
theorem probability_for_given_scenario : 
  probability_at_least_4_stay 8 3 5 (1/3) = 401/243 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_given_scenario_l1475_147588


namespace NUMINAMATH_CALUDE_range_of_a_l1475_147534

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1475_147534


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1475_147538

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence {a_n}, if a_3 + a_4 + a_5 + a_6 + a_7 = 25, then a_2 + a_8 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (a 3 + a 4 + a 5 + a 6 + a 7 = 25) → (a 2 + a 8 = 10) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1475_147538


namespace NUMINAMATH_CALUDE_mistaken_addition_l1475_147598

theorem mistaken_addition (N : ℤ) : (41 - N = 12) → (41 + N = 70) := by
  sorry

end NUMINAMATH_CALUDE_mistaken_addition_l1475_147598


namespace NUMINAMATH_CALUDE_find_z_when_y_is_6_l1475_147517

-- Define the direct variation relationship
def varies_directly (y z : ℝ) : Prop := ∃ k : ℝ, y^3 = k * z^(1/3)

-- State the theorem
theorem find_z_when_y_is_6 (y z : ℝ) (h1 : varies_directly y z) (h2 : y = 3 ∧ z = 8) :
  y = 6 → z = 4096 := by
  sorry

end NUMINAMATH_CALUDE_find_z_when_y_is_6_l1475_147517


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l1475_147541

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3742) % 17 = 1578 % 17 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3742) % 17 = 1578 % 17 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l1475_147541


namespace NUMINAMATH_CALUDE_max_product_of_functions_l1475_147515

theorem max_product_of_functions (f h : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.Icc (-5) 3) 
  (hh : ∀ x, h x ∈ Set.Icc (-3) 4) : 
  (⨆ x, f x * h x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_functions_l1475_147515


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l1475_147586

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → m ≤ n :=
by
  use 62
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l1475_147586


namespace NUMINAMATH_CALUDE_semicircle_problem_l1475_147505

/-- Given a large semicircle with diameter D and N congruent small semicircles
    fitting exactly on its diameter, if the ratio of the combined area of the
    small semicircles to the area of the large semicircle not covered by the
    small semicircles is 1:10, then N = 11. -/
theorem semicircle_problem (D : ℝ) (N : ℕ) (h : N > 0) :
  let r := D / (2 * N)
  let A := N * π * r^2 / 2
  let B := π * (N * r)^2 / 2 - A
  A / B = 1 / 10 → N = 11 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_problem_l1475_147505


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1475_147574

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 : ℝ) ^ (3 * x) = Real.sqrt 125 ∧ x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1475_147574


namespace NUMINAMATH_CALUDE_swim_club_prep_course_count_l1475_147556

/-- Represents a swim club with members, some of whom have passed a lifesaving test
    and some of whom have taken a preparatory course. -/
structure SwimClub where
  totalMembers : ℕ
  passedTest : ℕ
  notPassedNotTakenCourse : ℕ

/-- Calculates the number of members who have taken the preparatory course
    but not passed the test in a given swim club. -/
def membersInPreparatoryNotPassed (club : SwimClub) : ℕ :=
  club.totalMembers - club.passedTest - club.notPassedNotTakenCourse

/-- Theorem stating that in a swim club with 50 members, where 30% have passed
    the lifesaving test and 30 of those who haven't passed haven't taken the
    preparatory course, the number of members who have taken the preparatory
    course but not passed the test is 5. -/
theorem swim_club_prep_course_count :
  let club : SwimClub := {
    totalMembers := 50,
    passedTest := 15,  -- 30% of 50
    notPassedNotTakenCourse := 30
  }
  membersInPreparatoryNotPassed club = 5 := by
  sorry


end NUMINAMATH_CALUDE_swim_club_prep_course_count_l1475_147556


namespace NUMINAMATH_CALUDE_three_digit_sum_27_l1475_147537

theorem three_digit_sum_27 : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n / 100 + (n / 10) % 10 + n % 10 = 27) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_27_l1475_147537


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_l1475_147548

/-- Represents a caterer's pricing structure -/
structure Caterer where
  basic_fee : ℕ
  per_person : ℕ

/-- Calculates the total cost for a given number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.basic_fee + c.per_person * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { basic_fee := 150, per_person := 18 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { basic_fee := 250, per_person := 15 }

/-- Theorem stating the minimum number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper : 
  (∀ n : ℕ, n ≥ 34 → total_cost caterer2 n < total_cost caterer1 n) ∧
  (∀ n : ℕ, n < 34 → total_cost caterer2 n ≥ total_cost caterer1 n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_l1475_147548
