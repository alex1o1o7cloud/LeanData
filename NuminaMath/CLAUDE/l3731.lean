import Mathlib

namespace NUMINAMATH_CALUDE_max_power_of_15_l3731_373171

-- Define pow function
def pow (n : ℕ) : ℕ := sorry

-- Define the product of pow(n) from 2 to 2200
def product_pow : ℕ := sorry

-- Theorem statement
theorem max_power_of_15 :
  (∀ m : ℕ, m > 10 → ¬(pow (15^m) ∣ product_pow)) ∧
  (pow (15^10) ∣ product_pow) :=
sorry

end NUMINAMATH_CALUDE_max_power_of_15_l3731_373171


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3731_373127

theorem unique_triple_solution : 
  ∃! (p q n : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 0 ∧ q > 0 ∧ n > 0 ∧
    p * (p + 3) + q * (q + 3) = n * (n + 3) ∧
    p = 3 ∧ q = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3731_373127


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3731_373102

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 * x₂ + x₁ * x₂^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3731_373102


namespace NUMINAMATH_CALUDE_line_equations_specific_line_equations_l3731_373194

/-- Definition of a line passing through two points -/
def Line (A B : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)}

theorem line_equations (A B : ℝ × ℝ) (h : A ≠ B) :
  let l := Line A B
  -- Two-point form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ (y - B.2) / (A.2 - B.2) = (x - B.1) / (A.1 - B.1) ∧
  -- Point-slope form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y - B.2 = ((A.2 - B.2) / (A.1 - B.1)) * (x - B.1) ∧
  -- Slope-intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = ((A.2 - B.2) / (A.1 - B.1)) * x + (B.2 - ((A.2 - B.2) / (A.1 - B.1)) * B.1) ∧
  -- Intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ x / (-B.1 + (A.1 * B.2 - A.2 * B.1) / (A.2 - B.2)) + 
                             y / ((A.1 * B.2 - A.2 * B.1) / (A.1 - B.1)) = 1 :=
by
  sorry

-- Specific instance for points A(-2, 3) and B(4, -1)
theorem specific_line_equations :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (4, -1)
  let l := Line A B
  -- Two-point form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ (y + 1) / 4 = (x - 4) / (-6) ∧
  -- Point-slope form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y + 1 = -2/3 * (x - 4) ∧
  -- Slope-intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = -2/3 * x + 5/3 ∧
  -- Intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ x / (5/2) + y / (5/3) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equations_specific_line_equations_l3731_373194


namespace NUMINAMATH_CALUDE_locus_is_parabola_l3731_373176

-- Define the fixed point M
def M : ℝ × ℝ := (1, 0)

-- Define the fixed line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the locus of points P
def locus : Set (ℝ × ℝ) := {P : ℝ × ℝ | ∃ B ∈ l, dist P M = dist P B}

-- Theorem statement
theorem locus_is_parabola : 
  ∃ a b c : ℝ, locus = {P : ℝ × ℝ | P.2 = a * P.1^2 + b * P.1 + c} := by
  sorry

end NUMINAMATH_CALUDE_locus_is_parabola_l3731_373176


namespace NUMINAMATH_CALUDE_equation_solution_l3731_373172

theorem equation_solution : 
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3731_373172


namespace NUMINAMATH_CALUDE_sunflower_majority_on_tuesday_l3731_373199

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflower_seeds : Real
  other_seeds : Real

/-- Calculates the next day's feeder state -/
def next_day_state (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflower_seeds := state.sunflower_seeds * 0.7 + 0.2,
    other_seeds := state.other_seeds * 0.4 + 0.3 }

/-- Initial state of the feeder on Sunday -/
def initial_state : FeederState :=
  { day := 1,
    sunflower_seeds := 0.4,
    other_seeds := 0.6 }

/-- Theorem stating that on Day 3 (Tuesday), sunflower seeds make up more than half of the total seeds -/
theorem sunflower_majority_on_tuesday :
  let state₃ := next_day_state (next_day_state initial_state)
  state₃.sunflower_seeds > (state₃.sunflower_seeds + state₃.other_seeds) / 2 := by
  sorry


end NUMINAMATH_CALUDE_sunflower_majority_on_tuesday_l3731_373199


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3731_373141

theorem multiplication_addition_equality : 42 * 25 + 58 * 42 = 3486 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3731_373141


namespace NUMINAMATH_CALUDE_father_age_at_second_son_birth_l3731_373193

/-- Represents the ages of a father and his three sons -/
structure FamilyAges where
  father : ℕ
  son1 : ℕ
  son2 : ℕ
  son3 : ℕ

/-- The problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.son1 = ages.son2 + ages.son3 ∧
  ages.father * ages.son1 * ages.son2 * ages.son3 = 27090

/-- The main theorem -/
theorem father_age_at_second_son_birth (ages : FamilyAges) 
  (h : satisfiesConditions ages) : 
  ages.father - ages.son2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_father_age_at_second_son_birth_l3731_373193


namespace NUMINAMATH_CALUDE_vans_needed_l3731_373159

def van_capacity : ℕ := 5
def num_students : ℕ := 12
def num_adults : ℕ := 3

theorem vans_needed : 
  (num_students + num_adults + van_capacity - 1) / van_capacity = 3 := by
sorry

end NUMINAMATH_CALUDE_vans_needed_l3731_373159


namespace NUMINAMATH_CALUDE_tan_plus_4sin_30_deg_l3731_373188

theorem tan_plus_4sin_30_deg :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  let tan_30 : ℝ := sin_30 / cos_30
  tan_30 + 4 * sin_30 = (Real.sqrt 3 + 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_4sin_30_deg_l3731_373188


namespace NUMINAMATH_CALUDE_problem_1_l3731_373142

theorem problem_1 : 2023 * 2023 - 2024 * 2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3731_373142


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l3731_373185

theorem arithmetic_sequence_term_count 
  (a₁ aₙ d : ℤ) 
  (h₁ : a₁ = -25)
  (h₂ : aₙ = 96)
  (h₃ : d = 7)
  (h₄ : aₙ = a₁ + (n - 1) * d)
  : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l3731_373185


namespace NUMINAMATH_CALUDE_go_game_draw_probability_l3731_373198

theorem go_game_draw_probability 
  (p_not_lose : ℝ) 
  (p_win : ℝ) 
  (h1 : p_not_lose = 0.6) 
  (h2 : p_win = 0.5) : 
  p_not_lose - p_win = 0.1 := by
sorry

end NUMINAMATH_CALUDE_go_game_draw_probability_l3731_373198


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3731_373130

def total_crayons : ℕ := 15
def red_crayons : ℕ := 3
def non_red_crayons : ℕ := total_crayons - red_crayons
def selection_size : ℕ := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem crayon_selection_theorem :
  choose total_crayons selection_size - choose non_red_crayons selection_size = 2211 :=
by sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3731_373130


namespace NUMINAMATH_CALUDE_gcd_1426_1643_l3731_373174

theorem gcd_1426_1643 : Nat.gcd 1426 1643 = 31 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1426_1643_l3731_373174


namespace NUMINAMATH_CALUDE_sangita_flying_months_l3731_373117

/-- Calculates the number of months needed to complete flying hours for a pilot certificate. -/
def months_to_complete_flying (total_required : ℕ) (day_completed : ℕ) (night_completed : ℕ) (cross_country_completed : ℕ) (monthly_goal : ℕ) : ℕ :=
  let total_completed := day_completed + night_completed + cross_country_completed
  let remaining_hours := total_required - total_completed
  (remaining_hours + monthly_goal - 1) / monthly_goal

/-- Proves that Sangita needs 6 months to complete her flying hours. -/
theorem sangita_flying_months :
  months_to_complete_flying 1500 50 9 121 220 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sangita_flying_months_l3731_373117


namespace NUMINAMATH_CALUDE_jeromes_contacts_l3731_373196

theorem jeromes_contacts (classmates : ℕ) (total_contacts : ℕ) :
  classmates = 20 →
  total_contacts = 33 →
  3 = total_contacts - (classmates + classmates / 2) :=
by sorry

end NUMINAMATH_CALUDE_jeromes_contacts_l3731_373196


namespace NUMINAMATH_CALUDE_power_multiplication_l3731_373125

theorem power_multiplication (a : ℝ) : 4 * a^2 * a = 4 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3731_373125


namespace NUMINAMATH_CALUDE_negation_existence_statement_l3731_373122

theorem negation_existence_statement :
  (¬ ∃ x : ℝ, x < -1 ∧ x^2 ≥ 1) ↔ (∀ x : ℝ, x < -1 → x^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l3731_373122


namespace NUMINAMATH_CALUDE_homework_difference_l3731_373105

/-- The number of pages of reading homework -/
def reading_pages : ℕ := 6

/-- The number of pages of math homework -/
def math_pages : ℕ := 10

/-- The number of pages of science homework -/
def science_pages : ℕ := 3

/-- The number of pages of history homework -/
def history_pages : ℕ := 5

/-- The theorem states that the difference between math homework pages and the sum of reading, science, and history homework pages is -4 -/
theorem homework_difference : 
  (math_pages : ℤ) - (reading_pages + science_pages + history_pages : ℤ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_l3731_373105


namespace NUMINAMATH_CALUDE_seeds_in_first_plot_l3731_373144

/-- The number of seeds planted in the first plot -/
def seeds_first_plot : ℕ := sorry

/-- The number of seeds planted in the second plot -/
def seeds_second_plot : ℕ := 200

/-- The percentage of seeds that germinated in the first plot -/
def germination_rate_first : ℚ := 20 / 100

/-- The percentage of seeds that germinated in the second plot -/
def germination_rate_second : ℚ := 35 / 100

/-- The percentage of total seeds that germinated -/
def total_germination_rate : ℚ := 26 / 100

/-- Theorem stating that the number of seeds in the first plot is 300 -/
theorem seeds_in_first_plot :
  (seeds_first_plot : ℚ) * germination_rate_first + 
  (seeds_second_plot : ℚ) * germination_rate_second = 
  total_germination_rate * ((seeds_first_plot : ℚ) + seeds_second_plot) ∧
  seeds_first_plot = 300 := by sorry

end NUMINAMATH_CALUDE_seeds_in_first_plot_l3731_373144


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l3731_373184

/-- Given a company's yearly magazine subscription cost and desired budget cut,
    calculate the percentage reduction in the budget. -/
theorem magazine_budget_cut_percentage
  (original_cost : ℝ)
  (budget_cut : ℝ)
  (h_original_cost : original_cost = 940)
  (h_budget_cut : budget_cut = 611) :
  (budget_cut / original_cost) * 100 = 65 := by
sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l3731_373184


namespace NUMINAMATH_CALUDE_inequality_solution_l3731_373153

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3731_373153


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l3731_373126

theorem triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = Real.pi) (h5 : A ≤ B) (h6 : B ≤ C)
  (h7 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = Real.sqrt 3) :
  Real.sin B + Real.sin (2 * B) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l3731_373126


namespace NUMINAMATH_CALUDE_circle_rational_points_l3731_373169

theorem circle_rational_points :
  ∃ (S : Set (ℚ × ℚ)), Set.Infinite S ∧ ∀ (p : ℚ × ℚ), p ∈ S → p.1^2 + p.2^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_rational_points_l3731_373169


namespace NUMINAMATH_CALUDE_not_right_triangle_3_4_5_squared_l3731_373166

-- Define a function to check if three numbers can form a right triangle
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that 3^2, 4^2, 5^2 cannot form a right triangle
theorem not_right_triangle_3_4_5_squared :
  ¬ isRightTriangle (3^2) (4^2) (5^2) := by
  sorry


end NUMINAMATH_CALUDE_not_right_triangle_3_4_5_squared_l3731_373166


namespace NUMINAMATH_CALUDE_problem_statement_l3731_373140

theorem problem_statement :
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3) ∧
  (∀ (a : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a/2 ≥ x + 2*y + 2*z) ↔ (a ≤ 0 ∨ a ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3731_373140


namespace NUMINAMATH_CALUDE_four_tangent_circles_l3731_373155

-- Define a line in 2D space
def Line2D := (ℝ × ℝ) → Prop

-- Define a circle in 2D space
structure Circle2D where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency between a circle and a line
def isTangent (c : Circle2D) (l : Line2D) : Prop := sorry

-- Main theorem
theorem four_tangent_circles 
  (l1 l2 : Line2D) 
  (intersect : ∃ p : ℝ × ℝ, l1 p ∧ l2 p) 
  (r : ℝ) 
  (h : r > 0) : 
  ∃! (cs : Finset Circle2D), 
    cs.card = 4 ∧ 
    (∀ c ∈ cs, c.radius = r ∧ isTangent c l1 ∧ isTangent c l2) :=
sorry

end NUMINAMATH_CALUDE_four_tangent_circles_l3731_373155


namespace NUMINAMATH_CALUDE_half_sufficient_not_necessary_l3731_373134

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ → ℝ → Prop := λ x y => x + 2 * a * y - 1 = 0
  line2 : ℝ → ℝ → Prop := λ x y => (a - 1) * x - a * y - 1 = 0

/-- The lines are parallel -/
def are_parallel (l : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l.line1 x y ↔ l.line2 (k * x) (k * y)

/-- The statement that a = 1/2 is sufficient but not necessary for the lines to be parallel -/
theorem half_sufficient_not_necessary :
  (∃ l : TwoLines, l.a = 1/2 ∧ ¬are_parallel l) ∧
  (∃ l : TwoLines, l.a ≠ 1/2 ∧ are_parallel l) ∧
  (∀ l : TwoLines, l.a = 1/2 → are_parallel l) :=
sorry

end NUMINAMATH_CALUDE_half_sufficient_not_necessary_l3731_373134


namespace NUMINAMATH_CALUDE_min_packs_for_135_cans_l3731_373192

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (s : PackSize) : Nat :=
  match s with
  | .small => 8
  | .medium => 15
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : Nat :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Checks if a pack combination is valid for the target number of cans -/
def isValidCombination (c : PackCombination) (target : Nat) : Prop :=
  totalCans c = target

/-- Counts the total number of packs in a combination -/
def totalPacks (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- The main theorem: prove that the minimum number of packs to get 135 cans is 5 -/
theorem min_packs_for_135_cans :
  ∃ (c : PackCombination),
    isValidCombination c 135 ∧
    totalPacks c = 5 ∧
    (∀ (c' : PackCombination), isValidCombination c' 135 → totalPacks c ≤ totalPacks c') :=
by sorry

end NUMINAMATH_CALUDE_min_packs_for_135_cans_l3731_373192


namespace NUMINAMATH_CALUDE_calculation_proof_l3731_373146

theorem calculation_proof : (4.5 - 1.23) * 2.1 = 6.867 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3731_373146


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3731_373178

/-- A trinomial ax^2 + bxy + cy^2 is a perfect square if and only if b^2 = 4ac -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c

/-- The value of k for which 9x^2 - kxy + 4y^2 is a perfect square trinomial -/
theorem perfect_square_trinomial_k : 
  ∃ (k : ℝ), is_perfect_square_trinomial 9 (-k) 4 ∧ (k = 12 ∨ k = -12) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3731_373178


namespace NUMINAMATH_CALUDE_f_value_at_3_l3731_373138

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 2

-- State the theorem
theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l3731_373138


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l3731_373110

theorem smallest_distance_between_circles (z w : ℂ) :
  Complex.abs (z - (2 + 4*I)) = 2 →
  Complex.abs (w - (5 + 6*I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 4*I)) = 2 →
      Complex.abs (w' - (5 + 6*I)) = 4 →
      Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l3731_373110


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_removal_l3731_373124

theorem smallest_n_for_candy_removal : ∃ n : ℕ, 
  (∀ k : ℕ, k > 0 → k * (k + 1) / 2 ≥ 64 → n ≤ k) ∧ 
  n * (n + 1) / 2 ≥ 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_removal_l3731_373124


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3731_373151

theorem chocolate_distribution (minho : ℕ) (taemin kibum : ℕ) : 
  taemin = 5 * minho →
  kibum = 3 * minho →
  taemin + kibum = 160 →
  minho = 20 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3731_373151


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3731_373175

open Set Real

theorem condition_necessary_not_sufficient :
  let A : Set ℝ := {x | 0 < x ∧ x < 3}
  let B : Set ℝ := {x | log (x - 2) < 0}
  B ⊂ A ∧ B ≠ A := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3731_373175


namespace NUMINAMATH_CALUDE_sophomore_sample_size_l3731_373128

/-- Represents the number of students selected in a stratified sample -/
def stratifiedSample (totalPopulation : ℕ) (sampleSize : ℕ) (strataSize : ℕ) : ℕ :=
  (strataSize * sampleSize) / totalPopulation

/-- The problem statement -/
theorem sophomore_sample_size :
  let totalStudents : ℕ := 2800
  let sophomores : ℕ := 930
  let sampleSize : ℕ := 280
  stratifiedSample totalStudents sampleSize sophomores = 93 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_size_l3731_373128


namespace NUMINAMATH_CALUDE_appropriate_mass_units_l3731_373154

-- Define the mass units
inductive MassUnit
| Gram
| Ton
| Kilogram

-- Define a structure for an item with its mass value
structure MassItem where
  value : ℕ
  unit : MassUnit

-- Define the function to check if a mass unit is appropriate for a given item
def isAppropriateUnit (item : MassItem) : Prop :=
  match item with
  | ⟨1, MassUnit.Gram⟩ => true     -- Peanut kernel
  | ⟨8, MassUnit.Ton⟩ => true      -- Truck loading capacity
  | ⟨30, MassUnit.Kilogram⟩ => true -- Xiao Ming's weight
  | ⟨580, MassUnit.Gram⟩ => true   -- Basketball mass
  | _ => false

-- Theorem statement
theorem appropriate_mass_units :
  let peanut := MassItem.mk 1 MassUnit.Gram
  let truck := MassItem.mk 8 MassUnit.Ton
  let xiaoMing := MassItem.mk 30 MassUnit.Kilogram
  let basketball := MassItem.mk 580 MassUnit.Gram
  isAppropriateUnit peanut ∧
  isAppropriateUnit truck ∧
  isAppropriateUnit xiaoMing ∧
  isAppropriateUnit basketball :=
by sorry


end NUMINAMATH_CALUDE_appropriate_mass_units_l3731_373154


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3731_373189

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4)) ∧ 
  (∃ a b : ℝ, (a + b > 4 ∧ a * b > 4) ∧ ¬(a > 2 ∧ b > 2)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3731_373189


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3731_373164

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) : 
  total_workers = 12 → 
  num_technicians = 7 → 
  avg_salary_technicians = 12000 → 
  avg_salary_rest = 6000 → 
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest) / total_workers = 9500 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3731_373164


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_one_l3731_373108

theorem sqrt_difference_equals_one : 
  Real.sqrt 9 - Real.sqrt ((-2)^2) = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_one_l3731_373108


namespace NUMINAMATH_CALUDE_james_tshirt_cost_l3731_373120

def calculate_total_cost (num_shirts : ℕ) (discount_rate : ℚ) (original_price : ℚ) : ℚ :=
  num_shirts * (original_price * (1 - discount_rate))

theorem james_tshirt_cost :
  calculate_total_cost 6 (1/2) 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_tshirt_cost_l3731_373120


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l3731_373101

/-- The set of positive integers -/
def PositiveIntegers := {n : ℕ | n > 0}

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction := PositiveIntegers → PositiveIntegers

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : PositiveIntegers,
    Nat.gcd (f m) n + Nat.lcm m (f n) = Nat.gcd m (f n) + Nat.lcm (f m) n

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∃! f : PositiveIntegerFunction, SatisfiesEquation f ∧ (∀ n, f n = n) :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l3731_373101


namespace NUMINAMATH_CALUDE_child_b_share_l3731_373179

theorem child_b_share (total_money : ℝ) (ratios : Fin 5 → ℝ) : 
  total_money = 10800 ∧ 
  ratios 0 = 0.5 ∧ 
  ratios 1 = 1.5 ∧ 
  ratios 2 = 2.25 ∧ 
  ratios 3 = 3.5 ∧ 
  ratios 4 = 4.25 → 
  (ratios 1 * total_money) / (ratios 0 + ratios 1 + ratios 2 + ratios 3 + ratios 4) = 1350 := by
sorry

end NUMINAMATH_CALUDE_child_b_share_l3731_373179


namespace NUMINAMATH_CALUDE_divisibility_of_quadratic_l3731_373129

theorem divisibility_of_quadratic (n : ℤ) : 
  (∀ n, ¬(8 ∣ (n^2 - 6*n - 2))) ∧ 
  (∀ n, ¬(9 ∣ (n^2 - 6*n - 2))) ∧ 
  (∀ n, (11 ∣ (n^2 - 6*n - 2)) ↔ (n ≡ 3 [ZMOD 11])) ∧ 
  (∀ n, ¬(121 ∣ (n^2 - 6*n - 2))) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_quadratic_l3731_373129


namespace NUMINAMATH_CALUDE_even_function_range_theorem_l3731_373177

-- Define an even function f on ℝ
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_range_theorem (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_even : isEvenFunction f) 
  (h_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x, 2 * f x + x * f' x < 2) :
  {x : ℝ | x^2 * f x - 4 * f 2 < x^2 - 4} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_even_function_range_theorem_l3731_373177


namespace NUMINAMATH_CALUDE_classroom_capacity_l3731_373114

theorem classroom_capacity (total_students : ℕ) (num_classrooms : ℕ) 
  (h1 : total_students = 390) (h2 : num_classrooms = 13) :
  total_students / num_classrooms = 30 := by
  sorry

end NUMINAMATH_CALUDE_classroom_capacity_l3731_373114


namespace NUMINAMATH_CALUDE_chris_age_is_17_l3731_373123

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℕ
  ben : ℕ
  chris : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Six years ago, Chris was the same age as Amy is now
  ages.chris - 6 = ages.amy ∧
  -- In 3 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 3 = (3 * (ages.amy + 3)) / 4

/-- The theorem stating that Chris's age is 17 -/
theorem chris_age_is_17 :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.chris = 17 := by
  sorry


end NUMINAMATH_CALUDE_chris_age_is_17_l3731_373123


namespace NUMINAMATH_CALUDE_valid_speaking_orders_eq_600_l3731_373149

/-- The number of students in the class --/
def total_students : ℕ := 7

/-- The number of students to be selected for speaking --/
def selected_speakers : ℕ := 4

/-- The number of special students (A and B) --/
def special_students : ℕ := 2

/-- Function to calculate the number of valid speaking orders --/
def valid_speaking_orders : ℕ :=
  let one_special := special_students * (total_students - special_students).choose (selected_speakers - 1) * (selected_speakers).factorial
  let both_special := special_students.choose 2 * (total_students - special_students).choose (selected_speakers - 2) * (selected_speakers).factorial
  let adjacent := special_students.choose 2 * (total_students - special_students).choose (selected_speakers - 2) * (selected_speakers - 1).factorial * 2
  one_special + both_special - adjacent

/-- Theorem stating that the number of valid speaking orders is 600 --/
theorem valid_speaking_orders_eq_600 : valid_speaking_orders = 600 := by
  sorry

end NUMINAMATH_CALUDE_valid_speaking_orders_eq_600_l3731_373149


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l3731_373163

theorem arithmetic_progression_first_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n + 5) →  -- Common difference is 5
    a 21 = 103 →                 -- 21st term is 103
    a 1 = 3 :=                   -- First term is 3
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l3731_373163


namespace NUMINAMATH_CALUDE_distribute_problems_l3731_373167

theorem distribute_problems (n m : ℕ) (hn : n = 7) (hm : m = 15) :
  (Nat.choose m n) * (Nat.factorial n) = 32432400 := by
  sorry

end NUMINAMATH_CALUDE_distribute_problems_l3731_373167


namespace NUMINAMATH_CALUDE_john_vacation_expenses_l3731_373116

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  base8_to_base10 savings - ticket_cost

theorem john_vacation_expenses :
  remaining_money 5373 1500 = 1311 := by sorry

end NUMINAMATH_CALUDE_john_vacation_expenses_l3731_373116


namespace NUMINAMATH_CALUDE_birch_tree_spacing_probability_l3731_373137

def total_trees : ℕ := 15
def pine_trees : ℕ := 4
def maple_trees : ℕ := 5
def birch_trees : ℕ := 6

theorem birch_tree_spacing_probability :
  let non_birch_trees := pine_trees + maple_trees
  let total_arrangements := (total_trees.choose birch_trees : ℚ)
  let valid_arrangements := ((non_birch_trees + 1).choose birch_trees : ℚ)
  valid_arrangements / total_arrangements = 2 / 95 := by
sorry

end NUMINAMATH_CALUDE_birch_tree_spacing_probability_l3731_373137


namespace NUMINAMATH_CALUDE_polynomial_transformation_c_values_l3731_373107

/-- The number of distinct possible values of c in a polynomial transformation. -/
theorem polynomial_transformation_c_values
  (a b r s t : ℂ)
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_transform : ∀ z, (z - r) * (z - s) * (z - t) =
                      ((a * z + b) - c * r) * ((a * z + b) - c * s) * ((a * z + b) - c * t)) :
  ∃! (values : Finset ℂ), values.card = 4 ∧ ∀ c, c ∈ values ↔ 
    ∃ z, (z - r) * (z - s) * (z - t) =
         ((a * z + b) - c * r) * ((a * z + b) - c * s) * ((a * z + b) - c * t) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_c_values_l3731_373107


namespace NUMINAMATH_CALUDE_movie_ratio_is_half_l3731_373190

/-- The ratio of movies Theresa saw in 2009 to movies Timothy saw in 2009 -/
def movie_ratio (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℚ :=
  theresa_2009 / timothy_2009

theorem movie_ratio_is_half :
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2009 = 24 →
    timothy_2010 = timothy_2009 + 7 →
    theresa_2010 = 2 * timothy_2010 →
    timothy_2009 + theresa_2009 + timothy_2010 + theresa_2010 = 129 →
    movie_ratio timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_movie_ratio_is_half_l3731_373190


namespace NUMINAMATH_CALUDE_allocation_methods_six_individuals_l3731_373111

/-- The number of ways to allocate 6 individuals into 2 rooms -/
def allocation_methods (n : ℕ) : ℕ → ℕ
  | 1 => Nat.choose n 3  -- Exactly 3 per room
  | 2 => Nat.choose n 1 * Nat.choose (n-1) (n-1) +  -- 1 in first room
         Nat.choose n 2 * Nat.choose (n-2) (n-2) +  -- 2 in first room
         Nat.choose n 3 * Nat.choose (n-3) (n-3) +  -- 3 in first room
         Nat.choose n 4 * Nat.choose (n-4) (n-4) +  -- 4 in first room
         Nat.choose n 5 * Nat.choose (n-5) (n-5)    -- 5 in first room
  | _ => 0  -- For any other input

theorem allocation_methods_six_individuals :
  allocation_methods 6 1 = 20 ∧ allocation_methods 6 2 = 62 := by
  sorry

#eval allocation_methods 6 1  -- Should output 20
#eval allocation_methods 6 2  -- Should output 62

end NUMINAMATH_CALUDE_allocation_methods_six_individuals_l3731_373111


namespace NUMINAMATH_CALUDE_speed_in_still_water_problem_l3731_373121

/-- Calculates the speed in still water given the downstream speed and current speed. -/
def speed_in_still_water (downstream_speed current_speed : ℝ) : ℝ :=
  downstream_speed - current_speed

/-- Theorem: Given the conditions from the problem, the speed in still water is 30 kmph. -/
theorem speed_in_still_water_problem :
  let downstream_distance : ℝ := 0.24 -- 240 meters in km
  let downstream_time : ℝ := 24 / 3600 -- 24 seconds in hours
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let current_speed : ℝ := 6
  speed_in_still_water downstream_speed current_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_problem_l3731_373121


namespace NUMINAMATH_CALUDE_bonus_implication_l3731_373112

-- Define the universe of discourse
variable (Employee : Type)

-- Define the predicates
variable (completes_all_projects : Employee → Prop)
variable (receives_bonus : Employee → Prop)

-- Mr. Thompson's statement
variable (thompson_statement : ∀ (e : Employee), completes_all_projects e → receives_bonus e)

-- Theorem to prove
theorem bonus_implication :
  ∀ (e : Employee), ¬(receives_bonus e) → ¬(completes_all_projects e) := by
  sorry

end NUMINAMATH_CALUDE_bonus_implication_l3731_373112


namespace NUMINAMATH_CALUDE_problem_solution_l3731_373132

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/2 = 6*y) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3731_373132


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l3731_373135

theorem point_movement_on_number_line :
  ∀ (a b c : ℝ),
    b = a - 3 →
    c = b + 5 →
    c = 1 →
    a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l3731_373135


namespace NUMINAMATH_CALUDE_not_sum_of_six_odd_squares_l3731_373106

theorem not_sum_of_six_odd_squares (n : ℕ) : n = 1986 → ¬ ∃ (a b c d e f : ℕ), 
  (∃ (k₁ k₂ k₃ k₄ k₅ k₆ : ℕ), 
    a = 2 * k₁ + 1 ∧ 
    b = 2 * k₂ + 1 ∧ 
    c = 2 * k₃ + 1 ∧ 
    d = 2 * k₄ + 1 ∧ 
    e = 2 * k₅ + 1 ∧ 
    f = 2 * k₆ + 1) ∧ 
  n = a^2 + b^2 + c^2 + d^2 + e^2 + f^2 :=
by sorry

end NUMINAMATH_CALUDE_not_sum_of_six_odd_squares_l3731_373106


namespace NUMINAMATH_CALUDE_sin_cos_sum_21_39_l3731_373161

theorem sin_cos_sum_21_39 : 
  Real.sin (21 * π / 180) * Real.cos (39 * π / 180) + 
  Real.cos (21 * π / 180) * Real.sin (39 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_21_39_l3731_373161


namespace NUMINAMATH_CALUDE_heights_equal_on_equal_sides_l3731_373104

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define the relevant parts for our proof
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  height1 : ℝ
  height2 : ℝ
  is_isosceles : side1 = side2

-- Theorem statement
theorem heights_equal_on_equal_sides (t : IsoscelesTriangle) : 
  t.height1 = t.height2 := by
  sorry

end NUMINAMATH_CALUDE_heights_equal_on_equal_sides_l3731_373104


namespace NUMINAMATH_CALUDE_female_democrats_count_l3731_373109

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 990 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total / 3 : ℚ) →
  (female / 2 : ℕ) = 275 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3731_373109


namespace NUMINAMATH_CALUDE_fifty_factorial_trailing_zeros_l3731_373170

/-- The number of trailing zeros in n! -/
def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- 50! has exactly 12 trailing zeros -/
theorem fifty_factorial_trailing_zeros : trailing_zeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fifty_factorial_trailing_zeros_l3731_373170


namespace NUMINAMATH_CALUDE_circle_land_diagram_value_l3731_373148

/-- Represents a digit with circles in Circle Land -/
structure CircleDigit where
  digit : Nat
  circles : Nat

/-- Calculates the value of a CircleDigit -/
def circleValue (cd : CircleDigit) : Nat :=
  cd.digit * (10 ^ cd.circles)

/-- Represents a number in Circle Land -/
def CircleLandNumber (cds : List CircleDigit) : Nat :=
  cds.map circleValue |>.sum

/-- The specific diagram given in the problem -/
def problemDiagram : List CircleDigit :=
  [⟨3, 4⟩, ⟨1, 2⟩, ⟨5, 0⟩]

theorem circle_land_diagram_value :
  CircleLandNumber problemDiagram = 30105 := by
  sorry

end NUMINAMATH_CALUDE_circle_land_diagram_value_l3731_373148


namespace NUMINAMATH_CALUDE_min_value_theorem_l3731_373131

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y - x*y = 0) :
  ∃ (min : ℝ), min = 5 + 2 * Real.sqrt 6 ∧ ∀ (z : ℝ), z = 3*x + 2*y → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3731_373131


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3731_373103

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 ↔ x - a > 2 ∧ b - 2*x > 0) →
  (a + b)^2021 = -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3731_373103


namespace NUMINAMATH_CALUDE_root_magnitude_bound_l3731_373145

theorem root_magnitude_bound (p : ℝ) (r₁ r₂ : ℝ) 
  (h_distinct : r₁ ≠ r₂)
  (h_root₁ : r₁^2 + p*r₁ - 12 = 0)
  (h_root₂ : r₂^2 + p*r₂ - 12 = 0) :
  abs r₁ > 3 ∨ abs r₂ > 3 := by
sorry

end NUMINAMATH_CALUDE_root_magnitude_bound_l3731_373145


namespace NUMINAMATH_CALUDE_soccer_team_starters_l3731_373191

theorem soccer_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) 
  (h1 : total_players = 16) 
  (h2 : quadruplets = 4) 
  (h3 : starters = 7) : 
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l3731_373191


namespace NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l3731_373162

theorem integral_sin_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1)..1, (Real.sin x + Real.sqrt (1 - x^2)) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l3731_373162


namespace NUMINAMATH_CALUDE_triangle_with_unequal_angle_l3731_373139

theorem triangle_with_unequal_angle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal
  c = a - 10 →       -- Third angle is 10° less than the others
  c = 53.33 :=       -- Measure of the smallest angle
by sorry

end NUMINAMATH_CALUDE_triangle_with_unequal_angle_l3731_373139


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l3731_373181

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 2

-- State the theorem
theorem monotonic_function_a_range :
  (∀ a : ℝ, Monotone (f a) → a ∈ Set.Icc (- Real.sqrt 3) (Real.sqrt 3)) ∧
  (∀ a : ℝ, a ∈ Set.Icc (- Real.sqrt 3) (Real.sqrt 3) → Monotone (f a)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l3731_373181


namespace NUMINAMATH_CALUDE_simplify_expression_l3731_373119

theorem simplify_expression (b : ℝ) : (1:ℝ) * (3*b) * (5*b^2) * (7*b^3) * (9*b^4) = 945 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3731_373119


namespace NUMINAMATH_CALUDE_first_ball_odd_given_two_odd_one_even_l3731_373133

/-- The probability of selecting an odd-numbered ball from a box of 100 balls numbered 1 to 100 -/
def prob_odd_ball : ℚ := 1/2

/-- The probability of selecting an even-numbered ball from a box of 100 balls numbered 1 to 100 -/
def prob_even_ball : ℚ := 1/2

/-- The probability of selecting two odd-numbered balls and one even-numbered ball in any order when selecting 3 balls with replacement -/
def prob_two_odd_one_even : ℚ := 3 * prob_odd_ball * prob_odd_ball * prob_even_ball

theorem first_ball_odd_given_two_odd_one_even :
  let prob_first_odd := prob_odd_ball * (prob_odd_ball * prob_even_ball + prob_even_ball * prob_odd_ball)
  prob_first_odd / prob_two_odd_one_even = 1/4 := by sorry

end NUMINAMATH_CALUDE_first_ball_odd_given_two_odd_one_even_l3731_373133


namespace NUMINAMATH_CALUDE_right_handed_players_count_l3731_373115

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 52)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + ((total_players - throwers) * 2 / 3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l3731_373115


namespace NUMINAMATH_CALUDE_exists_term_divisible_by_2006_l3731_373197

theorem exists_term_divisible_by_2006 : ∃ n : ℤ, (2006 : ℤ) ∣ (n^3 - (2*n + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_exists_term_divisible_by_2006_l3731_373197


namespace NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l3731_373113

def dress_hours : List Nat := [15, 18, 20, 22, 24, 26, 28]
def weekly_pattern : List Nat := [5, 3, 6, 4]
def finalization_hours : Nat := 10

def total_sewing_hours : Nat := dress_hours.sum
def total_hours : Nat := total_sewing_hours + finalization_hours
def cycle_hours : Nat := weekly_pattern.sum

def weeks_to_complete : Nat :=
  let full_cycles := (total_hours + cycle_hours - 1) / cycle_hours
  full_cycles * 4 - 3

theorem bridesmaid_dresses_completion_time :
  weeks_to_complete = 37 := by sorry

end NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l3731_373113


namespace NUMINAMATH_CALUDE_min_three_digit_divisible_by_seven_l3731_373100

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def remove_middle_digit (n : ℕ) : ℕ :=
  (n / 100) * 10 + (n % 10)

theorem min_three_digit_divisible_by_seven :
  ∃ (N : ℕ),
    is_three_digit N ∧
    N % 7 = 0 ∧
    (remove_middle_digit N) % 7 = 0 ∧
    (∀ (M : ℕ), 
      is_three_digit M ∧ 
      M % 7 = 0 ∧ 
      (remove_middle_digit M) % 7 = 0 → 
      N ≤ M) ∧
    N = 154 := by
  sorry

end NUMINAMATH_CALUDE_min_three_digit_divisible_by_seven_l3731_373100


namespace NUMINAMATH_CALUDE_choose_four_captains_from_twelve_l3731_373150

theorem choose_four_captains_from_twelve (n : ℕ) (k : ℕ) : n = 12 ∧ k = 4 → Nat.choose n k = 990 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_captains_from_twelve_l3731_373150


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3731_373182

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3731_373182


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l3731_373173

/-- The number of valid 18-letter arrangements of 6 D's, 6 E's, and 6 F's -/
def valid_arrangements : ℕ :=
  Finset.sum (Finset.range 7) (fun m => (Nat.choose 6 m) ^ 3)

/-- Theorem stating the number of valid arrangements -/
theorem count_valid_arrangements :
  valid_arrangements =
    (Finset.sum (Finset.range 7) (fun m => (Nat.choose 6 m) ^ 3)) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l3731_373173


namespace NUMINAMATH_CALUDE_simplify_expression_l3731_373165

theorem simplify_expression (r : ℝ) : 180 * r - 88 * r = 92 * r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3731_373165


namespace NUMINAMATH_CALUDE_theater_seats_l3731_373186

theorem theater_seats (people_watching : ℕ) (empty_seats : ℕ) : 
  people_watching = 532 → empty_seats = 218 → people_watching + empty_seats = 750 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_l3731_373186


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3731_373156

/-- The perimeter of an equilateral triangle with an inscribed circle of radius 2 cm -/
theorem equilateral_triangle_perimeter (r : ℝ) (h : r = 2) :
  let a := 2 * r * Real.sqrt 3
  3 * a = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3731_373156


namespace NUMINAMATH_CALUDE_floor_e_squared_l3731_373157

theorem floor_e_squared : ⌊Real.exp 1 ^ 2⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_e_squared_l3731_373157


namespace NUMINAMATH_CALUDE_comic_books_triple_storybooks_l3731_373147

/-- The number of days after which the number of comic books is three times the number of storybooks -/
def days_until_triple_ratio : ℕ := 20

/-- The initial number of comic books -/
def initial_comic_books : ℕ := 140

/-- The initial number of storybooks -/
def initial_storybooks : ℕ := 100

/-- The number of books borrowed per day for each type -/
def daily_borrowing_rate : ℕ := 4

theorem comic_books_triple_storybooks :
  initial_comic_books - days_until_triple_ratio * daily_borrowing_rate =
  3 * (initial_storybooks - days_until_triple_ratio * daily_borrowing_rate) := by
  sorry

end NUMINAMATH_CALUDE_comic_books_triple_storybooks_l3731_373147


namespace NUMINAMATH_CALUDE_max_m_inequality_l3731_373180

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y ≥ m/(2*x + y)) ∧
  (∀ (n : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y ≥ n/(2*x + y)) → n ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l3731_373180


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3731_373152

theorem trigonometric_equation_solution (θ : ℝ) :
  3 * Real.sin (-3 * Real.pi + θ) + Real.cos (Real.pi - θ) = 0 →
  (Real.sin θ * Real.cos θ) / Real.cos (2 * θ) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3731_373152


namespace NUMINAMATH_CALUDE_correct_freshman_count_l3731_373168

/-- The number of students the college needs to admit into the freshman class each year
    to maintain a total enrollment of 3400 students, given specific dropout rates for each class. -/
def requiredFreshmen : ℕ :=
  let totalEnrollment : ℕ := 3400
  let freshmanDropoutRate : ℚ := 1/3
  let sophomoreDropouts : ℕ := 40
  let juniorDropoutRate : ℚ := 1/10
  5727

/-- Theorem stating that the required number of freshmen is 5727 -/
theorem correct_freshman_count :
  requiredFreshmen = 5727 :=
by sorry

end NUMINAMATH_CALUDE_correct_freshman_count_l3731_373168


namespace NUMINAMATH_CALUDE_sum_of_roots_is_6_l3731_373158

-- Define a quadratic function
variable (f : ℝ → ℝ)

-- Define the symmetry property
def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 + x) = f (3 - x)

-- Define the property of having two real roots
def has_two_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem sum_of_roots_is_6 (f : ℝ → ℝ) 
  (h_sym : is_symmetric_about_3 f) 
  (h_roots : has_two_real_roots f) :
  ∃ x₁ x₂ : ℝ, has_two_real_roots f ∧ x₁ + x₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_6_l3731_373158


namespace NUMINAMATH_CALUDE_six_lines_six_intersections_l3731_373118

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines intersect -/
def Line.intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- A configuration of six lines -/
structure SixLineConfig :=
  (lines : Fin 6 → Line)

/-- Count the number of intersection points in a configuration -/
def SixLineConfig.intersectionCount (config : SixLineConfig) : ℕ :=
  sorry

/-- Theorem: There exists a configuration of six lines with exactly six intersection points -/
theorem six_lines_six_intersections :
  ∃ (config : SixLineConfig), config.intersectionCount = 6 :=
sorry

end NUMINAMATH_CALUDE_six_lines_six_intersections_l3731_373118


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3731_373187

/-- Given that four identical canoes weigh the same as nine identical bowling balls,
    and one canoe weighs 36 pounds, prove that one bowling ball weighs 16 pounds. -/
theorem bowling_ball_weight (canoe_weight : ℝ) (ball_weight : ℝ) : 
  canoe_weight = 36 →  -- One canoe weighs 36 pounds
  4 * canoe_weight = 9 * ball_weight →  -- Four canoes weigh the same as nine bowling balls
  ball_weight = 16 :=  -- One bowling ball weighs 16 pounds
by
  sorry

#check bowling_ball_weight

end NUMINAMATH_CALUDE_bowling_ball_weight_l3731_373187


namespace NUMINAMATH_CALUDE_school_gender_ratio_l3731_373160

/-- The number of boys in the school -/
def num_boys : ℕ := 50

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys + 80

/-- The ratio of boys to girls as a pair of natural numbers -/
def boys_to_girls_ratio : ℕ × ℕ := (5, 13)

theorem school_gender_ratio :
  (num_boys, num_girls) = (boys_to_girls_ratio.1 * 10, boys_to_girls_ratio.2 * 10) := by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l3731_373160


namespace NUMINAMATH_CALUDE_complex_value_at_angle_l3731_373195

/-- The value of 1-2i at an angle of 267.5° is equal to -√2/2 -/
theorem complex_value_at_angle : 
  let z : ℂ := 1 - 2*I
  let angle : Real := 267.5 * (π / 180)  -- Convert to radians
  Complex.abs z * Complex.exp (I * angle) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_value_at_angle_l3731_373195


namespace NUMINAMATH_CALUDE_pqr_product_l3731_373143

theorem pqr_product (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p)
  (h1 : p + 2 / q = q + 2 / r) (h2 : q + 2 / r = r + 2 / p) :
  |p * q * r| = 2 := by
  sorry

end NUMINAMATH_CALUDE_pqr_product_l3731_373143


namespace NUMINAMATH_CALUDE_ceiling_sqrt_169_l3731_373183

theorem ceiling_sqrt_169 : ⌈Real.sqrt 169⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_169_l3731_373183


namespace NUMINAMATH_CALUDE_grace_walk_distance_l3731_373136

/-- The number of blocks Grace walked south -/
def blocks_south : ℕ := 4

/-- The number of blocks Grace walked west -/
def blocks_west : ℕ := 8

/-- The length of one block in miles -/
def block_length : ℚ := 1 / 4

/-- The total distance Grace walked in miles -/
def total_distance : ℚ := (blocks_south + blocks_west : ℚ) * block_length

theorem grace_walk_distance :
  total_distance = 3 := by sorry

end NUMINAMATH_CALUDE_grace_walk_distance_l3731_373136
