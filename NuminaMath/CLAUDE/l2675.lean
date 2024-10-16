import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l2675_267579

noncomputable section

variables (a : ℝ) (x m : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem f_properties (h1 : a > 0) (h2 : a ≠ 1) :
  -- f is an odd function
  (∀ x, f a (-x) = -f a x) ∧
  -- f is decreasing when 0 < a < 1
  ((0 < a ∧ a < 1) → (∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂)) ∧
  -- f is increasing when a > 1
  (a > 1 → (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂)) ∧
  -- For x ∈ (-1, 1), if f(m-1) + f(m) < 0, then:
  (∀ m, -1 < m ∧ m < 1 → f a (m-1) + f a m < 0 →
    ((0 < a ∧ a < 1 → 1/2 < m ∧ m < 1) ∧
     (a > 1 → 0 < m ∧ m < 1/2))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2675_267579


namespace NUMINAMATH_CALUDE_smallest_equal_probability_sum_l2675_267550

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The sum we want to compare with -/
def target_sum : ℕ := 1504

/-- The function to calculate the transformed sum -/
def transformed_sum (n : ℕ) : ℕ := 9 * n - target_sum

/-- The proposition that S is the smallest possible value satisfying the conditions -/
theorem smallest_equal_probability_sum : 
  ∃ (n : ℕ), n * sides ≥ target_sum ∧ 
  ∀ (m : ℕ), m < transformed_sum n → 
  ¬(∃ (k : ℕ), k * sides ≥ target_sum ∧ 
    transformed_sum k = m) :=
sorry

end NUMINAMATH_CALUDE_smallest_equal_probability_sum_l2675_267550


namespace NUMINAMATH_CALUDE_union_complement_theorem_l2675_267583

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_theorem : 
  M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_theorem_l2675_267583


namespace NUMINAMATH_CALUDE_roof_area_l2675_267517

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 4 * width →
  length - width = 45 →
  width * length = 900 := by
sorry

end NUMINAMATH_CALUDE_roof_area_l2675_267517


namespace NUMINAMATH_CALUDE_function_critical_points_and_inequality_l2675_267598

open Real

noncomputable def f (a x : ℝ) : ℝ := (x - 2) * exp x - a * x^2 + 2 * a * x - 2 * a

theorem function_critical_points_and_inequality (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, 0 < x → x < x₂ → f a x < -2 * a)) →
  a = exp 1 / 4 ∨ a = 2 * exp 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_critical_points_and_inequality_l2675_267598


namespace NUMINAMATH_CALUDE_binomial_factorial_product_l2675_267515

theorem binomial_factorial_product : (Nat.choose 60 3) * (Nat.factorial 10) = 124467072000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_factorial_product_l2675_267515


namespace NUMINAMATH_CALUDE_sin_cos_transformation_l2675_267554

theorem sin_cos_transformation (x : ℝ) : 
  (Real.sqrt 2 / 2) * Real.sin x + (Real.sqrt 2 / 2) * Real.cos x + 1 = Real.sin (x + π/4) + 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_l2675_267554


namespace NUMINAMATH_CALUDE_fable_village_impossible_total_l2675_267547

theorem fable_village_impossible_total (p h s c d k : ℕ) : 
  p = 4 * h ∧ 
  s = 5 * c ∧ 
  d = 2 * p ∧ 
  k = 2 * d → 
  p + h + s + c + d + k ≠ 90 :=
by sorry

end NUMINAMATH_CALUDE_fable_village_impossible_total_l2675_267547


namespace NUMINAMATH_CALUDE_smallest_number_is_21_l2675_267578

/-- A sequence of 25 consecutive natural numbers satisfying certain conditions -/
def ConsecutiveSequence (start : ℕ) : Prop :=
  ∃ (seq : Fin 25 → ℕ),
    (∀ i, seq i = start + i) ∧
    (((Finset.filter (λ i => seq i % 2 = 0) Finset.univ).card : ℚ) / 25 = 12 / 25) ∧
    (((Finset.filter (λ i => seq i < 30) Finset.univ).card : ℚ) / 25 = 9 / 25)

/-- The smallest number in the sequence is 21 -/
theorem smallest_number_is_21 :
  ∃ (start : ℕ), ConsecutiveSequence start ∧ ∀ s, ConsecutiveSequence s → start ≤ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_is_21_l2675_267578


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2675_267529

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x > 2) ↔ (∃ x : ℝ, x ≥ 1 ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2675_267529


namespace NUMINAMATH_CALUDE_mary_garden_apples_l2675_267531

/-- The number of pies Mary wants to bake -/
def num_pies : ℕ := 10

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 8

/-- The number of additional apples Mary needs to buy -/
def apples_to_buy : ℕ := 30

/-- The number of apples Mary harvested from her garden -/
def apples_from_garden : ℕ := num_pies * apples_per_pie - apples_to_buy

theorem mary_garden_apples : apples_from_garden = 50 := by
  sorry

end NUMINAMATH_CALUDE_mary_garden_apples_l2675_267531


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l2675_267560

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + 8 * y₀ - x₀ * y₀ = 0 ∧ x₀ * y₀ = 64) ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 2 * x' + 8 * y' - x' * y' = 0 → x' * y' ≥ 64) ∧
  (∃ (x₁ y₁ : ℝ), x₁ > 0 ∧ y₁ > 0 ∧ 2 * x₁ + 8 * y₁ - x₁ * y₁ = 0 ∧ x₁ + y₁ = 18) ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 2 * x' + 8 * y' - x' * y' = 0 → x' + y' ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l2675_267560


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2675_267563

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (Real.cos (α / 2))^2 = 4 * Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2675_267563


namespace NUMINAMATH_CALUDE_wednesday_occurs_five_times_in_august_l2675_267500

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- July of year N -/
def july : Month := { days := 31, firstDay := DayOfWeek.Tuesday }

/-- August of year N -/
def august : Month := { days := 31, firstDay := sorry }

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- The main theorem -/
theorem wednesday_occurs_five_times_in_august :
  (countDayOccurrences july DayOfWeek.Tuesday = 5) →
  (countDayOccurrences august DayOfWeek.Wednesday = 5) := by
  sorry

end NUMINAMATH_CALUDE_wednesday_occurs_five_times_in_august_l2675_267500


namespace NUMINAMATH_CALUDE_equation_proof_l2675_267551

theorem equation_proof : (12 : ℕ)^2 * 6^4 / 432 = 432 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2675_267551


namespace NUMINAMATH_CALUDE_polynomial_expansion_simplification_l2675_267587

theorem polynomial_expansion_simplification (x : ℝ) : 
  (x^3 - 3*x^2 + (1/2)*x - 1) * (x^2 + 3*x + 3/2) = 
  x^5 - (15/2)*x^3 - 4*x^2 - (9/4)*x - 3/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_simplification_l2675_267587


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2675_267561

-- Define the universe
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets A and B
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

-- State the theorem
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2675_267561


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2675_267582

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- All terms are positive
  (∀ k, ∃ q > 0, a (k + 1) = q * a k) →  -- Geometric sequence
  (a m * a n).sqrt = 8 * a 1 →  -- Given condition
  a 9 = a 8 + 2 * a 7 →  -- Given condition
  (∃ m' n', m' + n' = 8 ∧ 1 / m' + 4 / n' < 1 / m + 4 / n) →  -- Minimum condition
  1 / m + 4 / n = 17 / 15 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2675_267582


namespace NUMINAMATH_CALUDE_polynomial_root_not_all_real_l2675_267549

theorem polynomial_root_not_all_real (a b c d e : ℝ) :
  2 * a^2 < 5 * b →
  ∃ z : ℂ, z^5 + a*z^4 + b*z^3 + c*z^2 + d*z + e = 0 ∧ z.im ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_not_all_real_l2675_267549


namespace NUMINAMATH_CALUDE_defect_probability_is_22_900_l2675_267599

/-- Represents a machine in the production line -/
structure Machine where
  defectProb : ℝ
  productivityRatio : ℝ

/-- The production setup with three machines -/
def productionSetup : List Machine := [
  { defectProb := 0.02, productivityRatio := 3 },
  { defectProb := 0.03, productivityRatio := 1 },
  { defectProb := 0.04, productivityRatio := 0.5 }
]

/-- Calculates the probability of a randomly selected part being defective -/
def calculateDefectProbability (setup : List Machine) : ℝ :=
  sorry

/-- Theorem stating that the probability of a defective part is 22/900 -/
theorem defect_probability_is_22_900 :
  calculateDefectProbability productionSetup = 22 / 900 := by
  sorry

end NUMINAMATH_CALUDE_defect_probability_is_22_900_l2675_267599


namespace NUMINAMATH_CALUDE_feline_sanctuary_tigers_l2675_267530

theorem feline_sanctuary_tigers (lions cougars tigers : ℕ) : 
  lions = 12 →
  cougars = (lions + tigers) / 2 →
  lions + tigers + cougars = 39 →
  tigers = 14 := by
sorry

end NUMINAMATH_CALUDE_feline_sanctuary_tigers_l2675_267530


namespace NUMINAMATH_CALUDE_abc_sum_product_bound_l2675_267556

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 1) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_product_bound_l2675_267556


namespace NUMINAMATH_CALUDE_inequality_solution_l2675_267568

theorem inequality_solution (y : ℝ) : 
  (y^2 + y^3 - 3*y^4) / (y + y^2 - 3*y^3) ≥ -1 ↔ 
  y ∈ Set.Icc (-1) (-4/3) ∪ Set.Ioo (-4/3) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2675_267568


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l2675_267586

/-- Calculates the percentage of yellow tint in a new mixture after adding more yellow tint -/
theorem yellow_tint_percentage 
  (initial_volume : ℝ)
  (initial_yellow_percentage : ℝ)
  (added_yellow : ℝ) :
  initial_volume = 50 →
  initial_yellow_percentage = 25 →
  added_yellow = 10 →
  let initial_yellow := initial_volume * (initial_yellow_percentage / 100)
  let new_yellow := initial_yellow + added_yellow
  let new_volume := initial_volume + added_yellow
  (new_yellow / new_volume) * 100 = 37.5 := by
sorry


end NUMINAMATH_CALUDE_yellow_tint_percentage_l2675_267586


namespace NUMINAMATH_CALUDE_population_increase_rate_is_two_l2675_267546

/-- The rate of population increase in persons per minute, given that one person is added every 30 seconds. -/
def population_increase_rate (seconds_per_person : ℕ) : ℚ :=
  60 / seconds_per_person

/-- Theorem stating that if the population increases by one person every 30 seconds, 
    then the rate of population increase is 2 persons per minute. -/
theorem population_increase_rate_is_two :
  population_increase_rate 30 = 2 := by sorry

end NUMINAMATH_CALUDE_population_increase_rate_is_two_l2675_267546


namespace NUMINAMATH_CALUDE_april_production_l2675_267509

/-- Calculates the production after n months given an initial production and monthly growth rate -/
def production_after_months (initial_production : ℕ) (growth_rate : ℝ) (months : ℕ) : ℝ :=
  initial_production * (1 + growth_rate) ^ months

/-- Proves that the production in April is 926,100 pencils given the initial conditions -/
theorem april_production :
  let initial_production := 800000
  let growth_rate := 0.05
  let months := 3
  ⌊production_after_months initial_production growth_rate months⌋ = 926100 := by
  sorry

end NUMINAMATH_CALUDE_april_production_l2675_267509


namespace NUMINAMATH_CALUDE_shoes_per_person_l2675_267524

theorem shoes_per_person (num_pairs : ℕ) (num_people : ℕ) : 
  num_pairs = 36 → num_people = 36 → (num_pairs * 2) / num_people = 2 := by
  sorry

end NUMINAMATH_CALUDE_shoes_per_person_l2675_267524


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l2675_267573

def number_of_balls : ℕ := 20

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_is_even (a b : ℕ) : Prop := is_even (a + b)

theorem probability_of_even_sum :
  let total_outcomes := number_of_balls * (number_of_balls - 1)
  let favorable_outcomes := (number_of_balls / 2) * ((number_of_balls / 2) - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l2675_267573


namespace NUMINAMATH_CALUDE_coin_distribution_problem_l2675_267522

theorem coin_distribution_problem :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ 9 * x + 17 * y = 70 ∧ x = 4 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_problem_l2675_267522


namespace NUMINAMATH_CALUDE_shelves_used_l2675_267562

theorem shelves_used (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 130 → books_sold = 47 → books_per_shelf = 15 →
  (initial_stock - books_sold + books_per_shelf - 1) / books_per_shelf = 6 := by
  sorry

#eval (130 - 47 + 15 - 1) / 15

end NUMINAMATH_CALUDE_shelves_used_l2675_267562


namespace NUMINAMATH_CALUDE_train_distance_problem_l2675_267534

/-- Given two trains with specified lengths, speeds, and crossing time, 
    calculate the initial distance between them. -/
theorem train_distance_problem (length1 length2 speed1 speed2 crossing_time : ℝ) 
  (h1 : length1 = 100)
  (h2 : length2 = 150)
  (h3 : speed1 = 10)
  (h4 : speed2 = 15)
  (h5 : crossing_time = 60)
  (h6 : speed2 > speed1) : 
  (speed2 - speed1) * crossing_time = length1 + length2 + 50 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2675_267534


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2675_267559

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2675_267559


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2675_267545

def P : Set ℝ := {x | -x^2 + 3*x + 4 < 0}
def Q : Set ℝ := {x | 2*x - 5 > 0}

theorem intersection_P_Q : P ∩ Q = {x | x > 4} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2675_267545


namespace NUMINAMATH_CALUDE_sugar_salt_difference_is_two_l2675_267502

/-- A baking recipe with specified amounts of ingredients -/
structure Recipe where
  sugar : ℕ
  flour : ℕ
  salt : ℕ

/-- The amount of ingredients Mary has already added -/
structure Added where
  flour : ℕ

/-- Calculate the difference between required sugar and salt -/
def sugarSaltDifference (recipe : Recipe) : ℤ :=
  recipe.sugar - recipe.salt

/-- Theorem: The difference between required sugar and salt is 2 cups -/
theorem sugar_salt_difference_is_two (recipe : Recipe) (added : Added) :
  recipe.sugar = 11 →
  recipe.flour = 6 →
  recipe.salt = 9 →
  added.flour = 12 →
  sugarSaltDifference recipe = 2 := by
  sorry

#eval sugarSaltDifference { sugar := 11, flour := 6, salt := 9 }

end NUMINAMATH_CALUDE_sugar_salt_difference_is_two_l2675_267502


namespace NUMINAMATH_CALUDE_quadratic_root_inequality_l2675_267544

theorem quadratic_root_inequality (a b c : ℝ) (ha : a ≠ 0) 
  (h_root : ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + b * x + c = 0) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_inequality_l2675_267544


namespace NUMINAMATH_CALUDE_inequality_proof_l2675_267577

theorem inequality_proof (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2675_267577


namespace NUMINAMATH_CALUDE_problem_paths_l2675_267572

/-- Represents the number of ways to reach a specific arrow type -/
structure ArrowPaths where
  count : Nat
  arrows : Nat

/-- The modified hexagonal lattice structure -/
structure HexLattice where
  redPaths : ArrowPaths
  bluePaths : ArrowPaths
  greenPaths : ArrowPaths
  endPaths : Nat

/-- The specific hexagonal lattice in the problem -/
def problemLattice : HexLattice :=
  { redPaths := { count := 1, arrows := 1 }
    bluePaths := { count := 3, arrows := 2 }
    greenPaths := { count := 6, arrows := 2 }
    endPaths := 4 }

/-- Calculates the total number of paths in the lattice -/
def totalPaths (lattice : HexLattice) : Nat :=
  lattice.redPaths.count *
  lattice.bluePaths.count * lattice.bluePaths.arrows *
  lattice.greenPaths.count * lattice.greenPaths.arrows *
  lattice.endPaths

theorem problem_paths :
  totalPaths problemLattice = 288 := by sorry

end NUMINAMATH_CALUDE_problem_paths_l2675_267572


namespace NUMINAMATH_CALUDE_student_weight_l2675_267519

/-- Prove that the student's present weight is 71 kilograms -/
theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 104) :
  student_weight = 71 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_l2675_267519


namespace NUMINAMATH_CALUDE_hide_and_seek_l2675_267537

-- Define the participants
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem statement
theorem hide_and_seek :
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena := by sorry

end NUMINAMATH_CALUDE_hide_and_seek_l2675_267537


namespace NUMINAMATH_CALUDE_pancake_problem_l2675_267516

theorem pancake_problem (pancakes_made : ℕ) (family_size : ℕ) : pancakes_made = 12 → family_size = 8 → 
  (pancakes_made - family_size) + (family_size - (pancakes_made - family_size)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pancake_problem_l2675_267516


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l2675_267523

/-- Given the conversion rates between knicks, knacks, and knocks, 
    this theorem proves that 36 knocks are equivalent to 40 knicks. -/
theorem knicks_knacks_knocks_conversion : 
  ∀ (knick knack knock : ℚ),
  (5 * knick = 3 * knack) →
  (4 * knack = 6 * knock) →
  (36 * knock = 40 * knick) := by
sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l2675_267523


namespace NUMINAMATH_CALUDE_fold_crease_forms_ellipse_l2675_267596

/-- Given a circle with radius R centered at the origin and an internal point A at (a, 0),
    the set of all points P(x, y) that are equidistant from A and any point on the circle's circumference
    forms an ellipse. -/
theorem fold_crease_forms_ellipse (R a : ℝ) (h : 0 < a ∧ a < R) :
  ∀ x y : ℝ,
    (∃ α : ℝ, (x - R * Real.cos α)^2 + (y - R * Real.sin α)^2 = (x - a)^2 + y^2) ↔
    (2*x - a)^2 / R^2 + 4*y^2 / (R^2 - a^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fold_crease_forms_ellipse_l2675_267596


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l2675_267506

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l2675_267506


namespace NUMINAMATH_CALUDE_find_a_and_m_l2675_267527

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + (a-1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- State the theorem
theorem find_a_and_m :
  ∃ (a m : ℝ),
    (A ∪ B a = A) ∧
    (A ∩ B a = C m) ∧
    (a = 3) ∧
    (m = 3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_m_l2675_267527


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2675_267539

/-- Given a geometric sequence {a_n} with S_3 = 9/2 and a_3 = 3/2, prove that the common ratio q satisfies q = 1 or q = -1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : S 3 = 9/2) 
  (h2 : a 3 = 3/2) : 
  ∃ q : ℚ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ (q = 1 ∨ q = -1/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2675_267539


namespace NUMINAMATH_CALUDE_cookie_count_pastry_shop_cookies_l2675_267511

/-- Given a ratio of doughnuts, cookies, and muffins, and the number of doughnuts and muffins,
    calculate the number of cookies. -/
theorem cookie_count (doughnut_ratio cookie_ratio muffin_ratio : ℕ) 
                     (doughnut_count muffin_count : ℕ) : ℕ :=
  let total_ratio := doughnut_ratio + cookie_ratio + muffin_ratio
  let part_value := doughnut_count / doughnut_ratio
  cookie_ratio * part_value

/-- Prove that given the ratio of doughnuts, cookies, and muffins is 5 : 3 : 1,
    and there are 50 doughnuts and 10 muffins, the number of cookies is 30. -/
theorem pastry_shop_cookies : cookie_count 5 3 1 50 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_pastry_shop_cookies_l2675_267511


namespace NUMINAMATH_CALUDE_jeremy_remaining_money_l2675_267512

/-- Given an initial amount and the costs of various items, calculate the remaining amount --/
def remaining_amount (initial : ℕ) (jersey_cost : ℕ) (jersey_count : ℕ) (basketball_cost : ℕ) (shorts_cost : ℕ) : ℕ :=
  initial - (jersey_cost * jersey_count + basketball_cost + shorts_cost)

/-- Prove that Jeremy has $14 left after his purchases --/
theorem jeremy_remaining_money :
  remaining_amount 50 2 5 18 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_remaining_money_l2675_267512


namespace NUMINAMATH_CALUDE_car_speed_comparison_l2675_267538

/-- Proves that the average speed of Car A is less than or equal to the average speed of Car B -/
theorem car_speed_comparison
  (u v : ℝ) -- speeds in miles per hour
  (hu : u > 0) (hv : v > 0) -- speeds are positive
  (x : ℝ) -- average speed of Car A
  (hx : x = 3 / (1 / u + 2 / v)) -- definition of x
  (y : ℝ) -- average speed of Car B
  (hy : y = (u + 2 * v) / 3) -- definition of y
  : x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l2675_267538


namespace NUMINAMATH_CALUDE_factorial_division_equality_l2675_267558

theorem factorial_division_equality : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_equality_l2675_267558


namespace NUMINAMATH_CALUDE_all_solutions_irrational_l2675_267597

/-- A number is rational if it can be expressed as a ratio of two integers -/
def IsRational (x : ℝ) : Prop := ∃ (m n : ℤ), n ≠ 0 ∧ x = m / n

/-- A number is irrational if it is not rational -/
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

/-- The equation in question -/
def SatisfiesEquation (x : ℝ) : Prop := 0.001 * x^3 + x^2 - 1 = 0

theorem all_solutions_irrational :
  ∀ x : ℝ, SatisfiesEquation x → IsIrrational x := by
  sorry

end NUMINAMATH_CALUDE_all_solutions_irrational_l2675_267597


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l2675_267574

/-- A line through (1, -8) and (k, 15) is parallel to 6x + 9y = -12 iff k = -33.5 -/
theorem parallel_line_k_value : ∀ k : ℝ,
  (∃ m b : ℝ, (∀ x y : ℝ, y = m*x + b ↔ (x = 1 ∧ y = -8) ∨ (x = k ∧ y = 15)) ∧
               m = -2/3) ↔
  k = -33.5 := by sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l2675_267574


namespace NUMINAMATH_CALUDE_cheesecake_eggs_proof_l2675_267580

/-- The number of eggs needed for each chocolate cake -/
def chocolate_cake_eggs : ℕ := 3

/-- The number of eggs needed for each cheesecake -/
def cheesecake_eggs : ℕ := 8

/-- Proof that the number of eggs for each cheesecake is 8 -/
theorem cheesecake_eggs_proof : 
  9 * cheesecake_eggs = 5 * chocolate_cake_eggs + 57 :=
by sorry

end NUMINAMATH_CALUDE_cheesecake_eggs_proof_l2675_267580


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2675_267528

theorem polygon_interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 2340) → (180 * ((n - 3) - 2) = 1800) := by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2675_267528


namespace NUMINAMATH_CALUDE_cake_area_theorem_l2675_267588

/-- Represents the dimensions of a piece of cake -/
structure PieceDimensions where
  length : ℝ
  width : ℝ

/-- Represents a cake -/
structure Cake where
  pieces : ℕ
  pieceDimensions : PieceDimensions

/-- Calculates the total area of a cake -/
def cakeArea (c : Cake) : ℝ :=
  c.pieces * (c.pieceDimensions.length * c.pieceDimensions.width)

theorem cake_area_theorem (c : Cake) 
  (h1 : c.pieces = 25)
  (h2 : c.pieceDimensions.length = 4)
  (h3 : c.pieceDimensions.width = 4) :
  cakeArea c = 400 := by
  sorry

end NUMINAMATH_CALUDE_cake_area_theorem_l2675_267588


namespace NUMINAMATH_CALUDE_f_domain_l2675_267575

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (3 - Real.tan x ^ 2) + Real.sqrt (x * (Real.pi - x))

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem f_domain :
  domain f = Set.Icc 0 (Real.pi / 3) ∪ Set.Ioc (2 * Real.pi / 3) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_f_domain_l2675_267575


namespace NUMINAMATH_CALUDE_square_of_number_l2675_267566

theorem square_of_number (x : ℝ) : 2 * x = x / 5 + 9 → x^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_l2675_267566


namespace NUMINAMATH_CALUDE_log_product_equals_four_implies_y_equals_81_l2675_267504

theorem log_product_equals_four_implies_y_equals_81 (m y : ℝ) 
  (h : m > 0) (k : y > 0) (eq : Real.log y / Real.log m * Real.log m / Real.log 3 = 4) : 
  y = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_implies_y_equals_81_l2675_267504


namespace NUMINAMATH_CALUDE_inequality_max_value_inequality_range_l2675_267510

theorem inequality_max_value (x y : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 1 3) :
  (2 * x^2 + y^2) / (x * y) ≤ 2 * Real.sqrt 2 :=
by sorry

theorem inequality_range (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 3 → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_max_value_inequality_range_l2675_267510


namespace NUMINAMATH_CALUDE_shortest_light_path_length_shortest_light_path_equals_12_l2675_267501

/-- The shortest path length of a light ray reflecting off the x-axis -/
theorem shortest_light_path_length : ℝ :=
  let A : ℝ × ℝ := (-3, 9)
  let C : ℝ × ℝ := (2, 3)  -- Center of the circle
  let r : ℝ := 1  -- Radius of the circle
  let C' : ℝ × ℝ := (2, -3)  -- Reflection of C across x-axis
  let AC' : ℝ := Real.sqrt ((-3 - 2)^2 + (9 - (-3))^2)
  AC' - r

theorem shortest_light_path_equals_12 :
  shortest_light_path_length = 12 := by sorry

end NUMINAMATH_CALUDE_shortest_light_path_length_shortest_light_path_equals_12_l2675_267501


namespace NUMINAMATH_CALUDE_ursulas_purchases_l2675_267533

theorem ursulas_purchases (tea_price : ℝ) 
  (h1 : tea_price = 10)
  (h2 : tea_price > 0) :
  let cheese_price := tea_price / 2
  let butter_price := 0.8 * cheese_price
  let bread_price := butter_price / 2
  tea_price + cheese_price + butter_price + bread_price = 21 :=
by sorry

end NUMINAMATH_CALUDE_ursulas_purchases_l2675_267533


namespace NUMINAMATH_CALUDE_pi_approx_thousandth_l2675_267591

/-- The approximation of π to the thousandth place -/
def pi_approx : ℝ := 3.142

/-- The theorem stating that the approximation of π to the thousandth place is equal to 3.142 -/
theorem pi_approx_thousandth : |π - pi_approx| < 0.0005 := by
  sorry

end NUMINAMATH_CALUDE_pi_approx_thousandth_l2675_267591


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2675_267576

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 22

theorem total_cost_calculation : 
  (10 * mango_cost = 24 * rice_cost) → 
  (flour_cost = 2 * rice_cost) → 
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 248.6) := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2675_267576


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2675_267540

theorem reciprocal_of_negative_2023 : ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2675_267540


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2675_267595

theorem scientific_notation_equivalence : 
  56000000 = 5.6 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2675_267595


namespace NUMINAMATH_CALUDE_max_k_value_l2675_267508

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 84) →
  k ≤ 2 * Real.sqrt 29 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2675_267508


namespace NUMINAMATH_CALUDE_abs_2x_minus_1_lt_15_x_squared_plus_6x_minus_16_lt_0_abs_2x_plus_1_gt_13_x_squared_minus_2x_gt_0_l2675_267520

-- Question 1
theorem abs_2x_minus_1_lt_15 (x : ℝ) : 
  |2*x - 1| < 15 ↔ -7 < x ∧ x < 8 := by sorry

-- Question 2
theorem x_squared_plus_6x_minus_16_lt_0 (x : ℝ) : 
  x^2 + 6*x - 16 < 0 ↔ -8 < x ∧ x < 2 := by sorry

-- Question 3
theorem abs_2x_plus_1_gt_13 (x : ℝ) : 
  |2*x + 1| > 13 ↔ x < -7 ∨ x > 6 := by sorry

-- Question 4
theorem x_squared_minus_2x_gt_0 (x : ℝ) : 
  x^2 - 2*x > 0 ↔ x < 0 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_abs_2x_minus_1_lt_15_x_squared_plus_6x_minus_16_lt_0_abs_2x_plus_1_gt_13_x_squared_minus_2x_gt_0_l2675_267520


namespace NUMINAMATH_CALUDE_negation_of_existence_l2675_267569

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2675_267569


namespace NUMINAMATH_CALUDE_typing_difference_l2675_267535

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 20

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 40

/-- The difference in words typed per hour between Isaiah and Micah -/
theorem typing_difference : 
  (isaiah_speed * minutes_per_hour) - (micah_speed * minutes_per_hour) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_typing_difference_l2675_267535


namespace NUMINAMATH_CALUDE_basketball_weight_prove_basketball_weight_l2675_267571

theorem basketball_weight : ℝ → ℝ → ℝ → Prop :=
  fun basketball_weight tricycle_weight motorbike_weight =>
    (9 * basketball_weight = 6 * tricycle_weight) ∧
    (6 * tricycle_weight = 4 * motorbike_weight) ∧
    (2 * motorbike_weight = 144) →
    basketball_weight = 32

-- Proof
theorem prove_basketball_weight :
  ∃ (b t m : ℝ), basketball_weight b t m :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_weight_prove_basketball_weight_l2675_267571


namespace NUMINAMATH_CALUDE_subset_ratio_eight_elements_l2675_267553

theorem subset_ratio_eight_elements :
  let n : ℕ := 8
  let S : ℕ := 2^n
  let T : ℕ := n.choose 3
  (T : ℚ) / S = 7 / 32 := by
sorry

end NUMINAMATH_CALUDE_subset_ratio_eight_elements_l2675_267553


namespace NUMINAMATH_CALUDE_cars_produced_in_north_america_l2675_267590

theorem cars_produced_in_north_america :
  ∀ (total_cars europe_cars north_america_cars : ℕ),
    total_cars = 6755 →
    europe_cars = 2871 →
    total_cars = europe_cars + north_america_cars →
    north_america_cars = 3884 :=
by
  sorry

end NUMINAMATH_CALUDE_cars_produced_in_north_america_l2675_267590


namespace NUMINAMATH_CALUDE_jack_book_pages_l2675_267570

/-- Calculates the total number of pages in a book given the daily reading rate and the number of days to finish. -/
def total_pages (pages_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  pages_per_day * days_to_finish

/-- Proves that the book Jack is reading has 299 pages. -/
theorem jack_book_pages :
  let pages_per_day : ℕ := 23
  let days_to_finish : ℕ := 13
  total_pages pages_per_day days_to_finish = 299 := by
  sorry

end NUMINAMATH_CALUDE_jack_book_pages_l2675_267570


namespace NUMINAMATH_CALUDE_combinations_of_three_from_seven_l2675_267525

theorem combinations_of_three_from_seven (n k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_combinations_of_three_from_seven_l2675_267525


namespace NUMINAMATH_CALUDE_field_completion_time_l2675_267536

theorem field_completion_time (team1_time team2_time initial_days joint_days : ℝ) : 
  team1_time = 12 →
  team2_time = 0.75 * team1_time →
  initial_days = 5 →
  (initial_days / team1_time) + joint_days * (1 / team1_time + 1 / team2_time) = 1 →
  joint_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_completion_time_l2675_267536


namespace NUMINAMATH_CALUDE_total_drive_distance_l2675_267507

/-- The total distance of a drive given two drivers with different speed limits and driving times -/
theorem total_drive_distance (christina_speed : ℝ) (friend_speed : ℝ) (christina_time_min : ℝ) (friend_time_hr : ℝ) : 
  christina_speed = 30 →
  friend_speed = 40 →
  christina_time_min = 180 →
  friend_time_hr = 3 →
  christina_speed * (christina_time_min / 60) + friend_speed * friend_time_hr = 210 := by
  sorry

#check total_drive_distance

end NUMINAMATH_CALUDE_total_drive_distance_l2675_267507


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_four_consecutive_integers_sum_l2675_267565

theorem smallest_prime_factor_of_four_consecutive_integers_sum (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k ∧
  ∀ (p : ℕ), p < 2 → ¬(Prime p ∧ ∃ (m : ℤ), (n - 1) + n + (n + 1) + (n + 2) = p * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_four_consecutive_integers_sum_l2675_267565


namespace NUMINAMATH_CALUDE_sqrt_14400_l2675_267557

theorem sqrt_14400 : Real.sqrt 14400 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14400_l2675_267557


namespace NUMINAMATH_CALUDE_power_equation_l2675_267584

theorem power_equation (x : ℝ) (h : (10 : ℝ)^(2*x) = 25) : (10 : ℝ)^(1-x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l2675_267584


namespace NUMINAMATH_CALUDE_custom_mul_property_l2675_267503

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := (a + b + 1)^2

/-- Theorem stating that (x-1) * (1-x) = 1 for all real x -/
theorem custom_mul_property (x : ℝ) : custom_mul (x - 1) (1 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_property_l2675_267503


namespace NUMINAMATH_CALUDE_solution_difference_l2675_267514

theorem solution_difference (p q : ℝ) : 
  ((p - 5) * (p + 5) = 26 * p - 130) →
  ((q - 5) * (q + 5) = 26 * q - 130) →
  p ≠ q →
  p > q →
  p - q = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2675_267514


namespace NUMINAMATH_CALUDE_mode_is_25_l2675_267543

def sales_volumes : List ℕ := [10, 14, 25, 13]

def is_mode (x : ℕ) (list : List ℕ) : Prop :=
  ∀ y ∈ list, (list.count x ≥ list.count y)

theorem mode_is_25 (s : ℕ) : is_mode 25 (sales_volumes ++ [s]) := by
  sorry

end NUMINAMATH_CALUDE_mode_is_25_l2675_267543


namespace NUMINAMATH_CALUDE_b_2017_value_l2675_267592

/-- Given sequences a and b with the specified properties, b₂₀₁₇ equals 2016/2017 -/
theorem b_2017_value (a b : ℕ → ℚ) : 
  (b 1 = 0) →
  (∀ n : ℕ, n ≥ 1 → a n = 1 / (n * (n + 1))) →
  (∀ n : ℕ, n ≥ 2 → b n = b (n - 1) + a (n - 1)) →
  b 2017 = 2016 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_b_2017_value_l2675_267592


namespace NUMINAMATH_CALUDE_B_is_top_leftmost_l2675_267532

/-- Represents a rectangle with four sides labeled w, x, y, z --/
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- The set of all rectangles in the arrangement --/
def rectangles : Finset Rectangle := sorry

/-- Rectangle A --/
def A : Rectangle := ⟨5, 2, 8, 11⟩

/-- Rectangle B --/
def B : Rectangle := ⟨2, 1, 4, 7⟩

/-- Rectangle C --/
def C : Rectangle := ⟨4, 9, 6, 3⟩

/-- Rectangle D --/
def D : Rectangle := ⟨8, 6, 5, 9⟩

/-- Rectangle E --/
def E : Rectangle := ⟨10, 3, 9, 1⟩

/-- Rectangle F --/
def F : Rectangle := ⟨11, 4, 10, 2⟩

/-- Predicate to check if a rectangle is in the leftmost position --/
def isLeftmost (r : Rectangle) : Prop :=
  ∀ s ∈ rectangles, r.w ≤ s.w

/-- Predicate to check if a rectangle is in the top row --/
def isTopRow (r : Rectangle) : Prop := sorry

/-- The main theorem stating that B is the top leftmost rectangle --/
theorem B_is_top_leftmost : isLeftmost B ∧ isTopRow B := by sorry

end NUMINAMATH_CALUDE_B_is_top_leftmost_l2675_267532


namespace NUMINAMATH_CALUDE_vanessa_scoring_record_l2675_267589

/-- Vanessa's new scoring record in a basketball game -/
theorem vanessa_scoring_record (total_score : ℕ) (other_players : ℕ) (average_score : ℕ) 
  (h1 : total_score = 55)
  (h2 : other_players = 7)
  (h3 : average_score = 4) : 
  total_score - (other_players * average_score) = 27 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_scoring_record_l2675_267589


namespace NUMINAMATH_CALUDE_inverse_function_value_l2675_267541

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem inverse_function_value :
  f_inv 2 = -2/3 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_value_l2675_267541


namespace NUMINAMATH_CALUDE_housing_boom_calculation_l2675_267505

/-- The number of houses in Lawrence County before the housing boom. -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom. -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom. -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_calculation :
  houses_built = 574 :=
by sorry

end NUMINAMATH_CALUDE_housing_boom_calculation_l2675_267505


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2675_267518

theorem quadratic_equation_properties (m : ℝ) :
  let equation := fun x => x^2 - 2*m*x + m^2 - 4*m - 1
  (∃ x : ℝ, equation x = 0) ↔ m ≥ -1/4
  ∧
  equation 1 = 0 → m = 0 ∨ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2675_267518


namespace NUMINAMATH_CALUDE_Q_iff_a_in_range_P_xor_Q_iff_a_in_range_l2675_267555

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a + 1) + y^2 / (a - 2) = 1 ∧ (a + 1) * (a - 2) < 0

-- Theorem for part (1)
theorem Q_iff_a_in_range (a : ℝ) : Q a ↔ a ∈ Set.Ioo (-1) 2 := by sorry

-- Theorem for part (2)
theorem P_xor_Q_iff_a_in_range (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Ioo 1 2 ∪ Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_Q_iff_a_in_range_P_xor_Q_iff_a_in_range_l2675_267555


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l2675_267594

theorem hippopotamus_crayons (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 62)
  (h2 : remaining_crayons = 10) :
  initial_crayons - remaining_crayons = 52 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_crayons_l2675_267594


namespace NUMINAMATH_CALUDE_total_weight_moved_proof_l2675_267548

/-- Calculates the total weight moved during three sets of back squat, front squat, and deadlift exercises --/
def total_weight_moved (initial_back_squat : ℝ) (back_squat_increase : ℝ) : ℝ :=
  let updated_back_squat := initial_back_squat + back_squat_increase
  let front_squat_ratio := 0.8
  let deadlift_ratio := 1.2
  let back_squat_increase_ratio := 1.05
  let front_squat_increase_ratio := 1.04
  let deadlift_increase_ratio := 1.03
  let back_squat_performance_ratio := 1.0
  let front_squat_performance_ratio := 0.9
  let deadlift_performance_ratio := 0.85
  let back_squat_reps := 3
  let front_squat_reps := 3
  let deadlift_reps := 2

  let back_squat_set1 := updated_back_squat * back_squat_performance_ratio * back_squat_reps
  let back_squat_set2 := updated_back_squat * back_squat_increase_ratio * back_squat_performance_ratio * back_squat_reps
  let back_squat_set3 := updated_back_squat * back_squat_increase_ratio * back_squat_increase_ratio * back_squat_performance_ratio * back_squat_reps

  let front_squat_base := updated_back_squat * front_squat_ratio
  let front_squat_set1 := front_squat_base * front_squat_performance_ratio * front_squat_reps
  let front_squat_set2 := front_squat_base * front_squat_increase_ratio * front_squat_performance_ratio * front_squat_reps
  let front_squat_set3 := front_squat_base * front_squat_increase_ratio * front_squat_increase_ratio * front_squat_performance_ratio * front_squat_reps

  let deadlift_base := updated_back_squat * deadlift_ratio
  let deadlift_set1 := deadlift_base * deadlift_performance_ratio * deadlift_reps
  let deadlift_set2 := deadlift_base * deadlift_increase_ratio * deadlift_performance_ratio * deadlift_reps
  let deadlift_set3 := deadlift_base * deadlift_increase_ratio * deadlift_increase_ratio * deadlift_performance_ratio * deadlift_reps

  back_squat_set1 + back_squat_set2 + back_squat_set3 +
  front_squat_set1 + front_squat_set2 + front_squat_set3 +
  deadlift_set1 + deadlift_set2 + deadlift_set3

theorem total_weight_moved_proof (initial_back_squat : ℝ) (back_squat_increase : ℝ) :
  initial_back_squat = 200 → back_squat_increase = 50 →
  total_weight_moved initial_back_squat back_squat_increase = 5626.398 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_moved_proof_l2675_267548


namespace NUMINAMATH_CALUDE_problem_statement_l2675_267542

theorem problem_statement : (-1)^53 + 2^(4^4 + 3^3 - 5^2) = -1 + 2^258 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2675_267542


namespace NUMINAMATH_CALUDE_angle_conversion_l2675_267564

theorem angle_conversion (angle_deg : ℝ) (k : ℤ) (α : ℝ) :
  angle_deg = -1125 →
  (k = -4 ∧ α = (7 * π) / 4) →
  (0 ≤ α ∧ α < 2 * π) →
  angle_deg * π / 180 = 2 * k * π + α := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l2675_267564


namespace NUMINAMATH_CALUDE_simplify_expression_l2675_267521

theorem simplify_expression : (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = (1 / 2) * (3^16 - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2675_267521


namespace NUMINAMATH_CALUDE_sin_50_cos_80_cos_160_l2675_267513

theorem sin_50_cos_80_cos_160 :
  Real.sin (50 * π / 180) * Real.cos (80 * π / 180) * Real.cos (160 * π / 180) = -1/8 := by
sorry

end NUMINAMATH_CALUDE_sin_50_cos_80_cos_160_l2675_267513


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2675_267593

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 140000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.4
    exponent := 8
    coeff_range := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2675_267593


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_120_l2675_267567

theorem greatest_common_multiple_9_15_under_120 :
  ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  90 % 9 = 0 ∧ 90 % 15 = 0 ∧ 90 < 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_120_l2675_267567


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2675_267526

/-- The volume of ice cream in a cone with a cylinder on top -/
theorem ice_cream_volume (cone_height : ℝ) (cone_radius : ℝ) (cylinder_height : ℝ) : 
  cone_height = 12 → 
  cone_radius = 3 → 
  cylinder_height = 2 → 
  (1/3 * π * cone_radius^2 * cone_height) + (π * cone_radius^2 * cylinder_height) = 54 * π := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_volume_l2675_267526


namespace NUMINAMATH_CALUDE_ball_speed_equality_time_ball_speed_equality_time_specific_l2675_267552

/-- The time when a ball's average speed equals its instantaneous speed after being dropped from a height and experiencing a perfectly elastic collision. -/
theorem ball_speed_equality_time
  (h : ℝ)  -- Initial height
  (g : ℝ)  -- Acceleration due to gravity
  (h_pos : h > 0)
  (g_pos : g > 0)
  : ∃ (t : ℝ), t > 0 ∧ t = Real.sqrt (2 * h / g + 8 * h / g) :=
by
  sorry

/-- The specific case where h = 45 m and g = 10 m/s² -/
theorem ball_speed_equality_time_specific :
  ∃ (t : ℝ), t > 0 ∧ t = Real.sqrt 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_speed_equality_time_ball_speed_equality_time_specific_l2675_267552


namespace NUMINAMATH_CALUDE_not_perfect_square_with_digit_sum_2006_l2675_267581

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem not_perfect_square_with_digit_sum_2006 (n : ℕ) 
  (h : sum_of_digits n = 2006) : 
  ¬ ∃ (m : ℕ), n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_with_digit_sum_2006_l2675_267581


namespace NUMINAMATH_CALUDE_second_party_amount_2000_4_16_l2675_267585

/-- Calculates the amount received by the second party in a two-party division given a total amount and a ratio --/
def calculate_second_party_amount (total : ℕ) (ratio1 : ℕ) (ratio2 : ℕ) : ℕ :=
  let total_parts := ratio1 + ratio2
  let part_value := total / total_parts
  ratio2 * part_value

theorem second_party_amount_2000_4_16 :
  calculate_second_party_amount 2000 4 16 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_second_party_amount_2000_4_16_l2675_267585
