import Mathlib

namespace combined_molecular_weight_l912_91202

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def chlorine_weight : ℝ := 35.45
def sulfur_weight : ℝ := 32.07
def fluorine_weight : ℝ := 19.00

-- Define molecular compositions
def ccl4_carbon_count : ℕ := 1
def ccl4_chlorine_count : ℕ := 4
def sf6_sulfur_count : ℕ := 1
def sf6_fluorine_count : ℕ := 6

-- Define number of moles
def ccl4_moles : ℕ := 9
def sf6_moles : ℕ := 5

-- Theorem statement
theorem combined_molecular_weight :
  let ccl4_weight := carbon_weight * ccl4_carbon_count + chlorine_weight * ccl4_chlorine_count
  let sf6_weight := sulfur_weight * sf6_sulfur_count + fluorine_weight * sf6_fluorine_count
  ccl4_weight * ccl4_moles + sf6_weight * sf6_moles = 2114.64 := by
  sorry

end combined_molecular_weight_l912_91202


namespace leg_head_difference_l912_91243

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 13

/-- Calculates the total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- States that the difference between total legs and thrice the total heads is 13 -/
theorem leg_head_difference : total_legs - 3 * total_heads = 13 := by sorry

end leg_head_difference_l912_91243


namespace fraction_simplification_l912_91276

theorem fraction_simplification :
  (3 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 - Real.sqrt 18) = (3 * Real.sqrt 2) / 26 := by
  sorry

end fraction_simplification_l912_91276


namespace correct_calculation_l912_91248

theorem correct_calculation (x : ℤ) (h : x + 238 = 637) : x - 382 = 17 := by
  sorry

end correct_calculation_l912_91248


namespace no_valid_labeling_exists_l912_91233

/-- Represents a labeling of a 45-gon with digits 0-9 -/
def Labeling := Fin 45 → Fin 10

/-- Checks if a labeling is valid according to the problem conditions -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ i j : Fin 10, i ≠ j →
    ∃! k : Fin 45, (l k = i ∧ l (k + 1) = j) ∨ (l k = j ∧ l (k + 1) = i)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_labeling_exists : ¬∃ l : Labeling, is_valid_labeling l := by
  sorry

end no_valid_labeling_exists_l912_91233


namespace max_product_constraint_l912_91281

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/4 + b/5 = 1) :
  a * b ≤ 5 := by
sorry

end max_product_constraint_l912_91281


namespace three_std_dev_below_mean_undetermined_l912_91217

/-- Represents a non-normal probability distribution --/
structure NonNormalDistribution where
  mean : ℝ
  std_dev : ℝ
  skewness : ℝ
  kurtosis : ℝ
  is_non_normal : Bool

/-- The value that is exactly 3 standard deviations less than the mean --/
def three_std_dev_below_mean (d : NonNormalDistribution) : ℝ := sorry

/-- Theorem stating that the value 3 standard deviations below the mean cannot be determined
    for a non-normal distribution without additional information --/
theorem three_std_dev_below_mean_undetermined
  (d : NonNormalDistribution)
  (h_mean : d.mean = 15)
  (h_std_dev : d.std_dev = 1.5)
  (h_skewness : d.skewness = 0.5)
  (h_kurtosis : d.kurtosis = 0.6)
  (h_non_normal : d.is_non_normal = true) :
  ¬ ∃ (x : ℝ), three_std_dev_below_mean d = x :=
sorry

end three_std_dev_below_mean_undetermined_l912_91217


namespace modified_short_bingo_arrangements_l912_91292

theorem modified_short_bingo_arrangements : Nat.factorial 15 / Nat.factorial 8 = 1816214400 := by
  sorry

end modified_short_bingo_arrangements_l912_91292


namespace tv_cash_price_l912_91273

def installment_plan_cost (down_payment : ℕ) (monthly_payment : ℕ) (num_months : ℕ) : ℕ :=
  down_payment + monthly_payment * num_months

def cash_price (total_installment_cost : ℕ) (savings : ℕ) : ℕ :=
  total_installment_cost - savings

theorem tv_cash_price :
  let down_payment : ℕ := 120
  let monthly_payment : ℕ := 30
  let num_months : ℕ := 12
  let savings : ℕ := 80
  let total_installment_cost : ℕ := installment_plan_cost down_payment monthly_payment num_months
  cash_price total_installment_cost savings = 400 := by
  sorry

end tv_cash_price_l912_91273


namespace sin_shift_l912_91244

open Real

theorem sin_shift (x : ℝ) :
  sin (3 * (x - π / 12)) = sin (3 * x - π / 4) := by sorry

end sin_shift_l912_91244


namespace locus_of_point_C_l912_91271

/-- The locus of point C in a triangle ABC with given conditions forms an ellipse -/
theorem locus_of_point_C (A B C : ℝ × ℝ) : 
  (A = (-6, 0) ∧ B = (6, 0)) →  -- Coordinates of A and B
  (dist A B + dist B C + dist C A = 26) →  -- Perimeter condition
  (C.1 ≠ 7 ∧ C.1 ≠ -7) →  -- Exclude points where x = ±7
  (C.1^2 / 49 + C.2^2 / 13 = 1) :=  -- Equation of the ellipse
by sorry

end locus_of_point_C_l912_91271


namespace geometric_sequence_property_l912_91221

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry


end geometric_sequence_property_l912_91221


namespace valid_pairs_l912_91286

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def valid_pair (a b : ℕ) : Prop :=
  is_integer ((a^2 + b) / (b^2 - a)) ∧
  is_integer ((b^2 + a) / (a^2 - b))

theorem valid_pairs :
  ∀ a b : ℕ, valid_pair a b ↔
    ((a = 1 ∧ b = 2) ∨
     (a = 2 ∧ b = 1) ∨
     (a = 2 ∧ b = 2) ∨
     (a = 2 ∧ b = 3) ∨
     (a = 3 ∧ b = 2) ∨
     (a = 3 ∧ b = 3)) :=
sorry

end valid_pairs_l912_91286


namespace expression_equality_l912_91215

theorem expression_equality (x y : ℝ) (h : x^2 + y^2 = 1) :
  2*x^4 + 3*x^2*y^2 + y^4 + y^2 = 2 := by
  sorry

end expression_equality_l912_91215


namespace valentino_farm_birds_l912_91274

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ := chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∃ (chickens ducks turkeys : ℕ),
    chickens = 200 ∧
    ducks = 2 * chickens ∧
    turkeys = 3 * ducks ∧
    total_birds chickens ducks turkeys = 1800 := by
  sorry

end valentino_farm_birds_l912_91274


namespace expand_polynomial_l912_91240

theorem expand_polynomial (x : ℝ) : (4 * x + 3) * (2 * x - 7) + x = 8 * x^2 - 21 * x - 21 := by
  sorry

end expand_polynomial_l912_91240


namespace largest_angle_is_90_l912_91270

-- Define an isosceles triangle with angles α, β, and γ
structure IsoscelesTriangle where
  α : Real
  β : Real
  γ : Real
  isIsosceles : (α = β) ∨ (α = γ) ∨ (β = γ)
  sumIs180 : α + β + γ = 180
  nonNegative : α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0

-- Define the condition that two angles are in the ratio 1:2
def hasRatio1to2 (t : IsoscelesTriangle) : Prop :=
  (t.α = 2 * t.β) ∨ (t.β = 2 * t.α) ∨ (t.α = 2 * t.γ) ∨ (t.γ = 2 * t.α) ∨ (t.β = 2 * t.γ) ∨ (t.γ = 2 * t.β)

-- Theorem statement
theorem largest_angle_is_90 (t : IsoscelesTriangle) (h : hasRatio1to2 t) :
  max t.α (max t.β t.γ) = 90 := by sorry

end largest_angle_is_90_l912_91270


namespace cost_of_nuts_l912_91223

/-- Given Alyssa's purchases and refund, calculate the cost of the pack of nuts -/
theorem cost_of_nuts (grapes_cost refund_cherries total_spent : ℚ) 
  (h1 : grapes_cost = 12.08)
  (h2 : refund_cherries = 9.85)
  (h3 : total_spent = 26.35) :
  grapes_cost - refund_cherries + (total_spent - (grapes_cost - refund_cherries)) = 24.12 := by
  sorry

end cost_of_nuts_l912_91223


namespace john_mary_difference_l912_91260

/-- The number of chickens each person took -/
structure ChickenCount where
  ray : ℕ
  john : ℕ
  mary : ℕ

/-- The conditions of the chicken distribution problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.ray = 10 ∧
  c.john = c.ray + 11 ∧
  c.mary = c.ray + 6

/-- The theorem stating the difference between John's and Mary's chicken count -/
theorem john_mary_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.mary = 5 := by
  sorry

end john_mary_difference_l912_91260


namespace angle_from_terminal_point_l912_91277

theorem angle_from_terminal_point (θ : Real) :
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin θ = Real.sin (3 * Real.pi / 4) ∧ Real.cos θ = Real.cos (3 * Real.pi / 4)) →
  θ = 7 * Real.pi / 4 := by
  sorry

end angle_from_terminal_point_l912_91277


namespace wall_building_time_l912_91275

/-- Represents the time taken to build a wall given different workforce scenarios -/
theorem wall_building_time
  (original_men : ℕ)
  (original_days : ℕ)
  (new_total_men : ℕ)
  (fast_men : ℕ)
  (h1 : original_men = 20)
  (h2 : original_days = 6)
  (h3 : new_total_men = 30)
  (h4 : fast_men = 10)
  (h5 : fast_men ≤ new_total_men) :
  let effective_workforce := new_total_men - fast_men + 2 * fast_men
  let new_days := (original_men * original_days) / effective_workforce
  new_days = 3 := by sorry

end wall_building_time_l912_91275


namespace nine_integer_chords_l912_91293

/-- Represents a circle with a given radius and a point P at a given distance from the center -/
structure CircleWithPoint where
  radius : ℝ
  distanceToP : ℝ

/-- Counts the number of integer-length chords passing through P -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem nine_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 20)
  (h2 : c.distanceToP = 12) : 
  countIntegerChords c = 9 :=
sorry

end nine_integer_chords_l912_91293


namespace gcd_of_large_numbers_l912_91204

theorem gcd_of_large_numbers : Nat.gcd 1000000000 1000000005 = 5 := by
  sorry

end gcd_of_large_numbers_l912_91204


namespace cubic_polynomial_roots_l912_91287

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Predicate to check if a, b, c are roots of the polynomial -/
def has_roots (p : CubicPolynomial) : Prop :=
  (p.a^3 + p.a * p.a^2 + p.b * p.a + p.c = 0) ∧
  (p.b^3 + p.a * p.b^2 + p.b * p.b + p.c = 0) ∧
  (p.c^3 + p.a * p.c^2 + p.b * p.c + p.c = 0)

/-- The set of valid polynomials -/
def valid_polynomials : Set CubicPolynomial :=
  {⟨0, 0, 0⟩, ⟨1, -2, 0⟩}

theorem cubic_polynomial_roots (p : CubicPolynomial) :
  has_roots p ↔ p ∈ valid_polynomials := by sorry

end cubic_polynomial_roots_l912_91287


namespace total_age_is_23_l912_91222

/-- Proves that the total combined age of Ryanne, Hezekiah, and Jamison is 23 years -/
theorem total_age_is_23 (hezekiah_age : ℕ) 
  (ryanne_older : hezekiah_age + 7 = ryanne_age)
  (sum_ryanne_hezekiah : ryanne_age + hezekiah_age = 15)
  (jamison_twice : jamison_age = 2 * hezekiah_age) : 
  ryanne_age + hezekiah_age + jamison_age = 23 :=
by
  sorry

#check total_age_is_23

end total_age_is_23_l912_91222


namespace total_toys_l912_91284

/-- The number of toys each child has -/
structure ToyCount where
  jerry : ℕ
  gabriel : ℕ
  jaxon : ℕ

/-- The conditions of the problem -/
def toy_conditions (t : ToyCount) : Prop :=
  t.jerry = t.gabriel + 8 ∧
  t.gabriel = 2 * t.jaxon ∧
  t.jaxon = 15

/-- The theorem stating the total number of toys -/
theorem total_toys (t : ToyCount) (h : toy_conditions t) : 
  t.jerry + t.gabriel + t.jaxon = 83 := by
  sorry

end total_toys_l912_91284


namespace inverse_function_property_l912_91272

-- Define a function f with an inverse
def f_has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_function_property
  (f : ℝ → ℝ)
  (h_inverse : f_has_inverse f)
  (h_point : f 2 = -1) :
  ∃ f_inv : ℝ → ℝ, f_inv (-1) = 2 ∧ (∀ x, f_inv (f x) = x) ∧ (∀ y, f (f_inv y) = y) :=
sorry

end inverse_function_property_l912_91272


namespace smallest_gcd_of_multiples_l912_91294

theorem smallest_gcd_of_multiples (a b : ℕ+) (h : Nat.gcd a b = 18) :
  ∃ (m n : ℕ+), 12 * m = 12 * a ∧ 20 * n = 20 * b ∧ 
    Nat.gcd (12 * m) (20 * n) = 72 ∧
    ∀ (x y : ℕ+), 12 * x = 12 * a → 20 * y = 20 * b → 
      Nat.gcd (12 * x) (20 * y) ≥ 72 :=
by sorry

end smallest_gcd_of_multiples_l912_91294


namespace half_abs_diff_squares_15_13_l912_91212

theorem half_abs_diff_squares_15_13 : 
  (1/2 : ℝ) * |15^2 - 13^2| = 28 := by sorry

end half_abs_diff_squares_15_13_l912_91212


namespace mp3_player_problem_l912_91290

def initial_songs : Nat := 8
def deleted_songs : Nat := 5
def added_songs : Nat := 30
def added_song_durations : List Nat := [3, 4, 2, 6, 5, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 4, 3, 5, 6, 7, 8, 9, 10]

theorem mp3_player_problem :
  (initial_songs - deleted_songs + added_songs = 33) ∧
  (added_song_durations.sum = 145) := by
  sorry

end mp3_player_problem_l912_91290


namespace distinct_sums_count_l912_91216

def bag_A : Finset Nat := {1, 3, 5, 7}
def bag_B : Finset Nat := {2, 4, 6, 8}

def possible_sums : Finset Nat :=
  Finset.image (λ (pair : Nat × Nat) => pair.1 + pair.2) (bag_A.product bag_B)

theorem distinct_sums_count : Finset.card possible_sums = 7 := by
  sorry

end distinct_sums_count_l912_91216


namespace prescription_final_cost_l912_91295

/-- Calculates the final cost of a prescription after cashback and rebate --/
theorem prescription_final_cost
  (original_cost : ℝ)
  (cashback_percentage : ℝ)
  (rebate : ℝ)
  (h1 : original_cost = 150)
  (h2 : cashback_percentage = 0.1)
  (h3 : rebate = 25) :
  original_cost - (cashback_percentage * original_cost + rebate) = 110 :=
by sorry

end prescription_final_cost_l912_91295


namespace expand_and_simplify_l912_91265

theorem expand_and_simplify (x : ℝ) : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_and_simplify_l912_91265


namespace cos_symmetry_l912_91245

/-- The function f(x) = cos(2x + π/3) is symmetric about the line x = π/3 -/
theorem cos_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x + π / 3)
  ∀ y : ℝ, f (π / 3 + y) = f (π / 3 - y) := by
  sorry

end cos_symmetry_l912_91245


namespace f_max_min_on_interval_l912_91232

def f (x : ℝ) := 3 * x^3 - 9 * x + 5

theorem f_max_min_on_interval :
  ∃ (max min : ℝ) (x_max x_min : ℝ),
    x_max ∈ Set.Icc (-3 : ℝ) 3 ∧
    x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ f x_min) ∧
    f x_max = max ∧
    f x_min = min ∧
    max = 59 ∧
    min = -49 ∧
    x_max = 3 ∧
    x_min = -3 :=
by sorry

end f_max_min_on_interval_l912_91232


namespace tan_squared_gamma_equals_tan_alpha_tan_beta_l912_91250

theorem tan_squared_gamma_equals_tan_alpha_tan_beta 
  (α β γ : Real) 
  (h : (Real.sin γ)^2 / (Real.sin α)^2 = 1 - Real.tan (α - β) / Real.tan α) : 
  (Real.tan γ)^2 = Real.tan α * Real.tan β := by
  sorry

end tan_squared_gamma_equals_tan_alpha_tan_beta_l912_91250


namespace chi_square_significant_distribution_prob_sum_to_one_l912_91258

/-- Represents the total number of students surveyed -/
def total_students : ℕ := 2000

/-- Represents the percentage of students with myopia -/
def myopia_rate : ℚ := 2/5

/-- Represents the percentage of students spending more than 1 hour on phones daily -/
def long_phone_usage_rate : ℚ := 1/5

/-- Represents the myopia rate among students who spend more than 1 hour on phones daily -/
def myopia_rate_long_usage : ℚ := 1/2

/-- Represents the significance level for the Chi-square test -/
def significance_level : ℚ := 1/1000

/-- Represents the critical value for the Chi-square test at α = 0.001 -/
def critical_value : ℚ := 10828/1000

/-- Represents the number of myopic students randomly selected -/
def selected_myopic_students : ℕ := 8

/-- Represents the number of selected myopic students spending more than 1 hour on phones -/
def long_usage_in_selection : ℕ := 2

/-- Represents the number of students randomly selected from the 8 myopic students -/
def final_selection : ℕ := 3

/-- Calculates the Chi-square value for the given data -/
def chi_square_value : ℚ := 20833/1000

/-- Theorem stating that the Chi-square value is greater than the critical value -/
theorem chi_square_significant : chi_square_value > critical_value := by sorry

/-- Calculates the probability of X = 0 in the distribution table -/
def prob_X_0 : ℚ := 5/14

/-- Calculates the probability of X = 1 in the distribution table -/
def prob_X_1 : ℚ := 15/28

/-- Calculates the probability of X = 2 in the distribution table -/
def prob_X_2 : ℚ := 3/28

/-- Theorem stating that the probabilities in the distribution table sum up to 1 -/
theorem distribution_prob_sum_to_one : prob_X_0 + prob_X_1 + prob_X_2 = 1 := by sorry

end chi_square_significant_distribution_prob_sum_to_one_l912_91258


namespace expected_rolls_in_non_leap_year_l912_91205

/-- Represents the outcomes of rolling an eight-sided die -/
inductive DieOutcome
| Composite
| Prime
| RollAgain

/-- The probability of each outcome when rolling the die -/
def outcomeProbability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Composite => 2/8
  | DieOutcome.Prime => 4/8
  | DieOutcome.RollAgain => 2/8

/-- The expected number of rolls on a single day -/
def expectedRollsPerDay : ℚ :=
  4/3

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
def expectedRollsInYear : ℚ :=
  expectedRollsPerDay * daysInNonLeapYear

theorem expected_rolls_in_non_leap_year :
  expectedRollsInYear = 486 + 2/3 :=
sorry

end expected_rolls_in_non_leap_year_l912_91205


namespace f_2005_of_2006_eq_145_l912_91225

/-- Sum of squares of digits of a positive integer -/
def f (n : ℕ+) : ℕ := sorry

/-- Recursive application of f, k times -/
def f_k (k : ℕ) (n : ℕ+) : ℕ :=
  match k with
  | 0 => n.val
  | k + 1 => f (⟨f_k k n, sorry⟩)

/-- The main theorem to prove -/
theorem f_2005_of_2006_eq_145 : f_k 2005 ⟨2006, sorry⟩ = 145 := by sorry

end f_2005_of_2006_eq_145_l912_91225


namespace mat_weavers_problem_l912_91291

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 16

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 64

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 16

theorem mat_weavers_problem :
  first_group_weavers * second_group_mats * first_group_days =
  second_group_weavers * first_group_mats * second_group_days :=
by sorry

end mat_weavers_problem_l912_91291


namespace forest_coverage_growth_rate_l912_91209

theorem forest_coverage_growth_rate (x : ℝ) : 
  (0.63 * (1 + x)^2 = 0.68) ↔ 
  (∃ (rate : ℝ → ℝ), 
    rate 0 = 0.63 ∧ 
    rate 2 = 0.68 ∧ 
    ∀ t, 0 ≤ t → t ≤ 2 → rate t = 0.63 * (1 + x)^t) :=
sorry

end forest_coverage_growth_rate_l912_91209


namespace routes_in_grid_l912_91219

/-- The number of routes in a 3x3 grid from top-left to bottom-right -/
def num_routes : ℕ := Nat.choose 6 3

/-- The dimensions of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := 2 * grid_size

/-- The number of moves in each direction -/
def moves_per_direction : ℕ := grid_size

theorem routes_in_grid :
  num_routes = Nat.choose total_moves moves_per_direction :=
sorry

end routes_in_grid_l912_91219


namespace uncommon_roots_product_l912_91263

def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 8*x^2 - 6*x + 15
def Q (x : ℝ) : ℝ := x^3 + 4*x^2 - x - 10

theorem uncommon_roots_product : 
  ∃ (r₁ r₂ : ℝ), 
    P r₁ = 0 ∧ 
    Q r₂ = 0 ∧ 
    r₁ ≠ r₂ ∧
    (∀ x : ℝ, (P x = 0 ∧ Q x ≠ 0) ∨ (Q x = 0 ∧ P x ≠ 0) → x = r₁ ∨ x = r₂) ∧
    r₁ * r₂ = -2 * Real.sqrt 3 := by
  sorry

end uncommon_roots_product_l912_91263


namespace polynomial_simplification_l912_91256

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 2 * x^2 + 6 * x - 18) = 
  x^3 + 2 * x^2 + 3 * x + 13 := by
  sorry

end polynomial_simplification_l912_91256


namespace intersection_points_product_l912_91268

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - Real.sqrt 3 * y - m = 0

-- Define the intersection condition
def intersects (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    (x₁ - m)^2 + y₁^2 * (x₂ - m)^2 + y₂^2 = 1

-- Theorem statement
theorem intersection_points_product (m : ℝ) :
  intersects m ↔ m = 1 ∨ m = 1 + Real.sqrt 2 ∨ m = 1 - Real.sqrt 2 := by
  sorry

end intersection_points_product_l912_91268


namespace frank_uniform_number_l912_91203

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem frank_uniform_number 
  (d e f : ℕ) 
  (h1 : is_two_digit_prime d) 
  (h2 : is_two_digit_prime e) 
  (h3 : is_two_digit_prime f) 
  (h4 : d + f = 28) 
  (h5 : d + e = 24) 
  (h6 : e + f = 30) : 
  f = 17 := by
sorry

end frank_uniform_number_l912_91203


namespace point_inside_ellipse_l912_91200

theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) → (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) := by
  sorry

end point_inside_ellipse_l912_91200


namespace part_one_part_two_l912_91207

-- Part 1
theorem part_one (x y : ℝ) (hx : x = Real.sqrt 2 - 1) (hy : y = Real.sqrt 2 + 1) :
  y / x + x / y = 6 := by sorry

-- Part 2
theorem part_two :
  (Real.sqrt 3 + Real.sqrt 2 - 2) * (Real.sqrt 3 - Real.sqrt 2 + 2) = 4 * Real.sqrt 2 - 3 := by sorry

end part_one_part_two_l912_91207


namespace debby_dvd_count_l912_91288

/-- The number of DVDs Debby sold -/
def sold_dvds : ℕ := 6

/-- The number of DVDs Debby had left after selling -/
def remaining_dvds : ℕ := 7

/-- The initial number of DVDs Debby owned -/
def initial_dvds : ℕ := sold_dvds + remaining_dvds

theorem debby_dvd_count : initial_dvds = 13 := by sorry

end debby_dvd_count_l912_91288


namespace original_tomatoes_cost_l912_91231

def original_order : ℝ := 25
def new_tomatoes : ℝ := 2.20
def old_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def old_celery : ℝ := 1.96
def new_celery : ℝ := 2.00
def delivery_tip : ℝ := 8.00
def new_total : ℝ := 35

theorem original_tomatoes_cost (x : ℝ) : 
  x = 3.41 ↔ 
  x + old_lettuce + old_celery + delivery_tip = new_total ∧
  new_tomatoes + new_lettuce + new_celery + delivery_tip = new_total :=
by sorry

end original_tomatoes_cost_l912_91231


namespace product_mod_seven_l912_91269

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end product_mod_seven_l912_91269


namespace rectangle_area_l912_91214

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3/10)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w : ℝ, 
  w > 0 ∧ 
  x^2 = (3*w)^2 + w^2 ∧ 
  (3*w) * w = (3/10) * x^2 :=
sorry

end rectangle_area_l912_91214


namespace king_queen_ages_l912_91247

theorem king_queen_ages : ∃ (K Q : ℕ),
  -- The king is twice as old as the queen was when the king was as old as the queen is now
  K = 2 * (Q - (K - Q)) ∧
  -- When the queen is as old as the king is now, their combined ages will be 63 years
  Q + (K - (K - Q)) + K = 63 ∧
  -- The king's age is 28 and the queen's age is 21
  K = 28 ∧ Q = 21 := by
sorry

end king_queen_ages_l912_91247


namespace sqrt_6_simplest_l912_91262

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y z : ℝ, y * y = x → z * z = x → y = z ∨ y = -z

theorem sqrt_6_simplest :
  is_simplest_sqrt 6 ∧
  ¬ is_simplest_sqrt 8 ∧
  ¬ is_simplest_sqrt 12 ∧
  ¬ is_simplest_sqrt 0.3 :=
sorry

end sqrt_6_simplest_l912_91262


namespace largest_prime_factor_of_897_l912_91251

theorem largest_prime_factor_of_897 : ∃ (p : ℕ), Prime p ∧ p ∣ 897 ∧ ∀ (q : ℕ), Prime q → q ∣ 897 → q ≤ p :=
by
  -- The proof goes here
  sorry

end largest_prime_factor_of_897_l912_91251


namespace percentage_relation_l912_91230

theorem percentage_relation (x a b : ℝ) (ha : a = 0.07 * x) (hb : b = 0.14 * x) :
  a / b = 1 / 2 := by sorry

end percentage_relation_l912_91230


namespace horse_oats_consumption_l912_91289

theorem horse_oats_consumption 
  (num_horses : ℕ) 
  (meals_per_day : ℕ) 
  (days : ℕ) 
  (total_oats : ℕ) 
  (h1 : num_horses = 4) 
  (h2 : meals_per_day = 2) 
  (h3 : days = 3) 
  (h4 : total_oats = 96) : 
  total_oats / (num_horses * meals_per_day * days) = 4 := by
sorry

end horse_oats_consumption_l912_91289


namespace symmetry_preserves_circle_l912_91234

/-- A circle in R^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in R^2 of the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The given circle (x-1)^2 + (y-1)^2 = 1 -/
def given_circle : Circle := { center := (1, 1), radius := 1 }

/-- The given line y = 5x - 4 -/
def given_line : Line := { m := 5, b := -4 }

/-- Predicate to check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- The symmetrical circle with respect to a line -/
def symmetrical_circle (c : Circle) (l : Line) : Circle :=
  sorry -- Definition of symmetrical circle

theorem symmetry_preserves_circle (c : Circle) (l : Line) :
  point_on_line c.center l →
  symmetrical_circle c l = c := by
  sorry

end symmetry_preserves_circle_l912_91234


namespace cube_root_product_equals_48_l912_91235

theorem cube_root_product_equals_48 : 
  (64 : ℝ) ^ (1/3) * (27 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 48 := by
  sorry

end cube_root_product_equals_48_l912_91235


namespace chewbacca_gum_pack_size_l912_91220

theorem chewbacca_gum_pack_size :
  ∀ x : ℕ,
  (20 : ℚ) / 30 = (20 - x) / 30 →
  (20 : ℚ) / 30 = 20 / (30 + 5 * x) →
  x ≠ 0 →
  x = 14 := by
sorry

end chewbacca_gum_pack_size_l912_91220


namespace triangle_side_length_l912_91279

/-- Given a triangle ABC where the internal angles form an arithmetic sequence,
    and sides a = 4 and c = 3, prove that the length of side b is √13 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  B = (A + C) / 2 →      -- Angles form an arithmetic sequence
  a = 4 →                -- Length of side a
  c = 3 →                -- Length of side c
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →  -- Cosine rule
  b = Real.sqrt 13 := by
sorry

end triangle_side_length_l912_91279


namespace ellipse_properties_l912_91264

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the condition for points A and B
def SlopeCondition (xA yA xB yB : ℝ) : Prop :=
  xA ≠ 0 ∧ xB ≠ 0 ∧ (yA / xA) * (yB / xB) = -3/2

theorem ellipse_properties :
  -- Given conditions
  let D := (0, 2)
  let F := (c, 0)
  let E := (4*c/3, -2/3)
  -- The ellipse passes through D and E
  Ellipse D.1 D.2 ∧ Ellipse E.1 E.2 →
  -- |DF| = 3|EF|
  (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9 * ((E.1 - F.1)^2 + (E.2 - F.2)^2) →
  -- Theorem statements
  (∀ x y, Ellipse x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ xA yA xB yB,
    Ellipse xA yA → Ellipse xB yB → SlopeCondition xA yA xB yB →
    -1 ≤ (xA * xB + yA * yB) ∧ (xA * xB + yA * yB) ≤ 1 ∧
    (xA * xB + yA * yB) ≠ 0) :=
by sorry

end ellipse_properties_l912_91264


namespace unique_f_3_l912_91218

/-- A function satisfying the given functional equation and initial condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * (f x) * y) ∧ f 1 = 1

/-- The theorem stating that for any function satisfying the conditions, f(3) must equal 9 -/
theorem unique_f_3 (f : ℝ → ℝ) (hf : special_function f) : f 3 = 9 := by
  sorry

end unique_f_3_l912_91218


namespace joans_total_seashells_l912_91213

/-- The number of seashells Joan found on the beach -/
def initial_seashells : ℕ := 70

/-- The number of seashells Sam gave to Joan -/
def additional_seashells : ℕ := 27

/-- Theorem: Joan's total number of seashells is 97 -/
theorem joans_total_seashells : initial_seashells + additional_seashells = 97 := by
  sorry

end joans_total_seashells_l912_91213


namespace ladybug_leaf_count_l912_91228

theorem ladybug_leaf_count (ladybugs_per_leaf : ℕ) (total_ladybugs : ℕ) (h1 : ladybugs_per_leaf = 139) (h2 : total_ladybugs = 11676) :
  total_ladybugs / ladybugs_per_leaf = 84 := by
  sorry

end ladybug_leaf_count_l912_91228


namespace sam_remaining_money_l912_91255

/-- Calculates the remaining money in cents after Sam's purchases -/
def remaining_money (initial_dimes : Nat) (initial_quarters : Nat) 
  (candy_bars : Nat) (candy_bar_cost : Nat) (lollipop_cost : Nat) : Nat :=
  let remaining_dimes := initial_dimes - candy_bars * candy_bar_cost
  let remaining_quarters := initial_quarters - 1
  remaining_dimes * 10 + remaining_quarters * 25

/-- Proves that Sam has 195 cents left after her purchases -/
theorem sam_remaining_money :
  remaining_money 19 6 4 3 1 = 195 := by
  sorry

end sam_remaining_money_l912_91255


namespace khalil_dogs_count_l912_91283

/-- Represents the veterinary clinic problem -/
def veterinary_clinic_problem (dog_cost cat_cost : ℕ) (num_cats total_cost : ℕ) : Prop :=
  ∃ (num_dogs : ℕ), 
    dog_cost * num_dogs + cat_cost * num_cats = total_cost

/-- Proves that the number of dogs Khalil took to the clinic is 20 -/
theorem khalil_dogs_count : veterinary_clinic_problem 60 40 60 3600 → 
  ∃ (num_dogs : ℕ), num_dogs = 20 ∧ 60 * num_dogs + 40 * 60 = 3600 :=
by
  sorry


end khalil_dogs_count_l912_91283


namespace taxi_problem_l912_91249

/-- Represents the direction of travel -/
inductive Direction
| East
| West

/-- Represents a single trip -/
structure Trip where
  distance : ℝ
  direction : Direction

def trips : List Trip := [
  ⟨8, Direction.East⟩, ⟨6, Direction.West⟩, ⟨3, Direction.East⟩,
  ⟨7, Direction.West⟩, ⟨8, Direction.East⟩, ⟨4, Direction.East⟩,
  ⟨9, Direction.West⟩, ⟨4, Direction.West⟩, ⟨3, Direction.East⟩,
  ⟨3, Direction.East⟩
]

def totalTime : ℝ := 1.25 -- in hours

def startingFare : ℝ := 8
def additionalFarePerKm : ℝ := 2
def freeDistance : ℝ := 3

theorem taxi_problem (trips : List Trip) (totalTime : ℝ) 
    (startingFare additionalFarePerKm freeDistance : ℝ) :
  -- 1. Final position is 3 km east
  (trips.foldl (fun acc trip => 
    match trip.direction with
    | Direction.East => acc + trip.distance
    | Direction.West => acc - trip.distance
  ) 0 = 3) ∧
  -- 2. Average speed is 44 km/h
  ((trips.foldl (fun acc trip => acc + trip.distance) 0) / totalTime = 44) ∧
  -- 3. Total earnings are 130 yuan
  (trips.length * startingFare + 
   (trips.foldl (fun acc trip => acc + max (trip.distance - freeDistance) 0) 0) * additionalFarePerKm = 130) := by
  sorry

end taxi_problem_l912_91249


namespace workshop_workers_count_l912_91278

/-- Given a workshop with workers including technicians and non-technicians,
    prove that the total number of workers is 22 under the following conditions:
    - The average salary of all workers is 850
    - There are 7 technicians with an average salary of 1000
    - The average salary of non-technicians is 780 -/
theorem workshop_workers_count :
  ∀ (W : ℕ) (avg_salary tech_salary nontech_salary : ℚ),
    avg_salary = 850 →
    tech_salary = 1000 →
    nontech_salary = 780 →
    (W : ℚ) * avg_salary = 7 * tech_salary + (W - 7 : ℚ) * nontech_salary →
    W = 22 := by
  sorry


end workshop_workers_count_l912_91278


namespace ten_percent_of_x_l912_91257

theorem ten_percent_of_x (x c : ℝ) : 
  3 - (1/4)*2 - (1/3)*3 - (1/7)*x = c → 
  (10/100) * x = 0.7 * (1.5 - c) := by
sorry

end ten_percent_of_x_l912_91257


namespace farmhouse_blocks_l912_91229

def total_blocks : ℕ := 344
def building_blocks : ℕ := 80
def fenced_area_blocks : ℕ := 57
def leftover_blocks : ℕ := 84

theorem farmhouse_blocks :
  total_blocks - building_blocks - fenced_area_blocks - leftover_blocks = 123 := by
  sorry

end farmhouse_blocks_l912_91229


namespace lower_variance_more_stable_student_B_more_stable_l912_91224

/-- Represents a student's throwing performance -/
structure StudentPerformance where
  name : String
  variance : ℝ

/-- Defines the concept of stability in performance -/
def moreStable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students' performances, the one with lower variance is more stable -/
theorem lower_variance_more_stable (a b : StudentPerformance) :
    moreStable a b ↔ a.variance < b.variance :=
  by sorry

/-- The specific problem instance -/
def studentA : StudentPerformance :=
  { name := "A", variance := 0.2 }

def studentB : StudentPerformance :=
  { name := "B", variance := 0.09 }

/-- Theorem: Student B has more stable performance than Student A -/
theorem student_B_more_stable : moreStable studentB studentA :=
  by sorry

end lower_variance_more_stable_student_B_more_stable_l912_91224


namespace no_nontrivial_integer_solution_l912_91208

theorem no_nontrivial_integer_solution :
  ∀ (a b c d : ℤ), a^2 - b = c^2 ∧ b^2 - a = d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end no_nontrivial_integer_solution_l912_91208


namespace prob_red_is_three_tenths_l912_91246

-- Define the contents of the bags
def bag_A : Finset (Fin 10) := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def bag_B : Finset (Fin 10) := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the colors of the balls
inductive Color
| Red
| White
| Black

-- Define the color distribution in bag A
def color_A : Fin 10 → Color
| 0 | 1 | 2 => Color.Red
| 3 | 4 => Color.White
| _ => Color.Black

-- Define the color distribution in bag B
def color_B : Fin 10 → Color
| 0 | 1 | 2 => Color.Red
| 3 | 4 | 5 => Color.White
| _ => Color.Black

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_after_transfer : ℚ :=
  3 / 10

-- Theorem statement
theorem prob_red_is_three_tenths :
  prob_red_after_transfer = 3 / 10 := by
  sorry


end prob_red_is_three_tenths_l912_91246


namespace intersection_is_sinusoid_l912_91236

/-- Represents a cylinder with radius R and height H -/
structure Cylinder where
  R : ℝ
  H : ℝ

/-- Represents the inclined plane intersecting the cylinder -/
structure InclinedPlane where
  α : ℝ  -- Angle of inclination

/-- Represents a point on the unfolded lateral surface of the cylinder -/
structure UnfoldedPoint where
  x : ℝ  -- Horizontal distance along unwrapped cylinder
  z : ℝ  -- Vertical distance

/-- The equation of the intersection line on the unfolded surface -/
def intersectionLine (c : Cylinder) (p : InclinedPlane) (point : UnfoldedPoint) : Prop :=
  ∃ (A z₀ : ℝ), point.z = A * Real.sin (point.x / c.R) + z₀

/-- Theorem stating that the intersection line is sinusoidal -/
theorem intersection_is_sinusoid (c : Cylinder) (p : InclinedPlane) :
  ∀ point : UnfoldedPoint, intersectionLine c p point := by
  sorry

end intersection_is_sinusoid_l912_91236


namespace zeros_after_decimal_point_l912_91259

-- Define the fraction
def fraction : ℚ := 3 / (25^25)

-- Define the number of zeros after the decimal point
def num_zeros : ℕ := 18

-- Theorem statement
theorem zeros_after_decimal_point :
  (fraction * (10^num_zeros)).floor = 0 ∧
  (fraction * (10^(num_zeros + 1))).floor ≠ 0 :=
sorry

end zeros_after_decimal_point_l912_91259


namespace k_range_k_trapezoid_l912_91242

-- Define the circles and lines
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2
def circle_N (x y : ℝ) : Prop := x^2 + (y - 8)^2 = 40

def line_l1 (k x y : ℝ) : Prop := y = k * x
def line_l2 (k x y : ℝ) : Prop := y = -1/k * x

-- Define the intersection points
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry
def point_C (k : ℝ) : ℝ × ℝ := sorry
def point_D (k : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
def conditions (k : ℝ) : Prop :=
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ C ≠ D ∧
    circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
    circle_N C.1 C.2 ∧ circle_N D.1 D.2 ∧
    line_l1 k A.1 A.2 ∧ line_l1 k B.1 B.2 ∧
    line_l2 k C.1 C.2 ∧ line_l2 k D.1 D.2

-- Define the trapezoid condition
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := sorry

-- Theorem for the range of k
theorem k_range (k : ℝ) (h : conditions k) : 
  2 - Real.sqrt 3 < k ∧ k < Real.sqrt 15 / 3 := by sorry

-- Theorem for k when ABCD is a trapezoid
theorem k_trapezoid (k : ℝ) (h : conditions k) :
  (∃ (A B C D : ℝ × ℝ), is_trapezoid A B C D) → k = 1 := by sorry

end k_range_k_trapezoid_l912_91242


namespace anitas_class_size_l912_91252

/-- The number of students in Anita's class -/
def num_students : ℕ := 360 / 6

/-- Theorem: The number of students in Anita's class is 60 -/
theorem anitas_class_size :
  num_students = 60 :=
by sorry

end anitas_class_size_l912_91252


namespace total_water_consumption_in_week_l912_91211

/-- Represents the water consumption of a sibling -/
structure WaterConsumption where
  weekday : ℕ
  weekend : ℕ

/-- Calculates the total water consumption for a sibling in a week -/
def weeklyConsumption (wc : WaterConsumption) : ℕ :=
  wc.weekday * 5 + wc.weekend * 2

/-- Theorem: Total water consumption of siblings in a week -/
theorem total_water_consumption_in_week (theo mason roxy zara lily : WaterConsumption)
  (h_theo : theo = { weekday := 8, weekend := 10 })
  (h_mason : mason = { weekday := 7, weekend := 8 })
  (h_roxy : roxy = { weekday := 9, weekend := 11 })
  (h_zara : zara = { weekday := 10, weekend := 12 })
  (h_lily : lily = { weekday := 6, weekend := 7 }) :
  weeklyConsumption theo + weeklyConsumption mason + weeklyConsumption roxy +
  weeklyConsumption zara + weeklyConsumption lily = 296 := by
  sorry


end total_water_consumption_in_week_l912_91211


namespace justice_plants_l912_91282

/-- The number of plants Justice wants in her home -/
def desired_plants (ferns palms succulents additional : ℕ) : ℕ :=
  ferns + palms + succulents + additional

/-- Theorem stating the total number of plants Justice wants -/
theorem justice_plants : 
  desired_plants 3 5 7 9 = 24 := by
  sorry

end justice_plants_l912_91282


namespace train_speed_excluding_stoppages_l912_91241

/-- Proves that given a train with a speed of 90 kmph including stoppages
    and stopping for 15 minutes per hour, the speed of the train excluding
    stoppages is 120 kmph. -/
theorem train_speed_excluding_stoppages
  (speed_with_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_with_stoppages = 90)
  (h2 : stopping_time = 15/60) :
  let running_time := 1 - stopping_time
  speed_with_stoppages * 1 = speed_with_stoppages * running_time →
  speed_with_stoppages / running_time = 120 :=
by
  sorry

#check train_speed_excluding_stoppages

end train_speed_excluding_stoppages_l912_91241


namespace single_digit_sum_l912_91210

theorem single_digit_sum (a b : ℕ) : 
  a ∈ Finset.range 10 ∧ a ≠ 0 ∧
  b ∈ Finset.range 10 ∧ b ≠ 0 ∧
  82 * 10 * a + 7 + 6 * b = 190 →
  a + 2 * b = 7 := by
sorry

end single_digit_sum_l912_91210


namespace mendez_family_mean_age_l912_91201

/-- The Mendez family children problem -/
theorem mendez_family_mean_age :
  let ages : List ℝ := [5, 5, 10, 12, 15]
  let mean_age := (ages.sum) / (ages.length : ℝ)
  mean_age = 9.4 := by
sorry

end mendez_family_mean_age_l912_91201


namespace fraction_to_decimal_l912_91261

theorem fraction_to_decimal : (22 : ℚ) / 8 = 2.75 := by
  sorry

end fraction_to_decimal_l912_91261


namespace cricket_players_l912_91206

/-- The number of students who like to play basketball -/
def B : ℕ := 12

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 3

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 17

/-- The number of students who like to play cricket -/
def C : ℕ := B_or_C - B + B_and_C

theorem cricket_players : C = 8 := by
  sorry

end cricket_players_l912_91206


namespace PQ_length_l912_91285

-- Define the point R
def R : ℝ × ℝ := (10, 8)

-- Define the lines
def line1 (x y : ℝ) : Prop := 7 * y = 24 * x
def line2 (x y : ℝ) : Prop := 13 * y = 5 * x

-- Define P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P is on line1
axiom P_on_line1 : line1 P.1 P.2

-- State that Q is on line2
axiom Q_on_line2 : line2 Q.1 Q.2

-- R is the midpoint of PQ
axiom R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Theorem to prove
theorem PQ_length : 
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 4648 / 277 := by sorry

end PQ_length_l912_91285


namespace plywood_cut_perimeter_difference_l912_91227

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the original plywood and its cuts -/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Checks if a rectangle is a valid cut of the plywood -/
def is_valid_cut (p : Plywood) (r : Rectangle) : Prop :=
  p.length * p.width = p.num_pieces * r.length * r.width

theorem plywood_cut_perimeter_difference (p : Plywood) :
  p.length = 6 ∧ p.width = 9 ∧ p.num_pieces = 6 →
  ∃ (max_r min_r : Rectangle),
    is_valid_cut p max_r ∧
    is_valid_cut p min_r ∧
    ∀ (r : Rectangle), is_valid_cut p r →
      perimeter r ≤ perimeter max_r ∧
      perimeter min_r ≤ perimeter r ∧
      perimeter max_r - perimeter min_r = 10 := by
  sorry

end plywood_cut_perimeter_difference_l912_91227


namespace equation_solution_l912_91266

theorem equation_solution : 
  ∃! x : ℚ, (x - 17) / 3 = (3 * x + 8) / 6 :=
by
  use (-42 : ℚ)
  sorry

end equation_solution_l912_91266


namespace restore_original_example_l912_91297

def original_product : ℕ := 4 * 5 * 4 * 5 * 4

def changed_product : ℕ := 2247

def num_changed_digits : ℕ := 2

theorem restore_original_example :
  (original_product = 2240) ∧
  (∃ (a b : ℕ), a ≠ b ∧ a ≤ 9 ∧ b ≤ 9 ∧
    changed_product = original_product + a * 10 - b) :=
sorry

end restore_original_example_l912_91297


namespace running_speed_calculation_l912_91226

/-- Proves that the running speed is 8 km/hr given the problem conditions --/
theorem running_speed_calculation (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : total_distance = 8)
  (h3 : total_time = 1.5)
  (h4 : total_distance / 2 / walking_speed + total_distance / 2 / running_speed = total_time) :
  running_speed = 8 := by
  sorry


end running_speed_calculation_l912_91226


namespace walnut_trees_planted_l912_91299

/-- The number of walnut trees planted in a park -/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem: The number of walnut trees planted is the difference between
    the final number of trees and the initial number of trees -/
theorem walnut_trees_planted (initial final planted : ℕ) 
  (h1 : initial = 107)
  (h2 : final = 211)
  (h3 : planted = trees_planted initial final) :
  planted = 104 := by
  sorry

end walnut_trees_planted_l912_91299


namespace cookie_radius_l912_91296

/-- The equation of the cookie boundary -/
def cookie_boundary (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 4

/-- The cookie is a circle -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ x y : ℝ, cookie_boundary x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem cookie_radius :
  ∃ center : ℝ × ℝ, is_circle center 3 :=
sorry

end cookie_radius_l912_91296


namespace car_ownership_proof_l912_91280

def total_cars (cathy_cars : ℕ) : ℕ :=
  let carol_cars := 2 * cathy_cars
  let susan_cars := carol_cars - 2
  let lindsey_cars := cathy_cars + 4
  cathy_cars + carol_cars + susan_cars + lindsey_cars

theorem car_ownership_proof (cathy_cars : ℕ) (h : cathy_cars = 5) : total_cars cathy_cars = 32 := by
  sorry

end car_ownership_proof_l912_91280


namespace simplify_expression_l912_91238

theorem simplify_expression : ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := by
  sorry

end simplify_expression_l912_91238


namespace coefficient_corresponds_to_20th_term_l912_91267

/-- The general term of the arithmetic sequence -/
def a (n : ℕ) : ℤ := 3 * n - 5

/-- The coefficient of x^4 in the expansion of (1+x)^k -/
def coeff (k : ℕ) : ℕ := Nat.choose k 4

/-- The theorem stating that the 20th term of the sequence corresponds to
    the coefficient of x^4 in the given expansion -/
theorem coefficient_corresponds_to_20th_term :
  a 20 = (coeff 5 + coeff 6 + coeff 7) := by
  sorry

end coefficient_corresponds_to_20th_term_l912_91267


namespace star_replacement_impossibility_l912_91239

theorem star_replacement_impossibility : ∀ (f : Fin 9 → Bool),
  ∃ (result : ℤ), result ≠ 0 ∧
  result = (if f 0 then 1 else -1) +
           (if f 1 then 2 else -2) +
           (if f 2 then 3 else -3) +
           (if f 3 then 4 else -4) +
           (if f 4 then 5 else -5) +
           (if f 5 then 6 else -6) +
           (if f 6 then 7 else -7) +
           (if f 7 then 8 else -8) +
           (if f 8 then 9 else -9) +
           10 :=
by
  sorry

end star_replacement_impossibility_l912_91239


namespace flowers_left_l912_91253

theorem flowers_left (alissa_flowers melissa_flowers given_flowers : ℕ) :
  alissa_flowers = 16 →
  melissa_flowers = 16 →
  given_flowers = 18 →
  alissa_flowers + melissa_flowers - given_flowers = 14 := by
sorry

end flowers_left_l912_91253


namespace parabola_coeff_sum_l912_91254

/-- A parabola with equation y = ax^2 + bx + c, vertex at (2, 3), and passing through (5, 6) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 2
  vertex_y : ℝ := 3
  point_x : ℝ := 5
  point_y : ℝ := 6
  eq_at_vertex : 3 = a * 2^2 + b * 2 + c
  eq_at_point : 6 = a * 5^2 + b * 5 + c

/-- The sum of coefficients a, b, and c equals 4 -/
theorem parabola_coeff_sum (p : Parabola) : p.a + p.b + p.c = 4 := by
  sorry

end parabola_coeff_sum_l912_91254


namespace solution_set_of_equations_l912_91298

theorem solution_set_of_equations (x y : ℝ) :
  (x^2 - 2*x*y = 1 ∧ 5*x^2 - 2*x*y + 2*y^2 = 5) ↔
  ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ (x = 1/3 ∧ y = -4/3) ∨ (x = -1/3 ∧ y = 4/3)) :=
by sorry

end solution_set_of_equations_l912_91298


namespace integral_equation_solution_l912_91237

theorem integral_equation_solution (k : ℝ) : 
  (∫ (x : ℝ), 2*x - 3*x^2) = 0 → k = 0 ∨ k = 1 :=
by sorry

end integral_equation_solution_l912_91237
