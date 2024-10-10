import Mathlib

namespace quadratic_sufficient_not_necessary_l3631_363189

theorem quadratic_sufficient_not_necessary (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ 
  ((a > 0 ∧ b^2 - 4*a*c < 0) ∨ 
   ∃ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' > 0) ∧ ¬(a' > 0 ∧ b'^2 - 4*a'*c' < 0)) :=
by sorry

end quadratic_sufficient_not_necessary_l3631_363189


namespace evenBlueFaceCubesFor642Block_l3631_363184

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with an even number of blue faces in a painted block -/
def evenBlueFaceCubes (b : Block) : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that a 6x4x2 inch block has 20 cubes with an even number of blue faces -/
theorem evenBlueFaceCubesFor642Block :
  evenBlueFaceCubes { length := 6, width := 4, height := 2 } = 20 := by
  sorry

end evenBlueFaceCubesFor642Block_l3631_363184


namespace least_odd_prime_factor_of_2100_8_plus_1_l3631_363194

theorem least_odd_prime_factor_of_2100_8_plus_1 :
  (Nat.minFac (2100^8 + 1)) = 193 := by
  sorry

end least_odd_prime_factor_of_2100_8_plus_1_l3631_363194


namespace x_lt_one_necessary_not_sufficient_l3631_363136

theorem x_lt_one_necessary_not_sufficient :
  ∀ x : ℝ,
  (∀ x, (1 / x > 1 → x < 1)) ∧
  (∃ x, x < 1 ∧ ¬(1 / x > 1)) :=
by sorry

end x_lt_one_necessary_not_sufficient_l3631_363136


namespace parallelogram_solution_l3631_363146

-- Define the parallelogram EFGH
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ

-- Define the specific parallelogram from the problem
def specificParallelogram : Parallelogram where
  EF := 45
  FG := fun y ↦ 4 * y^2
  GH := fun x ↦ 3 * x + 6
  HE := 32

-- Theorem statement
theorem parallelogram_solution (p : Parallelogram) 
  (h1 : p = specificParallelogram) : 
  ∃ (x y : ℝ), p.GH x = p.EF ∧ p.FG y = p.HE ∧ x = 13 ∧ y = 2 * Real.sqrt 2 := by
  sorry

#check parallelogram_solution

end parallelogram_solution_l3631_363146


namespace number_difference_l3631_363169

theorem number_difference (n : ℕ) (h : n = 15) : n * 13 - n = 180 := by
  sorry

end number_difference_l3631_363169


namespace library_book_count_l3631_363113

theorem library_book_count (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 →
  loaned_books = 40 →
  return_rate = 4/5 →
  initial_books - loaned_books + (return_rate * loaned_books).floor = 67 := by
sorry

end library_book_count_l3631_363113


namespace max_value_of_z_l3631_363147

-- Define the objective function
def z (x y : ℝ) : ℝ := 4 * x + 3 * y

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x - y - 2 ≥ 0 ∧ 2 * x + y - 2 ≤ 0 ∧ y + 4 ≥ 0

-- Theorem statement
theorem max_value_of_z :
  ∃ (max : ℝ), max = 8 ∧
  (∀ x y : ℝ, feasible_region x y → z x y ≤ max) ∧
  (∃ x y : ℝ, feasible_region x y ∧ z x y = max) :=
sorry

end max_value_of_z_l3631_363147


namespace monotonic_function_characterization_l3631_363133

-- Define the types of our functions
def MonotonicFunction (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y → f x ≤ f y

def StrictlyMonotonicFunction (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem monotonic_function_characterization 
  (u : ℝ → ℝ) 
  (h_u_monotonic : MonotonicFunction u) 
  (h_exists_f : ∃ f : ℝ → ℝ, 
    StrictlyMonotonicFunction f ∧ 
    (∀ x y : ℝ, f (x + y) = f x * u y + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, u x = Real.exp (k * x) := by
sorry

end monotonic_function_characterization_l3631_363133


namespace card_distribution_proof_l3631_363150

/-- Represents the number of cards each player has -/
structure CardDistribution :=
  (alfred : ℕ)
  (bruno : ℕ)
  (christophe : ℕ)
  (damien : ℕ)

/-- The total number of cards in the deck -/
def totalCards : ℕ := 32

/-- Redistribution function for Alfred -/
def redistributeAlfred (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred - d.alfred / 2,
    bruno := d.bruno + d.alfred / 4,
    christophe := d.christophe + d.alfred / 4,
    damien := d.damien }

/-- Redistribution function for Bruno -/
def redistributeBruno (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred + d.bruno / 4,
    bruno := d.bruno - d.bruno / 2,
    christophe := d.christophe + d.bruno / 4,
    damien := d.damien }

/-- Redistribution function for Christophe -/
def redistributeChristophe (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred + d.christophe / 4,
    bruno := d.bruno + d.christophe / 4,
    christophe := d.christophe - d.christophe / 2,
    damien := d.damien }

/-- The initial distribution of cards -/
def initialDistribution : CardDistribution :=
  { alfred := 4, bruno := 7, christophe := 13, damien := 8 }

theorem card_distribution_proof :
  let finalDist := redistributeChristophe (redistributeBruno (redistributeAlfred initialDistribution))
  (finalDist.alfred = finalDist.bruno) ∧
  (finalDist.bruno = finalDist.christophe) ∧
  (finalDist.christophe = finalDist.damien) ∧
  (finalDist.alfred + finalDist.bruno + finalDist.christophe + finalDist.damien = totalCards) :=
by sorry

end card_distribution_proof_l3631_363150


namespace class_composition_l3631_363181

theorem class_composition (boys_avg : ℝ) (girls_avg : ℝ) (class_avg : ℝ) :
  boys_avg = 4 →
  girls_avg = 3.25 →
  class_avg = 3.6 →
  ∃ (boys girls : ℕ),
    boys + girls > 30 ∧
    boys + girls < 50 ∧
    (boys_avg * boys + girls_avg * girls) / (boys + girls) = class_avg ∧
    boys = 21 ∧
    girls = 24 := by
  sorry

end class_composition_l3631_363181


namespace square_perimeter_sum_l3631_363112

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 45) :
  4*x + 4*y = 4*Real.sqrt 65 + 8*Real.sqrt 5 := by
  sorry

end square_perimeter_sum_l3631_363112


namespace x_equals_seven_l3631_363163

theorem x_equals_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 := by
  sorry

end x_equals_seven_l3631_363163


namespace divisible_by_forty_l3631_363166

theorem divisible_by_forty (n : ℕ) 
  (h1 : ∃ k : ℕ, 2 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℕ, 3 * n + 1 = m ^ 2) : 
  40 ∣ n := by
sorry

end divisible_by_forty_l3631_363166


namespace total_animals_l3631_363198

/-- Given a field with cows, sheep, and goats, calculate the total number of animals -/
theorem total_animals (cows sheep goats : ℕ) 
  (h_cows : cows = 40)
  (h_sheep : sheep = 56)
  (h_goats : goats = 104) :
  cows + sheep + goats = 200 := by
  sorry

end total_animals_l3631_363198


namespace reading_time_difference_l3631_363111

/-- Proves that the difference in reading time between two readers is 144 minutes given their reading rates and book length. -/
theorem reading_time_difference 
  (xanthia_rate : ℝ) 
  (molly_rate : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_rate = 75) 
  (h2 : molly_rate = 45) 
  (h3 : book_pages = 270) : 
  (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 144 := by
  sorry

end reading_time_difference_l3631_363111


namespace range_of_f_on_interval_l3631_363144

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem range_of_f_on_interval :
  ∃ (y : ℝ), y ∈ Set.Icc 0 (Real.exp (Real.pi / 2)) ↔
  ∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ f x = y :=
by sorry

end range_of_f_on_interval_l3631_363144


namespace allyns_june_expenses_l3631_363174

/-- Calculates the total monthly electricity expenses for a given number of bulbs --/
def calculate_monthly_expenses (
  bulb_wattage : ℕ)  -- Wattage of each bulb
  (num_bulbs : ℕ)    -- Number of bulbs
  (days_in_month : ℕ) -- Number of days in the month
  (cost_per_watt : ℚ) -- Cost per watt in dollars
  : ℚ :=
  (bulb_wattage * num_bulbs * days_in_month : ℚ) * cost_per_watt

/-- Theorem stating that Allyn's monthly electricity expenses for June are $14400 --/
theorem allyns_june_expenses :
  calculate_monthly_expenses 60 40 30 (20 / 100) = 14400 := by
  sorry

end allyns_june_expenses_l3631_363174


namespace ladder_problem_l3631_363135

-- Define the ladder setup
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the theorem
theorem ladder_problem :
  -- Part 1: Horizontal distance
  ∃ (horizontal_distance : ℝ),
    horizontal_distance^2 + wall_height^2 = ladder_length^2 ∧
    horizontal_distance = 5 ∧
  -- Part 2: Height reached by 8-meter ladder
  ∃ (height_8m : ℝ),
    height_8m = (wall_height * 8) / ladder_length ∧
    height_8m = 96 / 13 := by
  sorry

end ladder_problem_l3631_363135


namespace right_triangle_ratio_l3631_363138

theorem right_triangle_ratio (x d : ℝ) (h1 : x > d) (h2 : d > 0) : 
  (x^2)^2 + (x^2 - d)^2 = (x^2 + d)^2 → x / d = 8 := by
  sorry

end right_triangle_ratio_l3631_363138


namespace equation_solution_l3631_363117

theorem equation_solution : 
  ∃! x : ℚ, (x - 30) / 3 = (5 - 3 * x) / 4 + 2 :=
by
  use 15
  sorry

end equation_solution_l3631_363117


namespace equation_solution_l3631_363199

theorem equation_solution : ∃ y : ℝ, (2 / y + 3 / y / (6 / y) = 1.5) ∧ y = 2 := by
  sorry

end equation_solution_l3631_363199


namespace vector_subtraction_l3631_363154

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end vector_subtraction_l3631_363154


namespace imaginary_part_of_z_l3631_363191

theorem imaginary_part_of_z (i : ℂ) : i * i = -1 → Complex.im ((1 + i) / i) = -1 := by
  sorry

end imaginary_part_of_z_l3631_363191


namespace wire_ratio_theorem_l3631_363193

theorem wire_ratio_theorem (B C : ℝ) (h1 : B > 0) (h2 : C > 0) (h3 : B + C = 80) : 
  ∃ (r : ℝ → ℝ → ℝ → Prop), r 16 B C ∧ 
  (∀ (x y z : ℝ), r x y z ↔ ∃ (k : ℝ), k > 0 ∧ x = 16 * k ∧ y = B * k ∧ z = C * k) :=
sorry

end wire_ratio_theorem_l3631_363193


namespace min_ones_23x23_l3631_363187

/-- Represents a tiling of a square grid --/
structure Tiling (n : ℕ) :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (valid : ones + 4 * twos + 9 * threes = n^2)

/-- The minimum number of 1x1 squares in a valid 23x23 tiling --/
def min_ones : ℕ := 1

theorem min_ones_23x23 :
  ∀ (t : Tiling 23), t.ones ≥ min_ones :=
sorry

end min_ones_23x23_l3631_363187


namespace power_equation_solution_l3631_363149

theorem power_equation_solution :
  ∃ x : ℕ, (1000 : ℝ)^7 / (10 : ℝ)^x = 10000 ∧ x = 17 := by sorry

end power_equation_solution_l3631_363149


namespace power_of_product_l3631_363178

theorem power_of_product (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_of_product_l3631_363178


namespace right_triangle_leg_sum_l3631_363175

/-- A right triangle with consecutive even number legs and hypotenuse 34 has leg sum 50 -/
theorem right_triangle_leg_sum (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 34 →  -- Hypotenuse is 34
  ∃ k : ℕ, a = 2*k ∧ b = 2*k + 2 →  -- Legs are consecutive even numbers
  a + b = 50 := by
sorry

end right_triangle_leg_sum_l3631_363175


namespace valuable_files_count_l3631_363134

def initial_download : ℕ := 800
def first_deletion_rate : ℚ := 70 / 100
def second_download : ℕ := 400
def second_deletion_rate : ℚ := 3 / 5

theorem valuable_files_count : 
  (initial_download - (initial_download * first_deletion_rate).floor) + 
  (second_download - (second_download * second_deletion_rate).floor) = 400 :=
by sorry

end valuable_files_count_l3631_363134


namespace most_likely_top_quality_count_l3631_363179

/-- The proportion of top-quality products -/
def p : ℝ := 0.31

/-- The number of products in the batch -/
def n : ℕ := 75

/-- The most likely number of top-quality products in the batch -/
def most_likely_count : ℕ := 23

/-- Theorem stating that the most likely number of top-quality products in the batch is 23 -/
theorem most_likely_top_quality_count :
  ⌊n * p⌋ = most_likely_count ∧
  (n * p - (1 - p) ≤ most_likely_count) ∧
  (most_likely_count ≤ n * p + p) :=
sorry

end most_likely_top_quality_count_l3631_363179


namespace product_of_solutions_l3631_363195

theorem product_of_solutions (x : ℝ) : 
  (x^2 + 6*x - 21 = 0) → 
  (∃ α β : ℝ, (α + β = -6) ∧ (α * β = -21)) := by
sorry

end product_of_solutions_l3631_363195


namespace f_is_quadratic_l3631_363172

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l3631_363172


namespace two_numbers_problem_l3631_363127

theorem two_numbers_problem (x y : ℝ) :
  (2 * (x + y) = x^2 - y^2) ∧ (2 * (x + y) = x * y / 4 - 56) →
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := by
  sorry

end two_numbers_problem_l3631_363127


namespace sufficient_but_not_necessary_l3631_363197

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (((0 ≤ a) ∧ (a ≤ 1) ∧ (0 ≤ b) ∧ (b ≤ 1)) → (0 ≤ a * b) ∧ (a * b ≤ 1)) ∧ 
  (∃ (a b : ℝ), ((0 ≤ a * b) ∧ (a * b ≤ 1)) ∧ ¬((0 ≤ a) ∧ (a ≤ 1) ∧ (0 ≤ b) ∧ (b ≤ 1))) :=
by sorry

end sufficient_but_not_necessary_l3631_363197


namespace bowtie_equation_solution_l3631_363104

-- Define the ⊗ operation
noncomputable def bowtie (a b : ℝ) : ℝ := a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem bowtie_equation_solution (h : ℝ) :
  bowtie 4 h = 10 → h = 2 := by
  sorry

end bowtie_equation_solution_l3631_363104


namespace logarithm_sum_l3631_363162

theorem logarithm_sum (a : ℝ) (h : 1 + a^3 = 9) : 
  Real.log a / Real.log (1/4) + Real.log 8 / Real.log a = 5/2 := by
  sorry

end logarithm_sum_l3631_363162


namespace problem_statement_l3631_363124

theorem problem_statement (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end problem_statement_l3631_363124


namespace square_areas_and_perimeters_l3631_363141

theorem square_areas_and_perimeters (x : ℝ) : 
  (x^2 + 8*x + 16 = (x + 4)^2) ∧ 
  (4*x^2 - 12*x + 9 = (2*x - 3)^2) ∧ 
  (4*(x + 4) + 4*(2*x - 3) = 32) → 
  x = 7/3 := by
sorry

end square_areas_and_perimeters_l3631_363141


namespace monomial_polynomial_multiplication_l3631_363182

theorem monomial_polynomial_multiplication :
  ∀ (x y : ℝ), -3 * x * y * (4 * y - 2 * x - 1) = -12 * x * y^2 + 6 * x^2 * y + 3 * x * y := by
  sorry

end monomial_polynomial_multiplication_l3631_363182


namespace goose_eggs_count_goose_eggs_solution_l3631_363151

theorem goose_eggs_count : ℕ → Prop :=
  fun total_eggs =>
    let hatched := (1 : ℚ) / 4 * total_eggs
    let survived_first_month := (4 : ℚ) / 5 * hatched
    let survived_six_months := (2 : ℚ) / 5 * survived_first_month
    let survived_first_year := (4 : ℚ) / 7 * survived_six_months
    survived_first_year = 120 ∧ total_eggs = 2625

/-- The number of goose eggs laid at the pond is 2625. -/
theorem goose_eggs_solution : goose_eggs_count 2625 := by
  sorry

end goose_eggs_count_goose_eggs_solution_l3631_363151


namespace units_digit_sum_factorials_plus_double_factorial_l3631_363120

/-- Double factorial of a natural number -/
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

/-- Sum of factorials from 1 to n -/
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => Nat.factorial (i + 1))

/-- Theorem: The units digit of the sum of factorials from 1 to 12 plus 12!! is 3 -/
theorem units_digit_sum_factorials_plus_double_factorial :
  (sum_factorials 12 + double_factorial 12) % 10 = 3 := by
  sorry

end units_digit_sum_factorials_plus_double_factorial_l3631_363120


namespace inequality_solution_l3631_363170

theorem inequality_solution (a x : ℝ) : 
  (x - a) * (x - a^2) < 0 ↔ 
  ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) ∨ 
  (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a) ∨ 
  (a = 0 ∨ a = 1 ∧ False) :=
by sorry

end inequality_solution_l3631_363170


namespace semicircles_area_ratio_l3631_363161

theorem semicircles_area_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let semicircle1_area := (π * r^2) / 2
  let semicircle2_area := (π * (r/2)^2) / 2
  (semicircle1_area + semicircle2_area) / circle_area = 5/8 := by
sorry

end semicircles_area_ratio_l3631_363161


namespace fifteenth_triangular_sum_fifteenth_sixteenth_triangular_l3631_363121

/-- Triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular : triangular_number 15 = 120 := by sorry

/-- The sum of the 15th and 16th triangular numbers is 256 -/
theorem sum_fifteenth_sixteenth_triangular : 
  triangular_number 15 + triangular_number 16 = 256 := by sorry

end fifteenth_triangular_sum_fifteenth_sixteenth_triangular_l3631_363121


namespace total_pencils_count_l3631_363140

/-- The number of people in the group -/
def num_people : ℕ := 5

/-- The number of pencils each person has -/
def pencils_per_person : ℕ := 15

/-- The total number of pencils for the group -/
def total_pencils : ℕ := num_people * pencils_per_person

theorem total_pencils_count : total_pencils = 75 := by
  sorry

end total_pencils_count_l3631_363140


namespace lucas_change_l3631_363110

def banana_cost : ℚ := 70 / 100
def orange_cost : ℚ := 80 / 100
def banana_quantity : ℕ := 5
def orange_quantity : ℕ := 2
def paid_amount : ℚ := 10

def total_cost : ℚ := banana_cost * banana_quantity + orange_cost * orange_quantity

theorem lucas_change :
  paid_amount - total_cost = 490 / 100 := by sorry

end lucas_change_l3631_363110


namespace empty_fixed_implies_empty_stable_l3631_363122

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Set of fixed points -/
def A (a b c : ℝ) : Set ℝ := {x | f a b c x = x}

/-- Set of stable points -/
def B (a b c : ℝ) : Set ℝ := {x | f a b c (f a b c x) = x}

/-- Theorem: If A is empty, then B is empty for quadratic functions -/
theorem empty_fixed_implies_empty_stable (a b c : ℝ) (ha : a ≠ 0) :
  A a b c = ∅ → B a b c = ∅ := by sorry

end empty_fixed_implies_empty_stable_l3631_363122


namespace negation_of_square_nonnegative_l3631_363100

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end negation_of_square_nonnegative_l3631_363100


namespace quadratic_coefficients_l3631_363119

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 2 = 3*x

-- Define the standard form of the quadratic equation
def standard_form (a b c x : ℝ) : Prop := a*x^2 + b*x + c = 0

-- Theorem statement
theorem quadratic_coefficients :
  ∃ (c : ℝ), ∀ (x : ℝ),
    quadratic_equation x ↔ standard_form 1 (-3) c x :=
by sorry

end quadratic_coefficients_l3631_363119


namespace smaller_std_dev_more_stable_smaller_variance_more_stable_smaller_mean_not_necessarily_more_stable_l3631_363159

-- Define a dataset as a list of real numbers
def Dataset := List ℝ

-- Define standard deviation
def standardDeviation (data : Dataset) : ℝ :=
  sorry

-- Define variance
def variance (data : Dataset) : ℝ :=
  sorry

-- Define mean
def mean (data : Dataset) : ℝ :=
  sorry

-- Define a measure of concentration and stability
def isConcentratedAndStable (data : Dataset) : Prop :=
  sorry

-- Theorem stating that smaller standard deviation implies more concentrated and stable distribution
theorem smaller_std_dev_more_stable (data1 data2 : Dataset) :
  standardDeviation data1 < standardDeviation data2 →
  isConcentratedAndStable data1 → isConcentratedAndStable data2 :=
sorry

-- Theorem stating that smaller variance implies more concentrated and stable distribution
theorem smaller_variance_more_stable (data1 data2 : Dataset) :
  variance data1 < variance data2 →
  isConcentratedAndStable data1 → isConcentratedAndStable data2 :=
sorry

-- Theorem stating that smaller mean does not necessarily imply more concentrated and stable distribution
theorem smaller_mean_not_necessarily_more_stable :
  ∃ (data1 data2 : Dataset), mean data1 < mean data2 ∧
  isConcentratedAndStable data2 ∧ ¬isConcentratedAndStable data1 :=
sorry

end smaller_std_dev_more_stable_smaller_variance_more_stable_smaller_mean_not_necessarily_more_stable_l3631_363159


namespace remainder_divisibility_l3631_363115

theorem remainder_divisibility (x : ℤ) : 
  (∃ k : ℤ, x = 63 * k + 27) → (∃ m : ℤ, x = 8 * m + 3) :=
by sorry

end remainder_divisibility_l3631_363115


namespace unique_face_reconstruction_l3631_363103

/-- Represents the numbers on the faces of a cube -/
structure CubeFaces where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Represents the sums on the edges of a cube -/
structure CubeEdges where
  ab : ℝ
  ac : ℝ
  ad : ℝ
  ae : ℝ
  bc : ℝ
  bf : ℝ
  cf : ℝ
  df : ℝ
  de : ℝ
  ef : ℝ
  bd : ℝ
  ce : ℝ

/-- Function to calculate edge sums from face numbers -/
def edgeSumsFromFaces (faces : CubeFaces) : CubeEdges :=
  { ab := faces.a + faces.b
  , ac := faces.a + faces.c
  , ad := faces.a + faces.d
  , ae := faces.a + faces.e
  , bc := faces.b + faces.c
  , bf := faces.b + faces.f
  , cf := faces.c + faces.f
  , df := faces.d + faces.f
  , de := faces.d + faces.e
  , ef := faces.e + faces.f
  , bd := faces.b + faces.d
  , ce := faces.c + faces.e }

/-- Theorem stating that face numbers can be uniquely reconstructed from edge sums -/
theorem unique_face_reconstruction (edges : CubeEdges) : 
  ∃! faces : CubeFaces, edgeSumsFromFaces faces = edges := by
  sorry


end unique_face_reconstruction_l3631_363103


namespace chord_length_l3631_363168

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := { center := (0, 0), radius := 6 }
def C2 : Circle := { center := (18, 0), radius := 12 }
def C3 : Circle := { center := (38, 0), radius := 38 }
def C4 : Circle := { center := (58, 0), radius := 20 }

-- Define the properties of the circles
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c2.radius - c1.radius)^2

def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Theorem statement
theorem chord_length :
  externally_tangent C1 C2 ∧
  internally_tangent C1 C3 ∧
  internally_tangent C2 C3 ∧
  externally_tangent C3 C4 ∧
  collinear C1.center C2.center C3.center →
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 7 :=
sorry

end chord_length_l3631_363168


namespace f_of_one_equals_two_l3631_363188

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem f_of_one_equals_two : f 1 = 2 := by
  sorry

end f_of_one_equals_two_l3631_363188


namespace right_angled_triangle_345_l3631_363130

theorem right_angled_triangle_345 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a / b = 3 / 4) (h5 : b / c = 4 / 5) : a^2 + b^2 = c^2 := by
sorry


end right_angled_triangle_345_l3631_363130


namespace halloween_candy_l3631_363143

theorem halloween_candy (debby_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) :
  debby_candy = 32 →
  sister_candy = 42 →
  remaining_candy = 39 →
  debby_candy + sister_candy - remaining_candy = 35 :=
by
  sorry

end halloween_candy_l3631_363143


namespace set_intersection_and_union_l3631_363148

def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+2)*x + 2*a^2 - a + 1 = 0}

theorem set_intersection_and_union (a : ℝ) :
  (A ∩ B a = {2} → a = 1/2) ∧
  (A ∪ B a = A → a ≤ 0 ∨ a = 1 ∨ a > 8/7) := by
  sorry

end set_intersection_and_union_l3631_363148


namespace quadratic_function_passes_through_points_l3631_363171

/-- The quadratic function f(x) = x² + 2x - 3 passes through the points (0, -3), (1, 0), and (-3, 0). -/
theorem quadratic_function_passes_through_points :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x - 3
  f 0 = -3 ∧ f 1 = 0 ∧ f (-3) = 0 := by
  sorry

end quadratic_function_passes_through_points_l3631_363171


namespace students_just_passed_l3631_363160

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) :
  total = 300 →
  first_div_percent = 29 / 100 →
  second_div_percent = 54 / 100 →
  (total : ℚ) * (1 - first_div_percent - second_div_percent) = 51 := by
sorry

end students_just_passed_l3631_363160


namespace area_of_square_II_l3631_363185

/-- Given a square I with diagonal √3ab, prove that the area of a square II 
    with three times the area of square I is 9(ab)²/2 -/
theorem area_of_square_II (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  let diagonal_I := Real.sqrt 3 * a * b
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 3 * area_I
  area_II = 9 * (a * b) ^ 2 / 2 := by
sorry

end area_of_square_II_l3631_363185


namespace sum_of_ages_in_three_years_l3631_363190

-- Define the current ages
def jeremy_current_age : ℕ := 40
def sebastian_current_age : ℕ := jeremy_current_age + 4
def sophia_future_age : ℕ := 60

-- Define the ages in three years
def jeremy_future_age : ℕ := jeremy_current_age + 3
def sebastian_future_age : ℕ := sebastian_current_age + 3
def sophia_current_age : ℕ := sophia_future_age - 3

-- Theorem to prove
theorem sum_of_ages_in_three_years :
  jeremy_future_age + sebastian_future_age + sophia_future_age = 150 := by
  sorry

end sum_of_ages_in_three_years_l3631_363190


namespace hardcover_non_fiction_count_l3631_363155

/-- Represents the number of books in Thabo's collection -/
def total_books : ℕ := 500

/-- Represents the fraction of fiction books in the collection -/
def fiction_fraction : ℚ := 2/5

/-- Represents the fraction of non-fiction books in the collection -/
def non_fiction_fraction : ℚ := 3/5

/-- Represents the difference between paperback and hardcover non-fiction books -/
def non_fiction_difference : ℕ := 50

/-- Represents the ratio of paperback to hardcover fiction books -/
def fiction_ratio : ℕ := 2

theorem hardcover_non_fiction_count :
  ∃ (hnf : ℕ), 
    (hnf : ℚ) + (hnf + non_fiction_difference : ℚ) = total_books * non_fiction_fraction ∧
    hnf = 125 := by
  sorry

end hardcover_non_fiction_count_l3631_363155


namespace soda_cost_lucille_soda_cost_l3631_363139

/-- The cost of Lucille's soda given her weeding earnings and remaining money -/
theorem soda_cost (cents_per_weed : ℕ) (flower_bed_weeds : ℕ) (vegetable_patch_weeds : ℕ) 
  (grass_weeds : ℕ) (remaining_cents : ℕ) : ℕ :=
  let total_weeds := flower_bed_weeds + vegetable_patch_weeds + grass_weeds / 2
  let total_earnings := total_weeds * cents_per_weed
  total_earnings - remaining_cents

/-- Proof that Lucille's soda cost 99 cents -/
theorem lucille_soda_cost : soda_cost 6 11 14 32 147 = 99 := by
  sorry

end soda_cost_lucille_soda_cost_l3631_363139


namespace mean_temperature_is_87_5_l3631_363123

def temperatures : List ℝ := [82, 80, 83, 88, 90, 92, 90, 95]

theorem mean_temperature_is_87_5 :
  (temperatures.sum / temperatures.length : ℝ) = 87.5 := by
  sorry

end mean_temperature_is_87_5_l3631_363123


namespace bananas_bought_l3631_363145

theorem bananas_bought (initial : ℕ) (eaten : ℕ) (remaining : ℕ) : 
  eaten = 1 → remaining = 11 → initial = eaten + remaining := by sorry

end bananas_bought_l3631_363145


namespace f_range_l3631_363105

def f (x : ℝ) : ℝ := -x^2

theorem f_range :
  ∀ y ∈ Set.range (f ∘ (Set.Icc (-3) 1).restrict f), -9 ≤ y ∧ y ≤ 0 :=
by sorry

end f_range_l3631_363105


namespace march_temperature_data_inconsistent_l3631_363142

/-- Represents the statistical data for March temperatures --/
structure MarchTemperatureData where
  mean : ℝ
  median : ℝ
  variance : ℝ
  mean_eq_zero : mean = 0
  median_eq_four : median = 4
  variance_eq : variance = 15.917

/-- Theorem stating that the given data is inconsistent --/
theorem march_temperature_data_inconsistent (data : MarchTemperatureData) :
  (data.mean - data.median)^2 > data.variance := by
  sorry

#check march_temperature_data_inconsistent

end march_temperature_data_inconsistent_l3631_363142


namespace detergent_per_pound_l3631_363137

/-- Given that Mrs. Hilt used 18 ounces of detergent to wash 9 pounds of clothes,
    prove that she uses 2 ounces of detergent per pound of clothes. -/
theorem detergent_per_pound (total_detergent : ℝ) (total_clothes : ℝ) 
  (h1 : total_detergent = 18) 
  (h2 : total_clothes = 9) : 
  total_detergent / total_clothes = 2 := by
sorry

end detergent_per_pound_l3631_363137


namespace fraction_irreducible_l3631_363156

theorem fraction_irreducible (n : ℤ) : 
  Nat.gcd (Int.natAbs (2*n^2 + 9*n - 17)) (Int.natAbs (n + 6)) = 1 := by
  sorry

end fraction_irreducible_l3631_363156


namespace problem_solution_l3631_363116

theorem problem_solution (p q r : ℝ) 
  (h1 : p / q = 5 / 4)
  (h2 : p = r^2)
  (h3 : Real.sin r = 3 / 5) : 
  2 * p + q = 44.8 := by
  sorry

end problem_solution_l3631_363116


namespace right_triangle_sets_l3631_363108

theorem right_triangle_sets : ∃! (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 5 ∧ b = 2 ∧ c = 5) ∨
   (a = 5 ∧ b = 12 ∧ c = 13)) ∧
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end right_triangle_sets_l3631_363108


namespace initial_mixture_amount_l3631_363196

/-- Represents the problem of finding the initial amount of mixture -/
theorem initial_mixture_amount (initial_mixture : ℝ) : 
  (0.1 * initial_mixture / initial_mixture = 0.1) →  -- Initial mixture is 10% grape juice
  (0.25 * (initial_mixture + 10) = 0.1 * initial_mixture + 10) →  -- Resulting mixture is 25% grape juice
  initial_mixture = 50 := by
  sorry

end initial_mixture_amount_l3631_363196


namespace limit_special_function_l3631_363128

/-- The limit of (2 - e^(x^2))^(1 / (1 - cos(π * x))) as x approaches 0 is e^(-2 / π^2) -/
theorem limit_special_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((2 - Real.exp (x^2))^(1 / (1 - Real.cos (π * x)))) - Real.exp (-2 / π^2)| < ε :=
by sorry

end limit_special_function_l3631_363128


namespace existence_of_x0_l3631_363153

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem existence_of_x0 
  (hcont : ContinuousOn f (Set.Icc 0 1))
  (hdiff : DifferentiableOn ℝ f (Set.Ioo 0 1))
  (hf0 : f 0 = 1)
  (hf1 : f 1 = 0) :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ 
    |deriv f x0| ≥ 2018 * (f x0)^2018 := by
  sorry

end existence_of_x0_l3631_363153


namespace puzzle_pieces_missing_l3631_363167

theorem puzzle_pieces_missing (total : ℕ) (border : ℕ) (trevor : ℕ) (joe : ℕ) 
  (h1 : total = 500)
  (h2 : border = 75)
  (h3 : trevor = 105)
  (h4 : joe = 3 * trevor) :
  total - (border + trevor + joe) = 5 := by
  sorry

end puzzle_pieces_missing_l3631_363167


namespace intersection_implies_a_leq_neg_one_l3631_363165

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3)}
def B (a : ℝ) : Set ℝ := {x | ∃ y, y = Real.sqrt (a - x)}

-- State the theorem
theorem intersection_implies_a_leq_neg_one (a : ℝ) : A ∩ B a = B a → a ≤ -1 := by
  sorry

end intersection_implies_a_leq_neg_one_l3631_363165


namespace point_coordinates_in_third_quadrant_l3631_363118

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- The distance between a point and the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate for a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y < 0

theorem point_coordinates_in_third_quadrant 
  (M : Point) 
  (h1 : isInThirdQuadrant M) 
  (h2 : distToXAxis M = 1) 
  (h3 : distToYAxis M = 2) : 
  M.x = -2 ∧ M.y = -1 := by
  sorry

end point_coordinates_in_third_quadrant_l3631_363118


namespace arithmetic_geometric_mean_ratio_l3631_363107

theorem arithmetic_geometric_mean_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |a / b - 34| < ε :=
sorry

end arithmetic_geometric_mean_ratio_l3631_363107


namespace process_never_stops_l3631_363125

/-- Represents a large number as a list of digits -/
def LargeNumber := List Nat

/-- The initial number with 900 digits, all 1s -/
def initial_number : LargeNumber := List.replicate 900 1

/-- Extracts the last two digits of a LargeNumber -/
def last_two_digits (n : LargeNumber) : Nat :=
  match n.reverse with
  | d1 :: d2 :: _ => d1 + 10 * d2
  | _ => 0

/-- Applies the transformation rule to a LargeNumber -/
def transform (n : LargeNumber) : Nat :=
  let a := n.foldl (fun acc d => acc * 10 + d) 0 / 100
  let b := last_two_digits n
  2 * a + 8 * b

/-- Predicate to check if a number is less than 100 -/
def is_less_than_100 (n : Nat) : Prop := n < 100

/-- Main theorem: The process will never stop -/
theorem process_never_stops :
  ∀ n : Nat, ∃ m : Nat, m > n ∧ ¬(is_less_than_100 (transform (List.replicate m 1))) :=
  sorry


end process_never_stops_l3631_363125


namespace second_train_length_second_train_length_solution_l3631_363152

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 165.12 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, abs (second_train_length 80 65 7.0752960452818945 120 - 165.12) < ε :=
by sorry

end second_train_length_second_train_length_solution_l3631_363152


namespace third_number_in_expression_l3631_363126

theorem third_number_in_expression (x : ℝ) : 
  (26.3 * 12 * x) / 3 + 125 = 2229 → x = 20 := by
  sorry

end third_number_in_expression_l3631_363126


namespace triangle_problem_l3631_363132

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end triangle_problem_l3631_363132


namespace cos_90_degrees_l3631_363129

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_90_degrees_l3631_363129


namespace smallest_n_square_and_cube_l3631_363176

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (j : ℕ), 5 * n = j^3) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (k : ℕ), 4 * m = k^2) → 
    (∃ (j : ℕ), 5 * m = j^3) → 
    m ≥ n) ∧
  n = 125 :=
sorry

end smallest_n_square_and_cube_l3631_363176


namespace sum_of_products_l3631_363164

theorem sum_of_products (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + 2*x^5 + x^4 + x^3 + x^2 + 2*x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) : 
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 2 := by
sorry

end sum_of_products_l3631_363164


namespace porter_buns_problem_l3631_363106

/-- Calculates the maximum number of buns that can be transported to a construction site. -/
def max_buns_transported (total_buns : ℕ) (buns_per_trip : ℕ) (buns_eaten_per_way : ℕ) : ℕ :=
  let num_trips : ℕ := total_buns / buns_per_trip
  let buns_eaten : ℕ := 2 * (num_trips - 1) * buns_eaten_per_way + buns_eaten_per_way
  total_buns - buns_eaten

/-- Theorem stating that given 200 total buns, 40 buns carried per trip, and 1 bun eaten per one-way trip,
    the maximum number of buns that can be transported to the construction site is 191. -/
theorem porter_buns_problem :
  max_buns_transported 200 40 1 = 191 := by
  sorry

end porter_buns_problem_l3631_363106


namespace initial_workers_correct_l3631_363186

/-- The number of workers initially working on the job -/
def initial_workers : ℕ := 6

/-- The number of days to finish the job initially -/
def initial_days : ℕ := 8

/-- The number of days worked before new workers join -/
def days_before_join : ℕ := 3

/-- The number of new workers that join -/
def new_workers : ℕ := 4

/-- The number of additional days needed to finish the job after new workers join -/
def additional_days : ℕ := 3

/-- Theorem stating that the initial number of workers is correct -/
theorem initial_workers_correct : 
  initial_workers * initial_days = 
  initial_workers * days_before_join + 
  (initial_workers + new_workers) * additional_days := by
  sorry

#check initial_workers_correct

end initial_workers_correct_l3631_363186


namespace perfect_square_trinomial_m_value_l3631_363183

/-- A perfect square trinomial in the form ax^2 + bx + c -/
structure PerfectSquareTrinomial (a b c : ℝ) : Prop where
  is_perfect_square : ∃ (p q : ℝ), a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, PerfectSquareTrinomial 1 (-m) 25 → m = 10 ∨ m = -10 := by
  sorry

end perfect_square_trinomial_m_value_l3631_363183


namespace sum_of_divisors_of_11_squared_l3631_363114

theorem sum_of_divisors_of_11_squared (a b c : ℕ+) : 
  a * b * c = 11^2 →
  a ∣ 11^2 ∧ b ∣ 11^2 ∧ c ∣ 11^2 →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 23 := by
sorry

end sum_of_divisors_of_11_squared_l3631_363114


namespace consecutive_odd_sum_fourth_power_l3631_363173

theorem consecutive_odd_sum_fourth_power (a b c : ℕ) : 
  (∃ n : ℕ, n < 10 ∧ a + b + c = n^4) ∧ 
  (Odd a ∧ Odd b ∧ Odd c) ∧
  (b = a + 2 ∧ c = b + 2) →
  ((a, b, c) = (25, 27, 29) ∨ (a, b, c) = (2185, 2187, 2189)) :=
by sorry

end consecutive_odd_sum_fourth_power_l3631_363173


namespace parallelogram_area_l3631_363177

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area :
  let base : ℝ := 20
  let height : ℝ := 4
  let area : ℝ := base * height
  area = 80 := by sorry

end parallelogram_area_l3631_363177


namespace algebraic_arithmetic_equivalence_l3631_363192

theorem algebraic_arithmetic_equivalence (a b : ℕ) (h : a > b) :
  (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end algebraic_arithmetic_equivalence_l3631_363192


namespace cos_equation_solution_l3631_363101

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos (2 * x) - 2 * Real.cos (4 * x))^2 = 9 + (Real.cos (5 * x))^2 ↔ 
  ∃ k : ℤ, x = π / 2 + k * π :=
sorry

end cos_equation_solution_l3631_363101


namespace find_other_number_l3631_363102

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 4 * b = 161) 
  (h2 : a = 17 ∨ b = 17) : (a = 31 ∧ b = 17) ∨ (a = 17 ∧ b = 31) :=
sorry

end find_other_number_l3631_363102


namespace train_speed_l3631_363157

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 3000) (h2 : time = 120) :
  length / time * (3600 / 1000) = 90 := by
  sorry

end train_speed_l3631_363157


namespace digit_250_of_13_over_17_is_8_l3631_363180

/-- The 250th decimal digit of 13/17 -/
def digit_250_of_13_over_17 : ℕ :=
  let decimal_expansion := (13 : ℚ) / 17
  let period := 16
  let position_in_period := 250 % period
  8

/-- Theorem: The 250th decimal digit in the decimal representation of 13/17 is 8 -/
theorem digit_250_of_13_over_17_is_8 :
  digit_250_of_13_over_17 = 8 := by
  sorry

end digit_250_of_13_over_17_is_8_l3631_363180


namespace scarlett_oil_addition_l3631_363158

/-- The amount of oil Scarlett needs to add to her measuring cup -/
def oil_to_add (current : ℚ) (desired : ℚ) : ℚ :=
  desired - current

/-- Theorem: Given the current amount of oil and the desired amount, 
    prove that Scarlett needs to add 0.67 cup of oil -/
theorem scarlett_oil_addition (current : ℚ) (desired : ℚ)
  (h1 : current = 17/100)
  (h2 : desired = 84/100) :
  oil_to_add current desired = 67/100 := by
  sorry

end scarlett_oil_addition_l3631_363158


namespace smallest_non_square_product_of_four_primes_l3631_363109

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a number is prime --/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

/-- A function that checks if a number is the product of four primes --/
def is_product_of_four_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ n = p * q * r * s

theorem smallest_non_square_product_of_four_primes :
  (∀ m : ℕ, m < 24 → ¬(is_product_of_four_primes m ∧ ¬is_perfect_square m)) ∧
  (is_product_of_four_primes 24 ∧ ¬is_perfect_square 24) :=
sorry

end smallest_non_square_product_of_four_primes_l3631_363109


namespace factorization_equality_l3631_363131

theorem factorization_equality (a b : ℝ) : a^2 - 2*a*b = a*(a - 2*b) := by
  sorry

end factorization_equality_l3631_363131
