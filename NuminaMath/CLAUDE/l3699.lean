import Mathlib

namespace circle_circumference_bounds_l3699_369941

/-- The circumference of a circle with diameter 1 is between 3 and 4 -/
theorem circle_circumference_bounds :
  ∀ C : ℝ, C = π * 1 → 3 < C ∧ C < 4 := by
  sorry

end circle_circumference_bounds_l3699_369941


namespace sum_greater_than_two_l3699_369939

theorem sum_greater_than_two (x y : ℝ) 
  (h1 : x^7 > y^6) 
  (h2 : y^7 > x^6) : 
  x + y > 2 := by
sorry

end sum_greater_than_two_l3699_369939


namespace books_movies_difference_l3699_369925

/-- The number of books in the "crazy silly school" series -/
def num_books : ℕ := 36

/-- The number of movies in the "crazy silly school" series -/
def num_movies : ℕ := 25

/-- The number of books read -/
def books_read : ℕ := 17

/-- The number of movies watched -/
def movies_watched : ℕ := 13

/-- Theorem stating the difference between the number of books and movies -/
theorem books_movies_difference : num_books - num_movies = 11 := by
  sorry

end books_movies_difference_l3699_369925


namespace square_pentagon_angle_sum_l3699_369943

/-- In a figure composed of a square and a regular pentagon, the sum of angles a° and b° is 324°. -/
theorem square_pentagon_angle_sum (a b : ℝ) : 
  -- The figure is composed of a square and a regular pentagon
  -- a and b are angles in degrees as shown in the diagram
  a + b = 324 := by sorry

end square_pentagon_angle_sum_l3699_369943


namespace sweater_markup_l3699_369951

theorem sweater_markup (wholesale : ℝ) (retail : ℝ) (h1 : retail > 0) (h2 : wholesale > 0) :
  (retail * (1 - 0.6) = wholesale * 1.2) →
  ((retail - wholesale) / wholesale * 100 = 200) := by
sorry

end sweater_markup_l3699_369951


namespace expression_evaluation_l3699_369969

theorem expression_evaluation (m n : ℚ) (hm : m = -1/3) (hn : n = 1/2) :
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2) = -5/3 := by
  sorry

end expression_evaluation_l3699_369969


namespace binomial_square_value_l3699_369972

theorem binomial_square_value (a : ℚ) : 
  (∃ p q : ℚ, ∀ x, 9*x^2 + 27*x + a = (p*x + q)^2) → a = 81/4 := by
  sorry

end binomial_square_value_l3699_369972


namespace binomial_coefficient_n_choose_2_l3699_369920

theorem binomial_coefficient_n_choose_2 (n : ℕ) (h : n ≥ 2) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_n_choose_2_l3699_369920


namespace george_total_blocks_l3699_369963

/-- The number of boxes George has -/
def num_boxes : ℕ := 2

/-- The number of blocks in each box -/
def blocks_per_box : ℕ := 6

/-- Theorem stating the total number of blocks George has -/
theorem george_total_blocks : num_boxes * blocks_per_box = 12 := by
  sorry

end george_total_blocks_l3699_369963


namespace solution_set_min_value_l3699_369962

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2*x - 5

-- Statement for the solution set
theorem solution_set : 
  {x : ℝ | |f x| + |g x| ≤ 2} = {x : ℝ | 5/3 ≤ x ∧ x ≤ 3} := by sorry

-- Statement for the minimum value
theorem min_value : 
  ∀ x : ℝ, |f (2*x)| + |g x| ≥ 1 := by sorry

end solution_set_min_value_l3699_369962


namespace smallest_m_for_distinct_roots_smallest_integer_m_for_distinct_roots_l3699_369926

theorem smallest_m_for_distinct_roots (m : ℤ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) ↔ m ≥ 0 :=
by sorry

theorem smallest_integer_m_for_distinct_roots : 
  ∃ m₀ : ℤ, m₀ ≥ 0 ∧ ∀ m : ℤ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) → m ≥ m₀ :=
by sorry

end smallest_m_for_distinct_roots_smallest_integer_m_for_distinct_roots_l3699_369926


namespace line_equation_point_slope_l3699_369996

/-- The point-slope form of a line with given slope and point. -/
def point_slope_form (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- Theorem: The point-slope form of a line with slope 2 passing through (2, -3) is y + 3 = 2(x - 2). -/
theorem line_equation_point_slope : 
  let k : ℝ := 2
  let x₀ : ℝ := 2
  let y₀ : ℝ := -3
  ∀ x y : ℝ, point_slope_form k x₀ y₀ x y ↔ y + 3 = 2 * (x - 2) :=
sorry

end line_equation_point_slope_l3699_369996


namespace log_equation_solution_l3699_369916

-- Define the logarithm function with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_equation_solution :
  ∃! x : ℝ, x > 1 ∧ log_one_third (x^2 + 3*x - 4) = log_one_third (2*x + 2) :=
by sorry

end log_equation_solution_l3699_369916


namespace complex_cube_equation_l3699_369909

theorem complex_cube_equation :
  ∃! (z : ℂ), ∃ (x y c : ℤ), 
    x > 0 ∧ y > 0 ∧ 
    z = x + y * I ∧
    z^3 = -74 + c * I ∧
    z = 1 + 5 * I :=
by sorry

end complex_cube_equation_l3699_369909


namespace small_planters_needed_l3699_369993

/-- Represents the types of seeds --/
inductive SeedType
  | Basil
  | Cilantro
  | Parsley

/-- Represents the types of planters --/
inductive PlanterType
  | Large
  | Medium
  | Small

/-- Represents the planting requirements for each seed type --/
def plantingRequirement (s : SeedType) : Set PlanterType :=
  match s with
  | SeedType.Basil => {PlanterType.Large, PlanterType.Medium}
  | SeedType.Cilantro => {PlanterType.Medium}
  | SeedType.Parsley => {PlanterType.Large, PlanterType.Medium, PlanterType.Small}

/-- The capacity of each planter type --/
def planterCapacity (p : PlanterType) : ℕ :=
  match p with
  | PlanterType.Large => 20
  | PlanterType.Medium => 10
  | PlanterType.Small => 4

/-- The number of each planter type available --/
def planterCount (p : PlanterType) : ℕ :=
  match p with
  | PlanterType.Large => 4
  | PlanterType.Medium => 8
  | PlanterType.Small => 0  -- We're solving for this

/-- The number of seeds for each seed type --/
def seedCount (s : SeedType) : ℕ :=
  match s with
  | SeedType.Basil => 200
  | SeedType.Cilantro => 160
  | SeedType.Parsley => 120

theorem small_planters_needed : 
  ∃ (n : ℕ), 
    n * planterCapacity PlanterType.Small = 
      seedCount SeedType.Parsley + 
      (seedCount SeedType.Cilantro - planterCount PlanterType.Medium * planterCapacity PlanterType.Medium) + 
      (seedCount SeedType.Basil - 
        (planterCount PlanterType.Large * planterCapacity PlanterType.Large + 
         (planterCount PlanterType.Medium - 
          (seedCount SeedType.Cilantro / planterCapacity PlanterType.Medium)) * 
           planterCapacity PlanterType.Medium)) ∧ 
    n = 50 := by
  sorry

end small_planters_needed_l3699_369993


namespace units_digit_17_2011_l3699_369906

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the property that powers of 17 have the same units digit as powers of 7
axiom units_digit_17_7 (n : ℕ) : unitsDigit (17^n) = unitsDigit (7^n)

-- Define the cycle of units digits for powers of 7
def sevenPowerCycle : List ℕ := [7, 9, 3, 1]

-- Theorem stating that the units digit of 17^2011 is 3
theorem units_digit_17_2011 : unitsDigit (17^2011) = 3 := by
  sorry

end units_digit_17_2011_l3699_369906


namespace school_population_l3699_369931

theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t := by sorry

end school_population_l3699_369931


namespace sample_size_is_100_l3699_369910

/-- A structure representing a statistical sampling process -/
structure SamplingProcess where
  totalStudents : Nat
  selectedStudents : Nat

/-- Definition of sample size for a SamplingProcess -/
def sampleSize (sp : SamplingProcess) : Nat := sp.selectedStudents

/-- Theorem stating that for the given sampling process, the sample size is 100 -/
theorem sample_size_is_100 (sp : SamplingProcess) 
  (h1 : sp.totalStudents = 1000) 
  (h2 : sp.selectedStudents = 100) : 
  sampleSize sp = 100 := by
  sorry

#check sample_size_is_100

end sample_size_is_100_l3699_369910


namespace min_value_expression_l3699_369942

theorem min_value_expression (x : ℝ) (h : x > 0) : 6 * x + 1 / x^6 ≥ 7 ∧ (6 * x + 1 / x^6 = 7 ↔ x = 1) := by
  sorry

end min_value_expression_l3699_369942


namespace square_side_increase_l3699_369924

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.1025 → p = 5 := by
sorry

end square_side_increase_l3699_369924


namespace hexagon_area_proof_l3699_369997

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Calculates the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Checks if lines are parallel -/
def areParallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if y-coordinates are distinct elements of a set -/
def distinctYCoordinates (h : Hexagon) (s : Set ℝ) : Prop := sorry

theorem hexagon_area_proof (h : Hexagon) :
  h.A = ⟨0, 0⟩ →
  h.B = ⟨2 * Real.sqrt 3, 3⟩ →
  h.F = ⟨-7 / 2 * Real.sqrt 3, 5⟩ →
  angle h.F h.A h.B = 150 * π / 180 →
  areParallel h.A h.B h.D h.E →
  areParallel h.B h.C h.E h.F →
  areParallel h.C h.D h.F h.A →
  isEquilateral h →
  distinctYCoordinates h {0, 1, 3, 5, 7, 9} →
  hexagonArea h = 77 / 2 * Real.sqrt 3 := by
  sorry

end hexagon_area_proof_l3699_369997


namespace factorization_proof_l3699_369980

theorem factorization_proof (a : ℝ) :
  74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) := by
  sorry

end factorization_proof_l3699_369980


namespace integral_problems_l3699_369934

theorem integral_problems :
  (∃ k : ℝ, (∫ x in (0:ℝ)..2, (3*x^2 + k)) = 10 ∧ k = 1) ∧
  (∫ x in (-1:ℝ)..8, x^(1/3)) = 45/4 :=
by sorry

end integral_problems_l3699_369934


namespace sum_first_six_multiples_of_twelve_l3699_369987

theorem sum_first_six_multiples_of_twelve : 
  (Finset.range 6).sum (fun i => 12 * (i + 1)) = 252 := by
  sorry

end sum_first_six_multiples_of_twelve_l3699_369987


namespace product_correction_l3699_369947

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (p q : Nat) :
  p ≥ 10 ∧ p < 100 →  -- p is a two-digit number
  q > 0 →  -- q is positive
  reverseDigits p * q = 221 →
  p * q = 923 := by
sorry

end product_correction_l3699_369947


namespace smallest_number_with_given_remainders_l3699_369979

theorem smallest_number_with_given_remainders : ∃ (n : ℕ), n = 838 ∧ 
  (∃ (a : ℕ), 0 ≤ a ∧ a ≤ 19 ∧ 
    n % 20 = a ∧ 
    n % 21 = a + 1 ∧ 
    n % 22 = 2) ∧ 
  (∀ (m : ℕ), m < n → 
    ¬(∃ (b : ℕ), 0 ≤ b ∧ b ≤ 19 ∧ 
      m % 20 = b ∧ 
      m % 21 = b + 1 ∧ 
      m % 22 = 2)) :=
by sorry

end smallest_number_with_given_remainders_l3699_369979


namespace inequality_proof_l3699_369927

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 ≥ 2*(a + b - 1) ∧ 
  (a > 0 ∧ b > 0 ∧ a + b = 3 → 1/a + 4/(b+1) ≥ 9/4) := by
  sorry

end inequality_proof_l3699_369927


namespace total_amount_paid_prove_total_amount_l3699_369944

/-- Calculate the total amount paid for grapes and mangoes -/
theorem total_amount_paid 
  (grape_quantity : ℕ) (grape_price : ℕ) 
  (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price

/-- Prove that the total amount paid for the given quantities and prices is 1135 -/
theorem prove_total_amount : 
  total_amount_paid 8 80 9 55 = 1135 := by
  sorry

end total_amount_paid_prove_total_amount_l3699_369944


namespace max_sum_of_factors_l3699_369973

theorem max_sum_of_factors (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  A + B + C ≤ 42 := by
sorry

end max_sum_of_factors_l3699_369973


namespace spatial_relationships_l3699_369936

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem spatial_relationships 
  (m n : Line) (α β : Plane) : 
  (∀ (m n : Line) (β : Plane), 
    perpendicular m β → perpendicular n β → parallel_lines m n) ∧
  (∀ (m : Line) (α β : Plane), 
    perpendicular m α → perpendicular m β → parallel_planes α β) :=
by sorry

end spatial_relationships_l3699_369936


namespace correct_distribution_l3699_369912

/-- Represents the amount of coins each person receives -/
structure CoinDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  e : ℚ

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (dist : CoinDistribution) : Prop :=
  -- The total amount is 5 coins
  dist.a + dist.b + dist.c + dist.d + dist.e = 5 ∧
  -- The difference between each person is equal
  (dist.b - dist.a = dist.c - dist.b) ∧
  (dist.c - dist.b = dist.d - dist.c) ∧
  (dist.d - dist.c = dist.e - dist.d) ∧
  -- The total amount received by A and B equals that received by C, D, and E
  dist.a + dist.b = dist.c + dist.d + dist.e

/-- The theorem stating the correct distribution -/
theorem correct_distribution :
  ∃ (dist : CoinDistribution),
    isValidDistribution dist ∧
    dist.a = 2/3 ∧
    dist.b = 5/6 ∧
    dist.c = 1 ∧
    dist.d = 7/6 ∧
    dist.e = 4/3 :=
  sorry

end correct_distribution_l3699_369912


namespace probability_non_defective_pencils_l3699_369901

/-- The probability of selecting 3 non-defective pencils from a box of 9 pencils with 2 defective pencils -/
theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 9
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 :=
by sorry

end probability_non_defective_pencils_l3699_369901


namespace arithmetic_sequence_sum_property_l3699_369975

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h : ArithmeticSequence a) (h2 : a 2 + a 12 = 32) :
  a 3 + a 11 = 32 := by
  sorry

end arithmetic_sequence_sum_property_l3699_369975


namespace imaginary_sum_equals_i_l3699_369956

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum_equals_i :
  i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by
  sorry

end imaginary_sum_equals_i_l3699_369956


namespace pizza_cost_l3699_369955

/-- Proves that the cost of each pizza is $11 given the conditions of the problem -/
theorem pizza_cost (total_money : ℕ) (initial_bill : ℕ) (final_bill : ℕ) (num_pizzas : ℕ) :
  total_money = 42 →
  initial_bill = 30 →
  final_bill = 39 →
  num_pizzas = 3 →
  ∃ (pizza_cost : ℕ), 
    pizza_cost * num_pizzas = total_money - (final_bill - initial_bill) ∧
    pizza_cost = 11 :=
by
  sorry

#check pizza_cost

end pizza_cost_l3699_369955


namespace cubic_root_sum_simplification_l3699_369991

theorem cubic_root_sum_simplification :
  (((9 : ℝ) / 16 + 25 / 36 + 4 / 9) ^ (1/3 : ℝ)) = (245 : ℝ) ^ (1/3) / 12 := by
  sorry

end cubic_root_sum_simplification_l3699_369991


namespace intersection_of_A_and_B_l3699_369957

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_A_and_B_l3699_369957


namespace eleven_pow_2023_mod_50_l3699_369919

theorem eleven_pow_2023_mod_50 : 11^2023 % 50 = 31 := by
  sorry

end eleven_pow_2023_mod_50_l3699_369919


namespace profit_percentage_calculation_l3699_369905

/-- Profit percentage calculation for Company N --/
theorem profit_percentage_calculation (R : ℝ) (P : ℝ) :
  R > 0 ∧ P > 0 →  -- Assuming positive revenue and profit
  (0.8 * R) * 0.14 = 0.112 * R →  -- 1999 profit calculation
  0.112 * R = 1.1200000000000001 * P →  -- Profit comparison between years
  P / R * 100 = 10 := by
sorry


end profit_percentage_calculation_l3699_369905


namespace largest_non_sum_of_composites_l3699_369933

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l3699_369933


namespace inverse_of_B_cubed_l3699_369937

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, -2; 1, 1] →
  (B^3)⁻¹ = !![13, -22; 11, -9] := by
  sorry

end inverse_of_B_cubed_l3699_369937


namespace xyz_value_l3699_369948

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 8/3) :
  x * y * z = (17 + Real.sqrt 285) / 2 := by
  sorry

end xyz_value_l3699_369948


namespace bird_migration_difference_l3699_369976

/-- The number of bird families that flew to Asia is greater than the number
    of bird families that flew to Africa by 47. -/
theorem bird_migration_difference :
  let mountain_families : ℕ := 38
  let africa_families : ℕ := 47
  let asia_families : ℕ := 94
  asia_families - africa_families = 47 := by sorry

end bird_migration_difference_l3699_369976


namespace sequence_exists_l3699_369930

theorem sequence_exists : ∃ (seq : Fin 2000 → ℝ), 
  (∀ i : Fin 1998, seq i + seq (i + 1) + seq (i + 2) < 0) ∧ 
  (Finset.sum Finset.univ seq > 0) := by
  sorry

end sequence_exists_l3699_369930


namespace john_hard_hat_ratio_l3699_369945

/-- Proves that the ratio of green to pink hard hats John took away is 2:1 -/
theorem john_hard_hat_ratio :
  let initial_pink : ℕ := 26
  let initial_green : ℕ := 15
  let initial_yellow : ℕ := 24
  let carl_pink : ℕ := 4
  let john_pink : ℕ := 6
  let remaining_total : ℕ := 43
  let initial_total : ℕ := initial_pink + initial_green + initial_yellow
  let john_green : ℕ := initial_total - carl_pink - john_pink - remaining_total
  (john_green : ℚ) / (john_pink : ℚ) = 2 / 1 :=
by sorry

end john_hard_hat_ratio_l3699_369945


namespace uncovered_area_is_64_l3699_369949

/-- Represents the dimensions of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the dimensions of a square carpet -/
structure Carpet :=
  (side : ℝ)

/-- Calculates the area of a rectangular floor -/
def floorArea (f : Floor) : ℝ :=
  f.length * f.width

/-- Calculates the area of a square carpet -/
def carpetArea (c : Carpet) : ℝ :=
  c.side * c.side

/-- Calculates the uncovered area when placing a carpet on a floor -/
def uncoveredArea (f : Floor) (c : Carpet) : ℝ :=
  floorArea f - carpetArea c

theorem uncovered_area_is_64 (f : Floor) (c : Carpet) 
    (h1 : f.length = 10)
    (h2 : f.width = 8)
    (h3 : c.side = 4) :
  uncoveredArea f c = 64 := by
  sorry

end uncovered_area_is_64_l3699_369949


namespace mixture_alcohol_percentage_l3699_369971

/-- Represents the alcohol content and volume of a solution -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the amount of pure alcohol in a solution -/
def alcoholContent (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

theorem mixture_alcohol_percentage 
  (x : Solution) 
  (y : Solution) 
  (h1 : x.volume = 250)
  (h2 : x.alcoholPercentage = 0.1)
  (h3 : y.alcoholPercentage = 0.3)
  (h4 : y.volume = 750) :
  let mixedSolution : Solution := ⟨x.volume + y.volume, (alcoholContent x + alcoholContent y) / (x.volume + y.volume)⟩
  mixedSolution.alcoholPercentage = 0.25 := by
sorry

end mixture_alcohol_percentage_l3699_369971


namespace correct_distribution_l3699_369918

/-- Represents the distribution of chestnuts among three girls -/
structure ChestnutDistribution where
  alya : ℕ
  valya : ℕ
  galya : ℕ

/-- Checks if the given distribution satisfies the problem conditions -/
def isValidDistribution (d : ChestnutDistribution) : Prop :=
  d.alya + d.valya + d.galya = 70 ∧
  4 * d.valya = 3 * d.alya ∧
  7 * d.alya = 6 * d.galya

/-- Theorem stating that the given distribution is correct -/
theorem correct_distribution :
  let d : ChestnutDistribution := ⟨24, 18, 28⟩
  isValidDistribution d := by
  sorry


end correct_distribution_l3699_369918


namespace smallest_number_of_groups_l3699_369995

theorem smallest_number_of_groups (total_campers : ℕ) (max_group_size : ℕ) : 
  total_campers = 36 → max_group_size = 12 → 
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) ∧
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) → num_groups = 3 :=
by sorry


end smallest_number_of_groups_l3699_369995


namespace greatest_common_divisor_780_180_240_l3699_369935

theorem greatest_common_divisor_780_180_240 :
  (∃ (d : ℕ), d ∣ 780 ∧ d ∣ 180 ∧ d ∣ 240 ∧ d < 100 ∧
    ∀ (x : ℕ), x ∣ 780 ∧ x ∣ 180 ∧ x ∣ 240 ∧ x < 100 → x ≤ d) ∧
  (60 ∣ 780 ∧ 60 ∣ 180 ∧ 60 ∣ 240 ∧ 60 < 100) :=
by sorry

end greatest_common_divisor_780_180_240_l3699_369935


namespace cat_average_weight_l3699_369953

theorem cat_average_weight : 
  let num_cats : ℕ := 4
  let weight_cat1 : ℝ := 12
  let weight_cat2 : ℝ := 12
  let weight_cat3 : ℝ := 14.7
  let weight_cat4 : ℝ := 9.3
  let total_weight : ℝ := weight_cat1 + weight_cat2 + weight_cat3 + weight_cat4
  let average_weight : ℝ := total_weight / num_cats
  average_weight = 12 := by
    sorry

end cat_average_weight_l3699_369953


namespace rectangle_max_area_l3699_369983

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) →  -- perimeter is 40 units
  (l * w ≤ 100) -- area is at most 100 square units
:= by sorry

end rectangle_max_area_l3699_369983


namespace frog_corner_probability_l3699_369903

/-- Represents a position on the 4x4 grid -/
inductive Position
| Corner
| Edge
| Middle

/-- Represents the state of the frog's movement -/
structure FrogState where
  position : Position
  hops : Nat

/-- Transition function for the frog's movement -/
def transition (state : FrogState) : FrogState :=
  sorry

/-- Probability of reaching a corner from a given state -/
def cornerProbability (state : FrogState) : Rat :=
  sorry

/-- The starting state of the frog -/
def initialState : FrogState :=
  { position := Position.Edge, hops := 0 }

/-- Main theorem: Probability of reaching a corner within 4 hops -/
theorem frog_corner_probability :
  cornerProbability { position := initialState.position, hops := 4 } = 35 / 64 := by
  sorry

end frog_corner_probability_l3699_369903


namespace probability_of_white_ball_l3699_369999

def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 10
def balls_removed : ℕ := 2

theorem probability_of_white_ball :
  let total_balls := initial_white_balls + initial_black_balls
  let remaining_balls := total_balls - balls_removed
  ∃ (p : ℚ), p = 37/98 ∧ 
    (∀ (w b : ℕ), w + b = remaining_balls → 
      (w : ℚ) / (w + b : ℚ) ≤ p) ∧
    (∃ (w b : ℕ), w + b = remaining_balls ∧ 
      (w : ℚ) / (w + b : ℚ) = p) :=
by sorry


end probability_of_white_ball_l3699_369999


namespace dans_age_l3699_369900

theorem dans_age (dan_age ben_age : ℕ) : 
  ben_age = dan_age - 3 →
  ben_age + dan_age = 53 →
  dan_age = 28 := by
sorry

end dans_age_l3699_369900


namespace max_intersection_points_circle_rectangle_l3699_369913

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rectangle in a plane --/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment --/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The number of intersection points between a circle and a rectangle --/
def intersectionPointsCircleRectangle (c : Circle) (r : Rectangle) : ℕ :=
  (intersectionPointsCircleLine c (r.corners 0) (r.corners 1)) +
  (intersectionPointsCircleLine c (r.corners 1) (r.corners 2)) +
  (intersectionPointsCircleLine c (r.corners 2) (r.corners 3)) +
  (intersectionPointsCircleLine c (r.corners 3) (r.corners 0))

/-- Theorem: The maximum number of intersection points between a circle and a rectangle is 8 --/
theorem max_intersection_points_circle_rectangle :
  ∀ c : Circle, ∀ r : Rectangle, intersectionPointsCircleRectangle c r ≤ 8 ∧
  ∃ c : Circle, ∃ r : Rectangle, intersectionPointsCircleRectangle c r = 8 :=
sorry

end max_intersection_points_circle_rectangle_l3699_369913


namespace quadratic_one_solution_l3699_369914

theorem quadratic_one_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 30 * x + 12 = 0) :
  ∃ x, a * x^2 + 30 * x + 12 = 0 ∧ x = -4/5 := by
  sorry

end quadratic_one_solution_l3699_369914


namespace valid_arrangements_count_l3699_369998

/-- The number of ways to arrange plates around a circular table. -/
def arrange_plates (blue red green orange : ℕ) : ℕ :=
  sorry

/-- The number of valid arrangements of plates. -/
def valid_arrangements : ℕ :=
  arrange_plates 5 3 2 1

/-- Theorem stating the correct number of valid arrangements. -/
theorem valid_arrangements_count : valid_arrangements = 361 := by
  sorry

end valid_arrangements_count_l3699_369998


namespace binomial_coefficient_sum_l3699_369964

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (x + 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  (a₁ + 3*a₃ + 5*a₅ + 7*a₇ + 9*a₉)^2 - (2*a₂ + 4*a₄ + 6*a₆ + 8*a₈)^2 = 3^12 := by
  sorry

end binomial_coefficient_sum_l3699_369964


namespace third_task_end_time_l3699_369952

-- Define the start time of the first task
def start_time : Nat := 13 * 60  -- 1:00 PM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 14 * 60 + 40  -- 2:40 PM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 3

-- Theorem statement
theorem third_task_end_time :
  let task_duration := (end_second_task - start_time) / 2
  let end_third_task := end_second_task + task_duration
  end_third_task = 15 * 60 + 30  -- 3:30 PM in minutes since midnight
  := by sorry

end third_task_end_time_l3699_369952


namespace jack_marbles_remaining_l3699_369968

/-- Given Jack starts with 62 marbles and shares 33 marbles, prove that he ends up with 29 marbles. -/
theorem jack_marbles_remaining (initial_marbles : ℕ) (shared_marbles : ℕ) 
  (h1 : initial_marbles = 62)
  (h2 : shared_marbles = 33) :
  initial_marbles - shared_marbles = 29 := by
  sorry

end jack_marbles_remaining_l3699_369968


namespace lcm_problem_l3699_369904

theorem lcm_problem (m n : ℕ+) :
  m - n = 189 →
  Nat.lcm m n = 133866 →
  m = 22311 ∧ n = 22122 := by
  sorry

end lcm_problem_l3699_369904


namespace modular_inverse_three_mod_187_l3699_369907

theorem modular_inverse_three_mod_187 :
  ∃ x : ℕ, x < 187 ∧ (3 * x) % 187 = 1 :=
by
  use 125
  sorry

end modular_inverse_three_mod_187_l3699_369907


namespace four_digit_perfect_square_palindrome_l3699_369946

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem four_digit_perfect_square_palindrome :
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n := by sorry

end four_digit_perfect_square_palindrome_l3699_369946


namespace work_completion_l3699_369965

/-- Given that 36 men can complete a piece of work in 25 hours,
    prove that 10 men can complete the same work in 90 hours. -/
theorem work_completion (work : ℝ) : 
  work = 36 * 25 → work = 10 * 90 := by
  sorry

end work_completion_l3699_369965


namespace min_abs_z_l3699_369938

open Complex

theorem min_abs_z (z : ℂ) (h : abs (z - 1) + abs (z - (3 + 2*I)) = 2 * Real.sqrt 2) :
  ∃ (w : ℂ), abs w ≤ abs z ∧ abs w = 1 :=
by sorry

end min_abs_z_l3699_369938


namespace siblings_selection_probability_l3699_369932

/-- Given the probabilities of selection for three siblings, prove that the probability of all three being selected is 3/28 -/
theorem siblings_selection_probability
  (p_ram : ℚ) (p_ravi : ℚ) (p_rani : ℚ)
  (h_ram : p_ram = 5 / 7)
  (h_ravi : p_ravi = 1 / 5)
  (h_rani : p_rani = 3 / 4) :
  p_ram * p_ravi * p_rani = 3 / 28 := by
  sorry

end siblings_selection_probability_l3699_369932


namespace polynomial_evaluation_l3699_369923

theorem polynomial_evaluation :
  ∀ x : ℝ, 
    x > 0 → 
    x^2 - 3*x - 10 = 0 → 
    x^3 - 3*x^2 - 9*x + 5 = 5 :=
by
  sorry

end polynomial_evaluation_l3699_369923


namespace wolf_tail_growth_l3699_369958

theorem wolf_tail_growth (x y : ℕ) : 1 * 2^x * 3^y = 864 ↔ x = 5 ∧ y = 3 := by
  sorry

end wolf_tail_growth_l3699_369958


namespace exercise_time_is_9_25_hours_l3699_369970

/-- Represents the exercise schedule for a week -/
structure ExerciseSchedule where
  initial_jogging : ℕ
  jogging_increment : ℕ
  swimming_increment : ℕ
  wednesday_reduction : ℕ
  friday_kickboxing : ℕ
  kickboxing_multiplier : ℕ

/-- Calculates the total exercise time for the week -/
def total_exercise_time (schedule : ExerciseSchedule) : ℚ :=
  sorry

/-- Theorem stating that the total exercise time is 9.25 hours -/
theorem exercise_time_is_9_25_hours (schedule : ExerciseSchedule) 
  (h1 : schedule.initial_jogging = 30)
  (h2 : schedule.jogging_increment = 5)
  (h3 : schedule.swimming_increment = 10)
  (h4 : schedule.wednesday_reduction = 10)
  (h5 : schedule.friday_kickboxing = 20)
  (h6 : schedule.kickboxing_multiplier = 2) :
  total_exercise_time schedule = 9.25 := by
  sorry

end exercise_time_is_9_25_hours_l3699_369970


namespace distribute_six_balls_three_boxes_l3699_369917

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 729 -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end distribute_six_balls_three_boxes_l3699_369917


namespace probability_ascending_rolls_eq_five_fiftyfour_l3699_369902

/-- A standard die has faces labeled from 1 to 6 -/
def standardDie : Finset Nat := Finset.range 6

/-- The number of times the die is rolled -/
def numRolls : Nat := 3

/-- The probability of rolling three dice and getting three distinct numbers in ascending order -/
def probabilityAscendingRolls : Rat :=
  (Nat.choose 6 3 : Rat) / (6 ^ numRolls)

theorem probability_ascending_rolls_eq_five_fiftyfour : 
  probabilityAscendingRolls = 5 / 54 := by
  sorry

end probability_ascending_rolls_eq_five_fiftyfour_l3699_369902


namespace elementary_school_coats_l3699_369982

theorem elementary_school_coats 
  (total_coats : ℕ) 
  (high_school_coats : ℕ) 
  (middle_school_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : high_school_coats = 6922)
  (h3 : middle_school_coats = 1825) :
  total_coats - (high_school_coats + middle_school_coats) = 690 := by
  sorry

end elementary_school_coats_l3699_369982


namespace min_value_theorem_l3699_369929

theorem min_value_theorem (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 36*c^2 = 4) :
  ∃ (m : ℝ), m = -2 * Real.sqrt 14 ∧ ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 36*z^2 = 4 → 3*x + 6*y + 12*z ≥ m :=
by sorry

end min_value_theorem_l3699_369929


namespace retailer_profit_percentage_l3699_369928

/-- Calculates the percentage profit of a retailer given the wholesale price, retail price, and discount percentage. -/
theorem retailer_profit_percentage 
  (wholesale_price retail_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : wholesale_price = 90) 
  (h2 : retail_price = 120) 
  (h3 : discount_percentage = 0.1) :
  let selling_price := retail_price * (1 - discount_percentage)
  let profit := selling_price - wholesale_price
  let profit_percentage := (profit / wholesale_price) * 100
  profit_percentage = 20 := by
  sorry

end retailer_profit_percentage_l3699_369928


namespace karen_tom_race_l3699_369940

/-- Karen's race against Tom -/
theorem karen_tom_race (karen_speed : ℝ) (karen_delay : ℝ) (lead_distance : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 4 / 60 →
  lead_distance = 4 →
  tom_distance = 24 →
  ∃ (tom_speed : ℝ), tom_speed = 45 ∧ 
    karen_speed * (tom_distance / karen_speed + lead_distance / karen_speed) = 
    tom_speed * (tom_distance / karen_speed + lead_distance / karen_speed + karen_delay) :=
by sorry

end karen_tom_race_l3699_369940


namespace bead_mixing_problem_l3699_369985

/-- Proves that the total number of boxes is 8 given the conditions of the bead mixing problem. -/
theorem bead_mixing_problem (red_cost yellow_cost mixed_cost : ℚ) 
  (boxes_per_color : ℕ) : 
  red_cost = 13/10 ∧ 
  yellow_cost = 2 ∧ 
  mixed_cost = 43/25 ∧ 
  boxes_per_color = 4 → 
  (red_cost * boxes_per_color + yellow_cost * boxes_per_color) / 
    (2 * boxes_per_color) = mixed_cost ∧
  2 * boxes_per_color = 8 := by
  sorry

end bead_mixing_problem_l3699_369985


namespace total_insect_legs_l3699_369915

/-- The number of insects in the laboratory -/
def num_insects : ℕ := 6

/-- The number of legs each insect has -/
def legs_per_insect : ℕ := 6

/-- The total number of legs for all insects in the laboratory -/
def total_legs : ℕ := num_insects * legs_per_insect

/-- Theorem stating that the total number of legs is 36 -/
theorem total_insect_legs : total_legs = 36 := by
  sorry

end total_insect_legs_l3699_369915


namespace derivative_f_at_3pi_4_l3699_369994

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x + 1

theorem derivative_f_at_3pi_4 :
  deriv f (3 * Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

end derivative_f_at_3pi_4_l3699_369994


namespace line_point_distance_l3699_369921

/-- Given a line x = 3y + 5 passing through points (m, n) and (m + d, n + p),
    where p = 0.6666666666666666, prove that d = 2 -/
theorem line_point_distance (m n d p : ℝ) : 
  p = 0.6666666666666666 →
  m = 3 * n + 5 →
  (m + d) = 3 * (n + p) + 5 →
  d = 2 := by
sorry

end line_point_distance_l3699_369921


namespace function_symmetry_l3699_369974

-- Define the function f
variable (f : ℝ → ℝ)
-- Define the constant a
variable (a : ℝ)

-- State the theorem
theorem function_symmetry 
  (h : ∀ x : ℝ, f (a - x) = -f (a + x)) : 
  ∀ x : ℝ, f (2 * a - x) = -f x := by
  sorry

end function_symmetry_l3699_369974


namespace conical_hopper_volume_l3699_369986

/-- The volume of a conical hopper with given dimensions -/
theorem conical_hopper_volume :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let height : ℝ := 0.6 * radius
  let volume : ℝ := (1 / 3) * Real.pi * radius^2 * height
  volume = 25 * Real.pi := by sorry

end conical_hopper_volume_l3699_369986


namespace lowest_price_option2_l3699_369966

def initial_amount : ℝ := 12000

def option1_price : ℝ := initial_amount * (1 - 0.15) * (1 - 0.10) * (1 - 0.05)

def option2_price : ℝ := initial_amount * (1 - 0.25) * (1 - 0.05)

def option3_price : ℝ := initial_amount * (1 - 0.20) - 500

theorem lowest_price_option2 :
  option2_price < option1_price ∧ option2_price < option3_price :=
by sorry

end lowest_price_option2_l3699_369966


namespace andy_ate_six_cookies_six_is_max_cookies_for_andy_l3699_369959

/-- Represents the number of cookies eaten by Andy -/
def andys_cookies : ℕ := sorry

/-- The total number of cookies baked -/
def total_cookies : ℕ := 36

/-- Theorem stating that Andy ate 6 cookies, given the problem conditions -/
theorem andy_ate_six_cookies : andys_cookies = 6 := by
  have h1 : andys_cookies + 2 * andys_cookies + 3 * andys_cookies = total_cookies := sorry
  have h2 : andys_cookies ≤ 6 := sorry
  sorry

/-- Theorem proving that 6 is the maximum number of cookies Andy could have eaten -/
theorem six_is_max_cookies_for_andy :
  ∀ n : ℕ, n > 6 → n + 2 * n + 3 * n > total_cookies := by
  sorry

end andy_ate_six_cookies_six_is_max_cookies_for_andy_l3699_369959


namespace at_least_five_roots_l3699_369981

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the period T
variable (T : ℝ)

-- Assumptions
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x + T) = f x)
variable (h_T_pos : T > 0)

-- Theorem statement
theorem at_least_five_roots :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
     x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
     x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
     x₄ ≠ x₅) ∧
    (x₁ ∈ Set.Icc (-T) T ∧
     x₂ ∈ Set.Icc (-T) T ∧
     x₃ ∈ Set.Icc (-T) T ∧
     x₄ ∈ Set.Icc (-T) T ∧
     x₅ ∈ Set.Icc (-T) T) ∧
    (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :=
by sorry

end at_least_five_roots_l3699_369981


namespace min_sum_of_primes_l3699_369988

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem min_sum_of_primes (a b c d n : ℕ) : 
  (∃ k : ℕ, a * 1000 + b * 100 + c * 10 + d = 3 * 3 * 11 * (n + 49)) →
  is_prime a → is_prime b → is_prime c → is_prime d →
  (∀ a' b' c' d' n' : ℕ, 
    (∃ k' : ℕ, a' * 1000 + b' * 100 + c' * 10 + d' = 3 * 3 * 11 * (n' + 49)) →
    is_prime a' → is_prime b' → is_prime c' → is_prime d' →
    a + b + c + d ≤ a' + b' + c' + d') →
  a + b + c + d = 70 := 
sorry

end min_sum_of_primes_l3699_369988


namespace french_students_count_l3699_369961

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 69)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 15) :
  ∃ french : ℕ, french = total - german + both - neither :=
by
  sorry

end french_students_count_l3699_369961


namespace work_together_time_l3699_369990

/-- 
Calculates the time taken to complete a job when two people work together, 
given their individual completion times.
-/
def time_together (time_david time_john : ℚ) : ℚ :=
  1 / (1 / time_david + 1 / time_john)

/-- 
Theorem: If David completes a job in 5 days and John completes the same job in 9 days,
then the time taken to complete the job when they work together is 45/14 days.
-/
theorem work_together_time : time_together 5 9 = 45 / 14 := by
  sorry

end work_together_time_l3699_369990


namespace arith_geom_seq_sum_30_l3699_369978

/-- An arithmetic-geometric sequence with its partial sums -/
structure ArithGeomSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums of the sequence
  is_arith_geom : ∀ n, (S (n + 10) - S n) / (S (n + 20) - S (n + 10)) = (S (n + 20) - S (n + 10)) / (S (n + 30) - S (n + 20))

/-- Theorem: For an arithmetic-geometric sequence, if S_10 = 10 and S_20 = 30, then S_30 = 70 -/
theorem arith_geom_seq_sum_30 (seq : ArithGeomSeq) (h1 : seq.S 10 = 10) (h2 : seq.S 20 = 30) : 
  seq.S 30 = 70 := by
  sorry

end arith_geom_seq_sum_30_l3699_369978


namespace planted_field_fraction_l3699_369960

theorem planted_field_fraction (a b h x : ℝ) (ha : a = 5) (hb : b = 12) (hh : h = 3) :
  let c := (a^2 + b^2).sqrt
  let s := x^2
  let triangle_area := a * b / 2
  h = (2 * triangle_area) / c - (b * x) / c →
  (triangle_area - s) / triangle_area = 431 / 480 :=
by sorry

end planted_field_fraction_l3699_369960


namespace union_of_sets_l3699_369908

theorem union_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-1, 1}
  A ∪ B = {-1, 1, 2, 3} := by
sorry

end union_of_sets_l3699_369908


namespace laborer_income_l3699_369911

theorem laborer_income (
  avg_expenditure_6months : ℝ)
  (fell_into_debt : Prop)
  (reduced_expenses_4months : ℝ)
  (debt_cleared_and_saved : ℝ) :
  avg_expenditure_6months = 85 →
  fell_into_debt →
  reduced_expenses_4months = 60 →
  debt_cleared_and_saved = 30 →
  ∃ (monthly_income : ℝ), monthly_income = 78 :=
by sorry

end laborer_income_l3699_369911


namespace hyperbola_equation_from_conditions_l3699_369950

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- The asymptotic lines of a hyperbola -/
def asymptotic_lines (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ y = -(3/4) * x

/-- The focus of a hyperbola -/
def focus (h : Hyperbola) : ℝ × ℝ := (0, 5)

/-- The main theorem -/
theorem hyperbola_equation_from_conditions (h : Hyperbola) :
  (∀ x y, asymptotic_lines h x y ↔ y = (3/4) * x ∨ y = -(3/4) * x) →
  focus h = (0, 5) →
  ∀ x y, hyperbola_equation h x y ↔ y^2 / 9 - x^2 / 16 = 1 :=
sorry

end hyperbola_equation_from_conditions_l3699_369950


namespace cosine_equality_l3699_369984

theorem cosine_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (745 * π / 180) →
  n = 25 ∨ n = -25 := by
sorry

end cosine_equality_l3699_369984


namespace largest_perfect_square_factor_3402_l3699_369992

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_3402 :
  largest_perfect_square_factor 3402 = 9 := by sorry

end largest_perfect_square_factor_3402_l3699_369992


namespace hyperbola_vertices_distance_l3699_369954

/-- The distance between the vertices of the hyperbola (x^2 / 121) - (y^2 / 49) = 1 is 22 -/
theorem hyperbola_vertices_distance : 
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ (x^2 / 121) - (y^2 / 49) - 1
  let vertices := {p : ℝ × ℝ | f p = 0 ∧ p.2 = 0}
  let distance := fun (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v1 ≠ v2 ∧ distance v1 v2 = 22 :=
by
  sorry

end hyperbola_vertices_distance_l3699_369954


namespace function_existence_l3699_369922

theorem function_existence : ∃ (f : ℤ → ℤ), ∀ (k : ℕ) (m : ℤ), k ≤ 1996 → ∃ (x : ℤ), f x + k * x = m := by
  sorry

end function_existence_l3699_369922


namespace employee_count_l3699_369989

theorem employee_count (initial_avg : ℝ) (new_avg : ℝ) (manager_salary : ℝ) : 
  initial_avg = 1500 →
  new_avg = 1900 →
  manager_salary = 11500 →
  ∃ n : ℕ, (n : ℝ) * initial_avg + manager_salary = new_avg * ((n : ℝ) + 1) ∧ n = 24 := by
sorry

end employee_count_l3699_369989


namespace root_sum_theorem_l3699_369977

theorem root_sum_theorem (a b c : ℝ) : 
  (a * b * c = -22) → 
  (a + b + c = 20) → 
  (a * b + b * c + c * a = 0) → 
  (b * c / a^2 + a * c / b^2 + a * b / c^2 = 3) := by
sorry

end root_sum_theorem_l3699_369977


namespace birds_to_africa_l3699_369967

/-- The number of bird families that flew away to Africa -/
def families_to_africa : ℕ := 118 - 80

/-- The number of bird families that flew away to Asia -/
def families_to_asia : ℕ := 80

/-- The total number of bird families that flew away for the winter -/
def total_families_away : ℕ := 118

/-- The number of bird families living near the mountain (not used in the proof) -/
def families_near_mountain : ℕ := 18

theorem birds_to_africa :
  families_to_africa = 38 ∧
  families_to_africa + families_to_asia = total_families_away :=
sorry

end birds_to_africa_l3699_369967
