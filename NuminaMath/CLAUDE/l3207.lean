import Mathlib

namespace abc_values_l3207_320726

theorem abc_values (A B C : ℝ) 
  (sum_eq : A + B = 10)
  (relation : 2 * A = 3 * B + 5)
  (product : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end abc_values_l3207_320726


namespace squirrels_and_nuts_l3207_320714

theorem squirrels_and_nuts (num_squirrels num_nuts : ℕ) 
  (h1 : num_squirrels = 4) 
  (h2 : num_nuts = 2) : 
  num_squirrels - num_nuts = 2 := by
  sorry

end squirrels_and_nuts_l3207_320714


namespace fraction_equality_l3207_320730

theorem fraction_equality (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end fraction_equality_l3207_320730


namespace correct_control_group_setup_l3207_320722

/-- Represents the different media types used in the experiment -/
inductive Medium
| BeefExtractPeptone
| SelectiveUreaDecomposing

/-- Represents the different inoculation methods -/
inductive InoculationMethod
| SoilSample
| SterileWater
| NoInoculation

/-- Represents a control group setup -/
structure ControlGroup :=
  (medium : Medium)
  (inoculation : InoculationMethod)

/-- The correct control group setup for the experiment -/
def correctControlGroup : ControlGroup :=
  { medium := Medium.BeefExtractPeptone,
    inoculation := InoculationMethod.SoilSample }

/-- The experiment setup -/
structure Experiment :=
  (name : String)
  (goal : String)
  (controlGroup : ControlGroup)

/-- Theorem stating that the correct control group is the one that inoculates
    the same soil sample liquid on beef extract peptone medium -/
theorem correct_control_group_setup
  (exp : Experiment)
  (h1 : exp.name = "Separating Bacteria that Decompose Urea in Soil")
  (h2 : exp.goal = "judge whether the separation effect has been achieved")
  : exp.controlGroup = correctControlGroup := by
  sorry

end correct_control_group_setup_l3207_320722


namespace incorrect_value_at_three_l3207_320754

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- Calculates the y-value for a given x-value using the linear function -/
def LinearFunction.eval (f : LinearFunction) (x : ℝ) : ℝ :=
  f.k * x + f.b

/-- Theorem: The value -2 for x = 3 is incorrect for the linear function passing through (-1, 3) and (0, 2) -/
theorem incorrect_value_at_three (f : LinearFunction) 
  (h1 : f.eval (-1) = 3)
  (h2 : f.eval 0 = 2) : 
  f.eval 3 ≠ -2 := by
  sorry

end incorrect_value_at_three_l3207_320754


namespace inequality_with_product_condition_l3207_320700

theorem inequality_with_product_condition (x y z : ℝ) (h : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) := by
  sorry

end inequality_with_product_condition_l3207_320700


namespace infinitely_many_k_with_Q_3k_geq_Q_3k1_l3207_320739

-- Define Q(n) as the sum of the decimal digits of n
def Q (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem infinitely_many_k_with_Q_3k_geq_Q_3k1 :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ Q (3^k) ≥ Q (3^(k+1)) := by
  sorry

end infinitely_many_k_with_Q_3k_geq_Q_3k1_l3207_320739


namespace no_k_exists_product_odd_primes_minus_one_is_power_l3207_320705

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k odd prime numbers -/
def productFirstKOddPrimes (k : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number k such that the product of the first k odd prime numbers minus 1 is an exact power of a natural number greater than one -/
theorem no_k_exists_product_odd_primes_minus_one_is_power :
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstKOddPrimes k = a^n + 1 := by
  sorry

end no_k_exists_product_odd_primes_minus_one_is_power_l3207_320705


namespace intersection_points_range_l3207_320728

theorem intersection_points_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    y₁ = Real.sqrt (4 - x₁^2) ∧
    y₂ = Real.sqrt (4 - x₂^2) ∧
    k * x₁ - y₁ - 2 * k + 4 = 0 ∧
    k * x₂ - y₂ - 2 * k + 4 = 0) ↔
  (3/4 < k ∧ k ≤ 1) :=
by sorry

end intersection_points_range_l3207_320728


namespace toms_total_coins_l3207_320776

/-- Represents the number of coins Tom has -/
structure TomCoins where
  quarters : ℕ
  nickels : ℕ

/-- The total number of coins Tom has -/
def total_coins (c : TomCoins) : ℕ :=
  c.quarters + c.nickels

/-- Tom's actual coin count -/
def toms_coins : TomCoins :=
  { quarters := 4, nickels := 8 }

theorem toms_total_coins :
  total_coins toms_coins = 12 := by
  sorry

end toms_total_coins_l3207_320776


namespace lcm_of_8_9_10_21_l3207_320775

theorem lcm_of_8_9_10_21 : Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 21)) = 2520 := by
  sorry

end lcm_of_8_9_10_21_l3207_320775


namespace shifted_sine_symmetry_l3207_320745

open Real

theorem shifted_sine_symmetry (φ : Real) (h1 : 0 < φ) (h2 : φ < π) :
  let f : Real → Real := λ x ↦ sin (3 * x + φ)
  let g : Real → Real := λ x ↦ f (x - π / 12)
  (∀ x, g x = g (-x)) → φ = 3 * π / 4 := by
  sorry

end shifted_sine_symmetry_l3207_320745


namespace remainder_problem_l3207_320704

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 38 = 32 := by
  sorry

end remainder_problem_l3207_320704


namespace vector_problem_l3207_320707

/-- Given vectors in 2D space -/
def OA : Fin 2 → ℝ := ![1, -2]
def OB : Fin 2 → ℝ := ![4, -1]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, m + 1]

/-- Vector AB -/
def AB : Fin 2 → ℝ := ![3, 1]

/-- Vector AC -/
def AC (m : ℝ) : Fin 2 → ℝ := ![m - 1, m + 3]

/-- Vector BC -/
def BC (m : ℝ) : Fin 2 → ℝ := ![m - 4, m + 2]

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- Two vectors are perpendicular if their dot product is zero -/
def are_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 = 0

/-- Triangle ABC is right-angled if any two of its sides are perpendicular -/
def is_right_angled (m : ℝ) : Prop :=
  are_perpendicular AB (AC m) ∨ are_perpendicular AB (BC m) ∨ are_perpendicular (AC m) (BC m)

theorem vector_problem (m : ℝ) :
  (are_parallel AB (OC m) → m = -3/2) ∧
  (is_right_angled m → m = 0 ∨ m = 5/2) := by
  sorry

end vector_problem_l3207_320707


namespace compute_expression_l3207_320723

theorem compute_expression : 3 * 3^4 + 9^60 / 9^58 = 324 := by sorry

end compute_expression_l3207_320723


namespace min_sum_with_geometric_mean_l3207_320750

theorem min_sum_with_geometric_mean (a b : ℝ) : 
  a > 0 → b > 0 → (Real.sqrt (3^a * 3^b) = Real.sqrt (3^a * 3^b)) → 
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt (3^x * 3^y) = Real.sqrt (3^x * 3^y) → x + y ≥ min :=
sorry

end min_sum_with_geometric_mean_l3207_320750


namespace range_of_a_l3207_320749

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem range_of_a (a : ℝ) :
  (∃ p q : ℝ, p < q ∧ 
    (∀ x : ℝ, x > 0 → (f a x ≤ 0 ↔ p ≤ x ∧ x ≤ q))) →
  0 < a ∧ a < 1 / Real.exp 1 :=
sorry

end range_of_a_l3207_320749


namespace largest_prime_factor_of_12321_l3207_320760

theorem largest_prime_factor_of_12321 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 12321 → q ≤ p :=
by sorry

end largest_prime_factor_of_12321_l3207_320760


namespace quadratic_inequality_solution_l3207_320716

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 8 * x + 3 < 0) ↔ (1/3 < x ∧ x < 3) := by
  sorry

end quadratic_inequality_solution_l3207_320716


namespace carries_tshirt_purchase_l3207_320751

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℚ := 9.95

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 20

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℚ := tshirt_cost * num_tshirts

/-- Theorem stating that the total cost of Carrie's t-shirt purchase is $199 -/
theorem carries_tshirt_purchase : total_cost = 199 := by
  sorry

end carries_tshirt_purchase_l3207_320751


namespace angle_U_measure_l3207_320790

-- Define the hexagon and its angles
structure Hexagon :=
  (F I G U R E : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.F + h.I + h.G + h.U + h.R + h.E = 720

def angles_congruent (h : Hexagon) : Prop :=
  h.F = h.I ∧ h.I = h.U

def angles_supplementary (h : Hexagon) : Prop :=
  h.G + h.R = 180 ∧ h.E + h.U = 180

-- Theorem statement
theorem angle_U_measure (h : Hexagon) 
  (valid : is_valid_hexagon h) 
  (congruent : angles_congruent h)
  (supplementary : angles_supplementary h) : 
  h.U = 120 := by
  sorry

end angle_U_measure_l3207_320790


namespace red_balls_count_l3207_320796

theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 30 →
  yellow = 8 →
  purple = 3 →
  prob = 88/100 →
  prob = (white + green + yellow : ℚ) / total →
  ∃ red : ℕ, red = 9 ∧ total = white + green + yellow + red + purple :=
by sorry

end red_balls_count_l3207_320796


namespace f_local_min_at_one_f_no_local_max_l3207_320762

/-- The function f(x) = (x^3 - 1)^2 + 1 -/
def f (x : ℝ) : ℝ := (x^3 - 1)^2 + 1

/-- f has a local minimum at x = 1 -/
theorem f_local_min_at_one : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1 :=
sorry

/-- f has no local maximum points -/
theorem f_no_local_max : 
  ¬∃ a, ∃ δ > 0, ∀ x, |x - a| < δ → f x ≤ f a :=
sorry

end f_local_min_at_one_f_no_local_max_l3207_320762


namespace park_diameter_is_40_l3207_320706

/-- Represents the circular park with its components -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of the jogging path -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.path_width)

/-- Theorem stating that for the given park dimensions, the outer boundary diameter is 40 feet -/
theorem park_diameter_is_40 :
  let park : CircularPark := {
    pond_diameter := 12,
    garden_width := 10,
    path_width := 4
  }
  outer_boundary_diameter park = 40 := by sorry

end park_diameter_is_40_l3207_320706


namespace survey_analysis_l3207_320779

/-- Represents the survey data and population information -/
structure SurveyData where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  surveyed_male : ℕ
  surveyed_female : ℕ
  male_enthusiasts : ℕ
  male_non_enthusiasts : ℕ
  female_enthusiasts : ℕ
  female_non_enthusiasts : ℕ

/-- Calculates the K² value for the chi-square test -/
def calculate_k_squared (data : SurveyData) : ℚ :=
  let n := data.surveyed_male + data.surveyed_female
  let a := data.male_enthusiasts
  let b := data.male_non_enthusiasts
  let c := data.female_enthusiasts
  let d := data.female_non_enthusiasts
  (n : ℚ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The main theorem to prove -/
theorem survey_analysis (data : SurveyData) 
    (h1 : data.total_students = 9000)
    (h2 : data.male_students = 4000)
    (h3 : data.female_students = 5000)
    (h4 : data.surveyed_male = 40)
    (h5 : data.surveyed_female = 50)
    (h6 : data.male_enthusiasts = 20)
    (h7 : data.male_non_enthusiasts = 20)
    (h8 : data.female_enthusiasts = 40)
    (h9 : data.female_non_enthusiasts = 10) :
    (data.surveyed_male : ℚ) / data.surveyed_female = data.male_students / data.female_students ∧
    calculate_k_squared data > 6635 / 1000 := by
  sorry

end survey_analysis_l3207_320779


namespace inserted_numbers_sum_l3207_320711

theorem inserted_numbers_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ r : ℝ, r > 0 ∧ a = 4 * r ∧ b = 4 * r^2) →  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ 16 = b + d) →           -- Arithmetic progression condition
  b = a + 4 →                                   -- Difference condition
  a + b = 8 + 4 * Real.sqrt 5 := by
sorry

end inserted_numbers_sum_l3207_320711


namespace squares_in_unit_square_l3207_320793

/-- Two squares with side lengths a and b contained in a unit square without sharing interior points have a + b ≤ 1 -/
theorem squares_in_unit_square (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) 
  (contained : a ≤ 1 ∧ b ≤ 1) 
  (no_overlap : ∃ (x y x' y' : ℝ), 
    0 ≤ x ∧ x + a ≤ 1 ∧ 
    0 ≤ y ∧ y + a ≤ 1 ∧
    0 ≤ x' ∧ x' + b ≤ 1 ∧ 
    0 ≤ y' ∧ y' + b ≤ 1 ∧
    (x + a ≤ x' ∨ x' + b ≤ x ∨ y + a ≤ y' ∨ y' + b ≤ y)) : 
  a + b ≤ 1 := by
sorry

end squares_in_unit_square_l3207_320793


namespace equation_solutions_l3207_320737

theorem equation_solutions :
  (∀ x : ℝ, (1/2 * x^2 = 5) ↔ (x = Real.sqrt 10 ∨ x = -Real.sqrt 10)) ∧
  (∀ x : ℝ, ((x - 1)^2 = 16) ↔ (x = 5 ∨ x = -3)) :=
by sorry

end equation_solutions_l3207_320737


namespace fraction_sum_equation_l3207_320798

theorem fraction_sum_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 5 * b) / (b + 5 * a) = 2) : 
  a / b = 0.6 := by sorry

end fraction_sum_equation_l3207_320798


namespace quadratic_function_property_l3207_320740

/-- Given a quadratic function f(x) = ax^2 + bx - 3 where a ≠ 0,
    if f(2) = f(4), then f(6) = -3 -/
theorem quadratic_function_property (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - 3
  f 2 = f 4 → f 6 = -3 := by
sorry

end quadratic_function_property_l3207_320740


namespace arithmetic_mean_midpoint_l3207_320719

/-- Given two points on a number line, their arithmetic mean is located halfway between them -/
theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m - a = b - m := by
  sorry

end arithmetic_mean_midpoint_l3207_320719


namespace arithmetic_sequence_product_l3207_320799

theorem arithmetic_sequence_product (a b c d : ℝ) (m n p : ℕ+) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  b - a = Real.sqrt 2 →
  c - b = Real.sqrt 2 →
  d - c = Real.sqrt 2 →
  a * b * c * d = 2021 →
  d = (m + Real.sqrt n) / Real.sqrt p →
  ∀ (q : ℕ+), q * q ∣ m → q = 1 →
  ∀ (q : ℕ+), q * q ∣ n → q = 1 →
  ∀ (q : ℕ+), q * q ∣ p → q = 1 →
  m + n + p = 100 := by
sorry

end arithmetic_sequence_product_l3207_320799


namespace jump_rope_record_time_l3207_320736

theorem jump_rope_record_time (record : ℕ) (jumps_per_second : ℕ) : 
  record = 54000 → jumps_per_second = 3 → 
  ∃ (hours : ℕ), hours = 5 ∧ hours * (jumps_per_second * 3600) > record :=
by sorry

end jump_rope_record_time_l3207_320736


namespace minimum_vases_for_70_flowers_l3207_320797

/-- Represents the capacity of each vase type -/
structure VaseCapacity where
  a : Nat
  b : Nat
  c : Nat

/-- Represents the number of each vase type -/
structure VaseCount where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the total number of flowers that can be held by the given vases -/
def totalFlowers (capacity : VaseCapacity) (count : VaseCount) : Nat :=
  capacity.a * count.a + capacity.b * count.b + capacity.c * count.c

/-- Checks if the given vase count is sufficient to hold the total number of flowers -/
def isSufficient (capacity : VaseCapacity) (count : VaseCount) (total : Nat) : Prop :=
  totalFlowers capacity count ≥ total

/-- Checks if the given vase count is the minimum required to hold the total number of flowers -/
def isMinimum (capacity : VaseCapacity) (count : VaseCount) (total : Nat) : Prop :=
  isSufficient capacity count total ∧
  ∀ (other : VaseCount), isSufficient capacity other total →
    count.a + count.b + count.c ≤ other.a + other.b + other.c

/-- Theorem: The minimum number of vases required to hold 70 flowers is 8 vases C, 1 vase B, and 0 vases A -/
theorem minimum_vases_for_70_flowers :
  let capacity := VaseCapacity.mk 4 6 8
  let count := VaseCount.mk 0 1 8
  isMinimum capacity count 70 := by
  sorry

end minimum_vases_for_70_flowers_l3207_320797


namespace imaginary_part_of_complex_expression_l3207_320733

theorem imaginary_part_of_complex_expression :
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / i * i
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_expression_l3207_320733


namespace semicircle_perimeter_approx_l3207_320755

/-- The perimeter of a semi-circle with radius 21.977625925131363 cm is approximately 113.024 cm. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 21.977625925131363
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 113.024) < ε :=
by sorry

end semicircle_perimeter_approx_l3207_320755


namespace quadratic_inequality_problem_l3207_320768

theorem quadratic_inequality_problem (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  (a = -12 ∧ b = -2) ∧
  (∀ x : ℝ, (a*x + b) / (x - 2) ≥ 0 ↔ -1/6 ≤ x ∧ x < 2) :=
sorry

end quadratic_inequality_problem_l3207_320768


namespace nine_integer_chord_lengths_l3207_320708

/-- Represents a circle with a given radius and a point inside it -/
structure CircleWithPoint where
  radius : ℝ
  pointDistance : ℝ

/-- Counts the number of different integer chord lengths containing the given point -/
def countIntegerChordLengths (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle of radius 25 and a point 13 units from the center,
    there are exactly 9 different integer chord lengths -/
theorem nine_integer_chord_lengths :
  let c := CircleWithPoint.mk 25 13
  countIntegerChordLengths c = 9 :=
sorry

end nine_integer_chord_lengths_l3207_320708


namespace exists_non_intersecting_line_l3207_320735

/-- Represents a domino on a chessboard -/
structure Domino where
  x : Fin 6
  y : Fin 6
  horizontal : Bool

/-- Represents a 6x6 chessboard covered by 18 dominoes -/
structure ChessboardCovering where
  dominoes : Finset Domino
  count : dominoes.card = 18
  valid_placement : ∀ d ∈ dominoes, 
    if d.horizontal
    then d.x < 5
    else d.y < 5
  covers_board : ∀ x y : Fin 6, ∃ d ∈ dominoes,
    (d.x = x ∧ d.y = y) ∨
    (d.horizontal ∧ d.x = x - 1 ∧ d.y = y) ∨
    (¬d.horizontal ∧ d.x = x ∧ d.y = y - 1)

/-- Main theorem: There exists a horizontal or vertical line that doesn't intersect any domino -/
theorem exists_non_intersecting_line (c : ChessboardCovering) :
  (∃ x : Fin 5, ∀ d ∈ c.dominoes, d.x ≠ x ∧ d.x ≠ x + 1) ∨
  (∃ y : Fin 5, ∀ d ∈ c.dominoes, d.y ≠ y ∧ d.y ≠ y + 1) :=
sorry

end exists_non_intersecting_line_l3207_320735


namespace product_of_fraction_is_111_l3207_320765

/-- The repeating decimal 0.009̄ as a real number -/
def repeating_decimal : ℚ := 1 / 111

/-- The product of numerator and denominator when the repeating decimal is expressed as a fraction in lowest terms -/
def product_of_fraction : ℕ := 111

/-- Theorem stating that the product of the numerator and denominator of the fraction representation of 0.009̄ in lowest terms is 111 -/
theorem product_of_fraction_is_111 : 
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n * d = product_of_fraction :=
by sorry

end product_of_fraction_is_111_l3207_320765


namespace count_primes_with_no_three_distinct_roots_l3207_320766

theorem count_primes_with_no_three_distinct_roots : 
  ∃ (S : Finset Nat), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (∀ p ∉ S, ¬Nat.Prime p ∨ 
      ∃ (x y z : Nat), x < p ∧ y < p ∧ z < p ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      (x^3 - 5*x^2 - 22*x + 56) % p = 0 ∧
      (y^3 - 5*y^2 - 22*y + 56) % p = 0 ∧
      (z^3 - 5*z^2 - 22*z + 56) % p = 0) ∧
    S.card = 4 := by
  sorry

end count_primes_with_no_three_distinct_roots_l3207_320766


namespace vidya_age_difference_l3207_320732

theorem vidya_age_difference : 
  let vidya_age : ℕ := 13
  let mother_age : ℕ := 44
  mother_age - 3 * vidya_age = 5 := by sorry

end vidya_age_difference_l3207_320732


namespace right_triangle_hypotenuse_l3207_320701

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) : 
  base = 12 →
  (1/2) * base * height = 30 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 13 := by
  sorry

end right_triangle_hypotenuse_l3207_320701


namespace students_just_passed_l3207_320753

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) (third_div_percent : ℚ) 
  (h_total : total = 500)
  (h_first : first_div_percent = 30 / 100)
  (h_second : second_div_percent = 45 / 100)
  (h_third : third_div_percent = 20 / 100)
  (h_sum_lt_1 : first_div_percent + second_div_percent + third_div_percent < 1) :
  total - (total * (first_div_percent + second_div_percent + third_div_percent)).floor = 25 := by
  sorry

end students_just_passed_l3207_320753


namespace cos_F_value_l3207_320717

-- Define the triangle
def Triangle (DE DF : ℝ) : Prop :=
  DE > 0 ∧ DF > 0 ∧ DE < DF

-- Define right triangle
def RightTriangle (DE DF : ℝ) : Prop :=
  Triangle DE DF ∧ DE^2 + (DF^2 - DE^2) = DF^2

-- Theorem statement
theorem cos_F_value (DE DF : ℝ) :
  RightTriangle DE DF → DE = 8 → DF = 17 → Real.cos (Real.arccos (DE / DF)) = 8 / 17 := by
  sorry

end cos_F_value_l3207_320717


namespace t_perimeter_is_14_l3207_320715

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a T-shaped figure formed by two rectangles -/
def t_perimeter (top : Rectangle) (bottom : Rectangle) : ℝ :=
  let exposed_top := top.width
  let exposed_sides := (top.width - bottom.width) + 2 * bottom.height
  let exposed_bottom := bottom.width
  exposed_top + exposed_sides + exposed_bottom

/-- Theorem stating that the perimeter of the T-shaped figure is 14 inches -/
theorem t_perimeter_is_14 :
  let top := Rectangle.mk 6 1
  let bottom := Rectangle.mk 3 4
  t_perimeter top bottom = 14 := by
  sorry

end t_perimeter_is_14_l3207_320715


namespace unique_magnitude_for_complex_roots_l3207_320777

theorem unique_magnitude_for_complex_roots (z : ℂ) : 
  z^2 - 6*z + 20 = 0 → ∃! m : ℝ, ∃ z : ℂ, z^2 - 6*z + 20 = 0 ∧ Complex.abs z = m :=
by
  sorry

end unique_magnitude_for_complex_roots_l3207_320777


namespace initial_channels_l3207_320792

theorem initial_channels (x : ℕ) : 
  x - 20 + 12 - 10 + 8 + 7 = 147 → x = 150 := by
  sorry

end initial_channels_l3207_320792


namespace rick_ironing_l3207_320783

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick has ironed -/
def total_pieces : ℕ := shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing :
  total_pieces = 27 := by sorry

end rick_ironing_l3207_320783


namespace simplify_and_rationalize_l3207_320764

theorem simplify_and_rationalize (x : ℝ) :
  x = 8 / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) →
  x = 2 * Real.sqrt 3 / 9 := by
sorry

end simplify_and_rationalize_l3207_320764


namespace complex_equation_implies_product_l3207_320757

theorem complex_equation_implies_product (x y : ℝ) : 
  (x + Complex.I) * (3 + y * Complex.I) = (2 : ℂ) + 4 * Complex.I → x * y = 1 := by
  sorry

end complex_equation_implies_product_l3207_320757


namespace factorial_of_factorial_divided_by_factorial_l3207_320710

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_of_factorial_divided_by_factorial_l3207_320710


namespace sqrt_equation_solution_l3207_320756

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l3207_320756


namespace team_ate_96_point_5_slices_l3207_320743

/-- The total number of pizza slices initially bought -/
def total_slices : ℝ := 116

/-- The number of pizza slices left after eating -/
def slices_left : ℝ := 19.5

/-- The number of pizza slices eaten by the team -/
def slices_eaten : ℝ := total_slices - slices_left

theorem team_ate_96_point_5_slices : slices_eaten = 96.5 := by
  sorry

end team_ate_96_point_5_slices_l3207_320743


namespace jim_can_bake_two_loaves_l3207_320758

/-- The amount of flour Jim has in the cupboard (in grams) -/
def flour_cupboard : ℕ := 200

/-- The amount of flour Jim has on the kitchen counter (in grams) -/
def flour_counter : ℕ := 100

/-- The amount of flour Jim has in the pantry (in grams) -/
def flour_pantry : ℕ := 100

/-- The amount of flour required for one loaf of bread (in grams) -/
def flour_per_loaf : ℕ := 200

/-- The total amount of flour Jim has (in grams) -/
def total_flour : ℕ := flour_cupboard + flour_counter + flour_pantry

/-- The number of loaves Jim can bake -/
def loaves_baked : ℕ := total_flour / flour_per_loaf

theorem jim_can_bake_two_loaves : loaves_baked = 2 := by
  sorry

end jim_can_bake_two_loaves_l3207_320758


namespace smallest_n_for_2007_l3207_320761

theorem smallest_n_for_2007 : 
  (∃ (n : ℕ) (S : Finset ℕ), 
    n > 1 ∧ 
    S.card = n ∧ 
    (∀ x ∈ S, x > 0) ∧ 
    S.prod id = 2007 ∧ 
    S.sum id = 2007 ∧ 
    (∀ m : ℕ, m > 1 → 
      (∃ T : Finset ℕ, 
        T.card = m ∧ 
        (∀ x ∈ T, x > 0) ∧ 
        T.prod id = 2007 ∧ 
        T.sum id = 2007) → 
      n ≤ m)) ∧ 
  (∀ S : Finset ℕ, 
    S.card > 1 ∧ 
    (∀ x ∈ S, x > 0) ∧ 
    S.prod id = 2007 ∧ 
    S.sum id = 2007 → 
    S.card ≥ 1337) :=
sorry

end smallest_n_for_2007_l3207_320761


namespace profit_at_4_max_profit_price_l3207_320713

noncomputable section

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := 10 / (x - 2) + 4 * (x - 6)^2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 2) * sales_volume x

-- Theorem for part (1)
theorem profit_at_4 : profit 4 = 42 := by sorry

-- Theorem for part (2)
theorem max_profit_price : 
  ∃ (x : ℝ), 2 < x ∧ x < 6 ∧ 
  (∀ (y : ℝ), 2 < y ∧ y < 6 → profit y ≤ profit x) ∧
  x = 10/3 := by sorry

end

end profit_at_4_max_profit_price_l3207_320713


namespace sum_of_solutions_eq_eleven_twelfths_l3207_320778

theorem sum_of_solutions_eq_eleven_twelfths :
  let f : ℝ → ℝ := λ x ↦ (4*x + 7)*(3*x - 8) + 12
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x + y = 11/12) :=
by
  sorry

end sum_of_solutions_eq_eleven_twelfths_l3207_320778


namespace fourth_corner_rectangle_area_l3207_320795

/-- Given a large rectangle divided into 9 smaller rectangles, where three corner rectangles
    have areas 9, 15, and 12, and the area ratios are the same between adjacent small rectangles,
    the area of the fourth corner rectangle is 20. -/
theorem fourth_corner_rectangle_area :
  ∀ (A B C D : ℝ),
    A = 9 →
    B = 15 →
    C = 12 →
    A / C = B / D →
    D = 20 := by
  sorry

end fourth_corner_rectangle_area_l3207_320795


namespace sqrt_14_less_than_4_l3207_320787

theorem sqrt_14_less_than_4 : Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_less_than_4_l3207_320787


namespace percentage_decrease_l3207_320788

theorem percentage_decrease (x y z : ℝ) : 
  x = 1.2 * y → x = 0.48 * z → y = 0.4 * z :=
by sorry

end percentage_decrease_l3207_320788


namespace dress_price_discount_l3207_320725

theorem dress_price_discount (P : ℝ) : P > 0 → 
  (1 - 0.35) * (1 - 0.30) * P = 0.455 * P :=
by
  sorry

end dress_price_discount_l3207_320725


namespace multiply_divide_equation_l3207_320769

theorem multiply_divide_equation : ∃ x : ℝ, (3.242 * x) / 100 = 0.04863 ∧ abs (x - 1.5) < 0.001 := by
  sorry

end multiply_divide_equation_l3207_320769


namespace average_height_problem_l3207_320794

/-- Proves that in a class of 50 students, if the average height of 10 students is 167 cm
    and the average height of the whole class is 168.6 cm, then the average height of
    the remaining 40 students is 169 cm. -/
theorem average_height_problem (total_students : ℕ) (group1_students : ℕ) 
  (group2_height : ℝ) (class_avg_height : ℝ) :
  total_students = 50 →
  group1_students = 40 →
  group2_height = 167 →
  class_avg_height = 168.6 →
  ∃ (group1_height : ℝ),
    group1_height = 169 ∧
    (group1_students : ℝ) * group1_height + (total_students - group1_students : ℝ) * group2_height =
      (total_students : ℝ) * class_avg_height :=
by sorry

end average_height_problem_l3207_320794


namespace chris_breath_holding_goal_l3207_320721

def breath_holding_sequence (n : ℕ) : ℕ :=
  10 * n

theorem chris_breath_holding_goal :
  breath_holding_sequence 6 = 60 := by
  sorry

end chris_breath_holding_goal_l3207_320721


namespace number_pair_theorem_l3207_320741

theorem number_pair_theorem (S P x y : ℝ) (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) ∧
  S^2 ≥ 4*P := by
  sorry

end number_pair_theorem_l3207_320741


namespace average_age_combined_group_l3207_320742

/-- Calculate the average age of a combined group of sixth-graders, parents, and teachers -/
theorem average_age_combined_group
  (n_students : ℕ) (avg_age_students : ℝ)
  (n_parents : ℕ) (avg_age_parents : ℝ)
  (n_teachers : ℕ) (avg_age_teachers : ℝ)
  (h_students : n_students = 40 ∧ avg_age_students = 12)
  (h_parents : n_parents = 50 ∧ avg_age_parents = 35)
  (h_teachers : n_teachers = 10 ∧ avg_age_teachers = 45) :
  let total_people := n_students + n_parents + n_teachers
  let total_age := n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  total_age / total_people = 26.8 :=
by sorry

end average_age_combined_group_l3207_320742


namespace three_planes_division_l3207_320720

/-- A plane in three-dimensional space -/
structure Plane3D where
  -- Add necessary fields to define a plane

/-- Represents the configuration of three planes in space -/
inductive PlaneConfiguration
  | AllParallel
  | TwoParallelOneIntersecting
  | IntersectAlongLine
  | IntersectPairwiseParallelLines
  | IntersectPairwiseAtPoint

/-- Counts the number of parts that three planes divide space into -/
def countParts (config : PlaneConfiguration) : ℕ :=
  match config with
  | .AllParallel => 4
  | .TwoParallelOneIntersecting => 6
  | .IntersectAlongLine => 6
  | .IntersectPairwiseParallelLines => 7
  | .IntersectPairwiseAtPoint => 8

/-- The set of possible numbers of parts -/
def possiblePartCounts : Set ℕ := {4, 6, 7, 8}

theorem three_planes_division (p1 p2 p3 : Plane3D) 
  (h : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
  ∃ (config : PlaneConfiguration), countParts config ∈ possiblePartCounts := by
  sorry

end three_planes_division_l3207_320720


namespace color_copies_comparison_l3207_320782

/-- The cost per color copy at print shop X -/
def cost_X : ℚ := 1.25

/-- The cost per color copy at print shop Y -/
def cost_Y : ℚ := 2.75

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℚ := 60

/-- The number of color copies being compared -/
def n : ℚ := 40

theorem color_copies_comparison :
  cost_Y * n = cost_X * n + additional_charge := by
  sorry

end color_copies_comparison_l3207_320782


namespace smallest_three_digit_square_append_l3207_320791

/-- A function that appends two numbers -/
def append (a b : ℕ) : ℕ := a * (10 ^ (Nat.digits 10 b).length) + b

/-- Predicate to check if a number satisfies the given condition -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (m : ℕ), append n (n + 1) = m ^ 2

/-- The smallest three-digit number satisfying the condition -/
def smallest_satisfying_number : ℕ := 183

theorem smallest_three_digit_square_append :
  (smallest_satisfying_number ≥ 100) ∧
  (smallest_satisfying_number < 1000) ∧
  satisfies_condition smallest_satisfying_number ∧
  ∀ n, n ≥ 100 ∧ n < smallest_satisfying_number → ¬(satisfies_condition n) :=
sorry

end smallest_three_digit_square_append_l3207_320791


namespace circumcenter_property_l3207_320759

-- Define the basic geometric structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def intersection_point (c1 c2 : Circle) : Point := sorry

def tangent_point (c : Circle) (p : Point) : Point := sorry

def is_parallelogram (p1 p2 p3 p4 : Point) : Prop := sorry

def is_circumcenter (p : Point) (t : Point × Point × Point) : Prop := sorry

-- State the theorem
theorem circumcenter_property 
  (X Y A B C P : Point) 
  (c1 c2 : Circle) :
  c1.center = X →
  c2.center = Y →
  A = intersection_point c1 c2 →
  B = tangent_point c1 A →
  C = tangent_point c2 A →
  is_parallelogram P X A Y →
  is_circumcenter P (B, C, A) :=
sorry

end circumcenter_property_l3207_320759


namespace derivative_f_at_one_l3207_320789

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 2 := by sorry

end derivative_f_at_one_l3207_320789


namespace gcd_of_390_455_546_l3207_320724

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end gcd_of_390_455_546_l3207_320724


namespace pigeonhole_on_permutation_sums_l3207_320712

theorem pigeonhole_on_permutation_sums (n : ℕ) :
  ∀ (p : Fin (2 * n) → Fin (2 * n)),
  Function.Bijective p →
  ∃ i j : Fin (2 * n), i ≠ j ∧ 
    (p i + i.val + 1) % (2 * n) = (p j + j.val + 1) % (2 * n) := by
  sorry

end pigeonhole_on_permutation_sums_l3207_320712


namespace isosceles_triangle_height_l3207_320752

/-- Given an isosceles triangle and a rectangle with the same area,
    where the base of the triangle equals the width of the rectangle,
    prove that the height of the triangle is twice the length of the rectangle. -/
theorem isosceles_triangle_height (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  (l * w = 1/2 * w * h) →  -- Areas are equal
  (h = 2 * l) := by
sorry

end isosceles_triangle_height_l3207_320752


namespace sheela_savings_percentage_l3207_320738

/-- Given Sheela's deposit and monthly income, prove the percentage of income deposited -/
theorem sheela_savings_percentage (deposit : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 5000)
  (h2 : monthly_income = 25000) :
  (deposit / monthly_income) * 100 = 20 := by
  sorry

end sheela_savings_percentage_l3207_320738


namespace smallest_prime_factor_of_reversed_difference_l3207_320763

theorem smallest_prime_factor_of_reversed_difference (A B C : ℕ) 
  (h1 : A ≠ C) 
  (h2 : A ≤ 9) (h3 : B ≤ 9) (h4 : C ≤ 9) 
  (h5 : A ≠ 0) :
  let ABC := 100 * A + 10 * B + C
  let CBA := 100 * C + 10 * B + A
  ∃ (k : ℕ), ABC - CBA = 3 * k ∧ 
  ∀ (p : ℕ), p < 3 → ¬(∃ (m : ℕ), ABC - CBA = p * m) :=
by sorry

end smallest_prime_factor_of_reversed_difference_l3207_320763


namespace antifreeze_concentration_l3207_320703

/-- Proves that the concentration of the certain antifreeze is 100% -/
theorem antifreeze_concentration
  (total_volume : ℝ)
  (final_concentration : ℝ)
  (certain_volume : ℝ)
  (other_concentration : ℝ)
  (h1 : total_volume = 55)
  (h2 : final_concentration = 0.20)
  (h3 : certain_volume = 6.11)
  (h4 : other_concentration = 0.10)
  : ∃ (certain_concentration : ℝ),
    certain_concentration = 1 ∧
    certain_volume * certain_concentration +
    (total_volume - certain_volume) * other_concentration =
    total_volume * final_concentration :=
by sorry

end antifreeze_concentration_l3207_320703


namespace prob_less_than_3_l3207_320718

/-- A fair cubic die with faces labeled 1 to 6 -/
structure FairDie :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The event of rolling a number less than 3 -/
def LessThan3 (d : FairDie) : Finset Nat :=
  d.faces.filter (λ x => x < 3)

/-- The probability of an event for a fair die -/
def Probability (d : FairDie) (event : Finset Nat) : Rat :=
  (event.card : Rat) / (d.faces.card : Rat)

theorem prob_less_than_3 (d : FairDie) :
  Probability d (LessThan3 d) = 1/3 := by
  sorry

end prob_less_than_3_l3207_320718


namespace age_difference_l3207_320781

/-- Given that the sum of A's and B's ages is 18 years more than the sum of B's and C's ages,
    prove that A is 18 years older than C. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a = c + 18 := by
  sorry

end age_difference_l3207_320781


namespace john_quilt_cost_l3207_320702

def quilt_cost (length width cost_per_sqft discount_rate tax_rate : ℝ) : ℝ :=
  let area := length * width
  let initial_cost := area * cost_per_sqft
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_cost := discounted_cost * (1 + tax_rate)
  total_cost

theorem john_quilt_cost :
  quilt_cost 12 15 70 0.1 0.05 = 11907 := by
  sorry

end john_quilt_cost_l3207_320702


namespace angle_between_vectors_l3207_320709

/-- Given non-zero vectors a and b in a real inner product space, 
    if |a| = √2|b| and (a - b) ⊥ (2a + 3b), 
    then the angle between a and b is 3π/4. -/
theorem angle_between_vectors 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a‖ = Real.sqrt 2 * ‖b‖) 
  (h2 : @inner ℝ V _ (a - b) (2 • a + 3 • b) = 0) : 
  Real.arccos ((@inner ℝ V _ a b) / (‖a‖ * ‖b‖)) = 3 * Real.pi / 4 := by
  sorry

end angle_between_vectors_l3207_320709


namespace power_multiplication_l3207_320731

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end power_multiplication_l3207_320731


namespace unique_valid_list_l3207_320774

def isValidList (l : List Nat) : Prop :=
  l.length = 10 ∧
  (∀ n ∈ l, n % 2 = 0 ∧ n > 0) ∧
  (∀ i ∈ l.enum.tail, 
    let (idx, n) := i
    if n > 2 then 
      l[idx-1]? = some (n - 2)
    else 
      true) ∧
  (∀ i ∈ l.enum.tail,
    let (idx, n) := i
    if n % 4 = 0 then
      l[idx-1]? = some (n - 1)
    else
      true)

theorem unique_valid_list : 
  ∃! l : List Nat, isValidList l :=
sorry

end unique_valid_list_l3207_320774


namespace set_equality_l3207_320727

def S : Set (ℕ × ℕ) := {(x, y) | 2 * x + 3 * y = 16}

theorem set_equality : S = {(2, 4), (5, 2), (8, 0)} := by
  sorry

end set_equality_l3207_320727


namespace exists_hole_for_unit_cube_l3207_320729

/-- A hole in a cube is represented by a rectangle on one face of the cube -/
structure Hole :=
  (width : ℝ)
  (height : ℝ)

/-- A cube is represented by its edge length -/
structure Cube :=
  (edge : ℝ)

/-- A proposition that states a cube can pass through a hole -/
def CanPassThrough (c : Cube) (h : Hole) : Prop :=
  c.edge ≤ h.width ∧ c.edge ≤ h.height

/-- The main theorem stating that there exists a hole in a unit cube through which another unit cube can pass -/
theorem exists_hole_for_unit_cube :
  ∃ (h : Hole), CanPassThrough (Cube.mk 1) h ∧ h.width < 1 ∧ h.height < 1 :=
sorry

end exists_hole_for_unit_cube_l3207_320729


namespace distribute_7_4_l3207_320770

/-- The number of ways to distribute n identical objects into k identical containers,
    with each container containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 identical apples into 4 identical packages,
    with each package containing at least one apple. -/
theorem distribute_7_4 : distribute 7 4 = 350 := by sorry

end distribute_7_4_l3207_320770


namespace parallel_line_with_y_intercept_l3207_320767

/-- Given a line mx + ny + 1 = 0 parallel to 4x + 3y + 5 = 0 with y-intercept 1/3, prove m = -4 and n = -3 -/
theorem parallel_line_with_y_intercept (m n : ℝ) : 
  (∀ x y, m * x + n * y + 1 = 0 ↔ 4 * x + 3 * y + 5 = 0) →  -- parallel condition
  (∃ y, m * 0 + n * y + 1 = 0 ∧ y = 1/3) →                  -- y-intercept condition
  m = -4 ∧ n = -3 :=
by sorry

end parallel_line_with_y_intercept_l3207_320767


namespace cube_spheres_diagonal_outside_length_l3207_320772

/-- Given a cube with edge length 1 and identical spheres centered at each vertex,
    where each sphere touches three neighboring spheres, the length of the part of
    the space diagonal of the cube that lies outside the spheres is √3 - 1. -/
theorem cube_spheres_diagonal_outside_length :
  let cube_edge_length : ℝ := 1
  let sphere_radius : ℝ := cube_edge_length / 2
  let cube_diagonal : ℝ := Real.sqrt 3
  let diagonal_inside_spheres : ℝ := 2 * sphere_radius
  cube_diagonal - diagonal_inside_spheres = Real.sqrt 3 - 1 := by
  sorry

#check cube_spheres_diagonal_outside_length

end cube_spheres_diagonal_outside_length_l3207_320772


namespace ninth_term_of_arithmetic_sequence_l3207_320773

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 7/11)
  (h_seventeenth : a 17 = 5/6) :
  a 9 = 97/132 := by
  sorry

end ninth_term_of_arithmetic_sequence_l3207_320773


namespace specific_ellipse_equation_l3207_320744

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis length
  c : ℝ  -- Distance from center to focus

/-- The equation of an ellipse given its parameters -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / (e.a^2 - e.c^2) = 1

/-- Theorem: The equation of an ellipse with specific properties -/
theorem specific_ellipse_equation :
  ∀ (e : Ellipse),
    e.a = 9 →  -- Half of the major axis length (18/2)
    e.c = 3 →  -- One-third of the semi-major axis (trisecting condition)
    ∀ (x y : ℝ),
      ellipse_equation e x y ↔ x^2 / 81 + y^2 / 72 = 1 := by
  sorry

end specific_ellipse_equation_l3207_320744


namespace complex_power_result_l3207_320786

theorem complex_power_result : (((Complex.I * Real.sqrt 2) / (1 + Complex.I)) ^ 100 : ℂ) = 1 := by
  sorry

end complex_power_result_l3207_320786


namespace jose_pool_charge_ratio_l3207_320747

/-- Represents the daily revenue from Jose's swimming pool --/
def daily_revenue (kids_charge : ℚ) (adults_charge : ℚ) : ℚ :=
  8 * kids_charge + 10 * adults_charge

/-- Represents the weekly revenue from Jose's swimming pool --/
def weekly_revenue (kids_charge : ℚ) (adults_charge : ℚ) : ℚ :=
  7 * daily_revenue kids_charge adults_charge

/-- Theorem stating the ratio of adult to kid charge in Jose's swimming pool --/
theorem jose_pool_charge_ratio :
  ∃ (adults_charge : ℚ),
    weekly_revenue 3 adults_charge = 588 ∧
    adults_charge / 3 = 2 := by
  sorry


end jose_pool_charge_ratio_l3207_320747


namespace felipe_build_time_l3207_320748

/-- Represents the time taken by each person to build their house, including break time. -/
structure BuildTime where
  felipe : ℝ
  emilio : ℝ
  carlos : ℝ

/-- Represents the break time taken by each person during construction. -/
structure BreakTime where
  felipe : ℝ
  emilio : ℝ
  carlos : ℝ

/-- The theorem stating Felipe's total build time is 27 months given the problem conditions. -/
theorem felipe_build_time (bt : BuildTime) (brt : BreakTime) : bt.felipe = 27 :=
  by
  have h1 : bt.felipe = bt.emilio / 2 := sorry
  have h2 : bt.carlos = bt.felipe + bt.emilio := sorry
  have h3 : bt.felipe + bt.emilio + bt.carlos = 10.5 * 12 := sorry
  have h4 : brt.felipe = 6 := sorry
  have h5 : brt.emilio = 2 * brt.felipe := sorry
  have h6 : brt.carlos = brt.emilio / 2 := sorry
  have h7 : bt.felipe + brt.felipe = 27 := sorry
  sorry

#check felipe_build_time

end felipe_build_time_l3207_320748


namespace mineral_water_case_price_l3207_320771

/-- The price of a case of mineral water -/
def case_price (daily_consumption : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / ((daily_consumption * days) / bottles_per_case)

/-- Theorem stating the price of a case of mineral water is $12 -/
theorem mineral_water_case_price :
  case_price (1/2) 240 24 60 = 12 := by
  sorry

end mineral_water_case_price_l3207_320771


namespace moving_circle_locus_l3207_320746

-- Define the fixed circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the locus of the center of the moving circle
def locus (x y : ℝ) : Prop := (x + 2)^2 - (y^2 / 13^2) = 1 ∧ x < -1

-- State the theorem
theorem moving_circle_locus :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), circle_M x' y' → (x - x')^2 + (y - y')^2 = r^2) ∧
    (∀ (x' y' : ℝ), circle_N x' y' → (x - x')^2 + (y - y')^2 = r^2)) →
  locus x y :=
by sorry

end moving_circle_locus_l3207_320746


namespace notebook_cost_l3207_320734

theorem notebook_cost (initial_amount : ℕ) (poster_cost : ℕ) (bookmark_cost : ℕ)
  (num_posters : ℕ) (num_notebooks : ℕ) (num_bookmarks : ℕ) (remaining_amount : ℕ) :
  initial_amount = 40 →
  poster_cost = 5 →
  bookmark_cost = 2 →
  num_posters = 2 →
  num_notebooks = 3 →
  num_bookmarks = 2 →
  remaining_amount = 14 →
  ∃ (notebook_cost : ℕ),
    initial_amount = num_posters * poster_cost + num_notebooks * notebook_cost +
      num_bookmarks * bookmark_cost + remaining_amount ∧
    notebook_cost = 4 :=
by sorry

end notebook_cost_l3207_320734


namespace third_median_length_l3207_320785

/-- An isosceles triangle with specific median lengths and area -/
structure SpecialIsoscelesTriangle where
  -- Two sides of equal length
  base : ℝ
  leg : ℝ
  -- Two medians of equal length
  equalMedian : ℝ
  -- The third median
  thirdMedian : ℝ
  -- Constraints
  isIsosceles : base > 0 ∧ leg > 0
  equalMedianLength : equalMedian = 4
  areaConstraint : area = 3 * Real.sqrt 15
  -- Area calculation (placeholder)
  area : ℝ := sorry

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : SpecialIsoscelesTriangle) : 
  t.thirdMedian = 2 * Real.sqrt 37 := by
  sorry

end third_median_length_l3207_320785


namespace ball_probability_l3207_320780

theorem ball_probability (p_red p_yellow p_blue : ℝ) : 
  p_red = 0.48 → p_yellow = 0.35 → p_red + p_yellow + p_blue = 1 → p_blue = 0.17 := by
  sorry

end ball_probability_l3207_320780


namespace median_circumradius_inequality_l3207_320784

/-- A triangle with medians and circumradius -/
structure Triangle where
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  R : ℝ

/-- Theorem about the relationship between medians and circumradius of a triangle -/
theorem median_circumradius_inequality (t : Triangle) :
  t.m_a^2 + t.m_b^2 + t.m_c^2 ≤ 27 * t.R^2 / 4 ∧
  t.m_a + t.m_b + t.m_c ≤ 9 * t.R / 2 :=
by sorry

end median_circumradius_inequality_l3207_320784
