import Mathlib

namespace NUMINAMATH_CALUDE_carolyn_silverware_knife_percentage_l2969_296913

/-- Represents the composition of a silverware set -/
structure Silverware :=
  (knives : ℕ)
  (forks : ℕ)
  (spoons : ℕ)

/-- Calculates the total number of pieces in a silverware set -/
def Silverware.total (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Represents a trade of silverware pieces -/
structure Trade :=
  (knives_gained : ℕ)
  (spoons_lost : ℕ)

/-- Applies a trade to a silverware set -/
def Silverware.apply_trade (s : Silverware) (t : Trade) : Silverware :=
  { knives := s.knives + t.knives_gained,
    forks := s.forks,
    spoons := s.spoons - t.spoons_lost }

/-- Calculates the percentage of knives in a silverware set -/
def Silverware.knife_percentage (s : Silverware) : ℚ :=
  (s.knives : ℚ) / (s.total : ℚ) * 100

theorem carolyn_silverware_knife_percentage :
  let initial_set : Silverware := { knives := 6, forks := 12, spoons := 6 * 3 }
  let trade : Trade := { knives_gained := 10, spoons_lost := 6 }
  let final_set := initial_set.apply_trade trade
  final_set.knife_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_silverware_knife_percentage_l2969_296913


namespace NUMINAMATH_CALUDE_second_meeting_time_l2969_296910

/-- The time in seconds for Racing Magic to complete one lap -/
def racing_magic_lap_time : ℕ := 150

/-- The number of laps Charging Bull completes in one hour -/
def charging_bull_laps_per_hour : ℕ := 40

/-- The time in minutes when both vehicles meet at the starting point for the second time -/
def meeting_time : ℕ := 15

/-- Theorem stating that the vehicles meet at the starting point for the second time after 15 minutes -/
theorem second_meeting_time :
  let racing_magic_lap_time_min : ℚ := racing_magic_lap_time / 60
  let charging_bull_lap_time_min : ℚ := 60 / charging_bull_laps_per_hour
  Nat.lcm (Nat.ceil (racing_magic_lap_time_min * 2)) (Nat.ceil (charging_bull_lap_time_min * 2)) / 2 = meeting_time :=
sorry

end NUMINAMATH_CALUDE_second_meeting_time_l2969_296910


namespace NUMINAMATH_CALUDE_isosceles_triangle_parallel_lines_l2969_296946

theorem isosceles_triangle_parallel_lines (base : ℝ) (line1 line2 : ℝ) : 
  base = 20 →
  line2 > line1 →
  line1 * line1 = (1/3) * base * base →
  line2 * line2 = (2/3) * base * base →
  line2 - line1 = (20 * (Real.sqrt 6 - Real.sqrt 3)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_parallel_lines_l2969_296946


namespace NUMINAMATH_CALUDE_archimedes_segment_theorem_l2969_296965

/-- Archimedes' Theorem applied to segments -/
theorem archimedes_segment_theorem 
  (b c : ℝ) 
  (CT AK CK AT AB AC : ℝ) 
  (h1 : CT = AK) 
  (h2 : CK = AK + AB) 
  (h3 : AT = CK) 
  (h4 : AC = b) : 
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := by
  sorry

#check archimedes_segment_theorem

end NUMINAMATH_CALUDE_archimedes_segment_theorem_l2969_296965


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2969_296996

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 3)^2 + 4*(a 3) + 1 = 0 →  -- a_3 is a root of x^2 + 4x + 1 = 0
  (a 15)^2 + 4*(a 15) + 1 = 0 →  -- a_15 is a root of x^2 + 4x + 1 = 0
  a 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2969_296996


namespace NUMINAMATH_CALUDE_shifted_cosine_to_sine_l2969_296967

/-- Given a cosine function shifted to create an odd function, 
    prove the value at a specific point. -/
theorem shifted_cosine_to_sine (f g : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.cos (x / 2 - π / 3)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, g x = f (x - φ)) →
  (∀ x, g x + g (-x) = 0) →
  g (2 * φ + π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_cosine_to_sine_l2969_296967


namespace NUMINAMATH_CALUDE_power_of_two_equation_l2969_296938

theorem power_of_two_equation (m : ℤ) : 
  2^2000 - 3 * 2^1999 + 2^1998 - 2^1997 + 2^1996 = m * 2^1996 → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l2969_296938


namespace NUMINAMATH_CALUDE_cloth_selling_price_l2969_296901

/-- Given a cloth with the following properties:
  * Total length: 60 meters
  * Cost price per meter: 128 Rs
  * Profit per meter: 12 Rs
  Prove that the total selling price is 8400 Rs. -/
theorem cloth_selling_price 
  (total_length : ℕ) 
  (cost_price_per_meter : ℕ) 
  (profit_per_meter : ℕ) 
  (h1 : total_length = 60)
  (h2 : cost_price_per_meter = 128)
  (h3 : profit_per_meter = 12) :
  (cost_price_per_meter + profit_per_meter) * total_length = 8400 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l2969_296901


namespace NUMINAMATH_CALUDE_cumulonimbus_cloud_count_l2969_296990

theorem cumulonimbus_cloud_count :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 4 * cumulus →
    cumulus = 12 * cumulonimbus →
    cumulonimbus > 0 →
    cirrus = 144 →
    cumulonimbus = 3 := by
  sorry

end NUMINAMATH_CALUDE_cumulonimbus_cloud_count_l2969_296990


namespace NUMINAMATH_CALUDE_intersection_of_line_and_curve_l2969_296994

/-- Line l is defined by the equation 2x - y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- Curve C is defined by the equation y² = 2x -/
def curve_C (x y : ℝ) : Prop := y^2 = 2 * x

/-- The intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) := {(2, 2), (1/2, -1)}

/-- Theorem stating that the intersection points of line l and curve C are (2, 2) and (1/2, -1) -/
theorem intersection_of_line_and_curve :
  ∀ p : ℝ × ℝ, (line_l p.1 p.2 ∧ curve_C p.1 p.2) ↔ p ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_line_and_curve_l2969_296994


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sum_over_sqrt2_l2969_296987

theorem sqrt_sum_squares_ge_sum_over_sqrt2 (a b : ℝ) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sum_over_sqrt2_l2969_296987


namespace NUMINAMATH_CALUDE_peanut_butter_jar_servings_l2969_296930

/-- The number of servings in a jar of peanut butter -/
def peanut_butter_servings (jar_contents : ℚ) (serving_size : ℚ) : ℚ :=
  jar_contents / serving_size

theorem peanut_butter_jar_servings :
  let jar_contents : ℚ := 35 + 4/5
  let serving_size : ℚ := 2 + 1/3
  peanut_butter_servings jar_contents serving_size = 15 + 17/35 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_jar_servings_l2969_296930


namespace NUMINAMATH_CALUDE_intersection_of_intervals_l2969_296915

open Set

theorem intersection_of_intervals (A B : Set ℝ) :
  A = {x | -1 < x ∧ x < 2} →
  B = {x | 1 < x ∧ x < 3} →
  A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_l2969_296915


namespace NUMINAMATH_CALUDE_population_difference_l2969_296961

/-- The population difference between thrice Willowdale and Roseville -/
theorem population_difference (willowdale roseville suncity : ℕ) : 
  willowdale = 2000 →
  suncity = 12000 →
  suncity = 2 * roseville + 1000 →
  roseville < 3 * willowdale →
  3 * willowdale - roseville = 500 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_l2969_296961


namespace NUMINAMATH_CALUDE_expression_value_l2969_296950

theorem expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2969_296950


namespace NUMINAMATH_CALUDE_linear_function_problem_l2969_296985

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) ∧ 
  (∀ x : ℝ, f x = 3 * (f⁻¹ x) + 9) ∧
  f 2 = 5 ∧
  f 3 = 9 →
  f 5 = 9 - 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_problem_l2969_296985


namespace NUMINAMATH_CALUDE_modulo_23_equivalence_l2969_296960

theorem modulo_23_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 58294 ≡ n [ZMOD 23] ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_modulo_23_equivalence_l2969_296960


namespace NUMINAMATH_CALUDE_upper_limit_n_l2969_296971

def is_integer (x : ℚ) : Prop := ∃ k : ℤ, x = k

def has_exactly_three_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  n = p * q * r

theorem upper_limit_n :
  ∀ n : ℕ, n > 0 →
  is_integer (14 * n / 60) →
  has_exactly_three_prime_factors n →
  n ≤ 210 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_n_l2969_296971


namespace NUMINAMATH_CALUDE_correct_urea_decomposing_bacteria_culture_l2969_296981

-- Define the types of culture media
inductive CultureMedium
| SelectiveNitrogen
| IdentificationPhenolRed

-- Define the process of bacterial culture
def BacterialCulture := List CultureMedium

-- Define the property of being a correct culture process
def IsCorrectCulture (process : BacterialCulture) : Prop :=
  process = [CultureMedium.SelectiveNitrogen, CultureMedium.IdentificationPhenolRed]

-- Theorem: The correct culture process for urea-decomposing bacteria
theorem correct_urea_decomposing_bacteria_culture :
  IsCorrectCulture [CultureMedium.SelectiveNitrogen, CultureMedium.IdentificationPhenolRed] :=
by sorry

end NUMINAMATH_CALUDE_correct_urea_decomposing_bacteria_culture_l2969_296981


namespace NUMINAMATH_CALUDE_shape_count_l2969_296914

theorem shape_count (total_shapes : ℕ) (total_edges : ℕ) 
  (h1 : total_shapes = 13) 
  (h2 : total_edges = 47) : 
  ∃ (triangles squares : ℕ),
    triangles + squares = total_shapes ∧ 
    3 * triangles + 4 * squares = total_edges ∧
    triangles = 5 ∧ 
    squares = 8 := by
  sorry

end NUMINAMATH_CALUDE_shape_count_l2969_296914


namespace NUMINAMATH_CALUDE_rectangle_area_l2969_296916

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2969_296916


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2969_296984

theorem power_of_three_mod_five : 3^2040 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2969_296984


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2969_296972

theorem prime_pairs_dividing_sum_of_powers (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q → (p * q ∣ (5^p + 5^q)) ↔ 
    ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ 
     (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) ∨ 
     (p = 5 ∧ q = 5) ∨ (p = 5 ∧ q = 313) ∨ 
     (p = 313 ∧ q = 5)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2969_296972


namespace NUMINAMATH_CALUDE_binomial_10_3_l2969_296933

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2969_296933


namespace NUMINAMATH_CALUDE_function_property_l2969_296912

theorem function_property (a : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) 1, 
    ∀ y ∈ Set.Icc (0 : ℝ) 1, 
    ∀ z ∈ Set.Icc (0 : ℝ) 1, 
    (1/2) * a * x^2 - (x - 1) * Real.exp x + 
    (1/2) * a * y^2 - (y - 1) * Real.exp y ≥ 
    (1/2) * a * z^2 - (z - 1) * Real.exp z) →
  a ∈ Set.Icc 1 4 := by
sorry

end NUMINAMATH_CALUDE_function_property_l2969_296912


namespace NUMINAMATH_CALUDE_max_min_values_of_f_on_interval_l2969_296980

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x + 5

-- Define the interval
def interval : Set ℝ := {x | -5/2 ≤ x ∧ x ≤ 3/2}

-- Theorem statement
theorem max_min_values_of_f_on_interval :
  (∃ x ∈ interval, f x = 9 ∧ ∀ y ∈ interval, f y ≤ 9) ∧
  (∃ x ∈ interval, f x = -11.25 ∧ ∀ y ∈ interval, f y ≥ -11.25) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_on_interval_l2969_296980


namespace NUMINAMATH_CALUDE_total_dogs_l2969_296958

theorem total_dogs (brown : ℕ) (white : ℕ) (black : ℕ)
  (h1 : brown = 20)
  (h2 : white = 10)
  (h3 : black = 15) :
  brown + white + black = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_l2969_296958


namespace NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2969_296929

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem ten_factorial_mod_thirteen : 
  factorial 10 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2969_296929


namespace NUMINAMATH_CALUDE_expected_value_Y_l2969_296997

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define Y as a function of X
def Y (X : ℝ → ℝ) : ℝ → ℝ := λ ω => 2 * (X ω) + 7

-- Define the expectation operator M
def M (Z : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem expected_value_Y (hX : M X = 4) : M (Y X) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_Y_l2969_296997


namespace NUMINAMATH_CALUDE_rectangle_area_stage_8_l2969_296921

/-- The area of a rectangle formed by n squares, each measuring s by s units -/
def rectangleArea (n : ℕ) (s : ℝ) : ℝ := n * (s * s)

/-- Theorem: The area of a rectangle formed by 8 squares, each measuring 4 inches by 4 inches, is 128 square inches -/
theorem rectangle_area_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_8_l2969_296921


namespace NUMINAMATH_CALUDE_number_plus_two_equals_six_l2969_296908

theorem number_plus_two_equals_six :
  ∃ x : ℝ, (2 + x = 6) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_number_plus_two_equals_six_l2969_296908


namespace NUMINAMATH_CALUDE_odd_symmetric_function_sum_l2969_296964

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is symmetric about x=2 if f(2+x) = f(2-x) for all x -/
def IsSymmetricAbout2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem odd_symmetric_function_sum (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_sym : IsSymmetricAbout2 f) 
    (h_f2 : f 2 = 2018) : 
  f 2018 + f 2016 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_odd_symmetric_function_sum_l2969_296964


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_times_three_l2969_296974

theorem radical_conjugate_sum_times_three : 
  let x := 15 - Real.sqrt 500
  let y := 15 + Real.sqrt 500
  3 * (x + y) = 90 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_times_three_l2969_296974


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2969_296905

/-- A quadratic function with vertex at (-1, 4) passing through (2, -5) -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 4

theorem quadratic_function_properties :
  (∀ x, f x = -x^2 - 2*x + 3) ∧
  (f (-1/2) = 11/4) ∧
  (∀ x, f x = 3 ↔ x = 0 ∨ x = -2) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2969_296905


namespace NUMINAMATH_CALUDE_first_number_in_sum_l2969_296918

theorem first_number_in_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3.622) 
  (b_eq : b = 0.014) 
  (c_eq : c = 0.458) : 
  a = 3.15 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_sum_l2969_296918


namespace NUMINAMATH_CALUDE_dwarf_truth_count_l2969_296959

/-- Represents the number of dwarfs who tell the truth -/
def truthful_dwarfs : ℕ := sorry

/-- Represents the number of dwarfs who lie -/
def lying_dwarfs : ℕ := sorry

/-- The total number of dwarfs -/
def total_dwarfs : ℕ := 10

/-- The number of dwarfs who raised their hands for vanilla ice cream -/
def vanilla_hands : ℕ := 10

/-- The number of dwarfs who raised their hands for chocolate ice cream -/
def chocolate_hands : ℕ := 5

/-- The number of dwarfs who raised their hands for fruit ice cream -/
def fruit_hands : ℕ := 1

theorem dwarf_truth_count :
  truthful_dwarfs + lying_dwarfs = total_dwarfs ∧
  truthful_dwarfs + 2 * lying_dwarfs = vanilla_hands + chocolate_hands + fruit_hands ∧
  truthful_dwarfs = 4 := by sorry

end NUMINAMATH_CALUDE_dwarf_truth_count_l2969_296959


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2969_296962

theorem inequality_equivalence (x y : ℝ) : 
  (y - x < Real.sqrt (x^2 + 1)) ↔ (y < x + Real.sqrt (x^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2969_296962


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2969_296982

theorem complex_equation_solution (a : ℝ) (ha : a ≥ 0) :
  let S := {z : ℂ | z^2 + 2 * Complex.abs z = a}
  S = {z : ℂ | z = -(1 - Real.sqrt (1 + a)) ∨ z = (1 - Real.sqrt (1 + a))} ∪
      (if 0 ≤ a ∧ a ≤ 1 then
        {z : ℂ | z = Complex.I * (1 + Real.sqrt (1 - a)) ∨
                 z = Complex.I * (-(1 + Real.sqrt (1 - a))) ∨
                 z = Complex.I * (1 - Real.sqrt (1 - a)) ∨
                 z = Complex.I * (-(1 - Real.sqrt (1 - a)))}
      else ∅) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2969_296982


namespace NUMINAMATH_CALUDE_functional_equation_problem_l2969_296975

/-- The functional equation problem -/
theorem functional_equation_problem :
  ∀ (f h : ℝ → ℝ),
  (∀ x y : ℝ, f (x^2 + y * h x) = x * h x + f (x * y)) →
  ((∃ a b : ℝ, (∀ x : ℝ, f x = a) ∧ 
                (∀ x : ℝ, x ≠ 0 → h x = 0) ∧ 
                (h 0 = b)) ∨
   (∃ a : ℝ, (∀ x : ℝ, f x = x + a) ∧ 
             (∀ x : ℝ, h x = x))) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l2969_296975


namespace NUMINAMATH_CALUDE_unique_subset_existence_l2969_296942

theorem unique_subset_existence : 
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (pair : ℤ × ℤ), 
    pair.1 ∈ X ∧ pair.2 ∈ X ∧ pair.1 + 2 * pair.2 = n := by
  sorry

end NUMINAMATH_CALUDE_unique_subset_existence_l2969_296942


namespace NUMINAMATH_CALUDE_number_exchange_ratio_l2969_296991

theorem number_exchange_ratio (a b p q : ℝ) (h : p * q ≠ 1) :
  ∃ z : ℝ, (z + a - a) + ((p * z - a) + a) = q * ((z + a + b) - ((p * z - a) - b)) →
  z = (a + b) * (q + 1) / (p * q - 1) :=
by sorry

end NUMINAMATH_CALUDE_number_exchange_ratio_l2969_296991


namespace NUMINAMATH_CALUDE_quadratic_one_zero_l2969_296902

def f (x : ℝ) := x^2 - 4*x + 4

theorem quadratic_one_zero :
  ∃! x, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_l2969_296902


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2969_296939

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proof that a train's length is approximately 129.96 meters -/
theorem train_length_proof (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 52)
  (h2 : time_s = 9) :
  ∃ ε > 0, |train_length speed_kmh time_s - 129.96| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2969_296939


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l2969_296904

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 395

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 950

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def mixture_ratio : ℚ := 3 / 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

theorem vegetable_ghee_weight : 
  weight_a * (mixture_ratio * total_volume / (1 + mixture_ratio)) + 
  weight_b * (total_volume / (1 + mixture_ratio)) = total_weight := by
  sorry

#check vegetable_ghee_weight

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l2969_296904


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2969_296919

/-- Given a function f: ℝ → ℝ with a tangent line y = -x + 8 at x = 5,
    prove that f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (fun x => -x + 8) = fun x => f 5 + (deriv f 5) * (x - 5)) : 
    f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2969_296919


namespace NUMINAMATH_CALUDE_taxi_charge_theorem_l2969_296920

/-- A taxi service with a given initial fee and per-distance charge -/
structure TaxiService where
  initialFee : ℚ
  chargePerIncrement : ℚ
  incrementDistance : ℚ

/-- Calculate the total charge for a given trip distance -/
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + service.chargePerIncrement * (distance / service.incrementDistance)

/-- Theorem: The total charge for a 3.6-mile trip with the given taxi service is $5.20 -/
theorem taxi_charge_theorem :
  let service : TaxiService := ⟨2.05, 0.35, 2/5⟩
  totalCharge service (36/10) = 26/5 := by
  sorry


end NUMINAMATH_CALUDE_taxi_charge_theorem_l2969_296920


namespace NUMINAMATH_CALUDE_value_of_a_satisfying_equation_l2969_296993

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- State the theorem
theorem value_of_a_satisfying_equation :
  ∃ a : ℝ, F a 3 8 = F a 5 12 ∧ a = -2/49 := by sorry

end NUMINAMATH_CALUDE_value_of_a_satisfying_equation_l2969_296993


namespace NUMINAMATH_CALUDE_power_calculation_l2969_296927

theorem power_calculation : 
  (27 : ℝ)^3 * 9^2 / 3^17 = 1/81 :=
by
  have h1 : (27 : ℝ) = 3^3 := by sorry
  have h2 : (9 : ℝ) = 3^2 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2969_296927


namespace NUMINAMATH_CALUDE_binary_1101001_is_105_and_odd_l2969_296956

-- Define the binary number as a list of bits
def binary_number : List Nat := [1, 1, 0, 1, 0, 0, 1]

-- Function to convert binary to decimal
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem statement
theorem binary_1101001_is_105_and_odd :
  (binary_to_decimal binary_number = 105) ∧ (105 % 2 = 1) := by
  sorry

#eval binary_to_decimal binary_number
#eval 105 % 2

end NUMINAMATH_CALUDE_binary_1101001_is_105_and_odd_l2969_296956


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l2969_296931

/-- A hyperbola with real axis length m and a point P on it -/
structure Hyperbola (m : ℝ) where
  /-- The distance from P to the right focus is m -/
  dist_right_focus : ℝ
  /-- The distance from P to the right focus equals m -/
  dist_right_focus_eq : dist_right_focus = m

/-- The theorem stating that the distance from P to the left focus is 2m -/
theorem distance_to_left_focus (m : ℝ) (h : Hyperbola m) : 
  ∃ (dist_left_focus : ℝ), dist_left_focus = 2 * m := by
  sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l2969_296931


namespace NUMINAMATH_CALUDE_investment_percentage_rate_l2969_296947

/-- Given an investment scenario, prove the percentage rate of the remaining investment --/
theorem investment_percentage_rate
  (total_investment : ℝ)
  (investment_at_five_percent : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 18000)
  (h2 : investment_at_five_percent = 6000)
  (h3 : total_interest = 660)
  : (total_interest - investment_at_five_percent * 0.05) / (total_investment - investment_at_five_percent) * 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_rate_l2969_296947


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2969_296998

theorem quadratic_equation_root (m : ℝ) : 
  (3 : ℝ) ^ 2 - 3 - m = 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2969_296998


namespace NUMINAMATH_CALUDE_power_mod_seven_l2969_296925

theorem power_mod_seven : 2^2004 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l2969_296925


namespace NUMINAMATH_CALUDE_transformed_triangle_area_l2969_296949

-- Define the function f on the domain {x₁, x₂, x₃}
variable (f : ℝ → ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the area of a triangle given three points
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area 
  (h1 : triangleArea (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 27) :
  triangleArea (x₁/2, 3 * f x₁) (x₂/2, 3 * f x₂) (x₃/2, 3 * f x₃) = 40.5 :=
sorry

end NUMINAMATH_CALUDE_transformed_triangle_area_l2969_296949


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2969_296976

/-- Given two parallel vectors a and b in R², prove that m = -3 --/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (1 + m, 1 - m)
  (∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) →
  m = -3 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2969_296976


namespace NUMINAMATH_CALUDE_parabola_shift_parabola_transformation_l2969_296900

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Theorem stating the equivalence of the original parabola after transformation and the shifted parabola
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by
  sorry

-- Theorem stating that the shifted parabola is the result of the described transformations
theorem parabola_transformation :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_parabola_transformation_l2969_296900


namespace NUMINAMATH_CALUDE_unused_ribbon_theorem_l2969_296937

/-- Represents the pattern of ribbon pieces -/
inductive RibbonPiece
  | two
  | four
  | six
  | eight
  | ten

/-- Returns the length of a ribbon piece in meters -/
def piece_length (p : RibbonPiece) : ℕ :=
  match p with
  | .two => 2
  | .four => 4
  | .six => 6
  | .eight => 8
  | .ten => 10

/-- Represents the pattern of ribbon usage -/
def ribbon_pattern : List RibbonPiece :=
  [.two, .two, .two, .four, .four, .six, .six, .six, .six, .eight, .ten, .ten]

/-- Calculates the unused ribbon length after following the pattern once -/
def unused_ribbon (total_length : ℕ) (pattern : List RibbonPiece) : ℕ :=
  let used := pattern.foldl (fun acc p => acc + piece_length p) 0
  total_length - (used % total_length)

theorem unused_ribbon_theorem :
  unused_ribbon 30 ribbon_pattern = 4 := by sorry

#eval unused_ribbon 30 ribbon_pattern

end NUMINAMATH_CALUDE_unused_ribbon_theorem_l2969_296937


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2969_296928

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2969_296928


namespace NUMINAMATH_CALUDE_total_distance_QY_l2969_296952

/-- Proves that the total distance between Q and Y is 45 km --/
theorem total_distance_QY (matthew_speed johnny_speed : ℝ)
  (johnny_distance : ℝ) (time_difference : ℝ) :
  matthew_speed = 3 →
  johnny_speed = 4 →
  johnny_distance = 24 →
  time_difference = 1 →
  ∃ (total_distance : ℝ), total_distance = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_total_distance_QY_l2969_296952


namespace NUMINAMATH_CALUDE_total_spent_calculation_l2969_296948

-- Define the prices and quantities
def shirt_price : ℝ := 15.00
def shirt_quantity : ℕ := 4
def pants_price : ℝ := 40.00
def pants_quantity : ℕ := 2
def suit_price : ℝ := 150.00
def suit_quantity : ℕ := 1
def sweater_price : ℝ := 30.00
def sweater_quantity : ℕ := 2
def tie_price : ℝ := 20.00
def tie_quantity : ℕ := 3
def shoes_price : ℝ := 80.00
def shoes_quantity : ℕ := 1

-- Define the discount rates
def shirt_discount : ℝ := 0.20
def pants_discount : ℝ := 0.30
def suit_discount : ℝ := 0.15
def coupon_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Define the theorem
theorem total_spent_calculation :
  let initial_total := shirt_price * shirt_quantity + pants_price * pants_quantity + 
                       suit_price * suit_quantity + sweater_price * sweater_quantity + 
                       tie_price * tie_quantity + shoes_price * shoes_quantity
  let discounted_shirts := shirt_price * shirt_quantity * (1 - shirt_discount)
  let discounted_pants := pants_price * pants_quantity * (1 - pants_discount)
  let discounted_suit := suit_price * suit_quantity * (1 - suit_discount)
  let discounted_total := discounted_shirts + discounted_pants + discounted_suit + 
                          sweater_price * sweater_quantity + tie_price * tie_quantity + 
                          shoes_price * shoes_quantity
  let coupon_applied := discounted_total * (1 - coupon_discount)
  let final_total := coupon_applied * (1 + sales_tax_rate)
  final_total = 407.77 := by
sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l2969_296948


namespace NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l2969_296926

theorem hyperbola_quadrilateral_area_ratio_max (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), x = a * b / (a^2 + b^2) ∧ ∀ (y : ℝ), y = a * b / (a^2 + b^2) → x ≥ y) →
  a * b / (a^2 + b^2) ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l2969_296926


namespace NUMINAMATH_CALUDE_pens_cost_gained_l2969_296970

/-- Represents the number of pens sold -/
def pens_sold : ℕ := 95

/-- Represents the gain percentage as a fraction -/
def gain_percentage : ℚ := 20 / 100

/-- Calculates the selling price given the cost price and gain percentage -/
def selling_price (cost : ℚ) : ℚ := cost * (1 + gain_percentage)

/-- Theorem stating that the number of pens' cost gained is 19 -/
theorem pens_cost_gained : 
  ∃ (cost : ℚ), cost > 0 ∧ 
  (pens_sold * (selling_price cost - cost) = 19 * cost) := by
  sorry

end NUMINAMATH_CALUDE_pens_cost_gained_l2969_296970


namespace NUMINAMATH_CALUDE_smallest_x_for_inequality_l2969_296986

theorem smallest_x_for_inequality : ∃ x : ℕ, (∀ y : ℕ, 27^y > 3^24 → x ≤ y) ∧ 27^x > 3^24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_inequality_l2969_296986


namespace NUMINAMATH_CALUDE_triangle_area_l2969_296906

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 15√3/4 when b = 7, c = 5, and B = 2π/3 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 7 → c = 5 → B = 2 * π / 3 → 
  (1/2) * b * c * Real.sin B = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2969_296906


namespace NUMINAMATH_CALUDE_profit_without_discount_is_fifty_percent_l2969_296923

/-- Represents the profit percentage and discount percentage as ratios -/
structure ProfitDiscount where
  profit : ℚ
  discount : ℚ

/-- Calculates the profit percentage without discount given the profit percentage with discount -/
def profit_without_discount (pd : ProfitDiscount) : ℚ :=
  (1 + pd.profit) / (1 - pd.discount) - 1

/-- Theorem stating that a 42.5% profit with a 5% discount results in a 50% profit without discount -/
theorem profit_without_discount_is_fifty_percent :
  let pd : ProfitDiscount := { profit := 425/1000, discount := 5/100 }
  profit_without_discount pd = 1/2 := by
sorry

end NUMINAMATH_CALUDE_profit_without_discount_is_fifty_percent_l2969_296923


namespace NUMINAMATH_CALUDE_graduating_class_size_l2969_296936

/-- The number of boys in the graduating class -/
def num_boys : ℕ := 208

/-- The difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 69

/-- The total number of students in the graduating class -/
def total_students : ℕ := num_boys + (num_boys + girl_boy_difference)

theorem graduating_class_size :
  total_students = 485 :=
by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l2969_296936


namespace NUMINAMATH_CALUDE_perpendicular_line_theorem_l2969_296969

/-- A figure in a plane -/
inductive PlaneFigure
  | Triangle
  | Trapezoid
  | CircleDiameters
  | HexagonSides

/-- Represents whether two lines in a figure are guaranteed to intersect -/
def guaranteed_intersection (figure : PlaneFigure) : Prop :=
  match figure with
  | PlaneFigure.Triangle => true
  | PlaneFigure.Trapezoid => false
  | PlaneFigure.CircleDiameters => true
  | PlaneFigure.HexagonSides => false

/-- A line perpendicular to two sides of a figure is perpendicular to the plane -/
def perpendicular_to_plane (figure : PlaneFigure) : Prop :=
  guaranteed_intersection figure

theorem perpendicular_line_theorem (figure : PlaneFigure) :
  perpendicular_to_plane figure ↔ (figure = PlaneFigure.Triangle ∨ figure = PlaneFigure.CircleDiameters) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_theorem_l2969_296969


namespace NUMINAMATH_CALUDE_rectangle_width_from_square_l2969_296922

theorem rectangle_width_from_square (square_side : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_length = 18 →
  4 * square_side = 2 * (rect_length + (4 * square_side - 2 * rect_length) / 2) →
  (4 * square_side - 2 * rect_length) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_from_square_l2969_296922


namespace NUMINAMATH_CALUDE_video_game_map_area_l2969_296911

-- Define the map dimensions
def map_width : ℝ := 10
def map_length : ℝ := 2

-- Define the area of a rectangle
def rectangle_area (width length : ℝ) : ℝ := width * length

-- Theorem statement
theorem video_game_map_area : rectangle_area map_width map_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_video_game_map_area_l2969_296911


namespace NUMINAMATH_CALUDE_initial_players_count_l2969_296992

/-- Represents a round-robin chess tournament. -/
structure ChessTournament where
  initial_players : ℕ
  matches_played : ℕ
  dropped_players : ℕ
  matches_per_dropped : ℕ

/-- Calculates the total number of matches in a round-robin tournament. -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the initial number of players in the tournament. -/
theorem initial_players_count (t : ChessTournament) 
  (h1 : t.matches_played = 84)
  (h2 : t.dropped_players = 2)
  (h3 : t.matches_per_dropped = 3) :
  t.initial_players = 15 := by
  sorry

/-- The specific tournament instance described in the problem. -/
def problem_tournament : ChessTournament := {
  initial_players := 15,  -- This is what we're proving
  matches_played := 84,
  dropped_players := 2,
  matches_per_dropped := 3
}

end NUMINAMATH_CALUDE_initial_players_count_l2969_296992


namespace NUMINAMATH_CALUDE_fraction_value_l2969_296932

theorem fraction_value : (5 * 7) / 10 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2969_296932


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2969_296944

theorem cubic_sum_theorem (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2969_296944


namespace NUMINAMATH_CALUDE_negation_equivalence_l2969_296957

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2969_296957


namespace NUMINAMATH_CALUDE_beaus_sons_age_l2969_296954

theorem beaus_sons_age (beau_age : ℕ) (sons_age : ℕ) : 
  beau_age = 42 →
  3 * (sons_age - 3) = beau_age - 3 →
  sons_age = 16 := by
sorry

end NUMINAMATH_CALUDE_beaus_sons_age_l2969_296954


namespace NUMINAMATH_CALUDE_savings_account_interest_rate_l2969_296977

theorem savings_account_interest_rate (initial_deposit : ℝ) (first_year_balance : ℝ) (total_increase_percentage : ℝ) :
  initial_deposit = 1000 →
  first_year_balance = 1100 →
  total_increase_percentage = 32 →
  let total_amount := initial_deposit * (1 + total_increase_percentage / 100)
  let second_year_increase := total_amount - first_year_balance
  let second_year_increase_percentage := (second_year_increase / first_year_balance) * 100
  second_year_increase_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_savings_account_interest_rate_l2969_296977


namespace NUMINAMATH_CALUDE_flower_shop_problem_l2969_296940

/-- Given information about flower purchases and sales, prove the cost price of the first batch and minimum selling price of the second batch -/
theorem flower_shop_problem (first_batch_cost second_batch_cost : ℝ) 
  (quantity_ratio : ℝ) (price_difference : ℝ) (min_total_profit : ℝ) 
  (first_batch_selling_price : ℝ) :
  first_batch_cost = 1000 →
  second_batch_cost = 2500 →
  quantity_ratio = 2 →
  price_difference = 0.5 →
  min_total_profit = 1500 →
  first_batch_selling_price = 3 →
  ∃ (first_batch_cost_price second_batch_min_selling_price : ℝ),
    first_batch_cost_price = 2 ∧
    second_batch_min_selling_price = 3.5 ∧
    (first_batch_cost / first_batch_cost_price) * quantity_ratio = 
      second_batch_cost / (first_batch_cost_price + price_difference) ∧
    (first_batch_cost / first_batch_cost_price) * 
      (first_batch_selling_price - first_batch_cost_price) +
    (second_batch_cost / (first_batch_cost_price + price_difference)) * 
      (second_batch_min_selling_price - (first_batch_cost_price + price_difference)) ≥ 
    min_total_profit := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_problem_l2969_296940


namespace NUMINAMATH_CALUDE_log_10_2_bounds_l2969_296955

theorem log_10_2_bounds :
  let log_10 (x : ℝ) := Real.log x / Real.log 10
  10^3 = 1000 ∧ 10^4 = 10000 ∧ 2^9 = 512 ∧ 2^14 = 16384 →
  2/7 < log_10 2 ∧ log_10 2 < 1/3 := by sorry

end NUMINAMATH_CALUDE_log_10_2_bounds_l2969_296955


namespace NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_9_l2969_296903

/-- A function that checks if a number has only odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- The smallest positive integer less than 10,000 with only odd digits that is a multiple of 9 -/
def smallestOddDigitMultipleOf9 : ℕ := 1117

theorem smallest_odd_digit_multiple_of_9 :
  smallestOddDigitMultipleOf9 < 10000 ∧
  hasOnlyOddDigits smallestOddDigitMultipleOf9 ∧
  smallestOddDigitMultipleOf9 % 9 = 0 ∧
  ∀ n : ℕ, n < 10000 → hasOnlyOddDigits n → n % 9 = 0 → smallestOddDigitMultipleOf9 ≤ n :=
by sorry

#eval smallestOddDigitMultipleOf9

end NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_9_l2969_296903


namespace NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l2969_296934

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y a : ℝ) : Prop := (x - 4)^2 + (y + a)^2 = 64

/-- Two circles have exactly two common tangents -/
def have_two_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : Prop := sorry

/-- Main theorem -/
theorem exists_a_with_two_common_tangents :
  ∃ a : ℕ, a > 0 ∧ have_two_common_tangents C₁ (C₂ · · a) :=
sorry

end NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l2969_296934


namespace NUMINAMATH_CALUDE_bottles_purchased_l2969_296917

/-- The number of large bottles purchased -/
def large_bottles : ℕ := 1380

/-- The cost of a large bottle in dollars -/
def large_bottle_cost : ℚ := 175/100

/-- The number of small bottles purchased -/
def small_bottles : ℕ := 690

/-- The cost of a small bottle in dollars -/
def small_bottle_cost : ℚ := 135/100

/-- The average price per bottle in dollars -/
def average_price : ℚ := 16163438256658595/10000000000000000

theorem bottles_purchased :
  (large_bottles * large_bottle_cost + small_bottles * small_bottle_cost) / 
  (large_bottles + small_bottles : ℚ) = average_price := by
  sorry

end NUMINAMATH_CALUDE_bottles_purchased_l2969_296917


namespace NUMINAMATH_CALUDE_only_valid_numbers_l2969_296963

/-- A six-digit number starting with 523 that is divisible by 7, 8, and 9 -/
def validNumber (n : ℕ) : Prop :=
  523000 ≤ n ∧ n < 524000 ∧ 7 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n

/-- The theorem stating that 523656 and 523152 are the only valid numbers -/
theorem only_valid_numbers :
  ∀ n : ℕ, validNumber n ↔ n = 523656 ∨ n = 523152 :=
by sorry

end NUMINAMATH_CALUDE_only_valid_numbers_l2969_296963


namespace NUMINAMATH_CALUDE_max_large_chips_l2969_296907

theorem max_large_chips (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 54 →
  ∃ (small large prime : ℕ), 
    is_prime prime ∧
    small + large = total ∧
    small = large + prime ∧
    ∀ (l : ℕ), (∃ (s p : ℕ), is_prime p ∧ s + l = total ∧ s = l + p) → l ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_max_large_chips_l2969_296907


namespace NUMINAMATH_CALUDE_last_score_entered_l2969_296935

def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

theorem last_score_entered (s : ℕ) :
  s ∈ scores →
  (s = 80 ↔ (List.sum scores - s) % 6 = 0 ∧
    ∀ t ∈ scores, t ≠ s → (List.sum scores - t) % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_last_score_entered_l2969_296935


namespace NUMINAMATH_CALUDE_triangle_expression_value_l2969_296983

theorem triangle_expression_value (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_expression_value_l2969_296983


namespace NUMINAMATH_CALUDE_expression_simplification_l2969_296953

theorem expression_simplification (m n : ℝ) (hm : m ≠ 0) :
  (m^(4/3) - 27 * m^(1/3) * n) / (m^(2/3) + 3 * (m*n)^(1/3) + 9 * n^(2/3)) / (1 - 3 * (n/m)^(1/3)) - m^(2/3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2969_296953


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2969_296945

/-- Given a car traveling for two hours with an average speed of 75 km/h
    and a speed of 90 km/h in the first hour, prove that the speed in
    the second hour must be 60 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (h_average : average_speed = 75)
  (h_first : first_hour_speed = 90)
  : (2 * average_speed - first_hour_speed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l2969_296945


namespace NUMINAMATH_CALUDE_y_value_l2969_296999

theorem y_value (x z y : ℝ) (h1 : x = 2 * z) (h2 : y = 3 * z - 1) (h3 : x = 40) : y = 59 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2969_296999


namespace NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l2969_296973

/-- The cost of jam used in Elmo's sandwiches --/
theorem elmo_sandwich_jam_cost :
  ∀ (N B J H : ℕ),
    N > 1 →
    B > 0 →
    J > 0 →
    H > 0 →
    N * (3 * B + 6 * J + 2 * H) = 342 →
    N * J * 6 = 270 := by
  sorry

end NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l2969_296973


namespace NUMINAMATH_CALUDE_tim_website_earnings_l2969_296968

/-- Calculates Tim's earnings from his website for a week -/
def website_earnings (
  daily_visitors : ℕ)  -- Number of visitors per day for the first 6 days
  (days : ℕ)            -- Number of days with constant visitors
  (last_day_multiplier : ℕ)  -- Multiplier for visitors on the last day
  (earnings_per_visit : ℚ)  -- Earnings per visit in dollars
  : ℚ :=
  let first_days_visitors := daily_visitors * days
  let last_day_visitors := first_days_visitors * last_day_multiplier
  let total_visitors := first_days_visitors + last_day_visitors
  (total_visitors : ℚ) * earnings_per_visit

/-- Theorem stating Tim's earnings for the week -/
theorem tim_website_earnings :
  website_earnings 100 6 2 (1/100) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tim_website_earnings_l2969_296968


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2969_296951

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem triangle_perimeter (a b c : ℕ) :
  a = 2 → b = 5 → is_odd c → a + b > c → b + c > a → c + a > b →
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2969_296951


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2969_296943

theorem angle_sum_around_point (y : ℝ) (h : y > 0) : 
  6 * y + 3 * y + 4 * y + 2 * y + y + 5 * y = 360 → y = 120 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2969_296943


namespace NUMINAMATH_CALUDE_tan_sum_ratio_l2969_296988

theorem tan_sum_ratio : 
  (Real.tan (20 * π / 180) + Real.tan (40 * π / 180) + Real.tan (120 * π / 180)) / 
  (Real.tan (20 * π / 180) * Real.tan (40 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_ratio_l2969_296988


namespace NUMINAMATH_CALUDE_min_value_when_a_is_1_range_of_a_for_bounded_f_l2969_296979

/-- The function f(x) defined as |2x-a| - |x+3| --/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| - |x + 3|

/-- Theorem stating the minimum value of f(x) when a = 1 --/
theorem min_value_when_a_is_1 :
  ∃ (m : ℝ), m = -7/2 ∧ ∀ (x : ℝ), f 1 x ≥ m := by sorry

/-- Theorem stating the range of a for which f(x) ≤ 4 when x ∈ [0,3] --/
theorem range_of_a_for_bounded_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 3 → f a x ≤ 4) ↔ a ∈ Set.Icc (-4) 7 := by sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_1_range_of_a_for_bounded_f_l2969_296979


namespace NUMINAMATH_CALUDE_arrangement_of_digits_and_blanks_l2969_296978

theorem arrangement_of_digits_and_blanks : 
  let n : ℕ := 6  -- total number of boxes
  let k : ℕ := 4  -- number of distinct digits
  let b : ℕ := 2  -- number of blank spaces
  n! / b! = 360 := by
sorry

end NUMINAMATH_CALUDE_arrangement_of_digits_and_blanks_l2969_296978


namespace NUMINAMATH_CALUDE_hayden_earnings_l2969_296989

/-- Calculates the total earnings for a limo driver based on given parameters. -/
def limo_driver_earnings (hourly_wage : ℕ) (hours_worked : ℕ) (ride_bonus : ℕ) (rides_given : ℕ) 
  (review_bonus : ℕ) (positive_reviews : ℕ) (gas_price : ℕ) (gas_used : ℕ) : ℕ :=
  hourly_wage * hours_worked + ride_bonus * rides_given + review_bonus * positive_reviews + gas_price * gas_used

/-- Proves that Hayden's earnings for the day equal $226 given the specified conditions. -/
theorem hayden_earnings : 
  limo_driver_earnings 15 8 5 3 20 2 3 17 = 226 := by
  sorry

end NUMINAMATH_CALUDE_hayden_earnings_l2969_296989


namespace NUMINAMATH_CALUDE_thirty_switch_network_connections_l2969_296941

/-- A network of switches with direct connections between them. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- The total number of connections in a switch network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  network.num_switches * network.connections_per_switch / 2

/-- Theorem: In a network of 30 switches, where each switch is directly
    connected to exactly 4 other switches, the total number of connections is 60. -/
theorem thirty_switch_network_connections :
  let network : SwitchNetwork := { num_switches := 30, connections_per_switch := 4 }
  total_connections network = 60 := by
  sorry


end NUMINAMATH_CALUDE_thirty_switch_network_connections_l2969_296941


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l2969_296924

/-- Represents the conversion rate between daps and dops -/
def daps_to_dops : ℚ := 5 / 4

/-- Represents the conversion rate between dops and dips -/
def dops_to_dips : ℚ := 3 / 9

/-- The number of dips we want to convert -/
def target_dips : ℚ := 54

theorem daps_equivalent_to_dips :
  (daps_to_dops * (1 / dops_to_dips) * target_dips : ℚ) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l2969_296924


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2969_296995

def last_two_digits (n : ℕ) : ℕ × ℕ :=
  ((n / 10) % 10, n % 10)

theorem last_two_digits_product (n : ℕ) 
  (h1 : n % 4 = 0) 
  (h2 : (last_two_digits n).1 + (last_two_digits n).2 = 14) : 
  (last_two_digits n).1 * (last_two_digits n).2 = 48 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2969_296995


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2969_296966

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + b^2 = 64 →
  (1/2) * a * b = 10 →
  c^2 + d^2 = (5*8)^2 →
  (1/2) * c * d = 250 →
  c + d = 51 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2969_296966


namespace NUMINAMATH_CALUDE_max_visible_cubes_l2969_296909

/-- Represents a transparent cube made of unit cubes --/
structure TransparentCube where
  size : Nat
  deriving Repr

/-- Calculates the number of visible unit cubes from a single point --/
def visibleUnitCubes (cube : TransparentCube) : Nat :=
  let fullFace := cube.size * cube.size
  let surfaceFaces := 2 * (cube.size * cube.size - (cube.size - 2) * (cube.size - 2))
  let sharedEdges := 3 * cube.size
  fullFace + surfaceFaces - sharedEdges + 1

/-- Theorem stating that the maximum number of visible unit cubes is 181 for a 12x12x12 cube --/
theorem max_visible_cubes (cube : TransparentCube) (h : cube.size = 12) :
  visibleUnitCubes cube = 181 := by
  sorry

#eval visibleUnitCubes { size := 12 }

end NUMINAMATH_CALUDE_max_visible_cubes_l2969_296909
