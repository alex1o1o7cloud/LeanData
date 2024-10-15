import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_product_l387_38745

/-- Given real numbers x, y, and z, if -1, x, y, z, -3 form a geometric sequence,
    then the product of x and z equals 3. -/
theorem geometric_sequence_product (x y z : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) →
  x * z = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l387_38745


namespace NUMINAMATH_CALUDE_next_repeated_year_correct_l387_38770

/-- A year consists of a repeated two-digit number if it can be written as ABAB where A and B are digits -/
def is_repeated_two_digit (year : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ year = a * 1000 + b * 100 + a * 10 + b

/-- The next year after 2020 with a repeated two-digit number -/
def next_repeated_year : ℕ := 2121

theorem next_repeated_year_correct :
  (next_repeated_year > 2020) ∧ 
  (is_repeated_two_digit next_repeated_year) ∧
  (∀ y : ℕ, 2020 < y ∧ y < next_repeated_year → ¬(is_repeated_two_digit y)) ∧
  (next_repeated_year - 2020 = 101) :=
sorry

end NUMINAMATH_CALUDE_next_repeated_year_correct_l387_38770


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_inequality_holds_l387_38720

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} :=
by sorry

-- Part 2
theorem range_of_a_when_inequality_holds :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_inequality_holds_l387_38720


namespace NUMINAMATH_CALUDE_triangle_inequality_l387_38722

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a * b + 1) / (a^2 + c * a + 1) + 
  (b * c + 1) / (b^2 + a * b + 1) + 
  (c * a + 1) / (c^2 + b * c + 1) > 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l387_38722


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l387_38712

theorem average_of_three_numbers (x : ℝ) : (15 + 19 + x) / 3 = 20 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l387_38712


namespace NUMINAMATH_CALUDE_lucas_sticker_redistribution_l387_38719

theorem lucas_sticker_redistribution
  (n : ℚ)  -- Noah's initial number of stickers
  (h1 : n > 0)  -- Ensure n is positive
  (emma : ℚ)  -- Emma's initial number of stickers
  (h2 : emma = 3 * n)  -- Emma has 3 times as many stickers as Noah
  (lucas : ℚ)  -- Lucas's initial number of stickers
  (h3 : lucas = 4 * emma)  -- Lucas has 4 times as many stickers as Emma
  : (lucas - (lucas + emma + n) / 3) / lucas = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_lucas_sticker_redistribution_l387_38719


namespace NUMINAMATH_CALUDE_population_growth_rate_l387_38759

/-- Proves that given an initial population of 1200, a 25% increase in the first year,
    and a final population of 1950 after two years, the percentage increase in the second year is 30%. -/
theorem population_growth_rate (initial_population : ℕ) (first_year_increase : ℚ) 
  (final_population : ℕ) (second_year_increase : ℚ) : 
  initial_population = 1200 →
  first_year_increase = 25 / 100 →
  final_population = 1950 →
  (initial_population * (1 + first_year_increase) * (1 + second_year_increase) = final_population) →
  second_year_increase = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l387_38759


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l387_38786

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) = 1680 → (n - 1) + n + (n + 1) + (n + 2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l387_38786


namespace NUMINAMATH_CALUDE_complex_parts_of_3i_times_1_plus_i_l387_38740

theorem complex_parts_of_3i_times_1_plus_i :
  let z : ℂ := 3 * Complex.I * (1 + Complex.I)
  (z.re = -3) ∧ (z.im = 3) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_3i_times_1_plus_i_l387_38740


namespace NUMINAMATH_CALUDE_tesseract_parallel_edges_l387_38726

/-- A tesseract is a four-dimensional hypercube -/
structure Tesseract where
  dim : Nat
  edges : Nat

/-- The number of pairs of parallel edges in a tesseract -/
def parallel_edge_pairs (t : Tesseract) : Nat :=
  sorry

/-- Theorem: A tesseract with 32 edges has 36 pairs of parallel edges -/
theorem tesseract_parallel_edges (t : Tesseract) (h1 : t.dim = 4) (h2 : t.edges = 32) :
  parallel_edge_pairs t = 36 := by
  sorry

end NUMINAMATH_CALUDE_tesseract_parallel_edges_l387_38726


namespace NUMINAMATH_CALUDE_series_sum_l387_38795

/-- The positive real solution to x³ + (1/4)x - 1 = 0 -/
noncomputable def s : ℝ := sorry

/-- The infinite series s³ + 2s⁷ + 3s¹¹ + 4s¹⁵ + ... -/
noncomputable def T : ℝ := sorry

/-- s is a solution to the equation x³ + (1/4)x - 1 = 0 -/
axiom s_def : s^3 + (1/4) * s - 1 = 0

/-- s is positive -/
axiom s_pos : s > 0

/-- T is equal to the infinite series s³ + 2s⁷ + 3s¹¹ + 4s¹⁵ + ... -/
axiom T_def : T = s^3 + 2*s^7 + 3*s^11 + 4*s^15 + sorry

theorem series_sum : T = 16 * s := by sorry

end NUMINAMATH_CALUDE_series_sum_l387_38795


namespace NUMINAMATH_CALUDE_hiker_first_pack_weight_hiker_first_pack_weight_proof_l387_38714

/-- Calculates the weight of the first pack for a hiker given specific conditions --/
theorem hiker_first_pack_weight
  (supplies_per_mile : Real)
  (hiking_rate : Real)
  (hours_per_day : Real)
  (days : Real)
  (resupply_ratio : Real)
  (h1 : supplies_per_mile = 0.5)
  (h2 : hiking_rate = 2.5)
  (h3 : hours_per_day = 8)
  (h4 : days = 5)
  (h5 : resupply_ratio = 0.25)
  : Real :=
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := supplies_per_mile * total_distance
  let resupply_weight := resupply_ratio * total_supplies
  let first_pack_weight := total_supplies - resupply_weight
  37.5

theorem hiker_first_pack_weight_proof : hiker_first_pack_weight 0.5 2.5 8 5 0.25 rfl rfl rfl rfl rfl = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_first_pack_weight_hiker_first_pack_weight_proof_l387_38714


namespace NUMINAMATH_CALUDE_equation_3y_plus_1_eq_6_is_linear_l387_38753

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 3y + 1 = 6 is a linear equation -/
theorem equation_3y_plus_1_eq_6_is_linear :
  is_linear_equation (λ y => 3 * y + 1) :=
by
  sorry

#check equation_3y_plus_1_eq_6_is_linear

end NUMINAMATH_CALUDE_equation_3y_plus_1_eq_6_is_linear_l387_38753


namespace NUMINAMATH_CALUDE_rational_fraction_implication_l387_38793

theorem rational_fraction_implication (x : ℝ) :
  (∃ a : ℚ, (x / (x^2 + x + 1) : ℝ) = a) →
  (∃ b : ℚ, (x^2 / (x^4 + x^2 + 1) : ℝ) = b) :=
by sorry

end NUMINAMATH_CALUDE_rational_fraction_implication_l387_38793


namespace NUMINAMATH_CALUDE_sampling_method_is_systematic_l387_38716

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory's production line -/
structure ProductionLine where
  product : Type
  conveyorBelt : Bool
  inspectionInterval : ℕ
  fixedPosition : Bool

/-- Determines the sampling method based on the production line characteristics -/
def determineSamplingMethod (line : ProductionLine) : SamplingMethod :=
  if line.conveyorBelt && line.inspectionInterval > 0 && line.fixedPosition then
    SamplingMethod.Systematic
  else
    SamplingMethod.Other

/-- Theorem: The sampling method for the given production line is systematic sampling -/
theorem sampling_method_is_systematic (line : ProductionLine) 
  (h1 : line.conveyorBelt = true)
  (h2 : line.inspectionInterval = 10)
  (h3 : line.fixedPosition = true) :
  determineSamplingMethod line = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_sampling_method_is_systematic_l387_38716


namespace NUMINAMATH_CALUDE_relation_between_x_and_y_l387_38771

theorem relation_between_x_and_y (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_relation_between_x_and_y_l387_38771


namespace NUMINAMATH_CALUDE_equity_investment_l387_38730

def total_investment : ℝ := 250000

theorem equity_investment (debt : ℝ) 
  (h1 : debt + 3 * debt = total_investment) : 
  3 * debt = 187500 := by
  sorry

#check equity_investment

end NUMINAMATH_CALUDE_equity_investment_l387_38730


namespace NUMINAMATH_CALUDE_purely_imaginary_product_l387_38783

theorem purely_imaginary_product (x : ℝ) : 
  (Complex.I * (x^4 + 2*x^3 + x^2 + 2*x) = (x + Complex.I) * ((x^2 + 1) + Complex.I) * ((x + 2) + Complex.I)) ↔ 
  (x = 0 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_product_l387_38783


namespace NUMINAMATH_CALUDE_lattice_points_count_l387_38763

/-- A triangular lattice -/
structure TriangularLattice where
  /-- The distance between adjacent points is 1 -/
  adjacent_distance : ℝ
  adjacent_distance_eq : adjacent_distance = 1

/-- An equilateral triangle on a triangular lattice -/
structure EquilateralTriangle (L : ℝ) where
  /-- The side length of the triangle -/
  side_length : ℝ
  side_length_eq : side_length = L
  /-- The triangle has no lattice points on its sides -/
  no_lattice_points_on_sides : Prop

/-- The number of lattice points inside an equilateral triangle -/
def lattice_points_inside (L : ℝ) (triangle : EquilateralTriangle L) : ℕ :=
  sorry

theorem lattice_points_count (L : ℝ) (triangle : EquilateralTriangle L) :
  lattice_points_inside L triangle = (L^2 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_lattice_points_count_l387_38763


namespace NUMINAMATH_CALUDE_exists_valid_partition_l387_38747

/-- A directed graph where each vertex has outdegree 2 -/
structure Graph (V : Type*) :=
  (edges : V → Finset V)
  (outdegree_two : ∀ v : V, (edges v).card = 2)

/-- A partition of vertices into three sets -/
def Partition (V : Type*) := V → Fin 3

/-- The main theorem statement -/
theorem exists_valid_partition {V : Type*} [Fintype V] (G : Graph V) :
  ∃ (p : Partition V), ∀ (v : V),
    ∃ (w : V), w ∈ G.edges v ∧ p w ≠ p v :=
sorry

end NUMINAMATH_CALUDE_exists_valid_partition_l387_38747


namespace NUMINAMATH_CALUDE_normal_dist_two_std_dev_below_mean_l387_38782

/-- For a normal distribution with mean μ and standard deviation σ,
    the value that is exactly 2 standard deviations less than the mean
    is equal to μ - 2σ. -/
theorem normal_dist_two_std_dev_below_mean (μ σ : ℝ) :
  let value := μ - 2 * σ
  μ = 16.5 → σ = 1.5 → value = 13.5 := by sorry

end NUMINAMATH_CALUDE_normal_dist_two_std_dev_below_mean_l387_38782


namespace NUMINAMATH_CALUDE_school_sampling_theorem_l387_38751

/-- Represents the types of schools --/
inductive SchoolType
  | Primary
  | Middle
  | University

/-- Represents the count of each school type --/
structure SchoolCounts where
  primary : Nat
  middle : Nat
  university : Nat

/-- Represents the result of stratified sampling --/
structure SamplingResult where
  primary : Nat
  middle : Nat
  university : Nat

/-- Calculates the stratified sampling result --/
def stratifiedSample (counts : SchoolCounts) (totalSample : Nat) : SamplingResult :=
  { primary := counts.primary * totalSample / (counts.primary + counts.middle + counts.university),
    middle := counts.middle * totalSample / (counts.primary + counts.middle + counts.university),
    university := counts.university * totalSample / (counts.primary + counts.middle + counts.university) }

/-- Calculates the probability of selecting two primary schools --/
def probabilityTwoPrimary (sample : SamplingResult) : Rat :=
  (sample.primary * (sample.primary - 1)) / (2 * (sample.primary + sample.middle + sample.university) * (sample.primary + sample.middle + sample.university - 1))

theorem school_sampling_theorem (counts : SchoolCounts) (h : counts = { primary := 21, middle := 14, university := 7 }) :
  let sample := stratifiedSample counts 6
  sample = { primary := 3, middle := 2, university := 1 } ∧
  probabilityTwoPrimary sample = 1/5 := by
  sorry

#check school_sampling_theorem

end NUMINAMATH_CALUDE_school_sampling_theorem_l387_38751


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l387_38738

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 →
  quarters = 22 :=
by sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l387_38738


namespace NUMINAMATH_CALUDE_product_selection_probability_l387_38785

/-- Given a set of products with some being first-class, this function calculates
    the probability that one of two randomly selected products is not first-class,
    given that one of them is first-class. -/
def conditional_probability (total : ℕ) (first_class : ℕ) : ℚ :=
  let not_first_class := total - first_class
  let total_combinations := (total.choose 2 : ℚ)
  let one_not_first_class := (first_class * not_first_class : ℚ)
  one_not_first_class / total_combinations

/-- The theorem states that for 8 total products with 6 being first-class,
    the conditional probability of selecting one non-first-class product
    given that one first-class product is selected is 12/13. -/
theorem product_selection_probability :
  conditional_probability 8 6 = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_product_selection_probability_l387_38785


namespace NUMINAMATH_CALUDE_determinant_cubic_roots_l387_38710

theorem determinant_cubic_roots (p q k : ℝ) (a b c : ℝ) : 
  a^3 + p*a + q = 0 → 
  b^3 + p*b + q = 0 → 
  c^3 + p*c + q = 0 → 
  Matrix.det !![k + a, 1, 1; 1, k + b, 1; 1, 1, k + c] = k^3 + k*p - q := by
  sorry

end NUMINAMATH_CALUDE_determinant_cubic_roots_l387_38710


namespace NUMINAMATH_CALUDE_age_problem_l387_38773

/-- Proves that given the age conditions, Mária is 36 2/3 years old and Anna is 7 1/3 years old -/
theorem age_problem (x y : ℚ) : 
  x + y = 44 → 
  x = 2 * (y - (-1/2 * x + 3/2 * (2/3 * y))) → 
  x = 110/3 ∧ y = 22/3 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l387_38773


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l387_38701

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (u v w : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hu : M.mulVec u = ![(-3), 4])
  (hv : M.mulVec v = ![2, (-7)])
  (hw : M.mulVec w = ![9, 0]) :
  M.mulVec (3 • u - 4 • v + 2 • w) = ![1, 40] := by
sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l387_38701


namespace NUMINAMATH_CALUDE_calculation_proof_l387_38792

theorem calculation_proof : (0.08 / 0.002) * 0.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l387_38792


namespace NUMINAMATH_CALUDE_oplus_problem_l387_38724

def oplus (a b : ℚ) : ℚ := a^3 / b

theorem oplus_problem : 
  let x := oplus (oplus 2 4) 6
  let y := oplus 2 (oplus 4 6)
  x - y = 7/12 := by sorry

end NUMINAMATH_CALUDE_oplus_problem_l387_38724


namespace NUMINAMATH_CALUDE_r_value_when_m_is_3_l387_38706

theorem r_value_when_m_is_3 (m : ℕ) (t : ℕ) (r : ℕ) : 
  m = 3 → 
  t = 3^m + 2 → 
  r = 4^t - 2*t → 
  r = 4^29 - 58 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_m_is_3_l387_38706


namespace NUMINAMATH_CALUDE_greatest_common_measure_l387_38762

theorem greatest_common_measure (a b c : ℕ) (ha : a = 700) (hb : b = 385) (hc : c = 1295) :
  Nat.gcd a (Nat.gcd b c) = 35 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l387_38762


namespace NUMINAMATH_CALUDE_max_integer_solutions_l387_38703

/-- A quadratic function f(x) = ax^2 + bx + c where a > 100 -/
def QuadraticFunction (a b c : ℝ) (h : a > 100) := fun (x : ℤ) => a * x^2 + b * x + c

/-- The maximum number of integer solutions for |f(x)| ≤ 50 is at most 2 -/
theorem max_integer_solutions (a b c : ℝ) (h : a > 100) :
  ∃ (n : ℕ), n ≤ 2 ∧ 
  ∀ (S : Finset ℤ), (∀ x ∈ S, |QuadraticFunction a b c h x| ≤ 50) → S.card ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_integer_solutions_l387_38703


namespace NUMINAMATH_CALUDE_east_northwest_angle_l387_38796

/-- A circle with ten equally spaced rays -/
structure TenRayCircle where
  rays : Fin 10 → ℝ
  north_ray : rays 0 = 0
  equally_spaced : ∀ i : Fin 10, rays i = (i : ℝ) * 36

/-- The angle between two rays in a TenRayCircle -/
def angle_between (c : TenRayCircle) (i j : Fin 10) : ℝ :=
  ((j - i : ℤ) % 10 : ℤ) * 36

theorem east_northwest_angle (c : TenRayCircle) :
  min (angle_between c 3 8) (angle_between c 8 3) = 144 :=
sorry

end NUMINAMATH_CALUDE_east_northwest_angle_l387_38796


namespace NUMINAMATH_CALUDE_f_neg_l387_38777

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 + 2*x - 3

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + 2*x + 3 := by sorry

end NUMINAMATH_CALUDE_f_neg_l387_38777


namespace NUMINAMATH_CALUDE_investment_problem_l387_38711

theorem investment_problem (initial_investment : ℝ) (growth_rate_year1 : ℝ) (growth_rate_year2 : ℝ) (final_value : ℝ) (amount_added : ℝ) : 
  initial_investment = 80 →
  growth_rate_year1 = 0.15 →
  growth_rate_year2 = 0.10 →
  final_value = 132 →
  amount_added = 28 →
  final_value = (initial_investment * (1 + growth_rate_year1) + amount_added) * (1 + growth_rate_year2) :=
by sorry

#check investment_problem

end NUMINAMATH_CALUDE_investment_problem_l387_38711


namespace NUMINAMATH_CALUDE_line_disjoint_from_circle_l387_38729

/-- Given a point M(a,b) inside the unit circle, prove that the line ax + by = 1 is disjoint from the circle -/
theorem line_disjoint_from_circle (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_line_disjoint_from_circle_l387_38729


namespace NUMINAMATH_CALUDE_probability_above_parabola_l387_38727

def is_single_digit_positive (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def point_above_parabola (a b : ℕ) : Prop :=
  is_single_digit_positive a ∧ is_single_digit_positive b ∧ b > a * a + b * a

def total_combinations : ℕ := 81

def valid_combinations : ℕ := 7

theorem probability_above_parabola :
  (valid_combinations : ℚ) / total_combinations = 7 / 81 := by sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l387_38727


namespace NUMINAMATH_CALUDE_james_toy_cost_l387_38755

/-- Calculates the cost per toy given the total number of toys, percentage sold, selling price, and profit. -/
def cost_per_toy (total_toys : ℕ) (percent_sold : ℚ) (selling_price : ℚ) (profit : ℚ) : ℚ :=
  let sold_toys : ℚ := total_toys * percent_sold
  let revenue : ℚ := sold_toys * selling_price
  let cost : ℚ := revenue - profit
  cost / sold_toys

/-- Proves that the cost per toy is $25 given the problem conditions. -/
theorem james_toy_cost :
  cost_per_toy 200 (80 / 100) 30 800 = 25 := by
  sorry

end NUMINAMATH_CALUDE_james_toy_cost_l387_38755


namespace NUMINAMATH_CALUDE_recurring_decimal_product_l387_38789

theorem recurring_decimal_product : 
  ∃ (s : ℚ), (s = 456 / 999) ∧ (7 * s = 355 / 111) := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_product_l387_38789


namespace NUMINAMATH_CALUDE_price_increase_x_l387_38788

/-- The annual price increase of commodity x -/
def annual_increase_x : ℚ := 30 / 100

/-- The annual price increase of commodity y -/
def annual_increase_y : ℚ := 20 / 100

/-- The price of commodity x in 2001 -/
def price_x_2001 : ℚ := 420 / 100

/-- The price of commodity y in 2001 -/
def price_y_2001 : ℚ := 440 / 100

/-- The number of years between 2001 and 2012 -/
def years : ℕ := 11

/-- The difference in price between commodities x and y in 2012 -/
def price_difference_2012 : ℚ := 90 / 100

theorem price_increase_x : 
  annual_increase_x * years + price_x_2001 = 
  annual_increase_y * years + price_y_2001 + price_difference_2012 :=
sorry

end NUMINAMATH_CALUDE_price_increase_x_l387_38788


namespace NUMINAMATH_CALUDE_unique_solution_l387_38736

/-- Represents a 3-digit number AAA where A is a single digit -/
def three_digit_AAA (A : ℕ) : ℕ := 100 * A + 10 * A + A

/-- Represents a 6-digit number AAABBB where A and B are single digits -/
def six_digit_AAABBB (A B : ℕ) : ℕ := 1000 * (three_digit_AAA A) + 100 * B + 10 * B + B

/-- Proves that the only solution to AAA × AAA + AAA = AAABBB is A = 9 and B = 0 -/
theorem unique_solution : 
  ∀ A B : ℕ, 
  A ≠ 0 → 
  A < 10 → 
  B < 10 → 
  (three_digit_AAA A) * (three_digit_AAA A) + (three_digit_AAA A) = six_digit_AAABBB A B → 
  A = 9 ∧ B = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l387_38736


namespace NUMINAMATH_CALUDE_odd_minus_odd_is_even_l387_38739

theorem odd_minus_odd_is_even (a b : ℤ) (ha : Odd a) (hb : Odd b) : Even (a - b) := by
  sorry

end NUMINAMATH_CALUDE_odd_minus_odd_is_even_l387_38739


namespace NUMINAMATH_CALUDE_semester_days_l387_38767

/-- Calculates the number of days given daily distance and total distance -/
def calculate_days (daily_distance : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / daily_distance

/-- Theorem stating that given the specific conditions, the number of days is 160 -/
theorem semester_days : calculate_days 10 1600 = 160 := by
  sorry

end NUMINAMATH_CALUDE_semester_days_l387_38767


namespace NUMINAMATH_CALUDE_divisor_problem_l387_38717

theorem divisor_problem (dividend : ℤ) (quotient : ℤ) (remainder : ℤ) (divisor : ℤ) : 
  dividend = 151 ∧ quotient = 11 ∧ remainder = -4 →
  divisor = 14 ∧ dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l387_38717


namespace NUMINAMATH_CALUDE_two_numbers_problem_l387_38735

theorem two_numbers_problem :
  ∃ (x y : ℤ),
    x + y = 44 ∧
    y < 0 ∧
    (x - y) * 100 = y * y ∧
    x = 264 ∧
    y = -220 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l387_38735


namespace NUMINAMATH_CALUDE_average_speed_calculation_l387_38705

/-- Calculates the average speed given distances and speeds for multiple segments of a ride -/
theorem average_speed_calculation 
  (d₁ d₂ d₃ : ℝ) 
  (v₁ v₂ v₃ : ℝ) 
  (h₁ : d₁ = 50)
  (h₂ : d₂ = 20)
  (h₃ : d₃ = 10)
  (h₄ : v₁ = 12)
  (h₅ : v₂ = 40)
  (h₆ : v₃ = 20) :
  (d₁ + d₂ + d₃) / ((d₁ / v₁) + (d₂ / v₂) + (d₃ / v₃)) = 480 / 31 := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l387_38705


namespace NUMINAMATH_CALUDE_abs_frac_inequality_l387_38754

theorem abs_frac_inequality (x : ℝ) : 
  |((x - 3) / x)| > ((x - 3) / x) ↔ 0 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_abs_frac_inequality_l387_38754


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_630_l387_38707

theorem sin_n_equals_cos_630 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (630 * π / 180) ↔ n = 0 ∨ n = 180 ∨ n = -180) :=
by sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_630_l387_38707


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l387_38780

def A : Set ℤ := {-2, 0, 1}
def B : Set ℤ := {x | x^2 > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l387_38780


namespace NUMINAMATH_CALUDE_brick_surface_area_is_54_l387_38766

/-- Represents the surface areas of a brick -/
structure BrickAreas where
  front : ℝ
  side : ℝ
  top : ℝ

/-- The surface areas of the three arrangements -/
def arrangement1 (b : BrickAreas) : ℝ := 4 * b.front + 4 * b.side + 2 * b.top
def arrangement2 (b : BrickAreas) : ℝ := 4 * b.front + 2 * b.side + 4 * b.top
def arrangement3 (b : BrickAreas) : ℝ := 2 * b.front + 4 * b.side + 4 * b.top

/-- The surface area of a single brick -/
def brickSurfaceArea (b : BrickAreas) : ℝ := 2 * (b.front + b.side + b.top)

theorem brick_surface_area_is_54 (b : BrickAreas) 
  (h1 : arrangement1 b = 72)
  (h2 : arrangement2 b = 96)
  (h3 : arrangement3 b = 102) : 
  brickSurfaceArea b = 54 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_is_54_l387_38766


namespace NUMINAMATH_CALUDE_cubic_root_h_value_l387_38764

theorem cubic_root_h_value (h : ℝ) : (3 : ℝ)^3 + h * 3 + 14 = 0 → h = -41/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_h_value_l387_38764


namespace NUMINAMATH_CALUDE_chris_breath_holding_start_l387_38743

def breath_holding_progression (start : ℕ) (days : ℕ) : ℕ :=
  start + 10 * (days - 1)

theorem chris_breath_holding_start :
  ∃ (start : ℕ),
    breath_holding_progression start 2 = 20 ∧
    breath_holding_progression start 6 = 90 ∧
    start = 10 := by
  sorry

end NUMINAMATH_CALUDE_chris_breath_holding_start_l387_38743


namespace NUMINAMATH_CALUDE_license_plate_difference_l387_38798

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible Sunshine license plates -/
def sunshine_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Prairie license plates -/
def prairie_plates : ℕ := num_letters^2 * num_digits^4

/-- The difference in the number of possible license plates between Sunshine and Prairie -/
def plate_difference : ℕ := sunshine_plates - prairie_plates

theorem license_plate_difference :
  plate_difference = 10816000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l387_38798


namespace NUMINAMATH_CALUDE_lens_discount_l387_38760

def old_camera_price : ℝ := 4000
def lens_original_price : ℝ := 400
def total_paid : ℝ := 5400
def price_increase_percentage : ℝ := 0.30

theorem lens_discount (new_camera_price : ℝ) (lens_paid : ℝ) 
  (h1 : new_camera_price = old_camera_price * (1 + price_increase_percentage))
  (h2 : total_paid = new_camera_price + lens_paid) :
  lens_original_price - lens_paid = 200 := by
sorry

end NUMINAMATH_CALUDE_lens_discount_l387_38760


namespace NUMINAMATH_CALUDE_intersection_sufficient_not_necessary_for_union_l387_38709

-- Define the sets M and P
def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x < 4}

-- State the theorem
theorem intersection_sufficient_not_necessary_for_union :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) := by
  sorry

end NUMINAMATH_CALUDE_intersection_sufficient_not_necessary_for_union_l387_38709


namespace NUMINAMATH_CALUDE_rearrangeable_shapes_exist_l387_38746

/-- Represents a shape that can be divided and rearranged -/
structure Divisible2DShape where
  area : ℝ
  can_form_square : Bool
  can_form_triangle : Bool

/-- Represents a set of shapes that can be rearranged -/
def ShapeSet := List Divisible2DShape

/-- Function to check if a shape set can form a square -/
def can_form_square (shapes : ShapeSet) : Bool :=
  shapes.any (·.can_form_square)

/-- Function to check if a shape set can form a triangle -/
def can_form_triangle (shapes : ShapeSet) : Bool :=
  shapes.any (·.can_form_triangle)

/-- The main theorem statement -/
theorem rearrangeable_shapes_exist (a : ℝ) (h : a > 0) :
  ∃ (shapes : ShapeSet),
    -- The total area of shapes is greater than the initial square
    (shapes.map (·.area)).sum > a^2 ∧
    -- The shape set can form two different squares
    can_form_square shapes ∧
    -- The shape set can form two different triangles
    can_form_triangle shapes :=
  sorry


end NUMINAMATH_CALUDE_rearrangeable_shapes_exist_l387_38746


namespace NUMINAMATH_CALUDE_equal_ratios_sum_l387_38776

theorem equal_ratios_sum (P Q : ℚ) :
  (4 : ℚ) / 9 = P / 63 ∧ (4 : ℚ) / 9 = 108 / Q → P + Q = 271 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_l387_38776


namespace NUMINAMATH_CALUDE_g_properties_l387_38700

-- Define the function g
def g : ℝ → ℝ := fun x ↦ -x

-- Theorem stating that g is an odd function and monotonically decreasing
theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ 
  (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry


end NUMINAMATH_CALUDE_g_properties_l387_38700


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_arithmetic_l387_38749

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_fifth_sum_arithmetic (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 3 = 13 →
  a 6 = 33 →
  a 7 = 38 →
  a 4 + a 5 = 41 := by
sorry

end NUMINAMATH_CALUDE_fourth_fifth_sum_arithmetic_l387_38749


namespace NUMINAMATH_CALUDE_min_exponent_sum_l387_38761

theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ) 
  (h1 : A = 2^α * 3^β * 5^γ)
  (h2 : ∃ (k : ℕ), A / 2 = k^2)
  (h3 : ∃ (m : ℕ), A / 3 = m^3)
  (h4 : ∃ (n : ℕ), A / 5 = n^5) :
  α + β + γ ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_min_exponent_sum_l387_38761


namespace NUMINAMATH_CALUDE_paper_strip_sequence_l387_38708

theorem paper_strip_sequence : ∃ (a : Fin 10 → ℝ), 
  a 0 = 9 ∧ 
  a 8 = 5 ∧ 
  ∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_sequence_l387_38708


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l387_38799

theorem trigonometric_expression_evaluation :
  (Real.cos (40 * π / 180) + Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) /
  (Real.sin (70 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l387_38799


namespace NUMINAMATH_CALUDE_prob_at_least_one_defective_l387_38787

/-- The probability of selecting at least one defective bulb when randomly choosing two bulbs from a box containing 22 bulbs, of which 4 are defective. -/
theorem prob_at_least_one_defective (total : Nat) (defective : Nat) (h1 : total = 22) (h2 : defective = 4) :
  (1 : ℚ) - (total - defective) * (total - defective - 1) / (total * (total - 1)) = 26 / 77 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_defective_l387_38787


namespace NUMINAMATH_CALUDE_line_slope_l387_38790

/-- The slope of a line given by the equation (x/2) + (y/3) = 1 is -3/2 -/
theorem line_slope : 
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ (x / 2 + y / 3 = 1)
  ∃ m b : ℝ, (∀ x y, line_eq x y ↔ y = m * x + b) ∧ m = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l387_38790


namespace NUMINAMATH_CALUDE_inverse_function_symmetry_l387_38794

def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

theorem inverse_function_symmetry 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h₁ : Function.Bijective f) 
  (h₂ : Function.RightInverse g f) 
  (h₃ : Function.LeftInverse g f)
  (h₄ : symmetric_about f (0, 1)) : 
  ∀ a : ℝ, g a + g (2 - a) = 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_symmetry_l387_38794


namespace NUMINAMATH_CALUDE_total_length_of_sticks_l387_38750

/-- The total length of 5 sticks with specific length relationships -/
theorem total_length_of_sticks : ∀ (stick1 stick2 stick3 stick4 stick5 : ℝ),
  stick1 = 3 →
  stick2 = 2 * stick1 →
  stick3 = stick2 - 1 →
  stick4 = stick3 / 2 →
  stick5 = 4 * stick4 →
  stick1 + stick2 + stick3 + stick4 + stick5 = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_total_length_of_sticks_l387_38750


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_range_l387_38778

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- State the theorem
theorem local_minimum_implies_b_range :
  ∀ b : ℝ, (∃ c ∈ Set.Ioo 0 1, IsLocalMin (f b) c) → 0 < b ∧ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_range_l387_38778


namespace NUMINAMATH_CALUDE_function_transformation_l387_38741

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) : f (x + 1) = 3 * x + 2 → f x = 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l387_38741


namespace NUMINAMATH_CALUDE_ladybugs_per_leaf_l387_38734

theorem ladybugs_per_leaf (total_leaves : ℕ) (total_ladybugs : ℕ) (h1 : total_leaves = 84) (h2 : total_ladybugs = 11676) :
  total_ladybugs / total_leaves = 139 :=
by sorry

end NUMINAMATH_CALUDE_ladybugs_per_leaf_l387_38734


namespace NUMINAMATH_CALUDE_rectangle_cut_and_rearrange_l387_38704

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Represents a cut of a rectangle into two parts -/
structure Cut where
  original : Rectangle
  part1 : Rectangle
  part2 : Rectangle

/-- Checks if a cut is valid (preserves area) -/
def Cut.isValid (c : Cut) : Prop :=
  c.original.area = c.part1.area + c.part2.area

/-- Theorem: A 14x6 rectangle can be cut into two parts that form a 21x4 rectangle -/
theorem rectangle_cut_and_rearrange :
  ∃ (c : Cut),
    c.original = { width := 14, height := 6 } ∧
    c.isValid ∧
    ∃ (new : Rectangle),
      new = { width := 21, height := 4 } ∧
      new.area = c.part1.area + c.part2.area :=
sorry

end NUMINAMATH_CALUDE_rectangle_cut_and_rearrange_l387_38704


namespace NUMINAMATH_CALUDE_four_solutions_to_equation_l387_38765

theorem four_solutions_to_equation :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2020 + y^2 = 2*y :=
sorry

end NUMINAMATH_CALUDE_four_solutions_to_equation_l387_38765


namespace NUMINAMATH_CALUDE_decimal_difference_l387_38791

-- Define the repeating decimal 0.3̄6
def repeating_decimal : ℚ := 4 / 11

-- Define the terminating decimal 0.36
def terminating_decimal : ℚ := 36 / 100

-- Theorem statement
theorem decimal_difference : 
  repeating_decimal - terminating_decimal = 4 / 1100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l387_38791


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l387_38725

-- Define the equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 1) + y^2 / (m + 2) = 1 ∧ (m + 1) * (m + 2) < 0

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m → -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l387_38725


namespace NUMINAMATH_CALUDE_no_three_similar_piles_l387_38756

theorem no_three_similar_piles (x : ℝ) (hx : x > 0) :
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    a ≤ b ∧ b ≤ c ∧
    c ≤ Real.sqrt 2 * b ∧
    b ≤ Real.sqrt 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_no_three_similar_piles_l387_38756


namespace NUMINAMATH_CALUDE_turn_on_all_in_four_moves_l387_38768

/-- Represents a light bulb on a 2D grid -/
structure Bulb where
  x : ℕ
  y : ℕ
  is_on : Bool

/-- Represents a line on a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the state of the grid -/
def GridState := List Bulb

/-- Checks if a bulb is on the specified side of a line -/
def is_on_side (b : Bulb) (l : Line) (positive_side : Bool) : Bool :=
  sorry

/-- Applies a move to the grid state -/
def apply_move (state : GridState) (l : Line) (positive_side : Bool) : GridState :=
  sorry

/-- Checks if all bulbs are on -/
def all_on (state : GridState) : Bool :=
  sorry

/-- Theorem: It's possible to turn on all bulbs in exactly four moves -/
theorem turn_on_all_in_four_moves :
  ∃ (moves : List (Line × Bool)),
    moves.length = 4 ∧
    let initial_state : GridState := [
      {x := 0, y := 0, is_on := false},
      {x := 0, y := 1, is_on := false},
      {x := 1, y := 0, is_on := false},
      {x := 1, y := 1, is_on := false}
    ]
    let final_state := moves.foldl (λ state move => apply_move state move.1 move.2) initial_state
    all_on final_state :=
  sorry

end NUMINAMATH_CALUDE_turn_on_all_in_four_moves_l387_38768


namespace NUMINAMATH_CALUDE_cookies_for_thanksgiving_l387_38797

/-- The number of cookies Helen baked three days ago -/
def cookies_day1 : ℕ := 31

/-- The number of cookies Helen baked two days ago -/
def cookies_day2 : ℕ := 270

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day3 : ℕ := 419

/-- The number of cookies Beaky ate from the first day's batch -/
def cookies_eaten_by_beaky : ℕ := 5

/-- The percentage of cookies that crumbled from the second day's batch -/
def crumble_percentage : ℚ := 15 / 100

/-- The number of cookies Helen gave away from the third day's batch -/
def cookies_given_away : ℕ := 30

/-- The number of cookies Helen received as a gift from Lucy -/
def cookies_gifted : ℕ := 45

/-- The total number of cookies available at Helen's house for Thanksgiving -/
def total_cookies : ℕ := 690

theorem cookies_for_thanksgiving :
  (cookies_day1 - cookies_eaten_by_beaky) +
  (cookies_day2 - Int.floor (crumble_percentage * cookies_day2)) +
  (cookies_day3 - cookies_given_away) +
  cookies_gifted = total_cookies := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_thanksgiving_l387_38797


namespace NUMINAMATH_CALUDE_root_distance_range_l387_38718

variables (a b c d : ℝ) (x₁ x₂ : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem root_distance_range (ha : a ≠ 0) 
  (hsum : a + b + c = 0) 
  (hf : f 0 * f 1 > 0) 
  (hx₁ : f x₁ = 0) 
  (hx₂ : f x₂ = 0) 
  (hx_distinct : x₁ ≠ x₂) :
  |x₁ - x₂| ∈ Set.Icc (Real.sqrt 3 / 3) (2 / 3) :=
sorry

end NUMINAMATH_CALUDE_root_distance_range_l387_38718


namespace NUMINAMATH_CALUDE_ellis_card_difference_l387_38775

/-- Represents the number of cards each player has -/
structure CardDistribution where
  ellis : ℕ
  orion : ℕ

/-- Calculates the card distribution based on the total cards and ratio -/
def distribute_cards (total : ℕ) (ellis_ratio : ℕ) (orion_ratio : ℕ) : CardDistribution :=
  let part_value := total / (ellis_ratio + orion_ratio)
  { ellis := ellis_ratio * part_value,
    orion := orion_ratio * part_value }

/-- Theorem stating that Ellis has 332 more cards than Orion -/
theorem ellis_card_difference (total : ℕ) (ellis_ratio : ℕ) (orion_ratio : ℕ)
  (h_total : total = 2500)
  (h_ellis_ratio : ellis_ratio = 17)
  (h_orion_ratio : orion_ratio = 13) :
  let distribution := distribute_cards total ellis_ratio orion_ratio
  distribution.ellis - distribution.orion = 332 := by
  sorry


end NUMINAMATH_CALUDE_ellis_card_difference_l387_38775


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l387_38742

theorem quadratic_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + 5*x₁ + k = 0 ∧ 
   x₂^2 + 5*x₂ + k = 0 ∧ 
   |x₁ - x₂| = 3) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l387_38742


namespace NUMINAMATH_CALUDE_library_visitors_l387_38733

theorem library_visitors (sunday_avg : ℕ) (month_days : ℕ) (month_avg : ℕ) :
  sunday_avg = 500 →
  month_days = 30 →
  month_avg = 200 →
  let sundays := (month_days + 6) / 7
  let other_days := month_days - sundays
  let other_avg := (month_days * month_avg - sundays * sunday_avg) / other_days
  other_avg = 140 := by
sorry

#eval (30 + 6) / 7  -- Should output 5, representing the number of Sundays

end NUMINAMATH_CALUDE_library_visitors_l387_38733


namespace NUMINAMATH_CALUDE_expression_simplification_l387_38748

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (((x^2 + 1) / (x - 1) - x + 1) / ((x^2) / (1 - x))) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l387_38748


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l387_38715

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_ratio : ℝ
  water_ratio : ℝ
  gravel_weight : ℝ

/-- Properties of the cement mixture -/
def is_valid_mixture (m : CementMixture) : Prop :=
  m.sand_ratio = 1/2 ∧
  m.water_ratio = 1/5 ∧
  m.gravel_weight = 15 ∧
  m.sand_ratio + m.water_ratio + m.gravel_weight / m.total_weight = 1

/-- Theorem stating that the total weight of the mixture is 50 pounds -/
theorem cement_mixture_weight (m : CementMixture) (h : is_valid_mixture m) : 
  m.total_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l387_38715


namespace NUMINAMATH_CALUDE_principal_is_15000_l387_38702

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest : ℚ) * 100 / (rate * time)

/-- Theorem: Given the specified conditions, the principal sum is 15000 -/
theorem principal_is_15000 :
  let simple_interest : ℕ := 2700
  let rate : ℚ := 6 / 100
  let time : ℕ := 3
  calculate_principal simple_interest rate time = 15000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_15000_l387_38702


namespace NUMINAMATH_CALUDE_f_is_quadratic_l387_38721

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 5y = 5y² -/
def f (y : ℝ) : ℝ := 5 * y^2 - 5 * y

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l387_38721


namespace NUMINAMATH_CALUDE_cars_return_to_start_l387_38728

/-- Represents a car on a circular race track -/
structure Car where
  position : ℝ  -- Position on the track (0 ≤ position < track_length)
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race -/
structure RaceState where
  track_length : ℝ
  cars : Vector Car n
  time : ℝ

/-- The race system evolves over time -/
def evolve_race (initial_state : RaceState) (t : ℝ) : RaceState :=
  sorry

/-- Predicate to check if all cars are at their initial positions -/
def all_cars_at_initial_positions (initial_state : RaceState) (current_state : RaceState) : Prop :=
  sorry

/-- Main theorem: There exists a time when all cars return to their initial positions -/
theorem cars_return_to_start {n : ℕ} (initial_state : RaceState) :
  ∃ t : ℝ, all_cars_at_initial_positions initial_state (evolve_race initial_state t) :=
  sorry

end NUMINAMATH_CALUDE_cars_return_to_start_l387_38728


namespace NUMINAMATH_CALUDE_ben_owes_rachel_l387_38758

theorem ben_owes_rachel (rate : ℚ) (lawns_mowed : ℚ) 
  (h1 : rate = 13 / 3) 
  (h2 : lawns_mowed = 8 / 5) : 
  rate * lawns_mowed = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ben_owes_rachel_l387_38758


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l387_38723

/-- Given a line passing through points (4, 0) and (-8, -6), 
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 :
  let m : ℚ := (0 - (-6)) / (4 - (-8))  -- Slope of the line
  let b : ℚ := 0 - m * 4                -- y-intercept of the line
  m * 10 + b = 3 := by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l387_38723


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l387_38737

/-- The sum of complex numbers 1 + i + i² + ... + i¹⁰ equals i -/
theorem sum_of_powers_of_i : 
  (Finset.range 11).sum (fun k => (Complex.I : ℂ) ^ k) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l387_38737


namespace NUMINAMATH_CALUDE_gold_copper_alloy_density_l387_38784

/-- The density of gold relative to water -/
def gold_density : ℝ := 10

/-- The density of copper relative to water -/
def copper_density : ℝ := 5

/-- The desired density of the alloy relative to water -/
def alloy_density : ℝ := 9

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 4

theorem gold_copper_alloy_density :
  ∀ (g c : ℝ),
  g > 0 → c > 0 →
  g / c = gold_copper_ratio →
  (gold_density * g + copper_density * c) / (g + c) = alloy_density :=
by sorry

end NUMINAMATH_CALUDE_gold_copper_alloy_density_l387_38784


namespace NUMINAMATH_CALUDE_specific_divisors_of_20_pow_30_l387_38731

def count_specific_divisors (n : ℕ) : ℕ :=
  let total_divisors := (60 + 1) * (30 + 1)
  let divisors_less_than_sqrt := (total_divisors - 1) / 2
  let divisors_of_sqrt := (30 + 1) * (15 + 1)
  divisors_less_than_sqrt - divisors_of_sqrt + 1

theorem specific_divisors_of_20_pow_30 :
  count_specific_divisors 20 = 450 := by
  sorry

end NUMINAMATH_CALUDE_specific_divisors_of_20_pow_30_l387_38731


namespace NUMINAMATH_CALUDE_infinite_product_equals_nine_l387_38744

/-- The series S is defined as the sum 1/2 + 2/4 + 3/8 + 4/16 + ... -/
def S : ℝ := 2

/-- The infinite product P is defined as 3^(1/2) * 9^(1/4) * 27^(1/8) * 81^(1/16) * ... -/
noncomputable def P : ℝ := Real.rpow 3 S

theorem infinite_product_equals_nine : P = 9 := by sorry

end NUMINAMATH_CALUDE_infinite_product_equals_nine_l387_38744


namespace NUMINAMATH_CALUDE_equation_solutions_l387_38713

theorem equation_solutions : 
  {x : ℝ | (2010 + 2*x)^2 = x^2} = {-2010, -670} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l387_38713


namespace NUMINAMATH_CALUDE_sum_of_squares_l387_38752

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : w * x = a^2) (h2 : w * y = b^2) (h3 : w * z = c^2) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l387_38752


namespace NUMINAMATH_CALUDE_range_of_a_l387_38779

def linear_function (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x + a - 8

def fractional_equation (a : ℝ) (y : ℝ) : Prop :=
  (y - 5) / (1 - y) + 3 = a / (y - 1)

theorem range_of_a (a : ℝ) :
  (∀ x y, y = linear_function a x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) →
  (∀ y, fractional_equation a y → y > -3) →
  1 < a ∧ a < 8 ∧ a ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l387_38779


namespace NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l387_38772

theorem positive_integer_sum_greater_than_product (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  m + n > m * n ↔ m = 1 ∨ n = 1 := by
sorry

end NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l387_38772


namespace NUMINAMATH_CALUDE_additional_income_needed_l387_38769

/-- Calculate the additional annual income needed to reach a target amount after expenses --/
theorem additional_income_needed
  (current_income : ℝ)
  (rent : ℝ)
  (groceries : ℝ)
  (gas : ℝ)
  (target_amount : ℝ)
  (h1 : current_income = 65000)
  (h2 : rent = 20000)
  (h3 : groceries = 5000)
  (h4 : gas = 8000)
  (h5 : target_amount = 42000) :
  current_income + 10000 - (rent + groceries + gas) ≥ target_amount ∧
  ∀ x : ℝ, x < 10000 → current_income + x - (rent + groceries + gas) < target_amount :=
by
  sorry

#check additional_income_needed

end NUMINAMATH_CALUDE_additional_income_needed_l387_38769


namespace NUMINAMATH_CALUDE_fib_units_digit_periodic_fib_15_value_units_digit_of_fib_fib_15_l387_38757

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_units_digit_periodic (n : ℕ) : fib n % 10 = fib (n % 60) % 10 := by sorry

theorem fib_15_value : fib 15 = 610 := by sorry

theorem units_digit_of_fib_fib_15 : fib (fib 15) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_fib_units_digit_periodic_fib_15_value_units_digit_of_fib_fib_15_l387_38757


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l387_38774

theorem positive_real_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2) ∧
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c ≥ 2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l387_38774


namespace NUMINAMATH_CALUDE_car_wash_earnings_l387_38732

/-- Proves that a car wash company cleaning 80 cars per day at $5 per car will earn $2000 in 5 days -/
theorem car_wash_earnings 
  (cars_per_day : ℕ) 
  (price_per_car : ℕ) 
  (num_days : ℕ) 
  (h1 : cars_per_day = 80) 
  (h2 : price_per_car = 5) 
  (h3 : num_days = 5) : 
  cars_per_day * price_per_car * num_days = 2000 := by
  sorry

#check car_wash_earnings

end NUMINAMATH_CALUDE_car_wash_earnings_l387_38732


namespace NUMINAMATH_CALUDE_distinguishable_triangles_l387_38781

def num_colors : ℕ := 8

def corner_configurations : ℕ := 
  num_colors + num_colors * (num_colors - 1) + (num_colors.choose 3)

def center_configurations : ℕ := num_colors * (num_colors - 1)

theorem distinguishable_triangles : 
  corner_configurations * center_configurations = 6720 := by sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_l387_38781
