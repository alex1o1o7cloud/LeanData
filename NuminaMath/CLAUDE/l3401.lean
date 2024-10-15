import Mathlib

namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3401_340156

/-- Proves that a side of beef with given initial and final weights loses approximately 30% of its weight during processing -/
theorem beef_weight_loss_percentage (initial_weight final_weight : ℝ) 
  (h1 : initial_weight = 714.2857142857143)
  (h2 : final_weight = 500) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((initial_weight - final_weight) / initial_weight * 100) - 30| < ε :=
sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3401_340156


namespace NUMINAMATH_CALUDE_systematic_sampling_identification_l3401_340116

/-- A sampling method is a function that selects elements from a population. -/
def SamplingMethod := Type → Type

/-- Systematic sampling is a method where samples are selected at regular intervals. -/
def IsSystematicSampling (m : SamplingMethod) : Prop := sorry

/-- Method 1: Sampling from numbered balls with a fixed interval. -/
def Method1 : SamplingMethod := sorry

/-- Method 2: Sampling products from a conveyor belt at fixed time intervals. -/
def Method2 : SamplingMethod := sorry

/-- Method 3: Random sampling at a shopping mall entrance. -/
def Method3 : SamplingMethod := sorry

/-- Method 4: Sampling moviegoers in specific seats. -/
def Method4 : SamplingMethod := sorry

/-- Theorem stating which methods are systematic sampling. -/
theorem systematic_sampling_identification :
  IsSystematicSampling Method1 ∧
  IsSystematicSampling Method2 ∧
  ¬IsSystematicSampling Method3 ∧
  IsSystematicSampling Method4 := by sorry

end NUMINAMATH_CALUDE_systematic_sampling_identification_l3401_340116


namespace NUMINAMATH_CALUDE_marketing_cost_per_book_l3401_340121

/-- The marketing cost per book for a publishing company --/
theorem marketing_cost_per_book 
  (fixed_cost : ℝ) 
  (selling_price : ℝ) 
  (break_even_quantity : ℕ) 
  (h1 : fixed_cost = 50000)
  (h2 : selling_price = 9)
  (h3 : break_even_quantity = 10000) :
  (selling_price * break_even_quantity - fixed_cost) / break_even_quantity = 4 := by
sorry


end NUMINAMATH_CALUDE_marketing_cost_per_book_l3401_340121


namespace NUMINAMATH_CALUDE_probability_all_truth_l3401_340192

theorem probability_all_truth (pA pB pC pD : ℝ) 
  (hA : pA = 0.55) 
  (hB : pB = 0.60) 
  (hC : pC = 0.45) 
  (hD : pD = 0.70) : 
  pA * pB * pC * pD = 0.10395 := by
sorry

end NUMINAMATH_CALUDE_probability_all_truth_l3401_340192


namespace NUMINAMATH_CALUDE_paper_area_difference_paper_area_difference_proof_l3401_340100

/-- The difference in combined area (front and back) between a square sheet of paper
    with side length 11 inches and a rectangular sheet of paper measuring 5.5 inches
    by 11 inches is 121 square inches. -/
theorem paper_area_difference : ℝ → Prop :=
  λ (inch : ℝ) =>
    let square_sheet_side := 11 * inch
    let rect_sheet_length := 5.5 * inch
    let rect_sheet_width := 11 * inch
    let square_sheet_area := 2 * (square_sheet_side * square_sheet_side)
    let rect_sheet_area := 2 * (rect_sheet_length * rect_sheet_width)
    square_sheet_area - rect_sheet_area = 121 * inch * inch

/-- Proof of the paper_area_difference theorem. -/
theorem paper_area_difference_proof : paper_area_difference 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_area_difference_paper_area_difference_proof_l3401_340100


namespace NUMINAMATH_CALUDE_five_foxes_weight_l3401_340152

/-- The weight of a single fox in kilograms. -/
def fox_weight : ℝ := sorry

/-- The weight of a single dog in kilograms. -/
def dog_weight : ℝ := fox_weight + 5

/-- The total weight of 3 foxes and 5 dogs in kilograms. -/
def total_weight : ℝ := 65

theorem five_foxes_weight :
  3 * fox_weight + 5 * dog_weight = total_weight →
  5 * fox_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_five_foxes_weight_l3401_340152


namespace NUMINAMATH_CALUDE_geometric_sequence_312th_term_l3401_340134

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a₁ : ℝ  -- first term
  r : ℝ   -- common ratio
  
/-- The nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a₁ * seq.r ^ (n - 1)

/-- Theorem: The 312th term of the specific geometric sequence -/
theorem geometric_sequence_312th_term :
  let seq : GeometricSequence := { a₁ := 12, r := -1/2 }
  seq.nthTerm 312 = -12 * (1/2)^311 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_312th_term_l3401_340134


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3401_340109

theorem quadratic_perfect_square (x : ℝ) (d : ℝ) :
  (∃ b : ℝ, ∀ x, x^2 + 60*x + d = (x + b)^2) ↔ d = 900 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3401_340109


namespace NUMINAMATH_CALUDE_smallest_block_size_l3401_340168

/-- Given a rectangular block formed by N congruent 1-cm cubes,
    where 252 cubes are invisible when three faces are viewed,
    the smallest possible value of N is 392. -/
theorem smallest_block_size (N : ℕ) : 
  (∃ l m n : ℕ, 
    l > 0 ∧ m > 0 ∧ n > 0 ∧
    (l - 1) * (m - 1) * (n - 1) = 252 ∧
    N = l * m * n) →
  N ≥ 392 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l3401_340168


namespace NUMINAMATH_CALUDE_fruit_count_correct_l3401_340123

structure FruitBasket :=
  (plums : ℕ)
  (oranges : ℕ)
  (apples : ℕ)
  (pears : ℕ)
  (cherries : ℕ)

def initial_basket : FruitBasket :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def exchanges (basket : FruitBasket) : FruitBasket :=
  { plums := basket.plums - 4 + 2,
    oranges := basket.oranges - 3 + 1,
    apples := basket.apples - 5 + 2,
    pears := basket.pears + 1 + 3,
    cherries := basket.cherries + 2 }

def final_basket : FruitBasket :=
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 }

theorem fruit_count_correct : exchanges initial_basket = final_basket := by
  sorry

end NUMINAMATH_CALUDE_fruit_count_correct_l3401_340123


namespace NUMINAMATH_CALUDE_lady_walking_distance_l3401_340126

theorem lady_walking_distance (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 12) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_lady_walking_distance_l3401_340126


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l3401_340103

/-- The length of a bridge given specific train characteristics and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that the bridge length is 205 meters given the specific conditions -/
theorem bridge_length_specific : bridge_length 170 45 30 = 205 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l3401_340103


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3401_340120

/-- Proves that the cost price of an article is 95 given the specified conditions -/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  marked_price = 125 ∧ 
  discount_rate = 0.05 ∧ 
  profit_rate = 0.25 →
  ∃ (cost_price : ℝ), 
    cost_price = 95 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3401_340120


namespace NUMINAMATH_CALUDE_female_population_l3401_340199

theorem female_population (total_population : ℕ) (male_ratio female_ratio : ℕ) 
  (h_total : total_population = 500)
  (h_ratio : male_ratio = 3 ∧ female_ratio = 2) : 
  (female_ratio * total_population) / (male_ratio + female_ratio) = 200 := by
  sorry

end NUMINAMATH_CALUDE_female_population_l3401_340199


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2_l3401_340141

theorem unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2 :
  ∃! n : ℕ+, 18 ∣ n ∧ 30 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2_l3401_340141


namespace NUMINAMATH_CALUDE_company_uniforms_l3401_340160

theorem company_uniforms (num_stores : ℕ) (uniforms_per_store : ℕ) 
  (h1 : num_stores = 32) (h2 : uniforms_per_store = 4) : 
  num_stores * uniforms_per_store = 128 := by
  sorry

end NUMINAMATH_CALUDE_company_uniforms_l3401_340160


namespace NUMINAMATH_CALUDE_count_with_zero_up_to_3500_l3401_340163

/-- Counts the number of integers from 1 to n that contain the digit 0 in base 10 -/
def count_with_zero (n : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 773 numbers containing 0 up to 3500 -/
theorem count_with_zero_up_to_3500 : count_with_zero 3500 = 773 := by sorry

end NUMINAMATH_CALUDE_count_with_zero_up_to_3500_l3401_340163


namespace NUMINAMATH_CALUDE_sum_of_nested_logs_l3401_340185

-- Define the logarithm functions
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem sum_of_nested_logs (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧
  log 3 (log 4 (log 2 y)) = 0 ∧
  log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nested_logs_l3401_340185


namespace NUMINAMATH_CALUDE_speed_increase_problem_l3401_340122

/-- The speed increase problem -/
theorem speed_increase_problem 
  (initial_speed : ℝ) 
  (distance : ℝ) 
  (late_time : ℝ) 
  (early_time : ℝ) 
  (h1 : initial_speed = 2) 
  (h2 : distance = 2) 
  (h3 : late_time = 1/6) 
  (h4 : early_time = 1/6) : 
  ∃ (speed_increase : ℝ), 
    speed_increase = 
      (distance / (distance / initial_speed - late_time - early_time)) - initial_speed ∧ 
    speed_increase = 1 := by
  sorry

#check speed_increase_problem

end NUMINAMATH_CALUDE_speed_increase_problem_l3401_340122


namespace NUMINAMATH_CALUDE_slope_range_l3401_340170

theorem slope_range (a : ℝ) :
  let k := -(1 / (a^2 + Real.sqrt 3))
  5 * Real.pi / 6 ≤ Real.arctan k ∧ Real.arctan k < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_slope_range_l3401_340170


namespace NUMINAMATH_CALUDE_sphere_radius_l3401_340135

/-- Given two spheres A and B, where A has radius 40 cm and the ratio of their surface areas is 16,
    prove that the radius of sphere B is 20 cm. -/
theorem sphere_radius (r : ℝ) : 
  let surface_area (radius : ℝ) := 4 * Real.pi * radius^2
  surface_area 40 / surface_area r = 16 → r = 20 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l3401_340135


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3401_340187

theorem negative_fraction_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3401_340187


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l3401_340193

theorem parabola_vertex_sum_max (a U : ℤ) (h_U : U ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let N := (3 * U / 2 : ℝ) + (- 9 * a * U^2 / 4 : ℝ)
  (passes_through 0 0) ∧ 
  (passes_through (3 * U) 0) ∧ 
  (passes_through (3 * U - 1) 12) →
  N ≤ 71/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l3401_340193


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_area_sum_l3401_340119

-- Define a parallelogram type
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

-- Define the property of having right or obtuse angles
def has_right_or_obtuse_angles (p : Parallelogram) : Prop :=
  sorry

-- Define the perimeter of the parallelogram
def perimeter (p : Parallelogram) : ℝ :=
  sorry

-- Define the area of the parallelogram
def area (p : Parallelogram) : ℝ :=
  sorry

-- Theorem statement
theorem parallelogram_perimeter_area_sum :
  ∀ p : Parallelogram,
  p.v1 = (6, 3) ∧ p.v2 = (9, 7) ∧ p.v3 = (2, 0) ∧
  has_right_or_obtuse_angles p →
  perimeter p + area p = 48 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_area_sum_l3401_340119


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3401_340172

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x / 2) ^ (1/3 : ℝ) = -4 :=
by
  -- The unique solution is x = 134
  use 134
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3401_340172


namespace NUMINAMATH_CALUDE_divide_fractions_three_sevenths_div_two_and_half_l3401_340114

theorem divide_fractions (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem three_sevenths_div_two_and_half :
  (3 : ℚ) / 7 / (5 / 2) = 6 / 35 := by sorry

end NUMINAMATH_CALUDE_divide_fractions_three_sevenths_div_two_and_half_l3401_340114


namespace NUMINAMATH_CALUDE_car_braking_distance_l3401_340113

/-- Represents the braking sequence of a car -/
def brakingSequence (initial : ℕ) (decrease : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => max 0 (brakingSequence initial decrease n - decrease)

/-- Calculates the total distance traveled during braking -/
def totalDistance (initial : ℕ) (decrease : ℕ) : ℕ :=
  (List.range 100).foldl (λ acc n => acc + brakingSequence initial decrease n) 0

/-- Theorem stating the total braking distance for the given conditions -/
theorem car_braking_distance :
  totalDistance 36 8 = 108 := by
  sorry


end NUMINAMATH_CALUDE_car_braking_distance_l3401_340113


namespace NUMINAMATH_CALUDE_knight_returns_to_start_l3401_340108

/-- A castle in Mara -/
structure Castle where
  id : ℕ

/-- The graph of castles and roads in Mara -/
structure MaraGraph where
  castles : Set Castle
  roads : Castle → Set Castle
  finite_castles : Set.Finite castles
  three_roads : ∀ c, (roads c).ncard = 3

/-- A turn direction -/
inductive Turn
| Left
| Right

/-- A path through the castles -/
structure KnightPath (G : MaraGraph) where
  path : ℕ → Castle
  turns : ℕ → Turn
  valid_path : ∀ n, G.roads (path n) (path (n + 1))
  alternating_turns : ∀ n, turns n ≠ turns (n + 1)

/-- The theorem stating that the knight will return to the original castle -/
theorem knight_returns_to_start (G : MaraGraph) (p : KnightPath G) :
  ∃ n m, n < m ∧ p.path n = p.path m := by sorry

end NUMINAMATH_CALUDE_knight_returns_to_start_l3401_340108


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3401_340111

/-- The polynomial x^4 - 3x^3 + mx + n -/
def f (m n x : ℂ) : ℂ := x^4 - 3*x^3 + m*x + n

/-- The polynomial x^2 - 2x + 4 -/
def g (x : ℂ) : ℂ := x^2 - 2*x + 4

theorem polynomial_divisibility (m n : ℂ) :
  (∀ x, g x = 0 → f m n x = 0) →
  g (1 + Complex.I * Real.sqrt 3) = 0 →
  g (1 - Complex.I * Real.sqrt 3) = 0 →
  m = 8 ∧ n = -24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3401_340111


namespace NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l3401_340132

theorem modular_inverse_35_mod_36 : ∃ x : ℤ, (35 * x) % 36 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l3401_340132


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l3401_340150

theorem at_least_one_geq_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l3401_340150


namespace NUMINAMATH_CALUDE_abs_three_minus_pi_l3401_340186

theorem abs_three_minus_pi : |3 - Real.pi| = Real.pi - 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_three_minus_pi_l3401_340186


namespace NUMINAMATH_CALUDE_element_in_set_implies_a_values_l3401_340155

theorem element_in_set_implies_a_values (a : ℝ) : 
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_implies_a_values_l3401_340155


namespace NUMINAMATH_CALUDE_constant_sum_and_square_sum_implies_constant_S_l3401_340137

theorem constant_sum_and_square_sum_implies_constant_S 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) : 
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * (a^2 + b^2 + c^2 + d^2) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_constant_sum_and_square_sum_implies_constant_S_l3401_340137


namespace NUMINAMATH_CALUDE_geometric_sequence_15th_term_l3401_340146

/-- Given a geometric sequence where the 8th term is 8 and the 11th term is 64,
    prove that the 15th term is 1024. -/
theorem geometric_sequence_15th_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (a 11 / a 8)^(1/3)) →
  a 8 = 8 →
  a 11 = 64 →
  a 15 = 1024 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_15th_term_l3401_340146


namespace NUMINAMATH_CALUDE_twentiethTerm_eq_97_l3401_340139

/-- The nth term of an arithmetic sequence -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence -/
def twentiethTerm : ℝ :=
  arithmeticSequence 2 5 20

theorem twentiethTerm_eq_97 : twentiethTerm = 97 := by
  sorry

end NUMINAMATH_CALUDE_twentiethTerm_eq_97_l3401_340139


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l3401_340125

/-- Given a point A with coordinates (1, 2), its symmetric point A' with respect to the origin has coordinates (-1, -2) -/
theorem symmetric_point_wrt_origin :
  let A : ℝ × ℝ := (1, 2)
  let A' : ℝ × ℝ := (-A.1, -A.2)
  A' = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l3401_340125


namespace NUMINAMATH_CALUDE_class_size_multiple_of_eight_l3401_340158

theorem class_size_multiple_of_eight (boys girls total : ℕ) : 
  girls = 7 * boys → total = boys + girls → ∃ k : ℕ, total = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_class_size_multiple_of_eight_l3401_340158


namespace NUMINAMATH_CALUDE_volume_formula_l3401_340173

/-- A pyramid with a rectangular base -/
structure Pyramid where
  /-- Length of side AB of the base -/
  ab : ℝ
  /-- Length of side AD of the base -/
  ad : ℝ
  /-- Angle AQB where Q is the apex -/
  θ : ℝ
  /-- Assertion that AB = 2 -/
  hab : ab = 2
  /-- Assertion that AD = 1 -/
  had : ad = 1
  /-- Assertion that Q is directly above the center of the base -/
  hcenter : True
  /-- Assertion that Q is equidistant from all vertices -/
  hequidistant : True

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  (2/3) * Real.sqrt (Real.tan p.θ ^ 2 + 1/4)

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula (p : Pyramid) : volume p = (2/3) * Real.sqrt (Real.tan p.θ ^ 2 + 1/4) := by
  sorry

end NUMINAMATH_CALUDE_volume_formula_l3401_340173


namespace NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l3401_340124

theorem sqrt_fifteen_over_two_equals_half_sqrt_thirty : 
  Real.sqrt (15 / 2) = (1 / 2) * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l3401_340124


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l3401_340131

theorem divisibility_by_eight : ∃ k : ℤ, 5^2001 + 7^2002 + 9^2003 + 11^2004 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l3401_340131


namespace NUMINAMATH_CALUDE_quartic_sum_theorem_l3401_340136

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + m
  at_zero : Q 0 = m
  at_one : Q 1 = 2 * m
  at_neg_one : Q (-1) = 4 * m
  at_two : Q 2 = 5 * m

/-- Theorem: For a quartic polynomial Q satisfying specific conditions, Q(2) + Q(-2) = 66m -/
theorem quartic_sum_theorem (m : ℝ) (qp : QuarticPolynomial m) : qp.Q 2 + qp.Q (-2) = 66 * m := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_theorem_l3401_340136


namespace NUMINAMATH_CALUDE_carolyns_essay_body_sections_l3401_340165

/-- Represents the structure of Carolyn's essay -/
structure EssayStructure where
  intro_length : ℕ
  conclusion_length : ℕ
  body_section_length : ℕ
  total_length : ℕ

/-- Calculates the number of body sections in Carolyn's essay -/
def calculate_body_sections (essay : EssayStructure) : ℕ :=
  let remaining_length := essay.total_length - (essay.intro_length + essay.conclusion_length)
  remaining_length / essay.body_section_length

/-- Theorem stating that Carolyn's essay has 4 body sections -/
theorem carolyns_essay_body_sections :
  let essay := EssayStructure.mk 450 (3 * 450) 800 5000
  calculate_body_sections essay = 4 := by
  sorry

end NUMINAMATH_CALUDE_carolyns_essay_body_sections_l3401_340165


namespace NUMINAMATH_CALUDE_map_distance_to_actual_distance_l3401_340166

theorem map_distance_to_actual_distance 
  (map_distance : ℝ) 
  (scale_map : ℝ) 
  (scale_actual : ℝ) 
  (h1 : map_distance = 18) 
  (h2 : scale_map = 0.5) 
  (h3 : scale_actual = 8) : 
  map_distance * (scale_actual / scale_map) = 288 := by
sorry

end NUMINAMATH_CALUDE_map_distance_to_actual_distance_l3401_340166


namespace NUMINAMATH_CALUDE_nested_root_equation_l3401_340179

theorem nested_root_equation (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → M^(1/d + 1/(d*e) + 1/(d*e*f)) = M^(17/24)) →
  e = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_root_equation_l3401_340179


namespace NUMINAMATH_CALUDE_city_population_dynamics_l3401_340133

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birthRate : ℝ  -- Average birth rate per second
  netIncrease : ℝ  -- Net population increase per second
  deathRate : ℝ  -- Average death rate per second

/-- Theorem stating the relationship between birth rate, net increase, and death rate --/
theorem city_population_dynamics (city : CityPopulation) 
  (h1 : city.birthRate = 3.5)
  (h2 : city.netIncrease = 2) :
  city.deathRate = 1.5 := by
  sorry

#check city_population_dynamics

end NUMINAMATH_CALUDE_city_population_dynamics_l3401_340133


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3401_340128

def sum_even_2_to_40 : ℕ := (20 / 2) * (2 + 40)

def sum_odd_1_to_39 : ℕ := (20 / 2) * (1 + 39)

theorem even_odd_sum_difference : sum_even_2_to_40 - sum_odd_1_to_39 = 20 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3401_340128


namespace NUMINAMATH_CALUDE_initial_bees_correct_l3401_340161

/-- The initial number of bees in the colony. -/
def initial_bees : ℕ := 80000

/-- The daily loss of bees. -/
def daily_loss : ℕ := 1200

/-- The number of days after which the colony reaches a fourth of its initial size. -/
def days : ℕ := 50

/-- Theorem stating that the initial number of bees is correct given the conditions. -/
theorem initial_bees_correct : 
  initial_bees = daily_loss * days * 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_bees_correct_l3401_340161


namespace NUMINAMATH_CALUDE_smallest_z_value_l3401_340107

theorem smallest_z_value (x y z : ℝ) : 
  (7 < x) → (x < 9) → (9 < y) → (y < z) → 
  (∃ (n : ℕ), y - x = n ∧ ∀ (m : ℕ), y - x ≤ m → m ≤ n) →
  (∀ (w : ℝ), (7 < w) → (w < 9) → (9 < y) → (y < z) → 
    ∃ (k : ℕ), y - w ≤ k ∧ k ≤ 7) →
  z ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l3401_340107


namespace NUMINAMATH_CALUDE_point_on_graph_l3401_340182

def is_on_graph (x y k : ℝ) : Prop := y = k * x - 2

theorem point_on_graph (k : ℝ) :
  is_on_graph 2 4 k → is_on_graph 1 1 k :=
by sorry

end NUMINAMATH_CALUDE_point_on_graph_l3401_340182


namespace NUMINAMATH_CALUDE_brick_width_is_four_l3401_340198

/-- The surface area of a rectangular prism given its length, width, and height -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular prism with length 10, height 2, and surface area 136 is 4 -/
theorem brick_width_is_four :
  ∃ w : ℝ, w > 0 ∧ surfaceArea 10 w 2 = 136 → w = 4 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_is_four_l3401_340198


namespace NUMINAMATH_CALUDE_deductive_reasoning_examples_l3401_340110

-- Define a type for the reasoning examples
inductive ReasoningExample
  | example1
  | example2
  | example3
  | example4

-- Define a function to check if a reasoning example is deductive
def isDeductive : ReasoningExample → Bool
  | ReasoningExample.example1 => false
  | ReasoningExample.example2 => true
  | ReasoningExample.example3 => false
  | ReasoningExample.example4 => true

-- Theorem statement
theorem deductive_reasoning_examples :
  ∀ (e : ReasoningExample), isDeductive e ↔ (e = ReasoningExample.example2 ∨ e = ReasoningExample.example4) := by
  sorry


end NUMINAMATH_CALUDE_deductive_reasoning_examples_l3401_340110


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l3401_340106

def range_size : ℕ := 60
def num_selections : ℕ := 3

def multiples_of_four (n : ℕ) : ℕ := (n + 3) / 4

theorem probability_at_least_one_multiple_of_four :
  let p := 1 - (1 - multiples_of_four range_size / range_size) ^ num_selections
  p = 37 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l3401_340106


namespace NUMINAMATH_CALUDE_billys_age_l3401_340189

theorem billys_age :
  ∀ (billy_age joe_age : ℚ),
    billy_age + 5 = 2 * joe_age →
    billy_age + joe_age = 60 →
    billy_age = 115 / 3 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l3401_340189


namespace NUMINAMATH_CALUDE_system_solution_l3401_340183

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k → 
  2*x + y = 1 → 
  x + y = 3 → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3401_340183


namespace NUMINAMATH_CALUDE_count_large_glasses_l3401_340144

/-- The number of jelly beans needed to fill a large drinking glass -/
def large_glass_jelly_beans : ℕ := 50

/-- The number of jelly beans needed to fill a small drinking glass -/
def small_glass_jelly_beans : ℕ := 25

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jelly beans used to fill all glasses -/
def total_jelly_beans : ℕ := 325

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

theorem count_large_glasses : 
  large_glass_jelly_beans * num_large_glasses + 
  small_glass_jelly_beans * num_small_glasses = total_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_count_large_glasses_l3401_340144


namespace NUMINAMATH_CALUDE_store_purchase_total_l3401_340177

/-- Calculate the total amount spent at the store -/
theorem store_purchase_total (initial_backpack_price initial_binder_price : ℚ)
  (backpack_increase binder_decrease : ℚ)
  (backpack_discount binder_deal sales_tax : ℚ)
  (num_binders : ℕ) :
  let new_backpack_price := initial_backpack_price + backpack_increase
  let new_binder_price := initial_binder_price - binder_decrease
  let discounted_backpack_price := new_backpack_price * (1 - backpack_discount)
  let binders_to_pay := (num_binders + 1) / 2
  let total_binder_price := new_binder_price * binders_to_pay
  let subtotal := discounted_backpack_price + total_binder_price
  let total_with_tax := subtotal * (1 + sales_tax)
  initial_backpack_price = 50 ∧
  initial_binder_price = 20 ∧
  backpack_increase = 5 ∧
  binder_decrease = 2 ∧
  backpack_discount = 0.1 ∧
  sales_tax = 0.06 ∧
  num_binders = 3 →
  total_with_tax = 90.63 :=
by sorry

end NUMINAMATH_CALUDE_store_purchase_total_l3401_340177


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l3401_340142

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_picks : ℕ := 7
def num_blue_picked : ℕ := 3

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose num_picks num_blue_picked) * 
  (probability_blue ^ num_blue_picked) * 
  (probability_red ^ (num_picks - num_blue_picked)) = 862 / 3417 :=
sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l3401_340142


namespace NUMINAMATH_CALUDE_tomatoes_picked_today_l3401_340174

/-- Represents the number of tomatoes in various states --/
structure TomatoCount where
  initial : ℕ
  pickedYesterday : ℕ
  leftAfterYesterday : ℕ

/-- Theorem: The number of tomatoes picked today is equal to the initial number
    minus the number left after yesterday's picking --/
theorem tomatoes_picked_today (t : TomatoCount)
  (h1 : t.initial = 160)
  (h2 : t.pickedYesterday = 56)
  (h3 : t.leftAfterYesterday = 104)
  : t.initial - t.leftAfterYesterday = 56 := by
  sorry


end NUMINAMATH_CALUDE_tomatoes_picked_today_l3401_340174


namespace NUMINAMATH_CALUDE_problem_solution_l3401_340153

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3401_340153


namespace NUMINAMATH_CALUDE_class_size_from_average_change_l3401_340178

theorem class_size_from_average_change 
  (original_mark : ℕ) 
  (incorrect_mark : ℕ)
  (mark_difference : ℕ)
  (average_increase : ℚ) :
  incorrect_mark = original_mark + mark_difference →
  mark_difference = 20 →
  average_increase = 1/2 →
  (mark_difference : ℚ) / (class_size : ℕ) = average_increase →
  class_size = 40 := by
sorry

end NUMINAMATH_CALUDE_class_size_from_average_change_l3401_340178


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3401_340149

theorem inequality_equivalence (x : ℝ) : 
  (1/2: ℝ) ^ (x^2 - 2*x + 3) < (1/2 : ℝ) ^ (2*x^2 + 3*x - 3) ↔ -6 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3401_340149


namespace NUMINAMATH_CALUDE_debby_pancakes_count_l3401_340176

/-- The number of pancakes Debby made with blueberries -/
def blueberry_pancakes : ℕ := 20

/-- The number of pancakes Debby made with bananas -/
def banana_pancakes : ℕ := 24

/-- The number of plain pancakes Debby made -/
def plain_pancakes : ℕ := 23

/-- The total number of pancakes Debby made -/
def total_pancakes : ℕ := blueberry_pancakes + banana_pancakes + plain_pancakes

theorem debby_pancakes_count : total_pancakes = 67 := by
  sorry

end NUMINAMATH_CALUDE_debby_pancakes_count_l3401_340176


namespace NUMINAMATH_CALUDE_ali_wallet_l3401_340162

def wallet_problem (num_five_dollar_bills : ℕ) (total_amount : ℕ) : ℕ := 
  let five_dollar_amount := 5 * num_five_dollar_bills
  let ten_dollar_amount := total_amount - five_dollar_amount
  ten_dollar_amount / 10

theorem ali_wallet :
  wallet_problem 7 45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ali_wallet_l3401_340162


namespace NUMINAMATH_CALUDE_vehicle_Y_ahead_distance_l3401_340180

-- Define the vehicles and their properties
structure Vehicle where
  speed : ℝ
  initialPosition : ℝ

-- Define the problem parameters
def time : ℝ := 5
def vehicleX : Vehicle := { speed := 36, initialPosition := 22 }
def vehicleY : Vehicle := { speed := 45, initialPosition := 0 }

-- Define the function to calculate the position of a vehicle after a given time
def position (v : Vehicle) (t : ℝ) : ℝ :=
  v.initialPosition + v.speed * t

-- Theorem statement
theorem vehicle_Y_ahead_distance : 
  position vehicleY time - position vehicleX time = 23 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_Y_ahead_distance_l3401_340180


namespace NUMINAMATH_CALUDE_binomial_largest_coefficient_l3401_340195

/-- 
Given a positive integer n, if the binomial coefficient in the expansion of (2+x)^n 
is largest in the 4th and 5th terms, then n = 7.
-/
theorem binomial_largest_coefficient (n : ℕ+) : 
  (∀ k : ℕ, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 3) ∧
  (∀ k : ℕ, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧
  Nat.choose n 3 = Nat.choose n 4 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_largest_coefficient_l3401_340195


namespace NUMINAMATH_CALUDE_sum_first_5_even_numbers_l3401_340147

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.map (fun i => 2 * i) (List.range n)

theorem sum_first_5_even_numbers :
  (first_n_even_numbers 5).sum = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_5_even_numbers_l3401_340147


namespace NUMINAMATH_CALUDE_two_thousand_second_non_diff_of_squares_non_diff_of_squares_form_eight_thousand_six_form_main_theorem_l3401_340194

/-- A function that returns true if a number is the difference of two squares -/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = x^2 - y^2

/-- A function that returns the nth number of the form 4k + 2 -/
def nth_non_diff_of_squares (n : ℕ) : ℕ :=
  4 * n - 2

/-- Theorem stating that 8006 is the 2002nd positive integer that is not the difference of two squares -/
theorem two_thousand_second_non_diff_of_squares :
  nth_non_diff_of_squares 2002 = 8006 ∧ ¬(is_diff_of_squares 8006) :=
sorry

/-- Theorem stating that numbers of the form 4k + 2 cannot be expressed as the difference of two squares -/
theorem non_diff_of_squares_form (k : ℕ) :
  ¬(is_diff_of_squares (4 * k + 2)) :=
sorry

/-- Theorem stating that 8006 is of the form 4k + 2 -/
theorem eight_thousand_six_form :
  ∃ k : ℕ, 8006 = 4 * k + 2 :=
sorry

/-- Main theorem combining the above results -/
theorem main_theorem :
  ∃ n : ℕ, n = 2002 ∧ nth_non_diff_of_squares n = 8006 ∧ ¬(is_diff_of_squares 8006) :=
sorry

end NUMINAMATH_CALUDE_two_thousand_second_non_diff_of_squares_non_diff_of_squares_form_eight_thousand_six_form_main_theorem_l3401_340194


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l3401_340143

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2 : ℝ) = π * (r₁^2 - r₂^2) :=
by sorry

/-- The area of a ring formed by concentric circles with radii 12 and 7 is 95π -/
theorem area_of_specific_ring : 
  (π * 12^2 - π * 7^2 : ℝ) = 95 * π :=
by sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l3401_340143


namespace NUMINAMATH_CALUDE_plugs_count_l3401_340118

/-- The number of pairs of mittens in the box -/
def mittens_pairs : ℕ := 150

/-- The number of pairs of plugs initially in the box -/
def initial_plugs_pairs : ℕ := mittens_pairs + 20

/-- The number of additional pairs of plugs added -/
def additional_plugs_pairs : ℕ := 30

/-- The total number of plugs after additions -/
def total_plugs : ℕ := 2 * (initial_plugs_pairs + additional_plugs_pairs)

theorem plugs_count : total_plugs = 400 := by
  sorry

end NUMINAMATH_CALUDE_plugs_count_l3401_340118


namespace NUMINAMATH_CALUDE_complex_product_l3401_340129

def z₁ : ℂ := 1 + 2 * Complex.I
def z₂ : ℂ := 2 - Complex.I

theorem complex_product : z₁ * z₂ = 4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_l3401_340129


namespace NUMINAMATH_CALUDE_extremal_points_sum_bound_l3401_340130

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + 1)

theorem extremal_points_sum_bound {a : ℝ} (ha : a > 0) 
  (x₁ x₂ : ℝ) (h_extremal : ∀ x, x ≠ x₁ → x ≠ x₂ → (deriv (f a)) x ≠ 0) :
  f a x₁ + f a x₂ < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_extremal_points_sum_bound_l3401_340130


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3401_340171

/-- The atomic mass of Deuterium (H-2) in atomic mass units (amu) -/
def mass_deuterium : ℝ := 2.014

/-- The atomic mass of Carbon-13 (C-13) in atomic mass units (amu) -/
def mass_carbon13 : ℝ := 13.003

/-- The atomic mass of Oxygen-16 (O-16) in atomic mass units (amu) -/
def mass_oxygen16 : ℝ := 15.995

/-- The atomic mass of Oxygen-18 (O-18) in atomic mass units (amu) -/
def mass_oxygen18 : ℝ := 17.999

/-- The number of Deuterium molecules in the compound -/
def num_deuterium : ℕ := 2

/-- The number of Carbon-13 molecules in the compound -/
def num_carbon13 : ℕ := 1

/-- The number of Oxygen-16 molecules in the compound -/
def num_oxygen16 : ℕ := 1

/-- The number of Oxygen-18 molecules in the compound -/
def num_oxygen18 : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  num_deuterium * mass_deuterium +
  num_carbon13 * mass_carbon13 +
  num_oxygen16 * mass_oxygen16 +
  num_oxygen18 * mass_oxygen18

theorem compound_molecular_weight :
  ∃ ε > 0, |molecular_weight - 69.024| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3401_340171


namespace NUMINAMATH_CALUDE_min_coins_for_distribution_l3401_340154

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 70) :
  min_additional_coins friends initial_coins = 50 := by
  sorry

#eval min_additional_coins 15 70

end NUMINAMATH_CALUDE_min_coins_for_distribution_l3401_340154


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3401_340175

theorem trig_expression_equality : 
  (1 + Real.cos (20 * π / 180)) / (2 * Real.sin (20 * π / 180)) - 
  Real.sin (10 * π / 180) * ((1 / Real.tan (5 * π / 180)) - Real.tan (5 * π / 180)) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3401_340175


namespace NUMINAMATH_CALUDE_f_passes_through_six_zero_f_vertex_at_four_neg_eight_l3401_340138

/-- A quadratic function passing through (6, 0) with vertex at (4, -8) -/
def f (x : ℝ) : ℝ := 2 * (x - 4)^2 - 8

/-- The function f passes through the point (6, 0) -/
theorem f_passes_through_six_zero : f 6 = 0 := by sorry

/-- The vertex of f is at (4, -8) -/
theorem f_vertex_at_four_neg_eight :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = a * (x - 4)^2 - 8) ∧
  (∀ (x : ℝ), f x ≥ f 4) := by sorry

end NUMINAMATH_CALUDE_f_passes_through_six_zero_f_vertex_at_four_neg_eight_l3401_340138


namespace NUMINAMATH_CALUDE_distance_between_squares_l3401_340196

/-- Given two squares, one with perimeter 8 cm and another with area 36 cm²,
    prove that the distance between opposite corners is √80 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ)
    (h1 : small_square_perimeter = 8)
    (h2 : large_square_area = 36) :
    Real.sqrt ((small_square_perimeter / 4 + Real.sqrt large_square_area) ^ 2 +
               (Real.sqrt large_square_area - small_square_perimeter / 4) ^ 2) = Real.sqrt 80 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_squares_l3401_340196


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3401_340127

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

/-- Theorem: The opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3401_340127


namespace NUMINAMATH_CALUDE_train_speed_calculation_train_speed_proof_l3401_340115

/-- The speed of two trains crossing each other -/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := 2 * train_length
  let relative_speed := total_distance / crossing_time
  let train_speed := relative_speed / 2
  let km_per_hour := train_speed * 3.6
  km_per_hour

/-- Proof that the speed of each train is approximately 12.01 km/hr -/
theorem train_speed_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_speed_calculation 120 36 - 12.01| < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_train_speed_proof_l3401_340115


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3401_340105

theorem arctan_equation_solution (y : ℝ) :
  4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 →
  y = 1251 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3401_340105


namespace NUMINAMATH_CALUDE_integer_triple_problem_l3401_340167

theorem integer_triple_problem (a b c : ℤ) :
  let N := ((a - b) * (b - c) * (c - a)) / 2 + 2
  (∃ m : ℕ, N = 1729^m ∧ N > 0) →
  ∃ k : ℤ, a = k + 2 ∧ b = k + 1 ∧ c = k :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_problem_l3401_340167


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3401_340184

/-- Given a configuration of four congruent rectangles arranged around an inner square,
    where the area of the outer square is 9 times the area of the inner square,
    the ratio of the longer side to the shorter side of each rectangle is 2. -/
theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) (h5 : x + s = 3*s) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3401_340184


namespace NUMINAMATH_CALUDE_mike_work_hours_l3401_340164

def wash_time : ℕ := 10
def oil_change_time : ℕ := 15
def tire_change_time : ℕ := 30
def paint_time : ℕ := 45
def engine_service_time : ℕ := 60

def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def tire_sets_changed : ℕ := 2
def cars_painted : ℕ := 4
def engines_serviced : ℕ := 3

def total_minutes : ℕ := 
  wash_time * cars_washed + 
  oil_change_time * cars_oil_changed + 
  tire_change_time * tire_sets_changed + 
  paint_time * cars_painted + 
  engine_service_time * engines_serviced

theorem mike_work_hours : total_minutes / 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mike_work_hours_l3401_340164


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3401_340148

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the complement of B
def complement_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_complement_B : A ∩ complement_B = {x | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3401_340148


namespace NUMINAMATH_CALUDE_john_average_score_l3401_340151

def john_scores : List ℝ := [88, 95, 90, 84, 91]

theorem john_average_score : (john_scores.sum / john_scores.length) = 89.6 := by
  sorry

end NUMINAMATH_CALUDE_john_average_score_l3401_340151


namespace NUMINAMATH_CALUDE_cubic_expression_equals_three_l3401_340157

theorem cubic_expression_equals_three (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) 
  (h4 : p + 2*q + 3*r = 0) : (p^3 + 2*q^3 + 3*r^3) / (p*q*r) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equals_three_l3401_340157


namespace NUMINAMATH_CALUDE_min_score_for_maria_l3401_340191

def min_score_for_advanced_class (scores : List ℚ) (required_average : ℚ) : ℚ :=
  let total_terms := 5
  let current_sum := scores.sum
  max ((required_average * total_terms) - current_sum) 0

theorem min_score_for_maria : 
  min_score_for_advanced_class [84/100, 80/100, 82/100, 83/100] (85/100) = 96/100 := by
  sorry

end NUMINAMATH_CALUDE_min_score_for_maria_l3401_340191


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3401_340181

/-- Proves that a train with given length crossing a bridge with given length in a given time has a specific speed. -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 275 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3401_340181


namespace NUMINAMATH_CALUDE_octagon_dissection_and_reassembly_l3401_340117

/-- Represents a regular octagon -/
structure RegularOctagon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a section of a regular octagon -/
structure OctagonSection where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if two OctagonSections are similar -/
def are_similar (s1 s2 : OctagonSection) : Prop :=
  sorry

/-- Checks if two RegularOctagons are congruent -/
def are_congruent (o1 o2 : RegularOctagon) : Prop :=
  sorry

/-- Represents the dissection of a RegularOctagon into OctagonSections -/
def dissect (o : RegularOctagon) : List OctagonSection :=
  sorry

/-- Represents the reassembly of OctagonSections into RegularOctagons -/
def reassemble (sections : List OctagonSection) : List RegularOctagon :=
  sorry

theorem octagon_dissection_and_reassembly 
  (o : RegularOctagon) : 
  let sections := dissect o
  ∃ (reassembled : List RegularOctagon),
    (reassembled = reassemble sections) ∧ 
    (sections.length = 8) ∧
    (∀ (s1 s2 : OctagonSection), s1 ∈ sections → s2 ∈ sections → are_similar s1 s2) ∧
    (reassembled.length = 8) ∧
    (∀ (o1 o2 : RegularOctagon), o1 ∈ reassembled → o2 ∈ reassembled → are_congruent o1 o2) :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_dissection_and_reassembly_l3401_340117


namespace NUMINAMATH_CALUDE_water_height_in_conical_tank_l3401_340169

/-- The height of water in an inverted conical tank -/
theorem water_height_in_conical_tank 
  (tank_radius : ℝ) 
  (tank_height : ℝ) 
  (water_volume_percentage : ℝ) 
  (h : water_volume_percentage = 0.4) 
  (r : tank_radius = 10) 
  (h : tank_height = 60) : 
  ∃ (water_height : ℝ), water_height = 12 * (3 ^ (1/3 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_water_height_in_conical_tank_l3401_340169


namespace NUMINAMATH_CALUDE_joan_remaining_books_l3401_340102

/-- Given an initial number of books and a number of books sold, 
    calculate the remaining number of books. -/
def remaining_books (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: Given 33 initial books and 26 books sold, 
    the remaining number of books is 7. -/
theorem joan_remaining_books :
  remaining_books 33 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_books_l3401_340102


namespace NUMINAMATH_CALUDE_profit_percentage_per_item_l3401_340190

theorem profit_percentage_per_item (total_cost : ℝ) (num_bought num_sold : ℕ) 
  (h1 : num_bought = 30)
  (h2 : num_sold = 20)
  (h3 : total_cost > 0)
  (h4 : num_bought > num_sold)
  (h5 : num_sold * (total_cost / num_bought) = total_cost) :
  (((total_cost / num_sold) - (total_cost / num_bought)) / (total_cost / num_bought)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_per_item_l3401_340190


namespace NUMINAMATH_CALUDE_elementary_classes_count_l3401_340112

/-- The number of elementary school classes in each school -/
def elementary_classes : ℕ := 4

/-- The number of schools -/
def num_schools : ℕ := 2

/-- The number of middle school classes in each school -/
def middle_classes : ℕ := 5

/-- The number of soccer balls donated per class -/
def balls_per_class : ℕ := 5

/-- The total number of soccer balls donated -/
def total_balls : ℕ := 90

theorem elementary_classes_count :
  elementary_classes * num_schools * balls_per_class +
  middle_classes * num_schools * balls_per_class = total_balls :=
by sorry

end NUMINAMATH_CALUDE_elementary_classes_count_l3401_340112


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_3_4_l3401_340197

/-- Given an angle α where a point on its terminal side has coordinates (3,4), prove that sin α = 4/5 -/
theorem sin_alpha_for_point_3_4 (α : Real) :
  (∃ (x y : Real), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_3_4_l3401_340197


namespace NUMINAMATH_CALUDE_relationship_abc_l3401_340188

theorem relationship_abc (a b c : ℝ) : 
  (∃ x y : ℝ, x + y = a ∧ x^2 + y^2 = b ∧ x^3 + y^3 = c) → 
  a^3 - 3*a*b + 2*c = 0 := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l3401_340188


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3401_340104

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((28457 + y) % 37 = 0 ∧ (28457 + y) % 59 = 0 ∧ (28457 + y) % 67 = 0)) ∧
  (28457 + x) % 37 = 0 ∧ (28457 + x) % 59 = 0 ∧ (28457 + x) % 67 = 0 →
  x = 117804 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3401_340104


namespace NUMINAMATH_CALUDE_complementary_angle_measure_l3401_340145

/-- Given two complementary angles A and B, where the measure of A is 3 times the measure of B,
    prove that the measure of angle A is 67.5° -/
theorem complementary_angle_measure (A B : ℝ) : 
  A + B = 90 →  -- angles A and B are complementary
  A = 3 * B →   -- measure of A is 3 times measure of B
  A = 67.5 :=   -- measure of A is 67.5°
by sorry

end NUMINAMATH_CALUDE_complementary_angle_measure_l3401_340145


namespace NUMINAMATH_CALUDE_discount_calculation_l3401_340101

theorem discount_calculation (price_per_person : ℕ) (num_people : ℕ) (total_cost_with_discount : ℕ) 
  (h1 : price_per_person = 147)
  (h2 : num_people = 2)
  (h3 : total_cost_with_discount = 266) :
  (price_per_person * num_people - total_cost_with_discount) / num_people = 14 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3401_340101


namespace NUMINAMATH_CALUDE_garden_cut_percentage_l3401_340140

theorem garden_cut_percentage (rows : ℕ) (flowers_per_row : ℕ) (remaining : ℕ) :
  rows = 50 →
  flowers_per_row = 400 →
  remaining = 8000 →
  (rows * flowers_per_row - remaining : ℚ) / (rows * flowers_per_row) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_cut_percentage_l3401_340140


namespace NUMINAMATH_CALUDE_base_prime_representation_441_l3401_340159

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ := sorry

/-- The list of primes up to a given number -/
def PrimesUpTo (n : ℕ) : List ℕ := sorry

theorem base_prime_representation_441 :
  let n := 441
  let primes := PrimesUpTo 7
  BasePrimeRepresentation n = [0, 2, 2, 0] ∧ 
  n = 3^2 * 7^2 ∧
  primes = [2, 3, 5, 7] := by sorry

end NUMINAMATH_CALUDE_base_prime_representation_441_l3401_340159
