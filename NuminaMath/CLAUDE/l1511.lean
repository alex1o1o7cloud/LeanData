import Mathlib

namespace NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l1511_151158

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 6

/-- The number of times we want to get an odd number -/
def target_odd : ℕ := 5

/-- The probability of getting exactly 'target_odd' odd numbers in 'num_rolls' rolls -/
def prob_target_odd : ℚ :=
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd ^ target_odd * (1 - prob_odd) ^ (num_rolls - target_odd)

theorem prob_five_odd_in_six_rolls : prob_target_odd = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l1511_151158


namespace NUMINAMATH_CALUDE_digit_difference_of_82_l1511_151139

theorem digit_difference_of_82 :
  let n : ℕ := 82
  let tens : ℕ := n / 10
  let ones : ℕ := n % 10
  (tens + ones = 10) → (tens - ones = 6) := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_of_82_l1511_151139


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_three_l1511_151179

def A : Set ℝ := {x | |x - 2| < 1}
def B (a : ℝ) : Set ℝ := {y | ∃ x, y = -x^2 + a}

theorem subset_implies_a_geq_three (a : ℝ) (h : A ⊆ B a) : a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_three_l1511_151179


namespace NUMINAMATH_CALUDE_binomial_expected_value_l1511_151176

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ    -- number of trials
  p : ℝ    -- probability of success
  h1 : 0 ≤ p ∧ p ≤ 1  -- probability is between 0 and 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Theorem: The expected value of ξ ~ B(6, 1/3) is 2 -/
theorem binomial_expected_value :
  ∀ ξ : BinomialRV, ξ.n = 6 ∧ ξ.p = 1/3 → expected_value ξ = 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expected_value_l1511_151176


namespace NUMINAMATH_CALUDE_coefficient_x4_expansion_l1511_151131

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem coefficient_x4_expansion :
  let n : ℕ := 8
  let k : ℕ := 4
  let a : ℝ := 1
  let b : ℝ := 3 * Real.sqrt 3
  binomial_coefficient n k * a^(n-k) * b^k = 51030 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_expansion_l1511_151131


namespace NUMINAMATH_CALUDE_f_properties_l1511_151174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + 1 / (a^x)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = f a x) ∧
  (∀ x y : ℝ, 0 ≤ x → x < y → f a x < f a y) ∧
  (∀ x y : ℝ, x < y → y ≤ 0 → f a x > f a y) ∧
  (Set.Ioo (-2 : ℝ) 0 = {x : ℝ | f a (x - 1) > f a (2*x + 1)}) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1511_151174


namespace NUMINAMATH_CALUDE_increasing_linear_function_not_in_fourth_quadrant_l1511_151134

/-- A linear function that passes through (-2, 0) and increases with x -/
structure IncreasingLinearFunction where
  k : ℝ
  b : ℝ
  k_neq_zero : k ≠ 0
  passes_through_neg_two_zero : 0 = -2 * k + b
  increasing : k > 0

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 < 0}

/-- The graph of a linear function -/
def graph (f : IncreasingLinearFunction) : Set (ℝ × ℝ) :=
  {p | p.2 = f.k * p.1 + f.b}

theorem increasing_linear_function_not_in_fourth_quadrant (f : IncreasingLinearFunction) :
  graph f ∩ fourth_quadrant = ∅ :=
sorry

end NUMINAMATH_CALUDE_increasing_linear_function_not_in_fourth_quadrant_l1511_151134


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_l1511_151191

/-- Given a rectangle ABCD with AB = c and AD = d, and points E on AB and F on AD
    such that AE = AF = x, the maximum area of quadrilateral CDFE is (c + d)^2 / 8. -/
theorem max_area_quadrilateral (c d : ℝ) (h_c : c > 0) (h_d : d > 0) :
  ∃ x : ℝ, 0 < x ∧ x < min c d ∧
    ∀ y : ℝ, 0 < y ∧ y < min c d →
      x * (c + d - 2*x) / 2 ≥ y * (c + d - 2*y) / 2 ∧
      x * (c + d - 2*x) / 2 = (c + d)^2 / 8 :=
by sorry


end NUMINAMATH_CALUDE_max_area_quadrilateral_l1511_151191


namespace NUMINAMATH_CALUDE_fencing_final_probability_l1511_151157

theorem fencing_final_probability (p_a : ℝ) (h1 : p_a = 0.41) :
  let p_b := 1 - p_a
  p_b = 0.59 := by
sorry

end NUMINAMATH_CALUDE_fencing_final_probability_l1511_151157


namespace NUMINAMATH_CALUDE_move_up_coordinates_l1511_151132

/-- Moving a point up in a 2D coordinate system -/
def move_up (p : ℝ × ℝ) (n : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + n)

/-- The theorem states that moving a point up by n units results in the expected coordinates -/
theorem move_up_coordinates (x y n : ℝ) :
  move_up (x, y) n = (x, y + n) := by
  sorry

end NUMINAMATH_CALUDE_move_up_coordinates_l1511_151132


namespace NUMINAMATH_CALUDE_box_triples_count_l1511_151123

/-- The number of ordered triples (a, b, c) satisfying the box conditions -/
def box_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    a ≤ b ∧ b ≤ c ∧ 2 * a * b * c = 2 * a * b + 2 * b * c + 2 * a * c)
    (Finset.product (Finset.range 100) (Finset.product (Finset.range 100) (Finset.range 100)))).card

/-- The main theorem stating that there are exactly 10 ordered triples satisfying the conditions -/
theorem box_triples_count : box_triples = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_triples_count_l1511_151123


namespace NUMINAMATH_CALUDE_wine_barrel_system_l1511_151184

/-- Represents the capacity of a large barrel in hu -/
def large_barrel_capacity : ℝ := sorry

/-- Represents the capacity of a small barrel in hu -/
def small_barrel_capacity : ℝ := sorry

/-- The total capacity of 6 large barrels and 4 small barrels is 48 hu -/
axiom first_equation : 6 * large_barrel_capacity + 4 * small_barrel_capacity = 48

/-- The total capacity of 5 large barrels and 3 small barrels is 38 hu -/
axiom second_equation : 5 * large_barrel_capacity + 3 * small_barrel_capacity = 38

/-- The system of equations representing the wine barrel problem -/
theorem wine_barrel_system :
  (6 * large_barrel_capacity + 4 * small_barrel_capacity = 48) ∧
  (5 * large_barrel_capacity + 3 * small_barrel_capacity = 38) := by
  sorry

end NUMINAMATH_CALUDE_wine_barrel_system_l1511_151184


namespace NUMINAMATH_CALUDE_min_value_theorem_l1511_151117

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_gm : 2 = Real.sqrt (4^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 2 = Real.sqrt (4^x * 2^y) → 2/x + 1/y ≥ 2/a + 1/b) → 
  2/a + 1/b = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1511_151117


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1511_151133

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three :
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1511_151133


namespace NUMINAMATH_CALUDE_khali_snow_volume_l1511_151148

/-- The volume of snow on Khali's driveway -/
def snow_volume (length width height : ℚ) : ℚ := length * width * height

theorem khali_snow_volume :
  snow_volume 30 4 (3/4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_khali_snow_volume_l1511_151148


namespace NUMINAMATH_CALUDE_composite_sum_product_l1511_151105

theorem composite_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a^2 + a*c - c^2 = b^2 + b*d - d^2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a*b + c*d = x*y :=
sorry

end NUMINAMATH_CALUDE_composite_sum_product_l1511_151105


namespace NUMINAMATH_CALUDE_base_equation_solution_l1511_151170

/-- Represents a number in a given base -/
def to_base (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Consecutive even positive integers -/
def consecutive_even (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ Even x ∧ Even y ∧ y = x + 2

theorem base_equation_solution (X Y : ℕ) :
  consecutive_even X Y →
  to_base 241 X + to_base 36 Y = to_base 94 (X + Y) →
  X + Y = 22 := by sorry

end NUMINAMATH_CALUDE_base_equation_solution_l1511_151170


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l1511_151115

def is_eventually_periodic (a : ℕ → ℚ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ ∀ n ≥ m, a (n + k) = a n

theorem sequence_eventually_periodic
  (a : ℕ → ℚ)
  (h1 : ∀ n : ℕ, |a (n + 1) - 2 * a n| = 2)
  (h2 : ∀ n : ℕ, |a n| ≤ 2)
  : is_eventually_periodic a :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l1511_151115


namespace NUMINAMATH_CALUDE_possible_x_value_for_simplest_radical_l1511_151159

/-- A number is a simplest quadratic radical if it's of the form √n where n is a positive integer
    and not a perfect square. -/
def is_simplest_quadratic_radical (n : ℝ) : Prop :=
  ∃ (m : ℕ), n = Real.sqrt m ∧ ¬ ∃ (k : ℕ), m = k^2

/-- The proposition states that 2 is a possible value for x that makes √(x+3) 
    the simplest quadratic radical. -/
theorem possible_x_value_for_simplest_radical : 
  ∃ (x : ℝ), is_simplest_quadratic_radical (Real.sqrt (x + 3)) ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_possible_x_value_for_simplest_radical_l1511_151159


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_l1511_151192

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4444^4444 -/
def A : ℕ := sum_of_digits (4444^4444)

/-- B is the sum of digits of A -/
def B : ℕ := sum_of_digits A

/-- Theorem: The sum of digits of B is 7 -/
theorem sum_of_digits_of_B : sum_of_digits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_l1511_151192


namespace NUMINAMATH_CALUDE_car_truck_sales_l1511_151177

theorem car_truck_sales (total_vehicles : ℕ) (car_truck_difference : ℕ) : 
  total_vehicles = 69 → car_truck_difference = 27 → 
  ∃ (trucks : ℕ), trucks = 21 ∧ trucks + (trucks + car_truck_difference) = total_vehicles := by
sorry

end NUMINAMATH_CALUDE_car_truck_sales_l1511_151177


namespace NUMINAMATH_CALUDE_floor_area_from_partial_coverage_l1511_151181

/-- The total area of a floor given a carpet covering a known percentage -/
theorem floor_area_from_partial_coverage (carpet_area : ℝ) (coverage_percentage : ℝ) 
  (h1 : carpet_area = 36) 
  (h2 : coverage_percentage = 0.45) : 
  carpet_area / coverage_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_floor_area_from_partial_coverage_l1511_151181


namespace NUMINAMATH_CALUDE_total_sides_is_75_l1511_151144

/-- Represents the number of sides for each shape --/
def sides_of_shape (shape : String) : ℕ :=
  match shape with
  | "triangle" => 3
  | "square" => 4
  | "hexagon" => 6
  | "octagon" => 8
  | "circle" => 0
  | "pentagon" => 5
  | _ => 0

/-- Calculates the total number of sides for a given shape and quantity --/
def total_sides (shape : String) (quantity : ℕ) : ℕ :=
  (sides_of_shape shape) * quantity

/-- Represents the cookie cutter drawer --/
structure CookieCutterDrawer :=
  (top_layer : ℕ)
  (middle_layer_squares : ℕ)
  (middle_layer_hexagons : ℕ)
  (bottom_layer_octagons : ℕ)
  (bottom_layer_circles : ℕ)
  (bottom_layer_pentagons : ℕ)

/-- Calculates the total number of sides for all cookie cutters in the drawer --/
def total_sides_in_drawer (drawer : CookieCutterDrawer) : ℕ :=
  total_sides "triangle" drawer.top_layer +
  total_sides "square" drawer.middle_layer_squares +
  total_sides "hexagon" drawer.middle_layer_hexagons +
  total_sides "octagon" drawer.bottom_layer_octagons +
  total_sides "circle" drawer.bottom_layer_circles +
  total_sides "pentagon" drawer.bottom_layer_pentagons

/-- The cookie cutter drawer described in the problem --/
def emery_drawer : CookieCutterDrawer :=
  { top_layer := 6,
    middle_layer_squares := 4,
    middle_layer_hexagons := 2,
    bottom_layer_octagons := 3,
    bottom_layer_circles := 5,
    bottom_layer_pentagons := 1 }

theorem total_sides_is_75 :
  total_sides_in_drawer emery_drawer = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_is_75_l1511_151144


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1511_151187

theorem x_minus_y_value (x y : ℤ) (h1 : x + y = 4) (h2 : x = 20) : x - y = 36 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1511_151187


namespace NUMINAMATH_CALUDE_odd_function_symmetry_symmetric_about_one_period_four_l1511_151185

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Statement 1
theorem odd_function_symmetry (h : ∀ x, f x = -f (-x)) :
  ∀ x, f (x - 1) = -f (-x + 1) :=
sorry

-- Statement 2
theorem symmetric_about_one (h : ∀ x, f (x - 1) = f (x + 1)) :
  ∀ x, f (1 - x) = f (1 + x) :=
sorry

-- Statement 4
theorem period_four (h1 : ∀ x, f (x + 1) = f (1 - x)) 
                    (h2 : ∀ x, f (x + 3) = f (3 - x)) :
  ∀ x, f x = f (x + 4) :=
sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_symmetric_about_one_period_four_l1511_151185


namespace NUMINAMATH_CALUDE_limit_proof_l1511_151109

theorem limit_proof (ε : ℝ) (hε : ε > 0) : 
  ∃ δ : ℝ, δ > 0 ∧ 
  ∀ x : ℝ, 0 < |x - 11| → |x - 11| < δ → 
  |(2*x^2 - 21*x - 11) / (x - 11) - 23| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_proof_l1511_151109


namespace NUMINAMATH_CALUDE_intersection_and_subset_condition_l1511_151108

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 4}
def N (a : ℝ) : Set ℝ := {x : ℝ | x ≤ 2*a - 5}

theorem intersection_and_subset_condition :
  (∀ x : ℝ, x ∈ M ∩ N 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧
  (∀ a : ℝ, M ⊆ N a ↔ a ≥ 9/2) := by sorry

end NUMINAMATH_CALUDE_intersection_and_subset_condition_l1511_151108


namespace NUMINAMATH_CALUDE_sum_differences_theorem_l1511_151138

def numeral1 : ℕ := 987348621829
def numeral2 : ℕ := 74693251

def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def face_value (digit : ℕ) : ℕ := digit

def difference_local_face (digit : ℕ) (position : ℕ) : ℕ :=
  local_value digit position - face_value digit

theorem sum_differences_theorem : 
  let first_8_pos := 8
  let second_8_pos := 1
  let seven_pos := 7
  (difference_local_face 8 first_8_pos + difference_local_face 8 second_8_pos) * 
  difference_local_face 7 seven_pos = 55999994048000192 := by
sorry

end NUMINAMATH_CALUDE_sum_differences_theorem_l1511_151138


namespace NUMINAMATH_CALUDE_problem_solution_l1511_151120

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2^x + a*x

theorem problem_solution (a : ℝ) : f a (f a 1) = 4 * a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1511_151120


namespace NUMINAMATH_CALUDE_difference_of_squares_l1511_151175

theorem difference_of_squares : 601^2 - 597^2 = 4792 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1511_151175


namespace NUMINAMATH_CALUDE_sine_theorem_l1511_151167

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β)
  (h2 : b / Real.sin β = c / Real.sin γ)
  (h3 : α + β + γ = Real.pi) : 
  (a = b * Real.cos γ + c * Real.cos β) ∧ 
  (b = c * Real.cos α + a * Real.cos γ) ∧ 
  (c = a * Real.cos β + b * Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_sine_theorem_l1511_151167


namespace NUMINAMATH_CALUDE_playlist_additional_time_l1511_151162

/-- Given a playlist with 3-minute and 2-minute songs, calculate the additional time needed to cover a run. -/
theorem playlist_additional_time (three_min_songs two_min_songs run_time : ℕ) :
  three_min_songs = 10 →
  two_min_songs = 15 →
  run_time = 100 →
  run_time - (three_min_songs * 3 + two_min_songs * 2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_playlist_additional_time_l1511_151162


namespace NUMINAMATH_CALUDE_sine_square_sum_condition_l1511_151128

theorem sine_square_sum_condition (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) : 
  (Real.sin α)^2 + (Real.sin β)^2 = (Real.sin (α + β))^2 ↔ α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_square_sum_condition_l1511_151128


namespace NUMINAMATH_CALUDE_parabola_translation_original_to_result_l1511_151183

/-- Represents a parabola in the form y = (x - h)^2 + k, where (h, k) is the vertex --/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { h := p.h - dx, k := p.k + dy }

theorem parabola_translation (p : Parabola) (dx dy : ℝ) :
  translate p dx dy = { h := p.h - dx, k := p.k + dy } := by sorry

theorem original_to_result :
  let original := Parabola.mk 2 (-8)
  let result := translate original 3 5
  result = Parabola.mk (-1) (-3) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_original_to_result_l1511_151183


namespace NUMINAMATH_CALUDE_impossible_filling_l1511_151145

/-- Represents a 7 × 3 table filled with 0s and 1s -/
def Table := Fin 7 → Fin 3 → Bool

/-- Checks if a 2 × 2 submatrix in the table has all the same values -/
def has_same_2x2_submatrix (t : Table) : Prop :=
  ∃ (i j : Fin 7) (k l : Fin 3), i < j ∧ k < l ∧
    t i k = t i l ∧ t i k = t j k ∧ t i k = t j l

/-- Theorem stating that any 7 × 3 table filled with 0s and 1s
    always has a 2 × 2 submatrix with all the same values -/
theorem impossible_filling :
  ∀ (t : Table), has_same_2x2_submatrix t :=
sorry

end NUMINAMATH_CALUDE_impossible_filling_l1511_151145


namespace NUMINAMATH_CALUDE_negation_equivalence_l1511_151166

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 - 2*x > 0) ↔ (∀ x : ℝ, x < 0 → x^2 - 2*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1511_151166


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1511_151186

-- Problem 1
theorem factorization_problem_1 (x : ℝ) :
  x^4 - 16 = (x-2)*(x+2)*(x^2+4) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  -9*x^2*y + 12*x*y^2 - 4*y^3 = -y*(3*x-2*y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1511_151186


namespace NUMINAMATH_CALUDE_cos_negative_third_quadrants_l1511_151151

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to determine the possible quadrants for a given cosine value
def possibleQuadrants (cosθ : ℝ) : Set Quadrant :=
  if cosθ > 0 then {Quadrant.First, Quadrant.Fourth}
  else if cosθ < 0 then {Quadrant.Second, Quadrant.Third}
  else {Quadrant.First, Quadrant.Second, Quadrant.Third, Quadrant.Fourth}

-- Theorem statement
theorem cos_negative_third_quadrants :
  let cosθ : ℝ := -1/3
  possibleQuadrants cosθ = {Quadrant.Second, Quadrant.Third} :=
by sorry


end NUMINAMATH_CALUDE_cos_negative_third_quadrants_l1511_151151


namespace NUMINAMATH_CALUDE_students_walking_home_l1511_151140

theorem students_walking_home (total : ℚ) (bus : ℚ) (auto : ℚ) (bike : ℚ) (walk : ℚ) : 
  bus = 1/3 * total → auto = 1/5 * total → bike = 1/15 * total → 
  walk = total - (bus + auto + bike) →
  walk = 2/5 * total :=
sorry

end NUMINAMATH_CALUDE_students_walking_home_l1511_151140


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1511_151163

theorem arctan_equation_solution (x : ℝ) :
  3 * Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/x) = π/4 →
  x = 34/13 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1511_151163


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l1511_151160

theorem cubic_root_sum_squares (p q r : ℝ) : 
  (p^3 - 18*p^2 + 40*p - 15 = 0) →
  (q^3 - 18*q^2 + 40*q - 15 = 0) →
  (r^3 - 18*r^2 + 40*r - 15 = 0) →
  (p + q + r = 18) →
  (p*q + q*r + r*p = 40) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 568 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l1511_151160


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l1511_151172

theorem multiples_of_four_between_100_and_300 :
  (Finset.filter (fun n => n % 4 = 0) (Finset.range 300 \ Finset.range 101)).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l1511_151172


namespace NUMINAMATH_CALUDE_units_digit_of_1505_odd_squares_sum_l1511_151141

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitsDigitOfOddSquaresSum (n : ℕ) : ℕ :=
  (n / 5 * 5) % 10

theorem units_digit_of_1505_odd_squares_sum :
  unitsDigitOfOddSquaresSum 1505 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_1505_odd_squares_sum_l1511_151141


namespace NUMINAMATH_CALUDE_john_apartment_number_l1511_151100

/-- Represents a skyscraper with 10 apartments on each floor. -/
structure Skyscraper where
  /-- John's apartment number -/
  john_apartment : ℕ
  /-- Mary's apartment number -/
  mary_apartment : ℕ
  /-- John's floor number -/
  john_floor : ℕ

/-- 
Given a skyscraper with 10 apartments on each floor, 
if John's floor number is equal to Mary's apartment number 
and the sum of their apartment numbers is 239, 
then John lives in apartment 217.
-/
theorem john_apartment_number (s : Skyscraper) : 
  s.john_floor = s.mary_apartment → 
  s.john_apartment + s.mary_apartment = 239 → 
  s.john_apartment = 217 := by
sorry

end NUMINAMATH_CALUDE_john_apartment_number_l1511_151100


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_condition_l1511_151169

theorem quadratic_two_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_condition_l1511_151169


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l1511_151119

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's problem -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ)
  (h1 : friends = 15)
  (h2 : initial_coins = 105) :
  min_additional_coins friends initial_coins = 15 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l1511_151119


namespace NUMINAMATH_CALUDE_train_length_calculation_l1511_151146

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length_calculation 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 480) 
  (h2 : crossing_time = 55) 
  (h3 : train_speed = 39.27272727272727) : 
  train_speed * crossing_time - bridge_length = 1680 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1511_151146


namespace NUMINAMATH_CALUDE_no_real_solutions_l1511_151110

theorem no_real_solutions :
  ∀ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1511_151110


namespace NUMINAMATH_CALUDE_exactly_two_triples_l1511_151147

/-- Least common multiple of two positive integers -/
def lcm (r s : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

/-- Theorem stating that there are exactly 2 ordered triples satisfying the conditions -/
theorem exactly_two_triples : 
  count_triples = 2 ∧ 
  ∀ a b c : ℕ+, 
    (lcm a b = 1250 ∧ lcm b c = 2500 ∧ lcm c a = 2500) → 
    (a, b, c) ∈ {x | count_triples > 0} :=
sorry

end NUMINAMATH_CALUDE_exactly_two_triples_l1511_151147


namespace NUMINAMATH_CALUDE_greatest_four_digit_sum_15_l1511_151106

/-- A function that returns true if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digits of the greatest four-digit number
    with digit product 36 is 15 -/
theorem greatest_four_digit_sum_15 :
  ∃ M : ℕ, is_four_digit M ∧ 
           digit_product M = 36 ∧ 
           (∀ n : ℕ, is_four_digit n → digit_product n = 36 → n ≤ M) ∧
           digit_sum M = 15 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_sum_15_l1511_151106


namespace NUMINAMATH_CALUDE_prob_exactly_two_prob_at_least_one_l1511_151164

/-- The probability of exactly two out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_exactly_two (p1 p2 p3 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3 = 
    0.398 ↔ p1 = 0.8 ∧ p2 = 0.7 ∧ p3 = 0.9 :=
sorry

/-- The probability of at least one out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_at_least_one (p1 p2 p3 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 
    0.994 ↔ p1 = 0.8 ∧ p2 = 0.7 ∧ p3 = 0.9 :=
sorry

end NUMINAMATH_CALUDE_prob_exactly_two_prob_at_least_one_l1511_151164


namespace NUMINAMATH_CALUDE_commission_problem_l1511_151154

/-- Calculates the total sales amount given the commission rate and commission amount -/
def calculateTotalSales (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem: Given a commission rate of 5% and a commission amount of 12.50, the total sales amount is 250 -/
theorem commission_problem :
  let commissionRate : ℚ := 5
  let commissionAmount : ℚ := 12.50
  calculateTotalSales commissionRate commissionAmount = 250 := by
  sorry

end NUMINAMATH_CALUDE_commission_problem_l1511_151154


namespace NUMINAMATH_CALUDE_function_from_derivative_and_point_l1511_151153

/-- Given a function f: ℝ → ℝ with f'(x) = 4x³ for all x and f(1) = -1, 
    prove that f(x) = x⁴ - 2 for all x ∈ ℝ -/
theorem function_from_derivative_and_point (f : ℝ → ℝ) 
    (h1 : ∀ x, deriv f x = 4 * x^3)
    (h2 : f 1 = -1) :
    ∀ x, f x = x^4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_from_derivative_and_point_l1511_151153


namespace NUMINAMATH_CALUDE_classroom_notebooks_l1511_151178

theorem classroom_notebooks (total_students : ℕ) 
  (notebooks_group1 : ℕ) (notebooks_group2 : ℕ) : 
  total_students = 28 →
  notebooks_group1 = 5 →
  notebooks_group2 = 3 →
  (total_students / 2 * notebooks_group1 + total_students / 2 * notebooks_group2) = 112 := by
  sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l1511_151178


namespace NUMINAMATH_CALUDE_total_blue_balloons_l1511_151143

theorem total_blue_balloons (joan_balloons melanie_balloons john_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : melanie_balloons = 41)
  (h3 : john_balloons = 55) :
  joan_balloons + melanie_balloons + john_balloons = 136 := by
sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l1511_151143


namespace NUMINAMATH_CALUDE_area_of_constrained_region_l1511_151104

/-- The area of the region defined by specific constraints in a coordinate plane --/
theorem area_of_constrained_region : 
  let S := {p : ℝ × ℝ | p.1 ≤ 0 ∧ p.2 + p.1 - 1 ≥ 0 ∧ p.2 ≤ 4}
  MeasureTheory.volume S = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_of_constrained_region_l1511_151104


namespace NUMINAMATH_CALUDE_f_properties_l1511_151180

/-- The function f(x) = 2^x / (2^x + 1) + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

/-- Main theorem about the properties of f -/
theorem f_properties (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) → a = -1/2) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) → 
    (∀ x k : ℝ, f a (x^2 - 2*x) + f a (2*x^2 - k) > 0 → k < -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1511_151180


namespace NUMINAMATH_CALUDE_inverse_of_periodic_function_l1511_151137

def PeriodicFunction (f : ℝ → ℝ) :=
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x

def SmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) :=
  PeriodicFunction f ∧ T > 0 ∧ ∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S

def InverseInInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ g : ℝ → ℝ, ∀ x ∈ Set.Ioo a b, g (f x) = x ∧ f (g x) = x

theorem inverse_of_periodic_function
  (f : ℝ → ℝ) (T : ℝ)
  (h_periodic : SmallestPositivePeriod f T)
  (h_inverse : InverseInInterval f 0 T) :
  ∃ g : ℝ → ℝ, ∀ x ∈ Set.Ioo T (2 * T),
    g (f x) = x ∧ f (g x) = x ∧ g x = (Classical.choose h_inverse) (x - T) + T :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_periodic_function_l1511_151137


namespace NUMINAMATH_CALUDE_karens_order_cost_l1511_151155

/-- The cost of Karen's fast-food order -/
def fast_food_order_cost (burger_cost sandwich_cost smoothie_cost : ℕ) 
  (burger_quantity sandwich_quantity smoothie_quantity : ℕ) : ℕ :=
  burger_cost * burger_quantity + sandwich_cost * sandwich_quantity + smoothie_cost * smoothie_quantity

/-- Theorem stating that Karen's fast-food order costs $17 -/
theorem karens_order_cost : 
  fast_food_order_cost 5 4 4 1 1 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_karens_order_cost_l1511_151155


namespace NUMINAMATH_CALUDE_parabola_vertex_l1511_151195

/-- The vertex of the parabola y = 1/2 * (x + 1)^2 - 1/2 is (-1, -1/2) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := λ x ↦ (1/2 : ℝ) * (x + 1)^2 - 1/2
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1/2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1511_151195


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l1511_151103

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the theorem
theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ↔ c ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l1511_151103


namespace NUMINAMATH_CALUDE_symmetric_function_property_l1511_151197

def symmetricAround (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f y = x

def symmetricAfterShift (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 1) = y ↔ f y = x

theorem symmetric_function_property (f : ℝ → ℝ)
  (h1 : symmetricAround f)
  (h2 : symmetricAfterShift f)
  (h3 : f 1 = 0) :
  f 2011 = -2010 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_property_l1511_151197


namespace NUMINAMATH_CALUDE_seans_total_spend_is_21_l1511_151136

/-- The total amount Sean spent on his Sunday purchases -/
def seans_total_spend : ℝ :=
  let almond_croissant : ℝ := 4.50
  let salami_cheese_croissant : ℝ := 4.50
  let plain_croissant : ℝ := 3.00
  let focaccia : ℝ := 4.00
  let latte : ℝ := 2.50
  let num_lattes : ℕ := 2

  almond_croissant + salami_cheese_croissant + plain_croissant + focaccia + (num_lattes : ℝ) * latte

/-- Theorem stating that Sean's total spend is $21.00 -/
theorem seans_total_spend_is_21 : seans_total_spend = 21 := by
  sorry

end NUMINAMATH_CALUDE_seans_total_spend_is_21_l1511_151136


namespace NUMINAMATH_CALUDE_alexandra_rearrangement_time_l1511_151107

/-- The number of letters in Alexandra's name -/
def name_length : ℕ := 8

/-- The number of rearrangements Alexandra can write per minute -/
def rearrangements_per_minute : ℕ := 16

/-- Calculate the time required to write all rearrangements in hours -/
def time_to_write_all_rearrangements : ℕ :=
  (Nat.factorial name_length / rearrangements_per_minute) / 60

theorem alexandra_rearrangement_time :
  time_to_write_all_rearrangements = 42 := by sorry

end NUMINAMATH_CALUDE_alexandra_rearrangement_time_l1511_151107


namespace NUMINAMATH_CALUDE_remainder_2519_div_4_l1511_151156

theorem remainder_2519_div_4 : 2519 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_4_l1511_151156


namespace NUMINAMATH_CALUDE_ellipse_properties_l1511_151199

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the condition for line l
def line_l (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 = 0

-- Define the equality of distances condition
def equal_distances (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1)^2 + y1^2 = (x2 - 1)^2 + y2^2

-- Main theorem
theorem ellipse_properties :
  -- Given conditions
  (∃ (x y : ℝ), x = 1/2 ∧ y = Real.sqrt 3 ∧ ellipse_C x y) →
  -- Conclusions
  (∀ (k m x1 y1 x2 y2 : ℝ),
    -- Line l intersects ellipse C at M(x1, y1) and N(x2, y2)
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line_l k m x1 y1 ∧ line_l k m x2 y2 ∧
    -- AM ⊥ AN and |AM| = |AN|
    perpendicular x1 y1 x2 y2 ∧ equal_distances x1 y1 x2 y2 →
    -- Then line l has one of these equations
    (k = Real.sqrt 5 ∧ m = -3/5 * Real.sqrt 5) ∨
    (k = -Real.sqrt 5 ∧ m = -3/5 * Real.sqrt 5) ∨
    (k = 0 ∧ m = -3/5)) ∧
  -- The locus of H
  (∀ (x y : ℝ), x ≠ 1 →
    (∃ (k m x1 y1 x2 y2 : ℝ),
      ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
      line_l k m x1 y1 ∧ line_l k m x2 y2 ∧
      perpendicular x1 y1 x2 y2 ∧
      -- H is on the perpendicular from A to MN
      (y - 0) / (x - 1) = -1 / k) ↔
    (x - 1/5)^2 + y^2 = 16/25) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1511_151199


namespace NUMINAMATH_CALUDE_simplify_expression_l1511_151193

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^7 = 1280 * 16^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1511_151193


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1511_151122

/-- An ellipse represented by the equation x^2 + ky^2 = 2 with foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the condition that foci are on y-axis

/-- The range of k for the given ellipse -/
def k_range (e : Ellipse) : Set ℝ :=
  {k : ℝ | 0 < k ∧ k < 1}

/-- Theorem stating that for the given ellipse, k is in the range (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : e.k ∈ k_range e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1511_151122


namespace NUMINAMATH_CALUDE_complex_norm_squared_l1511_151149

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 5 - (2*I)^2) : 
  Complex.normSq z = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l1511_151149


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l1511_151114

/-- Proves that the charge for each additional 1/5 mile is $0.40 given the initial and total charges -/
theorem taxi_fare_calculation (initial_charge : ℚ) (total_charge : ℚ) (ride_length : ℚ) 
  (h1 : initial_charge = 2.1)
  (h2 : total_charge = 17.7)
  (h3 : ride_length = 8) :
  let additional_increments := (ride_length * 5) - 1
  (total_charge - initial_charge) / additional_increments = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l1511_151114


namespace NUMINAMATH_CALUDE_system_solution_value_l1511_151173

theorem system_solution_value (x y a b : ℝ) : 
  3 * x - 2 * y + 20 = 0 →
  2 * x + 15 * y - 3 = 0 →
  a * x - b * y = 3 →
  6 * a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_value_l1511_151173


namespace NUMINAMATH_CALUDE_total_books_sold_l1511_151118

/-- Represents the sales data for a salesperson over 5 days -/
structure SalesData where
  monday : Float
  tuesday_multiplier : Float
  wednesday_multiplier : Float
  friday_multiplier : Float

/-- Calculates the total books sold by a salesperson over 5 days -/
def total_sales (data : SalesData) : Float :=
  let tuesday := data.monday * data.tuesday_multiplier
  let wednesday := tuesday * data.wednesday_multiplier
  data.monday + tuesday + wednesday + data.monday + (data.monday * data.friday_multiplier)

/-- Theorem stating the total books sold by all three salespeople -/
theorem total_books_sold (matias_data olivia_data luke_data : SalesData) 
  (h_matias : matias_data = { monday := 7, tuesday_multiplier := 2.5, wednesday_multiplier := 3.5, friday_multiplier := 4.2 })
  (h_olivia : olivia_data = { monday := 5, tuesday_multiplier := 1.5, wednesday_multiplier := 2.2, friday_multiplier := 3 })
  (h_luke : luke_data = { monday := 12, tuesday_multiplier := 0.75, wednesday_multiplier := 1.5, friday_multiplier := 0.8 }) :
  total_sales matias_data + total_sales olivia_data + total_sales luke_data = 227.75 := by
  sorry


end NUMINAMATH_CALUDE_total_books_sold_l1511_151118


namespace NUMINAMATH_CALUDE_positive_real_pair_with_integer_product_and_floor_sum_l1511_151116

theorem positive_real_pair_with_integer_product_and_floor_sum (x y : ℝ) : 
  x > 0 → y > 0 → (∃ n : ℤ, x * y = n) → x + y = ⌊x^2 - y^2⌋ → 
  ∃ d : ℕ, d ≥ 2 ∧ x = d ∧ y = d - 1 := by
sorry

end NUMINAMATH_CALUDE_positive_real_pair_with_integer_product_and_floor_sum_l1511_151116


namespace NUMINAMATH_CALUDE_room_width_calculation_l1511_151165

/-- Given a rectangular room with the following properties:
  * length: 5.5 meters
  * total paving cost: 16500 Rs
  * paving rate: 800 Rs per square meter
  This theorem proves that the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate : ℝ) :
  length = 5.5 →
  total_cost = 16500 →
  rate = 800 →
  (total_cost / rate) / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1511_151165


namespace NUMINAMATH_CALUDE_circle_equation_l1511_151190

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the properties of the circle
def is_tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

def center_on_line (c : Circle) : Prop :=
  c.center.1 = 3 * c.center.2

def cuts_chord_on_line (c : Circle) (chord_length : ℝ) : Prop :=
  ∃ (p q : ℝ × ℝ),
    p.1 - p.2 = 0 ∧ q.1 - q.2 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2 ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation (c : Circle) :
  is_tangent_to_y_axis c →
  center_on_line c →
  cuts_chord_on_line c (2 * Real.sqrt 7) →
  (∀ x y : ℝ, (x - 3)^2 + (y - 1)^2 = 9 ∨ (x + 3)^2 + (y + 1)^2 = 9 ↔
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1511_151190


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1511_151113

noncomputable def f (x : ℝ) : ℝ := (1 - 2 * x^3)^10

theorem f_derivative_at_one : 
  deriv f 1 = 60 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1511_151113


namespace NUMINAMATH_CALUDE_union_A_complementB_equals_result_l1511_151189

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 2*x < 0}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬(x ∈ B)}

-- Define the result set
def result : Set ℝ := {x | x ≤ 1 ∨ 2 ≤ x}

-- Theorem statement
theorem union_A_complementB_equals_result : A ∪ complementB = result := by
  sorry

end NUMINAMATH_CALUDE_union_A_complementB_equals_result_l1511_151189


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1511_151129

theorem right_triangle_perimeter (a b c : ℝ) :
  a = 3 ∧ b = 4 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2) →
  a + b + c = 12 ∨ a + b + c = 7 + Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1511_151129


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1511_151125

theorem triangle_angle_theorem (A B C : ℝ) : 
  A = 32 →
  B = 3 * A →
  C = 2 * A - 12 →
  A + B + C = 180 →
  C = 52 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1511_151125


namespace NUMINAMATH_CALUDE_expression_value_at_three_l1511_151196

theorem expression_value_at_three :
  let x : ℝ := 3
  x^5 - (5*x)^2 = 18 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l1511_151196


namespace NUMINAMATH_CALUDE_equality_of_gcd_lcm_sets_l1511_151194

theorem equality_of_gcd_lcm_sets (a b c : ℕ) :
  ({Nat.gcd a b, Nat.gcd b c, Nat.gcd a c} : Set ℕ) =
  ({Nat.lcm a b, Nat.lcm b c, Nat.lcm a c} : Set ℕ) →
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equality_of_gcd_lcm_sets_l1511_151194


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1511_151168

theorem expression_simplification_and_evaluation :
  ∀ x y : ℝ,
  x - y = 5 →
  x + 2*y = 2 →
  (x^2 - 4*x*y + 4*y^2) / (x^2 - x*y) / (x + y - 3*y^2 / (x - y)) + 1/x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1511_151168


namespace NUMINAMATH_CALUDE_intersection_x_difference_l1511_151152

/-- The difference between the x-coordinates of the intersection points of two parabolas -/
theorem intersection_x_difference (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 3*x^2 - 6*x + 5) 
  (h₂ : ∀ x, g x = -2*x^2 - 4*x + 6) : 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 6 / 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_x_difference_l1511_151152


namespace NUMINAMATH_CALUDE_cricket_players_l1511_151101

theorem cricket_players (B C Both Total : ℕ) : 
  B = 7 → 
  Both = 3 → 
  Total = 9 → 
  Total = B + C - Both → 
  C = 5 :=
by sorry

end NUMINAMATH_CALUDE_cricket_players_l1511_151101


namespace NUMINAMATH_CALUDE_georginas_parrot_days_l1511_151111

/-- The number of days Georgina has had her parrot -/
def days_with_parrot (total_phrases current_phrases_per_week initial_phrases days_per_week : ℕ) : ℕ :=
  ((total_phrases - initial_phrases) / current_phrases_per_week) * days_per_week

/-- Proof that Georgina has had her parrot for 49 days -/
theorem georginas_parrot_days : 
  days_with_parrot 17 2 3 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_georginas_parrot_days_l1511_151111


namespace NUMINAMATH_CALUDE_team_formation_theorem_l1511_151102

/-- The number of ways to form a team with at least one female student -/
def team_formation_count (male_count : ℕ) (female_count : ℕ) (team_size : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to form the team under given conditions -/
theorem team_formation_theorem :
  team_formation_count 5 3 4 = 780 :=
sorry

end NUMINAMATH_CALUDE_team_formation_theorem_l1511_151102


namespace NUMINAMATH_CALUDE_sum_positive_differences_equals_754152_l1511_151127

-- Define the set S
def S : Finset ℕ := Finset.range 11

-- Define the function to calculate 3^n
def pow3 (n : ℕ) : ℕ := 3^n

-- Define the sum of positive differences
def sumPositiveDifferences : ℕ :=
  Finset.sum S (fun i =>
    Finset.sum S (fun j =>
      if pow3 j > pow3 i then pow3 j - pow3 i else 0
    )
  )

-- Theorem statement
theorem sum_positive_differences_equals_754152 :
  sumPositiveDifferences = 754152 := by sorry

end NUMINAMATH_CALUDE_sum_positive_differences_equals_754152_l1511_151127


namespace NUMINAMATH_CALUDE_equation_solutions_l1511_151150

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 1, 2)} ∪ {(k, 2, 3*k) | k : ℕ} ∪ {(2, 3, 18)} ∪ {(1, 2*k, 3*k) | k : ℕ} ∪ {(2, 2, 6)}

theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x > 0 ∧ y > 0 ∧ z > 0 ∧ (1 : ℚ) / x + 2 / y - 3 / z = 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1511_151150


namespace NUMINAMATH_CALUDE_max_span_sum_of_digits_div_by_8_l1511_151112

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: Maximum span between numbers with sum of digits divisible by 8 -/
theorem max_span_sum_of_digits_div_by_8 (m : ℕ) (h1 : m > 0) (h2 : sumOfDigits m % 8 = 0) :
  ∃ (n : ℕ), n = 15 ∧
    sumOfDigits (m + n) % 8 = 0 ∧
    ∀ k : ℕ, 1 ≤ k → k < n → sumOfDigits (m + k) % 8 ≠ 0 ∧
    ∀ n' : ℕ, n' > n →
      ¬(sumOfDigits (m + n') % 8 = 0 ∧
        ∀ k : ℕ, 1 ≤ k → k < n' → sumOfDigits (m + k) % 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_span_sum_of_digits_div_by_8_l1511_151112


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1511_151142

/-- Proves that for a right circular cone and a sphere with the same radius,
    if the volume of the cone is one-third that of the sphere,
    then the ratio of the cone's altitude to its base radius is 4/3. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * r^2 * h) = (1 / 3 * (4 / 3 * π * r^3)) →
  h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1511_151142


namespace NUMINAMATH_CALUDE_distance_to_right_focus_is_18_l1511_151198

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Axiom: P is on the left branch of the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the distance from P to the left focus
def distance_to_left_focus : ℝ := 10

-- Define the distance from P to the right focus
def distance_to_right_focus : ℝ := sorry

-- Theorem to prove
theorem distance_to_right_focus_is_18 :
  distance_to_right_focus = 18 :=
sorry

end NUMINAMATH_CALUDE_distance_to_right_focus_is_18_l1511_151198


namespace NUMINAMATH_CALUDE_school_teachers_count_l1511_151188

theorem school_teachers_count 
  (total : ℕ) 
  (sample_size : ℕ) 
  (sample_students : ℕ) 
  (h1 : total = 2400)
  (h2 : sample_size = 120)
  (h3 : sample_students = 110)
  (h4 : sample_size ≤ total)
  (h5 : sample_students < sample_size) :
  (sample_size - sample_students) * total / sample_size = 200 := by
sorry

end NUMINAMATH_CALUDE_school_teachers_count_l1511_151188


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1511_151124

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  x : ℕ
  is_even : Even x

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.x + (t.x + 2) + (t.x + 4)

/-- Checks if the triangle inequality holds for an EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.x + (t.x + 2) > t.x + 4 ∧
  t.x + (t.x + 4) > t.x + 2 ∧
  (t.x + 2) + (t.x + 4) > t.x

/-- The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    perimeter t = 18 ∧
    ∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t' ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1511_151124


namespace NUMINAMATH_CALUDE_sunday_newspaper_cost_l1511_151182

/-- The cost of the Sunday edition of a newspaper -/
def sunday_cost (weekday_cost : ℚ) (total_cost : ℚ) (num_weeks : ℕ) : ℚ :=
  (total_cost - 3 * weekday_cost * num_weeks) / num_weeks

/-- Theorem stating that the Sunday edition costs $2.00 -/
theorem sunday_newspaper_cost : 
  let weekday_cost : ℚ := 1/2
  let total_cost : ℚ := 28
  let num_weeks : ℕ := 8
  sunday_cost weekday_cost total_cost num_weeks = 2 := by
sorry

end NUMINAMATH_CALUDE_sunday_newspaper_cost_l1511_151182


namespace NUMINAMATH_CALUDE_collinear_points_unique_k_l1511_151130

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that 5/2 is the only value of k that makes the points (1, k/3), (3, 1), and (6, k/2) collinear. -/
theorem collinear_points_unique_k : 
  ∃! k : ℝ, collinear 1 (k/3) 3 1 6 (k/2) ∧ k = 5/2 := by sorry

end NUMINAMATH_CALUDE_collinear_points_unique_k_l1511_151130


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1511_151135

/-- Geometric sequence with first term 3 and second sum 9 -/
def geometric_sequence (n : ℕ) : ℝ :=
  3 * 2^(n - 1)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℝ :=
  3 * (2^n - 1)

theorem geometric_sequence_properties :
  (geometric_sequence 1 = 3) ∧
  (geometric_sum 2 = 9) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sum n = 3 * (2^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1511_151135


namespace NUMINAMATH_CALUDE_sphere_truncated_cone_ratio_l1511_151161

/-- Theorem: The ratio of volumes of a sphere and its circumscribed truncated cone
    equals the ratio of their surface areas. -/
theorem sphere_truncated_cone_ratio
  (R h r₁ r₂ : ℝ)
  (h_pos : h > 0)
  (r₁_pos : r₁ > 0)
  (r₂_pos : r₂ > 0)
  (r₂_ge_r₁ : r₂ ≥ r₁)
  (sphere_inscribed : R ≤ min r₁ (h / 2)) :
  (4 / 3 * π * R^3) / (1 / 3 * π * h * (r₁^2 + r₁ * r₂ + r₂^2)) =
  (4 * π * R^2) / (π * r₁^2 + π * r₂^2 + π * (r₁ + r₂) * Real.sqrt (h^2 + (r₂ - r₁)^2)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_truncated_cone_ratio_l1511_151161


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1511_151121

theorem concentric_circles_radii_difference
  (s L : ℝ)
  (h_positive : s > 0)
  (h_ratio : L^2 / s^2 = 4) :
  L - s = s :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1511_151121


namespace NUMINAMATH_CALUDE_solution_equivalence_l1511_151126

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) := {p | |p.1| + |p.2| = p.1^2}

-- Define the set of points as described in the solution
def T : Set (ℝ × ℝ) := 
  {(0, 0)} ∪ 
  {p | p.1 ≥ 1 ∧ (p.2 = p.1^2 - p.1 ∨ p.2 = -(p.1^2 - p.1))} ∪
  {p | p.1 ≤ -1 ∧ (p.2 = p.1^2 + p.1 ∨ p.2 = -(p.1^2 + p.1))}

-- Theorem statement
theorem solution_equivalence : S = T := by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l1511_151126


namespace NUMINAMATH_CALUDE_diana_etienne_money_comparison_l1511_151171

/-- Proves that Diana's money is 21.25% greater than Etienne's after euro appreciation --/
theorem diana_etienne_money_comparison :
  let initial_rate : ℝ := 1.25  -- 1 euro = 1.25 dollars
  let diana_dollars : ℝ := 600
  let etienne_euros : ℝ := 350
  let appreciation_rate : ℝ := 1.08  -- 8% appreciation
  let new_rate : ℝ := initial_rate * appreciation_rate
  let etienne_dollars : ℝ := etienne_euros * new_rate
  let difference_percent : ℝ := (diana_dollars - etienne_dollars) / etienne_dollars * 100
  difference_percent = 21.25 := by
sorry

end NUMINAMATH_CALUDE_diana_etienne_money_comparison_l1511_151171
