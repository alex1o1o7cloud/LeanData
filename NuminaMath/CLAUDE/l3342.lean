import Mathlib

namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3342_334206

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1/12) :
  (∀ a b : ℕ+, a ≠ b → (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1/12 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) ∧
  (x : ℕ) + (y : ℕ) = 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3342_334206


namespace NUMINAMATH_CALUDE_max_product_with_constraints_l3342_334250

theorem max_product_with_constraints (a b : ℕ) :
  a + b = 100 →
  a % 5 = 2 →
  b % 6 = 3 →
  a * b ≤ 2331 ∧ ∃ (a' b' : ℕ), a' + b' = 100 ∧ a' % 5 = 2 ∧ b' % 6 = 3 ∧ a' * b' = 2331 :=
by sorry

end NUMINAMATH_CALUDE_max_product_with_constraints_l3342_334250


namespace NUMINAMATH_CALUDE_book_selling_price_l3342_334201

/-- Calculates the selling price of an item given its cost price and profit rate -/
def selling_price (cost_price : ℚ) (profit_rate : ℚ) : ℚ :=
  cost_price * (1 + profit_rate)

/-- Theorem: The selling price of a book with cost price Rs 50 and profit rate 40% is Rs 70 -/
theorem book_selling_price :
  selling_price 50 (40 / 100) = 70 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_l3342_334201


namespace NUMINAMATH_CALUDE_unique_pair_existence_l3342_334218

theorem unique_pair_existence : ∃! (c d : Real),
  c ∈ Set.Ioo 0 (Real.pi / 2) ∧
  d ∈ Set.Ioo 0 (Real.pi / 2) ∧
  c < d ∧
  Real.sin (Real.cos c) = c ∧
  Real.cos (Real.sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_existence_l3342_334218


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3342_334200

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + a_4 + a_10 + a_16 + a_19 = 150,
    prove that a_18 - 2a_14 = -30 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 1 + a 4 + a 10 + a 16 + a 19 = 150) : 
    a 18 - 2 * a 14 = -30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3342_334200


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3342_334238

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 8*x₁ = 9 ∧ x₂^2 + 8*x₂ = 9) ∧ 
  (x₁ = -9 ∧ x₂ = 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3342_334238


namespace NUMINAMATH_CALUDE_original_profit_margin_l3342_334289

theorem original_profit_margin (original_price selling_price : ℝ) : 
  original_price > 0 →
  selling_price > original_price →
  let new_price := 0.9 * original_price
  let original_margin := (selling_price - original_price) / original_price
  let new_margin := (selling_price - new_price) / new_price
  new_margin - original_margin = 0.12 →
  original_margin = 0.08 := by
sorry

end NUMINAMATH_CALUDE_original_profit_margin_l3342_334289


namespace NUMINAMATH_CALUDE_wooden_toy_price_is_20_l3342_334255

/-- The price of a wooden toy at the Craftee And Best store -/
def wooden_toy_price : ℕ := sorry

/-- The price of a hat at the Craftee And Best store -/
def hat_price : ℕ := 10

/-- The amount Kendra initially had -/
def initial_amount : ℕ := 100

/-- The number of wooden toys Kendra bought -/
def wooden_toys_bought : ℕ := 2

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount Kendra received in change -/
def change_received : ℕ := 30

theorem wooden_toy_price_is_20 :
  wooden_toy_price = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_wooden_toy_price_is_20_l3342_334255


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_smallest_addition_for_27452_div_9_smallest_addition_is_7_l3342_334268

theorem smallest_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem smallest_addition_for_27452_div_9 :
  ∃ (x : ℕ), x < 9 ∧ (27452 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (27452 + y) % 9 ≠ 0 :=
by
  apply smallest_addition_for_divisibility 27452 9
  norm_num

theorem smallest_addition_is_7 :
  7 < 9 ∧ (27452 + 7) % 9 = 0 ∧ ∀ (y : ℕ), y < 7 → (27452 + y) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_smallest_addition_for_27452_div_9_smallest_addition_is_7_l3342_334268


namespace NUMINAMATH_CALUDE_additional_machines_for_half_time_l3342_334277

/-- Represents the number of machines needed to complete a job in a given time -/
def machines_needed (initial_machines : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  initial_machines * initial_days / new_days

/-- Proof that 95 additional machines are needed to complete the job in half the time -/
theorem additional_machines_for_half_time (initial_machines : ℕ) (initial_days : ℕ) 
    (h1 : initial_machines = 5) (h2 : initial_days = 20) :
  machines_needed initial_machines initial_days (initial_days / 2) - initial_machines = 95 := by
  sorry

#eval machines_needed 5 20 10 - 5  -- Should output 95

end NUMINAMATH_CALUDE_additional_machines_for_half_time_l3342_334277


namespace NUMINAMATH_CALUDE_max_students_distribution_l3342_334270

theorem max_students_distribution (pens pencils erasers notebooks rulers : ℕ) 
  (h1 : pens = 3528) 
  (h2 : pencils = 3920) 
  (h3 : erasers = 3150) 
  (h4 : notebooks = 5880) 
  (h5 : rulers = 4410) : 
  Nat.gcd pens (Nat.gcd pencils (Nat.gcd erasers (Nat.gcd notebooks rulers))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3342_334270


namespace NUMINAMATH_CALUDE_maia_daily_work_requests_l3342_334222

/-- The number of client requests Maia receives daily -/
def daily_requests : ℕ := 6

/-- The number of days Maia works -/
def work_days : ℕ := 5

/-- The number of client requests remaining after the work period -/
def remaining_requests : ℕ := 10

/-- The number of client requests Maia works on each day -/
def daily_work_requests : ℕ := (daily_requests * work_days - remaining_requests) / work_days

theorem maia_daily_work_requests :
  daily_work_requests = 4 := by sorry

end NUMINAMATH_CALUDE_maia_daily_work_requests_l3342_334222


namespace NUMINAMATH_CALUDE_faye_coloring_books_l3342_334297

/-- Calculates the total number of coloring books Faye has after giving some away and buying more. -/
def total_coloring_books (initial : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - given_away + bought

/-- Proves that Faye ends up with 79 coloring books given the initial conditions. -/
theorem faye_coloring_books : total_coloring_books 34 3 48 = 79 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l3342_334297


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3342_334212

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-3) 7 6 = 87 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3342_334212


namespace NUMINAMATH_CALUDE_evaluate_expression_l3342_334226

theorem evaluate_expression : 
  (128 : ℝ)^(1/3) * (729 : ℝ)^(1/2) = 108 * 2^(1/3) :=
by
  -- Definitions based on given conditions
  have h1 : (128 : ℝ) = 2^7 := by sorry
  have h2 : (729 : ℝ) = 3^6 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3342_334226


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3342_334247

theorem polynomial_factorization (y : ℤ) :
  5 * (y + 4) * (y + 7) * (y + 9) * (y + 11) - 4 * y^2 =
  (y + 1) * (y + 9) * (5 * y^2 + 33 * y + 441) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3342_334247


namespace NUMINAMATH_CALUDE_angle_CBO_is_20_l3342_334252

-- Define the triangle ABC and point O
variable (A B C O : Point)

-- Define the angles as real numbers (in degrees)
variable (angle_BAO angle_CAO angle_CBO angle_ABO angle_ACO angle_BCO angle_AOC : ℝ)

-- State the theorem
theorem angle_CBO_is_20 
  (h1 : angle_BAO = angle_CAO)
  (h2 : angle_CBO = angle_ABO)
  (h3 : angle_ACO = angle_BCO)
  (h4 : angle_AOC = 110) :
  angle_CBO = 20 := by
    sorry

end NUMINAMATH_CALUDE_angle_CBO_is_20_l3342_334252


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3342_334272

theorem complex_equation_solution (a : ℝ) : 
  (a^2 - a : ℂ) + (3*a - 1 : ℂ)*Complex.I = 2 + 5*Complex.I → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3342_334272


namespace NUMINAMATH_CALUDE_valid_arrangement_5_cubes_valid_arrangement_6_cubes_l3342_334275

/-- A cube in 3D space --/
structure Cube where
  position : ℝ × ℝ × ℝ

/-- An arrangement of cubes in 3D space --/
def Arrangement (n : ℕ) := Fin n → Cube

/-- Predicate to check if two cubes share a polygonal face --/
def SharesFace (c1 c2 : Cube) : Prop := sorry

/-- Predicate to check if an arrangement is valid (each cube shares a face with every other) --/
def ValidArrangement (arr : Arrangement n) : Prop :=
  ∀ i j, i ≠ j → SharesFace (arr i) (arr j)

/-- Theorem stating the existence of a valid arrangement for 5 cubes --/
theorem valid_arrangement_5_cubes : ∃ (arr : Arrangement 5), ValidArrangement arr := sorry

/-- Theorem stating the existence of a valid arrangement for 6 cubes --/
theorem valid_arrangement_6_cubes : ∃ (arr : Arrangement 6), ValidArrangement arr := sorry

end NUMINAMATH_CALUDE_valid_arrangement_5_cubes_valid_arrangement_6_cubes_l3342_334275


namespace NUMINAMATH_CALUDE_inequality_proof_l3342_334208

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  a / (b^2 * (c + 1)) + b / (c^2 * (a + 1)) + c / (a^2 * (b + 1)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3342_334208


namespace NUMINAMATH_CALUDE_bird_feeder_problem_l3342_334215

theorem bird_feeder_problem (feeder_capacity : ℝ) (birds_per_cup : ℝ) (stolen_amount : ℝ) :
  feeder_capacity = 2 ∧ birds_per_cup = 14 ∧ stolen_amount = 0.5 →
  (feeder_capacity - stolen_amount) * birds_per_cup = 21 := by
  sorry

end NUMINAMATH_CALUDE_bird_feeder_problem_l3342_334215


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3342_334229

theorem initial_mixture_volume 
  (initial_x_percentage : Real) 
  (initial_y_percentage : Real)
  (added_x_volume : Real)
  (final_x_percentage : Real) :
  initial_x_percentage = 0.20 →
  initial_y_percentage = 0.80 →
  added_x_volume = 20 →
  final_x_percentage = 0.36 →
  ∃ (initial_volume : Real),
    initial_volume = 80 ∧
    (initial_x_percentage * initial_volume + added_x_volume) / (initial_volume + added_x_volume) = final_x_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l3342_334229


namespace NUMINAMATH_CALUDE_line_equation_correct_l3342_334232

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfies_equation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if a line equation represents a line with a given slope -/
def has_slope (eq : LineEquation) (m : ℝ) : Prop :=
  eq.a ≠ 0 ∧ eq.b ≠ 0 ∧ m = -eq.a / eq.b

theorem line_equation_correct (L : Line) (eq : LineEquation) : 
  L.point = (-2, 5) →
  L.slope = -3/4 →
  eq = ⟨3, 4, -14⟩ →
  satisfies_equation L.point eq ∧ has_slope eq L.slope :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l3342_334232


namespace NUMINAMATH_CALUDE_jar_size_proof_l3342_334283

/-- Proves that the size of the second jar type is 1/2 gallon given the problem conditions -/
theorem jar_size_proof (total_water : ℚ) (total_jars : ℕ) 
  (h1 : total_water = 28)
  (h2 : total_jars = 48)
  (h3 : ∃ (n : ℕ), n * (1 + x + 1/4) = total_jars ∧ n * (1 + x + 1/4) = total_water)
  : x = 1/2 := by
  sorry

#check jar_size_proof

end NUMINAMATH_CALUDE_jar_size_proof_l3342_334283


namespace NUMINAMATH_CALUDE_andrews_donation_l3342_334288

/-- The age when Andrew started donating -/
def start_age : ℕ := 11

/-- Andrew's current age -/
def current_age : ℕ := 29

/-- The amount Andrew donates each year in thousands -/
def yearly_donation : ℕ := 7

/-- Calculate the total amount Andrew has donated -/
def total_donation : ℕ := (current_age - start_age) * yearly_donation

/-- Theorem stating that Andrew's total donation is 126k -/
theorem andrews_donation : total_donation = 126 := by sorry

end NUMINAMATH_CALUDE_andrews_donation_l3342_334288


namespace NUMINAMATH_CALUDE_m_gt_n_gt_0_neither_sufficient_nor_necessary_l3342_334249

/-- Represents an ellipse defined by the equation mx² + ny² = 1 --/
structure Ellipse (m n : ℝ) :=
  (equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1)

/-- Predicate to check if an ellipse has foci on the x-axis --/
def has_foci_on_x_axis (e : Ellipse m n) : Prop :=
  n > m ∧ m > 0

/-- The main theorem stating that m > n > 0 is neither sufficient nor necessary
    for an ellipse to have foci on the x-axis --/
theorem m_gt_n_gt_0_neither_sufficient_nor_necessary :
  ¬(∀ m n : ℝ, m > n ∧ n > 0 → (∀ e : Ellipse m n, has_foci_on_x_axis e)) ∧
  ¬(∀ m n : ℝ, (∀ e : Ellipse m n, has_foci_on_x_axis e) → m > n ∧ n > 0) :=
sorry

end NUMINAMATH_CALUDE_m_gt_n_gt_0_neither_sufficient_nor_necessary_l3342_334249


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3342_334233

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3342_334233


namespace NUMINAMATH_CALUDE_probability_less_than_10_l3342_334241

theorem probability_less_than_10 (p_10_ring : ℝ) (h1 : p_10_ring = 0.22) :
  1 - p_10_ring = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_10_l3342_334241


namespace NUMINAMATH_CALUDE_modulus_of_z_l3342_334254

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3342_334254


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3342_334294

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x + 3 = 0) : 
  (x - 3) * (x + 2) = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3342_334294


namespace NUMINAMATH_CALUDE_stratified_sampling_appropriate_l3342_334274

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a student population -/
structure Population where
  male_count : Nat
  female_count : Nat

/-- Represents a survey -/
structure Survey where
  sample_size : Nat
  method : SamplingMethod

/-- Determines if a sampling method is appropriate for a given population and survey -/
def is_appropriate_method (pop : Population) (survey : Survey) : Prop :=
  pop.male_count = pop.female_count ∧ 
  pop.male_count + pop.female_count > survey.sample_size ∧
  survey.method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the appropriate method for the given scenario -/
theorem stratified_sampling_appropriate (pop : Population) (survey : Survey) :
  pop.male_count = 500 ∧ pop.female_count = 500 ∧ survey.sample_size = 100 →
  is_appropriate_method pop survey :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_appropriate_l3342_334274


namespace NUMINAMATH_CALUDE_line_perp_plane_iff_planes_perp_l3342_334276

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perpPlanes : Plane → Plane → Prop)
variable (perpLinePlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

variable (α β : Plane)
variable (l : Line)

-- State the theorem
theorem line_perp_plane_iff_planes_perp 
  (h_intersect : α ≠ β) 
  (h_subset : subset l α) :
  perpLinePlane l β ↔ perpPlanes α β := by
  sorry

end NUMINAMATH_CALUDE_line_perp_plane_iff_planes_perp_l3342_334276


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l3342_334258

/-- Given a natural number n, prove that if the sum of all coefficients in the expansion of (√x + 3/x)^n
    plus the sum of binomial coefficients equals 72, then the constant term in the expansion is 9. -/
theorem binomial_expansion_constant_term (n : ℕ) : 
  (4^n + 2^n = 72) → 
  (∃ (r : ℕ), r < n ∧ (n.choose r) * 3^r = 9) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l3342_334258


namespace NUMINAMATH_CALUDE_maple_pine_height_difference_l3342_334204

/-- The height of the pine tree in feet -/
def pine_height : ℚ := 24 + 1/4

/-- The height of the maple tree in feet -/
def maple_height : ℚ := 31 + 2/3

/-- The difference in height between the maple and pine trees -/
def height_difference : ℚ := maple_height - pine_height

theorem maple_pine_height_difference :
  height_difference = 7 + 5/12 := by sorry

end NUMINAMATH_CALUDE_maple_pine_height_difference_l3342_334204


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3342_334202

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth_term : a 5 = 10)
  (h_sum_first_three : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3342_334202


namespace NUMINAMATH_CALUDE_inequality_solution_l3342_334287

theorem inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| + |2*x - 1| < 7} = Set.Icc (-1.5) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3342_334287


namespace NUMINAMATH_CALUDE_three_squares_sum_l3342_334242

theorem three_squares_sum (n : ℤ) : 3*(n-1)^2 + 8 = (n-3)^2 + (n-1)^2 + (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_three_squares_sum_l3342_334242


namespace NUMINAMATH_CALUDE_impossible_to_reach_target_l3342_334240

/-- Represents the configuration of matchsticks on a square's vertices -/
structure SquareConfig where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ

/-- Calculates S for a given configuration -/
def S (config : SquareConfig) : ℤ :=
  config.a₁ - config.a₂ + config.a₃ - config.a₄

/-- Represents a valid move in the matchstick game -/
inductive Move
  | move_a₁ (k : ℕ)
  | move_a₂ (k : ℕ)
  | move_a₃ (k : ℕ)
  | move_a₄ (k : ℕ)

/-- Applies a move to a configuration -/
def apply_move (config : SquareConfig) (move : Move) : SquareConfig :=
  match move with
  | Move.move_a₁ k => ⟨config.a₁ - k, config.a₂ + k, config.a₃, config.a₄ + k⟩
  | Move.move_a₂ k => ⟨config.a₁ + k, config.a₂ - k, config.a₃ + k, config.a₄⟩
  | Move.move_a₃ k => ⟨config.a₁, config.a₂ + k, config.a₃ - k, config.a₄ + k⟩
  | Move.move_a₄ k => ⟨config.a₁ + k, config.a₂, config.a₃ + k, config.a₄ - k⟩

/-- The main theorem stating the impossibility of reaching the target configuration -/
theorem impossible_to_reach_target :
  ∀ (moves : List Move),
  let start_config := ⟨1, 0, 0, 0⟩
  let end_config := List.foldl apply_move start_config moves
  end_config ≠ ⟨1, 9, 8, 9⟩ := by
  sorry

/-- Lemma: S mod 3 is invariant under moves -/
lemma S_mod_3_invariant (config : SquareConfig) (move : Move) :
  (S config) % 3 = (S (apply_move config move)) % 3 := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_reach_target_l3342_334240


namespace NUMINAMATH_CALUDE_bike_ride_distance_l3342_334220

theorem bike_ride_distance (first_hour second_hour third_hour : ℝ) : 
  second_hour = first_hour * 1.2 →
  third_hour = second_hour * 1.25 →
  first_hour + second_hour + third_hour = 74 →
  second_hour = 24 := by
sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l3342_334220


namespace NUMINAMATH_CALUDE_problem_solved_probability_l3342_334262

theorem problem_solved_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 2/3) 
  (h2 : prob_B = 3/4) 
  : prob_A + prob_B - prob_A * prob_B = 11/12 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solved_probability_l3342_334262


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3342_334267

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (1 + Real.sqrt 3) / 2 ∧ y₂ = (1 - Real.sqrt 3) / 2 ∧
    2*y₁^2 - 2*y₁ - 1 = 0 ∧ 2*y₂^2 - 2*y₂ - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3342_334267


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3342_334292

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 3) :
  (1/x + 1/y : ℝ) ≥ 1 + 2*Real.sqrt 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3342_334292


namespace NUMINAMATH_CALUDE_sequence_bound_l3342_334253

theorem sequence_bound (a : ℕ → ℝ) 
  (h_pos : ∀ n, n ≥ 1 → a n > 0)
  (h_ineq : ∀ n, n ≥ 1 → (a (n + 1))^2 + (a n) * (a (n + 2)) ≤ a n + a (n + 2)) :
  a 2023 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_bound_l3342_334253


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l3342_334273

theorem base_4_9_digit_difference (n : ℕ) : n = 1296 →
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l3342_334273


namespace NUMINAMATH_CALUDE_precision_improves_with_sample_size_l3342_334217

/-- A structure representing a statistical sample -/
structure Sample (α : Type*) where
  data : List α
  size : Nat

/-- A measure of precision for an estimate -/
def precision (α : Type*) : Sample α → ℝ := sorry

/-- Theorem: As sample size increases, precision improves -/
theorem precision_improves_with_sample_size (α : Type*) :
  ∀ (s1 s2 : Sample α), s1.size < s2.size → precision α s1 < precision α s2 :=
sorry

end NUMINAMATH_CALUDE_precision_improves_with_sample_size_l3342_334217


namespace NUMINAMATH_CALUDE_fish_fillet_distribution_l3342_334257

theorem fish_fillet_distribution (total : ℕ) (second_team : ℕ) (third_team : ℕ) 
  (h1 : total = 500)
  (h2 : second_team = 131)
  (h3 : third_team = 180) :
  total - (second_team + third_team) = 189 := by
  sorry

end NUMINAMATH_CALUDE_fish_fillet_distribution_l3342_334257


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3342_334282

def inequality_solution (x : ℝ) : Prop :=
  x ∈ Set.Iio 0 ∪ Set.Ioo 0 5 ∪ Set.Ioi 5

theorem inequality_equivalence :
  ∀ x : ℝ, (x^2 / (x - 5)^2 > 0) ↔ inequality_solution x :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3342_334282


namespace NUMINAMATH_CALUDE_f_not_monotonic_implies_k_range_l3342_334285

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y ∨ f x > f y

theorem f_not_monotonic_implies_k_range (k : ℝ) :
  (∀ x, x > 0 → f x = x^2 - (1/2) * Real.log x + 1) →
  (¬ is_monotonic f (k - 1) (k + 1)) →
  k ∈ Set.Icc 1 (3/2) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_implies_k_range_l3342_334285


namespace NUMINAMATH_CALUDE_range_of_a_l3342_334279

theorem range_of_a (π : ℝ) (h : π > 0) : 
  ∀ a : ℝ, (∃ x : ℝ, x < 0 ∧ (1/π)^x = (1+a)/(1-a)) → 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3342_334279


namespace NUMINAMATH_CALUDE_sarah_brings_nine_photos_l3342_334284

/-- The number of photos Sarah brings to fill a photo album -/
def sarahs_photos (total_slots : ℕ) (cristina_photos : ℕ) (john_photos : ℕ) (clarissa_photos : ℕ) : ℕ :=
  total_slots - (cristina_photos + john_photos + clarissa_photos)

/-- Theorem stating that Sarah brings 9 photos given the conditions in the problem -/
theorem sarah_brings_nine_photos :
  sarahs_photos 40 7 10 14 = 9 := by
  sorry

#eval sarahs_photos 40 7 10 14

end NUMINAMATH_CALUDE_sarah_brings_nine_photos_l3342_334284


namespace NUMINAMATH_CALUDE_simplify_expression_l3342_334246

theorem simplify_expression (a : ℚ) : ((2 * a + 6) - 3 * a) / 2 = -a / 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3342_334246


namespace NUMINAMATH_CALUDE_teresa_jog_distance_l3342_334207

/-- Given a speed of 5 km/h and a time of 5 hours, prove that the distance traveled is 25 km. -/
theorem teresa_jog_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 5)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 25 := by
sorry

end NUMINAMATH_CALUDE_teresa_jog_distance_l3342_334207


namespace NUMINAMATH_CALUDE_quadratic_properties_l3342_334295

def f (x : ℝ) := (x - 1)^2 + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y → f ((x + y) / 2) < f x) ∧ 
  (∀ x : ℝ, f (x + 1) = f (1 - x)) ∧
  (f 1 = 3 ∧ ∀ x : ℝ, f x ≥ 3) ∧
  f 0 ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3342_334295


namespace NUMINAMATH_CALUDE_agnes_hourly_rate_l3342_334271

/-- Proves that Agnes's hourly rate is $15 given the conditions of the problem -/
theorem agnes_hourly_rate : 
  ∀ (mila_rate : ℝ) (agnes_weekly_hours : ℝ) (mila_equal_hours : ℝ) (weeks_per_month : ℝ),
  mila_rate = 10 →
  agnes_weekly_hours = 8 →
  mila_equal_hours = 48 →
  weeks_per_month = 4 →
  (agnes_weekly_hours * weeks_per_month * (mila_rate * mila_equal_hours / (agnes_weekly_hours * weeks_per_month))) = 15 :=
by sorry

end NUMINAMATH_CALUDE_agnes_hourly_rate_l3342_334271


namespace NUMINAMATH_CALUDE_seven_mile_taxi_cost_l3342_334256

/-- The cost of a taxi ride given the distance traveled -/
def taxi_cost (fixed_cost : ℚ) (per_mile_cost : ℚ) (miles : ℚ) : ℚ :=
  fixed_cost + per_mile_cost * miles

/-- Theorem: The cost of a 7-mile taxi ride is $4.10 -/
theorem seven_mile_taxi_cost :
  taxi_cost 2 0.3 7 = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_seven_mile_taxi_cost_l3342_334256


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l3342_334211

/-- The quadratic equation x^2 + (m-1)x + 1 = 0 has solutions in [0,2] if and only if m < -1 -/
theorem quadratic_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 + (m-1)*x + 1 = 0) ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l3342_334211


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l3342_334221

/-- Calculates the total number of pennies in a piggy bank after adding extra pennies -/
theorem piggy_bank_pennies (compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) :
  compartments = 20 →
  initial_pennies = 10 →
  added_pennies = 15 →
  compartments * (initial_pennies + added_pennies) = 500 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l3342_334221


namespace NUMINAMATH_CALUDE_trigonometric_equality_quadratic_equation_l3342_334263

theorem trigonometric_equality (x : ℝ) : 
  (1 - 2 * Real.sin x * Real.cos x) / (Real.cos x^2 - Real.sin x^2) = 
  (1 - Real.tan x) / (1 + Real.tan x) := by sorry

theorem quadratic_equation (θ a b : ℝ) :
  Real.tan θ + Real.sin θ = a ∧ Real.tan θ - Real.sin θ = b →
  (a^2 - b^2)^2 = 16 * a * b := by sorry

end NUMINAMATH_CALUDE_trigonometric_equality_quadratic_equation_l3342_334263


namespace NUMINAMATH_CALUDE_mini_croissant_cost_gala_luncheon_cost_l3342_334245

/-- Calculates the cost of mini croissants for a committee luncheon --/
theorem mini_croissant_cost (people : ℕ) (sandwiches_per_person : ℕ) 
  (croissants_per_pack : ℕ) (pack_price : ℚ) : ℕ → ℚ :=
  λ total_croissants =>
    let packs_needed := (total_croissants + croissants_per_pack - 1) / croissants_per_pack
    packs_needed * pack_price

/-- Proves that the cost of mini croissants for the committee luncheon is $32.00 --/
theorem gala_luncheon_cost : 
  mini_croissant_cost 24 2 12 8 (24 * 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_mini_croissant_cost_gala_luncheon_cost_l3342_334245


namespace NUMINAMATH_CALUDE_twentieth_term_is_59_l3342_334261

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 20th term of the arithmetic sequence with first term 2 and common difference 3 is 59 -/
theorem twentieth_term_is_59 :
  arithmeticSequenceTerm 2 3 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_59_l3342_334261


namespace NUMINAMATH_CALUDE_cheese_balls_per_serving_l3342_334298

/-- Given the information about cheese balls in barrels, calculate the number of cheese balls per serving -/
theorem cheese_balls_per_serving 
  (barrel_24oz : ℕ) 
  (barrel_35oz : ℕ) 
  (servings_24oz : ℕ) 
  (cheese_balls_35oz : ℕ) 
  (h1 : barrel_24oz = 24) 
  (h2 : barrel_35oz = 35) 
  (h3 : servings_24oz = 60) 
  (h4 : cheese_balls_35oz = 1050) : 
  (cheese_balls_35oz / barrel_35oz * barrel_24oz) / servings_24oz = 12 := by
  sorry


end NUMINAMATH_CALUDE_cheese_balls_per_serving_l3342_334298


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3342_334260

theorem consecutive_integers_sum (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + b + c + d = 274) → (b = 68) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3342_334260


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3342_334251

theorem quadratic_equation_solution : 
  {x : ℝ | 2 * x^2 + 5 * x = 0} = {0, -5/2} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3342_334251


namespace NUMINAMATH_CALUDE_pump_out_time_l3342_334234

/-- Calculates the time needed to pump out water from a flooded basement -/
theorem pump_out_time (length width depth : ℝ) (num_pumps pump_rate : ℝ) (conversion_rate : ℝ) :
  length = 20 ∧ 
  width = 40 ∧ 
  depth = 2 ∧ 
  num_pumps = 5 ∧ 
  pump_rate = 10 ∧ 
  conversion_rate = 7.5 →
  (length * width * depth * conversion_rate) / (num_pumps * pump_rate) = 240 := by
  sorry

end NUMINAMATH_CALUDE_pump_out_time_l3342_334234


namespace NUMINAMATH_CALUDE_max_absolute_value_on_circle_l3342_334235

theorem max_absolute_value_on_circle (z : ℂ) (h : Complex.abs (z - (1 - Complex.I)) = 1) :
  Complex.abs z ≤ Real.sqrt 2 + 1 ∧ ∃ w : ℂ, Complex.abs (w - (1 - Complex.I)) = 1 ∧ Complex.abs w = Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_max_absolute_value_on_circle_l3342_334235


namespace NUMINAMATH_CALUDE_matrix_power_identity_l3342_334223

variable {n : ℕ}

/-- Prove that for n×n complex matrices A, B, and C, if A^2 = B^2 = C^2 and B^3 = ABC + 2I, then A^6 = I. -/
theorem matrix_power_identity 
  (A B C : Matrix (Fin n) (Fin n) ℂ) 
  (h1 : A ^ 2 = B ^ 2)
  (h2 : B ^ 2 = C ^ 2)
  (h3 : B ^ 3 = A * B * C + 2 • 1) : 
  A ^ 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_matrix_power_identity_l3342_334223


namespace NUMINAMATH_CALUDE_jasons_initial_money_l3342_334296

theorem jasons_initial_money (initial_money : ℝ) : 
  let remaining_after_books := (3/4 : ℝ) * initial_money - 10
  let remaining_after_dvds := remaining_after_books - (2/5 : ℝ) * remaining_after_books - 8
  remaining_after_dvds = 130 → initial_money = 320 := by
sorry

end NUMINAMATH_CALUDE_jasons_initial_money_l3342_334296


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l3342_334236

theorem quadratic_real_roots_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l3342_334236


namespace NUMINAMATH_CALUDE_last_amoeba_is_B_l3342_334209

/-- Represents the type of a Martian amoeba -/
inductive AmoebType
  | A
  | B
  | C

/-- Represents the state of the amoeba population -/
structure AmoebState where
  countA : ℕ
  countB : ℕ
  countC : ℕ

/-- Defines the initial state of amoebas -/
def initialState : AmoebState :=
  { countA := 20, countB := 21, countC := 22 }

/-- Defines the merger rule for amoebas -/
def merge (a b : AmoebType) : AmoebType :=
  match a, b with
  | AmoebType.A, AmoebType.B => AmoebType.C
  | AmoebType.B, AmoebType.C => AmoebType.A
  | AmoebType.C, AmoebType.A => AmoebType.B
  | _, _ => a  -- This case should not occur in valid mergers

/-- Theorem: The last remaining amoeba is of type B -/
theorem last_amoeba_is_B (final : AmoebState) 
    (h_final : final.countA + final.countB + final.countC = 1) :
    ∃ (n : ℕ), n > 0 ∧ final = { countA := 0, countB := n, countC := 0 } :=
  sorry

#check last_amoeba_is_B

end NUMINAMATH_CALUDE_last_amoeba_is_B_l3342_334209


namespace NUMINAMATH_CALUDE_current_algae_count_l3342_334286

/-- The number of algae plants originally in Milford Lake -/
def original_algae : ℕ := 809

/-- The number of additional algae plants in Milford Lake -/
def additional_algae : ℕ := 2454

/-- Theorem stating the current total number of algae plants in Milford Lake -/
theorem current_algae_count : original_algae + additional_algae = 3263 := by
  sorry

end NUMINAMATH_CALUDE_current_algae_count_l3342_334286


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3342_334228

def f (x : ℝ) : ℝ := x^4 + 9*x^3 + 18*x^2 + 2023*x - 2021

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^12 + 9*x^11 + 18*x^10 + 2023*x^9 - 2021*x^8 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_unique_positive_solution_l3342_334228


namespace NUMINAMATH_CALUDE_factory_uses_systematic_sampling_factory_sampling_is_systematic_l3342_334281

-- Define the characteristics of the sampling method
structure SamplingMethod where
  regular_intervals : Bool
  fixed_position : Bool
  continuous_process : Bool

-- Define Systematic Sampling
def SystematicSampling : SamplingMethod :=
  { regular_intervals := true
  , fixed_position := true
  , continuous_process := true }

-- Define the factory's sampling method
def FactorySamplingMethod : SamplingMethod :=
  { regular_intervals := true  -- Every 5 minutes
  , fixed_position := true     -- Fixed position on conveyor belt
  , continuous_process := true -- Conveyor belt process
  }

-- Theorem to prove
theorem factory_uses_systematic_sampling :
  FactorySamplingMethod = SystematicSampling := by
  sorry

-- Additional theorem to show that the factory's method is indeed Systematic Sampling
theorem factory_sampling_is_systematic :
  FactorySamplingMethod.regular_intervals ∧
  FactorySamplingMethod.fixed_position ∧
  FactorySamplingMethod.continuous_process := by
  sorry

end NUMINAMATH_CALUDE_factory_uses_systematic_sampling_factory_sampling_is_systematic_l3342_334281


namespace NUMINAMATH_CALUDE_probability_green_is_9_31_l3342_334259

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue

/-- Calculates the probability of selecting a green jelly bean -/
def probabilityGreen (bag : JellyBeanBag) : ℚ :=
  bag.green / totalJellyBeans bag

/-- The specific bag of jelly beans described in the problem -/
def specificBag : JellyBeanBag :=
  { red := 10, green := 9, yellow := 5, blue := 7 }

/-- Theorem stating that the probability of selecting a green jelly bean
    from the specific bag is 9/31 -/
theorem probability_green_is_9_31 :
  probabilityGreen specificBag = 9 / 31 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_is_9_31_l3342_334259


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3342_334213

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I).im ≠ 0 ∧ 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I).re = 0 → 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3342_334213


namespace NUMINAMATH_CALUDE_smallest_number_l3342_334225

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, -4, 5}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3342_334225


namespace NUMINAMATH_CALUDE_rational_square_sum_l3342_334264

theorem rational_square_sum (a b c : ℚ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ∃ r : ℚ, (1 / (a - b)^2 + 1 / (b - c)^2 + 1 / (c - a)^2) = r^2 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_sum_l3342_334264


namespace NUMINAMATH_CALUDE_tv_show_average_episodes_l3342_334219

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_average_episodes_l3342_334219


namespace NUMINAMATH_CALUDE_value_of_expression_l3342_334237

theorem value_of_expression (x : ℝ) (h : x^2 - x = 1) : 1 + 2*x - 2*x^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3342_334237


namespace NUMINAMATH_CALUDE_exists_sequence_iff_N_ge_4_l3342_334299

/-- A sequence of positive integers -/
def PositiveIntegerSequence := ℕ+ → ℕ+

/-- Strictly increasing sequence -/
def StrictlyIncreasing (s : PositiveIntegerSequence) : Prop :=
  ∀ n m : ℕ+, n < m → s n < s m

/-- The property that the sequence satisfies for a given N -/
def SatisfiesProperty (s : PositiveIntegerSequence) (N : ℝ) : Prop :=
  ∀ n : ℕ+, (s (2 * n - 1) + s (2 * n)) / s n = N

/-- The main theorem -/
theorem exists_sequence_iff_N_ge_4 (N : ℝ) : 
  (∃ s : PositiveIntegerSequence, StrictlyIncreasing s ∧ SatisfiesProperty s N) ↔ N ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_sequence_iff_N_ge_4_l3342_334299


namespace NUMINAMATH_CALUDE_lcm_18_24_l3342_334203

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l3342_334203


namespace NUMINAMATH_CALUDE_grandfathers_age_is_79_l3342_334291

/-- The age of Caleb's grandfather based on the number of candles on the cake -/
def grandfathers_age (yellow_candles red_candles blue_candles : ℕ) : ℕ :=
  yellow_candles + red_candles + blue_candles

/-- Theorem stating that Caleb's grandfather's age is 79 given the number of candles -/
theorem grandfathers_age_is_79 :
  grandfathers_age 27 14 38 = 79 := by
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_is_79_l3342_334291


namespace NUMINAMATH_CALUDE_min_value_theorem_l3342_334243

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∀ z : ℝ, z = x^2 / (x + 2) + y^2 / (y + 1) → z ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3342_334243


namespace NUMINAMATH_CALUDE_max_value_of_f_l3342_334239

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x) ∧
  f x = Real.pi / 12 + Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3342_334239


namespace NUMINAMATH_CALUDE_sequence_property_l3342_334210

theorem sequence_property (a : ℕ → ℝ) :
  a 2 = 2 ∧ (∀ n : ℕ, n ≥ 2 → a (n + 1) - a n - 1 = 0) →
  ∀ n : ℕ, n ≥ 2 → a n = n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l3342_334210


namespace NUMINAMATH_CALUDE_two_solutions_exist_sum_of_solutions_l3342_334290

/-- Sum of digits of a positive integer in base 10 -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- The equation n - 3 * sum_of_digits n = 2022 has exactly two solutions -/
theorem two_solutions_exist :
  ∃ (n1 n2 : ℕ+),
    n1 - 3 * sum_of_digits n1 = 2022 ∧
    n2 - 3 * sum_of_digits n2 = 2022 ∧
    n1 ≠ n2 ∧
    ∀ (n : ℕ+), n - 3 * sum_of_digits n = 2022 → n = n1 ∨ n = n2 :=
  sorry

/-- The sum of the two solutions is 4107 -/
theorem sum_of_solutions :
  ∃ (n1 n2 : ℕ+),
    n1 - 3 * sum_of_digits n1 = 2022 ∧
    n2 - 3 * sum_of_digits n2 = 2022 ∧
    n1 ≠ n2 ∧
    n1 + n2 = 4107 :=
  sorry

end NUMINAMATH_CALUDE_two_solutions_exist_sum_of_solutions_l3342_334290


namespace NUMINAMATH_CALUDE_cubic_polynomial_proof_l3342_334278

theorem cubic_polynomial_proof : 
  let p : ℝ → ℝ := λ x => -5/6 * x^3 + 5 * x^2 - 85/6 * x - 5
  (p 1 = -10) ∧ (p 2 = -20) ∧ (p 3 = -30) ∧ (p 5 = -70) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_proof_l3342_334278


namespace NUMINAMATH_CALUDE_average_weight_increase_l3342_334266

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase
  (n : ℕ)                           -- number of people in the group
  (initial_weight : ℝ)              -- weight of the person being replaced
  (new_weight : ℝ)                  -- weight of the new person
  (h1 : n = 8)                      -- there are 8 people in the group
  (h2 : initial_weight = 55)        -- the initial person weighs 55 kg
  (h3 : new_weight = 75)            -- the new person weighs 75 kg
  : (new_weight - initial_weight) / n = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3342_334266


namespace NUMINAMATH_CALUDE_unique_tournaments_eq_fib_l3342_334269

/-- Represents a sequence of scores in descending order -/
def ScoreSequence (n : ℕ) := { a : Fin n → ℕ // ∀ i j, i ≤ j → a i ≥ a j }

/-- Represents a tournament outcome -/
structure Tournament (n : ℕ) where
  scores : ScoreSequence n
  team_scores : Fin n → ℕ

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The number of unique tournament outcomes for n teams -/
def uniqueTournaments (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of unique tournament outcomes is the (n+1)th Fibonacci number -/
theorem unique_tournaments_eq_fib (n : ℕ) : uniqueTournaments n = fib (n + 1) := by sorry

end NUMINAMATH_CALUDE_unique_tournaments_eq_fib_l3342_334269


namespace NUMINAMATH_CALUDE_sarah_trucks_l3342_334216

-- Define the initial number of trucks Sarah had
def initial_trucks : ℕ := 51

-- Define the number of trucks Sarah gave away
def trucks_given_away : ℕ := 13

-- Define the number of trucks Sarah has left
def trucks_left : ℕ := 38

-- Theorem statement
theorem sarah_trucks : 
  initial_trucks = trucks_given_away + trucks_left :=
by sorry

end NUMINAMATH_CALUDE_sarah_trucks_l3342_334216


namespace NUMINAMATH_CALUDE_equation_solution_l3342_334248

theorem equation_solution : ∀ x : ℝ, 3 * x + 15 = (1/3) * (6 * x + 45) → x - 5 = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3342_334248


namespace NUMINAMATH_CALUDE_even_sum_probability_l3342_334231

/-- Represents a die with faces numbered from 1 to n -/
structure Die (n : ℕ) where
  faces : Finset ℕ
  face_count : faces.card = n
  valid_faces : ∀ x ∈ faces, 1 ≤ x ∧ x ≤ n

/-- A regular die with faces numbered from 1 to 6 -/
def regular_die : Die 6 := {
  faces := Finset.range 6,
  face_count := sorry,
  valid_faces := sorry
}

/-- An odd-numbered die with faces 1, 3, 5, 7, 9, 11 -/
def odd_die : Die 6 := {
  faces := Finset.range 6,
  face_count := sorry,
  valid_faces := sorry
}

/-- The probability of an event occurring -/
def probability (event : Prop) : ℚ := sorry

/-- The sum of the top faces of three dice -/
def dice_sum (d1 d2 : Die 6) (d3 : Die 6) : ℕ := sorry

/-- The statement to be proved -/
theorem even_sum_probability :
  probability (∃ (r1 r2 : Die 6) (o : Die 6), 
    r1 = regular_die ∧ 
    r2 = regular_die ∧ 
    o = odd_die ∧ 
    Even (dice_sum r1 r2 o)) = 1/2 := sorry

end NUMINAMATH_CALUDE_even_sum_probability_l3342_334231


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3342_334244

/-- Given two rectangles A and B with sides proportional to a constant k, 
    prove that their area ratio is 4:1 -/
theorem rectangle_area_ratio 
  (k : ℝ) 
  (a b c d : ℝ) 
  (h_pos : k > 0) 
  (h_ka : a = k * a) 
  (h_kb : b = k * b) 
  (h_kc : c = k * c) 
  (h_kd : d = k * d) 
  (h_ratio : a / c = b / d) 
  (h_val : a / c = 2 / 5) : 
  (a * b) / (c * d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l3342_334244


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3342_334205

def total_republicans : Nat := 10
def total_democrats : Nat := 8
def subcommittee_republicans : Nat := 4
def subcommittee_democrats : Nat := 3
def senior_democrat : Nat := 1

def ways_to_form_subcommittee : Nat :=
  Nat.choose total_republicans subcommittee_republicans *
  Nat.choose (total_democrats - senior_democrat) (subcommittee_democrats - senior_democrat)

theorem subcommittee_formation_count :
  ways_to_form_subcommittee = 4410 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3342_334205


namespace NUMINAMATH_CALUDE_sons_age_l3342_334230

/-- Given a father and son where the father is 26 years older than the son,
    and in two years the father's age will be twice the son's age,
    prove that the son's current age is 24 years. -/
theorem sons_age (son_age father_age : ℕ) 
  (h1 : father_age = son_age + 26)
  (h2 : father_age + 2 = 2 * (son_age + 2)) : 
  son_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l3342_334230


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l3342_334224

theorem gasoline_tank_capacity : ∃ x : ℚ, 
  (3/4 : ℚ) * x - (1/3 : ℚ) * x = 18 ∧ x = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l3342_334224


namespace NUMINAMATH_CALUDE_total_flowers_l3342_334227

def flower_collection (arwen_tulips arwen_roses : ℕ) : ℕ :=
  let elrond_tulips := 2 * arwen_tulips
  let elrond_roses := 3 * arwen_roses
  let galadriel_tulips := 3 * elrond_tulips
  let galadriel_roses := 2 * arwen_roses
  arwen_tulips + arwen_roses + elrond_tulips + elrond_roses + galadriel_tulips + galadriel_roses

theorem total_flowers : flower_collection 20 18 = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l3342_334227


namespace NUMINAMATH_CALUDE_thirteenth_number_with_digit_sum_12_l3342_334280

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 13th number with digit sum 12 is 174 -/
theorem thirteenth_number_with_digit_sum_12 : 
  nth_number_with_digit_sum_12 13 = 174 := by sorry

end NUMINAMATH_CALUDE_thirteenth_number_with_digit_sum_12_l3342_334280


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l3342_334265

theorem x_positive_necessary_not_sufficient :
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (x - 4) ≥ 0) ∧
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l3342_334265


namespace NUMINAMATH_CALUDE_quadratic_to_vertex_form_l3342_334214

theorem quadratic_to_vertex_form (x : ℝ) : 
  x^2 - 4*x + 5 = (x - 2)^2 + 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_to_vertex_form_l3342_334214


namespace NUMINAMATH_CALUDE_cats_not_liking_tuna_or_chicken_l3342_334293

theorem cats_not_liking_tuna_or_chicken 
  (total : ℕ) (tuna : ℕ) (chicken : ℕ) (both : ℕ) :
  total = 80 → tuna = 15 → chicken = 60 → both = 10 →
  total - (tuna + chicken - both) = 15 := by
sorry

end NUMINAMATH_CALUDE_cats_not_liking_tuna_or_chicken_l3342_334293
