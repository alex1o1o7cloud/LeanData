import Mathlib

namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2014_201476

/-- Given a person's income and savings, calculate the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 10000) 
  (h2 : savings = 4000) : 
  (income : ℚ) / (income - savings) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2014_201476


namespace NUMINAMATH_CALUDE_no_real_roots_l2014_201475

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2014_201475


namespace NUMINAMATH_CALUDE_oscillation_period_l2014_201491

/-- The period of oscillation for a mass on a wire with small displacements -/
theorem oscillation_period
  (l d g : ℝ) 
  (G : ℝ) -- mass (not used in the formula but mentioned in the problem)
  (h₁ : l > d) 
  (h₂ : l > 0) 
  (h₃ : d > 0) 
  (h₄ : g > 0) :
  ∃ T : ℝ, T = π * l * Real.sqrt (Real.sqrt 2 / (g * Real.sqrt (l^2 - d^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_oscillation_period_l2014_201491


namespace NUMINAMATH_CALUDE_triangle_side_roots_l2014_201436

theorem triangle_side_roots (m : ℝ) : 
  (∃ a b c : ℝ, 
    (a - 1) * (a^2 - 2*a + m) = 0 ∧
    (b - 1) * (b^2 - 2*b + m) = 0 ∧
    (c - 1) * (c^2 - 2*c + m) = 0 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b) →
  3/4 < m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_roots_l2014_201436


namespace NUMINAMATH_CALUDE_first_train_length_l2014_201493

/-- The length of a train given its speed, the speed and length of an oncoming train, and the time they take to cross each other. -/
def trainLength (speed1 : ℝ) (speed2 : ℝ) (length2 : ℝ) (crossTime : ℝ) : ℝ :=
  (speed1 + speed2) * crossTime - length2

/-- Theorem stating the length of the first train given the problem conditions -/
theorem first_train_length :
  let speed1 := 120 * (1000 / 3600)  -- Convert 120 km/hr to m/s
  let speed2 := 80 * (1000 / 3600)   -- Convert 80 km/hr to m/s
  let length2 := 230.04              -- Length of the second train in meters
  let crossTime := 9                 -- Time to cross in seconds
  trainLength speed1 speed2 length2 crossTime = 269.96 := by
  sorry

end NUMINAMATH_CALUDE_first_train_length_l2014_201493


namespace NUMINAMATH_CALUDE_fraction_nonzero_digits_l2014_201483

/-- The number of non-zero digits to the right of the decimal point in the decimal representation of a rational number -/
def nonZeroDigitsAfterDecimal (q : ℚ) : ℕ :=
  sorry

/-- The fraction we're considering -/
def fraction : ℚ := 120 / (2^4 * 5^9)

theorem fraction_nonzero_digits :
  nonZeroDigitsAfterDecimal fraction = 3 :=
sorry

end NUMINAMATH_CALUDE_fraction_nonzero_digits_l2014_201483


namespace NUMINAMATH_CALUDE_negation_of_squared_nonnegative_l2014_201406

theorem negation_of_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_squared_nonnegative_l2014_201406


namespace NUMINAMATH_CALUDE_job_completion_time_l2014_201485

theorem job_completion_time (P Q : ℝ) (h1 : Q = 15) (h2 : 3 / P + 3 / Q + 1 / (5 * P) = 1) : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2014_201485


namespace NUMINAMATH_CALUDE_middle_term_expansion_l2014_201462

theorem middle_term_expansion (n a : ℕ+) (h1 : n > a) (h2 : 1 + a ^ (n : ℕ) = 65) :
  let middle_term := Nat.choose n.val (n.val / 2) * a ^ (n.val / 2)
  middle_term = 160 := by
sorry

end NUMINAMATH_CALUDE_middle_term_expansion_l2014_201462


namespace NUMINAMATH_CALUDE_cube_and_square_root_equality_l2014_201427

theorem cube_and_square_root_equality (x : ℝ) : 
  (x^3 = x ∧ x^2 = x) ↔ (x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_and_square_root_equality_l2014_201427


namespace NUMINAMATH_CALUDE_chessboard_tiling_impossible_l2014_201438

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the chessboard after removal of two squares -/
def ModifiedChessboard : Type := Fin 62 → Square

/-- A function to check if a tiling with dominoes is valid -/
def IsValidTiling (board : ModifiedChessboard) (tiling : List (Fin 62 × Fin 62)) : Prop :=
  ∀ (pair : Fin 62 × Fin 62), pair ∈ tiling →
    (board pair.1 ≠ board pair.2) ∧ 
    (∀ (i : Fin 62), i ∉ [pair.1, pair.2] → 
      ∀ (other_pair : Fin 62 × Fin 62), other_pair ∈ tiling → i ∉ [other_pair.1, other_pair.2])

theorem chessboard_tiling_impossible :
  ∀ (board : ModifiedChessboard),
    (∃ (white_count black_count : Nat), 
      (white_count + black_count = 62) ∧
      (white_count = 30) ∧ (black_count = 32) ∧
      (∀ (i : Fin 62), (board i = Square.White ↔ i.val < white_count))) →
    ¬∃ (tiling : List (Fin 62 × Fin 62)), IsValidTiling board tiling ∧ tiling.length = 31 :=
by sorry

end NUMINAMATH_CALUDE_chessboard_tiling_impossible_l2014_201438


namespace NUMINAMATH_CALUDE_division_reciprocal_equivalence_l2014_201446

theorem division_reciprocal_equivalence (x : ℝ) (hx : x ≠ 0) :
  1 / x = 1 * (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_division_reciprocal_equivalence_l2014_201446


namespace NUMINAMATH_CALUDE_min_omega_for_cos_symmetry_l2014_201454

theorem min_omega_for_cos_symmetry (ω : ℕ+) : 
  (∃ k : ℤ, ω = 6 * k + 2) → 
  (∀ ω' : ℕ+, (∃ k' : ℤ, ω' = 6 * k' + 2) → ω ≤ ω') → 
  ω = 2 := by sorry

end NUMINAMATH_CALUDE_min_omega_for_cos_symmetry_l2014_201454


namespace NUMINAMATH_CALUDE_function_inverse_fraction_l2014_201407

/-- Given a function f : ℝ \ {-1} → ℝ satisfying f((1-x)/(1+x)) = x for all x ≠ -1,
    prove that f(x) = (1-x)/(1+x) for all x ≠ -1 -/
theorem function_inverse_fraction (f : ℝ → ℝ) 
    (h : ∀ x ≠ -1, f ((1 - x) / (1 + x)) = x) :
    ∀ x ≠ -1, f x = (1 - x) / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_inverse_fraction_l2014_201407


namespace NUMINAMATH_CALUDE_smallest_positive_coterminal_angle_l2014_201434

/-- 
Given an angle of -660°, prove that the smallest positive angle 
with the same terminal side is 60°.
-/
theorem smallest_positive_coterminal_angle : 
  ∃ (k : ℤ), -660 + k * 360 = 60 ∧ 
  ∀ (m : ℤ), -660 + m * 360 > 0 → -660 + m * 360 ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_coterminal_angle_l2014_201434


namespace NUMINAMATH_CALUDE_line_slope_l2014_201452

/-- Given a line described by the equation 3y = 4x - 9 + 2z where z = 3,
    prove that the slope of this line is 4/3 -/
theorem line_slope (x y : ℝ) :
  3 * y = 4 * x - 9 + 2 * 3 →
  (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2014_201452


namespace NUMINAMATH_CALUDE_jack_hunting_problem_l2014_201466

theorem jack_hunting_problem (hunts_per_month : ℕ) (season_length : ℚ) 
  (deer_weight : ℕ) (kept_weight_ratio : ℚ) (total_kept_weight : ℕ) :
  hunts_per_month = 6 →
  season_length = 1 / 4 →
  deer_weight = 600 →
  kept_weight_ratio = 1 / 2 →
  total_kept_weight = 10800 →
  (total_kept_weight / kept_weight_ratio / deer_weight) / (hunts_per_month * (season_length * 12)) = 2 := by
sorry

end NUMINAMATH_CALUDE_jack_hunting_problem_l2014_201466


namespace NUMINAMATH_CALUDE_prob_zhong_guo_meng_correct_l2014_201463

/-- The number of cards labeled "中" -/
def num_zhong : ℕ := 2

/-- The number of cards labeled "国" -/
def num_guo : ℕ := 2

/-- The number of cards labeled "梦" -/
def num_meng : ℕ := 1

/-- The total number of cards -/
def total_cards : ℕ := num_zhong + num_guo + num_meng

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing cards that form "中国梦" -/
def prob_zhong_guo_meng : ℚ := 2 / 5

theorem prob_zhong_guo_meng_correct :
  (num_zhong * num_guo * num_meng : ℚ) / (total_cards.choose cards_drawn) = prob_zhong_guo_meng := by
  sorry

end NUMINAMATH_CALUDE_prob_zhong_guo_meng_correct_l2014_201463


namespace NUMINAMATH_CALUDE_group_bill_calculation_l2014_201451

/-- Calculates the total cost for a group at a restaurant where kids eat free. -/
def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Proves that the total cost for a group of 11 people, including 2 kids,
    at a restaurant where adult meals cost $8 and kids eat free, is $72. -/
theorem group_bill_calculation :
  restaurant_bill 11 2 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_group_bill_calculation_l2014_201451


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2014_201455

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  n > 2 → 
  exterior_angle = 72 → 
  (n : ℝ) * exterior_angle = 360 → 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2014_201455


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2014_201412

/-- Given a line in vector form, prove its slope-intercept form --/
theorem line_vector_to_slope_intercept :
  let vector_form : ℝ × ℝ → Prop := λ p => (3 : ℝ) * (p.1 - 2) + (-4 : ℝ) * (p.2 + 3) = 0
  ∃ m b : ℝ, m = 3/4 ∧ b = -9/2 ∧ ∀ x y : ℝ, vector_form (x, y) ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2014_201412


namespace NUMINAMATH_CALUDE_production_days_calculation_l2014_201482

/-- Given the average production and a new day's production, find the number of previous days. -/
theorem production_days_calculation (avg_n : ℝ) (new_prod : ℝ) (avg_n_plus_1 : ℝ) :
  avg_n = 50 →
  new_prod = 100 →
  avg_n_plus_1 = 55 →
  (avg_n * n + new_prod) / (n + 1) = avg_n_plus_1 →
  n = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l2014_201482


namespace NUMINAMATH_CALUDE_shelf_adjustment_theorem_l2014_201450

/-- The number of items on the shelf -/
def total_items : ℕ := 12

/-- The initial number of items on the upper layer -/
def initial_upper : ℕ := 4

/-- The initial number of items on the lower layer -/
def initial_lower : ℕ := 8

/-- The number of items to be moved from lower to upper layer -/
def items_to_move : ℕ := 2

/-- The number of ways to adjust the items -/
def adjustment_ways : ℕ := Nat.choose initial_lower items_to_move

theorem shelf_adjustment_theorem : adjustment_ways = 840 := by sorry

end NUMINAMATH_CALUDE_shelf_adjustment_theorem_l2014_201450


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_root_l2014_201461

theorem geometric_sequence_quadratic_root
  (a b c : ℝ)
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = a * r^2)
  (h_order : a ≤ b ∧ b ≤ c)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_one_root : ∃! x : ℝ, a * x^2 + b * x + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -1/8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_root_l2014_201461


namespace NUMINAMATH_CALUDE_binomial_variance_l2014_201413

/-- A random variable following a binomial distribution with two outcomes -/
structure BinomialRV where
  p : ℝ  -- Probability of success (X = 1)
  q : ℝ  -- Probability of failure (X = 0)
  sum_one : p + q = 1  -- Sum of probabilities is 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.p * X.q

/-- Theorem: The variance of a binomial random variable X is equal to pq -/
theorem binomial_variance (X : BinomialRV) : variance X = X.p * X.q := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_l2014_201413


namespace NUMINAMATH_CALUDE_f_range_l2014_201423

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x - 5

-- Define the domain
def domain : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem f_range : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | -9 ≤ y ∧ y ≤ 7} := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2014_201423


namespace NUMINAMATH_CALUDE_green_eyed_students_l2014_201472

theorem green_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) (green : ℕ) :
  total = 40 →
  3 * green = total - green - both - neither →
  both = 9 →
  neither = 4 →
  green = 9 := by
sorry

end NUMINAMATH_CALUDE_green_eyed_students_l2014_201472


namespace NUMINAMATH_CALUDE_jason_remaining_seashells_l2014_201456

def initial_seashells : ℕ := 49
def given_away : ℕ := 13

theorem jason_remaining_seashells :
  initial_seashells - given_away = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_remaining_seashells_l2014_201456


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2014_201484

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  (l.m * x₀ - y₀ + l.b)^2 = (c.radius^2 * (l.m^2 + 1))

theorem common_external_tangent_y_intercept :
  let c₁ : Circle := ⟨(1, 3), 3⟩
  let c₂ : Circle := ⟨(15, 10), 8⟩
  ∃ (l : Line), l.m > 0 ∧ isTangent l c₁ ∧ isTangent l c₂ ∧ l.b = 5/3 :=
sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2014_201484


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l2014_201460

theorem circles_internally_tangent (r1 r2 : ℝ) (d : ℝ) : 
  r1 + r2 = 5 ∧ 
  r1 * r2 = 3 ∧ 
  d = 3 → 
  r1 < r2 ∧ r2 - r1 < d ∧ d < r1 + r2 := by
  sorry

#check circles_internally_tangent

end NUMINAMATH_CALUDE_circles_internally_tangent_l2014_201460


namespace NUMINAMATH_CALUDE_ice_cream_cones_l2014_201478

theorem ice_cream_cones (cost_per_cone total_spent : ℕ) (h1 : cost_per_cone = 99) (h2 : total_spent = 198) :
  total_spent / cost_per_cone = 2 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cones_l2014_201478


namespace NUMINAMATH_CALUDE_grandmother_age_2000_birth_years_sum_alice_age_2005_l2014_201428

-- Define Alice's age at the end of 2000
def alice_age_2000 : ℕ := 32

-- Define the relationship between Alice's and her grandmother's ages
theorem grandmother_age_2000 : ℕ := 3 * alice_age_2000

-- Define the sum of their birth years
theorem birth_years_sum : ℕ := 3870

-- Theorem to prove Alice's age at the end of 2005
theorem alice_age_2005 :
  2000 - alice_age_2000 + (2000 - grandmother_age_2000) = birth_years_sum →
  alice_age_2000 + 5 = 37 :=
by
  sorry

#check alice_age_2005

end NUMINAMATH_CALUDE_grandmother_age_2000_birth_years_sum_alice_age_2005_l2014_201428


namespace NUMINAMATH_CALUDE_roots_form_parallelogram_l2014_201497

/-- The polynomial whose roots we're investigating -/
def f (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 5*b - 4)*z + 2

/-- Predicate to check if a set of complex numbers forms a parallelogram -/
def forms_parallelogram (s : Set ℂ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ), s = {z₁, z₂, z₃, z₄} ∧ 
    z₁ + z₃ = z₂ + z₄ ∧ z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄

/-- The main theorem stating the condition for the roots to form a parallelogram -/
theorem roots_form_parallelogram (b : ℝ) :
  forms_parallelogram {z : ℂ | f b z = 0} ↔ b = 1 ∨ b = 5/2 :=
sorry

end NUMINAMATH_CALUDE_roots_form_parallelogram_l2014_201497


namespace NUMINAMATH_CALUDE_tenths_of_2019_l2014_201469

theorem tenths_of_2019 : (2019 : ℚ) / 10 = 201.9 := by
  sorry

end NUMINAMATH_CALUDE_tenths_of_2019_l2014_201469


namespace NUMINAMATH_CALUDE_A_completes_in_15_days_l2014_201420

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the rate at which A and B work
variable (A_rate B_rate : ℝ)

-- Define the time it takes for A and B to complete the work alone
variable (A_time B_time : ℝ)

-- Conditions from the problem
axiom B_time_18 : B_time = 18
axiom B_rate_def : B_rate = W / B_time
axiom work_split : A_rate * 5 + B_rate * 12 = W
axiom A_rate_def : A_rate = W / A_time

-- Theorem to prove
theorem A_completes_in_15_days : A_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_A_completes_in_15_days_l2014_201420


namespace NUMINAMATH_CALUDE_fourth_day_jumps_l2014_201400

def jump_count (day : ℕ) : ℕ :=
  match day with
  | 0 => 0  -- day 0 is not defined in the problem, so we set it to 0
  | 1 => 15 -- first day
  | n + 1 => 2 * jump_count n -- subsequent days

theorem fourth_day_jumps :
  jump_count 4 = 120 :=
by sorry

end NUMINAMATH_CALUDE_fourth_day_jumps_l2014_201400


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l2014_201411

theorem max_product_constrained_sum (x y : ℕ+) (h : 7 * x + 5 * y = 140) :
  x * y ≤ 140 ∧ ∃ (a b : ℕ+), 7 * a + 5 * b = 140 ∧ a * b = 140 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l2014_201411


namespace NUMINAMATH_CALUDE_marker_carton_cost_l2014_201492

/-- Proves that the cost of each carton of markers is $20 given the specified conditions --/
theorem marker_carton_cost (
  pencil_cartons : ℕ)
  (pencil_boxes_per_carton : ℕ)
  (pencil_box_cost : ℕ)
  (marker_cartons : ℕ)
  (marker_boxes_per_carton : ℕ)
  (total_spent : ℕ)
  (h1 : pencil_cartons = 20)
  (h2 : pencil_boxes_per_carton = 10)
  (h3 : pencil_box_cost = 2)
  (h4 : marker_cartons = 10)
  (h5 : marker_boxes_per_carton = 5)
  (h6 : total_spent = 600)
  : (total_spent - pencil_cartons * pencil_boxes_per_carton * pencil_box_cost) / marker_cartons = 20 := by
  sorry

end NUMINAMATH_CALUDE_marker_carton_cost_l2014_201492


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l2014_201421

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 16 + y^2 / 4 = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral (hP : P ∈ C) (hQ : Q ∈ C) 
  (hSymmetric : Q = (-P.1, -P.2)) (hDistance : ‖P - Q‖ = ‖F₁ - F₂‖) :
  ‖P - F₁‖ * ‖P - F₂‖ = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l2014_201421


namespace NUMINAMATH_CALUDE_total_dolls_count_l2014_201408

def grandmother_dolls : ℕ := 50

def sister_dolls (grandmother_dolls : ℕ) : ℕ := grandmother_dolls + 2

def rene_dolls (sister_dolls : ℕ) : ℕ := 3 * sister_dolls

def total_dolls (grandmother_dolls sister_dolls rene_dolls : ℕ) : ℕ :=
  grandmother_dolls + sister_dolls + rene_dolls

theorem total_dolls_count :
  total_dolls grandmother_dolls (sister_dolls grandmother_dolls) (rene_dolls (sister_dolls grandmother_dolls)) = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l2014_201408


namespace NUMINAMATH_CALUDE_sin_cos_value_l2014_201416

theorem sin_cos_value (θ : Real) (h : Real.tan θ = 2) : Real.sin θ * Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l2014_201416


namespace NUMINAMATH_CALUDE_intersection_subset_l2014_201481

def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {1, 2, 3}

theorem intersection_subset : P ∩ Q ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_intersection_subset_l2014_201481


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_4_l2014_201499

theorem units_digit_of_3_pow_4 : (3^4 : ℕ) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_4_l2014_201499


namespace NUMINAMATH_CALUDE_parabola_focus_l2014_201494

/-- A parabola is defined by its equation in the form y = ax^2, where a is a non-zero real number. -/
structure Parabola where
  a : ℝ
  a_nonzero : a ≠ 0

/-- The focus of a parabola is a point (h, k) where h and k are real numbers. -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -1/8 * x^2, its focus is at the point (0, -2). -/
theorem parabola_focus (p : Parabola) (h : p.a = -1/8) : 
  ∃ (f : Focus), f.h = 0 ∧ f.k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2014_201494


namespace NUMINAMATH_CALUDE_sandy_savings_l2014_201435

theorem sandy_savings (last_year_salary : ℝ) (last_year_savings_rate : ℝ)
  (salary_increase_rate : ℝ) (savings_increase_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  salary_increase_rate = 0.10 →
  savings_increase_rate = 1.65 →
  (savings_increase_rate * last_year_savings_rate * last_year_salary) /
  (last_year_salary * (1 + salary_increase_rate)) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_sandy_savings_l2014_201435


namespace NUMINAMATH_CALUDE_baker_earnings_calculation_l2014_201437

def cakes_sold : ℕ := 453
def cake_price : ℕ := 12
def pies_sold : ℕ := 126
def pie_price : ℕ := 7

def baker_earnings : ℕ := cakes_sold * cake_price + pies_sold * pie_price

theorem baker_earnings_calculation : baker_earnings = 6318 := by
  sorry

end NUMINAMATH_CALUDE_baker_earnings_calculation_l2014_201437


namespace NUMINAMATH_CALUDE_Q_subset_complement_P_l2014_201464

-- Define the sets P and Q
def P : Set ℝ := {x | x > 4}
def Q : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the theorem
theorem Q_subset_complement_P : Q ⊆ (Set.univ \ P) := by sorry

end NUMINAMATH_CALUDE_Q_subset_complement_P_l2014_201464


namespace NUMINAMATH_CALUDE_no_perfect_cube_in_range_l2014_201465

theorem no_perfect_cube_in_range : 
  ¬∃ n : ℤ, 4 ≤ n ∧ n ≤ 12 ∧ ∃ k : ℤ, n^2 + 3*n + 2 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_in_range_l2014_201465


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2014_201401

/-- Given a point with polar coordinates (3, π/4), prove that its rectangular coordinates are (3√2/2, 3√2/2) -/
theorem polar_to_rectangular :
  let r : ℝ := 3
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2014_201401


namespace NUMINAMATH_CALUDE_charles_whistle_count_l2014_201405

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end NUMINAMATH_CALUDE_charles_whistle_count_l2014_201405


namespace NUMINAMATH_CALUDE_ticket_sales_proof_ticket_sales_result_l2014_201422

theorem ticket_sales_proof (reduced_first_week : ℕ) (total_tickets : ℕ) : ℕ :=
  let reduced_price_tickets := reduced_first_week
  let full_price_tickets := 5 * reduced_price_tickets
  let total := reduced_price_tickets + full_price_tickets
  
  have h1 : reduced_first_week = 5400 := by sorry
  have h2 : total_tickets = 25200 := by sorry
  have h3 : total = total_tickets := by sorry
  
  full_price_tickets

theorem ticket_sales_result : ticket_sales_proof 5400 25200 = 21000 := by sorry

end NUMINAMATH_CALUDE_ticket_sales_proof_ticket_sales_result_l2014_201422


namespace NUMINAMATH_CALUDE_free_throw_probability_l2014_201443

/-- The probability of making a single shot -/
def p : ℝ := sorry

/-- The probability of passing the test (making at least one shot out of three chances) -/
def prob_pass : ℝ := p + p * (1 - p) + p * (1 - p)^2

/-- Theorem stating that if the probability of passing is 0.784, then p is 0.4 -/
theorem free_throw_probability : prob_pass = 0.784 → p = 0.4 := by sorry

end NUMINAMATH_CALUDE_free_throw_probability_l2014_201443


namespace NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l2014_201445

/-- Represents the meeting point of Jack and Jill on their hill run. -/
structure MeetingPoint where
  /-- The time at which Jack and Jill meet, measured from Jill's start time. -/
  time : ℝ
  /-- The distance from the start point where Jack and Jill meet. -/
  distance : ℝ

/-- Calculates the meeting point of Jack and Jill given their running conditions. -/
def calculateMeetingPoint (totalDistance jackHeadStart uphillDistance : ℝ)
                          (jackUphillSpeed jackDownhillSpeed : ℝ)
                          (jillUphillSpeed jillDownhillSpeed : ℝ) : MeetingPoint :=
  sorry

/-- Theorem stating that Jack and Jill meet 2 km from the top of the hill. -/
theorem jack_and_jill_meeting_point :
  let meetingPoint := calculateMeetingPoint 12 (2/15) 7 12 18 14 20
  meetingPoint.distance = 5 ∧ uphillDistance - meetingPoint.distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l2014_201445


namespace NUMINAMATH_CALUDE_complement_of_P_intersection_P_M_range_of_a_l2014_201458

-- Define the sets P and M
def P : Set ℝ := {x | x * (x - 2) ≥ 0}
def M (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 3}

-- Theorem for the complement of P
theorem complement_of_P : 
  (Set.univ : Set ℝ) \ P = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for the intersection of P and M when a = 1
theorem intersection_P_M : 
  P ∩ M 1 = {x | 2 ≤ x ∧ x < 4} := by sorry

-- Theorem for the range of a when ∁ₗP ⊆ M
theorem range_of_a (a : ℝ) : 
  ((Set.univ : Set ℝ) \ P) ⊆ M a ↔ -1 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_complement_of_P_intersection_P_M_range_of_a_l2014_201458


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l2014_201473

theorem largest_four_digit_congruent_to_17_mod_26 :
  (∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧
    ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) →
  (∃ x : ℕ, x = 9972 ∧ x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧
    ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l2014_201473


namespace NUMINAMATH_CALUDE_janes_weekly_reading_l2014_201470

/-- Represents the number of pages Jane reads on a given day -/
structure DailyReading where
  morning : ℕ
  lunch : ℕ
  evening : ℕ
  extra : ℕ

/-- Calculates the total pages read in a day -/
def totalPagesPerDay (d : DailyReading) : ℕ :=
  d.morning + d.lunch + d.evening + d.extra

/-- Represents Jane's weekly reading schedule -/
def weeklySchedule : List DailyReading :=
  [
    { morning := 5,  lunch := 0, evening := 10, extra := 0  }, -- Monday
    { morning := 7,  lunch := 0, evening := 8,  extra := 0  }, -- Tuesday
    { morning := 5,  lunch := 0, evening := 5,  extra := 0  }, -- Wednesday
    { morning := 7,  lunch := 0, evening := 8,  extra := 15 }, -- Thursday
    { morning := 10, lunch := 5, evening := 0,  extra := 0  }, -- Friday
    { morning := 12, lunch := 0, evening := 20, extra := 0  }, -- Saturday
    { morning := 12, lunch := 0, evening := 0,  extra := 0  }  -- Sunday
  ]

/-- Theorem: Jane reads 129 pages in total over one week -/
theorem janes_weekly_reading : 
  (weeklySchedule.map totalPagesPerDay).sum = 129 := by
  sorry

end NUMINAMATH_CALUDE_janes_weekly_reading_l2014_201470


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l2014_201488

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1 + x, 7;
     3 - x, 8]

theorem matrix_not_invertible_iff (x : ℚ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 13/15 := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l2014_201488


namespace NUMINAMATH_CALUDE_rachel_money_theorem_l2014_201410

def rachel_money_left (initial_amount : ℚ) (lunch_fraction : ℚ) (dvd_fraction : ℚ) : ℚ :=
  initial_amount - (lunch_fraction * initial_amount) - (dvd_fraction * initial_amount)

theorem rachel_money_theorem :
  rachel_money_left 200 (1/4) (1/2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rachel_money_theorem_l2014_201410


namespace NUMINAMATH_CALUDE_sum_zero_iff_squared_sum_equal_l2014_201479

theorem sum_zero_iff_squared_sum_equal {a b c : ℝ} (h : ¬(a = b ∧ b = c)) :
  a + b + c = 0 ↔ a^2 + a*b + b^2 = b^2 + b*c + c^2 ∧ b^2 + b*c + c^2 = c^2 + c*a + a^2 :=
sorry

end NUMINAMATH_CALUDE_sum_zero_iff_squared_sum_equal_l2014_201479


namespace NUMINAMATH_CALUDE_real_roots_iff_m_le_25_4_m_eq_6_when_condition_satisfied_l2014_201448

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 5*x + m = 0

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, quadratic_equation x m

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

-- Define the additional condition for part 2
def satisfies_root_condition (x₁ x₂ : ℝ) : Prop := 3*x₁ - 2*x₂ = 5

-- Theorem 1: Equation has real roots iff m ≤ 25/4
theorem real_roots_iff_m_le_25_4 :
  ∀ m : ℝ, has_real_roots m ↔ m ≤ 25/4 :=
sorry

-- Theorem 2: If equation has two real roots satisfying the condition, then m = 6
theorem m_eq_6_when_condition_satisfied :
  ∀ m : ℝ, has_two_distinct_real_roots m →
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ satisfies_root_condition x₁ x₂) →
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_real_roots_iff_m_le_25_4_m_eq_6_when_condition_satisfied_l2014_201448


namespace NUMINAMATH_CALUDE_solve_prize_problem_l2014_201441

def prize_problem (x y m n w : ℝ) : Prop :=
  x + 2*y = 40 ∧
  2*x + 3*y = 70 ∧
  m + n = 60 ∧
  m ≥ n/2 ∧
  w = m*x + n*y

theorem solve_prize_problem :
  ∀ x y m n w,
  prize_problem x y m n w →
  (x = 20 ∧ y = 10) ∧
  (∀ m' n' w',
    prize_problem x y m' n' w' →
    w ≤ w') ∧
  (m = 20 ∧ n = 40 ∧ w = 800) :=
by sorry

end NUMINAMATH_CALUDE_solve_prize_problem_l2014_201441


namespace NUMINAMATH_CALUDE_special_trapezoid_not_isosceles_l2014_201453

/-- A trapezoid with the given properties --/
structure SpecialTrapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  is_trapezoid : base1 ≠ base2
  base_values : base1 = 3 ∧ base2 = 4
  diagonal_length : diagonal = 6
  diagonal_bisects_angle : Bool

/-- Theorem stating that a trapezoid with the given properties cannot be isosceles --/
theorem special_trapezoid_not_isosceles (t : SpecialTrapezoid) : 
  ¬(∃ (side : ℝ), side > 0 ∧ t.base1 < t.base2 → 
    (side = t.diagonal ∧ side^2 = (t.base2 - t.base1)^2 / 4 + side^2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_not_isosceles_l2014_201453


namespace NUMINAMATH_CALUDE_dog_speed_is_16_l2014_201404

/-- Represents the scenario of a man and a dog walking on a path -/
structure WalkingScenario where
  path_length : Real
  man_speed : Real
  dog_trips : Nat
  remaining_distance : Real
  dog_speed : Real

/-- Checks if the given scenario is valid -/
def is_valid_scenario (s : WalkingScenario) : Prop :=
  s.path_length = 0.625 ∧
  s.man_speed = 4 ∧
  s.dog_trips = 4 ∧
  s.remaining_distance = 0.081 ∧
  s.dog_speed > s.man_speed

/-- Theorem: Given the conditions, the dog's speed is 16 km/h -/
theorem dog_speed_is_16 (s : WalkingScenario) 
  (h : is_valid_scenario s) : s.dog_speed = 16 := by
  sorry

#check dog_speed_is_16

end NUMINAMATH_CALUDE_dog_speed_is_16_l2014_201404


namespace NUMINAMATH_CALUDE_max_min_values_of_f_on_interval_l2014_201418

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x + 5

-- Define the interval
def interval : Set ℝ := {x | -5/2 ≤ x ∧ x ≤ 3/2}

-- Theorem statement
theorem max_min_values_of_f_on_interval :
  (∃ x ∈ interval, f x = 9 ∧ ∀ y ∈ interval, f y ≤ 9) ∧
  (∃ x ∈ interval, f x = -11.25 ∧ ∀ y ∈ interval, f y ≥ -11.25) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_on_interval_l2014_201418


namespace NUMINAMATH_CALUDE_max_d_is_25_l2014_201433

/-- Sequence term definition -/
def a (n : ℕ) : ℕ := 100 + n^2 + 2*n

/-- Greatest common divisor of consecutive terms -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The maximum value of d_n is 25 -/
theorem max_d_is_25 : ∃ k : ℕ, d k = 25 ∧ ∀ n : ℕ, d n ≤ 25 :=
  sorry

end NUMINAMATH_CALUDE_max_d_is_25_l2014_201433


namespace NUMINAMATH_CALUDE_min_people_liking_both_l2014_201417

theorem min_people_liking_both (total : ℕ) (chopin : ℕ) (beethoven : ℕ) 
  (h1 : total = 120) (h2 : chopin = 95) (h3 : beethoven = 80) :
  ∃ both : ℕ, both ≥ 55 ∧ chopin + beethoven - both ≤ total := by
  sorry

end NUMINAMATH_CALUDE_min_people_liking_both_l2014_201417


namespace NUMINAMATH_CALUDE_band_percentage_of_ticket_price_l2014_201449

/-- Proves that the band receives 70% of the ticket price, given the concert conditions -/
theorem band_percentage_of_ticket_price : 
  ∀ (attendance : ℕ) (ticket_price : ℕ) (band_members : ℕ) (member_earnings : ℕ),
    attendance = 500 →
    ticket_price = 30 →
    band_members = 4 →
    member_earnings = 2625 →
    (band_members * member_earnings : ℚ) / (attendance * ticket_price) = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_band_percentage_of_ticket_price_l2014_201449


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2014_201457

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set M
def M : Set Int := {-1, 0, 1, 3}

-- Define set N
def N : Set Int := {-2, 0, 2, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2014_201457


namespace NUMINAMATH_CALUDE_abs_product_of_neg_two_and_four_l2014_201467

theorem abs_product_of_neg_two_and_four :
  ∀ x y : ℤ, x = -2 → y = 4 → |x * y| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_product_of_neg_two_and_four_l2014_201467


namespace NUMINAMATH_CALUDE_existence_of_m_n_l2014_201424

theorem existence_of_m_n (h k : ℕ+) (ε : ℝ) (hε : ε > 0) :
  ∃ (m n : ℕ+), ε < |h * Real.sqrt m - k * Real.sqrt n| ∧ |h * Real.sqrt m - k * Real.sqrt n| < 2 * ε :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l2014_201424


namespace NUMINAMATH_CALUDE_exists_easy_a_difficult_b_l2014_201409

structure TestConfiguration where
  variants : Type
  students : Type
  problems : Type
  solved : variants → students → problems → Prop

def easy_a (tc : TestConfiguration) : Prop :=
  ∀ v : tc.variants, ∀ p : tc.problems, ∃ s : tc.students, tc.solved v s p

def difficult_b (tc : TestConfiguration) : Prop :=
  ∀ v : tc.variants, ¬∃ s : tc.students, ∀ p : tc.problems, tc.solved v s p

theorem exists_easy_a_difficult_b :
  ∃ tc : TestConfiguration, easy_a tc ∧ difficult_b tc := by
  sorry

end NUMINAMATH_CALUDE_exists_easy_a_difficult_b_l2014_201409


namespace NUMINAMATH_CALUDE_shaded_fraction_of_specific_quilt_l2014_201442

/-- Represents a square quilt made of unit squares -/
structure Quilt where
  size : Nat
  divided_squares : Finset (Nat × Nat)
  shaded_squares : Finset (Nat × Nat)

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : Rat :=
  sorry

/-- Theorem stating the shaded fraction of the specific quilt configuration -/
theorem shaded_fraction_of_specific_quilt :
  ∃ (q : Quilt),
    q.size = 4 ∧
    q.shaded_squares.card = 6 ∧
    shaded_fraction q = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_specific_quilt_l2014_201442


namespace NUMINAMATH_CALUDE_hat_price_calculation_l2014_201429

theorem hat_price_calculation (total_hats green_hats : ℕ) (blue_price green_price : ℚ) 
  (h1 : total_hats = 85)
  (h2 : green_hats = 38)
  (h3 : blue_price = 6)
  (h4 : green_price = 7) :
  let blue_hats := total_hats - green_hats
  (blue_hats * blue_price + green_hats * green_price : ℚ) = 548 := by
  sorry

end NUMINAMATH_CALUDE_hat_price_calculation_l2014_201429


namespace NUMINAMATH_CALUDE_square_painting_size_l2014_201489

/-- Given the total area of an art collection and the areas of non-square paintings,
    prove that the side length of each square painting is 6 feet. -/
theorem square_painting_size 
  (total_area : ℝ) 
  (num_square_paintings : ℕ) 
  (num_small_paintings : ℕ) 
  (small_painting_width small_painting_height : ℝ)
  (num_large_paintings : ℕ)
  (large_painting_width large_painting_height : ℝ) :
  total_area = 282 ∧ 
  num_square_paintings = 3 ∧
  num_small_paintings = 4 ∧
  small_painting_width = 2 ∧
  small_painting_height = 3 ∧
  num_large_paintings = 1 ∧
  large_painting_width = 10 ∧
  large_painting_height = 15 →
  ∃ (square_side : ℝ), 
    square_side = 6 ∧ 
    num_square_paintings * square_side^2 + 
    num_small_paintings * small_painting_width * small_painting_height +
    num_large_paintings * large_painting_width * large_painting_height = total_area :=
by sorry

end NUMINAMATH_CALUDE_square_painting_size_l2014_201489


namespace NUMINAMATH_CALUDE_city_partition_l2014_201426

/-- A graph representing cities and flight routes -/
structure CityGraph where
  V : Type* -- Set of vertices (cities)
  E : V → V → Prop -- Edge relation (flight routes)

/-- A partition of edges into k sets representing k airlines -/
def AirlinePartition (G : CityGraph) (k : ℕ) :=
  ∃ (P : Fin k → (G.V → G.V → Prop)), 
    (∀ u v, G.E u v ↔ ∃ i, P i u v) ∧
    (∀ i, ∀ {u v w x}, P i u v → P i w x → (u = w ∨ u = x ∨ v = w ∨ v = x))

/-- A partition of vertices into k+2 sets -/
def VertexPartition (G : CityGraph) (k : ℕ) :=
  ∃ (f : G.V → Fin (k + 2)), ∀ u v, G.E u v → f u ≠ f v

theorem city_partition (G : CityGraph) (k : ℕ) :
  AirlinePartition G k → VertexPartition G k := by sorry

end NUMINAMATH_CALUDE_city_partition_l2014_201426


namespace NUMINAMATH_CALUDE_expression_simplification_l2014_201480

theorem expression_simplification (m : ℝ) (h : m = 10) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2014_201480


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2014_201414

theorem quadratic_equation_roots (k : ℕ) 
  (distinct_roots : ∃ x y : ℕ+, x ≠ y ∧ 
    (k^2 - 1) * x^2 - 6 * (3*k - 1) * x + 72 = 0 ∧
    (k^2 - 1) * y^2 - 6 * (3*k - 1) * y + 72 = 0) :
  k = 2 ∧ ∃ x y : ℕ+, x = 6 ∧ y = 4 ∧
    (k^2 - 1) * x^2 - 6 * (3*k - 1) * x + 72 = 0 ∧
    (k^2 - 1) * y^2 - 6 * (3*k - 1) * y + 72 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2014_201414


namespace NUMINAMATH_CALUDE_probability_rain_three_days_l2014_201459

theorem probability_rain_three_days
  (prob_friday : ℝ)
  (prob_saturday : ℝ)
  (prob_sunday : ℝ)
  (prob_sunday_given_saturday : ℝ)
  (h1 : prob_friday = 0.3)
  (h2 : prob_saturday = 0.5)
  (h3 : prob_sunday = 0.4)
  (h4 : prob_sunday_given_saturday = 0.7)
  : prob_friday * prob_saturday * prob_sunday_given_saturday = 0.105 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_three_days_l2014_201459


namespace NUMINAMATH_CALUDE_geometric_problem_l2014_201419

/-- Given a parabola and an ellipse with specific properties, prove the coordinates of intersection points, 
    the equation of a hyperbola, and the maximum area of a triangle. -/
theorem geometric_problem (a t : ℝ) (h_a_pos : a > 0) (h_a_range : a ∈ Set.Icc 1 2) (h_t : t > 4) :
  let C₁ := {(x, y) : ℝ × ℝ | y^2 = 4*a*x}
  let C := {(x, y) : ℝ × ℝ | x^2/(2*a^2) + y^2/a^2 = 1}
  let l := {(x, y) : ℝ × ℝ | y = x - a}
  let P := (4*a/3, a/3)
  let Q := ((3 - 2*Real.sqrt 2)*a, (2 - 2*Real.sqrt 2)*a)
  let A := (t, 0)
  let H := {(x, y) : ℝ × ℝ | 7*x^2 - 13*y^2 = 11*a^2}
  (P ∈ C ∧ P ∈ l) ∧
  (Q ∈ C₁ ∧ Q ∈ l) ∧
  (∃ Q' ∈ H, ∃ d : ℝ, d = 4*a ∧ (Q'.1 - Q.1)^2 + (Q'.2 - Q.2)^2 = d^2) ∧
  (∀ a' ∈ Set.Icc 1 2, 
    let S := abs ((P.1 - A.1)*(Q.2 - A.2) - (Q.1 - A.1)*(P.2 - A.2)) / 2
    S ≤ (Real.sqrt 2 - 5/6)*(2*t - 4)) ∧
  (∃ S : ℝ, S = (Real.sqrt 2 - 5/6)*(2*t - 4) ∧
    S = abs ((P.1 - A.1)*(Q.2 - A.2) - (Q.1 - A.1)*(P.2 - A.2)) / 2 ∧
    a = 2) :=
by sorry


end NUMINAMATH_CALUDE_geometric_problem_l2014_201419


namespace NUMINAMATH_CALUDE_max_y_value_l2014_201468

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : 
  ∃ (max_y : ℤ), y ≤ max_y ∧ max_y = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2014_201468


namespace NUMINAMATH_CALUDE_greatest_c_for_no_minus_seven_l2014_201415

theorem greatest_c_for_no_minus_seven : ∃ c : ℤ, 
  (∀ x : ℝ, x^2 + c*x + 20 ≠ -7) ∧
  (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 20 = -7) ∧
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_greatest_c_for_no_minus_seven_l2014_201415


namespace NUMINAMATH_CALUDE_polynomial_never_33_l2014_201403

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_never_33_l2014_201403


namespace NUMINAMATH_CALUDE_expression_value_l2014_201432

theorem expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 7) 
  (eq2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 113 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2014_201432


namespace NUMINAMATH_CALUDE_equation_solution_l2014_201440

theorem equation_solution (t : ℝ) : 
  (Real.sqrt (3 * Real.sqrt (3 * t - 6)) = (8 - t) ^ (1/4)) ↔ 
  (t = (-43 + Real.sqrt 2321) / 2 ∨ t = (-43 - Real.sqrt 2321) / 2) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2014_201440


namespace NUMINAMATH_CALUDE_floor_painting_problem_l2014_201498

def is_valid_pair (a b : ℕ) : Prop :=
  b > a ∧ 
  (a - 4) * (b - 4) = 2 * a * b / 3 ∧
  a > 4 ∧ b > 4

theorem floor_painting_problem :
  ∃! (pairs : List (ℕ × ℕ)), 
    pairs.length = 3 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ pairs ↔ is_valid_pair p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_floor_painting_problem_l2014_201498


namespace NUMINAMATH_CALUDE_carrots_theorem_l2014_201490

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 6

/-- The number of carrots Sam grew -/
def sam_carrots : ℕ := 3

/-- The total number of carrots grown -/
def total_carrots : ℕ := sandy_carrots + sam_carrots

theorem carrots_theorem : total_carrots = 9 := by sorry

end NUMINAMATH_CALUDE_carrots_theorem_l2014_201490


namespace NUMINAMATH_CALUDE_system_solution_l2014_201447

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = m - 3) → 
  (x - y = 2) → 
  (m = 8) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2014_201447


namespace NUMINAMATH_CALUDE_point_distance_and_inequality_l2014_201496

/-- The value of m for which the point P(m, 3) is at distance 4 from the line 4x-3y+1=0
    and satisfies the inequality 2x+y<3 -/
theorem point_distance_and_inequality (m : ℝ) : 
  (abs (4 * m - 3 * 3 + 1) / Real.sqrt (4^2 + (-3)^2) = 4) ∧ 
  (2 * m + 3 < 3) → 
  m = -3 := by sorry

end NUMINAMATH_CALUDE_point_distance_and_inequality_l2014_201496


namespace NUMINAMATH_CALUDE_wendys_bake_sale_l2014_201439

/-- Wendy's bake sale problem -/
theorem wendys_bake_sale
  (cupcakes : ℕ)
  (cookies : ℕ)
  (leftover : ℕ)
  (h1 : cupcakes = 4)
  (h2 : cookies = 29)
  (h3 : leftover = 24) :
  cupcakes + cookies - leftover = 9 :=
by sorry

end NUMINAMATH_CALUDE_wendys_bake_sale_l2014_201439


namespace NUMINAMATH_CALUDE_family_gave_forty_dollars_l2014_201486

/-- Represents the cost and composition of a family's movie outing -/
structure MovieOuting where
  regular_ticket_cost : ℕ
  child_discount : ℕ
  num_adults : ℕ
  num_children : ℕ
  change_received : ℕ

/-- Calculates the total amount given to the cashier for a movie outing -/
def total_amount_given (outing : MovieOuting) : ℕ :=
  let adult_cost := outing.regular_ticket_cost * outing.num_adults
  let child_cost := (outing.regular_ticket_cost - outing.child_discount) * outing.num_children
  let total_cost := adult_cost + child_cost
  total_cost + outing.change_received

/-- Theorem stating that the family gave the cashier $40 in total -/
theorem family_gave_forty_dollars :
  let outing : MovieOuting := {
    regular_ticket_cost := 9,
    child_discount := 2,
    num_adults := 2,
    num_children := 3,
    change_received := 1
  }
  total_amount_given outing = 40 := by sorry

end NUMINAMATH_CALUDE_family_gave_forty_dollars_l2014_201486


namespace NUMINAMATH_CALUDE_greek_cross_dissection_l2014_201477

/-- Represents a symmetric Greek cross -/
structure SymmetricGreekCross where
  -- Add necessary properties to define a symmetric Greek cross

/-- Represents a square -/
structure Square where
  -- Add necessary properties to define a square

/-- Represents a part of the dissected cross -/
inductive CrossPart
  | SmallCross : SymmetricGreekCross → CrossPart
  | OtherPart : CrossPart

/-- Theorem stating that a symmetric Greek cross can be dissected as described -/
theorem greek_cross_dissection (cross : SymmetricGreekCross) :
  ∃ (parts : Finset CrossPart) (square : Square),
    parts.card = 5 ∧
    (∃ small_cross : SymmetricGreekCross, CrossPart.SmallCross small_cross ∈ parts) ∧
    (∃ other_parts : Finset CrossPart,
      other_parts.card = 4 ∧
      (∀ p ∈ other_parts, p ∈ parts ∧ p ≠ CrossPart.SmallCross small_cross) ∧
      -- Here we would need to define how the other parts form the square
      True) := by
  sorry

end NUMINAMATH_CALUDE_greek_cross_dissection_l2014_201477


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2014_201431

-- Define the circle from part 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line x + y = 1
def line1 (x y : ℝ) : Prop := x + y = 1

-- Define the line y = -2x
def line2 (x y : ℝ) : Prop := y = -2 * x

-- Define the circle from part 2
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line from part 2
def line3 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem circle_and_line_properties :
  -- Part 1
  (circle1 2 (-1)) ∧ 
  (∃ (x y : ℝ), circle1 x y ∧ line1 x y) ∧
  (∃ (x y : ℝ), circle1 x y ∧ line2 x y) ∧
  -- Part 2
  (¬ circle2 2 (-2)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle2 x₁ y₁ ∧ circle2 x₂ y₂ ∧ 
    ((x₁ - 2) * (y₁ + 2) = 4) ∧ 
    ((x₂ - 2) * (y₂ + 2) = 4) ∧
    line3 x₁ y₁ ∧ line3 x₂ y₂) := by
  sorry

#check circle_and_line_properties

end NUMINAMATH_CALUDE_circle_and_line_properties_l2014_201431


namespace NUMINAMATH_CALUDE_hospital_opening_date_l2014_201402

theorem hospital_opening_date :
  ∃! (x y h : ℕ+),
    (x.val : ℤ) - (y.val : ℤ) = h.val ∨ (y.val : ℤ) - (x.val : ℤ) = h.val ∧
    x * (y * h - 1) = 1539 ∧
    h = 2 :=
by sorry

end NUMINAMATH_CALUDE_hospital_opening_date_l2014_201402


namespace NUMINAMATH_CALUDE_expression_evaluation_l2014_201430

theorem expression_evaluation (c d : ℤ) (hc : c = 2) (hd : d = 3) :
  (c^3 + d^2)^2 - (c^3 - d^2)^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2014_201430


namespace NUMINAMATH_CALUDE_sticker_distribution_equivalence_l2014_201474

/-- The number of ways to distribute n identical objects into k distinct containers --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute s identical stickers among p identical sheets,
    where each sheet must have at least 1 sticker --/
def distribute_stickers (s p : ℕ) : ℕ := stars_and_bars (s - p) p

theorem sticker_distribution_equivalence :
  distribute_stickers 10 5 = stars_and_bars 5 5 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_equivalence_l2014_201474


namespace NUMINAMATH_CALUDE_knight_seating_probability_correct_l2014_201495

/-- The probability of three knights seated at a round table with n chairs
    such that each knight has an empty chair on both sides. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else
    0

theorem knight_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n =
    (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knight_seating_probability_correct_l2014_201495


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2014_201425

theorem arithmetic_calculations :
  ((-20) + 3 + 5 + (-7) = -19) ∧
  (((-32) / 4) * (1 / 4) = -2) ∧
  ((2 / 7 - 1 / 4) * 28 = 1) ∧
  (-(2^4) * (((-3) * (-(2 + 1 + 1/3)) - (-5))) / ((-2/5)^2) = -1500) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2014_201425


namespace NUMINAMATH_CALUDE_diggers_holes_problem_l2014_201487

/-- Given that three diggers dug three holes in three hours,
    prove that six diggers will dig 10 holes in five hours. -/
theorem diggers_holes_problem (diggers_rate : ℚ) : 
  (diggers_rate = 3 / (3 * 3)) →  -- Rate of digging holes per digger per hour
  (6 * diggers_rate * 5 : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_diggers_holes_problem_l2014_201487


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2014_201444

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (3*(x-1) > x-6 ∧ 8-2*x+2*a ≥ 0) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  -3 ≤ a ∧ a < -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2014_201444


namespace NUMINAMATH_CALUDE_min_segments_polyline_l2014_201471

/-- Represents a square grid divided into n^2 smaller squares -/
structure SquareGrid (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents a polyline that passes through the centers of all smaller squares -/
structure Polyline (n : ℕ) where
  grid : SquareGrid n
  segments : ℕ
  passes_all_centers : segments ≥ 1

/-- Theorem stating the minimum number of segments in the polyline -/
theorem min_segments_polyline (n : ℕ) (h : n > 0) :
  ∃ (p : Polyline n), ∀ (q : Polyline n), p.segments ≤ q.segments ∧ p.segments = 2 * n - 2 :=
sorry

end NUMINAMATH_CALUDE_min_segments_polyline_l2014_201471
