import Mathlib

namespace NUMINAMATH_CALUDE_probability_calculations_l3392_339251

/-- Represents the number of students choosing each subject -/
structure SubjectCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  politics : ℕ
  history : ℕ
  geography : ℕ

/-- The total number of students -/
def totalStudents : ℕ := 1000

/-- The actual distribution of students across subjects -/
def actualCounts : SubjectCounts :=
  { physics := 300
  , chemistry := 200
  , biology := 100
  , politics := 200
  , history := 100
  , geography := 100 }

/-- Calculates the probability of an event given the number of favorable outcomes -/
def probability (favorableOutcomes : ℕ) : ℚ :=
  favorableOutcomes / totalStudents

/-- Theorem stating the probabilities of various events -/
theorem probability_calculations (counts : SubjectCounts) 
    (h : counts = actualCounts) : 
    probability counts.chemistry = 1/5 ∧ 
    probability (counts.biology + counts.history) = 1/5 ∧
    probability (counts.chemistry + counts.geography) = 3/10 := by
  sorry


end NUMINAMATH_CALUDE_probability_calculations_l3392_339251


namespace NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l3392_339271

/-- Conversion from rectangular to spherical coordinates -/
theorem rectangular_to_spherical_conversion
  (x y z : ℝ)
  (h_x : x = 3 * Real.sqrt 2)
  (h_y : y = -3)
  (h_z : z = 5)
  (h_rho_pos : 0 < Real.sqrt (x^2 + y^2 + z^2))
  (h_theta_range : 0 ≤ 2 * Real.pi - Real.arctan (1 / Real.sqrt 2) ∧ 
                   2 * Real.pi - Real.arctan (1 / Real.sqrt 2) < 2 * Real.pi)
  (h_phi_range : 0 ≤ Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2)) ∧ 
                 Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2)) ≤ Real.pi) :
  (Real.sqrt (x^2 + y^2 + z^2),
   2 * Real.pi - Real.arctan (1 / Real.sqrt 2),
   Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2))) =
  (Real.sqrt 52, 2 * Real.pi - Real.arctan (1 / Real.sqrt 2), Real.arccos (5 / Real.sqrt 52)) := by
  sorry

#check rectangular_to_spherical_conversion

end NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l3392_339271


namespace NUMINAMATH_CALUDE_soda_cost_is_one_l3392_339216

/-- The cost of one can of soda -/
def soda_cost : ℝ := 1

/-- The cost of one soup -/
def soup_cost : ℝ := 3 * soda_cost

/-- The cost of one sandwich -/
def sandwich_cost : ℝ := 3 * soup_cost

/-- The total cost of Sean's purchase -/
def total_cost : ℝ := 3 * soda_cost + 2 * soup_cost + sandwich_cost

theorem soda_cost_is_one :
  soda_cost = 1 ∧ total_cost = 18 := by sorry

end NUMINAMATH_CALUDE_soda_cost_is_one_l3392_339216


namespace NUMINAMATH_CALUDE_doors_per_apartment_l3392_339209

/-- Proves that the number of doors per apartment is 7, given the specifications of the apartment buildings and total doors needed. -/
theorem doors_per_apartment 
  (num_buildings : ℕ) 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (total_doors : ℕ) 
  (h1 : num_buildings = 2)
  (h2 : floors_per_building = 12)
  (h3 : apartments_per_floor = 6)
  (h4 : total_doors = 1008) :
  total_doors / (num_buildings * floors_per_building * apartments_per_floor) = 7 := by
  sorry

#check doors_per_apartment

end NUMINAMATH_CALUDE_doors_per_apartment_l3392_339209


namespace NUMINAMATH_CALUDE_equal_segments_l3392_339259

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (incenter : Point → Point → Point → Point)
variable (intersect_circle_line : Circle → Point → Point → Point)
variable (circle_through : Point → Point → Point → Circle)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem equal_segments 
  (A B C I X Y : Point) :
  I = incenter A B C →
  X = intersect_circle_line (circle_through A C I) B C →
  Y = intersect_circle_line (circle_through B C I) A C →
  length A Y = length B X :=
sorry

end NUMINAMATH_CALUDE_equal_segments_l3392_339259


namespace NUMINAMATH_CALUDE_parabola_properties_l3392_339230

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (2, 2) -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (parabola_equation 0 0) ∧ 
  (∃ p : ℝ, p > 0 ∧ parabola_equation p 0) ∧
  (parabola_equation 2 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3392_339230


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3392_339241

theorem greatest_divisor_with_remainders : 
  let a := 1657 - 6
  let b := 2037 - 5
  Nat.gcd a b = 127 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3392_339241


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l3392_339211

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 34 →
  (n * original_mean - n * decrement) / n = 166 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l3392_339211


namespace NUMINAMATH_CALUDE_congruent_triangles_sum_l3392_339220

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two triangles are congruent if their corresponding sides are equal -/
def congruent (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

theorem congruent_triangles_sum (x y : ℝ) :
  let t1 : Triangle := ⟨2, 5, x⟩
  let t2 : Triangle := ⟨y, 2, 6⟩
  congruent t1 t2 → x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_sum_l3392_339220


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3392_339250

theorem trigonometric_expression_value (α : Real) (h : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3392_339250


namespace NUMINAMATH_CALUDE_q_investment_proof_l3392_339293

/-- Calculates the investment of Q given the investment of P, total profit, and Q's profit share --/
def calculate_q_investment (p_investment : ℚ) (total_profit : ℚ) (q_profit_share : ℚ) : ℚ :=
  (p_investment * q_profit_share) / (total_profit - q_profit_share)

theorem q_investment_proof (p_investment : ℚ) (total_profit : ℚ) (q_profit_share : ℚ) 
  (h1 : p_investment = 54000)
  (h2 : total_profit = 18000)
  (h3 : q_profit_share = 6001.89) :
  calculate_q_investment p_investment total_profit q_profit_share = 27010 := by
  sorry

#eval calculate_q_investment 54000 18000 6001.89

end NUMINAMATH_CALUDE_q_investment_proof_l3392_339293


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l3392_339298

/-- Number of blue flags -/
def blue_flags : ℕ := 11

/-- Number of green flags -/
def green_flags : ℕ := 10

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Number of distinguishable arrangements -/
def M : ℕ := 660

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem flag_arrangement_remainder :
  M % 1000 = 660 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l3392_339298


namespace NUMINAMATH_CALUDE_sum_of_squares_l3392_339221

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 9)
  (eq2 : y^2 + 5*z = -9)
  (eq3 : z^2 + 7*x = -18) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3392_339221


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3392_339217

def a : ℝ × ℝ := (3, 0)
def b : ℝ × ℝ := (-5, 5)

theorem angle_between_vectors : 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3392_339217


namespace NUMINAMATH_CALUDE_no_solution_equation_simplify_fraction_l3392_339268

-- Problem 1
theorem no_solution_equation :
  ¬ ∃ x : ℝ, (3 - x) / (x - 4) - 1 / (4 - x) = 1 := by sorry

-- Problem 2
theorem simplify_fraction (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
  2 * x / (x^2 - 4) - 1 / (x + 2) = 1 / (x - 2) := by sorry

end NUMINAMATH_CALUDE_no_solution_equation_simplify_fraction_l3392_339268


namespace NUMINAMATH_CALUDE_min_value_of_f_l3392_339212

def f (x : ℝ) : ℝ := (x - 1)^2 + 3

theorem min_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3392_339212


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l3392_339208

theorem loss_percentage_calculation (CP : ℝ) :
  CP > 0 ∧ 
  240 = CP * (1 + 0.20) ∧ 
  170 < CP 
  → 
  (CP - 170) / CP * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l3392_339208


namespace NUMINAMATH_CALUDE_card_problem_solution_l3392_339277

/-- Represents the types of cards -/
inductive CardType
  | WW  -- White-White
  | BB  -- Black-Black
  | BW  -- Black-White

/-- Represents the state of a set of cards -/
structure CardSet where
  total : Nat
  blackUp : Nat

/-- Represents the problem setup -/
structure CardProblem where
  initialState : CardSet
  afterFirst : CardSet
  afterSecond : CardSet
  afterThird : CardSet

/-- The main theorem to prove -/
theorem card_problem_solution (p : CardProblem) : 
  p.initialState.total = 12 ∧ 
  p.initialState.blackUp = 9 ∧
  p.afterFirst.blackUp = 4 ∧
  p.afterSecond.blackUp = 6 ∧
  p.afterThird.blackUp = 5 →
  ∃ (bw ww : Nat), bw = 9 ∧ ww = 3 ∧ bw + ww = p.initialState.total := by
  sorry


end NUMINAMATH_CALUDE_card_problem_solution_l3392_339277


namespace NUMINAMATH_CALUDE_loop_termination_min_n_value_l3392_339247

def s (n : ℕ) : ℕ := 2010 / 2^n + 3 * (2^n - 1) / 2^(n-1)

theorem loop_termination : 
  ∀ k : ℕ, k < 5 → s k ≥ 120 ∧ s 5 < 120 :=
sorry

theorem min_n_value : (∃ n : ℕ, s n < 120) ∧ (∀ k : ℕ, s k < 120 → k ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_loop_termination_min_n_value_l3392_339247


namespace NUMINAMATH_CALUDE_interest_calculation_l3392_339260

/-- Calculates the total interest earned after 4 years given an initial investment
    and annual interest rates for each year. -/
def total_interest (initial_investment : ℝ) (rate1 rate2 rate3 rate4 : ℝ) : ℝ :=
  let final_amount := initial_investment * (1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4)
  final_amount - initial_investment

/-- Proves that the total interest earned after 4 years is approximately $572.36416
    given the specified initial investment and interest rates. -/
theorem interest_calculation :
  let initial_investment := 2000
  let rate1 := 0.05
  let rate2 := 0.06
  let rate3 := 0.07
  let rate4 := 0.08
  abs (total_interest initial_investment rate1 rate2 rate3 rate4 - 572.36416) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l3392_339260


namespace NUMINAMATH_CALUDE_lines_parallel_if_one_in_plane_one_parallel_to_plane_l3392_339291

-- Define the plane and lines
variable (α : Plane) (m n : Line)

-- Define the property of lines being coplanar
def coplanar (l₁ l₂ : Line) : Prop := sorry

-- Define the property of a line being contained in a plane
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- Define the property of a line being parallel to a plane
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the property of two lines being parallel
def parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem lines_parallel_if_one_in_plane_one_parallel_to_plane
  (h_coplanar : coplanar m n)
  (h_m_in_α : contained_in m α)
  (h_n_parallel_α : parallel_to_plane n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_if_one_in_plane_one_parallel_to_plane_l3392_339291


namespace NUMINAMATH_CALUDE_archies_sod_area_l3392_339263

/-- Calculates the area of sod needed for a rectangular backyard with a rectangular shed. -/
def area_of_sod_needed (backyard_length backyard_width shed_length shed_width : ℝ) : ℝ :=
  backyard_length * backyard_width - shed_length * shed_width

/-- Theorem: The area of sod needed for Archie's backyard is 245 square yards. -/
theorem archies_sod_area :
  area_of_sod_needed 20 13 3 5 = 245 := by
  sorry

end NUMINAMATH_CALUDE_archies_sod_area_l3392_339263


namespace NUMINAMATH_CALUDE_unique_prime_generator_l3392_339239

theorem unique_prime_generator : ∃! p : ℕ, Prime (p + 10) ∧ Prime (p + 14) :=
  ⟨3, 
    by {
      sorry -- Proof that 3 satisfies the conditions
    },
    by {
      sorry -- Proof that 3 is the only natural number satisfying the conditions
    }
  ⟩

end NUMINAMATH_CALUDE_unique_prime_generator_l3392_339239


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3392_339214

theorem largest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ a b : ℤ, 5 * x^2 + n * x + 48 = (5 * x + a) * (x + b)) → 
  n ≤ 241 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3392_339214


namespace NUMINAMATH_CALUDE_invalid_prism_diagonals_l3392_339227

/-- Represents the lengths of the extended diagonals of a right regular prism -/
structure PrismDiagonals where
  d1 : ℝ
  d2 : ℝ
  d3 : ℝ

/-- Checks if the given lengths can be the extended diagonals of a right regular prism -/
def is_valid_prism_diagonals (d : PrismDiagonals) : Prop :=
  ∃ (a b c : ℝ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (d.d1^2 = a^2 + b^2 ∧ d.d2^2 = b^2 + c^2 ∧ d.d3^2 = a^2 + c^2)

/-- The main theorem stating that {3, 4, 6} cannot be the lengths of extended diagonals -/
theorem invalid_prism_diagonals :
  ¬ is_valid_prism_diagonals ⟨3, 4, 6⟩ :=
sorry

end NUMINAMATH_CALUDE_invalid_prism_diagonals_l3392_339227


namespace NUMINAMATH_CALUDE_root_values_l3392_339222

theorem root_values (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_root_values_l3392_339222


namespace NUMINAMATH_CALUDE_prob_equals_927_4096_l3392_339218

/-- Recurrence relation for the number of valid sequences -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element sequence not containing three consecutive 1s -/
def prob : ℚ := b 12 / 2^12

theorem prob_equals_927_4096 : prob = 927 / 4096 := by sorry

end NUMINAMATH_CALUDE_prob_equals_927_4096_l3392_339218


namespace NUMINAMATH_CALUDE_sugar_problem_solution_l3392_339237

def sugar_problem (sugar_at_home : ℕ) (bags_bought : ℕ) (dozens : ℕ) 
  (sugar_per_dozen_batter : ℚ) (sugar_per_dozen_frosting : ℚ) : Prop :=
  ∃ (sugar_per_bag : ℕ),
    sugar_at_home = 3 ∧
    bags_bought = 2 ∧
    dozens = 5 ∧
    sugar_per_dozen_batter = 1 ∧
    sugar_per_dozen_frosting = 2 ∧
    sugar_at_home + bags_bought * sugar_per_bag = 
      dozens * (sugar_per_dozen_batter + sugar_per_dozen_frosting) ∧
    sugar_per_bag = 6

theorem sugar_problem_solution :
  sugar_problem 3 2 5 1 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_problem_solution_l3392_339237


namespace NUMINAMATH_CALUDE_equation_solution_l3392_339286

theorem equation_solution : ∃! y : ℝ, 4 + 2.3 * y = 1.7 * y - 20 :=
by
  use -40
  constructor
  · -- Prove that y = -40 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_equation_solution_l3392_339286


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3392_339245

-- Define the original number
def original_number : ℕ := 150000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.5 * (10 ^ 11)

-- Theorem to prove the equality
theorem original_equals_scientific : (original_number : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3392_339245


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3392_339275

theorem arctan_equation_solution :
  ∃ x : ℚ, 2 * Real.arctan (1/3) + 4 * Real.arctan (1/5) + Real.arctan (1/x) = π/4 ∧ x = -978/2029 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3392_339275


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3392_339265

/-- The number of questions asked to the Magic 8 Ball -/
def num_questions : ℕ := 7

/-- The number of possible responses from the Magic 8 Ball -/
def num_responses : ℕ := 3

/-- The probability of each type of response -/
def response_probability : ℚ := 1 / 3

/-- The number of desired positive responses -/
def desired_positive : ℕ := 3

/-- The number of desired neutral responses -/
def desired_neutral : ℕ := 2

/-- Theorem stating the probability of getting exactly 3 positive answers and 2 neutral answers
    when asking a Magic 8 Ball 7 questions, where each type of response has an equal probability of 1/3 -/
theorem magic_8_ball_probability :
  (Nat.choose num_questions desired_positive *
   Nat.choose (num_questions - desired_positive) desired_neutral *
   response_probability ^ num_questions) = 70 / 243 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3392_339265


namespace NUMINAMATH_CALUDE_max_tasty_compote_weight_l3392_339249

theorem max_tasty_compote_weight 
  (fresh_apples : ℝ) 
  (dried_apples : ℝ) 
  (fresh_water_content : ℝ) 
  (dried_water_content : ℝ) 
  (max_water_content : ℝ) :
  fresh_apples = 4 →
  dried_apples = 1 →
  fresh_water_content = 0.9 →
  dried_water_content = 0.12 →
  max_water_content = 0.95 →
  ∃ (max_compote : ℝ),
    max_compote = 25.6 ∧
    ∀ (added_water : ℝ),
      (fresh_apples * fresh_water_content + 
       dried_apples * dried_water_content + 
       added_water) / 
      (fresh_apples + dried_apples + added_water) ≤ max_water_content →
      fresh_apples + dried_apples + added_water ≤ max_compote :=
by sorry

end NUMINAMATH_CALUDE_max_tasty_compote_weight_l3392_339249


namespace NUMINAMATH_CALUDE_g_difference_l3392_339274

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

-- State the theorem
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l3392_339274


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l3392_339224

/-- Given two lines with the same non-zero y-intercept, where the first line has slope 12
    and x-intercept (u, 0), and the second line has slope 8 and x-intercept (v, 0),
    prove that the ratio of u to v is 2/3. -/
theorem ratio_of_x_intercepts (b : ℝ) (u v : ℝ) (h1 : b ≠ 0)
    (h2 : 12 * u + b = 0) (h3 : 8 * v + b = 0) : u / v = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l3392_339224


namespace NUMINAMATH_CALUDE_triangle_base_length_l3392_339202

/-- Theorem: The base of a triangle with specific side lengths -/
theorem triangle_base_length (left_side right_side base : ℝ) : 
  left_side = 12 →
  right_side = left_side + 2 →
  left_side + right_side + base = 50 →
  base = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3392_339202


namespace NUMINAMATH_CALUDE_smallest_special_number_l3392_339229

theorem smallest_special_number : ∃ (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (k : ℕ), n = 2 * k) ∧
  (∃ (k : ℕ), n + 1 = 3 * k) ∧
  (∃ (k : ℕ), n + 2 = 4 * k) ∧
  (∃ (k : ℕ), n + 3 = 5 * k) ∧
  (∃ (k : ℕ), n + 4 = 6 * k) ∧
  (∀ m : ℕ, m < n → 
    ¬((100 ≤ m ∧ m < 1000) ∧ 
      (∃ (k : ℕ), m = 2 * k) ∧
      (∃ (k : ℕ), m + 1 = 3 * k) ∧
      (∃ (k : ℕ), m + 2 = 4 * k) ∧
      (∃ (k : ℕ), m + 3 = 5 * k) ∧
      (∃ (k : ℕ), m + 4 = 6 * k))) ∧
  n = 122 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l3392_339229


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l3392_339213

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 8

/-- The number of independent transpositions -/
def num_transpositions : ℕ := 3

/-- The probability that a specific ball is chosen in any swap -/
def prob_chosen : ℚ := 1 / 4

/-- The probability that a ball is not chosen in a single swap -/
def prob_not_chosen : ℚ := 1 - prob_chosen

/-- The probability that a ball is in its original position after all transpositions -/
def prob_original_position : ℚ := prob_not_chosen ^ num_transpositions + 
  num_transpositions * prob_chosen ^ 2 * prob_not_chosen

/-- The expected number of balls in their original positions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

theorem expected_balls_in_original_position :
  expected_original_positions = 9 / 2 := by sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l3392_339213


namespace NUMINAMATH_CALUDE_tomato_plant_problem_l3392_339238

theorem tomato_plant_problem (initial_tomatoes : ℕ) : 
  (initial_tomatoes : ℚ) - (1/4 * initial_tomatoes + 20 + 40 : ℚ) = 15 → 
  initial_tomatoes = 100 := by
sorry

end NUMINAMATH_CALUDE_tomato_plant_problem_l3392_339238


namespace NUMINAMATH_CALUDE_transform_minus3_minus8i_l3392_339233

def rotate90 (z : ℂ) : ℂ := z * Complex.I

def dilate2 (z : ℂ) : ℂ := 2 * z

def transform (z : ℂ) : ℂ := dilate2 (rotate90 z)

theorem transform_minus3_minus8i :
  transform (-3 - 8 * Complex.I) = 16 - 6 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_transform_minus3_minus8i_l3392_339233


namespace NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_cos_54_l3392_339279

theorem cos_24_cos_36_minus_sin_24_cos_54 : 
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.sin (24 * π / 180) * Real.cos (54 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_cos_54_l3392_339279


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3392_339254

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3392_339254


namespace NUMINAMATH_CALUDE_velocity_maximum_at_lowest_point_l3392_339246

/-- Represents a point on the roller coaster track -/
structure TrackPoint where
  height : ℝ
  velocity : ℝ

/-- Represents the roller coaster system -/
structure RollerCoaster where
  points : List TrackPoint
  initial_velocity : ℝ
  g : ℝ  -- Acceleration due to gravity

/-- The total mechanical energy of the system -/
def total_energy (rc : RollerCoaster) (p : TrackPoint) : ℝ :=
  0.5 * p.velocity^2 + rc.g * p.height

/-- The point with minimum height has maximum velocity -/
theorem velocity_maximum_at_lowest_point (rc : RollerCoaster) :
  ∀ p q : TrackPoint,
    p ∈ rc.points →
    q ∈ rc.points →
    p.height < q.height →
    total_energy rc p = total_energy rc q →
    p.velocity > q.velocity :=
sorry

end NUMINAMATH_CALUDE_velocity_maximum_at_lowest_point_l3392_339246


namespace NUMINAMATH_CALUDE_max_dimes_count_l3392_339228

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Liam has in dollars -/
def total_money : ℚ := 4.80

/-- 
Given that Liam has $4.80 in U.S. coins and an equal number of dimes and nickels,
this theorem states that the maximum number of dimes he could have is 32.
-/
theorem max_dimes_count : 
  ∃ (d : ℕ), d * (dime_value + nickel_value) = total_money ∧ 
             ∀ (x : ℕ), x * (dime_value + nickel_value) ≤ total_money → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_dimes_count_l3392_339228


namespace NUMINAMATH_CALUDE_sports_conference_games_l3392_339289

/-- Calculates the number of games in a sports conference season -/
def conference_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

/-- Theorem: The number of games in the described sports conference is 232 -/
theorem sports_conference_games : 
  conference_games 16 8 3 1 = 232 := by sorry

end NUMINAMATH_CALUDE_sports_conference_games_l3392_339289


namespace NUMINAMATH_CALUDE_permutation_inequality_l3392_339225

theorem permutation_inequality (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : 
  a₁ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₂ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₃ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₄ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₅ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₆ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧
  a₅ ≠ a₆ →
  (a₁ + 1) / 2 * (a₂ + 2) / 2 * (a₃ + 3) / 2 * (a₄ + 4) / 2 * (a₅ + 5) / 2 * (a₆ + 6) / 2 < 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequality_l3392_339225


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l3392_339282

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l3392_339282


namespace NUMINAMATH_CALUDE_holly_throws_five_times_l3392_339252

/-- Represents the Frisbee throwing scenario -/
structure FrisbeeScenario where
  bess_throw_distance : ℕ
  bess_throw_count : ℕ
  holly_throw_distance : ℕ
  total_distance : ℕ

/-- Calculates the number of times Holly throws the Frisbee -/
def holly_throw_count (scenario : FrisbeeScenario) : ℕ :=
  (scenario.total_distance - 2 * scenario.bess_throw_distance * scenario.bess_throw_count) / scenario.holly_throw_distance

/-- Theorem stating that Holly throws the Frisbee 5 times in the given scenario -/
theorem holly_throws_five_times (scenario : FrisbeeScenario) 
  (h1 : scenario.bess_throw_distance = 20)
  (h2 : scenario.bess_throw_count = 4)
  (h3 : scenario.holly_throw_distance = 8)
  (h4 : scenario.total_distance = 200) :
  holly_throw_count scenario = 5 := by
  sorry

#eval holly_throw_count { bess_throw_distance := 20, bess_throw_count := 4, holly_throw_distance := 8, total_distance := 200 }

end NUMINAMATH_CALUDE_holly_throws_five_times_l3392_339252


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3392_339201

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3392_339201


namespace NUMINAMATH_CALUDE_larger_number_is_448_l3392_339272

/-- Given two positive integers with specific HCF and LCM properties, prove the larger number is 448 -/
theorem larger_number_is_448 (a b : ℕ+) : 
  (Nat.gcd a b = 32) →
  (∃ (x y : ℕ+), x = 13 ∧ y = 14 ∧ Nat.lcm a b = 32 * x * y) →
  max a b = 448 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_448_l3392_339272


namespace NUMINAMATH_CALUDE_number_operation_l3392_339281

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 24) / 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l3392_339281


namespace NUMINAMATH_CALUDE_geometric_mean_of_two_and_six_l3392_339240

theorem geometric_mean_of_two_and_six :
  ∃ (x : ℝ), x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = -2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_two_and_six_l3392_339240


namespace NUMINAMATH_CALUDE_total_salaries_l3392_339248

/-- The problem of calculating total salaries given specific conditions -/
theorem total_salaries (A_salary B_salary : ℝ) : 
  A_salary = 2250 →
  0.05 * A_salary = 0.15 * B_salary →
  A_salary + B_salary = 3000 := by
  sorry

#check total_salaries

end NUMINAMATH_CALUDE_total_salaries_l3392_339248


namespace NUMINAMATH_CALUDE_investment_sum_l3392_339283

/-- Proves that if a sum P is invested at 15% p.a. for two years instead of 12% p.a. for two years, 
    and the difference in interest is Rs. 840, then P = Rs. 14,000. -/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 840) → P = 14000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l3392_339283


namespace NUMINAMATH_CALUDE_rice_yield_and_conversion_l3392_339278

-- Define the yield per acre of ordinary rice
def ordinary_yield : ℝ := 600

-- Define the yield per acre of hybrid rice
def hybrid_yield : ℝ := 2 * ordinary_yield

-- Define the acreage difference between fields
def acreage_difference : ℝ := 4

-- Define the total yield of field A
def field_A_yield : ℝ := 9600

-- Define the total yield of field B
def field_B_yield : ℝ := 7200

-- Define the minimum total yield after conversion
def min_total_yield : ℝ := 17700

-- Theorem statement
theorem rice_yield_and_conversion :
  -- Prove that the ordinary yield is 600 kg/acre
  ordinary_yield = 600 ∧
  -- Prove that the hybrid yield is 1200 kg/acre
  hybrid_yield = 1200 ∧
  -- Prove that at least 1.5 acres of field B should be converted
  ∃ (converted_acres : ℝ),
    converted_acres ≥ 1.5 ∧
    field_A_yield + 
    ordinary_yield * (field_B_yield / ordinary_yield - converted_acres) + 
    hybrid_yield * converted_acres ≥ min_total_yield :=
by sorry

end NUMINAMATH_CALUDE_rice_yield_and_conversion_l3392_339278


namespace NUMINAMATH_CALUDE_circumcircle_radius_right_triangle_l3392_339264

/-- The radius of the circumcircle of a triangle with side lengths 8, 15, and 17 is 17/2 -/
theorem circumcircle_radius_right_triangle : 
  ∀ (a b c : ℝ), 
  a = 8 → b = 15 → c = 17 →
  a^2 + b^2 = c^2 →
  (∃ (r : ℝ), r = c / 2 ∧ r = 17 / 2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_right_triangle_l3392_339264


namespace NUMINAMATH_CALUDE_unique_arrangements_count_l3392_339231

/-- The number of letters in the word -/
def word_length : ℕ := 7

/-- The number of identical letters (B and S) -/
def identical_letters : ℕ := 2

/-- Calculates the number of unique arrangements for the given word -/
def unique_arrangements : ℕ := (Nat.factorial word_length) / (Nat.factorial identical_letters)

/-- Theorem stating that the number of unique arrangements is 2520 -/
theorem unique_arrangements_count : unique_arrangements = 2520 := by
  sorry

end NUMINAMATH_CALUDE_unique_arrangements_count_l3392_339231


namespace NUMINAMATH_CALUDE_no_natural_solutions_l3392_339210

theorem no_natural_solutions : ∀ x y : ℕ, x^2 + x*y + y^2 ≠ x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l3392_339210


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l3392_339256

/-- The positive difference between the two largest prime factors of 204204 is 16 -/
theorem largest_prime_factors_difference : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  p ∣ 204204 ∧ q ∣ 204204 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 204204 → r ≤ p) ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 204204 → r ≠ p → r ≤ q) ∧
  p - q = 16 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l3392_339256


namespace NUMINAMATH_CALUDE_parallel_lines_max_distance_l3392_339223

/-- Two parallel lines with maximum distance -/
theorem parallel_lines_max_distance :
  ∃ (k b₁ b₂ : ℝ),
    -- Line equations
    (∀ x y, y = k * x + b₁ ↔ 3 * x + 5 * y + 16 = 0) ∧
    (∀ x y, y = k * x + b₂ ↔ 3 * x + 5 * y - 18 = 0) ∧
    -- Lines pass through given points
    (-2 = k * (-2) + b₁) ∧
    (3 = k * 1 + b₂) ∧
    -- Lines are parallel
    (∀ x y₁ y₂, y₁ = k * x + b₁ ∧ y₂ = k * x + b₂ → y₂ - y₁ = b₂ - b₁) ∧
    -- Distance between lines is maximum
    (∀ k' b₁' b₂',
      ((-2 = k' * (-2) + b₁') ∧ (3 = k' * 1 + b₂')) →
      |b₂ - b₁| / Real.sqrt (1 + k^2) ≥ |b₂' - b₁'| / Real.sqrt (1 + k'^2)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_max_distance_l3392_339223


namespace NUMINAMATH_CALUDE_min_area_sum_l3392_339267

def point := ℝ × ℝ

def triangle_area (p1 p2 p3 : point) : ℝ := sorry

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem min_area_sum (m : ℝ) :
  let p1 : point := (2, 8)
  let p2 : point := (12, 20)
  let p3 : point := (8, m)
  is_integer m →
  (∀ k : ℝ, is_integer k → 
    k ≠ 15.2 → 
    triangle_area p1 p2 (8, k) ≥ triangle_area p1 p2 p3) →
  m ≠ 15.2 →
  (∃ n : ℝ, is_integer n ∧ 
    n ≠ 15.2 ∧
    triangle_area p1 p2 (8, n) = triangle_area p1 p2 p3 ∧
    |m - 15.2| + |n - 15.2| = |14 - 15.2| + |16 - 15.2|) →
  m + (30 - m) = 30 := by sorry

end NUMINAMATH_CALUDE_min_area_sum_l3392_339267


namespace NUMINAMATH_CALUDE_min_value_of_polynomial_l3392_339219

theorem min_value_of_polynomial (x : ℝ) : 
  ∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), x * (x + 1) * (x + 2) * (x + 3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_polynomial_l3392_339219


namespace NUMINAMATH_CALUDE_extremum_value_theorem_l3392_339285

/-- The function f(x) = x sin x achieves an extremum at x₀ -/
def has_extremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ f x ≥ f x₀

theorem extremum_value_theorem (x₀ : ℝ) :
  has_extremum (fun x => x * Real.sin x) x₀ →
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_extremum_value_theorem_l3392_339285


namespace NUMINAMATH_CALUDE_alyssa_soccer_games_l3392_339203

theorem alyssa_soccer_games (this_year last_year next_year total : ℕ) 
  (h1 : this_year = 11)
  (h2 : last_year = 13)
  (h3 : next_year = 15)
  (h4 : total = 39)
  (h5 : this_year + last_year + next_year = total) : 
  this_year - (total - (last_year + next_year)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_soccer_games_l3392_339203


namespace NUMINAMATH_CALUDE_problem_statement_l3392_339253

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + 2 * a + b = 16) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 16 ∧ x * y > a * b) →
    a * b ≤ 8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → 2 * x + y ≥ 8) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → x + y ≥ 6 * Real.sqrt 2 - 3) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → 1 / (x + 1) + 1 / (y + 2) > Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3392_339253


namespace NUMINAMATH_CALUDE_fraction_of_ivys_collectors_dolls_l3392_339294

/-- The number of dolls Dina has -/
def dinas_dolls : ℕ := 60

/-- The number of collectors edition dolls Ivy has -/
def ivys_collectors_dolls : ℕ := 20

/-- The number of dolls Ivy has -/
def ivys_dolls : ℕ := dinas_dolls / 2

theorem fraction_of_ivys_collectors_dolls : 
  (ivys_collectors_dolls : ℚ) / (ivys_dolls : ℚ) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_of_ivys_collectors_dolls_l3392_339294


namespace NUMINAMATH_CALUDE_number_count_proof_l3392_339226

theorem number_count_proof (total_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ) :
  total_avg = 2.5 →
  group1_avg = 1.1 →
  group2_avg = 1.4 →
  group3_avg = 5 →
  ∃ (n : ℕ), n = 6 ∧ 
    n * total_avg = 2 * group1_avg + 2 * group2_avg + 2 * group3_avg :=
by sorry

end NUMINAMATH_CALUDE_number_count_proof_l3392_339226


namespace NUMINAMATH_CALUDE_bookstore_problem_l3392_339270

/-- Represents the bookstore's notebook purchase and sales problem -/
theorem bookstore_problem (total_notebooks : ℕ) (cost_a cost_b total_cost : ℕ) 
  (sell_a sell_b : ℕ) (discount_a : ℚ) (profit_threshold : ℕ) :
  total_notebooks = 350 →
  cost_a = 12 →
  cost_b = 15 →
  total_cost = 4800 →
  sell_a = 20 →
  sell_b = 25 →
  discount_a = 0.7 →
  profit_threshold = 2348 →
  ∃ (notebooks_a notebooks_b m : ℕ),
    notebooks_a + notebooks_b = total_notebooks ∧
    cost_a * notebooks_a + cost_b * notebooks_b = total_cost ∧
    notebooks_a = 150 ∧
    notebooks_b = 200 ∧
    m = 111 ∧
    (m : ℚ) * (sell_a + sell_b : ℚ) + 
      ((notebooks_a - m : ℚ) * sell_a * discount_a + (notebooks_b - m : ℚ) * cost_b) - 
      (total_cost : ℚ) ≥ profit_threshold := by
  sorry

end NUMINAMATH_CALUDE_bookstore_problem_l3392_339270


namespace NUMINAMATH_CALUDE_gcd_65_169_l3392_339257

theorem gcd_65_169 : Nat.gcd 65 169 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_65_169_l3392_339257


namespace NUMINAMATH_CALUDE_complement_union_eq_result_l3392_339288

-- Define the universal set U
def U : Set ℕ := {x | 0 ≤ x ∧ x < 5}

-- Define sets P and Q
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {2, 4}

-- Theorem statement
theorem complement_union_eq_result : (U \ P) ∪ Q = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_eq_result_l3392_339288


namespace NUMINAMATH_CALUDE_all_transylvanians_answer_yes_l3392_339262

-- Define the types of Transylvanians
inductive TransylvanianType
  | SaneHuman
  | InsaneHuman
  | SaneVampire
  | InsaneVampire

-- Define the possible questions
inductive Question
  | ConsiderHuman
  | Reliable

-- Define the function that represents a Transylvanian's answer
def transylvanianAnswer (t : TransylvanianType) (q : Question) : Bool :=
  match q with
  | Question.ConsiderHuman => true
  | Question.Reliable => true

-- Theorem statement
theorem all_transylvanians_answer_yes
  (t : TransylvanianType) (q : Question) :
  transylvanianAnswer t q = true := by sorry

end NUMINAMATH_CALUDE_all_transylvanians_answer_yes_l3392_339262


namespace NUMINAMATH_CALUDE_people_dislike_both_radio_and_music_l3392_339206

def total_polled : ℕ := 1200
def radio_dislike_percent : ℚ := 30 / 100
def music_and_radio_dislike_percent : ℚ := 10 / 100

theorem people_dislike_both_radio_and_music :
  (total_polled : ℚ) * radio_dislike_percent * music_and_radio_dislike_percent = 36 := by
  sorry

end NUMINAMATH_CALUDE_people_dislike_both_radio_and_music_l3392_339206


namespace NUMINAMATH_CALUDE_anna_candy_store_l3392_339299

def candy_store_problem (initial_amount : ℚ) 
                        (gum_price : ℚ) (gum_quantity : ℕ)
                        (chocolate_price : ℚ) (chocolate_quantity : ℕ)
                        (candy_cane_price : ℚ) (candy_cane_quantity : ℕ) : Prop :=
  let total_spent := gum_price * gum_quantity + 
                     chocolate_price * chocolate_quantity + 
                     candy_cane_price * candy_cane_quantity
  initial_amount - total_spent = 1

theorem anna_candy_store : 
  candy_store_problem 10 1 3 1 5 (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_anna_candy_store_l3392_339299


namespace NUMINAMATH_CALUDE_supermarket_sales_l3392_339235

-- Define the sales volume function
def P (k b t : ℕ) : ℕ := k * t + b

-- Define the unit price function
def Q (t : ℕ) : ℕ :=
  if t < 25 then t + 20 else 80 - t

-- Define the daily sales revenue function
def Y (k b t : ℕ) : ℕ := P k b t * Q t

theorem supermarket_sales (k b : ℕ) :
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30) →
  P k b 5 = 55 →
  P k b 10 = 50 →
  (P k b 20 = 40) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → Y k b t ≤ 2395) ∧
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ Y k b t = 2395) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_sales_l3392_339235


namespace NUMINAMATH_CALUDE_inequality_proof_l3392_339276

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  (m - n) / (Real.log m - Real.log n) < (m + n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3392_339276


namespace NUMINAMATH_CALUDE_fraction_replacement_l3392_339215

theorem fraction_replacement (x : ℚ) :
  ((5 / 2 / x * 5 / 2) / (5 / 2 * x / (5 / 2))) = 25 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_replacement_l3392_339215


namespace NUMINAMATH_CALUDE_consecutive_integers_deduction_l3392_339284

theorem consecutive_integers_deduction (n : ℕ) (avg : ℚ) (new_avg : ℚ) : 
  n = 30 → 
  avg = 50 → 
  new_avg = 34.3 → 
  let sum := n * avg
  let first_deduction := 29
  let last_deduction := 1
  let deduction_sum := n.pred / 2 * (first_deduction + last_deduction)
  let final_deduction := 6 + 12 + 18
  let new_sum := sum - deduction_sum - final_deduction
  new_avg = new_sum / n := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_deduction_l3392_339284


namespace NUMINAMATH_CALUDE_train_speed_proof_l3392_339258

-- Define the given parameters
def train_length : ℝ := 155
def bridge_length : ℝ := 220
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def m_s_to_km_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_proof :
  let total_distance := train_length + bridge_length
  let speed_m_s := total_distance / crossing_time
  let speed_km_hr := speed_m_s * m_s_to_km_hr
  speed_km_hr = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l3392_339258


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l3392_339287

theorem min_value_quadratic_form (x y : ℝ) :
  2 * x^2 + 3 * x * y + 2 * y^2 ≥ 0 ∧
  (2 * x^2 + 3 * x * y + 2 * y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l3392_339287


namespace NUMINAMATH_CALUDE_parabola_focal_distance_l3392_339204

/-- Given a parabola y^2 = 2px with focus F and a point A(1, 2) on the parabola, |AF| = 2 -/
theorem parabola_focal_distance (p : ℝ) (F : ℝ × ℝ) :
  (∀ x y, y^2 = 2*p*x → (x, y) = (1, 2)) →  -- point A(1, 2) satisfies the parabola equation
  F.1 = p/2 →  -- x-coordinate of focus
  F.2 = 0 →  -- y-coordinate of focus
  let A := (1, 2)
  ((A.1 - F.1)^2 + (A.2 - F.2)^2).sqrt = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focal_distance_l3392_339204


namespace NUMINAMATH_CALUDE_common_divisor_and_remainder_l3392_339236

theorem common_divisor_and_remainder (a b c d : ℕ) : 
  a = 2613 ∧ b = 2243 ∧ c = 1503 ∧ d = 985 →
  ∃ (k : ℕ), k > 0 ∧ 
    k ∣ (a - b) ∧ k ∣ (b - c) ∧ k ∣ (c - d) ∧
    ∀ m : ℕ, m > k → ¬(m ∣ (a - b) ∧ m ∣ (b - c) ∧ m ∣ (c - d)) ∧
    a % k = b % k ∧ b % k = c % k ∧ c % k = d % k ∧
    k = 74 ∧ a % k = 23 :=
by sorry

end NUMINAMATH_CALUDE_common_divisor_and_remainder_l3392_339236


namespace NUMINAMATH_CALUDE_circumcircle_tangent_to_excircle_l3392_339280

-- Define the points and circles
variable (A B C D E B₁ C₁ I J S : Point)
variable (Ω : Circle)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D E : Point) : Prop := sorry

-- Define the common excircle of triangles
def common_excircle (A B C D E : Point) (Ω : Circle) : Prop := sorry

-- Define tangent points
def tangent_points (A E D B₁ C₁ : Point) (Ω : Circle) : Prop := sorry

-- Define incircle centers
def incircle_centers (A B E C D I J : Point) : Prop := sorry

-- Define intersection of IC₁ and JB₁
def segments_intersect_at (I C₁ J B₁ S : Point) : Prop := sorry

-- Define S lying on Ω
def point_on_circle (S : Point) (Ω : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def circumcircle (A E D : Point) : Circle := sorry

-- Define tangency of circles
def circles_tangent (c₁ c₂ : Circle) : Prop := sorry

-- Theorem statement
theorem circumcircle_tangent_to_excircle 
  (h₁ : is_convex_quadrilateral A B C D)
  (h₂ : diagonals_intersect_at A B C D E)
  (h₃ : common_excircle A B C D E Ω)
  (h₄ : tangent_points A E D B₁ C₁ Ω)
  (h₅ : incircle_centers A B E C D I J)
  (h₆ : segments_intersect_at I C₁ J B₁ S)
  (h₇ : point_on_circle S Ω) :
  circles_tangent (circumcircle A E D) Ω :=
sorry

end NUMINAMATH_CALUDE_circumcircle_tangent_to_excircle_l3392_339280


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3392_339296

/-- An isosceles triangle with sides a, b, and c, where at least two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : (a = b ∧ a > 0) ∨ (b = c ∧ b > 0) ∨ (a = c ∧ a > 0)

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: An isosceles triangle with one side of length 6 and another of length 5 
    has a perimeter of either 16 or 17 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 6 ∧ (t.b = 5 ∨ t.c = 5)) ∨ (t.b = 6 ∧ (t.a = 5 ∨ t.c = 5)) ∨ (t.c = 6 ∧ (t.a = 5 ∨ t.b = 5))) →
  (perimeter t = 16 ∨ perimeter t = 17) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3392_339296


namespace NUMINAMATH_CALUDE_max_y_over_x_l3392_339295

theorem max_y_over_x (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 ∧ ∃ (x₀ y₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ 1 / x₀ + 2 * y₀ = 3 ∧ y₀ / x₀ = 9 / 8 :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3392_339295


namespace NUMINAMATH_CALUDE_final_number_calculation_l3392_339266

theorem final_number_calculation : ∃ (n : ℕ), n = 5 ∧ (3 * ((2 * n) + 9) = 57) := by
  sorry

end NUMINAMATH_CALUDE_final_number_calculation_l3392_339266


namespace NUMINAMATH_CALUDE_car_speed_conversion_l3392_339292

/-- Conversion factor from m/s to km/h -/
def conversion_factor : ℝ := 3.6

/-- Speed of the car in m/s -/
def speed_ms : ℝ := 10

/-- Speed of the car in km/h -/
def speed_kmh : ℝ := speed_ms * conversion_factor

theorem car_speed_conversion :
  speed_kmh = 36 := by sorry

end NUMINAMATH_CALUDE_car_speed_conversion_l3392_339292


namespace NUMINAMATH_CALUDE_convex_polygon_23_sides_diagonals_l3392_339234

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 23 sides has 230 diagonals -/
theorem convex_polygon_23_sides_diagonals :
  num_diagonals 23 = 230 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_23_sides_diagonals_l3392_339234


namespace NUMINAMATH_CALUDE_photo_frame_border_area_l3392_339232

/-- The area of the border surrounding a rectangular photograph -/
theorem photo_frame_border_area (photo_height photo_width border_width : ℕ) : 
  photo_height = 12 →
  photo_width = 15 →
  border_width = 3 →
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - photo_height * photo_width = 198 := by
  sorry

#check photo_frame_border_area

end NUMINAMATH_CALUDE_photo_frame_border_area_l3392_339232


namespace NUMINAMATH_CALUDE_gym_membership_ratio_l3392_339200

theorem gym_membership_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (35 * f + 30 * m) / (f + m) = 32 → f / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_gym_membership_ratio_l3392_339200


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3392_339269

theorem fraction_inequality_solution_set (x : ℝ) :
  x ≠ 1 → ((x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3392_339269


namespace NUMINAMATH_CALUDE_astros_win_in_seven_l3392_339242

/-- The probability of the Dodgers winning a single game -/
def p_dodgers : ℚ := 3/4

/-- The probability of the Astros winning a single game -/
def p_astros : ℚ := 1 - p_dodgers

/-- The number of games needed to win the World Series -/
def games_to_win : ℕ := 4

/-- The total number of games in a full World Series -/
def total_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Astros winning the World Series in exactly 7 games -/
def p_astros_win_in_seven : ℚ := 135/4096

theorem astros_win_in_seven :
  p_astros_win_in_seven = (Nat.choose 6 3 : ℚ) * p_astros^3 * p_dodgers^3 * p_astros := by sorry

end NUMINAMATH_CALUDE_astros_win_in_seven_l3392_339242


namespace NUMINAMATH_CALUDE_more_birds_than_nests_l3392_339261

/-- Given 6 birds and 3 nests, prove that there are 3 more birds than nests. -/
theorem more_birds_than_nests (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) (h2 : nests = 3) : birds - nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_more_birds_than_nests_l3392_339261


namespace NUMINAMATH_CALUDE_tangent_line_intersection_three_distinct_solutions_l3392_339290

/-- The function f(x) = x³ - 9x -/
def f (x : ℝ) : ℝ := x^3 - 9*x

/-- The function g(x) = 3x² + a -/
def g (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 9

/-- The derivative of g -/
def g' (x : ℝ) : ℝ := 6*x

theorem tangent_line_intersection (a : ℝ) :
  (∃ m : ℝ, f' 0 = g' m ∧ f 0 + f' 0 * m = g a m) → a = 27/4 :=
sorry

theorem three_distinct_solutions (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = g a x ∧ f y = g a y ∧ f z = g a z) ↔ 
  -27 < a ∧ a < 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_three_distinct_solutions_l3392_339290


namespace NUMINAMATH_CALUDE_problem_solution_l3392_339243

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {a, 2, 2*a - 1}

-- State the theorem
theorem problem_solution :
  ∃ (a : ℝ), A ⊆ B a ∧ A = {2, 3} ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3392_339243


namespace NUMINAMATH_CALUDE_third_root_of_polynomial_l3392_339205

theorem third_root_of_polynomial (a b : ℚ) : 
  (∀ x : ℚ, a * x^3 + (a + 3*b) * x^2 + (2*b - 4*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = -24/19) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_polynomial_l3392_339205


namespace NUMINAMATH_CALUDE_min_value_of_function_l3392_339273

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let f := fun x => 4 * x + 2 / x
  (∀ y > 0, f y ≥ 4 * Real.sqrt 2) ∧ (∃ y > 0, f y = 4 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3392_339273


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3392_339207

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Checks if a point (x, y) lies on the quadratic function -/
def QuadraticFunction.passesThrough (f : QuadraticFunction) (x y : ℝ) : Prop :=
  f.evaluate x = y

theorem quadratic_function_theorem (f : QuadraticFunction) :
  f.c = -3 →
  f.passesThrough 2 (-3) →
  f.passesThrough (-1) 0 →
  (f.a = 1 ∧ f.b = -2) ∧
  (∃ k : ℝ, k = 4 ∧
    (∀ x : ℝ, (f.evaluate x + k = 0) → (∀ y : ℝ, y ≠ x → f.evaluate y + k ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3392_339207


namespace NUMINAMATH_CALUDE_composite_number_probability_l3392_339297

/-- Represents a standard 6-sided die -/
def StandardDie : Type := Fin 6

/-- Represents the special die with only prime numbers -/
def SpecialDie : Type := Fin 3

/-- The total number of possible outcomes when rolling 6 dice -/
def TotalOutcomes : ℕ := 6^5 * 3

/-- The number of non-composite outcomes -/
def NonCompositeOutcomes : ℕ := 4

/-- The probability of getting a composite number -/
def CompositeNumberProbability : ℚ := 5831 / 5832

/-- Theorem stating the probability of getting a composite number when rolling 6 dice
    (5 standard 6-sided dice and 1 special die with prime numbers 2, 3, 5) and
    multiplying their face values -/
theorem composite_number_probability :
  (TotalOutcomes - NonCompositeOutcomes : ℚ) / TotalOutcomes = CompositeNumberProbability :=
sorry

end NUMINAMATH_CALUDE_composite_number_probability_l3392_339297


namespace NUMINAMATH_CALUDE_natural_number_representation_l3392_339255

theorem natural_number_representation (n : ℕ) : 
  (∃ a b c : ℕ, (a + b + c)^2 = n * a * b * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ↔ 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 8 ∨ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_natural_number_representation_l3392_339255


namespace NUMINAMATH_CALUDE_correct_num_classes_l3392_339244

/-- The number of classes in a single round-robin basketball tournament -/
def num_classes : ℕ := 10

/-- The total number of games played in the tournament -/
def total_games : ℕ := 45

/-- Theorem stating that the number of classes is correct given the total number of games played -/
theorem correct_num_classes : 
  (num_classes * (num_classes - 1)) / 2 = total_games :=
by sorry

end NUMINAMATH_CALUDE_correct_num_classes_l3392_339244
