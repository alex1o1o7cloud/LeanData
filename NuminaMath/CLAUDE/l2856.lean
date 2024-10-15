import Mathlib

namespace NUMINAMATH_CALUDE_percentage_of_democrat_voters_l2856_285638

theorem percentage_of_democrat_voters (d r : ℝ) : 
  d + r = 100 →
  0.65 * d + 0.2 * r = 47 →
  d = 60 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_democrat_voters_l2856_285638


namespace NUMINAMATH_CALUDE_melissa_initial_oranges_l2856_285661

/-- The number of oranges Melissa has initially -/
def initial_oranges : ℕ := sorry

/-- The number of oranges John takes away -/
def oranges_taken : ℕ := 19

/-- The number of oranges Melissa has left -/
def oranges_left : ℕ := 51

/-- Theorem stating that Melissa's initial number of oranges is 70 -/
theorem melissa_initial_oranges : 
  initial_oranges = oranges_taken + oranges_left :=
sorry

end NUMINAMATH_CALUDE_melissa_initial_oranges_l2856_285661


namespace NUMINAMATH_CALUDE_smallest_common_factor_l2856_285683

theorem smallest_common_factor (n : ℕ) : 
  (∀ k < 60, ¬ ∃ m > 1, m ∣ (11 * k - 6) ∧ m ∣ (8 * k + 5)) ∧
  (∃ m > 1, m ∣ (11 * 60 - 6) ∧ m ∣ (8 * 60 + 5)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l2856_285683


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2856_285636

def f (x : ℝ) : ℝ := x^5 - 6*x^4 + 11*x^3 + 21*x^2 - 17*x + 10

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 84 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2856_285636


namespace NUMINAMATH_CALUDE_route_ratio_is_three_l2856_285682

-- Define the grid structure
structure Grid where
  -- Add necessary fields to represent the grid

-- Define a function to count routes
def countRoutes (g : Grid) (start : Nat × Nat) (steps : Nat) : Nat :=
  sorry

-- Define points A and B
def pointA : Nat × Nat := sorry
def pointB : Nat × Nat := sorry

-- Define the specific grid
def specificGrid : Grid := sorry

-- Theorem statement
theorem route_ratio_is_three :
  let m := countRoutes specificGrid pointA 4
  let n := countRoutes specificGrid pointB 4
  n / m = 3 := by sorry

end NUMINAMATH_CALUDE_route_ratio_is_three_l2856_285682


namespace NUMINAMATH_CALUDE_f_properties_l2856_285608

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_properties :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -1 ∧ x₂ < -1 → f x₁ > f x₂) ∧
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (f (-1) = -(1 / Real.exp 1)) ∧
  (∀ x, f x ≥ -(1 / Real.exp 1)) ∧
  (∀ y : ℝ, ∃ x, f x > y) ∧
  (∃ a : ℝ, a ≥ -2 ∧
    ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ →
      (f x₂ - f a) / (x₂ - a) > (f x₁ - f a) / (x₁ - a)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2856_285608


namespace NUMINAMATH_CALUDE_polynomial_value_l2856_285692

theorem polynomial_value (a : ℝ) (h : a^3 - a = 4) : (-a)^3 - (-a) - 5 = -9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2856_285692


namespace NUMINAMATH_CALUDE_room_length_calculation_l2856_285644

/-- Given a room with width 4 meters and a paving cost of 750 per square meter,
    if the total cost of paving is 16500, then the length of the room is 5.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) :
  width = 4 →
  cost_per_sqm = 750 →
  total_cost = 16500 →
  length * width * cost_per_sqm = total_cost →
  length = 5.5 := by
  sorry

#check room_length_calculation

end NUMINAMATH_CALUDE_room_length_calculation_l2856_285644


namespace NUMINAMATH_CALUDE_min_value_theorem_l2856_285600

theorem min_value_theorem (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (z w : ℝ), z^2 + w^2 = 2 → |z| ≠ |w| →
    1 / (z + w)^2 + 1 / (z - w)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2856_285600


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2856_285640

def MonotonousFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ ∀ x y, x ≤ y → f x ≥ f y

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_mono : MonotonousFunction f)
  (h_eq : ∀ x, f (f x) = f (-f x) ∧ f (f x) = (f x)^2) :
  (∀ x, f x = 0) ∨ (∀ x, f x = 1) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2856_285640


namespace NUMINAMATH_CALUDE_bons_win_probability_l2856_285667

theorem bons_win_probability : 
  let p : ℝ := (1 : ℝ) / 6  -- Probability of rolling a six
  let q : ℝ := 1 - p        -- Probability of not rolling a six
  ∃ (win_prob : ℝ), 
    win_prob = q * p + q * q * win_prob ∧ 
    win_prob = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_bons_win_probability_l2856_285667


namespace NUMINAMATH_CALUDE_intersection_of_D_sets_nonempty_l2856_285613

def D (n : ℕ) : Set ℕ :=
  {x | ∃ a b : ℕ, a * b = n ∧ a > b ∧ b > 0 ∧ x = a - b}

theorem intersection_of_D_sets_nonempty (k : ℕ) (hk : k > 1) :
  ∃ (n : Fin k → ℕ), (∀ i, n i > 1) ∧ 
  (∀ i j, i ≠ j → n i ≠ n j) ∧
  (∃ x y : ℕ, x ≠ y ∧ ∀ i, x ∈ D (n i) ∧ y ∈ D (n i)) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_D_sets_nonempty_l2856_285613


namespace NUMINAMATH_CALUDE_A_l2856_285670

def A' : ℕ → ℕ → ℕ → ℕ
  | 0, n, k => n + k
  | m+1, 0, k => A' m k 1
  | m+1, n+1, k => A' m (A' (m+1) n k) k

theorem A'_3_2_2 : A' 3 2 2 = 17 := by sorry

end NUMINAMATH_CALUDE_A_l2856_285670


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2856_285694

theorem complex_magnitude_squared (w : ℂ) (h : w^2 = -48 + 14*I) : 
  Complex.abs w = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2856_285694


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2856_285630

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, where a = (1, 2) and b = (2x, -3),
    if a is parallel to b, then x = 3 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2 * x, -3)
  parallel a b → x = 3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2856_285630


namespace NUMINAMATH_CALUDE_soccer_goals_proof_l2856_285653

def goals_first_6 : List Nat := [5, 2, 4, 3, 6, 2]

def total_goals_6 : Nat := goals_first_6.sum

theorem soccer_goals_proof (goals_7 goals_8 : Nat) : 
  goals_7 < 7 →
  goals_8 < 7 →
  (total_goals_6 + goals_7) % 7 = 0 →
  (total_goals_6 + goals_7 + goals_8) % 8 = 0 →
  goals_7 * goals_8 = 24 := by
  sorry

#eval total_goals_6

end NUMINAMATH_CALUDE_soccer_goals_proof_l2856_285653


namespace NUMINAMATH_CALUDE_total_students_is_600_l2856_285672

/-- Represents a school with boys and girls -/
structure School where
  numBoys : ℕ
  numGirls : ℕ
  avgAgeBoys : ℝ
  avgAgeGirls : ℝ
  avgAgeSchool : ℝ

/-- The conditions of the problem -/
def problemSchool : School :=
  { numBoys := 0,  -- We don't know this yet, so we set it to 0
    numGirls := 150,
    avgAgeBoys := 12,
    avgAgeGirls := 11,
    avgAgeSchool := 11.75 }

/-- The theorem stating that the total number of students is 600 -/
theorem total_students_is_600 (s : School) 
  (h1 : s.numGirls = problemSchool.numGirls)
  (h2 : s.avgAgeBoys = problemSchool.avgAgeBoys)
  (h3 : s.avgAgeGirls = problemSchool.avgAgeGirls)
  (h4 : s.avgAgeSchool = problemSchool.avgAgeSchool)
  (h5 : s.avgAgeSchool * (s.numBoys + s.numGirls) = 
        s.avgAgeBoys * s.numBoys + s.avgAgeGirls * s.numGirls) :
  s.numBoys + s.numGirls = 600 := by
  sorry

#check total_students_is_600

end NUMINAMATH_CALUDE_total_students_is_600_l2856_285672


namespace NUMINAMATH_CALUDE_birds_left_l2856_285679

theorem birds_left (initial_chickens ducks turkeys chickens_sold : ℕ) :
  initial_chickens ≥ chickens_sold →
  (initial_chickens - chickens_sold + ducks + turkeys : ℕ) =
    initial_chickens + ducks + turkeys - chickens_sold :=
by sorry

end NUMINAMATH_CALUDE_birds_left_l2856_285679


namespace NUMINAMATH_CALUDE_tiles_along_width_l2856_285664

theorem tiles_along_width (area : ℝ) (tile_size : ℝ) : 
  area = 360 → tile_size = 9 → (8 : ℝ) * Real.sqrt 5 = (12 * Real.sqrt (area / 2)) / tile_size := by
  sorry

end NUMINAMATH_CALUDE_tiles_along_width_l2856_285664


namespace NUMINAMATH_CALUDE_car_speed_problem_l2856_285693

/-- Given two cars leaving town A at the same time in the same direction,
    prove that if one car travels at 55 mph and they are 45 miles apart after 3 hours,
    then the speed of the other car must be 70 mph. -/
theorem car_speed_problem (v : ℝ) : 
  v * 3 - 55 * 3 = 45 → v = 70 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2856_285693


namespace NUMINAMATH_CALUDE_omega_value_l2856_285658

theorem omega_value (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x + π / 6)) →
  ω > 0 →
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 3 → f x < f y) →
  f (π / 4) = f (π / 2) →
  ω = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_omega_value_l2856_285658


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2856_285699

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a - 2 * i) * i = b - i →
  a^2 + b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2856_285699


namespace NUMINAMATH_CALUDE_weight_of_A_l2856_285609

-- Define the weights and ages of persons A, B, C, D, and E
variable (W_A W_B W_C W_D W_E : ℝ)
variable (Age_A Age_B Age_C Age_D Age_E : ℝ)

-- State the conditions from the problem
axiom avg_weight_ABC : (W_A + W_B + W_C) / 3 = 84
axiom avg_age_ABC : (Age_A + Age_B + Age_C) / 3 = 30
axiom avg_weight_ABCD : (W_A + W_B + W_C + W_D) / 4 = 80
axiom avg_age_ABCD : (Age_A + Age_B + Age_C + Age_D) / 4 = 28
axiom avg_weight_BCDE : (W_B + W_C + W_D + W_E) / 4 = 79
axiom avg_age_BCDE : (Age_B + Age_C + Age_D + Age_E) / 4 = 27
axiom weight_E : W_E = W_D + 7
axiom age_E : Age_E = Age_A - 3

-- State the theorem to be proved
theorem weight_of_A : W_A = 79 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_A_l2856_285609


namespace NUMINAMATH_CALUDE_male_rabbits_count_l2856_285631

theorem male_rabbits_count (white : ℕ) (black : ℕ) (female : ℕ) 
  (h1 : white = 12) 
  (h2 : black = 9) 
  (h3 : female = 8) : 
  white + black - female = 13 := by
  sorry

end NUMINAMATH_CALUDE_male_rabbits_count_l2856_285631


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2856_285622

/-- A line in the form y = k(x-1) + 2 always passes through the point (1, 2) -/
theorem line_passes_through_point (k : ℝ) : 
  let f : ℝ → ℝ := λ x => k * (x - 1) + 2
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2856_285622


namespace NUMINAMATH_CALUDE_fibSeriesSum_l2856_285628

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of the infinite series of Fibonacci numbers divided by powers of 5 -/
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- The sum of the infinite series of Fibonacci numbers divided by powers of 5 is 5/19 -/
theorem fibSeriesSum : fibSeries = 5 / 19 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l2856_285628


namespace NUMINAMATH_CALUDE_perimeter_is_96_l2856_285641

/-- A figure composed of perpendicular line segments -/
structure PerpendicularFigure where
  x : ℝ
  y : ℝ
  area : ℝ
  x_eq_2y : x = 2 * y
  area_eq_252 : area = 252

/-- The perimeter of the perpendicular figure -/
def perimeter (f : PerpendicularFigure) : ℝ :=
  16 * f.y

theorem perimeter_is_96 (f : PerpendicularFigure) : perimeter f = 96 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_96_l2856_285641


namespace NUMINAMATH_CALUDE_cos_neg_pi_third_l2856_285666

theorem cos_neg_pi_third : Real.cos (-π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_pi_third_l2856_285666


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2856_285646

theorem six_digit_divisibility (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  ∃ (k : ℕ), 100100 * a + 10010 * b + 1001 * c = 7 * 11 * 13 * k := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l2856_285646


namespace NUMINAMATH_CALUDE_two_people_walking_problem_l2856_285655

/-- Two people walking problem -/
theorem two_people_walking_problem (x y : ℝ) : 
  (∃ (distance : ℝ), distance = 18) →
  (∃ (time_meeting : ℝ), time_meeting = 2) →
  (∃ (time_catchup : ℝ), time_catchup = 4) →
  (∃ (time_headstart : ℝ), time_headstart = 1) →
  (2 * x + 2 * y = 18 ∧ 5 * x - 4 * y = 18) := by
sorry

end NUMINAMATH_CALUDE_two_people_walking_problem_l2856_285655


namespace NUMINAMATH_CALUDE_ellipse_proof_hyperbola_proof_l2856_285606

-- Ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 20 = 1

theorem ellipse_proof (major_axis_length : ℝ) (eccentricity : ℝ) 
  (h1 : major_axis_length = 12) 
  (h2 : eccentricity = 2/3) : 
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    ∃ a b : ℝ, a^2 * y^2 + b^2 * x^2 = a^2 * b^2 ∧ 
    2 * a = major_axis_length ∧ 
    (a^2 - b^2) / a^2 = eccentricity^2 :=
sorry

-- Hyperbola
def original_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def new_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 24 = 1

theorem hyperbola_proof :
  ∀ x y : ℝ, new_hyperbola x y ↔ 
    (∃ c : ℝ, (∀ x₀ y₀ : ℝ, original_hyperbola x₀ y₀ → 
      (x₀ - c)^2 - y₀^2 = c^2 ∧ (x₀ + c)^2 - y₀^2 = c^2) ∧
    new_hyperbola (-Real.sqrt 5 / 2) (-Real.sqrt 6)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_proof_hyperbola_proof_l2856_285606


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2856_285675

theorem cubic_expression_evaluation : 101^3 + 3*(101^2) - 3*101 + 9 = 1060610 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2856_285675


namespace NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l2856_285656

def total_cookies : ℝ := 41.0
def chocolate_chip_cookies : ℝ := 13.0
def cookies_per_bag : ℝ := 9.0

theorem oatmeal_cookie_baggies :
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ = 3 :=
by sorry

end NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l2856_285656


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l2856_285633

theorem largest_four_digit_divisible_by_12 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l2856_285633


namespace NUMINAMATH_CALUDE_multiplicative_inverse_484_mod_1123_l2856_285654

theorem multiplicative_inverse_484_mod_1123 :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 1123 ∧ (484 * n) % 1123 = 1 :=
by
  use 535
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_484_mod_1123_l2856_285654


namespace NUMINAMATH_CALUDE_equation_solution_l2856_285632

open Real

theorem equation_solution (x : ℝ) :
  (sin x ≠ 0) →
  (cos x ≠ 0) →
  (sin x + cos x ≥ 0) →
  (Real.sqrt (1 + tan x) = sin x + cos x) ↔
  (∃ n : ℤ, (x = π/4 + 2*π*↑n) ∨ (x = -π/4 + 2*π*↑n) ∨ (x = 3*π/4 + 2*π*↑n)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2856_285632


namespace NUMINAMATH_CALUDE_matchstick_20th_stage_l2856_285629

/-- Arithmetic sequence with first term 3 and common difference 3 -/
def matchstick_sequence (n : ℕ) : ℕ := 3 + (n - 1) * 3

/-- The 20th term of the matchstick sequence is 60 -/
theorem matchstick_20th_stage : matchstick_sequence 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_20th_stage_l2856_285629


namespace NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l2856_285697

theorem one_fourth_of_eight_point_eight : (8.8 : ℚ) / 4 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l2856_285697


namespace NUMINAMATH_CALUDE_point_below_line_range_l2856_285645

/-- Given a point (-2,t) located below the line 2x-3y+6=0, prove that the range of t is (-∞, 2/3) -/
theorem point_below_line_range (t : ℝ) : 
  (2 * (-2) - 3 * t + 6 > 0) → (t < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_point_below_line_range_l2856_285645


namespace NUMINAMATH_CALUDE_two_greater_than_sqrt_three_l2856_285685

theorem two_greater_than_sqrt_three : 2 > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_greater_than_sqrt_three_l2856_285685


namespace NUMINAMATH_CALUDE_fraction_equality_l2856_285604

theorem fraction_equality : (1998 - 998) / 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2856_285604


namespace NUMINAMATH_CALUDE_ellipse_iff_m_range_l2856_285690

-- Define the equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (2 + m) - y^2 / (m + 1) = 1

-- Define the condition for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    ellipse_equation x y m ↔ x^2 / a^2 + y^2 / b^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

-- State the theorem
theorem ellipse_iff_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_iff_m_range_l2856_285690


namespace NUMINAMATH_CALUDE_sues_shoe_probability_l2856_285681

/-- Represents the number of pairs of shoes for each color --/
structure ShoePairs where
  black : ℕ
  brown : ℕ
  gray : ℕ

/-- Calculates the probability of selecting two shoes of the same color,
    one left and one right, given the number of pairs for each color --/
def samePairColorProbability (pairs : ShoePairs) : ℚ :=
  let totalShoes := 2 * (pairs.black + pairs.brown + pairs.gray)
  let blackProb := (2 * pairs.black) * pairs.black / (totalShoes * (totalShoes - 1))
  let brownProb := (2 * pairs.brown) * pairs.brown / (totalShoes * (totalShoes - 1))
  let grayProb := (2 * pairs.gray) * pairs.gray / (totalShoes * (totalShoes - 1))
  blackProb + brownProb + grayProb

/-- Theorem stating that for Sue's shoe collection, the probability of
    selecting two shoes of the same color, one left and one right, is 7/33 --/
theorem sues_shoe_probability :
  samePairColorProbability ⟨6, 3, 2⟩ = 7 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sues_shoe_probability_l2856_285681


namespace NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l2856_285621

/-- Represents a type of algorithm statement -/
inductive AlgorithmStatement
  | INPUT
  | PRINT
  | IF_THEN
  | DO
  | END
  | WHILE
  | END_IF

/-- Defines the set of basic algorithm statements -/
def BasicAlgorithmStatements : Set AlgorithmStatement :=
  {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN,
   AlgorithmStatement.DO, AlgorithmStatement.WHILE}

/-- Theorem stating that the set of basic algorithm statements is correct -/
theorem basic_algorithm_statements_correct :
  BasicAlgorithmStatements = {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT,
    AlgorithmStatement.IF_THEN, AlgorithmStatement.DO, AlgorithmStatement.WHILE} := by
  sorry

end NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l2856_285621


namespace NUMINAMATH_CALUDE_integral_x_plus_exp_x_l2856_285611

theorem integral_x_plus_exp_x : ∫ x in (0:ℝ)..2, (x + Real.exp x) = Real.exp 2 + 1 := by sorry

end NUMINAMATH_CALUDE_integral_x_plus_exp_x_l2856_285611


namespace NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2856_285680

theorem yellow_jelly_bean_probability 
  (red_prob : ℝ) 
  (orange_prob : ℝ) 
  (blue_prob : ℝ) 
  (yellow_prob : ℝ)
  (h1 : red_prob = 0.1)
  (h2 : orange_prob = 0.4)
  (h3 : blue_prob = 0.2)
  (h4 : red_prob + orange_prob + blue_prob + yellow_prob = 1) :
  yellow_prob = 0.3 := by
sorry

end NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2856_285680


namespace NUMINAMATH_CALUDE_sara_cannot_have_two_l2856_285668

-- Define the set of cards
def Cards : Finset ℕ := {1, 2, 3, 4}

-- Define the players
inductive Player
| Ben
| Wendy
| Riley
| Sara

-- Define the distribution of cards
def Distribution := Player → ℕ

-- Define the conditions
def ValidDistribution (d : Distribution) : Prop :=
  (∀ p : Player, d p ∈ Cards) ∧
  (∀ p q : Player, p ≠ q → d p ≠ d q) ∧
  (d Player.Ben ≠ 1) ∧
  (d Player.Wendy = d Player.Riley + 1)

-- Theorem statement
theorem sara_cannot_have_two (d : Distribution) :
  ValidDistribution d → d Player.Sara ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_sara_cannot_have_two_l2856_285668


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l2856_285639

theorem ice_cream_sundaes (total_flavors : ℕ) (h : total_flavors = 8) :
  let vanilla_sundaes := total_flavors - 1
  vanilla_sundaes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l2856_285639


namespace NUMINAMATH_CALUDE_marble_box_count_l2856_285626

theorem marble_box_count (blue : ℕ) (red : ℕ) : 
  red = blue + 12 →
  (blue : ℚ) / (blue + red : ℚ) = 1 / 4 →
  blue + red = 24 :=
by sorry

end NUMINAMATH_CALUDE_marble_box_count_l2856_285626


namespace NUMINAMATH_CALUDE_min_operations_for_jugs_l2856_285623

/-- Represents the state of the two jugs -/
structure JugState :=
  (jug7 : ℕ)
  (jug5 : ℕ)

/-- Represents an operation on the jugs -/
inductive Operation
  | Fill7
  | Fill5
  | Empty7
  | Empty5
  | Pour7to5
  | Pour5to7

/-- Applies an operation to a JugState -/
def applyOperation (state : JugState) (op : Operation) : JugState :=
  match op with
  | Operation.Fill7 => ⟨7, state.jug5⟩
  | Operation.Fill5 => ⟨state.jug7, 5⟩
  | Operation.Empty7 => ⟨0, state.jug5⟩
  | Operation.Empty5 => ⟨state.jug7, 0⟩
  | Operation.Pour7to5 => 
      let amount := min state.jug7 (5 - state.jug5)
      ⟨state.jug7 - amount, state.jug5 + amount⟩
  | Operation.Pour5to7 => 
      let amount := min state.jug5 (7 - state.jug7)
      ⟨state.jug7 + amount, state.jug5 - amount⟩

/-- Checks if a sequence of operations results in the desired state -/
def isValidSolution (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.jug7 = 1 ∧ finalState.jug5 = 1

/-- The main theorem to be proved -/
theorem min_operations_for_jugs : 
  ∃ (ops : List Operation), isValidSolution ops ∧ ops.length = 42 ∧
  (∀ (other_ops : List Operation), isValidSolution other_ops → other_ops.length ≥ 42) :=
sorry

end NUMINAMATH_CALUDE_min_operations_for_jugs_l2856_285623


namespace NUMINAMATH_CALUDE_x_axis_intercept_l2856_285691

/-- The x-axis intercept of the line y = 2x + 1 is -1/2 -/
theorem x_axis_intercept (x : ℝ) : 
  (2 * x + 1 = 0) → (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_x_axis_intercept_l2856_285691


namespace NUMINAMATH_CALUDE_anna_transportation_tax_l2856_285689

/-- Calculates the transportation tax for a vehicle -/
def calculate_tax (engine_power : ℕ) (tax_rate : ℕ) (months_owned : ℕ) (months_in_year : ℕ) : ℕ :=
  (engine_power * tax_rate * months_owned) / months_in_year

/-- Represents the transportation tax problem for Anna Ivanovna -/
theorem anna_transportation_tax :
  let engine_power : ℕ := 250
  let tax_rate : ℕ := 75
  let months_owned : ℕ := 2
  let months_in_year : ℕ := 12
  calculate_tax engine_power tax_rate months_owned months_in_year = 3125 := by
  sorry


end NUMINAMATH_CALUDE_anna_transportation_tax_l2856_285689


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2856_285648

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | |x| > 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2856_285648


namespace NUMINAMATH_CALUDE_vegetables_in_soup_serving_l2856_285642

/-- Proves that the number of cups of vegetables in one serving of soup is 1 -/
theorem vegetables_in_soup_serving (V : ℝ) : V = 1 :=
  by
  -- One serving contains V cups of vegetables and 2.5 cups of broth
  have h1 : V + 2.5 = (14 * 2) / 8 := by sorry
  -- 8 servings require 14 pints of vegetables and broth combined
  -- 1 pint = 2 cups
  -- So, 14 pints = 14 * 2 cups = 28 cups
  -- Solve the equation: 8 * (V + 2.5) = 28
  sorry

end NUMINAMATH_CALUDE_vegetables_in_soup_serving_l2856_285642


namespace NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_self_l2856_285607

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem negation_of_forall_even_square_plus_self :
  (¬ ∀ n : ℕ, is_even (n^2 + n)) ↔ (∃ x : ℕ, ¬ is_even (x^2 + x)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_self_l2856_285607


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l2856_285620

/-- Given a cube with edge length 5 and a square-based pyramid with base edge length 10,
    prove that the height of the pyramid is 3.75 when their volumes are equal. -/
theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  (cube_edge ^ 3) = (1 / 3) * (pyramid_base ^ 2) * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l2856_285620


namespace NUMINAMATH_CALUDE_equation_solutions_l2856_285688

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 1)^2 = 100 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, (2*x - 1)^3 = -8 ↔ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2856_285688


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_25_l2856_285643

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def smallestPrimeBetween1And25 : ℕ := 2

def largestPrimeBetween1And25 : ℕ := 23

theorem sum_smallest_largest_prime_1_to_25 :
  isPrime smallestPrimeBetween1And25 ∧
  isPrime largestPrimeBetween1And25 ∧
  (∀ n : ℕ, 1 < n → n < 25 → isPrime n → smallestPrimeBetween1And25 ≤ n) ∧
  (∀ n : ℕ, 1 < n → n < 25 → isPrime n → n ≤ largestPrimeBetween1And25) →
  smallestPrimeBetween1And25 + largestPrimeBetween1And25 = 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_25_l2856_285643


namespace NUMINAMATH_CALUDE_combined_cost_increase_percentage_l2856_285686

def bicycle_initial_cost : ℝ := 200
def skates_initial_cost : ℝ := 50
def bicycle_increase_rate : ℝ := 0.06
def skates_increase_rate : ℝ := 0.15

theorem combined_cost_increase_percentage :
  let bicycle_new_cost := bicycle_initial_cost * (1 + bicycle_increase_rate)
  let skates_new_cost := skates_initial_cost * (1 + skates_increase_rate)
  let initial_total_cost := bicycle_initial_cost + skates_initial_cost
  let new_total_cost := bicycle_new_cost + skates_new_cost
  (new_total_cost - initial_total_cost) / initial_total_cost = 0.078 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_increase_percentage_l2856_285686


namespace NUMINAMATH_CALUDE_symmetry_axis_property_l2856_285601

/-- Given a function f(x) = 3sin(x) + 4cos(x), if x = θ is an axis of symmetry
    for the curve y = f(x), then cos(2θ) + sin(θ)cos(θ) = 19/25 -/
theorem symmetry_axis_property (θ : ℝ) :
  (∀ x, 3 * Real.sin x + 4 * Real.cos x = 3 * Real.sin (2 * θ - x) + 4 * Real.cos (2 * θ - x)) →
  Real.cos (2 * θ) + Real.sin θ * Real.cos θ = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_property_l2856_285601


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2856_285676

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2856_285676


namespace NUMINAMATH_CALUDE_parabola_shift_l2856_285678

def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 2)^2 + 7

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 2) + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l2856_285678


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2856_285665

theorem trigonometric_equation_solution (x : ℝ) : 
  (abs (Real.sin x) + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2856_285665


namespace NUMINAMATH_CALUDE_baseball_gear_expense_l2856_285605

theorem baseball_gear_expense (initial_amount : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : remaining_amount = 32) :
  initial_amount - remaining_amount = 47 := by
  sorry

end NUMINAMATH_CALUDE_baseball_gear_expense_l2856_285605


namespace NUMINAMATH_CALUDE_arrangements_with_A_must_go_arrangements_A_B_not_Japan_l2856_285696

-- Define the number of volunteers
def total_volunteers : ℕ := 6

-- Define the number of people to be selected
def selected_people : ℕ := 4

-- Define the number of pavilions
def num_pavilions : ℕ := 4

-- Function to calculate the number of arrangements when one person must be included
def arrangements_with_one_person (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose (n - 1) (k - 1)) * (Nat.factorial k)

-- Function to calculate the number of arrangements when two people cannot go to a specific pavilion
def arrangements_with_restriction (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose k 1) * (Nat.choose (n - 1) (k - 1)) * (Nat.factorial (k - 1))

-- Theorem for the first question
theorem arrangements_with_A_must_go :
  arrangements_with_one_person total_volunteers selected_people = 240 := by
  sorry

-- Theorem for the second question
theorem arrangements_A_B_not_Japan :
  arrangements_with_restriction total_volunteers selected_people = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_A_must_go_arrangements_A_B_not_Japan_l2856_285696


namespace NUMINAMATH_CALUDE_expansion_contains_2017_l2856_285635

/-- The first term in the expansion of n^3 -/
def first_term (n : ℕ) : ℕ := n^2 - n + 1

/-- The last term in the expansion of n^3 -/
def last_term (n : ℕ) : ℕ := n^2 + n - 1

/-- The sum of n consecutive odd numbers starting from the first term -/
def sum_expansion (n : ℕ) : ℕ := n * (first_term n + last_term n) / 2

theorem expansion_contains_2017 :
  ∃ (n : ℕ), n = 45 ∧ 
  first_term n ≤ 2017 ∧ 
  2017 ≤ last_term n ∧ 
  sum_expansion n = n^3 :=
sorry

end NUMINAMATH_CALUDE_expansion_contains_2017_l2856_285635


namespace NUMINAMATH_CALUDE_car_rental_rates_equal_l2856_285663

/-- The daily rate of Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate of Sunshine Car Rentals -/
def sunshine_mile_rate : ℝ := 0.18

/-- The per-mile rate of the second car rental company -/
def second_company_mile_rate : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48

/-- The daily rate of the second car rental company -/
def second_company_daily_rate : ℝ := 18.95

theorem car_rental_rates_equal :
  sunshine_daily_rate + sunshine_mile_rate * miles_driven =
  second_company_daily_rate + second_company_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rates_equal_l2856_285663


namespace NUMINAMATH_CALUDE_distinct_colorings_tetrahedron_l2856_285677

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the symmetry group size of a tetrahedron
def symmetry_group_size : ℕ := 12

-- Define the number of vertices in a tetrahedron
def num_vertices : ℕ := 4

-- Define the number of colorings fixed by identity rotation
def fixed_by_identity : ℕ := num_colors ^ num_vertices

-- Define the number of colorings fixed by 180° rotations
def fixed_by_180_rotation : ℕ := num_colors ^ 2

-- Define the number of 180° rotations
def num_180_rotations : ℕ := 3

-- Theorem statement
theorem distinct_colorings_tetrahedron :
  (fixed_by_identity + num_180_rotations * fixed_by_180_rotation) / symmetry_group_size = 36 :=
by sorry

end NUMINAMATH_CALUDE_distinct_colorings_tetrahedron_l2856_285677


namespace NUMINAMATH_CALUDE_diver_min_trips_l2856_285659

/-- Calculates the minimum number of trips required to transport objects --/
def min_trips (objects_per_trip : ℕ) (total_objects : ℕ) : ℕ :=
  (total_objects + objects_per_trip - 1) / objects_per_trip

/-- Theorem: Given a diver who can carry 3 objects at a time and has found 17 objects,
    the minimum number of trips required to transport all objects is 6 --/
theorem diver_min_trips :
  min_trips 3 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_diver_min_trips_l2856_285659


namespace NUMINAMATH_CALUDE_jeremy_songs_l2856_285637

theorem jeremy_songs (x : ℕ) (h1 : x + (x + 5) = 23) : x + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_songs_l2856_285637


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2856_285674

theorem inequality_solution_set (x : ℝ) : 
  (abs (2*x - 1) + abs (2*x + 3) < 5) ↔ (-3/2 ≤ x ∧ x < 3/4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2856_285674


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l2856_285647

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (n + 1) > (f n + f (f n)) / 2

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f →
  ∃ b : ℕ, (∀ n < b, f n = n) ∧ (∀ n ≥ b, f n = n + 1) :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_functions_l2856_285647


namespace NUMINAMATH_CALUDE_club_size_l2856_285603

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 3

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := sock_cost + 7

/-- The cost of a warm-up jacket in dollars -/
def jacket_cost : ℕ := 2 * jersey_cost

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (sock_cost + jersey_cost) + jacket_cost

/-- The total expenditure for the club in dollars -/
def total_expenditure : ℕ := 3276

/-- The number of players in the club -/
def num_players : ℕ := total_expenditure / player_cost

theorem club_size :
  num_players = 71 :=
sorry

end NUMINAMATH_CALUDE_club_size_l2856_285603


namespace NUMINAMATH_CALUDE_certain_number_proof_l2856_285657

theorem certain_number_proof : ∃ x : ℝ, x * 2 + (12 + 4) * (1 / 8) = 602 ∧ x = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2856_285657


namespace NUMINAMATH_CALUDE_food_company_inspection_l2856_285612

theorem food_company_inspection (large_companies medium_companies total_inspected medium_inspected : ℕ) 
  (h1 : large_companies = 4)
  (h2 : medium_companies = 20)
  (h3 : total_inspected = 40)
  (h4 : medium_inspected = 5) :
  ∃ (small_companies : ℕ), 
    small_companies = 136 ∧ 
    total_inspected = large_companies + medium_inspected + (total_inspected - large_companies - medium_inspected) :=
by sorry

end NUMINAMATH_CALUDE_food_company_inspection_l2856_285612


namespace NUMINAMATH_CALUDE_negation_p_iff_valid_range_l2856_285614

/-- The proposition p: There exists x₀ ∈ ℝ such that x₀² + ax₀ + a < 0 -/
def p (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + a*x₀ + a < 0

/-- The range of a for which ¬p holds -/
def valid_range (a : ℝ) : Prop := a ≤ 0 ∨ a ≥ 4

theorem negation_p_iff_valid_range (a : ℝ) :
  ¬(p a) ↔ valid_range a := by sorry

end NUMINAMATH_CALUDE_negation_p_iff_valid_range_l2856_285614


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2856_285610

/-- The eccentricity of an ellipse with equation x²/3 + y²/9 = 1 is √6/3 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt (a^2 - b^2) / a
  e = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2856_285610


namespace NUMINAMATH_CALUDE_silk_order_total_l2856_285671

/-- Calculates the total yards of silk dyed given the yards of each color and the percentage of red silk -/
def total_silk_dyed (green pink blue yellow : ℝ) (red_percent : ℝ) : ℝ :=
  let non_red := green + pink + blue + yellow
  let red := red_percent * non_red
  non_red + red

/-- Theorem stating the total yards of silk dyed for the given order -/
theorem silk_order_total :
  total_silk_dyed 61921 49500 75678 34874.5 0.1 = 245270.85 := by
  sorry

end NUMINAMATH_CALUDE_silk_order_total_l2856_285671


namespace NUMINAMATH_CALUDE_decimal_to_binary_and_remainder_l2856_285634

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def binary_to_decimal (b : List Bool) : ℕ :=
  sorry

def binary_division_remainder (dividend : List Bool) (divisor : List Bool) : List Bool :=
  sorry

theorem decimal_to_binary_and_remainder : 
  let binary_126 := decimal_to_binary 126
  let remainder := binary_division_remainder binary_126 [true, false, true, true]
  binary_126 = [true, true, true, true, true, true, false] ∧ 
  remainder = [true, false, false, true] :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_and_remainder_l2856_285634


namespace NUMINAMATH_CALUDE_unique_solution_l2856_285687

/-- A discrete random variable with three possible values -/
structure DiscreteRV where
  p₁ : ℝ
  p₂ : ℝ
  p₃ : ℝ
  sum_to_one : p₁ + p₂ + p₃ = 1
  nonnegative : 0 ≤ p₁ ∧ 0 ≤ p₂ ∧ 0 ≤ p₃

/-- The expected value of X -/
def expectation (X : DiscreteRV) : ℝ := -X.p₁ + X.p₃

/-- The expected value of X² -/
def expectation_squared (X : DiscreteRV) : ℝ := X.p₁ + X.p₃

/-- Theorem stating the unique solution for the given conditions -/
theorem unique_solution (X : DiscreteRV) 
  (h₁ : expectation X = 0.1) 
  (h₂ : expectation_squared X = 0.9) : 
  X.p₁ = 0.4 ∧ X.p₂ = 0.1 ∧ X.p₃ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2856_285687


namespace NUMINAMATH_CALUDE_palindrome_count_ratio_l2856_285625

/-- A palindrome is a natural number whose decimal representation reads the same from left to right and right to left. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Count of palindromes with even sum of digits between 10,000 and 999,999. -/
def evenSumPalindromeCount : ℕ := sorry

/-- Count of palindromes with odd sum of digits between 10,000 and 999,999. -/
def oddSumPalindromeCount : ℕ := sorry

theorem palindrome_count_ratio :
  evenSumPalindromeCount = 3 * oddSumPalindromeCount := by sorry

end NUMINAMATH_CALUDE_palindrome_count_ratio_l2856_285625


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l2856_285698

theorem books_per_bookshelf (total_books : ℕ) (num_bookshelves : ℕ) 
  (h1 : total_books = 621) 
  (h2 : num_bookshelves = 23) :
  total_books / num_bookshelves = 27 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l2856_285698


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l2856_285684

theorem stamp_collection_problem (current_stamps : ℕ) : 
  (current_stamps : ℚ) * (1 + 20 / 100) = 48 → current_stamps = 40 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l2856_285684


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_implies_m_bound_l2856_285619

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∨ (∀ x y : ℝ, x ≤ y → f y ≤ f x)

/-- The main theorem: If f(x) = x³ + x² + mx + 1 is monotonic on ℝ, then m ≥ 1/3. -/
theorem monotonic_cubic_function_implies_m_bound (m : ℝ) :
  Monotonic (fun x : ℝ => x^3 + x^2 + m*x + 1) → m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_implies_m_bound_l2856_285619


namespace NUMINAMATH_CALUDE_jerry_pool_time_l2856_285652

/-- Represents the time spent in the pool by each person --/
structure PoolTime where
  jerry : ℝ
  elaine : ℝ
  george : ℝ
  kramer : ℝ

/-- The conditions of the problem --/
def poolConditions (t : PoolTime) : Prop :=
  t.elaine = 2 * t.jerry ∧
  t.george = (1/3) * t.elaine ∧
  t.kramer = 0 ∧
  t.jerry + t.elaine + t.george + t.kramer = 11

/-- The theorem to be proved --/
theorem jerry_pool_time (t : PoolTime) :
  poolConditions t → t.jerry = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_pool_time_l2856_285652


namespace NUMINAMATH_CALUDE_seat_difference_l2856_285616

/-- Represents the seating configuration of a bus --/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- Theorem stating the difference in seats between left and right sides --/
theorem seat_difference (bus : BusSeating) : 
  bus.leftSeats = 15 →
  bus.backSeat = 12 →
  bus.seatCapacity = 3 →
  bus.totalCapacity = 93 →
  bus.leftSeats - bus.rightSeats = 3 := by
  sorry

#check seat_difference

end NUMINAMATH_CALUDE_seat_difference_l2856_285616


namespace NUMINAMATH_CALUDE_travel_ways_eq_nine_l2856_285624

/-- The number of different ways to travel from location A to location B in one day -/
def travel_ways (car_departures train_departures ship_departures : ℕ) : ℕ :=
  car_departures + train_departures + ship_departures

/-- Theorem: The number of different ways to travel is 9 given the specified departures -/
theorem travel_ways_eq_nine :
  travel_ways 3 4 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_eq_nine_l2856_285624


namespace NUMINAMATH_CALUDE_proposition_1_proposition_4_proposition_5_l2856_285662

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (point_not_on_line : Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Proposition 1
theorem proposition_1 (l m : Line) (α : Plane) :
  contains α m → contains α l → point_not_on_line m → skew l m :=
sorry

-- Proposition 4
theorem proposition_4 (l m : Line) (α : Plane) :
  line_perpendicular_plane m α → line_parallel_plane l α → perpendicular l m :=
sorry

-- Proposition 5
theorem proposition_5 (m n : Line) (α β : Plane) :
  skew m n → contains α m → line_parallel_plane m β → 
  contains β n → line_parallel_plane n α → parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_4_proposition_5_l2856_285662


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2856_285617

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and the length of the real axis is 4 and the length of the imaginary axis is 6,
    prove that the equation of its asymptotes is y = ±(3/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (real_axis : 2 * a = 4) (imag_axis : 2 * b = 6) :
  ∃ (k : ℝ), k = 3/2 ∧ (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2856_285617


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2856_285618

theorem line_tangent_to_circle (a : ℝ) (h1 : a > 0) :
  (∀ y : ℝ, (a - 1)^2 + y^2 = 4) →
  (∀ x y : ℝ, x = a → (x - 1)^2 + y^2 ≥ 4) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2856_285618


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2856_285651

/-- Proves that adding 6 liters of 75% alcohol solution to 6 liters of 25% alcohol solution results in a 50% alcohol solution -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.25
  let added_volume : ℝ := 6
  let added_concentration : ℝ := 0.75
  let target_concentration : ℝ := 0.50
  let final_volume : ℝ := initial_volume + added_volume
  let final_alcohol_amount : ℝ := initial_volume * initial_concentration + added_volume * added_concentration
  final_alcohol_amount / final_volume = target_concentration := by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2856_285651


namespace NUMINAMATH_CALUDE_A_div_B_equals_37_l2856_285650

-- Define the series A
def A : ℝ := sorry

-- Define the series B
def B : ℝ := sorry

-- Theorem statement
theorem A_div_B_equals_37 : A / B = 37 := by sorry

end NUMINAMATH_CALUDE_A_div_B_equals_37_l2856_285650


namespace NUMINAMATH_CALUDE_cut_cube_height_l2856_285649

/-- The height of a cube with a corner cut off -/
theorem cut_cube_height : 
  let s : ℝ := 2  -- side length of the original cube
  let triangle_side : ℝ := s * Real.sqrt 2  -- side length of the cut triangle
  let base_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2  -- area of the cut face
  let pyramid_volume : ℝ := s ^ 3 / 6  -- volume of the cut-off pyramid
  let h : ℝ := pyramid_volume / (base_area / 6)  -- height of the cut-off pyramid
  2 - h = 2 - (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_cut_cube_height_l2856_285649


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2856_285669

theorem unique_prime_solution :
  ∀ p m : ℕ,
    p.Prime →
    m > 0 →
    p * (p + m) + p = (m + 1)^3 →
    p = 2 ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2856_285669


namespace NUMINAMATH_CALUDE_division_problem_l2856_285602

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 1375) (h2 : quotient = 20) (h3 : remainder = 55) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 66 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2856_285602


namespace NUMINAMATH_CALUDE_approximate_0_9915_l2856_285660

theorem approximate_0_9915 : 
  ∃ (x : ℚ), (x = 0.956) ∧ 
  (∀ (y : ℚ), abs (y - 0.9915) < abs (x - 0.9915) → abs (y - 0.9915) ≥ 0.0005) :=
by sorry

end NUMINAMATH_CALUDE_approximate_0_9915_l2856_285660


namespace NUMINAMATH_CALUDE_arrangement_count_l2856_285695

def num_people : ℕ := 8

theorem arrangement_count :
  (num_people.factorial) / 6 * 2 = 13440 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l2856_285695


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2856_285615

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Theorem: Given the conditions, prove that the radius of circle k is 17 -/
theorem circle_radius_proof (k k1 k2 : Circle)
  (h1 : k1.radius = 8)
  (h2 : k2.radius = 15)
  (h3 : k1.radius < k.radius)
  (h4 : (k.radius ^ 2 - k1.radius ^ 2) * Real.pi = (k2.radius ^ 2 - k.radius ^ 2) * Real.pi) :
  k.radius = 17 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2856_285615


namespace NUMINAMATH_CALUDE_vector_problem_l2856_285673

theorem vector_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (3*π/2) (2*π))
  (h2 : (3*Real.sin α)*(2*Real.sin α) + (Real.cos α)*(5*Real.sin α - 4*Real.cos α) = 0) :
  Real.tan α = -4/3 ∧ Real.cos (α/2 + π/3) = -(2*Real.sqrt 5 + Real.sqrt 15)/10 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2856_285673


namespace NUMINAMATH_CALUDE_no_intersection_l2856_285627

theorem no_intersection : ¬∃ x : ℝ, |3 * x + 6| = -2 * |2 * x - 1| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l2856_285627
