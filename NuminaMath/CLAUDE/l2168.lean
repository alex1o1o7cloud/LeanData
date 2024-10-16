import Mathlib

namespace NUMINAMATH_CALUDE_buddy_gym_class_size_l2168_216893

/-- The number of students in Buddy's gym class -/
def total_students (group1 : ℕ) (group2 : ℕ) : ℕ := group1 + group2

/-- Theorem stating the total number of students in Buddy's gym class -/
theorem buddy_gym_class_size :
  total_students 34 37 = 71 := by
  sorry

end NUMINAMATH_CALUDE_buddy_gym_class_size_l2168_216893


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2168_216816

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 150) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 152 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2168_216816


namespace NUMINAMATH_CALUDE_fencing_rate_proof_l2168_216884

/-- Given a rectangular plot with the following properties:
  - The length is 10 meters more than the width
  - The perimeter is 300 meters
  - The total fencing cost is 1950 Rs
  Prove that the rate per meter for fencing is 6.5 Rs -/
theorem fencing_rate_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) (total_cost : ℝ) :
  length = width + 10 →
  perimeter = 300 →
  perimeter = 2 * (length + width) →
  total_cost = 1950 →
  total_cost / perimeter = 6.5 := by
sorry

end NUMINAMATH_CALUDE_fencing_rate_proof_l2168_216884


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l2168_216841

theorem inequality_solution_existence (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l2168_216841


namespace NUMINAMATH_CALUDE_value_of_b_l2168_216847

theorem value_of_b (b : ℚ) (h : b + b/4 = 3) : b = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2168_216847


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l2168_216887

theorem inequality_solution_implies_m_range (m : ℝ) : 
  (∀ x, (m - 1) * x > m - 1 ↔ x < 1) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l2168_216887


namespace NUMINAMATH_CALUDE_complement_intersection_equals_l2168_216853

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection_equals : (U \ (A ∩ B)) = {1, 2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_l2168_216853


namespace NUMINAMATH_CALUDE_part1_part2_l2168_216892

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection point of l₁ and l₂
def intersection : ℝ × ℝ := (-2, 2)

-- Define the line parallel to 3x + y - 1 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x + y - 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Part 1: Prove that if l passes through the intersection and is parallel to 3x + y - 1 = 0,
-- then its equation is 3x + y + 4 = 0
theorem part1 (l : ℝ → ℝ → Prop) :
  (∀ x y, l x y ↔ ∃ k, 3 * x + y + k = 0) →
  l (intersection.1) (intersection.2) →
  (∀ x y, l x y → parallel_line x y) →
  (∀ x y, l x y ↔ 3 * x + y + 4 = 0) :=
sorry

-- Part 2: Prove that if l passes through the intersection and the distance from A to l is 5,
-- then its equation is either x = -2 or 12x - 5y + 34 = 0
theorem part2 (l : ℝ → ℝ → Prop) :
  l (intersection.1) (intersection.2) →
  (∀ x y, l x y → (((x - point_A.1) ^ 2 + (y - point_A.2) ^ 2) : ℝ).sqrt = 5) →
  (∀ x y, l x y ↔ x = -2 ∨ 12 * x - 5 * y + 34 = 0) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2168_216892


namespace NUMINAMATH_CALUDE_eleven_girls_l2168_216862

/-- Represents a circular arrangement of girls -/
structure CircularArrangement where
  girls : ℕ  -- Total number of girls in the circle

/-- Defines the position of one girl relative to another in the circle -/
def position (c : CircularArrangement) (left right : ℕ) : Prop :=
  left + right + 2 = c.girls

/-- Theorem: If Florence is the 4th on the left and 7th on the right from Jess,
    then there are 11 girls in total -/
theorem eleven_girls (c : CircularArrangement) :
  position c 3 6 → c.girls = 11 := by
  sorry

#check eleven_girls

end NUMINAMATH_CALUDE_eleven_girls_l2168_216862


namespace NUMINAMATH_CALUDE_limit_sin_x_over_x_sin_x_over_x_squeeze_cos_continuous_limit_sin_x_over_x_equals_one_l2168_216808

open Real
open Topology
open Filter

theorem limit_sin_x_over_x : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |sin x / x - 1| < ε :=
by
  sorry

theorem sin_x_over_x_squeeze (x : ℝ) (h : x ≠ 0) (h' : |x| < π/2) :
  cos x < sin x / x ∧ sin x / x < 1 :=
by
  sorry

theorem cos_continuous : Continuous cos :=
by
  sorry

theorem limit_sin_x_over_x_equals_one :
  Tendsto (λ x => sin x / x) (𝓝[≠] 0) (𝓝 1) :=
by
  sorry

end NUMINAMATH_CALUDE_limit_sin_x_over_x_sin_x_over_x_squeeze_cos_continuous_limit_sin_x_over_x_equals_one_l2168_216808


namespace NUMINAMATH_CALUDE_g_of_5_l2168_216882

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3g(2 - x) = 4x^2 - 5x + 1 for all x ∈ ℝ,
    prove that g(5) = -5/4 -/
theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 5 * x + 1) :
  g 5 = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l2168_216882


namespace NUMINAMATH_CALUDE_graduate_ratio_l2168_216845

theorem graduate_ratio (N : ℝ) (G : ℝ) (C : ℝ) 
  (h1 : C = (2/3) * N) 
  (h2 : G / (G + C) = 0.15789473684210525) : 
  G = (1/8) * N := by
sorry

end NUMINAMATH_CALUDE_graduate_ratio_l2168_216845


namespace NUMINAMATH_CALUDE_time_subtraction_problem_l2168_216818

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Converts minutes to a Time structure -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60,
    minutes := m % 60,
    valid := by sorry }

/-- Subtracts two Time structures -/
def subtractTime (t1 t2 : Time) : Time :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  minutesToTime (totalMinutes1 - totalMinutes2)

theorem time_subtraction_problem :
  let currentTime : Time := { hours := 18, minutes := 27, valid := by sorry }
  let minutesToSubtract : ℕ := 2880717
  let resultTime : Time := subtractTime currentTime (minutesToTime minutesToSubtract)
  resultTime.hours = 6 ∧ resultTime.minutes = 30 := by sorry

end NUMINAMATH_CALUDE_time_subtraction_problem_l2168_216818


namespace NUMINAMATH_CALUDE_mean_increases_median_may_unchanged_variance_increases_l2168_216860

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (x_n_plus_1 : ℝ)

-- Assume n ≥ 3
axiom n_ge_3 : n ≥ 3

-- Define the median, mean, and variance of the original dataset
def median : ℝ := sorry
def mean : ℝ := sorry
def variance : ℝ := sorry

-- Assume x_n_plus_1 is much greater than any value in x
axiom x_n_plus_1_much_greater : ∀ i : Fin n, x_n_plus_1 > x i

-- Define the new dataset including x_n_plus_1
def new_dataset : Fin (n + 1) → ℝ :=
  λ i => if h : i.val < n then x ⟨i.val, h⟩ else x_n_plus_1

-- Define the new median, mean, and variance
def new_median : ℝ := sorry
def new_mean : ℝ := sorry
def new_variance : ℝ := sorry

-- Theorem statements
theorem mean_increases : new_mean > mean := sorry
theorem median_may_unchanged : new_median = median ∨ new_median > median := sorry
theorem variance_increases : new_variance > variance := sorry

end NUMINAMATH_CALUDE_mean_increases_median_may_unchanged_variance_increases_l2168_216860


namespace NUMINAMATH_CALUDE_estate_area_calculation_l2168_216836

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 350

/-- Represents the length of the rectangle on the map in inches -/
def map_length : ℝ := 9

/-- Represents the width of the rectangle on the map in inches -/
def map_width : ℝ := 6

/-- Calculates the actual length of the estate in miles -/
def actual_length : ℝ := scale * map_length

/-- Calculates the actual width of the estate in miles -/
def actual_width : ℝ := scale * map_width

/-- Calculates the actual area of the estate in square miles -/
def actual_area : ℝ := actual_length * actual_width

theorem estate_area_calculation :
  actual_area = 6615000 := by sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l2168_216836


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2168_216843

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x^2 + 16) = 12 ↔ x = 8 * Real.sqrt 2 ∨ x = -8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2168_216843


namespace NUMINAMATH_CALUDE_largest_value_in_interval_l2168_216871

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) :
  max (max (max (max x (x^3)) (3*x)) (x^(1/3))) (1/x) = 1/x := by sorry

end NUMINAMATH_CALUDE_largest_value_in_interval_l2168_216871


namespace NUMINAMATH_CALUDE_beach_population_evening_l2168_216852

/-- The number of people at the beach in the evening -/
def beach_population (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the total number of people at the beach in the evening -/
theorem beach_population_evening :
  beach_population 3 100 40 = 63 := by
  sorry

end NUMINAMATH_CALUDE_beach_population_evening_l2168_216852


namespace NUMINAMATH_CALUDE_product_357_sum_28_l2168_216831

theorem product_357_sum_28 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 357 →
  (a : ℕ) + b + c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_product_357_sum_28_l2168_216831


namespace NUMINAMATH_CALUDE_segment_length_l2168_216868

/-- Given a line segment CD with points R and S on it, prove that CD has length 146.2/11 -/
theorem segment_length (C D R S : ℝ) : 
  (R > C) →  -- R is to the right of C
  (S > R) →  -- S is to the right of R
  (D > S) →  -- D is to the right of S
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 4 / 7 →  -- S divides CD in ratio 4:7
  S - R = 1 →  -- RS = 1
  D - C = 146.2 / 11 := by
sorry

end NUMINAMATH_CALUDE_segment_length_l2168_216868


namespace NUMINAMATH_CALUDE_bernoulli_zero_success_l2168_216814

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is (5/7)^7 -/
theorem bernoulli_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by sorry

end NUMINAMATH_CALUDE_bernoulli_zero_success_l2168_216814


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2168_216878

theorem sum_of_four_numbers : 1357 + 7531 + 3175 + 5713 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2168_216878


namespace NUMINAMATH_CALUDE_second_week_rainfall_l2168_216851

/-- Proves that the rainfall during the second week of January was 15 inches,
    given the total rainfall and the relationship between the two weeks. -/
theorem second_week_rainfall (total_rainfall : ℝ) (first_week : ℝ) (second_week : ℝ) : 
  total_rainfall = 25 →
  second_week = 1.5 * first_week →
  total_rainfall = first_week + second_week →
  second_week = 15 := by
  sorry


end NUMINAMATH_CALUDE_second_week_rainfall_l2168_216851


namespace NUMINAMATH_CALUDE_copying_result_correct_l2168_216840

/-- Represents the copying cost and discount structure -/
structure CopyingCost where
  cost_per_5_pages : ℚ  -- Cost in cents for 5 pages
  budget : ℚ           -- Budget in dollars
  discount_rate : ℚ    -- Discount rate after 1000 pages
  discount_threshold : ℕ -- Number of pages after which discount applies

/-- Calculates the total number of pages that can be copied and the total cost with discount -/
def calculate_copying_result (c : CopyingCost) : ℕ × ℚ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem copying_result_correct (c : CopyingCost) :
  c.cost_per_5_pages = 10 ∧ 
  c.budget = 50 ∧ 
  c.discount_rate = 0.1 ∧
  c.discount_threshold = 1000 →
  calculate_copying_result c = (2500, 47) :=
sorry

end NUMINAMATH_CALUDE_copying_result_correct_l2168_216840


namespace NUMINAMATH_CALUDE_paving_rate_per_square_meter_l2168_216889

/-- Given a room with length 5.5 m and width 3.75 m, and a total paving cost of Rs. 16500,
    the rate of paving per square meter is Rs. 800. -/
theorem paving_rate_per_square_meter
  (length : ℝ)
  (width : ℝ)
  (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
sorry

end NUMINAMATH_CALUDE_paving_rate_per_square_meter_l2168_216889


namespace NUMINAMATH_CALUDE_inclination_angle_theorem_l2168_216809

-- Define the line equation
def line_equation (x y α : ℝ) : Prop := x * Real.cos α + Real.sqrt 3 * y + 2 = 0

-- Define the range of cos α
def cos_α_range (α : ℝ) : Prop := -1 ≤ Real.cos α ∧ Real.cos α ≤ 1

-- Define the range of θ
def θ_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < Real.pi

-- Define the inclination angle range
def inclination_angle_range (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

-- Theorem statement
theorem inclination_angle_theorem (x y α θ : ℝ) :
  line_equation x y α → cos_α_range α → θ_range θ →
  inclination_angle_range θ := by sorry

end NUMINAMATH_CALUDE_inclination_angle_theorem_l2168_216809


namespace NUMINAMATH_CALUDE_kittens_given_to_jessica_l2168_216869

theorem kittens_given_to_jessica (initial_kittens : ℕ) (received_kittens : ℕ) (final_kittens : ℕ) :
  initial_kittens = 6 →
  received_kittens = 9 →
  final_kittens = 12 →
  initial_kittens + received_kittens - final_kittens = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_kittens_given_to_jessica_l2168_216869


namespace NUMINAMATH_CALUDE_division_problem_addition_problem_multiplication_problem_l2168_216803

-- Problem 1
theorem division_problem : 246 / 73 = 3 + 27 / 73 := by sorry

-- Problem 2
theorem addition_problem : 9999 + 999 + 99 + 9 = 11106 := by sorry

-- Problem 3
theorem multiplication_problem : 25 * 29 * 4 = 2900 := by sorry

end NUMINAMATH_CALUDE_division_problem_addition_problem_multiplication_problem_l2168_216803


namespace NUMINAMATH_CALUDE_base_seven_5432_equals_1934_l2168_216813

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base_seven_5432_equals_1934 : 
  base_seven_to_ten [2, 3, 4, 5] = 1934 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_5432_equals_1934_l2168_216813


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2168_216863

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x^2 + y^2 = 4) 
  (h2 : (x - y)^2 = 5) : 
  (x + y)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2168_216863


namespace NUMINAMATH_CALUDE_course_choice_related_probability_three_males_l2168_216899

-- Define the total number of students
def total_students : ℕ := 200

-- Define the number of female students
def female_students : ℕ := 80

-- Define the number of female students majoring in the field
def female_major : ℕ := 70

-- Define the number of male students not majoring in the field
def male_non_major : ℕ := 40

-- Define the chi-square statistic threshold for 99.9% certainty
def chi_square_threshold : ℚ := 10828 / 1000

-- Define the function to calculate the chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem for the relationship between course choice, major, and gender
theorem course_choice_related :
  let male_students : ℕ := total_students - female_students
  let male_major : ℕ := male_students - male_non_major
  let female_non_major : ℕ := female_students - female_major
  chi_square female_major male_major female_non_major male_non_major > chi_square_threshold := by sorry

-- Theorem for the probability of selecting 3 males out of 5 students
theorem probability_three_males :
  (Nat.choose 4 3 : ℚ) / (Nat.choose 5 3) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_course_choice_related_probability_three_males_l2168_216899


namespace NUMINAMATH_CALUDE_sqrt_two_decomposition_l2168_216886

theorem sqrt_two_decomposition :
  ∃ (a : ℤ) (b : ℝ), 
    (Real.sqrt 2 = a + b) ∧ 
    (0 ≤ b) ∧ 
    (b < 1) ∧ 
    (a = 1) ∧ 
    (1 / b = Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_decomposition_l2168_216886


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l2168_216802

theorem sin_negative_1740_degrees : Real.sin (-(1740 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l2168_216802


namespace NUMINAMATH_CALUDE_range_of_a_for_positive_x_l2168_216866

theorem range_of_a_for_positive_x (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2*x - a = 3*x - 4) ↔ a < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_positive_x_l2168_216866


namespace NUMINAMATH_CALUDE_opposite_of_gold_is_olive_l2168_216854

-- Define the colors
inductive Color
  | Aqua | Maroon | Olive | Purple | Silver | Gold | Black

-- Define the cube faces
structure CubeFace where
  color : Color

-- Define the cube
structure Cube where
  faces : List CubeFace
  gold_face : CubeFace
  opposite_face : CubeFace

-- Define the cross pattern
structure CrossPattern where
  squares : List CubeFace

-- Function to fold the cross pattern into a cube
def fold_cross_to_cube (cross : CrossPattern) : Cube :=
  sorry

-- Theorem: The face opposite to Gold is Olive
theorem opposite_of_gold_is_olive (cross : CrossPattern) 
  (cube : Cube := fold_cross_to_cube cross) : 
  cube.gold_face.color = Color.Gold → cube.opposite_face.color = Color.Olive :=
sorry

end NUMINAMATH_CALUDE_opposite_of_gold_is_olive_l2168_216854


namespace NUMINAMATH_CALUDE_no_solution_when_x_is_five_l2168_216898

theorem no_solution_when_x_is_five (x : ℝ) (y : ℝ) :
  x = 5 → ¬∃y, 1 / (x + 5) + y = 1 / (x - 5) := by
sorry

end NUMINAMATH_CALUDE_no_solution_when_x_is_five_l2168_216898


namespace NUMINAMATH_CALUDE_least_exponent_sum_is_correct_l2168_216895

/-- Represents a sum of distinct powers of 2 -/
structure DistinctPowerSum where
  powers : List Nat
  distinct : powers.Pairwise (·≠·)
  sum_eq_500 : (powers.map (2^·)).sum = 500

/-- The least possible sum of exponents when expressing 500 as a sum of at least two distinct powers of 2 -/
def least_exponent_sum : Nat := 30

theorem least_exponent_sum_is_correct :
  ∀ (dps : DistinctPowerSum),
    dps.powers.length ≥ 2 →
    dps.powers.sum ≥ least_exponent_sum ∧
    ∃ (optimal : DistinctPowerSum),
      optimal.powers.length ≥ 2 ∧
      optimal.powers.sum = least_exponent_sum :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_is_correct_l2168_216895


namespace NUMINAMATH_CALUDE_equation_condition_l2168_216867

theorem equation_condition (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  (11 * a + b) * (11 * a + c) = 121 * a * (a + 1) + 11 * b * c ↔ b + c = 11 :=
sorry

end NUMINAMATH_CALUDE_equation_condition_l2168_216867


namespace NUMINAMATH_CALUDE_cut_out_pieces_border_l2168_216890

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a piece that can be cut out from the grid -/
inductive Piece
  | UnitSquare
  | LShape

/-- Represents the configuration of cut-out pieces -/
structure CutOutConfig :=
  (grid : Grid)
  (unitSquares : ℕ)
  (lShapes : ℕ)

/-- Predicate to check if two pieces border each other -/
def border (p1 p2 : Piece) : Prop := sorry

theorem cut_out_pieces_border
  (config : CutOutConfig)
  (h1 : config.grid.size = 55)
  (h2 : config.unitSquares = 500)
  (h3 : config.lShapes = 400) :
  ∃ (p1 p2 : Piece), p1 ≠ p2 ∧ border p1 p2 :=
sorry

end NUMINAMATH_CALUDE_cut_out_pieces_border_l2168_216890


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l2168_216804

-- Define the function f
def f (x : ℝ) : ℝ := |3*x + 2|

-- Theorem for part 1
theorem inequality_solution_set :
  {x : ℝ | f x < 4 - |x - 1|} = {x : ℝ | -5/4 < x ∧ x < 1/2} :=
sorry

-- Theorem for part 2
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ |x - a| - f x ≤ 1/m + 1/n) →
  ∃ a : ℝ, 0 < a ∧ a ≤ 10/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l2168_216804


namespace NUMINAMATH_CALUDE_complementary_angles_proof_l2168_216848

theorem complementary_angles_proof (A B : Real) : 
  A + B = 90 →  -- Angles A and B are complementary
  A = 4 * B →   -- Measure of angle A is 4 times angle B
  A = 72 ∧ B = 18 := by
sorry

end NUMINAMATH_CALUDE_complementary_angles_proof_l2168_216848


namespace NUMINAMATH_CALUDE_school_garden_flowers_l2168_216837

theorem school_garden_flowers :
  let total_flowers : ℕ := 96
  let green_flowers : ℕ := 9
  let red_flowers : ℕ := 3 * green_flowers
  let blue_flowers : ℕ := total_flowers / 2
  let yellow_flowers : ℕ := total_flowers - (green_flowers + red_flowers + blue_flowers)
  yellow_flowers = 12 := by
sorry

end NUMINAMATH_CALUDE_school_garden_flowers_l2168_216837


namespace NUMINAMATH_CALUDE_ethan_candle_coconut_oil_l2168_216823

/-- The amount of coconut oil used in each candle, given the total weight of candles,
    the number of candles, and the amount of beeswax per candle. -/
def coconut_oil_per_candle (total_weight : ℕ) (num_candles : ℕ) (beeswax_per_candle : ℕ) : ℚ :=
  (total_weight - num_candles * beeswax_per_candle) / num_candles

theorem ethan_candle_coconut_oil :
  coconut_oil_per_candle 63 (10 - 3) 8 = 1 := by sorry

end NUMINAMATH_CALUDE_ethan_candle_coconut_oil_l2168_216823


namespace NUMINAMATH_CALUDE_a_annual_income_l2168_216874

/-- Proves that A's annual income is 403200 given the specified conditions -/
theorem a_annual_income (c_income : ℕ) (h1 : c_income = 12000) : ∃ (a_income b_income : ℕ),
  (a_income : ℚ) / b_income = 5 / 2 ∧
  b_income = c_income + c_income * 12 / 100 ∧
  a_income * 12 = 403200 :=
by sorry

end NUMINAMATH_CALUDE_a_annual_income_l2168_216874


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2168_216879

theorem multiplication_subtraction_equality : 154 * 1836 - 54 * 1836 = 183600 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2168_216879


namespace NUMINAMATH_CALUDE_largest_class_size_l2168_216805

/-- Represents the number of students in the largest class of a school -/
def largest_class (n : ℕ) : ℕ := n

/-- Represents the total number of students in the school -/
def total_students (n : ℕ) : ℕ := 
  (largest_class n) + 
  (largest_class n - 2) + 
  (largest_class n - 4) + 
  (largest_class n - 6) + 
  (largest_class n - 8)

/-- Theorem stating that the largest class has 25 students -/
theorem largest_class_size : 
  (total_students 25 = 105) ∧ (largest_class 25 = 25) := by
  sorry

#check largest_class_size

end NUMINAMATH_CALUDE_largest_class_size_l2168_216805


namespace NUMINAMATH_CALUDE_books_to_read_l2168_216826

theorem books_to_read (total : ℕ) (mcgregor : ℕ) (floyd : ℕ) : 
  total = 89 → mcgregor = 34 → floyd = 32 → total - (mcgregor + floyd) = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_to_read_l2168_216826


namespace NUMINAMATH_CALUDE_one_fifths_in_ten_thirds_l2168_216885

theorem one_fifths_in_ten_thirds :
  (10 : ℚ) / 3 / (1 / 5) = 50 / 3 := by sorry

end NUMINAMATH_CALUDE_one_fifths_in_ten_thirds_l2168_216885


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2168_216888

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2168_216888


namespace NUMINAMATH_CALUDE_orange_harvest_difference_l2168_216844

/-- Represents the harvest rates for a type of orange --/
structure HarvestRate where
  ripe : ℕ
  unripe : ℕ

/-- Represents the harvest rates for weekdays and weekends --/
structure WeeklyHarvestRate where
  weekday : HarvestRate
  weekend : HarvestRate

/-- Calculates the total difference between ripe and unripe oranges for a week --/
def weeklyDifference (rate : WeeklyHarvestRate) : ℕ :=
  (rate.weekday.ripe * 5 + rate.weekend.ripe * 2) -
  (rate.weekday.unripe * 5 + rate.weekend.unripe * 2)

theorem orange_harvest_difference :
  let valencia := WeeklyHarvestRate.mk (HarvestRate.mk 90 38) (HarvestRate.mk 75 33)
  let navel := WeeklyHarvestRate.mk (HarvestRate.mk 125 65) (HarvestRate.mk 100 57)
  let blood := WeeklyHarvestRate.mk (HarvestRate.mk 60 42) (HarvestRate.mk 45 36)
  weeklyDifference valencia + weeklyDifference navel + weeklyDifference blood = 838 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_difference_l2168_216844


namespace NUMINAMATH_CALUDE_prob_one_boy_one_girl_prob_one_boy_one_girl_given_boy_prob_one_boy_one_girl_given_monday_boy_l2168_216855

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Represents the day of the week -/
inductive Day
| Monday
| OtherDay

/-- Represents a child with their gender and birth day -/
structure Child :=
  (gender : Gender)
  (birthDay : Day)

/-- Represents a family with two children -/
structure Family :=
  (child1 : Child)
  (child2 : Child)

/-- The probability of having a boy or a girl is equal -/
axiom equal_gender_probability : ℝ

/-- The probability of being born on a Monday -/
axiom monday_probability : ℝ

/-- Theorem for the probability of having one boy and one girl in a family with two children -/
theorem prob_one_boy_one_girl : ℝ := by sorry

/-- Theorem for the probability of having one boy and one girl, given that one child is a boy -/
theorem prob_one_boy_one_girl_given_boy : ℝ := by sorry

/-- Theorem for the probability of having one boy and one girl, given that one child is a boy born on a Monday -/
theorem prob_one_boy_one_girl_given_monday_boy : ℝ := by sorry

end NUMINAMATH_CALUDE_prob_one_boy_one_girl_prob_one_boy_one_girl_given_boy_prob_one_boy_one_girl_given_monday_boy_l2168_216855


namespace NUMINAMATH_CALUDE_paths_to_n_2_l2168_216820

/-- The number of possible paths from (0,0) to (x, y) -/
def f (x y : ℕ) : ℕ := sorry

/-- The theorem stating that f(n, 2) = (1/2)(n^2 + 3n + 2) for all natural numbers n -/
theorem paths_to_n_2 (n : ℕ) : f n 2 = (n^2 + 3*n + 2) / 2 := by sorry

end NUMINAMATH_CALUDE_paths_to_n_2_l2168_216820


namespace NUMINAMATH_CALUDE_nurse_lice_check_l2168_216877

/-- The number of Kindergarteners the nurse needs to check -/
def kindergarteners_to_check : ℕ := by sorry

theorem nurse_lice_check :
  let first_graders : ℕ := 19
  let second_graders : ℕ := 20
  let third_graders : ℕ := 25
  let minutes_per_check : ℕ := 2
  let total_hours : ℕ := 3
  let total_minutes : ℕ := total_hours * 60
  
  kindergarteners_to_check = 
    (total_minutes - 
      (first_graders + second_graders + third_graders) * minutes_per_check) / 
    minutes_per_check :=
by sorry

end NUMINAMATH_CALUDE_nurse_lice_check_l2168_216877


namespace NUMINAMATH_CALUDE_empire_state_building_height_l2168_216839

/-- The height of the Empire State Building to the top floor -/
def height_to_top_floor : ℝ := 1454 - 204

/-- The total height of the Empire State Building -/
def total_height : ℝ := 1454

/-- The height of the antenna spire -/
def antenna_height : ℝ := 204

theorem empire_state_building_height : height_to_top_floor = 1250 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_building_height_l2168_216839


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l2168_216891

theorem arithmetic_progression_problem (a d : ℚ) : 
  (3 * ((a - d) + a) = 2 * (a + d)) →
  ((a - 2)^2 = (a - d) * (a + d)) →
  ((a = 5 ∧ d = 4) ∨ (a = 5/4 ∧ d = 1)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l2168_216891


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2168_216849

theorem parallelogram_side_length 
  (s : ℝ) 
  (area : ℝ) 
  (angle : ℝ) :
  s > 0 → 
  angle = π / 6 → 
  area = 27 * Real.sqrt 3 → 
  3 * s * s * Real.sqrt 3 = area → 
  s = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2168_216849


namespace NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l2168_216856

-- Define 12.06 million
def twelve_point_zero_six_million : ℝ := 12.06 * 1000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.206 * 10^7

-- Theorem statement
theorem twelve_point_zero_six_million_scientific_notation :
  twelve_point_zero_six_million = scientific_notation :=
by sorry

end NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l2168_216856


namespace NUMINAMATH_CALUDE_exactly_one_even_iff_not_all_odd_or_two_even_l2168_216807

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨
  (Odd a ∧ Even b ∧ Odd c) ∨
  (Odd a ∧ Odd b ∧ Even c)

def all_odd_or_two_even (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∨
  (Even a ∧ Even b) ∨
  (Even a ∧ Even c) ∨
  (Even b ∧ Even c)

theorem exactly_one_even_iff_not_all_odd_or_two_even (a b c : ℕ) :
  exactly_one_even a b c ↔ ¬(all_odd_or_two_even a b c) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_even_iff_not_all_odd_or_two_even_l2168_216807


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l2168_216810

/-- Given the total weight of strawberries collected by Marco and his dad,
    the weight of strawberries lost by Marco's dad, and the weight of
    Marco's dad's remaining strawberries, prove that Marco's strawberries
    weigh 12 pounds. -/
theorem marcos_strawberries_weight
  (total_weight : ℕ)
  (dads_lost_weight : ℕ)
  (dads_remaining_weight : ℕ)
  (h1 : total_weight = 36)
  (h2 : dads_lost_weight = 8)
  (h3 : dads_remaining_weight = 16) :
  total_weight - (dads_remaining_weight + dads_lost_weight) = 12 :=
by sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l2168_216810


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2168_216881

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (hm_ne_n : m ≠ n) 
  (hα_ne_β : α ≠ β) 
  (hm_perp_α : perp m α) 
  (hm_perp_β : perp m β) 
  (hn_perp_α : perp n α) : 
  perp n β := by sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2168_216881


namespace NUMINAMATH_CALUDE_star_equation_solution_l2168_216834

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

/-- Theorem stating that if 6 ★ x = 45, then x = 19/3 -/
theorem star_equation_solution :
  (star 6 x = 45) → x = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2168_216834


namespace NUMINAMATH_CALUDE_alternative_interest_rate_l2168_216876

theorem alternative_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (chosen_rate : ℝ) 
  (interest_difference : ℝ) : ℝ :=
  let alternative_rate := 
    (principal * chosen_rate * time - interest_difference) / (principal * time)
  
  -- Assumptions
  have h1 : principal = 7000 := by sorry
  have h2 : time = 2 := by sorry
  have h3 : chosen_rate = 0.15 := by sorry
  have h4 : interest_difference = 420 := by sorry

  -- Theorem statement
  alternative_rate * 100

/- Proof
  sorry
-/

end NUMINAMATH_CALUDE_alternative_interest_rate_l2168_216876


namespace NUMINAMATH_CALUDE_chocolate_milk_total_ounces_l2168_216806

-- Define the ingredients per glass
def milk_per_glass : ℚ := 6
def syrup_per_glass : ℚ := 1.5
def cream_per_glass : ℚ := 0.5

-- Define the total available ingredients
def total_milk : ℚ := 130
def total_syrup : ℚ := 60
def total_cream : ℚ := 25

-- Define the size of each glass
def glass_size : ℚ := 8

-- Theorem to prove
theorem chocolate_milk_total_ounces :
  let max_glasses := min (total_milk / milk_per_glass) 
                         (min (total_syrup / syrup_per_glass) (total_cream / cream_per_glass))
  let full_glasses := ⌊max_glasses⌋
  full_glasses * glass_size = 168 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_total_ounces_l2168_216806


namespace NUMINAMATH_CALUDE_gcd_7584_18027_l2168_216830

theorem gcd_7584_18027 : Nat.gcd 7584 18027 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7584_18027_l2168_216830


namespace NUMINAMATH_CALUDE_extreme_values_imply_a_b_values_inequality_implies_m_range_l2168_216875

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x - b / x + Real.log x

def g (m x : ℝ) : ℝ := x^2 - 2 * m * x + m

def has_extreme_values (f : ℝ → ℝ) (x₁ x₂ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ (Set.Ioo (x₁ - ε) (x₁ + ε) ∪ Set.Ioo (x₂ - ε) (x₂ + ε)),
    f x ≤ f x₁ ∧ f x ≤ f x₂

theorem extreme_values_imply_a_b_values (a b : ℝ) :
  has_extreme_values (f a b) 1 (1/2) → a = -1/3 ∧ b = -1/3 :=
sorry

theorem inequality_implies_m_range (a b m : ℝ) :
  (a = -1/3 ∧ b = -1/3) →
  (∀ x₁ ∈ Set.Icc (1/2) 2, ∃ x₂ ∈ Set.Icc (1/2) 2, g m x₁ ≥ f a b x₂ - Real.log x₂) →
  m ≤ (3 + Real.sqrt 51) / 6 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_imply_a_b_values_inequality_implies_m_range_l2168_216875


namespace NUMINAMATH_CALUDE_log_zero_nonexistent_l2168_216880

theorem log_zero_nonexistent : ¬ ∃ x : ℝ, Real.log x = 0 := by sorry

end NUMINAMATH_CALUDE_log_zero_nonexistent_l2168_216880


namespace NUMINAMATH_CALUDE_probability_same_color_half_l2168_216827

/-- Represents a bag of colored balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing balls of the same color from two bags -/
def probability_same_color (bag_a bag_b : Bag) : ℚ :=
  let total_a := bag_a.white + bag_a.red
  let total_b := bag_b.white + bag_b.red
  (bag_a.white * bag_b.white + bag_a.red * bag_b.red) / (total_a * total_b)

/-- The main theorem stating that the probability of drawing balls of the same color
    from the given bags is 1/2 -/
theorem probability_same_color_half :
  let bag_a : Bag := ⟨8, 4⟩
  let bag_b : Bag := ⟨6, 6⟩
  probability_same_color bag_a bag_b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_probability_same_color_half_l2168_216827


namespace NUMINAMATH_CALUDE_simplify_expression_l2168_216842

theorem simplify_expression (z : ℝ) : z - 3 + 4*z + 5 - 6*z + 7 - 8*z + 9 = -9*z + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2168_216842


namespace NUMINAMATH_CALUDE_min_cubes_for_representation_l2168_216822

/-- The number of faces on each cube -/
def faces_per_cube : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The length of the number we want to represent -/
def number_length : ℕ := 30

/-- The minimum number of occurrences needed for digits 1-9 -/
def min_occurrences : ℕ := 30

/-- The minimum number of occurrences needed for digit 0 -/
def min_occurrences_zero : ℕ := 29

/-- The total number of digit occurrences needed -/
def total_occurrences : ℕ := (num_digits - 1) * min_occurrences + min_occurrences_zero

/-- The minimum number of cubes needed -/
def min_cubes : ℕ := (total_occurrences + faces_per_cube - 1) / faces_per_cube

theorem min_cubes_for_representation :
  min_cubes = 50 ∧
  min_cubes * faces_per_cube ≥ total_occurrences ∧
  (min_cubes - 1) * faces_per_cube < total_occurrences :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_for_representation_l2168_216822


namespace NUMINAMATH_CALUDE_garden_length_proof_l2168_216850

theorem garden_length_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 2 + 3 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 38 := by
sorry

end NUMINAMATH_CALUDE_garden_length_proof_l2168_216850


namespace NUMINAMATH_CALUDE_binomial_sum_modulo_1000_l2168_216832

theorem binomial_sum_modulo_1000 : 
  (Finset.sum (Finset.range 503) (fun k => Nat.choose 2011 (4 * k))) % 1000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_modulo_1000_l2168_216832


namespace NUMINAMATH_CALUDE_rational_coordinates_solution_l2168_216859

theorem rational_coordinates_solution (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 := by
  -- We claim that y = 1 - x satisfies the equation
  let y := 1 - x
  -- Existential introduction
  use y
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_rational_coordinates_solution_l2168_216859


namespace NUMINAMATH_CALUDE_average_weight_increase_l2168_216815

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 7 * initial_average
  let new_total_weight := initial_total_weight - 75 + 99.5
  let new_average := new_total_weight / 7
  new_average - initial_average = 3.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2168_216815


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l2168_216883

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  (Nat.factorial 4) * 
  (Nat.factorial chickens) * 
  (Nat.factorial dogs) * 
  (Nat.factorial cats) * 
  (Nat.factorial rabbits)

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 4 3 5 2 = 414720 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l2168_216883


namespace NUMINAMATH_CALUDE_equation_solution_l2168_216812

theorem equation_solution : 
  {x : ℝ | x + 45 / (x - 4) = -10} = {-1, -5} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2168_216812


namespace NUMINAMATH_CALUDE_probability_of_draw_l2168_216801

/-- Given two players A and B playing chess, this theorem proves the probability of a draw. -/
theorem probability_of_draw 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_draw_l2168_216801


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2168_216861

theorem arithmetic_mean_sqrt2 :
  let x := Real.sqrt 2 - 1
  (x + (1 / x)) / 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2168_216861


namespace NUMINAMATH_CALUDE_intersection_product_l2168_216828

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4*x

def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 3 * Real.sqrt 3 = 0

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define the intersection points P and Q
def isIntersection (p : ℝ × ℝ) : Prop :=
  C₁ p.1 p.2 ∧ C₂ p.1 p.2

-- State the theorem
theorem intersection_product :
  ∃ (P Q : ℝ × ℝ), isIntersection P ∧ isIntersection Q ∧ P ≠ Q ∧
    (P.1 - A.1)^2 + (P.2 - A.2)^2 * ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) = 3^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_l2168_216828


namespace NUMINAMATH_CALUDE_recurring_decimal_product_l2168_216858

theorem recurring_decimal_product : 
  (8 : ℚ) / 99 * (4 : ℚ) / 11 = (32 : ℚ) / 1089 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_product_l2168_216858


namespace NUMINAMATH_CALUDE_cookie_difference_l2168_216811

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def raisin_cookies : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_today : ℕ := 237

/-- The total number of chocolate chip cookies Helen baked -/
def total_choc_cookies : ℕ := cookies_yesterday + cookies_today

/-- The difference between chocolate chip cookies and raisin cookies -/
theorem cookie_difference : total_choc_cookies - raisin_cookies = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l2168_216811


namespace NUMINAMATH_CALUDE_shaded_area_of_square_with_removed_triangles_l2168_216819

/-- The area of a shape formed by removing four right triangles from a square -/
theorem shaded_area_of_square_with_removed_triangles 
  (square_side : ℝ) 
  (triangle_leg : ℝ) 
  (h1 : square_side = 6) 
  (h2 : triangle_leg = 2) : 
  square_side ^ 2 - 4 * (1 / 2 * triangle_leg ^ 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_with_removed_triangles_l2168_216819


namespace NUMINAMATH_CALUDE_june_sales_increase_l2168_216864

def normal_monthly_sales : ℕ := 21122
def june_july_combined_sales : ℕ := 46166

theorem june_sales_increase : 
  (june_july_combined_sales - 2 * normal_monthly_sales) = 3922 := by
  sorry

end NUMINAMATH_CALUDE_june_sales_increase_l2168_216864


namespace NUMINAMATH_CALUDE_robin_gum_count_l2168_216833

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 12

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 20

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 240 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l2168_216833


namespace NUMINAMATH_CALUDE_unicorn_tether_problem_l2168_216857

theorem unicorn_tether_problem (rope_length : ℝ) (tower_radius : ℝ) (unicorn_height : ℝ) 
  (rope_end_distance : ℝ) (p q r : ℕ) (h_rope_length : rope_length = 25)
  (h_tower_radius : tower_radius = 10) (h_unicorn_height : unicorn_height = 5)
  (h_rope_end_distance : rope_end_distance = 5) (h_r_prime : Nat.Prime r)
  (h_rope_tower_length : (p - Real.sqrt q) / r = 
    rope_length - Real.sqrt ((rope_end_distance + tower_radius)^2 + unicorn_height^2)) :
  p + q + r = 1128 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_tether_problem_l2168_216857


namespace NUMINAMATH_CALUDE_connor_date_cost_l2168_216846

/-- Calculates the total cost of Connor's movie date --/
def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price

/-- Theorem stating that the total cost of Connor's movie date is $36.00 --/
theorem connor_date_cost :
  movie_date_cost 10 11 2.5 = 36 :=
by sorry

end NUMINAMATH_CALUDE_connor_date_cost_l2168_216846


namespace NUMINAMATH_CALUDE_special_polynomial_at_five_l2168_216821

/-- A cubic polynomial satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ n ∈ ({1, 2, 3, 4, 6} : Set ℝ), p n = 1 / n^2) ∧
  p 0 = -1/25

/-- The main theorem -/
theorem special_polynomial_at_five 
  (p : ℝ → ℝ) 
  (h : special_polynomial p) : 
  p 5 = 20668/216000 := by
sorry

end NUMINAMATH_CALUDE_special_polynomial_at_five_l2168_216821


namespace NUMINAMATH_CALUDE_pizza_and_burgers_cost_l2168_216829

/-- The cost of a burger in dollars -/
def burger_cost : ℕ := 9

/-- The cost of a pizza in dollars -/
def pizza_cost : ℕ := 2 * burger_cost

/-- The total cost of one pizza and three burgers in dollars -/
def total_cost : ℕ := pizza_cost + 3 * burger_cost

theorem pizza_and_burgers_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_pizza_and_burgers_cost_l2168_216829


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2168_216897

theorem least_subtraction_for_divisibility (n m k : ℕ) (h : n - k ≥ 0) : 
  (∃ q : ℕ, n - k = m * q) ∧ 
  (∀ j : ℕ, j < k → ¬(∃ q : ℕ, n - j = m * q)) → 
  k = n % m :=
sorry

#check least_subtraction_for_divisibility 2361 23 15

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2168_216897


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l2168_216817

theorem hotel_room_charge_comparison (G P R : ℝ) 
  (hP_G : P = G * 0.8)
  (hR_G : R = G * 1.6) :
  (R - P) / R * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l2168_216817


namespace NUMINAMATH_CALUDE_not_perfect_square_infinitely_often_l2168_216800

theorem not_perfect_square_infinitely_often (a b : ℕ+) (h : ∃ p : ℕ, Nat.Prime p ∧ b = a + p) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, ¬∃ k : ℕ, (a^n + a + 1) * (b^n + b + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_infinitely_often_l2168_216800


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2168_216824

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 0 → 
  2 - x ≥ 0 → 
  (((Real.sqrt (2 - x) + 4 * x - 3) / x ≥ 2) ↔ (x < 0 ∨ (1 ≤ x ∧ x ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2168_216824


namespace NUMINAMATH_CALUDE_cross_in_square_l2168_216872

theorem cross_in_square (x : ℝ) : 
  x > 0 → 
  (5 / 8) * x^2 = 810 → 
  x = 36 :=
by sorry

end NUMINAMATH_CALUDE_cross_in_square_l2168_216872


namespace NUMINAMATH_CALUDE_common_chord_equation_l2168_216838

theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 = 4) ∧ (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
  (x - y + 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2168_216838


namespace NUMINAMATH_CALUDE_shaded_volume_is_112_l2168_216825

/-- The volume of a rectangular prism with dimensions a, b, and c -/
def volume (a b c : ℕ) : ℕ := a * b * c

/-- The dimensions of the larger prism -/
def large_prism : Fin 3 → ℕ
| 0 => 4
| 1 => 5
| 2 => 6
| _ => 0

/-- The dimensions of the smaller prism -/
def small_prism : Fin 3 → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| _ => 0

theorem shaded_volume_is_112 :
  volume (large_prism 0) (large_prism 1) (large_prism 2) -
  volume (small_prism 0) (small_prism 1) (small_prism 2) = 112 := by
  sorry

end NUMINAMATH_CALUDE_shaded_volume_is_112_l2168_216825


namespace NUMINAMATH_CALUDE_coefficient_a4_value_l2168_216873

theorem coefficient_a4_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5) →
  a₄ = -5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a4_value_l2168_216873


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_n_binomial_sum_l2168_216894

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem exists_n_binomial_sum : 
  ∃ n : ℕ, Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_n_binomial_sum_l2168_216894


namespace NUMINAMATH_CALUDE_no_primes_in_sequence_l2168_216896

def is_arithmetic_progression (a b c : ℕ) : Prop := 2 * b = a + c

def is_geometric_progression (a b c : ℕ) : Prop := b * b = a * c

def is_valid_sequence (seq : ℕ → ℕ) : Prop :=
  (∀ n, seq n < seq (n + 1)) ∧
  (∀ n, is_arithmetic_progression (seq n) (seq (n + 1)) (seq (n + 2)) ∨
        is_geometric_progression (seq n) (seq (n + 1)) (seq (n + 2))) ∧
  (4 ∣ seq 0) ∧ (4 ∣ seq 1)

theorem no_primes_in_sequence (seq : ℕ → ℕ) (h : is_valid_sequence seq) :
  ∀ n, ¬ Nat.Prime (seq n) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_sequence_l2168_216896


namespace NUMINAMATH_CALUDE_scooter_cost_l2168_216870

/-- The cost of a scooter given the amount saved and the additional amount needed. -/
theorem scooter_cost (saved : ℕ) (needed : ℕ) (cost : ℕ) 
  (h1 : saved = 57) 
  (h2 : needed = 33) : 
  cost = saved + needed := by
  sorry

end NUMINAMATH_CALUDE_scooter_cost_l2168_216870


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l2168_216835

theorem bicycle_price_calculation (original_price : ℝ) 
  (initial_discount_rate : ℝ) (additional_discount : ℝ) (sales_tax_rate : ℝ) :
  original_price = 200 →
  initial_discount_rate = 0.25 →
  additional_discount = 10 →
  sales_tax_rate = 0.05 →
  (original_price * (1 - initial_discount_rate) - additional_discount) * (1 + sales_tax_rate) = 147 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l2168_216835


namespace NUMINAMATH_CALUDE_translated_circle_center_l2168_216865

/-- Given a point A(1,1) and a point P(m,n) on the circle centered at A, 
    if P is symmetric with P' with respect to the origin after translation,
    then the coordinates of A' are (1-2m, 1-2n) -/
theorem translated_circle_center (m n : ℝ) : 
  let A : ℝ × ℝ := (1, 1)
  let P : ℝ × ℝ := (m, n)
  let O : ℝ × ℝ := (0, 0)
  ∃ (A' : ℝ × ℝ), 
    (∃ (P' : ℝ × ℝ), P'.1 = -P.1 ∧ P'.2 = -P.2) →  -- P and P' are symmetric about origin
    (∃ (r : ℝ), (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2) →  -- P is on circle centered at A
    A' = (1 - 2*m, 1 - 2*n) :=
sorry

end NUMINAMATH_CALUDE_translated_circle_center_l2168_216865
