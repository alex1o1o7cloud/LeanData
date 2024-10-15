import Mathlib

namespace NUMINAMATH_CALUDE_store_shirts_sold_l500_50016

theorem store_shirts_sold (num_jeans : ℕ) (shirt_price : ℕ) (total_earnings : ℕ) :
  num_jeans = 10 ∧ 
  shirt_price = 10 ∧ 
  total_earnings = 400 →
  ∃ (num_shirts : ℕ), 
    num_shirts * shirt_price + num_jeans * (2 * shirt_price) = total_earnings ∧
    num_shirts = 20 :=
by sorry

end NUMINAMATH_CALUDE_store_shirts_sold_l500_50016


namespace NUMINAMATH_CALUDE_eight_valid_numbers_l500_50072

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A predicate that checks if a number is a positive perfect square -/
def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ m * m = n

/-- The main theorem stating that there are exactly 8 two-digit numbers satisfying the condition -/
theorem eight_valid_numbers :
  ∃! (s : Finset ℕ),
    Finset.card s = 8 ∧
    ∀ n ∈ s, is_two_digit n ∧
      is_positive_perfect_square (n - reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_eight_valid_numbers_l500_50072


namespace NUMINAMATH_CALUDE_smallest_AAB_l500_50071

/-- Represents a two-digit number --/
def TwoDigitNumber (a b : Nat) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- Represents a three-digit number --/
def ThreeDigitNumber (a b : Nat) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- The value of a two-digit number AB --/
def ValueAB (a b : Nat) : Nat :=
  10 * a + b

/-- The value of a three-digit number AAB --/
def ValueAAB (a b : Nat) : Nat :=
  100 * a + 10 * a + b

theorem smallest_AAB :
  ∀ a b : Nat,
    TwoDigitNumber a b →
    ThreeDigitNumber a b →
    a ≠ b →
    8 * (ValueAB a b) = ValueAAB a b →
    ∀ x y : Nat,
      TwoDigitNumber x y →
      ThreeDigitNumber x y →
      x ≠ y →
      8 * (ValueAB x y) = ValueAAB x y →
      ValueAAB a b ≤ ValueAAB x y →
    ValueAAB a b = 224 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_l500_50071


namespace NUMINAMATH_CALUDE_initial_distance_proof_l500_50010

/-- The initial distance between two cars on a main road --/
def initial_distance : ℝ := 165

/-- The total distance traveled by the first car --/
def car1_distance : ℝ := 65

/-- The distance traveled by the second car --/
def car2_distance : ℝ := 62

/-- The final distance between the two cars --/
def final_distance : ℝ := 38

/-- Theorem stating that the initial distance is correct given the problem conditions --/
theorem initial_distance_proof :
  initial_distance = car1_distance + car2_distance + final_distance :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_proof_l500_50010


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l500_50031

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 4) - y^2 / (k + 4) = 1

-- Theorem statement
theorem hyperbola_k_range (k : ℝ) :
  is_hyperbola k → k < -4 ∨ k > 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l500_50031


namespace NUMINAMATH_CALUDE_unique_number_with_digit_sum_product_l500_50055

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that 251 is the unique positive integer whose product
    with the sum of its digits equals 2008 -/
theorem unique_number_with_digit_sum_product : ∃! n : ℕ+, (n : ℕ) * sum_of_digits n = 2008 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_sum_product_l500_50055


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l500_50070

open Set Real

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_M_complement_N : M ∩ (𝒰 \ N) = Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l500_50070


namespace NUMINAMATH_CALUDE_sum_possible_side_lengths_is_330_l500_50047

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  EF : ℝ
  angleE : ℝ
  sidesArithmeticProgression : Bool
  EFisMaxLength : Bool
  EFparallelGH : Bool

/-- Calculates the sum of all possible values for the length of one of the other sides -/
def sumPossibleSideLengths (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the sum of all possible values for the length of one of the other sides is 330 -/
theorem sum_possible_side_lengths_is_330 (q : ConvexQuadrilateral) :
  q.EF = 24 ∧ q.angleE = 45 ∧ q.sidesArithmeticProgression ∧ q.EFisMaxLength ∧ q.EFparallelGH →
  sumPossibleSideLengths q = 330 :=
by sorry

end NUMINAMATH_CALUDE_sum_possible_side_lengths_is_330_l500_50047


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l500_50083

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l500_50083


namespace NUMINAMATH_CALUDE_cookie_count_l500_50046

theorem cookie_count (bundles_per_box : ℕ) (cookies_per_bundle : ℕ) (num_boxes : ℕ) : 
  bundles_per_box = 9 → cookies_per_bundle = 7 → num_boxes = 13 →
  bundles_per_box * cookies_per_bundle * num_boxes = 819 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l500_50046


namespace NUMINAMATH_CALUDE_intersection_line_slope_l500_50006

/-- Given two circles in the xy-plane, this theorem proves that the slope of the line 
    passing through their intersection points is -2/3. -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 12 = 0) →
  (x^2 + y^2 - 10*x - 2*y + 22 = 0) →
  ∃ (m : ℝ), m = -2/3 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 12 = 0) →
    (x₁^2 + y₁^2 - 10*x₁ - 2*y₁ + 22 = 0) →
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 12 = 0) →
    (x₂^2 + y₂^2 - 10*x₂ - 2*y₂ + 22 = 0) →
    x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = m :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l500_50006


namespace NUMINAMATH_CALUDE_molecular_weight_CaO_l500_50054

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- A compound with 1 Calcium atom and 1 Oxygen atom -/
structure CaO where
  ca : ℕ := 1
  o : ℕ := 1

/-- The molecular weight of a compound is the sum of the atomic weights of its constituent atoms -/
def molecular_weight (c : CaO) : ℝ := c.ca * atomic_weight_Ca + c.o * atomic_weight_O

theorem molecular_weight_CaO :
  molecular_weight { ca := 1, o := 1 : CaO } = 56.08 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_CaO_l500_50054


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l500_50068

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = -1

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 16 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ (x y : ℝ),
  hyperbola_equation x y →
  (∃ (a b : ℝ), ellipse_equation a b) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l500_50068


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l500_50045

theorem a_gt_b_necessary_not_sufficient (a b c : ℝ) :
  (∀ c ≠ 0, a * c^2 > b * c^2 → a > b) ∧
  (∃ c, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l500_50045


namespace NUMINAMATH_CALUDE_max_integer_solution_inequality_system_l500_50036

theorem max_integer_solution_inequality_system :
  ∀ x : ℤ, (3 * x - 1 < x + 1 ∧ 2 * (2 * x - 1) ≤ 5 * x + 1) →
  x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_solution_inequality_system_l500_50036


namespace NUMINAMATH_CALUDE_correct_calculation_l500_50084

theorem correct_calculation (x : ℝ) (h : x * 3 = 18) : x / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l500_50084


namespace NUMINAMATH_CALUDE_triangles_count_l500_50065

/-- The number of triangles that can be made from a wire -/
def triangles_from_wire (original_length : ℕ) (remaining_length : ℕ) (triangle_wire_length : ℕ) : ℕ :=
  (original_length - remaining_length) / triangle_wire_length

/-- Theorem: Given the specified wire lengths, 24 triangles can be made -/
theorem triangles_count : triangles_from_wire 84 12 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangles_count_l500_50065


namespace NUMINAMATH_CALUDE_least_bananas_l500_50095

def banana_distribution (total : ℕ) : Prop :=
  ∃ (b₁ b₂ b₃ b₄ : ℕ),
    -- Total number of bananas
    b₁ + b₂ + b₃ + b₄ = total ∧
    -- First monkey's distribution
    ∃ (x₁ y₁ z₁ w₁ : ℕ),
      2 * b₁ = 3 * x₁ ∧
      b₁ - x₁ = 3 * y₁ ∧ y₁ = z₁ ∧ y₁ = w₁ ∧
    -- Second monkey's distribution
    ∃ (x₂ y₂ z₂ w₂ : ℕ),
      b₂ = 3 * y₂ ∧
      2 * b₂ = 3 * (x₂ + z₂ + w₂) ∧ x₂ = z₂ ∧ x₂ = w₂ ∧
    -- Third monkey's distribution
    ∃ (x₃ y₃ z₃ w₃ : ℕ),
      b₃ = 4 * z₃ ∧
      3 * b₃ = 4 * (x₃ + y₃ + w₃) ∧ x₃ = y₃ ∧ x₃ = w₃ ∧
    -- Fourth monkey's distribution
    ∃ (x₄ y₄ z₄ w₄ : ℕ),
      b₄ = 6 * w₄ ∧
      5 * b₄ = 6 * (x₄ + y₄ + z₄) ∧ x₄ = y₄ ∧ x₄ = z₄ ∧
    -- Final distribution ratio
    ∃ (k : ℕ),
      (2 * x₁ + y₂ + z₃ + w₄) = 4 * k ∧
      (y₁ + 2 * y₂ + z₃ + w₄) = 3 * k ∧
      (z₁ + y₂ + 2 * z₃ + w₄) = 2 * k ∧
      (w₁ + y₂ + z₃ + 2 * w₄) = k

theorem least_bananas : 
  ∀ n : ℕ, n < 1128 → ¬(banana_distribution n) ∧ banana_distribution 1128 := by
  sorry

end NUMINAMATH_CALUDE_least_bananas_l500_50095


namespace NUMINAMATH_CALUDE_average_rate_of_change_l500_50096

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the theorem
theorem average_rate_of_change (Δx : ℝ) :
  (f (1 + Δx) - f 1) / Δx = 2 + Δx :=
by sorry

end NUMINAMATH_CALUDE_average_rate_of_change_l500_50096


namespace NUMINAMATH_CALUDE_class_average_problem_l500_50013

theorem class_average_problem (x : ℝ) : 
  0.15 * x + 0.50 * 78 + 0.35 * 63 = 76.05 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l500_50013


namespace NUMINAMATH_CALUDE_isabella_hair_growth_l500_50080

def current_hair_length : ℕ := 18
def hair_growth : ℕ := 4

theorem isabella_hair_growth :
  current_hair_length + hair_growth = 22 :=
by sorry

end NUMINAMATH_CALUDE_isabella_hair_growth_l500_50080


namespace NUMINAMATH_CALUDE_inequality_proof_l500_50056

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  x^2 + y^2 + z^2 + (Real.sqrt 3 / 2) * Real.sqrt (x * y * z) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l500_50056


namespace NUMINAMATH_CALUDE_equal_squares_with_difference_one_l500_50063

theorem equal_squares_with_difference_one :
  ∃ (a b : ℝ), a = b + 1 ∧ a^2 = b^2 :=
by sorry

end NUMINAMATH_CALUDE_equal_squares_with_difference_one_l500_50063


namespace NUMINAMATH_CALUDE_zero_in_interval_l500_50048

theorem zero_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b) (hb' : b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ a^x + x - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l500_50048


namespace NUMINAMATH_CALUDE_area_outside_rectangle_within_square_l500_50022

/-- The area of the region outside a centered rectangle but within a square. -/
theorem area_outside_rectangle_within_square : 
  ∀ (square_side rectangle_length rectangle_width : ℝ),
    square_side = 10 →
    rectangle_length = 5 →
    rectangle_width = 2 →
    square_side > rectangle_length ∧ square_side > rectangle_width →
    square_side^2 - rectangle_length * rectangle_width = 90 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_rectangle_within_square_l500_50022


namespace NUMINAMATH_CALUDE_mary_balloon_count_l500_50008

/-- The number of black balloons Nancy has -/
def nancy_balloons : ℕ := 7

/-- The factor by which Mary's balloons exceed Nancy's -/
def mary_factor : ℕ := 4

/-- The number of black balloons Mary has -/
def mary_balloons : ℕ := nancy_balloons * mary_factor

theorem mary_balloon_count : mary_balloons = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_balloon_count_l500_50008


namespace NUMINAMATH_CALUDE_wedge_volume_l500_50064

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : θ = 60) :
  let r := d / 2
  let cylinder_volume := π * r^2 * d
  let wedge_volume := cylinder_volume * θ / 360
  d = 16 → wedge_volume = 341 * π :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_l500_50064


namespace NUMINAMATH_CALUDE_root_ordering_implies_a_range_l500_50005

/-- Given two quadratic equations and an ordering of their roots, 
    prove the range of the coefficient a. -/
theorem root_ordering_implies_a_range 
  (a b : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + 1 = 0)
  (h₂ : a * x₂^2 + b * x₂ + 1 = 0)
  (h₃ : a^2 * x₃^2 + b * x₃ + 1 = 0)
  (h₄ : a^2 * x₄^2 + b * x₄ + 1 = 0)
  (h_order : x₃ < x₁ ∧ x₁ < x₂ ∧ x₂ < x₄) : 
  0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_root_ordering_implies_a_range_l500_50005


namespace NUMINAMATH_CALUDE_divisor_problem_l500_50039

theorem divisor_problem (n m : ℕ) (h1 : n = 3830) (h2 : m = 5) : 
  (∃ d : ℕ, d > 0 ∧ (n - m) % d = 0 ∧ 
   ∀ k < m, ¬((n - k) % d = 0)) → 
  (n - m) % 15 = 0 ∧ 15 > 0 ∧ 
  ∀ k < m, ¬((n - k) % 15 = 0) :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l500_50039


namespace NUMINAMATH_CALUDE_temperature_difference_l500_50057

theorem temperature_difference (t1 t2 k1 k2 : ℚ) :
  t1 = 5 / 9 * (k1 - 32) →
  t2 = 5 / 9 * (k2 - 32) →
  t1 = 105 →
  t2 = 80 →
  k1 - k2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l500_50057


namespace NUMINAMATH_CALUDE_local_extremum_and_minimum_l500_50009

-- Define the function f
def f (a b x : ℝ) : ℝ := a^2 * x^3 + 3 * a * x^2 - b * x - 1

-- State the theorem
theorem local_extremum_and_minimum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) ∧
  (f a b 1 = 0) ∧
  (∀ x ≥ 0, f a b x ≥ -1) →
  a = -1/2 ∧ b = -9/4 ∧ ∀ x ≥ 0, f a b x ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_local_extremum_and_minimum_l500_50009


namespace NUMINAMATH_CALUDE_married_student_percentage_l500_50081

theorem married_student_percentage
  (total : ℝ)
  (total_positive : total > 0)
  (male_percentage : ℝ)
  (male_percentage_def : male_percentage = 0.7)
  (married_male_fraction : ℝ)
  (married_male_fraction_def : married_male_fraction = 1 / 7)
  (single_female_fraction : ℝ)
  (single_female_fraction_def : single_female_fraction = 1 / 3) :
  (male_percentage * married_male_fraction * total +
   (1 - male_percentage) * (1 - single_female_fraction) * total) / total = 0.3 := by
sorry

end NUMINAMATH_CALUDE_married_student_percentage_l500_50081


namespace NUMINAMATH_CALUDE_original_fraction_l500_50051

theorem original_fraction (x y : ℚ) : 
  (1.15 * x) / (0.92 * y) = 15 / 16 → x / y = 69 / 92 := by
sorry

end NUMINAMATH_CALUDE_original_fraction_l500_50051


namespace NUMINAMATH_CALUDE_bob_apples_correct_l500_50050

/-- The number of apples Bob and Carla share -/
def total_apples : ℕ := 30

/-- Represents the number of apples Bob eats -/
def bob_apples : ℕ := 10

/-- Carla eats twice as many apples as Bob -/
def carla_apples (b : ℕ) : ℕ := 2 * b

theorem bob_apples_correct :
  bob_apples + carla_apples bob_apples = total_apples := by sorry

end NUMINAMATH_CALUDE_bob_apples_correct_l500_50050


namespace NUMINAMATH_CALUDE_pipe_filling_time_l500_50027

theorem pipe_filling_time (fill_rate : ℝ → ℝ → ℝ) (time : ℝ → ℝ → ℝ) :
  (fill_rate 3 8 = 1) →
  (∀ n t, fill_rate n t * t = 1) →
  (time 2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l500_50027


namespace NUMINAMATH_CALUDE_factorization_equality_l500_50079

-- Define the equality we want to prove
theorem factorization_equality (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l500_50079


namespace NUMINAMATH_CALUDE_max_grid_mean_l500_50053

def Grid := Fin 3 → Fin 3 → ℕ

def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 9) ∧
  (∀ n, n ∈ Finset.range 9 → ∃ i j, g i j = n)

def circle_mean (g : Grid) (i j : Fin 2) : ℚ :=
  (g i j + g i (j+1) + g (i+1) j + g (i+1) (j+1)) / 4

def grid_mean (g : Grid) : ℚ :=
  (circle_mean g 0 0 + circle_mean g 0 1 + circle_mean g 1 0 + circle_mean g 1 1) / 4

theorem max_grid_mean :
  ∀ g : Grid, valid_grid g → grid_mean g ≤ 5.8125 :=
sorry

end NUMINAMATH_CALUDE_max_grid_mean_l500_50053


namespace NUMINAMATH_CALUDE_max_z_value_l500_50090

theorem max_z_value : 
  (∃ (z : ℝ), ∀ (w : ℝ), 
    (∃ (x y : ℝ), 4*x^2 + 4*y^2 + z^2 + x*y + y*z + x*z = 8) → 
    (∃ (x y : ℝ), 4*x^2 + 4*y^2 + w^2 + x*y + y*w + x*w = 8) → 
    w ≤ z) ∧ 
  (∃ (x y : ℝ), 4*x^2 + 4*y^2 + 3^2 + x*y + y*3 + x*3 = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_z_value_l500_50090


namespace NUMINAMATH_CALUDE_min_area_AOB_l500_50007

noncomputable section

-- Define the hyperbola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (2 * a^2) = 1 ∧ a > 0

-- Define the parabola C₂
def C₂ (a : ℝ) (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 3 * a * x

-- Define the focus F₁
def F₁ (a : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * a, 0)

-- Define a chord AB of C₂ passing through F₁
def chord_AB (a k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + Real.sqrt 3 * a) ∧ C₂ a x y

-- Define the area of triangle AOB
def area_AOB (a k : ℝ) : ℝ := 6 * a^2 * Real.sqrt (1 + 1 / k^2)

-- Main theorem
theorem min_area_AOB (a : ℝ) :
  (∃ k : ℝ, ∀ k' : ℝ, area_AOB a k ≤ area_AOB a k') ∧
  (∃ x : ℝ, x = -Real.sqrt 3 * a ∧ 
    ∀ k : ℝ, area_AOB a k ≥ 6 * a^2) :=
sorry

end

end NUMINAMATH_CALUDE_min_area_AOB_l500_50007


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_60_l500_50044

theorem product_of_five_consecutive_integers_divisible_by_60 (n : ℤ) : 
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_60_l500_50044


namespace NUMINAMATH_CALUDE_unknown_number_in_average_l500_50020

theorem unknown_number_in_average (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 50 + x) / 3 + 5 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_in_average_l500_50020


namespace NUMINAMATH_CALUDE_equation_solution_l500_50075

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4/3 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l500_50075


namespace NUMINAMATH_CALUDE_imoProof_l500_50082

theorem imoProof (a b : ℕ) (ha : a = 18) (hb : b = 1) : 
  ¬ (7 ∣ (a * b * (a + b))) ∧ 
  (7^7 ∣ ((a + b)^7 - a^7 - b^7)) := by
sorry

end NUMINAMATH_CALUDE_imoProof_l500_50082


namespace NUMINAMATH_CALUDE_total_tv_time_l500_50034

theorem total_tv_time : 
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  (reality_shows.sum + cartoons.sum + ad_breaks.sum) = 219 := by
  sorry

end NUMINAMATH_CALUDE_total_tv_time_l500_50034


namespace NUMINAMATH_CALUDE_fence_perimeter_is_200_l500_50032

/-- A square field enclosed by evenly spaced triangular posts -/
structure FenceSetup where
  total_posts : ℕ
  post_width : ℝ
  gap_width : ℝ

/-- Calculate the outer perimeter of the fence setup -/
def outer_perimeter (f : FenceSetup) : ℝ :=
  let posts_per_side := f.total_posts / 4
  let gaps_per_side := posts_per_side - 1
  let side_length := posts_per_side * f.post_width + gaps_per_side * f.gap_width
  4 * side_length

/-- Theorem: The outer perimeter of the given fence setup is 200 feet -/
theorem fence_perimeter_is_200 : 
  outer_perimeter ⟨36, 2, 4⟩ = 200 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_200_l500_50032


namespace NUMINAMATH_CALUDE_whale_length_in_crossing_scenario_l500_50029

/-- The length of a whale in a crossing scenario --/
theorem whale_length_in_crossing_scenario
  (v_fast : ℝ)  -- Initial speed of the faster whale
  (v_slow : ℝ)  -- Initial speed of the slower whale
  (a_fast : ℝ)  -- Acceleration of the faster whale
  (a_slow : ℝ)  -- Acceleration of the slower whale
  (t : ℝ)       -- Time taken for the faster whale to cross the slower whale
  (h_v_fast : v_fast = 18)
  (h_v_slow : v_slow = 15)
  (h_a_fast : a_fast = 1)
  (h_a_slow : a_slow = 0.5)
  (h_t : t = 15) :
  let d_fast := v_fast * t + (1/2) * a_fast * t^2
  let d_slow := v_slow * t + (1/2) * a_slow * t^2
  d_fast - d_slow = 101.25 := by
sorry


end NUMINAMATH_CALUDE_whale_length_in_crossing_scenario_l500_50029


namespace NUMINAMATH_CALUDE_ratio_of_two_numbers_l500_50052

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = 44) (h4 : a - b = 20) : a / b = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_two_numbers_l500_50052


namespace NUMINAMATH_CALUDE_right_angle_constraint_l500_50002

/-- Given two points A and B on the x-axis, and a point P on a line,
    prove that if ∠APB is a right angle, then the distance between A and B
    is at least 10 units. -/
theorem right_angle_constraint (m : ℝ) (h_m : m > 0) :
  (∃ (x y : ℝ), 3 * x + 4 * y + 25 = 0 ∧
    ((x + m) * (x - m) + y * y = 0)) →
  m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_right_angle_constraint_l500_50002


namespace NUMINAMATH_CALUDE_bernard_luke_age_problem_l500_50011

/-- Given that in 8 years, Mr. Bernard will be 3 times as old as Luke is now,
    prove that 10 years less than their average current age is 2 * L - 14,
    where L is Luke's current age. -/
theorem bernard_luke_age_problem (L : ℕ) : 
  (L + ((3 * L) - 8)) / 2 - 10 = 2 * L - 14 := by
  sorry

end NUMINAMATH_CALUDE_bernard_luke_age_problem_l500_50011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l500_50003

/-- Sum of first n terms of an arithmetic sequence -/
def T (b : ℚ) (n : ℕ) : ℚ := n * (2 * b + (n - 1) * 5) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term (b : ℚ) :
  (∃ k : ℚ, ∀ n : ℕ, n > 0 → T b (4 * n) / T b n = k) →
  b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l500_50003


namespace NUMINAMATH_CALUDE_centroid_of_equal_areas_l500_50015

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Check if a point is inside a triangle -/
def isInside (M : Point) (T : Triangle) : Prop :=
  sorry

/-- Calculate the area of a triangle -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Check if three triangles have equal areas -/
def equalAreas (T1 T2 T3 : Triangle) : Prop :=
  triangleArea T1.A T1.B T1.C = triangleArea T2.A T2.B T2.C ∧
  triangleArea T2.A T2.B T2.C = triangleArea T3.A T3.B T3.C

/-- Check if a point is the centroid of a triangle -/
def isCentroid (M : Point) (T : Triangle) : Prop :=
  sorry

theorem centroid_of_equal_areas (ABC : Triangle) (M : Point) 
  (h1 : isInside M ABC)
  (h2 : equalAreas (Triangle.mk M ABC.A ABC.B) (Triangle.mk M ABC.A ABC.C) (Triangle.mk M ABC.B ABC.C)) :
  isCentroid M ABC :=
sorry

end NUMINAMATH_CALUDE_centroid_of_equal_areas_l500_50015


namespace NUMINAMATH_CALUDE_min_value_and_range_l500_50099

theorem min_value_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + 3*b^2 = 3) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m) ∧ 
             m = 4) ∧
  (∀ x : ℝ, (2 * |x - 1| + |x| ≥ Real.sqrt 5 * a + b) ↔ (x ≤ -2/3 ∨ x ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_range_l500_50099


namespace NUMINAMATH_CALUDE_doll_ratio_l500_50024

/-- The ratio of Dina's dolls to Ivy's dolls is 2:1 -/
theorem doll_ratio : 
  ∀ (ivy_dolls : ℕ) (dina_dolls : ℕ),
  (2 : ℚ) / 3 * ivy_dolls = 20 →
  dina_dolls = 60 →
  (dina_dolls : ℚ) / ivy_dolls = 2 := by
  sorry

end NUMINAMATH_CALUDE_doll_ratio_l500_50024


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l500_50097

/-- The set A defined by the given condition -/
def A (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a^2 + 1 }

/-- The set B defined by the given condition -/
def B (a : ℝ) : Set ℝ := { x | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0 }

/-- Theorem stating the range of values for a where A is a subset of B -/
theorem range_of_a_for_subset (a : ℝ) : A a ⊆ B a ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l500_50097


namespace NUMINAMATH_CALUDE_angle_at_seven_l500_50023

/-- The number of parts the clock face is divided into -/
def clock_parts : ℕ := 12

/-- The angle of each part of the clock face in degrees -/
def part_angle : ℝ := 30

/-- The time in hours -/
def time : ℝ := 7

/-- The angle between the hour hand and the minute hand at a given time -/
def angle_between (t : ℝ) : ℝ := sorry

theorem angle_at_seven : angle_between time = 150 := by sorry

end NUMINAMATH_CALUDE_angle_at_seven_l500_50023


namespace NUMINAMATH_CALUDE_a_minus_b_value_l500_50033

theorem a_minus_b_value (a b : ℝ) 
  (ha : |a| = 4)
  (hb : |b| = 2)
  (hab : |a + b| = -(a + b)) :
  a - b = -2 ∨ a - b = -6 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l500_50033


namespace NUMINAMATH_CALUDE_odd_squares_difference_is_perfect_square_l500_50067

theorem odd_squares_difference_is_perfect_square (m n : ℤ) 
  (h_m_odd : Odd m) (h_n_odd : Odd n) 
  (h_divisible : ∃ k : ℤ, n^2 - 1 = k * (m^2 + 1 - n^2)) :
  ∃ k : ℤ, |m^2 + 1 - n^2| = k^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_squares_difference_is_perfect_square_l500_50067


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l500_50025

theorem product_purely_imaginary (x : ℝ) : 
  (∃ y : ℝ, (x + 2*I) * ((x + 1) + 3*I) * ((x + 2) + 4*I) = y*I) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_purely_imaginary_l500_50025


namespace NUMINAMATH_CALUDE_max_victory_margin_l500_50000

/-- Represents the vote count for a candidate in a specific time period -/
structure VoteCount where
  first_two_hours : ℕ
  last_two_hours : ℕ

/-- Represents the election results -/
structure ElectionResult where
  petya : VoteCount
  vasya : VoteCount

def total_votes (result : ElectionResult) : ℕ :=
  result.petya.first_two_hours + result.petya.last_two_hours +
  result.vasya.first_two_hours + result.vasya.last_two_hours

def petya_total (result : ElectionResult) : ℕ :=
  result.petya.first_two_hours + result.petya.last_two_hours

def vasya_total (result : ElectionResult) : ℕ :=
  result.vasya.first_two_hours + result.vasya.last_two_hours

def is_valid_result (result : ElectionResult) : Prop :=
  total_votes result = 27 ∧
  result.petya.first_two_hours = result.vasya.first_two_hours + 9 ∧
  result.vasya.last_two_hours = result.petya.last_two_hours + 9 ∧
  petya_total result > vasya_total result

def victory_margin (result : ElectionResult) : ℕ :=
  petya_total result - vasya_total result

theorem max_victory_margin :
  ∀ result : ElectionResult,
    is_valid_result result →
    victory_margin result ≤ 9 :=
by
  sorry

#check max_victory_margin

end NUMINAMATH_CALUDE_max_victory_margin_l500_50000


namespace NUMINAMATH_CALUDE_complex_number_problem_l500_50098

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * (1 + Complex.I) = 1 - Complex.I) →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 →
  z₂ = 4 + 2 * Complex.I ∧ Complex.abs z₂ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l500_50098


namespace NUMINAMATH_CALUDE_negation_of_existence_l500_50089

theorem negation_of_existence (m : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + m*x₀ - 2 > 0) ↔
  (∀ x : ℝ, x > 0 → x^2 + m*x - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l500_50089


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l500_50085

theorem opposite_of_negative_2023 : -(Int.neg 2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l500_50085


namespace NUMINAMATH_CALUDE_ratio_solution_l500_50094

theorem ratio_solution (x y z a : ℤ) : 
  (∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) → 
  y = 24 * a - 12 → 
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_solution_l500_50094


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_14_mod_21_l500_50076

theorem largest_four_digit_congruent_to_14_mod_21 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n ≡ 14 [MOD 21] → n ≤ 9979 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_14_mod_21_l500_50076


namespace NUMINAMATH_CALUDE_water_balloon_puddle_depth_l500_50078

/-- The depth of water in a cylindrical puddle formed from a burst spherical water balloon -/
theorem water_balloon_puddle_depth (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 → 
  r_cylinder = 12 → 
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h → 
  h = 1 / 4 := by
  sorry

#check water_balloon_puddle_depth

end NUMINAMATH_CALUDE_water_balloon_puddle_depth_l500_50078


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l500_50042

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem quadratic_symmetry 
  (a b c x₁ x₂ x₃ x₄ x₅ p q : ℝ) 
  (ha : a ≠ 0)
  (hx : x₁ ≠ x₂ + x₃ + x₄ + x₅)
  (hf₁ : f a b c x₁ = 5)
  (hf₂ : f a b c (x₂ + x₃ + x₄ + x₅) = 5)
  (hp : f a b c (x₁ + x₂) = p)
  (hq : f a b c (x₃ + x₄ + x₅) = q) :
  p - q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l500_50042


namespace NUMINAMATH_CALUDE_swim_trunks_price_l500_50066

def flat_rate_shipping : ℝ := 5.00
def shipping_threshold : ℝ := 50.00
def shipping_rate : ℝ := 0.20
def shirt_price : ℝ := 12.00
def shirt_quantity : ℕ := 3
def socks_price : ℝ := 5.00
def shorts_price : ℝ := 15.00
def shorts_quantity : ℕ := 2
def total_bill : ℝ := 102.00

def known_items_cost : ℝ := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity

theorem swim_trunks_price (x : ℝ) : 
  (known_items_cost + x + shipping_rate * (known_items_cost + x) = total_bill) → 
  x = 14.00 := by
  sorry

end NUMINAMATH_CALUDE_swim_trunks_price_l500_50066


namespace NUMINAMATH_CALUDE_min_value_expression_l500_50001

theorem min_value_expression (a b : ℝ) (hb : b ≠ 0) :
  a^2 + b^2 + a/b + 1/b^2 ≥ Real.sqrt 3 ∧
  ∃ (a₀ b₀ : ℝ) (hb₀ : b₀ ≠ 0), a₀^2 + b₀^2 + a₀/b₀ + 1/b₀^2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l500_50001


namespace NUMINAMATH_CALUDE_product_inspection_l500_50062

def total_products : ℕ := 100
def non_defective : ℕ := 98
def defective : ℕ := 2
def selected : ℕ := 3

theorem product_inspection :
  (Nat.choose total_products selected = 161700) ∧
  (Nat.choose defective 1 * Nat.choose non_defective (selected - 1) = 9506) ∧
  (Nat.choose total_products selected - Nat.choose non_defective selected = 9604) ∧
  (Nat.choose defective 1 * Nat.choose non_defective (selected - 1) * Nat.factorial selected = 57036) :=
by sorry

end NUMINAMATH_CALUDE_product_inspection_l500_50062


namespace NUMINAMATH_CALUDE_total_fireworks_count_l500_50037

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) +
  cherie_boxes * (cherie_sparklers + cherie_whistlers)

theorem total_fireworks_count : total_fireworks = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_fireworks_count_l500_50037


namespace NUMINAMATH_CALUDE_hyperbola_properties_l500_50012

/-- Hyperbola C with equation x^2 - 4y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 - 4*p.2^2 = 1}

/-- The asymptotes of hyperbola C -/
def asymptotes : Set (ℝ × ℝ) := {p | p.1 + 2*p.2 = 0 ∨ p.1 - 2*p.2 = 0}

/-- The imaginary axis length of hyperbola C -/
def imaginary_axis_length : ℝ := 1

/-- Theorem: The asymptotes and imaginary axis length of hyperbola C -/
theorem hyperbola_properties :
  (∀ p ∈ C, p ∈ asymptotes ↔ p.1^2 = 4*p.2^2) ∧
  imaginary_axis_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l500_50012


namespace NUMINAMATH_CALUDE_rectangle_area_l500_50061

/-- 
Given a rectangle with length l and width w, 
if the length is four times the width and the perimeter is 200,
then the area of the rectangle is 1600.
-/
theorem rectangle_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l500_50061


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_bound_l500_50030

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem prime_arithmetic_sequence_bound
  (a : ℕ → ℕ)
  (d : ℕ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_prime : ∀ n : ℕ, is_prime (a n))
  (h_d : d < 2000) :
  ∀ n : ℕ, n > 11 → ¬(is_prime (a n)) :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_bound_l500_50030


namespace NUMINAMATH_CALUDE_wheel_probability_l500_50058

theorem wheel_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 2/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/10 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l500_50058


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l500_50038

theorem quadratic_inequality_condition (a : ℝ) :
  (a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  (∃ a : ℝ, a < 0 ∧ ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l500_50038


namespace NUMINAMATH_CALUDE_john_used_one_nickel_l500_50069

/-- Calculates the number of nickels used in a purchase, given the number of quarters and dimes used, the cost of the item, and the change received. -/
def nickels_used (quarters : ℕ) (dimes : ℕ) (cost : ℕ) (change : ℕ) : ℕ :=
  let quarter_value := 25
  let dime_value := 10
  let nickel_value := 5
  let total_paid := cost + change
  let paid_without_nickels := quarters * quarter_value + dimes * dime_value
  (total_paid - paid_without_nickels) / nickel_value

theorem john_used_one_nickel :
  nickels_used 4 3 131 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_john_used_one_nickel_l500_50069


namespace NUMINAMATH_CALUDE_ribbon_length_reduction_l500_50060

theorem ribbon_length_reduction (original_length new_length : ℝ) : 
  (11 : ℝ) / 7 = original_length / new_length →
  new_length = 35 →
  original_length = 55 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_length_reduction_l500_50060


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l500_50040

theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 3*x + k - 2
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (k ≤ 17/4 ∧
   (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → (x₁ - 1)*(x₂ - 1) = -1 → k = -3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l500_50040


namespace NUMINAMATH_CALUDE_product_polynomials_l500_50017

theorem product_polynomials (g h : ℚ) :
  (∀ d : ℚ, (7*d^2 - 3*d + g) * (3*d^2 + h*d - 8) = 21*d^4 - 44*d^3 - 35*d^2 + 14*d - 16) →
  g + h = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_polynomials_l500_50017


namespace NUMINAMATH_CALUDE_residue_of_8_1234_mod_13_l500_50028

theorem residue_of_8_1234_mod_13 : (8 : ℤ)^1234 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_8_1234_mod_13_l500_50028


namespace NUMINAMATH_CALUDE_prob_at_least_one_to_museum_l500_50014

/-- The probability that at least one of two independent events occurs -/
def prob_at_least_one (p₁ p₂ : ℝ) : ℝ := 1 - (1 - p₁) * (1 - p₂)

/-- The probability that at least one of two people goes to the museum -/
theorem prob_at_least_one_to_museum (p_a p_b : ℝ) 
  (h_a : p_a = 0.8) 
  (h_b : p_b = 0.7) : 
  prob_at_least_one p_a p_b = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_to_museum_l500_50014


namespace NUMINAMATH_CALUDE_average_distance_and_monthly_expense_l500_50004

/-- Represents the daily distance traveled relative to the standard distance -/
def daily_distances : List ℤ := [-8, -11, -14, 0, -16, 41, 8]

/-- The standard distance in kilometers -/
def standard_distance : ℕ := 50

/-- Gasoline consumption in liters per 100 km -/
def gasoline_consumption : ℚ := 6 / 100

/-- Gasoline price in yuan per liter -/
def gasoline_price : ℚ := 77 / 10

/-- Number of days in a month -/
def days_in_month : ℕ := 30

theorem average_distance_and_monthly_expense :
  let avg_distance := standard_distance + (daily_distances.sum / daily_distances.length : ℚ)
  let monthly_expense := (days_in_month : ℚ) * avg_distance * gasoline_consumption * gasoline_price
  avg_distance = standard_distance ∧ monthly_expense = 693 := by
  sorry

end NUMINAMATH_CALUDE_average_distance_and_monthly_expense_l500_50004


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l500_50088

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 2*x - 3 < 0) → (-2 < x ∧ x < 3) ∧
  ∃ y : ℝ, -2 < y ∧ y < 3 ∧ ¬(y^2 - 2*y - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l500_50088


namespace NUMINAMATH_CALUDE_lending_period_equation_l500_50073

/-- Represents the lending period in years -/
def t : ℝ := sorry

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℝ := 3900

/-- The amount Manoj lent to Ramu -/
def lent_amount : ℝ := 5655

/-- The interest rate Anwar charged Manoj (in percentage) -/
def borrowing_rate : ℝ := 6

/-- The interest rate Manoj charged Ramu (in percentage) -/
def lending_rate : ℝ := 9

/-- Manoj's gain from the whole transaction -/
def gain : ℝ := 824.85

/-- Theorem stating the relationship between the lending period and the financial parameters -/
theorem lending_period_equation : 
  gain = (lent_amount * lending_rate * t / 100) - (borrowed_amount * borrowing_rate * t / 100) := by
  sorry

end NUMINAMATH_CALUDE_lending_period_equation_l500_50073


namespace NUMINAMATH_CALUDE_product_digit_sum_l500_50019

def first_number : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def second_number : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

theorem product_digit_sum : 
  let product := first_number * second_number
  let thousands_digit := (product / 1000) % 10
  let units_digit := product % 10
  thousands_digit + units_digit = 13 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l500_50019


namespace NUMINAMATH_CALUDE_play_seating_l500_50035

/-- The number of chairs put out for a play, given the number of rows and chairs per row -/
def total_chairs (rows : ℕ) (chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

/-- Theorem stating that 27 rows of 16 chairs each results in 432 chairs total -/
theorem play_seating : total_chairs 27 16 = 432 := by
  sorry

end NUMINAMATH_CALUDE_play_seating_l500_50035


namespace NUMINAMATH_CALUDE_total_marks_calculation_l500_50086

/-- Given 50 candidates in an examination with an average mark of 40,
    prove that the total marks is 2000. -/
theorem total_marks_calculation (num_candidates : ℕ) (average_mark : ℚ) :
  num_candidates = 50 →
  average_mark = 40 →
  (num_candidates : ℚ) * average_mark = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l500_50086


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l500_50059

theorem triangle_with_angle_ratio_1_2_3_is_right_angled (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l500_50059


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l500_50091

/-- A geometric sequence with a_3 = 1 and a_7 = 9 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  a 3 = 1 ∧ 
  a 7 = 9

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l500_50091


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l500_50041

theorem isosceles_triangle_perimeter : 
  ∀ x : ℝ, 
  x^2 - 8*x + 15 = 0 → 
  x > 0 →
  2*x + 7 > x →
  2*x + 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l500_50041


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l500_50077

/-- Given three real numbers a, b, and c satisfying certain conditions,
    prove that the average of a and b is 35. -/
theorem average_of_a_and_b (a b c : ℝ) 
    (h1 : (a + b) / 2 = 35)
    (h2 : (b + c) / 2 = 80)
    (h3 : c - a = 90) : 
  (a + b) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l500_50077


namespace NUMINAMATH_CALUDE_stick_cutting_l500_50092

theorem stick_cutting (short_length long_length : ℝ) : 
  long_length = short_length + 18 →
  short_length + long_length = 30 →
  long_length / short_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_stick_cutting_l500_50092


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l500_50074

theorem right_angled_triangle_set :
  ∀ (a b c : ℝ),
  (a = 3 ∧ b = 4 ∧ c = 5) →
  a^2 + b^2 = c^2 ∧
  ¬(1^2 + 2^2 = 3^2) ∧
  ¬(5^2 + 12^2 = 14^2) ∧
  ¬((Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2) :=
by
  sorry

#check right_angled_triangle_set

end NUMINAMATH_CALUDE_right_angled_triangle_set_l500_50074


namespace NUMINAMATH_CALUDE_geometric_series_sum_first_5_terms_l500_50087

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_first_5_terms :
  let a : ℚ := 2
  let r : ℚ := 1/4
  let n : ℕ := 5
  geometric_series_sum a r n = 341/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_first_5_terms_l500_50087


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l500_50018

/-- A triangle with angles satisfying specific ratios is right-angled -/
theorem triangle_is_right_angled (angle1 angle2 angle3 : ℝ) : 
  angle1 + angle2 + angle3 = 180 →
  angle1 = 3 * angle2 →
  angle3 = 2 * angle2 →
  angle1 = 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l500_50018


namespace NUMINAMATH_CALUDE_largest_expression_l500_50021

theorem largest_expression (a₁ a₂ b₁ b₂ : ℝ) 
  (ha : 0 < a₁ ∧ a₁ < a₂) 
  (hb : 0 < b₁ ∧ b₁ < b₂) 
  (ha_sum : a₁ + a₂ = 1) 
  (hb_sum : b₁ + b₂ = 1) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * a₂ + b₁ * b₂ ∧ 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l500_50021


namespace NUMINAMATH_CALUDE_part1_part2_part3_l500_50049

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ :=
  if f a x ≥ f' a x then f' a x else f a x

-- Part 1: Condition for f(x) ≤ f'(x) when x ∈ [-2, -1]
theorem part1 (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f a x ≤ f' a x) → a ≥ 3/2 :=
sorry

-- Part 2: Solutions to f(x) = |f'(x)|
theorem part2 (a : ℝ) (x : ℝ) :
  f a x = |f' a x| →
  ((a < -1 ∧ (x = -1 ∨ x = 1 - 2*a)) ∨
   (-1 ≤ a ∧ a ≤ 1 ∧ (x = 1 ∨ x = -1 ∨ x = 1 - 2*a ∨ x = -(1 + 2*a))) ∨
   (a > 1 ∧ (x = 1 ∨ x = -(1 + 2*a)))) :=
sorry

-- Part 3: Minimum value of g(x) for x ∈ [2, 4]
theorem part3 (a : ℝ) :
  (∃ m : ℝ, ∀ x ∈ Set.Icc 2 4, g a x ≥ m) ∧
  (a ≤ -4 → ∃ x ∈ Set.Icc 2 4, g a x = 8*a + 17) ∧
  (-4 < a ∧ a < -2 → ∃ x ∈ Set.Icc 2 4, g a x = 1 - a^2) ∧
  (-2 ≤ a ∧ a < -1/2 → ∃ x ∈ Set.Icc 2 4, g a x = 4*a + 5) ∧
  (a ≥ -1/2 → ∃ x ∈ Set.Icc 2 4, g a x = 2*a + 4) :=
sorry

end

end NUMINAMATH_CALUDE_part1_part2_part3_l500_50049


namespace NUMINAMATH_CALUDE_art_exhibition_tickets_l500_50093

theorem art_exhibition_tickets (advanced_price door_price total_tickets total_revenue : ℕ) 
  (h1 : advanced_price = 8)
  (h2 : door_price = 14)
  (h3 : total_tickets = 140)
  (h4 : total_revenue = 1720) :
  ∃ (advanced_tickets : ℕ),
    advanced_tickets * advanced_price + (total_tickets - advanced_tickets) * door_price = total_revenue ∧
    advanced_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_art_exhibition_tickets_l500_50093


namespace NUMINAMATH_CALUDE_abs_lt_one_iff_square_lt_one_l500_50026

theorem abs_lt_one_iff_square_lt_one (x : ℝ) : |x| < 1 ↔ x^2 < 1 := by sorry

end NUMINAMATH_CALUDE_abs_lt_one_iff_square_lt_one_l500_50026


namespace NUMINAMATH_CALUDE_geometric_sequence_consecutive_terms_l500_50043

theorem geometric_sequence_consecutive_terms (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → 
  x = 1 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_consecutive_terms_l500_50043
