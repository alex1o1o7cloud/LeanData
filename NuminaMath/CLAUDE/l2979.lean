import Mathlib

namespace NUMINAMATH_CALUDE_max_m_plus_2n_l2979_297944

-- Define the sets A and B
def A : Set ℕ := {x | ∃ k : ℕ+, x = 2 * k - 1}
def B : Set ℕ := {x | ∃ k : ℕ+, x = 8 * k - 8}

-- Define a function to calculate the sum of m different elements from A
def sumA (m : ℕ) : ℕ := m^2

-- Define a function to calculate the sum of n different elements from B
def sumB (n : ℕ) : ℕ := 4 * n^2 - 4 * n

-- State the theorem
theorem max_m_plus_2n (m n : ℕ) :
  sumA m + sumB n ≤ 967 → m + 2 * n ≤ 44 :=
sorry

end NUMINAMATH_CALUDE_max_m_plus_2n_l2979_297944


namespace NUMINAMATH_CALUDE_min_envelopes_correct_l2979_297955

/-- The number of different flags -/
def num_flags : ℕ := 12

/-- The number of flags in each envelope -/
def flags_per_envelope : ℕ := 2

/-- The probability threshold for having a repeated flag -/
def probability_threshold : ℚ := 1/2

/-- Calculates the probability of all flags being different when opening k envelopes -/
def prob_all_different (k : ℕ) : ℚ :=
  (num_flags.descFactorial (k * flags_per_envelope)) / (num_flags ^ (k * flags_per_envelope))

/-- The minimum number of envelopes to open -/
def min_envelopes : ℕ := 3

theorem min_envelopes_correct :
  (∀ k < min_envelopes, prob_all_different k > probability_threshold) ∧
  (prob_all_different min_envelopes ≤ probability_threshold) :=
sorry

end NUMINAMATH_CALUDE_min_envelopes_correct_l2979_297955


namespace NUMINAMATH_CALUDE_some_zens_not_cens_l2979_297937

-- Define the sets
variable (U : Type) -- Universe set
variable (Zen : Set U) -- Set of Zens
variable (Ben : Set U) -- Set of Bens
variable (Cen : Set U) -- Set of Cens

-- Define the hypotheses
variable (h1 : Zen ⊆ Ben) -- All Zens are Bens
variable (h2 : ∃ x, x ∈ Ben ∧ x ∉ Cen) -- Some Bens are not Cens

-- Theorem to prove
theorem some_zens_not_cens : ∃ x, x ∈ Zen ∧ x ∉ Cen :=
sorry

end NUMINAMATH_CALUDE_some_zens_not_cens_l2979_297937


namespace NUMINAMATH_CALUDE_geometric_relations_l2979_297914

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ₚ " => parallel_plane
local infix:50 " ⊥ₚ " => perpendicular_plane
local infix:50 " ∥ₗₚ " => line_parallel_plane
local infix:50 " ⊥ₗₚ " => line_perpendicular_plane

-- Theorem statement
theorem geometric_relations (m n : Line) (α β : Plane) :
  (((m ⊥ₗₚ α) ∧ (n ⊥ₗₚ β) ∧ (α ⊥ₚ β)) → (m ⊥ n)) ∧
  (((m ⊥ₗₚ α) ∧ (n ∥ₗₚ β) ∧ (α ∥ₚ β)) → (m ⊥ n)) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l2979_297914


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l2979_297950

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49, 16, x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l2979_297950


namespace NUMINAMATH_CALUDE_parabola_transformation_l2979_297986

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 3) - 2

-- State the theorem
theorem parabola_transformation :
  ∀ x : ℝ, transform original_parabola x = (x + 1)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2979_297986


namespace NUMINAMATH_CALUDE_unique_solution_sum_l2979_297959

theorem unique_solution_sum (x y : ℝ) : 
  (|x - 5| = |y - 11|) →
  (|x - 11| = 2*|y - 5|) →
  (x + y = 16) →
  (x + y = 16) :=
by
  sorry

#check unique_solution_sum

end NUMINAMATH_CALUDE_unique_solution_sum_l2979_297959


namespace NUMINAMATH_CALUDE_deposit_growth_condition_l2979_297942

theorem deposit_growth_condition 
  (X r s : ℝ) 
  (h_X_pos : X > 0) 
  (h_s_bound : s < 20) :
  X * (1 + r / 100) * (1 - s / 100) > X ↔ r > 100 * s / (100 - s) := by
  sorry

end NUMINAMATH_CALUDE_deposit_growth_condition_l2979_297942


namespace NUMINAMATH_CALUDE_deriv_two_zeros_neither_necessary_nor_sufficient_l2979_297985

open Set
open Function

/-- A function has two extreme points in an interval -/
def has_two_extreme_points (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (∀ x ∈ Ioo a b, f x ≤ f x₁ ∨ f x ≤ f x₂) ∧
    (∃ y₁ ∈ Ioo a b, f y₁ < f x₁) ∧
    (∃ y₂ ∈ Ioo a b, f y₂ < f x₂)

/-- The derivative of a function has two zeros in an interval -/
def deriv_has_two_zeros (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    deriv f x₁ = 0 ∧ deriv f x₂ = 0

/-- The main theorem stating that having two zeros in the derivative
    is neither necessary nor sufficient for having two extreme points -/
theorem deriv_two_zeros_neither_necessary_nor_sufficient :
  ¬(∀ f : ℝ → ℝ, DifferentiableOn ℝ f (Ioo 0 2) →
    (has_two_extreme_points f 0 2 ↔ deriv_has_two_zeros f 0 2)) :=
sorry

end NUMINAMATH_CALUDE_deriv_two_zeros_neither_necessary_nor_sufficient_l2979_297985


namespace NUMINAMATH_CALUDE_sector_arc_length_l2979_297920

/-- Given a circular sector with area 60π cm² and central angle 150°, its arc length is 10π cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 60 * Real.pi ∧ angle = 150 → arc_length = 10 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2979_297920


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l2979_297968

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P₀
def P₀ : ℝ × ℝ := (-1, -4)

-- Define the slope of the line parallel to the tangent
def parallel_slope : ℝ := 4

theorem tangent_point_and_perpendicular_line :
  -- The tangent line at P₀ is parallel to 4x - y - 1 = 0
  curve_derivative (P₀.1) = parallel_slope →
  -- P₀ is in the third quadrant
  P₀.1 < 0 ∧ P₀.2 < 0 →
  -- P₀ lies on the curve
  curve P₀.1 = P₀.2 →
  -- The equation of the perpendicular line passing through P₀
  ∃ (a b c : ℝ), a * P₀.1 + b * P₀.2 + c = 0 ∧
                 a = 1 ∧ b = 4 ∧ c = 17 ∧
                 -- The perpendicular line is indeed perpendicular to the tangent
                 a * parallel_slope + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l2979_297968


namespace NUMINAMATH_CALUDE_factorization_theorem_l2979_297949

theorem factorization_theorem (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) = (a^2+b^2)*(b^2+c^2)*(c^2+a^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_theorem_l2979_297949


namespace NUMINAMATH_CALUDE_linear_function_preserves_arithmetic_progression_l2979_297954

/-- A sequence (xₙ) is an arithmetic progression if there exists a constant d
    such that xₙ₊₁ = xₙ + d for all n. -/
def is_arithmetic_progression (x : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, x (n + 1) = x n + d

/-- A function f is linear if there exist constants k and b such that
    f(x) = kx + b for all x. -/
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x : ℝ, f x = k * x + b

theorem linear_function_preserves_arithmetic_progression
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (hf : is_linear_function f)
  (hx : is_arithmetic_progression x) :
  is_arithmetic_progression (fun n ↦ f (x n)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_preserves_arithmetic_progression_l2979_297954


namespace NUMINAMATH_CALUDE_andrews_age_l2979_297918

theorem andrews_age (andrew_age grandfather_age : ℕ) : 
  grandfather_age = 16 * andrew_age →
  grandfather_age - andrew_age = 60 →
  andrew_age = 4 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l2979_297918


namespace NUMINAMATH_CALUDE_expression_evaluation_l2979_297967

theorem expression_evaluation : 
  let x : ℚ := -1/2
  let expr := (x - 2) / ((x^2 + 4*x + 4) * ((x^2 + x - 6) / (x + 2) - x + 2))
  expr = 2/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2979_297967


namespace NUMINAMATH_CALUDE_power_function_odd_condition_l2979_297905

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_odd_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (m^2 - 5*m + 7) * x^(m-2)
  is_power_function f ∧ is_odd_function f → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_function_odd_condition_l2979_297905


namespace NUMINAMATH_CALUDE_photo_difference_l2979_297934

/-- The number of photos taken by Lisa -/
def L : ℕ := 50

/-- The number of photos taken by Mike -/
def M : ℕ := sorry

/-- The number of photos taken by Norm -/
def N : ℕ := 110

/-- The total of Lisa and Mike's photos is less than the sum of Mike's and Norm's -/
axiom photo_sum_inequality : L + M < M + N

/-- Norm's photos are 10 more than twice Lisa's photos -/
axiom norm_photos_relation : N = 2 * L + 10

theorem photo_difference : (M + N) - (L + M) = 60 := by sorry

end NUMINAMATH_CALUDE_photo_difference_l2979_297934


namespace NUMINAMATH_CALUDE_evaluate_expression_l2979_297913

theorem evaluate_expression : -(((16 / 4) * 6 - 50) + 5^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2979_297913


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l2979_297904

/-- The initial capacity of the fuel tank in liters -/
def initial_capacity : ℝ := 3000

/-- The amount of fuel remaining on January 1, 2006 in liters -/
def remaining_jan1 : ℝ := 180

/-- The amount of fuel remaining on May 1, 2006 in liters -/
def remaining_may1 : ℝ := 1238

/-- The total volume of fuel used from November 1, 2005 to May 1, 2006 in liters -/
def total_fuel_used : ℝ := 4582

/-- Proof that the initial capacity of the fuel tank is 3000 liters -/
theorem fuel_tank_capacity : 
  initial_capacity = 
    (total_fuel_used + remaining_may1 + remaining_jan1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l2979_297904


namespace NUMINAMATH_CALUDE_fraction_order_l2979_297932

theorem fraction_order : (25 : ℚ) / 21 < 23 / 19 ∧ 23 / 19 < 21 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l2979_297932


namespace NUMINAMATH_CALUDE_number_difference_l2979_297919

theorem number_difference (L S : ℕ) (h1 : L = 1608) (h2 : L = 6 * S + 15) : L - S = 1343 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2979_297919


namespace NUMINAMATH_CALUDE_line_equation_conversion_l2979_297903

/-- Given a line expressed as (2, -1) · ((x, y) - (5, -3)) = 0, prove that when written in the form y = mx + b, m = 2 and b = -13 -/
theorem line_equation_conversion :
  ∀ (x y : ℝ),
  (2 : ℝ) * (x - 5) + (-1 : ℝ) * (y - (-3)) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -13 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_conversion_l2979_297903


namespace NUMINAMATH_CALUDE_lcm_of_153_180_560_l2979_297983

theorem lcm_of_153_180_560 : Nat.lcm 153 (Nat.lcm 180 560) = 85680 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_153_180_560_l2979_297983


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2979_297915

theorem digit_sum_problem :
  ∃ (a b c d : ℕ),
    (1000 ≤ a ∧ a < 10000) ∧
    (1000 ≤ b ∧ b < 10000) ∧
    (1000 ≤ c ∧ c < 10000) ∧
    (1000 ≤ d ∧ d < 10000) ∧
    a + b = 4300 ∧
    c - d = 1542 ∧
    a + c = 5842 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2979_297915


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2979_297989

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a (n + 1) > a n) →
  a 5 ^ 2 = a 10 →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  ∃ c, ∀ n, a n = c * 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2979_297989


namespace NUMINAMATH_CALUDE_hidden_dots_count_l2979_297982

/-- Represents a standard six-sided die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def standardDieSum : ℕ := Finset.sum standardDie id

/-- The number of dice in the stack -/
def numDice : ℕ := 4

/-- The visible numbers on the dice -/
def visibleNumbers : Finset ℕ := {1, 2, 2, 3, 3, 4, 5, 6}

/-- The sum of visible numbers -/
def visibleSum : ℕ := Finset.sum visibleNumbers id

/-- The total number of dots on all dice -/
def totalDots : ℕ := numDice * standardDieSum

theorem hidden_dots_count : totalDots - visibleSum = 58 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l2979_297982


namespace NUMINAMATH_CALUDE_tony_age_proof_l2979_297979

/-- Represents Tony's age at the beginning of the nine-month period -/
def initial_age : ℕ := 10

/-- Represents the number of days Tony worked -/
def work_days : ℕ := 80

/-- Represents Tony's daily work hours -/
def daily_hours : ℕ := 3

/-- Represents Tony's base hourly wage in cents -/
def base_wage : ℕ := 75

/-- Represents the age-based hourly wage increase in cents -/
def age_wage_increase : ℕ := 25

/-- Represents Tony's total earnings in cents -/
def total_earnings : ℕ := 84000

theorem tony_age_proof :
  ∃ (x : ℕ), x ≤ work_days ∧
  (base_wage + age_wage_increase * initial_age) * daily_hours * x +
  (base_wage + age_wage_increase * (initial_age + 1)) * daily_hours * (work_days - x) =
  total_earnings := by sorry

end NUMINAMATH_CALUDE_tony_age_proof_l2979_297979


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l2979_297907

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of cars brought by guests -/
def guest_cars : ℕ := 10

/-- The number of cars belonging to Dylan's parents -/
def parent_cars : ℕ := 2

/-- The total number of cars in the parking lot -/
def total_cars : ℕ := guest_cars + parent_cars

/-- Theorem: The total number of car wheels in the parking lot is 48 -/
theorem total_wheels_in_parking_lot : 
  total_cars * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l2979_297907


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l2979_297973

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l2979_297973


namespace NUMINAMATH_CALUDE_complement_implies_sum_l2979_297999

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (m : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - m) > 0}

-- Define the complement of A in U
def C_UA (m n : ℝ) : Set ℝ := Set.Icc (-1) (-n)

-- Theorem statement
theorem complement_implies_sum (m n : ℝ) : 
  C_UA m n = Set.compl (A m) → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_complement_implies_sum_l2979_297999


namespace NUMINAMATH_CALUDE_first_degree_function_composition_l2979_297931

theorem first_degree_function_composition (f : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) →
  (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_first_degree_function_composition_l2979_297931


namespace NUMINAMATH_CALUDE_intersection_values_l2979_297962

/-- Definition of the circle M -/
def circle_M (x y : ℝ) : Prop := x^2 - 2*x + y^2 + 4*y - 10 = 0

/-- Definition of the intersecting line -/
def intersecting_line (x y : ℝ) (C : ℝ) : Prop := x + 3*y + C = 0

/-- Theorem stating the possible values of C -/
theorem intersection_values (C : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    circle_M A.1 A.2 ∧ 
    circle_M B.1 B.2 ∧
    intersecting_line A.1 A.2 C ∧
    intersecting_line B.1 B.2 C ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 20) →
  C = 15 ∨ C = -5 := by
sorry

end NUMINAMATH_CALUDE_intersection_values_l2979_297962


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2979_297916

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (l m : Line) 
  (α β : Plane) 
  (h_diff_lines : l ≠ m) 
  (h_diff_planes : α ≠ β) 
  (h_l_perp_m : perp_line l m) 
  (h_l_perp_α : perp_line_plane l α) 
  (h_m_perp_β : perp_line_plane m β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2979_297916


namespace NUMINAMATH_CALUDE_practice_for_five_months_l2979_297997

/-- Calculates the total piano practice hours over a given number of months -/
def total_practice_hours (weekly_hours : ℕ) (months : ℕ) : ℕ :=
  weekly_hours * 4 * months

/-- Theorem stating that practicing 4 hours weekly for 5 months results in 80 total hours -/
theorem practice_for_five_months : 
  total_practice_hours 4 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_practice_for_five_months_l2979_297997


namespace NUMINAMATH_CALUDE_cakes_sold_l2979_297924

theorem cakes_sold (made bought left : ℕ) 
  (h1 : made = 173)
  (h2 : bought = 103)
  (h3 : left = 190) :
  made + bought - left = 86 := by
sorry

end NUMINAMATH_CALUDE_cakes_sold_l2979_297924


namespace NUMINAMATH_CALUDE_circle_area_and_circumference_l2979_297951

/-- Given a circle described by the polar equation r = 4 cos θ + 3 sin θ,
    prove that its area is 25π/4 and its circumference is 5π. -/
theorem circle_area_and_circumference :
  ∀ θ : ℝ, ∃ r : ℝ, r = 4 * Real.cos θ + 3 * Real.sin θ →
  ∃ A C : ℝ, A = (25 * Real.pi) / 4 ∧ C = 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_and_circumference_l2979_297951


namespace NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l2979_297965

theorem min_people_with_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = (2 * n) / 5 → 
  hats = (3 * n) / 4 → 
  both = gloves + hats - n → 
  both ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l2979_297965


namespace NUMINAMATH_CALUDE_water_consumption_theorem_l2979_297908

/-- Calculates the number of glasses of water drunk per day given the bottle capacity,
    number of refills per week, glass size, and days in a week. -/
def glassesPerDay (bottleCapacity : ℕ) (refillsPerWeek : ℕ) (glassSize : ℕ) (daysPerWeek : ℕ) : ℕ :=
  (bottleCapacity * refillsPerWeek) / (glassSize * daysPerWeek)

/-- Theorem stating that given the specified conditions, the number of glasses of water
    drunk per day is equal to 4. -/
theorem water_consumption_theorem :
  let bottleCapacity : ℕ := 35
  let refillsPerWeek : ℕ := 4
  let glassSize : ℕ := 5
  let daysPerWeek : ℕ := 7
  glassesPerDay bottleCapacity refillsPerWeek glassSize daysPerWeek = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_theorem_l2979_297908


namespace NUMINAMATH_CALUDE_student_answer_difference_l2979_297939

theorem student_answer_difference (number : ℕ) (h : number = 288) : 
  (5 : ℚ) / 6 * number - (5 : ℚ) / 16 * number = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_answer_difference_l2979_297939


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2979_297991

def sum_of_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_of_range 30 50 + count_even_in_range 30 50 = 851 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2979_297991


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l2979_297994

theorem soccer_team_lineup_count :
  let team_size : ℕ := 16
  let positions_to_fill : ℕ := 5
  (team_size.factorial) / ((team_size - positions_to_fill).factorial) = 524160 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l2979_297994


namespace NUMINAMATH_CALUDE_rectangle_unique_symmetric_shape_l2979_297996

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | RegularPentagon

-- Define axisymmetry and central symmetry
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Rectangle => true
  | Shape.RegularPentagon => true

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.RegularPentagon => false

-- Theorem statement
theorem rectangle_unique_symmetric_shape :
  ∀ s : Shape, isAxisymmetric s ∧ isCentrallySymmetric s ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_unique_symmetric_shape_l2979_297996


namespace NUMINAMATH_CALUDE_m_less_than_n_l2979_297935

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence, M and N are defined as follows -/
def M (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a n * seq.a (n + 3)

def N (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a (n + 1) * seq.a (n + 2)

/-- For any arithmetic sequence with non-zero common difference, M < N -/
theorem m_less_than_n (seq : ArithmeticSequence) (n : ℕ) : M seq n < N seq n := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_n_l2979_297935


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2979_297953

/-- The diameter of a triangle's circumscribed circle, given one side and its opposite angle. -/
theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h_side : side = 18) (h_angle : angle = π/4) :
  let diameter := side / Real.sin angle
  diameter = 18 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2979_297953


namespace NUMINAMATH_CALUDE_circle_touching_three_circles_on_sphere_l2979_297993

/-- The radius of a circle touching three externally touching circles on a sphere -/
theorem circle_touching_three_circles_on_sphere
  (r a : ℝ) (hr : r > 0) (ha : a > 0) (ha_lt_r : a < r) :
  ∃ R : ℝ,
    R = r * Real.sin (Real.arcsin (a / r) - Real.arcsin (a * Real.sqrt 3 / (3 * r))) ∧
    R > 0 ∧
    R < r :=
by sorry

end NUMINAMATH_CALUDE_circle_touching_three_circles_on_sphere_l2979_297993


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2979_297927

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, 2 * k * x^2 + 36 * x + 3 * k = 0) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2979_297927


namespace NUMINAMATH_CALUDE_no_valid_function_l2979_297910

/-- The set M = {0, 1, 2, ..., 2022} -/
def M : Set Nat := Finset.range 2023

/-- The theorem stating that no function f satisfies both required conditions -/
theorem no_valid_function :
  ¬∃ (f : M → M → M),
    (∀ (a b : M), f a (f b a) = b) ∧
    (∀ (x : M), f x x ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_function_l2979_297910


namespace NUMINAMATH_CALUDE_speed_in_still_water_l2979_297928

/-- The speed of a man in still water given his upstream and downstream speeds -/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 60) 
  (h2 : downstream_speed = 90) : 
  (upstream_speed + downstream_speed) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l2979_297928


namespace NUMINAMATH_CALUDE_clothing_business_profit_l2979_297998

/-- Represents the daily profit function for a clothing business -/
def daily_profit (x : ℝ) : ℝ :=
  (40 - x) * (20 + 2 * x)

theorem clothing_business_profit :
  (∃ x : ℝ, x ≥ 0 ∧ daily_profit x = 1200) ∧
  (∀ y : ℝ, y ≥ 0 → daily_profit y ≠ 1800) := by
  sorry

end NUMINAMATH_CALUDE_clothing_business_profit_l2979_297998


namespace NUMINAMATH_CALUDE_jimmy_matchbooks_count_l2979_297963

/-- The number of matches in one matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of stamps Tonya initially had -/
def tonya_initial_stamps : ℕ := 13

/-- The number of stamps Tonya had left after trading -/
def tonya_final_stamps : ℕ := 3

/-- The number of matchbooks Jimmy had -/
def jimmy_matchbooks : ℕ := 5

theorem jimmy_matchbooks_count :
  jimmy_matchbooks * matches_per_matchbook = 
    (tonya_initial_stamps - tonya_final_stamps) * matches_per_stamp :=
by sorry

end NUMINAMATH_CALUDE_jimmy_matchbooks_count_l2979_297963


namespace NUMINAMATH_CALUDE_complex_sum_l2979_297975

theorem complex_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := a + b * i
  z = (1 - i)^2 / (1 + i) →
  a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_l2979_297975


namespace NUMINAMATH_CALUDE_infinitely_many_powers_of_two_in_floor_sqrt_two_n_l2979_297988

theorem infinitely_many_powers_of_two_in_floor_sqrt_two_n : 
  ∀ m : ℕ, ∃ n k : ℕ, n > m ∧ k > 0 ∧ ⌊Real.sqrt 2 * n⌋ = 2^k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_powers_of_two_in_floor_sqrt_two_n_l2979_297988


namespace NUMINAMATH_CALUDE_log_9_81_equals_2_l2979_297943

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_9_81_equals_2 : log 9 81 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_9_81_equals_2_l2979_297943


namespace NUMINAMATH_CALUDE_circular_track_circumference_l2979_297929

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference (speed1 speed2 meeting_time : ℝ) 
  (h1 : speed1 = 7)
  (h2 : speed2 = 8)
  (h3 : meeting_time = 40)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0)
  (h6 : meeting_time > 0) :
  speed1 * meeting_time + speed2 * meeting_time = 600 :=
by sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l2979_297929


namespace NUMINAMATH_CALUDE_race_probability_inconsistency_l2979_297977

-- Define the probabilities for each car to win
def prob_X_wins : ℚ := 1/2
def prob_Y_wins : ℚ := 1/4
def prob_Z_wins : ℚ := 1/3

-- Define the total probability of one of them winning
def total_prob : ℚ := 1.0833333333333333

-- Theorem stating the inconsistency of the given probabilities
theorem race_probability_inconsistency :
  prob_X_wins + prob_Y_wins + prob_Z_wins = total_prob ∧
  total_prob > 1 := by sorry

end NUMINAMATH_CALUDE_race_probability_inconsistency_l2979_297977


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l2979_297922

theorem sqrt_fraction_equality : 
  Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35)) = 256 / Real.sqrt 2049 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l2979_297922


namespace NUMINAMATH_CALUDE_min_value_theorem_l2979_297926

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 8) :
  (2/x + 3/y) ≥ 25/8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 8 ∧ 2/x₀ + 3/y₀ = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2979_297926


namespace NUMINAMATH_CALUDE_three_lines_plane_count_l2979_297990

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- Predicate to check if a line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Function to determine the number of planes formed by three lines -/
def num_planes_formed (l1 l2 l3 : Line3D) : ℕ :=
  sorry

/-- Theorem stating that three lines, where one intersects the other two,
    can form either 1, 2, or 3 planes -/
theorem three_lines_plane_count 
  (l1 l2 l3 : Line3D) 
  (h1 : intersects l1 l2) 
  (h2 : intersects l1 l3) : 
  let n := num_planes_formed l1 l2 l3
  n = 1 ∨ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_plane_count_l2979_297990


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l2979_297952

/-- Two cyclists meeting problem -/
theorem cyclists_meeting_time
  (b : ℝ) -- distance between towns A and B in km
  (peter_speed : ℝ) -- Peter's speed in km/h
  (john_speed : ℝ) -- John's speed in km/h
  (h1 : peter_speed = 7) -- Peter's speed is 7 km/h
  (h2 : john_speed = 5) -- John's speed is 5 km/h
  : ∃ p : ℝ, p = b / (peter_speed + john_speed) ∧ p = b / 12 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l2979_297952


namespace NUMINAMATH_CALUDE_max_additional_plates_is_24_l2979_297909

def initial_plates : ℕ := 3 * 2 * 4

def scenario1 : ℕ := (3 + 2) * 2 * 4
def scenario2 : ℕ := 3 * 2 * (4 + 2)
def scenario3 : ℕ := (3 + 1) * 2 * (4 + 1)
def scenario4 : ℕ := (3 + 1) * (2 + 1) * 4

def max_additional_plates : ℕ := max scenario1 (max scenario2 (max scenario3 scenario4)) - initial_plates

theorem max_additional_plates_is_24 : max_additional_plates = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_additional_plates_is_24_l2979_297909


namespace NUMINAMATH_CALUDE_auction_starting_value_l2979_297906

/-- The starting value of an auction satisfying the given conditions -/
def auctionStartingValue : ℝ → Prop := fun S =>
  let harryFirstBid := S + 200
  let secondBidderBid := 2 * harryFirstBid
  let thirdBidderBid := secondBidderBid + 3 * harryFirstBid
  let harryFinalBid := 4000
  harryFinalBid = thirdBidderBid + 1500

theorem auction_starting_value : ∃ S, auctionStartingValue S ∧ S = 300 := by
  sorry

end NUMINAMATH_CALUDE_auction_starting_value_l2979_297906


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2979_297940

theorem one_and_two_thirds_of_x_is_45 : ∃ x : ℝ, (5/3) * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2979_297940


namespace NUMINAMATH_CALUDE_gift_cost_problem_l2979_297946

theorem gift_cost_problem (initial_friends : ℕ) (dropped_out : ℕ) (extra_cost : ℝ) :
  initial_friends = 10 →
  dropped_out = 4 →
  extra_cost = 8 →
  ∃ (total_cost : ℝ),
    total_cost / (initial_friends - dropped_out : ℝ) = total_cost / initial_friends + extra_cost ∧
    total_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_problem_l2979_297946


namespace NUMINAMATH_CALUDE_unique_prime_cube_plus_two_l2979_297957

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The main theorem stating that there is exactly one positive integer n ≥ 2 
    such that n^3 + 2 is prime -/
theorem unique_prime_cube_plus_two :
  ∃! (n : ℕ), n ≥ 2 ∧ isPrime (n^3 + 2) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_cube_plus_two_l2979_297957


namespace NUMINAMATH_CALUDE_product_sampling_is_srs_l2979_297987

/-- Represents a sampling method --/
structure SamplingMethod where
  total : Nat
  sample_size : Nat
  method : String

/-- Defines what constitutes a simple random sample --/
def is_simple_random_sample (sm : SamplingMethod) : Prop :=
  sm.sample_size ≤ sm.total ∧
  sm.method = "drawing lots" ∧
  ∀ (subset : Finset (Fin sm.total)),
    subset.card = sm.sample_size →
    ∃ (p : ℝ), p > 0 ∧ p = 1 / (Nat.choose sm.total sm.sample_size)

/-- The sampling method described in the problem --/
def product_sampling : SamplingMethod :=
  { total := 10
  , sample_size := 3
  , method := "drawing lots" }

/-- Theorem stating that the product sampling method is a simple random sample --/
theorem product_sampling_is_srs : is_simple_random_sample product_sampling := by
  sorry


end NUMINAMATH_CALUDE_product_sampling_is_srs_l2979_297987


namespace NUMINAMATH_CALUDE_urn_ball_removal_l2979_297984

theorem urn_ball_removal (total : ℕ) (red_percent : ℚ) (blue_removed : ℕ) (new_red_percent : ℚ) : 
  total = 150 →
  red_percent = 2/5 →
  blue_removed = 75 →
  new_red_percent = 4/5 →
  (red_percent * total : ℚ) / (total - blue_removed : ℚ) = new_red_percent :=
by sorry

end NUMINAMATH_CALUDE_urn_ball_removal_l2979_297984


namespace NUMINAMATH_CALUDE_f_convex_when_a_negative_a_range_when_f_bounded_l2979_297960

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Define convexity condition
def is_convex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2

-- Theorem 1: f is convex when a < 0
theorem f_convex_when_a_negative (a : ℝ) (h : a < 0) : is_convex (f a) := by
  sorry

-- Theorem 2: Range of a when |f(x)| ≤ 1 for x ∈ [0, 1]
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1) → -2 ≤ a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_convex_when_a_negative_a_range_when_f_bounded_l2979_297960


namespace NUMINAMATH_CALUDE_log_xy_value_l2979_297972

theorem log_xy_value (x y : ℝ) 
  (h1 : Real.log (x^2 * y^5) = 2) 
  (h2 : Real.log (x^3 * y^2) = 2) : 
  Real.log (x * y) = 8 / 11 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l2979_297972


namespace NUMINAMATH_CALUDE_multiplicative_inverse_144_mod_941_l2979_297961

theorem multiplicative_inverse_144_mod_941 : ∃ n : ℤ, 
  0 ≤ n ∧ n < 941 ∧ (144 * n) % 941 = 1 := by
  use 364
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_144_mod_941_l2979_297961


namespace NUMINAMATH_CALUDE_k5_not_planar_l2979_297992

/-- A graph is planar if it satisfies the inequality E ≤ 3V - 6 -/
def is_planar (V E : ℕ) : Prop := E ≤ 3 * V - 6

/-- The number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- K5 is not planar -/
theorem k5_not_planar :
  ¬ (is_planar 5 (complete_graph_edges 5)) :=
sorry

end NUMINAMATH_CALUDE_k5_not_planar_l2979_297992


namespace NUMINAMATH_CALUDE_sandy_work_hours_l2979_297981

theorem sandy_work_hours (total_hours : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 45 → 
  num_days = 5 → 
  total_hours = num_days * hours_per_day → 
  hours_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_sandy_work_hours_l2979_297981


namespace NUMINAMATH_CALUDE_no_2014_ambiguous_numbers_l2979_297969

/-- A positive integer k is 2014-ambiguous if both x^2 + kx + 2014 and x^2 + kx - 2014 have two integer roots -/
def is_2014_ambiguous (k : ℕ+) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 + k * x₁ + 2014 = 0 ∧
    x₂^2 + k * x₂ + 2014 = 0 ∧
    y₁^2 + k * y₁ - 2014 = 0 ∧
    y₂^2 + k * y₂ - 2014 = 0

theorem no_2014_ambiguous_numbers : ¬∃ k : ℕ+, is_2014_ambiguous k := by
  sorry

end NUMINAMATH_CALUDE_no_2014_ambiguous_numbers_l2979_297969


namespace NUMINAMATH_CALUDE_eight_stairs_climb_ways_l2979_297938

/-- The number of ways to climb n stairs, taking 1, 2, 3, or 4 steps at a time. -/
def climbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => climbWays m + climbWays (m + 1) + climbWays (m + 2) + climbWays (m + 3)

theorem eight_stairs_climb_ways :
  climbWays 8 = 108 := by
  sorry

#eval climbWays 8

end NUMINAMATH_CALUDE_eight_stairs_climb_ways_l2979_297938


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_one_zero_l2979_297974

/-- A circle with center (a, a) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ

/-- The circle is tangent to the x-axis at (1, 0) -/
def isTangentAtOneZero (c : Circle) : Prop :=
  c.r = 1 ∧ c.a = 1

/-- The equation of the circle -/
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.a)^2 = c.r^2

theorem circle_tangent_to_x_axis_at_one_zero :
  ∀ c : Circle, isTangentAtOneZero c →
  ∀ x y : ℝ, circleEquation c x y ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_one_zero_l2979_297974


namespace NUMINAMATH_CALUDE_identity_function_proof_l2979_297933

theorem identity_function_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_identity_function_proof_l2979_297933


namespace NUMINAMATH_CALUDE_product_ab_is_zero_l2979_297900

theorem product_ab_is_zero (a b : ℝ) 
  (sum_eq : a + b = 4) 
  (sum_cubes_eq : a^3 + b^3 = 64) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_product_ab_is_zero_l2979_297900


namespace NUMINAMATH_CALUDE_largest_result_operation_l2979_297945

theorem largest_result_operation : 
  let a := -1
  let b := -(1/2)
  let add_result := a + b
  let sub_result := a - b
  let mul_result := a * b
  let div_result := a / b
  (div_result > add_result) ∧ 
  (div_result > sub_result) ∧ 
  (div_result > mul_result) ∧
  (div_result = 2) := by
sorry

end NUMINAMATH_CALUDE_largest_result_operation_l2979_297945


namespace NUMINAMATH_CALUDE_closed_polygonal_line_even_segments_l2979_297936

/-- Represents a segment of the polygonal line -/
structure Segment where
  x : Int
  y : Int

/-- Represents a closed polygonal line on a grid -/
structure ClosedPolygonalLine where
  segments : List Segment
  is_closed : segments.length > 0
  same_length : ∀ s ∈ segments, s.x^2 + s.y^2 = 1
  on_grid : ∀ s ∈ segments, s.x = 0 ∨ s.y = 0

/-- The main theorem stating that the number of segments in a closed polygonal line is even -/
theorem closed_polygonal_line_even_segments (p : ClosedPolygonalLine) : 
  Even p.segments.length := by
  sorry

end NUMINAMATH_CALUDE_closed_polygonal_line_even_segments_l2979_297936


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2979_297902

theorem imaginary_part_of_z (z : ℂ) (h : z * ((1 + Complex.I)^2 / 2) = 1 + 2 * Complex.I) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2979_297902


namespace NUMINAMATH_CALUDE_assignment_conditions_l2979_297923

/-- The number of ways to assign four students to three classes -/
def assignStudents : ℕ :=
  Nat.choose 4 2 * (3 * 2 * 1) - (3 * 2 * 1)

/-- Conditions of the problem -/
theorem assignment_conditions :
  (∀ (assignment : Fin 4 → Fin 3), 
    (∀ c : Fin 3, ∃ s : Fin 4, assignment s = c) ∧ 
    (assignment 0 ≠ assignment 1)) →
  (assignStudents = 30) :=
sorry

end NUMINAMATH_CALUDE_assignment_conditions_l2979_297923


namespace NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l2979_297911

theorem sunglasses_and_hats_probability
  (total_sunglasses : ℕ)
  (total_hats : ℕ)
  (prob_hat_and_sunglasses : ℚ)
  (h1 : total_sunglasses = 80)
  (h2 : total_hats = 50)
  (h3 : prob_hat_and_sunglasses = 3/5) :
  (prob_hat_and_sunglasses * total_hats) / total_sunglasses = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l2979_297911


namespace NUMINAMATH_CALUDE_rent_increase_calculation_l2979_297980

/-- Calculates the monthly rent after a price increase, given the total rental period,
    initial rent period, initial monthly rent, and total amount paid. -/
def calculate_increased_rent (total_years : ℕ) (initial_years : ℕ) (initial_rent : ℕ) (total_paid : ℕ) : ℕ :=
  let total_months := total_years * 12
  let initial_months := initial_years * 12
  let increased_months := total_months - initial_months
  let initial_total := initial_months * initial_rent
  let increased_total := total_paid - initial_total
  increased_total / increased_months

theorem rent_increase_calculation :
  calculate_increased_rent 5 3 300 19200 = 350 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_calculation_l2979_297980


namespace NUMINAMATH_CALUDE_b_share_is_1000_l2979_297976

/-- Given a partnership with investment ratios A:B:C as 2:2/3:1 and a total profit,
    calculate B's share of the profit. -/
def calculate_B_share (total_profit : ℚ) : ℚ :=
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 2/3
  let c_ratio : ℚ := 1
  let total_ratio : ℚ := a_ratio + b_ratio + c_ratio
  (b_ratio / total_ratio) * total_profit

/-- Theorem stating that given the investment ratios and a total profit of 5500,
    B's share of the profit is 1000. -/
theorem b_share_is_1000 :
  calculate_B_share 5500 = 1000 := by
  sorry

#eval calculate_B_share 5500

end NUMINAMATH_CALUDE_b_share_is_1000_l2979_297976


namespace NUMINAMATH_CALUDE_problem_solution_l2979_297947

/-- Given M = 2x + y, N = 2x - y, P = xy, M = 4, and N = 2, prove that P = 1.5 -/
theorem problem_solution (x y M N P : ℝ) 
  (hM : M = 2*x + y)
  (hN : N = 2*x - y)
  (hP : P = x*y)
  (hM_val : M = 4)
  (hN_val : N = 2) :
  P = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2979_297947


namespace NUMINAMATH_CALUDE_ab_value_l2979_297930

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B : Set ℝ := {x | ∃ a b : ℝ, x^2 - a*x + b = 0}

-- State the theorem
theorem ab_value (a b : ℝ) :
  A ∪ B = {2, 3, 5} →
  A ∩ B = {3} →
  B = {x | x^2 - a*x + b = 0} →
  a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2979_297930


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l2979_297958

theorem relationship_between_x_and_y (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l2979_297958


namespace NUMINAMATH_CALUDE_quadratic_roots_k_less_than_9_l2979_297995

theorem quadratic_roots_k_less_than_9 (k : ℝ) (h : k < 9) :
  (∃ x : ℝ, (k - 5) * x^2 - 2 * (k - 3) * x + k = 0) ∧
  ((∃! x : ℝ, (k - 5) * x^2 - 2 * (k - 3) * x + k = 0) ∨
   (∃ x y : ℝ, x ≠ y ∧ (k - 5) * x^2 - 2 * (k - 3) * x + k = 0 ∧
                      (k - 5) * y^2 - 2 * (k - 3) * y + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_k_less_than_9_l2979_297995


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l2979_297978

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := 3 * m * x + (m + 2) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + (m + 2) * y + 2 = 0

theorem parallel_lines_m_values :
  ∀ m : ℝ, parallel (3 * m) (m + 2) (m - 2) (m + 2) → m = -1 ∨ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l2979_297978


namespace NUMINAMATH_CALUDE_harlys_dogs_l2979_297912

theorem harlys_dogs (x : ℝ) : 
  (0.6 * x + 5 = 53) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_harlys_dogs_l2979_297912


namespace NUMINAMATH_CALUDE_other_communities_count_l2979_297948

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 32 / 100 →
  sikh_percent = 10 / 100 →
  ∃ (other_boys : ℕ), other_boys = 119 ∧ 
    (other_boys : ℚ) / total_boys = 1 - (muslim_percent + hindu_percent + sikh_percent) :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l2979_297948


namespace NUMINAMATH_CALUDE_h_min_neg_l2979_297925

-- Define the functions f, g, and h
variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def h (x : ℝ) := a * f x + b * g x + 2

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_odd : ∀ x, g (-x) = -g x

-- Define the maximum value of h on (0, +∞)
axiom h_max : ∀ x > 0, h x ≤ 5

-- State the theorem to be proved
theorem h_min_neg : (∀ x < 0, h x ≥ -1) := by sorry

end NUMINAMATH_CALUDE_h_min_neg_l2979_297925


namespace NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_thirteen_l2979_297971

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : Nat
  blue : Nat
  yellow : Nat

/-- The minimum number of tries required to guarantee obtaining one blue and one yellow ball -/
def minTriesForBlueAndYellow (counts : BallCounts) : Nat :=
  counts.purple + counts.blue + 1

theorem min_tries_for_blue_and_yellow_is_thirteen :
  let counts : BallCounts := { purple := 7, blue := 5, yellow := 11 }
  minTriesForBlueAndYellow counts = 13 := by sorry

end NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_thirteen_l2979_297971


namespace NUMINAMATH_CALUDE_square_1849_product_l2979_297921

theorem square_1849_product (x : ℤ) (h : x^2 = 1849) : (x + 2) * (x - 2) = 1845 := by
  sorry

end NUMINAMATH_CALUDE_square_1849_product_l2979_297921


namespace NUMINAMATH_CALUDE_jeff_daya_money_fraction_l2979_297941

/-- The amount of money Emma has in dollars -/
def emma_money : ℚ := 8

/-- The percentage more money Daya has compared to Emma -/
def daya_percentage : ℚ := 25 / 100

/-- The amount of money Brenda has in dollars -/
def brenda_money : ℚ := 8

/-- The difference in dollars between Brenda's and Jeff's money -/
def brenda_jeff_diff : ℚ := 4

/-- Calculates Daya's money based on Emma's money and the percentage more Daya has -/
def daya_money : ℚ := emma_money * (1 + daya_percentage)

/-- Calculates Jeff's money based on Brenda's money and the difference between them -/
def jeff_money : ℚ := brenda_money - brenda_jeff_diff

/-- The fraction of money Jeff has compared to Daya -/
def jeff_daya_fraction : ℚ := jeff_money / daya_money

theorem jeff_daya_money_fraction :
  jeff_daya_fraction = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_jeff_daya_money_fraction_l2979_297941


namespace NUMINAMATH_CALUDE_wooden_box_width_l2979_297956

def wooden_box_length : Real := 8
def wooden_box_height : Real := 6
def small_box_length : Real := 0.04
def small_box_width : Real := 0.07
def small_box_height : Real := 0.06
def max_small_boxes : Nat := 2000000

theorem wooden_box_width :
  ∃ (W : Real),
    W * wooden_box_length * wooden_box_height =
      (small_box_length * small_box_width * small_box_height) * max_small_boxes ∧
    W = 7 := by
  sorry

end NUMINAMATH_CALUDE_wooden_box_width_l2979_297956


namespace NUMINAMATH_CALUDE_problem_statement_l2979_297901

theorem problem_statement (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2979_297901


namespace NUMINAMATH_CALUDE_probability_non_red_face_l2979_297917

theorem probability_non_red_face (total_faces : ℕ) (red_faces : ℕ) (yellow_faces : ℕ) (blue_faces : ℕ) (green_faces : ℕ)
  (h1 : total_faces = 10)
  (h2 : red_faces = 5)
  (h3 : yellow_faces = 3)
  (h4 : blue_faces = 1)
  (h5 : green_faces = 1)
  (h6 : total_faces = red_faces + yellow_faces + blue_faces + green_faces) :
  (yellow_faces + blue_faces + green_faces : ℚ) / total_faces = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_red_face_l2979_297917


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2979_297964

def game_winner (M : ℕ) : Prop :=
  M ≤ 999 ∧ 
  2 * M < 1000 ∧ 
  6 * M < 1000 ∧ 
  12 * M < 1000 ∧ 
  36 * M > 999

theorem smallest_winning_number : 
  ∃ (M : ℕ), game_winner M ∧ ∀ (N : ℕ), N < M → ¬game_winner N :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2979_297964


namespace NUMINAMATH_CALUDE_bits_required_for_ABC12_l2979_297970

-- Define the hexadecimal number ABC12₁₆
def hex_number : ℕ := 0xABC12

-- Theorem stating that the number of bits required to represent ABC12₁₆ is 20
theorem bits_required_for_ABC12 :
  (Nat.log 2 hex_number).succ = 20 := by sorry

end NUMINAMATH_CALUDE_bits_required_for_ABC12_l2979_297970


namespace NUMINAMATH_CALUDE_function_property_center_of_symmetry_range_property_l2979_297966

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1 - a) / (a - x)

theorem function_property (a : ℝ) (x : ℝ) (h : x ≠ a) :
  f a x + f a (2 * a - x) + 2 = 0 := by sorry

theorem center_of_symmetry (a b : ℝ) 
  (h : ∀ x ≠ a, f a x + f a (6 - x) = 2 * b) :
  a + b = -4/7 := by sorry

theorem range_property (a : ℝ) :
  (∀ x ∈ Set.Icc (a + 1/2) (a + 1), f a x ∈ Set.Icc (-3) (-2)) ∧
  (∀ y ∈ Set.Icc (-3) (-2), ∃ x ∈ Set.Icc (a + 1/2) (a + 1), f a x = y) := by sorry

end NUMINAMATH_CALUDE_function_property_center_of_symmetry_range_property_l2979_297966
