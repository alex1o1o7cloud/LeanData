import Mathlib

namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_is_24_l3628_362829

/-- Given a boat with speed in still water and the relationship between upstream and downstream times,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) : ℝ :=
  let stream_speed := (boat_speed : ℝ) / 3
  have h1 : upstream_time = 2 * downstream_time := by sorry
  have h2 : boat_speed = 72 := by sorry
  have h3 : upstream_time * (boat_speed - stream_speed) = downstream_time * (boat_speed + stream_speed) := by sorry
  stream_speed

/-- The speed of the stream is 24 kmph. -/
theorem stream_speed_is_24 : stream_speed 72 1 0.5 = 24 := by sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_is_24_l3628_362829


namespace NUMINAMATH_CALUDE_curve_tangent_parallel_l3628_362824

theorem curve_tangent_parallel (k : ℝ) : 
  let f := fun x : ℝ => k * x + Real.log x
  let f' := fun x : ℝ => k + 1 / x
  (f' 1 = 2) → k = 1 := by
sorry

end NUMINAMATH_CALUDE_curve_tangent_parallel_l3628_362824


namespace NUMINAMATH_CALUDE_min_value_expression_l3628_362826

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x * y = 4) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3628_362826


namespace NUMINAMATH_CALUDE_proportional_function_k_l3628_362875

theorem proportional_function_k (k : ℝ) (h1 : k ≠ 0) (h2 : -5 = k * 3) : k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_k_l3628_362875


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3628_362897

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that the given conditions result in the correct principal -/
theorem simple_interest_problem :
  let interest : ℚ := 4016.25
  let rate : ℚ := 3
  let time : ℕ := 5
  calculate_principal interest rate time = 26775 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3628_362897


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l3628_362860

theorem simple_interest_time_period 
  (principal : ℝ)
  (amount1 : ℝ)
  (amount2 : ℝ)
  (rate_increase : ℝ)
  (h1 : principal = 825)
  (h2 : amount1 = 956)
  (h3 : amount2 = 1055)
  (h4 : rate_increase = 4) :
  ∃ (rate : ℝ) (time : ℝ),
    amount1 = principal + (principal * rate * time) / 100 ∧
    amount2 = principal + (principal * (rate + rate_increase) * time) / 100 ∧
    time = 3 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l3628_362860


namespace NUMINAMATH_CALUDE_complex_fraction_value_l3628_362889

theorem complex_fraction_value (a : ℝ) (z : ℂ) : 
  z = (a^2 - 1 : ℂ) + (a + 1 : ℂ) * Complex.I ∧ z.re = 0 → 
  (a + Complex.I^2016) / (1 + Complex.I) = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l3628_362889


namespace NUMINAMATH_CALUDE_exam_max_marks_l3628_362839

theorem exam_max_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) :
  percentage = 0.90 →
  scored_marks = 405 →
  percentage * max_marks = scored_marks →
  max_marks = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_max_marks_l3628_362839


namespace NUMINAMATH_CALUDE_soccer_ball_holes_percentage_l3628_362817

theorem soccer_ball_holes_percentage 
  (total_balls : ℕ) 
  (successfully_inflated : ℕ) 
  (overinflation_rate : ℚ) :
  total_balls = 100 →
  successfully_inflated = 48 →
  overinflation_rate = 1/5 →
  ∃ (x : ℚ), 
    0 ≤ x ∧ 
    x ≤ 1 ∧ 
    (1 - x) * (1 - overinflation_rate) * total_balls = successfully_inflated ∧
    x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_holes_percentage_l3628_362817


namespace NUMINAMATH_CALUDE_unique_single_digit_polynomial_exists_l3628_362887

/-- A polynomial with single-digit coefficients -/
def SingleDigitPolynomial (p : Polynomial ℤ) : Prop :=
  ∀ i, (p.coeff i) ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℤ)

/-- The theorem statement -/
theorem unique_single_digit_polynomial_exists (n : ℤ) :
  ∃! p : Polynomial ℤ, SingleDigitPolynomial p ∧ p.eval (-2) = n ∧ p.eval (-5) = n := by
  sorry

end NUMINAMATH_CALUDE_unique_single_digit_polynomial_exists_l3628_362887


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l3628_362825

theorem surrounding_circles_radius (r : ℝ) : 
  (∃ (A B C D : ℝ × ℝ),
    -- The centers of the surrounding circles form a square
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (4*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (4*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (4*r)^2 ∧
    -- The diagonal of the square
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = (2 + 2*r)^2 ∧
    -- The surrounding circles touch the central circle
    ∃ (O : ℝ × ℝ), (A.1 - O.1)^2 + (A.2 - O.2)^2 = (r + 1)^2) →
  r = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l3628_362825


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3628_362891

/-- Given a positive geometric sequence {a_n}, prove that a_8 * a_12 = 16,
    where a_1 and a_19 are the roots of x^2 - 10x + 16 = 0 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * r) →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 8 * a 12 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3628_362891


namespace NUMINAMATH_CALUDE_base6_154_to_decimal_l3628_362866

/-- Converts a list of digits in base 6 to its decimal (base 10) representation -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem base6_154_to_decimal :
  base6ToDecimal [4, 5, 1] = 70 := by
  sorry

#eval base6ToDecimal [4, 5, 1]

end NUMINAMATH_CALUDE_base6_154_to_decimal_l3628_362866


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3628_362801

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a^2 + b^2 = c^2 →  -- right-angled triangle condition
    a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
    c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3628_362801


namespace NUMINAMATH_CALUDE_simplify_expression_l3628_362861

theorem simplify_expression (y : ℝ) : 3*y + 9*y^2 + 10 - (5 - 3*y - 9*y^2) = 18*y^2 + 6*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3628_362861


namespace NUMINAMATH_CALUDE_negation_of_all_cars_are_fast_l3628_362846

variable (U : Type) -- Universe of discourse
variable (car : U → Prop) -- Predicate for being a car
variable (fast : U → Prop) -- Predicate for being fast

theorem negation_of_all_cars_are_fast :
  ¬(∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬(fast x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_cars_are_fast_l3628_362846


namespace NUMINAMATH_CALUDE_inequality_proof_l3628_362899

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) ∧
  ((a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l3628_362899


namespace NUMINAMATH_CALUDE_equation_solution_l3628_362815

theorem equation_solution (x y : ℝ) : (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3628_362815


namespace NUMINAMATH_CALUDE_pet_store_white_cats_l3628_362818

theorem pet_store_white_cats 
  (total : ℕ) 
  (black : ℕ) 
  (gray : ℕ) 
  (h1 : total = 15) 
  (h2 : black = 10) 
  (h3 : gray = 3) 
  (h4 : ∃ white : ℕ, total = white + black + gray) : 
  ∃ white : ℕ, white = 2 ∧ total = white + black + gray :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_white_cats_l3628_362818


namespace NUMINAMATH_CALUDE_school_population_l3628_362809

/-- Given a school population where:
  * b is the number of boys
  * g is the number of girls
  * t is the number of teachers
  * There are twice as many boys as girls
  * There are four times as many girls as teachers
Prove that the total population is 13t -/
theorem school_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) :
  b + g + t = 13 * t := by
  sorry

end NUMINAMATH_CALUDE_school_population_l3628_362809


namespace NUMINAMATH_CALUDE_circle_radius_l3628_362812

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3628_362812


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3628_362805

-- Define the function f(x) = x³ - 22 - x
def f (x : ℝ) := x^3 - 22 - x

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_root_exists_in_interval_l3628_362805


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3628_362836

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ -3 ↔ x - 5 > 3*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3628_362836


namespace NUMINAMATH_CALUDE_andrews_stickers_l3628_362816

theorem andrews_stickers (total : ℕ) (daniels : ℕ) (freds_extra : ℕ) 
  (h1 : total = 750)
  (h2 : daniels = 250)
  (h3 : freds_extra = 120) :
  total - (daniels + (daniels + freds_extra)) = 130 := by
  sorry

end NUMINAMATH_CALUDE_andrews_stickers_l3628_362816


namespace NUMINAMATH_CALUDE_mistake_position_l3628_362856

theorem mistake_position (n : ℕ) (a₁ : ℤ) (d : ℤ) (sum : ℤ) (k : ℕ) : 
  n = 21 →
  a₁ = 51 →
  d = 5 →
  sum = 2021 →
  k ∈ Finset.range n →
  sum = (n * (2 * a₁ + (n - 1) * d)) / 2 - 10 * k →
  k = 10 :=
by sorry

end NUMINAMATH_CALUDE_mistake_position_l3628_362856


namespace NUMINAMATH_CALUDE_sector_area_l3628_362814

theorem sector_area (α : Real) (l : Real) (S : Real) :
  α = π / 9 →
  l = π / 3 →
  S = (1 / 2) * l * (l / α) →
  S = π / 2 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l3628_362814


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l3628_362885

-- Define the function g
def g (c : ℤ) : ℝ → ℝ := λ x => 4 * x + c

-- State the theorem
theorem intersection_point_of_function_and_inverse (c : ℤ) :
  ∃ (d : ℤ), (g c (-4) = d ∧ g c d = -4) → d = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l3628_362885


namespace NUMINAMATH_CALUDE_violet_balloons_remaining_l3628_362853

def initial_violet_balloons : ℕ := 7
def lost_violet_balloons : ℕ := 3

theorem violet_balloons_remaining :
  initial_violet_balloons - lost_violet_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_violet_balloons_remaining_l3628_362853


namespace NUMINAMATH_CALUDE_firstYearStudents2012_is_set_l3628_362884

/-- A type representing a student -/
structure Student :=
  (name : String)
  (year : Nat)
  (school : String)
  (enrollmentYear : Nat)

/-- Definition of a well-defined criterion for set membership -/
def hasWellDefinedCriterion (s : Set Student) : Prop :=
  ∀ x : Student, (x ∈ s) ∨ (x ∉ s)

/-- The set of all first-year high school students at a certain school in 2012 -/
def firstYearStudents2012 (school : String) : Set Student :=
  {s : Student | s.year = 1 ∧ s.school = school ∧ s.enrollmentYear = 2012}

/-- Theorem stating that the collection of first-year students in 2012 forms a set -/
theorem firstYearStudents2012_is_set (school : String) :
  hasWellDefinedCriterion (firstYearStudents2012 school) :=
sorry

end NUMINAMATH_CALUDE_firstYearStudents2012_is_set_l3628_362884


namespace NUMINAMATH_CALUDE_range_of_k_in_linear_system_l3628_362813

/-- Given a system of linear equations and an inequality constraint,
    prove the range of the parameter k. -/
theorem range_of_k_in_linear_system (x y k : ℝ) :
  (2 * x - y = k + 1) →
  (x - y = -3) →
  (x + y > 2) →
  k > -4.5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_in_linear_system_l3628_362813


namespace NUMINAMATH_CALUDE_x_power_x_power_x_at_3_l3628_362879

theorem x_power_x_power_x_at_3 :
  let x : ℝ := 3
  (x^x)^(x^x) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_x_power_x_power_x_at_3_l3628_362879


namespace NUMINAMATH_CALUDE_xy_equals_four_l3628_362850

theorem xy_equals_four (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_x : x = w)
  (h_y : y = w)
  (h_w : w + w = w * w)
  (h_z : z = 3) : 
  x * y = 4 := by
sorry

end NUMINAMATH_CALUDE_xy_equals_four_l3628_362850


namespace NUMINAMATH_CALUDE_mistaken_division_l3628_362811

theorem mistaken_division (n : ℕ) (h : n = 172) :
  ∃! x : ℕ, x > 0 ∧ n % x = 7 ∧ n / x = n / 4 - 28 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_l3628_362811


namespace NUMINAMATH_CALUDE_vector_parallelism_transitivity_l3628_362888

/-- Given three non-zero vectors, if the first is parallel to the second and the second is parallel to the third, then the first is parallel to the third. -/
theorem vector_parallelism_transitivity 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : ∃ (k : ℝ), a = k • b) (hbc : ∃ (m : ℝ), b = m • c) : 
  ∃ (n : ℝ), a = n • c :=
sorry

end NUMINAMATH_CALUDE_vector_parallelism_transitivity_l3628_362888


namespace NUMINAMATH_CALUDE_prob_at_least_two_successes_l3628_362874

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 3

/-- The probability of exactly k successes in n trials -/
def binomialProb (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of at least 2 successes in 3 trials -/
theorem prob_at_least_two_successes : 
  binomialProb 2 + binomialProb 3 = 81/125 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_successes_l3628_362874


namespace NUMINAMATH_CALUDE_descending_order_abcd_l3628_362863

theorem descending_order_abcd (a b c d : ℚ) 
  (h1 : 2006 = 9 * a) 
  (h2 : 2006 = 15 * b) 
  (h3 : 2006 = 32 * c) 
  (h4 : 2006 = 68 * d) : 
  a > b ∧ b > c ∧ c > d := by
  sorry

end NUMINAMATH_CALUDE_descending_order_abcd_l3628_362863


namespace NUMINAMATH_CALUDE_curve_is_part_of_ellipse_l3628_362896

-- Define the curve
def curve (x y : ℝ) : Prop := x = Real.sqrt (1 - 4 * y^2)

-- Define an ellipse
def is_ellipse (x y : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem curve_is_part_of_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), curve x y → x ≥ 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_part_of_ellipse_l3628_362896


namespace NUMINAMATH_CALUDE_center_sum_coords_l3628_362845

/-- Defines a circle with the equation x^2 + y^2 = 6x - 8y + 24 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x - 8*y + 24

/-- Defines the center of a circle -/
def is_center (h x y : ℝ) : Prop :=
  ∀ (a b : ℝ), circle_equation a b → (a - x)^2 + (b - y)^2 = h^2

theorem center_sum_coords :
  ∃ (x y : ℝ), is_center 7 x y ∧ x + y = -1 :=
sorry

end NUMINAMATH_CALUDE_center_sum_coords_l3628_362845


namespace NUMINAMATH_CALUDE_picnic_theorem_l3628_362822

def picnic_problem (people : ℕ) (sandwich_price : ℚ) (fruit_salad_price : ℚ) (soda_price : ℚ) (sodas_per_person : ℕ) (snack_bags : ℕ) (total_spent : ℚ) : Prop :=
  let sandwich_cost := people * sandwich_price
  let fruit_salad_cost := people * fruit_salad_price
  let soda_cost := people * sodas_per_person * soda_price
  let food_cost := sandwich_cost + fruit_salad_cost + soda_cost
  let snack_cost := total_spent - food_cost
  snack_cost / snack_bags = 4

theorem picnic_theorem : 
  picnic_problem 4 5 3 2 2 3 60 := by
  sorry

end NUMINAMATH_CALUDE_picnic_theorem_l3628_362822


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l3628_362843

theorem binomial_expansion_terms (n : ℕ) : (Finset.range (2 * n + 1)).card = 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l3628_362843


namespace NUMINAMATH_CALUDE_empty_set_proof_l3628_362872

theorem empty_set_proof : {x : ℝ | x^2 + 1 = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l3628_362872


namespace NUMINAMATH_CALUDE_apples_sold_l3628_362882

/-- The amount of apples sold in a store --/
theorem apples_sold (kidney : ℕ) (golden : ℕ) (canada : ℕ) (left : ℕ) : 
  kidney + golden + canada - left = (kidney + golden + canada) - left :=
by sorry

end NUMINAMATH_CALUDE_apples_sold_l3628_362882


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_1000th_l3628_362895

def P (n : ℕ) : ℚ :=
  (2^(n-1) * Nat.factorial (n-1)) / (Nat.factorial (2*n-1) * (2*n+1))

theorem smallest_n_for_P_less_than_1000th (n : ℕ) : n = 18 ↔ 
  (n > 0 ∧ P n < 1/1000 ∧ ∀ m : ℕ, m > 0 ∧ m < n → P m ≥ 1/1000) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_1000th_l3628_362895


namespace NUMINAMATH_CALUDE_total_distance_is_490_l3628_362800

/-- Represents a segment of the journey -/
structure JourneySegment where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled in a journey segment -/
def distanceTraveled (segment : JourneySegment) : ℝ :=
  segment.speed * segment.time

/-- Represents the entire journey -/
def Journey : List JourneySegment := [
  { speed := 90, time := 2 },
  { speed := 60, time := 1 },
  { speed := 100, time := 2.5 }
]

/-- Theorem: The total distance traveled in the journey is 490 km -/
theorem total_distance_is_490 : 
  (Journey.map distanceTraveled).sum = 490 := by sorry

end NUMINAMATH_CALUDE_total_distance_is_490_l3628_362800


namespace NUMINAMATH_CALUDE_nomogram_relations_l3628_362894

-- Define the nomogram scales as real numbers
variables (x y z t r w v q s : ℝ)

-- Define y₁ as a function of y
def y₁ (y : ℝ) : ℝ := y

-- Theorem statement
theorem nomogram_relations :
  z = (x + 2 * y₁ y) / 3 ∧
  w = 2 * z ∧
  r = x - 2 ∧
  y + q = 6 ∧
  2 * s + t = 8 ∧
  3 * z - x - 2 * t + 6 = 0 ∧
  8 * z - 4 * t - v + 12 = 0 := by
sorry


end NUMINAMATH_CALUDE_nomogram_relations_l3628_362894


namespace NUMINAMATH_CALUDE_equation_solution_l3628_362851

theorem equation_solution : {x : ℝ | x^2 = 2*x} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3628_362851


namespace NUMINAMATH_CALUDE_min_distance_sum_five_digit_numbers_l3628_362865

theorem min_distance_sum_five_digit_numbers (x₁ x₂ x₃ x₄ x₅ : ℕ) :
  -- Define the constraints
  x₅ ≥ 9 →
  x₄ + x₅ ≥ 99 →
  x₃ + x₄ + x₅ ≥ 999 →
  x₂ + x₃ + x₄ + x₅ ≥ 9999 →
  x₁ + x₂ + x₃ + x₄ + x₅ = 99999 →
  -- The theorem to prove
  x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ ≥ 101105 :=
by sorry

#check min_distance_sum_five_digit_numbers

end NUMINAMATH_CALUDE_min_distance_sum_five_digit_numbers_l3628_362865


namespace NUMINAMATH_CALUDE_hong_travel_bound_l3628_362849

/-- Represents a town in the country -/
structure Town where
  coins : ℕ

/-- Represents the country with its towns and roads -/
structure Country where
  towns : Set Town
  roads : Set (Town × Town)
  initial_coins : ℕ

/-- Represents Hong's travel -/
structure Travel where
  country : Country
  days : ℕ

/-- The maximum number of days Hong can travel -/
def max_travel_days (n : ℕ) : ℕ := n + 2 * n^(2/3)

theorem hong_travel_bound (c : Country) (t : Travel) (h_infinite : Infinite c.towns)
    (h_all_connected : ∀ a b : Town, a ≠ b → (a, b) ∈ c.roads)
    (h_initial_coins : ∀ town ∈ c.towns, town.coins = c.initial_coins)
    (h_coin_transfer : ∀ k : ℕ, ∀ a b : Town, 
      (a, b) ∈ c.roads → t.days = k → b.coins = b.coins - k ∧ a.coins = a.coins + k)
    (h_road_usage : ∀ a b : Town, (a, b) ∈ c.roads → (b, a) ∉ c.roads) :
  t.days ≤ max_travel_days c.initial_coins :=
sorry

end NUMINAMATH_CALUDE_hong_travel_bound_l3628_362849


namespace NUMINAMATH_CALUDE_work_completion_l3628_362876

/-- The number of days B worked before leaving the job --/
def days_B_worked (a_rate b_rate : ℚ) (a_remaining_days : ℚ) : ℚ :=
  15 * (1 - 4 * a_rate)

theorem work_completion 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (a_remaining_days : ℚ) 
  (h1 : a_rate = 1 / 12)
  (h2 : b_rate = 1 / 15)
  (h3 : a_remaining_days = 4) :
  days_B_worked a_rate b_rate a_remaining_days = 10 := by
  sorry

#eval days_B_worked (1/12) (1/15) 4

end NUMINAMATH_CALUDE_work_completion_l3628_362876


namespace NUMINAMATH_CALUDE_emilys_spending_l3628_362837

/-- Given Emily's spending pattern over four days and the total amount spent,
    prove that the amount she spent on Friday is equal to the total divided by 18. -/
theorem emilys_spending (X Y : ℝ) : 
  X > 0 →  -- Assuming X is positive
  Y > 0 →  -- Assuming Y is positive
  X + 2*X + 3*X + 4*(3*X) = Y →  -- Total spending equation
  X = Y / 18 := by
sorry

end NUMINAMATH_CALUDE_emilys_spending_l3628_362837


namespace NUMINAMATH_CALUDE_trig_simplification_l3628_362833

theorem trig_simplification :
  (Real.cos (20 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_simplification_l3628_362833


namespace NUMINAMATH_CALUDE_complex_magnitude_l3628_362828

theorem complex_magnitude (z : ℂ) (h : (z - Complex.I) * (2 - Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3628_362828


namespace NUMINAMATH_CALUDE_rubber_band_calculation_l3628_362834

/-- The number of rubber bands in a small ball -/
def small_ball_rubber_bands : ℕ := 50

/-- The number of rubber bands in a large ball -/
def large_ball_rubber_bands : ℕ := 300

/-- The total number of rubber bands -/
def total_rubber_bands : ℕ := 5000

/-- The number of small balls made -/
def small_balls_made : ℕ := 22

/-- The number of large balls that can be made with remaining rubber bands -/
def large_balls_possible : ℕ := 13

theorem rubber_band_calculation :
  small_ball_rubber_bands * small_balls_made +
  large_ball_rubber_bands * large_balls_possible = total_rubber_bands :=
by sorry

end NUMINAMATH_CALUDE_rubber_band_calculation_l3628_362834


namespace NUMINAMATH_CALUDE_binary_multiplication_subtraction_l3628_362881

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def a : List Bool := [true, false, true, true, false, true, true]
def b : List Bool := [true, false, true, true, true]
def c : List Bool := [false, true, false, true, false, true]
def result : List Bool := [true, false, false, false, false, true, false, false, false, false, true]

theorem binary_multiplication_subtraction :
  nat_to_binary (binary_to_nat a * binary_to_nat b - binary_to_nat c) = result :=
sorry

end NUMINAMATH_CALUDE_binary_multiplication_subtraction_l3628_362881


namespace NUMINAMATH_CALUDE_pears_picked_total_l3628_362820

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 6

/-- The number of pears Tim picked -/
def tim_pears : ℕ := 5

/-- The total number of pears picked -/
def total_pears : ℕ := sara_pears + tim_pears

theorem pears_picked_total : total_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l3628_362820


namespace NUMINAMATH_CALUDE_olivias_race_time_l3628_362804

def total_time : ℕ := 112  -- 1 hour 52 minutes in minutes

theorem olivias_race_time (olivia_time : ℕ) 
  (h1 : olivia_time + (olivia_time - 4) = total_time) : 
  olivia_time = 58 := by
  sorry

end NUMINAMATH_CALUDE_olivias_race_time_l3628_362804


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3628_362841

theorem simplify_and_evaluate : 
  ∀ (a b : ℝ), 
    (a + 2*b)^2 - (a + b)*(a - b) = 4*a*b + 5*b^2 ∧
    (((-1/2) + 2*2)^2 - ((-1/2) + 2)*((-1/2) - 2) = 16) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3628_362841


namespace NUMINAMATH_CALUDE_circle_in_rectangle_l3628_362838

theorem circle_in_rectangle (r x : ℝ) : 
  r > 0 →  -- radius is positive
  2 * r = x →  -- width of rectangle is diameter of circle
  r + (2 * x) / 3 + r = 10 →  -- length of rectangle
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_in_rectangle_l3628_362838


namespace NUMINAMATH_CALUDE_f_properties_l3628_362807

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * log x

theorem f_properties (m : ℝ) (h : m ≥ 1) :
  (∃! (x : ℝ), x > 0 ∧ f m x = x^2 - (m + 1) * x) ∧
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f m x ≤ f m y) ∧
  (∃ (x : ℝ), x > 0 ∧ f m x = (m/2) * (1 - log m)) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l3628_362807


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l3628_362855

theorem quadratic_roots_sum_and_product :
  let f : ℝ → ℝ := λ x => x^2 - 18*x + 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ 
    (x₁ + x₂ = 18) ∧ (x₁ * x₂ = 16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l3628_362855


namespace NUMINAMATH_CALUDE_gym_class_distance_l3628_362869

/-- The total distance students have to run in gym class -/
def total_distance (track_length : ℕ) (completed_laps remaining_laps : ℕ) : ℕ :=
  track_length * (completed_laps + remaining_laps)

/-- Proof that the total distance to run is 1500 meters -/
theorem gym_class_distance : total_distance 150 6 4 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_gym_class_distance_l3628_362869


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l3628_362877

theorem unique_two_digit_multiple : ∃! t : ℕ, 10 ≤ t ∧ t < 100 ∧ (13 * t) % 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l3628_362877


namespace NUMINAMATH_CALUDE_pieces_with_high_product_bound_l3628_362842

/-- Represents an infinite chessboard with pieces placed on it. -/
structure InfiniteChessboard where
  m : ℕ  -- Total number of pieces
  piece_positions : Finset (ℕ × ℕ)  -- Positions of pieces
  piece_count : piece_positions.card = m  -- Ensure the number of pieces matches m

/-- Calculates the number of pieces in a given row -/
def pieces_in_row (board : InfiniteChessboard) (row : ℕ) : ℕ :=
  (board.piece_positions.filter (fun p => p.1 = row)).card

/-- Calculates the number of pieces in a given column -/
def pieces_in_column (board : InfiniteChessboard) (col : ℕ) : ℕ :=
  (board.piece_positions.filter (fun p => p.2 = col)).card

/-- Calculates the product of pieces in the row and column for a given position -/
def product_for_position (board : InfiniteChessboard) (pos : ℕ × ℕ) : ℕ :=
  (pieces_in_row board pos.1) * (pieces_in_column board pos.2)

/-- The main theorem to be proved -/
theorem pieces_with_high_product_bound (board : InfiniteChessboard) :
  (board.piece_positions.filter (fun pos => product_for_position board pos ≥ 10 * board.m)).card ≤ board.m / 10 :=
sorry

end NUMINAMATH_CALUDE_pieces_with_high_product_bound_l3628_362842


namespace NUMINAMATH_CALUDE_problem_solution_l3628_362868

def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) ∧
  (∀ k : ℝ, (∃ x : ℝ, (f 2 x + f 2 (-x)) / 3 < |k|) ↔ k < -2/3 ∨ k > 2/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3628_362868


namespace NUMINAMATH_CALUDE_length_comparison_l3628_362886

theorem length_comparison : 
  900/1000 < (2 : ℝ) ∧ (2 : ℝ) < 300/100 ∧ 300/100 < 80/10 ∧ 80/10 < 1000 := by
  sorry

end NUMINAMATH_CALUDE_length_comparison_l3628_362886


namespace NUMINAMATH_CALUDE_sector_area_l3628_362893

theorem sector_area (r a b : ℝ) : 
  r = 1 →  -- radius is 1 cm
  a = 1 →  -- arc length is 1 cm
  b = (1/2) * r * a →  -- area formula for a sector
  b = 1/2  -- the area of the sector is 1/2 cm²
:= by sorry

end NUMINAMATH_CALUDE_sector_area_l3628_362893


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l3628_362848

def total_money : ℚ := 90
def sandwich_fraction : ℚ := 1/5
def book_fraction : ℚ := 1/2
def money_left : ℚ := 12

theorem museum_ticket_fraction :
  let spent := total_money - money_left
  let sandwich_cost := sandwich_fraction * total_money
  let book_cost := book_fraction * total_money
  let museum_cost := spent - (sandwich_cost + book_cost)
  museum_cost / total_money = 1/6 := by sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l3628_362848


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3628_362870

theorem sum_of_coefficients (n : ℕ) : 
  (∀ x : ℝ, x ≠ 0 → (3 * x^2 + 1/x)^n = 256 → n = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3628_362870


namespace NUMINAMATH_CALUDE_cover_triangles_l3628_362806

/-- The side length of the small equilateral triangle -/
def small_side : ℝ := 0.5

/-- The side length of the large equilateral triangle -/
def large_side : ℝ := 10

/-- The minimum number of small triangles needed to cover the large triangle -/
def min_triangles : ℕ := 400

theorem cover_triangles : 
  ∀ (n : ℕ), n * (small_side^2 * Real.sqrt 3 / 4) ≥ large_side^2 * Real.sqrt 3 / 4 → n ≥ min_triangles :=
by sorry

end NUMINAMATH_CALUDE_cover_triangles_l3628_362806


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l3628_362857

noncomputable def m (a x : ℝ) : ℝ × ℝ := (a * Real.cos x, Real.cos x)
noncomputable def n (b x : ℝ) : ℝ × ℝ := (2 * Real.cos x, b * Real.sin x)

noncomputable def f (a b x : ℝ) : ℝ := (m a x).1 * (n b x).1 + (m a x).2 * (n b x).2

theorem vector_dot_product_problem (a b : ℝ) :
  (∃ x, f a b x = 2) ∧ 
  (f a b (π/3) = 1/2 + Real.sqrt 3/2) →
  (∃ x_min ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f a b x_min ≤ f a b x) ∧
  (∃ x_max ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f a b x ≤ f a b x_max) ∧
  (∀ θ, 0 < θ ∧ θ < π ∧ f a b (θ/2) = 3/2 → Real.tan θ = -(4 + Real.sqrt 7)/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l3628_362857


namespace NUMINAMATH_CALUDE_cost_of_goods_l3628_362819

/-- The cost of goods A, B, and C given certain conditions -/
theorem cost_of_goods (x y z : ℚ) 
  (h1 : 2*x + 4*y + z = 90)
  (h2 : 4*x + 10*y + z = 110) : 
  x + y + z = 80 := by
sorry

end NUMINAMATH_CALUDE_cost_of_goods_l3628_362819


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3628_362859

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 140 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3628_362859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3628_362827

/-- 
Given an arithmetic sequence where the third term is 3 and the eleventh term is 15,
prove that the first term is 0 and the common difference is 3/2.
-/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 3 = 3) 
  (h2 : a 11 = 15) 
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : 
  a 1 = 0 ∧ a 2 - a 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3628_362827


namespace NUMINAMATH_CALUDE_characterization_of_n_l3628_362880

/-- A positive integer is square-free if it is not divisible by any perfect square greater than 1 -/
def IsSquareFree (n : ℕ+) : Prop :=
  ∀ (d : ℕ+), d * d ∣ n → d = 1

/-- The condition that for all positive integers x and y, if n divides x^n - y^n, then n^2 divides x^n - y^n -/
def Condition (n : ℕ+) : Prop :=
  ∀ (x y : ℕ+), n ∣ (x ^ n.val - y ^ n.val) → n.val * n.val ∣ (x ^ n.val - y ^ n.val)

/-- The main theorem stating the characterization of n satisfying the condition -/
theorem characterization_of_n (n : ℕ+) :
  Condition n ↔ (∃ (m : ℕ+), IsSquareFree m ∧ (n = m ∨ n = 2 * m)) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_n_l3628_362880


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3628_362821

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane 
  (l n : Line) (α : Plane) :
  parallel l n → perpendicular_line_plane n α → perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3628_362821


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3628_362831

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) ∧
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3628_362831


namespace NUMINAMATH_CALUDE_place_value_ratio_l3628_362835

def number : ℚ := 86549.2047

theorem place_value_ratio : 
  let thousands_place_value : ℚ := 1000
  let tenths_place_value : ℚ := 0.1
  thousands_place_value / tenths_place_value = 10000 := by sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3628_362835


namespace NUMINAMATH_CALUDE_math_textbooks_same_box_probability_l3628_362823

/-- The probability of all mathematics textbooks ending up in the same box -/
theorem math_textbooks_same_box_probability :
  let total_books : ℕ := 15
  let math_books : ℕ := 4
  let box1_capacity : ℕ := 4
  let box2_capacity : ℕ := 5
  let box3_capacity : ℕ := 6
  
  -- Total number of ways to distribute books
  let total_distributions : ℕ := (Nat.choose total_books box1_capacity) * 
                                 (Nat.choose (total_books - box1_capacity) box2_capacity) *
                                 (Nat.choose (total_books - box1_capacity - box2_capacity) box3_capacity)
  
  -- Number of ways where all math books are in the same box
  let favorable_outcomes : ℕ := (Nat.choose (total_books - math_books) 0) +
                                (Nat.choose (total_books - math_books) 1) +
                                (Nat.choose (total_books - math_books) 2)
  
  (favorable_outcomes : ℚ) / total_distributions = 67 / 630630 :=
by sorry

end NUMINAMATH_CALUDE_math_textbooks_same_box_probability_l3628_362823


namespace NUMINAMATH_CALUDE_smallest_valid_arrangement_l3628_362878

/-- Represents a circular table with chairs -/
structure CircularTable :=
  (num_chairs : ℕ)

/-- Checks if a seating arrangement is valid -/
def is_valid_arrangement (table : CircularTable) (seated : ℕ) : Prop :=
  seated > 0 ∧ seated ≤ table.num_chairs ∧ 
  ∀ (new_seat : ℕ), new_seat ≤ table.num_chairs → 
    ∃ (occupied_seat : ℕ), occupied_seat ≤ table.num_chairs ∧ 
      (new_seat = occupied_seat + 1 ∨ new_seat = occupied_seat - 1 ∨ 
       (occupied_seat = table.num_chairs ∧ new_seat = 1) ∨ 
       (occupied_seat = 1 ∧ new_seat = table.num_chairs))

/-- The theorem to be proved -/
theorem smallest_valid_arrangement (table : CircularTable) 
  (h : table.num_chairs = 100) : 
  (∃ (n : ℕ), is_valid_arrangement table n ∧ 
    ∀ (m : ℕ), m < n → ¬is_valid_arrangement table m) ∧
  (∃ (n : ℕ), is_valid_arrangement table n ∧ n = 20) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_arrangement_l3628_362878


namespace NUMINAMATH_CALUDE_gumball_probability_l3628_362883

theorem gumball_probability (blue_twice_prob : ℚ) 
  (h1 : blue_twice_prob = 16/49) : 
  let blue_prob : ℚ := (blue_twice_prob.sqrt)
  let pink_prob : ℚ := 1 - blue_prob
  pink_prob = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l3628_362883


namespace NUMINAMATH_CALUDE_english_book_pages_l3628_362890

theorem english_book_pages :
  ∀ (english_pages chinese_pages : ℕ),
  english_pages = chinese_pages + 12 →
  3 * english_pages + 4 * chinese_pages = 1275 →
  english_pages = 189 :=
by
  sorry

end NUMINAMATH_CALUDE_english_book_pages_l3628_362890


namespace NUMINAMATH_CALUDE_shop_item_cost_prices_l3628_362840

theorem shop_item_cost_prices :
  ∀ (c1 c2 : ℝ),
    (0.30 * c1 - 0.15 * c1 = 120) →
    (0.25 * c2 - 0.10 * c2 = 150) →
    c1 = 800 ∧ c2 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_shop_item_cost_prices_l3628_362840


namespace NUMINAMATH_CALUDE_letter_F_perimeter_is_19_l3628_362810

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of the letter F given the specified conditions -/
def letter_F_perimeter (large : Rectangle) (small : Rectangle) (offset : ℝ) : ℝ :=
  2 * large.height + -- vertical sides of large rectangle
  (large.width - small.width) + -- uncovered top of large rectangle
  small.width -- bottom of small rectangle

/-- Theorem stating that the perimeter of the letter F is 19 inches -/
theorem letter_F_perimeter_is_19 :
  let large : Rectangle := { width := 2, height := 6 }
  let small : Rectangle := { width := 2, height := 2 }
  let offset : ℝ := 1
  letter_F_perimeter large small offset = 19 := by
  sorry

#eval letter_F_perimeter { width := 2, height := 6 } { width := 2, height := 2 } 1

end NUMINAMATH_CALUDE_letter_F_perimeter_is_19_l3628_362810


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3628_362808

theorem system_solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0)
  (eq1 : 8 * x - 5 * y = c) (eq2 : 10 * y - 16 * x = d) : d / c = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3628_362808


namespace NUMINAMATH_CALUDE_rectangle_ratio_in_square_arrangement_l3628_362867

/-- Represents the arrangement of rectangles around a square -/
structure SquareArrangement where
  s : ℝ  -- side length of the inner square
  x : ℝ  -- longer side of each rectangle
  y : ℝ  -- shorter side of each rectangle

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_ratio_in_square_arrangement
  (arr : SquareArrangement)
  (h1 : arr.s > 0)  -- inner square side length is positive
  (h2 : arr.s + 2 * arr.y = 3 * arr.s)  -- outer square side length relation
  (h3 : arr.x + arr.s = 3 * arr.s)  -- outer square side length relation in perpendicular direction
  : arr.x / arr.y = 2 := by
  sorry

#check rectangle_ratio_in_square_arrangement

end NUMINAMATH_CALUDE_rectangle_ratio_in_square_arrangement_l3628_362867


namespace NUMINAMATH_CALUDE_stating_crabapple_sequence_count_l3628_362864

/-- Represents the number of students in the class -/
def num_students : ℕ := 11

/-- Represents the number of days the class meets -/
def num_days : ℕ := 3

/-- 
  Calculates the number of possible sequences for selecting students to receive a crabapple,
  given that no student can be selected on consecutive days.
-/
def crabapple_sequences (n : ℕ) (d : ℕ) : ℕ :=
  if d = 1 then n
  else if d = 2 then n * (n - 1)
  else n * (n - 1) * (n - 1)

/-- 
  Theorem stating that the number of possible sequences for selecting students
  to receive a crabapple over three days in a class of 11 students,
  where no student can be selected on consecutive days, is 1100.
-/
theorem crabapple_sequence_count :
  crabapple_sequences num_students num_days = 1100 := by
  sorry

end NUMINAMATH_CALUDE_stating_crabapple_sequence_count_l3628_362864


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l3628_362858

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the line
structure Line where
  m : ℝ
  c : ℝ

-- Define the problem
theorem hyperbola_line_intersection
  (h : Hyperbola)
  (l : Line)
  (P Q R : Point)
  (h_eccentricity : h.e = Real.sqrt 3)
  (l_slope : l.m = 1)
  (intersect_y_axis : R.x = 0)
  (dot_product : P.x * Q.x + P.y * Q.y = -3)
  (segment_ratio : P.x - R.x = 3 * (R.x - Q.x))
  (on_line_P : P.y = l.m * P.x + l.c)
  (on_line_Q : Q.y = l.m * Q.x + l.c)
  (on_line_R : R.y = l.m * R.x + l.c)
  (on_hyperbola_P : 2 * P.x^2 - P.y^2 = 2 * h.a^2)
  (on_hyperbola_Q : 2 * Q.x^2 - Q.y^2 = 2 * h.a^2) :
  (l.c = 1 ∨ l.c = -1) ∧ h.a^2 = 1 ∧ h.b^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l3628_362858


namespace NUMINAMATH_CALUDE_total_persimmons_in_boxes_l3628_362832

/-- Given that each box contains 100 persimmons and there are 6 boxes,
    prove that the total number of persimmons is 600. -/
theorem total_persimmons_in_boxes : 
  let persimmons_per_box : ℕ := 100
  let number_of_boxes : ℕ := 6
  persimmons_per_box * number_of_boxes = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_persimmons_in_boxes_l3628_362832


namespace NUMINAMATH_CALUDE_mersenne_prime_condition_l3628_362873

theorem mersenne_prime_condition (a n : ℕ) : 
  a > 1 → n > 1 → Nat.Prime (a^n - 1) → a = 2 ∧ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_mersenne_prime_condition_l3628_362873


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l3628_362847

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x+2)^2 + 16y^2 = 64 is 2√5 -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (C.1 + 2)^2 / 16 + C.2^2 / 4 = 1 ∧  -- C is on the ellipse
    (D.1 + 2)^2 / 16 + D.2^2 / 4 = 1 ∧  -- D is on the ellipse
    C.2 = 0 ∧                           -- C is on the x-axis (major axis)
    D.1 = -2 ∧                          -- D is on the y-axis (minor axis)
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l3628_362847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3628_362892

/-- An arithmetic sequence with a non-zero common difference and positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a n > 0
  h3 : ∀ n, a (n + 1) = a n + d

/-- For an arithmetic sequence with non-zero common difference and positive terms, a₁ · a₈ < a₄ · a₅ -/
theorem arithmetic_sequence_inequality (seq : ArithmeticSequence) : seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3628_362892


namespace NUMINAMATH_CALUDE_line_symmetry_l3628_362830

-- Define a line by its coefficients a, b, and c in the equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define symmetry with respect to x = 1
def symmetricAboutX1 (l1 l2 : Line) : Prop :=
  ∀ x y : ℝ, l1.a * (2 - x) + l1.b * y + l1.c = 0 ↔ l2.a * x + l2.b * y + l2.c = 0

-- Theorem statement
theorem line_symmetry (l1 l2 : Line) :
  l1 = Line.mk 3 (-4) (-3) →
  symmetricAboutX1 l1 l2 →
  l2 = Line.mk 3 4 (-3) := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3628_362830


namespace NUMINAMATH_CALUDE_fraction_sum_product_l3628_362844

theorem fraction_sum_product : 
  24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_l3628_362844


namespace NUMINAMATH_CALUDE_water_speed_calculation_l3628_362803

/-- The speed of water in a river where a person who can swim at 4 km/h in still water
    takes 8 hours to swim 16 km against the current. -/
def water_speed : ℝ :=
  let still_water_speed : ℝ := 4
  let distance : ℝ := 16
  let time : ℝ := 8
  2

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ)
    (h1 : still_water_speed = 4)
    (h2 : distance = 16)
    (h3 : time = 8)
    (h4 : distance = (still_water_speed - water_speed) * time) :
  water_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l3628_362803


namespace NUMINAMATH_CALUDE_quadratic_and_system_solution_l3628_362898

theorem quadratic_and_system_solution :
  (∃ x₁ x₂ : ℚ, (4 * (x₁ - 1)^2 - 25 = 0 ∧ x₁ = 7/2) ∧
                (4 * (x₂ - 1)^2 - 25 = 0 ∧ x₂ = -3/2)) ∧
  (∃ x y : ℚ, (2*x - y = 4 ∧ 3*x + 2*y = 1) ∧ x = 9/7 ∧ y = -10/7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_and_system_solution_l3628_362898


namespace NUMINAMATH_CALUDE_christmas_ball_colors_l3628_362852

/-- Given a total number of balls and the number of balls per color, 
    calculate the number of colors used. -/
def number_of_colors (total_balls : ℕ) (balls_per_color : ℕ) : ℕ :=
  total_balls / balls_per_color

/-- Prove that the number of colors used is 10 given the problem conditions. -/
theorem christmas_ball_colors :
  let total_balls : ℕ := 350
  let balls_per_color : ℕ := 35
  number_of_colors total_balls balls_per_color = 10 := by
  sorry

end NUMINAMATH_CALUDE_christmas_ball_colors_l3628_362852


namespace NUMINAMATH_CALUDE_sqrt_calculation_and_exponent_simplification_l3628_362802

theorem sqrt_calculation_and_exponent_simplification :
  (∃ x : ℝ, x^2 = 18) ∧ (∃ y : ℝ, y^2 = 32) ∧ (∃ z : ℝ, z^2 = 2) →
  (∃ a : ℝ, a^2 = 3) →
  (∀ x y z : ℝ, x^2 = 18 ∧ y^2 = 32 ∧ z^2 = 2 → x - y + z = 0) ∧
  (∀ a : ℝ, a^2 = 3 → (a + 2)^2022 * (a - 2)^2021 * (a - 3) = 3 + a) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculation_and_exponent_simplification_l3628_362802


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l3628_362871

/-- The number of jelly beans initially in the barrel -/
def initial_jelly_beans : ℕ := 8000

/-- The number of people who took jelly beans -/
def total_people : ℕ := 10

/-- The number of people who took twice as many jelly beans -/
def first_group : ℕ := 6

/-- The number of people who took fewer jelly beans -/
def second_group : ℕ := 4

/-- The number of jelly beans taken by each person in the second group -/
def jelly_beans_per_second_group : ℕ := 400

/-- The number of jelly beans remaining in the barrel after everyone took their share -/
def remaining_jelly_beans : ℕ := 1600

theorem jelly_bean_problem :
  initial_jelly_beans = 
    (first_group * 2 * jelly_beans_per_second_group) + 
    (second_group * jelly_beans_per_second_group) + 
    remaining_jelly_beans :=
by
  sorry

#check jelly_bean_problem

end NUMINAMATH_CALUDE_jelly_bean_problem_l3628_362871


namespace NUMINAMATH_CALUDE_digital_earth_purpose_theorem_l3628_362854

/-- Represents the purpose of Digital Earth -/
structure DigitalEarthPurpose where
  dealWithIssues : Bool
  maximizeResources : Bool
  obtainInformation : Bool
  provideLocationData : Bool

/-- The developers of Digital Earth -/
inductive DigitalEarthDeveloper
  | ISDE
  | CAS

/-- The purpose of Digital Earth as developed by ISDE and CAS -/
def digitalEarthPurpose (developers : List DigitalEarthDeveloper) : DigitalEarthPurpose :=
  { dealWithIssues := true,
    maximizeResources := true,
    obtainInformation := true,
    provideLocationData := false }

theorem digital_earth_purpose_theorem (developers : List DigitalEarthDeveloper) 
  (h1 : DigitalEarthDeveloper.ISDE ∈ developers) 
  (h2 : DigitalEarthDeveloper.CAS ∈ developers) :
  let purpose := digitalEarthPurpose developers
  purpose.dealWithIssues ∧ purpose.maximizeResources ∧ purpose.obtainInformation ∧ ¬purpose.provideLocationData :=
by
  sorry

end NUMINAMATH_CALUDE_digital_earth_purpose_theorem_l3628_362854


namespace NUMINAMATH_CALUDE_intersection_union_when_a_2_complement_intersection_condition_l3628_362862

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | |x| < a}

theorem intersection_union_when_a_2 :
  A ∩ B 2 = {x | 1/2 ≤ x ∧ x < 2} ∧
  A ∪ B 2 = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_intersection_condition (a : ℝ) :
  (Aᶜ ∩ B a = B a) ↔ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_union_when_a_2_complement_intersection_condition_l3628_362862
