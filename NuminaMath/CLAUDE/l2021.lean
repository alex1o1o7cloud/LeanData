import Mathlib

namespace NUMINAMATH_CALUDE_base3_addition_l2021_202125

/-- Represents a number in base 3 --/
def Base3 := List Nat

/-- Converts a base 3 number to its decimal representation --/
def toDecimal (n : Base3) : Nat :=
  n.reverse.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The problem statement --/
theorem base3_addition :
  let a : Base3 := [2, 1, 0, 1]
  let b : Base3 := [1, 2, 0, 2]
  let c : Base3 := [1, 2, 0, 1, 1]
  let d : Base3 := [2, 1, 0, 2, 1]
  let result : Base3 := [2, 0, 0, 2, 1, 2]
  toDecimal a + toDecimal b + toDecimal c + toDecimal d = toDecimal result := by
  sorry


end NUMINAMATH_CALUDE_base3_addition_l2021_202125


namespace NUMINAMATH_CALUDE_jack_marbles_l2021_202120

theorem jack_marbles (initial : ℕ) (shared : ℕ) (final : ℕ) :
  initial = 62 →
  shared = 33 →
  final = initial - shared →
  final = 29 :=
by sorry

end NUMINAMATH_CALUDE_jack_marbles_l2021_202120


namespace NUMINAMATH_CALUDE_registration_methods_l2021_202136

/-- The number of ways to distribute n distinct objects into k non-empty distinct groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 students and 3 courses -/
def num_students : ℕ := 5
def num_courses : ℕ := 3

/-- Each student signs up for exactly one course -/
axiom one_course_per_student : distribute num_students num_courses > 0

/-- Each course must have at least one student enrolled -/
axiom non_empty_courses : ∀ (i : Fin num_courses), ∃ (student : Fin num_students), sorry

/-- The number of different registration methods is 150 -/
theorem registration_methods : distribute num_students num_courses = 150 := by sorry

end NUMINAMATH_CALUDE_registration_methods_l2021_202136


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l2021_202146

theorem polynomial_factor_theorem (c q k : ℝ) : 
  (∀ x, 3 * x^3 + c * x + 8 = (x^2 + q * x + 2) * (3 * x + k)) →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l2021_202146


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2021_202172

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 20 (2*x - 1) = Nat.choose 20 (x + 3)) → (x = 4 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2021_202172


namespace NUMINAMATH_CALUDE_dragon_legs_count_l2021_202185

/-- Represents the number of legs per centipede -/
def centipede_legs : ℕ := 40

/-- Represents the number of heads per dragon -/
def dragon_heads : ℕ := 9

/-- Represents the total number of heads in the cage -/
def total_heads : ℕ := 50

/-- Represents the total number of legs in the cage -/
def total_legs : ℕ := 220

/-- Represents the number of centipedes in the cage -/
def num_centipedes : ℕ := 40

/-- Represents the number of dragons in the cage -/
def num_dragons : ℕ := total_heads - num_centipedes

/-- Theorem stating that each dragon has 4 legs -/
theorem dragon_legs_count : 
  ∃ (dragon_legs : ℕ), 
    dragon_legs = 4 ∧ 
    num_centipedes * centipede_legs + num_dragons * dragon_legs = total_legs :=
sorry

end NUMINAMATH_CALUDE_dragon_legs_count_l2021_202185


namespace NUMINAMATH_CALUDE_complex_magnitude_plus_fraction_l2021_202106

theorem complex_magnitude_plus_fraction :
  Complex.abs (3/4 - 3*Complex.I) + 5/12 = (9*Real.sqrt 17 + 5) / 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_plus_fraction_l2021_202106


namespace NUMINAMATH_CALUDE_unique_pairs_satisfying_W_l2021_202115

def W (x : ℕ) : ℤ := x^4 - 3*x^3 + 5*x^2 - 9*x

theorem unique_pairs_satisfying_W :
  ∀ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ W a = W b ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_pairs_satisfying_W_l2021_202115


namespace NUMINAMATH_CALUDE_ellipse_condition_l2021_202150

/-- The equation of the graph -/
def graph_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 - 6 * x + 27 * y = k

/-- The condition for a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -93/4

/-- Theorem: The graph is a non-degenerate ellipse iff k > -93/4 -/
theorem ellipse_condition :
  ∀ k, (∃ x y, graph_equation x y k) ↔ is_non_degenerate_ellipse k :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2021_202150


namespace NUMINAMATH_CALUDE_sample_size_theorem_l2021_202126

theorem sample_size_theorem (N : ℕ) (sample_size : ℕ) (prob : ℚ) : 
  sample_size = 30 → prob = 1/4 → N * prob = sample_size → N = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l2021_202126


namespace NUMINAMATH_CALUDE_shoes_price_calculation_l2021_202117

/-- The price of shoes after a markup followed by a discount -/
def monday_price (thursday_price : ℝ) (friday_markup : ℝ) (monday_discount : ℝ) : ℝ :=
  thursday_price * (1 + friday_markup) * (1 - monday_discount)

/-- Theorem stating that the Monday price is $50.60 given the specified conditions -/
theorem shoes_price_calculation :
  monday_price 50 0.15 0.12 = 50.60 := by
  sorry

#eval monday_price 50 0.15 0.12

end NUMINAMATH_CALUDE_shoes_price_calculation_l2021_202117


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l2021_202121

theorem third_root_of_cubic (a b : ℚ) : 
  (∀ x : ℚ, a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 11/5) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l2021_202121


namespace NUMINAMATH_CALUDE_elder_person_age_l2021_202175

theorem elder_person_age (y e : ℕ) : 
  e = y + 16 →                     -- The ages differ by 16 years
  e - 6 = 3 * (y - 6) →            -- 6 years ago, elder was 3 times younger's age
  e = 30                           -- Elder's present age is 30
  := by sorry

end NUMINAMATH_CALUDE_elder_person_age_l2021_202175


namespace NUMINAMATH_CALUDE_roses_given_l2021_202134

-- Define the total number of students
def total_students : ℕ := 28

-- Define the relationship between flowers
def flower_relationship (daffodils roses tulips : ℕ) : Prop :=
  roses = 4 * daffodils ∧ tulips = 10 * roses

-- Define the total number of flowers given
def total_flowers (boys girls : ℕ) : ℕ := boys * girls

-- Define the constraint that the total number of students is the sum of boys and girls
def student_constraint (boys girls : ℕ) : Prop :=
  boys + girls = total_students

-- Theorem statement
theorem roses_given (boys girls daffodils roses tulips : ℕ) :
  student_constraint boys girls →
  flower_relationship daffodils roses tulips →
  total_flowers boys girls = daffodils + roses + tulips →
  roses = 16 := by
  sorry

end NUMINAMATH_CALUDE_roses_given_l2021_202134


namespace NUMINAMATH_CALUDE_largest_x_value_l2021_202196

theorem largest_x_value : ∃ (x : ℝ), 
  (∀ (z : ℝ), (|z - 3| = 8 ∧ 2*z + 1 ≤ 25) → z ≤ x) ∧ 
  |x - 3| = 8 ∧ 
  2*x + 1 ≤ 25 ∧
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_largest_x_value_l2021_202196


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2021_202143

/-- Sum of the first n terms of a geometric sequence -/
def geometric_sum (a₀ r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence
    with first term 1/3 and common ratio 1/3 is 6560/19683 -/
theorem geometric_sum_first_8_terms :
  geometric_sum (1/3) (1/3) 8 = 6560/19683 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2021_202143


namespace NUMINAMATH_CALUDE_coefficient_of_y_l2021_202142

theorem coefficient_of_y (x y a : ℝ) : 
  7 * x + y = 19 → 
  x + a * y = 1 → 
  2 * x + y = 5 → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l2021_202142


namespace NUMINAMATH_CALUDE_ten_times_average_letters_l2021_202105

def elida_letters : ℕ := 5

def adrianna_letters : ℕ := 2 * elida_letters - 2

def average_letters : ℚ := (elida_letters + adrianna_letters) / 2

theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end NUMINAMATH_CALUDE_ten_times_average_letters_l2021_202105


namespace NUMINAMATH_CALUDE_square_root_of_four_l2021_202187

theorem square_root_of_four : ∃ (x : ℝ), x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2021_202187


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2021_202107

theorem sqrt_sum_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2021_202107


namespace NUMINAMATH_CALUDE_bob_gardening_project_cost_l2021_202191

/-- The total cost of Bob's gardening project --/
def gardening_project_cost 
  (num_rose_bushes : ℕ) 
  (cost_per_rose_bush : ℕ) 
  (gardener_hourly_rate : ℕ) 
  (gardener_hours_per_day : ℕ) 
  (gardener_work_days : ℕ) 
  (soil_volume : ℕ) 
  (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush + 
  gardener_hourly_rate * gardener_hours_per_day * gardener_work_days + 
  soil_volume * soil_cost_per_unit

/-- Theorem stating that the total cost of Bob's gardening project is $4100 --/
theorem bob_gardening_project_cost : 
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_bob_gardening_project_cost_l2021_202191


namespace NUMINAMATH_CALUDE_juggler_count_l2021_202165

theorem juggler_count (balls_per_juggler : ℕ) (total_balls : ℕ) (h1 : balls_per_juggler = 6) (h2 : total_balls = 2268) :
  total_balls / balls_per_juggler = 378 := by
  sorry

end NUMINAMATH_CALUDE_juggler_count_l2021_202165


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_condition_l2021_202170

/-- A geometric sequence with first term a and common ratio q -/
def GeometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Predicate for an increasing sequence -/
def IsIncreasing (f : ℕ → ℝ) : Prop := ∀ n : ℕ, f n ≤ f (n + 1)

theorem neither_necessary_nor_sufficient_condition
  (a q : ℝ) :
  ¬(((a * q > 0) ↔ IsIncreasing (GeometricSequence a q))) :=
sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_condition_l2021_202170


namespace NUMINAMATH_CALUDE_product_x_z_l2021_202135

-- Define the parallelogram EFGH
structure Parallelogram :=
  (E F G H : ℝ × ℝ)
  (is_parallelogram : True)  -- This is a placeholder for the parallelogram property

-- Define the lengths of the sides
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem product_x_z (EFGH : Parallelogram) 
  (EF_length : side_length EFGH.E EFGH.F = 52)
  (FG_length : ∃ z, side_length EFGH.F EFGH.G = 4 * z^2 + 4)
  (GH_length : ∃ x, side_length EFGH.G EFGH.H = 5 * x + 6)
  (HE_length : side_length EFGH.H EFGH.E = 16) :
  ∃ x z, x * z = 46 * Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_product_x_z_l2021_202135


namespace NUMINAMATH_CALUDE_pen_count_problem_l2021_202161

theorem pen_count_problem (total_pens : ℕ) (difference : ℕ) 
  (h1 : total_pens = 140)
  (h2 : difference = 20) :
  ∃ (ballpoint_pens fountain_pens : ℕ),
    ballpoint_pens + fountain_pens = total_pens ∧
    ballpoint_pens + difference = fountain_pens ∧
    ballpoint_pens = 60 ∧
    fountain_pens = 80 := by
  sorry

end NUMINAMATH_CALUDE_pen_count_problem_l2021_202161


namespace NUMINAMATH_CALUDE_cubic_arithmetic_progression_complex_root_l2021_202151

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_form_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), p.a * (r - d)^3 + p.b * (r - d)^2 + p.c * (r - d) + p.d = 0 ∧
                p.a * r^3 + p.b * r^2 + p.c * r + p.d = 0 ∧
                p.a * (r + d)^3 + p.b * (r + d)^2 + p.c * (r + d) + p.d = 0

/-- One of the roots of a cubic polynomial is complex -/
def has_complex_root (p : CubicPolynomial) : Prop :=
  ∃ (z : ℂ), z.im ≠ 0 ∧ p.a * z^3 + p.b * z^2 + p.c * z + p.d = 0

/-- The main theorem -/
theorem cubic_arithmetic_progression_complex_root :
  ∃! (a : ℝ), roots_form_arithmetic_progression { a := 1, b := -9, c := 30, d := a } ∧
               has_complex_root { a := 1, b := -9, c := 30, d := a } ∧
               a = -12 := by sorry

end NUMINAMATH_CALUDE_cubic_arithmetic_progression_complex_root_l2021_202151


namespace NUMINAMATH_CALUDE_congruence_iff_divisible_l2021_202192

theorem congruence_iff_divisible (a b m : ℤ) : a ≡ b [ZMOD m] ↔ m ∣ (a - b) := by sorry

end NUMINAMATH_CALUDE_congruence_iff_divisible_l2021_202192


namespace NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l2021_202111

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve_C a x y ∧ line_l x y}

-- Theorem statement
theorem intersection_distance_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points a ∧ B ∈ intersection_points a ∧ 
   A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l2021_202111


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l2021_202147

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

def U (n : ℕ) : ℤ := 2 * T n

theorem sum_of_specific_terms : U 13 + T 25 + U 40 = -13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l2021_202147


namespace NUMINAMATH_CALUDE_third_month_sale_l2021_202195

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sale_month1 : ℕ := 6535
def sale_month2 : ℕ := 6927
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 4891

theorem third_month_sale :
  ∃ (sale_month3 : ℕ),
    sale_month3 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l2021_202195


namespace NUMINAMATH_CALUDE_number_of_divisors_of_M_l2021_202112

def M : ℕ := 2^6 * 3^4 * 5^2 * 7^2 * 11^1

theorem number_of_divisors_of_M : (Nat.divisors M).card = 630 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_M_l2021_202112


namespace NUMINAMATH_CALUDE_odd_sum_ends_with_1379_l2021_202113

theorem odd_sum_ends_with_1379 (S : Finset ℕ) 
  (h1 : S.card = 10000)
  (h2 : ∀ n ∈ S, Odd n)
  (h3 : ∀ n ∈ S, ¬(5 ∣ n)) :
  ∃ T ⊆ S, (T.sum id) % 10000 = 1379 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_ends_with_1379_l2021_202113


namespace NUMINAMATH_CALUDE_committee_selection_l2021_202145

theorem committee_selection (n : ℕ) (k : ℕ) : n = 9 → k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2021_202145


namespace NUMINAMATH_CALUDE_pencil_problem_l2021_202168

/-- Given the initial number of pencils, the number of containers, and the number of pencils that can be evenly distributed after receiving more, calculate the number of additional pencils received. -/
def additional_pencils (initial : ℕ) (containers : ℕ) (even_distribution : ℕ) : ℕ :=
  containers * even_distribution - initial

/-- Prove that given the specific conditions in the problem, the number of additional pencils is 30. -/
theorem pencil_problem : additional_pencils 150 5 36 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencil_problem_l2021_202168


namespace NUMINAMATH_CALUDE_completing_square_result_l2021_202183

theorem completing_square_result (x : ℝ) : 
  (x^2 - 6*x - 8 = 0) ↔ ((x - 3)^2 = 17) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2021_202183


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l2021_202188

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15) ∧ 
  x₁ ≠ x₂ ∧ 
  |x₁ - x₂| = 30 := by
sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l2021_202188


namespace NUMINAMATH_CALUDE_quadratic_comparison_l2021_202153

/-- Given a quadratic function f(x) = a(x-1)^2 + 3 where a < 0,
    if f(-1) = y₁ and f(2) = y₂, then y₁ < y₂ -/
theorem quadratic_comparison (a y₁ y₂ : ℝ) (ha : a < 0) 
    (h1 : y₁ = a * (-1 - 1)^2 + 3)
    (h2 : y₂ = a * (2 - 1)^2 + 3) :
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l2021_202153


namespace NUMINAMATH_CALUDE_abdul_binh_age_difference_l2021_202176

/- Define Susie's age -/
variable (S : ℤ)

/- Define Abdul's age in terms of Susie's -/
def A : ℤ := S + 9

/- Define Binh's age in terms of Susie's -/
def B : ℤ := S + 2

/- Theorem statement -/
theorem abdul_binh_age_difference : A - B = 7 := by
  sorry

end NUMINAMATH_CALUDE_abdul_binh_age_difference_l2021_202176


namespace NUMINAMATH_CALUDE_find_certain_number_l2021_202169

theorem find_certain_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + x + 45) / 3) + 5 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l2021_202169


namespace NUMINAMATH_CALUDE_marys_remaining_money_equals_50_minus_12p_l2021_202140

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def marys_remaining_money (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary's remaining money is equal to 50 - 12p -/
theorem marys_remaining_money_equals_50_minus_12p (p : ℝ) :
  marys_remaining_money p = 50 - 12 * p := by
  sorry

end NUMINAMATH_CALUDE_marys_remaining_money_equals_50_minus_12p_l2021_202140


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2021_202171

/-- Represents the outcome of a single coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of 8 coin flips -/
def CoinSequence := Vector CoinFlip 8

/-- Checks if a given sequence has exactly one pair of consecutive heads and one pair of consecutive tails -/
def hasExactlyOnePairEach (seq : CoinSequence) : Bool :=
  sorry

/-- The total number of possible 8-flip sequences -/
def totalSequences : Nat := 256

/-- The number of favorable sequences (with exactly one pair each of heads and tails) -/
def favorableSequences : Nat := 18

/-- The probability of getting exactly one pair each of heads and tails in 8 flips -/
def probability : Rat := favorableSequences / totalSequences

theorem coin_flip_probability :
  probability = 9 / 128 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2021_202171


namespace NUMINAMATH_CALUDE_total_candy_caught_l2021_202124

def candy_problem (tabitha : ℕ) (stan : ℕ) (julie : ℕ) (carlos : ℕ) : Prop :=
  tabitha = 22 ∧
  stan = 13 ∧
  julie = tabitha / 2 ∧
  carlos = 2 * stan ∧
  tabitha + stan + julie + carlos = 72

theorem total_candy_caught :
  ∃ (tabitha stan julie carlos : ℕ), candy_problem tabitha stan julie carlos :=
by
  sorry

end NUMINAMATH_CALUDE_total_candy_caught_l2021_202124


namespace NUMINAMATH_CALUDE_sum_of_products_l2021_202137

theorem sum_of_products : 
  12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l2021_202137


namespace NUMINAMATH_CALUDE_expression_evaluation_l2021_202186

theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/5
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2021_202186


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2021_202109

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2021_202109


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l2021_202179

/-- Calculates the earnings for a single cycle of lawn mowing -/
def cycle_earnings : ℕ := 5 + 10 + 15 + 20 + 25 + 30

/-- Calculates the total earnings for a given number of hours -/
def total_earnings (hours : ℕ) : ℕ := (hours / 6) * cycle_earnings

/-- The amount Jason borrowed is equal to his earnings from 24 hours of lawn mowing -/
theorem jason_borrowed_amount : total_earnings 24 = 420 := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l2021_202179


namespace NUMINAMATH_CALUDE_distance_product_l2021_202159

theorem distance_product (b₁ b₂ : ℝ) : 
  (∀ b : ℝ, (3*b - 5)^2 + (b - 3)^2 = 39 → b = b₁ ∨ b = b₂) →
  (3*b₁ - 5)^2 + (b₁ - 3)^2 = 39 →
  (3*b₂ - 5)^2 + (b₂ - 3)^2 = 39 →
  b₁ * b₂ = -(9/16) := by
sorry

end NUMINAMATH_CALUDE_distance_product_l2021_202159


namespace NUMINAMATH_CALUDE_lisa_walking_time_l2021_202160

/-- Given Lisa's walking speed and total distance over two days, prove she walks for 1 hour each day -/
theorem lisa_walking_time 
  (speed : ℝ)              -- Lisa's walking speed in meters per minute
  (total_distance : ℝ)     -- Total distance Lisa walks in two days
  (h1 : speed = 10)        -- Lisa walks 10 meters each minute
  (h2 : total_distance = 1200) -- Lisa walks 1200 meters in two days
  : (total_distance / 2) / speed / 60 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lisa_walking_time_l2021_202160


namespace NUMINAMATH_CALUDE_no_solution_cube_equation_mod_9_l2021_202108

theorem no_solution_cube_equation_mod_9 :
  ∀ (x y z : ℤ), (x^3 + y^3) % 9 ≠ (z^3 + 4) % 9 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cube_equation_mod_9_l2021_202108


namespace NUMINAMATH_CALUDE_calculate_savings_l2021_202189

/-- Given total expenses and savings rate, calculate the amount saved -/
theorem calculate_savings (total_expenses : ℝ) (savings_rate : ℝ) : 
  total_expenses = 24150 ∧ savings_rate = 0.1 → 
  ∃ amount_saved : ℝ, abs (amount_saved - 2683.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_calculate_savings_l2021_202189


namespace NUMINAMATH_CALUDE_cookie_box_lasts_nine_days_l2021_202166

/-- The number of days a box of cookies lasts for Jackson's sons -/
def cookie_box_duration (oldest_son_cookies : ℕ) (youngest_son_cookies : ℕ) (total_cookies : ℕ) : ℕ :=
  total_cookies / (oldest_son_cookies + youngest_son_cookies)

theorem cookie_box_lasts_nine_days :
  cookie_box_duration 4 2 54 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_box_lasts_nine_days_l2021_202166


namespace NUMINAMATH_CALUDE_circle_triangle_areas_l2021_202101

theorem circle_triangle_areas (a b c : ℝ) (A B C : ℝ) : 
  a = 15 → b = 20 → c = 25 →
  a^2 + b^2 = c^2 →
  A > 0 → B > 0 → C > 0 →
  C > A ∧ C > B →
  A + B + (1/2 * a * b) = C := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_areas_l2021_202101


namespace NUMINAMATH_CALUDE_y_value_l2021_202177

theorem y_value : ∀ y : ℚ, (1 / 3 - 1 / 4 : ℚ) = 4 / y → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2021_202177


namespace NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l2021_202103

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on a 2D grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Checks if a triangle is acute -/
def isAcute (t : GridTriangle) : Prop := sorry

/-- Checks if a point is inside or on the sides of a triangle -/
def isInsideOrOnSides (p : GridPoint) (t : GridTriangle) : Prop := sorry

/-- Main theorem: If a triangle on a grid is acute, there exists a grid point 
    (other than its vertices) inside or on its sides -/
theorem acute_triangle_contains_grid_point (t : GridTriangle) :
  isAcute t → ∃ p : GridPoint, p ≠ t.A ∧ p ≠ t.B ∧ p ≠ t.C ∧ isInsideOrOnSides p t := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l2021_202103


namespace NUMINAMATH_CALUDE_linear_coefficient_of_equation_l2021_202162

theorem linear_coefficient_of_equation : ∃ (a b c : ℝ), 
  (∀ x, (2*x + 1)*(x - 3) = x^2 + 1) → 
  (∀ x, a*x^2 + b*x + c = 0) ∧ 
  b = -5 := by
  sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_equation_l2021_202162


namespace NUMINAMATH_CALUDE_cindy_hit_eight_l2021_202163

-- Define the set of players
inductive Player : Type
| Alice : Player
| Ben : Player
| Cindy : Player
| Dave : Player
| Ellen : Player

-- Define the score function
def score : Player → ℕ
| Player.Alice => 10
| Player.Ben => 6
| Player.Cindy => 9
| Player.Dave => 15
| Player.Ellen => 19

-- Define the set of possible scores on the dartboard
def dartboard_scores : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define a function to check if a player's score can be composed of two different dartboard scores
def valid_score (p : Player) : Prop :=
  ∃ (a b : ℕ), a ∈ dartboard_scores ∧ b ∈ dartboard_scores ∧ a ≠ b ∧ a + b = score p

-- Theorem: Cindy is the only player who could have hit the section worth 8 points
theorem cindy_hit_eight :
  (∀ p : Player, valid_score p) →
  (∃! p : Player, ∃ (a : ℕ), a ∈ dartboard_scores ∧ a ≠ 8 ∧ a + 8 = score p) ∧
  (∃ (a : ℕ), a ∈ dartboard_scores ∧ a ≠ 8 ∧ a + 8 = score Player.Cindy) :=
by sorry

end NUMINAMATH_CALUDE_cindy_hit_eight_l2021_202163


namespace NUMINAMATH_CALUDE_min_value_bn_Sn_l2021_202123

def a (n : ℕ) : ℕ := n * (n + 1)

def S (n : ℕ) : ℚ := 1 - 1 / (n + 1)

def b (n : ℕ) : ℤ := n - 8

theorem min_value_bn_Sn :
  (∀ n : ℕ, (b n : ℚ) * S n ≥ -4) ∧
  (∃ n : ℕ, (b n : ℚ) * S n = -4) :=
sorry

end NUMINAMATH_CALUDE_min_value_bn_Sn_l2021_202123


namespace NUMINAMATH_CALUDE_system_two_solutions_l2021_202138

/-- The system of equations has exactly two solutions if and only if a = 49 or a = 289 -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ (|x| - 8)^2 + (|y| - 15)^2 = a) ↔ 
  (a = 49 ∨ a = 289) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l2021_202138


namespace NUMINAMATH_CALUDE_total_combinations_l2021_202193

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 4

/-- Theorem: The total number of combinations of color and painting method is 20 -/
theorem total_combinations : num_colors * num_methods = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_l2021_202193


namespace NUMINAMATH_CALUDE_books_obtained_l2021_202114

/-- Given an initial number of books and a final number of books,
    calculate the number of additional books obtained. -/
def additional_books (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that for the given initial and final book counts,
    the number of additional books is 23. -/
theorem books_obtained (initial : ℕ) (final : ℕ)
    (h1 : initial = 54)
    (h2 : final = 77) :
    additional_books initial final = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_obtained_l2021_202114


namespace NUMINAMATH_CALUDE_dihedral_angle_inscribed_spheres_l2021_202190

/-- Given two spheres inscribed in a dihedral angle, this theorem proves
    the relationship between the spheres' radii, their position, and the
    measure of the dihedral angle. -/
theorem dihedral_angle_inscribed_spheres 
  (R₁ R₂ : ℝ) -- Radii of the two spheres
  (h_touch : R₁ + R₂ > 0) -- The spheres touch (implied by positive sum of radii)
  (h_ratio : R₁ = 1.5 * R₂) -- Ratio of radii
  (h_angle : Real.cos (45 * π / 180) = Real.sqrt (1 / 2)) -- 45° angle with edge
  : Real.cos (θ / 2) = Real.sqrt ((1 + Real.sqrt (1 / 2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_inscribed_spheres_l2021_202190


namespace NUMINAMATH_CALUDE_min_value_theorem_l2021_202173

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  2 ≤ b / a + 3 / (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2021_202173


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2021_202118

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 145 ∧ 
  train_speed_kmh = 45 ∧ 
  bridge_length = 230 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2021_202118


namespace NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l2021_202174

/-- Represents the number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 4 indistinguishable boxes is 495 -/
theorem distribute_seven_balls_four_boxes : distribute_balls 7 4 = 495 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l2021_202174


namespace NUMINAMATH_CALUDE_arithmetic_sequence_of_squares_l2021_202181

theorem arithmetic_sequence_of_squares (a b c x y : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- a, b, c are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- a, b, c are distinct
  b - a = c - b ∧ -- a, b, c form an arithmetic sequence
  x^2 = a * b ∧ -- x is the geometric mean of a and b
  y^2 = b * c -- y is the geometric mean of b and c
  → 
  (y^2 - b^2 = b^2 - x^2) ∧ -- x^2, b^2, y^2 form an arithmetic sequence
  ¬(y^2 / b^2 = b^2 / x^2) -- x^2, b^2, y^2 do not form a geometric sequence
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_of_squares_l2021_202181


namespace NUMINAMATH_CALUDE_zunyi_conference_highest_temp_l2021_202194

/-- Given the lowest temperature and maximum temperature difference of a day,
    calculate the highest temperature of that day. -/
def highest_temperature (lowest_temp max_diff : ℝ) : ℝ :=
  lowest_temp + max_diff

/-- Theorem stating that given the specific conditions of the problem,
    the highest temperature of the day is 22°C. -/
theorem zunyi_conference_highest_temp :
  highest_temperature 18 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_zunyi_conference_highest_temp_l2021_202194


namespace NUMINAMATH_CALUDE_reciprocal_sum_and_product_l2021_202116

theorem reciprocal_sum_and_product (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 12) (h4 : x * y = 32) : 
  1 / x + 1 / y = 3 / 8 ∧ 1 / (x * y) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_and_product_l2021_202116


namespace NUMINAMATH_CALUDE_remaining_trees_correct_l2021_202129

/-- The number of oak trees remaining in the park after cutting down damaged trees -/
def remaining_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining trees is correct -/
theorem remaining_trees_correct (initial : ℕ) (cut_down : ℕ) 
  (h : cut_down ≤ initial) : 
  remaining_trees initial cut_down = initial - cut_down :=
by sorry

end NUMINAMATH_CALUDE_remaining_trees_correct_l2021_202129


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2021_202158

theorem inequality_solution_set (x : ℝ) : 5 * x + 1 ≥ 3 * x - 5 ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2021_202158


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2021_202100

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 3) - 2
  f (-3) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2021_202100


namespace NUMINAMATH_CALUDE_missing_integers_count_l2021_202182

theorem missing_integers_count (n : ℕ) (h : n = 2017) : 
  n - (n - n / 3 + n / 6 - n / 54) = 373 :=
by sorry

end NUMINAMATH_CALUDE_missing_integers_count_l2021_202182


namespace NUMINAMATH_CALUDE_black_friday_sales_l2021_202132

/-- Calculates the number of televisions sold after a given number of years,
    given an initial sale and yearly increase. -/
def televisionsSold (initialSale : ℕ) (yearlyIncrease : ℕ) (years : ℕ) : ℕ :=
  initialSale + yearlyIncrease * years

/-- Theorem stating that given an initial sale of 327 televisions and
    an increase of 50 televisions per year, the number of televisions
    sold after 3 years will be 477. -/
theorem black_friday_sales : televisionsSold 327 50 3 = 477 := by
  sorry

end NUMINAMATH_CALUDE_black_friday_sales_l2021_202132


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2021_202178

/-- The longest segment in a cylinder with radius 5 and height 12 is 2√61 -/
theorem longest_segment_in_cylinder : 
  let r : ℝ := 5
  let h : ℝ := 12
  let longest_segment := Real.sqrt ((2 * r) ^ 2 + h ^ 2)
  longest_segment = 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2021_202178


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2021_202197

def population_size : Nat := 1000
def num_groups : Nat := 10
def sample_size : Nat := 10

def systematic_sample (x : Nat) : List Nat :=
  List.range num_groups |>.map (fun k => (x + 33 * k) % 100)

def last_two_digits (n : Nat) : Nat := n % 100

theorem systematic_sampling_theorem :
  (systematic_sample 24 = [24, 57, 90, 23, 56, 89, 22, 55, 88, 21]) ∧
  (∀ x : Nat, x < population_size →
    (∃ n ∈ systematic_sample x, last_two_digits n = 87) →
    x ∈ [21, 22, 23, 54, 55, 56, 87, 88, 89, 90]) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2021_202197


namespace NUMINAMATH_CALUDE_mean_median_difference_l2021_202154

/-- Represents the score distribution in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_60_percent : ℚ
  score_75_percent : ℚ
  score_85_percent : ℚ
  score_90_percent : ℚ
  score_100_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (dist : ScoreDistribution) : ℚ :=
  (60 * dist.score_60_percent + 75 * dist.score_75_percent + 
   85 * dist.score_85_percent + 90 * dist.score_90_percent + 
   100 * dist.score_100_percent) / 1

/-- Calculates the median score given a score distribution -/
def median_score (dist : ScoreDistribution) : ℚ := 85

/-- Theorem stating the difference between mean and median scores -/
theorem mean_median_difference (dist : ScoreDistribution) : 
  dist.total_students = 25 ∧
  dist.score_60_percent = 15/100 ∧
  dist.score_75_percent = 20/100 ∧
  dist.score_85_percent = 30/100 ∧
  dist.score_90_percent = 20/100 ∧
  dist.score_100_percent = 15/100 →
  mean_score dist - median_score dist = 8/10 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2021_202154


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2021_202102

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2021_202102


namespace NUMINAMATH_CALUDE_log_sum_and_product_imply_average_l2021_202164

theorem log_sum_and_product_imply_average (x y : ℝ) : 
  x > 0 → y > 0 → (Real.log x / Real.log y + Real.log y / Real.log x = 4) → x * y = 81 → 
  (x + y) / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_log_sum_and_product_imply_average_l2021_202164


namespace NUMINAMATH_CALUDE_target_annual_revenue_l2021_202104

/-- Calculates the target annual revenue for a shoe company given their current monthly sales and required monthly increase. -/
theorem target_annual_revenue
  (current_monthly_sales : ℕ)
  (required_monthly_increase : ℕ)
  (months_per_year : ℕ)
  (h1 : current_monthly_sales = 4000)
  (h2 : required_monthly_increase = 1000)
  (h3 : months_per_year = 12) :
  (current_monthly_sales + required_monthly_increase) * months_per_year = 60000 :=
by
  sorry

#check target_annual_revenue

end NUMINAMATH_CALUDE_target_annual_revenue_l2021_202104


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l2021_202119

def total_marbles : ℕ := 12
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 4
def num_draws : ℕ := 8
def num_green_draws : ℕ := 5

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

theorem probability_five_green_marbles :
  (Nat.choose num_draws num_green_draws : ℚ) * 
  (prob_green ^ num_green_draws) * 
  (prob_purple ^ (num_draws - num_green_draws)) = 1792 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l2021_202119


namespace NUMINAMATH_CALUDE_wilted_flowers_count_l2021_202130

def initial_flowers : ℕ := 88
def flowers_per_bouquet : ℕ := 5
def bouquets_made : ℕ := 8

theorem wilted_flowers_count : 
  initial_flowers - (flowers_per_bouquet * bouquets_made) = 48 := by
  sorry

end NUMINAMATH_CALUDE_wilted_flowers_count_l2021_202130


namespace NUMINAMATH_CALUDE_complement_of_union_l2021_202155

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2021_202155


namespace NUMINAMATH_CALUDE_original_ghee_quantity_l2021_202133

/-- Represents the composition of ghee -/
structure GheeComposition where
  pure : Rat
  vanaspati : Rat
  impurities : Rat

/-- The original ghee composition -/
def originalComposition : GheeComposition :=
  { pure := 40/100, vanaspati := 30/100, impurities := 30/100 }

/-- The desired final ghee composition -/
def desiredComposition : GheeComposition :=
  { pure := 45/100, vanaspati := 25/100, impurities := 30/100 }

/-- The amount of pure ghee added (in kg) -/
def addedPureGhee : Rat := 20

/-- Theorem stating the original quantity of blended ghee -/
theorem original_ghee_quantity : 
  ∃ (x : Rat), 
    (originalComposition.pure * x + addedPureGhee = desiredComposition.pure * (x + addedPureGhee)) ∧
    (originalComposition.vanaspati * x = desiredComposition.vanaspati * (x + addedPureGhee)) ∧
    x = 220 := by
  sorry

end NUMINAMATH_CALUDE_original_ghee_quantity_l2021_202133


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l2021_202127

def total_cakes (initial : ℕ) (extra : ℕ) : ℕ :=
  initial + extra

theorem baker_cakes_theorem (initial : ℕ) (extra : ℕ) :
  total_cakes initial extra = initial + extra := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l2021_202127


namespace NUMINAMATH_CALUDE_bananas_lost_l2021_202156

theorem bananas_lost (initial_bananas final_bananas : ℕ) 
  (h1 : initial_bananas = 47) 
  (h2 : final_bananas = 2) : 
  initial_bananas - final_bananas = 45 := by
sorry

end NUMINAMATH_CALUDE_bananas_lost_l2021_202156


namespace NUMINAMATH_CALUDE_negation_of_existence_geq_l2021_202144

theorem negation_of_existence_geq (p : Prop) :
  (¬ (∃ x : ℝ, x^2 ≥ x)) ↔ (∀ x : ℝ, x^2 < x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_geq_l2021_202144


namespace NUMINAMATH_CALUDE_unique_prime_p_l2021_202139

theorem unique_prime_p : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (3 * p^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_p_l2021_202139


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2021_202141

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 9, 
    if the remainder when divided by x - 2 is 17, 
    then the remainder when divided by x + 2 is 33. -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D*x^4 + E*x^2 + F*x + 9
  (q 2 = 17) → (q (-2) = 33) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2021_202141


namespace NUMINAMATH_CALUDE_problem_solution_l2021_202199

theorem problem_solution :
  (12 / 60 = 0.2) ∧
  (0.2 = 4 / 20) ∧
  (0.2 = 20 / 100) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2021_202199


namespace NUMINAMATH_CALUDE_product_terminal_zeros_l2021_202149

/-- The number of terminal zeros in a positive integer -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 50 and 480 -/
def product : ℕ := 50 * 480

theorem product_terminal_zeros :
  terminalZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_terminal_zeros_l2021_202149


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l2021_202128

theorem smallest_c_for_inequality : ∃ c : ℕ, (∀ k : ℕ, 27^k > 3^24 → c ≤ k) ∧ 27^c > 3^24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l2021_202128


namespace NUMINAMATH_CALUDE_tim_balloon_count_l2021_202148

/-- Calculates the number of Tim's balloons given Dan's balloons and the multiplier -/
def tims_balloons (dans_balloons : ℕ) (multiplier : ℕ) : ℕ :=
  dans_balloons * multiplier

/-- Theorem: Given Dan has 59 violet balloons and Tim has 11 times more,
    Tim has 649 violet balloons -/
theorem tim_balloon_count :
  tims_balloons 59 11 = 649 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l2021_202148


namespace NUMINAMATH_CALUDE_five_not_in_A_and_B_l2021_202180

universe u

def U : Set Nat := {1, 2, 3, 4, 5}

theorem five_not_in_A_and_B
  (A B : Set Nat)
  (h_subset : A ⊆ U ∧ B ⊆ U)
  (h_inter : A ∩ B = {2, 4})
  (h_union : A ∪ B = {1, 2, 3, 4}) :
  5 ∉ A ∧ 5 ∉ B := by
  sorry


end NUMINAMATH_CALUDE_five_not_in_A_and_B_l2021_202180


namespace NUMINAMATH_CALUDE_min_value_problem_l2021_202110

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2/y = 3) :
  ∃ (m : ℝ), m = 8/3 ∧ ∀ z, z = 2/x + y → z ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2021_202110


namespace NUMINAMATH_CALUDE_words_per_page_smaller_type_l2021_202131

/-- Calculates words per page in smaller type given article details -/
def wordsPerPageSmallerType (totalWords : ℕ) (totalPages : ℕ) (smallerTypePages : ℕ) (wordsPerPageLargerType : ℕ) : ℕ :=
  let largerTypePages := totalPages - smallerTypePages
  let wordsInLargerType := largerTypePages * wordsPerPageLargerType
  let wordsInSmallerType := totalWords - wordsInLargerType
  wordsInSmallerType / smallerTypePages

/-- Proves that words per page in smaller type is 2400 for given article details -/
theorem words_per_page_smaller_type :
  wordsPerPageSmallerType 48000 21 17 1800 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_smaller_type_l2021_202131


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l2021_202167

theorem angle_terminal_side_point (α : Real) (a : Real) :
  (∃ (x y : Real), x = a ∧ y = -1 ∧ (Real.tan α) * x = y) →
  Real.tan α = -1/2 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l2021_202167


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l2021_202184

def f (x : ℝ) : ℝ := -3 * (x - 2)^2

theorem vertex_of_quadratic :
  ∃ (a : ℝ), a < 0 ∧ ∀ (x : ℝ), f x = a * (x - 2)^2 ∧ f 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l2021_202184


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_72_l2021_202122

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def six_digit_number (x y : ℕ) : ℕ := 640000 + x * 1000 + 720 + y

theorem two_digit_divisible_by_72 :
  ∀ x y : ℕ, is_two_digit (10 * x + y) →
  (six_digit_number x y ∣ 72) →
  (x = 8 ∧ y = 0) ∨ (x = 9 ∧ y = 8) :=
sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_72_l2021_202122


namespace NUMINAMATH_CALUDE_cd_cost_l2021_202157

theorem cd_cost (num_films : ℕ) (num_books : ℕ) (num_cds : ℕ) 
  (film_cost : ℕ) (book_cost : ℕ) (total_spent : ℕ) :
  num_films = 9 →
  num_books = 4 →
  num_cds = 6 →
  film_cost = 5 →
  book_cost = 4 →
  total_spent = 79 →
  (total_spent - (num_films * film_cost + num_books * book_cost)) / num_cds = 3 := by
  sorry

#eval (79 - (9 * 5 + 4 * 4)) / 6

end NUMINAMATH_CALUDE_cd_cost_l2021_202157


namespace NUMINAMATH_CALUDE_coefficient_of_x_fifth_power_l2021_202152

theorem coefficient_of_x_fifth_power (x : ℝ) : 
  ∃ (a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ), 
    (x - 2) * (x + 2)^5 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧ 
    a₅ = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fifth_power_l2021_202152


namespace NUMINAMATH_CALUDE_cupcakes_frosted_in_ten_minutes_l2021_202198

def mark_rate : ℚ := 1 / 15
def julia_rate : ℚ := 1 / 40
def total_time : ℚ := 10 * 60  -- 10 minutes in seconds

theorem cupcakes_frosted_in_ten_minutes : 
  ⌊(mark_rate + julia_rate) * total_time⌋ = 55 := by sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_in_ten_minutes_l2021_202198
