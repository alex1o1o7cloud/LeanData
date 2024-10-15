import Mathlib

namespace NUMINAMATH_CALUDE_min_sequence_length_l1639_163917

def S : Finset Nat := {1, 2, 3, 4}

def is_valid_sequence (a : List Nat) : Prop :=
  ∀ b : List Nat, b.length = 4 ∧ b.toFinset = S ∧ b.getLast? ≠ some 1 →
    ∃ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ ≤ a.length ∧
      (a.get? i₁, a.get? i₂, a.get? i₃, a.get? i₄) = (b.get? 0, b.get? 1, b.get? 2, b.get? 3)

theorem min_sequence_length :
  ∃ a : List Nat, a.length = 11 ∧ is_valid_sequence a ∧
    ∀ a' : List Nat, is_valid_sequence a' → a'.length ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_sequence_length_l1639_163917


namespace NUMINAMATH_CALUDE_square_area_l1639_163937

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The line y = 7 -/
def line : ℝ := 7

/-- The theorem stating the area of the square -/
theorem square_area : 
  ∃ (x₁ x₂ : ℝ), 
    parabola x₁ = line ∧ 
    parabola x₂ = line ∧ 
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 = 28 :=
sorry

end NUMINAMATH_CALUDE_square_area_l1639_163937


namespace NUMINAMATH_CALUDE_hospital_baby_probability_l1639_163976

/-- The probability of success for a single trial -/
def p : ℚ := 1/3

/-- The number of trials -/
def n : ℕ := 6

/-- The number of successes we're interested in -/
def k : ℕ := 3

/-- The probability of at least k successes in n trials with probability p -/
def prob_at_least (p : ℚ) (n k : ℕ) : ℚ :=
  1 - (Finset.range k).sum (λ i => Nat.choose n i * p^i * (1-p)^(n-i))

theorem hospital_baby_probability :
  prob_at_least p n k = 233/729 := by sorry

end NUMINAMATH_CALUDE_hospital_baby_probability_l1639_163976


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1639_163905

theorem smallest_factorization_coefficient (b : ℕ+) : 
  (∃ (r s : ℤ), (∀ x : ℝ, x^2 + b.val*x + 3258 = (x + r) * (x + s))) →
  b.val ≥ 1089 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1639_163905


namespace NUMINAMATH_CALUDE_emily_fish_weight_l1639_163959

/-- Calculates the total weight of fish caught by Emily -/
def total_fish_weight (trout_count catfish_count bluegill_count : ℕ)
                      (trout_weight catfish_weight bluegill_weight : ℝ) : ℝ :=
  (trout_count : ℝ) * trout_weight +
  (catfish_count : ℝ) * catfish_weight +
  (bluegill_count : ℝ) * bluegill_weight

/-- Proves that Emily caught 25 pounds of fish -/
theorem emily_fish_weight :
  total_fish_weight 4 3 5 2 1.5 2.5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_emily_fish_weight_l1639_163959


namespace NUMINAMATH_CALUDE_bagel_savings_l1639_163971

/-- The cost of an individual bagel in cents -/
def individual_cost : ℕ := 225

/-- The cost of a dozen bagels in dollars -/
def dozen_cost : ℕ := 24

/-- The number of bagels in a dozen -/
def bagels_per_dozen : ℕ := 12

/-- The savings per bagel in cents when buying a dozen -/
theorem bagel_savings : ℕ := by
  -- Convert individual cost to cents
  -- Calculate cost per bagel when buying a dozen
  -- Convert dozen cost to cents
  -- Calculate the difference
  sorry

end NUMINAMATH_CALUDE_bagel_savings_l1639_163971


namespace NUMINAMATH_CALUDE_evaluate_expression_l1639_163966

theorem evaluate_expression (a b : ℚ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a*b + b^2) = 97 / 7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1639_163966


namespace NUMINAMATH_CALUDE_banana_bread_recipe_l1639_163988

/-- Given a banana bread recipe and baking requirements, determine the number of loaves the recipe can make. -/
theorem banana_bread_recipe (total_loaves : ℕ) (total_bananas : ℕ) (bananas_per_recipe : ℕ) 
  (h1 : total_loaves = 99)
  (h2 : total_bananas = 33)
  (h3 : bananas_per_recipe = 1)
  (h4 : total_bananas > 0) :
  total_loaves / total_bananas = 3 := by
  sorry

#check banana_bread_recipe

end NUMINAMATH_CALUDE_banana_bread_recipe_l1639_163988


namespace NUMINAMATH_CALUDE_vector_properties_and_projection_l1639_163923

/-- Given vectors in ℝ², prove properties about their relationships and projections -/
theorem vector_properties_and_projection :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (x, 3)
  let c : ℝ × ℝ := (5, y)
  let d : ℝ × ℝ := (8, 6)

  -- b is parallel to d
  (b.2 / b.1 = d.2 / d.1) →
  -- 4a + d is perpendicular to c
  ((4 * a.1 + d.1) * c.1 + (4 * a.2 + d.2) * c.2 = 0) →

  -- Prove that b and c have specific values
  (b = (4, 3) ∧ c = (5, -2)) ∧
  -- Prove that the projection of c onto a is -7√2/2
  (let proj := (a.1 * c.1 + a.2 * c.2) / Real.sqrt (a.1^2 + a.2^2)
   proj = -7 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_and_projection_l1639_163923


namespace NUMINAMATH_CALUDE_scale_length_l1639_163977

-- Define the number of parts in the scale
def num_parts : ℕ := 5

-- Define the length of each part in inches
def part_length : ℕ := 16

-- Theorem stating the total length of the scale
theorem scale_length : num_parts * part_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_l1639_163977


namespace NUMINAMATH_CALUDE_difference_of_squares_64_36_l1639_163992

theorem difference_of_squares_64_36 : 64^2 - 36^2 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_64_36_l1639_163992


namespace NUMINAMATH_CALUDE_sin_negative_135_degrees_l1639_163998

theorem sin_negative_135_degrees : Real.sin (-(135 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_135_degrees_l1639_163998


namespace NUMINAMATH_CALUDE_tim_balloon_count_l1639_163908

/-- The number of violet balloons Dan has -/
def dan_balloons : ℕ := 29

/-- The factor by which Tim has more balloons than Dan -/
def tim_factor : ℕ := 7

/-- The number of violet balloons Tim has -/
def tim_balloons : ℕ := dan_balloons * tim_factor

theorem tim_balloon_count : tim_balloons = 203 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l1639_163908


namespace NUMINAMATH_CALUDE_work_completion_days_l1639_163906

theorem work_completion_days (total_men : ℕ) (absent_men : ℕ) (reduced_days : ℕ) 
  (h1 : total_men = 60)
  (h2 : absent_men = 10)
  (h3 : reduced_days = 60) :
  let remaining_men := total_men - absent_men
  let original_days := (remaining_men * reduced_days) / total_men
  original_days = 50 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l1639_163906


namespace NUMINAMATH_CALUDE_supermarket_spending_l1639_163975

theorem supermarket_spending (total : ℝ) :
  (1/2 : ℝ) * total +  -- Spent on fresh fruits and vegetables
  (1/3 : ℝ) * total +  -- Spent on meat products
  (1/10 : ℝ) * total + -- Spent on bakery products
  10 = total           -- Remaining spent on candy
  →
  total = 150 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1639_163975


namespace NUMINAMATH_CALUDE_valid_triangle_constructions_l1639_163932

-- Define the basic structure
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the given points
variable (A A₀ D E : ℝ × ℝ)

-- Define the midpoint property
def is_midpoint (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the median property
def is_median (M : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  is_midpoint M B C

-- Define the angle bisector property
def is_angle_bisector (D : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop := sorry

-- Define the perpendicular bisector property
def is_perpendicular_bisector (E : ℝ × ℝ) (A C : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem valid_triangle_constructions 
  (h1 : is_midpoint A₀ (Triangle.B t) (Triangle.C t))
  (h2 : is_median A₀ (Triangle.A t) (Triangle.B t) (Triangle.C t))
  (h3 : is_angle_bisector D (Triangle.A t) (Triangle.B t) (Triangle.C t))
  (h4 : is_perpendicular_bisector E (Triangle.A t) (Triangle.C t)) :
  ∃ (C₁ C₂ : ℝ × ℝ), C₁ ≠ C₂ ∧ 
    (∃ (t₁ t₂ : Triangle), 
      (t₁.A = A ∧ t₁.C = C₁) ∧ 
      (t₂.A = A ∧ t₂.C = C₂) ∧
      (is_midpoint A₀ t₁.B t₁.C) ∧
      (is_midpoint A₀ t₂.B t₂.C) ∧
      (is_median A₀ t₁.A t₁.B t₁.C) ∧
      (is_median A₀ t₂.A t₂.B t₂.C) ∧
      (is_angle_bisector D t₁.A t₁.B t₁.C) ∧
      (is_angle_bisector D t₂.A t₂.B t₂.C) ∧
      (is_perpendicular_bisector E t₁.A t₁.C) ∧
      (is_perpendicular_bisector E t₂.A t₂.C)) :=
sorry


end NUMINAMATH_CALUDE_valid_triangle_constructions_l1639_163932


namespace NUMINAMATH_CALUDE_tank_capacity_l1639_163938

/-- Represents the capacity of a tank and its inlet/outlet properties -/
structure Tank where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- Theorem stating the capacity of the tank given the conditions -/
theorem tank_capacity (t : Tank)
  (h1 : t.outlet_time = 10)
  (h2 : t.inlet_rate = 8 * 60)
  (h3 : t.combined_time = 16)
  : t.capacity = 1280 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1639_163938


namespace NUMINAMATH_CALUDE_factory_shift_cost_l1639_163909

/-- The cost to employ all workers for one 8-hour shift -/
def total_cost (total_employees : ℕ) (low_wage_employees : ℕ) (mid_wage_employees : ℕ) 
  (low_wage : ℕ) (mid_wage : ℕ) (high_wage : ℕ) (shift_hours : ℕ) : ℕ :=
  let high_wage_employees := total_employees - low_wage_employees - mid_wage_employees
  low_wage_employees * low_wage * shift_hours + 
  mid_wage_employees * mid_wage * shift_hours + 
  high_wage_employees * high_wage * shift_hours

/-- Theorem stating the total cost for the given scenario -/
theorem factory_shift_cost : 
  total_cost 300 200 40 12 14 17 8 = 31840 := by
  sorry

end NUMINAMATH_CALUDE_factory_shift_cost_l1639_163909


namespace NUMINAMATH_CALUDE_transform_sine_function_l1639_163990

/-- Given a function f and its transformed version g, 
    where g is obtained by shortening the abscissas of f to half their original length 
    and then shifting the resulting curve to the right by π/3 units,
    prove that f(x) = sin(x/2 + π/12) if g(x) = sin(x - π/4) -/
theorem transform_sine_function (f g : ℝ → ℝ) :
  (∀ x, g x = f ((x - π/3) / 2)) →
  (∀ x, g x = Real.sin (x - π/4)) →
  ∀ x, f x = Real.sin (x/2 + π/12) := by
sorry

end NUMINAMATH_CALUDE_transform_sine_function_l1639_163990


namespace NUMINAMATH_CALUDE_temperature_conversion_l1639_163953

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 68 → t = 20 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l1639_163953


namespace NUMINAMATH_CALUDE_percent_less_than_l1639_163907

theorem percent_less_than (P Q : ℝ) (h : P < Q) :
  (Q - P) / Q * 100 = 100 * (Q - P) / Q :=
by sorry

end NUMINAMATH_CALUDE_percent_less_than_l1639_163907


namespace NUMINAMATH_CALUDE_product_equals_half_l1639_163961

/-- Given that a * b * c * d = (√((a + 2) * (b + 3))) / (c + 1) * sin(d) for any a, b, c, and d,
    prove that 6 * 15 * 11 * 30 = 0.5 -/
theorem product_equals_half :
  (∀ a b c d : ℝ, a * b * c * d = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1) * Real.sin d) →
  6 * 15 * 11 * 30 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_half_l1639_163961


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1639_163920

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 →
  (6 * x - 3) / (x^2 - 9*x - 36) = (23/5) / (x - 12) + (7/5) / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1639_163920


namespace NUMINAMATH_CALUDE_nth_monomial_formula_l1639_163946

/-- A sequence of monomials is defined as follows:
    For n = 1: 2x
    For n = 2: -4x^2
    For n = 3: 6x^3
    For n = 4: -8x^4
    For n = 5: 10x^5
    ...
    This function represents the coefficient of the nth monomial in the sequence. -/
def monomial_coefficient (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n)

/-- This function represents the exponent of x in the nth monomial of the sequence. -/
def monomial_exponent (n : ℕ) : ℕ := n

/-- This theorem states that the nth monomial in the sequence
    can be expressed as (-1)^(n+1) * 2n * x^n for any positive integer n. -/
theorem nth_monomial_formula (n : ℕ) (h : n > 0) :
  monomial_coefficient n = (-1)^(n+1) * (2*n) ∧ monomial_exponent n = n :=
sorry

end NUMINAMATH_CALUDE_nth_monomial_formula_l1639_163946


namespace NUMINAMATH_CALUDE_carpet_area_l1639_163915

theorem carpet_area : 
  ∀ (length width : ℝ) (shoe_length : ℝ),
    shoe_length = 28 →
    length = 15 * shoe_length →
    width = 10 * shoe_length →
    length * width = 117600 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_l1639_163915


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1639_163978

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1639_163978


namespace NUMINAMATH_CALUDE_expected_value_Z_l1639_163922

/-- The probability mass function for the random variable Z --/
def pmf_Z (P : ℝ) (k : ℕ) : ℝ :=
  if k ≥ 2 then P * (1 - P)^(k - 1) + (1 - P) * P^(k - 1) else 0

/-- The expected value of Z --/
noncomputable def E_Z (P : ℝ) : ℝ :=
  ∑' k, k * pmf_Z P k

/-- Theorem stating the expected value of Z --/
theorem expected_value_Z (P : ℝ) (hP : 0 < P ∧ P < 1) :
  E_Z P = 1 / (P * (1 - P)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_Z_l1639_163922


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l1639_163972

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let total_distance := 2 * distance
  let completed_distance := distance + 0.1 * distance
  (completed_distance / total_distance) * 100 = 55 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l1639_163972


namespace NUMINAMATH_CALUDE_problem_statement_l1639_163927

theorem problem_statement (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1639_163927


namespace NUMINAMATH_CALUDE_triangle_area_l1639_163943

-- Define the linear functions
def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := -x - 4

-- Define the triangle
def Triangle := {(x, y) : ℝ × ℝ | (y = f x ∨ y = g x) ∧ y ≥ 0}

-- Theorem statement
theorem triangle_area : MeasureTheory.volume Triangle = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1639_163943


namespace NUMINAMATH_CALUDE_unique_point_on_circle_l1639_163921

-- Define the points A and B
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (2, 1)

-- Define the circle C
def C (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 16

-- Define the distance squared between two points
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem unique_point_on_circle (a : ℝ) : 
  (∃! P : ℝ × ℝ, C a P.1 P.2 ∧ distanceSquared P A + 2 * distanceSquared P B = 24) →
  a = -1 ∨ a = 3 := by
sorry


end NUMINAMATH_CALUDE_unique_point_on_circle_l1639_163921


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1639_163958

-- Define the lines
def line1 (x y : ℝ) : Prop := y = 3 * x + 14
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 40

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = -68 := by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1639_163958


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l1639_163941

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1} ∪ {x : ℝ | x < -5} := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x ≥ t^2 - (11/2)*t) ↔ (1/2 ≤ t ∧ t ≤ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l1639_163941


namespace NUMINAMATH_CALUDE_problem_solution_l1639_163903

def A (a : ℚ) : Set ℚ := {a^2, a+1, -3}
def B (a : ℚ) : Set ℚ := {a-3, 3*a-1, a^2+1}
def C (m : ℚ) : Set ℚ := {x | m*x = 1}

theorem problem_solution (a m : ℚ) 
  (h1 : A a ∩ B a = {-3}) 
  (h2 : C m ⊆ A a ∩ B a) : 
  a = -2/3 ∧ (m = 0 ∨ m = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1639_163903


namespace NUMINAMATH_CALUDE_sum_mod_9_equals_5_l1639_163945

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Given a list of natural numbers, compute the sum of their modulo 9 values
    after reducing each number to the sum of its digits -/
def sum_mod_9_of_digit_sums (numbers : List ℕ) : ℕ :=
  (numbers.map (fun n => sum_of_digits n % 9)).sum % 9

/-- The main theorem stating that the sum of modulo 9 values of the given numbers
    after reducing each to the sum of its digits is 5 -/
theorem sum_mod_9_equals_5 :
  sum_mod_9_of_digit_sums [1, 21, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999] = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_9_equals_5_l1639_163945


namespace NUMINAMATH_CALUDE_stuffed_animal_cost_l1639_163996

theorem stuffed_animal_cost (coloring_books_cost peanuts_cost total_spent : ℚ) : 
  coloring_books_cost = 8 →
  peanuts_cost = 6 →
  total_spent = 25 →
  total_spent - (coloring_books_cost + peanuts_cost) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_stuffed_animal_cost_l1639_163996


namespace NUMINAMATH_CALUDE_train_speed_on_time_l1639_163983

/-- Proves that the speed at which the train arrives on time is 84 km/h, given the conditions -/
theorem train_speed_on_time (d : ℝ) (t : ℝ) :
  (d = 80 * (t + 24/60)) →
  (d = 90 * (t - 32/60)) →
  (d / t = 84) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_on_time_l1639_163983


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1639_163985

theorem arithmetic_sequence_terms (a₁ : ℕ) (d : ℤ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 20 ∧ d = -2 ∧ aₙ = 10 ∧ aₙ = a₁ + (n - 1) * d → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1639_163985


namespace NUMINAMATH_CALUDE_waysToSelectIs186_l1639_163974

/-- The number of ways to select 5 balls from a bag containing 4 red balls and 6 white balls,
    such that the total score is at least 7 points (where red balls score 2 points and white balls score 1 point). -/
def waysToSelect : ℕ :=
  Nat.choose 4 4 * Nat.choose 6 1 +
  Nat.choose 4 3 * Nat.choose 6 2 +
  Nat.choose 4 2 * Nat.choose 6 3

/-- The theorem stating that the number of ways to select the balls is 186. -/
theorem waysToSelectIs186 : waysToSelect = 186 := by
  sorry

end NUMINAMATH_CALUDE_waysToSelectIs186_l1639_163974


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1639_163902

theorem complex_equation_solution (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = 5) →
  (a * b = t - 3 + 5 * Complex.I) →
  (t > 0) →
  (t = 3 + 10 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1639_163902


namespace NUMINAMATH_CALUDE_two_valid_permutations_l1639_163954

def S : Finset ℕ := Finset.range 2022

def is_valid_permutation (A : Fin 2022 → ℕ) : Prop :=
  Function.Injective A ∧ (∀ i, A i ∈ S) ∧
  (∀ n m : Fin 2022, (A n + A m) % (Nat.gcd n.val m.val) = 0)

theorem two_valid_permutations :
  ∃! (p : Finset (Fin 2022 → ℕ)), p.card = 2 ∧ ∀ A ∈ p, is_valid_permutation A :=
sorry

end NUMINAMATH_CALUDE_two_valid_permutations_l1639_163954


namespace NUMINAMATH_CALUDE_book_sale_fraction_l1639_163987

theorem book_sale_fraction (price : ℝ) (remaining : ℕ) (total_received : ℝ) :
  price = 3.5 →
  remaining = 36 →
  total_received = 252 →
  ∃ (total : ℕ) (sold : ℕ),
    total > 0 ∧
    sold = total - remaining ∧
    (sold : ℝ) / total = 2 / 3 ∧
    price * sold = total_received :=
by sorry

end NUMINAMATH_CALUDE_book_sale_fraction_l1639_163987


namespace NUMINAMATH_CALUDE_sample_capacity_proof_l1639_163916

theorem sample_capacity_proof (n : ℕ) (frequency : ℕ) (relative_frequency : ℚ) 
  (h1 : frequency = 30)
  (h2 : relative_frequency = 1/4)
  (h3 : relative_frequency = frequency / n) :
  n = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_proof_l1639_163916


namespace NUMINAMATH_CALUDE_acid_mixing_problem_l1639_163933

/-- Represents the acid mixing problem -/
theorem acid_mixing_problem 
  (volume_first : ℝ) 
  (percentage_second : ℝ) 
  (volume_final : ℝ) 
  (percentage_final : ℝ) 
  (h1 : volume_first = 4)
  (h2 : percentage_second = 75)
  (h3 : volume_final = 20)
  (h4 : percentage_final = 72) :
  ∃ (percentage_first : ℝ),
    percentage_first = 60 ∧
    volume_first * (percentage_first / 100) + 
    (volume_final - volume_first) * (percentage_second / 100) = 
    volume_final * (percentage_final / 100) :=
sorry

end NUMINAMATH_CALUDE_acid_mixing_problem_l1639_163933


namespace NUMINAMATH_CALUDE_mitch_savings_l1639_163964

/-- Represents the total amount of money Mitch has saved for his boating hobby -/
def total_saved : ℕ := 20000

/-- Cost of a new boat per foot in length -/
def boat_cost_per_foot : ℕ := 1500

/-- Amount Mitch needs to keep for license and registration -/
def license_registration_cost : ℕ := 500

/-- Maximum length of boat Mitch can buy -/
def max_boat_length : ℕ := 12

/-- Docking fee multiplier (relative to license and registration cost) -/
def docking_fee_multiplier : ℕ := 3

theorem mitch_savings :
  total_saved = 
    boat_cost_per_foot * max_boat_length + 
    license_registration_cost + 
    docking_fee_multiplier * license_registration_cost :=
by sorry

end NUMINAMATH_CALUDE_mitch_savings_l1639_163964


namespace NUMINAMATH_CALUDE_square_side_length_l1639_163965

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s * s = d * d / 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1639_163965


namespace NUMINAMATH_CALUDE_davids_mowing_hours_l1639_163994

theorem davids_mowing_hours (rate : ℝ) (days : ℕ) (remaining : ℝ) : 
  rate = 14 → days = 7 → remaining = 49 → 
  ∃ (hours : ℝ), 
    hours * rate * days / 2 / 2 = remaining ∧ 
    hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_davids_mowing_hours_l1639_163994


namespace NUMINAMATH_CALUDE_sum_of_f_at_lg2_and_lg_half_l1639_163955

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) + a

theorem sum_of_f_at_lg2_and_lg_half (a : ℝ) :
  f a 0 = 1 → f a (Real.log 2 / Real.log 10) + f a (Real.log (1/2) / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_at_lg2_and_lg_half_l1639_163955


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1639_163911

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) :
  (10 * x₁^2 + 15 * x₁ - 17 = 0) →
  (10 * x₂^2 + 15 * x₂ - 17 = 0) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 113/20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1639_163911


namespace NUMINAMATH_CALUDE_population_size_l1639_163934

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem population_size (P : ℝ) 
  (birth_rate : ℝ) (death_rate : ℝ) (net_growth_rate : ℝ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate = 2.1)
  (h4 : (birth_rate - death_rate) / P * 100 = net_growth_rate) :
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_population_size_l1639_163934


namespace NUMINAMATH_CALUDE_race_probability_l1639_163960

theorem race_probability (total_cars : ℕ) (prob_X : ℚ) (prob_Z : ℚ) (prob_XYZ : ℚ) :
  total_cars = 8 →
  prob_X = 1/2 →
  prob_Z = 1/3 →
  prob_XYZ = 13/12 →
  ∃ (prob_Y : ℚ), prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_Y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l1639_163960


namespace NUMINAMATH_CALUDE_additional_class_choices_l1639_163962

def total_classes : ℕ := 10
def compulsory_classes : ℕ := 1
def total_classes_to_take : ℕ := 4

theorem additional_class_choices : 
  Nat.choose (total_classes - compulsory_classes) (total_classes_to_take - compulsory_classes) = 84 := by
  sorry

end NUMINAMATH_CALUDE_additional_class_choices_l1639_163962


namespace NUMINAMATH_CALUDE_hamburger_count_l1639_163949

theorem hamburger_count (total_spent single_cost double_cost double_count : ℚ) 
  (h1 : total_spent = 64.5)
  (h2 : single_cost = 1)
  (h3 : double_cost = 1.5)
  (h4 : double_count = 29) :
  ∃ (single_count : ℚ), 
    single_count * single_cost + double_count * double_cost = total_spent ∧ 
    single_count + double_count = 50 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_count_l1639_163949


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1639_163948

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + 54 * a - 12 = 0) →
  (3 * b^3 - 9 * b^2 + 54 * b - 12 = 0) →
  (3 * c^3 - 9 * c^2 + 54 * c - 12 = 0) →
  (a + 2*b - 2)^3 + (b + 2*c - 2)^3 + (c + 2*a - 2)^3 = 162 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1639_163948


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_distinct_fixed_points_condition_l1639_163989

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

theorem fixed_points_for_specific_values :
  ∀ x : ℝ, is_fixed_point (f 1 (-2)) x ↔ (x = 3 ∨ x = -1) := by sorry

theorem two_distinct_fixed_points_condition :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔ (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_distinct_fixed_points_condition_l1639_163989


namespace NUMINAMATH_CALUDE_L8_2_7_exponent_is_columns_l1639_163984

/-- Represents an orthogonal array -/
structure OrthogonalArray where
  experiments : ℕ
  levels : ℕ
  columns : ℕ

/-- The specific orthogonal array L₈(2⁷) -/
def L8_2_7 : OrthogonalArray :=
  { experiments := 8
  , levels := 2
  , columns := 7 }

theorem L8_2_7_exponent_is_columns : L8_2_7.columns = 7 := by
  sorry

end NUMINAMATH_CALUDE_L8_2_7_exponent_is_columns_l1639_163984


namespace NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l1639_163991

/-- A function that checks if a natural number is a nine-digit number -/
def isNineDigit (n : ℕ) : Prop :=
  100000000 ≤ n ∧ n < 1000000000

/-- A function that checks if a natural number ends with 5 -/
def endsWithFive (n : ℕ) : Prop :=
  n % 10 = 5

/-- A function that checks if a natural number contains each of the digits 1-9 exactly once -/
def containsEachDigitOnce (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] →
    (∃! p : ℕ, p < 9 ∧ (n / 10^p) % 10 = d)

/-- The main theorem stating that no number satisfying the given conditions is a perfect square -/
theorem no_perfect_square_with_conditions :
  ¬∃ n : ℕ, isNineDigit n ∧ endsWithFive n ∧ containsEachDigitOnce n ∧ ∃ m : ℕ, n = m^2 := by
  sorry


end NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l1639_163991


namespace NUMINAMATH_CALUDE_tens_digit_of_6_to_2050_l1639_163926

theorem tens_digit_of_6_to_2050 : 6^2050 % 100 = 56 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_to_2050_l1639_163926


namespace NUMINAMATH_CALUDE_roots_product_theorem_l1639_163936

theorem roots_product_theorem (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) → 
  (b^3 - 15*b^2 + 25*b - 10 = 0) → 
  (c^3 - 15*c^2 + 25*c - 10 = 0) → 
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end NUMINAMATH_CALUDE_roots_product_theorem_l1639_163936


namespace NUMINAMATH_CALUDE_fraction_comparison_l1639_163939

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  a / d > b / c := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1639_163939


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l1639_163963

/-- A triangle with interior angles in the ratio 1:2:3 is a right triangle. -/
theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (α β γ : ℝ) :
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Angles are positive
  α + β + γ = 180 →        -- Sum of angles in a triangle is 180°
  β = 2 * α ∧ γ = 3 * α →  -- Angles are in the ratio 1:2:3
  γ = 90                   -- The largest angle is 90°
  := by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l1639_163963


namespace NUMINAMATH_CALUDE_angle_between_vectors_not_necessarily_alpha_minus_beta_l1639_163912

theorem angle_between_vectors_not_necessarily_alpha_minus_beta 
  (α β : ℝ) (a b : ℝ × ℝ) :
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  a ≠ b →
  ∃ θ, Real.cos θ = Real.cos α * Real.cos β + Real.sin α * Real.sin β ∧ θ ≠ α - β :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_not_necessarily_alpha_minus_beta_l1639_163912


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1639_163993

/-- Given a car traveling at a constant speed for a certain time, 
    calculate the distance covered. -/
def distance_covered (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem stating that a car traveling at 107 km/h for 6.5 hours
    covers a distance of 695.5 km. -/
theorem car_distance_theorem :
  distance_covered 107 6.5 = 695.5 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1639_163993


namespace NUMINAMATH_CALUDE_rectangle_x_value_l1639_163997

/-- A rectangle with specified side lengths -/
structure Rectangle where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X must be 7 in the given rectangle -/
theorem rectangle_x_value (r : Rectangle) 
    (h1 : r.top_left = 1)
    (h2 : r.top_middle = 2)
    (h3 : r.top_right = 3)
    (h4 : r.bottom_left = 4)
    (h5 : r.bottom_middle = 2)
    (h6 : r.bottom_right = 7)
    (h_rect : r.top_left + r.top_middle + X + r.top_right = 
              r.bottom_left + r.bottom_middle + r.bottom_right) : 
  X = 7 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_x_value_l1639_163997


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1639_163924

theorem decimal_to_fraction : 
  (3.76 : ℚ) = 94 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1639_163924


namespace NUMINAMATH_CALUDE_maria_has_four_l1639_163942

/-- Represents a player in the card game -/
inductive Player : Type
  | Maria
  | Josh
  | Laura
  | Neil
  | Eva

/-- The score of each player -/
def score (p : Player) : ℕ :=
  match p with
  | Player.Maria => 13
  | Player.Josh => 15
  | Player.Laura => 9
  | Player.Neil => 18
  | Player.Eva => 19

/-- The set of all possible cards -/
def cards : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

/-- Predicate to check if a pair of cards is valid for a player -/
def validCardPair (p : Player) (c1 c2 : ℕ) : Prop :=
  c1 ∈ cards ∧ c2 ∈ cards ∧ c1 + c2 = score p ∧ c1 ≠ c2

/-- Theorem stating that Maria must have received card number 4 -/
theorem maria_has_four :
  ∃ (c : ℕ), c ∈ cards ∧ c ≠ 4 ∧ validCardPair Player.Maria 4 c ∧
  (∀ (p : Player), p ≠ Player.Maria → ¬∃ (c1 c2 : ℕ), (c1 = 4 ∨ c2 = 4) ∧ validCardPair p c1 c2) :=
sorry

end NUMINAMATH_CALUDE_maria_has_four_l1639_163942


namespace NUMINAMATH_CALUDE_pig_farm_fence_length_l1639_163968

/-- Represents a rectangular pig farm with specific dimensions -/
structure PigFarm where
  /-- Length of the shorter sides of the rectangle -/
  short_side : ℝ
  /-- Ensures the short side is positive -/
  short_side_pos : short_side > 0

/-- Calculates the area of the pig farm -/
def PigFarm.area (farm : PigFarm) : ℝ :=
  2 * farm.short_side * farm.short_side

/-- Calculates the total fence length of the pig farm -/
def PigFarm.fence_length (farm : PigFarm) : ℝ :=
  4 * farm.short_side

/-- Theorem stating the fence length for a pig farm with area 1250 sq ft -/
theorem pig_farm_fence_length :
  ∃ (farm : PigFarm), farm.area = 1250 ∧ farm.fence_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_pig_farm_fence_length_l1639_163968


namespace NUMINAMATH_CALUDE_circle_sum_l1639_163931

def Circle := Fin 12 → ℝ

def is_valid_circle (c : Circle) : Prop :=
  (∀ i, c i ≠ 0) ∧
  (∀ i, i % 2 = 0 → c i = c ((i + 11) % 12) + c ((i + 1) % 12)) ∧
  (∀ i, i % 2 = 1 → c i = c ((i + 11) % 12) * c ((i + 1) % 12))

theorem circle_sum (c : Circle) (h : is_valid_circle c) :
  (Finset.sum Finset.univ c) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_l1639_163931


namespace NUMINAMATH_CALUDE_ball_max_height_l1639_163952

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 10

/-- The maximum height reached by the ball -/
def max_height : ℝ := 135

/-- Theorem stating that the maximum height reached by the ball is 135 meters -/
theorem ball_max_height : 
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1639_163952


namespace NUMINAMATH_CALUDE_sum_in_base_b_l1639_163995

/-- Given a base b, converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, converts a number from base 10 to base b -/
def fromBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a given base b satisfies the condition (14)(17)(18) = 5404 in base b -/
def isValidBase (b : ℕ) : Prop :=
  (toBase10 14 b) * (toBase10 17 b) * (toBase10 18 b) = toBase10 5404 b

theorem sum_in_base_b (b : ℕ) (h : isValidBase b) :
  fromBase10 ((toBase10 14 b) + (toBase10 17 b) + (toBase10 18 b)) b = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_b_l1639_163995


namespace NUMINAMATH_CALUDE_greatest_k_value_l1639_163973

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) → 
  k ≤ Real.sqrt 117 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1639_163973


namespace NUMINAMATH_CALUDE_program_arrangements_l1639_163980

/-- The number of solo segments in the program -/
def num_solo_segments : ℕ := 5

/-- The number of chorus segments in the program -/
def num_chorus_segments : ℕ := 3

/-- The number of spaces available for chorus segments after arranging solo segments -/
def num_spaces_for_chorus : ℕ := num_solo_segments + 1 - 1 -- +1 for spaces between solos, -1 for not placing first

/-- The number of different programs that can be arranged -/
def num_programs : ℕ := (Nat.factorial num_solo_segments) * (num_spaces_for_chorus.choose num_chorus_segments)

theorem program_arrangements :
  num_programs = 7200 :=
sorry

end NUMINAMATH_CALUDE_program_arrangements_l1639_163980


namespace NUMINAMATH_CALUDE_mode_of_sports_shoes_l1639_163970

/-- Represents the sales data for a particular shoe size -/
structure SalesData :=
  (size : Float)
  (sales : Nat)

/-- Finds the mode of a list of SalesData -/
def findMode (data : List SalesData) : Float :=
  sorry

/-- The sales data for the sports shoes -/
def salesData : List SalesData := [
  ⟨24, 1⟩,
  ⟨24.5, 3⟩,
  ⟨25, 10⟩,
  ⟨25.5, 4⟩,
  ⟨26, 2⟩
]

theorem mode_of_sports_shoes :
  findMode salesData = 25 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_sports_shoes_l1639_163970


namespace NUMINAMATH_CALUDE_vector_scalar_properties_l1639_163951

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_scalar_properties :
  (∀ (m : ℝ) (a b : V), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : V), (m - n) • a = m • a - n • a) ∧
  (∃ (m : ℝ) (a b : V), m • a = m • b ∧ a ≠ b) ∧
  (∀ (m n : ℝ) (a : V), a ≠ 0 → m • a = n • a → m = n) :=
by sorry

end NUMINAMATH_CALUDE_vector_scalar_properties_l1639_163951


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l1639_163967

/-- The line that is the perpendicular bisector of two points -/
def perpendicular_bisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P A = dist P B}

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point satisfies a line equation -/
def satisfies_equation (P : ℝ × ℝ) (L : LineEquation) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

theorem perpendicular_bisector_equation 
  (A B : ℝ × ℝ) 
  (hA : A = (7, -4)) 
  (hB : B = (-5, 6)) :
  ∃ L : LineEquation, 
    L.a = 6 ∧ L.b = -5 ∧ L.c = -1 ∧
    ∀ P, P ∈ perpendicular_bisector A B ↔ satisfies_equation P L :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l1639_163967


namespace NUMINAMATH_CALUDE_ultratown_block_perimeter_difference_l1639_163950

/-- Represents a rectangular city block with surrounding streets -/
structure CityBlock where
  length : ℝ
  width : ℝ
  street_width : ℝ

/-- Calculates the difference between outer and inner perimeters of a city block -/
def perimeter_difference (block : CityBlock) : ℝ :=
  2 * ((block.length + 2 * block.street_width) + (block.width + 2 * block.street_width)) -
  2 * (block.length + block.width)

/-- Theorem: The difference between outer and inner perimeters of the specified block is 200 feet -/
theorem ultratown_block_perimeter_difference :
  let block : CityBlock := {
    length := 500,
    width := 300,
    street_width := 25
  }
  perimeter_difference block = 200 := by
  sorry

end NUMINAMATH_CALUDE_ultratown_block_perimeter_difference_l1639_163950


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1639_163918

/-- Proves that a car can travel approximately 56.01 kilometers on a liter of fuel given specific conditions. -/
theorem car_fuel_efficiency (travel_time : Real) (fuel_used_gallons : Real) (speed_mph : Real)
    (gallons_to_liters : Real) (miles_to_km : Real)
    (h1 : travel_time = 5.7)
    (h2 : fuel_used_gallons = 3.9)
    (h3 : speed_mph = 91)
    (h4 : gallons_to_liters = 3.8)
    (h5 : miles_to_km = 1.6) :
    ∃ km_per_liter : Real, abs (km_per_liter - 56.01) < 0.01 ∧
    km_per_liter = (speed_mph * travel_time * miles_to_km) / (fuel_used_gallons * gallons_to_liters) :=
by
  sorry


end NUMINAMATH_CALUDE_car_fuel_efficiency_l1639_163918


namespace NUMINAMATH_CALUDE_max_fourth_power_sum_l1639_163904

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  ∃ (m : ℝ), m = 16 ∧ a^4 + b^4 + c^4 + d^4 ≤ m ∧
  ∃ (a' b' c' d' : ℝ), a'^3 + b'^3 + c'^3 + d'^3 = 8 ∧ a'^4 + b'^4 + c'^4 + d'^4 = m :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_power_sum_l1639_163904


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1639_163969

/-- The longest segment in a cylinder with radius 5 and height 10 is 10√2 -/
theorem longest_segment_in_cylinder : ∀ (r h : ℝ),
  r = 5 → h = 10 → 
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1639_163969


namespace NUMINAMATH_CALUDE_bird_count_l1639_163925

theorem bird_count (N t : ℚ) : 
  (3 / 5 * N + 1 / 4 * N + 10 * t = N) → 
  (3 / 5 * N = 40 * t) :=
by sorry

end NUMINAMATH_CALUDE_bird_count_l1639_163925


namespace NUMINAMATH_CALUDE_max_routes_in_network_l1639_163935

/-- A bus route network -/
structure BusNetwork where
  stops : Nat
  routes : Nat
  stops_per_route : Nat
  route_intersection : Nat

/-- The condition that any two routes either have no common stops or have exactly one common stop -/
def valid_intersection (network : BusNetwork) : Prop :=
  network.route_intersection = 0 ∨ network.route_intersection = 1

/-- The maximum number of routes possible given the constraints -/
def max_routes (network : BusNetwork) : Prop :=
  network.routes ≤ (network.stops * 4) / 3 ∧
  network.routes = 12

/-- Theorem stating the maximum number of routes in the given network -/
theorem max_routes_in_network (network : BusNetwork) 
  (h1 : network.stops = 9)
  (h2 : network.stops_per_route = 3)
  (h3 : valid_intersection network) :
  max_routes network :=
sorry

end NUMINAMATH_CALUDE_max_routes_in_network_l1639_163935


namespace NUMINAMATH_CALUDE_x_x_minus_one_sufficient_not_necessary_l1639_163929

theorem x_x_minus_one_sufficient_not_necessary (x : ℝ) :
  (∀ x, x * (x - 1) < 0 → x < 1) ∧
  (∃ x, x < 1 ∧ x * (x - 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_x_minus_one_sufficient_not_necessary_l1639_163929


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1639_163982

theorem gcd_of_three_numbers : Nat.gcd 9486 (Nat.gcd 13524 36582) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1639_163982


namespace NUMINAMATH_CALUDE_complex_real_condition_l1639_163999

theorem complex_real_condition (a : ℝ) : 
  let Z : ℂ := (a - 5) / (a^2 + 4*a - 5) + (a^2 + 2*a - 15) * Complex.I
  (Z.im = 0 ∧ (a^2 + 4*a - 5) ≠ 0) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1639_163999


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l1639_163901

theorem decimal_to_percentage (x : ℝ) (h : x = 0.02) : x * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l1639_163901


namespace NUMINAMATH_CALUDE_border_area_is_198_l1639_163900

/-- Calculates the area of the border for a framed picture -/
def border_area (picture_height : ℕ) (picture_width : ℕ) (border_width : ℕ) : ℕ :=
  let total_height := picture_height + 2 * border_width
  let total_width := picture_width + 2 * border_width
  total_height * total_width - picture_height * picture_width

/-- Theorem stating that the border area for the given dimensions is 198 square inches -/
theorem border_area_is_198 :
  border_area 12 15 3 = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_198_l1639_163900


namespace NUMINAMATH_CALUDE_simplify_expression_l1639_163944

theorem simplify_expression : 
  (9 * 10^12) / (3 * 10^4) + (2 * 10^8) / (4 * 10^2) = 300500000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1639_163944


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l1639_163919

theorem cubic_root_reciprocal_sum (a b c d : ℝ) (r s t : ℂ) 
  (ha : a ≠ 0) (hd : d ≠ 0)
  (h_cubic : ∀ x : ℂ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r ∨ x = s ∨ x = t) :
  1 / r^2 + 1 / s^2 + 1 / t^2 = (c^2 - 2 * b * d) / d^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l1639_163919


namespace NUMINAMATH_CALUDE_ratio_problem_l1639_163956

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : 
  x / y = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1639_163956


namespace NUMINAMATH_CALUDE_milk_container_problem_l1639_163914

theorem milk_container_problem (A B C : ℝ) : 
  A > 0 →  -- A is positive (container capacity)
  B = 0.375 * A →  -- B is 62.5% less than A
  C = A - B →  -- C contains the rest of the milk
  C - 152 = B + 152 →  -- After transfer, B and C are equal
  A = 608 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l1639_163914


namespace NUMINAMATH_CALUDE_coffee_shop_spending_l1639_163910

theorem coffee_shop_spending (ryan_spent : ℝ) (sarah_spent : ℝ) : 
  (sarah_spent = 0.60 * ryan_spent) →
  (ryan_spent = sarah_spent + 12.50) →
  (ryan_spent + sarah_spent = 50.00) :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_l1639_163910


namespace NUMINAMATH_CALUDE_smallest_cover_count_l1639_163957

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a rectangular region to be covered -/
structure Region where
  width : ℕ
  height : ℕ

def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

def Region.area (r : Region) : ℕ := r.width * r.height

/-- The number of rectangles needed to cover a region -/
def coverCount (r : Rectangle) (reg : Region) : ℕ :=
  Region.area reg / Rectangle.area r

theorem smallest_cover_count (r : Rectangle) (reg : Region) :
  r.width = 3 ∧ r.height = 4 ∧ reg.width = 12 ∧ reg.height = 12 →
  coverCount r reg = 12 ∧
  ∀ (r' : Rectangle) (reg' : Region),
    r'.width * r'.height ≤ r.width * r.height →
    reg'.width = 12 →
    coverCount r' reg' ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_cover_count_l1639_163957


namespace NUMINAMATH_CALUDE_cosine_amplitude_l1639_163979

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, d are positive constants,
    if the graph oscillates between 5 and 1, then a = 2. -/
theorem cosine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l1639_163979


namespace NUMINAMATH_CALUDE_total_stops_theorem_l1639_163928

def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

theorem total_stops_theorem : yoojeong_stops + namjoon_stops = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_stops_theorem_l1639_163928


namespace NUMINAMATH_CALUDE_correct_number_of_officers_l1639_163986

/-- Represents the number of officers in an office with given salary conditions. -/
def number_of_officers : ℕ :=
  let avg_salary_all : ℚ := 120
  let avg_salary_officers : ℚ := 420
  let avg_salary_non_officers : ℚ := 110
  let num_non_officers : ℕ := 450
  15

/-- Theorem stating that the number of officers is correct given the salary conditions. -/
theorem correct_number_of_officers :
  let avg_salary_all : ℚ := 120
  let avg_salary_officers : ℚ := 420
  let avg_salary_non_officers : ℚ := 110
  let num_non_officers : ℕ := 450
  let num_officers := number_of_officers
  (avg_salary_all * (num_officers + num_non_officers : ℚ) =
   avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_officers_l1639_163986


namespace NUMINAMATH_CALUDE_factorial_calculation_l1639_163913

theorem factorial_calculation : 
  Nat.factorial 8 - 7 * Nat.factorial 7 - 2 * Nat.factorial 6 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l1639_163913


namespace NUMINAMATH_CALUDE_hyperbola_point_range_l1639_163930

theorem hyperbola_point_range (x₀ y₀ : ℝ) : 
  (x₀^2 / 2 - y₀^2 = 1) →  -- Point on hyperbola
  (((-Real.sqrt 3 - x₀) * (Real.sqrt 3 - x₀) + (-y₀) * (-y₀)) ≤ 0) →  -- Dot product condition
  (-Real.sqrt 3 / 3 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_range_l1639_163930


namespace NUMINAMATH_CALUDE_max_area_is_1406_l1639_163940

/-- Represents a rectangular garden with integer side lengths. -/
structure RectangularGarden where
  width : ℕ
  length : ℕ
  perimeter_constraint : width * 2 + length * 2 = 150

/-- The area of a rectangular garden. -/
def garden_area (g : RectangularGarden) : ℕ :=
  g.width * g.length

/-- The maximum area of a rectangular garden with a perimeter of 150 feet. -/
def max_garden_area : ℕ := 1406

/-- Theorem stating that the maximum area of a rectangular garden with
    a perimeter of 150 feet and integer side lengths is 1406 square feet. -/
theorem max_area_is_1406 :
  ∀ g : RectangularGarden, garden_area g ≤ max_garden_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_is_1406_l1639_163940


namespace NUMINAMATH_CALUDE_equation_solution_l1639_163981

theorem equation_solution : ∃! x : ℝ, (x + 4) / (x - 2) = 3 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1639_163981


namespace NUMINAMATH_CALUDE_basketball_probability_l1639_163947

theorem basketball_probability (jack_prob jill_prob sandy_prob : ℚ) 
  (h1 : jack_prob = 1/6)
  (h2 : jill_prob = 1/7)
  (h3 : sandy_prob = 1/8) :
  (1 - jack_prob) * jill_prob * sandy_prob = 5/336 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l1639_163947
