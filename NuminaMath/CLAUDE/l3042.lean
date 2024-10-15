import Mathlib

namespace NUMINAMATH_CALUDE_inequality_and_sum_theorem_l3042_304244

def f (x : ℝ) : ℝ := |3*x - 1|

theorem inequality_and_sum_theorem :
  (∀ x : ℝ, f x - f (2 - x) > x ↔ x ∈ Set.Ioo (6/5) 4) ∧
  (∀ a b : ℝ, a + b = 2 → f (a^2) + f (b^2) ≥ 4) := by sorry

end NUMINAMATH_CALUDE_inequality_and_sum_theorem_l3042_304244


namespace NUMINAMATH_CALUDE_inequality_range_l3042_304277

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3) 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3042_304277


namespace NUMINAMATH_CALUDE_min_K_is_two_l3042_304234

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x - x^2

-- Define the property that f_K(x) = f(x) for all x ≥ 0
def f_K_equals_f (K : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≤ K

-- Theorem statement
theorem min_K_is_two :
  ∃ K : ℝ, (f_K_equals_f K ∧ ∀ K' : ℝ, K' < K → ¬f_K_equals_f K') ∧ K = 2 :=
sorry

end NUMINAMATH_CALUDE_min_K_is_two_l3042_304234


namespace NUMINAMATH_CALUDE_cos_unique_identifier_l3042_304263

open Real

theorem cos_unique_identifier (x : ℝ) (h1 : π / 2 < x) (h2 : x < π) :
  (sin x > 0 ∧ cos x < 0 ∧ cot x < 0) ∧
  (∀ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = cot →
    (f x < 0 → f = cos)) :=
by sorry

end NUMINAMATH_CALUDE_cos_unique_identifier_l3042_304263


namespace NUMINAMATH_CALUDE_trig_identity_l3042_304290

theorem trig_identity (α : ℝ) (h1 : α ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.sin (α + π/4) = -1/3) : 
  Real.sin (2*α) / Real.cos (π/4 - α) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3042_304290


namespace NUMINAMATH_CALUDE_hostel_expenditure_equation_l3042_304297

/-- Represents the average expenditure calculation for a student hostel with varying group costs. -/
theorem hostel_expenditure_equation 
  (A B C : ℕ) -- Original number of students in each group
  (a b c : ℕ) -- New students in each group
  (X Y Z : ℝ) -- Average expenditure for each group
  (h1 : A + B + C = 35) -- Total original students
  (h2 : a + b + c = 7)  -- Total new students
  : (A * X + B * Y + C * Z) / 35 - 1 = 
    ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42 := by
  sorry

#check hostel_expenditure_equation

end NUMINAMATH_CALUDE_hostel_expenditure_equation_l3042_304297


namespace NUMINAMATH_CALUDE_triangle_side_length_l3042_304286

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a = Real.sqrt 3 →
  Real.sin B = 1 / 2 →
  C = π / 6 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3042_304286


namespace NUMINAMATH_CALUDE_expression_evaluation_l3042_304259

theorem expression_evaluation (a b c : ℝ) : 
  (a - (b - c)) - ((a - b) - c) = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3042_304259


namespace NUMINAMATH_CALUDE_range_of_a_l3042_304209

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ 2*x^2 - 3*x + 1 ≤ 0) →
  (∀ x, q x ↔ (x - a)*(x - a - 1) ≤ 0) →
  (∀ x, p x → (1/2 : ℝ) ≤ x ∧ x ≤ 1) →
  (∀ x, q x → a ≤ x ∧ x ≤ a + 1) →
  (∀ x, ¬(p x) → ¬(q x)) →
  (∃ x, ¬(p x) ∧ q x) →
  0 ≤ a ∧ a ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3042_304209


namespace NUMINAMATH_CALUDE_lines_do_not_intersect_l3042_304202

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

theorem lines_do_not_intersect (k : ℝ) : 
  (parallel 
    { point := (1, 3), direction := (2, -5) }
    { point := (-1, 4), direction := (3, k) }) ↔ 
  k = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_lines_do_not_intersect_l3042_304202


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l3042_304271

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) → -3 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l3042_304271


namespace NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l3042_304217

theorem sandwich_non_condiment_percentage
  (total_weight : ℝ)
  (condiment_weight : ℝ)
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l3042_304217


namespace NUMINAMATH_CALUDE_dice_roll_probability_prob_first_less_than_second_l3042_304204

/-- The probability that when rolling two fair six-sided dice, the first roll is less than the second roll -/
theorem dice_roll_probability : ℚ :=
  5/12

/-- A fair six-sided die -/
def fair_die : Finset ℕ := Finset.range 6

/-- The sample space of rolling two dice -/
def two_dice_rolls : Finset (ℕ × ℕ) :=
  fair_die.product fair_die

/-- The event where the first roll is less than the second roll -/
def first_less_than_second : Set (ℕ × ℕ) :=
  {p | p.1 < p.2}

/-- The probability of the event where the first roll is less than the second roll -/
theorem prob_first_less_than_second :
  (two_dice_rolls.filter (λ p => p.1 < p.2)).card / two_dice_rolls.card = dice_roll_probability := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_prob_first_less_than_second_l3042_304204


namespace NUMINAMATH_CALUDE_aunt_gemma_dog_food_l3042_304275

/-- Calculates the amount of food each dog consumes per meal given the total amount of food,
    number of days it lasts, number of dogs, and number of meals per day. -/
def food_per_meal_per_dog (total_food : ℕ) (num_days : ℕ) (num_dogs : ℕ) (meals_per_day : ℕ) : ℕ :=
  (total_food * 1000) / (num_days * num_dogs * meals_per_day)

theorem aunt_gemma_dog_food :
  let num_sacks : ℕ := 2
  let weight_per_sack : ℕ := 50  -- in kg
  let num_days : ℕ := 50
  let num_dogs : ℕ := 4
  let meals_per_day : ℕ := 2
  food_per_meal_per_dog (num_sacks * weight_per_sack) num_days num_dogs meals_per_day = 250 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gemma_dog_food_l3042_304275


namespace NUMINAMATH_CALUDE_central_high_school_ratio_l3042_304294

theorem central_high_school_ratio (f s : ℚ) 
  (h1 : f > 0) (h2 : s > 0)
  (h3 : (3/7) * f = (2/3) * s) : f / s = 14/9 := by
  sorry

end NUMINAMATH_CALUDE_central_high_school_ratio_l3042_304294


namespace NUMINAMATH_CALUDE_regular_tetrahedron_height_l3042_304291

/-- Given a regular tetrahedron with an inscribed sphere, 
    prove that its height is 4 times the radius of the inscribed sphere -/
theorem regular_tetrahedron_height (h r : ℝ) : h = 4 * r :=
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_height_l3042_304291


namespace NUMINAMATH_CALUDE_fraction_equality_l3042_304249

theorem fraction_equality : (3 * 4 + 5) / 7 = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3042_304249


namespace NUMINAMATH_CALUDE_number_problem_l3042_304267

theorem number_problem : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  n / sum = 2 * diff ∧ n % sum = 50 ∧ n = 220050 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3042_304267


namespace NUMINAMATH_CALUDE_power_of_product_l3042_304273

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3042_304273


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3042_304219

-- Define the polynomial p
def p (x : ℝ) : ℝ := sorry

-- State the theorem
theorem polynomial_evaluation (y : ℝ) :
  (p (y^2 + 1) = 6 * y^4 - y^2 + 5) →
  (p (y^2 - 1) = 6 * y^4 - 25 * y^2 + 31) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3042_304219


namespace NUMINAMATH_CALUDE_casino_chip_loss_difference_l3042_304208

theorem casino_chip_loss_difference : 
  ∀ (x y : ℕ), 
    x + y = 16 →  -- Total number of chips lost
    20 * x + 100 * y = 880 →  -- Value of lost chips
    x - y = 2 :=  -- Difference in number of chips lost
by
  sorry

end NUMINAMATH_CALUDE_casino_chip_loss_difference_l3042_304208


namespace NUMINAMATH_CALUDE_multiple_remainder_l3042_304252

theorem multiple_remainder (x : ℕ) (h : x % 9 = 5) :
  ∃ k : ℕ, k > 0 ∧ (k * x) % 9 = 8 ∧ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiple_remainder_l3042_304252


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3042_304261

def A : Set ℝ := {x | |x - 3| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3042_304261


namespace NUMINAMATH_CALUDE_A_mod_126_l3042_304215

/-- A function that generates the number A by concatenating all three-digit numbers from 100 to 799 -/
def generate_A : ℕ := sorry

/-- Theorem stating that the number A is congruent to 91 modulo 126 -/
theorem A_mod_126 : generate_A % 126 = 91 := by sorry

end NUMINAMATH_CALUDE_A_mod_126_l3042_304215


namespace NUMINAMATH_CALUDE_intersection_sum_l3042_304218

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d → (x = 3 ∧ y = 3)) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l3042_304218


namespace NUMINAMATH_CALUDE_log_equation_l3042_304239

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l3042_304239


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3042_304248

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, x > 2 → (x + 1) * (x - 2) > 0) ∧
  (∃ x : ℝ, (x + 1) * (x - 2) > 0 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3042_304248


namespace NUMINAMATH_CALUDE_max_value_and_sum_l3042_304236

theorem max_value_and_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 4050) : 
  let M := a*c + 3*b*c + 2*c*d + 8*d*e
  ∃ (a_M b_M c_M d_M e_M : ℝ),
    (∀ a' b' c' d' e' : ℝ, 
      0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 
      a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 4050 → 
      a'*c' + 3*b'*c' + 2*c'*d' + 8*d'*e' ≤ M) ∧
    M = 4050 * Real.sqrt 14 ∧
    M + a_M + b_M + c_M + d_M + e_M = 4050 * Real.sqrt 14 + 90 :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_sum_l3042_304236


namespace NUMINAMATH_CALUDE_problem_solution_l3042_304251

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 30)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3042_304251


namespace NUMINAMATH_CALUDE_inequality_solution_range_of_a_l3042_304276

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (x a : ℝ) : ℝ := 2 * |x| + a

-- Part 1: Inequality solution
theorem inequality_solution :
  {x : ℝ | f x ≤ g x (-1)} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a
theorem range_of_a (h : ∃ x₀ : ℝ, f x₀ ≥ (1/2) * g x₀ a) :
  a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_of_a_l3042_304276


namespace NUMINAMATH_CALUDE_second_ruler_alignment_l3042_304214

/-- Represents a small ruler in relation to the large ruler -/
structure SmallRuler where
  large_units : ℚ  -- Number of units on the large ruler
  small_units : ℚ  -- Number of units on the small ruler

/-- Represents the set square system with two small rulers and a large ruler -/
structure SetSquare where
  first_ruler : SmallRuler
  second_ruler : SmallRuler
  point_b : ℚ  -- Position of point B on the large ruler

/-- Main theorem statement -/
theorem second_ruler_alignment (s : SetSquare) : 
  s.first_ruler = SmallRuler.mk 11 10 →   -- First ruler divides 11 units into 10
  s.second_ruler = SmallRuler.mk 9 10 →   -- Second ruler divides 9 units into 10
  18 < s.point_b ∧ s.point_b < 19 →       -- Point B is between 18 and 19
  (s.point_b + 3 * s.first_ruler.large_units / s.first_ruler.small_units).floor = 
    (s.point_b + 3 * s.first_ruler.large_units / s.first_ruler.small_units) →  
    -- 3rd unit of first ruler coincides with an integer
  ∃ k : ℕ, (s.point_b + 7 * s.second_ruler.large_units / s.second_ruler.small_units) = ↑k :=
by sorry

end NUMINAMATH_CALUDE_second_ruler_alignment_l3042_304214


namespace NUMINAMATH_CALUDE_x_powers_sum_l3042_304212

theorem x_powers_sum (x : ℝ) (h : x + 1/x = 10) : 
  x^2 + 1/x^2 = 98 ∧ x^3 + 1/x^3 = 970 := by
  sorry

end NUMINAMATH_CALUDE_x_powers_sum_l3042_304212


namespace NUMINAMATH_CALUDE_sector_perimeter_l3042_304292

/-- Given a circular sector with central angle 2/3π and area 3π, its perimeter is 6 + 2π. -/
theorem sector_perimeter (θ : Real) (S : Real) (R : Real) (l : Real) :
  θ = (2/3) * Real.pi →
  S = 3 * Real.pi →
  S = (1/2) * θ * R^2 →
  l = θ * R →
  (l + 2 * R) = 6 + 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3042_304292


namespace NUMINAMATH_CALUDE_tv_show_average_episodes_l3042_304223

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

#check tv_show_average_episodes

end NUMINAMATH_CALUDE_tv_show_average_episodes_l3042_304223


namespace NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l3042_304210

def initial_bales : ℕ := 15
def initial_cost_per_bale : ℕ := 20
def new_cost_per_bale : ℕ := 27

theorem additional_cost_for_new_requirements :
  (initial_bales * 3 * new_cost_per_bale) - (initial_bales * initial_cost_per_bale) = 915 := by
  sorry

end NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l3042_304210


namespace NUMINAMATH_CALUDE_jessica_milk_problem_l3042_304269

theorem jessica_milk_problem (initial_milk : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial_milk = 5 →
  given_away = 16 / 3 →
  remaining = initial_milk - given_away →
  remaining = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_jessica_milk_problem_l3042_304269


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_and_real_l3042_304250

theorem quadratic_roots_equal_and_real (a c : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, a * x^2 - 2 * x * Real.sqrt 2 + c = 0) ∧
  ((-2 * Real.sqrt 2)^2 - 4 * a * c = 0) →
  ∃! x : ℝ, a * x^2 - 2 * x * Real.sqrt 2 + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_equal_and_real_l3042_304250


namespace NUMINAMATH_CALUDE_journey_time_proof_l3042_304237

theorem journey_time_proof (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed1 = 40)
  (h4 : speed2 = 60) :
  ∃ (t1 : ℝ), t1 = 3 ∧ 
  ∃ (t2 : ℝ), t1 + t2 = total_time ∧ 
  speed1 * t1 + speed2 * t2 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_journey_time_proof_l3042_304237


namespace NUMINAMATH_CALUDE_romney_value_l3042_304220

theorem romney_value (N O : ℕ) (a b c d e f : ℕ) :
  (0 < N) → (N < O) →  -- N/O is a proper fraction
  (N = 4) → (O = 7) →  -- N/O = 4/7
  (0 ≤ a) → (a ≤ 9) → (0 ≤ b) → (b ≤ 9) → (0 ≤ c) → (c ≤ 9) →
  (0 ≤ d) → (d ≤ 9) → (0 ≤ e) → (e ≤ 9) → (0 ≤ f) → (f ≤ 9) →  -- Each letter is a digit
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (a ≠ e) → (a ≠ f) →
  (b ≠ c) → (b ≠ d) → (b ≠ e) → (b ≠ f) →
  (c ≠ d) → (c ≠ e) → (c ≠ f) →
  (d ≠ e) → (d ≠ f) →
  (e ≠ f) →  -- All letters are distinct
  (N : ℚ) / (O : ℚ) = 
    (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (d : ℚ) / 10000 + (e : ℚ) / 100000 + (f : ℚ) / 1000000 +
    (a : ℚ) / 1000000 + (b : ℚ) / 10000000 + (c : ℚ) / 100000000 + (d : ℚ) / 1000000000 + (e : ℚ) / 10000000000 + (f : ℚ) / 100000000000 +
    (a : ℚ) / 100000000000 + (b : ℚ) / 1000000000000 + (c : ℚ) / 10000000000000 + (d : ℚ) / 100000000000000 + (e : ℚ) / 1000000000000000 + (f : ℚ) / 10000000000000000 +
    (a : ℚ) / 10000000000000000 + (b : ℚ) / 100000000000000000 + (c : ℚ) / 1000000000000000000 + (d : ℚ) / 10000000000000000000 + (e : ℚ) / 100000000000000000000 + (f : ℚ) / 1000000000000000000000 →  -- Decimal representation
  a = 5 ∧ b = 7 ∧ c = 1 ∧ d = 4 ∧ e = 2 ∧ f = 8 := by
  sorry

end NUMINAMATH_CALUDE_romney_value_l3042_304220


namespace NUMINAMATH_CALUDE_total_dots_is_89_l3042_304253

/-- The number of ladybugs caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of dots on each ladybug caught on Monday -/
def monday_dots_per_ladybug : ℕ := 6

/-- The number of ladybugs caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of ladybugs caught on Wednesday -/
def wednesday_ladybugs : ℕ := 4

/-- The number of dots on each ladybug caught on Tuesday -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1

/-- The number of dots on each ladybug caught on Wednesday -/
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- The total number of dots on all ladybugs caught over three days -/
def total_dots : ℕ :=
  monday_ladybugs * monday_dots_per_ladybug +
  tuesday_ladybugs * tuesday_dots_per_ladybug +
  wednesday_ladybugs * wednesday_dots_per_ladybug

theorem total_dots_is_89 : total_dots = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_dots_is_89_l3042_304253


namespace NUMINAMATH_CALUDE_range_of_a_l3042_304270

/-- Proposition p: x^2 + 2ax + 4 > 0 holds for all x ∈ ℝ -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: x^2 - (a+1)x + 1 ≤ 0 has an empty solution set -/
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - (a+1)*x + 1 > 0

/-- The disjunction p ∨ q is true -/
axiom h1 (a : ℝ) : p a ∨ q a

/-- The conjunction p ∧ q is false -/
axiom h2 (a : ℝ) : ¬(p a ∧ q a)

/-- The range of values for a is (-3, -2] ∪ [1, 2) -/
theorem range_of_a : 
  {a : ℝ | (a > -3 ∧ a ≤ -2) ∨ (a ≥ 1 ∧ a < 2)} = {a : ℝ | p a ∨ q a ∧ ¬(p a ∧ q a)} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3042_304270


namespace NUMINAMATH_CALUDE_countMultiplesIs943_l3042_304283

/-- The number of integers between 1 and 3000 (inclusive) that are multiples of 5 or 7 but not multiples of 35 -/
def countMultiples : ℕ := sorry

theorem countMultiplesIs943 : countMultiples = 943 := by sorry

end NUMINAMATH_CALUDE_countMultiplesIs943_l3042_304283


namespace NUMINAMATH_CALUDE_equilateral_iff_complex_equation_l3042_304235

/-- A primitive cube root of unity -/
noncomputable def w : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))

/-- Definition of an equilateral triangle in the complex plane -/
def is_equilateral (z₁ z₂ z₃ : ℂ) : Prop :=
  Complex.abs (z₂ - z₁) = Complex.abs (z₃ - z₂) ∧
  Complex.abs (z₃ - z₂) = Complex.abs (z₁ - z₃)

/-- Definition of counterclockwise orientation -/
def is_counterclockwise (z₁ z₂ z₃ : ℂ) : Prop :=
  (z₂ - z₁).arg < (z₃ - z₁).arg ∧ (z₃ - z₁).arg < (z₂ - z₁).arg + Real.pi

/-- Theorem: A triangle is equilateral iff it satisfies the given complex equation -/
theorem equilateral_iff_complex_equation (z₁ z₂ z₃ : ℂ) :
  is_counterclockwise z₁ z₂ z₃ →
  is_equilateral z₁ z₂ z₃ ↔ z₁ + w * z₂ + w^2 * z₃ = 0 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_iff_complex_equation_l3042_304235


namespace NUMINAMATH_CALUDE_probability_kings_or_aces_value_l3042_304293

/-- A standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)

/-- The probability of drawing either three kings or at least 2 aces
    when 3 cards are selected randomly from a standard deck. -/
def probability_kings_or_aces (d : Deck) : ℚ :=
  sorry

/-- The theorem stating the probability of drawing either three kings or at least 2 aces
    when 3 cards are selected randomly from a standard deck. -/
theorem probability_kings_or_aces_value (d : Deck) 
  (h1 : d.total_cards = 52)
  (h2 : d.num_aces = 4)
  (h3 : d.num_kings = 4) :
  probability_kings_or_aces d = 74 / 5525 :=
sorry

end NUMINAMATH_CALUDE_probability_kings_or_aces_value_l3042_304293


namespace NUMINAMATH_CALUDE_line_plane_intersection_l3042_304280

/-- The point of intersection between a line and a plane in 3D space. -/
theorem line_plane_intersection
  (A1 A2 A3 A4 : ℝ × ℝ × ℝ)
  (h1 : A1 = (1, 2, -3))
  (h2 : A2 = (1, 0, 1))
  (h3 : A3 = (-2, -1, 6))
  (h4 : A4 = (0, -5, -4)) :
  ∃ P : ℝ × ℝ × ℝ,
    (∃ t : ℝ, P = A4 + t • (A4 - A1)) ∧
    (∃ u v : ℝ, P = A1 + u • (A2 - A1) + v • (A3 - A1)) :=
by sorry


end NUMINAMATH_CALUDE_line_plane_intersection_l3042_304280


namespace NUMINAMATH_CALUDE_ladder_distance_l3042_304288

theorem ladder_distance (angle : Real) (length : Real) (distance : Real) : 
  angle = 60 * π / 180 →
  length = 19 →
  distance = length * Real.cos angle →
  distance = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l3042_304288


namespace NUMINAMATH_CALUDE_f_monotone_implies_a_range_l3042_304211

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_monotone_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → 2 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_implies_a_range_l3042_304211


namespace NUMINAMATH_CALUDE_smallest_x_l3042_304245

theorem smallest_x (x a b : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (h3 : x > 0) : x ≥ 200000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_l3042_304245


namespace NUMINAMATH_CALUDE_product_equals_sum_exists_percentage_calculation_l3042_304200

-- Problem 1
theorem product_equals_sum_exists : ∃ (a b c : ℤ), a * b * c = a + b + c := by
  sorry

-- Problem 2
theorem percentage_calculation : (12.5 / 100) * 44 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_exists_percentage_calculation_l3042_304200


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3042_304222

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (7 * a) % 56 = 1 ∧ (13 * b) % 56 = 1 ∧ 
  (3 * a + 9 * b) % 56 = 29 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3042_304222


namespace NUMINAMATH_CALUDE_largest_element_of_A_l3042_304227

def A : Set ℝ := {x | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ x = n ^ (1 / n : ℝ)}

theorem largest_element_of_A : ∀ x ∈ A, x ≤ 3 ^ (1 / 3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_largest_element_of_A_l3042_304227


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3042_304298

theorem average_of_remaining_numbers
  (n : ℕ)
  (total : ℕ)
  (subset_sum : ℕ)
  (h1 : n = 5)
  (h2 : total = n * 20)
  (h3 : subset_sum = 48)
  (h4 : subset_sum < total) :
  (total - subset_sum) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3042_304298


namespace NUMINAMATH_CALUDE_walkers_speed_l3042_304206

theorem walkers_speed (speed_man2 : ℝ) (distance_apart : ℝ) (time : ℝ) (speed_man1 : ℝ) :
  speed_man2 = 12 →
  distance_apart = 2 →
  time = 1 →
  speed_man2 * time - speed_man1 * time = distance_apart →
  speed_man1 = 10 := by
sorry

end NUMINAMATH_CALUDE_walkers_speed_l3042_304206


namespace NUMINAMATH_CALUDE_max_protesters_l3042_304246

theorem max_protesters (population : ℕ) (reforms : ℕ) (dislike_per_reform : ℕ) :
  population = 96 →
  reforms = 5 →
  dislike_per_reform = population / 2 →
  (∀ r : ℕ, r ≤ reforms → dislike_per_reform = population / 2) →
  (∃ max_protesters : ℕ,
    max_protesters ≤ population ∧
    max_protesters * (reforms / 2 + 1) ≤ reforms * dislike_per_reform ∧
    ∀ n : ℕ, n ≤ population →
      n * (reforms / 2 + 1) ≤ reforms * dislike_per_reform →
      n ≤ max_protesters) →
  (∃ max_protesters : ℕ, max_protesters = 80) :=
by sorry

end NUMINAMATH_CALUDE_max_protesters_l3042_304246


namespace NUMINAMATH_CALUDE_root_sum_product_l3042_304287

theorem root_sum_product (p q : ℝ) : 
  (Complex.I * 2 - 1)^2 + p * (Complex.I * 2 - 1) + q = 0 → p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_product_l3042_304287


namespace NUMINAMATH_CALUDE_part1_part2_l3042_304279

-- Definition of "shifted equation"
def is_shifted_equation (f g : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x = 0 ∧ g y = 0 ∧ x = y + 1

-- Part 1
theorem part1 : is_shifted_equation (λ x => 2*x + 1) (λ x => 2*x + 3) := by sorry

-- Part 2
theorem part2 : ∃ m : ℝ, 
  is_shifted_equation 
    (λ x => 3*(x-1) - m - (m+3)/2) 
    (λ x => 2*(x-3) - 1 - (3-(x+1))) ∧ 
  m = 5 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3042_304279


namespace NUMINAMATH_CALUDE_james_caprisun_purchase_l3042_304258

/-- The total cost of James' Capri-sun purchase -/
def total_cost (num_boxes : ℕ) (pouches_per_box : ℕ) (cost_per_pouch : ℚ) : ℚ :=
  (num_boxes * pouches_per_box : ℕ) * cost_per_pouch

/-- Theorem stating the total cost of James' purchase -/
theorem james_caprisun_purchase :
  total_cost 10 6 (20 / 100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_caprisun_purchase_l3042_304258


namespace NUMINAMATH_CALUDE_cubic_function_range_l3042_304207

/-- If f(x) = x^3 - a and the graph of f(x) does not pass through the second quadrant, then a ∈ [0, +∞) -/
theorem cubic_function_range (a : ℝ) : 
  (∀ x : ℝ, (x ≤ 0 ∧ x^3 - a ≥ 0) → False) → 
  a ∈ Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_range_l3042_304207


namespace NUMINAMATH_CALUDE_common_remainder_exists_l3042_304268

theorem common_remainder_exists : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 312837 % n = 310650 % n ∧ 312837 % n = 96 := by
  sorry

end NUMINAMATH_CALUDE_common_remainder_exists_l3042_304268


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3042_304266

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3042_304266


namespace NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eight_l3042_304230

theorem thirteen_pow_seven_mod_eight : 13^7 % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eight_l3042_304230


namespace NUMINAMATH_CALUDE_asymptote_equation_correct_l3042_304265

/-- Represents a hyperbola with equation x^2 - y^2/b^2 = 1 and one focus at (2, 0) -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- The equation of the asymptotes of the hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (Real.sqrt 3 * x = y) ∨ (Real.sqrt 3 * x = -y)

/-- Theorem stating that the equation of the asymptotes is correct -/
theorem asymptote_equation_correct (h : Hyperbola) :
  asymptote_equation h = λ x y => (Real.sqrt 3 * x = y) ∨ (Real.sqrt 3 * x = -y) :=
by sorry

end NUMINAMATH_CALUDE_asymptote_equation_correct_l3042_304265


namespace NUMINAMATH_CALUDE_unique_third_rectangle_dimensions_l3042_304233

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Given three rectangles that form a larger rectangle without gaps and overlapping,
    where two of the rectangles are 3 cm × 8 cm and 2 cm × 5 cm,
    prove that there is only one possible set of dimensions for the third rectangle -/
theorem unique_third_rectangle_dimensions (r1 r2 r3 : Rectangle)
  (h1 : r1.width = 3 ∧ r1.height = 8)
  (h2 : r2.width = 2 ∧ r2.height = 5)
  (h_total_area : r1.area + r2.area + r3.area = (r1.width + r2.width + r3.width) * (r1.height + r2.height + r3.height)) :
  r3.width = 4 ∧ r3.height = 1 ∨ r3.width = 1 ∧ r3.height = 4 := by
  sorry

#check unique_third_rectangle_dimensions

end NUMINAMATH_CALUDE_unique_third_rectangle_dimensions_l3042_304233


namespace NUMINAMATH_CALUDE_sams_tuna_discount_l3042_304213

/-- Calculates the discount per coupon for a tuna purchase. -/
def discount_per_coupon (num_cans : ℕ) (num_coupons : ℕ) (paid : ℕ) (change : ℕ) (cost_per_can : ℕ) : ℕ :=
  let total_paid := paid - change
  let total_cost := num_cans * cost_per_can
  let total_discount := total_cost - total_paid
  total_discount / num_coupons

/-- Proves that the discount per coupon is 25 cents for Sam's tuna purchase. -/
theorem sams_tuna_discount :
  discount_per_coupon 9 5 2000 550 175 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sams_tuna_discount_l3042_304213


namespace NUMINAMATH_CALUDE_number_of_bags_l3042_304216

theorem number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 52) (h2 : cookies_per_bag = 2) :
  total_cookies / cookies_per_bag = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bags_l3042_304216


namespace NUMINAMATH_CALUDE_pendant_prices_and_optimal_plan_l3042_304262

/-- The price of a "Bing Dwen Dwen" pendant in yuan -/
def bing_price : ℝ := 8

/-- The price of a "Shuey Rong Rong" pendant in yuan -/
def shuey_price : ℝ := 10

/-- The cost of 2 "Bing Dwen Dwen" and 1 "Shuey Rong Rong" pendants -/
def cost1 : ℝ := 26

/-- The cost of 4 "Bing Dwen Dwen" and 3 "Shuey Rong Rong" pendants -/
def cost2 : ℝ := 62

/-- The total number of pendants to purchase -/
def total_pendants : ℕ := 100

/-- The number of "Bing Dwen Dwen" pendants in the optimal plan -/
def optimal_bing : ℕ := 75

/-- The number of "Shuey Rong Rong" pendants in the optimal plan -/
def optimal_shuey : ℕ := 25

/-- The minimum cost for the optimal plan -/
def min_cost : ℝ := 850

theorem pendant_prices_and_optimal_plan :
  (2 * bing_price + shuey_price = cost1) ∧
  (4 * bing_price + 3 * shuey_price = cost2) ∧
  (optimal_bing + optimal_shuey = total_pendants) ∧
  (3 * optimal_shuey ≥ optimal_bing) ∧
  (optimal_bing * bing_price + optimal_shuey * shuey_price = min_cost) ∧
  (∀ x y : ℕ, x + y = total_pendants → 3 * y ≥ x → 
    x * bing_price + y * shuey_price ≥ min_cost) :=
by sorry

#check pendant_prices_and_optimal_plan

end NUMINAMATH_CALUDE_pendant_prices_and_optimal_plan_l3042_304262


namespace NUMINAMATH_CALUDE_solve_jewelry_store_problem_l3042_304221

/-- Represents the jewelry store inventory problem --/
def jewelry_store_problem (necklace_capacity ring_capacity bracelet_capacity : ℕ)
  (current_rings current_bracelets : ℕ)
  (price_necklace price_ring price_bracelet : ℕ)
  (total_cost : ℕ) : Prop :=
  let rings_needed := ring_capacity - current_rings
  let bracelets_needed := bracelet_capacity - current_bracelets
  let necklaces_on_stand := necklace_capacity - 
    ((total_cost - price_ring * rings_needed - price_bracelet * bracelets_needed) / price_necklace)
  necklaces_on_stand = 5

/-- The main theorem stating the solution to the jewelry store problem --/
theorem solve_jewelry_store_problem :
  jewelry_store_problem 12 30 15 18 8 4 10 5 183 := by
  sorry

end NUMINAMATH_CALUDE_solve_jewelry_store_problem_l3042_304221


namespace NUMINAMATH_CALUDE_candidate_votes_proof_l3042_304203

theorem candidate_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 80 / 100 →
  ⌊(1 - invalid_percentage) * candidate_percentage * total_votes⌋ = 380800 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_proof_l3042_304203


namespace NUMINAMATH_CALUDE_unique_solution_l3042_304289

/-- The exponent function for our problem -/
def f (m : ℕ+) : ℤ := m^2 - 2*m - 3

/-- The condition that the exponent is negative -/
def condition1 (m : ℕ+) : Prop := f m < 0

/-- The condition that the exponent is odd -/
def condition2 (m : ℕ+) : Prop := ∃ k : ℤ, f m = 2*k + 1

/-- The theorem stating that 2 is the only positive integer satisfying all conditions -/
theorem unique_solution :
  ∃! m : ℕ+, condition1 m ∧ condition2 m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3042_304289


namespace NUMINAMATH_CALUDE_trapezoid_segment_property_l3042_304274

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 120 = longer_base
  midline_ratio_condition : midline_ratio = 3 / 4

/-- The main theorem -/
theorem trapezoid_segment_property (t : Trapezoid) : 
  ⌊(t.equal_area_segment^2) / 120⌋ = 217 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_property_l3042_304274


namespace NUMINAMATH_CALUDE_f_n_ratio_theorem_l3042_304278

noncomputable section

def f (x : ℝ) : ℝ := (x^2 + 1) / (2*x)

def f_n : ℕ → ℝ → ℝ
| 0, x => x
| n+1, x => f (f_n n x)

def N (n : ℕ) : ℕ := 2^n

theorem f_n_ratio_theorem (x : ℝ) (n : ℕ) (hx : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) :
  (f_n n x) / (f_n (n+1) x) = 1 + 1 / f ((((x+1)/(x-1)) ^ (N n))) :=
sorry

end NUMINAMATH_CALUDE_f_n_ratio_theorem_l3042_304278


namespace NUMINAMATH_CALUDE_addition_problem_l3042_304281

theorem addition_problem (x y : ℕ) :
  (x + y = x + 2000) ∧ (x + y = y + 6) →
  (x = 6 ∧ y = 2000 ∧ x + y = 2006) :=
by sorry

end NUMINAMATH_CALUDE_addition_problem_l3042_304281


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3042_304228

/-- The number of people sitting around the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of all math majors sitting in consecutive seats -/
def prob_consecutive_math : ℚ := 2 / 55

theorem math_majors_consecutive_probability :
  (total_people = math_majors + physics_majors + biology_majors) →
  (prob_consecutive_math = 2 / 55) := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3042_304228


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l3042_304238

/-- The average price of books Sandy bought given the conditions -/
def average_price_per_book (books_shop1 books_shop2 : ℕ) (price_shop1 price_shop2 : ℚ) : ℚ :=
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2)

/-- Theorem stating that the average price Sandy paid per book is $18 -/
theorem sandy_average_book_price :
  let books_shop1 : ℕ := 65
  let books_shop2 : ℕ := 55
  let price_shop1 : ℚ := 1280
  let price_shop2 : ℚ := 880
  average_price_per_book books_shop1 books_shop2 price_shop1 price_shop2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_sandy_average_book_price_l3042_304238


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_implies_segment_length_l3042_304229

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (0, 3)

-- Define the triangle area function
noncomputable def triangleArea (P A B : ℝ × ℝ) : ℝ := sorry

-- Define the length function
noncomputable def segmentLength (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_area_implies_segment_length :
  ∀ P A : ℝ × ℝ,
  ellipse P.1 P.2 →
  (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → triangleArea Q A B ≥ 1) →
  (∃ R : ℝ × ℝ, ellipse R.1 R.2 ∧ triangleArea R A B = 5) →
  segmentLength A B = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_implies_segment_length_l3042_304229


namespace NUMINAMATH_CALUDE_diff_eq_linear_solution_l3042_304260

/-- The differential equation y'' = 0 has a general solution of the form y = C₁x + C₂,
    where C₁ and C₂ are arbitrary constants. -/
theorem diff_eq_linear_solution (x : ℝ) :
  ∃ (y : ℝ → ℝ) (C₁ C₂ : ℝ), (∀ x, (deriv^[2] y) x = 0) ∧ (∀ x, y x = C₁ * x + C₂) := by
  sorry

end NUMINAMATH_CALUDE_diff_eq_linear_solution_l3042_304260


namespace NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l3042_304272

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b, if q satisfies ‖q - b‖ = 3 ‖q - a‖, 
    then q is at a fixed distance from (9/8)a + (-1/8)b. -/
theorem fixed_distance_from_linear_combination (a b q : E) 
  (h : ‖q - b‖ = 3 * ‖q - a‖) :
  ∃ (c : ℝ), ∀ (q : E), ‖q - b‖ = 3 * ‖q - a‖ → 
    ‖q - ((9/8 : ℝ) • a + (-1/8 : ℝ) • b)‖ = c :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l3042_304272


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_quadratic_equation_constant_geometric_sequence_ratio_exponential_expression_l3042_304225

-- Problem 1
theorem consecutive_odd_integers_sum (k : ℤ) : 
  k + (k + 2) + (k + 4) = 51 → k = 15 := by sorry

-- Problem 2
theorem quadratic_equation_constant (x k a C : ℝ) :
  x^2 + 6*x + k = (x + a)^2 + C → C = 6 := by sorry

-- Problem 3
theorem geometric_sequence_ratio (p q r s R : ℝ) :
  p/q = 2 ∧ q/r = 2 ∧ r/s = 2 ∧ R = p/s → R = 8 := by sorry

-- Problem 4
theorem exponential_expression (n : ℕ) (A : ℝ) :
  A = (3^n * 9^(n+1)) / 27^(n-1) → A = 729 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_quadratic_equation_constant_geometric_sequence_ratio_exponential_expression_l3042_304225


namespace NUMINAMATH_CALUDE_real_roots_range_l3042_304299

theorem real_roots_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) → 
  a ≤ -3/2 ∨ a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_real_roots_range_l3042_304299


namespace NUMINAMATH_CALUDE_point_p_coordinates_l3042_304247

/-- Given points A(2, 3) and B(4, -3), if a point P satisfies |AP| = 3/2 |PB|, 
    then P has coordinates (16/5, 0). -/
theorem point_p_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  ‖A - P‖ = (3/2) * ‖P - B‖ → 
  P = (16/5, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l3042_304247


namespace NUMINAMATH_CALUDE_sum_of_roots_l3042_304264

theorem sum_of_roots (p q r : ℕ+) : 
  4 * (7^(1/4) - 6^(1/4) : ℝ) = p^(1/4) + q^(1/4) - r^(1/4) → p + q + r = 122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3042_304264


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_proposition_l3042_304205

theorem arithmetic_geometric_sequence_proposition :
  let p : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) - a n = a 1 - a 0) → a 1 - a 0 ≠ 0
  let q : Prop := ∀ (g : ℕ → ℝ), (∀ n, g (n + 1) / g n = g 1 / g 0) → g 1 / g 0 ≠ 1
  ¬p ∧ ¬q → (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_proposition_l3042_304205


namespace NUMINAMATH_CALUDE_equation_solutions_l3042_304295

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 1 ∧ 
    (∀ x : ℝ, x * (x + 2) = (x + 2) ↔ x = x₁ ∨ x = x₂)) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (3 - Real.sqrt 7) / 2 ∧ y₂ = (3 + Real.sqrt 7) / 2 ∧ 
    (∀ x : ℝ, 2 * x^2 - 6 * x + 1 = 0 ↔ x = y₁ ∨ x = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3042_304295


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3042_304254

theorem solution_set_inequality (x : ℝ) : 
  1 / x < 1 / 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3042_304254


namespace NUMINAMATH_CALUDE_inverse_function_point_sum_l3042_304242

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Given condition: (2, 4) is on the graph of y = f(x)/3
axiom point_on_f : f 2 = 12

-- Theorem to prove
theorem inverse_function_point_sum :
  ∃ a b : ℝ, f_inv a = 3 * b ∧ a + b = 38 / 3 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_sum_l3042_304242


namespace NUMINAMATH_CALUDE_total_wheels_is_102_l3042_304296

/-- The number of wheels Dimitri saw at the park -/
def total_wheels : ℕ :=
  let bicycle_wheels := 2
  let tricycle_wheels := 3
  let unicycle_wheels := 1
  let scooter_wheels := 4
  let men_on_bicycles := 6
  let women_on_bicycles := 5
  let boys_on_tricycles := 8
  let girls_on_tricycles := 7
  let boys_on_unicycles := 2
  let girls_on_unicycles := 1
  let boys_on_scooters := 5
  let girls_on_scooters := 3
  (men_on_bicycles + women_on_bicycles) * bicycle_wheels +
  (boys_on_tricycles + girls_on_tricycles) * tricycle_wheels +
  (boys_on_unicycles + girls_on_unicycles) * unicycle_wheels +
  (boys_on_scooters + girls_on_scooters) * scooter_wheels

theorem total_wheels_is_102 : total_wheels = 102 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_102_l3042_304296


namespace NUMINAMATH_CALUDE_brother_age_proof_l3042_304284

def brother_age_in_5_years (nick_age : ℕ) : ℕ :=
  let sister_age := nick_age + 6
  let brother_age := (nick_age + sister_age) / 2
  brother_age + 5

theorem brother_age_proof (nick_age : ℕ) (h : nick_age = 13) :
  brother_age_in_5_years nick_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_brother_age_proof_l3042_304284


namespace NUMINAMATH_CALUDE_right_triangle_proof_l3042_304241

theorem right_triangle_proof (n : ℝ) (hn : n > 0) :
  let a := 2*n^2 + 2*n + 1
  let b := 2*n^2 + 2*n
  let c := 2*n + 1
  a^2 = b^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_proof_l3042_304241


namespace NUMINAMATH_CALUDE_distracted_scientist_waiting_time_l3042_304282

/-- The average waiting time for the first bite given the conditions of the distracted scientist problem -/
theorem distracted_scientist_waiting_time 
  (first_rod_bites : ℝ) 
  (second_rod_bites : ℝ) 
  (total_bites : ℝ) 
  (time_interval : ℝ) 
  (h1 : first_rod_bites = 3) 
  (h2 : second_rod_bites = 2) 
  (h3 : total_bites = first_rod_bites + second_rod_bites) 
  (h4 : time_interval = 6) : 
  (time_interval / total_bites) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_distracted_scientist_waiting_time_l3042_304282


namespace NUMINAMATH_CALUDE_division_problem_l3042_304285

theorem division_problem (total : ℚ) (a b c : ℚ) 
  (h1 : total = 544)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : c = 384 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3042_304285


namespace NUMINAMATH_CALUDE_polygon_sides_theorem_l3042_304232

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The measure of one interior angle in a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

/-- Predicate to check if a pair of numbers satisfies the polygon conditions -/
def satisfies_conditions (x y : ℕ) : Prop :=
  y = x + 10 ∧
  num_diagonals y - num_diagonals x = interior_angle x - 15

theorem polygon_sides_theorem :
  ∀ x y : ℕ, satisfies_conditions x y → (x = 5 ∧ y = 15) ∨ (x = 8 ∧ y = 18) :=
sorry

#check polygon_sides_theorem

end NUMINAMATH_CALUDE_polygon_sides_theorem_l3042_304232


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3042_304231

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_distance : ℕ) : 
  grasshopper_jump = 17 → extra_distance = 22 → grasshopper_jump + extra_distance = 39 :=
by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l3042_304231


namespace NUMINAMATH_CALUDE_discount_calculation_l3042_304256

theorem discount_calculation (CP : ℝ) (MP SP discount : ℝ) : 
  MP = 1.1 * CP → 
  SP = 0.99 * CP → 
  discount = MP - SP → 
  discount = 0.11 * CP :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l3042_304256


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3042_304226

theorem quadratic_factorization_sum : ∃ (a b c d : ℝ),
  (∀ x, x^2 + 23*x + 132 = (x + a) * (x + b)) ∧
  (∀ x, x^2 - 25*x + 168 = (x - c) * (x - d)) ∧
  (a + c + d = 42) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3042_304226


namespace NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l3042_304243

theorem no_three_digit_perfect_square_sum : 
  ∀ (a b c : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 1 ≤ a → 
  ¬∃ (m : ℕ), m^2 = 111 * (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l3042_304243


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3042_304224

theorem necessary_not_sufficient (a b h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, |a - 1| < h ∧ |b - 1| < h → |a - b| < 2 * h) ∧
  (∃ a b : ℝ, |a - b| < 2 * h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3042_304224


namespace NUMINAMATH_CALUDE_min_x_given_inequality_l3042_304255

theorem min_x_given_inequality (x : ℝ) :
  (∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) →
  x ≥ -1 ∧ ∀ y : ℝ, (∀ a : ℝ, a > 0 → y^2 ≤ 1 + a) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_min_x_given_inequality_l3042_304255


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3042_304201

def N : ℕ := 48 * 48 * 55 * 125 * 81

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3042_304201


namespace NUMINAMATH_CALUDE_paco_cookie_difference_l3042_304257

/-- Given Paco's cookie situation, prove that he ate 9 more cookies than he gave away -/
theorem paco_cookie_difference (initial_cookies : ℕ) (cookies_given : ℕ) (cookies_eaten : ℕ)
  (h1 : initial_cookies = 41)
  (h2 : cookies_given = 9)
  (h3 : cookies_eaten = 18) :
  cookies_eaten - cookies_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookie_difference_l3042_304257


namespace NUMINAMATH_CALUDE_larger_integer_value_l3042_304240

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℝ) / (b : ℝ) = 7 / 3)
  (h_product : (a : ℕ) * b = 168) : 
  (a : ℝ) = 14 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3042_304240
