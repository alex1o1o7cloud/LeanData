import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_of_factors_48_l3324_332454

theorem max_sum_of_factors_48 :
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧
  ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_48_l3324_332454


namespace NUMINAMATH_CALUDE_speed_difference_l3324_332486

/-- Proves that the difference between the car's and truck's average speeds is 18 km/h -/
theorem speed_difference (truck_distance : ℝ) (truck_time : ℝ) (car_time : ℝ) (distance_difference : ℝ)
  (h1 : truck_distance = 296)
  (h2 : truck_time = 8)
  (h3 : car_time = 5.5)
  (h4 : distance_difference = 6.5)
  (h5 : (truck_distance + distance_difference) / car_time > truck_distance / truck_time) :
  (truck_distance + distance_difference) / car_time - truck_distance / truck_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l3324_332486


namespace NUMINAMATH_CALUDE_negation_equivalence_l3324_332482

theorem negation_equivalence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + a > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3324_332482


namespace NUMINAMATH_CALUDE_triangle_angle_determination_l3324_332451

theorem triangle_angle_determination (a b c A B C : ℝ) : 
  a = Real.sqrt 3 → 
  b = Real.sqrt 2 → 
  B = π / 4 → 
  (a = 2 * Real.sin (A / 2)) → 
  (b = 2 * Real.sin (B / 2)) → 
  (c = 2 * Real.sin (C / 2)) → 
  A + B + C = π → 
  (A = π / 3 ∨ A = 2 * π / 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_determination_l3324_332451


namespace NUMINAMATH_CALUDE_zayne_revenue_l3324_332432

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  single_price : ℕ  -- Price of a single bracelet
  pair_price : ℕ    -- Price of a pair of bracelets
  initial_stock : ℕ -- Initial number of bracelets
  single_sale_revenue : ℕ -- Revenue from selling single bracelets

/-- Calculates the total revenue from selling bracelets -/
def total_revenue (sale : BraceletSale) : ℕ :=
  let single_bracelets_sold := sale.single_sale_revenue / sale.single_price
  let remaining_bracelets := sale.initial_stock - single_bracelets_sold
  let pairs_sold := remaining_bracelets / 2
  let pair_revenue := pairs_sold * sale.pair_price
  sale.single_sale_revenue + pair_revenue

/-- Theorem stating that Zayne's total revenue is $132 -/
theorem zayne_revenue :
  ∃ (sale : BraceletSale),
    sale.single_price = 5 ∧
    sale.pair_price = 8 ∧
    sale.initial_stock = 30 ∧
    sale.single_sale_revenue = 60 ∧
    total_revenue sale = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_zayne_revenue_l3324_332432


namespace NUMINAMATH_CALUDE_f_of_3_eq_one_over_17_l3324_332453

/-- Given f(x) = (x-2)/(4x+5), prove that f(3) = 1/17 -/
theorem f_of_3_eq_one_over_17 (f : ℝ → ℝ) (h : ∀ x, f x = (x - 2) / (4 * x + 5)) : 
  f 3 = 1 / 17 := by
sorry

end NUMINAMATH_CALUDE_f_of_3_eq_one_over_17_l3324_332453


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l3324_332464

def polynomial (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem real_roots_of_polynomial :
  ∀ x : ℝ, polynomial x = 0 ↔ x = 2 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l3324_332464


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3324_332425

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 4) (h2 : ∀ n : ℕ, a (n + 1) - a n = 3) :
  ∀ n : ℕ, a n = 3 * n + 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3324_332425


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_mean_equality_implies_y_value_proof_l3324_332496

theorem mean_equality_implies_y_value : ℝ → Prop :=
  fun y =>
    (((6 : ℝ) + 10 + 14 + 22) / 4 = (15 + y) / 2) → y = 11

-- The proof is omitted
theorem mean_equality_implies_y_value_proof : mean_equality_implies_y_value 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_mean_equality_implies_y_value_proof_l3324_332496


namespace NUMINAMATH_CALUDE_son_age_proof_l3324_332431

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l3324_332431


namespace NUMINAMATH_CALUDE_remaining_nails_l3324_332499

theorem remaining_nails (initial_nails : ℕ) (kitchen_percent : ℚ) (fence_percent : ℚ) : 
  initial_nails = 400 →
  kitchen_percent = 30 / 100 →
  fence_percent = 70 / 100 →
  initial_nails - (kitchen_percent * initial_nails).floor - 
    (fence_percent * (initial_nails - (kitchen_percent * initial_nails).floor)).floor = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_nails_l3324_332499


namespace NUMINAMATH_CALUDE_factorization_equality_l3324_332444

theorem factorization_equality (m n : ℝ) : 
  m^2 - n^2 + 2*m - 2*n = (m - n)*(m + n + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3324_332444


namespace NUMINAMATH_CALUDE_expression_simplification_l3324_332447

theorem expression_simplification (x : ℝ) : 7*x + 9 - 3*x + 15 * 2 = 4*x + 39 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3324_332447


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3324_332410

theorem solve_linear_equation (x : ℝ) (h : x - 3*x + 4*x = 120) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3324_332410


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3324_332434

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * x^2 - 8 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1) ∧
  (∀ x : ℝ, 5 * x^2 - 4 * x - 1 = 0 ↔ x = -1/5 ∨ x = 1) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3324_332434


namespace NUMINAMATH_CALUDE_sebastians_high_school_students_l3324_332417

theorem sebastians_high_school_students (s m : ℕ) : 
  s = 4 * m →  -- Sebastian's high school has 4 times as many students as Mia's
  s + m = 3000 →  -- The total number of students in both schools is 3000
  s = 2400 :=  -- Sebastian's high school has 2400 students
by sorry

end NUMINAMATH_CALUDE_sebastians_high_school_students_l3324_332417


namespace NUMINAMATH_CALUDE_max_steps_17_steps_17_possible_l3324_332463

/-- Represents the number of toothpicks used for n steps in Mandy's staircase -/
def toothpicks (n : ℕ) : ℕ := n * (n + 5)

/-- Theorem stating that 17 is the maximum number of steps that can be built with 380 toothpicks -/
theorem max_steps_17 :
  ∀ n : ℕ, toothpicks n ≤ 380 → n ≤ 17 :=
by
  sorry

/-- Theorem stating that 17 steps can indeed be built with 380 toothpicks -/
theorem steps_17_possible :
  toothpicks 17 ≤ 380 :=
by
  sorry

end NUMINAMATH_CALUDE_max_steps_17_steps_17_possible_l3324_332463


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l3324_332446

-- Define the complex cube root of unity
noncomputable def ω : ℂ := sorry

-- Define the theorem
theorem smallest_value_of_expression (a b c : ℤ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →  -- non-zero integers
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- distinct integers
  (Even a ∧ Even b ∧ Even c) →  -- even integers
  (ω^3 = 1 ∧ ω ≠ 1) →  -- properties of ω
  ∃ (min : ℝ), 
    min = Real.sqrt 12 ∧
    ∀ (x y z : ℤ), 
      (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) →
      (x ≠ y ∧ y ≠ z ∧ x ≠ z) →
      (Even x ∧ Even y ∧ Even z) →
      Complex.abs (x + y • ω + z • ω^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l3324_332446


namespace NUMINAMATH_CALUDE_second_platform_length_l3324_332419

/-- The length of the second platform given train and first platform details -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (time_first_platform : ℝ)
  (time_second_platform : ℝ)
  (h1 : train_length = 150)
  (h2 : first_platform_length = 150)
  (h3 : time_first_platform = 15)
  (h4 : time_second_platform = 20) :
  (time_second_platform * (train_length + first_platform_length) / time_first_platform) - train_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_second_platform_length_l3324_332419


namespace NUMINAMATH_CALUDE_exponent_simplification_l3324_332423

theorem exponent_simplification (x : ℝ) (hx : x ≠ 0) :
  x^5 * x^7 / x^3 = x^9 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3324_332423


namespace NUMINAMATH_CALUDE_beads_per_bracelet_l3324_332422

theorem beads_per_bracelet (num_friends : ℕ) (current_beads : ℕ) (additional_beads : ℕ) : 
  num_friends = 6 → 
  current_beads = 36 → 
  additional_beads = 12 → 
  (current_beads + additional_beads) / num_friends = 8 :=
by sorry

end NUMINAMATH_CALUDE_beads_per_bracelet_l3324_332422


namespace NUMINAMATH_CALUDE_composite_expression_l3324_332406

theorem composite_expression (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l3324_332406


namespace NUMINAMATH_CALUDE_inequality_proof_l3324_332483

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3324_332483


namespace NUMINAMATH_CALUDE_car_wash_earnings_l3324_332409

theorem car_wash_earnings (total : ℕ) (lisa : ℕ) (tommy : ℕ) : 
  total = 60 → 
  lisa = total / 2 → 
  tommy = lisa / 2 → 
  lisa - tommy = 15 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l3324_332409


namespace NUMINAMATH_CALUDE_softball_team_ratio_l3324_332462

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  men + women = 16 →
  women = men + 2 →
  (men : ℚ) / women = 7 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l3324_332462


namespace NUMINAMATH_CALUDE_range_of_T_l3324_332449

-- Define the function T
def T (x : ℝ) : ℝ := |2 * x - 1|

-- State the theorem
theorem range_of_T (x : ℝ) : 
  (∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) → 
  x ∈ Set.Ici 2 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_T_l3324_332449


namespace NUMINAMATH_CALUDE_ball_max_height_l3324_332479

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 16

/-- Theorem stating that the maximum height of the ball is 141 feet -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 141 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3324_332479


namespace NUMINAMATH_CALUDE_problem_statement_l3324_332485

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^2 + 3*b^2 ≥ 2*b*(a + b)) ∧ 
  ((1/a + 2/b = 1) → (2*a + b ≥ 8)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3324_332485


namespace NUMINAMATH_CALUDE_farmer_reward_distribution_l3324_332408

theorem farmer_reward_distribution (total_farmers : ℕ) (total_budget : ℕ) 
  (self_employed_reward : ℕ) (stable_employment_reward : ℕ) 
  (h1 : total_farmers = 60)
  (h2 : total_budget = 100000)
  (h3 : self_employed_reward = 1000)
  (h4 : stable_employment_reward = 2000) :
  ∃ (self_employed : ℕ) (stable_employment : ℕ),
    self_employed + stable_employment = total_farmers ∧
    self_employed * self_employed_reward + 
    stable_employment * (self_employed_reward + stable_employment_reward) = total_budget ∧
    self_employed = 40 ∧ 
    stable_employment = 20 := by
  sorry

end NUMINAMATH_CALUDE_farmer_reward_distribution_l3324_332408


namespace NUMINAMATH_CALUDE_john_daily_gallons_l3324_332439

-- Define the conversion rate from quarts to gallons
def quarts_per_gallon : ℚ := 4

-- Define the number of days in a week
def days_per_week : ℚ := 7

-- Define John's weekly water consumption in quarts
def john_weekly_quarts : ℚ := 42

-- Theorem to prove
theorem john_daily_gallons : 
  john_weekly_quarts / quarts_per_gallon / days_per_week = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_john_daily_gallons_l3324_332439


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3324_332458

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  terms : ℕ → ℕ
  first_term : terms 1 = 6
  is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, terms (n + 1) = (terms n : ℚ) * r

/-- The theorem stating the third term of the specific geometric sequence -/
theorem third_term_of_geometric_sequence
  (seq : GeometricSequence)
  (h_fourth : seq.terms 4 = 384) :
  seq.terms 3 = 96 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3324_332458


namespace NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l3324_332414

-- Define the complex number z
def z (a : ℝ) : ℂ := a + (a + 1) * Complex.I

-- Define the condition for a point to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Statement of the theorem
theorem first_quadrant_sufficient_not_necessary (a : ℝ) :
  (is_in_first_quadrant (z a) → a > -1) ∧
  ¬(a > -1 → is_in_first_quadrant (z a)) :=
sorry

end NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l3324_332414


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_one_l3324_332412

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  n_ge_2 : n ≥ 2
  x_not_all_equal : ∃ i j, i ≠ j ∧ x i ≠ x j
  points_on_line : ∀ i, y i = 3 * x i + 1

/-- The correlation coefficient of a set of sample data -/
def correlationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem stating that the correlation coefficient is 1 for the given conditions -/
theorem correlation_coefficient_is_one (data : SampleData) : 
  correlationCoefficient data = 1 :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_one_l3324_332412


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l3324_332489

-- Define the prices and quantities
def tshirt_price : ℚ := 8
def sweater_price : ℚ := 18
def jacket_price : ℚ := 80
def jacket_discount : ℚ := 0.1
def tshirt_quantity : ℕ := 6
def sweater_quantity : ℕ := 4
def jacket_quantity : ℕ := 5

-- Define the total cost before tax
def total_cost_before_tax : ℚ :=
  tshirt_price * tshirt_quantity +
  sweater_price * sweater_quantity +
  jacket_price * jacket_quantity * (1 - jacket_discount)

-- Define the total cost including tax
def total_cost_with_tax : ℚ := 504

-- Theorem: The sales tax percentage is 5%
theorem sales_tax_percentage :
  ∃ (tax_rate : ℚ), 
    tax_rate = 0.05 ∧
    total_cost_with_tax = total_cost_before_tax * (1 + tax_rate) :=
sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l3324_332489


namespace NUMINAMATH_CALUDE_trip_time_calculation_l3324_332445

theorem trip_time_calculation (distance : ℝ) (speed1 speed2 time1 : ℝ) 
  (h1 : speed1 = 100)
  (h2 : speed2 = 50)
  (h3 : time1 = 5)
  (h4 : distance = speed1 * time1) :
  distance / speed2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l3324_332445


namespace NUMINAMATH_CALUDE_ellipse_property_l3324_332474

/-- Definition of an ellipse with foci F₁ and F₂ -/
def Ellipse (F₁ F₂ : ℝ × ℝ) (a b : ℝ) :=
  {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1 ∧ a > b ∧ b > 0}

/-- The angle between two vectors -/
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_property (F₁ F₂ : ℝ × ℝ) (a b : ℝ) (P : ℝ × ℝ) :
  P ∈ Ellipse F₁ F₂ a b →
  angle (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) = π / 3 →
  triangle_area P F₁ F₂ = 3 * Real.sqrt 3 →
  b = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_property_l3324_332474


namespace NUMINAMATH_CALUDE_married_men_fraction_l3324_332492

-- Define the structure of the gathering
structure Gathering where
  single_women : ℕ
  married_couples : ℕ

-- Define the probability of a woman being single
def prob_single_woman (g : Gathering) : ℚ :=
  g.single_women / (g.single_women + g.married_couples)

-- Define the fraction of married men in the gathering
def fraction_married_men (g : Gathering) : ℚ :=
  g.married_couples / (g.single_women + 2 * g.married_couples)

-- Theorem statement
theorem married_men_fraction (g : Gathering) :
  prob_single_woman g = 1/3 → fraction_married_men g = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3324_332492


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3324_332418

theorem quadratic_root_value (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - 2*x + a^2 - 1 = 0 ↔ x = 0 ∨ x ≠ 0) →
  (a - 1 ≠ 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3324_332418


namespace NUMINAMATH_CALUDE_proportional_function_value_l3324_332403

/-- A function f is proportional if it can be written as f(x) = kx for some constant k -/
def IsProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = (m-2)x + m^2 - 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + m^2 - 4

theorem proportional_function_value :
  ∀ m : ℝ, IsProportional (f m) → f m (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_value_l3324_332403


namespace NUMINAMATH_CALUDE_max_above_average_students_l3324_332455

theorem max_above_average_students (n : ℕ) (h : n = 150) :
  ∃ (scores : Fin n → ℚ),
    (∃ (count : ℕ), count = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card ∧
                    count = n - 1) ∧
    ∀ (count : ℕ),
      count = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card →
      count ≤ n - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_above_average_students_l3324_332455


namespace NUMINAMATH_CALUDE_simplify_expression_l3324_332457

theorem simplify_expression (a b : ℝ) : 4 * (a - 2 * b) - 2 * (2 * a + 3 * b) = -14 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3324_332457


namespace NUMINAMATH_CALUDE_trig_simplification_l3324_332437

open Real

theorem trig_simplification (α : ℝ) (n : ℤ) :
  ((-sin (α + π) + sin (-α) - tan (2*π + α)) / 
   (tan (α + π) + cos (-α) + cos (π - α)) = -1) ∧
  ((sin (α + n*π) + sin (α - n*π)) / 
   (sin (α + n*π) * cos (α - n*π)) = 
     if n % 2 = 0 then 2 / cos α else -2 / cos α) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3324_332437


namespace NUMINAMATH_CALUDE_lower_right_is_one_l3324_332470

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a given grid satisfies the Latin square property -/
def is_latin_square (g : Grid) : Prop :=
  (∀ i j k, i ≠ k → g i j ≠ g k j) ∧ 
  (∀ i j k, j ≠ k → g i j ≠ g i k)

/-- The initial configuration of the grid -/
def initial_config (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 0 3 = 3 ∧ g 1 1 = 1 ∧ g 2 2 = 2

theorem lower_right_is_one (g : Grid) 
  (h1 : is_latin_square g) 
  (h2 : initial_config g) : 
  g 3 3 = 0 := by sorry

end NUMINAMATH_CALUDE_lower_right_is_one_l3324_332470


namespace NUMINAMATH_CALUDE_longest_side_length_l3324_332436

-- Define a triangle with angle ratio 1:2:3 and shortest side 5 cm
structure SpecialTriangle where
  -- a, b, c are the side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle A is opposite to side a, B to b, C to c
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  -- Conditions
  angle_ratio : angleA / angleB = 1/2 ∧ angleB / angleC = 2/3
  shortest_side : min a (min b c) = 5
  -- Triangle properties
  sum_angles : angleA + angleB + angleC = π
  -- Law of sines
  law_of_sines : a / (Real.sin angleA) = b / (Real.sin angleB)
                 ∧ b / (Real.sin angleB) = c / (Real.sin angleC)

-- Theorem statement
theorem longest_side_length (t : SpecialTriangle) : max t.a (max t.b t.c) = 10 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_length_l3324_332436


namespace NUMINAMATH_CALUDE_species_decline_year_l3324_332459

def species_decrease_rate : ℝ := 0.3
def threshold : ℝ := 0.05
def base_year : ℕ := 2010

def species_count (n : ℕ) : ℝ := (1 - species_decrease_rate) ^ n

theorem species_decline_year :
  ∃ k : ℕ, (species_count k < threshold) ∧ (∀ m : ℕ, m < k → species_count m ≥ threshold) ∧ (base_year + k = 2019) :=
sorry

end NUMINAMATH_CALUDE_species_decline_year_l3324_332459


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l3324_332484

theorem arithmetic_mean_geq_geometric_mean 
  (a b c : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l3324_332484


namespace NUMINAMATH_CALUDE_math_players_count_central_park_school_math_players_l3324_332427

theorem math_players_count (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) : ℕ :=
  let math_players := total_players - (physics_players - both_subjects)
  math_players

theorem central_park_school_math_players : 
  math_players_count 15 10 4 = 9 := by sorry

end NUMINAMATH_CALUDE_math_players_count_central_park_school_math_players_l3324_332427


namespace NUMINAMATH_CALUDE_vector_magnitude_l3324_332448

-- Define the vectors a and b
def a (t : ℝ) : Fin 2 → ℝ := ![t - 2, 3]
def b : Fin 2 → ℝ := ![3, -1]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

-- State the theorem
theorem vector_magnitude (t : ℝ) :
  (parallel (λ i => a t i + 2 * b i) b) →
  Real.sqrt ((a t 0) ^ 2 + (a t 1) ^ 2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3324_332448


namespace NUMINAMATH_CALUDE_max_tied_teams_seven_team_tournament_l3324_332461

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_teams : Nat)
  (no_draws : Bool)
  (round_robin : Bool)

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1)) / 2

/-- Represents the maximum number of teams that can be tied for the most wins --/
def max_tied_teams (t : Tournament) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem max_tied_teams_seven_team_tournament :
  ∀ t : Tournament, t.num_teams = 7 → t.no_draws = true → t.round_robin = true →
  max_tied_teams t = 6 :=
sorry

end NUMINAMATH_CALUDE_max_tied_teams_seven_team_tournament_l3324_332461


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3324_332421

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -300 ≡ n [ZMOD 31] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3324_332421


namespace NUMINAMATH_CALUDE_smallest_common_multiple_12_9_l3324_332466

theorem smallest_common_multiple_12_9 : ∃ n : ℕ, n > 0 ∧ 12 ∣ n ∧ 9 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 12 ∣ m ∧ 9 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_12_9_l3324_332466


namespace NUMINAMATH_CALUDE_arithmetic_progression_difference_l3324_332428

theorem arithmetic_progression_difference (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → k ≠ 0 → x ≠ y → y ≠ z → x ≠ z →
  ∃ d : ℝ, (y * (z - x) + k) - (x * (y - z) + k) = d ∧
           (z * (x - y) + k) - (y * (z - x) + k) = d →
  d = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_difference_l3324_332428


namespace NUMINAMATH_CALUDE_pasture_rent_l3324_332487

/-- Represents a milkman's grazing details -/
structure MilkmanGrazing where
  cows : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given the grazing details of milkmen -/
def totalRent (milkmen : List MilkmanGrazing) (aShare : ℕ) : ℕ :=
  let totalCowMonths := milkmen.foldl (fun acc m => acc + m.cows * m.months) 0
  let aMonths := (milkmen.head?).map (fun m => m.cows * m.months)
  match aMonths with
  | some months => (totalCowMonths * aShare) / months
  | none => 0

/-- Theorem stating that the total rent of the pasture is 3250 -/
theorem pasture_rent :
  let milkmen := [
    MilkmanGrazing.mk 24 3,  -- A
    MilkmanGrazing.mk 10 5,  -- B
    MilkmanGrazing.mk 35 4,  -- C
    MilkmanGrazing.mk 21 3   -- D
  ]
  totalRent milkmen 720 = 3250 := by
    sorry

end NUMINAMATH_CALUDE_pasture_rent_l3324_332487


namespace NUMINAMATH_CALUDE_eight_sided_die_product_l3324_332452

theorem eight_sided_die_product (x : ℕ) (h : 1 ≤ x ∧ x ≤ 8) : 
  192 ∣ (Nat.factorial 8 / x) := by sorry

end NUMINAMATH_CALUDE_eight_sided_die_product_l3324_332452


namespace NUMINAMATH_CALUDE_power_expression_l3324_332494

theorem power_expression (x y : ℝ) (a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) :
  10^(3*x + 2*y) = a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_l3324_332494


namespace NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l3324_332467

theorem polynomial_roots_in_arithmetic_progression (j k : ℝ) : 
  (∃ (b d : ℝ), d ≠ 0 ∧ 
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 400 = 0 ↔ 
      (x = b ∨ x = b + d ∨ x = b + 2*d ∨ x = b + 3*d)) ∧
    (b ≠ b + d) ∧ (b + d ≠ b + 2*d) ∧ (b + 2*d ≠ b + 3*d))
  → j = -200 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l3324_332467


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_k_l3324_332401

theorem quadratic_roots_imply_k (k : ℝ) : 
  (∀ x : ℂ, 8 * x^2 + 4 * x + k = 0 ↔ x = (-4 + Complex.I * Real.sqrt 380) / 16 ∨ x = (-4 - Complex.I * Real.sqrt 380) / 16) →
  k = 12.375 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_k_l3324_332401


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_M_l3324_332468

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x - 1| < 1}
def N : Set ℝ := {x : ℝ | x * (x - 3) < 0}

-- State the theorem
theorem M_intersect_N_equals_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_M_l3324_332468


namespace NUMINAMATH_CALUDE_seating_theorem_l3324_332402

/-- Represents a group of people seated around a round table -/
structure SeatingArrangement where
  num_men : ℕ
  num_women : ℕ

/-- A man is satisfied if at least one woman is sitting next to him -/
def is_satisfied (s : SeatingArrangement) : Prop :=
  ∃ (p : ℝ), p = 1 - (s.num_men - 1) / (s.num_men + s.num_women - 1) * (s.num_men - 2) / (s.num_men + s.num_women - 2)

/-- The probability of a specific man being satisfied -/
def satisfaction_probability (s : SeatingArrangement) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
def expected_satisfied_men (s : SeatingArrangement) : ℚ :=
  (s.num_men : ℚ) * (satisfaction_probability s)

/-- The main theorem about the seating arrangement -/
theorem seating_theorem (s : SeatingArrangement) 
  (h1 : s.num_men = 50) 
  (h2 : s.num_women = 50) : 
  is_satisfied s ∧ 
  satisfaction_probability s = 25 / 33 ∧ 
  expected_satisfied_men s = 1250 / 33 := by
  sorry


end NUMINAMATH_CALUDE_seating_theorem_l3324_332402


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l3324_332472

/-- Calculate the volume of a trapezoidal prism-shaped swimming pool -/
theorem swimming_pool_volume 
  (width : ℝ) 
  (length : ℝ) 
  (shallow_depth : ℝ) 
  (deep_depth : ℝ) 
  (h_width : width = 9) 
  (h_length : length = 12) 
  (h_shallow : shallow_depth = 1) 
  (h_deep : deep_depth = 4) : 
  (1 / 2) * (shallow_depth + deep_depth) * width * length = 270 := by
sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l3324_332472


namespace NUMINAMATH_CALUDE_quadratic_sum_l3324_332465

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 + 64 * x + 512 = a * (x + b)^2 + c) ∧ (a + b + c = 396) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3324_332465


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3324_332478

/-- Calculates the total cost of typing and revising a manuscript --/
def manuscript_cost (total_pages : ℕ) (once_revised : ℕ) (twice_revised : ℕ) 
  (initial_rate : ℕ) (revision_rate : ℕ) : ℕ :=
  let not_revised := total_pages - once_revised - twice_revised
  let initial_cost := total_pages * initial_rate
  let once_revised_cost := once_revised * revision_rate
  let twice_revised_cost := twice_revised * (2 * revision_rate)
  initial_cost + once_revised_cost + twice_revised_cost

/-- Theorem stating the total cost of the manuscript --/
theorem manuscript_cost_theorem :
  manuscript_cost 200 80 20 5 3 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3324_332478


namespace NUMINAMATH_CALUDE_g_range_g_range_complete_l3324_332426

noncomputable def g (x : ℝ) : ℝ :=
  (Real.cos x ^ 3 + 7 * Real.cos x ^ 2 - 2 * Real.cos x + 3 * Real.sin x ^ 2 - 12) / (Real.cos x - 1)

theorem g_range :
  ∀ y ∈ Set.range g, 5 ≤ y ∧ y < 9 :=
sorry

theorem g_range_complete :
  ∀ y, 5 ≤ y → y < 9 → ∃ x, Real.cos x ≠ 1 ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_g_range_g_range_complete_l3324_332426


namespace NUMINAMATH_CALUDE_pig_count_l3324_332460

theorem pig_count (initial_pigs joining_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joining_pigs = 22) : 
  initial_pigs + joining_pigs = 86 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l3324_332460


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_3_l3324_332498

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

theorem decreasing_quadratic_implies_a_geq_3 (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 3 → f a x₁ > f a x₂) →
  a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_3_l3324_332498


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l3324_332497

theorem quadratic_function_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x^2 + b * x + c| ≤ 1) →
  (a + b) * c ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l3324_332497


namespace NUMINAMATH_CALUDE_trigonometric_product_bounds_l3324_332473

theorem trigonometric_product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_bounds_l3324_332473


namespace NUMINAMATH_CALUDE_william_has_45_napkins_l3324_332441

/-- The number of napkins William has now -/
def williams_napkins (original : ℕ) (from_olivia : ℕ) (amelia_multiplier : ℕ) : ℕ :=
  original + from_olivia + amelia_multiplier * from_olivia

/-- Proof that William has 45 napkins given the conditions -/
theorem william_has_45_napkins :
  williams_napkins 15 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_william_has_45_napkins_l3324_332441


namespace NUMINAMATH_CALUDE_total_fish_count_l3324_332475

-- Define the number of fish for each person
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish
def jenny_fish : ℕ := bobby_fish - 4

-- Theorem to prove
theorem total_fish_count : billy_fish + tony_fish + sarah_fish + bobby_fish + jenny_fish = 211 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3324_332475


namespace NUMINAMATH_CALUDE_complex_set_equals_zero_two_neg_two_l3324_332430

def complex_set : Set ℂ := {z | ∃ n : ℤ, z = Complex.I ^ n + Complex.I ^ (-n)}

theorem complex_set_equals_zero_two_neg_two : 
  complex_set = {0, 2, -2} :=
sorry

end NUMINAMATH_CALUDE_complex_set_equals_zero_two_neg_two_l3324_332430


namespace NUMINAMATH_CALUDE_book_selection_probabilities_l3324_332416

def chinese_books : ℕ := 4
def math_books : ℕ := 3
def total_books : ℕ := chinese_books + math_books
def books_to_select : ℕ := 2

def total_combinations : ℕ := Nat.choose total_books books_to_select

theorem book_selection_probabilities :
  let prob_two_math : ℚ := (Nat.choose math_books books_to_select : ℚ) / total_combinations
  let prob_one_each : ℚ := (chinese_books * math_books : ℚ) / total_combinations
  prob_two_math = 1/7 ∧ prob_one_each = 4/7 := by sorry

end NUMINAMATH_CALUDE_book_selection_probabilities_l3324_332416


namespace NUMINAMATH_CALUDE_problem_sequence_sum_largest_fib_is_196418_l3324_332491

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The sequence in the problem -/
def problem_sequence : List ℤ :=
  [2, -3, -5, 8, 13, -21, -34, 55, 89, -144, -233, 377, 46368, -75025, -121393, 196418]

/-- The sum of the problem sequence -/
def sequence_sum : ℤ := problem_sequence.sum

/-- Theorem stating that the sum of the problem sequence equals 196418 -/
theorem problem_sequence_sum : sequence_sum = 196418 := by
  sorry

/-- The largest Fibonacci number in the sequence -/
def largest_fib : ℕ := 196418

/-- Theorem stating that the largest Fibonacci number in the sequence is 196418 -/
theorem largest_fib_is_196418 : fib 27 = largest_fib := by
  sorry

end NUMINAMATH_CALUDE_problem_sequence_sum_largest_fib_is_196418_l3324_332491


namespace NUMINAMATH_CALUDE_square_in_right_triangle_l3324_332429

/-- Given a right triangle PQR with PQ = 9, PR = 12, and right angle at P,
    if a square is fitted with one side on the hypotenuse QR and other vertices
    touching the legs of the triangle, then the length of the square's side is 3. -/
theorem square_in_right_triangle (P Q R : ℝ × ℝ) (s : ℝ) :
  let pq : ℝ := 9
  let pr : ℝ := 12
  -- P is the origin (0, 0)
  P = (0, 0) →
  -- Q is on the x-axis
  Q.2 = 0 →
  -- R is on the y-axis
  R.1 = 0 →
  -- PQ = 9
  Q.1 = pq →
  -- PR = 12
  R.2 = pr →
  -- s is positive
  s > 0 →
  -- One vertex of the square is on QR
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ x + y = s ∧ x^2 + y^2 = (pq - s)^2 + (pr - s)^2 →
  -- The square's side length is 3
  s = 3 :=
by sorry

end NUMINAMATH_CALUDE_square_in_right_triangle_l3324_332429


namespace NUMINAMATH_CALUDE_vet_clinic_dog_treatment_cost_l3324_332404

theorem vet_clinic_dog_treatment_cost (cat_cost : ℕ) (num_dogs num_cats total_cost : ℕ) :
  cat_cost = 40 →
  num_dogs = 20 →
  num_cats = 60 →
  total_cost = 3600 →
  ∃ (dog_cost : ℕ), dog_cost * num_dogs + cat_cost * num_cats = total_cost ∧ dog_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_vet_clinic_dog_treatment_cost_l3324_332404


namespace NUMINAMATH_CALUDE_mork_tax_rate_l3324_332424

/-- Represents the tax rates and incomes of Mork and Mindy -/
structure TaxData where
  mork_income : ℝ
  mork_tax_rate : ℝ
  mindy_tax_rate : ℝ
  combined_tax_rate : ℝ

/-- The conditions of the problem -/
def tax_conditions (data : TaxData) : Prop :=
  data.mork_income > 0 ∧
  data.mindy_tax_rate = 0.25 ∧
  data.combined_tax_rate = 0.29

/-- The theorem stating Mork's tax rate given the conditions -/
theorem mork_tax_rate (data : TaxData) :
  tax_conditions data →
  data.mork_tax_rate = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_mork_tax_rate_l3324_332424


namespace NUMINAMATH_CALUDE_function_upper_bound_implies_parameter_range_l3324_332456

theorem function_upper_bound_implies_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) →
  (∀ x, f x = Real.sin x ^ 2 + a * Real.cos x + a) →
  a ∈ Set.Iic 0 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_implies_parameter_range_l3324_332456


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3324_332433

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- State the theorem
theorem quadratic_inequality_range :
  ∀ a : ℝ, (¬ ∃ x : ℝ, f a x ≤ 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3324_332433


namespace NUMINAMATH_CALUDE_eighth_term_is_21_l3324_332488

/-- A Fibonacci-like sequence where each number after the second is the sum of the two preceding numbers -/
def fibonacci_like_sequence (a₁ a₂ : ℕ) : ℕ → ℕ
| 0 => a₁
| 1 => a₂
| (n + 2) => fibonacci_like_sequence a₁ a₂ n + fibonacci_like_sequence a₁ a₂ (n + 1)

/-- The theorem stating that the 8th term of the specific Fibonacci-like sequence is 21 -/
theorem eighth_term_is_21 :
  ∃ (seq : ℕ → ℕ), 
    seq = fibonacci_like_sequence 1 1 ∧
    seq 7 = 21 ∧
    seq 8 = 34 ∧
    seq 9 = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_21_l3324_332488


namespace NUMINAMATH_CALUDE_max_angles_theorem_l3324_332443

theorem max_angles_theorem (k : ℕ) :
  let n := 2 * k
  ∃ (max_angles : ℕ), max_angles = 2 * k - 1 ∧
    ∀ (num_angles : ℕ), num_angles ≤ max_angles :=
by sorry

end NUMINAMATH_CALUDE_max_angles_theorem_l3324_332443


namespace NUMINAMATH_CALUDE_line_segments_and_midpoints_l3324_332415

/-- The number of line segments that can be formed with n points on a line -/
def num_segments (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of unique midpoints of line segments formed by n points on a line -/
def num_midpoints (n : ℕ) : ℕ := 2 * n - 3

/-- The number of points on the line -/
def num_points : ℕ := 10

theorem line_segments_and_midpoints :
  num_segments num_points = 45 ∧ num_midpoints num_points = 17 := by
  sorry

end NUMINAMATH_CALUDE_line_segments_and_midpoints_l3324_332415


namespace NUMINAMATH_CALUDE_divisor_of_q_l3324_332476

theorem divisor_of_q (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 30)
  (h2 : Nat.gcd q.val r.val = 42)
  (h3 : Nat.gcd r.val s.val = 66)
  (h4 : 80 < Nat.gcd s.val p.val ∧ Nat.gcd s.val p.val < 120) :
  5 ∣ q.val :=
by sorry

end NUMINAMATH_CALUDE_divisor_of_q_l3324_332476


namespace NUMINAMATH_CALUDE_greatest_BAABC_div_11_l3324_332490

def is_valid_BAABC (n : ℕ) : Prop :=
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    n = 10000 * B + 1000 * A + 100 * A + 10 * B + C

theorem greatest_BAABC_div_11 :
  ∀ n : ℕ,
    is_valid_BAABC n →
    n ≤ 96619 ∧
    is_valid_BAABC 96619 ∧
    96619 % 11 = 0 ∧
    (n % 11 = 0 → n ≤ 96619) :=
by sorry

end NUMINAMATH_CALUDE_greatest_BAABC_div_11_l3324_332490


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3324_332450

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x + y) * (f x - f y)

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3324_332450


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3324_332413

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (p → q) ∧ ¬(q → p) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3324_332413


namespace NUMINAMATH_CALUDE_train_length_calculation_l3324_332438

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed = 36 * (1000 / 3600) →
  time_to_cross = 82.49340052795776 →
  bridge_length = 660 →
  train_speed * time_to_cross - bridge_length = 164.9340052795776 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3324_332438


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_two_thirds_l3324_332469

theorem mean_of_five_numbers_with_sum_two_thirds :
  ∀ (a b c d e : ℚ),
  a + b + c + d + e = 2/3 →
  (a + b + c + d + e) / 5 = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_two_thirds_l3324_332469


namespace NUMINAMATH_CALUDE_flower_position_l3324_332405

/-- Represents the number of students in the circle -/
def n : ℕ := 7

/-- Represents the number of times the drum is beaten -/
def k : ℕ := 50

/-- Function to calculate the final position after k rotations in a circle of n elements -/
def finalPosition (n k : ℕ) : ℕ := 
  (k % n) + 1

theorem flower_position : 
  finalPosition n k = 2 := by sorry

end NUMINAMATH_CALUDE_flower_position_l3324_332405


namespace NUMINAMATH_CALUDE_sugar_measurement_l3324_332400

theorem sugar_measurement (required_sugar : Rat) (cup_capacity : Rat) (fills : Nat) : 
  required_sugar = 15/4 ∧ cup_capacity = 1/3 → fills = 12 := by
  sorry

end NUMINAMATH_CALUDE_sugar_measurement_l3324_332400


namespace NUMINAMATH_CALUDE_friend_symmetry_iff_d_mod_7_eq_2_l3324_332481

def isFriend (d : ℕ) (M N : ℕ) : Prop :=
  ∀ k, k < d → (M + 10^k * ((N / 10^k) % 10 - (M / 10^k) % 10)) % 7 = 0

theorem friend_symmetry_iff_d_mod_7_eq_2 (d : ℕ) :
  (∀ M N : ℕ, M < 10^d → N < 10^d → (isFriend d M N ↔ isFriend d N M)) ↔ d % 7 = 2 :=
sorry

end NUMINAMATH_CALUDE_friend_symmetry_iff_d_mod_7_eq_2_l3324_332481


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l3324_332435

/-- Calculates the number of male athletes in a stratified sample -/
def maleAthletesInSample (totalMale : ℕ) (totalFemale : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalMale * sampleSize) / (totalMale + totalFemale)

/-- Theorem: In a stratified sample of 14 athletes drawn from a population of 32 male and 24 female athletes, the number of male athletes in the sample is 8 -/
theorem stratified_sample_male_count :
  maleAthletesInSample 32 24 14 = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_male_count_l3324_332435


namespace NUMINAMATH_CALUDE_speed_difference_l3324_332495

/-- Proves that the difference in average speed between two people traveling the same distance,
    where one travels at 12 miles per hour and the other completes the journey in 10 minutes,
    is 24 miles per hour. -/
theorem speed_difference (distance : ℝ) (speed_maya : ℝ) (time_naomi : ℝ) : 
  distance > 0 ∧ speed_maya = 12 ∧ time_naomi = 1/6 →
  (distance / time_naomi) - speed_maya = 24 :=
by sorry

end NUMINAMATH_CALUDE_speed_difference_l3324_332495


namespace NUMINAMATH_CALUDE_intersection_forms_right_triangle_l3324_332480

/-- An ellipse with equation x²/m + y² = 1, where m > 1 -/
structure Ellipse where
  m : ℝ
  h_m : m > 1

/-- A hyperbola with equation x²/n - y² = 1, where n > 0 -/
structure Hyperbola where
  n : ℝ
  h_n : n > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents two curves (ellipse and hyperbola) with shared foci -/
structure SharedFociCurves where
  e : Ellipse
  h : Hyperbola
  f₁ : Point  -- First focus
  f₂ : Point  -- Second focus

/-- A point P that lies on both the ellipse and the hyperbola -/
structure IntersectionPoint (curves : SharedFociCurves) where
  p : Point
  on_ellipse : p.x ^ 2 / curves.e.m + p.y ^ 2 = 1
  on_hyperbola : p.x ^ 2 / curves.h.n - p.y ^ 2 = 1

/-- The main theorem: Triangle F₁PF₂ is always a right triangle -/
theorem intersection_forms_right_triangle (curves : SharedFociCurves) 
  (p : IntersectionPoint curves) : 
  (p.p.x - curves.f₁.x) ^ 2 + (p.p.y - curves.f₁.y) ^ 2 +
  (p.p.x - curves.f₂.x) ^ 2 + (p.p.y - curves.f₂.y) ^ 2 =
  (curves.f₁.x - curves.f₂.x) ^ 2 + (curves.f₁.y - curves.f₂.y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_forms_right_triangle_l3324_332480


namespace NUMINAMATH_CALUDE_petyas_actual_time_greater_than_planned_l3324_332471

/-- Proves that Petya's actual time is greater than the planned time -/
theorem petyas_actual_time_greater_than_planned
  (D V : ℝ)
  (h_D : D > 0)
  (h_V : V > 0) :
  (D / (2 * 1.25 * V) + D / (2 * 0.8 * V)) > D / V := by
  sorry

end NUMINAMATH_CALUDE_petyas_actual_time_greater_than_planned_l3324_332471


namespace NUMINAMATH_CALUDE_function_composition_l3324_332411

theorem function_composition (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 - 3*x) → (∀ x, f x = x^2 - x - 2) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l3324_332411


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l3324_332493

/-- Probability of drawing two black balls without replacement -/
theorem probability_two_black_balls 
  (white : ℕ) 
  (black : ℕ) 
  (h1 : white = 7) 
  (h2 : black = 8) : 
  (black * (black - 1)) / ((white + black) * (white + black - 1)) = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_l3324_332493


namespace NUMINAMATH_CALUDE_problem_statement_l3324_332440

theorem problem_statement (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc1 : 0 ≤ c) (hc2 : c < -b) :
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3324_332440


namespace NUMINAMATH_CALUDE_plan1_more_profitable_l3324_332442

/-- Represents the monthly production and profit of a factory with two wastewater treatment plans -/
structure FactoryProduction where
  x : ℕ  -- Number of products produced per month
  y1 : ℤ -- Monthly profit for Plan 1 in yuan
  y2 : ℤ -- Monthly profit for Plan 2 in yuan

/-- Calculates the monthly profit for Plan 1 -/
def plan1Profit (x : ℕ) : ℤ :=
  24 * x - 30000

/-- Calculates the monthly profit for Plan 2 -/
def plan2Profit (x : ℕ) : ℤ :=
  18 * x

/-- Theorem stating that Plan 1 yields more profit when producing 6000 products per month -/
theorem plan1_more_profitable :
  let production : FactoryProduction := {
    x := 6000,
    y1 := plan1Profit 6000,
    y2 := plan2Profit 6000
  }
  production.y1 > production.y2 :=
by sorry

end NUMINAMATH_CALUDE_plan1_more_profitable_l3324_332442


namespace NUMINAMATH_CALUDE_smallest_base_for_square_property_l3324_332420

theorem smallest_base_for_square_property : ∃ (b x y : ℕ), 
  (b ≥ 2) ∧ 
  (x < b) ∧ 
  (y < b) ∧ 
  (x ≠ 0) ∧ 
  (y ≠ 0) ∧ 
  ((x * b + x)^2 = y * b^3 + y * b^2 + y * b + y) ∧
  (∀ b' x' y' : ℕ, 
    (b' ≥ 2) ∧ 
    (x' < b') ∧ 
    (y' < b') ∧ 
    (x' ≠ 0) ∧ 
    (y' ≠ 0) ∧ 
    ((x' * b' + x')^2 = y' * b'^3 + y' * b'^2 + y' * b' + y') →
    (b ≤ b')) ∧
  (b = 7) ∧ 
  (x = 5) ∧ 
  (y = 4) := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_square_property_l3324_332420


namespace NUMINAMATH_CALUDE_product_simplification_l3324_332477

theorem product_simplification (x : ℝ) (h : x ≠ 0) :
  (12 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (3/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3324_332477


namespace NUMINAMATH_CALUDE_water_depth_in_cistern_l3324_332407

/-- Calculates the depth of water in a rectangular cistern given its dimensions and wet surface area. -/
theorem water_depth_in_cistern
  (length : ℝ)
  (width : ℝ)
  (total_wet_surface_area : ℝ)
  (h1 : length = 8)
  (h2 : width = 6)
  (h3 : total_wet_surface_area = 83) :
  ∃ (depth : ℝ), depth = 1.25 ∧ 
    total_wet_surface_area = length * width + 2 * (length + width) * depth :=
by sorry


end NUMINAMATH_CALUDE_water_depth_in_cistern_l3324_332407
