import Mathlib

namespace NUMINAMATH_CALUDE_root_of_equations_l1500_150059

theorem root_of_equations (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + p = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + p * m + q = 0) :
  m^5 = q / p ∧ (p = q → ∃ k : Fin 5, m = Complex.exp (2 * Real.pi * I * (k : ℝ) / 5)) :=
sorry

end NUMINAMATH_CALUDE_root_of_equations_l1500_150059


namespace NUMINAMATH_CALUDE_evaluate_expression_l1500_150085

theorem evaluate_expression : (-3)^7 / 3^5 + 2^6 - 4^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1500_150085


namespace NUMINAMATH_CALUDE_square_side_length_l1500_150068

/-- The side length of a square with area equal to a 3 cm × 27 cm rectangle is 9 cm. -/
theorem square_side_length (square_area rectangle_area : ℝ) (square_side : ℝ) : 
  square_area = rectangle_area →
  rectangle_area = 3 * 27 →
  square_area = square_side ^ 2 →
  square_side = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1500_150068


namespace NUMINAMATH_CALUDE_area_of_region_l1500_150058

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 17 ∧ 
   A = Real.pi * (Real.sqrt ((x + 1)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 2*x - 4*y = 12) :=
by sorry

end NUMINAMATH_CALUDE_area_of_region_l1500_150058


namespace NUMINAMATH_CALUDE_negation_of_all_x_squared_nonnegative_l1500_150048

theorem negation_of_all_x_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_x_squared_nonnegative_l1500_150048


namespace NUMINAMATH_CALUDE_smallest_even_divisible_by_20_and_60_l1500_150087

theorem smallest_even_divisible_by_20_and_60 : ∃ n : ℕ, n > 0 ∧ Even n ∧ 20 ∣ n ∧ 60 ∣ n ∧ ∀ m : ℕ, m > 0 → Even m → 20 ∣ m → 60 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_divisible_by_20_and_60_l1500_150087


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1500_150034

theorem floor_equation_solutions : 
  (∃ (s : Finset ℕ), s.card = 110 ∧ 
    (∀ x : ℕ, x ∈ s ↔ ⌊(x : ℚ) / 10⌋ = ⌊(x : ℚ) / 11⌋ + 1)) := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1500_150034


namespace NUMINAMATH_CALUDE_square_and_sqrt_preserve_geometric_sequence_l1500_150039

-- Define the domain (−∞,0)∪(0,+∞)
def NonZeroReals : Set ℝ := {x : ℝ | x ≠ 0}

-- Define a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the "preserving geometric sequence" property
def PreservingGeometricSequence (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence (fun n ↦ f (a n))

-- State the theorem
theorem square_and_sqrt_preserve_geometric_sequence :
  (PreservingGeometricSequence (fun x ↦ x^2)) ∧
  (PreservingGeometricSequence (fun x ↦ Real.sqrt (abs x))) :=
sorry

end NUMINAMATH_CALUDE_square_and_sqrt_preserve_geometric_sequence_l1500_150039


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l1500_150078

theorem min_value_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : a + 2*b = 6) : 
  (∀ x y, x > 0 → y > 0 → x + 2*y = 6 → (1 / (a * b)) ≤ (1 / (x * y))) → 
  (1 / (a * b)) = 2/9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l1500_150078


namespace NUMINAMATH_CALUDE_rectangle_area_l1500_150062

theorem rectangle_area (x : ℝ) : 
  x > 0 → 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ 
  l = 3 * w ∧ 
  x^2 = l^2 + w^2 ∧
  w * l = (3/10) * x^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1500_150062


namespace NUMINAMATH_CALUDE_ice_cream_cup_cost_l1500_150086

/-- Calculates the cost of each ice-cream cup given the order details and total amount paid -/
theorem ice_cream_cup_cost
  (chapati_count : ℕ)
  (chapati_cost : ℕ)
  (rice_count : ℕ)
  (rice_cost : ℕ)
  (vegetable_count : ℕ)
  (vegetable_cost : ℕ)
  (ice_cream_count : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : chapati_cost = 6)
  (h3 : rice_count = 5)
  (h4 : rice_cost = 45)
  (h5 : vegetable_count = 7)
  (h6 : vegetable_cost = 70)
  (h7 : ice_cream_count = 6)
  (h8 : total_paid = 883)
  : (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 12 := by
  sorry

#check ice_cream_cup_cost

end NUMINAMATH_CALUDE_ice_cream_cup_cost_l1500_150086


namespace NUMINAMATH_CALUDE_magician_balls_l1500_150082

/-- Represents the number of balls in the box after each operation -/
def BallCount : ℕ → ℕ
  | 0 => 7  -- Initial count
  | n + 1 => BallCount n + 6 * (BallCount n - 1)  -- After each operation

/-- The form that the ball count must follow -/
def ValidForm (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k + 7

theorem magician_balls :
  (∀ n : ℕ, ValidForm (BallCount n)) ∧
  ValidForm 1993 ∧
  ¬ValidForm 1990 ∧
  ¬ValidForm 1991 ∧
  ¬ValidForm 1992 := by sorry

end NUMINAMATH_CALUDE_magician_balls_l1500_150082


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_range_l1500_150088

theorem sufficient_condition_implies_range (a : ℝ) : 
  (∀ x, (x - 1) * (x - 2) < 0 → x - a < 0) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_range_l1500_150088


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1500_150045

theorem quadratic_roots_theorem (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + 2*k*x + k^2 = x + 1 ↔ x = x₁ ∨ x = x₂) ∧
    (3*x₁ - x₂)*(x₁ - 3*x₂) = 19) →
  k = 0 ∨ k = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1500_150045


namespace NUMINAMATH_CALUDE_fraction_simplification_l1500_150027

theorem fraction_simplification (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 5) :
  (15 * x^2 * y^3) / (9 * x * y^2 * z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1500_150027


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1500_150072

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_proposition :
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1500_150072


namespace NUMINAMATH_CALUDE_oil_change_price_is_20_l1500_150077

def oil_change_price (repair_price car_wash_price : ℕ) 
                     (oil_changes repairs car_washes : ℕ) 
                     (total_earnings : ℕ) : Prop :=
  ∃ (x : ℕ), 
    repair_price = 30 ∧
    car_wash_price = 5 ∧
    oil_changes = 5 ∧
    repairs = 10 ∧
    car_washes = 15 ∧
    total_earnings = 475 ∧
    x * oil_changes + repair_price * repairs + car_wash_price * car_washes = total_earnings ∧
    x = 20

theorem oil_change_price_is_20 : 
  ∀ (repair_price car_wash_price oil_changes repairs car_washes total_earnings : ℕ),
    oil_change_price repair_price car_wash_price oil_changes repairs car_washes total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_oil_change_price_is_20_l1500_150077


namespace NUMINAMATH_CALUDE_sunflower_seed_tins_l1500_150050

theorem sunflower_seed_tins (candy_bags : ℕ) (candies_per_bag : ℕ) (seeds_per_tin : ℕ) (total_items : ℕ) : 
  candy_bags = 19 →
  candies_per_bag = 46 →
  seeds_per_tin = 170 →
  total_items = 1894 →
  (total_items - candy_bags * candies_per_bag) / seeds_per_tin = 6 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_seed_tins_l1500_150050


namespace NUMINAMATH_CALUDE_product_of_sines_l1500_150057

theorem product_of_sines (π : Real) : 
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) * 
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 
  (17 / 16 + 2 * Real.sin (π / 12)) * (17 / 16 + 2 * Real.sin (5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_l1500_150057


namespace NUMINAMATH_CALUDE_school_teachers_calculation_l1500_150053

/-- Calculates the number of teachers required in a school given specific conditions -/
theorem school_teachers_calculation (total_students : ℕ) (lessons_per_student : ℕ) 
  (lessons_per_teacher : ℕ) (students_per_class : ℕ) : 
  total_students = 1200 →
  lessons_per_student = 5 →
  lessons_per_teacher = 4 →
  students_per_class = 30 →
  (total_students * lessons_per_student) / (students_per_class * lessons_per_teacher) = 50 := by
  sorry

#check school_teachers_calculation

end NUMINAMATH_CALUDE_school_teachers_calculation_l1500_150053


namespace NUMINAMATH_CALUDE_square_difference_cubed_l1500_150049

theorem square_difference_cubed : (7^2 - 5^2)^3 = 13824 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_cubed_l1500_150049


namespace NUMINAMATH_CALUDE_m_range_for_p_necessary_not_sufficient_for_q_l1500_150074

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

-- Define the sets A and B
def A : Set ℝ := {x | p x}
def B (m : ℝ) : Set ℝ := {x | q x m}

-- Theorem statement
theorem m_range_for_p_necessary_not_sufficient_for_q :
  ∀ m : ℝ, (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x m)) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_for_p_necessary_not_sufficient_for_q_l1500_150074


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l1500_150041

/-- Calculates the share of profit for an investor given the total profit and investment ratios. -/
def calculate_share_of_profit (total_profit : ℚ) (investor_ratio : ℚ) (total_ratio : ℚ) : ℚ :=
  (investor_ratio / total_ratio) * total_profit

/-- Represents the problem of calculating Jose's share of profit in a business partnership. -/
theorem jose_share_of_profit 
  (tom_investment : ℚ) (tom_duration : ℕ) 
  (jose_investment : ℚ) (jose_duration : ℕ) 
  (total_profit : ℚ) : 
  tom_investment = 30000 → 
  tom_duration = 12 → 
  jose_investment = 45000 → 
  jose_duration = 10 → 
  total_profit = 54000 → 
  calculate_share_of_profit total_profit (jose_investment * jose_duration) 
    (tom_investment * tom_duration + jose_investment * jose_duration) = 30000 := by
  sorry

end NUMINAMATH_CALUDE_jose_share_of_profit_l1500_150041


namespace NUMINAMATH_CALUDE_raccoon_lock_problem_l1500_150019

theorem raccoon_lock_problem (first_lock_time second_lock_time both_locks_time : ℕ) :
  second_lock_time = 3 * first_lock_time - 3 →
  both_locks_time = 5 * second_lock_time →
  second_lock_time = 60 →
  first_lock_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_lock_problem_l1500_150019


namespace NUMINAMATH_CALUDE_quadratic_constraint_l1500_150069

/-- Quadratic function defined by a parameter a -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

/-- Theorem stating the condition on a for the given constraints -/
theorem quadratic_constraint (a : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : a ≠ 0)
  (h₂ : x₁ + x₂ = 2)
  (h₃ : x₁ < x₂)
  (h₄ : y₁ > y₂)
  (h₅ : quadratic_function a x₁ = y₁)
  (h₆ : quadratic_function a x₂ = y₂) :
  a < -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_constraint_l1500_150069


namespace NUMINAMATH_CALUDE_dans_marbles_l1500_150012

/-- 
Given that Dan gave some marbles to Mary and has some marbles left,
prove that the initial number of marbles is equal to the sum of
the marbles given and the marbles left.
-/
theorem dans_marbles (marbles_given : ℕ) (marbles_left : ℕ) :
  marbles_given + marbles_left = 64 → marbles_given = 14 → marbles_left = 50 := by
  sorry

#check dans_marbles

end NUMINAMATH_CALUDE_dans_marbles_l1500_150012


namespace NUMINAMATH_CALUDE_animal_lifespans_l1500_150056

theorem animal_lifespans (bat hamster frog tortoise : ℝ) : 
  hamster = bat - 6 →
  frog = 4 * hamster →
  tortoise = 2 * bat →
  bat + hamster + frog + tortoise = 62 →
  bat = 11.5 := by
sorry

end NUMINAMATH_CALUDE_animal_lifespans_l1500_150056


namespace NUMINAMATH_CALUDE_M_values_l1500_150046

theorem M_values (a b : ℚ) (h : a * b ≠ 0) :
  let M := |a| / a + b / |b|
  M = 0 ∨ M = 2 ∨ M = -2 := by
sorry

end NUMINAMATH_CALUDE_M_values_l1500_150046


namespace NUMINAMATH_CALUDE_candy_cost_proof_l1500_150017

/-- Represents the cost per pound of the second type of candy -/
def second_candy_cost : ℝ := 5

/-- Represents the weight of the first type of candy in pounds -/
def first_candy_weight : ℝ := 10

/-- Represents the cost per pound of the first type of candy -/
def first_candy_cost : ℝ := 8

/-- Represents the weight of the second type of candy in pounds -/
def second_candy_weight : ℝ := 20

/-- Represents the cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- Represents the total weight of the mixture in pounds -/
def total_weight : ℝ := first_candy_weight + second_candy_weight

theorem candy_cost_proof :
  first_candy_weight * first_candy_cost + second_candy_weight * second_candy_cost =
  total_weight * mixture_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l1500_150017


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1500_150096

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1500_150096


namespace NUMINAMATH_CALUDE_sin_2alpha_equals_one_minus_p_squared_l1500_150092

theorem sin_2alpha_equals_one_minus_p_squared (α : ℝ) (p : ℝ) 
  (h : Real.sin α - Real.cos α = p) : 
  Real.sin (2 * α) = 1 - p^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_equals_one_minus_p_squared_l1500_150092


namespace NUMINAMATH_CALUDE_fractional_parts_sum_not_one_l1500_150043

theorem fractional_parts_sum_not_one (x : ℚ) : 
  ¬(x - ⌊x⌋ + x^2 - ⌊x^2⌋ = 1) := by sorry

end NUMINAMATH_CALUDE_fractional_parts_sum_not_one_l1500_150043


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_l1500_150009

theorem sum_of_w_and_y (W X Y Z : ℤ) : 
  W ∈ ({1, 2, 5, 6} : Set ℤ) →
  X ∈ ({1, 2, 5, 6} : Set ℤ) →
  Y ∈ ({1, 2, 5, 6} : Set ℤ) →
  Z ∈ ({1, 2, 5, 6} : Set ℤ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / X + (Y : ℚ) / Z = 3 →
  W + Y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_l1500_150009


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1500_150036

/-- The probability of drawing a red ball from a bag with red and white balls -/
theorem probability_of_red_ball (red_balls white_balls : ℕ) :
  red_balls = 7 → white_balls = 3 →
  (red_balls : ℚ) / (red_balls + white_balls : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1500_150036


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1500_150071

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 4 = 12) : 
  a 1 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1500_150071


namespace NUMINAMATH_CALUDE_f_decreasing_range_l1500_150047

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a/x

/-- Theorem stating the range of a for f(x) to be decreasing -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  1/6 ≤ a ∧ a < 1/3 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_range_l1500_150047


namespace NUMINAMATH_CALUDE_square_root_problem_l1500_150067

theorem square_root_problem (a : ℝ) (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt x = 2*a - 1) (h3 : Real.sqrt x = -a + 3) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1500_150067


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1500_150013

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1500_150013


namespace NUMINAMATH_CALUDE_clever_cat_academy_count_l1500_150054

theorem clever_cat_academy_count :
  let jump : ℕ := 60
  let spin : ℕ := 35
  let fetch : ℕ := 40
  let jump_and_spin : ℕ := 25
  let spin_and_fetch : ℕ := 20
  let jump_and_fetch : ℕ := 22
  let all_three : ℕ := 12
  let none : ℕ := 10
  jump + spin + fetch - jump_and_spin - spin_and_fetch - jump_and_fetch + all_three + none = 92 :=
by sorry

end NUMINAMATH_CALUDE_clever_cat_academy_count_l1500_150054


namespace NUMINAMATH_CALUDE_angle_T_measure_l1500_150008

-- Define a pentagon
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

-- Define the properties of the pentagon
def is_valid_pentagon (p : Pentagon) : Prop :=
  p.P + p.Q + p.R + p.S + p.T = 540

def angles_congruent (p : Pentagon) : Prop :=
  p.P = p.R ∧ p.R = p.T

def angles_supplementary (p : Pentagon) : Prop :=
  p.Q + p.S = 180

-- Theorem statement
theorem angle_T_measure (p : Pentagon) 
  (h1 : is_valid_pentagon p) 
  (h2 : angles_congruent p) 
  (h3 : angles_supplementary p) : 
  p.T = 120 := by sorry

end NUMINAMATH_CALUDE_angle_T_measure_l1500_150008


namespace NUMINAMATH_CALUDE_subtraction_for_complex_equality_l1500_150040

theorem subtraction_for_complex_equality : ∃ (z : ℂ), (7 - 3*I) - z = 3 * ((2 + I) + (4 - 2*I)) ∧ z = -11 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_for_complex_equality_l1500_150040


namespace NUMINAMATH_CALUDE_sam_tutoring_hours_l1500_150032

/-- Sam's hourly rate for Math tutoring -/
def hourly_rate : ℕ := 10

/-- Sam's earnings for the first month -/
def first_month_earnings : ℕ := 200

/-- The additional amount Sam earned in the second month compared to the first -/
def second_month_increase : ℕ := 150

/-- The total number of hours Sam spent tutoring for two months -/
def total_hours : ℕ := 55

/-- Theorem stating that given the conditions, Sam spent 55 hours tutoring over two months -/
theorem sam_tutoring_hours :
  hourly_rate * total_hours = first_month_earnings + (first_month_earnings + second_month_increase) :=
by sorry

end NUMINAMATH_CALUDE_sam_tutoring_hours_l1500_150032


namespace NUMINAMATH_CALUDE_window_area_ratio_l1500_150081

/-- Represents the window design with a rectangle and semicircles at each end -/
structure WindowDesign where
  /-- Total length of the window, including semicircles -/
  ad : ℝ
  /-- Diameter of the semicircles (width of the window) -/
  ab : ℝ
  /-- Ratio of total length to semicircle diameter is 4:3 -/
  ratio_condition : ad / ab = 4 / 3
  /-- The width of the window is 40 inches -/
  width_condition : ab = 40

/-- The ratio of the rectangle area to the semicircles area is 8/(3π) -/
theorem window_area_ratio (w : WindowDesign) :
  let r := w.ab / 2  -- radius of semicircles
  let rect_length := w.ad - w.ab  -- length of rectangle
  let rect_area := rect_length * w.ab  -- area of rectangle
  let semicircles_area := π * r^2  -- area of semicircles (full circle)
  rect_area / semicircles_area = 8 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_window_area_ratio_l1500_150081


namespace NUMINAMATH_CALUDE_unique_right_triangle_perimeter_area_ratio_l1500_150044

theorem unique_right_triangle_perimeter_area_ratio :
  ∃! (a b : ℝ), a > 0 ∧ b > 0 ∧
  (a + b + Real.sqrt (a^2 + b^2)) / ((1/2) * a * b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_right_triangle_perimeter_area_ratio_l1500_150044


namespace NUMINAMATH_CALUDE_bird_feeder_theft_ratio_l1500_150095

/-- Given a bird feeder with the following properties:
  - Holds 2 cups of birdseed
  - Each cup of birdseed can feed 14 birds
  - The feeder actually feeds 21 birds weekly
  Prove that the ratio of birdseed stolen to total birdseed is 1:4 -/
theorem bird_feeder_theft_ratio 
  (total_cups : ℚ) 
  (birds_per_cup : ℕ) 
  (birds_fed : ℕ) : 
  total_cups = 2 →
  birds_per_cup = 14 →
  birds_fed = 21 →
  (total_cups - (birds_fed : ℚ) / (birds_per_cup : ℚ)) / total_cups = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bird_feeder_theft_ratio_l1500_150095


namespace NUMINAMATH_CALUDE_v_shaped_to_log_v_shaped_l1500_150010

/-- Definition of a V-shaped function -/
def is_v_shaped (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) ≤ f x₁ + f x₂

/-- Definition of a Logarithmic V-shaped function -/
def is_log_v_shaped (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x > 0) ∧
  (∀ x₁ x₂ : ℝ, Real.log (f (x₁ + x₂)) < Real.log (f x₁) + Real.log (f x₂))

/-- Theorem: If f is V-shaped and f(x) ≥ 2 for all x, then f is Logarithmic V-shaped -/
theorem v_shaped_to_log_v_shaped (f : ℝ → ℝ) 
    (hv : is_v_shaped f) (hf : ∀ x : ℝ, f x ≥ 2) : 
    is_log_v_shaped f := by
  sorry

end NUMINAMATH_CALUDE_v_shaped_to_log_v_shaped_l1500_150010


namespace NUMINAMATH_CALUDE_tin_can_equation_l1500_150084

/-- Represents the number of can bodies that can be made from one sheet of tinplate -/
def bodies_per_sheet : ℕ := 15

/-- Represents the number of can bottoms that can be made from one sheet of tinplate -/
def bottoms_per_sheet : ℕ := 42

/-- Represents the total number of available sheets of tinplate -/
def total_sheets : ℕ := 108

/-- Represents the number of can bottoms needed for one complete tin can -/
def bottoms_per_can : ℕ := 2

theorem tin_can_equation (x : ℕ) :
  x ≤ total_sheets →
  (bottoms_per_can * bodies_per_sheet * x = bottoms_per_sheet * (total_sheets - x)) ↔
  (2 * 15 * x = 42 * (108 - x)) :=
by sorry

end NUMINAMATH_CALUDE_tin_can_equation_l1500_150084


namespace NUMINAMATH_CALUDE_consecutive_products_not_3000000_l1500_150030

theorem consecutive_products_not_3000000 :
  ∀ n : ℕ, (n - 1) * n + n * (n + 1) + (n - 1) * (n + 1) ≠ 3000000 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_products_not_3000000_l1500_150030


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1500_150075

def symmetric_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem symmetry_of_point :
  symmetric_point_x_axis (1, 2) = (1, -2) := by
sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1500_150075


namespace NUMINAMATH_CALUDE_park_nests_l1500_150028

/-- Calculates the minimum number of nests required for birds in a park -/
def minimum_nests (sparrows pigeons starlings robins : ℕ) 
  (sparrow_nests pigeon_nests starling_nests robin_nests : ℕ) : ℕ :=
  sparrows * sparrow_nests + pigeons * pigeon_nests + 
  starlings * starling_nests + robins * robin_nests

/-- Theorem stating the minimum number of nests required for the given bird populations -/
theorem park_nests : 
  minimum_nests 5 3 6 2 1 2 3 4 = 37 := by
  sorry

#eval minimum_nests 5 3 6 2 1 2 3 4

end NUMINAMATH_CALUDE_park_nests_l1500_150028


namespace NUMINAMATH_CALUDE_longer_diagonal_squared_is_80_l1500_150055

/-- Represents a parallelogram LMNO with specific properties -/
structure Parallelogram where
  area : ℝ
  xy : ℝ
  zw : ℝ
  (area_positive : area > 0)
  (xy_positive : xy > 0)
  (zw_positive : zw > 0)

/-- The square of the longer diagonal of the parallelogram -/
def longer_diagonal_squared (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the square of the longer diagonal equals 80 -/
theorem longer_diagonal_squared_is_80 (p : Parallelogram) 
  (h1 : p.area = 24) 
  (h2 : p.xy = 8) 
  (h3 : p.zw = 10) : 
  longer_diagonal_squared p = 80 := by sorry

end NUMINAMATH_CALUDE_longer_diagonal_squared_is_80_l1500_150055


namespace NUMINAMATH_CALUDE_negative_among_expressions_l1500_150021

theorem negative_among_expressions : 
  (|(-3)| > 0) ∧ (-(-3) > 0) ∧ ((-3)^2 > 0) ∧ (-Real.sqrt 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_among_expressions_l1500_150021


namespace NUMINAMATH_CALUDE_unique_number_251_l1500_150037

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 251 is the unique positive integer whose product with its sum of digits is 2008 -/
theorem unique_number_251 : ∃! (n : ℕ), n > 0 ∧ n * sum_of_digits n = 2008 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_251_l1500_150037


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1500_150060

theorem z_in_first_quadrant (z : ℂ) : 
  ((1 + 2*Complex.I) / (z - 3) = -Complex.I) → 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1500_150060


namespace NUMINAMATH_CALUDE_player_B_most_consistent_l1500_150004

/-- Represents a player in the rope skipping test -/
inductive Player : Type
  | A : Player
  | B : Player
  | C : Player
  | D : Player

/-- Returns the variance of a player's performance -/
def variance (p : Player) : ℝ :=
  match p with
  | Player.A => 0.023
  | Player.B => 0.018
  | Player.C => 0.020
  | Player.D => 0.021

/-- States that Player B has the most consistent performance -/
theorem player_B_most_consistent :
  ∀ p : Player, p ≠ Player.B → variance Player.B < variance p :=
by sorry

end NUMINAMATH_CALUDE_player_B_most_consistent_l1500_150004


namespace NUMINAMATH_CALUDE_mod_37_5_l1500_150016

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_37_5_l1500_150016


namespace NUMINAMATH_CALUDE_jonessa_take_home_pay_l1500_150091

/-- Given Jonessa's pay and tax rate, calculate her take-home pay -/
theorem jonessa_take_home_pay (total_pay : ℝ) (tax_rate : ℝ) 
  (h1 : total_pay = 500)
  (h2 : tax_rate = 0.1) : 
  total_pay * (1 - tax_rate) = 450 := by
sorry

end NUMINAMATH_CALUDE_jonessa_take_home_pay_l1500_150091


namespace NUMINAMATH_CALUDE_cookie_distribution_l1500_150064

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1500_150064


namespace NUMINAMATH_CALUDE_reflection_property_l1500_150025

/-- A reflection in ℝ² -/
structure Reflection where
  /-- The function that performs the reflection -/
  apply : ℝ × ℝ → ℝ × ℝ

/-- Theorem stating that a reflection mapping (3, 2) to (1, 6) will map (2, -1) to (-2/5, -11/5) -/
theorem reflection_property (r : Reflection) 
  (h : r.apply (3, 2) = (1, 6)) :
  r.apply (2, -1) = (-2/5, -11/5) := by
  sorry

end NUMINAMATH_CALUDE_reflection_property_l1500_150025


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l1500_150080

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l1500_150080


namespace NUMINAMATH_CALUDE_probability_of_123456_l1500_150093

def num_cards : ℕ := 12
def num_distinct : ℕ := 6

def total_arrangements : ℕ := (Finset.prod (Finset.range num_distinct) (fun i => Nat.choose (num_cards - 2*i) 2))

def favorable_arrangements : ℕ := (Finset.prod (Finset.range num_distinct) (fun i => 2*i + 1))

theorem probability_of_123456 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 720 :=
sorry

end NUMINAMATH_CALUDE_probability_of_123456_l1500_150093


namespace NUMINAMATH_CALUDE_complex_number_existence_l1500_150011

theorem complex_number_existence : ∃ z : ℂ, 
  (z + 10 / z).im = 0 ∧ (z + 4).re = (z + 4).im :=
sorry

end NUMINAMATH_CALUDE_complex_number_existence_l1500_150011


namespace NUMINAMATH_CALUDE_complex_square_sum_l1500_150051

theorem complex_square_sum (a b : ℝ) (h : (1 + Complex.I)^2 = Complex.mk a b) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l1500_150051


namespace NUMINAMATH_CALUDE_altitude_division_ratio_l1500_150020

/-- Given a triangle with side lengths √3, 2, and √5, the altitude perpendicular 
    to the side of length 2 divides that side in the ratio 1:3 -/
theorem altitude_division_ratio (a b c : ℝ) (h₁ : a = Real.sqrt 3) 
    (h₂ : b = 2) (h₃ : c = Real.sqrt 5) :
    let m := Real.sqrt (3 - (1/2)^2)
    (1/2) / (3/2) = 1/3 := by sorry

end NUMINAMATH_CALUDE_altitude_division_ratio_l1500_150020


namespace NUMINAMATH_CALUDE_schedule_theorem_l1500_150029

/-- The number of periods in a day -/
def num_periods : ℕ := 7

/-- The number of subjects to be scheduled -/
def num_subjects : ℕ := 4

/-- Calculates the number of ways to schedule subjects -/
def schedule_ways : ℕ := Nat.choose num_periods num_subjects * Nat.factorial num_subjects

/-- Theorem stating that the number of ways to schedule 4 subjects in 7 periods
    with no consecutive subjects is 840 -/
theorem schedule_theorem : schedule_ways = 840 := by
  sorry

end NUMINAMATH_CALUDE_schedule_theorem_l1500_150029


namespace NUMINAMATH_CALUDE_range_of_m_l1500_150015

def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a^2 = m ∧ b^2 = 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m :
  ∀ m : ℝ, (¬p m ∧ q m) → (1 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1500_150015


namespace NUMINAMATH_CALUDE_linda_age_l1500_150003

theorem linda_age (carlos_age maya_age linda_age : ℕ) : 
  carlos_age = 12 →
  maya_age = carlos_age + 4 →
  linda_age = maya_age - 5 →
  linda_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_linda_age_l1500_150003


namespace NUMINAMATH_CALUDE_parabola_directrix_l1500_150083

/-- The equation of the directrix of the parabola y^2 = 2x is x = -1/2 -/
theorem parabola_directrix : ∀ x y : ℝ, y^2 = 2*x → (∃ p : ℝ, p > 0 ∧ x = -p/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1500_150083


namespace NUMINAMATH_CALUDE_red_car_position_implies_816_cars_l1500_150094

/-- The position of the red car in the parking lot -/
structure CarPosition where
  left : ℕ
  right : ℕ
  front : ℕ
  back : ℕ

/-- The dimensions of the parking lot -/
def parkingLotDimensions (pos : CarPosition) : ℕ × ℕ :=
  (pos.left + pos.right - 1, pos.front + pos.back - 1)

/-- The total number of cars in the parking lot -/
def totalCars (pos : CarPosition) : ℕ :=
  let (width, length) := parkingLotDimensions pos
  width * length

/-- Theorem stating that given the red car's position, the total number of cars is 816 -/
theorem red_car_position_implies_816_cars (pos : CarPosition)
    (h_left : pos.left = 19)
    (h_right : pos.right = 16)
    (h_front : pos.front = 14)
    (h_back : pos.back = 11) :
    totalCars pos = 816 := by
  sorry

#eval totalCars ⟨19, 16, 14, 11⟩

end NUMINAMATH_CALUDE_red_car_position_implies_816_cars_l1500_150094


namespace NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_wheat_profit_approximately_30_percent_l1500_150007

/-- Calculates the profit percentage for a wheat mixture sale --/
theorem wheat_mixture_profit_percentage 
  (wheat1_weight : ℝ) (wheat1_price : ℝ) 
  (wheat2_weight : ℝ) (wheat2_price : ℝ) 
  (selling_price : ℝ) : ℝ :=
  let total_cost := wheat1_weight * wheat1_price + wheat2_weight * wheat2_price
  let total_weight := wheat1_weight + wheat2_weight
  let selling_total := total_weight * selling_price
  let profit := selling_total - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage

/-- Proves that the profit percentage is approximately 30% --/
theorem wheat_profit_approximately_30_percent : 
  abs (wheat_mixture_profit_percentage 30 11.50 20 14.25 16.38 - 30) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_wheat_profit_approximately_30_percent_l1500_150007


namespace NUMINAMATH_CALUDE_dance_move_ratio_l1500_150033

/-- Frank's dance move sequence --/
structure DanceMove where
  initial_back : ℤ
  first_forward : ℤ
  second_back : ℤ
  final_forward : ℤ

/-- The dance move Frank performs --/
def franks_move : DanceMove :=
  { initial_back := 5
  , first_forward := 10
  , second_back := 2
  , final_forward := 4 }

/-- The final position relative to the starting point --/
def final_position (move : DanceMove) : ℤ :=
  -move.initial_back + move.first_forward - move.second_back + move.final_forward

/-- The theorem stating the ratio of final forward steps to second back steps --/
theorem dance_move_ratio (move : DanceMove) : 
  final_position move = 7 → 
  (move.final_forward : ℚ) / move.second_back = 2 := by
  sorry

#eval final_position franks_move

end NUMINAMATH_CALUDE_dance_move_ratio_l1500_150033


namespace NUMINAMATH_CALUDE_butterfat_mixture_proof_l1500_150000

-- Define the initial quantities and percentages
def initial_volume : ℝ := 8
def initial_butterfat_percentage : ℝ := 0.35
def added_butterfat_percentage : ℝ := 0.10
def target_butterfat_percentage : ℝ := 0.20

-- Define the volume of milk to be added
def added_volume : ℝ := 12

-- Theorem statement
theorem butterfat_mixture_proof :
  let total_volume := initial_volume + added_volume
  let total_butterfat := initial_volume * initial_butterfat_percentage + added_volume * added_butterfat_percentage
  total_butterfat / total_volume = target_butterfat_percentage := by
sorry


end NUMINAMATH_CALUDE_butterfat_mixture_proof_l1500_150000


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000_l1500_150035

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1300000 :
  to_scientific_notation 1300000 = ScientificNotation.mk 1.3 6 sorry :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000_l1500_150035


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1500_150038

theorem complex_magnitude_equation (s : ℝ) (hs : s > 0) :
  Complex.abs (3 + s * Complex.I) = 13 → s = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1500_150038


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1500_150070

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 12) ∣ (m^3 + 144))) ∧ 
  ((n + 12) ∣ (n^3 + 144)) ∧ 
  n = 84 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1500_150070


namespace NUMINAMATH_CALUDE_min_value_expression_l1500_150098

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x - 2*y)^2 = (x*y)^3) : 
  4/x^2 + 4/(x*y) + 1/y^2 ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1500_150098


namespace NUMINAMATH_CALUDE_play_school_kids_l1500_150090

/-- The number of kids in a play school -/
def total_kids (white : ℕ) (yellow : ℕ) (both : ℕ) : ℕ :=
  white + yellow - both

/-- Theorem: The total number of kids in the play school is 35 -/
theorem play_school_kids : total_kids 26 28 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_play_school_kids_l1500_150090


namespace NUMINAMATH_CALUDE_grid_diagonal_property_l1500_150022

/-- Represents a cell color in the grid -/
inductive Color
| Black
| White

/-- Represents a 100 x 100 grid -/
def Grid := Fin 100 → Fin 100 → Color

/-- A predicate that checks if a cell is on the boundary of the grid -/
def isBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- A predicate that checks if a 2x2 subgrid is monochromatic -/
def isMonochromatic (g : Grid) (i j : Fin 100) : Prop :=
  g i j = g (i+1) j ∧ g i j = g i (j+1) ∧ g i j = g (i+1) (j+1)

/-- A predicate that checks if a 2x2 subgrid has the desired diagonal property -/
def hasDiagonalProperty (g : Grid) (i j : Fin 100) : Prop :=
  (g i j = g (i+1) (j+1) ∧ g i (j+1) = g (i+1) j ∧ g i j ≠ g i (j+1))
  ∨ (g i j = g (i+1) (j+1) ∧ g i (j+1) = g (i+1) j ∧ g i (j+1) ≠ g i j)

theorem grid_diagonal_property (g : Grid) 
  (boundary_black : ∀ i j, isBoundary i j → g i j = Color.Black)
  (no_monochromatic : ∀ i j, ¬isMonochromatic g i j) :
  ∃ i j, hasDiagonalProperty g i j := by
  sorry

end NUMINAMATH_CALUDE_grid_diagonal_property_l1500_150022


namespace NUMINAMATH_CALUDE_negation_of_implication_l1500_150076

theorem negation_of_implication (a b : ℝ) :
  ¬(∀ x : ℝ, x < a → x < b) ↔ (∃ x : ℝ, x ≥ a ∧ x ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1500_150076


namespace NUMINAMATH_CALUDE_neighbor_rolls_count_l1500_150099

/-- The number of gift wrap rolls Nellie needs to sell for the fundraiser -/
def total_rolls : ℕ := 45

/-- The number of rolls Nellie sold to her grandmother -/
def grandmother_rolls : ℕ := 1

/-- The number of rolls Nellie sold to her uncle -/
def uncle_rolls : ℕ := 10

/-- The number of rolls Nellie still needs to sell to reach her goal -/
def remaining_rolls : ℕ := 28

/-- The number of rolls Nellie sold to her neighbor -/
def neighbor_rolls : ℕ := total_rolls - remaining_rolls - grandmother_rolls - uncle_rolls

theorem neighbor_rolls_count : neighbor_rolls = 6 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_rolls_count_l1500_150099


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1500_150097

theorem consecutive_integers_average (c d : ℝ) : 
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6) →
  ((d-2) + (d-1) + d + (d+1) + (d+2) + (d+3)) / 6 = c + 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1500_150097


namespace NUMINAMATH_CALUDE_two_sector_area_l1500_150031

/-- The area of a figure formed by two sectors of a circle -/
theorem two_sector_area (r : ℝ) (angle1 angle2 : ℝ) (h1 : r = 15) (h2 : angle1 = 90) (h3 : angle2 = 45) :
  (angle1 / 360) * π * r^2 + (angle2 / 360) * π * r^2 = 84.375 * π := by
  sorry

#check two_sector_area

end NUMINAMATH_CALUDE_two_sector_area_l1500_150031


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_progression_l1500_150052

def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_progression
  (a : ℕ → ℝ)
  (h_ap : arithmetic_progression a)
  (h_sum : a 1 + a 2 + a 3 = 168)
  (h_diff : a 2 - a 5 = 42) :
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_progression_l1500_150052


namespace NUMINAMATH_CALUDE_line_intersection_plane_parallel_l1500_150001

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the intersection of lines
variable (intersect : Line → Line → Prop)

-- Define parallel planes
variable (parallel : Plane → Plane → Prop)

-- Define the statement
theorem line_intersection_plane_parallel 
  (l m : Line) (α β : Plane) 
  (h1 : subset l α) (h2 : subset m β) :
  (¬ intersect l m → parallel α β) ∧ 
  ¬ (¬ intersect l m → parallel α β) ∧ 
  (parallel α β → ¬ intersect l m) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_plane_parallel_l1500_150001


namespace NUMINAMATH_CALUDE_hexagon_interior_exterior_angle_sum_l1500_150014

theorem hexagon_interior_exterior_angle_sum : 
  ∃! n : ℕ, n > 2 ∧ (n - 2) * 180 = 2 * 360 := by sorry

#check hexagon_interior_exterior_angle_sum

end NUMINAMATH_CALUDE_hexagon_interior_exterior_angle_sum_l1500_150014


namespace NUMINAMATH_CALUDE_system_solution_exists_iff_l1500_150023

theorem system_solution_exists_iff (k : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = 4 ∧ k * x^2 + y = 5) ↔ k > -1/36 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_iff_l1500_150023


namespace NUMINAMATH_CALUDE_privateer_overtakes_at_6_08_pm_l1500_150006

/-- Represents the chase scenario between a privateer and a merchantman -/
structure ChaseScenario where
  initial_distance : ℝ
  privateer_initial_speed : ℝ
  merchantman_speed : ℝ
  time_before_damage : ℝ
  new_speed_ratio_privateer : ℝ
  new_speed_ratio_merchantman : ℝ

/-- Calculates the time when the privateer overtakes the merchantman -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific chase scenario, the privateer overtakes the merchantman at 6:08 p.m. -/
theorem privateer_overtakes_at_6_08_pm (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 12)
  (h2 : scenario.privateer_initial_speed = 10)
  (h3 : scenario.merchantman_speed = 7)
  (h4 : scenario.time_before_damage = 3)
  (h5 : scenario.new_speed_ratio_privateer = 13)
  (h6 : scenario.new_speed_ratio_merchantman = 12) :
  overtake_time scenario = 8.1333333333 :=
  sorry

#eval 10 + 8.1333333333  -- Should output approximately 18.1333333333, representing 6:08 p.m.

end NUMINAMATH_CALUDE_privateer_overtakes_at_6_08_pm_l1500_150006


namespace NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l1500_150005

/-- Given a unit square AEFD, prove that if the ratio of length to width of rectangle ABCD
    equals the ratio of length to width of rectangle BCFE, then the length of AB (W) is (1 + √5) / 2. -/
theorem rectangle_ratio_golden_ratio (W : ℝ) : 
  W > 0 ∧ W / 1 = 1 / (W - 1) → W = (1 + Real.sqrt 5) / 2 := by
  sorry

#check rectangle_ratio_golden_ratio

end NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l1500_150005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1500_150061

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 13 = 40) →
  (a 8 + a 9 + a 10 = 60) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1500_150061


namespace NUMINAMATH_CALUDE_simple_annual_interest_rate_l1500_150079

/-- Calculate the simple annual interest rate given monthly interest payment and principal amount -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (principal : ℝ) 
  (h1 : monthly_interest = 234)
  (h2 : principal = 31200) :
  (monthly_interest * 12 / principal) * 100 = 8.99 := by
sorry

end NUMINAMATH_CALUDE_simple_annual_interest_rate_l1500_150079


namespace NUMINAMATH_CALUDE_equal_roots_condition_l1500_150002

theorem equal_roots_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2*x - (m^2 + 2)) / ((x^2 - 2)*(m - 2)) = x / m) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l1500_150002


namespace NUMINAMATH_CALUDE_square_area_above_line_l1500_150026

/-- Given a square with vertices at (2,1), (7,1), (7,6), and (2,6),
    and a line connecting points (2,1) and (7,3),
    the fraction of the square's area above this line is 4/5. -/
theorem square_area_above_line : 
  let square_vertices : List (ℝ × ℝ) := [(2,1), (7,1), (7,6), (2,6)]
  let line_points : List (ℝ × ℝ) := [(2,1), (7,3)]
  let total_area : ℝ := 25
  let area_above_line : ℝ := 20
  (area_above_line / total_area) = 4/5 := by sorry

end NUMINAMATH_CALUDE_square_area_above_line_l1500_150026


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1500_150089

def is_hyperbola (a b : ℝ) (h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1

def has_asymptote (h : ℝ → ℝ → Prop) : Prop :=
  ∀ x, h x (Real.sqrt 3 * x)

def has_focus (h : ℝ → ℝ → Prop) : Prop :=
  h 2 0

theorem hyperbola_equation (a b : ℝ) (h : ℝ → ℝ → Prop)
  (ha : a > 0) (hb : b > 0)
  (h_hyp : is_hyperbola a b h)
  (h_asym : has_asymptote h)
  (h_focus : has_focus h) :
  ∀ x y, h x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1500_150089


namespace NUMINAMATH_CALUDE_rhombus_diagonal_theorem_l1500_150063

/-- Represents a rhombus with given properties -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem stating the relationship between the diagonals and perimeter of a rhombus -/
theorem rhombus_diagonal_theorem (r : Rhombus) (h1 : r.diagonal1 = 24) (h2 : r.perimeter = 52) :
  r.diagonal2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_theorem_l1500_150063


namespace NUMINAMATH_CALUDE_james_hall_of_mirrors_glass_area_l1500_150018

/-- The total area of glass needed for three walls in a hall of mirrors --/
def total_glass_area (long_wall_length long_wall_height short_wall_length short_wall_height : ℝ) : ℝ :=
  2 * (long_wall_length * long_wall_height) + (short_wall_length * short_wall_height)

/-- Theorem: The total area of glass needed for James' hall of mirrors is 960 square feet --/
theorem james_hall_of_mirrors_glass_area :
  total_glass_area 30 12 20 12 = 960 := by
  sorry

end NUMINAMATH_CALUDE_james_hall_of_mirrors_glass_area_l1500_150018


namespace NUMINAMATH_CALUDE_apple_percentage_after_removal_l1500_150073

def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

def remaining_oranges : ℕ := initial_oranges - oranges_removed
def total_fruit_after : ℕ := initial_apples + remaining_oranges

theorem apple_percentage_after_removal : 
  (initial_apples : ℚ) / total_fruit_after * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_removal_l1500_150073


namespace NUMINAMATH_CALUDE_youngest_child_age_l1500_150042

/-- Given 5 children born at intervals of 3 years, if the sum of their ages is 50 years,
    then the age of the youngest child is 4 years. -/
theorem youngest_child_age (children : ℕ) (interval : ℕ) (total_age : ℕ) :
  children = 5 →
  interval = 3 →
  total_age = 50 →
  total_age = (children - 1) * children / 2 * interval + children * (youngest_age : ℕ) →
  youngest_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l1500_150042


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1500_150066

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate n 1 ++ [1 + 1 / n]
  (set.sum / set.length : ℚ) = 1 + 1 / (n * (n + 1)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1500_150066


namespace NUMINAMATH_CALUDE_max_third_altitude_l1500_150024

/-- A scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  /-- The length of the first altitude -/
  altitude1 : ℝ
  /-- The length of the second altitude -/
  altitude2 : ℝ
  /-- The length of the third altitude -/
  altitude3 : ℝ
  /-- Condition that the triangle is scalene -/
  scalene : altitude1 ≠ altitude2 ∧ altitude2 ≠ altitude3 ∧ altitude3 ≠ altitude1
  /-- Condition that two altitudes are 6 and 18 -/
  specific_altitudes : (altitude1 = 6 ∧ altitude2 = 18) ∨ (altitude1 = 18 ∧ altitude2 = 6) ∨
                       (altitude1 = 6 ∧ altitude3 = 18) ∨ (altitude1 = 18 ∧ altitude3 = 6) ∨
                       (altitude2 = 6 ∧ altitude3 = 18) ∨ (altitude2 = 18 ∧ altitude3 = 6)
  /-- Condition that the third altitude is an integer -/
  integer_altitude : ∃ n : ℤ, (altitude3 : ℝ) = n ∨ (altitude2 : ℝ) = n ∨ (altitude1 : ℝ) = n

/-- The maximum possible integer length of the third altitude is 8 -/
theorem max_third_altitude (t : ScaleneTriangle) : 
  ∃ (max_altitude : ℕ), max_altitude = 8 ∧ 
  ∀ (n : ℕ), (n : ℝ) = t.altitude1 ∨ (n : ℝ) = t.altitude2 ∨ (n : ℝ) = t.altitude3 → n ≤ max_altitude :=
sorry

end NUMINAMATH_CALUDE_max_third_altitude_l1500_150024


namespace NUMINAMATH_CALUDE_no_polynomial_exists_l1500_150065

theorem no_polynomial_exists : ¬∃ (P : ℝ → ℝ → ℝ), 
  (∀ x y, x ∈ ({1, 2, 3} : Set ℝ) → y ∈ ({1, 2, 3} : Set ℝ) → 
    P x y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 10} : Set ℝ)) ∧ 
  (∀ v, v ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 10} : Set ℝ) → 
    ∃! (x y : ℝ), x ∈ ({1, 2, 3} : Set ℝ) ∧ y ∈ ({1, 2, 3} : Set ℝ) ∧ P x y = v) ∧
  (∃ (a b c d e f : ℝ), ∀ x y, P x y = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f) :=
by sorry

end NUMINAMATH_CALUDE_no_polynomial_exists_l1500_150065
