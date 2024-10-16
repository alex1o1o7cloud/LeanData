import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3588_358815

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3588_358815


namespace NUMINAMATH_CALUDE_race_problem_l3588_358872

/-- The race problem on a circular lake -/
theorem race_problem (lake_circumference : ℝ) (serezha_speed : ℝ) (dima_run_speed : ℝ) 
  (serezha_time : ℝ) (dima_run_time : ℝ) :
  serezha_speed = 20 →
  dima_run_speed = 6 →
  serezha_time = 0.5 →
  dima_run_time = 0.25 →
  ∃ (total_time : ℝ), total_time = 37.5 / 60 := by
  sorry

#check race_problem

end NUMINAMATH_CALUDE_race_problem_l3588_358872


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l3588_358871

/-- Triangle ABC with vertices A(3,-4), B(6,0), and C(-5,2) -/
structure Triangle where
  A : ℝ × ℝ := (3, -4)
  B : ℝ × ℝ := (6, 0)
  C : ℝ × ℝ := (-5, 2)

/-- Line equation in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of altitude BD and median BE -/
theorem triangle_altitude_and_median (t : Triangle) :
  ∃ (altitude_bd median_be : LineEquation),
    altitude_bd = ⟨4, -3, -24⟩ ∧
    median_be = ⟨1, -7, -6⟩ := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l3588_358871


namespace NUMINAMATH_CALUDE_fraction_value_l3588_358887

theorem fraction_value : (20 + 24) / (20 - 24) = -11 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l3588_358887


namespace NUMINAMATH_CALUDE_shrink_ray_effect_l3588_358892

/-- Represents the shrink ray's effect on volume -/
def shrink_factor : ℝ := 0.5

/-- The number of coffee cups -/
def num_cups : ℕ := 5

/-- The initial volume of coffee in each cup (in ounces) -/
def initial_volume : ℝ := 8

/-- Calculates the total volume of coffee after shrinking -/
def total_volume_after_shrink : ℝ := num_cups * (initial_volume * shrink_factor)

theorem shrink_ray_effect :
  total_volume_after_shrink = 20 := by sorry

end NUMINAMATH_CALUDE_shrink_ray_effect_l3588_358892


namespace NUMINAMATH_CALUDE_system_solution_existence_l3588_358834

theorem system_solution_existence (k : ℝ) :
  (∃ (x y : ℝ), y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3588_358834


namespace NUMINAMATH_CALUDE_pentagon_divisible_hexagon_divisible_heptagon_not_divisible_l3588_358835

/-- A polygon is a closed shape with a certain number of sides and vertices. -/
structure Polygon where
  sides : ℕ
  vertices : ℕ

/-- A triangle is a polygon with 3 sides and 3 vertices. -/
def Triangle : Polygon := ⟨3, 3⟩

/-- A pentagon is a polygon with 5 sides and 5 vertices. -/
def Pentagon : Polygon := ⟨5, 5⟩

/-- A hexagon is a polygon with 6 sides and 6 vertices. -/
def Hexagon : Polygon := ⟨6, 6⟩

/-- A heptagon is a polygon with 7 sides and 7 vertices. -/
def Heptagon : Polygon := ⟨7, 7⟩

/-- A polygon can be divided into two triangles if there exists a way to combine two triangles to form that polygon. -/
def CanBeDividedIntoTwoTriangles (p : Polygon) : Prop :=
  ∃ (t1 t2 : Polygon), t1 = Triangle ∧ t2 = Triangle ∧ p.sides = t1.sides + t2.sides - 2 ∧ p.vertices = t1.vertices + t2.vertices - 2

theorem pentagon_divisible : CanBeDividedIntoTwoTriangles Pentagon := by sorry

theorem hexagon_divisible : CanBeDividedIntoTwoTriangles Hexagon := by sorry

theorem heptagon_not_divisible : ¬CanBeDividedIntoTwoTriangles Heptagon := by sorry

end NUMINAMATH_CALUDE_pentagon_divisible_hexagon_divisible_heptagon_not_divisible_l3588_358835


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_l3588_358890

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 = 1/3 →
  a 2 + a 5 = 4 →
  ∃ n : ℕ, a n = 33 →
  ∃ n : ℕ, a n = 33 ∧ n = 50 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_l3588_358890


namespace NUMINAMATH_CALUDE_second_train_speed_l3588_358812

/-- Given two trains traveling towards each other, prove that the speed of the second train is 16 km/hr -/
theorem second_train_speed
  (speed_train1 : ℝ)
  (total_distance : ℝ)
  (distance_difference : ℝ)
  (h1 : speed_train1 = 20)
  (h2 : total_distance = 630)
  (h3 : distance_difference = 70)
  : ∃ (speed_train2 : ℝ), speed_train2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l3588_358812


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3588_358820

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its possible cuts --/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Generates all possible ways to cut the plywood into congruent pieces --/
def possible_cuts (p : Plywood) : List Rectangle :=
  sorry -- Implementation details omitted

/-- Finds the maximum perimeter from a list of rectangles --/
def max_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

/-- Finds the minimum perimeter from a list of rectangles --/
def min_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

theorem plywood_cut_perimeter_difference :
  let p : Plywood := { length := 9, width := 6, num_pieces := 6 }
  let cuts := possible_cuts p
  max_perimeter cuts - min_perimeter cuts = 10 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3588_358820


namespace NUMINAMATH_CALUDE_lieutenant_age_l3588_358867

theorem lieutenant_age : ∃ (n : ℕ) (x : ℕ),
  n * (n + 5) = x * (n + 9) ∧
  x = 24 := by
  sorry

end NUMINAMATH_CALUDE_lieutenant_age_l3588_358867


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3588_358895

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : b₁ ≠ 0) (h₄ : b₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ a b : ℝ, a * b = k) →
  a₁ / a₂ = 3 / 5 →
  b₁ / b₂ = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3588_358895


namespace NUMINAMATH_CALUDE_michael_crates_tuesday_l3588_358845

/-- The number of crates Michael bought on Tuesday -/
def T : ℕ := sorry

/-- The number of crates Michael gave out -/
def crates_given_out : ℕ := 2

/-- The number of crates Michael bought on Thursday -/
def crates_bought_thursday : ℕ := 5

/-- The number of eggs each crate holds -/
def eggs_per_crate : ℕ := 30

/-- The total number of eggs Michael has now -/
def total_eggs : ℕ := 270

theorem michael_crates_tuesday : T = 6 := by
  sorry

end NUMINAMATH_CALUDE_michael_crates_tuesday_l3588_358845


namespace NUMINAMATH_CALUDE_ash_cloud_radius_l3588_358893

/-- Calculates the radius of an ash cloud from a volcano eruption -/
theorem ash_cloud_radius 
  (angle : Real) 
  (vertical_distance : Real) 
  (diameter_factor : Real) 
  (h1 : angle = 60) 
  (h2 : vertical_distance = 300) 
  (h3 : diameter_factor = 18) : 
  ∃ (radius : Real), abs (radius - 10228.74) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ash_cloud_radius_l3588_358893


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3588_358877

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3588_358877


namespace NUMINAMATH_CALUDE_number_of_female_managers_l3588_358859

/-- Represents a company with employees and managers -/
structure Company where
  totalEmployees : ℕ
  maleEmployees : ℕ
  femaleEmployees : ℕ
  totalManagers : ℕ
  maleManagers : ℕ
  femaleManagers : ℕ

/-- Theorem stating the number of female managers in the company -/
theorem number_of_female_managers (c : Company) : c.femaleManagers = 300 :=
  by
  have h1 : c.femaleEmployees = 750 := by sorry
  have h2 : c.totalManagers = (2 : ℕ) * c.totalEmployees / (5 : ℕ) := by sorry
  have h3 : c.maleManagers = (2 : ℕ) * c.maleEmployees / (5 : ℕ) := by sorry
  have h4 : c.totalEmployees = c.maleEmployees + c.femaleEmployees := by sorry
  have h5 : c.totalManagers = c.maleManagers + c.femaleManagers := by sorry
  sorry

#check number_of_female_managers

end NUMINAMATH_CALUDE_number_of_female_managers_l3588_358859


namespace NUMINAMATH_CALUDE_min_sum_abc_def_l3588_358822

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def are_distinct (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def to_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem min_sum_abc_def :
  ∀ a b c d e f : ℕ,
    is_valid_digit a → is_valid_digit b → is_valid_digit c →
    is_valid_digit d → is_valid_digit e → is_valid_digit f →
    are_distinct a b c d e f →
    459 ≤ to_number a b c + to_number d e f :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_def_l3588_358822


namespace NUMINAMATH_CALUDE_existence_of_integers_l3588_358816

theorem existence_of_integers (a b c : ℤ) : ∃ (p₁ q₁ r₁ p₂ q₂ r₂ : ℤ),
  (a = q₁ * r₂ - q₂ * r₁) ∧
  (b = r₁ * p₂ - r₂ * p₁) ∧
  (c = p₁ * q₂ - p₂ * q₁) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l3588_358816


namespace NUMINAMATH_CALUDE_perfect_square_two_pow_plus_three_l3588_358898

theorem perfect_square_two_pow_plus_three (n : ℕ) : 
  (∃ k : ℕ, 2^n + 3 = k^2) ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_perfect_square_two_pow_plus_three_l3588_358898


namespace NUMINAMATH_CALUDE_counterexample_exists_l3588_358801

/-- A function that returns the sum of digits of a natural number in base 4038 -/
def sumOfDigitsBase4038 (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is "good" (sum of digits in base 4038 is divisible by 2019) -/
def isGood (n : ℕ) : Prop := sumOfDigitsBase4038 n % 2019 = 0

theorem counterexample_exists (a : ℝ) : a ≥ 2019 →
  ∃ (seq : ℕ → ℕ), 
    (∀ m n : ℕ, m ≠ n → seq m ≠ seq n) ∧ 
    (∀ n : ℕ, (seq n : ℝ) ≤ a * n) ∧
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → isGood (seq n)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3588_358801


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_max_l3588_358854

theorem triangle_cosine_sum_max (A B C : ℝ) (h : Real.sin C = 2 * Real.cos A * Real.cos B) :
  ∃ (max : ℝ), max = (Real.sqrt 2 + 1) / 2 ∧ 
    ∀ (A' B' C' : ℝ), Real.sin C' = 2 * Real.cos A' * Real.cos B' →
      Real.cos A' ^ 2 + Real.cos B' ^ 2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_max_l3588_358854


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3588_358838

theorem expression_equals_zero :
  (-1)^2023 - |1 - Real.sqrt 3| + Real.sqrt 6 * Real.sqrt (1/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3588_358838


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3588_358850

theorem grade_assignment_count (num_students : ℕ) (num_grades : ℕ) :
  num_students = 10 →
  num_grades = 4 →
  (num_grades ^ num_students : ℕ) = 1048576 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3588_358850


namespace NUMINAMATH_CALUDE_existence_of_number_with_four_prime_factors_l3588_358821

theorem existence_of_number_with_four_prime_factors : ∃ N : ℕ,
  (∃ p₁ p₂ p₃ p₄ : ℕ, 
    (Nat.Prime p₁) ∧ (Nat.Prime p₂) ∧ (Nat.Prime p₃) ∧ (Nat.Prime p₄) ∧
    (p₁ ≠ p₂) ∧ (p₁ ≠ p₃) ∧ (p₁ ≠ p₄) ∧ (p₂ ≠ p₃) ∧ (p₂ ≠ p₄) ∧ (p₃ ≠ p₄) ∧
    (1 < p₁) ∧ (p₁ ≤ 100) ∧
    (1 < p₂) ∧ (p₂ ≤ 100) ∧
    (1 < p₃) ∧ (p₃ ≤ 100) ∧
    (1 < p₄) ∧ (p₄ ≤ 100) ∧
    (N = p₁ * p₂ * p₃ * p₄) ∧
    (∀ q : ℕ, Nat.Prime q → q ∣ N → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄))) ∧
  N = 210 :=
by
  sorry


end NUMINAMATH_CALUDE_existence_of_number_with_four_prime_factors_l3588_358821


namespace NUMINAMATH_CALUDE_curve_C_not_centrally_symmetric_l3588_358869

-- Define the curve C
def C : ℝ → ℝ := fun x ↦ x^3 - x + 2

-- Theorem statement
theorem curve_C_not_centrally_symmetric :
  ∀ (a b : ℝ), ¬(∀ (x y : ℝ), C x = y → C (2*a - x) = 2*b - y) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_not_centrally_symmetric_l3588_358869


namespace NUMINAMATH_CALUDE_sam_earnings_l3588_358873

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of pennies Sam earned -/
def num_pennies : ℕ := 15

/-- The number of nickels Sam earned -/
def num_nickels : ℕ := 11

/-- The number of dimes Sam earned -/
def num_dimes : ℕ := 21

/-- The number of quarters Sam earned -/
def num_quarters : ℕ := 29

/-- The total value of Sam's earnings in dollars -/
def total_value : ℚ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

theorem sam_earnings : total_value = 10.05 := by
  sorry

end NUMINAMATH_CALUDE_sam_earnings_l3588_358873


namespace NUMINAMATH_CALUDE_min_width_proof_l3588_358852

/-- The minimum width of a rectangular area satisfying given conditions -/
def min_width : ℝ := 5

/-- The length of the rectangular area in terms of its width -/
def length (w : ℝ) : ℝ := 2 * w + 10

/-- The area of the rectangular region -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 120 → w ≥ min_width) ∧
  (area min_width ≥ 120) ∧
  (min_width > 0) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l3588_358852


namespace NUMINAMATH_CALUDE_stamps_per_binder_l3588_358836

-- Define the number of notebooks and stamps per notebook
def num_notebooks : ℕ := 4
def stamps_per_notebook : ℕ := 20

-- Define the number of binders
def num_binders : ℕ := 2

-- Define the fraction of stamps kept
def fraction_kept : ℚ := 1/4

-- Define the number of stamps given away
def stamps_given_away : ℕ := 135

-- Theorem to prove
theorem stamps_per_binder :
  ∃ (x : ℕ), 
    (3/4 : ℚ) * (num_notebooks * stamps_per_notebook + num_binders * x) = stamps_given_away ∧
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_binder_l3588_358836


namespace NUMINAMATH_CALUDE_expression_equals_one_l3588_358851

theorem expression_equals_one (a b : ℝ) 
  (ha : a = Real.sqrt 2 + 0.8)
  (hb : b = Real.sqrt 2 - 0.2) :
  (((2-b)/(b-1)) + 2*((a-1)/(a-2))) / (b*((a-1)/(b-1)) + a*((2-b)/(a-2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3588_358851


namespace NUMINAMATH_CALUDE_problem_statement_l3588_358882

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 1/a + 9/b = 1) 
  (h_ineq : ∀ x : ℝ, a + b ≥ -x^2 + 4*x + 18 - m) : 
  m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3588_358882


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l3588_358848

/-- The area of Andrea's living room floor, given that 20% is covered by a 4ft by 9ft carpet -/
theorem andreas_living_room_area : 
  ∀ (carpet_length carpet_width carpet_area total_area : ℝ),
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_area = carpet_length * carpet_width →
  carpet_area / total_area = 1/5 →
  total_area = 180 := by
sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l3588_358848


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l3588_358870

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability mass function of a binomial random variable -/
def binomial_pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k : ℝ) * X.p^k * (1 - X.p)^(X.n - k)

/-- The main theorem -/
theorem binomial_probability_theorem (X : BinomialRV) 
  (h_p : X.p = 1/3) 
  (h_ev : expected_value X = 2) : 
  binomial_pmf X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l3588_358870


namespace NUMINAMATH_CALUDE_problem_solution_l3588_358813

theorem problem_solution (a b : ℝ) (h1 : |a| = 5) (h2 : b = -2) (h3 : a * b > 0) :
  a + b = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3588_358813


namespace NUMINAMATH_CALUDE_tire_price_problem_l3588_358843

theorem tire_price_problem (total_cost : ℝ) (discount_tire_price : ℝ) 
  (h1 : total_cost = 250)
  (h2 : discount_tire_price = 10) : 
  ∃ (regular_price : ℝ), 3 * regular_price + discount_tire_price = total_cost ∧ regular_price = 80 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_problem_l3588_358843


namespace NUMINAMATH_CALUDE_shells_added_proof_l3588_358860

/-- Given an initial amount of shells and a final amount after adding more,
    calculate the additional amount of shells added. -/
def additional_shells (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given the specific values in the problem,
    the additional amount of shells is 12 pounds. -/
theorem shells_added_proof :
  additional_shells 5 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_proof_l3588_358860


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3588_358896

/-- Proves that the cost of an adult ticket is $16 given the conditions of the problem -/
theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_attendance : ℕ) 
  (total_revenue : ℕ) (child_attendance : ℕ) :
  child_ticket_cost = 9 →
  total_attendance = 24 →
  total_revenue = 258 →
  child_attendance = 18 →
  (total_attendance - child_attendance) * 16 + child_attendance * child_ticket_cost = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3588_358896


namespace NUMINAMATH_CALUDE_select_gloves_count_l3588_358855

/-- The number of ways to select 4 gloves from 6 different pairs such that exactly two of the selected gloves are of the same color -/
def select_gloves : ℕ :=
  (Nat.choose 6 1) * (Nat.choose 10 2 - 5)

/-- Theorem stating that the number of ways to select the gloves is 240 -/
theorem select_gloves_count : select_gloves = 240 := by
  sorry

end NUMINAMATH_CALUDE_select_gloves_count_l3588_358855


namespace NUMINAMATH_CALUDE_p_is_cubic_l3588_358817

/-- The polynomial under consideration -/
def p (x : ℝ) : ℝ := 2^3 + 2^2*x - 2*x^2 - x^3

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

theorem p_is_cubic : degree p = 3 := by sorry

end NUMINAMATH_CALUDE_p_is_cubic_l3588_358817


namespace NUMINAMATH_CALUDE_existence_of_irrational_shifts_l3588_358862

theorem existence_of_irrational_shifts (n : ℕ) (a : Fin n → ℝ) :
  ∃ b : ℝ, ∀ i : Fin n, Irrational (a i + b) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irrational_shifts_l3588_358862


namespace NUMINAMATH_CALUDE_rotationally_invariant_unique_fixed_point_l3588_358885

/-- A function whose graph remains unchanged after rotation by π/2 around the origin -/
def RotationallyInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (-y) = x

/-- The main theorem stating that a rotationally invariant function
    has exactly one fixed point at the origin -/
theorem rotationally_invariant_unique_fixed_point
  (f : ℝ → ℝ) (h : RotationallyInvariant f) :
  (∃! x : ℝ, f x = x) ∧ (∀ x : ℝ, f x = x → x = 0) :=
by sorry


end NUMINAMATH_CALUDE_rotationally_invariant_unique_fixed_point_l3588_358885


namespace NUMINAMATH_CALUDE_sheep_herds_count_l3588_358888

theorem sheep_herds_count (sheep_per_herd : ℕ) (total_sheep : ℕ) (h1 : sheep_per_herd = 20) (h2 : total_sheep = 60) :
  total_sheep / sheep_per_herd = 3 := by
  sorry

end NUMINAMATH_CALUDE_sheep_herds_count_l3588_358888


namespace NUMINAMATH_CALUDE_pascal_zero_property_l3588_358864

/-- Pascal's triangle binomial coefficient -/
def pascal (n k : ℕ) : ℕ := Nat.choose n k

/-- Property that all elements except extremes are zero in a row -/
def all_zero_except_extremes (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k < n → pascal n k = 0

/-- Theorem: If s-th row has all elements zero except extremes,
    then s^k-th rows also have this property for all k ≥ 2 -/
theorem pascal_zero_property (s : ℕ) (hs : s > 1) :
  all_zero_except_extremes s →
  ∀ k, k ≥ 2 → all_zero_except_extremes (s^k) :=
by sorry

end NUMINAMATH_CALUDE_pascal_zero_property_l3588_358864


namespace NUMINAMATH_CALUDE_cats_on_ship_l3588_358844

/-- Represents the passengers on the Queen Mary II luxury liner -/
structure Passengers where
  sailors : ℕ
  cats : ℕ

/-- The total number of heads on the ship -/
def total_heads (p : Passengers) : ℕ := p.sailors + 1 + 1 + p.cats

/-- The total number of legs on the ship -/
def total_legs (p : Passengers) : ℕ := 2 * p.sailors + 2 + 1 + 4 * p.cats

/-- Theorem stating that there are 7 cats on the ship -/
theorem cats_on_ship : 
  ∃ (p : Passengers), total_heads p = 15 ∧ total_legs p = 43 ∧ p.cats = 7 := by
  sorry

end NUMINAMATH_CALUDE_cats_on_ship_l3588_358844


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3588_358841

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) :
  z.im = 5/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3588_358841


namespace NUMINAMATH_CALUDE_wetland_area_scientific_notation_l3588_358826

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a number to its scientific notation representation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem wetland_area_scientific_notation :
  toScientificNotation (29.47 * 1000) = ScientificNotation.mk 2.947 4 :=
sorry

end NUMINAMATH_CALUDE_wetland_area_scientific_notation_l3588_358826


namespace NUMINAMATH_CALUDE_unique_solution_l3588_358886

/-- A 3x3 matrix with special properties -/
structure SpecialMatrix where
  a : Matrix (Fin 3) (Fin 3) ℝ
  all_positive : ∀ i j, 0 < a i j
  row_sum_one : ∀ i, (Finset.univ.sum (λ j => a i j)) = 1
  col_sum_one : ∀ j, (Finset.univ.sum (λ i => a i j)) = 1
  diagonal_half : ∀ i, a i i = 1/2

/-- The system of equations -/
def system (m : SpecialMatrix) (x y z : ℝ) : Prop :=
  m.a 0 0 * x + m.a 0 1 * y + m.a 0 2 * z = 0 ∧
  m.a 1 0 * x + m.a 1 1 * y + m.a 1 2 * z = 0 ∧
  m.a 2 0 * x + m.a 2 1 * y + m.a 2 2 * z = 0

theorem unique_solution (m : SpecialMatrix) :
  ∀ x y z, system m x y z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3588_358886


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3588_358805

/-- Given 6 moles of a compound with a total weight of 252 grams, 
    the molecular weight of the compound is 42 grams/mole. -/
theorem molecular_weight_calculation (moles : ℝ) (total_weight : ℝ) :
  moles = 6 →
  total_weight = 252 →
  total_weight / moles = 42 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3588_358805


namespace NUMINAMATH_CALUDE_skateboard_distance_l3588_358878

/-- Sequence representing the distance covered by the skateboard in each second -/
def skateboardSequence (n : ℕ) : ℕ := 8 + 10 * (n - 1)

/-- The total distance traveled by the skateboard in 20 seconds -/
def totalDistance : ℕ := (Finset.range 20).sum skateboardSequence

/-- Theorem stating that the total distance traveled is 2060 inches -/
theorem skateboard_distance : totalDistance = 2060 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l3588_358878


namespace NUMINAMATH_CALUDE_avery_donation_l3588_358894

theorem avery_donation (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 4 → 
  pants = 2 * shirts → 
  shorts = pants / 2 → 
  shirts + pants + shorts = 16 := by
sorry

end NUMINAMATH_CALUDE_avery_donation_l3588_358894


namespace NUMINAMATH_CALUDE_inequality_proof_l3588_358889

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3588_358889


namespace NUMINAMATH_CALUDE_family_movie_night_l3588_358881

/-- Calculates the number of children in a family given ticket prices and payment information. -/
def number_of_children (regular_ticket_price : ℕ) (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_adults : ℕ) : ℕ :=
  let child_ticket_price := regular_ticket_price - child_discount
  let total_spent := total_payment - change
  let adult_tickets_cost := regular_ticket_price * num_adults
  let children_tickets_cost := total_spent - adult_tickets_cost
  children_tickets_cost / child_ticket_price

/-- Proves that the number of children in the family is 3 given the problem conditions. -/
theorem family_movie_night : number_of_children 9 2 40 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_movie_night_l3588_358881


namespace NUMINAMATH_CALUDE_min_abs_sum_l3588_358811

theorem min_abs_sum (x : ℝ) : 
  |x + 2| + |x + 4| + |x + 5| ≥ 3 ∧ ∃ y : ℝ, |y + 2| + |y + 4| + |y + 5| = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_sum_l3588_358811


namespace NUMINAMATH_CALUDE_special_geometric_sequence_a0_l3588_358819

/-- A geometric sequence with a special sum property -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  isGeometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2
  sumProperty : ∀ n : ℕ, (Finset.range n).sum a = 5^(n + 1) + a 0 - 5

/-- The value of a₀ in a SpecialGeometricSequence is -5 -/
theorem special_geometric_sequence_a0 (seq : SpecialGeometricSequence) : seq.a 0 = -5 := by
  sorry


end NUMINAMATH_CALUDE_special_geometric_sequence_a0_l3588_358819


namespace NUMINAMATH_CALUDE_ravon_has_card_4_l3588_358837

structure Player where
  name : String
  score : Nat
  cards : Finset Nat

def card_set : Finset Nat := Finset.range 10

theorem ravon_has_card_4 (players : Finset Player)
  (h1 : players.card = 5)
  (h2 : ∀ p ∈ players, p.cards ⊆ card_set)
  (h3 : ∀ p ∈ players, p.cards.card = 2)
  (h4 : ∀ p ∈ players, p.score = (p.cards.sum id))
  (h5 : ∃ p ∈ players, p.name = "Ravon" ∧ p.score = 11)
  (h6 : ∃ p ∈ players, p.name = "Oscar" ∧ p.score = 4)
  (h7 : ∃ p ∈ players, p.name = "Aditi" ∧ p.score = 7)
  (h8 : ∃ p ∈ players, p.name = "Tyrone" ∧ p.score = 16)
  (h9 : ∃ p ∈ players, p.name = "Kim" ∧ p.score = 17)
  (h10 : ∀ c ∈ card_set, (players.filter (λ p => c ∈ p.cards)).card = 1) :
  ∃ p ∈ players, p.name = "Ravon" ∧ 4 ∈ p.cards :=
sorry

end NUMINAMATH_CALUDE_ravon_has_card_4_l3588_358837


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l3588_358883

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci of the hyperbola
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define point A
def point_A : ℝ × ℝ := (6, -2)

-- Define the line passing through right focus and point A
def line_through_right_focus (x y : ℝ) : Prop := 2*x + y - 10 = 0

-- Define the perpendicular line passing through left focus
def perpendicular_line (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- The theorem to prove
theorem intersection_point_coordinates :
  ∃ (x y : ℝ), 
    hyperbola x y ∧
    line_through_right_focus x y ∧
    perpendicular_line x y ∧
    x = 3 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l3588_358883


namespace NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l3588_358853

theorem at_least_one_positive_discriminant (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l3588_358853


namespace NUMINAMATH_CALUDE_divisor_count_l3588_358899

def n : ℕ := 2028
def k : ℕ := 2004

theorem divisor_count (h : n = 2^2 * 3^2 * 13^2) : 
  (Finset.filter (fun d => (Finset.filter (fun x => x ∣ d) (Finset.range (n^k + 1))).card = n) 
   (Finset.filter (fun x => x ∣ n^k) (Finset.range (n^k + 1)))).card = 216 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_l3588_358899


namespace NUMINAMATH_CALUDE_car_speed_l3588_358839

/-- Represents a kilometer marker with two digits -/
structure Marker where
  tens : ℕ
  ones : ℕ
  h_digits : tens < 10 ∧ ones < 10

/-- Represents the time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24 ∧ minutes < 60

/-- Represents an observation of a marker at a specific time -/
structure Observation where
  time : Time
  marker : Marker

def speed_kmh (start_obs end_obs : Observation) : ℚ :=
  let time_diff := (end_obs.time.hours - start_obs.time.hours : ℚ) + 
                   ((end_obs.time.minutes - start_obs.time.minutes : ℚ) / 60)
  let distance := (end_obs.marker.tens * 10 + end_obs.marker.ones) - 
                  (start_obs.marker.tens * 10 + start_obs.marker.ones)
  distance / time_diff

theorem car_speed 
  (obs1 obs2 obs3 : Observation)
  (h_time1 : obs1.time = ⟨12, 0, by norm_num⟩)
  (h_time2 : obs2.time = ⟨12, 42, by norm_num⟩)
  (h_time3 : obs3.time = ⟨13, 0, by norm_num⟩)
  (h_marker1 : obs1.marker = ⟨obs1.marker.tens, obs1.marker.ones, by sorry⟩)
  (h_marker2 : obs2.marker = ⟨obs1.marker.ones, obs1.marker.tens, by sorry⟩)
  (h_marker3 : obs3.marker = ⟨obs1.marker.tens, obs1.marker.ones, by sorry⟩)
  (h_constant_speed : speed_kmh obs1 obs2 = speed_kmh obs2 obs3) :
  speed_kmh obs1 obs3 = 90 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_l3588_358839


namespace NUMINAMATH_CALUDE_coin_box_problem_l3588_358802

theorem coin_box_problem :
  ∃ (N B : ℕ), 
    N = 9 * (B - 2) ∧
    N - 6 * (B - 3) = 3 :=
by sorry

end NUMINAMATH_CALUDE_coin_box_problem_l3588_358802


namespace NUMINAMATH_CALUDE_cups_filled_l3588_358830

-- Define the volume of water in milliliters
def water_volume : ℕ := 1000

-- Define the cup size in milliliters
def cup_size : ℕ := 200

-- Theorem to prove
theorem cups_filled (water_volume : ℕ) (cup_size : ℕ) :
  water_volume = 1000 → cup_size = 200 → water_volume / cup_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_cups_filled_l3588_358830


namespace NUMINAMATH_CALUDE_spatial_geometry_l3588_358803

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (linePerpendicular : Line → Line → Prop)
variable (lineParallel : Line → Line → Prop)
variable (planePerpendicular : Line → Plane → Prop)

-- State the theorem
theorem spatial_geometry 
  (α β : Plane) (l m n : Line) 
  (h1 : perpendicular α β)
  (h2 : intersect α β = l)
  (h3 : contains α m)
  (h4 : contains β n)
  (h5 : linePerpendicular m n) :
  (lineParallel n l → planePerpendicular m β) ∧
  (planePerpendicular m β ∨ planePerpendicular n α) :=
sorry

end NUMINAMATH_CALUDE_spatial_geometry_l3588_358803


namespace NUMINAMATH_CALUDE_expand_product_l3588_358857

theorem expand_product (x : ℝ) : (x^2 - 3*x + 4) * (x^2 + 3*x + 1) = x^4 - 4*x^2 + 9*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3588_358857


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l3588_358897

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l3588_358897


namespace NUMINAMATH_CALUDE_second_to_first_rocket_height_ratio_l3588_358824

def first_rocket_height : ℝ := 500
def combined_height : ℝ := 1500

theorem second_to_first_rocket_height_ratio :
  (combined_height - first_rocket_height) / first_rocket_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_first_rocket_height_ratio_l3588_358824


namespace NUMINAMATH_CALUDE_complex_simplification_l3588_358875

theorem complex_simplification (i : ℂ) : i^2 = -1 → (2 + i^3) / (1 - i) = (3 + i) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3588_358875


namespace NUMINAMATH_CALUDE_four_line_corresponding_angles_l3588_358842

/-- Represents a line in a plane -/
structure Line

/-- Represents an intersection point of two lines -/
structure IntersectionPoint

/-- Represents a pair of corresponding angles -/
structure CorrespondingAnglePair

/-- A configuration of four lines intersecting pairwise -/
structure FourLineConfiguration where
  lines : Fin 4 → Line
  intersections : Fin 6 → IntersectionPoint
  no_triple_intersection : ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ∃ (p q : IntersectionPoint), p ≠ q ∧ 
    (p ∈ (Set.range intersections) ∧ q ∈ (Set.range intersections))

/-- The number of corresponding angle pairs in a four-line configuration -/
def num_corresponding_angles (config : FourLineConfiguration) : ℕ :=
  48

/-- Theorem stating that a four-line configuration has 48 pairs of corresponding angles -/
theorem four_line_corresponding_angles (config : FourLineConfiguration) :
  num_corresponding_angles config = 48 := by sorry

end NUMINAMATH_CALUDE_four_line_corresponding_angles_l3588_358842


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3588_358876

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, and c are digits
  (0 < y ∧ y ≤ 10) →  -- 0 < y ≤ 10
  (1000 * (1 : ℚ) / y = 100 * a + 10 * b + c) →  -- 0.abc = 1/y
  (∀ a' b' c' : ℕ, 
    (a' < 10 ∧ b' < 10 ∧ c' < 10) →
    (∃ y' : ℕ, 0 < y' ∧ y' ≤ 10 ∧ 1000 * (1 : ℚ) / y' = 100 * a' + 10 * b' + c') →
    a' + b' + c' ≤ a + b + c) →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3588_358876


namespace NUMINAMATH_CALUDE_pyramid_edge_length_l3588_358827

/-- A pyramid with 8 edges of equal length -/
structure Pyramid where
  edge_count : ℕ
  edge_length : ℝ
  total_length : ℝ
  edge_count_eq : edge_count = 8
  total_eq : total_length = edge_count * edge_length

/-- Theorem: If a pyramid has 8 edges of equal length, and the sum of all edges is 14.8 meters,
    then the length of each edge is 1.85 meters. -/
theorem pyramid_edge_length (p : Pyramid) (h : p.total_length = 14.8) :
  p.edge_length = 1.85 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edge_length_l3588_358827


namespace NUMINAMATH_CALUDE_yard_area_l3588_358884

/-- The area of a rectangular yard with a square cut-out -/
theorem yard_area (length width cut_side : ℝ) (h1 : length = 20) (h2 : width = 18) (h3 : cut_side = 4) :
  length * width - cut_side * cut_side = 344 :=
by sorry

end NUMINAMATH_CALUDE_yard_area_l3588_358884


namespace NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l3588_358849

/-- The measure of the largest angle in a convex hexagon with specific angle measures -/
theorem largest_angle_convex_hexagon : 
  ∀ x : ℝ,
  (x + 2) + (2*x + 3) + (3*x - 1) + (4*x + 2) + (5*x - 4) + (6*x - 3) = 720 →
  max (x + 2) (max (2*x + 3) (max (3*x - 1) (max (4*x + 2) (max (5*x - 4) (6*x - 3))))) = 203 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l3588_358849


namespace NUMINAMATH_CALUDE_polynomial_equation_properties_l3588_358891

theorem polynomial_equation_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) →
  (a₀ = 1 ∧ a₃ = -32 ∧ a₄ = 16 ∧ a₁ + a₂ + a₃ + a₄ = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_properties_l3588_358891


namespace NUMINAMATH_CALUDE_coloring_books_removed_l3588_358814

theorem coloring_books_removed (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 86 →
  shelves = 7 →
  books_per_shelf = 7 →
  initial_stock - (shelves * books_per_shelf) = 37 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_removed_l3588_358814


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l3588_358806

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem largest_two_digit_prime_factor_of_150_choose_75 :
  ∃ (p : ℕ), p = 47 ∧ 
  Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ binomial_coefficient 150 75 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial_coefficient 150 75 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l3588_358806


namespace NUMINAMATH_CALUDE_fraction_inequality_l3588_358856

theorem fraction_inequality (a b c d : ℕ+) (h1 : a + c < 1988) 
  (h2 : (1 : ℚ) - a / b - c / d > 0) : (1 : ℚ) - a / b - c / d > 1 / (1988^3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3588_358856


namespace NUMINAMATH_CALUDE_string_measurement_l3588_358880

theorem string_measurement (string_length : ℚ) (h : string_length = 2/3) :
  let folded_length := string_length / 4
  string_length - folded_length = 1/2 := by sorry

end NUMINAMATH_CALUDE_string_measurement_l3588_358880


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3588_358831

theorem cubic_sum_theorem (x y z a b c : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : x⁻¹ + y⁻¹ + z⁻¹ = c⁻¹)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = a^3 + (3/2) * (a^2 - b^2) * (c - a) := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3588_358831


namespace NUMINAMATH_CALUDE_unique_zero_point_between_consecutive_integers_l3588_358846

open Real

noncomputable def f (a x : ℝ) : ℝ := a * (x^2 + 2/x) - log x

theorem unique_zero_point_between_consecutive_integers (a : ℝ) (h : a > 0) :
  ∃ (x₀ m n : ℝ), 
    (∀ x ≠ x₀, f a x ≠ 0) ∧ 
    (f a x₀ = 0) ∧
    (m < x₀ ∧ x₀ < n) ∧
    (n = m + 1) ∧
    (m + n = 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_between_consecutive_integers_l3588_358846


namespace NUMINAMATH_CALUDE_composite_polynomial_l3588_358807

theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, 1 < k ∧ k < n^(5*n-1) + n^(5*n-2) + n^(5*n-3) + n + 1 ∧ 
  (n^(5*n-1) + n^(5*n-2) + n^(5*n-3) + n + 1) % k = 0 :=
by sorry

end NUMINAMATH_CALUDE_composite_polynomial_l3588_358807


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3588_358879

theorem proposition_equivalence (a : ℝ) : 
  (∀ x : ℝ, ((x < a ∨ x > a + 1) → (x ≤ 1/2 ∨ x ≥ 1)) ∧ 
   ∃ x : ℝ, (x ≤ 1/2 ∨ x ≥ 1) ∧ ¬(x < a ∨ x > a + 1)) ↔ 
  (0 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3588_358879


namespace NUMINAMATH_CALUDE_enclosing_polygon_sides_l3588_358808

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the enclosing arrangement -/
structure EnclosingArrangement :=
  (central : RegularPolygon)
  (enclosing : RegularPolygon)
  (num_enclosing : ℕ)

/-- Checks if the arrangement is symmetrical and without gaps or overlaps -/
def is_valid_arrangement (arr : EnclosingArrangement) : Prop :=
  arr.central.sides = arr.num_enclosing ∧
  arr.num_enclosing * (180 / arr.enclosing.sides) = arr.central.sides * (180 - (arr.central.sides - 2) * 180 / arr.central.sides) / 2

theorem enclosing_polygon_sides
  (arr : EnclosingArrangement)
  (h_valid : is_valid_arrangement arr)
  (h_central_sides : arr.central.sides = 15) :
  arr.enclosing.sides = 15 :=
sorry

end NUMINAMATH_CALUDE_enclosing_polygon_sides_l3588_358808


namespace NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l3588_358840

theorem nested_sqrt_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l3588_358840


namespace NUMINAMATH_CALUDE_broth_per_serving_is_two_point_five_l3588_358804

/-- Represents the number of cups in one pint -/
def cups_per_pint : ℚ := 2

/-- Represents the number of servings -/
def num_servings : ℕ := 8

/-- Represents the number of pints of vegetables and broth combined for all servings -/
def total_pints : ℚ := 14

/-- Represents the number of cups of vegetables in one serving -/
def vegetables_per_serving : ℚ := 1

/-- Calculates the number of cups of broth in one serving of soup -/
def broth_per_serving : ℚ :=
  (total_pints * cups_per_pint - num_servings * vegetables_per_serving) / num_servings

theorem broth_per_serving_is_two_point_five :
  broth_per_serving = 2.5 := by sorry

end NUMINAMATH_CALUDE_broth_per_serving_is_two_point_five_l3588_358804


namespace NUMINAMATH_CALUDE_count_eight_digit_integers_l3588_358818

/-- The number of different 8-digit positive integers -/
def eight_digit_integers : ℕ := 9 * (10^7)

/-- Theorem stating that the number of different 8-digit positive integers is 90,000,000 -/
theorem count_eight_digit_integers : eight_digit_integers = 90000000 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_integers_l3588_358818


namespace NUMINAMATH_CALUDE_first_consignment_cost_price_l3588_358863

/-- Represents a consignment of cloth -/
structure Consignment where
  length : ℕ
  profit_per_meter : ℚ

/-- Calculates the cost price per meter for a given consignment -/
def cost_price_per_meter (c : Consignment) (selling_price : ℚ) : ℚ :=
  (selling_price - c.profit_per_meter * c.length) / c.length

theorem first_consignment_cost_price 
  (c1 : Consignment)
  (c2 : Consignment)
  (c3 : Consignment)
  (selling_price : ℚ) :
  c1.length = 92 ∧ 
  c1.profit_per_meter = 24 ∧
  c2.length = 120 ∧
  c2.profit_per_meter = 30 ∧
  c3.length = 75 ∧
  c3.profit_per_meter = 20 ∧
  selling_price = 9890 →
  cost_price_per_meter c1 selling_price = 83.50 := by
    sorry

end NUMINAMATH_CALUDE_first_consignment_cost_price_l3588_358863


namespace NUMINAMATH_CALUDE_jeremy_age_l3588_358809

/-- Represents the ages of three people -/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  sophia : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.jeremy + 3) + (ages.sebastian + 3) + (ages.sophia + 3) = 150 ∧
  ages.sebastian = ages.jeremy + 4 ∧
  ages.sophia + 3 = 60

/-- The theorem stating Jeremy's current age -/
theorem jeremy_age (ages : Ages) (h : problem_conditions ages) : ages.jeremy = 40 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_age_l3588_358809


namespace NUMINAMATH_CALUDE_noemi_roulette_loss_l3588_358866

/-- Noemi's gambling problem -/
theorem noemi_roulette_loss 
  (initial_amount : ℕ) 
  (final_amount : ℕ) 
  (blackjack_loss : ℕ) 
  (h1 : initial_amount = 1700)
  (h2 : final_amount = 800)
  (h3 : blackjack_loss = 500) :
  initial_amount - final_amount - blackjack_loss = 400 := by
sorry

end NUMINAMATH_CALUDE_noemi_roulette_loss_l3588_358866


namespace NUMINAMATH_CALUDE_min_value_theorem_l3588_358823

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∃ (m : ℝ), m = 16/7 ∧ ∀ (z : ℝ), z ≥ m ↔ z ≥ x^2/(x+1) + y^2/(y+2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3588_358823


namespace NUMINAMATH_CALUDE_lunules_area_equals_triangle_area_l3588_358874

theorem lunules_area_equals_triangle_area (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_pythagorean : c^2 = a^2 + b^2) : 
  π * a^2 + π * b^2 - π * c^2 = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_lunules_area_equals_triangle_area_l3588_358874


namespace NUMINAMATH_CALUDE_unique_reverse_difference_l3588_358868

/-- Reverses the digits of a 4-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

/-- Checks if a number is a 4-digit number -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_reverse_difference :
  ∃! n : ℕ, isFourDigit n ∧ reverseDigits n = n + 8802 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_reverse_difference_l3588_358868


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_l3588_358861

theorem sin_five_pi_sixths : Real.sin (5 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_l3588_358861


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l3588_358865

theorem set_equality_implies_a_equals_one (a : ℝ) :
  let A : Set ℝ := {1, -2, a^2 - 1}
  let B : Set ℝ := {1, a^2 - 3*a, 0}
  A = B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l3588_358865


namespace NUMINAMATH_CALUDE_c_value_satisfies_equation_l3588_358832

/-- Definition of function F -/
def F (a b c d : ℝ) : ℝ := a * b^2 + c * d

/-- Theorem stating that c = 16 satisfies the equation when a = 2 -/
theorem c_value_satisfies_equation :
  ∃ c : ℝ, F 2 3 c 5 = F 2 5 c 3 ∧ c = 16 := by sorry

end NUMINAMATH_CALUDE_c_value_satisfies_equation_l3588_358832


namespace NUMINAMATH_CALUDE_max_submerged_cubes_is_five_l3588_358833

/-- Represents the properties of the cylinder and cubes -/
structure CylinderAndCubes where
  cylinder_diameter : ℝ
  initial_water_height : ℝ
  cube_edge_length : ℝ

/-- Calculates the maximum number of cubes that can be submerged -/
def max_submerged_cubes (props : CylinderAndCubes) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The main theorem stating the maximum number of submerged cubes -/
theorem max_submerged_cubes_is_five (props : CylinderAndCubes) 
  (h1 : props.cylinder_diameter = 2.9)
  (h2 : props.initial_water_height = 4)
  (h3 : props.cube_edge_length = 2) :
  max_submerged_cubes props = 5 := by
  sorry

#check max_submerged_cubes_is_five

end NUMINAMATH_CALUDE_max_submerged_cubes_is_five_l3588_358833


namespace NUMINAMATH_CALUDE_ellipse_equation_l3588_358829

noncomputable section

-- Define the ellipse C
def C (x y : ℝ) (a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

-- Define the foci
def F₁ (c : ℝ) : ℝ × ℝ := (-c, 0)
def F₂ (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define point A
def A : ℝ × ℝ := (2, 0)

-- Define the slope product condition
def slope_product (c : ℝ) : Prop :=
  let k_AF₁ := (0 - (-c)) / (2 - 0)
  let k_AF₂ := (0 - c) / (2 - 0)
  k_AF₁ * k_AF₂ = -1/4

-- Define the distance sum condition for point B
def distance_sum (a : ℝ) : Prop := 2*a = 2*Real.sqrt 2

-- Main theorem
theorem ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∃ c : ℝ, slope_product c ∧ distance_sum a) →
  (∀ x y : ℝ, C x y a b ↔ y^2/2 + x^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3588_358829


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3588_358825

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, 2 * X^6 - X^4 + 4 * X^2 - 7 = (X^2 + 4*X + 3) * q + (-704*X - 706) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3588_358825


namespace NUMINAMATH_CALUDE_smallest_coin_set_l3588_358858

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A function that checks if a given set of coins can make all amounts from 1 to 99 cents --/
def canMakeAllAmounts (coins : List Coin) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount ≤ 99 →
    ∃ (subset : List Coin), subset ⊆ coins ∧ (subset.map coinValue).sum = amount

/-- The theorem stating that 6 is the smallest number of coins needed --/
theorem smallest_coin_set :
  ∃ (coins : List Coin),
    coins.length = 6 ∧
    canMakeAllAmounts coins ∧
    ∀ (other_coins : List Coin),
      canMakeAllAmounts other_coins →
      other_coins.length ≥ 6 :=
by sorry

#check smallest_coin_set

end NUMINAMATH_CALUDE_smallest_coin_set_l3588_358858


namespace NUMINAMATH_CALUDE_sum_of_squares_ratio_l3588_358800

theorem sum_of_squares_ratio (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 3) 
  (h2 : a/x + b/y + c/z = -3) : 
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_ratio_l3588_358800


namespace NUMINAMATH_CALUDE_janes_age_l3588_358810

theorem janes_age (joe_age jane_age : ℕ) 
  (sum_of_ages : joe_age + jane_age = 54)
  (age_difference : joe_age - jane_age = 22) : 
  jane_age = 16 := by
sorry

end NUMINAMATH_CALUDE_janes_age_l3588_358810


namespace NUMINAMATH_CALUDE_factorization_equality_l3588_358847

theorem factorization_equality (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3588_358847


namespace NUMINAMATH_CALUDE_horse_speed_around_square_field_l3588_358828

/-- Given a square field with area 625 km^2 and a horse that runs around it in 4 hours,
    prove that the speed of the horse is 25 km/hour. -/
theorem horse_speed_around_square_field (area : ℝ) (time : ℝ) (horse_speed : ℝ) : 
  area = 625 → time = 4 → horse_speed = (4 * Real.sqrt area) / time → horse_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_horse_speed_around_square_field_l3588_358828
