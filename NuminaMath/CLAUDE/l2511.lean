import Mathlib

namespace NUMINAMATH_CALUDE_odd_three_digit_count_l2511_251199

/-- The set of digits that can be used for the first digit -/
def first_digit_set : Finset Nat := {0, 2}

/-- The set of digits that can be used for the second and third digits -/
def odd_digit_set : Finset Nat := {1, 3, 5}

/-- A function to check if a number is odd -/
def is_odd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a three-digit number has no repeating digits -/
def no_repeats (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The main theorem to be proved -/
theorem odd_three_digit_count : 
  (Finset.filter (λ n : Nat => 
    100 ≤ n ∧ n < 1000 ∧
    (n / 100) ∈ first_digit_set ∧
    ((n / 10) % 10) ∈ odd_digit_set ∧
    (n % 10) ∈ odd_digit_set ∧
    is_odd n ∧
    no_repeats n
  ) (Finset.range 1000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_odd_three_digit_count_l2511_251199


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2511_251146

theorem arithmetic_calculations :
  (1405 - (816 + 487) = 102) ∧
  (3450 - 107 * 13 = 2059) ∧
  (48306 / (311 - 145) = 291) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2511_251146


namespace NUMINAMATH_CALUDE_hyperbola_param_sum_l2511_251138

/-- A hyperbola with given center, focus, and vertex -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- Parameters of the hyperbola equation -/
structure HyperbolaParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given a hyperbola, compute its equation parameters -/
def computeParams (hyp : Hyperbola) : HyperbolaParams := sorry

theorem hyperbola_param_sum :
  let hyp : Hyperbola := {
    center := (1, -1),
    focus := (1, 5),
    vertex := (1, 1)
  }
  let params := computeParams hyp
  params.h + params.k + params.a + params.b = 2 + 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_param_sum_l2511_251138


namespace NUMINAMATH_CALUDE_deceased_member_income_l2511_251157

/-- Proves that given a family of 3 earning members with an average monthly income of Rs. 735,
    if one member dies and the new average income becomes Rs. 650,
    then the income of the deceased member was Rs. 905. -/
theorem deceased_member_income
  (total_income : ℕ)
  (remaining_income : ℕ)
  (h1 : total_income / 3 = 735)
  (h2 : remaining_income / 2 = 650)
  (h3 : total_income > remaining_income) :
  total_income - remaining_income = 905 := by
sorry

end NUMINAMATH_CALUDE_deceased_member_income_l2511_251157


namespace NUMINAMATH_CALUDE_equation_solution_l2511_251118

theorem equation_solution : ∃ x : ℚ, (1/8 : ℚ) + 8/x = 15/x + (1/15 : ℚ) ∧ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2511_251118


namespace NUMINAMATH_CALUDE_january_salary_l2511_251154

theorem january_salary (feb mar apr may : ℕ) 
  (h1 : (feb + mar + apr + may) / 4 = 8300)
  (h2 : may = 6500)
  (h3 : ∃ jan, (jan + feb + mar + apr) / 4 = 8000) :
  ∃ jan, (jan + feb + mar + apr) / 4 = 8000 ∧ jan = 5300 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l2511_251154


namespace NUMINAMATH_CALUDE_min_sum_squares_l2511_251196

theorem min_sum_squares (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, (a + 2) / x = a * x + 2 * b + 1) →
  ∃ m : ℝ, m = 1 / 100 ∧ ∀ a' b' : ℝ, 
    (∃ x ∈ Set.Icc 3 4, (a' + 2) / x = a' * x + 2 * b' + 1) → 
    a' ^ 2 + b' ^ 2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2511_251196


namespace NUMINAMATH_CALUDE_f_sum_theorem_l2511_251179

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_sum_theorem (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 2)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = (4:ℝ) ^ x) :
  f (-5/2) + f 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_sum_theorem_l2511_251179


namespace NUMINAMATH_CALUDE_smallest_range_of_four_integers_with_mean_2017_l2511_251111

/-- Given four different positive integers with a mean of 2017, 
    the smallest possible range between the largest and smallest of these integers is 4. -/
theorem smallest_range_of_four_integers_with_mean_2017 :
  ∀ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (a + b + c + d) / 4 = 2017 →
  (∀ (w x y z : ℕ), 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (w + x + y + z) / 4 = 2017 →
    max w (max x (max y z)) - min w (min x (min y z)) ≥ 4) ∧
  (∃ (p q r s : ℕ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 →
    (p + q + r + s) / 4 = 2017 →
    max p (max q (max r s)) - min p (min q (min r s)) = 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_range_of_four_integers_with_mean_2017_l2511_251111


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l2511_251155

theorem prime_quadratic_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x ≠ y ∧ 
    x^2 - 2*p*x + p^2 - 5*p - 1 = 0 ∧ 
    y^2 - 2*p*y + p^2 - 5*p - 1 = 0) → 
  p = 3 ∨ p = 7 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l2511_251155


namespace NUMINAMATH_CALUDE_negation_equivalence_function_property_l2511_251160

-- Define the statement for the negation of the existential proposition
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
sorry

-- Define the properties for functions f and g
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def positive_derivative_pos (h : ℝ → ℝ) : Prop := ∀ x > 0, deriv h x > 0

-- Theorem for the properties of functions f and g
theorem function_property (f g : ℝ → ℝ) 
  (hodd : odd_function f) (heven : even_function g)
  (hf_deriv : positive_derivative_pos f) (hg_deriv : positive_derivative_pos g) :
  ∀ x < 0, deriv f x > deriv g x :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_function_property_l2511_251160


namespace NUMINAMATH_CALUDE_garrison_provision_theorem_l2511_251167

/-- Calculates the initial number of days provisions were supposed to last for a garrison --/
def initial_provision_days (initial_garrison : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_garrison + reinforcement) * days_after_reinforcement / initial_garrison + days_before_reinforcement

theorem garrison_provision_theorem (initial_garrison : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) :
  initial_garrison = 2000 →
  reinforcement = 600 →
  days_before_reinforcement = 15 →
  days_after_reinforcement = 30 →
  initial_provision_days initial_garrison reinforcement days_before_reinforcement days_after_reinforcement = 39 :=
by
  sorry

#eval initial_provision_days 2000 600 15 30

end NUMINAMATH_CALUDE_garrison_provision_theorem_l2511_251167


namespace NUMINAMATH_CALUDE_work_earnings_equation_l2511_251165

theorem work_earnings_equation (t : ℚ) : (t + 2) * (4 * t - 5) = (2 * t + 1) * (2 * t + 3) + 3 ↔ t = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l2511_251165


namespace NUMINAMATH_CALUDE_ackermann_2_1_l2511_251189

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_2_1 : A 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ackermann_2_1_l2511_251189


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2511_251112

theorem sum_of_coefficients : 
  let p (x : ℝ) := -3*(x^8 - x^5 + 2*x^3 - 6) + 5*(x^4 + 3*x^2) - 4*(x^6 - 5)
  p 1 = 48 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2511_251112


namespace NUMINAMATH_CALUDE_unique_zero_quadratic_l2511_251105

/-- Given a quadratic function f(x) = 3x^2 + 2x - a with a unique zero in (-1, 1),
    prove that a ∈ (1, 5) ∪ {-1/3} -/
theorem unique_zero_quadratic (a : ℝ) :
  (∃! x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ 3 * x^2 + 2 * x - a = 0) →
  (a ∈ Set.Ioo 1 5 ∨ a = -1/3) :=
sorry

end NUMINAMATH_CALUDE_unique_zero_quadratic_l2511_251105


namespace NUMINAMATH_CALUDE_dance_pairing_l2511_251144

-- Define the types for boys and girls
variable {Boy Girl : Type}

-- Define the dancing relation
variable (danced_with : Boy → Girl → Prop)

-- State the theorem
theorem dance_pairing
  (h1 : ∀ b : Boy, ∃ g : Girl, ¬danced_with b g)
  (h2 : ∀ g : Girl, ∃ b : Boy, danced_with b g)
  : ∃ (g g' : Boy) (f f' : Girl),
    danced_with g f ∧ ¬danced_with g f' ∧
    danced_with g' f' ∧ ¬danced_with g' f :=
by sorry

end NUMINAMATH_CALUDE_dance_pairing_l2511_251144


namespace NUMINAMATH_CALUDE_incorrect_calculation_l2511_251101

theorem incorrect_calculation (a : ℝ) : a^3 + a^3 ≠ 2*a^6 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l2511_251101


namespace NUMINAMATH_CALUDE_immediate_prepayment_better_l2511_251191

/-- Represents a mortgage loan with fixed interest rate and annuity payments -/
structure MortgageLoan where
  S : ℝ  -- Initial loan balance
  T : ℝ  -- Monthly payment amount
  r : ℝ  -- Interest rate for the period
  (T_positive : T > 0)
  (r_nonnegative : r ≥ 0)
  (r_less_than_one : r < 1)

/-- Calculates the final balance after immediate partial prepayment -/
def final_balance_immediate (loan : MortgageLoan) : ℝ :=
  loan.S - 2 * loan.T + loan.r * loan.S - 0.5 * loan.r * loan.T + (0.5 * loan.r * loan.S)^2

/-- Calculates the final balance when waiting until the end of the period -/
def final_balance_waiting (loan : MortgageLoan) : ℝ :=
  loan.S - 2 * loan.T + loan.r * loan.S

/-- Theorem stating that immediate partial prepayment results in a lower final balance -/
theorem immediate_prepayment_better (loan : MortgageLoan) :
  final_balance_immediate loan < final_balance_waiting loan :=
by sorry

end NUMINAMATH_CALUDE_immediate_prepayment_better_l2511_251191


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2511_251120

theorem trigonometric_simplification (θ : Real) (h : 0 < θ ∧ θ < π) :
  ((1 + Real.sin θ + Real.cos θ) * (Real.sin (θ/2) - Real.cos (θ/2))) / 
  Real.sqrt (2 + 2 * Real.cos θ) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2511_251120


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2511_251175

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (1/a + 2/b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 2/b₀ = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2511_251175


namespace NUMINAMATH_CALUDE_inequality_proof_l2511_251172

theorem inequality_proof (a b : ℝ) (h : 1/a > 1/b ∧ 1/b > 0) : 
  a^3 < b^3 ∧ Real.sqrt b - Real.sqrt a < Real.sqrt (b - a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2511_251172


namespace NUMINAMATH_CALUDE_height_range_l2511_251148

def heights : List ℕ := [153, 167, 148, 170, 154, 166, 149, 159, 167, 153]

theorem height_range :
  (List.maximum heights).map (λ max =>
    (List.minimum heights).map (λ min =>
      max - min
    )
  ) = some 22 := by
  sorry

end NUMINAMATH_CALUDE_height_range_l2511_251148


namespace NUMINAMATH_CALUDE_remainder_of_x_plus_one_power_2011_l2511_251195

theorem remainder_of_x_plus_one_power_2011 (x : ℤ) :
  (x + 1)^2011 ≡ x [ZMOD (x^2 - x + 1)] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_x_plus_one_power_2011_l2511_251195


namespace NUMINAMATH_CALUDE_rajesh_savings_amount_l2511_251192

/-- Calculates Rajesh's monthly savings based on his salary and spending habits -/
def rajesh_savings (monthly_salary : ℕ) : ℕ :=
  let food_expense := (40 * monthly_salary) / 100
  let medicine_expense := (20 * monthly_salary) / 100
  let remaining := monthly_salary - (food_expense + medicine_expense)
  (60 * remaining) / 100

/-- Theorem stating that Rajesh's monthly savings are 3600 given his salary and spending habits -/
theorem rajesh_savings_amount :
  rajesh_savings 15000 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_rajesh_savings_amount_l2511_251192


namespace NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_equation_l2511_251127

theorem new_energy_vehicle_sales_growth_equation 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (growth_period : ℕ) 
  (x : ℝ) 
  (h1 : initial_sales = 298) 
  (h2 : final_sales = 850) 
  (h3 : growth_period = 2) :
  initial_sales * (1 + x)^growth_period = final_sales :=
by sorry

end NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_equation_l2511_251127


namespace NUMINAMATH_CALUDE_external_tangent_points_theorem_l2511_251170

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def intersect (c1 c2 : Circle) : Prop := sorry

def touches (c1 c2 : Circle) (p : Point) : Prop := sorry

def on_line (p : Point) (l : Line) : Prop := sorry

def passes_through (l : Line) (p : Point) : Prop := sorry

-- Main theorem
theorem external_tangent_points_theorem 
  (C C' : Circle) (X Y : Point) 
  (h1 : intersect C C') 
  (h2 : on_line X (Line.mk 0 1 0)) 
  (h3 : on_line Y (Line.mk 0 1 0)) :
  ∃ (T1 T2 T3 T4 : Point),
    ∀ (P Q R S : Point) (third_circle : Circle),
      touches C third_circle P →
      touches C' third_circle Q →
      on_line R (Line.mk 0 1 0) →
      on_line S (Line.mk 0 1 0) →
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_points_theorem_l2511_251170


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l2511_251107

theorem number_with_specific_remainders : ∃! (N : ℕ), N < 221 ∧ N % 13 = 11 ∧ N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l2511_251107


namespace NUMINAMATH_CALUDE_equation_always_has_real_root_l2511_251152

theorem equation_always_has_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_always_has_real_root_l2511_251152


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l2511_251122

theorem regular_polygon_interior_angle (n : ℕ) (h : n > 2) :
  (n - 2) * 180 / n = 140 → n = 9 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l2511_251122


namespace NUMINAMATH_CALUDE_max_value_of_roots_expression_l2511_251153

theorem max_value_of_roots_expression (a : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + a*x₁ + a = 2 →
  x₂^2 + a*x₂ + a = 2 →
  x₁ ≠ x₂ →
  ∀ b : ℝ, (x₁ - 2*x₂)*(x₂ - 2*x₁) ≤ -63/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_roots_expression_l2511_251153


namespace NUMINAMATH_CALUDE_quadratic_maximum_l2511_251174

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

/-- The point where the maximum occurs -/
def x_max : ℝ := 2

/-- The maximum value of the function -/
def y_max : ℝ := 24

theorem quadratic_maximum :
  (∀ x : ℝ, f x ≤ y_max) ∧ f x_max = y_max :=
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l2511_251174


namespace NUMINAMATH_CALUDE_perimeter_area_ratio_bound_l2511_251158

/-- A shape in the plane formed by a union of finitely many unit squares -/
structure UnitSquareShape where
  squares : Finset (ℤ × ℤ)

/-- The perimeter of a UnitSquareShape -/
def perimeter (S : UnitSquareShape) : ℝ := sorry

/-- The area of a UnitSquareShape -/
def area (S : UnitSquareShape) : ℝ := S.squares.card

/-- The theorem stating that the ratio of perimeter to area is at most 8 -/
theorem perimeter_area_ratio_bound (S : UnitSquareShape) :
  perimeter S / area S ≤ 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_area_ratio_bound_l2511_251158


namespace NUMINAMATH_CALUDE_shoe_pairs_in_box_l2511_251123

theorem shoe_pairs_in_box (total_shoes : ℕ) (prob_matching : ℚ) : 
  total_shoes = 200 →
  prob_matching = 1 / 199 →
  (total_shoes / 2 : ℕ) = 100 :=
by sorry

end NUMINAMATH_CALUDE_shoe_pairs_in_box_l2511_251123


namespace NUMINAMATH_CALUDE_room_area_l2511_251187

/-- The area of a room given the costs of floor replacement -/
theorem room_area (removal_cost : ℝ) (per_sqft_cost : ℝ) (total_cost : ℝ) : 
  removal_cost = 50 →
  per_sqft_cost = 1.25 →
  total_cost = 120 →
  (total_cost - removal_cost) / per_sqft_cost = 56 := by
sorry

end NUMINAMATH_CALUDE_room_area_l2511_251187


namespace NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_l2511_251151

theorem regular_polygon_with_144_degree_angle (n : ℕ) :
  n > 2 →
  (n - 2) * 180 = 144 * n →
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_l2511_251151


namespace NUMINAMATH_CALUDE_birthday_paradox_l2511_251100

theorem birthday_paradox (people : Finset ℕ) (birthdays : ℕ → Fin 366) :
  people.card = 367 → ∃ i j : ℕ, i ∈ people ∧ j ∈ people ∧ i ≠ j ∧ birthdays i = birthdays j :=
sorry

end NUMINAMATH_CALUDE_birthday_paradox_l2511_251100


namespace NUMINAMATH_CALUDE_apples_joan_can_buy_l2511_251149

def total_budget : ℕ := 60
def hummus_containers : ℕ := 2
def hummus_price : ℕ := 5
def chicken_price : ℕ := 20
def bacon_price : ℕ := 10
def vegetables_price : ℕ := 10
def apple_price : ℕ := 2

theorem apples_joan_can_buy :
  (total_budget - (hummus_containers * hummus_price + chicken_price + bacon_price + vegetables_price)) / apple_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_apples_joan_can_buy_l2511_251149


namespace NUMINAMATH_CALUDE_equation_solutions_l2511_251119

theorem equation_solutions :
  (∀ x : ℝ, (x + 2)^2 = 3*(x + 2) ↔ x = -2 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2511_251119


namespace NUMINAMATH_CALUDE_total_trophies_is_950_l2511_251128

/-- The total number of trophies Jack and Michael will have after five years -/
def totalTrophies (michaelCurrent : ℕ) (michaelIncrease : ℕ) (jackMultiplier : ℕ) : ℕ :=
  (michaelCurrent + michaelIncrease) + (jackMultiplier * michaelCurrent)

/-- Proof that the total number of trophies is 950 -/
theorem total_trophies_is_950 :
  totalTrophies 50 150 15 = 950 := by
  sorry

end NUMINAMATH_CALUDE_total_trophies_is_950_l2511_251128


namespace NUMINAMATH_CALUDE_remaining_seeds_l2511_251181

theorem remaining_seeds (initial_seeds : ℕ) (seeds_per_zone : ℕ) (num_zones : ℕ) : 
  initial_seeds = 54000 →
  seeds_per_zone = 3123 →
  num_zones = 7 →
  initial_seeds - (seeds_per_zone * num_zones) = 32139 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_seeds_l2511_251181


namespace NUMINAMATH_CALUDE_syrup_volume_proof_l2511_251176

def final_syrup_volume (original_volume : ℚ) (reduction_factor : ℚ) (sugar_added : ℚ) (cups_per_quart : ℚ) : ℚ :=
  original_volume * cups_per_quart * reduction_factor + sugar_added

theorem syrup_volume_proof (original_volume : ℚ) (reduction_factor : ℚ) (sugar_added : ℚ) (cups_per_quart : ℚ)
  (h1 : original_volume = 6)
  (h2 : reduction_factor = 1 / 12)
  (h3 : sugar_added = 1)
  (h4 : cups_per_quart = 4) :
  final_syrup_volume original_volume reduction_factor sugar_added cups_per_quart = 3 := by
  sorry

end NUMINAMATH_CALUDE_syrup_volume_proof_l2511_251176


namespace NUMINAMATH_CALUDE_sum_of_dimensions_for_specific_box_l2511_251115

/-- A rectangular box with dimensions A, B, and C -/
structure RectangularBox where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sum of dimensions of a rectangular box -/
def sum_of_dimensions (box : RectangularBox) : ℝ :=
  box.A + box.B + box.C

/-- Theorem: For a rectangular box with given surface areas, the sum of its dimensions is 27.67 -/
theorem sum_of_dimensions_for_specific_box :
  ∃ (box : RectangularBox),
    box.A * box.B = 40 ∧
    box.A * box.C = 90 ∧
    box.B * box.C = 100 ∧
    sum_of_dimensions box = 27.67 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_dimensions_for_specific_box_l2511_251115


namespace NUMINAMATH_CALUDE_two_digit_sum_l2511_251194

theorem two_digit_sum (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → a ≠ 0 → a - b = a * b → 
  10 * a + b + (10 * b + a) = 22 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_l2511_251194


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l2511_251132

theorem real_part_of_reciprocal (z : ℂ) (h1 : z ≠ 1) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 1) :
  (1 / (1 - z)).re = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l2511_251132


namespace NUMINAMATH_CALUDE_cistern_emptying_rate_l2511_251180

-- Define the rates of the pipes
def rate_A : ℚ := 1 / 60
def rate_B : ℚ := 1 / 75
def rate_combined : ℚ := 1 / 50

-- Define the theorem
theorem cistern_emptying_rate :
  ∃ (rate_C : ℚ), 
    rate_A + rate_B - rate_C = rate_combined ∧ 
    rate_C = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_rate_l2511_251180


namespace NUMINAMATH_CALUDE_tundra_electrification_l2511_251130

theorem tundra_electrification (x y : ℝ) : 
  x + y = 1 →                 -- Initial parts sum to 1
  2*x + 0.75*y = 1 →          -- Condition after changes
  0 ≤ x ∧ x ≤ 1 →             -- x is a fraction
  0 ≤ y ∧ y ≤ 1 →             -- y is a fraction
  y = 4/5 :=                  -- Conclusion: non-electrified part was 4/5
by sorry

end NUMINAMATH_CALUDE_tundra_electrification_l2511_251130


namespace NUMINAMATH_CALUDE_frood_game_theorem_l2511_251114

/-- Score for dropping n froods -/
def droppingScore (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Score for eating n froods -/
def eatingScore (n : ℕ) : ℕ := n^2

/-- The least number of froods for which dropping earns more points than eating -/
def leastFroods : ℕ := 21

theorem frood_game_theorem :
  (∀ k < leastFroods, droppingScore k ≤ eatingScore k) ∧
  (droppingScore leastFroods > eatingScore leastFroods) :=
sorry

end NUMINAMATH_CALUDE_frood_game_theorem_l2511_251114


namespace NUMINAMATH_CALUDE_quadratic_equations_one_common_root_l2511_251183

theorem quadratic_equations_one_common_root 
  (a b c d : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0) ↔ 
  ((a*d - b*c)*(c - a) = (b - d)^2 ∧ (a*d - b*c)*(c - a) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_one_common_root_l2511_251183


namespace NUMINAMATH_CALUDE_horner_v2_value_horner_method_correctness_l2511_251177

/-- Horner's method intermediate value -/
def v2 (x : ℝ) : ℝ := 2 * x - 3 * x + 1

/-- The polynomial function -/
def f (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

theorem horner_v2_value :
  v2 (-1) = 6 :=
by sorry

theorem horner_method_correctness (x : ℝ) :
  f x = (((2*x - 3)*x + 1)*x + 2)*x + 1 :=
by sorry

#check horner_v2_value
#check horner_method_correctness

end NUMINAMATH_CALUDE_horner_v2_value_horner_method_correctness_l2511_251177


namespace NUMINAMATH_CALUDE_complex_equality_l2511_251136

theorem complex_equality (z : ℂ) : z = -1 + I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z + 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2511_251136


namespace NUMINAMATH_CALUDE_cos_300_degrees_l2511_251188

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l2511_251188


namespace NUMINAMATH_CALUDE_invalid_set_l2511_251198

/-- A set of three positive real numbers representing the lengths of external diagonals of a right regular prism. -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- The condition for a valid set of external diagonals. -/
def is_valid (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ 
  d.b^2 + d.c^2 > d.a^2 ∧ 
  d.c^2 + d.a^2 > d.b^2

/-- The set {5,7,9} is not a valid set of external diagonals for a right regular prism. -/
theorem invalid_set : ¬ is_valid ⟨5, 7, 9, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_invalid_set_l2511_251198


namespace NUMINAMATH_CALUDE_power_of_prime_exponent_l2511_251193

theorem power_of_prime_exponent (x y p n k : ℕ) 
  (h_n_gt_1 : n > 1)
  (h_n_odd : Odd n)
  (h_p_prime : Nat.Prime p)
  (h_p_odd : Odd p)
  (h_eq : x^n + y^n = p^k) :
  ∃ m : ℕ, n = p^m :=
sorry

end NUMINAMATH_CALUDE_power_of_prime_exponent_l2511_251193


namespace NUMINAMATH_CALUDE_total_breakfast_cost_l2511_251150

def breakfast_cost (muffin_price fruit_cup_price francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins * muffin_price + francis_fruit_cups * fruit_cup_price) +
  (kiera_muffins * muffin_price + kiera_fruit_cups * fruit_cup_price)

theorem total_breakfast_cost :
  breakfast_cost 2 3 2 2 2 1 = 17 :=
by sorry

end NUMINAMATH_CALUDE_total_breakfast_cost_l2511_251150


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2511_251143

theorem polynomial_divisibility (x : ℝ) (m : ℝ) : 
  (5 * x^3 - 3 * x^2 - 12 * x + m) % (x - 4) = 0 ↔ m = -224 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2511_251143


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l2511_251106

theorem line_circle_intersection_range (m : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m ∧ 
    x₁^2 + y₁^2 + 4*x₁ + 2 = 0 ∧ 
    x₂^2 + y₂^2 + 4*x₂ + 2 = 0) → 
  0 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l2511_251106


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l2511_251137

/-- A square with side length 10 -/
structure Square :=
  (side : ℝ)
  (is_ten : side = 10)

/-- A circle passing through two opposite vertices of the square and tangent to one side -/
structure Circle (s : Square) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through_vertices : True)  -- This is a placeholder for the actual condition
  (tangent_to_side : True)  -- This is a placeholder for the actual condition

/-- The theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five (s : Square) (c : Circle s) : c.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l2511_251137


namespace NUMINAMATH_CALUDE_cube_number_sum_l2511_251139

theorem cube_number_sum : 
  ∀ (n : ℤ),
  (∀ (i : Fin 6), i.val < 6 → ∃ (face : ℤ), face = n + i.val) →
  (∃ (s : ℤ), s % 2 = 1 ∧ 
    (n + (n + 5) = s) ∧ 
    ((n + 1) + (n + 4) = s) ∧ 
    ((n + 2) + (n + 3) = s)) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 27) :=
by sorry


end NUMINAMATH_CALUDE_cube_number_sum_l2511_251139


namespace NUMINAMATH_CALUDE_count_numbers_3000_l2511_251197

/-- Returns true if the given number contains the digit '2' in its base-10 representation -/
def contains_two (n : ℕ) : Bool :=
  sorry

/-- Returns the count of numbers less than or equal to n that contain '2' and are divisible by 3 -/
def count_numbers (n : ℕ) : ℕ :=
  sorry

theorem count_numbers_3000 : count_numbers 3000 = 384 :=
  sorry

end NUMINAMATH_CALUDE_count_numbers_3000_l2511_251197


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l2511_251126

theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ ∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = x := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l2511_251126


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2511_251125

/-- Theorem about a triangle ABC with specific properties -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < C → C < Real.pi / 2 →
  a * c * Real.cos B - b * c * Real.cos A = 3 * b^2 →
  c = Real.sqrt 11 →
  Real.sin C = 2 * Real.sqrt 2 / 3 →
  (a / b = 2) ∧ (1/2 * a * b * Real.sin C = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2511_251125


namespace NUMINAMATH_CALUDE_rectangle_dimension_solution_l2511_251161

theorem rectangle_dimension_solution (x : ℝ) : 
  (3*x - 5 > 0) → (x + 7 > 0) → ((3*x - 5) * (x + 7) = 14*x - 35) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_solution_l2511_251161


namespace NUMINAMATH_CALUDE_distribute_5_3_l2511_251131

/-- The number of ways to distribute n identical objects into k identical containers,
    where at least one container must remain empty -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 26 ways to distribute 5 identical objects into 3 identical containers,
    where at least one container must remain empty -/
theorem distribute_5_3 : distribute 5 3 = 26 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2511_251131


namespace NUMINAMATH_CALUDE_octal_subtraction_theorem_l2511_251113

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Conversion from octal to decimal --/
def from_octal (n : OctalNumber) : ℕ :=
  sorry

theorem octal_subtraction_theorem :
  octal_sub (to_octal 52) (to_octal 27) = to_octal 25 :=
sorry

end NUMINAMATH_CALUDE_octal_subtraction_theorem_l2511_251113


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2511_251108

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2511_251108


namespace NUMINAMATH_CALUDE_fourth_part_diminished_l2511_251134

theorem fourth_part_diminished (x : ℝ) (y : ℝ) (h : x = 280) (h2 : x/5 + 7 = x/4 - y) : y = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_part_diminished_l2511_251134


namespace NUMINAMATH_CALUDE_similar_triangles_area_l2511_251110

-- Define the triangles and their properties
def Triangle : Type := Unit

def similar (t1 t2 : Triangle) : Prop := sorry

def similarityRatio (t1 t2 : Triangle) : ℚ := sorry

def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem similar_triangles_area 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_ratio : similarityRatio ABC DEF = 1 / 2) 
  (h_area_ABC : area ABC = 3) : 
  area DEF = 12 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_area_l2511_251110


namespace NUMINAMATH_CALUDE_binary_string_power_of_two_sum_l2511_251145

/-- A binary string is represented as a list of booleans, where true represents 1 and false represents 0. -/
def BinaryString := List Bool

/-- Count the number of ones in a binary string. -/
def countOnes (s : BinaryString) : Nat :=
  s.filter id |>.length

/-- Represents a way of inserting plus signs into a binary string. 
    true means "insert a plus sign after this digit", false means "don't insert". -/
def PlusInsertion := List Bool

/-- Compute the sum of a binary string with plus signs inserted according to a PlusInsertion. -/
def computeSum (s : BinaryString) (insertion : PlusInsertion) : Nat :=
  sorry  -- Implementation details omitted for brevity

/-- Check if a number is a power of two. -/
def isPowerOfTwo (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

/-- The main theorem statement. -/
theorem binary_string_power_of_two_sum 
  (s : BinaryString) 
  (h : countOnes s ≥ 2017) : 
  ∃ insertion : PlusInsertion, isPowerOfTwo (computeSum s insertion) := by
  sorry


end NUMINAMATH_CALUDE_binary_string_power_of_two_sum_l2511_251145


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2511_251141

theorem equal_roots_quadratic (k C : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + C = 0) →
  (∃! r : ℝ, 2 * x^2 + 4 * x + C = 0) →
  C = 2 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2511_251141


namespace NUMINAMATH_CALUDE_expression_simplification_l2511_251185

theorem expression_simplification (a : ℝ) (h : a^2 - a - (7/2) = 0) :
  a^2 - (a - (2*a)/(a+1)) / ((a^2 - 2*a + 1)/(a^2 - 1)) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2511_251185


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2511_251171

theorem polynomial_factorization (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b)^2 * (b - c) * (c - a) * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2511_251171


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_5_l2511_251135

-- Define the function representing the sum of distances
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem statement
theorem empty_solution_set_implies_a_leq_5 :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ a) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_5_l2511_251135


namespace NUMINAMATH_CALUDE_salt_concentration_increase_l2511_251190

/-- Given a 100 kg solution with 10% salt concentration and adding 20 kg of pure salt,
    the final salt concentration is 25%. -/
theorem salt_concentration_increase (initial_solution : ℝ) (initial_concentration : ℝ) 
    (added_salt : ℝ) (final_concentration : ℝ) : 
    initial_solution = 100 →
    initial_concentration = 0.1 →
    added_salt = 20 →
    final_concentration = (initial_solution * initial_concentration + added_salt) / 
                          (initial_solution + added_salt) →
    final_concentration = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_salt_concentration_increase_l2511_251190


namespace NUMINAMATH_CALUDE_max_price_correct_optimal_price_correct_max_profit_correct_l2511_251124

/-- Represents the beverage pricing and sales model for a food company. -/
structure BeverageModel where
  initial_price : ℝ
  initial_cost : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  marketing_cost : ℝ → ℝ
  sales_decrease : ℝ → ℝ

/-- The maximum price that ensures the total profit is not lower than the initial profit. -/
def max_price (model : BeverageModel) : ℝ :=
  model.initial_price + 5

/-- The price that maximizes the total profit under the new marketing strategy. -/
def optimal_price (model : BeverageModel) : ℝ := 19

/-- The maximum total profit under the new marketing strategy. -/
def max_profit (model : BeverageModel) : ℝ := 45.45

/-- Theorem stating the correctness of the maximum price. -/
theorem max_price_correct (model : BeverageModel) 
  (h1 : model.initial_price = 15)
  (h2 : model.initial_cost = 10)
  (h3 : model.initial_sales = 80000)
  (h4 : model.price_sensitivity = 8000) :
  max_price model = 20 := by sorry

/-- Theorem stating the correctness of the optimal price for maximum profit. -/
theorem optimal_price_correct (model : BeverageModel) 
  (h1 : model.initial_cost = 10)
  (h2 : ∀ x, x ≥ 16 → model.marketing_cost x = (33/4) * (x - 16))
  (h3 : ∀ x, model.sales_decrease x = 0.8 / ((x - 15)^2)) :
  optimal_price model = 19 := by sorry

/-- Theorem stating the correctness of the maximum total profit. -/
theorem max_profit_correct (model : BeverageModel) 
  (h1 : model.initial_cost = 10)
  (h2 : ∀ x, x ≥ 16 → model.marketing_cost x = (33/4) * (x - 16))
  (h3 : ∀ x, model.sales_decrease x = 0.8 / ((x - 15)^2)) :
  max_profit model = 45.45 := by sorry

end NUMINAMATH_CALUDE_max_price_correct_optimal_price_correct_max_profit_correct_l2511_251124


namespace NUMINAMATH_CALUDE_power_of_two_plus_five_l2511_251162

theorem power_of_two_plus_five : 2^5 + 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_five_l2511_251162


namespace NUMINAMATH_CALUDE_min_sum_distances_l2511_251169

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [OrderedAddCommGroup α]

def points (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α) : Prop :=
  P₁ < P₂ ∧ P₂ < P₃ ∧ P₃ < P₄ ∧ P₄ < P₅ ∧ P₅ < P₆ ∧ P₆ < P₇

def distance (x y : α) : α := abs (x - y)

def sum_distances (P : α) (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α) : α :=
  distance P P₁ + distance P P₂ + distance P P₃ + distance P P₄ +
  distance P P₅ + distance P P₆ + distance P P₇

theorem min_sum_distances
  (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α)
  (h : points P₁ P₂ P₃ P₄ P₅ P₆ P₇) :
  ∀ P, sum_distances P P₁ P₂ P₃ P₄ P₅ P₆ P₇ ≥ sum_distances P₄ P₁ P₂ P₃ P₄ P₅ P₆ P₇ ∧
  (sum_distances P P₁ P₂ P₃ P₄ P₅ P₆ P₇ = sum_distances P₄ P₁ P₂ P₃ P₄ P₅ P₆ P₇ ↔ P = P₄) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_l2511_251169


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2511_251116

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 10 → x * y = 200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2511_251116


namespace NUMINAMATH_CALUDE_nap_period_days_l2511_251133

-- Define the given conditions
def naps_per_week : ℕ := 3
def hours_per_nap : ℕ := 2
def total_nap_hours : ℕ := 60

-- Define the theorem
theorem nap_period_days : 
  (total_nap_hours / hours_per_nap / naps_per_week) * 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nap_period_days_l2511_251133


namespace NUMINAMATH_CALUDE_basement_water_pumping_time_l2511_251164

/-- Proves that pumping out water from a flooded basement takes 450 minutes given specific conditions. -/
theorem basement_water_pumping_time : ∀ (length width depth : ℝ) 
  (num_pumps pump_rate : ℕ) (water_density : ℝ),
  length = 30 →
  width = 40 →
  depth = 24 / 12 →
  num_pumps = 4 →
  pump_rate = 10 →
  water_density = 7.5 →
  (length * width * depth * water_density) / (num_pumps * pump_rate) = 450 := by
  sorry

end NUMINAMATH_CALUDE_basement_water_pumping_time_l2511_251164


namespace NUMINAMATH_CALUDE_bus_station_arrangement_count_l2511_251117

/-- The number of seats in the bus station -/
def num_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- The number of ways to arrange passengers with the required consecutive empty seats -/
def arrangement_count : ℕ := 480

/-- Theorem stating that the number of ways to arrange passengers with the required consecutive empty seats is correct -/
theorem bus_station_arrangement_count :
  (num_seats : ℕ) = 10 →
  (num_passengers : ℕ) = 4 →
  (consecutive_empty_seats : ℕ) = 5 →
  (arrangement_count : ℕ) = 480 := by
  sorry

end NUMINAMATH_CALUDE_bus_station_arrangement_count_l2511_251117


namespace NUMINAMATH_CALUDE_teds_age_l2511_251103

theorem teds_age (t s : ℝ) 
  (h1 : t = 3 * s - 10)  -- Ted's age is 10 years less than three times Sally's age
  (h2 : t + s = 60)      -- The sum of their ages is 60
  : t = 42.5 :=          -- Ted's age is 42.5
by sorry

end NUMINAMATH_CALUDE_teds_age_l2511_251103


namespace NUMINAMATH_CALUDE_roses_per_bouquet_l2511_251156

/-- Proves that the number of roses in each bouquet is 12 given the problem conditions -/
theorem roses_per_bouquet (total_bouquets : ℕ) (rose_bouquets : ℕ) (daisy_bouquets : ℕ) 
  (total_flowers : ℕ) (daisies_per_bouquet : ℕ) :
  total_bouquets = 20 →
  rose_bouquets = 10 →
  daisy_bouquets = 10 →
  rose_bouquets + daisy_bouquets = total_bouquets →
  total_flowers = 190 →
  daisies_per_bouquet = 7 →
  (total_flowers - daisy_bouquets * daisies_per_bouquet) / rose_bouquets = 12 :=
by sorry

end NUMINAMATH_CALUDE_roses_per_bouquet_l2511_251156


namespace NUMINAMATH_CALUDE_sufficient_condition_for_square_inequality_l2511_251178

theorem sufficient_condition_for_square_inequality (a b : ℝ) :
  a > b ∧ b > 0 → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_square_inequality_l2511_251178


namespace NUMINAMATH_CALUDE_milk_water_mixture_l2511_251109

/-- Given a mixture of milk and water with an initial ratio of 6:3 and a final ratio of 6:5 after
    adding 10 liters of water, the original quantity of milk is 30 liters. -/
theorem milk_water_mixture (milk : ℝ) (water : ℝ) : 
  milk / water = 6 / 3 →
  milk / (water + 10) = 6 / 5 →
  milk = 30 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_mixture_l2511_251109


namespace NUMINAMATH_CALUDE_finite_square_solutions_l2511_251121

theorem finite_square_solutions (a b : ℤ) (h : ¬ ∃ k : ℤ, b = k^2) :
  { x : ℤ | ∃ y : ℤ, x^2 + a*x + b = y^2 }.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_square_solutions_l2511_251121


namespace NUMINAMATH_CALUDE_negation_of_existential_negation_of_specific_existential_l2511_251186

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_specific_existential :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_negation_of_specific_existential_l2511_251186


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l2511_251163

theorem min_value_cubic_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 + y^3 - 5*x*y ≥ -125/27 ∧ ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x^3 + y^3 - 5*x*y = -125/27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l2511_251163


namespace NUMINAMATH_CALUDE_sequence_general_term_l2511_251104

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 6) 
  (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5^n) : 
  ∀ n : ℕ, n ≥ 1 → a n = 5^n + 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2511_251104


namespace NUMINAMATH_CALUDE_baby_age_at_weight_7200_l2511_251102

/-- The relationship between a baby's weight and age -/
def weight_age_relation (a : ℝ) (x : ℝ) : ℝ := a + 800 * x

/-- The theorem stating the age of the baby when their weight is 7200 grams -/
theorem baby_age_at_weight_7200 (a : ℝ) (x : ℝ) 
  (h1 : a = 3200) -- The baby's weight at birth is 3200 grams
  (h2 : weight_age_relation a x = 7200) -- The baby's weight is 7200 grams
  : x = 5 := by
  sorry

#check baby_age_at_weight_7200

end NUMINAMATH_CALUDE_baby_age_at_weight_7200_l2511_251102


namespace NUMINAMATH_CALUDE_inscribed_equiangular_triangle_exists_l2511_251173

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the concept of being inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

-- Define the concept of two triangles being equiangular
def isEquiangular (t1 t2 : Triangle) : Prop :=
  sorry

theorem inscribed_equiangular_triangle_exists 
  (c : Circle) (reference : Triangle) : 
  ∃ (t : Triangle), isInscribed t c ∧ isEquiangular t reference := by
  sorry

end NUMINAMATH_CALUDE_inscribed_equiangular_triangle_exists_l2511_251173


namespace NUMINAMATH_CALUDE_lukes_fishing_days_l2511_251129

/-- Proves the number of days Luke catches fish given the conditions -/
theorem lukes_fishing_days (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ) : 
  fish_per_day = 2 → 
  fillets_per_fish = 2 → 
  total_fillets = 120 → 
  (total_fillets / fillets_per_fish) / fish_per_day = 30 := by
  sorry

end NUMINAMATH_CALUDE_lukes_fishing_days_l2511_251129


namespace NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l2511_251140

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem cubic_inches_in_cubic_foot :
  (inches_per_foot ^ 3 : ℕ) = 1728 :=
sorry

end NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l2511_251140


namespace NUMINAMATH_CALUDE_fishing_ratio_l2511_251166

/-- Given that Tom caught 16 trout and Melanie caught 8 trout, 
    prove that the ratio of Tom's catch to Melanie's catch is 2. -/
theorem fishing_ratio (tom_catch melanie_catch : ℕ) 
  (h1 : tom_catch = 16) (h2 : melanie_catch = 8) : 
  (tom_catch : ℚ) / melanie_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_fishing_ratio_l2511_251166


namespace NUMINAMATH_CALUDE_andy_late_time_l2511_251147

def school_start_time : Nat := 8 * 60  -- 8:00 AM in minutes
def normal_travel_time : Nat := 30
def red_light_delay : Nat := 3
def num_red_lights : Nat := 4
def construction_delay : Nat := 10
def departure_time : Nat := 7 * 60 + 15  -- 7:15 AM in minutes

def total_delay : Nat := red_light_delay * num_red_lights + construction_delay

def actual_travel_time : Nat := normal_travel_time + total_delay

def arrival_time : Nat := departure_time + actual_travel_time

theorem andy_late_time : arrival_time - school_start_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_andy_late_time_l2511_251147


namespace NUMINAMATH_CALUDE_simplify_expression_l2511_251142

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2511_251142


namespace NUMINAMATH_CALUDE_no_valid_a_l2511_251159

theorem no_valid_a : ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_a_l2511_251159


namespace NUMINAMATH_CALUDE_inequality_proof_l2511_251182

theorem inequality_proof (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2511_251182


namespace NUMINAMATH_CALUDE_apple_bag_price_apple_bag_price_is_8_l2511_251168

/-- Calculates the selling price of one bag of apples given the harvest and sales information. -/
theorem apple_bag_price (total_harvest : ℕ) (juice_amount : ℕ) (restaurant_amount : ℕ) 
  (bag_size : ℕ) (total_revenue : ℕ) : ℕ :=
  let remaining := total_harvest - juice_amount - restaurant_amount
  let num_bags := remaining / bag_size
  total_revenue / num_bags

/-- Proves that the selling price of one bag of apples is $8 given the specific harvest and sales information. -/
theorem apple_bag_price_is_8 :
  apple_bag_price 405 90 60 5 408 = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_bag_price_apple_bag_price_is_8_l2511_251168


namespace NUMINAMATH_CALUDE_washing_machines_removed_per_box_l2511_251184

theorem washing_machines_removed_per_box :
  let num_crates : ℕ := 10
  let boxes_per_crate : ℕ := 6
  let initial_machines_per_box : ℕ := 4
  let total_machines_removed : ℕ := 60
  let total_boxes : ℕ := num_crates * boxes_per_crate
  let machines_removed_per_box : ℕ := total_machines_removed / total_boxes
  machines_removed_per_box = 1 := by
  sorry

end NUMINAMATH_CALUDE_washing_machines_removed_per_box_l2511_251184
