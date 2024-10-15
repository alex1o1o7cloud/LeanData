import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_properties_l1064_106455

def z : ℂ := 3 - 4 * Complex.I

theorem complex_number_properties : 
  (Complex.abs z = 5) ∧ 
  (∃ (y : ℝ), z - 3 = y * Complex.I) ∧
  (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1064_106455


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1064_106404

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ StrictMono f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1064_106404


namespace NUMINAMATH_CALUDE_mrs_hilt_impressed_fans_l1064_106477

/-- The number of sets of bleachers -/
def num_bleachers : ℕ := 3

/-- The number of fans on each set of bleachers -/
def fans_per_bleacher : ℕ := 812

/-- The total number of fans Mrs. Hilt impressed -/
def total_fans : ℕ := num_bleachers * fans_per_bleacher

theorem mrs_hilt_impressed_fans : total_fans = 2436 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_impressed_fans_l1064_106477


namespace NUMINAMATH_CALUDE_binomial_plus_three_l1064_106440

theorem binomial_plus_three : (Nat.choose 13 11) + 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_three_l1064_106440


namespace NUMINAMATH_CALUDE_shadow_length_indeterminate_l1064_106456

/-- Represents a person's shadow length under different light sources -/
structure Shadow where
  sunLength : ℝ
  streetLightLength : ℝ → ℝ

/-- The theorem states that given Xiao Ming's shadow is longer than Xiao Qiang's under sunlight,
    it's impossible to determine their relative shadow lengths under a streetlight -/
theorem shadow_length_indeterminate 
  (xiaoming xioaqiang : Shadow)
  (h_sun : xiaoming.sunLength > xioaqiang.sunLength) :
  ∃ (d₁ d₂ : ℝ), 
    xiaoming.streetLightLength d₁ > xioaqiang.streetLightLength d₂ ∧
    ∃ (d₃ d₄ : ℝ), 
      xiaoming.streetLightLength d₃ < xioaqiang.streetLightLength d₄ :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_indeterminate_l1064_106456


namespace NUMINAMATH_CALUDE_triangle_area_l1064_106498

/-- The area of a triangle with base 12 and height 15 is 90 -/
theorem triangle_area (base height area : ℝ) : 
  base = 12 → height = 15 → area = (1/2) * base * height → area = 90 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1064_106498


namespace NUMINAMATH_CALUDE_luke_garage_sale_games_l1064_106483

/-- Represents the number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := sorry

/-- Represents the number of games Luke bought from a friend -/
def games_from_friend : ℕ := 2

/-- Represents the number of games that didn't work -/
def non_working_games : ℕ := 2

/-- Represents the number of good games Luke ended up with -/
def good_games : ℕ := 2

theorem luke_garage_sale_games :
  games_from_garage_sale = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_garage_sale_games_l1064_106483


namespace NUMINAMATH_CALUDE_credit_card_more_beneficial_l1064_106451

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 8000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.0075

/-- Represents the monthly interest rate on the debit card -/
def debit_interest_rate : ℝ := 0.005

/-- Calculates the total income when using the credit card -/
def credit_card_income (amount : ℝ) : ℝ :=
  amount * credit_cashback_rate + amount * debit_interest_rate

/-- Calculates the total income when using the debit card -/
def debit_card_income (amount : ℝ) : ℝ :=
  amount * debit_cashback_rate

/-- Theorem stating that using the credit card is more beneficial -/
theorem credit_card_more_beneficial :
  credit_card_income purchase_amount > debit_card_income purchase_amount :=
sorry


end NUMINAMATH_CALUDE_credit_card_more_beneficial_l1064_106451


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1064_106463

def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def c : Fin 2 → ℝ := ![2, -4]

theorem vector_sum_magnitude : 
  ∃ (x y : ℝ), 
    (∀ i : Fin 2, (a x) i * c i = 0) ∧ 
    (∃ (k : ℝ), ∀ i : Fin 2, (b y) i = k * c i) →
    Real.sqrt ((a x 0 + b y 0)^2 + (a x 1 + b y 1)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1064_106463


namespace NUMINAMATH_CALUDE_order_of_abc_l1064_106486

/-- Given a = 0.1e^(0.1), b = 1/9, and c = -ln(0.9), prove that c < a < b -/
theorem order_of_abc (a b c : ℝ) (ha : a = 0.1 * Real.exp 0.1) (hb : b = 1/9) (hc : c = -Real.log 0.9) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1064_106486


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1064_106432

/-- The number of diagonals in a convex polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: The number of diagonals in a convex polygon with 30 sides is 375 -/
theorem diagonals_30_sided_polygon :
  numDiagonals 30 = 375 := by sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1064_106432


namespace NUMINAMATH_CALUDE_convex_bodies_with_coinciding_projections_intersect_l1064_106457

/-- A convex body in 3D space -/
structure ConvexBody3D where
  -- Add necessary fields/axioms for a convex body

/-- Projection of a convex body onto a coordinate plane -/
def projection (body : ConvexBody3D) (plane : Fin 3) : Set (Fin 2 → ℝ) :=
  sorry

/-- Two convex bodies intersect if they have a common point -/
def intersect (body1 body2 : ConvexBody3D) : Prop :=
  sorry

/-- Main theorem: If two convex bodies have coinciding projections on all coordinate planes, 
    then they must intersect -/
theorem convex_bodies_with_coinciding_projections_intersect 
  (body1 body2 : ConvexBody3D) 
  (h : ∀ (plane : Fin 3), projection body1 plane = projection body2 plane) : 
  intersect body1 body2 :=
sorry

end NUMINAMATH_CALUDE_convex_bodies_with_coinciding_projections_intersect_l1064_106457


namespace NUMINAMATH_CALUDE_min_value_problem_l1064_106427

/-- The problem statement -/
theorem min_value_problem (m n : ℝ) : 
  (∃ (x y : ℝ), x + y - 1 = 0 ∧ 3 * x - y - 7 = 0 ∧ m * x + y + n = 0) →
  (m * n > 0) →
  (∀ k : ℝ, (1 / m + 2 / n ≥ k) → k ≤ 8) ∧ 
  (∃ m₀ n₀ : ℝ, (∃ (x y : ℝ), x + y - 1 = 0 ∧ 3 * x - y - 7 = 0 ∧ m₀ * x + y + n₀ = 0) ∧ 
                (m₀ * n₀ > 0) ∧ 
                (1 / m₀ + 2 / n₀ = 8)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1064_106427


namespace NUMINAMATH_CALUDE_star_property_l1064_106473

def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a, b) (c, d) ↦ (a - c, b + d)

theorem star_property :
  ∀ x y : ℤ, star (x, y) (2, 3) = star (5, 4) (1, 1) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_property_l1064_106473


namespace NUMINAMATH_CALUDE_f_monotonicity_l1064_106464

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- State the theorem
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -1/3 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x y, -1/3 < x ∧ x < y ∧ y < 1 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_l1064_106464


namespace NUMINAMATH_CALUDE_correct_log_values_l1064_106489

/-- The logarithm base 10 function -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Representation of logarithmic values in terms of a, b, and c -/
structure LogValues where
  a : ℝ
  b : ℝ
  c : ℝ

theorem correct_log_values (v : LogValues) :
  (log10 0.021 = 2 * v.a + v.b + v.c - 3) →
  (log10 0.27 = 6 * v.a - 3 * v.b - 2) →
  (log10 2.8 = 1 - 2 * v.a + 2 * v.b - v.c) →
  (log10 3 = 2 * v.a - v.b) →
  (log10 5 = v.a + v.c) →
  (log10 6 = 1 + v.a - v.b - v.c) →
  (log10 8 = 3 - 3 * v.a - 3 * v.c) →
  (log10 9 = 4 * v.a - 2 * v.b) →
  (log10 14 = 1 - v.c + 2 * v.b) →
  (log10 1.5 = 3 * v.a - v.b + v.c - 1) ∧
  (log10 7 = 2 * v.b + v.c) := by
  sorry


end NUMINAMATH_CALUDE_correct_log_values_l1064_106489


namespace NUMINAMATH_CALUDE_work_rate_ratio_l1064_106407

/-- Given three workers with work rates, prove the ratio of combined work rates -/
theorem work_rate_ratio 
  (R₁ R₂ R₃ : ℝ) 
  (h₁ : R₂ + R₃ = 2 * R₁) 
  (h₂ : R₁ + R₃ = 3 * R₂) : 
  (R₁ + R₂) / R₃ = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_work_rate_ratio_l1064_106407


namespace NUMINAMATH_CALUDE_log_2_3_proof_l1064_106470

theorem log_2_3_proof (a b : ℝ) (h1 : a = Real.log 6 / Real.log 2) (h2 : b = Real.log 20 / Real.log 2) :
  Real.log 3 / Real.log 2 = (a - b + 1) / (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_2_3_proof_l1064_106470


namespace NUMINAMATH_CALUDE_not_perpendicular_to_y_axis_can_be_perpendicular_to_x_axis_l1064_106438

/-- A line in the form x = ky + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Predicate for a line being perpendicular to the y-axis -/
def perpendicular_to_y_axis (l : Line) : Prop :=
  ∃ c : ℝ, ∀ y : ℝ, l.k * y + l.b = c

/-- Predicate for a line being perpendicular to the x-axis -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  l.k = 0

/-- Theorem stating that lines in the form x = ky + b cannot be perpendicular to the y-axis -/
theorem not_perpendicular_to_y_axis :
  ¬ ∃ l : Line, perpendicular_to_y_axis l :=
sorry

/-- Theorem stating that lines in the form x = ky + b can be perpendicular to the x-axis -/
theorem can_be_perpendicular_to_x_axis :
  ∃ l : Line, perpendicular_to_x_axis l :=
sorry

end NUMINAMATH_CALUDE_not_perpendicular_to_y_axis_can_be_perpendicular_to_x_axis_l1064_106438


namespace NUMINAMATH_CALUDE_probability_two_male_finalists_l1064_106462

/-- The probability of selecting two male finalists from a group of 7 finalists (3 male, 4 female) -/
theorem probability_two_male_finalists (total : ℕ) (males : ℕ) (females : ℕ) 
  (h_total : total = 7)
  (h_males : males = 3)
  (h_females : females = 4)
  (h_sum : males + females = total) :
  (males.choose 2 : ℚ) / (total.choose 2) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_male_finalists_l1064_106462


namespace NUMINAMATH_CALUDE_max_students_planting_trees_l1064_106454

theorem max_students_planting_trees :
  ∃ (a b : ℕ), 3 * a + 5 * b = 115 ∧
  ∀ (x y : ℕ), 3 * x + 5 * y = 115 → x + y ≤ a + b ∧
  a + b = 37 := by
sorry

end NUMINAMATH_CALUDE_max_students_planting_trees_l1064_106454


namespace NUMINAMATH_CALUDE_sequence_problem_l1064_106446

-- Define arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0) :
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1064_106446


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l1064_106474

theorem quadratic_always_nonnegative_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*m*x + m + 2 ≥ 0) ↔ m ∈ Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l1064_106474


namespace NUMINAMATH_CALUDE_cube_of_negative_double_l1064_106450

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_double_l1064_106450


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_difference_l1064_106410

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  first_term : a 1 = 1
  arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_mean : a 2 ^ 2 = a 1 * a 6

/-- The common difference of the arithmetic sequence is either 0 or 3 -/
theorem arithmetic_sequence_special_difference (seq : ArithmeticSequence) : 
  seq.d = 0 ∨ seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_difference_l1064_106410


namespace NUMINAMATH_CALUDE_ice_cream_preference_l1064_106453

theorem ice_cream_preference (total : ℕ) (vanilla : ℕ) (strawberry : ℕ) (neither : ℕ) 
  (h1 : total = 50)
  (h2 : vanilla = 23)
  (h3 : strawberry = 20)
  (h4 : neither = 14) :
  total - neither - (vanilla + strawberry - (total - neither)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_preference_l1064_106453


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l1064_106424

def calculate_total_bill (hamburger_price : ℚ) (cracker_price : ℚ) (vegetable_price : ℚ) 
  (vegetable_quantity : ℕ) (cheese_price : ℚ) (chicken_price : ℚ) (cereal_price : ℚ) 
  (rewards_discount : ℚ) (meat_cheese_discount : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let discounted_hamburger := hamburger_price * (1 - rewards_discount)
  let discounted_crackers := cracker_price * (1 - rewards_discount)
  let discounted_vegetables := vegetable_price * (1 - rewards_discount) * vegetable_quantity
  let discounted_cheese := cheese_price * (1 - meat_cheese_discount)
  let discounted_chicken := chicken_price * (1 - meat_cheese_discount)
  let subtotal := discounted_hamburger + discounted_crackers + discounted_vegetables + 
                  discounted_cheese + discounted_chicken + cereal_price
  let total := subtotal * (1 + sales_tax_rate)
  total

theorem rays_grocery_bill : 
  calculate_total_bill 5 (7/2) 2 4 (7/2) (13/2) 4 (1/10) (1/20) (7/100) = (3035/100) := by
  sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l1064_106424


namespace NUMINAMATH_CALUDE_empty_solution_set_inequality_l1064_106494

theorem empty_solution_set_inequality (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_inequality_l1064_106494


namespace NUMINAMATH_CALUDE_janet_pill_count_l1064_106469

/-- Calculates the total number of pills Janet takes in a month -/
def total_pills (multivitamins_per_day : ℕ) (calcium_first_half : ℕ) (calcium_second_half : ℕ) : ℕ :=
  let days_in_month := 28
  let days_in_half := 14
  (multivitamins_per_day * days_in_month) + (calcium_first_half * days_in_half) + (calcium_second_half * days_in_half)

/-- Proves that Janet takes 112 pills in a month -/
theorem janet_pill_count : total_pills 2 3 1 = 112 := by
  sorry

end NUMINAMATH_CALUDE_janet_pill_count_l1064_106469


namespace NUMINAMATH_CALUDE_catering_weight_calculation_l1064_106408

/-- Calculates the total weight of catering items for an event --/
theorem catering_weight_calculation (
  silverware_weight : ℕ)
  (silverware_per_setting : ℕ)
  (plate_weight : ℕ)
  (plates_per_setting : ℕ)
  (glass_weight : ℕ)
  (glasses_per_setting : ℕ)
  (decoration_weight : ℕ)
  (num_tables : ℕ)
  (settings_per_table : ℕ)
  (backup_settings : ℕ)
  (decoration_per_table : ℕ)
  (h1 : silverware_weight = 4)
  (h2 : silverware_per_setting = 3)
  (h3 : plate_weight = 12)
  (h4 : plates_per_setting = 2)
  (h5 : glass_weight = 8)
  (h6 : glasses_per_setting = 2)
  (h7 : decoration_weight = 16)
  (h8 : num_tables = 15)
  (h9 : settings_per_table = 8)
  (h10 : backup_settings = 20)
  (h11 : decoration_per_table = 1) :
  (num_tables * settings_per_table + backup_settings) *
    (silverware_weight * silverware_per_setting +
     plate_weight * plates_per_setting +
     glass_weight * glasses_per_setting) +
  num_tables * decoration_weight * decoration_per_table = 7520 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_l1064_106408


namespace NUMINAMATH_CALUDE_perpendicular_parallel_planes_l1064_106460

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_parallel_planes
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel_lines m n)
  (h3 : parallel_planes α β) :
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_planes_l1064_106460


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1064_106459

/-- Given that the equation m/(x-2) + 1 = x/(2-x) has a non-negative solution for x,
    prove that the range of values for m is m ≤ 2 and m ≠ -2. -/
theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 2) + 1 = x / (2 - x)) →
  m ≤ 2 ∧ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1064_106459


namespace NUMINAMATH_CALUDE_uno_card_price_l1064_106481

/-- The original price of an Uno Giant Family Card -/
def original_price : ℝ := 12

/-- The number of cards purchased -/
def num_cards : ℕ := 10

/-- The discount applied to each card -/
def discount : ℝ := 2

/-- The total amount paid -/
def total_paid : ℝ := 100

/-- Theorem stating that the original price satisfies the given conditions -/
theorem uno_card_price : 
  num_cards * (original_price - discount) = total_paid := by
  sorry


end NUMINAMATH_CALUDE_uno_card_price_l1064_106481


namespace NUMINAMATH_CALUDE_exponent_division_l1064_106417

theorem exponent_division (x : ℝ) : x^8 / x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1064_106417


namespace NUMINAMATH_CALUDE_slower_train_speed_theorem_l1064_106458

/-- The speed of the faster train in km/h -/
def faster_train_speed : ℝ := 120

/-- The length of the first train in meters -/
def train_length_1 : ℝ := 500

/-- The length of the second train in meters -/
def train_length_2 : ℝ := 700

/-- The time taken for the trains to cross each other in seconds -/
def crossing_time : ℝ := 19.6347928529354

/-- The speed of the slower train in km/h -/
def slower_train_speed : ℝ := 100

theorem slower_train_speed_theorem :
  let total_length := train_length_1 + train_length_2
  let relative_speed := (slower_train_speed + faster_train_speed) * (1000 / 3600)
  total_length = relative_speed * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_theorem_l1064_106458


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l1064_106487

theorem smallest_integer_bound (a b c d e f : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f →  -- 6 different integers
  (a + b + c + d + e + f) / 6 = 85 →  -- average is 85
  f = 180 →  -- largest is 180
  e = 100 →  -- second largest is 100
  a ≥ -64 :=  -- smallest is not less than -64
by sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l1064_106487


namespace NUMINAMATH_CALUDE_range_of_a_l1064_106426

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a-1)^x < (a-1)^y

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a))) → 
  (∃ a : ℝ, (-1 < a ∧ a ≤ 2) ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1064_106426


namespace NUMINAMATH_CALUDE_mean_of_number_set_l1064_106435

def number_set : List ℝ := [1, 22, 23, 24, 25, 26, 27, 2]

theorem mean_of_number_set :
  (number_set.sum / number_set.length : ℝ) = 18.75 := by sorry

end NUMINAMATH_CALUDE_mean_of_number_set_l1064_106435


namespace NUMINAMATH_CALUDE_probability_cover_both_clubs_l1064_106492

/-- The probability of selecting two students that cover both clubs -/
theorem probability_cover_both_clubs 
  (total_students : Nat) 
  (robotics_members : Nat) 
  (science_members : Nat) 
  (h1 : total_students = 30)
  (h2 : robotics_members = 22)
  (h3 : science_members = 24) :
  (Nat.choose total_students 2 - (Nat.choose (robotics_members + science_members - total_students) 2 + 
   Nat.choose (robotics_members - (robotics_members + science_members - total_students)) 2 + 
   Nat.choose (science_members - (robotics_members + science_members - total_students)) 2)) / 
   Nat.choose total_students 2 = 392 / 435 := by
sorry

end NUMINAMATH_CALUDE_probability_cover_both_clubs_l1064_106492


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1064_106472

theorem divisibility_theorem (a b c : ℕ) (h1 : ∀ (p : ℕ), Nat.Prime p → c % (p^2) ≠ 0) 
  (h2 : (a^2) ∣ (b^2 * c)) : a ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1064_106472


namespace NUMINAMATH_CALUDE_integer_fraction_solutions_l1064_106497

theorem integer_fraction_solutions (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k) ↔
  (∃ n : ℕ+, (a = n ∧ b = 2 * n) ∨ 
             (a = 8 * n ^ 4 - n ∧ b = 2 * n) ∨ 
             (a = 2 * n ∧ b = 1)) :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_solutions_l1064_106497


namespace NUMINAMATH_CALUDE_gumball_difference_l1064_106478

/-- The number of gumballs Carl bought -/
def carl_gumballs : ℕ := 16

/-- The number of gumballs Lewis bought -/
def lewis_gumballs : ℕ := 12

/-- The minimum average number of gumballs -/
def min_average : ℚ := 19

/-- The maximum average number of gumballs -/
def max_average : ℚ := 25

/-- The number of people who bought gumballs -/
def num_people : ℕ := 3

theorem gumball_difference :
  ∃ (min_x max_x : ℕ),
    (∀ x : ℕ, 
      (min_average ≤ (carl_gumballs + lewis_gumballs + x : ℚ) / num_people ∧
       (carl_gumballs + lewis_gumballs + x : ℚ) / num_people ≤ max_average) →
      min_x ≤ x ∧ x ≤ max_x) ∧
    max_x - min_x = 18 := by
  sorry

end NUMINAMATH_CALUDE_gumball_difference_l1064_106478


namespace NUMINAMATH_CALUDE_remainder_property_l1064_106443

theorem remainder_property (N : ℤ) : ∃ (k : ℤ), N = 35 * k + 25 → ∃ (m : ℤ), N = 15 * m + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l1064_106443


namespace NUMINAMATH_CALUDE_counterexamples_count_l1064_106439

/-- Definition of digit sum -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Definition to check if a number has zero as a digit -/
def hasZeroDigit (n : ℕ) : Prop := sorry

/-- Definition of a prime number -/
def isPrime (n : ℕ) : Prop := sorry

theorem counterexamples_count :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n % 2 = 1 ∧ digitSum n = 4 ∧ ¬hasZeroDigit n ∧ ¬isPrime n) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_counterexamples_count_l1064_106439


namespace NUMINAMATH_CALUDE_largest_number_problem_l1064_106437

theorem largest_number_problem (a b c : ℝ) :
  a < b ∧ b < c →
  a + b + c = 82 →
  c - b = 8 →
  b - a = 4 →
  c = 34 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l1064_106437


namespace NUMINAMATH_CALUDE_field_ratio_l1064_106420

theorem field_ratio (field_length : ℝ) (pond_side : ℝ) (pond_area_ratio : ℝ) :
  field_length = 96 →
  pond_side = 8 →
  pond_area_ratio = 1 / 72 →
  (pond_side * pond_side) * (1 / pond_area_ratio) = field_length * (field_length / 2) →
  field_length / (field_length / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l1064_106420


namespace NUMINAMATH_CALUDE_initial_stock_theorem_l1064_106465

/-- Represents the stock management of a bicycle shop over 3 months -/
structure BikeShop :=
  (mountain_weekly_add : Fin 3 → ℕ)
  (road_weekly_add : ℕ)
  (hybrid_weekly_add : Fin 3 → ℕ)
  (mountain_monthly_sell : ℕ)
  (road_monthly_sell : Fin 3 → ℕ)
  (hybrid_monthly_sell : ℕ)
  (helmet_initial : ℕ)
  (helmet_weekly_add : ℕ)
  (helmet_weekly_sell : ℕ)
  (lock_initial : ℕ)
  (lock_weekly_add : ℕ)
  (lock_weekly_sell : ℕ)
  (final_mountain : ℕ)
  (final_road : ℕ)
  (final_hybrid : ℕ)
  (final_helmet : ℕ)
  (final_lock : ℕ)

/-- The theorem stating the initial stock of bicycles -/
theorem initial_stock_theorem (shop : BikeShop) 
  (h_mountain : shop.mountain_weekly_add = ![6, 4, 3])
  (h_road : shop.road_weekly_add = 4)
  (h_hybrid : shop.hybrid_weekly_add = ![2, 2, 3])
  (h_mountain_sell : shop.mountain_monthly_sell = 12)
  (h_road_sell : shop.road_monthly_sell = ![16, 16, 24])
  (h_hybrid_sell : shop.hybrid_monthly_sell = 10)
  (h_helmet : shop.helmet_initial = 100 ∧ shop.helmet_weekly_add = 10 ∧ shop.helmet_weekly_sell = 15)
  (h_lock : shop.lock_initial = 50 ∧ shop.lock_weekly_add = 5 ∧ shop.lock_weekly_sell = 3)
  (h_final : shop.final_mountain = 75 ∧ shop.final_road = 80 ∧ shop.final_hybrid = 45 ∧ 
             shop.final_helmet = 115 ∧ shop.final_lock = 62) :
  ∃ (initial_mountain initial_road initial_hybrid : ℕ),
    initial_mountain = 59 ∧ 
    initial_road = 88 ∧ 
    initial_hybrid = 47 :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_theorem_l1064_106465


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l1064_106476

theorem king_middle_school_teachers (total_students : ℕ) 
  (classes_per_student : ℕ) (classes_per_teacher : ℕ) 
  (students_per_class : ℕ) :
  total_students = 1500 →
  classes_per_student = 6 →
  classes_per_teacher = 5 →
  students_per_class = 25 →
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l1064_106476


namespace NUMINAMATH_CALUDE_heather_shared_blocks_l1064_106452

/-- The number of blocks Heather shared with Jose -/
def blocks_shared (initial final : ℕ) : ℕ := initial - final

/-- Theorem: The number of blocks Heather shared is the difference between her initial and final blocks -/
theorem heather_shared_blocks (initial final : ℕ) (h1 : initial = 86) (h2 : final = 45) :
  blocks_shared initial final = 41 := by
  sorry

end NUMINAMATH_CALUDE_heather_shared_blocks_l1064_106452


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1064_106430

def total_votes : ℕ := 8000
def loss_margin : ℕ := 2400

theorem candidate_vote_percentage :
  ∃ (p : ℚ),
    p * total_votes = (total_votes - loss_margin) / 2 ∧
    p = 35 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l1064_106430


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l1064_106436

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 6

/-- Number of rest days in Al's cycle -/
def al_rest_days : ℕ := 2

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 6

/-- Number of rest days in Barb's cycle -/
def barb_rest_days : ℕ := 1

/-- Total number of days -/
def total_days : ℕ := 1000

/-- The number of days both Al and Barb have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / (al_cycle * barb_cycle / Nat.gcd al_cycle barb_cycle)

theorem coinciding_rest_days_count :
  coinciding_rest_days = 166 :=
by sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l1064_106436


namespace NUMINAMATH_CALUDE_unique_perpendicular_plane_l1064_106496

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Perpendicularity of a plane to a line -/
def isPerpendicular (p : Plane3D) (l : Line3D) : Prop :=
  -- Definition of perpendicularity
  sorry

/-- A plane contains a point -/
def planeContainsPoint (p : Plane3D) (pt : Point3D) : Prop :=
  -- Definition of a plane containing a point
  sorry

theorem unique_perpendicular_plane 
  (M : Point3D) (h : Line3D) : 
  ∃! (p : Plane3D), planeContainsPoint p M ∧ isPerpendicular p h :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_plane_l1064_106496


namespace NUMINAMATH_CALUDE_earl_floor_problem_l1064_106471

theorem earl_floor_problem (total_floors : ℕ) (start_floor : ℕ) 
  (up_first : ℕ) (down : ℕ) (up_second : ℕ) :
  total_floors = 20 →
  start_floor = 1 →
  up_first = 5 →
  down = 2 →
  up_second = 7 →
  total_floors - (start_floor + up_first - down + up_second) = 9 :=
by sorry

end NUMINAMATH_CALUDE_earl_floor_problem_l1064_106471


namespace NUMINAMATH_CALUDE_part1_part2_l1064_106401

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + 3*x - 10 ≤ 0}

-- Define set B for part (1)
def B1 (m : ℝ) : Set ℝ := {x : ℝ | -2*m + 1 ≤ x ∧ x ≤ -m - 1}

-- Define set B for part (2)
def B2 (m : ℝ) : Set ℝ := {x : ℝ | -2*m + 1 ≤ x ∧ x ≤ -m - 1}

-- Theorem for part (1)
theorem part1 : ∀ m : ℝ, (A ∪ B1 m = A) → (2 < m ∧ m ≤ 3) := by sorry

-- Theorem for part (2)
theorem part2 : ∀ m : ℝ, (A ∪ B2 m = A) → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1064_106401


namespace NUMINAMATH_CALUDE_plate_729_circulation_plate_363_circulation_plate_255_circulation_l1064_106405

def is_valid_plate (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 999

def monday_rule (n : ℕ) : Prop := n % 2 = 1

def tuesday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (List.sum digits) ≥ 11

def wednesday_rule (n : ℕ) : Prop := n % 3 = 0

def thursday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (List.sum digits) ≤ 14

def friday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∃ (i j : Fin 3), i ≠ j ∧ digits[i] = digits[j]

def saturday_rule (n : ℕ) : Prop := n < 500

def sunday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≤ 5

theorem plate_729_circulation :
  is_valid_plate 729 ∧
  monday_rule 729 ∧
  tuesday_rule 729 ∧
  wednesday_rule 729 ∧
  ¬thursday_rule 729 ∧
  ¬friday_rule 729 ∧
  ¬saturday_rule 729 ∧
  ¬sunday_rule 729 := by sorry

theorem plate_363_circulation :
  is_valid_plate 363 ∧
  monday_rule 363 ∧
  tuesday_rule 363 ∧
  wednesday_rule 363 ∧
  thursday_rule 363 ∧
  friday_rule 363 ∧
  saturday_rule 363 ∧
  ¬sunday_rule 363 := by sorry

theorem plate_255_circulation :
  is_valid_plate 255 ∧
  monday_rule 255 ∧
  tuesday_rule 255 ∧
  wednesday_rule 255 ∧
  thursday_rule 255 ∧
  friday_rule 255 ∧
  saturday_rule 255 ∧
  sunday_rule 255 := by sorry

end NUMINAMATH_CALUDE_plate_729_circulation_plate_363_circulation_plate_255_circulation_l1064_106405


namespace NUMINAMATH_CALUDE_greatest_third_side_proof_l1064_106406

/-- The greatest integer length for the third side of a triangle with sides 5 and 10 -/
def greatest_third_side : ℕ :=
  14

theorem greatest_third_side_proof :
  ∀ (c : ℕ),
  (c > greatest_third_side → ¬(5 < c + 10 ∧ 10 < c + 5 ∧ c < 5 + 10)) ∧
  (c ≤ greatest_third_side → (5 < c + 10 ∧ 10 < c + 5 ∧ c < 5 + 10)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_side_proof_l1064_106406


namespace NUMINAMATH_CALUDE_isosceles_area_sum_l1064_106467

/-- Represents a right isosceles triangle constructed on a side of a right triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  area : ℝ

/-- Represents a 5-12-13 right triangle with right isosceles triangles on its sides -/
structure TriangleWithIsosceles where
  short_side1 : RightIsoscelesTriangle
  short_side2 : RightIsoscelesTriangle
  hypotenuse : RightIsoscelesTriangle

/-- The theorem to be proved -/
theorem isosceles_area_sum (t : TriangleWithIsosceles) : 
  t.short_side1.side = 5 ∧ 
  t.short_side2.side = 12 ∧ 
  t.hypotenuse.side = 13 ∧
  t.short_side1.area = (1/2) * t.short_side1.side * t.short_side1.side ∧
  t.short_side2.area = (1/2) * t.short_side2.side * t.short_side2.side ∧
  t.hypotenuse.area = (1/2) * t.hypotenuse.side * t.hypotenuse.side →
  t.short_side1.area + t.short_side2.area = t.hypotenuse.area := by
  sorry

end NUMINAMATH_CALUDE_isosceles_area_sum_l1064_106467


namespace NUMINAMATH_CALUDE_circle_ray_angle_l1064_106488

/-- In a circle with twelve evenly spaced rays, where one ray points north,
    the smaller angle between the north-pointing ray and the southeast-pointing ray is 90°. -/
theorem circle_ray_angle (n : ℕ) (θ : ℝ) : 
  n = 12 → θ = 360 / n → θ * 3 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_ray_angle_l1064_106488


namespace NUMINAMATH_CALUDE_candy_distribution_l1064_106418

theorem candy_distribution (n : ℕ) : n > 0 → (100 % n = 0) → (99 % n = 0) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1064_106418


namespace NUMINAMATH_CALUDE_number_of_products_l1064_106490

/-- Given fixed cost, marginal cost, and total cost, prove the number of products. -/
theorem number_of_products (fixed_cost marginal_cost total_cost : ℚ) (n : ℕ) : 
  fixed_cost = 12000 →
  marginal_cost = 200 →
  total_cost = 16000 →
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_products_l1064_106490


namespace NUMINAMATH_CALUDE_johnnys_hourly_rate_l1064_106482

def hours_worked : ℝ := 8
def total_earnings : ℝ := 26

theorem johnnys_hourly_rate : total_earnings / hours_worked = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_hourly_rate_l1064_106482


namespace NUMINAMATH_CALUDE_negative_negative_one_plus_abs_negative_one_equals_two_l1064_106493

theorem negative_negative_one_plus_abs_negative_one_equals_two : 
  -(-1) + |-1| = 2 := by sorry

end NUMINAMATH_CALUDE_negative_negative_one_plus_abs_negative_one_equals_two_l1064_106493


namespace NUMINAMATH_CALUDE_x_range_theorem_l1064_106414

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + sin x

-- State the theorem
theorem x_range_theorem (h1 : ∀ x ∈ Set.Ioo (-1) 1, deriv f x = 3 + cos x)
                        (h2 : f 0 = 0)
                        (h3 : ∀ x, f (1 - x) + f (1 - x^2) < 0) :
  ∃ x ∈ Set.Ioo 1 (Real.sqrt 2), True :=
sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1064_106414


namespace NUMINAMATH_CALUDE_division_exponent_rule_l1064_106468

theorem division_exponent_rule (x : ℝ) : -6 * x^5 / (2 * x^3) = -3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_division_exponent_rule_l1064_106468


namespace NUMINAMATH_CALUDE_circle_change_l1064_106475

/-- Represents the properties of a circle before and after diameter increase -/
structure CircleChange where
  d : ℝ  -- Initial diameter
  Q : ℝ  -- Increase in circumference

/-- Theorem stating the increase in circumference and area when diameter increases by 2π -/
theorem circle_change (c : CircleChange) :
  c.Q = 2 * Real.pi ^ 2 ∧
  (π * ((c.d + 2 * π) / 2) ^ 2 - π * (c.d / 2) ^ 2) = π ^ 2 * c.d + π ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_change_l1064_106475


namespace NUMINAMATH_CALUDE_pyramid_volume_l1064_106421

/-- The volume of a pyramid with a rectangular base, lateral edges of length l,
    and angles α and β between the lateral edges and adjacent sides of the base. -/
theorem pyramid_volume (l α β : ℝ) (hl : l > 0) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ, V = (4 / 3) * l^3 * Real.cos α * Real.cos β * Real.sqrt (-Real.cos (α + β) * Real.cos (α - β)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1064_106421


namespace NUMINAMATH_CALUDE_morgan_red_pens_l1064_106409

def total_pens : ℕ := 168
def blue_pens : ℕ := 45
def black_pens : ℕ := 58

def red_pens : ℕ := total_pens - (blue_pens + black_pens)

theorem morgan_red_pens : red_pens = 65 := by sorry

end NUMINAMATH_CALUDE_morgan_red_pens_l1064_106409


namespace NUMINAMATH_CALUDE_function_through_points_l1064_106495

/-- Given a function f(x) = a^x - k that passes through (1, 3) and (0, 2), 
    prove that f(x) = 2^x + 1 -/
theorem function_through_points 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h1 : ∀ x, f x = a^x - k) 
  (h2 : f 1 = 3) 
  (h3 : f 0 = 2) : 
  ∀ x, f x = 2^x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_through_points_l1064_106495


namespace NUMINAMATH_CALUDE_equal_area_triangles_l1064_106429

theorem equal_area_triangles (b c : ℝ) (h₁ : b > 0) (h₂ : c > 0) :
  let k : ℝ := (Real.sqrt 5 - 1) * c / 2
  let l : ℝ := (Real.sqrt 5 - 1) * b / 2
  let area_ABK : ℝ := b * k / 2
  let area_AKL : ℝ := l * c / 2
  let area_ADL : ℝ := (b * c - k * l) / 2
  area_ABK = area_AKL ∧ area_AKL = area_ADL := by sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l1064_106429


namespace NUMINAMATH_CALUDE_unique_function_property_l1064_106466

-- Define the property for the function
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

-- Define that the function is not identically zero
def not_zero_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ 0

-- Theorem statement
theorem unique_function_property :
  ∀ f : ℝ → ℝ, satisfies_property f → not_zero_function f →
  ∀ x : ℝ, f x = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_property_l1064_106466


namespace NUMINAMATH_CALUDE_gcd_of_product_3000_not_15_l1064_106449

theorem gcd_of_product_3000_not_15 (a b c : ℕ+) : 
  a * b * c = 3000 → Nat.gcd (a:ℕ) (Nat.gcd (b:ℕ) (c:ℕ)) ≠ 15 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_product_3000_not_15_l1064_106449


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1064_106419

theorem complex_equation_solution (z : ℂ) (m n : ℝ) : 
  (Complex.abs (1 - z) + z = 10 - 3 * Complex.I) →
  (z = 5 - 3 * Complex.I) ∧
  (z^2 + m * z + n = 1 - 3 * Complex.I) →
  (m = 14 ∧ n = -103) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1064_106419


namespace NUMINAMATH_CALUDE_grants_age_l1064_106444

theorem grants_age (hospital_age : ℕ) (grant_age : ℕ) : hospital_age = 40 →
  grant_age + 5 = (2 / 3) * (hospital_age + 5) →
  grant_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_grants_age_l1064_106444


namespace NUMINAMATH_CALUDE_average_of_numbers_l1064_106441

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

theorem average_of_numbers : (numbers.sum / numbers.length : ℚ) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1064_106441


namespace NUMINAMATH_CALUDE_proposition_analysis_l1064_106434

theorem proposition_analysis :
  let P : ℝ → ℝ → Prop := λ a b => a^2 + b^2 = 0
  let Q : ℝ → ℝ → Prop := λ a b => a^2 - b^2 = 0
  let contrapositive : Prop := ∀ a b : ℝ, ¬(Q a b) → ¬(P a b)
  let inverse : Prop := ∀ a b : ℝ, ¬(P a b) → ¬(Q a b)
  let converse : Prop := ∀ a b : ℝ, Q a b → P a b
  (contrapositive ∧ ¬inverse ∧ ¬converse) ∨
  (¬contrapositive ∧ inverse ∧ ¬converse) ∨
  (¬contrapositive ∧ ¬inverse ∧ converse) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l1064_106434


namespace NUMINAMATH_CALUDE_function_inequality_l1064_106423

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo 0 (π/2), tan x * deriv f x < f x) : 
  f (π/6) * sin 1 > (1/2) * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1064_106423


namespace NUMINAMATH_CALUDE_expression_evaluation_l1064_106491

theorem expression_evaluation : (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1064_106491


namespace NUMINAMATH_CALUDE_leftover_coin_value_l1064_106403

def quarters_per_roll : ℕ := 40
def dimes_per_roll : ℕ := 50
def half_dollars_per_roll : ℕ := 20

def james_quarters : ℕ := 120
def james_dimes : ℕ := 200
def james_half_dollars : ℕ := 90

def lindsay_quarters : ℕ := 150
def lindsay_dimes : ℕ := 310
def lindsay_half_dollars : ℕ := 160

def total_quarters : ℕ := james_quarters + lindsay_quarters
def total_dimes : ℕ := james_dimes + lindsay_dimes
def total_half_dollars : ℕ := james_half_dollars + lindsay_half_dollars

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll
def leftover_half_dollars : ℕ := total_half_dollars % half_dollars_per_roll

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.1
def half_dollar_value : ℚ := 0.5

theorem leftover_coin_value :
  (leftover_quarters : ℚ) * quarter_value +
  (leftover_dimes : ℚ) * dime_value +
  (leftover_half_dollars : ℚ) * half_dollar_value = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l1064_106403


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1064_106412

def satisfies_remainder_conditions (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 ∧
  n % 8 = 7 ∧ n % 9 = 8 ∧ n % 10 = 9 ∧ n % 11 = 10 ∧ n % 12 = 11

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → n % m ≠ 0

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ),
    n = 10079 ∧
    satisfies_remainder_conditions n ∧
    ¬∃ (m : ℕ), m * m = n ∧
    is_prime (sum_of_digits n) ∧
    n > 10000 ∧
    ∀ (k : ℕ), 10000 < k ∧ k < n →
      ¬(satisfies_remainder_conditions k ∧
        ¬∃ (m : ℕ), m * m = k ∧
        is_prime (sum_of_digits k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1064_106412


namespace NUMINAMATH_CALUDE_restaurant_hamburgers_l1064_106400

/-- Represents the number of hamburgers in various states --/
structure HamburgerCount where
  served : ℕ
  leftOver : ℕ

/-- Calculates the total number of hamburgers initially made --/
def totalHamburgers (h : HamburgerCount) : ℕ :=
  h.served + h.leftOver

/-- The theorem stating that for the given values, the total hamburgers is 9 --/
theorem restaurant_hamburgers :
  let h : HamburgerCount := { served := 3, leftOver := 6 }
  totalHamburgers h = 9 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_hamburgers_l1064_106400


namespace NUMINAMATH_CALUDE_max_value_of_f_l1064_106428

-- Define the function
def f (x : ℝ) : ℝ := -4 * x^2 + 10

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1064_106428


namespace NUMINAMATH_CALUDE_problem_solution_l1064_106447

theorem problem_solution : 
  (Real.sqrt 75 + Real.sqrt 27 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 8 * Real.sqrt 3 + Real.sqrt 6) ∧
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) - (Real.sqrt 5 - 1)^2 = 2 * Real.sqrt 5 - 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1064_106447


namespace NUMINAMATH_CALUDE_fish_tank_problem_l1064_106461

theorem fish_tank_problem (initial_fish : ℕ) : 
  (initial_fish - 4 = 8) → (initial_fish + 8 = 20) := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l1064_106461


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1064_106416

theorem completing_square_equivalence :
  ∀ x : ℝ, 2 * x^2 + 4 * x - 3 = 0 ↔ (x + 1)^2 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1064_106416


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l1064_106411

/-- Calculate the total interest earned on an investment with compound interest -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Theorem: The total interest earned on $2,000 at 8% annual interest rate after 5 years is approximately $938.66 -/
theorem investment_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let years : ℕ := 5
  abs (totalInterestEarned principal rate years - 938.66) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l1064_106411


namespace NUMINAMATH_CALUDE_group_size_l1064_106422

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  average_increase = 6 → old_weight = 45 → new_weight = 93 → 
  (new_weight - old_weight) / average_increase = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1064_106422


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1064_106402

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|x₁ + 8| = Real.sqrt 256) ∧
  (|x₂ + 8| = Real.sqrt 256) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 32 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1064_106402


namespace NUMINAMATH_CALUDE_constant_function_from_zero_derivative_l1064_106479

theorem constant_function_from_zero_derivative (f : ℝ → ℝ) (h : ∀ x, HasDerivAt f 0 x) :
  ∃ c, ∀ x, f x = c := by sorry

end NUMINAMATH_CALUDE_constant_function_from_zero_derivative_l1064_106479


namespace NUMINAMATH_CALUDE_max_voters_is_five_l1064_106425

/-- Represents a movie rating system where:
    - Scores are integers from 0 to 10
    - The rating is the sum of scores divided by the number of voters
    - At moment T, the rating is an integer
    - After moment T, each new vote decreases the rating by one unit -/
structure MovieRating where
  scores : List ℤ
  rating_at_T : ℤ

/-- The maximum number of viewers who could have voted after moment T
    while maintaining the property that each new vote decreases the rating by one unit -/
def max_voters_after_T (mr : MovieRating) : ℕ :=
  sorry

/-- All scores are between 0 and 10 -/
axiom scores_range (mr : MovieRating) : ∀ s ∈ mr.scores, 0 ≤ s ∧ s ≤ 10

/-- The rating at moment T is the sum of scores divided by the number of voters -/
axiom rating_calculation (mr : MovieRating) :
  mr.rating_at_T = (mr.scores.sum / mr.scores.length : ℤ)

/-- After moment T, each new vote decreases the rating by exactly one unit -/
axiom rating_decrease (mr : MovieRating) (new_score : ℤ) :
  let new_rating := ((mr.scores.sum + new_score) / (mr.scores.length + 1) : ℤ)
  new_rating = mr.rating_at_T - 1

/-- The maximum number of viewers who could have voted after moment T is 5 -/
theorem max_voters_is_five (mr : MovieRating) :
  max_voters_after_T mr = 5 :=
sorry

end NUMINAMATH_CALUDE_max_voters_is_five_l1064_106425


namespace NUMINAMATH_CALUDE_tire_usage_theorem_l1064_106499

/-- Represents the usage of tires on a car --/
structure TireUsage where
  total_tires : ℕ
  road_tires : ℕ
  total_miles : ℕ

/-- Calculates the miles each tire was used given equal usage --/
def miles_per_tire (usage : TireUsage) : ℕ :=
  (usage.total_miles * usage.road_tires) / usage.total_tires

/-- Theorem stating that for the given car configuration and mileage, each tire was used for 33333 miles --/
theorem tire_usage_theorem (usage : TireUsage) 
  (h1 : usage.total_tires = 6)
  (h2 : usage.road_tires = 4)
  (h3 : usage.total_miles = 50000) :
  miles_per_tire usage = 33333 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_theorem_l1064_106499


namespace NUMINAMATH_CALUDE_power_simplification_l1064_106413

theorem power_simplification : 2^6 * 8^3 * 2^12 * 8^6 = 2^45 := by sorry

end NUMINAMATH_CALUDE_power_simplification_l1064_106413


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1064_106442

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 100 ∧ x - y = 8 → x * y = 2484 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1064_106442


namespace NUMINAMATH_CALUDE_mark_kate_difference_l1064_106433

/-- Project hours charged by four people --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ
  sam : ℕ

/-- Conditions for the project hours --/
def valid_project_hours (h : ProjectHours) : Prop :=
  h.kate + h.pat + h.mark + h.sam = 198 ∧
  h.pat = 2 * h.kate ∧
  h.pat = h.mark / 3 ∧
  h.sam = (h.pat + h.mark) / 2

theorem mark_kate_difference (h : ProjectHours) (hvalid : valid_project_hours h) :
  h.mark - h.kate = 75 := by
  sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l1064_106433


namespace NUMINAMATH_CALUDE_church_full_capacity_l1064_106445

/-- Calculates the number of people needed to fill a church given the number of rows, chairs per row, and people per chair. -/
def church_capacity (rows : ℕ) (chairs_per_row : ℕ) (people_per_chair : ℕ) : ℕ :=
  rows * chairs_per_row * people_per_chair

/-- Theorem stating that a church with 20 rows, 6 chairs per row, and 5 people per chair can hold 600 people. -/
theorem church_full_capacity : church_capacity 20 6 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_church_full_capacity_l1064_106445


namespace NUMINAMATH_CALUDE_sum_8th_10th_is_230_l1064_106431

/-- An arithmetic sequence with given 4th and 6th terms -/
structure ArithmeticSequence where
  term4 : ℤ
  term6 : ℤ
  is_arithmetic : ∃ (a d : ℤ), term4 = a + 3 * d ∧ term6 = a + 5 * d

/-- The sum of the 8th and 10th terms of the arithmetic sequence -/
def sum_8th_10th_terms (seq : ArithmeticSequence) : ℤ :=
  let a : ℤ := seq.term4 - 3 * ((seq.term6 - seq.term4) / 2)
  let d : ℤ := (seq.term6 - seq.term4) / 2
  (a + 7 * d) + (a + 9 * d)

/-- Theorem stating that the sum of the 8th and 10th terms is 230 -/
theorem sum_8th_10th_is_230 (seq : ArithmeticSequence) 
  (h1 : seq.term4 = 25) (h2 : seq.term6 = 61) : 
  sum_8th_10th_terms seq = 230 := by
  sorry

end NUMINAMATH_CALUDE_sum_8th_10th_is_230_l1064_106431


namespace NUMINAMATH_CALUDE_no_real_solution_range_l1064_106415

theorem no_real_solution_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + x - a = 0) ↔ a < -1/4 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_range_l1064_106415


namespace NUMINAMATH_CALUDE_misread_weight_calculation_l1064_106484

theorem misread_weight_calculation (class_size : ℕ) (incorrect_avg : ℚ) (correct_avg : ℚ) (correct_weight : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  correct_avg = 58.7 →
  correct_weight = 62 →
  ∃ misread_weight : ℚ,
    class_size * correct_avg - class_size * incorrect_avg = correct_weight - misread_weight ∧
    misread_weight = 56 :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_calculation_l1064_106484


namespace NUMINAMATH_CALUDE_function_decreasing_on_interval_l1064_106485

/-- Given function f(x) = e^x - ax - 1, prove that there exists a real number a ≥ e^3
    such that f(x) is monotonically decreasing on the interval (-2, 3). -/
theorem function_decreasing_on_interval (a : ℝ) :
  ∃ (a : ℝ), a ≥ Real.exp 3 ∧
  ∀ (x y : ℝ), -2 < x ∧ x < y ∧ y < 3 →
    (Real.exp x - a * x - 1) > (Real.exp y - a * y - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_on_interval_l1064_106485


namespace NUMINAMATH_CALUDE_max_sides_is_12_l1064_106480

/-- A convex polygon that can be divided into right triangles with acute angles of 30 and 60 degrees -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool
  divisible_into_right_triangles : Bool
  acute_angles : Set ℝ
  acute_angles_eq : acute_angles = {30, 60}

/-- The maximum number of sides for the described convex polygon -/
def max_sides : ℕ := 12

/-- Theorem stating that the maximum number of sides for the described convex polygon is 12 -/
theorem max_sides_is_12 (p : ConvexPolygon) : p.sides ≤ max_sides := by
  sorry

end NUMINAMATH_CALUDE_max_sides_is_12_l1064_106480


namespace NUMINAMATH_CALUDE_perimeter_ratio_l1064_106448

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- The original large rectangle -/
def largeRectangle : Rectangle := { width := 4, height := 6 }

/-- One of the small rectangles after folding and cutting -/
def smallRectangle : Rectangle := { width := 2, height := 3 }

/-- Theorem stating the ratio of perimeters -/
theorem perimeter_ratio :
  perimeter smallRectangle / perimeter largeRectangle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_l1064_106448
