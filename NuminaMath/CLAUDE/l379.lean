import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l379_37934

theorem geometric_sequence_problem :
  ∃ (a r : ℝ), 
    -- Condition 1: a, ar, ar² form a geometric sequence
    (a * r * r - a * r = a * r - a) ∧
    -- Condition 2: ar² - a = 48
    (a * r * r - a = 48) ∧
    -- Condition 3: (ar²)² - a² = (208/217) * (a² + (ar)² + (ar²)²)
    ((a * r * r)^2 - a^2 = (208/217) * (a^2 + (a * r)^2 + (a * r * r)^2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l379_37934


namespace NUMINAMATH_CALUDE_german_enrollment_l379_37913

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 45)
  (h2 : both_subjects = 12)
  (h3 : only_english = 23)
  (h4 : total_students = only_english + both_subjects + (total_students - (only_english + both_subjects))) :
  total_students - (only_english + both_subjects) + both_subjects = 22 := by
  sorry

end NUMINAMATH_CALUDE_german_enrollment_l379_37913


namespace NUMINAMATH_CALUDE_function_inequality_l379_37962

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x : ℝ, f x < (deriv^[2] f) x) : 
  (Real.exp 2019 * f (-2019) < f 0) ∧ (f 2019 > Real.exp 2019 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l379_37962


namespace NUMINAMATH_CALUDE_joyful_point_properties_l379_37968

-- Define a "joyful point"
def is_joyful_point (m n : ℝ) : Prop := 2 * m = 6 - n

-- Define the point P
def P (m n : ℝ) : ℝ × ℝ := (m, n + 2)

theorem joyful_point_properties :
  -- Part 1: (1, 6) is a joyful point
  is_joyful_point 1 4 ∧
  P 1 4 = (1, 6) ∧
  -- Part 2: If P(a, -a+3) is a joyful point, then a = 5 and P is in the fourth quadrant
  (∀ a : ℝ, is_joyful_point a (-a + 3) → a = 5 ∧ 5 > 0 ∧ -2 < 0) ∧
  -- Part 3: The midpoint of OP is (5/2, -1)
  (let O : ℝ × ℝ := (0, 0);
   let P : ℝ × ℝ := (5, -2);
   (O.1 + P.1) / 2 = 5 / 2 ∧ (O.2 + P.2) / 2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_joyful_point_properties_l379_37968


namespace NUMINAMATH_CALUDE_iphone_purchase_savings_l379_37915

/-- The price of an iPhone X in dollars -/
def iphone_x_price : ℝ := 600

/-- The price of an iPhone Y in dollars -/
def iphone_y_price : ℝ := 800

/-- The discount rate for buying at least 2 smartphones of the same model -/
def same_model_discount : ℝ := 0.05

/-- The discount rate for mixed purchases of at least 3 smartphones -/
def mixed_purchase_discount : ℝ := 0.03

/-- The total cost of buying three iPhones individually -/
def individual_cost : ℝ := 2 * iphone_x_price + iphone_y_price

/-- The discounted price of two iPhone X models -/
def discounted_iphone_x : ℝ := 2 * (iphone_x_price * (1 - same_model_discount))

/-- The discounted price of one iPhone Y model -/
def discounted_iphone_y : ℝ := iphone_y_price * (1 - mixed_purchase_discount)

/-- The total cost of buying three iPhones together with discounts -/
def group_cost : ℝ := discounted_iphone_x + discounted_iphone_y

/-- The savings from buying three iPhones together vs. individually -/
def savings : ℝ := individual_cost - group_cost

theorem iphone_purchase_savings : savings = 84 := by sorry

end NUMINAMATH_CALUDE_iphone_purchase_savings_l379_37915


namespace NUMINAMATH_CALUDE_tank_filling_capacity_l379_37961

/-- Given a tank that can be filled with 28 buckets of 13.5 litres each,
    prove that if the same tank can be filled with 42 buckets of equal capacity,
    then the capacity of each bucket in the second case is 9 litres. -/
theorem tank_filling_capacity (tank_volume : ℝ) (bucket_count_1 bucket_count_2 : ℕ) 
    (bucket_capacity_1 : ℝ) :
  tank_volume = bucket_count_1 * bucket_capacity_1 →
  bucket_count_1 = 28 →
  bucket_capacity_1 = 13.5 →
  bucket_count_2 = 42 →
  ∃ bucket_capacity_2 : ℝ, 
    tank_volume = bucket_count_2 * bucket_capacity_2 ∧
    bucket_capacity_2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_tank_filling_capacity_l379_37961


namespace NUMINAMATH_CALUDE_parabola_directrix_l379_37964

/-- The directrix of a parabola y² = 2x is x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 2*x → (∃ (k : ℝ), k = -1/2 ∧ (∀ (x₀ y₀ : ℝ), y₀^2 = 2*x₀ → x₀ = k)) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l379_37964


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l379_37938

theorem farm_animal_ratio (total animals goats cows pigs : ℕ) : 
  total = 56 ∧ 
  goats = 11 ∧ 
  cows = goats + 4 ∧ 
  total = pigs + cows + goats → 
  pigs * 1 = cows * 2 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l379_37938


namespace NUMINAMATH_CALUDE_range_of_negative_values_l379_37936

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

-- State the theorem
theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on_neg f)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l379_37936


namespace NUMINAMATH_CALUDE_tangent_slope_implies_n_value_l379_37932

/-- The function f(x) defined as x^n + 3^x --/
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n + 3^x

/-- The derivative of f(x) --/
noncomputable def f_derivative (n : ℝ) (x : ℝ) : ℝ := n * x^(n-1) + 3^x * Real.log 3

theorem tangent_slope_implies_n_value (n : ℝ) :
  f n 1 = 4 →
  f_derivative n 1 = 3 + 3 * Real.log 3 →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_n_value_l379_37932


namespace NUMINAMATH_CALUDE_interest_is_37_cents_l379_37999

/-- Calculates the interest in cents given the initial principal and final amount after interest --/
def interest_in_cents (principal : ℚ) (final_amount : ℚ) : ℕ :=
  let interest_rate : ℚ := 3 / 100
  let time : ℚ := 1 / 4
  let interest : ℚ := final_amount - principal
  (interest * 100).floor.toNat

/-- Theorem stating that for some initial amount resulting in $310.45 after 3% annual simple interest for 3 months, the interest in cents is 37 --/
theorem interest_is_37_cents :
  ∃ (principal : ℚ),
    let final_amount : ℚ := 310.45
    interest_in_cents principal final_amount = 37 :=
sorry

end NUMINAMATH_CALUDE_interest_is_37_cents_l379_37999


namespace NUMINAMATH_CALUDE_hyperbola_equation_l379_37907

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / 12 + y^2 / 4 = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 - b^2 ∧ c^2 = 12 - 4) →
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x ∧ x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / 2 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l379_37907


namespace NUMINAMATH_CALUDE_inequality_equivalence_l379_37986

theorem inequality_equivalence (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l379_37986


namespace NUMINAMATH_CALUDE_domain_of_g_l379_37911

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function g
def g (f : Set ℝ) : Set ℝ := {x | x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g (f : Set ℝ) (hf : f = Set.Icc 0 4) : 
  g f = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l379_37911


namespace NUMINAMATH_CALUDE_paint_problem_l379_37930

theorem paint_problem (initial_paint : ℚ) : 
  initial_paint = 1 →
  let first_day_used := initial_paint / 2
  let first_day_remaining := initial_paint - first_day_used
  let second_day_first_op := first_day_remaining / 4
  let second_day_mid_remaining := first_day_remaining - second_day_first_op
  let second_day_second_op := second_day_mid_remaining / 8
  let final_remaining := second_day_mid_remaining - second_day_second_op
  final_remaining = (21 : ℚ) / 64 * initial_paint :=
by sorry

end NUMINAMATH_CALUDE_paint_problem_l379_37930


namespace NUMINAMATH_CALUDE_william_has_more_money_l379_37933

/-- Represents the amount of money in different currencies --/
structure Money where
  usd_20 : ℕ
  usd_10 : ℕ
  usd_5 : ℕ
  gbp_10 : ℕ
  eur_20 : ℕ

/-- Converts Money to USD --/
def to_usd (m : Money) (gbp_rate : ℚ) (eur_rate : ℚ) : ℚ :=
  (m.usd_20 * 20 + m.usd_10 * 10 + m.usd_5 * 5 + m.gbp_10 * 10 * gbp_rate + m.eur_20 * 20 * eur_rate : ℚ)

/-- Oliver's money --/
def oliver : Money := ⟨10, 0, 3, 12, 0⟩

/-- William's money --/
def william : Money := ⟨0, 15, 4, 0, 20⟩

/-- The exchange rates --/
def gbp_rate : ℚ := 138 / 100
def eur_rate : ℚ := 118 / 100

theorem william_has_more_money :
  to_usd william gbp_rate eur_rate - to_usd oliver gbp_rate eur_rate = 2614 / 10 := by
  sorry

end NUMINAMATH_CALUDE_william_has_more_money_l379_37933


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l379_37918

theorem complex_power_magnitude (z w : ℂ) (n : ℕ) :
  z = w ^ n → Complex.abs z ^ 2 = Complex.abs w ^ (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l379_37918


namespace NUMINAMATH_CALUDE_initial_distance_adrian_colton_initial_distance_l379_37941

/-- The initial distance between Adrian and Colton given their relative motion -/
theorem initial_distance (speed : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  let distance_run := speed * time
  distance_run + final_distance

/-- Proof of the initial distance between Adrian and Colton -/
theorem adrian_colton_initial_distance : 
  initial_distance 17 13 68 = 289 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_adrian_colton_initial_distance_l379_37941


namespace NUMINAMATH_CALUDE_trig_identity_l379_37928

theorem trig_identity (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + (Real.tan θ)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l379_37928


namespace NUMINAMATH_CALUDE_sum_of_solutions_l379_37937

-- Define the equations
def equation1 (x : ℝ) : Prop := x + Real.log x = 3
def equation2 (x : ℝ) : Prop := x + (10 : ℝ) ^ x = 3

-- State the theorem
theorem sum_of_solutions (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) (h2 : equation2 x₂) : x₁ + x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l379_37937


namespace NUMINAMATH_CALUDE_power_comparison_l379_37955

theorem power_comparison : 2^51 > 4^25 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l379_37955


namespace NUMINAMATH_CALUDE_complex_equation_solution_l379_37975

theorem complex_equation_solution (x y : ℝ) :
  (2 * x - 1 : ℂ) + I = -y - (3 - y) * I →
  x = -3/2 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l379_37975


namespace NUMINAMATH_CALUDE_probability_one_class_no_spot_l379_37995

/-- The number of spots for top students -/
def num_spots : ℕ := 6

/-- The number of classes -/
def num_classes : ℕ := 3

/-- The number of ways to distribute spots such that exactly one class doesn't receive a spot -/
def favorable_outcomes : ℕ := (num_classes.choose 2) * ((num_spots - 1).choose 1)

/-- The total number of ways to distribute spots among classes -/
def total_outcomes : ℕ := 
  (num_classes.choose 1) + 
  (num_classes.choose 2) * ((num_spots - 1).choose 1) + 
  (num_classes.choose 3) * ((num_spots - 1).choose 2)

/-- The probability that exactly one class does not receive a spot -/
theorem probability_one_class_no_spot : 
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_class_no_spot_l379_37995


namespace NUMINAMATH_CALUDE_clinton_wardrobe_problem_l379_37953

/-- Clinton's wardrobe problem -/
theorem clinton_wardrobe_problem (hats belts shoes : ℕ) :
  hats = 5 →
  belts = hats + 2 →
  shoes = 2 * belts →
  shoes = 14 := by sorry

end NUMINAMATH_CALUDE_clinton_wardrobe_problem_l379_37953


namespace NUMINAMATH_CALUDE_inequality_transformation_l379_37966

theorem inequality_transformation (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l379_37966


namespace NUMINAMATH_CALUDE_quadratic_minimum_l379_37959

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 10*x ≥ -25) ∧ (∃ x : ℝ, x^2 + 10*x = -25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l379_37959


namespace NUMINAMATH_CALUDE_proportional_segments_l379_37989

/-- Given four proportional line segments a, b, c, d, where b = 3, c = 4, and d = 6,
    prove that the length of line segment a is 2. -/
theorem proportional_segments (a b c d : ℝ) 
  (h_prop : a / b = c / d)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_d : d = 6) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l379_37989


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l379_37904

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The theorem states that if (a+i)(2+i) is a pure imaginary number, then a = 1/2 -/
theorem pure_imaginary_product (a : ℝ) : 
  is_pure_imaginary ((a + Complex.I) * (2 + Complex.I)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l379_37904


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l379_37900

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 16)
  (h_first : a 1 = 1) :
  a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l379_37900


namespace NUMINAMATH_CALUDE_soda_cost_l379_37970

theorem soda_cost (bob_burgers bob_sodas bob_total carol_burgers carol_sodas carol_total : ℕ) 
  (h_bob : bob_burgers = 4 ∧ bob_sodas = 3 ∧ bob_total = 500)
  (h_carol : carol_burgers = 3 ∧ carol_sodas = 4 ∧ carol_total = 540) :
  ∃ (burger_cost soda_cost : ℕ), 
    burger_cost * bob_burgers + soda_cost * bob_sodas = bob_total ∧
    burger_cost * carol_burgers + soda_cost * carol_sodas = carol_total ∧
    soda_cost = 94 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l379_37970


namespace NUMINAMATH_CALUDE_vector_equation_solution_l379_37967

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l379_37967


namespace NUMINAMATH_CALUDE_yellow_packs_bought_l379_37919

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℕ := 80

/-- The number of packs of green bouncy balls given away -/
def green_packs_given : ℕ := 4

/-- The number of packs of green bouncy balls bought -/
def green_packs_bought : ℕ := 4

/-- The theorem stating the number of packs of yellow bouncy balls Maggie bought -/
theorem yellow_packs_bought : 
  (total_balls / balls_per_pack : ℕ) = 8 :=
sorry

end NUMINAMATH_CALUDE_yellow_packs_bought_l379_37919


namespace NUMINAMATH_CALUDE_perfect_square_condition_l379_37939

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, ∀ x : ℤ, x^2 + 2*(m-3) + 16 = (x + k)^2) → (m = -1 ∨ m = 7) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l379_37939


namespace NUMINAMATH_CALUDE_g_of_nine_l379_37972

/-- Given a function g(x) = ax^7 - bx^3 + cx - 7 where g(-9) = 9, prove that g(9) = -23 -/
theorem g_of_nine (a b c : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = a * x^7 - b * x^3 + c * x - 7)
  (h2 : g (-9) = 9) : 
  g 9 = -23 := by sorry

end NUMINAMATH_CALUDE_g_of_nine_l379_37972


namespace NUMINAMATH_CALUDE_no_number_decreases_58_times_when_first_digit_removed_l379_37922

theorem no_number_decreases_58_times_when_first_digit_removed :
  ¬ ∃ (n : ℕ) (x y : ℕ), 
    n ≥ 2 ∧ 
    x > 0 ∧ x < 10 ∧
    y > 0 ∧
    x * 10^(n-1) + y = 58 * y :=
by sorry

end NUMINAMATH_CALUDE_no_number_decreases_58_times_when_first_digit_removed_l379_37922


namespace NUMINAMATH_CALUDE_triangle_angle_C_l379_37929

theorem triangle_angle_C (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  10 * a * Real.cos B = 3 * b * Real.cos A →
  Real.cos A = (5 * Real.sqrt 26) / 26 →
  C = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l379_37929


namespace NUMINAMATH_CALUDE_sin_double_angle_shift_graph_shift_equivalent_graphs_l379_37991

theorem sin_double_angle_shift (x : ℝ) :
  2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * (x + π / 6)) := by sorry

theorem graph_shift (x : ℝ) :
  2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * x + π / 3) := by sorry

theorem equivalent_graphs :
  ∀ x : ℝ, 2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * (x + π / 6)) := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_shift_graph_shift_equivalent_graphs_l379_37991


namespace NUMINAMATH_CALUDE_stratified_sampling_l379_37910

theorem stratified_sampling (seniors juniors freshmen sampled_freshmen : ℕ) :
  seniors = 1000 →
  juniors = 1200 →
  freshmen = 1500 →
  sampled_freshmen = 75 →
  (seniors + juniors + freshmen) * sampled_freshmen / freshmen = 185 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l379_37910


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l379_37954

-- Define the Cartesian coordinate system
def CartesianPoint := ℝ × ℝ

-- Define the fourth quadrant
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define pi as a real number
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant (π, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l379_37954


namespace NUMINAMATH_CALUDE_linear_function_intersection_k_range_l379_37940

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + (2 - 2 * k)

-- Define the intersection function
def intersection_function (x : ℝ) : ℝ := -x + 3

-- Define the domain
def in_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem linear_function_intersection_k_range :
  ∀ k : ℝ, 
  (∃ x : ℝ, in_domain x ∧ linear_function k x = intersection_function x) →
  ((k ≤ -2 ∨ k ≥ -1/2) ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_intersection_k_range_l379_37940


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l379_37925

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 8*y - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 4*y - 1 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-2, -4)
def radius1 : ℝ := 5
def center2 : ℝ × ℝ := (-2, -2)
def radius2 : ℝ := 3

-- Theorem statement
theorem circles_internally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = abs (radius1 - radius2) ∧ d < radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l379_37925


namespace NUMINAMATH_CALUDE_base_10_sum_45_l379_37997

/-- The sum of single-digit numbers in base b -/
def sum_single_digits (b : ℕ) : ℕ := (b - 1) * b / 2

/-- Checks if a number in base b has 5 as its units digit -/
def has_units_digit_5 (n : ℕ) (b : ℕ) : Prop := n % b = 5

theorem base_10_sum_45 :
  ∃ (b : ℕ), b > 1 ∧ sum_single_digits b = 45 ∧ has_units_digit_5 (sum_single_digits b) b ∧ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_10_sum_45_l379_37997


namespace NUMINAMATH_CALUDE_intersection_of_sets_l379_37958

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 0, 1, 2}
  let B : Set ℤ := {x | x ≥ 2}
  A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l379_37958


namespace NUMINAMATH_CALUDE_max_valid_rectangles_l379_37916

/-- Represents a grid with dimensions and unit square size -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (unit_size : ℕ)

/-- Represents a coloring of the grid -/
def Coloring := Grid → Fin 2

/-- Represents a cutting of the grid into rectangles -/
def Cutting := Grid → List (ℕ × ℕ)

/-- Counts the number of rectangles with at most one black square -/
def count_valid_rectangles (g : Grid) (c : Coloring) (cut : Cutting) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of valid rectangles -/
theorem max_valid_rectangles (g : Grid) 
  (h1 : g.width = 2020)
  (h2 : g.height = 2020)
  (h3 : g.unit_size = 11)
  (h4 : g.width / g.unit_size * (g.height / g.unit_size) = 400) :
  ∃ (c : Coloring) (cut : Cutting), 
    ∀ (c' : Coloring) (cut' : Cutting), 
      count_valid_rectangles g c cut ≥ count_valid_rectangles g c' cut' ∧ 
      count_valid_rectangles g c cut = 20 :=
sorry

end NUMINAMATH_CALUDE_max_valid_rectangles_l379_37916


namespace NUMINAMATH_CALUDE_complex_number_equidistant_l379_37926

theorem complex_number_equidistant (z : ℂ) :
  Complex.abs (z - Complex.I) = Complex.abs (z - 1) ∧
  Complex.abs (z - 1) = Complex.abs (z - 2015) →
  z = Complex.mk 1008 1008 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_equidistant_l379_37926


namespace NUMINAMATH_CALUDE_brother_twice_sister_age_l379_37981

theorem brother_twice_sister_age (brother_age_2010 sister_age_2010 : ℕ) : 
  brother_age_2010 = 16 →
  sister_age_2010 = 10 →
  ∃ (year : ℕ), year = 2006 ∧ 
    brother_age_2010 - (2010 - year) = 2 * (sister_age_2010 - (2010 - year)) :=
by sorry

end NUMINAMATH_CALUDE_brother_twice_sister_age_l379_37981


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l379_37957

-- Problem 1: Ellipse
theorem ellipse_equation (x y : ℝ) :
  let foci_ellipse : ℝ × ℝ → Prop := λ p => p.1^2 / 9 + p.2^2 / 4 = 1
  let passes_through : ℝ × ℝ → Prop := λ p => p = (-3, 2)
  let result_equation : ℝ × ℝ → Prop := λ p => p.1^2 / 15 + p.2^2 / 10 = 1
  (∃ e : Set (ℝ × ℝ), (∀ p ∈ e, result_equation p) ∧
    (∃ p ∈ e, passes_through p) ∧
    (∀ f : ℝ × ℝ, foci_ellipse f ↔ (∃ f' : ℝ × ℝ, (∀ p ∈ e, (p.1 - f.1)^2 + (p.2 - f.2)^2 =
                                                          (p.1 - f'.1)^2 + (p.2 - f'.2)^2) ∧
                                                 f.2 = f'.2 ∧ f.1 = -f'.1))) :=
by sorry

-- Problem 2: Hyperbola
theorem hyperbola_equation (x y : ℝ) :
  let passes_through : ℝ × ℝ → Prop := λ p => p = (2, -1)
  let has_asymptotes : (ℝ → ℝ) → Prop := λ f => f x = 3 * x ∨ f x = -3 * x
  let result_equation : ℝ × ℝ → Prop := λ p => p.1^2 / (35/9) - p.2^2 / 35 = 1
  (∃ h : Set (ℝ × ℝ), (∀ p ∈ h, result_equation p) ∧
    (∃ p ∈ h, passes_through p) ∧
    (∃ f g : ℝ → ℝ, has_asymptotes f ∧ has_asymptotes g ∧
      (∀ x : ℝ, (x, f x) ∉ h ∧ (x, g x) ∉ h) ∧
      (∀ ε > 0, ∃ δ > 0, ∀ p ∈ h, |p.1| > δ → (|p.2 - f p.1| < ε ∨ |p.2 - g p.1| < ε)))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l379_37957


namespace NUMINAMATH_CALUDE_difference_between_number_and_fraction_l379_37980

theorem difference_between_number_and_fraction (x : ℝ) (h : x = 155) : x - (3/5 * x) = 62 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_number_and_fraction_l379_37980


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l379_37984

theorem modular_inverse_of_5_mod_23 :
  ∃ a : ℕ, a ≤ 22 ∧ (5 * a) % 23 = 1 ∧ a = 14 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l379_37984


namespace NUMINAMATH_CALUDE_lori_beanie_babies_l379_37978

theorem lori_beanie_babies (sydney_beanie_babies : ℕ) 
  (h1 : sydney_beanie_babies + 15 * sydney_beanie_babies = 320) : 
  15 * sydney_beanie_babies = 300 := by
  sorry

end NUMINAMATH_CALUDE_lori_beanie_babies_l379_37978


namespace NUMINAMATH_CALUDE_good_numbers_in_set_l379_37977

/-- A number n is a "good number" if there exists a permutation of 1..n such that
    k + a[k] is a perfect square for all k in 1..n -/
def is_good_number (n : ℕ) : Prop :=
  ∃ a : Fin n → Fin n, Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k : ℕ) + (a k : ℕ) + 1 = m * m

theorem good_numbers_in_set : 
  is_good_number 13 ∧ 
  is_good_number 15 ∧ 
  is_good_number 17 ∧ 
  is_good_number 19 ∧ 
  ¬is_good_number 11 := by
  sorry

#check good_numbers_in_set

end NUMINAMATH_CALUDE_good_numbers_in_set_l379_37977


namespace NUMINAMATH_CALUDE_work_completion_time_l379_37912

/-- Given a work that can be completed by person A in 60 days, and together with person B in 24 days,
    this theorem proves that B can complete the remaining work alone in 40 days after A works for 15 days. -/
theorem work_completion_time (total_work : ℝ) : 
  (∃ (rate_a rate_b : ℝ), 
    rate_a * 60 = total_work ∧ 
    (rate_a + rate_b) * 24 = total_work ∧ 
    rate_b * 40 = total_work - rate_a * 15) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l379_37912


namespace NUMINAMATH_CALUDE_theta_range_l379_37950

theorem theta_range (θ : Real) : 
  (∀ x : Real, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) → 
  π/12 < θ ∧ θ < 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_theta_range_l379_37950


namespace NUMINAMATH_CALUDE_min_value_function_l379_37952

theorem min_value_function (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x ≥ 6 ∧ ∃ y > 0, 2 + 4*y + 1/y = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l379_37952


namespace NUMINAMATH_CALUDE_seed_cost_calculation_l379_37943

/-- Given that 2 pounds of seed cost $44.68, prove that 6 pounds of seed will cost $134.04. -/
theorem seed_cost_calculation (cost_for_two_pounds : ℝ) (pounds_needed : ℝ) : 
  cost_for_two_pounds = 44.68 → pounds_needed = 6 → 
  (pounds_needed / 2) * cost_for_two_pounds = 134.04 := by
sorry

end NUMINAMATH_CALUDE_seed_cost_calculation_l379_37943


namespace NUMINAMATH_CALUDE_no_valid_positive_x_for_equal_volume_increase_l379_37944

theorem no_valid_positive_x_for_equal_volume_increase (x : ℝ) : 
  x > 0 → 
  π * (5 + x)^2 * 10 - π * 5^2 * 10 ≠ π * 5^2 * (10 + x) - π * 5^2 * 10 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_positive_x_for_equal_volume_increase_l379_37944


namespace NUMINAMATH_CALUDE_nineteenth_row_red_squares_l379_37996

/-- Represents the number of squares in the nth row of a stair-step figure -/
def num_squares (n : ℕ) : ℕ := 3 * n - 1

/-- Represents the number of red squares in the nth row of a stair-step figure -/
def num_red_squares (n : ℕ) : ℕ := (num_squares n) / 2

theorem nineteenth_row_red_squares :
  num_red_squares 19 = 28 := by sorry

end NUMINAMATH_CALUDE_nineteenth_row_red_squares_l379_37996


namespace NUMINAMATH_CALUDE_triangle_cosine_value_l379_37935

theorem triangle_cosine_value (A B C : ℝ) (h : 7 * Real.sin B ^ 2 + 3 * Real.sin C ^ 2 = 2 * Real.sin A ^ 2 + 2 * Real.sin A * Real.sin B * Real.sin C) :
  Real.cos (A - π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_value_l379_37935


namespace NUMINAMATH_CALUDE_seven_sided_die_perfect_square_probability_l379_37902

/-- Represents a fair seven-sided die with faces numbered 1 through 7 -/
def SevenSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The number of times the die is rolled -/
def numRolls : ℕ := 4

/-- The total number of possible outcomes when rolling the die numRolls times -/
def totalOutcomes : ℕ := SevenSidedDie.card ^ numRolls

/-- The number of favorable outcomes (product of rolls is a perfect square) -/
def favorableOutcomes : ℕ := 164

theorem seven_sided_die_perfect_square_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 164 / 2401 :=
sorry

end NUMINAMATH_CALUDE_seven_sided_die_perfect_square_probability_l379_37902


namespace NUMINAMATH_CALUDE_island_puzzle_l379_37921

/-- Represents the nature of a person on the island -/
inductive PersonNature
| Knight
| Liar

/-- Represents the statement made by person A -/
def statement (nature : PersonNature) (treasures : Prop) : Prop :=
  (nature = PersonNature.Knight) ↔ treasures

/-- The main theorem about A's statement and the existence of treasures -/
theorem island_puzzle :
  ∀ (A_nature : PersonNature) (treasures : Prop),
    statement A_nature treasures →
    (¬ (A_nature = PersonNature.Knight ∨ A_nature = PersonNature.Liar) ∧ treasures) :=
by sorry

end NUMINAMATH_CALUDE_island_puzzle_l379_37921


namespace NUMINAMATH_CALUDE_sum_has_no_real_roots_l379_37983

/-- A quadratic polynomial with integer coefficients. -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate for an acceptable quadratic polynomial. -/
def is_acceptable (p : QuadraticPolynomial) : Prop :=
  abs p.a ≤ 2013 ∧ abs p.b ≤ 2013 ∧ abs p.c ≤ 2013 ∧
  ∃ (r₁ r₂ : ℤ), p.a * r₁^2 + p.b * r₁ + p.c = 0 ∧ p.a * r₂^2 + p.b * r₂ + p.c = 0

/-- The set of all acceptable quadratic polynomials. -/
def acceptable_polynomials : Set QuadraticPolynomial :=
  {p : QuadraticPolynomial | is_acceptable p}

/-- The sum of all acceptable quadratic polynomials. -/
noncomputable def sum_of_acceptable_polynomials : QuadraticPolynomial :=
  sorry

/-- Theorem stating that the sum of all acceptable quadratic polynomials has no real roots. -/
theorem sum_has_no_real_roots :
  ∃ (A C : ℤ), A > 0 ∧ C > 0 ∧
  sum_of_acceptable_polynomials.a = A ∧
  sum_of_acceptable_polynomials.b = 0 ∧
  sum_of_acceptable_polynomials.c = C :=
sorry

end NUMINAMATH_CALUDE_sum_has_no_real_roots_l379_37983


namespace NUMINAMATH_CALUDE_maggie_bought_ten_magazines_l379_37987

/-- The number of science magazines Maggie bought -/
def num_magazines : ℕ := 10

/-- The number of books Maggie bought -/
def num_books : ℕ := 10

/-- The cost of each book in dollars -/
def book_cost : ℕ := 15

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 2

/-- The total amount Maggie spent in dollars -/
def total_spent : ℕ := 170

/-- Proof that Maggie bought 10 science magazines -/
theorem maggie_bought_ten_magazines :
  num_magazines = 10 ∧
  num_books * book_cost + num_magazines * magazine_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_maggie_bought_ten_magazines_l379_37987


namespace NUMINAMATH_CALUDE_rectangle_y_value_l379_37927

/-- Given a rectangle with vertices at (-2, y), (8, y), (-2, 2), and (8, 2),
    with an area of 100 square units and y > 2, prove that y = 12. -/
theorem rectangle_y_value (y : ℝ) 
    (h1 : (8 - (-2)) * (y - 2) = 100)  -- Area of rectangle is 100
    (h2 : y > 2) :                     -- y is greater than 2
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l379_37927


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l379_37917

theorem largest_prime_factor_of_1729 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 1729 → q ≤ p ∧ p = 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l379_37917


namespace NUMINAMATH_CALUDE_agreed_period_is_18_months_prove_agreed_period_of_service_l379_37920

/-- Represents the agreed period of service in months -/
def agreed_period : ℕ := 18

/-- Represents the actual period served in months -/
def actual_period : ℕ := 9

/-- Represents the full payment amount in rupees -/
def full_payment : ℕ := 800

/-- Represents the actual payment received in rupees -/
def actual_payment : ℕ := 400

/-- Theorem stating that the agreed period of service is 18 months -/
theorem agreed_period_is_18_months :
  (actual_payment = full_payment / 2) →
  (actual_period * 2 = agreed_period) :=
by
  sorry

/-- Main theorem proving the agreed period of service -/
theorem prove_agreed_period_of_service :
  agreed_period = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_agreed_period_is_18_months_prove_agreed_period_of_service_l379_37920


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l379_37963

def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 12*x^2 + 20*x - 18

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l379_37963


namespace NUMINAMATH_CALUDE_square_root_of_four_l379_37949

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l379_37949


namespace NUMINAMATH_CALUDE_jack_morning_emails_l379_37923

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 16

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 2

theorem jack_morning_emails : 
  morning_emails = afternoon_emails + email_difference := by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l379_37923


namespace NUMINAMATH_CALUDE_perfect_square_primes_no_perfect_square_primes_l379_37951

theorem perfect_square_primes (p : Nat) : Prime p → (∃ n : Nat, (7^(p-1) - 1) / p = n^2) ↔ p = 3 :=
sorry

theorem no_perfect_square_primes (p : Nat) : Prime p → ¬∃ n : Nat, (11^(p-1) - 1) / p = n^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_primes_no_perfect_square_primes_l379_37951


namespace NUMINAMATH_CALUDE_number_of_girls_l379_37998

theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls neutral_boys : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : boys = 22)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 10)
  (h9 : happy_children + sad_children + neutral_children = total_children)
  : total_children - boys = 38 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l379_37998


namespace NUMINAMATH_CALUDE_trig_values_equal_for_same_terminal_side_l379_37985

-- Define what it means for two angles to have the same terminal side
def same_terminal_side (α β : Real) : Prop := sorry

-- Define a general trigonometric function
def trig_function (α : Real) : Real := sorry

theorem trig_values_equal_for_same_terminal_side :
  ∀ (α β : Real) (f : Real → Real),
  same_terminal_side α β →
  f = trig_function →
  f α = f β :=
sorry

end NUMINAMATH_CALUDE_trig_values_equal_for_same_terminal_side_l379_37985


namespace NUMINAMATH_CALUDE_victor_insect_stickers_l379_37971

/-- The number of insect stickers Victor has -/
def insect_stickers (flower_stickers : ℝ) (total_stickers : ℝ) : ℝ :=
  total_stickers - (2 * flower_stickers - 3.5) - (1.5 * flower_stickers + 5.5)

theorem victor_insect_stickers :
  insect_stickers 15 70 = 15.5 := by sorry

end NUMINAMATH_CALUDE_victor_insect_stickers_l379_37971


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l379_37965

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 4 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (hr : r > 0) : a r = 10 * r - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l379_37965


namespace NUMINAMATH_CALUDE_probability_one_good_product_l379_37948

def total_products : ℕ := 5
def good_products : ℕ := 3
def defective_products : ℕ := 2
def selections : ℕ := 2

theorem probability_one_good_product : 
  (Nat.choose good_products 1 * Nat.choose defective_products 1) / 
  Nat.choose total_products selections = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_one_good_product_l379_37948


namespace NUMINAMATH_CALUDE_unique_solution_two_and_five_l379_37901

theorem unique_solution_two_and_five (x : ℝ) : (x - 2) * (x - 5) = 0 ↔ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_two_and_five_l379_37901


namespace NUMINAMATH_CALUDE_rectangle_area_equals_44_l379_37909

/-- Given a triangle with sides a, b, c and a rectangle with one side length d,
    if the perimeters are equal and d = 8, then the area of the rectangle is 44. -/
theorem rectangle_area_equals_44 (a b c d : ℝ) : 
  a = 7.5 → b = 9 → c = 10.5 → d = 8 → 
  a + b + c = 2 * (d + (a + b + c) / 2 - d) → 
  d * ((a + b + c) / 2 - d) = 44 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_44_l379_37909


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l379_37960

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l379_37960


namespace NUMINAMATH_CALUDE_max_m_F_theorem_l379_37969

/-- The maximum value of m(F) for subsets F of {1, ..., 2n} with n elements -/
def max_m_F (n : ℕ) : ℕ :=
  if n = 2 ∨ n = 3 then 12
  else if n = 4 then 24
  else if n % 2 = 1 then 3 * (n + 1)
  else 3 * (n + 2)

/-- The theorem stating the maximum value of m(F) -/
theorem max_m_F_theorem (n : ℕ) (h : n ≥ 2) :
  ∀ (F : Finset ℕ),
    F ⊆ Finset.range (2 * n + 1) →
    F.card = n →
    (∀ (x y : ℕ), x ∈ F → y ∈ F → x ≠ y → Nat.lcm x y ≥ max_m_F n) :=
by sorry

end NUMINAMATH_CALUDE_max_m_F_theorem_l379_37969


namespace NUMINAMATH_CALUDE_seven_thousand_twenty_two_l379_37908

theorem seven_thousand_twenty_two : 7000 + 22 = 7022 := by
  sorry

end NUMINAMATH_CALUDE_seven_thousand_twenty_two_l379_37908


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l379_37905

theorem cyclic_sum_inequality (x y z : ℝ) (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x*y/z) + (y*z/x) + (z*x/y) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l379_37905


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l379_37903

theorem quadratic_roots_property (b c : ℝ) : 
  (3 * b^2 + 5 * b - 2 = 0) → 
  (3 * c^2 + 5 * c - 2 = 0) → 
  (b-1)*(c-1) = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l379_37903


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l379_37988

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- The parabola equation -/
def parabola_eq (x y p : ℝ) : Prop := y^2 = 2*p*x

/-- The directrix equation of the parabola -/
def directrix_eq (x p : ℝ) : Prop := x = -p/2

/-- The length of the line segment cut by the circle on the directrix -/
def segment_length (p : ℝ) : ℝ := 4

/-- The theorem to be proved -/
theorem parabola_circle_intersection (p : ℝ) 
  (h_p_pos : p > 0) 
  (h_segment : segment_length p = 4) : p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l379_37988


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l379_37974

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0,
    if one of its asymptotic lines is y = 2x, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∃ (x y : ℝ), x^2 - y^2/b^2 = 1 ∧ y = 2*x) → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l379_37974


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l379_37976

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 480 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃ (n : ℕ), n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 ∧ ∀ (m : ℕ), (m > 0 ∧ 24 ∣ m^2 ∧ 480 ∣ m^3) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l379_37976


namespace NUMINAMATH_CALUDE_production_consistency_gizmo_production_zero_l379_37942

/-- Represents the production rate of gadgets and gizmos -/
structure ProductionRate where
  workers : ℕ
  hours : ℕ
  gadgets : ℕ
  gizmos : ℕ

/-- Given production rates -/
def rate1 : ProductionRate := ⟨120, 1, 360, 240⟩
def rate2 : ProductionRate := ⟨40, 3, 240, 360⟩
def rate3 : ProductionRate := ⟨60, 4, 240, 0⟩

/-- Time to produce one gadget -/
def gadgetTime (r : ProductionRate) : ℚ :=
  (r.workers * r.hours : ℚ) / r.gadgets

/-- Time to produce one gizmo -/
def gizmoTime (r : ProductionRate) : ℚ :=
  (r.workers * r.hours : ℚ) / r.gizmos

theorem production_consistency (r1 r2 : ProductionRate) :
  gadgetTime r1 = gadgetTime r2 ∧ gizmoTime r1 = gizmoTime r2 := by sorry

theorem gizmo_production_zero :
  rate3.gizmos = 0 := by sorry

end NUMINAMATH_CALUDE_production_consistency_gizmo_production_zero_l379_37942


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l379_37924

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 130) (h2 : b = 95) (h3 : c = 115) (h4 : d = 110) (h5 : e = 87) : 
  720 - (a + b + c + d + e) = 183 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l379_37924


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l379_37982

/-- The sum of x-coordinates and y-coordinates of the intersection points of two parabolas -/
theorem intersection_sum_zero (x y : ℝ → ℝ) : 
  (∀ t, y t = (x t - 2)^2) →
  (∀ t, x t + 3 = (y t + 2)^2) →
  (∃ a b c d : ℝ, 
    (y a = (x a - 2)^2 ∧ x a + 3 = (y a + 2)^2) ∧
    (y b = (x b - 2)^2 ∧ x b + 3 = (y b + 2)^2) ∧
    (y c = (x c - 2)^2 ∧ x c + 3 = (y c + 2)^2) ∧
    (y d = (x d - 2)^2 ∧ x d + 3 = (y d + 2)^2) ∧
    (∀ t, y t = (x t - 2)^2 ∧ x t + 3 = (y t + 2)^2 → t = a ∨ t = b ∨ t = c ∨ t = d)) →
  x a + x b + x c + x d + y a + y b + y c + y d = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l379_37982


namespace NUMINAMATH_CALUDE_distance_between_stations_l379_37947

/-- The distance between two stations given train travel information -/
theorem distance_between_stations
  (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ)
  (h1 : speed1 = 20)
  (h2 : time1 = 3)
  (h3 : speed2 = 25)
  (h4 : time2 = 2)
  (h5 : speed1 * time1 + speed2 * time2 = speed1 * time1 + speed2 * time2) :
  speed1 * time1 + speed2 * time2 = 110 := by
  sorry

#check distance_between_stations

end NUMINAMATH_CALUDE_distance_between_stations_l379_37947


namespace NUMINAMATH_CALUDE_four_lines_max_regions_l379_37992

/-- The maximum number of regions a plane can be divided into by n lines -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions a plane can be divided into by four lines is 11 -/
theorem four_lines_max_regions : max_regions 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_four_lines_max_regions_l379_37992


namespace NUMINAMATH_CALUDE_gift_package_combinations_l379_37945

theorem gift_package_combinations : 
  let wrapping_paper_varieties : ℕ := 10
  let ribbon_colors : ℕ := 5
  let gift_card_types : ℕ := 5
  let gift_tag_types : ℕ := 2
  wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_tag_types = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_package_combinations_l379_37945


namespace NUMINAMATH_CALUDE_roberto_final_salary_l379_37990

/-- Calculates the final salary after raises, bonus, and taxes -/
def final_salary (starting_salary : ℝ) (first_raise_percent : ℝ) (second_raise_percent : ℝ) (bonus : ℝ) (tax_rate : ℝ) : ℝ :=
  let previous_salary := starting_salary * (1 + first_raise_percent)
  let current_salary := previous_salary * (1 + second_raise_percent)
  let total_income := current_salary + bonus
  let taxes := total_income * tax_rate
  total_income - taxes

/-- Theorem stating that Roberto's final salary is $104,550 -/
theorem roberto_final_salary :
  final_salary 80000 0.4 0.2 5000 0.25 = 104550 := by
  sorry

end NUMINAMATH_CALUDE_roberto_final_salary_l379_37990


namespace NUMINAMATH_CALUDE_cube_of_fraction_l379_37956

theorem cube_of_fraction (x y : ℝ) : 
  (-2/3 * x * y^2)^3 = -8/27 * x^3 * y^6 := by
sorry

end NUMINAMATH_CALUDE_cube_of_fraction_l379_37956


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l379_37946

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^2 - 18) / ((x - 1) * (x - 4) * (x + 2)) =
      P / (x - 1) + Q / (x - 4) + R / (x + 2) ∧
      P = 17/9 ∧ Q = 1/9 ∧ R = -5/9 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l379_37946


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l379_37993

theorem arithmetic_expression_evaluation : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l379_37993


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l379_37979

/-- Given two chemical solutions x and y, and their mixture, prove the percentage of chemical b in solution x -/
theorem chemical_mixture_problem (x_a : ℝ) (y_a y_b : ℝ) (mixture_a : ℝ) :
  x_a = 0.3 →
  y_a = 0.4 →
  y_b = 0.6 →
  mixture_a = 0.32 →
  0.8 * x_a + 0.2 * y_a = mixture_a →
  1 - x_a = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l379_37979


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l379_37906

/-- Represents a quadratic equation in standard form -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- The given equation: 3(x+1)^2 = 2(x+1) -/
def given_equation (x : ℝ) : Prop :=
  3 * (x + 1)^2 = 2 * (x + 1)

/-- Theorem stating that the given equation is equivalent to a quadratic equation -/
theorem given_equation_is_quadratic :
  ∃ (q : QuadraticEquation), ∀ x, given_equation x ↔ q.a * x^2 + q.b * x + q.c = 0 :=
sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l379_37906


namespace NUMINAMATH_CALUDE_square_ratio_problem_l379_37931

theorem square_ratio_problem (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 75 / 48 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l379_37931


namespace NUMINAMATH_CALUDE_dice_roll_probability_l379_37973

def is_valid_roll (m n : ℕ) : Prop := 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6

def angle_greater_than_90 (m n : ℕ) : Prop := m > n

def count_favorable_outcomes : ℕ := 15

def total_outcomes : ℕ := 36

theorem dice_roll_probability : 
  (count_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l379_37973


namespace NUMINAMATH_CALUDE_linear_equation_condition_l379_37994

/-- Given that (a-3)x^|a-2| + 4 = 0 is a linear equation in x and a-3 ≠ 0, prove that a = 1 -/
theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k, (a - 3) * x^(|a - 2|) + 4 = k * x + 4) ∧ 
  (a - 3 ≠ 0) → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l379_37994


namespace NUMINAMATH_CALUDE_not_p_sufficient_but_not_necessary_for_not_q_l379_37914

-- Define the conditions p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define what it means for one condition to be sufficient but not necessary for another
def sufficient_but_not_necessary (A B : Prop) : Prop :=
  (A → B) ∧ ¬(B → A)

-- Theorem statement
theorem not_p_sufficient_but_not_necessary_for_not_q :
  sufficient_but_not_necessary (¬∃ x, p x) (¬∃ x, q x) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_but_not_necessary_for_not_q_l379_37914
