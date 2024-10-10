import Mathlib

namespace matthew_initial_crackers_l1069_106990

/-- The number of crackers Matthew gave to each friend -/
def crackers_per_friend : ℕ := 2

/-- The number of friends Matthew gave crackers to -/
def number_of_friends : ℕ := 4

/-- The total number of crackers Matthew gave away -/
def total_crackers_given : ℕ := crackers_per_friend * number_of_friends

/-- Theorem stating that Matthew had at least 8 crackers initially -/
theorem matthew_initial_crackers :
  ∃ (initial_crackers : ℕ), initial_crackers ≥ total_crackers_given ∧ initial_crackers ≥ 8 := by
  sorry

end matthew_initial_crackers_l1069_106990


namespace nonzero_x_equality_l1069_106954

theorem nonzero_x_equality (x : ℝ) (hx : x ≠ 0) (h : (9 * x)^18 = (18 * x)^9) : x = 2/9 := by
  sorry

end nonzero_x_equality_l1069_106954


namespace complex_inside_unit_circle_l1069_106925

theorem complex_inside_unit_circle (x : ℝ) :
  (∀ z : ℂ, z = x - (1/3 : ℝ) * Complex.I → Complex.abs z < 1) →
  -2 * Real.sqrt 2 / 3 < x ∧ x < 2 * Real.sqrt 2 / 3 := by
  sorry

end complex_inside_unit_circle_l1069_106925


namespace average_roots_quadratic_l1069_106912

theorem average_roots_quadratic (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → 3*x^2 + 4*x - 8 = 0 → (x₁ + x₂) / 2 = -2/3 :=
by sorry

end average_roots_quadratic_l1069_106912


namespace unique_number_satisfying_conditions_l1069_106913

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_rearrangement (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
    (n : ℚ) / (digits_product n : ℚ) = 16 / 3 ∧
    is_rearrangement n (n - 9) ∧
    n = 32 := by sorry

end unique_number_satisfying_conditions_l1069_106913


namespace median_on_hypotenuse_of_right_triangle_l1069_106906

theorem median_on_hypotenuse_of_right_triangle 
  (a b : ℝ) (ha : a = 6) (hb : b = 8) : 
  let c := Real.sqrt (a^2 + b^2)
  let m := c / 2
  m = 5 := by sorry

end median_on_hypotenuse_of_right_triangle_l1069_106906


namespace shaded_area_is_73_l1069_106987

/-- The total area of two overlapping rectangles minus their common area -/
def total_shaded_area (length1 width1 length2 width2 overlap_area : ℕ) : ℕ :=
  length1 * width1 + length2 * width2 - overlap_area

/-- Theorem stating that the total shaded area is 73 for the given dimensions -/
theorem shaded_area_is_73 :
  total_shaded_area 8 5 4 9 3 = 73 := by
  sorry

end shaded_area_is_73_l1069_106987


namespace sequence_properties_l1069_106991

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 3 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 2^n

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem sequence_properties :
  (a 1 = 2) ∧
  (b 1 = 2) ∧
  (a 4 + b 4 = 27) ∧
  (S 4 - b 4 = 10) ∧
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n : ℕ, b (n + 1) / b n = b 2 / b 1) ∧
  (∃ m : ℕ+, 
    (4 / m : ℚ) * a (7 - m) = (b 1)^2 ∧
    (4 / m : ℚ) * a 7 = (b 2)^2 ∧
    (4 / m : ℚ) * a (7 + 4 * m) = (b 3)^2) :=
by sorry

#check sequence_properties

end sequence_properties_l1069_106991


namespace correction_is_15x_l1069_106963

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "half-dollar" => 50
  | "nickel" => 5
  | _ => 0

/-- Calculates the correction needed for miscounted coins -/
def correction_needed (x : ℕ) : ℤ :=
  let quarter_dime_diff := (coin_value "quarter" - coin_value "dime") * (2 * x)
  let half_dollar_nickel_diff := (coin_value "half-dollar" - coin_value "nickel") * x
  quarter_dime_diff - half_dollar_nickel_diff

theorem correction_is_15x (x : ℕ) : correction_needed x = 15 * x := by
  sorry

end correction_is_15x_l1069_106963


namespace find_m_value_l1069_106938

theorem find_m_value (n : ℕ) (m : ℕ) (h1 : n = 9998) (h2 : 72517 * (n + 1) = m) : 
  m = 725092483 := by
  sorry

end find_m_value_l1069_106938


namespace S_intersect_T_eq_T_l1069_106902

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l1069_106902


namespace debt_average_payment_l1069_106917

theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (additional_amount : ℚ) : 
  total_installments = 100 → 
  first_payment_count = 30 → 
  first_payment_amount = 620 → 
  additional_amount = 110 → 
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + additional_amount
  let total_amount := 
    (first_payment_count * first_payment_amount) + 
    (remaining_payment_count * remaining_payment_amount)
  total_amount / total_installments = 697 := by
sorry

end debt_average_payment_l1069_106917


namespace paper_boat_travel_time_l1069_106946

/-- Represents the problem of calculating the time for a paper boat to travel along an embankment --/
theorem paper_boat_travel_time 
  (embankment_length : ℝ)
  (boat_length : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h1 : embankment_length = 50)
  (h2 : boat_length = 10)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 4) :
  let downstream_speed := embankment_length / downstream_time
  let upstream_speed := embankment_length / upstream_time
  let boat_speed := (downstream_speed + upstream_speed) / 2
  let current_speed := (downstream_speed - upstream_speed) / 2
  (embankment_length / current_speed) = 40 := by
  sorry

end paper_boat_travel_time_l1069_106946


namespace product_of_special_integers_l1069_106939

theorem product_of_special_integers (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 1000 / (p * q * r) = 1) :
  p * q * r = 1600 := by
  sorry

end product_of_special_integers_l1069_106939


namespace lizzies_garbage_collection_l1069_106978

theorem lizzies_garbage_collection (x : ℝ) 
  (h1 : x + (x - 39) = 735) : x = 387 := by
  sorry

end lizzies_garbage_collection_l1069_106978


namespace square_area_equals_perimeter_l1069_106968

theorem square_area_equals_perimeter (s : ℝ) (h : s > 0) : 
  s^2 = 4*s → 4*s = 16 := by
sorry

end square_area_equals_perimeter_l1069_106968


namespace divisible_by_seven_l1069_106929

theorem divisible_by_seven : ∃ k : ℤ, (1 + 5)^4 - 1 = 7 * k := by
  sorry

end divisible_by_seven_l1069_106929


namespace james_age_when_thomas_grows_l1069_106958

/-- Given the ages and relationships of Thomas, Shay, and James, prove James' age when Thomas reaches his current age. -/
theorem james_age_when_thomas_grows (thomas_age : ℕ) (shay_thomas_diff : ℕ) (james_shay_diff : ℕ) : 
  thomas_age = 6 →
  shay_thomas_diff = 13 →
  james_shay_diff = 5 →
  thomas_age + shay_thomas_diff + james_shay_diff + shay_thomas_diff = 37 :=
by sorry

end james_age_when_thomas_grows_l1069_106958


namespace root_sum_theorem_l1069_106903

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 181/9 := by
sorry

end root_sum_theorem_l1069_106903


namespace triangle_third_side_l1069_106937

theorem triangle_third_side (a b x : ℝ) : 
  a = 3 ∧ b = 9 ∧ 
  (a + b > x) ∧ (a + x > b) ∧ (b + x > a) →
  x = 10 → True :=
by sorry

end triangle_third_side_l1069_106937


namespace inequality_contradiction_l1069_106933

theorem inequality_contradiction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ 
    (a + b) * (c + d) < a * b + c * d ∧ 
    (a + b) * c * d < a * b * (c + d)) :=
by sorry

end inequality_contradiction_l1069_106933


namespace boys_in_class_l1069_106974

theorem boys_in_class (total : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) 
  (h1 : total = 63)
  (h2 : girls_ratio = 4)
  (h3 : boys_ratio = 3)
  (h4 : others_ratio = 2)
  (h5 : ∃ k : ℕ, total = k * (girls_ratio + boys_ratio + others_ratio)) :
  ∃ num_boys : ℕ, num_boys = 21 ∧ num_boys * (girls_ratio + boys_ratio + others_ratio) = boys_ratio * total :=
by
  sorry

#check boys_in_class

end boys_in_class_l1069_106974


namespace simplify_and_rationalize_l1069_106930

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 7 / Real.sqrt 11) * (Real.sqrt 15 / Real.sqrt 2) = 
  (3 * Real.sqrt 154) / 22 := by
sorry

end simplify_and_rationalize_l1069_106930


namespace f_composition_l1069_106947

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_composition : ∀ x : ℝ, f (f x) = 4 * x + 3 := by sorry

end f_composition_l1069_106947


namespace circle_constraint_extrema_l1069_106976

theorem circle_constraint_extrema :
  ∀ x y : ℝ, x^2 + y^2 = 1 →
  (∀ a b : ℝ, a^2 + b^2 = 1 → (1 + x*y)*(1 - x*y) ≤ (1 + a*b)*(1 - a*b)) ∧
  (∃ a b : ℝ, a^2 + b^2 = 1 ∧ (1 + a*b)*(1 - a*b) = 1) ∧
  (∀ a b : ℝ, a^2 + b^2 = 1 → (1 + a*b)*(1 - a*b) ≥ 3/4) ∧
  (∃ a b : ℝ, a^2 + b^2 = 1 ∧ (1 + a*b)*(1 - a*b) = 3/4) :=
by sorry

end circle_constraint_extrema_l1069_106976


namespace train_crossing_time_l1069_106941

/-- Proves that a train crosses a man in 18 seconds given its speed and time to cross a platform. -/
theorem train_crossing_time (train_speed : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed = 20 →
  platform_length = 340 →
  platform_crossing_time = 35 →
  (train_speed * platform_crossing_time - platform_length) / train_speed = 18 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1069_106941


namespace arithmetic_sequence_sum_ratio_l1069_106957

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- sum formula
  (a 8 / a 7 = 13 / 5) →                -- given condition
  (S 15 / S 13 = 3) :=                  -- conclusion to prove
by sorry

end arithmetic_sequence_sum_ratio_l1069_106957


namespace quadratic_root_zero_l1069_106951

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → m = 1 ∨ m = -1 := by
  sorry

end quadratic_root_zero_l1069_106951


namespace least_subtraction_for_divisibility_l1069_106932

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(10 ∣ (724946 - y))) ∧ 
  (10 ∣ (724946 - x)) := by
  sorry

end least_subtraction_for_divisibility_l1069_106932


namespace complex_equation_solution_l1069_106944

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l1069_106944


namespace virginia_adrienne_difference_l1069_106949

/-- The combined total years of teaching for Virginia, Adrienne, and Dennis -/
def total_years : ℕ := 93

/-- The number of years Dennis has taught -/
def dennis_years : ℕ := 40

/-- The number of years Virginia has taught -/
def virginia_years : ℕ := dennis_years - 9

/-- The number of years Adrienne has taught -/
def adrienne_years : ℕ := total_years - dennis_years - virginia_years

/-- Theorem stating the difference in teaching years between Virginia and Adrienne -/
theorem virginia_adrienne_difference : virginia_years - adrienne_years = 9 := by
  sorry

end virginia_adrienne_difference_l1069_106949


namespace paco_cookies_l1069_106909

def cookie_problem (initial_cookies : ℕ) (given_to_friend1 : ℕ) (given_to_friend2 : ℕ) (eaten : ℕ) : Prop :=
  let total_given := given_to_friend1 + given_to_friend2
  eaten - total_given = 0

theorem paco_cookies : cookie_problem 100 15 25 40 := by
  sorry

end paco_cookies_l1069_106909


namespace harmonious_equations_have_real_roots_l1069_106956

/-- A harmonious equation is a quadratic equation ax² + bx + c = 0 where a ≠ 0 and b = a + c -/
def HarmoniousEquation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b = a + c

/-- The discriminant of a quadratic equation ax² + bx + c = 0 -/
def Discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

/-- A quadratic equation has real roots if and only if its discriminant is non-negative -/
def HasRealRoots (a b c : ℝ) : Prop :=
  Discriminant a b c ≥ 0

/-- Theorem: Harmonious equations always have real roots -/
theorem harmonious_equations_have_real_roots (a b c : ℝ) :
  HarmoniousEquation a b c → HasRealRoots a b c :=
by sorry

end harmonious_equations_have_real_roots_l1069_106956


namespace extremum_and_range_l1069_106921

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x + b

-- Theorem statement
theorem extremum_and_range :
  ∀ a b : ℝ,
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a b x ≥ f a b 2) ∧
  (f a b 2 = -8) →
  (a = 12 ∧ b = 8) ∧
  (∀ x ∈ Set.Icc (-3) 3, -8 ≤ f 12 8 x ∧ f 12 8 x ≤ 24) :=
by sorry

end extremum_and_range_l1069_106921


namespace sqrt_sum_simplification_l1069_106997

theorem sqrt_sum_simplification : ∃ (a b c : ℕ+),
  (Real.sqrt 8 + (Real.sqrt 8)⁻¹ + Real.sqrt 9 + (Real.sqrt 9)⁻¹ = (a * Real.sqrt 8 + b * Real.sqrt 9) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 8 + (Real.sqrt 8)⁻¹ + Real.sqrt 9 + (Real.sqrt 9)⁻¹ = (a' * Real.sqrt 8 + b' * Real.sqrt 9) / c') →
    c ≤ c') ∧
  (a + b + c = 158) :=
sorry

end sqrt_sum_simplification_l1069_106997


namespace factorization_proof_l1069_106984

theorem factorization_proof (a : ℝ) : (2*a + 1)*a - 4*a - 2 = (2*a + 1)*(a - 2) := by
  sorry

end factorization_proof_l1069_106984


namespace quadratic_inequality_solution_l1069_106993

theorem quadratic_inequality_solution (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end quadratic_inequality_solution_l1069_106993


namespace money_division_l1069_106967

theorem money_division (a_share b_share c_share total : ℝ) : 
  (b_share = 0.65 * a_share) →
  (c_share = 0.4 * a_share) →
  (c_share = 32) →
  (total = a_share + b_share + c_share) →
  total = 164 := by
sorry

end money_division_l1069_106967


namespace catering_weight_calculation_catering_weight_proof_l1069_106900

theorem catering_weight_calculation (silverware_weight : ℕ) (silverware_per_setting : ℕ)
  (plate_weight : ℕ) (plates_per_setting : ℕ) (num_tables : ℕ) (settings_per_table : ℕ)
  (backup_settings : ℕ) : ℕ :=
  let weight_per_setting := silverware_weight * silverware_per_setting + plate_weight * plates_per_setting
  let total_settings := num_tables * settings_per_table + backup_settings
  weight_per_setting * total_settings

theorem catering_weight_proof :
  catering_weight_calculation 4 3 12 2 15 8 20 = 5040 := by
  sorry

end catering_weight_calculation_catering_weight_proof_l1069_106900


namespace factor_calculation_l1069_106979

theorem factor_calculation (x : ℝ) (f : ℝ) : 
  x = 22.142857142857142 → 
  ((x + 5) * f / 5) - 5 = 66 / 2 → 
  f = 7 := by
  sorry

end factor_calculation_l1069_106979


namespace initial_stuffed_animals_l1069_106918

def stuffed_animals (x : ℕ) : Prop :=
  let after_mom := x + 2
  let from_dad := 3 * after_mom
  x + 2 + from_dad = 48

theorem initial_stuffed_animals :
  ∃ (x : ℕ), stuffed_animals x ∧ x = 10 :=
by
  sorry

end initial_stuffed_animals_l1069_106918


namespace diagonal_intersects_n_rhombuses_l1069_106969

/-- A regular hexagon with side length n -/
structure RegularHexagon (n : ℕ) where
  side_length : ℕ
  is_positive : 0 < side_length
  eq_n : side_length = n

/-- A rhombus with internal angles 60° and 120° -/
structure Rhombus where
  internal_angles : Fin 2 → ℝ
  angle_sum : internal_angles 0 + internal_angles 1 = 180
  angles_correct : (internal_angles 0 = 60 ∧ internal_angles 1 = 120) ∨ 
                   (internal_angles 0 = 120 ∧ internal_angles 1 = 60)

/-- Theorem: The diagonal of a regular hexagon intersects n rhombuses -/
theorem diagonal_intersects_n_rhombuses (n : ℕ) (h : RegularHexagon n) :
  ∃ (rhombuses : Finset Rhombus),
    (Finset.card rhombuses = 3 * n^2) ∧
    (∃ (intersected : Finset Rhombus),
      Finset.card intersected = n ∧
      intersected ⊆ rhombuses) := by
  sorry

end diagonal_intersects_n_rhombuses_l1069_106969


namespace complex_modulus_problem_l1069_106999

theorem complex_modulus_problem (z : ℂ) :
  (1 + Complex.I * Real.sqrt 3)^2 * z = 1 - Complex.I^3 →
  Complex.abs z = Real.sqrt 2 / 4 := by
sorry

end complex_modulus_problem_l1069_106999


namespace modulo_eleven_residue_l1069_106948

theorem modulo_eleven_residue : (312 - 3 * 52 + 9 * 165 + 6 * 22) % 11 = 2 := by
  sorry

end modulo_eleven_residue_l1069_106948


namespace line_through_points_l1069_106994

-- Define the points
def P₁ : ℝ × ℝ := (3, -1)
def P₂ : ℝ × ℝ := (-2, 1)

-- Define the slope-intercept form
def slope_intercept (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

-- Theorem statement
theorem line_through_points : 
  ∃ (m b : ℝ), m = -2/5 ∧ b = 1/5 ∧ 
  (slope_intercept m b P₁.1 P₁.2 ∧ slope_intercept m b P₂.1 P₂.2) :=
sorry

end line_through_points_l1069_106994


namespace equilateral_triangle_vertices_product_l1069_106931

theorem equilateral_triangle_vertices_product (a b : ℝ) : 
  (∀ z : ℂ, z^3 = 1 ∧ z ≠ 1 → (a + 18 * I) * z = b + 42 * I) →
  a * b = -2652 := by
  sorry

end equilateral_triangle_vertices_product_l1069_106931


namespace unique_solution_l1069_106985

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 3 = x + y + 2

/-- The main theorem stating that the function f(x) = x + 3 is the unique solution -/
theorem unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x + 3 := by
  sorry

end unique_solution_l1069_106985


namespace right_triangle_division_l1069_106981

theorem right_triangle_division (n : ℝ) (h : n > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    ∃ (x y : ℝ),
      0 < x ∧ x < c ∧
      0 < y ∧ y < c ∧
      x * y = a * b ∧
      (1/2) * x * a = n * x * y ∧
      (1/2) * y * b = (1/(4*n)) * x * y :=
sorry

end right_triangle_division_l1069_106981


namespace tangent_circles_bisector_l1069_106914

-- Define the basic geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

def Point := ℝ × ℝ

-- Define the geometric relations
def tangentCircles (c1 c2 : Circle) (p : Point) : Prop := sorry

def tangentLineToCircle (l : Line) (c : Circle) (a : Point) : Prop := sorry

def lineIntersectsCircle (l : Line) (c : Circle) (b c : Point) : Prop := sorry

def angleBisector (l : Line) (a b c : Point) : Prop := sorry

-- State the theorem
theorem tangent_circles_bisector
  (c1 c2 : Circle) (p a b c : Point) (d : Line) :
  tangentCircles c1 c2 p →
  tangentLineToCircle d c1 a →
  lineIntersectsCircle d c2 b c →
  angleBisector (Line.mk p a) p b c := by
  sorry

end tangent_circles_bisector_l1069_106914


namespace system_solution_l1069_106923

theorem system_solution :
  ∃ (x y : ℚ), (7 * x = -5 - 3 * y) ∧ (4 * x = 5 * y - 36) ∧ (x = -41/11) ∧ (y = 232/33) := by
  sorry

end system_solution_l1069_106923


namespace f_properties_l1069_106926

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end f_properties_l1069_106926


namespace base_conversion_sum_l1069_106945

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- Represents the digit C in base 13 -/
def C : Nat := 12

theorem base_conversion_sum : 
  let base_5_num := to_base_10 [2, 4, 3] 5
  let base_13_num := to_base_10 [9, C, 2] 13
  base_5_num + base_13_num = 600 := by sorry

end base_conversion_sum_l1069_106945


namespace range_of_m_l1069_106962

/-- The function f(x) = x^2 - 4x + 5 -/
def f (x : ℝ) := x^2 - 4*x + 5

/-- The maximum value of f on [0, m] is 5 -/
def max_value := 5

/-- The minimum value of f on [0, m] is 1 -/
def min_value := 1

/-- The range of m for which f has max_value and min_value on [0, m] -/
theorem range_of_m :
  ∃ (m : ℝ), m ∈ Set.Icc 2 4 ∧
  (∀ x ∈ Set.Icc 0 m, f x ≤ max_value) ∧
  (∃ x ∈ Set.Icc 0 m, f x = max_value) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ min_value) ∧
  (∃ x ∈ Set.Icc 0 m, f x = min_value) ∧
  (∀ m' > 4, ∃ x ∈ Set.Icc 0 m', f x > max_value) ∧
  (∀ m' < 2, ∀ x ∈ Set.Icc 0 m', f x > min_value) :=
by sorry

end range_of_m_l1069_106962


namespace probability_two_face_cards_total_20_l1069_106915

-- Define the deck
def deck_size : ℕ := 52

-- Define the number of face cards
def face_cards : ℕ := 12

-- Define the value of a face card
def face_card_value : ℕ := 10

-- Define the theorem
theorem probability_two_face_cards_total_20 :
  (face_cards : ℚ) * (face_cards - 1) / (deck_size * (deck_size - 1)) = 11 / 221 :=
sorry

end probability_two_face_cards_total_20_l1069_106915


namespace min_triangles_to_cover_l1069_106961

/-- The minimum number of unit equilateral triangles needed to cover a larger equilateral triangle and a square -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) (square_side : ℝ) :
  small_side = 1 →
  large_side = 12 →
  square_side = 4 →
  ∃ (n : ℕ), n = ⌈145 * Real.sqrt 3 + 64⌉ ∧
    n * (Real.sqrt 3 / 4 * small_side^2) ≥
      (Real.sqrt 3 / 4 * large_side^2) + square_side^2 :=
by sorry

end min_triangles_to_cover_l1069_106961


namespace maciek_purchase_l1069_106905

/-- The cost of a pack of pretzels in dollars -/
def pretzel_cost : ℚ := 4

/-- The cost of a pack of chips in dollars -/
def chip_cost : ℚ := pretzel_cost * (1 + 3/4)

/-- The total amount Maciek spent in dollars -/
def total_spent : ℚ := 22

/-- The number of packets of each type (chips and pretzels) that Maciek bought -/
def num_packets : ℚ := total_spent / (pretzel_cost + chip_cost)

theorem maciek_purchase : num_packets = 2 := by
  sorry

end maciek_purchase_l1069_106905


namespace bus_tour_tickets_l1069_106959

/-- Represents the total number of tickets sold in a local bus tour. -/
def total_tickets (senior_tickets : ℕ) (regular_tickets : ℕ) : ℕ :=
  senior_tickets + regular_tickets

/-- Represents the total sales from tickets. -/
def total_sales (senior_tickets : ℕ) (regular_tickets : ℕ) : ℕ :=
  10 * senior_tickets + 15 * regular_tickets

theorem bus_tour_tickets :
  ∃ (senior_tickets : ℕ),
    total_tickets senior_tickets 41 = 65 ∧
    total_sales senior_tickets 41 = 855 :=
by sorry

end bus_tour_tickets_l1069_106959


namespace sum_of_squares_204_l1069_106927

theorem sum_of_squares_204 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℕ+) 
  (h : a₁^2 + (2*a₂)^2 + (3*a₃)^2 + (4*a₄)^2 + (5*a₅)^2 + (6*a₆)^2 + (7*a₇)^2 + (8*a₈)^2 = 204) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 8 := by
sorry

end sum_of_squares_204_l1069_106927


namespace quadratic_expression_value_l1069_106998

theorem quadratic_expression_value (a b c : ℝ) 
  (h1 : a - b = 2 + Real.sqrt 3)
  (h2 : b - c = 2 - Real.sqrt 3) :
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 15 := by
sorry

end quadratic_expression_value_l1069_106998


namespace extreme_value_point_property_l1069_106977

/-- The function f(x) = x³ - x² + ax - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x - a

/-- Theorem: If f(x) has an extreme value point x₀ and f(x₁) = f(x₀) where x₁ ≠ x₀, then x₁ + 2x₀ = 1 -/
theorem extreme_value_point_property (a : ℝ) (x₀ x₁ : ℝ) 
  (h_extreme : ∃ ε > 0, ∀ x, |x - x₀| < ε → f a x ≤ f a x₀ ∨ f a x ≥ f a x₀)
  (h_equal : f a x₁ = f a x₀)
  (h_distinct : x₁ ≠ x₀) : 
  x₁ + 2*x₀ = 1 := by
  sorry

end extreme_value_point_property_l1069_106977


namespace simplify_expression_l1069_106907

theorem simplify_expression (m : ℝ) (hm : m > 0) :
  (m^(1/2) * 3*m * 4*m) / ((6*m)^5 * m^(1/4)) = 1 := by
  sorry

end simplify_expression_l1069_106907


namespace chess_tournament_win_loss_difference_l1069_106988

theorem chess_tournament_win_loss_difference 
  (total_games : ℕ) 
  (total_score : ℚ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (draws : ℕ) :
  total_games = 42 →
  total_score = 30 →
  wins + losses + draws = total_games →
  (wins : ℚ) + (1/2 : ℚ) * draws = total_score →
  wins - losses = 18 :=
by sorry

end chess_tournament_win_loss_difference_l1069_106988


namespace f_derivatives_l1069_106992

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x - 1

-- Theorem statement
theorem f_derivatives :
  (deriv f 2 = 0) ∧ (deriv f 1 = -1) := by
  sorry

end f_derivatives_l1069_106992


namespace total_ice_cream_scoops_l1069_106950

def single_cone : ℕ := 1
def double_cone : ℕ := 3
def milkshake : ℕ := 2  -- Rounded up from 1.5
def banana_split : ℕ := 4 * single_cone
def waffle_bowl : ℕ := banana_split + 2
def ice_cream_sandwich : ℕ := waffle_bowl - 3

theorem total_ice_cream_scoops : 
  single_cone + double_cone + milkshake + banana_split + waffle_bowl + ice_cream_sandwich = 19 := by
  sorry

end total_ice_cream_scoops_l1069_106950


namespace f_increasing_iff_three_distinct_roots_iff_l1069_106960

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * |2 * a - x| + 2 * x

-- Theorem 1: f(x) is increasing on ℝ iff -1 ≤ a ≤ 1
theorem f_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 2: f(x) - t f(2a) = 0 has 3 distinct real roots iff 1 < t < 9/8
theorem three_distinct_roots_iff (a t : ℝ) :
  (a ∈ Set.Icc (-2) 2) →
  (∃ x y z : ℝ, x < y ∧ y < z ∧ f a x = t * f a (2 * a) ∧ f a y = t * f a (2 * a) ∧ f a z = t * f a (2 * a)) ↔
  (1 < t ∧ t < 9/8) :=
sorry

end f_increasing_iff_three_distinct_roots_iff_l1069_106960


namespace james_rainwater_profit_l1069_106904

/-- Calculates the money James made from selling rainwater collected over two days -/
theorem james_rainwater_profit : 
  let gallons_per_inch : ℝ := 15
  let monday_rain : ℝ := 4
  let tuesday_rain : ℝ := 3
  let price_per_gallon : ℝ := 1.2
  let total_gallons := gallons_per_inch * (monday_rain + tuesday_rain)
  total_gallons * price_per_gallon = 126 := by
sorry


end james_rainwater_profit_l1069_106904


namespace complex_equation_solution_l1069_106996

theorem complex_equation_solution (Z : ℂ) (i : ℂ) : 
  i * i = -1 → Z = (2 - Z) * i → Z = 1 + i := by sorry

end complex_equation_solution_l1069_106996


namespace curve_C_properties_l1069_106928

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - t) + p.2^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for C to be an ellipse with foci on the X-axis
def is_ellipse_x_axis (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

-- State the theorem
theorem curve_C_properties :
  ∀ t : ℝ,
    (is_hyperbola t ↔ ∃ a b : ℝ, C t = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) ∧
    (is_ellipse_x_axis t ↔ ∃ a b : ℝ, C t = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ a > b) :=
by sorry

end curve_C_properties_l1069_106928


namespace borrowed_sum_l1069_106973

/-- Proves that given the conditions of the problem, the principal amount is 1050 --/
theorem borrowed_sum (P : ℝ) : 
  (P * 0.06 * 6 = P - 672) → P = 1050 := by
  sorry

end borrowed_sum_l1069_106973


namespace complement_of_union_equals_singleton_l1069_106971

def U : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6}
def A : Finset Int := {-1, 0, 1, 2, 3}
def B : Finset Int := {-2, 3, 4, 5, 6}

theorem complement_of_union_equals_singleton : U \ (A ∪ B) = {-3} := by
  sorry

end complement_of_union_equals_singleton_l1069_106971


namespace set_contains_all_integers_l1069_106953

def is_closed_under_subtraction (A : Set ℤ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A

theorem set_contains_all_integers (A : Set ℤ) 
  (h_closed : is_closed_under_subtraction A) 
  (h_four : 4 ∈ A) 
  (h_nine : 9 ∈ A) : 
  A = Set.univ :=
sorry

end set_contains_all_integers_l1069_106953


namespace arithmetic_progression_logarithm_l1069_106943

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the arithmetic progression property
def isArithmeticProgression (a b c : ℝ) : Prop := 2 * b = a + c

-- Theorem statement
theorem arithmetic_progression_logarithm :
  isArithmeticProgression (lg 3) (lg 6) (lg x) → x = 12 :=
by sorry

end arithmetic_progression_logarithm_l1069_106943


namespace number_of_green_balls_l1069_106986

/-- Given a total of 40 balls with red, blue, and green colors, where there are 11 blue balls
and the number of red balls is twice the number of blue balls, prove that there are 7 green balls. -/
theorem number_of_green_balls (total : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) : 
  total = 40 →
  blue = 11 →
  red = 2 * blue →
  total = red + blue + green →
  green = 7 := by
sorry

end number_of_green_balls_l1069_106986


namespace parabola_line_intersection_slope_l1069_106964

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- A line with slope k passing through the focus -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The distance between two points on the parabola -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ + x₂ + 2

theorem parabola_line_intersection_slope 
  (k : ℝ) 
  (h_k : k ≠ 0) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_A : parabola x₁ y₁) 
  (h_B : parabola x₂ y₂) 
  (h_l₁ : line k x₁ y₁) 
  (h_l₂ : line k x₂ y₂) 
  (h_dist : distance x₁ y₁ x₂ y₂ = 5) : 
  k = 2 ∨ k = -2 :=
sorry

end parabola_line_intersection_slope_l1069_106964


namespace regular_polygon_160_degrees_has_18_sides_l1069_106911

/-- A regular polygon with interior angles measuring 160° has 18 sides. -/
theorem regular_polygon_160_degrees_has_18_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (180 * (n - 2) : ℝ) / n = 160 →
  n = 18 := by
sorry

end regular_polygon_160_degrees_has_18_sides_l1069_106911


namespace even_function_implies_m_zero_l1069_106995

def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + 4

theorem even_function_implies_m_zero (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 0 := by
  sorry

end even_function_implies_m_zero_l1069_106995


namespace triangle_abc_properties_l1069_106908

-- Define the triangle vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 4)
def C : ℝ × ℝ := (-2, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the line equation 4x + 3y + m = 0
def line_equation (x y m : ℝ) : Prop := 4 * x + 3 * y + m = 0

-- Define the circumcircle equation (x-3)^2 + (y-4)^2 = 25
def circumcircle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Define the chord length
def chord_length : ℝ := 6

theorem triangle_abc_properties :
  (dot_product AB AC = 0) ∧ 
  (∃ m : ℝ, (m = -4 ∨ m = -44) ∧
    ∃ x y : ℝ, line_equation x y m ∧ circumcircle_equation x y) :=
sorry

end triangle_abc_properties_l1069_106908


namespace unique_triple_lcm_l1069_106972

theorem unique_triple_lcm : 
  ∃! (x y z : ℕ+), 
    Nat.lcm x y = 180 ∧ 
    Nat.lcm x z = 420 ∧ 
    Nat.lcm y z = 1260 := by
  sorry

end unique_triple_lcm_l1069_106972


namespace frustum_radius_l1069_106934

theorem frustum_radius (r : ℝ) 
  (h1 : (2 * π * (3 * r)) / (2 * π * r) = 3)
  (h2 : 3 = 3)  -- slant height
  (h3 : π * (r + 3 * r) * 3 = 84 * π) : r = 7 := by
  sorry

end frustum_radius_l1069_106934


namespace p_sufficient_not_necessary_l1069_106966

-- Define propositions p and q
variable (p q : Prop)

-- Define the given conditions
variable (h1 : p → q)
variable (h2 : ¬(¬p → ¬q))

-- State the theorem
theorem p_sufficient_not_necessary :
  (∃ (r : Prop), r → q) ∧ ¬(q → p) :=
sorry

end p_sufficient_not_necessary_l1069_106966


namespace union_of_sets_l1069_106924

theorem union_of_sets : 
  let M : Set ℤ := {4, -3}
  let N : Set ℤ := {0, -3}
  M ∪ N = {0, -3, 4} := by sorry

end union_of_sets_l1069_106924


namespace loan_problem_l1069_106970

/-- Proves that given the conditions of the loan problem, the time A lent money to C is 2/3 of a year. -/
theorem loan_problem (principal_B principal_C total_interest : ℚ) 
  (time_B : ℚ) (rate : ℚ) :
  principal_B = 5000 →
  principal_C = 3000 →
  time_B = 2 →
  rate = 9 / 100 →
  total_interest = 1980 →
  total_interest = principal_B * rate * time_B + principal_C * rate * (2 / 3) :=
by sorry

end loan_problem_l1069_106970


namespace length_to_height_ratio_l1069_106989

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def volume (b : BoxDimensions) : ℝ :=
  b.height * b.width * b.length

/-- Theorem stating the ratio of length to height for a specific box -/
theorem length_to_height_ratio (b : BoxDimensions) :
  b.height = 12 →
  b.length = 4 * b.width →
  volume b = 3888 →
  b.length / b.height = 3 := by
  sorry

end length_to_height_ratio_l1069_106989


namespace unequal_outcome_probability_l1069_106980

def num_grandchildren : ℕ := 10

theorem unequal_outcome_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_outcomes := Nat.choose num_grandchildren (num_grandchildren / 2)
  (total_outcomes - equal_outcomes) / total_outcomes = 193 / 256 := by
  sorry

end unequal_outcome_probability_l1069_106980


namespace coffee_blend_cost_calculation_l1069_106920

/-- The cost of the coffee blend given the prices and amounts of two types of coffee. -/
def coffee_blend_cost (price_a price_b : ℝ) (amount_a : ℝ) : ℝ :=
  amount_a * price_a + 2 * amount_a * price_b

/-- Theorem stating the total cost of the coffee blend under given conditions. -/
theorem coffee_blend_cost_calculation :
  coffee_blend_cost 4.60 5.95 67.52 = 1114.08 := by
  sorry

end coffee_blend_cost_calculation_l1069_106920


namespace russian_number_sequence_next_two_elements_l1069_106975

/-- Represents the first letter of a Russian number word -/
inductive RussianNumberLetter
| O  -- Один (One)
| D  -- Два (Two)
| T  -- Три (Three)
| C  -- Четыре (Four)
| P  -- Пять (Five)
| S  -- Шесть (Six)
| S' -- Семь (Seven)
| V  -- Восемь (Eight)

/-- Returns the RussianNumberLetter for a given natural number -/
def russianNumberLetter (n : ℕ) : RussianNumberLetter :=
  match n with
  | 1 => RussianNumberLetter.O
  | 2 => RussianNumberLetter.D
  | 3 => RussianNumberLetter.T
  | 4 => RussianNumberLetter.C
  | 5 => RussianNumberLetter.P
  | 6 => RussianNumberLetter.S
  | 7 => RussianNumberLetter.S'
  | 8 => RussianNumberLetter.V
  | _ => RussianNumberLetter.O  -- Default case, should not be reached for 1-8

theorem russian_number_sequence_next_two_elements :
  russianNumberLetter 7 = RussianNumberLetter.S' ∧
  russianNumberLetter 8 = RussianNumberLetter.V :=
by sorry

end russian_number_sequence_next_two_elements_l1069_106975


namespace train_speed_calculation_l1069_106935

/-- 
Proves that the speed of a train is 72 km/hr, given its length, 
the platform length it crosses, and the time it takes to cross the platform.
-/
theorem train_speed_calculation (train_length platform_length crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 220)
  (h3 : crossing_time = 26) :
  (train_length + platform_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1069_106935


namespace x1_value_l1069_106965

theorem x1_value (x1 x2 x3 x4 : Real) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 := by sorry

end x1_value_l1069_106965


namespace unknown_number_proof_l1069_106983

theorem unknown_number_proof :
  let N : ℕ := 15222392625570
  let a : ℕ := 1155
  let b : ℕ := 1845
  let product : ℕ := a * b
  let difference : ℕ := b - a
  let quotient : ℕ := 15 * (difference * difference)
  N / product = quotient ∧ N % product = 570 :=
by sorry

end unknown_number_proof_l1069_106983


namespace functional_equation_solution_l1069_106919

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^2 * g a = a^2 * g c

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) (h2 : g 3 ≠ 0) : 
  (g 6 - g 2) / g 3 = 32 / 9 := by sorry

end functional_equation_solution_l1069_106919


namespace english_only_enrollment_l1069_106955

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 32)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : total ≥ german)
  (h5 : german ≥ both) :
  total - german + both = 10 := by
  sorry

end english_only_enrollment_l1069_106955


namespace circle_E_equation_line_circle_disjoint_l1069_106916

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define point D
def point_D : ℝ × ℝ := (-2, 0)

-- Define line l passing through D with slope k
def line_l (k x y : ℝ) : Prop := y = k * (x + 2)

-- Theorem for the equation of circle E
theorem circle_E_equation : ∀ x y : ℝ,
  (∃ k : ℝ, line_l k x y) →
  ((x + 1)^2 + (y - 2)^2 = 5) ↔ 
  (∃ t : ℝ, x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 4 * t) :=
sorry

-- Theorem for the condition of line l and circle C being disjoint
theorem line_circle_disjoint : ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → ¬circle_C x y) ↔ k < 3/4 :=
sorry

end circle_E_equation_line_circle_disjoint_l1069_106916


namespace early_arrival_l1069_106952

/-- A boy's journey to school -/
def school_journey (usual_time : ℕ) (speed_factor : ℚ) : Prop :=
  let new_time := (usual_time : ℚ) * (1 / speed_factor)
  (usual_time : ℚ) - new_time = 7

theorem early_arrival : school_journey 49 (7/6) := by
  sorry

end early_arrival_l1069_106952


namespace phone_not_answered_probability_l1069_106901

theorem phone_not_answered_probability 
  (p1 p2 p3 p4 : ℝ) 
  (h1 : p1 = 0.1) 
  (h2 : p2 = 0.3) 
  (h3 : p3 = 0.4) 
  (h4 : p4 = 0.1) : 
  1 - (p1 + p2 + p3 + p4) = 0.1 := by
  sorry

end phone_not_answered_probability_l1069_106901


namespace tire_usage_proof_l1069_106910

/-- Represents the number of miles each tire is used when a car with 6 tires
    travels 40,000 miles, with each tire being used equally. -/
def miles_per_tire : ℕ := 26667

/-- The total number of tires. -/
def total_tires : ℕ := 6

/-- The number of tires used on the road at any given time. -/
def road_tires : ℕ := 4

/-- The total distance traveled by the car in miles. -/
def total_distance : ℕ := 40000

theorem tire_usage_proof :
  miles_per_tire * total_tires = total_distance * road_tires :=
sorry

end tire_usage_proof_l1069_106910


namespace sqrt_sum_implies_product_l1069_106942

theorem sqrt_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) := by
  sorry

end sqrt_sum_implies_product_l1069_106942


namespace unique_acute_prime_angled_triangle_l1069_106922

-- Define a structure for a triangle with three angles
structure Triangle where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a triangle to be acute
def isAcute (t : Triangle) : Prop := t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Define what it means for a triangle to have prime angles
def hasPrimeAngles (t : Triangle) : Prop := 
  isPrime t.angle1 ∧ isPrime t.angle2 ∧ isPrime t.angle3

-- Define what it means for a triangle to be valid (sum of angles is 180°)
def isValidTriangle (t : Triangle) : Prop := t.angle1 + t.angle2 + t.angle3 = 180

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := 
  t.angle1 = t.angle2 ∨ t.angle2 = t.angle3 ∨ t.angle3 = t.angle1

-- Theorem statement
theorem unique_acute_prime_angled_triangle : 
  ∃! t : Triangle, isAcute t ∧ hasPrimeAngles t ∧ isValidTriangle t ∧
  t.angle1 = 2 ∧ t.angle2 = 89 ∧ t.angle3 = 89 ∧ isIsosceles t :=
sorry

end unique_acute_prime_angled_triangle_l1069_106922


namespace unique_function_property_l1069_106936

theorem unique_function_property (f : ℕ → ℕ) : 
  (∀ (a b c : ℕ), (f a + f b + f c - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) → 
  (∀ (a : ℕ), f a = a ^ 2) := by
sorry

end unique_function_property_l1069_106936


namespace triangle_inequality_l1069_106940

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 1) :
    5 * (a^2 + b^2 + c^2) = 18 * a * b * c ∧ 18 * a * b * c ≥ 7/3 := by
  sorry

end triangle_inequality_l1069_106940


namespace coefficient_of_x_4_in_expansion_l1069_106982

def binomial_expansion (n : ℕ) (x : ℝ) : ℝ → ℝ := 
  fun a => (1 + a * x)^n

def coefficient_of_x_power (f : ℝ → ℝ) (n : ℕ) : ℝ := sorry

theorem coefficient_of_x_4_in_expansion : 
  coefficient_of_x_power (fun x => (1 + x^2) * binomial_expansion 5 (-2) x) 4 = 120 := by sorry

end coefficient_of_x_4_in_expansion_l1069_106982
