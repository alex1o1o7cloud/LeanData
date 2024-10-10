import Mathlib

namespace remainder_theorem_l79_7947

theorem remainder_theorem (T E N S E' N' S' : ℤ)
  (h1 : T = N * E + S)
  (h2 : N = N' * E' + S')
  : T % (E + E') = E * S' + S := by
  sorry

end remainder_theorem_l79_7947


namespace quadratic_polynomial_condition_l79_7957

/-- A second degree polynomial -/
structure QuadraticPolynomial where
  u : ℝ
  v : ℝ
  w : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.u * x^2 + p.v * x + p.w

theorem quadratic_polynomial_condition (p : QuadraticPolynomial) :
  (∀ a : ℝ, a ≥ 1 → p.eval (a^2 + a) ≥ a * p.eval (a + 1)) ↔
  (p.u > 0 ∧ p.w ≤ 4 * p.u) := by
  sorry

end quadratic_polynomial_condition_l79_7957


namespace cubic_function_monotonicity_l79_7919

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + x

/-- The derivative of f with respect to x -/
def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + 1

/-- The number of monotonic intervals of f -/
def monotonic_intervals (b : ℝ) : ℕ := sorry

theorem cubic_function_monotonicity (b : ℝ) :
  monotonic_intervals b = 3 → b ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ioi (Real.sqrt 3) :=
sorry

end cubic_function_monotonicity_l79_7919


namespace three_digit_number_puzzle_l79_7985

theorem three_digit_number_puzzle (A B : ℕ) : 
  (100 ≤ A * 100 + 30 + B) ∧ 
  (A * 100 + 30 + B < 1000) ∧ 
  (A * 100 + 30 + B - 41 = 591) → 
  B = 2 := by sorry

end three_digit_number_puzzle_l79_7985


namespace rain_on_tuesday_l79_7900

theorem rain_on_tuesday (rain_monday : ℝ) (rain_both : ℝ) (no_rain : ℝ) 
  (h1 : rain_monday = 0.6)
  (h2 : rain_both = 0.4)
  (h3 : no_rain = 0.25) :
  ∃ rain_tuesday : ℝ, rain_tuesday = 0.55 ∧ 
  rain_monday + rain_tuesday - rain_both + no_rain = 1 :=
by sorry

end rain_on_tuesday_l79_7900


namespace red_balls_count_l79_7945

theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  green = 18 ∧
  yellow = 5 ∧
  purple = 9 ∧
  prob = 3/4 ∧
  (white + green + yellow : ℚ) / total = prob →
  total - (white + green + yellow + purple) = 6 :=
by sorry

end red_balls_count_l79_7945


namespace binary_arithmetic_proof_l79_7981

theorem binary_arithmetic_proof : 
  let a : ℕ := 0b1100101
  let b : ℕ := 0b1101
  let c : ℕ := 0b101
  let result : ℕ := 0b11111010
  (a * b) / c = result := by sorry

end binary_arithmetic_proof_l79_7981


namespace min_value_x_plus_y_l79_7989

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 8/y = 1) :
  x + y ≥ 18 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 8/y = 1 ∧ x + y = 18 :=
sorry

end min_value_x_plus_y_l79_7989


namespace total_crayons_l79_7990

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 12) (h2 : num_children = 18) :
  crayons_per_child * num_children = 216 := by
  sorry

end total_crayons_l79_7990


namespace female_managers_count_l79_7942

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  male_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ
  female_managers : ℕ

/-- Conditions for the company -/
def ValidCompany (c : Company) : Prop :=
  c.female_employees = 1000 ∧
  c.total_employees = c.male_employees + c.female_employees ∧
  c.total_managers = c.male_managers + c.female_managers ∧
  5 * c.total_managers = 2 * c.total_employees ∧
  5 * c.male_managers = 2 * c.male_employees

/-- Theorem stating that in a valid company, the number of female managers is 400 -/
theorem female_managers_count (c : Company) (h : ValidCompany c) :
  c.female_managers = 400 := by
  sorry


end female_managers_count_l79_7942


namespace matrix_power_calculation_l79_7962

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_calculation :
  (2 • A)^10 = !![1024, 0; 20480, 1024] := by sorry

end matrix_power_calculation_l79_7962


namespace max_integer_difference_l79_7950

theorem max_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  ∃ (n : ℤ), n = ⌊y - x⌋ ∧ n ≤ 4 ∧ ∀ (m : ℤ), m = ⌊y - x⌋ → m ≤ n :=
by sorry

end max_integer_difference_l79_7950


namespace spider_web_paths_l79_7901

theorem spider_web_paths : Nat.choose 11 5 = 462 := by
  sorry

end spider_web_paths_l79_7901


namespace jaymee_is_22_l79_7977

def shara_age : ℕ := 10

def jaymee_age : ℕ := 2 * shara_age + 2

theorem jaymee_is_22 : jaymee_age = 22 := by
  sorry

end jaymee_is_22_l79_7977


namespace bisecting_plane_intersects_24_cubes_l79_7970

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ

/-- Represents a plane that bisects an internal diagonal of a cube -/
structure BisectingPlane where
  cube : LargeCube
  
/-- The number of unit cubes intersected by the bisecting plane -/
def intersected_cubes (plane : BisectingPlane) : ℕ := sorry

/-- Main theorem: A plane bisecting an internal diagonal of a 4x4x4 cube intersects 24 unit cubes -/
theorem bisecting_plane_intersects_24_cubes 
  (cube : LargeCube) 
  (plane : BisectingPlane) 
  (h1 : cube.side_length = 4) 
  (h2 : cube.total_cubes = 64) 
  (h3 : plane.cube = cube) :
  intersected_cubes plane = 24 := by sorry

end bisecting_plane_intersects_24_cubes_l79_7970


namespace joan_has_eight_kittens_l79_7959

/-- The number of kittens Joan has at the end, given the initial conditions and actions. -/
def joans_final_kittens (joan_initial : ℕ) (neighbor_initial : ℕ) 
  (joan_gave_away : ℕ) (neighbor_gave_away : ℕ) (joan_wants_to_adopt : ℕ) : ℕ :=
  let joan_after_giving := joan_initial - joan_gave_away
  let neighbor_after_giving := neighbor_initial - neighbor_gave_away
  let joan_can_adopt := min joan_wants_to_adopt neighbor_after_giving
  joan_after_giving + joan_can_adopt

/-- Theorem stating that Joan ends up with 8 kittens given the specific conditions. -/
theorem joan_has_eight_kittens : 
  joans_final_kittens 8 6 2 4 3 = 8 := by
  sorry

end joan_has_eight_kittens_l79_7959


namespace star_property_l79_7908

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the * operation
def star : Element → Element → Element
  | Element.one, x => x
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.five
  | Element.two, Element.five => Element.one
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.five
  | Element.three, Element.four => Element.one
  | Element.three, Element.five => Element.two
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.five
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.two
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.five
  | Element.five, Element.two => Element.one
  | Element.five, Element.three => Element.two
  | Element.five, Element.four => Element.three
  | Element.five, Element.five => Element.four

theorem star_property : 
  star (star Element.three Element.five) (star Element.two Element.four) = Element.one := by
  sorry

end star_property_l79_7908


namespace unique_solution_conditions_l79_7972

theorem unique_solution_conditions (n p : ℕ) :
  (∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p ^ z) ↔
  (p > 1 ∧ 
   (n - 1) % (p - 1) = 0 ∧
   ∀ k : ℕ, n ≠ p ^ k) :=
by sorry

end unique_solution_conditions_l79_7972


namespace continuous_at_3_l79_7935

def f (x : ℝ) : ℝ := -3 * x^2 - 9

theorem continuous_at_3 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end continuous_at_3_l79_7935


namespace smallest_multiple_l79_7975

theorem smallest_multiple (y : ℕ) : y = 32 ↔ 
  (y > 0 ∧ 
   900 * y % 1152 = 0 ∧ 
   ∀ z : ℕ, z > 0 → z < y → 900 * z % 1152 ≠ 0) := by
  sorry

end smallest_multiple_l79_7975


namespace train_length_l79_7997

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 122 →  -- speed in km/hr
  time = 4.425875438161669 →  -- time in seconds
  speed * (5 / 18) * time = 150 :=  -- length in meters
by sorry

end train_length_l79_7997


namespace extremum_values_l79_7941

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_values (a b : ℝ) :
  f_deriv a b 1 = 0 ∧ f a b 1 = 10 → a = 4 ∧ b = -11 := by
  sorry

end extremum_values_l79_7941


namespace max_ab_value_l79_7920

/-- Given a function f(x) = -a * ln(x) + (a+1)x - (1/2)x^2 where a > 0,
    if f(x) ≥ -(1/2)x^2 + ax + b holds for all x > 0,
    then the maximum value of ab is e/2 -/
theorem max_ab_value (a b : ℝ) (h_a : a > 0) :
  (∀ x > 0, -a * Real.log x + (a + 1) * x - (1/2) * x^2 ≥ -(1/2) * x^2 + a * x + b) →
  (∃ m : ℝ, m = Real.exp 1 / 2 ∧ a * b ≤ m ∧ ∀ c d : ℝ, c > 0 → (∀ x > 0, -c * Real.log x + (c + 1) * x - (1/2) * x^2 ≥ -(1/2) * x^2 + c * x + d) → c * d ≤ m) :=
by sorry

end max_ab_value_l79_7920


namespace f_extreme_values_l79_7963

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x ^ 2) + Real.sin (Real.sin x ^ 2)

theorem f_extreme_values (k : ℤ) :
  ∃ (x : ℝ), x = (k : ℝ) * (Real.pi / 4) ∧ 
  (∀ (y : ℝ), f y ≤ f x ∨ f y ≥ f x) :=
sorry

end f_extreme_values_l79_7963


namespace son_age_problem_l79_7938

theorem son_age_problem (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end son_age_problem_l79_7938


namespace power_mod_seventeen_l79_7964

theorem power_mod_seventeen : 4^2023 % 17 = 13 := by
  sorry

end power_mod_seventeen_l79_7964


namespace complement_of_A_is_closed_ray_l79_7954

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the domain of log(2-x)
def A : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem complement_of_A_is_closed_ray :
  Set.compl A = Set.Ici (2 : ℝ) := by sorry

end complement_of_A_is_closed_ray_l79_7954


namespace farm_feet_count_l79_7984

/-- Represents a farm with hens and cows -/
structure Farm where
  total_heads : ℕ
  num_hens : ℕ

/-- Calculates the total number of feet in the farm -/
def total_feet (f : Farm) : ℕ :=
  2 * f.num_hens + 4 * (f.total_heads - f.num_hens)

/-- Theorem stating that a farm with 48 total heads and 26 hens has 140 feet -/
theorem farm_feet_count : 
  ∀ (f : Farm), f.total_heads = 48 → f.num_hens = 26 → total_feet f = 140 := by
  sorry


end farm_feet_count_l79_7984


namespace true_discount_for_given_values_l79_7994

/-- Given a banker's discount and a sum due, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / sum_due)

/-- Theorem stating that for the given values, the true discount is 120 -/
theorem true_discount_for_given_values :
  true_discount 144 720 = 120 := by
  sorry

end true_discount_for_given_values_l79_7994


namespace triangle_perimeter_l79_7983

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angles and side lengths
def angle (t : Triangle) (v1 v2 v3 : ℝ × ℝ) : ℝ := sorry

def side_length (a b : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  side_length t.X t.Y + side_length t.Y t.Z + side_length t.Z t.X

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  angle t t.X t.Y t.Z = angle t t.X t.Z t.Y →
  side_length t.Y t.Z = 8 →
  side_length t.X t.Z = 10 →
  perimeter t = 28 := by
  sorry

end triangle_perimeter_l79_7983


namespace entire_line_purple_exactly_integers_purple_not_exactly_rationals_purple_l79_7948

-- Define the coloring function
def Coloring := ℝ → Bool

-- Define the property of being purple
def isPurple (c : Coloring) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ y z : ℝ, |x - y| < ε ∧ |x - z| < ε ∧ c y ≠ c z

-- Theorem for part a
theorem entire_line_purple :
  ∃ c : Coloring, ∀ x : ℝ, isPurple c x :=
sorry

-- Theorem for part b
theorem exactly_integers_purple :
  ∃ c : Coloring, ∀ x : ℝ, isPurple c x ↔ ∃ n : ℤ, x = n :=
sorry

-- Theorem for part c
theorem not_exactly_rationals_purple :
  ¬ ∃ c : Coloring, ∀ x : ℝ, isPurple c x ↔ ∃ q : ℚ, x = q :=
sorry

end entire_line_purple_exactly_integers_purple_not_exactly_rationals_purple_l79_7948


namespace factory_problem_l79_7979

/-- Represents a factory with workers and production methods -/
structure Factory where
  total_workers : ℕ
  production_increase : ℝ → ℝ
  new_method_factor : ℝ

/-- The conditions and proof goals for the factory problem -/
theorem factory_problem (f : Factory) : 
  (f.production_increase (40 / f.total_workers) = 1.2) →
  (f.production_increase 0.6 = 2.5) →
  (f.total_workers = 500 ∧ f.new_method_factor = 3.5) := by
  sorry


end factory_problem_l79_7979


namespace mean_temperature_l79_7960

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -9/7 := by
  sorry

end mean_temperature_l79_7960


namespace no_two_digit_product_equals_concatenation_l79_7907

theorem no_two_digit_product_equals_concatenation : ¬∃ (a b c d : ℕ), 
  (0 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (0 ≤ d ∧ d ≤ 9) ∧ 
  ((10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d) :=
by sorry

end no_two_digit_product_equals_concatenation_l79_7907


namespace mexico_city_car_restriction_l79_7926

/-- The minimum number of cars needed for a family in Mexico City -/
def min_cars : ℕ := 14

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of restricted days per car per week -/
def restricted_days_per_car : ℕ := 2

/-- The minimum number of cars that must be available each day -/
def min_available_cars : ℕ := 10

theorem mexico_city_car_restriction :
  ∀ n : ℕ,
  n ≥ min_cars →
  (∀ d : ℕ, d < days_in_week →
    n - (n * restricted_days_per_car / days_in_week) ≥ min_available_cars) ∧
  (∀ m : ℕ, m < min_cars →
    ∃ d : ℕ, d < days_in_week ∧
      m - (m * restricted_days_per_car / days_in_week) < min_available_cars) :=
by sorry


end mexico_city_car_restriction_l79_7926


namespace general_admission_tickets_l79_7991

theorem general_admission_tickets (student_price general_price : ℕ) 
  (total_tickets total_revenue : ℕ) : 
  student_price = 4 →
  general_price = 6 →
  total_tickets = 525 →
  total_revenue = 2876 →
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_tickets * student_price + general_tickets * general_price = total_revenue ∧
    general_tickets = 388 :=
by sorry

end general_admission_tickets_l79_7991


namespace simplify_radical_product_l79_7998

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (28 * x) * Real.sqrt (15 * x) * Real.sqrt (21 * x) = 42 * x * Real.sqrt (5 * x) := by
  sorry

end simplify_radical_product_l79_7998


namespace slope_intercept_sum_l79_7906

/-- Given points A, B, C, and F as the midpoint of AC, prove that the sum of the slope
    and y-intercept of the line passing through F and B is 3/4 -/
theorem slope_intercept_sum (A B C F : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 0) →
  C = (8, 0) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  let m := (F.2 - B.2) / (F.1 - B.1)
  let b := B.2
  m + b = 3/4 := by sorry

end slope_intercept_sum_l79_7906


namespace function_satisfies_equation_l79_7996

theorem function_satisfies_equation (x y : ℚ) (hx : 0 < x) (hy : 0 < y) :
  let f : ℚ → ℚ := λ t => 1 / (t^2)
  f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y) := by
  sorry

end function_satisfies_equation_l79_7996


namespace complex_modulus_problem_l79_7940

theorem complex_modulus_problem (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_problem_l79_7940


namespace sum_possible_constants_eq_1232_l79_7909

/-- 
Given a quadratic equation ax² + bx + c = 0 with two distinct negative integer roots,
where b = 24, this function computes the sum of all possible values for c.
-/
def sum_possible_constants : ℤ := by
  sorry

/-- The main theorem stating that the sum of all possible constant terms is 1232 -/
theorem sum_possible_constants_eq_1232 : sum_possible_constants = 1232 := by
  sorry

end sum_possible_constants_eq_1232_l79_7909


namespace polynomial_factorization_l79_7922

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 5*x^4 + 8*x^2 - 4 = (x-1)*(x+1)*(x^2-2)^2 := by
  sorry

end polynomial_factorization_l79_7922


namespace inequality_solution_set_l79_7993

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 4 * (a - 2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end inequality_solution_set_l79_7993


namespace factorial_300_trailing_zeros_l79_7934

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 300! is 74 -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by
  sorry

end factorial_300_trailing_zeros_l79_7934


namespace lcm_12_18_l79_7982

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l79_7982


namespace flu_probability_l79_7918

/-- The probability of a randomly selected person having the flu given the flu rates and population ratios for three areas -/
theorem flu_probability (flu_rate_A flu_rate_B flu_rate_C : ℝ) 
  (pop_ratio_A pop_ratio_B pop_ratio_C : ℕ) : 
  flu_rate_A = 0.06 →
  flu_rate_B = 0.05 →
  flu_rate_C = 0.04 →
  pop_ratio_A = 6 →
  pop_ratio_B = 5 →
  pop_ratio_C = 4 →
  (flu_rate_A * pop_ratio_A + flu_rate_B * pop_ratio_B + flu_rate_C * pop_ratio_C) / 
  (pop_ratio_A + pop_ratio_B + pop_ratio_C) = 77 / 1500 := by
sorry


end flu_probability_l79_7918


namespace rotated_angle_intersection_l79_7916

/-- 
Given an angle α, when its terminal side is rotated clockwise by π/2,
the intersection of the new angle with the unit circle centered at the origin
has coordinates (sin α, -cos α).
-/
theorem rotated_angle_intersection (α : Real) : 
  let rotated_angle := α - π / 2
  let x := Real.cos rotated_angle
  let y := Real.sin rotated_angle
  (x, y) = (Real.sin α, -Real.cos α) := by
sorry

end rotated_angle_intersection_l79_7916


namespace alphabetic_sequences_count_l79_7995

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the sequence -/
def sequence_length : ℕ := 2013

/-- The number of alphabetic sequences of given length with letters in alphabetic order -/
def alphabetic_sequences (n : ℕ) : ℕ := Nat.choose (n + alphabet_size - 1) (alphabet_size - 1)

theorem alphabetic_sequences_count : 
  alphabetic_sequences sequence_length = Nat.choose 2038 25 := by sorry

end alphabetic_sequences_count_l79_7995


namespace probability_is_two_thirty_thirds_l79_7944

/-- A square with side length 3 and 12 equally spaced points on its perimeter -/
structure SquareWithPoints where
  side_length : ℝ
  num_points : ℕ
  points_per_side : ℕ

/-- The probability of selecting two points that are one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  4 / (s.num_points.choose 2)

/-- The main theorem stating the probability is 2/33 -/
theorem probability_is_two_thirty_thirds :
  let s : SquareWithPoints := {
    side_length := 3,
    num_points := 12,
    points_per_side := 3
  }
  probability_one_unit_apart s = 2 / 33 := by
  sorry

end probability_is_two_thirty_thirds_l79_7944


namespace sequence_sum_divisible_by_five_l79_7956

/-- Represents a four-digit integer -/
structure FourDigitInt where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Represents a sequence of four FourDigitInts with the given property -/
structure SpecialSequence where
  term1 : FourDigitInt
  term2 : FourDigitInt
  term3 : FourDigitInt
  term4 : FourDigitInt
  property : term2.hundreds = term1.tens ∧ term2.tens = term1.units ∧
             term3.hundreds = term2.tens ∧ term3.tens = term2.units ∧
             term4.hundreds = term3.tens ∧ term4.tens = term3.units ∧
             term1.hundreds = term4.tens ∧ term1.tens = term4.units

/-- Calculates the sum of all terms in the sequence -/
def sequenceSum (seq : SpecialSequence) : Nat :=
  let toNum (t : FourDigitInt) := t.thousands * 1000 + t.hundreds * 100 + t.tens * 10 + t.units
  toNum seq.term1 + toNum seq.term2 + toNum seq.term3 + toNum seq.term4

theorem sequence_sum_divisible_by_five (seq : SpecialSequence) :
  ∃ k : Nat, sequenceSum seq = 5 * k := by
  sorry


end sequence_sum_divisible_by_five_l79_7956


namespace diameter_endpoint2_coordinates_l79_7910

def circle_center : ℝ × ℝ := (1, 2)
def diameter_endpoint1 : ℝ × ℝ := (4, 6)

theorem diameter_endpoint2_coordinates :
  let midpoint := circle_center
  let endpoint1 := diameter_endpoint1
  let endpoint2 := (2 * midpoint.1 - endpoint1.1, 2 * midpoint.2 - endpoint1.2)
  endpoint2 = (-2, -2) :=
sorry

end diameter_endpoint2_coordinates_l79_7910


namespace cube_sum_theorem_l79_7933

theorem cube_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 8)
  (h2 : a * b + a * c + b * c = 9)
  (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 := by
sorry

end cube_sum_theorem_l79_7933


namespace polygon_sides_l79_7967

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (2 : ℚ) / 9 * ((n - 2) * 180) = 360 → n = 11 :=
by
  sorry

end polygon_sides_l79_7967


namespace tom_rides_11860_miles_l79_7932

/-- Tom's daily bike riding distance for the first part of the year -/
def first_part_distance : ℕ := 30

/-- Number of days in the first part of the year -/
def first_part_days : ℕ := 183

/-- Tom's daily bike riding distance for the second part of the year -/
def second_part_distance : ℕ := 35

/-- Total number of days in a year -/
def total_days : ℕ := 365

/-- Calculate the total miles Tom rides in a year -/
def total_miles : ℕ := first_part_distance * first_part_days + 
                        second_part_distance * (total_days - first_part_days)

theorem tom_rides_11860_miles : total_miles = 11860 := by
  sorry

end tom_rides_11860_miles_l79_7932


namespace two_numbers_squares_sum_cube_cubes_sum_square_l79_7958

theorem two_numbers_squares_sum_cube_cubes_sum_square :
  ∃ (a b : ℕ), a ≠ b ∧ a > 0 ∧ b > 0 ∧
  (∃ (c : ℕ), a^2 + b^2 = c^3) ∧
  (∃ (d : ℕ), a^3 + b^3 = d^2) := by
sorry

end two_numbers_squares_sum_cube_cubes_sum_square_l79_7958


namespace sphere_radii_formula_l79_7924

/-- Given three mutually tangent spheres touched by a plane at points A, B, and C,
    where the sides of triangle ABC are a, b, and c, prove that the radii of the
    spheres (x, y, z) are given by the formulas stated. -/
theorem sphere_radii_formula (a b c x y z : ℝ) 
  (h1 : a = 2 * Real.sqrt (x * y))
  (h2 : b = 2 * Real.sqrt (y * z))
  (h3 : c = 2 * Real.sqrt (x * z))
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  x = a * c / (2 * b) ∧ 
  y = a * b / (2 * c) ∧ 
  z = b * c / (2 * a) := by
  sorry

end sphere_radii_formula_l79_7924


namespace one_fifth_of_eight_point_five_l79_7904

theorem one_fifth_of_eight_point_five : (8.5 : ℚ) / 5 = 17 / 10 := by
  sorry

end one_fifth_of_eight_point_five_l79_7904


namespace gcd_special_numbers_l79_7914

theorem gcd_special_numbers : Nat.gcd 33333333 666666666 = 3 := by sorry

end gcd_special_numbers_l79_7914


namespace integer_ratio_problem_l79_7952

theorem integer_ratio_problem (a b : ℤ) :
  1996 * a + b / 96 = a + b →
  b / a = 2016 ∨ a / b = 1 / 2016 := by
sorry

end integer_ratio_problem_l79_7952


namespace street_tree_count_l79_7961

theorem street_tree_count (road_length : ℕ) (interval : ℕ) (h1 : road_length = 2575) (h2 : interval = 25) : 
  2 * (road_length / interval + 1) = 208 := by
  sorry

end street_tree_count_l79_7961


namespace complex_arithmetic_equality_l79_7928

theorem complex_arithmetic_equality : (5 - Complex.I) - (3 - Complex.I) - 5 * Complex.I = 2 - 5 * Complex.I := by
  sorry

end complex_arithmetic_equality_l79_7928


namespace function_identity_l79_7927

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (2*x + 1) = 4*x^2 + 4*x) :
  ∀ x, f x = x^2 - 1 := by
  sorry

end function_identity_l79_7927


namespace negation_of_proposition_l79_7915

theorem negation_of_proposition :
  (¬ (∀ n : ℕ, ¬ (Nat.Prime (2^n - 2)))) ↔ (∃ n : ℕ, Nat.Prime (2^n - 2)) := by
  sorry

end negation_of_proposition_l79_7915


namespace sum_coordinates_endpoint_l79_7913

/-- Given a line segment CD with midpoint M(4,7) and endpoint C(6,2),
    the sum of coordinates of the other endpoint D is 14. -/
theorem sum_coordinates_endpoint (C D M : ℝ × ℝ) : 
  C = (6, 2) → M = (4, 7) → M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 14 := by
sorry

end sum_coordinates_endpoint_l79_7913


namespace sin_plus_two_cos_alpha_l79_7946

theorem sin_plus_two_cos_alpha (α : Real) :
  (∃ P : Real × Real, P.1 = 3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.sin α + 2 * Real.cos α = 2 := by
sorry

end sin_plus_two_cos_alpha_l79_7946


namespace rain_duration_problem_l79_7986

theorem rain_duration_problem (x : ℝ) : 
  let first_day := 10
  let second_day := first_day + x
  let third_day := 2 * second_day
  first_day + second_day + third_day = 46 → x = 2 := by
sorry

end rain_duration_problem_l79_7986


namespace cindys_calculation_l79_7939

theorem cindys_calculation (x : ℝ) (h : (x - 5) / 7 = 15) : (x - 7) / 5 = 20.6 := by
  sorry

end cindys_calculation_l79_7939


namespace curve_in_second_quadrant_l79_7976

-- Define the curve C
def C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem curve_in_second_quadrant :
  (∀ a : ℝ, ∀ x y : ℝ, C a x y → second_quadrant x y) →
  (∀ a : ℝ, a ∈ Set.Ioi 2) :=
sorry

end curve_in_second_quadrant_l79_7976


namespace intersection_line_circle_l79_7911

/-- Given a line intersecting a circle, prove the value of parameter a -/
theorem intersection_line_circle (a : ℝ) : 
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), x - y + 2*a = 0 → x^2 + y^2 - 2*a*y - 2 = 0 → (x, y) = A ∨ (x, y) = B) ∧
    ‖A - B‖ = 4 * Real.sqrt 3 →
    a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 := by
  sorry

end intersection_line_circle_l79_7911


namespace martha_blocks_theorem_l79_7930

/-- The number of blocks Martha starts with -/
def starting_blocks : ℕ := 11

/-- The number of blocks Martha finds -/
def found_blocks : ℕ := 129

/-- The total number of blocks Martha ends up with -/
def total_blocks : ℕ := starting_blocks + found_blocks

theorem martha_blocks_theorem : total_blocks = 140 := by
  sorry

end martha_blocks_theorem_l79_7930


namespace compound_interest_proof_l79_7925

/-- Calculate compound interest and prove the total interest earned --/
theorem compound_interest_proof (P : ℝ) (r : ℝ) (n : ℕ) (h1 : P = 1000) (h2 : r = 0.1) (h3 : n = 3) :
  (P * (1 + r)^n - P) = 331 := by
  sorry

end compound_interest_proof_l79_7925


namespace fourth_number_in_proportion_l79_7902

-- Define the proportion
def proportion (a b c d : ℝ) : Prop := a / b = c / d

-- State the theorem
theorem fourth_number_in_proportion : 
  proportion 0.75 1.35 5 9 := by sorry

end fourth_number_in_proportion_l79_7902


namespace least_k_for_inequality_l79_7903

theorem least_k_for_inequality (n : ℕ) : 
  (0.0010101 : ℝ) * (10 : ℝ) ^ ((1586 : ℝ) / 500) > (n^2 - 3*n + 5 : ℝ) / (n^3 + 1) ∧ 
  ∀ k : ℚ, k < 1586/500 → (0.0010101 : ℝ) * (10 : ℝ) ^ (k : ℝ) ≤ (n^2 - 3*n + 5 : ℝ) / (n^3 + 1) :=
sorry

end least_k_for_inequality_l79_7903


namespace exactly_two_identical_pairs_l79_7949

/-- Two lines in the xy-plane -/
structure TwoLines where
  a : ℝ
  d : ℝ

/-- The condition for two lines to be identical -/
def are_identical (l : TwoLines) : Prop :=
  (4 / l.a = -l.d / 3) ∧ (l.d / l.a = 6)

/-- The theorem stating that there are exactly two pairs (a, d) that make the lines identical -/
theorem exactly_two_identical_pairs :
  ∃! (s : Finset TwoLines), (∀ l ∈ s, are_identical l) ∧ s.card = 2 := by
  sorry

end exactly_two_identical_pairs_l79_7949


namespace election_percentage_l79_7917

theorem election_percentage (total_members votes_cast : ℕ) 
  (percentage_of_total : ℚ) (h1 : total_members = 1600) 
  (h2 : votes_cast = 525) (h3 : percentage_of_total = 19.6875 / 100) : 
  (percentage_of_total * total_members) / votes_cast = 60 / 100 := by
  sorry

end election_percentage_l79_7917


namespace abc_sum_sqrt_l79_7905

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 16)
  (h2 : c + a = 18)
  (h3 : a + b = 20) :
  Real.sqrt (a * b * c * (a + b + c)) = 231 := by
  sorry

end abc_sum_sqrt_l79_7905


namespace bug_path_tiles_l79_7988

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tiles_visited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- Theorem stating that a bug walking diagonally across an 18x24 rectangular floor visits 36 tiles -/
theorem bug_path_tiles : tiles_visited 18 24 = 36 := by
  sorry

end bug_path_tiles_l79_7988


namespace power_of_power_l79_7987

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l79_7987


namespace quadratic_equation_general_form_l79_7953

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (4 * x = x^2 - 8) ↔ (x^2 - 4*x - 8 = 0) :=
by sorry

end quadratic_equation_general_form_l79_7953


namespace modular_congruence_l79_7943

theorem modular_congruence (n : ℕ) : 37^29 ≡ 7 [ZMOD 65] :=
by sorry

end modular_congruence_l79_7943


namespace piggy_bank_coins_l79_7951

/-- The number of coins remaining after each day of withdrawal --/
def coins_remaining (initial : ℕ) : Fin 9 → ℕ
| 0 => initial  -- Initial number of coins
| 1 => initial * 8 / 9  -- After day 1
| 2 => initial * 8 * 7 / (9 * 8)  -- After day 2
| 3 => initial * 8 * 7 * 6 / (9 * 8 * 7)  -- After day 3
| 4 => initial * 8 * 7 * 6 * 5 / (9 * 8 * 7 * 6)  -- After day 4
| 5 => initial * 8 * 7 * 6 * 5 * 4 / (9 * 8 * 7 * 6 * 5)  -- After day 5
| 6 => initial * 8 * 7 * 6 * 5 * 4 * 3 / (9 * 8 * 7 * 6 * 5 * 4)  -- After day 6
| 7 => initial * 8 * 7 * 6 * 5 * 4 * 3 * 2 / (9 * 8 * 7 * 6 * 5 * 4 * 3)  -- After day 7
| 8 => initial * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2)  -- After day 8

theorem piggy_bank_coins (initial : ℕ) : 
  (coins_remaining initial 8 = 5) → (initial = 45) := by
  sorry

#eval coins_remaining 45 8  -- Should output 5

end piggy_bank_coins_l79_7951


namespace forty_three_base7_equals_thirty_four_base9_l79_7955

/-- Converts a number from base-7 to base-10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base-10 to a given base -/
def base10ToBase (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem forty_three_base7_equals_thirty_four_base9 :
  let n := 43
  let base7Value := base7ToBase10 n
  let reversedDigits := reverseDigits n
  base10ToBase base7Value 9 = reversedDigits := by sorry

end forty_three_base7_equals_thirty_four_base9_l79_7955


namespace root_value_implies_m_l79_7966

theorem root_value_implies_m (m : ℝ) : (∃ x : ℝ, x^2 + m*x - 3 = 0 ∧ x = 1) → m = 2 := by
  sorry

end root_value_implies_m_l79_7966


namespace x1_x2_range_l79_7936

noncomputable section

def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem x1_x2_range (m : ℝ) (x₁ x₂ : ℝ) (h₁ : F m x₁ = 0) (h₂ : F m x₂ = 0) (h₃ : x₁ ≠ x₂) :
  x₁ * x₂ < Real.sqrt (Real.exp 1) ∧ ∀ y : ℝ, ∃ m : ℝ, ∃ x₁ x₂ : ℝ, 
    F m x₁ = 0 ∧ F m x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ < y :=
by sorry

end

end x1_x2_range_l79_7936


namespace f_of_three_equals_seven_l79_7971

/-- Given a function f(x) = x^7 - ax^5 + bx^3 + cx + 2 where f(-3) = -3, prove that f(3) = 7 -/
theorem f_of_three_equals_seven 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^7 - a*x^5 + b*x^3 + c*x + 2)
  (h2 : f (-3) = -3) :
  f 3 = 7 := by
sorry

end f_of_three_equals_seven_l79_7971


namespace increasing_implies_a_geq_neg_two_l79_7921

/-- A quadratic function f(x) = x^2 + 2(a-1)x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x - 3

/-- The property of f being increasing on [3, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≥ 3 → y ≥ 3 → x < y → f a x < f a y

/-- Theorem: If f is increasing on [3, +∞), then a ≥ -2 -/
theorem increasing_implies_a_geq_neg_two (a : ℝ) :
  is_increasing_on_interval a → a ≥ -2 :=
by sorry

end increasing_implies_a_geq_neg_two_l79_7921


namespace system_solution_l79_7931

theorem system_solution (x y z u : ℝ) : 
  (x^3 * y^2 * z = 2 ∧ 
   z^3 * u^2 * x = 32 ∧ 
   y^3 * z^2 * u = 8 ∧ 
   u^3 * x^2 * y = 8) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
   (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
   (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
   (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2)) :=
by sorry

end system_solution_l79_7931


namespace stratified_sample_size_l79_7974

theorem stratified_sample_size 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sampled_male : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : male_employees = 90) 
  (h3 : sampled_male = 27) 
  (h4 : male_employees < total_employees) :
  (sampled_male : ℚ) / (male_employees : ℚ) * (total_employees : ℚ) = 36 := by
sorry

end stratified_sample_size_l79_7974


namespace banana_bunches_l79_7912

theorem banana_bunches (total_bananas : ℕ) (eight_bunch_count : ℕ) (bananas_per_eight_bunch : ℕ) : 
  total_bananas = 83 →
  eight_bunch_count = 6 →
  bananas_per_eight_bunch = 8 →
  ∃ (seven_bunch_count : ℕ),
    seven_bunch_count * 7 + eight_bunch_count * bananas_per_eight_bunch = total_bananas ∧
    seven_bunch_count = 5 :=
by sorry

end banana_bunches_l79_7912


namespace inequality_solution_range_l79_7992

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 4

-- State the theorem
theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x > a) → a < -4 := by
  sorry

end inequality_solution_range_l79_7992


namespace solution_set_of_inequality_l79_7937

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h_decreasing : is_decreasing f) 
  (h_point1 : f 0 = 1) 
  (h_point2 : f 3 = -1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by sorry

end solution_set_of_inequality_l79_7937


namespace correct_selection_count_l79_7973

/-- The number of ways to select course representatives from a class -/
def select_representatives (num_boys num_girls num_subjects : ℕ) : ℕ × ℕ × ℕ :=
  let scenario1 := sorry
  let scenario2 := sorry
  let scenario3 := sorry
  (scenario1, scenario2, scenario3)

/-- Theorem stating the correct number of ways to select representatives under different conditions -/
theorem correct_selection_count :
  select_representatives 6 4 5 = (22320, 12096, 1008) := by
  sorry

end correct_selection_count_l79_7973


namespace h_inverse_correct_l79_7968

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x^2 - 2
def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := Real.sqrt ((x + 3) / 12)

-- Theorem statement
theorem h_inverse_correct (x : ℝ) : h (h_inv x) = x ∧ h_inv (h x) = x :=
  sorry

end h_inverse_correct_l79_7968


namespace total_pies_is_119_l79_7965

/-- The number of pies Eddie can bake in a day -/
def eddie_daily_pies : ℕ := 3

/-- The number of pies Eddie's sister can bake in a day -/
def sister_daily_pies : ℕ := 6

/-- The number of pies Eddie's mother can bake in a day -/
def mother_daily_pies : ℕ := 8

/-- The number of days they will bake pies -/
def baking_days : ℕ := 7

/-- The total number of pies baked by Eddie, his sister, and his mother in 7 days -/
def total_pies : ℕ := (eddie_daily_pies + sister_daily_pies + mother_daily_pies) * baking_days

theorem total_pies_is_119 : total_pies = 119 := by
  sorry

end total_pies_is_119_l79_7965


namespace parabola_c_value_l79_7978

/-- A parabola in the xy-plane with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-3) = 2 →   -- Vertex condition
  p.x_coord (-5) = 0 →   -- Point condition
  p.c = -5/2 := by sorry

end parabola_c_value_l79_7978


namespace trapezoid_base_lengths_l79_7980

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  b : ℝ  -- smaller base
  h : ℝ  -- altitude
  B : ℝ  -- larger base
  d : ℝ  -- common difference
  arithmetic_progression : b = h - 2 * d ∧ B = h + 2 * d
  area : (b + B) * h / 2 = 48

/-- Theorem stating the base lengths of the trapezoid -/
theorem trapezoid_base_lengths (t : Trapezoid) : 
  t.b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ 
  t.B = Real.sqrt 48 + 2 * Real.sqrt 3 := by
  sorry

end trapezoid_base_lengths_l79_7980


namespace proposition_truth_count_l79_7929

theorem proposition_truth_count : 
  let P1 := ∀ x : ℝ, x > -3 → x > -6
  let P2 := ∀ x : ℝ, x > -6 → x > -3
  let P3 := ∀ x : ℝ, x ≤ -3 → x ≤ -6
  let P4 := ∀ x : ℝ, x ≤ -6 → x ≤ -3
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) ∨
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨
  (P1 ∧ P2 ∧ ¬P3 ∧ ¬P4) ∨
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨
  (¬P1 ∧ ¬P2 ∧ P3 ∧ P4) :=
by
  sorry

end proposition_truth_count_l79_7929


namespace sequence_nth_term_l79_7999

/-- Given a sequence {a_n} where the differences between successive terms form
    a geometric sequence with first term 1 and common ratio r, 
    prove that the nth term of the sequence is (1-r^(n-1))/(1-r) -/
theorem sequence_nth_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n+1) - a n = r^(n-1)) :
  ∀ n : ℕ, a n = (1 - r^(n-1)) / (1 - r) :=
sorry

end sequence_nth_term_l79_7999


namespace vertical_tangents_condition_l79_7969

/-- The function f(x) = x(a - 1/e^x) has two distinct points with vertical tangents
    if and only if a is in the open interval (0, 2/e) -/
theorem vertical_tangents_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, (x * (a - Real.exp (-x))) = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, (a - (1 + x) * Real.exp (-x)) = 0 → x = x₁ ∨ x = x₂)) ↔
  (0 < a ∧ a < 2 / Real.exp 1) :=
sorry

end vertical_tangents_condition_l79_7969


namespace one_incorrect_statement_l79_7923

-- Define the structure for a statistical statement
structure StatStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the list of statements
def statements : List StatStatement :=
  [
    { id := 1, content := "Residuals can be used to judge the effectiveness of model fitting", isCorrect := true },
    { id := 2, content := "Given a regression equation: ŷ=3-5x, when variable x increases by one unit, y increases by an average of 5 units", isCorrect := false },
    { id := 3, content := "The linear regression line: ŷ=b̂x+â must pass through the point (x̄, ȳ)", isCorrect := true },
    { id := 4, content := "In a 2×2 contingency table, it is calculated that χ²=13.079, thus there is a 99% confidence that there is a relationship between the two variables (where P(χ²≥10.828)=0.001)", isCorrect := true }
  ]

-- Theorem: Exactly one statement is incorrect
theorem one_incorrect_statement : 
  (statements.filter (fun s => !s.isCorrect)).length = 1 := by
  sorry

end one_incorrect_statement_l79_7923
