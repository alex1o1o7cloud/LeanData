import Mathlib

namespace min_value_sum_squares_l3260_326097

theorem min_value_sum_squares (x y z : ℝ) :
  x - 1 = 2 * (y + 1) ∧ x - 1 = 3 * (z + 2) →
  ∀ a b c : ℝ, a - 1 = 2 * (b + 1) ∧ a - 1 = 3 * (c + 2) →
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  ∃ x₀ y₀ z₀ : ℝ, x₀ - 1 = 2 * (y₀ + 1) ∧ x₀ - 1 = 3 * (z₀ + 2) ∧
                  x₀^2 + y₀^2 + z₀^2 = 293 / 49 :=
by sorry

end min_value_sum_squares_l3260_326097


namespace binomial_10_choose_3_l3260_326016

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l3260_326016


namespace cubic_function_theorem_l3260_326006

/-- A cubic function with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

/-- Theorem stating that under given conditions, c must equal 16 -/
theorem cubic_function_theorem (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : f a b c a = a^3)
  (h2 : f a b c b = b^3) :
  c = 16 := by sorry

end cubic_function_theorem_l3260_326006


namespace rectangle_area_diagonal_l3260_326088

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 10 / 29 := by
  sorry

end rectangle_area_diagonal_l3260_326088


namespace unique_solution_for_equation_l3260_326077

theorem unique_solution_for_equation (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end unique_solution_for_equation_l3260_326077


namespace isosceles_right_triangle_XY_length_l3260_326081

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop := sorry

def longerSide (t : Triangle) (s1 s2 : ℝ × ℝ) : Prop := sorry

def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem isosceles_right_triangle_XY_length 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : longerSide t t.X t.Y) 
  (h3 : triangleArea t = 36) : 
  ‖t.X - t.Y‖ = 12 := by sorry

end isosceles_right_triangle_XY_length_l3260_326081


namespace quadratic_above_x_axis_l3260_326047

/-- Given a quadratic function f(x) = ax^2 + x + 5, if f(x) > 0 for all real x, then a > 1/20 -/
theorem quadratic_above_x_axis (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x + 5 > 0) → a > 1/20 := by
  sorry

end quadratic_above_x_axis_l3260_326047


namespace zoo_visit_l3260_326074

/-- The number of children who saw giraffes but not pandas -/
def giraffes_not_pandas (total children_pandas children_giraffes pandas_not_giraffes : ℕ) : ℕ :=
  children_giraffes - (children_pandas - pandas_not_giraffes)

/-- Theorem stating the number of children who saw giraffes but not pandas -/
theorem zoo_visit (total children_pandas children_giraffes pandas_not_giraffes : ℕ) 
  (h1 : total = 50)
  (h2 : children_pandas = 36)
  (h3 : children_giraffes = 28)
  (h4 : pandas_not_giraffes = 15) :
  giraffes_not_pandas total children_pandas children_giraffes pandas_not_giraffes = 7 := by
  sorry


end zoo_visit_l3260_326074


namespace solve_problem_l3260_326090

def problem (basketballs soccer_balls volleyballs : ℕ) : Prop :=
  (soccer_balls = basketballs + 23) ∧
  (volleyballs + 18 = soccer_balls) ∧
  (volleyballs = 40)

theorem solve_problem :
  ∃ (basketballs soccer_balls volleyballs : ℕ),
    problem basketballs soccer_balls volleyballs ∧ basketballs = 35 := by
  sorry

end solve_problem_l3260_326090


namespace quadratic_solution_l3260_326029

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 + k * x + 2

-- Define the known root
def known_root : ℝ := -0.5

-- Theorem statement
theorem quadratic_solution :
  ∃ (k : ℝ),
    (quadratic_equation k known_root = 0) ∧
    (k = 6) ∧
    (∃ (other_root : ℝ), 
      (quadratic_equation k other_root = 0) ∧
      (other_root = -1)) := by
  sorry

end quadratic_solution_l3260_326029


namespace sum_of_roots_of_equation_l3260_326057

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end sum_of_roots_of_equation_l3260_326057


namespace matrix_power_2023_l3260_326082

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 : A ^ 2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_l3260_326082


namespace complex_number_problem_l3260_326033

theorem complex_number_problem (a : ℝ) (z₁ : ℂ) (h₁ : a > 0) (h₂ : z₁ = 1 + a * I) (h₃ : ∃ b : ℝ, z₁^2 = b * I) :
  z₁ = 1 + I ∧ Complex.abs (z₁ / (1 - I)) = 1 := by
  sorry

end complex_number_problem_l3260_326033


namespace geometric_sequence_problem_l3260_326045

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) 
  (h2 : 3 * a 3 ^ 2 - 25 * a 3 + 27 = 0) (h3 : 3 * a 11 ^ 2 - 25 * a 11 + 27 = 0) : a 7 = 3 := by
  sorry

end geometric_sequence_problem_l3260_326045


namespace simplify_trig_expression_find_sin_beta_plus_pi_over_4_l3260_326065

-- Part 1
theorem simplify_trig_expression :
  Real.sin (119 * π / 180) * Real.sin (181 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by sorry

-- Part 2
theorem find_sin_beta_plus_pi_over_4 (α β : Real) 
  (h1 : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5)
  (h2 : π < β ∧ β < 3*π/2) :  -- β is in the third quadrant
  Real.sin (β + π/4) = -7*Real.sqrt 2/10 := by sorry

end simplify_trig_expression_find_sin_beta_plus_pi_over_4_l3260_326065


namespace condition_sufficient_not_necessary_l3260_326000

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x * (x - 5) < 0) ∧
  (∃ x : ℝ, x * (x - 5) < 0 ∧ ¬(2 < x ∧ x < 3)) :=
by sorry

end condition_sufficient_not_necessary_l3260_326000


namespace survey_results_l3260_326086

theorem survey_results (total : ℕ) (support_a : ℕ) (support_b : ℕ) (support_both : ℕ) (support_neither : ℕ) : 
  total = 50 ∧
  support_a = (3 * total) / 5 ∧
  support_b = support_a + 3 ∧
  support_neither = support_both / 3 + 1 ∧
  total = support_a + support_b - support_both + support_neither →
  support_both = 21 ∧ support_neither = 8 := by
  sorry

end survey_results_l3260_326086


namespace fuel_consumption_statements_correct_l3260_326093

/-- Represents the fuel consumption data for a car journey -/
structure FuelConsumptionData where
  initial_fuel : ℝ
  distance_interval : ℝ
  fuel_decrease_per_interval : ℝ
  total_distance : ℝ

/-- Theorem stating the correctness of all fuel consumption statements -/
theorem fuel_consumption_statements_correct
  (data : FuelConsumptionData)
  (h_initial : data.initial_fuel = 45)
  (h_interval : data.distance_interval = 50)
  (h_decrease : data.fuel_decrease_per_interval = 4)
  (h_total : data.total_distance = 500) :
  (data.initial_fuel = 45) ∧
  ((data.fuel_decrease_per_interval / data.distance_interval) * 100 = 8) ∧
  (∀ x y : ℝ, y = data.initial_fuel - (data.fuel_decrease_per_interval / data.distance_interval) * x) ∧
  (data.initial_fuel - (data.fuel_decrease_per_interval / data.distance_interval) * data.total_distance = 5) :=
by sorry


end fuel_consumption_statements_correct_l3260_326093


namespace color_films_count_l3260_326048

theorem color_films_count (x y : ℝ) (h : x > 0) :
  let total_bw := 40 * x
  let selected_bw := 2 * y / 5
  let fraction_color := 0.9615384615384615
  let color_films := (fraction_color * (selected_bw + color_films)) / (1 - fraction_color)
  color_films = 10 * y := by
  sorry

end color_films_count_l3260_326048


namespace boys_in_biology_class_l3260_326092

/-- Given a Physics class with 200 students and a Biology class with half as many students,
    where the ratio of girls to boys in Biology is 3:1, prove that there are 25 boys in Biology. -/
theorem boys_in_biology_class
  (physics_students : ℕ)
  (biology_students : ℕ)
  (girls_to_boys_ratio : ℚ)
  (h1 : physics_students = 200)
  (h2 : biology_students = physics_students / 2)
  (h3 : girls_to_boys_ratio = 3)
  : biology_students / (girls_to_boys_ratio + 1) = 25 := by
  sorry

end boys_in_biology_class_l3260_326092


namespace range_equals_fixed_points_l3260_326037

theorem range_equals_fixed_points (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f (f m) + f n) : 
  {n : ℕ | ∃ k : ℕ, f k = n} = {n : ℕ | f n = n} := by
sorry

end range_equals_fixed_points_l3260_326037


namespace range_of_g_l3260_326096

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x) * (Real.arcsin x)

theorem range_of_g :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
    Real.arccos x + Real.arcsin x = π / 2 →
    ∃ y ∈ Set.Icc 0 (π^2 / 8), g y = g x ∧
    ∀ z ∈ Set.Icc (-1 : ℝ) 1, g z ≤ π^2 / 8 ∧ g z ≥ 0 :=
sorry

end range_of_g_l3260_326096


namespace complete_factorization_l3260_326069

theorem complete_factorization (x : ℝ) : 
  x^6 - 64 = (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end complete_factorization_l3260_326069


namespace sum_of_roots_l3260_326043

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 6*x^2 + 15*x = 12) 
  (hy : y^3 - 6*y^2 + 15*y = 16) : 
  x + y = 4 := by
sorry

end sum_of_roots_l3260_326043


namespace set_c_is_proportional_l3260_326017

/-- A set of four real numbers is proportional if the product of its extremes equals the product of its means -/
def isProportional (a b c d : ℝ) : Prop := a * d = b * c

/-- The set (2, 3, 4, 6) is proportional -/
theorem set_c_is_proportional : isProportional 2 3 4 6 := by
  sorry

end set_c_is_proportional_l3260_326017


namespace tan_theta_is_negative_three_l3260_326027

/-- Given vectors a and b with angle θ between them, if a • b = -1, a = (-1, 2), and |b| = √2, then tan θ = -3 -/
theorem tan_theta_is_negative_three (a b : ℝ × ℝ) (θ : ℝ) :
  a = (-1, 2) →
  a • b = -1 →
  ‖b‖ = Real.sqrt 2 →
  Real.tan θ = -3 := by
  sorry

end tan_theta_is_negative_three_l3260_326027


namespace expected_winnings_is_negative_half_dollar_l3260_326008

/-- Represents the sections of the spinner --/
inductive Section
  | Red
  | Blue
  | Green
  | Yellow

/-- Returns the probability of landing on a given section --/
def probability (s : Section) : ℚ :=
  match s with
  | Section.Red => 3/8
  | Section.Blue => 1/4
  | Section.Green => 1/4
  | Section.Yellow => 1/8

/-- Returns the winnings (in dollars) for a given section --/
def winnings (s : Section) : ℤ :=
  match s with
  | Section.Red => 2
  | Section.Blue => 4
  | Section.Green => -3
  | Section.Yellow => -6

/-- Calculates the expected winnings from spinning the spinner --/
def expectedWinnings : ℚ :=
  (probability Section.Red * winnings Section.Red) +
  (probability Section.Blue * winnings Section.Blue) +
  (probability Section.Green * winnings Section.Green) +
  (probability Section.Yellow * winnings Section.Yellow)

/-- Theorem stating that the expected winnings is -$0.50 --/
theorem expected_winnings_is_negative_half_dollar :
  expectedWinnings = -1/2 := by
  sorry

end expected_winnings_is_negative_half_dollar_l3260_326008


namespace coin_problem_l3260_326034

theorem coin_problem (total : ℕ) (difference : ℕ) (tails : ℕ) : 
  total = 1250 →
  difference = 124 →
  tails + (tails + difference) = total →
  tails = 563 := by
sorry

end coin_problem_l3260_326034


namespace sum_mod_nine_l3260_326025

theorem sum_mod_nine : (1234 + 1235 + 1236 + 1237 + 1238) % 9 = 3 := by
  sorry

end sum_mod_nine_l3260_326025


namespace veranda_area_l3260_326054

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) : 
  room_length = 21 →
  room_width = 12 →
  veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 148 := by
  sorry


end veranda_area_l3260_326054


namespace rectangular_table_capacity_l3260_326031

/-- The number of square tables arranged in a row -/
def num_tables : ℕ := 8

/-- The number of people that can sit evenly spaced around one square table -/
def people_per_square_table : ℕ := 12

/-- The number of sides in a square table -/
def sides_per_square : ℕ := 4

/-- Calculate the number of people that can sit on one side of a square table -/
def people_per_side : ℕ := people_per_square_table / sides_per_square

/-- The number of people that can sit on the long side of the rectangular table -/
def long_side_capacity : ℕ := num_tables * people_per_side

/-- The number of people that can sit on the short side of the rectangular table -/
def short_side_capacity : ℕ := 2 * people_per_side

/-- The total number of people that can sit around the rectangular table -/
def total_capacity : ℕ := 2 * long_side_capacity + 2 * short_side_capacity

theorem rectangular_table_capacity :
  total_capacity = 60 := by sorry

end rectangular_table_capacity_l3260_326031


namespace sqrt_fraction_sum_diff_l3260_326052

theorem sqrt_fraction_sum_diff (x : ℝ) : 
  x = Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36 - (1 : ℝ) / 100) → x = (Real.sqrt 13) / 15 :=
by sorry

end sqrt_fraction_sum_diff_l3260_326052


namespace susan_ate_six_candies_l3260_326012

/-- The number of candies Susan bought on Tuesday -/
def tuesday_candies : ℕ := 3

/-- The number of candies Susan bought on Thursday -/
def thursday_candies : ℕ := 5

/-- The number of candies Susan bought on Friday -/
def friday_candies : ℕ := 2

/-- The number of candies Susan has left -/
def candies_left : ℕ := 4

/-- The total number of candies Susan bought -/
def total_candies : ℕ := tuesday_candies + thursday_candies + friday_candies

/-- The number of candies Susan ate -/
def candies_eaten : ℕ := total_candies - candies_left

theorem susan_ate_six_candies : candies_eaten = 6 := by
  sorry

end susan_ate_six_candies_l3260_326012


namespace correct_equations_l3260_326044

/-- Represents the money held by a person -/
structure Money where
  amount : ℚ
  deriving Repr

/-- The problem setup -/
def problem_setup (a b : Money) : Prop :=
  (a.amount + (1/2) * b.amount = 50) ∧
  ((2/3) * a.amount + b.amount = 50)

/-- The theorem to prove -/
theorem correct_equations (a b : Money) :
  problem_setup a b ↔
  (a.amount + (1/2) * b.amount = 50 ∧ (2/3) * a.amount + b.amount = 50) :=
by sorry

end correct_equations_l3260_326044


namespace mean_equality_problem_l3260_326050

theorem mean_equality_problem (y : ℝ) : 
  (5 + 8 + 17) / 3 = (12 + y) / 2 → y = 8 := by sorry

end mean_equality_problem_l3260_326050


namespace gcd_765432_654321_l3260_326007

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l3260_326007


namespace less_than_preserved_subtraction_l3260_326032

theorem less_than_preserved_subtraction (a b : ℝ) : a < b → a - 1 < b - 1 := by
  sorry

end less_than_preserved_subtraction_l3260_326032


namespace blood_expires_february_5_l3260_326062

def seconds_per_day : ℕ := 24 * 60 * 60

def february_days : ℕ := 28

def blood_expiration_seconds : ℕ := Nat.factorial 9

def days_until_expiration : ℕ := blood_expiration_seconds / seconds_per_day

theorem blood_expires_february_5 :
  days_until_expiration = 4 →
  (1 : ℕ) + days_until_expiration = 5 :=
by sorry

end blood_expires_february_5_l3260_326062


namespace division_problem_l3260_326072

theorem division_problem (remainder quotient divisor dividend : ℕ) :
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  dividend = divisor * quotient + remainder →
  dividend = 86 := by
sorry

end division_problem_l3260_326072


namespace line_intersects_circle_twice_l3260_326087

/-- The line y = -x + a intersects the curve y = √(1 - x²) at two points
    if and only if a is in the range [1, √2). -/
theorem line_intersects_circle_twice (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   -x₁ + a = Real.sqrt (1 - x₁^2) ∧
   -x₂ + a = Real.sqrt (1 - x₂^2)) ↔ 
  1 ≤ a ∧ a < Real.sqrt 2 :=
sorry

end line_intersects_circle_twice_l3260_326087


namespace sum_of_f_values_l3260_326073

noncomputable def f (x : ℝ) : ℝ := (x * Real.exp x + x + 2) / (Real.exp x + 1) + Real.sin x

theorem sum_of_f_values : 
  f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 = 9 :=
by sorry

end sum_of_f_values_l3260_326073


namespace set_equality_implies_a_equals_one_l3260_326002

theorem set_equality_implies_a_equals_one (a : ℝ) :
  let A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
  let B : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}
  (A ∪ B) ⊆ (A ∩ B) → a = 1 := by
  sorry

end set_equality_implies_a_equals_one_l3260_326002


namespace slower_traveler_speed_l3260_326058

/-- Proves that given two people traveling in opposite directions for 1.5 hours,
    where one travels 3 miles per hour faster than the other, and they end up 19.5 miles apart,
    the slower person's speed is 5 miles per hour. -/
theorem slower_traveler_speed
  (time : ℝ)
  (distance_apart : ℝ)
  (speed_difference : ℝ)
  (h1 : time = 1.5)
  (h2 : distance_apart = 19.5)
  (h3 : speed_difference = 3)
  : ∃ (slower_speed : ℝ), slower_speed = 5 ∧
    distance_apart = time * (slower_speed + (slower_speed + speed_difference)) :=
by sorry

end slower_traveler_speed_l3260_326058


namespace triangle_angle_problem_l3260_326030

theorem triangle_angle_problem (left right top : ℝ) : 
  left + right + top = 250 →
  left = 2 * right →
  right = 60 →
  top = 70 := by sorry

end triangle_angle_problem_l3260_326030


namespace john_mileage_conversion_l3260_326039

/-- Converts a base-8 number represented as a list of digits to its base-10 equivalent -/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- The base-8 representation of John's mileage -/
def johnMileageBase8 : List Nat := [3, 4, 5, 2]

/-- Theorem: John's mileage in base-10 is 1834 miles -/
theorem john_mileage_conversion :
  base8ToBase10 johnMileageBase8 = 1834 := by
  sorry

end john_mileage_conversion_l3260_326039


namespace smallest_w_l3260_326055

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (13^2) (936 * w) → 
  ∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (13^2) (936 * v) → 
    w ≤ v → 
  w = 156 := by sorry

end smallest_w_l3260_326055


namespace min_total_faces_l3260_326013

/-- Represents a fair die with a given number of faces. -/
structure Die where
  faces : ℕ
  faces_gt_6 : faces > 6

/-- Calculates the number of ways to roll a given sum with two dice. -/
def waysToRoll (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- The probability of rolling a given sum with two dice. -/
def probOfSum (d1 d2 : Die) (sum : ℕ) : ℚ :=
  sorry

theorem min_total_faces (d1 d2 : Die) :
  (probOfSum d1 d2 8 = (1 : ℚ) / 2 * probOfSum d1 d2 11) →
  (probOfSum d1 d2 15 = (1 : ℚ) / 30) →
  d1.faces + d2.faces ≥ 18 :=
sorry

end min_total_faces_l3260_326013


namespace min_value_of_expression_min_value_attained_l3260_326061

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 :=
by sorry

end min_value_of_expression_min_value_attained_l3260_326061


namespace lisa_total_miles_l3260_326078

/-- The total miles flown by Lisa -/
def total_miles_flown (distance_per_trip : Float) (num_trips : Float) : Float :=
  distance_per_trip * num_trips

/-- Theorem stating that Lisa's total miles flown is 8192.0 -/
theorem lisa_total_miles :
  total_miles_flown 256.0 32.0 = 8192.0 := by
  sorry

end lisa_total_miles_l3260_326078


namespace smallest_n_with_four_pairs_l3260_326018

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 65 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m : ℕ, 0 < m → m < 65 → g m ≠ 4) ∧ g 65 = 4 := by
  sorry

end smallest_n_with_four_pairs_l3260_326018


namespace number_of_proper_subsets_of_A_l3260_326010

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define the complement of A with respect to U
def complement_A : Finset Nat := {2}

-- Define set A based on its complement
def A : Finset Nat := U \ complement_A

-- Theorem statement
theorem number_of_proper_subsets_of_A : 
  Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end number_of_proper_subsets_of_A_l3260_326010


namespace union_of_A_and_B_l3260_326095

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end union_of_A_and_B_l3260_326095


namespace chantel_bracelets_l3260_326024

/-- Represents the number of bracelets Chantel makes per day in the last four days -/
def x : ℕ := sorry

/-- The total number of bracelets Chantel has at the end -/
def total_bracelets : ℕ := 13

/-- The number of bracelets Chantel makes in the first 5 days -/
def first_phase_bracelets : ℕ := 2 * 5

/-- The number of bracelets Chantel gives away after the first phase -/
def first_giveaway : ℕ := 3

/-- The number of bracelets Chantel gives away after the second phase -/
def second_giveaway : ℕ := 6

/-- The number of days in the second phase -/
def second_phase_days : ℕ := 4

theorem chantel_bracelets : 
  first_phase_bracelets - first_giveaway + x * second_phase_days - second_giveaway = total_bracelets ∧ 
  x = 3 := by sorry

end chantel_bracelets_l3260_326024


namespace min_distinct_integers_for_progressions_l3260_326036

/-- A sequence of integers forms a geometric progression of length 5 -/
def is_geometric_progression (seq : Fin 5 → ℤ) : Prop :=
  ∃ (b q : ℤ), ∀ i : Fin 5, seq i = b * q ^ (i : ℕ)

/-- A sequence of integers forms an arithmetic progression of length 5 -/
def is_arithmetic_progression (seq : Fin 5 → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ i : Fin 5, seq i = a + (i : ℕ) * d

/-- The minimum number of distinct integers needed for both progressions -/
def min_distinct_integers : ℕ := 6

/-- Theorem stating the minimum number of distinct integers needed -/
theorem min_distinct_integers_for_progressions :
  ∀ (S : Finset ℤ),
  (∃ (seq_gp : Fin 5 → ℤ), (∀ i, seq_gp i ∈ S) ∧ is_geometric_progression seq_gp) ∧
  (∃ (seq_ap : Fin 5 → ℤ), (∀ i, seq_ap i ∈ S) ∧ is_arithmetic_progression seq_ap) →
  S.card ≥ min_distinct_integers :=
sorry

end min_distinct_integers_for_progressions_l3260_326036


namespace quadratic_function_inequality_max_l3260_326094

theorem quadratic_function_inequality_max (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = Real.sqrt 6 - 2 ∧ 
    (∀ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') → 
      b'^2 / (a'^2 + 2 * c'^2) ≤ M) ∧
    (∃ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') ∧ 
      b'^2 / (a'^2 + 2 * c'^2) = M)) :=
by sorry

end quadratic_function_inequality_max_l3260_326094


namespace area_of_bounded_region_l3260_326071

-- Define the lines that bound the region
def line1 (x y : ℝ) : Prop := x + y = 6
def line2 (y : ℝ) : Prop := y = 4
def line3 (x : ℝ) : Prop := x = 0
def line4 (y : ℝ) : Prop := y = 0

-- Define the vertices of the quadrilateral
def P : ℝ × ℝ := (6, 0)
def Q : ℝ × ℝ := (2, 4)
def R : ℝ × ℝ := (0, 6)
def O : ℝ × ℝ := (0, 0)

-- Define the area of the quadrilateral
def area_quadrilateral : ℝ := 18

-- Theorem statement
theorem area_of_bounded_region :
  area_quadrilateral = 18 :=
sorry

end area_of_bounded_region_l3260_326071


namespace smallest_n_satisfying_inequality_l3260_326089

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 9 else sorry

def sequence_sum (n : ℕ) : ℚ :=
  sorry

theorem smallest_n_satisfying_inequality : 
  (∀ n : ℕ, n > 0 → 3 * sequence_a (n + 1) + sequence_a n = 4) →
  sequence_a 1 = 9 →
  (∀ n : ℕ, n > 0 → |sequence_sum n - n - 6| < 1 / 125 → n ≥ 7) ∧
  |sequence_sum 7 - 7 - 6| < 1 / 125 :=
sorry

end smallest_n_satisfying_inequality_l3260_326089


namespace san_diego_zoo_tickets_l3260_326059

/-- Given a family of 7 members visiting the San Diego Zoo, prove that 3 adult tickets were purchased. --/
theorem san_diego_zoo_tickets (total_cost : ℕ) (adult_price child_price : ℕ) 
  (h1 : total_cost = 119)
  (h2 : adult_price = 21)
  (h3 : child_price = 14) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = 7 ∧
    adult_tickets * adult_price + child_tickets * child_price = total_cost ∧
    adult_tickets = 3 := by
  sorry

end san_diego_zoo_tickets_l3260_326059


namespace profit_maximized_at_12_marginal_profit_decreasing_l3260_326099

-- Define the revenue and cost functions
def R (x : ℕ) : ℝ := 3700 * x + 45 * x^2 - 10 * x^3
def C (x : ℕ) : ℝ := 460 * x + 5000

-- Define the profit function
def P (x : ℕ) : ℝ := R x - C x

-- Define the marginal function
def M (f : ℕ → ℝ) (x : ℕ) : ℝ := f (x + 1) - f x

-- Define the marginal profit function
def MP (x : ℕ) : ℝ := M P x

-- Theorem: Profit is maximized when 12 ships are built
theorem profit_maximized_at_12 :
  ∀ x : ℕ, 1 ≤ x → x ≤ 20 → P 12 ≥ P x :=
sorry

-- Theorem: Marginal profit function is decreasing on [1, 19]
theorem marginal_profit_decreasing :
  ∀ x y : ℕ, 1 ≤ x → x < y → y ≤ 19 → MP y < MP x :=
sorry

end profit_maximized_at_12_marginal_profit_decreasing_l3260_326099


namespace vector_problem_l3260_326009

/-- The angle between two 2D vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Checks if two 2D vectors are collinear -/
def collinear (v w : ℝ × ℝ) : Prop := sorry

/-- Checks if two 2D vectors are perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop := sorry

theorem vector_problem (a b c : ℝ × ℝ) 
  (ha : a = (1, 2))
  (hb : b = (-2, 6))
  (hc : c = (-1, 3)) : 
  angle a b = π/4 ∧ 
  collinear b c ∧ 
  perpendicular a (a - c) := by
  sorry

end vector_problem_l3260_326009


namespace greg_distance_when_azarah_finishes_l3260_326022

/-- Represents the constant speed of a runner -/
structure Speed : Type :=
  (value : ℝ)
  (pos : value > 0)

/-- Calculates the distance traveled given speed and time -/
def distance (s : Speed) (t : ℝ) : ℝ := s.value * t

theorem greg_distance_when_azarah_finishes 
  (azarah_speed charlize_speed greg_speed : Speed)
  (h1 : distance azarah_speed 1 = 100)
  (h2 : distance charlize_speed 1 = 80)
  (h3 : distance charlize_speed (100 / charlize_speed.value) = 100)
  (h4 : distance greg_speed (100 / charlize_speed.value) = 90) :
  distance greg_speed (100 / azarah_speed.value) = 72 :=
sorry

end greg_distance_when_azarah_finishes_l3260_326022


namespace smallest_a_l3260_326021

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex condition
def vertex_condition (a b c : ℝ) : Prop :=
  parabola a b c (1/3) = -4/3

-- Define the integer condition
def integer_condition (a b c : ℝ) : Prop :=
  ∃ n : ℤ, 3*a + 2*b + c = n

-- State the theorem
theorem smallest_a (a b c : ℝ) :
  vertex_condition a b c →
  integer_condition a b c →
  a > 0 →
  (∀ a' b' c' : ℝ, vertex_condition a' b' c' → integer_condition a' b' c' → a' > 0 → a' ≥ a) →
  a = 3 := by
  sorry

end smallest_a_l3260_326021


namespace marble_difference_l3260_326003

theorem marble_difference (total : ℕ) (yellow : ℕ) (blue_ratio : ℕ) (red_ratio : ℕ)
  (h_total : total = 19)
  (h_yellow : yellow = 5)
  (h_blue_ratio : blue_ratio = 3)
  (h_red_ratio : red_ratio = 4) :
  let remaining : ℕ := total - yellow
  let share : ℕ := remaining / (blue_ratio + red_ratio)
  let red : ℕ := red_ratio * share
  red - yellow = 3 := by sorry

end marble_difference_l3260_326003


namespace proportion_solution_l3260_326042

theorem proportion_solution (x : ℝ) : (0.75 / x = 3 / 8) → x = 2 := by
  sorry

end proportion_solution_l3260_326042


namespace opposite_of_negative_one_third_l3260_326001

theorem opposite_of_negative_one_third : 
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end opposite_of_negative_one_third_l3260_326001


namespace number_of_apricot_trees_apricot_trees_count_l3260_326026

/-- Proves that the number of apricot trees is 135, given the conditions stated in the problem. -/
theorem number_of_apricot_trees : ℕ → Prop :=
  fun n : ℕ =>
    (∃ peach_trees : ℕ,
      peach_trees = 300 ∧
      peach_trees = 2 * n + 30) →
    n = 135

/-- The main theorem stating that there are 135 apricot trees. -/
theorem apricot_trees_count : ∃ n : ℕ, number_of_apricot_trees n :=
  sorry

end number_of_apricot_trees_apricot_trees_count_l3260_326026


namespace money_division_l3260_326028

theorem money_division (A B C : ℚ) (h1 : A = (1/3) * (B + C))
                                   (h2 : ∃ x, B = x * (A + C))
                                   (h3 : A = B + 15)
                                   (h4 : A + B + C = 540) :
  ∃ x, B = x * (A + C) ∧ x = 2/9 := by
sorry

end money_division_l3260_326028


namespace EF_length_l3260_326060

/-- Configuration of line segments AB, CD, and EF -/
structure Configuration where
  AB_length : ℝ
  CD_length : ℝ
  EF_start_x : ℝ
  EF_end_x : ℝ
  AB_height : ℝ
  CD_height : ℝ
  EF_height : ℝ

/-- Conditions for the configuration -/
def valid_configuration (c : Configuration) : Prop :=
  c.AB_length = 120 ∧
  c.CD_length = 80 ∧
  c.EF_start_x = c.CD_length / 2 ∧
  c.EF_end_x = c.CD_length ∧
  c.AB_height > c.EF_height ∧
  c.EF_height > c.CD_height ∧
  c.EF_height = (c.AB_height + c.CD_height) / 2

/-- Theorem: The length of EF is 40 cm -/
theorem EF_length (c : Configuration) (h : valid_configuration c) : 
  c.EF_end_x - c.EF_start_x = 40 := by
  sorry

end EF_length_l3260_326060


namespace unique_solution_natural_equation_l3260_326005

theorem unique_solution_natural_equation :
  ∀ (a b x y : ℕ),
    x^(a + b) + y = x^a * y^b →
    (x = 2 ∧ y = 4) ∧ (∀ (x' y' : ℕ), x'^(a + b) + y' = x'^a * y'^b → x' = 2 ∧ y' = 4) := by
  sorry

end unique_solution_natural_equation_l3260_326005


namespace asymptote_slope_l3260_326041

-- Define the hyperbola parameters
def m : ℝ := 2

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / (m^2 + 12) - y^2 / (5*m - 1) = 1

-- Define the length of the real axis
def real_axis_length : ℝ := 8

-- Theorem statement
theorem asymptote_slope :
  hyperbola x y ∧ real_axis_length = 8 →
  ∃ (k : ℝ), k = 3/4 ∧ (y = k*x ∨ y = -k*x) :=
sorry

end asymptote_slope_l3260_326041


namespace sum_of_gcd_and_binary_l3260_326014

theorem sum_of_gcd_and_binary : ∃ (a b : ℕ),
  (Nat.gcd 98 63 = a) ∧
  (((1 : ℕ) * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = b) ∧
  (a + b = 58) := by
  sorry

end sum_of_gcd_and_binary_l3260_326014


namespace arithmetic_geometric_sequence_sum_l3260_326056

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 2) = a (n + 1) * r

theorem arithmetic_geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_ag : ArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end arithmetic_geometric_sequence_sum_l3260_326056


namespace average_w_x_is_half_l3260_326091

theorem average_w_x_is_half 
  (w x y : ℝ) 
  (h1 : 5 / w + 5 / x = 5 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end average_w_x_is_half_l3260_326091


namespace vacation_cost_splitting_l3260_326083

/-- Prove that the difference between what Tom and Dorothy owe Sammy is 20 dollars -/
theorem vacation_cost_splitting (tom_paid dorothy_paid sammy_paid t d : ℚ) : 
  tom_paid = 105 →
  dorothy_paid = 125 →
  sammy_paid = 175 →
  (tom_paid + dorothy_paid + sammy_paid) / 3 = tom_paid + t →
  (tom_paid + dorothy_paid + sammy_paid) / 3 = dorothy_paid + d →
  t - d = 20 := by
sorry


end vacation_cost_splitting_l3260_326083


namespace inequality_system_solution_l3260_326080

theorem inequality_system_solution (x : ℝ) :
  (-9 * x^2 + 12 * x + 5 > 0) ∧ (3 * x - 1 < 0) ↔ x < -1/3 :=
by sorry

end inequality_system_solution_l3260_326080


namespace lcm_6_15_l3260_326035

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end lcm_6_15_l3260_326035


namespace no_initial_values_satisfy_conditions_l3260_326068

/-- A sequence defined by the given recurrence relation -/
def RecurrenceSequence (x₀ x₁ : ℚ) : ℕ → ℚ
  | 0 => x₀
  | 1 => x₁
  | (n + 2) => (RecurrenceSequence x₀ x₁ n * RecurrenceSequence x₀ x₁ (n + 1)) /
               (3 * RecurrenceSequence x₀ x₁ n - 2 * RecurrenceSequence x₀ x₁ (n + 1))

/-- The property of a sequence containing infinitely many natural numbers -/
def ContainsInfinitelyManyNaturals (seq : ℕ → ℚ) : Prop :=
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ ∃ m : ℕ, seq n = m

/-- The main theorem stating that no initial values satisfy the conditions -/
theorem no_initial_values_satisfy_conditions :
  ¬∃ (x₀ x₁ : ℚ), ContainsInfinitelyManyNaturals (RecurrenceSequence x₀ x₁) :=
sorry

end no_initial_values_satisfy_conditions_l3260_326068


namespace solution_product_l3260_326066

theorem solution_product (p q : ℝ) : 
  (p - 6) * (2 * p + 10) = p^2 - 15 * p + 56 →
  (q - 6) * (2 * q + 10) = q^2 - 15 * q + 56 →
  p ≠ q →
  (p + 4) * (q + 4) = -40 := by
sorry

end solution_product_l3260_326066


namespace max_profit_at_nine_profit_function_correct_max_profit_at_nine_explicit_l3260_326020

-- Define the profit function
def profit (x : ℝ) : ℝ := x^3 - 30*x^2 + 288*x - 864

-- Define the theorem
theorem max_profit_at_nine :
  ∀ x ∈ Set.Icc 9 11,
    profit x ≤ profit 9 ∧
    profit 9 = 27 := by
  sorry

-- Define the selling price range
def selling_price_range : Set ℝ := Set.Icc 9 11

-- Define the annual sales volume function
def annual_sales (x : ℝ) : ℝ := (12 - x)^2

-- State that the profit function is correct
theorem profit_function_correct :
  ∀ x ∈ selling_price_range,
    profit x = (x - 6) * annual_sales x := by
  sorry

-- State that the maximum profit occurs at x = 9
theorem max_profit_at_nine_explicit :
  ∃ x ∈ selling_price_range,
    ∀ y ∈ selling_price_range,
      profit y ≤ profit x ∧
      x = 9 := by
  sorry

end max_profit_at_nine_profit_function_correct_max_profit_at_nine_explicit_l3260_326020


namespace nail_count_proof_l3260_326019

/-- The number of nails Violet has -/
def violet_nails : ℕ := 27

/-- The number of nails Tickletoe has -/
def tickletoe_nails : ℕ := (violet_nails - 3) / 2

/-- The number of nails SillySocks has -/
def sillysocks_nails : ℕ := 3 * tickletoe_nails - 2

/-- The total number of nails -/
def total_nails : ℕ := violet_nails + tickletoe_nails + sillysocks_nails

theorem nail_count_proof :
  total_nails = 73 :=
sorry

end nail_count_proof_l3260_326019


namespace prime_triplet_divisiblity_l3260_326053

theorem prime_triplet_divisiblity (p q r : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) →
  ((p = 2 ∧ q = 5 ∧ r = 3) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 5)) :=
sorry

end prime_triplet_divisiblity_l3260_326053


namespace infinitely_many_primes_with_solutions_l3260_326023

theorem infinitely_many_primes_with_solutions : 
  ¬ (∃ (S : Finset Nat), ∀ (p : Nat), 
    (Nat.Prime p ∧ (∃ (x y : ℤ), x^2 + x + 1 = p * y)) → p ∈ S) := by
  sorry

end infinitely_many_primes_with_solutions_l3260_326023


namespace x_142_equals_1995_unique_l3260_326011

def p (x : ℕ) : ℕ := sorry

def q (x : ℕ) : ℕ := 
  if p x = 2 then 1 else sorry

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n * p (x n) / q (x n)

theorem x_142_equals_1995_unique : 
  (x 142 = 1995) ∧ (∀ n : ℕ, n ≠ 142 → x n ≠ 1995) := by sorry

end x_142_equals_1995_unique_l3260_326011


namespace probability_two_defective_shipment_l3260_326079

/-- The probability of selecting two defective smartphones at random from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℝ :=
  let p1 := defective / total
  let p2 := (defective - 1) / (total - 1)
  p1 * p2

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_shipment :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  |probability_two_defective 250 76 - 0.0915632| < ε :=
sorry

end probability_two_defective_shipment_l3260_326079


namespace det_special_matrix_l3260_326076

-- Define the matrix as a function of y
def matrix (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![y + 1, y, y],
    ![y, y + 1, y],
    ![y, y, y + 1]]

-- State the theorem
theorem det_special_matrix (y : ℝ) :
  Matrix.det (matrix y) = 3 * y + 1 := by
  sorry

end det_special_matrix_l3260_326076


namespace least_integer_with_12_factors_and_consecutive_primes_l3260_326064

/-- A function that returns the number of positive factors of a given natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if two prime numbers are consecutive -/
def are_consecutive_primes (p q : ℕ) : Prop := sorry

/-- The theorem stating that 36 is the least positive integer with exactly 12 factors
    and consecutive prime factors -/
theorem least_integer_with_12_factors_and_consecutive_primes :
  ∀ n : ℕ, n > 0 → num_factors n = 12 →
  (∃ p q : ℕ, n = p^2 * q^2 ∧ are_consecutive_primes p q) →
  n ≥ 36 := by
  sorry

end least_integer_with_12_factors_and_consecutive_primes_l3260_326064


namespace solution_set_f_greater_than_two_range_of_t_l3260_326004

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1 ∨ x < -5} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} := by sorry

end solution_set_f_greater_than_two_range_of_t_l3260_326004


namespace thirtieth_term_of_specific_sequence_l3260_326067

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem thirtieth_term_of_specific_sequence :
  let a₁ := 3
  let a₂ := 7
  let d := a₂ - a₁
  arithmeticSequence a₁ d 30 = 119 := by
  sorry

end thirtieth_term_of_specific_sequence_l3260_326067


namespace tan_alpha_equals_one_l3260_326063

theorem tan_alpha_equals_one (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 2 * Real.sin (α - 15 * π / 180) - 1 = 0) : Real.tan α = 1 := by
  sorry

end tan_alpha_equals_one_l3260_326063


namespace infinite_sum_equals_nine_eighties_l3260_326049

/-- The infinite sum of 2n / (n^4 + 16) from n=1 to infinity equals 9/80 -/
theorem infinite_sum_equals_nine_eighties :
  (∑' n : ℕ+, (2 * n : ℝ) / (n^4 + 16)) = 9 / 80 := by sorry

end infinite_sum_equals_nine_eighties_l3260_326049


namespace max_red_socks_l3260_326098

def is_valid_sock_distribution (r b y : ℕ) : Prop :=
  let t := r + b + y
  t ≤ 2300 ∧
  (r * (r - 1) * (r - 2) + b * (b - 1) * (b - 2) + y * (y - 1) * (y - 2)) * 3 =
  t * (t - 1) * (t - 2)

theorem max_red_socks :
  ∀ r b y : ℕ, is_valid_sock_distribution r b y → r ≤ 897 :=
by sorry

end max_red_socks_l3260_326098


namespace games_played_l3260_326085

-- Define the total points scored
def total_points : ℝ := 120.0

-- Define the points scored per game
def points_per_game : ℝ := 12

-- Theorem to prove
theorem games_played : (total_points / points_per_game : ℝ) = 10 := by
  sorry

end games_played_l3260_326085


namespace factor_3x_squared_minus_75_l3260_326084

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_3x_squared_minus_75_l3260_326084


namespace cost_of_four_enchiladas_five_tacos_l3260_326046

/-- The price of an enchilada -/
def enchilada_price : ℝ := sorry

/-- The price of a taco -/
def taco_price : ℝ := sorry

/-- The first condition: 5 enchiladas and 2 tacos cost $4.30 -/
axiom condition1 : 5 * enchilada_price + 2 * taco_price = 4.30

/-- The second condition: 4 enchiladas and 3 tacos cost $4.50 -/
axiom condition2 : 4 * enchilada_price + 3 * taco_price = 4.50

/-- The theorem to prove -/
theorem cost_of_four_enchiladas_five_tacos :
  4 * enchilada_price + 5 * taco_price = 6.01 := by sorry

end cost_of_four_enchiladas_five_tacos_l3260_326046


namespace star_two_neg_three_l3260_326070

-- Define the new operation *
def star (a b : ℝ) : ℝ := a * b - (a + b)

-- Theorem statement
theorem star_two_neg_three : star 2 (-3) = -5 := by
  sorry

end star_two_neg_three_l3260_326070


namespace tan_two_implies_expression_one_l3260_326075

theorem tan_two_implies_expression_one (x : ℝ) (h : Real.tan x = 2) :
  4 * (Real.sin x)^2 - 3 * (Real.sin x) * (Real.cos x) - 5 * (Real.cos x)^2 = 1 := by
  sorry

end tan_two_implies_expression_one_l3260_326075


namespace expected_digits_is_nineteen_twelfths_l3260_326038

/-- Die numbers -/
def die_numbers : List ℕ := List.range 12 |>.map (· + 5)

/-- Count of digits in a number -/
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- Expected value calculation -/
def expected_digits : ℚ :=
  (die_numbers.map digit_count).sum / die_numbers.length

/-- Theorem: Expected number of digits is 19/12 -/
theorem expected_digits_is_nineteen_twelfths :
  expected_digits = 19 / 12 := by
  sorry

end expected_digits_is_nineteen_twelfths_l3260_326038


namespace existence_of_m_l3260_326015

theorem existence_of_m (a b : ℝ) (h : a > b) : ∃ m : ℝ, a * m < b * m := by
  sorry

end existence_of_m_l3260_326015


namespace intersection_and_perpendicular_line_l3260_326051

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2*x - 3*y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := x - y = 0

theorem intersection_and_perpendicular_line :
  -- P is on both l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧ 
  -- l is perpendicular to l₂ and passes through P
  (∀ x y : ℝ, l x y → (x - P.1) * 1 + (y - P.2) * 1 = 0) :=
sorry

end intersection_and_perpendicular_line_l3260_326051


namespace complex_modulus_squared_l3260_326040

/-- Given a complex number z satisfying z + |z| = 2 + 8i, prove that |z|² = 289 -/
theorem complex_modulus_squared (z : ℂ) (h : z + Complex.abs z = 2 + 8 * Complex.I) : 
  Complex.abs z ^ 2 = 289 := by
  sorry

end complex_modulus_squared_l3260_326040
