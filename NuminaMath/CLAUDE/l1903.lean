import Mathlib

namespace num_measurable_weights_l1903_190329

/-- Represents the number of weights of each type -/
def num_weights : ℕ := 3

/-- Represents the weight values -/
def weight_values : List ℕ := [1, 5, 50]

/-- Represents the maximum number of weights that can be placed on one side of the scale -/
def max_weights_per_side : ℕ := num_weights * weight_values.length

/-- Represents the set of all possible weight combinations on one side of the scale -/
def weight_combinations : Finset (List ℕ) :=
  sorry

/-- Calculates the total weight of a combination -/
def total_weight (combination : List ℕ) : ℕ :=
  sorry

/-- Represents the set of all possible positive weight differences -/
def measurable_weights : Finset ℕ :=
  sorry

/-- The main theorem stating that the number of different positive weights
    that can be measured is 63 -/
theorem num_measurable_weights : measurable_weights.card = 63 :=
  sorry

end num_measurable_weights_l1903_190329


namespace function_difference_l1903_190349

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + k * x - 8

-- State the theorem
theorem function_difference (k : ℝ) : 
  f 5 - g k 5 = 20 → k = 53 / 5 := by
  sorry

end function_difference_l1903_190349


namespace complex_equation_solution_l1903_190322

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I ^ 2018 → z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l1903_190322


namespace tangent_slope_implies_b_over_a_equals_two_l1903_190338

/-- A quadratic function f(x) = ax² + b with a tangent line of slope 2 at (1,3) -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

/-- The derivative of f -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

theorem tangent_slope_implies_b_over_a_equals_two (a b : ℝ) :
  f a b 1 = 3 ∧ f_derivative a 1 = 2 → b / a = 2 := by
  sorry

end tangent_slope_implies_b_over_a_equals_two_l1903_190338


namespace min_value_theorem_l1903_190385

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  9 ≤ 4*a + b ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ 4*a₀ + b₀ = 9 :=
by sorry

end min_value_theorem_l1903_190385


namespace largest_element_in_S_l1903_190383

def a : ℝ := -4

def S : Set ℝ := { -2 * a^2, 5 * a, 40 / a, 3 * a^2, 2 }

theorem largest_element_in_S : ∀ x ∈ S, x ≤ (3 * a^2) := by
  sorry

end largest_element_in_S_l1903_190383


namespace factorial_sum_remainder_20_l1903_190395

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_remainder_20 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 20 = 13 := by
  sorry

end factorial_sum_remainder_20_l1903_190395


namespace quadratic_equivalence_l1903_190361

/-- A quadratic function with vertex form parameters -/
def quad_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

/-- A quadratic function in standard form -/
def quad_standard_form (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the equivalence of the quadratic function with given properties -/
theorem quadratic_equivalence :
  ∃ (a b c : ℝ),
    (∀ x, quad_vertex_form (1/2) 4 3 x = quad_standard_form a b c x) ∧
    quad_standard_form a b c 2 = 5 ∧
    a = 1/2 ∧ b = -4 ∧ c = 11 := by
  sorry

end quadratic_equivalence_l1903_190361


namespace number_equation_l1903_190353

theorem number_equation (x : ℤ) : 8 * x + 64 = 336 ↔ x = 34 := by
  sorry

end number_equation_l1903_190353


namespace problem_solution_l1903_190379

/-- Proposition A: The solution set of x^2 + (a-1)x + a^2 ≤ 0 with respect to x is empty -/
def proposition_a (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + a^2 > 0

/-- Proposition B: The function y = (2a^2 - a)^x is increasing -/
def proposition_b (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (2*a^2 - a)^x₁ < (2*a^2 - a)^x₂

/-- The set of real numbers a for which at least one of A or B is true -/
def at_least_one_true (a : ℝ) : Prop :=
  proposition_a a ∨ proposition_b a

/-- The set of real numbers a for which exactly one of A or B is true -/
def exactly_one_true (a : ℝ) : Prop :=
  (proposition_a a ∧ ¬proposition_b a) ∨ (¬proposition_a a ∧ proposition_b a)

theorem problem_solution :
  (∀ a, at_least_one_true a ↔ (a < -1/2 ∨ a > 1/3)) ∧
  (∀ a, exactly_one_true a ↔ (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end problem_solution_l1903_190379


namespace continued_fraction_solution_l1903_190343

theorem continued_fraction_solution : 
  ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 := by
  sorry

end continued_fraction_solution_l1903_190343


namespace y_in_terms_of_x_l1903_190358

theorem y_in_terms_of_x (x y : ℝ) (h : y - 2*x = 5) : y = 2*x + 5 := by
  sorry

end y_in_terms_of_x_l1903_190358


namespace candle_burning_time_l1903_190388

-- Define the original length of the candle
def original_length : ℝ := 12

-- Define the rate of decrease in length per minute
def rate_of_decrease : ℝ := 0.08

-- Define the function for remaining length after x minutes
def remaining_length (x : ℝ) : ℝ := original_length - rate_of_decrease * x

-- Theorem statement
theorem candle_burning_time :
  ∃ (max_time : ℝ), max_time = 150 ∧ remaining_length max_time = 0 :=
sorry

end candle_burning_time_l1903_190388


namespace triangle_problem_l1903_190381

theorem triangle_problem (A B C a b c : Real) (t : Real) :
  -- Conditions
  (A + B + C = π) →
  (2 * B = A + C) →
  (b = Real.sqrt 7) →
  (a = 3) →
  (t = Real.sin A * Real.sin C) →
  -- Conclusions
  (c = 4 ∧ t ≤ Real.sqrt 3 / 2) :=
by sorry

end triangle_problem_l1903_190381


namespace complex_fraction_evaluation_l1903_190394

theorem complex_fraction_evaluation : 
  1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5)))))) = 968/3191 := by
  sorry

end complex_fraction_evaluation_l1903_190394


namespace geometric_sequence_a5_l1903_190341

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 * a 11 = 16 →
  a 5 = 1 := by
  sorry

end geometric_sequence_a5_l1903_190341


namespace newscast_advertising_time_l1903_190357

theorem newscast_advertising_time (total_time national_news international_news sports weather : ℕ)
  (h_total : total_time = 30)
  (h_national : national_news = 12)
  (h_international : international_news = 5)
  (h_sports : sports = 5)
  (h_weather : weather = 2) :
  total_time - (national_news + international_news + sports + weather) = 6 := by
  sorry

end newscast_advertising_time_l1903_190357


namespace added_number_after_doubling_l1903_190386

theorem added_number_after_doubling (x : ℝ) : 
  3 * (2 * 7 + x) = 69 → x = 9 := by
sorry

end added_number_after_doubling_l1903_190386


namespace average_difference_l1903_190327

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 80 + x) / 3) + 5 → x = 15 := by
  sorry

end average_difference_l1903_190327


namespace factor_expression_l1903_190318

theorem factor_expression (a b c : ℝ) :
  a^4*(b^3 - c^3) + b^4*(c^3 - a^3) + c^4*(a^3 - b^3) = 
  (a-b)*(b-c)*(c-a)*(a^2 + a*b + a*c + b^2 + b*c + c^2) := by
sorry

end factor_expression_l1903_190318


namespace triple_angle_sine_sin_18_degrees_l1903_190342

open Real

-- Define the sum of sines formula
axiom sum_of_sines (α β : ℝ) : sin (α + β) = sin α * cos β + cos α * sin β

-- Define the double angle formula for sine
axiom double_angle_sine (α : ℝ) : sin (2 * α) = 2 * sin α * cos α

-- Define the relation between sine and cosine
axiom sine_cosine_relation (α : ℝ) : sin α = cos (π / 2 - α)

-- Theorem 1: Triple angle formula for sine
theorem triple_angle_sine (α : ℝ) : sin (3 * α) = 3 * sin α - 4 * (sin α)^3 := by sorry

-- Theorem 2: Value of sin 18°
theorem sin_18_degrees : sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by sorry

end triple_angle_sine_sin_18_degrees_l1903_190342


namespace s_tends_to_infinity_l1903_190377

/-- Sum of digits in the decimal expansion of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- s_n is the sum of digits in the decimal expansion of 2^n -/
def s (n : ℕ) : ℕ := sum_of_digits (2^n)

/-- The sequence (s_n) tends to infinity -/
theorem s_tends_to_infinity : ∀ k : ℕ, ∃ N : ℕ, ∀ n ≥ N, s n ≥ k := by
  sorry

end s_tends_to_infinity_l1903_190377


namespace polygon_sides_from_angle_sum_l1903_190315

theorem polygon_sides_from_angle_sum (sum_of_angles : ℕ) (h : sum_of_angles = 1260) :
  ∃ n : ℕ, n ≥ 3 ∧ (n - 2) * 180 = sum_of_angles ∧ n = 9 :=
by sorry

end polygon_sides_from_angle_sum_l1903_190315


namespace sophie_savings_l1903_190387

/-- The amount of money saved in a year by not buying dryer sheets -/
def money_saved_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (cost_per_box : ℚ) : ℚ :=
  let sheets_per_week : ℕ := loads_per_week * sheets_per_load
  let sheets_per_year : ℕ := sheets_per_week * 52
  let boxes_per_year : ℚ := (sheets_per_year : ℚ) / (sheets_per_box : ℚ)
  boxes_per_year * cost_per_box

/-- Theorem stating the amount of money saved per year -/
theorem sophie_savings : money_saved_per_year 4 1 104 (11/2) = 11 := by
  sorry

end sophie_savings_l1903_190387


namespace min_sum_of_coefficients_l1903_190390

theorem min_sum_of_coefficients (b c : ℕ+) 
  (h1 : ∃ (x y : ℝ), x ≠ y ∧ 2 * x^2 + b * x + c = 0 ∧ 2 * y^2 + b * y + c = 0)
  (h2 : ∃ (x y : ℝ), x - y = 30 ∧ 2 * x^2 + b * x + c = 0 ∧ 2 * y^2 + b * y + c = 0) :
  (∀ (b' c' : ℕ+), 
    (∃ (x y : ℝ), x ≠ y ∧ 2 * x^2 + b' * x + c' = 0 ∧ 2 * y^2 + b' * y + c' = 0) →
    (∃ (x y : ℝ), x - y = 30 ∧ 2 * x^2 + b' * x + c' = 0 ∧ 2 * y^2 + b' * y + c' = 0) →
    b'.val + c'.val ≥ b.val + c.val) →
  b.val + c.val = 126 := by
sorry

end min_sum_of_coefficients_l1903_190390


namespace mrs_lovely_class_l1903_190316

/-- The number of students in Mrs. Lovely's class -/
def total_students : ℕ := 23

/-- The number of girls in the class -/
def girls : ℕ := 10

/-- The number of boys in the class -/
def boys : ℕ := girls + 3

/-- The total number of chocolates brought -/
def total_chocolates : ℕ := 500

/-- The number of chocolates left after distribution -/
def leftover_chocolates : ℕ := 10

theorem mrs_lovely_class :
  (girls * girls + boys * boys = total_chocolates - leftover_chocolates) ∧
  (girls + boys = total_students) := by
  sorry

end mrs_lovely_class_l1903_190316


namespace circle_triangle_count_l1903_190356

def points : ℕ := 9

def total_combinations : ℕ := Nat.choose points 3

def consecutive_triangles : ℕ := points

def valid_triangles : ℕ := total_combinations - consecutive_triangles

theorem circle_triangle_count :
  valid_triangles = 75 := by sorry

end circle_triangle_count_l1903_190356


namespace system_solution_l1903_190331

theorem system_solution :
  let x : ℚ := -89/43
  let y : ℚ := -202/129
  (4 * x - 3 * y = -14) ∧ (5 * x + 7 * y = 3) := by
  sorry

end system_solution_l1903_190331


namespace coordinate_square_area_l1903_190314

/-- A square in the coordinate plane with y-coordinates between 3 and 8 -/
structure CoordinateSquare where
  lowest_y : ℝ
  highest_y : ℝ
  is_square : lowest_y = 3 ∧ highest_y = 8

/-- The area of a CoordinateSquare is 25 -/
theorem coordinate_square_area (s : CoordinateSquare) : (s.highest_y - s.lowest_y) ^ 2 = 25 :=
sorry

end coordinate_square_area_l1903_190314


namespace derivative_f_at_one_l1903_190308

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x

theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 := by sorry

end derivative_f_at_one_l1903_190308


namespace total_cost_calculation_l1903_190368

theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) : 
  sandwich_price = 245/100 →
  soda_price = 87/100 →
  sandwich_quantity = 2 →
  soda_quantity = 4 →
  (sandwich_price * sandwich_quantity + soda_price * soda_quantity : ℚ) = 838/100 := by
  sorry

end total_cost_calculation_l1903_190368


namespace remainder_problem_l1903_190321

theorem remainder_problem (N : ℤ) : ∃ k : ℤ, N = 35 * k + 25 → ∃ m : ℤ, N = 15 * m + 10 := by
  sorry

end remainder_problem_l1903_190321


namespace octahedron_sum_l1903_190344

/-- A regular octahedron with numbers from 1 to 12 on its vertices -/
structure NumberedOctahedron where
  /-- The assignment of numbers to vertices -/
  vertex_numbers : Fin 6 → Fin 12
  /-- The property that each number from 1 to 12 is used exactly once -/
  all_numbers_used : Function.Injective vertex_numbers

/-- The sum of numbers on a face of the octahedron -/
def face_sum (o : NumberedOctahedron) (face : Fin 8) : ℕ := sorry

/-- The property that all face sums are equal -/
def all_face_sums_equal (o : NumberedOctahedron) : Prop :=
  ∀ (face1 face2 : Fin 8), face_sum o face1 = face_sum o face2

theorem octahedron_sum (o : NumberedOctahedron) (h : all_face_sums_equal o) :
  ∃ (face : Fin 8), face_sum o face = 39 := by sorry

end octahedron_sum_l1903_190344


namespace students_not_playing_sports_l1903_190399

theorem students_not_playing_sports (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (both : ℕ) : 
  total = 20 ∧ 
  basketball = total / 2 ∧ 
  volleyball = total * 2 / 5 ∧ 
  both = total / 10 → 
  total - (basketball + volleyball - both) = 4 :=
by sorry

end students_not_playing_sports_l1903_190399


namespace gloria_cypress_price_l1903_190307

/-- The amount Gloria gets for each cypress tree -/
def cypress_price : ℕ := sorry

theorem gloria_cypress_price :
  let cabin_price : ℕ := 129000
  let initial_cash : ℕ := 150
  let num_cypress : ℕ := 20
  let num_pine : ℕ := 600
  let num_maple : ℕ := 24
  let maple_price : ℕ := 300
  let pine_price : ℕ := 200
  let remaining_cash : ℕ := 350

  cypress_price * num_cypress + 
  pine_price * num_pine + 
  maple_price * num_maple + 
  initial_cash = 
  cabin_price + remaining_cash →
  
  cypress_price = 100 :=
by sorry

end gloria_cypress_price_l1903_190307


namespace kopeck_ruble_equivalence_l1903_190378

/-- Represents the denominations of coins available in kopecks -/
def coin_denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

/-- Represents a collection of coins, where each natural number is the count of coins for the corresponding denomination -/
def Coins := List ℕ

/-- Calculates the total value of a collection of coins in kopecks -/
def total_value (coins : Coins) : ℕ :=
  List.sum (List.zipWith (· * ·) coins coin_denominations)

/-- Calculates the total number of coins in a collection -/
def total_count (coins : Coins) : ℕ :=
  List.sum coins

theorem kopeck_ruble_equivalence (k m : ℕ) (coins : Coins) 
    (h1 : total_count coins = k)
    (h2 : total_value coins = m) :
  ∃ (new_coins : Coins), total_count new_coins = m ∧ total_value new_coins = k * 100 := by
  sorry

end kopeck_ruble_equivalence_l1903_190378


namespace inequality_proof_l1903_190326

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end inequality_proof_l1903_190326


namespace hyperbola_eccentricity_l1903_190397

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 (a > 0, b > 0)
    is √2, given that one of its asymptotes is parallel to the line x - y + 3 = 0. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_asymptote : ∃ (k : ℝ), b / a = k ∧ 1 = k) : 
    Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l1903_190397


namespace log_sine_absolute_value_sum_l1903_190309

theorem log_sine_absolute_value_sum (x : ℝ) (θ : ℝ) 
  (h : Real.log x / Real.log 2 = 2 + Real.sin θ) : 
  |x + 1| + |x - 10| = 11 := by
  sorry

end log_sine_absolute_value_sum_l1903_190309


namespace exists_all_met_l1903_190348

-- Define a type for participants
variable (Participant : Type)

-- Define a relation for "has met"
variable (has_met : Participant → Participant → Prop)

-- Define the number of participants
variable (n : ℕ)

-- Assume there are at least 4 participants
variable (h_n : n ≥ 4)

-- Define the set of all participants
variable (participants : Finset Participant)

-- Assume the number of participants matches n
variable (h_card : participants.card = n)

-- State the condition that among any 4 participants, one has met the other 3
variable (h_four_met : ∀ (a b c d : Participant), a ∈ participants → b ∈ participants → c ∈ participants → d ∈ participants →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (has_met a b ∧ has_met a c ∧ has_met a d) ∨
  (has_met b a ∧ has_met b c ∧ has_met b d) ∨
  (has_met c a ∧ has_met c b ∧ has_met c d) ∨
  (has_met d a ∧ has_met d b ∧ has_met d c))

-- Theorem statement
theorem exists_all_met :
  ∃ (x : Participant), x ∈ participants ∧ ∀ (y : Participant), y ∈ participants → y ≠ x → has_met x y :=
sorry

end exists_all_met_l1903_190348


namespace problem_statement_l1903_190312

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -5932 := by
  sorry

end problem_statement_l1903_190312


namespace triangle_abc_properties_l1903_190384

theorem triangle_abc_properties (A B C : Real) (R : Real) (BC AC : Real) :
  0 < R →
  0 < BC →
  0 < AC →
  C = 3 * Real.pi / 4 →
  Real.sin (A + C) = (BC / R) * Real.cos (A + B) →
  (1 / 2) * BC * AC * Real.sin C = 1 →
  (BC * AC = AC * (2 * BC)) ∧ 
  (AC * BC = A + B) ∧
  AC ^ 2 = 10 :=
by sorry

end triangle_abc_properties_l1903_190384


namespace constant_phi_is_cone_l1903_190311

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConstantPhiSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Define a cone (we'll use a simplified definition for this statement)
def Cone : Set SphericalCoord := sorry

-- Theorem statement
theorem constant_phi_is_cone (c : ℝ) : 
  ConstantPhiSet c = Cone := by sorry

end constant_phi_is_cone_l1903_190311


namespace jesse_blocks_l1903_190398

theorem jesse_blocks (cityscape farmhouse zoo fence1 fence2 fence3 left : ℕ) 
  (h1 : cityscape = 80)
  (h2 : farmhouse = 123)
  (h3 : zoo = 95)
  (h4 : fence1 = 57)
  (h5 : fence2 = 43)
  (h6 : fence3 = 62)
  (h7 : left = 84) :
  cityscape + farmhouse + zoo + fence1 + fence2 + fence3 + left = 544 := by
  sorry

end jesse_blocks_l1903_190398


namespace f_inf_fixed_point_l1903_190310

variable {A : Type*} [Fintype A]
variable (f : A → A)

def f_n : ℕ → (Set A) → Set A
  | 0, S => S
  | n + 1, S => f '' (f_n n S)

def f_inf (S : Set A) : Set A :=
  ⋂ n, f_n f n S

theorem f_inf_fixed_point (S : Set A) :
  f '' (f_inf f S) = f_inf f S := by sorry

end f_inf_fixed_point_l1903_190310


namespace roberto_chicken_investment_l1903_190393

def initial_cost : ℝ := 25 + 30 + 22 + 35
def weekly_feed_cost : ℝ := 1.5 + 1.3 + 1.1 + 0.9
def weekly_egg_production : ℕ := 4 + 3 + 5 + 2
def previous_egg_cost : ℝ := 2

def break_even_weeks : ℕ := 40

theorem roberto_chicken_investment (w : ℕ) :
  w = break_even_weeks ↔ 
  initial_cost + w * weekly_feed_cost = w * previous_egg_cost :=
sorry

end roberto_chicken_investment_l1903_190393


namespace min_S_value_l1903_190367

/-- Represents a 10x10 table arrangement of numbers from 1 to 100 -/
def Arrangement := Fin 10 → Fin 10 → Fin 100

/-- Checks if two positions in the table are adjacent -/
def isAdjacent (p1 p2 : Fin 10 × Fin 10) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if an arrangement satisfies the adjacent sum condition -/
def satisfiesCondition (arr : Arrangement) (S : ℕ) : Prop :=
  ∀ p1 p2 : Fin 10 × Fin 10, isAdjacent p1 p2 →
    (arr p1.1 p1.2).val + (arr p2.1 p2.2).val ≤ S

/-- The main theorem stating the minimum value of S -/
theorem min_S_value :
  (∃ (arr : Arrangement), satisfiesCondition arr 106) ∧
  (∀ S : ℕ, S < 106 → ¬∃ (arr : Arrangement), satisfiesCondition arr S) :=
sorry

end min_S_value_l1903_190367


namespace triangle_area_l1903_190391

/-- Given a triangle with perimeter 36 and inradius 2.5, its area is 45 -/
theorem triangle_area (p : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : p = 36) -- perimeter is 36
  (h2 : r = 2.5) -- inradius is 2.5
  (h3 : A = r * (p / 2)) -- area formula: A = r * s, where s is semiperimeter (p / 2)
  : A = 45 := by
  sorry

end triangle_area_l1903_190391


namespace obtuse_triangle_consecutive_sides_l1903_190302

theorem obtuse_triangle_consecutive_sides :
  ∀ (a b c : ℕ), 
    (a < b) → 
    (b < c) → 
    (c = a + 2) → 
    (a^2 + b^2 < c^2) → 
    (a = 2 ∧ b = 3 ∧ c = 4) :=
by sorry

end obtuse_triangle_consecutive_sides_l1903_190302


namespace geometric_sequence_property_l1903_190374

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : GeometricSequence a) 
  (h_sum : a 4 + a 8 = -3) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 := by
  sorry

end geometric_sequence_property_l1903_190374


namespace cookies_left_to_take_home_l1903_190359

-- Define the initial number of cookies
def initial_cookies : ℕ := 120

-- Define the number of cookies in a dozen
def cookies_per_dozen : ℕ := 12

-- Define the number of dozens sold in the morning
def morning_dozens_sold : ℕ := 3

-- Define the number of cookies sold during lunch
def lunch_cookies_sold : ℕ := 57

-- Define the number of cookies sold in the afternoon
def afternoon_cookies_sold : ℕ := 16

-- Theorem statement
theorem cookies_left_to_take_home :
  initial_cookies - (morning_dozens_sold * cookies_per_dozen + lunch_cookies_sold + afternoon_cookies_sold) = 11 := by
  sorry

end cookies_left_to_take_home_l1903_190359


namespace intersection_distance_l1903_190362

/-- Given a line y = kx - 2 intersecting a parabola y^2 = 8x at two points,
    if the x-coordinate of the midpoint of these points is 2,
    then the distance between these points is 2√15. -/
theorem intersection_distance (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧
    y₁^2 = 8*x₁ ∧ y₂^2 = 8*x₂ ∧
    y₁ = k*x₁ - 2 ∧ y₂ = k*x₂ - 2 ∧
    (x₁ + x₂) / 2 = 2) →
  ∃ A B : ℝ × ℝ, 
    A.1 ≠ B.1 ∧
    A.2^2 = 8*A.1 ∧ B.2^2 = 8*B.1 ∧
    A.2 = k*A.1 - 2 ∧ B.2 = k*B.1 - 2 ∧
    (A.1 + B.1) / 2 = 2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2 : ℝ) = 2 * (15^(1/2 : ℝ)) :=
by sorry

end intersection_distance_l1903_190362


namespace organization_size_l1903_190375

/-- Represents a committee in the organization -/
def Committee := Fin 6

/-- Represents a member of the organization -/
structure Member where
  committees : Finset Committee
  member_in_three : committees.card = 3

/-- The organization with its members -/
structure Organization where
  members : Finset Member
  all_triples_covered : ∀ (c1 c2 c3 : Committee), c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 →
    ∃! m : Member, m ∈ members ∧ c1 ∈ m.committees ∧ c2 ∈ m.committees ∧ c3 ∈ m.committees

theorem organization_size (org : Organization) : org.members.card = 20 := by
  sorry

end organization_size_l1903_190375


namespace find_b_value_l1903_190325

/-- Given the equation a * b * c = ( √ ( a + 2 ) ( b + 3 ) ) / ( c + 1 ),
    when a = 6, c = 3, and the left-hand side of the equation equals 3,
    prove that b = 15. -/
theorem find_b_value (a b c : ℝ) :
  a = 6 →
  c = 3 →
  a * b * c = ( Real.sqrt ((a + 2) * (b + 3)) ) / (c + 1) →
  a * b * c = 3 →
  b = 15 := by
  sorry


end find_b_value_l1903_190325


namespace probability_king_ace_standard_deck_l1903_190352

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (kings : Nat)
  (aces : Nat)

/-- The probability of drawing a King as the top card and an Ace as the second card -/
def probability_king_ace (d : Deck) : Rat :=
  (d.kings : Rat) / d.total_cards * d.aces / (d.total_cards - 1)

/-- Theorem: The probability of drawing a King as the top card and an Ace as the second card
    in a standard 52-card deck is 4/663 -/
theorem probability_king_ace_standard_deck :
  probability_king_ace ⟨52, 4, 4⟩ = 4 / 663 := by
  sorry

#eval probability_king_ace ⟨52, 4, 4⟩

end probability_king_ace_standard_deck_l1903_190352


namespace marble_selection_ways_l1903_190364

def total_marbles : ℕ := 20
def red_marbles : ℕ := 3
def green_marbles : ℕ := 3
def blue_marbles : ℕ := 2
def special_marbles : ℕ := red_marbles + green_marbles + blue_marbles
def other_marbles : ℕ := total_marbles - special_marbles
def chosen_marbles : ℕ := 5
def chosen_special : ℕ := 2

theorem marble_selection_ways :
  (Nat.choose red_marbles 2 +
   Nat.choose green_marbles 2 +
   Nat.choose blue_marbles 2 +
   Nat.choose red_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 +
   Nat.choose green_marbles 1 * Nat.choose blue_marbles 1) *
  Nat.choose other_marbles (chosen_marbles - chosen_special) = 6160 := by
  sorry

end marble_selection_ways_l1903_190364


namespace stratified_sample_sum_l1903_190339

/-- Represents the number of items in each food category -/
structure FoodCategories where
  grains : ℕ
  vegetableOils : ℕ
  animalDerived : ℕ
  fruitsAndVegetables : ℕ

/-- Calculates the total number of items across all categories -/
def totalItems (fc : FoodCategories) : ℕ :=
  fc.grains + fc.vegetableOils + fc.animalDerived + fc.fruitsAndVegetables

/-- Calculates the number of items to be sampled from a category in stratified sampling -/
def stratifiedSampleSize (categorySize sampleSize totalSize : ℕ) : ℕ :=
  (categorySize * sampleSize) / totalSize

/-- Theorem: In a stratified sample of 20 items from the given food categories,
    the sum of items from vegetable oils and fruits and vegetables is 6 -/
theorem stratified_sample_sum (fc : FoodCategories) 
    (h1 : fc.grains = 40)
    (h2 : fc.vegetableOils = 10)
    (h3 : fc.animalDerived = 30)
    (h4 : fc.fruitsAndVegetables = 20)
    (h5 : totalItems fc = 100)
    (sampleSize : ℕ)
    (h6 : sampleSize = 20) :
    stratifiedSampleSize fc.vegetableOils sampleSize (totalItems fc) +
    stratifiedSampleSize fc.fruitsAndVegetables sampleSize (totalItems fc) = 6 := by
  sorry


end stratified_sample_sum_l1903_190339


namespace ab_value_l1903_190376

theorem ab_value (a b : ℝ) (h : Real.sqrt (a - 1) + b^2 - 4*b + 4 = 0) : a * b = 2 := by
  sorry

end ab_value_l1903_190376


namespace john_calculation_l1903_190346

theorem john_calculation (n : ℕ) (h : n = 40) : n^2 - (n - 1)^2 = 2*n - 1 := by
  sorry

#check john_calculation

end john_calculation_l1903_190346


namespace imaginary_part_of_i_times_one_plus_i_l1903_190392

theorem imaginary_part_of_i_times_one_plus_i : 
  Complex.im (Complex.I * (1 + Complex.I)) = 1 := by sorry

end imaginary_part_of_i_times_one_plus_i_l1903_190392


namespace oil_mixture_price_l1903_190366

/-- Given two oils mixed together, prove the price of the first oil -/
theorem oil_mixture_price (volume1 volume2 : ℝ) (price2 mix_price : ℝ) (h1 : volume1 = 10)
    (h2 : volume2 = 5) (h3 : price2 = 66) (h4 : mix_price = 58.67) :
    ∃ (price1 : ℝ), price1 = 55.005 ∧ 
    volume1 * price1 + volume2 * price2 = (volume1 + volume2) * mix_price := by
  sorry

end oil_mixture_price_l1903_190366


namespace olivias_initial_money_l1903_190347

/-- Calculates the initial amount of money Olivia had given the number of card packs, their prices, and the change received. -/
def initialMoney (basketballPacks : ℕ) (basketballPrice : ℕ) (baseballDecks : ℕ) (baseballPrice : ℕ) (change : ℕ) : ℕ :=
  basketballPacks * basketballPrice + baseballDecks * baseballPrice + change

/-- Proves that Olivia's initial amount of money was $50 given the problem conditions. -/
theorem olivias_initial_money :
  initialMoney 2 3 5 4 24 = 50 := by
  sorry

end olivias_initial_money_l1903_190347


namespace min_translation_for_symmetry_l1903_190336

open Real

theorem min_translation_for_symmetry :
  let f (x m : ℝ) := Real.sqrt 3 * cos (x + m) + sin (x + m)
  ∃ (min_m : ℝ), min_m > 0 ∧
    (∀ (m : ℝ), m > 0 → 
      (∀ (x : ℝ), f x m = f (-x) m) → m ≥ min_m) ∧
    (∀ (x : ℝ), f x min_m = f (-x) min_m) ∧
    min_m = π / 6 := by
  sorry

end min_translation_for_symmetry_l1903_190336


namespace bobs_final_score_l1903_190323

/-- Bob's math knowledge competition score calculation -/
theorem bobs_final_score :
  let points_per_correct : ℕ := 5
  let points_per_incorrect : ℕ := 2
  let correct_answers : ℕ := 18
  let incorrect_answers : ℕ := 2
  let total_score := points_per_correct * correct_answers - points_per_incorrect * incorrect_answers
  total_score = 86 := by
  sorry

end bobs_final_score_l1903_190323


namespace coins_missing_fraction_l1903_190396

-- Define the initial number of coins
variable (x : ℚ)

-- Define the fractions based on the problem conditions
def lost_fraction : ℚ := 1 / 3
def found_fraction : ℚ := 5 / 6
def spent_fraction : ℚ := 1 / 4

-- Define the fraction of coins still missing
def missing_fraction : ℚ := 
  spent_fraction + (lost_fraction - lost_fraction * found_fraction)

-- Theorem to prove
theorem coins_missing_fraction : missing_fraction = 11 / 36 := by
  sorry

end coins_missing_fraction_l1903_190396


namespace wendy_phone_pictures_l1903_190340

/-- The number of pictures Wendy uploaded from her phone -/
def phone_pictures (total_albums : ℕ) (pictures_per_album : ℕ) (camera_pictures : ℕ) : ℕ :=
  total_albums * pictures_per_album - camera_pictures

/-- Proof that Wendy uploaded 22 pictures from her phone -/
theorem wendy_phone_pictures :
  phone_pictures 4 6 2 = 22 := by
  sorry

end wendy_phone_pictures_l1903_190340


namespace pool_and_deck_area_l1903_190306

/-- Calculates the total area of a rectangular pool and its surrounding deck. -/
def total_area (pool_length pool_width deck_width : ℝ) : ℝ :=
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width)

/-- Proves that the total area of a specific rectangular pool and its deck is 728 square feet. -/
theorem pool_and_deck_area :
  total_area 20 22 3 = 728 := by
  sorry

end pool_and_deck_area_l1903_190306


namespace second_divisor_problem_l1903_190332

theorem second_divisor_problem : ∃ (D N k m : ℕ+), N = 39 * k + 17 ∧ N = D * m + 4 ∧ D = 13 := by
  sorry

end second_divisor_problem_l1903_190332


namespace parallel_vectors_linear_combination_l1903_190369

/-- Given two parallel plane vectors a and b, prove their linear combination -/
theorem parallel_vectors_linear_combination (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, -8] := by
sorry

end parallel_vectors_linear_combination_l1903_190369


namespace simplify_fraction_l1903_190337

theorem simplify_fraction : (150 : ℚ) / 225 = 2 / 3 := by
  sorry

end simplify_fraction_l1903_190337


namespace negation_of_true_is_false_l1903_190305

theorem negation_of_true_is_false (p : Prop) : p → ¬p = False := by
  sorry

end negation_of_true_is_false_l1903_190305


namespace product_digits_sum_base7_l1903_190382

/-- Converts a base 7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits in base 7 of the product of 16₇ and 21₇ is equal to 3₇ --/
theorem product_digits_sum_base7 : 
  sumOfDigitsBase7 (toBase7 (toDecimal 16 * toDecimal 21)) = 3 := by sorry

end product_digits_sum_base7_l1903_190382


namespace odd_sum_probability_l1903_190330

structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  even_plus_odd : even + odd = total

def probability_odd_sum (a b : Wheel) : ℚ :=
  (a.even * b.odd + a.odd * b.even : ℚ) / (a.total * b.total : ℚ)

theorem odd_sum_probability 
  (a b : Wheel)
  (ha : a.even = a.odd)
  (hb : b.even = 3 * b.odd)
  (hta : a.total = 8)
  (htb : b.total = 8) :
  probability_odd_sum a b = 1/2 :=
sorry

end odd_sum_probability_l1903_190330


namespace f_eval_one_l1903_190363

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 200*x + c

/-- Theorem stating that f(1) = -28417 given the conditions -/
theorem f_eval_one (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -28417 := by
  sorry

end f_eval_one_l1903_190363


namespace equal_sum_and_product_sets_l1903_190304

theorem equal_sum_and_product_sets : ∃ (S₁ S₂ : Finset ℕ),
  S₁ ≠ S₂ ∧
  S₁.card = 8 ∧
  S₂.card = 8 ∧
  (S₁.sum id = S₁.prod id) ∧
  (S₂.sum id = S₂.prod id) :=
by
  sorry

end equal_sum_and_product_sets_l1903_190304


namespace shaded_area_approx_l1903_190373

/-- The area of a 4 x 6 rectangle minus a circle with diameter 2 is approximately 21 -/
theorem shaded_area_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (4 * 6 : ℝ) - Real.pi * (2 / 2)^2 = 21 + ε := by
  sorry

end shaded_area_approx_l1903_190373


namespace sum_of_numbers_l1903_190303

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9375) (h4 : y / x = 15) : 
  x + y = 400 := by sorry

end sum_of_numbers_l1903_190303


namespace vector_problem_l1903_190360

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_problem (k m : ℝ) :
  (parallel (3 • a - b) (a + k • b) → k = -1/3) ∧
  (perpendicular a (m • a - b) → m = -4/5) := by
  sorry

end vector_problem_l1903_190360


namespace ratio_problem_l1903_190335

theorem ratio_problem (x y : ℝ) : 
  (0.60 / x = 6 / 2) ∧ (x / y = 8 / 12) → x = 0.20 ∧ y = 0.30 := by
  sorry

end ratio_problem_l1903_190335


namespace floor_ceil_sum_l1903_190371

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(38.2 : ℝ)⌉ = 35 := by sorry

end floor_ceil_sum_l1903_190371


namespace estimate_eight_minus_two_sqrt_seven_l1903_190300

theorem estimate_eight_minus_two_sqrt_seven :
  2 < 8 - 2 * Real.sqrt 7 ∧ 8 - 2 * Real.sqrt 7 < 3 := by
  sorry

end estimate_eight_minus_two_sqrt_seven_l1903_190300


namespace complex_fraction_product_l1903_190354

theorem complex_fraction_product (a b : ℝ) : 
  (1 + Complex.I) / (1 - Complex.I) = Complex.mk a b → a * b = 0 := by
  sorry

end complex_fraction_product_l1903_190354


namespace parallelogram_base_length_l1903_190370

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude : ℝ) 
  (base : ℝ) 
  (h1 : area = 450) 
  (h2 : altitude = 2 * base) 
  (h3 : area = base * altitude) : 
  base = 15 := by
sorry

end parallelogram_base_length_l1903_190370


namespace range_of_m_l1903_190317

theorem range_of_m (m : ℝ) : 
  m ≠ 0 → 
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) → 
  m < -1/2 :=
by sorry

end range_of_m_l1903_190317


namespace line_passes_through_fixed_point_l1903_190301

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  (k + 1) * x - (2 * k - 1) * y + 3 * k = 0

/-- Theorem stating that the line passes through (-1, 1) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k (-1) 1 := by
  sorry

end line_passes_through_fixed_point_l1903_190301


namespace weighted_average_theorem_l1903_190334

def score1 : Rat := 55 / 100
def score2 : Rat := 67 / 100
def score3 : Rat := 76 / 100
def score4 : Rat := 82 / 100
def score5 : Rat := 85 / 100
def score6 : Rat := 48 / 60
def score7 : Rat := 150 / 200

def convertedScore6 : Rat := score6 * 100 / 60
def convertedScore7 : Rat := score7 * 100 / 200

def totalScores : Rat := score1 + score2 + score3 + score4 + score5 + convertedScore6 + convertedScore7
def numberOfScores : Nat := 7

theorem weighted_average_theorem :
  totalScores / numberOfScores = (55 + 67 + 76 + 82 + 85 + 80 + 75) / 7 := by sorry

end weighted_average_theorem_l1903_190334


namespace intersection_equality_l1903_190389

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.1 + p.2^2 ≤ 0}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 + a}

-- State the theorem
theorem intersection_equality (a : ℝ) : M ∩ N a = M ↔ a ≤ 1 - Real.sqrt 2 := by
  sorry

end intersection_equality_l1903_190389


namespace employee_share_l1903_190351

theorem employee_share (total_profit : ℝ) (num_employees : ℕ) (employer_percentage : ℝ) :
  total_profit = 50 ∧ num_employees = 9 ∧ employer_percentage = 0.1 →
  (total_profit - (employer_percentage * total_profit)) / num_employees = 5 := by
  sorry

end employee_share_l1903_190351


namespace binomial_18_6_l1903_190345

theorem binomial_18_6 : Nat.choose 18 6 = 4767 := by
  sorry

end binomial_18_6_l1903_190345


namespace sum_of_g_10_and_neg_10_l1903_190365

/-- A function g defined as a polynomial of even degree -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

/-- Theorem stating that g(10) + g(-10) = 4 given g(10) = 2 -/
theorem sum_of_g_10_and_neg_10 (a b c : ℝ) (h : g a b c 10 = 2) :
  g a b c 10 + g a b c (-10) = 4 := by
  sorry

end sum_of_g_10_and_neg_10_l1903_190365


namespace equation_value_proof_l1903_190372

theorem equation_value_proof (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 4)
  (eq2 : (2 * x + y) * (2 * z + w) = 20) :
  x * w + y * z = 8 := by
  sorry

end equation_value_proof_l1903_190372


namespace unpainted_cubes_count_l1903_190324

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  unpainted_center_size : Nat

/-- Calculates the number of unpainted unit cubes in the given PaintedCube -/
def count_unpainted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a 6x6x6 cube with 2x2 unpainted centers has 72 unpainted unit cubes -/
theorem unpainted_cubes_count (cube : PaintedCube) 
  (h1 : cube.size = 6)
  (h2 : cube.total_units = 216)
  (h3 : cube.unpainted_center_size = 2) : 
  count_unpainted_cubes cube = 72 := by
  sorry

end unpainted_cubes_count_l1903_190324


namespace cubic_expression_evaluation_l1903_190320

theorem cubic_expression_evaluation : (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := by
  sorry

end cubic_expression_evaluation_l1903_190320


namespace rectangle_dimension_change_l1903_190319

theorem rectangle_dimension_change (L W : ℝ) (L' W' : ℝ) : 
  L > 0 ∧ W > 0 →  -- Ensure positive dimensions
  L' = 1.4 * L →   -- Length increased by 40%
  L * W = L' * W' → -- Area remains constant
  (W - W') / W = 0.2857 := by
sorry

end rectangle_dimension_change_l1903_190319


namespace min_value_abs_plus_two_l1903_190313

theorem min_value_abs_plus_two (a : ℚ) : |a - 1| + 2 ≥ 2 := by
  sorry

end min_value_abs_plus_two_l1903_190313


namespace candidate_vote_percentage_l1903_190350

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 := by
  sorry

end candidate_vote_percentage_l1903_190350


namespace expand_expression_l1903_190380

theorem expand_expression (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := by
  sorry

end expand_expression_l1903_190380


namespace parallel_vectors_imply_a_equals_two_l1903_190328

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given two vectors m and n, if they are parallel, then the second component of n is 2 -/
theorem parallel_vectors_imply_a_equals_two (m n : ℝ × ℝ) 
    (hm : m = (2, 1)) 
    (hn : ∃ a : ℝ, n = (4, a)) 
    (h_parallel : are_parallel m n) : 
    n.2 = 2 := by
  sorry

end parallel_vectors_imply_a_equals_two_l1903_190328


namespace min_value_theorem_l1903_190333

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) ≥ 50/9 := by
sorry

end min_value_theorem_l1903_190333


namespace pictures_per_album_l1903_190355

/-- Given the number of pictures uploaded from a phone and a camera, and the number of albums,
    prove that the number of pictures in each album is correct. -/
theorem pictures_per_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (num_albums : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : num_albums = 5)
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 8 := by
sorry

end pictures_per_album_l1903_190355
