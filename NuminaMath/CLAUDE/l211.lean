import Mathlib

namespace NUMINAMATH_CALUDE_gloria_cypress_trees_l211_21105

def cabin_price : ℕ := 129000
def initial_cash : ℕ := 150
def final_cash : ℕ := 350
def pine_trees : ℕ := 600
def maple_trees : ℕ := 24
def pine_price : ℕ := 200
def maple_price : ℕ := 300
def cypress_price : ℕ := 100

theorem gloria_cypress_trees :
  ∃ (cypress_trees : ℕ),
    cypress_trees * cypress_price + 
    pine_trees * pine_price + 
    maple_trees * maple_price = 
    cabin_price + final_cash - initial_cash ∧
    cypress_trees = 20 := by
  sorry

end NUMINAMATH_CALUDE_gloria_cypress_trees_l211_21105


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l211_21156

/-- Represents an ellipse in a 2D Cartesian coordinate system -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  passing_point : ℝ × ℝ

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem: Given an ellipse with specific properties, prove its standard equation -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h_center : e.center = (0, 0))
  (h_left_focus : e.left_focus = (-Real.sqrt 3, 0))
  (h_passing_point : e.passing_point = (2, 0)) :
  ∀ x y : ℝ, standard_equation 4 1 x y :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l211_21156


namespace NUMINAMATH_CALUDE_polygon_with_720_degrees_is_hexagon_l211_21173

/-- A polygon with a sum of interior angles of 720° has 6 sides. -/
theorem polygon_with_720_degrees_is_hexagon :
  ∀ n : ℕ,
  (180 * (n - 2) = 720) →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_720_degrees_is_hexagon_l211_21173


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l211_21139

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m-1)*x - m*(m+2) = 0 → ∃ y : ℝ, y ≠ x ∧ y^2 - 2*(m-1)*y - m*(m+2) = 0) ∧
  ((-2)^2 - 2*(m-1)*(-2) - m*(m+2) = 0 → 2018 - 3*(m-1)^2 = 2015) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l211_21139


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l211_21196

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + 1/a + 1/b = 5 → x + y ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l211_21196


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l211_21177

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -5
  let y : ℝ := 2
  second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l211_21177


namespace NUMINAMATH_CALUDE_square_diff_inequality_l211_21115

theorem square_diff_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (a^2 + b^2) * (a - b) > (a^2 - b^2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_diff_inequality_l211_21115


namespace NUMINAMATH_CALUDE_debit_card_advantage_l211_21111

/-- Represents the benefit of using a credit card for N days -/
def credit_card_benefit (N : ℕ) : ℚ :=
  20 * N + 120

/-- Represents the benefit of using a debit card -/
def debit_card_benefit : ℚ := 240

/-- The maximum number of days for which using the debit card is more advantageous -/
def max_days_debit_advantageous : ℕ := 6

theorem debit_card_advantage :
  ∀ N : ℕ, N ≤ max_days_debit_advantageous ↔ debit_card_benefit ≥ credit_card_benefit N :=
by sorry

#check debit_card_advantage

end NUMINAMATH_CALUDE_debit_card_advantage_l211_21111


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l211_21112

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi →  -- given area
  A = Real.pi * r^2 →  -- definition of circle area
  d = 2 * r →  -- definition of diameter
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l211_21112


namespace NUMINAMATH_CALUDE_y_power_neg_x_value_l211_21123

theorem y_power_neg_x_value (x y : ℝ) (h : |y - 2*x| + (x + y - 3)^2 = 0) : y^(-x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_y_power_neg_x_value_l211_21123


namespace NUMINAMATH_CALUDE_square_side_length_l211_21102

theorem square_side_length (s : ℝ) (h : s > 0) : s^2 = 2 * (4 * s) → s = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l211_21102


namespace NUMINAMATH_CALUDE_relay_race_total_time_l211_21125

/-- The time taken by four athletes to complete a relay race -/
def relay_race_time (athlete1_time : ℕ) : ℕ :=
  let athlete2_time := athlete1_time + 10
  let athlete3_time := athlete2_time - 15
  let athlete4_time := athlete1_time - 25
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating that the total time for the relay race is 200 seconds -/
theorem relay_race_total_time : relay_race_time 55 = 200 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l211_21125


namespace NUMINAMATH_CALUDE_largest_angle_is_right_angle_l211_21124

/-- Given a triangle ABC with sides a, b, c and corresponding altitudes ha, hb, hc -/
theorem largest_angle_is_right_angle 
  (a b c ha hb hc : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ ha > 0 ∧ hb > 0 ∧ hc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_volumes : (1 : ℝ) / (ha^2 * a)^2 = 1 / (hb^2 * b)^2 + 1 / (hc^2 * c)^2) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) 
            (max (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) 
                 (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_is_right_angle_l211_21124


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l211_21143

-- Define the function f(x) = x³ + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0 ∨ p.1 = -1 ∧ p.2 = -4} =
  {p : ℝ × ℝ | f p.1 = p.2 ∧ f' p.1 = 4} :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l211_21143


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l211_21130

theorem quadratic_inequality_and_constraint (a b : ℝ) : 
  (∀ x, x < 1 ∨ x > b ↔ a * x^2 - 3 * x + 2 > 0) →
  b > 1 →
  (a = 1 ∧ b = 2) ∧
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ 8) ∧
  (∀ k, (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) ↔ -3 ≤ k ∧ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l211_21130


namespace NUMINAMATH_CALUDE_fourth_member_income_l211_21129

def family_size : ℕ := 4
def average_income : ℕ := 10000
def member1_income : ℕ := 8000
def member2_income : ℕ := 15000
def member3_income : ℕ := 6000

theorem fourth_member_income :
  let total_income := family_size * average_income
  let known_members_income := member1_income + member2_income + member3_income
  total_income - known_members_income = 11000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_member_income_l211_21129


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l211_21174

theorem complex_modulus_problem (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l211_21174


namespace NUMINAMATH_CALUDE_equations_not_equivalent_l211_21168

theorem equations_not_equivalent : 
  ¬(∀ x : ℝ, (Real.sqrt (x^2 + x - 5) = Real.sqrt (x - 1)) ↔ (x^2 + x - 5 = x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equations_not_equivalent_l211_21168


namespace NUMINAMATH_CALUDE_weight_of_six_meter_rod_l211_21133

/-- Given a uniform steel rod with specified properties, this theorem proves
    the weight of a 6 m piece of the same rod. -/
theorem weight_of_six_meter_rod (r : ℝ) (ρ : ℝ) : 
  let rod_length : ℝ := 11.25
  let rod_weight : ℝ := 42.75
  let piece_length : ℝ := 6
  let rod_volume := π * r^2 * rod_length
  let piece_volume := π * r^2 * piece_length
  let density := rod_weight / rod_volume
  piece_volume * density = 22.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_six_meter_rod_l211_21133


namespace NUMINAMATH_CALUDE_surrounded_pentagon_n_gons_l211_21159

/-- The number of sides of the central polygon -/
def m : ℕ := 5

/-- The number of surrounding polygons -/
def num_surrounding : ℕ := 5

/-- The interior angle of a regular polygon with k sides -/
def interior_angle (k : ℕ) : ℚ :=
  (k - 2 : ℚ) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
def exterior_angle (k : ℕ) : ℚ :=
  180 - interior_angle k

/-- Theorem stating that for a regular pentagon surrounded by 5 regular n-gons
    with no overlap and no gaps, n must equal 5 -/
theorem surrounded_pentagon_n_gons :
  ∃ (n : ℕ), n > 2 ∧ 
  exterior_angle m = 360 / n ∧
  num_surrounding * (360 / n) = 360 := by
  sorry

end NUMINAMATH_CALUDE_surrounded_pentagon_n_gons_l211_21159


namespace NUMINAMATH_CALUDE_classroom_benches_l211_21153

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of students that can be seated in the classroom -/
def studentsInBase7 : Nat := 321

/-- The number of students that sit on one bench -/
def studentsPerBench : Nat := 3

/-- The number of benches in the classroom -/
def numberOfBenches : Nat := base7ToBase10 studentsInBase7 / studentsPerBench

theorem classroom_benches :
  numberOfBenches = 54 := by
  sorry

end NUMINAMATH_CALUDE_classroom_benches_l211_21153


namespace NUMINAMATH_CALUDE_distance_is_49_l211_21144

/-- Represents a sign at a kilometer marker -/
structure Sign :=
  (to_yolkino : Nat)
  (to_palkino : Nat)

/-- Calculates the sum of digits of a natural number -/
def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

/-- The distance between Yolkino and Palkino -/
def distance_yolkino_palkino : Nat := sorry

theorem distance_is_49 :
  ∀ (k : Nat), k < distance_yolkino_palkino →
    ∃ (sign : Sign),
      sign.to_yolkino = k ∧
      sign.to_palkino = distance_yolkino_palkino - k ∧
      digit_sum sign.to_yolkino + digit_sum sign.to_palkino = 13 →
  distance_yolkino_palkino = 49 :=
sorry

end NUMINAMATH_CALUDE_distance_is_49_l211_21144


namespace NUMINAMATH_CALUDE_rectangular_field_area_l211_21198

/-- Calculates the area of a rectangular field given its perimeter and width. -/
theorem rectangular_field_area
  (perimeter : ℝ) (width : ℝ)
  (h_perimeter : perimeter = 30)
  (h_width : width = 5) :
  width * (perimeter / 2 - width) = 50 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l211_21198


namespace NUMINAMATH_CALUDE_six_houses_configurations_l211_21131

/-- Represents the material of a house -/
inductive Material
  | Brick
  | Wood

/-- A configuration of houses is a list of their materials -/
def Configuration := List Material

/-- Checks if a configuration is valid (no adjacent wooden houses) -/
def isValidConfiguration (config : Configuration) : Bool :=
  match config with
  | [] => true
  | [_] => true
  | Material.Wood :: Material.Wood :: _ => false
  | _ :: rest => isValidConfiguration rest

/-- Generates all possible configurations of n houses -/
def allConfigurations (n : Nat) : List Configuration :=
  match n with
  | 0 => [[]]
  | m + 1 => 
    let prev := allConfigurations m
    (prev.map (λ c => Material.Brick :: c)) ++ (prev.map (λ c => Material.Wood :: c))

/-- Counts the number of valid configurations for n houses -/
def countValidConfigurations (n : Nat) : Nat :=
  (allConfigurations n).filter isValidConfiguration |>.length

/-- The main theorem: there are 21 valid configurations for 6 houses -/
theorem six_houses_configurations :
  countValidConfigurations 6 = 21 := by
  sorry


end NUMINAMATH_CALUDE_six_houses_configurations_l211_21131


namespace NUMINAMATH_CALUDE_barbaras_candies_l211_21110

/-- Barbara's candy counting problem -/
theorem barbaras_candies (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 9 → bought = 18 → total = initial + bought → total = 27 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_candies_l211_21110


namespace NUMINAMATH_CALUDE_f_min_at_neg_seven_l211_21118

/-- The quadratic function we're minimizing -/
def f (x : ℝ) := x^2 + 14*x - 20

/-- The theorem stating that f attains its minimum at x = -7 -/
theorem f_min_at_neg_seven :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_seven_l211_21118


namespace NUMINAMATH_CALUDE_power_mod_prime_remainder_2_100_mod_101_l211_21180

theorem power_mod_prime (p : Nat) (a : Nat) (h_prime : Nat.Prime p) (h_not_div : ¬(p ∣ a)) :
  a^(p - 1) ≡ 1 [MOD p] := by sorry

theorem remainder_2_100_mod_101 :
  2^100 ≡ 1 [MOD 101] := by
  have h_prime : Nat.Prime 101 := sorry
  have h_not_div : ¬(101 ∣ 2) := sorry
  have h_fermat := power_mod_prime 101 2 h_prime h_not_div
  sorry

end NUMINAMATH_CALUDE_power_mod_prime_remainder_2_100_mod_101_l211_21180


namespace NUMINAMATH_CALUDE_equal_savings_l211_21119

theorem equal_savings (your_initial : ℕ) (your_weekly : ℕ) (friend_initial : ℕ) (friend_weekly : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  your_weekly = 7 →
  friend_initial = 210 →
  friend_weekly = 5 →
  weeks = 25 →
  your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks :=
by sorry

end NUMINAMATH_CALUDE_equal_savings_l211_21119


namespace NUMINAMATH_CALUDE_customers_per_car_l211_21138

theorem customers_per_car (num_cars : ℕ) (total_sales : ℕ) 
  (h1 : num_cars = 10) 
  (h2 : total_sales = 50) : 
  ∃ (customers_per_car : ℕ), 
    customers_per_car * num_cars = total_sales ∧ 
    customers_per_car = 5 := by
  sorry

end NUMINAMATH_CALUDE_customers_per_car_l211_21138


namespace NUMINAMATH_CALUDE_doughnut_profit_l211_21140

/-- Calculate the profit from selling doughnuts -/
theorem doughnut_profit 
  (expenses : ℕ) 
  (num_doughnuts : ℕ) 
  (price_per_doughnut : ℕ) 
  (h1 : expenses = 53)
  (h2 : num_doughnuts = 25)
  (h3 : price_per_doughnut = 3) : 
  num_doughnuts * price_per_doughnut - expenses = 22 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_profit_l211_21140


namespace NUMINAMATH_CALUDE_ellipse_equation_l211_21145

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the points A, B, and C
variable (A B C : ℝ × ℝ)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_equation 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a ≠ b)
  (h4 : ∀ x y, a * x^2 + b * y^2 = 1 ↔ (x, y) = A ∨ (x, y) = B)
  (h5 : A.1 + A.2 = 1 ∧ B.1 + B.2 = 1)
  (h6 : C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h7 : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)
  (h8 : (C.2 - O.2) / (C.1 - O.1) = Real.sqrt 2 / 2) :
  a = 1/3 ∧ b = Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l211_21145


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_right_triangle_geometric_progression_l211_21175

-- Define a right triangle with sides a, b, c
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

-- Define arithmetic progression for three numbers
def is_arithmetic_progression (x y z : ℝ) : Prop :=
  y - x = z - y

-- Define geometric progression for three numbers
def is_geometric_progression (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

theorem right_triangle_arithmetic_progression :
  ∃ t : RightTriangle, is_arithmetic_progression t.a t.b t.c ∧ t.a = 3 ∧ t.b = 4 ∧ t.c = 5 :=
sorry

theorem right_triangle_geometric_progression :
  ∃ t : RightTriangle, is_geometric_progression t.a t.b t.c ∧
    t.a = 1 ∧ t.b = Real.sqrt ((1 + Real.sqrt 5) / 2) ∧ t.c = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_right_triangle_geometric_progression_l211_21175


namespace NUMINAMATH_CALUDE_M_superset_N_l211_21154

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a ∈ M, x = a^2}

theorem M_superset_N : M ⊇ N := by
  sorry

end NUMINAMATH_CALUDE_M_superset_N_l211_21154


namespace NUMINAMATH_CALUDE_awards_distribution_l211_21161

/-- The number of ways to distribute n distinct awards to k students, where each student receives at least one award. -/
def distribute_awards (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

theorem awards_distribution :
  distribute_awards 5 3 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l211_21161


namespace NUMINAMATH_CALUDE_parallelepiped_net_theorem_l211_21146

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents a net formed from a parallelepiped -/
structure Net :=
  (squares : ℕ)

/-- Unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- Removes one square from a net -/
def remove_square (n : Net) : Net :=
  { squares := n.squares - 1 }

theorem parallelepiped_net_theorem (p : Parallelepiped) 
  (h1 : p.length = 2)
  (h2 : p.width = 1)
  (h3 : p.height = 1) :
  (remove_square (unfold p)).squares = 9 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_net_theorem_l211_21146


namespace NUMINAMATH_CALUDE_sixteen_solutions_l211_21108

/-- The function f(x) = x^2 - 3x --/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The fourth composition of f --/
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

/-- There are exactly 16 distinct real solutions to f(f(f(f(c)))) = 6 --/
theorem sixteen_solutions : ∃! (s : Finset ℝ), s.card = 16 ∧ ∀ c, c ∈ s ↔ f_4 c = 6 := by sorry

end NUMINAMATH_CALUDE_sixteen_solutions_l211_21108


namespace NUMINAMATH_CALUDE_abs_neg_three_fourths_l211_21151

theorem abs_neg_three_fourths : |(-3 : ℚ) / 4| = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_fourths_l211_21151


namespace NUMINAMATH_CALUDE_possible_values_of_a_l211_21197

theorem possible_values_of_a (a b c d : ℕ) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 3010)
  (h3 : a^2 - b^2 + c^2 - d^2 = 3010) :
  ∃! (s : Finset ℕ), s.card = 751 ∧ ∀ x, x ∈ s ↔ 
    ∃ (b' c' d' : ℕ), 
      a = x ∧
      x > b' ∧ b' > c' ∧ c' > d' ∧
      x + b' + c' + d' = 3010 ∧
      x^2 - b'^2 + c'^2 - d'^2 = 3010 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l211_21197


namespace NUMINAMATH_CALUDE_angle_C_60_not_sufficient_for_similarity_l211_21163

-- Define triangles ABC and A'B'C'
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angles of a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- State the given conditions
axiom triangle_ABC : Triangle
axiom triangle_A'B'C' : Triangle

axiom angle_B_is_right : angle triangle_ABC 1 = 90
axiom angle_B'_is_right : angle triangle_A'B'C' 1 = 90
axiom angle_A_is_30 : angle triangle_ABC 0 = 30

-- Define triangle similarity
def similar (t1 t2 : Triangle) : Prop :=
  sorry

-- State the theorem
theorem angle_C_60_not_sufficient_for_similarity :
  ¬(∀ (ABC A'B'C' : Triangle),
    angle ABC 1 = 90 →
    angle A'B'C' 1 = 90 →
    angle ABC 0 = 30 →
    angle ABC 2 = 60 →
    similar ABC A'B'C') :=
  sorry

end NUMINAMATH_CALUDE_angle_C_60_not_sufficient_for_similarity_l211_21163


namespace NUMINAMATH_CALUDE_number_raised_to_fourth_l211_21136

theorem number_raised_to_fourth : ∃ x : ℝ, 121 * x^4 = 75625 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_raised_to_fourth_l211_21136


namespace NUMINAMATH_CALUDE_commodity_sales_profit_l211_21137

/-- Profit function for a commodity sale --/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

/-- Sales quantity function --/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400

theorem commodity_sales_profit 
  (cost_price : ℝ) 
  (h_cost : cost_price = 10) 
  (h_domain : ∀ x, 0 < x → x ≤ 40 → sales_quantity x ≥ 0) :
  /- 1. Profit function is correct for the given domain -/
  (∀ x, 0 < x → x ≤ 40 → 
    profit_function x = (sales_quantity x) * (x - cost_price)) ∧
  /- 2. Selling price for $1250 profit that maximizes sales is $15 -/
  (∃ x, profit_function x = 1250 ∧ 
    sales_quantity x = (sales_quantity 15) ∧
    x = 15) ∧
  /- 3. Maximum profit when x ≥ 28 and y ≥ 50 is $2160 -/
  (∀ x, x ≥ 28 → sales_quantity x ≥ 50 → 
    profit_function x ≤ 2160) ∧
  (∃ x, x ≥ 28 ∧ sales_quantity x ≥ 50 ∧ 
    profit_function x = 2160) := by
  sorry

end NUMINAMATH_CALUDE_commodity_sales_profit_l211_21137


namespace NUMINAMATH_CALUDE_quarterback_passes_l211_21116

theorem quarterback_passes (total : ℕ) (left : ℕ) : 
  total = 50 → 
  left + 2 * left + (left + 2) = total → 
  left = 12 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l211_21116


namespace NUMINAMATH_CALUDE_median_squares_ratio_l211_21183

/-- Given a triangle with sides a, b, c and corresponding medians ma, mb, mc,
    the ratio of the sum of squares of medians to the sum of squares of sides is 3/4 -/
theorem median_squares_ratio (a b c ma mb mc : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (hmb : mb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (hmc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (ma^2 + mb^2 + mc^2) / (a^2 + b^2 + c^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_median_squares_ratio_l211_21183


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l211_21107

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l211_21107


namespace NUMINAMATH_CALUDE_simplify_expression_l211_21191

theorem simplify_expression : 
  Real.sqrt 27 - Real.sqrt (1/3) + Real.sqrt 12 = 14 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l211_21191


namespace NUMINAMATH_CALUDE_final_number_theorem_l211_21166

/-- Represents the state of the number on the board -/
structure BoardState where
  digits : List Nat
  deriving Repr

/-- Applies Operation 1 to the board state -/
def applyOperation1 (state : BoardState) : BoardState :=
  sorry

/-- Applies Operation 2 to the board state -/
def applyOperation2 (state : BoardState) : BoardState :=
  sorry

/-- Checks if a number is a valid final state (two digits) -/
def isValidFinalState (state : BoardState) : Bool :=
  sorry

/-- Generates the initial state with 100 fives -/
def initialState : BoardState :=
  { digits := List.replicate 100 5 }

/-- Theorem stating the final result of the operations -/
theorem final_number_theorem :
  ∃ (finalState : BoardState),
    (isValidFinalState finalState) ∧
    (finalState.digits = [8, 0] ∨ finalState.digits = [6, 6]) ∧
    (∃ (operations : List (BoardState → BoardState)),
      operations.foldl (λ state op => op state) initialState = finalState) :=
sorry

end NUMINAMATH_CALUDE_final_number_theorem_l211_21166


namespace NUMINAMATH_CALUDE_total_tape_area_l211_21160

/-- Calculate the total area of tape used for taping boxes -/
theorem total_tape_area (box1_length box1_width : ℕ) (box2_side : ℕ) (box3_length box3_width : ℕ)
  (box1_count box2_count box3_count : ℕ) (tape_width overlap : ℕ) :
  box1_length = 30 ∧ box1_width = 15 ∧ 
  box2_side = 40 ∧
  box3_length = 50 ∧ box3_width = 20 ∧
  box1_count = 5 ∧ box2_count = 2 ∧ box3_count = 3 ∧
  tape_width = 2 ∧ overlap = 2 →
  (box1_count * (box1_length + overlap + 2 * (box1_width + overlap)) +
   box2_count * (3 * (box2_side + overlap)) +
   box3_count * (box3_length + overlap + 2 * (box3_width + overlap))) * tape_width = 1740 := by
  sorry

end NUMINAMATH_CALUDE_total_tape_area_l211_21160


namespace NUMINAMATH_CALUDE_prime_triples_divisibility_l211_21181

theorem prime_triples_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  p ∣ q^r + 1 ∧ q ∣ r^p + 1 ∧ r ∣ p^q + 1 →
  (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_triples_divisibility_l211_21181


namespace NUMINAMATH_CALUDE_tangent_line_circle_l211_21165

-- Define the set of real numbers m+n should belong to
def tangent_range : Set ℝ :=
  {x | x ≤ 2 - 2 * Real.sqrt 2 ∨ x ≥ 2 + 2 * Real.sqrt 2}

-- Define the condition for the line to be tangent to the circle
def is_tangent (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), ((m + 1) * x + (n + 1) * y - 2 = 0) ∧
                ((x - 1)^2 + (y - 1)^2 = 1)

-- Theorem statement
theorem tangent_line_circle (m n : ℝ) :
  is_tangent m n → (m + n) ∈ tangent_range := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l211_21165


namespace NUMINAMATH_CALUDE_percentage_difference_l211_21172

theorem percentage_difference (A B : ℝ) (h : A = B * (1 + 0.25)) :
  B = A * (1 - 0.2) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l211_21172


namespace NUMINAMATH_CALUDE_constant_distance_l211_21147

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1/2 and y-intercept m -/
structure Line where
  m : ℝ

/-- Theorem stating the constant distance between points B and N -/
theorem constant_distance (E : Ellipse) (l : Line) : 
  E.a^2 * (1 / E.b^2 - 1 / E.a^2) = 3 / 4 →  -- eccentricity condition
  E.b = 1 →  -- passes through (0, 1)
  ∃ (A C : ℝ × ℝ), 
    (A.1^2 / E.a^2 + A.2^2 / E.b^2 = 1) ∧  -- A is on the ellipse
    (C.1^2 / E.a^2 + C.2^2 / E.b^2 = 1) ∧  -- C is on the ellipse
    (A.2 = A.1 / 2 + l.m) ∧  -- A is on the line
    (C.2 = C.1 / 2 + l.m) ∧  -- C is on the line
    ∃ (B D : ℝ × ℝ), 
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧  -- ABCD is a square
      (D.1 - A.1)^2 + (D.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
      (B.1 - C.1)^2 + (B.2 - C.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
      (B.1 - 2 * l.m)^2 + B.2^2 = 5 / 2  -- distance between B and N is √(5/2)
  := by sorry

end NUMINAMATH_CALUDE_constant_distance_l211_21147


namespace NUMINAMATH_CALUDE_opposite_of_pi_l211_21176

theorem opposite_of_pi : 
  ∃ (x : ℝ), x = -π ∧ x + π = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_pi_l211_21176


namespace NUMINAMATH_CALUDE_star_inequality_l211_21141

-- Define the * operation
def star (m n : Int) : Int := (m + 2) * 3 - n

-- Theorem statement
theorem star_inequality : star 2 (-2) > star (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_star_inequality_l211_21141


namespace NUMINAMATH_CALUDE_ellipse_equation_l211_21190

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1) ∧ 
  (2*a*b = 4) ∧
  (∃ (c : ℝ), a^2 - b^2 = c^2 ∧ c = Real.sqrt 3) →
  (∃ (x y : ℝ), x^2/4 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l211_21190


namespace NUMINAMATH_CALUDE_square_figure_division_l211_21114

/-- Represents a rectangular figure composed of squares -/
structure SquareFigure where
  width : ℕ
  height : ℕ
  pattern : List (List Bool)

/-- Represents a cut in the figure -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut

/-- Checks if a cut follows the sides of the squares -/
def isValidCut (figure : SquareFigure) (cut : Cut) : Prop :=
  match cut with
  | Cut.Vertical n => n > 0 ∧ n < figure.width
  | Cut.Horizontal n => n > 0 ∧ n < figure.height

/-- Checks if two cuts divide the figure into four parts -/
def dividesFourParts (figure : SquareFigure) (cut1 cut2 : Cut) : Prop :=
  isValidCut figure cut1 ∧ isValidCut figure cut2 ∧
  ((∃ n m, cut1 = Cut.Vertical n ∧ cut2 = Cut.Horizontal m) ∨
   (∃ n m, cut1 = Cut.Horizontal n ∧ cut2 = Cut.Vertical m))

/-- Checks if all parts are identical after cuts -/
def partsAreIdentical (figure : SquareFigure) (cut1 cut2 : Cut) : Prop :=
  sorry  -- Definition of identical parts

/-- Main theorem: The figure can be divided into four identical parts -/
theorem square_figure_division (figure : SquareFigure) :
  ∃ cut1 cut2, dividesFourParts figure cut1 cut2 ∧ partsAreIdentical figure cut1 cut2 :=
sorry


end NUMINAMATH_CALUDE_square_figure_division_l211_21114


namespace NUMINAMATH_CALUDE_bobs_current_time_l211_21142

/-- Given that Bob's sister runs a mile in 320 seconds, and Bob needs to improve his time by 50% to match his sister's time, prove that Bob's current time is 480 seconds. -/
theorem bobs_current_time (sister_time : ℝ) (improvement_rate : ℝ) (bob_time : ℝ) 
  (h1 : sister_time = 320)
  (h2 : improvement_rate = 0.5)
  (h3 : bob_time = sister_time + sister_time * improvement_rate) :
  bob_time = 480 := by
  sorry

end NUMINAMATH_CALUDE_bobs_current_time_l211_21142


namespace NUMINAMATH_CALUDE_compound_interest_rate_l211_21120

/-- The compound interest rate that satisfies the given conditions -/
def interest_rate : ℝ := 20

/-- The principal amount (initial deposit) -/
noncomputable def principal : ℝ := 
  3000 / (1 + interest_rate / 100) ^ 3

theorem compound_interest_rate : 
  (principal * (1 + interest_rate / 100) ^ 3 = 3000) ∧ 
  (principal * (1 + interest_rate / 100) ^ 4 = 3600) := by
  sorry

#check compound_interest_rate

end NUMINAMATH_CALUDE_compound_interest_rate_l211_21120


namespace NUMINAMATH_CALUDE_die_probability_l211_21100

/-- A fair 8-sided die -/
def Die : Finset ℕ := Finset.range 8

/-- Perfect squares from 1 to 8 -/
def PerfectSquares : Finset ℕ := {1, 4}

/-- Even numbers from 1 to 8 -/
def EvenNumbers : Finset ℕ := {2, 4, 6, 8}

/-- The probability of rolling a number that is either a perfect square or an even number -/
theorem die_probability : 
  (Finset.card (PerfectSquares ∪ EvenNumbers) : ℚ) / Finset.card Die = 5 / 8 :=
sorry

end NUMINAMATH_CALUDE_die_probability_l211_21100


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l211_21194

/-- The quadratic function f(x) = -2(x+1)^2-4 -/
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 - 4

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -4)

/-- Theorem: The vertex of f(x) = -2(x+1)^2-4 is at (-1, -4) -/
theorem vertex_of_quadratic :
  (∀ x, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l211_21194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l211_21195

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the properties of the sequence
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 2 + a 4 = 10)
  (h3 : ∃ r : ℝ, r ≠ 0 ∧ a 2 = a 1 * r ∧ a 5 = a 2 * r)
  (h4 : arithmetic_sequence a d) :
  a 1 = 1 ∧ ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l211_21195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l211_21189

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting at 2, with common difference 4, 
    and last term 2014 has 504 terms -/
theorem arithmetic_sequence_2_to_2014 : 
  arithmetic_sequence_length 2 4 2014 = 504 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l211_21189


namespace NUMINAMATH_CALUDE_cans_collected_l211_21135

theorem cans_collected (monday_cans tuesday_cans : ℕ) 
  (h1 : monday_cans = 71) 
  (h2 : tuesday_cans = 27) : 
  monday_cans + tuesday_cans = 98 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l211_21135


namespace NUMINAMATH_CALUDE_total_watching_time_l211_21187

def first_show_length : ℕ := 30
def second_show_multiplier : ℕ := 4

theorem total_watching_time :
  first_show_length + first_show_length * second_show_multiplier = 150 :=
by sorry

end NUMINAMATH_CALUDE_total_watching_time_l211_21187


namespace NUMINAMATH_CALUDE_geometry_number_theory_arrangement_l211_21104

theorem geometry_number_theory_arrangement (n_geometry : ℕ) (n_number_theory : ℕ) :
  n_geometry = 4 →
  n_number_theory = 5 →
  (number_of_arrangements : ℕ) =
    Nat.choose (n_number_theory + 1) n_geometry :=
by sorry

end NUMINAMATH_CALUDE_geometry_number_theory_arrangement_l211_21104


namespace NUMINAMATH_CALUDE_teacher_instruction_l211_21101

theorem teacher_instruction (x : ℝ) : ((x - 2) * 3 + 3) * 3 = 63 ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_teacher_instruction_l211_21101


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l211_21113

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 ∧ 
               y^2 + (k^2 - 4)*y + k - 1 = 0 ∧ 
               x = -y) → 
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l211_21113


namespace NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l211_21162

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  squares : Fin 9 → ℕ
  width : ℕ
  height : ℕ
  is_valid : width = squares 0 + squares 1 + squares 2 ∧
             height = squares 0 + squares 3 + squares 6 ∧
             width = squares 6 + squares 7 + squares 8 ∧
             height = squares 2 + squares 5 + squares 8

/-- The perimeter of a rectangle -/
def perimeter (rect : NineSquareRectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Theorem stating that the minimum perimeter of a valid NineSquareRectangle is 52 -/
theorem min_perimeter_nine_square_rectangle :
  ∃ (rect : NineSquareRectangle), perimeter rect = 52 ∧
  ∀ (other : NineSquareRectangle), perimeter other ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l211_21162


namespace NUMINAMATH_CALUDE_sqrt_56_58_fraction_existence_l211_21134

theorem sqrt_56_58_fraction_existence (q : ℕ+) :
  q ≠ 1 → q ≠ 3 → ∃ p : ℤ, Real.sqrt 56 < (p : ℚ) / q ∧ (p : ℚ) / q < Real.sqrt 58 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_56_58_fraction_existence_l211_21134


namespace NUMINAMATH_CALUDE_hypotenuse_division_l211_21171

/-- A right triangle with one acute angle of 30° and hypotenuse of length 8 -/
structure RightTriangle30 where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8 -/
  hyp_eq_8 : hypotenuse = 8
  /-- One acute angle is 30° -/
  acute_angle : ℝ
  acute_angle_eq_30 : acute_angle = 30

/-- The altitude from the right angle vertex to the hypotenuse -/
def altitude (t : RightTriangle30) : ℝ := sorry

/-- The shorter segment of the hypotenuse divided by the altitude -/
def short_segment (t : RightTriangle30) : ℝ := sorry

/-- The longer segment of the hypotenuse divided by the altitude -/
def long_segment (t : RightTriangle30) : ℝ := sorry

/-- Theorem stating that the altitude divides the hypotenuse into segments of length 4 and 6 -/
theorem hypotenuse_division (t : RightTriangle30) : 
  short_segment t = 4 ∧ long_segment t = 6 :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_division_l211_21171


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_greater_than_neg_three_fourths_l211_21157

theorem inequality_holds_iff_m_greater_than_neg_three_fourths (m : ℝ) :
  (∀ x : ℝ, m^2 * x^2 - 2*m*x > -x^2 - x - 1) ↔ m > -3/4 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_greater_than_neg_three_fourths_l211_21157


namespace NUMINAMATH_CALUDE_line_equation_proof_l211_21169

/-- A parameterization of a line in R² -/
structure LineParam where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The equation of a line in the form y = mx + b -/
structure LineEquation where
  m : ℝ
  b : ℝ

/-- Given parameterization represents a line -/
axiom is_line (p : LineParam) : True

/-- The given parameterization of the line -/
def given_param : LineParam where
  x := λ t => 2 * t + 4
  y := λ t => 4 * t - 5

/-- The equation we want to prove -/
def target_equation : LineEquation where
  m := 2
  b := -13

/-- Theorem: The given parameterized line has the equation y = 2x - 13 -/
theorem line_equation_proof :
  ∀ t : ℝ, (given_param.y t) = target_equation.m * (given_param.x t) + target_equation.b :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l211_21169


namespace NUMINAMATH_CALUDE_kelly_apples_l211_21106

/-- The number of apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly will have after picking -/
def total_apples : ℕ := 105

/-- The number of apples Kelly has now -/
def current_apples : ℕ := total_apples - apples_to_pick

theorem kelly_apples : current_apples = 56 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l211_21106


namespace NUMINAMATH_CALUDE_bill_proof_l211_21186

/-- The number of friends who can pay -/
def paying_friends : ℕ := 9

/-- The number of friends including the one who can't pay -/
def total_friends : ℕ := 10

/-- The additional amount each paying friend contributes -/
def additional_amount : ℕ := 3

/-- The total bill amount -/
def total_bill : ℕ := 270

theorem bill_proof :
  (paying_friends : ℚ) * (total_bill / total_friends + additional_amount) = total_bill := by
  sorry

end NUMINAMATH_CALUDE_bill_proof_l211_21186


namespace NUMINAMATH_CALUDE_no_power_of_three_and_five_l211_21184

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem no_power_of_three_and_five :
  ∀ n : ℕ, ∀ α β : ℕ+, v n ≠ (3 : ℤ)^(α : ℕ) * (5 : ℤ)^(β : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_no_power_of_three_and_five_l211_21184


namespace NUMINAMATH_CALUDE_range_of_a_l211_21126

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l211_21126


namespace NUMINAMATH_CALUDE_bacteria_division_theorem_l211_21150

/-- Represents a binary tree of bacteria -/
inductive BacteriaTree
  | Leaf : BacteriaTree
  | Node : BacteriaTree → BacteriaTree → BacteriaTree

/-- Counts the number of nodes in a BacteriaTree -/
def count_nodes : BacteriaTree → Nat
  | BacteriaTree.Leaf => 1
  | BacteriaTree.Node left right => count_nodes left + count_nodes right

/-- Checks if a subtree with the desired properties exists -/
def exists_balanced_subtree (tree : BacteriaTree) : Prop :=
  ∃ (subtree : BacteriaTree), 
    (count_nodes subtree ≥ 334 ∧ count_nodes subtree ≤ 667)

theorem bacteria_division_theorem (tree : BacteriaTree) 
  (h : count_nodes tree = 1000) : 
  exists_balanced_subtree tree :=
sorry

end NUMINAMATH_CALUDE_bacteria_division_theorem_l211_21150


namespace NUMINAMATH_CALUDE_no_integer_solution_l211_21148

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^137) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^117) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l211_21148


namespace NUMINAMATH_CALUDE_two_diamonds_balance_three_dots_l211_21182

-- Define the symbols
variable (triangle diamond dot : ℕ)

-- Define the balance relationships
axiom balance1 : 3 * triangle + diamond = 9 * dot
axiom balance2 : triangle = diamond + dot

-- Theorem to prove
theorem two_diamonds_balance_three_dots : 2 * diamond = 3 * dot := by
  sorry

end NUMINAMATH_CALUDE_two_diamonds_balance_three_dots_l211_21182


namespace NUMINAMATH_CALUDE_bobs_age_l211_21109

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- Theorem stating Bob's age given the problem conditions -/
theorem bobs_age (ages : SiblingAges) : 
  ages.susan = 15 ∧ 
  ages.arthur = ages.susan + 2 ∧ 
  ages.tom = ages.bob - 3 ∧ 
  ages.susan + ages.arthur + ages.tom + ages.bob = 51 →
  ages.bob = 11 := by
sorry

end NUMINAMATH_CALUDE_bobs_age_l211_21109


namespace NUMINAMATH_CALUDE_book_length_proof_l211_21199

theorem book_length_proof (pages_read : ℕ) (pages_difference : ℕ) : 
  pages_read = 2323 → pages_difference = 90 → 
  pages_read = (pages_read - pages_difference) + pages_difference → 
  pages_read + (pages_read - pages_difference) = 4556 :=
by
  sorry

end NUMINAMATH_CALUDE_book_length_proof_l211_21199


namespace NUMINAMATH_CALUDE_cos_36_degrees_l211_21155

theorem cos_36_degrees : Real.cos (36 * Real.pi / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l211_21155


namespace NUMINAMATH_CALUDE_trig_sum_zero_l211_21167

theorem trig_sum_zero (α β γ : ℝ) : 
  (Real.sin α / (Real.sin (α - β) * Real.sin (α - γ))) +
  (Real.sin β / (Real.sin (β - α) * Real.sin (β - γ))) +
  (Real.sin γ / (Real.sin (γ - α) * Real.sin (γ - β))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_zero_l211_21167


namespace NUMINAMATH_CALUDE_cube_sphere_volume_l211_21149

theorem cube_sphere_volume (n : ℕ) (hn : n > 2) : 
  (n^3 : ℝ) - (4/3 * Real.pi * (n/2)^3) = 2 * (4/3 * Real.pi * (n/2)^3) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_l211_21149


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l211_21152

/-- Represents the amount of peanut butter in tablespoons -/
def peanut_butter : ℚ := 29 + 5 / 7

/-- Represents the size of one serving in tablespoons -/
def serving_size : ℚ := 2

/-- Represents the number of servings in the jar -/
def num_servings : ℚ := peanut_butter / serving_size

/-- Theorem stating that the number of servings in the jar is 14 3/7 -/
theorem peanut_butter_servings : num_servings = 14 + 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l211_21152


namespace NUMINAMATH_CALUDE_kathryn_annie_difference_l211_21192

/-- Represents the number of pints of blueberries picked by each person -/
structure BlueberryPicks where
  annie : ℕ
  kathryn : ℕ
  ben : ℕ

/-- Theorem stating the difference between Kathryn's and Annie's blueberry picks -/
theorem kathryn_annie_difference (picks : BlueberryPicks) : 
  picks.annie = 8 →
  picks.ben = picks.kathryn - 3 →
  picks.annie + picks.kathryn + picks.ben = 25 →
  picks.kathryn - picks.annie = 2 := by
sorry

end NUMINAMATH_CALUDE_kathryn_annie_difference_l211_21192


namespace NUMINAMATH_CALUDE_plan1_more_cost_effective_l211_21193

/-- Represents the cost of a mobile phone plan based on talk time -/
def plan_cost (rental : ℝ) (rate : ℝ) (minutes : ℝ) : ℝ := rental + rate * minutes

/-- Theorem stating when Plan 1 is more cost-effective than Plan 2 -/
theorem plan1_more_cost_effective (minutes : ℝ) :
  minutes > 72 →
  plan_cost 36 0.1 minutes < plan_cost 0 0.6 minutes := by
  sorry

end NUMINAMATH_CALUDE_plan1_more_cost_effective_l211_21193


namespace NUMINAMATH_CALUDE_divisible_by_24_l211_21122

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l211_21122


namespace NUMINAMATH_CALUDE_orange_bin_theorem_l211_21185

def final_orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : ℕ :=
  initial - thrown_away + added

theorem orange_bin_theorem (initial : ℕ) (thrown_away : ℕ) (added : ℕ) 
  (h1 : thrown_away ≤ initial) :
  final_orange_count initial thrown_away added = initial - thrown_away + added :=
by
  sorry

#eval final_orange_count 31 9 38

end NUMINAMATH_CALUDE_orange_bin_theorem_l211_21185


namespace NUMINAMATH_CALUDE_B_initial_investment_correct_l211_21170

/-- Represents the initial investment of B in rupees -/
def B_initial_investment : ℝ := 4866.67

/-- Represents A's initial investment in rupees -/
def A_initial_investment : ℝ := 2000

/-- Represents the amount A withdraws after 8 months in rupees -/
def A_withdrawal : ℝ := 1000

/-- Represents the amount B advances after 8 months in rupees -/
def B_advance : ℝ := 1000

/-- Represents the total profit at the end of the year in rupees -/
def total_profit : ℝ := 630

/-- Represents A's share of the profit in rupees -/
def A_profit_share : ℝ := 175

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of months before A withdraws and B advances -/
def months_before_change : ℕ := 8

theorem B_initial_investment_correct :
  B_initial_investment * months_before_change +
  (B_initial_investment + B_advance) * (months_in_year - months_before_change) =
  (total_profit - A_profit_share) / A_profit_share *
  (A_initial_investment * months_in_year) :=
by sorry

end NUMINAMATH_CALUDE_B_initial_investment_correct_l211_21170


namespace NUMINAMATH_CALUDE_raft_sticks_ratio_l211_21188

theorem raft_sticks_ratio :
  ∀ (simon_sticks gerry_sticks micky_sticks : ℕ),
    simon_sticks = 36 →
    micky_sticks = simon_sticks + gerry_sticks + 9 →
    simon_sticks + gerry_sticks + micky_sticks = 129 →
    gerry_sticks * 3 = simon_sticks * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_ratio_l211_21188


namespace NUMINAMATH_CALUDE_f_one_ge_six_l211_21178

/-- A quadratic function f(x) = x^2 + 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

/-- Theorem: If f(x) is increasing on (-1, +∞), then f(1) ≥ 6 -/
theorem f_one_ge_six (a : ℝ) 
  (h : ∀ x y, -1 < x ∧ x < y → f a x < f a y) : 
  f a 1 ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_f_one_ge_six_l211_21178


namespace NUMINAMATH_CALUDE_simplify_expression_l211_21132

variable (R : Type*) [Ring R]
variable (a b : R)

theorem simplify_expression : (a - b) - (a + b) = -2 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l211_21132


namespace NUMINAMATH_CALUDE_no_real_roots_third_polynomial_l211_21103

/-- Given two quadratic polynomials with integer roots, prove the third has no real roots -/
theorem no_real_roots_third_polynomial (a b : ℝ) :
  (∃ x : ℤ, x^2 + a*x + b = 0) →
  (∃ y : ℤ, y^2 + a*y + (b+1) = 0) →
  ¬∃ z : ℝ, z^2 + a*z + (b+2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_third_polynomial_l211_21103


namespace NUMINAMATH_CALUDE_two_a_plus_b_value_l211_21121

theorem two_a_plus_b_value (a b : ℚ) 
  (eq1 : 3 * a - b = 8) 
  (eq2 : 4 * b + 7 * a = 13) : 
  2 * a + b = 73 / 19 := by
sorry

end NUMINAMATH_CALUDE_two_a_plus_b_value_l211_21121


namespace NUMINAMATH_CALUDE_total_cats_l211_21127

theorem total_cats (white : ℕ) (black : ℕ) (gray : ℕ) 
  (h_white : white = 2) 
  (h_black : black = 10) 
  (h_gray : gray = 3) : 
  white + black + gray = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l211_21127


namespace NUMINAMATH_CALUDE_sum_two_angles_greater_90_implies_acute_l211_21164

-- Define a triangle type
structure Triangle where
  α : Real
  β : Real
  γ : Real
  angle_sum : α + β + γ = 180
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ

-- Define the property of sum of any two angles being greater than 90°
def sum_of_two_angles_greater_than_90 (t : Triangle) : Prop :=
  t.α + t.β > 90 ∧ t.α + t.γ > 90 ∧ t.β + t.γ > 90

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.α < 90 ∧ t.β < 90 ∧ t.γ < 90

-- Theorem statement
theorem sum_two_angles_greater_90_implies_acute (t : Triangle) :
  sum_of_two_angles_greater_than_90 t → is_acute_triangle t :=
by
  sorry


end NUMINAMATH_CALUDE_sum_two_angles_greater_90_implies_acute_l211_21164


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l211_21128

theorem sin_cos_sum_equals_half : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (70 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l211_21128


namespace NUMINAMATH_CALUDE_floor_sqrt_10_l211_21158

theorem floor_sqrt_10 : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_10_l211_21158


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l211_21117

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_11 :
  ∃ (p : ℕ),
    isFourDigit p ∧
    p % 63 = 0 ∧
    (reverseDigits p) % 63 = 0 ∧
    p % 11 = 0 ∧
    ∀ (q : ℕ),
      isFourDigit q ∧
      q % 63 = 0 ∧
      (reverseDigits q) % 63 = 0 ∧
      q % 11 = 0 →
      q ≤ p ∧
    p = 9779 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l211_21117


namespace NUMINAMATH_CALUDE_least_difference_nm_l211_21179

/-- Given a triangle ABC with sides AB = x+6, AC = 4x, BC = x+12, prove that the least possible 
    value of n-m is 2.5, where m and n are defined such that 1.5 < x < 4, m = 1.5, and n = 4. -/
theorem least_difference_nm (x : ℝ) (m n : ℝ) : 
  x > 0 ∧ 
  (x + 6) + 4*x > (x + 12) ∧
  (x + 6) + (x + 12) > 4*x ∧
  4*x + (x + 12) > (x + 6) ∧
  x + 12 > x + 6 ∧
  x + 12 > 4*x ∧
  m = 1.5 ∧
  n = 4 ∧
  1.5 < x ∧
  x < 4 →
  n - m = 2.5 := by
sorry

end NUMINAMATH_CALUDE_least_difference_nm_l211_21179
