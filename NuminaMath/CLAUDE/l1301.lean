import Mathlib

namespace smallest_possible_d_l1301_130164

theorem smallest_possible_d (c d : ℝ) : 
  (2 < c) → 
  (c < d) → 
  (2 + c ≤ d) → 
  (2/c + 2/d ≤ 2) → 
  d ≥ 2 + Real.sqrt 2 :=
by sorry

end smallest_possible_d_l1301_130164


namespace chess_game_probability_l1301_130159

theorem chess_game_probability (p_win p_not_lose : ℝ) :
  p_win = 0.3 → p_not_lose = 0.8 → p_win + (p_not_lose - p_win) = 0.5 := by
  sorry

end chess_game_probability_l1301_130159


namespace ball_travel_distance_l1301_130168

/-- Represents an elliptical billiard table -/
structure EllipticalTable where
  majorAxis : ℝ
  focalDistance : ℝ

/-- Possible distances traveled by a ball on an elliptical table -/
def possibleDistances (table : EllipticalTable) : Set ℝ :=
  {4, 3, 1}

/-- Theorem: The distance traveled by a ball on a specific elliptical table -/
theorem ball_travel_distance (table : EllipticalTable) 
  (h1 : table.majorAxis = 2)
  (h2 : table.focalDistance = 1) :
  ∃ d ∈ possibleDistances table, d = 4 ∨ d = 3 ∨ d = 1 :=
by sorry

end ball_travel_distance_l1301_130168


namespace complex_sum_problem_l1301_130195

theorem complex_sum_problem (a b c d e f g h : ℂ) :
  b = 2 ∧ 
  g = -a - c - e ∧ 
  a + b * Complex.I + c + d * Complex.I + e + f * Complex.I + g + h * Complex.I = -3 * Complex.I →
  d + f + h = -5 := by
  sorry

end complex_sum_problem_l1301_130195


namespace one_acute_triangle_in_1997_gon_l1301_130173

/-- A convex regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- A decomposition of a polygon into triangles using non-intersecting diagonals -/
structure TriangularDecomposition (n : ℕ) where
  (polygon : RegularPolygon n)

/-- An acute triangle -/
structure AcuteTriangle

/-- The number of acute triangles in a triangular decomposition of a regular polygon -/
def num_acute_triangles (n : ℕ) (decomp : TriangularDecomposition n) : ℕ :=
  sorry

/-- The main theorem: In a regular 1997-gon, there is exactly one acute triangle
    in its triangular decomposition -/
theorem one_acute_triangle_in_1997_gon :
  ∀ (decomp : TriangularDecomposition 1997),
    num_acute_triangles 1997 decomp = 1 :=
  sorry

end one_acute_triangle_in_1997_gon_l1301_130173


namespace stellas_antique_shop_profit_l1301_130124

/-- Calculates the profit for Stella's antique shop given the inventory and prices --/
theorem stellas_antique_shop_profit :
  let dolls : ℕ := 6
  let clocks : ℕ := 4
  let glasses : ℕ := 8
  let vases : ℕ := 3
  let postcards : ℕ := 10
  let doll_price : ℕ := 8
  let clock_price : ℕ := 25
  let glass_price : ℕ := 6
  let vase_price : ℕ := 12
  let postcard_price : ℕ := 3
  let purchase_cost : ℕ := 250
  let revenue := dolls * doll_price + clocks * clock_price + glasses * glass_price + 
                 vases * vase_price + postcards * postcard_price
  let profit := revenue - purchase_cost
  profit = 12 := by sorry

end stellas_antique_shop_profit_l1301_130124


namespace like_terms_imply_xy_value_l1301_130190

theorem like_terms_imply_xy_value (a b : ℝ) (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ 2 * a^x * b^3 = k * (-a^2 * b^(1-y))) →
  x * y = -4 :=
sorry

end like_terms_imply_xy_value_l1301_130190


namespace joan_next_birthday_age_l1301_130132

theorem joan_next_birthday_age
  (joan larry kim : ℝ)
  (h1 : joan = 1.3 * larry)
  (h2 : larry = 0.75 * kim)
  (h3 : joan + larry + kim = 39)
  : ⌊joan⌋ + 1 = 15 :=
sorry

end joan_next_birthday_age_l1301_130132


namespace problem_solution_l1301_130150

theorem problem_solution (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 23 := by
  sorry

end problem_solution_l1301_130150


namespace substitution_result_l1301_130117

theorem substitution_result (x y : ℝ) :
  (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) →
  4 * x + 10 * x - 5 = 7 := by
sorry

end substitution_result_l1301_130117


namespace mortgage_repayment_duration_l1301_130125

theorem mortgage_repayment_duration (a : ℝ) (r : ℝ) (S : ℝ) (h1 : a = 400) (h2 : r = 2) (h3 : S = 819200) :
  ∃ n : ℕ, n = 11 ∧ S = a * (1 - r^n) / (1 - r) ∧ ∀ m : ℕ, m < n → S > a * (1 - r^m) / (1 - r) :=
sorry

end mortgage_repayment_duration_l1301_130125


namespace inverse_prop_percent_change_l1301_130183

/-- Theorem: Inverse Proportionality and Percentage Change

Given:
- x and y are inversely proportional and positive
- x decreases by q%

Prove:
y increases by (100q)/(100-q)%
-/
theorem inverse_prop_percent_change (x y k q : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_q : 0 < q) (h_q_lt_100 : q < 100)
  (h_inverse_prop : x * y = k) :
  let x' := x * (1 - q / 100)
  let y' := k / x'
  (y' - y) / y * 100 = 100 * q / (100 - q) :=
by sorry

end inverse_prop_percent_change_l1301_130183


namespace inequality_proof_l1301_130136

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) (h4 : a + b + c = 3) : 
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 ∧ 
  (a * b^2 + b * c^2 + c * a^2 = 27/8 ↔ a = 3/2 ∧ b = 3/2 ∧ c = 0) :=
by sorry

end inequality_proof_l1301_130136


namespace ball_max_height_l1301_130178

/-- The path of a ball thrown on a planet with stronger gravity -/
def ballPath (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

/-- The maximum height reached by the ball -/
def maxHeight : ℝ := 140

theorem ball_max_height :
  ∃ t : ℝ, ballPath t = maxHeight ∧ ∀ s : ℝ, ballPath s ≤ maxHeight := by
  sorry

end ball_max_height_l1301_130178


namespace rectangle_area_l1301_130109

theorem rectangle_area (perimeter : ℝ) (width : ℝ) (length : ℝ) : 
  perimeter = 40 →
  length = 2 * width →
  width * length = 800 / 9 := by
  sorry

end rectangle_area_l1301_130109


namespace function_properties_l1301_130107

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x + b

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

theorem function_properties (a b : ℝ) :
  f a b 0 = 2 →
  f' a 1 = 0 →
  (∃ (x : ℝ), f 3 2 x = f a b x) ∧
  (∀ (x : ℝ), x < -3 → (f' 3 x > 0)) ∧
  (∀ (x : ℝ), -3 < x ∧ x < 1 → (f' 3 x < 0)) ∧
  (∀ (x : ℝ), x > 1 → (f' 3 x > 0)) :=
by sorry

end function_properties_l1301_130107


namespace direct_proportion_exponent_l1301_130115

theorem direct_proportion_exponent (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, -2 * x^(m-2) = k * x) → m = 3 := by
  sorry

end direct_proportion_exponent_l1301_130115


namespace apple_difference_l1301_130104

theorem apple_difference (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 46)
  (h2 : remaining_apples = 14) : 
  initial_apples - remaining_apples = 32 := by
sorry

end apple_difference_l1301_130104


namespace cube_sum_inequality_l1301_130126

theorem cube_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := by
  sorry

end cube_sum_inequality_l1301_130126


namespace red_on_third_prob_l1301_130147

/-- A fair 10-sided die with exactly 3 red sides -/
structure RedDie :=
  (sides : Nat)
  (red_sides : Nat)
  (h_sides : sides = 10)
  (h_red : red_sides = 3)

/-- The probability of rolling a specific outcome on the RedDie -/
def roll_prob (d : RedDie) (is_red : Bool) : ℚ :=
  if is_red then d.red_sides / d.sides else (d.sides - d.red_sides) / d.sides

/-- The probability of the die landing with a red side up for the first time on the third roll -/
def red_on_third (d : RedDie) : ℚ :=
  (roll_prob d false) * (roll_prob d false) * (roll_prob d true)

theorem red_on_third_prob (d : RedDie) : 
  red_on_third d = 147 / 1000 := by sorry

end red_on_third_prob_l1301_130147


namespace number_of_students_l1301_130141

theorem number_of_students (x : ℕ) 
  (h1 : 3600 = (3600 / x) * x)  -- Retail price for x tools
  (h2 : 3600 = (3600 / (x + 60)) * (x + 60))  -- Wholesale price for x + 60 tools
  (h3 : (3600 / x) * 50 = (3600 / (x + 60)) * 60)  -- Cost equality condition
  : x = 300 := by
  sorry

end number_of_students_l1301_130141


namespace parabola_properties_l1301_130189

/-- Parabola represented by y = -3x^2 - 6x + 2 -/
def parabola (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 2

theorem parabola_properties :
  (∃ (x_max y_max : ℝ),
    (∀ x, parabola x ≤ parabola x_max) ∧
    parabola x_max = y_max ∧
    x_max = 1 ∧
    y_max = -7) ∧
  (∀ x, parabola (2 - x) = parabola x) := by
  sorry

end parabola_properties_l1301_130189


namespace sine_tangent_inequality_l1301_130133

theorem sine_tangent_inequality (x y : ℝ) : 
  (0 < Real.sin (50 * π / 180) ∧ Real.sin (50 * π / 180) < 1) →
  Real.tan (50 * π / 180) > 1 →
  (Real.sin (50 * π / 180))^x - (Real.tan (50 * π / 180))^x ≤ 
  (Real.sin (50 * π / 180))^(-y) - (Real.tan (50 * π / 180))^(-y) →
  x + y ≥ 0 := by
  sorry

end sine_tangent_inequality_l1301_130133


namespace hcf_problem_l1301_130131

theorem hcf_problem (a b : ℕ) (h1 : a = 280) (h2 : Nat.lcm a b = Nat.gcd a b * 13 * 14) :
  Nat.gcd a b = 5 := by
  sorry

end hcf_problem_l1301_130131


namespace right_triangle_area_l1301_130144

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 18 := by
  sorry

end right_triangle_area_l1301_130144


namespace quadrilateral_area_sum_l1301_130167

/-- Represents a quadrilateral PQRS -/
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

/-- Checks if a quadrilateral is convex -/
def is_convex (quad : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def area (quad : Quadrilateral) : ℝ := sorry

/-- Checks if a number has no perfect square factors greater than 1 -/
def no_perfect_square_factors (n : ℝ) : Prop := sorry

theorem quadrilateral_area_sum (quad : Quadrilateral) (a b c : ℝ) :
  is_convex quad →
  distance quad.P quad.Q = 7 →
  distance quad.Q quad.R = 3 →
  distance quad.R quad.S = 9 →
  distance quad.S quad.P = 9 →
  angle quad.R quad.S quad.P = π / 3 →
  ∃ (a b c : ℝ), area quad = Real.sqrt a + b * Real.sqrt c ∧
                  no_perfect_square_factors a ∧
                  no_perfect_square_factors c →
  a + b + c = 608.25 := by
  sorry

end quadrilateral_area_sum_l1301_130167


namespace arithmetic_sequence_properties_l1301_130192

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- Theorem about a specific arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 1 + seq.a 3 = 16)
    (h2 : seq.S 4 = 28) :
  (∀ n : ℕ, seq.a n = 12 - 2 * n) ∧
  (∀ n : ℕ, n ≤ 5 → seq.S n ≤ seq.S 5) ∧
  seq.S 5 = 45 := by
  sorry

end arithmetic_sequence_properties_l1301_130192


namespace parking_theorem_l1301_130151

/-- The number of ways to park 5 trains on 5 tracks with one restriction -/
def parking_arrangements (n : ℕ) (restricted_train : ℕ) (restricted_track : ℕ) : ℕ :=
  (n - 1) * Nat.factorial (n - 1)

theorem parking_theorem :
  parking_arrangements 5 1 1 = 96 :=
by sorry

end parking_theorem_l1301_130151


namespace min_value_abc_l1301_130142

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (a₀ + 3 * b₀) * (b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 64 :=
by sorry

end min_value_abc_l1301_130142


namespace triangle_properties_l1301_130122

noncomputable section

-- Define the triangle
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Theorem statement
theorem triangle_properties {A B C a b c : ℝ} (h : triangle A B C a b c) :
  a = 2 * Real.sqrt 3 ∧ 
  Real.cos (2 * A + π / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := by
  sorry

end

end triangle_properties_l1301_130122


namespace quadratic_root_difference_l1301_130197

theorem quadratic_root_difference (r₁ r₂ : ℝ) : 
  2 * r₁^2 - 10 * r₁ + 2 = 0 ∧
  2 * r₂^2 - 10 * r₂ + 2 = 0 ∧
  r₁^2 + r₂^2 = 23 →
  |r₁ - r₂| = Real.sqrt 21 :=
by sorry

end quadratic_root_difference_l1301_130197


namespace equation_solution_l1301_130148

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : (7*x - 4) / (x - 2) = 5 / (x - 2)) : x = 9/7 := by
  sorry

end equation_solution_l1301_130148


namespace unit_digit_of_3_to_2022_l1301_130165

def unit_digit (n : ℕ) : ℕ := n % 10

def power_of_3_unit_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | 0 => 1
  | _ => 0  -- This case should never occur

theorem unit_digit_of_3_to_2022 :
  unit_digit (3^2022) = 9 :=
by
  sorry

end unit_digit_of_3_to_2022_l1301_130165


namespace courtyard_paving_l1301_130112

/-- Calculate the number of bricks required to pave a courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℝ) 
  (h1 : courtyard_length = 35)
  (h2 : courtyard_width = 24)
  (h3 : brick_length = 0.15)
  (h4 : brick_width = 0.08) :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 70000 := by
  sorry

#eval (35 * 24) / (0.15 * 0.08)

end courtyard_paving_l1301_130112


namespace f_one_is_zero_five_zeros_symmetric_center_l1301_130160

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x - 1) = f (x + 1)
axiom decreasing_property : ∀ x₁ x₂, x₁ ∈ Set.Ioo 0 1 → x₂ ∈ Set.Ioo 0 1 → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

-- Theorem statements
theorem f_one_is_zero : f 1 = 0 := by sorry

theorem five_zeros : 
  f (-2) = 0 ∧ f (-1) = 0 ∧ f 0 = 0 ∧ f 1 = 0 ∧ f 2 = 0 := by sorry

theorem symmetric_center : 
  ∀ x, f (2014 + x) = -f (2014 - x) := by sorry

end f_one_is_zero_five_zeros_symmetric_center_l1301_130160


namespace volume_conversion_l1301_130154

/-- Proves that a volume of 108 cubic feet is equal to 4 cubic yards -/
theorem volume_conversion (box_volume_cubic_feet : ℝ) (cubic_feet_per_cubic_yard : ℝ) :
  box_volume_cubic_feet = 108 →
  cubic_feet_per_cubic_yard = 27 →
  box_volume_cubic_feet / cubic_feet_per_cubic_yard = 4 := by
sorry

end volume_conversion_l1301_130154


namespace sum_of_coefficients_is_64_l1301_130103

/-- The sum of the coefficients of the terms in the expansion of (x+y+3)^3 that do not contain y -/
def sum_of_coefficients (x y : ℝ) : ℝ := (x + 3)^3

/-- Theorem: The sum of the coefficients of the terms in the expansion of (x+y+3)^3 that do not contain y is 64 -/
theorem sum_of_coefficients_is_64 :
  ∀ x y : ℝ, sum_of_coefficients x y = 64 := by
sorry

end sum_of_coefficients_is_64_l1301_130103


namespace flight_cost_proof_l1301_130176

theorem flight_cost_proof (initial_cost : ℝ) : 
  (∃ (cost_per_person_4 cost_per_person_5 : ℝ),
    cost_per_person_4 = initial_cost / 4 ∧
    cost_per_person_5 = initial_cost / 5 ∧
    cost_per_person_4 - cost_per_person_5 = 30) →
  initial_cost = 600 := by
sorry

end flight_cost_proof_l1301_130176


namespace repeating_decimal_to_fraction_l1301_130194

theorem repeating_decimal_to_fraction : 
  ∃ (n : ℚ), n = 7 + 123 / 999 ∧ n = 593 / 111 := by
  sorry

end repeating_decimal_to_fraction_l1301_130194


namespace sin_alpha_value_l1301_130177

theorem sin_alpha_value (m : ℝ) (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (m, -3) ∧ P.1 * Real.cos α = P.2 * Real.sin α) →
  Real.tan α = -3/4 →
  Real.sin α = -3/5 := by
sorry

end sin_alpha_value_l1301_130177


namespace mary_warmth_hours_l1301_130161

/-- Represents the number of sticks of wood produced by different furniture types -/
structure FurnitureWood where
  chair : Nat
  table : Nat
  cabinet : Nat
  stool : Nat

/-- Represents the quantity of each furniture type Mary chops -/
structure ChoppedFurniture where
  chairs : Nat
  tables : Nat
  cabinets : Nat
  stools : Nat

/-- Calculates the total number of sticks of wood produced -/
def totalWood (fw : FurnitureWood) (cf : ChoppedFurniture) : Nat :=
  fw.chair * cf.chairs + fw.table * cf.tables + fw.cabinet * cf.cabinets + fw.stool * cf.stools

/-- Theorem stating that Mary can keep warm for 64 hours with the chopped firewood -/
theorem mary_warmth_hours (fw : FurnitureWood) (cf : ChoppedFurniture) (sticksPerHour : Nat) :
  fw.chair = 8 →
  fw.table = 12 →
  fw.cabinet = 16 →
  fw.stool = 3 →
  cf.chairs = 25 →
  cf.tables = 12 →
  cf.cabinets = 5 →
  cf.stools = 8 →
  sticksPerHour = 7 →
  totalWood fw cf / sticksPerHour = 64 := by
  sorry

#check mary_warmth_hours

end mary_warmth_hours_l1301_130161


namespace family_income_problem_l1301_130145

theorem family_income_problem (initial_members : ℕ) (deceased_income new_average : ℚ) 
  (h1 : initial_members = 4)
  (h2 : deceased_income = 1170)
  (h3 : new_average = 590) :
  let initial_average := (initial_members * new_average + deceased_income) / initial_members
  initial_average = 735 := by
sorry

end family_income_problem_l1301_130145


namespace exists_sum_of_digits_div_11_l1301_130169

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there exists one whose sum of digits is divisible by 11 -/
theorem exists_sum_of_digits_div_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (n + k) % 11 = 0) := by sorry

end exists_sum_of_digits_div_11_l1301_130169


namespace area_code_count_l1301_130111

/-- The number of uppercase letters available -/
def uppercaseLetters : Nat := 26

/-- The number of lowercase letters and digits available for the second character -/
def secondCharOptions : Nat := 36

/-- The number of special characters available -/
def specialChars : Nat := 10

/-- The number of digits available -/
def digits : Nat := 10

/-- The total number of unique area codes that can be created -/
def totalAreaCodes : Nat := 
  (uppercaseLetters * secondCharOptions) + 
  (uppercaseLetters * secondCharOptions * specialChars) + 
  (uppercaseLetters * secondCharOptions * specialChars * digits)

theorem area_code_count : totalAreaCodes = 103896 := by
  sorry

end area_code_count_l1301_130111


namespace cos_2alpha_value_l1301_130146

theorem cos_2alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (α - π/4))
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.cos (2 * α) = Real.sqrt 15 / 8 := by
  sorry

end cos_2alpha_value_l1301_130146


namespace amoli_driving_time_l1301_130198

-- Define the constants from the problem
def total_distance : ℝ := 369
def amoli_speed : ℝ := 42
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2
def remaining_distance : ℝ := 121

-- Define Amoli's driving time as a variable
def amoli_time : ℝ := 3

-- Theorem statement
theorem amoli_driving_time :
  amoli_speed * amoli_time + anayet_speed * anayet_time = total_distance - remaining_distance :=
by sorry

end amoli_driving_time_l1301_130198


namespace parabola_midpoint_to_directrix_l1301_130120

/-- A parabola with equation y^2 = 4x and two points on it -/
structure Parabola where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁^2 = 4 * x₁
  eq₂ : y₂^2 = 4 * x₂
  dist : (x₁ - x₂)^2 + (y₁ - y₂)^2 = 49  -- |AB|^2 = 7^2

/-- The distance from the midpoint of AB to the directrix of the parabola is 7/2 -/
theorem parabola_midpoint_to_directrix (p : Parabola) : 
  (p.x₁ + p.x₂) / 2 + 1 = 7/2 := by
  sorry

end parabola_midpoint_to_directrix_l1301_130120


namespace trailingZeros_100_factorial_l1301_130123

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailingZeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end trailingZeros_100_factorial_l1301_130123


namespace min_value_quadratic_sum_l1301_130139

theorem min_value_quadratic_sum (x y s : ℝ) (h : x + y = s) :
  (∀ a b : ℝ, a + b = s → 3 * a^2 + 2 * b^2 ≥ (6/5) * s^2) ∧
  ∃ x₀ y₀ : ℝ, x₀ + y₀ = s ∧ 3 * x₀^2 + 2 * y₀^2 = (6/5) * s^2 :=
by sorry

end min_value_quadratic_sum_l1301_130139


namespace max_subsets_100_l1301_130130

/-- Given a set S of n elements, f(n) returns the maximum number of non-empty subsets
    that can be chosen from S such that any two chosen subsets are either disjoint
    or one contains the other. -/
def max_subsets (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem stating that for a set of 100 elements, the maximum number of non-empty subsets
    that can be chosen, such that any two chosen subsets are either disjoint or one contains
    the other, is 199. -/
theorem max_subsets_100 : max_subsets 100 = 199 := by
  sorry

end max_subsets_100_l1301_130130


namespace evaluate_polynomial_l1301_130171

theorem evaluate_polynomial : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 := by
  sorry

end evaluate_polynomial_l1301_130171


namespace volleyball_lineup_count_is_151200_l1301_130193

/-- The number of ways to choose 6 players from a team of 10 players for 6 distinct positions -/
def volleyball_lineup_count : ℕ := 10 * 9 * 8 * 7 * 6 * 5

/-- Theorem stating that the number of ways to choose a volleyball lineup is 151,200 -/
theorem volleyball_lineup_count_is_151200 : volleyball_lineup_count = 151200 := by
  sorry

end volleyball_lineup_count_is_151200_l1301_130193


namespace sum_possible_angles_l1301_130179

/-- An isosceles triangle with one angle of 80 degrees -/
structure IsoscelesTriangle80 where
  /-- The measure of one of the angles in degrees -/
  angle1 : ℝ
  /-- The measure of another angle in degrees -/
  angle2 : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True
  /-- One of the angles is 80 degrees -/
  has80Angle : angle1 = 80 ∨ angle2 = 80
  /-- The sum of all angles in a triangle is 180 degrees -/
  angleSum : angle1 + angle2 + (180 - angle1 - angle2) = 180

/-- The theorem to be proved -/
theorem sum_possible_angles (t : IsoscelesTriangle80) :
  ∃ (y1 y2 y3 : ℝ), (y1 = t.angle1 ∧ y1 ≠ 80) ∨ 
                    (y1 = t.angle2 ∧ y1 ≠ 80) ∨
                    (y1 = 180 - t.angle1 - t.angle2 ∧ y1 ≠ 80) ∧
                    (y2 = t.angle1 ∧ y2 ≠ 80) ∨ 
                    (y2 = t.angle2 ∧ y2 ≠ 80) ∨
                    (y2 = 180 - t.angle1 - t.angle2 ∧ y2 ≠ 80) ∧
                    (y3 = t.angle1 ∧ y3 ≠ 80) ∨ 
                    (y3 = t.angle2 ∧ y3 ≠ 80) ∨
                    (y3 = 180 - t.angle1 - t.angle2 ∧ y3 ≠ 80) ∧
                    y1 + y2 + y3 = 150 :=
  sorry

end sum_possible_angles_l1301_130179


namespace min_trees_for_three_types_l1301_130129

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove where
  trees : Finset TreeType
  total_count : ℕ
  four_types_in_85 : ∀ (subset : Finset TreeType), subset.card = 85 → (∀ t : TreeType, t ∈ subset)

/-- The theorem to be proved -/
theorem min_trees_for_three_types (g : Grove) (h1 : g.total_count = 100) :
  ∃ (n : ℕ), n = 69 ∧ 
  (∀ (subset : Finset TreeType), subset.card ≥ n → 
    ∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset) ∧
  (∃ (subset : Finset TreeType), subset.card = n - 1 ∧ 
    ∀ (t1 t2 t3 : TreeType), t1 ≠ t2 → t1 ≠ t3 → t2 ≠ t3 → 
      ¬(t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset)) :=
by sorry

end min_trees_for_three_types_l1301_130129


namespace greatest_color_count_l1301_130116

theorem greatest_color_count (α β : ℝ) (h1 : 1 < α) (h2 : α < β) : 
  (∀ (r : ℕ), r > 2 → 
    ∃ (f : ℕ+ → Fin r), ∀ (x y : ℕ+), 
      f x = f y → (α : ℝ) ≤ (x : ℝ) / (y : ℝ) → (x : ℝ) / (y : ℝ) ≤ β → False) ∧
  (∀ (f : ℕ+ → Fin 2), ∃ (x y : ℕ+), 
    f x = f y ∧ (α : ℝ) ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β) :=
by sorry

end greatest_color_count_l1301_130116


namespace pool_width_l1301_130152

theorem pool_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 30 ∧ length = 10 ∧ area = length * width → width = 3 := by
  sorry

end pool_width_l1301_130152


namespace largest_divisor_with_remainders_l1301_130162

theorem largest_divisor_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℤ), 69 = k * n + 5) ∧ 
  (∃ (l : ℤ), 86 = l * n + 6) ∧ 
  (∀ (m : ℕ), m > n → 
    (¬∃ (k : ℤ), 69 = k * m + 5) ∨ 
    (¬∃ (l : ℤ), 86 = l * m + 6)) :=
by sorry

end largest_divisor_with_remainders_l1301_130162


namespace trace_of_matrix_minus_inverse_zero_l1301_130113

/-- Given a 2x2 matrix A with real entries a, 2, -3, and d,
    if A - A^(-1) is the zero matrix, then the trace of A is a + d. -/
theorem trace_of_matrix_minus_inverse_zero (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  (A - A⁻¹ = 0) → Matrix.trace A = a + d := by
  sorry

end trace_of_matrix_minus_inverse_zero_l1301_130113


namespace complete_square_integer_l1301_130110

theorem complete_square_integer (y : ℝ) : ∃ k : ℤ, y^2 + 12*y + 40 = (y + 6)^2 + k := by
  sorry

end complete_square_integer_l1301_130110


namespace geometric_arithmetic_progression_sum_l1301_130118

theorem geometric_arithmetic_progression_sum : ∃ x y : ℝ, 
  (5 < x ∧ x < y ∧ y < 12) ∧ 
  (∃ r : ℝ, r > 1 ∧ x = 5 * r ∧ y = 5 * r^2) ∧
  (∃ d : ℝ, d > 0 ∧ y = x + d ∧ 12 = y + d) ∧
  (abs (x + y - 16.2788) < 0.0001) := by
sorry

end geometric_arithmetic_progression_sum_l1301_130118


namespace geometric_sequence_sum_l1301_130127

theorem geometric_sequence_sum (a r : ℝ) (h1 : a + a*r + a*r^2 = 13) (h2 : a * (1 - r^7) / (1 - r) = 183) : 
  ∃ (ε : ℝ), abs (a + a*r + a*r^2 + a*r^3 + a*r^4 - 75.764) < ε ∧ ε > 0 := by
  sorry

end geometric_sequence_sum_l1301_130127


namespace mikes_video_games_l1301_130108

theorem mikes_video_games (non_working : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) : 
  non_working = 9 → price_per_game = 5 → total_earnings = 30 →
  non_working + (total_earnings / price_per_game) = 15 :=
by sorry

end mikes_video_games_l1301_130108


namespace pretzel_ratio_is_three_to_one_l1301_130101

/-- The number of pretzels Barry bought -/
def barry_pretzels : ℕ := 12

/-- The number of pretzels Angie bought -/
def angie_pretzels : ℕ := 18

/-- The number of pretzels Shelly bought -/
def shelly_pretzels : ℕ := barry_pretzels / 2

/-- The ratio of pretzels Angie bought to pretzels Shelly bought -/
def pretzel_ratio : ℚ := angie_pretzels / shelly_pretzels

theorem pretzel_ratio_is_three_to_one :
  pretzel_ratio = 3 := by sorry

end pretzel_ratio_is_three_to_one_l1301_130101


namespace min_value_theorem_achievable_lower_bound_l1301_130181

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + 2*n = 1) :
  (m + 1) * (n + 1) / (m * n) ≥ 8 + 4 * Real.sqrt 3 :=
by sorry

theorem achievable_lower_bound :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m + 2*n = 1 ∧ (m + 1) * (n + 1) / (m * n) = 8 + 4 * Real.sqrt 3 :=
by sorry

end min_value_theorem_achievable_lower_bound_l1301_130181


namespace arithmetic_sequence_length_l1301_130180

theorem arithmetic_sequence_length (a₁ aₙ d : ℕ) (h : a₁ = 6) (h' : aₙ = 206) (h'' : d = 4) :
  (aₙ - a₁) / d + 1 = 51 := by
  sorry

end arithmetic_sequence_length_l1301_130180


namespace absolute_value_inequality_l1301_130156

theorem absolute_value_inequality (x : ℝ) : 
  (abs x + abs (abs x - 1) = 1) → (x + 1) * (x - 1) ≤ 0 := by sorry

end absolute_value_inequality_l1301_130156


namespace john_quilt_cost_l1301_130134

/-- Calculates the cost of a rectangular quilt given its dimensions and cost per square foot -/
def quiltCost (length width costPerSqFt : ℝ) : ℝ :=
  length * width * costPerSqFt

/-- Proves that a 7ft by 8ft quilt at $40 per square foot costs $2240 -/
theorem john_quilt_cost :
  quiltCost 7 8 40 = 2240 := by
  sorry

end john_quilt_cost_l1301_130134


namespace bus_cyclist_speeds_l1301_130153

/-- The speed of the buses in km/h -/
def bus_speed : ℝ := 42

/-- The speed of the cyclist in km/h -/
def cyclist_speed : ℝ := 18

/-- The distance between points A and B in km -/
def distance : ℝ := 37

/-- The time in minutes from the start of the first bus to meeting the cyclist -/
def time_bus1_to_meeting : ℝ := 40

/-- The time in minutes from the start of the second bus to meeting the cyclist -/
def time_bus2_to_meeting : ℝ := 31

/-- The time in minutes from the start of the cyclist to meeting the first bus -/
def time_cyclist_to_bus1 : ℝ := 30

/-- The time in minutes from the start of the cyclist to meeting the second bus -/
def time_cyclist_to_bus2 : ℝ := 51

theorem bus_cyclist_speeds : 
  bus_speed * (time_bus1_to_meeting / 60) + cyclist_speed * (time_cyclist_to_bus1 / 60) = distance ∧
  bus_speed * (time_bus2_to_meeting / 60) + cyclist_speed * (time_cyclist_to_bus2 / 60) = distance :=
by sorry

end bus_cyclist_speeds_l1301_130153


namespace congruence_solution_l1301_130114

theorem congruence_solution (x : ℤ) : 
  (∃ (a m : ℤ), m ≥ 2 ∧ 0 ≤ a ∧ a < m ∧ x ≡ a [ZMOD m]) →
  ((10 * x + 3) ≡ 6 [ZMOD 15] ↔ x ≡ 0 [ZMOD 3]) := by
  sorry

end congruence_solution_l1301_130114


namespace x_to_y_ratio_l1301_130182

theorem x_to_y_ratio (x y : ℝ) (h : 3 * x = 0.12 * 250 * y) : x / y = 10 := by
  sorry

end x_to_y_ratio_l1301_130182


namespace two_digit_times_eleven_l1301_130196

theorem two_digit_times_eleven (A B : ℕ) (h : A + B ≥ 10) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * (A + B - 10) + B := by
  sorry

end two_digit_times_eleven_l1301_130196


namespace base_b_not_perfect_square_l1301_130155

theorem base_b_not_perfect_square (b : ℕ) (h : b ≥ 3) :
  ¬∃ (n : ℕ), 2 * b^2 + 2 * b + 1 = n^2 := by
  sorry

end base_b_not_perfect_square_l1301_130155


namespace trapezium_area_l1301_130106

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 10) (hh : h = 10) :
  (a + b) * h / 2 = 150 :=
by sorry

end trapezium_area_l1301_130106


namespace regression_lines_intersect_at_average_point_l1301_130149

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The average point of a dataset -/
structure AveragePoint where
  x : ℝ
  y : ℝ

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect_at_average_point 
  (l₁ l₂ : RegressionLine) 
  (avg : AveragePoint) : 
  (avg.x * l₁.slope + l₁.intercept = avg.y) ∧ 
  (avg.x * l₂.slope + l₂.intercept = avg.y) := by
  sorry

#check regression_lines_intersect_at_average_point

end regression_lines_intersect_at_average_point_l1301_130149


namespace salt_solution_problem_l1301_130187

theorem salt_solution_problem (initial_weight : ℝ) (added_salt : ℝ) (final_percentage : ℝ) :
  initial_weight = 60 →
  added_salt = 3 →
  final_percentage = 25 →
  let final_weight := initial_weight + added_salt
  let final_salt := (final_percentage / 100) * final_weight
  let initial_salt := final_salt - added_salt
  initial_salt / initial_weight * 100 = 21.25 := by
sorry

end salt_solution_problem_l1301_130187


namespace fraction_calculation_l1301_130137

theorem fraction_calculation (w x y : ℝ) 
  (h1 : w / y = 1 / 5)
  (h2 : (x + y) / y = 2.2) :
  w / x = 6 / 25 := by
  sorry

end fraction_calculation_l1301_130137


namespace complex_multiplication_l1301_130186

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l1301_130186


namespace arccos_gt_arctan_iff_l1301_130143

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1) (Real.sqrt 2 / 2) ∧ x ≠ Real.sqrt 2 / 2 :=
sorry

end arccos_gt_arctan_iff_l1301_130143


namespace stadium_length_conversion_l1301_130185

/-- Converts yards to feet given a conversion factor -/
def yards_to_feet (yards : ℕ) (conversion_factor : ℕ) : ℕ :=
  yards * conversion_factor

/-- Theorem: The stadium length of 62 yards is equal to 186 feet -/
theorem stadium_length_conversion :
  yards_to_feet 62 3 = 186 := by
  sorry

end stadium_length_conversion_l1301_130185


namespace area_of_region_s_l1301_130100

/-- A square with side length 4 -/
structure Square :=
  (side_length : ℝ)
  (is_four : side_length = 4)

/-- A region S in the square -/
structure Region (sq : Square) :=
  (area : ℝ)
  (in_square : area ≤ sq.side_length^2)
  (closer_to_vertex : area > 0)

/-- Theorem: The area of region S is 2 -/
theorem area_of_region_s (sq : Square) (S : Region sq) : S.area = 2 :=
sorry

end area_of_region_s_l1301_130100


namespace inequality_proof_l1301_130121

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + y^2)^2 ≥ (x+y+z)*(x-y+z)*(x+y-z)*(y+z-x) ∧
  ((x^2 + y^2)^2 = (x+y+z)*(x-y+z)*(x+y-z)*(y+z-x) ↔ x = y ∧ z = x*Real.sqrt 2) :=
sorry

end inequality_proof_l1301_130121


namespace felix_weight_lifting_l1301_130105

/-- Felix's weight lifting problem -/
theorem felix_weight_lifting (felix_weight : ℝ) (felix_lift : ℝ) (brother_lift : ℝ)
  (h1 : felix_lift = 150)
  (h2 : brother_lift = 600)
  (h3 : brother_lift = 3 * (2 * felix_weight)) :
  felix_lift / felix_weight = 1.5 := by
  sorry

end felix_weight_lifting_l1301_130105


namespace cookies_given_to_friend_l1301_130170

theorem cookies_given_to_friend (total : ℕ) (given_to_friend : ℕ) (given_to_family : ℕ) (eaten : ℕ) (left : ℕ) :
  total = 19 →
  given_to_family = (total - given_to_friend) / 2 →
  eaten = 2 →
  left = 5 →
  left = total - given_to_friend - given_to_family - eaten →
  given_to_friend = 5 := by
  sorry

end cookies_given_to_friend_l1301_130170


namespace function_properties_l1301_130128

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function -/
def Domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≠ 0

/-- The quadratic function we're considering -/
def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + n * x + 3 * m + n

/-- The theorem stating the properties of the function and its maximum value -/
theorem function_properties :
  ∃ (m n : ℝ),
    EvenFunction (f m n) ∧
    Domain (f m n) (m - 1) (2 * m) ∧
    m = 1/3 ∧
    n = 0 ∧
    (∀ x, m - 1 ≤ x ∧ x ≤ 2 * m → f m n x ≤ 31/27) :=
sorry

end function_properties_l1301_130128


namespace mika_birthday_stickers_l1301_130172

/-- The number of stickers Mika received for her birthday -/
def birthday_stickers (initial : ℕ) (bought : ℕ) (given_away : ℕ) (used : ℕ) (left : ℕ) : ℕ :=
  (left + given_away + used) - (initial + bought)

/-- Theorem stating that Mika received 20 stickers for her birthday -/
theorem mika_birthday_stickers : 
  birthday_stickers 20 26 6 58 2 = 20 := by
  sorry

end mika_birthday_stickers_l1301_130172


namespace triangle_properties_l1301_130191

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Define the triangle
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  -- Given conditions
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A →
  b + c = Real.sqrt 10 →
  a = 2 →
  -- Prove
  Real.cos A = 1/2 ∧
  (1/2 : ℝ) * b * c * Real.sin A = (7 * Real.sqrt 3) / 6 :=
by sorry

end triangle_properties_l1301_130191


namespace problem_1_problem_2_l1301_130199

-- Problem 1
theorem problem_1 : -1^2009 + Real.rpow 27 (1/3) - |1 - Real.sqrt 2| + Real.sqrt 8 = 3 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  y / x + x / y + 2 = 8 := by sorry

end problem_1_problem_2_l1301_130199


namespace only_negative_one_squared_is_negative_l1301_130157

theorem only_negative_one_squared_is_negative : 
  ((-1 : ℝ)^0 < 0 ∨ |(-1 : ℝ)| < 0 ∨ Real.sqrt 1 < 0 ∨ -(1 : ℝ)^2 < 0) ∧
  ((-1 : ℝ)^0 ≥ 0 ∧ |(-1 : ℝ)| ≥ 0 ∧ Real.sqrt 1 ≥ 0) ∧
  (-(1 : ℝ)^2 < 0) :=
by sorry

end only_negative_one_squared_is_negative_l1301_130157


namespace imaginary_part_of_complex_number_l1301_130184

theorem imaginary_part_of_complex_number (z : ℂ) (h : z = -1 + Complex.I) :
  z.im = 1 := by
  sorry

end imaginary_part_of_complex_number_l1301_130184


namespace profit_increase_may_to_june_l1301_130138

theorem profit_increase_may_to_june
  (march_to_april : Real)
  (april_to_may : Real)
  (march_to_june : Real)
  (h1 : march_to_april = 0.30)
  (h2 : april_to_may = -0.20)
  (h3 : march_to_june = 0.5600000000000001)
  : ∃ may_to_june : Real,
    (1 + march_to_april) * (1 + april_to_may) * (1 + may_to_june) = 1 + march_to_june ∧
    may_to_june = 0.50 := by
  sorry

end profit_increase_may_to_june_l1301_130138


namespace range_of_a_l1301_130188

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The theorem stating the range of a --/
theorem range_of_a (a : ℝ) : 
  (a > 0 ∧ 
   ∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, g a x₁ = f x₀) → 
  (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by sorry

end range_of_a_l1301_130188


namespace shares_distribution_l1301_130163

/-- Proves that given the conditions, the shares of A, B, C, D, and E are 50, 100, 300, 150, and 600 respectively. -/
theorem shares_distribution (total : ℝ) (a b c d e : ℝ) 
  (h_total : total = 1200)
  (h_ab : a = (1/2) * b)
  (h_bc : b = (1/3) * c)
  (h_cd : c = 2 * d)
  (h_de : d = (1/4) * e)
  (h_sum : a + b + c + d + e = total) :
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600 := by
  sorry

#check shares_distribution

end shares_distribution_l1301_130163


namespace highest_power_of_three_N_l1301_130135

/-- Concatenates a list of integers into a single integer -/
def concatenate_integers (list : List Int) : Int :=
  sorry

/-- Generates a list of 2-digit integers from 73 to 29 in descending order -/
def generate_list : List Int :=
  sorry

/-- The number N formed by concatenating 2-digit integers from 73 to 29 in descending order -/
def N : Int := concatenate_integers generate_list

/-- The highest power of 3 that divides a given integer -/
def highest_power_of_three (n : Int) : Int :=
  sorry

theorem highest_power_of_three_N :
  highest_power_of_three N = 0 := by
  sorry

end highest_power_of_three_N_l1301_130135


namespace area_triangle_abc_is_ten_l1301_130158

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Area of triangle ABC is 10 -/
theorem area_triangle_abc_is_ten
  (M₁ : Parabola)
  (M₂ : Parabola)
  (A : Point)
  (B : Point)
  (C : Point)
  (h₁ : M₂.b = -2 * M₂.a) -- M₂ is a horizontal translation of M₁
  (h₂ : A.y = M₂.a * A.x^2 + M₂.b * A.x + M₂.c) -- A is on M₂
  (h₃ : B.x = C.x) -- B and C are on the axis of symmetry of M₂
  (h₄ : C.x = 2 ∧ C.y = M₁.c - 5) -- Coordinates of C
  (h₅ : B.y = M₁.a * B.x^2 + M₁.c) -- B is on M₁
  (h₆ : C.y = M₂.a * C.x^2 + M₂.b * C.x + M₂.c) -- C is on M₂
  : (1/2 : ℝ) * |C.x - A.x| * |C.y - B.y| = 10 := by
  sorry


end area_triangle_abc_is_ten_l1301_130158


namespace garden_length_difference_l1301_130175

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular garden -/
def GardenProperties (garden : RectangularGarden) : Prop :=
  garden.length > 3 * garden.width ∧
  2 * (garden.length + garden.width) = 100 ∧
  garden.length = 38

theorem garden_length_difference (garden : RectangularGarden) 
  (h : GardenProperties garden) : 
  garden.length - 3 * garden.width = 2 := by
  sorry

end garden_length_difference_l1301_130175


namespace unique_function_satisfying_equation_l1301_130102

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y) ∧ f = fun x ↦ x - 1 := by
  sorry

end unique_function_satisfying_equation_l1301_130102


namespace sequence_property_l1301_130174

theorem sequence_property (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
sorry

end sequence_property_l1301_130174


namespace smallest_angle_solution_l1301_130140

theorem smallest_angle_solution (x : ℝ) : 
  (∀ y ∈ {y : ℝ | 0 < y ∧ y < x}, ¬(Real.sin (2*y) * Real.sin (3*y) = Real.cos (2*y) * Real.cos (3*y))) ∧
  (Real.sin (2*x) * Real.sin (3*x) = Real.cos (2*x) * Real.cos (3*x)) ∧
  (x * (180 / Real.pi) = 18) := by
  sorry

end smallest_angle_solution_l1301_130140


namespace largest_of_three_negatives_l1301_130166

theorem largest_of_three_negatives (a b c : ℝ) 
  (neg_a : a < 0) (neg_b : b < 0) (neg_c : c < 0)
  (h : c / (a + b) < a / (b + c) ∧ a / (b + c) < b / (c + a)) :
  c > a ∧ c > b := by
  sorry

end largest_of_three_negatives_l1301_130166


namespace smallest_y_value_l1301_130119

theorem smallest_y_value : ∃ y : ℝ, 
  (∀ z : ℝ, 3 * z^2 + 21 * z + 18 = z * (2 * z + 12) → y ≤ z) ∧
  (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧
  y = -6 := by
  sorry

end smallest_y_value_l1301_130119
