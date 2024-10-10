import Mathlib

namespace exists_monochromatic_equilateral_triangle_l2312_231274

-- Define a color type
inductive Color
| White
| Black

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  is_equilateral : sorry

-- Theorem statement
theorem exists_monochromatic_equilateral_triangle :
  ∃ (t : EquilateralTriangle), 
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end exists_monochromatic_equilateral_triangle_l2312_231274


namespace midpoint_of_fractions_l2312_231292

theorem midpoint_of_fractions : 
  let f1 : ℚ := 3/4
  let f2 : ℚ := 5/6
  let midpoint : ℚ := (f1 + f2) / 2
  midpoint = 19/24 := by sorry

end midpoint_of_fractions_l2312_231292


namespace sqrt_equality_condition_l2312_231268

theorem sqrt_equality_condition (a b c : ℝ) :
  Real.sqrt (4 * a^2 + 9 * b^2) = 2 * a + 3 * b + c ↔
  12 * a * b + 4 * a * c + 6 * b * c + c^2 = 0 ∧ 2 * a + 3 * b + c ≥ 0 :=
by sorry

end sqrt_equality_condition_l2312_231268


namespace scientific_notation_proof_l2312_231229

-- Define the original number
def original_number : ℝ := 0.0000084

-- Define the scientific notation components
def significand : ℝ := 8.4
def exponent : ℤ := -6

-- Theorem statement
theorem scientific_notation_proof :
  original_number = significand * (10 : ℝ) ^ exponent :=
by sorry

end scientific_notation_proof_l2312_231229


namespace only_parallel_corresponding_angles_has_converse_l2312_231225

-- Define the basic geometric concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define the geometric relations
def vertical_angles (a b : Angle) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def corresponding_angles (a b : Angle) (l1 l2 : Line) : Prop := sorry
def congruent_triangles (t1 t2 : Triangle) : Prop := sorry
def right_angle (a : Angle) : Prop := sorry

-- Define the theorems
def vertical_angles_theorem (a b : Angle) : 
  vertical_angles a b → a = b := sorry

def parallel_corresponding_angles_theorem (l1 l2 : Line) (a b : Angle) :
  parallel_lines l1 l2 → corresponding_angles a b l1 l2 → a = b := sorry

def congruent_triangles_angles_theorem (t1 t2 : Triangle) (a1 a2 : Angle) :
  congruent_triangles t1 t2 → corresponding_angles a1 a2 t1 t2 → a1 = a2 := sorry

def right_angles_equal_theorem (a b : Angle) :
  right_angle a → right_angle b → a = b := sorry

-- The main theorem to prove
theorem only_parallel_corresponding_angles_has_converse :
  ∃ (l1 l2 : Line) (a b : Angle),
    (corresponding_angles a b l1 l2 ∧ a = b → parallel_lines l1 l2) ∧
    (¬∃ (a b : Angle), a = b → vertical_angles a b) ∧
    (¬∃ (t1 t2 : Triangle) (a1 a2 a3 b1 b2 b3 : Angle),
      a1 = b1 ∧ a2 = b2 ∧ a3 = b3 → congruent_triangles t1 t2) ∧
    (¬∃ (a b : Angle), a = b → right_angle a ∧ right_angle b) := by
  sorry

end only_parallel_corresponding_angles_has_converse_l2312_231225


namespace circle_ring_area_floor_l2312_231252

theorem circle_ring_area_floor :
  let r : ℝ := 30 / 3 -- radius of small circles
  let R : ℝ := 30 -- radius of large circle C
  let K : ℝ := 3 * Real.pi * r^2 -- area between large circle and six small circles
  ⌊K⌋ = 942 := by
  sorry

end circle_ring_area_floor_l2312_231252


namespace trapezoid_diagonal_segments_l2312_231273

/-- 
Theorem: In a trapezoid with bases a and c, and sides b and d, 
the segments AO and OC of diagonal AC divided by diagonal BD are:
AO = (c / (a + c)) * √(ac + (ad² - cb²) / (a - c))
OC = (a / (a + c)) * √(ac + (ad² - cb²) / (a - c))
-/
theorem trapezoid_diagonal_segments 
  (a c b d : ℝ) 
  (ha : a > 0) 
  (hc : c > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hac : a ≠ c) :
  ∃ (AO OC : ℝ),
    AO = (c / (a + c)) * Real.sqrt (a * c + (a * d^2 - c * b^2) / (a - c)) ∧
    OC = (a / (a + c)) * Real.sqrt (a * c + (a * d^2 - c * b^2) / (a - c)) :=
by sorry

end trapezoid_diagonal_segments_l2312_231273


namespace quadratic_inequality_solution_sets_l2312_231215

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-2 : ℝ) 1 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 - b * x + a < 0} = Set.Ioo (-1 : ℝ) (1/2) := by
  sorry

end quadratic_inequality_solution_sets_l2312_231215


namespace fan_airflow_rate_l2312_231283

/-- Proves that the airflow rate of a fan is 10 liters per second, given the specified conditions. -/
theorem fan_airflow_rate : 
  ∀ (daily_operation_minutes : ℝ) (weekly_airflow_liters : ℝ),
    daily_operation_minutes = 10 →
    weekly_airflow_liters = 42000 →
    (weekly_airflow_liters / (daily_operation_minutes * 7 * 60)) = 10 := by
  sorry

end fan_airflow_rate_l2312_231283


namespace compound_carbon_atoms_l2312_231216

/-- Represents the number of Carbon atoms in a compound -/
def carbonAtoms (molecularWeight : ℕ) (hydrogenAtoms : ℕ) : ℕ :=
  (molecularWeight - hydrogenAtoms) / 12

/-- Proves that a compound with 6 Hydrogen atoms and a molecular weight of 78 amu contains 6 Carbon atoms -/
theorem compound_carbon_atoms :
  carbonAtoms 78 6 = 6 :=
by
  sorry

#eval carbonAtoms 78 6

end compound_carbon_atoms_l2312_231216


namespace speakers_cost_l2312_231220

def total_spent : ℚ := 387.85
def cd_player_cost : ℚ := 139.38
def new_tires_cost : ℚ := 112.46

theorem speakers_cost (total : ℚ) (cd : ℚ) (tires : ℚ) 
  (h1 : total = total_spent) 
  (h2 : cd = cd_player_cost) 
  (h3 : tires = new_tires_cost) : 
  total - (cd + tires) = 136.01 := by
  sorry

end speakers_cost_l2312_231220


namespace runners_meeting_time_l2312_231239

/-- The time (in seconds) after which two runners meet at the starting point -/
def meetingTime (p_time q_time : ℕ) : ℕ :=
  Nat.lcm p_time q_time

/-- Theorem stating that two runners with given lap times meet after a specific time -/
theorem runners_meeting_time :
  meetingTime 252 198 = 2772 := by
  sorry

end runners_meeting_time_l2312_231239


namespace abs_2x_plus_1_gt_3_l2312_231262

theorem abs_2x_plus_1_gt_3 (x : ℝ) : |2*x + 1| > 3 ↔ x > 1 ∨ x < -2 := by
  sorry

end abs_2x_plus_1_gt_3_l2312_231262


namespace no_integer_solution_l2312_231210

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 93^3 + b * 93^2 + c * 93 + d = 2) := by
sorry

end no_integer_solution_l2312_231210


namespace function_properties_l2312_231236

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_properties
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_neg : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, f (2 - x) = f x) ∧
  f 2 = f 0 :=
sorry

end function_properties_l2312_231236


namespace counterfeit_bag_identification_l2312_231254

/-- Represents a bag of coins -/
structure CoinBag where
  weight : ℕ  -- Weight of each coin in grams
  count : ℕ   -- Number of coins taken from the bag

/-- Creates a list of 10 coin bags with the specified counterfeit bag -/
def createBags (counterfeitBag : ℕ) : List CoinBag :=
  List.range 10 |>.map (fun i =>
    if i + 1 = counterfeitBag then
      { weight := 11, count := i + 1 }
    else
      { weight := 10, count := i + 1 })

/-- Calculates the total weight of coins from all bags -/
def totalWeight (bags : List CoinBag) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.weight * bag.count) 0

/-- The main theorem to prove -/
theorem counterfeit_bag_identification
  (counterfeitBag : ℕ) (h1 : 1 ≤ counterfeitBag) (h2 : counterfeitBag ≤ 10) :
  totalWeight (createBags counterfeitBag) - 550 = counterfeitBag := by
  sorry

#check counterfeit_bag_identification

end counterfeit_bag_identification_l2312_231254


namespace factorization_and_sum_of_coefficients_l2312_231288

theorem factorization_and_sum_of_coefficients :
  ∃ (a b c d e f : ℤ),
    (81 : ℚ) * x^4 - 256 * y^4 = (a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) ∧
    (a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) = (3 * x - 4 * y) * (3 * x + 4 * y) * (9 * x^2 + 16 * y^2) ∧
    a + b + c + d + e + f = 31 :=
by sorry

end factorization_and_sum_of_coefficients_l2312_231288


namespace optimal_seedlings_optimal_seedlings_count_l2312_231276

/-- Represents the profit per pot as a function of the number of seedlings -/
def profit_per_pot (n : ℕ) : ℝ :=
  n * (5 - 0.5 * (n - 4 : ℝ))

/-- The target profit per pot -/
def target_profit : ℝ := 24

/-- Theorem stating that 6 seedlings per pot achieves the target profit while minimizing costs -/
theorem optimal_seedlings :
  (profit_per_pot 6 = target_profit) ∧
  (∀ m : ℕ, m < 6 → profit_per_pot m < target_profit) ∧
  (∀ m : ℕ, m > 6 → profit_per_pot m ≤ target_profit) :=
sorry

/-- Corollary: 6 is the optimal number of seedlings per pot -/
theorem optimal_seedlings_count : ℕ :=
6

end optimal_seedlings_optimal_seedlings_count_l2312_231276


namespace union_equality_iff_m_range_l2312_231204

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + 2*m < 0}

-- State the theorem
theorem union_equality_iff_m_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (-1/2 ≤ m ∧ m ≤ 1) := by
  sorry

end union_equality_iff_m_range_l2312_231204


namespace infinite_hyperbolas_l2312_231214

/-- A hyperbola with asymptotes 2x ± 3y = 0 -/
def Hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), 4 * x^2 - 9 * y^2 = k ∧ k ≠ 0

/-- The set of all hyperbolas with asymptotes 2x ± 3y = 0 -/
def HyperbolaSet : Set ℝ :=
  {k : ℝ | Hyperbola k}

/-- Theorem stating that there are infinitely many hyperbolas with asymptotes 2x ± 3y = 0 -/
theorem infinite_hyperbolas : Set.Infinite HyperbolaSet := by
  sorry

end infinite_hyperbolas_l2312_231214


namespace infinitely_many_primes_dividing_special_form_l2312_231258

-- Define the set of primes we're interested in
def S : Set Nat :=
  {p | Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014)}

-- State the theorem
theorem infinitely_many_primes_dividing_special_form :
  Set.Infinite S :=
sorry

end infinitely_many_primes_dividing_special_form_l2312_231258


namespace arithmetic_computation_l2312_231279

theorem arithmetic_computation : -12 * 5 - (-4 * -2) + (-15 * -3) / 3 = -53 := by
  sorry

end arithmetic_computation_l2312_231279


namespace correct_calculation_l2312_231270

theorem correct_calculation : 
  (-2 - 3 = -5) ∧ 
  (-3^2 ≠ -6) ∧ 
  (1/2 / 2 ≠ 2 * 2) ∧ 
  ((-2/3)^2 ≠ 4/3) := by
sorry

end correct_calculation_l2312_231270


namespace square_division_into_rectangles_l2312_231260

theorem square_division_into_rectangles :
  ∃ (s : ℝ), s > 0 ∧
  ∃ (a : ℝ), a > 0 ∧
  7 * (2 * a^2) ≤ s^2 ∧
  2 * a ≤ s :=
sorry

end square_division_into_rectangles_l2312_231260


namespace tangent_line_and_inequality_l2312_231261

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := x * Real.log x
def g (a : ℝ) (x : ℝ) := -x^2 + a*x - 3

-- State the theorem
theorem tangent_line_and_inequality (a : ℝ) :
  -- Part 1: Tangent line equation
  (∀ x : ℝ, HasDerivAt f (x - 1) 1) ∧
  -- Part 2: Inequality condition
  (∀ x : ℝ, x > 0 → 2 * f x ≥ g a x) ↔ a ≤ 4 :=
sorry

end tangent_line_and_inequality_l2312_231261


namespace sum_of_roots_equation_l2312_231287

theorem sum_of_roots_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a ≠ b) →
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end sum_of_roots_equation_l2312_231287


namespace jacket_sale_profit_l2312_231286

/-- Represents a jacket sale with its cost and selling price -/
structure JacketSale where
  cost : ℝ
  sellingPrice : ℝ

/-- Calculates the profit or loss from a jacket sale -/
def profit (sale : JacketSale) : ℝ := sale.sellingPrice - sale.cost

theorem jacket_sale_profit :
  ∀ (jacket1 jacket2 : JacketSale),
    jacket1.sellingPrice = 80 ∧
    jacket2.sellingPrice = 80 ∧
    jacket1.sellingPrice = jacket1.cost * 1.6 ∧
    jacket2.sellingPrice = jacket2.cost * 0.8 →
    profit jacket1 + profit jacket2 = 10 := by
  sorry

end jacket_sale_profit_l2312_231286


namespace function_behavior_implies_a_range_l2312_231241

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + (a-1)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + (a-1)

theorem function_behavior_implies_a_range :
  ∀ a : ℝ,
  (∀ x ∈ Set.Ioo 1 4, (f_prime a x) < 0) →
  (∀ x ∈ Set.Ioi 6, (f_prime a x) > 0) →
  5 ≤ a ∧ a ≤ 7 :=
by sorry

end function_behavior_implies_a_range_l2312_231241


namespace percentage_of_male_employees_l2312_231212

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 6400) 
  (h2 : males_below_50 = 3120) : 
  (males_below_50 : ℚ) / (0.75 * total_employees) = 65 / 100 := by
  sorry

end percentage_of_male_employees_l2312_231212


namespace poles_not_moved_l2312_231217

theorem poles_not_moved (total_distance : ℕ) (original_spacing : ℕ) (new_spacing : ℕ) : 
  total_distance = 2340 ∧ 
  original_spacing = 45 ∧ 
  new_spacing = 60 → 
  (total_distance / (Nat.lcm original_spacing new_spacing)) - 1 = 12 := by
sorry

end poles_not_moved_l2312_231217


namespace rectangle_length_width_difference_l2312_231240

theorem rectangle_length_width_difference
  (perimeter : ℝ)
  (diagonal : ℝ)
  (h_perimeter : perimeter = 80)
  (h_diagonal : diagonal = 20 * Real.sqrt 2) :
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = perimeter ∧
    length^2 + width^2 = diagonal^2 ∧
    length - width = 0 := by
  sorry

end rectangle_length_width_difference_l2312_231240


namespace tangent_equation_solution_l2312_231201

open Real

theorem tangent_equation_solution (x : ℝ) : 
  8.482 * (3 * tan x - tan x ^ 3) / (1 - tan x ^ 2) * (cos (3 * x) + cos x) = 2 * sin (5 * x) ↔ 
  (∃ k : ℤ, x = k * π) ∨ (∃ k : ℤ, x = π / 8 * (2 * k + 1)) :=
sorry

end tangent_equation_solution_l2312_231201


namespace set_difference_equals_open_interval_l2312_231257

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x ≠ 1 ∧ x / (x - 1) ≤ 0}

-- Define the open interval (-1, 0)
def open_interval : Set ℝ := {x | -1 < x ∧ x < 0}

-- Theorem statement
theorem set_difference_equals_open_interval : M \ N = open_interval := by
  sorry

end set_difference_equals_open_interval_l2312_231257


namespace green_squares_count_l2312_231211

/-- Represents a grid with colored squares -/
structure ColoredGrid where
  rows : Nat
  squares_per_row : Nat
  red_rows : Nat
  red_squares_per_row : Nat
  blue_rows : Nat

/-- Calculates the number of green squares in the grid -/
def green_squares (grid : ColoredGrid) : Nat :=
  grid.rows * grid.squares_per_row - 
  (grid.red_rows * grid.red_squares_per_row + grid.blue_rows * grid.squares_per_row)

/-- Theorem stating that the number of green squares in the given grid configuration is 66 -/
theorem green_squares_count (grid : ColoredGrid) 
  (h1 : grid.rows = 10)
  (h2 : grid.squares_per_row = 15)
  (h3 : grid.red_rows = 4)
  (h4 : grid.red_squares_per_row = 6)
  (h5 : grid.blue_rows = 4) :
  green_squares grid = 66 := by
  sorry

#eval green_squares { rows := 10, squares_per_row := 15, red_rows := 4, red_squares_per_row := 6, blue_rows := 4 }

end green_squares_count_l2312_231211


namespace brothers_age_difference_l2312_231244

/-- Represents a year in the format ABCD where A, B, C, D are digits --/
structure Year :=
  (value : ℕ)
  (in_19th_century : value ≥ 1800 ∧ value < 1900)

/-- Represents a year in the format ABCD where A, B, C, D are digits --/
structure Year' :=
  (value : ℕ)
  (in_20th_century : value ≥ 1900 ∧ value < 2000)

/-- Sum of digits of a number --/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Age of a person born in year y at the current year current_year --/
def age (y : ℕ) (current_year : ℕ) : ℕ := current_year - y

theorem brothers_age_difference 
  (peter_birth : Year) 
  (paul_birth : Year') 
  (current_year : ℕ) 
  (h1 : age peter_birth.value current_year = sum_of_digits peter_birth.value)
  (h2 : age paul_birth.value current_year = sum_of_digits paul_birth.value) :
  age peter_birth.value current_year - age paul_birth.value current_year = 9 :=
sorry

end brothers_age_difference_l2312_231244


namespace share_price_is_31_l2312_231223

/-- The price at which an investor bought shares, given the dividend rate,
    face value, and return on investment. -/
def share_purchase_price (dividend_rate : ℚ) (face_value : ℚ) (roi : ℚ) : ℚ :=
  (dividend_rate * face_value) / roi

/-- Theorem stating that under the given conditions, the share purchase price is 31. -/
theorem share_price_is_31 :
  let dividend_rate : ℚ := 155 / 1000
  let face_value : ℚ := 50
  let roi : ℚ := 1 / 4
  share_purchase_price dividend_rate face_value roi = 31 := by
  sorry

end share_price_is_31_l2312_231223


namespace ms_jones_class_size_l2312_231243

theorem ms_jones_class_size :
  ∀ (num_students : ℕ),
    (num_students : ℝ) * 0.3 * (1/3) * 10 = 50 →
    num_students = 50 :=
by
  sorry

end ms_jones_class_size_l2312_231243


namespace triangle_cut_theorem_l2312_231285

theorem triangle_cut_theorem (x : ℝ) : 
  (∀ y : ℝ, y ≥ x → (12 - y) + (18 - y) ≤ 24 - y) →
  (∀ z : ℝ, z < x → ∃ a b c : ℝ, 
    a + b > c ∧ 
    a + c > b ∧ 
    b + c > a ∧
    a = 12 - z ∧ 
    b = 18 - z ∧ 
    c = 24 - z) →
  x = 6 := by
sorry

end triangle_cut_theorem_l2312_231285


namespace edward_money_problem_l2312_231245

theorem edward_money_problem (initial spent received final : ℤ) :
  spent = 17 →
  received = 10 →
  final = 7 →
  initial - spent + received = final →
  initial = 14 := by
sorry

end edward_money_problem_l2312_231245


namespace matt_writing_difference_l2312_231248

/-- The number of words Matt can write per minute with his right hand -/
def right_hand_speed : ℕ := 10

/-- The number of words Matt can write per minute with his left hand -/
def left_hand_speed : ℕ := 7

/-- The duration of time in minutes -/
def duration : ℕ := 5

/-- The difference in words written between Matt's right and left hands over the given duration -/
def word_difference : ℕ := (right_hand_speed - left_hand_speed) * duration

theorem matt_writing_difference : word_difference = 15 := by
  sorry

end matt_writing_difference_l2312_231248


namespace tan_double_angle_problem_l2312_231209

theorem tan_double_angle_problem (θ : Real) 
  (h1 : Real.tan (2 * θ) = -2) 
  (h2 : π < 2 * θ) 
  (h3 : 2 * θ < 2 * π) : 
  Real.sin θ ^ 4 - Real.cos θ ^ 4 = -1 / Real.sqrt 5 := by
  sorry

end tan_double_angle_problem_l2312_231209


namespace doubled_side_cube_weight_l2312_231218

/-- Represents the weight of a cube given its side length -/
def cube_weight (side_length : ℝ) : ℝ := sorry

theorem doubled_side_cube_weight (original_side : ℝ) :
  cube_weight original_side = 6 →
  cube_weight (2 * original_side) = 48 := by
  sorry

end doubled_side_cube_weight_l2312_231218


namespace tree_height_difference_l2312_231213

/-- The height of the birch tree in feet -/
def birch_height : ℚ := 49/4

/-- The height of the pine tree in feet -/
def pine_height : ℚ := 37/2

/-- The difference in height between the pine tree and the birch tree -/
def height_difference : ℚ := pine_height - birch_height

theorem tree_height_difference : height_difference = 25/4 := by sorry

end tree_height_difference_l2312_231213


namespace axis_of_symmetry_l2312_231299

-- Define a function f with the given symmetry property
def f : ℝ → ℝ := sorry

-- State the symmetry condition
axiom f_symmetry (x : ℝ) : f x = f (5 - x)

-- Define the line of symmetry
def line_of_symmetry : ℝ → Prop := λ x ↦ x = 2.5

-- Theorem stating that the line x = 2.5 is an axis of symmetry
theorem axis_of_symmetry :
  ∀ (x y : ℝ), f x = y → f (5 - x) = y → line_of_symmetry ((x + (5 - x)) / 2) := by
  sorry

end axis_of_symmetry_l2312_231299


namespace average_songs_theorem_l2312_231296

/-- Represents a band's performance schedule --/
structure BandPerformance where
  repertoire : ℕ
  first_set : ℕ
  second_set : ℕ
  encores : ℕ

/-- Calculates the average number of songs for the remaining sets --/
def average_remaining_songs (b : BandPerformance) : ℚ :=
  let songs_played := b.first_set + b.second_set + b.encores
  let remaining_songs := b.repertoire - songs_played
  let remaining_sets := 3
  (remaining_songs : ℚ) / remaining_sets

/-- Theorem stating the average number of songs for the remaining sets --/
theorem average_songs_theorem (b : BandPerformance) 
  (h1 : b.repertoire = 50)
  (h2 : b.first_set = 8)
  (h3 : b.second_set = 12)
  (h4 : b.encores = 4) :
  average_remaining_songs b = 26 / 3 := by
  sorry

end average_songs_theorem_l2312_231296


namespace largest_integer_less_than_M_div_100_l2312_231206

def factorial (n : ℕ) : ℕ := Nat.factorial n

def M : ℚ :=
  (1 / (factorial 3 * factorial 19) +
   1 / (factorial 4 * factorial 18) +
   1 / (factorial 5 * factorial 17) +
   1 / (factorial 6 * factorial 16) +
   1 / (factorial 7 * factorial 15) +
   1 / (factorial 8 * factorial 14) +
   1 / (factorial 9 * factorial 13) +
   1 / (factorial 10 * factorial 12)) * (factorial 1 * factorial 21)

theorem largest_integer_less_than_M_div_100 :
  Int.floor (M / 100) = 952 := by sorry

end largest_integer_less_than_M_div_100_l2312_231206


namespace grandfather_animals_l2312_231256

theorem grandfather_animals (h p k s : ℕ) : 
  h + p + k + s = 40 →
  h = 3 * k →
  s - 8 = h + p →
  40 - (1/4) * h + (3/4) * h = 46 →
  h = 12 ∧ p = 2 ∧ k = 4 ∧ s = 22 := by sorry

end grandfather_animals_l2312_231256


namespace semicircle_area_shaded_area_proof_l2312_231219

/-- The area of semicircles lined up along a line -/
theorem semicircle_area (diameter : Real) (length : Real) : 
  diameter > 0 → length > 0 → 
  (length / diameter) * (π * diameter^2 / 8) = 3 * π * length / 2 := by
  sorry

/-- The specific case for the given problem -/
theorem shaded_area_proof :
  let diameter : Real := 4
  let length : Real := 24  -- 2 feet in inches
  (length / diameter) * (π * diameter^2 / 8) = 12 * π := by
  sorry

end semicircle_area_shaded_area_proof_l2312_231219


namespace half_area_triangle_l2312_231203

/-- A square in a 2D plane -/
structure Square where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ
  w : ℝ × ℝ

/-- The area of a triangle given three points -/
def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

/-- The area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Theorem: The coordinates (3, 3) for point T' result in the area of triangle YZT' 
    being half the area of square XYZW, given that XYZW is a square with X at (0,0) 
    and Z at (3,3) -/
theorem half_area_triangle (xyzw : Square) 
  (h1 : xyzw.x = (0, 0))
  (h2 : xyzw.z = (3, 3))
  (t' : ℝ × ℝ)
  (h3 : t' = (3, 3)) : 
  triangleArea xyzw.y xyzw.z t' = (1/2) * squareArea xyzw := by
  sorry

end half_area_triangle_l2312_231203


namespace sons_age_l2312_231222

theorem sons_age (father_age son_age : ℕ) 
  (h1 : 2 * son_age + father_age = 70)
  (h2 : 2 * father_age + son_age = 95)
  (h3 : father_age = 40) : son_age = 15 := by
  sorry

end sons_age_l2312_231222


namespace class_size_l2312_231269

theorem class_size (chinese : ℕ) (math : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chinese = 15)
  (h2 : math = 18)
  (h3 : both = 8)
  (h4 : neither = 20) :
  chinese + math - both + neither = 45 := by
  sorry

end class_size_l2312_231269


namespace inequality_always_true_l2312_231295

theorem inequality_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end inequality_always_true_l2312_231295


namespace fruit_difference_l2312_231255

/-- Given the number of apples harvested and the ratio of peaches to apples,
    prove that the difference between the number of peaches and apples is 120. -/
theorem fruit_difference (apples : ℕ) (peach_ratio : ℕ) : apples = 60 → peach_ratio = 3 →
  peach_ratio * apples - apples = 120 := by
  sorry

end fruit_difference_l2312_231255


namespace square_inequality_negative_l2312_231232

theorem square_inequality_negative (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_negative_l2312_231232


namespace quadratic_properties_l2312_231205

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties (a b c m : ℝ) (h_a : a ≠ 0) :
  quadratic a b c (-2) = m ∧
  quadratic a b c (-1) = 1 ∧
  quadratic a b c 0 = -1 ∧
  quadratic a b c 1 = 1 ∧
  quadratic a b c 2 = 7 →
  (∀ x, quadratic a b c x = quadratic a b c (-x)) ∧  -- Symmetry axis at x = 0
  quadratic a b c 0 = -1 ∧                           -- Vertex at (0, -1)
  m = 7 ∧
  a > 0 := by sorry

end quadratic_properties_l2312_231205


namespace blue_to_red_ratio_l2312_231234

/-- Represents the number of socks of each color --/
structure SockCount where
  blue : ℕ
  black : ℕ
  red : ℕ
  white : ℕ

/-- The conditions of Joseph's sock collection --/
def josephsSocks : SockCount → Prop :=
  fun s => s.blue = s.black + 6 ∧
           s.red = s.white - 2 ∧
           s.red = 6 ∧
           s.blue + s.black + s.red + s.white = 28

/-- The theorem stating the ratio of blue to red socks --/
theorem blue_to_red_ratio (s : SockCount) (h : josephsSocks s) :
  s.blue / s.red = 7 / 3 := by
  sorry

end blue_to_red_ratio_l2312_231234


namespace marble_group_size_l2312_231202

theorem marble_group_size :
  ∀ (x : ℕ),
  (144 / x : ℚ) = (144 / (x + 2) : ℚ) + 1 →
  x = 16 :=
by
  sorry

end marble_group_size_l2312_231202


namespace price_per_working_game_l2312_231233

def total_games : ℕ := 10
def non_working_games : ℕ := 8
def total_earnings : ℕ := 12

theorem price_per_working_game :
  (total_earnings : ℚ) / (total_games - non_working_games) = 6 := by
  sorry

end price_per_working_game_l2312_231233


namespace curve_C₁_and_constant_product_l2312_231290

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop :=
  ∀ (x' y' : ℝ), C₂ x' y' → (x - x')^2 + (y - y')^2 ≥ (y + 2)^2

-- Define the line y = -4
def line_y_neg4 (x y : ℝ) : Prop := y = -4

-- Define the tangent lines from a point to C₂
def tangent_to_C₂ (x₀ y₀ x y : ℝ) : Prop :=
  ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ (x₀^2 - 9) * k^2 + 18 * x₀ * k + 72 = 0

-- Theorem statement
theorem curve_C₁_and_constant_product :
  (∀ x y : ℝ, C₁ x y ↔ x^2 = 20 * y) ∧
  (∀ x₀ : ℝ, x₀ ≠ 3 ∧ x₀ ≠ -3 →
    ∀ x₁ x₂ x₃ x₄ : ℝ,
    (∃ y₀, line_y_neg4 x₀ y₀) →
    (∃ y₁, C₁ x₁ y₁ ∧ tangent_to_C₂ x₀ (-4) x₁ y₁) →
    (∃ y₂, C₁ x₂ y₂ ∧ tangent_to_C₂ x₀ (-4) x₂ y₂) →
    (∃ y₃, C₁ x₃ y₃ ∧ tangent_to_C₂ x₀ (-4) x₃ y₃) →
    (∃ y₄, C₁ x₄ y₄ ∧ tangent_to_C₂ x₀ (-4) x₄ y₄) →
    x₁ * x₂ * x₃ * x₄ = 6400) :=
sorry

end curve_C₁_and_constant_product_l2312_231290


namespace number_reciprocal_relation_l2312_231259

theorem number_reciprocal_relation (x y : ℝ) : 
  x > 0 → x = 3 → x + y = 60 * (1 / x) → y = 17 := by
  sorry

end number_reciprocal_relation_l2312_231259


namespace calculate_expression_l2312_231242

theorem calculate_expression : 
  2 * Real.tan (60 * π / 180) - (-2023)^(0 : ℝ) + (1/2)^(-1 : ℝ) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 := by
  sorry

end calculate_expression_l2312_231242


namespace average_score_is_two_l2312_231249

/-- Represents the distribution of scores in a class test --/
structure ScoreDistribution where
  score3 : Real
  score2 : Real
  score1 : Real
  score0 : Real
  sum_to_one : score3 + score2 + score1 + score0 = 1

/-- Calculates the average score given a score distribution --/
def averageScore (d : ScoreDistribution) : Real :=
  3 * d.score3 + 2 * d.score2 + 1 * d.score1 + 0 * d.score0

/-- Theorem: The average score for the given distribution is 2.0 --/
theorem average_score_is_two :
  let d : ScoreDistribution := {
    score3 := 0.3,
    score2 := 0.5,
    score1 := 0.1,
    score0 := 0.1,
    sum_to_one := by norm_num
  }
  averageScore d = 2.0 := by sorry

end average_score_is_two_l2312_231249


namespace triangle_side_lengths_l2312_231221

-- Define the triangle ABC
theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  -- Conditions
  a = 1 →
  B = π / 4 → -- 45° in radians
  S = 2 →
  S = (1 / 2) * a * c * Real.sin B →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  -- Conclusion
  c = 4 * Real.sqrt 2 ∧ b = 5 := by
sorry


end triangle_side_lengths_l2312_231221


namespace solution_for_x_l2312_231294

theorem solution_for_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (eq1 : x + 1 / z = 15) (eq2 : z + 1 / x = 9 / 20) :
  x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by sorry

end solution_for_x_l2312_231294


namespace batsman_running_percentage_l2312_231251

theorem batsman_running_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (h1 : total_runs = 125)
  (h2 : boundaries = 5)
  (h3 : sixes = 5) :
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 60 := by
  sorry

end batsman_running_percentage_l2312_231251


namespace count_integers_satisfying_equation_l2312_231281

-- Define the function g
def g (n : ℤ) : ℤ := ⌈(101 * n : ℚ) / 102⌉ - ⌊(102 * n : ℚ) / 103⌋

-- State the theorem
theorem count_integers_satisfying_equation : 
  (∃ (S : Finset ℤ), (∀ n ∈ S, g n = 1) ∧ (∀ n ∉ S, g n ≠ 1) ∧ Finset.card S = 10506) :=
sorry

end count_integers_satisfying_equation_l2312_231281


namespace consecutive_divisible_numbers_l2312_231246

theorem consecutive_divisible_numbers :
  ∃ (n : ℕ),
    (5 ∣ n) ∧
    (4 ∣ n + 1) ∧
    (3 ∣ n + 2) ∧
    (∀ (m : ℕ), (5 ∣ m) ∧ (4 ∣ m + 1) ∧ (3 ∣ m + 2) → n ≤ m) ∧
    n = 55 := by
  sorry

end consecutive_divisible_numbers_l2312_231246


namespace min_product_of_three_numbers_l2312_231227

theorem min_product_of_three_numbers (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 2*y ∧ x ≤ 2*z ∧ y ≤ 2*x ∧ y ≤ 2*z ∧ z ≤ 2*x ∧ z ≤ 2*y →
  x * y * z ≥ 1/32 := by
sorry

end min_product_of_three_numbers_l2312_231227


namespace triangle_division_result_l2312_231265

-- Define the process of dividing triangles
def divide_triangles (n : ℕ) : ℕ := 3^n

-- Define the side length after n iterations
def side_length (n : ℕ) : ℚ := 1 / 2^n

-- Theorem statement
theorem triangle_division_result :
  let iterations : ℕ := 12
  let final_count : ℕ := divide_triangles iterations
  let final_side_length : ℚ := side_length iterations
  final_count = 531441 ∧ final_side_length = 1 / 2^12 := by
  sorry


end triangle_division_result_l2312_231265


namespace train_length_proof_l2312_231291

/-- Given a train that crosses a 500-meter platform in 48 seconds and a signal pole in 18 seconds,
    prove that its length is 300 meters. -/
theorem train_length_proof (L : ℝ) : (L + 500) / 48 = L / 18 ↔ L = 300 := by
  sorry

end train_length_proof_l2312_231291


namespace randy_piggy_bank_l2312_231289

/-- Calculates the initial amount in Randy's piggy bank -/
def initial_amount (spend_per_trip : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) (amount_left : ℕ) : ℕ :=
  spend_per_trip * trips_per_month * months_per_year + amount_left

/-- Proves that Randy initially had $200 in his piggy bank -/
theorem randy_piggy_bank : initial_amount 2 4 12 104 = 200 := by
  sorry

end randy_piggy_bank_l2312_231289


namespace shirt_count_l2312_231293

theorem shirt_count (total_shirt_price : ℝ) (total_sweater_price : ℝ) (sweater_count : ℕ) (price_difference : ℝ) :
  total_shirt_price = 400 →
  total_sweater_price = 1500 →
  sweater_count = 75 →
  (total_sweater_price / sweater_count) = (total_shirt_price / (total_shirt_price / 16)) + price_difference →
  price_difference = 4 →
  (total_shirt_price / 16 : ℝ) = 25 :=
by sorry

end shirt_count_l2312_231293


namespace mindy_tax_rate_is_25_percent_l2312_231271

/-- Calculates Mindy's tax rate given Mork's tax rate, their income ratio, and combined tax rate -/
def mindyTaxRate (morkTaxRate : ℚ) (incomeRatio : ℚ) (combinedTaxRate : ℚ) : ℚ :=
  (combinedTaxRate * (1 + incomeRatio) - morkTaxRate) / incomeRatio

/-- Proves that Mindy's tax rate is 25% given the specified conditions -/
theorem mindy_tax_rate_is_25_percent :
  mindyTaxRate (40 / 100) 4 (28 / 100) = 25 / 100 := by
  sorry

#eval mindyTaxRate (40 / 100) 4 (28 / 100)

end mindy_tax_rate_is_25_percent_l2312_231271


namespace parabola_symmetry_l2312_231277

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with inclination angle -/
structure Line where
  angle : ℝ

/-- Function to check if a point is on the parabola -/
def on_parabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (pt : Point) : Prop :=
  true -- Simplified for this problem

/-- Function to check if two points are symmetric with respect to a line -/
def symmetric_wrt_line (p1 p2 : Point) (l : Line) : Prop :=
  true -- Simplified for this problem

/-- Main theorem -/
theorem parabola_symmetry (para : Parabola) (l : Line) (p q : Point) :
  l.angle = π / 6 →
  passes_through l (Point.mk (para.p / 2) 0) →
  on_parabola para p →
  q = Point.mk 5 0 →
  symmetric_wrt_line p q l →
  para.p = 2 := by
  sorry

end parabola_symmetry_l2312_231277


namespace division_problem_l2312_231208

theorem division_problem : 
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 := by
  sorry

end division_problem_l2312_231208


namespace scale_division_l2312_231282

/-- Proves that dividing a scale of length 7 feet 12 inches into 4 equal parts results in parts that are 2 feet long each. -/
theorem scale_division (scale_length_feet : ℕ) (scale_length_inches : ℕ) (num_parts : ℕ) :
  scale_length_feet = 7 →
  scale_length_inches = 12 →
  num_parts = 4 →
  (scale_length_feet * 12 + scale_length_inches) / num_parts = 24 := by
  sorry

#check scale_division

end scale_division_l2312_231282


namespace same_color_marble_probability_l2312_231224

theorem same_color_marble_probability : 
  let total_marbles : ℕ := 3 + 6 + 8
  let red_marbles : ℕ := 3
  let white_marbles : ℕ := 6
  let blue_marbles : ℕ := 8
  let drawn_marbles : ℕ := 4
  
  (Nat.choose white_marbles drawn_marbles + Nat.choose blue_marbles drawn_marbles : ℚ) /
  (Nat.choose total_marbles drawn_marbles : ℚ) = 17 / 476 :=
by sorry

end same_color_marble_probability_l2312_231224


namespace square_of_trinomial_l2312_231284

theorem square_of_trinomial (a b c : ℝ) : 
  (a - 2*b - 3*c)^2 = a^2 - 4*a*b + 4*b^2 - 6*a*c + 12*b*c + 9*c^2 := by
  sorry

end square_of_trinomial_l2312_231284


namespace infinitely_many_special_integers_l2312_231272

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A function that checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

/-- A function that checks if a number is a perfect fifth power -/
def isPerfectFifthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

/-- The main theorem stating that there are infinitely many integers satisfying the conditions -/
theorem infinitely_many_special_integers :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
    ∀ k : ℕ, 
      isPerfectSquare (2 * f k) ∧ 
      isPerfectCube (3 * f k) ∧ 
      isPerfectFifthPower (5 * f k) :=
sorry

end infinitely_many_special_integers_l2312_231272


namespace article_font_pages_l2312_231280

theorem article_font_pages (total_words : ℕ) (large_font_words : ℕ) (small_font_words : ℕ) (total_pages : ℕ) :
  total_words = 48000 →
  large_font_words = 1800 →
  small_font_words = 2400 →
  total_pages = 21 →
  ∃ (large_pages : ℕ) (small_pages : ℕ),
    large_pages + small_pages = total_pages ∧
    large_pages * large_font_words + small_pages * small_font_words = total_words ∧
    large_pages = 4 :=
by sorry

end article_font_pages_l2312_231280


namespace cards_at_home_l2312_231278

def cards_in_hospital : ℕ := 403
def total_cards : ℕ := 690

theorem cards_at_home : total_cards - cards_in_hospital = 287 := by
  sorry

end cards_at_home_l2312_231278


namespace cube_root_8000_l2312_231235

theorem cube_root_8000 :
  ∃ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3 : ℝ) = (8000 : ℝ)^(1/3 : ℝ) ∧
  c = 20 ∧ d = 1 ∧ c + d = 21 ∧
  ∀ (c' d' : ℕ+), (c' : ℝ) * (d' : ℝ)^(1/3 : ℝ) = (8000 : ℝ)^(1/3 : ℝ) → d ≤ d' :=
by sorry

end cube_root_8000_l2312_231235


namespace kathleens_allowance_increase_l2312_231253

theorem kathleens_allowance_increase (middle_school_allowance senior_year_allowance : ℚ) : 
  middle_school_allowance = 8 + 2 →
  senior_year_allowance = 2 * middle_school_allowance + 5 →
  (senior_year_allowance - middle_school_allowance) / middle_school_allowance * 100 = 150 := by
  sorry

end kathleens_allowance_increase_l2312_231253


namespace cow_starting_weight_l2312_231298

/-- The starting weight of a cow, given certain conditions about its weight gain and value increase. -/
theorem cow_starting_weight (W : ℝ) : W = 400 :=
  -- Given conditions
  have weight_increase : W * 1.5 = W + W * 0.5 := by sorry
  have price_per_pound : ℝ := 3
  have value_increase : W * 1.5 * price_per_pound - W * price_per_pound = 600 := by sorry

  -- Proof
  sorry

end cow_starting_weight_l2312_231298


namespace bug_return_probability_l2312_231230

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - Q n)

/-- The probability of returning to the starting vertex on the eighth move -/
theorem bug_return_probability : Q 8 = 43/128 := by
  sorry

end bug_return_probability_l2312_231230


namespace algorithm_computes_gcd_l2312_231266

/-- The algorithm described in the problem -/
def algorithm (x y : ℕ) : ℕ :=
  let rec loop (m n : ℕ) : ℕ :=
    if m / n = m / n then n
    else loop n (m % n)
  loop (max x y) (min x y)

/-- Theorem stating that the algorithm computes the GCD -/
theorem algorithm_computes_gcd (x y : ℕ) :
  algorithm x y = Nat.gcd x y := by sorry

end algorithm_computes_gcd_l2312_231266


namespace decimal_division_multiplication_l2312_231238

theorem decimal_division_multiplication : (0.08 / 0.005) * 2 = 32 := by sorry

end decimal_division_multiplication_l2312_231238


namespace det_special_matrix_l2312_231263

/-- The determinant of the matrix [[1, x, x^2], [1, x+1, (x+1)^2], [1, x, (x+1)^2]] is equal to x + 1 -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![1, x, x^2; 1, x+1, (x+1)^2; 1, x, (x+1)^2] = x + 1 := by
  sorry

end det_special_matrix_l2312_231263


namespace davis_remaining_sticks_l2312_231207

/-- The number of popsicle sticks Miss Davis had initially -/
def initial_sticks : ℕ := 170

/-- The number of popsicle sticks given to each group -/
def sticks_per_group : ℕ := 15

/-- The number of groups in Miss Davis's class -/
def number_of_groups : ℕ := 10

/-- The number of popsicle sticks Miss Davis has left -/
def remaining_sticks : ℕ := initial_sticks - (sticks_per_group * number_of_groups)

theorem davis_remaining_sticks : remaining_sticks = 20 := by
  sorry

end davis_remaining_sticks_l2312_231207


namespace modulus_of_z_l2312_231267

def z : ℂ := 3 + 4 * Complex.I

theorem modulus_of_z : Complex.abs z = 5 := by sorry

end modulus_of_z_l2312_231267


namespace no_natural_solutions_l2312_231226

theorem no_natural_solutions (k x y z : ℕ) (h : k > 3) :
  x^2 + y^2 + z^2 ≠ k * x * y * z :=
sorry

end no_natural_solutions_l2312_231226


namespace x_eleven_percent_greater_than_80_l2312_231228

/-- If x is 11 percent greater than 80, then x equals 88.8 -/
theorem x_eleven_percent_greater_than_80 (x : ℝ) :
  x = 80 * (1 + 11 / 100) → x = 88.8 := by
  sorry

end x_eleven_percent_greater_than_80_l2312_231228


namespace log_condition_equivalence_l2312_231200

theorem log_condition_equivalence (m n : ℝ) 
  (hm_pos : m > 0) (hm_neq_one : m ≠ 1) (hn_pos : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) := by
  sorry

end log_condition_equivalence_l2312_231200


namespace seating_arrangement_theorem_l2312_231297

structure SeatingArrangement where
  rows_6 : Nat
  rows_8 : Nat
  rows_9 : Nat
  total_people : Nat
  max_rows : Nat

def is_valid (s : SeatingArrangement) : Prop :=
  s.rows_6 * 6 + s.rows_8 * 8 + s.rows_9 * 9 = s.total_people ∧
  s.rows_6 + s.rows_8 + s.rows_9 ≤ s.max_rows

theorem seating_arrangement_theorem :
  ∃ (s : SeatingArrangement),
    s.total_people = 58 ∧
    s.max_rows = 7 ∧
    is_valid s ∧
    s.rows_9 = 4 :=
by sorry

end seating_arrangement_theorem_l2312_231297


namespace max_correct_is_23_l2312_231247

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers for a given test score --/
def max_correct_answers (ts : TestScore) : ℕ :=
  sorry

/-- Theorem stating that for the given test conditions, the maximum number of correct answers is 23 --/
theorem max_correct_is_23 :
  let ts : TestScore := {
    total_questions := 30,
    correct_points := 4,
    incorrect_points := -1,
    total_score := 85
  }
  max_correct_answers ts = 23 := by
  sorry

end max_correct_is_23_l2312_231247


namespace max_value_of_f_on_interval_l2312_231231

def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 1 → f y ≤ f x) ∧
  f x = 3 :=
sorry

end max_value_of_f_on_interval_l2312_231231


namespace election_winner_percentage_l2312_231237

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 992 →
  margin = 384 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 :=
by
  sorry

end election_winner_percentage_l2312_231237


namespace equation_solution_l2312_231264

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 1 ∧ x ≠ -6 ∧ (3*x + 6)/((x^2 + 5*x - 6)) = (3 - x)/(x - 1) ∧ x = -4 := by
  sorry

end equation_solution_l2312_231264


namespace factor_expression_l2312_231250

theorem factor_expression (a b : ℝ) : 56 * b^2 * a^2 + 168 * b * a = 56 * b * a * (b * a + 3) := by
  sorry

end factor_expression_l2312_231250


namespace fourth_root_unity_sum_l2312_231275

/-- Given a nonreal complex number ω that is a fourth root of unity,
    prove that (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 -/
theorem fourth_root_unity_sum (ω : ℂ) 
  (h1 : ω^4 = 1) 
  (h2 : ω ≠ 1 ∧ ω ≠ -1 ∧ ω ≠ Complex.I ∧ ω ≠ -Complex.I) : 
  (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 := by
  sorry

end fourth_root_unity_sum_l2312_231275
