import Mathlib

namespace problem_solution_l3061_306127

theorem problem_solution (m n : ℝ) 
  (hm : 3 * m^2 + 5 * m - 3 = 0) 
  (hn : 3 * n^2 - 5 * n - 3 = 0) 
  (hmn : m * n ≠ 1) : 
  1 / n^2 + m / n - (5/3) * m = 25/9 := by
  sorry

end problem_solution_l3061_306127


namespace triangle_circumcircle_intersection_l3061_306122

/-- Triangle PQR with sides PQ = 47, QR = 14, and RP = 50 -/
structure Triangle (P Q R : ℝ × ℝ) :=
  (pq : dist P Q = 47)
  (qr : dist Q R = 14)
  (rp : dist R P = 50)

/-- The circumcircle of triangle PQR -/
def circumcircle (P Q R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {S | dist S P = dist S Q ∧ dist S Q = dist S R}

/-- The perpendicular bisector of RP -/
def perpBisector (R P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {S | dist S R = dist S P ∧ (S.1 - R.1) * (P.1 - R.1) + (S.2 - R.2) * (P.2 - R.2) = 0}

/-- S is on the opposite side of RP from Q -/
def oppositeSide (S Q R P : ℝ × ℝ) : Prop :=
  ((S.1 - R.1) * (P.2 - R.2) - (S.2 - R.2) * (P.1 - R.1)) *
  ((Q.1 - R.1) * (P.2 - R.2) - (Q.2 - R.2) * (P.1 - R.1)) < 0

theorem triangle_circumcircle_intersection
  (P Q R : ℝ × ℝ)
  (tri : Triangle P Q R)
  (S : ℝ × ℝ)
  (h1 : S ∈ circumcircle P Q R)
  (h2 : S ∈ perpBisector R P)
  (h3 : oppositeSide S Q R P) :
  dist P S = 8 * Real.sqrt 47 :=
sorry

end triangle_circumcircle_intersection_l3061_306122


namespace battery_life_comparison_l3061_306147

/-- Represents the charge of a battery -/
structure BatteryCharge where
  charge : ℝ
  positive : charge > 0

/-- Represents a clock powered by batteries -/
structure Clock where
  batteries : ℕ
  batteryType : BatteryCharge

/-- The problem statement -/
theorem battery_life_comparison 
  (battery_a battery_b : BatteryCharge)
  (clock_1 clock_2 : Clock)
  (h1 : battery_a.charge = 6 * battery_b.charge)
  (h2 : clock_1.batteries = 4 ∧ clock_1.batteryType = battery_a)
  (h3 : clock_2.batteries = 3 ∧ clock_2.batteryType = battery_b)
  (h4 : (clock_2.batteries : ℝ) * clock_2.batteryType.charge = 2)
  : (clock_1.batteries : ℝ) * clock_1.batteryType.charge - 
    (clock_2.batteries : ℝ) * clock_2.batteryType.charge = 14 := by
  sorry

#check battery_life_comparison

end battery_life_comparison_l3061_306147


namespace zoo_animals_l3061_306169

/-- The number of sea lions at the zoo -/
def sea_lions : ℕ := 42

/-- The number of penguins at the zoo -/
def penguins : ℕ := sea_lions + 84

/-- The number of flamingos at the zoo -/
def flamingos : ℕ := penguins + 42

theorem zoo_animals :
  (4 : ℚ) * sea_lions = 11 * sea_lions - 7 * 84 ∧
  7 * penguins = 11 * sea_lions + 7 * 42 ∧
  4 * flamingos = 7 * penguins + 4 * 42 :=
by sorry

#check zoo_animals

end zoo_animals_l3061_306169


namespace exactly_two_correct_l3061_306172

-- Define a mapping
def Mapping (A B : Type) := A → B

-- Define a function
def Function (α : Type) := α → ℝ

-- Define an odd function
def OddFunction (f : Function ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define propositions
def Proposition1 (A B : Type) : Prop :=
  ∃ (f : Mapping A B), ∃ b : B, ∀ a : A, f a ≠ b

def Proposition2 : Prop :=
  ∀ (f : Function ℝ) (t : ℝ), ∃! x : ℝ, f x = t

def Proposition3 (f : Function ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) → OddFunction f

def Proposition4 (f : Function ℝ) : Prop :=
  (∀ x, 0 ≤ f (2*x - 1) ∧ f (2*x - 1) ≤ 1) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 1)

-- Theorem statement
theorem exactly_two_correct :
  (Proposition1 ℝ ℝ) ∧
  (∃ f : Function ℝ, Proposition3 f) ∧
  ¬(Proposition2) ∧
  ¬(∃ f : Function ℝ, Proposition4 f) :=
sorry

end exactly_two_correct_l3061_306172


namespace farm_animals_l3061_306124

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (cows : ℕ) : 
  total_animals = 120 →
  total_legs = 350 →
  total_animals = chickens + cows →
  total_legs = 2 * chickens + 4 * cows →
  chickens = 65 := by
  sorry

end farm_animals_l3061_306124


namespace repeating_decimal_sum_div_fifth_l3061_306123

/-- Represents a repeating decimal with a two-digit repeat -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (a * 100 + b) / 99

theorem repeating_decimal_sum_div_fifth :
  let x := RepeatingDecimal 8 3
  let y := RepeatingDecimal 1 8
  (x + y) / (1/5) = 505/99 := by
  sorry

end repeating_decimal_sum_div_fifth_l3061_306123


namespace quadratic_integer_root_existence_l3061_306198

theorem quadratic_integer_root_existence (a b c : ℕ) (h : a + b + c = 2000) :
  ∃ (a' b' c' : ℤ), 
    (∃ (x : ℤ), a' * x^2 + b' * x + c' = 0) ∧ 
    (|a - a'| + |b - b'| + |c - c'| : ℤ) ≤ 1050 := by
  sorry

end quadratic_integer_root_existence_l3061_306198


namespace negation_of_existence_statement_l3061_306174

theorem negation_of_existence_statement :
  ¬(∃ x : ℝ, x ≤ 0) ↔ (∀ x : ℝ, x > 0) := by sorry

end negation_of_existence_statement_l3061_306174


namespace trivia_team_size_l3061_306153

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  absent_members = 2 →
  points_per_member = 6 →
  total_points = 18 →
  ∃ initial_members : ℕ, initial_members = 5 ∧ 
    points_per_member * (initial_members - absent_members) = total_points :=
by sorry

end trivia_team_size_l3061_306153


namespace ellipse_hyperbola_property_l3061_306125

/-- Given an ellipse and a hyperbola with shared foci, prove a property of the ellipse's semi-minor axis --/
theorem ellipse_hyperbola_property (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
   ∃ x' y' : ℝ, x'^2 - y'^2/4 = 1 ∧ 
   ∃ A B : ℝ × ℝ, 
     (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*a)^2 ∧
     (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
       (t*A.1 + (1-t)*B.1)^2/a^2 + (t*A.2 + (1-t)*B.2)^2/b^2 = 1 ∧
       ((1-t)*A.1 + t*B.1)^2/a^2 + ((1-t)*A.2 + t*B.2)^2/b^2 = 1 ∧
       t = 1/3)) →
  b^2 = 1/2 := by
sorry

end ellipse_hyperbola_property_l3061_306125


namespace arithmetic_sequence_problem_l3061_306114

/-- Given a positive arithmetic sequence {a_n} satisfying certain conditions, prove a_10 = 21 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  (∀ n, a n > 0) →  -- positive sequence
  a 1 + a 2 + a 3 = 15 →  -- sum condition
  (a 2 + 5)^2 = (a 1 + 2) * (a 3 + 13) →  -- geometric sequence condition
  a 10 = 21 := by
sorry

end arithmetic_sequence_problem_l3061_306114


namespace abs_sum_inequality_l3061_306173

theorem abs_sum_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := by
  sorry

end abs_sum_inequality_l3061_306173


namespace lindsey_squat_weight_l3061_306136

/-- The total weight Lindsey will squat given exercise bands and a dumbbell -/
theorem lindsey_squat_weight (num_bands : ℕ) (band_resistance : ℕ) (dumbbell_weight : ℕ) : 
  num_bands = 2 →
  band_resistance = 5 →
  dumbbell_weight = 10 →
  num_bands * band_resistance + dumbbell_weight = 20 := by
  sorry

end lindsey_squat_weight_l3061_306136


namespace national_bank_interest_rate_national_bank_interest_rate_is_five_percent_l3061_306167

theorem national_bank_interest_rate 
  (initial_investment : ℝ) 
  (additional_investment : ℝ) 
  (additional_rate : ℝ) 
  (total_income_rate : ℝ) : ℝ :=
  let total_investment := initial_investment + additional_investment
  let total_income := total_investment * total_income_rate
  let additional_income := additional_investment * additional_rate
  let national_bank_income := total_income - additional_income
  national_bank_income / initial_investment

#check national_bank_interest_rate 2400 600 0.1 0.06 -- Expected output: 0.05

theorem national_bank_interest_rate_is_five_percent :
  national_bank_interest_rate 2400 600 0.1 0.06 = 0.05 := by
  sorry

end national_bank_interest_rate_national_bank_interest_rate_is_five_percent_l3061_306167


namespace square_area_from_rectangle_l3061_306158

theorem square_area_from_rectangle (rectangle_area : ℝ) (rectangle_breadth : ℝ) : 
  rectangle_area = 100 →
  rectangle_breadth = 10 →
  ∃ (circle_radius : ℝ),
    (2 / 5 : ℝ) * circle_radius * rectangle_breadth = rectangle_area →
    circle_radius ^ 2 = 625 := by
  sorry

end square_area_from_rectangle_l3061_306158


namespace expression_factorization_l3061_306112

theorem expression_factorization (x : ℝ) :
  (20 * x^3 - 100 * x^2 + 30) - (5 * x^3 - 10 * x^2 + 3) = 3 * (5 * x^2 * (x - 6) + 9) := by
  sorry

end expression_factorization_l3061_306112


namespace necklaces_sold_l3061_306148

theorem necklaces_sold (total : ℕ) (given_away : ℕ) (left : ℕ) (sold : ℕ) : 
  total = 60 → given_away = 18 → left = 26 → sold = total - given_away - left → sold = 16 := by
  sorry

end necklaces_sold_l3061_306148


namespace ln_square_plus_ln_inequality_l3061_306181

theorem ln_square_plus_ln_inequality (x : ℝ) :
  (Real.log x)^2 + Real.log x < 0 ↔ Real.exp (-1) < x ∧ x < 1 := by
  sorry

end ln_square_plus_ln_inequality_l3061_306181


namespace can_measure_15_minutes_l3061_306171

/-- Represents an hourglass with a given duration in minutes -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of the timing system -/
structure TimingSystem where
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Defines the initial state of the timing system -/
def initialState : TimingSystem :=
  { hourglass1 := { duration := 7 },
    hourglass2 := { duration := 11 } }

/-- Represents a sequence of operations on the hourglasses -/
inductive Operation
  | FlipHourglass1
  | FlipHourglass2
  | Wait (minutes : ℕ)

/-- Applies a sequence of operations to the timing system -/
def applyOperations (state : TimingSystem) (ops : List Operation) : ℕ :=
  sorry

/-- Theorem stating that 15 minutes can be measured using the given hourglasses -/
theorem can_measure_15_minutes :
  ∃ (ops : List Operation), applyOperations initialState ops = 15 :=
sorry

end can_measure_15_minutes_l3061_306171


namespace lentil_dishes_count_l3061_306116

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_and_lentils : ℕ)
  (beans_and_seitan : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)

/-- The number of dishes including lentils in a vegan menu -/
def dishes_with_lentils (menu : VeganMenu) : ℕ :=
  menu.beans_and_lentils + menu.only_lentils

/-- Theorem stating the number of dishes including lentils in the given vegan menu -/
theorem lentil_dishes_count (menu : VeganMenu) 
  (h1 : menu.total_dishes = 10)
  (h2 : menu.beans_and_lentils = 2)
  (h3 : menu.beans_and_seitan = 2)
  (h4 : menu.only_beans = (menu.total_dishes - menu.beans_and_lentils - menu.beans_and_seitan) / 2)
  (h5 : menu.only_beans = 3 * menu.only_seitan)
  (h6 : menu.only_lentils = menu.total_dishes - menu.beans_and_lentils - menu.beans_and_seitan - menu.only_beans - menu.only_seitan) :
  dishes_with_lentils menu = 4 := by
  sorry


end lentil_dishes_count_l3061_306116


namespace linear_function_property_l3061_306139

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g where g(5) - g(1) = 16, prove that g(13) - g(1) = 48. -/
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) 
  (h_given : g 5 - g 1 = 16) : 
  g 13 - g 1 = 48 := by
  sorry


end linear_function_property_l3061_306139


namespace triangle_perimeter_l3061_306197

-- Define the lines
def line_through_origin (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}
def vertical_line (a : ℝ) := {(x, y) : ℝ × ℝ | x = a}
def sloped_line (m : ℝ) (b : ℝ) := {(x, y) : ℝ × ℝ | y = m * x + b}

-- Define the triangle
def right_triangle (m : ℝ) := 
  (0, 0) ∈ line_through_origin m ∧
  (1, -m) ∈ line_through_origin m ∧
  (1, -m) ∈ vertical_line 1 ∧
  (1, 1.5) ∈ sloped_line (1/2) 1 ∧
  (1, 1.5) ∈ vertical_line 1

-- Theorem statement
theorem triangle_perimeter :
  ∀ m : ℝ, right_triangle m → 
  (Real.sqrt ((1:ℝ)^2 + m^2) + Real.sqrt ((1:ℝ)^2 + (1.5 + m)^2) + 0.5) = 3 + Real.sqrt 5 :=
sorry

end triangle_perimeter_l3061_306197


namespace roots_custom_op_result_l3061_306143

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a * b - a - b

-- State the theorem
theorem roots_custom_op_result :
  ∀ x₁ x₂ : ℝ,
  (x₁^2 + x₁ - 1 = 0) →
  (x₂^2 + x₂ - 1 = 0) →
  customOp x₁ x₂ = 1 := by
sorry

end roots_custom_op_result_l3061_306143


namespace expression_simplification_l3061_306165

theorem expression_simplification :
  (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) / 
  ((2 * 3) * (3 * 4) * (4 * 5) * (5 * 6)) = 1/5 := by
  sorry

end expression_simplification_l3061_306165


namespace annual_increase_rate_proof_l3061_306185

/-- Proves that given an initial value of 32000 and a value of 40500 after two years,
    the annual increase rate is 0.125. -/
theorem annual_increase_rate_proof (initial_value final_value : ℝ) 
  (h1 : initial_value = 32000)
  (h2 : final_value = 40500)
  (h3 : final_value = initial_value * (1 + 0.125)^2) : 
  ∃ (r : ℝ), r = 0.125 ∧ final_value = initial_value * (1 + r)^2 := by
  sorry

end annual_increase_rate_proof_l3061_306185


namespace eight_even_painted_cubes_l3061_306100

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Represents a cube with a certain number of painted faces -/
structure Cube where
  painted_faces : Nat

/-- Function to determine if a number is even -/
def is_even (n : Nat) : Bool :=
  n % 2 = 0

/-- Function to calculate the number of cubes with even painted faces -/
def count_even_painted_cubes (block : Block) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that a 6x3x1 block has 8 cubes with even painted faces -/
theorem eight_even_painted_cubes (block : Block) 
  (h1 : block.length = 6) 
  (h2 : block.width = 3) 
  (h3 : block.height = 1) : 
  count_even_painted_cubes block = 8 := by
  sorry

end eight_even_painted_cubes_l3061_306100


namespace prob_sum_greater_than_8_is_5_18_l3061_306170

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to get a sum of 8 or less when rolling two dice -/
def sum_8_or_less : ℕ := 26

/-- The probability of rolling two dice and getting a sum greater than eight -/
def prob_sum_greater_than_8 : ℚ :=
  1 - (sum_8_or_less : ℚ) / total_outcomes

theorem prob_sum_greater_than_8_is_5_18 :
  prob_sum_greater_than_8 = 5 / 18 := by
  sorry

end prob_sum_greater_than_8_is_5_18_l3061_306170


namespace percentage_of_red_non_honda_cars_l3061_306159

theorem percentage_of_red_non_honda_cars
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (honda_red_percentage : ℚ)
  (total_red_percentage : ℚ)
  (h1 : total_cars = 900)
  (h2 : honda_cars = 500)
  (h3 : honda_red_percentage = 90 / 100)
  (h4 : total_red_percentage = 60 / 100)
  : (((total_red_percentage * total_cars) - (honda_red_percentage * honda_cars)) /
     (total_cars - honda_cars) : ℚ) = 225 / 1000 :=
by sorry

end percentage_of_red_non_honda_cars_l3061_306159


namespace albert_horses_l3061_306192

/-- Proves that Albert bought 4 horses given the conditions of the problem -/
theorem albert_horses : 
  ∀ (n : ℕ) (cow_price : ℕ),
  2000 * n + 9 * cow_price = 13400 →
  200 * n + 18 / 10 * cow_price = 1880 →
  n = 4 :=
by
  sorry

end albert_horses_l3061_306192


namespace multiplication_solution_l3061_306184

def possible_digits : Set Nat := {2, 4, 5, 6, 7, 8, 9}

def valid_multiplication (A B C D E : Nat) : Prop :=
  A ∈ possible_digits ∧ B ∈ possible_digits ∧ C ∈ possible_digits ∧ 
  D ∈ possible_digits ∧ E ∈ possible_digits ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E ∧
  E = 7 ∧
  (3 * (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1) = 
   100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1)

theorem multiplication_solution : 
  ∃ (A B C D E : Nat), valid_multiplication A B C D E ∧ A = 4 ∧ B = 2 ∧ C = 8 ∧ D = 5 := by
  sorry

end multiplication_solution_l3061_306184


namespace stating_first_alloy_amount_l3061_306157

/-- Represents an alloy with a specific ratio of lead to tin -/
structure Alloy where
  lead : ℝ
  tin : ℝ

/-- The first available alloy -/
def alloy1 : Alloy := { lead := 1, tin := 2 }

/-- The second available alloy -/
def alloy2 : Alloy := { lead := 2, tin := 3 }

/-- The desired new alloy -/
def newAlloy : Alloy := { lead := 4, tin := 7 }

/-- The total mass of the new alloy -/
def totalMass : ℝ := 22

/-- 
Theorem stating that 12 grams of the first alloy is needed to create the new alloy
with the desired properties
-/
theorem first_alloy_amount : 
  ∃ (x y : ℝ),
    x * (alloy1.lead + alloy1.tin) + y * (alloy2.lead + alloy2.tin) = totalMass ∧
    (x * alloy1.lead + y * alloy2.lead) / (x * alloy1.tin + y * alloy2.tin) = newAlloy.lead / newAlloy.tin ∧
    x * (alloy1.lead + alloy1.tin) = 12 := by
  sorry


end stating_first_alloy_amount_l3061_306157


namespace haley_growth_rate_l3061_306176

/-- Represents Haley's growth over time -/
structure Growth where
  initial_height : ℝ
  final_height : ℝ
  time_period : ℝ
  growth_rate : ℝ

/-- Theorem stating that given the initial conditions, Haley's growth rate is 3 inches per year -/
theorem haley_growth_rate (g : Growth) 
  (h1 : g.initial_height = 20)
  (h2 : g.final_height = 50)
  (h3 : g.time_period = 10)
  (h4 : g.growth_rate = (g.final_height - g.initial_height) / g.time_period) :
  g.growth_rate = 3 := by
  sorry

#check haley_growth_rate

end haley_growth_rate_l3061_306176


namespace hyperbola_eccentricity_k_range_l3061_306191

theorem hyperbola_eccentricity_k_range :
  ∀ (k : ℝ) (e : ℝ),
    (∀ (x y : ℝ), x^2 / 4 + y^2 / k = 1) →
    (1 < e ∧ e < 3) →
    (e = Real.sqrt (1 - k / 4)) →
    (-32 < k ∧ k < 0) :=
by sorry

end hyperbola_eccentricity_k_range_l3061_306191


namespace cistern_leak_time_l3061_306115

/-- Given a cistern with two pipes A and B, this theorem proves the time it takes for pipe B to leak out the full cistern. -/
theorem cistern_leak_time 
  (fill_time_A : ℝ) 
  (fill_time_both : ℝ) 
  (h1 : fill_time_A = 10) 
  (h2 : fill_time_both = 59.999999999999964) : 
  ∃ (leak_time_B : ℝ), leak_time_B = 12 ∧ 
  (1 / fill_time_A - 1 / leak_time_B = 1 / fill_time_both) := by
  sorry

end cistern_leak_time_l3061_306115


namespace tan_alpha_problem_l3061_306142

theorem tan_alpha_problem (α : Real) 
  (h1 : Real.tan (α + π/4) = -1/2) 
  (h2 : π/2 < α) 
  (h3 : α < π) : 
  (Real.sin (2*α) - 2*(Real.cos α)^2) / Real.sin (α - π/4) = -2*Real.sqrt 5/5 :=
by sorry

end tan_alpha_problem_l3061_306142


namespace intersection_when_a_zero_union_equals_A_iff_l3061_306131

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: When a = 0, A ∩ B = {x | -1 < x < 5}
theorem intersection_when_a_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: A ∪ B = A if and only if a ∈ (0, 1] ∪ [6, +∞)
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end intersection_when_a_zero_union_equals_A_iff_l3061_306131


namespace triangle_area_formulas_l3061_306182

/-- Given a triangle with area t, semiperimeter s, angles α, β, γ, and sides a, b, c,
    prove two formulas for the area. -/
theorem triangle_area_formulas (t s a b c α β γ : ℝ) 
  (h_area : t > 0)
  (h_semiperimeter : s > 0)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_semiperimeter_def : s = (a + b + c) / 2) :
  (t = s^2 * Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2)) ∧
  (t = (a*b*c/s) * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2)) := by
  sorry


end triangle_area_formulas_l3061_306182


namespace photo_arrangements_l3061_306150

/-- Represents the number of people in each category -/
structure People where
  teacher : Nat
  boys : Nat
  girls : Nat

/-- The total number of people -/
def total_people (p : People) : Nat :=
  p.teacher + p.boys + p.girls

/-- The number of arrangements for the given conditions -/
def arrangements (p : People) : Nat :=
  -- We'll define this function without implementation
  sorry

/-- Theorem stating the number of arrangements for the given problem -/
theorem photo_arrangements :
  let p := People.mk 1 2 2
  arrangements p = 24 := by
  sorry

end photo_arrangements_l3061_306150


namespace sqrt_198_between_14_and_15_l3061_306190

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end sqrt_198_between_14_and_15_l3061_306190


namespace function_not_satisfying_differential_equation_l3061_306164

open Real

theorem function_not_satisfying_differential_equation :
  ¬∃ y : ℝ → ℝ, ∀ x : ℝ,
    (y x = (x + 1) * (Real.exp (x^2))) ∧
    (deriv y x - 2 * x * y x = 2 * x * (Real.exp (x^2))) :=
sorry

end function_not_satisfying_differential_equation_l3061_306164


namespace greatest_of_three_consecutive_integers_l3061_306189

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 36) : 
  max x (max (x + 1) (x + 2)) = 13 := by
  sorry

end greatest_of_three_consecutive_integers_l3061_306189


namespace expression_evaluation_l3061_306107

theorem expression_evaluation :
  Real.sqrt ((16^6 + 2^18) / (16^3 + 2^21)) = (8 * Real.sqrt 65) / Real.sqrt 513 := by
  sorry

end expression_evaluation_l3061_306107


namespace fraction_sum_equals_negative_two_l3061_306196

theorem fraction_sum_equals_negative_two (a b : ℝ) (h1 : a + b = 0) (h2 : a * b ≠ 0) :
  b / a + a / b = -2 := by sorry

end fraction_sum_equals_negative_two_l3061_306196


namespace polynomial_symmetry_representation_l3061_306155

theorem polynomial_symmetry_representation (p : ℝ → ℝ) (a : ℝ) 
  (h_symmetry : ∀ x, p x = p (a - x)) :
  ∃ h : ℝ → ℝ, ∀ x, p x = h ((x - a / 2) ^ 2) := by
  sorry

end polynomial_symmetry_representation_l3061_306155


namespace lcm_gcd_product_equals_number_product_l3061_306126

theorem lcm_gcd_product_equals_number_product : 
  let a := 24
  let b := 36
  Nat.lcm a b * Nat.gcd a b = a * b :=
by sorry

end lcm_gcd_product_equals_number_product_l3061_306126


namespace three_numbers_problem_l3061_306163

theorem three_numbers_problem :
  ∃ (a b c : ℕ),
    (Nat.gcd a b = 8) ∧
    (Nat.gcd b c = 2) ∧
    (Nat.gcd a c = 6) ∧
    (Nat.lcm (Nat.lcm a b) c = 1680) ∧
    (max a (max b c) > 100) ∧
    (max a (max b c) ≤ 200) ∧
    ((∃ n : ℕ, a = n^4) ∨ (∃ n : ℕ, b = n^4) ∨ (∃ n : ℕ, c = n^4)) ∧
    ((a = 120 ∧ b = 16 ∧ c = 42) ∨ (a = 168 ∧ b = 16 ∧ c = 30)) :=
by
  sorry

#check three_numbers_problem

end three_numbers_problem_l3061_306163


namespace rhombus_perimeter_l3061_306195

/-- A rhombus with given diagonal lengths has a specific perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 40 := by
  sorry

end rhombus_perimeter_l3061_306195


namespace exists_permutation_divisible_by_seven_l3061_306118

def digits : List Nat := [1, 3, 7, 9]

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ ∀ x, l1.count x = l2.count x

def list_to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

theorem exists_permutation_divisible_by_seven :
  ∃ perm : List Nat, is_permutation digits perm ∧ 
    (list_to_number perm) % 7 = 0 := by
  sorry

end exists_permutation_divisible_by_seven_l3061_306118


namespace nes_sale_price_l3061_306101

theorem nes_sale_price 
  (snes_value : ℝ)
  (trade_in_percentage : ℝ)
  (additional_cash : ℝ)
  (change : ℝ)
  (game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : additional_cash = 80)
  (h4 : change = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + additional_cash - change - game_value = 160 :=
by
  sorry

#check nes_sale_price

end nes_sale_price_l3061_306101


namespace new_person_weight_specific_new_person_weight_l3061_306102

/-- Given a group of people, calculate the weight of a new person who causes the average weight to increase when replacing another person. -/
theorem new_person_weight (initial_size : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  let total_increase := initial_size * avg_increase
  replaced_weight + total_increase

/-- Prove that for the given conditions, the weight of the new person is 61.3 kg. -/
theorem specific_new_person_weight :
  new_person_weight 12 1.3 45.7 = 61.3 := by sorry

end new_person_weight_specific_new_person_weight_l3061_306102


namespace root_zero_iff_m_neg_three_l3061_306121

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + (3*m - 1) * x + m^2 - 9

/-- Theorem: One root of the quadratic equation is 0 iff m = -3 -/
theorem root_zero_iff_m_neg_three :
  ∀ m : ℝ, (∃ x : ℝ, quadratic_equation m x = 0 ∧ x = 0) ↔ m = -3 := by sorry

end root_zero_iff_m_neg_three_l3061_306121


namespace dress_discount_percentage_l3061_306151

theorem dress_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  (d * (100 - x) / 100) * 0.7 = 0.455 * d ↔ x = 35 := by
sorry

end dress_discount_percentage_l3061_306151


namespace unique_division_problem_l3061_306105

theorem unique_division_problem :
  ∃! (dividend divisor : ℕ),
    dividend ≥ 1000000 ∧ dividend < 2000000 ∧
    divisor ≥ 300 ∧ divisor < 400 ∧
    (dividend / divisor : ℚ) = 5243 / 1000 ∧
    dividend % divisor = 0 ∧
    ∃ (r1 r2 r3 : ℕ),
      r1 % 10 = 9 ∧
      r2 % 10 = 6 ∧
      r3 % 10 = 3 ∧
      r1 < divisor ∧
      r2 < divisor ∧
      r3 < divisor ∧
      dividend = 1000000 + (dividend / 100000 % 10) * 100000 + 50000 + (dividend % 10000) :=
by sorry

end unique_division_problem_l3061_306105


namespace unique_geometric_sequence_value_l3061_306152

/-- Two geometric sequences with specific conditions -/
structure GeometricSequences (a : ℝ) :=
  (a_seq : ℕ → ℝ)
  (b_seq : ℕ → ℝ)
  (a_positive : a > 0)
  (a_first : a_seq 1 = a)
  (b_minus_a_1 : b_seq 1 - a_seq 1 = 1)
  (b_minus_a_2 : b_seq 2 - a_seq 2 = 2)
  (b_minus_a_3 : b_seq 3 - a_seq 3 = 3)
  (a_geometric : ∀ n : ℕ, a_seq (n + 1) / a_seq n = a_seq 2 / a_seq 1)
  (b_geometric : ∀ n : ℕ, b_seq (n + 1) / b_seq n = b_seq 2 / b_seq 1)

/-- If the a_seq is unique, then a = 1/3 -/
theorem unique_geometric_sequence_value (a : ℝ) (h : GeometricSequences a) 
  (h_unique : ∃! q : ℝ, ∀ n : ℕ, h.a_seq (n + 1) = h.a_seq n * q) : 
  a = 1/3 := by
sorry

end unique_geometric_sequence_value_l3061_306152


namespace smallest_positive_integer_e_l3061_306162

theorem smallest_positive_integer_e (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -4 ∨ x = 6 ∨ x = 10 ∨ x = -1/2) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ 
      x = -4 ∨ x = 6 ∨ x = 10 ∨ x = -1/2) → 
    e ≤ e') →
  e = 200 := by
sorry

end smallest_positive_integer_e_l3061_306162


namespace nanometer_scientific_notation_l3061_306104

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem nanometer_scientific_notation :
  scientific_notation 0.000000022 = (2.2, -8) :=
sorry

end nanometer_scientific_notation_l3061_306104


namespace equation_roots_l3061_306140

theorem equation_roots : 
  let f (x : ℝ) := 20 / (x^2 - 9) - 3 / (x + 3) - 2
  let root1 := (-3 + Real.sqrt 385) / 4
  let root2 := (-3 - Real.sqrt 385) / 4
  f root1 = 0 ∧ f root2 = 0 := by
  sorry

end equation_roots_l3061_306140


namespace martha_coffee_spending_cut_l3061_306135

def coffee_spending_cut_percentage (latte_cost : ℚ) (iced_coffee_cost : ℚ)
  (lattes_per_week : ℕ) (iced_coffees_per_week : ℕ) (weeks_per_year : ℕ)
  (savings_goal : ℚ) : ℚ :=
  let weekly_spending := latte_cost * lattes_per_week + iced_coffee_cost * iced_coffees_per_week
  let annual_spending := weekly_spending * weeks_per_year
  (savings_goal / annual_spending) * 100

theorem martha_coffee_spending_cut (latte_cost : ℚ) (iced_coffee_cost : ℚ)
  (lattes_per_week : ℕ) (iced_coffees_per_week : ℕ) (weeks_per_year : ℕ)
  (savings_goal : ℚ) :
  latte_cost = 4 →
  iced_coffee_cost = 2 →
  lattes_per_week = 5 →
  iced_coffees_per_week = 3 →
  weeks_per_year = 52 →
  savings_goal = 338 →
  coffee_spending_cut_percentage latte_cost iced_coffee_cost lattes_per_week
    iced_coffees_per_week weeks_per_year savings_goal = 25 :=
by sorry

end martha_coffee_spending_cut_l3061_306135


namespace max_product_953_l3061_306175

/-- A type representing a valid digit for our problem -/
inductive Digit
  | three
  | five
  | six
  | eight
  | nine

/-- A function to convert our Digit type to a natural number -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8
  | Digit.nine => 9

/-- A type representing a valid combination of digits -/
structure DigitCombination where
  d1 : Digit
  d2 : Digit
  d3 : Digit
  d4 : Digit
  d5 : Digit
  all_different : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
                  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧
                  d3 ≠ d4 ∧ d3 ≠ d5 ∧
                  d4 ≠ d5

/-- Function to calculate the product of a three-digit and two-digit number from a DigitCombination -/
def calculate_product (dc : DigitCombination) : ℕ :=
  (100 * digit_to_nat dc.d1 + 10 * digit_to_nat dc.d2 + digit_to_nat dc.d3) *
  (10 * digit_to_nat dc.d4 + digit_to_nat dc.d5)

/-- The main theorem stating that 953 yields the maximum product -/
theorem max_product_953 :
  ∀ dc : DigitCombination,
  calculate_product dc ≤ calculate_product
    { d1 := Digit.nine, d2 := Digit.five, d3 := Digit.three,
      d4 := Digit.eight, d5 := Digit.six,
      all_different := by simp } :=
sorry

end max_product_953_l3061_306175


namespace lcd_of_fractions_l3061_306134

theorem lcd_of_fractions (a b c d e f : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) (hd : d = 6) (he : e = 8) (hf : f = 9) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e f)))) = 360 := by
  sorry

end lcd_of_fractions_l3061_306134


namespace comparison_of_powers_l3061_306144

theorem comparison_of_powers (a b c : ℝ) : 
  a = 10 ∧ b = -49 ∧ c = -50 → 
  a^b - 2 * a^c = 8 * a^c := by
  sorry

end comparison_of_powers_l3061_306144


namespace printing_speed_proof_l3061_306130

/-- Mike's initial printing speed in pamphlets per hour -/
def initial_speed : ℕ := 600

/-- Total number of pamphlets printed -/
def total_pamphlets : ℕ := 9400

/-- Mike's initial printing time in hours -/
def mike_initial_time : ℕ := 9

/-- Mike's reduced speed printing time in hours -/
def mike_reduced_time : ℕ := 2

/-- Leo's printing time in hours -/
def leo_time : ℕ := 3

theorem printing_speed_proof :
  initial_speed * mike_initial_time + 
  (initial_speed / 3) * mike_reduced_time + 
  (2 * initial_speed) * leo_time = total_pamphlets :=
by sorry

end printing_speed_proof_l3061_306130


namespace village_population_equality_l3061_306178

-- Define the initial populations and known rate of decrease
def population_X : ℕ := 70000
def population_Y : ℕ := 42000
def decrease_rate_X : ℕ := 1200
def years : ℕ := 14

-- Define the unknown rate of increase for Village Y
def increase_rate_Y : ℕ := sorry

-- Theorem statement
theorem village_population_equality :
  population_X - years * decrease_rate_X = population_Y + years * increase_rate_Y ∧
  increase_rate_Y = 800 := by sorry

end village_population_equality_l3061_306178


namespace a_fourth_zero_implies_a_squared_zero_l3061_306180

theorem a_fourth_zero_implies_a_squared_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
  sorry

end a_fourth_zero_implies_a_squared_zero_l3061_306180


namespace complex_equation_solution_l3061_306133

theorem complex_equation_solution (m A B : ℝ) : 
  (2 - m * Complex.I) / (1 + 2 * Complex.I) = Complex.mk A B →
  A + B = 0 →
  m = -2/3 := by sorry

end complex_equation_solution_l3061_306133


namespace abcd_sum_l3061_306145

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -2)
  (eq3 : a + c + d = 5)
  (eq4 : b + c + d = 4) :
  a * b + c * d = 26 / 9 := by
sorry

end abcd_sum_l3061_306145


namespace a_range_l3061_306161

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) → 
  a > -1 ∧ a < 1 := by
sorry

end a_range_l3061_306161


namespace quinn_caught_four_frogs_l3061_306160

-- Define the number of frogs caught by each person
def alster_frogs : ℕ := 2
def bret_frogs : ℕ := 12

-- Define Quinn's frogs in terms of Alster's
def quinn_frogs : ℕ := alster_frogs

-- Define the relationship between Bret's and Quinn's frogs
axiom bret_quinn_relation : bret_frogs = 3 * quinn_frogs

theorem quinn_caught_four_frogs : quinn_frogs = 4 := by
  sorry

end quinn_caught_four_frogs_l3061_306160


namespace sum_of_squares_theorem_l3061_306117

theorem sum_of_squares_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  (h_sum : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 3 := by
  sorry

end sum_of_squares_theorem_l3061_306117


namespace updated_mean_l3061_306149

theorem updated_mean (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 9 →
  (n * original_mean - n * decrement) / n = 191 := by
  sorry

end updated_mean_l3061_306149


namespace inequality_proof_l3061_306141

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y ≤ (y^2 / x) + (x^2 / y) := by
  sorry

end inequality_proof_l3061_306141


namespace initial_pears_eq_sum_l3061_306183

/-- The number of pears Sara picked initially -/
def initial_pears : ℕ := sorry

/-- The number of apples Sara picked -/
def apples : ℕ := 27

/-- The number of pears Sara gave to Dan -/
def pears_given : ℕ := 28

/-- The number of pears Sara has left -/
def pears_left : ℕ := 7

/-- Theorem stating that the initial number of pears is equal to the sum of pears given and pears left -/
theorem initial_pears_eq_sum : initial_pears = pears_given + pears_left := by sorry

end initial_pears_eq_sum_l3061_306183


namespace dealer_profit_selling_price_percentage_l3061_306132

theorem dealer_profit (list_price : ℝ) (purchase_price selling_price : ℝ) : 
  purchase_price = 3/4 * list_price →
  selling_price = 2 * purchase_price →
  selling_price = 3/2 * list_price :=
by sorry

theorem selling_price_percentage (list_price : ℝ) (selling_price : ℝ) :
  selling_price = 3/2 * list_price →
  (selling_price - list_price) / list_price = 1/2 :=
by sorry

end dealer_profit_selling_price_percentage_l3061_306132


namespace is_centre_of_hyperbola_l3061_306137

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0

/-- The centre of the hyperbola -/
def hyperbola_centre : ℝ × ℝ := (1, 2)

/-- Theorem stating that the given point is the centre of the hyperbola -/
theorem is_centre_of_hyperbola :
  let (h, k) := hyperbola_centre
  ∀ (a b : ℝ), hyperbola_equation (h + a) (k + b) ↔ hyperbola_equation (h - a) (k - b) :=
by sorry

end is_centre_of_hyperbola_l3061_306137


namespace lollipop_collection_time_l3061_306110

theorem lollipop_collection_time (total_sticks : ℕ) (visits_per_week : ℕ) (completion_percentage : ℚ) : 
  total_sticks = 400 →
  visits_per_week = 3 →
  completion_percentage = 3/5 →
  (total_sticks * completion_percentage / visits_per_week : ℚ) = 80 := by
sorry

end lollipop_collection_time_l3061_306110


namespace power_expression_equality_l3061_306103

theorem power_expression_equality : (3^5 / 3^2) * 2^7 = 3456 := by
  sorry

end power_expression_equality_l3061_306103


namespace second_metal_cost_l3061_306166

/-- Given two metals mixed in equal proportions, prove the cost of the second metal
    when the cost of the first metal and the resulting alloy are known. -/
theorem second_metal_cost (cost_first : ℝ) (cost_alloy : ℝ) : 
  cost_first = 68 → cost_alloy = 82 → 2 * cost_alloy - cost_first = 96 := by
  sorry

end second_metal_cost_l3061_306166


namespace eddy_spider_plant_production_l3061_306186

/-- A spider plant that produces baby plants -/
structure SpiderPlant where
  /-- Number of baby plants produced each time -/
  baby_per_time : ℕ
  /-- Total number of baby plants produced -/
  total_babies : ℕ
  /-- Number of years -/
  years : ℕ

/-- The number of times per year a spider plant produces baby plants -/
def times_per_year (plant : SpiderPlant) : ℚ :=
  (plant.total_babies : ℚ) / (plant.years * plant.baby_per_time : ℚ)

/-- Theorem stating that Eddy's spider plant produces baby plants 2 times per year -/
theorem eddy_spider_plant_production :
  ∃ (plant : SpiderPlant),
    plant.baby_per_time = 2 ∧
    plant.total_babies = 16 ∧
    plant.years = 4 ∧
    times_per_year plant = 2 := by
  sorry

end eddy_spider_plant_production_l3061_306186


namespace ellipse_parabola_intersection_range_l3061_306154

/-- Given an ellipse and a parabola with a common point, this theorem proves the range of parameter a. -/
theorem ellipse_parabola_intersection_range :
  ∀ (a x y : ℝ),
  (x^2 + 4*(y - a)^2 = 4) →  -- Ellipse equation
  (x^2 = 2*y) →              -- Parabola equation
  (-1 ≤ a ∧ a ≤ 17/8) :=     -- Range of a
by sorry

end ellipse_parabola_intersection_range_l3061_306154


namespace sweater_price_proof_l3061_306109

/-- Price of a T-shirt in dollars -/
def t_shirt_price : ℝ := 8

/-- Price of a jacket before discount in dollars -/
def jacket_price : ℝ := 80

/-- Discount rate for jackets -/
def jacket_discount : ℝ := 0.1

/-- Sales tax rate -/
def sales_tax : ℝ := 0.05

/-- Number of T-shirts purchased -/
def num_tshirts : ℕ := 6

/-- Number of sweaters purchased -/
def num_sweaters : ℕ := 4

/-- Number of jackets purchased -/
def num_jackets : ℕ := 5

/-- Total cost including tax in dollars -/
def total_cost : ℝ := 504

/-- Price of a sweater in dollars -/
def sweater_price : ℝ := 18

theorem sweater_price_proof :
  (num_tshirts * t_shirt_price +
   num_sweaters * sweater_price +
   num_jackets * jacket_price * (1 - jacket_discount)) *
  (1 + sales_tax) = total_cost := by
  sorry

end sweater_price_proof_l3061_306109


namespace pages_of_maps_skipped_l3061_306128

theorem pages_of_maps_skipped (total_pages read_pages pages_left : ℕ) 
  (h1 : total_pages = 372)
  (h2 : read_pages = 125)
  (h3 : pages_left = 231) :
  total_pages - (read_pages + pages_left) = 16 := by
  sorry

end pages_of_maps_skipped_l3061_306128


namespace negation_and_converse_l3061_306106

def last_digit (n : ℤ) : ℕ := (n % 10).natAbs

def divisible_by_five (n : ℤ) : Prop := n % 5 = 0

def statement (n : ℤ) : Prop :=
  (last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n

theorem negation_and_converse :
  (∀ n : ℤ, ¬statement n ↔ (last_digit n = 0 ∨ last_digit n = 5) ∧ ¬(divisible_by_five n)) ∧
  (∀ n : ℤ, (¬(last_digit n = 0 ∨ last_digit n = 5) → ¬(divisible_by_five n)) →
    ((last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n)) :=
sorry

end negation_and_converse_l3061_306106


namespace lateral_surface_area_square_pyramid_l3061_306199

/-- Lateral surface area of a regular square pyramid -/
theorem lateral_surface_area_square_pyramid 
  (base_edge : ℝ) 
  (height : ℝ) 
  (h : base_edge = 2 * Real.sqrt 3) 
  (h' : height = 1) : 
  4 * (1/2 * base_edge * Real.sqrt (base_edge^2/4 + height^2)) = 8 * Real.sqrt 3 :=
by sorry

end lateral_surface_area_square_pyramid_l3061_306199


namespace remainder_sum_product_l3061_306146

theorem remainder_sum_product (X Y Z E S T U s t q : ℕ) 
  (hX : X > Y) (hY : Y > Z)
  (hS : X % E = S) (hT : Y % E = T) (hU : Z % E = U)
  (hs : (X * Y * Z) % E = s) (ht : (S * T * U) % E = t)
  (hq : (X * Y * Z + S * T * U) % E = q) :
  q = (2 * s) % E :=
sorry

end remainder_sum_product_l3061_306146


namespace book_length_problem_l3061_306187

/-- Represents the problem of determining book lengths based on reading rates and times -/
theorem book_length_problem (book1_pages book2_pages_read : ℕ) 
  (book1_rate book2_rate : ℕ) (h1 : book1_rate = 40) (h2 : book2_rate = 60) :
  (2 * book1_pages / 3 = book1_pages / 3 + 30) →
  (2 * book1_pages / (3 * book1_rate) = book2_pages_read / book2_rate) →
  book1_pages = 90 ∧ book2_pages_read = 45 := by
  sorry

#check book_length_problem

end book_length_problem_l3061_306187


namespace hyperbola_m_range_l3061_306194

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (m^2 - 4) - y^2 / (m + 1) = 1

-- Define the condition that foci are on y-axis
def foci_on_y_axis (m : ℝ) : Prop :=
  -(m + 1) > 0 ∧ 4 - m^2 > 0

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, hyperbola_equation x y m) ∧ foci_on_y_axis m → 
  m > -2 ∧ m < -1 :=
sorry

end hyperbola_m_range_l3061_306194


namespace existence_of_decreasing_lcm_sequence_l3061_306129

theorem existence_of_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
by sorry

end existence_of_decreasing_lcm_sequence_l3061_306129


namespace parabola_coefficients_from_vertex_and_point_l3061_306179

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_coefficients_from_vertex_and_point
  (p : Parabola)
  (vertex_x vertex_y : ℝ)
  (point_x point_y : ℝ)
  (h_vertex : p.y vertex_x = vertex_y)
  (h_point : p.y point_x = point_y)
  (h_vertex_x : vertex_x = 4)
  (h_vertex_y : vertex_y = 3)
  (h_point_x : point_x = 2)
  (h_point_y : point_y = 1) :
  p.a = -1/2 ∧ p.b = 4 ∧ p.c = -5 := by
  sorry

end parabola_coefficients_from_vertex_and_point_l3061_306179


namespace odd_function_property_l3061_306111

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = x * (1 + x)) : 
  ∀ x > 0, f x = x * (1 - x) := by
sorry

end odd_function_property_l3061_306111


namespace base_2_representation_of_56_l3061_306120

/-- Represents a natural number in base 2 as a list of bits (least significant bit first) -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

/-- Converts a list of bits (least significant bit first) to a natural number -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem base_2_representation_of_56 :
  toBinary 56 = [false, false, false, true, true, true] := by
  sorry

end base_2_representation_of_56_l3061_306120


namespace apple_cost_price_l3061_306193

/-- The cost price of an apple, given its selling price and loss ratio. -/
def cost_price (selling_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  selling_price / (1 - loss_ratio)

/-- Theorem: The cost price of an apple is 20.4 when sold for 17 with a 1/6 loss. -/
theorem apple_cost_price :
  cost_price 17 (1/6) = 20.4 := by
sorry

end apple_cost_price_l3061_306193


namespace four_students_three_communities_l3061_306168

/-- The number of ways to distribute students among communities -/
def distribute_students (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for 4 students and 3 communities -/
theorem four_students_three_communities :
  distribute_students 4 3 = 36 := by sorry

end four_students_three_communities_l3061_306168


namespace choose_three_from_ten_l3061_306138

theorem choose_three_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end choose_three_from_ten_l3061_306138


namespace other_root_of_complex_quadratic_l3061_306119

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -39 - 52*I ∧ z = 5 - 7*I → (-z = -5 + 7*I ∧ (-z)^2 = -39 - 52*I) := by
  sorry

end other_root_of_complex_quadratic_l3061_306119


namespace total_weight_is_540_l3061_306113

def back_squat_initial : ℝ := 200
def back_squat_increase : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9
def number_of_triples : ℕ := 3

def calculate_total_weight : ℝ :=
  let back_squat_new := back_squat_initial + back_squat_increase
  let front_squat := back_squat_new * front_squat_ratio
  let triple_weight := front_squat * triple_ratio
  triple_weight * number_of_triples

theorem total_weight_is_540 :
  calculate_total_weight = 540 := by sorry

end total_weight_is_540_l3061_306113


namespace basketball_free_throws_l3061_306188

theorem basketball_free_throws (total_score : ℕ) (three_point_shots : ℕ) 
  (h1 : total_score = 79)
  (h2 : 3 * three_point_shots = 2 * (total_score - 3 * three_point_shots - free_throws) / 2)
  (h3 : free_throws = 2 * (total_score - 3 * three_point_shots - free_throws) / 2)
  (h4 : three_point_shots = 4) :
  free_throws = 12 :=
by
  sorry

#check basketball_free_throws

end basketball_free_throws_l3061_306188


namespace rectangle_shorter_side_length_l3061_306108

/-- Given a rectangle, square, and equilateral triangle with the same perimeter,
    if the square's side length is 9 cm, the rectangle's shorter side is 6 cm. -/
theorem rectangle_shorter_side_length
  (rectangle : Real × Real)
  (square : Real)
  (equilateral_triangle : Real)
  (h1 : 2 * (rectangle.1 + rectangle.2) = 4 * square)
  (h2 : 2 * (rectangle.1 + rectangle.2) = 3 * equilateral_triangle)
  (h3 : square = 9) :
  min rectangle.1 rectangle.2 = 6 := by
  sorry

end rectangle_shorter_side_length_l3061_306108


namespace imaginary_part_of_z_l3061_306177

theorem imaginary_part_of_z (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l3061_306177


namespace prime_pair_equation_solution_l3061_306156

theorem prime_pair_equation_solution :
  ∀ p q : ℕ,
  Prime p → Prime q →
  (∃ m : ℕ+, (p * q : ℚ) / (p + q) = (m.val^2 + 6 : ℚ) / (m.val + 1)) →
  (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := by
sorry

end prime_pair_equation_solution_l3061_306156
