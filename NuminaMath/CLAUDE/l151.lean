import Mathlib

namespace intersection_of_A_and_B_l151_15103

def set_A : Set ℝ := {x | (x - 1) / (x + 3) < 0}
def set_B : Set ℝ := {x | abs x < 2}

theorem intersection_of_A_and_B : 
  set_A ∩ set_B = {x | -2 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l151_15103


namespace quadratic_function_properties_l151_15141

/-- Given a quadratic function f(x) = ax^2 - 2x + c where the solution set of f(x) > 0 is {x | x ≠ 1/a},
    this theorem states that the minimum value of f(2) is 0 and when f(2) is minimum,
    the maximum value of m that satisfies f(x) + 4 ≥ m(x-2) for all x > 2 is 2√2. -/
theorem quadratic_function_properties (a c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 - 2*x + c)
  (h2 : ∀ x, f x > 0 ↔ x ≠ 1/a) :
  (∃ (f_min : ℝ → ℝ), (∀ x, f_min x = (1/2) * x^2 - 2*x + 2) ∧ 
   f_min 2 = 0 ∧ 
   (∀ m : ℝ, (∀ x > 2, f_min x + 4 ≥ m * (x - 2)) ↔ m ≤ 2 * Real.sqrt 2)) := by
  sorry

end quadratic_function_properties_l151_15141


namespace expression_evaluation_l151_15140

theorem expression_evaluation : 
  let a : ℚ := 7
  let b : ℚ := 11
  let c : ℚ := 13
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end expression_evaluation_l151_15140


namespace profit_percentage_calculation_l151_15125

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 240 →
  selling_price = 288 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end profit_percentage_calculation_l151_15125


namespace carlos_gummy_worms_l151_15154

/-- The number of gummy worms remaining after eating half for a given number of days -/
def gummy_worms_remaining (initial : ℕ) (days : ℕ) : ℕ :=
  initial / (2 ^ days)

/-- Theorem stating that Carlos has 4 gummy worms left after 4 days -/
theorem carlos_gummy_worms :
  gummy_worms_remaining 64 4 = 4 := by
  sorry

end carlos_gummy_worms_l151_15154


namespace percentage_less_l151_15178

theorem percentage_less (q y w z : ℝ) : 
  w = 0.6 * q →
  z = 0.54 * y →
  z = 1.5 * w →
  q = 0.6 * y :=
by sorry

end percentage_less_l151_15178


namespace arccos_one_over_sqrt_two_l151_15122

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l151_15122


namespace prime_power_theorem_l151_15119

theorem prime_power_theorem (p q : ℕ) : 
  p > 1 → q > 1 → 
  Nat.Prime p → Nat.Prime q → 
  Nat.Prime (7 * p + q) → Nat.Prime (p * q + 11) → 
  p^q = 8 ∨ p^q = 9 := by
sorry

end prime_power_theorem_l151_15119


namespace all_X_composite_except_101_l151_15104

def X (n : ℕ) : ℕ := 
  (10^(2*n + 1) - 1) / 9

theorem all_X_composite_except_101 (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ X n = a * b :=
sorry

end all_X_composite_except_101_l151_15104


namespace bus_problem_solution_l151_15121

def bus_problem (initial : ℕ) 
  (stop1_off stop1_on : ℕ) 
  (stop2_off stop2_on : ℕ) 
  (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off + stop1_on - stop2_off + stop2_on - stop3_off + stop3_on

theorem bus_problem_solution : 
  bus_problem 10 3 2 1 4 2 3 = 13 := by
  sorry

end bus_problem_solution_l151_15121


namespace strawberry_weight_sum_l151_15187

/-- The weight of Marco's strawberries in pounds -/
def marco_strawberries : ℕ := 3

/-- The weight of Marco's dad's strawberries in pounds -/
def dad_strawberries : ℕ := 17

/-- The total weight of Marco's and his dad's strawberries -/
def total_strawberries : ℕ := marco_strawberries + dad_strawberries

theorem strawberry_weight_sum :
  total_strawberries = 20 := by sorry

end strawberry_weight_sum_l151_15187


namespace log_inequality_l151_15151

theorem log_inequality (h1 : 4^5 < 7^4) (h2 : 11^4 < 7^5) : 
  Real.log 11 / Real.log 7 < Real.log 243 / Real.log 81 ∧ 
  Real.log 243 / Real.log 81 < Real.log 7 / Real.log 4 := by
  sorry

end log_inequality_l151_15151


namespace franks_daily_cookie_consumption_l151_15190

/-- Proves that Frank eats 1 cookie each day given the conditions of the problem -/
theorem franks_daily_cookie_consumption :
  let days : ℕ := 6
  let trays_per_day : ℕ := 2
  let cookies_per_tray : ℕ := 12
  let ted_cookies : ℕ := 4
  let cookies_left : ℕ := 134
  let total_baked : ℕ := days * trays_per_day * cookies_per_tray
  let franks_total_consumption : ℕ := total_baked - ted_cookies - cookies_left
  franks_total_consumption / days = 1 := by
  sorry

end franks_daily_cookie_consumption_l151_15190


namespace parallelepiped_surface_area_l151_15139

/-- Represents a parallelepiped composed of white and black unit cubes -/
structure Parallelepiped where
  white_cubes : ℕ
  black_cubes : ℕ
  length : ℕ
  width : ℕ
  height : ℕ

/-- Conditions for the parallelepiped -/
def valid_parallelepiped (p : Parallelepiped) : Prop :=
  p.white_cubes > 0 ∧
  p.black_cubes = p.white_cubes * 53 / 52 ∧
  p.length > 1 ∧ p.width > 1 ∧ p.height > 1 ∧
  p.length * p.width * p.height = p.white_cubes + p.black_cubes

/-- Surface area of a parallelepiped -/
def surface_area (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.width * p.height + p.height * p.length)

/-- Theorem stating the surface area of the parallelepiped is 142 -/
theorem parallelepiped_surface_area (p : Parallelepiped) 
  (h : valid_parallelepiped p) : surface_area p = 142 := by
  sorry

end parallelepiped_surface_area_l151_15139


namespace repeating_decimal_sum_l151_15113

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d →
  c + d = 11 := by
  sorry

end repeating_decimal_sum_l151_15113


namespace gasoline_distribution_impossible_l151_15129

theorem gasoline_distribution_impossible : 
  ¬ ∃ (A B C : ℝ), 
    A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧
    A + B + C = 50 ∧ 
    A = B + 10 ∧ 
    C + 26 = B :=
by sorry

end gasoline_distribution_impossible_l151_15129


namespace magnitude_of_w_l151_15198

theorem magnitude_of_w (w : ℂ) (h : w^2 = -7 + 24*I) : Complex.abs w = 5 := by
  sorry

end magnitude_of_w_l151_15198


namespace same_club_probability_l151_15153

theorem same_club_probability (n : ℕ) (h : n = 8) :
  let p := 1 / n
  (n : ℝ) * p * p = 1 / n :=
by
  sorry

end same_club_probability_l151_15153


namespace soccer_players_count_l151_15136

def total_students : ℕ := 400
def sports_proportion : ℚ := 52 / 100
def soccer_proportion : ℚ := 125 / 1000

theorem soccer_players_count :
  ⌊(total_students : ℚ) * sports_proportion * soccer_proportion⌋ = 26 := by
  sorry

end soccer_players_count_l151_15136


namespace zoe_has_16_crayons_l151_15148

/-- The number of crayons in each student's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ
  xavier : ℕ
  yasmine : ℕ
  zoe : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = 2 * boxes.gilbert ∧
  boxes.gilbert = 4 * boxes.judah ∧
  2 * boxes.gilbert = boxes.xavier ∧
  boxes.xavier = boxes.yasmine + 16 ∧
  boxes.yasmine = 3 * boxes.zoe ∧
  boxes.karen = 128

/-- The theorem to be proved -/
theorem zoe_has_16_crayons (boxes : CrayonBoxes) 
  (h : satisfies_conditions boxes) : boxes.zoe = 16 := by
  sorry

end zoe_has_16_crayons_l151_15148


namespace function_value_l151_15100

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (1/4 - a)*x + 2*a

theorem function_value (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a = 1/2 :=
by sorry

end function_value_l151_15100


namespace angle_z_is_100_l151_15115

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.X > 0 ∧ t.Y > 0 ∧ t.Z > 0 ∧ t.X + t.Y + t.Z = 180

-- Define the given conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.X + t.Y = 80 ∧ t.X = 2 * t.Y

-- Theorem statement
theorem angle_z_is_100 (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_conditions t) : 
  t.Z = 100 := by
  sorry

end angle_z_is_100_l151_15115


namespace snack_pack_distribution_l151_15195

/-- Given the number of pretzels, goldfish, suckers, and kids, calculate the number of items per baggie -/
def items_per_baggie (pretzels : ℕ) (goldfish_multiplier : ℕ) (suckers : ℕ) (kids : ℕ) : ℕ :=
  (pretzels + pretzels * goldfish_multiplier + suckers) / kids

/-- Theorem: Given 64 pretzels, 4 times as many goldfish, 32 suckers, and 16 kids, each baggie will contain 22 items -/
theorem snack_pack_distribution :
  items_per_baggie 64 4 32 16 = 22 := by
  sorry

end snack_pack_distribution_l151_15195


namespace base_6_addition_l151_15120

/-- Addition in base 6 -/
def add_base_6 (a b : ℕ) : ℕ :=
  (a + b) % 36

/-- Conversion from base 6 to decimal -/
def base_6_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

theorem base_6_addition :
  add_base_6 (base_6_to_decimal 4) (base_6_to_decimal 14) = base_6_to_decimal 22 := by
  sorry

#eval add_base_6 (base_6_to_decimal 4) (base_6_to_decimal 14)
#eval base_6_to_decimal 22

end base_6_addition_l151_15120


namespace sin_alpha_on_ray_l151_15179

/-- If the terminal side of angle α lies on the ray y = -√3x (x < 0), then sin α = √3/2 -/
theorem sin_alpha_on_ray (α : Real) : 
  (∃ (x y : Real), x < 0 ∧ y = -Real.sqrt 3 * x ∧ 
   (∃ (r : Real), x^2 + y^2 = r^2 ∧ Real.sin α = y / r)) →
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end sin_alpha_on_ray_l151_15179


namespace chantel_bracelet_count_l151_15163

/-- The number of bracelets Chantel has at the end of the process --/
def final_bracelet_count (initial_daily_production : ℕ) (initial_days : ℕ) 
  (first_giveaway : ℕ) (second_daily_production : ℕ) (second_days : ℕ) 
  (second_giveaway : ℕ) : ℕ :=
  initial_daily_production * initial_days - first_giveaway + 
  second_daily_production * second_days - second_giveaway

/-- Theorem stating that Chantel ends up with 13 bracelets --/
theorem chantel_bracelet_count : 
  final_bracelet_count 2 5 3 3 4 6 = 13 := by
  sorry

end chantel_bracelet_count_l151_15163


namespace arithmetic_sequence_60th_term_l151_15166

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_60th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic a)
  (h_first : a 1 = 3)
  (h_fifteenth : a 15 = 31) :
  a 60 = 121 :=
sorry

end arithmetic_sequence_60th_term_l151_15166


namespace max_price_of_roses_and_peonies_l151_15117

-- Define the price of a rose and a peony
variable (R P : ℝ)

-- Define the conditions
def condition1 : Prop := 4 * R + 5 * P ≥ 27
def condition2 : Prop := 6 * R + 3 * P ≤ 27

-- Define the objective function
def objective : ℝ := 3 * R + 4 * P

-- Theorem statement
theorem max_price_of_roses_and_peonies :
  condition1 R P → condition2 R P → ∃ (max : ℝ), max = 36 ∧ objective R P ≤ max :=
by sorry

end max_price_of_roses_and_peonies_l151_15117


namespace x_range_when_p_and_not_q_x_in_range_implies_p_and_not_q_l151_15162

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x : ℝ) : Prop := 1/(3 - x) > 1

-- Define the set representing the range of x
def range_x : Set ℝ := {x | x < -3 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3}

-- Theorem statement
theorem x_range_when_p_and_not_q (x : ℝ) :
  p x ∧ ¬(q x) → x ∈ range_x :=
by
  sorry

-- Theorem for the converse (to show equivalence)
theorem x_in_range_implies_p_and_not_q (x : ℝ) :
  x ∈ range_x → p x ∧ ¬(q x) :=
by
  sorry

end x_range_when_p_and_not_q_x_in_range_implies_p_and_not_q_l151_15162


namespace annulus_area_l151_15185

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  b : ℝ  -- radius of the larger circle
  c : ℝ  -- radius of the smaller circle
  h : b > c

/-- The configuration of the annulus with a tangent line. -/
structure AnnulusConfig extends Annulus where
  a : ℝ  -- length of the tangent line XZ
  d : ℝ  -- length of YZ
  e : ℝ  -- length of XY

/-- The area of an annulus is πa², where a is the length of a tangent line
    from a point on the smaller circle to the larger circle. -/
theorem annulus_area (config : AnnulusConfig) : 
  (config.b ^ 2 - config.c ^ 2) * π = config.a ^ 2 * π := by
  sorry

end annulus_area_l151_15185


namespace function_inequality_l151_15126

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h' : ∀ x, f' x > f x) : f (Real.log 2022) > 2022 * f 0 := by
  sorry

end function_inequality_l151_15126


namespace complex_fraction_equality_l151_15196

theorem complex_fraction_equality : (5 * Complex.I) / (2 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end complex_fraction_equality_l151_15196


namespace major_axis_length_for_given_conditions_l151_15102

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorToMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorToMinorRatio

/-- Theorem: The major axis length is 6 for a cylinder of radius 2 and 50% longer major axis --/
theorem major_axis_length_for_given_conditions :
  majorAxisLength 2 1.5 = 6 := by
  sorry

end major_axis_length_for_given_conditions_l151_15102


namespace julian_airplane_models_l151_15181

theorem julian_airplane_models : 
  ∀ (total_legos : ℕ) (legos_per_model : ℕ) (additional_legos_needed : ℕ),
    total_legos = 400 →
    legos_per_model = 240 →
    additional_legos_needed = 80 →
    (total_legos + additional_legos_needed) / legos_per_model = 2 := by
  sorry

end julian_airplane_models_l151_15181


namespace triangle_perimeter_from_inradius_and_area_l151_15193

theorem triangle_perimeter_from_inradius_and_area :
  ∀ (r A p : ℝ),
  r > 0 →
  A > 0 →
  r = 2.5 →
  A = 60 →
  A = r * (p / 2) →
  p = 48 := by
sorry

end triangle_perimeter_from_inradius_and_area_l151_15193


namespace rightmost_two_digits_l151_15116

theorem rightmost_two_digits : ∃ n : ℕ, (4^127 + 5^129 + 7^131) % 100 = 52 + 100 * n := by
  sorry

end rightmost_two_digits_l151_15116


namespace variance_of_surviving_trees_l151_15167

/-- The number of osmanthus trees transplanted -/
def n : ℕ := 4

/-- The probability of survival for each tree -/
def p : ℚ := 4/5

/-- The random variable representing the number of surviving trees -/
def X : ℕ → ℚ := sorry

/-- The expected value of X -/
def E_X : ℚ := n * p

/-- The variance of X -/
def Var_X : ℚ := n * p * (1 - p)

theorem variance_of_surviving_trees :
  Var_X = 16/25 := by sorry

end variance_of_surviving_trees_l151_15167


namespace quartic_root_ratio_l151_15147

theorem quartic_root_ratio (a b c d e : ℝ) (h : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = 3) : 
  d / e = 5 / 6 := by
sorry

end quartic_root_ratio_l151_15147


namespace max_third_term_arithmetic_sequence_greatest_third_term_l151_15171

theorem max_third_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) 
  (h3 : a + (a + d) + (a + 2*d) + (a + 3*d) = 52) : 
  ∀ (x y : ℕ), 0 < x → 0 < y → 
  x + (x + y) + (x + 2*y) + (x + 3*y) = 52 → 
  x + 2*y ≤ a + 2*d := by
sorry

theorem greatest_third_term : 
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ 
  a + (a + d) + (a + 2*d) + (a + 3*d) = 52 ∧
  a + 2*d = 17 ∧
  (∀ (x y : ℕ), 0 < x → 0 < y → 
   x + (x + y) + (x + 2*y) + (x + 3*y) = 52 → 
   x + 2*y ≤ 17) := by
sorry

end max_third_term_arithmetic_sequence_greatest_third_term_l151_15171


namespace monday_sales_correct_l151_15150

/-- Represents the inventory and sales of hand sanitizer bottles at Danivan Drugstore --/
structure DrugstoreInventory where
  initial_inventory : ℕ
  tuesday_sales : ℕ
  daily_sales_wed_to_sun : ℕ
  saturday_delivery : ℕ
  end_week_inventory : ℕ

/-- Calculates the number of bottles sold on Monday --/
def monday_sales (d : DrugstoreInventory) : ℕ :=
  d.initial_inventory - d.tuesday_sales - (5 * d.daily_sales_wed_to_sun) + d.saturday_delivery - d.end_week_inventory

/-- Theorem stating that the number of bottles sold on Monday is 2445 --/
theorem monday_sales_correct (d : DrugstoreInventory) 
  (h1 : d.initial_inventory = 4500)
  (h2 : d.tuesday_sales = 900)
  (h3 : d.daily_sales_wed_to_sun = 50)
  (h4 : d.saturday_delivery = 650)
  (h5 : d.end_week_inventory = 1555) :
  monday_sales d = 2445 := by
  sorry

end monday_sales_correct_l151_15150


namespace max_pages_copied_l151_15146

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the budget in dollars
def budget : ℕ := 15

-- Define the function to calculate the number of pages
def pages_copied (cost : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost

-- Theorem statement
theorem max_pages_copied :
  pages_copied cost_per_page budget = 500 := by
  sorry

end max_pages_copied_l151_15146


namespace inequality_solution_l151_15174

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / (x * (x + 2)) < 1 / 4) ↔ (x < -1 ∨ x > 1) := by
  sorry

end inequality_solution_l151_15174


namespace max_value_of_function_l151_15105

theorem max_value_of_function (x : ℝ) (h : x < 0) :
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 ∧ ∃ y < 0, 3 * y + 4 / y = -4 * Real.sqrt 3 := by
  sorry

end max_value_of_function_l151_15105


namespace sin_300_degrees_l151_15152

theorem sin_300_degrees : Real.sin (300 * Real.pi / 180) = -1/2 := by sorry

end sin_300_degrees_l151_15152


namespace smallest_perimeter_l151_15114

/- Define the triangle PQR -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  isosceles : dist P Q = dist P R

/- Define the intersection point J -/
def J (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

/- Define the perimeter of the triangle -/
def perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

/- Main theorem -/
theorem smallest_perimeter (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  dist Q (J P Q R) = 10 →
  ∃ (P' Q' R' : ℝ × ℝ), Triangle P' Q' R' ∧
    dist Q' (J P' Q' R') = 10 ∧
    perimeter P' Q' R' = 198 ∧
    ∀ (P'' Q'' R'' : ℝ × ℝ), Triangle P'' Q'' R'' →
      dist Q'' (J P'' Q'' R'') = 10 →
      perimeter P'' Q'' R'' ≥ 198 := by
  sorry

end smallest_perimeter_l151_15114


namespace job_completion_time_l151_15175

/-- Represents the time to complete a job given initial and final workforce conditions -/
def total_completion_time (n k t : ℕ) : ℝ :=
  t + 4 * (n + k)

/-- Theorem stating the total time to complete the job -/
theorem job_completion_time (n k t : ℕ) :
  (3 / 4 : ℝ) / t = n / total_completion_time n k t ∧
  (1 / 4 : ℝ) / (total_completion_time n k t - t) = (n + k) / (total_completion_time n k t - t) →
  total_completion_time n k t = t + 4 * (n + k) := by
  sorry

#check job_completion_time

end job_completion_time_l151_15175


namespace face_D_opposite_Y_l151_15189

-- Define the faces of the cube
inductive Face
| A | B | C | D | E | Y

-- Define the structure of the net
structure Net :=
  (faces : List Face)
  (adjacent : Face → Face → Bool)

-- Define the structure of the cube
structure Cube :=
  (faces : List Face)
  (opposite : Face → Face)

-- Define the folding operation
def fold (net : Net) : Cube :=
  sorry

-- The theorem to prove
theorem face_D_opposite_Y (net : Net) (cube : Cube) :
  net.faces = [Face.A, Face.B, Face.C, Face.D, Face.E, Face.Y] →
  cube = fold net →
  cube.opposite Face.Y = Face.D :=
sorry

end face_D_opposite_Y_l151_15189


namespace quadratic_inequality_condition_l151_15173

theorem quadratic_inequality_condition (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_inequality_condition_l151_15173


namespace interior_angle_sum_increases_interior_angle_sum_formula_l151_15169

/-- The sum of interior angles of a polygon with k sides -/
def interior_angle_sum (k : ℕ) : ℝ := (k - 2) * 180

/-- Theorem: The sum of interior angles increases as the number of sides increases -/
theorem interior_angle_sum_increases (k : ℕ) (h : k ≥ 3) :
  interior_angle_sum k < interior_angle_sum (k + 1) := by
  sorry

/-- Theorem: The sum of interior angles of a k-sided polygon is (k-2) * 180° -/
theorem interior_angle_sum_formula (k : ℕ) (h : k ≥ 3) :
  interior_angle_sum k = (k - 2) * 180 := by
  sorry

end interior_angle_sum_increases_interior_angle_sum_formula_l151_15169


namespace third_candidate_votes_l151_15164

theorem third_candidate_votes : 
  let total_votes : ℕ := 23400
  let candidate1_votes : ℕ := 7636
  let candidate2_votes : ℕ := 11628
  let winning_percentage : ℚ := 49.69230769230769 / 100
  ∀ (third_candidate_votes : ℕ),
    (candidate1_votes + candidate2_votes + third_candidate_votes = total_votes) ∧
    (candidate2_votes = (winning_percentage * total_votes).floor) →
    third_candidate_votes = 4136 := by
sorry

end third_candidate_votes_l151_15164


namespace johns_grass_height_l151_15184

/-- The height to which John cuts his grass -/
def cut_height : ℝ := 2

/-- The monthly growth rate of the grass in inches -/
def growth_rate : ℝ := 0.5

/-- The maximum height of the grass before cutting in inches -/
def max_height : ℝ := 4

/-- The number of times John cuts his grass per year -/
def cuts_per_year : ℕ := 3

/-- The number of months between each cutting -/
def months_between_cuts : ℕ := 4

theorem johns_grass_height :
  cut_height + growth_rate * months_between_cuts = max_height :=
sorry

end johns_grass_height_l151_15184


namespace remaining_seeds_l151_15156

def initial_seeds : ℝ := 8.75
def sowed_seeds : ℝ := 2.75

theorem remaining_seeds :
  initial_seeds - sowed_seeds = 6 := by sorry

end remaining_seeds_l151_15156


namespace distinct_roots_of_f_l151_15124

-- Define the function
def f (x : ℝ) : ℝ := (x - 5) * (x + 3)^2

-- Theorem statement
theorem distinct_roots_of_f :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ ∀ x, f x = 0 → x = r₁ ∨ x = r₂ :=
sorry

end distinct_roots_of_f_l151_15124


namespace new_cards_count_l151_15161

theorem new_cards_count (cards_per_page : ℕ) (old_cards : ℕ) (total_pages : ℕ) : 
  cards_per_page = 3 → old_cards = 16 → total_pages = 8 → 
  total_pages * cards_per_page - old_cards = 8 := by
  sorry

end new_cards_count_l151_15161


namespace repeating_decimal_ratio_l151_15101

-- Define repeating decimal 0.overline{36}
def repeating_36 : ℚ := 36 / 99

-- Define repeating decimal 0.overline{09}
def repeating_09 : ℚ := 9 / 99

-- Theorem statement
theorem repeating_decimal_ratio : repeating_36 / repeating_09 = 4 := by
  sorry

end repeating_decimal_ratio_l151_15101


namespace vector_parallel_condition_l151_15186

/-- Given two vectors a and b in R², where a = (2,1) and b = (k,3),
    if a + 2b is parallel to 2a - b, then k = 6 -/
theorem vector_parallel_condition (k : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![k, 3]
  (∃ (t : ℝ), t ≠ 0 ∧ (a + 2 • b) = t • (2 • a - b)) →
  k = 6 :=
by sorry

end vector_parallel_condition_l151_15186


namespace base_10_to_base_5_88_l151_15107

def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_10_to_base_5_88 : to_base_5 88 = [3, 2, 3] := by
  sorry

end base_10_to_base_5_88_l151_15107


namespace book_purchases_l151_15199

theorem book_purchases (people_A : ℕ) (people_B : ℕ) (people_both : ℕ) (people_only_B : ℕ) (people_only_A : ℕ) : 
  people_A = 2 * people_B →
  people_both = 500 →
  people_both = 2 * people_only_B →
  people_A = people_only_A + people_both →
  people_B = people_only_B + people_both →
  people_only_A = 1000 := by
sorry

end book_purchases_l151_15199


namespace locus_definition_correct_l151_15106

-- Define the space we're working in (e.g., a metric space)
variable {X : Type*} [MetricSpace X]

-- Define the locus and the distance
variable (P : X) (r : ℝ) (locus : Set X)

-- Define the condition for a point to be at distance r from P
def atDistanceR (x : X) := dist x P = r

-- State the theorem
theorem locus_definition_correct :
  (∀ x : X, atDistanceR P r x → x ∈ locus) ∧
  (∀ x : X, x ∈ locus → atDistanceR P r x) :=
sorry

end locus_definition_correct_l151_15106


namespace intersection_of_A_and_B_l151_15183

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | 1 - x > 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l151_15183


namespace pizza_slice_ratio_l151_15130

theorem pizza_slice_ratio : 
  ∀ (total_slices lunch_slices : ℕ),
    total_slices = 12 →
    lunch_slices ≤ total_slices →
    (total_slices - lunch_slices) / 3 + 4 = total_slices - lunch_slices →
    lunch_slices = total_slices / 2 :=
by
  sorry

end pizza_slice_ratio_l151_15130


namespace triangle_sides_with_inscribed_rhombus_l151_15143

/-- A right triangle with a 60° angle and an inscribed rhombus -/
structure TriangleWithRhombus where
  /-- Side length of the inscribed rhombus -/
  rhombus_side : ℝ
  /-- The rhombus shares the 60° angle with the triangle -/
  shares_angle : Bool
  /-- All vertices of the rhombus lie on the sides of the triangle -/
  vertices_on_sides : Bool

/-- Theorem about the sides of the triangle given the inscribed rhombus -/
theorem triangle_sides_with_inscribed_rhombus 
  (t : TriangleWithRhombus) 
  (h1 : t.rhombus_side = 6) 
  (h2 : t.shares_angle) 
  (h3 : t.vertices_on_sides) : 
  ∃ (a b c : ℝ), a = 9 ∧ b = 9 * Real.sqrt 3 ∧ c = 18 := by
  sorry

end triangle_sides_with_inscribed_rhombus_l151_15143


namespace security_guard_schedule_l151_15192

structure Guard where
  id : Nat
  hours : Nat

def valid_schedule (g2 g3 g4 g5 : Guard) : Prop :=
  g2.id = 2 ∧ g3.id = 3 ∧ g4.id = 4 ∧ g5.id = 5 ∧
  g2.hours + g3.hours + g4.hours + g5.hours = 6 ∧
  g2.hours ≤ 2 ∧
  g3.hours ≤ 3 ∧
  g4.hours = g5.hours + 1 ∧
  g5.hours > 0

theorem security_guard_schedule :
  ∃ (g2 g3 g4 g5 : Guard), valid_schedule g2 g3 g4 g5 :=
sorry

end security_guard_schedule_l151_15192


namespace min_value_inequality_l151_15155

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + 2 * y + 6) :
  1 / x + 1 / (2 * y) ≥ 1 / 3 := by
sorry

end min_value_inequality_l151_15155


namespace employee_payment_percentage_l151_15133

theorem employee_payment_percentage (total_payment y_payment x_payment : ℝ) :
  total_payment = 770 →
  y_payment = 350 →
  x_payment + y_payment = total_payment →
  x_payment / y_payment = 1.2 := by
  sorry

end employee_payment_percentage_l151_15133


namespace ben_basketball_boxes_ben_basketball_boxes_correct_l151_15194

theorem ben_basketball_boxes (basketball_cards_per_box : ℕ) 
                              (baseball_boxes : ℕ) 
                              (baseball_cards_per_box : ℕ) 
                              (cards_given_away : ℕ) 
                              (cards_left : ℕ) : ℕ :=
  let total_baseball_cards := baseball_boxes * baseball_cards_per_box
  let total_cards_before := cards_given_away + cards_left
  let basketball_boxes := (total_cards_before - total_baseball_cards) / basketball_cards_per_box
  basketball_boxes

#check ben_basketball_boxes 10 5 8 58 22 = 4

theorem ben_basketball_boxes_correct : ben_basketball_boxes 10 5 8 58 22 = 4 := by
  sorry

end ben_basketball_boxes_ben_basketball_boxes_correct_l151_15194


namespace triangle_area_tangent_circles_l151_15159

/-- Given two non-overlapping circles with radii r₁ and r₂, where one common internal tangent
    is perpendicular to one common external tangent, the area S of the triangle formed by
    these tangents and the third common tangent satisfies one of two formulas. -/
theorem triangle_area_tangent_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₁ ≠ r₂) :
  ∃ S : ℝ, (S = (r₁ * r₂ * (r₁ + r₂)) / |r₁ - r₂|) ∨ (S = (r₁ * r₂ * |r₁ - r₂|) / (r₁ + r₂)) :=
by sorry

end triangle_area_tangent_circles_l151_15159


namespace science_club_membership_l151_15127

theorem science_club_membership (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 10 := by
  sorry

end science_club_membership_l151_15127


namespace climbing_time_problem_l151_15108

/-- The time it takes for Jason to be 42 feet higher than Matt, given their climbing rates. -/
theorem climbing_time_problem (matt_rate jason_rate : ℝ) (height_difference : ℝ)
  (h_matt : matt_rate = 6)
  (h_jason : jason_rate = 12)
  (h_diff : height_difference = 42) :
  (height_difference / (jason_rate - matt_rate)) = 7 :=
by sorry

end climbing_time_problem_l151_15108


namespace polynomial_value_l151_15188

theorem polynomial_value (a b c d : ℝ) : 
  (∀ x, a * x^5 + b * x^3 + c * x + d = 
    (fun x => a * x^5 + b * x^3 + c * x + d) x) →
  (a * 0^5 + b * 0^3 + c * 0 + d = -5) →
  (a * (-3)^5 + b * (-3)^3 + c * (-3) + d = 7) →
  (a * 3^5 + b * 3^3 + c * 3 + d = -17) := by
sorry

end polynomial_value_l151_15188


namespace infinitely_many_primes_with_large_primitive_root_l151_15138

theorem infinitely_many_primes_with_large_primitive_root (n : ℕ) (hn : n > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ p ∈ S,
    Nat.Prime p ∧ ∀ m ∈ Finset.range n, ∃ x, x^2 ≡ m [MOD p] :=
sorry

end infinitely_many_primes_with_large_primitive_root_l151_15138


namespace x_minus_y_equals_half_l151_15118

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x y : ℝ) : Set ℝ := {1/x, |x|, y/x}

-- State the theorem
theorem x_minus_y_equals_half (x y : ℝ) : A x = B x y → x - y = 1/2 := by
  sorry

end x_minus_y_equals_half_l151_15118


namespace tetrahedron_divides_space_l151_15137

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  faces : Fin 4 → Plane
  edges : Fin 6 → Line
  vertices : Fin 4 → Point

/-- The number of regions formed by the planes of a tetrahedron's faces -/
def num_regions (t : Tetrahedron) : ℕ := 15

/-- Theorem stating that the planes of a tetrahedron's faces divide space into 15 regions -/
theorem tetrahedron_divides_space (t : Tetrahedron) : 
  num_regions t = 15 := by sorry

end tetrahedron_divides_space_l151_15137


namespace weight_of_eight_moles_l151_15182

/-- The total weight of a given number of moles of a compound -/
def total_weight (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

/-- Proof that 8 moles of a compound with molecular weight 496 g/mol has a total weight of 3968 g -/
theorem weight_of_eight_moles :
  let molecular_weight : ℝ := 496
  let moles : ℝ := 8
  total_weight molecular_weight moles = 3968 := by
  sorry

end weight_of_eight_moles_l151_15182


namespace smallest_digit_change_correct_change_l151_15157

def original_sum : ℕ := 738 + 625 + 841
def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 2204

def change_digit (n : ℕ) (place : ℕ) (new_digit : ℕ) : ℕ :=
  n - (n / 10^place % 10) * 10^place + new_digit * 10^place

theorem smallest_digit_change :
  ∀ (d : ℕ),
    d < 6 →
    ¬∃ (n : ℕ) (place : ℕ),
      (n = 738 ∨ n = 625 ∨ n = 841) ∧
      change_digit n place d + 
        (if n = 738 then 625 + 841
         else if n = 625 then 738 + 841
         else 738 + 625) = correct_sum :=
by sorry

theorem correct_change :
  change_digit 625 2 5 + 738 + 841 = correct_sum :=
by sorry

end smallest_digit_change_correct_change_l151_15157


namespace no_common_root_with_specific_values_l151_15144

theorem no_common_root_with_specific_values : ¬ ∃ (P₁ P₂ : ℤ → ℤ) (a b : ℤ),
  (∀ x, ∃ (c : ℤ), P₁ x = c) ∧  -- P₁ has integer coefficients
  (∀ x, ∃ (c : ℤ), P₂ x = c) ∧  -- P₂ has integer coefficients
  a < 0 ∧                       -- a is strictly negative
  P₁ a = 0 ∧                    -- a is a root of P₁
  P₂ a = 0 ∧                    -- a is a root of P₂
  b > 0 ∧                       -- b is positive
  P₁ b = 2007 ∧                 -- P₁ evaluates to 2007 at b
  P₂ b = 2008                   -- P₂ evaluates to 2008 at b
  := by sorry

end no_common_root_with_specific_values_l151_15144


namespace quadratic_real_root_condition_l151_15109

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end quadratic_real_root_condition_l151_15109


namespace center_C_range_l151_15128

-- Define the points and line
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define circle C
def C (a : ℝ) : ℝ × ℝ := (a, l a)
def radius_C : ℝ := 1

-- Define moving point M
def M : ℝ × ℝ → Prop := λ (x, y) => (x^2 + (y - 3)^2) = 4 * (x^2 + y^2)

-- Define the intersection condition
def intersects (C : ℝ × ℝ) (M : ℝ × ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), M (x, y) ∧ (x - C.1)^2 + (y - C.2)^2 = radius_C^2

-- Theorem statement
theorem center_C_range (a : ℝ) :
  (C a).2 = l (C a).1 →  -- Center of C lies on line l
  intersects (C a) M →   -- M intersects with C
  0 ≤ a ∧ a ≤ 12/5 :=
by sorry

end center_C_range_l151_15128


namespace christophers_gabrielas_age_ratio_l151_15123

/-- Proves that given Christopher is 2 times as old as Gabriela and Christopher is 24 years old, 
    the ratio of Christopher's age to Gabriela's age nine years ago is 5:1. -/
theorem christophers_gabrielas_age_ratio : 
  ∀ (christopher_age gabriela_age : ℕ),
    christopher_age = 2 * gabriela_age →
    christopher_age = 24 →
    (christopher_age - 9) / (gabriela_age - 9) = 5 := by
  sorry

end christophers_gabrielas_age_ratio_l151_15123


namespace negation_of_universal_statement_l151_15110

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end negation_of_universal_statement_l151_15110


namespace cheese_calories_per_serving_l151_15170

/-- Represents the number of calories in a serving of cheese -/
def calories_per_serving (total_servings : ℕ) (eaten_servings : ℕ) (remaining_calories : ℕ) : ℕ :=
  remaining_calories / (total_servings - eaten_servings)

/-- Theorem stating that the number of calories in a serving of cheese is 110 -/
theorem cheese_calories_per_serving :
  calories_per_serving 16 5 1210 = 110 := by
  sorry

end cheese_calories_per_serving_l151_15170


namespace meghan_weight_conversion_l151_15172

/-- Given a base b, this function calculates the value of a number represented as 451 in base b -/
def value_in_base_b (b : ℕ) : ℕ := 4 * b^2 + 5 * b + 1

/-- Given a base b, this function calculates the value of a number represented as 127 in base 2b -/
def value_in_base_2b (b : ℕ) : ℕ := 1 * (2*b)^2 + 2 * (2*b) + 7

/-- Theorem stating that if a number is represented as 451 in base b and 127 in base 2b, 
    then it is equal to 175 in base 10 -/
theorem meghan_weight_conversion (b : ℕ) : 
  value_in_base_b b = value_in_base_2b b → value_in_base_b b = 175 := by
  sorry

#eval value_in_base_b 6  -- Should output 175
#eval value_in_base_2b 6  -- Should also output 175

end meghan_weight_conversion_l151_15172


namespace circle_radius_is_one_l151_15134

-- Define the polar equation of the circle
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the Cartesian equation of the circle
def cartesian_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_radius_is_one :
  ∀ ρ θ x y : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  cartesian_equation x y →
  1 = (x^2 + y^2).sqrt :=
by sorry


end circle_radius_is_one_l151_15134


namespace perfect_square_quadratic_l151_15191

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m*x + 16 = y^2) → (m = 8 ∨ m = -8) :=
by sorry

end perfect_square_quadratic_l151_15191


namespace slide_boys_count_l151_15145

/-- The number of boys who initially went down the slide -/
def initial_boys : ℕ := 22

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := initial_boys + additional_boys

theorem slide_boys_count : total_boys = 35 := by
  sorry

end slide_boys_count_l151_15145


namespace probability_of_meeting_l151_15111

/-- Two people arrive independently and uniformly at random within a 2-hour interval -/
def arrival_interval : ℝ := 2

/-- Each person stays for 30 minutes after arrival -/
def stay_duration : ℝ := 0.5

/-- The maximum arrival time for each person is 30 minutes before the end of the 2-hour interval -/
def max_arrival_time : ℝ := arrival_interval - stay_duration

/-- The probability of two people seeing each other given the conditions -/
theorem probability_of_meeting :
  let total_area : ℝ := arrival_interval ^ 2
  let overlap_area : ℝ := total_area - 2 * (stay_duration ^ 2 / 2)
  overlap_area / total_area = 15 / 16 := by sorry

end probability_of_meeting_l151_15111


namespace number_problem_l151_15180

theorem number_problem : ∃ x : ℚ, x = 15 + (x * 9/64) + (x * 1/2) ∧ x = 960/23 := by
  sorry

end number_problem_l151_15180


namespace square_value_proof_l151_15135

theorem square_value_proof : ∃ (square : ℚ), 
  (13.5 / (11 + (2.25 / (1 - square))) - 1 / 7) * (7/6) = 1 ∧ square = 1/10 := by
  sorry

end square_value_proof_l151_15135


namespace unique_divisible_by_18_l151_15142

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_18 :
  ∀ n : ℕ, n < 10 →
    (is_divisible_by (7120 + n) 18 ↔ n = 8) :=
by sorry

end unique_divisible_by_18_l151_15142


namespace special_function_value_l151_15158

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value 
  (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : f 45 = 15) : 
  f 60 = 11.25 := by
  sorry

end special_function_value_l151_15158


namespace circle_ratio_after_radius_increase_l151_15112

theorem circle_ratio_after_radius_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end circle_ratio_after_radius_increase_l151_15112


namespace weight_of_BaCl2_l151_15177

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The number of moles of BaCl2 -/
def moles_BaCl2 : ℝ := 8

/-- The molecular weight of BaCl2 in g/mol -/
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

/-- The total weight of BaCl2 in grams -/
def total_weight_BaCl2 : ℝ := molecular_weight_BaCl2 * moles_BaCl2

theorem weight_of_BaCl2 : total_weight_BaCl2 = 1665.84 := by
  sorry

end weight_of_BaCl2_l151_15177


namespace factorization_problems_l151_15160

theorem factorization_problems :
  (∀ a : ℝ, 18 * a^2 - 32 = 2 * (3*a + 4) * (3*a - 4)) ∧
  (∀ x y : ℝ, y - 6*x*y + 9*x^2*y = y * (1 - 3*x)^2) :=
by sorry

end factorization_problems_l151_15160


namespace circle_circumference_approximation_l151_15131

/-- The circumference of a circle with radius 0.4997465213085514 meters is approximately 3.140093 meters. -/
theorem circle_circumference_approximation :
  let r : ℝ := 0.4997465213085514
  let π : ℝ := Real.pi
  let C : ℝ := 2 * π * r
  ∃ ε > 0, |C - 3.140093| < ε :=
by
  sorry

end circle_circumference_approximation_l151_15131


namespace work_earnings_equality_l151_15197

theorem work_earnings_equality (t : ℝ) 
  (my_hours : ℝ := t - 4)
  (my_rate : ℝ := 3*t - 7)
  (bob_hours : ℝ := 3*t - 12)
  (bob_rate : ℝ := t - 6)
  (h : my_hours * my_rate = bob_hours * bob_rate) : 
  t = 44 := by
sorry

end work_earnings_equality_l151_15197


namespace root_product_value_l151_15168

theorem root_product_value (m n : ℝ) : 
  m^2 - 2019*m - 1 = 0 → 
  n^2 - 2019*n - 1 = 0 → 
  (m^2 - 2019*m + 3) * (n^2 - 2019*n + 4) = 20 := by
sorry

end root_product_value_l151_15168


namespace group_arrangements_eq_40_l151_15132

/-- The number of ways to divide 2 teachers and 6 students into two groups,
    each consisting of 1 teacher and 3 students. -/
def group_arrangements : ℕ :=
  (Nat.choose 2 1) * (Nat.choose 6 3)

/-- Theorem stating that the number of group arrangements is 40. -/
theorem group_arrangements_eq_40 : group_arrangements = 40 := by
  sorry

end group_arrangements_eq_40_l151_15132


namespace polygon_sides_l151_15165

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ sum_interior_angles = 180 * (n - 2) := by
  sorry

end polygon_sides_l151_15165


namespace seokjin_paper_count_l151_15176

theorem seokjin_paper_count (jimin_count : ℕ) (difference : ℕ) 
  (h1 : jimin_count = 41)
  (h2 : difference = 1)
  (h3 : jimin_count = seokjin_count + difference) :
  seokjin_count = 40 := by
  sorry

end seokjin_paper_count_l151_15176


namespace school_picnic_volunteers_l151_15149

theorem school_picnic_volunteers (total_parents supervise_parents refreshment_parents : ℕ) 
  (h1 : total_parents = 84)
  (h2 : supervise_parents = 25)
  (h3 : refreshment_parents = 42)
  (h4 : refreshment_parents = (3/2 : ℚ) * (total_parents - supervise_parents - refreshment_parents + both_parents)) :
  ∃ both_parents : ℕ, both_parents = 11 := by
  sorry

end school_picnic_volunteers_l151_15149
