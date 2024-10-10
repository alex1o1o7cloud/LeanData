import Mathlib

namespace base_seven_divisibility_l1603_160385

theorem base_seven_divisibility (y : ℕ) : 
  y ≤ 6 → (∃! y, (5 * 7^2 + y * 7 + 2) % 19 = 0 ∧ y ≤ 6) → y = 0 := by
  sorry

end base_seven_divisibility_l1603_160385


namespace derivative_of_y_l1603_160354

noncomputable def y (x : ℝ) : ℝ :=
  (1 / Real.sqrt 8) * Real.log ((4 + Real.sqrt 8 * Real.tanh (x / 2)) / (4 - Real.sqrt 8 * Real.tanh (x / 2)))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (2 * (Real.cosh (x / 2) ^ 2 + 1)) :=
sorry

end derivative_of_y_l1603_160354


namespace ratio_problem_l1603_160379

theorem ratio_problem (x : ℝ) : (20 / 1 = x / 10) → x = 200 := by
  sorry

end ratio_problem_l1603_160379


namespace circplus_square_sum_diff_l1603_160326

/-- Custom operation ⊕ for real numbers -/
def circplus (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the equality for (x+y)^2 ⊕ (x-y)^2 -/
theorem circplus_square_sum_diff (x y : ℝ) : 
  circplus ((x + y)^2) ((x - y)^2) = 4 * (x^2 + y^2)^2 := by
  sorry

end circplus_square_sum_diff_l1603_160326


namespace carpenter_logs_l1603_160389

/-- Proves that the carpenter currently has 8 logs given the conditions of the problem -/
theorem carpenter_logs :
  ∀ (total_woodblocks : ℕ) (woodblocks_per_log : ℕ) (additional_logs_needed : ℕ),
    total_woodblocks = 80 →
    woodblocks_per_log = 5 →
    additional_logs_needed = 8 →
    (total_woodblocks - additional_logs_needed * woodblocks_per_log) / woodblocks_per_log = 8 :=
by
  sorry

end carpenter_logs_l1603_160389


namespace geometric_sequence_first_term_l1603_160392

/-- Given a geometric sequence where the third term is 24 and the fourth term is 36, 
    the first term of the sequence is 32/3. -/
theorem geometric_sequence_first_term (a : ℚ) (r : ℚ) : 
  a * r^2 = 24 ∧ a * r^3 = 36 → a = 32/3 := by
  sorry

end geometric_sequence_first_term_l1603_160392


namespace min_value_expression_min_value_achievable_l1603_160344

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + 2*y^2)).sqrt) / (x*y) ≥ 2 + Real.sqrt 2 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (((x^2 + y^2) * (4*x^2 + 2*y^2)).sqrt) / (x*y) = 2 + Real.sqrt 2 :=
sorry

end min_value_expression_min_value_achievable_l1603_160344


namespace right_triangle_conditions_l1603_160305

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles A, B, C in radians

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.C = t.B
def condition2 (t : Triangle) : Prop := ∃ (k : ℝ), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k
def condition3 (t : Triangle) : Prop := ∃ (AB BC AC : ℝ), 3*AB = 4*BC ∧ 4*BC = 5*AC
def condition4 (t : Triangle) : Prop := t.A = t.B ∧ t.B = t.C

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

-- Theorem statement
theorem right_triangle_conditions (t : Triangle) :
  (condition1 t → is_right_triangle t) ∧
  (condition2 t → is_right_triangle t) ∧
  ¬(condition3 t → is_right_triangle t) ∧
  ¬(condition4 t → is_right_triangle t) :=
sorry

end right_triangle_conditions_l1603_160305


namespace isosceles_triangle_perimeter_l1603_160378

/-- An isosceles triangle with two sides of length 5 and one side of length 2 has perimeter 12 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b + c = 12 := by
sorry

end isosceles_triangle_perimeter_l1603_160378


namespace chocolates_left_first_method_candies_left_second_method_chocolates_left_second_method_total_items_is_35_l1603_160341

/-- Represents the number of bags packed using the first method -/
def x : ℕ := 1

/-- Represents the number of bags packed using the second method -/
def y : ℕ := 0

/-- The total number of chocolates initially -/
def total_chocolates : ℕ := 3 * x + 5 * y + 25

/-- The total number of fruit candies initially -/
def total_candies : ℕ := 7 * x + 5 * y

/-- Condition: When fruit candies are used up in the first method, 25 chocolates are left -/
theorem chocolates_left_first_method : total_chocolates - (3 * x + 5 * y) = 25 := by sorry

/-- Condition: In the second method, 4 fruit candies are left in the end -/
theorem candies_left_second_method : total_candies - (7 * x + 5 * y) = 4 := by sorry

/-- Condition: In the second method, 1 chocolate is left in the end -/
theorem chocolates_left_second_method : total_chocolates - (3 * x + 5 * y) - 4 = 1 := by sorry

/-- The main theorem: The total number of chocolates and fruit candies is 35 -/
theorem total_items_is_35 : total_chocolates + total_candies = 35 := by sorry

end chocolates_left_first_method_candies_left_second_method_chocolates_left_second_method_total_items_is_35_l1603_160341


namespace concentric_circles_radii_product_l1603_160331

theorem concentric_circles_radii_product (r₁ r₂ r₃ : ℝ) : 
  r₁ = 2 →
  (r₂^2 - r₁^2 = r₁^2) →
  (r₃^2 - r₂^2 = r₁^2) →
  (r₁ * r₂ * r₃)^2 = 384 :=
by sorry

end concentric_circles_radii_product_l1603_160331


namespace intersection_sum_l1603_160338

theorem intersection_sum (c d : ℝ) : 
  (2 = (1/5) * 3 + c) → 
  (3 = (1/5) * 2 + d) → 
  c + d = 4 := by
sorry

end intersection_sum_l1603_160338


namespace jade_transactions_l1603_160347

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 18 →
  jade = 84 :=
by sorry

end jade_transactions_l1603_160347


namespace inequality_proof_l1603_160325

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
sorry

end inequality_proof_l1603_160325


namespace inscribed_trapezoids_equal_diagonals_l1603_160391

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  vertices : Fin 4 → ℝ × ℝ

-- Define the property of being inscribed in a circle
def inscribed (t : IsoscelesTrapezoid) (c : Circle) : Prop := sorry

-- Define the property of sides being parallel
def parallel_sides (t1 t2 : IsoscelesTrapezoid) : Prop := sorry

-- Define the length of a diagonal
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := sorry

-- Main theorem
theorem inscribed_trapezoids_equal_diagonals 
  (c : Circle) (t1 t2 : IsoscelesTrapezoid) 
  (h1 : inscribed t1 c) (h2 : inscribed t2 c) 
  (h3 : parallel_sides t1 t2) : 
  diagonal_length t1 = diagonal_length t2 := by sorry

end inscribed_trapezoids_equal_diagonals_l1603_160391


namespace special_function_properties_l1603_160396

/-- An increasing function f: ℝ₊ → ℝ₊ satisfying f(xy) = f(x)f(y) for all x, y > 0, and f(2) = 4 -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 → y > 0 → f (x * y) = f x * f y) ∧
  (∀ x y, x > y → x > 0 → y > 0 → f x > f y) ∧
  f 2 = 4

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 1 = 1 ∧ f 8 = 64 ∧ Set.Ioo 3 (7/2) = {x | 16 * f (1 / (x - 3)) ≥ f (2 * x + 1)} := by
  sorry

end special_function_properties_l1603_160396


namespace jenny_activities_lcm_l1603_160352

theorem jenny_activities_lcm : Nat.lcm (Nat.lcm 6 12) 18 = 36 := by
  sorry

end jenny_activities_lcm_l1603_160352


namespace work_completion_time_l1603_160315

theorem work_completion_time (a_time b_time : ℝ) (ha : a_time = 10) (hb : b_time = 10) :
  1 / (1 / a_time + 1 / b_time) = 5 := by sorry

end work_completion_time_l1603_160315


namespace water_left_l1603_160307

theorem water_left (initial : ℚ) (used : ℚ) (left : ℚ) : 
  initial = 3 → used = 9/4 → left = initial - used → left = 3/4 := by sorry

end water_left_l1603_160307


namespace game_score_product_l1603_160361

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 0

def allie_rolls : List ℕ := [6, 3, 2, 4]
def betty_rolls : List ℕ := [5, 2, 3, 6]

theorem game_score_product : 
  (allie_rolls.map g).sum * (betty_rolls.map g).sum = 143 := by
  sorry

end game_score_product_l1603_160361


namespace george_borrowing_weeks_l1603_160319

def loan_amount : ℝ := 100
def initial_fee_rate : ℝ := 0.05
def total_fee : ℝ := 15

def fee_after_weeks (weeks : ℕ) : ℝ :=
  loan_amount * initial_fee_rate * (2 ^ weeks - 1)

theorem george_borrowing_weeks :
  ∃ (weeks : ℕ), weeks > 0 ∧ fee_after_weeks weeks ≤ total_fee ∧ fee_after_weeks (weeks + 1) > total_fee :=
by sorry

end george_borrowing_weeks_l1603_160319


namespace distribution_percent_below_mean_plus_std_dev_l1603_160324

-- Define a symmetric distribution with mean m and standard deviation d
def SymmetricDistribution (μ : ℝ) (σ : ℝ) (F : ℝ → ℝ) : Prop :=
  ∀ x, F (μ + x) + F (μ - x) = 1

-- Define the condition that 36% of the distribution lies within one standard deviation of the mean
def WithinOneStdDev (μ : ℝ) (σ : ℝ) (F : ℝ → ℝ) : Prop :=
  F (μ + σ) - F (μ - σ) = 0.36

-- Theorem statement
theorem distribution_percent_below_mean_plus_std_dev
  (μ σ : ℝ) (F : ℝ → ℝ) 
  (h_symmetric : SymmetricDistribution μ σ F)
  (h_within_one_std_dev : WithinOneStdDev μ σ F) :
  F (μ + σ) = 0.68 := by
  sorry

end distribution_percent_below_mean_plus_std_dev_l1603_160324


namespace candy_count_l1603_160349

theorem candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409) 
  (h2 : red = 145) 
  (h3 : total = red + blue) : blue = 3264 := by
  sorry

end candy_count_l1603_160349


namespace fraction_of_fraction_of_fraction_product_of_fractions_l1603_160397

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem product_of_fractions :
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end fraction_of_fraction_of_fraction_product_of_fractions_l1603_160397


namespace complex_equation_solution_l1603_160343

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 - z) → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l1603_160343


namespace carries_shopping_money_l1603_160375

theorem carries_shopping_money (initial_amount : ℕ) (sweater_cost : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) 
  (h1 : initial_amount = 91)
  (h2 : sweater_cost = 24)
  (h3 : tshirt_cost = 6)
  (h4 : shoes_cost = 11) :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end carries_shopping_money_l1603_160375


namespace sum_of_last_three_digits_of_fibonacci_factorial_series_l1603_160311

def fibonacci_factorial_series : List Nat := [1, 2, 3, 5, 8, 13, 21]

def last_three_digits (n : Nat) : Nat :=
  n % 1000

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_three_digits_of_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_three_digits (factorial n))).sum % 1000 = 249 := by
  sorry

end sum_of_last_three_digits_of_fibonacci_factorial_series_l1603_160311


namespace bus_stop_walking_time_l1603_160366

theorem bus_stop_walking_time 
  (usual_speed : ℝ) 
  (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_speed * usual_time = (4/5 * usual_speed) * (usual_time + 7)) :
  usual_time = 28 := by
sorry

end bus_stop_walking_time_l1603_160366


namespace new_average_age_l1603_160387

def initial_people : ℕ := 6
def initial_average_age : ℚ := 25
def leaving_age : ℕ := 20
def entering_age : ℕ := 30

theorem new_average_age :
  let initial_total_age : ℚ := initial_people * initial_average_age
  let new_total_age : ℚ := initial_total_age - leaving_age + entering_age
  new_total_age / initial_people = 26.67 := by
sorry

end new_average_age_l1603_160387


namespace bleacher_runs_theorem_l1603_160358

/-- The number of times a player runs up and down the bleachers -/
def number_of_trips (stairs_one_way : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ) : ℕ :=
  total_calories_burned / (4 * stairs_one_way * calories_per_stair)

/-- Theorem stating the number of times players run up and down the bleachers -/
theorem bleacher_runs_theorem (stairs_one_way : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ)
    (h1 : stairs_one_way = 32)
    (h2 : calories_per_stair = 2)
    (h3 : total_calories_burned = 5120) :
    number_of_trips stairs_one_way calories_per_stair total_calories_burned = 40 := by
  sorry

end bleacher_runs_theorem_l1603_160358


namespace jorge_goals_this_season_l1603_160345

/-- Given that Jorge scored 156 goals last season and his total goals are 343,
    prove that the number of goals he scored this season is 343 - 156. -/
theorem jorge_goals_this_season 
  (goals_last_season : ℕ) 
  (total_goals : ℕ) 
  (h1 : goals_last_season = 156) 
  (h2 : total_goals = 343) : 
  total_goals - goals_last_season = 343 - 156 :=
by sorry

end jorge_goals_this_season_l1603_160345


namespace kite_area_from_shifted_triangles_l1603_160336

/-- The area of a kite-shaped figure formed by the intersection of two equilateral triangles -/
theorem kite_area_from_shifted_triangles (square_side : ℝ) (shift : ℝ) : 
  square_side = 4 →
  shift = 1 →
  let triangle_side := square_side
  let triangle_height := (Real.sqrt 3 / 2) * triangle_side
  let kite_base := square_side - shift
  let kite_area := kite_base * triangle_height
  kite_area = 6 * Real.sqrt 3 := by
  sorry

end kite_area_from_shifted_triangles_l1603_160336


namespace focus_of_specific_parabola_l1603_160332

/-- A parabola is defined by its quadratic equation coefficients -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (x, y) -/
def focus (p : Parabola) : ℝ × ℝ :=
  sorry

/-- Theorem: The focus of the parabola y = 9x^2 + 6x - 2 is at (-1/3, -107/36) -/
theorem focus_of_specific_parabola :
  let p : Parabola := { a := 9, b := 6, c := -2 }
  focus p = (-1/3, -107/36) := by
  sorry

end focus_of_specific_parabola_l1603_160332


namespace additive_inverse_of_zero_l1603_160313

theorem additive_inverse_of_zero : (0 : ℤ) + (0 : ℤ) = (0 : ℤ) := by sorry

end additive_inverse_of_zero_l1603_160313


namespace pentagon_diagonal_equality_l1603_160318

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A pentagon defined by five points -/
structure Pentagon :=
  (A B C D E : Point)

/-- Checks if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Checks if a line segment bisects an angle -/
def bisects_angle (P Q R S : Point) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersection (P Q R S : Point) : Point := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

theorem pentagon_diagonal_equality (p : Pentagon) 
  (h_convex : is_convex p)
  (h_bd_bisect1 : bisects_angle p.C p.B p.E p.D)
  (h_bd_bisect2 : bisects_angle p.C p.D p.A p.B)
  (h_ce_bisect1 : bisects_angle p.A p.C p.D p.E)
  (h_ce_bisect2 : bisects_angle p.B p.E p.D p.C)
  (K : Point) (h_K : K = intersection p.B p.E p.A p.C)
  (L : Point) (h_L : L = intersection p.B p.E p.A p.D) :
  distance p.C K = distance p.D L := by sorry

end pentagon_diagonal_equality_l1603_160318


namespace aloks_age_l1603_160398

theorem aloks_age (alok_age bipin_age chandan_age : ℕ) : 
  bipin_age = 6 * alok_age →
  bipin_age + 10 = 2 * (chandan_age + 10) →
  chandan_age = 10 →
  alok_age = 5 := by
sorry

end aloks_age_l1603_160398


namespace road_length_l1603_160350

theorem road_length (trees : ℕ) (tree_space : ℕ) (between_space : ℕ) : 
  trees = 13 → tree_space = 1 → between_space = 12 → 
  trees * tree_space + (trees - 1) * between_space = 157 := by
  sorry

end road_length_l1603_160350


namespace expand_binomials_l1603_160360

theorem expand_binomials (x : ℝ) : (2*x + 3) * (4*x - 7) = 8*x^2 - 2*x - 21 := by
  sorry

end expand_binomials_l1603_160360


namespace cake_mix_tray_difference_l1603_160372

theorem cake_mix_tray_difference :
  ∀ (tray1_capacity tray2_capacity : ℕ),
    tray1_capacity + tray2_capacity = 500 →
    tray2_capacity = 240 →
    tray1_capacity > tray2_capacity →
    tray1_capacity - tray2_capacity = 20 :=
by
  sorry

end cake_mix_tray_difference_l1603_160372


namespace paint_cans_theorem_l1603_160321

/-- Represents the number of rooms that can be painted with the available paint -/
def initialRooms : ℕ := 50

/-- Represents the number of paint cans misplaced -/
def misplacedCans : ℕ := 5

/-- Represents the number of rooms that can be painted after misplacing some cans -/
def remainingRooms : ℕ := 37

/-- Calculates the number of cans used to paint the remaining rooms -/
def cansUsed : ℕ := 15

theorem paint_cans_theorem : 
  ∀ (initial : ℕ) (misplaced : ℕ) (remaining : ℕ),
  initial = initialRooms → 
  misplaced = misplacedCans → 
  remaining = remainingRooms → 
  cansUsed = 15 :=
by sorry

end paint_cans_theorem_l1603_160321


namespace hernandez_state_tax_l1603_160393

/-- Calculates the state tax for a resident given their taxable income and months of residence --/
def calculate_state_tax (taxable_income : ℕ) (months_of_residence : ℕ) : ℕ :=
  let adjusted_income := taxable_income - 5000
  let tax_bracket1 := min adjusted_income 10000
  let tax_bracket2 := min (max (adjusted_income - 10000) 0) 20000
  let tax_bracket3 := min (max (adjusted_income - 30000) 0) 30000
  let tax_bracket4 := max (adjusted_income - 60000) 0
  let total_tax := tax_bracket1 / 100 + tax_bracket2 * 3 / 100 + tax_bracket3 * 5 / 100 + tax_bracket4 * 7 / 100
  let tax_credit := if months_of_residence < 10 then 500 else 0
  total_tax - tax_credit

/-- The theorem stating that Mr. Hernandez's state tax is $575 --/
theorem hernandez_state_tax :
  calculate_state_tax 42500 9 = 575 :=
by sorry

end hernandez_state_tax_l1603_160393


namespace product_prices_and_savings_l1603_160339

-- Define the discount rates
def discount_A : ℚ := 0.2
def discount_B : ℚ := 0.25

-- Define the equations from the conditions
def equation1 (x y : ℚ) : Prop := 6 * x + 3 * y = 600
def equation2 (x y : ℚ) : Prop := 50 * (1 - discount_A) * x + 40 * (1 - discount_B) * y = 5200

-- Define the prices we want to prove
def price_A : ℚ := 40
def price_B : ℚ := 120

-- Define the savings calculation
def savings (x y : ℚ) : ℚ :=
  80 * x + 100 * y - (80 * (1 - discount_A) * x + 100 * (1 - discount_B) * y)

-- Theorem statement
theorem product_prices_and_savings :
  equation1 price_A price_B ∧
  equation2 price_A price_B ∧
  savings price_A price_B = 3640 := by
  sorry

end product_prices_and_savings_l1603_160339


namespace cube_sum_of_roots_l1603_160383

theorem cube_sum_of_roots (a b c : ℝ) : 
  (3 * a^3 - 2 * a^2 + 5 * a - 7 = 0) ∧ 
  (3 * b^3 - 2 * b^2 + 5 * b - 7 = 0) ∧ 
  (3 * c^3 - 2 * c^2 + 5 * c - 7 = 0) → 
  a^3 + b^3 + c^3 = 137 / 27 := by
sorry

end cube_sum_of_roots_l1603_160383


namespace estevan_blanket_ratio_l1603_160314

/-- The ratio of polka-dot blankets to total blankets before Estevan's birthday -/
theorem estevan_blanket_ratio :
  let total_blankets : ℕ := 24
  let new_polka_dot_blankets : ℕ := 2
  let total_polka_dot_blankets : ℕ := 10
  let initial_polka_dot_blankets : ℕ := total_polka_dot_blankets - new_polka_dot_blankets
  (initial_polka_dot_blankets : ℚ) / total_blankets = 1 / 3 := by
  sorry

end estevan_blanket_ratio_l1603_160314


namespace equation_solution_l1603_160328

theorem equation_solution (x : ℚ) : 
  (30 * x^2 + 17 = 47 * x - 6) →
  (x = 3/5 ∨ x = 23/36) :=
by sorry

end equation_solution_l1603_160328


namespace triangle_free_edge_bound_l1603_160380

/-- A graph with n vertices and k edges, where no three edges form a triangle -/
structure TriangleFreeGraph where
  n : ℕ  -- number of vertices
  k : ℕ  -- number of edges
  no_triangle : True  -- represents the condition that no three edges form a triangle

/-- Theorem: In a triangle-free graph, the number of edges is at most ⌊n²/4⌋ -/
theorem triangle_free_edge_bound (G : TriangleFreeGraph) : G.k ≤ (G.n^2) / 4 := by
  sorry

end triangle_free_edge_bound_l1603_160380


namespace fraction_equality_l1603_160340

theorem fraction_equality : (1 : ℚ) / 2 + (1 : ℚ) / 4 = 9 / 12 := by
  sorry

end fraction_equality_l1603_160340


namespace annual_population_change_l1603_160376

def town_population (initial_pop : ℕ) (new_people : ℕ) (moved_out : ℕ) (years : ℕ) (final_pop : ℕ) : ℤ :=
  let pop_after_changes : ℤ := initial_pop + new_people - moved_out
  let total_change : ℤ := pop_after_changes - final_pop
  total_change / years

theorem annual_population_change :
  town_population 780 100 400 4 60 = -105 :=
sorry

end annual_population_change_l1603_160376


namespace christine_speed_l1603_160359

/-- Given a distance of 80 miles traveled in 4 hours, prove that the speed is 20 miles per hour. -/
theorem christine_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 4) :
  distance / time = 20 := by
  sorry

end christine_speed_l1603_160359


namespace inequality_proof_l1603_160373

theorem inequality_proof (a b : ℝ) (m : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ m + (1 + b / a) ^ m ≥ 2^(m + 1) := by
  sorry

end inequality_proof_l1603_160373


namespace f_is_K_function_l1603_160334

-- Define a K function
def is_K_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0)

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Theorem stating that f is a K function
theorem f_is_K_function : is_K_function f := by sorry

end f_is_K_function_l1603_160334


namespace water_level_rise_l1603_160381

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) : 
  cube_edge = 5 →
  vessel_length = 10 →
  vessel_width = 5 →
  (cube_edge^3) / (vessel_length * vessel_width) = 2.5 := by
  sorry


end water_level_rise_l1603_160381


namespace incorrect_proposition_l1603_160316

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (different : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem incorrect_proposition
  (α β : Plane) (m n : Line)
  (h1 : different α β)
  (h2 : m ≠ n)
  : ¬(parallel_lines m n ∧ intersect α β m →
      parallel_line_plane n α ∧ parallel_line_plane n β) :=
sorry

end incorrect_proposition_l1603_160316


namespace smallest_a_for_coeff_70_l1603_160317

/-- The coefficient of x^4 in the expansion of (1-3x+ax^2)^8 -/
def coeff_x4 (a : ℝ) : ℝ := 28 * a^2 + 1512 * a + 4725

/-- The problem statement -/
theorem smallest_a_for_coeff_70 :
  ∃ a : ℝ, (∀ b : ℝ, coeff_x4 b = 70 → a ≤ b) ∧ coeff_x4 a = 70 ∧ a = -50 := by
  sorry

end smallest_a_for_coeff_70_l1603_160317


namespace girls_in_school_l1603_160357

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girl_boy_diff : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girl_boy_diff = 20 →
  ∃ (girls : ℕ) (boys : ℕ),
    girls + boys = total_students ∧
    girls * sample_size = (total_students - girls) * (sample_size - girl_boy_diff) ∧
    girls = 720 :=
by sorry

end girls_in_school_l1603_160357


namespace total_candy_cases_l1603_160330

/-- The Sweet Shop's candy inventory -/
structure CandyInventory where
  chocolate_cases : ℕ
  lollipop_cases : ℕ

/-- The total number of candy cases in the inventory -/
def total_cases (inventory : CandyInventory) : ℕ :=
  inventory.chocolate_cases + inventory.lollipop_cases

/-- Theorem: The total number of candy cases is 80 -/
theorem total_candy_cases :
  ∃ (inventory : CandyInventory),
    inventory.chocolate_cases = 25 ∧
    inventory.lollipop_cases = 55 ∧
    total_cases inventory = 80 := by
  sorry

end total_candy_cases_l1603_160330


namespace painters_work_days_l1603_160368

/-- Represents the number of work-days required for a given number of painters to complete a job -/
noncomputable def workDays (numPainters : ℕ) : ℝ :=
  sorry

theorem painters_work_days :
  (workDays 6 = 1.5) →
  (∀ (n m : ℕ), n * workDays n = m * workDays m) →
  workDays 4 = 2.25 :=
by
  sorry

end painters_work_days_l1603_160368


namespace integer_solutions_xy_eq_x_plus_y_l1603_160322

theorem integer_solutions_xy_eq_x_plus_y :
  ∀ x y : ℤ, x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) := by
  sorry

end integer_solutions_xy_eq_x_plus_y_l1603_160322


namespace burger_length_l1603_160335

theorem burger_length (share : ℝ) (h1 : share = 6) : 2 * share = 12 := by
  sorry

#check burger_length

end burger_length_l1603_160335


namespace expression_evaluation_l1603_160310

theorem expression_evaluation : 6^2 + 4*5 - 2^3 + 4^2/2 = 56 := by
  sorry

end expression_evaluation_l1603_160310


namespace five_hundredth_barrel_is_four_l1603_160304

/-- The labeling function for barrels based on their position in the sequence -/
def barrel_label (n : ℕ) : ℕ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | 6 => 10 - 6
  | 7 => 10 - 7
  | 0 => 10 - 8
  | _ => 0  -- This case should never occur due to properties of modulo

/-- The theorem stating that the 500th barrel is labeled 4 -/
theorem five_hundredth_barrel_is_four :
  barrel_label 500 = 4 := by
  sorry


end five_hundredth_barrel_is_four_l1603_160304


namespace card_problem_l1603_160394

theorem card_problem (w x y z : ℝ) : 
  x = w / 2 →
  y = w + x →
  z = 400 →
  w + x + y + z = 1000 →
  w = 200 := by
sorry

end card_problem_l1603_160394


namespace smallest_k_for_720k_square_and_cube_l1603_160374

theorem smallest_k_for_720k_square_and_cube :
  (∀ n : ℕ+, n < 1012500 → ¬(∃ a b : ℕ+, 720 * n = a^2 ∧ 720 * n = b^3)) ∧
  (∃ a b : ℕ+, 720 * 1012500 = a^2 ∧ 720 * 1012500 = b^3) := by
  sorry

end smallest_k_for_720k_square_and_cube_l1603_160374


namespace calculation_proof_l1603_160308

theorem calculation_proof : (-3) / (-1 - 3/4) * (3/4) / (3/7) = 3 := by
  sorry

end calculation_proof_l1603_160308


namespace quadratic_root_value_l1603_160333

theorem quadratic_root_value (d : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + d = 0 ↔ x = (-14 + Real.sqrt 14) / 4 ∨ x = (-14 - Real.sqrt 14) / 4) →
  d = 91/4 := by
sorry

end quadratic_root_value_l1603_160333


namespace thief_hiding_speeds_l1603_160388

/-- Configuration of roads, houses, and police movement --/
structure Configuration where
  road_distance : ℝ
  house_size : ℝ
  house_spacing : ℝ
  house_road_distance : ℝ
  police_speed : ℝ
  police_interval : ℝ

/-- Thief's movement relative to police --/
inductive ThiefMovement
  | Opposite
  | Same

/-- Proposition that the thief can stay hidden --/
def can_stay_hidden (config : Configuration) (thief_speed : ℝ) (direction : ThiefMovement) : Prop :=
  match direction with
  | ThiefMovement.Opposite => thief_speed = 2 * config.police_speed
  | ThiefMovement.Same => thief_speed = config.police_speed / 2

/-- Theorem stating the only two viable speeds for the thief --/
theorem thief_hiding_speeds (config : Configuration) 
  (h1 : config.road_distance = 30)
  (h2 : config.house_size = 10)
  (h3 : config.house_spacing = 20)
  (h4 : config.house_road_distance = 10)
  (h5 : config.police_interval = 90)
  (thief_speed : ℝ)
  (direction : ThiefMovement) :
  can_stay_hidden config thief_speed direction ↔ 
    (thief_speed = 2 * config.police_speed ∧ direction = ThiefMovement.Opposite) ∨
    (thief_speed = config.police_speed / 2 ∧ direction = ThiefMovement.Same) :=
  sorry

end thief_hiding_speeds_l1603_160388


namespace dog_max_distance_l1603_160353

/-- The maximum distance a dog can be from the origin when tied to a post -/
theorem dog_max_distance (post_x post_y rope_length : ℝ) :
  post_x = 6 ∧ post_y = 8 ∧ rope_length = 15 →
  ∃ (max_distance : ℝ),
    max_distance = 25 ∧
    ∀ (x y : ℝ),
      (x - post_x)^2 + (y - post_y)^2 ≤ rope_length^2 →
      x^2 + y^2 ≤ max_distance^2 :=
by sorry

end dog_max_distance_l1603_160353


namespace equal_cost_at_200_unique_equal_cost_l1603_160364

/-- Represents the price per book in yuan -/
def base_price : ℕ := 40

/-- Represents the discount factor for supplier A -/
def discount_a : ℚ := 9/10

/-- Represents the discount factor for supplier B on books over 100 -/
def discount_b : ℚ := 8/10

/-- Represents the threshold for supplier B's discount -/
def threshold : ℕ := 100

/-- Calculates the cost for supplier A given the number of books -/
def cost_a (n : ℕ) : ℚ := n * base_price * discount_a

/-- Calculates the cost for supplier B given the number of books -/
def cost_b (n : ℕ) : ℚ :=
  if n ≤ threshold then n * base_price
  else threshold * base_price + (n - threshold) * base_price * discount_b

/-- Theorem stating that the costs are equal when 200 books are ordered -/
theorem equal_cost_at_200 : cost_a 200 = cost_b 200 := by sorry

/-- Theorem stating that 200 is the unique number of books where costs are equal -/
theorem unique_equal_cost (n : ℕ) : cost_a n = cost_b n ↔ n = 200 := by sorry

end equal_cost_at_200_unique_equal_cost_l1603_160364


namespace point_c_coordinates_l1603_160362

/-- Given point A, vector AB, and vector BC in a 2D Cartesian coordinate system,
    prove that the coordinates of point C are as calculated. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) (AB BC : ℝ × ℝ) :
  A = (0, 1) →
  AB = (-4, -3) →
  BC = (-7, -4) →
  B = (A.1 + AB.1, A.2 + AB.2) →
  C = (B.1 + BC.1, B.2 + BC.2) →
  C = (-11, -6) := by
  sorry

end point_c_coordinates_l1603_160362


namespace triangle_perimeter_from_inradius_and_area_l1603_160337

/-- Given a triangle with inradius 2.0 cm and area 28 cm², its perimeter is 28 cm. -/
theorem triangle_perimeter_from_inradius_and_area :
  ∀ (p : ℝ), 
    (2.0 : ℝ) * p / 2 = 28 →
    p = 28 := by
  sorry

end triangle_perimeter_from_inradius_and_area_l1603_160337


namespace orange_apple_difference_l1603_160309

def apples : ℕ := 14
def dozen : ℕ := 12
def oranges : ℕ := 2 * dozen

theorem orange_apple_difference : oranges - apples = 10 := by
  sorry

end orange_apple_difference_l1603_160309


namespace greatest_integer_inequality_l1603_160342

theorem greatest_integer_inequality (y : ℤ) : (8 : ℚ) / 11 > (y : ℚ) / 15 ↔ y ≤ 10 := by sorry

end greatest_integer_inequality_l1603_160342


namespace quadratic_roots_condition_l1603_160399

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -2 ∧ x₂ > -2 ∧
    x₁^2 + (2*m + 6)*x₁ + 4*m + 12 = 0 ∧
    x₂^2 + (2*m + 6)*x₂ + 4*m + 12 = 0) ↔
  m ≤ -3 :=
by sorry

end quadratic_roots_condition_l1603_160399


namespace min_value_case1_min_value_case2_l1603_160346

/-- The function f(x) defined as x^2 + |x-a| + 1 -/
def f (a x : ℝ) : ℝ := x^2 + |x-a| + 1

/-- The minimum value of f(x) when a ≤ -1/2 and x ≥ a -/
theorem min_value_case1 (a : ℝ) (h : a ≤ -1/2) :
  ∀ x ≥ a, f a x ≥ 3/4 - a :=
sorry

/-- The minimum value of f(x) when a > -1/2 and x ≥ a -/
theorem min_value_case2 (a : ℝ) (h : a > -1/2) :
  ∀ x ≥ a, f a x ≥ a^2 + 1 :=
sorry

end min_value_case1_min_value_case2_l1603_160346


namespace line_segment_endpoint_l1603_160377

/-- Given a line segment with midpoint (2, -3) and one endpoint (3, 1),
    prove that the other endpoint is (1, -7) -/
theorem line_segment_endpoint (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁ = 3 ∧ y₁ = 1) →  -- One endpoint is (3, 1)
  ((x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = -3) →  -- Midpoint is (2, -3)
  x₂ = 1 ∧ y₂ = -7  -- Other endpoint is (1, -7)
  := by sorry

end line_segment_endpoint_l1603_160377


namespace original_equals_scientific_l1603_160384

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 0.056

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 5.6
    exponent := -2
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  original_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l1603_160384


namespace inequality_solution_set_l1603_160301

theorem inequality_solution_set : 
  {x : ℝ | -2 ≤ x ∧ x ≤ 1} = {x : ℝ | 2 - x - x^2 ≥ 0} := by
  sorry

end inequality_solution_set_l1603_160301


namespace g_of_5_l1603_160355

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_of_5 : g 5 = 13 / 7 := by
  sorry

end g_of_5_l1603_160355


namespace dress_price_difference_l1603_160367

theorem dress_price_difference : 
  let original_price := 78.2 / 0.85
  let discounted_price := 78.2
  let final_price := discounted_price * 1.25
  final_price - original_price = 5.75 := by
sorry

end dress_price_difference_l1603_160367


namespace tangent_line_at_x_1_l1603_160329

/-- The function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_at_x_1 : 
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2*x - y + 1 = 0) :=
by sorry

end tangent_line_at_x_1_l1603_160329


namespace fixed_point_power_function_l1603_160386

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

theorem fixed_point_power_function 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (α : ℝ) 
  (h3 : ∀ x : ℝ, (x : ℝ)^α = x^α) -- To ensure g is a power function
  (h4 : (2 : ℝ)^α = 1/4) -- g passes through (2, 1/4)
  : (1/2 : ℝ)^α = 4 := by
  sorry

end fixed_point_power_function_l1603_160386


namespace employee_transfer_theorem_l1603_160369

/-- Represents the number of employees transferred to the tertiary industry -/
def x : ℕ+ := sorry

/-- Represents the profit multiplier for transferred employees -/
def a : ℝ := sorry

/-- The total number of employees -/
def total_employees : ℕ := 1000

/-- The initial average profit per employee per year in ten thousands of yuan -/
def initial_profit : ℝ := 10

/-- The profit increase rate for remaining employees -/
def profit_increase_rate : ℝ := 0.002

/-- The condition that the total profit after transfer is not less than the initial total profit -/
def profit_condition (x : ℕ+) : Prop :=
  (initial_profit * (total_employees - x) * (1 + profit_increase_rate * x) ≥ initial_profit * total_employees)

/-- The condition that the profit from transferred employees is not more than the profit from remaining employees -/
def transfer_condition (x : ℕ+) (a : ℝ) : Prop :=
  (initial_profit * a * x ≤ initial_profit * (total_employees - x) * (1 + profit_increase_rate * x))

theorem employee_transfer_theorem :
  (∃ max_x : ℕ+, max_x = 500 ∧ 
    (∀ y : ℕ+, y > max_x → ¬profit_condition y) ∧
    (∀ y : ℕ+, y ≤ max_x → profit_condition y)) ∧
  (∀ x : ℕ+, x ≤ 500 →
    (∀ a : ℝ, 0 < a ∧ a ≤ 5 → transfer_condition x a) ∧
    (∀ a : ℝ, a > 5 → ¬transfer_condition x a)) :=
sorry

end employee_transfer_theorem_l1603_160369


namespace line_intersects_segment_l1603_160323

/-- A line defined by the equation 2x + y - b = 0 intersects the line segment
    between points (1,0) and (-1,0) if and only if -2 ≤ b ≤ 2. -/
theorem line_intersects_segment (b : ℝ) :
  (∃ (x y : ℝ), 2*x + y - b = 0 ∧ 
    ((x = 1 ∧ y = 0) ∨ 
     (x = -1 ∧ y = 0) ∨ 
     (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ x = -1 + 2*t ∧ y = 0)))
  ↔ -2 ≤ b ∧ b ≤ 2 :=
by sorry

end line_intersects_segment_l1603_160323


namespace plastic_bag_estimate_l1603_160327

def plastic_bag_data : List Nat := [33, 25, 28, 26, 25, 31]
def total_students : Nat := 45

theorem plastic_bag_estimate :
  let average := (plastic_bag_data.sum / plastic_bag_data.length)
  average * total_students = 1260 := by
  sorry

end plastic_bag_estimate_l1603_160327


namespace computer_price_increase_l1603_160302

theorem computer_price_increase (x : ℝ) (h : x + 0.3 * x = 351) : x + 351 = 621 := by
  sorry

end computer_price_increase_l1603_160302


namespace set_relationships_evaluation_l1603_160390

theorem set_relationships_evaluation :
  let s1 : Set (Set ℕ) := {{0}, {2, 3, 4}}
  let s2 : Set ℕ := {0}
  let s3 : Set ℤ := {-1, 0, 1}
  let s4 : Set ℤ := {0, -1, 1}
  ({0} ∈ s1) = false ∧
  (∅ ⊆ s2) = true ∧
  (s3 = s4) = true ∧
  (0 ∈ (∅ : Set ℕ)) = false :=
by sorry

end set_relationships_evaluation_l1603_160390


namespace complex_magnitude_l1603_160371

theorem complex_magnitude (r s : ℝ) (z : ℂ) 
  (h1 : |r| < 4) 
  (h2 : s ≠ 0) 
  (h3 : s * z + 1 / z = r) : 
  Complex.abs z = Real.sqrt (2 * (r^2 - 2*s) + 2*r * Real.sqrt (r^2 - 4*s)) / (2 * |s|) :=
sorry

end complex_magnitude_l1603_160371


namespace functional_equation_solution_l1603_160312

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x - f y) = f (f y) + x * f x + x^2) ↔ 
  (∀ x : ℝ, f x = 1 - x^2 / 2) :=
by sorry

end functional_equation_solution_l1603_160312


namespace f_inequality_l1603_160365

noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1))

theorem f_inequality : f (1 / Real.exp 1) < f 0 ∧ f 0 < f (Real.exp 1) := by
  sorry

end f_inequality_l1603_160365


namespace roy_pens_count_l1603_160303

def blue_pens : ℕ := 5

def black_pens : ℕ := 3 * blue_pens

def red_pens : ℕ := 2 * black_pens - 4

def total_pens : ℕ := blue_pens + black_pens + red_pens

theorem roy_pens_count : total_pens = 46 := by
  sorry

end roy_pens_count_l1603_160303


namespace inequality_solution_set_l1603_160370

theorem inequality_solution_set :
  {x : ℝ | x * |x - 1| > 0} = (Set.Ioo 0 1) ∪ (Set.Ioi 1) := by
  sorry

end inequality_solution_set_l1603_160370


namespace fruit_mix_grapes_l1603_160382

theorem fruit_mix_grapes (b r g c : ℚ) : 
  b + r + g + c = 400 →
  r = 3 * b →
  g = 2 * c →
  c = 5 * r →
  g = 12000 / 49 := by
sorry

end fruit_mix_grapes_l1603_160382


namespace couscous_per_dish_l1603_160306

theorem couscous_per_dish 
  (shipment1 : ℕ) 
  (shipment2 : ℕ) 
  (shipment3 : ℕ) 
  (num_dishes : ℕ) 
  (h1 : shipment1 = 7)
  (h2 : shipment2 = 13)
  (h3 : shipment3 = 45)
  (h4 : num_dishes = 13) :
  (shipment1 + shipment2 + shipment3) / num_dishes = 5 := by
  sorry

#check couscous_per_dish

end couscous_per_dish_l1603_160306


namespace power_of_power_l1603_160395

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1603_160395


namespace sector_central_angle_l1603_160351

theorem sector_central_angle (circumference area : ℝ) (h_circ : circumference = 6) (h_area : area = 2) :
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 2 * r + l = circumference ∧ (1 / 2) * r * l = area ∧
  (l / r = 1 ∨ l / r = 4) :=
sorry

end sector_central_angle_l1603_160351


namespace birdseed_mixture_cost_per_pound_l1603_160348

theorem birdseed_mixture_cost_per_pound
  (millet_weight : ℝ)
  (millet_cost_per_lb : ℝ)
  (sunflower_weight : ℝ)
  (sunflower_cost_per_lb : ℝ)
  (h1 : millet_weight = 100)
  (h2 : millet_cost_per_lb = 0.60)
  (h3 : sunflower_weight = 25)
  (h4 : sunflower_cost_per_lb = 1.10) :
  (millet_weight * millet_cost_per_lb + sunflower_weight * sunflower_cost_per_lb) /
  (millet_weight + sunflower_weight) = 0.70 := by
  sorry

#check birdseed_mixture_cost_per_pound

end birdseed_mixture_cost_per_pound_l1603_160348


namespace prove_a_equals_two_l1603_160356

/-- Given a > 1 and f(x) = a^x + 1, prove that a = 2 if f(2) - f(1) = 2 -/
theorem prove_a_equals_two (a : ℝ) (h1 : a > 1) : 
  (fun x => a^x + 1) 2 - (fun x => a^x + 1) 1 = 2 → a = 2 := by
sorry

end prove_a_equals_two_l1603_160356


namespace percentage_with_neither_is_twenty_percent_l1603_160320

/-- Represents the study of adults in a neighborhood -/
structure NeighborhoodStudy where
  total : ℕ
  insomnia : ℕ
  migraines : ℕ
  both : ℕ

/-- Calculates the percentage of adults with neither insomnia nor migraines -/
def percentageWithNeither (study : NeighborhoodStudy) : ℚ :=
  let withNeither := study.total - (study.insomnia + study.migraines - study.both)
  (withNeither : ℚ) / study.total * 100

/-- The main theorem stating that the percentage of adults with neither condition is 20% -/
theorem percentage_with_neither_is_twenty_percent (study : NeighborhoodStudy)
  (h_total : study.total = 150)
  (h_insomnia : study.insomnia = 90)
  (h_migraines : study.migraines = 60)
  (h_both : study.both = 30) :
  percentageWithNeither study = 20 := by
  sorry

#eval percentageWithNeither { total := 150, insomnia := 90, migraines := 60, both := 30 }

end percentage_with_neither_is_twenty_percent_l1603_160320


namespace min_value_parallel_vectors_l1603_160363

/-- Given vectors a and b, where a is parallel to b, prove the minimum value of 1/m + 8/n is 9/2 -/
theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ (k : ℝ), a = k • b) → 
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 4 → 1/x + 8/y ≥ 9/2) :=
by sorry

end min_value_parallel_vectors_l1603_160363


namespace olympic_inequalities_l1603_160300

/-- Given positive real numbers a, b, c, d such that a + b + c + d = 3,
    prove the following inequalities:
    1. (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2) ≤ 1/(a^2*b^2*c^2*d^2)
    2. (1/a^3 + 1/b^3 + 1/c^3 + 1/d^3) ≤ 1/(a^3*b^3*c^3*d^3) -/
theorem olympic_inequalities (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 3) :
  (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2)) ∧
  (1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a^3*b^3*c^3*d^3)) := by
  sorry

end olympic_inequalities_l1603_160300
