import Mathlib

namespace bounded_sequence_l3719_371901

/-- A sequence defined recursively with a parameter c -/
def x (c : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => (x c n)^2 + c

/-- The theorem stating the condition for boundedness of the sequence -/
theorem bounded_sequence (c : ℝ) (h : c > 0) :
  (∀ n, |x c n| < 2016) ↔ c ≤ 1/4 := by
  sorry

end bounded_sequence_l3719_371901


namespace cobbler_hourly_rate_l3719_371972

theorem cobbler_hourly_rate 
  (mold_cost : ℝ) 
  (work_hours : ℝ) 
  (discount_rate : ℝ) 
  (total_payment : ℝ) 
  (h1 : mold_cost = 250)
  (h2 : work_hours = 8)
  (h3 : discount_rate = 0.8)
  (h4 : total_payment = 730) :
  ∃ hourly_rate : ℝ, 
    hourly_rate = 75 ∧ 
    total_payment = mold_cost + discount_rate * work_hours * hourly_rate :=
by
  sorry

end cobbler_hourly_rate_l3719_371972


namespace log_difference_equals_two_l3719_371913

theorem log_difference_equals_two :
  (Real.log 80 / Real.log 2) / (Real.log 2 / Real.log 40) -
  (Real.log 160 / Real.log 2) / (Real.log 2 / Real.log 20) = 2 := by
  sorry

end log_difference_equals_two_l3719_371913


namespace system_solution_l3719_371947

theorem system_solution :
  let x : ℚ := 57 / 31
  let y : ℚ := 195 / 62
  (3 * x - 4 * y = -7) ∧ (4 * x + 5 * y = 23) := by
  sorry

end system_solution_l3719_371947


namespace expression_factorization_l3719_371982

theorem expression_factorization (x : ℚ) :
  (x^2 - 3*x + 2) - (x^2 - x + 6) + (x - 1)*(x - 2) + x^2 + 2 = (2*x - 1)*(x - 2) := by
  sorry

end expression_factorization_l3719_371982


namespace gcf_90_150_l3719_371918

theorem gcf_90_150 : Nat.gcd 90 150 = 30 := by
  sorry

end gcf_90_150_l3719_371918


namespace intersection_of_logarithmic_functions_l3719_371992

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) := by sorry

end intersection_of_logarithmic_functions_l3719_371992


namespace boat_race_spacing_l3719_371967

theorem boat_race_spacing (river_width : ℝ) (num_boats : ℕ) (boat_width : ℝ)
  (hw : river_width = 42)
  (hn : num_boats = 8)
  (hb : boat_width = 3) :
  (river_width - num_boats * boat_width) / (num_boats + 1) = 2 :=
by sorry

end boat_race_spacing_l3719_371967


namespace sum_of_numbers_l3719_371916

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 5 / y) : x + y = 24 * Real.sqrt 5 / 5 := by
  sorry

end sum_of_numbers_l3719_371916


namespace derivative_at_negative_one_l3719_371964

/-- Given a function f(x) = x², prove that the derivative of f at x = -1 is -2. -/
theorem derivative_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = x^2) :
  deriv f (-1) = -2 := by
  sorry

end derivative_at_negative_one_l3719_371964


namespace store_discount_percentage_l3719_371933

theorem store_discount_percentage (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.20 * C
  let new_year_price := 1.25 * initial_price
  let final_price := 1.32 * C
  let discount_percentage := (new_year_price - final_price) / new_year_price * 100
  discount_percentage = 12 := by sorry

end store_discount_percentage_l3719_371933


namespace response_rate_percentage_l3719_371954

def responses_needed : ℕ := 300
def questionnaires_mailed : ℕ := 600

theorem response_rate_percentage : 
  (responses_needed : ℚ) / questionnaires_mailed * 100 = 50 := by
  sorry

end response_rate_percentage_l3719_371954


namespace alcohol_percentage_problem_l3719_371953

theorem alcohol_percentage_problem (initial_volume : Real) 
  (added_alcohol : Real) (final_percentage : Real) :
  initial_volume = 6 →
  added_alcohol = 3.6 →
  final_percentage = 50 →
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := final_volume * (final_percentage / 100)
  let initial_alcohol := final_alcohol - added_alcohol
  initial_alcohol / initial_volume * 100 = 20 := by
sorry


end alcohol_percentage_problem_l3719_371953


namespace polynomial_roots_l3719_371949

/-- The polynomial x^3 - 3x^2 - x + 3 -/
def p (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- The roots of the polynomial -/
def roots : Set ℝ := {1, -1, 3}

theorem polynomial_roots : 
  (∀ x ∈ roots, p x = 0) ∧ 
  (∀ x : ℝ, p x = 0 → x ∈ roots) := by
  sorry

end polynomial_roots_l3719_371949


namespace garden_radius_increase_l3719_371965

theorem garden_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 40)
  (h2 : final_circumference = 50) :
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end garden_radius_increase_l3719_371965


namespace alternating_hexagon_area_l3719_371960

/-- A hexagon with alternating side lengths and specified corner triangles -/
structure AlternatingHexagon where
  short_side : ℝ
  long_side : ℝ
  corner_triangle_base : ℝ
  corner_triangle_altitude : ℝ

/-- The area of an alternating hexagon -/
def area (h : AlternatingHexagon) : ℝ := sorry

/-- Theorem: The area of the specified hexagon is 36 square units -/
theorem alternating_hexagon_area :
  let h : AlternatingHexagon := {
    short_side := 2,
    long_side := 4,
    corner_triangle_base := 2,
    corner_triangle_altitude := 3
  }
  area h = 36 := by sorry

end alternating_hexagon_area_l3719_371960


namespace chord_length_is_two_l3719_371958

/-- The chord length intercepted by y = 1 - x on x² + y² + 2y - 2 = 0 is 2 -/
theorem chord_length_is_two (x y : ℝ) : 
  (x^2 + y^2 + 2*y - 2 = 0) → 
  (y = 1 - x) → 
  ∃ (a b : ℝ), (a^2 + b^2 = 1) ∧ 
               ((x - a)^2 + (y - b)^2 = 2^2 / 4) :=
by sorry

end chord_length_is_two_l3719_371958


namespace bumper_car_line_problem_l3719_371971

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 4 + 8 = 11) → initial_people = 7 := by
  sorry

end bumper_car_line_problem_l3719_371971


namespace f_of_two_equals_five_l3719_371928

/-- Given a function f(x) = x^2 + 2x - 3, prove that f(2) = 5 -/
theorem f_of_two_equals_five (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x - 3) : f 2 = 5 := by
  sorry

end f_of_two_equals_five_l3719_371928


namespace square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven_l3719_371959

theorem square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven :
  ∀ a : ℝ, a = Real.sqrt 11 - 1 → a^2 + 2*a + 1 = 11 := by
  sorry

end square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven_l3719_371959


namespace not_necessary_nor_sufficient_l3719_371988

theorem not_necessary_nor_sufficient : ∃ (x y : ℝ), 
  ((x / y > 1 ∧ x ≤ y) ∨ (x / y ≤ 1 ∧ x > y)) := by
  sorry

end not_necessary_nor_sufficient_l3719_371988


namespace determinant_equality_l3719_371920

theorem determinant_equality (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 3 → 
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 3 := by
sorry

end determinant_equality_l3719_371920


namespace compound_interest_rate_l3719_371984

/-- Compound interest calculation -/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (I : ℝ) (r : ℝ) : 
  P = 6000 → t = 2 → I = 1260.000000000001 → 
  P * (1 + r)^t = P + I → r = 0.1 := by
  sorry

end compound_interest_rate_l3719_371984


namespace lice_check_time_proof_l3719_371903

theorem lice_check_time_proof (kindergarteners first_graders second_graders third_graders : ℕ)
  (time_per_check : ℕ) (h1 : kindergarteners = 26) (h2 : first_graders = 19)
  (h3 : second_graders = 20) (h4 : third_graders = 25) (h5 : time_per_check = 2) :
  (kindergarteners + first_graders + second_graders + third_graders) * time_per_check / 60 = 3 :=
by sorry

end lice_check_time_proof_l3719_371903


namespace machine_chip_production_l3719_371978

/-- The number of computer chips produced by a machine in a day, given the number of
    video game consoles it can supply chips for and the number of chips per console. -/
def chips_per_day (consoles_per_day : ℕ) (chips_per_console : ℕ) : ℕ :=
  consoles_per_day * chips_per_console

/-- Theorem stating that a machine supplying chips for 93 consoles per day,
    with 5 chips per console, produces 465 chips per day. -/
theorem machine_chip_production :
  chips_per_day 93 5 = 465 := by
  sorry

end machine_chip_production_l3719_371978


namespace chair_cost_l3719_371952

/-- Proves that the cost of one chair is $11 given the conditions of Nadine's garage sale purchase. -/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  ∃ (chair_cost : ℕ), 
    chair_cost * num_chairs = total_spent - table_cost ∧
    chair_cost = 11 :=
by sorry

end chair_cost_l3719_371952


namespace decimal_to_fraction_l3719_371999

theorem decimal_to_fraction : (2.25 : ℚ) = 9 / 4 := by sorry

end decimal_to_fraction_l3719_371999


namespace solution_set_when_a_is_one_range_of_a_for_f_geq_three_halves_l3719_371936

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |a*x + 1| + |x - a|
def g (x : ℝ) : ℝ := x^2 + x

-- State the theorems
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, g x ≥ f 1 x ↔ x ≤ -3 ∨ x ≥ 1 := by sorry

theorem range_of_a_for_f_geq_three_halves :
  (∀ x : ℝ, f a x ≥ 3/2) → a ≥ Real.sqrt 2 / 2 := by sorry

-- Note: We assume 'a' is positive as given in the original problem
variable (a : ℝ) (ha : a > 0)

end solution_set_when_a_is_one_range_of_a_for_f_geq_three_halves_l3719_371936


namespace max_cashback_selection_l3719_371941

structure Category where
  name : String
  cashback : Float
  expenses : Float

def calculate_cashback (c : Category) : Float :=
  c.cashback * c.expenses / 100

def total_cashback (categories : List Category) : Float :=
  categories.map calculate_cashback |> List.sum

theorem max_cashback_selection (categories : List Category) :
  let transport := { name := "Transport", cashback := 5, expenses := 2000 }
  let groceries := { name := "Groceries", cashback := 3, expenses := 5000 }
  let clothing := { name := "Clothing", cashback := 4, expenses := 3000 }
  let entertainment := { name := "Entertainment", cashback := 5, expenses := 3000 }
  let sports := { name := "Sports", cashback := 6, expenses := 1500 }
  let all_categories := [transport, groceries, clothing, entertainment, sports]
  let best_selection := [groceries, entertainment, clothing]
  categories = all_categories →
  (∀ selection : List Category,
    selection.length ≤ 3 →
    selection ⊆ categories →
    total_cashback selection ≤ total_cashback best_selection) :=
by sorry

end max_cashback_selection_l3719_371941


namespace max_colors_is_six_l3719_371926

/-- A cube is a structure with edges and a coloring function. -/
structure Cube where
  edges : Finset (Fin 12)
  coloring : Fin 12 → Nat

/-- Two edges are adjacent if they share a common vertex. -/
def adjacent (e1 e2 : Fin 12) : Prop := sorry

/-- A valid coloring satisfies the problem conditions. -/
def valid_coloring (c : Cube) : Prop :=
  ∀ (color1 color2 : Nat), color1 ≠ color2 →
    ∃ (e1 e2 : Fin 12), adjacent e1 e2 ∧ c.coloring e1 = color1 ∧ c.coloring e2 = color2

/-- The maximum number of colors that can be used. -/
def max_colors (c : Cube) : Nat :=
  Finset.card (Finset.image c.coloring c.edges)

/-- The main theorem: The maximum number of colors is 6. -/
theorem max_colors_is_six (c : Cube) (h : valid_coloring c) : max_colors c = 6 := by
  sorry

end max_colors_is_six_l3719_371926


namespace problem_solution_l3719_371995

theorem problem_solution (x y : ℝ) : 
  let A := 2 * x^2 - x + y - 3 * x * y
  let B := x^2 - 2 * x - y + x * y
  (A - 2 * B = 3 * x + 3 * y - 5 * x * y) ∧ 
  (x + y = 4 → x * y = -1/5 → A - 2 * B = 13) := by
sorry

end problem_solution_l3719_371995


namespace max_value_on_ellipse_l3719_371940

/-- Given a curve C with equation 4x^2 + 9y^2 = 36, 
    the maximum value of 3x + 4y for any point (x,y) on C is √145. -/
theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 145 ∧ 
  (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 → 3 * x + 4 * y ≤ M) ∧
  (∃ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 ∧ 3 * x + 4 * y = M) := by
  sorry

end max_value_on_ellipse_l3719_371940


namespace quadratic_inequality_solution_l3719_371908

theorem quadratic_inequality_solution (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, (a * x^2 + b * x + 2 < 0) ↔ (x < -1/2 ∨ x > 1/3)) →
  (a - b) / a = 5/6 := by
  sorry

end quadratic_inequality_solution_l3719_371908


namespace onesDigit_73_pow_355_l3719_371997

-- Define a function to get the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem onesDigit_73_pow_355 : onesDigit (73^355) = 7 := by
  sorry

end onesDigit_73_pow_355_l3719_371997


namespace prob_at_least_one_female_l3719_371951

/-- The probability of selecting at least one female student when randomly choosing 2 students
    from a group of 3 males and 1 female is equal to 1/2. -/
theorem prob_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (team_size : ℕ) (h1 : total_students = male_students + female_students) 
  (h2 : male_students = 3) (h3 : female_students = 1) (h4 : team_size = 2) :
  1 - (Nat.choose male_students team_size : ℚ) / (Nat.choose total_students team_size : ℚ) = 1/2 :=
sorry

end prob_at_least_one_female_l3719_371951


namespace cross_section_distance_in_pyramid_l3719_371956

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Theorem about the distance of a cross section in a right hexagonal pyramid -/
theorem cross_section_distance_in_pyramid 
  (pyramid : RightHexagonalPyramid)
  (section1 section2 : CrossSection) :
  section1.area = 216 * Real.sqrt 3 →
  section2.area = 486 * Real.sqrt 3 →
  |section1.distance_from_apex - section2.distance_from_apex| = 8 →
  section2.distance_from_apex = 24 :=
by sorry

end cross_section_distance_in_pyramid_l3719_371956


namespace gcd_lcm_sum_8_12_l3719_371944

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l3719_371944


namespace girls_who_left_l3719_371996

theorem girls_who_left (initial_boys : ℕ) (initial_girls : ℕ) (final_students : ℕ) :
  initial_boys = 24 →
  initial_girls = 14 →
  final_students = 30 →
  ∃ (left_girls : ℕ),
    left_girls = initial_girls - (final_students - (initial_boys - left_girls)) ∧
    left_girls = 4 := by
  sorry

end girls_who_left_l3719_371996


namespace proportionality_analysis_l3719_371904

/-- Represents a relationship between x and y -/
inductive Relationship
  | DirectlyProportional
  | InverselyProportional
  | Neither

/-- Determines the relationship between x and y given an equation -/
def determineRelationship (equation : ℝ → ℝ → Prop) : Relationship :=
  sorry

/-- Equation A: x + y = 0 -/
def equationA (x y : ℝ) : Prop := x + y = 0

/-- Equation B: 3xy = 10 -/
def equationB (x y : ℝ) : Prop := 3 * x * y = 10

/-- Equation C: x = 5y -/
def equationC (x y : ℝ) : Prop := x = 5 * y

/-- Equation D: x^2 + 3x + y = 10 -/
def equationD (x y : ℝ) : Prop := x^2 + 3*x + y = 10

/-- Equation E: x/y = √3 -/
def equationE (x y : ℝ) : Prop := x / y = Real.sqrt 3

theorem proportionality_analysis :
  (determineRelationship equationA = Relationship.DirectlyProportional) ∧
  (determineRelationship equationB = Relationship.InverselyProportional) ∧
  (determineRelationship equationC = Relationship.DirectlyProportional) ∧
  (determineRelationship equationD = Relationship.Neither) ∧
  (determineRelationship equationE = Relationship.DirectlyProportional) :=
by
  sorry

end proportionality_analysis_l3719_371904


namespace parabola_c_value_l3719_371924

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a parabola, its vertex, and a point it passes through, prove that c = 9/2 -/
theorem parabola_c_value (p : Parabola) (h1 : p.a * (-1)^2 + p.b * (-1) + p.c = 5)
    (h2 : p.a * 1^2 + p.b * 1 + p.c = 3) : p.c = 9/2 := by
  sorry


end parabola_c_value_l3719_371924


namespace original_price_l3719_371950

theorem original_price (p q : ℝ) (h : p ≠ 0 ∧ q ≠ 0) :
  let x := (20000 : ℝ) / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  let final_price := x * (1 + p/100) * (1 + q/100) * (1 - q/100) * (1 - p/100)
  final_price = 2 := by sorry

end original_price_l3719_371950


namespace ellipse_chord_slope_l3719_371907

/-- The slope of a chord in an ellipse with given midpoint -/
theorem ellipse_chord_slope (x y : ℝ) :
  (x^2 / 16 + y^2 / 9 = 1) →  -- ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / 16 + y₁^2 / 9 = 1 ∧  -- endpoint 1 on ellipse
    x₂^2 / 16 + y₂^2 / 9 = 1 ∧  -- endpoint 2 on ellipse
    (x₁ + x₂) / 2 = -1 ∧        -- x-coordinate of midpoint
    (y₁ + y₂) / 2 = 2 ∧         -- y-coordinate of midpoint
    (y₂ - y₁) / (x₂ - x₁) = 9 / 32) -- slope of chord
  := by sorry

end ellipse_chord_slope_l3719_371907


namespace coffee_milk_problem_l3719_371934

/-- Represents the liquid mixture in a cup -/
structure Mixture where
  coffee : ℚ
  milk : ℚ

/-- The process of mixing and transferring liquids -/
def mix_and_transfer (coffee_cup milk_cup : Mixture) : Mixture :=
  let transferred_coffee := coffee_cup.coffee / 3
  let mixed_cup := Mixture.mk (milk_cup.coffee + transferred_coffee) milk_cup.milk
  let total_mixed := mixed_cup.coffee + mixed_cup.milk
  let transferred_back := total_mixed / 2
  let coffee_ratio := mixed_cup.coffee / total_mixed
  let milk_ratio := mixed_cup.milk / total_mixed
  Mixture.mk 
    (coffee_cup.coffee - transferred_coffee + transferred_back * coffee_ratio)
    (transferred_back * milk_ratio)

theorem coffee_milk_problem :
  let initial_coffee_cup := Mixture.mk 6 0
  let initial_milk_cup := Mixture.mk 0 3
  let final_coffee_cup := mix_and_transfer initial_coffee_cup initial_milk_cup
  final_coffee_cup.milk / (final_coffee_cup.coffee + final_coffee_cup.milk) = 3 / 13 := by
  sorry

end coffee_milk_problem_l3719_371934


namespace sequence_general_term_l3719_371979

theorem sequence_general_term 
  (a : ℕ+ → ℝ) 
  (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = 3 * n.val ^ 2 - 2 * n.val) :
  ∀ n : ℕ+, a n = 6 * n.val - 5 :=
sorry

end sequence_general_term_l3719_371979


namespace boys_camp_total_l3719_371976

theorem boys_camp_total (total : ℕ) : 
  (total * 20 / 100 : ℕ) > 0 →  -- Ensure there are boys from school A
  (((total * 20 / 100) * 70 / 100 : ℕ) = 35) →  -- 35 boys from school A not studying science
  total = 250 := by
sorry

end boys_camp_total_l3719_371976


namespace arrangement_counts_l3719_371946

/-- Counts the number of valid arrangements of crosses and zeros -/
def countArrangements (n : ℕ) (zeros : ℕ) : ℕ :=
  sorry

theorem arrangement_counts :
  (countArrangements 29 14 = 15) ∧ (countArrangements 28 14 = 120) := by
  sorry

end arrangement_counts_l3719_371946


namespace solve_equation_l3719_371990

theorem solve_equation : ∃ x : ℝ, 3 * x + 15 = (1/3) * (6 * x + 45) ∧ x = 0 := by
  sorry

end solve_equation_l3719_371990


namespace point_inside_circle_l3719_371909

theorem point_inside_circle (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 ↔ -1 < a ∧ a < 1 := by
  sorry

end point_inside_circle_l3719_371909


namespace existence_implies_lower_bound_l3719_371970

theorem existence_implies_lower_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a ≤ a*x - 3) → a ≥ 7 := by
  sorry

end existence_implies_lower_bound_l3719_371970


namespace cubic_equation_roots_l3719_371977

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 38 := by
sorry

end cubic_equation_roots_l3719_371977


namespace simplify_expression_l3719_371962

theorem simplify_expression (a b c : ℝ) (h : a * b ≠ c^2) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - c^2) = a / b + 1 := by
  sorry

end simplify_expression_l3719_371962


namespace progress_primary_grade3_students_l3719_371945

/-- The number of students in Grade 3 of Progress Primary School -/
def total_students (num_classes : ℕ) (special_class_size : ℕ) (regular_class_size : ℕ) : ℕ :=
  special_class_size + (num_classes - 1) * regular_class_size

/-- Theorem stating the total number of students in Grade 3 of Progress Primary School -/
theorem progress_primary_grade3_students :
  total_students 10 48 50 = 48 + 9 * 50 := by
  sorry

end progress_primary_grade3_students_l3719_371945


namespace at_least_one_not_less_than_two_l3719_371993

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l3719_371993


namespace ones_and_seven_primality_l3719_371917

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def ones_and_seven (n : ℕ) : ℕ :=
  if n = 1 then 7
  else (10^(n-1) - 1) / 9 + 7 * 10^((n-1) / 2)

theorem ones_and_seven_primality (n : ℕ) :
  is_prime (ones_and_seven n) ↔ n = 1 ∨ n = 2 :=
sorry

end ones_and_seven_primality_l3719_371917


namespace largest_multiple_of_3_and_5_under_800_l3719_371939

theorem largest_multiple_of_3_and_5_under_800 : 
  ∀ n : ℕ, n < 800 ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ 795 :=
by sorry

end largest_multiple_of_3_and_5_under_800_l3719_371939


namespace modular_congruence_solution_l3719_371987

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -3457 [ZMOD 13] ∧ n = 1 := by
  sorry

end modular_congruence_solution_l3719_371987


namespace mobius_speed_theorem_l3719_371911

theorem mobius_speed_theorem (total_distance : ℝ) (loaded_speed : ℝ) (total_time : ℝ) (rest_time : ℝ) :
  total_distance = 286 →
  loaded_speed = 11 →
  total_time = 26 →
  rest_time = 2 →
  ∃ v : ℝ, v > 0 ∧ (total_distance / 2) / loaded_speed + (total_distance / 2) / v = total_time - rest_time ∧ v = 13 := by
  sorry

end mobius_speed_theorem_l3719_371911


namespace soccer_team_bottles_l3719_371966

theorem soccer_team_bottles (total_bottles : ℕ) (football_players : ℕ) (football_bottles_per_player : ℕ)
  (lacrosse_extra_bottles : ℕ) (rugby_bottles : ℕ) :
  total_bottles = 254 →
  football_players = 11 →
  football_bottles_per_player = 6 →
  lacrosse_extra_bottles = 12 →
  rugby_bottles = 49 →
  total_bottles - (football_players * football_bottles_per_player + 
    (football_players * football_bottles_per_player + lacrosse_extra_bottles) + 
    rugby_bottles) = 61 := by
  sorry

#check soccer_team_bottles

end soccer_team_bottles_l3719_371966


namespace trapezoid_sides_l3719_371938

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithCircle where
  r : ℝ  -- radius of the inscribed circle
  a : ℝ  -- shorter base
  b : ℝ  -- longer base
  c : ℝ  -- left side
  d : ℝ  -- right side (hypotenuse)
  h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r > 0  -- all lengths are positive
  ha : a = 4*r/3  -- shorter base condition
  hsum : a + b = c + d  -- sum of bases equals sum of non-parallel sides
  hright : c^2 + a^2 = d^2  -- right angle condition

/-- The sides of the trapezoid are 2r, 4r/3, 10r/3, and 4r -/
theorem trapezoid_sides (t : RightTrapezoidWithCircle) :
  t.c = 2*t.r ∧ t.a = 4*t.r/3 ∧ t.b = 10*t.r/3 ∧ t.d = 4*t.r :=
sorry

end trapezoid_sides_l3719_371938


namespace mango_dishes_l3719_371955

theorem mango_dishes (total_dishes : ℕ) (mango_salsa_dishes : ℕ) (mango_jelly_dishes : ℕ) (oliver_willing_dishes : ℕ) :
  total_dishes = 36 →
  mango_salsa_dishes = 3 →
  mango_jelly_dishes = 1 →
  oliver_willing_dishes = 28 →
  let fresh_mango_dishes : ℕ := total_dishes / 6
  let pickable_fresh_mango_dishes : ℕ := total_dishes - (oliver_willing_dishes + mango_salsa_dishes + mango_jelly_dishes)
  pickable_fresh_mango_dishes = 4 := by
sorry

end mango_dishes_l3719_371955


namespace remainder_proof_l3719_371985

theorem remainder_proof : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end remainder_proof_l3719_371985


namespace false_proposition_implies_plane_plane_line_l3719_371905

-- Define geometric figures
inductive GeometricFigure
  | Line
  | Plane

-- Define perpendicular and parallel relations
def perpendicular (a b : GeometricFigure) : Prop := sorry
def parallel (a b : GeometricFigure) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricFigure) : Prop :=
  perpendicular x y → parallel y z → perpendicular x z

-- Theorem statement
theorem false_proposition_implies_plane_plane_line :
  ∀ x y z : GeometricFigure,
  ¬(proposition x y z) →
  (x = GeometricFigure.Plane ∧ y = GeometricFigure.Plane ∧ z = GeometricFigure.Line) :=
sorry

end false_proposition_implies_plane_plane_line_l3719_371905


namespace equation_solution_l3719_371991

theorem equation_solution (x y : ℝ) 
  (hx0 : x ≠ 0) (hx3 : x ≠ 3) (hy0 : y ≠ 0) (hy4 : y ≠ 4)
  (h_eq : 3 / x + 2 / y = 5 / 6) :
  x = 18 * y / (5 * y - 12) := by
sorry

end equation_solution_l3719_371991


namespace root_sum_reciprocal_products_l3719_371932

theorem root_sum_reciprocal_products (p q r s t : ℂ) : 
  p^5 - 4*p^4 + 7*p^3 - 3*p^2 + p - 1 = 0 →
  q^5 - 4*q^4 + 7*q^3 - 3*q^2 + q - 1 = 0 →
  r^5 - 4*r^4 + 7*r^3 - 3*r^2 + r - 1 = 0 →
  s^5 - 4*s^4 + 7*s^3 - 3*s^2 + s - 1 = 0 →
  t^5 - 4*t^4 + 7*t^3 - 3*t^2 + t - 1 = 0 →
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 → t ≠ 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 7 := by
sorry

end root_sum_reciprocal_products_l3719_371932


namespace nested_fraction_evaluation_l3719_371989

theorem nested_fraction_evaluation : 
  2 + 3 / (4 + 5 / (6 + 7/8)) = 137/52 := by
  sorry

end nested_fraction_evaluation_l3719_371989


namespace profit_difference_theorem_l3719_371919

/-- Represents the profit distribution for a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of a and c --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.a_investment + pd.b_investment + pd.c_investment
  let total_profit := pd.b_profit * total_investment / pd.b_investment
  let a_profit := total_profit * pd.a_investment / total_investment
  let c_profit := total_profit * pd.c_investment / total_investment
  c_profit - a_profit

/-- Theorem stating the difference between profit shares of a and c --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 3500) :
  profit_difference pd = 1400 := by
  sorry

end profit_difference_theorem_l3719_371919


namespace scientific_notation_of_899000_l3719_371981

/-- Theorem: 899,000 expressed in scientific notation is 8.99 × 10^5 -/
theorem scientific_notation_of_899000 :
  899000 = 8.99 * (10 ^ 5) := by
  sorry

end scientific_notation_of_899000_l3719_371981


namespace triangle_problem_l3719_371931

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  c > b →
  a = Real.sqrt 21 →
  S = Real.sqrt 3 →
  (1/2) * b * c * Real.sin A = S →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (∃ (B C : ℝ), A + B + C = π ∧ 
    a / Real.sin A = b / Real.sin B ∧
    Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)) →
  (b = 1 ∧ c = 4) ∧
  Real.sin B + Real.cos C = (Real.sqrt 7 + 2 * Real.sqrt 21) / 14 := by
  sorry


end triangle_problem_l3719_371931


namespace expected_worth_unfair_coin_l3719_371915

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ
  fixed_cost : ℝ

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails) - c.fixed_cost

/-- Theorem: The expected worth of the specific unfair coin is -1/3 -/
theorem expected_worth_unfair_coin :
  let c : UnfairCoin := {
    prob_heads := 1/3,
    prob_tails := 2/3,
    gain_heads := 6,
    loss_tails := 2,
    fixed_cost := 1
  }
  expected_worth c = -1/3 := by
  sorry

end expected_worth_unfair_coin_l3719_371915


namespace probability_matching_letter_l3719_371910

def word1 : String := "MATHEMATICS"
def word2 : String := "CALCULUS"

def is_in_word2 (c : Char) : Bool :=
  word2.contains c

def count_matching_letters : Nat :=
  word1.toList.filter is_in_word2 |>.length

theorem probability_matching_letter :
  (count_matching_letters : ℚ) / word1.length = 3 / 8 :=
sorry

end probability_matching_letter_l3719_371910


namespace line_parallel_perp_plane_implies_perp_line_l3719_371998

/-- In three-dimensional space -/
structure Space :=
  (points : Type*)
  (vectors : Type*)

/-- A line in space -/
structure Line (S : Space) :=
  (point : S.points)
  (direction : S.vectors)

/-- A plane in space -/
structure Plane (S : Space) :=
  (point : S.points)
  (normal : S.vectors)

/-- Parallel relation between lines -/
def parallel (S : Space) (a b : Line S) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def perp_line_plane (S : Space) (l : Line S) (α : Plane S) : Prop := sorry

/-- Perpendicular relation between lines -/
def perp_line_line (S : Space) (l1 l2 : Line S) : Prop := sorry

/-- The main theorem -/
theorem line_parallel_perp_plane_implies_perp_line 
  (S : Space) (a b l : Line S) (α : Plane S) :
  parallel S a b → perp_line_plane S l α → perp_line_line S l b := by
  sorry

end line_parallel_perp_plane_implies_perp_line_l3719_371998


namespace largest_root_of_g_l3719_371943

def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

theorem largest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (7/5) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end largest_root_of_g_l3719_371943


namespace borrowing_lending_period_l3719_371942

theorem borrowing_lending_period (principal : ℝ) (borrowing_rate : ℝ) (lending_rate : ℝ) (gain_per_year : ℝ) :
  principal = 9000 ∧ 
  borrowing_rate = 0.04 ∧ 
  lending_rate = 0.06 ∧ 
  gain_per_year = 180 → 
  (gain_per_year / (principal * (lending_rate - borrowing_rate))) = 1 := by
sorry

end borrowing_lending_period_l3719_371942


namespace pebbles_count_l3719_371929

/-- The width of a splash made by a pebble in meters -/
def pebble_splash : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash : ℚ := 2

/-- The number of rocks tossed -/
def rocks_tossed : ℕ := 3

/-- The number of boulders tossed -/
def boulders_tossed : ℕ := 2

/-- The total width of all splashes in meters -/
def total_splash_width : ℚ := 7

/-- The number of pebbles tossed -/
def pebbles_tossed : ℕ := 6

theorem pebbles_count :
  pebbles_tossed * pebble_splash + 
  rocks_tossed * rock_splash + 
  boulders_tossed * boulder_splash = 
  total_splash_width := by sorry

end pebbles_count_l3719_371929


namespace impossible_table_l3719_371994

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the table -/
def Table := Cell → Int

/-- Two cells are adjacent if they share a side -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- The table satisfies the adjacency condition -/
def satisfies_adjacency (t : Table) : Prop :=
  ∀ c1 c2 : Cell, adjacent c1 c2 → |t c1 - t c2| ≤ 18

/-- The table contains different integers -/
def all_different (t : Table) : Prop :=
  ∀ c1 c2 : Cell, c1 ≠ c2 → t c1 ≠ t c2

/-- The main theorem -/
theorem impossible_table : ¬∃ t : Table, satisfies_adjacency t ∧ all_different t := by
  sorry

end impossible_table_l3719_371994


namespace square_field_area_l3719_371902

/-- Given a square field with barbed wire drawn around it, if the total cost of the wire
    at a specific rate per meter is a certain amount, then we can determine the area of the field. -/
theorem square_field_area (wire_cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) :
  wire_cost_per_meter = 1 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 666 →
  ∃ (side_length : ℝ), 
    side_length > 0 ∧
    (4 * side_length - num_gates * gate_width) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end square_field_area_l3719_371902


namespace cycle_gains_and_overall_gain_l3719_371968

def cycle1_purchase : ℚ := 900
def cycle1_sale : ℚ := 1440
def cycle2_purchase : ℚ := 1200
def cycle2_sale : ℚ := 1680
def cycle3_purchase : ℚ := 1500
def cycle3_sale : ℚ := 1950

def gain_percentage (purchase : ℚ) (sale : ℚ) : ℚ :=
  ((sale - purchase) / purchase) * 100

def total_purchase : ℚ := cycle1_purchase + cycle2_purchase + cycle3_purchase
def total_sale : ℚ := cycle1_sale + cycle2_sale + cycle3_sale

theorem cycle_gains_and_overall_gain :
  (gain_percentage cycle1_purchase cycle1_sale = 60) ∧
  (gain_percentage cycle2_purchase cycle2_sale = 40) ∧
  (gain_percentage cycle3_purchase cycle3_sale = 30) ∧
  (gain_percentage total_purchase total_sale = 40 + 5/6) :=
sorry

end cycle_gains_and_overall_gain_l3719_371968


namespace solve_rational_equation_l3719_371963

theorem solve_rational_equation (x : ℚ) :
  (x^2 - 10*x + 9) / (x - 1) + (2*x^2 + 17*x - 15) / (2*x - 3) = -5 →
  x = -1/2 := by
  sorry

end solve_rational_equation_l3719_371963


namespace min_value_of_expression_l3719_371900

theorem min_value_of_expression (x y z w : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 1) 
  (hz : -1 < z ∧ z < 1) 
  (hw : -2 < w ∧ w < 2) :
  (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w/2)) + 
   1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w/2))) ≥ 2 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0) * (1 - 0/2)) + 
   1 / ((1 + 0) * (1 + 0) * (1 + 0) * (1 + 0/2))) = 2 :=
by sorry

end min_value_of_expression_l3719_371900


namespace proportion_equality_l3719_371923

theorem proportion_equality (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 := by
  sorry

end proportion_equality_l3719_371923


namespace second_larger_perfect_square_l3719_371983

theorem second_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m ^ 2) ∧
  (∀ y : ℕ, y > x ∧ (∃ l : ℕ, y = l ^ 2) → y ≥ n) ∧
  n = x + 4 * (x.sqrt) + 4 :=
sorry

end second_larger_perfect_square_l3719_371983


namespace solve_for_z_l3719_371927

theorem solve_for_z (x y z : ℝ) : 
  x^2 - 3*x + 6 = y - 10 → 
  y = 2*z → 
  x = -5 → 
  z = 28 := by
sorry

end solve_for_z_l3719_371927


namespace quadratic_root_implies_m_l3719_371906

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ x = -1) → m = -3 := by
sorry

end quadratic_root_implies_m_l3719_371906


namespace hyperbola_equation_l3719_371914

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x = 2 ∧ x^2 / a^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = 3/2) →
  a^2 = 4 ∧ b^2 = 5 := by
sorry

end hyperbola_equation_l3719_371914


namespace ray_AB_not_equal_ray_BA_l3719_371986

-- Define a point type
def Point := ℝ × ℝ

-- Define a ray type
structure Ray where
  start : Point
  direction : Point

-- Define an equality relation for rays
def ray_eq (r1 r2 : Ray) : Prop :=
  r1.start = r2.start ∧ r1.direction = r2.direction

-- Theorem statement
theorem ray_AB_not_equal_ray_BA (A B : Point) (h : A ≠ B) :
  ¬(ray_eq (Ray.mk A B) (Ray.mk B A)) := by
  sorry

end ray_AB_not_equal_ray_BA_l3719_371986


namespace quadratic_equation_solution_l3719_371948

theorem quadratic_equation_solution (C : ℝ) (h : C = 3) :
  ∃ x : ℝ, 3 * x^2 - 6 * x + C = 0 ∧ x = 1 := by
  sorry

end quadratic_equation_solution_l3719_371948


namespace sum_of_squares_of_coefficients_l3719_371969

/-- The expression to be simplified -/
def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 5 * (x^3 - 2*x^2 + 4*x - 1)

/-- The fully simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := -5*x^3 + 13*x^2 - 29*x + 14

/-- The coefficients of the simplified expression -/
def coefficients : List ℝ := [-5, 13, -29, 14]

/-- Theorem stating that the sum of squares of coefficients equals 1231 -/
theorem sum_of_squares_of_coefficients :
  (coefficients.map (λ c => c^2)).sum = 1231 := by sorry

end sum_of_squares_of_coefficients_l3719_371969


namespace james_profit_l3719_371937

/-- Calculates the profit from selling toys --/
def calculate_profit (initial_quantity : ℕ) (buy_price sell_price : ℚ) (sell_percentage : ℚ) : ℚ :=
  let total_cost := initial_quantity * buy_price
  let sold_quantity := (initial_quantity : ℚ) * sell_percentage
  let total_revenue := sold_quantity * sell_price
  total_revenue - total_cost

/-- Proves that James' profit is $800 --/
theorem james_profit :
  calculate_profit 200 20 30 (4/5) = 800 := by
  sorry

end james_profit_l3719_371937


namespace log_ratio_problem_l3719_371922

theorem log_ratio_problem (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h_log : Real.log p / Real.log 16 = Real.log q / Real.log 20 ∧ 
           Real.log p / Real.log 16 = Real.log (p + q) / Real.log 25) : 
  p / q = (Real.sqrt 5 - 1) / 2 := by
  sorry

end log_ratio_problem_l3719_371922


namespace negative_integer_sum_square_twelve_l3719_371912

theorem negative_integer_sum_square_twelve (M : ℤ) : 
  M < 0 → M^2 + M = 12 → M = -4 := by sorry

end negative_integer_sum_square_twelve_l3719_371912


namespace highlighter_difference_l3719_371980

theorem highlighter_difference (total pink blue yellow : ℕ) : 
  total = 40 →
  yellow = 7 →
  blue = pink + 5 →
  total = yellow + pink + blue →
  pink - yellow = 7 := by
sorry

end highlighter_difference_l3719_371980


namespace cubic_root_sum_l3719_371925

theorem cubic_root_sum (p q r : ℕ) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) :
  (∃ x : ℝ, 27 * x^3 - 11 * x^2 - 11 * x - 3 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / r) →
  p + q + r = 782 := by
sorry


end cubic_root_sum_l3719_371925


namespace sum_of_powers_half_l3719_371930

theorem sum_of_powers_half : 
  (-1/2 : ℚ)^3 + (-1/2 : ℚ)^2 + (-1/2 : ℚ)^1 + (1/2 : ℚ)^1 + (1/2 : ℚ)^2 + (1/2 : ℚ)^3 = 1/2 := by
  sorry

end sum_of_powers_half_l3719_371930


namespace mrs_heine_biscuits_l3719_371975

/-- Calculates the total number of biscuits needed for Mrs. Heine's pets -/
def total_biscuits (num_dogs : ℕ) (num_cats : ℕ) (num_birds : ℕ) 
                   (biscuits_per_dog : ℕ) (biscuits_per_cat : ℕ) (biscuits_per_bird : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog + num_cats * biscuits_per_cat + num_birds * biscuits_per_bird

/-- Theorem stating that Mrs. Heine needs to buy 11 biscuits in total -/
theorem mrs_heine_biscuits : 
  total_biscuits 2 1 3 3 2 1 = 11 := by
  sorry

end mrs_heine_biscuits_l3719_371975


namespace equation_solution_l3719_371961

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by sorry

end equation_solution_l3719_371961


namespace sin_cos_identity_l3719_371921

theorem sin_cos_identity : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) - 
  Real.sin (253 * π / 180) * Real.cos (43 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l3719_371921


namespace point_on_unit_circle_l3719_371935

/-- A point on the unit circle reached by moving counterclockwise from (1,0) along an arc length of 2π/3 has coordinates (-1/2, √3/2). -/
theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Real.cos (2 * Real.pi / 3) = Q.1 ∧ Real.sin (2 * Real.pi / 3) = Q.2) →  -- Q is reached by moving 2π/3 radians counterclockwise from (1,0)
  (Q.1 = -1/2 ∧ Q.2 = Real.sqrt 3 / 2) :=  -- Q has coordinates (-1/2, √3/2)
by sorry

end point_on_unit_circle_l3719_371935


namespace symmetric_line_l3719_371973

/-- Given a point (x, y) on the line y = -x + 2, prove that it is symmetric to a point on the line y = x about the line x = 1 -/
theorem symmetric_line (x y : ℝ) : 
  y = -x + 2 → 
  ∃ (x' y' : ℝ), 
    (y' = x') ∧  -- Point (x', y') is on the line y = x
    ((x + x') / 2 = 1) ∧  -- Midpoint of x and x' is on the line x = 1
    (y = y') -- y-coordinates are the same
    := by sorry

end symmetric_line_l3719_371973


namespace complement_union_theorem_l3719_371974

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 4} := by sorry

end complement_union_theorem_l3719_371974


namespace gain_percentage_is_20_percent_l3719_371957

def selling_price : ℝ := 90
def gain : ℝ := 15

theorem gain_percentage_is_20_percent :
  let cost_price := selling_price - gain
  (gain / cost_price) * 100 = 20 := by sorry

end gain_percentage_is_20_percent_l3719_371957
