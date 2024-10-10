import Mathlib

namespace two_x_power_x_eq_sqrt_two_solutions_l1092_109282

theorem two_x_power_x_eq_sqrt_two_solutions (x : ℝ) :
  x > 0 ∧ 2 * (x ^ x) = Real.sqrt 2 ↔ x = 1/2 ∨ x = 1/4 :=
sorry

end two_x_power_x_eq_sqrt_two_solutions_l1092_109282


namespace arithmetic_sequence_75th_term_l1092_109227

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_75th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_15 : a 15 = 8)
  (h_60 : a 60 = 20) :
  a 75 = 24 := by
  sorry


end arithmetic_sequence_75th_term_l1092_109227


namespace min_area_special_square_l1092_109228

/-- A square with one side on y = 2x - 17 and two vertices on y = x^2 -/
structure SpecialSquare where
  -- Coordinates of the two vertices on the parabola
  x₁ : ℝ
  x₂ : ℝ
  -- Conditions
  vertex_on_parabola : x₁ < x₂ ∧ (x₁, x₁^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2} ∧ (x₂, x₂^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  side_on_line : ∃ (a b : ℝ), (a, 2*a - 17) ∈ {p : ℝ × ℝ | p.2 = 2*p.1 - 17} ∧ 
                               (b, 2*b - 17) ∈ {p : ℝ × ℝ | p.2 = 2*p.1 - 17} ∧
                               (b - a)^2 + (2*b - 17 - (2*a - 17))^2 = (x₂ - x₁)^2 + (x₂^2 - x₁^2)^2

/-- The area of a SpecialSquare -/
def area (s : SpecialSquare) : ℝ := (s.x₂ - s.x₁)^2 + (s.x₂^2 - s.x₁^2)^2

/-- Theorem stating the minimum area of a SpecialSquare is 80 -/
theorem min_area_special_square : 
  ∀ s : SpecialSquare, area s ≥ 80 ∧ ∃ s' : SpecialSquare, area s' = 80 := by
  sorry

end min_area_special_square_l1092_109228


namespace probability_specific_arrangement_l1092_109293

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem probability_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 35 := by sorry

end probability_specific_arrangement_l1092_109293


namespace no_primes_divisible_by_45_l1092_109240

theorem no_primes_divisible_by_45 : ∀ p : ℕ, Nat.Prime p → ¬(45 ∣ p) := by
  sorry

end no_primes_divisible_by_45_l1092_109240


namespace min_value_x_plus_2y_l1092_109246

/-- Given x > 0 and y > 0, the minimum value of (x+2y)^+ is 9 -/
theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (m : ℝ), m = 9 ∧ ∀ z, z = x + 2*y → z ≥ m :=
sorry

end min_value_x_plus_2y_l1092_109246


namespace hundredth_term_is_one_l1092_109276

/-- Defines the sequence term at position n -/
def sequenceTerm (n : ℕ) : ℕ :=
  sorry

/-- The number of elements in the first n groups -/
def elementsInGroups (n : ℕ) : ℕ :=
  n^2

theorem hundredth_term_is_one :
  sequenceTerm 100 = 1 :=
sorry

end hundredth_term_is_one_l1092_109276


namespace total_sweets_l1092_109297

theorem total_sweets (red_sweets : ℕ) (green_sweets : ℕ) (other_sweets : ℕ)
  (h1 : red_sweets = 49)
  (h2 : green_sweets = 59)
  (h3 : other_sweets = 177) :
  red_sweets + green_sweets + other_sweets = 285 := by
sorry

end total_sweets_l1092_109297


namespace rectangle_area_l1092_109298

/-- Rectangle ABCD with specific properties -/
structure Rectangle where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side AD -/
  AD : ℝ
  /-- AD is 9 units longer than AB -/
  length_diff : AD = AB + 9
  /-- Area of trapezoid ABCE is 5 times the area of triangle ADE -/
  area_ratio : AB * AD = 6 * ((1/2) * AB * (1/3 * AD))
  /-- Perimeter difference between trapezoid ABCE and triangle ADE -/
  perimeter_diff : AB + (2/3 * AB) - (1/3 * AB) = 68

/-- The area of the rectangle ABCD is 3060 square units -/
theorem rectangle_area (rect : Rectangle) : rect.AB * rect.AD = 3060 := by
  sorry


end rectangle_area_l1092_109298


namespace one_point_zero_six_million_scientific_notation_l1092_109235

theorem one_point_zero_six_million_scientific_notation :
  (1.06 : ℝ) * (1000000 : ℝ) = (1.06 : ℝ) * (10 ^ 6 : ℝ) := by
  sorry

end one_point_zero_six_million_scientific_notation_l1092_109235


namespace female_democrats_count_l1092_109275

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 660 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 110 :=
by
  sorry

end female_democrats_count_l1092_109275


namespace halfway_fraction_l1092_109255

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b + ((c : ℚ) / d - (a : ℚ) / b) / 2 = 19 / 24 := by
  sorry

end halfway_fraction_l1092_109255


namespace machine_work_time_equation_l1092_109251

theorem machine_work_time_equation (x : ℝ) (hx : x > 0) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end machine_work_time_equation_l1092_109251


namespace polynomial_root_problem_l1092_109241

theorem polynomial_root_problem (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ t : ℝ, t^3 + a*t^2 + b*t + 20 = 0 ↔ t = x ∨ t = y ∨ t = z)) →
  (∀ t : ℝ, t^3 + a*t^2 + b*t + 20 = 0 → t^4 + t^3 + b*t^2 + c*t + 200 = 0) →
  (1 : ℝ)^4 + (1 : ℝ)^3 + b*(1 : ℝ)^2 + c*(1 : ℝ) + 200 = 132 :=
by sorry

end polynomial_root_problem_l1092_109241


namespace additional_pots_in_warm_hour_l1092_109220

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time in minutes to produce a pot when the machine is cold -/
def cold_production_time : ℕ := 6

/-- The time in minutes to produce a pot when the machine is warm -/
def warm_production_time : ℕ := 5

/-- Theorem stating the difference in pot production between warm and cold hours -/
theorem additional_pots_in_warm_hour :
  (minutes_per_hour / warm_production_time) - (minutes_per_hour / cold_production_time) = 2 :=
by sorry

end additional_pots_in_warm_hour_l1092_109220


namespace equal_distribution_l1092_109259

/-- Proves that when Rs 42,900 is distributed equally among 22 persons, each person receives Rs 1,950. -/
theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) : 
  total_amount = 42900 → 
  num_persons = 22 → 
  amount_per_person = total_amount / num_persons → 
  amount_per_person = 1950 := by
sorry

end equal_distribution_l1092_109259


namespace expression_undefined_at_nine_l1092_109205

/-- The expression (3x^3 + 4) / (x^2 - 18x + 81) is undefined when x = 9 -/
theorem expression_undefined_at_nine :
  ∀ x : ℝ, x = 9 → (x^2 - 18*x + 81 = 0) := by
  sorry

end expression_undefined_at_nine_l1092_109205


namespace labourer_savings_labourer_savings_specific_l1092_109281

theorem labourer_savings (monthly_income : ℕ) (initial_expense : ℕ) (reduced_expense : ℕ) 
  (initial_months : ℕ) (reduced_months : ℕ) : ℕ :=
  let initial_total_expense := initial_months * initial_expense
  let initial_total_income := initial_months * monthly_income
  let debt := if initial_total_expense > initial_total_income 
    then initial_total_expense - initial_total_income 
    else 0
  let reduced_total_expense := reduced_months * reduced_expense
  let reduced_total_income := reduced_months * monthly_income
  let savings := reduced_total_income - (reduced_total_expense + debt)
  savings

theorem labourer_savings_specific : 
  labourer_savings 72 75 60 6 4 = 30 := by
  sorry

end labourer_savings_labourer_savings_specific_l1092_109281


namespace b_four_lt_b_seven_l1092_109248

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (b n α + 1 / α (n + 1))

theorem b_four_lt_b_seven (α : ℕ → ℕ) (h : ∀ k, α k ≥ 1) :
  b 4 α < b 7 α := by
  sorry

end b_four_lt_b_seven_l1092_109248


namespace smallest_lcm_four_digit_gcd_five_l1092_109231

theorem smallest_lcm_four_digit_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  201000 ≤ Nat.lcm k l ∧ 
  ∃ (k₀ l₀ : ℕ), 1000 ≤ k₀ ∧ k₀ < 10000 ∧ 
                 1000 ≤ l₀ ∧ l₀ < 10000 ∧ 
                 Nat.gcd k₀ l₀ = 5 ∧
                 Nat.lcm k₀ l₀ = 201000 :=
by sorry

end smallest_lcm_four_digit_gcd_five_l1092_109231


namespace liam_nickels_problem_l1092_109292

theorem liam_nickels_problem :
  ∃! n : ℕ, 120 < n ∧ n < 400 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 6 = 4 ∧
    n = 374 := by
  sorry

end liam_nickels_problem_l1092_109292


namespace boxes_needed_l1092_109207

def initial_games : ℕ := 76
def sold_games : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : 
  (initial_games - sold_games) / games_per_box = 6 :=
by sorry

end boxes_needed_l1092_109207


namespace unique_root_of_equation_l1092_109272

theorem unique_root_of_equation (a b c d : ℝ) 
  (h1 : a + d = 2016)
  (h2 : b + c = 2016)
  (h3 : a ≠ c) :
  ∃! x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) ∧ x = 1008 :=
by sorry

end unique_root_of_equation_l1092_109272


namespace remainder_seven_fourth_mod_hundred_l1092_109266

theorem remainder_seven_fourth_mod_hundred : 7^4 % 100 = 1 := by
  sorry

end remainder_seven_fourth_mod_hundred_l1092_109266


namespace cube_difference_factorization_l1092_109299

theorem cube_difference_factorization (a b : ℝ) :
  a^3 - 8*b^3 = (a - 2*b)*(a^2 + 2*a*b + 4*b^2) := by
  sorry

end cube_difference_factorization_l1092_109299


namespace max_product_sum_300_l1092_109242

theorem max_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 := by
  sorry

end max_product_sum_300_l1092_109242


namespace students_history_not_statistics_l1092_109291

/-- Given a group of students with the following properties:
  * There are 150 students in total
  * 58 students are taking history
  * 42 students are taking statistics
  * 95 students are taking history or statistics or both
  Then the number of students taking history but not statistics is 53. -/
theorem students_history_not_statistics 
  (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 150 →
  history = 58 →
  statistics = 42 →
  history_or_statistics = 95 →
  history - (history + statistics - history_or_statistics) = 53 := by
sorry

end students_history_not_statistics_l1092_109291


namespace distance_AB_DB1_is_12_div_5_l1092_109284

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  AB : ℝ
  BC : ℝ
  BB1 : ℝ

/-- The distance between AB and DB₁ in a rectangular prism -/
def distance_AB_DB1 (prism : RectangularPrism) : ℝ := sorry

theorem distance_AB_DB1_is_12_div_5 (prism : RectangularPrism) 
  (h1 : prism.AB = 5)
  (h2 : prism.BC = 4)
  (h3 : prism.BB1 = 3) :
  distance_AB_DB1 prism = 12 / 5 := by sorry

end distance_AB_DB1_is_12_div_5_l1092_109284


namespace equation_solution_l1092_109210

theorem equation_solution (x : ℝ) (number : ℝ) :
  x = 32 →
  35 - (23 - (15 - x)) = 12 * 2 / (number / 2) →
  number = -2.4 := by
sorry

end equation_solution_l1092_109210


namespace f_of_3_equals_15_l1092_109232

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement
theorem f_of_3_equals_15 : f 3 = 15 := by
  sorry

end f_of_3_equals_15_l1092_109232


namespace girls_percentage_l1092_109289

/-- The percentage of girls in a school with 150 total students and 60 boys is 60%. -/
theorem girls_percentage (total : ℕ) (boys : ℕ) (h1 : total = 150) (h2 : boys = 60) :
  (total - boys : ℚ) / total * 100 = 60 := by
  sorry

end girls_percentage_l1092_109289


namespace stratified_sample_fourth_unit_l1092_109274

/-- Represents a stratified sample from four units -/
structure StratifiedSample :=
  (total : ℕ)
  (unit_samples : Fin 4 → ℕ)
  (is_arithmetic : ∃ d : ℤ, ∀ i : Fin 3, (unit_samples i.succ : ℤ) - (unit_samples i) = d)
  (sum_to_total : (Finset.univ.sum unit_samples) = total)

/-- The theorem statement -/
theorem stratified_sample_fourth_unit 
  (sample : StratifiedSample)
  (total_collected : ℕ)
  (h_total : sample.total = 150)
  (h_collected : total_collected = 1000)
  (h_second_unit : sample.unit_samples 1 = 30) :
  sample.unit_samples 3 = 60 :=
sorry

end stratified_sample_fourth_unit_l1092_109274


namespace ellipse_theorem_l1092_109250

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 6 / 9
  h_major : 2 * a = 2 * Real.sqrt 3

/-- A line that intersects the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1) ∧
    (x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1) ∧
    (y₁ = k * x₁ + m) ∧
    (y₂ = k * x₂ + m) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- The main theorem -/
theorem ellipse_theorem (E : Ellipse) (L : IntersectingLine E) :
  (E.a^2 = 3 ∧ E.b^2 = 1) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1) →
    (x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1) →
    (y₁ = L.k * x₁ + L.m) →
    (y₂ = L.k * x₂ + L.m) →
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    (x₁ * x₂ + y₁ * y₂ = 0) →
    (abs L.m / Real.sqrt (1 + L.k^2) = Real.sqrt 3 / 2)) := by
  sorry

end ellipse_theorem_l1092_109250


namespace circle_symmetry_max_ab_l1092_109224

/-- Given a circle x^2 + y^2 - 4ax + 2by + b^2 = 0 (where a > 0 and b > 0) 
    symmetric about the line x - y - 1 = 0, the maximum value of ab is 1/8 -/
theorem circle_symmetry_max_ab (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 → 
    ∃ x' y' : ℝ, x' - y' - 1 = 0 ∧ x^2 + y^2 - 4*a*x + 2*b*y + b^2 = (x' - x)^2 + (y' - y)^2) →
  a * b ≤ 1/8 :=
by sorry

end circle_symmetry_max_ab_l1092_109224


namespace last_score_is_86_l1092_109208

def scores : List ℕ := [68, 74, 78, 83, 86, 95]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℕ, (subset.sum : ℚ) / subset.length = n

def satisfies_conditions (last_score : ℕ) : Prop :=
  last_score ∈ scores ∧
  ∀ k : ℕ, k ∈ (List.range 6) →
    is_integer_average (List.take k ((scores.filter (· ≠ last_score)) ++ [last_score]))

theorem last_score_is_86 :
  ∃ (last_score : ℕ), last_score = 86 ∧ 
    satisfies_conditions last_score ∧
    ∀ (other_score : ℕ), other_score ∈ scores → other_score ≠ 86 →
      satisfies_conditions other_score → False :=
sorry

end last_score_is_86_l1092_109208


namespace item_price_ratio_l1092_109243

theorem item_price_ratio (c p q : ℝ) (h1 : p = 0.8 * c) (h2 : q = 1.2 * c) : q / p = 3 / 2 := by
  sorry

end item_price_ratio_l1092_109243


namespace shoveling_time_bounds_l1092_109264

/-- Represents the snow shoveling scenario -/
structure SnowShoveling where
  initialRate : ℕ  -- Initial shoveling rate in cubic yards per hour
  rateDecrease : ℕ  -- Rate decrease per hour in cubic yards
  driveWidth : ℕ  -- Driveway width in yards
  driveLength : ℕ  -- Driveway length in yards
  snowDepth : ℕ  -- Snow depth in yards

/-- Calculates the time taken to shovel the driveway clean -/
def shovelingTime (s : SnowShoveling) : ℕ :=
  sorry

/-- Theorem stating that it takes at least 9 hours and less than 10 hours to clear the driveway -/
theorem shoveling_time_bounds (s : SnowShoveling) 
  (h1 : s.initialRate = 30)
  (h2 : s.rateDecrease = 2)
  (h3 : s.driveWidth = 4)
  (h4 : s.driveLength = 10)
  (h5 : s.snowDepth = 5) :
  9 ≤ shovelingTime s ∧ shovelingTime s < 10 :=
by
  sorry

end shoveling_time_bounds_l1092_109264


namespace symmetric_function_properties_l1092_109217

/-- A function that is symmetric about the line x=1 and the point (2,0) -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f x) ∧ 
  (∀ x, f (2 + x) = -f x)

theorem symmetric_function_properties (f : ℝ → ℝ) (h : SymmetricFunction f) :
  (∀ x, f (2 - x) = f x) ∧
  (∀ x, f (4 - x) = -f x) ∧
  (∀ x, f (4 + x) = f x) := by
  sorry

end symmetric_function_properties_l1092_109217


namespace isabellas_house_paintable_area_l1092_109201

/-- Calculates the total paintable wall area in a house with specified room dimensions and non-paintable areas. -/
def total_paintable_area (num_bedrooms num_bathrooms : ℕ)
                         (bedroom_length bedroom_width bedroom_height : ℝ)
                         (bathroom_length bathroom_width bathroom_height : ℝ)
                         (bedroom_nonpaintable_area bathroom_nonpaintable_area : ℝ) : ℝ :=
  let bedroom_wall_area := 2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height)
  let bathroom_wall_area := 2 * (bathroom_length * bathroom_height + bathroom_width * bathroom_height)
  let paintable_bedroom_area := bedroom_wall_area - bedroom_nonpaintable_area
  let paintable_bathroom_area := bathroom_wall_area - bathroom_nonpaintable_area
  num_bedrooms * paintable_bedroom_area + num_bathrooms * paintable_bathroom_area

/-- The total paintable wall area in Isabella's house is 1637 square feet. -/
theorem isabellas_house_paintable_area :
  total_paintable_area 3 2 14 11 9 9 7 8 55 30 = 1637 := by
  sorry

end isabellas_house_paintable_area_l1092_109201


namespace min_area_rectangle_l1092_109269

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 → w > 0 → 2 * (l + w) = 150 → l * w ≥ 74 := by
  sorry

end min_area_rectangle_l1092_109269


namespace perimeter_is_18_l1092_109258

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define points A and B on the left branch
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line passing through F1, A, and B
def line_through_F1AB (p : ℝ × ℝ) : Prop := sorry

-- State that A and B are on the hyperbola
axiom A_on_hyperbola : hyperbola A.1 A.2
axiom B_on_hyperbola : hyperbola B.1 B.2

-- State that A and B are on the line passing through F1
axiom A_on_line : line_through_F1AB A
axiom B_on_line : line_through_F1AB B

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State that the distance between A and B is 5
axiom AB_distance : distance A B = 5

-- Define the perimeter of triangle AF2B
def perimeter_AF2B : ℝ := distance A F2 + distance B F2 + distance A B

-- Theorem to prove
theorem perimeter_is_18 : perimeter_AF2B = 18 := by sorry

end perimeter_is_18_l1092_109258


namespace largest_expression_l1092_109212

theorem largest_expression : 
  let a := 3 + 1 + 4
  let b := 3 * 1 + 4
  let c := 3 + 1 * 4
  let d := 3 * 1 * 4
  let e := 3 + 0 * 1 + 4
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e := by
  sorry

end largest_expression_l1092_109212


namespace absolute_value_inequality_solution_l1092_109247

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x - 4| ≤ 6) ↔ (-2 ≤ x ∧ x ≤ 10) := by sorry

end absolute_value_inequality_solution_l1092_109247


namespace truck_fuel_distance_l1092_109216

/-- Given a truck that travels 300 miles on 10 gallons of fuel,
    prove that it will travel 450 miles on 15 gallons of fuel,
    assuming a proportional relationship between fuel consumption and distance. -/
theorem truck_fuel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ)
    (h1 : initial_distance = 300)
    (h2 : initial_fuel = 10)
    (h3 : new_fuel = 15)
    (h4 : initial_fuel > 0) :
  (new_fuel / initial_fuel) * initial_distance = 450 := by
  sorry

end truck_fuel_distance_l1092_109216


namespace square_difference_l1092_109286

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l1092_109286


namespace complement_of_M_l1092_109238

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M :
  Set.compl M = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end complement_of_M_l1092_109238


namespace hexagon_problem_l1092_109236

-- Define the regular hexagon
structure RegularHexagon :=
  (side_length : ℝ)
  (A B C D E F : ℝ × ℝ)

-- Define the intersection point L
def L (hex : RegularHexagon) : ℝ × ℝ := sorry

-- Define point K
def K (hex : RegularHexagon) : ℝ × ℝ := sorry

-- Function to check if a point is outside the hexagon
def is_outside (hex : RegularHexagon) (point : ℝ × ℝ) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem hexagon_problem (hex : RegularHexagon) 
  (h1 : hex.side_length = 2) :
  is_outside hex (K hex) ∧ 
  distance (K hex) hex.B = (2 * Real.sqrt 3) / 3 := by sorry

end hexagon_problem_l1092_109236


namespace range_of_a_l1092_109214

theorem range_of_a (a : ℝ) : 
  (a + 1)^(-1/2 : ℝ) < (3 - 2*a)^(-1/2 : ℝ) → 
  2/3 < a ∧ a < 3/2 :=
by sorry

end range_of_a_l1092_109214


namespace square_difference_given_sum_and_weighted_sum_l1092_109202

theorem square_difference_given_sum_and_weighted_sum (x y : ℝ) 
  (h1 : x + y = 15) (h2 : 3 * x + y = 20) : x^2 - y^2 = -150 := by
  sorry

end square_difference_given_sum_and_weighted_sum_l1092_109202


namespace parallel_sides_equal_or_complementary_l1092_109267

/-- Two angles in space -/
structure AngleInSpace where
  -- Define the necessary components of an angle in space
  -- This is a simplified representation
  measure : ℝ

/-- Predicate to check if two angles have parallel sides -/
def has_parallel_sides (a b : AngleInSpace) : Prop :=
  -- This is a placeholder for the actual condition of parallel sides
  True

/-- Predicate to check if two angles are equal -/
def are_equal (a b : AngleInSpace) : Prop :=
  a.measure = b.measure

/-- Predicate to check if two angles are complementary -/
def are_complementary (a b : AngleInSpace) : Prop :=
  a.measure + b.measure = 90

/-- Theorem: If two angles in space have parallel sides, 
    then they are either equal or complementary -/
theorem parallel_sides_equal_or_complementary (a b : AngleInSpace) :
  has_parallel_sides a b → (are_equal a b ∨ are_complementary a b) := by
  sorry

end parallel_sides_equal_or_complementary_l1092_109267


namespace field_trip_bus_occupancy_l1092_109273

/-- Proves that given the conditions from the field trip problem, 
    the number of people in each bus is 18.0 --/
theorem field_trip_bus_occupancy 
  (num_vans : ℝ) 
  (num_buses : ℝ) 
  (people_per_van : ℝ) 
  (additional_people_in_buses : ℝ) 
  (h1 : num_vans = 6.0)
  (h2 : num_buses = 8.0)
  (h3 : people_per_van = 6.0)
  (h4 : additional_people_in_buses = 108.0) :
  (num_vans * people_per_van + additional_people_in_buses) / num_buses = 18.0 := by
  sorry

#eval (6.0 * 6.0 + 108.0) / 8.0  -- This should evaluate to 18.0

end field_trip_bus_occupancy_l1092_109273


namespace work_equals_2pi_l1092_109221

/-- The force field F --/
def F (x y : ℝ) : ℝ × ℝ := (x - y, 1)

/-- The curve L --/
def L : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4 ∧ p.2 ≥ 0}

/-- Starting point --/
def M : ℝ × ℝ := (2, 0)

/-- Ending point --/
def N : ℝ × ℝ := (-2, 0)

/-- Work done by force F along curve L from M to N --/
noncomputable def work : ℝ := sorry

theorem work_equals_2pi : work = 2 * Real.pi := by sorry

end work_equals_2pi_l1092_109221


namespace cylinder_volume_l1092_109265

/-- The volume of a solid cylinder in a cubic container --/
theorem cylinder_volume (container_side : ℝ) (exposed_height : ℝ) (base_area_ratio : ℝ) :
  container_side = 20 →
  exposed_height = 8 →
  base_area_ratio = 1/8 →
  (container_side - exposed_height) * (container_side * container_side * base_area_ratio) = 650 :=
by sorry

end cylinder_volume_l1092_109265


namespace neighborhood_glass_panels_l1092_109215

/-- Represents the number of houses of each type -/
def num_houses_A : ℕ := 4
def num_houses_B : ℕ := 3
def num_houses_C : ℕ := 3

/-- Represents the number of glass panels in each type of house -/
def panels_per_house_A : ℕ := 
  4 * 6 + -- double windows downstairs
  8 * 3 + -- single windows upstairs
  2 * 6 + -- bay windows
  1 * 2 + -- front door
  1 * 3   -- back door

def panels_per_house_B : ℕ := 
  8 * 5 + -- double windows downstairs
  6 * 4 + -- single windows upstairs
  1 * 7 + -- bay window
  1 * 4   -- front door

def panels_per_house_C : ℕ := 
  5 * 4 + -- double windows downstairs
  10 * 2 + -- single windows upstairs
  3 * 1   -- skylights

/-- The total number of glass panels in the neighborhood -/
def total_panels : ℕ := 
  num_houses_A * panels_per_house_A +
  num_houses_B * panels_per_house_B +
  num_houses_C * panels_per_house_C

/-- Theorem stating that the total number of glass panels in the neighborhood is 614 -/
theorem neighborhood_glass_panels : total_panels = 614 := by
  sorry

end neighborhood_glass_panels_l1092_109215


namespace light_wash_water_usage_l1092_109256

/-- Represents the water usage of a washing machine -/
structure WashingMachine where
  heavyWashWater : ℕ
  regularWashWater : ℕ
  lightWashWater : ℕ
  heavyWashCount : ℕ
  regularWashCount : ℕ
  lightWashCount : ℕ
  bleachedLoadsCount : ℕ
  totalWaterUsage : ℕ

/-- Theorem stating that the light wash water usage is 2 gallons -/
theorem light_wash_water_usage 
  (wm : WashingMachine) 
  (heavy_wash : wm.heavyWashWater = 20)
  (regular_wash : wm.regularWashWater = 10)
  (wash_counts : wm.heavyWashCount = 2 ∧ wm.regularWashCount = 3 ∧ wm.lightWashCount = 1)
  (bleached_loads : wm.bleachedLoadsCount = 2)
  (total_water : wm.totalWaterUsage = 76)
  (water_balance : wm.totalWaterUsage = 
    wm.heavyWashWater * wm.heavyWashCount + 
    wm.regularWashWater * wm.regularWashCount + 
    wm.lightWashWater * (wm.lightWashCount + wm.bleachedLoadsCount)) :
  wm.lightWashWater = 2 := by
  sorry

end light_wash_water_usage_l1092_109256


namespace certain_number_problem_l1092_109225

theorem certain_number_problem (A B : ℝ) (h1 : A + B = 15) (h2 : A = 7) :
  ∃ C : ℝ, C * B = 5 * A - 11 ∧ C = 3 := by
sorry

end certain_number_problem_l1092_109225


namespace gcf_lcm_300_105_l1092_109209

theorem gcf_lcm_300_105 : ∃ (gcf lcm : ℕ),
  (Nat.gcd 300 105 = gcf) ∧
  (Nat.lcm 300 105 = lcm) ∧
  (gcf = 15) ∧
  (lcm = 2100) := by
  sorry

end gcf_lcm_300_105_l1092_109209


namespace g_value_at_five_sixths_l1092_109290

/-- Given a function f and g with specific properties, prove that g(5/6) = -√3/2 -/
theorem g_value_at_five_sixths 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, f x = Real.sqrt 2 * Real.sin (a * x + π / 4))
  (h3 : ∀ x, x ≥ 0 → g x = g (x - 1))
  (h4 : ∀ x, x < 0 → g x = Real.sin (a * x))
  (h5 : ∃ T, T > 0 ∧ T = 1 ∧ ∀ x, f (x + T) = f x) :
  g (5/6) = -Real.sqrt 3 / 2 := by
sorry

end g_value_at_five_sixths_l1092_109290


namespace ball_attendance_l1092_109237

theorem ball_attendance :
  ∀ n m : ℕ,
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  (∃ k : ℕ, n = 20 * k ∧ m = 21 * k) →
  n + m = 41 :=
λ n m h1 h2 h3 =>
  sorry

end ball_attendance_l1092_109237


namespace gcd_15_2015_l1092_109244

theorem gcd_15_2015 : Nat.gcd 15 2015 = 5 := by
  sorry

end gcd_15_2015_l1092_109244


namespace original_sugar_percentage_l1092_109263

/-- Given a solution where one fourth is replaced by a 42% sugar solution,
    resulting in an 18% sugar solution, prove that the original solution
    must have been 10% sugar. -/
theorem original_sugar_percentage
  (original : ℝ)
  (replaced : ℝ := 1/4)
  (second_solution : ℝ := 42)
  (final_solution : ℝ := 18)
  (h : (1 - replaced) * original + replaced * second_solution = final_solution) :
  original = 10 :=
sorry

end original_sugar_percentage_l1092_109263


namespace ellipse_triangle_perimeter_l1092_109234

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- State that AB is perpendicular to x-axis
def AB_perpendicular_to_x : Prop := sorry

-- Define the perimeter of triangle AF₁B
def perimeter_AF1B : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse point_A.1 point_A.2 ∧
  ellipse point_B.1 point_B.2 ∧
  AB_perpendicular_to_x →
  perimeter_AF1B = 24 := by sorry

end ellipse_triangle_perimeter_l1092_109234


namespace group_division_ways_l1092_109280

theorem group_division_ways (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) : 
  Nat.choose n k = 20 := by
  sorry

end group_division_ways_l1092_109280


namespace stratified_sample_size_l1092_109285

/-- Represents the structure of a company's workforce -/
structure Company where
  staff : ℕ
  middle_managers : ℕ
  senior_managers : ℕ

/-- Calculates the total number of employees in the company -/
def Company.total (c : Company) : ℕ := c.staff + c.middle_managers + c.senior_managers

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  company : Company
  sample_size : ℕ
  selected_senior_managers : ℕ

/-- Theorem stating the correct sample size for the given conditions -/
theorem stratified_sample_size
  (c : Company)
  (sample : StratifiedSample)
  (h1 : c.staff = 160)
  (h2 : c.middle_managers = 30)
  (h3 : c.senior_managers = 10)
  (h4 : sample.company = c)
  (h5 : sample.selected_senior_managers = 1) :
  sample.sample_size = 20 := by
  sorry


end stratified_sample_size_l1092_109285


namespace negative495_terminates_as_225_l1092_109229

-- Define the set of possible answers
inductive PossibleAnswer
  | angle135  : PossibleAnswer
  | angle45   : PossibleAnswer
  | angle225  : PossibleAnswer
  | angleNeg225 : PossibleAnswer

-- Define a function to convert PossibleAnswer to real number (in degrees)
def toRealDegrees (a : PossibleAnswer) : ℝ :=
  match a with
  | PossibleAnswer.angle135   => 135
  | PossibleAnswer.angle45    => 45
  | PossibleAnswer.angle225   => 225
  | PossibleAnswer.angleNeg225 => -225

-- Define what it means for two angles to terminate in the same direction
def terminatesSameDirection (a b : ℝ) : Prop :=
  ∃ k : ℤ, a - b = 360 * (k : ℝ)

-- State the theorem
theorem negative495_terminates_as_225 :
  ∃ (answer : PossibleAnswer), terminatesSameDirection (-495) (toRealDegrees answer) ∧
  answer = PossibleAnswer.angle225 :=
sorry

end negative495_terminates_as_225_l1092_109229


namespace ricks_savings_to_gift_ratio_l1092_109277

def gift_cost : ℕ := 250
def cake_cost : ℕ := 25
def erikas_savings : ℕ := 155
def money_left : ℕ := 5

def total_savings : ℕ := gift_cost + cake_cost - money_left

def ricks_savings : ℕ := total_savings - erikas_savings

theorem ricks_savings_to_gift_ratio :
  (ricks_savings : ℚ) / gift_cost = 23 / 50 := by sorry

end ricks_savings_to_gift_ratio_l1092_109277


namespace sum_equals_250_l1092_109288

theorem sum_equals_250 : 157 + 18 + 32 + 43 = 250 := by
  sorry

end sum_equals_250_l1092_109288


namespace quadratic_function_properties_l1092_109254

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hno_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (¬ (a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x)) ∧
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end quadratic_function_properties_l1092_109254


namespace jim_distance_in_24_steps_l1092_109283

-- Define the number of steps for Carly and Jim to cover the same distance
def carly_steps : ℕ := 3
def jim_steps : ℕ := 4

-- Define the length of Carly's step in meters
def carly_step_length : ℚ := 1/2

-- Define the number of Jim's steps we're interested in
def jim_target_steps : ℕ := 24

-- Theorem to prove
theorem jim_distance_in_24_steps :
  (jim_target_steps : ℚ) * (carly_steps * carly_step_length) / jim_steps = 9 := by
  sorry

end jim_distance_in_24_steps_l1092_109283


namespace quadratic_square_form_sum_l1092_109230

theorem quadratic_square_form_sum (x : ℝ) :
  ∃ (a b c : ℤ), a > 0 ∧
  (25 * x^2 + 30 * x - 35 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 52 := by sorry

end quadratic_square_form_sum_l1092_109230


namespace probability_B_outscores_A_is_correct_l1092_109261

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : Rat

/-- The probability that team B finishes with more points than team A -/
def probability_B_outscores_A (tournament : SoccerTournament) : Rat :=
  793 / 2048

/-- Theorem stating the probability that team B finishes with more points than team A -/
theorem probability_B_outscores_A_is_correct (tournament : SoccerTournament) 
  (h1 : tournament.num_teams = 8)
  (h2 : tournament.games_per_team = 7)
  (h3 : tournament.win_probability = 1 / 2) : 
  probability_B_outscores_A tournament = 793 / 2048 := by
  sorry

end probability_B_outscores_A_is_correct_l1092_109261


namespace toy_purchase_problem_l1092_109223

theorem toy_purchase_problem (toy_cost : ℝ) (discount_rate : ℝ) (total_paid : ℝ) :
  toy_cost = 3 →
  discount_rate = 0.2 →
  total_paid = 12 →
  (1 - discount_rate) * (toy_cost * (total_paid / ((1 - discount_rate) * toy_cost))) = total_paid →
  ∃ n : ℕ, n = 5 ∧ n * toy_cost = total_paid / (1 - discount_rate) := by
  sorry

end toy_purchase_problem_l1092_109223


namespace concert_ticket_sales_l1092_109268

/-- Represents the number of non-student tickets sold at an annual concert --/
def non_student_tickets : ℕ := 60

/-- Represents the number of student tickets sold at an annual concert --/
def student_tickets : ℕ := 150 - non_student_tickets

/-- The price of a student ticket in dollars --/
def student_price : ℕ := 5

/-- The price of a non-student ticket in dollars --/
def non_student_price : ℕ := 8

/-- The total revenue from ticket sales in dollars --/
def total_revenue : ℕ := 930

/-- The total number of tickets sold --/
def total_tickets : ℕ := 150

theorem concert_ticket_sales :
  (student_tickets * student_price + non_student_tickets * non_student_price = total_revenue) ∧
  (student_tickets + non_student_tickets = total_tickets) :=
by sorry

end concert_ticket_sales_l1092_109268


namespace three_sqrt_two_gt_sqrt_seventeen_l1092_109279

theorem three_sqrt_two_gt_sqrt_seventeen : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end three_sqrt_two_gt_sqrt_seventeen_l1092_109279


namespace farm_animals_l1092_109245

theorem farm_animals (cows chickens ducks : ℕ) : 
  (4 * cows + 2 * chickens + 2 * ducks = 2 * (cows + chickens + ducks) + 22) →
  (chickens + ducks = 2 * cows) →
  (cows = 11) := by
sorry

end farm_animals_l1092_109245


namespace three_digit_sum_property_l1092_109249

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    N = 100 * a + 10 * b + c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    (a + b + 1 = (a + b + c) / 3 ∨
     a + (b + 1) + 1 = (a + b + c) / 3)

theorem three_digit_sum_property :
  ∀ N : ℕ, is_valid_number N → (N = 207 ∨ N = 117 ∨ N = 108) :=
by sorry

end three_digit_sum_property_l1092_109249


namespace total_bird_wings_l1092_109271

/-- The number of birds in the sky -/
def num_birds : ℕ := 13

/-- The number of wings each bird has -/
def wings_per_bird : ℕ := 2

/-- Theorem: The total number of bird wings in the sky is 26 -/
theorem total_bird_wings : num_birds * wings_per_bird = 26 := by
  sorry

end total_bird_wings_l1092_109271


namespace percentage_increase_l1092_109219

theorem percentage_increase (x y z : ℝ) : 
  y = 0.5 * z →  -- y is 50% less than z
  x = 0.6 * z →  -- x is 60% of z
  x = 1.2 * y    -- x is 20% more than y (equivalent to 120% of y)
  := by sorry

end percentage_increase_l1092_109219


namespace quadratic_equation_always_real_roots_l1092_109257

theorem quadratic_equation_always_real_roots (m : ℝ) :
  ∃ x : ℝ, m * x^2 - (5*m - 1) * x + (4*m - 4) = 0 :=
by sorry

end quadratic_equation_always_real_roots_l1092_109257


namespace correct_average_l1092_109213

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 16 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 35 → 
  (n * incorrect_avg + (correct_num - incorrect_num)) / n = 17 := by
sorry

end correct_average_l1092_109213


namespace book_price_calculation_l1092_109296

theorem book_price_calculation (discounted_price original_price : ℝ) : 
  discounted_price = 8 →
  discounted_price = (1 / 8) * original_price →
  original_price = 64 := by
sorry

end book_price_calculation_l1092_109296


namespace intersection_of_lines_l1092_109270

/-- Proves the existence and uniqueness of the intersection point of two lines, if it exists -/
theorem intersection_of_lines (a b c d e f : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) (h2 : c ≠ 0 ∨ d ≠ 0) 
  (h3 : a * d ≠ b * c) : 
  ∃! p : ℝ × ℝ, a * p.1 + b * p.2 + e = 0 ∧ c * p.1 + d * p.2 + f = 0 := by
  sorry

end intersection_of_lines_l1092_109270


namespace product_of_integers_l1092_109295

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 18) 
  (diff_squares_eq : x^2 - y^2 = 36) : 
  x * y = 80 := by
  sorry

end product_of_integers_l1092_109295


namespace pencil_buyers_difference_l1092_109200

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := 13

/-- The number of cents paid by eighth graders -/
def eighth_grade_total : ℕ := 208

/-- The number of cents paid by seventh graders -/
def seventh_grade_total : ℕ := 181

/-- The number of cents paid by sixth graders -/
def sixth_grade_total : ℕ := 234

/-- The number of sixth graders -/
def sixth_graders : ℕ := 45

theorem pencil_buyers_difference :
  (sixth_grade_total / pencil_cost) - (seventh_grade_total / pencil_cost) = 4 :=
by sorry

end pencil_buyers_difference_l1092_109200


namespace prob_at_least_one_red_l1092_109253

theorem prob_at_least_one_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  (red_balls + white_balls = total_balls) →
  (probability_at_least_one_red : ℚ) =
    1 - (white_balls / total_balls * (white_balls - 1) / (total_balls - 1)) →
  probability_at_least_one_red = 7 / 10 := by
  sorry

#check prob_at_least_one_red

end prob_at_least_one_red_l1092_109253


namespace cubic_sum_over_product_l1092_109278

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + b - c = 0) : (a^3 + b^3 + c^3) / (a * b * c) = 5 := by
  sorry

end cubic_sum_over_product_l1092_109278


namespace peggy_final_doll_count_l1092_109226

/-- Calculates the final number of dolls Peggy has after a series of events --/
def finalDollCount (initialDolls : ℕ) (grandmotherGift : ℕ) : ℕ :=
  let birthdayGift := grandmotherGift / 2
  let afterBirthday := initialDolls + grandmotherGift + birthdayGift
  let afterSpringCleaning := afterBirthday - (afterBirthday / 10)
  let easterGift := birthdayGift / 3
  let afterEaster := afterSpringCleaning + easterGift
  let afterExchange := afterEaster - 1
  let christmasGift := easterGift + (easterGift / 5)
  let afterChristmas := afterExchange + christmasGift
  afterChristmas - 3

/-- Theorem stating that Peggy ends up with 50 dolls --/
theorem peggy_final_doll_count :
  finalDollCount 6 28 = 50 := by
  sorry

end peggy_final_doll_count_l1092_109226


namespace parallel_vectors_l1092_109239

/-- Given two vectors a and b in R², prove that ka + b is parallel to a - 3b iff k = -1/3 -/
theorem parallel_vectors (a b : Fin 2 → ℝ) (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : b 0 = -3) (h4 : b 1 = 2) :
  (∃ k : ℝ, ∀ i : Fin 2, k * (a i) + (b i) = c * ((a i) - 3 * (b i)) ∧ c ≠ 0) ↔ k = -1/3 :=
by sorry

end parallel_vectors_l1092_109239


namespace no_square_root_representation_l1092_109222

theorem no_square_root_representation : ¬ ∃ (A B : ℤ), (A + B * Real.sqrt 3) ^ 2 = 99999 + 111111 * Real.sqrt 3 := by
  sorry

end no_square_root_representation_l1092_109222


namespace regular_triangular_pyramid_properties_l1092_109252

structure RegularTriangularPyramid where
  PA : ℝ
  θ : ℝ

def distance_to_base (p : RegularTriangularPyramid) : ℝ :=
  sorry

def surface_area (p : RegularTriangularPyramid) : ℝ :=
  sorry

theorem regular_triangular_pyramid_properties
  (p : RegularTriangularPyramid)
  (h1 : p.PA = 2)
  (h2 : 0 < p.θ ∧ p.θ ≤ π / 2) :
  (distance_to_base { PA := 2, θ := π / 2 } = 2 * Real.sqrt 3 / 3) ∧
  (∀ θ₁ θ₂, 0 < θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ π / 2 →
    surface_area { PA := 2, θ := θ₁ } < surface_area { PA := 2, θ := θ₂ }) :=
sorry

end regular_triangular_pyramid_properties_l1092_109252


namespace triangle_side_ratio_l1092_109204

theorem triangle_side_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a →
  b / a = Real.sqrt 2 := by
sorry

end triangle_side_ratio_l1092_109204


namespace deepak_age_l1092_109233

theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (future_years : ℕ) (rahul_future_age : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 3 →
  future_years = 6 →
  rahul_future_age = 18 →
  ∃ (x : ℚ), rahul_ratio * x + future_years = rahul_future_age ∧ deepak_ratio * x = 9 :=
by sorry

end deepak_age_l1092_109233


namespace four_variable_equation_consecutive_evens_l1092_109262

theorem four_variable_equation_consecutive_evens :
  ∃ (x y z w : ℕ), 
    (x + y + z + w = 100) ∧ 
    (∃ (k : ℕ), x = 2 * k) ∧
    (∃ (l : ℕ), y = 2 * l) ∧
    (∃ (m : ℕ), z = 2 * m) ∧
    (∃ (n : ℕ), w = 2 * n) ∧
    (y = x + 2) ∧
    (z = x + 4) ∧
    (w = x + 6) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (w > 0) := by
  sorry

end four_variable_equation_consecutive_evens_l1092_109262


namespace infinitely_many_special_triangles_l1092_109294

/-- A triangle with integer side lengths and area, where one side is 4 and the difference between the other two sides is 2. -/
structure SpecialTriangle where
  a : ℕ+  -- First side length
  b : ℕ+  -- Second side length
  c : ℕ+  -- Third side length (always 4)
  area : ℕ+  -- Area of the triangle
  h_c : c = 4  -- One side is 4
  h_diff : a - b = 2 ∨ b - a = 2  -- Difference between other two sides is 2
  h_triangle : a + b > c ∧ b + c > a ∧ a + c > b  -- Triangle inequality
  h_area : 4 * area ^ 2 = (a + b + c) * (a + b - c) * (b + c - a) * (a + c - b)  -- Heron's formula

/-- There are infinitely many special triangles. -/
theorem infinitely_many_special_triangles : ∀ n : ℕ, ∃ m > n, ∃ t : SpecialTriangle, m = t.a.val := by
  sorry

end infinitely_many_special_triangles_l1092_109294


namespace house_count_l1092_109206

/-- The number of houses in a development with specific features -/
theorem house_count (G P GP N : ℕ) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) :
  G + P - GP + N = 65 := by
  sorry

#check house_count

end house_count_l1092_109206


namespace sun_energy_china_equivalence_l1092_109287

/-- The energy received from the sun in one year on 1 square kilometer of land,
    measured in kilograms of coal equivalent -/
def energy_per_sq_km : ℝ := 1.3 * 10^8

/-- The approximate land area of China in square kilometers -/
def china_area : ℝ := 9.6 * 10^6

/-- The total energy received from the sun on China's land area,
    measured in kilograms of coal equivalent -/
def total_energy : ℝ := energy_per_sq_km * china_area

theorem sun_energy_china_equivalence :
  total_energy = 1.248 * 10^15 := by
  sorry

end sun_energy_china_equivalence_l1092_109287


namespace chocolate_box_theorem_l1092_109203

/-- Represents a box of chocolates -/
structure ChocolateBox where
  total : ℕ  -- Total number of chocolates originally
  rows : ℕ   -- Number of rows
  cols : ℕ   -- Number of columns

/-- Míša's actions on the chocolate box -/
def misaActions (box : ChocolateBox) : Prop :=
  ∃ (eaten1 eaten2 : ℕ),
    -- After all actions, 1/3 of chocolates remain
    box.total / 3 = box.total - eaten1 - eaten2 - (box.rows - 1) - (box.cols - 1) ∧
    -- After first rearrangement, 3 rows are filled except for one space
    3 * box.cols - 1 = box.total - eaten1 ∧
    -- After second rearrangement, 5 columns are filled except for one space
    5 * box.rows - 1 = box.total - eaten1 - eaten2 - (box.rows - 1)

theorem chocolate_box_theorem (box : ChocolateBox) :
  misaActions box →
  box.total = 60 ∧ box.rows = 5 ∧ box.cols = 12 ∧
  ∃ (eaten1 : ℕ), eaten1 = 25 := by
  sorry

end chocolate_box_theorem_l1092_109203


namespace negation_equivalence_l1092_109260

theorem negation_equivalence :
  (¬ ∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x + 2 = 0) ↔
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → a * x^2 - 3 * x + 2 ≠ 0) :=
by sorry

end negation_equivalence_l1092_109260


namespace neighborhood_to_gina_litter_ratio_l1092_109211

/-- Given the following conditions:
  * Gina collected 2 bags of litter
  * Each bag of litter weighs 4 pounds
  * Total litter collected by everyone is 664 pounds
  Prove that the ratio of litter collected by the rest of the neighborhood
  to the amount collected by Gina is 82:1 -/
theorem neighborhood_to_gina_litter_ratio :
  let gina_bags : ℕ := 2
  let bag_weight : ℕ := 4
  let total_litter : ℕ := 664
  let gina_litter := gina_bags * bag_weight
  let neighborhood_litter := total_litter - gina_litter
  neighborhood_litter / gina_litter = 82 ∧ gina_litter ≠ 0 :=
by sorry

end neighborhood_to_gina_litter_ratio_l1092_109211


namespace cows_husk_consumption_l1092_109218

/-- The number of bags of husk eaten by a given number of cows in 45 days -/
def bags_eaten (num_cows : ℕ) : ℕ :=
  num_cows

/-- Theorem stating that 45 cows eat 45 bags of husk in 45 days -/
theorem cows_husk_consumption :
  bags_eaten 45 = 45 := by
  sorry

end cows_husk_consumption_l1092_109218
