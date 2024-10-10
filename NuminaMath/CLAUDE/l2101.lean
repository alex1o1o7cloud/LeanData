import Mathlib

namespace observation_probability_l2101_210192

theorem observation_probability 
  (total_students : Nat) 
  (total_periods : Nat) 
  (zi_shi_duration : Nat) 
  (total_duration : Nat) :
  total_students = 4 →
  total_periods = 4 →
  zi_shi_duration = 2 →
  total_duration = 8 →
  (zi_shi_duration : ℚ) / total_duration = 1 / 4 := by
  sorry

end observation_probability_l2101_210192


namespace outside_county_attendance_l2101_210116

/-- The number of kids from Lawrence county who went to camp -/
def lawrence_camp : ℕ := 34044

/-- The total number of kids who attended the camp -/
def total_camp : ℕ := 458988

/-- The number of kids from outside the county who attended the camp -/
def outside_county : ℕ := total_camp - lawrence_camp

theorem outside_county_attendance : outside_county = 424944 := by
  sorry

end outside_county_attendance_l2101_210116


namespace sufficient_not_necessary_condition_l2101_210131

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∃ a b : ℝ, a = -1 ∧ b = 2 ∧ a * b = -2) ∧
  (∃ a b : ℝ, a * b = -2 ∧ (a ≠ -1 ∨ b ≠ 2)) := by
  sorry

end sufficient_not_necessary_condition_l2101_210131


namespace power_of_128_l2101_210129

theorem power_of_128 : (128 : ℝ) ^ (7/3) = 65536 * (2 : ℝ) ^ (1/3) := by sorry

end power_of_128_l2101_210129


namespace tile_arrangement_l2101_210157

/-- The internal angle of a square in degrees -/
def square_angle : ℝ := 90

/-- The internal angle of an octagon in degrees -/
def octagon_angle : ℝ := 135

/-- The sum of angles around a vertex in degrees -/
def vertex_sum : ℝ := 360

/-- The number of square tiles around a vertex -/
def num_square_tiles : ℕ := 1

/-- The number of octagonal tiles around a vertex -/
def num_octagon_tiles : ℕ := 2

theorem tile_arrangement :
  num_square_tiles * square_angle + num_octagon_tiles * octagon_angle = vertex_sum :=
by sorry

end tile_arrangement_l2101_210157


namespace distance_origin_to_line_l2101_210189

/-- The distance from the origin to the line x = 1 is 1. -/
theorem distance_origin_to_line : ∃ d : ℝ, d = 1 ∧ 
  ∀ (x y : ℝ), x = 1 → d = |x| := by sorry

end distance_origin_to_line_l2101_210189


namespace alice_gadget_sales_l2101_210161

/-- The worth of gadgets Alice sold -/
def gadget_worth : ℝ := 2500

/-- Alice's monthly basic salary -/
def basic_salary : ℝ := 240

/-- Alice's commission rate -/
def commission_rate : ℝ := 0.02

/-- Amount Alice saves -/
def savings : ℝ := 29

/-- Percentage of total earnings Alice saves -/
def savings_rate : ℝ := 0.10

/-- Alice's total earnings -/
def total_earnings : ℝ := basic_salary + commission_rate * gadget_worth

theorem alice_gadget_sales :
  gadget_worth = 2500 ∧
  basic_salary = 240 ∧
  commission_rate = 0.02 ∧
  savings = 29 ∧
  savings_rate = 0.10 ∧
  savings = savings_rate * total_earnings :=
by sorry

end alice_gadget_sales_l2101_210161


namespace middle_school_students_l2101_210167

theorem middle_school_students (band_percentage : ℝ) (band_students : ℕ) (total_students : ℕ) : 
  band_percentage = 0.20 →
  band_students = 168 →
  (band_percentage * total_students : ℝ) = band_students →
  total_students = 840 := by
sorry

end middle_school_students_l2101_210167


namespace final_game_score_l2101_210150

/-- Represents the points scored by each player in the basketball game -/
structure PlayerPoints where
  bailey : ℕ
  michiko : ℕ
  akiko : ℕ
  chandra : ℕ

/-- Calculates the total points scored by the team -/
def total_points (p : PlayerPoints) : ℕ :=
  p.bailey + p.michiko + p.akiko + p.chandra

/-- Proves that the team scored 54 points in the final game -/
theorem final_game_score :
  ∃ (p : PlayerPoints),
    p.bailey = 14 ∧
    p.michiko = p.bailey / 2 ∧
    p.akiko = p.michiko + 4 ∧
    p.chandra = 2 * p.akiko ∧
    total_points p = 54 := by
  sorry

end final_game_score_l2101_210150


namespace quadratic_function_property_l2101_210148

/-- Given a positive real number a and a function f(x) = ax^2 + 2ax + 1,
    if f(m) < 0 for some real m, then f(m+2) > 1 -/
theorem quadratic_function_property (a : ℝ) (m : ℝ) (h_a : a > 0) :
  let f := λ x : ℝ ↦ a * x^2 + 2 * a * x + 1
  f m < 0 → f (m + 2) > 1 := by sorry

end quadratic_function_property_l2101_210148


namespace function_satisfies_equation_l2101_210109

theorem function_satisfies_equation :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = |x + 1| :=
by
  -- Define f(x) = √(x + 1)
  let f := λ x : ℝ ↦ Real.sqrt (x + 1)
  
  -- Prove that this f satisfies the equation
  -- for all x ∈ ℝ
  sorry

end function_satisfies_equation_l2101_210109


namespace bernoulli_inequality_l2101_210177

theorem bernoulli_inequality (h : ℝ) (hgt : h > -1) :
  (∀ x > 1, (1 + h)^x > 1 + h*x) ∧
  (∀ x < 0, (1 + h)^x > 1 + h*x) ∧
  (∀ x, 0 < x → x < 1 → (1 + h)^x < 1 + h*x) := by
  sorry

end bernoulli_inequality_l2101_210177


namespace tank_plastering_cost_l2101_210112

/-- Calculates the cost of plastering a tank's walls and bottom -/
def plasteringCost (length width depth : ℝ) (costPerSqMeter : ℝ) : ℝ :=
  let bottomArea := length * width
  let longWallsArea := 2 * (length * depth)
  let shortWallsArea := 2 * (width * depth)
  let totalArea := bottomArea + longWallsArea + shortWallsArea
  totalArea * costPerSqMeter

/-- Theorem stating the cost of plastering the given tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.25 = 186 := by
  sorry

end tank_plastering_cost_l2101_210112


namespace circle_tangent_point_relation_l2101_210159

/-- Given a circle C and a point A satisfying certain conditions, prove that a + (3/2)b = 3 -/
theorem circle_tangent_point_relation (a b : ℝ) : 
  (∃ (x y : ℝ), (x - 2)^2 + (y - 3)^2 = 1) →  -- Circle C equation
  (∃ (m_x m_y : ℝ), (m_x - 2)^2 + (m_y - 3)^2 = 1 ∧ 
    ((m_x - a) * (m_x - 2) + (m_y - b) * (m_y - 3) = 0)) →  -- AM is tangent to C at M
  ((a - 2)^2 + (b - 3)^2 - 1 = a^2 + b^2) →  -- |AM| = |AO|
  a + (3/2) * b = 3 := by
sorry

end circle_tangent_point_relation_l2101_210159


namespace perpendicular_transitivity_l2101_210152

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_diff_lines : m ≠ n)
  (h1 : perp n α)
  (h2 : perp n β)
  (h3 : perp m α)
  : perp m β :=
sorry

end perpendicular_transitivity_l2101_210152


namespace kg_to_ton_conversion_min_to_hour_conversion_kg_to_g_conversion_l2101_210132

-- Define conversion rates
def kg_to_ton : ℝ := 1000
def min_to_hour : ℝ := 60
def kg_to_g : ℝ := 1000

-- Theorem statements
theorem kg_to_ton_conversion : 56 / kg_to_ton = 0.056 := by sorry

theorem min_to_hour_conversion : 45 / min_to_hour = 0.75 := by sorry

theorem kg_to_g_conversion : 0.3 * kg_to_g = 300 := by sorry

end kg_to_ton_conversion_min_to_hour_conversion_kg_to_g_conversion_l2101_210132


namespace hyperbola_focus_m_value_l2101_210198

/-- Given a hyperbola with equation (y^2/m) - (x^2/9) = 1 and a focus at (0,5), prove that m = 16 -/
theorem hyperbola_focus_m_value (m : ℝ) : 
  (∃ (x y : ℝ), y^2/m - x^2/9 = 1) →  -- Hyperbola equation exists
  (0, 5) ∈ {p : ℝ × ℝ | p.1 = 0 ∧ p.2^2 = m + 9} →  -- (0,5) is a focus
  m = 16 := by
sorry

end hyperbola_focus_m_value_l2101_210198


namespace quadratic_root_m_value_l2101_210187

theorem quadratic_root_m_value : ∀ m : ℝ,
  ((-1 : ℝ)^2 + m * (-1) - 1 = 0) → m = 0 := by
  sorry

end quadratic_root_m_value_l2101_210187


namespace series_sum_l2101_210126

/-- The sum of the infinite series Σ(n=1 to ∞) ((3n - 2) / (n(n+1)(n+3))) equals 61/24 -/
theorem series_sum : ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3)) = 61 / 24 := by
  sorry

end series_sum_l2101_210126


namespace integral_sin_over_square_l2101_210184

open Real MeasureTheory

/-- The definite integral of sin(x) / (1 + cos(x) + sin(x))^2 from 0 to π/2 equals ln(2) - 1/2 -/
theorem integral_sin_over_square : ∫ x in (0)..(π/2), sin x / (1 + cos x + sin x)^2 = log 2 - 1/2 := by
  sorry

end integral_sin_over_square_l2101_210184


namespace sanya_washing_days_l2101_210146

/-- Represents the number of days needed to wash all towels -/
def days_needed (towels_per_wash : ℕ) (hours_per_day : ℕ) (total_towels : ℕ) : ℕ :=
  (total_towels + towels_per_wash * hours_per_day - 1) / (towels_per_wash * hours_per_day)

/-- Theorem stating that Sanya needs 7 days to wash all towels -/
theorem sanya_washing_days :
  days_needed 7 2 98 = 7 :=
by sorry

#eval days_needed 7 2 98

end sanya_washing_days_l2101_210146


namespace cube_sum_equals_94_l2101_210154

theorem cube_sum_equals_94 (a b c : ℝ) 
  (h1 : a + b + c = 7) 
  (h2 : a * b + a * c + b * c = 11) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = 94 := by
sorry

end cube_sum_equals_94_l2101_210154


namespace even_and_mono_decreasing_implies_ordering_l2101_210160

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define a monotonically decreasing function on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- Main theorem
theorem even_and_mono_decreasing_implies_ordering (f : ℝ → ℝ)
  (h1 : EvenFunction f)
  (h2 : MonoDecreasing (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end even_and_mono_decreasing_implies_ordering_l2101_210160


namespace perimeter_inequality_l2101_210140

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the foot of the perpendicular from a point to a line -/
def perpendicularFoot (p : Point) (l : Point × Point) : Point := sorry

/-- Main theorem -/
theorem perimeter_inequality 
  (ABC : Triangle) 
  (h_acute : isAcute ABC) 
  (D : Point) (E : Point) (F : Point)
  (P : Point) (Q : Point) (R : Point)
  (h_D : D = perpendicularFoot ABC.A (ABC.B, ABC.C))
  (h_E : E = perpendicularFoot ABC.B (ABC.C, ABC.A))
  (h_F : F = perpendicularFoot ABC.C (ABC.A, ABC.B))
  (h_P : P = perpendicularFoot ABC.A (E, F))
  (h_Q : Q = perpendicularFoot ABC.B (F, D))
  (h_R : R = perpendicularFoot ABC.C (D, E))
  : perimeter ABC * perimeter {A := P, B := Q, C := R} ≥ (perimeter {A := D, B := E, C := F})^2 := by
  sorry

end perimeter_inequality_l2101_210140


namespace sum_of_cubes_l2101_210193

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by sorry

end sum_of_cubes_l2101_210193


namespace max_value_part_i_one_root_condition_part_ii_inequality_condition_part_iii_l2101_210113

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

-- Part I
theorem max_value_part_i :
  ∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 0, (f 1 x) * (g x) ≤ M :=
sorry

-- Part II
theorem one_root_condition_part_ii :
  ∀ k : ℝ, (∃! x : ℝ, f (-1) x = k * g x) ↔ 
  (k > 0 ∧ k < Real.exp (-1)) ∨ (k > 3 * Real.exp (-2)) :=
sorry

-- Part III
theorem inequality_condition_part_iii :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔ 
  (a ≥ -1 ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end

end max_value_part_i_one_root_condition_part_ii_inequality_condition_part_iii_l2101_210113


namespace cube_volume_l2101_210153

theorem cube_volume (n : ℝ) : 
  (∃ (s : ℝ), s * Real.sqrt 2 = 4 ∧ s^3 = n * Real.sqrt 2) → n = 16 := by
  sorry

end cube_volume_l2101_210153


namespace last_segment_speed_l2101_210182

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 96)
  (h2 : total_time = 90 / 60)
  (h3 : speed1 = 60)
  (h4 : speed2 = 65)
  (h5 : (speed1 + speed2 + (3 * total_distance / total_time - speed1 - speed2)) / 3 = total_distance / total_time) :
  3 * total_distance / total_time - speed1 - speed2 = 67 := by
  sorry

end last_segment_speed_l2101_210182


namespace quadratic_inequality_specific_case_l2101_210118

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
by sorry

theorem specific_case :
  ∀ x : ℝ, x^2 - 5*x + 4 > 0 ↔ x < 1 ∨ x > 4 :=
by sorry

end quadratic_inequality_specific_case_l2101_210118


namespace farm_animal_ratio_l2101_210108

theorem farm_animal_ratio : 
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let total_goats_chickens : ℕ := goats + chickens
  let ducks : ℕ := 99  -- We define this to match the problem constraints
  let pigs : ℕ := ducks / 3
  goats = pigs + 33 →
  (ducks : ℚ) / total_goats_chickens = 1 / 2 := by
sorry

end farm_animal_ratio_l2101_210108


namespace brick_width_is_10_l2101_210191

-- Define the dimensions of the brick and wall
def brick_length : ℝ := 20
def brick_height : ℝ := 7.5
def wall_length : ℝ := 2700  -- 27 m in cm
def wall_width : ℝ := 200    -- 2 m in cm
def wall_height : ℝ := 75    -- 0.75 m in cm
def num_bricks : ℕ := 27000

-- Theorem to prove the width of the brick
theorem brick_width_is_10 :
  ∃ (brick_width : ℝ),
    brick_width = 10 ∧
    brick_length * brick_width * brick_height * num_bricks =
    wall_length * wall_width * wall_height :=
by sorry

end brick_width_is_10_l2101_210191


namespace problem_statement_l2101_210138

def p : Prop := ∀ x : ℝ, x^2 - 1 ≥ -1

def q : Prop := 4 + 2 = 7

theorem problem_statement : 
  p ∧ ¬q ∧ ¬(p ∧ q) ∧ (p ∨ q) := by sorry

end problem_statement_l2101_210138


namespace cookie_count_consistency_l2101_210178

theorem cookie_count_consistency (total_cookies : ℕ) (eaten_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 32)
  (h2 : eaten_cookies = 9)
  (h3 : remaining_cookies = 23) :
  total_cookies - eaten_cookies = remaining_cookies := by
  sorry

#check cookie_count_consistency

end cookie_count_consistency_l2101_210178


namespace alcohol_concentration_proof_l2101_210105

theorem alcohol_concentration_proof :
  ∀ (vessel1_capacity vessel2_capacity total_liquid final_capacity : ℝ)
    (vessel2_concentration final_concentration : ℝ),
  vessel1_capacity = 2 →
  vessel2_capacity = 6 →
  vessel2_concentration = 0.4 →
  total_liquid = 8 →
  final_capacity = 10 →
  final_concentration = 0.29000000000000004 →
  ∃ (vessel1_concentration : ℝ),
    vessel1_concentration = 0.25 ∧
    vessel1_concentration * vessel1_capacity + vessel2_concentration * vessel2_capacity =
      final_concentration * final_capacity :=
by
  sorry

#check alcohol_concentration_proof

end alcohol_concentration_proof_l2101_210105


namespace soda_consumption_theorem_l2101_210127

/-- The number of bottles of soda left after a given period -/
def bottles_left (bottles_per_pack : ℕ) (packs_bought : ℕ) (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  (bottles_per_pack * packs_bought : ℚ) - (bottles_per_day * days)

/-- Theorem stating that given the conditions, 4 bottles will be left after 4 weeks -/
theorem soda_consumption_theorem :
  bottles_left 6 3 (1/2) (4 * 7) = 4 := by
  sorry

end soda_consumption_theorem_l2101_210127


namespace max_real_part_of_roots_l2101_210124

open Complex

-- Define the polynomial
def p (z : ℂ) : ℂ := z^6 - z^4 + z^2 - 1

-- Theorem statement
theorem max_real_part_of_roots :
  ∃ (z : ℂ), p z = 0 ∧ 
  ∀ (w : ℂ), p w = 0 → z.re ≥ w.re ∧
  z.re = 1 := by
  sorry

end max_real_part_of_roots_l2101_210124


namespace total_profit_is_135000_l2101_210103

/-- Represents an investor in the partnership business -/
structure Investor where
  name : String
  investment : ℕ
  months : ℕ

/-- Calculates the total profit given the investors and C's profit share -/
def calculateTotalProfit (investors : List Investor) (cProfit : ℕ) : ℕ :=
  let totalInvestmentMonths := investors.map (λ i => i.investment * i.months) |>.sum
  let cInvestmentMonths := (investors.find? (λ i => i.name = "C")).map (λ i => i.investment * i.months)
  match cInvestmentMonths with
  | some im => cProfit * totalInvestmentMonths / im
  | none => 0

/-- Theorem stating that the total profit is 135000 given the specified conditions -/
theorem total_profit_is_135000 (investors : List Investor) (h1 : investors.length = 5)
    (h2 : investors.any (λ i => i.name = "A" ∧ i.investment = 12000 ∧ i.months = 6))
    (h3 : investors.any (λ i => i.name = "B" ∧ i.investment = 16000 ∧ i.months = 12))
    (h4 : investors.any (λ i => i.name = "C" ∧ i.investment = 20000 ∧ i.months = 12))
    (h5 : investors.any (λ i => i.name = "D" ∧ i.investment = 24000 ∧ i.months = 12))
    (h6 : investors.any (λ i => i.name = "E" ∧ i.investment = 18000 ∧ i.months = 6))
    (h7 : calculateTotalProfit investors 36000 = 135000) : 
  calculateTotalProfit investors 36000 = 135000 := by
  sorry


end total_profit_is_135000_l2101_210103


namespace borrowing_methods_count_l2101_210173

/-- Represents the number of books of each type -/
structure BookCounts where
  physics : Nat
  history : Nat
  mathematics : Nat

/-- Represents the number of students of each type -/
structure StudentCounts where
  science : Nat
  liberal_arts : Nat

/-- Calculates the number of ways to distribute books to students -/
def calculate_borrowing_methods (books : BookCounts) (students : StudentCounts) : Nat :=
  sorry

/-- Theorem stating the correct number of borrowing methods -/
theorem borrowing_methods_count :
  let books := BookCounts.mk 3 2 4
  let students := StudentCounts.mk 4 3
  calculate_borrowing_methods books students = 76 := by
  sorry

end borrowing_methods_count_l2101_210173


namespace digit_123_is_1_l2101_210100

/-- The decimal representation of 47/740 -/
def decimal_rep : ℚ := 47 / 740

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 12

/-- The position we're interested in -/
def target_position : ℕ := 123

/-- The function that returns the nth digit after the decimal point in the decimal representation of 47/740 -/
noncomputable def nth_digit (n : ℕ) : ℕ :=
  sorry

theorem digit_123_is_1 : nth_digit target_position = 1 := by
  sorry

end digit_123_is_1_l2101_210100


namespace no_three_digit_base_7_equals_two_digit_base_6_l2101_210102

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if a number is representable as a two-digit number in base 6 --/
def is_two_digit_base_6 (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), d1 < 6 ∧ d2 < 6 ∧ n = to_base_10 [d1, d2] 6

theorem no_three_digit_base_7_equals_two_digit_base_6 :
  ¬ ∃ (d1 d2 d3 : ℕ), 
    d1 > 0 ∧ d1 < 7 ∧ d2 < 7 ∧ d3 < 7 ∧ 
    is_two_digit_base_6 (to_base_10 [d1, d2, d3] 7) :=
by sorry

end no_three_digit_base_7_equals_two_digit_base_6_l2101_210102


namespace rectangle_area_l2101_210128

/-- Given a rectangle with length 15 cm and perimeter-to-width ratio of 5:1, its area is 150 cm² -/
theorem rectangle_area (w : ℝ) (h1 : (2 * 15 + 2 * w) / w = 5) : w * 15 = 150 := by
  sorry

end rectangle_area_l2101_210128


namespace determine_y_from_one_point_determine_y_from_k_one_additional_data_necessary_and_sufficient_l2101_210115

/-- A structure representing a proportional relationship between x and y --/
structure ProportionalRelationship where
  k : ℝ  -- Constant of proportionality
  proportional : ∀ (x y : ℝ), y = k * x

/-- Given a proportional relationship and one point, we can determine y for any x --/
theorem determine_y_from_one_point 
  (rel : ProportionalRelationship) (x₀ y₀ : ℝ) (h : y₀ = rel.k * x₀) :
  ∀ (x : ℝ), ∃! (y : ℝ), y = rel.k * x :=
sorry

/-- Given a proportional relationship and k, we can determine y for any x --/
theorem determine_y_from_k (rel : ProportionalRelationship) :
  ∀ (x : ℝ), ∃! (y : ℝ), y = rel.k * x :=
sorry

/-- One additional piece of data (either k or a point) is necessary and sufficient --/
theorem one_additional_data_necessary_and_sufficient :
  ∀ (x y : ℝ → ℝ), (∃ (k : ℝ), ∀ (t : ℝ), y t = k * x t) →
  ((∃ (k : ℝ), ∀ (t : ℝ), y t = k * x t) ∨ 
   (∃ (x₀ y₀ : ℝ), y x₀ = y₀ ∧ ∀ (t : ℝ), y t = (y₀ / x₀) * x t)) ∧
  (∀ (t : ℝ), ∃! (yt : ℝ), y t = yt) :=
sorry

end determine_y_from_one_point_determine_y_from_k_one_additional_data_necessary_and_sufficient_l2101_210115


namespace function_properties_l2101_210176

/-- The function f(x) = x³ + 2ax² + bx + a -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b

theorem function_properties (a b : ℝ) :
  f a b (-1) = 1 ∧ f_derivative a b (-1) = 0 →
  a = 1 ∧ b = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 1 x ≤ 5 ∧ f 1 1 1 = 5 :=
by sorry

end function_properties_l2101_210176


namespace bugs_meeting_time_l2101_210155

/-- Two circles tangent at point O with radii 7 and 3 inches, and bugs moving at 4π and 3π inches per minute respectively -/
structure CircleSetup where
  r1 : ℝ
  r2 : ℝ
  v1 : ℝ
  v2 : ℝ
  h_r1 : r1 = 7
  h_r2 : r2 = 3
  h_v1 : v1 = 4 * Real.pi
  h_v2 : v2 = 3 * Real.pi

/-- Time taken for bugs to meet again at point O -/
def meetingTime (setup : CircleSetup) : ℝ :=
  7

/-- Theorem stating that the meeting time is 7 minutes -/
theorem bugs_meeting_time (setup : CircleSetup) :
  meetingTime setup = 7 := by
  sorry

end bugs_meeting_time_l2101_210155


namespace number_puzzle_solution_l2101_210143

theorem number_puzzle_solution : 
  ∃ x : ℚ, x - (3/5) * x = 50 ∧ x = 125 := by
  sorry

end number_puzzle_solution_l2101_210143


namespace approximate_12000_accuracy_l2101_210170

/-- Represents an approximate number with its value and significant digits -/
structure ApproximateNumber where
  value : ℕ
  significantDigits : ℕ

/-- Determines the number of significant digits in an approximate number -/
def countSignificantDigits (n : ℕ) : ℕ :=
  sorry

theorem approximate_12000_accuracy :
  let n : ApproximateNumber := ⟨12000, countSignificantDigits 12000⟩
  n.significantDigits = 2 := by sorry

end approximate_12000_accuracy_l2101_210170


namespace equilateral_triangle_area_perimeter_ratio_l2101_210125

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  (area / perimeter) = (5 * Real.sqrt 3) / 6 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l2101_210125


namespace min_value_expression_l2101_210175

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 3 := by
  sorry

end min_value_expression_l2101_210175


namespace cylinder_line_segment_distance_l2101_210121

/-- Represents a cylinder with a square axial cross-section -/
structure SquareCylinder where
  -- We don't need to define specific properties here

/-- Represents a line segment connecting points on the top and bottom bases of the cylinder -/
structure LineSegment where
  length : ℝ
  angle : ℝ

/-- 
Theorem: For a cylinder with a square axial cross-section, given a line segment of length l 
connecting points on the top and bottom base circumferences and making an angle α with the base plane, 
the distance d from this line segment to the cylinder axis is (l/2) * sqrt(-cos(2α)), 
and the valid range for α is π/4 < α < 3π/4.
-/
theorem cylinder_line_segment_distance (c : SquareCylinder) (seg : LineSegment) :
  let l := seg.length
  let α := seg.angle
  let d := (l / 2) * Real.sqrt (-Real.cos (2 * α))
  d > 0 ∧ π / 4 < α ∧ α < 3 * π / 4 := by
  sorry


end cylinder_line_segment_distance_l2101_210121


namespace yard_length_26_trees_l2101_210106

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees, 
    with trees at each end and 12 meters between consecutive trees, is 300 meters -/
theorem yard_length_26_trees : 
  yard_length 26 12 = 300 := by sorry

end yard_length_26_trees_l2101_210106


namespace special_line_equation_l2101_210144

/-- A line passing through point (1,4) with the sum of its x and y intercepts equal to zero -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (1,4) -/
  passes_through_point : slope * 1 + y_intercept = 4
  /-- The sum of x and y intercepts is zero -/
  sum_of_intercepts_zero : (-y_intercept / slope) + y_intercept = 0

/-- The equation of a SpecialLine is either 4x - y = 0 or x - y + 3 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end special_line_equation_l2101_210144


namespace cube_adjacent_diagonals_perpendicular_l2101_210181

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- A face diagonal is a line segment that connects opposite corners of a face -/
structure FaceDiagonal where
  cube : Cube
  face : Nat  -- We can use natural numbers to identify faces (1 to 6)

/-- The angle between two face diagonals -/
def angle_between_diagonals (d1 d2 : FaceDiagonal) : ℝ := sorry

/-- Two faces of a cube are adjacent if they share an edge -/
def adjacent_faces (f1 f2 : Nat) : Prop := sorry

/-- Theorem: The angle between the diagonals of any two adjacent faces of a cube is 90 degrees -/
theorem cube_adjacent_diagonals_perpendicular (c : Cube) (f1 f2 : Nat) (d1 d2 : FaceDiagonal)
  (h1 : d1.cube = c) (h2 : d2.cube = c) (h3 : d1.face = f1) (h4 : d2.face = f2)
  (h5 : adjacent_faces f1 f2) :
  angle_between_diagonals d1 d2 = 90 := by sorry

end cube_adjacent_diagonals_perpendicular_l2101_210181


namespace sum_of_fractions_geq_four_l2101_210165

theorem sum_of_fractions_geq_four (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a * d + b * c) / (b * d) + (b * c + a * d) / (a * c) ≥ 4 := by
  sorry

end sum_of_fractions_geq_four_l2101_210165


namespace expression_evaluation_l2101_210147

theorem expression_evaluation : 
  (((15^15 / 15^14)^3 * 8^3) / 4^6 : ℚ) = 1728000 / 4096 := by
  sorry

end expression_evaluation_l2101_210147


namespace max_value_of_d_l2101_210110

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + 5 * Real.sqrt 21) / 2 := by
sorry

end max_value_of_d_l2101_210110


namespace dihedral_angle_cosine_l2101_210199

/-- Represents a regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a circle in 3D space -/
structure Circle3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Function to check if a line intersects the base side of the pyramid -/
def intersectsBaseSide (p : RegularHexagonalPyramid) (l : Line3D) : Prop :=
  sorry

/-- Function to get the circle inscribed around a lateral face -/
def lateralFaceInscribedCircle (p : RegularHexagonalPyramid) : Circle3D :=
  sorry

/-- Function to get the circle inscribed around the larger diagonal cross-section -/
def diagonalCrossSectionInscribedCircle (p : RegularHexagonalPyramid) : Circle3D :=
  sorry

/-- Function to check if a line passes through the centers of two circles -/
def passesThroughCenters (l : Line3D) (c1 c2 : Circle3D) : Prop :=
  sorry

/-- Function to calculate the dihedral angle at the base -/
def dihedralAngleAtBase (p : RegularHexagonalPyramid) : ℝ :=
  sorry

theorem dihedral_angle_cosine (p : RegularHexagonalPyramid) (l : Line3D) :
  intersectsBaseSide p l ∧
  passesThroughCenters l (lateralFaceInscribedCircle p) (diagonalCrossSectionInscribedCircle p) →
  Real.cos (dihedralAngleAtBase p) = Real.sqrt (3 / 13) :=
  sorry

end dihedral_angle_cosine_l2101_210199


namespace point_transformation_l2101_210168

-- Define the rotation function
def rotate180 (x y : ℝ) : ℝ × ℝ := (2 - x, 10 - y)

-- Define the reflection function
def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ := (y, x)

-- Theorem statement
theorem point_transformation (a b : ℝ) :
  let (x', y') := rotate180 a b
  let (x'', y'') := reflect_y_eq_x x' y'
  (x'' = 3 ∧ y'' = -6) → b - a = -1 := by
  sorry

end point_transformation_l2101_210168


namespace unique_intersection_point_l2101_210136

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- Theorem statement
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-3, -3) :=
sorry

end unique_intersection_point_l2101_210136


namespace quadratic_prime_roots_unique_k_l2101_210197

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots_unique_k : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p + q = 58 ∧ 
    p * q = k ∧ 
    ∀ x : ℝ, x^2 - 58*x + k = 0 ↔ (x = p ∨ x = q) :=
sorry

end quadratic_prime_roots_unique_k_l2101_210197


namespace shaded_region_perimeter_l2101_210107

/-- The perimeter of a region formed by four 90° arcs of circles with circumference 48 -/
theorem shaded_region_perimeter (c : ℝ) (h : c = 48) : 
  4 * (90 / 360 * c) = 48 := by sorry

end shaded_region_perimeter_l2101_210107


namespace greatest_constant_for_triangle_inequality_l2101_210166

theorem greatest_constant_for_triangle_inequality (a b c : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (a + b > c) → (b + c > a) → (c + a > b) →
  (∃ (N : ℝ), ∀ (a b c : ℝ), 
    (a > 0) → (b > 0) → (c > 0) →
    (a + b > c) → (b + c > a) → (c + a > b) →
    (a^2 + b^2 + a*b) / c^2 > N) ∧
  (∀ (M : ℝ), 
    (∀ (a b c : ℝ), 
      (a > 0) → (b > 0) → (c > 0) →
      (a + b > c) → (b + c > a) → (c + a > b) →
      (a^2 + b^2 + a*b) / c^2 > M) →
    M ≤ 3/4) :=
sorry

end greatest_constant_for_triangle_inequality_l2101_210166


namespace percentage_of_non_roses_l2101_210104

theorem percentage_of_non_roses (roses tulips daisies : ℕ) 
  (h_roses : roses = 25)
  (h_tulips : tulips = 40)
  (h_daisies : daisies = 35) :
  (100 : ℚ) * (tulips + daisies : ℚ) / (roses + tulips + daisies : ℚ) = 75 := by
  sorry

end percentage_of_non_roses_l2101_210104


namespace line_through_point_with_slope_l2101_210139

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The equation of the line in the form ax + by + c = 0 -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  l.slope * x - y + l.yIntercept = 0

theorem line_through_point_with_slope (x₀ y₀ m : ℝ) :
  ∃ (l : Line), l.slope = m ∧ l.containsPoint x₀ y₀ ∧
  ∀ (x y : ℝ), l.equation x y ↔ (2 : ℝ) * x - y + 3 = 0 :=
sorry

end line_through_point_with_slope_l2101_210139


namespace zain_coins_count_and_value_l2101_210190

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100
def half_dollar_value : ℚ := 50 / 100

def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def emerie_pennies : ℕ := 10
def emerie_half_dollars : ℕ := 2

def zain_more_coins : ℕ := 10

def zain_quarters : ℕ := emerie_quarters + zain_more_coins
def zain_dimes : ℕ := emerie_dimes + zain_more_coins
def zain_nickels : ℕ := emerie_nickels + zain_more_coins
def zain_pennies : ℕ := emerie_pennies + zain_more_coins
def zain_half_dollars : ℕ := emerie_half_dollars + zain_more_coins

def zain_total_coins : ℕ := zain_quarters + zain_dimes + zain_nickels + zain_pennies + zain_half_dollars

def zain_total_value : ℚ :=
  zain_quarters * quarter_value +
  zain_dimes * dime_value +
  zain_nickels * nickel_value +
  zain_pennies * penny_value +
  zain_half_dollars * half_dollar_value

theorem zain_coins_count_and_value :
  zain_total_coins = 80 ∧ zain_total_value ≤ 20 := by sorry

end zain_coins_count_and_value_l2101_210190


namespace factor_tree_proof_l2101_210114

theorem factor_tree_proof (A B C D E : ℕ) 
  (hB : B = 4 * D)
  (hC : C = 7 * E)
  (hA : A = B * C)
  (hD : D = 4 * 3)
  (hE : E = 7 * 3) :
  A = 7056 := by
  sorry

end factor_tree_proof_l2101_210114


namespace max_child_fraction_is_11_20_l2101_210169

/-- Represents the babysitting scenario for Jane -/
structure BabysittingScenario where
  jane_start_age : ℕ
  jane_current_age : ℕ
  years_since_stopped : ℕ
  oldest_babysat_current_age : ℕ

/-- The maximum fraction of Jane's age that a child she babysat could be -/
def max_child_fraction (scenario : BabysittingScenario) : ℚ :=
  let jane_stop_age := scenario.jane_current_age - scenario.years_since_stopped
  let child_age_when_jane_stopped := scenario.oldest_babysat_current_age - scenario.years_since_stopped
  child_age_when_jane_stopped / jane_stop_age

/-- The theorem stating the maximum fraction of Jane's age a child could be -/
theorem max_child_fraction_is_11_20 (scenario : BabysittingScenario)
  (h1 : scenario.jane_start_age = 18)
  (h2 : scenario.jane_current_age = 32)
  (h3 : scenario.years_since_stopped = 12)
  (h4 : scenario.oldest_babysat_current_age = 23) :
  max_child_fraction scenario = 11/20 := by
  sorry

end max_child_fraction_is_11_20_l2101_210169


namespace caiden_roofing_problem_l2101_210123

theorem caiden_roofing_problem (cost_per_foot : ℝ) (free_feet : ℝ) (remaining_cost : ℝ) :
  cost_per_foot = 8 →
  free_feet = 250 →
  remaining_cost = 400 →
  ∃ (total_feet : ℝ), total_feet = 300 ∧ (total_feet - free_feet) * cost_per_foot = remaining_cost :=
by sorry

end caiden_roofing_problem_l2101_210123


namespace sum_of_powers_divisible_by_ten_l2101_210163

theorem sum_of_powers_divisible_by_ten (n : ℕ) (h : ¬ (4 ∣ n)) :
  10 ∣ (1^n + 2^n + 3^n + 4^n) := by
  sorry

end sum_of_powers_divisible_by_ten_l2101_210163


namespace divide_into_triominoes_l2101_210119

/-- An L-shaped triomino is a shape consisting of three connected cells in an L shape -/
def LShapedTriomino : Type := Unit

/-- A grid is represented by its size, which is always of the form 6n+1 for some natural number n -/
structure Grid :=
  (n : ℕ)

/-- A cell in the grid, represented by its row and column coordinates -/
structure Cell :=
  (row : ℕ)
  (col : ℕ)

/-- A configuration is a grid with one cell removed -/
structure Configuration :=
  (grid : Grid)
  (removed_cell : Cell)

/-- A division of a configuration into L-shaped triominoes -/
def Division (config : Configuration) : Type := Unit

/-- The main theorem: any configuration can be divided into L-shaped triominoes -/
theorem divide_into_triominoes (config : Configuration) : 
  ∃ (d : Division config), True :=
sorry

end divide_into_triominoes_l2101_210119


namespace garden_fencing_theorem_l2101_210134

/-- Calculates the perimeter of a rectangular garden with given length and width. -/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: A rectangular garden with length 60 yards and width equal to half its length
    requires 180 yards of fencing to enclose it. -/
theorem garden_fencing_theorem :
  let length : ℝ := 60
  let width : ℝ := length / 2
  garden_perimeter length width = 180 := by
sorry

end garden_fencing_theorem_l2101_210134


namespace polygon_sides_l2101_210162

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : 
  n > 2 ∧ sum_angles = 2190 ∧ sum_angles = (n - 3) * 180 → n = 15 := by
  sorry

end polygon_sides_l2101_210162


namespace gasoline_spending_increase_l2101_210196

theorem gasoline_spending_increase (P Q : ℝ) (P_positive : P > 0) (Q_positive : Q > 0) :
  let new_price := 1.25 * P
  let new_quantity := 0.88 * Q
  let original_spending := P * Q
  let new_spending := new_price * new_quantity
  (new_spending - original_spending) / original_spending = 0.1 := by
sorry

end gasoline_spending_increase_l2101_210196


namespace min_distance_circle_parabola_l2101_210142

/-- The minimum distance between a point on a circle and a point on a parabola -/
theorem min_distance_circle_parabola :
  ∀ (A B : ℝ × ℝ),
  (A.1^2 + A.2^2 = 16) →
  (B.2 = B.1^2 - 4) →
  (∃ (θ : ℝ), A = (4 * Real.cos θ, 4 * Real.sin θ)) →
  (∃ (x : ℝ), B = (x, x^2 - 4)) →
  (∃ (d : ℝ), d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (∀ (d' : ℝ), d' ≥ d) →
  (∃ (x : ℝ), -2*(4*Real.cos θ - x) + 2*(4*Real.sin θ - (x^2 - 4))*(-2*x) = 0) :=
by sorry

end min_distance_circle_parabola_l2101_210142


namespace surface_area_cube_with_holes_l2101_210120

/-- The surface area of a cube with smaller cubes dug out from each face -/
theorem surface_area_cube_with_holes (edge_length : ℝ) (hole_length : ℝ) : 
  edge_length = 10 →
  hole_length = 2 →
  (6 * edge_length^2) - (6 * hole_length^2) + (6 * 5 * hole_length^2) = 696 := by
  sorry

end surface_area_cube_with_holes_l2101_210120


namespace equal_candies_after_sharing_l2101_210151

/-- Proves that Minyoung and Taehyung will have the same number of candies
    if Minyoung gives 3 candies to Taehyung. -/
theorem equal_candies_after_sharing (minyoung_initial : ℕ) (taehyung_initial : ℕ) 
  (candies_shared : ℕ) : 
  minyoung_initial = 9 →
  taehyung_initial = 3 →
  candies_shared = 3 →
  minyoung_initial - candies_shared = taehyung_initial + candies_shared :=
by
  sorry

#check equal_candies_after_sharing

end equal_candies_after_sharing_l2101_210151


namespace prime_pairs_divisibility_l2101_210111

theorem prime_pairs_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → 
  (∃ k : ℤ, 30 * q - 1 = k * p) → 
  (∃ m : ℤ, 30 * p - 1 = m * q) → 
  ((p = 7 ∧ q = 11) ∨ (p = 11 ∧ q = 7) ∨ (p = 59 ∧ q = 61) ∨ (p = 61 ∧ q = 59)) :=
by sorry

end prime_pairs_divisibility_l2101_210111


namespace reciprocal_of_negative_one_sixth_l2101_210174

theorem reciprocal_of_negative_one_sixth : 
  ((-1 / 6 : ℚ)⁻¹ : ℚ) = -6 := by sorry

end reciprocal_of_negative_one_sixth_l2101_210174


namespace intersection_y_intercept_sum_l2101_210186

/-- Given two lines that intersect at (3,3), prove that the sum of their y-intercepts is 4 -/
theorem intersection_y_intercept_sum (c d : ℝ) : 
  (3 = (1/3)*3 + c) → (3 = (1/3)*3 + d) → c + d = 4 := by
  sorry

end intersection_y_intercept_sum_l2101_210186


namespace alvin_wood_gathering_l2101_210145

theorem alvin_wood_gathering (total_needed wood_from_friend wood_from_brother : ℕ) 
  (h1 : total_needed = 376)
  (h2 : wood_from_friend = 123)
  (h3 : wood_from_brother = 136) :
  total_needed - (wood_from_friend + wood_from_brother) = 117 := by
  sorry

end alvin_wood_gathering_l2101_210145


namespace f_greater_than_one_factorial_inequality_l2101_210122

noncomputable def f (x : ℝ) : ℝ := (1/x + 1/2) * Real.log (x + 1)

theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f x > 1 := by sorry

theorem factorial_inequality (n : ℕ) :
  5/6 < Real.log (n.factorial : ℝ) - (n + 1/2) * Real.log n + n ∧
  Real.log (n.factorial : ℝ) - (n + 1/2) * Real.log n + n ≤ 1 := by sorry

end f_greater_than_one_factorial_inequality_l2101_210122


namespace tom_nail_purchase_l2101_210156

/-- The number of additional nails Tom needs to buy for his project -/
def additional_nails_needed (initial : ℝ) (toolshed : ℝ) (drawer : ℝ) (neighbor : ℝ) (thank_you : ℝ) (required : ℝ) : ℝ :=
  required - (initial + toolshed + drawer + neighbor + thank_you)

/-- Theorem stating the number of additional nails Tom needs to buy -/
theorem tom_nail_purchase (initial : ℝ) (toolshed : ℝ) (drawer : ℝ) (neighbor : ℝ) (thank_you : ℝ) (required : ℝ)
    (h1 : initial = 247.5)
    (h2 : toolshed = 144.25)
    (h3 : drawer = 0.75)
    (h4 : neighbor = 58.75)
    (h5 : thank_you = 37.25)
    (h6 : required = 761.58) :
    additional_nails_needed initial toolshed drawer neighbor thank_you required = 273.08 := by
  sorry

end tom_nail_purchase_l2101_210156


namespace intersection_M_N_l2101_210101

def M : Set ℕ := {1, 2, 3, 5, 7}

def N : Set ℕ := {x | ∃ k ∈ M, x = 2 * k - 1}

theorem intersection_M_N : M ∩ N = {1, 3, 5} := by
  sorry

end intersection_M_N_l2101_210101


namespace balance_scale_theorem_l2101_210172

/-- Represents a weight on the balance scale -/
structure Weight where
  pan : Bool  -- true for left pan, false for right pan
  value : ℝ
  number : ℕ

/-- Represents the state of the balance scale -/
structure BalanceScale where
  k : ℕ  -- number of weights on each pan
  weights : List Weight

/-- Checks if the left pan is heavier -/
def leftPanHeavier (scale : BalanceScale) : Prop :=
  let leftSum := (scale.weights.filter (fun w => w.pan)).map (fun w => w.value) |>.sum
  let rightSum := (scale.weights.filter (fun w => !w.pan)).map (fun w => w.value) |>.sum
  leftSum > rightSum

/-- Checks if swapping weights with the same number makes the right pan heavier or balances the pans -/
def swapMakesRightHeavierOrBalance (scale : BalanceScale) : Prop :=
  ∀ i, i ≤ scale.k →
    let swappedWeights := scale.weights.map (fun w => if w.number = i then { w with pan := !w.pan } else w)
    let swappedLeftSum := (swappedWeights.filter (fun w => w.pan)).map (fun w => w.value) |>.sum
    let swappedRightSum := (swappedWeights.filter (fun w => !w.pan)).map (fun w => w.value) |>.sum
    swappedRightSum ≥ swappedLeftSum

/-- The main theorem stating that k can only be 1 or 2 -/
theorem balance_scale_theorem (scale : BalanceScale) :
  leftPanHeavier scale →
  swapMakesRightHeavierOrBalance scale →
  scale.k = 1 ∨ scale.k = 2 :=
by
  sorry

end balance_scale_theorem_l2101_210172


namespace similar_quadrilaterals_rectangle_areas_l2101_210149

/-- Given two similar quadrilaterals with sides (a, b, c, d) and (a', b', c', d') respectively,
    prove that the areas of rectangles formed by pairs of corresponding sides
    are in proportion to the squares of the sides of the original quadrilaterals. -/
theorem similar_quadrilaterals_rectangle_areas
  (a b c d a' b' c' d' : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c ∧ d' = k * d) :
  ∃ (m : ℝ), m > 0 ∧
    a * a' / (b * b') = a^2 / b^2 ∧
    b * b' / (c * c') = b^2 / c^2 ∧
    c * c' / (d * d') = c^2 / d^2 ∧
    d * d' / (a * a') = d^2 / a^2 :=
by sorry

end similar_quadrilaterals_rectangle_areas_l2101_210149


namespace pyramid_sphere_theorem_l2101_210179

/-- Represents a triangular pyramid with a sphere touching its edges -/
structure PyramidWithSphere where
  -- Base triangle side length
  base_side : ℝ
  -- Height of the pyramid
  height : ℝ
  -- Radius of the inscribed sphere
  sphere_radius : ℝ

/-- Properties of the pyramid and sphere system -/
def pyramid_sphere_properties (p : PyramidWithSphere) : Prop :=
  -- Base is an equilateral triangle
  p.base_side = 8 ∧
  -- Height of the pyramid
  p.height = 15 ∧
  -- Sphere touches edges of the pyramid
  ∃ (aa₁ : ℝ) (dist_o_bc : ℝ),
    -- Distance from vertex A to point of contact A₁
    aa₁ = 6 ∧
    -- Distance from sphere center O to edge BC
    dist_o_bc = 18 / 5 ∧
    -- Radius of the sphere
    p.sphere_radius = 4 * Real.sqrt 39 / 5

/-- Theorem stating the properties of the pyramid and sphere system -/
theorem pyramid_sphere_theorem (p : PyramidWithSphere) :
  pyramid_sphere_properties p → 
  ∃ (aa₁ : ℝ) (dist_o_bc : ℝ),
    aa₁ = 6 ∧
    dist_o_bc = 18 / 5 ∧
    p.sphere_radius = 4 * Real.sqrt 39 / 5 :=
by sorry

end pyramid_sphere_theorem_l2101_210179


namespace fraction_unchanged_l2101_210130

theorem fraction_unchanged (x y : ℝ) (h : x + y ≠ 0) :
  (3 * (2 * y)) / (2 * x + 2 * y) = (3 * y) / (x + y) :=
by sorry

end fraction_unchanged_l2101_210130


namespace simple_interest_problem_l2101_210188

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4016.25 →
  rate = 9 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 8925 := by
sorry

end simple_interest_problem_l2101_210188


namespace sum_of_roots_eq_one_l2101_210195

theorem sum_of_roots_eq_one : 
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 4) - 20
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 1 :=
by sorry

end sum_of_roots_eq_one_l2101_210195


namespace selenas_remaining_money_is_38_l2101_210194

/-- Calculates the remaining money for Selena after her meal -/
def selenas_remaining_money (tip : ℚ) (steak_price : ℚ) (steak_count : ℕ) 
  (burger_price : ℚ) (burger_count : ℕ) (icecream_price : ℚ) (icecream_count : ℕ) : ℚ :=
  tip - (steak_price * steak_count + burger_price * burger_count + icecream_price * icecream_count)

/-- Theorem stating that Selena will be left with $38 after her meal -/
theorem selenas_remaining_money_is_38 :
  selenas_remaining_money 99 24 2 3.5 2 2 3 = 38 := by
  sorry

end selenas_remaining_money_is_38_l2101_210194


namespace expression_factorization_l2101_210141

theorem expression_factorization (x : ℝ) : 
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x) = 3 * x * (5 * x^3 - 7 * x^2 + 12) := by
  sorry

end expression_factorization_l2101_210141


namespace absolute_difference_x_y_l2101_210171

theorem absolute_difference_x_y (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 2.4)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 5.1) : 
  |x - y| = 3.3 := by
sorry

end absolute_difference_x_y_l2101_210171


namespace lily_correct_answers_percentage_l2101_210135

theorem lily_correct_answers_percentage
  (t : ℝ)
  (h_t_positive : t > 0)
  (h_max_alone : 0.7 * (t / 2) = 0.35 * t)
  (h_max_total : 0.82 * t = 0.82 * t)
  (h_lily_alone : 0.85 * (t / 2) = 0.425 * t)
  (h_solved_together : 0.82 * t - 0.35 * t = 0.47 * t) :
  (0.425 * t + 0.47 * t) / t = 0.895 := by
  sorry

#check lily_correct_answers_percentage

end lily_correct_answers_percentage_l2101_210135


namespace current_speed_l2101_210137

/-- The speed of the current given a motorboat's constant speed and trip times -/
theorem current_speed (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 30)
  (h2 : upstream_time = 40 / 60)
  (h3 : downstream_time = 25 / 60) :
  ∃ c : ℝ, c = 90 / 13 ∧ 
  (boat_speed - c) * upstream_time = (boat_speed + c) * downstream_time :=
sorry

end current_speed_l2101_210137


namespace parabola_intersection_l2101_210185

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 9 * x - 8
  let g (x : ℝ) := x^2 - 3 * x + 4
  (f 3 = g 3 ∧ f 3 = -8) ∧ (f (-2) = g (-2) ∧ f (-2) = 22) :=
by sorry

end parabola_intersection_l2101_210185


namespace millet_sunflower_exceed_half_on_tuesday_l2101_210158

/-- Represents the proportion of seeds in the feeder -/
structure SeedMix where
  millet : ℝ
  sunflower : ℝ
  other : ℝ

/-- Calculates the next day's seed mix based on consumption and refilling -/
def nextDayMix (mix : SeedMix) : SeedMix :=
  { millet := 0.2 + 0.75 * mix.millet,
    sunflower := 0.3 + 0.5 * mix.sunflower,
    other := 0.5 }

/-- The initial seed mix on Monday -/
def initialMix : SeedMix :=
  { millet := 0.2, sunflower := 0.3, other := 0.5 }

/-- Theorem: On Tuesday, millet and sunflower seeds combined exceed 50% of total seeds -/
theorem millet_sunflower_exceed_half_on_tuesday :
  let tuesdayMix := nextDayMix initialMix
  tuesdayMix.millet + tuesdayMix.sunflower > 0.5 := by
  sorry


end millet_sunflower_exceed_half_on_tuesday_l2101_210158


namespace ac_length_l2101_210117

/-- A quadrilateral with diagonals intersecting at O --/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (OA : ℝ)
  (OC : ℝ)
  (OD : ℝ)
  (OB : ℝ)
  (BD : ℝ)
  (hOA : dist O A = OA)
  (hOC : dist O C = OC)
  (hOD : dist O D = OD)
  (hOB : dist O B = OB)
  (hBD : dist B D = BD)

/-- The theorem stating the length of AC in the given quadrilateral --/
theorem ac_length (q : Quadrilateral) 
  (h1 : q.OA = 6)
  (h2 : q.OC = 9)
  (h3 : q.OD = 6)
  (h4 : q.OB = 7)
  (h5 : q.BD = 10) :
  dist q.A q.C = 11.5 := by sorry

end ac_length_l2101_210117


namespace shane_minimum_score_l2101_210180

def exam_count : ℕ := 5
def max_score : ℕ := 100
def goal_average : ℕ := 86
def first_three_scores : List ℕ := [81, 72, 93]

theorem shane_minimum_score :
  let total_needed : ℕ := goal_average * exam_count
  let scored_so_far : ℕ := first_three_scores.sum
  let remaining_needed : ℕ := total_needed - scored_so_far
  remaining_needed - max_score = 84 :=
by sorry

end shane_minimum_score_l2101_210180


namespace parabola_c_value_l2101_210164

/-- A parabola passing through two given points has a specific c-value -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x, 2 = x^2 + b*x + c → x = 1 ∨ x = 5) →
  c = 7 := by
  sorry

end parabola_c_value_l2101_210164


namespace function_satisfying_condition_l2101_210133

theorem function_satisfying_condition (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, |f x - f y| = 2 * |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = 2 * x + c) ∨ (∀ x : ℝ, f x = -2 * x + c) :=
by sorry

end function_satisfying_condition_l2101_210133


namespace factorial_ratio_l2101_210183

theorem factorial_ratio : Nat.factorial 45 / Nat.factorial 42 = 85140 := by sorry

end factorial_ratio_l2101_210183
