import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_geq_two_l2441_244111

theorem square_sum_geq_two (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_two_l2441_244111


namespace NUMINAMATH_CALUDE_f_properties_l2441_244158

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem f_properties :
  let π := Real.pi
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
    T = π ∧
    (∀ x ∈ Set.Icc (5*π/12) (11*π/12), ∀ y ∈ Set.Icc (5*π/12) (11*π/12), x < y → f y < f x) ∧
    (∀ A b c : ℝ, 
      f (A/2 + π/4) = 1 → 
      2 = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) → 
      2 < b + c ∧ b + c ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2441_244158


namespace NUMINAMATH_CALUDE_intersection_when_a_4_range_of_a_for_sufficient_condition_l2441_244179

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x ∈ Set.Icc 2 3, y = -2^x}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3*x - a^2 - 3*a > 0}

-- Part 1: Intersection when a = 4
theorem intersection_when_a_4 :
  A ∩ B 4 = {x | -8 < x ∧ x < -7} := by sorry

-- Part 2: Range of a for sufficient but not necessary condition
theorem range_of_a_for_sufficient_condition :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) ↔ -4 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_4_range_of_a_for_sufficient_condition_l2441_244179


namespace NUMINAMATH_CALUDE_unique_power_of_two_with_prepended_digit_l2441_244131

theorem unique_power_of_two_with_prepended_digit : 
  ∃! n : ℕ, 
    (∃ k : ℕ, n = 2^k) ∧ 
    (∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ ∃ m : ℕ, 10 * n + d = 2^m) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_power_of_two_with_prepended_digit_l2441_244131


namespace NUMINAMATH_CALUDE_min_abs_z_l2441_244116

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - (3 + 2*I)) = 7) :
  ∃ (z_min : ℂ), ∀ (w : ℂ), Complex.abs (w - 2*I) + Complex.abs (w - (3 + 2*I)) = 7 →
    Complex.abs z_min ≤ Complex.abs w ∧ Complex.abs z_min = 2 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l2441_244116


namespace NUMINAMATH_CALUDE_garden_area_l2441_244162

/-- A rectangular garden with specific length-width relationship and perimeter has an area of 12000 square meters. -/
theorem garden_area (w : ℝ) (h1 : w > 0) : 
  let l := 3 * w + 20
  2 * l + 2 * w = 520 →
  w * l = 12000 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2441_244162


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2441_244101

theorem complex_equation_solution (i z : ℂ) (hi : i^2 = -1) (hz : i * z = 2 * z + 1) :
  z = -2/5 - 1/5 * i := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2441_244101


namespace NUMINAMATH_CALUDE_van_distance_calculation_l2441_244165

theorem van_distance_calculation (original_time : ℝ) (new_speed : ℝ) : 
  original_time = 5 →
  new_speed = 58 →
  ∃ (distance : ℝ), distance = new_speed * (3/2 * original_time) ∧ distance = 435 :=
by
  sorry

end NUMINAMATH_CALUDE_van_distance_calculation_l2441_244165


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2441_244181

theorem inequality_equivalence (x : ℝ) : 
  3/16 + |x - 17/64| < 7/32 ↔ 15/64 < x ∧ x < 19/64 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2441_244181


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2441_244120

theorem complex_fraction_evaluation : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2441_244120


namespace NUMINAMATH_CALUDE_mindy_tax_rate_l2441_244187

theorem mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_income_ratio : ℝ) 
  (combined_rate : ℝ)
  (h1 : mork_rate = 0.45)
  (h2 : mindy_income_ratio = 4)
  (h3 : combined_rate = 0.25) :
  ∃ mindy_rate : ℝ,
    mindy_rate * mindy_income_ratio * mork_rate + mork_rate = 
    combined_rate * (mindy_income_ratio + 1) ∧ 
    mindy_rate = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_mindy_tax_rate_l2441_244187


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l2441_244146

theorem gcd_of_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 528 * k) :
  Nat.gcd (3 * b^3 + b^2 + 4 * b + 66).natAbs b.natAbs = 66 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l2441_244146


namespace NUMINAMATH_CALUDE_investment_interest_l2441_244152

/-- Calculates the interest earned on an investment with annual compounding --/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- The interest earned on a $500 investment at 2% annual interest for 3 years is $31 --/
theorem investment_interest : 
  ∃ ε > 0, |interest_earned 500 0.02 3 - 31| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_interest_l2441_244152


namespace NUMINAMATH_CALUDE_yolka_probability_l2441_244100

/-- Represents the time in minutes after 15:00 -/
def Time := Fin 60

/-- The waiting time for Vasya in minutes -/
def vasyaWaitTime : ℕ := 15

/-- The waiting time for Boris in minutes -/
def borisWaitTime : ℕ := 10

/-- The probability that Anya arrives last -/
def probAnyaLast : ℚ := 1/3

/-- The area in the time square where Boris and Vasya meet -/
def meetingArea : ℕ := 3500

/-- The total area of the time square -/
def totalArea : ℕ := 3600

/-- The probability that all three go to Yolka together -/
def probAllTogether : ℚ := probAnyaLast * (meetingArea / totalArea)

theorem yolka_probability :
  probAllTogether = 1/3 * (3500/3600) :=
sorry

end NUMINAMATH_CALUDE_yolka_probability_l2441_244100


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2441_244151

theorem arithmetic_mean_problem (a b c : ℝ) :
  (a + b + c + 105) / 4 = 93 →
  (a + b + c) / 3 = 89 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2441_244151


namespace NUMINAMATH_CALUDE_a_work_days_l2441_244194

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 16

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 6

/-- The theorem stating that A can finish the work alone in 4 days -/
theorem a_work_days : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    together_days * (1/x + 1/b_days) + b_alone_days * (1/b_days) = 1 ∧ 
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_work_days_l2441_244194


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2441_244193

theorem two_numbers_problem (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 375) :
  |a - b| = 10 := by sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2441_244193


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l2441_244136

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines 
  (m : ℝ) -- Parameter m in the second line equation
  (h1 : x + 2 * y - 1 = 0) -- First line equation
  (h2 : 2 * x + m * y + 4 = 0) -- Second line equation
  (h_parallel : m = 4) -- Condition for lines to be parallel
  : 
  -- The distance between the lines
  (|(-1) - 2| / Real.sqrt (1 + 4)) = 3 / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l2441_244136


namespace NUMINAMATH_CALUDE_same_solution_for_both_systems_l2441_244114

theorem same_solution_for_both_systems :
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 2*x + 3*y = 7 ∧ 3*x - 2*y = 4 ∧ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_same_solution_for_both_systems_l2441_244114


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l2441_244184

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ cos θ = a -/
structure PolarLine where
  a : ℝ

/-- Calculates the distance from a point in polar coordinates to a polar line -/
def distanceFromPointToLine (p : PolarPoint) (l : PolarLine) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  let p : PolarPoint := ⟨1, Real.pi / 2⟩
  let l : PolarLine := ⟨2⟩
  distanceFromPointToLine p l = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l2441_244184


namespace NUMINAMATH_CALUDE_good_time_more_prevalent_l2441_244167

-- Define the clock hands
structure ClockHand where
  angle : ℝ
  speed : ℝ  -- angular speed in radians per hour

-- Define the clock
structure Clock where
  hour : ClockHand
  minute : ClockHand
  second : ClockHand

-- Define good time
def isGoodTime (c : Clock) : Prop :=
  ∃ (d : ℝ), (c.hour.angle - d) * (c.minute.angle - d) ≥ 0 ∧
             (c.hour.angle - d) * (c.second.angle - d) ≥ 0 ∧
             (c.minute.angle - d) * (c.second.angle - d) ≥ 0

-- Define the duration of good time in a day
def goodTimeDuration : ℝ :=
  sorry

-- Define the duration of bad time in a day
def badTimeDuration : ℝ :=
  sorry

-- The theorem to prove
theorem good_time_more_prevalent : goodTimeDuration > badTimeDuration := by
  sorry

end NUMINAMATH_CALUDE_good_time_more_prevalent_l2441_244167


namespace NUMINAMATH_CALUDE_ladder_problem_l2441_244197

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l2441_244197


namespace NUMINAMATH_CALUDE_pizza_bill_division_l2441_244147

theorem pizza_bill_division (total_price : ℝ) (num_people : ℕ) (individual_payment : ℝ) :
  total_price = 40 →
  num_people = 5 →
  individual_payment = total_price / num_people →
  individual_payment = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_bill_division_l2441_244147


namespace NUMINAMATH_CALUDE_riddle_count_l2441_244178

theorem riddle_count (josh ivory taso : ℕ) : 
  josh = 8 → 
  ivory = josh + 4 → 
  taso = 2 * ivory → 
  taso = 24 := by sorry

end NUMINAMATH_CALUDE_riddle_count_l2441_244178


namespace NUMINAMATH_CALUDE_bottle_weight_problem_l2441_244138

/-- Given the weight of 3 glass bottles and the weight difference between glass and plastic bottles,
    calculate the total weight of 4 glass bottles and 5 plastic bottles. -/
theorem bottle_weight_problem (weight_3_glass : ℕ) (weight_diff : ℕ) : 
  weight_3_glass = 600 → weight_diff = 150 → 
  (4 * (weight_3_glass / 3 + weight_diff) + 5 * (weight_3_glass / 3 - weight_diff / 3)) = 1050 := by
  sorry

end NUMINAMATH_CALUDE_bottle_weight_problem_l2441_244138


namespace NUMINAMATH_CALUDE_minimum_freight_charges_l2441_244169

theorem minimum_freight_charges 
  (total_trucks : ℕ) 
  (large_capacity small_capacity : ℕ) 
  (total_sugar : ℕ) 
  (large_freight_A small_freight_A : ℕ) 
  (large_freight_B small_freight_B : ℕ) 
  (trucks_to_A : ℕ) 
  (min_sugar_A : ℕ) :
  total_trucks = 20 →
  large_capacity = 15 →
  small_capacity = 10 →
  total_sugar = 240 →
  large_freight_A = 630 →
  small_freight_A = 420 →
  large_freight_B = 750 →
  small_freight_B = 550 →
  trucks_to_A = 10 →
  min_sugar_A = 115 →
  ∃ (large_A small_A large_B small_B : ℕ),
    large_A + small_A = trucks_to_A ∧
    large_B + small_B = total_trucks - trucks_to_A ∧
    large_A + large_B = 8 ∧
    small_A + small_B = 12 ∧
    large_capacity * large_A + small_capacity * small_A ≥ min_sugar_A ∧
    large_capacity * (large_A + large_B) + small_capacity * (small_A + small_B) = total_sugar ∧
    large_freight_A * large_A + small_freight_A * small_A + 
    large_freight_B * large_B + small_freight_B * small_B = 11330 ∧
    ∀ (x y z w : ℕ),
      x + y = trucks_to_A →
      z + w = total_trucks - trucks_to_A →
      x + z = 8 →
      y + w = 12 →
      large_capacity * x + small_capacity * y ≥ min_sugar_A →
      large_capacity * (x + z) + small_capacity * (y + w) = total_sugar →
      large_freight_A * x + small_freight_A * y + 
      large_freight_B * z + small_freight_B * w ≥ 11330 :=
by
  sorry


end NUMINAMATH_CALUDE_minimum_freight_charges_l2441_244169


namespace NUMINAMATH_CALUDE_raspberry_pie_degrees_l2441_244148

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The total number of students in the class -/
def total_students : ℕ := 48

/-- The number of students preferring chocolate pie -/
def chocolate_pref : ℕ := 18

/-- The number of students preferring apple pie -/
def apple_pref : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_pref : ℕ := 8

/-- Theorem stating that the number of degrees for raspberry pie in the pie chart is 45 -/
theorem raspberry_pie_degrees : 
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let raspberry_pref := remaining / 2
  (raspberry_pref : ℚ) / total_students * full_circle = 45 := by sorry

end NUMINAMATH_CALUDE_raspberry_pie_degrees_l2441_244148


namespace NUMINAMATH_CALUDE_ceiling_sqrt_180_l2441_244129

theorem ceiling_sqrt_180 : ⌈Real.sqrt 180⌉ = 14 := by
  have h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14 := by sorry
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_180_l2441_244129


namespace NUMINAMATH_CALUDE_stickers_on_bottles_elizabeth_stickers_l2441_244112

/-- Calculate the total number of stickers used on water bottles -/
theorem stickers_on_bottles 
  (initial_bottles : ℕ) 
  (lost_bottles : ℕ) 
  (stolen_bottles : ℕ) 
  (stickers_per_bottle : ℕ) 
  (h1 : initial_bottles ≥ lost_bottles + stolen_bottles) : 
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle = 
  (initial_bottles - (lost_bottles + stolen_bottles)) * stickers_per_bottle :=
by sorry

/-- Specific case for Elizabeth's water bottles -/
theorem elizabeth_stickers : 
  (10 : ℕ) - 2 - 1 = 7 ∧ 7 * 3 = 21 :=
by sorry

end NUMINAMATH_CALUDE_stickers_on_bottles_elizabeth_stickers_l2441_244112


namespace NUMINAMATH_CALUDE_outfit_combinations_l2441_244191

/-- The number of different outfits that can be created given a set of clothing items -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem: Given 8 shirts, 4 pairs of pants, 5 ties, and 2 types of belts,
    where an outfit requires a shirt and pants, and can optionally include a tie and/or a belt,
    the total number of different outfits that can be created is 576. -/
theorem outfit_combinations : number_of_outfits 8 4 5 2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2441_244191


namespace NUMINAMATH_CALUDE_curve_tangent_acute_angle_l2441_244106

/-- The curve C: y = x^3 - 2ax^2 + 2ax -/
def C (a : ℤ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℤ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

/-- The condition that the tangent line has an acute angle with the x-axis -/
def acute_angle_condition (a : ℤ) : Prop :=
  ∀ x : ℝ, C_derivative a x > 0

theorem curve_tangent_acute_angle (a : ℤ) (h : acute_angle_condition a) : a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_tangent_acute_angle_l2441_244106


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2441_244189

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) 
  (h1 : n = 20) 
  (h2 : total_games = 380) : 
  total_games = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2441_244189


namespace NUMINAMATH_CALUDE_min_sum_cube_relation_l2441_244161

theorem min_sum_cube_relation (m n : ℕ+) (h : 50 * m = n^3) : 
  (∀ m' n' : ℕ+, 50 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 30 := by
sorry

end NUMINAMATH_CALUDE_min_sum_cube_relation_l2441_244161


namespace NUMINAMATH_CALUDE_equation_roots_and_m_values_l2441_244121

-- Define the equation
def equation (x m : ℝ) : Prop := (x + m)^2 - 4 = 0

-- Theorem statement
theorem equation_roots_and_m_values :
  -- For all real m
  ∀ m : ℝ,
  -- There exist two distinct real roots
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ m ∧ equation x₂ m ∧
  -- If the roots p and q satisfy pq = p + q
  (∀ p q : ℝ, equation p m → equation q m → p * q = p + q →
  -- Then m equals one of these two values
  (m = Real.sqrt 5 - 1 ∨ m = -Real.sqrt 5 - 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_and_m_values_l2441_244121


namespace NUMINAMATH_CALUDE_fraction_simplification_l2441_244172

theorem fraction_simplification : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2441_244172


namespace NUMINAMATH_CALUDE_salary_percentage_difference_l2441_244123

/-- Given two employees M and N with a total salary and N's individual salary,
    calculate the percentage difference between M's and N's salaries. -/
theorem salary_percentage_difference
  (total_salary : ℚ)
  (n_salary : ℚ)
  (h1 : total_salary = 616)
  (h2 : n_salary = 280) :
  (total_salary - n_salary) / n_salary * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_difference_l2441_244123


namespace NUMINAMATH_CALUDE_third_nonagon_side_length_l2441_244163

/-- Represents a regular nonagon with a given side length -/
structure RegularNonagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a regular nonagon given its side length -/
def nonagonArea (n : RegularNonagon) : ℝ := n.sideLength^2

/-- Theorem: Given three concentric regular nonagons with parallel sides,
    where two have side lengths of 8 and 56, and the third divides the area
    between them in a 1:7 ratio (measured from the smaller nonagon),
    the side length of the third nonagon is 8√7. -/
theorem third_nonagon_side_length
  (n1 n2 n3 : RegularNonagon)
  (h1 : n1.sideLength = 8)
  (h2 : n2.sideLength = 56)
  (h3 : (nonagonArea n3 - nonagonArea n1) / (nonagonArea n2 - nonagonArea n3) = 1 / 7) :
  n3.sideLength = 8 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_third_nonagon_side_length_l2441_244163


namespace NUMINAMATH_CALUDE_product_sequence_mod_six_l2441_244154

theorem product_sequence_mod_six : ∃ (seq : List Nat), 
  (seq.length = 10) ∧ 
  (∀ i, i ∈ seq → ∃ k, i = 10 * k + 3) ∧
  (seq.prod % 6 = 3) := by
sorry

end NUMINAMATH_CALUDE_product_sequence_mod_six_l2441_244154


namespace NUMINAMATH_CALUDE_digits_of_2_to_70_l2441_244159

theorem digits_of_2_to_70 : ∃ n : ℕ, n = 22 ∧ (2^70 : ℕ) < 10^n ∧ 10^(n-1) ≤ 2^70 :=
  sorry

end NUMINAMATH_CALUDE_digits_of_2_to_70_l2441_244159


namespace NUMINAMATH_CALUDE_invalid_statement_d_l2441_244119

/-- Represents a mathematical statement --/
structure MathStatement where
  content : String

/-- Represents a mathematical proof --/
structure Proof where
  premises : List MathStatement
  conclusion : MathStatement

/-- Checks if a statement is true --/
def isTrue (s : MathStatement) : Bool :=
  sorry

/-- Checks if a proof is valid --/
def isValidProof (p : Proof) : Bool :=
  sorry

/-- Checks if a statement is used in deriving the conclusion of a proof --/
def isUsedInConclusion (s : MathStatement) (p : Proof) : Bool :=
  sorry

theorem invalid_statement_d :
  ¬∀ (p : Proof) (s : MathStatement),
    (s ∈ p.premises ∧ ¬isTrue s ∧ ¬isUsedInConclusion s p) →
    (isValidProof p → isTrue p.conclusion) :=
  sorry

end NUMINAMATH_CALUDE_invalid_statement_d_l2441_244119


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2441_244145

-- Define the repeating decimal 7.036̅
def repeating_decimal : ℚ := 7 + 36 / 999

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 781 / 111 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2441_244145


namespace NUMINAMATH_CALUDE_arctg_difference_bound_l2441_244188

theorem arctg_difference_bound (a b : ℝ) : 
  |Real.arctan a - Real.arctan b| ≤ |b - a| := by sorry

end NUMINAMATH_CALUDE_arctg_difference_bound_l2441_244188


namespace NUMINAMATH_CALUDE_unique_solution_is_three_l2441_244102

/-- An arithmetic progression with the first three terms as given functions of x -/
def arithmetic_progression (x : ℝ) : ℕ → ℝ
  | 0 => 2 * x - 1
  | 1 => 3 * x + 4
  | 2 => 5 * x + 6
  | _ => 0  -- We only care about the first three terms

/-- The common difference of the arithmetic progression -/
def common_difference (x : ℝ) : ℝ := arithmetic_progression x 1 - arithmetic_progression x 0

/-- Theorem stating that x = 3 is the unique solution for the given arithmetic progression -/
theorem unique_solution_is_three :
  ∃! x : ℝ, 
    (arithmetic_progression x 1 - arithmetic_progression x 0 = common_difference x) ∧
    (arithmetic_progression x 2 - arithmetic_progression x 1 = common_difference x) ∧
    x = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_three_l2441_244102


namespace NUMINAMATH_CALUDE_nancy_small_gardens_l2441_244144

/-- The number of small gardens Nancy had given her seed distribution --/
def small_gardens_count (total_seeds capsicum_seeds cucumber_seeds tomato_seeds big_garden_tomato : ℕ) : ℕ :=
  let remaining_tomato := tomato_seeds - big_garden_tomato
  remaining_tomato / 2

theorem nancy_small_gardens 
  (h1 : total_seeds = 85)
  (h2 : tomato_seeds = 42)
  (h3 : capsicum_seeds = 26)
  (h4 : cucumber_seeds = 17)
  (h5 : big_garden_tomato = 24)
  (h6 : total_seeds = tomato_seeds + capsicum_seeds + cucumber_seeds) :
  small_gardens_count total_seeds capsicum_seeds cucumber_seeds tomato_seeds big_garden_tomato = 9 := by
  sorry

#eval small_gardens_count 85 26 17 42 24

end NUMINAMATH_CALUDE_nancy_small_gardens_l2441_244144


namespace NUMINAMATH_CALUDE_jeff_total_distance_l2441_244198

/-- Represents a segment of Jeff's journey --/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled in a segment --/
def distanceInSegment (s : Segment) : ℝ := s.speed * s.duration

/-- Jeff's journey segments --/
def jeffJourney : List Segment := [
  ⟨80, 3⟩, ⟨50, 2⟩, ⟨70, 1⟩, ⟨60, 1.5⟩, ⟨45, 1.5⟩,
  ⟨60, 1.5⟩, ⟨35, 2⟩, ⟨40, 2⟩, ⟨30, 2.5⟩, ⟨25, 1⟩
]

/-- Theorem: The total distance Jeff traveled is 907.5 miles --/
theorem jeff_total_distance :
  (jeffJourney.map distanceInSegment).sum = 907.5 := by
  sorry


end NUMINAMATH_CALUDE_jeff_total_distance_l2441_244198


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2441_244182

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 12 ∧ 
  a = 8 ∧ 
  b = 22 ∧ 
  c = d → 
  c * d = 81 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2441_244182


namespace NUMINAMATH_CALUDE_eggs_per_unit_l2441_244199

/-- Given that Joan bought 6 units of eggs and 72 eggs in total, 
    prove that the number of eggs in one unit is 12. -/
theorem eggs_per_unit (units : ℕ) (total_eggs : ℕ) 
  (h1 : units = 6) (h2 : total_eggs = 72) : 
  total_eggs / units = 12 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_unit_l2441_244199


namespace NUMINAMATH_CALUDE_star_not_commutative_l2441_244156

def star (x y : ℝ) : ℝ := |x - 2*y + 3|

theorem star_not_commutative : ∃ x y : ℝ, star x y ≠ star y x := by sorry

end NUMINAMATH_CALUDE_star_not_commutative_l2441_244156


namespace NUMINAMATH_CALUDE_combined_stickers_l2441_244128

theorem combined_stickers (june_initial : ℕ) (bonnie_initial : ℕ) (birthday_gift : ℕ) :
  june_initial = 76 →
  bonnie_initial = 63 →
  birthday_gift = 25 →
  june_initial + bonnie_initial + 2 * birthday_gift = 189 :=
by sorry

end NUMINAMATH_CALUDE_combined_stickers_l2441_244128


namespace NUMINAMATH_CALUDE_percent_relation_l2441_244192

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y) 
  (h2 : z = 2 * x) : 
  y = 0.75 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l2441_244192


namespace NUMINAMATH_CALUDE_initial_fee_is_2_25_l2441_244166

/-- Represents the taxi service charges -/
structure TaxiCharge where
  initialFee : ℝ
  additionalChargePerSegment : ℝ
  segmentLength : ℝ
  tripLength : ℝ
  totalCharge : ℝ

/-- Proves that the initial fee for the given taxi trip is $2.25 -/
theorem initial_fee_is_2_25 (tc : TaxiCharge) 
  (h1 : tc.additionalChargePerSegment = 0.4)
  (h2 : tc.segmentLength = 2/5)
  (h3 : tc.tripLength = 3.6)
  (h4 : tc.totalCharge = 5.85) :
  tc.initialFee = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_initial_fee_is_2_25_l2441_244166


namespace NUMINAMATH_CALUDE_specific_pyramid_has_180_balls_l2441_244168

/-- Represents a pyramid display of balls -/
structure PyramidDisplay where
  bottomLayer : ℕ
  topLayer : ℕ
  difference : ℤ

/-- Calculates the total number of balls in a pyramid display -/
def totalBalls (p : PyramidDisplay) : ℕ :=
  sorry

/-- Theorem stating that the specific pyramid display has 180 balls -/
theorem specific_pyramid_has_180_balls :
  let p : PyramidDisplay := {
    bottomLayer := 35,
    topLayer := 1,
    difference := -4
  }
  totalBalls p = 180 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_has_180_balls_l2441_244168


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2441_244171

theorem coefficient_x_squared_in_expansion : 
  let n : ℕ := 7
  let expansion := (1 - X : Polynomial ℚ) ^ n
  (expansion.coeff 2 : ℚ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2441_244171


namespace NUMINAMATH_CALUDE_trig_expressions_given_tan_alpha_l2441_244150

theorem trig_expressions_given_tan_alpha (α : Real) (h : Real.tan α = -2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = 5 ∧
  1 / (Real.sin α * Real.cos α) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expressions_given_tan_alpha_l2441_244150


namespace NUMINAMATH_CALUDE_original_mean_calculation_l2441_244125

theorem original_mean_calculation (n : ℕ) (decrease : ℝ) (new_mean : ℝ) (h1 : n = 50) (h2 : decrease = 15) (h3 : new_mean = 185) : 
  (n : ℝ) * new_mean + n * decrease = n * 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l2441_244125


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l2441_244124

def total_respondents : ℕ := 180
def brand_x_preference : ℕ := 150

theorem brand_preference_ratio :
  let brand_y_preference := total_respondents - brand_x_preference
  (brand_x_preference : ℚ) / brand_y_preference = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l2441_244124


namespace NUMINAMATH_CALUDE_calculate_remaining_student_age_l2441_244155

/-- Given a class of students with known average ages for subgroups,
    calculate the age of the remaining student. -/
theorem calculate_remaining_student_age
  (total_students : ℕ)
  (total_average : ℕ)
  (group1_students : ℕ)
  (group1_average : ℕ)
  (group2_students : ℕ)
  (group2_average : ℕ)
  (h1 : total_students = 25)
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28)
  (h7 : group1_students + group2_students + 1 = total_students) :
  total_students * total_average =
    group1_students * group1_average +
    group2_students * group2_average + 13 :=
by sorry

end NUMINAMATH_CALUDE_calculate_remaining_student_age_l2441_244155


namespace NUMINAMATH_CALUDE_expected_pine_saplings_l2441_244160

/-- Represents the number of pine saplings in a stratified sample. -/
def pine_saplings_in_sample (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℚ :=
  (pine_saplings : ℚ) / (total_saplings : ℚ) * (sample_size : ℚ)

/-- Theorem stating the expected number of pine saplings in the sample. -/
theorem expected_pine_saplings :
  pine_saplings_in_sample 30000 4000 150 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_pine_saplings_l2441_244160


namespace NUMINAMATH_CALUDE_cook_carrots_problem_l2441_244140

theorem cook_carrots_problem (initial_carrots : ℕ) 
  (fraction_used_before_lunch : ℚ) (carrots_not_used : ℕ) : 
  initial_carrots = 300 →
  fraction_used_before_lunch = 2/5 →
  carrots_not_used = 72 →
  (initial_carrots - fraction_used_before_lunch * initial_carrots - carrots_not_used) / 
  (initial_carrots - fraction_used_before_lunch * initial_carrots) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cook_carrots_problem_l2441_244140


namespace NUMINAMATH_CALUDE_M_equals_N_l2441_244142

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {a | ∃ x ∈ M, a * x = 1}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l2441_244142


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_two_l2441_244180

/-- The cost of a single topping in dollars -/
def topping_cost : ℚ := 1/2

/-- The number of toppings on the sundae -/
def num_toppings : ℕ := 10

/-- The total cost of the sundae in dollars -/
def sundae_cost : ℚ := 7

/-- The cost of the ice cream in dollars -/
def ice_cream_cost : ℚ := sundae_cost - num_toppings * topping_cost

theorem ice_cream_cost_is_two :
  ice_cream_cost = 2 :=
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_is_two_l2441_244180


namespace NUMINAMATH_CALUDE_eighteenth_roots_of_unity_ninth_power_real_l2441_244175

theorem eighteenth_roots_of_unity_ninth_power_real : 
  ∀ z : ℂ, z^18 = 1 → ∃ r : ℝ, z^9 = r :=
by sorry

end NUMINAMATH_CALUDE_eighteenth_roots_of_unity_ninth_power_real_l2441_244175


namespace NUMINAMATH_CALUDE_final_shell_count_l2441_244127

def shell_collection (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (extra_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + extra_shells

theorem final_shell_count :
  shell_collection 20 5 3 6 = 41 := by
  sorry

end NUMINAMATH_CALUDE_final_shell_count_l2441_244127


namespace NUMINAMATH_CALUDE_hair_growth_calculation_l2441_244186

theorem hair_growth_calculation (current_length desired_after_donation donation_length : ℕ) 
  (h1 : current_length = 14)
  (h2 : desired_after_donation = 12)
  (h3 : donation_length = 23) :
  donation_length + desired_after_donation - current_length = 21 := by
  sorry

end NUMINAMATH_CALUDE_hair_growth_calculation_l2441_244186


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2441_244170

/-- Set A defined by the quadratic inequality -/
def A (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- Set B defined by the given inequality -/
def B : Set ℝ := {x | (x-3)*(2-x) ≥ 0}

/-- Theorem stating the condition for A to be a necessary but not sufficient condition for B -/
theorem necessary_not_sufficient_condition (a : ℝ) :
  (a > 0) → (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B) ↔ a > 1 ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2441_244170


namespace NUMINAMATH_CALUDE_equality_relations_l2441_244143

theorem equality_relations (a b c d : ℝ) (h : a * b = c * d) : 
  (a / c = d / b) ∧ 
  (a / d = c / b) ∧ 
  ((a + c) / c = (d + b) / b) ∧ 
  ¬ ∀ (a b c d : ℝ), a * b = c * d → (a + 1) / (c + 1) = (d + 1) / (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_equality_relations_l2441_244143


namespace NUMINAMATH_CALUDE_desk_height_proof_l2441_244185

theorem desk_height_proof (block_length block_width desk_height : ℝ) : 
  desk_height + 2 * block_length = 50 ∧ 
  desk_height + 2 * block_width = 40 →
  desk_height = 30 := by
sorry

end NUMINAMATH_CALUDE_desk_height_proof_l2441_244185


namespace NUMINAMATH_CALUDE_average_page_count_l2441_244141

theorem average_page_count (total_students : ℕ) 
  (group1_count group2_count group3_count group4_count : ℕ)
  (group1_pages group2_pages group3_pages group4_pages : ℕ) : 
  total_students = 30 →
  group1_count = 8 →
  group2_count = 10 →
  group3_count = 7 →
  group4_count = 5 →
  group1_pages = 3 →
  group2_pages = 5 →
  group3_pages = 2 →
  group4_pages = 4 →
  (group1_count * group1_pages + 
   group2_count * group2_pages + 
   group3_count * group3_pages + 
   group4_count * group4_pages : ℚ) / total_students = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_average_page_count_l2441_244141


namespace NUMINAMATH_CALUDE_cone_height_l2441_244139

/-- A cone with volume 8000π cubic inches and a vertical cross section with a 90-degree vertex angle has a height of 20 × ∛6 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : ℝ) :
  V = 8000 * Real.pi ∧ θ = Real.pi / 2 →
  h = 20 * (6 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l2441_244139


namespace NUMINAMATH_CALUDE_no_rain_probability_l2441_244115

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2441_244115


namespace NUMINAMATH_CALUDE_train_speed_l2441_244130

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 10) :
  length / time = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2441_244130


namespace NUMINAMATH_CALUDE_expression_simplification_l2441_244149

theorem expression_simplification (a b : ℝ) (ha : a = -1) (hb : b = 1) :
  (4/5) * a * b - (2 * a * b^2 - 4 * (-(1/5) * a * b + 3 * a^2 * b)) + 2 * a * b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2441_244149


namespace NUMINAMATH_CALUDE_product_of_exponents_l2441_244117

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^3 = 320 → 
  3^r + 27 = 54 → 
  2^5 + 7^s = 375 → 
  p * r * s = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l2441_244117


namespace NUMINAMATH_CALUDE_circle_radius_order_l2441_244137

theorem circle_radius_order (r_A : ℝ) (c_B : ℝ) (a_C : ℝ) :
  r_A = 3 * Real.pi →
  c_B = 10 * Real.pi →
  a_C = 16 * Real.pi →
  ∃ (r_B r_C : ℝ),
    c_B = 2 * Real.pi * r_B ∧
    a_C = Real.pi * r_C^2 ∧
    r_C < r_B ∧ r_B < r_A :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_order_l2441_244137


namespace NUMINAMATH_CALUDE_gallop_waddle_length_difference_l2441_244177

/-- The number of waddles Percy takes between consecutive lampposts -/
def percy_waddles : ℕ := 36

/-- The number of gallops Zelda takes between consecutive lampposts -/
def zelda_gallops : ℕ := 15

/-- The number of the last lamppost -/
def last_lamppost : ℕ := 31

/-- The distance in feet from the first to the last lamppost -/
def total_distance : ℕ := 3720

/-- The difference between Zelda's gallop length and Percy's waddle length -/
def gallop_waddle_difference : ℚ := 31 / 15

theorem gallop_waddle_length_difference :
  let percy_waddle_length : ℚ := total_distance / (percy_waddles * (last_lamppost - 1))
  let zelda_gallop_length : ℚ := total_distance / (zelda_gallops * (last_lamppost - 1))
  zelda_gallop_length - percy_waddle_length = gallop_waddle_difference := by
  sorry

end NUMINAMATH_CALUDE_gallop_waddle_length_difference_l2441_244177


namespace NUMINAMATH_CALUDE_kolya_de_segment_time_l2441_244103

/-- Represents a point in the park --/
structure Point :=
  (name : String)

/-- Represents a route in the park --/
structure Route :=
  (points : List Point)

/-- Represents a biker in the park --/
structure Biker :=
  (name : String)
  (route : Route)
  (speed : ℝ)

/-- The time taken by a biker to complete their route --/
def routeTime (b : Biker) : ℝ := 12

theorem kolya_de_segment_time :
  ∀ (petya kolya : Biker) (a b c d e f : Point),
    petya.route = Route.mk [a, b, c] →
    kolya.route = Route.mk [a, d, e, f, c] →
    routeTime petya = routeTime kolya →
    kolya.speed = 1.2 * petya.speed →
    ∃ (t : ℝ), t = 1 ∧ (∀ (de_segment : Route), de_segment = Route.mk [d, e] → routeTime (Biker.mk kolya.name de_segment kolya.speed) = t) :=
by
  sorry


end NUMINAMATH_CALUDE_kolya_de_segment_time_l2441_244103


namespace NUMINAMATH_CALUDE_max_receivable_amount_l2441_244108

/-- Represents the denominations of chips available at the casino. -/
inductive ChipDenomination
  | TwentyFive
  | SeventyFive
  | TwoFifty

/-- Represents the number of chips lost for each denomination. -/
structure LostChips where
  twentyFive : ℕ
  seventyFive : ℕ
  twoFifty : ℕ

/-- Calculates the total value of chips based on the number of chips for each denomination. -/
def chipValue (chips : LostChips) : ℕ :=
  25 * chips.twentyFive + 75 * chips.seventyFive + 250 * chips.twoFifty

/-- Represents the conditions of the gambling problem. -/
structure GamblingProblem where
  initialValue : ℕ
  lostChips : LostChips
  haveTwentyFiveLeft : Prop
  haveSeventyFiveLeft : Prop
  haveTwoFiftyLeft : Prop
  totalLostChips : ℕ
  lostTwentyFiveTwiceSeventyFive : Prop
  lostSeventyFiveHalfTwoFifty : Prop

/-- Theorem stating the maximum amount the gambler could have received back. -/
theorem max_receivable_amount (problem : GamblingProblem)
  (h1 : problem.initialValue = 15000)
  (h2 : problem.totalLostChips = 40)
  (h3 : problem.lostTwentyFiveTwiceSeventyFive)
  (h4 : problem.lostSeventyFiveHalfTwoFifty)
  (h5 : problem.haveTwentyFiveLeft)
  (h6 : problem.haveSeventyFiveLeft)
  (h7 : problem.haveTwoFiftyLeft) :
  problem.initialValue - chipValue problem.lostChips = 10000 := by
  sorry

#check max_receivable_amount

end NUMINAMATH_CALUDE_max_receivable_amount_l2441_244108


namespace NUMINAMATH_CALUDE_ryan_leaf_collection_l2441_244126

theorem ryan_leaf_collection (lost broken left initial : ℕ) : 
  lost = 24 → broken = 43 → left = 22 → initial = lost + broken + left :=
by sorry

end NUMINAMATH_CALUDE_ryan_leaf_collection_l2441_244126


namespace NUMINAMATH_CALUDE_no_valid_n_l2441_244122

theorem no_valid_n : ¬∃ (n : ℕ), 
  n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l2441_244122


namespace NUMINAMATH_CALUDE_doughnuts_per_person_l2441_244113

def samuel_doughnuts : ℕ := 2 * 12
def cathy_doughnuts : ℕ := 3 * 12
def total_friends : ℕ := 8
def total_people : ℕ := total_friends + 2

theorem doughnuts_per_person :
  (samuel_doughnuts + cathy_doughnuts) / total_people = 6 :=
sorry

end NUMINAMATH_CALUDE_doughnuts_per_person_l2441_244113


namespace NUMINAMATH_CALUDE_arrangement_exists_linear_not_circular_l2441_244104

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def validLinearArrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 16 ∧
  arrangement.toFinset = Finset.range 16 ∧
  ∀ i : ℕ, i < 15 → isPerfectSquare (arrangement[i]! + arrangement[i+1]!)

def validCircularArrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 16 ∧
  arrangement.toFinset = Finset.range 16 ∧
  ∀ i : ℕ, i < 16 → isPerfectSquare (arrangement[i]! + arrangement[(i+1) % 16]!)

theorem arrangement_exists_linear_not_circular :
  (∃ arrangement : List ℕ, validLinearArrangement arrangement) ∧
  (¬ ∃ arrangement : List ℕ, validCircularArrangement arrangement) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_exists_linear_not_circular_l2441_244104


namespace NUMINAMATH_CALUDE_number_equation_solution_l2441_244190

theorem number_equation_solution : 
  ∃ x : ℝ, (5020 - (1004 / x) = 4970) ∧ (x = 20.08) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2441_244190


namespace NUMINAMATH_CALUDE_smallest_product_of_digits_1234_l2441_244135

/-- Given a list of four distinct digits, returns all possible pairs of two-digit numbers that can be formed using each digit exactly once. -/
def generatePairs (digits : List Nat) : List (Nat × Nat) :=
  sorry

/-- Calculates the product of a pair of numbers. -/
def pairProduct (pair : Nat × Nat) : Nat :=
  pair.1 * pair.2

/-- Finds the smallest product among a list of pairs. -/
def smallestProduct (pairs : List (Nat × Nat)) : Nat :=
  sorry

theorem smallest_product_of_digits_1234 :
  let digits := [1, 2, 3, 4]
  let pairs := generatePairs digits
  smallestProduct pairs = 312 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_of_digits_1234_l2441_244135


namespace NUMINAMATH_CALUDE_gina_hourly_wage_l2441_244133

-- Define Gina's painting rates
def rose_rate : ℝ := 6
def lily_rate : ℝ := 7
def sunflower_rate : ℝ := 5
def orchid_rate : ℝ := 8

-- Define the orders
def order1_roses : ℝ := 6
def order1_lilies : ℝ := 14
def order1_sunflowers : ℝ := 4
def order1_payment : ℝ := 120

def order2_orchids : ℝ := 10
def order2_roses : ℝ := 2
def order2_payment : ℝ := 80

def order3_sunflowers : ℝ := 8
def order3_orchids : ℝ := 4
def order3_payment : ℝ := 70

-- Define the theorem
theorem gina_hourly_wage :
  let total_time := (order1_roses / rose_rate + order1_lilies / lily_rate + order1_sunflowers / sunflower_rate) +
                    (order2_orchids / orchid_rate + order2_roses / rose_rate) +
                    (order3_sunflowers / sunflower_rate + order3_orchids / orchid_rate)
  let total_payment := order1_payment + order2_payment + order3_payment
  let hourly_wage := total_payment / total_time
  ∃ ε > 0, |hourly_wage - 36.08| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_gina_hourly_wage_l2441_244133


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2441_244109

/-- Given a hyperbola and a line intersecting at two points, prove a relation between a and b -/
theorem hyperbola_line_intersection (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (P Q : ℝ × ℝ),
    -- P and Q lie on the hyperbola
    (P.1^2 / a - P.2^2 / b = 1) ∧
    (Q.1^2 / a - Q.2^2 / b = 1) ∧
    -- P and Q lie on the line
    (P.1 + P.2 = 1) ∧
    (Q.1 + Q.2 = 1) ∧
    -- OP is perpendicular to OQ
    (P.1 * Q.1 + P.2 * Q.2 = 0)) →
  1 / a - 1 / b = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2441_244109


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2441_244173

theorem cubic_root_sum (p q : ℝ) : 
  (∃ x : ℂ, x^3 + p*x + q = 0 ∧ x = 2 + Complex.I) → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2441_244173


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2441_244153

-- Define the condition that x^2 - 2ax + a > 0 holds for all real x
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*a*x + a > 0

-- Define the solution set
def solution_set (t : ℝ) : Prop :=
  t < -3 ∨ t > 1

-- State the theorem
theorem inequality_solution_set :
  ∀ a : ℝ, always_positive a →
    (∀ t : ℝ, a^(t^2) + 2*t - 3 < 1 ↔ solution_set t) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2441_244153


namespace NUMINAMATH_CALUDE_three_semi_fixed_points_l2441_244164

/-- A function f has a semi-fixed point at x₀ if f(x₀) = -x₀ -/
def has_semi_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = -x₀

/-- The function f(x) = ax^3 - 3x^2 - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^3 - 3 * x^2 - x + 1

/-- The theorem stating the condition for f to have exactly three semi-fixed points -/
theorem three_semi_fixed_points (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    has_semi_fixed_point (f a) x₁ ∧
    has_semi_fixed_point (f a) x₂ ∧
    has_semi_fixed_point (f a) x₃ ∧
    (∀ x : ℝ, has_semi_fixed_point (f a) x → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  (a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2) :=
sorry

end NUMINAMATH_CALUDE_three_semi_fixed_points_l2441_244164


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cube_l2441_244157

structure Cube where
  edge : ℝ
  volume : ℝ
  volume_eq : volume = edge^3

structure Pyramid where
  base_area : ℝ
  height : ℝ
  volume : ℝ
  volume_eq : volume = (1/3) * base_area * height

theorem pyramid_volume_in_cube (c : Cube) (p : Pyramid) :
  c.volume = 8 →
  p.base_area = 2 →
  p.height = c.edge →
  p.volume = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cube_l2441_244157


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2441_244107

/-- Given a hyperbola with equation x²/4 - y² = 1, prove its transverse axis length and asymptote equations -/
theorem hyperbola_properties :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/4 - y^2 = 1}
  ∃ (transverse_axis_length : ℝ) (asymptote_slope : ℝ),
    transverse_axis_length = 4 ∧
    asymptote_slope = 1/2 ∧
    (∀ (x y : ℝ), (x, y) ∈ hyperbola → 
      (y = asymptote_slope * x ∨ y = -asymptote_slope * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2441_244107


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2441_244105

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + d * (n - 1)

theorem arithmetic_sequence_formula :
  let a := arithmetic_sequence 2 3
  ∀ n : ℕ, a n = 3 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2441_244105


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2441_244183

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- The set of roots of f -/
def roots_of_f : Set ℂ := {z : ℂ | f z = 0}

/-- n^th roots of unity -/
def nth_roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

/-- Statement: 9 is the smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n^th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ∃ z ∈ roots_of_f, z ∉ nth_roots_of_unity m) ∧
  (∀ z ∈ roots_of_f, z ∈ nth_roots_of_unity n) ∧
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2441_244183


namespace NUMINAMATH_CALUDE_newspaper_profit_bounds_l2441_244176

/-- Profit function for newspaper sales --/
def profit (x : ℝ) : ℝ := 0.95 * x - 90

/-- Domain of the profit function --/
def valid_sales (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 200

theorem newspaper_profit_bounds :
  ∀ x : ℝ, valid_sales x →
    profit x ≤ 100 ∧
    profit x ≥ -90 ∧
    (∃ x₁ x₂ : ℝ, valid_sales x₁ ∧ valid_sales x₂ ∧ profit x₁ = 100 ∧ profit x₂ = -90) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_profit_bounds_l2441_244176


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2441_244195

theorem possible_values_of_a (a b c d : ℕ) : 
  a > b ∧ b > c ∧ c > d ∧ 
  a + b + c + d = 2010 ∧ 
  a^2 - b^2 + c^2 - d^2 = 2010 →
  (∃ (s : Finset ℕ), s.card = 501 ∧ ∀ x, x ∈ s ↔ (∃ b' c' d' : ℕ, 
    x > b' ∧ b' > c' ∧ c' > d' ∧ 
    x + b' + c' + d' = 2010 ∧ 
    x^2 - b'^2 + c'^2 - d'^2 = 2010)) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2441_244195


namespace NUMINAMATH_CALUDE_total_height_difference_l2441_244196

/-- Given the height relationships between family members, calculate the total height difference --/
theorem total_height_difference (anne bella cathy daisy ellie : ℝ) : 
  anne = 2 * cathy ∧ 
  bella = 3 * anne ∧ 
  daisy = 1.5 * cathy ∧ 
  ellie = 1.75 * bella ∧ 
  anne = 80 → 
  |bella - cathy| + |bella - daisy| + |bella - ellie| + 
  |cathy - daisy| + |cathy - ellie| + |daisy - ellie| = 1320 := by
sorry

end NUMINAMATH_CALUDE_total_height_difference_l2441_244196


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l2441_244174

/-- The number of ways to arrange animals in cages. -/
def arrange_animals (num_chickens num_dogs num_cats : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial num_chickens * Nat.factorial num_dogs * Nat.factorial num_cats

/-- The theorem stating the number of arrangements for the given problem. -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 3 3 4 = 5184 :=
by sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l2441_244174


namespace NUMINAMATH_CALUDE_minimum_cost_for_227_students_l2441_244134

/-- Represents the cost structure for notebooks -/
structure NotebookPricing where
  single_cost : ℝ
  dozen_cost : ℝ
  bulk_dozen_cost : ℝ
  bulk_threshold : ℕ

/-- Calculates the minimum cost for a given number of notebooks -/
def minimum_cost (pricing : NotebookPricing) (num_students : ℕ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem minimum_cost_for_227_students :
  let pricing : NotebookPricing := {
    single_cost := 0.3,
    dozen_cost := 3.0,
    bulk_dozen_cost := 2.7,
    bulk_threshold := 10
  }
  minimum_cost pricing 227 = 51.3 := by sorry

end NUMINAMATH_CALUDE_minimum_cost_for_227_students_l2441_244134


namespace NUMINAMATH_CALUDE_complex_point_on_line_l2441_244132

theorem complex_point_on_line (a : ℝ) : 
  let z₁ : ℂ := 1 - a * Complex.I
  let z₂ : ℂ := (2 + Complex.I) ^ 2
  let z : ℂ := z₁ / z₂
  (5 * z.re - 5 * z.im + 3 = 0) → a = 22 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l2441_244132


namespace NUMINAMATH_CALUDE_logarithm_bijection_l2441_244118

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- State the theorem
theorem logarithm_bijection (a : ℝ) (ha : a > 1) :
  ∃ f : PositiveReals → ℝ, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_logarithm_bijection_l2441_244118


namespace NUMINAMATH_CALUDE_angle_D_measure_l2441_244110

-- Define the angles in degrees
def angle_A : ℝ := 50
def angle_B : ℝ := 35
def angle_C : ℝ := 40

-- Define the configuration
structure TriangleConfiguration where
  -- Triangle 1
  internal_angle_A : ℝ
  internal_angle_B : ℝ
  -- External triangle
  external_angle_C : ℝ
  -- Constraints
  angle_A_eq : internal_angle_A = angle_A
  angle_B_eq : internal_angle_B = angle_B
  angle_C_eq : external_angle_C = angle_C

-- Theorem statement
theorem angle_D_measure (config : TriangleConfiguration) :
  ∃ (angle_D : ℝ), angle_D = 125 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2441_244110
