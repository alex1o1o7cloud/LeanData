import Mathlib

namespace NUMINAMATH_CALUDE_leadership_selection_count_l1759_175970

/-- The number of ways to choose a president, vice president, and a 3-person committee from a group of people. -/
def choose_leadership (total : ℕ) (males : ℕ) (females : ℕ) : ℕ :=
  let remaining := total - 2  -- After choosing president and vice president
  let committee_choices := 
    (males.choose 1 * females.choose 2) +  -- 1 male and 2 females
    (males.choose 2 * females.choose 1)    -- 2 males and 1 female
  (total * (total - 1)) * committee_choices

/-- The theorem stating the number of ways to choose leadership positions from a specific group. -/
theorem leadership_selection_count : 
  choose_leadership 10 6 4 = 8640 := by
  sorry


end NUMINAMATH_CALUDE_leadership_selection_count_l1759_175970


namespace NUMINAMATH_CALUDE_equation_solution_l1759_175945

theorem equation_solution :
  ∃ x : ℝ, x = 1 ∧ 2021 * x = 2022 * (x^2021)^(1/2021) - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1759_175945


namespace NUMINAMATH_CALUDE_circle_symmetry_l1759_175982

/-- Given two circles and a line of symmetry, prove that the parameter 'a' in the first circle's equation must equal 2 for the circles to be symmetrical. -/
theorem circle_symmetry (x y : ℝ) (a : ℝ) : 
  (∀ x y, x^2 + y^2 - a*x + 2*y + 1 = 0) →  -- First circle equation
  (∀ x y, x^2 + y^2 = 1) →                  -- Second circle equation
  (∀ x y, x - y = 1) →                      -- Line of symmetry
  a = 2 := by
sorry


end NUMINAMATH_CALUDE_circle_symmetry_l1759_175982


namespace NUMINAMATH_CALUDE_set_difference_N_M_l1759_175963

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 2, 3, 7}

theorem set_difference_N_M : N \ M = {7} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_N_M_l1759_175963


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l1759_175958

theorem triangle_side_length_range (a b c : ℝ) :
  (|a + b - 4| + (a - b + 2)^2 = 0) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (2 < c ∧ c < 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l1759_175958


namespace NUMINAMATH_CALUDE_softball_team_composition_l1759_175968

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 14 → ratio = 5/9 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 4 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_composition_l1759_175968


namespace NUMINAMATH_CALUDE_not_always_complete_gear_possible_l1759_175974

-- Define the number of teeth on each gear
def num_teeth : ℕ := 13

-- Define the number of pairs of teeth removed
def num_removed : ℕ := 4

-- Define a type for the positions of removed teeth
def RemovedTeeth := Fin num_teeth

-- Define a function to check if two positions overlap after rotation
def overlaps (x y : RemovedTeeth) (rotation : ℕ) : Prop :=
  (x.val + rotation) % num_teeth = y.val

-- State the theorem
theorem not_always_complete_gear_possible : ∃ (removed : Fin num_removed → RemovedTeeth),
  ∀ (rotation : ℕ), ∃ (i j : Fin num_removed), i ≠ j ∧ overlaps (removed i) (removed j) rotation :=
sorry

end NUMINAMATH_CALUDE_not_always_complete_gear_possible_l1759_175974


namespace NUMINAMATH_CALUDE_fencing_cost_is_5300_l1759_175925

/-- Calculates the total cost of fencing a rectangular plot -/
def totalFencingCost (length width fenceCostPerMeter : ℝ) : ℝ :=
  2 * (length + width) * fenceCostPerMeter

/-- Theorem: The total cost of fencing the given rectangular plot is $5300 -/
theorem fencing_cost_is_5300 :
  let length : ℝ := 70
  let width : ℝ := 30
  let fenceCostPerMeter : ℝ := 26.50
  totalFencingCost length width fenceCostPerMeter = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_5300_l1759_175925


namespace NUMINAMATH_CALUDE_david_presents_l1759_175915

theorem david_presents (christmas : ℕ) (easter : ℕ) (birthday : ℕ) 
  (h1 : christmas = 60)
  (h2 : birthday = 3 * easter)
  (h3 : easter = christmas / 2 - 10) : 
  christmas + easter + birthday = 140 := by
  sorry

end NUMINAMATH_CALUDE_david_presents_l1759_175915


namespace NUMINAMATH_CALUDE_train_overtake_time_l1759_175901

/-- The time (in seconds) for a faster train to overtake a slower train after they meet -/
def overtake_time (v1 v2 l : ℚ) : ℚ :=
  (2 * l) / ((v2 - v1) / 3600)

theorem train_overtake_time :
  let v1 : ℚ := 50  -- speed of slower train (mph)
  let v2 : ℚ := 70  -- speed of faster train (mph)
  let l : ℚ := 1/6  -- length of each train (miles)
  overtake_time v1 v2 l = 60 := by
sorry

end NUMINAMATH_CALUDE_train_overtake_time_l1759_175901


namespace NUMINAMATH_CALUDE_candy_mixture_cost_per_pound_l1759_175905

/-- Calculates the desired cost per pound of a candy mixture --/
theorem candy_mixture_cost_per_pound 
  (weight_expensive : ℝ) 
  (price_expensive : ℝ) 
  (weight_cheap : ℝ) 
  (price_cheap : ℝ) 
  (h1 : weight_expensive = 20) 
  (h2 : price_expensive = 10) 
  (h3 : weight_cheap = 80) 
  (h4 : price_cheap = 5) : 
  (weight_expensive * price_expensive + weight_cheap * price_cheap) / (weight_expensive + weight_cheap) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_per_pound_l1759_175905


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1759_175951

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1759_175951


namespace NUMINAMATH_CALUDE_terry_age_proof_l1759_175967

/-- Nora's current age -/
def nora_age : ℕ := 10

/-- Terry's age in 10 years -/
def terry_future_age : ℕ := 4 * nora_age

/-- Terry's current age -/
def terry_current_age : ℕ := terry_future_age - 10

theorem terry_age_proof : terry_current_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_terry_age_proof_l1759_175967


namespace NUMINAMATH_CALUDE_sequence_product_l1759_175965

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that the product of the second term of the geometric sequence and
    the difference of the second and first terms of the arithmetic sequence is -8. -/
theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (∀ d : ℝ, -9 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -1) →  -- arithmetic sequence condition
  (∃ r : ℝ, -9 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -1) →  -- geometric sequence condition
  b₂ * (a₂ - a₁) = -8 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l1759_175965


namespace NUMINAMATH_CALUDE_juan_running_time_l1759_175927

theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 80) (h2 : speed = 10) :
  distance / speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_time_l1759_175927


namespace NUMINAMATH_CALUDE_ruby_reading_homework_l1759_175921

theorem ruby_reading_homework (nina_math : ℕ) (nina_reading : ℕ) (ruby_math : ℕ) (ruby_reading : ℕ) :
  nina_math = 4 * ruby_math →
  nina_reading = 8 * ruby_reading →
  ruby_math = 6 →
  nina_math + nina_reading = 48 →
  ruby_reading = 3 := by
sorry

end NUMINAMATH_CALUDE_ruby_reading_homework_l1759_175921


namespace NUMINAMATH_CALUDE_number_plus_four_equals_six_l1759_175977

theorem number_plus_four_equals_six (x : ℤ) : x + 4 = 6 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_four_equals_six_l1759_175977


namespace NUMINAMATH_CALUDE_ratio_to_percentage_difference_l1759_175998

theorem ratio_to_percentage_difference (A B : ℝ) (h : A / B = 3 / 4) :
  (B - A) / B = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_difference_l1759_175998


namespace NUMINAMATH_CALUDE_church_rows_count_l1759_175997

/-- Represents the seating arrangement in a church --/
structure ChurchSeating where
  chairs_per_row : ℕ
  people_per_chair : ℕ
  total_people : ℕ

/-- Calculates the number of rows in the church --/
def number_of_rows (s : ChurchSeating) : ℕ :=
  s.total_people / (s.chairs_per_row * s.people_per_chair)

/-- Theorem stating the number of rows in the church --/
theorem church_rows_count (s : ChurchSeating) 
  (h1 : s.chairs_per_row = 6)
  (h2 : s.people_per_chair = 5)
  (h3 : s.total_people = 600) :
  number_of_rows s = 20 := by
  sorry

#eval number_of_rows ⟨6, 5, 600⟩

end NUMINAMATH_CALUDE_church_rows_count_l1759_175997


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1759_175972

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ (b : ℝ), a + (5 * Complex.I) / (1 - 2 * Complex.I) = b * Complex.I) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1759_175972


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l1759_175992

theorem absolute_value_inequality_solution (x : ℝ) :
  (|2*x - 3| < 5) ↔ (-1 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l1759_175992


namespace NUMINAMATH_CALUDE_f_not_in_second_quadrant_l1759_175922

/-- A linear function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of f(x) = 2x - 3 does not pass through the second quadrant -/
theorem f_not_in_second_quadrant :
  ∀ x y : ℝ, f x = y → ¬(second_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_f_not_in_second_quadrant_l1759_175922


namespace NUMINAMATH_CALUDE_income_comparison_l1759_175904

/-- Given that Mary's income is 60% more than Tim's income, and Tim's income is 20% less than Juan's income, 
    prove that Mary's income is 128% of Juan's income. -/
theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.8)
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.28 := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l1759_175904


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1759_175910

/-- Given a function f: ℝ → ℝ satisfying certain conditions,
    prove that the solution set of f(x) + 1 > 2023 * exp(x) is (-∞, 0) -/
theorem solution_set_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, (deriv f) x - f x < 1)
  (h2 : f 0 = 2022) :
  {x : ℝ | f x + 1 > 2023 * Real.exp x} = Set.Iio 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1759_175910


namespace NUMINAMATH_CALUDE_circle_equation_holds_l1759_175988

/-- A circle in the Cartesian plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in general form --/
def CircleEquation (x y : ℝ) := x^2 + y^2 - 6*x = 0

/-- The circle represented by the equation x^2 + y^2 - 6x = 0 --/
def specificCircle : Circle := { center := (3, 0), radius := 3 }

/-- Theorem stating that the specificCircle satisfies the given equation --/
theorem circle_equation_holds (x y : ℝ) :
  CircleEquation x y ↔ (x - specificCircle.center.1)^2 + (y - specificCircle.center.2)^2 = specificCircle.radius^2 := by
  sorry

#check circle_equation_holds

end NUMINAMATH_CALUDE_circle_equation_holds_l1759_175988


namespace NUMINAMATH_CALUDE_correct_logarithms_l1759_175971

-- Define the logarithm function
noncomputable def log (x : ℝ) : ℝ := Real.log x

-- Define the variables a, b, and c
variable (a b c : ℝ)

-- Define the given logarithmic relationships
axiom log_3 : log 3 = 2*a - b
axiom log_5 : log 5 = a + c
axiom log_2 : log 2 = 1 - a - c
axiom log_9 : log 9 = 4*a - 2*b
axiom log_14 : log 14 = 1 - c + 2*b

-- State the theorem to be proved
theorem correct_logarithms :
  log 1.5 = 3*a - b + c - 1 ∧ log 7 = 2*b + c :=
by sorry

end NUMINAMATH_CALUDE_correct_logarithms_l1759_175971


namespace NUMINAMATH_CALUDE_pig_count_l1759_175987

theorem pig_count (initial_pigs : ℕ) : initial_pigs + 86 = 150 → initial_pigs = 64 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l1759_175987


namespace NUMINAMATH_CALUDE_distance_to_line_segment_equidistant_points_vertical_line_equidistant_points_diagonal_l1759_175949

-- Define the distance function from a point to a line segment
def distance_point_to_segment (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the line segment l: x-y-3=0 (3 ≤ x ≤ 5)
def line_segment_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 3 = 0 ∧ 3 ≤ p.1 ∧ p.1 ≤ 5}

-- Theorem 1
theorem distance_to_line_segment :
  distance_point_to_segment (1, 1) line_segment_l = Real.sqrt 5 := by sorry

-- Define the set of points equidistant from two line segments
def equidistant_points (l₁ l₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | distance_point_to_segment p l₁ = distance_point_to_segment p l₂}

-- Define line segments AB and CD for Theorem 2
def line_segment_AB : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
def line_segment_CD : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Theorem 2
theorem equidistant_points_vertical_line :
  equidistant_points line_segment_AB line_segment_CD = {p : ℝ × ℝ | p.1 = 0} := by sorry

-- Define line segments AB and CD for Theorem 3
def line_segment_AB' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0 ∧ -1 ≤ p.1 ∧ p.1 ≤ 1}
def line_segment_CD' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem 3
theorem equidistant_points_diagonal :
  equidistant_points line_segment_AB' line_segment_CD' = {p : ℝ × ℝ | p.1^2 - p.2^2 = 0} := by sorry

end NUMINAMATH_CALUDE_distance_to_line_segment_equidistant_points_vertical_line_equidistant_points_diagonal_l1759_175949


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1759_175989

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if S_m = 2 and S_2m = 10, then S_3m = 24. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →  -- Definition of S_n for arithmetic sequence
  (S m = 2) →
  (S (2 * m) = 10) →
  (S (3 * m) = 24) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1759_175989


namespace NUMINAMATH_CALUDE_prob_third_term_four_sum_of_fraction_parts_l1759_175933

/-- Set of permutations of 1,2,3,4,5,6 with restrictions -/
def T : Set (Fin 6 → Fin 6) :=
  { σ | Function.Bijective σ ∧ 
        σ 0 ≠ 0 ∧ σ 0 ≠ 1 ∧
        σ 1 ≠ 2 }

/-- The cardinality of set T -/
def T_size : ℕ := 48

/-- The number of permutations in T where the third term is 4 -/
def favorable_outcomes : ℕ := 12

/-- The probability of the third term being 4 in a randomly chosen permutation from T -/
theorem prob_third_term_four : 
  (favorable_outcomes : ℚ) / T_size = 1 / 4 :=
sorry

/-- The sum of numerator and denominator in the probability fraction -/
theorem sum_of_fraction_parts : 
  1 + 4 = 5 :=
sorry

end NUMINAMATH_CALUDE_prob_third_term_four_sum_of_fraction_parts_l1759_175933


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1759_175969

theorem fraction_power_equality : (125000 : ℝ)^5 / (25000 : ℝ)^5 = 3125 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1759_175969


namespace NUMINAMATH_CALUDE_tv_price_calculation_l1759_175984

/-- Calculates the final price of an item given the original price, discount rate, tax rate, and rebate amount. -/
def finalPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) (rebate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discountRate)
  let priceWithTax := salePrice * (1 + taxRate)
  priceWithTax - rebate

/-- Theorem stating that the final price of a $1200 item with 30% discount, 8% tax, and $50 rebate is $857.2. -/
theorem tv_price_calculation :
  finalPrice 1200 0.30 0.08 50 = 857.2 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_calculation_l1759_175984


namespace NUMINAMATH_CALUDE_line_equation_proof_l1759_175932

/-- Given a line defined by (3, -4) · ((x, y) - (2, 7)) = 0, prove that its slope-intercept form y = mx + b has m = 3/4 and b = 11/2 -/
theorem line_equation_proof (x y : ℝ) : 
  (3 * (x - 2) + (-4) * (y - 7) = 0) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 3/4 ∧ b = 11/2) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1759_175932


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1759_175912

theorem solution_set_inequality (x : ℝ) : 
  (x * (2 - x) > 0) ↔ (0 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1759_175912


namespace NUMINAMATH_CALUDE_probability_12th_roll_last_proof_l1759_175940

/-- The probability of the 12th roll being the last roll when rolling a standard 
    eight-sided die until getting the same number on consecutive rolls -/
def probability_12th_roll_last : ℚ :=
  (7^10 : ℚ) / (8^11 : ℚ)

/-- The number of sides on the standard die -/
def num_sides : ℕ := 8

/-- The number of rolls -/
def num_rolls : ℕ := 12

theorem probability_12th_roll_last_proof :
  probability_12th_roll_last = (7^(num_rolls - 2) : ℚ) / (num_sides^(num_rolls - 1) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_12th_roll_last_proof_l1759_175940


namespace NUMINAMATH_CALUDE_john_duck_profit_l1759_175906

/-- Calculates the profit from selling ducks given the following conditions:
  * number_of_ducks: The number of ducks bought and sold
  * cost_per_duck: The cost of each duck when buying
  * weight_per_duck: The weight of each duck in pounds
  * price_per_pound: The selling price per pound of duck
-/
def duck_profit (number_of_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (price_per_pound : ℚ) : ℚ :=
  let total_cost := number_of_ducks * cost_per_duck
  let revenue_per_duck := weight_per_duck * price_per_pound
  let total_revenue := number_of_ducks * revenue_per_duck
  total_revenue - total_cost

/-- Theorem stating that under the given conditions, the profit is $300 -/
theorem john_duck_profit :
  duck_profit 30 10 4 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_john_duck_profit_l1759_175906


namespace NUMINAMATH_CALUDE_garden_area_l1759_175941

/-- Represents a rectangular garden with specific properties -/
structure Garden where
  width : ℝ
  length : ℝ
  perimeter_minus_one_side : ℝ

/-- The properties of the garden as described in the problem -/
def garden_properties (g : Garden) : Prop :=
  g.perimeter_minus_one_side = 60 ∧
  g.length = 2 * g.width

/-- The theorem stating that a garden with the given properties has an area of 450 square meters -/
theorem garden_area (g : Garden) (h : garden_properties g) : g.width * g.length = 450 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_l1759_175941


namespace NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l1759_175907

theorem not_divisible_by_1000_power_minus_1 (m : ℕ) :
  ¬(1000^m - 1 ∣ 1978^m - 1) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l1759_175907


namespace NUMINAMATH_CALUDE_lcm_n_n_plus_3_l1759_175920

theorem lcm_n_n_plus_3 (n : ℕ) :
  lcm n (n + 3) = if n % 3 = 0 then n * (n + 3) / 3 else n * (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_lcm_n_n_plus_3_l1759_175920


namespace NUMINAMATH_CALUDE_min_sum_product_2400_l1759_175954

theorem min_sum_product_2400 (x y z : ℕ+) (h : x * y * z = 2400) :
  x + y + z ≥ 43 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_product_2400_l1759_175954


namespace NUMINAMATH_CALUDE_two_prime_roots_equation_l1759_175942

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem two_prime_roots_equation (n : ℕ) (h_pos : n > 0) :
  ∃ (x₁ x₂ : ℕ), 
    is_prime x₁ ∧ 
    is_prime x₂ ∧ 
    x₁ ≠ x₂ ∧
    2 * x₁^2 - 8*n*x₁ + 10*x₁ - n^2 + 35*n - 76 = 0 ∧
    2 * x₂^2 - 8*n*x₂ + 10*x₂ - n^2 + 35*n - 76 = 0 →
  n = 3 ∧ x₁ = 2 ∧ x₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_two_prime_roots_equation_l1759_175942


namespace NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l1759_175916

theorem gcf_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l1759_175916


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1759_175956

theorem binomial_coefficient_two (n : ℕ+) : (n.val.choose 2) = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1759_175956


namespace NUMINAMATH_CALUDE_initial_men_count_l1759_175900

/-- Proves that the initial number of men is 1000, given the conditions of the problem. -/
theorem initial_men_count (initial_days : ℝ) (joined_days : ℝ) (joined_men : ℕ) : 
  initial_days = 20 →
  joined_days = 16.67 →
  joined_men = 200 →
  (∃ (initial_men : ℕ), initial_men * initial_days = (initial_men + joined_men) * joined_days ∧ initial_men = 1000) :=
by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1759_175900


namespace NUMINAMATH_CALUDE_margaret_score_l1759_175946

theorem margaret_score (average_score : ℝ) (marco_percentage : ℝ) (margaret_difference : ℝ) : 
  average_score = 90 →
  marco_percentage = 0.1 →
  margaret_difference = 5 →
  let marco_score := average_score * (1 - marco_percentage)
  let margaret_score := marco_score + margaret_difference
  margaret_score = 86 := by sorry

end NUMINAMATH_CALUDE_margaret_score_l1759_175946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_one_l1759_175926

theorem arithmetic_sequence_before_one (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 100 → d = -7 → n = 15 →
  a₁ + (n - 1) * d = 1 ∧ n - 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_one_l1759_175926


namespace NUMINAMATH_CALUDE_problem_statement_l1759_175990

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := Real.log x - x + 2

theorem problem_statement :
  (∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 1) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ≥ 1 → m * f x ≥ (x - 1) / (x + 1)) ↔ m ≥ 1/2) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi/2 →
    ((0 < α ∧ α < Real.pi/4 → f (Real.tan α) < -Real.cos (2*α)) ∧
     (α = Real.pi/4 → f (Real.tan α) = -Real.cos (2*α)) ∧
     (Real.pi/4 < α ∧ α < Real.pi/2 → f (Real.tan α) > -Real.cos (2*α)))) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1759_175990


namespace NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l1759_175999

theorem quadratic_vertex_ordinate 
  (a b c : ℝ) 
  (d : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b^2 - 4*a*c > 0) 
  (h3 : d = (Real.sqrt (b^2 - 4*a*c)) / a) :
  ∃! y : ℝ, y = -a * d^2 / 4 ∧ 
    y = a * (-b / (2*a))^2 + b * (-b / (2*a)) + c :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l1759_175999


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_of_n_l1759_175976

-- Define the number we're working with
def n : ℕ := 9999

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define a function to check if a number is a divisor of n
def is_divisor_of_n (d : ℕ) : Prop := n % d = 0

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem stating the sum of digits of the greatest prime divisor of n is 2
theorem sum_of_digits_of_greatest_prime_divisor_of_n : 
  ∃ p : ℕ, is_prime p ∧ is_divisor_of_n p ∧ 
    (∀ q : ℕ, is_prime q → is_divisor_of_n q → q ≤ p) ∧
    sum_of_digits p = 2 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_of_n_l1759_175976


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l1759_175944

/-- The number of ways to arrange volunteers among events --/
def arrangeVolunteers (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to arrange volunteers among events, excluding one event --/
def arrangeVolunteersExcludeOne (n : ℕ) (k : ℕ) : ℕ := k * (k-1)^n

/-- The number of ways to arrange volunteers to only one event --/
def arrangeVolunteersToOne (n : ℕ) (k : ℕ) : ℕ := k

theorem volunteer_arrangement_count :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 3  -- number of events
  arrangeVolunteers n k - k * arrangeVolunteersExcludeOne n (k-1) + arrangeVolunteersToOne n k = 150 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l1759_175944


namespace NUMINAMATH_CALUDE_exists_cheaper_bulk_purchase_l1759_175943

/-- The original price of a notebook --/
def original_price : ℝ := 8

/-- The discounted price of a notebook when buying more than 100 --/
def discounted_price : ℝ := original_price - 2

/-- The cost of buying n books under Plan 1 (n ≤ 100) --/
def cost_plan1 (n : ℝ) : ℝ := original_price * n

/-- The cost of buying n books under Plan 2 (n > 100) --/
def cost_plan2 (n : ℝ) : ℝ := discounted_price * n

/-- Theorem stating that there exists a scenario where buying n books (n > 100) 
    costs less than buying 80 books under Plan 1 --/
theorem exists_cheaper_bulk_purchase :
  ∃ n : ℝ, n > 100 ∧ cost_plan2 n < cost_plan1 80 := by
  sorry

end NUMINAMATH_CALUDE_exists_cheaper_bulk_purchase_l1759_175943


namespace NUMINAMATH_CALUDE_cubic_equation_roots_inequality_l1759_175948

/-- Given a cubic equation x³ + ax² + bx + c = 0 with three real roots p ≤ q ≤ r,
    prove that a² - 3b ≥ 0 and √(a² - 3b) ≤ r - p -/
theorem cubic_equation_roots_inequality (a b c p q r : ℝ) :
  p ≤ q ∧ q ≤ r ∧
  p^3 + a*p^2 + b*p + c = 0 ∧
  q^3 + a*q^2 + b*q + c = 0 ∧
  r^3 + a*r^2 + b*r + c = 0 →
  a^2 - 3*b ≥ 0 ∧ Real.sqrt (a^2 - 3*b) ≤ r - p :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_inequality_l1759_175948


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_for_positive_x_l1759_175953

theorem necessary_but_not_sufficient_condition_for_positive_x :
  (∀ x : ℝ, x > 0 → x > -2) ∧
  (∃ x : ℝ, x > -2 ∧ x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_for_positive_x_l1759_175953


namespace NUMINAMATH_CALUDE_f_36_l1759_175978

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, f (x * y) = f x + f y)
variable (h2 : f 2 = p)
variable (h3 : f 3 = q)

-- State the theorem
theorem f_36 (p q : ℝ) : f 36 = 2 * (p + q) := by sorry

end NUMINAMATH_CALUDE_f_36_l1759_175978


namespace NUMINAMATH_CALUDE_total_prizes_l1759_175957

def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def yo_yos : ℕ := 18

theorem total_prizes : stuffed_animals + frisbees + yo_yos = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_prizes_l1759_175957


namespace NUMINAMATH_CALUDE_divisor_property_l1759_175938

theorem divisor_property (k : ℕ) (h : 5^k - k^5 = 1) : 15^k = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l1759_175938


namespace NUMINAMATH_CALUDE_equality_condition_for_squared_sum_equals_product_sum_l1759_175994

theorem equality_condition_for_squared_sum_equals_product_sum (a b c : ℝ) :
  (a^2 + b^2 + c^2 = a*b + b*c + c*a) ↔ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_equality_condition_for_squared_sum_equals_product_sum_l1759_175994


namespace NUMINAMATH_CALUDE_positive_intervals_l1759_175985

-- Define the expression
def f (x : ℝ) : ℝ := (x - 2) * (x + 3)

-- Theorem statement
theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ x < -3 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_intervals_l1759_175985


namespace NUMINAMATH_CALUDE_hilt_detergent_usage_l1759_175930

/-- The amount of detergent Mrs. Hilt uses per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The number of pounds of clothes to be washed -/
def pounds_of_clothes : ℝ := 9

/-- Theorem: Mrs. Hilt will use 18 ounces of detergent to wash 9 pounds of clothes -/
theorem hilt_detergent_usage : detergent_per_pound * pounds_of_clothes = 18 := by
  sorry

end NUMINAMATH_CALUDE_hilt_detergent_usage_l1759_175930


namespace NUMINAMATH_CALUDE_second_chapter_pages_l1759_175947

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ

/-- Properties of the book -/
def book_properties (b : Book) : Prop :=
  b.total_pages = 93 ∧ b.first_chapter_pages = 60 ∧ b.total_pages = b.first_chapter_pages + b.second_chapter_pages

/-- Theorem stating that the second chapter has 33 pages -/
theorem second_chapter_pages (b : Book) (h : book_properties b) : b.second_chapter_pages = 33 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l1759_175947


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1759_175929

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (y : ℝ), x^4 + 16 = (x^2 - 4*x + 4) * y :=
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1759_175929


namespace NUMINAMATH_CALUDE_arccos_of_neg_one_eq_pi_l1759_175962

theorem arccos_of_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_neg_one_eq_pi_l1759_175962


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_theorem_l1759_175959

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

def geometric_sum_reciprocals (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  (1 / a) * (1 - (1 / r)^n) / (1 - (1 / r))

theorem geometric_sequence_ratio_theorem :
  let a : ℚ := 1 / 4
  let r : ℚ := 2
  let n : ℕ := 10
  let S := geometric_sum a r n
  let S' := geometric_sum_reciprocals a r n
  S / S' = 32 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_theorem_l1759_175959


namespace NUMINAMATH_CALUDE_binary_10110011_equals_179_l1759_175995

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_10110011_equals_179 :
  binary_to_decimal [true, true, false, false, true, true, false, true] = 179 := by
  sorry

end NUMINAMATH_CALUDE_binary_10110011_equals_179_l1759_175995


namespace NUMINAMATH_CALUDE_store_a_more_cost_effective_for_large_x_l1759_175931

/-- Represents the cost of purchasing table tennis rackets from Store A -/
def cost_store_a (x : ℕ) : ℚ :=
  if x ≤ 10 then 30 * x else 300 + 21 * (x - 10)

/-- Represents the cost of purchasing table tennis rackets from Store B -/
def cost_store_b (x : ℕ) : ℚ := 25.5 * x

/-- Theorem stating that Store A is more cost-effective than Store B for x > 20 -/
theorem store_a_more_cost_effective_for_large_x :
  ∀ x : ℕ, x > 20 → cost_store_a x < cost_store_b x :=
by
  sorry

/-- Helper lemma to show that cost_store_a simplifies to 21x + 90 for x > 10 -/
lemma cost_store_a_simplification (x : ℕ) (h : x > 10) :
  cost_store_a x = 21 * x + 90 :=
by
  sorry

end NUMINAMATH_CALUDE_store_a_more_cost_effective_for_large_x_l1759_175931


namespace NUMINAMATH_CALUDE_problem_solution_l1759_175961

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := f (x + 1) - x

theorem problem_solution :
  (∃ (x_max : ℝ), ∀ (x : ℝ), g x ≤ g x_max ∧ g x_max = 0) ∧
  (∀ (n : ℕ), n > 0 → (1 + 1 / n : ℝ) ^ n < Real.exp 1) ∧
  (∀ (a b : ℝ), 0 < a → a < b → f b - f a > 2 * a * (b - a) / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1759_175961


namespace NUMINAMATH_CALUDE_daycare_count_l1759_175964

/-- The real number of toddlers in the daycare -/
def real_count (bill_count playground_count new_count double_counted missed : ℕ) : ℕ :=
  bill_count - double_counted + missed - playground_count + new_count

/-- Theorem stating the real number of toddlers given the conditions -/
theorem daycare_count : real_count 28 6 4 9 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_daycare_count_l1759_175964


namespace NUMINAMATH_CALUDE_house_selling_price_l1759_175975

/-- Represents the total number of houses in the village -/
def total_houses : ℕ := 15

/-- Represents the total cost of construction for the entire village in millions of units -/
def total_cost : ℕ := 150 + 105 + 225 + 45

/-- Represents the markup percentage as a rational number -/
def markup : ℚ := 1 / 5

/-- Theorem: The selling price of each house in the village is 42 million units -/
theorem house_selling_price : 
  ∃ (cost_per_house : ℕ) (selling_price : ℕ),
    cost_per_house * total_houses = total_cost ∧
    selling_price = cost_per_house + cost_per_house * markup ∧
    selling_price = 42 :=
by sorry

end NUMINAMATH_CALUDE_house_selling_price_l1759_175975


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l1759_175924

theorem opposite_of_negative_three_fourths :
  let x : ℚ := -3/4
  let y : ℚ := 3/4
  (∀ z : ℚ, z + x = 0 ↔ z = y) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l1759_175924


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1759_175952

theorem quadratic_minimum (x : ℝ) (h : x > 0) : x^2 - 2*x + 3 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1759_175952


namespace NUMINAMATH_CALUDE_line_passes_through_center_line_is_diameter_l1759_175935

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Theorem: The line passes through the center of the circle
theorem line_passes_through_center :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y :=
sorry

-- Theorem: The line is a diameter of the circle
theorem line_is_diameter :
  ∀ (x y : ℝ), circle_eq x y → line_eq x y → 
  ∃ (x' y' : ℝ), circle_eq x' y' ∧ line_eq x' y' ∧ 
  (x - x')^2 + (y - y')^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_center_line_is_diameter_l1759_175935


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1759_175928

theorem solution_satisfies_system :
  ∃ (x y z w : ℝ), 
    (x = 2 ∧ y = 2 ∧ z = 0 ∧ w = 0) ∧
    (x + y + Real.sqrt z = 4) ∧
    (Real.sqrt x * Real.sqrt y - Real.sqrt w = 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1759_175928


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l1759_175973

theorem pencil_buyers_difference : ∀ (pencil_cost : ℕ) 
  (seventh_graders : ℕ) (sixth_graders : ℕ),
  pencil_cost > 0 →
  pencil_cost * seventh_graders = 143 →
  pencil_cost * sixth_graders = 195 →
  sixth_graders ≤ 30 →
  sixth_graders - seventh_graders = 4 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l1759_175973


namespace NUMINAMATH_CALUDE_union_equals_A_l1759_175902

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - a = 0}

-- State the theorem
theorem union_equals_A (a : ℝ) : (A ∪ B a = A) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l1759_175902


namespace NUMINAMATH_CALUDE_tims_books_l1759_175936

theorem tims_books (sandy_books : ℕ) (benny_books : ℕ) (total_books : ℕ) :
  sandy_books = 10 →
  benny_books = 24 →
  total_books = 67 →
  ∃ tim_books : ℕ, tim_books = total_books - (sandy_books + benny_books) ∧ tim_books = 33 :=
by sorry

end NUMINAMATH_CALUDE_tims_books_l1759_175936


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l1759_175955

/-- The line equation y = x + a is tangent to the curve y = x^3 - x^2 + 1 at the point (-1/3, 23/27) when a = 32/27 -/
theorem line_tangent_to_curve :
  let line (x : ℝ) := x + 32/27
  let curve (x : ℝ) := x^3 - x^2 + 1
  let tangent_point : ℝ × ℝ := (-1/3, 23/27)
  (∀ x, line x ≠ curve x ∨ x = tangent_point.1) ∧
  (line tangent_point.1 = curve tangent_point.1) ∧
  (HasDerivAt curve (line tangent_point.1) tangent_point.1) :=
by sorry


end NUMINAMATH_CALUDE_line_tangent_to_curve_l1759_175955


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1759_175986

theorem quadratic_one_root (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃! x : ℝ, x^2 + 6*m*x + m - n = 0) →
  (0 < m ∧ m < 1/9 ∧ n = m - 9*m^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1759_175986


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1759_175914

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  m : ℝ

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_intersection_theorem (E : Ellipse) (L : IntersectingLine) : 
  (E.a^2 - E.b^2 = 1) →  -- Focal length is 2
  (1 / E.a^2 + (9/4) / E.b^2 = 1) →  -- Ellipse passes through (1, 3/2)
  (∃ (x₁ y₁ x₂ y₂ : ℝ),  -- Intersection points exist
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    y₁ = 3/2 * x₁ + L.m ∧
    y₂ = 3/2 * x₂ + L.m) →
  (∃ (k₁ k₂ : ℝ),  -- Slope ratio condition
    k₁ / k₂ = 2 ∧
    k₁ = y₂ / (x₂ + 2) ∧
    k₂ = y₁ / (x₁ - 2)) →
  L.m = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1759_175914


namespace NUMINAMATH_CALUDE_line_through_points_l1759_175983

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨8, 9⟩
  let p2 : Point := ⟨2, -3⟩
  let p3 : Point := ⟨5, 3⟩
  let p4 : Point := ⟨6, 6⟩
  let p5 : Point := ⟨3, 0⟩
  let p6 : Point := ⟨0, -9⟩
  let p7 : Point := ⟨4, 1⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p7 ∧ 
  ¬collinear p1 p2 p4 ∧ 
  ¬collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1759_175983


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1759_175911

/-- Given two points P and Q in a 2D plane, we define a circle with PQ as its diameter. -/
def circle_with_diameter (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {point | (point.1 - (P.1 + Q.1) / 2)^2 + (point.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4}

/-- The theorem states that for P(4,0) and Q(0,2), the equation of the circle with PQ as diameter is (x-2)^2 + (y-1)^2 = 5. -/
theorem circle_equation_proof :
  circle_with_diameter (4, 0) (0, 2) = {point : ℝ × ℝ | (point.1 - 2)^2 + (point.2 - 1)^2 = 5} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1759_175911


namespace NUMINAMATH_CALUDE_vertex_angle_is_40_l1759_175918

-- Define an isosceles triangle
structure IsoscelesTriangle where
  vertexAngle : ℝ
  baseAngle : ℝ
  sum_of_angles : vertexAngle + 2 * baseAngle = 180
  base_angle_relation : baseAngle = vertexAngle + 30

-- Theorem statement
theorem vertex_angle_is_40 (t : IsoscelesTriangle) : t.vertexAngle = 40 :=
by sorry

end NUMINAMATH_CALUDE_vertex_angle_is_40_l1759_175918


namespace NUMINAMATH_CALUDE_max_value_ab_l1759_175939

theorem max_value_ab (a b : ℕ) : 
  a > 1 → b > 1 → a^b * b^a + a^b + b^a = 5329 → a^b ≤ 64 := by
sorry

end NUMINAMATH_CALUDE_max_value_ab_l1759_175939


namespace NUMINAMATH_CALUDE_min_y_value_l1759_175919

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 50*y + 64) : y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_min_y_value_l1759_175919


namespace NUMINAMATH_CALUDE_final_debt_calculation_l1759_175950

def calculate_debt (initial_loan : ℝ) (repayment1_percent : ℝ) (loan2 : ℝ) 
                   (repayment2_percent : ℝ) (loan3 : ℝ) (repayment3_percent : ℝ) : ℝ :=
  let debt1 := initial_loan * (1 - repayment1_percent)
  let debt2 := debt1 + loan2
  let debt3 := debt2 * (1 - repayment2_percent)
  let debt4 := debt3 + loan3
  debt4 * (1 - repayment3_percent)

theorem final_debt_calculation :
  calculate_debt 40 0.25 25 0.5 30 0.1 = 51.75 := by
  sorry

end NUMINAMATH_CALUDE_final_debt_calculation_l1759_175950


namespace NUMINAMATH_CALUDE_intersection_complement_sets_l1759_175966

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_sets : M ∩ (U \ N) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_sets_l1759_175966


namespace NUMINAMATH_CALUDE_orvin_max_balloons_l1759_175908

/-- Represents the price of a balloon in cents -/
def regularPrice : ℕ := 200

/-- Represents the number of balloons Orvin can afford at regular price -/
def regularAffordable : ℕ := 40

/-- Represents the maximum number of discounted balloons -/
def maxDiscounted : ℕ := 10

/-- Calculates the total money Orvin has in cents -/
def totalMoney : ℕ := regularPrice * regularAffordable

/-- Calculates the price of a discounted balloon in cents -/
def discountedPrice : ℕ := regularPrice / 2

/-- Calculates the cost of buying a regular and a discounted balloon in cents -/
def pairCost : ℕ := regularPrice + discountedPrice

/-- Represents the maximum number of balloons Orvin can buy -/
def maxBalloons : ℕ := 42

theorem orvin_max_balloons :
  regularPrice > 0 →
  (totalMoney - (maxDiscounted / 2 * pairCost)) / regularPrice + maxDiscounted = maxBalloons :=
by sorry

end NUMINAMATH_CALUDE_orvin_max_balloons_l1759_175908


namespace NUMINAMATH_CALUDE_fraction_simplification_l1759_175996

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  12 / (x^2 - 9) - 2 / (x - 3) = -2 / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1759_175996


namespace NUMINAMATH_CALUDE_passengers_per_bus_l1759_175923

def total_people : ℕ := 1230
def num_buses : ℕ := 26

theorem passengers_per_bus :
  (total_people / num_buses : ℕ) = 47 := by sorry

end NUMINAMATH_CALUDE_passengers_per_bus_l1759_175923


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1759_175917

theorem solve_linear_equation :
  ∀ x : ℝ, 5 + 3.6 * x = 2.1 * x - 25 → x = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1759_175917


namespace NUMINAMATH_CALUDE_eugene_apples_proof_l1759_175937

def apples_from_eugene (initial_apples final_apples : ℝ) : ℝ :=
  final_apples - initial_apples

theorem eugene_apples_proof (initial_apples final_apples : ℝ) :
  apples_from_eugene initial_apples final_apples =
  final_apples - initial_apples :=
by
  sorry

#eval apples_from_eugene 20.0 27.0

end NUMINAMATH_CALUDE_eugene_apples_proof_l1759_175937


namespace NUMINAMATH_CALUDE_base_eight_47_equals_39_l1759_175960

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-eight number 47 is equal to the base-ten number 39 -/
theorem base_eight_47_equals_39 : base_eight_to_ten 4 7 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_47_equals_39_l1759_175960


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l1759_175979

/-- In a pentagon with angles 104°, 97°, x°, 2x°, and R°, where the sum of all angles is 540°, 
    the measure of angle R is 204°. -/
theorem pentagon_angle_measure (x : ℝ) (R : ℝ) : 
  104 + 97 + x + 2*x + R = 540 → R = 204 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l1759_175979


namespace NUMINAMATH_CALUDE_bill_due_time_l1759_175993

/-- Proves that given a bill with specified face value, true discount, and annual interest rate,
    the time until the bill is due is 9 months. -/
theorem bill_due_time (face_value : ℝ) (true_discount : ℝ) (annual_interest_rate : ℝ) :
  face_value = 2240 →
  true_discount = 240 →
  annual_interest_rate = 0.16 →
  (face_value / (face_value - true_discount) - 1) / annual_interest_rate * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bill_due_time_l1759_175993


namespace NUMINAMATH_CALUDE_smallest_y_coordinate_l1759_175991

theorem smallest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_coordinate_l1759_175991


namespace NUMINAMATH_CALUDE_distinct_collections_count_l1759_175909

/-- Represents the collection of letters in MATHEMATICAL -/
def mathematical : Finset Char := {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

/-- Represents the vowels in MATHEMATICAL -/
def vowels : Finset Char := {'A', 'E', 'I'}

/-- Represents the consonants in MATHEMATICAL -/
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'L'}

/-- Function to count occurrences of a character in MATHEMATICAL -/
def count (c : Char) : Nat := (mathematical.filter (· = c)).card

/-- The number of distinct possible collections of letters in the bag -/
def distinct_collections : Nat := sorry

theorem distinct_collections_count : distinct_collections = 220 := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l1759_175909


namespace NUMINAMATH_CALUDE_min_value_expression_l1759_175903

theorem min_value_expression (x : ℝ) : 
  (∃ (m : ℝ), ∀ (y : ℝ), (15 - y) * (13 - y) * (15 + y) * (13 + y) + 200 * y^2 ≥ m) ∧ 
  (∃ (z : ℝ), (15 - z) * (13 - z) * (15 + z) * (13 + z) + 200 * z^2 = 33) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1759_175903


namespace NUMINAMATH_CALUDE_problem_solution_l1759_175913

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * ↑(⌊x⌋) = 72) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1759_175913


namespace NUMINAMATH_CALUDE_h_is_even_l1759_175934

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the function h
def h (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g x * |f x|

-- State the theorem
theorem h_is_even (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) : 
  IsEven (h f g) := by
  sorry

end NUMINAMATH_CALUDE_h_is_even_l1759_175934


namespace NUMINAMATH_CALUDE_game_probability_l1759_175981

/-- The probability of a specific outcome in a game with 8 rounds -/
theorem game_probability : 
  -- Total number of rounds
  (total_rounds : ℕ) →
  -- Alex's probability of winning a round
  (alex_prob : ℚ) →
  -- Mel's probability of winning a round
  (mel_prob : ℚ) →
  -- Chelsea's probability of winning a round
  (chelsea_prob : ℚ) →
  -- Number of rounds Alex wins
  (alex_wins : ℕ) →
  -- Number of rounds Mel wins
  (mel_wins : ℕ) →
  -- Number of rounds Chelsea wins
  (chelsea_wins : ℕ) →
  -- Conditions
  total_rounds = 8 →
  alex_prob = 2/5 →
  mel_prob = 3 * chelsea_prob →
  alex_prob + mel_prob + chelsea_prob = 1 →
  alex_wins + mel_wins + chelsea_wins = total_rounds →
  alex_wins = 3 →
  mel_wins = 4 →
  chelsea_wins = 1 →
  -- Conclusion
  (Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins *
   alex_prob ^ alex_wins * mel_prob ^ mel_wins * chelsea_prob ^ chelsea_wins : ℚ) = 881/1000 := by
sorry

end NUMINAMATH_CALUDE_game_probability_l1759_175981


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1759_175980

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + y) = f x + f y) : 
  ∃ a : ℤ, ∀ x : ℤ, f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1759_175980
