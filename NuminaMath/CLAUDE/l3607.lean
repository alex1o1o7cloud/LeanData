import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l3607_360795

theorem complex_equation_solution :
  ∀ z : ℂ, (2 - 5*I) * z = 29 → z = 2 + 5*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3607_360795


namespace NUMINAMATH_CALUDE_power_equation_solution_l3607_360746

theorem power_equation_solution (m : ℝ) : (7 : ℝ) ^ (4 * m) = (1 / 7) ^ (2 * m - 18) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3607_360746


namespace NUMINAMATH_CALUDE_word_arrangements_l3607_360732

/-- The number of distinct letters in the word -/
def n : ℕ := 6

/-- The number of units to be arranged after combining the T's -/
def k : ℕ := 5

/-- The number of ways to arrange the T's within their unit -/
def t : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := k.factorial * t.factorial

theorem word_arrangements : total_arrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_word_arrangements_l3607_360732


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3607_360702

/-- Given a geometric sequence {a_n}, prove that if a_1 + a_2 = 40 and a_3 + a_4 = 60, then a_7 + a_8 = 135 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_sum1 : a 1 + a 2 = 40) (h_sum2 : a 3 + a 4 = 60) : a 7 + a 8 = 135 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3607_360702


namespace NUMINAMATH_CALUDE_yao_ming_shots_l3607_360744

/-- Represents the scoring details of a basketball player in a game -/
structure ScoringDetails where
  total_shots_made : ℕ
  total_points : ℕ
  three_pointers_made : ℕ

/-- Calculates the number of 2-point shots and free throws made given the scoring details -/
def calculate_shots (details : ScoringDetails) : ℕ × ℕ :=
  let two_pointers := (details.total_points - 3 * details.three_pointers_made) / 2
  let free_throws := details.total_shots_made - details.three_pointers_made - two_pointers
  (two_pointers, free_throws)

/-- Theorem stating that given Yao Ming's scoring details, he made 8 2-point shots and 3 free throws -/
theorem yao_ming_shots :
  let details : ScoringDetails := {
    total_shots_made := 14,
    total_points := 28,
    three_pointers_made := 3
  }
  calculate_shots details = (8, 3) := by sorry

end NUMINAMATH_CALUDE_yao_ming_shots_l3607_360744


namespace NUMINAMATH_CALUDE_rational_roots_quadratic_l3607_360714

theorem rational_roots_quadratic (m : ℤ) :
  (∃ x y : ℚ, m * x^2 - (m - 1) * x + 1 = 0 ∧ m * y^2 - (m - 1) * y + 1 = 0 ∧ x ≠ y) →
  m = 6 ∧ (1/2 : ℚ) * m - (m - 1) * (1/2) + 1 = 0 ∧ (1/3 : ℚ) * m - (m - 1) * (1/3) + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_quadratic_l3607_360714


namespace NUMINAMATH_CALUDE_center_locus_is_single_point_l3607_360784

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] where
  P : α
  Q : α

/-- A circle passing through two fixed points with constant radius -/
structure Circle (α : Type*) [NormedAddCommGroup α] where
  center : α
  radius : ℝ
  fixedPoints : FixedPoints α

/-- The locus of centers of circles passing through two fixed points -/
def CenterLocus (α : Type*) [NormedAddCommGroup α] (a : ℝ) : Set α :=
  {C : α | ∃ (circ : Circle α), circ.center = C ∧ circ.radius = a}

/-- The theorem stating that the locus of centers is a single point -/
theorem center_locus_is_single_point
  (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α]
  (a : ℝ) (points : FixedPoints α)
  (h : ‖points.P - points.Q‖ = 2 * a) :
  ∃! C, C ∈ CenterLocus α a :=
sorry

end NUMINAMATH_CALUDE_center_locus_is_single_point_l3607_360784


namespace NUMINAMATH_CALUDE_shower_tiles_l3607_360755

/-- Calculates the total number of tiles in a shower --/
def total_tiles (sides : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  sides * width * height

/-- Theorem: The total number of tiles in a 3-sided shower with 8 tiles in width and 20 tiles in height is 480 --/
theorem shower_tiles : total_tiles 3 8 20 = 480 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_l3607_360755


namespace NUMINAMATH_CALUDE_ancient_chinese_fruit_problem_l3607_360700

/-- Represents the ancient Chinese fruit problem -/
theorem ancient_chinese_fruit_problem 
  (x y : ℚ) -- x: number of bitter fruits, y: number of sweet fruits
  (h1 : x + y = 1000) -- total number of fruits
  (h2 : 7 * (4 / 7 : ℚ) = 4) -- cost of 7 bitter fruits
  (h3 : 9 * (11 / 9 : ℚ) = 11) -- cost of 9 sweet fruits
  (h4 : (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = 999) -- total cost
  : (x + y = 1000 ∧ (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = 999) :=
by sorry

end NUMINAMATH_CALUDE_ancient_chinese_fruit_problem_l3607_360700


namespace NUMINAMATH_CALUDE_lucy_bank_balance_l3607_360705

theorem lucy_bank_balance (initial_balance deposit withdrawal : ℕ) :
  initial_balance = 65 →
  deposit = 15 →
  withdrawal = 4 →
  initial_balance + deposit - withdrawal = 76 := by
sorry

end NUMINAMATH_CALUDE_lucy_bank_balance_l3607_360705


namespace NUMINAMATH_CALUDE_minimum_dresses_for_six_colors_one_style_l3607_360764

theorem minimum_dresses_for_six_colors_one_style 
  (num_colors : ℕ) 
  (num_styles : ℕ) 
  (max_extraction_time : ℕ) 
  (h1 : num_colors = 10)
  (h2 : num_styles = 9)
  (h3 : max_extraction_time = 60) :
  ∃ (min_dresses : ℕ),
    (∀ (n : ℕ), n < min_dresses → 
      ¬(∃ (style : ℕ), style < num_styles ∧ 
        (∃ (colors : Finset ℕ), colors.card = 6 ∧ 
          (∀ (c : ℕ), c ∈ colors → c < num_colors) ∧
          (∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2 → 
            ∃ (t1 t2 : ℕ), t1 < max_extraction_time ∧ t2 < max_extraction_time ∧ t1 ≠ t2)))) ∧
    (∃ (style : ℕ), style < num_styles ∧ 
      (∃ (colors : Finset ℕ), colors.card = 6 ∧ 
        (∀ (c : ℕ), c ∈ colors → c < num_colors) ∧
        (∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2 → 
          ∃ (t1 t2 : ℕ), t1 < max_extraction_time ∧ t2 < max_extraction_time ∧ t1 ≠ t2))) ∧
    min_dresses = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_dresses_for_six_colors_one_style_l3607_360764


namespace NUMINAMATH_CALUDE_two_number_difference_l3607_360711

theorem two_number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : 
  |y - x| = 80 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l3607_360711


namespace NUMINAMATH_CALUDE_reading_time_l3607_360737

theorem reading_time (total_pages : ℕ) (first_half_speed second_half_speed : ℕ) : 
  total_pages = 500 → 
  first_half_speed = 10 → 
  second_half_speed = 5 → 
  (total_pages / 2 / first_half_speed + total_pages / 2 / second_half_speed) = 75 := by
sorry

end NUMINAMATH_CALUDE_reading_time_l3607_360737


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l3607_360742

/-- Definition of the sequence x_n -/
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

/-- Theorem stating that no term in the sequence is a perfect square -/
theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℤ, x n = m * m :=
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l3607_360742


namespace NUMINAMATH_CALUDE_lotto_ticket_cost_l3607_360728

/-- Proves that the cost per ticket is $2 given the lottery conditions --/
theorem lotto_ticket_cost (total_tickets : ℕ) (winning_percentage : ℚ)
  (five_dollar_winners_percentage : ℚ) (grand_prize_tickets : ℕ)
  (grand_prize_amount : ℕ) (other_winners_average : ℕ) (total_profit : ℕ) :
  total_tickets = 200 →
  winning_percentage = 1/5 →
  five_dollar_winners_percentage = 4/5 →
  grand_prize_tickets = 1 →
  grand_prize_amount = 5000 →
  other_winners_average = 10 →
  total_profit = 4830 →
  ∃ (cost_per_ticket : ℚ), cost_per_ticket = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_lotto_ticket_cost_l3607_360728


namespace NUMINAMATH_CALUDE_house_population_total_l3607_360797

/-- Represents the number of people on each floor of a three-story house. -/
structure HousePopulation where
  ground : ℕ
  first : ℕ
  second : ℕ

/-- Proves that given the conditions, the total number of people in the house is 60. -/
theorem house_population_total (h : HousePopulation) :
  (h.ground + h.first + h.second = 60) ∧
  (h.first + h.second = 35) ∧
  (h.ground + h.first = 45) ∧
  (h.first = (h.ground + h.first + h.second) / 3) :=
by sorry

end NUMINAMATH_CALUDE_house_population_total_l3607_360797


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3607_360774

theorem quadratic_roots_property : ∀ m n : ℝ, 
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = m ∨ x = n) → 
  m + n - m*n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3607_360774


namespace NUMINAMATH_CALUDE_bob_age_proof_l3607_360760

theorem bob_age_proof :
  ∃! x : ℕ, 
    x > 0 ∧ 
    ∃ y : ℕ, x - 2 = y^2 ∧
    ∃ z : ℕ, x + 2 = z^3 ∧
    x = 123 := by
  sorry

end NUMINAMATH_CALUDE_bob_age_proof_l3607_360760


namespace NUMINAMATH_CALUDE_coin_ratio_l3607_360707

/-- Given the total number of coins, the fraction Amalie spends, and the number of coins
    Amalie has left, prove the ratio of Elsa's coins to Amalie's original coins. -/
theorem coin_ratio (total : ℕ) (amalie_spent_fraction : ℚ) (amalie_left : ℕ)
  (h1 : total = 440)
  (h2 : amalie_spent_fraction = 3/4)
  (h3 : amalie_left = 90) :
  (total - (amalie_left / (1 - amalie_spent_fraction))) / (amalie_left / (1 - amalie_spent_fraction)) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_coin_ratio_l3607_360707


namespace NUMINAMATH_CALUDE_final_sections_count_l3607_360725

/-- Represents the school's student distribution before and after admitting new students -/
structure School where
  initial_students_per_section : ℕ
  final_students_per_section : ℕ
  new_sections : ℕ
  new_students : ℕ

/-- The theorem stating the final number of sections in the school -/
theorem final_sections_count (s : School) 
  (h1 : s.initial_students_per_section = 24)
  (h2 : s.final_students_per_section = 21)
  (h3 : s.new_sections = 3)
  (h4 : s.new_students = 24) :
  ∃ (initial_sections : ℕ), 
    initial_sections * s.initial_students_per_section + s.new_students = 
    (initial_sections + s.new_sections) * s.final_students_per_section ∧
    initial_sections + s.new_sections = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_sections_count_l3607_360725


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3607_360701

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3607_360701


namespace NUMINAMATH_CALUDE_incorrect_exponent_equality_l3607_360757

theorem incorrect_exponent_equality : (-2)^2 ≠ -(2^2) :=
by
  -- Assuming the other equalities are true
  have h1 : 2^0 = 1 := by sorry
  have h2 : (-5)^3 = -(5^3) := by sorry
  have h3 : (-1/2)^3 = -1/8 := by sorry
  
  -- Proof that (-2)^2 ≠ -(2^2)
  sorry

end NUMINAMATH_CALUDE_incorrect_exponent_equality_l3607_360757


namespace NUMINAMATH_CALUDE_graces_coins_worth_l3607_360753

/-- The total worth of Grace's coins in pennies -/
def total_worth (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes + 25 * quarters

/-- Theorem stating that Grace's coins are worth 550 pennies -/
theorem graces_coins_worth : total_worth 25 15 20 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_graces_coins_worth_l3607_360753


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l3607_360743

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = x - y ∧ f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l3607_360743


namespace NUMINAMATH_CALUDE_roots_sum_magnitude_l3607_360762

theorem roots_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ →
  r₁^2 + p*r₁ + 18 = 0 →
  r₂^2 + p*r₂ + 18 = 0 →
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_magnitude_l3607_360762


namespace NUMINAMATH_CALUDE_boys_passed_exam_l3607_360716

theorem boys_passed_exam (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 36 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed : ℕ), passed = 105 ∧ 
    (passed : ℚ) * passed_avg + (total - passed : ℚ) * failed_avg = (total : ℚ) * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_boys_passed_exam_l3607_360716


namespace NUMINAMATH_CALUDE_abc_value_for_specific_factorization_l3607_360783

theorem abc_value_for_specific_factorization (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = (x - 1) * (x - 2)) → a * b * c = -6 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_for_specific_factorization_l3607_360783


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3607_360730

theorem quadratic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3607_360730


namespace NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l3607_360786

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem specific_square_root_squared : (Real.sqrt 529441) ^ 2 = 529441 := by
  apply square_root_squared
  norm_num

end NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l3607_360786


namespace NUMINAMATH_CALUDE_quadratic_two_unequal_real_roots_l3607_360766

theorem quadratic_two_unequal_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (2 * x₁^2 - 3 * x₁ - 4 = 0) ∧ (2 * x₂^2 - 3 * x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_unequal_real_roots_l3607_360766


namespace NUMINAMATH_CALUDE_length_AC_l3607_360733

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 49}

structure PointsOnCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A ∈ Circle
  h_B : B ∈ Circle
  h_AB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64
  h_C : C ∈ Circle
  h_C_midpoint : C.1 = (A.1 + B.1) / 2 ∧ C.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem length_AC (points : PointsOnCircle) :
  (points.A.1 - points.C.1)^2 + (points.A.2 - points.C.2)^2 = 98 - 14 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_length_AC_l3607_360733


namespace NUMINAMATH_CALUDE_quadratic_parabola_properties_l3607_360736

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given quadratic equation and parabola have two distinct real roots and specific form when intersecting x-axis symmetrically -/
theorem quadratic_parabola_properties (m : ℝ) :
  let q : QuadraticEquation := ⟨1, -2*m, m^2 - 4⟩
  let p : Parabola := ⟨1, -2*m, m^2 - 4⟩
  -- The quadratic equation has two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ q.a * x₁^2 + q.b * x₁ + q.c = 0 ∧ q.a * x₂^2 + q.b * x₂ + q.c = 0 ∧
  -- When the parabola intersects x-axis symmetrically, it has the form y = x^2 - 4
  (∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ x₁ = -x₂ ∧ 
   p.a * x₁^2 + p.b * x₁ + p.c = 0 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 0) →
  p = ⟨1, 0, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_parabola_properties_l3607_360736


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3607_360761

/-- Represents a cube with given side length -/
structure Cube where
  side : ℝ
  side_pos : side > 0

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def originalCube : Cube := ⟨4, by norm_num⟩

/-- Represents a corner cube -/
def cornerCube : Cube := ⟨2, by norm_num⟩

/-- The number of corners in a cube -/
def numCorners : ℕ := 8

theorem surface_area_unchanged : 
  surfaceArea originalCube = surfaceArea originalCube - numCorners * (
    3 * cornerCube.side^2 - 3 * cornerCube.side^2
  ) := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l3607_360761


namespace NUMINAMATH_CALUDE_doll_cost_is_15_l3607_360756

/-- Represents the cost of gifts for each sister -/
def gift_cost : ℕ := 60

/-- Represents the number of dolls bought for the younger sister -/
def num_dolls : ℕ := 4

/-- Represents the number of Lego sets bought for the older sister -/
def num_lego_sets : ℕ := 3

/-- Represents the cost of each Lego set -/
def lego_set_cost : ℕ := 20

/-- Theorem stating that the cost of each doll is $15 -/
theorem doll_cost_is_15 : 
  gift_cost = num_lego_sets * lego_set_cost ∧ 
  gift_cost = num_dolls * 15 := by
  sorry

end NUMINAMATH_CALUDE_doll_cost_is_15_l3607_360756


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l3607_360747

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) : 
  Prime p → 0 < k → k < p → p ∣ Nat.choose p k := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l3607_360747


namespace NUMINAMATH_CALUDE_equal_weekend_days_count_l3607_360778

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Checks if starting the month on a given day results in equal Saturdays and Sundays -/
def equalWeekendDays (startDay : DayOfWeek) : Bool :=
  sorry

/-- Counts the number of days that result in equal Saturdays and Sundays when used as the start day -/
def countEqualWeekendDays : Nat :=
  sorry

theorem equal_weekend_days_count :
  countEqualWeekendDays = 2 :=
sorry

end NUMINAMATH_CALUDE_equal_weekend_days_count_l3607_360778


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l3607_360765

/-- Given two workers A and B, their combined work efficiency, and A's individual efficiency,
    prove the ratio of their efficiencies. -/
theorem work_efficiency_ratio 
  (total_days : ℝ) 
  (a_days : ℝ) 
  (h1 : total_days = 12) 
  (h2 : a_days = 16) : 
  (1 / a_days) / ((1 / total_days) - (1 / a_days)) = 3 := by
  sorry

#check work_efficiency_ratio

end NUMINAMATH_CALUDE_work_efficiency_ratio_l3607_360765


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3607_360791

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  (a^3 + b^3 + c^3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3607_360791


namespace NUMINAMATH_CALUDE_equal_discriminants_l3607_360721

theorem equal_discriminants (p1 p2 q1 q2 a1 a2 b1 b2 : ℝ) 
  (hP : ∀ x, x^2 + p1*x + q1 = (x - a1)*(x - a2))
  (hQ : ∀ x, x^2 + p2*x + q2 = (x - b1)*(x - b2))
  (ha : a1 ≠ a2)
  (hb : b1 ≠ b2)
  (h_eq : (b1^2 + p1*b1 + q1) + (b2^2 + p1*b2 + q1) = 
          (a1^2 + p2*a1 + q2) + (a2^2 + p2*a2 + q2)) :
  (a1 - a2)^2 = (b1 - b2)^2 := by
  sorry

end NUMINAMATH_CALUDE_equal_discriminants_l3607_360721


namespace NUMINAMATH_CALUDE_root_implies_m_values_l3607_360793

theorem root_implies_m_values (m : ℝ) : 
  ((m + 2) * 1^2 - 2 * 1 + m^2 - 2*m - 6 = 0) → (m = -2 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_values_l3607_360793


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3607_360775

/-- Two lines ax+2y+1=0 and 3x+(a-1)y+1=0 are parallel if and only if a = -2 -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + 1 = 0 ∧ 3*x + (a-1)*y + 1 = 0) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3607_360775


namespace NUMINAMATH_CALUDE_hannah_strawberries_l3607_360771

theorem hannah_strawberries (x : ℕ) : 
  (30 * x - 20 - 30 = 100) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_hannah_strawberries_l3607_360771


namespace NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l3607_360780

theorem tenth_power_sum_of_roots (u v : ℝ) : 
  u^2 - 2*u*Real.sqrt 3 + 1 = 0 ∧ 
  v^2 - 2*v*Real.sqrt 3 + 1 = 0 → 
  u^10 + v^10 = 93884 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l3607_360780


namespace NUMINAMATH_CALUDE_class_gender_ratio_l3607_360741

/-- Proves that given the boys' average score of 90, girls' average score of 96,
    and overall class average of 94, the ratio of boys to girls in the class is 1:2. -/
theorem class_gender_ratio (B G : ℕ) (B_pos : B > 0) (G_pos : G > 0) : 
  (90 * B + 96 * G) / (B + G) = 94 → B = G / 2 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l3607_360741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3607_360703

def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (a 3 = 5) →
  (∀ n : ℕ, a n = arithmeticSequence (a 1) ((a 3 - a 1) / 2) n) →
  2 * (a 9) - (a 10) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3607_360703


namespace NUMINAMATH_CALUDE_proportional_relationship_l3607_360769

theorem proportional_relationship (k : ℝ) (x z : ℝ → ℝ) :
  (∀ t, x t = k / (z t * Real.sqrt (z t))) →
  x 9 = 8 →
  x 64 = 27 / 64 := by
sorry

end NUMINAMATH_CALUDE_proportional_relationship_l3607_360769


namespace NUMINAMATH_CALUDE_michael_has_52_robots_l3607_360719

/-- The number of flying robots Tom has -/
def tom_robots : ℕ := 12

/-- The ratio of Michael's robots to Tom's robots -/
def michael_to_tom_ratio : ℕ := 4

/-- The number of robots Tom gives away for every group of robots he has -/
def tom_giveaway_ratio : ℕ := 1

/-- The size of the group of robots Tom considers when giving away -/
def tom_group_size : ℕ := 3

/-- Calculates the number of flying robots Michael has in total -/
def michael_total_robots : ℕ :=
  (michael_to_tom_ratio * tom_robots) + (tom_robots / tom_group_size)

/-- Theorem stating that Michael has 52 flying robots in total -/
theorem michael_has_52_robots : michael_total_robots = 52 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_52_robots_l3607_360719


namespace NUMINAMATH_CALUDE_divisibility_property_implies_factor_of_99_l3607_360723

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_property_implies_factor_of_99 (k : ℕ) :
  (∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) →
  99 ∣ k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_implies_factor_of_99_l3607_360723


namespace NUMINAMATH_CALUDE_light_travel_distance_l3607_360796

/-- The distance light travels in one year, in kilometers -/
def light_year : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℕ := 300

/-- The distance light travels in the given number of years -/
def light_distance : ℝ := light_year * years

theorem light_travel_distance : 
  light_distance = 28382 * (10 : ℝ)^13 := by sorry

end NUMINAMATH_CALUDE_light_travel_distance_l3607_360796


namespace NUMINAMATH_CALUDE_num_al_sandwiches_l3607_360717

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 6

/-- Represents whether turkey is available. -/
def has_turkey : Prop := True

/-- Represents whether roast beef is available. -/
def has_roast_beef : Prop := True

/-- Represents whether swiss cheese is available. -/
def has_swiss_cheese : Prop := True

/-- Represents whether rye bread is available. -/
def has_rye_bread : Prop := True

/-- Represents the number of sandwich combinations with turkey and swiss cheese. -/
def turkey_swiss_combos : ℕ := num_bread

/-- Represents the number of sandwich combinations with rye bread and roast beef. -/
def rye_roast_beef_combos : ℕ := num_cheese

/-- Theorem stating the number of different sandwiches Al can order. -/
theorem num_al_sandwiches : 
  num_bread * num_meat * num_cheese - turkey_swiss_combos - rye_roast_beef_combos = 199 := by
  sorry

end NUMINAMATH_CALUDE_num_al_sandwiches_l3607_360717


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3607_360724

-- Define the complex number
def z : ℂ := Complex.I * (1 - Complex.I)

-- Theorem statement
theorem z_in_first_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3607_360724


namespace NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l3607_360745

/-- Given a total number of votes and a loss margin, calculate the percentage of votes received by the losing candidate. -/
def calculate_vote_percentage (total_votes : ℕ) (loss_margin : ℕ) : ℚ :=
  let candidate_votes := (total_votes - loss_margin) / 2
  (candidate_votes : ℚ) / total_votes * 100

/-- Theorem stating that given 7000 total votes and a loss margin of 2100 votes, the losing candidate received 35% of the votes. -/
theorem losing_candidate_vote_percentage :
  calculate_vote_percentage 7000 2100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l3607_360745


namespace NUMINAMATH_CALUDE_max_binomial_probability_l3607_360773

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem max_binomial_probability :
  ∃ (k : ℕ), k ≤ 5 ∧
  ∀ (j : ℕ), j ≤ 5 →
    binomial_probability 5 k (1/4) ≥ binomial_probability 5 j (1/4) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_max_binomial_probability_l3607_360773


namespace NUMINAMATH_CALUDE_triangle_problem_l3607_360720

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Sides a, b, c are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  b * Real.sin (C + π/3) - c * Real.sin B = 0 →
  -- Area condition
  1/2 * a * b * Real.sin C = 10 * Real.sqrt 3 →
  -- D is midpoint of AC
  D.1 = (0 + a * Real.cos B) / 2 ∧ D.2 = (0 + a * Real.sin B) / 2 →
  -- Prove:
  C = π/3 ∧ 
  (∀ (BD : Real), BD^2 ≥ a^2 + b^2/4 - a*b*Real.cos C → BD ≥ 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3607_360720


namespace NUMINAMATH_CALUDE_bisecting_line_sum_of_squares_l3607_360722

/-- A line with slope 4 that bisects a 3x3 unit square into two equal areas -/
def bisecting_line (a b c : ℝ) : Prop :=
  -- The line has slope 4
  a / b = 4 ∧
  -- The line equation is of the form ax = by + c
  ∀ x y, a * x = b * y + c ↔ y = 4 * x ∧
  -- The line bisects the square into two equal areas
  ∃ x₁ y₁ x₂ y₂, 
    0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 ∧
    0 ≤ y₁ ∧ y₁ < y₂ ∧ y₂ ≤ 3 ∧
    a * x₁ = b * y₁ + c ∧
    a * x₂ = b * y₂ + c ∧
    (3 * y₁ + (3 - y₂) * 3) / 2 = 9 / 2

theorem bisecting_line_sum_of_squares (a b c : ℝ) :
  bisecting_line a b c → a^2 + b^2 + c^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_of_squares_l3607_360722


namespace NUMINAMATH_CALUDE_hexagon_extended_point_distance_l3607_360718

/-- Regular hexagon with side length 1 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (is_regular : ∀ (X Y : ℝ × ℝ), (X, Y) ∈ [(A, B), (B, C), (C, D), (D, E), (E, F), (F, A)] → dist X Y = 1)

/-- Point Y extended from A such that BY = 4AB -/
def extend_point (h : RegularHexagon) (Y : ℝ × ℝ) : Prop :=
  dist h.B Y = 4 * dist h.A h.B

/-- The length of segment EY is √21 -/
theorem hexagon_extended_point_distance (h : RegularHexagon) (Y : ℝ × ℝ) 
  (h_extend : extend_point h Y) : 
  dist h.E Y = Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_hexagon_extended_point_distance_l3607_360718


namespace NUMINAMATH_CALUDE_smallest_factorization_constant_l3607_360704

theorem smallest_factorization_constant (b : ℕ) : 
  (∃ (p q : ℤ), (∀ x : ℝ, x^2 + b * x + 2016 = (x + p) * (x + q))) →
  b ≥ 95 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_constant_l3607_360704


namespace NUMINAMATH_CALUDE_louisa_travel_l3607_360712

/-- Louisa's vacation travel problem -/
theorem louisa_travel (average_speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  average_speed = 33.333333333333336 →
  second_day_distance = 350 →
  time_difference = 3 →
  ∃ (first_day_distance : ℝ),
    first_day_distance = average_speed * (second_day_distance / average_speed - time_difference) ∧
    first_day_distance = 250 :=
by sorry

end NUMINAMATH_CALUDE_louisa_travel_l3607_360712


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_l3607_360710

theorem largest_number_from_hcf_lcm (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Nat.gcd a b = 210 →
  Nat.gcd (Nat.gcd a b) c = 210 →
  Nat.lcm (Nat.lcm a b) c = 902910 →
  max a (max b c) = 4830 :=
sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_l3607_360710


namespace NUMINAMATH_CALUDE_stating_chess_team_arrangements_l3607_360763

/-- Represents the number of boys on the chess team -/
def num_boys : Nat := 3

/-- Represents the number of girls on the chess team -/
def num_girls : Nat := 3

/-- Represents the total number of students on the chess team -/
def total_students : Nat := num_boys + num_girls

/-- 
Represents the number of ways to arrange the chess team in a row 
such that all boys are at the ends and exactly one boy is in the middle
-/
def num_arrangements : Nat := 36

/-- 
Theorem stating that the number of arrangements of the chess team
satisfying the given conditions is equal to 36
-/
theorem chess_team_arrangements : 
  (num_boys = 3 ∧ num_girls = 3) → num_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_stating_chess_team_arrangements_l3607_360763


namespace NUMINAMATH_CALUDE_zoo_trip_theorem_l3607_360777

/-- Calculates the remaining money for lunch and snacks after a zoo trip for two people -/
def zoo_trip_remaining_money (ticket_price : ℚ) (bus_fare_one_way : ℚ) (total_money : ℚ) : ℚ :=
  let num_people : ℚ := 2
  let total_ticket_cost := ticket_price * num_people
  let total_bus_fare := bus_fare_one_way * num_people * 2
  let total_trip_cost := total_ticket_cost + total_bus_fare
  total_money - total_trip_cost

theorem zoo_trip_theorem :
  zoo_trip_remaining_money 5 1.5 40 = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_theorem_l3607_360777


namespace NUMINAMATH_CALUDE_functional_equation_proof_l3607_360768

open Real

theorem functional_equation_proof (x : ℝ) (hx : x ≠ 0) :
  let f : ℝ → ℝ := λ x => (x / 3) + (2 / (3 * x))
  2 * f x - f (1 / x) = 1 / x := by sorry

end NUMINAMATH_CALUDE_functional_equation_proof_l3607_360768


namespace NUMINAMATH_CALUDE_number_divided_by_two_equals_number_minus_five_l3607_360772

theorem number_divided_by_two_equals_number_minus_five : ∃! x : ℝ, x / 2 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_two_equals_number_minus_five_l3607_360772


namespace NUMINAMATH_CALUDE_f_max_min_l3607_360740

-- Define the function f(x) = 2x² - x⁴
def f (x : ℝ) : ℝ := 2 * x^2 - x^4

-- Theorem statement
theorem f_max_min :
  (∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, f y ≤ 1) ∧
  (∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_l3607_360740


namespace NUMINAMATH_CALUDE_imaginary_number_condition_l3607_360738

theorem imaginary_number_condition (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 1) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_number_condition_l3607_360738


namespace NUMINAMATH_CALUDE_ace_then_king_probability_l3607_360759

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of Aces in a standard deck -/
def numAces : ℕ := 4

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The probability of drawing an Ace followed by a King from a standard deck -/
theorem ace_then_king_probability :
  (numAces / deckSize) * (numKings / (deckSize - 1)) = 4 / 663 := by
  sorry


end NUMINAMATH_CALUDE_ace_then_king_probability_l3607_360759


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l3607_360731

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem P_sufficient_not_necessary_for_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l3607_360731


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_18_16_l3607_360749

theorem half_abs_diff_squares_18_16 : (1 / 2 : ℝ) * |18^2 - 16^2| = 34 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_18_16_l3607_360749


namespace NUMINAMATH_CALUDE_P_inter_Q_l3607_360754

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem P_inter_Q : P ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_P_inter_Q_l3607_360754


namespace NUMINAMATH_CALUDE_eight_and_half_minutes_in_seconds_l3607_360758

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we're converting -/
def minutes : ℚ := 8.5

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℚ) : ℚ := m * seconds_per_minute

theorem eight_and_half_minutes_in_seconds :
  minutes_to_seconds minutes = 510 := by
  sorry

end NUMINAMATH_CALUDE_eight_and_half_minutes_in_seconds_l3607_360758


namespace NUMINAMATH_CALUDE_max_value_sine_function_l3607_360792

theorem max_value_sine_function (x : ℝ) (h : x ∈ Set.Icc 0 (π/4)) :
  (∃ (max_y : ℝ), max_y = Real.sqrt 3 ∧
    (∀ y : ℝ, y = Real.sqrt 3 * Real.sin (2*x + π/4) → y ≤ max_y) ∧
    max_y = Real.sqrt 3 * Real.sin (2*(π/8) + π/4)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_function_l3607_360792


namespace NUMINAMATH_CALUDE_exam_logic_l3607_360776

structure Student where
  name : String
  score : ℝ
  grade : String

def exam_rule (s : Student) : Prop :=
  s.score ≥ 0.8 → s.grade = "A"

theorem exam_logic (s : Student) (h : exam_rule s) :
  (s.grade ≠ "A" → s.score < 0.8) ∧
  (s.score ≥ 0.8 → s.grade = "A") := by
  sorry

end NUMINAMATH_CALUDE_exam_logic_l3607_360776


namespace NUMINAMATH_CALUDE_parallel_perpendicular_line_coefficient_l3607_360708

/-- Given two lines in the plane, if there exists a third line parallel to one and perpendicular to the other, prove that the coefficient k in the equations must be zero. -/
theorem parallel_perpendicular_line_coefficient (k : ℝ) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), (3 * x - k * y + c = 0) ∧ 
    ((3 * x - k * y + c = 0) ↔ (3 * x - k * y + 6 = 0)) ∧
    ((3 * k + (-k) * 1 = 0) ↔ (k * x + y + 1 = 0))) → 
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_line_coefficient_l3607_360708


namespace NUMINAMATH_CALUDE_problem_solution_l3607_360734

theorem problem_solution (x y : ℚ) : 
  x = 51 → x^3 * y - 3 * x^2 * y + 2 * x * y = 122650 → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3607_360734


namespace NUMINAMATH_CALUDE_interior_angle_sum_l3607_360735

theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) :=
by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_l3607_360735


namespace NUMINAMATH_CALUDE_pi_minus_2023_power_0_minus_one_third_power_neg_2_l3607_360713

theorem pi_minus_2023_power_0_minus_one_third_power_neg_2 :
  (π - 2023) ^ (0 : ℝ) - (1 / 3 : ℝ) ^ (-2 : ℝ) = -8 := by sorry

end NUMINAMATH_CALUDE_pi_minus_2023_power_0_minus_one_third_power_neg_2_l3607_360713


namespace NUMINAMATH_CALUDE_susan_homework_time_l3607_360779

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem susan_homework_time : 
  let homeworkStart : Time := { hours := 13, minutes := 59 }
  let homeworkDuration : Nat := 96
  let practiceStart : Time := { hours := 16, minutes := 0 }
  let homeworkEnd := addMinutes homeworkStart homeworkDuration
  timeDifference homeworkEnd practiceStart = 25 := by
  sorry

end NUMINAMATH_CALUDE_susan_homework_time_l3607_360779


namespace NUMINAMATH_CALUDE_leisure_park_ticket_cost_l3607_360729

/-- The cost of tickets for a family visit to a leisure park -/
theorem leisure_park_ticket_cost :
  ∀ (child_ticket : ℕ),
  child_ticket * 5 + (child_ticket + 8) * 2 + (child_ticket + 4) * 2 = 150 →
  child_ticket + 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_leisure_park_ticket_cost_l3607_360729


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l3607_360727

theorem rulers_in_drawer (initial_rulers : ℕ) (added_rulers : ℕ) : 
  initial_rulers = 46 → added_rulers = 25 → initial_rulers + added_rulers = 71 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l3607_360727


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3607_360798

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℝ
  positive : side > 0

/-- Calculates the surface area of a cube -/
def surfaceArea (c : CubeDimensions) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def originalCube : CubeDimensions := ⟨4, by norm_num⟩

/-- Represents the cube to be removed -/
def removedCube : CubeDimensions := ⟨2, by norm_num⟩

/-- The number of faces of the removed cube that were initially exposed -/
def initiallyExposedFaces : ℕ := 3

theorem surface_area_unchanged :
  surfaceArea originalCube = 
  surfaceArea originalCube - 
  (initiallyExposedFaces : ℝ) * surfaceArea removedCube + 
  (initiallyExposedFaces : ℝ) * surfaceArea removedCube :=
sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l3607_360798


namespace NUMINAMATH_CALUDE_complex_modulus_product_l3607_360785

theorem complex_modulus_product : 
  Complex.abs ((5 * Real.sqrt 3 - 5 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l3607_360785


namespace NUMINAMATH_CALUDE_eggs_per_box_l3607_360790

/-- Given that there are 6 eggs in 2 boxes and each box contains some eggs,
    prove that the number of eggs in each box is 3. -/
theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) (eggs_per_box : ℕ) 
  (h1 : total_eggs = 6)
  (h2 : num_boxes = 2)
  (h3 : eggs_per_box * num_boxes = total_eggs)
  (h4 : eggs_per_box > 0) :
  eggs_per_box = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3607_360790


namespace NUMINAMATH_CALUDE_average_score_is_116_l3607_360748

def mock_exam_scores : List ℕ := [115, 118, 115]

theorem average_score_is_116 : 
  (List.sum mock_exam_scores) / (List.length mock_exam_scores) = 116 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_116_l3607_360748


namespace NUMINAMATH_CALUDE_cone_base_radius_l3607_360750

theorem cone_base_radius (r : ℝ) (θ : ℝ) (base_radius : ℝ) : 
  r = 9 → θ = 240 * π / 180 → base_radius = r * θ / (2 * π) → base_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3607_360750


namespace NUMINAMATH_CALUDE_shortest_path_length_on_cube_l3607_360739

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a path on the surface of a cube -/
structure SurfacePath (c : Cube) where
  length : ℝ
  isOnSurface : Bool

/-- The shortest path on the surface of a cube from the center of one face to the center of the opposite face -/
def shortestPath (c : Cube) : SurfacePath c :=
  sorry

/-- Theorem stating that the shortest path on a cube with edge length 2 has length 3 -/
theorem shortest_path_length_on_cube :
  let c : Cube := { edgeLength := 2 }
  (shortestPath c).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_length_on_cube_l3607_360739


namespace NUMINAMATH_CALUDE_melissas_total_points_l3607_360770

/-- Calculates the total points scored in multiple games -/
def totalPoints (gamesPlayed : ℕ) (pointsPerGame : ℕ) : ℕ :=
  gamesPlayed * pointsPerGame

/-- Proves that Melissa's total points is 81 -/
theorem melissas_total_points :
  let gamesPlayed : ℕ := 3
  let pointsPerGame : ℕ := 27
  totalPoints gamesPlayed pointsPerGame = 81 := by
  sorry

end NUMINAMATH_CALUDE_melissas_total_points_l3607_360770


namespace NUMINAMATH_CALUDE_point_B_coordinate_l3607_360789

/-- Given two points A and B on a number line, where A is 3 units to the left of the origin
    and the distance between A and B is 1, prove that the coordinate of B is either -4 or -2. -/
theorem point_B_coordinate (A B : ℝ) : 
  A = -3 → abs (B - A) = 1 → (B = -4 ∨ B = -2) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinate_l3607_360789


namespace NUMINAMATH_CALUDE_factory_task_excess_l3607_360782

theorem factory_task_excess (first_half : Rat) (second_half : Rat)
  (h1 : first_half = 2 / 3)
  (h2 : second_half = 3 / 5) :
  first_half + second_half - 1 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_factory_task_excess_l3607_360782


namespace NUMINAMATH_CALUDE_perpendicular_planes_condition_l3607_360799

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_condition
  (a b : Line) (α β : Plane)
  (skew : a ≠ b)
  (perp_a_α : perpendicular a α)
  (perp_b_β : perpendicular b β)
  (not_subset_a_β : ¬subset a β)
  (not_subset_b_α : ¬subset b α) :
  perpendicular_planes α β ↔ perpendicular_lines a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_condition_l3607_360799


namespace NUMINAMATH_CALUDE_smallest_geometric_distinct_digits_l3607_360794

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem smallest_geometric_distinct_digits : 
  (∀ n : ℕ, is_three_digit n → 
    digits_are_distinct n → 
    is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) → 
    124 ≤ n) ∧ 
  is_three_digit 124 ∧ 
  digits_are_distinct 124 ∧ 
  is_geometric_sequence (124 / 100) ((124 / 10) % 10) (124 % 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_geometric_distinct_digits_l3607_360794


namespace NUMINAMATH_CALUDE_mary_initial_amount_l3607_360715

def marco_initial : ℕ := 24

theorem mary_initial_amount (mary_initial : ℕ) : mary_initial = 27 :=
  by
  have h1 : mary_initial + marco_initial / 2 > marco_initial / 2 := by sorry
  have h2 : mary_initial - 5 = marco_initial / 2 + 10 := by sorry
  sorry


end NUMINAMATH_CALUDE_mary_initial_amount_l3607_360715


namespace NUMINAMATH_CALUDE_percent_of_360_l3607_360787

theorem percent_of_360 : (35 / 100) * 360 = 126 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_360_l3607_360787


namespace NUMINAMATH_CALUDE_walnut_trees_increase_l3607_360709

/-- Calculates the total number of walnut trees after planting given the initial number and percentage increase -/
def total_trees_after_planting (initial_trees : ℕ) (percent_increase : ℕ) : ℕ :=
  initial_trees + (initial_trees * percent_increase) / 100

/-- Theorem stating that with 22 initial trees and 150% increase, the total after planting is 55 -/
theorem walnut_trees_increase :
  total_trees_after_planting 22 150 = 55 := by
  sorry

#eval total_trees_after_planting 22 150

end NUMINAMATH_CALUDE_walnut_trees_increase_l3607_360709


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_450_l3607_360726

/-- Represents a quadrilateral ABCD with a diagonal BD and offsets AE and CF -/
structure Quadrilateral where
  bd : ℝ  -- Length of diagonal BD
  ae : ℝ  -- Length of offset AE
  cf : ℝ  -- Length of offset CF
  abd_angle : ℝ  -- Angle ABD in degrees

/-- Calculate the area of the quadrilateral ABCD -/
def area (q : Quadrilateral) : ℝ :=
  0.5 * q.bd * q.ae + 0.5 * q.bd * q.cf

/-- Theorem stating that the area of the given quadrilateral is 450 -/
theorem quadrilateral_area_is_450 (q : Quadrilateral)
    (h1 : q.bd = 50)
    (h2 : q.ae = 10)
    (h3 : q.cf = 8)
    (h4 : 0 < q.abd_angle ∧ q.abd_angle < 180) :
    area q = 450 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_450_l3607_360726


namespace NUMINAMATH_CALUDE_computer_music_time_l3607_360781

def total_time : ℕ := 120
def piano_time : ℕ := 30
def reading_time : ℕ := 38
def exerciser_time : ℕ := 27

theorem computer_music_time : 
  total_time - (piano_time + reading_time + exerciser_time) = 25 := by
sorry

end NUMINAMATH_CALUDE_computer_music_time_l3607_360781


namespace NUMINAMATH_CALUDE_equal_pieces_after_exchanges_l3607_360706

theorem equal_pieces_after_exchanges (initial_white : ℕ) (initial_black : ℕ) 
  (exchange_count : ℕ) (pieces_per_exchange : ℕ) :
  initial_white = 80 →
  initial_black = 50 →
  pieces_per_exchange = 3 →
  exchange_count = 5 →
  initial_white - exchange_count * pieces_per_exchange = 
  initial_black + exchange_count * pieces_per_exchange :=
by
  sorry

#check equal_pieces_after_exchanges

end NUMINAMATH_CALUDE_equal_pieces_after_exchanges_l3607_360706


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_water_bottle_l3607_360788

theorem min_blue_eyes_and_water_bottle 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (water_bottle : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 18) 
  (h3 : water_bottle = 25) : 
  ∃ (both : ℕ), both ≥ 8 ∧ 
    both ≤ blue_eyes ∧ 
    both ≤ water_bottle ∧ 
    (∀ (x : ℕ), x < both → 
      x > blue_eyes - (total_students - water_bottle) ∨ 
      x > water_bottle - (total_students - blue_eyes)) :=
by sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_water_bottle_l3607_360788


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l3607_360752

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1233 :=
by sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l3607_360752


namespace NUMINAMATH_CALUDE_boys_without_notebooks_l3607_360767

/-- Proves the number of boys without notebooks in Ms. Johnson's class -/
theorem boys_without_notebooks
  (total_boys : ℕ)
  (students_with_notebooks : ℕ)
  (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24)
  (h2 : students_with_notebooks = 30)
  (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_notebooks_l3607_360767


namespace NUMINAMATH_CALUDE_vegetable_garden_theorem_l3607_360751

def vegetable_garden_total (potatoes cucumbers tomatoes peppers carrots : ℕ) : Prop :=
  potatoes = 1200 ∧
  cucumbers = potatoes - 160 ∧
  tomatoes = 4 * cucumbers ∧
  peppers * peppers = cucumbers * tomatoes ∧
  carrots = (cucumbers + tomatoes) + (cucumbers + tomatoes) / 5 ∧
  potatoes + cucumbers + tomatoes + peppers + carrots = 14720

theorem vegetable_garden_theorem :
  ∃ (potatoes cucumbers tomatoes peppers carrots : ℕ),
    vegetable_garden_total potatoes cucumbers tomatoes peppers carrots :=
by
  sorry

end NUMINAMATH_CALUDE_vegetable_garden_theorem_l3607_360751
