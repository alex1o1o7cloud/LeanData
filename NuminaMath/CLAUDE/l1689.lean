import Mathlib

namespace sum_three_numbers_l1689_168949

theorem sum_three_numbers (a b c N : ℚ) : 
  a + b + c = 84 ∧ 
  a - 7 = N ∧ 
  b + 7 = N ∧ 
  c / 7 = N → 
  N = 28 / 3 := by
sorry

end sum_three_numbers_l1689_168949


namespace circle_reflection_translation_l1689_168934

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -4) →
  (translate_right (reflect_x center) 5) = (8, 4) := by
  sorry

end circle_reflection_translation_l1689_168934


namespace fraction_simplification_l1689_168970

variable (x : ℝ)

theorem fraction_simplification :
  (x^3 + 4*x^2 + 7*x + 4) / (x^3 + 2*x^2 + x - 4) = (x + 1) / (x - 1) ∧
  2 * (24*x^3 + 46*x^2 + 33*x + 9) / (24*x^3 + 10*x^2 - 9*x - 9) = (4*x + 3) / (4*x - 3) :=
by
  sorry

end fraction_simplification_l1689_168970


namespace peggy_final_doll_count_l1689_168974

/-- Calculates the final number of dolls Peggy has -/
def peggy_dolls (initial : ℕ) (grandmother_gift : ℕ) : ℕ :=
  initial + grandmother_gift + (grandmother_gift / 2)

/-- Theorem stating that Peggy's final doll count is 51 -/
theorem peggy_final_doll_count :
  peggy_dolls 6 30 = 51 := by
  sorry

end peggy_final_doll_count_l1689_168974


namespace horse_purchase_problem_l1689_168995

theorem horse_purchase_problem (cuirassier_total : ℝ) (dragoon_total : ℝ) (dragoon_extra : ℕ) (price_diff : ℝ) :
  cuirassier_total = 11250 ∧ 
  dragoon_total = 16000 ∧ 
  dragoon_extra = 15 ∧ 
  price_diff = 50 →
  ∃ (cuirassier_count dragoon_count : ℕ) (cuirassier_price dragoon_price : ℝ),
    cuirassier_count = 25 ∧
    dragoon_count = 40 ∧
    cuirassier_price = 450 ∧
    dragoon_price = 400 ∧
    cuirassier_count * cuirassier_price = cuirassier_total ∧
    dragoon_count * dragoon_price = dragoon_total ∧
    dragoon_count = cuirassier_count + dragoon_extra ∧
    cuirassier_price = dragoon_price + price_diff :=
by sorry

end horse_purchase_problem_l1689_168995


namespace math_club_team_selection_l1689_168931

theorem math_club_team_selection (total_boys : ℕ) (total_girls : ℕ) 
  (team_boys : ℕ) (team_girls : ℕ) : 
  total_boys = 7 → 
  total_girls = 9 → 
  team_boys = 4 → 
  team_girls = 3 → 
  (team_boys + team_girls : ℕ) = 7 → 
  (Nat.choose total_boys team_boys) * (Nat.choose total_girls team_girls) = 2940 := by
  sorry

end math_club_team_selection_l1689_168931


namespace number_times_power_of_five_l1689_168952

theorem number_times_power_of_five (x : ℝ) : x * (5^4) = 75625 → x = 121 := by
  sorry

end number_times_power_of_five_l1689_168952


namespace fraction_meaningful_iff_not_equal_two_l1689_168924

theorem fraction_meaningful_iff_not_equal_two (x : ℝ) : 
  (∃ y : ℝ, y = 7 / (x - 2)) ↔ x ≠ 2 := by
sorry

end fraction_meaningful_iff_not_equal_two_l1689_168924


namespace contact_lenses_sales_l1689_168904

theorem contact_lenses_sales (soft_price hard_price : ℕ) 
  (soft_hard_difference total_sales : ℕ) : 
  soft_price = 150 →
  hard_price = 85 →
  soft_hard_difference = 5 →
  total_sales = 1455 →
  ∃ (soft hard : ℕ), 
    soft = hard + soft_hard_difference ∧
    soft_price * soft + hard_price * hard = total_sales ∧
    soft + hard = 11 :=
by sorry

end contact_lenses_sales_l1689_168904


namespace apartment_number_exists_unique_l1689_168958

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def contains_digit (n d : ℕ) : Prop :=
  ∃ k m, n = 100 * k + 10 * m + d ∧ 0 ≤ k ∧ k < 10 ∧ 0 ≤ m ∧ m < 10

theorem apartment_number_exists_unique :
  ∃! n : ℕ, is_three_digit n ∧
            n % 11 = 0 ∧
            n % 2 = 0 ∧
            n % 5 = 0 ∧
            ¬ contains_digit n 7 :=
sorry

end apartment_number_exists_unique_l1689_168958


namespace m_range_l1689_168987

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < -2 ∨ x > 10

def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0

-- Define the condition that ¬q is sufficient but not necessary for ¬p
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(q x m) → ¬(p x)) ∧ ∃ x, ¬(p x) ∧ q x m

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) ∧ (sufficient_not_necessary m) ↔ 0 < m ∧ m ≤ 3 :=
sorry

end m_range_l1689_168987


namespace bernoulli_inequality_l1689_168985

theorem bernoulli_inequality (x : ℝ) (n : ℕ+) (h1 : x ≠ 0) (h2 : x > -1) :
  (1 + x)^(n : ℝ) ≥ n * x := by
  sorry

end bernoulli_inequality_l1689_168985


namespace f_is_quadratic_l1689_168928

/-- A quadratic equation in terms of x is a polynomial equation of degree 2 in x. -/
def IsQuadraticInX (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation x(x-1) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 1)

/-- Theorem stating that f is a quadratic equation in terms of x -/
theorem f_is_quadratic : IsQuadraticInX f := by sorry

end f_is_quadratic_l1689_168928


namespace tim_cell_phone_cost_l1689_168910

/-- Calculates the total cost of a cell phone plan -/
def calculate_total_cost (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
                         (free_hours : ℝ) (texts_sent : ℝ) (hours_talked : ℝ) : ℝ :=
  let text_total := text_cost * texts_sent
  let extra_minutes := (hours_talked - free_hours) * 60
  let extra_minute_total := extra_minute_cost * extra_minutes
  base_cost + text_total + extra_minute_total

theorem tim_cell_phone_cost :
  let base_cost : ℝ := 30
  let text_cost : ℝ := 0.04
  let extra_minute_cost : ℝ := 0.15
  let free_hours : ℝ := 40
  let texts_sent : ℝ := 200
  let hours_talked : ℝ := 42
  calculate_total_cost base_cost text_cost extra_minute_cost free_hours texts_sent hours_talked = 56 := by
  sorry


end tim_cell_phone_cost_l1689_168910


namespace shaded_cubes_count_l1689_168903

/-- Represents a cube made up of smaller cubes --/
structure Cube where
  size : Nat
  shaded_corners : Bool
  shaded_center : Bool

/-- Counts the number of smaller cubes with at least one face shaded --/
def count_shaded_cubes (c : Cube) : Nat :=
  sorry

/-- Theorem stating that a 4x4x4 cube with shaded corners and centers has 14 shaded cubes --/
theorem shaded_cubes_count (c : Cube) :
  c.size = 4 ∧ c.shaded_corners ∧ c.shaded_center →
  count_shaded_cubes c = 14 :=
by sorry

end shaded_cubes_count_l1689_168903


namespace discriminant_of_5x2_plus_3x_minus_8_l1689_168930

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Proof that the discriminant of 5x² + 3x - 8 is 169 -/
theorem discriminant_of_5x2_plus_3x_minus_8 :
  discriminant 5 3 (-8) = 169 := by
  sorry

end discriminant_of_5x2_plus_3x_minus_8_l1689_168930


namespace car_travel_time_l1689_168913

/-- Proves that a car with given specifications traveling for a certain time uses the specified amount of fuel -/
theorem car_travel_time (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (fuel_used_ratio : ℝ) : 
  speed = 40 →
  fuel_efficiency = 40 →
  tank_capacity = 12 →
  fuel_used_ratio = 0.4166666666666667 →
  (fuel_used_ratio * tank_capacity * fuel_efficiency) / speed = 5 := by
  sorry

end car_travel_time_l1689_168913


namespace restaurant_bill_total_l1689_168996

theorem restaurant_bill_total (num_people : ℕ) (individual_payment : ℚ) (total_bill : ℚ) : 
  num_people = 9 → 
  individual_payment = 514.19 → 
  total_bill = num_people * individual_payment → 
  total_bill = 4627.71 := by
sorry

end restaurant_bill_total_l1689_168996


namespace integer_root_values_l1689_168951

theorem integer_root_values (b : ℤ) : 
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ b ∈ ({-21, 19, -17, -4, 3} : Set ℤ) := by
  sorry

end integer_root_values_l1689_168951


namespace sqrt_expression_simplification_l1689_168998

theorem sqrt_expression_simplification :
  (Real.sqrt 7 - 1)^2 - (Real.sqrt 14 - Real.sqrt 2) * (Real.sqrt 14 + Real.sqrt 2) = -4 - 2 * Real.sqrt 7 :=
by sorry

end sqrt_expression_simplification_l1689_168998


namespace apple_bags_sum_l1689_168920

theorem apple_bags_sum : 
  let golden_delicious : ℚ := 17/100
  let macintosh : ℚ := 17/100
  let cortland : ℚ := 33/100
  golden_delicious + macintosh + cortland = 67/100 := by
  sorry

end apple_bags_sum_l1689_168920


namespace extended_fishing_rod_length_l1689_168984

theorem extended_fishing_rod_length 
  (original_length : ℝ) 
  (increase_factor : ℝ) 
  (extended_length : ℝ) : 
  original_length = 48 → 
  increase_factor = 1.33 → 
  extended_length = original_length * increase_factor → 
  extended_length = 63.84 :=
by sorry

end extended_fishing_rod_length_l1689_168984


namespace rectangle_length_l1689_168991

/-- Represents a rectangle with length, width, diagonal, and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  diagonal : ℝ
  perimeter : ℝ

/-- Theorem: A rectangle with diagonal 17 cm and perimeter 46 cm has a length of 15 cm -/
theorem rectangle_length (r : Rectangle) 
  (h_diagonal : r.diagonal = 17)
  (h_perimeter : r.perimeter = 46)
  (h_perimeter_def : r.perimeter = 2 * (r.length + r.width))
  (h_diagonal_def : r.diagonal ^ 2 = r.length ^ 2 + r.width ^ 2) :
  r.length = 15 := by
  sorry


end rectangle_length_l1689_168991


namespace one_in_set_zero_one_l1689_168935

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by
  sorry

end one_in_set_zero_one_l1689_168935


namespace baking_contest_votes_l1689_168944

/-- The number of votes for the witch cake -/
def witch_votes : ℕ := 7

/-- The number of votes for the unicorn cake -/
def unicorn_votes : ℕ := 3 * witch_votes

/-- The number of votes for the dragon cake -/
def dragon_votes : ℕ := witch_votes + 25

/-- The total number of votes cast -/
def total_votes : ℕ := witch_votes + unicorn_votes + dragon_votes

theorem baking_contest_votes : total_votes = 60 := by
  sorry

end baking_contest_votes_l1689_168944


namespace system_solution_l1689_168953

theorem system_solution : 
  ∀ x y : ℝ, (y^2 + x*y = 15 ∧ x^2 + x*y = 10) ↔ ((x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3)) := by
  sorry

end system_solution_l1689_168953


namespace contrapositive_square_inequality_l1689_168978

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x^2 > y^2) → ¬(x > y)) ↔ (x^2 ≤ y^2 → x ≤ y) := by sorry

end contrapositive_square_inequality_l1689_168978


namespace maxwell_brad_meeting_time_l1689_168983

/-- Proves that Maxwell walks for 2 hours before meeting Brad -/
theorem maxwell_brad_meeting_time
  (distance : ℝ)
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (head_start : ℝ)
  (h_distance : distance = 14)
  (h_maxwell_speed : maxwell_speed = 4)
  (h_brad_speed : brad_speed = 6)
  (h_head_start : head_start = 1) :
  ∃ (t : ℝ), t + head_start = 2 ∧ maxwell_speed * (t + head_start) + brad_speed * t = distance :=
by sorry

end maxwell_brad_meeting_time_l1689_168983


namespace square_root_of_sixteen_l1689_168914

theorem square_root_of_sixteen (x : ℝ) : x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end square_root_of_sixteen_l1689_168914


namespace power_of_25_equals_power_of_5_l1689_168927

theorem power_of_25_equals_power_of_5 : (25 : ℕ) ^ 5 = 5 ^ 10 := by
  sorry

end power_of_25_equals_power_of_5_l1689_168927


namespace nancy_shoe_multiple_l1689_168954

/-- Given Nancy's shoe collection, prove the multiple relating heels to boots and slippers --/
theorem nancy_shoe_multiple :
  ∀ (boots slippers heels : ℕ),
  boots = 6 →
  slippers = boots + 9 →
  2 * (boots + slippers + heels) = 168 →
  ∃ (m : ℕ), heels = m * (boots + slippers) ∧ m = 3 :=
by sorry

end nancy_shoe_multiple_l1689_168954


namespace total_cost_of_shirts_proof_total_cost_of_shirts_l1689_168950

/-- The total cost of two shirts, where the first shirt costs $6 more than the second,
    and the first shirt costs $15, is $24. -/
theorem total_cost_of_shirts : ℕ → Prop :=
  fun n => n = 24 ∧ ∃ (cost1 cost2 : ℕ),
    cost1 = 15 ∧
    cost1 = cost2 + 6 ∧
    n = cost1 + cost2

/-- Proof of the theorem -/
theorem proof_total_cost_of_shirts : total_cost_of_shirts 24 := by
  sorry

end total_cost_of_shirts_proof_total_cost_of_shirts_l1689_168950


namespace monkeys_required_for_new_bananas_l1689_168967

/-- Represents the number of monkeys eating bananas -/
def num_monkeys : ℕ := 5

/-- Represents the number of bananas eaten in the initial scenario -/
def initial_bananas : ℕ := 5

/-- Represents the time taken to eat the initial number of bananas -/
def initial_time : ℕ := 5

/-- Represents the number of bananas to be eaten in the new scenario -/
def new_bananas : ℕ := 15

/-- Theorem stating that the number of monkeys required to eat the new number of bananas
    is equal to the initial number of monkeys -/
theorem monkeys_required_for_new_bananas :
  (num_monkeys : ℕ) = (num_monkeys : ℕ) := by sorry

end monkeys_required_for_new_bananas_l1689_168967


namespace production_days_calculation_l1689_168941

/-- Proves that given the conditions of the production problem, n must equal 9 -/
theorem production_days_calculation (n : ℕ) 
  (h1 : (n : ℝ) * 50 / n = 50)  -- Average for n days is 50
  (h2 : ((n : ℝ) * 50 + 90) / (n + 1) = 54)  -- New average for n+1 days is 54
  : n = 9 := by
  sorry

end production_days_calculation_l1689_168941


namespace min_mozart_and_bach_not_beethoven_l1689_168956

def total_surveyed : ℕ := 150
def mozart_fans : ℕ := 120
def bach_fans : ℕ := 105
def beethoven_fans : ℕ := 45

theorem min_mozart_and_bach_not_beethoven :
  ∃ (mozart_set bach_set beethoven_set : Finset (Fin total_surveyed)),
    mozart_set.card = mozart_fans ∧
    bach_set.card = bach_fans ∧
    beethoven_set.card = beethoven_fans ∧
    ((mozart_set ∩ bach_set) \ beethoven_set).card ≥ 75 ∧
    ∀ (m b be : Finset (Fin total_surveyed)),
      m.card = mozart_fans →
      b.card = bach_fans →
      be.card = beethoven_fans →
      ((m ∩ b) \ be).card ≥ 75 :=
by sorry

end min_mozart_and_bach_not_beethoven_l1689_168956


namespace dhoni_leftover_percentage_l1689_168982

/-- Represents the percentage of Dhoni's earnings spent on rent -/
def rent_percentage : ℝ := 20

/-- Represents the difference in percentage between rent and dishwasher expenses -/
def dishwasher_difference : ℝ := 5

/-- Calculates the percentage of earnings spent on the dishwasher -/
def dishwasher_percentage : ℝ := rent_percentage - dishwasher_difference

/-- Calculates the total percentage of earnings spent -/
def total_spent_percentage : ℝ := rent_percentage + dishwasher_percentage

/-- Represents the total percentage (100%) -/
def total_percentage : ℝ := 100

/-- Theorem: The percentage of Dhoni's earning left over is 65% -/
theorem dhoni_leftover_percentage : 
  total_percentage - total_spent_percentage = 65 := by
  sorry

end dhoni_leftover_percentage_l1689_168982


namespace max_checkers_8x8_l1689_168960

/-- Represents a chess board -/
structure Board :=
  (size : Nat)

/-- Represents a configuration of checkers on a board -/
structure CheckerConfiguration :=
  (board : Board)
  (numCheckers : Nat)

/-- Predicate to check if a configuration is valid (all checkers under attack) -/
def isValidConfiguration (config : CheckerConfiguration) : Prop := sorry

/-- The maximum number of checkers that can be placed on a board -/
def maxCheckers (b : Board) : Nat := sorry

/-- Theorem stating the maximum number of checkers on an 8x8 board -/
theorem max_checkers_8x8 :
  ∃ (config : CheckerConfiguration),
    config.board.size = 8 ∧
    isValidConfiguration config ∧
    config.numCheckers = maxCheckers config.board ∧
    config.numCheckers = 32 :=
  sorry

end max_checkers_8x8_l1689_168960


namespace school_total_students_l1689_168957

/-- The total number of students in a school with a given number of grades and students per grade -/
def total_students (num_grades : ℕ) (students_per_grade : ℕ) : ℕ :=
  num_grades * students_per_grade

/-- Theorem stating that the total number of students in a school with 304 grades and 75 students per grade is 22800 -/
theorem school_total_students :
  total_students 304 75 = 22800 := by
  sorry

end school_total_students_l1689_168957


namespace geometric_sequence_property_l1689_168932

/-- Given a geometric sequence {a_n} where the 5th term is equal to the constant term
    in the expansion of (x + 1/x)^4, prove that a_3 * a_7 = 36 -/
theorem geometric_sequence_property (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 5 = 6) →  -- 5th term is equal to the constant term in (x + 1/x)^4
  a 3 * a 7 = 36 := by
sorry

end geometric_sequence_property_l1689_168932


namespace projective_transformation_uniqueness_l1689_168946

/-- A projective transformation on a straight line -/
structure ProjectiveTransformation :=
  (f : ℝ → ℝ)

/-- The property that a projective transformation preserves cross-ratio -/
def PreservesCrossRatio (t : ProjectiveTransformation) : Prop :=
  ∀ a b c d : ℝ, (a - c) * (b - d) / ((b - c) * (a - d)) = 
    (t.f a - t.f c) * (t.f b - t.f d) / ((t.f b - t.f c) * (t.f a - t.f d))

/-- Two projective transformations are equal if they agree on three distinct points -/
theorem projective_transformation_uniqueness 
  (t₁ t₂ : ProjectiveTransformation)
  (h₁ : PreservesCrossRatio t₁)
  (h₂ : PreservesCrossRatio t₂)
  (a b c : ℝ)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (eq_a : t₁.f a = t₂.f a)
  (eq_b : t₁.f b = t₂.f b)
  (eq_c : t₁.f c = t₂.f c) :
  ∀ x : ℝ, t₁.f x = t₂.f x :=
sorry

end projective_transformation_uniqueness_l1689_168946


namespace solution_characterization_l1689_168919

/-- Represents a 3-digit integer abc --/
structure ThreeDigitInt where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a < 10
  h3 : b < 10
  h4 : c < 10

/-- Converts a ThreeDigitInt to its numerical value --/
def toInt (n : ThreeDigitInt) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Checks if a ThreeDigitInt satisfies the given equation --/
def satisfiesEquation (n : ThreeDigitInt) : Prop :=
  n.b * (10 * n.a + n.c) = n.c * (10 * n.a + n.b) + 10

/-- The set of all ThreeDigitInt that satisfy the equation --/
def solutionSet : Set ThreeDigitInt :=
  {n : ThreeDigitInt | satisfiesEquation n}

/-- The theorem to be proved --/
theorem solution_characterization :
  solutionSet = {
    ⟨1, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 2, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 3, 4, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 4, 5, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 5, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 6, 7, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 7, 8, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 8, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩
  } := by sorry

#eval toInt ⟨1, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 2, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 3, 4, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 4, 5, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 5, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 6, 7, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 7, 8, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 8, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩

end solution_characterization_l1689_168919


namespace simplify_fraction_product_l1689_168965

theorem simplify_fraction_product : 8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end simplify_fraction_product_l1689_168965


namespace intersection_probability_correct_l1689_168918

/-- Given a positive integer n, this function calculates the probability that
    the intersection of two randomly selected non-empty subsets from {1, 2, ..., n}
    is not empty. -/
def intersection_probability (n : ℕ+) : ℚ :=
  (4^n.val - 3^n.val : ℚ) / (2^n.val - 1)^2

/-- Theorem stating that the probability of non-empty intersection of two randomly
    selected non-empty subsets from {1, 2, ..., n} is given by the function
    intersection_probability. -/
theorem intersection_probability_correct (n : ℕ+) :
  intersection_probability n =
    (4^n.val - 3^n.val : ℚ) / (2^n.val - 1)^2 := by
  sorry

end intersection_probability_correct_l1689_168918


namespace gcd_2023_2052_l1689_168961

theorem gcd_2023_2052 : Nat.gcd 2023 2052 = 1 := by
  sorry

end gcd_2023_2052_l1689_168961


namespace salary_calculation_l1689_168908

theorem salary_calculation (S : ℚ) 
  (food_expense : S / 5 = S * (1 / 5))
  (rent_expense : S / 10 = S * (1 / 10))
  (clothes_expense : S * 3 / 5 = S * (3 / 5))
  (remaining : S - (S / 5) - (S / 10) - (S * 3 / 5) = 18000) :
  S = 180000 := by
sorry

end salary_calculation_l1689_168908


namespace parallel_lines_perpendicular_lines_l1689_168980

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) ↔ a = -1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * 1 + 2 * a = 0) ↔ a = 1/3 :=
sorry

end parallel_lines_perpendicular_lines_l1689_168980


namespace solution_set_of_inequality_range_of_a_l1689_168979

-- Define the function f(x) for part (1)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the function f(x) for part (2) with parameter a
def f_with_a (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part (1)
theorem solution_set_of_inequality (x : ℝ) :
  f x ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 := by sorry

-- Part (2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_with_a a x ≥ 2) → (a = -1 ∨ a = 3) := by sorry

end solution_set_of_inequality_range_of_a_l1689_168979


namespace hexagonal_field_fencing_cost_l1689_168907

/-- Represents the cost of fencing for a single side of the hexagonal field -/
structure SideCost where
  length : ℝ
  costPerMeter : ℝ

/-- Calculates the total cost of fencing for an irregular hexagonal field -/
def totalFencingCost (sides : List SideCost) : ℝ :=
  sides.foldl (fun acc side => acc + side.length * side.costPerMeter) 0

/-- Theorem stating that the total cost of fencing for the given hexagonal field is 289 rs. -/
theorem hexagonal_field_fencing_cost :
  let sides : List SideCost := [
    ⟨10, 3⟩, ⟨20, 2⟩, ⟨15, 4⟩, ⟨18, 3.5⟩, ⟨12, 2.5⟩, ⟨22, 3⟩
  ]
  totalFencingCost sides = 289 := by
  sorry


end hexagonal_field_fencing_cost_l1689_168907


namespace circle_triangle_area_constraint_l1689_168993

/-- The range of r for which there are exactly two points on the circle
    (x-2)^2 + y^2 = r^2 that form triangles with area 4 with given points A and B -/
theorem circle_triangle_area_constraint (r : ℝ) : 
  r > 0 →
  (∃! M N : ℝ × ℝ, 
    (M.1 - 2)^2 + M.2^2 = r^2 ∧
    (N.1 - 2)^2 + N.2^2 = r^2 ∧
    abs ((M.1 + 3) * (-2) - (M.2 - 0) * (-2)) / 2 = 4 ∧
    abs ((N.1 + 3) * (-2) - (N.2 - 0) * (-2)) / 2 = 4) →
  r ∈ Set.Ioo (Real.sqrt 2 / 2) (9 * Real.sqrt 2 / 2) :=
by sorry

end circle_triangle_area_constraint_l1689_168993


namespace divisor_function_ratio_l1689_168966

/-- τ(n) denotes the number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := sorry

theorem divisor_function_ratio (n : ℕ+) (h : τ (n^2) / τ n = 3) : 
  τ (n^7) / τ n = 29 := by sorry

end divisor_function_ratio_l1689_168966


namespace unique_digit_product_l1689_168945

theorem unique_digit_product : ∃! (x y z : ℕ), 
  (x < 10 ∧ y < 10 ∧ z < 10) ∧ 
  (10 ≤ 10 * x + y) ∧ (10 * x + y ≤ 99) ∧
  (1 ≤ z) ∧
  (x * (10 * x + y) = 111 * z) := by
sorry

end unique_digit_product_l1689_168945


namespace linear_equation_solution_l1689_168968

theorem linear_equation_solution (x y : ℝ) :
  4 * x - 5 * y = 9 → y = (4 * x - 9) / 5 := by
  sorry

end linear_equation_solution_l1689_168968


namespace equal_area_rectangles_width_l1689_168929

/-- Given two rectangles of equal area, where one rectangle has dimensions 5 inches by 24 inches,
    and the other rectangle has a length of 3 inches, prove that the width of the second rectangle
    is 40 inches. -/
theorem equal_area_rectangles_width (area : ℝ) (width : ℝ) :
  area = 5 * 24 →  -- Carol's rectangle area
  area = 3 * width →  -- Jordan's rectangle area
  width = 40 := by
  sorry

end equal_area_rectangles_width_l1689_168929


namespace not_octal_7857_l1689_168900

def is_octal_digit (d : Nat) : Prop := d ≤ 7

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem not_octal_7857 : ¬ is_octal_number 7857 := by
  sorry

end not_octal_7857_l1689_168900


namespace dick_jane_age_problem_l1689_168999

theorem dick_jane_age_problem :
  ∃ (d n : ℕ), 
    d > 27 ∧
    10 ≤ 27 + n ∧ 27 + n ≤ 99 ∧
    10 ≤ d + n ∧ d + n ≤ 99 ∧
    ∃ (a b : ℕ), 
      27 + n = 10 * a + b ∧
      d + n = 10 * b + a ∧
      Nat.Prime (a + b) ∧
      1 ≤ a ∧ a < b ∧ b ≤ 9 :=
by sorry

end dick_jane_age_problem_l1689_168999


namespace infected_and_positive_probability_l1689_168981

/-- The infection rate of the novel coronavirus -/
def infection_rate : ℝ := 0.005

/-- The probability of testing positive given infection -/
def positive_given_infection : ℝ := 0.99

/-- The probability that a citizen is infected and tests positive -/
def infected_and_positive : ℝ := infection_rate * positive_given_infection

theorem infected_and_positive_probability :
  infected_and_positive = 0.00495 := by sorry

end infected_and_positive_probability_l1689_168981


namespace pen_and_pencil_cost_l1689_168975

theorem pen_and_pencil_cost (pencil_cost : ℝ) (pen_cost : ℝ) : 
  pencil_cost = 8 → pen_cost = pencil_cost / 2 → pencil_cost + pen_cost = 12 := by
  sorry

end pen_and_pencil_cost_l1689_168975


namespace sets_equality_independent_of_order_l1689_168972

theorem sets_equality_independent_of_order (A B : Set ℕ) : 
  (∀ x, x ∈ A ↔ x ∈ B) → A = B :=
by sorry

end sets_equality_independent_of_order_l1689_168972


namespace stock_price_theorem_l1689_168915

def stock_price_evolution (initial_price : ℝ) (year1_change : ℝ) (year2_change : ℝ) (year3_change : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_change)
  let price_after_year2 := price_after_year1 * (1 + year2_change)
  let price_after_year3 := price_after_year2 * (1 + year3_change)
  price_after_year3

theorem stock_price_theorem :
  stock_price_evolution 150 0.5 (-0.3) 0.2 = 189 := by
  sorry

end stock_price_theorem_l1689_168915


namespace ellipse_equation_l1689_168959

/-- An ellipse with one focus at (1,0) and eccentricity 1/2 has the standard equation x²/4 + y²/3 = 1 -/
theorem ellipse_equation (F : ℝ × ℝ) (e : ℝ) (h1 : F = (1, 0)) (h2 : e = 1/2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

end ellipse_equation_l1689_168959


namespace taxi_distance_is_ten_miles_l1689_168989

/-- Calculates the taxi fare distance given the total fare, initial fare, initial distance, and additional fare per unit distance -/
def taxi_fare_distance (total_fare : ℚ) (initial_fare : ℚ) (initial_distance : ℚ) (additional_fare_per_unit : ℚ) : ℚ :=
  initial_distance + (total_fare - initial_fare) / additional_fare_per_unit

/-- Theorem: Given the specified fare structure and total fare, the distance traveled is 10 miles -/
theorem taxi_distance_is_ten_miles :
  let total_fare : ℚ := 59
  let initial_fare : ℚ := 10
  let initial_distance : ℚ := 1/5
  let additional_fare_per_unit : ℚ := 1/(1/5)
  taxi_fare_distance total_fare initial_fare initial_distance additional_fare_per_unit = 10 := by
  sorry

#eval taxi_fare_distance 59 10 (1/5) 5

end taxi_distance_is_ten_miles_l1689_168989


namespace solve_a_l1689_168943

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a : ∃ (a : ℝ), star a 5 = 9 ∧ a = 17 := by sorry

end solve_a_l1689_168943


namespace smiths_bakery_pies_l1689_168906

theorem smiths_bakery_pies (mcgees_pies : ℕ) (smiths_pies : ℕ) : 
  mcgees_pies = 16 → 
  smiths_pies = 4 * mcgees_pies + 6 → 
  smiths_pies = 70 := by
sorry

end smiths_bakery_pies_l1689_168906


namespace systematic_sampling_fourth_group_l1689_168973

/-- Systematic sampling function -/
def systematic_sample (class_size : ℕ) (sample_size : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun group => start + (group - 1) * (class_size / sample_size)

theorem systematic_sampling_fourth_group 
  (class_size : ℕ) 
  (sample_size : ℕ) 
  (second_group_number : ℕ) :
  class_size = 72 →
  sample_size = 6 →
  second_group_number = 16 →
  systematic_sample class_size sample_size second_group_number 4 = 40 :=
by
  sorry

#check systematic_sampling_fourth_group

end systematic_sampling_fourth_group_l1689_168973


namespace ultramen_defeat_monster_l1689_168937

theorem ultramen_defeat_monster (monster_health : ℕ) (ultraman1_rate : ℕ) (ultraman2_rate : ℕ) :
  monster_health = 100 →
  ultraman1_rate = 12 →
  ultraman2_rate = 8 →
  (monster_health : ℚ) / (ultraman1_rate + ultraman2_rate : ℚ) = 5 :=
by sorry

end ultramen_defeat_monster_l1689_168937


namespace inequality_proof_l1689_168912

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a) + (4 / b) ≥ 9 / (a + b) := by sorry

end inequality_proof_l1689_168912


namespace base_12_addition_l1689_168971

/-- Addition in base 12 --/
def base_12_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 12 --/
def to_base_12 (n : ℕ) : ℕ := sorry

/-- Conversion from base 12 to base 10 --/
def from_base_12 (n : ℕ) : ℕ := sorry

theorem base_12_addition :
  base_12_add (from_base_12 528) (from_base_12 274) = to_base_12 940 :=
sorry

end base_12_addition_l1689_168971


namespace max_books_borrowed_l1689_168962

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat) (avg_books : Nat) :
  total_students = 30 →
  zero_books = 5 →
  one_book = 12 →
  two_books = 8 →
  avg_books = 2 →
  ∃ (max_books : Nat), max_books = 20 ∧ 
    ∀ (student_books : Nat), student_books ≤ max_books ∧
    (total_students * avg_books = 
      zero_books * 0 + one_book * 1 + two_books * 2 + 
      (total_students - zero_books - one_book - two_books - 1) * 3 + max_books) :=
by sorry

end max_books_borrowed_l1689_168962


namespace division_of_powers_l1689_168969

theorem division_of_powers (x : ℝ) (h : x ≠ 0) : (-6 * x^3) / (-2 * x^2) = 3 * x := by
  sorry

end division_of_powers_l1689_168969


namespace kekai_sales_ratio_l1689_168986

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := shirts_sold * shirt_price + pants_sold * pants_price

def money_given_to_parents : ℕ := total_earnings - money_left

theorem kekai_sales_ratio :
  (money_given_to_parents : ℚ) / total_earnings = 1 / 2 := by sorry

end kekai_sales_ratio_l1689_168986


namespace escalator_ride_time_l1689_168936

/-- Represents the time it takes Clea to descend the escalator under different conditions -/
structure EscalatorTime where
  stationary : ℝ  -- Time to walk down stationary escalator
  moving : ℝ      -- Time to walk down moving escalator
  riding : ℝ      -- Time to ride down without walking

/-- The theorem states that given the times for walking down stationary and moving escalators,
    the time to ride without walking can be determined -/
theorem escalator_ride_time (et : EscalatorTime) 
  (h1 : et.stationary = 75) 
  (h2 : et.moving = 30) : 
  et.riding = 50 := by
  sorry

end escalator_ride_time_l1689_168936


namespace finite_solutions_of_equation_l1689_168938

theorem finite_solutions_of_equation : 
  Finite {xyz : ℕ × ℕ × ℕ | (1 : ℚ) / xyz.1 + (1 : ℚ) / xyz.2.1 + (1 : ℚ) / xyz.2.2 = (1 : ℚ) / 1983} :=
by sorry

end finite_solutions_of_equation_l1689_168938


namespace relay_team_selection_l1689_168905

-- Define the number of sprinters
def total_sprinters : ℕ := 6

-- Define the number of sprinters to be selected
def selected_sprinters : ℕ := 4

-- Define a function to calculate the number of ways to select and arrange sprinters
def relay_arrangements (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

-- Define a function to calculate the number of ways with restrictions
def restricted_arrangements (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

theorem relay_team_selection :
  restricted_arrangements total_sprinters selected_sprinters = 252 :=
sorry

end relay_team_selection_l1689_168905


namespace sin_cos_alpha_l1689_168947

theorem sin_cos_alpha (α : Real) 
  (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2/5 := by
  sorry

end sin_cos_alpha_l1689_168947


namespace math_only_students_l1689_168909

theorem math_only_students (total : ℕ) (math : ℕ) (foreign : ℕ) 
  (h1 : total = 93) 
  (h2 : math = 70) 
  (h3 : foreign = 54) : 
  math - (math + foreign - total) = 39 := by
  sorry

end math_only_students_l1689_168909


namespace circle_area_equality_l1689_168992

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 25) (h₃ : r₃ = Real.sqrt 481) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 :=
by sorry

end circle_area_equality_l1689_168992


namespace complete_square_formula_l1689_168976

theorem complete_square_formula (x y : ℝ) : x^2 - 2*x*y + y^2 = (x - y)^2 := by
  sorry

end complete_square_formula_l1689_168976


namespace second_exam_sleep_for_average_85_l1689_168997

/-- Represents the relationship between sleep hours and test score -/
structure ExamData where
  sleep : ℝ
  score : ℝ

/-- The constant product of sleep hours and test score -/
def sleepScoreProduct (data : ExamData) : ℝ := data.sleep * data.score

theorem second_exam_sleep_for_average_85 
  (first_exam : ExamData)
  (h_first_exam : first_exam.sleep = 6 ∧ first_exam.score = 60)
  (h_inverse_relation : ∀ exam : ExamData, sleepScoreProduct exam = sleepScoreProduct first_exam)
  (second_exam : ExamData)
  (h_second_exam : second_exam.sleep = 3.3) :
  (first_exam.score + second_exam.score) / 2 = 85 := by
sorry

end second_exam_sleep_for_average_85_l1689_168997


namespace divisor_sum_relation_l1689_168963

theorem divisor_sum_relation (n f g : ℕ) : 
  n > 1 → 
  (∃ d1 d2 : ℕ, d1 ∣ n ∧ d2 ∣ n ∧ d1 ≤ d2 ∧ ∀ d : ℕ, d ∣ n → d = d1 ∨ d ≥ d2 → f = d1 + d2) →
  (∃ d3 d4 : ℕ, d3 ∣ n ∧ d4 ∣ n ∧ d3 ≥ d4 ∧ ∀ d : ℕ, d ∣ n → d = d3 ∨ d ≤ d4 → g = d3 + d4) →
  n = (g * (f - 1)) / f :=
sorry

end divisor_sum_relation_l1689_168963


namespace special_function_omega_value_l1689_168926

/-- A function f with the properties described in the problem -/
structure SpecialFunction (ω : ℝ) where
  f : ℝ → ℝ
  eq : ∀ x, f x = 3 * Real.sin (ω * x + π / 3)
  positive_ω : ω > 0
  equal_values : f (π / 6) = f (π / 3)
  min_no_max : ∃ x₀ ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x₀ ≤ f x
             ∧ ¬∃ x₁ ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x ≤ f x₁

/-- The main theorem stating that ω must be 14/3 -/
theorem special_function_omega_value {ω : ℝ} (sf : SpecialFunction ω) : ω = 14 / 3 := by
  sorry

end special_function_omega_value_l1689_168926


namespace negation_of_proposition_negation_of_inequality_l1689_168916

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 > 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 ≤ 0) := by sorry

end negation_of_proposition_negation_of_inequality_l1689_168916


namespace concentric_circles_equal_areas_l1689_168901

theorem concentric_circles_equal_areas (R : ℝ) (R₁ R₂ : ℝ) 
  (h₁ : R > 0) 
  (h₂ : R₁ > 0) 
  (h₃ : R₂ > 0) 
  (h₄ : R₁ < R₂) 
  (h₅ : R₂ < R) 
  (h₆ : π * R₁^2 = (π * R^2) / 3) 
  (h₇ : π * R₂^2 - π * R₁^2 = (π * R^2) / 3) 
  (h₈ : π * R^2 - π * R₂^2 = (π * R^2) / 3) : 
  R₁ = (R * Real.sqrt 3) / 3 ∧ R₂ = (R * Real.sqrt 6) / 3 := by
  sorry

end concentric_circles_equal_areas_l1689_168901


namespace x_positive_necessary_not_sufficient_l1689_168988

theorem x_positive_necessary_not_sufficient :
  (∃ x : ℝ, x > 0 ∧ ¬(|x - 1| < 1)) ∧
  (∀ x : ℝ, |x - 1| < 1 → x > 0) :=
by sorry

end x_positive_necessary_not_sufficient_l1689_168988


namespace blue_left_handed_fraction_proof_l1689_168948

/-- The fraction of "blue" world participants who are left-handed -/
def blue_left_handed_fraction : ℝ := 0.66

theorem blue_left_handed_fraction_proof :
  let red_to_blue_ratio : ℝ := 2
  let red_left_handed_fraction : ℝ := 1/3
  let total_left_handed_fraction : ℝ := 0.44222222222222224
  blue_left_handed_fraction = 
    (3 * total_left_handed_fraction - 2 * red_left_handed_fraction) / red_to_blue_ratio :=
by sorry

#check blue_left_handed_fraction_proof

end blue_left_handed_fraction_proof_l1689_168948


namespace remainder_three_power_800_mod_17_l1689_168933

theorem remainder_three_power_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end remainder_three_power_800_mod_17_l1689_168933


namespace ivan_petrovich_savings_l1689_168977

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proof of Ivan Petrovich's retirement savings --/
theorem ivan_petrovich_savings : 
  simple_interest 750000 0.08 12 = 1470000 := by
  sorry

end ivan_petrovich_savings_l1689_168977


namespace division_problem_l1689_168940

theorem division_problem : (100 : ℚ) / ((5 / 2) * 3) = 40 / 3 := by
  sorry

end division_problem_l1689_168940


namespace min_book_cover_area_l1689_168925

/-- Given a book cover with reported dimensions of 5 inches by 7 inches,
    where each dimension can vary by ±0.5 inches, the minimum possible area
    of the book cover is 29.25 square inches. -/
theorem min_book_cover_area (reported_length : ℝ) (reported_width : ℝ)
    (actual_length : ℝ) (actual_width : ℝ) :
  reported_length = 5 →
  reported_width = 7 →
  abs (actual_length - reported_length) ≤ 0.5 →
  abs (actual_width - reported_width) ≤ 0.5 →
  ∀ area : ℝ, area = actual_length * actual_width →
    area ≥ 29.25 :=
by sorry

end min_book_cover_area_l1689_168925


namespace rectangle_circle_propositions_l1689_168921

theorem rectangle_circle_propositions (p q : Prop) 
  (hp : p) 
  (hq : ¬q) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end rectangle_circle_propositions_l1689_168921


namespace alexs_score_l1689_168939

theorem alexs_score (total_students : ℕ) (initial_students : ℕ) (initial_avg : ℕ) (final_avg : ℕ) :
  total_students = 20 →
  initial_students = 19 →
  initial_avg = 76 →
  final_avg = 78 →
  (initial_students * initial_avg + (total_students - initial_students) * x) / total_students = final_avg →
  x = 116 :=
by sorry

end alexs_score_l1689_168939


namespace first_year_after_2000_with_digit_sum_15_l1689_168911

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2000 ∧ sum_of_digits year = 15

theorem first_year_after_2000_with_digit_sum_15 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2049 :=
sorry

end first_year_after_2000_with_digit_sum_15_l1689_168911


namespace sum_inequality_l1689_168955

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) :
  a + c > b + d := by
  sorry

end sum_inequality_l1689_168955


namespace equation_solution_l1689_168922

theorem equation_solution : 
  ∃ y : ℚ, (1 : ℚ) / 3 + 1 / y = (4 : ℚ) / 5 ↔ y = 15 / 7 := by
  sorry

end equation_solution_l1689_168922


namespace deepak_age_l1689_168964

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, prove Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end deepak_age_l1689_168964


namespace continued_fraction_equality_l1689_168994

theorem continued_fraction_equality : 
  1 + 1 / (2 + 1 / (2 + 1 / 3)) = 24 / 17 := by
sorry

end continued_fraction_equality_l1689_168994


namespace cos_120_degrees_l1689_168990

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by sorry

end cos_120_degrees_l1689_168990


namespace total_ingredients_for_batches_l1689_168902

/-- The amount of flour needed for one batch of cookies, in cups. -/
def flour_per_batch : ℝ := 4

/-- The amount of sugar needed for one batch of cookies, in cups. -/
def sugar_per_batch : ℝ := 1.5

/-- The number of batches we want to make. -/
def num_batches : ℕ := 8

/-- Theorem: The total amount of flour and sugar needed for 8 batches of cookies is 44 cups. -/
theorem total_ingredients_for_batches : 
  (flour_per_batch + sugar_per_batch) * num_batches = 44 := by sorry

end total_ingredients_for_batches_l1689_168902


namespace geometric_sequence_sum_l1689_168917

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_sum1 : a 1 + a 2 = 4/9)
  (h_sum2 : a 3 + a 4 + a 5 + a 6 = 40) :
  (a 7 + a 8 + a 9) / 9 = 117 := by
  sorry

end geometric_sequence_sum_l1689_168917


namespace hypotenuse_length_squared_l1689_168923

/-- Given complex numbers p, q, and r that are zeros of a polynomial Q(z) = z^3 + sz + t,
    if |p|^2 + |q|^2 + |r|^2 = 300, p + q + r = 0, and p, q, and r form a right triangle
    in the complex plane, then the square of the length of the hypotenuse of this triangle is 450. -/
theorem hypotenuse_length_squared (p q r s t : ℂ) : 
  (Q : ℂ → ℂ) = (fun z ↦ z^3 + s*z + t) →
  p^3 + s*p + t = 0 →
  q^3 + s*q + t = 0 →
  r^3 + s*r + t = 0 →
  Complex.abs p^2 + Complex.abs q^2 + Complex.abs r^2 = 300 →
  p + q + r = 0 →
  ∃ (a b : ℝ), Complex.abs (p - q)^2 = a^2 ∧ Complex.abs (q - r)^2 = b^2 ∧ Complex.abs (p - r)^2 = a^2 + b^2 →
  Complex.abs (p - r)^2 = 450 :=
by sorry

end hypotenuse_length_squared_l1689_168923


namespace cyclist_heartbeats_l1689_168942

/-- Calculates the total number of heartbeats for a cyclist during a race. -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that a cyclist's heart beats 16800 times during a 35-mile race. -/
theorem cyclist_heartbeats :
  let heart_rate := 120  -- heartbeats per minute
  let race_distance := 35  -- miles
  let pace := 4  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 16800 :=
by sorry

end cyclist_heartbeats_l1689_168942
