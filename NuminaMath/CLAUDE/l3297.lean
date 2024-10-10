import Mathlib

namespace perpendicular_line_through_point_l3297_329788

/-- The line passing through (-1, 0) and perpendicular to x+y=0 has equation x-y+1=0 -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + y = 0 → (x + 1 = 0 ∧ y = 0) → x - y + 1 = 0) := by
  sorry

end perpendicular_line_through_point_l3297_329788


namespace cube_root_of_negative_eight_l3297_329778

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l3297_329778


namespace exactly_three_combinations_l3297_329728

/-- Represents a combination of banknotes -/
structure BanknoteCombination :=
  (n_2000 : Nat)
  (n_1000 : Nat)
  (n_500  : Nat)
  (n_200  : Nat)

/-- Checks if a combination is valid according to the problem conditions -/
def isValidCombination (c : BanknoteCombination) : Prop :=
  c.n_2000 + c.n_1000 + c.n_500 + c.n_200 = 10 ∧
  2000 * c.n_2000 + 1000 * c.n_1000 + 500 * c.n_500 + 200 * c.n_200 = 5000

/-- The set of all valid combinations -/
def validCombinations : Set BanknoteCombination :=
  { c | isValidCombination c }

/-- The three specific combinations mentioned in the solution -/
def solution1 : BanknoteCombination := ⟨0, 0, 10, 0⟩
def solution2 : BanknoteCombination := ⟨1, 0, 4, 5⟩
def solution3 : BanknoteCombination := ⟨0, 3, 2, 5⟩

/-- Theorem stating that there are exactly three valid combinations -/
theorem exactly_three_combinations :
  validCombinations = {solution1, solution2, solution3} := by sorry

#check exactly_three_combinations

end exactly_three_combinations_l3297_329728


namespace button_probability_l3297_329755

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Calculate the probability of choosing a blue button from a jar -/
def blueProbability (jar : Jar) : ℚ :=
  jar.blue / (jar.red + jar.blue)

theorem button_probability (jarA jarB : Jar) : 
  jarA.red = 6 ∧ 
  jarA.blue = 10 ∧ 
  jarB.red = 3 ∧ 
  jarB.blue = 5 ∧ 
  (jarA.red + jarA.blue : ℚ) = 2/3 * (6 + 10) →
  blueProbability jarA * blueProbability jarB = 25/64 := by
  sorry

#check button_probability

end button_probability_l3297_329755


namespace integral_of_root_and_polynomial_l3297_329764

open Real

theorem integral_of_root_and_polynomial (x : ℝ) :
  let f := λ x : ℝ => x^(1/2) * (3 + 2*x^(3/4))^(1/2)
  let F := λ x : ℝ => (2/15) * (3 + 2*x^(3/4))^(5/2) - (2/3) * (3 + 2*x^(3/4))^(3/2)
  deriv F x = f x := by sorry

end integral_of_root_and_polynomial_l3297_329764


namespace fourth_six_probability_l3297_329752

/-- Represents a six-sided die --/
structure Die :=
  (prob_six : ℚ)
  (prob_other : ℚ)
  (valid_probs : prob_six + 5 * prob_other = 1)

/-- The fair die --/
def fair_die : Die :=
  { prob_six := 1/6,
    prob_other := 1/6,
    valid_probs := by norm_num }

/-- The biased die --/
def biased_die : Die :=
  { prob_six := 3/4,
    prob_other := 1/20,
    valid_probs := by norm_num }

/-- The probability of rolling three sixes with a given die --/
def prob_three_sixes (d : Die) : ℚ := d.prob_six^3

/-- The probability of the fourth roll being a six given the first three were sixes --/
def prob_fourth_six (fair : Die) (biased : Die) : ℚ :=
  let p_fair := prob_three_sixes fair
  let p_biased := prob_three_sixes biased
  let total := p_fair + p_biased
  (p_fair / total) * fair.prob_six + (p_biased / total) * biased.prob_six

theorem fourth_six_probability :
  prob_fourth_six fair_die biased_die = 685 / 922 :=
sorry

end fourth_six_probability_l3297_329752


namespace sqrt_of_nine_l3297_329750

theorem sqrt_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end sqrt_of_nine_l3297_329750


namespace smallest_student_count_l3297_329761

/-- Represents the number of students in each grade --/
structure GradeCount where
  sixth : ℕ
  eighth : ℕ
  ninth : ℕ

/-- Checks if the given counts satisfy the required ratios --/
def satisfiesRatios (counts : GradeCount) : Prop :=
  5 * counts.sixth = 3 * counts.eighth ∧
  7 * counts.ninth = 4 * counts.eighth

/-- The total number of students --/
def totalStudents (counts : GradeCount) : ℕ :=
  counts.sixth + counts.eighth + counts.ninth

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : GradeCount),
    satisfiesRatios counts ∧
    totalStudents counts = 76 ∧
    ∀ (other : GradeCount),
      satisfiesRatios other →
      totalStudents other ≥ 76 :=
by sorry

end smallest_student_count_l3297_329761


namespace almost_order_lineup_correct_almost_order_lineup_10_l3297_329785

/-- Represents the number of ways to line up n people in almost-order -/
def almost_order_lineup (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 3 => almost_order_lineup (n + 1) + almost_order_lineup (n + 2)

/-- The height difference between consecutive people -/
def height_diff : ℕ := 5

/-- The maximum allowed height difference for almost-order -/
def max_height_diff : ℕ := 8

/-- The height of the shortest person -/
def min_height : ℕ := 140

theorem almost_order_lineup_correct (n : ℕ) :
  (∀ i j, i < j → j ≤ n → min_height + i * height_diff ≤ min_height + j * height_diff + max_height_diff) →
  almost_order_lineup n = if n ≤ 2 then n else almost_order_lineup (n - 1) + almost_order_lineup (n - 2) :=
sorry

theorem almost_order_lineup_10 : almost_order_lineup 10 = 89 :=
sorry

end almost_order_lineup_correct_almost_order_lineup_10_l3297_329785


namespace isosceles_right_triangle_properties_l3297_329758

/-- A right triangle with two equal angles and hypotenuse length 12 -/
structure IsoscelesRightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle equality
  angle_equality : a = b
  -- Hypotenuse length
  hypotenuse_length : c = 12

/-- Theorem about the properties of an isosceles right triangle -/
theorem isosceles_right_triangle_properties (t : IsoscelesRightTriangle) :
  t.a = 6 * Real.sqrt 2 ∧ (1/2 : ℝ) * t.a * t.b = 36 := by
  sorry

#check isosceles_right_triangle_properties

end isosceles_right_triangle_properties_l3297_329758


namespace equation_solution_l3297_329742

theorem equation_solution :
  ∃ y : ℝ, ∀ x : ℝ, x + 0.35 * y - (x + y) = 200 :=
by
  -- The proof would go here
  sorry

end equation_solution_l3297_329742


namespace purely_imaginary_complex_number_l3297_329797

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = 3/2 := by
  sorry

end purely_imaginary_complex_number_l3297_329797


namespace range_of_k_for_two_distinct_roots_l3297_329772

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ (k - 1) * x₂^2 + 2 * x₂ - 2 = 0

/-- The range of k values for which the quadratic equation has two distinct real roots -/
theorem range_of_k_for_two_distinct_roots :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ k > 1/2 ∧ k ≠ 1 :=
by sorry

end range_of_k_for_two_distinct_roots_l3297_329772


namespace shop_equations_correct_l3297_329706

/-- A shop with rooms and guests satisfying certain conditions -/
structure Shop where
  rooms : ℕ
  guests : ℕ
  seven_per_room_overflow : 7 * rooms + 7 = guests
  nine_per_room_empty : 9 * (rooms - 1) = guests

/-- The theorem stating that the system of equations correctly describes the shop's situation -/
theorem shop_equations_correct (s : Shop) : 
  (7 * s.rooms + 7 = s.guests) ∧ (9 * (s.rooms - 1) = s.guests) := by
  sorry

end shop_equations_correct_l3297_329706


namespace units_digit_of_k_squared_plus_two_to_k_l3297_329777

def k : ℕ := 2015^2 + 2^2015

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2015^2 + 2^2015 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l3297_329777


namespace kennedy_drive_home_l3297_329782

/-- Calculates the remaining miles that can be driven given the car's efficiency,
    initial gas amount, and distance already driven. -/
def remaining_miles (efficiency : ℝ) (initial_gas : ℝ) (driven_miles : ℝ) : ℝ :=
  efficiency * initial_gas - driven_miles

theorem kennedy_drive_home 
  (efficiency : ℝ) 
  (initial_gas : ℝ) 
  (school_miles : ℝ) 
  (softball_miles : ℝ) 
  (restaurant_miles : ℝ) 
  (friend_miles : ℝ) 
  (h1 : efficiency = 19)
  (h2 : initial_gas = 2)
  (h3 : school_miles = 15)
  (h4 : softball_miles = 6)
  (h5 : restaurant_miles = 2)
  (h6 : friend_miles = 4) :
  remaining_miles efficiency initial_gas (school_miles + softball_miles + restaurant_miles + friend_miles) = 11 := by
  sorry

end kennedy_drive_home_l3297_329782


namespace min_bailing_rate_is_8_l3297_329738

/-- Represents the fishing scenario with Steve and LeRoy -/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  max_water_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach the shore without sinking -/
def min_bailing_rate (scenario : FishingScenario) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the minimum bailing rate for the given scenario is 8 gallons per minute -/
theorem min_bailing_rate_is_8 (scenario : FishingScenario) 
  (h1 : scenario.distance_to_shore = 1)
  (h2 : scenario.water_intake_rate = 10)
  (h3 : scenario.max_water_capacity = 30)
  (h4 : scenario.rowing_speed = 4) :
  min_bailing_rate scenario = 8 := by
  sorry

end min_bailing_rate_is_8_l3297_329738


namespace fraction_ratio_equality_l3297_329711

theorem fraction_ratio_equality : 
  let certain_fraction : ℚ := 84 / 25
  let given_fraction : ℚ := 6 / 5
  let comparison_fraction : ℚ := 2 / 5
  let answer : ℚ := 1 / 7  -- 0.14285714285714288 is approximately 1/7
  (certain_fraction / given_fraction) = (comparison_fraction / answer) :=
by sorry

end fraction_ratio_equality_l3297_329711


namespace handshake_count_l3297_329792

/-- Represents the number of women in each age group -/
def women_per_group : ℕ := 5

/-- Represents the number of age groups -/
def num_groups : ℕ := 3

/-- Calculates the number of inter-group handshakes -/
def inter_group_handshakes : ℕ := women_per_group * women_per_group * (num_groups.choose 2)

/-- Calculates the number of intra-group handshakes for a single group -/
def intra_group_handshakes : ℕ := women_per_group.choose 2

/-- Calculates the total number of handshakes -/
def total_handshakes : ℕ := inter_group_handshakes + num_groups * intra_group_handshakes

/-- Theorem stating that the total number of handshakes is 105 -/
theorem handshake_count : total_handshakes = 105 := by
  sorry

end handshake_count_l3297_329792


namespace town_square_length_l3297_329703

/-- The length of the town square in miles -/
def square_length : ℝ := 5.25

/-- The number of times runners go around the square -/
def laps : ℕ := 7

/-- The time (in minutes) it took the winner to finish the race this year -/
def winner_time : ℝ := 42

/-- The time (in minutes) it took last year's winner to finish the race -/
def last_year_time : ℝ := 47.25

/-- The time difference (in minutes) for running one mile between this year and last year -/
def speed_improvement : ℝ := 1

theorem town_square_length :
  square_length = (last_year_time - winner_time) / speed_improvement :=
by sorry

end town_square_length_l3297_329703


namespace S_3_5_equals_42_l3297_329774

-- Define the operation S
def S (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem to prove
theorem S_3_5_equals_42 : S 3 5 = 42 := by
  sorry

end S_3_5_equals_42_l3297_329774


namespace team_games_total_l3297_329766

theorem team_games_total (first_games : ℕ) (first_win_rate : ℚ) (remaining_win_rate : ℚ) (total_win_rate : ℚ) : 
  first_games = 30 →
  first_win_rate = 2/5 →
  remaining_win_rate = 4/5 →
  total_win_rate = 3/5 →
  ∃ (total_games : ℕ), total_games = 60 ∧ 
    (first_win_rate * first_games + remaining_win_rate * (total_games - first_games) = total_win_rate * total_games) :=
by
  sorry

#check team_games_total

end team_games_total_l3297_329766


namespace function_identification_l3297_329730

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- State the theorem
theorem function_identification (a b c : ℝ) :
  f a b c 0 = 1 ∧ 
  (∃ k m : ℝ, k = 4 * a * 1^3 + 2 * b * 1 ∧ 
              m = a * 1^4 + b * 1^2 + c ∧ 
              k = 1 ∧ 
              m = -1) →
  ∀ x, f a b c x = 5/2 * x^4 - 9/2 * x^2 + 1 :=
by sorry

end function_identification_l3297_329730


namespace shoe_price_problem_l3297_329753

theorem shoe_price_problem (first_pair_price : ℝ) (total_paid : ℝ) :
  first_pair_price = 40 →
  total_paid = 60 →
  ∃ (second_pair_price : ℝ),
    second_pair_price ≥ first_pair_price ∧
    total_paid = (3/4) * (first_pair_price + (1/2) * second_pair_price) ∧
    second_pair_price = 80 :=
by
  sorry

#check shoe_price_problem

end shoe_price_problem_l3297_329753


namespace square_equals_product_plus_seven_l3297_329767

theorem square_equals_product_plus_seven (a b : ℕ) : 
  (a^2 = b * (b + 7)) ↔ ((a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9)) :=
sorry

end square_equals_product_plus_seven_l3297_329767


namespace red_apples_count_l3297_329746

theorem red_apples_count (red : ℕ) (green : ℕ) : 
  green = red + 12 →
  red + green = 44 →
  red = 16 := by
sorry

end red_apples_count_l3297_329746


namespace a_4_equals_8_l3297_329717

def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q^(n - 1)

theorem a_4_equals_8 
  (a : ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q 6 + geometric_sequence a q 2 = 34)
  (h2 : geometric_sequence a q 6 - geometric_sequence a q 2 = 30) :
  geometric_sequence a q 4 = 8 :=
sorry

end a_4_equals_8_l3297_329717


namespace olly_minimum_cost_l3297_329735

/-- Represents the number of each type of pet Olly has -/
structure Pets where
  dogs : Nat
  cats : Nat
  ferrets : Nat

/-- Represents the pricing and discount structure for Pack A -/
structure PackA where
  small_shoe_price : ℝ
  medium_shoe_price : ℝ
  small_shoe_discount : ℝ
  medium_shoe_discount : ℝ

/-- Represents the pricing and discount structure for Pack B -/
structure PackB where
  small_shoe_price : ℝ
  medium_shoe_price : ℝ
  small_shoe_free_ratio : Nat
  medium_shoe_free_ratio : Nat

/-- Calculates the minimum cost for Olly to purchase shoes for all his pets -/
def minimum_cost (pets : Pets) (pack_a : PackA) (pack_b : PackB) : ℝ := by
  sorry

/-- Theorem stating that the minimum cost for Olly to purchase shoes for all his pets is $64 -/
theorem olly_minimum_cost :
  let pets := Pets.mk 3 2 1
  let pack_a := PackA.mk 12 16 0.2 0.15
  let pack_b := PackB.mk 7 9 3 4
  minimum_cost pets pack_a pack_b = 64 := by
  sorry

end olly_minimum_cost_l3297_329735


namespace smallest_y_squared_l3297_329714

/-- An isosceles trapezoid with a inscribed circle --/
structure IsoscelesTrapezoidWithCircle where
  -- Length of the longer base
  AB : ℝ
  -- Length of the shorter base
  CD : ℝ
  -- Length of the legs
  y : ℝ
  -- The circle's center is on AB and it's tangent to AD and BC
  has_inscribed_circle : Bool

/-- The smallest possible y value for the given trapezoid configuration --/
def smallest_y (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  sorry

/-- Theorem stating that the square of the smallest y is 900 --/
theorem smallest_y_squared (t : IsoscelesTrapezoidWithCircle) 
  (h1 : t.AB = 100)
  (h2 : t.CD = 64)
  (h3 : t.has_inscribed_circle = true) :
  (smallest_y t) ^ 2 = 900 :=
sorry

end smallest_y_squared_l3297_329714


namespace coin_value_theorem_l3297_329737

theorem coin_value_theorem (n d : ℕ) : 
  n + d = 25 →
  (10 * n + 5 * d) - (5 * n + 10 * d) = 100 →
  5 * n + 10 * d = 140 := by
  sorry

end coin_value_theorem_l3297_329737


namespace midway_point_distance_l3297_329743

/-- The distance from Yooseon's house to the midway point of her path to school -/
def midway_distance (house_to_hospital : ℕ) (hospital_to_school : ℕ) : ℕ :=
  (house_to_hospital + hospital_to_school) / 2

theorem midway_point_distance :
  let house_to_hospital := 1700
  let hospital_to_school := 900
  midway_distance house_to_hospital hospital_to_school = 1300 := by
  sorry

end midway_point_distance_l3297_329743


namespace sqrt_two_squared_l3297_329726

theorem sqrt_two_squared : (Real.sqrt 2)^2 = 2 := by
  sorry

end sqrt_two_squared_l3297_329726


namespace reciprocal_inequality_l3297_329712

theorem reciprocal_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end reciprocal_inequality_l3297_329712


namespace floor_of_e_l3297_329757

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end floor_of_e_l3297_329757


namespace number_ratio_l3297_329756

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 15) = 75) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l3297_329756


namespace prove_a_minus_b_l3297_329795

-- Define the equation
def equation (a b c x : ℝ) : Prop :=
  (2*x - 3)^2 = a*x^2 + b*x + c

-- Theorem statement
theorem prove_a_minus_b (a b c : ℝ) 
  (h : ∀ x : ℝ, equation a b c x) : a - b = 16 := by
  sorry

end prove_a_minus_b_l3297_329795


namespace rectangle_area_l3297_329771

/-- Given a rectangle with width 5 inches and length 3 times its width, prove its area is 75 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 → length = 3 * width → area = length * width → area = 75 := by
  sorry

end rectangle_area_l3297_329771


namespace janes_calculation_l3297_329747

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by
sorry

end janes_calculation_l3297_329747


namespace arithmetic_computation_l3297_329770

theorem arithmetic_computation : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end arithmetic_computation_l3297_329770


namespace height_comparison_l3297_329796

theorem height_comparison (height_a height_b : ℝ) (h : height_a = 0.75 * height_b) :
  (height_b - height_a) / height_a = 1/3 := by
  sorry

end height_comparison_l3297_329796


namespace football_game_attendance_l3297_329727

/-- Prove the total attendance at a football game -/
theorem football_game_attendance
  (adult_price : ℚ)
  (child_price : ℚ)
  (total_collected : ℚ)
  (num_adults : ℕ)
  (h1 : adult_price = 60 / 100)
  (h2 : child_price = 25 / 100)
  (h3 : total_collected = 140)
  (h4 : num_adults = 200) :
  num_adults + ((total_collected - (↑num_adults * adult_price)) / child_price) = 280 :=
by sorry

end football_game_attendance_l3297_329727


namespace farm_trip_chaperones_l3297_329715

theorem farm_trip_chaperones (num_students : ℕ) (student_fee adult_fee total_fee : ℚ) : 
  num_students = 35 →
  student_fee = 5 →
  adult_fee = 6 →
  total_fee = 199 →
  ∃ (num_adults : ℕ), num_adults * adult_fee + num_students * student_fee = total_fee ∧ num_adults = 4 :=
by sorry

end farm_trip_chaperones_l3297_329715


namespace factorization_proof_l3297_329722

theorem factorization_proof (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end factorization_proof_l3297_329722


namespace camel_cannot_end_adjacent_l3297_329732

/-- Represents a hexagonal board with side length m -/
structure HexBoard where
  m : ℕ

/-- The total number of fields on a hexagonal board -/
def HexBoard.total_fields (board : HexBoard) : ℕ :=
  3 * board.m^2 - 3 * board.m + 1

/-- The number of moves a camel makes on the board -/
def HexBoard.camel_moves (board : HexBoard) : ℕ :=
  board.total_fields - 1

/-- Theorem stating that a camel cannot end on an adjacent field to its starting position -/
theorem camel_cannot_end_adjacent (board : HexBoard) :
  ∃ (start finish : ℕ), start ≠ finish ∧ 
  finish ≠ (start + 1) ∧ finish ≠ (start - 1) ∧
  finish = (start + board.camel_moves) % board.total_fields :=
sorry

end camel_cannot_end_adjacent_l3297_329732


namespace contrapositive_equivalence_l3297_329705

def is_decreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem contrapositive_equivalence (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 2)) / 2 < a (n + 1) → is_decreasing a) ↔
  (¬ is_decreasing a → ∀ n : ℕ+, (a n + a (n + 2)) / 2 ≥ a (n + 1)) :=
by sorry

end contrapositive_equivalence_l3297_329705


namespace donut_selection_problem_l3297_329760

theorem donut_selection_problem :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 56 := by
sorry

end donut_selection_problem_l3297_329760


namespace trapezoid_area_between_triangles_l3297_329734

/-- Given two equilateral triangles, one inside the other, this theorem calculates
    the area of one of the three congruent trapezoids formed between them. -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ)
  (inner_area : ℝ)
  (h_outer : outer_area = 36)
  (h_inner : inner_area = 4)
  (h_positive : 0 < inner_area ∧ inner_area < outer_area) :
  (outer_area - inner_area) / 3 = 32 / 3 := by
  sorry

end trapezoid_area_between_triangles_l3297_329734


namespace apple_orange_difference_l3297_329716

theorem apple_orange_difference (total : Nat) (apples : Nat) (h1 : total = 301) (h2 : apples = 164) (h3 : apples > total - apples) : apples - (total - apples) = 27 := by
  sorry

end apple_orange_difference_l3297_329716


namespace complex_number_coordinates_i_times_one_minus_i_l3297_329776

theorem complex_number_coordinates : Complex → Complex → Prop :=
  fun z w => z = w

theorem i_times_one_minus_i (i : Complex) (h : i * i = -1) :
  complex_number_coordinates (i * (1 - i)) (1 + i) := by
  sorry

end complex_number_coordinates_i_times_one_minus_i_l3297_329776


namespace outfit_combinations_l3297_329779

theorem outfit_combinations (n : ℕ) (h : n = 7) : n^3 - n = 336 := by
  sorry

end outfit_combinations_l3297_329779


namespace g_neg_501_l3297_329794

-- Define the function g
variable (g : ℝ → ℝ)

-- State the conditions
axiom func_eq : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_neg_one : g (-1) = 7

-- State the theorem to be proved
theorem g_neg_501 : g (-501) = 507 := by sorry

end g_neg_501_l3297_329794


namespace some_students_not_honor_society_l3297_329783

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Scholarship : U → Prop)
variable (HonorSociety : U → Prop)

-- State the theorem
theorem some_students_not_honor_society :
  (∃ x, Student x ∧ ¬Scholarship x) →
  (∀ x, HonorSociety x → Scholarship x) →
  (∃ x, Student x ∧ ¬HonorSociety x) :=
by
  sorry


end some_students_not_honor_society_l3297_329783


namespace square_58_sexagesimal_l3297_329704

/-- Represents a number in sexagesimal form a•b, where the value is a*60 + b -/
structure Sexagesimal where
  a : ℕ
  b : ℕ
  h : b < 60

/-- Converts a natural number to its sexagesimal representation -/
def to_sexagesimal (n : ℕ) : Sexagesimal :=
  ⟨n / 60, n % 60, sorry⟩

/-- The statement to be proved -/
theorem square_58_sexagesimal : 
  to_sexagesimal (58^2) = Sexagesimal.mk 56 4 sorry := by sorry

end square_58_sexagesimal_l3297_329704


namespace solution_set_part1_range_of_a_part2_l3297_329718

-- Define the function f
def f (x t : ℝ) : ℝ := |x - 1| + |x - t|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 > 2} = {x : ℝ | x < (1/2) ∨ x > (5/2)} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ t x : ℝ, t ∈ [1, 2] → x ∈ [-1, 3] → f x t ≥ a + x) → a ≤ -1 := by sorry

end solution_set_part1_range_of_a_part2_l3297_329718


namespace quadratic_coefficients_l3297_329719

/-- A quadratic function with vertex (-2, 5) passing through (0, 3) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_coefficients :
  ∃ (a b c : ℝ),
    (∀ x, f a b c x = a * x^2 + b * x + c) ∧
    (f a b c (-2) = 5) ∧
    (∀ x, f a b c (x) = f a b c (-x - 4)) ∧
    (f a b c 0 = 3) ∧
    (a = -1/2 ∧ b = -2 ∧ c = 3) :=
by sorry

end quadratic_coefficients_l3297_329719


namespace three_digit_permutations_l3297_329751

def digits : Finset Nat := {1, 5, 8}

theorem three_digit_permutations (d : Finset Nat) (h : d = digits) :
  (d.toList.permutations.filter (fun l => l.length = 3)).length = 6 := by
  sorry

end three_digit_permutations_l3297_329751


namespace A_older_than_B_by_two_l3297_329733

-- Define the ages of A, B, and C
def B : ℕ := 14
def C : ℕ := B / 2
def A : ℕ := 37 - B - C

-- Theorem statement
theorem A_older_than_B_by_two : A = B + 2 := by
  sorry

end A_older_than_B_by_two_l3297_329733


namespace square_area_from_diagonal_l3297_329762

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  (d^2 / 2 : ℝ) = 72 := by
  sorry

#check square_area_from_diagonal

end square_area_from_diagonal_l3297_329762


namespace bakery_earnings_for_five_days_l3297_329799

/-- Represents the daily production and prices of baked goods in Uki's bakery --/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ

/-- Calculates the total earnings for a given number of days --/
def total_earnings (data : BakeryData) (days : ℕ) : ℝ :=
  let daily_earnings := 
    data.cupcake_price * data.cupcakes_per_day +
    data.cookie_price * data.cookie_packets_per_day +
    data.biscuit_price * data.biscuit_packets_per_day
  daily_earnings * days

/-- Theorem stating that the total earnings for 5 days is $350 --/
theorem bakery_earnings_for_five_days :
  let data := BakeryData.mk 1.5 2 1 20 10 20
  total_earnings data 5 = 350 := by
  sorry

end bakery_earnings_for_five_days_l3297_329799


namespace triangle_area_l3297_329775

/-- The area of a triangle with vertices at (0,0), (8,8), and (-8,8) is 64 -/
theorem triangle_area : 
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (8, 8)
  let B : ℝ × ℝ := (-8, 8)
  let base := |A.1 - B.1|
  let height := A.2
  (1 / 2) * base * height = 64 := by sorry

end triangle_area_l3297_329775


namespace lines_planes_perpendicular_l3297_329745

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem lines_planes_perpendicular 
  (m n : Line) (α β : Plane) :
  parallel m n →
  contains α m →
  perpendicular n β →
  plane_perpendicular α β :=
sorry

end lines_planes_perpendicular_l3297_329745


namespace f_min_value_l3297_329773

/-- The function f to be minimized -/
def f (x y z : ℝ) : ℝ :=
  x^2 + 2*y^2 + 3*z^2 + 2*x*y + 4*y*z + 2*z*x - 6*x - 10*y - 12*z

/-- Theorem stating that -14 is the minimum value of f -/
theorem f_min_value :
  ∀ x y z : ℝ, f x y z ≥ -14 :=
by sorry

end f_min_value_l3297_329773


namespace existence_of_solution_l3297_329713

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Condition: For any positive integers k₁ and k₂, ⌊k₁α⌋ ≠ ⌊k₂β⌋ -/
def condition (α β : ℝ) : Prop :=
  ∀ (k₁ k₂ : ℕ), k₁ > 0 ∧ k₂ > 0 → floor (k₁ * α) ≠ floor (k₂ * β)

/-- Theorem: If the condition holds for positive real numbers α and β,
    then there exist positive integers m₁ and m₂ such that (m₁/α) + (m₂/β) = 1 -/
theorem existence_of_solution (α β : ℝ) (hα : α > 0) (hβ : β > 0) 
    (h : condition α β) : 
    ∃ (m₁ m₂ : ℕ), m₁ > 0 ∧ m₂ > 0 ∧ (m₁ : ℝ) / α + (m₂ : ℝ) / β = 1 :=
  sorry

end existence_of_solution_l3297_329713


namespace octagon_pebble_arrangements_l3297_329789

/-- The number of symmetries (rotations and reflections) of a regular octagon -/
def octagon_symmetries : ℕ := 16

/-- The number of vertices in a regular octagon -/
def octagon_vertices : ℕ := 8

/-- The number of distinct arrangements of pebbles on a regular octagon -/
def distinct_arrangements : ℕ := Nat.factorial octagon_vertices / octagon_symmetries

theorem octagon_pebble_arrangements :
  distinct_arrangements = 2520 := by sorry

end octagon_pebble_arrangements_l3297_329789


namespace expression_simplification_l3297_329710

theorem expression_simplification (x : ℝ) :
  2 * x * (4 * x^2 - 3) - 6 * (x^2 - 3 * x + 8) = 8 * x^3 - 6 * x^2 + 12 * x - 48 := by
  sorry

end expression_simplification_l3297_329710


namespace fraction_inequality_l3297_329759

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end fraction_inequality_l3297_329759


namespace triangle_median_inequality_l3297_329744

/-- Given a triangle with side lengths a, b, c, medians m_a, m_b, m_c, 
    and circumscribed circle diameter D, the following inequality holds. -/
theorem triangle_median_inequality 
  (a b c m_a m_b m_c D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : 4 * m_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h_median_b : 4 * m_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h_median_c : 4 * m_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (h_circumradius : D = 2 * (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b ≤ 6 * D := by
  sorry

end triangle_median_inequality_l3297_329744


namespace optimal_bus_rental_plan_l3297_329729

/-- Represents a bus rental plan -/
structure BusRentalPlan where
  modelA : ℕ
  modelB : ℕ

/-- Calculates the total capacity of a bus rental plan -/
def totalCapacity (plan : BusRentalPlan) : ℕ :=
  40 * plan.modelA + 55 * plan.modelB

/-- Calculates the total cost of a bus rental plan -/
def totalCost (plan : BusRentalPlan) : ℕ :=
  600 * plan.modelA + 700 * plan.modelB

/-- Checks if a bus rental plan is valid -/
def isValidPlan (plan : BusRentalPlan) : Prop :=
  plan.modelA + plan.modelB = 10 ∧ 
  plan.modelA ≥ 1 ∧ 
  plan.modelB ≥ 1 ∧
  totalCapacity plan ≥ 502

/-- Theorem stating the properties of the optimal bus rental plan -/
theorem optimal_bus_rental_plan :
  ∃ (optimalPlan : BusRentalPlan),
    isValidPlan optimalPlan ∧
    optimalPlan.modelA = 3 ∧
    optimalPlan.modelB = 7 ∧
    totalCost optimalPlan = 6700 ∧
    (∀ (plan : BusRentalPlan), isValidPlan plan → totalCost plan ≥ totalCost optimalPlan) ∧
    (∀ (plan : BusRentalPlan), isValidPlan plan → plan.modelA ≤ 3) :=
  sorry


end optimal_bus_rental_plan_l3297_329729


namespace square_sum_or_product_l3297_329707

theorem square_sum_or_product (a b c : ℕ+) (p : ℕ) :
  a + b = b * (a - c) →
  c + 1 = p^2 →
  Nat.Prime p →
  (∃ k : ℕ, (a + b : ℕ) = k^2) ∨ (∃ k : ℕ, (a * b : ℕ) = k^2) := by
  sorry

end square_sum_or_product_l3297_329707


namespace lake_fish_population_l3297_329791

/-- Represents the fish population in a lake --/
structure FishPopulation where
  initial_tagged : ℕ
  second_catch : ℕ
  tagged_in_second_catch : ℕ
  new_migrants : ℕ

/-- Calculates the approximate total number of fish in the lake --/
def approximate_total_fish (fp : FishPopulation) : ℕ :=
  (fp.initial_tagged * fp.second_catch) / fp.tagged_in_second_catch

/-- The main theorem stating the approximate number of fish in the lake --/
theorem lake_fish_population (fp : FishPopulation) 
  (h1 : fp.initial_tagged = 500)
  (h2 : fp.second_catch = 300)
  (h3 : fp.tagged_in_second_catch = 6)
  (h4 : fp.new_migrants = 250) :
  approximate_total_fish fp = 25000 := by
  sorry

#eval approximate_total_fish { initial_tagged := 500, second_catch := 300, tagged_in_second_catch := 6, new_migrants := 250 }

end lake_fish_population_l3297_329791


namespace car_trip_equation_correct_l3297_329748

/-- Represents a car trip with a break -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  breakDuration : ℝ
  speedBefore : ℝ
  speedAfter : ℝ

/-- The equation representing the relationship between time before break and total distance -/
def tripEquation (trip : CarTrip) (t : ℝ) : Prop :=
  trip.speedBefore * t + trip.speedAfter * (trip.totalTime - trip.breakDuration / 60 - t) = trip.totalDistance

theorem car_trip_equation_correct (trip : CarTrip) : 
  trip.totalDistance = 295 ∧ 
  trip.totalTime = 3.25 ∧ 
  trip.breakDuration = 15 ∧ 
  trip.speedBefore = 85 ∧ 
  trip.speedAfter = 115 → 
  ∃ t, tripEquation trip t ∧ t > 0 ∧ t < trip.totalTime - trip.breakDuration / 60 :=
sorry

end car_trip_equation_correct_l3297_329748


namespace tony_weightlifting_ratio_l3297_329702

/-- Given Tony's weightlifting capabilities, prove the ratio of squat to military press weight -/
theorem tony_weightlifting_ratio :
  let curl_weight : ℝ := 90
  let military_press_weight : ℝ := 2 * curl_weight
  let squat_weight : ℝ := 900
  squat_weight / military_press_weight = 5 := by sorry

end tony_weightlifting_ratio_l3297_329702


namespace quadratic_function_theorem_l3297_329769

/-- A quadratic function symmetric about the y-axis -/
def QuadraticFunction (a c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + c

theorem quadratic_function_theorem (a c : ℝ) :
  (QuadraticFunction a c 0 = -2) →
  (QuadraticFunction a c 1 = -1) →
  (∃ (x : ℝ), QuadraticFunction a c x = QuadraticFunction a c (-x)) →
  (QuadraticFunction a c = fun x ↦ x^2 - 2) ∧
  (∃! (x y : ℝ), x ≠ y ∧ QuadraticFunction a c x = 0 ∧ QuadraticFunction a c y = 0) :=
by sorry

end quadratic_function_theorem_l3297_329769


namespace probability_correct_l3297_329736

/-- Represents a standard six-sided die --/
def Die := Fin 6

/-- The probability of the event described in the problem --/
def probability : ℚ :=
  (5 * 4^9) / (6^11)

/-- The function that calculates the probability of the event --/
def calculate_probability : ℚ :=
  -- First roll: any number (1)
  -- Rolls 2 to 10: different from previous, not 4 on 11th (5/6 * (4/5)^9)
  -- 11th and 12th rolls both 4 (1/6 * 1/6)
  1 * (5/6) * (4/5)^9 * (1/6)^2

theorem probability_correct :
  calculate_probability = probability := by sorry

end probability_correct_l3297_329736


namespace periodic_trig_function_l3297_329708

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, and β are constants, 
    if f(2009) = 5, then f(2010) = 3 -/
theorem periodic_trig_function 
  (a b α β : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4) 
  (h2 : f 2009 = 5) : 
  f 2010 = 3 := by
sorry

end periodic_trig_function_l3297_329708


namespace least_people_for_cheaper_second_service_l3297_329741

/-- Represents a catering service with a basic fee and per-person charge -/
structure CateringService where
  basicFee : ℕ
  perPersonCharge : ℕ

/-- Calculates the total cost for a catering service given the number of people -/
def totalCost (service : CateringService) (people : ℕ) : ℕ :=
  service.basicFee + service.perPersonCharge * people

/-- The first catering service -/
def service1 : CateringService := { basicFee := 150, perPersonCharge := 18 }

/-- The second catering service -/
def service2 : CateringService := { basicFee := 250, perPersonCharge := 15 }

/-- Theorem stating that 34 is the least number of people for which the second service is cheaper -/
theorem least_people_for_cheaper_second_service :
  (∀ n : ℕ, n < 34 → totalCost service1 n ≤ totalCost service2 n) ∧
  totalCost service2 34 < totalCost service1 34 := by
  sorry

end least_people_for_cheaper_second_service_l3297_329741


namespace fourth_day_earning_l3297_329781

/-- Represents the daily earnings of a mechanic for a week -/
def MechanicEarnings : Type := Fin 7 → ℝ

/-- The average earning for the first 4 days is 18 -/
def avg_first_four (e : MechanicEarnings) : Prop :=
  (e 0 + e 1 + e 2 + e 3) / 4 = 18

/-- The average earning for the last 4 days is 22 -/
def avg_last_four (e : MechanicEarnings) : Prop :=
  (e 3 + e 4 + e 5 + e 6) / 4 = 22

/-- The average earning for the whole week is 21 -/
def avg_whole_week (e : MechanicEarnings) : Prop :=
  (e 0 + e 1 + e 2 + e 3 + e 4 + e 5 + e 6) / 7 = 21

/-- The theorem stating that given the conditions, the earning on the fourth day is 13 -/
theorem fourth_day_earning (e : MechanicEarnings) 
  (h1 : avg_first_four e) 
  (h2 : avg_last_four e) 
  (h3 : avg_whole_week e) : 
  e 3 = 13 := by sorry

end fourth_day_earning_l3297_329781


namespace coefficient_a2_value_l3297_329723

theorem coefficient_a2_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, x^2 + (x+1)^7 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7) →
  a₂ = -20 := by
sorry

end coefficient_a2_value_l3297_329723


namespace unique_matching_number_l3297_329731

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  t_range : tens ≥ 0 ∧ tens ≤ 9
  u_range : units ≥ 0 ∧ units ≤ 9

/-- Checks if two ThreeDigitNumbers match in exactly one digit place -/
def matchesOneDigit (a b : ThreeDigitNumber) : Prop :=
  (a.hundreds = b.hundreds ∧ a.tens ≠ b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens = b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens ≠ b.tens ∧ a.units = b.units)

/-- The theorem to be proved -/
theorem unique_matching_number : ∃! n : ThreeDigitNumber,
  matchesOneDigit n ⟨1, 0, 9, by sorry, by sorry, by sorry⟩ ∧
  matchesOneDigit n ⟨7, 0, 4, by sorry, by sorry, by sorry⟩ ∧
  matchesOneDigit n ⟨1, 2, 4, by sorry, by sorry, by sorry⟩ ∧
  n = ⟨7, 2, 9, by sorry, by sorry, by sorry⟩ :=
sorry

end unique_matching_number_l3297_329731


namespace quadratic_real_roots_condition_l3297_329798

theorem quadratic_real_roots_condition 
  (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ (a ≠ 0 ∧ b^2 - 4*a*c ≥ 0) :=
sorry

end quadratic_real_roots_condition_l3297_329798


namespace smallest_integer_with_given_remainders_l3297_329780

theorem smallest_integer_with_given_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 3 = 2 ∧
  n % 5 = 4 ∧
  n % 7 = 6 ∧
  n % 11 = 10 ∧
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 5 = 4 ∧ m % 7 = 6 ∧ m % 11 = 10 → n ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_integer_with_given_remainders_l3297_329780


namespace rachel_money_theorem_l3297_329790

def rachel_money_problem (initial_earnings : ℝ) 
  (lunch_fraction : ℝ) (clothes_percent : ℝ) (dvd_cost : ℝ) (supplies_percent : ℝ) : Prop :=
  let lunch_cost := initial_earnings * lunch_fraction
  let clothes_cost := initial_earnings * (clothes_percent / 100)
  let supplies_cost := initial_earnings * (supplies_percent / 100)
  let total_expenses := lunch_cost + clothes_cost + dvd_cost + supplies_cost
  let money_left := initial_earnings - total_expenses
  money_left = 74.50

theorem rachel_money_theorem :
  rachel_money_problem 200 0.25 15 24.50 10.5 := by
  sorry

end rachel_money_theorem_l3297_329790


namespace symmetric_point_coordinates_l3297_329787

def point (x y : ℝ) := (x, y)

def symmetric_point_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := point (-2) 1
  let Q : ℝ × ℝ := symmetric_point_x_axis P
  Q = point (-2) (-1) := by sorry

end symmetric_point_coordinates_l3297_329787


namespace max_rectangles_in_6x6_square_l3297_329740

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents the maximum number of rectangles that can fit in a square -/
def max_rectangles_in_square (r : Rectangle) (s : Square) : ℕ :=
  sorry

/-- The theorem stating the maximum number of 4×1 rectangles in a 6×6 square -/
theorem max_rectangles_in_6x6_square :
  let r : Rectangle := ⟨4, 1⟩
  let s : Square := ⟨6⟩
  max_rectangles_in_square r s = 8 := by
  sorry

end max_rectangles_in_6x6_square_l3297_329740


namespace expected_adjacent_red_pairs_l3297_329793

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCards : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def probAdjacentRed : ℚ := 25 / 51

theorem expected_adjacent_red_pairs (deckSize : ℕ) (redCards : ℕ) (probAdjacentRed : ℚ) :
  deckSize = 52 → redCards = 26 → probAdjacentRed = 25 / 51 →
  (redCards : ℚ) * probAdjacentRed = 650 / 51 := by
  sorry

end expected_adjacent_red_pairs_l3297_329793


namespace initial_pencils_count_l3297_329768

/-- The number of pencils Eric takes from the box -/
def pencils_taken : ℕ := 4

/-- The number of pencils left in the box after Eric takes some -/
def pencils_left : ℕ := 75

/-- The initial number of pencils in the box -/
def initial_pencils : ℕ := pencils_taken + pencils_left

theorem initial_pencils_count : initial_pencils = 79 := by
  sorry

end initial_pencils_count_l3297_329768


namespace fib_2n_square_sum_l3297_329700

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: For a Fibonacci sequence, f_{2n} = f_{n-1}^2 + f_n^2 for all natural numbers n -/
theorem fib_2n_square_sum (n : ℕ) : fib (2 * n) = (fib (n - 1))^2 + (fib n)^2 := by
  sorry

end fib_2n_square_sum_l3297_329700


namespace bike_riders_count_l3297_329709

theorem bike_riders_count (total : ℕ) (hikers : ℕ) (bikers : ℕ) :
  total = hikers + bikers →
  hikers = bikers + 178 →
  total = 676 →
  bikers = 249 := by
sorry

end bike_riders_count_l3297_329709


namespace ten_by_ten_not_tileable_l3297_329725

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile -/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- Defines a function to check if a checkerboard can be tiled with given tiles -/
def can_tile (board : Checkerboard) (tile : Tile) : Prop :=
  ∃ (n : ℕ), board.rows * board.cols = n * tile.width * tile.height

/-- Theorem stating that a 10x10 checkerboard cannot be tiled with 1x4 tiles -/
theorem ten_by_ten_not_tileable :
  ¬ can_tile (Checkerboard.mk 10 10) (Tile.mk 1 4) :=
sorry

end ten_by_ten_not_tileable_l3297_329725


namespace main_theorem_l3297_329721

/-- Proposition p: for all positive x, x + a/x ≥ 2 -/
def p (a : ℝ) : Prop :=
  ∀ x > 0, x + a / x ≥ 2

/-- Proposition q: for all real k, the line kx - y + 2 = 0 intersects with the ellipse x^2 + y^2/a^2 = 1 -/
def q (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

/-- The main theorem: (p ∨ q) ∧ ¬(p ∧ q) is true if and only if 1 ≤ a < 2 -/
theorem main_theorem (a : ℝ) (h : a > 0) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ 1 ≤ a ∧ a < 2 :=
by sorry

end main_theorem_l3297_329721


namespace component_usage_impossibility_l3297_329784

theorem component_usage_impossibility (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧
    (2 * x + y = 2 * p + q + 1) ∧
    (y + z = q + r) := by
  sorry

end component_usage_impossibility_l3297_329784


namespace mans_speed_against_current_l3297_329724

/-- Calculates the man's speed against the current with wind and waves -/
def speed_against_current_with_wind_and_waves 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (wind_effect : ℝ) 
  (wave_effect : ℝ) : ℝ :=
  speed_with_current - current_speed - wind_effect - current_speed - wave_effect

/-- Theorem stating the man's speed against the current with wind and waves -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (wind_effect : ℝ) 
  (wave_effect : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : current_speed = 5) 
  (h3 : wind_effect = 2) 
  (h4 : wave_effect = 1) : 
  speed_against_current_with_wind_and_waves speed_with_current current_speed wind_effect wave_effect = 7 := by
  sorry

end mans_speed_against_current_l3297_329724


namespace max_M_inequality_l3297_329720

theorem max_M_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (∃ (M : ℝ), ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
    a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(a-b)*(b-c)*(c-a)) ↔ 
  (M ≤ Real.sqrt (9 + 6 * Real.sqrt 3)) :=
by sorry

end max_M_inequality_l3297_329720


namespace two_different_buttons_l3297_329765

/-- Represents the size of a button -/
inductive Size
| Big
| Small

/-- Represents the color of a button -/
inductive Color
| White
| Black

/-- Represents a button with a size and color -/
structure Button :=
  (size : Size)
  (color : Color)

/-- A set of buttons satisfying the given conditions -/
structure ButtonSet :=
  (buttons : Set Button)
  (has_big : ∃ b ∈ buttons, b.size = Size.Big)
  (has_small : ∃ b ∈ buttons, b.size = Size.Small)
  (has_white : ∃ b ∈ buttons, b.color = Color.White)
  (has_black : ∃ b ∈ buttons, b.color = Color.Black)

/-- Theorem stating that there exist two buttons with different size and color -/
theorem two_different_buttons (bs : ButtonSet) :
  ∃ (b1 b2 : Button), b1 ∈ bs.buttons ∧ b2 ∈ bs.buttons ∧
  b1.size ≠ b2.size ∧ b1.color ≠ b2.color :=
sorry

end two_different_buttons_l3297_329765


namespace min_omega_l3297_329754

theorem min_omega (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin (ω * x)) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), f x = -2) →
  ω ≥ 3/2 ∧ ∀ ω' > 0, (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) ≥ -2) →
    (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) = -2) → ω' ≥ 3/2 :=
by sorry

end min_omega_l3297_329754


namespace system_solution_l3297_329739

theorem system_solution (x y z : ℝ) : 
  (Real.sqrt (2 * x^2 + 2) = y + 1 ∧
   Real.sqrt (2 * y^2 + 2) = z + 1 ∧
   Real.sqrt (2 * z^2 + 2) = x + 1) →
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end system_solution_l3297_329739


namespace rain_probability_rain_probability_in_both_areas_l3297_329786

theorem rain_probability (P₁ P₂ : ℝ) 
  (h₁ : 0 < P₁ ∧ P₁ < 1) 
  (h₂ : 0 < P₂ ∧ P₂ < 1) 
  (h_independent : True) -- Representing independence condition
  : ℝ :=
(1 - P₁) * (1 - P₂)

theorem rain_probability_in_both_areas (P₁ P₂ : ℝ) 
  (h₁ : 0 < P₁ ∧ P₁ < 1) 
  (h₂ : 0 < P₂ ∧ P₂ < 1) 
  (h_independent : True) -- Representing independence condition
  : rain_probability P₁ P₂ h₁ h₂ h_independent = (1 - P₁) * (1 - P₂) :=
sorry

end rain_probability_rain_probability_in_both_areas_l3297_329786


namespace evaluate_expression_l3297_329701

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 5
  (2 * x - a + 4) = a + 14 := by sorry

end evaluate_expression_l3297_329701


namespace pure_imaginary_fraction_l3297_329749

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * ((a + 2 * Complex.I) / (1 + Complex.I))).re = ((a + 2 * Complex.I) / (1 + Complex.I)).re → a = -2 := by
  sorry

end pure_imaginary_fraction_l3297_329749


namespace min_equal_fruits_cost_l3297_329763

/-- Represents a package of fruits -/
structure Package where
  apples : ℕ
  oranges : ℕ
  cost : ℕ

/-- The two available packages -/
def package1 : Package := ⟨3, 12, 5⟩
def package2 : Package := ⟨20, 5, 13⟩

/-- The minimum nonzero amount to spend for equal apples and oranges -/
def minEqualFruitsCost : ℕ := 64

/-- Theorem stating the minimum cost for equal fruits -/
theorem min_equal_fruits_cost :
  ∀ x y : ℕ,
    x * package1.apples + y * package2.apples = x * package1.oranges + y * package2.oranges →
    x > 0 ∨ y > 0 →
    x * package1.cost + y * package2.cost ≥ minEqualFruitsCost :=
sorry

end min_equal_fruits_cost_l3297_329763
