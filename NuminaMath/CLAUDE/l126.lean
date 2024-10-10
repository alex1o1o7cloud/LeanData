import Mathlib

namespace scarlet_earrings_cost_l126_12683

/-- Calculates the cost of earrings given initial savings, necklace cost, and remaining money --/
def earrings_cost (initial_savings : ℕ) (necklace_cost : ℕ) (remaining : ℕ) : ℕ :=
  initial_savings - necklace_cost - remaining

/-- Proves that the cost of earrings is 23 given the problem conditions --/
theorem scarlet_earrings_cost :
  let initial_savings : ℕ := 80
  let necklace_cost : ℕ := 48
  let remaining : ℕ := 9
  earrings_cost initial_savings necklace_cost remaining = 23 := by
  sorry

end scarlet_earrings_cost_l126_12683


namespace correct_seat_notation_l126_12642

/-- Represents a cinema seat notation -/
def CinemaSeat := ℕ × ℕ

/-- Converts a row and seat number to cinema seat notation -/
def toSeatNotation (row : ℕ) (seat : ℕ) : CinemaSeat := (row, seat)

theorem correct_seat_notation :
  toSeatNotation 2 5 = (2, 5) := by sorry

end correct_seat_notation_l126_12642


namespace remaining_money_l126_12663

def initial_amount : ℕ := 400
def dress_count : ℕ := 5
def dress_price : ℕ := 20
def pants_count : ℕ := 3
def pants_price : ℕ := 12
def jacket_count : ℕ := 4
def jacket_price : ℕ := 30
def transportation_cost : ℕ := 5

def total_expense : ℕ := 
  dress_count * dress_price + 
  pants_count * pants_price + 
  jacket_count * jacket_price + 
  transportation_cost

theorem remaining_money : 
  initial_amount - total_expense = 139 := by
  sorry

end remaining_money_l126_12663


namespace certain_number_proof_l126_12660

theorem certain_number_proof (y x : ℝ) (h1 : y > 0) (h2 : (1/2) * Real.sqrt x = y^(1/3)) (h3 : y = 64) : x = 64 := by
  sorry

end certain_number_proof_l126_12660


namespace complex_norm_squared_l126_12638

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 6*I) : 
  Complex.normSq z = 13/2 := by
sorry

end complex_norm_squared_l126_12638


namespace cricket_average_l126_12645

theorem cricket_average (initial_innings : ℕ) (next_innings_runs : ℕ) (average_increase : ℕ) 
  (h1 : initial_innings = 20)
  (h2 : next_innings_runs = 137)
  (h3 : average_increase = 5) :
  ∃ (initial_average : ℕ),
    (initial_innings * initial_average + next_innings_runs) / (initial_innings + 1) = 
    initial_average + average_increase ∧ initial_average = 32 := by
  sorry

end cricket_average_l126_12645


namespace constant_term_expansion_l126_12617

theorem constant_term_expansion (x : ℝ) (x_neq_0 : x ≠ 0) :
  ∃ (c : ℝ), (x - 1/x)^4 = c + (terms_with_x : ℝ) ∧ c = 6 :=
sorry

end constant_term_expansion_l126_12617


namespace even_decreasing_inequality_l126_12624

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x ∧ x < y → f y < f x

theorem even_decreasing_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_dec : decreasing_nonneg f) :
  ∀ m : ℝ, f (1 - m) < f m ↔ m < (1 : ℝ) / 2 := by sorry

end even_decreasing_inequality_l126_12624


namespace at_least_one_first_class_l126_12605

def total_parts : ℕ := 20
def first_class_parts : ℕ := 16
def second_class_parts : ℕ := 4
def selections : ℕ := 3

theorem at_least_one_first_class :
  (Nat.choose first_class_parts 1 * Nat.choose second_class_parts 2) +
  (Nat.choose first_class_parts 2 * Nat.choose second_class_parts 1) +
  (Nat.choose first_class_parts 3) =
  Nat.choose total_parts selections - Nat.choose second_class_parts selections :=
sorry

end at_least_one_first_class_l126_12605


namespace shaded_region_area_is_zero_l126_12655

/-- A rectangle with height 8 and width 12 -/
structure Rectangle where
  height : ℝ
  width : ℝ
  height_eq : height = 8
  width_eq : width = 12

/-- A right triangle with base 12 and height 8 -/
structure RightTriangle where
  base : ℝ
  height : ℝ
  base_eq : base = 12
  height_eq : height = 8

/-- The shaded region formed by the segment and parts of the rectangle and triangle -/
def shadedRegion (r : Rectangle) (t : RightTriangle) : ℝ := sorry

/-- The theorem stating that the area of the shaded region is 0 -/
theorem shaded_region_area_is_zero (r : Rectangle) (t : RightTriangle) :
  shadedRegion r t = 0 := by sorry

end shaded_region_area_is_zero_l126_12655


namespace tallest_player_height_calculation_l126_12629

/-- The height of the tallest player on a basketball team, given the height of the shortest player and the difference in height between the tallest and shortest players. -/
def tallest_player_height (shortest_player_height : ℝ) (height_difference : ℝ) : ℝ :=
  shortest_player_height + height_difference

/-- Theorem stating that given a shortest player height of 68.25 inches and a height difference of 9.5 inches, the tallest player's height is 77.75 inches. -/
theorem tallest_player_height_calculation :
  tallest_player_height 68.25 9.5 = 77.75 := by
  sorry

end tallest_player_height_calculation_l126_12629


namespace min_value_expression_l126_12693

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 4) * (b^2 + 4*b + 4) * (c^2 + 4*c + 4) / (a*b*c) ≥ 64 ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀^2 + 4*a₀ + 4) * (b₀^2 + 4*b₀ + 4) * (c₀^2 + 4*c₀ + 4) / (a₀*b₀*c₀) = 64) :=
by sorry

end min_value_expression_l126_12693


namespace five_balls_three_boxes_l126_12675

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 21 := by sorry

end five_balls_three_boxes_l126_12675


namespace first_three_digits_of_expression_l126_12698

-- Define the expression
def expression : ℝ := (10^2003 + 1)^(11/9)

-- Define a function to get the first three digits after the decimal point
def firstThreeDecimalDigits (x : ℝ) : ℕ × ℕ × ℕ := sorry

-- Theorem statement
theorem first_three_digits_of_expression :
  firstThreeDecimalDigits expression = (2, 2, 2) := by sorry

end first_three_digits_of_expression_l126_12698


namespace imaginary_part_of_complex_reciprocal_l126_12694

theorem imaginary_part_of_complex_reciprocal (z : ℂ) (h : z = 1 + 2*I) : 
  Complex.im (z⁻¹) = -2/5 := by
  sorry

end imaginary_part_of_complex_reciprocal_l126_12694


namespace jims_remaining_distance_l126_12635

/-- Given a total journey distance and the distance already driven, 
    calculate the remaining distance to drive. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: For Jim's journey of 1200 miles, having driven 923 miles, 
    the remaining distance is 277 miles. -/
theorem jims_remaining_distance : 
  remaining_distance 1200 923 = 277 := by
  sorry

end jims_remaining_distance_l126_12635


namespace max_value_xy_l126_12615

theorem max_value_xy (x y : ℝ) (h1 : x * y + 6 = x + 9 * y) (h2 : y < 1) :
  (∃ (z : ℝ), ∀ (a b : ℝ), a * b + 6 = a + 9 * b → b < 1 → (a + 3) * (b + 1) ≤ z) ∧
  (∃ (x' y' : ℝ), x' * y' + 6 = x' + 9 * y' ∧ y' < 1 ∧ (x' + 3) * (y' + 1) = 27 - 12 * Real.sqrt 2) :=
by sorry

end max_value_xy_l126_12615


namespace problem_statement_l126_12627

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 12) : 5 * (x + 4) = 10 := by
  sorry

end problem_statement_l126_12627


namespace missing_fraction_sum_l126_12662

theorem missing_fraction_sum (x : ℚ) : 
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-9/20) + x = 45/100 → x = 27/60 := by
  sorry

end missing_fraction_sum_l126_12662


namespace problem_solution_l126_12626

theorem problem_solution :
  (∃ x : ℝ, x > 0 ∧ x + 4/x = 6) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 2/x + 1/y ≥ 8) := by
  sorry

end problem_solution_l126_12626


namespace ten_tall_flags_made_l126_12669

/-- Calculates the number of tall flags made given the total fabric area,
    dimensions of each flag type, number of square and wide flags made,
    and remaining fabric area. -/
def tall_flags_made (total_fabric : ℕ) (square_side : ℕ) (wide_length wide_width : ℕ)
  (tall_length tall_width : ℕ) (square_flags_made wide_flags_made : ℕ)
  (fabric_left : ℕ) : ℕ :=
  let square_area := square_side * square_side
  let wide_area := wide_length * wide_width
  let tall_area := tall_length * tall_width
  let used_area := square_area * square_flags_made + wide_area * wide_flags_made
  let tall_flags_area := total_fabric - used_area - fabric_left
  tall_flags_area / tall_area

/-- Theorem stating that given the problem conditions, 10 tall flags were made. -/
theorem ten_tall_flags_made :
  tall_flags_made 1000 4 5 3 3 5 16 20 294 = 10 := by
  sorry


end ten_tall_flags_made_l126_12669


namespace unusual_arithmetic_l126_12641

/-- In a country with unusual arithmetic, given that 1/3 of 4 equals 6,
    if 1/6 of a number is 15, then that number is 405. -/
theorem unusual_arithmetic (country_multiplier : ℚ) : 
  (1/3 : ℚ) * 4 * country_multiplier = 6 →
  ∃ (x : ℚ), (1/6 : ℚ) * x * country_multiplier = 15 ∧ x * country_multiplier = 405 :=
by
  sorry


end unusual_arithmetic_l126_12641


namespace time_difference_between_arrivals_l126_12666

/-- Represents the problem of calculating the time difference between arrivals of a car and a minivan --/
theorem time_difference_between_arrivals 
  (car_speed : ℝ) 
  (minivan_speed : ℝ) 
  (passing_time_before_arrival : ℝ) :
  car_speed = 40 →
  minivan_speed = 50 →
  passing_time_before_arrival = 1/6 →
  ∃ (time_difference : ℝ),
    time_difference = passing_time_before_arrival - (car_speed * passing_time_before_arrival) / minivan_speed ∧
    time_difference * 60 = 2 := by
  sorry


end time_difference_between_arrivals_l126_12666


namespace probability_of_pair_l126_12649

def deck_size : ℕ := 60
def cards_per_number : ℕ := 6
def numbers_in_deck : ℕ := 10
def pairs_removed : ℕ := 3

def remaining_cards : ℕ := deck_size - 2 * pairs_removed

def ways_to_choose_two : ℕ := remaining_cards.choose 2

def full_ranks : ℕ := numbers_in_deck - pairs_removed
def cards_in_full_rank : ℕ := cards_per_number

def pairs_from_full_ranks : ℕ := full_ranks * cards_in_full_rank.choose 2

def affected_ranks : ℕ := pairs_removed
def cards_in_affected_rank : ℕ := cards_per_number - 2

def pairs_from_affected_ranks : ℕ := affected_ranks * cards_in_affected_rank.choose 2

def total_pairs : ℕ := pairs_from_full_ranks + pairs_from_affected_ranks

theorem probability_of_pair (h : ways_to_choose_two = 1431 ∧ total_pairs = 123) :
  (total_pairs : ℚ) / ways_to_choose_two = 123 / 1431 :=
sorry

end probability_of_pair_l126_12649


namespace perpendicular_vectors_l126_12612

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (1, 2) and b = (x, 1), if they are perpendicular, then x = -2 -/
theorem perpendicular_vectors (x : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  perpendicular a b → x = -2 := by sorry

end perpendicular_vectors_l126_12612


namespace polynomial_factorization_l126_12608

theorem polynomial_factorization (a : ℝ) : a^3 + 10*a^2 + 25*a = a*(a+5)^2 := by
  sorry

end polynomial_factorization_l126_12608


namespace pythagorean_orthogonal_sum_zero_l126_12661

theorem pythagorean_orthogonal_sum_zero
  (a b c d : ℝ)
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : a*c + b*d = 0) :
  a*b + c*d = 0 := by
sorry

end pythagorean_orthogonal_sum_zero_l126_12661


namespace min_value_of_expression_l126_12614

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 81/x^3 ≥ 18 * Real.sqrt 3 ∧
  ∃ y > 0, y^2 + 12*y + 81/y^3 = 18 * Real.sqrt 3 :=
by sorry

end min_value_of_expression_l126_12614


namespace complex_modulus_one_l126_12692

theorem complex_modulus_one (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a ∧ a < (n + 1) / (n - 1 : ℝ)) 
  (h3 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by sorry

end complex_modulus_one_l126_12692


namespace picture_distribution_l126_12673

theorem picture_distribution (total_pictures : ℕ) 
  (first_albums : ℕ) (first_album_capacity : ℕ) (remaining_albums : ℕ) :
  total_pictures = 100 →
  first_albums = 2 →
  first_album_capacity = 15 →
  remaining_albums = 3 →
  ∃ (pictures_per_remaining_album : ℕ) (leftover : ℕ),
    pictures_per_remaining_album = 23 ∧
    leftover = 1 ∧
    total_pictures = 
      (first_albums * first_album_capacity) + 
      (remaining_albums * pictures_per_remaining_album) + 
      leftover :=
by sorry

end picture_distribution_l126_12673


namespace average_of_three_numbers_l126_12668

theorem average_of_three_numbers (y : ℝ) : (15 + 18 + y) / 3 = 21 → y = 30 := by
  sorry

end average_of_three_numbers_l126_12668


namespace rps_win_on_sixth_game_l126_12603

/-- The probability of a tie in a single game of Rock Paper Scissors -/
def tie_prob : ℚ := 1 / 3

/-- The probability of not tying (i.e., someone wins) in a single game -/
def win_prob : ℚ := 1 - tie_prob

/-- The number of consecutive ties before the winning game -/
def num_ties : ℕ := 5

theorem rps_win_on_sixth_game : 
  tie_prob ^ num_ties * win_prob = 2 / 729 := by sorry

end rps_win_on_sixth_game_l126_12603


namespace teachers_students_arrangement_l126_12678

def num_students : ℕ := 4
def num_teachers : ℕ := 3

-- Function to calculate permutations
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

-- Theorem statement
theorem teachers_students_arrangement :
  permutations num_teachers num_teachers * permutations num_students num_students = 144 :=
by sorry

end teachers_students_arrangement_l126_12678


namespace flower_pot_profit_equation_l126_12651

/-- Represents a flower pot system with variable number of plants and profit. -/
structure FlowerPot where
  initial_plants : ℕ
  initial_profit_per_plant : ℝ
  profit_decrease_per_plant : ℝ

/-- Calculates the total profit for a given number of additional plants. -/
def total_profit (fp : FlowerPot) (additional_plants : ℝ) : ℝ :=
  (fp.initial_plants + additional_plants) * 
  (fp.initial_profit_per_plant - additional_plants * fp.profit_decrease_per_plant)

/-- Theorem stating that the equation (x+3)(10-x)=40 correctly represents
    the total profit of 40 yuan for the given flower pot system. -/
theorem flower_pot_profit_equation (x : ℝ) : 
  let fp : FlowerPot := ⟨3, 10, 1⟩
  total_profit fp x = 40 ↔ (x + 3) * (10 - x) = 40 := by
  sorry

end flower_pot_profit_equation_l126_12651


namespace average_speed_is_55_l126_12677

-- Define the problem parameters
def initial_reading : ℕ := 2332
def final_reading : ℕ := 2772
def total_time : ℕ := 8

-- Define the average speed calculation
def average_speed : ℚ := (final_reading - initial_reading : ℚ) / total_time

-- Theorem statement
theorem average_speed_is_55 : average_speed = 55 := by
  sorry

end average_speed_is_55_l126_12677


namespace expression_simplification_l126_12667

theorem expression_simplification :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l126_12667


namespace hypotenuse_length_of_right_isosceles_triangle_l126_12625

/-- A right triangle with a 45-45-90 degree angle configuration. -/
structure RightIsoscelesTriangle where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The length of the hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The leg is √2 times the inradius -/
  leg_eq : leg = Real.sqrt 2 * inradius
  /-- The hypotenuse is √2 times the leg -/
  hypotenuse_eq : hypotenuse = Real.sqrt 2 * leg

/-- 
If a right isosceles triangle has an inscribed circle with radius 4,
then its hypotenuse has length 8.
-/
theorem hypotenuse_length_of_right_isosceles_triangle 
  (t : RightIsoscelesTriangle) (h : t.inradius = 4) : 
  t.hypotenuse = 8 := by
  sorry

end hypotenuse_length_of_right_isosceles_triangle_l126_12625


namespace novelist_writing_speed_l126_12613

/-- Calculates the average writing speed given total words, total hours, and break hours. -/
def average_writing_speed (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) : ℚ :=
  total_words / (total_hours - break_hours)

/-- Proves that the average writing speed is 625 words per hour for the given conditions. -/
theorem novelist_writing_speed :
  average_writing_speed 50000 100 20 = 625 := by
  sorry

end novelist_writing_speed_l126_12613


namespace product_of_complements_lower_bound_l126_12643

theorem product_of_complements_lower_bound 
  (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) 
  (h₄ : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end product_of_complements_lower_bound_l126_12643


namespace weak_arithmetic_progression_of_three_weak_arithmetic_progression_in_large_subset_l126_12665

-- Definition of weak arithmetic progression
def is_weak_arithmetic_progression (a : Fin M → ℝ) : Prop :=
  ∃ x : Fin (M + 1) → ℝ, ∀ i : Fin M, x i ≤ a i ∧ a i < x (i + 1)

-- Part (a)
theorem weak_arithmetic_progression_of_three (a₁ a₂ a₃ : ℝ) (h : a₁ < a₂ ∧ a₂ < a₃) :
  is_weak_arithmetic_progression (fun i => [a₁, a₂, a₃].get i) :=
sorry

-- Part (b)
theorem weak_arithmetic_progression_in_large_subset :
  ∀ (S : Finset (Fin 1000)),
    S.card ≥ 730 →
    ∃ (a : Fin 10 → Fin 1000), (∀ i, a i ∈ S) ∧ is_weak_arithmetic_progression (fun i => (a i : ℝ)) :=
sorry

end weak_arithmetic_progression_of_three_weak_arithmetic_progression_in_large_subset_l126_12665


namespace unique_quadratic_solution_l126_12604

theorem unique_quadratic_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 36 * x + 12 = 0) : 
  ∃ x, a * x^2 + 36 * x + 12 = 0 ∧ x = -2/3 := by
sorry

end unique_quadratic_solution_l126_12604


namespace prize_orders_count_l126_12621

/-- Represents a tournament with n players -/
structure Tournament (n : ℕ) where
  players : Fin n

/-- The playoff structure for a 6-player tournament -/
def playoff_structure (t : Tournament 6) : Prop :=
  ∃ (order : Fin 6 → Fin 6),
    -- Each player gets a unique position
    Function.Injective order ∧
    -- The structure follows the described playoff

    -- #6 vs #5, loser gets 6th place
    (order 5 = 6 ∨ order 4 = 6) ∧
    
    -- Winner of #6 vs #5 plays against #4
    (order 5 ≠ 6 → order 5 < order 3) ∧
    (order 4 ≠ 6 → order 4 < order 3) ∧
    
    -- Subsequent matches
    order 3 < order 2 ∧
    order 2 < order 1 ∧
    order 1 < order 0

/-- The number of possible prize orders in a 6-player tournament with the given playoff structure -/
def num_prize_orders (t : Tournament 6) : ℕ := 32

/-- Theorem stating that the number of possible prize orders is 32 -/
theorem prize_orders_count (t : Tournament 6) :
  playoff_structure t → num_prize_orders t = 32 := by
  sorry


end prize_orders_count_l126_12621


namespace fraction_meaningful_l126_12676

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by
sorry

end fraction_meaningful_l126_12676


namespace first_half_time_l126_12659

/-- Represents the time taken for the elevator to travel down different sections of floors -/
structure ElevatorTime where
  firstHalf : ℕ
  secondQuarter : ℕ
  thirdQuarter : ℕ

/-- The total number of floors the elevator needs to travel -/
def totalFloors : ℕ := 20

/-- The time taken per floor for the second quarter of the journey -/
def timePerFloorSecondQuarter : ℕ := 5

/-- The time taken per floor for the third quarter of the journey -/
def timePerFloorThirdQuarter : ℕ := 16

/-- The total time taken for the elevator to reach the bottom floor -/
def totalTime : ℕ := 120

/-- Calculates the time taken for the second quarter of the journey -/
def secondQuarterTime : ℕ := (totalFloors / 4) * timePerFloorSecondQuarter

/-- Calculates the time taken for the third quarter of the journey -/
def thirdQuarterTime : ℕ := (totalFloors / 4) * timePerFloorThirdQuarter

/-- Theorem stating that the time taken for the first half of the journey is 15 minutes -/
theorem first_half_time (t : ElevatorTime) : 
  t.firstHalf = 15 ∧ 
  t.secondQuarter = secondQuarterTime ∧ 
  t.thirdQuarter = thirdQuarterTime ∧ 
  t.firstHalf + t.secondQuarter + t.thirdQuarter = totalTime := by
  sorry

end first_half_time_l126_12659


namespace rectangular_solid_diagonal_l126_12620

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : a * b = 6)
  (h2 : a * c = 8)
  (h3 : b * c = 12)
  : Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry

end rectangular_solid_diagonal_l126_12620


namespace double_series_convergence_l126_12690

/-- The double series ∑_{m=1}^∞ ∑_{n=1}^∞ 1/(m(m+n+2)) converges to 1 -/
theorem double_series_convergence :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * (m + n + 2))) = 1 := by
  sorry

end double_series_convergence_l126_12690


namespace max_value_abc_l126_12610

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683/472392 :=
sorry

end max_value_abc_l126_12610


namespace tile_difference_after_border_l126_12646

/-- Proves that the difference between green and blue tiles after adding a border is 11 -/
theorem tile_difference_after_border (initial_blue : ℕ) (initial_green : ℕ) 
  (sides : ℕ) (tiles_per_side : ℕ) : 
  initial_blue = 13 → initial_green = 6 → sides = 6 → tiles_per_side = 3 →
  (initial_green + sides * tiles_per_side) - initial_blue = 11 := by
  sorry

end tile_difference_after_border_l126_12646


namespace good_pair_exists_l126_12670

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2 := by
  sorry

end good_pair_exists_l126_12670


namespace B_interval_when_m_less_than_half_A_union_B_equals_A_iff_l126_12619

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m+1)*x + 2*m < 0}

-- Statement 1: When m < 1/2, B = (2m, 1)
theorem B_interval_when_m_less_than_half (m : ℝ) (h : m < 1/2) :
  B m = Set.Ioo (2*m) 1 :=
sorry

-- Statement 2: A ∪ B = A if and only if -1/2 ≤ m ≤ 1
theorem A_union_B_equals_A_iff (m : ℝ) :
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 :=
sorry

end B_interval_when_m_less_than_half_A_union_B_equals_A_iff_l126_12619


namespace water_fee_calculation_l126_12600

-- Define the water fee structure
structure WaterFee where
  a : ℝ  -- rate for usage ≤ 6m³
  b : ℝ  -- rate for usage > 6m³

-- Define the water usage data
structure WaterUsage where
  usage : ℝ
  fee : ℝ

-- Theorem statement
theorem water_fee_calculation (wf : WaterFee) (march : WaterUsage) (april : WaterUsage)
  (h1 : march.usage = 5 ∧ march.fee = 7.5)
  (h2 : april.usage = 9 ∧ april.fee = 27)
  (h3 : march.usage ≤ 6 ∧ april.usage > 6) :
  (∀ x > 6, wf.a * 6 + wf.b * (x - 6) = 6 * x - 27) ∧
  (∃ x > 6, wf.a * 6 + wf.b * (x - 6) = 39 ∧ x = 11) := by
  sorry

end water_fee_calculation_l126_12600


namespace sum_of_coordinates_A_l126_12682

/-- Given three points A, B, and C in a 2D plane satisfying certain conditions,
    prove that the sum of the coordinates of A is 8.5. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 8) →
  C = (5, 14) →
  A.1 + A.2 = 8.5 := by
sorry

end sum_of_coordinates_A_l126_12682


namespace journey_mpg_l126_12602

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer : ℕ) (final_odometer : ℕ) (odometer_error : ℕ) 
                (initial_fuel : ℕ) (refill1 : ℕ) (refill2 : ℕ) : ℚ :=
  let actual_distance := (final_odometer - odometer_error) - initial_odometer
  let total_fuel := initial_fuel + refill1 + refill2
  (actual_distance : ℚ) / total_fuel

/-- Theorem stating that the average miles per gallon for the given journey is 20.8 -/
theorem journey_mpg : 
  let mpg := average_mpg 68300 69350 10 10 15 25
  ∃ (n : ℕ), (n : ℚ) / 10 = mpg ∧ n = 208 :=
by sorry

end journey_mpg_l126_12602


namespace C_is_integer_l126_12628

/-- Represents a number consisting of k ones -/
def ones (k : ℕ) : ℕ :=
  if k = 0 then 0 else 10^(k-1) + ones (k-1)

/-- The factorial-like function [n]! -/
def special_factorial : ℕ → ℕ
  | 0 => 1
  | n+1 => (ones (n+1)) * special_factorial n

/-- The combinatorial-like function C[m, n] -/
def C (m n : ℕ) : ℚ :=
  (special_factorial (m + n)) / ((special_factorial m) * (special_factorial n))

/-- Theorem stating that C[m, n] is always an integer -/
theorem C_is_integer (m n : ℕ) : ∃ k : ℤ, C m n = k :=
  sorry

end C_is_integer_l126_12628


namespace at_operation_four_three_l126_12695

def at_operation (a b : ℝ) : ℝ := 4 * a^2 - 2 * b

theorem at_operation_four_three : at_operation 4 3 = 58 := by
  sorry

end at_operation_four_three_l126_12695


namespace dangerous_animals_remaining_is_231_l126_12636

/-- The number of dangerous animals remaining in the swamp after some animals migrate --/
def dangerous_animals_remaining : ℕ :=
  let initial_crocodiles : ℕ := 42
  let initial_alligators : ℕ := 35
  let initial_vipers : ℕ := 10
  let initial_water_moccasins : ℕ := 28
  let initial_cottonmouth_snakes : ℕ := 15
  let initial_piranha_fish : ℕ := 120
  let migrating_crocodiles : ℕ := 9
  let migrating_alligators : ℕ := 7
  let migrating_vipers : ℕ := 3
  let total_initial := initial_crocodiles + initial_alligators + initial_vipers + 
                       initial_water_moccasins + initial_cottonmouth_snakes + initial_piranha_fish
  let total_migrating := migrating_crocodiles + migrating_alligators + migrating_vipers
  total_initial - total_migrating

/-- Theorem stating that the number of dangerous animals remaining in the swamp is 231 --/
theorem dangerous_animals_remaining_is_231 : dangerous_animals_remaining = 231 := by
  sorry

end dangerous_animals_remaining_is_231_l126_12636


namespace exactly_two_favor_policy_l126_12630

/-- The probability of a person favoring the policy -/
def p : ℝ := 0.6

/-- The number of people surveyed -/
def n : ℕ := 5

/-- The number of people who favor the policy in the desired outcome -/
def k : ℕ := 2

/-- The probability of exactly k out of n people favoring the policy -/
def prob_exactly_k (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_favor_policy :
  prob_exactly_k p n k = 0.2304 := by sorry

end exactly_two_favor_policy_l126_12630


namespace inequalities_hold_l126_12691

theorem inequalities_hold (a b : ℝ) (h : a * b > 0) :
  (2 * (a^2 + b^2) ≥ (a + b)^2) ∧
  (b / a + a / b ≥ 2) ∧
  ((a + 1 / a) * (b + 1 / b) ≥ 4) := by
  sorry

end inequalities_hold_l126_12691


namespace height_weight_most_suitable_for_regression_l126_12697

-- Define the types of variables
inductive Variable
| CircleArea
| CircleRadius
| Height
| Weight
| ColorBlindness
| Gender
| AcademicPerformance

-- Define a pair of variables
structure VariablePair where
  var1 : Variable
  var2 : Variable

-- Define the types of relationships between variables
inductive Relationship
| Functional
| Correlated
| Unrelated

-- Function to determine the relationship between a pair of variables
def relationshipBetween (pair : VariablePair) : Relationship :=
  match pair with
  | ⟨Variable.CircleArea, Variable.CircleRadius⟩ => Relationship.Functional
  | ⟨Variable.ColorBlindness, Variable.Gender⟩ => Relationship.Unrelated
  | ⟨Variable.Height, Variable.AcademicPerformance⟩ => Relationship.Unrelated
  | ⟨Variable.Height, Variable.Weight⟩ => Relationship.Correlated
  | _ => Relationship.Unrelated  -- Default case

-- Function to determine if a pair is suitable for regression analysis
def suitableForRegression (pair : VariablePair) : Prop :=
  relationshipBetween pair = Relationship.Correlated

-- Theorem stating that height and weight is the most suitable pair for regression
theorem height_weight_most_suitable_for_regression :
  suitableForRegression ⟨Variable.Height, Variable.Weight⟩ ∧
  ¬suitableForRegression ⟨Variable.CircleArea, Variable.CircleRadius⟩ ∧
  ¬suitableForRegression ⟨Variable.ColorBlindness, Variable.Gender⟩ ∧
  ¬suitableForRegression ⟨Variable.Height, Variable.AcademicPerformance⟩ :=
by sorry

end height_weight_most_suitable_for_regression_l126_12697


namespace triangle_inequality_theorem_l126_12679

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  can_form_triangle 8 6 4 ∧
  ¬can_form_triangle 2 4 6 ∧
  ¬can_form_triangle 14 6 7 ∧
  ¬can_form_triangle 2 3 6 :=
sorry

end triangle_inequality_theorem_l126_12679


namespace extra_fruits_theorem_l126_12631

/-- Represents the quantities of fruits ordered and wanted --/
structure FruitQuantities where
  redAppleOrdered : Nat
  redAppleWanted : Nat
  greenAppleOrdered : Nat
  greenAppleWanted : Nat
  orangeOrdered : Nat
  orangeWanted : Nat
  bananaOrdered : Nat
  bananaWanted : Nat

/-- Represents the extra fruits for each type --/
structure ExtraFruits where
  redApple : Nat
  greenApple : Nat
  orange : Nat
  banana : Nat

/-- Calculates the extra fruits given the ordered and wanted quantities --/
def calculateExtraFruits (quantities : FruitQuantities) : ExtraFruits :=
  { redApple := quantities.redAppleOrdered - quantities.redAppleWanted,
    greenApple := quantities.greenAppleOrdered - quantities.greenAppleWanted,
    orange := quantities.orangeOrdered - quantities.orangeWanted,
    banana := quantities.bananaOrdered - quantities.bananaWanted }

/-- The theorem stating that the calculated extra fruits match the expected values --/
theorem extra_fruits_theorem (quantities : FruitQuantities) 
  (h : quantities = { redAppleOrdered := 6, redAppleWanted := 5,
                      greenAppleOrdered := 15, greenAppleWanted := 8,
                      orangeOrdered := 10, orangeWanted := 6,
                      bananaOrdered := 8, bananaWanted := 7 }) :
  calculateExtraFruits quantities = { redApple := 1, greenApple := 7, orange := 4, banana := 1 } := by
  sorry

end extra_fruits_theorem_l126_12631


namespace simplify_expression_l126_12686

theorem simplify_expression (y : ℝ) : 4*y + 9*y^2 + 8 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 5 := by
  sorry

end simplify_expression_l126_12686


namespace sum_odd_integers_9_to_49_l126_12685

/-- The sum of odd integers from 9 through 49, inclusive, is 609. -/
theorem sum_odd_integers_9_to_49 : 
  (Finset.range 21).sum (fun i => 2*i + 9) = 609 := by
  sorry

end sum_odd_integers_9_to_49_l126_12685


namespace vector_to_line_l126_12658

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if a vector is on a parametric line -/
def vector_on_line (v : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = v.1 ∧ l.y t = v.2

/-- Check if two vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_to_line (l : ParametricLine) (v : ℝ × ℝ) :
  l.x t = 3 * t + 5 →
  l.y t = 2 * t + 1 →
  vector_on_line v l →
  parallel v (3, 2) →
  v = (21/2, 7) := by sorry

end vector_to_line_l126_12658


namespace statement_A_statement_B_statement_C_statement_D_l126_12637

/-- Given polynomials M and N -/
def M (a x : ℝ) : ℝ := a * x^2 - 2*x + 3
def N (b x : ℝ) : ℝ := x^2 - b*x - 1

/-- Statement A -/
theorem statement_A : ∃ x : ℝ, M 1 x - N 2 x ≠ -4*x + 2 := by sorry

/-- Statement B -/
theorem statement_B : ∀ x : ℝ, M (-1) x + N 2 x = -4*x + 2 := by sorry

/-- Statement C -/
theorem statement_C : ∃ x : ℝ, x ≠ 1 ∧ |M 1 x - N 4 x| = 6 := by sorry

/-- Statement D -/
theorem statement_D : (∀ x : ℝ, ∃ c : ℝ, 2 * M a x + N b x = c) → a = -1/2 ∧ b = -4 := by sorry

end statement_A_statement_B_statement_C_statement_D_l126_12637


namespace m_div_125_eq_2_pow_124_l126_12653

/-- The smallest positive integer that is a multiple of 125 and has exactly 125 positive integral divisors -/
def m : ℕ := sorry

/-- m is a multiple of 125 -/
axiom m_multiple_of_125 : 125 ∣ m

/-- m has exactly 125 positive integral divisors -/
axiom m_divisors_count : (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 125

/-- m is the smallest such positive integer -/
axiom m_is_smallest : ∀ k : ℕ, k < m → ¬(125 ∣ k ∧ (Finset.filter (· ∣ k) (Finset.range (k + 1))).card = 125)

/-- The main theorem to prove -/
theorem m_div_125_eq_2_pow_124 : m / 125 = 2^124 := by sorry

end m_div_125_eq_2_pow_124_l126_12653


namespace probability_no_three_consecutive_ones_l126_12687

/-- Represents a sequence of 0s and 1s of length n that doesn't contain three consecutive 1s -/
def ValidSequence (n : ℕ) := Fin n → Fin 2

/-- The number of valid sequences of length n -/
def countValidSequences : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => countValidSequences (n + 2) + countValidSequences (n + 1) + countValidSequences n

/-- The total number of possible binary sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

theorem probability_no_three_consecutive_ones :
  (countValidSequences 15 : ℚ) / (totalSequences 15 : ℚ) = 10609 / 32768 := by
  sorry

#eval countValidSequences 15 + totalSequences 15

end probability_no_three_consecutive_ones_l126_12687


namespace pizza_size_increase_l126_12674

theorem pizza_size_increase (r : ℝ) (h : r > 0) :
  let R := r * Real.sqrt 1.21
  (R ^ 2 - r ^ 2) / r ^ 2 = 0.21000000000000018 →
  (R - r) / r = 0.1 := by
sorry

end pizza_size_increase_l126_12674


namespace sum_a5_a4_l126_12654

def S (n : ℕ) : ℕ := n^2 + 2*n - 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_a5_a4 : a 5 + a 4 = 20 := by
  sorry

end sum_a5_a4_l126_12654


namespace largest_prime_factor_of_expression_l126_12652

theorem largest_prime_factor_of_expression :
  let n : ℤ := 25^2 + 35^3 - 10^5
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n.natAbs ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ n.natAbs → q ≤ p ∧ p = 113 :=
by sorry

end largest_prime_factor_of_expression_l126_12652


namespace least_four_digit_square_and_cube_l126_12671

theorem least_four_digit_square_and_cube : ∃ n : ℕ,
  (1000 ≤ n ∧ n < 10000) ∧  -- four-digit number
  (∃ a : ℕ, n = a^2) ∧      -- perfect square
  (∃ b : ℕ, n = b^3) ∧      -- perfect cube
  (∀ m : ℕ, 
    (1000 ≤ m ∧ m < 10000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧
  n = 4096 := by
sorry

end least_four_digit_square_and_cube_l126_12671


namespace partition_theorem_l126_12656

theorem partition_theorem (a b : ℝ) (ha : 1 < a) (hab : a < 2) (hb : 2 < b) :
  (¬ ∃ (A₀ A₁ : Set ℕ), (A₀ ∪ A₁ = Set.univ) ∧ (A₀ ∩ A₁ = ∅) ∧
    (∀ (j : Fin 2) (m n : ℕ), m ∈ (if j = 0 then A₀ else A₁) → n ∈ (if j = 0 then A₀ else A₁) →
      (n / m : ℝ) < a ∨ (n / m : ℝ) > b)) ∧
  ((∃ (A₀ A₁ A₂ : Set ℕ), (A₀ ∪ A₁ ∪ A₂ = Set.univ) ∧ (A₀ ∩ A₁ = ∅) ∧ (A₀ ∩ A₂ = ∅) ∧ (A₁ ∩ A₂ = ∅) ∧
    (∀ (j : Fin 3) (m n : ℕ),
      m ∈ (if j = 0 then A₀ else if j = 1 then A₁ else A₂) →
      n ∈ (if j = 0 then A₀ else if j = 1 then A₁ else A₂) →
        (n / m : ℝ) < a ∨ (n / m : ℝ) > b)) ↔ b ≤ a^2) :=
by sorry

end partition_theorem_l126_12656


namespace comparison_theorem_l126_12623

theorem comparison_theorem (n : ℕ) (h : n ≥ 4) : 3 * 2^(n - 1) > n^2 + 3 := by
  sorry

end comparison_theorem_l126_12623


namespace cell_phone_providers_l126_12688

theorem cell_phone_providers (n : ℕ) (k : ℕ) : n = 20 ∧ k = 4 →
  (n.factorial / (n - k).factorial) = 116280 := by
  sorry

end cell_phone_providers_l126_12688


namespace exponent_division_l126_12689

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end exponent_division_l126_12689


namespace calculate_expression_l126_12657

theorem calculate_expression : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
sorry

end calculate_expression_l126_12657


namespace sum_of_squares_difference_l126_12644

theorem sum_of_squares_difference : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end sum_of_squares_difference_l126_12644


namespace square_in_S_l126_12639

/-- The set S of numbers where n-1, n, and n+1 can be expressed as sums of squares of two positive integers -/
def S : Set ℕ := {n | ∃ a b k l p q : ℕ+, 
  (a^2 + b^2 : ℕ) = n - 1 ∧ 
  (k^2 + l^2 : ℕ) = n ∧ 
  (p^2 + q^2 : ℕ) = n + 1}

/-- If n is in S, then n^2 is also in S -/
theorem square_in_S (n : ℕ) (hn : n ∈ S) : n^2 ∈ S := by
  sorry

end square_in_S_l126_12639


namespace chess_team_boys_l126_12601

theorem chess_team_boys (total : ℕ) (attendees : ℕ) 
  (junior_girls : ℕ) (senior_girls : ℕ) (boys : ℕ) :
  total = 32 →
  attendees = 18 →
  junior_girls + senior_girls + boys = total →
  (junior_girls / 3 + senior_girls / 2 + boys : ℚ) = attendees →
  boys = 4 :=
by sorry

end chess_team_boys_l126_12601


namespace probability_at_least_one_correct_william_guessing_probability_l126_12650

theorem probability_at_least_one_correct (n : Nat) (k : Nat) :
  (n ≥ 1) →
  (k ≥ 1) →
  (1 : ℚ) - (((k - 1 : ℚ) / k) ^ n) = (k ^ n - (k - 1) ^ n) / (k ^ n) :=
by sorry

theorem william_guessing_probability :
  let n : Nat := 4  -- number of questions William guesses
  let k : Nat := 5  -- number of answer choices per question
  (1 : ℚ) - (((k - 1 : ℚ) / k) ^ n) = 369 / 625 :=
by sorry

end probability_at_least_one_correct_william_guessing_probability_l126_12650


namespace dime_probability_l126_12648

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 1200
  | Coin.Dime => 800
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Dime + coin_count Coin.Penny

/-- Theorem: The probability of randomly choosing a dime from the jar is 40/314 -/
theorem dime_probability : 
  (coin_count Coin.Dime : ℚ) / total_coins = 40 / 314 := by sorry

end dime_probability_l126_12648


namespace square_properties_l126_12681

/-- A square in a 2D plane -/
structure Square where
  /-- The line representing the center's x-coordinate -/
  center_line1 : ℝ → ℝ → Prop
  /-- The line representing the center's y-coordinate -/
  center_line2 : ℝ → ℝ → Prop
  /-- The equation of one side of the square -/
  side1 : ℝ → ℝ → Prop
  /-- The equation of the second side of the square -/
  side2 : ℝ → ℝ → Prop
  /-- The equation of the third side of the square -/
  side3 : ℝ → ℝ → Prop
  /-- The equation of the fourth side of the square -/
  side4 : ℝ → ℝ → Prop

/-- Theorem stating the properties of the square -/
theorem square_properties (s : Square) :
  s.center_line1 = fun x y => x - y + 1 = 0 ∧
  s.center_line2 = fun x y => 2*x + y + 2 = 0 ∧
  s.side1 = fun x y => x + 3*y - 2 = 0 →
  s.side2 = fun x y => x + 3*y + 4 = 0 ∧
  s.side3 = fun x y => 3*x - y = 0 ∧
  s.side4 = fun x y => 3*x - y + 6 = 0 :=
by sorry


end square_properties_l126_12681


namespace calculation_proof_l126_12611

theorem calculation_proof :
  (2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15/2) ∧
  (-3^2 * (-1/3)^2 + (3/4 - 1/6 + 3/8) * (-24) = -24) := by
  sorry

end calculation_proof_l126_12611


namespace probability_sum_less_than_five_l126_12672

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 6

theorem probability_sum_less_than_five (p : ℚ) : 
  p = favorable_outcomes / dice_outcomes → p = 1 / 6 := by
  sorry

end probability_sum_less_than_five_l126_12672


namespace least_distance_between_ticks_l126_12632

theorem least_distance_between_ticks (n m : ℕ) (hn : n = 11) (hm : m = 13) :
  let lcm := Nat.lcm n m
  1 / lcm = (1 : ℚ) / 143 := by sorry

end least_distance_between_ticks_l126_12632


namespace negative_inequality_transform_l126_12640

theorem negative_inequality_transform {a b : ℝ} (h : a > b) : -a < -b := by
  sorry

end negative_inequality_transform_l126_12640


namespace symmetry_of_point_wrt_y_axis_l126_12633

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The y-axis symmetry operation on a point -/
def yAxisSymmetry (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetry_of_point_wrt_y_axis :
  let P : Point2D := { x := 2, y := 1 }
  yAxisSymmetry P = { x := -2, y := 1 } := by
  sorry

end symmetry_of_point_wrt_y_axis_l126_12633


namespace parallel_line_through_point_l126_12634

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation x - 2y - 1 = 0
    passes through the point (1, 0) and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ↔ 
  ((x = 1 ∧ y = 0) ∨ 
   ∃ (t : ℝ), x = 1 + t ∧ y = t/2) :=
sorry

end parallel_line_through_point_l126_12634


namespace solve_for_y_l126_12618

theorem solve_for_y : ∃ y : ℝ, (2 * y) / 5 = 10 ∧ y = 25 := by
  sorry

end solve_for_y_l126_12618


namespace constant_slope_on_parabola_l126_12606

/-- Given a parabola and two fixed points, prove that the slope of the line formed by 
    the intersections of lines from a moving point on the parabola to the fixed points 
    is constant. -/
theorem constant_slope_on_parabola 
  (p : ℝ) (x₀ y₀ : ℝ) (h_p : p > 0) :
  let A : ℝ × ℝ := (x₀, y₀)
  let B : ℝ × ℝ := (y₀^2 / p - x₀, y₀)
  let parabola := {P : ℝ × ℝ | P.2^2 = 2 * p * P.1}
  ∀ P ∈ parabola, ∃ C D : ℝ × ℝ,
    (C ∈ parabola ∧ D ∈ parabola) ∧
    (∃ t : ℝ, P = (2 * p * t^2, 2 * p * t)) ∧
    (C ≠ P ∧ D ≠ P) ∧
    (C.2 - A.2) / (C.1 - A.1) = (P.2 - A.2) / (P.1 - A.1) ∧
    (D.2 - B.2) / (D.1 - B.1) = (P.2 - B.2) / (P.1 - B.1) ∧
    (D.2 - C.2) / (D.1 - C.1) = p / y₀ :=
by sorry

end constant_slope_on_parabola_l126_12606


namespace april_coffee_expenditure_l126_12699

/-- Calculates the total expenditure on coffee for a month given the number of coffees per day, 
    cost per coffee, and number of days in the month. -/
def coffee_expenditure (coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_month : ℕ) : ℕ :=
  coffees_per_day * cost_per_coffee * days_in_month

theorem april_coffee_expenditure :
  coffee_expenditure 2 2 30 = 120 := by
  sorry

end april_coffee_expenditure_l126_12699


namespace root_product_sum_zero_l126_12609

noncomputable def sqrt10 : ℝ := Real.sqrt 10

theorem root_product_sum_zero 
  (y₁ y₂ y₃ : ℝ) 
  (h_roots : ∀ x, x^3 - 6*sqrt10*x^2 + 10 = 0 ↔ x = y₁ ∨ x = y₂ ∨ x = y₃)
  (h_order : y₁ < y₂ ∧ y₂ < y₃) : 
  y₂ * (y₁ + y₃) = 0 := by
sorry

end root_product_sum_zero_l126_12609


namespace weight_problem_l126_12664

theorem weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 42)
  (h3 : (b + c) / 2 = 43) :
  b = 35 := by
sorry

end weight_problem_l126_12664


namespace rectangular_prism_parallel_edges_l126_12622

-- Define a rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0
  unequal_faces : length ≠ width ∧ width ≠ height ∧ height ≠ length

-- Define a function to count parallel edge pairs
def count_parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  12

-- Theorem statement
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  count_parallel_edge_pairs prism = 12 := by
  sorry

end rectangular_prism_parallel_edges_l126_12622


namespace fraction_sum_bounds_l126_12696

theorem fraction_sum_bounds (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_log_sum : Real.log a + Real.log b + Real.log c = 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 := by
  sorry


end fraction_sum_bounds_l126_12696


namespace least_four_digit_multiple_l126_12680

theorem least_four_digit_multiple : ∃ n : ℕ,
  (n ≥ 1000 ∧ n < 10000) ∧
  n % 15 = 0 ∧ n % 25 = 0 ∧ n % 40 = 0 ∧ n % 75 = 0 ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 15 = 0 ∧ m % 25 = 0 ∧ m % 40 = 0 ∧ m % 75 = 0 → m ≥ n) ∧
  n = 1200 :=
by sorry


end least_four_digit_multiple_l126_12680


namespace c_possible_values_l126_12616

/-- Represents a string of base-ten digits -/
def DigitString : Type := List Nat

/-- Represents the number of valid splits of a digit string -/
def c (m : Nat) (S : DigitString) : Nat :=
  sorry

/-- Theorem stating the possible values of c(S) -/
theorem c_possible_values (m : Nat) (S : DigitString) :
  m > 1 → ∃ n : Nat, c m S = 0 ∨ c m S = 2^n := by
  sorry

end c_possible_values_l126_12616


namespace blocks_for_house_l126_12647

/-- Given that Randy used 80 blocks in total for a tower and a house, 
    and 27 blocks for the tower, prove that he used 53 blocks for the house. -/
theorem blocks_for_house (total : ℕ) (tower : ℕ) (house : ℕ) 
    (h1 : total = 80) (h2 : tower = 27) (h3 : total = tower + house) : house = 53 := by
  sorry

end blocks_for_house_l126_12647


namespace milk_bag_probability_l126_12684

theorem milk_bag_probability (total_bags : ℕ) (expired_bags : ℕ) (selected_bags : ℕ) : 
  total_bags = 5 → 
  expired_bags = 2 → 
  selected_bags = 2 → 
  (Nat.choose (total_bags - expired_bags) selected_bags : ℚ) / (Nat.choose total_bags selected_bags) = 3/10 := by
sorry

end milk_bag_probability_l126_12684


namespace sum_of_x_and_y_positive_l126_12607

theorem sum_of_x_and_y_positive (x y : ℝ) (h : 2 * x + 3 * y > 2 - y + 3 - x) : x + y > 0 := by
  sorry

end sum_of_x_and_y_positive_l126_12607
