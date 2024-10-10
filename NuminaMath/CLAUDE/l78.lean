import Mathlib

namespace difference_of_squares_l78_7856

theorem difference_of_squares (x y : ℝ) : 
  x + y = 15 → x - y = 10 → x^2 - y^2 = 150 := by
  sorry

end difference_of_squares_l78_7856


namespace yellow_white_flowers_l78_7844

theorem yellow_white_flowers (total : ℕ) (red_yellow : ℕ) (red_white : ℕ) (red_minus_white : ℕ) :
  total = 44 →
  red_yellow = 17 →
  red_white = 14 →
  red_minus_white = 4 →
  red_yellow + red_white - (red_white + (total - red_yellow - red_white)) = red_minus_white →
  total - red_yellow - red_white = 13 := by
sorry

end yellow_white_flowers_l78_7844


namespace polygon_sides_and_diagonals_l78_7847

theorem polygon_sides_and_diagonals :
  ∀ n : ℕ,
  (180 * (n - 2) = 3 * 360 + 180) →
  (n = 9 ∧ (n * (n - 3)) / 2 = 27) :=
by
  sorry

end polygon_sides_and_diagonals_l78_7847


namespace runners_meeting_point_l78_7867

/-- Represents the marathon track setup and runners' speeds -/
structure MarathonTrack where
  totalLength : ℝ
  uphillLength : ℝ
  jackHeadStart : ℝ
  jackUphillSpeed : ℝ
  jackDownhillSpeed : ℝ
  jillUphillSpeed : ℝ
  jillDownhillSpeed : ℝ

/-- Calculates the distance from the top of the hill where runners meet -/
def distanceFromTop (track : MarathonTrack) : ℝ :=
  sorry

/-- Theorem stating the distance from the top where runners meet -/
theorem runners_meeting_point (track : MarathonTrack)
  (h1 : track.totalLength = 16)
  (h2 : track.uphillLength = 8)
  (h3 : track.jackHeadStart = 0.25)
  (h4 : track.jackUphillSpeed = 12)
  (h5 : track.jackDownhillSpeed = 18)
  (h6 : track.jillUphillSpeed = 14)
  (h7 : track.jillDownhillSpeed = 20) :
  distanceFromTop track = 511 / 32 := by
  sorry

end runners_meeting_point_l78_7867


namespace factorization_of_16x_squared_minus_1_l78_7885

theorem factorization_of_16x_squared_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4*x + 1) * (4*x - 1) := by
  sorry

end factorization_of_16x_squared_minus_1_l78_7885


namespace probability_no_distinct_roots_l78_7814

-- Define the range of b and c
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Define the condition for not having distinct real roots
def no_distinct_roots (b c : Int) : Prop := b^2 - 4*c ≤ 0

-- Define the total number of possible pairs
def total_pairs : Nat := (15 * 15 : Nat)

-- Define the number of pairs that don't have distinct roots
def pairs_without_distinct_roots : Nat := 180

-- Theorem statement
theorem probability_no_distinct_roots :
  (pairs_without_distinct_roots : Rat) / total_pairs = 4 / 5 := by
  sorry

end probability_no_distinct_roots_l78_7814


namespace team_selection_with_twins_l78_7877

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of players to be chosen
def chosen_players : ℕ := 7

-- Define the number of twins
def num_twins : ℕ := 2

-- Theorem statement
theorem team_selection_with_twins :
  (Nat.choose total_players chosen_players) - 
  (Nat.choose (total_players - num_twins) chosen_players) = 20384 :=
by sorry

end team_selection_with_twins_l78_7877


namespace max_δ_is_seven_l78_7846

/-- The sequence a_n = 1 + n^3 -/
def a (n : ℕ) : ℕ := 1 + n^3

/-- The greatest common divisor of consecutive terms in the sequence -/
def δ (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

/-- The maximum value of δ_n is 7 -/
theorem max_δ_is_seven : ∃ (n : ℕ), δ n = 7 ∧ ∀ (m : ℕ), δ m ≤ 7 := by
  sorry

end max_δ_is_seven_l78_7846


namespace propositions_truth_l78_7824

theorem propositions_truth :
  (∀ a b : ℝ, a * b > 0 → a > b → 1 / a < 1 / b) ∧
  (∀ a b : ℝ, a > abs b → a^2 > b^2) ∧
  (¬ ∀ a b c d : ℝ, a > b → c > d → a - c > b - d) ∧
  (∀ a b m : ℝ, 0 < a → a < b → m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end propositions_truth_l78_7824


namespace tangent_line_sum_l78_7859

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (1, f(1)) has equation y = 1/2 * x + 2
def has_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m = (1/2 : ℝ) ∧ b = 2 ∧ 
  ∀ (x : ℝ), f 1 + m * (x - 1) = m * x + b

-- State the theorem
theorem tangent_line_sum (f : ℝ → ℝ) (h : has_tangent_line f) :
  f 1 + deriv f 1 = 3 :=
sorry

end tangent_line_sum_l78_7859


namespace unit_vectors_equal_magnitude_l78_7888

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors a and b
variable (a b : V)

-- State the theorem
theorem unit_vectors_equal_magnitude
  (ha : ‖a‖ = 1) -- a is a unit vector
  (hb : ‖b‖ = 1) -- b is a unit vector
  : ‖a‖ = ‖b‖ := by
  sorry

end unit_vectors_equal_magnitude_l78_7888


namespace sum_of_digits_equation_l78_7865

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_equation : 
  ∃ (n : ℕ), n + sum_of_digits n = 2018 :=
by
  -- The proof would go here
  sorry

end sum_of_digits_equation_l78_7865


namespace birth_probability_l78_7894

theorem birth_probability (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 1/2) :
  let prob_all_same := p^n
  let prob_three_two := (n.choose 3) * p^n
  prob_three_two > prob_all_same :=
by sorry

end birth_probability_l78_7894


namespace largest_non_representable_amount_l78_7804

/-- Represents the denominations of coins in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ i => 5^(n - i) * 7^i)

/-- Determines if a number is representable using the given coin denominations -/
def is_representable (s n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The main theorem stating the largest non-representable amount -/
theorem largest_non_representable_amount (n : ℕ) :
  ∀ s : ℕ, s > 2 * 7^(n+1) - 3 * 5^(n+1) → is_representable s n ∧
  ¬is_representable (2 * 7^(n+1) - 3 * 5^(n+1)) n :=
sorry

end largest_non_representable_amount_l78_7804


namespace fraction_value_in_system_l78_7821

theorem fraction_value_in_system (a b x y : ℝ) (hb : b ≠ 0) 
  (eq1 : 4 * x - 2 * y = a) (eq2 : 5 * y - 10 * x = b) : a / b = -2 / 5 := by
  sorry

end fraction_value_in_system_l78_7821


namespace find_other_number_l78_7871

theorem find_other_number (A B : ℕ+) (h1 : A = 24) (h2 : Nat.gcd A B = 14) (h3 : Nat.lcm A B = 312) :
  B = 182 := by
  sorry

end find_other_number_l78_7871


namespace quadratic_factorization_l78_7850

theorem quadratic_factorization (a b : ℤ) :
  (∀ x : ℝ, 20 * x^2 - 90 * x - 22 = (5 * x + a) * (4 * x + b)) →
  a + 3 * b = -65 := by
  sorry

end quadratic_factorization_l78_7850


namespace quadratic_one_solution_l78_7849

theorem quadratic_one_solution (k : ℝ) :
  (∃! x, 4 * x^2 + k * x + 4 = 0) ↔ (k = 8 ∨ k = -8) := by
  sorry

end quadratic_one_solution_l78_7849


namespace eva_marks_difference_l78_7838

/-- Represents Eva's marks in a single semester -/
structure SemesterMarks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Represents Eva's marks for the entire year -/
structure YearMarks where
  first : SemesterMarks
  second : SemesterMarks

def total_marks (year : YearMarks) : ℕ :=
  year.first.maths + year.first.arts + year.first.science +
  year.second.maths + year.second.arts + year.second.science

theorem eva_marks_difference (eva : YearMarks) : 
  eva.second.maths = 80 →
  eva.second.arts = 90 →
  eva.second.science = 90 →
  eva.first.maths = eva.second.maths + 10 →
  eva.first.science = eva.second.science - (eva.second.science / 3) →
  total_marks eva = 485 →
  eva.second.arts - eva.first.arts = 75 := by
  sorry

end eva_marks_difference_l78_7838


namespace eighth_diagram_fully_shaded_l78_7827

/-- The number of shaded triangles in the nth diagram -/
def shaded_triangles (n : ℕ) : ℕ := n^2

/-- The total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := n^2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded_triangles n / total_triangles n

theorem eighth_diagram_fully_shaded :
  shaded_fraction 8 = 1 := by sorry

end eighth_diagram_fully_shaded_l78_7827


namespace arithmetic_geometric_sequence_l78_7896

def arithmetic_sequence (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_geometric_sequence (d : ℚ) (h : d ≠ 0) :
  let a := arithmetic_sequence 1 d
  (a 2) * (a 9) = (a 4)^2 →
  (∃ q : ℚ, q = 5/2 ∧
    (∀ n : ℕ, arithmetic_sequence 1 d n = 3*n - 2) ∧
    (∀ n : ℕ, sum_arithmetic_sequence 1 d n = (3*n^2 - n) / 2)) :=
by sorry

end arithmetic_geometric_sequence_l78_7896


namespace laundry_day_lcm_l78_7874

theorem laundry_day_lcm : Nat.lcm 6 (Nat.lcm 9 (Nat.lcm 12 15)) = 180 := by
  sorry

end laundry_day_lcm_l78_7874


namespace quadratic_roots_relation_l78_7829

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 5 = 0 ∧ 
               2 * s^2 - 4 * s - 5 = 0 ∧
               ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  c = 1/2 := by
sorry

end quadratic_roots_relation_l78_7829


namespace baseball_card_value_decrease_l78_7826

theorem baseball_card_value_decrease (initial_value : ℝ) (h_initial_positive : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.5)
  let second_year_decrease_percent := (0.55 * initial_value - 0.5 * initial_value) / first_year_value
  second_year_decrease_percent = 0.1 := by
sorry

end baseball_card_value_decrease_l78_7826


namespace chess_tournament_games_l78_7866

/-- Represents a chess tournament with players of two ranks -/
structure ChessTournament where
  a_players : Nat
  b_players : Nat

/-- Calculates the total number of games in a chess tournament -/
def total_games (t : ChessTournament) : Nat :=
  t.a_players * t.b_players

/-- Theorem: In a chess tournament with 3 'A' players and 3 'B' players, 
    where each 'A' player faces all 'B' players, the total number of games is 9 -/
theorem chess_tournament_games :
  ∀ (t : ChessTournament), 
  t.a_players = 3 → t.b_players = 3 → total_games t = 9 := by
  sorry


end chess_tournament_games_l78_7866


namespace isosceles_triangle_base_length_l78_7818

theorem isosceles_triangle_base_length 
  (eq_perimeter : ℝ) 
  (is_perimeter : ℝ) 
  (eq_side : ℝ) 
  (is_equal_side : ℝ) 
  (is_base : ℝ) 
  (vertex_angle : ℝ) 
  (h1 : eq_perimeter = 45) 
  (h2 : is_perimeter = 40) 
  (h3 : 3 * eq_side = eq_perimeter) 
  (h4 : 2 * is_equal_side + is_base = is_perimeter) 
  (h5 : is_equal_side = eq_side) 
  (h6 : 100 < vertex_angle ∧ vertex_angle < 120) : 
  is_base = 10 := by sorry

end isosceles_triangle_base_length_l78_7818


namespace average_of_eight_numbers_l78_7833

theorem average_of_eight_numbers :
  ∀ (a₁ a₂ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ),
    (a₁ + a₂) / 2 = 20 →
    (b₁ + b₂ + b₃) / 3 = 26 →
    c₁ = c₂ - 4 →
    c₁ = c₃ - 6 →
    c₃ = 30 →
    (a₁ + a₂ + b₁ + b₂ + b₃ + c₁ + c₂ + c₃) / 8 = 25 := by
  sorry


end average_of_eight_numbers_l78_7833


namespace technician_round_trip_l78_7881

theorem technician_round_trip (D : ℝ) (h : D > 0) :
  let total_distance := 2 * D
  let distance_traveled := 0.55 * total_distance
  let return_distance := distance_traveled - D
  return_distance / D = 0.1 := by
sorry

end technician_round_trip_l78_7881


namespace inequality_solution_implies_a_range_l78_7873

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) → a > -1 := by
  sorry

end inequality_solution_implies_a_range_l78_7873


namespace spring_length_dependent_on_mass_l78_7831

/-- Represents the relationship between spring length and object mass -/
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

theorem spring_length_dependent_on_mass :
  ∃ (f : ℝ → ℝ), ∀ (mass : ℝ), spring_length mass = f mass ∧
  ¬ (∃ (g : ℝ → ℝ), ∀ (length : ℝ), mass = g length) :=
sorry

end spring_length_dependent_on_mass_l78_7831


namespace f_minimum_value_l78_7883

noncomputable def f (x : ℝ) : ℝ :=
  x + (2*x)/(x^2 + 1) + (x*(x + 5))/(x^2 + 3) + (3*(x + 3))/(x*(x^2 + 3))

theorem f_minimum_value (x : ℝ) (h : x > 0) : f x ≥ 5.5 := by
  sorry

end f_minimum_value_l78_7883


namespace gas_station_candy_boxes_l78_7800

theorem gas_station_candy_boxes : 3 + 5 + 2 + 4 + 7 = 21 := by
  sorry

end gas_station_candy_boxes_l78_7800


namespace cone_volume_l78_7837

theorem cone_volume (s : ℝ) (θ : ℝ) (h : s = 6 ∧ θ = 2 * π / 3) :
  ∃ (V : ℝ), V = (16 * Real.sqrt 2 / 3) * π ∧
  V = (1 / 3) * π * (s * θ / (2 * π))^2 * Real.sqrt (s^2 - (s * θ / (2 * π))^2) :=
by sorry

end cone_volume_l78_7837


namespace golf_cost_calculation_l78_7816

/-- Proves that given the cost of one round of golf and the number of rounds that can be played,
    the total amount of money is correctly calculated. -/
theorem golf_cost_calculation (cost_per_round : ℕ) (num_rounds : ℕ) (total_money : ℕ) :
  cost_per_round = 80 →
  num_rounds = 5 →
  total_money = cost_per_round * num_rounds →
  total_money = 400 := by
sorry

end golf_cost_calculation_l78_7816


namespace burrito_cost_burrito_cost_is_six_l78_7870

/-- Calculates the cost of burritos given the following conditions:
  * There are 10 burritos with 120 calories each
  * 5 burgers with 400 calories each cost $8
  * Burgers provide 50 more calories per dollar than burritos
-/
theorem burrito_cost : ℝ → Prop :=
  fun cost : ℝ =>
    let burrito_count : ℕ := 10
    let burrito_calories : ℕ := 120
    let burger_count : ℕ := 5
    let burger_calories : ℕ := 400
    let burger_cost : ℝ := 8
    let calorie_difference : ℝ := 50

    let total_burrito_calories : ℕ := burrito_count * burrito_calories
    let total_burger_calories : ℕ := burger_count * burger_calories
    let burger_calories_per_dollar : ℝ := total_burger_calories / burger_cost
    let burrito_calories_per_dollar : ℝ := burger_calories_per_dollar - calorie_difference

    cost = total_burrito_calories / burrito_calories_per_dollar ∧
    cost = 6

theorem burrito_cost_is_six : burrito_cost 6 := by
  sorry

end burrito_cost_burrito_cost_is_six_l78_7870


namespace linear_inequality_equivalence_l78_7893

theorem linear_inequality_equivalence :
  ∀ x : ℝ, (2 * x - 4 > 0) ↔ (x > 2) := by sorry

end linear_inequality_equivalence_l78_7893


namespace probability_six_even_numbers_l78_7876

def integers_range : Set ℤ := {x | -9 ≤ x ∧ x ≤ 9}

def even_numbers (S : Set ℤ) : Set ℤ := {x ∈ S | x % 2 = 0}

def total_count : ℕ := Finset.card (Finset.range 19)

def even_count (S : Set ℤ) : ℕ := Finset.card (Finset.filter (λ x => x % 2 = 0) (Finset.range 19))

theorem probability_six_even_numbers :
  let S := integers_range
  let n := total_count
  let k := even_count S
  (k.choose 6 : ℚ) / (n.choose 6 : ℚ) = 1 / 76 := by sorry

end probability_six_even_numbers_l78_7876


namespace remainder_problem_l78_7855

theorem remainder_problem (greatest_divisor remainder_4521 : ℕ) 
  (h1 : greatest_divisor = 88)
  (h2 : remainder_4521 = 33)
  (h3 : ∃ q1 : ℕ, 3815 = greatest_divisor * q1 + (3815 % greatest_divisor))
  (h4 : ∃ q2 : ℕ, 4521 = greatest_divisor * q2 + remainder_4521) :
  3815 % greatest_divisor = 31 := by
sorry

end remainder_problem_l78_7855


namespace morning_ride_l78_7852

theorem morning_ride (x : ℝ) (h : x + 5*x = 12) : x = 2 := by
  sorry

end morning_ride_l78_7852


namespace scientific_notation_of_2590000_l78_7840

theorem scientific_notation_of_2590000 :
  2590000 = 2.59 * (10 ^ 6) := by
  sorry

end scientific_notation_of_2590000_l78_7840


namespace third_month_sale_l78_7884

def average_sale : ℝ := 6500
def num_months : ℕ := 6
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 4991

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 6855 := by
  sorry

end third_month_sale_l78_7884


namespace total_length_of_stationery_l78_7805

/-- Given the lengths of a rubber, pen, and pencil with specific relationships,
    prove that their total length is 29 centimeters. -/
theorem total_length_of_stationery (rubber pen pencil : ℝ) : 
  pen = rubber + 3 →
  pencil = pen + 2 →
  pencil = 12 →
  rubber + pen + pencil = 29 := by
sorry

end total_length_of_stationery_l78_7805


namespace sealant_cost_per_square_foot_l78_7841

/-- Calculates the cost per square foot of sealant for a deck -/
theorem sealant_cost_per_square_foot
  (length : ℝ)
  (width : ℝ)
  (construction_cost_per_sqft : ℝ)
  (total_paid : ℝ)
  (h1 : length = 30)
  (h2 : width = 40)
  (h3 : construction_cost_per_sqft = 3)
  (h4 : total_paid = 4800) :
  (total_paid - construction_cost_per_sqft * length * width) / (length * width) = 1 := by
  sorry

end sealant_cost_per_square_foot_l78_7841


namespace virginia_eggs_l78_7830

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 96 → taken_eggs = 3 → final_eggs = initial_eggs - taken_eggs → final_eggs = 93 := by
  sorry

end virginia_eggs_l78_7830


namespace power_two_99_mod_7_l78_7813

theorem power_two_99_mod_7 : 2^99 % 7 = 1 := by
  sorry

end power_two_99_mod_7_l78_7813


namespace expression_simplification_l78_7869

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (18 * x^3) * (8 * y) * (1 / (6 * x * y)^2) = 4 * x / y := by
  sorry

end expression_simplification_l78_7869


namespace solve_for_s_l78_7828

theorem solve_for_s (s t : ℤ) (eq1 : 8 * s + 7 * t = 156) (eq2 : s = t - 3) : s = 9 := by
  sorry

end solve_for_s_l78_7828


namespace nabla_calculation_l78_7801

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := a + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 3 2) 2 = 2059 := by
  sorry

end nabla_calculation_l78_7801


namespace cos_2alpha_plus_pi_12_l78_7898

theorem cos_2alpha_plus_pi_12 (α : Real) (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : Real.sin (α - π/8) = Real.sqrt 3 / 3) : 
  Real.cos (2*α + π/12) = (1 - 2*Real.sqrt 6) / 6 := by
  sorry

end cos_2alpha_plus_pi_12_l78_7898


namespace binomial_20_choose_7_l78_7875

theorem binomial_20_choose_7 : Nat.choose 20 7 = 5536 := by
  sorry

end binomial_20_choose_7_l78_7875


namespace solution_system_equations_l78_7899

theorem solution_system_equations (a : ℝ) (ha : a ≠ 0) :
  let x₁ := a / 2 + (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let y₁ := a / 2 - (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let x₂ := a / 2 - (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let y₂ := a / 2 + (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  (x₁ + y₁ = a ∧ x₁^5 + y₁^5 = 2 * a^5) ∧
  (x₂ + y₂ = a ∧ x₂^5 + y₂^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x + y = a ∧ x^5 + y^5 = 2 * a^5 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end solution_system_equations_l78_7899


namespace quadratic_real_roots_l78_7839

/-- 
For a quadratic equation kx^2 + 2x + 1 = 0, where k is a real number,
this theorem states that the equation has real roots if and only if k ≤ 1 and k ≠ 0.
-/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_l78_7839


namespace probability_of_circle_l78_7858

theorem probability_of_circle (total_figures : ℕ) (circles : ℕ) 
  (h1 : total_figures = 10)
  (h2 : circles = 4) :
  (circles : ℚ) / total_figures = 2 / 5 := by
  sorry

end probability_of_circle_l78_7858


namespace circle_radius_l78_7853

theorem circle_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + k = 0 → 
   ∃ h a : ℝ, (x - h)^2 + (y - a)^2 = 5^2) ↔ 
  k = -8 := by sorry

end circle_radius_l78_7853


namespace abc_inequality_l78_7878

theorem abc_inequality : 
  let a : ℝ := Real.rpow 7 (1/3)
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2
  a < c ∧ c < b := by sorry

end abc_inequality_l78_7878


namespace marble_distribution_l78_7890

theorem marble_distribution (n : ℕ) (hn : n = 720) :
  (Finset.filter (fun m => m > 1 ∧ m < n ∧ n % m = 0) (Finset.range (n + 1))).card = 28 := by
  sorry

end marble_distribution_l78_7890


namespace overtime_increase_is_25_percent_l78_7868

/-- Calculates the percentage increase for overtime pay given basic pay and total wage information. -/
def overtime_percentage_increase (basic_pay : ℚ) (total_wage : ℚ) (basic_hours : ℕ) (total_hours : ℕ) : ℚ :=
  let basic_rate : ℚ := basic_pay / basic_hours
  let overtime_hours : ℕ := total_hours - basic_hours
  let overtime_pay : ℚ := total_wage - basic_pay
  let overtime_rate : ℚ := overtime_pay / overtime_hours
  ((overtime_rate - basic_rate) / basic_rate) * 100

/-- Theorem stating that given the specified conditions, the overtime percentage increase is 25%. -/
theorem overtime_increase_is_25_percent :
  overtime_percentage_increase 20 25 40 48 = 25 := by
  sorry

end overtime_increase_is_25_percent_l78_7868


namespace tangent_angle_at_one_l78_7811

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_angle_at_one (x : ℝ) :
  let slope := f' 1
  let angle := Real.arctan slope
  angle = π/4 := by sorry

end tangent_angle_at_one_l78_7811


namespace construction_material_total_l78_7861

theorem construction_material_total (gravel sand : ℝ) 
  (h1 : gravel = 5.91) (h2 : sand = 8.11) : 
  gravel + sand = 14.02 := by
  sorry

end construction_material_total_l78_7861


namespace same_terminal_side_negative_pi_sixth_same_terminal_side_l78_7836

theorem same_terminal_side (θ₁ θ₂ : ℝ) : ∃ k : ℤ, θ₂ = θ₁ + 2 * π * k → 
  (θ₁.cos = θ₂.cos ∧ θ₁.sin = θ₂.sin) :=
by sorry

theorem negative_pi_sixth_same_terminal_side : 
  ∃ k : ℤ, (11 * π / 6 : ℝ) = -π / 6 + 2 * π * k :=
by sorry

end same_terminal_side_negative_pi_sixth_same_terminal_side_l78_7836


namespace specific_pentagon_area_l78_7843

/-- Pentagon PQRST with given side lengths and angles -/
structure Pentagon where
  PQ : ℝ
  QR : ℝ
  ST : ℝ
  perimeter : ℝ
  angle_QRS : ℝ
  angle_RST : ℝ
  angle_STP : ℝ

/-- The area of a pentagon with given properties -/
def pentagon_area (p : Pentagon) : ℝ :=
  sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∀ (p : Pentagon),
    p.PQ = 13 ∧
    p.QR = 18 ∧
    p.ST = 30 ∧
    p.perimeter = 82 ∧
    p.angle_QRS = 90 ∧
    p.angle_RST = 90 ∧
    p.angle_STP = 90 →
    pentagon_area p = 270 :=
  sorry

end specific_pentagon_area_l78_7843


namespace unique_g_property_l78_7895

theorem unique_g_property : ∃! (g : ℕ+), 
  (∀ (p : ℕ) (hp : Nat.Prime p) (ho : Odd p), 
    ∃ (n : ℕ+), 
      (p ∣ g.val^n.val - n.val) ∧ 
      (p ∣ g.val^(n.val + 1) - (n.val + 1))) ∧ 
  g.val = 2 :=
by sorry

end unique_g_property_l78_7895


namespace g_of_neg_three_eq_one_l78_7880

/-- Given a function g(x) = (3x + 4) / (x - 2), prove that g(-3) = 1 -/
theorem g_of_neg_three_eq_one (g : ℝ → ℝ) (h : ∀ x, x ≠ 2 → g x = (3 * x + 4) / (x - 2)) : 
  g (-3) = 1 := by
  sorry

end g_of_neg_three_eq_one_l78_7880


namespace inequality_holds_l78_7832

theorem inequality_holds (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end inequality_holds_l78_7832


namespace right_triangle_inequality_l78_7848

theorem right_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b < c) (right_triangle : a^2 + b^2 = c^2) :
  (1/a + 1/b + 1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) ∧
  ∀ M > 5 + 3 * Real.sqrt 2, ∃ a' b' c' : ℝ,
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' ≤ b' ∧ b' < c' ∧ a'^2 + b'^2 = c'^2 ∧
    (1/a' + 1/b' + 1/c') < M / (a' + b' + c') := by
  sorry

end right_triangle_inequality_l78_7848


namespace angle_mor_measure_l78_7879

/-- A regular octagon with vertices LMNOPQR -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices j) (vertices (j + 1))

/-- The measure of an angle in radians -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the measure of angle MOR in a regular octagon is π/8 radians (22.5°) -/
theorem angle_mor_measure (octagon : RegularOctagon) :
  let vertices := octagon.vertices
  angle_measure (vertices 0) (vertices 3) (vertices 5) = π / 8 := by sorry

end angle_mor_measure_l78_7879


namespace amy_music_files_l78_7854

theorem amy_music_files (total : ℕ) (video picture : ℝ) (h1 : total = 48) (h2 : video = 21.0) (h3 : picture = 23.0) :
  total - (video + picture) = 4 := by
sorry

end amy_music_files_l78_7854


namespace rounding_down_less_than_exact_sum_l78_7823

def fraction_a : ℚ := 2 / 3
def fraction_b : ℚ := 5 / 4

def round_down (q : ℚ) : ℤ := ⌊q⌋

theorem rounding_down_less_than_exact_sum :
  (round_down fraction_a : ℚ) + (round_down fraction_b : ℚ) ≤ fraction_a + fraction_b := by
  sorry

end rounding_down_less_than_exact_sum_l78_7823


namespace dave_pays_more_than_doug_l78_7845

/-- Represents the cost and composition of a pizza -/
structure Pizza where
  slices : Nat
  base_cost : Nat
  olive_slices : Nat
  olive_cost : Nat
  mushroom_slices : Nat
  mushroom_cost : Nat

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : Nat :=
  p.base_cost + p.olive_cost + p.mushroom_cost

/-- Calculates the cost of a given number of slices -/
def slice_cost (p : Pizza) (n : Nat) (with_olive : Nat) (with_mushroom : Nat) : Nat :=
  let base := n * p.base_cost / p.slices
  let olive := with_olive * p.olive_cost / p.olive_slices
  let mushroom := with_mushroom * p.mushroom_cost / p.mushroom_slices
  base + olive + mushroom

/-- Theorem: Dave pays 10 dollars more than Doug -/
theorem dave_pays_more_than_doug (p : Pizza) 
    (h1 : p.slices = 12)
    (h2 : p.base_cost = 12)
    (h3 : p.olive_slices = 3)
    (h4 : p.olive_cost = 3)
    (h5 : p.mushroom_slices = 6)
    (h6 : p.mushroom_cost = 4) :
  slice_cost p 8 2 6 - slice_cost p 4 0 0 = 10 := by
  sorry


end dave_pays_more_than_doug_l78_7845


namespace equation_solution_l78_7857

theorem equation_solution :
  ∃! x : ℚ, 7 + 3.5 * x = 2.1 * x - 30 * 1.5 ∧ x = -520 / 14 := by sorry

end equation_solution_l78_7857


namespace total_fruits_in_garden_l78_7891

def papaya_production : List Nat := [10, 12]
def mango_production : List Nat := [18, 20, 22]
def apple_production : List Nat := [14, 15, 16, 17]
def orange_production : List Nat := [20, 23, 25, 27, 30]

theorem total_fruits_in_garden : 
  (papaya_production.sum + mango_production.sum + 
   apple_production.sum + orange_production.sum) = 269 := by
  sorry

end total_fruits_in_garden_l78_7891


namespace tangent_segment_area_l78_7887

theorem tangent_segment_area (r : ℝ) (l : ℝ) (h_r : r = 3) (h_l : l = 6) :
  let outer_radius := (r^2 + (l/2)^2).sqrt
  (π * outer_radius^2 - π * r^2) = 9 * π := by sorry

end tangent_segment_area_l78_7887


namespace decimal_calculation_l78_7842

theorem decimal_calculation : (3.15 * 2.5) - 1.75 = 6.125 := by
  sorry

end decimal_calculation_l78_7842


namespace min_value_theorem_l78_7809

theorem min_value_theorem (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (m : ℝ), m = -2*Real.sqrt 3 ∧ ∀ x y : ℝ, x^2 + 2*y^2 = 6 → m ≤ x + Real.sqrt 2 * y :=
sorry

end min_value_theorem_l78_7809


namespace geometric_sequence_sum_l78_7810

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence satisfying certain conditions, prove that a₇ + a₁₀ = 27/2 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h1 : a 3 + a 6 = 6) 
  (h2 : a 5 + a 8 = 9) : 
  a 7 + a 10 = 27/2 := by
sorry

end geometric_sequence_sum_l78_7810


namespace third_shot_scores_l78_7863

/-- Represents a shooter's scores across 5 shots -/
structure ShooterScores where
  scores : Fin 5 → ℕ

/-- The problem setup -/
def ShootingProblem (shooter1 shooter2 : ShooterScores) : Prop :=
  -- The first three shots resulted in the same number of points
  (shooter1.scores 0 + shooter1.scores 1 + shooter1.scores 2 =
   shooter2.scores 0 + shooter2.scores 1 + shooter2.scores 2) ∧
  -- In the last three shots, the first shooter scored three times as many points as the second shooter
  (shooter1.scores 2 + shooter1.scores 3 + shooter1.scores 4 =
   3 * (shooter2.scores 2 + shooter2.scores 3 + shooter2.scores 4))

/-- The theorem to prove -/
theorem third_shot_scores (shooter1 shooter2 : ShooterScores)
    (h : ShootingProblem shooter1 shooter2) :
    shooter1.scores 2 = 10 ∧ shooter2.scores 2 = 2 := by
  sorry


end third_shot_scores_l78_7863


namespace fraction_simplification_l78_7834

theorem fraction_simplification :
  (-45 : ℚ) / 25 / (15 : ℚ) / 40 = -24 / 5 := by sorry

end fraction_simplification_l78_7834


namespace fish_pond_area_increase_l78_7820

/-- Proves that the increase in area of a rectangular fish pond is (20x-4) square meters
    when both length and width are increased by 2 meters. -/
theorem fish_pond_area_increase (x : ℝ) :
  let original_length : ℝ := 5 * x
  let original_width : ℝ := 5 * x - 4
  let new_length : ℝ := original_length + 2
  let new_width : ℝ := original_width + 2
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 20 * x - 4 := by
sorry

end fish_pond_area_increase_l78_7820


namespace consecutive_integers_product_272_sum_33_l78_7864

theorem consecutive_integers_product_272_sum_33 :
  ∀ x y : ℕ,
  x > 0 →
  y = x + 1 →
  x * y = 272 →
  x + y = 33 := by
sorry

end consecutive_integers_product_272_sum_33_l78_7864


namespace stratified_sampling_l78_7892

theorem stratified_sampling (first_grade : ℕ) (second_grade : ℕ) (sample_size : ℕ) (third_grade_sampled : ℕ) :
  first_grade = 24 →
  second_grade = 36 →
  sample_size = 20 →
  third_grade_sampled = 10 →
  ∃ (total_parts : ℕ) (third_grade : ℕ) (second_grade_sampled : ℕ),
    total_parts = first_grade + second_grade + third_grade ∧
    third_grade = 60 ∧
    second_grade_sampled = 6 ∧
    (third_grade : ℚ) / total_parts = (third_grade_sampled : ℚ) / sample_size ∧
    (second_grade : ℚ) / total_parts = (second_grade_sampled : ℚ) / sample_size :=
by sorry

end stratified_sampling_l78_7892


namespace max_n_for_factorizable_quadratic_l78_7807

/-- Given a quadratic expression 5x^2 + nx + 60 that can be factored as the product
    of two linear factors with integer coefficients, the maximum possible value of n is 301. -/
theorem max_n_for_factorizable_quadratic : 
  ∀ n : ℤ, 
  (∃ a b : ℤ, ∀ x : ℤ, 5 * x^2 + n * x + 60 = (5 * x + a) * (x + b)) →
  n ≤ 301 :=
by sorry

end max_n_for_factorizable_quadratic_l78_7807


namespace divisible_by_nine_l78_7825

theorem divisible_by_nine (x y : ℕ) (h : x < 10 ∧ y < 10) :
  (300000 + 10000 * x + 5700 + 70 * y + 2) % 9 = 0 →
  x + y = 1 ∨ x + y = 10 := by
  sorry

end divisible_by_nine_l78_7825


namespace max_product_with_851_l78_7822

def digits : Finset Nat := {1, 5, 6, 8, 9}

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit_number (d e : Nat) : Nat := 10 * d + e

theorem max_product_with_851 :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    three_digit_number a b c * two_digit_number d e ≤ three_digit_number 8 5 1 * two_digit_number 9 6 :=
sorry

end max_product_with_851_l78_7822


namespace special_function_properties_l78_7815

/-- A function satisfying the given properties -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g x > 0) ∧ (∀ a b, g a * g b = g (a * b))

theorem special_function_properties (g : ℝ → ℝ) (h : SpecialFunction g) :
  (g 1 = 1) ∧ (∀ a, g (1 / a) = 1 / g a) := by
  sorry

end special_function_properties_l78_7815


namespace hiking_rate_ratio_l78_7889

/-- Proves that the ratio of the rate down to the rate up is 1.5 given the hiking conditions -/
theorem hiking_rate_ratio 
  (time_equal : ℝ) -- The time for both routes is the same
  (rate_up : ℝ) -- The rate up the mountain
  (time_up : ℝ) -- The time to go up the mountain
  (distance_down : ℝ) -- The distance of the route down the mountain
  (h_rate_up : rate_up = 5) -- The rate up is 5 miles per day
  (h_time_up : time_up = 2) -- It takes 2 days to go up
  (h_distance_down : distance_down = 15) -- The route down is 15 miles long
  : (distance_down / time_equal) / rate_up = 1.5 := by
  sorry

end hiking_rate_ratio_l78_7889


namespace product_of_roots_l78_7835

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 50*x + 35 = 0) → 
  (∃ a b c : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - a) * (x - b) * (x - c) ∧ a * b * c = -35) := by
sorry

end product_of_roots_l78_7835


namespace equation_represents_point_l78_7806

theorem equation_represents_point :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 7 = 0 ↔ x = 2 ∧ y = 1 := by
sorry

end equation_represents_point_l78_7806


namespace geometric_sum_first_seven_l78_7897

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_seven :
  geometric_sum (1/4) (1/4) 7 = 16383/49152 := by
  sorry

end geometric_sum_first_seven_l78_7897


namespace a_range_l78_7886

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 :=
by sorry


end a_range_l78_7886


namespace max_points_for_successful_teams_l78_7851

/-- Represents the number of teams in the tournament -/
def num_teams : ℕ := 15

/-- Represents the number of teams that scored at least N points -/
def num_successful_teams : ℕ := 6

/-- Represents the points awarded for a win -/
def win_points : ℕ := 3

/-- Represents the points awarded for a draw -/
def draw_points : ℕ := 1

/-- Represents the points awarded for a loss -/
def loss_points : ℕ := 0

/-- Theorem stating the maximum value of N -/
theorem max_points_for_successful_teams :
  ∃ (N : ℕ), 
    (∀ (n : ℕ), n > N → 
      ¬∃ (team_scores : Fin num_teams → ℕ),
        (∀ i j, i ≠ j → team_scores i + team_scores j ≤ win_points) ∧
        (∃ (successful : Fin num_teams → Prop),
          (∃ (k : Fin num_successful_teams), ∀ i, successful i ↔ team_scores i ≥ n))) ∧
    (∃ (team_scores : Fin num_teams → ℕ),
      (∀ i j, i ≠ j → team_scores i + team_scores j ≤ win_points) ∧
      (∃ (successful : Fin num_teams → Prop),
        (∃ (k : Fin num_successful_teams), ∀ i, successful i ↔ team_scores i ≥ N))) ∧
    N = 34 := by
  sorry

end max_points_for_successful_teams_l78_7851


namespace expression_value_l78_7803

theorem expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by sorry

end expression_value_l78_7803


namespace max_min_values_of_f_l78_7812

-- Define the function f(x)
def f (x : ℝ) : ℝ := 6 - 12*x + x^3

-- Define the interval
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem max_min_values_of_f :
  (∃ x ∈ interval, f x = 22 ∧ ∀ y ∈ interval, f y ≤ 22) ∧
  (∃ x ∈ interval, f x = -5 ∧ ∀ y ∈ interval, f y ≥ -5) := by
  sorry

end max_min_values_of_f_l78_7812


namespace ice_cream_revenue_l78_7860

/-- Calculate the total revenue from ice cream sales with discounts --/
theorem ice_cream_revenue : 
  let chocolate : ℕ := 50
  let mango : ℕ := 54
  let vanilla : ℕ := 80
  let strawberry : ℕ := 40
  let price : ℚ := 2
  let chocolate_sold : ℚ := 3 / 5 * chocolate
  let mango_sold : ℚ := 2 / 3 * mango
  let vanilla_sold : ℚ := 75 / 100 * vanilla
  let strawberry_sold : ℚ := 5 / 8 * strawberry
  let discount : ℚ := 1 / 2
  let apply_discount (x : ℚ) : ℚ := if x ≥ 10 then x * discount else 0

  let total_revenue : ℚ := 
    (chocolate_sold + mango_sold + vanilla_sold + strawberry_sold) * price - 
    (apply_discount chocolate_sold + apply_discount mango_sold + 
     apply_discount vanilla_sold + apply_discount strawberry_sold)

  total_revenue = 226.5 := by sorry

end ice_cream_revenue_l78_7860


namespace square_perimeter_when_area_equals_side_l78_7819

/-- A square with area numerically equal to its side length has a perimeter of 4 units. -/
theorem square_perimeter_when_area_equals_side : ∀ s : ℝ,
  s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end square_perimeter_when_area_equals_side_l78_7819


namespace cube_root_equation_solution_l78_7808

theorem cube_root_equation_solution (y : ℝ) :
  (6 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 2/33 := by
  sorry

end cube_root_equation_solution_l78_7808


namespace cookies_ratio_l78_7817

/-- Proves the ratio of cookies eaten by Monica's mother to her father -/
theorem cookies_ratio :
  ∀ (total mother_cookies father_cookies brother_cookies left : ℕ),
  total = 30 →
  father_cookies = 10 →
  brother_cookies = mother_cookies + 2 →
  left = 8 →
  total = mother_cookies + father_cookies + brother_cookies + left →
  (mother_cookies : ℚ) / father_cookies = 1 / 2 := by
sorry

end cookies_ratio_l78_7817


namespace base8_arithmetic_result_l78_7882

/-- Convert a base 8 number to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Convert a base 10 number to base 8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Perform base 8 arithmetic: multiply by 2 and subtract --/
def base8Arithmetic (a b : ℕ) : ℕ :=
  base10ToBase8 ((base8ToBase10 a) * 2 - (base8ToBase10 b))

theorem base8_arithmetic_result :
  base8Arithmetic 45 76 = 14 := by sorry

end base8_arithmetic_result_l78_7882


namespace min_b_value_l78_7802

/-- The function f(x) = x^2 + 2bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x

/-- The function g(x) = |x-1| -/
def g (x : ℝ) : ℝ := |x - 1|

/-- The theorem stating the minimum value of b -/
theorem min_b_value :
  ∀ b : ℝ,
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 →
    f b x₁ - f b x₂ < g x₁ - g x₂) →
  b ≥ -1/2 :=
by sorry

end min_b_value_l78_7802


namespace unique_solution_system_l78_7862

theorem unique_solution_system (x y : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧
  y * Real.sqrt (2 * x) - x * Real.sqrt (2 * y) = 6 ∧
  x * y^2 - x^2 * y = 30 →
  x = 1/2 ∧ y = 8 :=
by sorry

end unique_solution_system_l78_7862


namespace divisibility_by_eleven_l78_7872

theorem divisibility_by_eleven (B : Nat) : 
  (B = 5 → 11 ∣ 15675) → 
  (∀ n : Nat, n < 10 → (11 ∣ (15670 + n) ↔ n = B)) :=
sorry

end divisibility_by_eleven_l78_7872
