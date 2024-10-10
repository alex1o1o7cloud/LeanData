import Mathlib

namespace d_value_l4113_411302

theorem d_value (d : ℚ) (h : 10 * d + 8 = 528) : 2 * d = 104 := by
  sorry

end d_value_l4113_411302


namespace decagon_triangles_l4113_411356

def regularDecagonVertices : ℕ := 10

def trianglesFromDecagon : ℕ :=
  Nat.choose regularDecagonVertices 3

theorem decagon_triangles :
  trianglesFromDecagon = 120 := by sorry

end decagon_triangles_l4113_411356


namespace ada_original_seat_l4113_411310

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends --/
inductive Friend
| ada
| bea
| ceci
| dee
| edie
| fred

/-- Represents the movement of a friend --/
structure Movement where
  friend : Friend
  displacement : Int

/-- The seating arrangement before Ada left --/
def initial_arrangement : Friend → Seat := sorry

/-- The seating arrangement after all movements --/
def final_arrangement : Friend → Seat := sorry

/-- The list of all movements --/
def movements : List Movement := sorry

/-- Calculates the net displacement of all movements --/
def net_displacement (mvs : List Movement) : Int := sorry

/-- Checks if a seat is an end seat --/
def is_end_seat (s : Seat) : Prop := s = Seat.one ∨ s = Seat.six

theorem ada_original_seat (h1 : net_displacement movements = 0)
                          (h2 : is_end_seat (final_arrangement Friend.ada)) :
  is_end_seat (initial_arrangement Friend.ada) := by sorry

end ada_original_seat_l4113_411310


namespace parabola_zeros_difference_l4113_411352

/-- Represents a quadratic function ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Finds the zeros of a quadratic function -/
def QuadraticFunction.zeros (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.eval x = 0}

theorem parabola_zeros_difference (f : QuadraticFunction) :
  f.eval 3 = -9 →  -- vertex at (3, -9)
  f.eval 5 = 7 →   -- passes through (5, 7)
  ∃ m n, m ∈ f.zeros ∧ n ∈ f.zeros ∧ m > n ∧ m - n = 3 := by
  sorry

end parabola_zeros_difference_l4113_411352


namespace lineup_calculation_1_lineup_calculation_2_l4113_411388

/-- Represents a basketball team -/
structure BasketballTeam where
  veterans : Nat
  newPlayers : Nat

/-- Represents the conditions for lineup selection -/
structure LineupConditions where
  specificVeteranMustPlay : Bool
  specificNewPlayersCannotPlay : Nat
  forwardPlayers : Nat
  guardPlayers : Nat
  versatilePlayers : Nat

/-- Calculates the number of different lineups under given conditions -/
def calculateLineups (team : BasketballTeam) (conditions : LineupConditions) : Nat :=
  sorry

/-- Theorem for the first lineup calculation -/
theorem lineup_calculation_1 (team : BasketballTeam) (conditions : LineupConditions) :
  team.veterans = 7 ∧ team.newPlayers = 5 ∧
  conditions.specificVeteranMustPlay = true ∧
  conditions.specificNewPlayersCannotPlay = 2 →
  calculateLineups team conditions = 126 :=
sorry

/-- Theorem for the second lineup calculation -/
theorem lineup_calculation_2 (team : BasketballTeam) (conditions : LineupConditions) :
  team.veterans + team.newPlayers = 12 ∧
  conditions.forwardPlayers = 6 ∧
  conditions.guardPlayers = 4 ∧
  conditions.versatilePlayers = 2 →
  calculateLineups team conditions = 636 :=
sorry

end lineup_calculation_1_lineup_calculation_2_l4113_411388


namespace imaginary_part_of_complex_division_l4113_411303

theorem imaginary_part_of_complex_division : 
  Complex.im ((3 + 4 * Complex.I) / Complex.I) = -3 := by
  sorry

end imaginary_part_of_complex_division_l4113_411303


namespace tv_production_last_five_days_l4113_411326

theorem tv_production_last_five_days 
  (total_days : Nat) 
  (first_period : Nat) 
  (avg_first_period : Nat) 
  (avg_total : Nat) 
  (h1 : total_days = 30)
  (h2 : first_period = 25)
  (h3 : avg_first_period = 50)
  (h4 : avg_total = 48) :
  (total_days * avg_total - first_period * avg_first_period) / (total_days - first_period) = 38 := by
  sorry

#check tv_production_last_five_days

end tv_production_last_five_days_l4113_411326


namespace merchant_profit_l4113_411325

/-- Calculates the profit percentage given the ratio of cost price to selling price -/
def profit_percentage (cost_price : ℚ) (selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Proves that if the cost price of 19 articles is equal to the selling price of 16 articles,
    then the merchant makes a profit of 18.75% -/
theorem merchant_profit :
  ∀ (cost_price selling_price : ℚ),
  19 * cost_price = 16 * selling_price →
  profit_percentage cost_price selling_price = 18.75 := by
sorry

#eval profit_percentage 16 19 -- Should evaluate to 18.75

end merchant_profit_l4113_411325


namespace merchant_profit_percentage_l4113_411367

/-- If the cost price of 29 articles equals the selling price of 24 articles,
    then the percentage of profit is 5/24 * 100. -/
theorem merchant_profit_percentage (C S : ℝ) (h : 29 * C = 24 * S) :
  (S - C) / C * 100 = 5 / 24 * 100 := by
  sorry

end merchant_profit_percentage_l4113_411367


namespace shoe_cost_l4113_411328

/-- The cost of shoes given an initial budget and remaining amount --/
theorem shoe_cost (initial_budget remaining : ℚ) (h1 : initial_budget = 999) (h2 : remaining = 834) :
  initial_budget - remaining = 165 := by
  sorry

end shoe_cost_l4113_411328


namespace friends_with_boxes_eq_two_l4113_411390

/-- The number of pencils in one color box -/
def pencils_per_box : ℕ := 7

/-- The total number of pencils Serenity and her friends have -/
def total_pencils : ℕ := 21

/-- The number of color boxes Serenity bought -/
def serenity_boxes : ℕ := 1

/-- The number of Serenity's friends who bought the color box -/
def friends_with_boxes : ℕ := (total_pencils / pencils_per_box) - serenity_boxes

theorem friends_with_boxes_eq_two : friends_with_boxes = 2 := by
  sorry

end friends_with_boxes_eq_two_l4113_411390


namespace assembled_figure_surface_area_l4113_411320

/-- The surface area of a figure assembled from four identical bars -/
def figureSurfaceArea (barSurfaceArea : ℝ) (lostAreaPerJunction : ℝ) : ℝ :=
  4 * (barSurfaceArea - lostAreaPerJunction)

/-- Theorem: The surface area of the assembled figure is 64 cm² -/
theorem assembled_figure_surface_area :
  figureSurfaceArea 18 2 = 64 := by
  sorry

end assembled_figure_surface_area_l4113_411320


namespace salaries_sum_l4113_411396

theorem salaries_sum (A_salary B_salary : ℝ) : 
  A_salary = 5250 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 7000 :=
by
  sorry

end salaries_sum_l4113_411396


namespace centrally_symmetric_implies_congruent_l4113_411350

-- Define a shape
def Shape : Type := sorry

-- Define central symmetry
def centrally_symmetric (s1 s2 : Shape) : Prop := 
  ∃ p : ℝ × ℝ, ∃ rotation : Shape → Shape, 
    rotation s1 = s2 ∧ 
    (∀ x : Shape, rotation (rotation x) = x)

-- Define congruence
def congruent (s1 s2 : Shape) : Prop := sorry

-- Theorem statement
theorem centrally_symmetric_implies_congruent (s1 s2 : Shape) :
  centrally_symmetric s1 s2 → congruent s1 s2 := by sorry

end centrally_symmetric_implies_congruent_l4113_411350


namespace negative_eight_to_four_thirds_equals_sixteen_l4113_411393

theorem negative_eight_to_four_thirds_equals_sixteen :
  (-8 : ℝ) ^ (4/3) = 16 := by
  sorry

end negative_eight_to_four_thirds_equals_sixteen_l4113_411393


namespace initial_amount_proof_l4113_411307

/-- The initial amount given on interest -/
def P : ℝ := 1250

/-- The interest rate per annum (in decimal form) -/
def r : ℝ := 0.04

/-- The number of years -/
def n : ℕ := 2

/-- The difference between compound and simple interest -/
def interest_difference : ℝ := 2.0000000000002274

theorem initial_amount_proof :
  P * ((1 + r)^n - (1 + r * n)) = interest_difference := by
  sorry

end initial_amount_proof_l4113_411307


namespace total_with_tax_calculation_l4113_411386

def total_before_tax : ℝ := 150
def sales_tax_rate : ℝ := 0.08

theorem total_with_tax_calculation :
  total_before_tax * (1 + sales_tax_rate) = 162 := by
  sorry

end total_with_tax_calculation_l4113_411386


namespace number_problem_l4113_411395

theorem number_problem (x : ℝ) : 4 * x = 166.08 → (x / 4) + 0.48 = 10.86 := by
  sorry

end number_problem_l4113_411395


namespace john_personal_payment_l4113_411316

def hearing_aid_cost : ℝ := 2500
def insurance_coverage_percentage : ℝ := 80
def number_of_hearing_aids : ℕ := 2

theorem john_personal_payment (total_cost : ℝ) (insurance_payment : ℝ) (john_payment : ℝ) :
  total_cost = number_of_hearing_aids * hearing_aid_cost →
  insurance_payment = (insurance_coverage_percentage / 100) * total_cost →
  john_payment = total_cost - insurance_payment →
  john_payment = 1000 := by
sorry

end john_personal_payment_l4113_411316


namespace min_sum_of_product_100_l4113_411324

theorem min_sum_of_product_100 (a b : ℤ) (h : a * b = 100) :
  ∀ (x y : ℤ), x * y = 100 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 100 ∧ a₀ + b₀ = -101 :=
by sorry

end min_sum_of_product_100_l4113_411324


namespace gasoline_reduction_percentage_l4113_411383

theorem gasoline_reduction_percentage
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percentage : ℝ)
  (spending_increase_percentage : ℝ)
  (h1 : price_increase_percentage = 0.25)
  (h2 : spending_increase_percentage = 0.05)
  (h3 : original_price > 0)
  (h4 : original_quantity > 0) :
  let new_price := original_price * (1 + price_increase_percentage)
  let new_total_cost := original_price * original_quantity * (1 + spending_increase_percentage)
  let new_quantity := new_total_cost / new_price
  (1 - new_quantity / original_quantity) * 100 = 16 := by
sorry

end gasoline_reduction_percentage_l4113_411383


namespace parabola_focus_l4113_411399

/-- The parabola equation --/
def parabola (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola --/
def focus (p q : ℝ) : Prop := p = 0 ∧ q = 2

theorem parabola_focus :
  ∀ x y : ℝ, parabola x y → ∃ p q : ℝ, focus p q := by
  sorry

end parabola_focus_l4113_411399


namespace f_derivative_at_2_l4113_411364

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem f_derivative_at_2 : 
  (deriv f) 2 = -120 := by sorry

end f_derivative_at_2_l4113_411364


namespace altitude_division_ratio_l4113_411371

/-- Given a triangle with side lengths √3, 2, and √5, the altitude perpendicular 
    to the side of length 2 divides that side in the ratio 1:3 -/
theorem altitude_division_ratio (a b c : ℝ) (h₁ : a = Real.sqrt 3) 
    (h₂ : b = 2) (h₃ : c = Real.sqrt 5) :
    let m := Real.sqrt (3 - (1/2)^2)
    (1/2) / (3/2) = 1/3 := by sorry

end altitude_division_ratio_l4113_411371


namespace boat_distance_theorem_l4113_411385

/-- Calculates the distance a boat travels along a stream in one hour, given its speed in still water and its distance against the stream in one hour. -/
def distance_along_stream (boat_speed : ℝ) (distance_against : ℝ) : ℝ :=
  let stream_speed := boat_speed - distance_against
  boat_speed + stream_speed

/-- Theorem stating that a boat with a speed of 7 km/hr in still water,
    traveling 3 km against the stream in one hour,
    will travel 11 km along the stream in one hour. -/
theorem boat_distance_theorem :
  distance_along_stream 7 3 = 11 := by
  sorry

end boat_distance_theorem_l4113_411385


namespace pyramid_volume_l4113_411305

/-- The volume of a pyramid with a rectangular base and a slant edge perpendicular to two adjacent sides of the base. -/
theorem pyramid_volume (base_length base_width slant_edge : ℝ) 
  (hl : base_length = 10) 
  (hw : base_width = 6) 
  (hs : slant_edge = 20) : 
  (1 / 3 : ℝ) * base_length * base_width * Real.sqrt (slant_edge^2 - base_length^2) = 200 * Real.sqrt 3 := by
  sorry

end pyramid_volume_l4113_411305


namespace sequence_gcd_theorem_l4113_411392

theorem sequence_gcd_theorem (d m : ℕ) (hd : d > 1) :
  ∃ k l : ℕ, k ≠ l ∧ Nat.gcd (2^(2^k) + d) (2^(2^l) + d) > m := by
  sorry

end sequence_gcd_theorem_l4113_411392


namespace work_completion_time_l4113_411372

-- Define the work rates for a, b, and c
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 30
def work_rate_c : ℚ := 1 / 40

-- Define the combined work rate of a, b, and c
def combined_rate : ℚ := work_rate_a + work_rate_b + work_rate_c

-- Define the combined work rate of a and b
def combined_rate_ab : ℚ := work_rate_a + work_rate_b

-- Define the total days to complete the work
def total_days : ℚ := 11

-- Theorem statement
theorem work_completion_time :
  (total_days - 4) * combined_rate + 4 * combined_rate_ab = 1 :=
sorry

end work_completion_time_l4113_411372


namespace employee_transfer_solution_l4113_411347

/-- Represents the company's employee transfer problem -/
def EmployeeTransfer (a : ℝ) (x : ℕ) : Prop :=
  let total_employees : ℕ := 100
  let manufacturing_before : ℝ := a * total_employees
  let manufacturing_after : ℝ := a * 1.2 * (total_employees - x)
  let service_output : ℝ := 3.5 * a * x
  (manufacturing_after ≥ manufacturing_before) ∧ 
  (service_output ≥ 0.5 * manufacturing_before) ∧
  (x ≤ total_employees)

/-- Theorem stating the solution to the employee transfer problem -/
theorem employee_transfer_solution (a : ℝ) (h : a > 0) :
  ∃ x : ℕ, EmployeeTransfer a x ∧ x ≥ 15 ∧ x ≤ 16 :=
sorry

end employee_transfer_solution_l4113_411347


namespace geometric_sequence_sum_l4113_411334

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
  sorry

end geometric_sequence_sum_l4113_411334


namespace books_sold_l4113_411382

/-- Proves the number of books Adam sold given initial count, books bought, and final count -/
theorem books_sold (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 33 → bought = 23 → final = 45 → initial - (initial - final + bought) = 11 := by
  sorry

end books_sold_l4113_411382


namespace square_inequality_for_negatives_l4113_411363

theorem square_inequality_for_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_for_negatives_l4113_411363


namespace no_sequence_exists_l4113_411339

theorem no_sequence_exists : ¬ ∃ (a : Fin 7 → ℝ), 
  (∀ i, 0 ≤ a i) ∧ 
  (a 0 = 0) ∧ 
  (a 6 = 0) ∧ 
  (∀ i ∈ Finset.range 5, a (i + 2) + a i > Real.sqrt 3 * a (i + 1)) := by
sorry

end no_sequence_exists_l4113_411339


namespace gcd_polynomial_and_y_l4113_411345

theorem gcd_polynomial_and_y (y : ℤ) (h : ∃ k : ℤ, y = 46896 * k) :
  let g := fun (y : ℤ) => (3*y+5)*(8*y+3)*(16*y+9)*(y+16)
  Nat.gcd (Int.natAbs (g y)) (Int.natAbs y) = 2160 := by
  sorry

end gcd_polynomial_and_y_l4113_411345


namespace decimal_sum_as_fraction_l4113_411317

/-- The sum of 0.01, 0.002, 0.0003, 0.00004, and 0.000005 is equal to 2469/200000 -/
theorem decimal_sum_as_fraction : 
  (0.01 : ℚ) + 0.002 + 0.0003 + 0.00004 + 0.000005 = 2469 / 200000 := by
  sorry

end decimal_sum_as_fraction_l4113_411317


namespace closest_beetle_positions_l4113_411366

structure Table where
  sugar_position : ℝ × ℝ
  ant_radius : ℝ
  beetle_radius : ℝ
  ant_initial_position : ℝ × ℝ
  beetle_initial_position : ℝ × ℝ

def closest_positions (t : Table) : Set (ℝ × ℝ) :=
  {(2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3)}

theorem closest_beetle_positions (t : Table) 
  (h1 : t.sugar_position = (0, 0))
  (h2 : t.ant_radius = 2)
  (h3 : t.beetle_radius = 4)
  (h4 : t.ant_initial_position = (-1, Real.sqrt 3))
  (h5 : t.beetle_initial_position = (2 * Real.sqrt 3, 2)) :
  closest_positions t = {(2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3)} := by
  sorry

end closest_beetle_positions_l4113_411366


namespace max_games_buyable_l4113_411375

def total_earnings : ℝ := 180
def blade_percentage : ℝ := 0.35
def game_cost : ℝ := 12.50
def tax_rate : ℝ := 0.05

def remaining_money : ℝ := total_earnings * (1 - blade_percentage)
def game_cost_with_tax : ℝ := game_cost * (1 + tax_rate)

theorem max_games_buyable : 
  ⌊remaining_money / game_cost_with_tax⌋ = 8 :=
sorry

end max_games_buyable_l4113_411375


namespace smallest_side_difference_l4113_411384

theorem smallest_side_difference (P Q R : ℕ) : 
  P + Q + R = 2021 →  -- Perimeter condition
  P < Q →             -- PQ < PR
  Q ≤ R →             -- PR ≤ QR
  P + R > Q →         -- Triangle inequality
  P + Q > R →         -- Triangle inequality
  Q + R > P →         -- Triangle inequality
  (∀ P' Q' R' : ℕ, 
    P' + Q' + R' = 2021 → 
    P' < Q' → 
    Q' ≤ R' → 
    P' + R' > Q' → 
    P' + Q' > R' → 
    Q' + R' > P' → 
    Q' - P' ≥ Q - P) →
  Q - P = 1 := by
sorry

end smallest_side_difference_l4113_411384


namespace lanas_final_page_count_l4113_411327

theorem lanas_final_page_count (lana_initial : ℕ) (duane_total : ℕ) : 
  lana_initial = 8 → duane_total = 42 → lana_initial + duane_total / 2 = 29 := by
  sorry

end lanas_final_page_count_l4113_411327


namespace committee_count_l4113_411357

/-- Represents a department with male and female professors -/
structure Department where
  male_profs : Nat
  female_profs : Nat

/-- Represents the configuration of the science division -/
structure ScienceDivision where
  departments : Fin 3 → Department

/-- Represents a committee formation -/
structure Committee where
  members : Fin 6 → Nat
  department_count : Fin 3 → Nat
  male_count : Nat
  female_count : Nat

def is_valid_committee (sd : ScienceDivision) (c : Committee) : Prop :=
  c.male_count = 3 ∧ 
  c.female_count = 3 ∧ 
  (∀ d : Fin 3, c.department_count d = 2)

def count_valid_committees (sd : ScienceDivision) : Nat :=
  sorry

theorem committee_count (sd : ScienceDivision) : 
  (∀ d : Fin 3, sd.departments d = ⟨3, 3⟩) → 
  count_valid_committees sd = 1215 := by
  sorry

end committee_count_l4113_411357


namespace bowling_ball_weight_is_correct_l4113_411394

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 30

/-- Theorem stating that the weight of one bowling ball is 18.75 pounds -/
theorem bowling_ball_weight_is_correct : bowling_ball_weight = 18.75 := by
  -- Define the relationship between bowling balls and canoes
  have h1 : 8 * bowling_ball_weight = 5 * canoe_weight := by sorry
  
  -- Define the relationship between canoes and their total weight
  have h2 : 4 * canoe_weight = 120 := by sorry
  
  -- Prove that the bowling ball weight is correct
  sorry

#eval bowling_ball_weight

end bowling_ball_weight_is_correct_l4113_411394


namespace smallest_sum_of_reciprocals_l4113_411315

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
  (x : ℤ) + y ≤ (a : ℤ) + b → 
  (x : ℤ) + y = 64 :=
by sorry

end smallest_sum_of_reciprocals_l4113_411315


namespace walking_speed_ratio_l4113_411362

/-- The ratio of a slower walking speed to a usual walking speed, given the times taken for the same distance. -/
theorem walking_speed_ratio (usual_time slower_time : ℝ) 
  (h1 : usual_time = 32)
  (h2 : slower_time = 40) :
  (usual_time / slower_time) = 4 / 5 := by
  sorry

end walking_speed_ratio_l4113_411362


namespace abs_m_minus_n_equals_2_sqrt_3_l4113_411319

theorem abs_m_minus_n_equals_2_sqrt_3 (m n p : ℝ) 
  (h1 : m * n = 6)
  (h2 : m + n + p = 7)
  (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 := by
  sorry

end abs_m_minus_n_equals_2_sqrt_3_l4113_411319


namespace prob_four_same_face_five_coins_l4113_411397

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The probability of getting at least 'num_same' coins showing the same face when flipping 'num_coins' fair coins -/
def prob_same_face (num_same : ℕ) : ℚ :=
  let total_outcomes := 2^num_coins
  let favorable_outcomes := 2 * (Nat.choose num_coins (num_coins - num_same + 1))
  favorable_outcomes / total_outcomes

/-- The probability of getting at least 4 coins showing the same face when flipping 5 fair coins is 3/8 -/
theorem prob_four_same_face_five_coins : prob_same_face 4 = 3/8 := by
  sorry

end prob_four_same_face_five_coins_l4113_411397


namespace d₁_d₂_not_divisible_by_3_l4113_411332

-- Define d₁ and d₂ as functions of a
def d₁ (a : ℕ) : ℕ := a^3 + 3^a + a * 3^((a+1)/2)
def d₂ (a : ℕ) : ℕ := a^3 + 3^a - a * 3^((a+1)/2)

-- Define the main theorem
theorem d₁_d₂_not_divisible_by_3 :
  ∀ a : ℕ, 1 ≤ a → a ≤ 251 → ¬(3 ∣ (d₁ a * d₂ a)) :=
by sorry

end d₁_d₂_not_divisible_by_3_l4113_411332


namespace nobel_laureates_count_l4113_411321

/-- Represents the number of scientists at a workshop with various prize combinations -/
structure WorkshopScientists where
  total : Nat
  wolf : Nat
  wolfAndNobel : Nat
  nonWolfNobel : Nat
  nonWolfNonNobel : Nat

/-- The conditions of the workshop -/
def workshop : WorkshopScientists where
  total := 50
  wolf := 31
  wolfAndNobel := 12
  nonWolfNobel := (50 - 31 + 3) / 2
  nonWolfNonNobel := (50 - 31 - 3) / 2

/-- Theorem stating the total number of Nobel prize laureates -/
theorem nobel_laureates_count (w : WorkshopScientists) (h1 : w = workshop) :
  w.wolfAndNobel + w.nonWolfNobel = 23 := by
  sorry

#check nobel_laureates_count

end nobel_laureates_count_l4113_411321


namespace negation_statement_is_false_l4113_411378

theorem negation_statement_is_false : ¬(
  (∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x)
) := by sorry

end negation_statement_is_false_l4113_411378


namespace simplify_expression_l4113_411335

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  x⁻¹ - 3*x + 2 = -(3*x^2 - 2*x - 1) / x := by sorry

end simplify_expression_l4113_411335


namespace initial_investment_interest_rate_l4113_411373

/-- Proves that the interest rate of the initial investment is 5% given the problem conditions --/
theorem initial_investment_interest_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2000)
  (h2 : additional_investment = 1000)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : ∃ r : ℝ, r * initial_investment + additional_rate * additional_investment = 
        total_rate * (initial_investment + additional_investment)) :
  ∃ r : ℝ, r = 0.05 :=
sorry

end initial_investment_interest_rate_l4113_411373


namespace prob_three_games_correct_constant_term_is_three_f_one_half_l4113_411323

/-- Represents the probability of player A winning a single game -/
def p : ℝ := sorry

/-- Assumption that p is between 0 and 1 -/
axiom p_range : 0 ≤ p ∧ p ≤ 1

/-- The probability of the match ending in three games -/
def prob_three_games : ℝ := p^3 + (1-p)^3

/-- The expected number of games played in the match -/
def f (p : ℝ) : ℝ := 6*p^4 - 12*p^3 + 3*p^2 + 3*p + 3

/-- Theorem: The probability of the match ending in three games is p³ + (1-p)³ -/
theorem prob_three_games_correct : prob_three_games = p^3 + (1-p)^3 := by sorry

/-- Theorem: The constant term of f(p) is 3 -/
theorem constant_term_is_three : f 0 = 3 := by sorry

/-- Theorem: f(1/2) = 33/8 -/
theorem f_one_half : f (1/2) = 33/8 := by sorry

end prob_three_games_correct_constant_term_is_three_f_one_half_l4113_411323


namespace mistaken_division_l4113_411314

theorem mistaken_division (n : ℕ) : 
  (n % 32 = 0 ∧ n / 32 = 3) → n / 4 = 24 := by
  sorry

end mistaken_division_l4113_411314


namespace imaginary_part_of_complex_fraction_l4113_411368

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + I) / I
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_fraction_l4113_411368


namespace largest_prime_factor_of_expression_l4113_411359

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p ∧
  p = 307 := by
  sorry

end largest_prime_factor_of_expression_l4113_411359


namespace expression_range_l4113_411379

def expression_value (parenthesization : List (List Nat)) : ℚ :=
  sorry

theorem expression_range :
  ∀ p : List (List Nat),
    (∀ n, n ∈ p.join → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) →
    (∀ n, n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → n ∈ p.join) →
    1 / 362880 ≤ expression_value p ∧ expression_value p ≤ 181440 :=
  sorry

end expression_range_l4113_411379


namespace expression_evaluation_l4113_411336

theorem expression_evaluation : ∃ (n m k : ℕ),
  (n > 0 ∧ m > 0 ∧ k > 0) ∧
  (2 * n - 1 = 2025) ∧
  (2 * m = 2024) ∧
  (2^k = 1024) →
  (Finset.sum (Finset.range n) (λ i => 2 * i + 5)) -
  (Finset.sum (Finset.range m) (λ i => 2 * i + 4)) +
  2 * (Finset.sum (Finset.range k) (λ i => 2^i)) = 5104 := by
  sorry

end expression_evaluation_l4113_411336


namespace meeting_speed_l4113_411377

/-- Given two people 55 miles apart, where one walks at 6 mph and the other walks 25 miles before they meet, prove that the speed of the second person is 5 mph. -/
theorem meeting_speed (total_distance : ℝ) (fred_speed : ℝ) (sam_distance : ℝ) :
  total_distance = 55 →
  fred_speed = 6 →
  sam_distance = 25 →
  (total_distance - sam_distance) / fred_speed = sam_distance / ((total_distance - sam_distance) / fred_speed) :=
by sorry

end meeting_speed_l4113_411377


namespace exactly_one_success_probability_l4113_411344

/-- The probability of success in a single trial -/
def p : ℚ := 1/3

/-- The number of trials -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 1

/-- The binomial coefficient function -/
def binomial_coef (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coef n k * p^k * (1 - p)^(n - k)

theorem exactly_one_success_probability :
  binomial_probability n k p = 4/9 := by sorry

end exactly_one_success_probability_l4113_411344


namespace bobs_mile_time_l4113_411304

/-- Bob's mile run time problem -/
theorem bobs_mile_time (sister_time : ℝ) (improvement_percent : ℝ) (bob_time : ℝ) : 
  sister_time = 9 * 60 + 42 →
  improvement_percent = 9.062499999999996 →
  bob_time = sister_time * (1 + improvement_percent / 100) →
  bob_time = 634.5 := by
  sorry

end bobs_mile_time_l4113_411304


namespace problem_statement_l4113_411340

theorem problem_statement (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) 
  (h4 : k > 0) : k = Real.sqrt 15 := by
  sorry

end problem_statement_l4113_411340


namespace ast_equation_solutions_l4113_411301

-- Define the operation ※
def ast (a b : ℝ) : ℝ := a + b^2

-- Theorem statement
theorem ast_equation_solutions :
  ∃! (s : Set ℝ), s = {x : ℝ | ast x (x + 1) = 5} ∧ s = {1, -4} :=
by sorry

end ast_equation_solutions_l4113_411301


namespace vector_norm_equation_solution_l4113_411370

theorem vector_norm_equation_solution :
  let v : ℝ × ℝ := (3, -2)
  let w : ℝ × ℝ := (6, -1)
  let norm_squared (x : ℝ × ℝ) := x.1^2 + x.2^2
  { k : ℝ | norm_squared (k * v.1 - w.1, k * v.2 - w.2) = 34 } = {3, 1/13} := by
  sorry

end vector_norm_equation_solution_l4113_411370


namespace truth_telling_probability_l4113_411374

theorem truth_telling_probability (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.85) 
  (h_B : prob_B = 0.60) : 
  prob_A * prob_B = 0.51 := by
  sorry

end truth_telling_probability_l4113_411374


namespace rebecca_earring_ratio_l4113_411358

/-- Proves the ratio of gemstones to buttons for Rebecca's earrings --/
theorem rebecca_earring_ratio 
  (magnets_per_earring : ℕ)
  (buttons_to_magnets_ratio : ℚ)
  (sets_of_earrings : ℕ)
  (total_gemstones : ℕ)
  (h1 : magnets_per_earring = 2)
  (h2 : buttons_to_magnets_ratio = 1/2)
  (h3 : sets_of_earrings = 4)
  (h4 : total_gemstones = 24) :
  (total_gemstones : ℚ) / ((sets_of_earrings * 2 * magnets_per_earring * buttons_to_magnets_ratio) : ℚ) = 3 := by
  sorry

#check rebecca_earring_ratio

end rebecca_earring_ratio_l4113_411358


namespace factorization_equality_l4113_411376

theorem factorization_equality (x : ℝ) : 
  75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := by
  sorry

end factorization_equality_l4113_411376


namespace middle_manager_sample_size_l4113_411337

/-- Calculates the number of middle-level managers to be sampled in a stratified sampling scenario -/
theorem middle_manager_sample_size (total_employees : ℕ) (middle_managers : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : middle_managers = 150)
  (h3 : sample_size = 200) :
  (middle_managers : ℚ) / (total_employees : ℚ) * (sample_size : ℚ) = 30 := by
  sorry

end middle_manager_sample_size_l4113_411337


namespace dance_group_equality_l4113_411360

def dance_group_total (initial_boys initial_girls weekly_boys_increase weekly_girls_increase : ℕ) : ℕ :=
  let weeks := (initial_boys - initial_girls) / (weekly_girls_increase - weekly_boys_increase)
  let final_boys := initial_boys + weeks * weekly_boys_increase
  let final_girls := initial_girls + weeks * weekly_girls_increase
  final_boys + final_girls

theorem dance_group_equality (initial_boys initial_girls weekly_boys_increase weekly_girls_increase : ℕ) 
  (h1 : initial_boys = 39)
  (h2 : initial_girls = 23)
  (h3 : weekly_boys_increase = 6)
  (h4 : weekly_girls_increase = 8) :
  dance_group_total initial_boys initial_girls weekly_boys_increase weekly_girls_increase = 174 := by
  sorry

end dance_group_equality_l4113_411360


namespace problem_solution_l4113_411369

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ = 2)
  (h2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ = 15)
  (h3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ = 130) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ = 347 := by
  sorry

end problem_solution_l4113_411369


namespace fraction_decomposition_l4113_411343

theorem fraction_decomposition :
  ∀ (x : ℝ) (C D : ℚ),
    (C / (x - 2) + D / (3 * x + 7) = (3 * x^2 + 7 * x - 20) / (3 * x^2 - x - 14)) →
    (C = -14/13 ∧ D = 81/13) := by
  sorry

end fraction_decomposition_l4113_411343


namespace max_sundays_in_45_days_l4113_411306

/-- Represents the day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date within the first 45 days of a year -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Returns the number of Sundays in the first 45 days of a year -/
def countSundays (startDay : DayOfWeek) : Nat :=
  sorry

/-- The maximum number of Sundays in the first 45 days of a year -/
def maxSundays : Nat :=
  sorry

theorem max_sundays_in_45_days :
  maxSundays = 7 :=
sorry

end max_sundays_in_45_days_l4113_411306


namespace quadratic_distinct_roots_l4113_411300

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + m = 0 ∧ y^2 - 4*y + m = 0) → m < 4 :=
by sorry

end quadratic_distinct_roots_l4113_411300


namespace largest_a_for_fibonacci_sum_l4113_411333

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Property: Fₐ, Fᵦ, Fᶜ form an increasing arithmetic sequence -/
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

/-- Main theorem -/
theorem largest_a_for_fibonacci_sum (a b c : ℕ) :
  is_arithmetic_seq a b c →
  a + b + c ≤ 3000 →
  a ≤ 998 :=
by sorry

end largest_a_for_fibonacci_sum_l4113_411333


namespace triangle_unique_solution_l4113_411365

open Real

theorem triangle_unique_solution (a b : ℝ) (A : ℝ) (ha : a = 30) (hb : b = 25) (hA : A = 150 * π / 180) :
  ∃! B : ℝ, 0 < B ∧ B < π ∧ sin B = (b / a) * sin A :=
sorry

end triangle_unique_solution_l4113_411365


namespace dentist_age_problem_l4113_411313

/-- Given a dentist's current age and the relationship between his past and future ages,
    calculate how many years ago his age was being considered. -/
theorem dentist_age_problem (current_age : ℕ) (h : current_age = 32) : 
  ∃ (x : ℕ), (1 / 6 : ℚ) * (current_age - x) = (1 / 10 : ℚ) * (current_age + 8) ∧ x = 8 :=
by sorry

end dentist_age_problem_l4113_411313


namespace tangent_trapezoid_ratio_l4113_411331

/-- Represents a trapezoid with a circle tangent to two sides -/
structure TangentTrapezoid where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  parallel : ef ≠ gh
  /-- Circle with center Q on EF is tangent to FG and HE -/
  tangent : True

/-- The ratio of EQ to QF in the trapezoid -/
def ratio (t : TangentTrapezoid) : ℚ :=
  12 / 37

theorem tangent_trapezoid_ratio (t : TangentTrapezoid) 
  (h1 : t.ef = 40)
  (h2 : t.fg = 25)
  (h3 : t.gh = 12)
  (h4 : t.he = 35) :
  ratio t = 12 / 37 := by
  sorry

end tangent_trapezoid_ratio_l4113_411331


namespace hamburger_cost_satisfies_conditions_l4113_411354

/-- The cost of a pack of hamburger meat that satisfies the given conditions -/
def hamburger_cost : ℝ :=
  let crackers : ℝ := 3.50
  let vegetables : ℝ := 4 * 2.00
  let cheese : ℝ := 3.50
  let discount_rate : ℝ := 0.10
  let total_after_discount : ℝ := 18.00
  5.00

/-- Theorem stating that the hamburger cost satisfies the given conditions -/
theorem hamburger_cost_satisfies_conditions :
  let crackers : ℝ := 3.50
  let vegetables : ℝ := 4 * 2.00
  let cheese : ℝ := 3.50
  let discount_rate : ℝ := 0.10
  let total_after_discount : ℝ := 18.00
  total_after_discount = (hamburger_cost + crackers + vegetables + cheese) * (1 - discount_rate) := by
  sorry

#eval hamburger_cost

end hamburger_cost_satisfies_conditions_l4113_411354


namespace complement_of_M_l4113_411355

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M (x : ℝ) : x ∈ (Set.compl M) ↔ x < -2 ∨ x > 2 := by
  sorry

end complement_of_M_l4113_411355


namespace binomial_coefficient_n_minus_two_l4113_411391

theorem binomial_coefficient_n_minus_two (n : ℕ) (hn : n > 0) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
sorry

end binomial_coefficient_n_minus_two_l4113_411391


namespace school_attendance_problem_l4113_411338

theorem school_attendance_problem (boys : ℕ) (girls : ℕ) :
  boys = 2000 →
  (boys + girls : ℝ) = 1.4 * boys →
  girls = 800 := by
sorry

end school_attendance_problem_l4113_411338


namespace hyperbola_equation_proof_l4113_411346

/-- The hyperbola equation -/
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Point on asymptote condition -/
def point_on_asymptote (a b : ℝ) : Prop :=
  4 / 3 = b / a

/-- Perpendicular foci condition -/
def perpendicular_foci (c : ℝ) : Prop :=
  4 / (3 + c) * (4 / (3 - c)) = -1

/-- Relationship between a, b, and c -/
def foci_distance (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Main theorem -/
theorem hyperbola_equation_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : point_on_asymptote a b)
  (h_foci : ∃ c, perpendicular_foci c ∧ foci_distance a b c) :
  hyperbola_equation 3 4 x y :=
sorry

end hyperbola_equation_proof_l4113_411346


namespace seventh_term_is_84_l4113_411361

/-- A sequence where the differences between consecutive terms form a quadratic sequence -/
def CookieSequence (a : ℕ → ℕ) : Prop :=
  ∃ p q r : ℕ,
    (∀ n, a (n + 1) - a n = p * n * n + q * n + r) ∧
    a 1 = 5 ∧ a 2 = 9 ∧ a 3 = 14 ∧ a 4 = 22 ∧ a 5 = 35

theorem seventh_term_is_84 (a : ℕ → ℕ) (h : CookieSequence a) : a 7 = 84 := by
  sorry

end seventh_term_is_84_l4113_411361


namespace planted_field_fraction_l4113_411309

theorem planted_field_fraction (a b x : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : x = 3) :
  let total_area := (a * b) / 2
  let square_area := x^2
  let planted_area := total_area - square_area
  planted_area / total_area = 7 / 10 := by
sorry

end planted_field_fraction_l4113_411309


namespace questions_for_first_project_l4113_411308

/-- Given a mathematician who completes a fixed number of questions per day for a week and needs to write a specific number of questions for one project, this theorem calculates the number of questions for the other project. -/
theorem questions_for_first_project 
  (questions_per_day : ℕ) 
  (days_in_week : ℕ) 
  (questions_for_second_project : ℕ) 
  (h1 : questions_per_day = 142) 
  (h2 : days_in_week = 7) 
  (h3 : questions_for_second_project = 476) : 
  questions_per_day * days_in_week - questions_for_second_project = 518 := by
  sorry

#eval 142 * 7 - 476  -- Should output 518

end questions_for_first_project_l4113_411308


namespace sqrt_expression_equality_l4113_411318

theorem sqrt_expression_equality : 
  (Real.sqrt 48 - Real.sqrt 27) / Real.sqrt 3 + Real.sqrt 6 * 2 * Real.sqrt (1/3) = 1 + 2 * Real.sqrt 2 := by
  sorry

end sqrt_expression_equality_l4113_411318


namespace tan_sin_equation_l4113_411312

theorem tan_sin_equation (m : ℝ) : 
  Real.tan (20 * π / 180) + m * Real.sin (20 * π / 180) = Real.sqrt 3 → m = 4 := by
  sorry

end tan_sin_equation_l4113_411312


namespace fountain_area_l4113_411341

theorem fountain_area (AB DC : ℝ) (h1 : AB = 24) (h2 : DC = 14) : 
  let AD : ℝ := AB / 3
  let R : ℝ := Real.sqrt (AD^2 + DC^2)
  π * R^2 = 260 * π := by sorry

end fountain_area_l4113_411341


namespace sin_thirteen_pi_fourths_l4113_411322

theorem sin_thirteen_pi_fourths : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end sin_thirteen_pi_fourths_l4113_411322


namespace sqrt_sum_eq_8_implies_product_l4113_411351

theorem sqrt_sum_eq_8_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (25 - x) = 8 →
  (8 + x) * (25 - x) = 961 / 4 := by
sorry

end sqrt_sum_eq_8_implies_product_l4113_411351


namespace hyosung_mimi_distance_l4113_411330

/-- Calculates the remaining distance between two people walking towards each other. -/
def remaining_distance (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  initial_distance - (speed1 + speed2) * time

/-- Theorem stating the remaining distance between Hyosung and Mimi after 15 minutes. -/
theorem hyosung_mimi_distance :
  let initial_distance : ℝ := 2.5
  let mimi_speed : ℝ := 2.4
  let hyosung_speed : ℝ := 0.08 * 60
  let time : ℝ := 15 / 60
  remaining_distance initial_distance mimi_speed hyosung_speed time = 0.7 := by
  sorry

end hyosung_mimi_distance_l4113_411330


namespace factors_of_6000_l4113_411329

/-- The number of positive integer factors of a number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- The number of positive integer factors of a number that are perfect squares -/
def num_square_factors (n : ℕ) : ℕ := sorry

theorem factors_of_6000 :
  let n : ℕ := 6000
  let factorization : List (ℕ × ℕ) := [(2, 4), (3, 1), (5, 3)]
  (num_factors n = 40) ∧
  (num_factors n - num_square_factors n = 34) := by sorry

end factors_of_6000_l4113_411329


namespace last_colored_square_l4113_411380

/-- Represents a position in the rectangle --/
structure Position where
  row : Nat
  col : Nat

/-- Represents the dimensions of the rectangle --/
structure Dimensions where
  width : Nat
  height : Nat

/-- Represents the spiral coloring process --/
def spiralColor (dims : Dimensions) : Position :=
  sorry

/-- Theorem stating the last colored square in a 200x100 rectangle --/
theorem last_colored_square :
  spiralColor ⟨200, 100⟩ = ⟨51, 50⟩ := by
  sorry

end last_colored_square_l4113_411380


namespace foci_of_given_hyperbola_l4113_411311

/-- A hyperbola is defined by its equation and foci coordinates -/
structure Hyperbola where
  a_squared : ℝ
  b_squared : ℝ
  equation : (x y : ℝ) → Prop := λ x y => x^2 / a_squared - y^2 / b_squared = 1

/-- The foci of a hyperbola are the two fixed points used in its geometric definition -/
def foci (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = h.a_squared + h.b_squared ∧ p.2 = 0}

/-- The given hyperbola from the problem -/
def given_hyperbola : Hyperbola :=
  { a_squared := 7
    b_squared := 3 }

/-- The theorem states that the foci of the given hyperbola are (√10, 0) and (-√10, 0) -/
theorem foci_of_given_hyperbola :
  foci given_hyperbola = {(Real.sqrt 10, 0), (-Real.sqrt 10, 0)} := by
  sorry

end foci_of_given_hyperbola_l4113_411311


namespace triangle_reciprocal_sum_l4113_411349

/-- Given a triangle ABC with angle ratio A:B:C = 4:2:1, 
    prove that 1/a + 1/b = 1/c, where a, b, and c are the 
    sides opposite to angles A, B, and C respectively. -/
theorem triangle_reciprocal_sum (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_ratio : A = 4 * (π / 7) ∧ B = 2 * (π / 7) ∧ C = π / 7)
  (h_sides : a = 2 * Real.sin A ∧ b = 2 * Real.sin B ∧ c = 2 * Real.sin C) :
  1 / a + 1 / b = 1 / c :=
sorry

end triangle_reciprocal_sum_l4113_411349


namespace prime_fraction_sum_l4113_411398

theorem prime_fraction_sum (p q x y : ℕ) : 
  Prime p → Prime q → x > 0 → y > 0 → x < p → y < q → 
  (∃ k : ℤ, (p : ℚ) / x + (q : ℚ) / y = k) → x = y := by
sorry

end prime_fraction_sum_l4113_411398


namespace bubble_sort_probability_main_result_l4113_411387

def n : ℕ := 50

/-- The probability that r₂₅ ends up in the 35th position after one bubble pass -/
def probability : ℚ := 1 / 1190

theorem bubble_sort_probability (r : Fin n → ℕ) (h : Function.Injective r) :
  probability = (Nat.factorial 33) / (Nat.factorial 35) :=
sorry

theorem main_result : probability.num + probability.den = 1191 :=
sorry

end bubble_sort_probability_main_result_l4113_411387


namespace rectangle_perimeter_l4113_411353

/-- Given a triangle with sides 9, 12, and 15 units, and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6)
    (h5 : w * (a * b / 2 / w) = a * b / 2) : 2 * (w + a * b / 2 / w) = 30 := by
  sorry

end rectangle_perimeter_l4113_411353


namespace solution_for_given_condition_l4113_411348

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) / (x^2 - 1)

theorem solution_for_given_condition (a : ℝ) :
  (∀ x, f a x > 0 ↔ a > 1/3) → ∃ x, x = 3 ∧ f (1/3) x = 0 :=
by sorry

end solution_for_given_condition_l4113_411348


namespace quadratic_rational_root_even_coefficient_l4113_411342

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end quadratic_rational_root_even_coefficient_l4113_411342


namespace divisibility_property_l4113_411381

theorem divisibility_property (p : ℕ) (h_odd : Odd p) (h_gt_one : p > 1) :
  ∃ k : ℤ, (p - 1) ^ ((p - 1) / 2) - 1 = (p - 2) * k :=
sorry

end divisibility_property_l4113_411381


namespace quadratic_real_solutions_l4113_411389

theorem quadratic_real_solutions : ∃ (x : ℝ), x^2 + 3*x - 2 = 0 ∧
  (∀ (x : ℝ), 2*x^2 - x + 1 ≠ 0) ∧
  (∀ (x : ℝ), x^2 - 2*x + 2 ≠ 0) ∧
  (∀ (x : ℝ), x^2 + 2 ≠ 0) := by
  sorry

end quadratic_real_solutions_l4113_411389
