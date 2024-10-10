import Mathlib

namespace monday_temperature_l2010_201093

theorem monday_temperature
  (temp : Fin 5 → ℝ)
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 40)
  (some_day_42 : ∃ i, temp i = 42)
  (friday_10 : temp 4 = 10) :
  temp 0 = 42 := by
sorry

end monday_temperature_l2010_201093


namespace cricketer_stats_l2010_201040

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  wickets : ℕ
  runs : ℕ
  balls : ℕ

/-- Calculates the bowling average (runs per wicket) -/
def bowlingAverage (stats : BowlingStats) : ℚ :=
  stats.runs / stats.wickets

/-- Calculates the strike rate (balls per wicket) -/
def strikeRate (stats : BowlingStats) : ℚ :=
  stats.balls / stats.wickets

/-- Theorem about the cricketer's statistics -/
theorem cricketer_stats 
  (initial : BowlingStats) 
  (current_match : BowlingStats) 
  (new_stats : BowlingStats) :
  initial.wickets ≥ 50 →
  bowlingAverage initial = 124/10 →
  strikeRate initial = 30 →
  current_match.wickets = 5 →
  current_match.runs = 26 →
  bowlingAverage new_stats = bowlingAverage initial - 4/10 →
  strikeRate new_stats = 28 →
  new_stats.wickets = initial.wickets + current_match.wickets →
  new_stats.runs = initial.runs + current_match.runs →
  initial.wickets = 85 ∧ initial.balls = 2550 := by
  sorry


end cricketer_stats_l2010_201040


namespace simplify_expression_l2010_201068

theorem simplify_expression (a : ℝ) : 2*a*(2*a^2 + a) - a^2 = 4*a^3 + a^2 := by
  sorry

end simplify_expression_l2010_201068


namespace rahim_average_price_per_book_l2010_201044

/-- The average price per book given two purchases -/
def average_price_per_book (books1 : ℕ) (cost1 : ℕ) (books2 : ℕ) (cost2 : ℕ) : ℚ :=
  (cost1 + cost2) / (books1 + books2)

/-- Theorem: The average price per book for Rahim's purchases is 85 -/
theorem rahim_average_price_per_book :
  average_price_per_book 65 6500 35 2000 = 85 := by
  sorry

end rahim_average_price_per_book_l2010_201044


namespace rectangle_property_l2010_201079

/-- Represents a complex number --/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Represents a rectangle in the complex plane --/
structure ComplexRectangle where
  A : ComplexNumber
  B : ComplexNumber
  C : ComplexNumber
  D : ComplexNumber

/-- The theorem stating the properties of the given rectangle --/
theorem rectangle_property (rect : ComplexRectangle) :
  rect.A = ComplexNumber.mk 2 3 →
  rect.B = ComplexNumber.mk 3 2 →
  rect.C = ComplexNumber.mk (-2) (-3) →
  rect.D = ComplexNumber.mk (-3) (-2) :=
by sorry

end rectangle_property_l2010_201079


namespace parallelogram_max_area_l2010_201001

/-- Given a parallelogram with perimeter 60 units and one side three times the length of the other,
    the maximum possible area is 168.75 square units. -/
theorem parallelogram_max_area :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a = 3 * b →
  2 * a + 2 * b = 60 →
  ∀ (θ : ℝ),
  0 < θ → θ < π →
  a * b * Real.sin θ ≤ 168.75 :=
by
  sorry

end parallelogram_max_area_l2010_201001


namespace inverse_proportion_ratio_l2010_201054

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_ratio
  (x y : ℝ → ℝ)
  (h_inv_prop : InverselyProportional x y)
  (x₁ x₂ y₁ y₂ : ℝ)
  (h_x_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0)
  (h_y_nonzero : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_x_ratio : x₁ / x₂ = 4 / 5)
  (h_y_corr : y₁ = y (x.invFun x₁) ∧ y₂ = y (x.invFun x₂)) :
  y₁ / y₂ = 5 / 4 := by
  sorry

end inverse_proportion_ratio_l2010_201054


namespace square_diagonal_triangle_dimensions_l2010_201038

theorem square_diagonal_triangle_dimensions :
  ∀ (square_side : ℝ) (triangle_leg1 triangle_leg2 triangle_hypotenuse : ℝ),
    square_side = 10 →
    triangle_leg1 = square_side →
    triangle_leg2 = square_side →
    triangle_hypotenuse^2 = triangle_leg1^2 + triangle_leg2^2 →
    (triangle_leg1 = 10 ∧ triangle_leg2 = 10 ∧ triangle_hypotenuse = 10 * Real.sqrt 2) :=
by
  sorry

end square_diagonal_triangle_dimensions_l2010_201038


namespace intersection_complement_equals_set_l2010_201066

def U : Set ℕ := {x | x^2 - 4*x - 5 ≤ 0}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3, 5}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {0, 2} := by sorry

end intersection_complement_equals_set_l2010_201066


namespace equal_roots_quadratic_l2010_201037

theorem equal_roots_quadratic : ∃ (x : ℝ), x^2 - 2*x + 1 = 0 ∧ 
  ∀ (y : ℝ), y^2 - 2*y + 1 = 0 → y = x :=
by sorry

end equal_roots_quadratic_l2010_201037


namespace symmetric_hexagon_relationship_l2010_201000

/-- A hexagon that is both inscribed and circumscribed, and symmetric about the perpendicular bisector of one of its sides. -/
structure SymmetricHexagon where
  R : ℝ  -- radius of circumscribed circle
  r : ℝ  -- radius of inscribed circle
  c : ℝ  -- distance between centers of circles
  R_pos : 0 < R
  r_pos : 0 < r
  c_pos : 0 < c
  inscribed : True  -- represents that the hexagon is inscribed
  circumscribed : True  -- represents that the hexagon is circumscribed
  symmetric : True  -- represents that the hexagon is symmetric about the perpendicular bisector of one of its sides

/-- The relationship between R, r, and c for a symmetric hexagon -/
theorem symmetric_hexagon_relationship (h : SymmetricHexagon) :
  3 * (h.R^2 - h.c^2)^4 - 4 * h.r^2 * (h.R^2 - h.c^2)^2 * (h.R^2 + h.c^2) - 16 * h.R^2 * h.c^2 * h.r^4 = 0 := by
  sorry

end symmetric_hexagon_relationship_l2010_201000


namespace largest_multiple_of_60_with_7_and_0_l2010_201091

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def consists_of_7_and_0 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7 ∨ d = 0

theorem largest_multiple_of_60_with_7_and_0 :
  ∃ n : ℕ,
    is_multiple_of n 60 ∧
    consists_of_7_and_0 n ∧
    (∀ m : ℕ, m > n → ¬(is_multiple_of m 60 ∧ consists_of_7_and_0 m)) ∧
    n / 15 = 518 := by
  sorry

end largest_multiple_of_60_with_7_and_0_l2010_201091


namespace mixture_quantity_proof_l2010_201043

theorem mixture_quantity_proof (petrol kerosene diesel : ℝ) 
  (h1 : petrol / kerosene = 3 / 2)
  (h2 : petrol / diesel = 3 / 5)
  (h3 : (petrol - 6) / ((kerosene - 4) + 20) = 2 / 3)
  (h4 : (petrol - 6) / (diesel - 10) = 2 / 5)
  (h5 : petrol + kerosene + diesel > 0) :
  petrol + kerosene + diesel = 100 := by
sorry

end mixture_quantity_proof_l2010_201043


namespace tens_digit_of_3_to_2017_l2010_201048

theorem tens_digit_of_3_to_2017 : ∃ n : ℕ, 3^2017 ≡ 87 + 100*n [ZMOD 100] := by
  sorry

end tens_digit_of_3_to_2017_l2010_201048


namespace currency_multiplication_invalid_l2010_201086

-- Define currency types
inductive Currency
| Ruble
| Kopeck

-- Define a structure for money
structure Money where
  amount : ℚ
  currency : Currency

-- Define conversion rate
def conversionRate : ℚ := 100

-- Define equality for Money
def Money.eq (a b : Money) : Prop :=
  (a.currency = b.currency ∧ a.amount = b.amount) ∨
  (a.currency = Currency.Ruble ∧ b.currency = Currency.Kopeck ∧ a.amount * conversionRate = b.amount) ∨
  (a.currency = Currency.Kopeck ∧ b.currency = Currency.Ruble ∧ a.amount = b.amount * conversionRate)

-- Define multiplication for Money (this operation is not well-defined for real currencies)
def Money.mul (a b : Money) : Money :=
  { amount := a.amount * b.amount,
    currency := 
      match a.currency, b.currency with
      | Currency.Ruble, Currency.Ruble => Currency.Ruble
      | Currency.Kopeck, Currency.Kopeck => Currency.Kopeck
      | _, _ => Currency.Ruble }

-- Theorem statement
theorem currency_multiplication_invalid :
  ∃ (a b c d : Money),
    Money.eq a b ∧ Money.eq c d ∧
    ¬(Money.eq (Money.mul a c) (Money.mul b d)) := by
  sorry

end currency_multiplication_invalid_l2010_201086


namespace problem_one_problem_two_l2010_201039

-- Problem 1
theorem problem_one : (Real.sqrt 12 - Real.sqrt 6) / Real.sqrt 3 + 2 / Real.sqrt 2 = 2 := by sorry

-- Problem 2
theorem problem_two : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) + (2 - Real.sqrt 3)^2 = 8 - 4 * Real.sqrt 3 := by sorry

end problem_one_problem_two_l2010_201039


namespace sqrt_23_minus_1_lt_4_l2010_201053

theorem sqrt_23_minus_1_lt_4 : Real.sqrt 23 - 1 < 4 := by sorry

end sqrt_23_minus_1_lt_4_l2010_201053


namespace lowest_dropped_score_l2010_201085

theorem lowest_dropped_score (scores : Fin 4 → ℕ) 
  (avg_all : (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 45)
  (avg_after_drop : ∃ i, (scores 0 + scores 1 + scores 2 + scores 3 - scores i) / 3 = 50) :
  ∃ i, scores i = 30 ∧ ∀ j, scores j ≥ scores i := by
  sorry

end lowest_dropped_score_l2010_201085


namespace tangent_perpendicular_to_line_l2010_201002

/-- The curve y = x^3 + 2x has a tangent line at (1, 3) perpendicular to ax - y + 2019 = 0 -/
theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f (x : ℝ) := x^3 + 2*x
  let f' (x : ℝ) := 3*x^2 + 2
  let tangent_slope := f' 1
  let perpendicular_slope := a
  (f 1 = 3) ∧ (tangent_slope * perpendicular_slope = -1) → a = -1/5 := by
  sorry

end tangent_perpendicular_to_line_l2010_201002


namespace f_properties_l2010_201065

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi ∧
  (∀ (x y : ℝ), x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) →
    y ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) → x < y → f y < f x) :=
by sorry

end f_properties_l2010_201065


namespace smallest_angle_quadrilateral_l2010_201021

theorem smallest_angle_quadrilateral (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a + b + c + d = 360) →
  (b = 5/4 * a) → (c = 3/2 * a) → (d = 7/4 * a) →
  a = 720 / 11 := by
sorry

end smallest_angle_quadrilateral_l2010_201021


namespace simplify_expression_l2010_201034

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : x/y + y/x - 3/(x*y) = 1 := by
  sorry

end simplify_expression_l2010_201034


namespace triangle_existence_l2010_201055

theorem triangle_existence (n : ℕ) (h : n ≥ 2) : 
  ∃ (points : Finset (Fin (2*n))) (segments : Finset (Fin (2*n) × Fin (2*n))),
    Finset.card segments = n^2 + 1 →
    ∃ (a b c : Fin (2*n)), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (a, c) ∈ segments :=
by sorry


end triangle_existence_l2010_201055


namespace model_M_completion_time_l2010_201022

/-- The time (in minutes) taken by a model N computer to complete the task -/
def model_N_time : ℝ := 18

/-- The number of model M computers used -/
def num_model_M : ℝ := 12

/-- The time (in minutes) taken to complete the task when using both models -/
def total_time : ℝ := 1

/-- The time (in minutes) taken by a model M computer to complete the task -/
def model_M_time : ℝ := 36

theorem model_M_completion_time :
  (num_model_M / model_M_time + num_model_M / model_N_time) * total_time = num_model_M :=
by sorry

end model_M_completion_time_l2010_201022


namespace g_continuous_c_plus_d_equals_negative_three_l2010_201031

-- Define the piecewise function g(x)
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - d

-- Theorem stating the continuity condition
theorem g_continuous (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) ↔ c = -4 ∧ d = 1 := by
  sorry

-- Corollary for the sum of c and d
theorem c_plus_d_equals_negative_three (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) → c + d = -3 := by
  sorry

end g_continuous_c_plus_d_equals_negative_three_l2010_201031


namespace prob_A_leading_after_three_prob_B_wins_3_2_l2010_201045

-- Define the probability of Team A winning a single game
def p_A_win : ℝ := 0.60

-- Define the probability of Team B winning a single game
def p_B_win : ℝ := 1 - p_A_win

-- Define the number of games needed to win the match
def games_to_win : ℕ := 3

-- Define the total number of games in a full match
def total_games : ℕ := 5

-- Theorem for the probability of Team A leading after the first three games
theorem prob_A_leading_after_three : 
  (Finset.sum (Finset.range 2) (λ k => Nat.choose 3 (3 - k) * p_A_win ^ (3 - k) * p_B_win ^ k)) = 0.648 := by sorry

-- Theorem for the probability of Team B winning the match with a score of 3:2
theorem prob_B_wins_3_2 : 
  (Nat.choose 4 2 * p_A_win ^ 2 * p_B_win ^ 2 * p_B_win) = 0.138 := by sorry

end prob_A_leading_after_three_prob_B_wins_3_2_l2010_201045


namespace unique_solution_l2010_201062

def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ Nat.gcd a b = 1 ∧ (a + 12) * b = 3 * a * (b + 12)

theorem unique_solution : ∀ a b : ℕ, is_solution a b ↔ a = 2 ∧ b = 9 := by sorry

end unique_solution_l2010_201062


namespace area_fraction_on_7x7_grid_l2010_201081

/-- Represents a square grid of points -/
structure PointGrid :=
  (size : ℕ)

/-- Represents a square on the grid -/
structure GridSquare :=
  (sideLength : ℕ)

/-- The larger square formed by the outer points of the grid -/
def outerSquare (grid : PointGrid) : GridSquare :=
  { sideLength := grid.size - 1 }

/-- The shaded square inside the grid -/
def innerSquare : GridSquare :=
  { sideLength := 2 }

/-- Calculate the area of a square -/
def area (square : GridSquare) : ℕ :=
  square.sideLength * square.sideLength

/-- The fraction of the outer square's area occupied by the inner square -/
def areaFraction (grid : PointGrid) : ℚ :=
  (area innerSquare : ℚ) / (area (outerSquare grid))

theorem area_fraction_on_7x7_grid :
  areaFraction { size := 7 } = 1 / 9 := by
  sorry

end area_fraction_on_7x7_grid_l2010_201081


namespace complex_power_difference_l2010_201090

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := by sorry

end complex_power_difference_l2010_201090


namespace female_democrat_ratio_is_half_l2010_201050

/-- Represents the number of participants in a meeting with given conditions -/
structure Meeting where
  total : ℕ
  maleDemocratRatio : ℚ
  totalDemocratRatio : ℚ
  femaleDemocrats : ℕ
  male : ℕ
  female : ℕ

/-- The ratio of female democrats to total female participants -/
def femaleDemocratRatio (m : Meeting) : ℚ :=
  m.femaleDemocrats / m.female

theorem female_democrat_ratio_is_half (m : Meeting) 
  (h1 : m.total = 660)
  (h2 : m.maleDemocratRatio = 1/4)
  (h3 : m.totalDemocratRatio = 1/3)
  (h4 : m.femaleDemocrats = 110)
  (h5 : m.male + m.female = m.total)
  (h6 : m.maleDemocratRatio * m.male + m.femaleDemocrats = m.totalDemocratRatio * m.total) :
  femaleDemocratRatio m = 1/2 := by
  sorry

end female_democrat_ratio_is_half_l2010_201050


namespace hat_number_sum_l2010_201026

theorem hat_number_sum : ∀ (alice_num bob_num : ℕ),
  alice_num ∈ Finset.range 51 →
  bob_num ∈ Finset.range 51 →
  alice_num ≠ bob_num →
  (∃ (x : ℕ), alice_num < x ∧ x ≤ 50) →
  (∃ (y : ℕ), y < bob_num ∧ y ≤ 50) →
  bob_num % 3 = 0 →
  ∃ (k : ℕ), 2 * bob_num + alice_num = k^2 →
  alice_num + bob_num = 22 :=
by sorry

end hat_number_sum_l2010_201026


namespace one_third_1206_is_100_5_percent_of_400_l2010_201064

theorem one_third_1206_is_100_5_percent_of_400 :
  (1206 / 3) / 400 = 1.005 := by
  sorry

end one_third_1206_is_100_5_percent_of_400_l2010_201064


namespace dog_travel_time_l2010_201083

theorem dog_travel_time (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 20)
  (h2 : speed1 = 10)
  (h3 : speed2 = 5) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 3 := by
  sorry

end dog_travel_time_l2010_201083


namespace interval_intersection_l2010_201023

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1/2 < x ∧ x < 0.6) := by
  sorry

end interval_intersection_l2010_201023


namespace julia_puppy_cost_l2010_201032

def adoption_fee : ℝ := 20
def dog_food : ℝ := 20
def treats_price : ℝ := 2.5
def treats_quantity : ℕ := 2
def toys : ℝ := 15
def crate : ℝ := 20
def bed : ℝ := 20
def collar_leash : ℝ := 15
def discount_rate : ℝ := 0.2

def total_cost : ℝ :=
  adoption_fee +
  (1 - discount_rate) * (dog_food + treats_price * treats_quantity + toys + crate + bed + collar_leash)

theorem julia_puppy_cost :
  total_cost = 96 := by sorry

end julia_puppy_cost_l2010_201032


namespace margin_in_terms_of_selling_price_l2010_201047

theorem margin_in_terms_of_selling_price
  (C S M n : ℝ)
  (h1 : M = (2 / n) * C)
  (h2 : S - M = C)
  : M = 2 * S / (n + 2) := by
sorry

end margin_in_terms_of_selling_price_l2010_201047


namespace mikis_sandcastle_height_l2010_201078

/-- The height of Miki's sandcastle given the height of her sister's sandcastle and the difference in height -/
theorem mikis_sandcastle_height 
  (sisters_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : sisters_height = 0.5)
  (h2 : height_difference = 0.3333333333333333) : 
  sisters_height + height_difference = 0.8333333333333333 :=
by sorry

end mikis_sandcastle_height_l2010_201078


namespace sin_alpha_value_l2010_201074

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := by
  sorry

end sin_alpha_value_l2010_201074


namespace chess_positions_l2010_201082

/-- The number of different positions on a chessboard after both players make one move each -/
def num_positions : ℕ :=
  let pawns_per_player := 8
  let knights_per_player := 2
  let pawn_moves := 2
  let knight_moves := 2
  let moves_per_player := pawns_per_player * pawn_moves + knights_per_player * knight_moves
  moves_per_player * moves_per_player

theorem chess_positions : num_positions = 400 := by
  sorry

end chess_positions_l2010_201082


namespace difference_of_a_and_reciprocal_l2010_201007

theorem difference_of_a_and_reciprocal (a : ℝ) (h : a + 1/a = Real.sqrt 13) :
  a - 1/a = 3 ∨ a - 1/a = -3 := by
sorry

end difference_of_a_and_reciprocal_l2010_201007


namespace fixed_point_of_exponential_function_l2010_201041

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) - 1
  f (-1) = 0 ∧ ∀ x : ℝ, f x = 0 → x = -1 := by
  sorry

end fixed_point_of_exponential_function_l2010_201041


namespace complex_number_equality_l2010_201059

theorem complex_number_equality : ∀ z : ℂ, z = (Complex.I ^ 3) / (1 + Complex.I) → z = (-1 - Complex.I) / 2 := by
  sorry

end complex_number_equality_l2010_201059


namespace angus_token_count_l2010_201063

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

/-- The value of each token in dollars -/
def token_value : ℕ := 4

/-- The difference in token value between Elsa and Angus in dollars -/
def token_value_difference : ℕ := 20

/-- The number of tokens Angus has -/
def angus_tokens : ℕ := elsa_tokens - (token_value_difference / token_value)

theorem angus_token_count : angus_tokens = 55 := by
  sorry

end angus_token_count_l2010_201063


namespace sqrt_sum_equals_four_l2010_201009

theorem sqrt_sum_equals_four : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 := by
sorry

end sqrt_sum_equals_four_l2010_201009


namespace expression_values_l2010_201027

theorem expression_values : 
  (0.64^(-1/2) - (-1/8)^0 + 8^(2/3) + (9/16)^(1/2) = 6) ∧ 
  (Real.log 2^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1) := by
  sorry

end expression_values_l2010_201027


namespace complex_equation_solution_l2010_201057

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l2010_201057


namespace max_choir_members_correct_l2010_201015

/-- The maximum number of choir members that satisfies the given conditions. -/
def max_choir_members : ℕ := 266

/-- Predicate to check if a number satisfies the square formation condition. -/
def is_square_formation (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k + 11

/-- Predicate to check if a number satisfies the rectangular formation condition. -/
def is_rectangular_formation (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 5)

/-- Theorem stating that max_choir_members satisfies both formation conditions
    and is the maximum number that does so. -/
theorem max_choir_members_correct :
  is_square_formation max_choir_members ∧
  is_rectangular_formation max_choir_members ∧
  ∀ m : ℕ, m > max_choir_members →
    ¬(is_square_formation m ∧ is_rectangular_formation m) :=
by sorry

end max_choir_members_correct_l2010_201015


namespace self_checkout_increase_is_20_percent_l2010_201088

/-- The percentage increase in complaints when the self-checkout is broken -/
def self_checkout_increase (normal_complaints : ℕ) (short_staffed_increase : ℚ) (total_complaints : ℕ) : ℚ :=
  let short_staffed_complaints := normal_complaints * (1 + short_staffed_increase)
  let daily_complaints_both := total_complaints / 3
  (daily_complaints_both - short_staffed_complaints) / short_staffed_complaints * 100

/-- Theorem stating that the percentage increase when self-checkout is broken is 20% -/
theorem self_checkout_increase_is_20_percent :
  self_checkout_increase 120 (1/3) 576 = 20 := by
  sorry

end self_checkout_increase_is_20_percent_l2010_201088


namespace number_line_steps_l2010_201056

theorem number_line_steps (total_distance : ℝ) (num_steps : ℕ) (step_to_x : ℕ) : 
  total_distance = 32 →
  num_steps = 8 →
  step_to_x = 6 →
  (total_distance / num_steps) * step_to_x = 24 := by
sorry

end number_line_steps_l2010_201056


namespace max_a_items_eleven_a_items_possible_l2010_201052

/-- Represents the number of items purchased for each stationery type -/
structure Stationery where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of the stationery purchase -/
def totalCost (s : Stationery) : ℕ :=
  3 * s.a + 2 * s.b + s.c

/-- Checks if the purchase satisfies all conditions -/
def isValidPurchase (s : Stationery) : Prop :=
  s.b = s.a - 2 ∧
  3 * s.a ≤ 33 ∧
  totalCost s = 66

/-- Theorem stating that the maximum number of A items that can be purchased is 11 -/
theorem max_a_items : ∀ s : Stationery, isValidPurchase s → s.a ≤ 11 :=
  sorry

/-- Theorem stating that 11 A items can actually be purchased -/
theorem eleven_a_items_possible : ∃ s : Stationery, isValidPurchase s ∧ s.a = 11 :=
  sorry

end max_a_items_eleven_a_items_possible_l2010_201052


namespace old_edition_pages_l2010_201029

/-- The number of pages in the new edition of the Geometry book -/
def new_edition_pages : ℕ := 450

/-- The difference between twice the number of pages in the old edition and the new edition -/
def page_difference : ℕ := 230

/-- Theorem stating that the old edition of the Geometry book had 340 pages -/
theorem old_edition_pages : 
  ∃ (x : ℕ), 2 * x - page_difference = new_edition_pages ∧ x = 340 := by
  sorry

end old_edition_pages_l2010_201029


namespace smallest_prime_divisor_of_sum_l2010_201095

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^24 + 8^15) ∧ ∀ q, Nat.Prime q → q ∣ (3^24 + 8^15) → p ≤ q := by
  sorry

end smallest_prime_divisor_of_sum_l2010_201095


namespace container_capacity_l2010_201042

theorem container_capacity : 
  ∀ (capacity : ℝ), 
  (1/4 : ℝ) * capacity + 120 = (2/3 : ℝ) * capacity → 
  capacity = 288 := by
  sorry

end container_capacity_l2010_201042


namespace average_of_first_20_multiples_of_17_l2010_201013

theorem average_of_first_20_multiples_of_17 : 
  let n : ℕ := 20
  let first_multiple : ℕ := 17
  let sum_of_multiples : ℕ := n * (first_multiple + n * first_multiple) / 2
  (sum_of_multiples : ℚ) / n = 178.5 := by sorry

end average_of_first_20_multiples_of_17_l2010_201013


namespace cookie_milk_calculation_l2010_201060

/-- Given that 12 cookies require 2 quarts of milk and 1 quart equals 2 pints,
    prove that 3 cookies require 1 pint of milk. -/
theorem cookie_milk_calculation 
  (cookies_per_recipe : ℕ := 12)
  (quarts_per_recipe : ℚ := 2)
  (pints_per_quart : ℕ := 2)
  (target_cookies : ℕ := 3) :
  let pints_per_recipe := quarts_per_recipe * pints_per_quart
  let pints_per_cookie := pints_per_recipe / cookies_per_recipe
  target_cookies * pints_per_cookie = 1 := by
sorry

end cookie_milk_calculation_l2010_201060


namespace kangaroo_jumps_l2010_201028

theorem kangaroo_jumps (time_for_4_jumps : ℝ) (jumps_to_calculate : ℕ) : 
  time_for_4_jumps = 6 → jumps_to_calculate = 30 → 
  (time_for_4_jumps / 4) * jumps_to_calculate = 45 := by
sorry

end kangaroo_jumps_l2010_201028


namespace vector_equation_l2010_201024

/-- Given vectors a, b, and c in ℝ², prove that if a = (5, 2), b = (-4, -3), 
    and 3a - 2b + c = 0, then c = (-23, -12). -/
theorem vector_equation (a b c : ℝ × ℝ) : 
  a = (5, 2) → 
  b = (-4, -3) → 
  3 • a - 2 • b + c = (0, 0) → 
  c = (-23, -12) := by sorry

end vector_equation_l2010_201024


namespace function_identity_proof_l2010_201073

theorem function_identity_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, (n - 1)^2 < f n * f (f n) ∧ f n * f (f n) < n^2 + n) : 
  ∀ n : ℕ, f n = n := by
  sorry

end function_identity_proof_l2010_201073


namespace reading_time_calculation_l2010_201098

def total_time : ℕ := 120
def piano_time : ℕ := 30
def writing_time : ℕ := 25
def exerciser_time : ℕ := 27

theorem reading_time_calculation :
  total_time - piano_time - writing_time - exerciser_time = 38 := by
  sorry

end reading_time_calculation_l2010_201098


namespace distribution_6_boxes_8_floors_l2010_201004

/-- The number of ways to distribute boxes among floors with at least two on the top floor -/
def distributionWays (numBoxes numFloors : ℕ) : ℕ :=
  numFloors^numBoxes - (numFloors - 1)^numBoxes - numBoxes * (numFloors - 1)^(numBoxes - 1)

/-- Theorem: For 6 boxes and 8 floors, the number of distributions with at least 2 boxes on the top floor -/
theorem distribution_6_boxes_8_floors :
  distributionWays 6 8 = 8^6 - 13 * 7^5 := by
  sorry

end distribution_6_boxes_8_floors_l2010_201004


namespace cube_triangle_areas_sum_l2010_201030

/-- Represents a 2 × 2 × 2 cube -/
structure Cube where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The sum of areas of all triangles with vertices on the cube -/
def sum_triangle_areas (c : Cube) : ℝ := sorry

/-- The sum can be expressed as m + √n + √p -/
def sum_representation (m n p : ℕ) (c : Cube) : Prop :=
  sum_triangle_areas c = m + Real.sqrt n + Real.sqrt p

theorem cube_triangle_areas_sum (c : Cube) :
  ∃ (m n p : ℕ), sum_representation m n p c ∧ m + n + p = 5424 := by sorry

end cube_triangle_areas_sum_l2010_201030


namespace hens_count_l2010_201016

/-- Given a total number of heads and feet, and the number of feet for hens and cows,
    calculate the number of hens. -/
def count_hens (total_heads : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : ℕ :=
  sorry

theorem hens_count :
  let total_heads := 44
  let total_feet := 140
  let hen_feet := 2
  let cow_feet := 4
  count_hens total_heads total_feet hen_feet cow_feet = 18 := by
  sorry

end hens_count_l2010_201016


namespace optimal_schedule_l2010_201096

/-- Represents the construction teams -/
inductive Team
| A
| B

/-- Represents the construction schedule -/
structure Schedule where
  teamA_months : ℕ
  teamB_months : ℕ

/-- Calculates the total work done by a team given the months worked and their efficiency -/
def work_done (months : ℕ) (efficiency : ℚ) : ℚ :=
  months * efficiency

/-- Calculates the cost of a schedule given the monthly rates -/
def schedule_cost (s : Schedule) (rateA rateB : ℕ) : ℕ :=
  s.teamA_months * rateA + s.teamB_months * rateB

/-- Checks if a schedule is valid according to the given constraints -/
def is_valid_schedule (s : Schedule) : Prop :=
  s.teamA_months > 0 ∧ s.teamA_months ≤ 6 ∧
  s.teamB_months > 0 ∧ s.teamB_months ≤ 24 ∧
  s.teamA_months + s.teamB_months ≤ 24

/-- The main theorem to be proved -/
theorem optimal_schedule :
  ∃ (s : Schedule),
    is_valid_schedule s ∧
    work_done s.teamA_months (1 / 18) + work_done s.teamB_months (1 / 27) = 1 ∧
    ∀ (s' : Schedule),
      is_valid_schedule s' ∧
      work_done s'.teamA_months (1 / 18) + work_done s'.teamB_months (1 / 27) = 1 →
      schedule_cost s 80000 50000 ≤ schedule_cost s' 80000 50000 ∧
    s.teamA_months = 2 ∧ s.teamB_months = 24 :=
  sorry

end optimal_schedule_l2010_201096


namespace hiker_speed_l2010_201070

/-- Proves that given a cyclist traveling at 10 miles per hour who passes a hiker, 
    stops 5 minutes later, and waits 7.5 minutes for the hiker to catch up, 
    the hiker's constant speed is 50/7.5 miles per hour. -/
theorem hiker_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) :
  cyclist_speed = 10 →
  cyclist_travel_time = 5 / 60 →
  hiker_catch_up_time = 7.5 / 60 →
  (cyclist_speed * cyclist_travel_time) / hiker_catch_up_time = 50 / 7.5 := by
  sorry

#eval (50 : ℚ) / 7.5

end hiker_speed_l2010_201070


namespace vector_collinearity_l2010_201018

/-- Given vectors a and b, prove that k makes k*a + b collinear with a - 3*b -/
theorem vector_collinearity (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 2) →
  b = (-3, 2) →
  k = -1/3 →
  ∃ (t : ℝ), t • (k • a + b) = a - 3 • b := by
  sorry

end vector_collinearity_l2010_201018


namespace f_properties_l2010_201080

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := sin x + m * x

theorem f_properties (m : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ (deriv (f m)) x₁ = (deriv (f m)) x₂) ∧
  (∃ s : ℕ → ℝ, ∀ i j, i ≠ j → (deriv (f m)) (s i) = (deriv (f m)) (s j)) ∧
  (∃ t : ℕ → ℝ, ∀ i j, (deriv (f m)) (t i) = (deriv (f m)) (t j)) :=
sorry

end f_properties_l2010_201080


namespace beams_per_panel_is_two_l2010_201014

/-- Represents the number of fence panels in the fence -/
def num_panels : ℕ := 10

/-- Represents the number of metal sheets in each fence panel -/
def sheets_per_panel : ℕ := 3

/-- Represents the number of metal rods in each sheet -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal rods in each beam -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the fence -/
def total_rods : ℕ := 380

/-- Calculates the number of metal beams in each fence panel -/
def beams_per_panel : ℕ := 
  let total_sheets := num_panels * sheets_per_panel
  let rods_for_sheets := total_sheets * rods_per_sheet
  let remaining_rods := total_rods - rods_for_sheets
  let total_beams := remaining_rods / rods_per_beam
  total_beams / num_panels

/-- Theorem stating that the number of metal beams in each fence panel is 2 -/
theorem beams_per_panel_is_two : beams_per_panel = 2 := by sorry

end beams_per_panel_is_two_l2010_201014


namespace exists_divisible_by_sum_of_digits_l2010_201092

/-- Sum of digits of a three-digit number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- Theorem: Among any 18 consecutive three-digit numbers, there exists one divisible by its sum of digits -/
theorem exists_divisible_by_sum_of_digits (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sumOfDigits k = 0 := by
  sorry

end exists_divisible_by_sum_of_digits_l2010_201092


namespace fred_balloons_l2010_201072

theorem fred_balloons (initial_balloons given_balloons : ℕ) 
  (h1 : initial_balloons = 709)
  (h2 : given_balloons = 221) :
  initial_balloons - given_balloons = 488 :=
by sorry

end fred_balloons_l2010_201072


namespace geometric_sequence_problem_l2010_201017

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_roots : a 4 > 0 ∧ a 8 > 0 ∧ a 4^2 - 4*a 4 + 3 = 0 ∧ a 8^2 - 4*a 8 + 3 = 0) :
  a 6 = Real.sqrt 3 := by
  sorry

end geometric_sequence_problem_l2010_201017


namespace starting_number_sequence_l2010_201049

theorem starting_number_sequence (n : ℕ) : 
  (n ≤ 79) →                          -- Last number is less than or equal to 79
  (n % 11 = 0) →                      -- Last number is divisible by 11
  (∃ (m : ℕ), n = m * 11) →           -- n is a multiple of 11
  (∃ (k : ℕ), n = 11 * 7 - k * 11) →  -- n is the 7th number in the sequence
  (11 : ℕ) = n - 6 * 11               -- Starting number is 11
  := by sorry

end starting_number_sequence_l2010_201049


namespace clock_cost_price_l2010_201008

/-- The cost price of each clock -/
def cost_price : ℝ := 125

/-- The number of clocks purchased -/
def total_clocks : ℕ := 150

/-- The number of clocks sold at 12% gain -/
def clocks_12_percent : ℕ := 60

/-- The number of clocks sold at 18% gain -/
def clocks_18_percent : ℕ := 90

/-- The gain percentage for the first group of clocks -/
def gain_12_percent : ℝ := 0.12

/-- The gain percentage for the second group of clocks -/
def gain_18_percent : ℝ := 0.18

/-- The uniform gain percentage -/
def uniform_gain : ℝ := 0.16

/-- The difference in total selling price -/
def price_difference : ℝ := 75

theorem clock_cost_price : 
  (clocks_12_percent : ℝ) * cost_price * (1 + gain_12_percent) + 
  (clocks_18_percent : ℝ) * cost_price * (1 + gain_18_percent) = 
  (total_clocks : ℝ) * cost_price * (1 + uniform_gain) + price_difference :=
by sorry

end clock_cost_price_l2010_201008


namespace sufficient_condition_for_a_gt_b_l2010_201075

theorem sufficient_condition_for_a_gt_b (a b : ℝ) : 
  (1 / a < 1 / b) ∧ (1 / b < 0) → a > b := by
  sorry

end sufficient_condition_for_a_gt_b_l2010_201075


namespace optimal_sapling_positions_l2010_201003

/-- Represents the number of trees planted -/
def num_trees : ℕ := 20

/-- Represents the distance between adjacent trees in meters -/
def tree_spacing : ℕ := 10

/-- Calculates the total distance walked by students for given sapling positions -/
def total_distance (pos1 pos2 : ℕ) : ℕ := sorry

/-- Theorem stating that positions 10 and 11 minimize the total distance -/
theorem optimal_sapling_positions :
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ num_trees →
    total_distance 10 11 ≤ total_distance a b :=
by sorry

end optimal_sapling_positions_l2010_201003


namespace percentage_calculation_l2010_201084

theorem percentage_calculation (N I P : ℝ) : 
  N = 93.75 →
  I = 0.4 * N →
  (P / 100) * I = 6 →
  P = 16 := by
sorry

end percentage_calculation_l2010_201084


namespace cone_distance_theorem_l2010_201011

/-- Represents a right circular cone -/
structure RightCircularCone where
  slantHeight : ℝ
  topRadius : ℝ

/-- The shortest distance between two points on a cone's surface -/
def shortestDistance (cone : RightCircularCone) (pointA pointB : ℝ × ℝ) : ℝ := sorry

theorem cone_distance_theorem (cone : RightCircularCone) 
  (h1 : cone.slantHeight = 21)
  (h2 : cone.topRadius = 14) :
  let midpoint : ℝ × ℝ := (cone.slantHeight / 2, 0)
  let oppositePoint : ℝ × ℝ := (cone.slantHeight / 2, cone.topRadius)
  Int.floor (shortestDistance cone midpoint oppositePoint) = 18 := by sorry

end cone_distance_theorem_l2010_201011


namespace power_of_product_square_l2010_201033

theorem power_of_product_square (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_of_product_square_l2010_201033


namespace arithmetic_computation_l2010_201076

theorem arithmetic_computation : 12 + 4 * (2 * 3 - 8 + 1)^2 = 16 := by
  sorry

end arithmetic_computation_l2010_201076


namespace student_count_l2010_201097

theorem student_count (stars_per_student : ℕ) (total_stars : ℕ) (h1 : stars_per_student = 3) (h2 : total_stars = 372) :
  total_stars / stars_per_student = 124 := by
  sorry

end student_count_l2010_201097


namespace function_behavior_l2010_201019

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_increasing : ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
variable (h_f7 : f 7 = 6)

-- State the theorem
theorem function_behavior :
  (∀ x y, -7 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f y ≤ f x) ∧
  (∀ x, -7 ≤ x ∧ x ≤ 7 → f x ≤ 6) :=
sorry

end function_behavior_l2010_201019


namespace square_number_problem_l2010_201077

theorem square_number_problem : ∃ x : ℤ, 
  (∃ m : ℤ, x + 15 = m^2) ∧ 
  (∃ n : ℤ, x - 74 = n^2) ∧ 
  x = 2010 := by
  sorry

end square_number_problem_l2010_201077


namespace jim_savings_rate_l2010_201025

/-- 
Given:
- Sara has already saved 4100 dollars
- Sara saves 10 dollars per week
- Jim saves x dollars per week
- After 820 weeks, Sara and Jim have saved the same amount

Prove: x = 15
-/

theorem jim_savings_rate (x : ℚ) : 
  (4100 + 820 * 10 = 820 * x) → x = 15 := by
  sorry

end jim_savings_rate_l2010_201025


namespace sam_final_marbles_l2010_201087

/-- Represents the number of marbles each person has -/
structure Marbles where
  steve : ℕ
  sam : ℕ
  sally : ℕ

/-- Represents the initial distribution of marbles -/
def initial_marbles (steve_marbles : ℕ) : Marbles :=
  { steve := steve_marbles,
    sam := 2 * steve_marbles,
    sally := 2 * steve_marbles - 5 }

/-- Represents the distribution of marbles after the exchange -/
def final_marbles (m : Marbles) : Marbles :=
  { steve := m.steve + 3,
    sam := m.sam - 6,
    sally := m.sally + 3 }

/-- Theorem stating that Sam ends up with 8 marbles -/
theorem sam_final_marbles :
  ∀ (initial : Marbles),
    initial.sam = 2 * initial.steve →
    initial.sally = initial.sam - 5 →
    (final_marbles initial).steve = 10 →
    (final_marbles initial).sam = 8 :=
by sorry

end sam_final_marbles_l2010_201087


namespace outfits_count_l2010_201071

theorem outfits_count (shirts : ℕ) (hats : ℕ) : shirts = 5 → hats = 3 → shirts * hats = 15 := by
  sorry

end outfits_count_l2010_201071


namespace license_plate_count_l2010_201051

def alphabet_size : ℕ := 26
def digit_count : ℕ := 10

def license_plate_combinations : ℕ :=
  -- Choose first repeated letter
  alphabet_size *
  -- Choose second repeated letter
  (alphabet_size - 1) *
  -- Choose two other unique letters
  (Nat.choose (alphabet_size - 2) 2) *
  -- Positions for first repeated letter
  (Nat.choose 6 2) *
  -- Positions for second repeated letter
  (Nat.choose 4 2) *
  -- Arrange two unique letters
  2 *
  -- Choose first digit
  digit_count *
  -- Choose second digit
  (digit_count - 1) *
  -- Choose third digit
  (digit_count - 2)

theorem license_plate_count :
  license_plate_combinations = 241164000 := by
  sorry

end license_plate_count_l2010_201051


namespace union_of_M_and_N_l2010_201061

def M : Set ℝ := {x : ℝ | x^2 + 2*x = 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end union_of_M_and_N_l2010_201061


namespace circle_area_after_folding_l2010_201020

theorem circle_area_after_folding (original_area : ℝ) (sector_area : ℝ) : 
  sector_area = 5 → original_area / 64 = sector_area → original_area = 320 := by
  sorry

end circle_area_after_folding_l2010_201020


namespace quadratic_sum_l2010_201058

/-- A quadratic function g(x) = ax^2 + bx + c satisfying g(1) = 2 and g(2) = 3 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: For a quadratic function g(x) = ax^2 + bx + c, if g(1) = 2 and g(2) = 3, then a + 2b + 3c = 7 -/
theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c 1 = 2) →
  (QuadraticFunction a b c 2 = 3) →
  a + 2 * b + 3 * c = 7 := by
  sorry

end quadratic_sum_l2010_201058


namespace representatives_selection_l2010_201069

/-- The number of ways to select representatives from a group of male and female students. -/
def select_representatives (num_male num_female num_total num_min_female : ℕ) : ℕ :=
  (Nat.choose num_female 2 * Nat.choose num_male 2) +
  (Nat.choose num_female 3 * Nat.choose num_male 1) +
  (Nat.choose num_female 4 * Nat.choose num_male 0)

/-- Theorem stating that selecting 4 representatives from 5 male and 4 female students,
    with at least 2 females, can be done in 81 ways. -/
theorem representatives_selection :
  select_representatives 5 4 4 2 = 81 := by
  sorry

end representatives_selection_l2010_201069


namespace remainder_theorem_l2010_201036

theorem remainder_theorem (m : ℤ) (k : ℤ) : 
  m = 40 * k - 1 → (m^2 + 3*m + 5) % 40 = 3 := by
  sorry

end remainder_theorem_l2010_201036


namespace correct_articles_for_newton_discovery_l2010_201089

/-- Represents the possible article choices for each blank --/
inductive Article
  | A : Article  -- represents "a"
  | The : Article  -- represents "the"
  | None : Article  -- represents no article

/-- Represents the context of the discovery --/
structure DiscoveryContext where
  is_specific : Bool
  is_previously_mentioned : Bool

/-- Represents the usage of "man" in the sentence --/
structure ManUsage where
  represents_mankind : Bool

/-- Determines the correct article choice given the context --/
def correct_article (context : DiscoveryContext) (man_usage : ManUsage) : Article × Article :=
  sorry

/-- Theorem stating the correct article choice for the given sentence --/
theorem correct_articles_for_newton_discovery 
  (context : DiscoveryContext)
  (man_usage : ManUsage)
  (h1 : context.is_specific = true)
  (h2 : context.is_previously_mentioned = false)
  (h3 : man_usage.represents_mankind = true) :
  correct_article context man_usage = (Article.A, Article.The) :=
sorry

end correct_articles_for_newton_discovery_l2010_201089


namespace smallest_y_in_triangle_l2010_201010

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_y_in_triangle (A B C x y : ℕ) : 
  A + B + C = 180 →
  isPrime A ∧ isPrime C →
  B ≤ A ∧ B ≤ C →
  2 * x + y = 180 →
  isPrime x →
  (∀ z : ℕ, z < y → ¬(isPrime z ∧ ∃ w : ℕ, isPrime w ∧ 2 * w + z = 180)) →
  y = 101 := by
sorry

end smallest_y_in_triangle_l2010_201010


namespace sin_75_cos_75_l2010_201012

theorem sin_75_cos_75 : Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 4 := by
  sorry

end sin_75_cos_75_l2010_201012


namespace greg_trousers_bought_l2010_201046

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a tie -/
def tie_cost : ℝ := sorry

/-- The number of trousers bought in the first scenario -/
def trousers_bought : ℕ := sorry

theorem greg_trousers_bought : 
  (3 * shirt_cost + trousers_bought * trouser_cost + 2 * tie_cost = 90) ∧
  (7 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50) ∧
  (5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 70) →
  trousers_bought = 4 := by
sorry

end greg_trousers_bought_l2010_201046


namespace fourth_root_sum_equals_expression_l2010_201035

theorem fourth_root_sum_equals_expression : 
  (1 + Real.sqrt 2 + Real.sqrt 3)^4 = 
    Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608 := by
  sorry

end fourth_root_sum_equals_expression_l2010_201035


namespace min_value_zero_l2010_201005

/-- The quadratic form as a function of x, y, and k -/
def f (x y k : ℝ) : ℝ :=
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9

/-- The theorem stating that 3/2 is the value of k that makes the minimum of f zero -/
theorem min_value_zero (k : ℝ) : 
  (∀ x y : ℝ, f x y k ≥ 0) ∧ (∃ x y : ℝ, f x y k = 0) ↔ k = 3/2 := by
  sorry

end min_value_zero_l2010_201005


namespace simple_interest_doubling_l2010_201099

/-- The factor by which a sum of money increases under simple interest -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

/-- Theorem: Given a simple interest rate of 25% per annum over 4 years, 
    the factor by which an initial sum of money increases is 2 -/
theorem simple_interest_doubling : 
  simple_interest_factor 0.25 4 = 2 := by
sorry

end simple_interest_doubling_l2010_201099


namespace mollys_age_l2010_201006

/-- Given the ratio of Sandy's age to Molly's age and Sandy's future age, 
    prove Molly's current age -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  sandy_age / molly_age = 4 / 3 →
  sandy_age + 6 = 38 →
  molly_age = 24 := by
  sorry

#check mollys_age

end mollys_age_l2010_201006


namespace solve_for_k_l2010_201094

theorem solve_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) →
  k = 8 := by
sorry

end solve_for_k_l2010_201094


namespace cone_volume_approximation_l2010_201067

theorem cone_volume_approximation (L h : ℝ) (h1 : L > 0) (h2 : h > 0) :
  (1 / 75 : ℝ) * L^2 * h = (1 / 3 : ℝ) * ((25 / 4) / 4) * L^2 * h := by
  sorry

end cone_volume_approximation_l2010_201067
