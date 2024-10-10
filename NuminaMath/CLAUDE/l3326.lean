import Mathlib

namespace sqrt_plus_reciprocal_inequality_l3326_332614

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (h : x > 0) :
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 ∧
  (Real.sqrt x + 1 / Real.sqrt x = 2 ↔ x = 1) := by
  sorry

end sqrt_plus_reciprocal_inequality_l3326_332614


namespace williams_tips_l3326_332641

/-- Williams works at a resort for 7 months. Let A be the average monthly tips for 6 of these months.
In August, he made 8 times the average of the other months. -/
theorem williams_tips (A : ℚ) : 
  let august_tips := 8 * A
  let total_tips := 15 * A
  august_tips / total_tips = 8 / 15 := by sorry

end williams_tips_l3326_332641


namespace modular_inverse_of_7_mod_31_l3326_332634

theorem modular_inverse_of_7_mod_31 :
  ∃ x : ℕ, x ≤ 30 ∧ (7 * x) % 31 = 1 :=
by
  use 9
  sorry

end modular_inverse_of_7_mod_31_l3326_332634


namespace ellipse_equation_l3326_332615

theorem ellipse_equation (x y : ℝ) :
  let a : ℝ := 4
  let b : ℝ := Real.sqrt 7
  let ε : ℝ := 0.75
  let passes_through : Prop := (-3)^2 / a^2 + 1.75^2 / b^2 = 1
  let eccentricity : Prop := ε = Real.sqrt (a^2 - b^2) / a
  passes_through ∧ eccentricity →
  x^2 / 16 + y^2 / 7 = 1 :=
by sorry

end ellipse_equation_l3326_332615


namespace perfect_square_polynomial_l3326_332644

theorem perfect_square_polynomial (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 := by
  sorry

end perfect_square_polynomial_l3326_332644


namespace chucks_team_leads_l3326_332655

/-- Represents a team's scoring in a single quarter -/
structure QuarterScore where
  fieldGoals : ℕ
  threePointers : ℕ
  freeThrows : ℕ

/-- Calculates the total points for a quarter -/
def quarterPoints (qs : QuarterScore) : ℕ :=
  2 * qs.fieldGoals + 3 * qs.threePointers + qs.freeThrows

/-- Represents a team's scoring for the entire game -/
structure GameScore where
  q1 : QuarterScore
  q2 : QuarterScore
  q3 : QuarterScore
  q4 : QuarterScore
  technicalFouls : ℕ

/-- Calculates the total points for a team in the game -/
def totalPoints (gs : GameScore) : ℕ :=
  quarterPoints gs.q1 + quarterPoints gs.q2 + quarterPoints gs.q3 + quarterPoints gs.q4 + gs.technicalFouls

theorem chucks_team_leads :
  let chucksTeam : GameScore := {
    q1 := { fieldGoals := 9, threePointers := 0, freeThrows := 5 },
    q2 := { fieldGoals := 6, threePointers := 3, freeThrows := 0 },
    q3 := { fieldGoals := 4, threePointers := 2, freeThrows := 6 },
    q4 := { fieldGoals := 8, threePointers := 1, freeThrows := 0 },
    technicalFouls := 3
  }
  let yellowTeam : GameScore := {
    q1 := { fieldGoals := 7, threePointers := 4, freeThrows := 0 },
    q2 := { fieldGoals := 5, threePointers := 2, freeThrows := 3 },
    q3 := { fieldGoals := 6, threePointers := 2, freeThrows := 0 },
    q4 := { fieldGoals := 4, threePointers := 3, freeThrows := 2 },
    technicalFouls := 2
  }
  totalPoints chucksTeam - totalPoints yellowTeam = 2 := by
  sorry

end chucks_team_leads_l3326_332655


namespace pictures_picked_out_l3326_332607

def total_pictures : ℕ := 10
def jim_bought : ℕ := 3
def probability : ℚ := 7/15

theorem pictures_picked_out :
  ∃ n : ℕ, n > 0 ∧ n < total_pictures ∧
  (Nat.choose (total_pictures - jim_bought) n : ℚ) / (Nat.choose total_pictures n : ℚ) = probability ∧
  n = 2 := by
  sorry

end pictures_picked_out_l3326_332607


namespace samson_sandwich_difference_l3326_332686

/-- The number of sandwiches Samson ate for lunch on Monday -/
def monday_lunch : ℕ := 3

/-- The number of sandwiches Samson ate for dinner on Monday -/
def monday_dinner : ℕ := 2 * monday_lunch

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

/-- The total number of sandwiches Samson ate on Monday -/
def monday_total : ℕ := monday_lunch + monday_dinner

/-- The difference between the number of sandwiches Samson ate on Monday and Tuesday -/
def sandwich_difference : ℕ := monday_total - tuesday_breakfast

theorem samson_sandwich_difference : sandwich_difference = 8 := by
  sorry

end samson_sandwich_difference_l3326_332686


namespace total_remaining_candle_life_l3326_332659

/-- Calculates the total remaining candle life in a house given the number of candles and their remaining life percentages in different rooms. -/
theorem total_remaining_candle_life
  (bedroom_candles : ℕ)
  (bedroom_life : ℚ)
  (living_room_candles : ℕ)
  (living_room_life : ℚ)
  (hallway_candles : ℕ)
  (hallway_life : ℚ)
  (study_room_life : ℚ)
  (h1 : bedroom_candles = 20)
  (h2 : living_room_candles = bedroom_candles / 2)
  (h3 : hallway_candles = 20)
  (h4 : bedroom_life = 60 / 100)
  (h5 : living_room_life = 80 / 100)
  (h6 : hallway_life = 50 / 100)
  (h7 : study_room_life = 70 / 100) :
  let study_room_candles := bedroom_candles + living_room_candles + 5
  (bedroom_candles : ℚ) * bedroom_life +
  (living_room_candles : ℚ) * living_room_life +
  (hallway_candles : ℚ) * hallway_life +
  (study_room_candles : ℚ) * study_room_life = 54.5 :=
sorry

end total_remaining_candle_life_l3326_332659


namespace solution_mixture_percentage_l3326_332660

/-- Proves that in a mixture of solutions X and Y, where X is 40% chemical A and Y is 50% chemical A,
    if the final mixture is 47% chemical A, then the percentage of solution X in the mixture is 30%. -/
theorem solution_mixture_percentage (x y : ℝ) :
  x + y = 100 →
  0.40 * x + 0.50 * y = 47 →
  x = 30 := by sorry

end solution_mixture_percentage_l3326_332660


namespace candy_sampling_percentage_l3326_332600

theorem candy_sampling_percentage (caught_percent : ℝ) (not_caught_ratio : ℝ) 
  (h1 : caught_percent = 22)
  (h2 : not_caught_ratio = 0.2) : 
  (caught_percent / (1 - not_caught_ratio)) = 27.5 := by
  sorry

end candy_sampling_percentage_l3326_332600


namespace vertex_of_quadratic_l3326_332603

/-- The quadratic function f(x) = (x-1)^2 - 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- The vertex of f(x) -/
def vertex : ℝ × ℝ := (1, -2)

theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end vertex_of_quadratic_l3326_332603


namespace barn_painted_area_l3326_332609

/-- Calculates the total area to be painted in a barn with given dimensions and conditions -/
def total_painted_area (length width height : ℝ) (window_side : ℝ) (num_windows : ℕ) : ℝ :=
  let long_wall_area := length * height
  let wide_wall_area := width * height
  let ceiling_area := length * width
  let window_area := window_side * window_side * num_windows
  let total_wall_area := 2 * (2 * long_wall_area + 2 * wide_wall_area - window_area)
  total_wall_area + ceiling_area

/-- The total area to be painted in the barn is 796 square yards -/
theorem barn_painted_area :
  total_painted_area 12 15 6 2 2 = 796 := by
  sorry

end barn_painted_area_l3326_332609


namespace base_difference_equals_59_l3326_332647

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def base_6_number : List Nat := [5, 2, 3]
def base_5_number : List Nat := [1, 3, 2]

theorem base_difference_equals_59 :
  to_base_10 base_6_number 6 - to_base_10 base_5_number 5 = 59 := by
  sorry

end base_difference_equals_59_l3326_332647


namespace prove_M_value_l3326_332648

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : Int
  diff : Int

/-- The row sequence -/
def rowSeq : ArithmeticSequence := { first := 12, diff := -7 }

/-- The first column sequence -/
def col1Seq : ArithmeticSequence := { first := -11, diff := 9 }

/-- The second column sequence -/
def col2Seq : ArithmeticSequence := { first := -35, diff := 5 }

/-- Get the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : Nat) : Int :=
  seq.first + seq.diff * (n - 1)

theorem prove_M_value : 
  nthTerm rowSeq 1 = 12 ∧ 
  nthTerm col1Seq 4 = 7 ∧ 
  nthTerm col1Seq 5 = 16 ∧
  nthTerm col2Seq 5 = -10 ∧
  col2Seq.first = -35 := by sorry

end prove_M_value_l3326_332648


namespace frozen_food_storage_temp_l3326_332601

def standard_temp : ℝ := -18
def temp_range : ℝ := 2

def is_within_range (temp : ℝ) : Prop :=
  (standard_temp - temp_range) ≤ temp ∧ temp ≤ (standard_temp + temp_range)

theorem frozen_food_storage_temp :
  ¬(is_within_range (-21)) ∧
  is_within_range (-19) ∧
  is_within_range (-18) ∧
  is_within_range (-17) := by
sorry

end frozen_food_storage_temp_l3326_332601


namespace cubic_equation_root_l3326_332624

theorem cubic_equation_root (a b : ℚ) :
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = 2 + Real.sqrt 5) →
  b = 29 := by
sorry

end cubic_equation_root_l3326_332624


namespace geometric_sum_first_7_terms_l3326_332682

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_7_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/2
  geometric_sum a r 7 = 127/192 := by sorry

end geometric_sum_first_7_terms_l3326_332682


namespace second_number_value_l3326_332694

theorem second_number_value (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7) :
  y = 240 / 7 := by
sorry

end second_number_value_l3326_332694


namespace exam_maximum_marks_l3326_332681

/-- The maximum marks for an exam -/
def maximum_marks : ℝ := sorry

/-- The passing mark as a percentage of the maximum marks -/
def passing_percentage : ℝ := 0.45

/-- The marks obtained by the student -/
def student_marks : ℝ := 150

/-- The number of marks by which the student failed -/
def failing_margin : ℝ := 30

theorem exam_maximum_marks : 
  (passing_percentage * maximum_marks = student_marks + failing_margin) → 
  maximum_marks = 400 := by
  sorry

end exam_maximum_marks_l3326_332681


namespace no_integer_solutions_l3326_332635

theorem no_integer_solutions (p₁ p₂ α n : ℕ) : 
  Prime p₁ → Prime p₂ → Odd p₁ → Odd p₂ → α > 1 → n > 1 →
  ¬ ∃ (α n : ℕ), ((p₂ - 1) / 2)^p₁ + ((p₂ + 1) / 2)^p₁ = α^n :=
by sorry

end no_integer_solutions_l3326_332635


namespace heesu_has_greatest_sum_l3326_332666

def sora_numbers : Fin 2 → ℕ
| 0 => 4
| 1 => 6

def heesu_numbers : Fin 2 → ℕ
| 0 => 7
| 1 => 5

def jiyeon_numbers : Fin 2 → ℕ
| 0 => 3
| 1 => 8

def sum_numbers (numbers : Fin 2 → ℕ) : ℕ :=
  (numbers 0) + (numbers 1)

theorem heesu_has_greatest_sum :
  sum_numbers heesu_numbers > sum_numbers sora_numbers ∧
  sum_numbers heesu_numbers > sum_numbers jiyeon_numbers :=
by sorry

end heesu_has_greatest_sum_l3326_332666


namespace infinite_solutions_exist_l3326_332688

theorem infinite_solutions_exist :
  ∃ f : ℕ → ℕ → ℕ × ℕ × ℕ,
    ∀ u v : ℕ, u > 1 → v > 1 →
      let (x, y, z) := f u v
      x^2015 + y^2015 = z^2016 ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end infinite_solutions_exist_l3326_332688


namespace cucumber_weight_after_evaporation_l3326_332632

/-- Given 100 pounds of cucumbers with initial 99% water composition by weight,
    prove that after water evaporation resulting in 95% water composition,
    the new weight is 20 pounds. -/
theorem cucumber_weight_after_evaporation
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 0.99)
  (h3 : final_water_percentage = 0.95) :
  let solid_weight := initial_weight * (1 - initial_water_percentage)
  let final_weight := solid_weight / (1 - final_water_percentage)
  final_weight = 20 :=
by sorry

end cucumber_weight_after_evaporation_l3326_332632


namespace fraction_bounds_l3326_332677

theorem fraction_bounds (n : ℕ+) : 1/2 ≤ (n : ℚ) / (n + 1) ∧ (n : ℚ) / (n + 1) < 1 := by
  sorry

end fraction_bounds_l3326_332677


namespace coin_collection_average_l3326_332629

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => arithmetic_sequence a₁ d n k + d

theorem coin_collection_average :
  let a₁ : ℝ := 5
  let d : ℝ := 6
  let n : ℕ := 7
  let seq := arithmetic_sequence a₁ d n
  (seq 0 + seq (n - 1)) / 2 = 23 := by
  sorry

end coin_collection_average_l3326_332629


namespace rays_dog_walking_problem_l3326_332618

/-- Ray's dog walking problem -/
theorem rays_dog_walking_problem (x : ℕ) : 
  (∀ (total_blocks : ℕ), total_blocks = 3 * (x + 7 + 11) → total_blocks = 66) → 
  x = 4 := by
  sorry

end rays_dog_walking_problem_l3326_332618


namespace circle_and_line_properties_l3326_332683

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - a)^2 = 36 ∧ a = 4

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  3*x - 4*y - 16 = 0 ∨ x = 0

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle M passes through A(√2, -√2) and B(10, 4)
  circle_M (Real.sqrt 2) (-Real.sqrt 2) ∧ circle_M 10 4 ∧
  -- The center of circle M lies on the line y = x
  ∃ (a : ℝ), circle_M a a ∧
  -- A line m passing through (0, -4) intersects circle M to form a chord of length 4√5
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    circle_M x₁ y₁ ∧ circle_M x₂ y₂ ∧
    line_m 0 (-4) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 80 →
  -- The standard equation of circle M is (x-4)² + (y-4)² = 36
  ∀ (x y : ℝ), circle_M x y ↔ (x - 4)^2 + (y - 4)^2 = 36 ∧
  -- The equation of line m is either 3x - 4y - 16 = 0 or x = 0
  ∀ (x y : ℝ), line_m x y ↔ (3*x - 4*y - 16 = 0 ∨ x = 0) :=
by sorry

end circle_and_line_properties_l3326_332683


namespace bill_sunday_miles_l3326_332662

/-- Represents the number of miles run by Bill and Julia over two days --/
structure RunningMiles where
  billSaturday : ℕ
  billSunday : ℕ
  juliaSunday : ℕ

/-- The conditions of the running problem --/
def runningProblem (r : RunningMiles) : Prop :=
  r.billSunday = r.billSaturday + 4 ∧
  r.juliaSunday = 2 * r.billSunday ∧
  r.billSaturday + r.billSunday + r.juliaSunday = 36

theorem bill_sunday_miles (r : RunningMiles) :
  runningProblem r → r.billSunday = 10 := by
  sorry

end bill_sunday_miles_l3326_332662


namespace intersection_condition_l3326_332625

-- Define the line l
def line (k x y : ℝ) : Prop := y + k*x + 2 = 0

-- Define the curve C in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Theorem stating the condition for intersection
theorem intersection_condition (k : ℝ) :
  (∃ x y : ℝ, line k x y ∧ curve_cartesian x y) → k ≤ -3/4 :=
sorry

end intersection_condition_l3326_332625


namespace figurine_cost_l3326_332616

def televisions : ℕ := 5
def television_cost : ℕ := 50
def figurines : ℕ := 10
def total_spent : ℕ := 260

theorem figurine_cost :
  (total_spent - televisions * television_cost) / figurines = 1 :=
sorry

end figurine_cost_l3326_332616


namespace complex_number_quadrant_l3326_332642

theorem complex_number_quadrant (z : ℂ) (m : ℝ) 
  (h1 : z * Complex.I = Complex.I + m)
  (h2 : z.im = 1) : 
  0 < z.re :=
sorry

end complex_number_quadrant_l3326_332642


namespace fold_paper_sum_l3326_332639

/-- The fold line equation --/
def fold_line (x y : ℝ) : Prop := y = 2 * x - 4

/-- The relation between (8,4) and (m,n) --/
def point_relation (m n : ℝ) : Prop := 2 * n - 8 = -m + 8

/-- The theorem stating that m + n = 32/3 --/
theorem fold_paper_sum (m n : ℝ) 
  (h1 : fold_line ((1 + 5) / 2) ((3 + 1) / 2))
  (h2 : fold_line ((8 + m) / 2) ((4 + n) / 2))
  (h3 : point_relation m n) :
  m + n = 32 / 3 := by sorry

end fold_paper_sum_l3326_332639


namespace points_needed_theorem_l3326_332665

/-- Represents the points scored in each game -/
structure GameScores where
  lastHome : ℕ
  firstAway : ℕ
  secondAway : ℕ
  thirdAway : ℕ

/-- Calculates the points needed in the next game -/
def pointsNeededNextGame (scores : GameScores) : ℕ :=
  4 * scores.lastHome - (scores.lastHome + scores.firstAway + scores.secondAway + scores.thirdAway)

/-- Theorem stating the conditions and the result to be proved -/
theorem points_needed_theorem (scores : GameScores) 
  (h1 : scores.lastHome = 2 * scores.firstAway)
  (h2 : scores.secondAway = scores.firstAway + 18)
  (h3 : scores.thirdAway = scores.secondAway + 2)
  (h4 : scores.lastHome = 62) :
  pointsNeededNextGame scores = 55 := by
  sorry

#eval pointsNeededNextGame ⟨62, 31, 49, 51⟩

end points_needed_theorem_l3326_332665


namespace smallest_number_with_conditions_l3326_332636

def has_exactly_six_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 6

def all_divisors_accommodate (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → n % d = 0

theorem smallest_number_with_conditions : 
  ∃ n : ℕ, 
    n % 18 = 0 ∧ 
    has_exactly_six_divisors n ∧
    all_divisors_accommodate n ∧
    (∀ m : ℕ, m < n → 
      ¬(m % 18 = 0 ∧ 
        has_exactly_six_divisors m ∧ 
        all_divisors_accommodate m)) ∧
    n = 72 :=
  sorry

end smallest_number_with_conditions_l3326_332636


namespace otimes_identity_l3326_332670

-- Define the new operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y^3

-- Theorem statement
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 + k^6 + 6*k^7 + k^9 := by
  sorry

end otimes_identity_l3326_332670


namespace library_sunday_visitors_l3326_332608

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_sunday_visitors
  (total_days : ℕ) 
  (non_sunday_visitors : ℕ) 
  (overall_average : ℕ) 
  (h1 : total_days = 30)
  (h2 : non_sunday_visitors = 240)
  (h3 : overall_average = 285) :
  let sundays : ℕ := total_days / 7 + 1
  let non_sundays : ℕ := total_days - sundays
  let sunday_visitors : ℕ := (overall_average * total_days - non_sunday_visitors * non_sundays) / sundays
  sunday_visitors = 510 := by
sorry

end library_sunday_visitors_l3326_332608


namespace total_cars_sold_l3326_332680

def cars_sold_day1 : ℕ := 14
def cars_sold_day2 : ℕ := 16
def cars_sold_day3 : ℕ := 27

theorem total_cars_sold : cars_sold_day1 + cars_sold_day2 + cars_sold_day3 = 57 := by
  sorry

end total_cars_sold_l3326_332680


namespace product_last_two_digits_perfect_square_even_l3326_332605

theorem product_last_two_digits_perfect_square_even (n : ℤ) : 
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (n^2 % 100 = 10 * a + b) ∧ Even (a * b) :=
sorry

end product_last_two_digits_perfect_square_even_l3326_332605


namespace negation_of_existence_proposition_l3326_332661

theorem negation_of_existence_proposition :
  (¬ ∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0) := by
  sorry

end negation_of_existence_proposition_l3326_332661


namespace intersection_of_A_and_B_l3326_332691

def A : Set ℝ := {x | x^2 - 1 = 0}
def B : Set ℝ := {-1, 2, 5}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by
  sorry

end intersection_of_A_and_B_l3326_332691


namespace extra_interest_proof_l3326_332672

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem extra_interest_proof (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) :
  principal = 15000 →
  rate1 = 0.15 →
  rate2 = 0.12 →
  time = 2 →
  simple_interest principal rate1 time - simple_interest principal rate2 time = 900 := by
  sorry

end extra_interest_proof_l3326_332672


namespace petya_wins_against_sasha_l3326_332689

/-- Represents a player in the knockout tennis tournament -/
inductive Player : Type
| Petya : Player
| Sasha : Player
| Misha : Player

/-- The number of rounds played by each player -/
def rounds_played (p : Player) : ℕ :=
  match p with
  | Player.Petya => 12
  | Player.Sasha => 7
  | Player.Misha => 11

/-- The total number of games played in the tournament -/
def total_games : ℕ := (rounds_played Player.Petya + rounds_played Player.Sasha + rounds_played Player.Misha) / 2

/-- The number of games a player did not play -/
def games_not_played (p : Player) : ℕ := total_games - rounds_played p

/-- Theorem stating that Petya won 4 times against Sasha -/
theorem petya_wins_against_sasha : 
  games_not_played Player.Misha = 4 ∧ 
  (∀ p : Player, games_not_played p + rounds_played p = total_games) ∧
  (rounds_played Player.Sasha = 7 → games_not_played Player.Sasha = 8) :=
sorry

end petya_wins_against_sasha_l3326_332689


namespace population_average_age_l3326_332650

/-- Given a population with females and males, calculate the average age -/
theorem population_average_age
  (female_ratio male_ratio : ℕ)
  (female_avg_age male_avg_age : ℝ)
  (h_ratio : female_ratio = 11 ∧ male_ratio = 10)
  (h_female_age : female_avg_age = 34)
  (h_male_age : male_avg_age = 32) :
  let total_people := female_ratio + male_ratio
  let total_age_sum := female_ratio * female_avg_age + male_ratio * male_avg_age
  total_age_sum / total_people = 33 + 1 / 21 :=
by sorry

end population_average_age_l3326_332650


namespace discount_percentage_calculation_l3326_332687

theorem discount_percentage_calculation (marked_price : ℝ) (h1 : marked_price > 0) : 
  let cost_price := 0.64 * marked_price
  let gain := 0.375 * cost_price
  let selling_price := cost_price + gain
  let discount := marked_price - selling_price
  (discount / marked_price) * 100 = 12 := by
sorry

end discount_percentage_calculation_l3326_332687


namespace four_dogs_food_consumption_l3326_332602

/-- The total daily food consumption of four dogs -/
def total_dog_food_consumption (dog1 dog2 dog3 dog4 : ℚ) : ℚ :=
  dog1 + dog2 + dog3 + dog4

/-- Theorem stating the total daily food consumption of four specific dogs -/
theorem four_dogs_food_consumption :
  total_dog_food_consumption (1/8) (1/4) (3/8) (1/2) = 5/4 := by
  sorry

end four_dogs_food_consumption_l3326_332602


namespace triangle_area_is_25_over_3_l3326_332679

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The area of a triangle given three lines that form its sides -/
def triangleArea (l1 l2 l3 : Line) : ℝ :=
  sorry

/-- The three lines that form the triangle -/
def line1 : Line := { slope := 2, intercept := 4 }
def line2 : Line := { slope := -1, intercept := 3 }
def line3 : Line := { slope := 0, intercept := 0 }

theorem triangle_area_is_25_over_3 :
  triangleArea line1 line2 line3 = 25 / 3 := by
  sorry

end triangle_area_is_25_over_3_l3326_332679


namespace modulus_of_complex_fraction_l3326_332604

theorem modulus_of_complex_fraction : 
  Complex.abs ((2 - Complex.I) / (1 + Complex.I)) = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_complex_fraction_l3326_332604


namespace some_number_value_l3326_332606

theorem some_number_value (x y n : ℝ) 
  (h1 : x / (2 * y) = 3 / n) 
  (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : 
  n = 2 := by
  sorry

end some_number_value_l3326_332606


namespace intersection_condition_minimum_condition_l3326_332643

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^3 - x^2 else a * x * Real.exp x

-- Theorem for the range of m
theorem intersection_condition (a : ℝ) (h : a > 0) :
  ∀ m : ℝ, (∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ f a x = m) ↔ (0 ≤ m ∧ m ≤ 4) ∨ m = -4/27 :=
sorry

-- Theorem for the range of a
theorem minimum_condition :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ -a) ↔ a ≥ 4/27 :=
sorry

end intersection_condition_minimum_condition_l3326_332643


namespace problem_solution_l3326_332695

-- Define a function to check if a number is square-free
def is_square_free (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p * p ∣ n) → p = 1

-- Define the condition for the problem
def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧ p ≥ 3 ∧
  ∀ (q : ℕ), Nat.Prime q → q < p →
    is_square_free (p - p / q * q)

-- State the theorem
theorem problem_solution :
  {p : ℕ | satisfies_condition p} = {3, 5, 7, 13} :=
sorry

end problem_solution_l3326_332695


namespace bisection_uses_all_structures_l3326_332640

/-- Represents the different algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents the bisection method for a specific equation -/
structure BisectionMethod where
  equation : ℝ → ℝ
  approximateRoot : ℝ → ℝ → ℝ → ℝ

/-- The bisection method for x^2 - 10 = 0 -/
def bisectionForXSquaredMinus10 : BisectionMethod :=
  { equation := λ x => x^2 - 10,
    approximateRoot := sorry }

/-- Checks if a given algorithm structure is used in the bisection method -/
def usesStructure (b : BisectionMethod) (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Sequential => sorry
  | AlgorithmStructure.Conditional => sorry
  | AlgorithmStructure.Loop => sorry

theorem bisection_uses_all_structures :
  ∀ s : AlgorithmStructure, usesStructure bisectionForXSquaredMinus10 s := by
  sorry

end bisection_uses_all_structures_l3326_332640


namespace linear_equation_solution_l3326_332669

theorem linear_equation_solution (m : ℝ) : 
  (1 : ℝ) * m - 3 = 3 → m = 6 := by sorry

end linear_equation_solution_l3326_332669


namespace total_rectangles_in_diagram_l3326_332621

/-- Represents a rectangle in the diagram -/
structure Rectangle where
  id : Nat

/-- Represents the diagram with rectangles -/
structure Diagram where
  rectangles : List Rectangle

/-- Counts the number of unique rectangles in the diagram -/
def count_unique_rectangles (d : Diagram) : Nat :=
  d.rectangles.length

/-- Theorem stating the total number of unique rectangles in the specific diagram -/
theorem total_rectangles_in_diagram :
  ∃ (d : Diagram),
    (∃ (r1 r2 r3 : Rectangle), r1 ∈ d.rectangles ∧ r2 ∈ d.rectangles ∧ r3 ∈ d.rectangles) ∧  -- 3 large rectangles
    (∃ (r4 r5 r6 r7 : Rectangle), r4 ∈ d.rectangles ∧ r5 ∈ d.rectangles ∧ r6 ∈ d.rectangles ∧ r7 ∈ d.rectangles) ∧  -- 4 small rectangles
    (∀ (r s : Rectangle), r ∈ d.rectangles → s ∈ d.rectangles → ∃ (t : Rectangle), t ∈ d.rectangles) →  -- Combination of rectangles
    count_unique_rectangles d = 11 :=
by
  sorry


end total_rectangles_in_diagram_l3326_332621


namespace tan_inequality_solution_set_l3326_332690

open Real

theorem tan_inequality_solution_set (x : ℝ) :
  (3 * tan x + Real.sqrt 3 > 0) ↔
  ∃ k : ℤ, x ∈ Set.Ioo ((-(π / 6) : ℝ) + k * π) ((π / 6 : ℝ) + k * π) :=
by sorry

end tan_inequality_solution_set_l3326_332690


namespace lines_parallel_iff_l3326_332628

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2

/-- The first line: ax + 2y + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 1 = 0

/-- The second line: x + y + 4 = 0 -/
def line2 (x y : ℝ) : Prop :=
  x + y + 4 = 0

theorem lines_parallel_iff (a : ℝ) :
  parallel a 2 1 1 1 4 ↔ a = 2 :=
sorry

end lines_parallel_iff_l3326_332628


namespace consecutive_even_count_l3326_332673

def is_consecutive_even (a b : ℕ) : Prop := b = a + 2

def sum_consecutive_even (start : ℕ) (count : ℕ) : ℕ :=
  (count * (2 * start + count - 1))

theorem consecutive_even_count :
  ∃ (count : ℕ), 
    sum_consecutive_even 80 count = 246 ∧
    count = 3 :=
by sorry

end consecutive_even_count_l3326_332673


namespace profit_ratio_theorem_l3326_332611

/-- Represents the investment and profit information for a partner -/
structure Partner where
  investment : ℕ
  months : ℕ

/-- Calculates the profit factor for a partner -/
def profitFactor (p : Partner) : ℕ := p.investment * p.months

/-- Theorem stating the profit ratio of two partners given their investments and time periods -/
theorem profit_ratio_theorem (p q : Partner) 
  (h_investment_ratio : p.investment * 5 = q.investment * 7)
  (h_p_months : p.months = 5)
  (h_q_months : q.months = 9) :
  profitFactor p * 9 = profitFactor q * 7 := by
  sorry

#check profit_ratio_theorem

end profit_ratio_theorem_l3326_332611


namespace hannah_ran_9km_on_monday_l3326_332698

/-- The distance Hannah ran on Wednesday in meters -/
def wednesday_distance : ℕ := 4816

/-- The distance Hannah ran on Friday in meters -/
def friday_distance : ℕ := 2095

/-- The additional distance Hannah ran on Monday compared to Wednesday and Friday combined, in meters -/
def monday_additional_distance : ℕ := 2089

/-- The number of meters in a kilometer -/
def meters_per_kilometer : ℕ := 1000

/-- Theorem stating that Hannah ran 9 kilometers on Monday -/
theorem hannah_ran_9km_on_monday : 
  (wednesday_distance + friday_distance + monday_additional_distance) / meters_per_kilometer = 9 := by
  sorry

end hannah_ran_9km_on_monday_l3326_332698


namespace original_equals_scientific_l3326_332667

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The original number to be represented -/
def original_number : ℝ := 43050000

/-- The scientific notation representation -/
def scientific_repr : ScientificNotation :=
  { coefficient := 4.305,
    exponent := 7,
    h1 := by sorry }

/-- Theorem stating that the original number equals its scientific notation representation -/
theorem original_equals_scientific :
  original_number = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent :=
by sorry

end original_equals_scientific_l3326_332667


namespace vector_dot_product_problem_l3326_332613

theorem vector_dot_product_problem (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-1, 2)) :
  (a + 2 • b) • b = 14 := by
  sorry

end vector_dot_product_problem_l3326_332613


namespace parabola_area_l3326_332649

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the roots of the parabola
def root1 : ℝ := 1
def root2 : ℝ := 3

-- State the theorem
theorem parabola_area : 
  (∫ (x : ℝ) in root1..root2, -f x) = 4/3 := by sorry

end parabola_area_l3326_332649


namespace arithmetic_sequence_a9_l3326_332620

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = 16)
  (h_a3 : a 3 = 1) :
  a 9 = 15 :=
sorry

end arithmetic_sequence_a9_l3326_332620


namespace minimum_value_at_one_l3326_332627

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (a + 1) * x^2 - (a^2 + 3*a - 3) * x

theorem minimum_value_at_one (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 1) → a = 2 :=
by sorry

end minimum_value_at_one_l3326_332627


namespace factor_in_range_l3326_332638

theorem factor_in_range : ∃ (n : ℕ), 
  1210000 < n ∧ 
  n < 1220000 ∧ 
  1464101210001 % n = 0 :=
by
  -- The proof would go here
  sorry

end factor_in_range_l3326_332638


namespace sum_of_real_roots_of_quartic_l3326_332610

theorem sum_of_real_roots_of_quartic (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^4 - 6*x^2 - 2*x - 1
  ∃ (r₁ r₂ : ℝ), (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = -Real.sqrt 2 :=
by sorry

end sum_of_real_roots_of_quartic_l3326_332610


namespace total_fish_count_l3326_332626

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * white_ducks + 
                      fish_per_black_duck * black_ducks + 
                      fish_per_multicolor_duck * multicolor_ducks

theorem total_fish_count : total_fish = 157 := by
  sorry

end total_fish_count_l3326_332626


namespace altitude_to_base_l3326_332646

/-- Given a triangle ABC with known sides and area, prove the altitude to base AB -/
theorem altitude_to_base (a b c area h : ℝ) : 
  a = 30 → b = 17 → c = 25 → area = 120 → 
  area = (1/2) * a * h → h = 8 := by sorry

end altitude_to_base_l3326_332646


namespace point_on_y_axis_l3326_332652

/-- A point P with coordinates (m+2, 2m-4) that lies on the y-axis has coordinates (0, -8). -/
theorem point_on_y_axis (m : ℝ) :
  (m + 2 = 0) → (m + 2, 2 * m - 4) = (0, -8) := by
  sorry

end point_on_y_axis_l3326_332652


namespace max_value_with_constraint_l3326_332617

theorem max_value_with_constraint (x y z : ℝ) (h : 4 * x^2 + y^2 + 16 * z^2 = 1) :
  7 * x + 2 * y + 8 * z ≤ 9 / 2 :=
by sorry

end max_value_with_constraint_l3326_332617


namespace mock_exam_is_systematic_sampling_l3326_332678

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Cluster

/-- Represents an examination room --/
structure ExamRoom where
  seats : Fin 30 → Nat
  selected_seat : Nat

/-- Represents the mock exam setup --/
structure MockExam where
  rooms : Fin 80 → ExamRoom
  selection_method : SamplingMethod

/-- The mock exam setup as described in the problem --/
def mock_exam : MockExam :=
  { rooms := λ _ => { seats := λ _ => Nat.succ (Nat.zero), selected_seat := 15 },
    selection_method := SamplingMethod.Systematic }

/-- Theorem stating that the sampling method used in the mock exam is systematic sampling --/
theorem mock_exam_is_systematic_sampling :
  mock_exam.selection_method = SamplingMethod.Systematic :=
by sorry

end mock_exam_is_systematic_sampling_l3326_332678


namespace first_team_speed_calculation_l3326_332653

/-- The speed of the first team in miles per hour -/
def first_team_speed : ℝ := 20

/-- The speed of the second team in miles per hour -/
def second_team_speed : ℝ := 30

/-- The radio range in miles -/
def radio_range : ℝ := 125

/-- The time until radio contact is lost in hours -/
def time_until_lost_contact : ℝ := 2.5

theorem first_team_speed_calculation :
  first_team_speed = (radio_range / time_until_lost_contact) - second_team_speed := by
  sorry

#check first_team_speed_calculation

end first_team_speed_calculation_l3326_332653


namespace statement_equivalence_l3326_332693

/-- Represents the property of being happy -/
def happy : Prop := sorry

/-- Represents the property of possessing the food item -/
def possess : Prop := sorry

/-- The statement "Happy people all possess it" -/
def original_statement : Prop := happy → possess

/-- The statement "People who do not possess it are unhappy" -/
def equivalent_statement : Prop := ¬possess → ¬happy

/-- Theorem stating that the original statement is logically equivalent to the equivalent statement -/
theorem statement_equivalence : original_statement ↔ equivalent_statement :=
  sorry

end statement_equivalence_l3326_332693


namespace rebecca_egg_groups_l3326_332676

/-- Given a total number of eggs and the number of eggs per group, 
    calculate the number of groups that can be created. -/
def calculate_groups (total_eggs : ℕ) (eggs_per_group : ℕ) : ℕ :=
  total_eggs / eggs_per_group

/-- Theorem stating that with 15 eggs and 5 eggs per group, 
    the number of groups is 3. -/
theorem rebecca_egg_groups : 
  calculate_groups 15 5 = 3 := by
  sorry

end rebecca_egg_groups_l3326_332676


namespace apple_distribution_l3326_332623

theorem apple_distribution (t x : ℕ) (h1 : t = 4) (h2 : (9 * t * x) / 10 - 6 = 48) : x = 15 := by
  sorry

end apple_distribution_l3326_332623


namespace competition_scores_l3326_332663

def student_scores : List ℝ := [80, 84, 86, 90]

theorem competition_scores (fifth_score : ℝ) 
  (h1 : (fifth_score :: student_scores).length = 5)
  (h2 : (fifth_score :: student_scores).sum / 5 = 87) :
  fifth_score = 95 ∧ 
  let all_scores := fifth_score :: student_scores
  (all_scores.map (λ x => (x - 87)^2)).sum / 5 = 26.4 := by
  sorry

end competition_scores_l3326_332663


namespace mark_total_eggs_l3326_332656

/-- The number of people sharing the eggs -/
def num_people : ℕ := 4

/-- The number of eggs each person gets when distributed equally -/
def eggs_per_person : ℕ := 6

/-- The total number of eggs Mark has -/
def total_eggs : ℕ := num_people * eggs_per_person

theorem mark_total_eggs : total_eggs = 24 := by
  sorry

end mark_total_eggs_l3326_332656


namespace evaluate_power_l3326_332619

-- Define the problem
theorem evaluate_power : (81 : ℝ) ^ (11/4) = 177147 := by
  sorry

end evaluate_power_l3326_332619


namespace ellipse_hyperbola_equations_l3326_332612

/-- Definition of an ellipse with given properties -/
def Ellipse (e : ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ), 
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (p.1 - f₁.1)^2 + p.2^2 + (p.1 - f₂.1)^2 + p.2^2 = d^2 ∧
    (f₁.1 - f₂.1)^2 = (e * d)^2}

/-- Definition of a hyperbola with given properties -/
def Hyperbola (c : ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ),
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (f₁.1 - f₂.1)^2 = 4 * c^2 ∧
    (p.2 = k * p.1 → p.1^2 * (1 + k^2) = c^2 * (1 + k^2)^2)}

/-- Main theorem statement -/
theorem ellipse_hyperbola_equations :
  ∀ (x y : ℝ),
    (x, y) ∈ Ellipse (1/2) 8 ↔ x^2/16 + y^2/12 = 1 ∧
    (x, y) ∈ Hyperbola 2 (Real.sqrt 3) ↔ x^2 - y^2/3 = 1 :=
by sorry

end ellipse_hyperbola_equations_l3326_332612


namespace mikes_work_days_l3326_332651

theorem mikes_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 →
  total_hours = 15 →
  days * hours_per_day = total_hours →
  days = 5 := by
sorry

end mikes_work_days_l3326_332651


namespace set_B_proof_l3326_332633

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem set_B_proof (A B : Finset Nat) 
  (h1 : A ∩ (U \ B) = {1,3})
  (h2 : U \ (A ∪ B) = {2,4}) :
  B = {5,6,7,8} := by
sorry

end set_B_proof_l3326_332633


namespace first_player_wins_l3326_332622

/-- Represents the game state -/
structure GameState where
  stones : ℕ
  last_move : ℕ

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : ℕ) : Prop :=
  move > 0 ∧ move ≤ state.stones ∧
  (state.last_move = 0 ∨ state.last_move % move = 0)

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.stones = 0

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (first_move : ℕ),
    valid_move { stones := 1992, last_move := 0 } first_move ∧
    ∀ (second_move : ℕ),
      valid_move { stones := 1992 - first_move, last_move := first_move } second_move →
      ∃ (strategy : GameState → ℕ),
        (∀ (state : GameState),
          valid_move state (strategy state)) ∧
        (∀ (state : GameState),
          ¬is_winning_state state →
          is_winning_state { stones := state.stones - strategy state, last_move := strategy state }) :=
sorry

end first_player_wins_l3326_332622


namespace cristine_lemons_l3326_332685

theorem cristine_lemons (initial_lemons : ℕ) (given_away_fraction : ℚ) (exchanged_fraction : ℚ) : 
  initial_lemons = 12 →
  given_away_fraction = 1/4 →
  exchanged_fraction = 1/3 →
  (initial_lemons - initial_lemons * given_away_fraction) * (1 - exchanged_fraction) = 6 := by
sorry

end cristine_lemons_l3326_332685


namespace cos_difference_value_l3326_332668

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.cos A + Real.cos B = 1/2) 
  (h2 : Real.sin A + Real.sin B = 3/2) : 
  Real.cos (A - B) = 1/4 := by
sorry

end cos_difference_value_l3326_332668


namespace solution_pairs_l3326_332645

theorem solution_pairs (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * y - Real.sqrt (y / x) - 6 * Real.sqrt (x * y) + 2 = 0 ∧
   x^2 + 81 * x^2 * y^4 = 2 * y^2) ↔
  ((x = Real.sqrt (Real.sqrt 31) / 12 ∧ y = Real.sqrt (Real.sqrt 31) / 3) ∨
   (x = 1/3 ∧ y = 1/3)) :=
by sorry

end solution_pairs_l3326_332645


namespace problem_1_problem_2_l3326_332696

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = 13 - 8*x + Real.sqrt 2 * x^2) →
  (deriv f x₀ = 4) →
  x₀ = 3 * Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) :
  (∀ x, f x = x^2 + 2*x*(deriv f 0)) →
  ¬∃ y, deriv f 0 = y := by sorry

end problem_1_problem_2_l3326_332696


namespace student_not_asked_probability_l3326_332637

/-- The probability of a student not being asked in either of two consecutive lessons -/
theorem student_not_asked_probability
  (total_students : ℕ)
  (selected_students : ℕ)
  (previous_lesson_pool : ℕ)
  (h1 : total_students = 30)
  (h2 : selected_students = 3)
  (h3 : previous_lesson_pool = 10)
  : ℚ :=
  11 / 30

/-- The proof of the theorem -/
lemma student_not_asked_probability_proof :
  student_not_asked_probability 30 3 10 rfl rfl rfl = 11 / 30 := by
  sorry

end student_not_asked_probability_l3326_332637


namespace missing_number_proof_l3326_332684

theorem missing_number_proof : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ∧ x = 11 := by
  sorry

end missing_number_proof_l3326_332684


namespace R_is_converse_negation_of_P_l3326_332699

-- Define the proposition P
def P : Prop := ∀ x y : ℝ, x + y = 0 → (x = -y ∧ y = -x)

-- Define the negation of P (Q)
def Q : Prop := ¬P

-- Define the inverse of Q (R)
def R : Prop := ∀ x y : ℝ, ¬(x = -y ∧ y = -x) → x + y ≠ 0

-- Theorem stating that R is the converse negation of P
theorem R_is_converse_negation_of_P : R = (∀ x y : ℝ, ¬(x = -y ∧ y = -x) → x + y ≠ 0) := by
  sorry

end R_is_converse_negation_of_P_l3326_332699


namespace triangle_area_at_most_half_parallelogram_l3326_332674

/-- A parallelogram in a 2D plane -/
structure Parallelogram :=
  (P Q R S : ℝ × ℝ)

/-- A triangle in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Calculate the area of a parallelogram -/
def area_parallelogram (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of a triangle -/
def area_triangle (t : Triangle) : ℝ :=
  sorry

/-- Check if a triangle is inside a parallelogram -/
def is_inside (t : Triangle) (p : Parallelogram) : Prop :=
  sorry

/-- Theorem: The area of a triangle inside a parallelogram is at most half the area of the parallelogram -/
theorem triangle_area_at_most_half_parallelogram (p : Parallelogram) (t : Triangle) 
  (h : is_inside t p) : area_triangle t ≤ (1/2) * area_parallelogram p :=
  sorry

end triangle_area_at_most_half_parallelogram_l3326_332674


namespace simplify_product_of_square_roots_l3326_332657

theorem simplify_product_of_square_roots (y : ℝ) (hy : y > 0) :
  Real.sqrt (50 * y^3) * Real.sqrt (18 * y) * Real.sqrt (98 * y^5) = 210 * y^4 * Real.sqrt (2 * y) :=
by sorry

end simplify_product_of_square_roots_l3326_332657


namespace arcsin_sqrt3_div2_l3326_332671

theorem arcsin_sqrt3_div2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by sorry

end arcsin_sqrt3_div2_l3326_332671


namespace f_positive_solution_set_m_upper_bound_l3326_332654

def f (x : ℝ) := |x - 2| - |2*x + 1|

theorem f_positive_solution_set :
  {x : ℝ | f x > 0} = Set.Ioo (-3) (1/3) :=
sorry

theorem m_upper_bound (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ > 2*m + 1) → m < 3/4 :=
sorry

end f_positive_solution_set_m_upper_bound_l3326_332654


namespace sum_of_rectangle_areas_l3326_332630

/-- Given six rectangles with width 2 and lengths 1, 4, 9, 16, 25, and 36, 
    prove that the sum of their areas is 182. -/
theorem sum_of_rectangle_areas : 
  let width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36]
  let areas : List ℕ := lengths.map (λ l => l * width)
  areas.sum = 182 := by
sorry

end sum_of_rectangle_areas_l3326_332630


namespace largest_odd_between_1_and_7_l3326_332697

theorem largest_odd_between_1_and_7 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 7 ∧ Odd n → n ≤ 7 :=
by sorry

end largest_odd_between_1_and_7_l3326_332697


namespace cup_stacking_l3326_332692

theorem cup_stacking (a₁ a₂ a₄ a₅ : ℕ) (h1 : a₁ = 17) (h2 : a₂ = 21) (h4 : a₄ = 29) (h5 : a₅ = 33)
  (h_pattern : ∃ d : ℕ, a₂ = a₁ + d ∧ a₄ = a₂ + 2*d ∧ a₅ = a₄ + d) :
  ∃ a₃ : ℕ, a₃ = 25 ∧ a₃ = a₂ + (a₂ - a₁) := by
  sorry

end cup_stacking_l3326_332692


namespace house_transaction_net_change_l3326_332631

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  ownsHouse : Bool

/-- Represents a house transaction -/
structure Transaction where
  seller : String
  buyer : String
  price : Int

/-- Calculate the net change in wealth after transactions -/
def netChangeInWealth (initial : FinancialState) (final : FinancialState) (initialHouseValue : Int) : Int :=
  final.cash - initial.cash + (if final.ownsHouse then initialHouseValue else 0) - (if initial.ownsHouse then initialHouseValue else 0)

theorem house_transaction_net_change :
  let initialHouseValue := 15000
  let initialA := FinancialState.mk 15000 true
  let initialB := FinancialState.mk 20000 false
  let transaction1 := Transaction.mk "A" "B" 18000
  let transaction2 := Transaction.mk "B" "A" 12000
  let finalA := FinancialState.mk 21000 true
  let finalB := FinancialState.mk 14000 false
  (netChangeInWealth initialA finalA initialHouseValue = 6000) ∧
  (netChangeInWealth initialB finalB initialHouseValue = -6000) := by
  sorry

end house_transaction_net_change_l3326_332631


namespace expression_simplification_l3326_332664

theorem expression_simplification (x : ℝ) (h : x = Real.pi ^ 0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end expression_simplification_l3326_332664


namespace solve_for_y_l3326_332658

theorem solve_for_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) (h4 : x / y = 81) : 
  y = 2 / 9 := by
sorry

end solve_for_y_l3326_332658


namespace parallel_vectors_y_value_l3326_332675

def vector_a : Fin 2 → ℝ := ![4, 2]
def vector_b (y : ℝ) : Fin 2 → ℝ := ![6, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem parallel_vectors_y_value :
  parallel vector_a (vector_b y) → y = 3 := by
  sorry

end parallel_vectors_y_value_l3326_332675
