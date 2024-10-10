import Mathlib

namespace pencils_per_row_indeterminate_l3424_342493

theorem pencils_per_row_indeterminate (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 7 →
  crayons_per_row = 30 →
  total_crayons = 210 →
  ∀ (pencils_per_row : ℕ), ∃ (total_pencils : ℕ),
    total_pencils = rows * pencils_per_row :=
by sorry

end pencils_per_row_indeterminate_l3424_342493


namespace principal_calculation_l3424_342460

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal is 44625 -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 1 / 100
  let time : ℕ := 9
  calculate_principal simple_interest rate time = 44625 := by
  sorry

end principal_calculation_l3424_342460


namespace nails_needed_l3424_342403

theorem nails_needed (nails_per_plank : ℕ) (num_planks : ℕ) : 
  nails_per_plank = 2 → num_planks = 2 → nails_per_plank * num_planks = 4 := by
  sorry

#check nails_needed

end nails_needed_l3424_342403


namespace exists_non_isosceles_with_isosceles_bisector_base_l3424_342486

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define an angle bisector
def AngleBisector (T : Triangle) (vertex : ℕ) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the base of an angle bisector
def BaseBisector (T : Triangle) (vertex : ℕ) : ℝ × ℝ := sorry

-- Define isosceles property for a triangle
def IsIsosceles (T : Triangle) : Prop := sorry

-- Define the triangle formed by the bases of angle bisectors
def BisectorBaseTriangle (T : Triangle) : Triangle :=
  { A := BaseBisector T 0,
    B := BaseBisector T 1,
    C := BaseBisector T 2 }

theorem exists_non_isosceles_with_isosceles_bisector_base :
  ∃ T : Triangle,
    IsIsosceles (BisectorBaseTriangle T) ∧
    ¬IsIsosceles T :=
  sorry

end exists_non_isosceles_with_isosceles_bisector_base_l3424_342486


namespace smallest_number_of_eggs_l3424_342462

theorem smallest_number_of_eggs :
  ∀ (total_eggs : ℕ) (num_containers : ℕ),
    total_eggs > 150 →
    total_eggs = 15 * num_containers - 3 →
    (∀ smaller_total : ℕ, smaller_total > 150 → smaller_total = 15 * (smaller_total / 15) - 3 → smaller_total ≥ total_eggs) →
    total_eggs = 162 :=
by
  sorry

end smallest_number_of_eggs_l3424_342462


namespace quadrilateral_diagonal_l3424_342438

theorem quadrilateral_diagonal (sides : Finset ℝ) 
  (h_sides : sides = {1, 2, 2.8, 5, 7.5}) : 
  ∃ (diagonal : ℝ), diagonal ∈ sides ∧
  (∀ (a b c : ℝ), a ∈ sides → b ∈ sides → c ∈ sides → 
   a ≠ diagonal → b ≠ diagonal → c ≠ diagonal → 
   a + b > diagonal ∧ b + c > diagonal ∧ a + c > diagonal) ∧
  diagonal = 2.8 :=
sorry

end quadrilateral_diagonal_l3424_342438


namespace mixture_percentage_l3424_342479

theorem mixture_percentage (solution1 solution2 : ℝ) 
  (percent1 percent2 : ℝ) (h1 : solution1 = 6) 
  (h2 : solution2 = 4) (h3 : percent1 = 0.2) 
  (h4 : percent2 = 0.6) : 
  (percent1 * solution1 + percent2 * solution2) / (solution1 + solution2) = 0.36 := by
  sorry

end mixture_percentage_l3424_342479


namespace fractional_equation_elimination_l3424_342497

theorem fractional_equation_elimination (x : ℝ) : 
  (1 - (5*x + 2) / (x * (x + 1)) = 3 / (x + 1)) → 
  (x^2 - 7*x - 2 = 0) :=
by
  sorry

end fractional_equation_elimination_l3424_342497


namespace oliver_final_amount_l3424_342414

def oliver_money (initial : ℕ) (spent : ℕ) (received : ℕ) : ℕ :=
  initial - spent + received

theorem oliver_final_amount :
  oliver_money 33 4 32 = 61 := by sorry

end oliver_final_amount_l3424_342414


namespace cab_driver_income_l3424_342406

/-- The cab driver's income problem -/
theorem cab_driver_income (day1 day2 day3 day5 : ℕ) (average : ℕ) 
  (h1 : day1 = 400)
  (h2 : day2 = 250)
  (h3 : day3 = 650)
  (h5 : day5 = 500)
  (h_avg : average = 440)
  (h_total : day1 + day2 + day3 + day5 + (5 * average - (day1 + day2 + day3 + day5)) = 5 * average) :
  5 * average - (day1 + day2 + day3 + day5) = 400 := by
  sorry

end cab_driver_income_l3424_342406


namespace athlete_A_one_win_one_loss_l3424_342483

/-- The probability of athlete A winning against athlete B -/
def prob_A_wins_B : ℝ := 0.8

/-- The probability of athlete A winning against athlete C -/
def prob_A_wins_C : ℝ := 0.7

/-- The probability of athlete A achieving one win and one loss -/
def prob_one_win_one_loss : ℝ := prob_A_wins_B * (1 - prob_A_wins_C) + (1 - prob_A_wins_B) * prob_A_wins_C

theorem athlete_A_one_win_one_loss : prob_one_win_one_loss = 0.38 := by
  sorry

end athlete_A_one_win_one_loss_l3424_342483


namespace prob_B_wins_value_l3424_342424

/-- The probability of player B winning in a chess game -/
def prob_B_wins (prob_A_wins : ℝ) (prob_draw : ℝ) : ℝ :=
  1 - prob_A_wins - prob_draw

/-- Theorem: The probability of player B winning is 0.3 -/
theorem prob_B_wins_value :
  prob_B_wins 0.3 0.4 = 0.3 := by
sorry

end prob_B_wins_value_l3424_342424


namespace vision_data_median_l3424_342419

structure VisionData where
  values : List Float
  frequencies : List Nat
  total_students : Nat

def median (data : VisionData) : Float :=
  sorry

theorem vision_data_median :
  let data : VisionData := {
    values := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    frequencies := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39
  }
  median data = 4.6 := by sorry

end vision_data_median_l3424_342419


namespace sqrt_meaningful_range_l3424_342490

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) → x ≥ 2 := by
  sorry

end sqrt_meaningful_range_l3424_342490


namespace quadratic_coefficient_sum_l3424_342404

/-- A quadratic function passing through (2, 5) with vertex at (1, 3) has a - b + c = 11 -/
theorem quadratic_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 1)^2 + 3) →  -- vertex form
  a * 2^2 + b * 2 + c = 5 →                         -- passes through (2, 5)
  a - b + c = 11 := by
sorry

end quadratic_coefficient_sum_l3424_342404


namespace five_integer_chords_l3424_342405

/-- A circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Count of integer length chords passing through P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The specific circle and point from the problem -/
def problem_circle : CircleWithPoint :=
  { radius := 17,
    distance_to_p := 8 }

/-- The theorem stating that there are 5 integer length chords -/
theorem five_integer_chords :
  count_integer_chords problem_circle = 5 := by
  sorry

end five_integer_chords_l3424_342405


namespace smallest_AAAB_l3424_342466

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def AB (a b : ℕ) : ℕ := 10 * a + b

def AAAB (a b : ℕ) : ℕ := 1000 * a + 100 * a + 10 * a + b

theorem smallest_AAAB :
  ∀ a b : ℕ,
    a ≠ b →
    a < 10 →
    b < 10 →
    is_two_digit (AB a b) →
    is_four_digit (AAAB a b) →
    7 * (AB a b) = AAAB a b →
    ∀ a' b' : ℕ,
      a' ≠ b' →
      a' < 10 →
      b' < 10 →
      is_two_digit (AB a' b') →
      is_four_digit (AAAB a' b') →
      7 * (AB a' b') = AAAB a' b' →
      AAAB a b ≤ AAAB a' b' →
    AAAB a b = 6661 :=
by sorry

end smallest_AAAB_l3424_342466


namespace line_properties_l3424_342491

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Define points A and P
def point_A : ℝ × ℝ := (3, 2)
def point_P : ℝ × ℝ := (3, 0)

-- Define the perpendicular line l₁
def line_l1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the parallel lines l₂
def line_l2_1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line_l2_2 (x y : ℝ) : Prop := 2 * x + y - 11 = 0

-- Theorem statement
theorem line_properties :
  (∀ x y : ℝ, line_l1 x y ↔ (x = point_A.1 ∧ y = point_A.2) ∨ 
    (∃ k : ℝ, x = point_A.1 + k ∧ y = point_A.2 - k/2)) ∧
  (∀ x y : ℝ, (line_l2_1 x y ∨ line_l2_2 x y) ↔
    (∃ k : ℝ, x = k ∧ y = -2*k + 1) ∧
    (|2 * point_P.1 + point_P.2 + 1| / Real.sqrt 5 = Real.sqrt 5 ∨
     |2 * point_P.1 + point_P.2 + 11| / Real.sqrt 5 = Real.sqrt 5)) :=
by sorry


end line_properties_l3424_342491


namespace sin_780_degrees_l3424_342454

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_780_degrees_l3424_342454


namespace twelve_sided_die_expected_value_l3424_342458

/-- The number of sides on the die -/
def n : ℕ := 12

/-- The expected value of rolling an n-sided die with faces numbered from 1 to n -/
def expected_value (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / 2

/-- Theorem: The expected value of rolling a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem twelve_sided_die_expected_value :
  expected_value n = 13/2 := by sorry

end twelve_sided_die_expected_value_l3424_342458


namespace problem_solution_l3424_342425

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x, f m (x - 3) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 2) →
  m = 2 ∧
  (∃ t : ℝ, ∀ x, ∃ y : ℝ, f 2 y ≥ |2*x - 1| - t^2 + (3/2)*t + 1 ↔ t ≤ 1/2 ∨ t ≥ 1) :=
by sorry

end problem_solution_l3424_342425


namespace unique_solution_l3424_342435

theorem unique_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 2*x - 2*y + 1/z = 1/2014)
  (eq2 : 2*y - 2*z + 1/x = 1/2014)
  (eq3 : 2*z - 2*x + 1/y = 1/2014) :
  x = 2014 ∧ y = 2014 ∧ z = 2014 := by
sorry

end unique_solution_l3424_342435


namespace triangle_max_tan_diff_l3424_342496

open Real

theorem triangle_max_tan_diff (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a * cos B - b * cos A = c / 2 →
  (∀ θ, 0 < θ ∧ θ < π → tan (A - B) ≤ tan (A - θ)) →
  B = π / 6 :=
sorry

end triangle_max_tan_diff_l3424_342496


namespace inequalities_hold_l3424_342488

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ((a + b) * (1/a + 1/b) ≥ 4) ∧ 
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧ 
  (Real.sqrt (abs (a - b)) ≥ Real.sqrt a - Real.sqrt b) :=
by sorry

end inequalities_hold_l3424_342488


namespace cake_recipe_ratio_l3424_342470

/-- Given a recipe with 60 eggs and a total of 90 cups of flour and eggs,
    prove that the ratio of cups of flour to eggs is 1:2. -/
theorem cake_recipe_ratio : 
  ∀ (flour eggs : ℕ), 
    eggs = 60 →
    flour + eggs = 90 →
    (flour : ℚ) / (eggs : ℚ) = 1 / 2 := by
  sorry

end cake_recipe_ratio_l3424_342470


namespace hexagon_diagonals_intersect_l3424_342421

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A hexagon in a 2D plane -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A line dividing a side of a triangle into three equal parts -/
def dividingLine (T : Triangle) (vertex : Fin 3) : ℝ × ℝ → ℝ × ℝ → Prop :=
  sorry

/-- The hexagon formed by the dividing lines -/
def formHexagon (T : Triangle) : Hexagon :=
  sorry

/-- The diagonals of a hexagon -/
def diagonals (H : Hexagon) : List (ℝ × ℝ → ℝ × ℝ → Prop) :=
  sorry

/-- The intersection point of lines -/
def intersectionPoint (lines : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Option (ℝ × ℝ) :=
  sorry

/-- Main theorem -/
theorem hexagon_diagonals_intersect (T : Triangle) :
  let H := formHexagon T
  let diag := diagonals H
  ∃ p : ℝ × ℝ, intersectionPoint diag = some p :=
by sorry

end hexagon_diagonals_intersect_l3424_342421


namespace polynomial_remainder_l3424_342432

/-- Given a polynomial Q with Q(10) = 5 and Q(50) = 15, 
    the remainder when Q is divided by (x - 10)(x - 50) is (1/4)x + 2.5 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 10 = 5) (h2 : Q 50 = 15) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 10) * (x - 50) * R x + 1/4 * x + 5/2 := by
  sorry

end polynomial_remainder_l3424_342432


namespace subtraction_of_negative_integers_l3424_342477

theorem subtraction_of_negative_integers : -3 - 2 = -5 := by
  sorry

end subtraction_of_negative_integers_l3424_342477


namespace bags_given_away_bags_given_away_equals_two_l3424_342476

def initial_purchase : ℕ := 3
def second_purchase : ℕ := 3
def remaining_bags : ℕ := 4

theorem bags_given_away : ℕ := by
  sorry

theorem bags_given_away_equals_two : bags_given_away = 2 := by
  sorry

end bags_given_away_bags_given_away_equals_two_l3424_342476


namespace living_room_set_cost_l3424_342401

theorem living_room_set_cost (couch_cost sectional_cost other_cost : ℕ)
  (discount_rate : ℚ) (h1 : couch_cost = 2500) (h2 : sectional_cost = 3500)
  (h3 : other_cost = 2000) (h4 : discount_rate = 1/10) :
  (couch_cost + sectional_cost + other_cost) * (1 - discount_rate) = 7200 :=
by sorry

end living_room_set_cost_l3424_342401


namespace always_two_real_roots_unique_m_value_l3424_342451

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 4*m*x + 3*m^2 = 0

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

-- Theorem 2: When m > 0 and the difference between roots is 2, m = 1
theorem unique_m_value (m : ℝ) (h₁ : m > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ x₁ - x₂ = 2) →
  m = 1 :=
sorry

end always_two_real_roots_unique_m_value_l3424_342451


namespace linear_systems_solution_and_expression_l3424_342436

theorem linear_systems_solution_and_expression (a b : ℝ) : 
  (∃ x y : ℝ, (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
              (2 * x + 5 * y = -26 ∧ a * x - b * y = -4)) →
  (∃ x y : ℝ, x = 2 ∧ y = -6 ∧
              (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
              (2 * x + 5 * y = -26 ∧ a * x - b * y = -4)) ∧
  (2 * a + b)^2023 = 1 :=
by sorry

end linear_systems_solution_and_expression_l3424_342436


namespace power_of_power_at_three_l3424_342429

theorem power_of_power_at_three :
  (3^3)^(3^3) = 27^27 := by sorry

end power_of_power_at_three_l3424_342429


namespace temperature_reaches_target_l3424_342445

/-- The temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The target temperature -/
def target_temp : ℝ := 80

/-- The latest time when the temperature reaches the target -/
def latest_time : ℝ := 10

theorem temperature_reaches_target :
  (∃ t : ℝ, temperature t = target_temp) ∧
  (∀ t : ℝ, temperature t = target_temp → t ≤ latest_time) ∧
  (temperature latest_time = target_temp) := by
  sorry

end temperature_reaches_target_l3424_342445


namespace min_cuts_for_eleven_sided_polygons_l3424_342416

/-- Represents a straight-line cut on a piece of paper -/
structure Cut where
  -- Add necessary fields

/-- Represents a polygon on the table -/
structure Polygon where
  sides : ℕ

/-- Represents the state of the paper after a series of cuts -/
structure PaperState where
  polygons : List Polygon

/-- Function to apply a cut to a paper state -/
def applyCut (state : PaperState) (cut : Cut) : PaperState :=
  sorry

/-- Function to count the number of eleven-sided polygons in a paper state -/
def countElevenSidedPolygons (state : PaperState) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_eleven_sided_polygons :
  ∀ (initial : PaperState),
    (∃ (cuts : List Cut),
      cuts.length = 2015 ∧
      countElevenSidedPolygons (cuts.foldl applyCut initial) ≥ 252) ∧
    (∀ (cuts : List Cut),
      cuts.length < 2015 →
      countElevenSidedPolygons (cuts.foldl applyCut initial) < 252) :=
by
  sorry

end min_cuts_for_eleven_sided_polygons_l3424_342416


namespace chess_tournament_games_l3424_342423

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 7 players, where each player plays every other player once,
    the total number of games played is 21. --/
theorem chess_tournament_games :
  num_games 7 = 21 := by
  sorry

end chess_tournament_games_l3424_342423


namespace chinese_chess_draw_probability_l3424_342478

theorem chinese_chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.6) 
  (h_not_lose : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.3 := by
  sorry

end chinese_chess_draw_probability_l3424_342478


namespace tire_usage_calculation_l3424_342427

/- Define the problem parameters -/
def total_miles : ℕ := 42000
def total_tires : ℕ := 7
def tires_used_simultaneously : ℕ := 6

/- Theorem statement -/
theorem tire_usage_calculation :
  let total_tire_miles : ℕ := total_miles * tires_used_simultaneously
  let miles_per_tire : ℕ := total_tire_miles / total_tires
  miles_per_tire = 36000 := by
  sorry

end tire_usage_calculation_l3424_342427


namespace multiply_993_879_l3424_342474

theorem multiply_993_879 : 993 * 879 = 872847 := by
  -- Define the method
  let a := 993
  let b := 879
  let n := 7
  
  -- Step 1: Subtract n from b
  let b_minus_n := b - n
  
  -- Step 2: Add n to a
  let a_plus_n := a + n
  
  -- Step 3: Multiply results of steps 1 and 2
  let product_step3 := b_minus_n * a_plus_n
  
  -- Step 4: Calculate the difference
  let diff := a - b_minus_n
  
  -- Step 5: Multiply the difference by n
  let product_step5 := diff * n
  
  -- Step 6: Add results of steps 3 and 5
  let result := product_step3 + product_step5
  
  -- Prove that the result equals 872847
  sorry

end multiply_993_879_l3424_342474


namespace min_phi_value_l3424_342481

/-- Given a function f and a constant φ, this theorem proves that under certain conditions,
    the minimum value of φ is 5π/12. -/
theorem min_phi_value (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x) * Real.cos (2 * φ) + Real.cos (2 * x) * Real.sin (2 * φ)) →
  φ > 0 →
  (∀ x, f x = f (2 * π / 3 - x)) →
  ∃ k : ℤ, φ = k * π / 2 - π / 12 ∧ 
  (∀ m : ℤ, m * π / 2 - π / 12 > 0 → φ ≤ m * π / 2 - π / 12) :=
sorry

end min_phi_value_l3424_342481


namespace triangle_heights_sum_ge_nine_times_inradius_l3424_342492

/-- Given a triangle with heights h₁, h₂, h₃ and an inscribed circle of radius r,
    the sum of the heights is greater than or equal to 9 times the radius. -/
theorem triangle_heights_sum_ge_nine_times_inradius 
  (h₁ h₂ h₃ r : ℝ) 
  (height_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (inradius_positive : r > 0)
  (triangle_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = 2 * (a * b * c).sqrt / (a * (a + b + c)) ∧
    h₂ = 2 * (a * b * c).sqrt / (b * (a + b + c)) ∧
    h₃ = 2 * (a * b * c).sqrt / (c * (a + b + c)))
  (inradius_def : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    r = (a * b * c).sqrt / (a + b + c)) :
  h₁ + h₂ + h₃ ≥ 9 * r := by
  sorry

end triangle_heights_sum_ge_nine_times_inradius_l3424_342492


namespace cow_chicken_problem_l3424_342484

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 16) → cows = 8 := by
  sorry

end cow_chicken_problem_l3424_342484


namespace sum_of_reciprocals_l3424_342475

theorem sum_of_reciprocals (a b : ℝ) (ha : a^2 - 3*a + 2 = 0) (hb : b^2 - 3*b + 2 = 0) (hab : a ≠ b) :
  1/a + 1/b = 3/2 := by sorry

end sum_of_reciprocals_l3424_342475


namespace existence_of_sum_equality_l3424_342473

theorem existence_of_sum_equality (n : ℕ) (a : Fin (n + 1) → ℤ)
  (h_n : n > 3)
  (h_a : ∀ i j : Fin (n + 1), i < j → a i < a j)
  (h_lower : a 0 ≥ 1)
  (h_upper : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin (n + 1)),
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
    k ≠ l ∧ k ≠ m ∧
    l ≠ m ∧
    a i + a j = a k + a l ∧ a k + a l = a m :=
by sorry

end existence_of_sum_equality_l3424_342473


namespace arrangement_theorem_l3424_342447

/-- The number of ways to arrange n distinct objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct objects with k specific objects always adjacent --/
def permutations_with_adjacent (n k : ℕ) : ℕ :=
  permutations (n - k + 1) * permutations k

/-- The number of ways to arrange n distinct objects with k specific objects always adjacent
    and m specific objects never adjacent --/
def permutations_with_adjacent_and_not_adjacent (n k m : ℕ) : ℕ :=
  permutations_with_adjacent n k - permutations_with_adjacent (n - m + 1) (k + m - 1)

theorem arrangement_theorem :
  (permutations_with_adjacent 5 2 = 48) ∧
  (permutations_with_adjacent_and_not_adjacent 5 2 1 = 36) := by
  sorry

end arrangement_theorem_l3424_342447


namespace isosceles_triangle_smallest_base_l3424_342407

theorem isosceles_triangle_smallest_base 
  (α : ℝ) 
  (q : ℝ) 
  (h_α : 0 < α ∧ α < π) 
  (h_q : q > 0) :
  let base (a : ℝ) := 
    Real.sqrt (q^2 * ((1 - Real.cos α) / 2) + 2 * (1 + Real.cos α) * (a - q/2)^2)
  ∀ a, 0 < a ∧ a < q → base (q/2) ≤ base a :=
by sorry

end isosceles_triangle_smallest_base_l3424_342407


namespace complex_equation_solution_l3424_342498

/-- Given a complex number z and a real number a satisfying the equation (2+i)z = a+2i,
    where the real part of z is twice its imaginary part, prove that a = 3/2. -/
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
    (h1 : (2 + Complex.I) * z = a + 2 * Complex.I)
    (h2 : z.re = 2 * z.im) : 
  a = 3/2 := by sorry

end complex_equation_solution_l3424_342498


namespace infinite_series_sum_l3424_342428

theorem infinite_series_sum : 
  let r : ℝ := (1 : ℝ) / 1000
  let series_sum := ∑' n, (n : ℝ)^2 * r^(n - 1)
  series_sum = (r + 1) / ((1 - r)^3) := by
  sorry

end infinite_series_sum_l3424_342428


namespace emily_jumps_in_75_seconds_l3424_342463

/-- Emily's jumping rate in jumps per second -/
def jumping_rate : ℚ := 52 / 60

/-- The number of jumps Emily makes in a given time -/
def jumps (time : ℚ) : ℚ := jumping_rate * time

theorem emily_jumps_in_75_seconds : 
  jumps 75 = 65 := by sorry

end emily_jumps_in_75_seconds_l3424_342463


namespace min_ratio_two_digit_integers_l3424_342464

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → -- x and y are two-digit positive integers
  (x + y) / 2 = 55 → -- mean of x and y is 55
  ∀ a b : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ (a + b) / 2 = 55 →
  x / y ≤ a / b →
  x / y = 1 / 9 := by
sorry

end min_ratio_two_digit_integers_l3424_342464


namespace minimum_fourth_quarter_score_l3424_342494

def required_average : ℝ := 85
def num_quarters : ℕ := 4
def first_quarter_score : ℝ := 84
def second_quarter_score : ℝ := 80
def third_quarter_score : ℝ := 78

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let current_total := first_quarter_score + second_quarter_score + third_quarter_score
  let minimum_score := total_required - current_total
  minimum_score = 98 := by sorry

end minimum_fourth_quarter_score_l3424_342494


namespace megan_lead_actress_percentage_l3424_342400

def total_plays : ℕ := 100
def not_lead_plays : ℕ := 20

theorem megan_lead_actress_percentage :
  (total_plays - not_lead_plays) * 100 / total_plays = 80 := by
  sorry

end megan_lead_actress_percentage_l3424_342400


namespace rainwater_chickens_l3424_342430

/-- Mr. Rainwater's farm animals -/
structure Farm where
  cows : ℕ
  goats : ℕ
  chickens : ℕ

/-- The conditions of Mr. Rainwater's farm -/
def rainwater_farm (f : Farm) : Prop :=
  f.cows = 9 ∧ f.goats = 4 * f.cows ∧ f.goats = 2 * f.chickens

/-- Theorem: Mr. Rainwater has 18 chickens -/
theorem rainwater_chickens (f : Farm) (h : rainwater_farm f) : f.chickens = 18 := by
  sorry

end rainwater_chickens_l3424_342430


namespace product_of_roots_l3424_342480

theorem product_of_roots (x : ℝ) : 
  (x^3 - 9*x^2 + 27*x - 64 = 0) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 64 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 64) := by
sorry

end product_of_roots_l3424_342480


namespace selling_price_calculation_l3424_342443

theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 45 →
  gain_percentage = 30 →
  ∃ (cost_price : ℝ) (selling_price : ℝ),
    gain = (gain_percentage / 100) * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 195 :=
by sorry

end selling_price_calculation_l3424_342443


namespace book_loss_percentage_l3424_342485

/-- If the cost price of 8 books equals the selling price of 16 books, then the loss percentage is 50% -/
theorem book_loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : (C - S) / C * 100 = 50 := by
  sorry

end book_loss_percentage_l3424_342485


namespace product_of_polynomials_l3424_342440

theorem product_of_polynomials (p q : ℤ) : 
  (∀ d : ℤ, (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) →
  p = -5 ∧ q = 8 := by
sorry

end product_of_polynomials_l3424_342440


namespace sandcastle_height_difference_l3424_342431

-- Define the heights of the sandcastles
def miki_height : ℝ := 0.8333333333333334
def sister_height : ℝ := 0.5

-- Theorem to prove
theorem sandcastle_height_difference :
  miki_height - sister_height = 0.3333333333333334 := by
  sorry

end sandcastle_height_difference_l3424_342431


namespace robyn_packs_l3424_342469

def lucy_packs : ℕ := 19
def total_packs : ℕ := 35

theorem robyn_packs : total_packs - lucy_packs = 16 := by sorry

end robyn_packs_l3424_342469


namespace product_digit_sum_l3424_342420

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem product_digit_sum :
  let c : ℕ := 777
  let d : ℕ := 444
  sum_of_digits (7 * c * d) = 27 := by sorry

end product_digit_sum_l3424_342420


namespace power_of_power_three_l3424_342487

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_l3424_342487


namespace square_expansion_area_increase_l3424_342444

theorem square_expansion_area_increase (a : ℝ) : 
  (a + 2)^2 - a^2 = 4*a + 4 := by
  sorry

end square_expansion_area_increase_l3424_342444


namespace absolute_value_inequality_range_l3424_342422

theorem absolute_value_inequality_range (k : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < k) ↔ k > -3 := by sorry

end absolute_value_inequality_range_l3424_342422


namespace tan_alpha_plus_pi_sixth_l3424_342455

theorem tan_alpha_plus_pi_sixth (α : Real) (h : α > 0) (h' : α < π / 2) 
  (h_eq : Real.sqrt 3 * Real.sin α + Real.cos α = 8 / 5) : 
  Real.tan (α + π / 6) = 4 / 3 := by
sorry

end tan_alpha_plus_pi_sixth_l3424_342455


namespace polynomial_simplification_l3424_342441

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  (((x^2 + a)^2) / ((a - b)*(a - c)) + ((x^2 + b)^2) / ((b - a)*(b - c)) + ((x^2 + c)^2) / ((c - a)*(c - b))) =
  x^4 + x^2*(a + b + c) + (a^2 + b^2 + c^2) := by
  sorry

#check polynomial_simplification

end polynomial_simplification_l3424_342441


namespace cubic_polynomial_integer_root_l3424_342409

theorem cubic_polynomial_integer_root 
  (b c : ℚ) 
  (h1 : (3 - Real.sqrt 5)^3 + b*(3 - Real.sqrt 5) + c = 0) 
  (h2 : ∃ (n : ℤ), n^3 + b*n + c = 0) :
  ∃ (n : ℤ), n^3 + b*n + c = 0 ∧ n = -6 := by
  sorry

end cubic_polynomial_integer_root_l3424_342409


namespace tank_length_proof_l3424_342410

/-- Proves that a rectangular tank with given dimensions and plastering cost has a specific length -/
theorem tank_length_proof (width depth cost_per_sqm total_cost : ℝ) 
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost_per_sqm : cost_per_sqm = 0.70)
  (h_total_cost : total_cost = 520.8)
  : ∃ length : ℝ, 
    length = 25 ∧ 
    total_cost = (2 * width * depth + 2 * length * depth + width * length) * cost_per_sqm :=
by sorry

end tank_length_proof_l3424_342410


namespace no_adjacent_standing_probability_l3424_342439

def num_people : ℕ := 10

-- Define a recursive function to calculate the number of favorable arrangements
def favorable_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => favorable_arrangements (n + 1) + favorable_arrangements (n + 2)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^num_people

-- Define the probability
def probability : ℚ := favorable_arrangements num_people / total_outcomes

-- Theorem statement
theorem no_adjacent_standing_probability :
  probability = 123 / 1024 := by sorry

end no_adjacent_standing_probability_l3424_342439


namespace symmetric_points_sum_l3424_342449

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

/-- Given two points A(a, 4) and B(-3, b) symmetric with respect to the origin, prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin (a, 4) (-3, b)) : 
  a + b = -1 := by
  sorry

end symmetric_points_sum_l3424_342449


namespace inscribed_rectangle_area_l3424_342411

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 - 12*x + 32

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define the conditions of the problem
def inscribedRectangle (r : Rectangle) : Prop :=
  ∃ t : ℝ,
    r.base = 2*t ∧
    r.height = (2*t)/3 ∧
    parabola (6 - t) = r.height ∧
    t > 0

-- The theorem to prove
theorem inscribed_rectangle_area :
  ∀ r : Rectangle, inscribedRectangle r →
    r.base * r.height = 91 + 25 * Real.sqrt 13 := by
  sorry

end inscribed_rectangle_area_l3424_342411


namespace tangent_lines_theorem_l3424_342472

-- Define the function f(x)
def f (t : ℝ) (x : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

-- Define the derivative of f(x)
def f_deriv (t : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (t - 1) * x

theorem tangent_lines_theorem (t k : ℝ) (hk : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f_deriv t x₁ = k ∧
    f_deriv t x₂ = k ∧
    f t x₁ = 2 * x₁ - 1 ∧
    f t x₂ = 2 * x₂ - 1) →
  t + k = 7 := by
sorry

end tangent_lines_theorem_l3424_342472


namespace battle_station_staffing_l3424_342468

theorem battle_station_staffing (n m : ℕ) (h1 : n = 12) (h2 : m = 4) :
  (n.factorial / ((n - m).factorial * m.factorial)) = 11880 := by
  sorry

end battle_station_staffing_l3424_342468


namespace correct_calculation_l3424_342453

theorem correct_calculation (a b : ℝ) : (a * b)^2 / (-a * b) = -a * b := by
  sorry

end correct_calculation_l3424_342453


namespace similar_polygon_area_sum_l3424_342402

/-- Given two similar polygons, constructs a third similar polygon with area equal to the sum of the given polygons' areas -/
theorem similar_polygon_area_sum 
  (t₁ t₂ : ℝ) 
  (a₁ a₂ : ℝ) 
  (h_positive : t₁ > 0 ∧ t₂ > 0 ∧ a₁ > 0 ∧ a₂ > 0)
  (h_similar : t₁ / (a₁^2) = t₂ / (a₂^2)) :
  let b := Real.sqrt (a₁^2 + a₂^2)
  let t₃ := t₁ + t₂
  t₃ / b^2 = t₁ / a₁^2 := by sorry

end similar_polygon_area_sum_l3424_342402


namespace inscribed_hexagon_area_l3424_342442

/-- The area of a regular hexagon inscribed in a circle with area 324π -/
theorem inscribed_hexagon_area :
  ∀ (circle_area hexagon_area : ℝ),
  circle_area = 324 * Real.pi →
  hexagon_area = 6 * (((Real.sqrt (circle_area / Real.pi)) ^ 2 * Real.sqrt 3) / 4) →
  hexagon_area = 486 * Real.sqrt 3 :=
by sorry

end inscribed_hexagon_area_l3424_342442


namespace garden_fence_posts_l3424_342408

/-- Calculates the minimum number of fence posts needed for a rectangular garden -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let wall_side := max length width
  let fenced_perimeter := perimeter - wall_side
  let posts_on_long_side := wall_side / post_spacing + 1
  let posts_on_short_sides := 2 * (fenced_perimeter - wall_side) / post_spacing
  posts_on_long_side + posts_on_short_sides

/-- Proves that for a 50m by 80m garden with 10m post spacing, 17 posts are needed -/
theorem garden_fence_posts :
  min_fence_posts 80 50 10 = 17 := by
  sorry

#eval min_fence_posts 80 50 10

end garden_fence_posts_l3424_342408


namespace trajectory_is_ellipse_l3424_342415

theorem trajectory_is_ellipse (x y : ℝ) 
  (h1 : (2*y)^2 = (1+x)*(1-x)) 
  (h2 : y ≠ 0) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end trajectory_is_ellipse_l3424_342415


namespace students_just_passed_l3424_342418

/-- The number of students who just passed an examination -/
theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 26 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent < 1) :
  total - (first_div_percent * total).floor - (second_div_percent * total).floor = 60 := by
  sorry

end students_just_passed_l3424_342418


namespace min_value_expression_l3424_342417

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 :=
by sorry

end min_value_expression_l3424_342417


namespace circle_passes_through_intersection_point_l3424_342495

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2*y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (4, 3)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 25

-- Theorem statement
theorem circle_passes_through_intersection_point :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ circle_equation x y :=
sorry

end circle_passes_through_intersection_point_l3424_342495


namespace cafeteria_apples_theorem_l3424_342426

/-- Given the initial number of apples, the number of pies made, and the number of apples per pie,
    calculate the number of apples handed out to students. -/
def apples_handed_out (initial_apples : ℕ) (pies_made : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - pies_made * apples_per_pie

/-- Theorem stating that for the given problem, 30 apples were handed out to students. -/
theorem cafeteria_apples_theorem :
  apples_handed_out 86 7 8 = 30 := by
  sorry

#eval apples_handed_out 86 7 8

end cafeteria_apples_theorem_l3424_342426


namespace first_class_product_rate_l3424_342489

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    calculate the overall rate of first-class products. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_qualified : ℝ)
  (h1 : pass_rate = 0.95)
  (h2 : first_class_rate_qualified = 0.2) :
  pass_rate * first_class_rate_qualified = 0.19 :=
by sorry

end first_class_product_rate_l3424_342489


namespace seating_arrangements_with_restrictions_l3424_342457

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to seat n people in a row where two specific people must sit together -/
def arrangementsWithPairTogether (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * (Nat.factorial 2)

/-- The number of ways to seat n people in a row where three specific people must sit together -/
def arrangementsWithTrioTogether (n : ℕ) : ℕ := (Nat.factorial (n - 2)) * (Nat.factorial 3)

/-- The number of ways to seat 7 people in a row where 3 specific people cannot sit next to each other -/
theorem seating_arrangements_with_restrictions : 
  totalArrangements 7 - 3 * arrangementsWithPairTogether 7 + arrangementsWithTrioTogether 7 = 1440 := by
  sorry

end seating_arrangements_with_restrictions_l3424_342457


namespace sqrt_inequality_fraction_product_inequality_l3424_342465

-- Part 1
theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

-- Part 2
theorem fraction_product_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end sqrt_inequality_fraction_product_inequality_l3424_342465


namespace min_xyz_l3424_342433

theorem min_xyz (x y z : ℝ) (h1 : x * y + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10) : 
  ∀ (a b c : ℝ), a * b * c ≥ -28 → x * y * z ≥ -28 :=
by sorry

end min_xyz_l3424_342433


namespace circle_tangent_y_intercept_l3424_342459

/-- Two circles with given centers and radii have a common external tangent with y-intercept 135/28 -/
theorem circle_tangent_y_intercept :
  ∃ (m b : ℝ),
    m > 0 ∧
    b = 135 / 28 ∧
    ∀ (x y : ℝ),
      (y = m * x + b) →
      ((x - 1)^2 + (y - 3)^2 = 3^2 ∨ (x - 10)^2 + (y - 8)^2 = 6^2) →
      ∀ (x' y' : ℝ),
        ((x' - 1)^2 + (y' - 3)^2 < 3^2 ∧ (x' - 10)^2 + (y' - 8)^2 < 6^2) →
        (y' ≠ m * x' + b) :=
by sorry

end circle_tangent_y_intercept_l3424_342459


namespace symmetry_properties_l3424_342471

/-- Two rational numbers are symmetric about a point with a given symmetric radius. -/
def symmetric (m n p r : ℚ) : Prop :=
  m ≠ n ∧ m ≠ p ∧ n ≠ p ∧ |m - p| = r ∧ |n - p| = r

theorem symmetry_properties :
  (∃ x r : ℚ, symmetric 3 x 1 r ∧ x = -1 ∧ r = 2) ∧
  (∃ a b r : ℚ, symmetric a b 2 r ∧ |a| = 2 * |b| ∧ (r = 2/3 ∨ r = 6)) :=
by sorry

end symmetry_properties_l3424_342471


namespace first_month_sale_is_6435_l3424_342434

/-- Represents the sales data for a grocery shop over 6 months -/
structure GrocerySales where
  average_target : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the sale in the first month given the sales data -/
def first_month_sale (s : GrocerySales) : ℕ :=
  6 * s.average_target - (s.month2 + s.month3 + s.month4 + s.month5 + s.month6)

/-- Theorem stating that the first month's sale is 6435 given the specific sales data -/
theorem first_month_sale_is_6435 :
  let s : GrocerySales := {
    average_target := 6500,
    month2 := 6927,
    month3 := 6855,
    month4 := 7230,
    month5 := 6562,
    month6 := 4991
  }
  first_month_sale s = 6435 := by
  sorry

end first_month_sale_is_6435_l3424_342434


namespace square_area_from_diagonal_l3424_342412

/-- The area of a square with diagonal length 20 is 200 -/
theorem square_area_from_diagonal : 
  ∀ s : ℝ, s > 0 → s * s * 2 = 20 * 20 → s * s = 200 := by
  sorry

end square_area_from_diagonal_l3424_342412


namespace solution_set_when_a_is_neg_one_a_range_when_f_bounded_l3424_342413

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x - 4|

-- Theorem for part I
theorem solution_set_when_a_is_neg_one :
  {x : ℝ | f (-1) x ≥ 4} = {x : ℝ | x ≥ 7/2} := by sorry

-- Theorem for part II
theorem a_range_when_f_bounded :
  (∀ x : ℝ, |f a x| ≤ 2) → a ∈ Set.Icc 2 6 := by sorry

end solution_set_when_a_is_neg_one_a_range_when_f_bounded_l3424_342413


namespace birthday_theorem_l3424_342437

def birthday_money (age : ℕ) : ℕ := age * 5

theorem birthday_theorem : 
  ∀ (age : ℕ), age = 3 + 3 * 3 → birthday_money age = 60 := by
  sorry

end birthday_theorem_l3424_342437


namespace subtracted_number_l3424_342452

theorem subtracted_number (t k x : ℝ) : 
  t = 5/9 * (k - x) → 
  t = 20 → 
  k = 68 → 
  x = 32 := by
sorry

end subtracted_number_l3424_342452


namespace no_primes_in_perm_numbers_l3424_342467

/-- A permutation of the digits 1, 2, 3, 4, 5 -/
def Perm5 : Type := Fin 5 → Fin 5

/-- Converts a permutation to a 5-digit number -/
def toNumber (p : Perm5) : ℕ :=
  10000 * (p 0).val + 1000 * (p 1).val + 100 * (p 2).val + 10 * (p 3).val + (p 4).val + 11111

/-- The set of all 5-digit numbers formed by permutations of 1, 2, 3, 4, 5 -/
def PermNumbers : Set ℕ :=
  {n | ∃ p : Perm5, toNumber p = n}

theorem no_primes_in_perm_numbers : ∀ n ∈ PermNumbers, ¬ Nat.Prime n := by
  sorry

end no_primes_in_perm_numbers_l3424_342467


namespace division_sum_theorem_l3424_342448

theorem division_sum_theorem (dividend : Nat) (divisor : Nat) (quotient : Nat) :
  dividend = 82502 →
  divisor ≥ 100 ∧ divisor < 1000 →
  dividend = divisor * quotient →
  divisor + quotient = 723 := by
  sorry

end division_sum_theorem_l3424_342448


namespace find_heaviest_coin_l3424_342450

/-- Represents a weighing scale that may be faulty -/
structure Scale :=
  (isDefective : Bool)

/-- Represents a coin with a certain mass -/
structure Coin :=
  (mass : ℕ)

/-- The minimum number of weighings needed to find the heaviest coin -/
def minWeighings (n : ℕ) : ℕ := 2 * n - 1

theorem find_heaviest_coin (n : ℕ) (h : n > 2) 
  (coins : Fin n → Coin) 
  (scales : Fin n → Scale) 
  (all_different_masses : ∀ i j, i ≠ j → (coins i).mass ≠ (coins j).mass)
  (one_faulty_scale : ∃ i, (scales i).isDefective) :
  ∃ (num_weighings : ℕ), 
    num_weighings = minWeighings n ∧ 
    (∃ heaviest : Fin n, ∀ i, (coins heaviest).mass ≥ (coins i).mass) :=
sorry

end find_heaviest_coin_l3424_342450


namespace min_value_theorem_l3424_342461

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 4/b ≥ 9/2 := by sorry

end min_value_theorem_l3424_342461


namespace triangle_formation_check_l3424_342482

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 9), (50, 60, 12), (11, 11, 31), (20, 30, 50)]

theorem triangle_formation_check :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ 
    let (a, b, c) := set
    can_form_triangle a b c :=
by sorry

end triangle_formation_check_l3424_342482


namespace quadratic_equation_result_l3424_342499

theorem quadratic_equation_result (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) : 
  (14 * y - 5)^2 = 333 := by
sorry

end quadratic_equation_result_l3424_342499


namespace tiffany_bag_collection_l3424_342446

/-- Represents the number of bags of cans Tiffany collected over three days -/
structure BagCollection where
  monday : Nat
  nextDay : Nat
  dayAfter : Nat
  total : Nat

/-- Theorem stating that given the conditions from the problem, 
    the number of bags collected on the next day must be 3 -/
theorem tiffany_bag_collection (bc : BagCollection) 
  (h1 : bc.monday = 10)
  (h2 : bc.dayAfter = 7)
  (h3 : bc.total = 20)
  (h4 : bc.monday + bc.nextDay + bc.dayAfter = bc.total) :
  bc.nextDay = 3 := by
  sorry

end tiffany_bag_collection_l3424_342446


namespace potato_division_l3424_342456

theorem potato_division (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end potato_division_l3424_342456
