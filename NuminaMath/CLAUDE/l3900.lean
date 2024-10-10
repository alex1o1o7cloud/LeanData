import Mathlib

namespace flywheel_power_l3900_390093

/-- Calculates the power of a flywheel's driving machine in horsepower -/
theorem flywheel_power (r : ℝ) (m : ℝ) (n : ℝ) (t : ℝ) : 
  r = 3 →
  m = 6000 →
  n = 800 →
  t = 3 →
  ∃ (p : ℝ), abs (p - 1431) < 1 ∧ 
  p = (m * (r * n * 2 * Real.pi / 60)^2) / (2 * t * 60 * 746) := by
  sorry

end flywheel_power_l3900_390093


namespace multiple_of_seven_in_range_l3900_390065

theorem multiple_of_seven_in_range (y : ℕ) (h1 : ∃ k : ℕ, y = 7 * k)
    (h2 : y * y > 225) (h3 : y < 30) : y = 21 := by
  sorry

end multiple_of_seven_in_range_l3900_390065


namespace stacked_squares_area_l3900_390045

/-- Represents a square sheet of paper -/
structure Square where
  side_length : ℝ

/-- Represents the configuration of four stacked squares -/
structure StackedSquares where
  base : Square
  rotated45 : Square
  middle : Square
  rotated90 : Square

/-- The area of the polygon formed by the stacked squares -/
def polygon_area (s : StackedSquares) : ℝ := sorry

theorem stacked_squares_area :
  ∀ (s : StackedSquares),
    s.base.side_length = 8 ∧
    s.rotated45.side_length = 8 ∧
    s.middle.side_length = 8 ∧
    s.rotated90.side_length = 8 →
    polygon_area s = 192 - 128 * Real.sqrt 2 :=
by sorry

end stacked_squares_area_l3900_390045


namespace problem_1_l3900_390001

theorem problem_1 (a b : ℝ) (h : a ≠ 0) (h' : b ≠ 0) :
  (-3*b/(2*a)) * (6*a/b^3) = -9/b^2 :=
sorry

end problem_1_l3900_390001


namespace shredder_capacity_l3900_390083

/-- Given a paper shredder that can shred 6 pages at a time,
    and 44 shredding operations, prove that the total number
    of pages shredded is 264. -/
theorem shredder_capacity (pages_per_operation : Nat) (num_operations : Nat) :
  pages_per_operation = 6 → num_operations = 44 → pages_per_operation * num_operations = 264 := by
  sorry

end shredder_capacity_l3900_390083


namespace rectangle_perimeter_l3900_390039

/-- Given a rectangle A composed of 3 equal squares with a perimeter of 112 cm,
    prove that a rectangle B composed of 4 of the same squares will have a perimeter of 140 cm. -/
theorem rectangle_perimeter (side_length : ℝ) : 
  (3 * side_length * 2 + 2 * side_length) = 112 → 
  (4 * side_length * 2 + 2 * side_length) = 140 := by
sorry

end rectangle_perimeter_l3900_390039


namespace corrected_mean_l3900_390032

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 30 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (30.5 : ℚ) :=
by sorry

end corrected_mean_l3900_390032


namespace chocolate_leftover_l3900_390096

/-- Calculates the amount of chocolate left over when making cookies -/
theorem chocolate_leftover (dough : ℝ) (total_chocolate : ℝ) (chocolate_percentage : ℝ) : 
  dough = 36 → 
  total_chocolate = 13 → 
  chocolate_percentage = 0.20 → 
  (total_chocolate - (chocolate_percentage * (dough + (chocolate_percentage * (dough + total_chocolate) / (1 - chocolate_percentage))))) = 4 := by
  sorry

end chocolate_leftover_l3900_390096


namespace number_of_divisors_of_360_l3900_390027

theorem number_of_divisors_of_360 : Finset.card (Nat.divisors 360) = 24 := by
  sorry

end number_of_divisors_of_360_l3900_390027


namespace oliver_monster_club_cards_l3900_390023

/-- Represents Oliver's card collection --/
structure CardCollection where
  alien_baseball : ℕ
  monster_club : ℕ
  battle_gremlins : ℕ

/-- The conditions of Oliver's card collection --/
def oliver_collection : CardCollection :=
  { alien_baseball := 18,
    monster_club := 27,
    battle_gremlins := 72 }

/-- Theorem stating the number of Monster Club cards Oliver has --/
theorem oliver_monster_club_cards :
  oliver_collection.monster_club = 27 ∧
  oliver_collection.monster_club = (3 / 2 : ℚ) * oliver_collection.alien_baseball ∧
  oliver_collection.battle_gremlins = 72 ∧
  oliver_collection.battle_gremlins = 4 * oliver_collection.alien_baseball :=
by
  sorry

end oliver_monster_club_cards_l3900_390023


namespace range_of_a_l3900_390079

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (B a ⊆ Aᶜ) ↔ a ≤ -1 :=
sorry

end range_of_a_l3900_390079


namespace hyperbola_equation_l3900_390037

/-- Proves that a hyperbola with given conditions has the equation x²/3 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.sqrt 3 / 3) →
  (Real.sqrt 3 * a / 3 = 1) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 = 1) :=
by sorry

end hyperbola_equation_l3900_390037


namespace fourth_student_added_25_l3900_390013

/-- The number of jellybeans added by the fourth student to the average of the first three guesses -/
def jellybeans_added (first_guess : ℕ) (fourth_guess : ℕ) : ℕ :=
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let average := (first_guess + second_guess + third_guess) / 3
  fourth_guess - average

/-- Theorem stating that given the conditions in the problem, the fourth student added 25 jellybeans -/
theorem fourth_student_added_25 :
  jellybeans_added 100 525 = 25 := by
  sorry

#eval jellybeans_added 100 525

end fourth_student_added_25_l3900_390013


namespace paul_filled_three_bags_sunday_l3900_390011

/-- Calculates the number of bags filled on Sunday given the total cans collected,
    bags filled on Saturday, and cans per bag. -/
def bags_filled_sunday (total_cans : ℕ) (saturday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag

/-- Proves that for the given problem, Paul filled 3 bags on Sunday. -/
theorem paul_filled_three_bags_sunday :
  bags_filled_sunday 72 6 8 = 3 := by
  sorry

end paul_filled_three_bags_sunday_l3900_390011


namespace star_value_l3900_390066

theorem star_value : ∃ (x : ℚ), 
  45 - ((28 * 3) - (37 - (15 / (x - 2)))) = 57 ∧ x = 103/59 := by
  sorry

end star_value_l3900_390066


namespace euro_problem_l3900_390003

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem euro_problem (n : ℝ) :
  euro 8 (euro 4 n) = 640 → n = 5 := by
  sorry

end euro_problem_l3900_390003


namespace more_wins_probability_correct_l3900_390070

/-- The number of matches played by the team -/
def num_matches : ℕ := 5

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The probability of ending with more wins than losses -/
def more_wins_probability : ℚ := 16/243

theorem more_wins_probability_correct :
  (∀ (outcome : Fin num_matches → Fin 3),
    (∃ (wins losses : ℕ),
      wins > losses ∧
      wins + losses ≤ num_matches ∧
      (∀ i, outcome i = 0 → wins > 0) ∧
      (∀ i, outcome i = 1 → losses > 0))) →
  ∃ (favorable_outcomes : ℕ),
    favorable_outcomes = 16 ∧
    (favorable_outcomes : ℚ) / (3 ^ num_matches) = more_wins_probability :=
sorry

end more_wins_probability_correct_l3900_390070


namespace stating_max_regions_correct_l3900_390087

/-- 
Given two points A and B on a plane, with m lines passing through A and n lines passing through B,
this function calculates the maximum number of regions these m+n lines can divide the plane into.
-/
def max_regions (m n : ℕ) : ℕ :=
  m * n + 2 * m + 2 * n - 1

/-- 
Theorem stating that for any positive natural numbers m and n, 
the maximum number of regions formed by m+n lines (m through point A, n through point B) 
is given by the function max_regions.
-/
theorem max_regions_correct (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  max_regions m n = m * n + 2 * m + 2 * n - 1 :=
by sorry

end stating_max_regions_correct_l3900_390087


namespace identity_function_unique_l3900_390059

/-- A function satisfying the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧ 
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = f x / x^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem identity_function_unique (f : ℝ → ℝ) (h : satisfying_function f) : 
  ∀ x : ℝ, f x = x := by
sorry

end identity_function_unique_l3900_390059


namespace diamonds_in_F20_l3900_390061

/-- Definition of the number of diamonds in figure F_n -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 9
  else n^2 + (n-1)^2

/-- Theorem: The number of diamonds in F_20 is 761 -/
theorem diamonds_in_F20 :
  num_diamonds 20 = 761 := by sorry

end diamonds_in_F20_l3900_390061


namespace problem_statement_l3900_390031

theorem problem_statement (t : ℝ) : 
  let x := 3 - 2*t
  let y := 5*t + 3
  x = 1 → y = 8 := by
sorry

end problem_statement_l3900_390031


namespace line_PQ_parallel_to_x_axis_l3900_390006

-- Define the points P and Q
def P : ℝ × ℝ := (6, -6)
def Q : ℝ × ℝ := (-6, -6)

-- Define a line as parallel to x-axis if y-coordinates are equal
def parallel_to_x_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2

-- Theorem statement
theorem line_PQ_parallel_to_x_axis :
  parallel_to_x_axis P Q := by sorry

end line_PQ_parallel_to_x_axis_l3900_390006


namespace park_fencing_cost_l3900_390028

/-- Proves that for a rectangular park with sides in the ratio 3:2, area of 4704 sq m,
    and a total fencing cost of 140, the cost of fencing per meter is 50 paise. -/
theorem park_fencing_cost (length width : ℝ) (area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = 4704 →
  length * width = area →
  perimeter = 2 * (length + width) →
  total_cost = 140 →
  (total_cost / perimeter) * 100 = 50 := by
  sorry

end park_fencing_cost_l3900_390028


namespace number_problem_l3900_390005

theorem number_problem (x : ℝ) : (0.5 * x = (3/5) * x - 10) → x = 100 := by
  sorry

end number_problem_l3900_390005


namespace david_pushups_l3900_390082

theorem david_pushups (zachary_pushups : ℕ) : 
  zachary_pushups + (zachary_pushups + 49) = 53 →
  zachary_pushups + 49 = 51 := by
  sorry

end david_pushups_l3900_390082


namespace tangent_line_to_circle_l3900_390030

theorem tangent_line_to_circle (x y : ℝ) :
  (x^2 + y^2 = 4) →  -- Circle equation
  (1^2 + (Real.sqrt 3)^2 = 4) →  -- Point (1, √3) is on the circle
  (x + Real.sqrt 3 * y = 4) →  -- Proposed tangent line equation
  ∃ (k : ℝ), k * (x - 1) + Real.sqrt 3 * k * (y - Real.sqrt 3) = 0  -- Tangent line property
  :=
by sorry

end tangent_line_to_circle_l3900_390030


namespace square_sum_value_l3900_390002

theorem square_sum_value (x y : ℝ) (h : (x^2 + y^2)^4 - 6*(x^2 + y^2)^2 + 9 = 0) : 
  x^2 + y^2 = 3 := by
sorry

end square_sum_value_l3900_390002


namespace geometric_sequence_sum_l3900_390041

theorem geometric_sequence_sum (a r : ℚ) (n : ℕ) (h1 : a = 1/3) (h2 : r = 1/3) :
  (a * (1 - r^n) / (1 - r) = 26/81) → n = 4 := by
  sorry

end geometric_sequence_sum_l3900_390041


namespace water_jars_problem_l3900_390026

theorem water_jars_problem (C1 C2 C3 W : ℚ) : 
  W > 0 ∧ C1 > 0 ∧ C2 > 0 ∧ C3 > 0 →
  W = (1/7) * C1 ∧ W = (2/9) * C2 ∧ W = (3/11) * C3 →
  C3 ≥ C1 ∧ C3 ≥ C2 →
  (3 * W) / C3 = 9/11 := by
sorry


end water_jars_problem_l3900_390026


namespace exist_good_numbers_without_digit_sum_property_l3900_390036

/-- A natural number is "good" if its decimal representation contains only zeros and ones. -/
def isGood (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The sum of digits of a natural number in base 10. -/
def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that there exist good numbers whose product is good,
    but the sum of digits property doesn't hold. -/
theorem exist_good_numbers_without_digit_sum_property :
  ∃ (A B : ℕ), isGood A ∧ isGood B ∧ isGood (A * B) ∧
    digitSum (A * B) ≠ digitSum A * digitSum B := by
  sorry


end exist_good_numbers_without_digit_sum_property_l3900_390036


namespace arithmetic_sequence_sum_l3900_390088

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 3 + a 13 = -4) := by
  sorry

end arithmetic_sequence_sum_l3900_390088


namespace dollar_composition_30_l3900_390068

/-- The dollar function as defined in the problem -/
noncomputable def dollar (N : ℝ) : ℝ := 0.75 * N + 2

/-- The statement to be proved -/
theorem dollar_composition_30 : dollar (dollar (dollar 30)) = 17.28125 := by
  sorry

end dollar_composition_30_l3900_390068


namespace cube_edge_ratio_l3900_390040

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 64) : a / b = 4 := by
  sorry

end cube_edge_ratio_l3900_390040


namespace table_size_lower_bound_l3900_390097

/-- A table with 10 columns and n rows, where each cell contains a digit -/
structure Table (n : ℕ) :=
  (cells : Fin n → Fin 10 → Fin 10)

/-- The property that for each row and any two columns, there exists a row
    that differs from it in exactly these two columns -/
def has_differing_rows (t : Table n) : Prop :=
  ∀ (row : Fin n) (col1 col2 : Fin 10),
    col1 ≠ col2 →
    ∃ (diff_row : Fin n),
      (∀ (col : Fin 10), col ≠ col1 ∧ col ≠ col2 → t.cells diff_row col = t.cells row col) ∧
      t.cells diff_row col1 ≠ t.cells row col1 ∧
      t.cells diff_row col2 ≠ t.cells row col2

theorem table_size_lower_bound {n : ℕ} (t : Table n) (h : has_differing_rows t) :
  n ≥ 512 :=
sorry

end table_size_lower_bound_l3900_390097


namespace time_is_point_eight_hours_l3900_390052

/-- The number of unique letters in the name --/
def name_length : ℕ := 6

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 15

/-- Calculates the time in hours required to write all possible rearrangements of a name --/
def time_to_write_all_rearrangements : ℚ :=
  (Nat.factorial name_length : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Theorem stating that the time to write all rearrangements is 0.8 hours --/
theorem time_is_point_eight_hours :
  time_to_write_all_rearrangements = 4/5 := by sorry

end time_is_point_eight_hours_l3900_390052


namespace edwards_remaining_money_l3900_390044

/-- Calculates the remaining money after a purchase -/
def remainingMoney (initialAmount spentAmount : ℕ) : ℕ :=
  initialAmount - spentAmount

/-- Theorem: Edward's remaining money is $6 -/
theorem edwards_remaining_money :
  remainingMoney 22 16 = 6 := by
  sorry

end edwards_remaining_money_l3900_390044


namespace integral_inequality_l3900_390008

theorem integral_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_incr : Monotone f) (h_f0 : f 0 = 0) : 
  ∫ x in (0)..(1), f x * (deriv f x) ≥ (1/2) * (∫ x in (0)..(1), f x)^2 := by
  sorry

end integral_inequality_l3900_390008


namespace quadratic_equation_integer_root_l3900_390056

theorem quadratic_equation_integer_root (k : ℕ) : 
  (∃ x : ℕ, x^2 - 34*x + 34*k - 1 = 0) → k = 1 := by
  sorry

end quadratic_equation_integer_root_l3900_390056


namespace square_area_error_percentage_l3900_390034

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.01
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 2.01 := by
sorry


end square_area_error_percentage_l3900_390034


namespace total_price_calculation_l3900_390051

theorem total_price_calculation (refrigerator_price washing_machine_price total_price : ℕ) : 
  refrigerator_price = 4275 →
  washing_machine_price = refrigerator_price - 1490 →
  total_price = refrigerator_price + washing_machine_price →
  total_price = 7060 := by
sorry

end total_price_calculation_l3900_390051


namespace billys_dime_piles_l3900_390010

/-- Given Billy's coin arrangement, prove the number of dime piles -/
theorem billys_dime_piles 
  (quarter_piles : ℕ) 
  (coins_per_pile : ℕ) 
  (total_coins : ℕ) 
  (h1 : quarter_piles = 2)
  (h2 : coins_per_pile = 4)
  (h3 : total_coins = 20) :
  (total_coins - quarter_piles * coins_per_pile) / coins_per_pile = 3 := by
sorry

end billys_dime_piles_l3900_390010


namespace tangent_circle_radius_l3900_390009

/-- A square with side length 8 -/
structure Square :=
  (side : ℝ)
  (is_eight : side = 8)

/-- A circle passing through two opposite vertices of a square and tangent to the opposite side -/
structure TangentCircle (s : Square) :=
  (radius : ℝ)
  (passes_through_vertices : True)  -- This is a simplification, as we can't directly represent geometric relations
  (tangent_to_side : True)  -- This is a simplification, as we can't directly represent geometric relations

/-- The radius of the tangent circle is 5 -/
theorem tangent_circle_radius (s : Square) (c : TangentCircle s) : c.radius = 5 := by
  sorry

end tangent_circle_radius_l3900_390009


namespace inequality_proof_l3900_390090

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + 3*c = 9) : 1/a + 1/b + 1/c ≥ 9/2 := by
  sorry

end inequality_proof_l3900_390090


namespace remainder_444_power_444_mod_13_l3900_390007

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l3900_390007


namespace rectangle_perimeter_l3900_390058

/-- Given a large square with side length z and an inscribed smaller square with side length w,
    prove that the perimeter of one of the four identical rectangles formed between the two squares is w + z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (hzw : z > w) : 
  let long_side := w
  let short_side := (z - w) / 2
  2 * long_side + 2 * short_side = w + z := by
  sorry

end rectangle_perimeter_l3900_390058


namespace complex_product_squared_l3900_390078

theorem complex_product_squared (P R S : ℂ) : 
  P = 3 + 4*I ∧ R = 2*I ∧ S = 3 - 4*I → (P * R * S)^2 = -2500 := by
  sorry

end complex_product_squared_l3900_390078


namespace system_solution_l3900_390015

theorem system_solution :
  ∃ (x₁ x₂ x₃ : ℝ),
    (3 * x₁ - 2 * x₂ + x₃ = -10) ∧
    (2 * x₁ + 3 * x₂ - 4 * x₃ = 16) ∧
    (x₁ - 4 * x₂ + 3 * x₃ = -18) ∧
    (x₁ = -1) ∧ (x₂ = 2) ∧ (x₃ = -3) :=
by
  sorry

end system_solution_l3900_390015


namespace root_product_l3900_390054

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  (lg x)^2 + (lg 2 + lg 3) * lg x + lg 2 * lg 3 = 0

-- State the theorem
theorem root_product (x₁ x₂ : ℝ) :
  equation x₁ ∧ equation x₂ ∧ x₁ ≠ x₂ → x₁ * x₂ = 1/6 := by
  sorry

end root_product_l3900_390054


namespace asymptotes_of_hyperbola_l3900_390095

theorem asymptotes_of_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e1 := Real.sqrt (a^2 - b^2) / a
  let e2 := Real.sqrt (a^2 + b^2) / a
  let C1 := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let C2 := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  e1 * e2 = Real.sqrt 15 / 4 →
  (∀ x y, C2 x y → (x + 2*y = 0 ∨ x - 2*y = 0)) :=
by sorry

end asymptotes_of_hyperbola_l3900_390095


namespace range_of_a_l3900_390020

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici (-1 : ℝ), f a x ≥ a) →
  -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 :=
by sorry

end range_of_a_l3900_390020


namespace select_two_each_select_at_least_one_each_select_with_restriction_student_selection_methods_l3900_390073

/-- The number of female students -/
def num_females : ℕ := 5

/-- The number of male students -/
def num_males : ℕ := 4

/-- The number of students to be selected -/
def num_selected : ℕ := 4

/-- Theorem for the number of ways to select 2 males and 2 females -/
theorem select_two_each : ℕ := by sorry

/-- Theorem for the number of ways to select at least 1 male and 1 female -/
theorem select_at_least_one_each : ℕ := by sorry

/-- Theorem for the number of ways to select at least 1 male and 1 female, 
    but not both male A and female B -/
theorem select_with_restriction : ℕ := by sorry

/-- Main theorem combining all selection methods -/
theorem student_selection_methods :
  select_two_each = 1440 ∧
  select_at_least_one_each = 2880 ∧
  select_with_restriction = 2376 := by sorry

end select_two_each_select_at_least_one_each_select_with_restriction_student_selection_methods_l3900_390073


namespace second_polygon_sides_l3900_390046

theorem second_polygon_sides (n₁ n₂ : ℕ) (s₁ s₂ : ℝ) :
  n₁ = 50 →
  s₁ = 3 * s₂ →
  n₁ * s₁ = n₂ * s₂ →
  n₂ = 150 := by
sorry

end second_polygon_sides_l3900_390046


namespace triangle_angle_inequalities_l3900_390024

theorem triangle_angle_inequalities (α β γ : ℝ) 
  (h_triangle : α + β + γ = Real.pi) : 
  ((1 - Real.cos α) * (1 - Real.cos β) * (1 - Real.cos γ) ≥ Real.cos α * Real.cos β * Real.cos γ) ∧
  (12 * Real.cos α * Real.cos β * Real.cos γ ≤ 
   2 * Real.cos α * Real.cos β + 2 * Real.cos α * Real.cos γ + 2 * Real.cos β * Real.cos γ) ∧
  (2 * Real.cos α * Real.cos β + 2 * Real.cos α * Real.cos γ + 2 * Real.cos β * Real.cos γ ≤ 
   Real.cos α + Real.cos β + Real.cos γ) :=
by sorry

end triangle_angle_inequalities_l3900_390024


namespace binomial_coefficient_sum_l3900_390080

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end binomial_coefficient_sum_l3900_390080


namespace unique_solution_system_l3900_390094

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    x^2 - 22*y - 69*z + 703 = 0 ∧
    y^2 + 23*x + 23*z - 1473 = 0 ∧
    z^2 - 63*x + 66*y + 2183 = 0 ∧
    x = 20 ∧ y = -22 ∧ z = 23 := by
  sorry

end unique_solution_system_l3900_390094


namespace pet_store_cages_l3900_390016

theorem pet_store_cages (total_puppies : ℕ) (puppies_per_cage : ℕ) (last_cage_puppies : ℕ) :
  total_puppies = 38 →
  puppies_per_cage = 6 →
  last_cage_puppies = 4 →
  (total_puppies / puppies_per_cage + 1 : ℕ) = 7 :=
by sorry

end pet_store_cages_l3900_390016


namespace y_intercept_for_specific_line_l3900_390050

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -2 and x-intercept (5,0), the y-intercept is (0,10). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -2, x_intercept := (5, 0) }
  y_intercept l = (0, 10) := by
  sorry

end y_intercept_for_specific_line_l3900_390050


namespace georges_expenses_l3900_390074

theorem georges_expenses (B : ℝ) (B_pos : B > 0) : ∃ (m s : ℝ),
  m = 0.25 * (B - s) ∧
  s = 0.05 * (B - m) ∧
  m + s = B :=
by sorry

end georges_expenses_l3900_390074


namespace standard_normal_probability_l3900_390060

/-- Standard normal distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- Probability density function of the standard normal distribution -/
noncomputable def φ : ℝ → ℝ := sorry

theorem standard_normal_probability (X : ℝ → ℝ) : 
  (∀ (a b : ℝ), a < b → ∫ x in a..b, φ x = Φ b - Φ a) →
  Φ 2 - Φ (-1) = 0.8185 := by
  sorry

end standard_normal_probability_l3900_390060


namespace largest_quotient_l3900_390019

def S : Set ℤ := {-36, -6, -4, 3, 7, 9}

def quotient (a b : ℤ) : ℚ := (a : ℚ) / (b : ℚ)

def valid_quotient (q : ℚ) : Prop :=
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ b ≠ 0 ∧ q = quotient a b

theorem largest_quotient :
  ∃ (max_q : ℚ), valid_quotient max_q ∧ 
  (∀ (q : ℚ), valid_quotient q → q ≤ max_q) ∧
  max_q = 9 := by sorry

end largest_quotient_l3900_390019


namespace first_number_value_l3900_390000

theorem first_number_value : ∃ x y : ℤ, 
  (x + 2 * y = 124) ∧ (y = 43) → x = 38 := by
  sorry

end first_number_value_l3900_390000


namespace largest_solution_and_ratio_l3900_390042

theorem largest_solution_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 4 - 2 = 5 / x) → 
  (x = (a + b * Real.sqrt c) / d) → 
  (∀ y : ℝ, (7 * y / 4 - 2 = 5 / y) → y ≤ x) →
  (x = (4 + 2 * Real.sqrt 39) / 7 ∧ a * c * d / b = 546) := by
  sorry

end largest_solution_and_ratio_l3900_390042


namespace largest_integer_square_four_digits_base7_l3900_390075

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 66

/-- Convert a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Count the number of digits in a number's base 7 representation -/
def digitCountBase7 (n : ℕ) : ℕ :=
  (toBase7 n).length

theorem largest_integer_square_four_digits_base7 :
  (M * M ≥ 7^3) ∧
  (M * M < 7^4) ∧
  (digitCountBase7 (M * M) = 4) ∧
  (∀ n : ℕ, n > M → digitCountBase7 (n * n) ≠ 4) := by
  sorry

end largest_integer_square_four_digits_base7_l3900_390075


namespace negation_of_existence_squared_greater_than_two_l3900_390072

theorem negation_of_existence_squared_greater_than_two :
  (¬ ∃ x : ℝ, x^2 > 2) ↔ (∀ x : ℝ, x^2 ≤ 2) := by sorry

end negation_of_existence_squared_greater_than_two_l3900_390072


namespace positive_solution_x_l3900_390029

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 10 - 3 * x - 2 * y)
  (eq2 : y * z = 10 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 2 * z)
  (x_pos : x > 0) :
  x = 8 := by
  sorry

end positive_solution_x_l3900_390029


namespace abs_add_ge_abs_add_x_range_when_eq_possible_values_l3900_390076

-- 1. Triangle inequality for absolute values
theorem abs_add_ge_abs_add (a b : ℚ) : |a| + |b| ≥ |a + b| := by sorry

-- 2. Range of x when |x|+2015=|x-2015|
theorem x_range_when_eq (x : ℝ) (h : |x| + 2015 = |x - 2015|) : x ≤ 0 := by sorry

-- 3. Possible values of a₁+a₂ given conditions
theorem possible_values (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : |a₁ + a₂| + |a₃ + a₄| = 15) 
  (h2 : |a₁ + a₂ + a₃ + a₄| = 5) : 
  (a₁ + a₂ = 10) ∨ (a₁ + a₂ = -10) ∨ (a₁ + a₂ = 5) ∨ (a₁ + a₂ = -5) := by sorry

end abs_add_ge_abs_add_x_range_when_eq_possible_values_l3900_390076


namespace cubic_foot_to_cubic_inch_l3900_390035

theorem cubic_foot_to_cubic_inch :
  (1 : ℝ) * (foot ^ 3) = 1728 * (inch ^ 3) :=
by
  -- Define the relationship between foot and inch
  have foot_to_inch : (1 : ℝ) * foot = 12 * inch := sorry
  
  -- Cube both sides of the equation
  have cubed_equality : ((1 : ℝ) * foot) ^ 3 = (12 * inch) ^ 3 := sorry
  
  -- Simplify the left side
  have left_side : ((1 : ℝ) * foot) ^ 3 = (1 : ℝ) * (foot ^ 3) := sorry
  
  -- Simplify the right side
  have right_side : (12 * inch) ^ 3 = 1728 * (inch ^ 3) := sorry
  
  -- Combine the steps to prove the theorem
  sorry

end cubic_foot_to_cubic_inch_l3900_390035


namespace repeating_decimal_ratio_l3900_390043

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem repeating_decimal_ratio : 
  (RepeatingDecimal 5 4) / (RepeatingDecimal 1 8) = 3 := by
  sorry

end repeating_decimal_ratio_l3900_390043


namespace stratified_sampling_correct_l3900_390017

/-- Represents the categories of teachers -/
inductive TeacherCategory
  | Senior
  | Intermediate
  | Junior

/-- Represents the school's teacher population -/
structure SchoolPopulation where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the selected sample of teachers -/
structure SelectedSample where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Checks if the sample maintains the same proportion as the population -/
def isProportionalSample (pop : SchoolPopulation) (sample : SelectedSample) : Prop :=
  pop.senior * sample.total = sample.senior * pop.total ∧
  pop.intermediate * sample.total = sample.intermediate * pop.total ∧
  pop.junior * sample.total = sample.junior * pop.total

/-- The main theorem stating that the given sample is proportional -/
theorem stratified_sampling_correct 
  (pop : SchoolPopulation)
  (sample : SelectedSample)
  (h1 : pop.total = 150)
  (h2 : pop.senior = 15)
  (h3 : pop.intermediate = 45)
  (h4 : pop.junior = 90)
  (h5 : sample.total = 30)
  (h6 : sample.senior = 3)
  (h7 : sample.intermediate = 9)
  (h8 : sample.junior = 18) :
  isProportionalSample pop sample :=
sorry

end stratified_sampling_correct_l3900_390017


namespace die_roll_probability_l3900_390063

/-- The probability of getting a different outcome on a six-sided die -/
def prob_different : ℚ := 5 / 6

/-- The probability of getting the same outcome on a six-sided die -/
def prob_same : ℚ := 1 / 6

/-- The number of rolls before the consecutive identical rolls -/
def num_rolls : ℕ := 10

theorem die_roll_probability : 
  prob_different ^ num_rolls * prob_same = 5^10 / 6^11 := by
  sorry

end die_roll_probability_l3900_390063


namespace number_of_nieces_l3900_390069

def hand_mitts_price : ℚ := 14
def apron_price : ℚ := 16
def utensils_price : ℚ := 10
def knife_price : ℚ := 2 * utensils_price
def discount_rate : ℚ := 1/4
def total_spending : ℚ := 135

def discounted_price (price : ℚ) : ℚ :=
  price * (1 - discount_rate)

def gift_set_price : ℚ :=
  discounted_price hand_mitts_price +
  discounted_price apron_price +
  discounted_price utensils_price +
  discounted_price knife_price

theorem number_of_nieces :
  total_spending / gift_set_price = 3 := by sorry

end number_of_nieces_l3900_390069


namespace initial_alcohol_percentage_l3900_390014

/-- Proves that the initial alcohol percentage in a mixture is 20% given the conditions --/
theorem initial_alcohol_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_percentage : ℝ) : ℝ :=
  by
  have h1 : initial_volume = 18 := by sorry
  have h2 : added_water = 3 := by sorry
  have h3 : final_percentage = 17.14285714285715 := by sorry
  
  let final_volume : ℝ := initial_volume + added_water
  let initial_percentage : ℝ := (final_percentage * final_volume) / initial_volume
  
  have h4 : initial_percentage = 20 := by sorry
  
  exact initial_percentage

end initial_alcohol_percentage_l3900_390014


namespace frog_arrangement_count_l3900_390077

/-- Represents the number of valid arrangements of frogs -/
def validFrogArrangements (n g r b : ℕ) : ℕ :=
  if n = g + r + b ∧ g ≥ 1 ∧ r ≥ 1 ∧ b = 1 then
    2 * (Nat.factorial r * Nat.factorial g)
  else
    0

/-- Theorem stating the number of valid frog arrangements for the given problem -/
theorem frog_arrangement_count :
  validFrogArrangements 8 3 4 1 = 288 := by
  sorry

end frog_arrangement_count_l3900_390077


namespace football_player_goals_l3900_390098

theorem football_player_goals (average_increase : ℝ) (fifth_match_goals : ℕ) : 
  average_increase = 0.3 →
  fifth_match_goals = 2 →
  ∃ (initial_average : ℝ),
    initial_average * 4 + fifth_match_goals = (initial_average + average_increase) * 5 ∧
    (initial_average + average_increase) * 5 = 4 := by
  sorry

end football_player_goals_l3900_390098


namespace ellipse_equal_angles_l3900_390081

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a chord of the ellipse -/
structure Chord where
  A : Point
  B : Point

/-- Checks if a chord passes through a given point -/
def chordThroughPoint (c : Chord) (p : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = c.A.x + t * (c.B.x - c.A.x) ∧
    p.y = c.A.y + t * (c.B.y - c.A.y)

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ :=
  sorry -- Angle calculation implementation

/-- The main theorem to prove -/
theorem ellipse_equal_angles (e : Ellipse) (F P : Point) :
  e.a = 2 ∧ e.b = 1 ∧
  F.x = Real.sqrt 3 ∧ F.y = 0 ∧
  P.x = 2 ∧ P.y = 0 →
  ∀ (c : Chord), onEllipse e c.A ∧ onEllipse e c.B ∧ chordThroughPoint c F →
    angle c.A P F = angle c.B P F :=
  sorry

end ellipse_equal_angles_l3900_390081


namespace geometric_subseq_implies_arithmetic_indices_l3900_390085

/-- Given a geometric sequence with common ratio q ≠ 1, if three terms form a geometric sequence,
    then their indices form an arithmetic sequence. -/
theorem geometric_subseq_implies_arithmetic_indices
  (a : ℕ → ℝ) (q : ℝ) (m n p : ℕ) (hq : q ≠ 1)
  (h_geom : ∀ k, a (k + 1) = q * a k)
  (h_subseq : (a n)^2 = a m * a p) :
  2 * n = m + p :=
sorry

end geometric_subseq_implies_arithmetic_indices_l3900_390085


namespace distance_between_sasha_and_kolya_l3900_390004

-- Define the race distance
def race_distance : ℝ := 100

-- Define the runners' speeds
variable (v_S v_L v_K : ℝ)

-- Define the conditions
axiom positive_speeds : 0 < v_S ∧ 0 < v_L ∧ 0 < v_K
axiom lyosha_behind_sasha : v_L / v_S = 0.9
axiom kolya_behind_lyosha : v_K / v_L = 0.9

-- Define the theorem
theorem distance_between_sasha_and_kolya :
  let t_S := race_distance / v_S
  let d_K := v_K * t_S
  race_distance - d_K = 19 := by sorry

end distance_between_sasha_and_kolya_l3900_390004


namespace rectangles_in_5x4_grid_l3900_390022

/-- The number of different rectangles in a rectangular grid --/
def num_rectangles (rows : ℕ) (cols : ℕ) : ℕ :=
  (rows.choose 2) * (cols.choose 2)

/-- Theorem: In a 5x4 grid, the number of different rectangles is 60 --/
theorem rectangles_in_5x4_grid :
  num_rectangles 5 4 = 60 := by
  sorry

end rectangles_in_5x4_grid_l3900_390022


namespace expression_simplification_l3900_390089

theorem expression_simplification (m n : ℝ) 
  (h : Real.sqrt (m - 1/2) + (n + 2)^2 = 0) : 
  ((3*m + n) * (m + n) - (2*m - n)^2 + (m + 2*n) * (m - 2*n)) / (2*n) = 6 := by
  sorry

end expression_simplification_l3900_390089


namespace geometric_mean_max_l3900_390071

theorem geometric_mean_max (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_arithmetic_mean : (a + b) / 2 = 4) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) / 2 = 4 ∧ 
  Real.sqrt (x * y) = 4 ∧ 
  ∀ (c d : ℝ), c > 0 → d > 0 → (c + d) / 2 = 4 → Real.sqrt (c * d) ≤ 4 := by
  sorry

end geometric_mean_max_l3900_390071


namespace garden_area_l3900_390062

/-- The total area of two triangles with given bases and a shared altitude -/
theorem garden_area (base1 base2 : ℝ) (area1 : ℝ) (h : ℝ) : 
  base1 = 50 →
  base2 = 40 →
  area1 = 800 →
  area1 = (1/2) * base1 * h →
  (1/2) * base1 * h + (1/2) * base2 * h = 1440 := by
  sorry

#check garden_area

end garden_area_l3900_390062


namespace new_customers_count_l3900_390091

-- Define the given conditions
def initial_customers : ℕ := 13
def final_customers : ℕ := 9
def total_left : ℕ := 8

-- Theorem to prove
theorem new_customers_count :
  ∃ (new_customers : ℕ),
    new_customers = total_left + (final_customers - (initial_customers - total_left)) :=
by
  sorry

end new_customers_count_l3900_390091


namespace calculation_proof_l3900_390033

theorem calculation_proof : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := by
  sorry

end calculation_proof_l3900_390033


namespace arcsin_equation_solution_l3900_390012

theorem arcsin_equation_solution : 
  Real.arcsin (Real.sqrt (2/51)) + Real.arcsin (3 * Real.sqrt (2/51)) = π/4 := by
  sorry

end arcsin_equation_solution_l3900_390012


namespace ashas_borrowed_amount_l3900_390047

theorem ashas_borrowed_amount (brother mother granny savings spent_fraction remaining : ℚ)
  (h1 : brother = 20)
  (h2 : mother = 30)
  (h3 : granny = 70)
  (h4 : savings = 100)
  (h5 : spent_fraction = 3/4)
  (h6 : remaining = 65)
  (h7 : (1 - spent_fraction) * (brother + mother + granny + savings + father) = remaining) :
  father = 40 := by
  sorry

end ashas_borrowed_amount_l3900_390047


namespace smallest_x_satisfying_equation_l3900_390099

theorem smallest_x_satisfying_equation : 
  (∃ x : ℝ, x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ 
  (∀ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 8 → y ≥ 89/9) := by
  sorry

end smallest_x_satisfying_equation_l3900_390099


namespace smallest_integer_solution_l3900_390025

theorem smallest_integer_solution (x : ℤ) : 
  (x^4 - 40*x^2 + 324 = 0) → x ≥ -4 :=
by sorry

end smallest_integer_solution_l3900_390025


namespace kilmer_park_tree_height_l3900_390067

/-- Calculates the height of a tree in inches after a given number of years -/
def tree_height_in_inches (initial_height : ℕ) (growth_rate : ℕ) (years : ℕ) (inches_per_foot : ℕ) : ℕ :=
  (initial_height + growth_rate * years) * inches_per_foot

/-- Proves that the height of the tree in Kilmer Park after 8 years is 1104 inches -/
theorem kilmer_park_tree_height : tree_height_in_inches 52 5 8 12 = 1104 := by
  sorry

end kilmer_park_tree_height_l3900_390067


namespace sqrt_three_irrational_other_numbers_rational_sqrt_three_unique_irrational_l3900_390057

theorem sqrt_three_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)) :=
by sorry

theorem other_numbers_rational :
  ∃ (a b c d e f : ℤ), 
    b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧
    (-32 : ℚ) / 7 = (a : ℚ) / (b : ℚ) ∧
    (0 : ℚ) = (c : ℚ) / (d : ℚ) ∧
    (3.5 : ℚ) = (e : ℚ) / (f : ℚ) :=
by sorry

theorem sqrt_three_unique_irrational 
  (h1 : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)))
  (h2 : ∃ (a b c d e f : ℤ), 
    b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧
    (-32 : ℚ) / 7 = (a : ℚ) / (b : ℚ) ∧
    (0 : ℚ) = (c : ℚ) / (d : ℚ) ∧
    (3.5 : ℚ) = (e : ℚ) / (f : ℚ)) :
  Real.sqrt 3 = Real.sqrt 3 :=
by sorry

end sqrt_three_irrational_other_numbers_rational_sqrt_three_unique_irrational_l3900_390057


namespace cube_root_sum_equals_three_l3900_390092

theorem cube_root_sum_equals_three :
  ∃ (a b : ℝ), 
    a^3 = 9 + 4 * Real.sqrt 5 ∧ 
    b^3 = 9 - 4 * Real.sqrt 5 ∧ 
    (∃ (k : ℤ), a + b = k) → 
    a + b = 3 := by
  sorry

end cube_root_sum_equals_three_l3900_390092


namespace matt_card_trade_profit_l3900_390086

/-- Represents the profit made from trading cards -/
def card_trade_profit (cards_traded : ℕ) (value_per_traded_card : ℕ) 
                      (received_cards_1 : ℕ) (value_per_received_card_1 : ℕ)
                      (received_cards_2 : ℕ) (value_per_received_card_2 : ℕ) : ℤ :=
  (received_cards_1 * value_per_received_card_1 + received_cards_2 * value_per_received_card_2) -
  (cards_traded * value_per_traded_card)

/-- The profit Matt makes from trading two $6 cards for three $2 cards and one $9 card is $3 -/
theorem matt_card_trade_profit : card_trade_profit 2 6 3 2 1 9 = 3 := by
  sorry

end matt_card_trade_profit_l3900_390086


namespace ab_length_is_twelve_l3900_390064

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define the theorem
theorem ab_length_is_twelve
  (ABC : Triangle) (CBD : Triangle)
  (h1 : isIsosceles ABC)
  (h2 : isIsosceles CBD)
  (h3 : perimeter CBD = 24)
  (h4 : perimeter ABC = 26)
  (h5 : CBD.c = 10) :
  ABC.c = 12 := by
  sorry

end ab_length_is_twelve_l3900_390064


namespace valentines_remaining_l3900_390055

/-- Given that Mrs. Wong initially had 30 Valentines and gave 8 away,
    prove that she has 22 Valentines left. -/
theorem valentines_remaining (initial : Nat) (given_away : Nat) :
  initial = 30 → given_away = 8 → initial - given_away = 22 := by
  sorry

end valentines_remaining_l3900_390055


namespace rectangle_area_l3900_390018

theorem rectangle_area (a b : ℝ) (h : a^2 + b^2 - 8*a - 6*b + 25 = 0) : a * b = 12 := by
  sorry

end rectangle_area_l3900_390018


namespace moon_arrangements_count_l3900_390048

/-- The number of letters in the word "MOON" -/
def word_length : ℕ := 4

/-- The number of repeated letters (O's) in the word "MOON" -/
def repeated_letters : ℕ := 2

/-- The number of unique arrangements of the letters in "MOON" -/
def moon_arrangements : ℕ := Nat.factorial word_length / Nat.factorial repeated_letters

theorem moon_arrangements_count : moon_arrangements = 12 := by
  sorry

end moon_arrangements_count_l3900_390048


namespace egyptian_fraction_identity_l3900_390084

theorem egyptian_fraction_identity (n : ℕ+) :
  (2 : ℚ) / (2 * n + 1) = 1 / (n + 1) + 1 / ((n + 1) * (2 * n + 1)) := by
  sorry

end egyptian_fraction_identity_l3900_390084


namespace scientific_notation_of_1230000_l3900_390053

theorem scientific_notation_of_1230000 :
  (1230000 : ℝ) = 1.23 * (10 : ℝ) ^ 6 := by
  sorry

end scientific_notation_of_1230000_l3900_390053


namespace tensor_range_theorem_l3900_390038

/-- Custom operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem stating the range of k given the condition -/
theorem tensor_range_theorem (k : ℝ) :
  (∀ x : ℝ, tensor k x > 0) → k ∈ Set.Ioo 0 4 := by
  sorry

end tensor_range_theorem_l3900_390038


namespace du_chin_pies_l3900_390021

/-- The number of meat pies Du Chin bakes in a day -/
def num_pies : ℕ := 200

/-- The price of each meat pie in dollars -/
def price_per_pie : ℕ := 20

/-- The fraction of sales used to buy ingredients for the next day -/
def ingredient_fraction : ℚ := 3/5

/-- The amount remaining after setting aside money for ingredients -/
def remaining_amount : ℕ := 1600

/-- Theorem stating that the number of pies baked satisfies the given conditions -/
theorem du_chin_pies :
  (num_pies * price_per_pie : ℚ) * (1 - ingredient_fraction) = remaining_amount := by
  sorry

end du_chin_pies_l3900_390021


namespace angle_pairs_same_terminal_side_l3900_390049

def same_terminal_side (a b : Int) : Prop :=
  ∃ k : Int, b - a = k * 360

theorem angle_pairs_same_terminal_side :
  ¬ same_terminal_side 390 690 ∧
  same_terminal_side (-330) 750 ∧
  ¬ same_terminal_side 480 (-420) ∧
  ¬ same_terminal_side 3000 (-840) :=
by sorry

end angle_pairs_same_terminal_side_l3900_390049
