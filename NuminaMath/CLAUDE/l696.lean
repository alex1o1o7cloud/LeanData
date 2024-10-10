import Mathlib

namespace sausage_pepperoni_difference_l696_69675

def pizza_problem (pepperoni ham sausage : ℕ) : Prop :=
  let total_slices : ℕ := 6
  let meat_per_slice : ℕ := 22
  pepperoni = 30 ∧
  ham = 2 * pepperoni ∧
  sausage > pepperoni ∧
  (pepperoni + ham + sausage) / total_slices = meat_per_slice

theorem sausage_pepperoni_difference :
  ∀ (pepperoni ham sausage : ℕ),
    pizza_problem pepperoni ham sausage →
    sausage - pepperoni = 12 :=
by
  sorry

end sausage_pepperoni_difference_l696_69675


namespace employee_count_l696_69672

theorem employee_count (avg_salary : ℝ) (new_avg_salary : ℝ) (manager_salary : ℝ) : 
  avg_salary = 1500 →
  new_avg_salary = 2500 →
  manager_salary = 22500 →
  ∃ (E : ℕ), (E : ℝ) * avg_salary + manager_salary = new_avg_salary * ((E : ℝ) + 1) ∧ E = 20 :=
by
  sorry

end employee_count_l696_69672


namespace carpet_width_l696_69653

/-- Proves that a rectangular carpet covering 30% of a 120 square feet floor with a length of 9 feet has a width of 4 feet. -/
theorem carpet_width (floor_area : ℝ) (carpet_coverage : ℝ) (carpet_length : ℝ) :
  floor_area = 120 →
  carpet_coverage = 0.3 →
  carpet_length = 9 →
  (floor_area * carpet_coverage) / carpet_length = 4 := by
  sorry

end carpet_width_l696_69653


namespace parabola_no_real_roots_l696_69606

def parabola (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem parabola_no_real_roots :
  ∀ x : ℝ, parabola x ≠ 0 := by
sorry

end parabola_no_real_roots_l696_69606


namespace abc_inequality_l696_69668

theorem abc_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end abc_inequality_l696_69668


namespace line_tangent_to_circle_l696_69688

/-- A line that bisects a circle passes through its center -/
axiom line_bisects_circle_passes_through_center 
  (a b c d : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = c^2 → y = d*x + (b - d*a)) → 
  b = d*a + c^2/(2*d)

/-- The equation of a circle -/
def is_on_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- The equation of a line -/
def is_on_line (x y m c : ℝ) : Prop :=
  y = m*x + c

theorem line_tangent_to_circle 
  (a b r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, is_on_circle x y 1 2 2 ↔ is_on_circle x y a b r) →
  (∀ x y : ℝ, is_on_line x y 1 1 → is_on_circle x y a b r) →
  ∀ y : ℝ, is_on_circle 3 y a b r ↔ y = 2 :=
sorry

end line_tangent_to_circle_l696_69688


namespace inequality_transformation_l696_69690

theorem inequality_transformation (a b : ℝ) : a ≤ b → -a/2 ≥ -b/2 := by
  sorry

end inequality_transformation_l696_69690


namespace largest_integer_satisfying_inequality_l696_69661

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 3 ↔ 3 * x + 4 < 5 * x - 2 :=
by sorry

end largest_integer_satisfying_inequality_l696_69661


namespace manolo_face_masks_l696_69666

/-- Represents the number of face-masks Manolo can make in a given time period -/
def face_masks (first_hour_rate : ℚ) (subsequent_rate : ℚ) (hours : ℚ) : ℚ :=
  let first_hour := min 1 hours
  let remaining_hours := max 0 (hours - 1)
  (60 / first_hour_rate) * first_hour + (60 / subsequent_rate) * remaining_hours

/-- Theorem stating that Manolo makes 45 face-masks in a four-hour shift -/
theorem manolo_face_masks :
  face_masks (4 : ℚ) (6 : ℚ) (4 : ℚ) = 45 := by
  sorry

#eval face_masks (4 : ℚ) (6 : ℚ) (4 : ℚ)

end manolo_face_masks_l696_69666


namespace smallest_terminating_with_two_l696_69647

/-- A function that checks if a positive integer contains the digit 2 -/
def containsDigitTwo (n : ℕ+) : Prop := sorry

/-- A function that checks if the reciprocal of a positive integer is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := sorry

/-- Theorem stating that 2 is the smallest positive integer n such that 1/n is a terminating decimal and n contains the digit 2 -/
theorem smallest_terminating_with_two :
  (∀ m : ℕ+, m < 2 → ¬(isTerminatingDecimal m ∧ containsDigitTwo m)) ∧
  (isTerminatingDecimal 2 ∧ containsDigitTwo 2) :=
sorry

end smallest_terminating_with_two_l696_69647


namespace normal_price_after_discounts_l696_69643

theorem normal_price_after_discounts (price : ℝ) : 
  price * (1 - 0.1) * (1 - 0.2) = 144 → price = 200 := by
  sorry

end normal_price_after_discounts_l696_69643


namespace min_value_on_circle_l696_69665

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (min : ℝ), (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 6*y' + 12 = 0 →
    |2*x' - y' - 2| ≥ min) ∧ min = 5 - Real.sqrt 5 := by
  sorry

end min_value_on_circle_l696_69665


namespace negation_of_existence_l696_69603

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2*a*x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2*a*x - 1 ≥ 0) := by
  sorry

end negation_of_existence_l696_69603


namespace candy_sales_l696_69664

theorem candy_sales (initial_candy : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) :
  initial_candy = 80 →
  sold_monday = 15 →
  remaining_wednesday = 7 →
  initial_candy - sold_monday - remaining_wednesday = 58 := by
  sorry

end candy_sales_l696_69664


namespace linear_system_solution_l696_69630

theorem linear_system_solution (x y : ℝ) 
  (eq1 : x + 4*y = 5) 
  (eq2 : 5*x + 6*y = 7) : 
  3*x + 5*y = 6 := by
sorry

end linear_system_solution_l696_69630


namespace largest_common_divisor_525_385_l696_69646

theorem largest_common_divisor_525_385 : Nat.gcd 525 385 = 35 := by
  sorry

end largest_common_divisor_525_385_l696_69646


namespace paul_filled_six_bags_saturday_l696_69683

/-- The number of bags Paul filled on Saturday -/
def bags_saturday : ℕ := sorry

/-- The number of bags Paul filled on Sunday -/
def bags_sunday : ℕ := 3

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 8

/-- The total number of cans collected -/
def total_cans : ℕ := 72

/-- Theorem stating that Paul filled 6 bags on Saturday -/
theorem paul_filled_six_bags_saturday : 
  bags_saturday = 6 := by sorry

end paul_filled_six_bags_saturday_l696_69683


namespace largest_three_digit_number_with_gcd_condition_l696_69626

theorem largest_three_digit_number_with_gcd_condition :
  ∃ (x : ℕ), 
    x ≤ 990 ∧ 
    100 ≤ x ∧ 
    x % 3 = 0 ∧
    Nat.gcd 15 (Nat.gcd x 20) = 5 ∧
    ∀ (y : ℕ), 
      100 ≤ y ∧ 
      y ≤ 999 ∧ 
      y % 3 = 0 ∧ 
      Nat.gcd 15 (Nat.gcd y 20) = 5 → 
      y ≤ x :=
by sorry

end largest_three_digit_number_with_gcd_condition_l696_69626


namespace unique_solution_for_equation_l696_69670

theorem unique_solution_for_equation (y : ℝ) : y + 49 / y = 14 ↔ y = 7 := by sorry

end unique_solution_for_equation_l696_69670


namespace qr_length_l696_69693

-- Define a right triangle
structure RightTriangle where
  QP : ℝ
  QR : ℝ
  cosQ : ℝ
  right_angle : cosQ = QP / QR

-- Theorem statement
theorem qr_length (t : RightTriangle) (h1 : t.cosQ = 0.5) (h2 : t.QP = 10) : t.QR = 20 := by
  sorry

end qr_length_l696_69693


namespace factorial_sum_equality_l696_69637

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 6 = 5040 := by
  sorry

end factorial_sum_equality_l696_69637


namespace f_four_times_one_l696_69631

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_four_times_one : f (f (f (f 1))) = 4 := by
  sorry

end f_four_times_one_l696_69631


namespace raquel_has_40_dollars_l696_69673

-- Define the amounts of money for each person
def raquel_money : ℝ := sorry
def nataly_money : ℝ := sorry
def tom_money : ℝ := sorry

-- State the theorem
theorem raquel_has_40_dollars :
  -- Conditions
  (tom_money = (1/4) * nataly_money) →
  (nataly_money = 3 * raquel_money) →
  (tom_money + nataly_money + raquel_money = 190) →
  -- Conclusion
  raquel_money = 40 := by
  sorry

end raquel_has_40_dollars_l696_69673


namespace distance_right_focus_to_line_l696_69620

/-- The distance from the right focus of the hyperbola x²/4 - y²/5 = 1 to the line x + 2y - 8 = 0 is √5 -/
theorem distance_right_focus_to_line : ∃ (d : ℝ), d = Real.sqrt 5 ∧ 
  ∀ (x y : ℝ), 
    (x^2 / 4 - y^2 / 5 = 1) →  -- Hyperbola equation
    (x + 2*y - 8 = 0) →       -- Line equation
    d = Real.sqrt ((x - 3)^2 + y^2) := by
  sorry

end distance_right_focus_to_line_l696_69620


namespace polynomial_equality_l696_69677

def polynomial (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) (x : ℝ) : ℝ :=
  a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8

theorem polynomial_equality 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x - 3)^3 * (2*x + 1)^5 = polynomial a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ x) →
  (a₀ = -27 ∧ a₀ + a₂ + a₄ + a₆ + a₈ = -940) :=
by sorry

end polynomial_equality_l696_69677


namespace gdp_scientific_notation_equality_l696_69645

/-- Represents the gross domestic product in billions of yuan -/
def gdp : ℝ := 2502.7

/-- The scientific notation representation of the GDP -/
def scientific_notation : ℝ := 2.5027 * (10 ^ 11)

/-- Theorem stating that the GDP in billions of yuan is equal to its scientific notation representation -/
theorem gdp_scientific_notation_equality : gdp * 10^9 = scientific_notation := by
  sorry

end gdp_scientific_notation_equality_l696_69645


namespace classroom_contribution_prove_classroom_contribution_l696_69618

/-- Proves that the amount contributed by each of the eight families is $10 --/
theorem classroom_contribution : ℝ → Prop :=
  fun x =>
    let goal : ℝ := 200
    let raised_from_two : ℝ := 2 * 20
    let raised_from_ten : ℝ := 10 * 5
    let raised_from_eight : ℝ := 8 * x
    let total_raised : ℝ := raised_from_two + raised_from_ten + raised_from_eight
    let remaining : ℝ := 30
    total_raised + remaining = goal → x = 10

/-- Proof of the classroom_contribution theorem --/
theorem prove_classroom_contribution : classroom_contribution 10 := by
  sorry

end classroom_contribution_prove_classroom_contribution_l696_69618


namespace M_intersect_N_l696_69674

def M : Set ℕ := {1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem M_intersect_N : M ∩ N = {1} := by
  sorry

end M_intersect_N_l696_69674


namespace classroom_gpa_proof_l696_69651

/-- Proves that the grade point average of one third of a classroom is 30,
    given the grade point average of two thirds is 33 and the overall average is 32. -/
theorem classroom_gpa_proof (gpa_two_thirds : ℝ) (gpa_overall : ℝ) : ℝ :=
  let gpa_one_third : ℝ := 30
  by
    have h1 : gpa_two_thirds = 33 := by sorry
    have h2 : gpa_overall = 32 := by sorry
    have h3 : (1/3 : ℝ) * gpa_one_third + (2/3 : ℝ) * gpa_two_thirds = gpa_overall := by sorry
    sorry

end classroom_gpa_proof_l696_69651


namespace special_card_survives_l696_69667

/-- Represents a deck of cards with a specific card at a given position -/
structure Deck :=
  (size : Nat)
  (special_card_pos : Nat)

/-- Represents a removal operation on the deck -/
inductive Removal
  | left : Nat → Removal
  | right : Nat → Removal

/-- Checks if a card at the given position survives a removal operation -/
def survives (d : Deck) (r : Removal) : Bool :=
  match r with
  | Removal.left n => d.special_card_pos > n
  | Removal.right n => d.size - d.special_card_pos ≥ n

/-- Theorem stating that a card in position 26 or 27 of a 52-card deck can always survive 51 removals -/
theorem special_card_survives (initial_pos : Nat) 
    (h : initial_pos = 26 ∨ initial_pos = 27) : 
    ∀ (removals : List Removal), 
      removals.length = 51 → 
      ∃ (final_deck : Deck), 
        final_deck.size = 1 ∧ 
        final_deck.special_card_pos = 1 :=
  sorry

#check special_card_survives

end special_card_survives_l696_69667


namespace water_level_lowered_l696_69604

/-- Proves that removing 4500 gallons of water from a 60ft by 20ft pool lowers the water level by 6 inches -/
theorem water_level_lowered (pool_length pool_width : ℝ) 
  (water_removed : ℝ) (conversion_factor : ℝ) :
  pool_length = 60 →
  pool_width = 20 →
  water_removed = 4500 →
  conversion_factor = 7.5 →
  (water_removed / conversion_factor) / (pool_length * pool_width) * 12 = 6 := by
  sorry

end water_level_lowered_l696_69604


namespace simplify_expression_l696_69615

theorem simplify_expression (a : ℝ) (h : a ≠ -1) :
  a - 1 + 1 / (a + 1) = a^2 / (a + 1) := by
  sorry

end simplify_expression_l696_69615


namespace consecutive_integers_divisibility_l696_69669

theorem consecutive_integers_divisibility (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ n : ℕ, ∃ x y z : ℕ,
    x ∈ Finset.range (2 * c) ∧
    y ∈ Finset.range (2 * c) ∧
    z ∈ Finset.range (2 * c) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (a * b * c) ∣ (x * y * z) :=
by sorry

end consecutive_integers_divisibility_l696_69669


namespace intersection_M_N_l696_69691

def M : Set ℝ := {x | x^2 - x ≥ 0}
def N : Set ℝ := {x | x < 2}

theorem intersection_M_N : 
  ∀ x : ℝ, x ∈ M ∩ N ↔ x ≤ 0 ∨ (1 ≤ x ∧ x < 2) := by
  sorry

end intersection_M_N_l696_69691


namespace arithmetic_expression_equality_l696_69663

theorem arithmetic_expression_equality : 10 - 9 + 8 * (7 - 6) + 5 * 4 - 3 + 2 - 1 = 25 := by
  sorry

end arithmetic_expression_equality_l696_69663


namespace carlos_singles_percentage_l696_69681

/-- Represents the hit statistics for Carlos during the baseball season -/
structure HitStats :=
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (strikeouts : ℕ)

/-- Calculates the percentage of singles among successful hits -/
def percentage_singles (stats : HitStats) : ℚ :=
  let successful_hits := stats.total_hits - stats.strikeouts
  let non_single_hits := stats.home_runs + stats.triples + stats.doubles
  let singles := successful_hits - non_single_hits
  (singles : ℚ) / (successful_hits : ℚ) * 100

/-- The hit statistics for Carlos -/
def carlos_stats : HitStats :=
  { total_hits := 50
  , home_runs := 4
  , triples := 2
  , doubles := 8
  , strikeouts := 6 }

/-- Theorem stating that the percentage of singles for Carlos is approximately 68.18% -/
theorem carlos_singles_percentage :
  abs (percentage_singles carlos_stats - 68.18) < 0.01 := by
  sorry

end carlos_singles_percentage_l696_69681


namespace total_additions_in_half_hour_l696_69611

/-- The number of additions a single computer can perform per second -/
def additions_per_second : ℕ := 15000

/-- The number of computers -/
def num_computers : ℕ := 3

/-- The number of seconds in half an hour -/
def seconds_in_half_hour : ℕ := 1800

/-- The total number of additions performed by all computers in half an hour -/
def total_additions : ℕ := additions_per_second * num_computers * seconds_in_half_hour

theorem total_additions_in_half_hour :
  total_additions = 81000000 := by sorry

end total_additions_in_half_hour_l696_69611


namespace parametric_to_standard_equation_l696_69682

/-- Given parametric equations x = √t, y = 2√(1-t), prove they are equivalent to x² + y²/4 = 1, where 0 ≤ x ≤ 1 and 0 ≤ y ≤ 2 -/
theorem parametric_to_standard_equation (t : ℝ) (x y : ℝ) 
    (hx : x = Real.sqrt t) (hy : y = 2 * Real.sqrt (1 - t)) :
    x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2 := by
  sorry

end parametric_to_standard_equation_l696_69682


namespace trig_simplification_l696_69694

theorem trig_simplification (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  (4 * (Real.cos x)^2 * Real.sin x) / (1 + 2 * Real.cos x * Real.cos (2 * x)) := by
  sorry

end trig_simplification_l696_69694


namespace cos_alpha_value_l696_69654

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = 1 / 3) : 
  Real.cos α = 7 / 9 := by
  sorry

end cos_alpha_value_l696_69654


namespace equation_solution_l696_69628

theorem equation_solution :
  ∃! x : ℝ, (3 / (x - 3) = 4 / (x - 4)) ∧ x = 0 :=
by sorry

end equation_solution_l696_69628


namespace basketball_team_selection_l696_69610

/-- The number of players in the basketball team -/
def total_players : ℕ := 16

/-- The number of players to be chosen for a game -/
def team_size : ℕ := 7

/-- The number of players excluding the twins -/
def players_without_twins : ℕ := total_players - 2

/-- The number of ways to choose the team with the given conditions -/
def ways_to_choose_team : ℕ := Nat.choose players_without_twins team_size + Nat.choose players_without_twins (team_size - 2)

theorem basketball_team_selection :
  ways_to_choose_team = 5434 := by sorry

end basketball_team_selection_l696_69610


namespace expression_simplification_l696_69625

theorem expression_simplification (a b : ℝ) : 
  32 * a^2 * b^2 * (a^2 + b^2)^2 + (a^2 - b^2)^4 + 
  8 * a * b * (a^2 + b^2) * Real.sqrt (16 * a^2 * b^2 * (a^2 + b^2)^2 + (a^2 - b^2)^4) = 
  (a + b)^8 := by sorry

end expression_simplification_l696_69625


namespace tan_sum_17_28_l696_69685

theorem tan_sum_17_28 : 
  (Real.tan (17 * π / 180) + Real.tan (28 * π / 180)) / 
  (1 - Real.tan (17 * π / 180) * Real.tan (28 * π / 180)) = 1 :=
by sorry

end tan_sum_17_28_l696_69685


namespace sum_factorials_mod_12_l696_69602

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_mod_12 :
  sum_factorials 7 % 12 = (factorial 1 + factorial 2 + factorial 3) % 12 :=
sorry

end sum_factorials_mod_12_l696_69602


namespace inequality_proof_l696_69684

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end inequality_proof_l696_69684


namespace third_sum_third_term_ratio_l696_69680

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  sum : ℕ → ℝ
  first_third_sum : a 1 + a 3 = 5/2
  second_fourth_sum : a 2 + a 4 = 5/4
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating that S₃/a₃ = 6 for the given arithmetic progression -/
theorem third_sum_third_term_ratio (ap : ArithmeticProgression) :
  ap.sum 3 / ap.a 3 = 6 := by
  sorry

end third_sum_third_term_ratio_l696_69680


namespace inequality_proof_l696_69641

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1/3)) := by
  sorry

end inequality_proof_l696_69641


namespace train_distance_l696_69660

/-- The distance between two trains after 30 seconds -/
theorem train_distance (speed1 speed2 : ℝ) (time : ℝ) : 
  speed1 = 36 →
  speed2 = 48 →
  time = 30 / 3600 →
  let d1 := speed1 * time * 1000
  let d2 := speed2 * time * 1000
  Real.sqrt (d1^2 + d2^2) = 500 := by
  sorry

end train_distance_l696_69660


namespace quadratic_through_point_l696_69644

/-- Prove that for a quadratic function y = ax² passing through the point (-1, 4), the value of a is 4. -/
theorem quadratic_through_point (a : ℝ) : (∀ x : ℝ, (a * x^2) = 4) ↔ a = 4 := by
  sorry

end quadratic_through_point_l696_69644


namespace triangle_problem_l696_69619

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_problem (ABC : Triangle) 
  (h1 : ABC.a * Real.sin ABC.A + ABC.c * Real.sin ABC.C = Real.sqrt 2 * ABC.a * Real.sin ABC.C + ABC.b * Real.sin ABC.B)
  (h2 : ABC.A = 5 * Real.pi / 12) :
  ABC.B = Real.pi / 4 ∧ ABC.a = 1 + Real.sqrt 3 ∧ ABC.c = Real.sqrt 6 := by
  sorry

end triangle_problem_l696_69619


namespace equation_solutions_l696_69636

def equation (x : ℝ) : Prop :=
  6 / (Real.sqrt (x - 8) - 9) + 1 / (Real.sqrt (x - 8) - 4) + 
  7 / (Real.sqrt (x - 8) + 4) + 12 / (Real.sqrt (x - 8) + 9) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {17, 44} := by sorry

end equation_solutions_l696_69636


namespace nedy_crackers_l696_69656

/-- The number of cracker packs Nedy ate from Monday to Thursday -/
def monday_to_thursday : ℕ := 8

/-- The number of cracker packs Nedy ate on Friday -/
def friday : ℕ := 2 * monday_to_thursday

/-- The total number of cracker packs Nedy ate from Monday to Friday -/
def total : ℕ := monday_to_thursday + friday

theorem nedy_crackers : total = 24 := by sorry

end nedy_crackers_l696_69656


namespace quadrilateral_area_is_5_sqrt_2_l696_69689

/-- A rectangular prism with dimensions length, width, and height -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A quadrilateral formed by four points in 3D space -/
structure Quadrilateral3D where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The area of a quadrilateral formed by the intersection of a plane with a rectangular prism -/
def quadrilateral_area (prism : RectangularPrism) (quad : Quadrilateral3D) : ℝ := sorry

/-- The main theorem stating the area of the quadrilateral ABCD -/
theorem quadrilateral_area_is_5_sqrt_2 (prism : RectangularPrism) (quad : Quadrilateral3D) :
  prism.length = 2 ∧ prism.width = 1 ∧ prism.height = 3 →
  quad.A = ⟨0, 0, 0⟩ ∧ quad.C = ⟨2, 1, 3⟩ →
  quad.B.x = 1 ∧ quad.B.y = 1 ∧ quad.B.z = 0 →
  quad.D.x = 1 ∧ quad.D.y = 0 ∧ quad.D.z = 3 →
  quadrilateral_area prism quad = 5 * Real.sqrt 2 := by
  sorry

end quadrilateral_area_is_5_sqrt_2_l696_69689


namespace ellipse_line_intersection_l696_69640

/-- Given an ellipse C and a line l, prove that under certain conditions, 
    a point derived from l lies on a specific circle. -/
theorem ellipse_line_intersection (a b : ℝ) (k m : ℝ) : 
  a > b ∧ b > 0 ∧ 
  (a^2 - b^2) / a^2 = 3 / 4 ∧
  b = 1 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧
    x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    (y₁ * x₂) / (x₁ * y₂) = 5 / 4) →
  m^2 + k^2 = 5 / 4 := by
sorry

end ellipse_line_intersection_l696_69640


namespace fixed_point_on_line_l696_69616

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * (7/2 : ℝ) - (m + 3) * (5/2 : ℝ) - (m - 11) = 0 := by
  sorry

end fixed_point_on_line_l696_69616


namespace triangle_determinant_l696_69662

theorem triangle_determinant (A B C : Real) : 
  A + B + C = π → 
  A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2 →
  Matrix.det !![Real.tan A, 1, 1; 1, Real.tan B, 1; 1, 1, Real.tan C] = 2 := by
sorry

end triangle_determinant_l696_69662


namespace sqrt_18_minus_sqrt_2_equality_l696_69671

theorem sqrt_18_minus_sqrt_2_equality (a b : ℝ) :
  Real.sqrt 18 - Real.sqrt 2 = a * Real.sqrt 2 - Real.sqrt 2 ∧
  a * Real.sqrt 2 - Real.sqrt 2 = b * Real.sqrt 2 →
  a * b = 6 := by
  sorry

end sqrt_18_minus_sqrt_2_equality_l696_69671


namespace total_spent_is_2100_l696_69609

/-- Calculates the total amount spent on a computer setup -/
def total_spent (computer_cost monitor_peripheral_ratio original_video_card_cost new_video_card_ratio : ℚ) : ℚ :=
  computer_cost + 
  (monitor_peripheral_ratio * computer_cost) + 
  (new_video_card_ratio * original_video_card_cost - original_video_card_cost)

/-- Proves that the total amount spent is $2100 given the specified costs and ratios -/
theorem total_spent_is_2100 : 
  total_spent 1500 (1/5) 300 2 = 2100 := by
  sorry

end total_spent_is_2100_l696_69609


namespace isosceles_triangle_proof_l696_69650

theorem isosceles_triangle_proof (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : 2 * (Real.cos B) * (Real.sin A) = Real.sin C) : A = B :=
sorry

end isosceles_triangle_proof_l696_69650


namespace cos_330_degrees_l696_69629

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l696_69629


namespace copper_percentage_in_first_alloy_l696_69607

/-- Prove that the percentage of copper in the first alloy is 20% -/
theorem copper_percentage_in_first_alloy
  (final_mixture_weight : ℝ)
  (final_copper_percentage : ℝ)
  (first_alloy_weight : ℝ)
  (second_alloy_copper_percentage : ℝ)
  (h1 : final_mixture_weight = 100)
  (h2 : final_copper_percentage = 24.9)
  (h3 : first_alloy_weight = 30)
  (h4 : second_alloy_copper_percentage = 27)
  : ∃ (first_alloy_copper_percentage : ℝ),
    first_alloy_copper_percentage = 20 ∧
    (first_alloy_copper_percentage / 100) * first_alloy_weight +
    (second_alloy_copper_percentage / 100) * (final_mixture_weight - first_alloy_weight) =
    (final_copper_percentage / 100) * final_mixture_weight :=
by sorry

end copper_percentage_in_first_alloy_l696_69607


namespace solution_range_l696_69676

theorem solution_range (m : ℝ) : 
  (∃ x y : ℝ, x + y = -1 ∧ 5 * x + 2 * y = 6 * m + 7 ∧ 2 * x - y < 19) → 
  m < 3/2 := by
sorry

end solution_range_l696_69676


namespace special_function_property_l696_69659

/-- A function from positive reals to positive reals satisfying the given condition -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x + f y) ≥ f (x + y) + f y)

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∀ x > 0, f x > x :=
by sorry

end special_function_property_l696_69659


namespace subtract_inequality_from_less_than_l696_69698

theorem subtract_inequality_from_less_than (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : 
  a - b < 0 := by
  sorry

end subtract_inequality_from_less_than_l696_69698


namespace jimmy_notebooks_l696_69624

/-- The number of notebooks Jimmy bought -/
def num_notebooks : ℕ := sorry

/-- The cost of one pen -/
def pen_cost : ℕ := 1

/-- The cost of one notebook -/
def notebook_cost : ℕ := 3

/-- The cost of one folder -/
def folder_cost : ℕ := 5

/-- The number of pens Jimmy bought -/
def num_pens : ℕ := 3

/-- The number of folders Jimmy bought -/
def num_folders : ℕ := 2

/-- The amount Jimmy paid with -/
def paid_amount : ℕ := 50

/-- The amount Jimmy received as change -/
def change_amount : ℕ := 25

theorem jimmy_notebooks :
  num_notebooks = 4 :=
sorry

end jimmy_notebooks_l696_69624


namespace sine_inequality_l696_69692

theorem sine_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π/4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) := by
  sorry

end sine_inequality_l696_69692


namespace two_sixty_billion_scientific_notation_l696_69649

-- Define 260 billion
def two_hundred_sixty_billion : ℝ := 260000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.6 * (10 ^ 11)

-- Theorem stating that 260 billion is equal to its scientific notation
theorem two_sixty_billion_scientific_notation : 
  two_hundred_sixty_billion = scientific_notation := by
  sorry

end two_sixty_billion_scientific_notation_l696_69649


namespace f_positive_range_f_greater_g_range_l696_69639

-- Define the functions f and g
def f (x : ℝ) := x^2 - x - 6
def g (b x : ℝ) := b*x - 10

-- Theorem for the range of x where f(x) > 0
theorem f_positive_range (x : ℝ) : 
  f x > 0 ↔ x < -2 ∨ x > 3 :=
sorry

-- Theorem for the range of b where f(x) > g(x) for all real x
theorem f_greater_g_range (b : ℝ) : 
  (∀ x : ℝ, f x > g b x) ↔ b < -5 ∨ b > 3 :=
sorry

end f_positive_range_f_greater_g_range_l696_69639


namespace z_value_l696_69634

theorem z_value (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x*y - 9) : z = 0 := by
  sorry

end z_value_l696_69634


namespace inserted_numbers_sum_l696_69652

theorem inserted_numbers_sum : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧ 
  (∃ r : ℝ, r > 0 ∧ x = 4 * r ∧ y = 4 * r^2) ∧
  (∃ d : ℝ, y = x + d ∧ 64 = y + d) ∧
  x + y = 131 + 3 * Real.sqrt 129 :=
sorry

end inserted_numbers_sum_l696_69652


namespace polar_to_rectangular_conversion_l696_69678

theorem polar_to_rectangular_conversion :
  ∀ (x y ρ θ : ℝ),
  ρ = Real.sin θ + Real.cos θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  ρ^2 = x^2 + y^2 →
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by sorry

end polar_to_rectangular_conversion_l696_69678


namespace max_value_fg_unique_root_condition_inequality_condition_l696_69687

noncomputable section

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

-- Part 1
theorem max_value_fg (x : ℝ) (hx : x ∈ Set.Icc (-2) 0) :
  (f 1 x) * (g x) ≤ 1 :=
sorry

-- Part 2
theorem unique_root_condition (k : ℝ) :
  (∃! x, f (-1) x = k * g x) ↔ (k > 3 / Real.exp 2 ∨ 0 < k ∧ k < 1 / Real.exp 1) :=
sorry

-- Part 3
theorem inequality_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ →
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end max_value_fg_unique_root_condition_inequality_condition_l696_69687


namespace sqrt_inequality_l696_69605

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > (1 : ℝ) / 2 ↔ 
  -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 :=
sorry

end sqrt_inequality_l696_69605


namespace floor_slab_rate_l696_69627

/-- Proves that for a rectangular room with given dimensions and total flooring cost,
    the rate per square meter is 900 Rs. -/
theorem floor_slab_rate (length width total_cost : ℝ) :
  length = 5 →
  width = 4.75 →
  total_cost = 21375 →
  total_cost / (length * width) = 900 := by
sorry

end floor_slab_rate_l696_69627


namespace point_trajectory_l696_69613

/-- The trajectory of a point satisfying a specific equation -/
theorem point_trajectory (x y : ℝ) :
  (Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) →
  ((x^2 / 16 - y^2 / 9 = 1) ∧ (x > 0)) :=
by sorry

end point_trajectory_l696_69613


namespace common_chord_circle_through_AB_center_on_line_smallest_circle_through_AB_l696_69600

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the line y = -x
def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

-- Theorem for the common chord
theorem common_chord : ∀ x y : ℝ, C₁ x y ∧ C₂ x y → x - 2*y + 4 = 0 :=
sorry

-- Theorem for the circle passing through A and B with center on y = -x
theorem circle_through_AB_center_on_line : ∃ h k : ℝ, 
  line_y_eq_neg_x h k ∧ 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 10 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
sorry

-- Theorem for the circle with smallest area passing through A and B
theorem smallest_circle_through_AB : ∀ x y : ℝ,
  (x + 2)^2 + (y - 1)^2 = 5 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) :=
sorry

end common_chord_circle_through_AB_center_on_line_smallest_circle_through_AB_l696_69600


namespace triangle_perimeter_bound_l696_69699

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 ∧ b = 23 ∧ 0 < s ∧ s < a + b ∧ a < b + s ∧ b < a + s → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + s → p < n := by
  sorry

end triangle_perimeter_bound_l696_69699


namespace arithmetic_sequence_sum_l696_69635

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2017 = 10 →
  a 1 * a 2017 = 16 →
  a 2 + a 1009 + a 2016 = 15 := by
  sorry

end arithmetic_sequence_sum_l696_69635


namespace tangerines_most_numerous_l696_69696

/-- Represents the number of boxes for each fruit type -/
structure BoxCounts where
  tangerines : Nat
  apples : Nat
  pears : Nat

/-- Represents the number of fruits per box for each fruit type -/
structure FruitsPerBox where
  tangerines : Nat
  apples : Nat
  pears : Nat

/-- Calculates the total number of fruits for each type -/
def totalFruits (boxes : BoxCounts) (perBox : FruitsPerBox) : BoxCounts :=
  { tangerines := boxes.tangerines * perBox.tangerines
  , apples := boxes.apples * perBox.apples
  , pears := boxes.pears * perBox.pears }

/-- Proves that tangerines are the most numerous fruit -/
theorem tangerines_most_numerous (boxes : BoxCounts) (perBox : FruitsPerBox) :
  boxes.tangerines = 5 →
  boxes.apples = 3 →
  boxes.pears = 4 →
  perBox.tangerines = 30 →
  perBox.apples = 20 →
  perBox.pears = 15 →
  let totals := totalFruits boxes perBox
  totals.tangerines > totals.apples ∧ totals.tangerines > totals.pears :=
by
  sorry


end tangerines_most_numerous_l696_69696


namespace aaron_matthews_more_cows_l696_69632

/-- Represents the number of cows each person has -/
structure CowCounts where
  aaron : ℕ
  matthews : ℕ
  marovich : ℕ

/-- The conditions of the problem -/
def cow_problem (c : CowCounts) : Prop :=
  c.aaron = 4 * c.matthews ∧
  c.matthews = 60 ∧
  c.aaron + c.matthews + c.marovich = 570

/-- The theorem to prove -/
theorem aaron_matthews_more_cows (c : CowCounts) 
  (h : cow_problem c) : c.aaron + c.matthews - c.marovich = 30 := by
  sorry


end aaron_matthews_more_cows_l696_69632


namespace not_p_sufficient_not_necessary_for_q_l696_69617

theorem not_p_sufficient_not_necessary_for_q :
  ∃ (x : ℝ), (x > 1 → 1 / x < 1) ∧ (1 / x < 1 → ¬(x > 1)) := by
  sorry

end not_p_sufficient_not_necessary_for_q_l696_69617


namespace perpendicular_line_equation_l696_69642

/-- The equation of a line perpendicular to x - 3y + 2 = 0 and passing through (1, -2) -/
theorem perpendicular_line_equation :
  let l₁ : ℝ → ℝ → Prop := λ x y => x - 3 * y + 2 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => 3 * x + y - 1 = 0
  let P : ℝ × ℝ := (1, -2)
  (∀ x y, l₁ x y → (3 * x + y = 0 → False)) ∧ 
  l₂ P.1 P.2 ∧
  ∀ x y, l₂ x y → (x - 3 * y = 0 → False) :=
by sorry

end perpendicular_line_equation_l696_69642


namespace point_in_second_quadrant_l696_69633

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let P : ℝ × ℝ := (-1, 2)
  second_quadrant P.1 P.2 :=
by
  sorry

end point_in_second_quadrant_l696_69633


namespace sin_geq_tan_minus_half_tan_cubed_l696_69622

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  Real.sin x ≥ Real.tan x - (1/2) * (Real.tan x)^3 := by
  sorry

end sin_geq_tan_minus_half_tan_cubed_l696_69622


namespace cab_delay_l696_69621

/-- Proves that a cab with reduced speed arrives 15 minutes late -/
theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) : 
  usual_time = 75 → speed_ratio = 5/6 → 
  (usual_time / speed_ratio) - usual_time = 15 := by
  sorry

end cab_delay_l696_69621


namespace students_enjoying_both_music_and_sports_l696_69657

theorem students_enjoying_both_music_and_sports 
  (total : ℕ) (music : ℕ) (sports : ℕ) (neither : ℕ) : 
  total = 55 → music = 35 → sports = 45 → neither = 4 → 
  music + sports - (total - neither) = 29 := by
sorry

end students_enjoying_both_music_and_sports_l696_69657


namespace hyperbola_vertices_distance_l696_69648

/-- The distance between the vertices of the hyperbola x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertices_distance : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/16 - y^2/9 = 1
  ∃ (v₁ v₂ : ℝ × ℝ), 
    (h v₁.1 v₁.2 ∧ h v₂.1 v₂.2) ∧ 
    (v₁.2 = 0 ∧ v₂.2 = 0) ∧
    (v₁.1 = -v₂.1) ∧
    abs (v₁.1 - v₂.1) = 8 :=
by sorry

end hyperbola_vertices_distance_l696_69648


namespace prob_one_of_each_specific_jar_l696_69658

/-- Represents the number of marbles of each color in the jar -/
structure MarbleJar :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)

/-- Calculates the probability of drawing one red, one blue, and one yellow marble -/
def prob_one_of_each (jar : MarbleJar) : ℚ :=
  sorry

/-- The theorem statement -/
theorem prob_one_of_each_specific_jar :
  prob_one_of_each ⟨3, 8, 9⟩ = 18 / 95 := by
  sorry

end prob_one_of_each_specific_jar_l696_69658


namespace second_person_age_l696_69695

/-- Given a group of 7 people, if adding a 39-year-old increases the average age by 2,
    and adding another person decreases the average age by 1,
    then the age of the second person added is 15 years old. -/
theorem second_person_age (initial_group : Finset ℕ) 
  (initial_total_age : ℕ) (second_person_age : ℕ) :
  (initial_group.card = 7) →
  (initial_total_age / 7 + 2 = (initial_total_age + 39) / 8) →
  (initial_total_age / 7 - 1 = (initial_total_age + second_person_age) / 8) →
  second_person_age = 15 := by
  sorry


end second_person_age_l696_69695


namespace smallest_satisfying_arrangement_l696_69686

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_guests : ℕ

/-- Checks if a seating arrangement satisfies the condition -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  ∀ (i : ℕ), i < seating.total_chairs →
    ∃ (j : ℕ), j < seating.seated_guests ∧
      (i % (seating.total_chairs / seating.seated_guests) = 0 ∨
       (i + 1) % (seating.total_chairs / seating.seated_guests) = 0)

/-- The main theorem to be proved -/
theorem smallest_satisfying_arrangement :
  ∀ (n : ℕ), n < 20 →
    ¬(satisfies_condition { total_chairs := 120, seated_guests := n }) ∧
    satisfies_condition { total_chairs := 120, seated_guests := 20 } :=
by sorry


end smallest_satisfying_arrangement_l696_69686


namespace parabola_intercepts_sum_l696_69601

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 5

-- Define the x-intercept
def x_intercept : ℝ := parabola 0

-- Define the y-intercepts
def y_intercepts : Set ℝ := {y | parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧
  x_intercept + b + c = 8 := by sorry

end parabola_intercepts_sum_l696_69601


namespace square_property_implies_equality_l696_69608

theorem square_property_implies_equality (n : ℕ) (a : ℕ) (a_list : List ℕ) 
  (h : ∀ k : ℕ, ∃ m : ℕ, a * k + 1 = m ^ 2 → 
    ∃ (i : ℕ) (hi : i < a_list.length) (p : ℕ), a_list[i] * k + 1 = p ^ 2) :
  a ∈ a_list := by
  sorry

end square_property_implies_equality_l696_69608


namespace race_start_relation_l696_69614

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceCondition where
  a : Runner
  b : Runner
  c : Runner
  race_length : ℝ
  a_c_start : ℝ
  b_c_start : ℝ

/-- Theorem stating the relation between the starts given by runners -/
theorem race_start_relation (cond : RaceCondition) 
  (h1 : cond.race_length = 1000)
  (h2 : cond.a_c_start = 600)
  (h3 : cond.b_c_start = 428.57) :
  ∃ (a_b_start : ℝ), a_b_start = 750 ∧ 
    (cond.race_length - a_b_start) / cond.race_length = 
    (cond.race_length - cond.b_c_start) / (cond.race_length - cond.a_c_start) :=
by sorry

end race_start_relation_l696_69614


namespace sarah_shopping_theorem_l696_69623

theorem sarah_shopping_theorem (toy_car1 toy_car2_orig scarf_orig beanie gloves book necklace_orig : ℚ)
  (toy_car2_discount scarf_discount beanie_tax necklace_discount : ℚ)
  (remaining : ℚ)
  (h1 : toy_car1 = 12)
  (h2 : toy_car2_orig = 15)
  (h3 : toy_car2_discount = 0.1)
  (h4 : scarf_orig = 10)
  (h5 : scarf_discount = 0.2)
  (h6 : beanie = 14)
  (h7 : beanie_tax = 0.08)
  (h8 : necklace_orig = 20)
  (h9 : necklace_discount = 0.05)
  (h10 : gloves = 12)
  (h11 : book = 15)
  (h12 : remaining = 7) :
  toy_car1 +
  (toy_car2_orig - toy_car2_orig * toy_car2_discount) +
  (scarf_orig - scarf_orig * scarf_discount) +
  (beanie + beanie * beanie_tax) +
  (necklace_orig - necklace_orig * necklace_discount) +
  gloves +
  book +
  remaining = 101.62 := by
sorry

end sarah_shopping_theorem_l696_69623


namespace repeating_decimal_sum_l696_69655

theorem repeating_decimal_sum (b c : ℕ) : 
  b < 10 → c < 10 →
  (10 * b + c : ℚ) / 99 + (100 * c + 10 * b + c : ℚ) / 999 = 83 / 222 →
  b = 1 ∧ c = 1 := by sorry

end repeating_decimal_sum_l696_69655


namespace floor_times_self_equals_72_l696_69638

theorem floor_times_self_equals_72 (x : ℝ) :
  x > 0 ∧ ⌊x⌋ * x = 72 → x = 9 := by
  sorry

end floor_times_self_equals_72_l696_69638


namespace boat_travel_time_l696_69612

/-- Given a boat that travels 2 miles in 5 minutes, prove that it takes 90 minutes to travel 36 miles at the same speed. -/
theorem boat_travel_time (distance : ℝ) (time : ℝ) (total_distance : ℝ) 
  (h1 : distance = 2) 
  (h2 : time = 5) 
  (h3 : total_distance = 36) : 
  (total_distance / (distance / time)) = 90 := by
  sorry


end boat_travel_time_l696_69612


namespace complex_on_real_axis_l696_69679

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  (z.im = 0) → a = 1 := by
sorry

end complex_on_real_axis_l696_69679


namespace min_distance_sum_l696_69697

theorem min_distance_sum (x a b : ℚ) : 
  x ≠ a ∧ x ≠ b ∧ a ≠ b →
  a > b →
  (∀ y : ℚ, |y - a| + |y - b| ≥ 2) ∧ (∃ z : ℚ, |z - a| + |z - b| = 2) →
  2022 + a - b = 2024 :=
by sorry

end min_distance_sum_l696_69697
