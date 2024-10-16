import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_symmetry_l4085_408592

/-- Given a polynomial g(x) = ax^2 + bx^3 + cx + d where g(-3) = 2, prove that g(3) = 0 -/
theorem polynomial_symmetry (a b c d : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = a * x^2 + b * x^3 + c * x + d)
  (h2 : g (-3) = 2) : 
  g 3 = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l4085_408592


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l4085_408500

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecularWeight (carbon_atoms : ℕ) (hydrogen_atoms : ℕ) (oxygen_atoms : ℕ) 
  (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight + oxygen_atoms * oxygen_weight

/-- Theorem stating that the molecular weight of the given compound is 192.124 g/mol -/
theorem compound_molecular_weight :
  molecularWeight 6 8 7 12.01 1.008 16.00 = 192.124 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l4085_408500


namespace NUMINAMATH_CALUDE_ant_expected_moves_l4085_408504

/-- Represents the possible parities of the ant's position -/
inductive Parity
  | Even
  | Odd

/-- Defines the ant's position on the coordinate plane -/
structure AntPosition :=
  (x : Parity)
  (y : Parity)

/-- Calculates the expected number of moves to reach an anthill from a given position -/
noncomputable def expectedMoves (pos : AntPosition) : ℝ :=
  match pos with
  | ⟨Parity.Even, Parity.Even⟩ => 4
  | ⟨Parity.Odd, Parity.Odd⟩ => 0
  | _ => 3

/-- The main theorem to be proved -/
theorem ant_expected_moves :
  let initialPos : AntPosition := ⟨Parity.Even, Parity.Even⟩
  expectedMoves initialPos = 4 := by sorry

end NUMINAMATH_CALUDE_ant_expected_moves_l4085_408504


namespace NUMINAMATH_CALUDE_minuend_value_l4085_408506

theorem minuend_value (minuend subtrahend : ℝ) 
  (h : minuend + subtrahend + (minuend - subtrahend) = 25) : 
  minuend = 12.5 := by
sorry

end NUMINAMATH_CALUDE_minuend_value_l4085_408506


namespace NUMINAMATH_CALUDE_perpendicular_lines_l4085_408584

/-- Two lines ax+y-1=0 and x-y+3=0 are perpendicular if and only if a = 1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (a*x + y = 1 ∧ x - y = -3) → 
   ((-a) * 1 = -1)) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l4085_408584


namespace NUMINAMATH_CALUDE_excess_hour_cost_correct_l4085_408563

/-- The cost per hour in excess of 2 hours for a parking garage -/
def excess_hour_cost : ℝ := 1.75

/-- The cost to park for up to 2 hours -/
def initial_cost : ℝ := 15

/-- The average cost per hour to park for 9 hours -/
def average_cost_9_hours : ℝ := 3.0277777777777777

/-- Theorem stating that the excess hour cost is correct given the initial cost and average cost -/
theorem excess_hour_cost_correct : 
  (initial_cost + 7 * excess_hour_cost) / 9 = average_cost_9_hours :=
by sorry

end NUMINAMATH_CALUDE_excess_hour_cost_correct_l4085_408563


namespace NUMINAMATH_CALUDE_patch_net_profit_l4085_408572

/-- Calculates the net profit from selling patches --/
theorem patch_net_profit (order_quantity : ℕ) (cost_per_patch : ℚ) (sell_price : ℚ) : 
  order_quantity = 100 ∧ cost_per_patch = 125/100 ∧ sell_price = 12 →
  (sell_price * order_quantity) - (cost_per_patch * order_quantity) = 1075 := by
  sorry

end NUMINAMATH_CALUDE_patch_net_profit_l4085_408572


namespace NUMINAMATH_CALUDE_point_on_x_axis_l4085_408543

/-- 
If a point P(a+2, a-3) lies on the x-axis, then its coordinates are (5, 0).
-/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a + 2 ∧ P.2 = a - 3 ∧ P.2 = 0) → 
  (∃ P : ℝ × ℝ, P = (5, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l4085_408543


namespace NUMINAMATH_CALUDE_min_sum_a7_a14_l4085_408510

/-- An arithmetic sequence of positive real numbers -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 0 < a n ∧ ∃ d : ℝ, a (n + 1) = a n + d

/-- The theorem stating the minimum value of a_7 + a_14 in the given sequence -/
theorem min_sum_a7_a14 (a : ℕ → ℝ) (h_arith : ArithmeticSequence a) (h_prod : a 1 * a 20 = 100) :
  ∀ x y : ℝ, x = a 7 ∧ y = a 14 → x + y ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_sum_a7_a14_l4085_408510


namespace NUMINAMATH_CALUDE_right_triangle_tangent_sum_l4085_408568

theorem right_triangle_tangent_sum (α β : Real) (k : Real) : 
  α > 0 → β > 0 → α + β = π / 2 →
  (1 / 2) * Real.cos α * Real.cos β = k →
  Real.tan α + Real.tan β = 2 * k := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_sum_l4085_408568


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l4085_408501

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 50 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 4500 →
  (n1 + n2 : ℚ) = 80 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / (n1 + n2 : ℚ) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l4085_408501


namespace NUMINAMATH_CALUDE_min_marked_price_for_profit_l4085_408509

theorem min_marked_price_for_profit (num_sets : ℕ) (purchase_price : ℝ) (discount_rate : ℝ) (desired_profit : ℝ) :
  let marked_price := (desired_profit + num_sets * purchase_price) / (num_sets * (1 - discount_rate))
  marked_price ≥ 200 ∧ 
  num_sets * (1 - discount_rate) * marked_price - num_sets * purchase_price ≥ desired_profit ∧
  ∀ x < marked_price, num_sets * (1 - discount_rate) * x - num_sets * purchase_price < desired_profit :=
by sorry

end NUMINAMATH_CALUDE_min_marked_price_for_profit_l4085_408509


namespace NUMINAMATH_CALUDE_square_sum_product_l4085_408513

theorem square_sum_product (x : ℝ) : 
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) → (10 + x) * (30 - x) = 144 := by
sorry

end NUMINAMATH_CALUDE_square_sum_product_l4085_408513


namespace NUMINAMATH_CALUDE_problem_solution_l4085_408585

theorem problem_solution : ∃ N : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  N / sum = 2 * diff ∧ N % sum = 80 ∧ N = 220080 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4085_408585


namespace NUMINAMATH_CALUDE_monica_has_eight_cookies_left_l4085_408593

/-- The number of cookies left for Monica --/
def cookies_left_for_monica (total_cookies : ℕ) (father_cookies : ℕ) : ℕ :=
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  total_cookies - (father_cookies + mother_cookies + brother_cookies)

/-- Theorem stating that Monica has 8 cookies left --/
theorem monica_has_eight_cookies_left :
  cookies_left_for_monica 30 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_monica_has_eight_cookies_left_l4085_408593


namespace NUMINAMATH_CALUDE_sequence_growth_l4085_408517

/-- A sequence of integers satisfying the given conditions -/
def Sequence (a : ℕ → ℤ) : Prop :=
  a 1 > a 0 ∧ a 1 > 0 ∧ ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r

/-- The main theorem -/
theorem sequence_growth (a : ℕ → ℤ) (h : Sequence a) : a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l4085_408517


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l4085_408540

/-- In a parallelogram where one angle exceeds the other by 10 degrees, 
    the measure of the smaller angle is 85 degrees. -/
theorem parallelogram_angle_measure : 
  ∀ (a b : ℝ), 
  (a > 0 ∧ b > 0) →  -- Ensure angles are positive
  (a + b = 180) →    -- Sum of adjacent angles in a parallelogram is 180°
  (b = a + 10) →     -- One angle exceeds the other by 10°
  a = 85 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l4085_408540


namespace NUMINAMATH_CALUDE_impossible_time_reduction_l4085_408521

/-- Proves that it's impossible to reduce the time per kilometer by 1 minute when starting from a speed of 60 km/h. -/
theorem impossible_time_reduction (initial_speed : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → time_reduction = 1 → ¬ (∃ (new_speed : ℝ), new_speed > 0 ∧ (1 / new_speed) * 60 = (1 / initial_speed) * 60 - time_reduction) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_time_reduction_l4085_408521


namespace NUMINAMATH_CALUDE_machine_N_output_fraction_l4085_408571

/-- Represents the production time of a machine relative to machine N -/
structure MachineTime where
  relative_to_N : ℚ

/-- Represents the production rate of a machine -/
def production_rate (m : MachineTime) : ℚ := 1 / m.relative_to_N

/-- The production time of machine T -/
def machine_T : MachineTime := ⟨3/4⟩

/-- The production time of machine N -/
def machine_N : MachineTime := ⟨1⟩

/-- The production time of machine O -/
def machine_O : MachineTime := ⟨3/2⟩

/-- The total production rate of all machines -/
def total_rate : ℚ :=
  production_rate machine_T + production_rate machine_N + production_rate machine_O

/-- The fraction of total output produced by machine N -/
def fraction_by_N : ℚ := production_rate machine_N / total_rate

theorem machine_N_output_fraction :
  fraction_by_N = 1/3 := by sorry

end NUMINAMATH_CALUDE_machine_N_output_fraction_l4085_408571


namespace NUMINAMATH_CALUDE_marble_game_winner_l4085_408552

/-- Represents the distribution of marbles in a single game -/
structure GameDistribution where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the total marbles each player has after all games -/
structure FinalMarbles where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The main theorem statement -/
theorem marble_game_winner
  (p q r : ℕ)
  (h_p_lt_q : p < q)
  (h_q_lt_r : q < r)
  (h_p_pos : 0 < p)
  (h_sum : p + q + r = 13)
  (final_marbles : FinalMarbles)
  (h_final_a : final_marbles.a = 20)
  (h_final_b : final_marbles.b = 10)
  (h_final_c : final_marbles.c = 9)
  (h_b_last : ∃ (g1 g2 : GameDistribution), g1.b + g2.b + r = 10)
  : ∃ (g1 g2 : GameDistribution),
    g1.c = q ∧ g2.c ≠ q ∧
    g1.a + g2.a + final_marbles.a - 20 = p + q + r ∧
    g1.b + g2.b + final_marbles.b - 10 = p + q + r ∧
    g1.c + g2.c + final_marbles.c - 9 = p + q + r :=
sorry

end NUMINAMATH_CALUDE_marble_game_winner_l4085_408552


namespace NUMINAMATH_CALUDE_product_diversity_l4085_408562

theorem product_diversity (n k : ℕ+) :
  ∃ (m : ℕ), m = n + k - 2 ∧
  ∀ (A : Finset ℝ) (B : Finset ℝ),
    A.card = k ∧ B.card = n →
    (A.product B).card ≥ m ∧
    ∀ (m' : ℕ), m' > m →
      ∃ (A' : Finset ℝ) (B' : Finset ℝ),
        A'.card = k ∧ B'.card = n ∧ (A'.product B').card < m' :=
by sorry

end NUMINAMATH_CALUDE_product_diversity_l4085_408562


namespace NUMINAMATH_CALUDE_similar_right_triangles_shortest_side_l4085_408545

theorem similar_right_triangles_shortest_side 
  (leg1 : ℝ) (hyp1 : ℝ) (hyp2 : ℝ) 
  (h_right : leg1 ^ 2 + (hyp1 ^ 2 - leg1 ^ 2) = hyp1 ^ 2) 
  (h_leg1 : leg1 = 15) 
  (h_hyp1 : hyp1 = 17) 
  (h_hyp2 : hyp2 = 51) : 
  (leg1 * hyp2 / hyp1) = 24 := by sorry

end NUMINAMATH_CALUDE_similar_right_triangles_shortest_side_l4085_408545


namespace NUMINAMATH_CALUDE_number_of_employees_l4085_408525

-- Define the given constants
def average_salary : ℚ := 1500
def salary_increase : ℚ := 100
def manager_salary : ℚ := 3600

-- Define the number of employees (excluding manager) as a variable
variable (n : ℚ)

-- Define the theorem
theorem number_of_employees : 
  (n * average_salary + manager_salary) / (n + 1) = average_salary + salary_increase →
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_employees_l4085_408525


namespace NUMINAMATH_CALUDE_right_handed_players_count_l4085_408514

/-- Calculates the total number of right-handed players in a cricket team. -/
theorem right_handed_players_count 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h4 : throwers ≤ total_players) : 
  throwers + ((total_players - throwers) * 2 / 3) = 59 := by
  sorry

#check right_handed_players_count

end NUMINAMATH_CALUDE_right_handed_players_count_l4085_408514


namespace NUMINAMATH_CALUDE_probability_blue_then_yellow_l4085_408577

def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 4
def pink_marbles : ℕ := 9

def total_marbles : ℕ := blue_marbles + yellow_marbles + pink_marbles

theorem probability_blue_then_yellow :
  (blue_marbles : ℚ) / total_marbles * yellow_marbles / (total_marbles - 1) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_blue_then_yellow_l4085_408577


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l4085_408581

/-- Systematic sampling function that generates a sample number for a given group -/
def sampleNumber (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 100 + 100 * k

/-- Generates the full sample of 10 numbers given an initial value x -/
def generateSample (x : ℕ) : List ℕ :=
  List.range 10 |>.map (sampleNumber x)

/-- Checks if a number ends with the digits 87 -/
def endsWith87 (n : ℕ) : Bool :=
  n % 100 = 87

/-- Set of possible x values that result in a sample number ending with 87 -/
def possibleXValues : Set ℕ :=
  {x | x ∈ Finset.range 100 ∧ ∃ k, k ∈ Finset.range 10 ∧ endsWith87 (sampleNumber x k)}

theorem systematic_sampling_theorem :
  (generateSample 24 = [24, 157, 290, 423, 556, 689, 822, 955, 88, 221]) ∧
  (possibleXValues = {21, 22, 23, 54, 55, 56, 87, 88, 89, 90}) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l4085_408581


namespace NUMINAMATH_CALUDE_problem_statement_l4085_408556

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 1| + |x + a|

-- Define the theorem
theorem problem_statement 
  (a b : ℝ)
  (m n : ℝ)
  (h1 : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1)
  (h2 : m > 0)
  (h3 : n > 0)
  (h4 : 1/(2*m) + 2/n + 2*a = 0) :
  a = -1 ∧ 4*m^2 + n^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4085_408556


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_is_all_reals_l4085_408505

theorem quadratic_inequality_solution_is_all_reals :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 5 * x + 6
  ∀ x : ℝ, f x < 0 ∨ f x > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_is_all_reals_l4085_408505


namespace NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l4085_408502

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * log x) / (3*x + 1)

theorem tangent_perpendicular_and_inequality (a : ℝ) :
  (∃ k : ℝ, (deriv (f a) 1 = k) ∧ (k * (-1) = -1)) →
  (∃ m : ℝ, ∀ x ∈ Set.Icc 1 (exp 1), f a x ≤ m * x) →
  (a = 0 ∧ ∀ m : ℝ, (∀ x ∈ Set.Icc 1 (exp 1), f a x ≤ m * x) → m ≥ 4 / (3 * exp 1 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l4085_408502


namespace NUMINAMATH_CALUDE_expression_equality_l4085_408573

theorem expression_equality : -1^4 + (-2)^3 / 4 * (5 - (-3)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4085_408573


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l4085_408575

-- Define the slopes of the two lines
def slope1 : ℚ := -2/3
def slope2 (b : ℚ) : ℚ := -b/3

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  ∃ b : ℚ, perpendicular b ∧ b = -9/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l4085_408575


namespace NUMINAMATH_CALUDE_remainder_theorem_l4085_408551

theorem remainder_theorem (A : ℕ) 
  (h1 : A % 1981 = 35) 
  (h2 : A % 1982 = 35) : 
  A % 14 = 7 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4085_408551


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l4085_408560

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 14 → x ≥ 8 ∧ 8 < 3*8 - 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l4085_408560


namespace NUMINAMATH_CALUDE_cookie_ratio_l4085_408576

def cookie_problem (initial_white : ℕ) (black_white_difference : ℕ) (remaining_total : ℕ) : Prop :=
  let initial_black : ℕ := initial_white + black_white_difference
  let eaten_black : ℕ := initial_black / 2
  let remaining_black : ℕ := initial_black - eaten_black
  let remaining_white : ℕ := remaining_total - remaining_black
  let eaten_white : ℕ := initial_white - remaining_white
  (eaten_white : ℚ) / initial_white = 3 / 4

theorem cookie_ratio :
  cookie_problem 80 50 85 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l4085_408576


namespace NUMINAMATH_CALUDE_teaching_competition_score_l4085_408542

theorem teaching_competition_score (teaching_design_weight : ℝ) 
                                   (on_site_demo_weight : ℝ) 
                                   (teaching_design_score : ℝ) 
                                   (on_site_demo_score : ℝ) 
                                   (h1 : teaching_design_weight = 0.2)
                                   (h2 : on_site_demo_weight = 0.8)
                                   (h3 : teaching_design_score = 90)
                                   (h4 : on_site_demo_score = 95) :
  teaching_design_weight * teaching_design_score + 
  on_site_demo_weight * on_site_demo_score = 94 := by
sorry

end NUMINAMATH_CALUDE_teaching_competition_score_l4085_408542


namespace NUMINAMATH_CALUDE_cos_2alpha_eq_neg_one_seventh_l4085_408546

theorem cos_2alpha_eq_neg_one_seventh (α : Real) 
  (h : 3 * Real.sin (α - π/6) = Real.sin (α + π/6)) : 
  Real.cos (2 * α) = -1/7 := by sorry

end NUMINAMATH_CALUDE_cos_2alpha_eq_neg_one_seventh_l4085_408546


namespace NUMINAMATH_CALUDE_inequality_solution_l4085_408507

theorem inequality_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6)) :=
by sorry


end NUMINAMATH_CALUDE_inequality_solution_l4085_408507


namespace NUMINAMATH_CALUDE_line_vector_at_t_one_l4085_408564

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  vector : ℝ → Fin 2 → ℝ

/-- The theorem stating the properties of the given line and the vector to be proved -/
theorem line_vector_at_t_one
  (L : ParameterizedLine)
  (h1 : L.vector 4 = ![2, 5])
  (h2 : L.vector 5 = ![4, -3]) :
  L.vector 1 = ![8, -19] := by
  sorry

end NUMINAMATH_CALUDE_line_vector_at_t_one_l4085_408564


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l4085_408574

theorem complex_modulus_equality (m : ℝ) (h : m > 0) :
  Complex.abs (4 + m * Complex.I) = 4 * Real.sqrt 13 → m = 8 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l4085_408574


namespace NUMINAMATH_CALUDE_intersection_range_chord_length_l4085_408589

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The range of m for which the line intersects the ellipse -/
theorem intersection_range (m : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 := by sorry

/-- The length of the chord when the line passes through (1,0) -/
theorem chord_length : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ (-1) ∧ line x₂ y₂ (-1) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (4 / 3) * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_intersection_range_chord_length_l4085_408589


namespace NUMINAMATH_CALUDE_paint_problem_solution_l4085_408565

def paint_problem (total_paint : ℚ) (second_week_fraction : ℚ) (total_used : ℚ) (first_week_fraction : ℚ) : Prop :=
  total_paint > 0 ∧
  second_week_fraction > 0 ∧ second_week_fraction < 1 ∧
  total_used > 0 ∧ total_used < total_paint ∧
  first_week_fraction > 0 ∧ first_week_fraction < 1 ∧
  first_week_fraction * total_paint + second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used

theorem paint_problem_solution :
  paint_problem 360 (1/3) 180 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_paint_problem_solution_l4085_408565


namespace NUMINAMATH_CALUDE_remaining_money_for_seat_and_tape_l4085_408537

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def wheel_cost : ℕ := 25

theorem remaining_money_for_seat_and_tape :
  initial_amount - (frame_cost + wheel_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_for_seat_and_tape_l4085_408537


namespace NUMINAMATH_CALUDE_biking_distance_difference_l4085_408558

/-- The distance biked by Daniela in four hours -/
def daniela_distance : ℤ := 75

/-- The distance biked by Carlos in four hours -/
def carlos_distance : ℤ := 60

/-- The distance biked by Emilio in four hours -/
def emilio_distance : ℤ := 45

/-- Theorem stating the difference between Daniela's distance and the sum of Carlos' and Emilio's distances -/
theorem biking_distance_difference : 
  daniela_distance - (carlos_distance + emilio_distance) = -30 := by
  sorry

end NUMINAMATH_CALUDE_biking_distance_difference_l4085_408558


namespace NUMINAMATH_CALUDE_true_conjunction_with_negation_l4085_408523

theorem true_conjunction_with_negation (p q : Prop) (hp : p) (hq : ¬q) : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_true_conjunction_with_negation_l4085_408523


namespace NUMINAMATH_CALUDE_matthew_hotdogs_l4085_408587

/-- The number of hotdogs Ella wants -/
def ella_hotdogs : ℕ := 2

/-- The number of hotdogs Emma wants -/
def emma_hotdogs : ℕ := 2

/-- The number of hotdogs Luke wants -/
def luke_hotdogs : ℕ := 2 * (ella_hotdogs + emma_hotdogs)

/-- The number of hotdogs Hunter wants -/
def hunter_hotdogs : ℕ := (3 * (ella_hotdogs + emma_hotdogs)) / 2

/-- The total number of hotdogs Matthew needs to cook -/
def total_hotdogs : ℕ := ella_hotdogs + emma_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_matthew_hotdogs_l4085_408587


namespace NUMINAMATH_CALUDE_bake_sale_chips_l4085_408566

/-- The number of cups of chocolate chips needed for one recipe -/
def chips_per_recipe : ℕ := 2

/-- The number of recipes to be made -/
def num_recipes : ℕ := 23

/-- The total number of cups of chocolate chips needed -/
def total_chips : ℕ := chips_per_recipe * num_recipes

theorem bake_sale_chips : total_chips = 46 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_chips_l4085_408566


namespace NUMINAMATH_CALUDE_club_assignment_count_l4085_408528

/-- Represents a club -/
inductive Club
| LittleGrassLiteratureSociety
| StreetDanceClub
| FootballHouse
| CyclingClub

/-- Represents a student -/
inductive Student
| A
| B
| C
| D
| E

/-- A valid club assignment is a function from Student to Club -/
def ClubAssignment := Student → Club

/-- Predicate to check if a club assignment is valid -/
def isValidAssignment (assignment : ClubAssignment) : Prop :=
  (∀ c : Club, ∃ s : Student, assignment s = c) ∧
  (assignment Student.A ≠ Club.StreetDanceClub)

/-- The number of valid club assignments -/
def numValidAssignments : ℕ := sorry

theorem club_assignment_count :
  numValidAssignments = 180 := by sorry

end NUMINAMATH_CALUDE_club_assignment_count_l4085_408528


namespace NUMINAMATH_CALUDE_ratio_equality_l4085_408588

theorem ratio_equality : (240 : ℚ) / 1547 / (2 / 13) = (5 : ℚ) / 34 / (7 / 48) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l4085_408588


namespace NUMINAMATH_CALUDE_f_of_three_equals_e_squared_l4085_408557

theorem f_of_three_equals_e_squared 
  (f : ℝ → ℝ) 
  (h : ∀ x > 0, f (Real.log x + 1) = x) : 
  f 3 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_e_squared_l4085_408557


namespace NUMINAMATH_CALUDE_sean_patch_selling_price_l4085_408548

/-- Proves that the selling price per patch is $12 given the conditions of Sean's patch business. -/
theorem sean_patch_selling_price
  (num_patches : ℕ)
  (cost_per_patch : ℚ)
  (net_profit : ℚ)
  (h_num_patches : num_patches = 100)
  (h_cost_per_patch : cost_per_patch = 1.25)
  (h_net_profit : net_profit = 1075) :
  (cost_per_patch * num_patches + net_profit) / num_patches = 12 := by
  sorry

end NUMINAMATH_CALUDE_sean_patch_selling_price_l4085_408548


namespace NUMINAMATH_CALUDE_zoey_reading_schedule_l4085_408538

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (n : ℕ) : ℕ := 
  n * (2 * 2 + (n - 1) * 1) / 2

def weekday (start_day : ℕ) (days_passed : ℕ) : ℕ := 
  (start_day + days_passed) % 7

theorem zoey_reading_schedule :
  let number_of_books : ℕ := 20
  let friday : ℕ := 5
  (total_days number_of_books = 230) ∧ 
  (weekday friday (total_days number_of_books) = 4) := by
  sorry

#check zoey_reading_schedule

end NUMINAMATH_CALUDE_zoey_reading_schedule_l4085_408538


namespace NUMINAMATH_CALUDE_total_sales_correct_l4085_408527

/-- Represents the sales data for a house --/
structure HouseSale where
  boxes : Nat
  price_per_box : Float
  discount_rate : Float
  discount_threshold : Nat

/-- Represents the sales data for the neighbor --/
structure NeighborSale where
  boxes : Nat
  price_per_box : Float
  exchange_rate : Float

/-- Calculates the total sales in US dollars --/
def calculate_total_sales (green : HouseSale) (yellow : HouseSale) (brown : HouseSale) (neighbor : NeighborSale) (tax_rate : Float) : Float :=
  sorry

/-- The main theorem to prove --/
theorem total_sales_correct (green : HouseSale) (yellow : HouseSale) (brown : HouseSale) (neighbor : NeighborSale) (tax_rate : Float) :
  let green_sale := HouseSale.mk 3 4 0.1 2
  let yellow_sale := HouseSale.mk 3 (13/3) 0 0
  let brown_sale := HouseSale.mk 9 2 0.05 3
  let neighbor_sale := NeighborSale.mk 3 4.5 1.1
  calculate_total_sales green_sale yellow_sale brown_sale neighbor_sale 0.07 = 57.543 :=
  sorry

end NUMINAMATH_CALUDE_total_sales_correct_l4085_408527


namespace NUMINAMATH_CALUDE_box_length_calculation_l4085_408570

/-- The length of a cubic box given total volume, cost per box, and total cost -/
theorem box_length_calculation (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) :
  total_volume = 1080000 ∧ cost_per_box = 0.8 ∧ total_cost = 480 →
  ∃ (length : ℝ), abs (length - (total_volume / (total_cost / cost_per_box))^(1/3)) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_box_length_calculation_l4085_408570


namespace NUMINAMATH_CALUDE_triangle_area_l4085_408582

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → 
  a = 2 * c → 
  B = π / 3 → 
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l4085_408582


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4085_408569

/-- Given that x³ and y vary inversely, x and y are always positive, and y = 8 when x = 2,
    prove that x = 2/5 when y = 500. -/
theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ (k : ℝ), ∀ (x y : ℝ), x^3 * y = k) 
  (h4 : 2^3 * 8 = (2 : ℝ)^3 * 8) : 
  (y = 500 → x = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4085_408569


namespace NUMINAMATH_CALUDE_no_intersection_AB_in_C_l4085_408541

open Set Real

theorem no_intersection_AB_in_C (a b x y : ℝ) : 
  0 < x → x < π/2 → 0 < y → y < π/2 →
  let A := {(x, y) | a * (sin x + sin y) + (b - 1) * (cos x + cos y) = 0}
  let B := {(x, y) | (b + 1) * sin (x + y) - a * cos (x + y) = a}
  let C := {(a, b) | ∀ z, z^2 - 2*(a - b)*z + (a + b)^2 - 2 > 0}
  (a, b) ∈ C → A ∩ B = ∅ := by
sorry

end NUMINAMATH_CALUDE_no_intersection_AB_in_C_l4085_408541


namespace NUMINAMATH_CALUDE_sum_due_theorem_l4085_408599

/-- Calculates the sum due given the true discount, interest rate, and time period. -/
def sum_due (true_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  let present_value := (true_discount * 100) / (interest_rate * time)
  present_value + true_discount

/-- Proves that the sum due is 568 given the specified conditions. -/
theorem sum_due_theorem :
  sum_due 168 14 3 = 568 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_theorem_l4085_408599


namespace NUMINAMATH_CALUDE_cookies_sum_l4085_408547

/-- The number of cookies Mona brought -/
def mona_cookies : ℕ := 20

/-- The number of cookies Jasmine brought -/
def jasmine_cookies : ℕ := mona_cookies - 5

/-- The number of cookies Rachel brought -/
def rachel_cookies : ℕ := jasmine_cookies + 10

/-- The total number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies : ℕ := mona_cookies + jasmine_cookies + rachel_cookies

theorem cookies_sum : total_cookies = 60 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sum_l4085_408547


namespace NUMINAMATH_CALUDE_initial_customers_l4085_408544

theorem initial_customers (stayed : ℕ) (left : ℕ) : stayed = 3 → left = stayed + 5 → stayed + left = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_l4085_408544


namespace NUMINAMATH_CALUDE_root_product_theorem_l4085_408508

theorem root_product_theorem (c d m r s : ℝ) : 
  (c^2 - m*c + 3 = 0) →
  (d^2 - m*d + 3 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l4085_408508


namespace NUMINAMATH_CALUDE_three_connected_iff_sequence_from_K4_l4085_408524

/-- A simple graph. -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- A graph is 3-connected if removing any two vertices does not disconnect the graph. -/
def ThreeConnected (G : Graph V) : Prop := sorry

/-- The complete graph on 4 vertices. -/
def K4 (V : Type*) : Graph V := sorry

/-- Remove an edge from a graph. -/
def removeEdge (G : Graph V) (e : V × V) : Graph V := sorry

/-- Theorem 3.2.3 (Tutte, 1966): A graph is 3-connected if and only if it can be constructed
    from K4 by adding edges one at a time. -/
theorem three_connected_iff_sequence_from_K4 {V : Type*} (G : Graph V) :
  ThreeConnected G ↔ 
  ∃ (n : ℕ) (sequence : ℕ → Graph V),
    (sequence 0 = K4 V) ∧
    (sequence n = G) ∧
    (∀ i < n, ∃ e, sequence i = removeEdge (sequence (i + 1)) e) :=
  sorry

end NUMINAMATH_CALUDE_three_connected_iff_sequence_from_K4_l4085_408524


namespace NUMINAMATH_CALUDE_fraction_simplification_l4085_408549

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  (m^2 - 3*m) / (9 - m^2) = -m / (m + 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4085_408549


namespace NUMINAMATH_CALUDE_pelican_migration_l4085_408580

/-- Represents the number of Pelicans originally in Shark Bite Cove -/
def original_pelicans : ℕ := 30

/-- Represents the number of sharks in Pelican Bay -/
def sharks_in_pelican_bay : ℕ := 60

/-- Represents the number of Pelicans remaining in Shark Bite Cove -/
def remaining_pelicans : ℕ := 20

/-- The fraction of Pelicans that moved from Shark Bite Cove to Pelican Bay -/
def fraction_moved : ℚ := 1 / 3

theorem pelican_migration :
  (sharks_in_pelican_bay = 2 * original_pelicans) ∧
  (remaining_pelicans < original_pelicans) ∧
  (fraction_moved = (original_pelicans - remaining_pelicans : ℚ) / original_pelicans) :=
by sorry

end NUMINAMATH_CALUDE_pelican_migration_l4085_408580


namespace NUMINAMATH_CALUDE_log_equation_solution_l4085_408530

noncomputable def LogEquation (a b x : ℝ) : Prop :=
  5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)

theorem log_equation_solution (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  LogEquation a b x → (b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4085_408530


namespace NUMINAMATH_CALUDE_min_ratio_four_digit_number_l4085_408529

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The theorem stating that 1099 minimizes x/y for four-digit numbers -/
theorem min_ratio_four_digit_number :
  ∀ (x : FourDigitNumber),
    (x.val : ℚ) / digit_sum x.val ≥ 1099 / digit_sum 1099 := by sorry

end NUMINAMATH_CALUDE_min_ratio_four_digit_number_l4085_408529


namespace NUMINAMATH_CALUDE_password_unique_l4085_408526

def is_valid_password (n : ℕ) : Prop :=
  -- The password is an eight-digit number
  100000000 > n ∧ n ≥ 10000000 ∧
  -- The password is a multiple of both 3 and 25
  n % 3 = 0 ∧ n % 25 = 0 ∧
  -- The password is between 20,000,000 and 30,000,000
  n > 20000000 ∧ n < 30000000 ∧
  -- The millions place and the hundred thousands place digits are the same
  (n / 1000000) % 10 = (n / 100000) % 10 ∧
  -- The hundreds digit is 2 less than the ten thousands digit
  (n / 100) % 10 + 2 = (n / 10000) % 10 ∧
  -- The digits in the hundred thousands, ten thousands, and thousands places form a three-digit number
  -- which, when divided by the two-digit number formed by the digits in the ten millions and millions places,
  -- gives a quotient of 25
  ((n / 100000) % 1000) / ((n / 1000000) % 100) = 25

theorem password_unique : ∀ n : ℕ, is_valid_password n ↔ n = 26650350 := by
  sorry

end NUMINAMATH_CALUDE_password_unique_l4085_408526


namespace NUMINAMATH_CALUDE_no_equal_sums_l4085_408579

theorem no_equal_sums : ¬∃ (n : ℕ+), 
  (5 * n * (n + 1) : ℚ) / 2 = (5 * n * (n + 7) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_equal_sums_l4085_408579


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4085_408535

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 5 = 28 →
  a 3 + a 4 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4085_408535


namespace NUMINAMATH_CALUDE_sum_less_than_one_l4085_408583

theorem sum_less_than_one (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) : 
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_one_l4085_408583


namespace NUMINAMATH_CALUDE_messages_sent_l4085_408520

theorem messages_sent (lucia_day1 : ℕ) : 
  lucia_day1 > 20 →
  let alina_day1 := lucia_day1 - 20
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := 2 * alina_day1
  let lucia_day3 := lucia_day1
  let alina_day3 := alina_day1
  lucia_day1 + alina_day1 + lucia_day2 + alina_day2 + lucia_day3 + alina_day3 = 680 →
  lucia_day1 = 120 := by
sorry

end NUMINAMATH_CALUDE_messages_sent_l4085_408520


namespace NUMINAMATH_CALUDE_polygon_side_theorem_l4085_408553

def polygon_side_proof (total_area : ℝ) (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) (unknown_side_min unknown_side_max : ℝ) : Prop :=
  let rect1_area := rect1_length * rect1_width
  let rect2_area := rect2_length * rect2_width
  let unknown_rect_area := total_area - rect1_area - rect2_area
  ∃ (x : ℝ), (x = 7 ∨ x = 6) ∧ 
             unknown_rect_area = x * (unknown_rect_area / x) ∧
             x > unknown_side_min ∧ x < unknown_side_max

theorem polygon_side_theorem : 
  polygon_side_proof 72 10 1 5 4 5 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_side_theorem_l4085_408553


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l4085_408554

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l4085_408554


namespace NUMINAMATH_CALUDE_same_sign_as_B_l4085_408559

/-- A line in 2D space defined by the equation Ax + By + C = 0 -/
structure Line2D where
  A : ℝ
  B : ℝ
  C : ℝ
  A_nonzero : A ≠ 0
  B_nonzero : B ≠ 0

/-- Determines if a point (x, y) is above a given line -/
def IsAboveLine (l : Line2D) (x y : ℝ) : Prop :=
  l.A * x + l.B * y + l.C > 0

/-- The main theorem stating that for a point above the line, 
    Ax + By + C has the same sign as B -/
theorem same_sign_as_B (l : Line2D) (x y : ℝ) 
    (h : IsAboveLine l x y) : 
    (l.A * x + l.B * y + l.C) * l.B > 0 := by
  sorry

end NUMINAMATH_CALUDE_same_sign_as_B_l4085_408559


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4085_408522

/-- Calculates the total surface area of a cube with holes --/
def cube_surface_area_with_holes (cube_edge_length : ℝ) (hole_side_length : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge_length^2
  let hole_area := 6 * hole_side_length^2
  let exposed_internal_area := 6 * 4 * hole_side_length^2
  original_surface_area - hole_area + exposed_internal_area

/-- Theorem stating the total surface area of the cube with holes --/
theorem cube_with_holes_surface_area :
  cube_surface_area_with_holes 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4085_408522


namespace NUMINAMATH_CALUDE_batting_average_calculation_l4085_408503

/-- Calculates the new batting average after a match -/
def newBattingAverage (currentAverage : ℚ) (matchesPlayed : ℕ) (runsScored : ℕ) : ℚ :=
  (currentAverage * matchesPlayed + runsScored) / (matchesPlayed + 1)

/-- Theorem: Given the conditions, the new batting average will be 54 -/
theorem batting_average_calculation (currentAverage : ℚ) (matchesPlayed : ℕ) (runsScored : ℕ)
  (h1 : currentAverage = 51)
  (h2 : matchesPlayed = 5)
  (h3 : runsScored = 69) :
  newBattingAverage currentAverage matchesPlayed runsScored = 54 := by
  sorry

#eval newBattingAverage 51 5 69

end NUMINAMATH_CALUDE_batting_average_calculation_l4085_408503


namespace NUMINAMATH_CALUDE_kelly_games_giveaway_l4085_408590

theorem kelly_games_giveaway (initial_games : ℕ) (remaining_games : ℕ) : 
  initial_games = 50 → remaining_games = 35 → initial_games - remaining_games = 15 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_giveaway_l4085_408590


namespace NUMINAMATH_CALUDE_vampire_blood_requirement_l4085_408597

-- Define the constants
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Define the theorem
theorem vampire_blood_requirement :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := by
  sorry

end NUMINAMATH_CALUDE_vampire_blood_requirement_l4085_408597


namespace NUMINAMATH_CALUDE_tangerine_sales_theorem_l4085_408550

/-- Represents the daily sales data for a week -/
def sales_data : List Int := [300, -400, -200, 100, -600, 1200, 500]

/-- The planned daily sales amount in kilograms -/
def planned_daily_sales : Nat := 20000

/-- The selling price per kilogram in yuan -/
def selling_price : Nat := 6

/-- The express delivery cost and other expenses per kilogram in yuan -/
def expenses : Nat := 2

/-- The number of days in a week -/
def days_in_week : Nat := 7

theorem tangerine_sales_theorem :
  (List.maximum? sales_data).isSome ∧ 
  (List.minimum? sales_data).isSome →
  (∃ max min : Int, 
    (List.maximum? sales_data) = some max ∧
    (List.minimum? sales_data) = some min ∧
    max - min = 1800) ∧
  (planned_daily_sales * days_in_week + (List.sum sales_data)) * 
    (selling_price - expenses) = 563600 := by
  sorry

end NUMINAMATH_CALUDE_tangerine_sales_theorem_l4085_408550


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_le_eight_l4085_408536

theorem inequality_holds_iff_p_le_eight (p : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < π/2 → (1 + 1/Real.sin x)^3 ≥ p/(Real.tan x)^2) ↔ p ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_le_eight_l4085_408536


namespace NUMINAMATH_CALUDE_reverse_clock_theorem_l4085_408567

/-- Represents a clock with a reverse-moving minute hand -/
structure ReverseClock :=
  (hour : ℝ)
  (minute : ℝ)

/-- Converts a ReverseClock time to a standard clock time -/
def to_standard_time (c : ReverseClock) : ℝ := sorry

/-- Checks if the hands of a ReverseClock coincide -/
def hands_coincide (c : ReverseClock) : Prop := sorry

theorem reverse_clock_theorem :
  ∀ (c : ReverseClock),
    4 < c.hour ∧ c.hour < 5 →
    hands_coincide c →
    to_standard_time c = 4 + 36 / 60 + 12 / (13 * 60) :=
by sorry

end NUMINAMATH_CALUDE_reverse_clock_theorem_l4085_408567


namespace NUMINAMATH_CALUDE_initial_kola_percentage_l4085_408561

/-- Proves that the initial percentage of concentrated kola in a solution is 6% -/
theorem initial_kola_percentage (
  initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 80)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_kola = 6.8)
  (h6 : final_sugar_percentage = 14.111111111111112)
  : ∃ (initial_kola_percentage : ℝ),
    initial_kola_percentage = 6 ∧
    (initial_volume - initial_water_percentage / 100 * initial_volume - initial_kola_percentage / 100 * initial_volume + added_sugar) /
    (initial_volume + added_sugar + added_water + added_kola) =
    final_sugar_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_kola_percentage_l4085_408561


namespace NUMINAMATH_CALUDE_hockey_games_per_month_l4085_408598

/-- 
Given a hockey season with the following properties:
- The season lasts for 14 months
- There are 182 hockey games in the season

This theorem proves that the number of hockey games played each month is 13.
-/
theorem hockey_games_per_month (season_length : ℕ) (total_games : ℕ) 
  (h1 : season_length = 14) (h2 : total_games = 182) :
  total_games / season_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_hockey_games_per_month_l4085_408598


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l4085_408578

theorem mod_equivalence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l4085_408578


namespace NUMINAMATH_CALUDE_red_bead_count_l4085_408555

/-- Represents a necklace with blue and red beads. -/
structure Necklace where
  blue_count : Nat
  red_count : Nat
  is_valid : Bool

/-- Checks if a necklace satisfies the given conditions. -/
def is_valid_necklace (n : Necklace) : Prop :=
  n.blue_count = 30 ∧
  n.is_valid = true ∧
  n.red_count > 0 ∧
  n.red_count % 2 = 0 ∧
  n.red_count = 2 * n.blue_count

theorem red_bead_count (n : Necklace) :
  is_valid_necklace n → n.red_count = 60 := by
  sorry

#check red_bead_count

end NUMINAMATH_CALUDE_red_bead_count_l4085_408555


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l4085_408511

def point_symmetric_to_x_axis (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem symmetric_point_x_axis :
  let M : ℝ × ℝ := (1, 2)
  point_symmetric_to_x_axis M.1 M.2 = (1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l4085_408511


namespace NUMINAMATH_CALUDE_no_zeros_of_g_l4085_408539

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) + f(x)/x > 0 for all x ≠ 0,
    the function g(x) = f(x) + 1/x has no zeros. -/
theorem no_zeros_of_g (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x ≠ 0, deriv f x + f x / x > 0) :
    ∀ x ≠ 0, f x + 1 / x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeros_of_g_l4085_408539


namespace NUMINAMATH_CALUDE_circle_exists_with_conditions_l4085_408534

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a circle in 3D space
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D  -- Normal vector to the plane of the circle

def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

def circle_touches_plane (c : Circle3D) (p : Plane3D) : Prop := sorry

def circle_passes_through_points (c : Circle3D) (p1 p2 : Point3D) : Prop := sorry

theorem circle_exists_with_conditions 
  (p1 p2 : Point3D) 
  (projection_plane : Plane3D) : 
  ∃ (c : Circle3D), 
    circle_passes_through_points c p1 p2 ∧ 
    angle_between_planes (Plane3D.mk c.normal.x c.normal.y c.normal.z 0) projection_plane = π/3 ∧
    circle_touches_plane c projection_plane := by
  sorry

end NUMINAMATH_CALUDE_circle_exists_with_conditions_l4085_408534


namespace NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l4085_408594

theorem maximize_x_cube_y_fourth (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 40) :
  x^3 * y^4 ≤ 24^3 * 32^4 ∧ x^3 * y^4 = 24^3 * 32^4 ↔ x = 24 ∧ y = 32 := by
  sorry

end NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l4085_408594


namespace NUMINAMATH_CALUDE_jane_has_nine_cans_l4085_408516

/-- The number of sunflower seeds Jane has -/
def total_seeds : ℕ := 54

/-- The number of seeds Jane places in each can -/
def seeds_per_can : ℕ := 6

/-- The number of cans Jane has -/
def number_of_cans : ℕ := total_seeds / seeds_per_can

/-- Proof that Jane has 9 cans -/
theorem jane_has_nine_cans : number_of_cans = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_has_nine_cans_l4085_408516


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4085_408586

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^2 + 3*n = m^2) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l4085_408586


namespace NUMINAMATH_CALUDE_negation_of_universal_quantification_l4085_408531

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x + Real.log x > 0) ↔ (∃ x : ℝ, x + Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantification_l4085_408531


namespace NUMINAMATH_CALUDE_complement_of_intersection_l4085_408512

open Set

theorem complement_of_intersection (A B : Set ℝ) : 
  A = {x : ℝ | x ≤ 1} → 
  B = {x : ℝ | x < 2} → 
  (A ∩ B)ᶜ = {x : ℝ | x > 1} := by
sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l4085_408512


namespace NUMINAMATH_CALUDE_shelby_today_stars_l4085_408596

-- Define the variables
def yesterday_stars : ℕ := 4
def total_stars : ℕ := 7

-- State the theorem
theorem shelby_today_stars : 
  total_stars - yesterday_stars = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelby_today_stars_l4085_408596


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l4085_408532

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l4085_408532


namespace NUMINAMATH_CALUDE_meeting_point_relationship_l4085_408591

/-- Represents the scenario of two vehicles meeting on a road --/
structure MeetingScenario where
  S : ℝ  -- Distance between village and city
  x : ℝ  -- Speed of the truck
  y : ℝ  -- Speed of the car
  t : ℝ  -- Time taken to meet under normal conditions
  t1 : ℝ  -- Time taken to meet if truck leaves 45 minutes earlier
  t2 : ℝ  -- Time taken to meet if car leaves 20 minutes earlier

/-- The theorem stating the relationship between the meeting points --/
theorem meeting_point_relationship (scenario : MeetingScenario) :
  scenario.t = scenario.S / (scenario.x + scenario.y) →
  scenario.t1 = (scenario.S - 0.75 * scenario.x) / (scenario.x + scenario.y) →
  scenario.t2 = (scenario.S - scenario.y / 3) / (scenario.x + scenario.y) →
  0.75 * scenario.x + (scenario.S - 0.75 * scenario.x) * scenario.x / (scenario.x + scenario.y) - scenario.S * scenario.x / (scenario.x + scenario.y) = 18 →
  scenario.S * scenario.x / (scenario.x + scenario.y) - (scenario.S - scenario.y / 3) * scenario.x / (scenario.x + scenario.y) = 8 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_relationship_l4085_408591


namespace NUMINAMATH_CALUDE_pyramid_layers_l4085_408519

/-- Calculates the number of exterior golfballs for a given layer -/
def exteriorGolfballs (layer : ℕ) : ℕ :=
  if layer ≤ 2 then layer * layer else 4 * (layer - 1)

/-- Calculates the total number of exterior golfballs up to a given layer -/
def totalExteriorGolfballs (n : ℕ) : ℕ :=
  (List.range n).map exteriorGolfballs |>.sum

/-- Theorem stating that a pyramid with 145 exterior golfballs has 9 layers -/
theorem pyramid_layers (n : ℕ) : totalExteriorGolfballs n = 145 ↔ n = 9 := by
  sorry

#eval totalExteriorGolfballs 9  -- Should output 145

end NUMINAMATH_CALUDE_pyramid_layers_l4085_408519


namespace NUMINAMATH_CALUDE_cloak_change_theorem_l4085_408515

/-- Represents the price of an invisibility cloak and the change received in different scenarios -/
structure CloakTransaction where
  silverPaid : ℕ
  goldChange : ℕ

/-- Calculates the number of silver coins received as change when buying a cloak with gold coins -/
def silverChangeForGoldPurchase (transaction1 transaction2 : CloakTransaction) (goldPaid : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct change in silver coins when buying a cloak for 14 gold coins -/
theorem cloak_change_theorem (transaction1 transaction2 : CloakTransaction) 
  (h1 : transaction1.silverPaid = 20 ∧ transaction1.goldChange = 4)
  (h2 : transaction2.silverPaid = 15 ∧ transaction2.goldChange = 1) :
  silverChangeForGoldPurchase transaction1 transaction2 14 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_change_theorem_l4085_408515


namespace NUMINAMATH_CALUDE_coin_distribution_l4085_408595

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem coin_distribution : distribute 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_coin_distribution_l4085_408595


namespace NUMINAMATH_CALUDE_rectangle_breadth_l4085_408518

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) : 
  square_area = 16 → 
  rectangle_area = 220 → 
  ∃ (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ),
    circle_radius = Real.sqrt square_area ∧
    rectangle_length = 5 * circle_radius ∧
    rectangle_area = rectangle_length * rectangle_breadth ∧
    rectangle_breadth = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l4085_408518


namespace NUMINAMATH_CALUDE_cannot_simplify_further_l4085_408533

theorem cannot_simplify_further (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by sorry

end NUMINAMATH_CALUDE_cannot_simplify_further_l4085_408533
