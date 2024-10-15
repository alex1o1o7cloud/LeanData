import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l2148_214814

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (h_intersection : m + 4*n = 1) : 
  (1/m + 4/n) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2148_214814


namespace NUMINAMATH_CALUDE_jean_jane_money_total_jean_jane_money_total_proof_l2148_214806

/-- Given that Jean has three times as much money as Jane, and Jean has $57,
    prove that their combined total is $76. -/
theorem jean_jane_money_total : ℕ → ℕ → Prop :=
  fun jean_money jane_money =>
    (jean_money = 3 * jane_money) →
    (jean_money = 57) →
    (jean_money + jane_money = 76)

/-- The actual theorem instance -/
theorem jean_jane_money_total_proof : jean_jane_money_total 57 19 := by
  sorry

end NUMINAMATH_CALUDE_jean_jane_money_total_jean_jane_money_total_proof_l2148_214806


namespace NUMINAMATH_CALUDE_number_of_rooms_l2148_214898

theorem number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) :
  total_paintings / paintings_per_room = 4 := by
sorry

end NUMINAMATH_CALUDE_number_of_rooms_l2148_214898


namespace NUMINAMATH_CALUDE_ben_win_probability_l2148_214856

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : lose_prob + win_prob = 1) : win_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l2148_214856


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2148_214833

theorem complex_power_modulus : Complex.abs ((2 + 2 * Complex.I) ^ 6 + 3) = 515 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2148_214833


namespace NUMINAMATH_CALUDE_meetings_percentage_of_work_day_l2148_214800

/-- Represents the duration of a work day in minutes -/
def work_day_duration : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 80

/-- Represents the duration of the break between meetings in minutes -/
def break_duration : ℕ := 15

/-- Calculates the total time spent in meetings and on break -/
def total_meeting_time : ℕ :=
  first_meeting_duration + (3 * first_meeting_duration) + break_duration

/-- Theorem stating that the percentage of work day spent in meetings and on break is 56% -/
theorem meetings_percentage_of_work_day :
  (total_meeting_time : ℚ) / work_day_duration * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_work_day_l2148_214800


namespace NUMINAMATH_CALUDE_perfect_power_multiple_l2148_214875

theorem perfect_power_multiple : ∃ (n : ℕ), 
  n > 0 ∧ 
  ∃ (a b c : ℕ), 
    2 * n = a^2 ∧ 
    3 * n = b^3 ∧ 
    5 * n = c^5 := by
  sorry

end NUMINAMATH_CALUDE_perfect_power_multiple_l2148_214875


namespace NUMINAMATH_CALUDE_stone_piles_theorem_l2148_214883

/-- Represents the state of stone piles after operations -/
structure StonePiles :=
  (num_piles : Nat)
  (initial_stones : Nat)
  (operations : Nat)
  (pile_a_stones : Nat)
  (pile_b_stones : Nat)

/-- The theorem to prove -/
theorem stone_piles_theorem (sp : StonePiles) : 
  sp.num_piles = 20 →
  sp.initial_stones = 2006 →
  sp.operations < 20 →
  sp.pile_a_stones = 1990 →
  2080 ≤ sp.pile_b_stones →
  sp.pile_b_stones ≤ 2100 →
  sp.pile_b_stones = 2090 := by
sorry

end NUMINAMATH_CALUDE_stone_piles_theorem_l2148_214883


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2148_214818

theorem solution_set_equivalence (x : ℝ) :
  (x + 1) * (x - 1) < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2148_214818


namespace NUMINAMATH_CALUDE_gift_items_solution_l2148_214820

theorem gift_items_solution :
  ∃ (x y z : ℕ) (x' y' z' : ℕ),
    x + y + z = 20 ∧
    60 * x + 50 * y + 10 * z = 720 ∧
    x' + y' + z' = 20 ∧
    60 * x' + 50 * y' + 10 * z' = 720 ∧
    ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) ∧
    ((x' = 4 ∧ y' = 8 ∧ z' = 8) ∨ (x' = 8 ∧ y' = 3 ∧ z' = 9)) ∧
    ¬(x = x' ∧ y = y' ∧ z = z') :=
by
  sorry

#check gift_items_solution

end NUMINAMATH_CALUDE_gift_items_solution_l2148_214820


namespace NUMINAMATH_CALUDE_min_discount_factor_proof_l2148_214839

/-- Proves the minimum discount factor for a product with given cost and marked prices, ensuring a minimum profit margin. -/
theorem min_discount_factor_proof (cost_price marked_price : ℝ) (min_profit_margin : ℝ) 
  (h_cost : cost_price = 800)
  (h_marked : marked_price = 1200)
  (h_margin : min_profit_margin = 0.2) :
  ∃ x : ℝ, x = 0.8 ∧ 
    ∀ y : ℝ, (marked_price * y - cost_price ≥ cost_price * min_profit_margin) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_min_discount_factor_proof_l2148_214839


namespace NUMINAMATH_CALUDE_log3_45_not_expressible_l2148_214830

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Given conditions
axiom log3_27 : log 3 27 = 3
axiom log3_81 : log 3 81 = 4

-- Define the property of being expressible without logarithmic tables
def expressible_without_tables (x : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ), f (log 3 27) (log 3 81) = log 3 x

-- Theorem statement
theorem log3_45_not_expressible :
  ¬ expressible_without_tables 45 :=
sorry

end NUMINAMATH_CALUDE_log3_45_not_expressible_l2148_214830


namespace NUMINAMATH_CALUDE_whole_number_between_constraints_l2148_214831

theorem whole_number_between_constraints (N : ℤ) : 
  (6 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7.5) ↔ N ∈ ({25, 26, 27, 28, 29} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_whole_number_between_constraints_l2148_214831


namespace NUMINAMATH_CALUDE_farm_problem_l2148_214843

/-- The farm problem -/
theorem farm_problem (H C : ℕ) : 
  (H - 15) / (C + 15) = 3 →  -- After transaction, ratio is 3:1
  H - 15 = C + 15 + 70 →    -- After transaction, 70 more horses than cows
  H / C = 6                  -- Initial ratio is 6:1
:= by sorry

end NUMINAMATH_CALUDE_farm_problem_l2148_214843


namespace NUMINAMATH_CALUDE_contractor_payment_result_l2148_214861

def contractor_payment (total_days : ℕ) (working_pay : ℚ) (absence_fine : ℚ) (absent_days : ℕ) : ℚ :=
  let working_days := total_days - absent_days
  let total_earnings := working_days * working_pay
  let total_fines := absent_days * absence_fine
  total_earnings - total_fines

theorem contractor_payment_result :
  contractor_payment 30 25 7.5 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_result_l2148_214861


namespace NUMINAMATH_CALUDE_count_leftmost_seven_eq_diff_l2148_214817

/-- The set of powers of 7 from 0 to 3000 -/
def U : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 3000 ∧ n = 7^k}

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a natural number -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The number of elements in U with 7 as the leftmost digit -/
def count_leftmost_seven (U : Set ℕ) : ℕ := sorry

theorem count_leftmost_seven_eq_diff :
  num_digits (7^3000) = 2510 →
  leftmost_digit (7^3000) = 7 →
  count_leftmost_seven U = 3000 - 2509 := by sorry

end NUMINAMATH_CALUDE_count_leftmost_seven_eq_diff_l2148_214817


namespace NUMINAMATH_CALUDE_valid_schedule_count_is_twelve_l2148_214892

/-- Represents the four subjects in the class schedule -/
inductive Subject
| Chinese
| Mathematics
| English
| PhysicalEducation

/-- Represents a schedule of four periods -/
def Schedule := Fin 4 → Subject

/-- Checks if a schedule is valid (PE is not in first or fourth period) -/
def isValidSchedule (s : Schedule) : Prop :=
  s 0 ≠ Subject.PhysicalEducation ∧ s 3 ≠ Subject.PhysicalEducation

/-- The number of valid schedules -/
def validScheduleCount : ℕ := sorry

/-- Theorem stating that the number of valid schedules is 12 -/
theorem valid_schedule_count_is_twelve : validScheduleCount = 12 := by sorry

end NUMINAMATH_CALUDE_valid_schedule_count_is_twelve_l2148_214892


namespace NUMINAMATH_CALUDE_equation_solution_l2148_214801

theorem equation_solution (x : ℝ) (h_pos : x > 0) :
  7.74 * Real.sqrt (Real.log x / Real.log 5) + (Real.log x / Real.log 5) ^ (1/3) = 2 →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2148_214801


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2148_214826

theorem smallest_integer_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 1 ∧ 
  n % 6 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 4 = 1 → m % 5 = 1 → m % 6 = 1 → n ≤ m) ∧
  n = 61 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2148_214826


namespace NUMINAMATH_CALUDE_system_solution_l2148_214840

theorem system_solution :
  let solutions : List (ℝ × ℝ × ℝ) := [
    (1, -1, 1), (1, 3/2, -2/3), (-2, 1/2, 1), (-2, 3/2, 1/3), (3, -1, 1/3), (3, 1/2, -2/3)
  ]
  ∀ (x y z : ℝ),
    (x + 2*y + 3*z = 2 ∧
     1/x + 1/(2*y) + 1/(3*z) = 5/6 ∧
     x*y*z = -1) ↔
    (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2148_214840


namespace NUMINAMATH_CALUDE_range_of_sum_of_squares_l2148_214860

theorem range_of_sum_of_squares (x y : ℝ) (h : x^2 - 2*x*y + 5*y^2 = 4) :
  3 - Real.sqrt 5 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_of_squares_l2148_214860


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l2148_214893

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l2148_214893


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2148_214880

-- Define the point
def point : ℝ × ℝ := (12, -5)

-- Theorem statement
theorem distance_from_origin_to_point :
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2148_214880


namespace NUMINAMATH_CALUDE_garden_dimensions_l2148_214823

theorem garden_dimensions :
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.2 > p.1 ∧ 
      (p.1 - 6) * (p.2 - 6) = 12 ∧ 
      p.1 ≥ 7 ∧ p.2 ≥ 7)
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ 
  n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_dimensions_l2148_214823


namespace NUMINAMATH_CALUDE_valid_numbers_l2148_214851

def isValid (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 : ℕ), d1 > d2 ∧ d2 > d3 ∧
    d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
    d1 + d2 + d3 = 1457 ∧
    ∀ (d : ℕ), d ∣ n → d ≤ d1

theorem valid_numbers : ∀ (n : ℕ), isValid n ↔ n = 987 ∨ n = 1023 ∨ n = 1085 ∨ n = 1175 := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l2148_214851


namespace NUMINAMATH_CALUDE_paving_cost_l2148_214834

/-- The cost of paving a rectangular floor given its dimensions and the rate per square meter. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 600) :
  length * width * rate = 12375 := by sorry

end NUMINAMATH_CALUDE_paving_cost_l2148_214834


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l2148_214845

theorem complex_magnitude_proof : Complex.abs (3/5 - 5/4 * Complex.I) = Real.sqrt 769 / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l2148_214845


namespace NUMINAMATH_CALUDE_parabola_cubic_intersection_l2148_214812

def parabola (x y : ℝ) : Prop := y = 3 * x^2 - 12 * x - 15

def cubic (x y : ℝ) : Prop := y = x^3 - 6 * x^2 + 11 * x - 6

def intersection_points : Set (ℝ × ℝ) := {(-1, 0), (1, -24), (9, 162)}

theorem parabola_cubic_intersection :
  ∀ x y : ℝ, (parabola x y ∧ cubic x y) ↔ (x, y) ∈ intersection_points :=
sorry

end NUMINAMATH_CALUDE_parabola_cubic_intersection_l2148_214812


namespace NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l2148_214847

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  3 * x - 4 * y - 23 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop :=
  4 * x + 3 * y + 1 = 0

-- Theorem statement
theorem perpendicular_bisector_and_parallel_line :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ∧
    (x - (A.1 + B.1) / 2) ^ 2 + (y - (A.2 + B.2) / 2) ^ 2 = 
    ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) / 4) ∧
  (∀ x y : ℝ, parallel_line x y ↔
    (y - P.2) * (B.1 - A.1) = (x - P.1) * (B.2 - A.2)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l2148_214847


namespace NUMINAMATH_CALUDE_triangle_base_length_l2148_214802

/-- Given a triangle with area 36 cm² and height 8 cm, its base length is 9 cm. -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 36 →
  height = 8 →
  area = (base * height) / 2 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2148_214802


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_l2148_214864

-- Define the conditions
def p (x : ℝ) : Prop := x ≤ 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Statement to prove
theorem neg_p_sufficient_not_necessary :
  (∀ x : ℝ, ¬(p x) → q x) ∧ ¬(∀ x : ℝ, q x → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_l2148_214864


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_l2148_214855

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (yellow : ℕ)
  (white : ℕ)
  (green : ℕ)

/-- The minimum number of gumballs needed to ensure obtaining four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : ℕ :=
  13

/-- Theorem stating that for the given gumball machine, 
    the minimum number of gumballs needed to ensure 
    obtaining four of the same color is 13 -/
theorem min_gumballs_for_four_same_color 
  (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.yellow = 6)
  (h3 : machine.white = 8)
  (h4 : machine.green = 9) :
  minGumballsForFourSameColor machine = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_l2148_214855


namespace NUMINAMATH_CALUDE_math_competition_scores_l2148_214869

/-- Represents the scoring system for a math competition. -/
structure ScoringSystem where
  num_questions : ℕ
  correct_points : ℕ
  no_answer_points : ℕ
  wrong_answer_deduction : ℕ

/-- Calculates the number of different possible scores for a given scoring system. -/
def num_different_scores (s : ScoringSystem) : ℕ :=
  sorry

/-- Theorem stating that for the given scoring system, there are 35 different possible scores. -/
theorem math_competition_scores :
  let s : ScoringSystem := {
    num_questions := 10,
    correct_points := 4,
    no_answer_points := 0,
    wrong_answer_deduction := 1
  }
  num_different_scores s = 35 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_scores_l2148_214869


namespace NUMINAMATH_CALUDE_no_geometric_progression_2_3_5_l2148_214836

theorem no_geometric_progression_2_3_5 : 
  ¬ (∃ (a r : ℝ) (k n : ℕ), 
    a > 0 ∧ r > 0 ∧ 
    a * r^0 = 2 ∧
    a * r^k = 3 ∧
    a * r^n = 5 ∧
    0 < k ∧ k < n) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_progression_2_3_5_l2148_214836


namespace NUMINAMATH_CALUDE_expression_evaluation_l2148_214876

theorem expression_evaluation : (-7)^3 / 7^2 + 4^3 - 5 * 2^2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2148_214876


namespace NUMINAMATH_CALUDE_exists_more_than_20_components_l2148_214811

/-- A diagonal in a cell can be either left-to-right or right-to-left -/
inductive Diagonal
| LeftToRight
| RightToLeft

/-- A grid is represented as a function from coordinates to diagonals -/
def Grid := Fin 8 → Fin 8 → Diagonal

/-- A point in the grid -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Two points are connected if there's a path of adjacent diagonals between them -/
def Connected (g : Grid) (p q : Point) : Prop := sorry

/-- A connected component is a maximal set of connected points -/
def ConnectedComponent (g : Grid) (s : Set Point) : Prop := sorry

/-- The number of connected components in a grid -/
def NumComponents (g : Grid) : ℕ := sorry

/-- There exists a configuration with more than 20 connected components -/
theorem exists_more_than_20_components : ∃ g : Grid, NumComponents g > 20 := by sorry

end NUMINAMATH_CALUDE_exists_more_than_20_components_l2148_214811


namespace NUMINAMATH_CALUDE_total_oranges_picked_l2148_214829

def monday_pick : ℕ := 100
def tuesday_pick : ℕ := 3 * monday_pick
def wednesday_pick : ℕ := 70

theorem total_oranges_picked : monday_pick + tuesday_pick + wednesday_pick = 470 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l2148_214829


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l2148_214819

theorem integer_fraction_characterization (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℤ) = k * (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1)) ↔
  (∃ l : ℕ+, (a = 2 * l ∧ b = 1) ∨
             (a = l ∧ b = 2 * l) ∨
             (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l2148_214819


namespace NUMINAMATH_CALUDE_triangle_area_sines_l2148_214884

theorem triangle_area_sines (a b c : ℝ) (h_a : a = 5) (h_b : b = 4 * Real.sqrt 2) (h_c : c = 7) :
  let R := (a * b * c) / (4 * Real.sqrt (((a + b + c)/2) * (((a + b + c)/2) - a) * (((a + b + c)/2) - b) * (((a + b + c)/2) - c)));
  let sin_A := a / (2 * R);
  let sin_B := b / (2 * R);
  let sin_C := c / (2 * R);
  let s := (sin_A + sin_B + sin_C) / 2;
  Real.sqrt (s * (s - sin_A) * (s - sin_B) * (s - sin_C)) = 7 / 25 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_sines_l2148_214884


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l2148_214852

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (4, -1)
def d : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance : 
  let line1 := fun (t : ℝ) => a + t • d
  let line2 := fun (s : ℝ) => b + s • d
  (∃ (p q : ℝ × ℝ), p ∈ Set.range line1 ∧ q ∈ Set.range line2 ∧ 
    ∀ (x y : ℝ × ℝ), x ∈ Set.range line1 → y ∈ Set.range line2 → 
      ‖p - q‖ ≤ ‖x - y‖) →
  ∃ (p q : ℝ × ℝ), p ∈ Set.range line1 ∧ q ∈ Set.range line2 ∧ 
    ‖p - q‖ = (5 * Real.sqrt 29) / 29 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l2148_214852


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l2148_214859

/-- Given real x, vectors a and b, with a parallel to b, 
    prove that the magnitude of their sum is √5 -/
theorem magnitude_of_sum (x : ℝ) (a b : ℝ × ℝ) :
  a = (x, 1) →
  b = (4, -2) →
  ∃ (k : ℝ), a = k • b →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l2148_214859


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l2148_214849

theorem sum_of_squares_divisible_by_seven (a b : ℤ) : 
  (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l2148_214849


namespace NUMINAMATH_CALUDE_triangle_sides_proof_l2148_214846

theorem triangle_sides_proof (a b c : ℝ) (h : ℝ) (x : ℝ) :
  b - c = 3 →
  h = 10 →
  (a / 2 + 6) - (a / 2 - 6) = 12 →
  a^2 = 427 / 3 ∧
  b = Real.sqrt (427 / 3) + 3 / 2 ∧
  c = Real.sqrt (427 / 3) - 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_proof_l2148_214846


namespace NUMINAMATH_CALUDE_triangle_translation_l2148_214853

structure Point where
  x : ℝ
  y : ℝ

def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem triangle_translation :
  let A : Point := ⟨2, 1⟩
  let B : Point := ⟨4, 3⟩
  let C : Point := ⟨0, 2⟩
  let A' : Point := ⟨-1, 5⟩
  let dx : ℝ := A'.x - A.x
  let dy : ℝ := A'.y - A.y
  let C' : Point := translate C dx dy
  C'.x = -3 ∧ C'.y = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_translation_l2148_214853


namespace NUMINAMATH_CALUDE_olympic_numbers_l2148_214858

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def all_digits_different (x y : ℕ) : Prop :=
  ∀ d, is_valid_digit d → (d ∈ x.digits 10 ↔ d ∉ y.digits 10)

theorem olympic_numbers :
  ∀ x y : ℕ,
    x < 1000 ∧ x ≥ 100 ∧  -- x is a three-digit number
    y < 10000 ∧ y ≥ 1000 ∧  -- y is a four-digit number
    (∀ d, d ∈ x.digits 10 → is_valid_digit d) ∧
    (∀ d, d ∈ y.digits 10 → is_valid_digit d) ∧
    all_digits_different x y ∧
    1 ∉ x.digits 10 ∧
    9 ∉ x.digits 10 ∧
    x / y = 1 / 9  -- Rational division
    →
    x = 163 ∨ x = 318 ∨ x = 729 ∨ x = 1638 ∨ x = 1647 :=
by sorry

end NUMINAMATH_CALUDE_olympic_numbers_l2148_214858


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2148_214828

/-- The distance between the foci of a hyperbola defined by x^2 - 2xy + y^2 = 2 is 4 -/
theorem hyperbola_foci_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - 2*x*y + y^2 = 2}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ hyperbola ∧ f₂ ∈ hyperbola ∧
    (∀ p ∈ hyperbola, dist p f₁ - dist p f₂ = 2 ∨ dist p f₂ - dist p f₁ = 2) ∧
    dist f₁ f₂ = 4 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2148_214828


namespace NUMINAMATH_CALUDE_farmer_horses_count_l2148_214838

/-- Calculates the number of horses a farmer owns based on hay production and consumption --/
def farmer_horses (last_year_bales : ℕ) (last_year_acres : ℕ) (additional_acres : ℕ) 
                  (bales_per_horse_per_day : ℕ) (remaining_bales : ℕ) : ℕ :=
  let total_acres := last_year_acres + additional_acres
  let bales_per_month := (last_year_bales / last_year_acres) * total_acres
  let feeding_months := 4  -- September to December
  let total_bales := bales_per_month * feeding_months + remaining_bales
  let feeding_days := 122  -- Total days from September 1st to December 31st
  let bales_per_horse := bales_per_horse_per_day * feeding_days
  total_bales / bales_per_horse

/-- Theorem stating the number of horses owned by the farmer --/
theorem farmer_horses_count : 
  farmer_horses 560 5 7 3 12834 = 49 := by
  sorry

end NUMINAMATH_CALUDE_farmer_horses_count_l2148_214838


namespace NUMINAMATH_CALUDE_rectangle_side_length_l2148_214844

theorem rectangle_side_length (a b c d : ℝ) : 
  a / c = 3 / 4 → 
  b / d = 3 / 4 → 
  c = 4 → 
  d = 8 → 
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l2148_214844


namespace NUMINAMATH_CALUDE_network_connections_l2148_214866

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  (n * k) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l2148_214866


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2148_214813

theorem nested_fraction_equality : 
  1 + 1 / (1 + 1 / (1 + 1 / 2)) = 8 / 5 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2148_214813


namespace NUMINAMATH_CALUDE_inequality_implication_l2148_214896

theorem inequality_implication (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19) :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implication_l2148_214896


namespace NUMINAMATH_CALUDE_system_solution_l2148_214824

theorem system_solution (x y : ℝ) 
  (eq1 : 2019 * x + 2020 * y = 2018)
  (eq2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2148_214824


namespace NUMINAMATH_CALUDE_dave_total_wage_l2148_214825

/-- Represents the daily wage information --/
structure DailyWage where
  hourly_rate : ℕ
  hours_worked : ℕ

/-- Calculates the total wage for a given day --/
def daily_total (dw : DailyWage) : ℕ :=
  dw.hourly_rate * dw.hours_worked

/-- Dave's wage information for Monday to Thursday --/
def dave_wages : List DailyWage := [
  ⟨6, 6⟩,  -- Monday
  ⟨7, 2⟩,  -- Tuesday
  ⟨9, 3⟩,  -- Wednesday
  ⟨8, 5⟩   -- Thursday
]

theorem dave_total_wage :
  (dave_wages.map daily_total).sum = 117 := by
  sorry

#eval (dave_wages.map daily_total).sum

end NUMINAMATH_CALUDE_dave_total_wage_l2148_214825


namespace NUMINAMATH_CALUDE_triangle_count_l2148_214868

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def vertices_per_triangle : ℕ := 3

theorem triangle_count :
  choose num_points vertices_per_triangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l2148_214868


namespace NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l2148_214832

/-- Two lines are symmetric about the y-axis if and only if their coefficients satisfy a specific relation -/
theorem lines_symmetric_about_y_axis 
  (a b c p q m : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hm : m ≠ 0) :
  (∃ (k : ℝ), k ≠ 0 ∧ -a = k*p ∧ b = k*q ∧ c = k*m) ↔ 
  (∀ (x y : ℝ), a*x + b*y + c = 0 ↔ p*(-x) + q*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l2148_214832


namespace NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_equals_three_sqrt_two_l2148_214862

theorem sqrt_eight_plus_sqrt_two_equals_three_sqrt_two : 
  Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_equals_three_sqrt_two_l2148_214862


namespace NUMINAMATH_CALUDE_division_problem_l2148_214863

theorem division_problem (x : ℝ) (h1 : 10 * x = 50) : 20 / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2148_214863


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l2148_214821

theorem sandwich_jam_cost (N B J : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 7 * J) = 276) : 
  (N * J * 7 : ℚ) / 100 = 0.14 * J := by
  sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l2148_214821


namespace NUMINAMATH_CALUDE_problem_solution_l2148_214837

theorem problem_solution (x y : ℝ) : 
  x + y = 150 ∧ 1.20 * y - 0.80 * x = 0.75 * (x + y) → x = 33.75 ∧ y = 116.25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2148_214837


namespace NUMINAMATH_CALUDE_q_contribution_l2148_214873

/-- Represents the contribution and time in the business for a partner -/
structure Partner where
  contribution : ℕ
  time : ℕ

/-- Calculates the weighted contribution of a partner -/
def weightedContribution (p : Partner) : ℕ := p.contribution * p.time

/-- Represents the business scenario -/
structure Business where
  p : Partner
  q : Partner
  profitRatio : Fraction

theorem q_contribution (b : Business) : b.q.contribution = 9000 :=
  sorry

end NUMINAMATH_CALUDE_q_contribution_l2148_214873


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l2148_214890

theorem quadratic_roots_nature (a : ℝ) (h : a < -1) :
  ∃ (x₁ x₂ : ℝ), 
    (a^3 + 1) * x₁^2 + (a^2 + 1) * x₁ - (a + 1) = 0 ∧
    (a^3 + 1) * x₂^2 + (a^2 + 1) * x₂ - (a + 1) = 0 ∧
    x₁ > 0 ∧ x₂ < 0 ∧ |x₂| < x₁ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l2148_214890


namespace NUMINAMATH_CALUDE_abc_inequality_l2148_214870

theorem abc_inequality (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a + b + c = 6) 
  (h4 : a * b + b * c + a * c = 9) : 
  0 < a ∧ a < 1 ∧ 1 < b ∧ b < 3 ∧ 3 < c ∧ c < 4 :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2148_214870


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_l2148_214894

theorem sqrt_six_times_sqrt_two : Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_l2148_214894


namespace NUMINAMATH_CALUDE_race_head_start_l2148_214899

/-- Represents the race scenario where A runs twice as fast as B -/
structure Race where
  speed_a : ℝ
  speed_b : ℝ
  course_length : ℝ
  head_start : ℝ
  speed_ratio : speed_a = 2 * speed_b
  course_length_value : course_length = 142

/-- Theorem stating that for the given race conditions, the head start must be 71 meters -/
theorem race_head_start (r : Race) : r.head_start = 71 := by
  sorry

#check race_head_start

end NUMINAMATH_CALUDE_race_head_start_l2148_214899


namespace NUMINAMATH_CALUDE_prob_two_spades_is_one_seventeenth_l2148_214886

/-- A standard deck of cards --/
structure Deck :=
  (total_cards : Nat)
  (spade_cards : Nat)
  (h_total : total_cards = 52)
  (h_spades : spade_cards = 13)

/-- The probability of drawing two spades as the first two cards --/
def prob_two_spades (d : Deck) : ℚ :=
  (d.spade_cards : ℚ) / d.total_cards * (d.spade_cards - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing two spades as the first two cards is 1/17 --/
theorem prob_two_spades_is_one_seventeenth (d : Deck) : prob_two_spades d = 1 / 17 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_spades_is_one_seventeenth_l2148_214886


namespace NUMINAMATH_CALUDE_cube_split_contains_2015_l2148_214891

def split_sum (m : ℕ) : ℕ := (m + 2) * (m - 1) / 2

theorem cube_split_contains_2015 (m : ℕ) (h1 : m > 1) :
  (split_sum m ≥ 1007) ∧ (split_sum (m - 1) < 1007) → m = 45 :=
sorry

end NUMINAMATH_CALUDE_cube_split_contains_2015_l2148_214891


namespace NUMINAMATH_CALUDE_water_intersection_points_l2148_214850

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents the water level in the cube -/
def waterLevel (c : Cube) (vol : ℝ) : ℝ :=
  vol * c.edgeLength

theorem water_intersection_points (c : Cube) (waterVol : ℝ) :
  c.edgeLength = 1 →
  waterVol = 5/6 →
  ∃ (x : ℝ), 
    0.26 < x ∧ x < 0.28 ∧ 
    0.72 < (1 - x) ∧ (1 - x) < 0.74 ∧
    (waterLevel c waterVol = x ∨ waterLevel c waterVol = 1 - x) := by
  sorry

#check water_intersection_points

end NUMINAMATH_CALUDE_water_intersection_points_l2148_214850


namespace NUMINAMATH_CALUDE_solution_set_correct_l2148_214888

def solution_set : Set ℝ := {1, 2, 3, 4, 5}

def equation (x : ℝ) : Prop :=
  (x^2 - 5*x + 5)^(x^2 - 9*x + 20) = 1

theorem solution_set_correct :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2148_214888


namespace NUMINAMATH_CALUDE_points_subtracted_per_wrong_answer_l2148_214889

theorem points_subtracted_per_wrong_answer
  (total_problems : ℕ)
  (total_score : ℕ)
  (points_per_correct : ℕ)
  (wrong_answers : ℕ)
  (h1 : total_problems = 25)
  (h2 : total_score = 85)
  (h3 : points_per_correct = 4)
  (h4 : wrong_answers = 3)
  : (total_problems * points_per_correct - total_score) / wrong_answers = 1 := by
  sorry

end NUMINAMATH_CALUDE_points_subtracted_per_wrong_answer_l2148_214889


namespace NUMINAMATH_CALUDE_max_a1_is_26_l2148_214848

/-- A sequence of positive integers satisfying the given conditions --/
def GoodSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ n, a (n + 1) ≤ a n + 5) ∧
  (∀ n, n ∣ a n)

/-- The maximum possible value of a₁ in a good sequence --/
def MaxA1 : ℕ := 26

/-- The theorem stating that the maximum possible value of a₁ in a good sequence is 26 --/
theorem max_a1_is_26 :
  (∃ a, GoodSequence a ∧ a 1 = MaxA1) ∧
  (∀ a, GoodSequence a → a 1 ≤ MaxA1) :=
sorry

end NUMINAMATH_CALUDE_max_a1_is_26_l2148_214848


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l2148_214827

/-- Given a container with alcohol and water, prove the ratio after adding water. -/
theorem alcohol_water_ratio 
  (initial_alcohol : ℚ) 
  (initial_water : ℚ) 
  (added_water : ℚ) 
  (h1 : initial_alcohol = 4) 
  (h2 : initial_water = 4) 
  (h3 : added_water = 2666666666666667 / 1000000000000000) : 
  (initial_alcohol / (initial_water + added_water)) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l2148_214827


namespace NUMINAMATH_CALUDE_sum_a_b_is_one_third_l2148_214854

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The theorem stating that a + b = 1/3 given the conditions -/
theorem sum_a_b_is_one_third
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + 3 * a + b)
  (h2 : IsEven f)
  (h3 : Set.Icc (a - 1) (2 * a) = Set.range f) :
  a + b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_is_one_third_l2148_214854


namespace NUMINAMATH_CALUDE_total_baseball_cards_l2148_214816

/-- The number of people with baseball cards -/
def num_people : ℕ := 6

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 52

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_baseball_cards : total_cards = 312 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l2148_214816


namespace NUMINAMATH_CALUDE_snowball_distribution_l2148_214857

/-- Represents the number of snowballs each person has -/
structure Snowballs :=
  (charlie : ℕ)
  (lucy : ℕ)
  (linus : ℕ)

/-- The initial state of snowballs -/
def initial_state : Snowballs :=
  { charlie := 19 + 31,  -- Lucy's snowballs + difference
    lucy := 19,
    linus := 0 }

/-- The final state after Charlie gives half his snowballs to Linus -/
def final_state : Snowballs :=
  { charlie := (19 + 31) / 2,
    lucy := 19,
    linus := (19 + 31) / 2 }

/-- Theorem stating the correct distribution of snowballs after sharing -/
theorem snowball_distribution :
  final_state.charlie = 25 ∧
  final_state.lucy = 19 ∧
  final_state.linus = 25 :=
by sorry

end NUMINAMATH_CALUDE_snowball_distribution_l2148_214857


namespace NUMINAMATH_CALUDE_pet_ownership_percentages_l2148_214887

def total_students : ℕ := 500
def dog_owners : ℕ := 125
def cat_owners : ℕ := 100
def rabbit_owners : ℕ := 50

def percent_dog_owners : ℚ := dog_owners / total_students * 100
def percent_cat_owners : ℚ := cat_owners / total_students * 100
def percent_rabbit_owners : ℚ := rabbit_owners / total_students * 100

theorem pet_ownership_percentages :
  percent_dog_owners = 25 ∧
  percent_cat_owners = 20 ∧
  percent_rabbit_owners = 10 :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_percentages_l2148_214887


namespace NUMINAMATH_CALUDE_mahi_share_l2148_214879

structure Friend where
  name : String
  age : ℕ
  distance : ℕ
  removed_amount : ℚ
  ratio : ℕ

def total_amount : ℚ := 2200

def friends : List Friend := [
  ⟨"Neha", 25, 5, 5, 2⟩,
  ⟨"Sabi", 32, 8, 8, 8⟩,
  ⟨"Mahi", 30, 7, 4, 6⟩,
  ⟨"Ravi", 28, 10, 6, 4⟩,
  ⟨"Priya", 35, 4, 10, 10⟩
]

def distance_bonus : ℚ := 10

theorem mahi_share (mahi : Friend) 
  (h1 : mahi ∈ friends)
  (h2 : mahi.name = "Mahi")
  (h3 : ∀ f : Friend, f ∈ friends → 
    f.age * (mahi.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + mahi.removed_amount + mahi.distance * distance_bonus) = 
    mahi.age * (f.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + f.removed_amount + f.distance * distance_bonus)) :
  mahi.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + mahi.removed_amount + mahi.distance * distance_bonus = 507.38 := by
  sorry

end NUMINAMATH_CALUDE_mahi_share_l2148_214879


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_numbers_l2148_214885

theorem largest_common_divisor_of_consecutive_odd_numbers (n : ℕ) :
  (n % 2 = 0 ∧ n > 0) →
  ∃ (k : ℕ), k = 45 ∧ 
    (∀ (m : ℕ), m ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → m ≤ k) ∧
    k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
by sorry


end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_numbers_l2148_214885


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2148_214822

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2148_214822


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l2148_214897

/-- The minimum distance between a point on the line y = 2 and a point on the circle (x - 1)² + y² = 1 is 1 -/
theorem min_distance_line_circle : 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = 2) → 
    ((Q.1 - 1)^2 + Q.2^2 = 1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_l2148_214897


namespace NUMINAMATH_CALUDE_james_birthday_stickers_l2148_214835

/-- The number of stickers James gets for his birthday -/
def birthday_stickers (initial_stickers total_stickers : ℕ) : ℕ :=
  total_stickers - initial_stickers

/-- Theorem: James got 22 stickers for his birthday -/
theorem james_birthday_stickers :
  birthday_stickers 39 61 = 22 := by
  sorry

end NUMINAMATH_CALUDE_james_birthday_stickers_l2148_214835


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2148_214874

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 30 and x = 3y, then when x = -6, y = -28.125 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
  (h1 : x * y = k)  -- x and y are inversely proportional
  (h2 : x + y = 30) -- sum condition
  (h3 : x = 3 * y)  -- x is three times y
  : x = -6 → y = -28.125 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2148_214874


namespace NUMINAMATH_CALUDE_interval_sum_l2148_214808

/-- The theorem states that for an interval [a, b] satisfying the given inequality,
    the sum of its endpoints is 12. -/
theorem interval_sum (a b : ℝ) : 
  (∀ x ∈ Set.Icc a b, |3*x - 80| ≤ |2*x - 105|) → a + b = 12 := by
  sorry

#check interval_sum

end NUMINAMATH_CALUDE_interval_sum_l2148_214808


namespace NUMINAMATH_CALUDE_angle_sum_bounds_l2148_214867

theorem angle_sum_bounds (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_sum_sin_sq : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ < 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_bounds_l2148_214867


namespace NUMINAMATH_CALUDE_football_tournament_max_points_l2148_214865

theorem football_tournament_max_points (n : ℕ) : 
  (∃ (scores : Fin 15 → ℕ), 
    (∀ i j : Fin 15, i ≠ j → scores i + scores j ≤ 3) ∧ 
    (∃ (successful : Finset (Fin 15)), 
      successful.card = 6 ∧ 
      ∀ i ∈ successful, n ≤ scores i)) →
  n ≤ 34 :=
sorry

end NUMINAMATH_CALUDE_football_tournament_max_points_l2148_214865


namespace NUMINAMATH_CALUDE_triangle_ratio_specific_l2148_214895

noncomputable def triangle_ratio (BC AC : ℝ) (angle_C : ℝ) : ℝ :=
  let AB := Real.sqrt (BC^2 + AC^2 - 2*BC*AC*(Real.cos angle_C))
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let AD := 2 * area / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  let AH := AD - BD / 2
  let HD := BD / 2
  AH / HD

theorem triangle_ratio_specific : 
  triangle_ratio 6 (3 * Real.sqrt 3) (π / 4) = (2 * Real.sqrt 6 - 4) / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_specific_l2148_214895


namespace NUMINAMATH_CALUDE_lcm_180_560_l2148_214810

theorem lcm_180_560 : Nat.lcm 180 560 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_lcm_180_560_l2148_214810


namespace NUMINAMATH_CALUDE_chessboard_ratio_sum_l2148_214881

/-- The number of rectangles formed on an 8x8 chessboard with 9 horizontal and 9 vertical lines -/
def total_rectangles : ℕ := 1296

/-- The number of squares formed on an 8x8 chessboard with 9 horizontal and 9 vertical lines -/
def total_squares : ℕ := 204

/-- The ratio of squares to rectangles as a simplified fraction -/
def square_rectangle_ratio : ℚ := total_squares / total_rectangles

theorem chessboard_ratio_sum :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ square_rectangle_ratio = m / n ∧ m + n = 125 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_ratio_sum_l2148_214881


namespace NUMINAMATH_CALUDE_subtract_largest_3digit_from_smallest_5digit_l2148_214805

def largest_3digit : ℕ := 999
def smallest_5digit : ℕ := 10000

theorem subtract_largest_3digit_from_smallest_5digit :
  smallest_5digit - largest_3digit = 9001 := by
  sorry

end NUMINAMATH_CALUDE_subtract_largest_3digit_from_smallest_5digit_l2148_214805


namespace NUMINAMATH_CALUDE_CaO_weight_calculation_l2148_214872

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of moles of CaO -/
def moles_CaO : ℝ := 7

/-- The molecular weight of CaO in g/mol -/
def molecular_weight_CaO : ℝ := calcium_weight + oxygen_weight

/-- The total weight of CaO in grams -/
def total_weight_CaO : ℝ := molecular_weight_CaO * moles_CaO

theorem CaO_weight_calculation : total_weight_CaO = 392.56 := by
  sorry

end NUMINAMATH_CALUDE_CaO_weight_calculation_l2148_214872


namespace NUMINAMATH_CALUDE_football_team_practice_hours_l2148_214871

/-- Given a football team's practice schedule, calculate the total practice hours in a week with one missed day. -/
theorem football_team_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 5 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 30 := by
sorry

end NUMINAMATH_CALUDE_football_team_practice_hours_l2148_214871


namespace NUMINAMATH_CALUDE_solve_sales_problem_l2148_214842

def sales_problem (m1 m2 m4 m5 m6 : ℕ) (average : ℚ) : Prop :=
  ∃ m3 : ℕ,
    (m1 + m2 + m3 + m4 + m5 + m6 : ℚ) / 6 = average ∧
    m3 = 5207

theorem solve_sales_problem :
  sales_problem 5124 5366 5399 6124 4579 (5400 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_solve_sales_problem_l2148_214842


namespace NUMINAMATH_CALUDE_min_time_30_seconds_l2148_214804

/-- Represents a person moving along the perimeter of a square -/
structure Person where
  start_position : ℕ  -- Starting vertex (0 = A, 1 = B, 2 = C, 3 = D)
  speed : ℕ           -- Speed in meters per second

/-- Calculates the minimum time for two people to be on the same side of a square -/
def min_time_same_side (side_length : ℕ) (person_a : Person) (person_b : Person) : ℕ :=
  sorry

/-- Theorem stating that the minimum time for the given scenario is 30 seconds -/
theorem min_time_30_seconds (side_length : ℕ) (person_a person_b : Person) :
  side_length = 50 ∧ 
  person_a = { start_position := 0, speed := 5 } ∧
  person_b = { start_position := 2, speed := 3 } →
  min_time_same_side side_length person_a person_b = 30 :=
sorry

end NUMINAMATH_CALUDE_min_time_30_seconds_l2148_214804


namespace NUMINAMATH_CALUDE_area_of_shaded_region_l2148_214807

/-- Given a square composed of 25 congruent smaller squares with a diagonal of 10 cm,
    the total area of all 25 squares is 50 square cm. -/
theorem area_of_shaded_region (diagonal : ℝ) (num_squares : ℕ) : 
  diagonal = 10 → num_squares = 25 → (diagonal^2 / 2) = 50 := by sorry

end NUMINAMATH_CALUDE_area_of_shaded_region_l2148_214807


namespace NUMINAMATH_CALUDE_wire_length_proof_l2148_214878

theorem wire_length_proof (part1 part2 total : ℕ) : 
  part1 = 106 →
  part2 = 74 →
  part1 = part2 + 32 →
  total = part1 + part2 →
  total = 180 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2148_214878


namespace NUMINAMATH_CALUDE_tourists_eq_scientific_l2148_214882

/-- Represents the number of domestic tourists during the "May Day" holiday in 2023 (in millions) -/
def tourists : ℝ := 274

/-- Represents the scientific notation of the number of tourists -/
def tourists_scientific : ℝ := 2.74 * (10 ^ 8)

/-- Theorem stating that the number of tourists in millions is equal to its scientific notation representation -/
theorem tourists_eq_scientific : tourists * (10 ^ 6) = tourists_scientific := by sorry

end NUMINAMATH_CALUDE_tourists_eq_scientific_l2148_214882


namespace NUMINAMATH_CALUDE_identity_polynomial_form_l2148_214809

/-- A polynomial that satisfies the given identity. -/
def IdentityPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * P (x - 1) = (x - 2) * P x

/-- The theorem stating the form of polynomials satisfying the identity. -/
theorem identity_polynomial_form (P : ℝ → ℝ) (h : IdentityPolynomial P) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^2 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_identity_polynomial_form_l2148_214809


namespace NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l2148_214877

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the absolute value of the coefficient for a given r
def coeff (r : ℕ) : ℕ := binomial 7 r

-- State the theorem
theorem largest_coefficient_in_expansion :
  ∀ r : ℕ, r ≤ 7 → coeff r ≤ coeff 4 :=
sorry

end NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l2148_214877


namespace NUMINAMATH_CALUDE_max_stores_visited_is_four_l2148_214841

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  total_visits : ℕ
  num_shoppers : ℕ
  double_visitors : ℕ
  max_stores_visited : ℕ

/-- The specific shopping scenario described in the problem -/
def town_scenario : ShoppingScenario :=
  { num_stores := 7
  , total_visits := 21
  , num_shoppers := 11
  , double_visitors := 7
  , max_stores_visited := 4 }

/-- Theorem stating that the maximum number of stores visited by any single person is 4 -/
theorem max_stores_visited_is_four (s : ShoppingScenario) 
  (h1 : s.num_stores = town_scenario.num_stores)
  (h2 : s.total_visits = town_scenario.total_visits)
  (h3 : s.num_shoppers = town_scenario.num_shoppers)
  (h4 : s.double_visitors = town_scenario.double_visitors)
  (h5 : s.double_visitors * 2 + (s.num_shoppers - s.double_visitors) ≤ s.total_visits) :
  s.max_stores_visited = town_scenario.max_stores_visited :=
by sorry


end NUMINAMATH_CALUDE_max_stores_visited_is_four_l2148_214841


namespace NUMINAMATH_CALUDE_third_quiz_score_l2148_214803

theorem third_quiz_score (score1 score2 score3 : ℕ) : 
  score1 = 91 → 
  score2 = 92 → 
  (score1 + score2 + score3) / 3 = 91 → 
  score3 = 90 := by
sorry

end NUMINAMATH_CALUDE_third_quiz_score_l2148_214803


namespace NUMINAMATH_CALUDE_coefficient_of_x4_l2148_214815

theorem coefficient_of_x4 (x : ℝ) : 
  let expr := 5*(x^4 - 2*x^5) + 3*(2*x^2 - x^6 + x^3) - (2*x^6 - 3*x^4 + x^2)
  ∃ (a b c d e f : ℝ), expr = 8*x^4 + a*x^6 + b*x^5 + c*x^3 + d*x^2 + e*x + f :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_l2148_214815
