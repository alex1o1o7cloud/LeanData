import Mathlib

namespace tower_surface_area_calculation_l3475_347590

def cube_surface_area (s : ℕ) : ℕ := 6 * s^2

def tower_surface_area (edge_lengths : List ℕ) : ℕ :=
  let n := edge_lengths.length
  edge_lengths.enum.foldl (fun acc (i, s) => 
    if i = 0 
    then acc + cube_surface_area s
    else acc + cube_surface_area s - s^2
  ) 0

theorem tower_surface_area_calculation :
  tower_surface_area [4, 5, 6, 7, 8, 9, 10] = 1871 :=
sorry

end tower_surface_area_calculation_l3475_347590


namespace problem_solution_l3475_347529

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem problem_solution :
  let a₁ := 5
  let d := 3
  let aₙ := 38
  let n := (aₙ - a₁) / d + 1
  let a := arithmetic_sum a₁ d n
  let b := sum_of_digits a
  let c := b ^ 2
  let d := c / 3
  d = 75 := by sorry

end problem_solution_l3475_347529


namespace p_sufficient_not_necessary_l3475_347578

-- Define the ratio between p and k
def ratio_p_k (p k : ℝ) : Prop := p / k = Real.sqrt 3

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop := 2 / Real.sqrt (k^2 + 1) = 1

-- Define the theorem
theorem p_sufficient_not_necessary (p q : ℝ) : 
  (∃ k, ratio_p_k p k ∧ is_tangent k) → 
  (∃ k, ratio_p_k q k ∧ is_tangent k) → 
  (∃ k, ratio_p_k p k ∧ is_tangent k → ∃ k', ratio_p_k q k' ∧ is_tangent k') ∧ 
  ¬(∀ k, ratio_p_k q k ∧ is_tangent k → ∃ k', ratio_p_k p k' ∧ is_tangent k') :=
sorry

end p_sufficient_not_necessary_l3475_347578


namespace cash_realized_specific_case_l3475_347591

/-- Given a total amount including brokerage and a brokerage rate,
    calculates the cash realized without brokerage -/
def cash_realized (total : ℚ) (brokerage_rate : ℚ) : ℚ :=
  total / (1 + brokerage_rate)

/-- Theorem stating that given the specific conditions of the problem,
    the cash realized is equal to 43200/401 -/
theorem cash_realized_specific_case :
  cash_realized 108 (1/400) = 43200/401 := by
  sorry

end cash_realized_specific_case_l3475_347591


namespace radio_listening_time_l3475_347520

/-- Calculates the time spent listening to the radio during a flight --/
theorem radio_listening_time (total_flight_time reading_time movie_time dinner_time game_time nap_time : ℕ) :
  total_flight_time = 680 →
  reading_time = 120 →
  movie_time = 240 →
  dinner_time = 30 →
  game_time = 70 →
  nap_time = 180 →
  total_flight_time - (reading_time + movie_time + dinner_time + game_time + nap_time) = 40 :=
by sorry

end radio_listening_time_l3475_347520


namespace dividend_calculation_l3475_347579

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 15)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 140 := by
  sorry

end dividend_calculation_l3475_347579


namespace sugar_calculation_l3475_347580

theorem sugar_calculation (initial_sugar : ℕ) (used_sugar : ℕ) (bought_sugar : ℕ) 
  (h1 : initial_sugar = 65)
  (h2 : used_sugar = 18)
  (h3 : bought_sugar = 50) :
  initial_sugar - used_sugar + bought_sugar = 97 := by
  sorry

end sugar_calculation_l3475_347580


namespace sin_to_cos_transformation_l3475_347587

theorem sin_to_cos_transformation (x : ℝ) :
  Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) =
  Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) := by
  sorry

end sin_to_cos_transformation_l3475_347587


namespace caravan_feet_heads_difference_l3475_347511

/-- Represents the number of feet for each animal type -/
def feet_per_animal : Nat → Nat
| 0 => 2  -- Hens
| 1 => 4  -- Goats
| 2 => 4  -- Camels
| 3 => 2  -- Keepers
| _ => 0  -- Other (shouldn't occur)

/-- Calculates the total number of feet for a given animal type and count -/
def total_feet (animal_type : Nat) (count : Nat) : Nat :=
  count * feet_per_animal animal_type

/-- Theorem: In a caravan with 60 hens, 35 goats, 6 camels, and 10 keepers,
    the difference between the total number of feet and the total number of heads is 193 -/
theorem caravan_feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let total_heads := hens + goats + camels + keepers
  let total_feet := total_feet 0 hens + total_feet 1 goats + total_feet 2 camels + total_feet 3 keepers
  total_feet - total_heads = 193 := by
  sorry

end caravan_feet_heads_difference_l3475_347511


namespace set_intersection_example_l3475_347559

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2, 4} := by
sorry

end set_intersection_example_l3475_347559


namespace square_tiles_problem_l3475_347526

/-- 
Given a square area tiled with congruent square tiles,
if the total number of tiles on the two diagonals is 25,
then the total number of tiles covering the entire square area is 169.
-/
theorem square_tiles_problem (n : ℕ) : 
  n > 0 → 
  2 * n - 1 = 25 → 
  n ^ 2 = 169 := by
  sorry

end square_tiles_problem_l3475_347526


namespace second_month_bill_l3475_347508

/-- Represents Elvin's telephone bill structure -/
structure TelephoneBill where
  internetCharge : ℝ
  callCharge : ℝ

/-- Calculates the total bill given internet and call charges -/
def totalBill (bill : TelephoneBill) : ℝ :=
  bill.internetCharge + bill.callCharge

theorem second_month_bill 
  (januaryBill : TelephoneBill) 
  (h1 : totalBill januaryBill = 40) 
  (secondMonthBill : TelephoneBill) 
  (h2 : secondMonthBill.internetCharge = januaryBill.internetCharge)
  (h3 : secondMonthBill.callCharge = 2 * januaryBill.callCharge) :
  totalBill secondMonthBill = 40 + januaryBill.callCharge := by
  sorry

#check second_month_bill

end second_month_bill_l3475_347508


namespace vector_sum_parallel_l3475_347564

theorem vector_sum_parallel (y : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2, y]
  (∃ (k : ℝ), a = k • b) →
  (a + 2 • b) = ![5, 10] := by
sorry

end vector_sum_parallel_l3475_347564


namespace book_purchase_change_l3475_347532

/-- Calculates the change received when buying two items with given prices and paying with a given amount. -/
def calculate_change (price1 : ℝ) (price2 : ℝ) (payment : ℝ) : ℝ :=
  payment - (price1 + price2)

/-- Theorem stating that buying two books priced at 5.5£ and 6.5£ with a 20£ bill results in 8£ change. -/
theorem book_purchase_change : calculate_change 5.5 6.5 20 = 8 := by
  sorry

end book_purchase_change_l3475_347532


namespace line_relationship_l3475_347502

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- Define the intersecting relationship between lines
variable (intersecting : Line → Line → Prop)

-- Define the skew relationship between lines
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem line_relationship (a b c : Line) 
  (h1 : parallel a c) 
  (h2 : ¬ parallel b c) : 
  intersecting a b ∨ skew a b :=
sorry

end line_relationship_l3475_347502


namespace average_length_is_10_over_3_l3475_347576

-- Define the lengths of the strings
def string1_length : ℚ := 2
def string2_length : ℚ := 5
def string3_length : ℚ := 3

-- Define the number of strings
def num_strings : ℕ := 3

-- Define the average length calculation
def average_length : ℚ := (string1_length + string2_length + string3_length) / num_strings

-- Theorem statement
theorem average_length_is_10_over_3 : average_length = 10 / 3 := by
  sorry

end average_length_is_10_over_3_l3475_347576


namespace a_gt_2_sufficient_not_necessary_l3475_347516

theorem a_gt_2_sufficient_not_necessary :
  (∀ a : ℝ, a > 2 → 2^a - a - 1 > 0) ∧
  (∃ a : ℝ, a ≤ 2 ∧ 2^a - a - 1 > 0) :=
by sorry

end a_gt_2_sufficient_not_necessary_l3475_347516


namespace gcd_lcm_sum_180_4620_l3475_347507

theorem gcd_lcm_sum_180_4620 : 
  Nat.gcd 180 4620 + Nat.lcm 180 4620 = 13920 := by
  sorry

end gcd_lcm_sum_180_4620_l3475_347507


namespace z_ninth_power_l3475_347592

theorem z_ninth_power (z : ℂ) : z = (-Real.sqrt 3 + Complex.I) / 2 → z^9 = -Complex.I := by
  sorry

end z_ninth_power_l3475_347592


namespace line_equation_from_slope_and_intercept_l3475_347522

/-- Given a line with slope 2 and y-intercept -3, its equation is 2x - y - 3 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (x y : ℝ), 
    (∃ (m b : ℝ), m = 2 ∧ b = -3 ∧ y = m * x + b) →
    2 * x - y - 3 = 0 :=
by sorry

end line_equation_from_slope_and_intercept_l3475_347522


namespace absolute_value_equation_solution_l3475_347595

theorem absolute_value_equation_solution : 
  ∃ (x : ℝ), (|x - 3| = 5 - 2*x) ∧ (x = 8/3 ∨ x = 2) := by
  sorry

end absolute_value_equation_solution_l3475_347595


namespace rachel_math_problems_l3475_347557

theorem rachel_math_problems (minutes_before_bed : ℕ) (problems_next_day : ℕ) (total_problems : ℕ)
  (h1 : minutes_before_bed = 12)
  (h2 : problems_next_day = 16)
  (h3 : total_problems = 76) :
  ∃ (problems_per_minute : ℕ),
    problems_per_minute * minutes_before_bed + problems_next_day = total_problems ∧
    problems_per_minute = 5 := by
  sorry

end rachel_math_problems_l3475_347557


namespace centroid_altitude_distance_l3475_347550

/-- Triangle XYZ with sides a, b, c and centroid G -/
structure Triangle where
  a : ℝ  -- side XY
  b : ℝ  -- side XZ
  c : ℝ  -- side YZ
  G : ℝ × ℝ  -- centroid coordinates

/-- The foot of the altitude from a point to a line segment -/
def altitudeFoot (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In a triangle with sides 12, 15, and 23, the distance from the centroid
    to the foot of the altitude from the centroid to the longest side is 40/23 -/
theorem centroid_altitude_distance (t : Triangle) 
    (h1 : t.a = 12) (h2 : t.b = 15) (h3 : t.c = 23) : 
    let Q := altitudeFoot t.G (⟨0, 0⟩, ⟨t.c, 0⟩)  -- Assuming YZ is on x-axis
    distance t.G Q = 40 / 23 := by
  sorry

end centroid_altitude_distance_l3475_347550


namespace zero_neither_positive_nor_negative_l3475_347577

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by
  sorry

#check zero_neither_positive_nor_negative

end zero_neither_positive_nor_negative_l3475_347577


namespace hyperbola_sum_l3475_347515

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -4 ∧ 
  c = Real.sqrt 53 ∧ 
  a = 4 ∧ 
  c^2 = a^2 + b^2 → 
  h + k + a + b = 3 + Real.sqrt 37 := by
sorry

end hyperbola_sum_l3475_347515


namespace triangle_passing_theorem_l3475_347568

/-- A triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- Whether a triangle can pass through another triangle -/
def can_pass_through (t1 t2 : Triangle) : Prop := sorry

theorem triangle_passing_theorem (T Q : Triangle) 
  (h_T_area : Triangle.area T < 4)
  (h_Q_area : Triangle.area Q = 3) :
  can_pass_through T Q := by sorry

end triangle_passing_theorem_l3475_347568


namespace friday_zoo_visitors_l3475_347569

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := 3750

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := saturday_visitors / 3

/-- Theorem stating that 1250 people visited the zoo on Friday -/
theorem friday_zoo_visitors : friday_visitors = 1250 := by
  sorry

end friday_zoo_visitors_l3475_347569


namespace root_sum_sixth_power_l3475_347566

theorem root_sum_sixth_power (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r ≠ s →
  r^6 + s^6 = 970 := by
sorry

end root_sum_sixth_power_l3475_347566


namespace original_workers_is_seven_l3475_347549

/-- Represents the work scenario described in the problem -/
structure WorkScenario where
  planned_days : ℕ
  absent_workers : ℕ
  actual_days : ℕ

/-- Calculates the original number of workers given a work scenario -/
def original_workers (scenario : WorkScenario) : ℕ :=
  (scenario.absent_workers * scenario.actual_days) / (scenario.actual_days - scenario.planned_days)

/-- The specific work scenario from the problem -/
def problem_scenario : WorkScenario :=
  { planned_days := 8
  , absent_workers := 3
  , actual_days := 14 }

/-- Theorem stating that the original number of workers in the problem scenario is 7 -/
theorem original_workers_is_seven :
  original_workers problem_scenario = 7 := by
  sorry

end original_workers_is_seven_l3475_347549


namespace candy_mixture_problem_l3475_347542

/-- Given two types of candy mixed to produce a specific mixture, 
    prove the amount of the second type of candy. -/
theorem candy_mixture_problem (X Y : ℝ) : 
  X + Y = 10 →
  3.50 * X + 4.30 * Y = 40 →
  Y = 6.25 := by
  sorry

end candy_mixture_problem_l3475_347542


namespace caleb_spent_66_50_l3475_347501

/-- The total amount spent on hamburgers -/
def total_spent (total_burgers : ℕ) (single_cost double_cost : ℚ) (double_count : ℕ) : ℚ :=
  let single_count := total_burgers - double_count
  double_count * double_cost + single_count * single_cost

/-- Theorem stating that Caleb spent $66.50 on hamburgers -/
theorem caleb_spent_66_50 :
  total_spent 50 1 (3/2) 33 = 133/2 := by
  sorry

end caleb_spent_66_50_l3475_347501


namespace election_votes_l3475_347574

theorem election_votes (total_votes : ℕ) 
  (winning_percentage : ℚ) (vote_majority : ℕ) : 
  winning_percentage = 70 / 100 → 
  vote_majority = 160 → 
  (winning_percentage * total_votes : ℚ) - 
    ((1 - winning_percentage) * total_votes : ℚ) = vote_majority → 
  total_votes = 400 := by
sorry

end election_votes_l3475_347574


namespace salary_changes_l3475_347506

theorem salary_changes (S : ℝ) (S_pos : S > 0) :
  S * (1 + 0.3) * (1 - 0.2) * (1 + 0.1) * (1 - 0.25) = S * 1.04 := by
  sorry

end salary_changes_l3475_347506


namespace max_knights_and_courtiers_l3475_347539

/-- Represents the number of people at each table -/
structure TableCounts where
  king : ℕ
  courtiers : ℕ
  knights : ℕ

/-- Checks if the table counts are valid according to the problem constraints -/
def is_valid_table_counts (tc : TableCounts) : Prop :=
  tc.king = 7 ∧ 
  12 ≤ tc.courtiers ∧ tc.courtiers ≤ 18 ∧
  10 ≤ tc.knights ∧ tc.knights ≤ 20

/-- The rule that the sum of a knight's portion and a courtier's portion equals the king's portion -/
def satisfies_portion_rule (tc : TableCounts) : Prop :=
  (1 : ℚ) / tc.courtiers + (1 : ℚ) / tc.knights = (1 : ℚ) / tc.king

/-- The main theorem stating the maximum number of knights and corresponding courtiers -/
theorem max_knights_and_courtiers :
  ∃ (tc : TableCounts), 
    is_valid_table_counts tc ∧ 
    satisfies_portion_rule tc ∧
    tc.knights = 14 ∧ 
    tc.courtiers = 14 ∧
    (∀ (tc' : TableCounts), 
      is_valid_table_counts tc' ∧ 
      satisfies_portion_rule tc' → 
      tc'.knights ≤ tc.knights) :=
by sorry

end max_knights_and_courtiers_l3475_347539


namespace centroid_line_intersection_l3475_347594

/-- Given a triangle ABC with centroid G, and a line through G intersecting AB at M and AC at N,
    where AM = x * AB and AN = y * AC, prove that 1/x + 1/y = 3 -/
theorem centroid_line_intersection (A B C G M N : ℝ × ℝ) (x y : ℝ) :
  (G = (1/3 : ℝ) • (A + B + C)) →  -- G is the centroid
  (∃ (t : ℝ), M = A + t • (G - A) ∧ N = A + t • (G - A)) →  -- M and N are on the line through G
  (M = A + x • (B - A)) →  -- AM = x * AB
  (N = A + y • (C - A)) →  -- AN = y * AC
  (1 / x + 1 / y = 3) :=
by sorry

end centroid_line_intersection_l3475_347594


namespace integer_pairs_satisfying_equation_l3475_347524

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | x * (x + 1) * (x + 2) * (x + 3) = y * (y + 1)} = 
  {(0, 0), (-1, 0), (-2, 0), (-3, 0), (0, -1), (-1, -1), (-2, -1), (-3, -1)} := by
  sorry

end integer_pairs_satisfying_equation_l3475_347524


namespace loss_percentage_is_five_percent_l3475_347533

def original_price : ℚ := 490
def sale_price : ℚ := 465.5

def loss_amount : ℚ := original_price - sale_price

def loss_percentage : ℚ := (loss_amount / original_price) * 100

theorem loss_percentage_is_five_percent :
  loss_percentage = 5 := by
  sorry

end loss_percentage_is_five_percent_l3475_347533


namespace six_by_six_grid_shaded_half_l3475_347548

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)
  (shaded_per_row : ℕ)

/-- Calculates the percentage of shaded area in a grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (g.size * g.shaded_per_row : ℚ) / (g.size * g.size)

/-- The main theorem: for a 6x6 grid with 3 shaded squares per row,
    the shaded percentage is 50% -/
theorem six_by_six_grid_shaded_half :
  let g : Grid := { size := 6, shaded_per_row := 3 }
  shaded_percentage g = 1/2 := by sorry

end six_by_six_grid_shaded_half_l3475_347548


namespace power_function_property_l3475_347599

/-- A power function is a function of the form f(x) = x^a for some real number a -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

theorem power_function_property (f : ℝ → ℝ) (h1 : IsPowerFunction f) (h2 : f 4 = 2) :
  f (1/4) = 1/2 := by
  sorry

end power_function_property_l3475_347599


namespace dilution_proof_l3475_347519

/-- Given a solution of 12 ounces with 60% alcohol concentration, 
    adding 6 ounces of water results in a 40% alcohol solution -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
                       (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧ 
  water_added = 6 ∧ 
  final_concentration = 0.4 → 
  initial_volume * initial_concentration = 
  (initial_volume + water_added) * final_concentration := by
  sorry

#check dilution_proof

end dilution_proof_l3475_347519


namespace max_non_managers_l3475_347555

/-- The maximum number of non-managers in a department with 9 managers, 
    given that the ratio of managers to non-managers must be greater than 7:37 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 37 →
  non_managers ≤ 47 :=
by sorry

end max_non_managers_l3475_347555


namespace geometric_sequence_first_term_l3475_347589

theorem geometric_sequence_first_term (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
  a = 1/4 := by
sorry

end geometric_sequence_first_term_l3475_347589


namespace peas_soybean_mixture_ratio_l3475_347517

/-- Proves that the ratio of peas to soybean in a mixture costing Rs. 19/kg is 2:1,
    given that peas cost Rs. 16/kg and soybean costs Rs. 25/kg. -/
theorem peas_soybean_mixture_ratio : 
  ∀ (x y : ℝ), 
    x > 0 → y > 0 →
    16 * x + 25 * y = 19 * (x + y) →
    x / y = 2 := by
  sorry

end peas_soybean_mixture_ratio_l3475_347517


namespace fraction_transformation_l3475_347572

theorem fraction_transformation (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 8 : ℚ) / (d + 8) = (1 : ℚ) / 3 →
  d = 25 := by
sorry

end fraction_transformation_l3475_347572


namespace real_roots_iff_k_geq_quarter_l3475_347571

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ :=
  (k - 1)^2 * x^2 + (2*k + 1) * x + 1

-- Theorem statement
theorem real_roots_iff_k_geq_quarter :
  ∀ k : ℝ, (∃ x : ℝ, quadratic_equation k x = 0) ↔ k ≥ 1/4 := by
  sorry

end real_roots_iff_k_geq_quarter_l3475_347571


namespace equation_solution_l3475_347598

theorem equation_solution (x y : ℚ) 
  (eq1 : 3 * x + y = 6) 
  (eq2 : x + 3 * y = 8) : 
  9 * x^2 + 15 * x * y + 9 * y^2 = 1629 / 16 := by
sorry

end equation_solution_l3475_347598


namespace altitude_construction_possible_l3475_347521

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a triangle in a plane -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a construction step -/
inductive ConstructionStep
  | DrawLine (p1 p2 : Point)
  | DrawCircle (center : Point) (through : Point)
  | MarkPoints (points : List Point)

/-- Represents an erasing step -/
structure EraseStep :=
  (points : List Point)

/-- Function to check if a triangle is acute-angled and non-equilateral -/
def isAcuteNonEquilateral (t : Triangle) : Prop := sorry

/-- Function to check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Function to construct an altitude of a triangle -/
def constructAltitude (t : Triangle) (v : Point) : Line := sorry

/-- Theorem stating that altitudes can be constructed despite point erasure -/
theorem altitude_construction_possible 
  (t : Triangle) 
  (h_acute : isAcuteNonEquilateral t) :
  ∃ (steps : List ConstructionStep),
    ∀ (erases : List EraseStep),
      (∀ e ∈ erases, e.points.length ≤ 3) →
      ∃ (a b c : Line),
        isPointOnLine t.a a ∧ 
        isPointOnLine t.b b ∧ 
        isPointOnLine t.c c ∧
        a = constructAltitude t t.a ∧
        b = constructAltitude t t.b ∧
        c = constructAltitude t t.c :=
sorry

end altitude_construction_possible_l3475_347521


namespace water_in_pool_is_34_l3475_347536

/-- Calculates the amount of water in Carol's pool after five hours of filling and leaking -/
def water_in_pool : ℕ :=
  let first_hour : ℕ := 8
  let next_two_hours : ℕ := 10 * 2
  let fourth_hour : ℕ := 14
  let leak : ℕ := 8
  (first_hour + next_two_hours + fourth_hour) - leak

/-- Theorem stating that the amount of water in the pool after five hours is 34 gallons -/
theorem water_in_pool_is_34 : water_in_pool = 34 := by
  sorry

end water_in_pool_is_34_l3475_347536


namespace min_sum_m_n_l3475_347593

theorem min_sum_m_n (m n : ℕ+) (h : 300 * m = n^3) : 
  ∀ (m' n' : ℕ+), 300 * m' = n'^3 → m + n ≤ m' + n' :=
by
  sorry

end min_sum_m_n_l3475_347593


namespace expression_simplification_and_evaluation_l3475_347504

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) ∧
  (3 + 1 - 3 / (3 - 1)) / ((3^2 - 4*3 + 4) / (3 - 1)) = 5 := by
  sorry

end expression_simplification_and_evaluation_l3475_347504


namespace original_average_age_proof_l3475_347525

theorem original_average_age_proof (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 ∧
  new_students = 12 ∧
  new_avg = 32 ∧
  avg_decrease = 6 →
  original_avg = 40 := by
sorry

end original_average_age_proof_l3475_347525


namespace min_a_for_inequality_l3475_347570

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x > a, 2 * x + 3 ≥ 7) ↔ a < 2 := by sorry

end min_a_for_inequality_l3475_347570


namespace no_function_satisfies_inequality_l3475_347503

/-- There does not exist a function satisfying the given inequality for all real numbers. -/
theorem no_function_satisfies_inequality :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| := by
  sorry

end no_function_satisfies_inequality_l3475_347503


namespace sequence_gcd_property_l3475_347534

theorem sequence_gcd_property :
  (¬∃(a : ℕ → ℕ), ∀i j, i < j → Nat.gcd (a i + j) (a j + i) = 1) ∧
  (∀p, Prime p ∧ Odd p → ∃(a : ℕ → ℕ), ∀i j, i < j → ¬(p ∣ Nat.gcd (a i + j) (a j + i))) :=
by sorry

end sequence_gcd_property_l3475_347534


namespace remainder_N_mod_45_l3475_347513

def N : ℕ := sorry

theorem remainder_N_mod_45 : N % 45 = 9 := by
  sorry

end remainder_N_mod_45_l3475_347513


namespace ruby_candy_sharing_l3475_347582

theorem ruby_candy_sharing (total_candies : ℕ) (candies_per_friend : ℕ) 
  (h1 : total_candies = 36)
  (h2 : candies_per_friend = 4) :
  total_candies / candies_per_friend = 9 := by
  sorry

end ruby_candy_sharing_l3475_347582


namespace hyperbola_center_correct_l3475_347547

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y + 8)^2 / 16^2 - (5 * x - 15)^2 / 9^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, -2)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem hyperbola_center_correct :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
by sorry

end hyperbola_center_correct_l3475_347547


namespace isosceles_triangle_base_length_l3475_347510

/-- An isosceles triangle with a median to the leg dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  /-- Length of the leg of the isosceles triangle -/
  leg : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : leg > 0
  /-- The median to the leg divides the perimeter into two parts -/
  medianDivides : leg + leg + base = 27
  /-- One part of the divided perimeter is 15 -/
  part1 : leg + leg / 2 = 15 ∨ leg / 2 + base = 15

/-- The theorem stating the possible base lengths of the isosceles triangle -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangleWithMedian) :
  t.base = 7 ∨ t.base = 11 := by
  sorry

end isosceles_triangle_base_length_l3475_347510


namespace negation_of_universal_proposition_l3475_347541

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by sorry

end negation_of_universal_proposition_l3475_347541


namespace rachel_removed_bottle_caps_l3475_347554

/-- The number of bottle caps Rachel removed from a jar --/
def bottleCapsRemoved (originalCount remainingCount : ℕ) : ℕ :=
  originalCount - remainingCount

/-- Theorem: The number of bottle caps Rachel removed is equal to the difference
    between the original number and the remaining number of bottle caps --/
theorem rachel_removed_bottle_caps :
  bottleCapsRemoved 87 40 = 47 := by
  sorry

end rachel_removed_bottle_caps_l3475_347554


namespace max_value_expression_l3475_347514

theorem max_value_expression (x k : ℕ) (hx : x > 0) (hk : k > 0) : 
  let y := k * x
  ∃ (max : ℚ), max = 2 ∧ ∀ (x' k' : ℕ), x' > 0 → k' > 0 → 
    let y' := k' * x'
    (x' + y')^2 / (x'^2 + y'^2 : ℚ) ≤ max :=
by sorry

end max_value_expression_l3475_347514


namespace abs_diff_of_abs_l3475_347553

theorem abs_diff_of_abs : ∀ a b : ℝ, 
  (abs a = 3 ∧ abs b = 5) → abs (abs (a + b) - abs (a - b)) = 6 := by
  sorry

end abs_diff_of_abs_l3475_347553


namespace race_time_calculation_l3475_347531

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  time_difference : ℝ
  distance_difference : ℝ

/-- The theorem to prove -/
theorem race_time_calculation (race : Race) 
  (h1 : race.distance = 1000)
  (h2 : race.time_difference = 10)
  (h3 : race.distance_difference = 20) :
  ∃ (t : ℝ), t = 490 ∧ t * race.runner_a.speed = race.distance :=
sorry

end race_time_calculation_l3475_347531


namespace unique_a_with_prime_roots_l3475_347512

theorem unique_a_with_prime_roots : ∃! a : ℕ+, 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (2 : ℝ) * p^2 - 30 * p + (a : ℝ) = 0 ∧
  (2 : ℝ) * q^2 - 30 * q + (a : ℝ) = 0 ∧
  a = 52 := by
sorry

end unique_a_with_prime_roots_l3475_347512


namespace circle_and_tangent_lines_l3475_347565

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A, B, and M
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (-5, 5)
def M : ℝ × ℝ := (-2, 9)

-- Define the line l: x + y + 3 = 0
def l (p : ℝ × ℝ) : Prop := p.1 + p.2 + 3 = 0

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    -- C lies on line l
    l C ∧
    -- Circle passes through A and B
    A ∈ Circle C r ∧ B ∈ Circle C r ∧
    -- Standard equation of the circle
    (∀ (x y : ℝ), (x, y) ∈ Circle C r ↔ (x + 5)^2 + (y - 2)^2 = 9) ∧
    -- Tangent lines through M
    (∀ (x y : ℝ),
      ((x = -2) ∨ (20 * x - 21 * y + 229 = 0)) ↔
      ((x, y) ∈ Circle C r → (x - M.1) * (x - C.1) + (y - M.2) * (y - C.2) = 0)) :=
sorry

end circle_and_tangent_lines_l3475_347565


namespace hyperbola_standard_equation_l3475_347583

/-- A hyperbola is defined by its standard equation and properties -/
structure Hyperbola where
  /-- The standard equation of the hyperbola: y²/a² - x²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The hyperbola passes through a given point -/
  passes_through : ℝ × ℝ → Prop
  /-- The asymptotic equations of the hyperbola -/
  asymptotic_equations : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop)

/-- Theorem: Given a hyperbola that passes through (√3, 4) with asymptotic equations 2x ± y = 0,
    its standard equation is y²/4 - x² = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  h.passes_through (Real.sqrt 3, 4) ∧
  h.asymptotic_equations = ((fun x y => 2*x = y), (fun x y => 2*x = -y)) →
  h.equation = fun x y => y^2/4 - x^2 = 1 := by
  sorry


end hyperbola_standard_equation_l3475_347583


namespace cubic_equation_root_problem_l3475_347581

theorem cubic_equation_root_problem (c d : ℚ) : 
  (∃ x : ℝ, x^3 + c*x^2 + d*x + 15 = 0 ∧ x = 3 + Real.sqrt 5) → d = -37/2 :=
by sorry

end cubic_equation_root_problem_l3475_347581


namespace flour_bags_theorem_l3475_347556

def measurements : List Int := [3, 1, 0, 2, 6, -1, 2, 1, -4, 1]

def standard_weight : Int := 100

theorem flour_bags_theorem (measurements : List Int) (standard_weight : Int) :
  measurements = [3, 1, 0, 2, 6, -1, 2, 1, -4, 1] →
  standard_weight = 100 →
  (∀ m ∈ measurements, |0| ≤ |m|) ∧
  (measurements.sum = 11) ∧
  (measurements.length * standard_weight + measurements.sum = 1011) :=
by sorry

end flour_bags_theorem_l3475_347556


namespace cube_root_equivalence_l3475_347563

theorem cube_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/4))^(1/3) = x^(3/4) := by sorry

end cube_root_equivalence_l3475_347563


namespace regression_line_equation_l3475_347528

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation of the form y = mx + b -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Given a regression line with slope 1.2 passing through (4,5), prove its equation is ŷ = 1.2x + 0.2 -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point)
  (h1 : slope = 1.2)
  (h2 : center = ⟨4, 5⟩)
  : ∃ (eq : LinearEquation), 
    eq.slope = slope ∧ 
    eq.intercept = 0.2 ∧ 
    center.y = eq.slope * center.x + eq.intercept := by
  sorry

end regression_line_equation_l3475_347528


namespace smallest_argument_in_circle_l3475_347573

theorem smallest_argument_in_circle (p : ℂ) : 
  (Complex.abs (p - 25 * Complex.I) ≤ 15) →
  Complex.arg p ≥ Complex.arg (12 + 16 * Complex.I) :=
by sorry

end smallest_argument_in_circle_l3475_347573


namespace outlet_pipe_emptying_time_l3475_347505

theorem outlet_pipe_emptying_time 
  (fill_time_1 : ℝ) 
  (fill_time_2 : ℝ) 
  (combined_fill_time : ℝ) 
  (h1 : fill_time_1 = 18) 
  (h2 : fill_time_2 = 30) 
  (h3 : combined_fill_time = 0.06666666666666665) :
  let fill_rate_1 := 1 / fill_time_1
  let fill_rate_2 := 1 / fill_time_2
  let combined_fill_rate := 1 / combined_fill_time
  ∃ (empty_time : ℝ), 
    fill_rate_1 + fill_rate_2 - (1 / empty_time) = combined_fill_rate ∧ 
    empty_time = 45 :=
by sorry

end outlet_pipe_emptying_time_l3475_347505


namespace mat_equation_solution_l3475_347543

theorem mat_equation_solution :
  ∃! x : ℝ, (589 + x) + (544 - x) + 80 * x = 2013 := by
  sorry

end mat_equation_solution_l3475_347543


namespace jenny_meal_combinations_l3475_347584

/-- Represents the number of choices for each meal component -/
structure MealChoices where
  mainDishes : Nat
  drinks : Nat
  desserts : Nat
  sideDishes : Nat

/-- Calculates the total number of possible meal combinations -/
def totalMealCombinations (choices : MealChoices) : Nat :=
  choices.mainDishes * choices.drinks * choices.desserts * choices.sideDishes

/-- Theorem stating that Jenny can arrange 48 distinct possible meals -/
theorem jenny_meal_combinations :
  let jennyChoices : MealChoices := {
    mainDishes := 4,
    drinks := 2,
    desserts := 2,
    sideDishes := 3
  }
  totalMealCombinations jennyChoices = 48 := by
  sorry

end jenny_meal_combinations_l3475_347584


namespace second_week_collection_l3475_347509

def total_goal : ℕ := 500
def first_week : ℕ := 158
def cans_needed : ℕ := 83

theorem second_week_collection : 
  total_goal - first_week - cans_needed = 259 := by
  sorry

end second_week_collection_l3475_347509


namespace digit_product_over_21_l3475_347551

theorem digit_product_over_21 (c d : ℕ) : 
  (c < 10 ∧ d < 10) → -- c and d are base-10 digits
  (7 * 7 * 7 + 6 * 7 + 5 = 400 + 10 * c + d) → -- 765₇ = 4cd₁₀
  (c * d : ℚ) / 21 = 9 / 7 := by
  sorry

end digit_product_over_21_l3475_347551


namespace inequality_solution_set_l3475_347552

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | a * x - 1 > 0} = {x : ℝ | x < 1 / a} :=
by sorry

end inequality_solution_set_l3475_347552


namespace expression_equality_l3475_347560

theorem expression_equality : 
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 + 
  (2013^2 * 2014 - 2015) / Nat.factorial 2014 = 
  1 / Nat.factorial 2009 + 1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 1 / Nat.factorial 2014 := by
  sorry

end expression_equality_l3475_347560


namespace power_sum_problem_l3475_347537

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : 2*a*x + 3*b*y = 6)
  (h2 : 2*a*x^2 + 3*b*y^2 = 14)
  (h3 : 2*a*x^3 + 3*b*y^3 = 33)
  (h4 : 2*a*x^4 + 3*b*y^4 = 87) :
  2*a*x^5 + 3*b*y^5 = 528 := by
sorry

end power_sum_problem_l3475_347537


namespace power_difference_l3475_347597

theorem power_difference (a m n : ℝ) (hm : a^m = 9) (hn : a^n = 3) :
  a^(m - n) = 3 := by
  sorry

end power_difference_l3475_347597


namespace cost_of_gums_in_dollars_l3475_347561

-- Define the cost of one piece of gum in cents
def cost_of_one_gum : ℕ := 2

-- Define the number of pieces of gum
def number_of_gums : ℕ := 500

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_gums_in_dollars : 
  (number_of_gums * cost_of_one_gum) / cents_per_dollar = 10 := by
  sorry


end cost_of_gums_in_dollars_l3475_347561


namespace susan_remaining_money_l3475_347544

def susan_fair_spending (initial_amount food_cost : ℕ) : ℕ :=
  let game_cost := 3 * food_cost
  let total_spent := food_cost + game_cost
  initial_amount - total_spent

theorem susan_remaining_money :
  susan_fair_spending 90 20 = 10 := by
  sorry

end susan_remaining_money_l3475_347544


namespace factorial_fraction_simplification_l3475_347535

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end factorial_fraction_simplification_l3475_347535


namespace chord_intersection_diameter_segments_l3475_347527

theorem chord_intersection_diameter_segments (r : ℝ) (chord_length : ℝ) : 
  r = 6 → chord_length = 10 → ∃ (s₁ s₂ : ℝ), s₁ = 6 - Real.sqrt 11 ∧ s₂ = 6 + Real.sqrt 11 ∧ s₁ + s₂ = 2 * r :=
by sorry

end chord_intersection_diameter_segments_l3475_347527


namespace doctor_team_formations_l3475_347546

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem doctor_team_formations :
  let total_doctors : ℕ := 9
  let male_doctors : ℕ := 5
  let female_doctors : ℕ := 4
  let team_size : ℕ := 3
  let one_male_two_female : ℕ := choose male_doctors 1 * choose female_doctors 2
  let two_male_one_female : ℕ := choose male_doctors 2 * choose female_doctors 1
  one_male_two_female + two_male_one_female = 70 :=
sorry

end doctor_team_formations_l3475_347546


namespace carol_distance_behind_anna_l3475_347567

/-- Represents the position of a runner in a race -/
structure Position :=
  (distance : ℝ)

/-- Represents a runner in the race -/
structure Runner :=
  (speed : ℝ)
  (position : Position)

/-- The race setup -/
structure Race :=
  (length : ℝ)
  (anna : Runner)
  (bridgit : Runner)
  (carol : Runner)

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧
  r.anna.speed > 0 ∧
  r.bridgit.speed > 0 ∧
  r.carol.speed > 0 ∧
  r.anna.speed > r.bridgit.speed ∧
  r.bridgit.speed > r.carol.speed ∧
  r.length - r.bridgit.position.distance = 16 ∧
  r.length - r.carol.position.distance = 25 + (r.length - r.bridgit.position.distance)

theorem carol_distance_behind_anna (r : Race) (h : race_conditions r) :
  r.length - r.carol.position.distance = 37 :=
sorry

end carol_distance_behind_anna_l3475_347567


namespace base_nine_ones_triangular_l3475_347545

theorem base_nine_ones_triangular (k : ℕ+) : ∃ n : ℕ, (9^k.val - 1) / 8 = n * (n + 1) / 2 := by
  sorry

end base_nine_ones_triangular_l3475_347545


namespace sixteenth_selected_student_number_l3475_347540

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalStudents : ℕ
  numGroups : ℕ
  interval : ℕ
  firstSelected : ℕ

/-- Calculates the number of the nth selected student in a systematic sampling. -/
def nthSelectedStudent (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.firstSelected + (n - 1) * s.interval

theorem sixteenth_selected_student_number
  (s : SystematicSampling)
  (h1 : s.totalStudents = 800)
  (h2 : s.numGroups = 50)
  (h3 : s.interval = s.totalStudents / s.numGroups)
  (h4 : nthSelectedStudent s 3 = 36) :
  nthSelectedStudent s 16 = 244 := by
  sorry

end sixteenth_selected_student_number_l3475_347540


namespace first_seven_primes_sum_mod_eighth_prime_l3475_347596

theorem first_seven_primes_sum_mod_eighth_prime : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 := by
  sorry

end first_seven_primes_sum_mod_eighth_prime_l3475_347596


namespace pasha_wins_l3475_347500

/-- Represents the game state -/
structure GameState where
  n : ℕ  -- Number of tokens
  k : ℕ  -- Game parameter

/-- Represents a move in the game -/
inductive Move
  | pasha : Move
  | roma : Move

/-- Represents the result of the game -/
inductive GameResult
  | pashaWins : GameResult
  | romaWins : GameResult

/-- The game progression function -/
def playGame (state : GameState) (strategy : GameState → Move) : GameResult :=
  sorry

/-- Pasha's winning strategy -/
def pashaStrategy (state : GameState) : Move :=
  sorry

/-- Theorem stating that Pasha can ensure at least one token reaches the end -/
theorem pasha_wins (n k : ℕ) (h : n > k * 2^k) :
  ∃ (strategy : GameState → Move),
    playGame ⟨n, k⟩ strategy = GameResult.pashaWins :=
  sorry

end pasha_wins_l3475_347500


namespace square_field_area_l3475_347588

/-- Given a square field with side length s, prove that the area is 27889 square meters
    when the cost of barbed wire at 1.20 per meter for (4s - 2) meters equals 799.20. -/
theorem square_field_area (s : ℝ) : 
  (4 * s - 2) * 1.20 = 799.20 → s^2 = 27889 := by
  sorry

end square_field_area_l3475_347588


namespace vector_problem_l3475_347518

/-- Given two non-collinear vectors e₁ and e₂ in a real vector space,
    prove that if CB = e₁ + 3e₂, CD = 2e₁ - e₂, BF = 3e₁ - ke₂,
    and points B, D, and F are collinear, then k = 12. -/
theorem vector_problem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (e₁ e₂ : V) (hne : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (CB CD BF : V)
  (hCB : CB = e₁ + 3 • e₂)
  (hCD : CD = 2 • e₁ - e₂)
  (k : ℝ)
  (hBF : BF = 3 • e₁ - k • e₂)
  (hcollinear : ∃ (t : ℝ), BF = t • (CD - CB)) :
  k = 12 := by sorry

end vector_problem_l3475_347518


namespace zhonghuan_cup_exam_l3475_347586

theorem zhonghuan_cup_exam (total : ℕ) (english : ℕ) (chinese : ℕ) (both : ℕ) 
  (h1 : total = 45)
  (h2 : english = 35)
  (h3 : chinese = 31)
  (h4 : both = 24) :
  total - (english + chinese - both) = 3 := by
  sorry

end zhonghuan_cup_exam_l3475_347586


namespace coefficient_of_3_squared_x_squared_l3475_347538

/-- Definition of a coefficient in an algebraic term -/
def is_coefficient (c : ℝ) (term : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, term x = c * f x

/-- The coefficient of 3^2 * x^2 is 3^2 -/
theorem coefficient_of_3_squared_x_squared :
  is_coefficient (3^2) (λ x => 3^2 * x^2) :=
sorry

end coefficient_of_3_squared_x_squared_l3475_347538


namespace expression_evaluation_l3475_347585

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 4 + x * (2 + x) - 2^2
  let denominator := x - 2 + x^2
  numerator / denominator = 3 / 4 := by sorry

end expression_evaluation_l3475_347585


namespace combine_like_terms_l3475_347523

-- Define the theorem
theorem combine_like_terms (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end combine_like_terms_l3475_347523


namespace compound_weight_l3475_347530

/-- Given a compound with a molecular weight of 2670 grams/mole, 
    prove that the total weight of 10 moles of this compound is 26700 grams. -/
theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 2670 → moles = 10 → moles * molecular_weight = 26700 := by
  sorry

end compound_weight_l3475_347530


namespace system_has_three_solutions_l3475_347562

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The system of equations -/
def system (x : ℝ) : Prop :=
  3 * x^2 - 45 * (floor x) + 60 = 0 ∧ 2 * x - 3 * (floor x) + 1 = 0

/-- The theorem stating that the system has exactly 3 real solutions -/
theorem system_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ system x :=
sorry

end system_has_three_solutions_l3475_347562


namespace plant_structure_l3475_347575

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  branches : ℕ
  smallBranchesPerBranch : ℕ

/-- The total count of parts in the plant (main stem + branches + small branches). -/
def Plant.totalCount (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranchesPerBranch

/-- The plant satisfies the given conditions. -/
def validPlant (p : Plant) : Prop :=
  p.branches = p.smallBranchesPerBranch ∧ p.totalCount = 43

theorem plant_structure : ∃ (p : Plant), validPlant p ∧ p.smallBranchesPerBranch = 6 := by
  sorry

end plant_structure_l3475_347575


namespace chip_division_percentage_l3475_347558

theorem chip_division_percentage (total_chips : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) 
  (h_total : total_chips = 100)
  (h_ratio : ratio_small + ratio_large = 10)
  (h_ratio_order : ratio_large > ratio_small)
  (h_ratio_large : ratio_large = 6) :
  (ratio_large : ℚ) / (ratio_small + ratio_large : ℚ) * 100 = 60 := by
  sorry

end chip_division_percentage_l3475_347558
