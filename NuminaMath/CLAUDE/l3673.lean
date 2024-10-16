import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l3673_367360

theorem square_difference_divided_by_nine : (109^2 - 100^2) / 9 = 209 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l3673_367360


namespace NUMINAMATH_CALUDE_total_time_calculation_l3673_367367

-- Define the constants
def performance_time : ℕ := 6
def practice_ratio : ℕ := 3
def tantrum_ratio : ℕ := 5

-- Define the theorem
theorem total_time_calculation :
  performance_time * (1 + practice_ratio + tantrum_ratio) = 54 :=
by sorry

end NUMINAMATH_CALUDE_total_time_calculation_l3673_367367


namespace NUMINAMATH_CALUDE_multiply_72515_9999_l3673_367380

theorem multiply_72515_9999 : 72515 * 9999 = 725077485 := by sorry

end NUMINAMATH_CALUDE_multiply_72515_9999_l3673_367380


namespace NUMINAMATH_CALUDE_perfect_square_mod_four_l3673_367374

theorem perfect_square_mod_four (n : ℕ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_mod_four_l3673_367374


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l3673_367382

theorem bingo_prize_distribution (total_prize : ℚ) (first_winner_fraction : ℚ) 
  (num_subsequent_winners : ℕ) (each_subsequent_winner_prize : ℚ) :
  total_prize = 2400 →
  first_winner_fraction = 1/3 →
  num_subsequent_winners = 10 →
  each_subsequent_winner_prize = 160 →
  let remaining_prize := total_prize - first_winner_fraction * total_prize
  (each_subsequent_winner_prize / remaining_prize) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_bingo_prize_distribution_l3673_367382


namespace NUMINAMATH_CALUDE_sally_quarters_l3673_367355

/-- Given an initial quantity of quarters and an additional amount received,
    calculate the total number of quarters. -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 760 initial quarters and 418 additional quarters,
    the total number of quarters is 1178. -/
theorem sally_quarters : total_quarters 760 418 = 1178 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l3673_367355


namespace NUMINAMATH_CALUDE_wanda_crayon_count_l3673_367334

/-- The number of crayons Wanda, Dina, and Jacob have. -/
structure CrayonCount where
  wanda : ℕ
  dina : ℕ
  jacob : ℕ

/-- The given conditions for the crayon problem. -/
def crayon_problem (c : CrayonCount) : Prop :=
  c.dina = 28 ∧
  c.jacob = c.dina - 2 ∧
  c.wanda + c.dina + c.jacob = 116

/-- Theorem stating that Wanda has 62 crayons given the conditions. -/
theorem wanda_crayon_count (c : CrayonCount) (h : crayon_problem c) : c.wanda = 62 := by
  sorry

end NUMINAMATH_CALUDE_wanda_crayon_count_l3673_367334


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3673_367342

/-- Given a square with diagonal length 10√2 cm, its area is 100 cm². -/
theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 10 * Real.sqrt 2 → area = diagonal ^ 2 / 2 → area = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3673_367342


namespace NUMINAMATH_CALUDE_walkway_area_calculation_l3673_367321

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the layout of the garden -/
structure GardenLayout where
  bed : FlowerBed
  rows : ℕ
  columns : ℕ
  walkwayWidth : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkwayArea (layout : GardenLayout) : ℝ :=
  let totalWidth := layout.columns * layout.bed.length + (layout.columns + 1) * layout.walkwayWidth
  let totalHeight := layout.rows * layout.bed.width + (layout.rows + 1) * layout.walkwayWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := layout.rows * layout.columns * layout.bed.length * layout.bed.width
  totalArea - bedArea

theorem walkway_area_calculation (layout : GardenLayout) : 
  layout.bed.length = 6 ∧ 
  layout.bed.width = 2 ∧ 
  layout.rows = 3 ∧ 
  layout.columns = 2 ∧ 
  layout.walkwayWidth = 1 → 
  walkwayArea layout = 78 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_calculation_l3673_367321


namespace NUMINAMATH_CALUDE_range_of_a_l3673_367368

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3673_367368


namespace NUMINAMATH_CALUDE_bicycle_problem_l3673_367338

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 ∧ 
  speed_ratio = 1.2 ∧ 
  time_difference = 1/6 →
  ∃ (speed_B : ℝ),
    speed_B = 12 ∧
    distance / speed_B - time_difference = distance / (speed_ratio * speed_B) :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_problem_l3673_367338


namespace NUMINAMATH_CALUDE_sales_revenue_equilibrium_l3673_367388

/-- Represents the sales revenue increase percentage for product C -/
def revenue_increase_percentage : ℝ := 0.3

/-- Represents last year's total sales revenue -/
def last_year_revenue : ℝ := 1

/-- Represents the proportion of product C's revenue in last year's total revenue -/
def product_c_proportion : ℝ := 0.4

/-- Represents the decrease percentage for products A and B -/
def decrease_percentage : ℝ := 0.2

theorem sales_revenue_equilibrium :
  product_c_proportion * last_year_revenue * (1 + revenue_increase_percentage) +
  (1 - product_c_proportion) * last_year_revenue * (1 - decrease_percentage) =
  last_year_revenue :=
sorry

end NUMINAMATH_CALUDE_sales_revenue_equilibrium_l3673_367388


namespace NUMINAMATH_CALUDE_cookies_left_l3673_367332

def dozen : ℕ := 12

theorem cookies_left (total : ℕ) (eaten_percent : ℚ) (h1 : total = 2 * dozen) (h2 : eaten_percent = 1/4) :
  total - (eaten_percent * total).floor = 18 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3673_367332


namespace NUMINAMATH_CALUDE_fair_attendance_proof_l3673_367306

def fair_attendance (last_year this_year next_year : ℕ) : Prop :=
  (this_year = 600) ∧
  (next_year = 2 * this_year) ∧
  (last_year = next_year - 200)

theorem fair_attendance_proof :
  ∃ (last_year this_year next_year : ℕ),
    fair_attendance last_year this_year next_year ∧
    last_year = 1000 ∧ this_year = 600 ∧ next_year = 1200 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_proof_l3673_367306


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l3673_367364

theorem square_area_equal_perimeter (triangle_area : ℝ) : 
  triangle_area = 16 * Real.sqrt 3 → 
  ∃ (triangle_side square_side : ℝ), 
    triangle_side > 0 ∧ 
    square_side > 0 ∧ 
    (triangle_side^2 * Real.sqrt 3) / 4 = triangle_area ∧ 
    3 * triangle_side = 4 * square_side ∧ 
    square_side^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l3673_367364


namespace NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l3673_367313

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ := x * (x * (3 * x + 2) + 1) + 1

-- Theorem statement
theorem horner_method_equals_f_at_2 : 
  horner_method 2 = f 2 ∧ f 2 = 35 := by sorry

end NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l3673_367313


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3673_367365

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + 2 = 0) ∧ (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3673_367365


namespace NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3673_367378

/-- Represents the prices of fruits in dollars -/
structure FruitPrices where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitPurchaseConditions (p : FruitPrices) : Prop :=
  p.apples + p.bananas + p.cantaloupe + p.dates = 30 ∧
  p.dates = 3 * p.apples ∧
  p.cantaloupe = p.apples - p.bananas

/-- The theorem stating the cost of bananas and cantaloupe -/
theorem banana_cantaloupe_cost (p : FruitPrices) 
  (h : fruitPurchaseConditions p) : 
  p.bananas + p.cantaloupe = 6 := by
  sorry


end NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3673_367378


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3673_367357

theorem complex_arithmetic_equality : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3673_367357


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3673_367302

theorem expand_and_simplify (x : ℝ) : (1 + x^3) * (1 - x^4) = 1 + x^3 - x^4 - x^7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3673_367302


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l3673_367394

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  λ i => a₁ + d * (i : ℝ)

theorem common_difference_of_arithmetic_sequence
  (a₁ : ℝ) (aₙ : ℝ) (S : ℝ) (n : ℕ) (d : ℝ)
  (h₁ : a₁ = 5)
  (h₂ : aₙ = 50)
  (h₃ : S = 275)
  (h₄ : aₙ = a₁ + d * (n - 1))
  (h₅ : S = n / 2 * (a₁ + aₙ))
  : d = 5 := by
  sorry

#check common_difference_of_arithmetic_sequence

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l3673_367394


namespace NUMINAMATH_CALUDE_sqrt_equation_l3673_367341

theorem sqrt_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l3673_367341


namespace NUMINAMATH_CALUDE_initial_overs_correct_l3673_367387

/-- Represents the number of overs played initially in a cricket game. -/
def initial_overs : ℝ := 10

/-- The target score for the cricket game. -/
def target_score : ℝ := 282

/-- The initial run rate in runs per over. -/
def initial_run_rate : ℝ := 6.2

/-- The required run rate for the remaining overs in runs per over. -/
def required_run_rate : ℝ := 5.5

/-- The number of remaining overs. -/
def remaining_overs : ℝ := 40

/-- Theorem stating that the initial number of overs is correct given the conditions. -/
theorem initial_overs_correct : 
  target_score = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by sorry

end NUMINAMATH_CALUDE_initial_overs_correct_l3673_367387


namespace NUMINAMATH_CALUDE_late_average_speed_l3673_367369

def journey_length : ℝ := 225
def original_speed : ℝ := 60
def delay_time : ℝ := 0.75 -- 45 minutes in hours

theorem late_average_speed (v : ℝ) : 
  journey_length / original_speed = journey_length / v - delay_time →
  v = 50 := by
sorry

end NUMINAMATH_CALUDE_late_average_speed_l3673_367369


namespace NUMINAMATH_CALUDE_delegation_selection_l3673_367300

theorem delegation_selection (n k : ℕ) (h1 : n = 12) (h2 : k = 3) : 
  Nat.choose n k = 220 := by
  sorry

end NUMINAMATH_CALUDE_delegation_selection_l3673_367300


namespace NUMINAMATH_CALUDE_max_square_plots_exists_valid_partition_8_largest_num_square_plots_l3673_367305

def field_length : ℕ := 30
def field_width : ℕ := 60
def available_fencing : ℕ := 2500

def is_valid_partition (s : ℕ) : Prop :=
  s ∣ field_length ∧ s ∣ field_width ∧
  (field_length / s - 1) * field_width + (field_width / s - 1) * field_length ≤ available_fencing

def num_plots (s : ℕ) : ℕ :=
  (field_length / s) * (field_width / s)

theorem max_square_plots :
  ∀ s : ℕ, is_valid_partition s → num_plots s ≤ 8 :=
by sorry

theorem exists_valid_partition_8 :
  ∃ s : ℕ, is_valid_partition s ∧ num_plots s = 8 :=
by sorry

theorem largest_num_square_plots : 
  (∃ s : ℕ, is_valid_partition s ∧ num_plots s = 8) ∧
  (∀ s : ℕ, is_valid_partition s → num_plots s ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_max_square_plots_exists_valid_partition_8_largest_num_square_plots_l3673_367305


namespace NUMINAMATH_CALUDE_zero_in_interval_implies_m_leq_neg_one_l3673_367345

/-- A function f(x) = x² + (m-1)x + 1 has a zero point in the interval [0, 2] -/
def has_zero_in_interval (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m-1)*x + 1 = 0

/-- If f(x) = x² + (m-1)x + 1 has a zero point in the interval [0, 2], then m ≤ -1 -/
theorem zero_in_interval_implies_m_leq_neg_one (m : ℝ) :
  has_zero_in_interval m → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_implies_m_leq_neg_one_l3673_367345


namespace NUMINAMATH_CALUDE_probability_three_games_probability_best_of_five_l3673_367356

-- Define the probability of A winning a single game
def p_A : ℚ := 2/3

-- Define the probability of B winning a single game
def p_B : ℚ := 1/3

-- Theorem for part (1)
theorem probability_three_games 
  (h1 : p_A + p_B = 1) 
  (h2 : p_A = 2/3) 
  (h3 : p_B = 1/3) :
  let p_A_wins_two := 3 * (p_A^2 * p_B)
  let p_B_wins_at_least_one := 1 - p_A^3
  (p_A_wins_two = 4/9) ∧ (p_B_wins_at_least_one = 19/27) := by
  sorry

-- Theorem for part (2)
theorem probability_best_of_five
  (h1 : p_A + p_B = 1)
  (h2 : p_A = 2/3)
  (h3 : p_B = 1/3) :
  let p_A_wins_three_straight := p_A^3
  let p_A_wins_in_four := 3 * (p_A^3 * p_B)
  let p_A_wins_in_five := 6 * (p_A^3 * p_B^2)
  p_A_wins_three_straight + p_A_wins_in_four + p_A_wins_in_five = 64/81 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_games_probability_best_of_five_l3673_367356


namespace NUMINAMATH_CALUDE_final_weight_gain_l3673_367396

def weight_change (initial_weight : ℕ) (final_weight : ℕ) : ℕ :=
  let first_loss := 12
  let second_gain := 2 * first_loss
  let third_loss := 3 * first_loss
  let weight_after_third_loss := initial_weight - first_loss + second_gain - third_loss
  final_weight - weight_after_third_loss

theorem final_weight_gain (initial_weight final_weight : ℕ) 
  (h1 : initial_weight = 99)
  (h2 : final_weight = 81) :
  weight_change initial_weight final_weight = 6 := by
  sorry

#eval weight_change 99 81

end NUMINAMATH_CALUDE_final_weight_gain_l3673_367396


namespace NUMINAMATH_CALUDE_office_salary_problem_l3673_367354

/-- Represents the average salary of non-officers in Rs/month -/
def average_salary_non_officers : ℝ := 110

theorem office_salary_problem (total_employees : ℕ) (officers : ℕ) (non_officers : ℕ)
  (avg_salary_all : ℝ) (avg_salary_officers : ℝ) :
  total_employees = officers + non_officers →
  total_employees = 495 →
  officers = 15 →
  non_officers = 480 →
  avg_salary_all = 120 →
  avg_salary_officers = 440 →
  average_salary_non_officers = 
    (total_employees * avg_salary_all - officers * avg_salary_officers) / non_officers :=
by sorry

end NUMINAMATH_CALUDE_office_salary_problem_l3673_367354


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_sqrt_five_l3673_367346

theorem cube_root_sum_equals_two_sqrt_five :
  (((17 * Real.sqrt 5 + 38) ^ (1/3 : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1/3 : ℝ))) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_sqrt_five_l3673_367346


namespace NUMINAMATH_CALUDE_cody_dumplings_l3673_367353

theorem cody_dumplings (A B : ℕ) (P1 Q1 Q2 P2 : ℚ) : 
  A = 14 → 
  B = 20 → 
  P1 = 1/2 → 
  Q1 = 1/4 → 
  Q2 = 2/5 → 
  P2 = 3/20 → 
  ∃ (remaining : ℕ), remaining = 16 ∧ 
    remaining = A - Int.floor (P1 * A) - Int.floor (Q1 * (A - Int.floor (P1 * A))) + 
                B - Int.floor (Q2 * B) - 
                Int.floor (P2 * (A - Int.floor (P1 * A) - Int.floor (Q1 * (A - Int.floor (P1 * A))) + 
                                 B - Int.floor (Q2 * B))) :=
by sorry

end NUMINAMATH_CALUDE_cody_dumplings_l3673_367353


namespace NUMINAMATH_CALUDE_marathon_distance_theorem_l3673_367376

/-- Represents the length of a marathon in miles and yards. -/
structure MarathonLength where
  miles : ℕ
  yards : ℕ

/-- Calculates the total distance in miles and yards after running multiple marathons. -/
def totalDistance (marathonLength : MarathonLength) (numMarathons : ℕ) : MarathonLength :=
  let totalMiles := marathonLength.miles * numMarathons
  let totalYards := marathonLength.yards * numMarathons
  let extraMiles := totalYards / 1760
  let remainingYards := totalYards % 1760
  { miles := totalMiles + extraMiles, yards := remainingYards }

theorem marathon_distance_theorem :
  let marathonLength : MarathonLength := { miles := 26, yards := 385 }
  let numMarathons : ℕ := 15
  let result := totalDistance marathonLength numMarathons
  result.miles = 393 ∧ result.yards = 495 := by
  sorry

end NUMINAMATH_CALUDE_marathon_distance_theorem_l3673_367376


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3673_367352

theorem quadratic_root_relation (p : ℚ) : 
  (∃ x y : ℚ, x = 3 * y ∧ 
   x^2 - (3*p - 2)*x + p^2 - 1 = 0 ∧ 
   y^2 - (3*p - 2)*y + p^2 - 1 = 0) ↔ 
  (p = 2 ∨ p = 14/11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3673_367352


namespace NUMINAMATH_CALUDE_vector_at_zero_l3673_367331

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  vector : ℝ → Fin 3 → ℝ

/-- The vector at a given parameter value -/
def vectorAt (line : ParameterizedLine) (t : ℝ) : Fin 3 → ℝ := line.vector t

theorem vector_at_zero (line : ParameterizedLine) 
  (h1 : vectorAt line 1 = ![2, 4, 9])
  (h2 : vectorAt line (-1) = ![-1, 1, 2]) :
  vectorAt line 0 = ![1/2, 5/2, 11/2] := by
  sorry

end NUMINAMATH_CALUDE_vector_at_zero_l3673_367331


namespace NUMINAMATH_CALUDE_smallest_excluded_number_l3673_367398

theorem smallest_excluded_number : ∃ n : ℕ, 
  (∀ k ∈ Finset.range 200, k + 1 ≠ 128 ∧ k + 1 ≠ 129 → n % (k + 1) = 0) ∧
  (∀ m : ℕ, m < 128 → 
    ¬∃ n : ℕ, (∀ k ∈ Finset.range 200, k + 1 ≠ m ∧ k + 1 ≠ m + 1 → n % (k + 1) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_excluded_number_l3673_367398


namespace NUMINAMATH_CALUDE_last_number_is_odd_l3673_367399

/-- The operation of choosing two numbers and replacing them with their absolute difference -/
def boardOperation (numbers : List Int) : List Int :=
  sorry

/-- The process of repeatedly applying the operation until only one number remains -/
def boardProcess (initialNumbers : List Int) : Int :=
  sorry

/-- The list of integers from 1 to 2018 -/
def initialBoard : List Int :=
  List.range 2018

theorem last_number_is_odd :
  Odd (boardProcess initialBoard) :=
by sorry

end NUMINAMATH_CALUDE_last_number_is_odd_l3673_367399


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3673_367329

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The area of overlap between a rectangle and a square -/
def overlap_area (r : Rectangle) (s : Square) : ℝ := sorry

/-- The theorem stating the ratio of rectangle's width to height -/
theorem rectangle_square_overlap_ratio 
  (r : Rectangle) 
  (s : Square) 
  (h1 : overlap_area r s = 0.6 * r.width * r.height) 
  (h2 : overlap_area r s = 0.3 * s.side * s.side) : 
  r.width / r.height = 12.5 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3673_367329


namespace NUMINAMATH_CALUDE_oranges_and_cookies_donation_l3673_367349

theorem oranges_and_cookies_donation (total_oranges : ℕ) (total_cookies : ℕ) (num_children : ℕ) 
  (h_oranges : total_oranges = 81)
  (h_cookies : total_cookies = 65)
  (h_children : num_children = 7) :
  (total_oranges % num_children = 4) ∧ (total_cookies % num_children = 2) :=
by sorry

end NUMINAMATH_CALUDE_oranges_and_cookies_donation_l3673_367349


namespace NUMINAMATH_CALUDE_assignment_increases_by_one_l3673_367393

-- Define the assignment operation
def assign (x : ℕ) : ℕ := x + 1

-- Theorem stating that the assignment n = n + 1 increases n by 1
theorem assignment_increases_by_one (n : ℕ) : assign n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_assignment_increases_by_one_l3673_367393


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l3673_367392

theorem cube_root_of_negative_eight_squared (x : ℝ) : x^3 = (-8)^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l3673_367392


namespace NUMINAMATH_CALUDE_quadratic_greatest_lower_bound_l3673_367337

/-- The greatest lower bound of a quadratic function -/
theorem quadratic_greatest_lower_bound (a b : ℝ) (ha : a ≠ 0) (hnz : a ≠ 0 ∨ b ≠ 0) (hpos : a > 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x
  ∃ M : ℝ, M = -b^2 / (4 * a) ∧ ∀ x, f x ≥ M ∧ ∀ N, (∀ x, f x ≥ N) → N ≤ M :=
by sorry

end NUMINAMATH_CALUDE_quadratic_greatest_lower_bound_l3673_367337


namespace NUMINAMATH_CALUDE_expand_expression_l3673_367361

theorem expand_expression (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3673_367361


namespace NUMINAMATH_CALUDE_license_plate_increase_l3673_367340

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^4 * 10^2
  new_plates / old_plates = 26^2 / 10 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3673_367340


namespace NUMINAMATH_CALUDE_right_triangle_area_arithmetic_sides_l3673_367366

/-- A right-angled triangle with area 37.5 m² and sides forming an arithmetic sequence has side lengths 7.5 m, 10 m, and 12.5 m. -/
theorem right_triangle_area_arithmetic_sides (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a < b ∧ b < c →  -- Ordered side lengths
  b - a = c - b →  -- Arithmetic sequence condition
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right-angled triangle)
  (1/2) * a * b = 37.5 →  -- Area condition
  (a, b, c) = (7.5, 10, 12.5) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_arithmetic_sides_l3673_367366


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l3673_367322

theorem events_mutually_exclusive (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 1/5)
  (hB : P B = 1/3)
  (hAB : P (A ∪ B) = 8/15) :
  P (A ∪ B) = P A + P B :=
by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l3673_367322


namespace NUMINAMATH_CALUDE_divisible_by_eight_l3673_367381

theorem divisible_by_eight (n : ℕ) : ∃ k : ℤ, 6 * n^2 + 4 * n + (-1)^n * 9 + 7 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l3673_367381


namespace NUMINAMATH_CALUDE_mingyoungs_animals_l3673_367384

theorem mingyoungs_animals (chickens ducks rabbits : ℕ) : 
  chickens = 4 * ducks →
  ducks = rabbits + 17 →
  rabbits = 8 →
  chickens + ducks + rabbits = 133 := by
sorry

end NUMINAMATH_CALUDE_mingyoungs_animals_l3673_367384


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3673_367330

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sampleSize : ℕ
  interval : ℕ
  startPoint : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.startPoint + k * s.interval ∧ k < s.sampleSize

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_interval : s.interval = s.total / s.sampleSize)
  (h_start : s.startPoint = 6)
  (h_contains_6 : s.contains 6)
  (h_contains_32 : s.contains 32)
  (h_contains_45 : s.contains 45) :
  s.contains 19 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3673_367330


namespace NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l3673_367316

theorem max_sum_abs_on_unit_sphere :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l3673_367316


namespace NUMINAMATH_CALUDE_no_simultaneous_divisibility_l3673_367379

theorem no_simultaneous_divisibility (k n : ℕ) (hk : k > 1) (hn : n > 1)
  (hk_odd : Odd k) (hn_odd : Odd n)
  (h_exists : ∃ a : ℕ, k ∣ 2^a + 1 ∧ n ∣ 2^a - 1) :
  ¬∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_divisibility_l3673_367379


namespace NUMINAMATH_CALUDE_trigonometric_fraction_equals_two_l3673_367397

theorem trigonometric_fraction_equals_two :
  (3 - Real.sin (70 * π / 180)) / (2 - Real.cos (10 * π / 180) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_equals_two_l3673_367397


namespace NUMINAMATH_CALUDE_smallest_product_l3673_367391

def number_list : List Int := [-5, -3, -1, 2, 4, 6]

def is_valid_product (p : Int) : Prop :=
  ∃ (a b c : Int), a ∈ number_list ∧ b ∈ number_list ∧ c ∈ number_list ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ p = a * b * c

theorem smallest_product :
  ∀ p, is_valid_product p → p ≥ -120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l3673_367391


namespace NUMINAMATH_CALUDE_homework_time_calculation_l3673_367307

theorem homework_time_calculation (total_time : ℝ) :
  (0.3 * total_time = 0.3 * total_time) ∧  -- Time spent on math
  (0.4 * total_time = 0.4 * total_time) ∧  -- Time spent on science
  (total_time - 0.3 * total_time - 0.4 * total_time = 45) →  -- Time spent on other subjects
  total_time = 150 := by
sorry

end NUMINAMATH_CALUDE_homework_time_calculation_l3673_367307


namespace NUMINAMATH_CALUDE_equation_representation_l3673_367317

theorem equation_representation (a : ℝ) : (3 * a + 5 = 9) ↔ 
  (∃ x : ℝ, x = 3 * a + 5 ∧ x = 9) :=
sorry

end NUMINAMATH_CALUDE_equation_representation_l3673_367317


namespace NUMINAMATH_CALUDE_divisible_by_two_l3673_367358

theorem divisible_by_two (a b : ℕ) : 
  (2 ∣ (a * b)) → (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_l3673_367358


namespace NUMINAMATH_CALUDE_stating_parking_arrangement_count_l3673_367311

/-- Represents the number of parking spaces -/
def num_spaces : ℕ := 8

/-- Represents the number of trucks -/
def num_trucks : ℕ := 2

/-- Represents the number of cars -/
def num_cars : ℕ := 2

/-- Represents the total number of vehicles -/
def total_vehicles : ℕ := num_trucks + num_cars

/-- 
Represents the number of ways to arrange trucks and cars in a row of parking spaces,
where vehicles of the same type must be adjacent.
-/
def parking_arrangements (spaces : ℕ) (trucks : ℕ) (cars : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the number of ways to arrange 2 trucks and 2 cars
in a row of 8 parking spaces, where vehicles of the same type must be adjacent,
is equal to 120.
-/
theorem parking_arrangement_count :
  parking_arrangements num_spaces num_trucks num_cars = 120 := by
  sorry

end NUMINAMATH_CALUDE_stating_parking_arrangement_count_l3673_367311


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3673_367318

-- Define the types for line and plane
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the parallel and perpendicular relations
variable (parallel : L → P → Prop)
variable (perpendicular : L → P → Prop)
variable (plane_perpendicular : P → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : L) (α β : P) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3673_367318


namespace NUMINAMATH_CALUDE_robotics_club_theorem_l3673_367309

theorem robotics_club_theorem (total : ℕ) (cs : ℕ) (eng : ℕ) (both : ℕ) 
  (h1 : total = 120)
  (h2 : cs = 75)
  (h3 : eng = 50)
  (h4 : both = 10) :
  total - (cs + eng - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_theorem_l3673_367309


namespace NUMINAMATH_CALUDE_power_equality_natural_numbers_l3673_367326

theorem power_equality_natural_numbers (a b : ℕ) :
  a ^ b = b ^ a ↔ (a = b) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) := by
sorry

end NUMINAMATH_CALUDE_power_equality_natural_numbers_l3673_367326


namespace NUMINAMATH_CALUDE_darwin_gas_expense_l3673_367370

def initial_amount : ℝ := 600
def final_amount : ℝ := 300

theorem darwin_gas_expense (x : ℝ) 
  (h1 : 0 < x ∧ x < 1) 
  (h2 : final_amount = initial_amount - x * initial_amount - (1/4) * (initial_amount - x * initial_amount)) :
  x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_darwin_gas_expense_l3673_367370


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3673_367359

theorem students_not_enrolled (total : ℕ) (football : ℕ) (swimming : ℕ) (both : ℕ) :
  total = 100 →
  football = 37 →
  swimming = 40 →
  both = 15 →
  total - (football + swimming - both) = 38 := by
sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3673_367359


namespace NUMINAMATH_CALUDE_no_specific_m_value_l3673_367385

theorem no_specific_m_value (m : ℝ) (z₁ z₂ : ℂ) 
  (h₁ : z₁ = m + 2*I) 
  (h₂ : z₂ = 3 - 4*I) : 
  ∀ (n : ℝ), ∃ (m' : ℝ), m' ≠ n ∧ z₁ = m' + 2*I :=
sorry

end NUMINAMATH_CALUDE_no_specific_m_value_l3673_367385


namespace NUMINAMATH_CALUDE_books_bought_l3673_367324

theorem books_bought (initial_amount : ℕ) (cost_per_book : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  cost_per_book = 7 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / cost_per_book = 9 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_l3673_367324


namespace NUMINAMATH_CALUDE_triangle_with_seven_points_forms_fifteen_triangles_l3673_367333

/-- The number of smaller triangles formed in a triangle with interior points -/
def num_smaller_triangles (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem: A triangle with 7 interior points forms 15 smaller triangles -/
theorem triangle_with_seven_points_forms_fifteen_triangles :
  num_smaller_triangles 7 = 15 := by
  sorry

#eval num_smaller_triangles 7  -- Should output 15

end NUMINAMATH_CALUDE_triangle_with_seven_points_forms_fifteen_triangles_l3673_367333


namespace NUMINAMATH_CALUDE_proportion_problem_l3673_367323

theorem proportion_problem (x y : ℚ) : 
  (3/4 : ℚ) / x = 7/8 → x / y = 5/6 → x = 6/7 ∧ y = 36/35 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l3673_367323


namespace NUMINAMATH_CALUDE_parallelogram_sides_l3673_367363

-- Define the triangle ABC
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the parallelogram BKLM
structure Parallelogram :=
  (BM : ℝ)
  (BK : ℝ)

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.AB = 18 ∧ t.BC = 12

def parallelogram_conditions (t : Triangle) (p : Parallelogram) : Prop :=
  -- Area of BKLM is 4/9 of the area of ABC
  p.BM * p.BK = (4/9) * (1/2) * t.AB * t.BC

-- Theorem statement
theorem parallelogram_sides (t : Triangle) (p : Parallelogram) :
  triangle_conditions t →
  parallelogram_conditions t p →
  ((p.BM = 8 ∧ p.BK = 6) ∨ (p.BM = 4 ∧ p.BK = 12)) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_sides_l3673_367363


namespace NUMINAMATH_CALUDE_joes_bath_shop_problem_l3673_367339

theorem joes_bath_shop_problem (bottles_per_box : ℕ) (total_sold : ℕ) 
  (h1 : bottles_per_box = 19)
  (h2 : total_sold = 95)
  (h3 : ∃ (bar_boxes bottle_boxes : ℕ), bar_boxes * total_sold = bottle_boxes * total_sold)
  (h4 : ∀ x : ℕ, x > 1 ∧ x * total_sold = bottles_per_box * total_sold → x ≥ 5) :
  ∃ (bars_per_box : ℕ), bars_per_box > 1 ∧ bars_per_box * total_sold = bottles_per_box * total_sold ∧ bars_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_joes_bath_shop_problem_l3673_367339


namespace NUMINAMATH_CALUDE_f_eval_approx_l3673_367344

/-- The polynomial function f(x) -/
def f (x : ℝ) : ℝ := 1 + x + 0.5*x^2 + 0.16667*x^3 + 0.04167*x^4 + 0.00833*x^5

/-- The evaluation point -/
def x₀ : ℝ := -0.2

/-- The theorem stating that f(x₀) is approximately equal to 0.81873 -/
theorem f_eval_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |f x₀ - 0.81873| < ε := by
  sorry

end NUMINAMATH_CALUDE_f_eval_approx_l3673_367344


namespace NUMINAMATH_CALUDE_square_of_sum_23_2_l3673_367386

theorem square_of_sum_23_2 : 23^2 + 2*(23*2) + 2^2 = 625 := by sorry

end NUMINAMATH_CALUDE_square_of_sum_23_2_l3673_367386


namespace NUMINAMATH_CALUDE_ken_steak_purchase_l3673_367351

/-- The cost of one pound of steak, given the conditions of Ken's purchase -/
def steak_cost (total_pounds : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  (paid - change) / total_pounds

theorem ken_steak_purchase :
  steak_cost 2 20 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ken_steak_purchase_l3673_367351


namespace NUMINAMATH_CALUDE_min_probability_is_601_1225_l3673_367348

/-- The number of cards in the deck -/
def num_cards : ℕ := 52

/-- The probability that Charlie and Jane are on the same team, given that they draw cards a and a+11 -/
def p (a : ℕ) : ℚ :=
  let remaining_combinations := (num_cards - 2).choose 2
  let lower_team_combinations := (a - 1).choose 2
  let higher_team_combinations := (num_cards - (a + 11) - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / remaining_combinations

/-- The minimum value of a for which p(a) is at least 1/2 -/
def min_a : ℕ := 36

theorem min_probability_is_601_1225 :
  p min_a = 601 / 1225 ∧ ∀ a : ℕ, 1 ≤ a ∧ a ≤ num_cards - 11 → p a ≥ 1 / 2 → p a ≥ p min_a :=
sorry

end NUMINAMATH_CALUDE_min_probability_is_601_1225_l3673_367348


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3673_367395

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 3 = 35904 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3673_367395


namespace NUMINAMATH_CALUDE_inequality_implies_a_positive_l3673_367312

theorem inequality_implies_a_positive (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → x^2 + x + a > 0) →
  a > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_positive_l3673_367312


namespace NUMINAMATH_CALUDE_q_value_approximation_l3673_367328

theorem q_value_approximation (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p*q = 16/3) : 
  ∃ ε > 0, |q - 7.27| < ε :=
sorry

end NUMINAMATH_CALUDE_q_value_approximation_l3673_367328


namespace NUMINAMATH_CALUDE_custom_calculator_results_l3673_367319

/-- A custom operation that satisfies specific properties -/
noncomputable def customOp (a b : ℕ) : ℕ :=
  sorry

/-- Addition operation -/
def add : ℕ → ℕ → ℕ := (·+·)

axiom custom_op_self (a : ℕ) : customOp a a = a

axiom custom_op_zero (a : ℕ) : customOp a 0 = 2 * a

axiom custom_op_distributive (a b c d : ℕ) :
  add (customOp a b) (customOp c d) = add (customOp a c) (customOp b d)

theorem custom_calculator_results :
  (customOp (add 2 3) (add 0 3) = 7) ∧
  (customOp 1024 48 = 2000) := by
  sorry

end NUMINAMATH_CALUDE_custom_calculator_results_l3673_367319


namespace NUMINAMATH_CALUDE_cost_per_book_l3673_367377

def total_books : ℕ := 14
def total_spent : ℕ := 224

theorem cost_per_book : total_spent / total_books = 16 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_book_l3673_367377


namespace NUMINAMATH_CALUDE_work_completion_days_l3673_367314

/-- Proves that the original number of days planned to complete the work is 15,
    given the conditions of the problem. -/
theorem work_completion_days : ∀ (total_men : ℕ) (absent_men : ℕ) (actual_days : ℕ),
  total_men = 48 →
  absent_men = 8 →
  actual_days = 18 →
  (total_men - absent_men) * actual_days = total_men * 15 :=
by
  sorry

#check work_completion_days

end NUMINAMATH_CALUDE_work_completion_days_l3673_367314


namespace NUMINAMATH_CALUDE_annulus_area_l3673_367301

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (R r t : ℝ) (h1 : R > r) (h2 : R^2 = r^2 + t^2) : 
  π * R^2 - π * r^2 = π * t^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l3673_367301


namespace NUMINAMATH_CALUDE_gcd_problem_l3673_367375

theorem gcd_problem (h1 : Nat.Prime 361) 
                    (h2 : 172 = 2 * 2 * 43) 
                    (h3 : 473 = 43 * 11) 
                    (h4 : 360 = 4 * 90) : 
  Nat.gcd (360 * 473) (172 * 361) = 172 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3673_367375


namespace NUMINAMATH_CALUDE_savings_calculation_l3673_367304

def calculate_savings (initial_winnings : ℚ) (first_saving_ratio : ℚ) (profit_ratio : ℚ) (second_saving_ratio : ℚ) : ℚ :=
  let first_saving := initial_winnings * first_saving_ratio
  let second_bet := initial_winnings * (1 - first_saving_ratio)
  let second_bet_earnings := second_bet * (1 + profit_ratio)
  let second_saving := second_bet_earnings * second_saving_ratio
  first_saving + second_saving

theorem savings_calculation :
  calculate_savings 100 (1/2) (3/5) (1/2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3673_367304


namespace NUMINAMATH_CALUDE_product_base_8_units_digit_l3673_367315

theorem product_base_8_units_digit : 
  (123 * 58) % 8 = 6 := by sorry

end NUMINAMATH_CALUDE_product_base_8_units_digit_l3673_367315


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3673_367336

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3673_367336


namespace NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l3673_367362

/-- The optimal selection method popularized by Hua Luogeng -/
structure OptimalSelectionMethod where
  author : String
  concept : String

/-- Definition of Hua Luogeng's optimal selection method -/
def huaMethod : OptimalSelectionMethod :=
  { author := "Hua Luogeng"
  , concept := "golden ratio" }

/-- Theorem stating that Hua Luogeng's optimal selection method uses the golden ratio -/
theorem hua_method_uses_golden_ratio :
  huaMethod.concept = "golden ratio" := by
  sorry

end NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l3673_367362


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3673_367371

theorem quadratic_minimum_value :
  ∃ (min : ℝ), min = -3 ∧ ∀ x : ℝ, (x - 1)^2 - 3 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3673_367371


namespace NUMINAMATH_CALUDE_probability_two_girls_l3673_367320

def total_students : ℕ := 5
def num_boys : ℕ := 2
def num_girls : ℕ := 3
def students_selected : ℕ := 2

theorem probability_two_girls :
  (Nat.choose num_girls students_selected) / (Nat.choose total_students students_selected) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l3673_367320


namespace NUMINAMATH_CALUDE_find_k_l3673_367383

theorem find_k (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k)*(x + k) = x^3 + k*(x^2 - x - 7)) → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3673_367383


namespace NUMINAMATH_CALUDE_chocolate_probability_l3673_367373

/-- Represents a chocolate bar with dark and white segments -/
structure ChocolateBar :=
  (segments : List (Float × Bool))  -- List of (length, isDark) pairs

/-- The process of cutting and switching chocolate bars -/
def cutAndSwitch (p : Float) (bar1 bar2 : ChocolateBar) : ChocolateBar × ChocolateBar :=
  sorry

/-- Checks if the chocolate at 1/3 and 2/3 are the same type -/
def sameTypeAt13And23 (bar : ChocolateBar) : Bool :=
  sorry

/-- Performs the cutting and switching process for n steps -/
def processSteps (n : Nat) (bar1 bar2 : ChocolateBar) : ChocolateBar × ChocolateBar :=
  sorry

/-- The probability of getting the same type at 1/3 and 2/3 after n steps -/
def probabilitySameType (n : Nat) : Float :=
  sorry

theorem chocolate_probability :
  probabilitySameType 100 = 1/2 * (1 + (1/3)^100) :=
sorry

end NUMINAMATH_CALUDE_chocolate_probability_l3673_367373


namespace NUMINAMATH_CALUDE_no_solution_for_socks_l3673_367327

theorem no_solution_for_socks : ¬∃ (n m : ℕ), n + m = 2009 ∧ (n^2 - 2*n*m + m^2) = (n + m) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_socks_l3673_367327


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3673_367347

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 90 ∧ percentage = 50 ∧ final = initial * (1 + percentage / 100) →
  final = 135 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3673_367347


namespace NUMINAMATH_CALUDE_m_range_theorem_l3673_367325

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem m_range_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-2) 2, f x ≠ 0 → True)  -- f is defined on [-2, 2]
  (h2 : is_even f)
  (h3 : monotone_decreasing_on f 0 2)
  (h4 : ∀ m, f (1 - m) < f m) :
  ∀ m, -2 ≤ m ∧ m < (1/2) := by
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3673_367325


namespace NUMINAMATH_CALUDE_integer_solutions_x_squared_plus_15_eq_y_squared_l3673_367372

theorem integer_solutions_x_squared_plus_15_eq_y_squared :
  {(x, y) : ℤ × ℤ | x^2 + 15 = y^2} =
  {(7, 8), (-7, -8), (-7, 8), (7, -8), (1, 4), (-1, -4), (-1, 4), (1, -4)} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_x_squared_plus_15_eq_y_squared_l3673_367372


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3673_367303

theorem quadratic_factorization (x : ℝ) :
  x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3673_367303


namespace NUMINAMATH_CALUDE_scientific_notation_of_chip_size_l3673_367389

theorem scientific_notation_of_chip_size :
  ∃ (a : ℝ) (n : ℤ), 0.000000014 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_chip_size_l3673_367389


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l3673_367343

-- Define the original proposition
def original_prop (a c : ℝ) : Prop := a > 0 → a * c^2 ≥ 0

-- Define the inverse proposition
def inverse_prop (a c : ℝ) : Prop := a * c^2 ≥ 0 → a > 0

-- Theorem stating that inverse_prop is the inverse of original_prop
theorem inverse_of_proposition :
  ∀ a c : ℝ, inverse_prop a c ↔ ¬(∃ a c : ℝ, original_prop a c ∧ ¬(inverse_prop a c)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l3673_367343


namespace NUMINAMATH_CALUDE_newton_method_convergence_l3673_367310

noncomputable def newtonSequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => (newtonSequence n ^ 2 + 2) / (2 * newtonSequence n)

theorem newton_method_convergence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |newtonSequence n - Real.sqrt 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_newton_method_convergence_l3673_367310


namespace NUMINAMATH_CALUDE_cosine_function_period_l3673_367308

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    if the graph covers two periods in an interval of 2π, then b = 2. -/
theorem cosine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * x + c) + d) →
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * (x + 2 * Real.pi) + c) + d) →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_cosine_function_period_l3673_367308


namespace NUMINAMATH_CALUDE_power_inequality_l3673_367335

theorem power_inequality (p q a : ℝ) (h1 : p > q) (h2 : q > 1) (h3 : 0 < a) (h4 : a < 1) :
  p ^ a > q ^ a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3673_367335


namespace NUMINAMATH_CALUDE_sector_area_l3673_367390

/-- The area of a sector with a central angle of 120° and a radius of 3 is 3π. -/
theorem sector_area (angle : Real) (radius : Real) : 
  angle = 120 * π / 180 → radius = 3 → (1/2) * angle * radius^2 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3673_367390


namespace NUMINAMATH_CALUDE_cube_difference_equals_product_plus_constant_l3673_367350

theorem cube_difference_equals_product_plus_constant
  (x y : ℤ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x^3 - y^3 = x*y + 61) :
  x = 6 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_equals_product_plus_constant_l3673_367350
