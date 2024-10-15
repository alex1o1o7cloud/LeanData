import Mathlib

namespace NUMINAMATH_CALUDE_divisible_by_4_6_9_less_than_300_l1806_180647

theorem divisible_by_4_6_9_less_than_300 : 
  (Finset.filter (fun n : ℕ => n % 4 = 0 ∧ n % 6 = 0 ∧ n % 9 = 0) (Finset.range 300)).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_4_6_9_less_than_300_l1806_180647


namespace NUMINAMATH_CALUDE_geometric_progression_with_means_l1806_180678

theorem geometric_progression_with_means
  (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  let q := (b / a) ^ (1 / (n + 1 : ℝ))
  ∀ k : ℕ, ∃ r : ℝ, a * q ^ k = a * (b / a) ^ (k / (n + 1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_with_means_l1806_180678


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1806_180621

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x > 0) ↔ (∃ x : ℝ, x^2 - 2*x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1806_180621


namespace NUMINAMATH_CALUDE_lisa_photos_l1806_180699

/-- The number of photos Lisa took this weekend -/
def total_photos (animal_photos flower_photos scenery_photos : ℕ) : ℕ :=
  animal_photos + flower_photos + scenery_photos

theorem lisa_photos : 
  ∀ (animal_photos flower_photos scenery_photos : ℕ),
    animal_photos = 10 →
    flower_photos = 3 * animal_photos →
    scenery_photos = flower_photos - 10 →
    total_photos animal_photos flower_photos scenery_photos = 60 := by
  sorry

#check lisa_photos

end NUMINAMATH_CALUDE_lisa_photos_l1806_180699


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l1806_180690

theorem fraction_sum_equals_one (m : ℝ) (h : m ≠ 1) :
  m / (m - 1) + 1 / (1 - m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l1806_180690


namespace NUMINAMATH_CALUDE_problem_solution_l1806_180650

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (2 * x + 1)) :
  (3 * x - 3 * y + x * y) / (4 * x * y) = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1806_180650


namespace NUMINAMATH_CALUDE_problem_solution_l1806_180677

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

theorem problem_solution (k : ℝ) (h : k > 0) :
  (∀ x ∈ Set.Ioo 0 4, f_deriv k x < 0) → k = 1/3 ∧
  (∀ x ∈ Set.Ioo 0 4, f_deriv k x ≤ 0) ↔ 0 < k ∧ k ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1806_180677


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1806_180604

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x^2 + 7 * x = 3.5 * (x - 4)^2 + 1.5 * (x - 2) * (x - 4) + 18 * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1806_180604


namespace NUMINAMATH_CALUDE_train_length_problem_l1806_180613

/-- Proves that the length of each train is 25 meters given the specified conditions -/
theorem train_length_problem (speed_fast speed_slow : ℝ) (passing_time : ℝ) :
  speed_fast = 46 →
  speed_slow = 36 →
  passing_time = 18 →
  let relative_speed := (speed_fast - speed_slow) * (5 / 18)
  let train_length := (relative_speed * passing_time) / 2
  train_length = 25 := by
sorry


end NUMINAMATH_CALUDE_train_length_problem_l1806_180613


namespace NUMINAMATH_CALUDE_irrational_among_options_l1806_180668

theorem irrational_among_options : 
  (¬ (∃ (a b : ℤ), -Real.sqrt 3 = (a : ℚ) / (b : ℚ) ∧ b ≠ 0)) ∧
  (∃ (a b : ℤ), (-2 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) ∧
  (∃ (a b : ℤ), (0.1010 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) ∧
  (∃ (a b : ℤ), (1/3 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_irrational_among_options_l1806_180668


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l1806_180607

/-- Represents the salary distribution of a company -/
structure SalaryDistribution where
  ceo : ℕ × ℕ
  senior_manager : ℕ × ℕ
  manager : ℕ × ℕ
  assistant_manager : ℕ × ℕ
  clerk : ℕ × ℕ

/-- The total number of employees in the company -/
def total_employees (sd : SalaryDistribution) : ℕ :=
  sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 + sd.assistant_manager.1 + sd.clerk.1

/-- The median index in a list of salaries -/
def median_index (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Finds the median salary given a salary distribution -/
def median_salary (sd : SalaryDistribution) : ℕ :=
  let total := total_employees sd
  let median_idx := median_index total
  if median_idx ≤ sd.ceo.1 then sd.ceo.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 then sd.senior_manager.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 then sd.manager.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 + sd.assistant_manager.1 then sd.assistant_manager.2
  else sd.clerk.2

/-- The company's salary distribution -/
def company_salaries : SalaryDistribution := {
  ceo := (1, 140000),
  senior_manager := (4, 95000),
  manager := (15, 80000),
  assistant_manager := (7, 55000),
  clerk := (40, 25000)
}

theorem median_salary_is_25000 :
  median_salary company_salaries = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l1806_180607


namespace NUMINAMATH_CALUDE_number_divided_by_six_l1806_180644

theorem number_divided_by_six : ∃ n : ℝ, n / 6 = 26 ∧ n = 156 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_six_l1806_180644


namespace NUMINAMATH_CALUDE_paper_strip_division_l1806_180664

theorem paper_strip_division (total_fraction : ℚ) (num_books : ℕ) : 
  total_fraction = 5/8 ∧ num_books = 5 → 
  total_fraction / num_books = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_division_l1806_180664


namespace NUMINAMATH_CALUDE_descending_order_XYZ_l1806_180665

theorem descending_order_XYZ : ∀ (X Y Z : ℝ),
  X = 0.6 * 0.5 + 0.4 →
  Y = 0.6 * 0.5 / 0.4 →
  Z = 0.6 * 0.5 * 0.4 →
  Y > X ∧ X > Z :=
by
  sorry

end NUMINAMATH_CALUDE_descending_order_XYZ_l1806_180665


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1806_180636

/-- Given a right triangle with medians from acute angles 6 and √30, prove the hypotenuse is 2√52.8 -/
theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a^2 + b^2 = (a + b)^2 / 4)
  (h_median1 : b^2 + (a/2)^2 = 30) (h_median2 : a^2 + (b/2)^2 = 36) :
  (2*a)^2 + (2*b)^2 = 4 * 52.8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1806_180636


namespace NUMINAMATH_CALUDE_total_husk_bags_eaten_l1806_180695

-- Define the number of cows
def num_cows : ℕ := 26

-- Define the number of days
def num_days : ℕ := 26

-- Define the rate at which one cow eats husk
def cow_husk_rate : ℚ := 1 / 26

-- Theorem to prove
theorem total_husk_bags_eaten : 
  (num_cows : ℚ) * cow_husk_rate * (num_days : ℚ) = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_husk_bags_eaten_l1806_180695


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1806_180624

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) -- a is the arithmetic sequence
  (h1 : a 5 = 3) -- given condition: a_5 = 3
  (h2 : a 6 = -2) -- given condition: a_6 = -2
  : ∃ d : ℤ, d = -5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1806_180624


namespace NUMINAMATH_CALUDE_mona_unique_players_l1806_180609

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (total_groups : ℕ) (players_per_group : ℕ) (repeated_players : ℕ) : ℕ :=
  total_groups * players_per_group - repeated_players

/-- Theorem: Mona grouped with 33 unique players --/
theorem mona_unique_players :
  unique_players 9 4 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l1806_180609


namespace NUMINAMATH_CALUDE_floor_tiles_theorem_l1806_180688

/-- Represents a square floor divided into four congruent sections -/
structure SquareFloor :=
  (section_side : ℕ)

/-- The number of tiles on the main diagonal of the entire floor -/
def main_diagonal_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.section_side - 3

/-- The total number of tiles covering the entire floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  (4 * floor.section_side) ^ 2

/-- Theorem stating the relationship between the number of tiles on the main diagonal
    and the total number of tiles on the floor -/
theorem floor_tiles_theorem (floor : SquareFloor) 
  (h : main_diagonal_tiles floor = 75) : total_tiles floor = 25600 := by
  sorry

#check floor_tiles_theorem

end NUMINAMATH_CALUDE_floor_tiles_theorem_l1806_180688


namespace NUMINAMATH_CALUDE_total_cookies_l1806_180648

/-- The number of baking trays Lara is using. -/
def num_trays : ℕ := 4

/-- The number of rows of cookies on each tray. -/
def rows_per_tray : ℕ := 5

/-- The number of cookies in one row. -/
def cookies_per_row : ℕ := 6

/-- Theorem: The total number of cookies Lara is baking is 120. -/
theorem total_cookies : 
  num_trays * rows_per_tray * cookies_per_row = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1806_180648


namespace NUMINAMATH_CALUDE_inequality_proof_l1806_180643

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (abs a > abs b) ∧ (b / a < a / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1806_180643


namespace NUMINAMATH_CALUDE_divisors_of_500_l1806_180610

theorem divisors_of_500 : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n ∣ 500 ∧ 1 ≤ n ∧ n ≤ 500) ∧ 
    (∀ n, n ∣ 500 ∧ 1 ≤ n ∧ n ≤ 500 → n ∈ S) ∧ 
    Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_500_l1806_180610


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1806_180626

theorem age_ratio_problem (j e : ℕ) (h1 : j - 6 = 4 * (e - 6)) (h2 : j - 4 = 3 * (e - 4)) :
  ∃ x : ℕ, x = 14 ∧ (j + x) * 2 = (e + x) * 3 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1806_180626


namespace NUMINAMATH_CALUDE_mod_power_seventeen_seven_l1806_180620

theorem mod_power_seventeen_seven (m : ℕ) : 
  17^7 % 11 = m ∧ 0 ≤ m ∧ m < 11 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_power_seventeen_seven_l1806_180620


namespace NUMINAMATH_CALUDE_problem_solution_l1806_180628

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 0)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 1) :
  b / (a + b) + c / (b + c) + a / (c + a) = 5/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1806_180628


namespace NUMINAMATH_CALUDE_tanner_savings_l1806_180682

theorem tanner_savings (september_savings : ℤ) : 
  september_savings + 48 + 25 - 49 = 41 → september_savings = 17 := by
  sorry

end NUMINAMATH_CALUDE_tanner_savings_l1806_180682


namespace NUMINAMATH_CALUDE_briannes_yard_length_l1806_180635

theorem briannes_yard_length (derricks_length : ℝ) (alexs_length : ℝ) (briannes_length : ℝ) : 
  derricks_length = 10 →
  alexs_length = derricks_length / 2 →
  briannes_length = 6 * alexs_length →
  briannes_length = 30 := by sorry

end NUMINAMATH_CALUDE_briannes_yard_length_l1806_180635


namespace NUMINAMATH_CALUDE_max_notebooks_lucy_can_buy_l1806_180646

def lucy_money : ℕ := 2145
def notebook_cost : ℕ := 230

theorem max_notebooks_lucy_can_buy :
  (lucy_money / notebook_cost : ℕ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_lucy_can_buy_l1806_180646


namespace NUMINAMATH_CALUDE_plan_y_cost_effective_l1806_180600

/-- The cost in cents for Plan X given the number of minutes used -/
def planXCost (minutes : ℕ) : ℕ := 15 * minutes

/-- The cost in cents for Plan Y given the number of minutes used -/
def planYCost (minutes : ℕ) : ℕ := 3000 + 10 * minutes

/-- The minimum number of minutes for Plan Y to be cost-effective -/
def minMinutes : ℕ := 601

theorem plan_y_cost_effective : 
  ∀ m : ℕ, m ≥ minMinutes → planYCost m < planXCost m :=
by
  sorry

#check plan_y_cost_effective

end NUMINAMATH_CALUDE_plan_y_cost_effective_l1806_180600


namespace NUMINAMATH_CALUDE_m_range_l1806_180655

theorem m_range (m : ℝ) :
  (m^2 + m)^(3/5) ≤ (3 - m)^(3/5) → -3 ≤ m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_l1806_180655


namespace NUMINAMATH_CALUDE_rhombus_area_l1806_180673

/-- The area of a rhombus with diagonals measuring 9 cm and 14 cm is 63 square centimeters. -/
theorem rhombus_area (d1 d2 area : ℝ) : 
  d1 = 9 → d2 = 14 → area = (d1 * d2) / 2 → area = 63 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1806_180673


namespace NUMINAMATH_CALUDE_cosine_arcsine_tangent_arccos_equation_l1806_180632

theorem cosine_arcsine_tangent_arccos_equation :
  ∃! x : ℝ, x ∈ [(-1 : ℝ), 1] ∧
    Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x ∧
    x = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_arcsine_tangent_arccos_equation_l1806_180632


namespace NUMINAMATH_CALUDE_chinese_spanish_difference_l1806_180614

def hours_english : ℕ := 2
def hours_chinese : ℕ := 5
def hours_spanish : ℕ := 4

theorem chinese_spanish_difference : hours_chinese - hours_spanish = 1 := by
  sorry

end NUMINAMATH_CALUDE_chinese_spanish_difference_l1806_180614


namespace NUMINAMATH_CALUDE_crescent_area_equals_rectangle_area_l1806_180662

theorem crescent_area_equals_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_area := 4 * a * b
  let circle_area := π * (a^2 + b^2)
  let semicircles_area := π * a^2 + π * b^2
  let crescent_area := semicircles_area + rectangle_area - circle_area
  crescent_area = rectangle_area := by
  sorry

end NUMINAMATH_CALUDE_crescent_area_equals_rectangle_area_l1806_180662


namespace NUMINAMATH_CALUDE_positive_expression_l1806_180674

theorem positive_expression (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) :
  0 < b + a^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l1806_180674


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l1806_180639

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

theorem min_value_reciprocal_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b = 3 + 2 * Real.sqrt 2) ↔ (b / a = 2 * a / b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l1806_180639


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l1806_180629

/-- Calculates the total cost of tickets sold at a snooker tournament --/
theorem snooker_tournament_ticket_sales 
  (total_tickets : ℕ) 
  (vip_price general_price : ℚ) 
  (ticket_difference : ℕ) 
  (h1 : total_tickets = 320)
  (h2 : vip_price = 45)
  (h3 : general_price = 20)
  (h4 : ticket_difference = 276) :
  let general_tickets := (total_tickets + ticket_difference) / 2
  let vip_tickets := total_tickets - general_tickets
  vip_price * vip_tickets + general_price * general_tickets = 6950 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l1806_180629


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1806_180616

/-- Given a quadratic function f(x) = ax^2 - x + c with range [0, +∞),
    the minimum value of 2/a + 2/c is 8 -/
theorem min_value_sum_reciprocals (a c : ℝ) : 
  (∀ x, ∃ y ≥ 0, y = a * x^2 - x + c) →
  (∃ x, a * x^2 - x + c = 0) →
  (∀ x, a * x^2 - x + c ≥ 0) →
  (2 / a + 2 / c ≥ 8) ∧ (∃ a c, 2 / a + 2 / c = 8) := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1806_180616


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1806_180642

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area (a b h x : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  let area := x * (a - b) * (h - x) / h
  area = x * (a - b) * (h - x) / h :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1806_180642


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1806_180681

theorem absolute_value_simplification : |(-5^2 + 7 - 3)| = 21 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1806_180681


namespace NUMINAMATH_CALUDE_dataset_transformation_l1806_180619

/-- Represents a dataset with mean and variance -/
structure Dataset where
  mean : ℝ
  variance : ℝ

/-- Represents the transformation of adding a constant to each data point -/
def add_constant (d : Dataset) (c : ℝ) : Dataset :=
  { mean := d.mean + c,
    variance := d.variance }

theorem dataset_transformation (d : Dataset) :
  d.mean = 2.8 →
  d.variance = 3.6 →
  (add_constant d 60).mean = 62.8 ∧ (add_constant d 60).variance = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_dataset_transformation_l1806_180619


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l1806_180601

def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

theorem parallel_vectors_imply_x_equals_two (x : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b x) = k • (4 • (b x) - 2 • a)) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l1806_180601


namespace NUMINAMATH_CALUDE_money_sharing_problem_l1806_180679

theorem money_sharing_problem (amanda_ratio ben_ratio carlos_ratio : ℕ) 
  (ben_share : ℕ) (total : ℕ) : 
  amanda_ratio = 3 → 
  ben_ratio = 5 → 
  carlos_ratio = 8 → 
  ben_share = 25 → 
  total = amanda_ratio * (ben_share / ben_ratio) + 
          ben_share + 
          carlos_ratio * (ben_share / ben_ratio) → 
  total = 80 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l1806_180679


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1806_180653

theorem sufficient_not_necessary :
  (∀ x : ℝ, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1806_180653


namespace NUMINAMATH_CALUDE_fraction_less_than_mode_l1806_180659

def data_list : List ℕ := [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than (l : List ℕ) (n : ℕ) : ℕ :=
  l.filter (· < n) |>.length

theorem fraction_less_than_mode :
  (count_less_than data_list (mode data_list) : ℚ) / data_list.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_mode_l1806_180659


namespace NUMINAMATH_CALUDE_ln_1_1_approx_fourth_root_17_approx_l1806_180637

-- Define the required accuracy
def accuracy : ℝ := 0.0001

-- Theorem for ln(1.1)
theorem ln_1_1_approx : |Real.log 1.1 - 0.0953| < accuracy := by sorry

-- Theorem for ⁴√17
theorem fourth_root_17_approx : |((17 : ℝ) ^ (1/4)) - 2.0305| < accuracy := by sorry

end NUMINAMATH_CALUDE_ln_1_1_approx_fourth_root_17_approx_l1806_180637


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l1806_180622

theorem modulus_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (3 - i) * (1 + 3*i)
  Complex.abs z = 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l1806_180622


namespace NUMINAMATH_CALUDE_point_transformation_l1806_180670

-- Define the transformations
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (rotate_x_90 (reflect_xy (rotate_z_90 p)))

-- Theorem statement
theorem point_transformation :
  transform (2, 3, 4) = (3, 4, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1806_180670


namespace NUMINAMATH_CALUDE_quadratic_rational_solution_l1806_180611

/-- The quadratic equation kx^2 + 16x + k = 0 has rational solutions if and only if k = 8, where k is a positive integer. -/
theorem quadratic_rational_solution (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rational_solution_l1806_180611


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1806_180663

theorem cube_volume_surface_area (y : ℝ) (h1 : y > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*y ∧ 6*s^2 = 6*y) → y = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1806_180663


namespace NUMINAMATH_CALUDE_xyz_sum_l1806_180689

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = y.val * z.val + x.val)
  (h2 : y.val * z.val + x.val = z.val * x.val + y.val)
  (h3 : z.val * x.val + y.val = x.val * y.val + z.val)
  (h4 : x.val * y.val + z.val = 56) : 
  x.val + y.val + z.val = 21 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l1806_180689


namespace NUMINAMATH_CALUDE_arrangement_count_l1806_180658

theorem arrangement_count (volunteers : ℕ) (elderly : ℕ) : 
  volunteers = 4 ∧ elderly = 1 → (volunteers.factorial : ℕ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1806_180658


namespace NUMINAMATH_CALUDE_sequence_inequality_l1806_180634

theorem sequence_inequality (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n, a (n + 2) ≤ (2023 * a n) / (a n * a (n + 1) + 2023)) :
  a 2023 < 1 ∨ a 2024 < 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1806_180634


namespace NUMINAMATH_CALUDE_y_value_proof_l1806_180660

theorem y_value_proof (y : ℝ) (h : 9 / (y^3) = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1806_180660


namespace NUMINAMATH_CALUDE_bus_fare_difference_sam_alex_l1806_180657

/-- The cost difference between two people's bus fares for a given number of trips -/
def busFareDifference (alexFare samFare : ℚ) (numTrips : ℕ) : ℚ :=
  numTrips * (samFare - alexFare)

/-- Theorem stating the cost difference between Sam and Alex's bus fares for 20 trips -/
theorem bus_fare_difference_sam_alex :
  busFareDifference (25/10) 3 20 = 15 := by sorry

end NUMINAMATH_CALUDE_bus_fare_difference_sam_alex_l1806_180657


namespace NUMINAMATH_CALUDE_cube_angles_l1806_180669

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Calculates the angle between two skew lines in a cube -/
def angle_between_skew_lines (c : Cube) (l1 l2 : Fin 2 → Fin 8) : ℝ :=
  sorry

/-- Calculates the angle between a line and a plane in a cube -/
def angle_between_line_and_plane (c : Cube) (l : Fin 2 → Fin 8) (p : Fin 4 → Fin 8) : ℝ :=
  sorry

/-- Theorem stating the angles in a cube -/
theorem cube_angles (c : Cube) : 
  angle_between_skew_lines c ![7, 1] ![0, 2] = 60 ∧ 
  angle_between_line_and_plane c ![7, 1] ![7, 5, 2, 3] = 30 :=
sorry

end NUMINAMATH_CALUDE_cube_angles_l1806_180669


namespace NUMINAMATH_CALUDE_total_spots_l1806_180692

/-- The number of spots on each dog -/
structure DogSpots where
  rover : ℕ
  cisco : ℕ
  granger : ℕ
  sparky : ℕ
  bella : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (d : DogSpots) : Prop :=
  d.rover = 46 ∧
  d.cisco = d.rover / 2 - 5 ∧
  d.granger = 5 * d.cisco ∧
  d.sparky = 3 * d.rover ∧
  d.bella = 2 * (d.granger + d.sparky)

/-- The theorem to be proven -/
theorem total_spots (d : DogSpots) (h : satisfiesConditions d) : 
  d.granger + d.cisco + d.sparky + d.bella = 702 := by
  sorry

end NUMINAMATH_CALUDE_total_spots_l1806_180692


namespace NUMINAMATH_CALUDE_is_rectangle_l1806_180652

/-- Given points A, B, C, and D in a 2D plane, prove that ABCD is a rectangle -/
theorem is_rectangle (A B C D : ℝ × ℝ) : 
  A = (-2, 0) → B = (1, 6) → C = (5, 4) → D = (2, -2) →
  (B.1 - A.1, B.2 - A.2) = (C.1 - D.1, C.2 - D.2) ∧
  (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0 := by
  sorry

#check is_rectangle

end NUMINAMATH_CALUDE_is_rectangle_l1806_180652


namespace NUMINAMATH_CALUDE_total_players_is_fifty_l1806_180661

/-- The number of cricket players -/
def cricket_players : ℕ := 12

/-- The number of hockey players -/
def hockey_players : ℕ := 17

/-- The number of football players -/
def football_players : ℕ := 11

/-- The number of softball players -/
def softball_players : ℕ := 10

/-- The total number of players on the ground -/
def total_players : ℕ := cricket_players + hockey_players + football_players + softball_players

/-- Theorem stating that the total number of players is 50 -/
theorem total_players_is_fifty : total_players = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_players_is_fifty_l1806_180661


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1806_180680

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4
def circle_N (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define the property of being externally tangent
def externally_tangent (x y R : ℝ) : Prop :=
  ∃ (x_m y_m : ℝ), circle_M x_m y_m ∧ (x - x_m)^2 + (y - y_m)^2 = (R + 2)^2

-- Define the property of being internally tangent
def internally_tangent (x y R : ℝ) : Prop :=
  ∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = (10 - R)^2

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y R : ℝ),
    externally_tangent x y R →
    internally_tangent x y R →
    x^2 / 36 + y^2 / 27 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1806_180680


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1806_180666

/-- Number of diagonals in a convex polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1806_180666


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1806_180612

theorem binomial_expansion_example : 57^3 + 3*(57^2)*4 + 3*57*(4^2) + 4^3 = 226981 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1806_180612


namespace NUMINAMATH_CALUDE_part_one_part_two_l1806_180618

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem for the first part
theorem part_one (m : ℝ) : 
  (∀ x : ℝ, m + f x > 0) ↔ m > -2 :=
sorry

-- Theorem for the second part
theorem part_two (m : ℝ) :
  (∃ x : ℝ, m - f x > 0) ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1806_180618


namespace NUMINAMATH_CALUDE_polygon_sides_l1806_180603

theorem polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 40) :
  (360 : ℝ) / exterior_angle = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1806_180603


namespace NUMINAMATH_CALUDE_circle_locus_is_spherical_triangle_l1806_180684

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  vertex : Point3D
  isRightAngled : Bool

/-- The locus of circle centers touching the faces of a right-angled trihedral angle -/
def circleLocus (t : TrihedralAngle) (r : ℝ) : Set Point3D :=
  {p : Point3D | ∃ (c : Circle3D), c.radius = r ∧ 
    c.center = p ∧ 
    (c.center.x ≤ r ∧ c.center.y ≤ r ∧ c.center.z ≤ r) ∧
    (c.center.x ≥ 0 ∧ c.center.y ≥ 0 ∧ c.center.z ≥ 0) ∧
    (c.center.x ^ 2 + c.center.y ^ 2 + c.center.z ^ 2 = 2 * r ^ 2)}

theorem circle_locus_is_spherical_triangle (t : TrihedralAngle) (r : ℝ) 
  (h : t.isRightAngled = true) :
  circleLocus t r = {p : Point3D | 
    p.x ^ 2 + p.y ^ 2 + p.z ^ 2 = 2 * r ^ 2 ∧
    p.x ≤ r ∧ p.y ≤ r ∧ p.z ≤ r ∧
    p.x ≥ 0 ∧ p.y ≥ 0 ∧ p.z ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_circle_locus_is_spherical_triangle_l1806_180684


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l1806_180627

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, f a x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f a x = -3) →
  a = Real.sqrt 5 ∨ a = -Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l1806_180627


namespace NUMINAMATH_CALUDE_vertex_of_given_function_l1806_180676

/-- A quadratic function of the form y = a(x - h)^2 + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a quadratic function -/
def vertex (f : QuadraticFunction) : ℝ × ℝ := (f.h, f.k)

/-- The given quadratic function y = -2(x+1)^2 + 5 -/
def given_function : QuadraticFunction := ⟨-2, -1, 5⟩

theorem vertex_of_given_function :
  vertex given_function = (-1, 5) := by sorry

end NUMINAMATH_CALUDE_vertex_of_given_function_l1806_180676


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1806_180625

def T : Finset Nat := Finset.range 15

def disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (T : Finset Nat) (h : T = Finset.range 15) :
  disjoint_subsets T % 1000 = 686 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1806_180625


namespace NUMINAMATH_CALUDE_count_pairs_eq_27_l1806_180623

open Set

def S : Finset Char := {'a', 'b', 'c'}

/-- The number of ordered pairs (A, B) of subsets of S such that A ∪ B = S and A ≠ B -/
def count_pairs : ℕ :=
  (Finset.powerset S).card * (Finset.powerset S).card -
  (Finset.powerset S).card

theorem count_pairs_eq_27 : count_pairs = 27 := by sorry

end NUMINAMATH_CALUDE_count_pairs_eq_27_l1806_180623


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1806_180671

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  0 < b → b < a → a + b = 7 * (a - b) → a / b = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1806_180671


namespace NUMINAMATH_CALUDE_sin_cos_sum_zero_l1806_180630

theorem sin_cos_sum_zero : 
  Real.sin (35 * π / 6) + Real.cos (-11 * π / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_zero_l1806_180630


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_negative_one_l1806_180696

theorem purely_imaginary_implies_a_equals_negative_one :
  ∀ (a : ℝ), (Complex.I * (a - 1) : ℂ).im ≠ 0 →
  (a^2 - 1 + Complex.I * (a - 1) : ℂ).re = 0 →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_negative_one_l1806_180696


namespace NUMINAMATH_CALUDE_min_fraction_value_l1806_180645

theorem min_fraction_value (x y : ℝ) (h : Real.sqrt (x - 1) + Real.sqrt (y - 1) = 1) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (z : ℝ), z = x/y → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_value_l1806_180645


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1806_180683

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 4 →
  set1_mean = 10 →
  set2_count = 8 →
  set2_mean = 21 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 52 / 3 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1806_180683


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1806_180606

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  6 * x + 1 / (x^2) ≥ 7 * (6^(1/3)) := by
  sorry

theorem equality_condition : 
  6 * ((1/6)^(1/3)) + 1 / (((1/6)^(1/3))^2) = 7 * (6^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1806_180606


namespace NUMINAMATH_CALUDE_hilt_fountain_trips_l1806_180631

/-- The number of trips to the water fountain -/
def number_of_trips (distance_to_fountain : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / distance_to_fountain

/-- Theorem: Mrs. Hilt will go to the water fountain 4 times -/
theorem hilt_fountain_trips :
  let distance_to_fountain : ℕ := 30
  let total_distance : ℕ := 120
  number_of_trips distance_to_fountain total_distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_hilt_fountain_trips_l1806_180631


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1806_180608

theorem pure_imaginary_product (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 4 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1806_180608


namespace NUMINAMATH_CALUDE_postcard_height_l1806_180697

theorem postcard_height (perimeter width : ℝ) (h_perimeter : perimeter = 20) (h_width : width = 6) :
  let height := (perimeter - 2 * width) / 2
  height = 4 := by sorry

end NUMINAMATH_CALUDE_postcard_height_l1806_180697


namespace NUMINAMATH_CALUDE_parabola_vertex_fourth_quadrant_l1806_180640

/-- A parabola with equation y = -2(x+a)^2 + c -/
structure Parabola (a c : ℝ) where
  equation : ℝ → ℝ
  eq_def : ∀ x, equation x = -2 * (x + a)^2 + c

/-- The vertex of a parabola -/
def vertex (p : Parabola a c) : ℝ × ℝ := (-a, c)

/-- A point is in the fourth quadrant if its x-coordinate is positive and y-coordinate is negative -/
def in_fourth_quadrant (point : ℝ × ℝ) : Prop :=
  point.1 > 0 ∧ point.2 < 0

/-- Theorem: For a parabola y = -2(x+a)^2 + c with its vertex in the fourth quadrant, a < 0 and c < 0 -/
theorem parabola_vertex_fourth_quadrant {a c : ℝ} (p : Parabola a c) 
  (h : in_fourth_quadrant (vertex p)) : a < 0 ∧ c < 0 := by
  sorry


end NUMINAMATH_CALUDE_parabola_vertex_fourth_quadrant_l1806_180640


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1806_180694

/-- Proves that a boat's speed in still water is 20 km/hr given specific conditions -/
theorem boat_speed_in_still_water :
  let current_speed : ℝ := 3
  let downstream_distance : ℝ := 9.2
  let downstream_time : ℝ := 24 / 60
  let downstream_speed : ℝ → ℝ := λ v => v + current_speed
  ∃ (v : ℝ), downstream_speed v * downstream_time = downstream_distance ∧ v = 20 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1806_180694


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1806_180691

theorem min_value_expression (x : ℝ) (h : x > 0) :
  x^2 / x + 2 + 5 / x ≥ 2 * Real.sqrt 5 + 2 :=
sorry

theorem min_value_achievable :
  ∃ (x : ℝ), x > 0 ∧ x^2 / x + 2 + 5 / x = 2 * Real.sqrt 5 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1806_180691


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l1806_180687

/-- Represents a repeating decimal with a repeating part of length 2 -/
def RepeatingDecimal2 (a b : ℕ) : ℚ :=
  (a * 10 + b : ℚ) / 99

/-- Represents a repeating decimal with a repeating part of length 1 -/
def RepeatingDecimal1 (a : ℕ) : ℚ :=
  (a : ℚ) / 9

/-- The product of 0.overline{03} and 0.overline{3} is equal to 1/99 -/
theorem product_of_repeating_decimals :
  (RepeatingDecimal2 0 3) * (RepeatingDecimal1 3) = 1 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l1806_180687


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_to_12000_l1806_180651

theorem last_three_digits_of_3_to_12000 (h : 3^400 ≡ 1 [ZMOD 1000]) :
  3^12000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_to_12000_l1806_180651


namespace NUMINAMATH_CALUDE_paper_products_distribution_l1806_180615

theorem paper_products_distribution (total : ℕ) 
  (h1 : total = 20)
  (h2 : total / 2 + total / 4 + total / 5 < total) :
  total - (total / 2 + total / 4 + total / 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_products_distribution_l1806_180615


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1806_180685

theorem arithmetic_calculations : 
  (2 - 7 * (-3) + 10 + (-2) = 31) ∧ 
  (-1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1806_180685


namespace NUMINAMATH_CALUDE_unique_integer_pair_for_equal_W_values_l1806_180698

/-- The polynomial W(x) = x^4 - 3x^3 + 5x^2 - 9x -/
def W (x : ℤ) : ℤ := x^4 - 3*x^3 + 5*x^2 - 9*x

/-- Theorem: The only pair of different integers (a, b) satisfying W(a) = W(b) is (1, 2) -/
theorem unique_integer_pair_for_equal_W_values :
  ∀ a b : ℤ, a ≠ b ∧ W a = W b ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_pair_for_equal_W_values_l1806_180698


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_l1806_180667

theorem no_real_solution_for_equation :
  ¬ ∃ x : ℝ, (Real.sqrt (4 * x + 2) + 1) / Real.sqrt (8 * x + 10) = 2 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_l1806_180667


namespace NUMINAMATH_CALUDE_binomial_sum_even_n_l1806_180649

theorem binomial_sum_even_n (n : ℕ) (h : Even n) :
  (Finset.sum (Finset.range (n + 1)) (fun k =>
    if k % 2 = 0 then (1 : ℕ) * Nat.choose n k
    else 2 * Nat.choose n k)) = 3 * 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_even_n_l1806_180649


namespace NUMINAMATH_CALUDE_triangle_side_length_l1806_180602

-- Define the triangle DEF
structure Triangle (D E F : ℝ) where
  -- Angle sum property of a triangle
  angle_sum : D + E + F = Real.pi

-- Define the main theorem
theorem triangle_side_length 
  (D E F : ℝ) 
  (t : Triangle D E F) 
  (h1 : Real.cos (3 * D - E) + Real.sin (D + E) = 2) 
  (h2 : 6 = 6) :  -- DE = 6, but we use 6 = 6 as Lean doesn't know DE yet
  ∃ (EF : ℝ), EF = 3 * Real.sqrt (2 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1806_180602


namespace NUMINAMATH_CALUDE_recurrence_sequence_a9_l1806_180617

/-- An increasing sequence of positive integers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, 1 ≤ n → a (n + 2) = a (n + 1) + a n)

theorem recurrence_sequence_a9 (a : ℕ → ℕ) (h : RecurrenceSequence a) (h6 : a 6 = 56) :
  a 9 = 270 := by
  sorry

#check recurrence_sequence_a9

end NUMINAMATH_CALUDE_recurrence_sequence_a9_l1806_180617


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1806_180633

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1806_180633


namespace NUMINAMATH_CALUDE_common_tangent_lower_bound_l1806_180686

/-- Given two curves C₁: y = ax² (a > 0) and C₂: y = e^x, if they have a common tangent line,
    then a ≥ e²/4 -/
theorem common_tangent_lower_bound (a : ℝ) (h_pos : a > 0) :
  (∃ x₁ x₂ : ℝ, (2 * a * x₁ = Real.exp x₂) ∧ 
                (a * x₁^2 - Real.exp x₂ = 2 * a * x₁ * (x₁ - x₂))) →
  a ≥ Real.exp 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_common_tangent_lower_bound_l1806_180686


namespace NUMINAMATH_CALUDE_shorter_base_length_l1806_180654

/-- A trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midline_length : ℝ

/-- The property that the line joining the midpoints of the diagonals 
    is half the difference of the bases -/
def midline_property (t : Trapezoid) : Prop :=
  t.midline_length = (t.long_base - t.short_base) / 2

/-- Theorem stating the length of the shorter base given the conditions -/
theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.long_base = 115)
  (h2 : t.midline_length = 6)
  (h3 : midline_property t) : 
  t.short_base = 103 := by
sorry

end NUMINAMATH_CALUDE_shorter_base_length_l1806_180654


namespace NUMINAMATH_CALUDE_max_boat_shipments_l1806_180656

theorem max_boat_shipments (B : ℕ) (h1 : B ≥ 120) (h2 : B % 24 = 0) :
  ∃ S : ℕ, S ≠ 24 ∧ B % S = 0 ∧ ∀ T : ℕ, T ≠ 24 → B % T = 0 → T ≤ S :=
by
  sorry

end NUMINAMATH_CALUDE_max_boat_shipments_l1806_180656


namespace NUMINAMATH_CALUDE_ellie_and_hank_weight_l1806_180693

/-- The weights of Ellie, Frank, Gina, and Hank satisfy the given conditions
    and Ellie and Hank weigh 355 pounds together. -/
theorem ellie_and_hank_weight (e f g h : ℝ) 
    (ef_sum : e + f = 310)
    (fg_sum : f + g = 280)
    (gh_sum : g + h = 325)
    (g_minus_h : g = h + 10) :
  e + h = 355 := by
  sorry

end NUMINAMATH_CALUDE_ellie_and_hank_weight_l1806_180693


namespace NUMINAMATH_CALUDE_rabbit_measurement_probability_l1806_180675

theorem rabbit_measurement_probability :
  let total_rabbits : ℕ := 5
  let measured_rabbits : ℕ := 3
  let selected_rabbits : ℕ := 3
  let favorable_outcomes : ℕ := (measured_rabbits.choose 2) * ((total_rabbits - measured_rabbits).choose 1)
  let total_outcomes : ℕ := total_rabbits.choose selected_rabbits
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_rabbit_measurement_probability_l1806_180675


namespace NUMINAMATH_CALUDE_ram_selection_probability_l1806_180641

/-- Given two brothers Ram and Ravi, where the probability of Ravi's selection is 1/5
    and the probability of both being selected is 0.11428571428571428,
    prove that the probability of Ram's selection is 0.5714285714285714 -/
theorem ram_selection_probability
  (p_ravi : ℝ)
  (p_both : ℝ)
  (h1 : p_ravi = 1 / 5)
  (h2 : p_both = 0.11428571428571428) :
  p_both / p_ravi = 0.5714285714285714 := by
  sorry

end NUMINAMATH_CALUDE_ram_selection_probability_l1806_180641


namespace NUMINAMATH_CALUDE_at_least_one_positive_l1806_180672

theorem at_least_one_positive (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (x : ℝ) (hx : x = a^2 - b*c)
  (y : ℝ) (hy : y = b^2 - c*a)
  (z : ℝ) (hz : z = c^2 - a*b) :
  x > 0 ∨ y > 0 ∨ z > 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l1806_180672


namespace NUMINAMATH_CALUDE_line_equation_from_parametric_l1806_180638

/-- The equation of a line parameterized by (3t + 6, 5t - 7) is y = (5/3)x - 17 -/
theorem line_equation_from_parametric : 
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 → y = (5/3) * x - 17 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_parametric_l1806_180638


namespace NUMINAMATH_CALUDE_sum_of_products_l1806_180605

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 12)
  (h2 : y^2 + y*z + z^2 = 25)
  (h3 : z^2 + x*z + x^2 = 37) :
  x*y + y*z + x*z = 20 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_products_l1806_180605
