import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l3548_354840

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l3548_354840


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3548_354842

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 7*x - 12) → (∃ y : ℝ, y^2 = 7*y - 12 ∧ x + y = 7) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3548_354842


namespace NUMINAMATH_CALUDE_meeting_time_correct_l3548_354817

/-- Represents the time in hours after 7:00 AM -/
def time_after_seven (hours minutes : ℕ) : ℚ :=
  hours + minutes / 60

/-- The problem setup -/
structure TravelProblem where
  julia_speed : ℚ
  mark_speed : ℚ
  total_distance : ℚ
  mark_departure_time : ℚ

/-- The solution to the problem -/
def meeting_time (p : TravelProblem) : ℚ :=
  (p.total_distance + p.mark_speed * p.mark_departure_time) / (p.julia_speed + p.mark_speed)

/-- The theorem statement -/
theorem meeting_time_correct (p : TravelProblem) : 
  p.julia_speed = 15 ∧ 
  p.mark_speed = 20 ∧ 
  p.total_distance = 85 ∧ 
  p.mark_departure_time = 0.75 →
  meeting_time p = time_after_seven 2 51 := by
  sorry

#eval time_after_seven 2 51

end NUMINAMATH_CALUDE_meeting_time_correct_l3548_354817


namespace NUMINAMATH_CALUDE_odd_function_sum_l3548_354814

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : IsOdd f) (h_f_neg_one : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3548_354814


namespace NUMINAMATH_CALUDE_alpha_value_l3548_354892

theorem alpha_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) 
  (h_min : ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → 1/m + 16/n ≤ 1/x + 16/y) 
  (h_curve : (m/5)^α = m/4) : α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3548_354892


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3548_354864

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m % 17 = 0 → 10 ≤ m → m ≤ 99 → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3548_354864


namespace NUMINAMATH_CALUDE_book_price_change_l3548_354877

theorem book_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 + 20 / 100) = P * (1 + 16 / 100) → 
  x = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_book_price_change_l3548_354877


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3548_354857

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 3 = 2 → m ≥ n) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3548_354857


namespace NUMINAMATH_CALUDE_solution_set_and_range_l3548_354879

def f (x a : ℝ) : ℝ := |x - a| + 2 * x

theorem solution_set_and_range :
  (∀ x : ℝ, f x (-1) ≤ 0 ↔ x ≤ -1/3) ∧
  ((∀ x : ℝ, x ≥ -1 → f x a ≥ 0) → (a ≤ -3 ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l3548_354879


namespace NUMINAMATH_CALUDE_horse_speed_l3548_354836

/-- Given a square field with area 1600 km^2 and a horse that takes 10 hours to run around it,
    the speed of the horse is 16 km/h. -/
theorem horse_speed (field_area : ℝ) (run_time : ℝ) (horse_speed : ℝ) : 
  field_area = 1600 → run_time = 10 → horse_speed = (4 * Real.sqrt field_area) / run_time → 
  horse_speed = 16 := by sorry

end NUMINAMATH_CALUDE_horse_speed_l3548_354836


namespace NUMINAMATH_CALUDE_square_perimeter_unchanged_l3548_354849

/-- The perimeter of a square with side length 5 remains unchanged after cutting out four small rectangles from its corners. -/
theorem square_perimeter_unchanged (side_length : ℝ) (h : side_length = 5) :
  let original_perimeter := 4 * side_length
  let modified_perimeter := original_perimeter
  modified_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_unchanged_l3548_354849


namespace NUMINAMATH_CALUDE_loop_iterations_count_l3548_354841

theorem loop_iterations_count (i : ℕ) : 
  i = 1 → (∀ j, j ≥ 1 ∧ j < 21 → i + j = j + 1) → i + 20 = 21 :=
by sorry

end NUMINAMATH_CALUDE_loop_iterations_count_l3548_354841


namespace NUMINAMATH_CALUDE_machine_value_is_35000_l3548_354896

/-- Represents the denomination of a bill in dollars -/
inductive BillType
  | five
  | ten
  | twenty

/-- Returns the value of a bill in dollars -/
def billValue : BillType → Nat
  | BillType.five => 5
  | BillType.ten => 10
  | BillType.twenty => 20

/-- Represents a bundle of bills -/
structure Bundle where
  billType : BillType
  count : Nat

/-- Represents a cash machine -/
structure CashMachine where
  bundles : List Bundle

/-- The number of bills in each bundle -/
def billsPerBundle : Nat := 100

/-- The number of bundles for each bill type -/
def bundlesPerType : Nat := 10

/-- Calculates the total value of a bundle -/
def bundleValue (b : Bundle) : Nat :=
  billValue b.billType * b.count

/-- Calculates the total value of all bundles in the machine -/
def machineValue (m : CashMachine) : Nat :=
  m.bundles.map bundleValue |>.sum

/-- The cash machine configuration -/
def filledMachine : CashMachine :=
  { bundles := [
    { billType := BillType.five, count := billsPerBundle },
    { billType := BillType.ten, count := billsPerBundle },
    { billType := BillType.twenty, count := billsPerBundle }
  ] }

/-- Theorem: The total amount of money required to fill the machine is $35,000 -/
theorem machine_value_is_35000 : 
  machineValue filledMachine = 35000 := by sorry

end NUMINAMATH_CALUDE_machine_value_is_35000_l3548_354896


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l3548_354850

theorem factorial_fraction_simplification (N : ℕ) (h : N ≥ 2) :
  (Nat.factorial (N - 2) * N * (N - 1)) / Nat.factorial (N + 2) = 1 / ((N + 1) * (N + 2)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l3548_354850


namespace NUMINAMATH_CALUDE_investment_average_interest_rate_l3548_354882

/-- Proves that given a total investment split into two parts with different interest rates
    and equal annual returns, the average interest rate is as calculated. -/
theorem investment_average_interest_rate 
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h_total : total_investment = 4500)
  (h_rates : rate1 = 0.04 ∧ rate2 = 0.06)
  (h_equal_returns : ∃ (x : ℝ), 
    x > 0 ∧ x < total_investment ∧
    rate1 * (total_investment - x) = rate2 * x) :
  (rate1 * (total_investment - x) + rate2 * x) / total_investment = 0.048 := by
  sorry

#check investment_average_interest_rate

end NUMINAMATH_CALUDE_investment_average_interest_rate_l3548_354882


namespace NUMINAMATH_CALUDE_probability_one_suit_each_probability_calculation_correct_l3548_354861

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards drawn -/
def NumberOfDraws : ℕ := 4

/-- Represents the probability of drawing one card from each suit in four draws with replacement -/
def ProbabilityOneSuitEach : ℚ := 3 / 32

/-- Theorem stating that the probability of drawing one card from each suit
    in four draws with replacement from a standard 52-card deck is 3/32 -/
theorem probability_one_suit_each :
  (3 / 4 : ℚ) * (1 / 2 : ℚ) * (1 / 4 : ℚ) = ProbabilityOneSuitEach :=
by sorry

/-- Theorem stating that the calculated probability is correct -/
theorem probability_calculation_correct :
  ProbabilityOneSuitEach = (3 : ℚ) / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_suit_each_probability_calculation_correct_l3548_354861


namespace NUMINAMATH_CALUDE_marie_lost_erasers_l3548_354827

/-- The number of erasers Marie lost -/
def erasers_lost (initial final : ℕ) : ℕ := initial - final

/-- Theorem stating that Marie lost 42 erasers -/
theorem marie_lost_erasers : 
  let initial := 95
  let final := 53
  erasers_lost initial final = 42 := by
sorry

end NUMINAMATH_CALUDE_marie_lost_erasers_l3548_354827


namespace NUMINAMATH_CALUDE_tribe_leadership_organization_l3548_354848

def tribe_size : ℕ := 12
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 3
def inferior_officers_per_chief : ℕ := 2

theorem tribe_leadership_organization :
  (tribe_size.choose num_chiefs) *
  ((tribe_size - num_chiefs).choose 1) *
  ((tribe_size - num_chiefs - 1).choose 1) *
  ((tribe_size - num_chiefs - 2).choose 1) *
  ((tribe_size - num_chiefs - num_supporting_chiefs).choose inferior_officers_per_chief) *
  ((tribe_size - num_chiefs - num_supporting_chiefs - inferior_officers_per_chief).choose inferior_officers_per_chief) *
  ((tribe_size - num_chiefs - num_supporting_chiefs - 2 * inferior_officers_per_chief).choose inferior_officers_per_chief) = 1069200 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_organization_l3548_354848


namespace NUMINAMATH_CALUDE_problem_solution_l3548_354878

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b + 3) :
  (∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 2 * x * y = x + 2 * y + 3 → x + 2 * y ≥ min) ∧
  (¬ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * x * y = x + 2 * y + 3 ∧ x^2 + 4 * y^2 = 17) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3548_354878


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3548_354854

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3548_354854


namespace NUMINAMATH_CALUDE_mr_callen_wooden_toys_solution_is_eight_l3548_354883

/-- Proves that the number of wooden toys bought is 8, given the conditions of Mr. Callen's purchase and sale. -/
theorem mr_callen_wooden_toys : ℕ :=
  let num_paintings : ℕ := 10
  let painting_cost : ℚ := 40
  let toy_cost : ℚ := 20
  let painting_discount : ℚ := 0.1
  let toy_discount : ℚ := 0.15
  let total_loss : ℚ := 64

  let painting_revenue := num_paintings * (painting_cost * (1 - painting_discount))
  let toy_revenue (num_toys : ℕ) := num_toys * (toy_cost * (1 - toy_discount))
  let total_cost (num_toys : ℕ) := num_paintings * painting_cost + num_toys * toy_cost
  let total_revenue (num_toys : ℕ) := painting_revenue + toy_revenue num_toys

  have h : ∃ (num_toys : ℕ), total_cost num_toys - total_revenue num_toys = total_loss :=
    sorry

  Classical.choose h

/-- The solution to the problem is 8 wooden toys. -/
theorem solution_is_eight : mr_callen_wooden_toys = 8 := by
  sorry

end NUMINAMATH_CALUDE_mr_callen_wooden_toys_solution_is_eight_l3548_354883


namespace NUMINAMATH_CALUDE_book_page_increase_l3548_354843

/-- Represents a book with chapters that increase in page count -/
structure Book where
  total_pages : ℕ
  num_chapters : ℕ
  first_chapter_pages : ℕ
  page_increase : ℕ

/-- Calculates the total pages in a book based on its structure -/
def calculate_total_pages (b : Book) : ℕ :=
  b.first_chapter_pages * b.num_chapters + 
  (b.num_chapters * (b.num_chapters - 1) * b.page_increase) / 2

/-- Theorem stating the page increase for the given book specifications -/
theorem book_page_increase (b : Book) 
  (h1 : b.total_pages = 95)
  (h2 : b.num_chapters = 5)
  (h3 : b.first_chapter_pages = 13)
  (h4 : calculate_total_pages b = b.total_pages) :
  b.page_increase = 3 := by
  sorry

#eval calculate_total_pages { total_pages := 95, num_chapters := 5, first_chapter_pages := 13, page_increase := 3 }

end NUMINAMATH_CALUDE_book_page_increase_l3548_354843


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3548_354865

theorem complex_equation_solution : ∃ (z : ℂ), z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) ∧ z = (1/2 : ℂ) - Complex.I * ((Real.sqrt 3)/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3548_354865


namespace NUMINAMATH_CALUDE_notebook_purchase_l3548_354811

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The cost of the notebook in cents -/
def notebook_cost : ℕ := 130

/-- The number of nickels needed to pay for the notebook -/
def nickels_needed : ℕ := notebook_cost / nickel_value

theorem notebook_purchase :
  nickels_needed = 26 := by sorry

end NUMINAMATH_CALUDE_notebook_purchase_l3548_354811


namespace NUMINAMATH_CALUDE_red_highest_probability_l3548_354803

def num_red : ℕ := 5
def num_yellow : ℕ := 4
def num_white : ℕ := 1
def num_blue : ℕ := 3

def total_balls : ℕ := num_red + num_yellow + num_white + num_blue

theorem red_highest_probability :
  (num_red : ℚ) / total_balls > max ((num_yellow : ℚ) / total_balls)
                                    (max ((num_white : ℚ) / total_balls)
                                         ((num_blue : ℚ) / total_balls)) :=
by sorry

end NUMINAMATH_CALUDE_red_highest_probability_l3548_354803


namespace NUMINAMATH_CALUDE_intersection_point_l3548_354834

def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 4

theorem intersection_point : 
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-12/7, 22/7) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_l3548_354834


namespace NUMINAMATH_CALUDE_fourth_sphere_radius_l3548_354837

/-- Given four spheres where each touches the other three, and three of them have radius R,
    the radius of the fourth sphere is R/3. -/
theorem fourth_sphere_radius (R : ℝ) (R_pos : R > 0) : ℝ :=
  let fourth_radius := R / 3
  fourth_radius

#check fourth_sphere_radius

end NUMINAMATH_CALUDE_fourth_sphere_radius_l3548_354837


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3548_354871

def U : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}

theorem intersection_with_complement : P ∩ (U \ Q) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3548_354871


namespace NUMINAMATH_CALUDE_office_absenteeism_l3548_354852

-- Define the number of persons in the office
variable (p : ℕ) 

-- Define the fraction of absent members
def absent_fraction : ℚ := 1 / 7

-- Define the work increase percentage
def work_increase_percentage : ℚ := 100 * (1 / 7)

-- Theorem statement
theorem office_absenteeism (p : ℕ) (hp : p > 0) : 
  (1 / (1 - absent_fraction) - 1) * 100 = work_increase_percentage := by
  sorry

#check office_absenteeism

end NUMINAMATH_CALUDE_office_absenteeism_l3548_354852


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3548_354890

/-- Given a quadratic equation x^2 + bx + c = 0 whose roots are each three more than
    the roots of 2x^2 - 4x - 8 = 0, prove that c = 11 -/
theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ p q : ℝ, (2 * p^2 - 4 * p - 8 = 0) ∧ 
              (2 * q^2 - 4 * q - 8 = 0) ∧ 
              ((p + 3)^2 + b * (p + 3) + c = 0) ∧ 
              ((q + 3)^2 + b * (q + 3) + c = 0)) →
  c = 11 := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_relation_l3548_354890


namespace NUMINAMATH_CALUDE_marys_income_percentage_l3548_354847

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.12 := by
  sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l3548_354847


namespace NUMINAMATH_CALUDE_outfit_cost_theorem_l3548_354868

/-- The cost of an outfit given the prices of individual items -/
def outfit_cost (pant_price t_shirt_price jacket_price : ℚ) : ℚ :=
  pant_price + 4 * t_shirt_price + jacket_price

/-- The theorem stating the cost of the outfit given the constraints -/
theorem outfit_cost_theorem (pant_price t_shirt_price jacket_price : ℚ) :
  (4 * pant_price + 8 * t_shirt_price + 2 * jacket_price = 2400) →
  (2 * pant_price + 14 * t_shirt_price + 3 * jacket_price = 2400) →
  (3 * pant_price + 6 * t_shirt_price = 1500) →
  outfit_cost pant_price t_shirt_price jacket_price = 860 := by
  sorry

#eval outfit_cost 340 80 200

end NUMINAMATH_CALUDE_outfit_cost_theorem_l3548_354868


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3548_354899

/-- Given the expansion of (1+2x)^10, prove properties about its coefficients -/
theorem binomial_expansion_properties :
  let n : ℕ := 10
  let expansion := fun (k : ℕ) => (n.choose k) * (2^k)
  let sum_first_three := 1 + 2 * n.choose 1 + 4 * n.choose 2
  -- Condition: sum of coefficients of first three terms is 201
  sum_first_three = 201 →
  -- 1. The binomial coefficient is largest for the 6th term
  (∀ k, k ≠ 5 → n.choose 5 ≥ n.choose k) ∧
  -- 2. The coefficient is largest for the 8th term
  (∀ k, k ≠ 7 → expansion 7 ≥ expansion k) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3548_354899


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_300_l3548_354806

theorem distinct_prime_factors_of_300 : Nat.card (Nat.factors 300).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_300_l3548_354806


namespace NUMINAMATH_CALUDE_max_value_d_l3548_354858

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_d_l3548_354858


namespace NUMINAMATH_CALUDE_batsman_average_is_60_l3548_354891

/-- Represents a batsman's performance statistics -/
structure BatsmanStats where
  total_innings : ℕ
  highest_score : ℕ
  score_difference : ℕ
  avg_excluding_extremes : ℕ

/-- Calculates the overall batting average -/
def overall_average (stats : BatsmanStats) : ℚ :=
  let lowest_score := stats.highest_score - stats.score_difference
  let total_runs := (stats.total_innings - 2) * stats.avg_excluding_extremes + stats.highest_score + lowest_score
  total_runs / stats.total_innings

/-- Theorem stating the overall batting average is 60 runs given the specific conditions -/
theorem batsman_average_is_60 (stats : BatsmanStats) 
  (h_innings : stats.total_innings = 46)
  (h_highest : stats.highest_score = 199)
  (h_diff : stats.score_difference = 190)
  (h_avg : stats.avg_excluding_extremes = 58) :
  overall_average stats = 60 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_is_60_l3548_354891


namespace NUMINAMATH_CALUDE_beth_crayon_packs_l3548_354844

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons not in packs -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := 46

/-- The number of packs of crayons Beth has -/
def num_packs : ℕ := (total_crayons - extra_crayons) / crayons_per_pack

theorem beth_crayon_packs :
  num_packs = 4 :=
sorry

end NUMINAMATH_CALUDE_beth_crayon_packs_l3548_354844


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3548_354886

/-- Given an ellipse with center at the origin, one focus at (0, -2), and one endpoint
    of a semi-major axis at (0, 5), its semi-minor axis has length √21. -/
theorem ellipse_semi_minor_axis (c a b : ℝ) : 
  c = 2 →  -- distance from center to focus
  a = 5 →  -- length of semi-major axis
  b^2 = a^2 - c^2 →  -- relationship between a, b, and c in an ellipse
  b = Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3548_354886


namespace NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l3548_354874

/-- 
Represents a line in polar coordinates passing through a given point and parallel to the polar axis.
-/
def polar_line_parallel_to_axis (r : ℝ) (θ : ℝ) : Prop :=
  ∀ ρ θ', ρ * Real.sin θ' = r * Real.sin θ

theorem polar_line_through_point_parallel_to_axis 
  (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π/4) :
  polar_line_parallel_to_axis r θ ↔ 
  ∀ ρ θ', ρ * Real.sin θ' = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l3548_354874


namespace NUMINAMATH_CALUDE_min_omega_two_max_sine_l3548_354802

theorem min_omega_two_max_sine (ω : Real) : ω > 0 → (∃ x₁ x₂ : Real, 
  0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₁)) ∧
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₂))) → 
  ω ≥ 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_min_omega_two_max_sine_l3548_354802


namespace NUMINAMATH_CALUDE_odd_perfect_number_l3548_354828

/-- Sum of positive divisors of n -/
def sigma (n : ℕ) : ℕ := sorry

/-- A number is perfect if σ(n) = 2n -/
def isPerfect (n : ℕ) : Prop := sigma n = 2 * n

theorem odd_perfect_number (n : ℕ) (h : n > 0) (h_sigma : (sigma n : ℚ) / n = 5 / 3) :
  isPerfect (5 * n) ∧ Odd (5 * n) := by sorry

end NUMINAMATH_CALUDE_odd_perfect_number_l3548_354828


namespace NUMINAMATH_CALUDE_separation_of_homologous_chromosomes_unique_l3548_354815

-- Define the cell division processes
inductive CellDivisionProcess
  | ChromosomeReplication
  | SeparationOfHomologousChromosomes
  | SeparationOfChromatids
  | Cytokinesis

-- Define the types of cell division
inductive CellDivision
  | Mitosis
  | Meiosis

-- Define a function that determines if a process occurs in a given cell division
def occursIn (process : CellDivisionProcess) (division : CellDivision) : Prop :=
  match division with
  | CellDivision.Mitosis =>
    process ≠ CellDivisionProcess.SeparationOfHomologousChromosomes
  | CellDivision.Meiosis => True

-- Theorem statement
theorem separation_of_homologous_chromosomes_unique :
  ∀ (process : CellDivisionProcess),
    (occursIn process CellDivision.Meiosis ∧ ¬occursIn process CellDivision.Mitosis) →
    process = CellDivisionProcess.SeparationOfHomologousChromosomes :=
by sorry

end NUMINAMATH_CALUDE_separation_of_homologous_chromosomes_unique_l3548_354815


namespace NUMINAMATH_CALUDE_function_value_at_a_plus_one_l3548_354824

theorem function_value_at_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_a_plus_one_l3548_354824


namespace NUMINAMATH_CALUDE_range_of_H_l3548_354805

-- Define the function H
def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H : 
  Set.range H = {-1, 5} := by sorry

end NUMINAMATH_CALUDE_range_of_H_l3548_354805


namespace NUMINAMATH_CALUDE_hcd_8100_270_minus_8_l3548_354884

theorem hcd_8100_270_minus_8 : Nat.gcd 8100 270 - 8 = 262 := by
  sorry

end NUMINAMATH_CALUDE_hcd_8100_270_minus_8_l3548_354884


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3548_354898

/-- Proves that given a journey of 12 hours covering 560 km, where the first half of the distance 
is traveled at 35 kmph, the speed for the second half of the journey is 70 kmph. -/
theorem journey_speed_calculation (total_time : ℝ) (total_distance : ℝ) (first_half_speed : ℝ) :
  total_time = 12 →
  total_distance = 560 →
  first_half_speed = 35 →
  (total_distance / 2) / first_half_speed + (total_distance / 2) / ((total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed)) = total_time →
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 70 := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_journey_speed_calculation_l3548_354898


namespace NUMINAMATH_CALUDE_root_property_l3548_354880

theorem root_property (a : ℝ) : 3 * a^2 - 5 * a - 2 = 0 → 6 * a^2 - 10 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l3548_354880


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l3548_354888

theorem quadratic_roots_max_value (s p r₁ : ℝ) (h1 : r₁ ≠ 0) : 
  (r₁ + (-r₁) = 0) → 
  (r₁ * (-r₁) = p) → 
  (∀ (n : ℕ), n ≤ 2005 → r₁^(2*n) + (-r₁)^(2*n) = 2 * r₁^(2*n)) →
  (∃ (x : ℝ), x^2 - s*x + p = 0) →
  (∀ (y : ℝ), (1 / r₁^2006) + (1 / (-r₁)^2006) ≤ y) →
  y = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l3548_354888


namespace NUMINAMATH_CALUDE_syrup_box_cost_l3548_354862

/-- Represents the cost of syrup boxes for a convenience store -/
def SyrupCost (total_soda : ℕ) (soda_per_box : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (total_soda / soda_per_box)

/-- Theorem: The cost per box of syrup is $40 -/
theorem syrup_box_cost :
  SyrupCost 180 30 240 = 40 := by
  sorry

end NUMINAMATH_CALUDE_syrup_box_cost_l3548_354862


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3548_354833

/-- The value of a for which the line x - y - 1 = 0 is tangent to the parabola y = ax² --/
theorem line_tangent_to_parabola :
  ∃! (a : ℝ), ∀ (x y : ℝ),
    (x - y - 1 = 0 ∧ y = a * x^2) →
    (∃! p : ℝ × ℝ, p.1 - p.2 - 1 = 0 ∧ p.2 = a * p.1^2) ∧
    a = 1/4 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3548_354833


namespace NUMINAMATH_CALUDE_wire_cutting_l3548_354804

theorem wire_cutting (x : ℝ) :
  let total_length := Real.sqrt 600 + 12 * x
  let A := (Real.sqrt 600 + 15 * x - 9 * x^2) / 2
  let B := (Real.sqrt 600 + 9 * x - 9 * x^2) / 2
  let C := 9 * x^2
  (A = B + 3 * x) ∧
  (C = (A - B)^2) ∧
  (A + B + C = total_length) :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_l3548_354804


namespace NUMINAMATH_CALUDE_rectangle_diagonal_parts_l3548_354894

theorem rectangle_diagonal_parts (m n : ℕ) (hm : m = 1000) (hn : n = 1979) :
  m + n - Nat.gcd m n = 2978 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_parts_l3548_354894


namespace NUMINAMATH_CALUDE_min_value_problem_l3548_354846

theorem min_value_problem (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0)
  (h1 : 2 * x + y = 1)
  (h2 : ∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 1 → a / x' + 1 / y' ≥ 9)
  (h3 : ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 2 * x' + y' = 1 ∧ a / x' + 1 / y' = 9) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3548_354846


namespace NUMINAMATH_CALUDE_cupcakes_problem_l3548_354856

/-- Calculates the number of cupcakes per package given the initial number of cupcakes,
    the number of cupcakes eaten, and the number of packages. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Proves that given 50 initial cupcakes, 5 cupcakes eaten, and 9 equal packages,
    the number of cupcakes in each package is 5. -/
theorem cupcakes_problem :
  cupcakes_per_package 50 5 9 = 5 := by
  sorry

#eval cupcakes_per_package 50 5 9

end NUMINAMATH_CALUDE_cupcakes_problem_l3548_354856


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_is_13_l3548_354876

/-- Conference attendee information -/
structure ConferenceData where
  total : Nat
  writers : Nat
  editors : Nat
  both : Nat
  neither : Nat
  editors_gt_38 : editors > 38
  neither_eq_2both : neither = 2 * both
  total_sum : total = writers + editors - both + neither

/-- The maximum number of people who can be both writers and editors -/
def max_both_writers_and_editors (data : ConferenceData) : Nat :=
  13

/-- Theorem stating that 13 is the maximum number of people who can be both writers and editors -/
theorem max_both_writers_and_editors_is_13 (data : ConferenceData) 
  (h : data.total = 110 ∧ data.writers = 45) :
  max_both_writers_and_editors data = 13 := by
  sorry

#check max_both_writers_and_editors_is_13

end NUMINAMATH_CALUDE_max_both_writers_and_editors_is_13_l3548_354876


namespace NUMINAMATH_CALUDE_tetrahedron_to_polyhedron_ratios_l3548_354845

/-- Regular tetrahedron -/
structure RegularTetrahedron where
  surface_area : ℝ
  volume : ℝ

/-- Polyhedron G formed by removing four smaller tetrahedrons from a regular tetrahedron -/
structure PolyhedronG where
  surface_area : ℝ
  volume : ℝ

/-- Given a regular tetrahedron and the polyhedron G formed from it, 
    prove that the surface area ratio is 9/7 and the volume ratio is 27/23 -/
theorem tetrahedron_to_polyhedron_ratios 
  (p : RegularTetrahedron) 
  (g : PolyhedronG) 
  (h : g = PolyhedronG.mk ((28/36) * p.surface_area) ((23/27) * p.volume)) : 
  p.surface_area / g.surface_area = 9/7 ∧ p.volume / g.volume = 27/23 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_to_polyhedron_ratios_l3548_354845


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l3548_354867

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 1.02

/-- Converts speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * seconds_per_hour

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3672 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l3548_354867


namespace NUMINAMATH_CALUDE_trebled_result_l3548_354826

theorem trebled_result (initial_number : ℕ) : 
  initial_number = 17 → 
  3 * (2 * initial_number + 5) = 117 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_l3548_354826


namespace NUMINAMATH_CALUDE_find_divisor_l3548_354819

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 132 →
  quotient = 8 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3548_354819


namespace NUMINAMATH_CALUDE_cost_difference_is_six_l3548_354809

/-- Represents the cost and consumption of a pizza -/
structure PizzaCost where
  totalSlices : ℕ
  plainCost : ℚ
  toppingCost : ℚ
  daveToppedSlices : ℕ
  davePlainSlices : ℕ

/-- Calculates the difference in cost between Dave's and Doug's portions -/
def costDifference (p : PizzaCost) : ℚ :=
  let totalCost := p.plainCost + p.toppingCost
  let costPerSlice := totalCost / p.totalSlices
  let daveCost := costPerSlice * (p.daveToppedSlices + p.davePlainSlices)
  let dougSlices := p.totalSlices - p.daveToppedSlices - p.davePlainSlices
  let dougCost := (p.plainCost / p.totalSlices) * dougSlices
  daveCost - dougCost

/-- Theorem stating that the cost difference is $6 -/
theorem cost_difference_is_six (p : PizzaCost) 
  (h1 : p.totalSlices = 12)
  (h2 : p.plainCost = 12)
  (h3 : p.toppingCost = 3)
  (h4 : p.daveToppedSlices = 6)
  (h5 : p.davePlainSlices = 2) :
  costDifference p = 6 := by
  sorry

#eval costDifference { totalSlices := 12, plainCost := 12, toppingCost := 3, daveToppedSlices := 6, davePlainSlices := 2 }

end NUMINAMATH_CALUDE_cost_difference_is_six_l3548_354809


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l3548_354835

theorem andreas_living_room_area :
  ∀ (room_area : ℝ),
  (0.60 * room_area = 4 * 9) →
  room_area = 60 := by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l3548_354835


namespace NUMINAMATH_CALUDE_total_weight_calculation_l3548_354851

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 1280

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- The total weight of the compound in grams -/
def total_weight : ℝ := molecular_weight * number_of_moles

theorem total_weight_calculation :
  total_weight = 10240 := by sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l3548_354851


namespace NUMINAMATH_CALUDE_binomial_constant_term_l3548_354887

theorem binomial_constant_term (n : ℕ) : 
  (∃ r : ℕ, r ≤ n ∧ 4*n = 5*r) ↔ n = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_constant_term_l3548_354887


namespace NUMINAMATH_CALUDE_y_value_proof_l3548_354820

/-- Proves that y = 8 on an equally spaced number line from 0 to 32 with 8 steps,
    where y is 2 steps before the midpoint -/
theorem y_value_proof (total_distance : ℝ) (num_steps : ℕ) (y : ℝ) :
  total_distance = 32 →
  num_steps = 8 →
  y = (total_distance / 2) - 2 * (total_distance / num_steps) →
  y = 8 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l3548_354820


namespace NUMINAMATH_CALUDE_part_one_part_two_l3548_354870

-- Define the function f
def f (x m : ℝ) : ℝ := 3 * x^2 + m * (m - 6) * x + 5

-- Theorem for part 1
theorem part_one (m : ℝ) : f 1 m > 0 ↔ m > 4 ∨ m < 2 := by sorry

-- Theorem for part 2
theorem part_two (m n : ℝ) : 
  (∀ x, f x m < n ↔ -1 < x ∧ x < 4) → m = 3 ∧ n = 17 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3548_354870


namespace NUMINAMATH_CALUDE_at_most_one_root_l3548_354821

theorem at_most_one_root {f : ℝ → ℝ} (h : ∀ a b, a < b → f a < f b) :
  ∃! x, f x = 0 ∨ ∀ x, f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_root_l3548_354821


namespace NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l3548_354881

/-- The cost of football equipment relative to shorts -/
def FootballEquipmentCost (x : ℝ) : Prop :=
  -- x is the cost of shorts
  x > 0 ∧
  -- Shorts + T-shirt cost twice as much as shorts alone
  ∃ t, x + t = 2 * x ∧
  -- Shorts + boots cost five times as much as shorts alone
  ∃ b, x + b = 5 * x ∧
  -- Shorts + shin guards cost three times as much as shorts alone
  ∃ s, x + s = 3 * x

/-- The total cost of all items is 8 times the cost of shorts -/
theorem total_cost_is_eight_times_shorts (x : ℝ) 
  (h : FootballEquipmentCost x) : 
  ∃ total, total = 8 * x := by
  sorry


end NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l3548_354881


namespace NUMINAMATH_CALUDE_equal_area_perimeter_rectangle_dimensions_l3548_354889

/-- A rectangle with integer side lengths where the area equals the perimeter. -/
structure EqualAreaPerimeterRectangle where
  width : ℕ
  length : ℕ
  area_eq_perimeter : width * length = 2 * (width + length)

/-- The possible dimensions of a rectangle with integer side lengths where the area equals the perimeter. -/
def valid_dimensions : Set (ℕ × ℕ) :=
  {(4, 4), (3, 6), (6, 3)}

/-- Theorem stating that the only valid dimensions for a rectangle with integer side lengths
    where the area equals the perimeter are 4x4, 3x6, or 6x3. -/
theorem equal_area_perimeter_rectangle_dimensions (r : EqualAreaPerimeterRectangle) :
  (r.width, r.length) ∈ valid_dimensions := by
  sorry

#check equal_area_perimeter_rectangle_dimensions

end NUMINAMATH_CALUDE_equal_area_perimeter_rectangle_dimensions_l3548_354889


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l3548_354869

theorem volleyball_team_starters (n m k : ℕ) (h1 : n = 14) (h2 : m = 6) (h3 : k = 3) :
  Nat.choose (n - k) (m - k) = 165 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l3548_354869


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l3548_354822

/-- A regular tetrahedron with one vertex on the axis of a cylinder and the other three vertices on the lateral surface of the cylinder. -/
structure TetrahedronInCylinder where
  R : ℝ  -- Radius of the cylinder's base
  edge_length : ℝ  -- Edge length of the tetrahedron

/-- The edge length of the tetrahedron is either R√3 or (R√11)/3. -/
theorem tetrahedron_edge_length (t : TetrahedronInCylinder) :
  t.edge_length = t.R * Real.sqrt 3 ∨ t.edge_length = t.R * Real.sqrt 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l3548_354822


namespace NUMINAMATH_CALUDE_binomial_expansion_difference_l3548_354831

theorem binomial_expansion_difference : 
  3^7 + (Nat.choose 7 2) * 3^5 + (Nat.choose 7 4) * 3^3 + (Nat.choose 7 6) * 3 -
  ((Nat.choose 7 1) * 3^6 + (Nat.choose 7 3) * 3^4 + (Nat.choose 7 5) * 3^2 + 1) = 128 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_difference_l3548_354831


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l3548_354860

theorem trig_fraction_equality (α : ℝ) (h : (1 + Real.sin α) / Real.cos α = -1/2) :
  Real.cos α / (Real.sin α - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l3548_354860


namespace NUMINAMATH_CALUDE_negative_correlation_from_negative_coefficient_given_equation_negative_correlation_l3548_354801

/-- Represents a linear regression equation -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Defines negative correlation between x and y -/
def negatively_correlated (eq : LinearRegression) : Prop :=
  eq.b < 0

/-- Theorem: If the coefficient of x in a linear regression equation is negative,
    then x and y are negatively correlated -/
theorem negative_correlation_from_negative_coefficient (eq : LinearRegression) :
  eq.b < 0 → negatively_correlated eq :=
by
  sorry

/-- The given empirical regression equation -/
def given_equation : LinearRegression :=
  { a := 2, b := -1 }

/-- Theorem: The given equation represents a negative correlation between x and y -/
theorem given_equation_negative_correlation :
  negatively_correlated given_equation :=
by
  sorry

end NUMINAMATH_CALUDE_negative_correlation_from_negative_coefficient_given_equation_negative_correlation_l3548_354801


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l3548_354859

theorem tracy_art_fair_sales : 
  let total_customers : ℕ := 20
  let group1_customers : ℕ := 4
  let group1_paintings_per_customer : ℕ := 2
  let group2_customers : ℕ := 12
  let group2_paintings_per_customer : ℕ := 1
  let group3_customers : ℕ := 4
  let group3_paintings_per_customer : ℕ := 4
  let total_paintings_sold := 
    group1_customers * group1_paintings_per_customer +
    group2_customers * group2_paintings_per_customer +
    group3_customers * group3_paintings_per_customer
  total_customers = group1_customers + group2_customers + group3_customers →
  total_paintings_sold = 36 := by
sorry


end NUMINAMATH_CALUDE_tracy_art_fair_sales_l3548_354859


namespace NUMINAMATH_CALUDE_upper_side_length_l3548_354855

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  lower_side : ℝ
  upper_side : ℝ
  height : ℝ
  area : ℝ
  upper_shorter : upper_side = lower_side - 6
  height_value : height = 8
  area_value : area = 72
  area_formula : area = (lower_side + upper_side) / 2 * height

/-- Theorem: The length of the upper side of the trapezoid is 6 cm -/
theorem upper_side_length (t : Trapezoid) : t.upper_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_upper_side_length_l3548_354855


namespace NUMINAMATH_CALUDE_cricket_bats_profit_percentage_l3548_354812

/-- Calculate the overall profit percentage for three cricket bats --/
theorem cricket_bats_profit_percentage
  (selling_price_A selling_price_B selling_price_C : ℝ)
  (profit_A profit_B profit_C : ℝ)
  (h1 : selling_price_A = 900)
  (h2 : selling_price_B = 1200)
  (h3 : selling_price_C = 1500)
  (h4 : profit_A = 300)
  (h5 : profit_B = 400)
  (h6 : profit_C = 500) :
  let total_cost_price := (selling_price_A - profit_A) + (selling_price_B - profit_B) + (selling_price_C - profit_C)
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let total_profit := total_selling_price - total_cost_price
  (total_profit / total_cost_price) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bats_profit_percentage_l3548_354812


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3548_354832

theorem consecutive_integers_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) = 504 → n + (n + 1) + (n + 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3548_354832


namespace NUMINAMATH_CALUDE_product_of_roots_l3548_354885

theorem product_of_roots (x z : ℝ) (h1 : x - z = 6) (h2 : x^3 - z^3 = 108) : x * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3548_354885


namespace NUMINAMATH_CALUDE_class_size_l3548_354813

theorem class_size (top_rank bottom_rank : ℕ) (h1 : top_rank = 17) (h2 : bottom_rank = 15) :
  top_rank + bottom_rank - 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3548_354813


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_4_and_5_l3548_354816

theorem largest_three_digit_divisible_by_4_and_5 : ∀ n : ℕ,
  n ≤ 999 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 → n ≤ 980 := by
  sorry

#check largest_three_digit_divisible_by_4_and_5

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_4_and_5_l3548_354816


namespace NUMINAMATH_CALUDE_solution_set_inequality_range_of_a_l3548_354807

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * |x + 1|
def g (x : ℝ) : ℝ := 4 + |2*x - 1|

-- Part 1: Solution set of f(x) + 2 ≤ g(x)
theorem solution_set_inequality (x : ℝ) :
  f x + 2 ≤ g x ↔ x ∈ Set.Iic (1/4 : ℝ) :=
sorry

-- Part 2: Range of a for which f(x) + g(x) ≥ 2a^2 - 13a for all real x
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x + g x ≥ 2*a^2 - 13*a) ↔ a ∈ Set.Icc (-1/2 : ℝ) 7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_range_of_a_l3548_354807


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3548_354829

/-- Given a point (a,b) outside the circle x^2 + y^2 = r^2, 
    the line ax + by = r^2 intersects the circle and does not pass through the center. -/
theorem line_circle_intersection (a b r : ℝ) (hr : r > 0) (h_outside : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_l3548_354829


namespace NUMINAMATH_CALUDE_biff_break_even_time_biff_break_even_time_is_three_l3548_354872

/-- Calculates the break-even time for Biff's bus trip -/
theorem biff_break_even_time 
  (ticket_cost : ℝ) 
  (snacks_cost : ℝ) 
  (headphones_cost : ℝ) 
  (work_rate : ℝ) 
  (wifi_cost : ℝ) : ℝ :=
  let total_cost := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := work_rate - wifi_cost
  total_cost / net_hourly_rate

/-- Proves that Biff's break-even time is 3 hours given the specific costs and rates -/
theorem biff_break_even_time_is_three :
  biff_break_even_time 11 3 16 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_biff_break_even_time_biff_break_even_time_is_three_l3548_354872


namespace NUMINAMATH_CALUDE_quarters_in_school_year_l3548_354875

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The number of students in the art club -/
def students : ℕ := 15

/-- The number of artworks each student makes per quarter -/
def artworks_per_student_per_quarter : ℕ := 2

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- Theorem stating that the number of quarters in a school year is 4 -/
theorem quarters_in_school_year : 
  quarters_per_year * 2 * students * artworks_per_student_per_quarter = total_artworks :=
by sorry

end NUMINAMATH_CALUDE_quarters_in_school_year_l3548_354875


namespace NUMINAMATH_CALUDE_range_of_f_l3548_354810

def f (x : ℤ) : ℤ := x^2 - 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 4}

theorem range_of_f : {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3, 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3548_354810


namespace NUMINAMATH_CALUDE_existence_of_a_values_l3548_354866

theorem existence_of_a_values (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_values_l3548_354866


namespace NUMINAMATH_CALUDE_equal_weight_partition_l3548_354893

theorem equal_weight_partition : ∃ (A B C : Finset Nat), 
  (A ∪ B ∪ C = Finset.range 556 \ {0}) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (A.sum id = B.sum id) ∧ (B.sum id = C.sum id) := by
  sorry

#check equal_weight_partition

end NUMINAMATH_CALUDE_equal_weight_partition_l3548_354893


namespace NUMINAMATH_CALUDE_one_third_displayed_l3548_354825

/-- Represents an art gallery with paintings and sculptures -/
structure ArtGallery where
  total_pieces : ℕ
  displayed_pieces : ℕ
  displayed_sculptures : ℕ
  not_displayed_paintings : ℕ
  not_displayed_sculptures : ℕ

/-- Conditions for the art gallery problem -/
def gallery_conditions (g : ArtGallery) : Prop :=
  g.total_pieces = 900 ∧
  g.not_displayed_sculptures = 400 ∧
  g.displayed_sculptures = g.displayed_pieces / 6 ∧
  g.not_displayed_paintings = (g.total_pieces - g.displayed_pieces) / 3

/-- Theorem stating that 1/3 of the pieces are displayed -/
theorem one_third_displayed (g : ArtGallery) 
  (h : gallery_conditions g) : 
  g.displayed_pieces = g.total_pieces / 3 := by
  sorry

#check one_third_displayed

end NUMINAMATH_CALUDE_one_third_displayed_l3548_354825


namespace NUMINAMATH_CALUDE_video_game_points_sum_l3548_354873

theorem video_game_points_sum : 
  let paul_points : ℕ := 3103
  let cousin_points : ℕ := 2713
  paul_points + cousin_points = 5816 :=
by sorry

end NUMINAMATH_CALUDE_video_game_points_sum_l3548_354873


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3548_354823

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), 
  s > 0 → 
  6 * s^2 = 150 → 
  s^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3548_354823


namespace NUMINAMATH_CALUDE_cistern_emptying_l3548_354839

/-- Represents the rate at which a pipe can empty a cistern -/
structure EmptyingRate where
  fraction : ℚ
  time : ℕ

/-- Represents the operation of pipes emptying a cistern -/
def empty_cistern (pipe1 : EmptyingRate) (pipe2 : EmptyingRate) (time1 : ℕ) (time2 : ℕ) : ℚ :=
  sorry

theorem cistern_emptying :
  let pipe1 : EmptyingRate := ⟨3/4, 12⟩
  let pipe2 : EmptyingRate := ⟨1/2, 15⟩
  empty_cistern pipe1 pipe2 4 10 = 7/12 :=
by sorry

end NUMINAMATH_CALUDE_cistern_emptying_l3548_354839


namespace NUMINAMATH_CALUDE_jakes_lawn_mowing_time_l3548_354830

/-- Jake's lawn mowing problem -/
theorem jakes_lawn_mowing_time
  (desired_hourly_rate : ℝ)
  (flower_planting_time : ℝ)
  (flower_planting_charge : ℝ)
  (lawn_mowing_pay : ℝ)
  (h1 : desired_hourly_rate = 20)
  (h2 : flower_planting_time = 2)
  (h3 : flower_planting_charge = 45)
  (h4 : lawn_mowing_pay = 15) :
  (flower_planting_charge + lawn_mowing_pay) / desired_hourly_rate - flower_planting_time = 1 := by
  sorry

#check jakes_lawn_mowing_time

end NUMINAMATH_CALUDE_jakes_lawn_mowing_time_l3548_354830


namespace NUMINAMATH_CALUDE_expression_evaluation_l3548_354808

theorem expression_evaluation (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d + Nat.factorial (d-1))^2 = 39204 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3548_354808


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3548_354838

/-- Given an income, dividend rate, and investment amount, calculate the price of a stock. -/
theorem stock_price_calculation (income investment : ℚ) (dividend_rate : ℚ) : 
  income = 650 →
  dividend_rate = 1/10 →
  investment = 6240 →
  (investment / (income / dividend_rate)) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3548_354838


namespace NUMINAMATH_CALUDE_B_completes_work_in_8_days_l3548_354897

/-- The number of days B takes to complete the work alone -/
def B : ℕ := 8

/-- The rate at which A completes the work -/
def rate_A : ℚ := 1 / 20

/-- The rate at which B completes the work -/
def rate_B : ℚ := 1 / B

/-- The amount of work completed by A and B together in 3 days -/
def work_together : ℚ := 3 * (rate_A + rate_B)

/-- The amount of work completed by B alone in 3 days -/
def work_B_alone : ℚ := 3 * rate_B

theorem B_completes_work_in_8_days :
  work_together + work_B_alone = 1 ∧ B = 8 := by
  sorry

end NUMINAMATH_CALUDE_B_completes_work_in_8_days_l3548_354897


namespace NUMINAMATH_CALUDE_expression_evaluation_l3548_354800

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3548_354800


namespace NUMINAMATH_CALUDE_expression_evaluation_l3548_354853

theorem expression_evaluation :
  let a : ℤ := 1
  let b : ℤ := 10
  let c : ℤ := 100
  let d : ℤ := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3548_354853


namespace NUMINAMATH_CALUDE_poker_night_cards_l3548_354818

theorem poker_night_cards (half_decks full_decks thrown_away remaining : ℕ) : 
  half_decks = 3 →
  full_decks = 3 →
  thrown_away = 34 →
  remaining = 200 →
  ∃ (cards_per_full_deck cards_per_half_deck : ℕ),
    cards_per_half_deck = cards_per_full_deck / 2 ∧
    remaining + thrown_away = half_decks * cards_per_half_deck + full_decks * cards_per_full_deck ∧
    cards_per_full_deck = 52 :=
by sorry

end NUMINAMATH_CALUDE_poker_night_cards_l3548_354818


namespace NUMINAMATH_CALUDE_smallest_y_coordinate_on_ellipse_l3548_354863

/-- The ellipse is defined by the equation (x^2/49) + ((y-3)^2/25) = 1 -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2/49 + (y-3)^2/25 = 1

/-- The smallest y-coordinate of any point on the ellipse -/
def smallest_y_coordinate : ℝ := -2

/-- Theorem stating that the smallest y-coordinate of any point on the ellipse is -2 -/
theorem smallest_y_coordinate_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y → y ≥ smallest_y_coordinate :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_coordinate_on_ellipse_l3548_354863


namespace NUMINAMATH_CALUDE_father_daughter_ages_l3548_354895

theorem father_daughter_ages (father daughter : ℕ) : 
  father = 4 * daughter ∧ 
  father + 20 = 2 * (daughter + 20) → 
  father = 40 ∧ daughter = 10 := by
sorry

end NUMINAMATH_CALUDE_father_daughter_ages_l3548_354895
