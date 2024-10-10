import Mathlib

namespace base_4_to_16_digits_l213_21321

theorem base_4_to_16_digits : ∀ n : ℕ,
  (4^4 ≤ n) ∧ (n < 4^5) →
  (16^2 ≤ n) ∧ (n < 16^3) :=
by sorry

end base_4_to_16_digits_l213_21321


namespace trajectory_equation_l213_21314

/-- The trajectory of point M(x, y) with distance ratio 2 from F(4,0) and line x = 3 -/
theorem trajectory_equation (x y : ℝ) : 
  (((x - 4)^2 + y^2) / ((x - 3)^2)) = 4 → 
  3 * x^2 - y^2 - 16 * x + 20 = 0 := by
sorry

end trajectory_equation_l213_21314


namespace prime_sequence_l213_21368

def is_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem prime_sequence (a p : ℕ → ℕ) 
  (h_inc : is_increasing a)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_div : ∀ n, (p n) ∣ (a n))
  (h_diff : ∀ n k, a n - a k = p n - p k) :
  ∀ n, a n = p n :=
sorry

end prime_sequence_l213_21368


namespace min_intersection_size_l213_21349

theorem min_intersection_size (total blue_eyes backpack : ℕ) 
  (h_total : total = 35)
  (h_blue : blue_eyes = 18)
  (h_backpack : backpack = 24) :
  blue_eyes + backpack - total ≤ (blue_eyes ⊓ backpack) :=
by sorry

end min_intersection_size_l213_21349


namespace not_both_perfect_squares_l213_21311

theorem not_both_perfect_squares (x y z t : ℕ+) 
  (h1 : x.val * y.val - z.val * t.val = x.val + y.val)
  (h2 : x.val + y.val = z.val + t.val) :
  ¬(∃ (a c : ℕ), x.val * y.val = a^2 ∧ z.val * t.val = c^2) :=
by sorry

end not_both_perfect_squares_l213_21311


namespace league_games_l213_21335

theorem league_games (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 30) (h2 : k = 2) (h3 : m = 6) :
  (n.choose k) * m = 2610 := by
  sorry

end league_games_l213_21335


namespace largest_angle_in_triangle_l213_21398

theorem largest_angle_in_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines holds
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given conditions
  Real.cos A = 3/4 →
  C = 2 * A →
  -- Conclusion
  C > A ∧ C > B :=
by sorry

end largest_angle_in_triangle_l213_21398


namespace sum_of_quotient_dividend_divisor_l213_21318

theorem sum_of_quotient_dividend_divisor (n d : ℕ) (h1 : n = 45) (h2 : d = 3) :
  n / d + n + d = 63 := by
  sorry

end sum_of_quotient_dividend_divisor_l213_21318


namespace scientific_notation_of_105_9_billion_l213_21364

theorem scientific_notation_of_105_9_billion : 
  (105.9 : ℝ) * 1000000000 = 1.059 * (10 : ℝ)^10 := by sorry

end scientific_notation_of_105_9_billion_l213_21364


namespace distinct_digit_count_is_5040_l213_21380

/-- The number of four-digit integers with distinct digits, including those starting with 0 -/
def distinctDigitCount : ℕ := 10 * 9 * 8 * 7

/-- Theorem stating that the count of four-digit integers with distinct digits is 5040 -/
theorem distinct_digit_count_is_5040 : distinctDigitCount = 5040 := by
  sorry

end distinct_digit_count_is_5040_l213_21380


namespace binomial_coefficient_23_5_l213_21369

theorem binomial_coefficient_23_5 (h1 : Nat.choose 21 3 = 1330)
                                  (h2 : Nat.choose 21 4 = 5985)
                                  (h3 : Nat.choose 21 5 = 20349) :
  Nat.choose 23 5 = 33649 := by
  sorry

end binomial_coefficient_23_5_l213_21369


namespace triangle_angle_measure_l213_21308

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < π →
  (a - b + c) * (a + b + c) = 3 * a * c →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  B = π / 3 := by
sorry

end triangle_angle_measure_l213_21308


namespace sphere_volume_surface_area_ratio_l213_21352

theorem sphere_volume_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 := by
  sorry

end sphere_volume_surface_area_ratio_l213_21352


namespace greatest_three_digit_number_l213_21387

theorem greatest_three_digit_number : ∃ (n : ℕ), 
  n = 982 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∃ (k : ℕ), n = 7 * k + 2 ∧ 
  ∃ (m : ℕ), n = 6 * m + 4 ∧ 
  ∀ (x : ℕ), (100 ≤ x ∧ x ≤ 999 ∧ ∃ (a : ℕ), x = 7 * a + 2 ∧ ∃ (b : ℕ), x = 6 * b + 4) → x ≤ n :=
by
  sorry

end greatest_three_digit_number_l213_21387


namespace sqrt_equation_solution_l213_21362

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end sqrt_equation_solution_l213_21362


namespace students_registered_correct_registration_l213_21351

theorem students_registered (students_yesterday : ℕ) (absent_today : ℕ) : ℕ :=
  let twice_yesterday := 2 * students_yesterday
  let ten_percent := twice_yesterday / 10
  let attending_today := twice_yesterday - ten_percent
  let total_registered := attending_today + absent_today
  total_registered

theorem correct_registration : students_registered 70 30 = 156 := by
  sorry

end students_registered_correct_registration_l213_21351


namespace friendly_numbers_solution_l213_21356

/-- Two rational numbers are friendly if their sum is 66 -/
def friendly (m n : ℚ) : Prop := m + n = 66

/-- Given that 7x and -18 are friendly numbers, prove that x = 12 -/
theorem friendly_numbers_solution : 
  ∀ x : ℚ, friendly (7 * x) (-18) → x = 12 := by
  sorry

end friendly_numbers_solution_l213_21356


namespace total_gratuity_is_23_02_l213_21392

-- Define the structure for menu items
structure MenuItem where
  name : String
  basePrice : Float
  taxRate : Float

-- Define the menu items
def nyStriploin : MenuItem := ⟨"NY Striploin", 80, 0.10⟩
def wineGlass : MenuItem := ⟨"Glass of wine", 10, 0.15⟩
def dessert : MenuItem := ⟨"Dessert", 12, 0.05⟩
def waterBottle : MenuItem := ⟨"Bottle of water", 3, 0⟩

-- Define the gratuity rate
def gratuityRate : Float := 0.20

-- Function to calculate the total price with tax for an item
def totalPriceWithTax (item : MenuItem) : Float :=
  item.basePrice * (1 + item.taxRate)

-- Function to calculate the gratuity for an item
def calculateGratuity (item : MenuItem) : Float :=
  totalPriceWithTax item * gratuityRate

-- Theorem stating that the total gratuity is $23.02
theorem total_gratuity_is_23_02 :
  calculateGratuity nyStriploin +
  calculateGratuity wineGlass +
  calculateGratuity dessert +
  calculateGratuity waterBottle = 23.02 := by
  sorry -- Proof is omitted as per instructions

end total_gratuity_is_23_02_l213_21392


namespace sequence_21st_term_l213_21327

theorem sequence_21st_term (a : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = (4 * a n + 3) / 4) →
  a 1 = 1 →
  a 21 = 16 := by
sorry

end sequence_21st_term_l213_21327


namespace homogeneous_polynomial_terms_l213_21338

/-- The number of distinct terms in a homogeneous polynomial -/
def num_distinct_terms (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: The number of distinct terms in a homogeneous polynomial of degree 6 with 5 variables is 210 -/
theorem homogeneous_polynomial_terms :
  num_distinct_terms 6 5 = 210 := by sorry

end homogeneous_polynomial_terms_l213_21338


namespace print_time_calculation_l213_21309

/-- Represents a printer with warm-up time and printing speed -/
structure Printer where
  warmupTime : ℕ
  pagesPerMinute : ℕ

/-- Calculates the total time required to print a given number of pages -/
def totalPrintTime (printer : Printer) (pages : ℕ) : ℕ :=
  printer.warmupTime + (pages + printer.pagesPerMinute - 1) / printer.pagesPerMinute

theorem print_time_calculation (printer : Printer) (pages : ℕ) :
  printer.warmupTime = 2 →
  printer.pagesPerMinute = 15 →
  pages = 225 →
  totalPrintTime printer pages = 17 :=
by
  sorry

#eval totalPrintTime ⟨2, 15⟩ 225

end print_time_calculation_l213_21309


namespace logarithm_domain_l213_21385

theorem logarithm_domain (a : ℝ) : 
  (∀ x : ℝ, x < 2 → ∃ y : ℝ, y = Real.log (a - 3 * x)) → a = 6 := by
  sorry

end logarithm_domain_l213_21385


namespace leifs_oranges_l213_21316

theorem leifs_oranges (apples : ℕ) (oranges : ℕ) : apples = 14 → oranges = apples + 10 → oranges = 24 := by
  sorry

end leifs_oranges_l213_21316


namespace sum_odd_integers_21_to_51_l213_21395

/-- The sum of all odd integers from 21 through 51, inclusive, is 576. -/
theorem sum_odd_integers_21_to_51 : 
  (Finset.range 16).sum (fun i => 21 + 2 * i) = 576 := by
  sorry

end sum_odd_integers_21_to_51_l213_21395


namespace diamond_zero_not_always_double_l213_21376

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 2 * |x - y|

-- Statement to prove
theorem diamond_zero_not_always_double : ¬ (∀ x : ℝ, diamond x 0 = 2 * x) := by
  sorry

end diamond_zero_not_always_double_l213_21376


namespace sequence_general_term_l213_21377

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n + 1) →
  (∀ n : ℕ, n ≥ 1 → a n = n^2) :=
by sorry

end sequence_general_term_l213_21377


namespace distribute_five_into_four_l213_21334

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else (n - k) + 1

theorem distribute_five_into_four :
  distribute_objects 5 4 = 4 := by
  sorry

end distribute_five_into_four_l213_21334


namespace expression_simplification_l213_21382

theorem expression_simplification (a : ℝ) (h : a = 2 * Real.cos (π / 3) + 1) :
  (a - a^2 / (a + 1)) / (a^2 / (a^2 - 1)) = 1/2 := by
  sorry

end expression_simplification_l213_21382


namespace spheres_in_base_of_165_pyramid_l213_21306

/-- The number of spheres in a regular triangular pyramid with n levels -/
def pyramid_spheres (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of spheres in the base of a regular triangular pyramid with n levels -/
def base_spheres (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: In a regular triangular pyramid with exactly 165 identical spheres,
    the number of spheres in the base is 45 -/
theorem spheres_in_base_of_165_pyramid :
  ∃ n : ℕ, pyramid_spheres n = 165 ∧ base_spheres n = 45 :=
sorry

end spheres_in_base_of_165_pyramid_l213_21306


namespace product_cde_value_l213_21396

theorem product_cde_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := by
  sorry

end product_cde_value_l213_21396


namespace set_operations_l213_21320

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  ((Aᶜ ∩ Bᶜ) = {x : ℝ | x < -1 ∨ x > 4}) := by
  sorry

end set_operations_l213_21320


namespace supermarket_product_sales_l213_21340

-- Define the linear function
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (sales_quantity x) * (x - 60)

-- Define the given data points
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

theorem supermarket_product_sales :
  -- 1. The function fits the given data points
  (∀ (point : ℝ × ℝ), point ∈ data_points → sales_quantity point.1 = point.2) ∧
  -- 2. The selling price of 70 or 90 dollars per kilogram results in a daily profit of $600
  (profit 70 = 600 ∧ profit 90 = 600) ∧
  -- 3. The maximum daily profit is $800, achieved at a selling price of 80 dollars per kilogram
  (∀ (x : ℝ), profit x ≤ 800) ∧ (profit 80 = 800) :=
by sorry

end supermarket_product_sales_l213_21340


namespace triangle_altitude_l213_21373

theorem triangle_altitude (a b : ℝ) (B : ℝ) (h : ℝ) : 
  a = 2 → b = Real.sqrt 7 → B = π / 3 → h = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_altitude_l213_21373


namespace perfect_squares_exist_l213_21389

theorem perfect_squares_exist : ∃ (a b c d : ℤ),
  (∃ (x : ℤ), (a + b) = x^2) ∧
  (∃ (y : ℤ), (a + c) = y^2) ∧
  (∃ (z : ℤ), (a + d) = z^2) ∧
  (∃ (w : ℤ), (b + c) = w^2) ∧
  (∃ (v : ℤ), (b + d) = v^2) ∧
  (∃ (u : ℤ), (c + d) = u^2) ∧
  (∃ (t : ℤ), (a + b + c + d) = t^2) :=
by
  sorry

end perfect_squares_exist_l213_21389


namespace triangle_obtuse_iff_tangent_product_less_than_one_l213_21300

theorem triangle_obtuse_iff_tangent_product_less_than_one 
  (α β γ : Real) (h_sum : α + β + γ = Real.pi) (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  γ > Real.pi / 2 ↔ Real.tan α * Real.tan β < 1 :=
by sorry

end triangle_obtuse_iff_tangent_product_less_than_one_l213_21300


namespace go_out_to_sea_is_better_l213_21379

/-- Represents the decision to go out to sea or not -/
inductive Decision
| GoOut
| StayIn

/-- Represents the weather condition -/
inductive Weather
| Good
| Bad

/-- The profit or loss for each scenario -/
def profit (d : Decision) (w : Weather) : ℤ :=
  match d, w with
  | Decision.GoOut, Weather.Good => 6000
  | Decision.GoOut, Weather.Bad => -8000
  | Decision.StayIn, _ => -1000

/-- The probability of each weather condition -/
def weather_prob (w : Weather) : ℚ :=
  match w with
  | Weather.Good => 1/2
  | Weather.Bad => 4/10

/-- The expected value of a decision -/
def expected_value (d : Decision) : ℚ :=
  (weather_prob Weather.Good * profit d Weather.Good) +
  (weather_prob Weather.Bad * profit d Weather.Bad)

/-- Theorem stating that going out to sea has a higher expected value -/
theorem go_out_to_sea_is_better :
  expected_value Decision.GoOut > expected_value Decision.StayIn :=
by sorry

end go_out_to_sea_is_better_l213_21379


namespace frank_reading_speed_l213_21301

/-- Given a book with a certain number of pages and the number of days to read it,
    calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that Frank read 102 pages per day. -/
theorem frank_reading_speed :
  pages_per_day 612 6 = 102 := by
  sorry

end frank_reading_speed_l213_21301


namespace quadratic_max_l213_21359

/-- Given a quadratic function f(x) = ax^2 + bx + c where a < 0,
    and x₀ satisfies 2ax + b = 0, then for all x ∈ ℝ, f(x) ≤ f(x₀) -/
theorem quadratic_max (a b c : ℝ) (x₀ : ℝ) (h₁ : a < 0) (h₂ : 2 * a * x₀ + b = 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≤ a * x₀^2 + b * x₀ + c :=
by sorry

end quadratic_max_l213_21359


namespace impossible_shape_l213_21354

/-- Represents a square sheet of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents a folded paper -/
structure FoldedPaper :=
  (paper : Paper)
  (folds : ℕ)
  (is_valid : folds ≤ 2)

/-- Represents a shape cut from the paper -/
inductive Shape
  | CrossesBothFolds
  | CrossesOneFold
  | CrossesNoFolds
  | ContainsCenter

/-- Represents a cut made on the folded paper -/
structure Cut :=
  (folded_paper : FoldedPaper)
  (resulting_shape : Shape)

/-- Theorem stating that a shape crossing both folds without containing the center is impossible -/
theorem impossible_shape (p : Paper) (fp : FoldedPaper) (c : Cut) :
  fp.paper = p →
  fp.folds = 2 →
  c.folded_paper = fp →
  c.resulting_shape = Shape.CrossesBothFolds →
  ¬(c.resulting_shape = Shape.ContainsCenter) →
  False :=
sorry

end impossible_shape_l213_21354


namespace parallel_vectors_x_value_l213_21361

/-- Two 2D vectors are parallel if the cross product of their coordinates is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, prove that if they are parallel, then x = 3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, x)
  let b : ℝ × ℝ := (2, x - 1)
  are_parallel a b → x = 3 := by
  sorry

end parallel_vectors_x_value_l213_21361


namespace vasya_petya_notebooks_different_l213_21337

theorem vasya_petya_notebooks_different (S : Finset ℝ) (h : S.card = 10) :
  let vasya_set := Finset.image (fun (p : ℝ × ℝ) => (p.1 - p.2)^2) (S.product S)
  let petya_set := Finset.image (fun (p : ℝ × ℝ) => |p.1^2 - p.2^2|) (S.product S)
  vasya_set ≠ petya_set :=
by sorry

end vasya_petya_notebooks_different_l213_21337


namespace lottery_jackpot_probability_l213_21330

/-- The number of balls for the MegaBall draw -/
def megaBallCount : ℕ := 30

/-- The number of balls for the WinnerBalls draw -/
def winnerBallCount : ℕ := 49

/-- The number of WinnerBalls drawn -/
def winnerBallsDraw : ℕ := 6

/-- The probability of winning the jackpot in the lottery -/
def jackpotProbability : ℚ := 1 / 419514480

/-- Theorem stating that the probability of winning the jackpot in the given lottery system
    is equal to 1/419,514,480 -/
theorem lottery_jackpot_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / (winnerBallCount.choose winnerBallsDraw) = jackpotProbability := by
  sorry


end lottery_jackpot_probability_l213_21330


namespace quadratic_inequality_range_quadratic_inequality_range_set_l213_21371

/-- For a real number a, if ax^2 + ax + a + 3 > 0 for all real x, then a ≥ 0 -/
theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) → a ≥ 0 := by
  sorry

/-- The set of all real numbers a satisfying the quadratic inequality for all x is [0, +∞) -/
theorem quadratic_inequality_range_set : 
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0} = Set.Ici (0 : ℝ) := by
  sorry

end quadratic_inequality_range_quadratic_inequality_range_set_l213_21371


namespace ellipse_intersection_properties_l213_21374

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem ellipse_intersection_properties :
  let A := (0, -1)
  let B := (4/3, 1/3)
  (A ∈ intersection_points) ∧
  (B ∈ intersection_points) ∧
  (∃ (AB : ℝ), AB = (4 * Real.sqrt 2) / 3 ∧
    AB = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) ∧
  (∃ (S : ℝ), S = 4/3 ∧
    S = (1/2) * ((4 * Real.sqrt 2) / 3) * Real.sqrt 2) :=
by sorry

end ellipse_intersection_properties_l213_21374


namespace computer_factory_month_days_l213_21386

/-- Proves that given a factory producing 5376 computers per month at a constant rate,
    and 4 computers built every 30 minutes, the number of days in the month is 28. -/
theorem computer_factory_month_days : 
  ∀ (computers_per_month : ℕ) (computers_per_30min : ℕ),
    computers_per_month = 5376 →
    computers_per_30min = 4 →
    (computers_per_month / (48 * computers_per_30min) : ℕ) = 28 := by
  sorry

#check computer_factory_month_days

end computer_factory_month_days_l213_21386


namespace sarahs_number_l213_21329

theorem sarahs_number :
  ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 144 ∣ n ∧ 45 ∣ n ∧ n = 2880 := by
  sorry

end sarahs_number_l213_21329


namespace median_salary_is_24000_l213_21303

structure SalaryGroup where
  title : String
  count : Nat
  salary : Nat

def company_data : List SalaryGroup := [
  ⟨"President", 1, 140000⟩,
  ⟨"Vice-President", 4, 92000⟩,
  ⟨"Director", 12, 75000⟩,
  ⟨"Associate Director", 8, 55000⟩,
  ⟨"Administrative Specialist", 38, 24000⟩
]

def total_employees : Nat := (company_data.map (λ g => g.count)).sum

theorem median_salary_is_24000 :
  total_employees = 63 →
  (∃ median_index : Nat, median_index = (total_employees + 1) / 2) →
  (∃ median_salary : Nat, 
    (company_data.map (λ g => List.replicate g.count g.salary)).join.get! (median_index - 1) = median_salary ∧
    median_salary = 24000) :=
by sorry

end median_salary_is_24000_l213_21303


namespace average_rate_for_trip_l213_21366

/-- Given a trip with the following conditions:
  - Total distance is 640 miles
  - First half is driven at 80 miles per hour
  - Second half takes 200% longer than the first half
  Prove that the average rate for the entire trip is 40 miles per hour -/
theorem average_rate_for_trip (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 :=
by sorry

end average_rate_for_trip_l213_21366


namespace exactly_three_combinations_l213_21331

/-- Represents the number of games played -/
def total_games : ℕ := 15

/-- Represents the total points scored -/
def total_points : ℕ := 33

/-- Represents the points earned for a win -/
def win_points : ℕ := 3

/-- Represents the points earned for a draw -/
def draw_points : ℕ := 1

/-- Represents the points earned for a loss -/
def loss_points : ℕ := 0

/-- A combination of wins, draws, and losses -/
structure GameCombination where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Checks if a combination is valid according to the given conditions -/
def is_valid_combination (c : GameCombination) : Prop :=
  c.wins + c.draws + c.losses = total_games ∧
  c.wins * win_points + c.draws * draw_points + c.losses * loss_points = total_points

/-- The theorem to be proved -/
theorem exactly_three_combinations :
  ∃! (combinations : List GameCombination),
    (∀ c ∈ combinations, is_valid_combination c) ∧
    combinations.length = 3 :=
sorry

end exactly_three_combinations_l213_21331


namespace hundredth_digit_is_one_l213_21307

/-- The decimal representation of 7/33 has a repeating pattern of length 2 -/
def decimal_rep_period (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 
    (7 : ℚ) / 33 = (a * 10 + b : ℚ) / 100 + (7 : ℚ) / (33 * 100)

/-- The 100th digit after the decimal point in 7/33 -/
def hundredth_digit : ℕ :=
  sorry

theorem hundredth_digit_is_one :
  decimal_rep_period 2 → hundredth_digit = 1 := by
  sorry

end hundredth_digit_is_one_l213_21307


namespace cubic_inequality_l213_21328

theorem cubic_inequality (x : ℝ) : x^3 - 10*x^2 > -25*x ↔ (0 < x ∧ x < 5) ∨ x > 5 := by
  sorry

end cubic_inequality_l213_21328


namespace perfect_square_identification_l213_21319

theorem perfect_square_identification (a b : ℝ) : 
  (∃ x : ℝ, a^2 - 4*a + 4 = x^2) ∧ 
  (¬∃ x : ℝ, 1 + 4*a^2 = x^2) ∧ 
  (¬∃ x : ℝ, 4*b^2 + 4*b - 1 = x^2) ∧ 
  (¬∃ x : ℝ, a^2 + a*b + b^2 = x^2) := by
sorry

end perfect_square_identification_l213_21319


namespace residue_mod_14_l213_21344

theorem residue_mod_14 : (182 * 12 - 15 * 7 + 3) % 14 = 10 := by sorry

end residue_mod_14_l213_21344


namespace arctan_equation_solution_l213_21333

theorem arctan_equation_solution :
  ∀ x : ℝ, 2 * Real.arctan (1/5) + Real.arctan (1/15) + Real.arctan (1/x) = π/3 → x = -49 := by
  sorry

end arctan_equation_solution_l213_21333


namespace product_fraction_inequality_l213_21325

theorem product_fraction_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (sum_eq_two : a + b + c = 2) : 
  (a / (1 - a)) * (b / (1 - b)) * (c / (1 - c)) ≥ 8 := by
  sorry

end product_fraction_inequality_l213_21325


namespace probability_theorem_l213_21326

def total_marbles : ℕ := 30
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def marbles_selected : ℕ := 4

def probability_two_red_one_blue_one_green : ℚ :=
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
  Nat.choose total_marbles marbles_selected

theorem probability_theorem :
  probability_two_red_one_blue_one_green = 350 / 1827 := by
  sorry

end probability_theorem_l213_21326


namespace money_problem_l213_21313

theorem money_problem (p q r s t : ℚ) : 
  p = q + r + 35 →
  q = (2/5) * p →
  r = (1/7) * p →
  s = 2 * p →
  t = (1/2) * (q + r) →
  p + q + r + s + t = 291.03125 := by
sorry

end money_problem_l213_21313


namespace intersection_points_line_l213_21323

theorem intersection_points_line (s : ℝ) :
  let x : ℝ := (41 * s + 13) / 11
  let y : ℝ := -(2 * s + 6) / 11
  (2 * x - 3 * y = 8 * s + 4) ∧ 
  (x + 4 * y = 3 * s - 1) →
  y = (-22 * x + 272) / 451 := by
sorry

end intersection_points_line_l213_21323


namespace geometric_series_sum_l213_21304

/-- Sum of a geometric series with n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric series -/
def a : ℚ := 1/4

/-- Common ratio of the geometric series -/
def r : ℚ := 1/4

/-- Number of terms to sum -/
def n : ℕ := 6

theorem geometric_series_sum :
  geometricSum a r n = 4095/12288 := by
  sorry

end geometric_series_sum_l213_21304


namespace sqrt_square_sum_zero_implies_diff_l213_21375

theorem sqrt_square_sum_zero_implies_diff (x y : ℝ) : 
  Real.sqrt (8 - x) + (y + 4)^2 = 0 → x - y = 12 := by
  sorry

end sqrt_square_sum_zero_implies_diff_l213_21375


namespace find_N_l213_21347

theorem find_N (X Y Z N : ℝ) 
  (h1 : 0.15 * X = 0.25 * N + Y) 
  (h2 : X + Y = Z) : 
  N = 4.6 * X - 4 * Z := by
sorry

end find_N_l213_21347


namespace arithmetic_mean_of_sixty_integers_from_three_l213_21367

def arithmetic_sequence_sum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_mean_of_sixty_integers_from_three (a₁ n : ℕ) (h₁ : a₁ = 3) (h₂ : n = 60) :
  (arithmetic_sequence_sum a₁ n 1 : ℚ) / n = 32.5 := by
  sorry

end arithmetic_mean_of_sixty_integers_from_three_l213_21367


namespace new_lines_satisfy_axioms_l213_21302

-- Define the type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the type for new lines (parabolas and vertical lines)
inductive NewLine
  | Parabola (a b : ℝ)  -- y = (x + a)² + b
  | VerticalLine (c : ℝ)  -- x = c

-- Define when a point lies on a new line
def lies_on (p : Point) (l : NewLine) : Prop :=
  match l with
  | NewLine.Parabola a b => p.y = (p.x + a)^2 + b
  | NewLine.VerticalLine c => p.x = c

-- Axiom 1: For any two distinct points, there exists a unique new line passing through them
axiom exists_unique_newline (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! l : NewLine, lies_on p1 l ∧ lies_on p2 l

-- Axiom 2: Any two distinct new lines intersect in at most one point
axiom at_most_one_intersection (l1 l2 : NewLine) (h : l1 ≠ l2) :
  ∃! p : Point, lies_on p l1 ∧ lies_on p l2

-- Axiom 3: For any new line and a point not on it, there exists a unique new line
--          passing through the point and not intersecting the given line
axiom exists_unique_parallel (l : NewLine) (p : Point) (h : ¬lies_on p l) :
  ∃! l' : NewLine, lies_on p l' ∧ ∀ q : Point, ¬(lies_on q l ∧ lies_on q l')

-- Theorem: The set of new lines satisfies the three axioms
theorem new_lines_satisfy_axioms :
  (∀ p1 p2 : Point, p1 ≠ p2 → ∃! l : NewLine, lies_on p1 l ∧ lies_on p2 l) ∧
  (∀ l1 l2 : NewLine, l1 ≠ l2 → ∃! p : Point, lies_on p l1 ∧ lies_on p l2) ∧
  (∀ l : NewLine, ∀ p : Point, ¬lies_on p l →
    ∃! l' : NewLine, lies_on p l' ∧ ∀ q : Point, ¬(lies_on q l ∧ lies_on q l')) :=
by sorry

end new_lines_satisfy_axioms_l213_21302


namespace race_outcomes_l213_21343

theorem race_outcomes (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 4) : 
  n * (n - 1) * (n - 2) * (n - 3) = 360 := by
  sorry

end race_outcomes_l213_21343


namespace power_2023_preserves_order_l213_21372

theorem power_2023_preserves_order (a b : ℝ) (h : a > b) : a^2023 > b^2023 := by
  sorry

end power_2023_preserves_order_l213_21372


namespace min_box_height_l213_21315

def box_height (x : ℝ) : ℝ := 2 * x + 2

def box_surface_area (x : ℝ) : ℝ := 9 * x^2 + 8 * x

theorem min_box_height :
  ∃ (x : ℝ), 
    x > 0 ∧
    box_surface_area x ≥ 110 ∧
    (∀ y : ℝ, y > 0 → box_surface_area y ≥ 110 → x ≤ y) ∧
    box_height x = 10 := by
  sorry

end min_box_height_l213_21315


namespace age_difference_correct_l213_21324

/-- The age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Emma's age when her sister is 56 -/
def emma_future_age : ℕ := 47

/-- Emma's sister's age when Emma is 47 -/
def sister_future_age : ℕ := 56

/-- Proof that the age difference is correct -/
theorem age_difference_correct : 
  emma_future_age + age_difference = sister_future_age :=
by sorry

end age_difference_correct_l213_21324


namespace orange_juice_profit_l213_21390

/-- Represents the number of orange trees each sister has -/
def trees_per_sister : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges_per_tree : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges_per_tree : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the price of one cup of juice in dollars -/
def price_per_cup : ℕ := 4

/-- Theorem stating the total money earned from selling orange juice -/
theorem orange_juice_profit : 
  (trees_per_sister * gabriela_oranges_per_tree + 
   trees_per_sister * alba_oranges_per_tree + 
   trees_per_sister * maricela_oranges_per_tree) / oranges_per_cup * price_per_cup = 220000 := by
  sorry

end orange_juice_profit_l213_21390


namespace map_area_ratio_map_area_ratio_not_scale_l213_21350

/-- Represents the scale of a map --/
structure MapScale where
  ratio : ℚ

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℚ := r.length * r.width

/-- Theorem: For a map with scale 1:500, the ratio of map area to actual area is 1:250000 --/
theorem map_area_ratio (scale : MapScale) (map_rect : Rectangle) (actual_rect : Rectangle)
    (h_scale : scale.ratio = 1 / 500)
    (h_length : map_rect.length * 500 = actual_rect.length)
    (h_width : map_rect.width * 500 = actual_rect.width) :
    area map_rect / area actual_rect = 1 / 250000 := by
  sorry

/-- The ratio of map area to actual area is not 1:500 --/
theorem map_area_ratio_not_scale (scale : MapScale) (map_rect : Rectangle) (actual_rect : Rectangle)
    (h_scale : scale.ratio = 1 / 500)
    (h_length : map_rect.length * 500 = actual_rect.length)
    (h_width : map_rect.width * 500 = actual_rect.width) :
    area map_rect / area actual_rect ≠ 1 / 500 := by
  sorry

end map_area_ratio_map_area_ratio_not_scale_l213_21350


namespace octal_subtraction_example_l213_21305

/-- Represents a number in base 8 as a list of digits (least significant first) --/
def OctalNumber := List Nat

/-- Subtraction operation for octal numbers --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from a natural number to its octal representation --/
def to_octal (n : Nat) : OctalNumber :=
  sorry

theorem octal_subtraction_example :
  octal_subtract [4, 3, 5, 7] [7, 6, 2, 3] = [3, 4, 2, 4] :=
sorry

end octal_subtraction_example_l213_21305


namespace same_color_probability_l213_21332

def total_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem same_color_probability :
  let prob_same_color := prob_red^2 + prob_white^2
  prob_same_color = 5/9 := by sorry

end same_color_probability_l213_21332


namespace f_composition_negative_three_equals_zero_l213_21358

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 2/x - 3
  else Real.log (x^2 + 1) / Real.log 10

-- State the theorem
theorem f_composition_negative_three_equals_zero :
  f (f (-3)) = 0 := by
  sorry

end f_composition_negative_three_equals_zero_l213_21358


namespace triangle_is_obtuse_l213_21383

theorem triangle_is_obtuse (A : ℝ) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  π/2 < A ∧ A < π :=
by sorry

end triangle_is_obtuse_l213_21383


namespace distribute_four_to_three_eq_36_l213_21353

/-- The number of ways to distribute four distinct objects into three non-empty groups -/
def distribute_four_to_three : ℕ := 36

/-- Theorem stating that the number of ways to distribute four distinct objects 
    into three non-empty groups is 36 -/
theorem distribute_four_to_three_eq_36 : 
  distribute_four_to_three = 36 := by
  sorry

end distribute_four_to_three_eq_36_l213_21353


namespace population_exceeds_target_in_2125_l213_21365

-- Define the initial year and population
def initialYear : ℕ := 1950
def initialPopulation : ℕ := 750

-- Define the doubling period
def doublingPeriod : ℕ := 35

-- Define the target population
def targetPopulation : ℕ := 15000

-- Function to calculate population after n doubling periods
def populationAfterPeriods (n : ℕ) : ℕ :=
  initialPopulation * 2^n

-- Function to calculate the year after n doubling periods
def yearAfterPeriods (n : ℕ) : ℕ :=
  initialYear + n * doublingPeriod

-- Theorem to prove
theorem population_exceeds_target_in_2125 :
  ∃ n : ℕ, yearAfterPeriods n = 2125 ∧ populationAfterPeriods n > targetPopulation ∧
  ∀ m : ℕ, m < n → populationAfterPeriods m ≤ targetPopulation :=
sorry

end population_exceeds_target_in_2125_l213_21365


namespace coloring_problem_l213_21370

/-- Represents the number of objects colored by each person -/
def objects_per_person (total_colors : ℕ) (num_people : ℕ) : ℕ :=
  total_colors / num_people

/-- Represents the total number of objects colored -/
def total_objects (objects_per_person : ℕ) (num_people : ℕ) : ℕ :=
  objects_per_person * num_people

theorem coloring_problem (total_colors : ℕ) (num_people : ℕ) 
  (h1 : total_colors = 24) 
  (h2 : num_people = 3) :
  total_objects (objects_per_person total_colors num_people) num_people = 24 := by
sorry

end coloring_problem_l213_21370


namespace rental_cost_difference_l213_21322

/-- Calculates the difference in rental costs between a ski boat and a sailboat for a given duration. -/
theorem rental_cost_difference 
  (sailboat_cost_per_day : ℕ)
  (ski_boat_cost_per_hour : ℕ)
  (hours_per_day : ℕ)
  (num_days : ℕ)
  (h1 : sailboat_cost_per_day = 60)
  (h2 : ski_boat_cost_per_hour = 80)
  (h3 : hours_per_day = 3)
  (h4 : num_days = 2) :
  ski_boat_cost_per_hour * hours_per_day * num_days - sailboat_cost_per_day * num_days = 360 :=
by
  sorry

#check rental_cost_difference

end rental_cost_difference_l213_21322


namespace courier_delivery_patterns_l213_21339

/-- Represents the number of acceptable delivery patterns for n offices -/
def P : ℕ → ℕ
| 0 => 1  -- Base case: only one way to deliver to 0 offices
| 1 => 2  -- Can either deliver or not deliver to 1 office
| 2 => 4  -- All combinations for 2 offices
| 3 => 8  -- All combinations for 3 offices
| 4 => 15 -- All combinations for 4 offices, excluding all non-deliveries
| n + 5 => P (n + 4) + P (n + 3) + P (n + 2) + P (n + 1)

theorem courier_delivery_patterns :
  P 12 = 927 := by
  sorry


end courier_delivery_patterns_l213_21339


namespace contrapositive_equivalence_l213_21393

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 1 → a^2 > 1) ↔ (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by sorry

end contrapositive_equivalence_l213_21393


namespace largest_n_with_negative_sum_l213_21397

theorem largest_n_with_negative_sum 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) 
  (h_sum : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) 
  (h_a6_neg : a 6 < 0) 
  (h_a4_a9_pos : a 4 + a 9 > 0) : 
  (∀ n > 11, S n ≥ 0) ∧ S 11 < 0 :=
sorry

end largest_n_with_negative_sum_l213_21397


namespace quadratic_roots_constraint_l213_21345

theorem quadratic_roots_constraint (x y : ℤ) : 
  (∃ α β : ℝ, α^2 + β^2 < 4 ∧ ∀ t : ℝ, t^2 + x*t + y = 0 ↔ t = α ∨ t = β) →
  (x = -2 ∧ y = 1) ∨
  (x = -1 ∧ y = -1) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 0 ∧ y = -1) ∨
  (x = 0 ∧ y = 0) ∨
  (x = 1 ∧ y = 0) ∨
  (x = 1 ∧ y = -1) ∨
  (x = 2 ∧ y = 1) :=
by sorry

end quadratic_roots_constraint_l213_21345


namespace inequality_solution_count_l213_21363

theorem inequality_solution_count : ∃! (n : ℤ), (n - 2) * (n + 4) * (n - 3) < 0 := by sorry

end inequality_solution_count_l213_21363


namespace symmetry_of_lines_l213_21310

/-- Given two lines in a 2D plane, this function returns true if they are symmetric with respect to a third line. -/
def are_symmetric_lines (line1 line2 axis : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, line1 x y ↔ line2 (y - 1) (x + 1)

/-- The line y = 2x + 3 -/
def line1 (x y : ℝ) : Prop := y = 2 * x + 3

/-- The line y = x + 1 (the axis of symmetry) -/
def axis (x y : ℝ) : Prop := y = x + 1

/-- The line x = 2y (which is equivalent to x - 2y = 0) -/
def line2 (x y : ℝ) : Prop := x = 2 * y

theorem symmetry_of_lines : are_symmetric_lines line1 line2 axis := by
  sorry

end symmetry_of_lines_l213_21310


namespace mountain_height_proof_l213_21341

def mountain_height (h : ℝ) : Prop :=
  h > 7900 ∧ h < 8000

theorem mountain_height_proof (h : ℝ) 
  (peter_false : ¬(h ≥ 8000))
  (mary_false : ¬(h ≤ 7900))
  (john_false : ¬(h ≤ 7500)) :
  mountain_height h :=
sorry

end mountain_height_proof_l213_21341


namespace least_integer_with_2035_divisors_l213_21317

/-- The number of divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the least positive integer with exactly 2035 distinct positive divisors -/
def n : ℕ := sorry

/-- m and k are integers such that n = m * 6^k and 6 is not a divisor of m -/
def m : ℕ := sorry
def k : ℕ := sorry

theorem least_integer_with_2035_divisors :
  num_divisors n = 2035 ∧
  n = m * 6^k ∧
  ¬(6 ∣ m) ∧
  ∀ i : ℕ, i < n → num_divisors i < 2035 →
  m + k = 26 := by sorry

end least_integer_with_2035_divisors_l213_21317


namespace max_value_of_expression_l213_21388

theorem max_value_of_expression (x y : ℝ) (h : x * y > 0) :
  (∃ (z : ℝ), ∀ (a b : ℝ), a * b > 0 → x / (x + y) + 2 * y / (x + 2 * y) ≤ z) ∧
  (x / (x + y) + 2 * y / (x + 2 * y) ≤ 4 - 2 * Real.sqrt 2) :=
sorry

end max_value_of_expression_l213_21388


namespace equation_solution_l213_21348

theorem equation_solution (x : ℝ) (h : x > 1) :
  (x^2 / (x - 1)) + Real.sqrt (x - 1) + (Real.sqrt (x - 1) / x^2) =
  ((x - 1) / x^2) + (1 / Real.sqrt (x - 1)) + (x^2 / Real.sqrt (x - 1)) ↔
  x = 2 := by
sorry

end equation_solution_l213_21348


namespace prob_laurent_ge_2chloe_l213_21391

/-- Represents a uniform distribution over a real interval -/
structure UniformDist (a b : ℝ) where
  (a_le_b : a ≤ b)

/-- The probability that a random variable from distribution Y is at least twice 
    a random variable from distribution X -/
noncomputable def prob_y_ge_2x (X : UniformDist 0 1000) (Y : UniformDist 0 2000) : ℝ :=
  (1000 * 1000 / 2) / (1000 * 2000)

/-- Theorem stating that the probability of Laurent's number being at least 
    twice Chloe's number is 1/4 -/
theorem prob_laurent_ge_2chloe :
  ∀ (X : UniformDist 0 1000) (Y : UniformDist 0 2000),
  prob_y_ge_2x X Y = 1/4 := by sorry

end prob_laurent_ge_2chloe_l213_21391


namespace factory_production_l213_21312

/-- Calculates the number of toys produced per day in a factory -/
def toys_per_day (total_toys : ℕ) (work_days : ℕ) : ℕ :=
  total_toys / work_days

theorem factory_production :
  let total_weekly_production := 6000
  let work_days_per_week := 4
  toys_per_day total_weekly_production work_days_per_week = 1500 := by
  sorry

end factory_production_l213_21312


namespace volumes_equal_l213_21336

-- Define the region for V₁
def region_V1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ x ≥ -4 ∧ x ≤ 4

-- Define the region for V₂
def region_V2 (x y : ℝ) : Prop :=
  x^2 * y^2 ≤ 16 ∧ x^2 + (y-2)^2 ≥ 4 ∧ x^2 + (y+2)^2 ≥ 4

-- Define the volume of revolution around y-axis
noncomputable def volume_of_revolution (region : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- State the theorem
theorem volumes_equal :
  volume_of_revolution region_V1 = volume_of_revolution region_V2 :=
sorry

end volumes_equal_l213_21336


namespace vertical_pairwise_sets_l213_21357

/-- Definition of a vertical pairwise set -/
def is_vertical_pairwise_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

/-- Set M₁: y = 1/x² -/
def M₁ : Set (ℝ × ℝ) :=
  {p | p.2 = 1 / (p.1 ^ 2) ∧ p.1 ≠ 0}

/-- Set M₂: y = sin x + 1 -/
def M₂ : Set (ℝ × ℝ) :=
  {p | p.2 = Real.sin p.1 + 1}

/-- Set M₄: y = 2ˣ - 2 -/
def M₄ : Set (ℝ × ℝ) :=
  {p | p.2 = 2 ^ p.1 - 2}

/-- Theorem: M₁, M₂, and M₄ are vertical pairwise sets -/
theorem vertical_pairwise_sets :
  is_vertical_pairwise_set M₁ ∧
  is_vertical_pairwise_set M₂ ∧
  is_vertical_pairwise_set M₄ := by
  sorry

end vertical_pairwise_sets_l213_21357


namespace different_signs_implies_range_l213_21384

theorem different_signs_implies_range (m : ℝ) : 
  ((2 - m) * (|m| - 3) < 0) → ((-3 < m ∧ m < 2) ∨ m > 3) := by
sorry

end different_signs_implies_range_l213_21384


namespace data_transmission_time_l213_21360

theorem data_transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 80 →
  chunks_per_block = 640 →
  transmission_rate = 160 →
  (blocks * chunks_per_block : ℚ) / transmission_rate = 320 :=
by sorry

end data_transmission_time_l213_21360


namespace complex_on_imaginary_axis_l213_21355

theorem complex_on_imaginary_axis (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 1) → z.re = 0 := by
  sorry

end complex_on_imaginary_axis_l213_21355


namespace rectangle_perimeter_10_l213_21342

/-- A rectangle with sides a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- The perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The sum of three sides of a rectangle. -/
def sum_three_sides (r : Rectangle) : Set ℝ := {2 * r.a + r.b, r.a + 2 * r.b}

/-- Theorem stating that there exists a rectangle with perimeter 10,
    given that the sum of the lengths of three different sides can be equal to 6 or 9. -/
theorem rectangle_perimeter_10 :
  ∃ r : Rectangle, (6 ∈ sum_three_sides r ∨ 9 ∈ sum_three_sides r) ∧ perimeter r = 10 := by
  sorry

end rectangle_perimeter_10_l213_21342


namespace cloth_cost_price_l213_21346

/-- Represents the cost price per metre of cloth -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Theorem: Given a cloth of 300 metres sold for Rs. 9000 with a loss of Rs. 6 per metre,
    the cost price for one metre of cloth is Rs. 36 -/
theorem cloth_cost_price :
  cost_price_per_metre 300 9000 6 = 36 := by
  sorry

end cloth_cost_price_l213_21346


namespace horner_method_correct_l213_21381

def horner_polynomial (x : ℚ) : ℚ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v4 (x : ℚ) : ℚ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 79
  v3 * x - 8

theorem horner_method_correct :
  horner_v4 (-4) = 220 :=
sorry

end horner_method_correct_l213_21381


namespace tangent_curves_l213_21378

/-- The value of α for which e^x is tangent to αx^2 -/
theorem tangent_curves (f g : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, g x = α * x^2) →
  (∃ x, f x = g x ∧ deriv f x = deriv g x) →
  α = Real.exp 2 / 4 := by
  sorry

end tangent_curves_l213_21378


namespace roberto_outfits_l213_21394

/-- The number of pairs of trousers Roberto has -/
def trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def shirts : ℕ := 6

/-- The number of jackets Roberto has -/
def jackets : ℕ := 4

/-- The number of pairs of shoes Roberto has -/
def shoes : ℕ := 3

/-- The number of jackets with shoe restrictions -/
def restricted_jackets : ℕ := 1

/-- The number of shoes that can be worn with the restricted jacket -/
def shoes_per_restricted_jacket : ℕ := 2

/-- The total number of outfits Roberto can put together -/
def total_outfits : ℕ := trousers * shirts * (
  (jackets - restricted_jackets) * shoes +
  restricted_jackets * shoes_per_restricted_jacket
)

theorem roberto_outfits :
  total_outfits = 330 :=
by sorry

end roberto_outfits_l213_21394


namespace no_ten_digit_divisor_with_different_digits_l213_21399

/-- The number consisting of 1000 ones -/
def number_of_ones : ℕ := 10^1000 - 1

/-- A function to check if a natural number has exactly 10 digits -/
def has_ten_digits (n : ℕ) : Prop := 10^9 ≤ n ∧ n < 10^10

/-- A function to check if all digits in a number are different -/
def all_digits_different (n : ℕ) : Prop :=
  ∀ d₁ d₂, 0 ≤ d₁ ∧ d₁ < 10 ∧ 0 ≤ d₂ ∧ d₂ < 10 → d₁ ≠ d₂ → (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)

/-- The main theorem stating that the number of 1000 ones has no ten-digit divisor with all different digits -/
theorem no_ten_digit_divisor_with_different_digits : 
  ¬ ∃ (d : ℕ), d ∣ number_of_ones ∧ has_ten_digits d ∧ all_digits_different d := by
  sorry

end no_ten_digit_divisor_with_different_digits_l213_21399
