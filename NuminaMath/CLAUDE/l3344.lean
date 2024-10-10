import Mathlib

namespace unique_four_digit_solution_l3344_334448

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_equation (a b c d : ℕ) : Prop :=
  1000 * a + 100 * b + 10 * c + d - (100 * a + 10 * b + c) - (10 * a + b) - a = 1995

theorem unique_four_digit_solution :
  ∃! (abcd : ℕ), is_four_digit abcd ∧ 
    ∃ (a b c d : ℕ), abcd = 1000 * a + 100 * b + 10 * c + d ∧ digit_equation a b c d ∧
    a ≠ 0 := by sorry

end unique_four_digit_solution_l3344_334448


namespace harriets_siblings_product_l3344_334453

/-- Given a family where Harry has 4 sisters and 6 brothers, and Harriet is one of Harry's sisters,
    this theorem proves that the product of the number of Harriet's sisters and brothers is 24. -/
theorem harriets_siblings_product (harry_sisters : ℕ) (harry_brothers : ℕ) 
  (harriet_sisters : ℕ) (harriet_brothers : ℕ) :
  harry_sisters = 4 →
  harry_brothers = 6 →
  harriet_sisters = harry_sisters - 1 →
  harriet_brothers = harry_brothers →
  harriet_sisters * harriet_brothers = 24 :=
by sorry

end harriets_siblings_product_l3344_334453


namespace linear_function_quadrants_l3344_334463

/-- A linear function y = (1-2m)x + m + 1 passes through the first, second, and third quadrants
    if and only if -1 < m < 1/2 -/
theorem linear_function_quadrants (m : ℝ) :
  (∀ x y : ℝ, y = (1 - 2*m)*x + m + 1 →
    (∃ x₁ y₁, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = (1 - 2*m)*x₁ + m + 1) ∧
    (∃ x₂ y₂, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = (1 - 2*m)*x₂ + m + 1) ∧
    (∃ x₃ y₃, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = (1 - 2*m)*x₃ + m + 1)) ↔
  -1 < m ∧ m < 1/2 :=
sorry

end linear_function_quadrants_l3344_334463


namespace sales_equation_l3344_334491

theorem sales_equation (x : ℝ) 
  (h1 : x > 0) 
  (h2 : 5000 / (x + 1) > 0) -- sales quantity last year is positive
  (h3 : 5000 / x > 0) -- sales quantity this year is positive
  (h4 : 5000 / (x + 1) = 5000 / x) -- sales quantity remains the same
  : 5000 / (x + 1) = 5000 * (1 - 0.2) / x := by
  sorry

end sales_equation_l3344_334491


namespace planet_can_be_fully_explored_l3344_334465

/-- Represents a spherical planet -/
structure Planet :=
  (equatorial_length : ℝ)

/-- Represents a rover's exploration path on the planet -/
structure ExplorationPath :=
  (length : ℝ)
  (covers_all_points : Bool)

/-- Checks if an exploration path fully explores the planet -/
def fully_explores (p : Planet) (path : ExplorationPath) : Prop :=
  path.length ≤ 600 ∧ path.covers_all_points = true

/-- Theorem stating that the planet can be fully explored -/
theorem planet_can_be_fully_explored (p : Planet) 
  (h : p.equatorial_length = 400) : 
  ∃ path : ExplorationPath, fully_explores p path :=
sorry

end planet_can_be_fully_explored_l3344_334465


namespace range_of_m_l3344_334451

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (9 : ℝ)^x - m*(3 : ℝ)^x + 4 ≤ 0) → m ≥ 4 := by
  sorry

end range_of_m_l3344_334451


namespace ratio_sum_to_last_l3344_334452

theorem ratio_sum_to_last {a b c : ℝ} (h : a / c = 3 / 7 ∧ b / c = 4 / 7) :
  (a + b + c) / c = 2 := by
  sorry

end ratio_sum_to_last_l3344_334452


namespace sheila_weekly_earnings_l3344_334493

/-- Calculates the weekly earnings of a worker given their work schedule and hourly wage. -/
def weekly_earnings (hours_per_day_1 : ℕ) (days_1 : ℕ) (hours_per_day_2 : ℕ) (days_2 : ℕ) (hourly_wage : ℕ) : ℕ :=
  (hours_per_day_1 * days_1 + hours_per_day_2 * days_2) * hourly_wage

/-- Proves that Sheila's weekly earnings are $216 given her work schedule and hourly wage. -/
theorem sheila_weekly_earnings :
  weekly_earnings 8 3 6 2 6 = 216 := by
  sorry

end sheila_weekly_earnings_l3344_334493


namespace perimeter_of_fourth_figure_l3344_334495

/-- Given four planar figures composed of identical triangles, prove that the perimeter of the fourth figure is 10 cm. -/
theorem perimeter_of_fourth_figure
  (p₁ : ℝ) (p₂ : ℝ) (p₃ : ℝ) (p₄ : ℝ)
  (h₁ : p₁ = 8)
  (h₂ : p₂ = 11.4)
  (h₃ : p₃ = 14.7)
  (h_relation : p₁ + p₂ + p₄ = 2 * p₃) :
  p₄ = 10 := by
  sorry

end perimeter_of_fourth_figure_l3344_334495


namespace distance_between_points_l3344_334460

theorem distance_between_points (m : ℝ) :
  let P : ℝ × ℝ × ℝ := (m, 0, 0)
  let P₁ : ℝ × ℝ × ℝ := (4, 1, 2)
  (m - 4)^2 + 1^2 + 2^2 = 30 → m = 9 ∨ m = -1 := by
  sorry

end distance_between_points_l3344_334460


namespace min_value_theorem_l3344_334411

/-- Triangle ABC with area 2 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (area : ℝ)
  (area_eq : area = 2)

/-- Function f mapping a point to areas of subtriangles -/
def f (T : Triangle) (P : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the minimum value of 1/x + 4/y -/
theorem min_value_theorem (T : Triangle) :
  ∀ P : ℝ × ℝ, 
  ∀ x y : ℝ,
  f T P = (1, x, y) →
  (∀ a b : ℝ, f T (a, b) = (1, x, y) → 1/x + 4/y ≥ 9) ∧ 
  (∃ a b : ℝ, f T (a, b) = (1, x, y) ∧ 1/x + 4/y = 9) :=
sorry

end min_value_theorem_l3344_334411


namespace base7_246_equals_base10_132_l3344_334419

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Proves that 246 in base 7 is equal to 132 in base 10 -/
theorem base7_246_equals_base10_132 : base7ToBase10 2 4 6 = 132 := by
  sorry

end base7_246_equals_base10_132_l3344_334419


namespace sum_of_five_integers_l3344_334418

theorem sum_of_five_integers (a b c d e : ℕ) :
  a ∈ Finset.range 20 ∧ 
  b ∈ Finset.range 20 ∧ 
  c ∈ Finset.range 20 ∧ 
  d ∈ Finset.range 20 ∧ 
  e ∈ Finset.range 20 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  15 ≤ a + b + c + d + e ∧ a + b + c + d + e ≤ 90 := by
  sorry

end sum_of_five_integers_l3344_334418


namespace y1_greater_than_y2_l3344_334458

/-- Given that (-4, y₁) and (2, y₂) both lie on the line y = -2x + 3, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  y₁ = -2 * (-4) + 3 → y₂ = -2 * 2 + 3 → y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l3344_334458


namespace train_speed_l3344_334410

/-- The speed of a train given the time to pass a pole and a stationary train -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) :
  t_pole = 8 →
  t_stationary = 18 →
  l_stationary = 400 →
  ∃ (speed : ℝ), speed = 144 ∧ speed * 1000 / 3600 * t_pole = speed * 1000 / 3600 * t_stationary - l_stationary :=
by sorry

end train_speed_l3344_334410


namespace total_attendance_l3344_334428

def wedding_reception (bride_couples groom_couples friends : ℕ) : ℕ :=
  2 * (bride_couples + groom_couples) + friends

theorem total_attendance : wedding_reception 20 20 100 = 180 := by
  sorry

end total_attendance_l3344_334428


namespace hcf_problem_l3344_334442

theorem hcf_problem (a b H : ℕ+) : 
  (Nat.gcd a b = H) →
  (Nat.lcm a b = H * 13 * 14) →
  (max a b = 322) →
  (H = 14) := by
sorry

end hcf_problem_l3344_334442


namespace product_of_binomials_l3344_334420

theorem product_of_binomials (a : ℝ) : (a + 2) * (2 * a - 3) = 2 * a^2 + a - 6 := by
  sorry

end product_of_binomials_l3344_334420


namespace intersection_points_sum_l3344_334483

def f (x : ℝ) : ℝ := (x + 2) * (x - 4)

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (2 * x)

def intersection_points_fg : ℕ := 2

def intersection_points_fh : ℕ := 2

theorem intersection_points_sum : 
  10 * intersection_points_fg + intersection_points_fh = 22 := by sorry

end intersection_points_sum_l3344_334483


namespace direct_proportion_second_fourth_quadrants_l3344_334422

/-- A function f(x) = ax^b is a direct proportion function if and only if b = 1 -/
def is_direct_proportion (a : ℝ) (b : ℝ) : Prop :=
  b = 1

/-- A function f(x) = ax^b has its graph in the second and fourth quadrants if and only if a < 0 -/
def in_second_and_fourth_quadrants (a : ℝ) : Prop :=
  a < 0

/-- The main theorem stating that for y=(m+1)x^(m^2-3) to be a direct proportion function
    with its graph in the second and fourth quadrants, m must be -2 -/
theorem direct_proportion_second_fourth_quadrants :
  ∀ m : ℝ, is_direct_proportion (m + 1) (m^2 - 3) ∧ 
            in_second_and_fourth_quadrants (m + 1) →
            m = -2 :=
by sorry

end direct_proportion_second_fourth_quadrants_l3344_334422


namespace hypotenuse_ratio_l3344_334426

/-- Represents a right-angled triangle with a 30° angle -/
structure Triangle30 where
  hypotenuse : ℝ
  shared_side : ℝ
  hypotenuse_gt_shared : hypotenuse > shared_side

/-- The three triangles in our problem -/
def three_triangles (a b c : Triangle30) : Prop :=
  a.shared_side = b.shared_side ∧ 
  b.shared_side = c.shared_side ∧ 
  a.hypotenuse ≠ b.hypotenuse ∧ 
  b.hypotenuse ≠ c.hypotenuse ∧ 
  a.hypotenuse ≠ c.hypotenuse

theorem hypotenuse_ratio (a b c : Triangle30) :
  three_triangles a b c →
  (∃ (k : ℝ), k > 0 ∧ 
    (max a.hypotenuse (max b.hypotenuse c.hypotenuse) = 2 * k) ∧
    (max (min a.hypotenuse b.hypotenuse) c.hypotenuse = 2 * k / Real.sqrt 3) ∧
    (min a.hypotenuse (min b.hypotenuse c.hypotenuse) = k)) :=
by sorry

end hypotenuse_ratio_l3344_334426


namespace angle_sum_at_point_l3344_334413

theorem angle_sum_at_point (y : ℝ) : 
  150 + y + 2*y = 360 → y = 70 := by
sorry

end angle_sum_at_point_l3344_334413


namespace sandys_books_l3344_334476

theorem sandys_books (total_spent : ℕ) (books_second_shop : ℕ) (avg_price : ℕ) :
  total_spent = 1920 →
  books_second_shop = 55 →
  avg_price = 16 →
  ∃ (books_first_shop : ℕ), 
    books_first_shop = 65 ∧
    avg_price * (books_first_shop + books_second_shop) = total_spent :=
by sorry

end sandys_books_l3344_334476


namespace contrapositive_equivalence_l3344_334485

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a - 8 > b - 8) → ¬(a > b)) ↔ ((a - 8 ≤ b - 8) → (a ≤ b)) := by sorry

end contrapositive_equivalence_l3344_334485


namespace pencil_count_l3344_334497

theorem pencil_count (reeta_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils = 2 * reeta_pencils + 4 →
  anika_pencils + reeta_pencils = 64 := by
sorry

end pencil_count_l3344_334497


namespace sum_of_a_and_b_equals_four_l3344_334406

theorem sum_of_a_and_b_equals_four (a b : ℝ) (h : b + (a - 2) * Complex.I = 1 + Complex.I) : a + b = 4 := by
  sorry

end sum_of_a_and_b_equals_four_l3344_334406


namespace toy_selling_price_l3344_334437

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the number of toys whose cost price equals the total gain. -/
def total_selling_price (num_toys : ℕ) (cost_price : ℕ) (gain_toys : ℕ) : ℕ :=
  num_toys * cost_price + gain_toys * cost_price

/-- Proves that the total selling price of 18 toys is 23100,
    given a cost price of 1100 per toy and a gain equal to the cost of 3 toys. -/
theorem toy_selling_price :
  total_selling_price 18 1100 3 = 23100 := by
  sorry

end toy_selling_price_l3344_334437


namespace problem_solution_l3344_334436

theorem problem_solution : ∃! n : ℕ, n > 1 ∧ Nat.Prime n ∧ Even n ∧ n ≠ 9 ∧ ¬(15 ∣ n) :=
by
  sorry

end problem_solution_l3344_334436


namespace pants_price_l3344_334447

theorem pants_price (total_cost : ℝ) (shirt_price : ℝ → ℝ) (shoes_price : ℝ → ℝ) 
  (h1 : total_cost = 340)
  (h2 : ∀ p, shirt_price p = 3/4 * p)
  (h3 : ∀ p, shoes_price p = p + 10) :
  ∃ p, p = 120 ∧ total_cost = shirt_price p + p + shoes_price p :=
sorry

end pants_price_l3344_334447


namespace recreation_spending_percentage_l3344_334446

theorem recreation_spending_percentage
  (last_week_wages : ℝ)
  (last_week_recreation_percent : ℝ)
  (wage_decrease_percent : ℝ)
  (this_week_recreation_increase : ℝ)
  (h1 : last_week_recreation_percent = 10)
  (h2 : wage_decrease_percent = 10)
  (h3 : this_week_recreation_increase = 360) :
  let this_week_wages := last_week_wages * (1 - wage_decrease_percent / 100)
  let last_week_recreation := last_week_wages * (last_week_recreation_percent / 100)
  let this_week_recreation := last_week_recreation * (this_week_recreation_increase / 100)
  this_week_recreation / this_week_wages * 100 = 40 := by
sorry

end recreation_spending_percentage_l3344_334446


namespace glove_profit_is_810_l3344_334434

/-- Calculates the profit from selling gloves given the purchase and sales information. -/
def glove_profit (total_pairs : ℕ) (cost_per_pair : ℚ) (sold_pairs_high : ℕ) (price_high : ℚ) (price_low : ℚ) : ℚ :=
  let remaining_pairs := total_pairs - sold_pairs_high
  let total_cost := cost_per_pair * total_pairs
  let revenue_high := price_high * sold_pairs_high
  let revenue_low := price_low * remaining_pairs
  let total_revenue := revenue_high + revenue_low
  total_revenue - total_cost

/-- The profit from selling gloves under the given conditions is 810 yuan. -/
theorem glove_profit_is_810 :
  glove_profit 600 12 470 14 11 = 810 := by
  sorry

#eval glove_profit 600 12 470 14 11

end glove_profit_is_810_l3344_334434


namespace carries_hourly_wage_l3344_334480

/-- Carrie's work and savings scenario --/
theorem carries_hourly_wage (hours_per_week : ℕ) (weeks : ℕ) (bike_cost : ℕ) (leftover : ℕ) :
  hours_per_week = 35 →
  weeks = 4 →
  bike_cost = 400 →
  leftover = 720 →
  ∃ (hourly_wage : ℚ), hourly_wage = 8 ∧ 
    (hourly_wage * (hours_per_week * weeks : ℚ) : ℚ) = (bike_cost + leftover : ℚ) := by
  sorry


end carries_hourly_wage_l3344_334480


namespace integral_proof_l3344_334404

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  deriv (fun x => log (abs (x + 1)) - 1 / (2 * (x + 2)^2)) x =
  (x^3 + 6*x^2 + 13*x + 9) / ((x + 1) * (x + 2)^3) := by
sorry

end integral_proof_l3344_334404


namespace set_of_integers_between_10_and_16_l3344_334478

def S : Set ℤ := {n | 10 < n ∧ n < 16}

theorem set_of_integers_between_10_and_16 : S = {11, 12, 13, 14, 15} := by
  sorry

end set_of_integers_between_10_and_16_l3344_334478


namespace paint_cans_theorem_l3344_334408

/-- The number of rooms that can be painted with one can of paint -/
def rooms_per_can : ℚ :=
  (40 - 32) / 4

/-- The number of cans needed to paint 32 rooms -/
def cans_for_32_rooms : ℚ :=
  32 / rooms_per_can

theorem paint_cans_theorem :
  cans_for_32_rooms = 16 := by
  sorry

end paint_cans_theorem_l3344_334408


namespace largest_two_digit_multiple_minus_one_l3344_334435

theorem largest_two_digit_multiple_minus_one : ∃ (n : ℕ), n = 83 ∧ 
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ 
    (∃ k : ℕ, m + 1 = 3 * k) ∧ 
    (∃ k : ℕ, m + 1 = 4 * k) ∧ 
    (∃ k : ℕ, m + 1 = 5 * k) ∧ 
    (∃ k : ℕ, m + 1 = 7 * k) → 
  m ≤ n) := by
  sorry

end largest_two_digit_multiple_minus_one_l3344_334435


namespace quadratic_real_roots_l3344_334445

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℂ, x^2 - (2*Complex.I - 1)*x + 3*m - Complex.I = 0 ∧ x.im = 0) → m = 1/12 := by
  sorry

end quadratic_real_roots_l3344_334445


namespace f_composition_equal_range_l3344_334440

/-- The function f(x) = x^2 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

/-- The theorem stating the range of a -/
theorem f_composition_equal_range (a : ℝ) :
  ({y | ∃ x, y = f a (f a x)} = {y | ∃ x, y = f a x}) →
  (a ≥ 4 ∨ a ≤ -2) :=
by sorry

end f_composition_equal_range_l3344_334440


namespace total_charts_brought_l3344_334423

/-- Represents the number of associate professors -/
def associate_profs : ℕ := 2

/-- Represents the number of assistant professors -/
def assistant_profs : ℕ := 7

/-- Represents the total number of people present -/
def total_people : ℕ := 9

/-- Represents the total number of pencils brought -/
def total_pencils : ℕ := 11

/-- Represents the number of pencils each associate professor brings -/
def pencils_per_associate : ℕ := 2

/-- Represents the number of pencils each assistant professor brings -/
def pencils_per_assistant : ℕ := 1

/-- Represents the number of charts each associate professor brings -/
def charts_per_associate : ℕ := 1

/-- Represents the number of charts each assistant professor brings -/
def charts_per_assistant : ℕ := 2

theorem total_charts_brought : 
  associate_profs * charts_per_associate + assistant_profs * charts_per_assistant = 16 :=
by sorry

end total_charts_brought_l3344_334423


namespace arithmetic_mean_of_18_24_42_l3344_334405

theorem arithmetic_mean_of_18_24_42 :
  let numbers : List ℕ := [18, 24, 42]
  (numbers.sum : ℚ) / numbers.length = 28 := by sorry

end arithmetic_mean_of_18_24_42_l3344_334405


namespace intersection_of_A_and_B_l3344_334430

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l3344_334430


namespace consecutive_pages_sum_l3344_334462

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end consecutive_pages_sum_l3344_334462


namespace quadrilateral_inequality_l3344_334489

theorem quadrilateral_inequality (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h1 : a + b > e) (h2 : c + d > e) (h3 : a + d > f) (h4 : b + c > f) :
  (a + b + c + d) * (e + f) > 2 * (e^2 + f^2) := by
sorry

end quadrilateral_inequality_l3344_334489


namespace hyperbola_max_eccentricity_l3344_334427

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and a point P on the right branch of the hyperbola satisfying |PF₁| = 4|PF₂|,
    the maximum value of the eccentricity e is 5/3. -/
theorem hyperbola_max_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ e_max : ℝ, e_max = 5/3 ∧
  ∀ (x y e : ℝ),
    x^2/a^2 - y^2/b^2 = 1 →
    x ≥ a →
    ∃ (F₁ F₂ : ℝ × ℝ),
      let P := (x, y)
      let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
      let d₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
      d₁ = 4 * d₂ →
      e = Real.sqrt (1 + b^2/a^2) →
      e ≤ e_max :=
by sorry

end hyperbola_max_eccentricity_l3344_334427


namespace leak_empty_time_correct_l3344_334443

/-- Represents a tank with a leak and an inlet pipe -/
structure Tank where
  capacity : ℝ
  inletRate : ℝ
  emptyTimeWithInlet : ℝ

/-- Calculates the time it takes for the leak alone to empty the tank -/
def leakEmptyTime (t : Tank) : ℝ :=
  -- Definition to be proved
  9

/-- Theorem stating the correct leak empty time for the given tank -/
theorem leak_empty_time_correct (t : Tank) 
  (h1 : t.capacity = 12960)
  (h2 : t.inletRate = 6 * 60)  -- 6 litres per minute converted to per hour
  (h3 : t.emptyTimeWithInlet = 12) : 
  leakEmptyTime t = 9 := by
  sorry

#check leak_empty_time_correct

end leak_empty_time_correct_l3344_334443


namespace sin_2010th_derivative_l3344_334441

open Real

-- Define the recursive function for the nth derivative of sin x
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

-- State the theorem
theorem sin_2010th_derivative :
  ∀ x, f 2010 x = -sin x :=
by
  sorry

end sin_2010th_derivative_l3344_334441


namespace sum_of_fractions_l3344_334494

theorem sum_of_fractions : (10 + 20 + 30 + 40) / 10 + 10 / (10 + 20 + 30 + 40) = 10.1 := by
  sorry

end sum_of_fractions_l3344_334494


namespace emily_egg_collection_l3344_334431

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ :=
  let set_a_eggs := 200 * 36 + 250 * 24
  let set_b_eggs := 375 * 42 - 80
  let set_c_eggs := (560 / 2) * 50 + (560 / 2) * 32
  set_a_eggs + set_b_eggs + set_c_eggs

/-- Theorem stating that Emily collected 51830 eggs in total -/
theorem emily_egg_collection : total_eggs = 51830 := by
  sorry

end emily_egg_collection_l3344_334431


namespace modified_binomial_coefficient_integrality_l3344_334450

theorem modified_binomial_coefficient_integrality 
  (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) * (n.factorial) = 
    (k + 2 : ℤ) * m * (k.factorial) * ((n - k).factorial) := by
  sorry

end modified_binomial_coefficient_integrality_l3344_334450


namespace distance_from_origin_to_point_l3344_334407

-- Define the point
def point : ℝ × ℝ := (8, -15)

-- Theorem statement
theorem distance_from_origin_to_point :
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 17 := by
  sorry

end distance_from_origin_to_point_l3344_334407


namespace fraction_equality_l3344_334438

theorem fraction_equality : 
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) / 
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 := by
  sorry

end fraction_equality_l3344_334438


namespace tan_equality_periodic_l3344_334486

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = -30 := by
  sorry

end tan_equality_periodic_l3344_334486


namespace parallelogram_properties_l3344_334482

def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

theorem parallelogram_properties :
  let D : ℂ := 4 + 3 * Complex.I
  let diagonal_BD : ℂ := D - B
  (A + C = B + D) ∧ 
  (Complex.abs diagonal_BD = 3 * Real.sqrt 2) := by
  sorry

end parallelogram_properties_l3344_334482


namespace perpendicular_line_proof_l3344_334464

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (0, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → given_line x y → x * x + y * y = 0) ∧
  perpendicular_line point.1 point.2 :=
by sorry

end perpendicular_line_proof_l3344_334464


namespace smallest_n_is_34_l3344_334424

/-- Given a natural number n ≥ 16, this function represents the set {16, 17, ..., n} -/
def S (n : ℕ) : Set ℕ := {x | 16 ≤ x ∧ x ≤ n}

/-- This function checks if a sequence of 15 natural numbers satisfies the required conditions -/
def valid_sequence (n : ℕ) (a : Fin 15 → ℕ) : Prop :=
  (∀ i : Fin 15, a i ∈ S n) ∧
  (∀ i : Fin 15, (i.val + 1) ∣ a i) ∧
  (∀ i j : Fin 15, i ≠ j → a i ≠ a j)

/-- The main theorem stating that 34 is the smallest n satisfying the conditions -/
theorem smallest_n_is_34 :
  (∃ a : Fin 15 → ℕ, valid_sequence 34 a) ∧
  (∀ m : ℕ, m < 34 → ¬∃ a : Fin 15 → ℕ, valid_sequence m a) :=
sorry

end smallest_n_is_34_l3344_334424


namespace gcd_factorial_plus_two_l3344_334454

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_plus_two : 
  Nat.gcd (factorial 6 + 2) (factorial 8 + 2) = 2 := by
  sorry

end gcd_factorial_plus_two_l3344_334454


namespace female_guests_from_jays_family_l3344_334456

def total_guests : ℕ := 240
def female_percentage : ℚ := 60 / 100
def jays_family_percentage : ℚ := 50 / 100

theorem female_guests_from_jays_family :
  (total_guests : ℚ) * female_percentage * jays_family_percentage = 72 := by
  sorry

end female_guests_from_jays_family_l3344_334456


namespace inequality_proof_l3344_334466

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.sin (80 * π / 180))
  (hb : b = (1/2)⁻¹)
  (hc : c = Real.log 3 / Real.log (1/2)) :
  b > a ∧ a > c := by sorry

end inequality_proof_l3344_334466


namespace children_outnumber_parents_l3344_334461

/-- Represents a family unit in the apartment block -/
structure Family where
  parents : Nat
  boys : Nat
  girls : Nat

/-- Represents the apartment block -/
structure ApartmentBlock where
  families : List Family

/-- Every couple has at least one child -/
axiom at_least_one_child (f : Family) : f.boys + f.girls ≥ 1

/-- Every child has exactly two parents -/
axiom two_parents (f : Family) : f.parents = 2

/-- Every little boy has a sister -/
axiom boys_have_sisters (f : Family) : f.boys > 0 → f.girls > 0

/-- Among the children, there are more boys than girls -/
axiom more_boys_than_girls (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.boys)).sum > (ab.families.map (λ f => f.girls)).sum

/-- There are no grandparents living in the building -/
axiom no_grandparents (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.parents)).sum = 2 * ab.families.length

theorem children_outnumber_parents (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.boys + f.girls)).sum > (ab.families.map (λ f => f.parents)).sum :=
sorry

end children_outnumber_parents_l3344_334461


namespace bonus_remainder_l3344_334472

theorem bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 := by
  sorry

end bonus_remainder_l3344_334472


namespace no_solution_for_quadratic_congruence_l3344_334492

theorem no_solution_for_quadratic_congruence :
  ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
sorry

end no_solution_for_quadratic_congruence_l3344_334492


namespace average_candies_per_packet_l3344_334417

def candy_counts : List Nat := [5, 7, 9, 11, 13, 15]
def num_packets : Nat := 6

theorem average_candies_per_packet :
  (candy_counts.sum / num_packets : ℚ) = 10 := by sorry

end average_candies_per_packet_l3344_334417


namespace variance_scaling_l3344_334433

-- Define a set of data points
def DataSet : Type := List ℝ

-- Define the variance function
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define a function to multiply each data point by a scalar
def scaleData (data : DataSet) (scalar : ℝ) : DataSet :=
  data.map (· * scalar)

-- Theorem statement
theorem variance_scaling (data : DataSet) (s : ℝ) :
  variance data = s^2 → variance (scaleData data 2) = 4 * s^2 := by
  sorry

end variance_scaling_l3344_334433


namespace quadratic_equation_roots_l3344_334487

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 13*x₁ + 40 = 0) ∧
  (x₂^2 - 13*x₂ + 40 = 0) ∧
  x₁ = 5 ∧
  x₂ = 8 ∧
  x₁ > 0 ∧
  x₂ > 0 ∧
  x₂ > x₁ :=
by
  sorry

end quadratic_equation_roots_l3344_334487


namespace rope_and_well_l3344_334467

theorem rope_and_well (x y : ℝ) (h : (1/4) * x = y + 3) : (1/5) * x = y + 2 := by
  sorry

end rope_and_well_l3344_334467


namespace sum_first_five_multiples_of_twelve_l3344_334459

theorem sum_first_five_multiples_of_twelve : 
  (Finset.range 5).sum (fun i => 12 * (i + 1)) = 180 := by
  sorry

end sum_first_five_multiples_of_twelve_l3344_334459


namespace meaningful_expression_l3344_334409

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 2)) ↔ x > 2 :=
by sorry

end meaningful_expression_l3344_334409


namespace library_visitors_sunday_visitors_proof_l3344_334490

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_visitors (non_sunday_visitors : ℕ) (total_average : ℕ) : ℕ :=
  let total_days : ℕ := 30
  let sundays : ℕ := 5
  let non_sundays : ℕ := total_days - sundays
  let sunday_visitors : ℕ := 
    (total_average * total_days - non_sunday_visitors * non_sundays) / sundays
  sunday_visitors

/-- Proves that the average number of Sunday visitors is 510 given the conditions -/
theorem sunday_visitors_proof (h1 : library_visitors 240 285 = 510) : 
  library_visitors 240 285 = 510 := by
  sorry

end library_visitors_sunday_visitors_proof_l3344_334490


namespace mistaken_addition_correction_l3344_334484

theorem mistaken_addition_correction (x : ℤ) : x + 16 = 64 → x - 16 = 32 := by
  sorry

end mistaken_addition_correction_l3344_334484


namespace corveus_sleep_lack_l3344_334481

/-- The number of hours Corveus lacks sleep in a week -/
def sleep_lack_per_week (actual_sleep : ℕ) (recommended_sleep : ℕ) (days_in_week : ℕ) : ℕ :=
  (recommended_sleep - actual_sleep) * days_in_week

/-- Theorem stating that Corveus lacks 14 hours of sleep in a week -/
theorem corveus_sleep_lack :
  sleep_lack_per_week 4 6 7 = 14 := by
  sorry

end corveus_sleep_lack_l3344_334481


namespace true_propositions_l3344_334400

-- Define the propositions p and q
def p : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2)
def q : Prop := ∀ y : ℝ, y > 0 → ∃ x : ℝ, y = 3^x

-- Define the set of derived propositions
def derived_props : Set Prop := {p ∨ q, p ∧ q, ¬p, ¬q}

-- Define the set of true propositions
def true_props : Set Prop := {p ∨ q, ¬p}

-- Theorem statement
theorem true_propositions : 
  {prop ∈ derived_props | prop} = true_props := by sorry

end true_propositions_l3344_334400


namespace remainder_theorem_l3344_334449

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 100 * k - 1) : (n^2 - n + 4) % 100 = 6 := by
  sorry

end remainder_theorem_l3344_334449


namespace workers_count_l3344_334429

/-- Given a work that can be completed by some workers in 35 days,
    and adding 10 workers reduces the completion time by 10 days,
    prove that the original number of workers is 25. -/
theorem workers_count (work : ℕ) : ∃ (workers : ℕ), 
  (workers * 35 = (workers + 10) * 25) ∧ 
  workers = 25 := by
  sorry

end workers_count_l3344_334429


namespace sock_pairs_problem_l3344_334496

theorem sock_pairs_problem (n : ℕ) : 
  (2 * n * (2 * n - 1)) / 2 = 6 → n = 2 := by
  sorry

end sock_pairs_problem_l3344_334496


namespace intersection_probability_l3344_334479

/-- Given probabilities for events a and b, prove their intersection probability -/
theorem intersection_probability (a b : Set α) (p : Set α → ℝ) 
  (ha : p a = 0.18)
  (hb : p b = 0.5)
  (hba : p (b ∩ a) / p a = 0.2) :
  p (a ∩ b) = 0.036 := by
  sorry

end intersection_probability_l3344_334479


namespace female_students_count_l3344_334416

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 84 →
  female_average = 92 →
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 24 :=
by sorry

end female_students_count_l3344_334416


namespace travel_problem_solution_l3344_334402

/-- Represents the speeds and distance in the problem -/
structure TravelData where
  pedestrian_speed : ℝ
  cyclist_speed : ℝ
  rider_speed : ℝ
  distance_AB : ℝ

/-- The conditions of the problem -/
def problem_conditions (data : TravelData) : Prop :=
  data.cyclist_speed = 2 * data.pedestrian_speed ∧
  2 * data.cyclist_speed + 2 * data.rider_speed = data.distance_AB ∧
  2.8 * data.pedestrian_speed + 2.8 * data.rider_speed = data.distance_AB ∧
  2 * data.rider_speed = data.distance_AB / 2 - 3 ∧
  2 * data.cyclist_speed = data.distance_AB / 2 + 3

/-- The theorem to prove -/
theorem travel_problem_solution :
  ∃ (data : TravelData),
    problem_conditions data ∧
    data.pedestrian_speed = 6 ∧
    data.cyclist_speed = 12 ∧
    data.rider_speed = 9 ∧
    data.distance_AB = 42 :=
by
  sorry

end travel_problem_solution_l3344_334402


namespace rice_cost_difference_l3344_334439

/-- Represents the rice purchase and distribution scenario -/
structure RiceScenario where
  total_rice : ℝ
  price1 : ℝ
  price2 : ℝ
  price3 : ℝ
  quantity1 : ℝ
  quantity2 : ℝ
  quantity3 : ℝ
  kept_ratio : ℝ

/-- Calculates the cost difference between kept and given rice -/
def cost_difference (scenario : RiceScenario) : ℝ :=
  let total_cost := scenario.price1 * scenario.quantity1 + 
                    scenario.price2 * scenario.quantity2 + 
                    scenario.price3 * scenario.quantity3
  let kept_quantity := scenario.kept_ratio * scenario.total_rice
  let given_quantity := scenario.total_rice - kept_quantity
  let kept_cost := scenario.price1 * scenario.quantity1 + 
                   scenario.price2 * (kept_quantity - scenario.quantity1) + 
                   scenario.price3 * (kept_quantity - scenario.quantity1 - scenario.quantity2)
  let given_cost := total_cost - kept_cost
  kept_cost - given_cost

/-- The main theorem stating the cost difference for the given scenario -/
theorem rice_cost_difference : 
  let scenario : RiceScenario := {
    total_rice := 50,
    price1 := 1.2,
    price2 := 1.5,
    price3 := 2,
    quantity1 := 20,
    quantity2 := 25,
    quantity3 := 5,
    kept_ratio := 0.7
  }
  cost_difference scenario = 41.5 := by sorry


end rice_cost_difference_l3344_334439


namespace target_state_reachable_l3344_334475

/-- Represents the state of the urn -/
structure UrnState where
  black : ℕ
  white : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | replaceBlacks
  | replaceBlackWhite
  | replaceWhiteBlack
  | replaceWhites

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replaceBlacks => 
      if state.black ≥ 3 then UrnState.mk (state.black - 1) state.white
      else state
  | Operation.replaceBlackWhite => 
      if state.black ≥ 2 && state.white ≥ 1 then UrnState.mk (state.black - 2) (state.white + 1)
      else state
  | Operation.replaceWhiteBlack => 
      if state.black ≥ 1 && state.white ≥ 2 then UrnState.mk state.black (state.white - 1)
      else state
  | Operation.replaceWhites => 
      if state.white ≥ 3 then UrnState.mk (state.black + 1) (state.white - 3)
      else state

/-- Checks if the target state is reachable from the initial state -/
def isReachable (initial : UrnState) (target : UrnState) : Prop :=
  ∃ (sequence : List Operation), 
    List.foldl applyOperation initial sequence = target

/-- The main theorem stating that the target state is reachable -/
theorem target_state_reachable : 
  isReachable (UrnState.mk 80 120) (UrnState.mk 1 2) := by
  sorry

end target_state_reachable_l3344_334475


namespace product_equals_result_l3344_334455

theorem product_equals_result : 582964 * 99999 = 58295817036 := by
  sorry

end product_equals_result_l3344_334455


namespace min_offers_for_conviction_l3344_334415

/-- The minimum number of additional offers needed to be convinced with high probability. -/
def min_additional_offers : ℕ := 58

/-- The probability threshold for conviction. -/
def conviction_threshold : ℝ := 0.99

/-- The number of models already observed. -/
def observed_models : ℕ := 12

theorem min_offers_for_conviction :
  ∀ n : ℕ, n > observed_models →
    (observed_models : ℝ) / n ^ min_additional_offers < 1 - conviction_threshold :=
by sorry

end min_offers_for_conviction_l3344_334415


namespace max_acute_angles_in_hexagon_l3344_334432

/-- A convex hexagon is a polygon with 6 sides where all interior points are on the same side of any line through two vertices. -/
structure ConvexHexagon where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- An angle is acute if it is less than 90 degrees. -/
def is_acute (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

/-- The sum of interior angles of a hexagon is 720 degrees. -/
axiom hexagon_angle_sum (h : ConvexHexagon) : 
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ), a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720

/-- The theorem stating that the maximum number of acute angles in a convex hexagon is 3. -/
theorem max_acute_angles_in_hexagon (h : ConvexHexagon) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (is_acute a₁ ∧ is_acute a₂ ∧ is_acute a₃) ∧
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720 ∧
    ¬∃ (b₁ b₂ b₃ b₄ : ℝ),
      (is_acute b₁ ∧ is_acute b₂ ∧ is_acute b₃ ∧ is_acute b₄) ∧
      ∃ (b₅ b₆ : ℝ), b₁ + b₂ + b₃ + b₄ + b₅ + b₆ = 720 :=
by
  sorry


end max_acute_angles_in_hexagon_l3344_334432


namespace stamp_collection_problem_l3344_334499

theorem stamp_collection_problem (tom_initial : ℕ) (tom_final : ℕ) (harry_extra : ℕ) :
  tom_initial = 3000 →
  tom_final = 3061 →
  harry_extra = 10 →
  ∃ (mike : ℕ),
    mike = 17 ∧
    tom_final = tom_initial + mike + (2 * mike + harry_extra) :=
by sorry

end stamp_collection_problem_l3344_334499


namespace lucy_money_problem_l3344_334425

theorem lucy_money_problem (x : ℝ) : 
  let doubled := 2 * x
  let after_giving := doubled * (4/5)
  let after_losing := after_giving * (2/3)
  let after_spending := after_losing * (3/4)
  after_spending = 15 → x = 18.75 := by sorry

end lucy_money_problem_l3344_334425


namespace expression_evaluation_l3344_334498

theorem expression_evaluation (a b : ℝ) (h1 : a = 1) (h2 : b = -2) :
  (a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b) = -8 := by
  sorry

end expression_evaluation_l3344_334498


namespace gcd_power_of_two_minus_one_l3344_334457

theorem gcd_power_of_two_minus_one :
  Nat.gcd (2^2022 - 1) (2^2036 - 1) = 2^14 - 1 := by
  sorry

end gcd_power_of_two_minus_one_l3344_334457


namespace fence_cost_per_foot_l3344_334403

/-- The cost per foot of building a fence around a square plot -/
theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) : area = 289 → total_cost = 4080 → (total_cost / (4 * Real.sqrt area)) = 60 := by
  sorry

end fence_cost_per_foot_l3344_334403


namespace elisas_painting_l3344_334444

theorem elisas_painting (monday : ℝ) 
  (h1 : monday > 0)
  (h2 : monday + 2 * monday + monday / 2 = 105) : 
  monday = 30 := by
  sorry

end elisas_painting_l3344_334444


namespace auto_store_sales_time_l3344_334468

theorem auto_store_sales_time (total_cars : ℕ) (salespeople : ℕ) (cars_per_person : ℕ) :
  total_cars = 500 →
  salespeople = 10 →
  cars_per_person = 10 →
  (total_cars / (salespeople * cars_per_person) : ℚ) = 5 := by
  sorry

end auto_store_sales_time_l3344_334468


namespace square_side_length_l3344_334401

/-- Given an arrangement of rectangles and squares forming a larger rectangle,
    this theorem proves that the side length of square S2 is 900 units. -/
theorem square_side_length (r : ℕ) : 
  (2 * r + 900 = 2800) ∧ (2 * r + 3 * 900 = 4600) → 900 = 900 := by
  sorry

end square_side_length_l3344_334401


namespace factor_expression_l3344_334471

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) := by
  sorry

end factor_expression_l3344_334471


namespace quadratic_inequality_properties_l3344_334474

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_solution_set : ∀ x, f a b c x > 0 ↔ -1/2 < x ∧ x < 3) :
  c > 0 ∧ 4*a + 2*b + c > 0 :=
sorry

end quadratic_inequality_properties_l3344_334474


namespace sumata_vacation_miles_l3344_334421

/-- The total miles driven during a vacation -/
def total_miles (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

/-- Theorem: The Sumata family drove 1250 miles during their 5.0-day vacation -/
theorem sumata_vacation_miles :
  total_miles 5.0 250 = 1250 := by
  sorry

end sumata_vacation_miles_l3344_334421


namespace cookies_per_person_l3344_334473

/-- The number of cookie batches Beth bakes in a week -/
def batches : ℕ := 8

/-- The number of dozens of cookies in each batch -/
def dozens_per_batch : ℕ := 5

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of people sharing the cookies -/
def people : ℕ := 30

/-- Theorem: If 8 batches of 5 dozen cookies are shared equally among 30 people,
    each person will receive 16 cookies -/
theorem cookies_per_person :
  (batches * dozens_per_batch * cookies_per_dozen) / people = 16 := by
  sorry

end cookies_per_person_l3344_334473


namespace smallest_angle_3_4_5_triangle_l3344_334469

theorem smallest_angle_3_4_5_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a^2 + b^2 = c^2 →
    a / c = 3 / 5 ∧ b / c = 4 / 5 →
    min (Real.arctan (a / b)) (Real.arctan (b / a)) = Real.arctan (3 / 4) :=
by sorry

end smallest_angle_3_4_5_triangle_l3344_334469


namespace bracelet_selling_price_l3344_334477

theorem bracelet_selling_price 
  (total_bracelets : ℕ)
  (given_away : ℕ)
  (material_cost : ℚ)
  (profit : ℚ)
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : material_cost = 3)
  (h4 : profit = 8) :
  let sold_bracelets := total_bracelets - given_away
  let total_sales := profit + material_cost
  let price_per_bracelet := total_sales / sold_bracelets
  price_per_bracelet = 1/4 := by
sorry

end bracelet_selling_price_l3344_334477


namespace product_of_roots_l3344_334488

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 22 → ∃ y : ℝ, (x + 3) * (x - 5) = 22 ∧ (y + 3) * (y - 5) = 22 ∧ x * y = -37 := by
  sorry

end product_of_roots_l3344_334488


namespace equation_one_solution_equation_two_solution_l3344_334412

-- Equation 1
theorem equation_one_solution (x : ℝ) : 9 * x^2 = 27 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : -2 * (x - 3)^3 + 16 = 0 ↔ x = 5 := by sorry

end equation_one_solution_equation_two_solution_l3344_334412


namespace tank_height_is_16_l3344_334414

/-- The height of a cylindrical water tank with specific conditions -/
def tank_height : ℝ := 16

/-- The base radius of the cylindrical water tank -/
def base_radius : ℝ := 3

/-- Theorem stating that the height of the tank is 16 cm under given conditions -/
theorem tank_height_is_16 :
  tank_height = 16 ∧
  base_radius = 3 ∧
  (π * base_radius^2 * (tank_height / 2) = 2 * (4/3) * π * base_radius^3) :=
by sorry

end tank_height_is_16_l3344_334414


namespace complex_modulus_example_l3344_334470

theorem complex_modulus_example : 
  let z : ℂ := 1 - (5/4)*I
  Complex.abs z = Real.sqrt 41 / 4 := by sorry

end complex_modulus_example_l3344_334470
