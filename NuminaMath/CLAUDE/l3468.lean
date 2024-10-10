import Mathlib

namespace arithmetic_sequence_sum_l3468_346824

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d > 0 →  -- positive common difference
  a 1 + a 2 + a 3 = 15 →  -- first condition
  a 1 * a 2 * a 3 = 80 →  -- second condition
  a 11 + a 12 + a 13 = 105 :=  -- conclusion
by
  sorry

end arithmetic_sequence_sum_l3468_346824


namespace f_derivative_l3468_346809

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem f_derivative :
  deriv f = fun x => x * Real.cos x := by sorry

end f_derivative_l3468_346809


namespace conjugate_sum_product_l3468_346894

theorem conjugate_sum_product (c d : ℝ) : 
  ((c + Real.sqrt d) + (c - Real.sqrt d) = -6) →
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 4) →
  c + d = 2 := by
  sorry

end conjugate_sum_product_l3468_346894


namespace min_sum_given_product_l3468_346842

theorem min_sum_given_product (a b c : ℕ+) : a * b * c = 2310 → (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) → a + b + c = 42 := by
  sorry

end min_sum_given_product_l3468_346842


namespace expression_evaluation_l3468_346866

theorem expression_evaluation (b : ℝ) :
  let x : ℝ := b + 9
  2 * x - b + 5 = b + 23 := by sorry

end expression_evaluation_l3468_346866


namespace polynomial_calculation_l3468_346851

/-- A polynomial of degree 4 with specific properties -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- Theorem stating the result of the calculation -/
theorem polynomial_calculation (a b c d : ℝ) 
  (h1 : P a b c d 1 = 1993)
  (h2 : P a b c d 2 = 3986)
  (h3 : P a b c d 3 = 5979) :
  (1/4) * (P a b c d 11 + P a b c d (-7)) = 4693 := by
  sorry

end polynomial_calculation_l3468_346851


namespace complement_A_subset_B_l3468_346888

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Define set B
def B : Set ℝ := {y | y ≥ 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x ≥ 1}

theorem complement_A_subset_B : complementA ⊆ B := by
  sorry

end complement_A_subset_B_l3468_346888


namespace johns_sister_age_l3468_346823

/-- Given the ages of John, his dad, and his sister, prove that John's sister is 37.5 years old -/
theorem johns_sister_age :
  ∀ (john dad sister : ℝ),
  dad = john + 15 →
  john + dad = 100 →
  sister = john - 5 →
  sister = 37.5 := by
sorry

end johns_sister_age_l3468_346823


namespace coin_stack_arrangements_l3468_346865

/-- The number of distinguishable arrangements of coins -/
def coin_arrangements (gold : Nat) (silver : Nat) : Nat :=
  Nat.choose (gold + silver) gold * (gold + silver + 1)

/-- Theorem stating the number of distinguishable arrangements for the given problem -/
theorem coin_stack_arrangements :
  coin_arrangements 5 3 = 504 := by
  sorry

end coin_stack_arrangements_l3468_346865


namespace equation_solution_l3468_346873

theorem equation_solution : ∃ y : ℝ, 
  (y^2 - 3*y - 10)/(y + 2) + (4*y^2 + 17*y - 15)/(4*y - 1) = 5 ∧ y = -5/2 := by
  sorry

end equation_solution_l3468_346873


namespace integral_equals_half_l3468_346841

open Real MeasureTheory Interval

/-- The definite integral of 1 / (1 + sin x - cos x)^2 from 2 arctan(1/2) to π/2 equals 1/2 -/
theorem integral_equals_half :
  ∫ x in 2 * arctan (1/2)..π/2, 1 / (1 + sin x - cos x)^2 = 1/2 := by
  sorry

end integral_equals_half_l3468_346841


namespace rain_probability_l3468_346895

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by
  sorry

end rain_probability_l3468_346895


namespace min_distance_to_locus_l3468_346875

open Complex

theorem min_distance_to_locus (z : ℂ) :
  (abs (z - 1) = abs (z + 2*I)) →
  ∃ min_val : ℝ, (min_val = (9 * Real.sqrt 5) / 10) ∧
  (∀ w : ℂ, abs (z - 1) = abs (z + 2*I) → abs (w - 1 - I) ≥ min_val) ∧
  (∃ z₀ : ℂ, abs (z₀ - 1) = abs (z₀ + 2*I) ∧ abs (z₀ - 1 - I) = min_val) :=
by sorry

end min_distance_to_locus_l3468_346875


namespace machine_production_time_l3468_346853

/-- The number of items the machine can produce in one hour -/
def items_per_hour : ℕ := 90

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The time it takes to produce one item in minutes -/
def time_per_item : ℚ := minutes_per_hour / items_per_hour

theorem machine_production_time : time_per_item = 2/3 := by
  sorry

end machine_production_time_l3468_346853


namespace modular_arithmetic_problem_l3468_346857

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (7 * a) % 72 = 1 ∧ (13 * b) % 72 = 1 →
  (3 * a + 9 * b) % 72 = 18 := by
sorry

end modular_arithmetic_problem_l3468_346857


namespace rivet_distribution_l3468_346883

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℕ
  y : ℕ

/-- Checks if a point is inside a rectangle -/
def Point.insideRectangle (p : Point) (r : Rectangle) : Prop :=
  p.x < r.width ∧ p.y < r.height

/-- Checks if a point is on the grid lines of a rectangle divided into unit squares -/
def Point.onGridLines (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

/-- Theorem: In a 9x11 rectangle divided into unit squares, 
    with 200 points inside and not on grid lines, 
    there exists at least one unit square with 3 or more points -/
theorem rivet_distribution (points : List Point) : 
  points.length = 200 → 
  (∀ p ∈ points, p.insideRectangle ⟨9, 11⟩ ∧ ¬p.onGridLines) →
  ∃ (x y : ℕ), x < 9 ∧ y < 11 ∧ 
    (points.filter (λ p => p.x ≥ x ∧ p.x < x + 1 ∧ p.y ≥ y ∧ p.y < y + 1)).length ≥ 3 :=
by sorry

end rivet_distribution_l3468_346883


namespace library_shelves_l3468_346878

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
  sorry

end library_shelves_l3468_346878


namespace equation_solution_l3468_346892

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (9 + Real.sqrt (15 + 5*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 5 + Real.sqrt 15) ∧ x = -2 := by
  sorry

end equation_solution_l3468_346892


namespace coin_distribution_l3468_346874

theorem coin_distribution (a b c d e : ℚ) : 
  a + b + c + d + e = 5 →  -- total is 5 coins
  ∃ (x y : ℚ), (a = x - 2*y ∧ b = x - y ∧ c = x ∧ d = x + y ∧ e = x + 2*y) →  -- arithmetic sequence
  a + b = c + d + e →  -- sum of first two equals sum of last three
  b = 4/3 :=  -- second person receives 4/3 coins
by sorry

end coin_distribution_l3468_346874


namespace chocolate_chip_cookies_baked_l3468_346818

/-- The number of dozens of cookies Ann baked for each type -/
structure CookieBatch where
  oatmeal_raisin : ℚ
  sugar : ℚ
  chocolate_chip : ℚ

/-- The number of dozens of cookies Ann gave away for each type -/
structure CookiesGivenAway where
  oatmeal_raisin : ℚ
  sugar : ℚ
  chocolate_chip : ℚ

def cookies_kept (baked : CookieBatch) (given_away : CookiesGivenAway) : ℚ :=
  (baked.oatmeal_raisin - given_away.oatmeal_raisin +
   baked.sugar - given_away.sugar +
   baked.chocolate_chip - given_away.chocolate_chip) * 12

theorem chocolate_chip_cookies_baked 
  (baked : CookieBatch)
  (given_away : CookiesGivenAway)
  (h1 : baked.oatmeal_raisin = 3)
  (h2 : baked.sugar = 2)
  (h3 : given_away.oatmeal_raisin = 2)
  (h4 : given_away.sugar = 3/2)
  (h5 : given_away.chocolate_chip = 5/2)
  (h6 : cookies_kept baked given_away = 36) :
  baked.chocolate_chip = 4 := by
sorry

end chocolate_chip_cookies_baked_l3468_346818


namespace parabola_properties_l3468_346829

/-- Parabola passing through (-1, 0) -/
def parabola (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x - 3

theorem parabola_properties :
  ∃ (b : ℝ),
    (parabola b (-1) = 0) ∧
    (b = -4) ∧
    (∃ (h k : ℝ), h = -2 ∧ k = 1 ∧ ∀ x, parabola b x = -(x - h)^2 + k) ∧
    (∀ y₁ y₂ : ℝ, parabola b 1 = y₁ → parabola b (-1) = y₂ → y₁ < y₂) :=
by sorry

end parabola_properties_l3468_346829


namespace complex_magnitude_equality_l3468_346830

theorem complex_magnitude_equality (t : ℝ) (h1 : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 2 * Real.sqrt 17 → t = Real.sqrt 59 := by
  sorry

end complex_magnitude_equality_l3468_346830


namespace first_wave_infections_count_l3468_346803

/-- The number of infections per day during the first wave of coronavirus -/
def first_wave_infections : ℕ := 375

/-- The number of infections per day during the second wave of coronavirus -/
def second_wave_infections : ℕ := 4 * first_wave_infections

/-- The duration of the second wave in days -/
def second_wave_duration : ℕ := 14

/-- The total number of infections during the second wave -/
def total_second_wave_infections : ℕ := 21000

/-- Theorem stating that the number of infections per day during the first wave was 375 -/
theorem first_wave_infections_count : 
  first_wave_infections = 375 ∧ 
  second_wave_infections = 4 * first_wave_infections ∧
  total_second_wave_infections = second_wave_infections * second_wave_duration :=
sorry

end first_wave_infections_count_l3468_346803


namespace inequality_proof_l3468_346828

theorem inequality_proof (a b c : ℝ) 
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  2 * (a + b + c) * (a^2 + b^2 + c^2) / 3 > a^3 + b^3 + c^3 + a*b*c :=
by sorry

end inequality_proof_l3468_346828


namespace zoe_family_cost_l3468_346897

/-- The total cost of soda and pizza for a group --/
def total_cost (num_people : ℕ) (soda_cost pizza_cost : ℚ) : ℚ :=
  num_people * (soda_cost + pizza_cost)

/-- Theorem: The total cost for Zoe and her family is $9 --/
theorem zoe_family_cost : 
  total_cost 6 (1/2) 1 = 9 := by sorry

end zoe_family_cost_l3468_346897


namespace circle_divides_sides_l3468_346826

/-- An isosceles trapezoid with bases in ratio 3:2 and a circle on the larger base -/
structure IsoscelesTrapezoidWithCircle where
  /-- Length of the smaller base -/
  b : ℝ
  /-- Length of the larger base -/
  a : ℝ
  /-- The bases are in ratio 3:2 -/
  base_ratio : a = (3/2) * b
  /-- The trapezoid is isosceles -/
  isosceles : True
  /-- Radius of the circle (half of the larger base) -/
  r : ℝ
  circle_diameter : r = a / 2
  /-- Length of the segment cut off on the smaller base by the circle -/
  m : ℝ
  segment_half_base : m = b / 2

/-- The circle divides the non-parallel sides of the trapezoid in the ratio 1:2 -/
theorem circle_divides_sides (t : IsoscelesTrapezoidWithCircle) :
  ∃ (x y : ℝ), x + y = t.a - t.b ∧ x / y = 1 / 2 := by
  sorry

end circle_divides_sides_l3468_346826


namespace new_person_weight_is_102_l3468_346856

/-- The weight of a new person joining a group, given the initial group size,
    average weight increase, and weight of the person being replaced. -/
def new_person_weight (initial_group_size : ℕ) (avg_weight_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + initial_group_size * avg_weight_increase

/-- Theorem stating that the weight of the new person is 102 kg -/
theorem new_person_weight_is_102 :
  new_person_weight 6 4.5 75 = 102 := by
  sorry

end new_person_weight_is_102_l3468_346856


namespace betty_sugar_purchase_l3468_346846

theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s ≥ 4 := by
sorry

end betty_sugar_purchase_l3468_346846


namespace problem_statement_l3468_346868

theorem problem_statement : (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 := by
  sorry

end problem_statement_l3468_346868


namespace new_boarders_count_l3468_346854

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ) (final_ratio_boarders : ℕ) (final_ratio_day : ℕ) :
  initial_boarders = 220 →
  initial_ratio_boarders = 5 →
  initial_ratio_day = 12 →
  final_ratio_boarders = 1 →
  final_ratio_day = 2 →
  ∃ (new_boarders : ℕ),
    new_boarders = 44 ∧
    (initial_boarders + new_boarders) * final_ratio_day = initial_boarders * initial_ratio_day * final_ratio_boarders :=
by sorry


end new_boarders_count_l3468_346854


namespace average_problem_l3468_346839

theorem average_problem (x : ℝ) : (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end average_problem_l3468_346839


namespace point_y_value_l3468_346848

/-- An angle with vertex at the origin and initial side on the non-negative x-axis -/
structure AngleAtOrigin where
  α : ℝ
  initial_side_on_x_axis : 0 ≤ α ∧ α < 2 * Real.pi

/-- A point on the terminal side of an angle -/
structure PointOnTerminalSide (angle : AngleAtOrigin) where
  x : ℝ
  y : ℝ
  on_terminal_side : x = 6 ∧ y = 6 * Real.tan angle.α

/-- Theorem: For an angle α with sin α = -4/5, if P(6, y) is on its terminal side, then y = -8 -/
theorem point_y_value (angle : AngleAtOrigin) 
  (h_sin : Real.sin angle.α = -4/5) 
  (point : PointOnTerminalSide angle) : 
  point.y = -8 := by
  sorry

end point_y_value_l3468_346848


namespace no_positive_integer_solutions_l3468_346833

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ+), x^2 + 2*y^2 = 2*x^3 - x := by sorry

end no_positive_integer_solutions_l3468_346833


namespace expression_evaluation_l3468_346886

/-- Proves that the given expression evaluates to -5 when x = -2 and y = -1 -/
theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -1) :
  2 * (x + y) * (-x - y) - (2 * x + y) * (-2 * x + y) = -5 := by
  sorry

end expression_evaluation_l3468_346886


namespace inequality_range_l3468_346816

-- Define the inequality function
def f (x a : ℝ) : Prop := x^2 + a*x > 4*x + a - 3

-- State the theorem
theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x, f x a ↔ x < -1 ∨ x > 3 :=
sorry

end inequality_range_l3468_346816


namespace smallest_dual_palindrome_l3468_346801

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a number to its representation in a given base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- The length of a number's representation in a given base -/
def digitCount (n : ℕ) (base : ℕ) : ℕ :=
  sorry

theorem smallest_dual_palindrome :
  ∀ m : ℕ, m < 17 →
    ¬(isPalindrome m 2 ∧ digitCount m 2 = 5 ∧
      isPalindrome m 3 ∧ digitCount m 3 = 3) →
  isPalindrome 17 2 ∧ digitCount 17 2 = 5 ∧
  isPalindrome 17 3 ∧ digitCount 17 3 = 3 :=
by sorry

end smallest_dual_palindrome_l3468_346801


namespace min_additional_packs_needed_l3468_346876

/-- The number of sticker packs in each basket -/
def packsPerBasket : ℕ := 7

/-- The current number of sticker packs Matilda has -/
def currentPacks : ℕ := 40

/-- The minimum number of additional packs needed -/
def minAdditionalPacks : ℕ := 2

/-- Theorem stating the minimum number of additional packs needed -/
theorem min_additional_packs_needed : 
  ∃ (totalPacks : ℕ), 
    totalPacks = currentPacks + minAdditionalPacks ∧ 
    totalPacks % packsPerBasket = 0 ∧
    ∀ (k : ℕ), k < minAdditionalPacks → 
      (currentPacks + k) % packsPerBasket ≠ 0 :=
sorry

end min_additional_packs_needed_l3468_346876


namespace smallest_number_with_remainders_l3468_346890

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = k - 1) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 12 ∧ m % k ≠ k - 1) ∧
  n = 27719 :=
by sorry

end smallest_number_with_remainders_l3468_346890


namespace smallest_n_with_odd_digits_l3468_346813

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d, d ∈ (97 * n).digits 10 → d % 2 = 1

theorem smallest_n_with_odd_digits :
  ∀ n : ℕ, n > 1 →
    (all_digits_odd n → n ≥ 35) ∧
    (all_digits_odd 35) :=
sorry

end smallest_n_with_odd_digits_l3468_346813


namespace log_inequality_l3468_346867

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (Real.log b) / (a - 1) = (a + 1) / a) : 
  Real.log b / Real.log a > 2 := by
sorry

end log_inequality_l3468_346867


namespace algebraic_expression_value_l3468_346849

theorem algebraic_expression_value : 
  let x : ℝ := Real.sqrt 3 + 2
  (x^2 - 4*x + 3) = 2 := by sorry

end algebraic_expression_value_l3468_346849


namespace min_value_reciprocal_sum_l3468_346884

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 3) :
  (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end min_value_reciprocal_sum_l3468_346884


namespace original_number_l3468_346858

theorem original_number : ∃ x : ℤ, 63 - 2 * x = 51 ∧ x = 6 := by
  sorry

end original_number_l3468_346858


namespace equivalence_of_statements_l3468_346827

theorem equivalence_of_statements (p q : Prop) :
  (¬p ∧ ¬q → p ∨ q) ↔ (p ∧ ¬q ∨ ¬p ∧ q) := by sorry

end equivalence_of_statements_l3468_346827


namespace roadwork_problem_l3468_346845

/-- Roadwork problem statement -/
theorem roadwork_problem (total_length pitch_day3 : ℝ) (h1 h2 h3 : ℕ) : 
  total_length = 16 ∧ 
  pitch_day3 = 6 ∧ 
  h1 = 2 ∧ 
  h2 = 5 ∧ 
  h3 = 3 → 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < total_length ∧ 
    (2 * x - 1) > 0 ∧ 
    (2 * x - 1) < total_length ∧
    3 * x - 1 = total_length - (pitch_day3 / h2) / h3 ∧ 
    x = 5 := by
  sorry

end roadwork_problem_l3468_346845


namespace price_increase_and_discount_l3468_346847

theorem price_increase_and_discount (original_price : ℝ) (increase_percentage : ℝ) :
  original_price * (1 + increase_percentage) * (1 - 0.2) = original_price →
  increase_percentage = 0.25 := by
  sorry

end price_increase_and_discount_l3468_346847


namespace circle_origin_outside_l3468_346889

theorem circle_origin_outside (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (x^2 + y^2 > 0)) → 
  (0 < m ∧ m < 1/2) := by
  sorry

end circle_origin_outside_l3468_346889


namespace circle_equation_from_diameter_endpoints_l3468_346821

/-- Given two points A and B as endpoints of a diameter of a circle,
    this theorem proves the equation of the circle. -/
theorem circle_equation_from_diameter_endpoints 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 4)) 
  (h_B : B = (3, -2)) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    r^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 ∧
    ∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 2)^2 + (y - 1)^2 = 10 :=
sorry

end circle_equation_from_diameter_endpoints_l3468_346821


namespace intersection_triangle_area_l3468_346879

-- Define the line L: x - 2y - 5 = 0
def L (x y : ℝ) : Prop := x - 2*y - 5 = 0

-- Define the circle C: x^2 + y^2 = 50
def C (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the intersection points
def A : ℝ × ℝ := (-5, -5)
def B : ℝ × ℝ := (7, 1)

-- Theorem statement
theorem intersection_triangle_area :
  L A.1 A.2 ∧ L B.1 B.2 ∧ C A.1 A.2 ∧ C B.1 B.2 →
  abs ((A.1 * B.2 - B.1 * A.2) / 2) = 15 :=
by sorry

end intersection_triangle_area_l3468_346879


namespace SetA_eq_SetB_l3468_346805

/-- Set A: integers representable as x^2 + 2y^2 where x and y are integers -/
def SetA : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 2*y^2}

/-- Set B: integers representable as x^2 + 6xy + 11y^2 where x and y are integers -/
def SetB : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 6*x*y + 11*y^2}

/-- Theorem stating that Set A and Set B are equal -/
theorem SetA_eq_SetB : SetA = SetB := by sorry

end SetA_eq_SetB_l3468_346805


namespace circle_radius_l3468_346819

/-- Given a circle with area M cm² and circumference N cm,
    where M/N = 15 and the area is 60π cm²,
    prove that the radius of the circle is 2√15 cm. -/
theorem circle_radius (M N : ℝ) (h1 : M / N = 15) (h2 : M = 60 * Real.pi) :
  ∃ (r : ℝ), r = 2 * Real.sqrt 15 ∧ M = Real.pi * r^2 ∧ N = 2 * Real.pi * r :=
sorry

end circle_radius_l3468_346819


namespace student_count_problem_l3468_346872

theorem student_count_problem :
  ∃! (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    22 * (a + b + c) = 2 * (100 * a + 10 * b + c) ∧
    100 * a + 10 * b + c = 198 :=
by sorry

end student_count_problem_l3468_346872


namespace rectangle_y_value_l3468_346862

/-- A rectangle with vertices at (-3, y), (5, y), (-3, -2), and (5, -2) has an area of 96 square units. -/
def rectangle_area (y : ℝ) : Prop :=
  (5 - (-3)) * (y - (-2)) = 96

/-- The theorem states that if y is negative and satisfies the rectangle_area condition, then y = -14. -/
theorem rectangle_y_value (y : ℝ) (h1 : y < 0) (h2 : rectangle_area y) : y = -14 := by
  sorry

end rectangle_y_value_l3468_346862


namespace treasure_trap_probability_l3468_346859

/-- The number of islands --/
def num_islands : ℕ := 5

/-- The probability of an island having treasure and no traps --/
def p_treasure : ℚ := 1/5

/-- The probability of an island having traps but no treasure --/
def p_traps : ℚ := 1/5

/-- The probability of an island having neither traps nor treasure --/
def p_neither : ℚ := 3/5

/-- The number of islands with treasure --/
def treasure_islands : ℕ := 2

/-- The number of islands with traps --/
def trap_islands : ℕ := 2

/-- Theorem stating the probability of encountering exactly 2 islands with treasure and 2 with traps --/
theorem treasure_trap_probability : 
  (Nat.choose num_islands treasure_islands) * 
  (Nat.choose (num_islands - treasure_islands) trap_islands) * 
  (p_treasure ^ treasure_islands) * 
  (p_traps ^ trap_islands) * 
  (p_neither ^ (num_islands - treasure_islands - trap_islands)) = 18/625 := by
sorry

end treasure_trap_probability_l3468_346859


namespace tomato_price_per_pound_l3468_346843

/-- Calculates the price per pound of tomatoes in Scott's ratatouille recipe --/
theorem tomato_price_per_pound
  (eggplant_weight : ℝ) (eggplant_price : ℝ)
  (zucchini_weight : ℝ) (zucchini_price : ℝ)
  (tomato_weight : ℝ)
  (onion_weight : ℝ) (onion_price : ℝ)
  (basil_weight : ℝ) (basil_price : ℝ)
  (yield_quarts : ℝ) (price_per_quart : ℝ)
  (h1 : eggplant_weight = 5)
  (h2 : eggplant_price = 2)
  (h3 : zucchini_weight = 4)
  (h4 : zucchini_price = 2)
  (h5 : tomato_weight = 4)
  (h6 : onion_weight = 3)
  (h7 : onion_price = 1)
  (h8 : basil_weight = 1)
  (h9 : basil_price = 2.5 * 2)  -- $2.50 per half pound, so double for 1 pound
  (h10 : yield_quarts = 4)
  (h11 : price_per_quart = 10) :
  (yield_quarts * price_per_quart - 
   (eggplant_weight * eggplant_price + 
    zucchini_weight * zucchini_price + 
    onion_weight * onion_price + 
    basil_weight * basil_price)) / tomato_weight = 3.5 := by
  sorry


end tomato_price_per_pound_l3468_346843


namespace distance_ratio_on_rough_terrain_l3468_346814

theorem distance_ratio_on_rough_terrain
  (total_distance : ℝ)
  (speed_ratio : ℝ → ℝ → Prop)
  (rough_terrain_speed : ℝ → ℝ)
  (rough_terrain_length : ℝ)
  (meeting_point : ℝ)
  (h1 : speed_ratio 2 3)
  (h2 : ∀ x, rough_terrain_speed x = x / 2)
  (h3 : rough_terrain_length = 2 / 3 * total_distance)
  (h4 : meeting_point = total_distance / 2) :
  ∃ (d1 d2 : ℝ), d1 + d2 = rough_terrain_length ∧ d1 / d2 = 1 / 3 := by
sorry

end distance_ratio_on_rough_terrain_l3468_346814


namespace geometric_sequence_common_ratio_l3468_346812

/-- Given a geometric sequence with positive terms and a specific arithmetic sequence condition, prove the common ratio. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_positive : ∀ n, a n > 0)
  (h_arith : a 1 + 2 * a 2 = a 3) : 
  q = 1 + Real.sqrt 2 := by
sorry

end geometric_sequence_common_ratio_l3468_346812


namespace set_intersection_complement_l3468_346800

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := {1, 4}

def N : Set ℕ := {1, 3, 5}

theorem set_intersection_complement :
  N ∩ (U \ M) = {3, 5} := by sorry

end set_intersection_complement_l3468_346800


namespace exercise_books_quantity_l3468_346869

/-- Given a ratio of items and the quantity of one item, calculate the quantity of another item in the ratio. -/
def calculate_quantity (ratio_a : ℕ) (ratio_b : ℕ) (quantity_a : ℕ) : ℕ :=
  (quantity_a * ratio_b) / ratio_a

/-- Prove that given 140 pencils and a ratio of 14 : 4 : 3 for pencils : pens : exercise books, 
    the number of exercise books is 30. -/
theorem exercise_books_quantity (pencils : ℕ) (ratio_pencils ratio_pens ratio_books : ℕ) 
    (h1 : pencils = 140)
    (h2 : ratio_pencils = 14)
    (h3 : ratio_pens = 4)
    (h4 : ratio_books = 3) :
  calculate_quantity ratio_pencils ratio_books pencils = 30 := by
  sorry

#eval calculate_quantity 14 3 140

end exercise_books_quantity_l3468_346869


namespace rectangle_width_l3468_346840

theorem rectangle_width (width : ℝ) (h1 : width > 0) : 
  (2 * width) * width = 50 → width = 5 := by
  sorry

end rectangle_width_l3468_346840


namespace one_alligator_per_week_l3468_346836

/-- The number of Burmese pythons -/
def num_pythons : ℕ := 5

/-- The number of alligators eaten in the given time period -/
def num_alligators : ℕ := 15

/-- The number of weeks in the given time period -/
def num_weeks : ℕ := 3

/-- The number of alligators one Burmese python can eat per week -/
def alligators_per_python_per_week : ℚ := num_alligators / (num_pythons * num_weeks)

theorem one_alligator_per_week : 
  alligators_per_python_per_week = 1 :=
sorry

end one_alligator_per_week_l3468_346836


namespace avery_build_time_l3468_346832

theorem avery_build_time (tom_time : ℝ) (joint_work_time : ℝ) (tom_remaining_time : ℝ)
  (h1 : tom_time = 2.5)
  (h2 : joint_work_time = 1)
  (h3 : tom_remaining_time = 2/3) :
  ∃ avery_time : ℝ, 
    (1 / avery_time + 1 / tom_time) * joint_work_time + 
    (1 / tom_time) * tom_remaining_time = 1 ∧ 
    avery_time = 3 := by
sorry

end avery_build_time_l3468_346832


namespace alice_number_problem_l3468_346882

theorem alice_number_problem (x : ℝ) : ((x + 3) * 3 - 5) / 3 = 10 → x = 26 / 3 := by
  sorry

end alice_number_problem_l3468_346882


namespace ratio_a_to_c_l3468_346811

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 2 / 1)
  (hdb : d / b = 2 / 5) :
  a / c = 25 / 8 := by
  sorry

end ratio_a_to_c_l3468_346811


namespace siblings_age_multiple_l3468_346835

theorem siblings_age_multiple (kay_age : ℕ) (oldest_age : ℕ) (num_siblings : ℕ) : 
  kay_age = 32 →
  oldest_age = 44 →
  num_siblings = 14 →
  ∃ (youngest_age : ℕ), 
    youngest_age = kay_age / 2 - 5 ∧
    oldest_age / youngest_age = 4 := by
  sorry

end siblings_age_multiple_l3468_346835


namespace total_apples_proof_l3468_346891

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73
def benny_apples : ℕ := 48
def lucy_sales : ℕ := 15

theorem total_apples_proof :
  pinky_apples + danny_apples + benny_apples = 157 :=
by sorry

end total_apples_proof_l3468_346891


namespace fortune_telling_app_probability_l3468_346808

theorem fortune_telling_app_probability :
  let n : ℕ := 7  -- Total number of trials
  let k : ℕ := 3  -- Number of successful trials
  let p : ℚ := 1/3  -- Probability of success in each trial
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
  sorry

end fortune_telling_app_probability_l3468_346808


namespace expression_simplification_l3468_346880

theorem expression_simplification (x y : ℚ) (hx : x = 1/8) (hy : y = -4) :
  ((x * y - 2) * (x * y + 2) - 2 * x^2 * y^2 + 4) / (-x * y) = -1/2 := by
  sorry

end expression_simplification_l3468_346880


namespace inverse_proposition_l3468_346834

theorem inverse_proposition :
  (∀ a : ℝ, a > 0 → a > 1) →
  (∀ a : ℝ, a > 1 → a > 0) :=
by sorry

end inverse_proposition_l3468_346834


namespace range_of_m_specific_m_value_l3468_346898

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m - 1

-- Define the condition for two real roots
def has_two_real_roots (m : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0

-- Theorem for the range of m
theorem range_of_m (m : ℝ) (h : has_two_real_roots m) : m ≤ 2 := by sorry

-- Theorem for the specific value of m
theorem specific_m_value (m : ℝ) (h : has_two_real_roots m) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0 ∧ x₁^2 + x₂^2 = 6*x₁*x₂) →
  m = 3/2 := by sorry

end range_of_m_specific_m_value_l3468_346898


namespace total_balls_count_l3468_346861

/-- The number of balls owned by Jungkook -/
def jungkook_balls : ℕ := 3

/-- The number of balls owned by Yoongi -/
def yoongi_balls : ℕ := 2

/-- The total number of balls owned by Jungkook and Yoongi -/
def total_balls : ℕ := jungkook_balls + yoongi_balls

theorem total_balls_count : total_balls = 5 := by
  sorry

end total_balls_count_l3468_346861


namespace curve_is_circle_l3468_346820

theorem curve_is_circle (θ : Real) (r : Real → Real) :
  (∀ θ, r θ = 1 / (1 - Real.sin θ)) →
  ∃ (x y : Real → Real), ∀ θ,
    x θ ^ 2 + (y θ - 1) ^ 2 = 1 :=
by sorry

end curve_is_circle_l3468_346820


namespace seven_thousand_six_hundred_scientific_notation_l3468_346871

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem seven_thousand_six_hundred_scientific_notation :
  toScientificNotation 7600 = ScientificNotation.mk 7.6 3 sorry sorry :=
sorry

end seven_thousand_six_hundred_scientific_notation_l3468_346871


namespace divisibility_condition_l3468_346863

theorem divisibility_condition (a b : ℤ) (ha : a ≥ 3) (hb : b ≥ 3) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k :=
sorry

end divisibility_condition_l3468_346863


namespace rectangle_configuration_exists_l3468_346864

/-- Represents a rectangle with vertical and horizontal sides -/
structure Rectangle where
  x : ℝ × ℝ  -- x-coordinates of left and right sides
  y : ℝ × ℝ  -- y-coordinates of bottom and top sides

/-- Checks if two rectangles meet (have at least one point in common) -/
def rectangles_meet (r1 r2 : Rectangle) : Prop :=
  (r1.x.1 ≤ r2.x.2 ∧ r2.x.1 ≤ r1.x.2) ∧ (r1.y.1 ≤ r2.y.2 ∧ r2.y.1 ≤ r1.y.2)

/-- Checks if two rectangles follow each other based on their indices -/
def rectangles_follow (i j n : ℕ) : Prop :=
  i % n = (j + 1) % n ∨ j % n = (i + 1) % n

/-- Represents a valid configuration of n rectangles -/
def valid_configuration (n : ℕ) (rectangles : Fin n → Rectangle) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    rectangles_meet (rectangles i) (rectangles j) ↔ ¬rectangles_follow i.val j.val n

/-- The main theorem stating that a valid configuration exists if and only if n ≤ 5 -/
theorem rectangle_configuration_exists (n : ℕ) (h : n ≥ 1) :
  (∃ rectangles : Fin n → Rectangle, valid_configuration n rectangles) ↔ n ≤ 5 :=
sorry

end rectangle_configuration_exists_l3468_346864


namespace consecutive_integers_cube_sum_l3468_346870

theorem consecutive_integers_cube_sum (a b c d : ℕ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 9340) →
  (a^3 + b^3 + c^3 + d^3 = 457064) :=
by sorry

end consecutive_integers_cube_sum_l3468_346870


namespace fib_50_div_5_l3468_346817

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The 50th Fibonacci number is divisible by 5 -/
theorem fib_50_div_5 : 5 ∣ fib 50 := by sorry

end fib_50_div_5_l3468_346817


namespace three_heads_in_eight_tosses_l3468_346837

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def coinTossProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 3 heads when tossing a fair coin 8 times is 7/32 -/
theorem three_heads_in_eight_tosses :
  coinTossProbability 8 3 = 7 / 32 := by
  sorry

end three_heads_in_eight_tosses_l3468_346837


namespace distinct_x_intercepts_l3468_346838

/-- The number of distinct real solutions to the equation (x-5)(x^2 - x - 6) = 0 -/
def num_solutions : ℕ := 3

/-- The equation representing the x-intercepts of the graph -/
def equation (x : ℝ) : ℝ := (x - 5) * (x^2 - x - 6)

theorem distinct_x_intercepts :
  ∃ (s : Finset ℝ), (∀ x ∈ s, equation x = 0) ∧ s.card = num_solutions :=
sorry

end distinct_x_intercepts_l3468_346838


namespace dropped_student_score_l3468_346822

theorem dropped_student_score (initial_students : ℕ) (remaining_students : ℕ) 
  (initial_average : ℚ) (new_average : ℚ) : 
  initial_students = 16 →
  remaining_students = 15 →
  initial_average = 60.5 →
  new_average = 64 →
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 8 := by
  sorry

end dropped_student_score_l3468_346822


namespace ellipse_equation_l3468_346844

/-- The equation of an ellipse with specific properties -/
theorem ellipse_equation (a b c : ℝ) (h1 : a + b = 10) (h2 : 2 * c = 4 * Real.sqrt 5) 
  (h3 : a^2 = c^2 + b^2) (h4 : a > b) (h5 : b > 0) :
  ∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1 :=
sorry

end ellipse_equation_l3468_346844


namespace car_numbers_proof_l3468_346899

theorem car_numbers_proof :
  ∃! (x y : ℕ), 
    100 ≤ x ∧ x ≤ 999 ∧
    100 ≤ y ∧ y ≤ 999 ∧
    (∃ (a b c d : ℕ), x = 100 * a + 10 * b + 3 ∧ (c = 3 ∨ d = 3)) ∧
    (∃ (a b c d : ℕ), y = 100 * a + 10 * b + 3 ∧ (c = 3 ∨ d = 3)) ∧
    119 * x + 179 * y = 105080 ∧
    x = 337 ∧ y = 363 := by
  sorry

end car_numbers_proof_l3468_346899


namespace triangle_square_perimeter_ratio_l3468_346806

theorem triangle_square_perimeter_ratio : 
  let square_side : ℝ := 4
  let square_perimeter : ℝ := 4 * square_side
  let triangle_leg : ℝ := square_side
  let triangle_hypotenuse : ℝ := square_side * Real.sqrt 2
  let triangle_perimeter : ℝ := 2 * triangle_leg + triangle_hypotenuse
  triangle_perimeter / square_perimeter = 1/2 + Real.sqrt 2 / 4 := by
sorry

end triangle_square_perimeter_ratio_l3468_346806


namespace expression_equals_one_l3468_346810

theorem expression_equals_one :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90-18)*(90+18)) / ((120-9)*(120+9)) = 1 := by
  sorry

end expression_equals_one_l3468_346810


namespace exponential_equation_solution_l3468_346860

theorem exponential_equation_solution :
  ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^6 ∧ x = 3 := by
  sorry

end exponential_equation_solution_l3468_346860


namespace principal_amount_calculation_l3468_346802

/-- Proves that given a principal amount P put at simple interest for 2 years,
    if an increase of 4% in the interest rate results in Rs. 60 more interest,
    then P = 750. -/
theorem principal_amount_calculation (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 4) * 2) / 100 = (P * R * 2) / 100 + 60 → P = 750 := by
  sorry

end principal_amount_calculation_l3468_346802


namespace geometric_means_equality_l3468_346815

/-- Represents a quadrilateral with sides a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents the geometric means of sides in the quadrilateral -/
structure GeometricMeans (q : Quadrilateral) where
  k : ℝ
  l : ℝ
  m : ℝ
  n : ℝ
  hk : k^2 = q.a * q.d
  hl : l^2 = q.a * q.d
  hm : m^2 = q.b * q.c
  hn : n^2 = q.b * q.c

/-- The main theorem stating the condition for KL = MN -/
theorem geometric_means_equality (q : Quadrilateral) (g : GeometricMeans q) :
  (g.k - g.l)^2 = (g.m - g.n)^2 ↔ (q.a + q.b = q.c + q.d ∨ q.a + q.c = q.b + q.d) :=
by sorry


end geometric_means_equality_l3468_346815


namespace hyperbola_equation_l3468_346881

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  asymptote_eq : ∀ (x y : ℝ), x = 2 * y ∨ x = -2 * y
  point_on_curve : (4 : ℝ)^2 / a^2 - 1^2 / b^2 = 1

/-- The specific equation of the hyperbola -/
def specific_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 12 - y^2 / 3 = 1

/-- Theorem stating that the specific equation holds for the given hyperbola -/
theorem hyperbola_equation (h : Hyperbola) :
  ∀ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ↔ specific_equation h x y :=
sorry

end hyperbola_equation_l3468_346881


namespace intersection_when_a_is_one_intersection_contains_one_integer_l3468_346855

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0}

-- Theorem for part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 1 < x ∧ x ≤ 1 + Real.sqrt 2} := by sorry

-- Theorem for part II
theorem intersection_contains_one_integer (a : ℝ) :
  (∃! (n : ℤ), (n : ℝ) ∈ A ∩ B a) ↔ 3/4 ≤ a ∧ a < 4/3 := by sorry

end intersection_when_a_is_one_intersection_contains_one_integer_l3468_346855


namespace lottery_probability_maximum_l3468_346831

/-- The probability of winning in one draw -/
def p₀ (n : ℕ) : ℚ := (10 * n) / ((n + 5) * (n + 4))

/-- The probability of exactly one win in three draws -/
def p (n : ℕ) : ℚ := 3 * p₀ n * (1 - p₀ n)^2

/-- The statement to prove -/
theorem lottery_probability_maximum (n : ℕ) (h : n > 1) :
  ∃ (max_n : ℕ) (max_p : ℚ),
    max_n > 1 ∧
    max_p = p max_n ∧
    ∀ m, m > 1 → p m ≤ max_p ∧
    max_n = 20 ∧
    max_p = 4/9 := by
  sorry

end lottery_probability_maximum_l3468_346831


namespace no_integer_solutions_l3468_346896

theorem no_integer_solutions :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
sorry

end no_integer_solutions_l3468_346896


namespace xyz_value_l3468_346893

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 192 := by
sorry

end xyz_value_l3468_346893


namespace tournament_theorem_l3468_346850

/-- A tournament is a complete directed graph -/
structure Tournament (n : ℕ) where
  edges : Fin n → Fin n → Bool
  complete : ∀ i j, i ≠ j → edges i j ≠ edges j i
  no_self_edges : ∀ i, edges i i = false

/-- A set of edges in a tournament -/
def EdgeSet (n : ℕ) := Fin n → Fin n → Bool

/-- Reverse the orientation of edges in the given set -/
def reverseEdges (T : Tournament n) (S : EdgeSet n) : Tournament n where
  edges i j := if S i j then ¬(T.edges i j) else T.edges i j
  complete := sorry
  no_self_edges := sorry

/-- A graph contains a cycle -/
def hasCycle (T : Tournament n) : Prop := sorry

/-- A graph is acyclic -/
def isAcyclic (T : Tournament n) : Prop := ¬(hasCycle T)

/-- The number of edges in an edge set -/
def edgeCount (S : EdgeSet n) : ℕ := sorry

theorem tournament_theorem (n : ℕ) (h : n = 8) :
  (∃ T : Tournament n, ∀ S : EdgeSet n, edgeCount S ≤ 7 → hasCycle (reverseEdges T S)) ∧
  (∀ T : Tournament n, ∃ S : EdgeSet n, edgeCount S ≤ 8 ∧ isAcyclic (reverseEdges T S)) :=
sorry

end tournament_theorem_l3468_346850


namespace zero_of_f_l3468_346825

-- Define the function f
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = -2 := by sorry

end zero_of_f_l3468_346825


namespace theater_ticket_sales_l3468_346885

/-- Proves that the number of child tickets sold is 63 given the theater conditions --/
theorem theater_ticket_sales (total_seats : ℕ) (adult_price child_price : ℕ) (total_revenue : ℕ) 
  (h1 : total_seats = 80)
  (h2 : adult_price = 12)
  (h3 : child_price = 5)
  (h4 : total_revenue = 519) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_seats ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 63 := by
  sorry

end theater_ticket_sales_l3468_346885


namespace factorize_x4_plus_81_l3468_346807

theorem factorize_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end factorize_x4_plus_81_l3468_346807


namespace fraction_and_percentage_l3468_346887

theorem fraction_and_percentage (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 :=
by
  sorry

end fraction_and_percentage_l3468_346887


namespace unique_integer_solution_l3468_346804

theorem unique_integer_solution : ∃! x : ℕ+, (4 * x)^2 - x = 2100 :=
by
  -- Proof goes here
  sorry

end unique_integer_solution_l3468_346804


namespace existence_of_prime_and_power_l3468_346877

/-- The distance from a real number to its nearest integer -/
noncomputable def dist_to_nearest_int (x : ℝ) : ℝ :=
  |x - round x|

/-- The statement of the theorem -/
theorem existence_of_prime_and_power (a b : ℕ+) :
  ∃ (p : ℕ) (k : ℕ), Prime p ∧ p % 2 = 1 ∧
    dist_to_nearest_int (a / p^k : ℝ) +
    dist_to_nearest_int (b / p^k : ℝ) +
    dist_to_nearest_int ((a + b) / p^k : ℝ) = 1 := by
  sorry

end existence_of_prime_and_power_l3468_346877


namespace average_cat_weight_in_pounds_l3468_346852

def cat_weights : List Real := [3.5, 7.2, 4.8, 6, 5.5, 9, 4, 7.5]
def kg_to_pounds : Real := 2.20462

theorem average_cat_weight_in_pounds :
  let total_weight_kg := cat_weights.sum
  let average_weight_kg := total_weight_kg / cat_weights.length
  let average_weight_pounds := average_weight_kg * kg_to_pounds
  average_weight_pounds = 13.0925 := by sorry

end average_cat_weight_in_pounds_l3468_346852
