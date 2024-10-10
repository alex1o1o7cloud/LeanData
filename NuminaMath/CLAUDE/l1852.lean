import Mathlib

namespace prime_sequence_ones_digit_l1852_185253

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to get the ones digit of a number
def onesDigit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) : 
  isPrime p → isPrime q → isPrime r → isPrime s →
  p > 3 →
  q = p + 4 →
  r = q + 4 →
  s = r + 4 →
  onesDigit p = 9 := by
  sorry

end prime_sequence_ones_digit_l1852_185253


namespace least_reducible_fraction_l1852_185251

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ n - 17 ≠ 0 ∧ 7*n + 8 ≠ 0 ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7*n + 8)) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    m - 17 = 0 ∨ 7*m + 8 = 0 ∨
    (∀ (j : ℕ), j > 1 → ¬(j ∣ (m - 17) ∧ j ∣ (7*m + 8)))) ∧
  n = 144 :=
by sorry


end least_reducible_fraction_l1852_185251


namespace sin_690_degrees_l1852_185242

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end sin_690_degrees_l1852_185242


namespace points_needed_for_average_l1852_185235

/-- 
Given a basketball player who has scored 333 points in 10 games, 
this theorem proves that the player needs to score 41 points in the 11th game 
to achieve an average of 34 points over 11 games.
-/
theorem points_needed_for_average (total_points : ℕ) (num_games : ℕ) (target_average : ℕ) :
  total_points = 333 →
  num_games = 10 →
  target_average = 34 →
  (total_points + 41) / (num_games + 1) = target_average := by
  sorry

end points_needed_for_average_l1852_185235


namespace min_value_sum_of_reciprocals_min_value_sum_of_reciprocals_achieved_l1852_185209

theorem min_value_sum_of_reciprocals (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 4) :
  (1 / (x - 1) + 1 / (y - 1)) ≥ 2 :=
sorry

theorem min_value_sum_of_reciprocals_achieved (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 4) :
  (1 / (x - 1) + 1 / (y - 1) = 2) ↔ (x = 2 ∧ y = 2) :=
sorry

end min_value_sum_of_reciprocals_min_value_sum_of_reciprocals_achieved_l1852_185209


namespace walkway_area_is_416_l1852_185288

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the configuration of a garden -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed : FlowerBed
  walkway_width : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℝ :=
  let total_width := g.columns * g.bed.length + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed.width + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed.length * g.bed.width
  total_area - beds_area

/-- Theorem stating that the walkway area for the given garden configuration is 416 square feet -/
theorem walkway_area_is_416 (g : Garden) 
  (h_rows : g.rows = 4)
  (h_columns : g.columns = 3)
  (h_bed_length : g.bed.length = 8)
  (h_bed_width : g.bed.width = 3)
  (h_walkway_width : g.walkway_width = 2) :
  walkway_area g = 416 := by
  sorry

end walkway_area_is_416_l1852_185288


namespace rectangular_piece_too_large_l1852_185266

theorem rectangular_piece_too_large (square_area : ℝ) (rect_area : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) :
  square_area = 400 →
  rect_area = 300 →
  ratio_length = 3 →
  ratio_width = 2 →
  ∃ (rect_length : ℝ), 
    rect_length * (rect_length * ratio_width / ratio_length) = rect_area ∧
    rect_length > Real.sqrt square_area :=
by sorry

end rectangular_piece_too_large_l1852_185266


namespace system_of_equations_solution_l1852_185212

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (3 * x + y = 8) ∧ (2 * x - y = 7) :=
by
  -- Proof goes here
  sorry

end system_of_equations_solution_l1852_185212


namespace square_area_l1852_185245

/-- Given a square ABCD composed of two identical rectangles and two squares with side lengths 2 cm and 4 cm respectively, prove that the area of ABCD is 36 cm². -/
theorem square_area (s : ℝ) (h1 : s > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  a + 2 = 4 ∧
  b + 4 = s ∧
  s = 6 ∧
  s^2 = 36 := by
sorry

end square_area_l1852_185245


namespace max_table_sum_l1852_185256

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 4 ∧ 
  (∀ x ∈ top, x ∈ primes) ∧ 
  (∀ x ∈ left, x ∈ primes) ∧
  17 ∈ top ∧
  (∀ x ∈ primes, x ∈ top ∨ x ∈ left) ∧
  (∀ x ∈ top, ∀ y ∈ left, x ≠ y)

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum) * (left.sum)

theorem max_table_sum :
  ∀ top left, is_valid_arrangement top left →
  table_sum top left ≤ 825 :=
sorry

end max_table_sum_l1852_185256


namespace prime_equation_solutions_l1852_185281

theorem prime_equation_solutions :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  (p^(2*q) + q^(2*p)) / (p^3 - p*q + q^3) = r →
  ((p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 3 ∧ q = 2 ∧ r = 5)) :=
by sorry

end prime_equation_solutions_l1852_185281


namespace complex_roots_l1852_185299

theorem complex_roots (a' b' c' d' k' : ℂ) 
  (h1 : a' * k' ^ 2 + b' * k' + c' = 0)
  (h2 : b' * k' ^ 2 + c' * k' + d' = 0)
  (h3 : d' = a')
  (h4 : k' ≠ 0) :
  k' = 1 ∨ k' = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ k' = (-1 - Complex.I * Real.sqrt 3) / 2 := by
  sorry

end complex_roots_l1852_185299


namespace piggy_bank_value_l1852_185254

-- Define the number of pennies and dimes in one piggy bank
def pennies_per_bank : ℕ := 100
def dimes_per_bank : ℕ := 50

-- Define the value of pennies and dimes in cents
def penny_value : ℕ := 1
def dime_value : ℕ := 10

-- Define the number of piggy banks
def num_banks : ℕ := 2

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem piggy_bank_value :
  (num_banks * (pennies_per_bank * penny_value + dimes_per_bank * dime_value)) / cents_per_dollar = 12 := by
  sorry

end piggy_bank_value_l1852_185254


namespace age_difference_l1852_185200

/-- Given three people a, b, and c, where the total age of a and b is 20 years more than
    the total age of b and c, prove that c is 20 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 20) : a = c + 20 := by
  sorry

end age_difference_l1852_185200


namespace mistake_in_report_l1852_185287

def reported_numbers : List Nat := [3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]

def num_boys : Nat := 7
def num_girls : Nat := 8

theorem mistake_in_report :
  (List.sum reported_numbers) % 2 = 0 →
  ¬(∃ (boys_sum : Nat), 
    boys_sum * 2 = List.sum reported_numbers ∧
    boys_sum = num_girls * (List.sum reported_numbers / (num_boys + num_girls))) :=
by sorry

end mistake_in_report_l1852_185287


namespace least_positive_integer_multiple_of_43_l1852_185269

theorem least_positive_integer_multiple_of_43 :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), y < x → ¬(43 ∣ (2*y)^2 + 2*33*(2*y) + 33^2)) ∧ 
    (43 ∣ (2*x)^2 + 2*33*(2*x) + 33^2) ∧
    x = 5 := by
  sorry

end least_positive_integer_multiple_of_43_l1852_185269


namespace train_length_train_length_approx_145m_l1852_185276

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proof that a train's length is approximately 145 meters -/
theorem train_length_approx_145m (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 58)
  (h2 : time_s = 9) :
  ∃ ε > 0, |train_length speed_kmh time_s - 145| < ε :=
by
  sorry

end train_length_train_length_approx_145m_l1852_185276


namespace constant_product_on_circle_l1852_185250

theorem constant_product_on_circle (x₀ y₀ : ℝ) :
  x₀ ≠ 0 →
  y₀ ≠ 0 →
  x₀^2 + y₀^2 = 4 →
  |2 + 2*x₀/(y₀-2)| * |2 + 2*y₀/(x₀-2)| = 8 := by
sorry

end constant_product_on_circle_l1852_185250


namespace max_profit_selling_price_l1852_185296

/-- Represents the profit function for a product sale --/
def profit_function (initial_cost initial_price initial_sales price_sensitivity : ℝ) (x : ℝ) : ℝ :=
  (x - initial_cost) * (initial_sales - (x - initial_price) * price_sensitivity)

/-- Theorem stating the maximum profit and optimal selling price --/
theorem max_profit_selling_price 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_sensitivity : ℝ)
  (h_initial_cost : initial_cost = 8)
  (h_initial_price : initial_price = 10)
  (h_initial_sales : initial_sales = 60)
  (h_price_sensitivity : price_sensitivity = 10) :
  ∃ (max_profit optimal_price : ℝ),
    max_profit = 160 ∧ 
    optimal_price = 12 ∧
    ∀ x, profit_function initial_cost initial_price initial_sales price_sensitivity x ≤ max_profit :=
by sorry

end max_profit_selling_price_l1852_185296


namespace coinciding_directrices_l1852_185279

/-- Given a hyperbola and a parabola with coinciding directrices, prove that p = 3 -/
theorem coinciding_directrices (p : ℝ) : p > 0 → ∃ (x y : ℝ),
  (x^2 / 3 - y^2 = 1 ∧ y^2 = 2*p*x ∧ 
   (x = -3/2 ∨ x = 3/2) ∧ x = -p/2) → p = 3 := by
  sorry

end coinciding_directrices_l1852_185279


namespace smallest_number_is_21_l1852_185239

/-- A sequence of 25 consecutive natural numbers satisfying certain conditions -/
def ConsecutiveSequence (start : ℕ) : Prop :=
  ∃ (seq : Fin 25 → ℕ),
    (∀ i, seq i = start + i) ∧
    (((Finset.filter (λ i => seq i % 2 = 0) Finset.univ).card : ℚ) / 25 = 12 / 25) ∧
    (((Finset.filter (λ i => seq i < 30) Finset.univ).card : ℚ) / 25 = 9 / 25)

/-- The smallest number in the sequence is 21 -/
theorem smallest_number_is_21 :
  ∃ (start : ℕ), ConsecutiveSequence start ∧ ∀ s, ConsecutiveSequence s → start ≤ s :=
by sorry

end smallest_number_is_21_l1852_185239


namespace game_lives_distribution_l1852_185257

/-- Given a game with initial players, new players joining, and a total number of lives,
    calculate the number of lives per player. -/
def lives_per_player (initial_players : ℕ) (new_players : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players + new_players)

/-- Theorem: In a game with 4 initial players, 5 new players joining, and a total of 27 lives,
    each player has 3 lives. -/
theorem game_lives_distribution :
  lives_per_player 4 5 27 = 3 := by
  sorry

end game_lives_distribution_l1852_185257


namespace floor_sqrt_sum_equality_l1852_185285

theorem floor_sqrt_sum_equality (n : ℕ) : 
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (n + 1) + Real.sqrt (n + 2)⌋ := by
  sorry

end floor_sqrt_sum_equality_l1852_185285


namespace angle_sum_less_than_three_halves_pi_l1852_185260

theorem angle_sum_less_than_three_halves_pi
  (α β : Real)
  (h1 : π / 2 < α ∧ α < π)
  (h2 : π / 2 < β ∧ β < π)
  (h3 : Real.tan α < Real.tan (π / 2 - β)) :
  α + β < 3 * π / 2 := by
  sorry

end angle_sum_less_than_three_halves_pi_l1852_185260


namespace last_three_digits_of_3_800_l1852_185227

theorem last_three_digits_of_3_800 (h : 3^400 ≡ 1 [ZMOD 500]) :
  3^800 ≡ 1 [ZMOD 1000] := by
sorry

end last_three_digits_of_3_800_l1852_185227


namespace intersection_at_single_point_l1852_185284

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 10

/-- The line equation -/
def line (k : ℝ) : ℝ := k

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! y, parabola y = line k

/-- Theorem stating the value of k for which the line intersects the parabola at exactly one point -/
theorem intersection_at_single_point :
  ∀ k : ℝ, single_intersection k ↔ k = 34/3 :=
sorry

end intersection_at_single_point_l1852_185284


namespace percentage_good_fruits_is_87_point_6_percent_l1852_185268

/-- Calculates the percentage of fruits in good condition given the quantities and spoilage rates --/
def percentageGoodFruits (oranges bananas apples pears : ℕ) 
  (orangesSpoilage bananaSpoilage appleSpoilage pearSpoilage : ℚ) : ℚ :=
  let totalFruits := oranges + bananas + apples + pears
  let goodOranges := oranges - (oranges * orangesSpoilage).floor
  let goodBananas := bananas - (bananas * bananaSpoilage).floor
  let goodApples := apples - (apples * appleSpoilage).floor
  let goodPears := pears - (pears * pearSpoilage).floor
  let totalGoodFruits := goodOranges + goodBananas + goodApples + goodPears
  (totalGoodFruits : ℚ) / (totalFruits : ℚ) * 100

/-- Theorem stating that the percentage of good fruits is 87.6% given the problem conditions --/
theorem percentage_good_fruits_is_87_point_6_percent :
  percentageGoodFruits 600 400 800 200 (15/100) (3/100) (12/100) (25/100) = 876/10 := by
  sorry


end percentage_good_fruits_is_87_point_6_percent_l1852_185268


namespace room_width_l1852_185202

/-- Given a rectangular room with length 18 m and unknown width, surrounded by a 2 m wide veranda on all sides, 
    if the area of the veranda is 136 m², then the width of the room is 12 m. -/
theorem room_width (w : ℝ) : 
  w > 0 →  -- Ensure width is positive
  (22 * (w + 4) - 18 * w = 136) →  -- Area of veranda equation
  w = 12 := by
sorry

end room_width_l1852_185202


namespace lauren_mail_total_l1852_185295

/-- The total number of pieces of mail sent by Lauren over four days -/
def total_mail (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem stating the total number of pieces of mail sent by Lauren -/
theorem lauren_mail_total : ∃ (monday tuesday wednesday thursday : ℕ),
  monday = 65 ∧
  tuesday = monday + 10 ∧
  wednesday = tuesday - 5 ∧
  thursday = wednesday + 15 ∧
  total_mail monday tuesday wednesday thursday = 295 :=
by sorry

end lauren_mail_total_l1852_185295


namespace hyperbola_foci_distance_l1852_185206

/-- The distance between the foci of a hyperbola defined by xy = 4 is 8 -/
theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 * f₁.2 = 4 ∧ f₂.1 * f₂.2 = 4) ∧ 
    ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2)^(1/2 : ℝ) = 8 := by
  sorry


end hyperbola_foci_distance_l1852_185206


namespace divisor_condition_solutions_l1852_185244

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The condition that the number of divisors equals the cube root of 4n -/
def divisor_condition (n : ℕ) : Prop :=
  num_divisors n = (4 * n : ℝ) ^ (1/3 : ℝ)

/-- The main theorem stating that the divisor condition is satisfied only for 2, 128, and 2000 -/
theorem divisor_condition_solutions :
  ∀ n : ℕ, n > 0 → (divisor_condition n ↔ n = 2 ∨ n = 128 ∨ n = 2000) := by
  sorry


end divisor_condition_solutions_l1852_185244


namespace geometric_sequence_properties_l1852_185210

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n => (a n)^2)) ∧
  (is_geometric_sequence (fun n => a (2*n))) ∧
  (is_geometric_sequence (fun n => 1 / (a n))) ∧
  (is_geometric_sequence (fun n => |a n|)) :=
by sorry

end geometric_sequence_properties_l1852_185210


namespace parabola_y_axis_intersection_l1852_185270

/-- The parabola function -/
def f (x : ℝ) : ℝ := -(x + 2)^2 + 6

/-- The y-axis -/
def y_axis : Set ℝ := {x | x = 0}

/-- Theorem: The intersection point of the parabola and the y-axis is (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃! p : ℝ × ℝ, p.1 ∈ y_axis ∧ p.2 = f p.1 ∧ p = (0, 2) := by
sorry

end parabola_y_axis_intersection_l1852_185270


namespace cube_sum_and_reciprocal_l1852_185272

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) :
  x^3 + 1/x^3 = -18 := by sorry

end cube_sum_and_reciprocal_l1852_185272


namespace geometric_sequence_sum_l1852_185252

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 = 1 - a 1 →
  a 4 = 9 - a 3 →
  a 4 + a 5 = 27 := by
sorry

end geometric_sequence_sum_l1852_185252


namespace absolute_value_of_w_l1852_185205

theorem absolute_value_of_w (w : ℂ) : w^2 - 6*w + 40 = 0 → Complex.abs w = Real.sqrt 40 := by
  sorry

end absolute_value_of_w_l1852_185205


namespace solve_candy_problem_l1852_185292

def candy_problem (packs : ℕ) (paid : ℕ) (change : ℕ) : Prop :=
  packs = 3 ∧ paid = 20 ∧ change = 11 →
  (paid - change) / packs = 3

theorem solve_candy_problem : candy_problem 3 20 11 := by
  sorry

end solve_candy_problem_l1852_185292


namespace max_value_problem_l1852_185214

theorem max_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 2 * Real.sqrt (x * y) - 4 * x^2 - y^2 ≤ 2 * Real.sqrt (a * b) - 4 * a^2 - b^2) →
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2 :=
by sorry

end max_value_problem_l1852_185214


namespace sss_sufficient_for_angle_construction_l1852_185217

/-- A triangle in a plane -/
structure Triangle :=
  (A B C : Point)

/-- Congruence relation between triangles -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

/-- Length of a side in a triangle -/
def SideLength (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- Angle measure in a triangle -/
def AngleMeasure (t : Triangle) (angle : Fin 3) : ℝ := sorry

/-- SSS congruence criterion -/
axiom sss_congruence (t1 t2 : Triangle) :
  (∀ i : Fin 3, SideLength t1 i = SideLength t2 i) → Congruent t1 t2

/-- Compass and straightedge construction -/
def ConstructibleAngle (θ : ℝ) : Prop := sorry

/-- Theorem: SSS is sufficient for angle construction -/
theorem sss_sufficient_for_angle_construction (θ : ℝ) (t : Triangle) :
  (∃ i : Fin 3, AngleMeasure t i = θ) →
  ConstructibleAngle θ :=
sorry

end sss_sufficient_for_angle_construction_l1852_185217


namespace one_third_percent_of_180_l1852_185201

-- Define the percentage as a fraction
def one_third_percent : ℚ := 1 / 3 / 100

-- Define the value we're calculating the percentage of
def base_value : ℚ := 180

-- Theorem statement
theorem one_third_percent_of_180 : one_third_percent * base_value = 0.6 := by
  sorry

end one_third_percent_of_180_l1852_185201


namespace min_sum_of_reciprocal_sum_eq_one_l1852_185275

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 1 / a + 1 / b = 1) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → a + b ≤ x + y ∧ a + b = 4 := by
sorry

end min_sum_of_reciprocal_sum_eq_one_l1852_185275


namespace monotonic_function_constraint_l1852_185238

theorem monotonic_function_constraint (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x ∈ Set.Icc (-1) 2, Monotone (fun x => -1/3 * x^3 + a * x^2 + b * x)) →
  a + b ≥ 5/2 := by
  sorry

end monotonic_function_constraint_l1852_185238


namespace rectangle_area_ratio_l1852_185297

/-- Given three rectangles with specific side length ratios, prove the ratio of areas -/
theorem rectangle_area_ratio (a b c d e f : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) 
  (h3 : a / e = 7 / 4) 
  (h4 : b / f = 7 / 4) 
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : c ≠ 0) (h8 : d ≠ 0) (h9 : e ≠ 0) (h10 : f ≠ 0) :
  (a * b) / ((c * d) + (e * f)) = 441 / 1369 := by
  sorry

end rectangle_area_ratio_l1852_185297


namespace plane_perpendicular_theorem_l1852_185236

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_theorem 
  (α β : Plane) (n : Line) 
  (h1 : contains β n) 
  (h2 : perpendicular n α) :
  perpendicularPlanes α β :=
sorry

end plane_perpendicular_theorem_l1852_185236


namespace area_of_arcsin_cos_l1852_185224

open Set
open MeasureTheory
open Interval

noncomputable def f (x : ℝ) := Real.arcsin (Real.cos x)

theorem area_of_arcsin_cos (a b : ℝ) (h : 0 ≤ a ∧ b = 2 * Real.pi) :
  (∫ x in a..b, |f x| ) = Real.pi^2 / 2 := by
  sorry

end area_of_arcsin_cos_l1852_185224


namespace sports_club_non_athletic_parents_l1852_185249

/-- Represents a sports club with members and their parents' athletic status -/
structure SportsClub where
  total_members : ℕ
  athletic_dads : ℕ
  athletic_moms : ℕ
  both_athletic : ℕ
  no_dads : ℕ

/-- Calculates the number of members with non-athletic parents in a sports club -/
def members_with_non_athletic_parents (club : SportsClub) : ℕ :=
  club.total_members - (club.athletic_dads + club.athletic_moms - club.both_athletic - club.no_dads)

/-- Theorem stating the number of members with non-athletic parents in the given sports club -/
theorem sports_club_non_athletic_parents :
  let club : SportsClub := {
    total_members := 50,
    athletic_dads := 25,
    athletic_moms := 30,
    both_athletic := 10,
    no_dads := 5
  }
  members_with_non_athletic_parents club = 10 := by
  sorry

end sports_club_non_athletic_parents_l1852_185249


namespace triangle_proof_l1852_185282

open Real

theorem triangle_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  -- Given condition
  a * cos C - c / 2 = b →
  -- Part I
  A = 2 * π / 3 ∧
  -- Part II
  a = 3 →
  -- Perimeter range
  let l := a + b + c
  6 < l ∧ l ≤ 3 + 2 * sqrt 3 :=
by sorry

end triangle_proof_l1852_185282


namespace parabola_focus_l1852_185298

/-- The focus of a parabola y^2 = 8x with directrix x + 2 = 0 is at (2,0) -/
theorem parabola_focus (x y : ℝ) : 
  (y^2 = 8*x) →  -- point (x,y) is on the parabola
  (∀ (a b : ℝ), (a + 2 = 0) → ((x - a)^2 + (y - b)^2 = 4)) → -- distance to directrix equals distance to (2,0)
  (x = 2 ∧ y = 0) -- focus is at (2,0)
  := by sorry

end parabola_focus_l1852_185298


namespace subtract_three_from_M_l1852_185232

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def M : List Bool := [false, false, false, false, true, true, false, true]

theorem subtract_three_from_M :
  decimal_to_binary (binary_to_decimal M - 3) = 
    [true, false, true, true, false, true, false, true] := by
  sorry

end subtract_three_from_M_l1852_185232


namespace sqrt2_power0_plus_neg2_power3_l1852_185267

theorem sqrt2_power0_plus_neg2_power3 : (Real.sqrt 2) ^ 0 + (-2) ^ 3 = -7 := by
  sorry

end sqrt2_power0_plus_neg2_power3_l1852_185267


namespace B_power_15_minus_3_power_14_l1852_185290

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end B_power_15_minus_3_power_14_l1852_185290


namespace units_digit_of_subtraction_is_seven_l1852_185215

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its integer value -/
def to_int (n : ThreeDigitNumber) : Int :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  is_valid := by sorry

/-- The main theorem -/
theorem units_digit_of_subtraction_is_seven (n : ThreeDigitNumber) 
  (h : n.hundreds = n.units + 3) : 
  (to_int n - to_int (reverse n)) % 10 = 7 := by
  sorry

end units_digit_of_subtraction_is_seven_l1852_185215


namespace only_negative_two_less_than_negative_one_l1852_185248

theorem only_negative_two_less_than_negative_one : ∀ x : ℚ, 
  (x = 0 ∨ x = -1/2 ∨ x = 1 ∨ x = -2) → (x < -1 ↔ x = -2) :=
by
  sorry

end only_negative_two_less_than_negative_one_l1852_185248


namespace quadratic_coefficients_l1852_185237

/-- A quadratic function with vertex (4, -1) passing through (0, 7) has coefficients a = 1/2, b = -4, and c = 7. -/
theorem quadratic_coefficients :
  ∀ (f : ℝ → ℝ) (a b c : ℝ),
    (∀ x, f x = a * x^2 + b * x + c) →
    (∀ x, f x = f (8 - x)) →
    f 4 = -1 →
    f 0 = 7 →
    a = (1/2 : ℝ) ∧ b = -4 ∧ c = 7 := by
  sorry

end quadratic_coefficients_l1852_185237


namespace remainder_102938475610_div_12_l1852_185294

theorem remainder_102938475610_div_12 : 102938475610 % 12 = 10 := by
  sorry

end remainder_102938475610_div_12_l1852_185294


namespace simplify_expression_1_simplify_expression_2_l1852_185246

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  -3*x*y - 3*x^2 + 4*x*y + 2*x^2 = x*y - x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  3*(a^2 - 2*a*b) - 5*(a^2 + 4*a*b) = -2*a^2 - 26*a*b := by sorry

end simplify_expression_1_simplify_expression_2_l1852_185246


namespace new_cube_edge_l1852_185243

/-- Given three cubes with edges 6 cm, 8 cm, and 10 cm, prove that when melted and formed into a new cube, the edge of the new cube is 12 cm. -/
theorem new_cube_edge (cube1 cube2 cube3 new_cube : ℝ) : 
  cube1 = 6 → cube2 = 8 → cube3 = 10 → 
  (cube1^3 + cube2^3 + cube3^3)^(1/3) = new_cube → 
  new_cube = 12 := by
sorry

#eval (6^3 + 8^3 + 10^3)^(1/3) -- This should evaluate to 12

end new_cube_edge_l1852_185243


namespace scaling_transformation_correct_l1852_185261

/-- Scaling transformation function -/
def scale (sx sy : ℚ) (p : ℚ × ℚ) : ℚ × ℚ :=
  (sx * p.1, sy * p.2)

/-- The initial point -/
def initial_point : ℚ × ℚ := (1, 2)

/-- The scaling factors -/
def sx : ℚ := 1/2
def sy : ℚ := 1/3

/-- The expected result after transformation -/
def expected_result : ℚ × ℚ := (1/2, 2/3)

theorem scaling_transformation_correct :
  scale sx sy initial_point = expected_result := by
  sorry

end scaling_transformation_correct_l1852_185261


namespace smaller_bill_denomination_l1852_185247

/-- Given a cashier with bills of two denominations, prove the value of the smaller denomination. -/
theorem smaller_bill_denomination
  (total_bills : ℕ)
  (total_value : ℕ)
  (smaller_bills : ℕ)
  (twenty_bills : ℕ)
  (h_total_bills : total_bills = smaller_bills + twenty_bills)
  (h_total_bills_value : total_bills = 30)
  (h_total_value : total_value = 330)
  (h_smaller_bills : smaller_bills = 27)
  (h_twenty_bills : twenty_bills = 3) :
  ∃ (x : ℕ), x * smaller_bills + 20 * twenty_bills = total_value ∧ x = 10 := by
sorry


end smaller_bill_denomination_l1852_185247


namespace verandah_area_l1852_185273

/-- The area of a verandah surrounding a rectangular room -/
theorem verandah_area (room_length room_width verandah_width : ℝ) :
  room_length = 15 ∧ room_width = 12 ∧ verandah_width = 2 →
  (room_length + 2 * verandah_width) * (room_width + 2 * verandah_width) -
  room_length * room_width = 124 := by sorry

end verandah_area_l1852_185273


namespace circle_condition_chord_length_l1852_185218

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem for the range of m
theorem circle_condition (m : ℝ) :
  (∃ x y, circle_equation x y m) ↔ (m < 1 ∨ m > 4) :=
sorry

-- Theorem for the chord length
theorem chord_length :
  let m : ℝ := -2
  let center : ℝ × ℝ := (-2, 2)
  let radius : ℝ := 3 * Real.sqrt 2
  let d : ℝ := Real.sqrt 5
  2 * Real.sqrt (radius^2 - d^2) = 2 * Real.sqrt 13 :=
sorry

end circle_condition_chord_length_l1852_185218


namespace tangency_lines_through_diagonal_intersection_l1852_185222

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Vector Point 4

-- Function to check if a quadrilateral is circumscribed around a circle
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Function to get tangency points
def tangency_points (q : Quadrilateral) (c : Circle) : Vector Point 4 := sorry

-- Function to get lines connecting opposite tangency points
def opposite_tangency_lines (q : Quadrilateral) (c : Circle) : Vector Line 2 := sorry

-- Function to get diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : Vector Line 2 := sorry

-- Function to check if two lines intersect
def lines_intersect (l1 : Line) (l2 : Line) : Prop := sorry

-- Function to get intersection point of two lines
def intersection_point (l1 : Line) (l2 : Line) : Point := sorry

-- Theorem statement
theorem tangency_lines_through_diagonal_intersection 
  (q : Quadrilateral) (c : Circle) : 
  is_circumscribed q c → 
  let tl := opposite_tangency_lines q c
  let d := diagonals q
  lines_intersect tl[0] tl[1] ∧ 
  lines_intersect d[0] d[1] ∧
  intersection_point tl[0] tl[1] = intersection_point d[0] d[1] := by
  sorry

end tangency_lines_through_diagonal_intersection_l1852_185222


namespace smallest_max_sum_l1852_185241

theorem smallest_max_sum (a b c d e : ℕ+) (h : a + b + c + d + e = 2020) :
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
   (∀ a' b' c' d' e' : ℕ+, a' + b' + c' + d' + e' = 2020 →
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) ≥ M) ∧
   M = 674) :=
sorry

end smallest_max_sum_l1852_185241


namespace saloon_prices_l1852_185265

/-- The cost of items in a saloon -/
structure SaloonPrices where
  sandwich : ℚ
  coffee : ℚ
  donut : ℚ

/-- The total cost of a purchase -/
def total_cost (p : SaloonPrices) (s c d : ℕ) : ℚ :=
  s * p.sandwich + c * p.coffee + d * p.donut

/-- The prices in the saloon satisfy the given conditions -/
def satisfies_conditions (p : SaloonPrices) : Prop :=
  total_cost p 4 1 10 = 169/100 ∧ total_cost p 3 1 7 = 126/100

theorem saloon_prices (p : SaloonPrices) (h : satisfies_conditions p) :
  total_cost p 1 1 1 = 40/100 := by
  sorry

end saloon_prices_l1852_185265


namespace other_number_proof_l1852_185240

/-- Given two positive integers with specific LCM, HCF, and one known value, prove the value of the other integer -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.lcm A B = 2310 → 
  Nat.gcd A B = 30 → 
  A = 210 → 
  B = 330 := by
sorry

end other_number_proof_l1852_185240


namespace quadratic_discriminant_l1852_185225

theorem quadratic_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) →
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4*a*c = -1/2 := by
sorry

end quadratic_discriminant_l1852_185225


namespace expression_evaluation_l1852_185208

theorem expression_evaluation (x : ℝ) (h : x < 0) :
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2*x + 1)) := by
  sorry

end expression_evaluation_l1852_185208


namespace power_of_four_in_expression_l1852_185278

theorem power_of_four_in_expression (x : ℕ) : 
  (2 * x + 5 + 2 = 29) → x = 11 := by sorry

end power_of_four_in_expression_l1852_185278


namespace f_lower_bound_g_inequality_min_a_l1852_185203

noncomputable section

variables (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := a * Real.exp (2*x - 1) - x^2 * (Real.log x + 1/2)

def g (x : ℝ) (a : ℝ) : ℝ := x * f x a + x^2 / Real.exp x

theorem f_lower_bound (h : x > 0) : f x 0 ≥ x^2/2 - x^3 := by sorry

theorem g_inequality_min_a :
  (∀ x > 1, x * g (Real.log x / (x - 1)) a < g (x * Real.log x / (x - 1)) a) ↔ a ≥ 1 / Real.exp 1 := by sorry

end

end f_lower_bound_g_inequality_min_a_l1852_185203


namespace truck_rental_percentage_l1852_185289

/-- The percentage of trucks returned given the total number of trucks,
    the number of trucks rented out, and the number of trucks returned -/
def percentage_returned (total : ℕ) (rented : ℕ) (returned : ℕ) : ℚ :=
  (returned : ℚ) / (rented : ℚ) * 100

theorem truck_rental_percentage (total : ℕ) (rented : ℕ) (returned : ℕ)
  (h_total : total = 24)
  (h_rented : rented = total)
  (h_returned : returned ≥ 12) :
  percentage_returned total rented returned = 50 := by
sorry

end truck_rental_percentage_l1852_185289


namespace percentage_of_360_equals_115_2_l1852_185216

theorem percentage_of_360_equals_115_2 : 
  let whole : ℝ := 360
  let part : ℝ := 115.2
  let percentage : ℝ := (part / whole) * 100
  percentage = 32 := by sorry

end percentage_of_360_equals_115_2_l1852_185216


namespace expected_no_allergies_is_75_l1852_185233

/-- The probability that an American does not suffer from allergies -/
def prob_no_allergies : ℚ := 1/4

/-- The size of the random sample of Americans -/
def sample_size : ℕ := 300

/-- The expected number of people in the sample who do not suffer from allergies -/
def expected_no_allergies : ℚ := prob_no_allergies * sample_size

theorem expected_no_allergies_is_75 : expected_no_allergies = 75 := by
  sorry

end expected_no_allergies_is_75_l1852_185233


namespace isosceles_right_triangle_in_circle_l1852_185204

/-- Given an isosceles right triangle inscribed in a circle with radius √2,
    where the side lengths are in the ratio 1:1:√2,
    prove that the area of the triangle is 2 and the circumference of the circle is 2π√2. -/
theorem isosceles_right_triangle_in_circle 
  (r : ℝ) 
  (h_r : r = Real.sqrt 2) 
  (a b c : ℝ) 
  (h_abc : a = b ∧ c = a * Real.sqrt 2) 
  (h_inscribed : c = 2 * r) : 
  (1/2 * a * b = 2) ∧ (2 * Real.pi * r = 2 * Real.pi * Real.sqrt 2) := by
  sorry

#check isosceles_right_triangle_in_circle

end isosceles_right_triangle_in_circle_l1852_185204


namespace sum_of_xy_l1852_185271

theorem sum_of_xy (x y : ℝ) 
  (eq1 : x^2 + 3*x*y + y^2 = 909)
  (eq2 : 3*x^2 + x*y + 3*y^2 = 1287) :
  x + y = 27 ∨ x + y = -27 := by
  sorry

end sum_of_xy_l1852_185271


namespace S_is_circle_l1852_185274

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 - 4 * Complex.I)}

-- Theorem stating that S is a circle
theorem S_is_circle : ∃ (c : ℂ) (r : ℝ), S = {z : ℂ | Complex.abs (z - c) = r} := by
  sorry

end S_is_circle_l1852_185274


namespace winning_team_arrangements_winning_team_groupings_winning_team_selections_l1852_185283

/-- A debate team with male and female members -/
structure DebateTeam where
  male_members : ℕ
  female_members : ℕ

/-- The national winning debate team -/
def winning_team : DebateTeam :=
  { male_members := 3, female_members := 5 }

/-- Number of arrangements with male members not adjacent -/
def non_adjacent_arrangements (team : DebateTeam) : ℕ := sorry

/-- Number of ways to divide into pairs for classes -/
def pair_groupings (team : DebateTeam) (num_classes : ℕ) : ℕ := sorry

/-- Number of ways to select debaters with at least one male -/
def debater_selections (team : DebateTeam) (num_debaters : ℕ) : ℕ := sorry

theorem winning_team_arrangements :
  non_adjacent_arrangements winning_team = 14400 := by sorry

theorem winning_team_groupings :
  pair_groupings winning_team 4 = 2520 := by sorry

theorem winning_team_selections :
  debater_selections winning_team 4 = 1560 := by sorry

end winning_team_arrangements_winning_team_groupings_winning_team_selections_l1852_185283


namespace hemisphere_volume_l1852_185234

/-- Given a hemisphere with surface area (excluding the base) of 256π cm²,
    prove that its volume is (2048√2)/3 π cm³. -/
theorem hemisphere_volume (r : ℝ) (h : 2 * Real.pi * r^2 = 256 * Real.pi) :
  (2/3) * Real.pi * r^3 = (2048 * Real.sqrt 2 / 3) * Real.pi := by
  sorry

end hemisphere_volume_l1852_185234


namespace pizza_distribution_l1852_185207

/-- Given 6 people sharing 3 pizzas with 8 slices each, if they all eat the same amount and finish all the pizzas, each person will eat 4 slices. -/
theorem pizza_distribution (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 6)
  (h2 : num_pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (num_pizzas * slices_per_pizza) / num_people = 4 :=
by
  sorry

#check pizza_distribution

end pizza_distribution_l1852_185207


namespace complex_fraction_simplification_l1852_185211

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - i) / (2 + i) = 1 - i := by sorry

end complex_fraction_simplification_l1852_185211


namespace triangle_side_length_l1852_185230

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + C = 2 * B → 
  A + B + C = π → 
  a = 1 → 
  b = Real.sqrt 3 → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos B) →
  c = Real.sqrt 3 := by
sorry

end triangle_side_length_l1852_185230


namespace f_is_even_l1852_185231

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x, f g (-x) = f g x := by
  sorry

end f_is_even_l1852_185231


namespace solution_set_characterization_l1852_185293

/-- A differentiable function satisfying certain conditions -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  differentiable : Differentiable ℝ f
  domain : ∀ x, x < 0 → f x ≠ 0
  condition : ∀ x, x < 0 → 3 * f x + x * deriv f x < 0

/-- The solution set of the inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | (x + 2016)^3 * f (x + 2016) + 8 * f (-2) < 0}

theorem solution_set_characterization (f : ℝ → ℝ) [SpecialFunction f] :
  SolutionSet f = Set.Ioo (-2018) (-2016) := by
  sorry

end solution_set_characterization_l1852_185293


namespace crayons_left_in_drawer_l1852_185280

theorem crayons_left_in_drawer (initial_crayons : ℕ) (crayons_taken : ℕ) : 
  initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by sorry

end crayons_left_in_drawer_l1852_185280


namespace right_triangle_integer_area_l1852_185286

theorem right_triangle_integer_area (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ S : ℕ, S * 2 = a * b := by
sorry

end right_triangle_integer_area_l1852_185286


namespace stamp_collection_theorem_l1852_185219

/-- Calculates the total value of a stamp collection given the following conditions:
    - The total number of stamps in the collection
    - The number of stamps in a subset of the collection
    - The total value of the subset of stamps
    Assumes that all stamps have the same value. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * (subset_value / subset_stamps)

/-- Proves that a collection of 18 stamps, where 6 stamps are worth 18 dollars,
    has a total value of 54 dollars. -/
theorem stamp_collection_theorem :
  stamp_collection_value 18 6 18 = 54 := by
  sorry

end stamp_collection_theorem_l1852_185219


namespace complement_union_equality_l1852_185262

-- Define the sets M, N, and U
variable (M N U : Set α)

-- Define the conditions
variable (hM : M.Nonempty)
variable (hN : N.Nonempty)
variable (hU : U.Nonempty)
variable (hMN : M ⊆ N)
variable (hNU : N ⊆ U)

-- State the theorem
theorem complement_union_equality :
  (U \ M) ∪ (U \ N) = U \ M :=
sorry

end complement_union_equality_l1852_185262


namespace fourth_intersection_point_l1852_185277

/-- The curve defined by xy = 2 -/
def curve (x y : ℝ) : Prop := x * y = 2

/-- An arbitrary ellipse in the coordinate plane -/
def ellipse (x y : ℝ) : Prop := sorry

/-- The four points of intersection satisfy both the curve and ellipse equations -/
axiom intersection_points (x y : ℝ) : curve x y ∧ ellipse x y ↔ 
  (x = 3 ∧ y = 2/3) ∨ (x = -4 ∧ y = -1/2) ∨ (x = 1/4 ∧ y = 8) ∨ (x = -2/3 ∧ y = -3)

theorem fourth_intersection_point : 
  ∃ (x y : ℝ), curve x y ∧ ellipse x y ∧ x = -2/3 ∧ y = -3 :=
sorry

end fourth_intersection_point_l1852_185277


namespace not_even_if_symmetric_to_x_squared_l1852_185220

-- Define the function g(x) = x^2 for x ≥ 0
def g (x : ℝ) : ℝ := x^2

-- Define symmetry with respect to y = x
def symmetricToYEqualsX (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define an even function
def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem not_even_if_symmetric_to_x_squared (f : ℝ → ℝ) 
  (h_sym : symmetricToYEqualsX f g) : ¬ (isEven f) := by
  sorry

end not_even_if_symmetric_to_x_squared_l1852_185220


namespace box_surface_area_l1852_185258

/-- The surface area of a box formed by removing triangles from corners of a rectangle --/
theorem box_surface_area (length width triangle_side : ℕ) : 
  length = 25 →
  width = 40 →
  triangle_side = 4 →
  (length * width) - (4 * (triangle_side * triangle_side / 2)) = 968 :=
by
  sorry


end box_surface_area_l1852_185258


namespace left_handed_rock_lovers_l1852_185263

theorem left_handed_rock_lovers (total : Nat) (left_handed : Nat) (rock_lovers : Nat) (right_handed_non_rock : Nat)
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_lovers = 18)
  (h4 : right_handed_non_rock = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : Nat, x = left_handed + rock_lovers - total + right_handed_non_rock ∧ x = 6 := by
  sorry

end left_handed_rock_lovers_l1852_185263


namespace appropriate_presentation_length_l1852_185259

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration : Type := { d : ℝ // 20 ≤ d ∧ d ≤ 40 }

/-- The speaking rate in words per minute -/
def speakingRate : ℝ := 160

/-- Calculates the number of words for a given duration -/
def wordsForDuration (d : PresentationDuration) : ℝ := d.val * speakingRate

/-- Theorem stating that 5000 words is an appropriate length for the presentation -/
theorem appropriate_presentation_length :
  ∃ (d : PresentationDuration), 5000 = wordsForDuration d := by
  sorry

end appropriate_presentation_length_l1852_185259


namespace sum_of_quadratic_roots_l1852_185223

theorem sum_of_quadratic_roots (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 - 6*y₁ + 8 = 0 ∧ y₂^2 - 6*y₂ + 8 = 0 ∧ y₁ + y₂ = 6) :=
by sorry

end sum_of_quadratic_roots_l1852_185223


namespace repair_cost_equals_profit_l1852_185228

/-- Proves that the repair cost equals the profit under given conditions --/
theorem repair_cost_equals_profit (original_cost : ℝ) : 
  let repair_cost := 0.1 * original_cost
  let selling_price := 1.2 * original_cost
  let profit := selling_price - (original_cost + repair_cost)
  profit = 1100 ∧ profit / original_cost = 0.2 → repair_cost = 1100 := by
sorry

end repair_cost_equals_profit_l1852_185228


namespace certain_number_is_five_l1852_185255

theorem certain_number_is_five (n d : ℕ) (h1 : d > 0) (h2 : n % d = 3) (h3 : (n^2) % d = 4) : d = 5 := by
  sorry

end certain_number_is_five_l1852_185255


namespace initial_volumes_l1852_185226

/-- Represents a cubic container with water --/
structure Container where
  capacity : ℝ
  initialVolume : ℝ
  currentVolume : ℝ

/-- The problem setup --/
def problemSetup : (Container × Container × Container) → Prop := fun (a, b, c) =>
  -- Capacities in ratio 1:8:27
  b.capacity = 8 * a.capacity ∧ c.capacity = 27 * a.capacity ∧
  -- Initial volumes in ratio 1:2:3
  b.initialVolume = 2 * a.initialVolume ∧ c.initialVolume = 3 * a.initialVolume ∧
  -- Same depth after first transfer
  a.currentVolume / a.capacity = b.currentVolume / b.capacity ∧
  b.currentVolume / b.capacity = c.currentVolume / c.capacity ∧
  -- Transfer from C to B
  ∃ (transferCB : ℝ), transferCB = 128 * (4/7) ∧
    c.currentVolume = c.initialVolume - transferCB ∧
    b.currentVolume = b.initialVolume + transferCB ∧
  -- Transfer from B to A, A's depth becomes twice B's
  ∃ (transferBA : ℝ), 
    a.currentVolume / a.capacity = 2 * (b.currentVolume - transferBA) / b.capacity ∧
  -- A has 10θ liters less than initially
  ∃ (θ : ℝ), a.currentVolume = a.initialVolume - 10 * θ

/-- The theorem to prove --/
theorem initial_volumes (a b c : Container) :
  problemSetup (a, b, c) →
  a.initialVolume = 500 ∧ b.initialVolume = 1000 ∧ c.initialVolume = 1500 := by
  sorry

end initial_volumes_l1852_185226


namespace johns_out_of_pocket_expense_l1852_185291

/-- Calculates the amount of money John spent out of pocket to buy a computer and accessories after selling his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value discount_rate : ℝ) 
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (h4 : discount_rate = 0.2) : 
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
  sorry

end johns_out_of_pocket_expense_l1852_185291


namespace pyramid_properties_l1852_185264

/-- Represents a pyramid ABCD with given edge lengths -/
structure Pyramid where
  DA : ℝ
  DB : ℝ
  DC : ℝ
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- The specific pyramid from the problem -/
def specific_pyramid : Pyramid :=
  { DA := 15
    DB := 12
    DC := 12
    AB := 9
    AC := 9
    BC := 3 }

/-- Calculates the radius of the circumscribed sphere around the pyramid -/
def circumscribed_sphere_radius (p : Pyramid) : ℝ := sorry

/-- Calculates the volume of the pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

theorem pyramid_properties :
  circumscribed_sphere_radius specific_pyramid = 7.5 ∧
  pyramid_volume specific_pyramid = 18 * Real.sqrt 3 := by
  sorry

end pyramid_properties_l1852_185264


namespace solution_of_square_eq_zero_l1852_185221

theorem solution_of_square_eq_zero :
  ∀ x : ℝ, x^2 = 0 ↔ x = 0 := by sorry

end solution_of_square_eq_zero_l1852_185221


namespace cylinder_base_area_ratio_l1852_185213

/-- Represents a cylinder with base area S and volume V -/
structure Cylinder where
  S : ℝ
  V : ℝ

/-- 
Given two cylinders with equal lateral areas and a volume ratio of 3/2,
prove that the ratio of their base areas is 9/4
-/
theorem cylinder_base_area_ratio 
  (A B : Cylinder) 
  (h1 : A.V / B.V = 3 / 2) 
  (h2 : ∃ (r₁ r₂ h₁ h₂ : ℝ), 
    A.S = π * r₁^2 ∧ 
    B.S = π * r₂^2 ∧ 
    A.V = π * r₁^2 * h₁ ∧ 
    B.V = π * r₂^2 * h₂ ∧ 
    2 * π * r₁ * h₁ = 2 * π * r₂ * h₂) : 
  A.S / B.S = 9 / 4 := by
sorry

end cylinder_base_area_ratio_l1852_185213


namespace grid_game_winner_l1852_185229

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the game state on a rectangular grid -/
structure GridGame where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- Determines the winner of the game based on the grid dimensions -/
def winner (game : GridGame) : Player :=
  if (game.m + game.n) % 2 = 0 then Player.Second else Player.First

/-- Theorem stating the winning condition for the grid game -/
theorem grid_game_winner (game : GridGame) :
  (winner game = Player.Second ↔ (game.m + game.n) % 2 = 0) ∧
  (winner game = Player.First ↔ (game.m + game.n) % 2 = 1) :=
sorry

end grid_game_winner_l1852_185229
