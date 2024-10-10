import Mathlib

namespace equation_solution_l3066_306691

theorem equation_solution (x : ℝ) : x ≠ -2 → (-2 * x^2 = (4 * x + 2) / (x + 2)) ↔ (x = -1) := by
  sorry

end equation_solution_l3066_306691


namespace yanna_payment_l3066_306654

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def num_sandals : ℕ := 3
def change : ℕ := 41

theorem yanna_payment :
  shirt_price * num_shirts + sandal_price * num_sandals + change = 100 := by
  sorry

end yanna_payment_l3066_306654


namespace carpet_border_problem_l3066_306607

theorem carpet_border_problem :
  let count_valid_pairs := 
    (Finset.filter 
      (fun pair : ℕ × ℕ => 
        let (p, q) := pair
        q > p ∧ (p - 6) * (q - 6) = 48 ∧ p > 6 ∧ q > 6)
      (Finset.product (Finset.range 100) (Finset.range 100))).card
  count_valid_pairs = 5 := by
sorry

end carpet_border_problem_l3066_306607


namespace multiples_difference_squared_l3066_306628

def a : ℕ := (Finset.filter (λ x => x % 7 = 0) (Finset.range 60)).card

def b : ℕ := (Finset.filter (λ x => x % 3 = 0 ∨ x % 7 = 0) (Finset.range 60)).card

theorem multiples_difference_squared : (a - b)^2 = 289 := by
  sorry

end multiples_difference_squared_l3066_306628


namespace nicole_bought_23_candies_l3066_306614

def nicole_candies (x : ℕ) : Prop :=
  ∃ (y : ℕ), 
    (2 * x) / 3 = y + 5 + 10 ∧ 
    y ≥ 0 ∧
    x > 0

theorem nicole_bought_23_candies : 
  ∃ (x : ℕ), nicole_candies x ∧ x = 23 := by sorry

end nicole_bought_23_candies_l3066_306614


namespace intersection_of_A_and_B_l3066_306662

def A : Set ℝ := {x | x < 3}
def B : Set ℝ := {x | 2 - x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l3066_306662


namespace gum_to_candy_ratio_l3066_306688

/-- The cost of a candy bar in dollars -/
def candy_cost : ℚ := 3/2

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 6

/-- The number of packs of gum purchased -/
def gum_packs : ℕ := 2

/-- The number of candy bars purchased -/
def candy_bars : ℕ := 3

theorem gum_to_candy_ratio :
  ∃ (gum_cost : ℚ), 
    gum_cost * gum_packs + candy_cost * candy_bars = total_cost ∧
    gum_cost / candy_cost = 1/2 := by
  sorry

end gum_to_candy_ratio_l3066_306688


namespace fixed_point_on_line_l3066_306693

theorem fixed_point_on_line (a b : ℝ) : (2 * a + b) * (-2) + (a + b) * 3 + a - b = 0 := by
  sorry

end fixed_point_on_line_l3066_306693


namespace quartic_roots_difference_l3066_306690

/-- A quartic polynomial with roots forming an arithmetic sequence -/
def quartic_with_arithmetic_roots (a : ℝ) (x : ℝ) : ℝ := 
  a * (x^4 - 10*x^2 + 9)

/-- The derivative of the quartic polynomial -/
def quartic_derivative (a : ℝ) (x : ℝ) : ℝ := 
  4 * a * x * (x^2 - 5)

theorem quartic_roots_difference (a : ℝ) (h : a ≠ 0) :
  let f := quartic_with_arithmetic_roots a
  let f' := quartic_derivative a
  let max_root := Real.sqrt 5
  let min_root := -Real.sqrt 5
  (∀ x, f' x = 0 → x ≤ max_root) ∧ 
  (∀ x, f' x = 0 → x ≥ min_root) ∧
  (max_root - min_root = 2 * Real.sqrt 5) :=
sorry

end quartic_roots_difference_l3066_306690


namespace smaller_number_of_product_and_difference_l3066_306657

theorem smaller_number_of_product_and_difference (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ x * y = 323 ∧ x - y = 2 → y = 17 := by
  sorry

end smaller_number_of_product_and_difference_l3066_306657


namespace expected_voters_for_candidate_A_l3066_306630

theorem expected_voters_for_candidate_A (total_voters : ℝ) (dem_percent : ℝ) 
  (dem_for_A : ℝ) (rep_for_A : ℝ) (h1 : dem_percent = 0.6) 
  (h2 : dem_for_A = 0.75) (h3 : rep_for_A = 0.3) : 
  (dem_percent * dem_for_A + (1 - dem_percent) * rep_for_A) * 100 = 57 := by
  sorry

end expected_voters_for_candidate_A_l3066_306630


namespace banana_arrangement_count_l3066_306676

/-- The number of distinct arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of occurrences of the letter A in BANANA -/
def num_a : ℕ := 3

/-- The number of occurrences of the letter N in BANANA -/
def num_n : ℕ := 2

/-- The number of occurrences of the letter B in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_a) * (Nat.factorial num_n)) :=
by sorry

end banana_arrangement_count_l3066_306676


namespace adjacent_points_probability_l3066_306624

/-- The number of points around the square -/
def n : ℕ := 12

/-- The number of pairs of adjacent points -/
def adjacent_pairs : ℕ := 12

/-- The total number of ways to choose 2 points from n points -/
def total_combinations : ℕ := n * (n - 1) / 2

/-- The probability of choosing two adjacent points -/
def probability : ℚ := adjacent_pairs / total_combinations

theorem adjacent_points_probability : probability = 2 / 11 := by
  sorry

end adjacent_points_probability_l3066_306624


namespace brick_width_calculation_l3066_306606

/-- Calculates the width of a brick given the dimensions of a wall and the number of bricks needed. -/
theorem brick_width_calculation (wall_length wall_width wall_height : ℝ)
  (brick_length brick_height : ℝ) (num_bricks : ℝ) :
  wall_length = 9 →
  wall_width = 5 →
  wall_height = 18.5 →
  brick_length = 0.21 →
  brick_height = 0.08 →
  num_bricks = 4955.357142857142 →
  ∃ (brick_width : ℝ), abs (brick_width - 0.295) < 0.001 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry


end brick_width_calculation_l3066_306606


namespace h_negative_a_equals_negative_two_l3066_306653

-- Define the functions
variable (f g h : ℝ → ℝ)
variable (a : ℝ)

-- Define the properties of the functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem h_negative_a_equals_negative_two 
  (hf_even : is_even f)
  (hg_odd : is_odd g)
  (hf_a : f a = 2)
  (hg_a : g a = 3)
  (hh : ∀ x, h x = f x + g x - 1) :
  h (-a) = -2 := by sorry

end h_negative_a_equals_negative_two_l3066_306653


namespace china_population_scientific_notation_l3066_306611

/-- Represents the population of China in millions at the end of 2021 -/
def china_population : ℝ := 1412.60

/-- Proves that the population of China expressed in scientific notation is 1.4126 × 10^9 -/
theorem china_population_scientific_notation :
  (china_population * 1000000 : ℝ) = 1.4126 * (10 ^ 9) := by
  sorry

end china_population_scientific_notation_l3066_306611


namespace triangle_altitude_excircle_radii_inequality_l3066_306639

/-- Given a triangle ABC with sides a, b, and c, altitude mc from vertex C to side AB,
    and radii ra and rb of the excircles opposite to vertices A and B respectively,
    prove that the altitude mc is at most the geometric mean of ra and rb. -/
theorem triangle_altitude_excircle_radii_inequality 
  (a b c mc ra rb : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ mc > 0 ∧ ra > 0 ∧ rb > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_altitude : mc = (2 * (a * b * c).sqrt) / (a + b + c)) 
  (h_excircle_a : ra = (a * b * c).sqrt / (b + c - a)) 
  (h_excircle_b : rb = (a * b * c).sqrt / (a + c - b)) : 
  mc ≤ Real.sqrt (ra * rb) :=
sorry

end triangle_altitude_excircle_radii_inequality_l3066_306639


namespace quadratic_root_relation_l3066_306671

/-- Given two quadratic equations, if the roots of the first are each three less than
    the roots of the second, then the constant term of the first equation is 24/5. -/
theorem quadratic_root_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 5*y^2 - 4*y - 9 = 0 ∧ x = y - 3) →
  c = 24/5 := by
sorry

end quadratic_root_relation_l3066_306671


namespace quadrilaterals_on_circle_l3066_306652

/-- The number of distinct convex quadrilaterals formed from points on a circle -/
theorem quadrilaterals_on_circle (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 4) :
  (Nat.choose n k) = 495 := by
  sorry

end quadrilaterals_on_circle_l3066_306652


namespace clubsuit_symmetry_forms_intersecting_lines_l3066_306672

-- Define the operation ♣
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Theorem statement
theorem clubsuit_symmetry_forms_intersecting_lines :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), clubsuit x y = clubsuit y x ↔ (x, y) ∈ l₁ ∪ l₂) ∧
    (l₁ ≠ l₂) ∧
    (∃ (p : ℝ × ℝ), p ∈ l₁ ∧ p ∈ l₂) :=
sorry


end clubsuit_symmetry_forms_intersecting_lines_l3066_306672


namespace percent_increase_in_sales_l3066_306600

theorem percent_increase_in_sales (sales_this_year sales_last_year : ℝ) 
  (h1 : sales_this_year = 460)
  (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 := by
  sorry

end percent_increase_in_sales_l3066_306600


namespace age_difference_l3066_306689

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 25 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 27 := by
sorry

end age_difference_l3066_306689


namespace no_solution_for_divisibility_l3066_306644

theorem no_solution_for_divisibility (n : ℕ) (hn : n ≥ 1) : ¬(9 ∣ (7^n + n^3)) := by
  sorry

end no_solution_for_divisibility_l3066_306644


namespace regular_octagon_side_length_l3066_306683

/-- A regular octagon is a polygon with 8 sides of equal length and 8 angles of equal measure. -/
structure RegularOctagon where
  sideLength : ℝ
  perimeter : ℝ

/-- The perimeter of a regular octagon is 8 times the length of one side. -/
def RegularOctagon.perimeterFormula (o : RegularOctagon) : ℝ :=
  8 * o.sideLength

theorem regular_octagon_side_length (o : RegularOctagon) 
    (h : o.perimeter = 23.6) : o.sideLength = 2.95 := by
  sorry

end regular_octagon_side_length_l3066_306683


namespace variance_transformation_l3066_306695

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable (a b : ℝ)

def variance (y : Fin n → ℝ) : ℝ := sorry

theorem variance_transformation (h1 : variance x = 3) 
  (h2 : variance (fun i => a * x i + b) = 12) : a = 2 ∨ a = -2 := by sorry

end variance_transformation_l3066_306695


namespace mechanic_work_hours_l3066_306622

/-- Calculates the number of hours a mechanic worked given the total cost, 
    cost of parts, and labor rate per minute. -/
theorem mechanic_work_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (num_parts : ℕ) 
  (labor_rate_per_minute : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_cost = 20) 
  (h3 : num_parts = 2) 
  (h4 : labor_rate_per_minute = 0.5) : 
  (total_cost - part_cost * num_parts) / (labor_rate_per_minute * 60) = 6 := by
sorry

end mechanic_work_hours_l3066_306622


namespace fruit_cost_theorem_l3066_306660

/-- Calculates the cost of remaining fruit after discount and loss -/
def remaining_fruit_cost (pear_price apple_price pineapple_price plum_price : ℚ)
                         (pear_qty apple_qty pineapple_qty plum_qty : ℕ)
                         (apple_discount : ℚ) (fruit_loss_ratio : ℚ) : ℚ :=
  let total_cost := pear_price * pear_qty + 
                    apple_price * apple_qty * (1 - apple_discount) + 
                    pineapple_price * pineapple_qty + 
                    plum_price * plum_qty
  total_cost * (1 - fruit_loss_ratio)

/-- The theorem to be proved -/
theorem fruit_cost_theorem : 
  remaining_fruit_cost 1.5 0.75 2 0.5 6 4 2 1 0.25 0.5 = 7.88 := by
  sorry

end fruit_cost_theorem_l3066_306660


namespace square_difference_ratio_l3066_306605

theorem square_difference_ratio : (1722^2 - 1715^2) / (1730^2 - 1705^2) = 7/25 := by
  sorry

end square_difference_ratio_l3066_306605


namespace equation_solution_l3066_306680

theorem equation_solution :
  ∃ x : ℝ, -((1 : ℝ) / 3) * x - 5 = 4 ∧ x = -27 := by
  sorry

end equation_solution_l3066_306680


namespace inequality_solution_set_l3066_306659

theorem inequality_solution_set : 
  ¬(∀ x : ℝ, -3 * x > 9 ↔ x < -3) :=
sorry

end inequality_solution_set_l3066_306659


namespace right_triangle_area_l3066_306679

theorem right_triangle_area (p : ℝ) (h : p > 0) : ∃ (x y z : ℝ),
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^2 + y^2 = z^2 ∧
  x + y + z = 3*p ∧
  x = z/2 ∧
  (1/2) * x * y = (p^2 * Real.sqrt 3) / 4 :=
by sorry

end right_triangle_area_l3066_306679


namespace geometric_progression_values_l3066_306617

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*p - 1) = |p - 8| * r ∧ (4*p + 5) = (2*p - 1) * r) ↔ 
  (p = -1 ∨ p = 39/8) := by
sorry

end geometric_progression_values_l3066_306617


namespace nail_polish_difference_l3066_306623

theorem nail_polish_difference (kim heidi karen : ℕ) : 
  kim = 12 →
  heidi = kim + 5 →
  karen + heidi = 25 →
  karen < kim →
  kim - karen = 4 := by
sorry

end nail_polish_difference_l3066_306623


namespace sequence_sum_99_100_l3066_306610

def sequence_term (n : ℕ) : ℚ :=
  let group := (n.sqrt : ℕ)
  let position := n - (group - 1) * group
  ↑(group + 1 - position) / position

theorem sequence_sum_99_100 : 
  sequence_term 99 + sequence_term 100 = 37 / 24 := by sorry

end sequence_sum_99_100_l3066_306610


namespace tangent_line_of_odd_function_l3066_306603

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a - 2) * x^2 + 2 * x

-- State the theorem
theorem tangent_line_of_odd_function (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is an odd function
  (f a 1 = 3) →                 -- f(1) = 3
  ∃ m b : ℝ, m = 5 ∧ b = -2 ∧
    ∀ x, (f a x - f a 1) = m * (x - 1) + b :=
by sorry

end tangent_line_of_odd_function_l3066_306603


namespace line_slope_l3066_306635

/-- The slope of the line 4x - 7y = 28 is 4/7 -/
theorem line_slope (x y : ℝ) : 4 * x - 7 * y = 28 → (y - (-4)) / (x - 0) = 4 / 7 := by
  sorry

end line_slope_l3066_306635


namespace inequality_proof_l3066_306619

theorem inequality_proof (n : ℕ+) : (2 * n.val ^ 2 + 3 * n.val + 1) ^ n.val ≥ 6 ^ n.val * (n.val.factorial) ^ 2 := by
  sorry

end inequality_proof_l3066_306619


namespace integral_equals_zero_l3066_306681

theorem integral_equals_zero : 
  ∫ x in (0 : ℝ)..1, (4 * Real.sqrt (1 - x) - Real.sqrt (3 * x + 1)) / 
    ((Real.sqrt (3 * x + 1) + 4 * Real.sqrt (1 - x)) * (3 * x + 1)^2) = 0 := by sorry

end integral_equals_zero_l3066_306681


namespace multiplication_puzzle_l3066_306612

theorem multiplication_puzzle :
  ∀ A B C K : ℕ,
    A ∈ Finset.range 10 →
    B ∈ Finset.range 10 →
    C ∈ Finset.range 10 →
    K ∈ Finset.range 10 →
    A < B →
    A ≠ B ∧ A ≠ C ∧ A ≠ K ∧ B ≠ C ∧ B ≠ K ∧ C ≠ K →
    (10 * A + C) * (10 * B + C) = 111 * K →
    K * 111 = 100 * K + 10 * K + K →
    A = 2 ∧ B = 3 ∧ C = 7 ∧ K = 9 := by
  sorry

end multiplication_puzzle_l3066_306612


namespace square_area_proof_l3066_306670

theorem square_area_proof (x : ℝ) : 
  (5 * x - 20 : ℝ) = (25 - 2 * x : ℝ) → 
  (5 * x - 20 : ℝ) > 0 → 
  ((5 * x - 20 : ℝ) * (5 * x - 20 : ℝ)) = 7225 / 49 := by
  sorry

end square_area_proof_l3066_306670


namespace inequality_not_always_true_l3066_306687

theorem inequality_not_always_true (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  ¬ (∀ (a b c : ℝ), a > 0 → b > 0 → a > b → c ≠ 0 → a / c > b / c) :=
by sorry

end inequality_not_always_true_l3066_306687


namespace expression_values_l3066_306640

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (a / abs a) + (b / abs b) + (c / abs c) + (d / abs d) + ((a * b * c * d) / abs (a * b * c * d))
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end expression_values_l3066_306640


namespace least_integer_absolute_value_inequality_l3066_306648

theorem least_integer_absolute_value_inequality :
  ∃ (x : ℤ), (3 * |x| + 4 < 19) ∧ (∀ (y : ℤ), y < x → 3 * |y| + 4 ≥ 19) :=
by
  sorry

end least_integer_absolute_value_inequality_l3066_306648


namespace min_value_implies_a_l3066_306663

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a - 1

theorem min_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = -2) → 
  a = -1 ∨ a = 2 := by sorry

end min_value_implies_a_l3066_306663


namespace candy_bar_cost_l3066_306696

theorem candy_bar_cost (candy_bars : ℕ) (lollipops : ℕ) (lollipop_cost : ℚ)
  (snow_shoveling_fraction : ℚ) (driveway_charge : ℚ) (driveways : ℕ) :
  candy_bars = 2 →
  lollipops = 4 →
  lollipop_cost = 1/4 →
  snow_shoveling_fraction = 1/6 →
  driveway_charge = 3/2 →
  driveways = 10 →
  (driveway_charge * driveways * snow_shoveling_fraction - lollipops * lollipop_cost) / candy_bars = 3/4 := by
  sorry

end candy_bar_cost_l3066_306696


namespace inequality_of_means_l3066_306666

theorem inequality_of_means (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : (x - y)^2 > 396*x*y) (h2 : 2.0804*x*y > x^2 + y^2) :
  1.01 * Real.sqrt (x*y) > (x + y)/2 ∧ (x + y)/2 > 100 * (2*x*y/(x + y)) := by
  sorry

end inequality_of_means_l3066_306666


namespace shems_earnings_l3066_306668

/-- Proves that Shem's earnings for an 8-hour workday is $80, given the conditions. -/
theorem shems_earnings (kem_hourly_rate : ℝ) (shem_multiplier : ℝ) (workday_hours : ℕ) :
  kem_hourly_rate = 4 →
  shem_multiplier = 2.5 →
  workday_hours = 8 →
  kem_hourly_rate * shem_multiplier * workday_hours = 80 := by
  sorry

#check shems_earnings

end shems_earnings_l3066_306668


namespace ninth_grade_maximizes_profit_l3066_306602

/-- Represents the profit function for a product with different quality grades. -/
def profit_function (k : ℕ) : ℝ :=
  let profit_per_piece := 8 + 2 * (k - 1)
  let pieces_produced := 60 - 3 * (k - 1)
  (profit_per_piece * pieces_produced : ℝ)

/-- Theorem stating that the 9th quality grade maximizes the profit. -/
theorem ninth_grade_maximizes_profit :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → profit_function k ≤ profit_function 9 := by
  sorry

#check ninth_grade_maximizes_profit

end ninth_grade_maximizes_profit_l3066_306602


namespace initial_persons_count_l3066_306601

/-- Represents the number of days to complete the work initially -/
def initial_days : ℕ := 18

/-- Represents the number of days worked before adding more persons -/
def days_before_addition : ℕ := 6

/-- Represents the number of persons added -/
def persons_added : ℕ := 4

/-- Represents the number of days to complete the remaining work after adding persons -/
def remaining_days : ℕ := 9

/-- Represents the total amount of work -/
def total_work : ℚ := 1

/-- Theorem stating the initial number of persons working on the project -/
theorem initial_persons_count : 
  ∃ (P : ℕ), 
    (P * initial_days : ℚ) * total_work = 
    (P * days_before_addition + (P + persons_added) * remaining_days : ℚ) * total_work ∧ 
    P = 12 := by
  sorry

end initial_persons_count_l3066_306601


namespace seating_arrangement_l3066_306665

theorem seating_arrangement (total_people : ℕ) (rows_of_nine : ℕ) (rows_of_ten : ℕ) : 
  total_people = 54 →
  total_people = 9 * rows_of_nine + 10 * rows_of_ten →
  rows_of_nine > 0 →
  rows_of_ten = 0 := by
sorry

end seating_arrangement_l3066_306665


namespace jeff_average_skips_l3066_306699

-- Define the number of rounds
def num_rounds : ℕ := 4

-- Define Sam's skips per round
def sam_skips : ℕ := 16

-- Define Jeff's skips for each round
def jeff_round1 : ℕ := sam_skips - 1
def jeff_round2 : ℕ := sam_skips - 3
def jeff_round3 : ℕ := sam_skips + 4
def jeff_round4 : ℕ := sam_skips / 2

-- Define Jeff's total skips
def jeff_total : ℕ := jeff_round1 + jeff_round2 + jeff_round3 + jeff_round4

-- Theorem to prove
theorem jeff_average_skips :
  jeff_total / num_rounds = 14 := by sorry

end jeff_average_skips_l3066_306699


namespace twenty_two_oclock_is_ten_pm_l3066_306673

/-- Converts 24-hour time format to 12-hour time format -/
def convert_24_to_12 (hour : ℕ) : ℕ × String :=
  if hour < 12 then (if hour = 0 then 12 else hour, "AM")
  else ((if hour = 12 then 12 else hour - 12), "PM")

/-- Theorem stating that 22:00 in 24-hour format is equivalent to 10:00 PM in 12-hour format -/
theorem twenty_two_oclock_is_ten_pm :
  convert_24_to_12 22 = (10, "PM") :=
sorry

end twenty_two_oclock_is_ten_pm_l3066_306673


namespace f_of_5_equals_22_l3066_306669

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 2

-- State the theorem
theorem f_of_5_equals_22 : f 5 = 22 := by
  sorry

end f_of_5_equals_22_l3066_306669


namespace original_rectangle_area_l3066_306686

/-- Given a rectangle whose dimensions are doubled to form a new rectangle with an area of 32 square meters, 
    prove that the area of the original rectangle is 8 square meters. -/
theorem original_rectangle_area (original_width original_height : ℝ) : 
  original_width > 0 → 
  original_height > 0 → 
  (2 * original_width) * (2 * original_height) = 32 → 
  original_width * original_height = 8 := by
sorry

end original_rectangle_area_l3066_306686


namespace solution_value_l3066_306664

theorem solution_value (a b : ℝ) : 
  (2 * a + b = 3) → (6 * a + 3 * b - 1 = 8) := by
  sorry

end solution_value_l3066_306664


namespace factorization_problem1_factorization_problem2_l3066_306661

-- Problem 1
theorem factorization_problem1 (x y : ℝ) :
  x^3 + 2*x^2*y + x*y^2 = x*(x + y)^2 := by sorry

-- Problem 2
theorem factorization_problem2 (m n : ℝ) :
  4*m^2 - n^2 - 4*m + 1 = (2*m - 1 + n)*(2*m - 1 - n) := by sorry

end factorization_problem1_factorization_problem2_l3066_306661


namespace baker_cakes_l3066_306692

/-- Calculates the remaining number of cakes after buying and selling -/
def remaining_cakes (initial bought sold : ℕ) : ℕ :=
  initial + bought - sold

/-- Theorem: The number of cakes Baker still has is 190 -/
theorem baker_cakes : remaining_cakes 173 103 86 = 190 := by
  sorry

end baker_cakes_l3066_306692


namespace triangle_side_length_l3066_306677

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b = 6 ∧
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 ∧
  A = π/3 →
  a = 2 * Real.sqrt 7 := by
sorry

end triangle_side_length_l3066_306677


namespace current_rate_calculation_l3066_306637

/-- Given a boat traveling downstream, calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ)             -- Speed of the boat in still water (km/hr)
  (distance : ℝ)               -- Distance traveled downstream (km)
  (time : ℝ)                   -- Time taken for the downstream journey (hr)
  (h1 : boat_speed = 20)       -- The boat's speed in still water is 20 km/hr
  (h2 : distance = 10)         -- The distance traveled downstream is 10 km
  (h3 : time = 24 / 60)        -- The time taken is 24 minutes, converted to hours
  : ∃ (current_rate : ℝ), 
    distance = (boat_speed + current_rate) * time ∧ 
    current_rate = 5 := by
  sorry

end current_rate_calculation_l3066_306637


namespace rowing_speed_l3066_306650

theorem rowing_speed (downstream_distance upstream_distance : ℝ)
                     (total_time : ℝ)
                     (current_speed : ℝ)
                     (h1 : downstream_distance = 3.5)
                     (h2 : upstream_distance = 3.5)
                     (h3 : total_time = 5/3)
                     (h4 : current_speed = 2) :
  ∃ still_water_speed : ℝ,
    still_water_speed = 5 ∧
    downstream_distance / (still_water_speed + current_speed) +
    upstream_distance / (still_water_speed - current_speed) = total_time :=
by sorry

end rowing_speed_l3066_306650


namespace problem_solution_l3066_306646

theorem problem_solution :
  ∀ (a b c : ℝ),
    (∃ (x : ℝ), x > 0 ∧ (a - 2)^2 = x ∧ (7 - 2*a)^2 = x) →
    ((3*b + 1)^(1/3) = -2) →
    (c = ⌊Real.sqrt 39⌋) →
    (a = 5 ∧ b = -3 ∧ c = 6 ∧ 
     (∃ (y : ℝ), y^2 = 5*a + 2*b - c ∧ (y = Real.sqrt 13 ∨ y = -Real.sqrt 13))) :=
by sorry

end problem_solution_l3066_306646


namespace pascal_triangle_entries_l3066_306631

/-- The number of entries in the n-th row of Pascal's Triangle -/
def entriesInRow (n : ℕ) : ℕ := n + 1

/-- The sum of entries in the first n rows of Pascal's Triangle -/
def sumOfEntries (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem pascal_triangle_entries : sumOfEntries 30 = 465 := by
  sorry

end pascal_triangle_entries_l3066_306631


namespace arithmetic_calculations_l3066_306638

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) + (-15) = 8) ∧ 
  (-2^3 + (-5)^2 * (2/5) - |(-3)| = -1) := by
  sorry

end arithmetic_calculations_l3066_306638


namespace intersection_of_A_and_B_l3066_306658

-- Define set A
def A : Set ℝ := {y | ∃ x, y = |x|}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 1 - 2*x - x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 2} := by sorry

end intersection_of_A_and_B_l3066_306658


namespace fraction_comparison_l3066_306613

theorem fraction_comparison :
  (373737 : ℚ) / 777777 = 37 / 77 ∧ (41 : ℚ) / 61 < 411 / 611 := by
  sorry

end fraction_comparison_l3066_306613


namespace project_solution_l3066_306685

/-- Represents the time (in days) required for a person to complete the project alone. -/
structure ProjectTime where
  personA : ℝ
  personB : ℝ
  personC : ℝ

/-- Defines the conditions of the engineering project. -/
def ProjectConditions (t : ProjectTime) : Prop :=
  -- Person B works alone for 4 days
  4 / t.personB +
  -- Persons A and C work together for 6 days
  6 * (1 / t.personA + 1 / t.personC) +
  -- Person A completes the remaining work in 9 days
  9 / t.personA = 1 ∧
  -- Work completed by Person B is 1/3 of the work completed by Person A
  t.personB = 3 * t.personA ∧
  -- Work completed by Person C is 2 times the work completed by Person B
  t.personC = t.personB / 2

/-- Theorem stating the solution to the engineering project problem. -/
theorem project_solution :
  ∃ t : ProjectTime, ProjectConditions t ∧ t.personA = 30 ∧ t.personB = 24 ∧ t.personC = 18 :=
by sorry

end project_solution_l3066_306685


namespace equation_solution_l3066_306651

theorem equation_solution :
  let x : ℝ := 32
  let equation (number : ℝ) := 35 - (23 - (15 - x)) = 12 * 2 / (1 / number)
  ∃ (number : ℝ), equation number ∧ number = -4.8 := by
  sorry

end equation_solution_l3066_306651


namespace f_min_value_l3066_306627

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The minimum value of f(x) is 2 -/
theorem f_min_value : ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 2 := by
  sorry

end f_min_value_l3066_306627


namespace companion_pair_example_companion_pair_value_companion_pair_expression_l3066_306675

/-- Definition of companion rational number pairs -/
def is_companion_pair (a b : ℚ) : Prop := a - b = a * b + 1

/-- Theorem 1: (-1/2, -3) is a companion rational number pair -/
theorem companion_pair_example : is_companion_pair (-1/2) (-3) := by sorry

/-- Theorem 2: When (x+1, 5) is a companion rational number pair, x = -5/2 -/
theorem companion_pair_value (x : ℚ) : 
  is_companion_pair (x + 1) 5 → x = -5/2 := by sorry

/-- Theorem 3: For any companion rational number pair (a,b), 
    3ab-a+1/2(a+b-5ab)+1 = 1/2 -/
theorem companion_pair_expression (a b : ℚ) :
  is_companion_pair a b → 3*a*b - a + 1/2*(a+b-5*a*b) + 1 = 1/2 := by sorry

end companion_pair_example_companion_pair_value_companion_pair_expression_l3066_306675


namespace circle_segment_area_l3066_306698

theorem circle_segment_area (r chord_length intersection_dist : ℝ) 
  (hr : r = 45)
  (hc : chord_length = 84)
  (hi : intersection_dist = 15) : 
  ∃ (m n d : ℝ), 
    (m = 506.25 ∧ n = 1012.5 ∧ d = 1) ∧
    (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) :=
by sorry

end circle_segment_area_l3066_306698


namespace simplify_trigonometric_expression_l3066_306667

theorem simplify_trigonometric_expression :
  let x := Real.sin (15 * π / 180)
  let y := Real.cos (15 * π / 180)
  Real.sqrt (x^4 + 4 * y^2) - Real.sqrt (y^4 + 4 * x^2) = (1 / 2) * Real.sqrt 3 := by
  sorry

end simplify_trigonometric_expression_l3066_306667


namespace sum_of_squares_l3066_306608

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4) 
  (h2 : a/x + b/y + c/z = 3) : 
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 10 := by
sorry

end sum_of_squares_l3066_306608


namespace george_blocks_count_l3066_306649

/-- Calculates the total number of blocks given the number of large boxes, small boxes per large box,
    blocks per small box, and individual blocks outside the boxes. -/
def totalBlocks (largBoxes smallBoxesPerLarge blocksPerSmall individualBlocks : ℕ) : ℕ :=
  largBoxes * smallBoxesPerLarge * blocksPerSmall + individualBlocks

/-- Proves that George has 366 blocks in total -/
theorem george_blocks_count :
  totalBlocks 5 8 9 6 = 366 := by
  sorry

end george_blocks_count_l3066_306649


namespace A_intersect_B_eq_singleton_one_l3066_306643

def A : Set ℝ := {-1, 1, 1/2, 3}

def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem A_intersect_B_eq_singleton_one : A ∩ B = {1} := by sorry

end A_intersect_B_eq_singleton_one_l3066_306643


namespace square_root_equation_l3066_306656

theorem square_root_equation (a b : ℝ) 
  (h1 : Real.sqrt a = 2*b - 3)
  (h2 : Real.sqrt a = 3*b + 8) : 
  a = 25 ∧ b = -1 := by
sorry

end square_root_equation_l3066_306656


namespace complete_square_result_l3066_306645

/-- Given a quadratic equation 16x^2 + 32x - 512 = 0, prove that when solved by completing the square
    to the form (x + r)^2 = s, the value of s is 33. -/
theorem complete_square_result (x r s : ℝ) : 
  (16 * x^2 + 32 * x - 512 = 0) →
  ((x + r)^2 = s) →
  (s = 33) := by
sorry

end complete_square_result_l3066_306645


namespace exponent_and_square_of_negative_two_l3066_306674

theorem exponent_and_square_of_negative_two :
  (-2^2 = -4) ∧ ((-2)^3 = -8) ∧ ((-2)^2 = 4) := by
  sorry

end exponent_and_square_of_negative_two_l3066_306674


namespace abs_equation_solution_l3066_306684

theorem abs_equation_solution : ∃! x : ℝ, |x - 3| = 5 - 2*x := by
  sorry

end abs_equation_solution_l3066_306684


namespace fishing_catch_difference_l3066_306636

theorem fishing_catch_difference (father_catch son_catch transfer : ℚ) : 
  (father_catch - transfer = son_catch + transfer) →
  (father_catch + transfer = 2 * (son_catch - transfer)) →
  (father_catch - son_catch) / son_catch = 2/5 := by
  sorry

end fishing_catch_difference_l3066_306636


namespace triangle_height_l3066_306634

theorem triangle_height (C b area h : Real) : 
  C = π / 3 → 
  b = 4 → 
  area = 2 * Real.sqrt 3 → 
  area = (1 / 2) * b * h * Real.sin C → 
  h = 2 * Real.sqrt 3 := by
sorry

end triangle_height_l3066_306634


namespace circle_area_with_diameter_10_l3066_306682

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  ∀ (circle_area : ℝ → ℝ) (pi : ℝ),
  (∀ r, circle_area r = pi * r^2) →
  circle_area 5 = 25 * pi := by
  sorry

end circle_area_with_diameter_10_l3066_306682


namespace line_parameterization_l3066_306621

/-- Given a line y = 2x - 30 parameterized by (x,y) = (f(t), 20t - 14), prove that f(t) = 10t + 8 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 2 * f t - 30 = 20 * t - 14) → 
  (∀ t : ℝ, f t = 10 * t + 8) := by
sorry

end line_parameterization_l3066_306621


namespace circle_sum_of_center_and_radius_l3066_306655

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*y - 4 = -y^2 + 12*x - 12

-- Define the center and radius of the circle
def circle_center_radius (a' b' r' : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a')^2 + (y - b')^2 = r'^2

-- Theorem statement
theorem circle_sum_of_center_and_radius :
  ∃ a' b' r', circle_center_radius a' b' r' ∧ a' + b' + r' = 3 + Real.sqrt 37 :=
sorry

end circle_sum_of_center_and_radius_l3066_306655


namespace middle_zero_product_l3066_306604

theorem middle_zero_product (a b c d : ℕ) : ∃ (x y z w : ℕ), 
  (x ≠ 0 ∧ z ≠ 0 ∧ y ≠ 0 ∧ w ≠ 0) ∧ 
  (100 * x + 0 * 10 + y) * z = 100 * a + 0 * 10 + b ∧
  (100 * x + 0 * 10 + y) * w = 100 * c + d * 10 + e ∧
  d ≠ 0 := by
  sorry

end middle_zero_product_l3066_306604


namespace platform_length_calculation_l3066_306647

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmph = 80 →
  crossing_time = 22 →
  ∃ (platform_length : ℝ), abs (platform_length - 288.84) < 0.01 := by
  sorry


end platform_length_calculation_l3066_306647


namespace max_value_implies_t_equals_one_l3066_306609

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| ≤ 2) →
  (∃ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| = 2) →
  t = 1 := by
sorry

end max_value_implies_t_equals_one_l3066_306609


namespace disco_probabilities_l3066_306616

/-- Represents the content of a music case -/
structure MusicCase where
  disco : ℕ
  techno : ℕ

/-- The probability of selecting a disco tape from a given music case -/
def prob_disco (case : MusicCase) : ℚ :=
  case.disco / (case.disco + case.techno)

/-- The probability of selecting a second disco tape when the first is returned -/
def prob_disco_returned (case : MusicCase) : ℚ :=
  prob_disco case

/-- The probability of selecting a second disco tape when the first is not returned -/
def prob_disco_not_returned (case : MusicCase) : ℚ :=
  (case.disco - 1) / (case.disco + case.techno - 1)

/-- Theorem stating the probabilities for the given scenario -/
theorem disco_probabilities (case : MusicCase) (h : case = ⟨20, 10⟩) :
  prob_disco case = 2/3 ∧
  prob_disco_returned case = 2/3 ∧
  prob_disco_not_returned case = 19/29 := by
  sorry


end disco_probabilities_l3066_306616


namespace total_balloons_l3066_306697

/-- Given a set of balloons divided into 7 equal groups with 5 balloons in each group,
    the total number of balloons is 35. -/
theorem total_balloons (num_groups : ℕ) (balloons_per_group : ℕ) 
  (h1 : num_groups = 7) (h2 : balloons_per_group = 5) : 
  num_groups * balloons_per_group = 35 := by
  sorry

end total_balloons_l3066_306697


namespace fifth_day_distance_l3066_306633

def running_distance (day : ℕ) : ℕ :=
  2 + (day - 1)

theorem fifth_day_distance : running_distance 5 = 6 := by
  sorry

end fifth_day_distance_l3066_306633


namespace complex_magnitude_problem_l3066_306629

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I * Real.sqrt 3) * z = Complex.I) :
  Complex.abs z = 1 / 2 := by
  sorry

end complex_magnitude_problem_l3066_306629


namespace fibonacci_seventh_term_l3066_306632

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_seventh_term : fibonacci 6 = 13 := by
  sorry

end fibonacci_seventh_term_l3066_306632


namespace sheep_problem_l3066_306641

theorem sheep_problem (mary_initial : ℕ) (bob_multiplier : ℕ) (bob_additional : ℕ) (difference : ℕ) : 
  mary_initial = 300 →
  bob_multiplier = 2 →
  bob_additional = 35 →
  difference = 69 →
  (mary_initial + (bob_multiplier * mary_initial + bob_additional - difference - mary_initial)) = 566 :=
by
  sorry

end sheep_problem_l3066_306641


namespace divisibility_property_l3066_306618

theorem divisibility_property (n : ℕ) : ∃ k : ℤ, 1 + ⌊(3 + Real.sqrt 5)^n⌋ = k * 2^n := by
  sorry

end divisibility_property_l3066_306618


namespace max_plots_for_given_garden_l3066_306642

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : ℝ
  length : ℝ

/-- Represents the constraints for partitioning the garden -/
structure PartitionConstraints where
  fencing_available : ℝ
  min_plots_per_row : ℕ

/-- Calculates the maximum number of square plots given garden dimensions and constraints -/
def max_square_plots (garden : GardenDimensions) (constraints : PartitionConstraints) : ℕ :=
  sorry

/-- Theorem stating the maximum number of square plots for the given problem -/
theorem max_plots_for_given_garden :
  let garden := GardenDimensions.mk 30 60
  let constraints := PartitionConstraints.mk 3000 4
  max_square_plots garden constraints = 1250 := by
  sorry

end max_plots_for_given_garden_l3066_306642


namespace cyclist_speed_problem_l3066_306620

theorem cyclist_speed_problem (v : ℝ) :
  v > 0 →
  (20 : ℝ) / (9 / v + 11 / 9) = 9.8019801980198 →
  ∃ ε > 0, |v - 11.03| < ε :=
by
  sorry

end cyclist_speed_problem_l3066_306620


namespace find_b_value_l3066_306625

theorem find_b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end find_b_value_l3066_306625


namespace circle_intersection_symmetry_l3066_306694

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (c1 c2 : Circle) (A B : ℝ × ℝ) : Prop :=
  -- The circles intersect at points A and B
  A ∈ {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2} ∧
  A ∈ {p : ℝ × ℝ | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2} ∧
  B ∈ {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2} ∧
  B ∈ {p : ℝ × ℝ | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2} ∧
  -- Centers of both circles are on the x-axis
  c1.center.2 = 0 ∧
  c2.center.2 = 0 ∧
  -- Coordinates of point A are (-3, 2)
  A = (-3, 2)

-- Theorem statement
theorem circle_intersection_symmetry (c1 c2 : Circle) (A B : ℝ × ℝ) :
  problem_setup c1 c2 A B → B = (-3, -2) :=
by
  sorry

end circle_intersection_symmetry_l3066_306694


namespace greatest_divisor_four_consecutive_integers_l3066_306626

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
sorry

end greatest_divisor_four_consecutive_integers_l3066_306626


namespace print_output_l3066_306615

-- Define a simple output function to represent PRINT
def print (a : ℕ) (b : ℕ) : String :=
  s!"{a}, {b}"

-- Theorem statement
theorem print_output : print 3 (3 + 2) = "3, 5" := by
  sorry

end print_output_l3066_306615


namespace sin_shift_equivalence_l3066_306678

theorem sin_shift_equivalence :
  ∀ x : ℝ, Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end sin_shift_equivalence_l3066_306678
