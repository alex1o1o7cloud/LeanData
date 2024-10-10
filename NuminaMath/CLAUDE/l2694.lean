import Mathlib

namespace binomial_6_choose_3_l2694_269445

theorem binomial_6_choose_3 : Nat.choose 6 3 = 20 := by
  sorry

end binomial_6_choose_3_l2694_269445


namespace arithmetic_sequence_average_l2694_269463

/-- The average of an arithmetic sequence with 21 terms, 
    starting at -180 and ending at 180, with a common difference of 6, is 0. -/
theorem arithmetic_sequence_average : 
  let first_term : ℤ := -180
  let last_term : ℤ := 180
  let num_terms : ℕ := 21
  let common_diff : ℤ := 6
  let sequence := fun i => first_term + (i : ℤ) * common_diff
  (first_term + last_term) / 2 = 0 ∧ 
  last_term = first_term + (num_terms - 1 : ℕ) * common_diff :=
by sorry

end arithmetic_sequence_average_l2694_269463


namespace american_flag_problem_l2694_269474

theorem american_flag_problem (total_stripes : ℕ) (total_red_stripes : ℕ) : 
  total_stripes = 13 →
  total_red_stripes = 70 →
  (total_stripes - 1) / 2 + 1 = 7 →
  total_red_stripes / ((total_stripes - 1) / 2 + 1) = 10 := by
sorry

end american_flag_problem_l2694_269474


namespace tv_cost_l2694_269480

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 880 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 220 := by
sorry

end tv_cost_l2694_269480


namespace unique_prime_with_square_divisor_sum_l2694_269443

theorem unique_prime_with_square_divisor_sum : 
  ∃! p : ℕ, Prime p ∧ 
  ∃ n : ℕ, (1 + p + p^2 + p^3 + p^4 : ℕ) = n^2 := by
  sorry

end unique_prime_with_square_divisor_sum_l2694_269443


namespace g_of_3_eq_200_l2694_269439

-- Define the function g
def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

-- Theorem statement
theorem g_of_3_eq_200 : g 3 = 200 := by
  sorry

end g_of_3_eq_200_l2694_269439


namespace factorization_yx_squared_minus_y_l2694_269415

theorem factorization_yx_squared_minus_y (x y : ℝ) : y * x^2 - y = y * (x + 1) * (x - 1) := by
  sorry

end factorization_yx_squared_minus_y_l2694_269415


namespace smallest_positive_multiple_of_45_l2694_269405

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l2694_269405


namespace undefined_fraction_l2694_269438

theorem undefined_fraction (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b - 2) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 := by
sorry

end undefined_fraction_l2694_269438


namespace initial_soldiers_count_l2694_269422

theorem initial_soldiers_count (provisions : ℝ) : ∃ (initial_soldiers : ℕ),
  provisions = initial_soldiers * 3 * 30 ∧
  provisions = (initial_soldiers + 528) * 2.5 * 25 ∧
  initial_soldiers = 1200 := by
sorry

end initial_soldiers_count_l2694_269422


namespace binomial_product_l2694_269465

theorem binomial_product (x : ℝ) : (2*x^2 + 3*x - 4)*(x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 := by
  sorry

end binomial_product_l2694_269465


namespace min_value_of_f_l2694_269416

/-- The function f(x) = 2x^3 - 6x^2 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -37) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ -37) := by
  sorry

end min_value_of_f_l2694_269416


namespace mystery_discount_rate_l2694_269431

/-- Represents the discount rate for books -/
structure DiscountRate :=
  (biography : ℝ)
  (mystery : ℝ)

/-- Represents the problem parameters -/
structure BookstoreParams :=
  (biography_price : ℝ)
  (mystery_price : ℝ)
  (biography_count : ℕ)
  (mystery_count : ℕ)
  (total_savings : ℝ)
  (total_discount_rate : ℝ)

/-- Theorem stating that given the problem conditions, the discount rate on mysteries is 37.5% -/
theorem mystery_discount_rate 
  (params : BookstoreParams)
  (h1 : params.biography_price = 20)
  (h2 : params.mystery_price = 12)
  (h3 : params.biography_count = 5)
  (h4 : params.mystery_count = 3)
  (h5 : params.total_savings = 19)
  (h6 : params.total_discount_rate = 43)
  : ∃ (d : DiscountRate), 
    d.biography + d.mystery = params.total_discount_rate ∧ 
    params.biography_count * params.biography_price * (d.biography / 100) + 
    params.mystery_count * params.mystery_price * (d.mystery / 100) = params.total_savings ∧
    d.mystery = 37.5 := by
  sorry

end mystery_discount_rate_l2694_269431


namespace ellipse_major_axis_length_l2694_269442

/-- The length of the major axis of an ellipse with given foci and tangent to x-axis -/
theorem ellipse_major_axis_length : 
  let f1 : ℝ × ℝ := (5, 15)
  let f2 : ℝ × ℝ := (40, 45)
  ∀ (E : Set (ℝ × ℝ)), 
    (∀ p ∈ E, dist p f1 + dist p f2 = dist p f1 + dist p f2) →  -- E is an ellipse with foci f1 and f2
    (∃ x, (x, 0) ∈ E) →  -- E is tangent to x-axis
    (∃ a : ℝ, ∀ p ∈ E, dist p f1 + dist p f2 = 2 * a) →  -- Definition of ellipse
    2 * (dist f1 f2) = 10 * Real.sqrt 193 :=
by
  sorry


end ellipse_major_axis_length_l2694_269442


namespace red_ball_probability_l2694_269478

/-- The probability of selecting a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- Theorem: The probability of selecting a red ball from a bag with 15 balls, 
    of which 3 are red, is 1/5 -/
theorem red_ball_probability :
  probability_red_ball 15 3 = 1 / 5 := by
  sorry

#eval probability_red_ball 15 3

end red_ball_probability_l2694_269478


namespace honey_ratio_proof_l2694_269429

/-- Given the conditions of James' honey production and jar requirements, 
    prove that the ratio of honey his friend is bringing jars for to the total honey produced is 1:2 -/
theorem honey_ratio_proof (hives : ℕ) (honey_per_hive : ℝ) (jar_capacity : ℝ) (james_jars : ℕ) 
  (h1 : hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : jar_capacity = 0.5)
  (h4 : james_jars = 100) :
  (↑james_jars : ℝ) / ((↑hives * honey_per_hive) / jar_capacity) = 1 / 2 :=
by sorry

end honey_ratio_proof_l2694_269429


namespace ln_product_eq_sum_of_ln_l2694_269488

-- Define the formal power series type
def FormalPowerSeries (α : Type*) := ℕ → α

-- Define the logarithm operation for formal power series
noncomputable def Ln (f : FormalPowerSeries ℝ) : FormalPowerSeries ℝ := sorry

-- Define the multiplication operation for formal power series
def mul (f g : FormalPowerSeries ℝ) : FormalPowerSeries ℝ := sorry

-- Theorem statement
theorem ln_product_eq_sum_of_ln 
  (f h : FormalPowerSeries ℝ) 
  (hf : f 0 = 1) 
  (hh : h 0 = 1) : 
  Ln (mul f h) = λ n => (Ln f n) + (Ln h n) := by sorry

end ln_product_eq_sum_of_ln_l2694_269488


namespace blue_shirts_count_l2694_269401

/-- Represents the number of people at a school dance --/
structure DanceAttendance where
  boys : ℕ
  girls : ℕ
  teachers : ℕ

/-- Calculates the number of people wearing blue shirts at the dance --/
def blueShirts (attendance : DanceAttendance) : ℕ :=
  let blueShirtedBoys := (attendance.boys * 20) / 100
  let blueShirtedMaleTeachers := (attendance.teachers * 25) / 100
  blueShirtedBoys + blueShirtedMaleTeachers

/-- Theorem stating the number of people wearing blue shirts at the dance --/
theorem blue_shirts_count (attendance : DanceAttendance) :
  attendance.boys * 4 = attendance.girls * 3 →
  attendance.teachers * 9 = attendance.boys + attendance.girls →
  attendance.girls = 108 →
  blueShirts attendance = 21 := by
  sorry

#eval blueShirts { boys := 81, girls := 108, teachers := 21 }

end blue_shirts_count_l2694_269401


namespace quadratic_equation_roots_l2694_269469

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end quadratic_equation_roots_l2694_269469


namespace equation_solutions_l2694_269476

def solutions_7 : Set (ℤ × ℤ) := {(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)}
def solutions_25 : Set (ℤ × ℤ) := {(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)}

theorem equation_solutions (a b : ℤ) :
  (1 / a + 1 / b = 1 / 7 → (a, b) ∈ solutions_7) ∧
  (1 / a + 1 / b = 1 / 25 → (a, b) ∈ solutions_25) := by
  sorry

end equation_solutions_l2694_269476


namespace base_conversion_sum_l2694_269487

-- Define the base conversion function
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def n1 : Nat := to_base_10 [2, 1, 4] 8
def n2 : Nat := to_base_10 [3, 2] 5
def n3 : Nat := to_base_10 [3, 4, 3] 9
def n4 : Nat := to_base_10 [1, 3, 3] 4

-- State the theorem
theorem base_conversion_sum :
  (n1 : ℚ) / n2 + (n3 : ℚ) / n4 = 9134 / 527 := by sorry

end base_conversion_sum_l2694_269487


namespace magic_square_sum_l2694_269407

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : Fin 3 → Fin 3 → ℕ
  sum_row : ∀ i, (Finset.univ.sum (λ j => a i j)) = (Finset.univ.sum (λ j => a 0 j))
  sum_col : ∀ j, (Finset.univ.sum (λ i => a i j)) = (Finset.univ.sum (λ j => a 0 j))
  sum_diag1 : (Finset.univ.sum (λ i => a i i)) = (Finset.univ.sum (λ j => a 0 j))
  sum_diag2 : (Finset.univ.sum (λ i => a i (2 - i))) = (Finset.univ.sum (λ j => a 0 j))

/-- The theorem to be proved -/
theorem magic_square_sum (s : MagicSquare) 
  (h1 : s.a 0 0 = 25)
  (h2 : s.a 0 2 = 23)
  (h3 : s.a 1 0 = 18)
  (h4 : s.a 2 1 = 22) :
  s.a 1 2 + s.a 0 1 = 45 := by
  sorry

end magic_square_sum_l2694_269407


namespace train_passing_time_l2694_269456

/-- The time taken for a train to pass a telegraph post -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 90 →
  train_speed_kmh = 36 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end train_passing_time_l2694_269456


namespace sum_of_even_coefficients_l2694_269454

theorem sum_of_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x - 1) * (x + 1)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
                                   a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ = 1 := by
sorry

end sum_of_even_coefficients_l2694_269454


namespace smallest_n_divisible_by_1000_l2694_269430

theorem smallest_n_divisible_by_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬(1000 ∣ (m+1)*(m+2)*(m+3)*(m+4))) ∧ 
  (1000 ∣ (n+1)*(n+2)*(n+3)*(n+4)) ∧ n = 121 := by
  sorry

end smallest_n_divisible_by_1000_l2694_269430


namespace jim_journey_distance_l2694_269447

/-- The total distance of Jim's journey -/
def total_distance (miles_driven : ℕ) (miles_remaining : ℕ) : ℕ :=
  miles_driven + miles_remaining

/-- Theorem: The total distance of Jim's journey is 1200 miles -/
theorem jim_journey_distance :
  total_distance 768 432 = 1200 := by
  sorry

end jim_journey_distance_l2694_269447


namespace function_inequality_condition_l2694_269423

theorem function_inequality_condition (f : ℝ → ℝ) (a c : ℝ) :
  (∀ x, f x = 2 * x + 3) →
  a > 0 →
  c > 0 →
  (∀ x, |x + 5| < c → |f x + 5| < a) ↔
  c > a / 2 := by sorry

end function_inequality_condition_l2694_269423


namespace number_of_girls_in_school_l2694_269483

/-- Proves the number of girls in a school given specific conditions --/
theorem number_of_girls_in_school (total_students : ℕ) 
  (avg_age_boys avg_age_girls avg_age_school : ℚ) :
  total_students = 604 →
  avg_age_boys = 12 →
  avg_age_girls = 11 →
  avg_age_school = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 151 ∧ 
    (num_girls : ℚ) * avg_age_girls + (total_students - num_girls : ℚ) * avg_age_boys = 
      total_students * avg_age_school :=
by
  sorry

end number_of_girls_in_school_l2694_269483


namespace birds_in_trees_l2694_269496

theorem birds_in_trees (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  stones = 40 →
  trees = 3 * stones →
  birds = 2 * (trees + stones) →
  birds = 400 := by
sorry

end birds_in_trees_l2694_269496


namespace transformation_interval_l2694_269424

theorem transformation_interval (x : ℝ) :
  x ∈ Set.Icc 0 1 → (8 * x - 2) ∈ Set.Icc (-2) 6 := by
  sorry

end transformation_interval_l2694_269424


namespace range_of_g_l2694_269457

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by sorry

end range_of_g_l2694_269457


namespace b_55_mod_56_l2694_269402

/-- b_n is the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The theorem states that b_55 mod 56 = 0 -/
theorem b_55_mod_56 : b 55 % 56 = 0 := by sorry

end b_55_mod_56_l2694_269402


namespace max_value_theorem_l2694_269455

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 = 4) :
  ∃ (max : ℝ), max = (1 + Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z = x*y / (x + 2*y - 2) → z ≤ max :=
sorry

end max_value_theorem_l2694_269455


namespace complex_equation_solution_l2694_269404

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 1) : 
  z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end complex_equation_solution_l2694_269404


namespace collatz_100th_term_l2694_269417

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => 6
  | m + 1 => collatz (collatzSequence n m)

theorem collatz_100th_term :
  collatzSequence 100 99 = 4 := by sorry

end collatz_100th_term_l2694_269417


namespace abs_two_implies_two_or_neg_two_l2694_269499

theorem abs_two_implies_two_or_neg_two (x : ℝ) : |x| = 2 → x = 2 ∨ x = -2 := by
  sorry

end abs_two_implies_two_or_neg_two_l2694_269499


namespace number_puzzle_solution_l2694_269461

theorem number_puzzle_solution : 
  ∃ x : ℚ, 3 * (2 * x + 7) = 99 ∧ x = 13 := by sorry

end number_puzzle_solution_l2694_269461


namespace choose_cooks_l2694_269479

theorem choose_cooks (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end choose_cooks_l2694_269479


namespace gcd_of_polynomial_and_multiple_l2694_269450

theorem gcd_of_polynomial_and_multiple (y : ℤ) : 
  (∃ k : ℤ, y = 30492 * k) →
  Int.gcd ((3*y+4)*(8*y+3)*(11*y+5)*(y+11)) y = 660 := by
  sorry

end gcd_of_polynomial_and_multiple_l2694_269450


namespace least_distinct_values_l2694_269452

/-- Given a list of 2023 positive integers with a unique mode occurring exactly 15 times,
    the least number of distinct values in the list is 145. -/
theorem least_distinct_values (l : List ℕ+) 
  (h_length : l.length = 2023)
  (h_unique_mode : ∃! m : ℕ+, l.count m = 15 ∧ ∀ n : ℕ+, l.count n ≤ 15) :
  (l.toFinset.card : ℕ) = 145 ∧ 
  ∀ k : ℕ, k < 145 → ¬∃ l' : List ℕ+, 
    l'.length = 2023 ∧ 
    (∃! m : ℕ+, l'.count m = 15 ∧ ∀ n : ℕ+, l'.count n ≤ 15) ∧
    (l'.toFinset.card : ℕ) = k :=
by sorry

end least_distinct_values_l2694_269452


namespace factory_machines_l2694_269467

/-- Represents the number of machines in the factory -/
def num_machines : ℕ := 7

/-- Represents the time (in hours) taken by 6 machines to fill the order -/
def time_6_machines : ℕ := 42

/-- Represents the time (in hours) taken by all machines to fill the order -/
def time_all_machines : ℕ := 36

/-- Theorem stating that the number of machines in the factory is 7 -/
theorem factory_machines :
  (6 : ℚ) * time_all_machines * num_machines = time_6_machines * num_machines - 
  6 * time_6_machines := by sorry

end factory_machines_l2694_269467


namespace strawberry_cost_l2694_269434

/-- The price of one basket of strawberries in dollars -/
def price_per_basket : ℚ := 16.5

/-- The number of baskets to be purchased -/
def number_of_baskets : ℕ := 4

/-- The total cost of purchasing the strawberries -/
def total_cost : ℚ := price_per_basket * number_of_baskets

/-- Theorem stating that the total cost of 4 baskets of strawberries at $16.50 each is $66.00 -/
theorem strawberry_cost : total_cost = 66 := by
  sorry

end strawberry_cost_l2694_269434


namespace point_on_line_l2694_269444

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let A : Point := ⟨1, -5⟩
  let B : Point := ⟨3, -1⟩
  let C : Point := ⟨4.5, 2⟩
  collinear A B C := by
  sorry

end point_on_line_l2694_269444


namespace no_solution_iff_k_eq_seven_l2694_269491

theorem no_solution_iff_k_eq_seven (k : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 :=
by sorry

end no_solution_iff_k_eq_seven_l2694_269491


namespace rectangle_length_proof_l2694_269418

/-- Proves that a rectangle with length double its width, when modified as described, has an original length of 40. -/
theorem rectangle_length_proof (w : ℝ) (h1 : w > 0) : 
  (2*w - 5) * (w + 5) = 2*w*w + 75 → 2*w = 40 := by
  sorry

#check rectangle_length_proof

end rectangle_length_proof_l2694_269418


namespace ninety_squared_l2694_269475

theorem ninety_squared : 90 * 90 = 8100 := by
  sorry

end ninety_squared_l2694_269475


namespace notebooks_needed_correct_l2694_269498

/-- The number of notebooks needed to achieve a profit of $40 -/
def notebooks_needed : ℕ := 96

/-- The cost of 4 notebooks in dollars -/
def cost_of_four : ℚ := 15

/-- The selling price of 6 notebooks in dollars -/
def sell_price_of_six : ℚ := 25

/-- The desired profit in dollars -/
def desired_profit : ℚ := 40

/-- Theorem stating that the number of notebooks needed to achieve the desired profit is correct -/
theorem notebooks_needed_correct : 
  (notebooks_needed : ℚ) * (sell_price_of_six / 6 - cost_of_four / 4) ≥ desired_profit ∧
  ((notebooks_needed - 1) : ℚ) * (sell_price_of_six / 6 - cost_of_four / 4) < desired_profit :=
by sorry

end notebooks_needed_correct_l2694_269498


namespace parallel_line_through_point_l2694_269451

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define parallelism between two lines
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define when a point lies on a line
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parallel_line_through_point :
  ∃ (l : Line2D),
    parallel l (Line2D.mk 2 1 (-1)) ∧
    point_on_line (Point2D.mk 1 2) l ∧
    l = Line2D.mk 2 1 (-4) := by
  sorry

end parallel_line_through_point_l2694_269451


namespace five_digit_divisibility_l2694_269462

/-- A function that removes the middle digit of a five-digit number -/
def removeMidDigit (n : ℕ) : ℕ :=
  (n / 10000) * 1000 + (n % 1000)

/-- A predicate that checks if a number is five-digit -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem five_digit_divisibility (A : ℕ) :
  isFiveDigit A →
  (∃ k : ℕ, A = k * (removeMidDigit A)) ↔ (∃ m : ℕ, A = m * 1000) :=
by sorry

end five_digit_divisibility_l2694_269462


namespace daily_wage_c_value_l2694_269497

/-- The daily wage of worker c given the conditions of the problem -/
def daily_wage_c (days_a days_b days_c : ℕ) 
                 (ratio_a ratio_b ratio_c : ℕ) 
                 (total_earning : ℚ) : ℚ :=
  let wage_a := total_earning * 3 / (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c)
  wage_a * ratio_c / ratio_a

/-- Theorem stating that the daily wage of c is $66.67 given the problem conditions -/
theorem daily_wage_c_value : 
  daily_wage_c 6 9 4 3 4 5 1480 = 200/3 := by sorry

end daily_wage_c_value_l2694_269497


namespace bobs_hair_length_at_last_cut_l2694_269470

/-- The length of Bob's hair at his last haircut, given his current hair length,
    hair growth rate, and time since last haircut. -/
def hair_length_at_last_cut (current_length : ℝ) (growth_rate : ℝ) (years_since_cut : ℝ) : ℝ :=
  current_length - growth_rate * 12 * years_since_cut

/-- Theorem stating that Bob's hair length at his last haircut was 6 inches,
    given the provided conditions. -/
theorem bobs_hair_length_at_last_cut :
  hair_length_at_last_cut 36 0.5 5 = 6 := by
  sorry

end bobs_hair_length_at_last_cut_l2694_269470


namespace pizza_slices_l2694_269492

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (slices_per_pizza : ℕ) : 
  total_pizzas = 21 → total_slices = 168 → slices_per_pizza * total_pizzas = total_slices → slices_per_pizza = 8 := by
  sorry

end pizza_slices_l2694_269492


namespace sum_of_powers_of_i_is_zero_l2694_269484

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_is_zero :
  i^12345 + i^12346 + i^12347 + i^12348 = 0 :=
by
  sorry

end sum_of_powers_of_i_is_zero_l2694_269484


namespace monotonic_square_exists_l2694_269464

/-- A function that returns the number of digits of a positive integer in base 10 -/
def numDigits (x : ℕ+) : ℕ := sorry

/-- A function that checks if a positive integer is monotonic in base 10 -/
def isMonotonic (x : ℕ+) : Prop := sorry

/-- For every positive integer n, there exists an n-digit monotonic number which is a perfect square -/
theorem monotonic_square_exists (n : ℕ+) : ∃ x : ℕ+, 
  (numDigits x = n) ∧ 
  isMonotonic x ∧ 
  ∃ y : ℕ+, x = y * y := by
  sorry

end monotonic_square_exists_l2694_269464


namespace quadratic_function_range_l2694_269413

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem quadratic_function_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, ∃ y ∈ Set.Icc (-4 : ℝ) 12, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-4 : ℝ) 12 :=
by sorry

end quadratic_function_range_l2694_269413


namespace ellipse_m_range_l2694_269409

/-- The equation of an ellipse with foci on the x-axis -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Conditions for the ellipse -/
def ellipse_conditions (m : ℝ) : Prop :=
  10 - m > 0 ∧ m - 2 > 0 ∧ 10 - m > m - 2

/-- The range of m for which the ellipse exists -/
theorem ellipse_m_range :
  ∀ m : ℝ, ellipse_conditions m ↔ 2 < m ∧ m < 6 :=
sorry

end ellipse_m_range_l2694_269409


namespace distance_between_complex_points_l2694_269400

theorem distance_between_complex_points :
  let z₁ : ℂ := -3 + I
  let z₂ : ℂ := 1 - I
  Complex.abs (z₂ - z₁) = Real.sqrt 20 := by sorry

end distance_between_complex_points_l2694_269400


namespace arithmetic_sequence_sum_l2694_269419

/-- Given an arithmetic sequence {a_n}, if a_3 + a_4 + a_5 = 12, 
    then a_1 + a_2 + ... + a_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →
  (a 3 + a 4 + a 5 = 12) →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by sorry

end arithmetic_sequence_sum_l2694_269419


namespace geometric_sequence_sum_l2694_269482

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geo : isGeometric a)
  (h_pos : ∀ n, a n > 0)
  (h_sum : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l2694_269482


namespace inscribed_circle_radius_squared_l2694_269489

/-- A quadrilateral with an inscribed circle -/
structure InscribedCircleQuadrilateral where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- Length of AP -/
  ap : ℝ
  /-- Length of PB -/
  pb : ℝ
  /-- Length of CQ -/
  cq : ℝ
  /-- Length of QD -/
  qd : ℝ
  /-- The circle is tangent to AB at P and to CD at Q -/
  tangent_condition : True

/-- The theorem stating that for the given quadrilateral, the square of the radius is 13325 -/
theorem inscribed_circle_radius_squared
  (quad : InscribedCircleQuadrilateral)
  (h1 : quad.ap = 25)
  (h2 : quad.pb = 35)
  (h3 : quad.cq = 30)
  (h4 : quad.qd = 40) :
  quad.r ^ 2 = 13325 := by
  sorry


end inscribed_circle_radius_squared_l2694_269489


namespace log_equality_l2694_269421

theorem log_equality (a b : ℝ) (h1 : a = Real.log 900 / Real.log 4) (h2 : b = Real.log 30 / Real.log 2) : a = b := by
  sorry

end log_equality_l2694_269421


namespace system_solution_correct_l2694_269448

theorem system_solution_correct (x y : ℝ) : 
  x = 3 ∧ y = 1 → (2 * x - 3 * y = 3 ∧ x + 2 * y = 5) := by
sorry

end system_solution_correct_l2694_269448


namespace sum_of_x_and_y_l2694_269420

theorem sum_of_x_and_y (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 27) : x + y = 9 := by
  sorry

end sum_of_x_and_y_l2694_269420


namespace eldorado_license_plates_l2694_269441

theorem eldorado_license_plates : 
  let letter_choices : ℕ := 26
  let digit_choices : ℕ := 10
  let letter_spots : ℕ := 3
  let digit_spots : ℕ := 4
  letter_choices ^ letter_spots * digit_choices ^ digit_spots = 175760000 :=
by sorry

end eldorado_license_plates_l2694_269441


namespace arithmetic_expression_result_l2694_269437

theorem arithmetic_expression_result : 5 + 12 / 3 - 4 * 2 + 3^2 = 10 := by
  sorry

end arithmetic_expression_result_l2694_269437


namespace stephens_ant_farm_l2694_269485

/-- The number of ants in Stephen's ant farm satisfies the given conditions -/
theorem stephens_ant_farm (total_ants : ℕ) : 
  (total_ants / 2 : ℚ) * (80 / 100 : ℚ) = 44 → total_ants = 110 :=
by
  sorry

#check stephens_ant_farm

end stephens_ant_farm_l2694_269485


namespace max_value_theorem_l2694_269433

theorem max_value_theorem (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 4 + 9 * y * z ≤ Real.sqrt 13 := by
  sorry

end max_value_theorem_l2694_269433


namespace walker_cyclist_speed_ratio_l2694_269406

/-- Given two people, a walker and a cyclist, prove that the walker is twice as slow as the cyclist
    when the cyclist's speed is three times the walker's speed. -/
theorem walker_cyclist_speed_ratio
  (S : ℝ) -- distance between home and lake
  (x : ℝ) -- walking speed
  (h1 : 0 < x) -- walking speed is positive
  (h2 : 0 < S) -- distance is positive
  (v : ℝ) -- cycling speed
  (h3 : v = 3 * x) -- cyclist speed is 3 times walker speed
  : (S / x) / (S / v) = 2 := by
  sorry

end walker_cyclist_speed_ratio_l2694_269406


namespace binomial_expansion_coefficient_l2694_269493

/-- Given that the coefficient of x^-3 in the expansion of (2x - a/x)^7 is 84, prove that a = -1 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 7 5 : ℝ) * 2^2 * (-a)^5 = 84 → a = -1 := by
  sorry

end binomial_expansion_coefficient_l2694_269493


namespace triangle_properties_l2694_269403

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = t.b * Real.sin t.A)
  (h2 : t.b = 3 * Real.sqrt 2)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a + t.b + t.c = 6 + 3 * Real.sqrt 2 := by
  sorry

end triangle_properties_l2694_269403


namespace garden_vegetable_difference_l2694_269472

/-- Represents the number of vegetables in a garden -/
structure GardenVegetables where
  potatoes : ℕ
  cucumbers : ℕ
  peppers : ℕ

/-- Theorem stating the difference between potatoes and cucumbers in the garden -/
theorem garden_vegetable_difference (g : GardenVegetables) :
  g.potatoes = 237 →
  g.peppers = 2 * g.cucumbers →
  g.potatoes + g.cucumbers + g.peppers = 768 →
  g.potatoes - g.cucumbers = 60 := by
  sorry

#check garden_vegetable_difference

end garden_vegetable_difference_l2694_269472


namespace angle_C_is_60_degrees_angle_B_and_area_l2694_269426

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def given_condition (t : Triangle) : Prop :=
  ((t.a + t.b)^2 - t.c^2) / (3 * t.a * t.b) = 1

/-- Part 1 of the theorem -/
theorem angle_C_is_60_degrees (t : Triangle) (h : given_condition t) :
  t.C = Real.pi / 3 := by sorry

/-- Part 2 of the theorem -/
theorem angle_B_and_area (t : Triangle) 
  (h1 : t.c = Real.sqrt 3) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : t.C = Real.pi / 3) :
  t.B = Real.pi / 4 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (3 + Real.sqrt 3) / 4 := by sorry

end angle_C_is_60_degrees_angle_B_and_area_l2694_269426


namespace limit_log_div_power_l2694_269436

open Real

-- Define the function f(x) = ln(x) / x^α
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (log x) / (x ^ α)

-- State the theorem
theorem limit_log_div_power (α : ℝ) (h₁ : α > 0) :
  ∀ ε > 0, ∃ N, ∀ x ≥ N, x > 0 → |f α x - 0| < ε :=
sorry

end limit_log_div_power_l2694_269436


namespace intersection_points_on_circle_l2694_269449

/-- Given two parabolas that intersect at four points, prove that these points lie on a circle with radius squared equal to 5/2 -/
theorem intersection_points_on_circle (x y : ℝ) : 
  (y = (x - 2)^2) ∧ (x - 3 = (y + 1)^2) →
  ∃ (center : ℝ × ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = 5/2 :=
by sorry

end intersection_points_on_circle_l2694_269449


namespace saeyoung_money_conversion_l2694_269427

/-- The exchange rate from yuan to yen -/
def exchange_rate : ℝ := 17.25

/-- The value of Saeyoung's 1000 yuan bill -/
def bill_value : ℝ := 1000

/-- The value of Saeyoung's 10 yuan coin -/
def coin_value : ℝ := 10

/-- The total value of Saeyoung's Chinese money in yen -/
def total_yen : ℝ := (bill_value + coin_value) * exchange_rate

theorem saeyoung_money_conversion :
  total_yen = 17422.5 := by sorry

end saeyoung_money_conversion_l2694_269427


namespace sally_quarters_l2694_269477

theorem sally_quarters (x : ℕ) : 
  (x + 418 = 1178) → (x = 760) := by
  sorry

end sally_quarters_l2694_269477


namespace orange_groups_l2694_269490

theorem orange_groups (total_oranges : ℕ) (num_groups : ℕ) 
  (h1 : total_oranges = 384) (h2 : num_groups = 16) :
  total_oranges / num_groups = 24 := by
sorry

end orange_groups_l2694_269490


namespace smallest_other_integer_l2694_269481

theorem smallest_other_integer (x : ℕ) (a b : ℕ) : 
  a = 45 →
  a > 0 →
  b > 0 →
  x > 0 →
  Nat.gcd a b = x + 5 →
  Nat.lcm a b = x * (x + 5) →
  a + b < 100 →
  ∃ (b_min : ℕ), b_min = 12 ∧ ∀ (b' : ℕ), b' ≠ a ∧ 
    Nat.gcd a b' = x + 5 ∧
    Nat.lcm a b' = x * (x + 5) ∧
    a + b' < 100 →
    b' ≥ b_min :=
sorry

end smallest_other_integer_l2694_269481


namespace inverse_function_point_l2694_269468

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x-1) passes through (1, 2)
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f (1 - 1) = 2

-- Define the inverse function of f
noncomputable def f_inverse (f : ℝ → ℝ) : ℝ → ℝ :=
  Function.invFun f

-- Theorem statement
theorem inverse_function_point (f : ℝ → ℝ) :
  passes_through_point f → f_inverse f 2 = 0 :=
by
  sorry

end inverse_function_point_l2694_269468


namespace eight_b_equals_sixteen_l2694_269486

theorem eight_b_equals_sixteen (a b : ℝ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  8 * b = 16 := by
sorry

end eight_b_equals_sixteen_l2694_269486


namespace own_square_and_cube_root_l2694_269410

theorem own_square_and_cube_root : 
  ∀ x : ℝ, (x^2 = x ∧ x^3 = x) ↔ x = 0 :=
by sorry

end own_square_and_cube_root_l2694_269410


namespace triangle_side_length_l2694_269411

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (b * Real.sin A = 3 * c * Real.sin B) →
  (a = 3) →
  (Real.cos B = 2/3) →
  -- Triangle inequality (to ensure it's a valid triangle)
  (a + b > c) → (b + c > a) → (c + a > b) →
  -- Positive side lengths
  (a > 0) → (b > 0) → (c > 0) →
  -- Conclusion
  b = Real.sqrt 6 := by
sorry

end triangle_side_length_l2694_269411


namespace digit_222_is_zero_l2694_269466

/-- The decimal representation of 41/777 -/
def decimal_rep : ℚ := 41 / 777

/-- The length of the repeating block in the decimal representation of 41/777 -/
def repeating_block_length : ℕ := 42

/-- The position of the 222nd digit within the repeating block -/
def position_in_block : ℕ := 222 % repeating_block_length

/-- The 222nd digit after the decimal point in the decimal representation of 41/777 -/
def digit_222 : ℕ := 0

/-- Theorem stating that the 222nd digit after the decimal point 
    in the decimal representation of 41/777 is 0 -/
theorem digit_222_is_zero : digit_222 = 0 := by sorry

end digit_222_is_zero_l2694_269466


namespace cos_alpha_value_l2694_269459

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/6) = 1/3) :
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end cos_alpha_value_l2694_269459


namespace barbara_wins_2023_barbara_wins_2024_l2694_269473

/-- Represents the players in the coin removal game -/
inductive Player
| Barbara
| Jenna

/-- Represents the state of the game -/
structure GameState where
  coins : ℕ
  currentPlayer : Player

/-- Defines a valid move for a player -/
def validMove (player : Player) (coins : ℕ) : Set ℕ :=
  match player with
  | Player.Barbara => {2, 4, 5}
  | Player.Jenna => {1, 3, 5}

/-- Determines if a game state is winning for the current player -/
def isWinningState : GameState → Prop :=
  sorry

/-- Theorem stating that Barbara wins with 2023 coins -/
theorem barbara_wins_2023 :
  isWinningState ⟨2023, Player.Barbara⟩ :=
  sorry

/-- Theorem stating that Barbara wins with 2024 coins -/
theorem barbara_wins_2024 :
  isWinningState ⟨2024, Player.Barbara⟩ :=
  sorry

end barbara_wins_2023_barbara_wins_2024_l2694_269473


namespace circle_and_line_equations_l2694_269460

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧
                 (2 - a)^2 + (4 - b)^2 = r^2 ∧
                 (1 - a)^2 + (3 - b)^2 = r^2 ∧
                 a - b + 1 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 1

-- Define the dot product of OM and ON
def dot_product_OM_ON (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem circle_and_line_equations :
  ∀ (k : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l x₁ y₁ k ∧ line_l x₂ y₂ k ∧
    dot_product_OM_ON x₁ y₁ x₂ y₂ = 12) →
  (∀ (x y : ℝ), circle_C x y ↔ (x - 2)^2 + (y - 3)^2 = 1) ∧
  k = 1 :=
sorry

end circle_and_line_equations_l2694_269460


namespace pages_to_read_tonight_l2694_269494

/-- The number of pages in Juwella's book -/
def total_pages : ℕ := 500

/-- The number of pages Juwella read three nights ago -/
def pages_three_nights_ago : ℕ := 20

/-- The number of pages Juwella read two nights ago -/
def pages_two_nights_ago : ℕ := pages_three_nights_ago^2 + 5

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The number of pages Juwella read last night -/
def pages_last_night : ℕ := 3 * sum_of_digits pages_two_nights_ago

/-- The total number of pages Juwella has read so far -/
def total_pages_read : ℕ := pages_three_nights_ago + pages_two_nights_ago + pages_last_night

/-- Theorem stating the number of pages Juwella will read tonight -/
theorem pages_to_read_tonight : total_pages - total_pages_read = 48 := by
  sorry

end pages_to_read_tonight_l2694_269494


namespace complement_of_A_in_U_l2694_269435

def U : Set Int := {-2, 0, 1, 2}

def A : Set Int := {x ∈ U | x^2 + x - 2 = 0}

theorem complement_of_A_in_U :
  (U \ A) = {0, 2} := by sorry

end complement_of_A_in_U_l2694_269435


namespace six_objects_three_parts_l2694_269408

/-- The number of ways to partition n indistinguishable objects into at most k non-empty parts -/
def partition_count (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to partition 6 indistinguishable objects into at most 3 non-empty parts -/
theorem six_objects_three_parts : partition_count 6 3 = 7 := by
  sorry

end six_objects_three_parts_l2694_269408


namespace frequency_table_purpose_l2694_269425

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable where
  /-- The table analyzes sample data -/
  analyzes_sample_data : Bool
  /-- The table groups data into categories -/
  groups_data : Bool

/-- The purpose of creating a frequency distribution table -/
def purpose_of_frequency_table (table : FrequencyDistributionTable) : Prop :=
  table.analyzes_sample_data ∧ 
  table.groups_data → 
  (∃ (proportion_understanding : Prop) (population_estimation : Prop),
    proportion_understanding ∧ population_estimation)

/-- Theorem stating the purpose of creating a frequency distribution table -/
theorem frequency_table_purpose (table : FrequencyDistributionTable) : 
  purpose_of_frequency_table table :=
sorry

end frequency_table_purpose_l2694_269425


namespace division_calculation_l2694_269428

theorem division_calculation : (-1/30) / (2/3 - 1/10 + 1/6 - 2/5) = -1/10 := by
  sorry

end division_calculation_l2694_269428


namespace approx_cube_root_25_correct_l2694_269446

/-- Approximate value of the cube root of 25 -/
def approx_cube_root_25 : ℝ := 2.926

/-- Generalized binomial theorem approximation for small x -/
def binomial_approx (α x : ℝ) : ℝ := 1 + α * x

/-- Cube root of 27 -/
def cube_root_27 : ℝ := 3

theorem approx_cube_root_25_correct :
  let x := -2/27
  let α := 1/3
  approx_cube_root_25 = cube_root_27 * binomial_approx α x := by sorry

end approx_cube_root_25_correct_l2694_269446


namespace stone_121_is_10_l2694_269453

/-- The number of stones in the sequence -/
def n : ℕ := 11

/-- The length of a full cycle (left-to-right and right-to-left) -/
def cycle_length : ℕ := 2 * n - 1

/-- The position of a stone in the original left-to-right count, given its count number -/
def stone_position (count : ℕ) : ℕ :=
  (count - 1) % cycle_length + 1

/-- The theorem stating that the 121st count corresponds to the 10th stone -/
theorem stone_121_is_10 : stone_position 121 = 10 := by
  sorry

end stone_121_is_10_l2694_269453


namespace octal_to_decimal_1743_l2694_269458

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the number -/
def octal_digits : List Nat := [3, 4, 7, 1]

theorem octal_to_decimal_1743 :
  octal_to_decimal octal_digits = 995 := by
  sorry

end octal_to_decimal_1743_l2694_269458


namespace polynomial_divisibility_l2694_269412

theorem polynomial_divisibility (a b : ℝ) : 
  (∀ (X : ℝ), (X - 1)^2 ∣ (a * X^4 + b * X^3 + 1)) ↔ 
  (a = 3 ∧ b = -4) := by
  sorry

end polynomial_divisibility_l2694_269412


namespace infinite_series_sum_l2694_269414

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 3^k is equal to 6 -/
theorem infinite_series_sum : ∑' k, (k^2 : ℝ) / 3^k = 6 := by sorry

end infinite_series_sum_l2694_269414


namespace ladder_tournament_rankings_ten_player_tournament_rankings_l2694_269471

/-- The number of possible rankings in a ladder-style tournament with n players. -/
def num_rankings (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2^(n-1)

/-- Theorem: The number of possible rankings in a ladder-style tournament with n players (n ≥ 2) is 2^(n-1). -/
theorem ladder_tournament_rankings (n : ℕ) (h : n ≥ 2) :
  num_rankings n = 2^(n-1) := by
  sorry

/-- Corollary: For a tournament with 10 players, there are 512 possible rankings. -/
theorem ten_player_tournament_rankings :
  num_rankings 10 = 512 := by
  sorry

end ladder_tournament_rankings_ten_player_tournament_rankings_l2694_269471


namespace largest_number_with_equal_quotient_and_remainder_l2694_269440

theorem largest_number_with_equal_quotient_and_remainder : ∀ A B C : ℕ,
  A = 8 * B + C →
  B = C →
  C < 8 →
  A ≤ 63 ∧ ∃ A₀ : ℕ, A₀ = 63 ∧ ∃ B₀ C₀ : ℕ, A₀ = 8 * B₀ + C₀ ∧ B₀ = C₀ ∧ C₀ < 8 :=
by
  sorry

end largest_number_with_equal_quotient_and_remainder_l2694_269440


namespace g_of_2_eq_neg_1_l2694_269495

/-- The function g defined as g(x) = x^2 - 3x + 1 -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem stating that g(2) = -1 -/
theorem g_of_2_eq_neg_1 : g 2 = -1 := by sorry

end g_of_2_eq_neg_1_l2694_269495


namespace decimal_sum_to_fraction_l2694_269432

theorem decimal_sum_to_fraction :
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 : ℚ) = 22839 / 50000 := by
  sorry

end decimal_sum_to_fraction_l2694_269432
