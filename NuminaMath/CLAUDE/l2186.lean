import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_expansion_l2186_218687

theorem polynomial_expansion (x : ℝ) : 
  (3*x^3 + x^2 - 5*x + 9)*(x + 2) - (x + 2)*(2*x^3 - 4*x + 8) + (x^2 - 6*x + 13)*(x + 2)*(x - 3) = 
  2*x^4 + x^3 + 9*x^2 + 23*x + 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2186_218687


namespace NUMINAMATH_CALUDE_min_value_trig_function_l2186_218658

/-- Given a function f(x) = 3sin(x) + 4cos(x), if f(x) ≥ f(α) for any x ∈ ℝ, 
    then tan(α) = 3/4 -/
theorem min_value_trig_function (f : ℝ → ℝ) (α : ℝ) 
    (h1 : ∀ x, f x = 3 * Real.sin x + 4 * Real.cos x)
    (h2 : ∀ x, f x ≥ f α) : 
    Real.tan α = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l2186_218658


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_two_l2186_218601

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop :=
  (3*x + 2*y = 0) ∧ (3*x - 2*y = 0)

theorem hyperbola_asymptote_implies_a_equals_two :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, hyperbola_equation x y a → asymptote_equations x y) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_two_l2186_218601


namespace NUMINAMATH_CALUDE_square_land_side_length_l2186_218684

theorem square_land_side_length (area : ℝ) (side : ℝ) : 
  area = 1024 → side * side = area → side = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l2186_218684


namespace NUMINAMATH_CALUDE_percent_above_sixty_percent_l2186_218622

theorem percent_above_sixty_percent (P Q : ℝ) (h : P > Q) :
  (P - 0.6 * Q) / Q * 100 = (100 * P - 60 * Q) / Q := by
  sorry

end NUMINAMATH_CALUDE_percent_above_sixty_percent_l2186_218622


namespace NUMINAMATH_CALUDE_point_difference_l2186_218636

/-- The value of a touchdown in points -/
def touchdown_value : ℕ := 7

/-- The number of touchdowns scored by Brayden and Gavin -/
def brayden_gavin_touchdowns : ℕ := 7

/-- The number of touchdowns scored by Cole and Freddy -/
def cole_freddy_touchdowns : ℕ := 9

/-- The point difference between Cole and Freddy's team and Brayden and Gavin's team -/
theorem point_difference : 
  cole_freddy_touchdowns * touchdown_value - brayden_gavin_touchdowns * touchdown_value = 14 :=
by sorry

end NUMINAMATH_CALUDE_point_difference_l2186_218636


namespace NUMINAMATH_CALUDE_max_min_on_interval_l2186_218652

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = min) ∧
    max = 5 ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l2186_218652


namespace NUMINAMATH_CALUDE_pauls_tips_amount_l2186_218627

def pauls_tips : ℕ := 14
def vinnies_earnings : ℕ := 30

theorem pauls_tips_amount :
  (∃ (p : ℕ), p = pauls_tips ∧ vinnies_earnings = p + 16) →
  pauls_tips = 14 := by
  sorry

end NUMINAMATH_CALUDE_pauls_tips_amount_l2186_218627


namespace NUMINAMATH_CALUDE_special_right_triangle_area_l2186_218665

/-- Represents a right triangle with an incircle that evenly trisects a median -/
structure SpecialRightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Incircle radius
  r : ℝ
  -- Median length
  m : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  hypotenuse : c = 24
  trisected_median : m = 3 * r
  area_condition : a * b = 288

/-- The main theorem -/
theorem special_right_triangle_area (t : SpecialRightTriangle) : 
  ∃ (m n : ℕ), t.a * t.b / 2 = m * Real.sqrt n ∧ m = 144 ∧ n = 1 ∧ ¬ ∃ (p : ℕ), Prime p ∧ n % (p^2) = 0 :=
sorry

end NUMINAMATH_CALUDE_special_right_triangle_area_l2186_218665


namespace NUMINAMATH_CALUDE_opponent_total_runs_is_67_l2186_218632

/-- Represents the scores of a baseball team in a series of games. -/
structure BaseballScores :=
  (scores : List Nat)
  (lostByTwoGames : Nat)
  (wonByTripleGames : Nat)

/-- Calculates the total runs scored by the opponents. -/
def opponentTotalRuns (bs : BaseballScores) : Nat :=
  sorry

/-- The theorem states that given the specific conditions of the baseball team's games,
    the total runs scored by their opponents is 67. -/
theorem opponent_total_runs_is_67 :
  let bs : BaseballScores := {
    scores := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    lostByTwoGames := 5,
    wonByTripleGames := 5
  }
  opponentTotalRuns bs = 67 := by sorry

end NUMINAMATH_CALUDE_opponent_total_runs_is_67_l2186_218632


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2186_218605

theorem unique_solution_for_equation : ∃! (x y : ℤ), (x + 2)^4 - x^4 = y^3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2186_218605


namespace NUMINAMATH_CALUDE_inequality_proof_l2186_218618

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2186_218618


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l2186_218638

/-- Given a piece of wood of length x and a rope of length y, 
    if there are 4.5 feet of rope left when measuring the wood 
    and 1 foot left when measuring with half the rope, 
    then the system of equations y - x = 4.5 and x - y/2 = 1 holds. -/
theorem sunzi_wood_measurement (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : x - y / 2 = 1) : 
  y - x = 4.5 ∧ x - y / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l2186_218638


namespace NUMINAMATH_CALUDE_remainder_theorem_l2186_218671

theorem remainder_theorem : (8 * 20^34 + 3^34) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2186_218671


namespace NUMINAMATH_CALUDE_thin_mints_price_l2186_218673

theorem thin_mints_price (samoas_price : ℝ) (fudge_delights_price : ℝ) (sugar_cookies_price : ℝ)
  (samoas_quantity : ℕ) (thin_mints_quantity : ℕ) (fudge_delights_quantity : ℕ) (sugar_cookies_quantity : ℕ)
  (total_earned : ℝ) :
  samoas_price = 4 →
  fudge_delights_price = 5 →
  sugar_cookies_price = 2 →
  samoas_quantity = 3 →
  thin_mints_quantity = 2 →
  fudge_delights_quantity = 1 →
  sugar_cookies_quantity = 9 →
  total_earned = 42 →
  (total_earned - (samoas_price * samoas_quantity + fudge_delights_price * fudge_delights_quantity + sugar_cookies_price * sugar_cookies_quantity)) / thin_mints_quantity = 3.5 := by
sorry

end NUMINAMATH_CALUDE_thin_mints_price_l2186_218673


namespace NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l2186_218655

/-- The sum of the infinite series Σ(n=1 to ∞) [3^n / (1 + 3^n + 3^(n+1) + 3^(2n+2))] is equal to 1/4. -/
theorem series_sum_equals_one_fourth :
  ∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l2186_218655


namespace NUMINAMATH_CALUDE_c_increases_as_n_increases_l2186_218645

/-- Given a formula for C, prove that C increases as n increases. -/
theorem c_increases_as_n_increases
  (e R r : ℝ)
  (he : e > 0)
  (hR : R > 0)
  (hr : r > 0)
  (C : ℝ → ℝ)
  (hC : ∀ n, n > 0 → C n = (e^2 * n) / (R + n*r)) :
  ∀ n₁ n₂, 0 < n₁ → n₁ < n₂ → C n₁ < C n₂ :=
by sorry

end NUMINAMATH_CALUDE_c_increases_as_n_increases_l2186_218645


namespace NUMINAMATH_CALUDE_additional_amount_needed_l2186_218689

/-- The cost of the perfume --/
def perfume_cost : ℚ := 75

/-- The amount Christian saved --/
def christian_saved : ℚ := 5

/-- The amount Sue saved --/
def sue_saved : ℚ := 7

/-- The number of yards Christian mowed --/
def yards_mowed : ℕ := 6

/-- The price Christian charged per yard --/
def price_per_yard : ℚ := 6

/-- The number of dogs Sue walked --/
def dogs_walked : ℕ := 8

/-- The price Sue charged per dog --/
def price_per_dog : ℚ := 3

/-- The theorem stating the additional amount needed --/
theorem additional_amount_needed : 
  perfume_cost - (christian_saved + sue_saved + yards_mowed * price_per_yard + dogs_walked * price_per_dog) = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_l2186_218689


namespace NUMINAMATH_CALUDE_assemble_cook_time_is_five_l2186_218688

/-- The time it takes to assemble and cook one omelet -/
def assemble_cook_time (
  pepper_chop_time : ℕ)  -- Time to chop one pepper
  (onion_chop_time : ℕ)  -- Time to chop one onion
  (cheese_grate_time : ℕ)  -- Time to grate cheese for one omelet
  (num_peppers : ℕ)  -- Number of peppers to chop
  (num_onions : ℕ)  -- Number of onions to chop
  (num_omelets : ℕ)  -- Number of omelets to make
  (total_time : ℕ)  -- Total time for preparing and cooking all omelets
  : ℕ :=
  let prep_time := pepper_chop_time * num_peppers + onion_chop_time * num_onions + cheese_grate_time * num_omelets
  (total_time - prep_time) / num_omelets

/-- Theorem stating that it takes 5 minutes to assemble and cook one omelet -/
theorem assemble_cook_time_is_five :
  assemble_cook_time 3 4 1 4 2 5 50 = 5 := by
  sorry


end NUMINAMATH_CALUDE_assemble_cook_time_is_five_l2186_218688


namespace NUMINAMATH_CALUDE_marble_probability_l2186_218639

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 20) (h2 : blue = 6) (h3 : red = 9) :
  let white := total - (blue + red)
  (red + white : ℚ) / total = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l2186_218639


namespace NUMINAMATH_CALUDE_correct_statements_count_l2186_218635

-- Define a structure for a programming statement
structure ProgrammingStatement :=
  (text : String)
  (is_correct : Bool)

-- Define the four statements
def statement1 : ProgrammingStatement :=
  ⟨"INPUT \"a, b, c=\"; a, b; c", false⟩

def statement2 : ProgrammingStatement :=
  ⟨"PRINT S=7", false⟩

def statement3 : ProgrammingStatement :=
  ⟨"9=r", false⟩

def statement4 : ProgrammingStatement :=
  ⟨"PRINT 20.3*2", true⟩

-- Define a list of all statements
def all_statements : List ProgrammingStatement :=
  [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (all_statements.filter (λ s => s.is_correct)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l2186_218635


namespace NUMINAMATH_CALUDE_ellipse_equation_l2186_218660

theorem ellipse_equation (x y : ℝ) : 
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m ≠ n ∧
    m * 2^2 + n * Real.sqrt 2^2 = 1 ∧
    m * (Real.sqrt 2)^2 + n * (Real.sqrt 3)^2 = 1) →
  (x^2 / 8 + y^2 / 4 = 1 ↔ m * x^2 + n * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2186_218660


namespace NUMINAMATH_CALUDE_train_length_l2186_218664

/-- The length of a train given its speed, the speed of a man walking in the same direction,
    and the time it takes for the train to cross the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 71.99424046076314 →
  (train_speed - man_speed) * (5 / 18) * crossing_time = 1199.9040076793857 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2186_218664


namespace NUMINAMATH_CALUDE_dimitri_burgers_per_day_l2186_218602

/-- The number of calories in each burger -/
def calories_per_burger : ℕ := 20

/-- The total number of calories Dimitri consumes in two days -/
def total_calories : ℕ := 120

/-- The number of days -/
def days : ℕ := 2

/-- The number of burgers Dimitri eats per day -/
def burgers_per_day : ℕ := total_calories / (calories_per_burger * days)

theorem dimitri_burgers_per_day : burgers_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_dimitri_burgers_per_day_l2186_218602


namespace NUMINAMATH_CALUDE_number_difference_l2186_218625

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 125000)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100)
  (a_div_5 : 5 ∣ a)
  (b_div_5 : 5 ∣ b) :
  b - a = 122265 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2186_218625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2186_218681

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms

/-- Properties of the arithmetic sequence -/
def ArithmeticSequenceProperties (seq : ArithmeticSequence) : Prop :=
  (seq.a 2 = 3) ∧ (seq.S 9 = 6 * seq.S 3)

/-- Theorem: The common difference of the arithmetic sequence is 1 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : ArithmeticSequenceProperties seq) :
  seq.d = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2186_218681


namespace NUMINAMATH_CALUDE_sum_difference_equality_l2186_218683

theorem sum_difference_equality : 3.59 + 2.4 - 1.67 = 4.32 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equality_l2186_218683


namespace NUMINAMATH_CALUDE_logarithmic_expression_equality_algebraic_expression_equality_l2186_218697

-- Part 1
theorem logarithmic_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 25 ^ (Real.log 3 / Real.log 5) = -7 := by sorry

-- Part 2
theorem algebraic_expression_equality : 
  (9 / 4) ^ (1 / 2) - (-7.8) ^ 0 - (27 / 8) ^ (2 / 3) + (2 / 3) ^ (-2) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_logarithmic_expression_equality_algebraic_expression_equality_l2186_218697


namespace NUMINAMATH_CALUDE_middle_number_proof_l2186_218620

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b = 12) (h4 : a + c = 17) (h5 : b + c = 19) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2186_218620


namespace NUMINAMATH_CALUDE_shelter_cats_l2186_218634

theorem shelter_cats (x : ℚ) 
  (h1 : x + x/2 - 3 + 5 - 1 = 19) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_shelter_cats_l2186_218634


namespace NUMINAMATH_CALUDE_problem_solution_l2186_218630

theorem problem_solution : 
  let X := (354 * 28)^2
  let Y := (48 * 14)^2
  (X * 9) / (Y * 2) = 2255688 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2186_218630


namespace NUMINAMATH_CALUDE_problem_statement_l2186_218685

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) : 
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) ∧ 
  (∃ (m : ℝ), m = 2 * Real.sqrt 6 + 3 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = x * y → |2*x - 1| + |3*y - 1| ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2186_218685


namespace NUMINAMATH_CALUDE_local_min_implies_a_range_l2186_218694

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*a*x + 3*a

-- Define what it means for f to have a local minimum in (0,1)
def has_local_min_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 0 < x ∧ x < 1 ∧ ∃ δ > 0, ∀ y, |y - x| < δ → f y ≥ f x

-- The theorem statement
theorem local_min_implies_a_range (a : ℝ) :
  has_local_min_in_interval (f a) → 0 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_local_min_implies_a_range_l2186_218694


namespace NUMINAMATH_CALUDE_i_power_difference_l2186_218619

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the cyclic property of i with period 4
axiom i_cyclic (n : ℤ) : i^n = i^(n % 4)

-- Theorem to prove
theorem i_power_difference : i^37 - i^29 = 0 := by sorry

end NUMINAMATH_CALUDE_i_power_difference_l2186_218619


namespace NUMINAMATH_CALUDE_special_polynomial_sum_l2186_218606

/-- A monic polynomial of degree 4 satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a b c d : ℝ, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  p 1 = 20 ∧ p 2 = 40 ∧ p 3 = 60

/-- The sum of p(0) and p(4) for a special polynomial p is 92 -/
theorem special_polynomial_sum (p : ℝ → ℝ) (h : SpecialPolynomial p) : 
  p 0 + p 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_sum_l2186_218606


namespace NUMINAMATH_CALUDE_sam_watermelons_l2186_218642

/-- The number of watermelons Sam grew initially -/
def initial_watermelons : ℕ := 4

/-- The number of additional watermelons Sam grew -/
def additional_watermelons : ℕ := 3

/-- The total number of watermelons Sam has -/
def total_watermelons : ℕ := initial_watermelons + additional_watermelons

theorem sam_watermelons : total_watermelons = 7 := by
  sorry

end NUMINAMATH_CALUDE_sam_watermelons_l2186_218642


namespace NUMINAMATH_CALUDE_students_left_early_l2186_218661

theorem students_left_early (original_groups : ℕ) (students_per_group : ℕ) (remaining_students : ℕ)
  (h1 : original_groups = 3)
  (h2 : students_per_group = 8)
  (h3 : remaining_students = 22) :
  original_groups * students_per_group - remaining_students = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_left_early_l2186_218661


namespace NUMINAMATH_CALUDE_factorial_ratio_l2186_218663

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2186_218663


namespace NUMINAMATH_CALUDE_intersection_M_N_l2186_218624

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2186_218624


namespace NUMINAMATH_CALUDE_unique_c_value_l2186_218604

theorem unique_c_value (c : ℝ) : c ≠ 0 ∧
  (∃! (b₁ b₂ b₃ : ℝ), b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧ b₁ ≠ b₂ ∧ b₂ ≠ b₃ ∧ b₁ ≠ b₃ ∧
    (∀ x : ℝ, x^2 + 2*(b₁ + 1/b₁)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₁ + 1/b₁)*y + c = 0) ∧
    (∀ x : ℝ, x^2 + 2*(b₂ + 1/b₂)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₂ + 1/b₂)*y + c = 0) ∧
    (∀ x : ℝ, x^2 + 2*(b₃ + 1/b₃)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₃ + 1/b₃)*y + c = 0)) →
  c = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_c_value_l2186_218604


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2186_218650

def expression : ℕ := 7^7 - 7^3

theorem sum_of_distinct_prime_factors : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (expression + 1)))
    (λ p => if p ∣ expression then p else 0)) = 17 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2186_218650


namespace NUMINAMATH_CALUDE_circle_arcs_angle_sum_l2186_218621

theorem circle_arcs_angle_sum (n : ℕ) (x y : ℝ) : 
  n = 18 → 
  x = 3 * (360 / n) / 2 →
  y = 5 * (360 / n) / 2 →
  x + y = 80 := by
  sorry

end NUMINAMATH_CALUDE_circle_arcs_angle_sum_l2186_218621


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2186_218614

theorem arithmetic_calculations :
  ((294.4 - 19.2 * 6) / (6 + 8) = 12.8) ∧
  (12.5 * 0.4 * 8 * 2.5 = 100) ∧
  (333 * 334 + 999 * 222 = 333000) ∧
  (999 + 99.9 + 9.99 + 0.999 = 1109.889) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2186_218614


namespace NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l2186_218686

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 8 := by
  sorry

theorem max_sum_achieved : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l2186_218686


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l2186_218633

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_d_for_inverse :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → x ≠ y → g x ≠ g y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l2186_218633


namespace NUMINAMATH_CALUDE_compound_propositions_truth_l2186_218653

-- Define proposition p
def p : Prop := ∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ∧ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a^(1/2) > b^(1/2) ↔ Real.log a > Real.log b

theorem compound_propositions_truth : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ((¬p) ∨ (¬q)) ∧ (p ∧ (¬q)) := by sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_l2186_218653


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2186_218643

theorem quadratic_equation_solution (x c d : ℝ) : 
  x^2 + 14*x = 92 → 
  (∃ c d : ℕ+, x = Real.sqrt c - d) →
  (∃ c d : ℕ+, x = Real.sqrt c - d ∧ c + d = 148) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2186_218643


namespace NUMINAMATH_CALUDE_profit_of_c_l2186_218675

def total_profit : ℕ := 56700
def ratio_a : ℕ := 8
def ratio_b : ℕ := 9
def ratio_c : ℕ := 10

theorem profit_of_c :
  let total_ratio := ratio_a + ratio_b + ratio_c
  let part_value := total_profit / total_ratio
  part_value * ratio_c = 21000 := by sorry

end NUMINAMATH_CALUDE_profit_of_c_l2186_218675


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2186_218659

theorem unknown_number_proof (x : ℝ) : (x + 48 / 69) * 69 = 1980 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2186_218659


namespace NUMINAMATH_CALUDE_josh_marbles_l2186_218646

theorem josh_marbles (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 16 → lost = 7 → remaining = initial - lost → remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l2186_218646


namespace NUMINAMATH_CALUDE_find_N_l2186_218631

theorem find_N : ∃ N : ℕ, (10 + 11 + 12 + 13) / 4 = (1000 + 1001 + 1002 + 1003) / N ∧ N = 348 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l2186_218631


namespace NUMINAMATH_CALUDE_circumscribing_sphere_surface_area_l2186_218611

/-- A right triangular rectangular pyramid with side length a -/
structure RightTriangularRectangularPyramid where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A sphere containing the vertices of a right triangular rectangular pyramid -/
structure CircumscribingSphere (p : RightTriangularRectangularPyramid) where
  radius : ℝ
  radius_pos : radius > 0

/-- The theorem stating the surface area of the circumscribing sphere -/
theorem circumscribing_sphere_surface_area
  (p : RightTriangularRectangularPyramid)
  (s : CircumscribingSphere p) :
  4 * Real.pi * s.radius ^ 2 = 3 * Real.pi * p.side_length ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribing_sphere_surface_area_l2186_218611


namespace NUMINAMATH_CALUDE_yoongi_number_division_l2186_218691

theorem yoongi_number_division (x : ℤ) : 
  x - 17 = 55 → x / 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_number_division_l2186_218691


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l2186_218648

theorem cos_pi_fourth_plus_alpha (α : ℝ) (h : Real.sin (α - π/4) = 1/3) :
  Real.cos (π/4 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l2186_218648


namespace NUMINAMATH_CALUDE_ln_is_control_function_of_x_ln_x_l2186_218603

/-- Definition of a control function -/
def is_control_function (f g : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → 
    g x₁ ≤ (f x₁ - f x₂) / (x₁ - x₂) ∧ (f x₁ - f x₂) / (x₁ - x₂) ≤ g x₂

/-- The main theorem to prove -/
theorem ln_is_control_function_of_x_ln_x :
  is_control_function (fun x => x * Real.log x) Real.log :=
sorry

end NUMINAMATH_CALUDE_ln_is_control_function_of_x_ln_x_l2186_218603


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2186_218678

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 < 2 ↔ n ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2186_218678


namespace NUMINAMATH_CALUDE_determine_x_value_l2186_218629

theorem determine_x_value (w y z x : ℕ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 8) : 
  x = 138 := by
  sorry

end NUMINAMATH_CALUDE_determine_x_value_l2186_218629


namespace NUMINAMATH_CALUDE_distance_a_beats_b_proof_l2186_218692

/-- The distance A can beat B when running 4.5 km -/
def distance_a_beats_b (a_speed : ℝ) (time_diff : ℝ) : ℝ :=
  a_speed * time_diff

/-- Theorem stating that the distance A beats B is equal to A's speed multiplied by the time difference -/
theorem distance_a_beats_b_proof (a_speed : ℝ) (time_diff : ℝ) (a_time : ℝ) (b_time : ℝ) 
    (h1 : a_speed = 4.5 / a_time)
    (h2 : time_diff = b_time - a_time)
    (h3 : a_time = 90)
    (h4 : b_time = 180) :
  distance_a_beats_b a_speed time_diff = 4.5 := by
  sorry

#check distance_a_beats_b_proof

end NUMINAMATH_CALUDE_distance_a_beats_b_proof_l2186_218692


namespace NUMINAMATH_CALUDE_tyrone_eric_marbles_l2186_218647

theorem tyrone_eric_marbles (tyrone_initial : ℕ) (eric_initial : ℕ) 
  (h1 : tyrone_initial = 97) 
  (h2 : eric_initial = 11) : 
  ∃ (marbles_given : ℕ), 
    marbles_given = 25 ∧ 
    (tyrone_initial - marbles_given = 2 * (eric_initial + marbles_given)) := by
  sorry

end NUMINAMATH_CALUDE_tyrone_eric_marbles_l2186_218647


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2186_218669

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  angleB : ℝ

-- Define the theorem
theorem triangle_area_theorem (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : t.b = 1) (h3 : t.angleB = π / 6) :
  ∃ (S : ℝ), (S = Real.sqrt 3 / 2 ∨ S = Real.sqrt 3 / 4) ∧ 
  (∃ (angleA angleC : ℝ), 
    angleA + t.angleB + angleC = π ∧
    S = 1/2 * t.a * t.b * Real.sin angleC) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2186_218669


namespace NUMINAMATH_CALUDE_log_expression_equals_zero_l2186_218649

theorem log_expression_equals_zero : 
  (Real.log 270 / Real.log 3) / (Real.log 3 / Real.log 54) - 
  (Real.log 540 / Real.log 3) / (Real.log 3 / Real.log 27) = 0 := by
sorry

end NUMINAMATH_CALUDE_log_expression_equals_zero_l2186_218649


namespace NUMINAMATH_CALUDE_lcm_of_165_and_396_l2186_218628

theorem lcm_of_165_and_396 : Nat.lcm 165 396 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_165_and_396_l2186_218628


namespace NUMINAMATH_CALUDE_solve_equation_l2186_218662

theorem solve_equation (x : ℝ) (h : 3 * x = (26 - x) + 26) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2186_218662


namespace NUMINAMATH_CALUDE_kennedy_car_drive_l2186_218654

theorem kennedy_car_drive (miles_per_gallon : ℝ) (initial_gas : ℝ) 
  (to_school : ℝ) (to_softball : ℝ) (to_restaurant : ℝ) (to_home : ℝ) 
  (to_friend : ℝ) : 
  miles_per_gallon = 19 →
  initial_gas = 2 →
  to_school = 15 →
  to_softball = 6 →
  to_restaurant = 2 →
  to_home = 11 →
  miles_per_gallon * initial_gas = to_school + to_softball + to_restaurant + to_friend + to_home →
  to_friend = 4 := by sorry

end NUMINAMATH_CALUDE_kennedy_car_drive_l2186_218654


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2186_218656

theorem smallest_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  ∀ m : ℕ, m > 0 ∧ 
    m % 2 = 1 ∧
    m % 3 = 2 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2186_218656


namespace NUMINAMATH_CALUDE_ellipse_point_coordinates_l2186_218693

theorem ellipse_point_coordinates (x y α : ℝ) : 
  x = 4 * Real.cos α → 
  y = 2 * Real.sqrt 3 * Real.sin α → 
  x > 0 → 
  y > 0 → 
  y / x = Real.sqrt 3 → 
  (x, y) = (4 * Real.sqrt 5 / 5, 4 * Real.sqrt 15 / 5) := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_coordinates_l2186_218693


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_9_l2186_218608

theorem largest_integer_less_than_100_with_remainder_5_mod_9 :
  ∀ n : ℕ, n < 100 → n % 9 = 5 → n ≤ 95 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_9_l2186_218608


namespace NUMINAMATH_CALUDE_cube_root_sum_l2186_218668

theorem cube_root_sum (a : ℝ) (h : a^3 = 7) :
  (0.007 : ℝ)^(1/3) + 7000^(1/3) = 10.1 * a := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_l2186_218668


namespace NUMINAMATH_CALUDE_fraction_equality_l2186_218615

theorem fraction_equality (a b c d : ℝ) : 
  (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4 →
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2186_218615


namespace NUMINAMATH_CALUDE_negative_1651_mod_9_l2186_218651

theorem negative_1651_mod_9 : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1651 ≡ n [ZMOD 9] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_negative_1651_mod_9_l2186_218651


namespace NUMINAMATH_CALUDE_slower_speed_fraction_l2186_218600

/-- Given that a person arrives at a bus stop 9 minutes later than normal when walking
    at a certain fraction of their usual speed, and it takes 36 minutes to walk to the
    bus stop at their usual speed, prove that the fraction of the usual speed they
    were walking at is 4/5. -/
theorem slower_speed_fraction (usual_time : ℕ) (delay : ℕ) (usual_time_eq : usual_time = 36) (delay_eq : delay = 9) :
  (usual_time : ℚ) / (usual_time + delay : ℚ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_fraction_l2186_218600


namespace NUMINAMATH_CALUDE_min_c_value_l2186_218617

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a| + |x - b| + |x - c|) :
  ∀ c' : ℕ, (0 < c' ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < c' ∧
    ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a'| + |x - b'| + |x - c'|) → c' ≥ 1002 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2186_218617


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2186_218644

theorem min_value_trig_expression (γ δ : ℝ) :
  ∃ (min : ℝ), min = 36 ∧
  ∀ (γ' δ' : ℝ), (3 * Real.cos γ' + 4 * Real.sin δ' - 7)^2 + 
    (3 * Real.sin γ' + 4 * Real.cos δ' - 12)^2 ≥ min ∧
  ∃ (γ₀ δ₀ : ℝ), (3 * Real.cos γ₀ + 4 * Real.sin δ₀ - 7)^2 + 
    (3 * Real.sin γ₀ + 4 * Real.cos δ₀ - 12)^2 = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2186_218644


namespace NUMINAMATH_CALUDE_total_groups_created_l2186_218623

def group_size : ℕ := 6
def eggs : ℕ := 18
def bananas : ℕ := 72
def marbles : ℕ := 66

theorem total_groups_created : 
  (eggs / group_size + bananas / group_size + marbles / group_size) = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_groups_created_l2186_218623


namespace NUMINAMATH_CALUDE_min_value_of_f_l2186_218672

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), f x_min = Real.exp (-1) ∧ ∀ (x : ℝ), f x ≥ Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2186_218672


namespace NUMINAMATH_CALUDE_equation_root_approximation_l2186_218657

/-- The equation whose root we need to find -/
def equation (x : ℝ) : Prop :=
  (Real.sqrt 5 - Real.sqrt 2) * (1 + x) = (Real.sqrt 6 - Real.sqrt 3) * (1 - x)

/-- The approximate root of the equation -/
def approximate_root : ℝ := -0.068

/-- Theorem stating that the approximate root satisfies the equation within a small error -/
theorem equation_root_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((Real.sqrt 5 - Real.sqrt 2) * (1 + approximate_root) - 
    (Real.sqrt 6 - Real.sqrt 3) * (1 - approximate_root))| < ε :=
sorry

end NUMINAMATH_CALUDE_equation_root_approximation_l2186_218657


namespace NUMINAMATH_CALUDE_order_of_logarithms_and_fraction_l2186_218682

theorem order_of_logarithms_and_fraction :
  let a := Real.log 5 / Real.log 8
  let b := Real.log 3 / Real.log 4
  let c := 2 / 3
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_logarithms_and_fraction_l2186_218682


namespace NUMINAMATH_CALUDE_inequality_on_unit_circle_l2186_218637

/-- The complex unit circle -/
def unit_circle : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- The inequality holds for all points on the unit circle -/
theorem inequality_on_unit_circle :
  ∀ z ∈ unit_circle, (Complex.abs (z + 1) - Real.sqrt 2) * (Complex.abs (z - 1) - Real.sqrt 2) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_on_unit_circle_l2186_218637


namespace NUMINAMATH_CALUDE_inequality_relationship_l2186_218616

theorem inequality_relationship (x : ℝ) : 
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧ 
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2186_218616


namespace NUMINAMATH_CALUDE_vacuuming_time_ratio_l2186_218699

theorem vacuuming_time_ratio : 
  ∀ (time_downstairs : ℝ),
  time_downstairs > 0 →
  27 = time_downstairs + 5 →
  38 = 27 + time_downstairs →
  (27 : ℝ) / time_downstairs = 27 / 22 :=
by
  sorry

end NUMINAMATH_CALUDE_vacuuming_time_ratio_l2186_218699


namespace NUMINAMATH_CALUDE_vector_identity_l2186_218640

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, and D in a vector space, 
    CB + AD - AB = CD -/
theorem vector_identity (A B C D : V) : C - B + (D - A) - (B - A) = D - C := by
  sorry

end NUMINAMATH_CALUDE_vector_identity_l2186_218640


namespace NUMINAMATH_CALUDE_equation_solution_l2186_218613

theorem equation_solution : 
  ∃ x : ℝ, (3 * x^2 + 6 = |(-25 + x)|) ∧ 
  (x = (-1 + Real.sqrt 229) / 6 ∨ x = (-1 - Real.sqrt 229) / 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2186_218613


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2186_218610

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  2 * x < 3 * (x - 3) + 1 ∧ (3 * x + 2) / 4 > x + a

-- Define the condition for having exactly four integer solutions
def has_four_integer_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℤ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    (∀ x : ℤ, inequality_system x a ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- The theorem to be proved
theorem inequality_system_solution_range :
  ∀ a : ℝ, has_four_integer_solutions a → -11/4 ≤ a ∧ a < -5/2 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2186_218610


namespace NUMINAMATH_CALUDE_min_average_books_borrowed_l2186_218670

theorem min_average_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 25)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 4)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let min_total_books := one_book * 1 + two_books * 2 + remaining_students * 3
  (min_total_books : ℚ) / total_students ≥ 1.64 := by
  sorry

end NUMINAMATH_CALUDE_min_average_books_borrowed_l2186_218670


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2186_218607

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem f_decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2186_218607


namespace NUMINAMATH_CALUDE_business_card_exchanges_count_l2186_218677

/-- Represents a business conference with two groups of people -/
structure BusinessConference where
  total_people : ℕ
  group1_size : ℕ
  group2_size : ℕ
  h_total : total_people = group1_size + group2_size
  h_group1 : group1_size = 25
  h_group2 : group2_size = 15

/-- Calculates the number of business card exchanges in a business conference -/
def business_card_exchanges (conf : BusinessConference) : ℕ :=
  conf.group1_size * conf.group2_size

/-- Theorem stating that the number of business card exchanges is 375 -/
theorem business_card_exchanges_count (conf : BusinessConference) :
  business_card_exchanges conf = 375 := by
  sorry

#eval business_card_exchanges ⟨40, 25, 15, rfl, rfl, rfl⟩

end NUMINAMATH_CALUDE_business_card_exchanges_count_l2186_218677


namespace NUMINAMATH_CALUDE_happy_street_traffic_happy_street_traffic_proof_l2186_218698

theorem happy_street_traffic (tuesday : ℕ) (thursday friday weekend_day : ℕ) 
  (total : ℕ) : ℕ :=
  let monday := tuesday - tuesday / 5
  let thursday_to_sunday := thursday + friday + 2 * weekend_day
  let monday_to_wednesday := total - thursday_to_sunday
  let wednesday := monday_to_wednesday - (monday + tuesday)
  wednesday - monday

#check happy_street_traffic 25 10 10 5 97 = 2

theorem happy_street_traffic_proof : 
  happy_street_traffic 25 10 10 5 97 = 2 := by
sorry

end NUMINAMATH_CALUDE_happy_street_traffic_happy_street_traffic_proof_l2186_218698


namespace NUMINAMATH_CALUDE_base_three_sum_l2186_218626

/-- Represents a number in base 3 --/
def BaseThree : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def toDecimal (n : BaseThree) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The theorem to prove --/
theorem base_three_sum :
  let a : BaseThree := [2]
  let b : BaseThree := [0, 2, 1]
  let c : BaseThree := [1, 2, 0, 2]
  let d : BaseThree := [2, 0, 1, 1]
  let result : BaseThree := [2, 2, 2, 2]
  toDecimal a + toDecimal b + toDecimal c + toDecimal d = toDecimal result :=
by sorry

end NUMINAMATH_CALUDE_base_three_sum_l2186_218626


namespace NUMINAMATH_CALUDE_matthews_friends_l2186_218695

/-- Given that Matthew had 30 cakes and each person ate 15 cakes, 
    prove that the number of friends he shared with is 2. -/
theorem matthews_friends (total_cakes : ℕ) (cakes_per_person : ℕ) 
  (h1 : total_cakes = 30) 
  (h2 : cakes_per_person = 15) :
  total_cakes / cakes_per_person = 2 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l2186_218695


namespace NUMINAMATH_CALUDE_count_congruent_is_71_l2186_218679

/-- The number of positive integers less than 500 that are congruent to 3 (mod 7) -/
def count_congruent : ℕ :=
  (Finset.filter (fun n => n % 7 = 3) (Finset.range 500)).card

/-- Theorem: The count of positive integers less than 500 that are congruent to 3 (mod 7) is 71 -/
theorem count_congruent_is_71 : count_congruent = 71 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_is_71_l2186_218679


namespace NUMINAMATH_CALUDE_calculation_proof_l2186_218696

theorem calculation_proof : (3.6 * 0.3) / 0.2 = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2186_218696


namespace NUMINAMATH_CALUDE_brinley_zoo_count_l2186_218676

/-- The number of animals Brinley counted at the San Diego Zoo --/
def total_animals (snakes arctic_foxes leopards bee_eaters cheetahs alligators : ℕ) : ℕ :=
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

/-- Theorem stating the total number of animals Brinley counted at the zoo --/
theorem brinley_zoo_count : ∃ (snakes arctic_foxes leopards bee_eaters cheetahs alligators : ℕ),
  snakes = 100 ∧
  arctic_foxes = 80 ∧
  leopards = 20 ∧
  bee_eaters = 10 * leopards ∧
  cheetahs = snakes / 2 ∧
  alligators = 2 * (arctic_foxes + leopards) ∧
  total_animals snakes arctic_foxes leopards bee_eaters cheetahs alligators = 650 :=
by
  sorry


end NUMINAMATH_CALUDE_brinley_zoo_count_l2186_218676


namespace NUMINAMATH_CALUDE_cubic_root_of_unity_expression_l2186_218667

theorem cubic_root_of_unity_expression : 
  ∀ ω : ℂ, ω ^ 3 = 1 → ω ≠ (1 : ℂ) → (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_of_unity_expression_l2186_218667


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2186_218641

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α + π/4) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2186_218641


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l2186_218690

theorem root_sum_reciprocals (p q r s : ℂ) : 
  p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0 →
  q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0 →
  r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0 →
  s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l2186_218690


namespace NUMINAMATH_CALUDE_number_divisibility_l2186_218680

theorem number_divisibility (n m : ℤ) : 
  n = 859622 ∧ m = 859560 → 
  ∃ k : ℤ, k ≠ 0 ∧ m = n + (-62) ∧ m % k = 0 :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l2186_218680


namespace NUMINAMATH_CALUDE_square_side_length_l2186_218666

theorem square_side_length (width height : ℝ) (h1 : width = 3320) (h2 : height = 2025) : ∃ (r s : ℝ),
  2 * r + s = height ∧
  2 * r + 3 * s = width ∧
  s = 647.5 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2186_218666


namespace NUMINAMATH_CALUDE_angle_bac_equals_arcsin_four_fifths_l2186_218609

-- Define the triangle ABC and point O
structure Triangle :=
  (A B C O : ℝ × ℝ)

-- Define the distances OA, OB, OC
def distOA (t : Triangle) : ℝ := 15
def distOB (t : Triangle) : ℝ := 12
def distOC (t : Triangle) : ℝ := 20

-- Define the property that the feet of perpendiculars form an equilateral triangle
def perpendicularsFormEquilateralTriangle (t : Triangle) : Prop := sorry

-- Define the angle BAC
def angleBac (t : Triangle) : ℝ := sorry

-- State the theorem
theorem angle_bac_equals_arcsin_four_fifths (t : Triangle) :
  distOA t = 15 →
  distOB t = 12 →
  distOC t = 20 →
  perpendicularsFormEquilateralTriangle t →
  angleBac t = Real.arcsin (4/5) :=
by sorry

end NUMINAMATH_CALUDE_angle_bac_equals_arcsin_four_fifths_l2186_218609


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l2186_218612

/-- The value of m for which the parabola y = x^2 + 4 is tangent to the hyperbola y^2 - mx^2 = 4 -/
def tangency_value : ℝ := 8

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 + 4

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 4

/-- Theorem stating that the parabola is tangent to the hyperbola if and only if m = 8 -/
theorem parabola_tangent_hyperbola :
  ∀ (m : ℝ), (∃ (x : ℝ), hyperbola m x (parabola x) ∧
    ∀ (x' : ℝ), x' ≠ x → ¬(hyperbola m x' (parabola x'))) ↔ m = tangency_value :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l2186_218612


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l2186_218674

theorem quadratic_root_implies_k (k : ℚ) : 
  (4 * ((-15 - Real.sqrt 165) / 8)^2 + 15 * ((-15 - Real.sqrt 165) / 8) + k = 0) → 
  k = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l2186_218674
