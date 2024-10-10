import Mathlib

namespace certain_fraction_proof_l1764_176474

theorem certain_fraction_proof (x y : ℚ) : 
  x / y ≠ 0 → -- Ensure division by y is valid
  (x / y) / (1 / 5) = (3 / 4) / (2 / 5) →
  x / y = 8 / 3 := by
sorry

end certain_fraction_proof_l1764_176474


namespace system_solution_l1764_176404

theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 13 →
  x^2 + y^2 + z^2 = 61 →
  x*y + x*z = 2*y*z →
  ((x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3)) :=
by sorry

end system_solution_l1764_176404


namespace maryville_population_increase_l1764_176455

/-- The average annual population increase in Maryville between 2000 and 2005 -/
def average_annual_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating that the average annual population increase in Maryville between 2000 and 2005 is 3400 -/
theorem maryville_population_increase :
  average_annual_increase 450000 467000 = 3400 := by
  sorry

end maryville_population_increase_l1764_176455


namespace cubic_system_unique_solution_l1764_176419

theorem cubic_system_unique_solution (x y : ℝ) 
  (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
by sorry

end cubic_system_unique_solution_l1764_176419


namespace razorback_shop_tshirt_profit_l1764_176464

/-- The amount of money made per t-shirt, given the number of t-shirts sold and the total revenue from t-shirt sales. -/
def amount_per_tshirt (num_tshirts : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / num_tshirts

/-- Theorem stating that the amount made per t-shirt is $215, given the conditions. -/
theorem razorback_shop_tshirt_profit :
  amount_per_tshirt 20 4300 = 215 := by
  sorry

end razorback_shop_tshirt_profit_l1764_176464


namespace oscar_cd_distribution_l1764_176497

/-- Represents the number of CDs Oscar can pack in each box -/
def max_cds_per_box : ℕ := 2

/-- Represents the number of rock CDs Oscar needs to ship -/
def rock_cds : ℕ := 14

/-- Represents the number of pop CDs Oscar needs to ship -/
def pop_cds : ℕ := 8

/-- Theorem stating that for any non-negative integer n, if Oscar ships 2n classical CDs
    along with the rock and pop CDs, the total number of CDs can be evenly distributed
    into boxes of 2 CDs each -/
theorem oscar_cd_distribution (n : ℕ) :
  ∃ (total_boxes : ℕ), (rock_cds + 2*n + pop_cds) = max_cds_per_box * total_boxes :=
sorry

end oscar_cd_distribution_l1764_176497


namespace inequality_solution_range_l1764_176443

theorem inequality_solution_range (m : ℚ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
   (∀ x ∈ s, x < 0 ∧ (x - 1) / 2 + 3 > (x + m) / 3) ∧
   (∀ x : ℤ, x < 0 → (x - 1) / 2 + 3 > (x + m) / 3 → x ∈ s)) ↔ 
  (11 / 2 : ℚ) ≤ m ∧ m < 6 :=
sorry

end inequality_solution_range_l1764_176443


namespace prime_plus_three_prime_l1764_176496

theorem prime_plus_three_prime (p : ℕ) (hp : Nat.Prime p) (hp3 : Nat.Prime (p + 3)) :
  p^11 - 52 = 1996 := by
  sorry

end prime_plus_three_prime_l1764_176496


namespace integer_solution_2017_l1764_176463

theorem integer_solution_2017 (x y z : ℤ) : 
  x + y + z + x*y + y*z + z*x + x*y*z = 2017 ↔ 
  ((x = 0 ∧ y = 1 ∧ z = 1008) ∨
   (x = 0 ∧ y = 1008 ∧ z = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = 1008) ∨
   (x = 1 ∧ y = 1008 ∧ z = 0) ∨
   (x = 1008 ∧ y = 0 ∧ z = 1) ∨
   (x = 1008 ∧ y = 1 ∧ z = 0)) :=
by sorry

#check integer_solution_2017

end integer_solution_2017_l1764_176463


namespace sequence_problem_l1764_176414

theorem sequence_problem (n : ℕ) (a_n : ℕ → ℕ) : 
  (∀ k, a_n k = 3 * k + 4) → a_n n = 13 → n = 6 := by
  sorry

end sequence_problem_l1764_176414


namespace power_division_equals_integer_l1764_176454

theorem power_division_equals_integer : 3^18 / 27^3 = 19683 := by sorry

end power_division_equals_integer_l1764_176454


namespace message_difference_l1764_176407

/-- The number of messages sent by Lucia and Alina over three days -/
def total_messages : ℕ := 680

/-- The number of messages sent by Lucia on the first day -/
def lucia_day1 : ℕ := 120

/-- The number of messages sent by Alina on the first day -/
def alina_day1 : ℕ := lucia_day1 - 20

/-- Calculates the total number of messages sent over three days -/
def calculate_total (a : ℕ) : ℕ :=
  a + lucia_day1 +  -- Day 1
  (2 * a) + (lucia_day1 / 3) +  -- Day 2
  a + lucia_day1  -- Day 3

/-- Theorem stating that the difference between Lucia's and Alina's messages on the first day is 20 -/
theorem message_difference :
  calculate_total alina_day1 = total_messages ∧
  alina_day1 < lucia_day1 ∧
  lucia_day1 - alina_day1 = 20 :=
by sorry

end message_difference_l1764_176407


namespace arithmetic_sqrt_def_l1764_176499

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (a : ℝ) : ℝ := Real.sqrt a

-- State the theorem
theorem arithmetic_sqrt_def (a : ℝ) (h : 0 < a) : 
  arithmetic_sqrt a = Real.sqrt a := by sorry

end arithmetic_sqrt_def_l1764_176499


namespace arithmetic_sequence_61st_term_l1764_176485

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_15 : a 15 = 33
  a_45 : a 45 = 153

/-- Theorem: In the given arithmetic sequence, the 61st term is 217 -/
theorem arithmetic_sequence_61st_term (seq : ArithmeticSequence) : seq.a 61 = 217 := by
  sorry

end arithmetic_sequence_61st_term_l1764_176485


namespace exists_valid_coloring_l1764_176478

-- Define the colors
inductive Color
| Red
| Blue

-- Define the coloring function type
def ColoringFunction := ℕ → Color

-- Define an infinite arithmetic progression
def IsArithmeticProgression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define a property that a coloring function contains both colors in any arithmetic progression
def ContainsBothColors (f : ColoringFunction) : Prop :=
  ∀ (a : ℕ → ℕ) (d : ℕ), 
    IsArithmeticProgression a d → 
    (∃ n : ℕ, f (a n) = Color.Red) ∧ (∃ m : ℕ, f (a m) = Color.Blue)

-- The main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction, ContainsBothColors f := by
  sorry

end exists_valid_coloring_l1764_176478


namespace coin_distribution_l1764_176427

theorem coin_distribution (x y z : ℕ) : 
  x + 2*y + 5*z = 71 →  -- total value is 71 kopecks
  x = y →  -- number of 1-kopeck coins equals number of 2-kopeck coins
  x + y + z = 31 →  -- total number of coins is 31
  (x = 12 ∧ y = 12 ∧ z = 7) :=
by sorry

end coin_distribution_l1764_176427


namespace largest_package_size_l1764_176461

theorem largest_package_size (liam_markers zoe_markers : ℕ) 
  (h1 : liam_markers = 60) 
  (h2 : zoe_markers = 36) : 
  Nat.gcd liam_markers zoe_markers = 12 := by
sorry

end largest_package_size_l1764_176461


namespace quadratic_inequality_l1764_176480

/-- A quadratic function with a symmetry axis at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry axis of f is at x = 2 -/
def symmetry_axis (b c : ℝ) : Prop := ∀ x : ℝ, f b c (2 - x) = f b c (2 + x)

theorem quadratic_inequality (b c : ℝ) (h : symmetry_axis b c) : 
  f b c 2 > f b c 1 ∧ f b c 1 > f b c 4 :=
sorry

end quadratic_inequality_l1764_176480


namespace find_certain_number_l1764_176488

theorem find_certain_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((x + 70 + 16) / 3) + 8 → x = 10 := by
sorry

end find_certain_number_l1764_176488


namespace least_possible_smallest_integer_l1764_176420

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- They are distinct
  (h_average : (a + b + c + d) / 4 = 70) -- Their average is 70
  (h_largest : d = 90 ∧ d ≥ a ∧ d ≥ b ∧ d ≥ c) -- d is the largest and equals 90
  : a ≥ 184 -- The smallest integer is at least 184
:= by sorry

end least_possible_smallest_integer_l1764_176420


namespace N_rightmost_ten_l1764_176437

/-- A number with 1999 digits where each pair of consecutive digits
    is either a multiple of 17 or 23, and the sum of all digits is 9599 -/
def N : ℕ :=
  sorry

/-- Checks if a two-digit number is a multiple of 17 or 23 -/
def is_valid_pair (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- The property that each pair of consecutive digits in N
    is either a multiple of 17 or 23 -/
def valid_pairs (n : ℕ) : Prop :=
  ∀ i, i < 1998 → is_valid_pair ((n / 10^i) % 100)

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The rightmost ten digits of a natural number -/
def rightmost_ten (n : ℕ) : ℕ :=
  n % 10^10

theorem N_rightmost_ten :
  N ≥ 10^1998 ∧
  N < 10^1999 ∧
  valid_pairs N ∧
  digit_sum N = 9599 →
  rightmost_ten N = 3469234685 :=
sorry

end N_rightmost_ten_l1764_176437


namespace school_boys_count_l1764_176417

theorem school_boys_count (total_pupils : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_pupils = 485 → girls = 232 → boys = total_pupils - girls → boys = 253 := by
  sorry

end school_boys_count_l1764_176417


namespace xy_neq_6_sufficient_not_necessary_l1764_176402

theorem xy_neq_6_sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x * y ≠ 6 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 3) ∧ x * y = 6) :=
sorry

end xy_neq_6_sufficient_not_necessary_l1764_176402


namespace brothers_ticket_cost_l1764_176423

/-- Proves that each brother's ticket costs $10 given the problem conditions -/
theorem brothers_ticket_cost (isabelle_ticket_cost : ℕ) 
  (total_savings : ℕ) (weeks_worked : ℕ) (wage_per_week : ℕ) :
  isabelle_ticket_cost = 20 →
  total_savings = 10 →
  weeks_worked = 10 →
  wage_per_week = 3 →
  let total_money := total_savings + weeks_worked * wage_per_week
  let remaining_money := total_money - isabelle_ticket_cost
  remaining_money / 2 = 10 := by
  sorry

end brothers_ticket_cost_l1764_176423


namespace allison_video_upload_ratio_l1764_176495

/-- Represents the problem of calculating the ratio of days Allison uploaded videos at her initial pace to the total days in June. -/
theorem allison_video_upload_ratio :
  ∀ (x y : ℕ), 
    x + y = 30 →  -- Total days in June
    10 * x + 20 * y = 450 →  -- Total video hours uploaded
    (x : ℚ) / 30 = 1 / 2 :=
by sorry

end allison_video_upload_ratio_l1764_176495


namespace min_rectangle_dimensions_l1764_176435

/-- A rectangle with length twice its width and area at least 500 square feet has minimum dimensions of width = 5√10 feet and length = 10√10 feet. -/
theorem min_rectangle_dimensions (w : ℝ) (h : w > 0) :
  (2 * w ^ 2 ≥ 500) → (∀ x > 0, 2 * x ^ 2 ≥ 500 → w ≤ x) → w = 5 * Real.sqrt 10 :=
by sorry

end min_rectangle_dimensions_l1764_176435


namespace total_spent_is_correct_l1764_176475

def total_spent (robert_pens julia_pens_multiplier dorothy_pens_divisor : ℕ)
                (pen_cost : ℚ)
                (robert_pencils julia_pencils_difference dorothy_pencils_multiplier : ℕ)
                (pencil_cost : ℚ)
                (dorothy_notebooks julia_notebooks_addition robert_notebooks_divisor : ℕ)
                (notebook_cost : ℚ) : ℚ :=
  let julia_pens := julia_pens_multiplier * robert_pens
  let dorothy_pens := julia_pens / dorothy_pens_divisor
  let total_pens := robert_pens + julia_pens + dorothy_pens
  let pens_cost := (total_pens : ℚ) * pen_cost

  let julia_pencils := robert_pencils - julia_pencils_difference
  let dorothy_pencils := dorothy_pencils_multiplier * julia_pencils
  let total_pencils := robert_pencils + julia_pencils + dorothy_pencils
  let pencils_cost := (total_pencils : ℚ) * pencil_cost

  let julia_notebooks := dorothy_notebooks + julia_notebooks_addition
  let robert_notebooks := julia_notebooks / robert_notebooks_divisor
  let total_notebooks := dorothy_notebooks + julia_notebooks + robert_notebooks
  let notebooks_cost := (total_notebooks : ℚ) * notebook_cost

  pens_cost + pencils_cost + notebooks_cost

theorem total_spent_is_correct :
  total_spent 4 3 2 (3/2) 12 5 2 (3/4) 3 1 2 4 = 93.75 := by sorry

end total_spent_is_correct_l1764_176475


namespace bulletin_board_width_l1764_176401

/-- Proves that a rectangular bulletin board with area 6400 cm² and length 160 cm has a width of 40 cm -/
theorem bulletin_board_width :
  ∀ (area length width : ℝ),
  area = 6400 ∧ length = 160 ∧ area = length * width →
  width = 40 := by
  sorry

end bulletin_board_width_l1764_176401


namespace f_sum_equals_four_l1764_176442

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x^5 + c * x^3 + 2

-- State the theorem
theorem f_sum_equals_four (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 := by
  sorry

end f_sum_equals_four_l1764_176442


namespace binomial_10_2_l1764_176446

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end binomial_10_2_l1764_176446


namespace profit_sharing_ratio_l1764_176440

/-- Represents the capital contribution and time invested by a partner --/
structure Investment where
  capital : ℕ
  months : ℕ

/-- Calculates the capital-months for an investment --/
def capitalMonths (inv : Investment) : ℕ := inv.capital * inv.months

theorem profit_sharing_ratio 
  (a_investment : Investment) 
  (b_investment : Investment) 
  (h1 : a_investment.capital = 3500) 
  (h2 : a_investment.months = 12) 
  (h3 : b_investment.capital = 10500) 
  (h4 : b_investment.months = 6) :
  (capitalMonths a_investment) / (capitalMonths a_investment).gcd (capitalMonths b_investment) = 2 ∧ 
  (capitalMonths b_investment) / (capitalMonths a_investment).gcd (capitalMonths b_investment) = 3 :=
sorry

end profit_sharing_ratio_l1764_176440


namespace range_of_p_l1764_176411

-- Define the set A
def A (p : ℝ) : Set ℝ := {x | x^2 + (p+2)*x + 1 = 0}

-- Theorem statement
theorem range_of_p (p : ℝ) : (A p ∩ Set.Ioi 0 = ∅) → p > -4 := by
  sorry

end range_of_p_l1764_176411


namespace sum_of_a_and_b_l1764_176406

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 3 * a + 5 * b = 47) 
  (eq2 : 4 * a + 2 * b = 38) : 
  a + b = 85 / 7 := by
sorry

end sum_of_a_and_b_l1764_176406


namespace half_abs_diff_squares_20_15_l1764_176424

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_20_15_l1764_176424


namespace smallest_prime_factor_in_C_l1764_176403

def C : Finset Nat := {67, 71, 73, 76, 85}

theorem smallest_prime_factor_in_C : 
  ∃ (n : Nat), n ∈ C ∧ (∀ m ∈ C, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 76 := by
  sorry

end smallest_prime_factor_in_C_l1764_176403


namespace odd_function_negative_domain_l1764_176429

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = -x^2 - 2*x := by
sorry

end odd_function_negative_domain_l1764_176429


namespace jessica_paper_count_l1764_176434

def paper_weight : ℚ := 1/5
def envelope_weight : ℚ := 2/5

def total_weight (num_papers : ℕ) : ℚ :=
  paper_weight * num_papers + envelope_weight

theorem jessica_paper_count :
  ∃ (num_papers : ℕ),
    (1 < total_weight num_papers) ∧
    (total_weight num_papers ≤ 2) ∧
    (num_papers = 8) := by
  sorry

end jessica_paper_count_l1764_176434


namespace bike_ride_distance_l1764_176473

/-- Calculates the total distance traveled in a 3-hour bike ride given specific conditions -/
theorem bike_ride_distance (second_hour_distance : ℝ) 
  (h1 : second_hour_distance = 12)
  (h2 : second_hour_distance = 1.2 * (second_hour_distance / 1.2))
  (h3 : 1.25 * second_hour_distance = 15) : 
  (second_hour_distance / 1.2) + second_hour_distance + (1.25 * second_hour_distance) = 37 := by
  sorry

#check bike_ride_distance

end bike_ride_distance_l1764_176473


namespace function_property_l1764_176425

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

theorem function_property (a b : ℝ) :
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = f a b x₂) →
  f a b (2 * a) = f a b (3 * b) →
  a + b = -Real.sqrt 6 / 2 + 3 := by
  sorry

end function_property_l1764_176425


namespace marble_distribution_l1764_176445

/-- Given the distribution of marbles between two classes, prove the difference between
    the number of marbles each male in Class 2 receives and the total number of marbles
    taken by Class 1. -/
theorem marble_distribution (total_marbles : ℕ) (class1_marbles : ℕ) (class2_marbles : ℕ)
  (boys_marbles : ℕ) (girls_marbles : ℕ) (num_boys : ℕ) :
  total_marbles = 1000 →
  class1_marbles = class2_marbles + 50 →
  class1_marbles + class2_marbles = total_marbles →
  boys_marbles = girls_marbles + 35 →
  boys_marbles + girls_marbles = class2_marbles →
  num_boys = 17 →
  class1_marbles - (boys_marbles / num_boys) = 510 :=
by sorry

end marble_distribution_l1764_176445


namespace total_population_l1764_176491

/-- Represents the number of boys, girls, and teachers in a school -/
structure School where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls
  t : ℕ  -- number of teachers

/-- The conditions of the school population -/
def school_conditions (s : School) : Prop :=
  s.b = 4 * s.g ∧ s.g = 5 * s.t

/-- The theorem stating that the total population is 26 times the number of teachers -/
theorem total_population (s : School) (h : school_conditions s) : 
  s.b + s.g + s.t = 26 * s.t := by
  sorry

end total_population_l1764_176491


namespace limit_calculation_l1764_176481

open Real

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f (1 - 2*Δx)) / Δx + 3/exp 1| < ε :=
sorry

end limit_calculation_l1764_176481


namespace trailer_count_proof_l1764_176436

theorem trailer_count_proof (initial_count : ℕ) (initial_avg_age : ℝ) (current_avg_age : ℝ) :
  initial_count = 30 ∧ initial_avg_age = 12 ∧ current_avg_age = 10 →
  ∃ (new_count : ℕ), 
    (initial_count * (initial_avg_age + 4) + new_count * 4) / (initial_count + new_count) = current_avg_age ∧
    new_count = 30 := by
  sorry

end trailer_count_proof_l1764_176436


namespace f_properties_l1764_176426

-- Define the function f(x) = x³ - 3x² + 3
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem f_properties :
  -- 1. The tangent line at (1, f(1)) is 3x + y - 4 = 0
  (∀ x y : ℝ, y = f' 1 * (x - 1) + f 1 ↔ 3*x + y - 4 = 0) ∧
  -- 2. The function has exactly 3 zeros
  (∃! (a b c : ℝ), a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  -- 3. The function is symmetric about the point (1, 1)
  (∀ x : ℝ, f (1 + x) - 1 = -(f (1 - x) - 1)) :=
sorry

end f_properties_l1764_176426


namespace cindy_earnings_l1764_176468

/-- Calculates the earnings for teaching one math course in a month -/
def earnings_per_course (total_courses : ℕ) (total_hours_per_week : ℕ) (hourly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (total_hours_per_week / total_courses) * weeks_per_month * hourly_rate

/-- Theorem: Cindy's earnings for one math course in a month is $1200 -/
theorem cindy_earnings : 
  earnings_per_course 4 48 25 4 = 1200 := by
  sorry

end cindy_earnings_l1764_176468


namespace average_growth_rate_proof_l1764_176422

def initial_sales : ℝ := 50000
def final_sales : ℝ := 72000
def time_period : ℝ := 2

theorem average_growth_rate_proof :
  (final_sales / initial_sales) ^ (1 / time_period) - 1 = 0.2 := by
  sorry

end average_growth_rate_proof_l1764_176422


namespace vector_relations_l1764_176409

def a : ℝ × ℝ := (6, 2)
def b : ℝ → ℝ × ℝ := λ k => (-2, k)

theorem vector_relations (k : ℝ) :
  (∃ (c : ℝ), b k = c • a) → k = -2/3 ∧
  (a.1 * (b k).1 + a.2 * (b k).2 = 0) → k = 6 ∧
  (a.1 * (b k).1 + a.2 * (b k).2 < 0 ∧ ¬∃ (c : ℝ), b k = c • a) → k < 6 ∧ k ≠ -2/3 :=
by sorry

end vector_relations_l1764_176409


namespace polynomial_value_theorem_l1764_176459

theorem polynomial_value_theorem (x : ℝ) (h : x^2 - 8*x - 3 = 0) :
  (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 := by
  sorry

end polynomial_value_theorem_l1764_176459


namespace original_game_points_l1764_176428

/-- The number of points in the original game -/
def P : ℕ := 60

/-- X can give Y 20 points in a game of P points -/
def X_gives_Y (p : ℕ) : Prop := p - 20 > 0

/-- X can give Z 30 points in a game of P points -/
def X_gives_Z (p : ℕ) : Prop := p - 30 > 0

/-- In a game of 120 points, Y can give Z 30 points -/
def Y_gives_Z_120 : Prop := 120 - 30 > 0

/-- The ratio of scores when Y and Z play against X is equal to their ratio in a 120-point game -/
def score_ratio (p : ℕ) : Prop := (p - 20) * 90 = (p - 30) * 120

theorem original_game_points :
  X_gives_Y P ∧ X_gives_Z P ∧ Y_gives_Z_120 ∧ score_ratio P → P = 60 :=
by sorry

end original_game_points_l1764_176428


namespace x_coordinate_difference_at_y_20_l1764_176412

/-- A line in a 2D coordinate system --/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Calculate the x-coordinate for a given y-coordinate on a line --/
def xCoordAtY (line : Line) (y : ℚ) : ℚ :=
  (y - line.intercept) / line.slope

/-- Create a line from two points --/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  intercept := y1 - (y2 - y1) / (x2 - x1) * x1

theorem x_coordinate_difference_at_y_20 :
  let l := lineFromPoints 0 5 3 0
  let m := lineFromPoints 0 4 6 0
  let x_l := xCoordAtY l 20
  let x_m := xCoordAtY m 20
  |x_l - x_m| = 15 := by sorry

end x_coordinate_difference_at_y_20_l1764_176412


namespace polynomial_simplification_l1764_176405

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 9 * x^9 + 5 * x^8) + (2 * x^12 + x^10 + 2 * x^9 + 3 * x^8 + 4 * x^4 + 6 * x^2 + 9) =
  2 * x^12 + 13 * x^10 + 11 * x^9 + 8 * x^8 + 4 * x^4 + 6 * x^2 + 9 := by
  sorry

end polynomial_simplification_l1764_176405


namespace simplify_sqrt_sum_l1764_176479

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l1764_176479


namespace statement_is_proposition_l1764_176489

-- Define what a proposition is
def is_proposition (s : Prop) : Prop := True

-- Define the statement we're examining
def statement : Prop := ∀ a : ℤ, Prime a → Odd a

-- Theorem stating that our statement is a proposition
theorem statement_is_proposition : is_proposition statement := by sorry

end statement_is_proposition_l1764_176489


namespace square_root_of_81_l1764_176467

theorem square_root_of_81 : 
  {x : ℝ | x^2 = 81} = {9, -9} := by sorry

end square_root_of_81_l1764_176467


namespace constant_digit_sum_characterization_l1764_176431

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Characterization of numbers with constant digit sum property -/
theorem constant_digit_sum_characterization (M : ℕ) :
  (M > 0 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → S (M * k) = S M) ↔
  (M = 1 ∨ ∃ n : ℕ, M = 10^n - 1) :=
sorry

end constant_digit_sum_characterization_l1764_176431


namespace tan_alpha_equals_one_l1764_176486

theorem tan_alpha_equals_one (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end tan_alpha_equals_one_l1764_176486


namespace cookies_bought_l1764_176453

theorem cookies_bought (total_groceries cake_packs : ℕ) 
  (h1 : total_groceries = 14)
  (h2 : cake_packs = 12)
  (h3 : ∃ cookie_packs : ℕ, cookie_packs + cake_packs = total_groceries) :
  ∃ cookie_packs : ℕ, cookie_packs = 2 ∧ cookie_packs + cake_packs = total_groceries :=
by
  sorry

end cookies_bought_l1764_176453


namespace empty_can_weight_is_two_l1764_176458

/-- Calculates the weight of each empty can given the total weight, number of soda cans, soda weight per can, and number of empty cans. -/
def empty_can_weight (total_weight : ℕ) (soda_cans : ℕ) (soda_weight_per_can : ℕ) (empty_cans : ℕ) : ℕ :=
  (total_weight - soda_cans * soda_weight_per_can) / (soda_cans + empty_cans)

/-- Proves that each empty can weighs 2 ounces given the problem conditions. -/
theorem empty_can_weight_is_two :
  empty_can_weight 88 6 12 2 = 2 := by
  sorry

end empty_can_weight_is_two_l1764_176458


namespace largest_812_double_l1764_176487

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-12 number -/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is an 8-12 double -/
def is812Double (n : ℕ) : Prop :=
  fromBase12 (toBase8 n) = 2 * n

theorem largest_812_double :
  (∀ m : ℕ, is812Double m → m ≤ 4032) ∧ is812Double 4032 :=
sorry

end largest_812_double_l1764_176487


namespace square_perimeter_sum_l1764_176447

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 42) :
  4*x + 4*y = 4*Real.sqrt 86 + 8*Real.sqrt 11 := by
  sorry

end square_perimeter_sum_l1764_176447


namespace total_books_l1764_176466

-- Define the number of books for each person
def harry_books : ℕ := 50
def flora_books : ℕ := 2 * harry_books
def gary_books : ℕ := harry_books / 2

-- Theorem to prove
theorem total_books : harry_books + flora_books + gary_books = 175 := by
  sorry

end total_books_l1764_176466


namespace parallel_line_through_point_l1764_176438

/-- Given a line L1 with equation x - y + 1 = 0 and a point P (2, -4),
    prove that the line L2 passing through P and parallel to L1
    has the equation x - y - 6 = 0 -/
theorem parallel_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - y + 1 = 0
  let P : ℝ × ℝ := (2, -4)
  let L2 : ℝ → ℝ → Prop := λ x y => x - y - 6 = 0
  (∀ x y, L2 x y ↔ (x - y = 6)) ∧
  L2 P.1 P.2 ∧
  (∀ x1 y1 x2 y2, L1 x1 y1 ∧ L2 x2 y2 → x1 - y1 = x2 - y2) :=
by sorry

end parallel_line_through_point_l1764_176438


namespace problem_solution_l1764_176483

theorem problem_solution (x y : ℝ) 
  (h1 : x = 52) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y + 100 = 540000) : 
  y = 10 := by
  sorry

end problem_solution_l1764_176483


namespace expression_value_l1764_176408

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (x * y + 1)) : 
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (x * y + 1) = 2 := by
  sorry

end expression_value_l1764_176408


namespace contrapositive_truth_square_less_than_one_implies_absolute_less_than_one_contrapositive_of_square_less_than_one_is_true_l1764_176469

theorem contrapositive_truth (P Q : Prop) :
  (P → Q) → (¬Q → ¬P) := by sorry

theorem square_less_than_one_implies_absolute_less_than_one :
  ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1 := by sorry

theorem contrapositive_of_square_less_than_one_is_true :
  (∀ x : ℝ, ¬(-1 < x ∧ x < 1) → ¬(x^2 < 1)) := by sorry

end contrapositive_truth_square_less_than_one_implies_absolute_less_than_one_contrapositive_of_square_less_than_one_is_true_l1764_176469


namespace round_trip_speed_calculation_l1764_176477

/-- Proves that given specific conditions for a round trip, the return speed is 37.5 mph -/
theorem round_trip_speed_calculation (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) :
  distance = 150 →
  speed_ab = 75 →
  avg_speed = 50 →
  (2 * distance) / (distance / speed_ab + distance / ((2 * distance) / (2 * distance / avg_speed - distance / speed_ab))) = avg_speed →
  (2 * distance) / (2 * distance / avg_speed - distance / speed_ab) = 37.5 := by
  sorry

end round_trip_speed_calculation_l1764_176477


namespace sam_seashells_l1764_176470

/-- The number of seashells Sam has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sam has 17 seashells after giving away 18 from his initial 35 -/
theorem sam_seashells : remaining_seashells 35 18 = 17 := by
  sorry

end sam_seashells_l1764_176470


namespace recess_time_calculation_l1764_176449

/-- Calculates the total recess time based on grade distribution -/
def total_recess_time (base_time : ℕ) (a_count b_count c_count d_count : ℕ) : ℕ :=
  base_time + 2 * a_count + b_count - d_count

/-- Theorem stating that given the specific grade distribution, the total recess time is 47 minutes -/
theorem recess_time_calculation :
  let base_time : ℕ := 20
  let a_count : ℕ := 10
  let b_count : ℕ := 12
  let c_count : ℕ := 14
  let d_count : ℕ := 5
  total_recess_time base_time a_count b_count c_count d_count = 47 := by
  sorry

#eval total_recess_time 20 10 12 14 5

end recess_time_calculation_l1764_176449


namespace sin_alpha_plus_7pi_over_6_l1764_176452

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = (4/5) * Real.sqrt 3) : 
  Real.sin (α + 7*π/6) = -4/5 := by sorry

end sin_alpha_plus_7pi_over_6_l1764_176452


namespace reflection_matrix_values_l1764_176444

theorem reflection_matrix_values (a b : ℚ) :
  let R : Matrix (Fin 2) (Fin 2) ℚ := !![a, 9/26; b, 17/26]
  (R * R = 1) → (a = -17/26 ∧ b = 0) := by
  sorry

end reflection_matrix_values_l1764_176444


namespace brownie_pieces_theorem_l1764_176451

/-- The number of big square pieces the brownies were cut into -/
def num_pieces : ℕ := sorry

/-- The total amount of money Tamara made from selling brownies -/
def total_amount : ℕ := 32

/-- The cost of each brownie -/
def cost_per_brownie : ℕ := 2

/-- The number of pans of brownies made -/
def num_pans : ℕ := 2

theorem brownie_pieces_theorem :
  num_pieces = total_amount / cost_per_brownie :=
sorry

end brownie_pieces_theorem_l1764_176451


namespace inequality_implication_l1764_176432

theorem inequality_implication (a b : ℝ) (h : a > b) : a + 2 > b + 1 := by
  sorry

#check inequality_implication

end inequality_implication_l1764_176432


namespace rectangular_stadium_length_l1764_176492

theorem rectangular_stadium_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 800) 
  (h2 : breadth = 300) : 
  2 * (breadth + 100) = perimeter :=
by sorry

end rectangular_stadium_length_l1764_176492


namespace weight_of_top_l1764_176410

/-- Given 9 robots each weighing 0.8 kg and 7 tops with a total weight of 10.98 kg,
    the weight of one top is 0.54 kg. -/
theorem weight_of_top (robot_weight : ℝ) (total_weight : ℝ) (num_robots : ℕ) (num_tops : ℕ) :
  robot_weight = 0.8 →
  num_robots = 9 →
  num_tops = 7 →
  total_weight = 10.98 →
  total_weight = (↑num_robots * robot_weight) + (↑num_tops * 0.54) :=
by sorry

end weight_of_top_l1764_176410


namespace prob_catch_carp_l1764_176430

/-- The probability of catching a carp in a pond with given conditions -/
theorem prob_catch_carp (num_carp num_tilapia : ℕ) (prob_grass_carp : ℚ) : 
  num_carp = 1600 →
  num_tilapia = 800 →
  prob_grass_carp = 1/2 →
  (num_carp : ℚ) / (num_carp + num_tilapia + (prob_grass_carp⁻¹ - 1) * (num_carp + num_tilapia)) = 1/3 :=
by sorry

end prob_catch_carp_l1764_176430


namespace three_digit_45_arithmetic_sequence_l1764_176413

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b = (a + c) / 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_45_arithmetic_sequence :
  ∀ n : ℕ, is_three_digit n →
            n % 45 = 0 →
            is_arithmetic_sequence (n / 100) ((n / 10) % 10) (n % 10) →
            (n = 135 ∨ n = 630 ∨ n = 765) :=
sorry

end three_digit_45_arithmetic_sequence_l1764_176413


namespace jack_book_sale_l1764_176418

/-- Calculates the amount received from selling books after a year --/
def amount_received (books_per_month : ℕ) (cost_per_book : ℕ) (months : ℕ) (loss : ℕ) : ℕ :=
  books_per_month * months * cost_per_book - loss

/-- Proves that Jack received $500 from selling the books --/
theorem jack_book_sale : amount_received 3 20 12 220 = 500 := by
  sorry

end jack_book_sale_l1764_176418


namespace convex_nonagon_diagonals_l1764_176494

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem convex_nonagon_diagonals : 
  nonagon_diagonals = 27 := by sorry

end convex_nonagon_diagonals_l1764_176494


namespace largest_b_in_box_l1764_176433

theorem largest_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 :=
sorry

end largest_b_in_box_l1764_176433


namespace eugene_model_house_l1764_176472

/-- Eugene's model house building problem --/
theorem eugene_model_house (toothpicks_per_card : ℕ) (cards_in_deck : ℕ) 
  (boxes_used : ℕ) (toothpicks_per_box : ℕ) : 
  toothpicks_per_card = 75 →
  cards_in_deck = 52 →
  boxes_used = 6 →
  toothpicks_per_box = 450 →
  cards_in_deck - (boxes_used * toothpicks_per_box) / toothpicks_per_card = 16 :=
by
  sorry

end eugene_model_house_l1764_176472


namespace bicycle_speed_correct_l1764_176484

/-- The speed of bicycles that satisfies the given conditions -/
def bicycle_speed : ℝ := 15

theorem bicycle_speed_correct :
  let distance : ℝ := 10
  let car_speed : ℝ → ℝ := λ x => 2 * x
  let bicycle_time : ℝ → ℝ := λ x => distance / x
  let car_time : ℝ → ℝ := λ x => distance / (car_speed x)
  let time_difference : ℝ := 1 / 3
  bicycle_time bicycle_speed = car_time bicycle_speed + time_difference :=
by sorry

end bicycle_speed_correct_l1764_176484


namespace equal_cost_at_48_miles_l1764_176416

-- Define the daily rates and per-mile charges
def sunshine_daily_rate : ℝ := 17.99
def sunshine_per_mile : ℝ := 0.18
def city_daily_rate : ℝ := 18.95
def city_per_mile : ℝ := 0.16

-- Define the cost functions for each rental company
def sunshine_cost (miles : ℝ) : ℝ := sunshine_daily_rate + sunshine_per_mile * miles
def city_cost (miles : ℝ) : ℝ := city_daily_rate + city_per_mile * miles

-- Theorem stating that the costs are equal at 48 miles
theorem equal_cost_at_48_miles :
  sunshine_cost 48 = city_cost 48 := by sorry

end equal_cost_at_48_miles_l1764_176416


namespace order_of_abc_l1764_176482

theorem order_of_abc : 
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  c < a ∧ a < b := by sorry

end order_of_abc_l1764_176482


namespace average_weight_l1764_176421

/-- Given three weights a, b, and c, prove that their average is 45 kg
    under the following conditions:
    1. The average of a and b is 40 kg
    2. The average of b and c is 43 kg
    3. The weight of b is 31 kg -/
theorem average_weight (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 40)
  (avg_bc : (b + c) / 2 = 43)
  (weight_b : b = 31) :
  (a + b + c) / 3 = 45 := by
  sorry


end average_weight_l1764_176421


namespace distinct_products_between_squares_l1764_176450

theorem distinct_products_between_squares (n a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 →
  a * d ≠ b * c :=
by sorry

end distinct_products_between_squares_l1764_176450


namespace circle_center_coordinates_l1764_176476

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem: The center coordinates of the circle x^2 + y^2 + 2x - 4y = 0 are (-1, 2) -/
theorem circle_center_coordinates :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 5 :=
by sorry

end circle_center_coordinates_l1764_176476


namespace probability_sum_seven_is_one_sixth_l1764_176456

def dice_faces : ℕ := 6

def favorable_outcomes : ℕ := 6

def total_outcomes : ℕ := dice_faces * dice_faces

def probability_sum_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_seven_is_one_sixth : 
  probability_sum_seven = 1 / 6 := by sorry

end probability_sum_seven_is_one_sixth_l1764_176456


namespace class_test_percentages_l1764_176462

theorem class_test_percentages
  (percent_first : ℝ)
  (percent_second : ℝ)
  (percent_both : ℝ)
  (h1 : percent_first = 75)
  (h2 : percent_second = 35)
  (h3 : percent_both = 30) :
  100 - (percent_first + percent_second - percent_both) = 20 := by
  sorry

end class_test_percentages_l1764_176462


namespace paris_weekday_study_hours_l1764_176441

/-- The number of hours Paris studies each weekday. -/
def weekday_study_hours : ℝ := 3

/-- The number of weeks in the fall semester. -/
def semester_weeks : ℕ := 15

/-- The number of hours Paris studies on Saturday. -/
def saturday_study_hours : ℝ := 4

/-- The number of hours Paris studies on Sunday. -/
def sunday_study_hours : ℝ := 5

/-- The total number of hours Paris studies during the semester. -/
def total_study_hours : ℝ := 360

/-- Theorem stating that the number of hours Paris studies each weekday is 3. -/
theorem paris_weekday_study_hours :
  weekday_study_hours * (5 * semester_weeks) +
  (saturday_study_hours + sunday_study_hours) * semester_weeks =
  total_study_hours :=
sorry

end paris_weekday_study_hours_l1764_176441


namespace arithmetic_mean_difference_l1764_176400

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 27) : 
  r - p = 34 := by
sorry

end arithmetic_mean_difference_l1764_176400


namespace max_n_value_l1764_176460

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : (a - b)⁻¹ + (b - c)⁻¹ ≥ n / (a - c)) : n ≤ 4 :=
by sorry

end max_n_value_l1764_176460


namespace class_size_difference_l1764_176490

theorem class_size_difference (students : ℕ) (teachers : ℕ) (enrollments : List ℕ) : 
  students = 120 →
  teachers = 6 →
  enrollments = [60, 30, 15, 5, 5, 5] →
  (enrollments.sum = students) →
  (enrollments.length = teachers) →
  let t : ℚ := (enrollments.sum : ℚ) / teachers
  let s : ℚ := (enrollments.map (λ x => x * x)).sum / students
  t - s = -20 := by
  sorry

#check class_size_difference

end class_size_difference_l1764_176490


namespace annie_money_left_l1764_176465

/-- Calculates the amount of money Annie has left after buying hamburgers and milkshakes. -/
def money_left (initial_money hamburger_price milkshake_price hamburger_count milkshake_count : ℕ) : ℕ :=
  initial_money - (hamburger_price * hamburger_count + milkshake_price * milkshake_count)

/-- Proves that Annie has $70 left after her purchases. -/
theorem annie_money_left :
  money_left 132 4 5 8 6 = 70 := by
  sorry

end annie_money_left_l1764_176465


namespace B_power_200_is_identity_l1764_176457

def B : Matrix (Fin 4) (Fin 4) ℝ := !![0,0,0,1; 1,0,0,0; 0,1,0,0; 0,0,1,0]

theorem B_power_200_is_identity :
  B ^ 200 = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end B_power_200_is_identity_l1764_176457


namespace radical_simplification_l1764_176415

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (45 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 675 * q * Real.sqrt q :=
by sorry

end radical_simplification_l1764_176415


namespace cubic_minus_four_ab_squared_factorization_l1764_176498

theorem cubic_minus_four_ab_squared_factorization (a b : ℝ) :
  a^3 - 4*a*b^2 = a*(a+2*b)*(a-2*b) := by sorry

end cubic_minus_four_ab_squared_factorization_l1764_176498


namespace rhombus_area_from_square_circumference_l1764_176493

/-- The area of a rhombus formed by connecting the midpoints of a square's sides,
    given the square's circumference. -/
theorem rhombus_area_from_square_circumference (circumference : ℝ) :
  circumference = 96 →
  let square_side := circumference / 4
  let rhombus_area := square_side^2 / 2
  rhombus_area = 288 := by
  sorry

end rhombus_area_from_square_circumference_l1764_176493


namespace years_until_double_age_l1764_176448

/-- Proves the number of years until a man's age is twice his son's age -/
theorem years_until_double_age (son_age : ℕ) (age_difference : ℕ) (years : ℕ) : 
  son_age = 44 → 
  age_difference = 46 → 
  (son_age + age_difference + years) = 2 * (son_age + years) → 
  years = 2 := by
sorry

end years_until_double_age_l1764_176448


namespace sqrt_x_plus_y_plus_five_halves_l1764_176439

theorem sqrt_x_plus_y_plus_five_halves (x y : ℝ) : 
  y = Real.sqrt (2 * x - 3) + Real.sqrt (3 - 2 * x) + 5 →
  Real.sqrt (x + y + 5 / 2) = 3 ∨ Real.sqrt (x + y + 5 / 2) = -3 :=
by sorry

end sqrt_x_plus_y_plus_five_halves_l1764_176439


namespace coffee_cost_theorem_l1764_176471

/-- The cost of coffee A per kilogram -/
def coffee_A_cost : ℝ := 10

/-- The cost of coffee B per kilogram -/
def coffee_B_cost : ℝ := 12

/-- The selling price of the mixture per kilogram -/
def mixture_price : ℝ := 11

/-- The total weight of the mixture in kilograms -/
def total_mixture : ℝ := 480

/-- The weight of coffee A used in the mixture in kilograms -/
def coffee_A_weight : ℝ := 240

/-- The weight of coffee B used in the mixture in kilograms -/
def coffee_B_weight : ℝ := 240

theorem coffee_cost_theorem :
  coffee_A_weight * coffee_A_cost + coffee_B_weight * coffee_B_cost = total_mixture * mixture_price :=
by sorry

end coffee_cost_theorem_l1764_176471
