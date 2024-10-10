import Mathlib

namespace tan_value_from_ratio_l2601_260193

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 2) : 
  Real.tan α = -12/5 := by
  sorry

end tan_value_from_ratio_l2601_260193


namespace price_relationship_total_cost_max_toy_A_l2601_260151

/- Define the unit prices of toys A and B -/
def price_A : ℕ := 50
def price_B : ℕ := 75

/- Define the relationship between prices -/
theorem price_relationship : price_B = price_A + 25 := by sorry

/- Define the total cost of 2B and 1A -/
theorem total_cost : 2 * price_B + price_A = 200 := by sorry

/- Define the function for total cost given number of A -/
def total_cost_function (num_A : ℕ) : ℕ := price_A * num_A + price_B * (2 * num_A)

/- Define the maximum budget -/
def max_budget : ℕ := 20000

/- Theorem to prove the maximum number of toy A that can be purchased -/
theorem max_toy_A : 
  (∀ n : ℕ, total_cost_function n ≤ max_budget → n ≤ 100) ∧ 
  total_cost_function 100 ≤ max_budget := by sorry

end price_relationship_total_cost_max_toy_A_l2601_260151


namespace quadratic_extremum_l2601_260157

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = -b^2 / (3a),
    prove that the graph of y = f(x) has a maximum if a < 0 and a minimum if a > 0 -/
theorem quadratic_extremum (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - b^2 / (3 * a)
  (a < 0 → ∃ x₀, ∀ x, f x ≤ f x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, f x ≥ f x₀) :=
by sorry


end quadratic_extremum_l2601_260157


namespace product_of_primes_minus_one_l2601_260164

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 0 ∧ m < n → n % m = 0 → m = 1

axiom every_nat_is_product_of_primes :
  ∀ n : Nat, n > 1 → ∃ (factors : List Nat), n = factors.prod ∧ ∀ p ∈ factors, isPrime p

theorem product_of_primes_minus_one (h : isPrime 11 ∧ isPrime 19) :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
by sorry

end product_of_primes_minus_one_l2601_260164


namespace problem_statement_l2601_260179

open Real

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log x - 2*a*x + 2*a

theorem problem_statement (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 1/4) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ →
    |g a x₁ - g a x₂| < 2*a*|1/x₁ - 1/x₂|) →
  a = 1/4 := by
sorry

end problem_statement_l2601_260179


namespace hotel_loss_calculation_l2601_260176

/-- Calculates the loss incurred by a hotel given its operations expenses and client payments --/
def hotel_loss (expenses : ℝ) (client_payment_ratio : ℝ) : ℝ :=
  expenses - (client_payment_ratio * expenses)

/-- Theorem: A hotel with $100 expenses and client payments of 3/4 of expenses incurs a $25 loss --/
theorem hotel_loss_calculation :
  hotel_loss 100 (3/4) = 25 := by
  sorry

end hotel_loss_calculation_l2601_260176


namespace first_year_interest_l2601_260110

def initial_deposit : ℝ := 1000
def first_year_balance : ℝ := 1100
def second_year_increase_rate : ℝ := 0.20
def total_increase_rate : ℝ := 0.32

theorem first_year_interest :
  let second_year_balance := first_year_balance * (1 + second_year_increase_rate)
  second_year_balance = initial_deposit * (1 + total_increase_rate) →
  first_year_balance - initial_deposit = 100 := by
sorry

end first_year_interest_l2601_260110


namespace not_all_zero_iff_at_least_one_nonzero_l2601_260106

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end not_all_zero_iff_at_least_one_nonzero_l2601_260106


namespace no_savings_on_joint_purchase_l2601_260190

/-- Calculates the number of paid windows given the total number of windows needed -/
def paidWindows (total : ℕ) : ℕ :=
  total - (total / 3)

/-- Calculates the cost of windows before any flat discount -/
def windowCost (paid : ℕ) : ℕ :=
  paid * 150

/-- Applies the flat discount if the cost is over 1000 -/
def applyDiscount (cost : ℕ) : ℕ :=
  if cost > 1000 then cost - 200 else cost

theorem no_savings_on_joint_purchase (dave_windows doug_windows : ℕ) 
  (h_dave : dave_windows = 9) (h_doug : doug_windows = 10) :
  let dave_cost := applyDiscount (windowCost (paidWindows dave_windows))
  let doug_cost := applyDiscount (windowCost (paidWindows doug_windows))
  let separate_cost := dave_cost + doug_cost
  let joint_windows := dave_windows + doug_windows
  let joint_cost := applyDiscount (windowCost (paidWindows joint_windows))
  separate_cost = joint_cost := by
  sorry

end no_savings_on_joint_purchase_l2601_260190


namespace correct_outfit_count_l2601_260150

/-- The number of outfits that can be made with given clothing items, 
    where shirts and hats cannot be the same color. -/
def number_of_outfits (red_shirts green_shirts pants blue_hats red_hats scarves : ℕ) : ℕ :=
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves)

/-- Theorem stating the correct number of outfits given specific quantities of clothing items. -/
theorem correct_outfit_count : 
  number_of_outfits 7 8 10 10 10 5 = 7500 := by
  sorry

end correct_outfit_count_l2601_260150


namespace brothers_ages_theorem_l2601_260175

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  kolya : ℕ
  vanya : ℕ
  petya : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.petya = 10 ∧
  ages.kolya = ages.petya + 3 ∧
  ages.vanya = ages.petya - 1

/-- The theorem to be proved -/
theorem brothers_ages_theorem (ages : BrothersAges) :
  satisfiesConditions ages → ages.vanya = 9 ∧ ages.kolya = 13 := by
  sorry

#check brothers_ages_theorem

end brothers_ages_theorem_l2601_260175


namespace max_intersections_l2601_260138

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The configuration of figures on the plane. -/
structure Configuration where
  circle : Circle
  lines : Fin 3 → Line

/-- The number of intersection points between a circle and a line. -/
def circleLineIntersections (c : Circle) (l : Line) : ℕ := sorry

/-- The number of intersection points between two lines. -/
def lineLineIntersections (l1 l2 : Line) : ℕ := sorry

/-- The total number of intersection points in a configuration. -/
def totalIntersections (config : Configuration) : ℕ := sorry

/-- The theorem stating that the maximum number of intersections is 9. -/
theorem max_intersections :
  ∃ (config : Configuration), totalIntersections config = 9 ∧
  ∀ (other : Configuration), totalIntersections other ≤ 9 :=
sorry

end max_intersections_l2601_260138


namespace ladder_length_proof_l2601_260186

theorem ladder_length_proof (ladder_length wall_height : ℝ) : 
  wall_height = ladder_length + 8/3 →
  ∃ (ladder_base ladder_top : ℝ),
    ladder_base = 3/5 * ladder_length ∧
    ladder_top = 2/5 * wall_height ∧
    ladder_length^2 = ladder_base^2 + ladder_top^2 →
  ladder_length = 8/3 := by
sorry

end ladder_length_proof_l2601_260186


namespace simplify_trig_expression_l2601_260145

theorem simplify_trig_expression (x : ℝ) : 
  ((1 + Real.sin x) / Real.cos x) * (Real.sin (2 * x) / (2 * (Real.cos (π/4 - x/2))^2)) = 2 * Real.sin x :=
by sorry

end simplify_trig_expression_l2601_260145


namespace average_of_solutions_is_zero_l2601_260189

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (5 * x^2 + 4) = Real.sqrt 29}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end average_of_solutions_is_zero_l2601_260189


namespace equation_to_lines_l2601_260130

theorem equation_to_lines :
  ∀ x y : ℝ,
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end equation_to_lines_l2601_260130


namespace intersection_complement_theorem_l2601_260187

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 1}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem intersection_complement_theorem :
  N ∩ (Mᶜ) = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_complement_theorem_l2601_260187


namespace xiaopang_mom_money_l2601_260119

/-- The price of apples per kilogram -/
def apple_price : ℝ := 5

/-- The amount of money Xiaopang's mom had -/
def total_money : ℝ := 21.5

/-- The amount of apples Xiaopang's mom wanted to buy initially -/
def initial_amount : ℝ := 5

/-- The amount of apples Xiaopang's mom actually bought -/
def actual_amount : ℝ := 4

/-- The amount of money Xiaopang's mom was short for the initial amount -/
def short_amount : ℝ := 3.5

/-- The amount of money Xiaopang's mom had left after buying the actual amount -/
def left_amount : ℝ := 1.5

theorem xiaopang_mom_money :
  total_money = actual_amount * apple_price + left_amount ∧
  total_money = initial_amount * apple_price - short_amount :=
by sorry

end xiaopang_mom_money_l2601_260119


namespace partner_investment_period_l2601_260117

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invests for 16 months, this theorem proves that p invests for 8 months. -/
theorem partner_investment_period (x : ℝ) (t : ℝ) : 
  (7 * x * t) / (5 * x * 16) = 7 / 10 → t = 8 := by
  sorry

end partner_investment_period_l2601_260117


namespace inscribed_triangle_angle_l2601_260103

/-- A triangle ABC inscribed in the parabola y = x^2 with specific properties -/
structure InscribedTriangle where
  /-- x-coordinate of point A -/
  a : ℝ
  /-- x-coordinate of point C -/
  c : ℝ
  /-- A and B have the same y-coordinate (AB parallel to x-axis) -/
  hParallel : a > 0
  /-- C is closer to x-axis than AB -/
  hCloser : 0 ≤ c ∧ c < a
  /-- Length of AB is 1 shorter than altitude CH -/
  hAltitude : a^2 - c^2 = 2*a + 1

/-- The angle ACB of the inscribed triangle is π/4 -/
theorem inscribed_triangle_angle (t : InscribedTriangle) : 
  Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by sorry

end inscribed_triangle_angle_l2601_260103


namespace calculate_expression_l2601_260168

theorem calculate_expression : ((-2 : ℤ)^2 : ℝ) - |(-5 : ℤ)| - Real.sqrt 144 = -13 := by
  sorry

end calculate_expression_l2601_260168


namespace seungjus_class_size_l2601_260132

theorem seungjus_class_size :
  ∃! n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 :=
sorry

end seungjus_class_size_l2601_260132


namespace inscribed_triangle_radius_l2601_260136

theorem inscribed_triangle_radius 
  (S : ℝ) 
  (α : ℝ) 
  (h1 : S > 0) 
  (h2 : 0 < α ∧ α < 2 * Real.pi) : 
  ∃ R : ℝ, R > 0 ∧ 
    R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
sorry

end inscribed_triangle_radius_l2601_260136


namespace cube_diagonal_length_l2601_260133

theorem cube_diagonal_length (s : ℝ) (h : s = 15) :
  let diagonal := Real.sqrt (3 * s^2)
  diagonal = 15 * Real.sqrt 3 :=
by sorry

end cube_diagonal_length_l2601_260133


namespace heptagon_interior_angle_sum_heptagon_interior_angle_sum_proof_l2601_260128

/-- The sum of the interior angles of a heptagon is 900 degrees. -/
theorem heptagon_interior_angle_sum : ℝ :=
  900

/-- A heptagon is a polygon with 7 sides. -/
def heptagon_sides : ℕ := 7

/-- The formula for the sum of interior angles of a polygon with n sides. -/
def polygon_interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem heptagon_interior_angle_sum_proof :
  polygon_interior_angle_sum heptagon_sides = heptagon_interior_angle_sum :=
by
  sorry

end heptagon_interior_angle_sum_heptagon_interior_angle_sum_proof_l2601_260128


namespace person_b_correct_probability_l2601_260118

theorem person_b_correct_probability 
  (prob_a_correct : ℝ) 
  (prob_b_correct_given_a_incorrect : ℝ) 
  (h1 : prob_a_correct = 0.4) 
  (h2 : prob_b_correct_given_a_incorrect = 0.5) : 
  (1 - prob_a_correct) * prob_b_correct_given_a_incorrect = 0.3 := by
  sorry

end person_b_correct_probability_l2601_260118


namespace sequence_existence_theorem_l2601_260184

theorem sequence_existence_theorem :
  (¬ ∃ (a : ℕ → ℕ), ∀ n, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) ∧
  (∃ (a : ℤ → ℝ), (∀ n, Irrational (a n)) ∧ (∀ n, (a (n - 1))^2 ≥ 2 * (a n) * (a (n - 2)))) :=
by sorry

end sequence_existence_theorem_l2601_260184


namespace triangle_inequality_l2601_260134

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > a^4 + b^4 + c^4 := by
  sorry

end triangle_inequality_l2601_260134


namespace alexis_alyssa_age_multiple_l2601_260129

theorem alexis_alyssa_age_multiple :
  ∀ (alexis_age alyssa_age : ℝ),
    alexis_age = 45 →
    alyssa_age = 45 →
    ∃ k : ℝ, alexis_age = k * alyssa_age - 162 →
    k = 4.6 :=
by
  sorry

end alexis_alyssa_age_multiple_l2601_260129


namespace max_ab_value_l2601_260152

/-- Given a > 0, b > 0, and f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1,
    the maximum value of ab is 9. -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → 
    (let g := fun x => 4 * x^3 - c * x^2 - 2 * d * x + 2
     ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), g x ≤ g 1 ∨ g x ≥ g 1) →
    a * b ≥ c * d) ∧ a * b = 9 := by
  sorry

end max_ab_value_l2601_260152


namespace equation_solutions_l2601_260148

theorem equation_solutions :
  (∀ x, x^2 - 7*x = 0 ↔ x = 0 ∨ x = 7) ∧
  (∀ x, 2*x^2 - 6*x + 1 = 0 ↔ x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) := by
  sorry

end equation_solutions_l2601_260148


namespace last_four_digits_of_7_to_5000_l2601_260191

theorem last_four_digits_of_7_to_5000 (h : 7^250 ≡ 1 [ZMOD 1250]) : 
  7^5000 ≡ 1 [ZMOD 1250] := by
  sorry

end last_four_digits_of_7_to_5000_l2601_260191


namespace biathlon_average_speed_l2601_260181

def cycling_speed : ℝ := 18
def running_speed : ℝ := 8

theorem biathlon_average_speed :
  let harmonic_mean := 2 / (1 / cycling_speed + 1 / running_speed)
  harmonic_mean = 144 / 13 := by
  sorry

end biathlon_average_speed_l2601_260181


namespace new_students_average_age_l2601_260127

/-- Calculates the average age of new students joining a class --/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_new_strength := original_strength + new_students
  let total_age_after := (original_strength + new_students) * new_average
  let total_age_before := original_strength * original_average
  let total_age_new_students := total_age_after - total_age_before
  (total_age_new_students / new_students : ℝ) = 32 := by
  sorry

end new_students_average_age_l2601_260127


namespace necessary_not_sufficient_l2601_260199

theorem necessary_not_sufficient :
  (∀ x : ℝ, -1 ≤ x ∧ x < 2 → -1 ≤ x ∧ x < 3) ∧
  ¬(∀ x : ℝ, -1 ≤ x ∧ x < 3 → -1 ≤ x ∧ x < 2) :=
by sorry

end necessary_not_sufficient_l2601_260199


namespace S_intersect_T_eq_T_l2601_260161

-- Define the sets S and T
def S : Set ℝ := {y | ∃ x, y = 3*x + 2}
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- Statement to prove
theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end S_intersect_T_eq_T_l2601_260161


namespace tiling_ways_eq_fib_l2601_260171

/-- The number of ways to tile a 2 × n strip with 1 × 2 or 2 × 1 bricks -/
def tiling_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => tiling_ways (k + 1) + tiling_ways k

/-- The Fibonacci sequence -/
def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => fib (k + 1) + fib k

theorem tiling_ways_eq_fib (n : ℕ) : tiling_ways n = fib n := by
  sorry

end tiling_ways_eq_fib_l2601_260171


namespace batsman_average_runs_l2601_260141

/-- The average runs scored by a batsman in a series of cricket matches. -/
def AverageRuns (first_10_avg : ℝ) (total_matches : ℕ) (overall_avg : ℝ) : Prop :=
  let first_10_total := first_10_avg * 10
  let total_runs := overall_avg * total_matches
  let next_10_total := total_runs - first_10_total
  let next_10_avg := next_10_total / 10
  next_10_avg = 30

/-- Theorem stating that given the conditions, the average runs scored in the next 10 matches is 30. -/
theorem batsman_average_runs : AverageRuns 40 20 35 := by
  sorry

end batsman_average_runs_l2601_260141


namespace factor_expression_l2601_260144

theorem factor_expression (a : ℝ) : 198 * a^2 + 36 * a + 54 = 18 * (11 * a^2 + 2 * a + 3) := by
  sorry

end factor_expression_l2601_260144


namespace area_is_60_l2601_260198

/-- Two perpendicular lines intersecting at point A(6,8) with y-intercepts P and Q -/
structure PerpendicularLines where
  A : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular : True  -- Represents that the lines are perpendicular
  intersect_at_A : True -- Represents that the lines intersect at A
  A_coords : A = (6, 8)
  P_is_y_intercept : P.1 = 0
  Q_is_y_intercept : Q.1 = 0
  sum_of_y_intercepts_zero : P.2 + Q.2 = 0

/-- The area of triangle APQ -/
def triangle_area (lines : PerpendicularLines) : ℝ := sorry

/-- Theorem stating that the area of triangle APQ is 60 -/
theorem area_is_60 (lines : PerpendicularLines) : triangle_area lines = 60 := by
  sorry

end area_is_60_l2601_260198


namespace james_tin_collection_l2601_260159

/-- The number of tins James collects in a week -/
def total_tins : ℕ := 500

/-- The number of tins James collects on the first day -/
def first_day_tins : ℕ := 50

/-- The number of tins James collects on the second day -/
def second_day_tins : ℕ := 3 * first_day_tins

/-- The number of tins James collects on each of the remaining days (4th to 7th) -/
def remaining_days_tins : ℕ := 50

/-- The total number of tins James collects on the remaining days (4th to 7th) -/
def total_remaining_days_tins : ℕ := 4 * remaining_days_tins

/-- The number of tins James collects on the third day -/
def third_day_tins : ℕ := total_tins - first_day_tins - second_day_tins - total_remaining_days_tins

theorem james_tin_collection :
  second_day_tins - third_day_tins = 50 :=
sorry

end james_tin_collection_l2601_260159


namespace remainder_of_1394_divided_by_2535_l2601_260195

theorem remainder_of_1394_divided_by_2535 : Int.mod 1394 2535 = 1394 := by
  sorry

end remainder_of_1394_divided_by_2535_l2601_260195


namespace second_class_average_l2601_260135

/-- Proves that the average mark of the second class is 69.83 given the conditions of the problem -/
theorem second_class_average (students_class1 : ℕ) (students_class2 : ℕ) 
  (avg_class1 : ℝ) (total_avg : ℝ) :
  students_class1 = 39 →
  students_class2 = 35 →
  avg_class1 = 45 →
  total_avg = 56.75 →
  let total_students := students_class1 + students_class2
  let avg_class2 := (total_avg * total_students - avg_class1 * students_class1) / students_class2
  avg_class2 = 69.83 := by
sorry

end second_class_average_l2601_260135


namespace triangle_angle_measurement_l2601_260100

theorem triangle_angle_measurement (A B C : ℝ) : 
  A = 70 ∧ 
  B = 2 * C + 30 ∧ 
  A + B + C = 180 →
  C = 80 / 3 := by
sorry

end triangle_angle_measurement_l2601_260100


namespace min_socks_for_given_problem_l2601_260131

/-- The minimum number of socks to pull out to guarantee at least one of each color -/
def min_socks_to_pull (red blue green khaki : ℕ) : ℕ :=
  (red + blue + green + khaki) - min red (min blue (min green khaki)) + 1

/-- Theorem stating the minimum number of socks to pull out for the given problem -/
theorem min_socks_for_given_problem :
  min_socks_to_pull 10 20 30 40 = 91 := by
  sorry

end min_socks_for_given_problem_l2601_260131


namespace ball_distribution_ratio_l2601_260107

theorem ball_distribution_ratio : 
  let total_balls : ℕ := 20
  let num_bins : ℕ := 5
  let config_A : List ℕ := [2, 6, 4, 4, 4]
  let config_B : List ℕ := [4, 4, 4, 4, 4]
  
  let prob_A := (Nat.choose num_bins 1) * (Nat.choose (num_bins - 1) 1) * 
                (Nat.choose total_balls 2) * (Nat.choose (total_balls - 2) 6) * 
                (Nat.choose (total_balls - 2 - 6) 4) * (Nat.choose (total_balls - 2 - 6 - 4) 4) * 
                (Nat.choose (total_balls - 2 - 6 - 4 - 4) 4)
  
  let prob_B := (Nat.choose total_balls 4) * (Nat.choose (total_balls - 4) 4) * 
                (Nat.choose (total_balls - 4 - 4) 4) * (Nat.choose (total_balls - 4 - 4 - 4) 4) * 
                (Nat.choose (total_balls - 4 - 4 - 4 - 4) 4)
  
  prob_A / prob_B = 10 := by
  sorry

#check ball_distribution_ratio

end ball_distribution_ratio_l2601_260107


namespace i_power_sum_l2601_260194

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the property that powers of i repeat every 4 powers
axiom i_power_cycle (n : ℤ) : i^n = i^(n % 4)

-- State the theorem
theorem i_power_sum : i^17 + i^2023 = 0 := by
  sorry

end i_power_sum_l2601_260194


namespace inequality_holds_for_even_positive_integers_l2601_260169

theorem inequality_holds_for_even_positive_integers (n : ℕ) (hn : Even n) (hn_pos : 0 < n) :
  ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2 := by
  sorry

end inequality_holds_for_even_positive_integers_l2601_260169


namespace boys_camp_total_l2601_260147

theorem boys_camp_total (total_boys : ℕ) : 
  (total_boys : ℝ) * 0.2 * 0.7 = 21 → total_boys = 150 := by
  sorry

end boys_camp_total_l2601_260147


namespace sqrt_twelve_over_sqrt_two_equals_sqrt_six_l2601_260167

theorem sqrt_twelve_over_sqrt_two_equals_sqrt_six : 
  (Real.sqrt 12) / (Real.sqrt 2) = Real.sqrt 6 := by
  sorry

end sqrt_twelve_over_sqrt_two_equals_sqrt_six_l2601_260167


namespace sqrt_floor_problem_l2601_260143

theorem sqrt_floor_problem (a b c : ℝ) : 
  (abs a = 4) → 
  (b^2 = 9) → 
  (c^3 = -8) → 
  (a > c) → 
  (c > b) → 
  Int.floor (Real.sqrt (a - b - 2*c)) = 3 := by
  sorry

end sqrt_floor_problem_l2601_260143


namespace cosine_equality_in_range_l2601_260185

theorem cosine_equality_in_range (n : ℤ) :
  100 ≤ n ∧ n ≤ 300 ∧ Real.cos (n * π / 180) = Real.cos (140 * π / 180) → n = 220 := by
  sorry

end cosine_equality_in_range_l2601_260185


namespace extreme_values_of_f_l2601_260196

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem extreme_values_of_f :
  ∃ (a b : ℝ), (∀ x : ℝ, f x ≤ f a ∨ f x ≥ f b) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≤ f c ∨ f x ≥ f c) → c = a ∨ c = b) :=
sorry

end extreme_values_of_f_l2601_260196


namespace uncovered_cells_bound_l2601_260102

/-- Represents a rectangular board with dominoes -/
structure Board where
  m : ℕ  -- width of the board
  n : ℕ  -- height of the board
  uncovered : ℕ  -- number of uncovered cells

/-- Theorem stating that the number of uncovered cells is less than both mn/4 and mn/5 -/
theorem uncovered_cells_bound (b : Board) : 
  b.uncovered < min (b.m * b.n / 4) (b.m * b.n / 5) := by
  sorry

#check uncovered_cells_bound

end uncovered_cells_bound_l2601_260102


namespace maze_paths_count_l2601_260114

/-- Represents a maze with specific branching structure -/
structure Maze where
  initial_branches : Nat
  subsequent_branches : Nat
  final_paths : Nat

/-- Calculates the number of unique paths through the maze -/
def count_paths (m : Maze) : Nat :=
  m.initial_branches * m.subsequent_branches.pow m.final_paths

/-- Theorem stating that a maze with given properties has 16 unique paths -/
theorem maze_paths_count :
  ∀ (m : Maze), m.initial_branches = 2 ∧ m.subsequent_branches = 2 ∧ m.final_paths = 3 →
  count_paths m = 16 := by
  sorry

#eval count_paths ⟨2, 2, 3⟩  -- Should output 16

end maze_paths_count_l2601_260114


namespace smallest_prime_factor_in_C_l2601_260120

def C : Finset Nat := {51, 53, 54, 56, 57}

def has_smallest_prime_factor (n : Nat) (s : Finset Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, (Nat.minFac n ≤ Nat.minFac m)

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 54 C := by
  sorry

end smallest_prime_factor_in_C_l2601_260120


namespace exam_score_theorem_l2601_260126

theorem exam_score_theorem (total_students : ℕ) 
                            (assigned_day_percentage : ℚ) 
                            (makeup_day_percentage : ℚ) 
                            (makeup_day_average : ℚ) 
                            (class_average : ℚ) :
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_day_percentage = 30 / 100 →
  makeup_day_average = 80 / 100 →
  class_average = 66 / 100 →
  ∃ (assigned_day_average : ℚ),
    assigned_day_average = 60 / 100 ∧
    class_average * total_students = 
      (assigned_day_percentage * total_students * assigned_day_average) +
      (makeup_day_percentage * total_students * makeup_day_average) :=
by sorry

end exam_score_theorem_l2601_260126


namespace forgot_capsules_days_l2601_260108

/-- The number of days in July -/
def july_days : ℕ := 31

/-- The number of days Adam took his capsules in July -/
def days_took_capsules : ℕ := 27

/-- The number of days Adam forgot to take his capsules in July -/
def days_forgot_capsules : ℕ := july_days - days_took_capsules

theorem forgot_capsules_days : days_forgot_capsules = 4 := by
  sorry

end forgot_capsules_days_l2601_260108


namespace fish_population_estimate_l2601_260178

theorem fish_population_estimate (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) 
  (h1 : initial_marked = 100)
  (h2 : second_catch = 200)
  (h3 : marked_in_second = 25) :
  (initial_marked * second_catch) / marked_in_second = 800 :=
by sorry

end fish_population_estimate_l2601_260178


namespace matrix_transformation_l2601_260137

theorem matrix_transformation (P Q : Matrix (Fin 3) (Fin 3) ℝ) : 
  P = !![3, 0, 0; 0, 0, 1; 0, 1, 0] → 
  (∀ a b c d e f g h i : ℝ, 
    Q = !![a, b, c; d, e, f; g, h, i] → 
    P * Q = !![3*a, 3*b, 3*c; g, h, i; d, e, f]) :=
by sorry

end matrix_transformation_l2601_260137


namespace souvenir_cost_in_usd_l2601_260172

/-- Calculates the cost in USD given the cost in yen and the exchange rate -/
def cost_in_usd (cost_yen : ℚ) (exchange_rate : ℚ) : ℚ :=
  cost_yen / exchange_rate

theorem souvenir_cost_in_usd :
  let cost_yen : ℚ := 500
  let exchange_rate : ℚ := 120
  cost_in_usd cost_yen exchange_rate = 25 / 6 := by sorry

end souvenir_cost_in_usd_l2601_260172


namespace sin_2theta_value_l2601_260149

theorem sin_2theta_value (θ : Real) (h : Real.sin (θ + π/4) = 1/3) : 
  Real.sin (2*θ) = -7/9 := by
  sorry

end sin_2theta_value_l2601_260149


namespace li_family_cinema_cost_l2601_260160

def adult_ticket_price : ℝ := 10
def child_discount : ℝ := 0.4
def senior_discount : ℝ := 0.3
def handling_fee : ℝ := 5
def num_adults : ℕ := 2
def num_children : ℕ := 1
def num_seniors : ℕ := 1

def total_cost : ℝ :=
  (num_adults * adult_ticket_price) +
  (num_children * adult_ticket_price * (1 - child_discount)) +
  (num_seniors * adult_ticket_price * (1 - senior_discount)) +
  handling_fee

theorem li_family_cinema_cost : total_cost = 38 := by
  sorry

end li_family_cinema_cost_l2601_260160


namespace three_numbers_sum_l2601_260177

theorem three_numbers_sum : ∀ (a b c : ℝ),
  a ≤ b ∧ b ≤ c →  -- Arrange numbers in ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 66 := by
sorry

end three_numbers_sum_l2601_260177


namespace area_of_trapezoid_TUVW_l2601_260121

/-- Represents a triangle in the problem -/
structure Triangle where
  isIsosceles : Bool
  area : ℝ

/-- Represents the large triangle XYZ -/
def XYZ : Triangle where
  isIsosceles := true
  area := 135

/-- Represents a small triangle -/
def SmallTriangle : Triangle where
  isIsosceles := true
  area := 3

/-- The number of small triangles in XYZ -/
def numSmallTriangles : ℕ := 9

/-- The number of small triangles in trapezoid TUVW -/
def numSmallTrianglesInTUVW : ℕ := 4

/-- The area of trapezoid TUVW -/
def areaTUVW : ℝ := numSmallTrianglesInTUVW * SmallTriangle.area

theorem area_of_trapezoid_TUVW : areaTUVW = 123 := by
  sorry

end area_of_trapezoid_TUVW_l2601_260121


namespace volume_maximized_when_perpendicular_l2601_260173

/-- A tetrahedron with edge lengths u, v, and w. -/
structure Tetrahedron (u v w : ℝ) where
  edge_u : ℝ := u
  edge_v : ℝ := v
  edge_w : ℝ := w

/-- The volume of a tetrahedron. -/
noncomputable def volume (t : Tetrahedron u v w) : ℝ :=
  sorry

/-- Mutually perpendicular edges of a tetrahedron. -/
def mutually_perpendicular (t : Tetrahedron u v w) : Prop :=
  sorry

/-- Theorem: The volume of a tetrahedron is maximized when its edges are mutually perpendicular. -/
theorem volume_maximized_when_perpendicular (u v w : ℝ) (t : Tetrahedron u v w) :
  mutually_perpendicular t ↔ ∀ (t' : Tetrahedron u v w), volume t ≥ volume t' :=
sorry

end volume_maximized_when_perpendicular_l2601_260173


namespace smallest_divisor_of_28_l2601_260162

theorem smallest_divisor_of_28 : ∀ d : ℕ, d > 0 → d ∣ 28 → d ≥ 1 :=
by
  sorry

end smallest_divisor_of_28_l2601_260162


namespace parabola_vertex_y_coordinate_l2601_260140

/-- The y-coordinate of the vertex of the parabola y = 3x^2 - 6x + 4 is 1 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 4
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = 1 :=
by sorry

end parabola_vertex_y_coordinate_l2601_260140


namespace triangle_centroid_distance_sum_l2601_260116

/-- Given a triangle ABC with centroid G, prove that if GA^2 + GB^2 + GC^2 = 58, 
    then AB^2 + AC^2 + BC^2 = 174. -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →  -- G is the centroid
  (dist G A)^2 + (dist G B)^2 + (dist G C)^2 = 58 →       -- Given condition
  (dist A B)^2 + (dist A C)^2 + (dist B C)^2 = 174 :=     -- Conclusion to prove
by
  sorry

#check triangle_centroid_distance_sum

end triangle_centroid_distance_sum_l2601_260116


namespace three_numbers_problem_l2601_260153

theorem three_numbers_problem (a b c : ℝ) :
  ((a + 1) * (b + 1) * (c + 1) = a * b * c + 1) →
  ((a + 2) * (b + 2) * (c + 2) = a * b * c + 2) →
  (a = -1 ∧ b = -1 ∧ c = -1) :=
by sorry

end three_numbers_problem_l2601_260153


namespace quadratic_always_positive_l2601_260183

theorem quadratic_always_positive : ∀ x : ℝ, 15 * x^2 - 8 * x + 3 > 0 := by
  sorry

end quadratic_always_positive_l2601_260183


namespace second_run_time_l2601_260111

/-- Represents the time in seconds for various parts of the obstacle course challenge -/
structure ObstacleCourseTime where
  totalSecondRun : ℕ
  doorOpenTime : ℕ

/-- Calculates the time for the second run without backpack -/
def secondRunWithoutBackpack (t : ObstacleCourseTime) : ℕ :=
  t.totalSecondRun - t.doorOpenTime

/-- Theorem stating that for the given times, the second run without backpack takes 801 seconds -/
theorem second_run_time (t : ObstacleCourseTime) 
    (h1 : t.totalSecondRun = 874)
    (h2 : t.doorOpenTime = 73) : 
  secondRunWithoutBackpack t = 801 := by
  sorry

end second_run_time_l2601_260111


namespace hyperbola_condition_l2601_260122

theorem hyperbola_condition (m : ℝ) :
  m > 0 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), x^2 / (2 + m) - y^2 / (1 + m) = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end hyperbola_condition_l2601_260122


namespace problem_statement_l2601_260158

theorem problem_statement : 
  let p := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2
  let q := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0
  (¬p) ∧ q := by sorry

end problem_statement_l2601_260158


namespace right_triangle_third_side_l2601_260165

theorem right_triangle_third_side (a b x : ℝ) :
  (a - 3)^2 + |b - 4| = 0 →
  (x^2 = a^2 + b^2 ∨ x^2 + a^2 = b^2 ∨ x^2 + b^2 = a^2) →
  x = 5 ∨ x = Real.sqrt 7 := by
  sorry

end right_triangle_third_side_l2601_260165


namespace regular_20gon_symmetry_sum_l2601_260166

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_20gon_symmetry_sum :
  ∀ (p : RegularPolygon 20),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 38 := by sorry

end regular_20gon_symmetry_sum_l2601_260166


namespace pascal_interior_sum_8_9_l2601_260123

/-- Sum of interior numbers in row n of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The sum of interior numbers of the 8th and 9th rows of Pascal's Triangle is 380 -/
theorem pascal_interior_sum_8_9 : interior_sum 8 + interior_sum 9 = 380 := by
  sorry

end pascal_interior_sum_8_9_l2601_260123


namespace water_jar_problem_l2601_260180

theorem water_jar_problem (c1 c2 c3 : ℝ) (h1 : c1 > 0) (h2 : c2 > 0) (h3 : c3 > 0) 
  (h4 : c1 < c2) (h5 : c2 < c3) 
  (h6 : c1 / 6 = c2 / 5) (h7 : c2 / 5 = c3 / 7) : 
  (c1 / 6 + c2 / 5) / c3 = 2 / 7 := by
sorry

end water_jar_problem_l2601_260180


namespace serezha_puts_more_berries_l2601_260125

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  serezha_rate : ℕ → ℕ  -- Function representing Serezha's picking pattern
  dima_rate : ℕ → ℕ     -- Function representing Dima's picking pattern
  serezha_speed : ℕ
  dima_speed : ℕ

/-- The specific berry picking scenario from the problem -/
def berry_scenario : BerryPicking :=
  { total_berries := 450
  , serezha_rate := λ n => n / 2  -- 1 out of every 2
  , dima_rate := λ n => 2 * n / 3 -- 2 out of every 3
  , serezha_speed := 2
  , dima_speed := 1 }

/-- Theorem stating the difference in berries put in basket -/
theorem serezha_puts_more_berries (bp : BerryPicking) (h : bp = berry_scenario) :
  ∃ (s d : ℕ), s = bp.serezha_rate (bp.serezha_speed * bp.total_berries / (bp.serezha_speed + bp.dima_speed)) ∧
                d = bp.dima_rate (bp.dima_speed * bp.total_berries / (bp.serezha_speed + bp.dima_speed)) ∧
                s - d = 50 := by
  sorry


end serezha_puts_more_berries_l2601_260125


namespace contrapositive_evenness_l2601_260146

theorem contrapositive_evenness (a b : ℤ) : 
  (Odd (a + b) → Odd a ∨ Odd b) = False :=
sorry

end contrapositive_evenness_l2601_260146


namespace circle_C_and_point_M_l2601_260109

/-- Circle C passing through points A and B, bisected by a line, with point M satisfying certain conditions -/
structure CircleC where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- Point A on the circle -/
  A : ℝ × ℝ
  /-- Point B on the circle -/
  B : ℝ × ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point M on the circle -/
  M : ℝ × ℝ
  /-- Circle passes through A -/
  passes_through_A : (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2
  /-- Circle passes through B -/
  passes_through_B : (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2
  /-- Line x-3y-4=0 bisects the circle -/
  bisected_by_line : center.1 - 3 * center.2 - 4 = 0
  /-- |MP|/|MQ| = 2 -/
  distance_ratio : ((M.1 - P.1)^2 + (M.2 - P.2)^2) = 4 * ((M.1 - Q.1)^2 + (M.2 - Q.2)^2)

/-- Theorem about the equation of circle C and coordinates of point M -/
theorem circle_C_and_point_M (c : CircleC)
  (h_A : c.A = (0, 2))
  (h_B : c.B = (6, 4))
  (h_P : c.P = (-6, 0))
  (h_Q : c.Q = (6, 0)) :
  (c.center = (4, 0) ∧ c.radius^2 = 20) ∧
  (c.M = (10/3, 4*Real.sqrt 11/3) ∨ c.M = (10/3, -4*Real.sqrt 11/3)) := by
  sorry

end circle_C_and_point_M_l2601_260109


namespace calculator_correction_l2601_260101

theorem calculator_correction : (0.024 * 3.08) / 0.4 = 0.1848 := by
  sorry

end calculator_correction_l2601_260101


namespace distribute_subtraction_l2601_260163

theorem distribute_subtraction (a b c : ℝ) : 5*a - (b + 2*c) = 5*a - b - 2*c := by
  sorry

end distribute_subtraction_l2601_260163


namespace equation_solution_l2601_260124

theorem equation_solution : ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 := by
  sorry

end equation_solution_l2601_260124


namespace xy_and_x2y_2xy2_values_l2601_260197

theorem xy_and_x2y_2xy2_values (x y : ℝ) 
  (h1 : x - 2*y = 3) 
  (h2 : x^2 - 2*x*y + 4*y^2 = 11) : 
  x * y = 1 ∧ x^2 * y - 2 * x * y^2 = 3 := by
  sorry

end xy_and_x2y_2xy2_values_l2601_260197


namespace factorial_division_l2601_260139

theorem factorial_division (h : Nat.factorial 7 = 5040) :
  Nat.factorial 7 / Nat.factorial 4 = 210 := by
  sorry

end factorial_division_l2601_260139


namespace power_fraction_simplification_l2601_260174

theorem power_fraction_simplification :
  (16 : ℕ) ^ 24 / (64 : ℕ) ^ 8 = (16 : ℕ) ^ 12 := by sorry

end power_fraction_simplification_l2601_260174


namespace pta_funds_remaining_l2601_260156

def initial_amount : ℚ := 600

def amount_after_supplies (initial : ℚ) : ℚ :=
  initial - (2 / 5) * initial

def amount_after_food (after_supplies : ℚ) : ℚ :=
  after_supplies - (30 / 100) * after_supplies

def final_amount (after_food : ℚ) : ℚ :=
  after_food - (1 / 3) * after_food

theorem pta_funds_remaining :
  final_amount (amount_after_food (amount_after_supplies initial_amount)) = 168 := by
  sorry

end pta_funds_remaining_l2601_260156


namespace scavenger_hunt_items_l2601_260192

theorem scavenger_hunt_items (tanya samantha lewis : ℕ) : 
  tanya = 4 ∧ 
  samantha = 4 * tanya ∧ 
  lewis = samantha + 4 → 
  lewis = 20 := by
sorry

end scavenger_hunt_items_l2601_260192


namespace ellipse_axis_endpoint_distance_l2601_260182

/-- Given an ellipse with equation 9(x-1)^2 + y^2 = 36, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√10 -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), 9 * (x - 1)^2 + y^2 = 36 → 
      ((x = A.1 ∧ y = A.2) ∨ (x = -A.1 ∧ y = -A.2)) ∨ 
      ((x = B.1 ∧ y = B.2) ∨ (x = -B.1 ∧ y = -B.2))) →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 40 :=
by sorry

end ellipse_axis_endpoint_distance_l2601_260182


namespace overall_discount_rate_l2601_260115

def bag_marked : ℕ := 200
def shirt_marked : ℕ := 80
def shoes_marked : ℕ := 150
def hat_marked : ℕ := 50
def jacket_marked : ℕ := 220

def bag_sold : ℕ := 120
def shirt_sold : ℕ := 60
def shoes_sold : ℕ := 105
def hat_sold : ℕ := 40
def jacket_sold : ℕ := 165

def total_marked : ℕ := bag_marked + shirt_marked + shoes_marked + hat_marked + jacket_marked
def total_sold : ℕ := bag_sold + shirt_sold + shoes_sold + hat_sold + jacket_sold

theorem overall_discount_rate :
  (1 - (total_sold : ℚ) / total_marked) * 100 = 30 := by sorry

end overall_discount_rate_l2601_260115


namespace ironman_age_is_48_l2601_260112

-- Define the ages as natural numbers
def thor_age : ℕ := 1456
def captain_america_age : ℕ := thor_age / 13
def peter_parker_age : ℕ := captain_america_age / 7
def doctor_strange_age : ℕ := peter_parker_age * 4
def ironman_age : ℕ := peter_parker_age + 32

-- State the theorem
theorem ironman_age_is_48 :
  (thor_age = 13 * captain_america_age) ∧
  (captain_america_age = 7 * peter_parker_age) ∧
  (4 * peter_parker_age = doctor_strange_age) ∧
  (ironman_age = peter_parker_age + 32) ∧
  (thor_age = 1456) →
  ironman_age = 48 := by
  sorry

end ironman_age_is_48_l2601_260112


namespace joaos_chocolates_l2601_260142

theorem joaos_chocolates :
  ∃! n : ℕ, 30 < n ∧ n < 100 ∧ n % 7 = 1 ∧ n % 10 = 2 ∧ n = 92 := by
  sorry

end joaos_chocolates_l2601_260142


namespace report_card_recess_num_of_ds_l2601_260154

/-- Calculates the number of Ds on report cards given the recess rules and grades --/
theorem report_card_recess (normal_recess : ℕ) (a_bonus : ℕ) (b_bonus : ℕ) (d_penalty : ℕ)
  (num_a : ℕ) (num_b : ℕ) (num_c : ℕ) (total_recess : ℕ) : ℕ :=
  let extra_time := num_a * a_bonus + num_b * b_bonus
  let expected_time := normal_recess + extra_time
  let reduced_time := expected_time - total_recess
  reduced_time / d_penalty

/-- Proves that there are 5 Ds on the report cards --/
theorem num_of_ds : report_card_recess 20 2 1 1 10 12 14 47 = 5 := by
  sorry

end report_card_recess_num_of_ds_l2601_260154


namespace parabola_comparison_l2601_260170

theorem parabola_comparison :
  ∀ x : ℝ, -x^2 + 2*x + 3 > x^2 - 2*x + 3 :=
by sorry

end parabola_comparison_l2601_260170


namespace weight_of_replaced_person_l2601_260113

/-- Proves that given 8 persons, if replacing one person with a new person weighing 93 kg
    increases the average weight by 3.5 kg, then the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 3.5)
  (h3 : new_person_weight = 93)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end weight_of_replaced_person_l2601_260113


namespace march_largest_drop_l2601_260104

/-- Represents the months of interest --/
inductive Month
  | january
  | february
  | march
  | april

/-- The price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january => -1.25
  | Month.february => 0.75
  | Month.march => -3.00
  | Month.april => 0.25

/-- A month has the largest price drop if its price change is negative and smaller than or equal to all other negative price changes --/
def has_largest_price_drop (m : Month) : Prop :=
  price_change m < 0 ∧
  ∀ n : Month, price_change n < 0 → price_change m ≤ price_change n

theorem march_largest_drop :
  has_largest_price_drop Month.march :=
sorry


end march_largest_drop_l2601_260104


namespace circle_radius_l2601_260155

theorem circle_radius (x y : ℝ) : 
  (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 1 := by
sorry

end circle_radius_l2601_260155


namespace volunteer_distribution_l2601_260105

theorem volunteer_distribution (n : ℕ) (h : n = 5) :
  (n.choose 1) * ((n - 1).choose 2 / 2) = 15 := by
  sorry

end volunteer_distribution_l2601_260105


namespace unique_be_length_l2601_260188

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a unit square ABCD -/
def UnitSquare : (Point × Point × Point × Point) :=
  (⟨0, 0⟩, ⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 1⟩)

/-- Definition of perpendicularity between two line segments -/
def Perpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

/-- Theorem: In a unit square ABCD, with points E on BC, F on CD, and G on DA,
    if AE ⊥ EF, EF ⊥ FG, and GA = 404/1331, then BE = 9/11 -/
theorem unique_be_length (A B C D E F G : Point)
  (square : (A, B, C, D) = UnitSquare)
  (e_on_bc : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = ⟨1, t⟩)
  (f_on_cd : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = ⟨1 - t, 1⟩)
  (g_on_da : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = ⟨0, 1 - t⟩)
  (ae_perp_ef : Perpendicular A E E F)
  (ef_perp_fg : Perpendicular E F F G)
  (ga_length : (G.x - A.x)^2 + (G.y - A.y)^2 = (404/1331)^2) :
  (E.x - B.x)^2 + (E.y - B.y)^2 = (9/11)^2 := by
  sorry

end unique_be_length_l2601_260188
