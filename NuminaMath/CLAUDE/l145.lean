import Mathlib

namespace min_A_over_C_is_zero_l145_14563

theorem min_A_over_C_is_zero (A C x : ℝ) (hA : A > 0) (hC : C > 0) (hx : x > 0)
  (hAx : x^2 + 1/x^2 = A) (hCx : x + 1/x = C) :
  ∀ ε > 0, ∃ A' C' x', A' > 0 ∧ C' > 0 ∧ x' > 0 ∧
    x'^2 + 1/x'^2 = A' ∧ x' + 1/x' = C' ∧ A' / C' < ε :=
sorry

end min_A_over_C_is_zero_l145_14563


namespace trigonometric_expression_value_l145_14577

theorem trigonometric_expression_value :
  Real.sin (315 * π / 180) * Real.sin (-1260 * π / 180) + 
  Real.cos (390 * π / 180) * Real.sin (-1020 * π / 180) = 3/4 := by
  sorry

end trigonometric_expression_value_l145_14577


namespace slope_angle_of_line_l145_14508

/-- The slope angle of the line x + √3y - 3 = 0 is 5π/6 -/
theorem slope_angle_of_line (x y : ℝ) : 
  x + Real.sqrt 3 * y - 3 = 0 → 
  ∃ α : ℝ, α = 5 * Real.pi / 6 ∧ 
    (Real.tan α = -(1 / Real.sqrt 3) ∨ Real.tan α = -(Real.sqrt 3 / 3)) :=
by sorry

end slope_angle_of_line_l145_14508


namespace function_properties_l145_14528

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_sym1 : ∀ x, f ω φ x = f ω φ ((2 * π) / 3 - x))
  (h_sym2 : ∀ x, f ω φ x = -f ω φ (π - x))
  (h_period : ∃ T > π / 2, ∀ x, f ω φ (x + T) = f ω φ x) :
  (∀ x, f ω φ (x + (2 * π) / 3) = f ω φ x) ∧
  (∀ x, f ω φ x = f ω φ (-x)) :=
sorry

end function_properties_l145_14528


namespace method_one_saves_more_money_l145_14517

/-- Represents the discount methods available at the store -/
inductive DiscountMethod
  | BuyRacketGetShuttlecock
  | PayPercentage

/-- Calculates the cost of purchase using the given discount method -/
def calculateCost (racketPrice shuttlecockPrice : ℕ) (racketCount shuttlecockCount : ℕ) (method : DiscountMethod) : ℚ :=
  match method with
  | DiscountMethod.BuyRacketGetShuttlecock =>
      (racketCount * racketPrice + (shuttlecockCount - racketCount) * shuttlecockPrice : ℚ)
  | DiscountMethod.PayPercentage =>
      ((racketCount * racketPrice + shuttlecockCount * shuttlecockPrice) * 92 / 100 : ℚ)

/-- Theorem stating that discount method ① saves more money than method ② -/
theorem method_one_saves_more_money (racketPrice shuttlecockPrice : ℕ) (racketCount shuttlecockCount : ℕ)
    (h1 : racketPrice = 20)
    (h2 : shuttlecockPrice = 5)
    (h3 : racketCount = 4)
    (h4 : shuttlecockCount = 30) :
    calculateCost racketPrice shuttlecockPrice racketCount shuttlecockCount DiscountMethod.BuyRacketGetShuttlecock <
    calculateCost racketPrice shuttlecockPrice racketCount shuttlecockCount DiscountMethod.PayPercentage :=
  sorry

end method_one_saves_more_money_l145_14517


namespace min_value_reciprocal_sum_l145_14546

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 4 :=
by sorry

end min_value_reciprocal_sum_l145_14546


namespace product_inequality_l145_14555

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end product_inequality_l145_14555


namespace boy_age_problem_l145_14585

theorem boy_age_problem (present_age : ℕ) (h : present_age = 16) : 
  ∃ (years_ago : ℕ), 
    (present_age + 4 = 2 * (present_age - years_ago)) ∧ 
    (present_age - years_ago = (present_age + 4) / 2) ∧
    years_ago = 6 := by
  sorry

end boy_age_problem_l145_14585


namespace probability_is_three_fourths_l145_14589

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a point (x, y) satisfies x + 2y < 4 when randomly and uniformly chosen from the given square --/
def probabilityLessThan4 (s : Square) : ℝ :=
  sorry

/-- The square with vertices (0, 0), (0, 3), (3, 3), and (3, 0) --/
def givenSquare : Square :=
  { bottomLeft := (0, 0), topRight := (3, 3) }

theorem probability_is_three_fourths :
  probabilityLessThan4 givenSquare = 3/4 := by
  sorry

end probability_is_three_fourths_l145_14589


namespace beverage_probabilities_l145_14545

/-- The probability of a single bottle of X beverage being qualified -/
def p_qualified : ℝ := 0.8

/-- The number of people drinking the beverage -/
def num_people : ℕ := 3

/-- The number of bottles each person drinks -/
def bottles_per_person : ℕ := 2

/-- The probability that a person drinks two qualified bottles -/
def p_two_qualified : ℝ := p_qualified ^ bottles_per_person

/-- The probability that exactly two out of three people drink two qualified bottles -/
def p_two_out_of_three : ℝ := 
  (num_people.choose 2 : ℝ) * p_two_qualified ^ 2 * (1 - p_two_qualified) ^ (num_people - 2)

theorem beverage_probabilities :
  p_two_qualified = 0.64 ∧ p_two_out_of_three = 0.44 := by sorry

end beverage_probabilities_l145_14545


namespace sugar_purchase_proof_l145_14533

/-- The number of pounds of sugar bought by the housewife -/
def sugar_pounds : ℕ := 24

/-- The price per pound of sugar in cents -/
def price_per_pound : ℕ := 9

/-- The total cost of the sugar purchase in cents -/
def total_cost : ℕ := 216

/-- Proves that the number of pounds of sugar bought is correct given the conditions -/
theorem sugar_purchase_proof :
  (sugar_pounds * price_per_pound = total_cost) ∧
  (sugar_pounds + 3) * (price_per_pound - 1) = total_cost :=
by sorry

#check sugar_purchase_proof

end sugar_purchase_proof_l145_14533


namespace perpendicular_bisector_value_l145_14531

/-- The perpendicular bisector of a line segment from (x₁, y₁) to (x₂, y₂) is defined as
    the line that passes through the midpoint of the segment and is perpendicular to it. --/
def is_perpendicular_bisector (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  -- The line ax + by + c = 0 passes through the midpoint
  a * midpoint_x + b * midpoint_y + c = 0 ∧
  -- The line is perpendicular to the segment
  a * (x₂ - x₁) + b * (y₂ - y₁) = 0

/-- Given that the line x + y = b is the perpendicular bisector of the line segment 
    from (0, 5) to (8, 10), prove that b = 11.5 --/
theorem perpendicular_bisector_value : 
  is_perpendicular_bisector 1 1 (-b) 0 5 8 10 → b = 11.5 := by
  sorry

end perpendicular_bisector_value_l145_14531


namespace walk_legs_count_l145_14571

/-- The number of legs of a human -/
def human_legs : ℕ := 2

/-- The number of legs of a dog -/
def dog_legs : ℕ := 4

/-- The number of humans on the walk -/
def num_humans : ℕ := 2

/-- The number of dogs on the walk -/
def num_dogs : ℕ := 2

/-- The total number of legs of all organisms on the walk -/
def total_legs : ℕ := human_legs * num_humans + dog_legs * num_dogs

theorem walk_legs_count : total_legs = 12 := by
  sorry

end walk_legs_count_l145_14571


namespace nine_balls_distribution_l145_14524

/-- The number of ways to distribute n identical objects into 3 distinct boxes,
    where box i must contain at least i objects (for i = 1, 2, 3) -/
def distribute_balls (n : ℕ) : ℕ := Nat.choose (n - 1 - 2 - 3 + 3 - 1) 3

/-- Theorem stating that there are 10 ways to distribute 9 balls into 3 boxes
    with the given constraints -/
theorem nine_balls_distribution : distribute_balls 9 = 10 := by
  sorry

end nine_balls_distribution_l145_14524


namespace square_perimeter_relation_l145_14503

/-- Given a square C with perimeter 40 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (40√3)/3 cm. -/
theorem square_perimeter_relation (C D : Real) : 
  (C = 10) →  -- Side length of square C (derived from perimeter 40)
  (D^2 = (C^2) / 3) →  -- Area of D is one-third of area of C
  (4 * D = (40 * Real.sqrt 3) / 3) :=  -- Perimeter of D
by sorry

end square_perimeter_relation_l145_14503


namespace trigonometric_simplification_l145_14538

theorem trigonometric_simplification :
  let tan_sum := Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
                 Real.tan (40 * π / 180) + Real.tan (60 * π / 180)
  tan_sum / Real.sin (80 * π / 180) = 
    2 * (Real.cos (40 * π / 180) / (Real.sqrt 3 * Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) + 
         2 / Real.cos (40 * π / 180)) := by
  sorry

end trigonometric_simplification_l145_14538


namespace green_shirt_pairs_l145_14553

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 140 →
  red_students = 60 →
  green_students = 80 →
  total_pairs = 70 →
  red_red_pairs = 10 →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 20 ∧ 
    green_green_pairs + red_red_pairs + (total_pairs - green_green_pairs - red_red_pairs) = total_pairs :=
by sorry

end green_shirt_pairs_l145_14553


namespace complex_magnitude_l145_14550

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_magnitude_l145_14550


namespace sqrt_sum_reciprocal_l145_14530

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end sqrt_sum_reciprocal_l145_14530


namespace max_min_y_over_x_l145_14594

theorem max_min_y_over_x :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 3 →
  (∀ (z w : ℝ), (z - 2)^2 + w^2 = 3 → w / z ≤ Real.sqrt 3) ∧
  (∀ (z w : ℝ), (z - 2)^2 + w^2 = 3 → w / z ≥ -Real.sqrt 3) ∧
  (∃ (x₁ y₁ : ℝ), (x₁ - 2)^2 + y₁^2 = 3 ∧ y₁ / x₁ = Real.sqrt 3) ∧
  (∃ (x₂ y₂ : ℝ), (x₂ - 2)^2 + y₂^2 = 3 ∧ y₂ / x₂ = -Real.sqrt 3) :=
by sorry

end max_min_y_over_x_l145_14594


namespace students_taking_physics_or_chemistry_but_not_both_l145_14574

theorem students_taking_physics_or_chemistry_but_not_both 
  (both : ℕ) 
  (physics : ℕ) 
  (only_chemistry : ℕ) 
  (h1 : both = 12) 
  (h2 : physics = 22) 
  (h3 : only_chemistry = 9) : 
  (physics - both) + only_chemistry = 19 := by
sorry

end students_taking_physics_or_chemistry_but_not_both_l145_14574


namespace total_cost_of_balls_l145_14522

theorem total_cost_of_balls (basketball_price : ℕ) (volleyball_price : ℕ) 
  (basketball_quantity : ℕ) (volleyball_quantity : ℕ) :
  basketball_price = 48 →
  basketball_price = volleyball_price + 18 →
  basketball_quantity = 3 →
  volleyball_quantity = 5 →
  basketball_price * basketball_quantity + volleyball_price * volleyball_quantity = 294 := by
  sorry

end total_cost_of_balls_l145_14522


namespace square_of_101_l145_14565

theorem square_of_101 : (101 : ℕ)^2 = 10201 := by
  sorry

end square_of_101_l145_14565


namespace sequence_problem_l145_14579

/-- An arithmetic sequence where no term is 0 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m ∧ ∀ k, a k ≠ 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : a 5 - a 7 ^ 2 + a 9 = 0)
    (h_b7 : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 := by
  sorry

end sequence_problem_l145_14579


namespace rectangle_arrangement_perimeter_bounds_l145_14575

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents an arrangement of rectangles -/
structure Arrangement where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of an arrangement -/
def perimeter (a : Arrangement) : ℝ :=
  2 * (a.length + a.width)

/-- The set of all possible arrangements of four 7x5 rectangles -/
def possible_arrangements : Set Arrangement :=
  sorry

theorem rectangle_arrangement_perimeter_bounds :
  let r : Rectangle := { length := 7, width := 5 }
  let arrangements := possible_arrangements
  ∃ (max_arr min_arr : Arrangement),
    max_arr ∈ arrangements ∧
    min_arr ∈ arrangements ∧
    (∀ a ∈ arrangements, perimeter a ≤ perimeter max_arr) ∧
    (∀ a ∈ arrangements, perimeter a ≥ perimeter min_arr) ∧
    perimeter max_arr = 66 ∧
    perimeter min_arr = 48 := by
  sorry

end rectangle_arrangement_perimeter_bounds_l145_14575


namespace max_term_binomial_expansion_l145_14548

theorem max_term_binomial_expansion :
  let n : ℕ := 212
  let x : ℝ := Real.sqrt 11
  let term (k : ℕ) : ℝ := (n.choose k) * (x ^ k)
  ∃ k : ℕ, k = 163 ∧ ∀ j : ℕ, j ≠ k → j ≤ n → term k ≥ term j :=
by sorry

end max_term_binomial_expansion_l145_14548


namespace young_worker_proportion_is_three_fifths_l145_14520

/-- The proportion of young workers in a steel works -/
def young_worker_proportion : ℚ := 3/5

/-- The statement that the proportion of young workers is three-fifths -/
theorem young_worker_proportion_is_three_fifths : 
  young_worker_proportion = 3/5 := by
  sorry

end young_worker_proportion_is_three_fifths_l145_14520


namespace dorchester_earnings_l145_14569

def daily_fixed_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def puppies_washed : ℕ := 16

theorem dorchester_earnings :
  daily_fixed_pay + pay_per_puppy * (puppies_washed : ℝ) = 76 := by
  sorry

end dorchester_earnings_l145_14569


namespace cubic_common_root_identity_l145_14525

theorem cubic_common_root_identity (p p' q q' : ℝ) (x : ℝ) :
  (x^3 + p*x + q = 0) ∧ (x^3 + p'*x + q' = 0) →
  (p*q' - q*p') * (p - p')^2 = (q - q')^3 := by
  sorry

end cubic_common_root_identity_l145_14525


namespace sin_theta_in_terms_of_x_l145_14539

theorem sin_theta_in_terms_of_x (θ : Real) (x : Real) (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt (x / (2 * x + 1))) :
  Real.sin θ = (2 * Real.sqrt (x * (x + 1))) / (2 * x + 1) := by
  sorry

end sin_theta_in_terms_of_x_l145_14539


namespace prob_tails_at_least_twice_eq_half_l145_14586

/-- Probability of getting tails k times in n flips of a fair coin -/
def binomialProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The number of coin flips -/
def numFlips : ℕ := 3

/-- Probability of getting tails at least twice but not more than 3 times in 3 flips -/
def probTailsAtLeastTwice : ℚ :=
  binomialProbability numFlips 2 + binomialProbability numFlips 3

theorem prob_tails_at_least_twice_eq_half :
  probTailsAtLeastTwice = 1 / 2 := by
  sorry

end prob_tails_at_least_twice_eq_half_l145_14586


namespace faye_crayons_count_l145_14543

/-- The number of rows of crayons --/
def num_rows : ℕ := 15

/-- The number of crayons in each row --/
def crayons_per_row : ℕ := 42

/-- The total number of crayons --/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem faye_crayons_count : total_crayons = 630 := by
  sorry

end faye_crayons_count_l145_14543


namespace probability_human_given_id_as_human_l145_14549

-- Define the total population
def total_population : ℝ := 1000

-- Define the proportion of vampires and humans
def vampire_proportion : ℝ := 0.99
def human_proportion : ℝ := 1 - vampire_proportion

-- Define the correct identification rates
def vampire_correct_id_rate : ℝ := 0.9
def human_correct_id_rate : ℝ := 0.9

-- Define the number of vampires and humans
def num_vampires : ℝ := vampire_proportion * total_population
def num_humans : ℝ := human_proportion * total_population

-- Define the number of correctly and incorrectly identified vampires and humans
def vampires_id_as_vampires : ℝ := vampire_correct_id_rate * num_vampires
def vampires_id_as_humans : ℝ := (1 - vampire_correct_id_rate) * num_vampires
def humans_id_as_humans : ℝ := human_correct_id_rate * num_humans
def humans_id_as_vampires : ℝ := (1 - human_correct_id_rate) * num_humans

-- Define the total number of individuals identified as humans
def total_id_as_humans : ℝ := vampires_id_as_humans + humans_id_as_humans

-- Theorem statement
theorem probability_human_given_id_as_human :
  humans_id_as_humans / total_id_as_humans = 1 / 12 := by
  sorry

end probability_human_given_id_as_human_l145_14549


namespace abs_equation_solution_difference_l145_14526

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 :=
by sorry

end abs_equation_solution_difference_l145_14526


namespace cookie_eating_contest_l145_14567

theorem cookie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 5/6)
  (h2 : second_student = 7/12) :
  first_student - second_student = 1/4 := by
  sorry

end cookie_eating_contest_l145_14567


namespace count_integer_pairs_l145_14516

theorem count_integer_pairs : ∃ (count : ℕ),
  count = (Finset.filter (fun p : ℕ × ℕ => 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 2012 ∧ 
    (5 : ℝ)^n < (2 : ℝ)^m ∧ 
    (2 : ℝ)^m < (2 : ℝ)^(m+2) ∧ 
    (2 : ℝ)^(m+2) < (5 : ℝ)^(n+1))
  (Finset.product (Finset.range 2013) (Finset.range (2014 + 1)))).card ∧
  (2 : ℝ)^2013 < (5 : ℝ)^867 ∧ (5 : ℝ)^867 < (2 : ℝ)^2014 ∧
  count = 279 := by
  sorry

end count_integer_pairs_l145_14516


namespace empty_container_mass_l145_14595

/-- The mass of an empty container, given its mass when filled with kerosene and water, and the densities of kerosene and water. -/
theorem empty_container_mass
  (mass_with_kerosene : ℝ)
  (mass_with_water : ℝ)
  (density_water : ℝ)
  (density_kerosene : ℝ)
  (h1 : mass_with_kerosene = 20)
  (h2 : mass_with_water = 24)
  (h3 : density_water = 1000)
  (h4 : density_kerosene = 800) :
  ∃ (empty_mass : ℝ), empty_mass = 4 ∧
  mass_with_kerosene = empty_mass + density_kerosene * ((mass_with_water - mass_with_kerosene) / (density_water - density_kerosene)) ∧
  mass_with_water = empty_mass + density_water * ((mass_with_water - mass_with_kerosene) / (density_water - density_kerosene)) :=
by
  sorry


end empty_container_mass_l145_14595


namespace M_superset_P_l145_14554

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = x^2 - 4}
def P : Set ℝ := {y | |y - 3| ≤ 1}

-- State the theorem
theorem M_superset_P : M ⊇ P := by
  sorry

end M_superset_P_l145_14554


namespace exists_n_plus_sum_of_digits_eq_125_l145_14510

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem stating the existence of a natural number n such that n + S(n) = 125 -/
theorem exists_n_plus_sum_of_digits_eq_125 :
  ∃ n : ℕ, n + sumOfDigits n = 125 ∧ n = 121 :=
sorry

end exists_n_plus_sum_of_digits_eq_125_l145_14510


namespace brazil_championship_prob_l145_14506

-- Define the probabilities and point system
def win_prob : ℚ := 1/2
def draw_prob : ℚ := 1/3
def loss_prob : ℚ := 1/6
def win_points : ℕ := 3
def draw_points : ℕ := 1
def loss_points : ℕ := 0

-- Define the number of group stage matches and minimum points to advance
def group_matches : ℕ := 3
def min_points : ℕ := 4

-- Define the probability of winning a penalty shootout
def penalty_win_prob : ℚ := 3/5

-- Define the number of knockout stage matches
def knockout_matches : ℕ := 4

-- Define the function to calculate the probability of winning the championship
-- with exactly one match decided by penalty shootout
def championship_prob : ℚ := sorry

-- State the theorem
theorem brazil_championship_prob : championship_prob = 1/12 := by sorry

end brazil_championship_prob_l145_14506


namespace cos_double_angle_special_case_l145_14501

/-- Given a vector a = (cos α, 1/2) with magnitude √2/2, prove that cos(2α) = -1/2 -/
theorem cos_double_angle_special_case (α : ℝ) :
  let a : ℝ × ℝ := (Real.cos α, 1/2)
  (a.1^2 + a.2^2 = 1/2) →
  Real.cos (2 * α) = -1/2 := by
  sorry

end cos_double_angle_special_case_l145_14501


namespace set_equality_l145_14515

theorem set_equality : {x : ℤ | -3 < x ∧ x < 1} = {-2, -1, 0} := by
  sorry

end set_equality_l145_14515


namespace inequality_solution_set_l145_14599

theorem inequality_solution_set (x : ℝ) : 3 * x + 2 > 5 ↔ x > 1 := by
  sorry

end inequality_solution_set_l145_14599


namespace equation_solution_l145_14583

theorem equation_solution : 
  ∃! x : ℚ, (3 * x - 1) / (4 * x - 4) = 2 / 3 ∧ x = -5 := by sorry

end equation_solution_l145_14583


namespace factorial_ratio_l145_14558

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_ratio (n : ℕ) (h : n ≥ 2) : 
  (factorial n) / (factorial (n - 2)) = n * (n - 1) := by
  sorry

#eval factorial 100 / factorial 98  -- Should output 9900

end factorial_ratio_l145_14558


namespace positive_sum_one_inequality_l145_14556

theorem positive_sum_one_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 := by
  sorry

end positive_sum_one_inequality_l145_14556


namespace ab_value_l145_14512

theorem ab_value (a b : ℝ) (h1 : (a + b)^2 = 4) (h2 : (a - b)^2 = 3) : a * b = 1/4 := by
  sorry

end ab_value_l145_14512


namespace arithmetic_sequence_sum_l145_14576

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a (n + 3) = 39 →
  a (n + 1) + a (n + 2) = 60 := by
  sorry

end arithmetic_sequence_sum_l145_14576


namespace equation_solution_l145_14532

theorem equation_solution : ∃ (x y : ℝ), 
  (1 / 6 + 6 / x = 14 / x + 1 / 14 + y) ∧ (x = 84) ∧ (y = 0) := by
  sorry

end equation_solution_l145_14532


namespace singh_family_seating_arrangements_l145_14535

/-- Represents a family with parents and children -/
structure Family :=
  (parents : ℕ)
  (children : ℕ)

/-- Represents a van with front and back seats -/
structure Van :=
  (front_seats : ℕ)
  (back_seats : ℕ)

/-- Calculates the number of seating arrangements for a family in a van -/
def seating_arrangements (f : Family) (v : Van) : ℕ :=
  sorry

/-- The Singh family -/
def singh_family : Family :=
  { parents := 2, children := 3 }

/-- The Singh family van -/
def singh_van : Van :=
  { front_seats := 2, back_seats := 3 }

theorem singh_family_seating_arrangements :
  seating_arrangements singh_family singh_van = 48 :=
sorry

end singh_family_seating_arrangements_l145_14535


namespace greatest_three_digit_divisible_by_3_5_6_l145_14559

theorem greatest_three_digit_divisible_by_3_5_6 : ∃ n : ℕ, 
  n < 1000 ∧ 
  n ≥ 100 ∧ 
  n % 3 = 0 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧
  ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

end greatest_three_digit_divisible_by_3_5_6_l145_14559


namespace equation_solution_l145_14537

theorem equation_solution : 
  {x : ℝ | x + 60 / (x - 3) = -12} = {-3, -6} := by sorry

end equation_solution_l145_14537


namespace nisos_population_estimate_l145_14578

/-- The initial population of Nisos in the year 2000 -/
def initial_population : ℕ := 400

/-- The number of years between 2000 and 2030 -/
def years_passed : ℕ := 30

/-- The number of years it takes for the population to double -/
def doubling_period : ℕ := 20

/-- The estimated population of Nisos in 2030 -/
def estimated_population_2030 : ℕ := 1131

/-- Theorem stating that the estimated population of Nisos in 2030 is approximately 1131 -/
theorem nisos_population_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (initial_population : ℝ) * (2 : ℝ) ^ (years_passed / doubling_period : ℝ) ∈ 
  Set.Icc (estimated_population_2030 - ε) (estimated_population_2030 + ε) :=
sorry

end nisos_population_estimate_l145_14578


namespace expected_cases_correct_l145_14552

/-- The probability of an American having the disease -/
def disease_probability : ℚ := 1 / 3

/-- The total number of Americans in the sample -/
def sample_size : ℕ := 450

/-- The expected number of Americans with the disease in the sample -/
def expected_cases : ℕ := 150

/-- Theorem stating that the expected number of cases is correct -/
theorem expected_cases_correct : 
  ↑expected_cases = ↑sample_size * disease_probability := by sorry

end expected_cases_correct_l145_14552


namespace rectangle_to_square_l145_14597

-- Define the rectangle dimensions
def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 4

-- Define the number of parts
def num_parts : ℕ := 3

-- Define the square side length
def square_side : ℕ := 6

-- Theorem statement
theorem rectangle_to_square :
  ∃ (part1 part2 part3 : ℕ × ℕ),
    -- The parts fit within the original rectangle
    part1.1 ≤ rectangle_length ∧ part1.2 ≤ rectangle_width ∧
    part2.1 ≤ rectangle_length ∧ part2.2 ≤ rectangle_width ∧
    part3.1 ≤ rectangle_length ∧ part3.2 ≤ rectangle_width ∧
    -- The total area of the parts equals the area of the original rectangle
    part1.1 * part1.2 + part2.1 * part2.2 + part3.1 * part3.2 = rectangle_length * rectangle_width ∧
    -- The parts can form a square
    (part1.1 = square_side ∨ part1.2 = square_side) ∧
    (part2.1 + part3.1 = square_side ∨ part2.2 + part3.2 = square_side) :=
by sorry

#check rectangle_to_square

end rectangle_to_square_l145_14597


namespace cubic_function_uniqueness_l145_14561

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

-- State the theorem
theorem cubic_function_uniqueness :
  -- f is a cubic function
  (∃ a b c d : ℝ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d) →
  -- f has a local maximum value of 4 when x = 1
  (∃ ε > 0, ∀ x, |x - 1| < ε → f x ≤ f 1) ∧ f 1 = 4 →
  -- f has a local minimum value of 0 when x = 3
  (∃ δ > 0, ∀ x, |x - 3| < δ → f x ≥ f 3) ∧ f 3 = 0 →
  -- The graph of f passes through the origin
  f 0 = 0 →
  -- Conclusion: f(x) = x³ - 6x² + 9x for all x
  ∀ x, f x = x^3 - 6*x^2 + 9*x :=
by sorry

end cubic_function_uniqueness_l145_14561


namespace nautical_mile_conversion_l145_14596

/-- Proves that under given conditions, one nautical mile equals 1.15 land miles -/
theorem nautical_mile_conversion (speed_one_sail : ℝ) (speed_two_sails : ℝ) 
  (time_one_sail : ℝ) (time_two_sails : ℝ) (total_distance : ℝ) :
  speed_one_sail = 25 →
  speed_two_sails = 50 →
  time_one_sail = 4 →
  time_two_sails = 4 →
  total_distance = 345 →
  speed_one_sail * time_one_sail + speed_two_sails * time_two_sails = total_distance →
  (1 : ℝ) * (345 / 300) = 1.15 := by
  sorry

#check nautical_mile_conversion

end nautical_mile_conversion_l145_14596


namespace quadratic_root_implies_a_l145_14536

theorem quadratic_root_implies_a (a : ℝ) : 
  (2^2 - a*2 + 6 = 0) → a = 5 := by
  sorry

end quadratic_root_implies_a_l145_14536


namespace inequality_proof_l145_14502

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + y^3 ≥ x^3 + y^4) : x^3 + y^3 ≤ 2 := by
  sorry

end inequality_proof_l145_14502


namespace unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1_l145_14519

theorem unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1 :
  ∃! n : ℕ+, (21 ∣ n) ∧ (9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.1) :=
by sorry

end unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1_l145_14519


namespace min_value_sum_fractions_l145_14511

theorem min_value_sum_fractions (a b c k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (a / (k * b) + b / (k * c) + c / (k * a)) ≥ 3 / k ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀ / (k * b₀) + b₀ / (k * c₀) + c₀ / (k * a₀)) = 3 / k :=
by sorry

end min_value_sum_fractions_l145_14511


namespace intersection_range_l145_14568

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

-- State the theorem
theorem intersection_range (r : ℝ) (h1 : r > 0) (h2 : M ∩ N r = N r) :
  r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
  sorry

-- Note: Set.Ioo represents an open interval (a, b)

end intersection_range_l145_14568


namespace sum_interior_angles_regular_polygon_l145_14540

/-- Given a regular polygon where each exterior angle measures 40°,
    prove that the sum of its interior angles is 1260°. -/
theorem sum_interior_angles_regular_polygon :
  ∀ (n : ℕ), n > 2 →
  (360 : ℝ) / (40 : ℝ) = n →
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end sum_interior_angles_regular_polygon_l145_14540


namespace function_defined_on_reals_l145_14566

/-- The function f(x) = (x^2 - 2)/(x^2 + 1) is defined for all real numbers x. -/
theorem function_defined_on_reals : ∀ x : ℝ, ∃ y : ℝ, y = (x^2 - 2)/(x^2 + 1) :=
sorry

end function_defined_on_reals_l145_14566


namespace magnitude_of_3_minus_4i_l145_14509

theorem magnitude_of_3_minus_4i :
  Complex.abs (3 - 4*Complex.I) = 5 := by
  sorry

end magnitude_of_3_minus_4i_l145_14509


namespace largest_even_digit_multiple_of_9_is_correct_l145_14541

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_9 : ℕ := 882

theorem largest_even_digit_multiple_of_9_is_correct :
  (has_only_even_digits largest_even_digit_multiple_of_9) ∧
  (largest_even_digit_multiple_of_9 < 1000) ∧
  (largest_even_digit_multiple_of_9 % 9 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_9 →
    ¬(has_only_even_digits m ∧ m < 1000 ∧ m % 9 = 0)) :=
by sorry

end largest_even_digit_multiple_of_9_is_correct_l145_14541


namespace profit_and_pricing_analysis_l145_14598

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

/-- Represents the profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 50) * (sales_quantity x)

/-- Represents the new profit function after cost price increase -/
def new_profit (x a : ℝ) : ℝ := (x - 50 - a) * (sales_quantity x)

theorem profit_and_pricing_analysis 
  (cost_price : ℝ) 
  (a : ℝ) 
  (h1 : cost_price = 50) 
  (h2 : a > 0) :
  (∃ x₁ x₂, profit x₁ = 800 ∧ profit x₂ = 800 ∧ x₁ ≠ x₂) ∧ 
  (∃ x_max, ∀ x, profit x ≤ profit x_max) ∧
  (∃ x, 50 + a ≤ x ∧ x ≤ 70 ∧ new_profit x a = 960 ∧ a = 4) := by
  sorry


end profit_and_pricing_analysis_l145_14598


namespace min_throws_for_repeat_sum_l145_14584

/-- Represents a fair six-sided die -/
def Die : Type := Fin 6

/-- The sum of four dice rolls -/
def DiceSum : Type := Nat

/-- The minimum possible sum when rolling four dice -/
def minSum : Nat := 4

/-- The maximum possible sum when rolling four dice -/
def maxSum : Nat := 24

/-- The number of possible unique sums when rolling four dice -/
def uniqueSums : Nat := maxSum - minSum + 1

/-- 
  Theorem: The minimum number of throws needed to ensure the same sum 
  is rolled twice with four fair six-sided dice is 22.
-/
theorem min_throws_for_repeat_sum : 
  (uniqueSums + 1 : Nat) = 22 := by sorry

end min_throws_for_repeat_sum_l145_14584


namespace smallest_candy_count_l145_14523

theorem smallest_candy_count : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (7 ∣ (n + 6)) ∧ 
  (4 ∣ (n - 9)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (7 ∣ (m + 6)) ∧ (4 ∣ (m - 9))) → False) ∧
  n = 113 := by
sorry

end smallest_candy_count_l145_14523


namespace lizzy_candy_spending_l145_14593

/-- The amount of money Lizzy spent on candy --/
def candy_spent : ℕ := sorry

/-- The amount of money Lizzy received from her mother --/
def mother_gave : ℕ := 80

/-- The amount of money Lizzy received from her father --/
def father_gave : ℕ := 40

/-- The amount of money Lizzy received from her uncle --/
def uncle_gave : ℕ := 70

/-- The total amount of money Lizzy has now --/
def current_total : ℕ := 140

theorem lizzy_candy_spending :
  candy_spent = 50 ∧
  current_total = mother_gave + father_gave - candy_spent + uncle_gave :=
sorry

end lizzy_candy_spending_l145_14593


namespace cafeteria_pies_correct_l145_14542

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_correct :
  cafeteria_pies 50 5 5 = 9 := by
  sorry

end cafeteria_pies_correct_l145_14542


namespace lucien_ball_count_l145_14529

/-- Proves that Lucien has 200 balls given the conditions of the problem -/
theorem lucien_ball_count :
  ∀ (lucca_balls lucca_basketballs lucien_basketballs : ℕ) 
    (lucien_balls : ℕ),
  lucca_balls = 100 →
  lucca_basketballs = lucca_balls / 10 →
  lucien_basketballs = lucien_balls / 5 →
  lucca_basketballs + lucien_basketballs = 50 →
  lucien_balls = 200 := by
sorry

end lucien_ball_count_l145_14529


namespace expression_evaluation_l145_14560

theorem expression_evaluation :
  let x : ℚ := -1/4
  (2*x + 1) * (2*x - 1) - (x - 2)^2 - 3*x^2 = -6 :=
by sorry

end expression_evaluation_l145_14560


namespace circle_tangent_y_axis_a_value_l145_14572

/-- A circle is tangent to the y-axis if and only if the absolute value of its center's x-coordinate equals its radius -/
axiom circle_tangent_y_axis {a r : ℝ} (h : ∀ x y : ℝ, (x - a)^2 + (y + 4)^2 = r^2) :
  (∃ y : ℝ, (0 - a)^2 + (y + 4)^2 = r^2) ↔ |a| = r

/-- If a circle with equation (x-a)^2+(y+4)^2=9 is tangent to the y-axis, then a = 3 or a = -3 -/
theorem circle_tangent_y_axis_a_value (h : ∀ x y : ℝ, (x - a)^2 + (y + 4)^2 = 9) 
  (tangent : ∃ y : ℝ, (0 - a)^2 + (y + 4)^2 = 9) : 
  a = 3 ∨ a = -3 := by
  sorry

end circle_tangent_y_axis_a_value_l145_14572


namespace problem_statement_l145_14527

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → a * b < m / 2) ↔ m > 2) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 9 / a + 1 / b ≥ |x - 1| + |x + 2|) ↔ -9/2 ≤ x ∧ x ≤ 7/2) :=
by sorry

end problem_statement_l145_14527


namespace pizzas_bought_l145_14570

def total_slices : ℕ := 32
def slices_left : ℕ := 7
def slices_per_pizza : ℕ := 8

theorem pizzas_bought : (total_slices - slices_left) / slices_per_pizza = 4 := by
  sorry

end pizzas_bought_l145_14570


namespace electronic_devices_bought_l145_14557

theorem electronic_devices_bought (original_price discount_price total_discount : ℕ) 
  (h1 : original_price = 800000)
  (h2 : discount_price = 450000)
  (h3 : total_discount = 16450000) :
  (total_discount / (original_price - discount_price) : ℕ) = 47 := by
  sorry

end electronic_devices_bought_l145_14557


namespace marble_ratio_l145_14521

/-- Proves that the ratio of marbles in a clay pot to marbles in a jar is 3:1 -/
theorem marble_ratio (jars : ℕ) (clay_pots : ℕ) (marbles_per_jar : ℕ) (total_marbles : ℕ) :
  jars = 16 →
  jars = 2 * clay_pots →
  marbles_per_jar = 5 →
  total_marbles = 200 →
  ∃ (marbles_per_pot : ℕ), 
    marbles_per_pot * clay_pots + marbles_per_jar * jars = total_marbles ∧
    marbles_per_pot / marbles_per_jar = 3 :=
by sorry

end marble_ratio_l145_14521


namespace base_conversion_440_to_octal_l145_14544

theorem base_conversion_440_to_octal :
  (440 : ℕ) = 6 * 8^2 + 7 * 8^1 + 0 * 8^0 :=
by sorry

end base_conversion_440_to_octal_l145_14544


namespace work_completion_time_l145_14591

/-- Given that Ravi can do a piece of work in 15 days and Prakash can do it in 30 days,
    prove that they will finish it together in 10 days. -/
theorem work_completion_time (ravi_time prakash_time : ℝ) (h1 : ravi_time = 15) (h2 : prakash_time = 30) :
  1 / (1 / ravi_time + 1 / prakash_time) = 10 := by
  sorry

end work_completion_time_l145_14591


namespace hawkeye_battery_budget_l145_14534

/-- Hawkeye's battery charging problem -/
theorem hawkeye_battery_budget
  (cost_per_charge : ℝ)
  (num_charges : ℕ)
  (money_left : ℝ)
  (h1 : cost_per_charge = 3.5)
  (h2 : num_charges = 4)
  (h3 : money_left = 6) :
  cost_per_charge * num_charges + money_left = 20 := by
  sorry

end hawkeye_battery_budget_l145_14534


namespace garden_length_l145_14562

theorem garden_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 → 
  length = 2 * width → 
  perimeter = 2 * length + 2 * width → 
  perimeter = 900 → 
  length = 300 := by
sorry

end garden_length_l145_14562


namespace painted_cubes_equality_l145_14587

theorem painted_cubes_equality (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 := by
  sorry

end painted_cubes_equality_l145_14587


namespace relay_team_count_l145_14573

/-- The number of sprinters --/
def total_sprinters : ℕ := 6

/-- The number of sprinters to be selected --/
def selected_sprinters : ℕ := 4

/-- The number of ways to form the relay team --/
def relay_team_formations : ℕ := 252

/-- Theorem stating the number of ways to form the relay team --/
theorem relay_team_count :
  (total_sprinters = 6) →
  (selected_sprinters = 4) →
  (∃ A B : ℕ, A ≠ B ∧ A ≤ total_sprinters ∧ B ≤ total_sprinters) →
  relay_team_formations = 252 :=
by sorry

end relay_team_count_l145_14573


namespace symmetry_xoz_plane_l145_14507

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the xOz plane
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

-- Define symmetry with respect to the xOz plane
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_xoz_plane :
  let P := Point3D.mk 3 1 5
  let Q := Point3D.mk 3 (-1) 5
  symmetricPointXOZ P = Q := by sorry

end symmetry_xoz_plane_l145_14507


namespace extreme_points_sum_lower_bound_l145_14564

theorem extreme_points_sum_lower_bound 
  (a : ℝ) 
  (ha : 0 < a ∧ a < 1/8) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x - a * x^2 - Real.log x) 
  (x₁ x₂ : ℝ) 
  (hx : x₁ + x₂ = 1 / (2*a) ∧ x₁ * x₂ = 1 / (2*a)) :
  f x₁ + f x₂ > 3 - 2 * Real.log 2 := by
  sorry

end extreme_points_sum_lower_bound_l145_14564


namespace min_sum_inequality_l145_14514

theorem min_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ 3 * (1 / Real.rpow 105 (1/3)) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (a' / (3 * b') + b' / (5 * c') + c' / (7 * a')) = 3 * (1 / Real.rpow 105 (1/3)) :=
sorry

end min_sum_inequality_l145_14514


namespace min_pizzas_correct_l145_14582

/-- The cost of the car John bought -/
def car_cost : ℕ := 5000

/-- The amount John earns per pizza delivered -/
def earnings_per_pizza : ℕ := 10

/-- The amount John spends on gas per pizza delivered -/
def gas_cost_per_pizza : ℕ := 3

/-- The net profit John makes per pizza delivered -/
def net_profit_per_pizza : ℕ := earnings_per_pizza - gas_cost_per_pizza

/-- The minimum number of pizzas John must deliver to earn back the car cost -/
def min_pizzas : ℕ := (car_cost + net_profit_per_pizza - 1) / net_profit_per_pizza

theorem min_pizzas_correct :
  min_pizzas * net_profit_per_pizza ≥ car_cost ∧
  ∀ n : ℕ, n < min_pizzas → n * net_profit_per_pizza < car_cost :=
by sorry

end min_pizzas_correct_l145_14582


namespace weight_to_lose_in_may_l145_14500

/-- Given Michael's weight loss goal and the amounts he lost in March and April,
    prove that the weight he needs to lose in May is the difference between
    his goal and the sum of weight lost in March and April. -/
theorem weight_to_lose_in_may
  (total_goal : ℕ)
  (march_loss : ℕ)
  (april_loss : ℕ)
  (may_loss : ℕ)
  (h1 : total_goal = 10)
  (h2 : march_loss = 3)
  (h3 : april_loss = 4)
  (h4 : may_loss = total_goal - (march_loss + april_loss)) :
  may_loss = 3 :=
by sorry

end weight_to_lose_in_may_l145_14500


namespace q_at_4_equals_6_l145_14588

-- Define the function q(x)
def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3*|x - 3|^(1/5) + 2

-- Theorem statement
theorem q_at_4_equals_6 : q 4 = 6 := by
  sorry

end q_at_4_equals_6_l145_14588


namespace total_full_price_tickets_is_16525_l145_14590

/-- Represents the ticket sales data for a play over three weeks -/
structure PlayTicketSales where
  total_tickets : ℕ
  week1_tickets : ℕ
  week2_tickets : ℕ
  week3_tickets : ℕ
  week2_full_price_ratio : ℕ
  week3_full_price_ratio : ℕ

/-- Calculates the total number of full-price tickets sold during the play's run -/
def total_full_price_tickets (sales : PlayTicketSales) : ℕ :=
  let week2_full_price := sales.week2_tickets * sales.week2_full_price_ratio / (sales.week2_full_price_ratio + 1)
  let week3_full_price := sales.week3_tickets * sales.week3_full_price_ratio / (sales.week3_full_price_ratio + 1)
  week2_full_price + week3_full_price

/-- Theorem stating that given the specific ticket sales data, the total number of full-price tickets is 16525 -/
theorem total_full_price_tickets_is_16525 (sales : PlayTicketSales) 
  (h1 : sales.total_tickets = 25200)
  (h2 : sales.week1_tickets = 5400)
  (h3 : sales.week2_tickets = 7200)
  (h4 : sales.week3_tickets = 13400)
  (h5 : sales.week2_full_price_ratio = 2)
  (h6 : sales.week3_full_price_ratio = 7) :
  total_full_price_tickets sales = 16525 := by
  sorry


end total_full_price_tickets_is_16525_l145_14590


namespace daisy_sales_difference_l145_14504

/-- Represents the sales of daisies at Daisy's Flower Shop over four days -/
structure DaisySales where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  total : ℕ

/-- Theorem stating the difference in sales between day 2 and day 1 -/
theorem daisy_sales_difference (s : DaisySales) : 
  s.day1 = 45 ∧ 
  s.day2 > s.day1 ∧ 
  s.day3 = 2 * s.day2 - 10 ∧ 
  s.day4 = 120 ∧ 
  s.total = 350 ∧ 
  s.total = s.day1 + s.day2 + s.day3 + s.day4 →
  s.day2 - s.day1 = 20 := by
  sorry

#check daisy_sales_difference

end daisy_sales_difference_l145_14504


namespace second_smallest_number_l145_14581

def digits : List Nat := [1, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10 = 5) ∧ (n % 10 ∈ digits) ∧ (n / 10 ∈ digits)

def count_smaller (n : Nat) : Nat :=
  (digits.filter (λ d => d < n % 10)).length

theorem second_smallest_number :
  ∃ n : Nat, is_valid_number n ∧ count_smaller n = 1 ∧ n = 56 := by
  sorry

end second_smallest_number_l145_14581


namespace f_max_min_difference_l145_14551

noncomputable def f (x : ℝ) := Real.exp (Real.sin x + Real.cos x) - (1/2) * Real.sin (2 * x)

theorem f_max_min_difference :
  (⨆ (x : ℝ), f x) - (⨅ (x : ℝ), f x) = Real.exp (Real.sqrt 2) - Real.exp (-Real.sqrt 2) := by
  sorry

end f_max_min_difference_l145_14551


namespace smallest_sum_of_perfect_squares_l145_14592

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → (∀ a b : ℕ, a^2 - b^2 = 221 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 24421 := by
sorry

end smallest_sum_of_perfect_squares_l145_14592


namespace largest_common_divisor_of_consecutive_odd_products_l145_14505

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def consecutive_odd_integers (a b c d : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

theorem largest_common_divisor_of_consecutive_odd_products :
  ∀ a b c d : ℕ,
  consecutive_odd_integers a b c d →
  (∃ k : ℕ, a * b * c * d = 3 * k) ∧
  (∀ m : ℕ, m > 3 → ∃ x y z w : ℕ, 
    consecutive_odd_integers x y z w ∧ 
    ¬(∃ k : ℕ, x * y * z * w = m * k)) :=
by sorry

end largest_common_divisor_of_consecutive_odd_products_l145_14505


namespace multiplication_grid_problem_l145_14518

theorem multiplication_grid_problem :
  ∃ (a b : ℕ+), 
    a * b = 1843 ∧ 
    (1843 % 10 = 3) ∧ 
    ((1843 / 10) % 10 = 8) := by
  sorry

end multiplication_grid_problem_l145_14518


namespace h_function_iff_strictly_increasing_l145_14580

/-- A function f: ℝ → ℝ is an "H function" if for any two distinct real numbers x₁ and x₂,
    the condition x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁ holds. -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function f: ℝ → ℝ is strictly increasing if for any two real numbers x₁ and x₂,
    x₁ < x₂ implies f x₁ < f x₂. -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

/-- Theorem: A function is an "H function" if and only if it is strictly increasing. -/
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_H_function f ↔ strictly_increasing f :=
sorry

end h_function_iff_strictly_increasing_l145_14580


namespace students_between_minyoung_and_hoseok_l145_14547

/-- Given 13 students in a line, with Minyoung at the 8th position from the left
    and Hoseok at the 9th position from the right, prove that the number of
    students between Minyoung and Hoseok is 2. -/
theorem students_between_minyoung_and_hoseok :
  let total_students : ℕ := 13
  let minyoung_position : ℕ := 8
  let hoseok_position_from_right : ℕ := 9
  let hoseok_position : ℕ := total_students - hoseok_position_from_right + 1
  (minyoung_position - hoseok_position - 1 : ℕ) = 2 := by
  sorry

end students_between_minyoung_and_hoseok_l145_14547


namespace sqrt_x_cubed_sqrt_x_l145_14513

theorem sqrt_x_cubed_sqrt_x (x : ℝ) (hx : x > 0) : Real.sqrt (x^3 * Real.sqrt x) = x^(7/4) := by
  sorry

end sqrt_x_cubed_sqrt_x_l145_14513
