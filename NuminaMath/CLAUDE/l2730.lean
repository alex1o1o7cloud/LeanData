import Mathlib

namespace monomial_exponents_sum_l2730_273067

/-- Two monomials are like terms if their variables have the same exponents -/
def are_like_terms (a b c d : ℕ) : Prop :=
  a = c ∧ b = d

theorem monomial_exponents_sum (m n : ℕ) : 
  are_like_terms 5 (2*n) m 4 → m + n = 7 := by
  sorry

end monomial_exponents_sum_l2730_273067


namespace f_properties_l2730_273041

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def has_max_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ c

-- State the theorem
theorem f_properties (hsym : symmetric_about_origin f)
                     (hinc : increasing_on f 3 7)
                     (hmax : has_max_value f 3 7 5) :
  increasing_on f (-7) (-3) ∧ has_max_value f (-7) (-3) (-5) := by
  sorry

end f_properties_l2730_273041


namespace lcm_factor_proof_l2730_273046

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ+) (h1 : Nat.gcd A B = 23)
  (h2 : Nat.lcm A B = 23 * X * 16) (h3 : A = 368) : X = 1 := by
  sorry

end lcm_factor_proof_l2730_273046


namespace sum_three_consecutive_terms_l2730_273065

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_three_consecutive_terms
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 21) :
  a 4 + a 5 + a 6 = 63 :=
sorry

end sum_three_consecutive_terms_l2730_273065


namespace min_length_roots_l2730_273020

/-- Given a quadratic function f(x) = a x^2 + (16 - a^3) x - 16 a^2, where a > 0,
    the minimum length of the line segment connecting its roots is 12. -/
theorem min_length_roots (a : ℝ) (ha : a > 0) :
  let f := fun x : ℝ => a * x^2 + (16 - a^3) * x - 16 * a^2
  let roots := {x : ℝ | f x = 0}
  let length := fun (r₁ r₂ : ℝ) => |r₁ - r₂|
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧
    ∀ (s₁ s₂ : ℝ), s₁ ∈ roots → s₂ ∈ roots → s₁ ≠ s₂ →
      length r₁ r₂ ≤ length s₁ s₂ ∧ length r₁ r₂ = 12 :=
by sorry


end min_length_roots_l2730_273020


namespace nori_crayons_l2730_273012

theorem nori_crayons (initial_boxes : ℕ) (crayons_per_box : ℕ) (crayons_left : ℕ) (extra_to_lea : ℕ) :
  initial_boxes = 4 →
  crayons_per_box = 8 →
  crayons_left = 15 →
  extra_to_lea = 7 →
  ∃ (crayons_to_mae : ℕ),
    initial_boxes * crayons_per_box = crayons_left + crayons_to_mae + (crayons_to_mae + extra_to_lea) ∧
    crayons_to_mae = 5 :=
by sorry

end nori_crayons_l2730_273012


namespace max_product_digits_l2730_273061

theorem max_product_digits : ∀ a b : ℕ,
  10000 ≤ a ∧ a < 100000 →
  1000 ≤ b ∧ b < 10000 →
  a * b < 1000000000 := by
sorry

end max_product_digits_l2730_273061


namespace fifteenth_digit_of_sum_l2730_273085

def decimal_rep_1_9 : ℚ := 1 / 9
def decimal_rep_1_11 : ℚ := 1 / 11

def sum_of_reps : ℚ := decimal_rep_1_9 + decimal_rep_1_11

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_of_sum :
  nth_digit_after_decimal sum_of_reps 15 = 1 := by sorry

end fifteenth_digit_of_sum_l2730_273085


namespace tangent_point_coordinates_l2730_273029

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_coordinates (a : ℝ) :
  (∀ x, f_deriv a (-x) = -f_deriv a x) →
  ∃ x₀ y₀, f a x₀ = y₀ ∧ f_deriv a x₀ = 3/2 →
  x₀ = Real.log 2 ∧ y₀ = 5/2 := by sorry

end tangent_point_coordinates_l2730_273029


namespace two_machines_half_hour_copies_l2730_273043

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.copies_per_minute * minutes

/-- Theorem: Two copy machines working together for half an hour will produce 3300 copies -/
theorem two_machines_half_hour_copies :
  let machine1 : CopyMachine := ⟨35⟩
  let machine2 : CopyMachine := ⟨75⟩
  let half_hour : ℕ := 30
  (copies_made machine1 half_hour) + (copies_made machine2 half_hour) = 3300 :=
by
  sorry

end two_machines_half_hour_copies_l2730_273043


namespace circle_delta_area_l2730_273099

/-- Circle δ with points A and B -/
structure Circle_delta where
  center : ℝ × ℝ
  radius : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Conditions for the circle δ -/
def circle_conditions (δ : Circle_delta) : Prop :=
  δ.A = (2, 9) ∧ 
  δ.B = (10, 5) ∧
  (δ.A.1 - δ.center.1)^2 + (δ.A.2 - δ.center.2)^2 = δ.radius^2 ∧
  (δ.B.1 - δ.center.1)^2 + (δ.B.2 - δ.center.2)^2 = δ.radius^2

/-- Tangent lines intersection condition -/
def tangent_intersection (δ : Circle_delta) : Prop :=
  ∃ x : ℝ, 
    let slope_AB := (δ.B.2 - δ.A.2) / (δ.B.1 - δ.A.1)
    let perp_slope := -1 / slope_AB
    let midpoint := ((δ.A.1 + δ.B.1) / 2, (δ.A.2 + δ.B.2) / 2)
    perp_slope * (x - midpoint.1) + midpoint.2 = 0

/-- Theorem stating the area of circle δ -/
theorem circle_delta_area (δ : Circle_delta) 
  (h1 : circle_conditions δ) (h2 : tangent_intersection δ) : 
  π * δ.radius^2 = 83.44 * π := by
  sorry

end circle_delta_area_l2730_273099


namespace perimeter_of_8x4_formation_l2730_273044

/-- A rectangular formation of students -/
structure Formation :=
  (rows : ℕ)
  (columns : ℕ)

/-- The number of elements on the perimeter of a formation -/
def perimeter_count (f : Formation) : ℕ :=
  2 * (f.rows + f.columns) - 4

theorem perimeter_of_8x4_formation :
  let f : Formation := ⟨8, 4⟩
  perimeter_count f = 20 := by sorry

end perimeter_of_8x4_formation_l2730_273044


namespace fractional_inequality_solution_set_l2730_273058

theorem fractional_inequality_solution_set (x : ℝ) :
  (x + 5) / (1 - x) ≤ 0 ↔ (x ≤ -5 ∨ x > 1) ∧ x ≠ 1 :=
sorry

end fractional_inequality_solution_set_l2730_273058


namespace intersecting_circles_m_plus_c_l2730_273086

/-- Two circles intersect at points A and B, with the centers of the circles lying on a line. -/
structure IntersectingCircles where
  m : ℝ
  c : ℝ
  A : ℝ × ℝ := (1, 3)
  B : ℝ × ℝ := (m, -1)
  centers_line : ℝ → ℝ := fun x ↦ x + c

/-- The value of m+c for the given intersecting circles configuration is 3. -/
theorem intersecting_circles_m_plus_c (circles : IntersectingCircles) : 
  circles.m + circles.c = 3 := by
  sorry


end intersecting_circles_m_plus_c_l2730_273086


namespace recruitment_plans_count_l2730_273076

/-- Represents the daily installation capacity of workers -/
structure WorkerCapacity where
  skilled : ℕ
  new : ℕ

/-- Represents a recruitment plan -/
structure RecruitmentPlan where
  skilled : ℕ
  new : ℕ

/-- Checks if a recruitment plan is valid given the constraints -/
def isValidPlan (plan : RecruitmentPlan) : Prop :=
  1 < plan.skilled ∧ plan.skilled < 8 ∧ 0 < plan.new

/-- Checks if a recruitment plan can complete the task -/
def canCompleteTask (capacity : WorkerCapacity) (plan : RecruitmentPlan) : Prop :=
  15 * (capacity.skilled * plan.skilled + capacity.new * plan.new) = 360

/-- Main theorem to prove -/
theorem recruitment_plans_count 
  (capacity : WorkerCapacity)
  (h1 : 2 * capacity.skilled + capacity.new = 10)
  (h2 : 3 * capacity.skilled + 2 * capacity.new = 16) :
  ∃! (plans : Finset RecruitmentPlan), 
    plans.card = 4 ∧ 
    (∀ plan ∈ plans, isValidPlan plan ∧ canCompleteTask capacity plan) ∧
    (∀ plan, isValidPlan plan ∧ canCompleteTask capacity plan → plan ∈ plans) :=
  sorry

end recruitment_plans_count_l2730_273076


namespace triangle_altitude_sum_square_l2730_273060

theorem triangle_altitude_sum_square (a b c : ℕ) : 
  (∃ (h_a h_b h_c : ℝ), h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ 
    h_a = (2 * (a * h_a / 2)) / a ∧ 
    h_b = (2 * (b * h_b / 2)) / b ∧ 
    h_c = (2 * (c * h_c / 2)) / c ∧ 
    h_a = h_b + h_c) → 
  ∃ (k : ℚ), a^2 + b^2 + c^2 = k^2 := by
sorry

end triangle_altitude_sum_square_l2730_273060


namespace carla_marbles_l2730_273089

/-- The number of marbles Carla bought -/
def marbles_bought (initial final : ℕ) : ℕ := final - initial

/-- Proof that Carla bought 134 marbles -/
theorem carla_marbles : marbles_bought 53 187 = 134 := by
  sorry

end carla_marbles_l2730_273089


namespace m_less_than_neg_two_l2730_273070

/-- A quadratic function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The proposition that there exists a positive x_0 where f(x_0) < 0 -/
def exists_positive_root (m : ℝ) : Prop :=
  ∃ x_0 : ℝ, x_0 > 0 ∧ f m x_0 < 0

/-- Theorem: If there exists a positive x_0 where f(x_0) < 0, then m < -2 -/
theorem m_less_than_neg_two (m : ℝ) (h : exists_positive_root m) : m < -2 := by
  sorry

end m_less_than_neg_two_l2730_273070


namespace amanda_notebooks_l2730_273008

/-- Calculate the final number of notebooks Amanda has -/
def final_notebooks (initial ordered lost : ℕ) : ℕ :=
  initial + ordered - lost

/-- Theorem stating that Amanda's final number of notebooks is 74 -/
theorem amanda_notebooks : final_notebooks 65 23 14 = 74 := by
  sorry

end amanda_notebooks_l2730_273008


namespace quadratic_factorization_l2730_273098

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25*x^2 - 85*x - 90 = (5*x + c) * (5*x + d)) → c + 2*d = -24 := by
  sorry

end quadratic_factorization_l2730_273098


namespace gnuff_tutoring_time_l2730_273031

/-- Proves that given Gnuff's tutoring rates and total amount paid, the number of minutes tutored is 18 --/
theorem gnuff_tutoring_time (flat_rate per_minute_rate total_paid : ℕ) : 
  flat_rate = 20 → 
  per_minute_rate = 7 → 
  total_paid = 146 → 
  (total_paid - flat_rate) / per_minute_rate = 18 := by
sorry

end gnuff_tutoring_time_l2730_273031


namespace harold_marbles_distribution_l2730_273052

def marble_distribution (total_marbles : ℕ) (kept_marbles : ℕ) (num_friends : ℕ) : ℕ :=
  (total_marbles - kept_marbles) / num_friends

theorem harold_marbles_distribution :
  marble_distribution 100 20 5 = 16 := by
  sorry

end harold_marbles_distribution_l2730_273052


namespace exists_digit_satisfying_equation_l2730_273013

theorem exists_digit_satisfying_equation : ∃ a : ℕ, 
  0 ≤ a ∧ a ≤ 9 ∧ 1111 * a - 1 = (a - 1) ^ (a - 2) := by
  sorry

end exists_digit_satisfying_equation_l2730_273013


namespace circle_equation_l2730_273072

/-- The equation of a circle passing through point P(2,5) with center C(8,-3) -/
theorem circle_equation (x y : ℝ) : 
  let P : ℝ × ℝ := (2, 5)
  let C : ℝ × ℝ := (8, -3)
  (x - C.1)^2 + (y - C.2)^2 = (P.1 - C.1)^2 + (P.2 - C.2)^2 ↔ 
  (x - 8)^2 + (y + 3)^2 = 100 :=
by sorry

end circle_equation_l2730_273072


namespace good_number_ending_8_has_9_before_l2730_273048

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

def ends_with_8 (n : ℕ) : Prop :=
  n % 10 = 8

def second_to_last_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem good_number_ending_8_has_9_before :
  ∀ n : ℕ, is_good n → ends_with_8 n → second_to_last_digit n = 9 := by
  sorry

end good_number_ending_8_has_9_before_l2730_273048


namespace hockey_championship_wins_l2730_273049

theorem hockey_championship_wins (total_matches : ℕ) (total_points : ℤ) 
  (win_points loss_points : ℤ) (h_total_matches : total_matches = 38) 
  (h_total_points : total_points = 60) (h_win_points : win_points = 12) 
  (h_loss_points : loss_points = 5) : 
  ∃! wins : ℕ, ∃ losses draws : ℕ,
    wins + losses + draws = total_matches ∧ 
    wins * win_points - losses * loss_points = total_points ∧
    losses > 0 := by
  sorry

#check hockey_championship_wins

end hockey_championship_wins_l2730_273049


namespace smallest_valid_number_l2730_273083

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  ∃ k, n = k * (lcm 3 (lcm 4 (lcm 5 (lcm 6 7)))) + 1

theorem smallest_valid_number : 
  is_valid_number 61 ∧ ∀ m, is_valid_number m → m ≥ 61 :=
sorry

end smallest_valid_number_l2730_273083


namespace fabric_needed_calculation_l2730_273019

/-- Calculates the additional fabric needed for dresses -/
def additional_fabric_needed (yards_per_dress : Float) (dresses : Nat) (available : Float) : Float :=
  yards_per_dress * dresses.toFloat * 3 - available

theorem fabric_needed_calculation (floral_yards_per_dress : Float) 
                                  (striped_yards_per_dress : Float)
                                  (polka_dot_yards_per_dress : Float)
                                  (floral_available : Float)
                                  (striped_available : Float)
                                  (polka_dot_available : Float) :
  floral_yards_per_dress = 5.25 →
  striped_yards_per_dress = 6.75 →
  polka_dot_yards_per_dress = 7.15 →
  floral_available = 12 →
  striped_available = 6 →
  polka_dot_available = 15 →
  additional_fabric_needed floral_yards_per_dress 2 floral_available = 19.5 ∧
  additional_fabric_needed striped_yards_per_dress 2 striped_available = 34.5 ∧
  additional_fabric_needed polka_dot_yards_per_dress 2 polka_dot_available = 27.9 :=
by sorry

end fabric_needed_calculation_l2730_273019


namespace inverse_proportion_problem_l2730_273055

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x - y = 10) :
  x = 4 → y = 50 := by
  sorry

end inverse_proportion_problem_l2730_273055


namespace students_in_circle_l2730_273022

/-- 
Given a circle of students where the 6th and 16th students are opposite each other,
prove that the total number of students is 18.
-/
theorem students_in_circle (n : ℕ) 
  (h1 : n > 0) -- Ensure there are students in the circle
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ a ≤ n ∧ b ≤ n) -- 6th and 16th students exist
  (h3 : (16 - 6) * 2 + 2 = n) -- Condition for 6th and 16th being opposite
  : n = 18 := by
  sorry

end students_in_circle_l2730_273022


namespace tadd_number_count_l2730_273001

theorem tadd_number_count : 
  let n : ℕ := 20  -- number of rounds
  let a : ℕ := 1   -- first term of the sequence
  let d : ℕ := 2   -- common difference
  let l : ℕ := a + d * (n - 1)  -- last term
  (n : ℚ) / 2 * (a + l) = 400 := by
  sorry

end tadd_number_count_l2730_273001


namespace smallest_r_value_l2730_273025

theorem smallest_r_value (p q r : ℕ) : 
  0 < p ∧ p < q ∧ q < r ∧                   -- p, q, r are positive integers and p < q < r
  (2 * q = p + r) ∧                         -- arithmetic progression
  (r * r = p * q) →                         -- geometric progression
  r ≥ 5 ∧ ∃ (p' q' r' : ℕ), 
    0 < p' ∧ p' < q' ∧ q' < r' ∧ 
    (2 * q' = p' + r') ∧ 
    (r' * r' = p' * q') ∧ 
    r' = 5 :=
by sorry

end smallest_r_value_l2730_273025


namespace pool_perimeter_l2730_273027

theorem pool_perimeter (garden_length garden_width pool_area : ℝ) 
  (h1 : garden_length = 8)
  (h2 : garden_width = 6)
  (h3 : pool_area = 24)
  (h4 : ∃ x : ℝ, (garden_length - 2*x) * (garden_width - 2*x) = pool_area ∧ 
                 x > 0 ∧ x < garden_length/2 ∧ x < garden_width/2) :
  ∃ pool_length pool_width : ℝ,
    pool_length * pool_width = pool_area ∧
    pool_length < garden_length ∧
    pool_width < garden_width ∧
    2 * pool_length + 2 * pool_width = 20 :=
by sorry

end pool_perimeter_l2730_273027


namespace employee_satisfaction_theorem_l2730_273007

-- Define the total number of people surveyed
def total_people : ℕ := 200

-- Define the number of people satisfied with both
def satisfied_both : ℕ := 50

-- Define the number of people satisfied with employee dedication
def satisfied_dedication : ℕ := (40 * total_people) / 100

-- Define the number of people satisfied with management level
def satisfied_management : ℕ := (45 * total_people) / 100

-- Define the chi-square function
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value
def critical_value : ℚ := 6635 / 1000

-- Define the probability of being satisfied with both
def prob_both : ℚ := satisfied_both / total_people

-- Theorem statement
theorem employee_satisfaction_theorem :
  let a := satisfied_both
  let b := satisfied_dedication - satisfied_both
  let c := satisfied_management - satisfied_both
  let d := total_people - satisfied_dedication - satisfied_management + satisfied_both
  (chi_square a b c d > critical_value) ∧
  (3 * prob_both * (1 - prob_both)^2 + 2 * 3 * prob_both^2 * (1 - prob_both) + 3 * prob_both^3 = 3/4) :=
by sorry

end employee_satisfaction_theorem_l2730_273007


namespace one_cow_one_bag_days_l2730_273077

/-- The number of days it takes for one cow to eat one bag of husk -/
def days_for_one_cow_one_bag (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) : ℚ :=
  (num_days : ℚ) * (num_cows : ℚ) / (num_bags : ℚ)

/-- Theorem stating that one cow will eat one bag of husk in 36 days -/
theorem one_cow_one_bag_days :
  days_for_one_cow_one_bag 60 75 45 = 36 := by
  sorry

#eval days_for_one_cow_one_bag 60 75 45

end one_cow_one_bag_days_l2730_273077


namespace set_operations_l2730_273081

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | -3 ≤ x ∧ x < -1}) ∧
  (A ∪ B = {x | x ≤ 4 ∨ x > 5}) ∧
  ((Set.compl A) ∩ (Set.compl B) = {x | 4 < x ∧ x ≤ 5}) :=
sorry

end set_operations_l2730_273081


namespace least_four_digit_number_with_conditions_l2730_273066

/-- A function that checks if a number has all different digits -/
def hasDifferentDigits (n : ℕ) : Prop := sorry

/-- A function that checks if a number includes the digit 5 -/
def includesFive (n : ℕ) : Prop := sorry

/-- A function that checks if a number is divisible by all of its digits -/
def divisibleByAllDigits (n : ℕ) : Prop := sorry

theorem least_four_digit_number_with_conditions :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    hasDifferentDigits n ∧
    includesFive n ∧
    divisibleByAllDigits n →
    1536 ≤ n :=
by sorry

end least_four_digit_number_with_conditions_l2730_273066


namespace donation_distribution_l2730_273023

theorem donation_distribution (giselle_amount sam_amount isabella_amount : ℕ) :
  giselle_amount = 120 →
  isabella_amount = giselle_amount + 15 →
  isabella_amount = sam_amount + 45 →
  (isabella_amount + giselle_amount + sam_amount) / 3 = 115 :=
by sorry

end donation_distribution_l2730_273023


namespace arctan_sum_problem_l2730_273030

theorem arctan_sum_problem (a b : ℝ) : 
  a = 1/3 → 
  (a + 2) * (b + 2) = 15 → 
  Real.arctan a + Real.arctan b = 5 * π / 6 := by
sorry

end arctan_sum_problem_l2730_273030


namespace slope_problem_l2730_273011

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = 2 * m) : m = (3 + Real.sqrt 41) / 4 := by
  sorry

end slope_problem_l2730_273011


namespace hajis_mother_sales_l2730_273090

/-- Haji's mother's sales problem -/
theorem hajis_mother_sales (tough_week_sales : ℕ) (total_sales : ℕ) :
  tough_week_sales = 800 →
  total_sales = 10400 →
  ∃ (good_weeks : ℕ),
    good_weeks * (2 * tough_week_sales) + 3 * tough_week_sales = total_sales ∧
    good_weeks = 5 :=
by sorry

end hajis_mother_sales_l2730_273090


namespace probability_two_female_contestants_l2730_273016

theorem probability_two_female_contestants (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 8 →
  female = 5 →
  male = 3 →
  (female.choose 2 : ℚ) / (total.choose 2) = 5 / 14 := by
  sorry

end probability_two_female_contestants_l2730_273016


namespace permutation_calculation_l2730_273063

/-- Definition of permutation notation -/
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem stating that A₆² A₄² equals 360 -/
theorem permutation_calculation : A 6 2 * A 4 2 = 360 := by
  sorry

end permutation_calculation_l2730_273063


namespace vacation_miles_theorem_l2730_273056

/-- Calculates the total miles driven during a vacation -/
def total_miles_driven (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that a 5-day vacation driving 250 miles per day results in 1250 total miles -/
theorem vacation_miles_theorem :
  total_miles_driven 5 250 = 1250 := by
  sorry

end vacation_miles_theorem_l2730_273056


namespace set_A_at_most_one_element_iff_a_in_range_l2730_273005

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + a = 0}

-- Theorem statement
theorem set_A_at_most_one_element_iff_a_in_range :
  ∀ a : ℝ, (∃ (x y : ℝ), x ∈ A a ∧ y ∈ A a ∧ x ≠ y) ↔ a ∈ {a : ℝ | a < -1 ∨ (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)} :=
by sorry

end set_A_at_most_one_element_iff_a_in_range_l2730_273005


namespace sum_of_squares_inequality_l2730_273033

theorem sum_of_squares_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) ≥ 1/2 ∧
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1/2 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end sum_of_squares_inequality_l2730_273033


namespace ellipse_minimum_value_l2730_273059

theorem ellipse_minimum_value (x y : ℝ) :
  x > 0 → y > 0 → x^2 / 16 + y^2 / 12 = 1 →
  x / (4 - x) + 3 * y / (6 - y) ≥ 4 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 / 16 + y₀^2 / 12 = 1 ∧
    x₀ / (4 - x₀) + 3 * y₀ / (6 - y₀) = 4 := by
  sorry

end ellipse_minimum_value_l2730_273059


namespace committee_probability_l2730_273000

/-- The number of members in the Literature club -/
def total_members : ℕ := 24

/-- The number of boys in the Literature club -/
def num_boys : ℕ := 12

/-- The number of girls in the Literature club -/
def num_girls : ℕ := 12

/-- The size of the committee to be chosen -/
def committee_size : ℕ := 5

/-- The probability of choosing a committee with at least 2 boys and at least 2 girls -/
theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (2 * Nat.choose num_boys committee_size + 
    Nat.choose num_boys 1 * Nat.choose num_girls 4 + 
    Nat.choose num_girls 1 * Nat.choose num_boys 4)) / 
   Nat.choose total_members committee_size = 121 / 177 := by
  sorry

end committee_probability_l2730_273000


namespace parabola_equation_l2730_273080

/-- A parabola with the same shape and orientation as y = -2x^2 + 2 and vertex (4, -2) -/
structure Parabola where
  shape_coeff : ℝ
  vertex : ℝ × ℝ
  shape_matches : shape_coeff = -2
  vertex_coords : vertex = (4, -2)

/-- The analytical expression of the parabola -/
def parabola_expression (p : Parabola) (x : ℝ) : ℝ :=
  p.shape_coeff * (x - p.vertex.1)^2 + p.vertex.2

theorem parabola_equation (p : Parabola) :
  ∀ x, parabola_expression p x = -2 * (x - 4)^2 - 2 :=
sorry

end parabola_equation_l2730_273080


namespace arithmetic_sequence_problem_l2730_273009

/-- Given an arithmetic sequence {a_n} where S_n is the sum of the first n terms,
    if (S_2016 / 2016) - (S_2015 / 2015) = 3, then a_2016 - a_2014 = 12. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (S 2016 / 2016 - S 2015 / 2015 = 3) →
  a 2016 - a 2014 = 12 := by
  sorry

end arithmetic_sequence_problem_l2730_273009


namespace no_identical_lines_l2730_273017

theorem no_identical_lines : ¬∃ (a d : ℝ), ∀ (x y : ℝ),
  (5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) := by
  sorry

end no_identical_lines_l2730_273017


namespace smallest_divisible_by_15_13_18_l2730_273064

theorem smallest_divisible_by_15_13_18 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 13 ∣ n ∧ 18 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 13 ∣ m → 18 ∣ m → n ≤ m :=
by sorry

end smallest_divisible_by_15_13_18_l2730_273064


namespace complex_magnitude_l2730_273068

theorem complex_magnitude (z : ℂ) (h : z * (1 - 2*I) = 3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l2730_273068


namespace midpoint_trajectory_l2730_273042

theorem midpoint_trajectory (a b : ℝ) : 
  a^2 + b^2 = 1 → ∃ x y : ℝ, x = a ∧ y = b/2 ∧ x^2 + 4*y^2 = 1 := by
  sorry

end midpoint_trajectory_l2730_273042


namespace knicks_knacks_knocks_conversion_l2730_273075

/-- Given the conversion rates between knicks, knacks, and knocks, 
    prove that 30 knocks are equal to 20 knicks. -/
theorem knicks_knacks_knocks_conversion :
  ∀ (knicks knacks knocks : ℚ),
    (5 * knicks = 3 * knacks) →
    (2 * knacks = 5 * knocks) →
    (30 * knocks = 20 * knicks) :=
by
  sorry

end knicks_knacks_knocks_conversion_l2730_273075


namespace average_cost_approx_1_50_l2730_273069

/-- Calculates the average cost per piece of fruit given specific quantities and prices. -/
def average_cost_per_fruit (apple_price banana_price orange_price grape_price kiwi_price : ℚ)
  (apple_qty banana_qty orange_qty grape_qty kiwi_qty : ℕ) : ℚ :=
  let apple_cost := if apple_qty ≥ 10 then (apple_qty - 2) * apple_price else apple_qty * apple_price
  let orange_cost := if orange_qty ≥ 3 then (orange_qty - (orange_qty / 3)) * orange_price else orange_qty * orange_price
  let grape_cost := if grape_qty * grape_price > 10 then grape_qty * grape_price * (1 - 0.2) else grape_qty * grape_price
  let kiwi_cost := if kiwi_qty ≥ 10 then kiwi_qty * kiwi_price * (1 - 0.15) else kiwi_qty * kiwi_price
  let banana_cost := banana_qty * banana_price
  let total_cost := apple_cost + orange_cost + grape_cost + kiwi_cost + banana_cost
  let total_pieces := apple_qty + orange_qty + grape_qty + kiwi_qty + banana_qty
  total_cost / total_pieces

/-- The average cost per piece of fruit is approximately $1.50 given the specific conditions. -/
theorem average_cost_approx_1_50 :
  ∃ ε > 0, |average_cost_per_fruit 2 1 3 (3/2) (7/4) 12 4 4 10 10 - (3/2)| < ε :=
by sorry

end average_cost_approx_1_50_l2730_273069


namespace seaweed_for_fires_l2730_273028

theorem seaweed_for_fires (total_seaweed livestock_feed : ℝ)
  (h1 : total_seaweed = 400)
  (h2 : livestock_feed = 150)
  (h3 : livestock_feed = 0.75 * (1 - fire_percentage / 100) * total_seaweed) :
  fire_percentage = 50 := by
  sorry

end seaweed_for_fires_l2730_273028


namespace system_solution_l2730_273006

theorem system_solution :
  ∃ x y : ℝ, 
    (4 * x - 3 * y = -0.75) ∧
    (5 * x + 3 * y = 5.35) ∧
    (abs (x - 0.5111) < 0.0001) ∧
    (abs (y - 0.9315) < 0.0001) := by
  sorry

end system_solution_l2730_273006


namespace mary_circus_change_l2730_273014

/-- Calculates the change Mary receives after buying circus tickets for herself and her children -/
theorem mary_circus_change (num_children : ℕ) (adult_price child_price payment : ℚ) : 
  num_children = 3 ∧ 
  adult_price = 2 ∧ 
  child_price = 1 ∧ 
  payment = 20 → 
  payment - (adult_price + num_children * child_price) = 15 := by
  sorry

end mary_circus_change_l2730_273014


namespace cubic_equation_result_l2730_273021

theorem cubic_equation_result (x : ℝ) (h : x^3 + 2*x = 4) : x^7 + 32*x^2 = 64 := by
  sorry

end cubic_equation_result_l2730_273021


namespace smallest_positive_root_l2730_273088

theorem smallest_positive_root (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 3) (hd : |d| ≤ 2) :
  ∃ (s : ℝ), s > 0 ∧ s^3 + b*s^2 + c*s + d = 0 ∧
  ∀ (x : ℝ), x > 0 ∧ x^3 + b*x^2 + c*x + d = 0 → s ≤ x :=
by
  sorry

end smallest_positive_root_l2730_273088


namespace orange_distribution_l2730_273078

/-- The number of ways to distribute distinct oranges to sons. -/
def distribute_oranges (num_oranges : ℕ) (num_sons : ℕ) : ℕ :=
  (num_sons.choose num_oranges) * num_oranges.factorial

/-- Theorem: The number of ways to distribute 5 distinct oranges to 8 sons is 6720. -/
theorem orange_distribution :
  distribute_oranges 5 8 = 6720 := by
  sorry

#eval distribute_oranges 5 8

end orange_distribution_l2730_273078


namespace division_of_fractions_l2730_273092

theorem division_of_fractions : (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end division_of_fractions_l2730_273092


namespace geometric_sequence_max_value_l2730_273074

theorem geometric_sequence_max_value (a b c d : ℝ) : 
  (∃ r : ℝ, a * r = b ∧ b * r = c ∧ c * r = d) →  -- geometric sequence condition
  (∀ x : ℝ, Real.log (x + 2) - x ≤ c) →           -- maximum value condition
  (Real.log (b + 2) - b = c) →                    -- maximum occurs at x = b
  a * d = -1 := by
sorry

end geometric_sequence_max_value_l2730_273074


namespace parabola_equation_theorem_l2730_273096

/-- Define a parabola by its focus and directrix -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

/-- The equation of a parabola in general form -/
def parabola_equation (a b c d e f : ℤ) (x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0

/-- The given parabola -/
def given_parabola : Parabola :=
  { focus := (4, 4),
    directrix := λ x y => 4 * x + 8 * y - 32 }

/-- Theorem stating the equation of the given parabola -/
theorem parabola_equation_theorem :
  ∃ (a b c d e f : ℤ),
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | parabola_equation a b c d e f p.1 p.2} ↔ 
      (x - given_parabola.focus.1)^2 + (y - given_parabola.focus.2)^2 = 
      (given_parabola.directrix x y)^2 / (4^2 + 8^2)) ∧
    a > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1 ∧
    a = 16 ∧ b = -64 ∧ c = 64 ∧ d = -128 ∧ e = -256 ∧ f = 768 := by
  sorry

end parabola_equation_theorem_l2730_273096


namespace percentage_prefer_x_is_zero_l2730_273040

def total_employees : ℕ := 200
def relocated_to_x : ℚ := 30 / 100
def relocated_to_y : ℚ := 70 / 100
def prefer_y : ℚ := 40 / 100
def max_satisfied : ℕ := 140

theorem percentage_prefer_x_is_zero :
  ∃ (prefer_x : ℚ),
    prefer_x ≥ 0 ∧
    prefer_x + prefer_y = 1 ∧
    (prefer_x * total_employees).floor + (prefer_y * total_employees).floor ≤ max_satisfied ∧
    prefer_x = 0 := by sorry

end percentage_prefer_x_is_zero_l2730_273040


namespace distance_between_points_l2730_273091

theorem distance_between_points (a : ℝ) : 
  let A : ℝ × ℝ := (a, -2)
  let B : ℝ × ℝ := (0, 3)
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 7^2) → (a = 2 * Real.sqrt 6 ∨ a = -2 * Real.sqrt 6) :=
by sorry

end distance_between_points_l2730_273091


namespace max_sum_of_factors_l2730_273079

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 360 →
  a + b + c + d ≤ 66 := by
sorry

end max_sum_of_factors_l2730_273079


namespace line_perp_plane_implies_planes_perp_l2730_273051

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : subset m α)
  (h4 : subset n β)
  (h5 : perp m β) :
  perpPlanes α β :=
sorry

end line_perp_plane_implies_planes_perp_l2730_273051


namespace completing_square_proof_l2730_273038

theorem completing_square_proof (x : ℝ) : 
  (x^2 - 8*x + 5 = 0) ↔ ((x - 4)^2 = 11) := by
  sorry

end completing_square_proof_l2730_273038


namespace two_out_of_three_probability_l2730_273024

-- Define the success rate of the basketball player
def success_rate : ℚ := 3 / 5

-- Define the number of shots taken
def total_shots : ℕ := 3

-- Define the number of successful shots we're interested in
def successful_shots : ℕ := 2

-- Theorem statement
theorem two_out_of_three_probability :
  (Nat.choose total_shots successful_shots : ℚ) * success_rate ^ successful_shots * (1 - success_rate) ^ (total_shots - successful_shots) = 54 / 125 := by
  sorry

end two_out_of_three_probability_l2730_273024


namespace polynomial_divisibility_l2730_273082

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
noncomputable def ω : ℂ := sorry

/-- The property that ω^2 + ω + 1 = 0 -/
axiom ω_property : ω^2 + ω + 1 = 0

/-- The polynomial x^104 + Ax^3 + Bx -/
def polynomial (A B : ℝ) (x : ℂ) : ℂ := x^104 + A * x^3 + B * x

/-- The divisibility condition -/
def is_divisible (A B : ℝ) : Prop :=
  polynomial A B ω = 0

theorem polynomial_divisibility (A B : ℝ) :
  is_divisible A B → A + B = 0 := by sorry

end polynomial_divisibility_l2730_273082


namespace sequence_decreasing_equivalence_l2730_273097

def IsDecreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) < a n

theorem sequence_decreasing_equivalence (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) ↔ IsDecreasing a :=
sorry

end sequence_decreasing_equivalence_l2730_273097


namespace circumscribed_sphere_surface_area_l2730_273071

/-- The surface area of a sphere circumscribing a right square prism -/
theorem circumscribed_sphere_surface_area (a h : ℝ) (ha : a = 2) (hh : h = 3) :
  let R := (1 / 2 : ℝ) * Real.sqrt (h^2 + 2 * a^2)
  4 * Real.pi * R^2 = 17 * Real.pi :=
sorry

end circumscribed_sphere_surface_area_l2730_273071


namespace sum_of_a_and_b_l2730_273054

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 31) 
  (eq2 : 4 * a + 3 * b = 35) : 
  a + b = 68 / 7 := by
sorry

end sum_of_a_and_b_l2730_273054


namespace walking_speed_calculation_l2730_273034

/-- Proves that given a distance that takes 2 hours 45 minutes to walk and 40 minutes to run at 16.5 kmph, the walking speed is 4 kmph. -/
theorem walking_speed_calculation (distance : ℝ) : 
  distance / (2 + 45 / 60) = 4 → distance / (40 / 60) = 16.5 → distance / (2 + 45 / 60) = 4 :=
by sorry

end walking_speed_calculation_l2730_273034


namespace frog_jump_probability_l2730_273010

/-- Represents a jump as a vector in 3D space -/
structure Jump where
  x : ℝ
  y : ℝ
  z : ℝ
  magnitude_is_one : x^2 + y^2 + z^2 = 1

/-- Represents the frog's position after a series of jumps -/
def FinalPosition (jumps : List Jump) : ℝ × ℝ × ℝ :=
  let sum := jumps.foldl (fun (ax, ay, az) j => (ax + j.x, ay + j.y, az + j.z)) (0, 0, 0)
  sum

/-- The probability of the frog's final position being exactly 1 meter from the start -/
noncomputable def probability_one_meter_away (num_jumps : ℕ) : ℝ :=
  sorry

/-- Theorem stating the probability for 4 jumps is 1/8 -/
theorem frog_jump_probability :
  probability_one_meter_away 4 = 1/8 := by sorry

end frog_jump_probability_l2730_273010


namespace enclosure_blocks_l2730_273047

/-- Calculates the number of blocks required for a rectangular enclosure --/
def blocks_required (length width height : ℕ) : ℕ :=
  let external_volume := length * width * height
  let internal_length := length - 2
  let internal_width := width - 2
  let internal_height := height - 2
  let internal_volume := internal_length * internal_width * internal_height
  external_volume - internal_volume

/-- Proves that the number of blocks required for the given dimensions is 598 --/
theorem enclosure_blocks : blocks_required 15 13 6 = 598 := by
  sorry

end enclosure_blocks_l2730_273047


namespace sum_lower_bound_l2730_273093

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end sum_lower_bound_l2730_273093


namespace smallest_bob_number_l2730_273045

def alice_number : ℕ := 24

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ),
    bob_number > 0 ∧
    has_all_prime_factors alice_number bob_number ∧
    (∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → bob_number ≤ m) ∧
    bob_number = 6 :=
sorry

end smallest_bob_number_l2730_273045


namespace inequality_holds_iff_l2730_273057

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end inequality_holds_iff_l2730_273057


namespace square_area_problem_l2730_273095

theorem square_area_problem : 
  ∀ x : ℚ, 
  (5 * x - 20 : ℚ) = (25 - 2 * x : ℚ) → 
  (5 * x - 20 : ℚ) > 0 →
  (5 * x - 20 : ℚ)^2 = 7225 / 49 := by
sorry

end square_area_problem_l2730_273095


namespace tangent_line_to_ln_l2730_273002

theorem tangent_line_to_ln (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 := by
  sorry

end tangent_line_to_ln_l2730_273002


namespace cube_number_sum_l2730_273018

theorem cube_number_sum :
  ∀ (a b c d e f : ℕ),
  -- The numbers are consecutive whole numbers between 15 and 20
  15 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f ≤ 20 →
  -- The sum of opposite faces is equal
  a + f = b + e ∧ b + e = c + d →
  -- The middle number in the range is the largest on one face
  (d = 18 ∨ c = 18) →
  -- The sum of all numbers is 105
  a + b + c + d + e + f = 105 :=
by sorry

end cube_number_sum_l2730_273018


namespace fraction_power_five_l2730_273084

theorem fraction_power_five : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end fraction_power_five_l2730_273084


namespace power_product_simplification_l2730_273087

theorem power_product_simplification :
  (5 / 3 : ℚ) ^ 2023 * (6 / 10 : ℚ) ^ 2022 = 5 / 3 := by sorry

end power_product_simplification_l2730_273087


namespace pen_count_l2730_273073

theorem pen_count (num_pencils : ℕ) (max_students : ℕ) (h1 : num_pencils = 910) (h2 : max_students = 91) 
  (h3 : max_students ∣ num_pencils) : 
  ∃ num_pens : ℕ, num_pens = num_pencils :=
by sorry

end pen_count_l2730_273073


namespace matrix_inverse_zero_if_singular_l2730_273037

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 8; -2, -4]

theorem matrix_inverse_zero_if_singular :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end matrix_inverse_zero_if_singular_l2730_273037


namespace work_time_proof_l2730_273053

/-- Represents the time taken by A to complete the work alone -/
def time_A : ℝ := 6

/-- Represents the time taken by B to complete the work alone -/
def time_B : ℝ := 8

/-- Represents the time taken by A, B, and C together to complete the work -/
def time_ABC : ℝ := 3

/-- Represents A's share of the payment -/
def share_A : ℝ := 300

/-- Represents B's share of the payment -/
def share_B : ℝ := 225

/-- Represents C's share of the payment -/
def share_C : ℝ := 75

/-- Represents the total payment for the work -/
def total_payment : ℝ := 600

theorem work_time_proof :
  (1 / time_A + 1 / time_B + (share_C / share_A) / time_A = 1 / time_ABC) ∧
  (share_A / share_B = 4 / 3) ∧
  (share_A / share_C = 4) ∧
  (share_A + share_B + share_C = total_payment) →
  time_A = 6 := by sorry

end work_time_proof_l2730_273053


namespace min_triangle_area_in_cube_l2730_273035

/-- Given a cube with edge length a, the minimum area of triangles formed by 
    intersections of a plane parallel to the base with specific lines is 7a²/32 -/
theorem min_triangle_area_in_cube (a : ℝ) (ha : a > 0) : 
  ∃ (S : ℝ), S = (7 * a^2) / 32 ∧ 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ a → 
    S ≤ (1/4) * |2*x^2 - 3*a*x + 2*a^2| := by
  sorry

end min_triangle_area_in_cube_l2730_273035


namespace sin_2alpha_in_terms_of_k_l2730_273026

theorem sin_2alpha_in_terms_of_k (k α : ℝ) (h : Real.cos (π / 4 - α) = k) :
  Real.sin (2 * α) = 2 * k^2 - 1 := by
  sorry

end sin_2alpha_in_terms_of_k_l2730_273026


namespace dans_stickers_l2730_273032

theorem dans_stickers (bob_stickers : ℕ) (tom_stickers : ℕ) (dan_stickers : ℕ)
  (h1 : bob_stickers = 12)
  (h2 : tom_stickers = 3 * bob_stickers)
  (h3 : dan_stickers = 2 * tom_stickers) :
  dan_stickers = 72 := by
sorry

end dans_stickers_l2730_273032


namespace sufficient_not_necessary_l2730_273062

theorem sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ (∃ a : ℝ, a ≤ 1 ∧ 1 / a < 1) := by sorry

end sufficient_not_necessary_l2730_273062


namespace complement_union_theorem_l2730_273004

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_union_theorem : 
  (U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end complement_union_theorem_l2730_273004


namespace cos_sin_pi_12_product_l2730_273003

theorem cos_sin_pi_12_product (π : Real) : 
  (Real.cos (π / 12) - Real.sin (π / 12)) * (Real.cos (π / 12) + Real.sin (π / 12)) = Real.sqrt 3 / 2 := by
  sorry

end cos_sin_pi_12_product_l2730_273003


namespace systematic_sampling_removal_l2730_273039

/-- The number of individuals to be removed from a population to make it divisible by a given sample size -/
def individualsToRemove (populationSize sampleSize : ℕ) : ℕ :=
  populationSize - sampleSize * (populationSize / sampleSize)

/-- Theorem stating that 4 individuals should be removed from a population of 3,204 to make it divisible by 80 -/
theorem systematic_sampling_removal :
  individualsToRemove 3204 80 = 4 := by
  sorry

end systematic_sampling_removal_l2730_273039


namespace expression_value_l2730_273036

theorem expression_value : 65 + (120 / 15) + (15 * 18) - 250 - (405 / 9) + 3^3 = 75 := by
  sorry

end expression_value_l2730_273036


namespace ice_pack_price_is_three_l2730_273015

/-- The price of a pack of 10 bags of ice for Chad's BBQ -/
def ice_pack_price (total_people : ℕ) (ice_per_person : ℕ) (bags_per_pack : ℕ) (total_spent : ℚ) : ℚ :=
  let total_ice := total_people * ice_per_person
  total_spent / (total_ice / bags_per_pack)

/-- Theorem: The price of a pack of 10 bags of ice is $3 -/
theorem ice_pack_price_is_three :
  ice_pack_price 15 2 10 9 = 3 := by
  sorry

#eval ice_pack_price 15 2 10 9

end ice_pack_price_is_three_l2730_273015


namespace linear_function_passes_through_point_l2730_273050

/-- A linear function y = kx - k (k ≠ 0) that passes through the point (-1, 4) also passes through the point (1, 0). -/
theorem linear_function_passes_through_point (k : ℝ) (hk : k ≠ 0) :
  (∃ y : ℝ, y = k * (-1) - k ∧ y = 4) →
  (∃ y : ℝ, y = k * 1 - k ∧ y = 0) :=
by sorry

end linear_function_passes_through_point_l2730_273050


namespace quadratic_inequality_properties_l2730_273094

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | x < -2 or x > 3}, prove the following statements -/
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧
  (a + b + c < 0) ∧
  (∀ x, c*x^2 - b*x + a < 0 ↔ x < -1/3 ∨ x > 1/2) :=
sorry

end quadratic_inequality_properties_l2730_273094
