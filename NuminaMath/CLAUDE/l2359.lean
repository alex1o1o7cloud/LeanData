import Mathlib

namespace distinct_prime_factors_of_divisor_sum_360_l2359_235948

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 360 is 4 -/
theorem distinct_prime_factors_of_divisor_sum_360 : 
  num_distinct_prime_factors (sum_of_divisors 360) = 4 := by sorry

end distinct_prime_factors_of_divisor_sum_360_l2359_235948


namespace unique_valid_statement_l2359_235945

theorem unique_valid_statement : ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
  (Real.sqrt (a^2 + b^2) = |a - b|) ∧
  ¬(Real.sqrt (a^2 + b^2) = a^2 - b^2) ∧
  ¬(Real.sqrt (a^2 + b^2) = a + b) ∧
  ¬(Real.sqrt (a^2 + b^2) = |a| + |b|) :=
by
  sorry

end unique_valid_statement_l2359_235945


namespace piggy_bank_pennies_l2359_235969

theorem piggy_bank_pennies (initial_pennies : ℕ) : 
  (12 * (initial_pennies + 6) = 96) → initial_pennies = 2 := by
  sorry

end piggy_bank_pennies_l2359_235969


namespace dino_money_theorem_l2359_235986

/-- Calculates the money Dino has left at the end of the month based on his work hours, rates, and expenses. -/
def dino_money_left (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem stating that Dino has $500 left at the end of the month. -/
theorem dino_money_theorem : dino_money_left 20 30 5 10 20 40 500 = 500 := by
  sorry

end dino_money_theorem_l2359_235986


namespace negation_of_proposition_negation_of_specific_proposition_l2359_235982

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l2359_235982


namespace complement_of_A_in_U_l2359_235952

def U : Set ℝ := {x | x ≥ 0}
def A : Set ℝ := {x | x ≥ 1}

theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end complement_of_A_in_U_l2359_235952


namespace largest_three_digit_perfect_square_diff_l2359_235955

/-- A function that returns the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a three-digit number. -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The main theorem stating that 919 is the largest three-digit number
    such that the number minus the sum of its digits is a perfect square. -/
theorem largest_three_digit_perfect_square_diff :
  ∀ n : ℕ, is_three_digit n →
    (∃ k : ℕ, n - sum_of_digits n = k^2) →
    n ≤ 919 := by sorry

end largest_three_digit_perfect_square_diff_l2359_235955


namespace complex_square_roots_l2359_235950

theorem complex_square_roots : 
  let z₁ : ℂ := Complex.mk (3 * Real.sqrt 2) (-55 * Real.sqrt 2 / 6)
  let z₂ : ℂ := Complex.mk (-3 * Real.sqrt 2) (55 * Real.sqrt 2 / 6)
  z₁^2 = Complex.mk (-121) (-110) ∧ z₂^2 = Complex.mk (-121) (-110) :=
by sorry

end complex_square_roots_l2359_235950


namespace min_value_of_exponential_sum_l2359_235995

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (z : ℝ), 3^x + 9^y ≥ z ∧ (∃ (a b : ℝ), a + 2 * b = 2 ∧ 3^a + 9^b = z) → m ≤ z :=
by sorry

end min_value_of_exponential_sum_l2359_235995


namespace orange_juice_distribution_l2359_235938

theorem orange_juice_distribution (pitcher_capacity : ℝ) (h : pitcher_capacity > 0) :
  let juice_volume := (2/3) * pitcher_capacity
  let num_cups := 8
  let juice_per_cup := juice_volume / num_cups
  juice_per_cup / pitcher_capacity = 1/12 := by
  sorry

end orange_juice_distribution_l2359_235938


namespace acute_triangle_sine_sum_l2359_235983

theorem acute_triangle_sine_sum (α β γ : Real) 
  (acute_triangle : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2)
  (angle_sum : α + β + γ = π) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end acute_triangle_sine_sum_l2359_235983


namespace compute_fraction_expression_l2359_235960

theorem compute_fraction_expression : 8 * (1/3)^2 * (2/7) = 16/63 := by
  sorry

end compute_fraction_expression_l2359_235960


namespace simplify_trig_expression_l2359_235977

theorem simplify_trig_expression :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end simplify_trig_expression_l2359_235977


namespace fair_coin_prob_heads_l2359_235914

-- Define a fair coin
def fair_coin : Type := Unit

-- Define the probability of landing heads for a fair coin
def prob_heads (c : fair_coin) : ℚ := 1 / 2

-- Define a sequence of coin tosses
def coin_tosses : ℕ → fair_coin
  | _ => ()

-- State the theorem
theorem fair_coin_prob_heads (n : ℕ) : 
  prob_heads (coin_tosses n) = 1 / 2 := by
  sorry

end fair_coin_prob_heads_l2359_235914


namespace subtracted_number_l2359_235926

theorem subtracted_number (x N : ℤ) (h1 : 3 * x = (N - x) + 16) (h2 : x = 13) : N = 36 := by
  sorry

end subtracted_number_l2359_235926


namespace marks_team_three_pointers_l2359_235946

/-- Represents the number of 3-pointers scored by Mark's team -/
def marks_three_pointers : ℕ := sorry

/-- The total points scored by both teams -/
def total_points : ℕ := 201

/-- The number of 2-pointers scored by Mark's team -/
def marks_two_pointers : ℕ := 25

/-- The number of free throws scored by Mark's team -/
def marks_free_throws : ℕ := 10

theorem marks_team_three_pointers :
  marks_three_pointers = 8 ∧
  (2 * marks_two_pointers + 3 * marks_three_pointers + marks_free_throws) +
  (2 * (2 * marks_two_pointers) + 3 * (marks_three_pointers / 2) + (marks_free_throws / 2)) = total_points :=
sorry

end marks_team_three_pointers_l2359_235946


namespace freelancer_earnings_l2359_235988

theorem freelancer_earnings (x : ℝ) : 
  x + (50 + 2*x) + 4*(x + (50 + 2*x)) = 5500 → x = 5300/15 := by
  sorry

end freelancer_earnings_l2359_235988


namespace square_difference_evaluation_l2359_235954

theorem square_difference_evaluation (c d : ℕ) (h1 : c = 5) (h2 : d = 3) :
  (c^2 + d)^2 - (c^2 - d)^2 = 300 := by
  sorry

end square_difference_evaluation_l2359_235954


namespace line_intersects_circle_l2359_235900

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- The point P -/
def P : ℝ × ℝ := (4, 0)

/-- The circle ⊙P -/
def circleP : Circle := { center := P, radius := 5 }

/-- The line y = kx + 2 -/
def line (k : ℝ) : Line := { k := k, b := 2 }

/-- Theorem: The line y = kx + 2 (k ≠ 0) always intersects the circle ⊙P -/
theorem line_intersects_circle (k : ℝ) (h : k ≠ 0) : 
  ∃ (x y : ℝ), (y = k * x + 2) ∧ 
  ((x - circleP.center.1)^2 + (y - circleP.center.2)^2 = circleP.radius^2) :=
sorry

end line_intersects_circle_l2359_235900


namespace walking_speed_solution_l2359_235947

/-- Represents the problem of finding A's walking speed -/
def walking_speed_problem (v : ℝ) : Prop :=
  let b_speed : ℝ := 20
  let time_diff : ℝ := 3
  let catch_up_distance : ℝ := 60
  let catch_up_time : ℝ := catch_up_distance / b_speed
  v * (time_diff + catch_up_time) = catch_up_distance ∧ v = 10

/-- Theorem stating that the solution to the walking speed problem is 10 kmph -/
theorem walking_speed_solution :
  ∃ v : ℝ, walking_speed_problem v :=
sorry

end walking_speed_solution_l2359_235947


namespace parabola_c_value_l2359_235993

/-- A parabola passing through two points -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_point_1 : 2 * 1^2 + b * 1 + c = 4
  pass_point_2 : 2 * 3^2 + b * 3 + c = 16

/-- The value of c for the parabola -/
def c_value (p : Parabola) : ℝ := p.c

theorem parabola_c_value (p : Parabola) : c_value p = 4 := by
  sorry

end parabola_c_value_l2359_235993


namespace quadratic_equation_no_real_roots_l2359_235971

theorem quadratic_equation_no_real_roots 
  (p q a b c : ℝ) 
  (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hpq : p ≠ q)
  (hgeom : a^2 = p*q)  -- Geometric sequence condition
  (harith1 : b - p = c - b) (harith2 : c - b = q - c)  -- Arithmetic sequence conditions
  : (2*a)^2 - 4*b*c < 0 := by
  sorry

#check quadratic_equation_no_real_roots

end quadratic_equation_no_real_roots_l2359_235971


namespace f_has_one_zero_in_interval_l2359_235967

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 7

-- State the theorem
theorem f_has_one_zero_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end f_has_one_zero_in_interval_l2359_235967


namespace derivative_ln_2x_squared_plus_1_l2359_235934

open Real

theorem derivative_ln_2x_squared_plus_1 (x : ℝ) :
  deriv (λ x => Real.log (2 * x^2 + 1)) x = (4 * x) / (2 * x^2 + 1) := by
  sorry

end derivative_ln_2x_squared_plus_1_l2359_235934


namespace cos_sum_eleventh_l2359_235970

theorem cos_sum_eleventh : 
  Real.cos (2 * Real.pi / 11) + Real.cos (6 * Real.pi / 11) + Real.cos (8 * Real.pi / 11) = 
    (-1 + Real.sqrt (-11)) / 4 := by
  sorry

end cos_sum_eleventh_l2359_235970


namespace third_number_value_l2359_235922

theorem third_number_value (a b c : ℝ) : 
  a + b + c = 500 →
  a = 200 →
  b = 2 * c →
  c = 100 := by
sorry

end third_number_value_l2359_235922


namespace gcd_power_two_minus_one_l2359_235930

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^1200 - 1) (2^1230 - 1) = 2^30 - 1 := by
  sorry

end gcd_power_two_minus_one_l2359_235930


namespace cost_price_calculation_l2359_235978

theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : marked_price = 200)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2) :
  ∃ (cost_price : ℝ),
    cost_price = 150 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by
  sorry

end cost_price_calculation_l2359_235978


namespace cool_function_periodic_l2359_235915

/-- A function is cool if there exist real numbers a and b such that
    f(x + a) is even and f(x + b) is odd. -/
def IsCool (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x, f (x + a) = f (-x + a)) ∧ (∀ x, f (x + b) = -f (-x + b))

/-- Every cool function is periodic. -/
theorem cool_function_periodic (f : ℝ → ℝ) (h : IsCool f) :
    ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
  sorry

end cool_function_periodic_l2359_235915


namespace function_roots_bound_l2359_235925

/-- The function f(x) defined with given parameters has no more than 14 positive roots -/
theorem function_roots_bound 
  (a b c d : ℝ) 
  (k l m p q r : ℕ) 
  (h1 : k ≥ l ∧ l ≥ m) 
  (h2 : p ≥ q ∧ q ≥ r) :
  let f : ℝ → ℝ := λ x => a*(x+1)^k * (x+2)^p + b*(x+1)^l * (x+2)^q + c*(x+1)^m * (x+2)^r - d
  ∃ (S : Finset ℝ), (∀ x ∈ S, x > 0 ∧ f x = 0) ∧ Finset.card S ≤ 14 := by
  sorry

end function_roots_bound_l2359_235925


namespace R_equals_eleven_l2359_235924

def F : ℝ := 2^121 - 1

def Q : ℕ := 120

theorem R_equals_eleven :
  Real.sqrt (Real.log (1 + F) / Real.log 2) = 11 := by sorry

end R_equals_eleven_l2359_235924


namespace system_solution_l2359_235957

theorem system_solution :
  let x : ℝ := -1
  let y : ℝ := (Real.sqrt 3 + 1) / 2
  (Real.sqrt 3 * x + 2 * y = 1) ∧ (x + 2 * y = Real.sqrt 3) := by
  sorry

end system_solution_l2359_235957


namespace distribute_five_into_four_l2359_235974

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else if k = 1 then 1
  else k

theorem distribute_five_into_four :
  distribute_objects 5 4 = 4 := by
  sorry

end distribute_five_into_four_l2359_235974


namespace arithmetic_progression_square_sum_l2359_235996

def is_four_identical_digits (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 9 ∧ n = k * 1111

theorem arithmetic_progression_square_sum (n : ℕ) : 
  is_four_identical_digits ((n - 2)^2 + n^2 + (n + 2)^2) ↔ n = 43 :=
sorry

end arithmetic_progression_square_sum_l2359_235996


namespace student_contribution_l2359_235994

theorem student_contribution
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : num_students = 19) :
  (total_contribution - class_funds) / num_students = 4 :=
by sorry

end student_contribution_l2359_235994


namespace fishing_theorem_l2359_235953

def fishing_problem (jordan_catch perry_catch alex_catch bird_steal release_fraction : ℕ) : ℕ :=
  let total_catch := jordan_catch + perry_catch + alex_catch
  let after_bird := total_catch - bird_steal
  let to_release := (after_bird * release_fraction) / 3
  after_bird - to_release

theorem fishing_theorem :
  fishing_problem 4 8 36 2 1 = 31 :=
by sorry

end fishing_theorem_l2359_235953


namespace sqrt3_div3_sufficient_sqrt3_div3_not_necessary_sqrt3_div3_sufficient_not_necessary_l2359_235927

/-- The condition for a line to be tangent to a circle --/
def is_tangent (k : ℝ) : Prop :=
  let line := fun x => k * (x + 2)
  let circle := fun x y => x^2 + y^2 = 1
  ∃ x y, circle x y ∧ y = line x ∧
  ∀ x' y', circle x' y' → (y' - line x')^2 ≥ 0

/-- k = √3/3 is sufficient for tangency --/
theorem sqrt3_div3_sufficient :
  is_tangent (Real.sqrt 3 / 3) := by sorry

/-- k = √3/3 is not necessary for tangency --/
theorem sqrt3_div3_not_necessary :
  ∃ k, k ≠ Real.sqrt 3 / 3 ∧ is_tangent k := by sorry

/-- k = √3/3 is a sufficient but not necessary condition for tangency --/
theorem sqrt3_div3_sufficient_not_necessary :
  (is_tangent (Real.sqrt 3 / 3)) ∧
  (∃ k, k ≠ Real.sqrt 3 / 3 ∧ is_tangent k) := by sorry

end sqrt3_div3_sufficient_sqrt3_div3_not_necessary_sqrt3_div3_sufficient_not_necessary_l2359_235927


namespace quadratic_root_difference_l2359_235906

theorem quadratic_root_difference (a b c : ℝ) (h : a ≠ 0) :
  let eq := fun x => a * x^2 + b * x + c
  let r1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  eq 1 + 40 + 300 = -64 →
  |r1 - r2| = 12 := by
sorry

end quadratic_root_difference_l2359_235906


namespace angle_calculation_l2359_235929

-- Define the triangles and angles
def Triangle (a b c : ℝ) := a + b + c = 180

-- Theorem statement
theorem angle_calculation (T1_angle1 T1_angle2 T2_angle1 T2_angle2 α β : ℝ) 
  (h1 : Triangle T1_angle1 T1_angle2 (180 - α))
  (h2 : Triangle T2_angle1 T2_angle2 β)
  (h3 : T1_angle1 = 70)
  (h4 : T1_angle2 = 50)
  (h5 : T2_angle1 = 45)
  (h6 : T2_angle2 = 50) :
  α = 120 ∧ β = 85 := by
  sorry

end angle_calculation_l2359_235929


namespace arithmetic_sequence_property_l2359_235931

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 4 + a 10 + a 16 = 30) : a 18 - 2 * a 14 = -10 := by
  sorry

end arithmetic_sequence_property_l2359_235931


namespace parabola_directrix_l2359_235985

/-- A parabola with equation x = -1/4 * y^2 has a directrix with equation x = 1 -/
theorem parabola_directrix (y : ℝ) : 
  (∃ x : ℝ, x = -(1/4) * y^2) → 
  (∃ d : ℝ, d = 1 ∧ ∀ p : ℝ × ℝ, p.1 = d ↔ p ∈ {q : ℝ × ℝ | q.1 = 1}) :=
by sorry

end parabola_directrix_l2359_235985


namespace stratified_sampling_count_l2359_235976

-- Define the total number of students and their gender distribution
def total_students : ℕ := 60
def female_students : ℕ := 24
def male_students : ℕ := 36

-- Define the number of students to be selected
def selected_students : ℕ := 20

-- Define the number of female and male students to be selected
def selected_female : ℕ := 8
def selected_male : ℕ := 12

-- Theorem statement
theorem stratified_sampling_count :
  (Nat.choose female_students selected_female) * (Nat.choose male_students selected_male) =
  (Nat.choose female_students selected_female) * (Nat.choose male_students selected_male) := by
  sorry

-- Ensure the conditions are met
axiom total_students_sum : female_students + male_students = total_students
axiom selected_students_sum : selected_female + selected_male = selected_students

end stratified_sampling_count_l2359_235976


namespace ellipse_properties_l2359_235999

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define perpendicular rays from origin
def perpendicular_rays (m1 m2 n1 n2 : ℝ) : Prop :=
  m1 * n1 + m2 * n2 = 0

-- Define points M and N on the ellipse
def points_on_ellipse (m1 m2 n1 n2 : ℝ) : Prop :=
  ellipse m1 m2 ∧ ellipse n1 n2

-- Theorem statement
theorem ellipse_properties :
  ∀ (m1 m2 n1 n2 : ℝ),
  perpendicular_rays m1 m2 n1 n2 →
  points_on_ellipse m1 m2 n1 n2 →
  (∃ (e : ℝ), e = 1/2 ∧ e = Real.sqrt (1 - 3/4)) ∧
  (∃ (d : ℝ), d = 2 * Real.sqrt 21 / 7 ∧
    ∀ (k b : ℝ), (m2 = k * m1 + b ∧ n2 = k * n1 + b) →
      d = |b| / Real.sqrt (k^2 + 1)) :=
sorry

end ellipse_properties_l2359_235999


namespace factory_underpayment_l2359_235917

/-- The hourly wage in yuan -/
def hourly_wage : ℚ := 6

/-- The nominal work day duration in hours -/
def nominal_work_day : ℚ := 8

/-- The time for clock hands to coincide in the inaccurate clock (in minutes) -/
def inaccurate_coincidence_time : ℚ := 69

/-- The time for clock hands to coincide in an accurate clock (in minutes) -/
def accurate_coincidence_time : ℚ := 720 / 11

/-- Calculate the actual work time based on the inaccurate clock -/
def actual_work_time : ℚ :=
  (inaccurate_coincidence_time * nominal_work_day) / accurate_coincidence_time

/-- Calculate the underpayment amount -/
def underpayment : ℚ := hourly_wage * (actual_work_time - nominal_work_day)

theorem factory_underpayment :
  underpayment = 13/5 :=
by sorry

end factory_underpayment_l2359_235917


namespace circle_ratio_problem_l2359_235907

theorem circle_ratio_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 3 * (π * a^2) → a / b = 1 / 2 := by
  sorry

end circle_ratio_problem_l2359_235907


namespace complex_fraction_simplification_l2359_235928

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 + i) / (3 - i) = (1 + 2*i) / 5 := by sorry

end complex_fraction_simplification_l2359_235928


namespace equidistant_function_property_l2359_235932

open Complex

theorem equidistant_function_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ z : ℂ, abs ((a + b * I) * z - z) = abs ((a + b * I) * z)) →
  abs (a + b * I) = 5 →
  b^2 = 99/4 := by sorry

end equidistant_function_property_l2359_235932


namespace chocolate_box_problem_l2359_235981

theorem chocolate_box_problem (C : ℝ) : 
  C > 0 →  -- Ensure the number of chocolates is positive
  (C / 2 - 0.8 * (C / 2)) + (C / 2 - 0.5 * (C / 2)) = 28 →
  C = 80 := by
sorry

end chocolate_box_problem_l2359_235981


namespace well_depth_l2359_235940

/-- The depth of a well given specific conditions -/
theorem well_depth : 
  -- Define the distance function
  let distance (t : ℝ) : ℝ := 16 * t^2
  -- Define the speed of sound
  let sound_speed : ℝ := 1120
  -- Define the total time
  let total_time : ℝ := 7.7
  -- Define the depth of the well
  let depth : ℝ := distance (total_time - depth / sound_speed)
  -- Prove that the depth is 784 feet
  depth = 784 := by sorry

end well_depth_l2359_235940


namespace delta_y_over_delta_x_l2359_235910

/-- Given a function f(x) = -x² + x and two points on its graph,
    prove that Δy/Δx = 3 - Δx -/
theorem delta_y_over_delta_x (f : ℝ → ℝ) (Δx Δy : ℝ) :
  (∀ x, f x = -x^2 + x) →
  f (-1) = -2 →
  f (-1 + Δx) = -2 + Δy →
  Δx ≠ 0 →
  Δy / Δx = 3 - Δx :=
by sorry

end delta_y_over_delta_x_l2359_235910


namespace quadratic_equation_properties_l2359_235991

/-- Given a quadratic equation x^2 - (k+3)x + 2k + 2 = 0, prove:
    1. The equation always has two real roots
    2. When one root is positive and less than 1, -1 < k < 0 -/
theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (k+3)*x + 2*k + 2
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 → -1 < k ∧ k < 0) :=
by sorry

end quadratic_equation_properties_l2359_235991


namespace max_entropy_is_n_minus_two_l2359_235936

/-- A configuration of children around a circular table -/
structure Configuration (n : ℕ) :=
  (boys : Fin n → Bool)
  (girls : Fin n → Bool)
  (valid : ∀ i, boys i ≠ girls i)

/-- The entropy of a configuration -/
def entropy (n : ℕ) (config : Configuration n) : ℕ :=
  sorry

/-- Theorem: The maximal entropy for any configuration is n-2 when n > 3 -/
theorem max_entropy_is_n_minus_two (n : ℕ) (h : n > 3) :
  ∃ (config : Configuration n), entropy n config = n - 2 ∧
  ∀ (other : Configuration n), entropy n other ≤ n - 2 :=
sorry

end max_entropy_is_n_minus_two_l2359_235936


namespace sum_of_reciprocal_relations_l2359_235912

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by sorry

end sum_of_reciprocal_relations_l2359_235912


namespace cassidy_poster_count_l2359_235939

/-- Represents Cassidy's poster collection over time -/
structure PosterCollection where
  initial : Nat  -- Initial number of posters 3 years ago
  lost : Nat     -- Number of posters lost
  sold : Nat     -- Number of posters sold
  future : Nat   -- Number of posters to be added this summer

/-- Calculates the current number of posters in Cassidy's collection -/
def currentPosters (c : PosterCollection) : Nat :=
  2 * c.initial - 6

theorem cassidy_poster_count (c : PosterCollection) 
  (h1 : c.initial = 18)
  (h2 : c.lost = 2)
  (h3 : c.sold = 5)
  (h4 : c.future = 6) :
  currentPosters c = 30 := by
  sorry

#eval currentPosters { initial := 18, lost := 2, sold := 5, future := 6 }

end cassidy_poster_count_l2359_235939


namespace complement_intersection_problem_l2359_235989

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 5, 6}
def B : Set Nat := {3, 5}

theorem complement_intersection_problem : (U \ A) ∩ B = {3} := by
  sorry

end complement_intersection_problem_l2359_235989


namespace ted_green_mushrooms_l2359_235951

/-- The number of green mushrooms Ted gathered -/
def green_mushrooms : ℕ := sorry

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := 6

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The total number of white-spotted mushrooms gathered -/
def total_white_spotted : ℕ := 17

theorem ted_green_mushrooms :
  green_mushrooms = 0 :=
by
  sorry

end ted_green_mushrooms_l2359_235951


namespace MNP_collinear_tangent_equals_PA_l2359_235987

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def S : Circle := sorry
def S₁ : Circle := sorry
def A : Point := sorry
def B : Point := sorry
def M : Point := sorry
def N : Point := sorry
def P : Point := sorry

-- Define the conditions
axiom chord_divides_circle : sorry
axiom S₁_touches_AB_at_M : sorry
axiom S₁_touches_arc_at_N : sorry
axiom P_is_midpoint_of_other_arc : sorry

-- Define helper functions
def collinear (p q r : Point) : Prop := sorry
def tangent_length (p : Point) (c : Circle) : ℝ := sorry
def distance (p q : Point) : ℝ := sorry

-- State the theorems to be proved
theorem MNP_collinear : collinear M N P := sorry

theorem tangent_equals_PA : tangent_length P S₁ = distance P A := sorry

end MNP_collinear_tangent_equals_PA_l2359_235987


namespace max_value_on_curve_l2359_235949

noncomputable def max_value (b : ℝ) : ℝ :=
  if 0 < b ∧ b ≤ 4 then b^2 / 4 + 4 else 2 * b

theorem max_value_on_curve (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 → x^2 + 2*y ≤ max_value b) ∧
  (∃ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 ∧ x^2 + 2*y = max_value b) :=
sorry

end max_value_on_curve_l2359_235949


namespace det_E_l2359_235942

/-- A 3x3 matrix representing a dilation centered at the origin with scale factor 4 -/
def E : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ ↦ 4)

/-- Theorem: The determinant of E is 64 -/
theorem det_E : Matrix.det E = 64 := by
  sorry

end det_E_l2359_235942


namespace lyle_friends_served_l2359_235901

/-- Calculates the maximum number of friends who can have a sandwich and a juice pack. -/
def max_friends_served (sandwich_cost juice_cost total_money : ℚ) : ℕ :=
  let cost_per_person := sandwich_cost + juice_cost
  let total_servings := (total_money / cost_per_person).floor
  (total_servings - 1).natAbs

/-- Proves that Lyle can buy a sandwich and a juice pack for 4 friends. -/
theorem lyle_friends_served :
  max_friends_served 0.30 0.20 2.50 = 4 := by
  sorry

#eval max_friends_served 0.30 0.20 2.50

end lyle_friends_served_l2359_235901


namespace min_boxes_to_eliminate_l2359_235921

/-- Represents the game setup with total boxes and valuable boxes -/
structure GameSetup :=
  (total_boxes : ℕ)
  (valuable_boxes : ℕ)

/-- Calculates the probability of holding a valuable box -/
def probability (setup : GameSetup) (eliminated : ℕ) : ℚ :=
  setup.valuable_boxes / (setup.total_boxes - eliminated)

/-- Theorem stating the minimum number of boxes to eliminate -/
theorem min_boxes_to_eliminate (setup : GameSetup) 
  (h1 : setup.total_boxes = 30)
  (h2 : setup.valuable_boxes = 9) :
  ∃ (n : ℕ), 
    (n = 3) ∧ 
    (probability setup n ≥ 1/3) ∧ 
    (∀ m : ℕ, m < n → probability setup m < 1/3) :=
sorry

end min_boxes_to_eliminate_l2359_235921


namespace jeff_swimming_laps_l2359_235943

/-- The number of laps Jeff swam on Saturday -/
def saturday_laps : ℕ := 27

/-- The number of laps Jeff swam on Sunday morning -/
def sunday_morning_laps : ℕ := 15

/-- The number of laps Jeff had remaining after the break -/
def remaining_laps : ℕ := 56

/-- The total number of laps Jeff's coach required him to swim over the weekend -/
def total_required_laps : ℕ := saturday_laps + sunday_morning_laps + remaining_laps

theorem jeff_swimming_laps : total_required_laps = 98 := by
  sorry

end jeff_swimming_laps_l2359_235943


namespace basketball_game_result_l2359_235919

/-- Represents a basketball player with their score and penalties -/
structure Player where
  score : ℕ
  penalties : List ℕ

/-- Calculates the total points for a player after applying penalties -/
def playerPoints (p : Player) : ℤ :=
  p.score - (List.sum p.penalties)

/-- Calculates the total points for a team -/
def teamPoints (team : List Player) : ℤ :=
  List.sum (team.map playerPoints)

theorem basketball_game_result :
  let team_a := [
    Player.mk 12 [1, 2],
    Player.mk 18 [1, 2, 3],
    Player.mk 5 [],
    Player.mk 7 [1, 2],
    Player.mk 6 [1]
  ]
  let team_b := [
    Player.mk 10 [1, 2],
    Player.mk 9 [1],
    Player.mk 12 [],
    Player.mk 8 [1, 2, 3],
    Player.mk 5 [1, 2],
    Player.mk 4 [1]
  ]
  teamPoints team_a - teamPoints team_b = 1 := by
  sorry


end basketball_game_result_l2359_235919


namespace expression_simplification_l2359_235990

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 + 1 / (x^2 - 1)) / (x^2 / (x^2 + 2*x + 1)) = 1 + Real.sqrt 2 := by
  sorry

end expression_simplification_l2359_235990


namespace farm_animal_ratio_l2359_235975

/-- Proves that the initial ratio of horses to cows is 4:1 given the problem conditions --/
theorem farm_animal_ratio :
  ∀ (h c : ℕ),  -- Initial number of horses and cows
  (h - 15 : ℚ) / (c + 15 : ℚ) = 13 / 7 →  -- Ratio after transaction
  h - 15 = c + 15 + 30 →  -- Difference after transaction
  h / c = 4 / 1 := by
sorry

end farm_animal_ratio_l2359_235975


namespace shaded_region_perimeter_l2359_235965

/-- Given a circle with center O and radius 8, where the shaded region is half of the circle plus two radii, 
    the perimeter of the shaded region is 16 + 8π. -/
theorem shaded_region_perimeter (O : Point) (r : ℝ) (h1 : r = 8) : 
  let perimeter := 2 * r + (π * r)
  perimeter = 16 + 8 * π := by sorry

end shaded_region_perimeter_l2359_235965


namespace rings_arrangement_count_l2359_235984

def rings : ℕ := 10
def fingers : ℕ := 5
def rings_to_arrange : ℕ := 6

def arrange_rings (total_rings : ℕ) (fingers : ℕ) (rings_to_arrange : ℕ) : ℕ :=
  (total_rings.choose rings_to_arrange) * fingers * (rings_to_arrange.factorial)

theorem rings_arrangement_count :
  arrange_rings rings fingers rings_to_arrange = 756000 := by
  sorry

end rings_arrangement_count_l2359_235984


namespace min_value_of_f_l2359_235980

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 3) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y ∧ f a x = -29 :=
by sorry

end min_value_of_f_l2359_235980


namespace bicycle_price_increase_l2359_235935

theorem bicycle_price_increase (original_price new_price : ℝ) 
  (h1 : original_price = 220)
  (h2 : new_price = 253) :
  (new_price - original_price) / original_price * 100 = 15 := by
  sorry

end bicycle_price_increase_l2359_235935


namespace triangle_side_equation_l2359_235998

/-- Given a triangle ABC with vertex A at (1,4), and angle bisectors of B and C
    represented by the equations x + y - 1 = 0 and x - 2y = 0 respectively,
    the side BC lies on the line with equation 4x + 17y + 12 = 0. -/
theorem triangle_side_equation (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (1, 4)
  let angle_bisector_B : ℝ → ℝ → Prop := λ x y => x + y - 1 = 0
  let angle_bisector_C : ℝ → ℝ → Prop := λ x y => x - 2*y = 0
  let line_BC : ℝ → ℝ → Prop := λ x y => 4*x + 17*y + 12 = 0
  (∀ x y, x = B.1 ∧ y = B.2 → angle_bisector_B x y) →
  (∀ x y, x = C.1 ∧ y = C.2 → angle_bisector_C x y) →
  (∀ x y, (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2) → line_BC x y) :=
by sorry

end triangle_side_equation_l2359_235998


namespace largest_n_for_product_l2359_235908

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ := fun n => a₁ + (n - 1) * d

theorem largest_n_for_product (x y : ℤ) (hxy : x < y) :
  let a := ArithmeticSequence 2 x
  let b := ArithmeticSequence 3 y
  (∃ n : ℕ, a n * b n = 1638) →
  (∀ m : ℕ, a m * b m = 1638 → m ≤ 35) ∧
  (a 35 * b 35 = 1638) :=
sorry

end largest_n_for_product_l2359_235908


namespace range_of_x_l2359_235992

theorem range_of_x (x : ℝ) : 
  (1 / x < 3) → (1 / x > -2) → (2 * x - 5 > 0) → (x > 5 / 2) := by
  sorry

end range_of_x_l2359_235992


namespace red_jellybeans_count_l2359_235973

def total_jellybeans : ℕ := 200
def blue_jellybeans : ℕ := 14
def purple_jellybeans : ℕ := 26
def orange_jellybeans : ℕ := 40

theorem red_jellybeans_count :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 := by
  sorry

end red_jellybeans_count_l2359_235973


namespace absolute_value_inequality_solution_l2359_235956

theorem absolute_value_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) → a = -3 :=
by sorry

end absolute_value_inequality_solution_l2359_235956


namespace expression_bounds_l2359_235961

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 16 :=
by sorry

end expression_bounds_l2359_235961


namespace exponent_sum_l2359_235997

theorem exponent_sum (x a b c : ℝ) (h1 : x ≠ 1) (h2 : x * x^a * x^b * x^c = x^2024) : 
  a + b + c = 2023 := by
sorry

end exponent_sum_l2359_235997


namespace line_equation_for_triangle_l2359_235944

/-- Given a line passing through (-a, 0) and cutting a triangle with area T in the second quadrant,
    prove that the equation of the line is 2Tx - a²y + 2aT = 0 --/
theorem line_equation_for_triangle (a T : ℝ) (h_a : a > 0) (h_T : T > 0) :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → (x = -a ∧ y = 0) ∨ (x ≥ 0 ∧ y ≥ 0)) ∧ 
    (1/2 * a * (b : ℝ) = T) ∧
    (∀ x y : ℝ, y = m * x + b ↔ 2 * T * x - a^2 * y + 2 * a * T = 0) :=
by sorry

end line_equation_for_triangle_l2359_235944


namespace smallest_prime_divisor_of_6_12_plus_5_13_l2359_235972

theorem smallest_prime_divisor_of_6_12_plus_5_13 :
  (Nat.minFac (6^12 + 5^13) = 5) := by sorry

end smallest_prime_divisor_of_6_12_plus_5_13_l2359_235972


namespace expression_evaluation_l2359_235905

theorem expression_evaluation : 7^2 - 4^2 + 2*5 - 3^3 = 16 := by
  sorry

end expression_evaluation_l2359_235905


namespace race_distance_l2359_235920

theorem race_distance (total : ℝ) (selena : ℝ) (josh : ℝ) 
  (h1 : total = 36)
  (h2 : selena + josh = total)
  (h3 : josh = selena / 2) : 
  selena = 24 := by
sorry

end race_distance_l2359_235920


namespace x_is_perfect_square_l2359_235979

/-- The sequence x_n as defined in the problem -/
def x : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | (n + 4) => ((n^2 + n + 1) * (n + 1) / n) * x (n + 3) + 
               (n^2 + n + 1) * x (n + 2) - 
               ((n + 1) / n) * x (n + 1)

/-- The theorem stating that all members of x_n are perfect squares -/
theorem x_is_perfect_square : ∀ n : ℕ, ∃ y : ℤ, x n = (y : ℚ)^2 := by
  sorry

end x_is_perfect_square_l2359_235979


namespace subset_condition_l2359_235909

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem subset_condition (m : ℝ) : B m ⊆ A → -1 ≤ m ∧ m ≤ 4 := by
  sorry

end subset_condition_l2359_235909


namespace carmela_difference_l2359_235933

def cecil_money : ℕ := 600
def catherine_money : ℕ := 2 * cecil_money - 250
def total_money : ℕ := 2800

theorem carmela_difference : ℕ := by
  have h1 : cecil_money + catherine_money + (2 * cecil_money + (total_money - (cecil_money + catherine_money))) = total_money := by sorry
  have h2 : total_money - (cecil_money + catherine_money) = 50 := by sorry
  exact 50

#check carmela_difference

end carmela_difference_l2359_235933


namespace ellipse_properties_l2359_235937

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the semi-major axis, semi-minor axis, and semi-focal distance
def semi_major_axis : ℝ := 5
def semi_minor_axis : ℝ := 4
def semi_focal_distance : ℝ := 3

-- Theorem statement
theorem ellipse_properties :
  (∀ x y : ℝ, ellipse_equation x y) →
  semi_major_axis = 5 ∧ semi_minor_axis = 4 ∧ semi_focal_distance = 3 :=
by sorry

end ellipse_properties_l2359_235937


namespace quadratic_function_values_l2359_235941

def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem quadratic_function_values (a b : ℝ) (h : a ≠ 0) :
  (∀ x ∈ Set.Icc 2 3, f a b x ≤ 5) ∧
  (∃ x ∈ Set.Icc 2 3, f a b x = 5) ∧
  (∀ x ∈ Set.Icc 2 3, f a b x ≥ 2) ∧
  (∃ x ∈ Set.Icc 2 3, f a b x = 2) →
  ((a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3)) :=
sorry

end quadratic_function_values_l2359_235941


namespace domino_set_0_to_12_l2359_235959

/-- The number of tiles in a domino set with values from 0 to n -/
def dominoCount (n : ℕ) : ℕ := Nat.choose (n + 1) 2

/-- The number of tiles in a standard domino set (0 to 6) -/
def standardDominoCount : ℕ := 28

theorem domino_set_0_to_12 : dominoCount 12 = 91 := by sorry

end domino_set_0_to_12_l2359_235959


namespace total_cost_is_56_15_l2359_235962

-- Define the prices and quantities
def spam_price : ℚ := 3
def peanut_butter_price : ℚ := 5
def bread_price : ℚ := 2
def spam_quantity : ℕ := 12
def peanut_butter_quantity : ℕ := 3
def bread_quantity : ℕ := 4

-- Define the discount and tax rates
def spam_discount : ℚ := 0.1
def peanut_butter_tax : ℚ := 0.05

-- Define the total cost function
def total_cost : ℚ :=
  (spam_price * spam_quantity * (1 - spam_discount)) +
  (peanut_butter_price * peanut_butter_quantity * (1 + peanut_butter_tax)) +
  (bread_price * bread_quantity)

-- Theorem statement
theorem total_cost_is_56_15 : total_cost = 56.15 := by
  sorry

end total_cost_is_56_15_l2359_235962


namespace rightmost_three_digits_of_7_to_1987_l2359_235904

theorem rightmost_three_digits_of_7_to_1987 :
  7^1987 ≡ 643 [MOD 1000] := by
  sorry

end rightmost_three_digits_of_7_to_1987_l2359_235904


namespace smallest_m_is_251_l2359_235968

/-- Represents a circular arrangement of grids with real numbers -/
def CircularGrids (n : ℕ) := Fin n → ℝ

/-- Checks if the difference condition is satisfied for a given grid and step -/
def satisfiesDifferenceCondition (grids : CircularGrids 999) (a k : Fin 999) : Prop :=
  (grids a - grids ((a + k) % 999) = k) ∨ (grids a - grids ((999 + a - k) % 999) = k)

/-- Checks if the consecutive condition is satisfied for a given starting grid -/
def satisfiesConsecutiveCondition (grids : CircularGrids 999) (s : Fin 999) : Prop :=
  (∀ k : Fin 998, grids ((s + k) % 999) = grids s + k) ∨
  (∀ k : Fin 998, grids ((999 + s - k) % 999) = grids s + k)

/-- The main theorem stating that 251 is the smallest positive integer satisfying the conditions -/
theorem smallest_m_is_251 : 
  ∀ m : ℕ+, 
    (m = 251 ↔ 
      (∀ grids : CircularGrids 999, 
        (∀ a : Fin 999, ∀ k : Fin m, satisfiesDifferenceCondition grids a k) →
        (∃ s : Fin 999, satisfiesConsecutiveCondition grids s)) ∧
      (∀ m' : ℕ+, m' < m →
        ∃ grids : CircularGrids 999, 
          (∀ a : Fin 999, ∀ k : Fin m', satisfiesDifferenceCondition grids a k) ∧
          (∀ s : Fin 999, ¬satisfiesConsecutiveCondition grids s))) :=
sorry

end smallest_m_is_251_l2359_235968


namespace dima_wins_l2359_235963

-- Define the game board as a set of integers from 1 to 100
def GameBoard : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100}

-- Define a type for player strategies
def Strategy := GameBoard → ℕ

-- Define the winning condition for Mitya
def MityaWins (a b : ℕ) : Prop := (a + b) % 7 = 0

-- Define the game result
inductive GameResult
| MityaVictory
| DimaVictory

-- Define the game play function
def playGame (mityaStrategy dimaStrategy : Strategy) : GameResult :=
  sorry -- Actual game logic would go here

-- Theorem statement
theorem dima_wins :
  ∃ (dimaStrategy : Strategy),
    ∀ (mityaStrategy : Strategy),
      playGame mityaStrategy dimaStrategy = GameResult.DimaVictory :=
sorry

end dima_wins_l2359_235963


namespace sin_product_45_deg_l2359_235911

theorem sin_product_45_deg (α β : Real) 
  (h1 : Real.sin (α + β) = 0.2) 
  (h2 : Real.cos (α - β) = 0.3) : 
  Real.sin (α + Real.pi/4) * Real.sin (β + Real.pi/4) = 0.25 := by
  sorry

end sin_product_45_deg_l2359_235911


namespace partnership_profit_l2359_235923

/-- Calculates the total profit of a partnership given the investments, time periods, and one partner's profit. -/
def total_profit (a_investment : ℕ) (b_investment : ℕ) (a_period : ℕ) (b_period : ℕ) (b_profit : ℕ) : ℕ :=
  let profit_ratio := (a_investment * a_period) / (b_investment * b_period)
  let total_parts := profit_ratio + 1
  total_parts * b_profit

/-- Theorem stating that under the given conditions, the total profit is 42000. -/
theorem partnership_profit : 
  ∀ (b_investment : ℕ) (b_period : ℕ),
    b_investment > 0 → b_period > 0 →
    total_profit (3 * b_investment) b_investment (2 * b_period) b_period 6000 = 42000 :=
by sorry

end partnership_profit_l2359_235923


namespace f_monotone_decreasing_iff_a_in_range_l2359_235916

/-- Piecewise function f(x) defined by parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5*a - 4)*x + 7*a - 3 else (2*a - 1)^x

/-- The range of a for which f is monotonically decreasing -/
def a_range : Set ℝ := Set.Icc (3/5) (4/5)

/-- Theorem stating that f is monotonically decreasing iff a is in the specified range -/
theorem f_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) ↔ a ∈ a_range :=
sorry

end f_monotone_decreasing_iff_a_in_range_l2359_235916


namespace zachary_did_more_pushups_l2359_235958

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference in push-ups between Zachary and David -/
def pushup_difference : ℕ := zachary_pushups - david_pushups

theorem zachary_did_more_pushups : pushup_difference = 7 := by
  sorry

end zachary_did_more_pushups_l2359_235958


namespace profit_to_cost_ratio_l2359_235966

/-- Given an article with a sale price and cost price, this theorem proves
    that if the ratio of sale price to cost price is 6:2,
    then the ratio of profit to cost price is 2:1. -/
theorem profit_to_cost_ratio
  (sale_price cost_price : ℚ)
  (h : sale_price / cost_price = 6 / 2) :
  (sale_price - cost_price) / cost_price = 2 / 1 := by
  sorry

end profit_to_cost_ratio_l2359_235966


namespace pie_crust_flour_calculation_l2359_235903

/-- Given that 40 smaller pie crusts each use 1/8 cup of flour,
    prove that 25 larger pie crusts using the same total amount of flour
    will each require 1/5 cup of flour. -/
theorem pie_crust_flour_calculation (small_crusts : ℕ) (large_crusts : ℕ)
  (small_flour : ℚ) (large_flour : ℚ) :
  small_crusts = 40 →
  large_crusts = 25 →
  small_flour = 1/8 →
  small_crusts * small_flour = large_crusts * large_flour →
  large_flour = 1/5 := by
sorry

end pie_crust_flour_calculation_l2359_235903


namespace imaginary_part_of_complex_fraction_l2359_235913

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 - 4*i) / (1 - i)
  (z.im : ℝ) = -1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l2359_235913


namespace number_puzzle_l2359_235964

theorem number_puzzle (x : ℝ) : (x - 26) / 2 = 37 → 48 - x / 4 = 23 := by
  sorry

end number_puzzle_l2359_235964


namespace cubic_function_properties_l2359_235918

/-- The cubic function f(x) = x^3 - 12x + 12 -/
def f (x : ℝ) : ℝ := x^3 - 12*x + 12

theorem cubic_function_properties :
  (∃ x : ℝ, f x = 28) ∧  -- Maximum value is 28
  (f 2 = -4) ∧           -- Extreme value at x = 2 is -4
  (∀ x ∈ Set.Icc (-3) 3, f x ≥ -4) ∧  -- Minimum value on [-3, 3] is -4
  (∃ x ∈ Set.Icc (-3) 3, f x = -4) -- The minimum is attained on [-3, 3]
  := by sorry

end cubic_function_properties_l2359_235918


namespace binomial_sum_one_l2359_235902

theorem binomial_sum_one (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (a*x - 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ = 80 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 := by
sorry

end binomial_sum_one_l2359_235902
