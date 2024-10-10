import Mathlib

namespace constant_in_toll_formula_l3709_370978

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  constant + 0.50 * (x - 2 : ℝ)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 9

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 5

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧
    constant = 1.50 := by sorry

end constant_in_toll_formula_l3709_370978


namespace sequence_equality_l3709_370918

theorem sequence_equality (a b : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n : ℕ, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n : ℕ, a n > 0) :
  a 1 = b 1 := by
  sorry

end sequence_equality_l3709_370918


namespace medium_revenue_is_24_l3709_370913

/-- Represents the revenue from Tonya's lemonade stand -/
structure LemonadeRevenue where
  total : ℕ
  small : ℕ
  large_cups : ℕ
  small_price : ℕ
  medium_price : ℕ
  large_price : ℕ

/-- Calculates the revenue from medium lemonades -/
def medium_revenue (r : LemonadeRevenue) : ℕ :=
  r.total - r.small - (r.large_cups * r.large_price)

/-- Theorem: The revenue from medium lemonades is 24 -/
theorem medium_revenue_is_24 (r : LemonadeRevenue) 
  (h1 : r.total = 50)
  (h2 : r.small = 11)
  (h3 : r.large_cups = 5)
  (h4 : r.small_price = 1)
  (h5 : r.medium_price = 2)
  (h6 : r.large_price = 3) :
  medium_revenue r = 24 := by
  sorry

end medium_revenue_is_24_l3709_370913


namespace number_of_persons_working_prove_number_of_persons_working_l3709_370911

/-- The number of days it takes for some persons to finish the job -/
def group_days : ℕ := 8

/-- The number of days it takes for the first person to finish the job -/
def first_person_days : ℕ := 24

/-- The number of days it takes for the second person to finish the job -/
def second_person_days : ℕ := 12

/-- The work rate of a person is the fraction of the job they can complete in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- The theorem stating that the number of persons working on the job is 2 -/
theorem number_of_persons_working : ℕ :=
  2

/-- Proof that the number of persons working on the job is 2 -/
theorem prove_number_of_persons_working :
  work_rate group_days = work_rate first_person_days + work_rate second_person_days →
  number_of_persons_working = 2 := by
  sorry

end number_of_persons_working_prove_number_of_persons_working_l3709_370911


namespace high_school_harriers_loss_percentage_l3709_370985

theorem high_school_harriers_loss_percentage
  (total_games : ℝ)
  (games_won : ℝ)
  (games_lost : ℝ)
  (games_tied : ℝ)
  (h1 : games_won / games_lost = 5 / 3)
  (h2 : games_tied = 0.2 * total_games)
  (h3 : total_games = games_won + games_lost + games_tied) :
  games_lost / total_games = 0.3 := by
sorry

end high_school_harriers_loss_percentage_l3709_370985


namespace reggies_lost_games_l3709_370943

/-- Given the conditions of Reggie's marble game, prove the number of games he lost. -/
theorem reggies_lost_games 
  (total_games : ℕ) 
  (initial_marbles : ℕ) 
  (bet_per_game : ℕ) 
  (final_marbles : ℕ) 
  (h1 : total_games = 9)
  (h2 : initial_marbles = 100)
  (h3 : bet_per_game = 10)
  (h4 : final_marbles = 90) :
  (initial_marbles - final_marbles) / bet_per_game = 1 :=
by sorry

end reggies_lost_games_l3709_370943


namespace danny_thrice_jane_age_l3709_370935

/-- Proves that Danny was thrice as old as Jane 19 years ago -/
theorem danny_thrice_jane_age (danny_age : ℕ) (jane_age : ℕ) 
  (h1 : danny_age = 40) (h2 : jane_age = 26) : 
  ∃ x : ℕ, x = 19 ∧ (danny_age - x) = 3 * (jane_age - x) :=
by sorry

end danny_thrice_jane_age_l3709_370935


namespace six_graduates_distribution_l3709_370924

/-- The number of ways to distribute n graduates among 2 employers, 
    with each employer receiving at least k graduates -/
def distribution_schemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 graduates among 2 employers, 
    with each employer receiving at least 2 graduates, is 50 -/
theorem six_graduates_distribution : distribution_schemes 6 2 = 50 := by sorry

end six_graduates_distribution_l3709_370924


namespace eighth_term_value_l3709_370915

/-- An arithmetic sequence with 30 terms, first term 5, and last term 86 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (86 - 5) / 29
  5 + (n - 1) * d

theorem eighth_term_value :
  arithmetic_sequence 8 = 592 / 29 :=
by sorry

end eighth_term_value_l3709_370915


namespace unique_promotion_solution_l3709_370951

/-- Represents the promotional offer for pencils -/
structure PencilPromotion where
  base : ℕ  -- The number of pencils Pete's mom gave money for
  bonus : ℕ -- The additional pencils Pete could buy due to the promotion

/-- Defines the specific promotion where Pete buys 12 more pencils -/
def specificPromotion : PencilPromotion := { base := 49, bonus := 12 }

/-- Theorem stating that the specific promotion is the only one satisfying the conditions -/
theorem unique_promotion_solution : 
  ∀ (p : PencilPromotion), p.bonus = 12 → p.base = 49 := by
  sorry

#check unique_promotion_solution

end unique_promotion_solution_l3709_370951


namespace inequality_proof_l3709_370968

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.exp (-3))
  (hb : b = Real.log 1.02)
  (hc : c = Real.sin 0.04) : 
  b < c ∧ c < a := by
  sorry

end inequality_proof_l3709_370968


namespace expression_evaluation_l3709_370998

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  3 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 1/2) + x^2 * y) + 6 * x * y^2 = 1/2 :=
by sorry

end expression_evaluation_l3709_370998


namespace sin_cos_derivative_l3709_370990

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos (2 * x) := by
  sorry

end sin_cos_derivative_l3709_370990


namespace complex_expression_proof_l3709_370938

theorem complex_expression_proof :
  let A : ℂ := 5 - 2*I
  let M : ℂ := -3 + 2*I
  let S : ℂ := 2*I
  let P : ℂ := 3
  2 * (A - M + S - P) = 10 - 4*I :=
by sorry

end complex_expression_proof_l3709_370938


namespace base4_calculation_l3709_370923

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Converts a base 4 number to its decimal representation --/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal number to its base 4 representation --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Multiplication operation for base 4 numbers --/
def mul_base4 (a b : Base4) : Base4 := 
  to_base4 (to_decimal a * to_decimal b)

/-- Division operation for base 4 numbers --/
def div_base4 (a b : Base4) : Base4 := 
  to_base4 (to_decimal a / to_decimal b)

theorem base4_calculation : 
  mul_base4 (div_base4 (to_base4 210) (to_base4 3)) (to_base4 21) = to_base4 1102 := by sorry

end base4_calculation_l3709_370923


namespace third_derivative_y_l3709_370931

noncomputable def y (x : ℝ) : ℝ := (1 / x) * Real.sin (2 * x)

theorem third_derivative_y (x : ℝ) (hx : x ≠ 0) :
  (deriv^[3] y) x = 
    ((-6 / x^4 + 12 / x^2) * Real.sin (2 * x) + 
     (12 / x^3 - 8 / x) * Real.cos (2 * x)) :=
by sorry

end third_derivative_y_l3709_370931


namespace z_in_first_quadrant_l3709_370945

theorem z_in_first_quadrant (z : ℂ) : 
  (z - 2*I) * (1 + I) = Complex.abs (1 - Real.sqrt 3 * I) → 
  0 < z.re ∧ 0 < z.im := by
  sorry

end z_in_first_quadrant_l3709_370945


namespace least_possible_c_l3709_370994

theorem least_possible_c (a b c : ℕ+) : 
  (a + b + c : ℚ) / 3 = 20 →
  a ≤ b →
  b ≤ c →
  b = a + 13 →
  c ≥ 45 ∧ ∃ (a₀ b₀ c₀ : ℕ+), 
    (a₀ + b₀ + c₀ : ℚ) / 3 = 20 ∧
    a₀ ≤ b₀ ∧
    b₀ ≤ c₀ ∧
    b₀ = a₀ + 13 ∧
    c₀ = 45 :=
by sorry

end least_possible_c_l3709_370994


namespace find_B_l3709_370962

theorem find_B (A C B : ℤ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 := by
  sorry

end find_B_l3709_370962


namespace reverse_digits_problem_l3709_370956

/-- Given a two-digit number, returns the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The problem statement -/
theorem reverse_digits_problem : ∃ (v : ℕ), 57 + v = reverse_digits 57 :=
  sorry

end reverse_digits_problem_l3709_370956


namespace betty_age_l3709_370950

/-- Given the relationships between Albert, Mary, and Betty's ages, prove Betty's age -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14) :
  betty = 7 := by
  sorry

end betty_age_l3709_370950


namespace simplify_expression_calculate_expression_find_expression_value_l3709_370973

-- Question 1
theorem simplify_expression (a b : ℝ) :
  2 * (a - b)^2 - 4 * (a - b)^2 + 7 * (a - b)^2 = 5 * (a - b)^2 := by sorry

-- Question 2
theorem calculate_expression (a b : ℝ) (h : a^2 - 2*b^2 - 3 = 0) :
  -3*a^2 + 6*b^2 + 2032 = 2023 := by sorry

-- Question 3
theorem find_expression_value (a b : ℝ) (h1 : a^2 + 2*a*b = 15) (h2 : b^2 + 2*a*b = 6) :
  2*a^2 - 4*b^2 - 4*a*b = 6 := by sorry

end simplify_expression_calculate_expression_find_expression_value_l3709_370973


namespace school_dance_relationship_l3709_370930

theorem school_dance_relationship (b g : ℕ) : 
  (b > 0) →  -- There is at least one boy
  (g ≥ 7) →  -- There are at least 7 girls (for the first boy)
  (∀ i : ℕ, i > 0 ∧ i ≤ b → (7 + i - 1) ≤ g) →  -- Each boy can dance with his required number of girls
  (7 + b - 1 = g) →  -- The last boy dances with all girls
  b = g - 6 := by
sorry

end school_dance_relationship_l3709_370930


namespace slope_of_CD_is_one_l3709_370919

/-- Given a line l passing through the origin O and intersecting y = e^(x-1) at two different points A and B,
    and lines parallel to y-axis drawn through A and B intersecting y = ln x at C and D respectively,
    prove that the slope of line CD is 1. -/
theorem slope_of_CD_is_one (k : ℝ) (hk : k > 0) : ∃ x₁ x₂ : ℝ, 
  x₁ ≠ x₂ ∧ 
  k * x₁ = Real.exp (x₁ - 1) ∧ 
  k * x₂ = Real.exp (x₂ - 1) ∧ 
  (Real.log (k * x₁) - Real.log (k * x₂)) / (k * x₁ - k * x₂) = 1 := by
  sorry


end slope_of_CD_is_one_l3709_370919


namespace complement_A_implies_m_eq_4_l3709_370971

def S : Finset ℕ := {1, 2, 3, 4}

def A (m : ℕ) : Finset ℕ := S.filter (λ x => x^2 - 5*x + m = 0)

theorem complement_A_implies_m_eq_4 :
  (S \ A m) = {2, 3} → m = 4 := by
  sorry

end complement_A_implies_m_eq_4_l3709_370971


namespace worth_of_cloth_is_8540_l3709_370966

/-- Commission rates and sales data for an agent --/
structure SalesData where
  cloth_rate : ℝ
  electronics_rate_low : ℝ
  electronics_rate_high : ℝ
  electronics_threshold : ℝ
  stationery_rate_low : ℝ
  stationery_rate_high : ℝ
  stationery_threshold : ℕ
  total_commission : ℝ
  electronics_sales : ℝ
  stationery_units : ℕ

/-- Calculate the worth of cloth sold given sales data --/
def worth_of_cloth_sold (data : SalesData) : ℝ :=
  sorry

/-- Theorem stating that the worth of cloth sold is 8540 given the specific sales data --/
theorem worth_of_cloth_is_8540 (data : SalesData) 
  (h1 : data.cloth_rate = 0.025)
  (h2 : data.electronics_rate_low = 0.035)
  (h3 : data.electronics_rate_high = 0.045)
  (h4 : data.electronics_threshold = 3000)
  (h5 : data.stationery_rate_low = 10)
  (h6 : data.stationery_rate_high = 15)
  (h7 : data.stationery_threshold = 5)
  (h8 : data.total_commission = 418)
  (h9 : data.electronics_sales = 3100)
  (h10 : data.stationery_units = 8) :
  worth_of_cloth_sold data = 8540 := by
  sorry

end worth_of_cloth_is_8540_l3709_370966


namespace road_building_time_l3709_370936

/-- Given that 60 workers can build a road in 5 days, prove that 40 workers
    working at the same rate will take 7.5 days to build the same road. -/
theorem road_building_time (workers_initial : ℕ) (days_initial : ℝ)
    (workers_new : ℕ) (days_new : ℝ) : 
    workers_initial = 60 → days_initial = 5 → workers_new = 40 → 
    (workers_initial : ℝ) * days_initial = workers_new * days_new →
    days_new = 7.5 := by
  sorry

#check road_building_time

end road_building_time_l3709_370936


namespace remaining_budget_l3709_370906

/-- Proves that given a weekly food budget of $80, after purchasing a $12 bucket of fried chicken
    and 5 pounds of beef at $3 per pound, the remaining budget is $53. -/
theorem remaining_budget (weekly_budget : ℕ) (chicken_cost : ℕ) (beef_price : ℕ) (beef_amount : ℕ) :
  weekly_budget = 80 →
  chicken_cost = 12 →
  beef_price = 3 →
  beef_amount = 5 →
  weekly_budget - (chicken_cost + beef_price * beef_amount) = 53 := by
  sorry

end remaining_budget_l3709_370906


namespace parallel_line_equation_l3709_370914

/-- A line passing through a point and parallel to another line -/
def parallel_line (p : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 = m * q.1 + (p.2 - m * p.1)}

theorem parallel_line_equation :
  let p : ℝ × ℝ := (0, 7)
  let m : ℝ := -4
  parallel_line p m = {q : ℝ × ℝ | q.2 = -4 * q.1 + 7} := by
  sorry

end parallel_line_equation_l3709_370914


namespace sock_selection_theorem_l3709_370946

/-- The number of ways to choose 4 socks from 6 socks (where one is blue and the rest are different colors), 
    such that at least one chosen sock is blue. -/
def choose_socks (total_socks : ℕ) (blue_socks : ℕ) (choose : ℕ) : ℕ :=
  Nat.choose (total_socks - blue_socks) (choose - 1)

/-- Theorem stating that there are 10 ways to choose 4 socks from 6 socks, 
    where at least one is blue. -/
theorem sock_selection_theorem :
  choose_socks 6 1 4 = 10 := by
  sorry

#eval choose_socks 6 1 4

end sock_selection_theorem_l3709_370946


namespace two_digit_condition_l3709_370961

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property that we want to prove
def satisfiesCondition (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = 6 * sumOfDigits (n + 7)

-- Statement of the theorem
theorem two_digit_condition :
  ∀ n : ℕ, satisfiesCondition n ↔ (n = 24 ∨ n = 78) := by
  sorry

end two_digit_condition_l3709_370961


namespace wendy_facebook_pictures_l3709_370965

theorem wendy_facebook_pictures (total_albums : ℕ) (pics_in_first_album : ℕ) 
  (pics_per_other_album : ℕ) (other_albums : ℕ) :
  total_albums = other_albums + 1 →
  pics_in_first_album = 44 →
  pics_per_other_album = 7 →
  other_albums = 5 →
  pics_in_first_album + other_albums * pics_per_other_album = 79 := by
  sorry

end wendy_facebook_pictures_l3709_370965


namespace circle_alignment_exists_l3709_370905

/-- Represents a circle with a circumference of 100 cm -/
structure Circle :=
  (circumference : ℝ)
  (h_circumference : circumference = 100)

/-- Represents a set of marked points on a circle -/
structure MarkedPoints :=
  (circle : Circle)
  (num_points : ℕ)

/-- Represents a set of arcs on a circle -/
structure Arcs :=
  (circle : Circle)
  (total_length : ℝ)
  (h_length : total_length < 1)

/-- Represents an alignment of two circles -/
def Alignment := ℝ

/-- Checks if a marked point coincides with an arc for a given alignment -/
def coincides (mp : MarkedPoints) (a : Arcs) (alignment : Alignment) : Prop :=
  sorry

theorem circle_alignment_exists (c1 c2 : Circle) 
  (mp : MarkedPoints) (a : Arcs) 
  (h_mp : mp.circle = c1) (h_a : a.circle = c2) 
  (h_num_points : mp.num_points = 100) :
  ∃ (alignment : Alignment), ∀ (p : ℕ) (h_p : p < mp.num_points), 
    ¬ coincides mp a alignment :=
sorry

end circle_alignment_exists_l3709_370905


namespace student_count_l3709_370992

theorem student_count (N : ℕ) 
  (h1 : N / 5 + N / 4 + N / 2 + 5 = N) : N = 100 := by
  sorry

#check student_count

end student_count_l3709_370992


namespace x_value_proof_l3709_370941

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 := by
  sorry

end x_value_proof_l3709_370941


namespace yard_sale_books_theorem_l3709_370963

/-- Represents Melanie's book collection --/
structure BookCollection where
  initial_books : ℕ
  current_books : ℕ
  magazines : ℕ

/-- Calculates the number of books bought at the yard sale --/
def books_bought (collection : BookCollection) : ℕ :=
  collection.current_books - collection.initial_books

/-- Theorem stating that the number of books bought at the yard sale
    is the difference between current and initial book counts --/
theorem yard_sale_books_theorem (collection : BookCollection)
    (h1 : collection.initial_books = 83)
    (h2 : collection.current_books = 167)
    (h3 : collection.magazines = 57) :
    books_bought collection = 84 := by
  sorry

end yard_sale_books_theorem_l3709_370963


namespace S_not_algorithmically_solvable_l3709_370955

-- Define a type for expressions
inductive Expression
  | finite : Nat → Expression  -- Represents finite sums
  | infinite : Expression      -- Represents infinite sums

-- Define what it means for an expression to be algorithmically solvable
def is_algorithmically_solvable (e : Expression) : Prop :=
  match e with
  | Expression.finite _ => True
  | Expression.infinite => False

-- Define the infinite sum S = 1 + 2 + 3 + ...
def S : Expression := Expression.infinite

-- Theorem statement
theorem S_not_algorithmically_solvable :
  ¬(is_algorithmically_solvable S) :=
sorry

end S_not_algorithmically_solvable_l3709_370955


namespace prob_at_least_four_girls_value_l3709_370969

def num_children : ℕ := 7
def prob_girl : ℚ := 3/5
def prob_boy : ℚ := 2/5

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_at_least_four_girls : ℚ :=
  (binomial num_children 4 : ℚ) * (prob_girl ^ 4) * (prob_boy ^ 3) +
  (binomial num_children 5 : ℚ) * (prob_girl ^ 5) * (prob_boy ^ 2) +
  (binomial num_children 6 : ℚ) * (prob_girl ^ 6) * (prob_boy ^ 1) +
  (binomial num_children 7 : ℚ) * (prob_girl ^ 7) * (prob_boy ^ 0)

theorem prob_at_least_four_girls_value : prob_at_least_four_girls = 35325/78125 := by
  sorry

end prob_at_least_four_girls_value_l3709_370969


namespace circle_parameter_value_l3709_370949

theorem circle_parameter_value (θ : Real) : 
  0 ≤ θ ∧ θ < 2 * Real.pi →
  4 * Real.cos θ = -2 →
  4 * Real.sin θ = 2 * Real.sqrt 3 →
  θ = 2 * Real.pi / 3 := by
sorry

end circle_parameter_value_l3709_370949


namespace remote_sensing_primary_for_sea_level_info_l3709_370967

/-- Represents different technologies used in geographic information systems -/
inductive GISTechnology
  | RemoteSensing
  | GPS
  | GIS
  | DigitalEarth

/-- Represents the capability of a technology to acquire sea level rise information -/
def can_acquire_sea_level_info (tech : GISTechnology) : Prop :=
  match tech with
  | GISTechnology.RemoteSensing => true
  | _ => false

/-- Theorem stating that Remote Sensing is the primary technology for acquiring sea level rise information -/
theorem remote_sensing_primary_for_sea_level_info :
  ∀ (tech : GISTechnology),
    can_acquire_sea_level_info tech → tech = GISTechnology.RemoteSensing :=
by
  sorry


end remote_sensing_primary_for_sea_level_info_l3709_370967


namespace pyramid_sum_l3709_370904

theorem pyramid_sum (x : ℝ) : 
  let row2_left : ℝ := 11
  let row2_middle : ℝ := 6 + x
  let row2_right : ℝ := x + 7
  let row3_left : ℝ := row2_left + row2_middle
  let row3_right : ℝ := row2_middle + row2_right
  let row4 : ℝ := row3_left + row3_right
  row4 = 60 → x = 10 := by
sorry

end pyramid_sum_l3709_370904


namespace markese_earnings_l3709_370948

/-- Proves that Markese earned 16 dollars given the conditions of the problem -/
theorem markese_earnings (E : ℕ) 
  (h1 : E - 5 + E = 37) : 
  E - 5 = 16 := by
  sorry

end markese_earnings_l3709_370948


namespace sin_squared_sum_l3709_370909

theorem sin_squared_sum (α β : ℝ) : 
  Real.sin (α + β) ^ 2 = Real.cos α ^ 2 + Real.cos β ^ 2 - 2 * Real.cos α * Real.cos β * Real.cos (α + β) := by
  sorry

end sin_squared_sum_l3709_370909


namespace expression_evaluation_l3709_370981

theorem expression_evaluation : 8 - 6 / (4 - 2) = 5 := by sorry

end expression_evaluation_l3709_370981


namespace workshop_workers_count_l3709_370932

/-- Given a workshop with workers including technicians, prove the total number of workers. -/
theorem workshop_workers_count 
  (total_avg : ℝ)  -- Average salary of all workers
  (tech_count : ℕ) -- Number of technicians
  (tech_avg : ℝ)   -- Average salary of technicians
  (rest_avg : ℝ)   -- Average salary of the rest of the workers
  (h1 : total_avg = 850)
  (h2 : tech_count = 7)
  (h3 : tech_avg = 1000)
  (h4 : rest_avg = 780) :
  ∃ (total_workers : ℕ), total_workers = 22 :=
by sorry

end workshop_workers_count_l3709_370932


namespace max_product_logarithms_l3709_370976

/-- Given a, b, c > 1 satisfying the given equations, the maximum value of lg a · lg c is 16/3 -/
theorem max_product_logarithms (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq1 : Real.log a / Real.log 10 + Real.log c / Real.log b = 3)
  (eq2 : Real.log b / Real.log 10 + Real.log c / Real.log a = 4) :
  (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ 16/3 :=
by sorry

end max_product_logarithms_l3709_370976


namespace youngest_daughter_cost_l3709_370903

/-- Represents the cost of dresses and hats bought by the daughters -/
structure Purchase where
  dresses : ℕ
  hats : ℕ
  cost : ℕ

/-- The problem setup -/
def merchant_problem : Prop :=
  ∃ (dress_cost hat_cost : ℕ),
    let eldest := Purchase.mk 6 3 105
    let second := Purchase.mk 3 5 70
    let youngest := Purchase.mk 1 2 0
    eldest.cost = eldest.dresses * dress_cost + eldest.hats * hat_cost ∧
    second.cost = second.dresses * dress_cost + second.hats * hat_cost ∧
    youngest.dresses * dress_cost + youngest.hats * hat_cost = 25

/-- The theorem to be proved -/
theorem youngest_daughter_cost :
  merchant_problem := by sorry

end youngest_daughter_cost_l3709_370903


namespace walmart_sales_theorem_walmart_december_sales_l3709_370925

/-- Calculates the total sales amount for Wal-Mart in December -/
theorem walmart_sales_theorem (thermometer_price : ℕ) (hot_water_bottle_price : ℕ) 
  (thermometer_to_bottle_ratio : ℕ) (hot_water_bottles_sold : ℕ) : ℕ :=
  let thermometers_sold := thermometer_to_bottle_ratio * hot_water_bottles_sold
  let thermometer_sales := thermometers_sold * thermometer_price
  let hot_water_bottle_sales := hot_water_bottles_sold * hot_water_bottle_price
  thermometer_sales + hot_water_bottle_sales

/-- Proves that the total sales amount for Wal-Mart in December is $1200 -/
theorem walmart_december_sales : 
  walmart_sales_theorem 2 6 7 60 = 1200 := by
  sorry

end walmart_sales_theorem_walmart_december_sales_l3709_370925


namespace joes_lift_l3709_370954

theorem joes_lift (first_lift second_lift : ℕ) 
  (h1 : first_lift + second_lift = 1800)
  (h2 : 2 * first_lift = second_lift + 300) : 
  first_lift = 700 := by
sorry

end joes_lift_l3709_370954


namespace disprove_square_implies_greater_l3709_370926

theorem disprove_square_implies_greater : ∃ a b : ℝ, a^2 > b^2 ∧ a ≤ b :=
  let a := -3
  let b := 2
  have h1 : a^2 > b^2 := by sorry
  have h2 : a ≤ b := by sorry
  ⟨a, b, h1, h2⟩

#check disprove_square_implies_greater

end disprove_square_implies_greater_l3709_370926


namespace sum_of_y_coefficients_correct_expressions_equal_l3709_370922

/-- The sum of coefficients of terms containing y in (5x+3y+2)(2x+5y+3) -/
def sum_of_y_coefficients : ℤ := 65

/-- The original expression -/
def original_expression (x y : ℚ) : ℚ := (5*x + 3*y + 2) * (2*x + 5*y + 3)

/-- Expanded form of the original expression -/
def expanded_expression (x y : ℚ) : ℚ := 
  10*x^2 + 31*x*y + 19*x + 15*y^2 + 19*y + 6

/-- Theorem stating that the sum of coefficients of terms containing y 
    in the expanded expression is equal to sum_of_y_coefficients -/
theorem sum_of_y_coefficients_correct : 
  (31 : ℤ) + 15 + 19 = sum_of_y_coefficients := by sorry

/-- Theorem stating that the original expression and expanded expression are equal -/
theorem expressions_equal (x y : ℚ) : 
  original_expression x y = expanded_expression x y := by sorry

end sum_of_y_coefficients_correct_expressions_equal_l3709_370922


namespace empty_solution_set_range_l3709_370934

theorem empty_solution_set_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end empty_solution_set_range_l3709_370934


namespace complex_product_minus_p_l3709_370908

theorem complex_product_minus_p :
  let P : ℂ := 7 + 3 * Complex.I
  let Q : ℂ := 2 * Complex.I
  let R : ℂ := 7 - 3 * Complex.I
  (P * Q * R) - P = 113 * Complex.I - 7 := by sorry

end complex_product_minus_p_l3709_370908


namespace simplify_expression_l3709_370972

theorem simplify_expression :
  let x : ℝ := Real.sqrt 2
  let y : ℝ := Real.sqrt 3
  (x + 1) ^ (y - 1) / (x - 1) ^ (y + 1) = 3 - 2 * x := by
  sorry

end simplify_expression_l3709_370972


namespace weight_ratio_proof_l3709_370991

/-- Given the weights of Antoinette and Rupert, prove their weight ratio -/
theorem weight_ratio_proof (A R : ℚ) (k : ℚ) : 
  A = 63 → 
  A + R = 98 → 
  A = k * R - 7 → 
  A / R = 9 / 5 := by
  sorry

end weight_ratio_proof_l3709_370991


namespace parabola_reflection_l3709_370983

/-- A parabola is a function of the form y = a(x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Reflection of a parabola along the y-axis -/
def reflect_y_axis (p : Parabola) : Parabola :=
  { a := p.a, h := -p.h, k := p.k }

theorem parabola_reflection :
  let original := Parabola.mk 2 1 (-4)
  let reflected := reflect_y_axis original
  reflected = Parabola.mk 2 (-1) (-4) := by sorry

end parabola_reflection_l3709_370983


namespace power_27_mod_13_l3709_370952

theorem power_27_mod_13 : 27^482 ≡ 1 [ZMOD 13] := by sorry

end power_27_mod_13_l3709_370952


namespace inverse_proportion_y_relationship_l3709_370979

/-- Given three points on an inverse proportion function, prove their y-coordinates' relationship -/
theorem inverse_proportion_y_relationship
  (k : ℝ) (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)
  (h_k : k < 0)
  (h_x : x₁ < x₂ ∧ x₂ < 0 ∧ 0 < x₃)
  (h_y₁ : y₁ = k / x₁)
  (h_y₂ : y₂ = k / x₂)
  (h_y₃ : y₃ = k / x₃) :
  y₂ > y₁ ∧ y₁ > y₃ := by
sorry

end inverse_proportion_y_relationship_l3709_370979


namespace complex_division_l3709_370982

theorem complex_division (i : ℂ) (h : i^2 = -1) : 
  1 / (1 + i)^2 = -1/2 * i := by sorry

end complex_division_l3709_370982


namespace smiley_face_tulips_l3709_370933

/-- Calculates the total number of tulips needed for a smiley face design. -/
def total_tulips : ℕ :=
  let red_eyes : ℕ := 8 * 2
  let purple_eyebrows : ℕ := 5 * 2
  let red_nose : ℕ := 12
  let red_smile : ℕ := 18
  let yellow_background : ℕ := 9 * red_smile
  red_eyes + purple_eyebrows + red_nose + red_smile + yellow_background

/-- Theorem stating that the total number of tulips needed is 218. -/
theorem smiley_face_tulips : total_tulips = 218 := by
  sorry

end smiley_face_tulips_l3709_370933


namespace water_force_on_trapezoidal_dam_water_force_on_trapezoidal_dam_proof_l3709_370999

/-- The force exerted by water on a dam with an isosceles trapezoidal cross-section --/
theorem water_force_on_trapezoidal_dam
  (ρ : Real) -- density of water
  (g : Real) -- acceleration due to gravity
  (a : Real) -- top base of trapezoid
  (b : Real) -- bottom base of trapezoid
  (h : Real) -- height of trapezoid
  (hρ : ρ = 1000) -- density of water in kg/m³
  (hg : g = 10) -- acceleration due to gravity in m/s²
  (ha : a = 6.3) -- top base in meters
  (hb : b = 10.2) -- bottom base in meters
  (hh : h = 4.0) -- height in meters
  : Real :=
  -- The force F in Newtons
  608000

/-- Proof of the theorem --/
theorem water_force_on_trapezoidal_dam_proof
  (ρ g a b h : Real)
  (hρ : ρ = 1000)
  (hg : g = 10)
  (ha : a = 6.3)
  (hb : b = 10.2)
  (hh : h = 4.0)
  : water_force_on_trapezoidal_dam ρ g a b h hρ hg ha hb hh = 608000 := by
  sorry

end water_force_on_trapezoidal_dam_water_force_on_trapezoidal_dam_proof_l3709_370999


namespace midpoint_chain_l3709_370901

/-- Given points A, B, C, D, E, F on a line segment AB, where:
    C is the midpoint of AB,
    D is the midpoint of AC,
    E is the midpoint of AD,
    F is the midpoint of AE,
    and AF = 3,
    prove that AB = 48. -/
theorem midpoint_chain (A B C D E F : ℝ) 
  (hC : C = (A + B) / 2) 
  (hD : D = (A + C) / 2)
  (hE : E = (A + D) / 2)
  (hF : F = (A + E) / 2)
  (hAF : F - A = 3) : 
  B - A = 48 := by
  sorry

end midpoint_chain_l3709_370901


namespace complement_intersection_theorem_l3709_370928

def U : Set Nat := {x | x > 0 ∧ x < 9}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 8} := by sorry

end complement_intersection_theorem_l3709_370928


namespace train_passing_time_l3709_370910

/-- The time it takes for a train to pass a person moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 240 →
  train_speed = 100 * (5/18) →
  person_speed = 8 * (5/18) →
  (train_length / (train_speed + person_speed)) = 8 := by
  sorry

end train_passing_time_l3709_370910


namespace continued_fraction_evaluation_l3709_370986

theorem continued_fraction_evaluation :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
  sorry

end continued_fraction_evaluation_l3709_370986


namespace ratio_first_term_to_common_difference_l3709_370939

/-- An arithmetic progression where the sum of the first twenty terms
    is six times the sum of the first ten terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (20 * a + 190 * d) = 6 * (10 * a + 45 * d)

/-- The ratio of the first term to the common difference is 2 -/
theorem ratio_first_term_to_common_difference
  (ap : ArithmeticProgression) : ap.a / ap.d = 2 := by
  sorry

end ratio_first_term_to_common_difference_l3709_370939


namespace anya_lost_games_correct_l3709_370975

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- The total number of games played --/
def total_games : ℕ := 19

/-- The number of games each girl played --/
def games_played (g : Girl) : ℕ :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- A game is represented by its number and the two girls who played --/
structure Game where
  number : ℕ
  player1 : Girl
  player2 : Girl

/-- The set of all games played --/
def all_games : Set Game := sorry

/-- The set of games where Anya lost --/
def anya_lost_games : Set ℕ := {4, 8, 12, 16}

/-- The main theorem to prove --/
theorem anya_lost_games_correct :
  ∀ (g : Game), g ∈ all_games → 
    (g.player1 = Girl.Anya ∨ g.player2 = Girl.Anya) → 
    g.number ∈ anya_lost_games :=
  sorry

end anya_lost_games_correct_l3709_370975


namespace tangent_line_equation_l3709_370942

/-- The equation of the tangent line to y = 2x - x³ at (1, 1) is x + y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = 2*x - x^3) → -- The curve equation
  ((1 : ℝ) = 1 → (2*(1 : ℝ) - (1 : ℝ)^3) = 1) → -- The point (1, 1) lies on the curve
  (x + y - 2 = 0) ↔ -- The tangent line equation
  (∃ (m : ℝ), y - 1 = m * (x - 1) ∧ 
              m = (2 - 3*(1 : ℝ)^2)) -- Slope of the tangent line at x = 1
  := by sorry

end tangent_line_equation_l3709_370942


namespace line_through_points_l3709_370912

def point_on_line (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

theorem line_through_points : 
  point_on_line 4 8 8 16 2 4 ∧
  point_on_line 6 12 8 16 2 4 ∧
  point_on_line 10 20 8 16 2 4 ∧
  ¬ point_on_line 5 11 8 16 2 4 ∧
  ¬ point_on_line 3 7 8 16 2 4 :=
by sorry

end line_through_points_l3709_370912


namespace circle_equation_l3709_370970

/-- The standard equation of a circle with center (-2, 1) passing through (0, 1) -/
theorem circle_equation :
  let center : ℝ × ℝ := (-2, 1)
  let point_on_circle : ℝ × ℝ := (0, 1)
  ∀ (x y : ℝ),
    (x + 2)^2 + (y - 1)^2 = 4 ↔
    (x - center.1)^2 + (y - center.2)^2 = (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 :=
by sorry

end circle_equation_l3709_370970


namespace estimate_proportion_approx_5_7_percent_l3709_370974

/-- Represents the survey data and population information -/
structure SurveyData where
  total_households : ℕ
  ordinary_households : ℕ
  high_income_households : ℕ
  sample_ordinary : ℕ
  sample_high_income : ℕ
  total_with_3plus_housing : ℕ
  ordinary_with_3plus_housing : ℕ
  high_income_with_3plus_housing : ℕ

/-- Calculates the estimated proportion of households with 3+ housing sets -/
def estimate_proportion (data : SurveyData) : ℝ :=
  sorry

/-- Theorem stating that the estimated proportion is approximately 5.7% -/
theorem estimate_proportion_approx_5_7_percent (data : SurveyData)
  (h1 : data.total_households = 100000)
  (h2 : data.ordinary_households = 99000)
  (h3 : data.high_income_households = 1000)
  (h4 : data.sample_ordinary = 990)
  (h5 : data.sample_high_income = 100)
  (h6 : data.total_with_3plus_housing = 120)
  (h7 : data.ordinary_with_3plus_housing = 50)
  (h8 : data.high_income_with_3plus_housing = 70) :
  ∃ ε > 0, |estimate_proportion data - 0.057| < ε :=
sorry

end estimate_proportion_approx_5_7_percent_l3709_370974


namespace quadratic_equation_roots_l3709_370997

theorem quadratic_equation_roots (a m : ℤ) : 
  (∃ x : ℤ, (a - 1) * x^2 + a * x + 1 = 0 ∧ (m^2 + m) * x^2 + 3 * m * x - 3 = 0) →
  (a = -2 ∧ (m = -1 ∨ m = 3)) :=
by sorry

end quadratic_equation_roots_l3709_370997


namespace evenProductProbabilityFor6And4_l3709_370957

/-- Represents a spinner with n equal segments numbered from 1 to n -/
structure Spinner :=
  (n : ℕ)

/-- The probability of getting an even product when spinning two spinners -/
def evenProductProbability (spinnerA spinnerB : Spinner) : ℚ :=
  sorry

/-- Theorem stating that the probability of getting an even product
    when spinning a 6-segment spinner and a 4-segment spinner is 1/2 -/
theorem evenProductProbabilityFor6And4 :
  evenProductProbability (Spinner.mk 6) (Spinner.mk 4) = 1/2 :=
sorry

end evenProductProbabilityFor6And4_l3709_370957


namespace total_pencils_l3709_370988

theorem total_pencils (num_boxes : ℕ) (pencils_per_box : ℕ) (h1 : num_boxes = 3) (h2 : pencils_per_box = 9) :
  num_boxes * pencils_per_box = 27 := by
  sorry

end total_pencils_l3709_370988


namespace orange_purchase_ratio_l3709_370995

/-- Proves the ratio of weekly orange purchases --/
theorem orange_purchase_ratio 
  (initial_purchase : ℕ) 
  (additional_purchase : ℕ) 
  (total_after_three_weeks : ℕ) 
  (h1 : initial_purchase = 10)
  (h2 : additional_purchase = 5)
  (h3 : total_after_three_weeks = 75) :
  (total_after_three_weeks - (initial_purchase + additional_purchase)) / 2 = 
  2 * (initial_purchase + additional_purchase) :=
by
  sorry

#check orange_purchase_ratio

end orange_purchase_ratio_l3709_370995


namespace sale_final_prices_correct_l3709_370940

/-- Calculates the final price of an item after a series of percentage discounts and flat discounts --/
def finalPrice (originalPrice : ℝ) (percentDiscounts : List ℝ) (flatDiscounts : List ℝ) : ℝ :=
  let applyPercentDiscount (price : ℝ) (discount : ℝ) := price * (1 - discount)
  let applyFlatDiscount (price : ℝ) (discount : ℝ) := price - discount
  let priceAfterPercentDiscounts := percentDiscounts.foldl applyPercentDiscount originalPrice
  flatDiscounts.foldl applyFlatDiscount priceAfterPercentDiscounts

/-- Proves that the final prices of the electronic item and clothing item are correct after the 4-day sale --/
theorem sale_final_prices_correct (electronicOriginalPrice clothingOriginalPrice : ℝ) 
  (h1 : electronicOriginalPrice = 480)
  (h2 : clothingOriginalPrice = 260) : 
  let electronicFinalPrice := finalPrice electronicOriginalPrice [0.1, 0.14, 0.12, 0.08] []
  let clothingFinalPrice := finalPrice clothingOriginalPrice [0.1, 0.12, 0.05] [20]
  (electronicFinalPrice = 300.78 ∧ clothingFinalPrice = 176.62) := by
  sorry


end sale_final_prices_correct_l3709_370940


namespace odd_numbers_theorem_l3709_370907

theorem odd_numbers_theorem (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 :=
by sorry

end odd_numbers_theorem_l3709_370907


namespace seven_balls_four_boxes_l3709_370916

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by
  sorry

end seven_balls_four_boxes_l3709_370916


namespace x_value_l3709_370900

theorem x_value : ∃ x : ℝ, x ≠ 0 ∧ x = 3 * (1/x * (-x)) + 5 → x = 2 := by
  sorry

end x_value_l3709_370900


namespace complex_sum_zero_l3709_370927

theorem complex_sum_zero : 
  let z : ℂ := (1 / 2 : ℂ) + (Complex.I * Real.sqrt 3 / 2)
  z + z^2 + z^3 + z^4 + z^5 + z^6 = 0 := by
  sorry

end complex_sum_zero_l3709_370927


namespace permutations_of_six_l3709_370977

theorem permutations_of_six (n : Nat) : n = 6 → Nat.factorial n = 720 := by
  sorry

end permutations_of_six_l3709_370977


namespace sine_sum_equality_l3709_370902

theorem sine_sum_equality : 
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) + 
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sine_sum_equality_l3709_370902


namespace preimage_of_three_one_l3709_370929

/-- A mapping from A to B -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (2, 1) is the preimage of (3, 1) under f -/
theorem preimage_of_three_one :
  f (2, 1) = (3, 1) ∧ ∀ p : ℝ × ℝ, f p = (3, 1) → p = (2, 1) :=
by sorry

end preimage_of_three_one_l3709_370929


namespace dress_trim_cuff_length_l3709_370937

/-- Proves that the length of each cuff is 50 cm given the dress trimming conditions --/
theorem dress_trim_cuff_length :
  let hem_length : ℝ := 300
  let waist_length : ℝ := hem_length / 3
  let neck_ruffles : ℕ := 5
  let ruffle_length : ℝ := 20
  let lace_cost_per_meter : ℝ := 6
  let total_spent : ℝ := 36
  let total_lace_length : ℝ := total_spent / lace_cost_per_meter * 100
  let hem_waist_neck_length : ℝ := hem_length + waist_length + (neck_ruffles : ℝ) * ruffle_length
  let cuff_total_length : ℝ := total_lace_length - hem_waist_neck_length
  cuff_total_length / 2 = 50 := by sorry

end dress_trim_cuff_length_l3709_370937


namespace eighteen_picks_required_l3709_370947

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : Nat
  black : Nat
  yellow : Nat

/-- The minimum number of picks required to guarantee at least one ball of each color -/
def minPicksRequired (counts : BallCounts) : Nat :=
  counts.white + counts.black + 1

/-- Theorem stating that for the given ball counts, 18 picks are required -/
theorem eighteen_picks_required (counts : BallCounts) 
  (h_white : counts.white = 8)
  (h_black : counts.black = 9)
  (h_yellow : counts.yellow = 7) : 
  minPicksRequired counts = 18 := by
  sorry

end eighteen_picks_required_l3709_370947


namespace ticket_price_values_l3709_370960

theorem ticket_price_values (x : ℕ) : 
  (∃ (a b c : ℕ), x * a = 72 ∧ x * b = 90 ∧ x * c = 45) ↔ 
  (x = 1 ∨ x = 3 ∨ x = 9) := by
sorry

end ticket_price_values_l3709_370960


namespace solution_pairs_l3709_370920

theorem solution_pairs (x y : ℕ+) : 
  let d := Nat.gcd x y
  x * y * d = x + y + d^2 ↔ (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end solution_pairs_l3709_370920


namespace total_beakers_l3709_370984

theorem total_beakers (copper_beakers : ℕ) (drops_per_test : ℕ) (total_drops : ℕ) (non_copper_tested : ℕ) :
  copper_beakers = 8 →
  drops_per_test = 3 →
  total_drops = 45 →
  non_copper_tested = 7 →
  copper_beakers + non_copper_tested = total_drops / drops_per_test :=
by sorry

end total_beakers_l3709_370984


namespace positive_integer_solutions_inequality_l3709_370944

theorem positive_integer_solutions_inequality (x : ℕ+) :
  2 * (x.val - 1) < 7 - x.val ↔ x = 1 ∨ x = 2 := by sorry

end positive_integer_solutions_inequality_l3709_370944


namespace integer_root_values_l3709_370917

def polynomial (a x : ℤ) : ℤ := x^3 + 3*x^2 + a*x + 9

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_values :
  {a : ℤ | has_integer_root a} = {-109, -21, -13, 3, 11, 53} := by sorry

end integer_root_values_l3709_370917


namespace set_equality_implies_sum_l3709_370964

theorem set_equality_implies_sum (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → a^2022 + b^2023 = 1 := by
  sorry

end set_equality_implies_sum_l3709_370964


namespace vehicle_distance_time_l3709_370980

/-- Proves that two vehicles traveling in opposite directions for 4 hours
    will be 384 miles apart, given their respective speeds -/
theorem vehicle_distance_time (slower_speed faster_speed : ℝ) 
    (h1 : slower_speed = 44)
    (h2 : faster_speed = slower_speed + 8)
    (distance : ℝ) (h3 : distance = 384) : 
    (slower_speed + faster_speed) * 4 = distance := by
  sorry

end vehicle_distance_time_l3709_370980


namespace bug_path_tiles_l3709_370989

/-- Represents a rectangular floor with integer dimensions -/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.width + floor.length - Nat.gcd floor.width floor.length

theorem bug_path_tiles (floor : RectangularFloor) 
  (h_width : floor.width = 9) 
  (h_length : floor.length = 13) : 
  tilesVisited floor = 21 := by
sorry

#eval tilesVisited ⟨9, 13⟩

end bug_path_tiles_l3709_370989


namespace only_point_A_in_region_l3709_370987

def plane_region (x y : ℝ) : Prop := x + y - 1 < 0

def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (2, 4)
def point_C : ℝ × ℝ := (-1, 4)
def point_D : ℝ × ℝ := (1, 8)

theorem only_point_A_in_region :
  plane_region point_A.1 point_A.2 ∧
  ¬plane_region point_B.1 point_B.2 ∧
  ¬plane_region point_C.1 point_C.2 ∧
  ¬plane_region point_D.1 point_D.2 := by
  sorry

end only_point_A_in_region_l3709_370987


namespace smith_children_age_problem_l3709_370996

theorem smith_children_age_problem :
  ∀ (age1 age2 age3 : ℕ),
  age1 = 6 →
  age2 = 8 →
  (age1 + age2 + age3) / 3 = 9 →
  age3 = 13 := by
sorry

end smith_children_age_problem_l3709_370996


namespace justine_colored_sheets_l3709_370959

/-- Given a total number of sheets and binders, calculate the number of sheets Justine colored. -/
def sheets_colored (total_sheets : ℕ) (num_binders : ℕ) : ℕ :=
  let sheets_per_binder := total_sheets / num_binders
  (2 * sheets_per_binder) / 3

/-- Prove that Justine colored 356 sheets given the problem conditions. -/
theorem justine_colored_sheets :
  sheets_colored 3750 7 = 356 := by
  sorry

end justine_colored_sheets_l3709_370959


namespace james_overtime_multiple_l3709_370958

/-- Harry's pay rate for the first 24 hours -/
def harry_base_rate (x : ℝ) : ℝ := x

/-- Harry's pay rate for additional hours -/
def harry_overtime_rate (x : ℝ) : ℝ := 1.5 * x

/-- James's pay rate for the first 40 hours -/
def james_base_rate (x : ℝ) : ℝ := x

/-- James's pay rate for additional hours -/
def james_overtime_rate (x : ℝ) (m : ℝ) : ℝ := m * x

/-- Total hours worked by James -/
def james_total_hours : ℝ := 41

/-- Theorem stating the multiple of x dollars for James's overtime -/
theorem james_overtime_multiple (x : ℝ) (m : ℝ) :
  (harry_base_rate x * 24 + harry_overtime_rate x * (james_total_hours - 24) =
   james_base_rate x * 40 + james_overtime_rate x m * (james_total_hours - 40)) →
  m = 9.5 := by
  sorry

#check james_overtime_multiple

end james_overtime_multiple_l3709_370958


namespace stratified_sample_theorem_l3709_370993

/-- Represents the distribution of students across four years -/
structure StudentDistribution :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- Calculates the total number of students -/
def total_students (d : StudentDistribution) : ℕ :=
  d.first + d.second + d.third + d.fourth

/-- Calculates the number of students in a stratified sample for a given year -/
def stratified_sample_size (total : ℕ) (year_count : ℕ) (sample_size : ℕ) : ℕ :=
  (year_count * sample_size) / total

theorem stratified_sample_theorem (d : StudentDistribution) 
  (h1 : d.first = 400)
  (h2 : d.second = 300)
  (h3 : d.third = 200)
  (h4 : d.fourth = 100)
  (h5 : total_students d = 1000)
  (sample_size : ℕ)
  (h6 : sample_size = 200) :
  stratified_sample_size (total_students d) d.third sample_size = 40 := by
  sorry

#check stratified_sample_theorem

end stratified_sample_theorem_l3709_370993


namespace f_odd_and_increasing_l3709_370953

def f (x : ℝ) : ℝ := x^3

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end f_odd_and_increasing_l3709_370953


namespace unique_score_with_three_combinations_l3709_370921

/-- Represents a scoring combination for the test -/
structure ScoringCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given scoring combination -/
def calculateScore (sc : ScoringCombination) : ℕ :=
  6 * sc.correct + 3 * sc.unanswered

/-- Checks if a scoring combination is valid (sums to 25 questions) -/
def isValidCombination (sc : ScoringCombination) : Prop :=
  sc.correct + sc.unanswered + sc.incorrect = 25

/-- Theorem: 78 is the only score achievable in exactly three ways -/
theorem unique_score_with_three_combinations :
  ∃! score : ℕ,
    (∃ (combinations : Finset ScoringCombination),
      combinations.card = 3 ∧
      (∀ sc ∈ combinations, isValidCombination sc ∧ calculateScore sc = score) ∧
      (∀ sc : ScoringCombination, isValidCombination sc ∧ calculateScore sc = score → sc ∈ combinations)) ∧
    score = 78 := by
  sorry

end unique_score_with_three_combinations_l3709_370921
