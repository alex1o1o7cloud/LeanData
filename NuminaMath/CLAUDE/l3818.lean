import Mathlib

namespace f_inequality_l3818_381874

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_def (x : ℝ) : f (Real.tan (2 * x)) = Real.tan x ^ 4 + (1 / Real.tan x) ^ 4

theorem f_inequality : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) ≥ 196 := by
  sorry

end f_inequality_l3818_381874


namespace factorization_m_squared_minus_3m_l3818_381863

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m-3) := by
  sorry

end factorization_m_squared_minus_3m_l3818_381863


namespace geometric_sequence_ratio_sum_l3818_381800

theorem geometric_sequence_ratio_sum (m a₂ a₃ b₂ b₃ x y : ℝ) 
  (hm : m ≠ 0)
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (hxy : x ≠ y)
  (ha₂ : a₂ = m * x)
  (ha₃ : a₃ = m * x^2)
  (hb₂ : b₂ = m * y)
  (hb₃ : b₃ = m * y^2)
  (heq : a₃ - b₃ = 3 * (a₂ - b₂)) :
  x + y = 3 := by sorry

end geometric_sequence_ratio_sum_l3818_381800


namespace marys_next_birthday_age_l3818_381899

theorem marys_next_birthday_age 
  (d : ℝ) -- Danielle's age
  (j : ℝ) -- John's age
  (s : ℝ) -- Sally's age
  (m : ℝ) -- Mary's age
  (h1 : j = 1.15 * d) -- John is 15% older than Danielle
  (h2 : s = 1.30 * d) -- Sally is 30% older than Danielle
  (h3 : m = 1.25 * s) -- Mary is 25% older than Sally
  (h4 : j + d + s + m = 80) -- Sum of ages is 80
  : Int.floor m + 1 = 26 := by
  sorry

#check marys_next_birthday_age

end marys_next_birthday_age_l3818_381899


namespace max_m_proof_l3818_381884

/-- The maximum value of m given the condition -/
def max_m : ℝ := -2

/-- The condition function -/
def condition (x : ℝ) : Prop := x^2 - 2*x - 8 > 0

/-- The main theorem -/
theorem max_m_proof :
  (∀ x : ℝ, x < max_m → condition x) ∧
  (∃ x : ℝ, x < max_m ∧ ¬condition x) ∧
  (∀ m : ℝ, m > max_m → ∃ x : ℝ, x < m ∧ ¬condition x) :=
sorry

end max_m_proof_l3818_381884


namespace carol_initial_cupcakes_l3818_381831

/-- Given that if Carol sold 9 cupcakes and made 28 more, she would have 49 cupcakes,
    prove that Carol initially made 30 cupcakes. -/
theorem carol_initial_cupcakes : 
  ∀ (initial : ℕ), 
  (initial - 9 + 28 = 49) → 
  initial = 30 := by
sorry

end carol_initial_cupcakes_l3818_381831


namespace line_through_three_points_l3818_381852

/-- Given that the points (0, 2), (10, m), and (25, -3) lie on the same line, prove that m = 0. -/
theorem line_through_three_points (m : ℝ) : 
  (∀ t : ℝ, ∃ a b : ℝ, t * (10 - 0) + 0 = 10 ∧ 
                       t * (m - 2) + 2 = m ∧ 
                       t * (25 - 0) + 0 = 25 ∧ 
                       t * (-3 - 2) + 2 = -3) → 
  m = 0 := by
  sorry

end line_through_three_points_l3818_381852


namespace equal_non_overlapping_areas_l3818_381869

-- Define two congruent triangles
def Triangle : Type := ℝ × ℝ × ℝ

-- Define a function to calculate the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the hexagon formed by the intersection
def Hexagon : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define a function to calculate the area of a hexagon
def hexagon_area (h : Hexagon) : ℝ := sorry

-- Define the overlapping triangles and their intersection
def triangles_overlap (t1 t2 : Triangle) (h : Hexagon) : Prop :=
  ∃ (a1 a2 : ℝ), 
    area t1 = hexagon_area h + a1 ∧
    area t2 = hexagon_area h + a2

-- Theorem statement
theorem equal_non_overlapping_areas 
  (t1 t2 : Triangle) 
  (h : Hexagon) 
  (congruent : area t1 = area t2) 
  (overlap : triangles_overlap t1 t2 h) : 
  ∃ (a : ℝ), 
    area t1 = hexagon_area h + a ∧ 
    area t2 = hexagon_area h + a := 
by sorry

end equal_non_overlapping_areas_l3818_381869


namespace f_increasing_when_x_gt_1_l3818_381873

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem f_increasing_when_x_gt_1 :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end f_increasing_when_x_gt_1_l3818_381873


namespace sequence_problem_l3818_381827

theorem sequence_problem (a : ℕ → ℚ) (m : ℕ) :
  a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → a n - a (n + 1) = a (n + 1) * a n) →
  8 * a m = 1 →
  m = 8 :=
by sorry

end sequence_problem_l3818_381827


namespace unique_a_value_l3818_381856

open Real

theorem unique_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 π, f x = (cos (2 * x) + a) / sin x) →
  (∀ x ∈ Set.Ioo 0 π, |f x| ≤ 3) →
  a = -1 :=
sorry

end unique_a_value_l3818_381856


namespace movies_watched_correct_l3818_381816

/-- The number of movies watched in the 'crazy silly school' series --/
def moviesWatched (totalMovies : ℕ) (moviesToWatch : ℕ) : ℕ :=
  totalMovies - moviesToWatch

/-- Theorem: The number of movies watched is correct --/
theorem movies_watched_correct (totalMovies moviesToWatch : ℕ) 
  (h1 : totalMovies = 17) 
  (h2 : moviesToWatch = 10) : 
  moviesWatched totalMovies moviesToWatch = 7 := by
  sorry

#eval moviesWatched 17 10

end movies_watched_correct_l3818_381816


namespace square_tiles_count_l3818_381809

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 35)
  (h_total_edges : total_edges = 140) :
  ∃ (t s p : ℕ),
    t + s + p = total_tiles ∧
    3 * t + 4 * s + 5 * p = total_edges ∧
    s = 35 := by
  sorry

end square_tiles_count_l3818_381809


namespace order_of_rational_numbers_l3818_381821

theorem order_of_rational_numbers (a b : ℚ) 
  (ha : a > 0) (hb : b < 0) (hab : |a| < |b|) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end order_of_rational_numbers_l3818_381821


namespace floor_inequality_l3818_381879

theorem floor_inequality (x y : ℝ) : 
  ⌊2*x⌋ + ⌊2*y⌋ ≥ ⌊x⌋ + ⌊y⌋ + ⌊x + y⌋ :=
sorry

end floor_inequality_l3818_381879


namespace product_of_digits_not_divisible_by_4_l3818_381833

def numbers : List Nat := [4624, 4634, 4644, 4652, 4672]

def is_divisible_by_4 (n : Nat) : Bool :=
  n % 4 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_4 :
  ∃ n ∈ numbers, ¬is_divisible_by_4 n ∧ units_digit n * tens_digit n = 12 := by
  sorry

end product_of_digits_not_divisible_by_4_l3818_381833


namespace apple_harvest_l3818_381862

/-- Proves that the initial number of apples is 569 given the harvesting conditions -/
theorem apple_harvest (new_apples : ℕ) (rotten_apples : ℕ) (current_apples : ℕ)
  (h1 : new_apples = 419)
  (h2 : rotten_apples = 263)
  (h3 : current_apples = 725) :
  current_apples + rotten_apples - new_apples = 569 := by
  sorry

#check apple_harvest

end apple_harvest_l3818_381862


namespace math_club_members_l3818_381885

/-- 
Given a Math club where:
- There are two times as many males as females
- There are 6 female members
Prove that the total number of members in the Math club is 18.
-/
theorem math_club_members :
  ∀ (female_members male_members total_members : ℕ),
    female_members = 6 →
    male_members = 2 * female_members →
    total_members = female_members + male_members →
    total_members = 18 := by
  sorry

end math_club_members_l3818_381885


namespace radio_loss_percentage_l3818_381875

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The loss percentage for a radio with cost price 1900 and selling price 1330 is 30% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1900
  let selling_price : ℚ := 1330
  loss_percentage cost_price selling_price = 30 := by
sorry

end radio_loss_percentage_l3818_381875


namespace x_squared_coefficient_in_binomial_expansion_l3818_381858

/-- Given a natural number n, returns the binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The exponent n for which the 5th term in (x-1/x)^n has the largest coefficient -/
def n : ℕ := 8

/-- The coefficient of x^2 in the expansion of (x-1/x)^n -/
def coefficient_x_squared : ℤ := -56

theorem x_squared_coefficient_in_binomial_expansion :
  coefficient_x_squared = (-1)^3 * binomial n 3 := by sorry

end x_squared_coefficient_in_binomial_expansion_l3818_381858


namespace alien_alphabet_l3818_381851

theorem alien_alphabet (total : ℕ) (both : ℕ) (triangle_only : ℕ) 
  (h1 : total = 120)
  (h2 : both = 32)
  (h3 : triangle_only = 72)
  (h4 : total = both + triangle_only + (total - (both + triangle_only))) :
  total - (both + triangle_only) = 16 := by
  sorry

end alien_alphabet_l3818_381851


namespace sin_90_degrees_l3818_381860

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l3818_381860


namespace rectangular_plot_fence_poles_l3818_381859

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Theorem: A 90m by 40m rectangular plot with fence poles 5m apart needs 52 poles -/
theorem rectangular_plot_fence_poles :
  fence_poles 90 40 5 = 52 := by
  sorry

end rectangular_plot_fence_poles_l3818_381859


namespace midpoint_of_AB_l3818_381877

-- Define the point F
def F : ℝ × ℝ := (0, 1)

-- Define the line y = -5
def line_y_neg5 (x : ℝ) : ℝ := -5

-- Define the line x - 4y + 2 = 0
def line_l (x y : ℝ) : Prop := x - 4*y + 2 = 0

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ 
    (Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) + 4 = Real.sqrt ((P.1 - P.1)^2 + (P.2 - line_y_neg5 P.1)^2))

-- Define the trajectory of P (parabola)
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  trajectory A.1 A.2 ∧ trajectory B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_AB :
  ∀ (P A B : ℝ × ℝ),
  distance_condition P →
  intersection_points A B →
  (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 5/8 :=
sorry

end midpoint_of_AB_l3818_381877


namespace bicycles_in_garage_l3818_381824

theorem bicycles_in_garage (tricycles unicycles total_wheels : ℕ) 
  (h1 : tricycles = 4)
  (h2 : unicycles = 7)
  (h3 : total_wheels = 25) : ∃ bicycles : ℕ, 
  bicycles * 2 + tricycles * 3 + unicycles * 1 = total_wheels ∧ bicycles = 3 := by
  sorry

end bicycles_in_garage_l3818_381824


namespace min_value_theorem_l3818_381842

theorem min_value_theorem (x y : ℝ) : (x + y)^2 + (x - 2/y)^2 ≥ 4 := by
  sorry

end min_value_theorem_l3818_381842


namespace factors_of_48_l3818_381829

/-- The number of distinct positive factors of 48 is 10. -/
theorem factors_of_48 : Nat.card (Nat.divisors 48) = 10 := by sorry

end factors_of_48_l3818_381829


namespace tent_production_equation_correct_l3818_381883

/-- Represents the tent production scenario -/
structure TentProduction where
  original_plan : ℕ
  increase_percentage : ℚ
  days_ahead : ℕ
  daily_increase : ℕ

/-- The equation representing the tent production scenario -/
def production_equation (tp : TentProduction) (x : ℚ) : Prop :=
  (tp.original_plan : ℚ) / (x - tp.daily_increase) - 
  (tp.original_plan * (1 + tp.increase_percentage)) / x = tp.days_ahead

/-- Theorem stating that the equation correctly represents the given conditions -/
theorem tent_production_equation_correct (tp : TentProduction) (x : ℚ) 
  (h1 : tp.original_plan = 7200)
  (h2 : tp.increase_percentage = 1/5)
  (h3 : tp.days_ahead = 4)
  (h4 : tp.daily_increase = 720)
  (h5 : x > tp.daily_increase) :
  production_equation tp x := by
  sorry

end tent_production_equation_correct_l3818_381883


namespace ice_cream_cup_cost_l3818_381834

/-- Calculates the cost of each ice-cream cup given the order details and total amount paid -/
theorem ice_cream_cup_cost
  (chapati_count : ℕ)
  (chapati_cost : ℕ)
  (rice_count : ℕ)
  (rice_cost : ℕ)
  (vegetable_count : ℕ)
  (vegetable_cost : ℕ)
  (ice_cream_count : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : chapati_cost = 6)
  (h3 : rice_count = 5)
  (h4 : rice_cost = 45)
  (h5 : vegetable_count = 7)
  (h6 : vegetable_cost = 70)
  (h7 : ice_cream_count = 6)
  (h8 : total_paid = 883)
  : (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 12 := by
  sorry

#check ice_cream_cup_cost

end ice_cream_cup_cost_l3818_381834


namespace line_through_points_l3818_381891

/-- Given a line y = ax + b passing through points (3,7) and (7,19), prove that a - b = 5 -/
theorem line_through_points (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) → 
  (7 : ℝ) = a * 3 + b → 
  (19 : ℝ) = a * 7 + b → 
  a - b = 5 := by
  sorry

end line_through_points_l3818_381891


namespace trajectory_of_Q_l3818_381895

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

-- Define point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the relation between O, Q, and P
def relation_OQP (qx qy px py : ℝ) : Prop :=
  2 * (qx - 0, qy - 0) = (px - qx, py - qy)

-- Theorem statement
theorem trajectory_of_Q (qx qy : ℝ) :
  (∃ px py, point_P px py ∧ relation_OQP qx qy px py) →
  2 * qx + 4 * qy + 1 = 0 :=
by sorry

end trajectory_of_Q_l3818_381895


namespace min_value_expression_l3818_381870

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x - 2*y)^2 = (x*y)^3) : 
  4/x^2 + 4/(x*y) + 1/y^2 ≥ 4 * Real.sqrt 2 := by
  sorry

end min_value_expression_l3818_381870


namespace b_sixth_mod_n_l3818_381818

theorem b_sixth_mod_n (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) :
  b^6 ≡ 1 [ZMOD n] := by
  sorry

end b_sixth_mod_n_l3818_381818


namespace cyclist_catchup_time_l3818_381890

/-- Two cyclists A and B travel from station A to station B -/
structure Cyclist where
  speed : ℝ
  startTime : ℝ

/-- The problem setup -/
def cyclistProblem (A B : Cyclist) (distance : ℝ) : Prop :=
  A.speed * 30 = distance ∧  -- A takes 30 minutes to reach station B
  B.speed * 40 = distance ∧  -- B takes 40 minutes to reach station B
  B.startTime = A.startTime - 5  -- B starts 5 minutes earlier than A

/-- The theorem to prove -/
theorem cyclist_catchup_time (A B : Cyclist) (distance : ℝ) 
  (h : cyclistProblem A B distance) : 
  ∃ t : ℝ, t = 15 ∧ A.speed * t = B.speed * (t + 5) :=
sorry

end cyclist_catchup_time_l3818_381890


namespace recruit_line_total_l3818_381889

/-- Represents the position of a person in the line of recruits -/
structure Position where
  front : Nat
  behind : Nat

/-- The line of recruits -/
structure RecruitLine where
  peter : Position
  nikolai : Position
  denis : Position
  total : Nat

/-- The conditions of the problem -/
def initial_conditions : RecruitLine := {
  peter := { front := 50, behind := 0 },
  nikolai := { front := 100, behind := 0 },
  denis := { front := 170, behind := 0 },
  total := 0
}

/-- The condition after turning around -/
def turn_around_condition (line : RecruitLine) : Prop :=
  (line.peter.behind = 50 ∧ line.nikolai.behind = 100 ∧ line.denis.behind = 170) ∧
  ((4 * line.peter.front = line.nikolai.front ∧ line.peter.behind = 4 * line.nikolai.behind) ∨
   (4 * line.nikolai.front = line.denis.front ∧ line.nikolai.behind = 4 * line.denis.behind) ∨
   (4 * line.peter.front = line.denis.front ∧ line.peter.behind = 4 * line.denis.behind))

/-- The theorem to prove -/
theorem recruit_line_total (line : RecruitLine) :
  turn_around_condition line →
  line.total = 211 :=
by sorry

end recruit_line_total_l3818_381889


namespace chinese_team_gold_medal_probability_l3818_381847

theorem chinese_team_gold_medal_probability 
  (prob_A prob_B : ℚ)
  (h1 : prob_A = 3 / 7)
  (h2 : prob_B = 1 / 4)
  (h3 : ∀ x y : ℚ, x + y = prob_A + prob_B → x ≤ prob_A ∧ y ≤ prob_B) :
  prob_A + prob_B = 19 / 28 := by
sorry

end chinese_team_gold_medal_probability_l3818_381847


namespace solve_linear_equation_l3818_381888

theorem solve_linear_equation (x : ℝ) : 5 * x + 3 = 10 * x - 22 → x = 5 := by
  sorry

end solve_linear_equation_l3818_381888


namespace binomial_expansion_properties_l3818_381825

theorem binomial_expansion_properties :
  let f := fun x => (2 * x + 1) ^ 4
  ∃ (a b c d e : ℤ),
    f x = a * x^4 + b * x^3 + c * x^2 + d * x + e ∧
    c = 24 ∧
    a + b + c + d + e = 81 :=
by sorry

end binomial_expansion_properties_l3818_381825


namespace complex_equation_solution_l3818_381878

theorem complex_equation_solution (i : ℂ) (n : ℝ) 
  (h1 : i * i = -1) 
  (h2 : (2 : ℂ) / (1 - i) = 1 + n * i) : 
  n = 1 := by
  sorry

end complex_equation_solution_l3818_381878


namespace hyperbola_focal_length_l3818_381819

-- Define the hyperbola parameters
def hyperbola_equation (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 20 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2 * x

-- Define the focal length calculation
def focal_length (a : ℝ) : ℝ := 2 * a

-- Theorem statement
theorem hyperbola_focal_length (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, hyperbola_equation x y a → asymptote_equation x y) :
  focal_length a = 10 := by
  sorry

end hyperbola_focal_length_l3818_381819


namespace transaction_yearly_loss_l3818_381803

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- Represents the financial transaction described in the problem -/
structure FinancialTransaction where
  borrowAmount : ℚ
  borrowRate : ℚ
  lendRate : ℚ
  timeInYears : ℚ

/-- Calculates the yearly loss in the given financial transaction -/
def yearlyLoss (transaction : FinancialTransaction) : ℚ :=
  let borrowInterest := simpleInterest transaction.borrowAmount transaction.borrowRate transaction.timeInYears
  let lendInterest := simpleInterest transaction.borrowAmount transaction.lendRate transaction.timeInYears
  (borrowInterest - lendInterest) / transaction.timeInYears

/-- Theorem stating that the yearly loss in the given transaction is 140 -/
theorem transaction_yearly_loss :
  let transaction : FinancialTransaction := {
    borrowAmount := 7000
    borrowRate := 4
    lendRate := 6
    timeInYears := 2
  }
  yearlyLoss transaction = 140 := by sorry

end transaction_yearly_loss_l3818_381803


namespace advantages_of_early_license_l3818_381864

-- Define the type for advantages
inductive Advantage
  | CostSavings
  | RentalFlexibility
  | EmploymentOpportunities

-- Define a function to check if an advantage applies to getting a license at 18
def is_advantage_at_18 (a : Advantage) : Prop :=
  match a with
  | Advantage.CostSavings => true
  | Advantage.RentalFlexibility => true
  | Advantage.EmploymentOpportunities => true

-- Define a function to check if an advantage applies to getting a license at 30
def is_advantage_at_30 (a : Advantage) : Prop :=
  match a with
  | Advantage.CostSavings => false
  | Advantage.RentalFlexibility => false
  | Advantage.EmploymentOpportunities => false

-- Theorem stating that there are at least three distinct advantages
-- of getting a license at 18 compared to 30
theorem advantages_of_early_license :
  ∃ (a b c : Advantage), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  is_advantage_at_18 a ∧ is_advantage_at_18 b ∧ is_advantage_at_18 c ∧
  ¬is_advantage_at_30 a ∧ ¬is_advantage_at_30 b ∧ ¬is_advantage_at_30 c :=
sorry

end advantages_of_early_license_l3818_381864


namespace gcd_of_ones_l3818_381840

theorem gcd_of_ones (m n : ℕ+) :
  Nat.gcd ((10^(m.val) - 1) / 9) ((10^(n.val) - 1) / 9) = (10^(Nat.gcd m.val n.val) - 1) / 9 := by
  sorry

end gcd_of_ones_l3818_381840


namespace intersection_union_eq_l3818_381849

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}

theorem intersection_union_eq : (A ∪ B) ∩ C = {0, 3, 4} := by sorry

end intersection_union_eq_l3818_381849


namespace sum_of_xy_l3818_381894

theorem sum_of_xy (x y : ℕ+) 
  (eq1 : 10 * x + y = 75)
  (eq2 : 10 * y + x = 57) : 
  x + y = 12 := by
sorry

end sum_of_xy_l3818_381894


namespace cyclist_speed_l3818_381886

/-- Proves that a cyclist's speed is 24 km/h given specific conditions -/
theorem cyclist_speed (hiker_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) 
  (hiker_speed_positive : 0 < hiker_speed)
  (cyclist_travel_time_positive : 0 < cyclist_travel_time)
  (hiker_catch_up_time_positive : 0 < hiker_catch_up_time)
  (hiker_speed_val : hiker_speed = 4)
  (cyclist_travel_time_val : cyclist_travel_time = 5 / 60)
  (hiker_catch_up_time_val : hiker_catch_up_time = 25 / 60) : 
  ∃ (cyclist_speed : ℝ), cyclist_speed = 24 := by
  sorry


end cyclist_speed_l3818_381886


namespace f_is_quadratic_l3818_381801

/-- A function f : ℝ → ℝ is quadratic if there exist constants a, b, c : ℝ 
    such that f(x) = ax² + bx + c for all x : ℝ, and a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

/-- The function f(x) = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f := by
  sorry


end f_is_quadratic_l3818_381801


namespace product_of_sines_l3818_381814

theorem product_of_sines : 
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) * 
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end product_of_sines_l3818_381814


namespace vacation_duration_l3818_381835

/-- Represents the vacation of a family -/
structure Vacation where
  total_days : ℕ
  rain_days : ℕ
  clear_afternoons : ℕ

/-- Theorem stating that given the conditions, the total number of days is 18 -/
theorem vacation_duration (v : Vacation) 
  (h1 : v.rain_days = 13)
  (h2 : v.clear_afternoons = 12)
  (h3 : v.rain_days + v.clear_afternoons ≤ v.total_days)
  (h4 : v.total_days ≤ v.rain_days + v.clear_afternoons + 1) :
  v.total_days = 18 := by
  sorry

#check vacation_duration

end vacation_duration_l3818_381835


namespace correct_time_is_two_five_and_five_elevenths_l3818_381854

/-- Represents a time between 2 and 3 o'clock --/
structure Time where
  hour : ℕ
  minute : ℚ
  h_hour : hour = 2
  h_minute : 0 ≤ minute ∧ minute < 60

/-- Converts a Time to minutes past 2:00 --/
def timeToMinutes (t : Time) : ℚ :=
  60 * (t.hour - 2) + t.minute

/-- Represents the misread time by swapping hour and minute hands --/
def misreadTime (t : Time) : ℚ :=
  60 * (t.minute / 5) + 5 * t.hour

theorem correct_time_is_two_five_and_five_elevenths (t : Time) :
  misreadTime t = timeToMinutes t - 55 →
  t.hour = 2 ∧ t.minute = 5 + 5 / 11 := by
  sorry

end correct_time_is_two_five_and_five_elevenths_l3818_381854


namespace train_problem_l3818_381893

/-- The speed of the freight train in km/h given the conditions of the train problem -/
def freight_train_speed : ℝ := by sorry

theorem train_problem (passenger_length freight_length : ℝ) (passing_time : ℝ) (speed_ratio : ℚ) :
  passenger_length = 200 →
  freight_length = 280 →
  passing_time = 18 →
  speed_ratio = 5 / 3 →
  freight_train_speed = 36 := by sorry

end train_problem_l3818_381893


namespace doughnut_cost_calculation_l3818_381845

theorem doughnut_cost_calculation (num_doughnuts : ℕ) (price_per_doughnut : ℚ) (profit : ℚ) :
  let total_revenue := num_doughnuts * price_per_doughnut
  let cost_of_ingredients := total_revenue - profit
  cost_of_ingredients = num_doughnuts * price_per_doughnut - profit :=
by sorry

-- Example usage with the given values
def dorothy_example : ℚ :=
  let num_doughnuts : ℕ := 25
  let price_per_doughnut : ℚ := 3
  let profit : ℚ := 22
  num_doughnuts * price_per_doughnut - profit

#eval dorothy_example -- This should evaluate to 53

end doughnut_cost_calculation_l3818_381845


namespace sqrt_7225_minus_55_cube_l3818_381853

theorem sqrt_7225_minus_55_cube (c d : ℕ) (hc : c > 0) (hd : d > 0) 
  (h : Real.sqrt 7225 - 55 = (Real.sqrt c - d)^3) : c + d = 19 := by
  sorry

end sqrt_7225_minus_55_cube_l3818_381853


namespace absolute_value_equality_l3818_381881

theorem absolute_value_equality (x : ℝ) : |x - 2| = |x + 3| → x = -1/2 := by
  sorry

end absolute_value_equality_l3818_381881


namespace A_subset_B_l3818_381897

def A : Set ℝ := {x | |x - 2| < 1}
def B : Set ℝ := {x | (x - 1) * (x - 4) < 0}

theorem A_subset_B : A ⊆ B := by sorry

end A_subset_B_l3818_381897


namespace divisibility_by_three_l3818_381850

theorem divisibility_by_three (n : ℕ+) : 
  (∃ k : ℤ, n = 6*k + 1 ∨ n = 6*k + 2) ↔ 
  (∃ m : ℤ, n * 2^(n : ℕ) + 1 = 3 * m) := by
sorry

end divisibility_by_three_l3818_381850


namespace number_ratio_proof_l3818_381813

theorem number_ratio_proof (N P : ℚ) (h1 : N = 280) (h2 : (1/5) * N + 7 = P - 7) :
  (P - 7) / N = 9 / 40 := by
  sorry

end number_ratio_proof_l3818_381813


namespace semicircle_bounded_rectangle_perimeter_l3818_381861

theorem semicircle_bounded_rectangle_perimeter :
  let rectangle_length : ℝ := 4 / π
  let rectangle_width : ℝ := 1 / π
  let long_side_arcs_perimeter : ℝ := 2 * π * rectangle_length / 2
  let short_side_arcs_perimeter : ℝ := π * rectangle_width
  long_side_arcs_perimeter + short_side_arcs_perimeter = 9 := by
  sorry

end semicircle_bounded_rectangle_perimeter_l3818_381861


namespace smooth_flow_probability_l3818_381844

def cable_capacities : List Nat := [1, 1, 2, 2, 3, 4]

def total_combinations : Nat := Nat.choose 6 3

def smooth_flow_combinations : Nat := 5

theorem smooth_flow_probability :
  (smooth_flow_combinations : ℚ) / total_combinations = 1 / 4 := by sorry

end smooth_flow_probability_l3818_381844


namespace neighbor_rolls_count_l3818_381871

/-- The number of gift wrap rolls Nellie needs to sell for the fundraiser -/
def total_rolls : ℕ := 45

/-- The number of rolls Nellie sold to her grandmother -/
def grandmother_rolls : ℕ := 1

/-- The number of rolls Nellie sold to her uncle -/
def uncle_rolls : ℕ := 10

/-- The number of rolls Nellie still needs to sell to reach her goal -/
def remaining_rolls : ℕ := 28

/-- The number of rolls Nellie sold to her neighbor -/
def neighbor_rolls : ℕ := total_rolls - remaining_rolls - grandmother_rolls - uncle_rolls

theorem neighbor_rolls_count : neighbor_rolls = 6 := by
  sorry

end neighbor_rolls_count_l3818_381871


namespace inequality_proof_l3818_381815

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  a / b + b / c + c / a + b / a + a / c + c / b + 6 ≥ 2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
by sorry

end inequality_proof_l3818_381815


namespace final_digit_is_two_l3818_381822

/-- Represents the state of the board with counts of zeros, ones, and twos -/
structure BoardState where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents an operation on the board -/
inductive Operation
  | ZeroOne   -- Erase 0 and 1, write 2
  | ZeroTwo   -- Erase 0 and 2, write 1
  | OneTwo    -- Erase 1 and 2, write 0

/-- Applies an operation to the board state -/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.ZeroOne => ⟨state.zeros - 1, state.ones - 1, state.twos + 1⟩
  | Operation.ZeroTwo => ⟨state.zeros - 1, state.ones + 1, state.twos - 1⟩
  | Operation.OneTwo => ⟨state.zeros + 1, state.ones - 1, state.twos - 1⟩

/-- Checks if the board state has only one digit remaining -/
def isFinalState (state : BoardState) : Bool :=
  (state.zeros + state.ones + state.twos = 1)

/-- Theorem: The final digit is always 2, regardless of the order of operations -/
theorem final_digit_is_two (initialState : BoardState) (ops : List Operation) :
  isFinalState (ops.foldl applyOperation initialState) →
  (ops.foldl applyOperation initialState).twos = 1 := by
  sorry

end final_digit_is_two_l3818_381822


namespace carlas_quadruple_batch_cans_l3818_381887

/-- Represents the number of cans of each ingredient in a normal batch of Carla's chili -/
structure ChiliBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Calculates the total number of cans in a batch -/
def totalCans (batch : ChiliBatch) : ℕ :=
  batch.chilis + batch.beans + batch.tomatoes

/-- Represents Carla's normal chili batch -/
def carlasNormalBatch : ChiliBatch :=
  { chilis := 1
  , beans := 2
  , tomatoes := 3 }  -- 50% more than beans: 2 * 1.5 = 3

/-- The number of times Carla is multiplying her normal batch -/
def batchMultiplier : ℕ := 4

/-- Theorem: The total number of cans for Carla's quadruple batch is 24 -/
theorem carlas_quadruple_batch_cans : 
  totalCans carlasNormalBatch * batchMultiplier = 24 := by
  sorry


end carlas_quadruple_batch_cans_l3818_381887


namespace component_probability_l3818_381880

theorem component_probability (p : ℝ) : 
  p ∈ Set.Icc 0 1 →
  (1 - (1 - p)^3 = 0.999) →
  p = 0.9 := by
sorry

end component_probability_l3818_381880


namespace original_number_proof_l3818_381811

theorem original_number_proof : ∃ x : ℝ, 3 * (2 * x + 9) = 81 ∧ x = 9 := by
  sorry

end original_number_proof_l3818_381811


namespace stock_increase_factor_l3818_381855

def initial_investment : ℝ := 900
def num_stocks : ℕ := 3
def stock_c_loss_factor : ℝ := 0.5
def final_total_value : ℝ := 1350

theorem stock_increase_factor :
  let initial_per_stock := initial_investment / num_stocks
  let stock_c_final_value := initial_per_stock * stock_c_loss_factor
  let stock_ab_final_value := final_total_value - stock_c_final_value
  let stock_ab_initial_value := initial_per_stock * 2
  stock_ab_final_value / stock_ab_initial_value = 2 := by sorry

end stock_increase_factor_l3818_381855


namespace solution_set_of_system_l3818_381892

theorem solution_set_of_system : ∃! S : Set (ℝ × ℝ),
  S = {(-1, 2), (2, -1), (-2, 7)} ∧
  ∀ (x y : ℝ), (x, y) ∈ S ↔ 
    (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ 
    (y - x + 1 = x^2 - 3*x) ∧ 
    (x ≠ 0) ∧ 
    (x ≠ 3) := by
  sorry

end solution_set_of_system_l3818_381892


namespace trig_identity_l3818_381876

theorem trig_identity :
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (47 * π / 180) * Real.cos (103 * π / 180) = 1/2 := by
  sorry

end trig_identity_l3818_381876


namespace play_school_kids_l3818_381872

/-- The number of kids in a play school -/
def total_kids (white : ℕ) (yellow : ℕ) (both : ℕ) : ℕ :=
  white + yellow - both

/-- Theorem: The total number of kids in the play school is 35 -/
theorem play_school_kids : total_kids 26 28 19 = 35 := by
  sorry

end play_school_kids_l3818_381872


namespace charge_difference_l3818_381830

/-- The charge for a single color copy at print shop X -/
def charge_X : ℚ := 1.25

/-- The charge for a single color copy at print shop Y -/
def charge_Y : ℚ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 80

/-- The theorem stating the difference in charges between print shops Y and X for 80 color copies -/
theorem charge_difference : (num_copies : ℚ) * charge_Y - (num_copies : ℚ) * charge_X = 120 := by
  sorry

end charge_difference_l3818_381830


namespace max_value_cosine_sine_l3818_381808

theorem max_value_cosine_sine (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (max : Real), max = (4 * Real.sqrt 3) / 9 ∧
    ∀ x, 0 < x ∧ x < π →
      Real.cos (x / 2) * (1 + Real.sin x) ≤ max ∧
      ∃ y, 0 < y ∧ y < π ∧ Real.cos (y / 2) * (1 + Real.sin y) = max :=
by sorry

end max_value_cosine_sine_l3818_381808


namespace external_tangent_intercept_l3818_381867

/-- Definition of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Function to check if a line is a common external tangent to two circles -/
def isCommonExternalTangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

theorem external_tangent_intercept : 
  let c1 : Circle := { center := (2, 4), radius := 4 }
  let c2 : Circle := { center := (14, 9), radius := 9 }
  ∃ l : Line, l.slope > 0 ∧ isCommonExternalTangent l c1 c2 ∧ l.intercept = 912 / 119 :=
sorry

end external_tangent_intercept_l3818_381867


namespace hyperbola_eccentricity_l3818_381817

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and one of its asymptotes is y = √2 x, prove that the eccentricity of C is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_l3818_381817


namespace solution_characterization_l3818_381866

/-- The set of solutions to the equation (n+1)^k = n! + 1 for natural numbers n and k -/
def SolutionSet : Set (ℕ × ℕ) :=
  {(1, 1), (2, 1), (4, 2)}

/-- The equation (n+1)^k = n! + 1 -/
def EquationHolds (n k : ℕ) : Prop :=
  (n + 1) ^ k = Nat.factorial n + 1

theorem solution_characterization :
  ∀ (n k : ℕ), EquationHolds n k ↔ (n, k) ∈ SolutionSet := by
  sorry

#check solution_characterization

end solution_characterization_l3818_381866


namespace partial_fraction_decomposition_product_l3818_381896

theorem partial_fraction_decomposition_product (A B C : ℝ) : 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 3 → 
    (x^2 - 19) / (x^3 - 2*x^2 - 5*x + 6) = A / (x - 1) + B / (x + 2) + C / (x - 3)) →
  A * B * C = 3 := by
sorry

end partial_fraction_decomposition_product_l3818_381896


namespace tangent_slope_constraint_implies_a_range_l3818_381838

theorem tangent_slope_constraint_implies_a_range
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = -x^3 + a*x^2 + b)
  (h2 : ∀ x, (deriv f x) < 1) :
  -Real.sqrt 3 < a ∧ a < Real.sqrt 3 :=
sorry

end tangent_slope_constraint_implies_a_range_l3818_381838


namespace race_result_theorem_l3818_381805

-- Define the girls
inductive Girl : Type
  | Anna : Girl
  | Bella : Girl
  | Csilla : Girl
  | Dora : Girl

-- Define the positions
inductive Position : Type
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

def race_result : Girl → Position := sorry

-- Define the statements
def anna_statement : Prop := race_result Girl.Anna ≠ Position.First ∧ race_result Girl.Anna ≠ Position.Fourth
def bella_statement : Prop := race_result Girl.Bella ≠ Position.First
def csilla_statement : Prop := race_result Girl.Csilla = Position.First
def dora_statement : Prop := race_result Girl.Dora = Position.Fourth

-- Define the condition that three statements are true and one is false
def statements_condition : Prop :=
  (anna_statement ∧ bella_statement ∧ csilla_statement ∧ ¬dora_statement) ∨
  (anna_statement ∧ bella_statement ∧ ¬csilla_statement ∧ dora_statement) ∨
  (anna_statement ∧ ¬bella_statement ∧ csilla_statement ∧ dora_statement) ∨
  (¬anna_statement ∧ bella_statement ∧ csilla_statement ∧ dora_statement)

-- Theorem to prove
theorem race_result_theorem :
  statements_condition →
  (¬dora_statement ∧ race_result Girl.Csilla = Position.First) := by
  sorry

end race_result_theorem_l3818_381805


namespace prove_age_difference_l3818_381865

def age_difference (freyja_age eli_age sarah_age kaylin_age : ℕ) : Prop :=
  freyja_age = 10 ∧
  eli_age = freyja_age + 9 ∧
  sarah_age = 2 * eli_age ∧
  kaylin_age = 33 ∧
  sarah_age - kaylin_age = 5

theorem prove_age_difference :
  ∃ (freyja_age eli_age sarah_age kaylin_age : ℕ),
    age_difference freyja_age eli_age sarah_age kaylin_age :=
by
  sorry

end prove_age_difference_l3818_381865


namespace triangle_area_ordering_l3818_381806

/-- The area of the first triangle -/
def m : ℚ := 15/2

/-- The area of the second triangle -/
def n : ℚ := 13/2

/-- The area of the third triangle -/
def p : ℚ := 7

/-- The side length of the square -/
def square_side : ℚ := 4

/-- The area of the square -/
def square_area : ℚ := square_side * square_side

/-- Theorem stating that the areas of the triangles satisfy n < p < m -/
theorem triangle_area_ordering : n < p ∧ p < m := by
  sorry

end triangle_area_ordering_l3818_381806


namespace age_difference_richard_david_l3818_381812

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrothersAges) : Prop :=
  ages.david > ages.scott ∧
  ages.richard > ages.david ∧
  ages.david = ages.scott + 8 ∧
  ages.david = 14 ∧
  ages.richard + 8 = 2 * (ages.scott + 8)

/-- The theorem to be proved -/
theorem age_difference_richard_david (ages : BrothersAges) :
  problem_conditions ages → ages.richard - ages.david = 6 := by
  sorry


end age_difference_richard_david_l3818_381812


namespace largest_among_four_l3818_381804

theorem largest_among_four (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  b > (1/2 : ℝ) ∧ b > 2*a*b ∧ b > a^2 + b^2 := by
  sorry

end largest_among_four_l3818_381804


namespace curtain_price_is_30_l3818_381810

/-- The cost of Emily's order, given the number of curtain pairs, wall prints, their prices, and installation cost. -/
def order_cost (curtain_pairs : ℕ) (curtain_price : ℝ) (wall_prints : ℕ) (print_price : ℝ) (installation : ℝ) : ℝ :=
  curtain_pairs * curtain_price + wall_prints * print_price + installation

/-- Theorem stating that the cost of each pair of curtains is $30.00 -/
theorem curtain_price_is_30 :
  ∃ (curtain_price : ℝ),
    order_cost 2 curtain_price 9 15 50 = 245 ∧ curtain_price = 30 := by
  sorry

#check curtain_price_is_30

end curtain_price_is_30_l3818_381810


namespace sequence_common_difference_l3818_381898

theorem sequence_common_difference (k x a : ℝ) : 
  (20 + k = x) ∧ (50 + k = a * x) ∧ (100 + k = a^2 * x) → a = 5/3 := by
  sorry

end sequence_common_difference_l3818_381898


namespace jackson_williams_money_ratio_l3818_381828

/-- Given that Jackson and Williams have a total of $150 and Jackson has $125,
    prove that the ratio of Jackson's money to Williams' money is 5:1 -/
theorem jackson_williams_money_ratio :
  ∀ (jackson_money williams_money : ℝ),
    jackson_money + williams_money = 150 →
    jackson_money = 125 →
    jackson_money / williams_money = 5 := by
  sorry

end jackson_williams_money_ratio_l3818_381828


namespace arithmetic_calculation_l3818_381857

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 3 * 2 = 192 := by
  sorry

end arithmetic_calculation_l3818_381857


namespace bobs_family_adults_l3818_381843

/-- The number of adults in Bob's family -/
def num_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) : ℕ :=
  (total_apples - num_children * apples_per_child) / apples_per_adult

/-- Theorem stating that the number of adults in Bob's family is 40 -/
theorem bobs_family_adults :
  num_adults 450 33 10 3 = 40 := by
  sorry

#eval num_adults 450 33 10 3

end bobs_family_adults_l3818_381843


namespace polynomial_sum_l3818_381839

def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), f a b x = g c d x) ∧
  (f a b (-a/2) = g c d (-c/2)) ∧
  (g c d (-a/2) = 0) ∧
  (f a b (-c/2) = 0) ∧
  (f a b 50 = -200) ∧
  (g c d 50 = -200) →
  a + c = -200 := by sorry

end polynomial_sum_l3818_381839


namespace geometric_sequence_sum_l3818_381837

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/2
  let n : ℕ := 7
  geometric_sum a r n = 127/256 := by
sorry

end geometric_sequence_sum_l3818_381837


namespace limit_of_sequence_l3818_381848

def a (n : ℕ) : ℚ := (4 * n - 3) / (2 * n + 1)

theorem limit_of_sequence : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end limit_of_sequence_l3818_381848


namespace lesser_fraction_l3818_381826

theorem lesser_fraction (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 13/14) (h_product : x * y = 1/5) : 
  min x y = 87/700 := by
sorry

end lesser_fraction_l3818_381826


namespace cost_for_five_point_five_kg_l3818_381807

-- Define the relationship between strawberries picked and cost
def strawberry_cost (x : ℝ) : ℝ := 16 * x + 2.5

-- Theorem stating the cost for 5.5kg of strawberries
theorem cost_for_five_point_five_kg :
  strawberry_cost 5.5 = 90.5 := by
  sorry

end cost_for_five_point_five_kg_l3818_381807


namespace hilt_bee_count_l3818_381846

/-- The number of bees Mrs. Hilt saw on the first day -/
def first_day_bees : ℕ := 144

/-- The multiplier for the number of bees on the second day -/
def day_two_multiplier : ℕ := 3

/-- The number of bees Mrs. Hilt saw on the second day -/
def second_day_bees : ℕ := first_day_bees * day_two_multiplier

/-- Theorem stating that Mrs. Hilt saw 432 bees on the second day -/
theorem hilt_bee_count : second_day_bees = 432 := by
  sorry

end hilt_bee_count_l3818_381846


namespace exactly_one_correct_l3818_381836

-- Define the four propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)

def proposition2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  ¬p ↔ ∀ x : ℝ, x^2 + 2*x > 0

def proposition3 : Prop :=
  ¬(∀ x : ℝ, x^2 - 2*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 < 0)

def proposition4 : Prop :=
  ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that exactly one proposition is correct
theorem exactly_one_correct : 
  (proposition2 ∧ ¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4) :=
sorry

end exactly_one_correct_l3818_381836


namespace f_is_linear_equation_one_var_l3818_381823

/-- A linear equation with one variable is of the form ax + b = 0, where a and b are real numbers and a ≠ 0 -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f(x) = x - 1 -/
def f (x : ℝ) : ℝ := x - 1

theorem f_is_linear_equation_one_var :
  is_linear_equation_one_var f :=
sorry

end f_is_linear_equation_one_var_l3818_381823


namespace quadratic_root_value_l3818_381868

theorem quadratic_root_value (m : ℝ) : 
  m^2 - m - 2 = 0 → 2*m^2 - 2*m + 2022 = 2026 := by
  sorry

end quadratic_root_value_l3818_381868


namespace pyramid_base_edge_length_l3818_381802

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Slant height from apex to midpoint of a base side -/
  slant_height : ℝ
  /-- Radius of the hemisphere -/
  hemisphere_radius : ℝ

/-- Theorem stating the edge-length of the pyramid's base -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.height = 8)
  (h2 : p.slant_height = 10)
  (h3 : p.hemisphere_radius = 3) :
  ∃ (edge_length : ℝ), edge_length = 6 * Real.sqrt 2 := by
  sorry

end pyramid_base_edge_length_l3818_381802


namespace inequality_solution_l3818_381820

theorem inequality_solution (x : ℝ) : (x + 1) / x > 1 ↔ x > 0 := by
  sorry

end inequality_solution_l3818_381820


namespace composition_result_l3818_381832

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 4 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 2

-- State the theorem
theorem composition_result (c d : ℝ) :
  (∀ x, f c (g c x) = 12 * x + d) → d = 11 :=
by
  sorry

end composition_result_l3818_381832


namespace magnitude_of_sum_l3818_381841

def a : ℝ × ℝ := (2, 3)
def b (m : ℝ) : ℝ × ℝ := (m, -6)

theorem magnitude_of_sum (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  Real.sqrt ((2 * a.1 + (b m).1)^2 + (2 * a.2 + (b m).2)^2) = 13 :=
by sorry

end magnitude_of_sum_l3818_381841


namespace negation_of_proposition_negation_of_specific_proposition_l3818_381882

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬∀ x : ℝ, x^2 - 2*x > 0) ↔ (∃ x : ℝ, x^2 - 2*x ≤ 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l3818_381882
