import Mathlib

namespace complex_number_equality_l3348_334801

theorem complex_number_equality : Complex.I * (1 - Complex.I)^2 = 2 := by
  sorry

end complex_number_equality_l3348_334801


namespace inequality_solution_l3348_334871

theorem inequality_solution (x : ℝ) : 
  x ≠ 2 → (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ x ∈ Set.Ici 1 ∩ Set.Iio 2 ∪ Set.Ioi (32/7) :=
sorry

end inequality_solution_l3348_334871


namespace monday_sales_calculation_l3348_334838

def total_stock : ℕ := 1300
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 69.07692307692308

theorem monday_sales_calculation :
  ∃ (monday_sales : ℕ),
    monday_sales = total_stock - tuesday_sales - wednesday_sales - thursday_sales - friday_sales -
      (unsold_percentage / 100 * total_stock).floor ∧
    monday_sales = 75 := by
  sorry

end monday_sales_calculation_l3348_334838


namespace arrange_balls_theorem_l3348_334854

/-- The number of ways to arrange balls of different types in a row -/
def arrangeMultisetBalls (n₁ n₂ n₃ : ℕ) : ℕ :=
  Nat.factorial (n₁ + n₂ + n₃) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃)

/-- Theorem stating that arranging 5 basketballs, 3 volleyballs, and 2 footballs yields 2520 ways -/
theorem arrange_balls_theorem : arrangeMultisetBalls 5 3 2 = 2520 := by
  sorry

end arrange_balls_theorem_l3348_334854


namespace max_product_under_constraint_l3348_334836

theorem max_product_under_constraint :
  ∀ x y : ℕ, 27 * x + 35 * y ≤ 1000 →
  x * y ≤ 252 ∧ ∃ a b : ℕ, 27 * a + 35 * b ≤ 1000 ∧ a * b = 252 := by
sorry

end max_product_under_constraint_l3348_334836


namespace square_minus_two_x_plus_one_l3348_334813

theorem square_minus_two_x_plus_one (x : ℝ) : x = Real.sqrt 3 + 1 → x^2 - 2*x + 1 = 3 := by
  sorry

end square_minus_two_x_plus_one_l3348_334813


namespace fraction_power_equality_l3348_334819

theorem fraction_power_equality (x y : ℚ) 
  (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3 : ℚ) * x^7 * y^8 = 2/5 := by
  sorry

end fraction_power_equality_l3348_334819


namespace unique_intersection_l3348_334851

/-- The value of m for which the line x = m intersects the parabola x = -3y² - 4y + 7 at exactly one point -/
def intersection_point : ℚ := 25 / 3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

theorem unique_intersection :
  ∀ m : ℝ, (∃! y : ℝ, parabola y = m) ↔ m = intersection_point := by sorry

end unique_intersection_l3348_334851


namespace exists_arithmetic_not_m_sequence_l3348_334869

/-- Definition of "M sequence" -/
def is_m_sequence (b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  (∀ n, b n < b (n + 1)) ∧ 
  (∀ n, c n < c (n + 1)) ∧
  (∀ n, ∃ m, c n ≤ b m ∧ b m ≤ c (n + 1))

/-- Arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) - a n = d

/-- Partial sum sequence -/
def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sum a n + a (n + 1)

/-- Main theorem -/
theorem exists_arithmetic_not_m_sequence :
  ∃ a : ℕ → ℝ, is_arithmetic a ∧ ¬(is_m_sequence a (partial_sum a)) := by
  sorry

end exists_arithmetic_not_m_sequence_l3348_334869


namespace average_of_numbers_l3348_334823

def numbers : List ℝ := [12, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125830.7 := by sorry

end average_of_numbers_l3348_334823


namespace age_problem_solution_l3348_334887

/-- Represents the ages of two people --/
structure Ages where
  your_age : ℕ
  my_age : ℕ

/-- The conditions of the age problem --/
def age_conditions (ages : Ages) : Prop :=
  -- Condition 1: I am twice as old as you were when I was as old as you are now
  ages.your_age = 2 * (2 * ages.my_age - ages.your_age) ∧
  -- Condition 2: When you are as old as I am now, the sum of our ages will be 140 years
  ages.my_age + (2 * ages.my_age - ages.your_age) = 140

/-- The theorem stating the solution to the age problem --/
theorem age_problem_solution :
  ∃ (ages : Ages), age_conditions ages ∧ ages.your_age = 112 ∧ ages.my_age = 84 := by
  sorry

end age_problem_solution_l3348_334887


namespace hospital_age_l3348_334848

/-- Proves that the hospital's current age is 40 years, given Grant's current age and the relationship between their ages in 5 years. -/
theorem hospital_age (grant_current_age : ℕ) (hospital_age : ℕ) : 
  grant_current_age = 25 →
  grant_current_age + 5 = 2 / 3 * (hospital_age + 5) →
  hospital_age = 40 := by
  sorry

end hospital_age_l3348_334848


namespace hyperbola_equation_l3348_334882

-- Define the hyperbola C
structure Hyperbola where
  -- The equation of the hyperbola in the form ax² + by² = c
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions of the problem
def hyperbola_conditions (C : Hyperbola) : Prop :=
  -- Center at origin (implied by the standard form)
  -- Asymptote y = √2x
  C.a / C.b = -2 ∧
  -- Point P(2√2, -√2) lies on C
  C.a * (2 * Real.sqrt 2)^2 + C.b * (-Real.sqrt 2)^2 = C.c

-- The theorem to prove
theorem hyperbola_equation (C : Hyperbola) :
  hyperbola_conditions C →
  C.a = 1/7 ∧ C.b = -1/14 ∧ C.c = 1 :=
by sorry

end hyperbola_equation_l3348_334882


namespace eight_bead_necklace_arrangements_l3348_334890

/-- The number of distinct arrangements of beads on a necklace with specific properties. -/
def necklaceArrangements (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- Theorem stating that the number of distinct arrangements of 8 beads
    on a necklace with a fixed pendant and reflectional symmetry is 8! / 2. -/
theorem eight_bead_necklace_arrangements :
  necklaceArrangements 8 = 20160 := by
  sorry

#eval necklaceArrangements 8

end eight_bead_necklace_arrangements_l3348_334890


namespace least_coins_ten_coins_coins_in_wallet_l3348_334815

theorem least_coins (n : ℕ) : (n % 7 = 3 ∧ n % 4 = 2) → n ≥ 10 :=
by sorry

theorem ten_coins : (10 % 7 = 3) ∧ (10 % 4 = 2) :=
by sorry

theorem coins_in_wallet : ∃ (n : ℕ), n % 7 = 3 ∧ n % 4 = 2 ∧ ∀ (m : ℕ), (m % 7 = 3 ∧ m % 4 = 2) → m ≥ n :=
by sorry

end least_coins_ten_coins_coins_in_wallet_l3348_334815


namespace triangle_ratio_l3348_334859

/-- Given an acute triangle ABC and a point D inside it, 
    if ∠ADB = ∠ACB + 90° and AC · BD = AD · BC, 
    then (AB · CD) / (AC · BD) = √2 -/
theorem triangle_ratio (A B C D : ℝ × ℝ) : 
  let triangle_is_acute : Bool := sorry
  let D_inside_triangle : Bool := sorry
  let angle_ADB : ℝ := sorry
  let angle_ACB : ℝ := sorry
  let AC : ℝ := sorry
  let BD : ℝ := sorry
  let AD : ℝ := sorry
  let BC : ℝ := sorry
  let AB : ℝ := sorry
  let CD : ℝ := sorry
  triangle_is_acute ∧ 
  D_inside_triangle ∧
  angle_ADB = angle_ACB + π/2 ∧ 
  AC * BD = AD * BC →
  (AB * CD) / (AC * BD) = Real.sqrt 2 := by
sorry

end triangle_ratio_l3348_334859


namespace converse_A_false_others_true_l3348_334883

-- Define the basic geometric concepts
structure Triangle where
  angles : Fin 3 → ℝ
  sides : Fin 3 → ℝ

def is_congruent (t1 t2 : Triangle) : Prop := sorry

def is_right_triangle (t : Triangle) : Prop := sorry

def is_equilateral (t : Triangle) : Prop := sorry

def are_complementary (a b : ℝ) : Prop := sorry

-- Define the statements and their converses
def statement_A (t1 t2 : Triangle) : Prop :=
  is_congruent t1 t2 → ∀ i : Fin 3, t1.angles i = t2.angles i

def converse_A (t1 t2 : Triangle) : Prop :=
  (∀ i : Fin 3, t1.angles i = t2.angles i) → is_congruent t1 t2

def statement_B (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.angles i = t.angles j) → (∀ i j : Fin 3, t.sides i = t.sides j)

def converse_B (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.sides i = t.sides j) → (∀ i j : Fin 3, t.angles i = t.angles j)

def statement_C (t : Triangle) : Prop :=
  is_right_triangle t → are_complementary (t.angles 0) (t.angles 1)

def converse_C (t : Triangle) : Prop :=
  are_complementary (t.angles 0) (t.angles 1) → is_right_triangle t

def statement_D (t : Triangle) : Prop :=
  is_equilateral t → (∀ i j : Fin 3, t.angles i = t.angles j)

def converse_D (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.angles i = t.angles j) → is_equilateral t

-- Main theorem
theorem converse_A_false_others_true :
  (∃ t1 t2 : Triangle, converse_A t1 t2 = false) ∧
  (∀ t : Triangle, converse_B t = true) ∧
  (∀ t : Triangle, converse_C t = true) ∧
  (∀ t : Triangle, converse_D t = true) := by sorry

end converse_A_false_others_true_l3348_334883


namespace video_votes_l3348_334855

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 140 ∧ like_percentage = 70 / 100 → 
  ∃ (total_votes : ℕ), 
    (like_percentage : ℚ) * total_votes - (1 - like_percentage) * total_votes = score ∧
    total_votes = 350 := by
  sorry

end video_votes_l3348_334855


namespace sarah_cans_yesterday_l3348_334802

theorem sarah_cans_yesterday (sarah_yesterday : ℕ) 
  (h1 : sarah_yesterday + (sarah_yesterday + 30) = 40 + 70 + 20) : 
  sarah_yesterday = 50 := by
  sorry

end sarah_cans_yesterday_l3348_334802


namespace shopkeeper_profit_l3348_334891

/-- Proves that if a shopkeeper sells an article with a 4% discount and earns a 20% profit,
    then the profit percentage without discount would be 25%. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_pos : 0 < cost_price) :
  let discount_rate : ℝ := 0.04
  let profit_rate_with_discount : ℝ := 0.20
  let selling_price_with_discount : ℝ := cost_price * (1 + profit_rate_with_discount)
  let marked_price : ℝ := selling_price_with_discount / (1 - discount_rate)
  let profit_rate_without_discount : ℝ := (marked_price - cost_price) / cost_price
  profit_rate_without_discount = 0.25 := by
sorry

end shopkeeper_profit_l3348_334891


namespace abs_fraction_inequality_l3348_334850

theorem abs_fraction_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end abs_fraction_inequality_l3348_334850


namespace exist_irrational_with_natural_power_l3348_334875

theorem exist_irrational_with_natural_power : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Irrational a ∧ Irrational b ∧ ∃ (n : ℕ), a^b = n :=
sorry

end exist_irrational_with_natural_power_l3348_334875


namespace arcade_tickets_l3348_334834

theorem arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) :
  initial_tickets ≥ spent_tickets →
  initial_tickets - spent_tickets + additional_tickets =
    initial_tickets + additional_tickets - spent_tickets :=
by sorry

end arcade_tickets_l3348_334834


namespace function_inequality_l3348_334863

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, x * (deriv (deriv f) x) + f x > 0

theorem function_inequality {f : ℝ → ℝ} (hf : Differentiable ℝ f) 
    (hf' : Differentiable ℝ (deriv f)) (hcond : SatisfiesCondition f) 
    {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) : 
    a * f a > b * f b := by
  sorry

end function_inequality_l3348_334863


namespace cab_driver_income_l3348_334832

theorem cab_driver_income (day1 day3 day4 day5 average : ℕ) 
  (h1 : day1 = 300)
  (h2 : day3 = 750)
  (h3 : day4 = 200)
  (h4 : day5 = 600)
  (h5 : average = 400)
  (h6 : (day1 + day3 + day4 + day5 + (5 * average - (day1 + day3 + day4 + day5))) / 5 = average) :
  5 * average - (day1 + day3 + day4 + day5) = 150 := by
  sorry

end cab_driver_income_l3348_334832


namespace complement_of_union_l3348_334874

open Set

def U : Finset ℕ := {1, 2, 3, 4}
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {4} := by
  sorry

end complement_of_union_l3348_334874


namespace governor_addresses_l3348_334845

theorem governor_addresses (sandoval hawkins sloan : ℕ) : 
  hawkins = sandoval / 2 →
  sloan = sandoval + 10 →
  sandoval + hawkins + sloan = 40 →
  sandoval = 12 := by
sorry

end governor_addresses_l3348_334845


namespace kitty_window_cleaning_time_l3348_334804

/-- Represents the weekly cleaning time for various tasks in minutes -/
structure CleaningTime where
  pickup : ℕ
  vacuum : ℕ
  dust : ℕ
  window : ℕ

/-- Calculates the total cleaning time for a given number of weeks -/
def totalCleaningTime (ct : CleaningTime) (weeks : ℕ) : ℕ :=
  (ct.pickup + ct.vacuum + ct.dust + ct.window) * weeks

/-- The main theorem about Kitty's cleaning time -/
theorem kitty_window_cleaning_time :
  ∀ (ct : CleaningTime),
    ct.pickup = 5 →
    ct.vacuum = 20 →
    ct.dust = 10 →
    totalCleaningTime ct 4 = 200 →
    ct.window = 15 :=
by sorry

end kitty_window_cleaning_time_l3348_334804


namespace factor_difference_of_squares_l3348_334812

theorem factor_difference_of_squares (x : ℝ) : 
  81 - 16 * (x - 1)^2 = (13 - 4*x) * (5 + 4*x) := by
  sorry

end factor_difference_of_squares_l3348_334812


namespace negation_of_existence_proposition_l3348_334806

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end negation_of_existence_proposition_l3348_334806


namespace ben_time_to_school_l3348_334856

/-- Represents the walking parameters of a person -/
structure WalkingParams where
  steps_per_minute : ℕ
  step_length : ℕ
  time_to_school : ℕ

/-- Calculates the time it takes for a person to walk to school given their walking parameters and the distance to school -/
def time_to_school (params : WalkingParams) (distance : ℕ) : ℚ :=
  distance / (params.steps_per_minute * params.step_length)

theorem ben_time_to_school 
  (amy : WalkingParams)
  (ben : WalkingParams)
  (h1 : amy.steps_per_minute = 80)
  (h2 : amy.step_length = 70)
  (h3 : amy.time_to_school = 20)
  (h4 : ben.steps_per_minute = 120)
  (h5 : ben.step_length = 50) :
  time_to_school ben (amy.steps_per_minute * amy.step_length * amy.time_to_school) = 56/3 := by
  sorry

end ben_time_to_school_l3348_334856


namespace angela_unfinished_problems_l3348_334852

theorem angela_unfinished_problems (total : Nat) (martha : Nat) (jenna : Nat) (mark : Nat)
  (h1 : total = 20)
  (h2 : martha = 2)
  (h3 : jenna = 4 * martha - 2)
  (h4 : mark = jenna / 2)
  (h5 : martha + jenna + mark ≤ total) :
  total - (martha + jenna + mark) = 9 := by
sorry

end angela_unfinished_problems_l3348_334852


namespace cylinder_height_comparison_l3348_334898

/-- Theorem: Comparing cylinder heights with equal volumes and different radii -/
theorem cylinder_height_comparison (r₁ h₁ r₂ h₂ : ℝ) 
  (volume_eq : r₁ ^ 2 * h₁ = r₂ ^ 2 * h₂)
  (radius_relation : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
sorry

end cylinder_height_comparison_l3348_334898


namespace alik_collection_l3348_334827

theorem alik_collection (badges bracelets : ℕ) (n : ℚ) : 
  badges > bracelets →
  badges + n * bracelets = 100 →
  n * badges + bracelets = 101 →
  ((badges = 34 ∧ bracelets = 33) ∨ (badges = 66 ∧ bracelets = 33)) :=
by sorry

end alik_collection_l3348_334827


namespace area_of_inscribed_square_l3348_334825

/-- A right triangle with an inscribed square -/
structure RightTriangleWithInscribedSquare where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Side length of the inscribed square BCFE -/
  x : ℝ
  /-- The inscribed square's side is perpendicular to both legs of the right triangle -/
  perpendicular : True
  /-- The inscribed square touches both legs of the right triangle -/
  touches_legs : True

/-- Theorem: Area of inscribed square in right triangle -/
theorem area_of_inscribed_square 
  (triangle : RightTriangleWithInscribedSquare) 
  (h1 : triangle.ab = 36)
  (h2 : triangle.cd = 64) :
  triangle.x^2 = 2304 := by
  sorry

end area_of_inscribed_square_l3348_334825


namespace quadratic_max_value_l3348_334824

/-- Given a quadratic function f(x) = -x^2 + 4x + a on the interval [0, 1] 
    with a minimum value of -2, prove that its maximum value is 1. -/
theorem quadratic_max_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = -x^2 + 4*x + a) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f x ≤ f y) →
  (∃ x ∈ Set.Icc 0 1, f x = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x) →
  (∃ x ∈ Set.Icc 0 1, f x = 1) :=
by sorry

end quadratic_max_value_l3348_334824


namespace veranda_width_l3348_334886

/-- Proves that the width of a veranda surrounding a 20 m × 12 m rectangular room is 2 m,
    given that the area of the veranda is 144 m². -/
theorem veranda_width (room_length : ℝ) (room_width : ℝ) (veranda_area : ℝ) :
  room_length = 20 →
  room_width = 12 →
  veranda_area = 144 →
  ∃ w : ℝ, w > 0 ∧ (room_length + 2*w) * (room_width + 2*w) - room_length * room_width = veranda_area ∧ w = 2 :=
by sorry

end veranda_width_l3348_334886


namespace connie_marbles_proof_l3348_334829

/-- The number of marbles Connie had initially -/
def initial_marbles : ℕ := 2856

/-- The number of marbles Connie had after losing half -/
def marbles_after_loss : ℕ := initial_marbles / 2

/-- The number of marbles Connie had after giving away 2/3 of the remaining marbles -/
def final_marbles : ℕ := 476

theorem connie_marbles_proof : 
  initial_marbles = 2856 ∧ 
  marbles_after_loss = initial_marbles / 2 ∧
  final_marbles = marbles_after_loss / 3 ∧
  final_marbles = 476 := by sorry

end connie_marbles_proof_l3348_334829


namespace total_fruits_picked_l3348_334837

theorem total_fruits_picked (joan_oranges sara_oranges carlos_oranges 
                             alyssa_pears ben_pears vanessa_pears 
                             tim_apples linda_apples : ℕ) 
                            (h1 : joan_oranges = 37)
                            (h2 : sara_oranges = 10)
                            (h3 : carlos_oranges = 25)
                            (h4 : alyssa_pears = 30)
                            (h5 : ben_pears = 40)
                            (h6 : vanessa_pears = 20)
                            (h7 : tim_apples = 15)
                            (h8 : linda_apples = 10) :
  joan_oranges + sara_oranges + carlos_oranges + 
  alyssa_pears + ben_pears + vanessa_pears + 
  tim_apples + linda_apples = 187 := by
  sorry

end total_fruits_picked_l3348_334837


namespace partner_profit_percentage_l3348_334817

theorem partner_profit_percentage (total_profit : ℝ) (majority_owner_percentage : ℝ) 
  (combined_amount : ℝ) (num_partners : ℕ) :
  total_profit = 80000 →
  majority_owner_percentage = 0.25 →
  combined_amount = 50000 →
  num_partners = 4 →
  let remaining_profit := total_profit * (1 - majority_owner_percentage)
  let partner_share := (combined_amount - total_profit * majority_owner_percentage) / 2
  (partner_share / remaining_profit) = 0.25 := by
  sorry

end partner_profit_percentage_l3348_334817


namespace stating_max_squares_correct_max_squares_1000_l3348_334876

/-- 
Represents the maximum number of squares that can be chosen on an m × n chessboard 
such that no three chosen squares have two in the same row and two in the same column.
-/
def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

/-- 
Theorem stating that max_squares gives the correct maximum number of squares
that can be chosen on an m × n chessboard under the given constraints.
-/
theorem max_squares_correct (m n : ℕ) (h : m ≤ n) :
  max_squares m n = 
    if m = 1 
    then n
    else m + n - 2 :=
by sorry

/-- 
Corollary for the specific case of a 1000 × 1000 chessboard.
-/
theorem max_squares_1000 : max_squares 1000 1000 = 1998 :=
by sorry

end stating_max_squares_correct_max_squares_1000_l3348_334876


namespace vector_perpendicular_condition_l3348_334843

/-- Given vectors a and b in ℝ², if a + b is perpendicular to b, then the second component of a is 9. -/
theorem vector_perpendicular_condition (m : ℝ) : 
  let a : ℝ × ℝ := (5, m)
  let b : ℝ × ℝ := (2, -2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → m = 9 := by
sorry

end vector_perpendicular_condition_l3348_334843


namespace pyramid_total_area_l3348_334826

-- Define the square side length and pyramid height
def squareSide : ℝ := 6
def pyramidHeight : ℝ := 4

-- Define the structure of our pyramid
structure Pyramid where
  base : ℝ
  height : ℝ

-- Define our specific pyramid
def ourPyramid : Pyramid :=
  { base := squareSide,
    height := pyramidHeight }

-- Theorem statement
theorem pyramid_total_area (p : Pyramid) (h : p = ourPyramid) :
  let diagonal := p.base * Real.sqrt 2
  let slantHeight := Real.sqrt (p.height^2 + (diagonal/2)^2)
  let triangleHeight := Real.sqrt (slantHeight^2 - (p.base/2)^2)
  let squareArea := p.base^2
  let triangleArea := 4 * (p.base * triangleHeight / 2)
  squareArea + triangleArea = 96 := by sorry

end pyramid_total_area_l3348_334826


namespace edwards_earnings_l3348_334840

/-- Edward's lawn mowing business earnings --/
theorem edwards_earnings (summer_earnings : ℕ) (supplies_cost : ℕ) (total_earnings : ℕ)
  (h1 : summer_earnings = 27)
  (h2 : supplies_cost = 5)
  (h3 : total_earnings = 24)
  : ∃ spring_earnings : ℕ,
    spring_earnings + (summer_earnings - supplies_cost) = total_earnings ∧
    spring_earnings = 2 := by
  sorry

end edwards_earnings_l3348_334840


namespace sandbox_cost_l3348_334897

/-- Calculates the cost of filling an L-shaped sandbox with sand -/
theorem sandbox_cost (short_length short_width short_depth long_length long_width long_depth sand_cost discount_threshold discount_rate : ℝ) :
  let short_volume := short_length * short_width * short_depth
  let long_volume := long_length * long_width * long_depth
  let total_volume := short_volume + long_volume
  let base_cost := total_volume * sand_cost
  let discounted_cost := if total_volume > discount_threshold then base_cost * (1 - discount_rate) else base_cost
  short_length = 3 ∧ 
  short_width = 2 ∧ 
  short_depth = 2 ∧ 
  long_length = 5 ∧ 
  long_width = 2 ∧ 
  long_depth = 2 ∧ 
  sand_cost = 3 ∧ 
  discount_threshold = 20 ∧ 
  discount_rate = 0.1 →
  discounted_cost = 86.4 := by
  sorry

end sandbox_cost_l3348_334897


namespace quadratic_root_transformation_l3348_334858

theorem quadratic_root_transformation (p q r : ℝ) (u v : ℝ) :
  (p * u^2 + q * u + r = 0) ∧ (p * v^2 + q * v + r = 0) →
  ((q * u + p)^2 - p * (q * u + p) + q * r = 0) ∧ ((q * v + p)^2 - p * (q * v + p) + q * r = 0) :=
by sorry

end quadratic_root_transformation_l3348_334858


namespace merchant_profit_percentage_l3348_334899

theorem merchant_profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) :
  (S - C) / C * 100 = 6.25 := by
  sorry

end merchant_profit_percentage_l3348_334899


namespace triangle_problem_l3348_334849

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = 2 →
  c = Real.sqrt 2 →
  Real.cos A = Real.sqrt 2 / 4 →
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c →
  a < b + c ∧ b < a + c ∧ c < a + b →
  -- Conclusions
  Real.sin C = Real.sqrt 7 / 4 ∧
  b = 1 ∧
  Real.cos (2 * A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry


end triangle_problem_l3348_334849


namespace triangle_properties_l3348_334885

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 4)
  (h2 : t.b = 6)
  (h3 : Real.sin t.A = Real.sin (2 * t.B)) :
  Real.cos t.B = 1/3 ∧ 
  1/2 * t.a * t.c * Real.sin t.B = 8 * Real.sqrt 2 := by
  sorry


end triangle_properties_l3348_334885


namespace nick_quarters_count_l3348_334839

-- Define the total number of quarters
def total_quarters : ℕ := 35

-- Define the fraction of state quarters
def state_quarter_fraction : ℚ := 2 / 5

-- Define the fraction of Pennsylvania quarters among state quarters
def pennsylvania_quarter_fraction : ℚ := 1 / 2

-- Define the number of Pennsylvania quarters
def pennsylvania_quarters : ℕ := 7

-- Theorem statement
theorem nick_quarters_count :
  (pennsylvania_quarter_fraction * state_quarter_fraction * total_quarters : ℚ) = pennsylvania_quarters :=
by sorry

end nick_quarters_count_l3348_334839


namespace smallest_sum_four_consecutive_composites_l3348_334867

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def fourConsecutiveComposites (n : ℕ) : Prop :=
  isComposite n ∧ isComposite (n + 1) ∧ isComposite (n + 2) ∧ isComposite (n + 3)

theorem smallest_sum_four_consecutive_composites :
  ∃ n : ℕ, fourConsecutiveComposites n ∧
    (∀ m : ℕ, fourConsecutiveComposites m → n ≤ m) ∧
    n + (n + 1) + (n + 2) + (n + 3) = 102 :=
sorry

end smallest_sum_four_consecutive_composites_l3348_334867


namespace smallest_three_digit_multiple_of_17_l3348_334877

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
sorry

end smallest_three_digit_multiple_of_17_l3348_334877


namespace shaded_area_sum_l3348_334879

/-- Given an equilateral triangle with side length 10 cm and an inscribed circle
    whose diameter is a side of the triangle, the sum of the areas of the two regions
    between the circle and the triangle can be expressed as a*π - b*√c,
    where a + b + c = 143/6. -/
theorem shaded_area_sum (a b c : ℝ) : 
  let side_length : ℝ := 10
  let triangle_area := side_length^2 * Real.sqrt 3 / 4
  let circle_radius := side_length / 2
  let sector_area := π * circle_radius^2 / 3
  let shaded_area := 2 * (sector_area - triangle_area / 2)
  (∃ (a b c : ℝ), shaded_area = a * π - b * Real.sqrt c ∧ a + b + c = 143/6) := by
  sorry

end shaded_area_sum_l3348_334879


namespace expand_expression_l3348_334872

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := by
  sorry

end expand_expression_l3348_334872


namespace no_valid_n_l3348_334846

theorem no_valid_n : ¬ ∃ (n : ℕ), n > 0 ∧ 
  (3*n - 3 + 2*n + 7 > 4*n + 6) ∧
  (3*n - 3 + 4*n + 6 > 2*n + 7) ∧
  (2*n + 7 + 4*n + 6 > 3*n - 3) ∧
  (2*n + 7 > 4*n + 6) ∧
  (4*n + 6 > 3*n - 3) :=
sorry

end no_valid_n_l3348_334846


namespace kids_at_camp_l3348_334821

theorem kids_at_camp (total : ℕ) (home : ℕ) (difference : ℕ) : 
  total = home + (home + difference) → 
  home = 668278 → 
  difference = 150780 → 
  home + difference = 409529 :=
by sorry

end kids_at_camp_l3348_334821


namespace intersection_not_empty_l3348_334805

theorem intersection_not_empty : ∃ (n : ℕ) (k : ℕ), n > 1 ∧ 2^n - n = k^2 := by sorry

end intersection_not_empty_l3348_334805


namespace y_intercepts_of_curve_l3348_334828

/-- The y-intercepts of the curve 3x + 5y^2 = 25 are (0, √5) and (0, -√5) -/
theorem y_intercepts_of_curve (x y : ℝ) :
  3*x + 5*y^2 = 25 ∧ x = 0 ↔ y = Real.sqrt 5 ∨ y = -Real.sqrt 5 := by
  sorry

end y_intercepts_of_curve_l3348_334828


namespace unique_number_with_three_prime_divisors_including_11_l3348_334847

theorem unique_number_with_three_prime_divisors_including_11 :
  ∀ (x n : ℕ), 
    x = 9^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
    11 ∣ x →
    x = 59048 := by
  sorry

end unique_number_with_three_prime_divisors_including_11_l3348_334847


namespace blackboard_division_l3348_334868

theorem blackboard_division : (96 : ℕ) / 8 = 12 := by
  sorry

end blackboard_division_l3348_334868


namespace point_inside_ellipse_l3348_334888

/-- A point A(a, 1) is inside the ellipse x²/4 + y²/2 = 1 if and only if -√2 < a < √2 -/
theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) ↔ (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) :=
by sorry

end point_inside_ellipse_l3348_334888


namespace production_average_proof_l3348_334800

/-- Calculates the new average daily production after adding a new day's production -/
def newAverageProduction (n : ℕ) (oldAverage : ℚ) (newProduction : ℚ) : ℚ :=
  ((n : ℚ) * oldAverage + newProduction) / ((n : ℚ) + 1)

theorem production_average_proof :
  let n : ℕ := 4
  let oldAverage : ℚ := 50
  let newProduction : ℚ := 90
  newAverageProduction n oldAverage newProduction = 58 := by
sorry

end production_average_proof_l3348_334800


namespace divisibility_by_five_l3348_334862

theorem divisibility_by_five (x y : ℕ) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x := by
  sorry

end divisibility_by_five_l3348_334862


namespace tan_90_degrees_undefined_l3348_334842

theorem tan_90_degrees_undefined :
  let θ : Real := 90 * Real.pi / 180  -- Convert 90 degrees to radians
  ∀ (tan sin cos : Real → Real),
    (∀ α, tan α = sin α / cos α) →    -- Definition of tangent
    sin θ = 1 →                       -- Given: sin 90° = 1
    cos θ = 0 →                       -- Given: cos 90° = 0
    ¬∃ (x : Real), tan θ = x          -- tan 90° is undefined
  := by sorry

end tan_90_degrees_undefined_l3348_334842


namespace regular_17gon_symmetry_sum_l3348_334809

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle for rotational symmetry in degrees -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ p : RegularPolygon 17,
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 649 / 17 := by
  sorry

end regular_17gon_symmetry_sum_l3348_334809


namespace tan_equality_implies_75_l3348_334808

theorem tan_equality_implies_75 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) :
  Real.tan (n • π / 180) = Real.tan (255 • π / 180) → n = 75 := by
  sorry

end tan_equality_implies_75_l3348_334808


namespace base_h_solution_l3348_334816

/-- Represents a digit in base h --/
def Digit (h : ℕ) := {d : ℕ // d < h}

/-- Converts a natural number to its representation in base h --/
def toBaseH (n h : ℕ) : List (Digit h) :=
  sorry

/-- Performs addition in base h --/
def addBaseH (a b : List (Digit h)) : List (Digit h) :=
  sorry

/-- The given addition problem --/
def additionProblem (h : ℕ) : Prop :=
  let a := toBaseH 5342 h
  let b := toBaseH 6421 h
  let result := toBaseH 14263 h
  addBaseH a b = result

theorem base_h_solution :
  ∃ h : ℕ, h > 0 ∧ additionProblem h ∧ h = 8 :=
sorry

end base_h_solution_l3348_334816


namespace boat_downstream_distance_l3348_334810

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 25 km/hr in still water, traveling downstream
    in a stream with a speed of 5 km/hr for 4 hours, travels a distance of 120 km. -/
theorem boat_downstream_distance :
  distance_downstream 25 5 4 = 120 := by
  sorry

#eval distance_downstream 25 5 4

end boat_downstream_distance_l3348_334810


namespace gcd_sum_fraction_eq_half_iff_special_triples_l3348_334860

/-- Given positive integers a, b, c satisfying a < b < c, prove that
    (a.gcd b + b.gcd c + c.gcd a) / (a + b + c) = 1/2
    if and only if there exists a positive integer d such that
    (a, b, c) = (d, 2*d, 3*d) or (a, b, c) = (d, 3*d, 6*d) -/
theorem gcd_sum_fraction_eq_half_iff_special_triples
  (a b c : ℕ+) (h1 : a < b) (h2 : b < c) :
  (a.gcd b + b.gcd c + c.gcd a : ℚ) / (a + b + c) = 1/2 ↔
  (∃ d : ℕ+, (a, b, c) = (d, 2*d, 3*d) ∨ (a, b, c) = (d, 3*d, 6*d)) := by
  sorry

end gcd_sum_fraction_eq_half_iff_special_triples_l3348_334860


namespace sum_of_a_and_b_min_value_of_expression_l3348_334870

/-- Given a > 0, b > 0, and the minimum value of |x+a| + |x-b| is 4, then a + b = 4 -/
theorem sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_min : ∀ x, |x + a| + |x - b| ≥ 4) : a + b = 4 := by sorry

/-- Given a + b = 4, the minimum value of (1/4)a² + (1/9)b² is 16/13 -/
theorem min_value_of_expression (a b : ℝ) (h : a + b = 4) :
  ∀ x y, x > 0 → y > 0 → x + y = 4 → (1/4) * a^2 + (1/9) * b^2 ≤ (1/4) * x^2 + (1/9) * y^2 := by sorry

end sum_of_a_and_b_min_value_of_expression_l3348_334870


namespace gym_budget_problem_l3348_334892

/-- Proves that given a budget that allows for the purchase of 10 softballs at $9 each after a 20% increase,
    the original budget would allow for the purchase of 15 dodgeballs at $5 each. -/
theorem gym_budget_problem (original_budget : ℝ) 
  (h1 : original_budget * 1.2 = 10 * 9) 
  (h2 : original_budget > 0) : 
  original_budget / 5 = 15 := by
sorry

end gym_budget_problem_l3348_334892


namespace rectangular_plot_length_difference_l3348_334865

theorem rectangular_plot_length_difference (breadth : ℝ) (x : ℝ) : 
  breadth > 0 →
  x > 0 →
  breadth + x = 60 →
  4 * breadth + 2 * x = 200 →
  x = 20 :=
by sorry

end rectangular_plot_length_difference_l3348_334865


namespace max_profit_at_max_price_max_profit_value_mall_sale_max_profit_l3348_334803

/-- Represents the shopping mall's clothing sale scenario -/
structure ClothingSale where
  cost : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ
  min_price : ℝ
  max_price : ℝ

/-- The specific clothing sale scenario as described in the problem -/
def mall_sale : ClothingSale :=
  { cost := 60
  , sales_function := λ x => -x + 120
  , profit_function := λ x => (x - 60) * (-x + 120)
  , min_price := 60
  , max_price := 84
  }

/-- Theorem stating that the maximum profit is achieved at the highest allowed price -/
theorem max_profit_at_max_price (sale : ClothingSale) :
  ∀ x ∈ Set.Icc sale.min_price sale.max_price,
    sale.profit_function x ≤ sale.profit_function sale.max_price :=
sorry

/-- Theorem stating that the maximum profit is 864 dollars -/
theorem max_profit_value (sale : ClothingSale) :
  sale.profit_function sale.max_price = 864 :=
sorry

/-- Main theorem combining the above results -/
theorem mall_sale_max_profit :
  ∃ x ∈ Set.Icc mall_sale.min_price mall_sale.max_price,
    mall_sale.profit_function x = 864 ∧
    ∀ y ∈ Set.Icc mall_sale.min_price mall_sale.max_price,
      mall_sale.profit_function y ≤ 864 :=
sorry

end max_profit_at_max_price_max_profit_value_mall_sale_max_profit_l3348_334803


namespace mikes_games_last_year_l3348_334893

/-- The number of basketball games Mike went to this year -/
def games_this_year : ℕ := 15

/-- The number of basketball games Mike missed this year -/
def games_missed : ℕ := 41

/-- The total number of basketball games Mike went to -/
def total_games : ℕ := 54

/-- The number of basketball games Mike went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem mikes_games_last_year : games_last_year = 39 := by
  sorry

end mikes_games_last_year_l3348_334893


namespace rolling_cube_path_length_l3348_334841

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Represents the path of a point on a rolling cube -/
def RollingCubePath (c : Cube) : ℝ := sorry

/-- Theorem stating the length of the path followed by the center point on the top face of a rolling cube -/
theorem rolling_cube_path_length (c : Cube) (h : c.sideLength = 2) :
  RollingCubePath c = 4 * Real.pi := by sorry

end rolling_cube_path_length_l3348_334841


namespace largest_integer_satisfying_inequality_l3348_334833

theorem largest_integer_satisfying_inequality :
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), 3 * (m^2007 : ℝ) < 3^4015 → m ≤ n) ∧
  (3 * ((n : ℝ)^2007) < 3^4015) :=
sorry

end largest_integer_satisfying_inequality_l3348_334833


namespace special_equation_result_l3348_334822

/-- If y is a real number satisfying y + 1/y = 3, then y^13 - 5y^9 + y^5 = 0 -/
theorem special_equation_result (y : ℝ) (h : y + 1/y = 3) : y^13 - 5*y^9 + y^5 = 0 := by
  sorry

end special_equation_result_l3348_334822


namespace luncheon_attendance_l3348_334864

theorem luncheon_attendance (invited : ℕ) (table_capacity : ℕ) (tables_used : ℕ) 
  (h1 : invited = 24) 
  (h2 : table_capacity = 7) 
  (h3 : tables_used = 2) : 
  invited - (table_capacity * tables_used) = 10 := by
  sorry

end luncheon_attendance_l3348_334864


namespace tax_rate_percentage_l3348_334878

/-- Given a tax rate of $82 per $100.00, prove that it is equivalent to 82% -/
theorem tax_rate_percentage : 
  let tax_amount : ℚ := 82
  let base_amount : ℚ := 100
  (tax_amount / base_amount) * 100 = 82 := by sorry

end tax_rate_percentage_l3348_334878


namespace root_difference_of_arithmetic_progression_l3348_334866

-- Define the polynomial coefficients
def a : ℝ := 81
def b : ℝ := -171
def c : ℝ := 107
def d : ℝ := -18

-- Define the polynomial
def p (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem root_difference_of_arithmetic_progression :
  ∃ (r₁ r₂ r₃ : ℝ),
    -- The roots satisfy the polynomial equation
    p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0 ∧
    -- The roots are in arithmetic progression
    r₂ - r₁ = r₃ - r₂ ∧
    -- The difference between the largest and smallest roots is approximately 1.66
    abs (max r₁ (max r₂ r₃) - min r₁ (min r₂ r₃) - 1.66) < 0.01 :=
by
  sorry


end root_difference_of_arithmetic_progression_l3348_334866


namespace expression_value_l3348_334889

theorem expression_value : 
  (128^2 - 5^2) / (72^2 - 13^2) * ((72-13)*(72+13)) / ((128-5)*(128+5)) * (128+5) / (72+13) = 133/85 := by
sorry

end expression_value_l3348_334889


namespace vasya_fool_count_l3348_334881

/-- Represents the number of times a player was left as the "fool" -/
structure FoolCount where
  count : ℕ
  positive : count > 0

/-- The game "Fool" with four players -/
structure FoolGame where
  misha : FoolCount
  petya : FoolCount
  kolya : FoolCount
  vasya : FoolCount
  total_games : misha.count + petya.count + kolya.count + vasya.count = 16
  misha_most : misha.count > petya.count ∧ misha.count > kolya.count ∧ misha.count > vasya.count
  petya_kolya_sum : petya.count + kolya.count = 9

theorem vasya_fool_count (game : FoolGame) : game.vasya.count = 1 := by
  sorry

end vasya_fool_count_l3348_334881


namespace total_weight_carrots_cucumbers_l3348_334894

def carrot_weight : ℝ := 250
def cucumber_multiplier : ℝ := 2.5

theorem total_weight_carrots_cucumbers : 
  carrot_weight + cucumber_multiplier * carrot_weight = 875 := by
  sorry

end total_weight_carrots_cucumbers_l3348_334894


namespace inequality_system_solution_l3348_334884

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (-1 < x ∧ x < 1) ↔ (2*x - a < 1 ∧ x - 2*b > 3)) → 
  (a + 1) * (b - 1) = -6 := by
  sorry

end inequality_system_solution_l3348_334884


namespace stratified_sampling_theorem_l3348_334807

theorem stratified_sampling_theorem (first_grade second_grade third_grade total_selected : ℕ) 
  (h1 : first_grade = 120)
  (h2 : second_grade = 180)
  (h3 : third_grade = 150)
  (h4 : total_selected = 90) :
  let total_students := first_grade + second_grade + third_grade
  let sampling_ratio := total_selected / total_students
  (sampling_ratio * first_grade : ℕ) = 24 ∧
  (sampling_ratio * second_grade : ℕ) = 36 ∧
  (sampling_ratio * third_grade : ℕ) = 30 := by
sorry

end stratified_sampling_theorem_l3348_334807


namespace sphere_radius_from_hole_l3348_334835

/-- Given a sphere intersecting a plane, if the resulting circular hole has a diameter of 30 cm
    and a depth of 10 cm, then the radius of the sphere is 16.25 cm. -/
theorem sphere_radius_from_hole (r : ℝ) (h : r > 0) :
  (∃ x : ℝ, x > 0 ∧ x^2 + 15^2 = (x + 10)^2 ∧ r^2 = x^2 + 15^2) →
  r = 16.25 := by
  sorry

end sphere_radius_from_hole_l3348_334835


namespace two_thousand_fourteen_between_powers_of_ten_l3348_334820

theorem two_thousand_fourteen_between_powers_of_ten : 10^3 < 2014 ∧ 2014 < 10^4 := by
  sorry

end two_thousand_fourteen_between_powers_of_ten_l3348_334820


namespace travel_equation_correct_l3348_334873

/-- Represents the scenario of Confucius and his students traveling to a school -/
structure TravelScenario where
  distance : ℝ
  student_speed : ℝ
  cart_speed_multiplier : ℝ
  head_start : ℝ

/-- The equation representing the travel times is correct for the given scenario -/
theorem travel_equation_correct (scenario : TravelScenario) 
  (h_distance : scenario.distance = 30)
  (h_cart_speed : scenario.cart_speed_multiplier = 1.5)
  (h_head_start : scenario.head_start = 1)
  (h_student_speed_pos : scenario.student_speed > 0) :
  scenario.distance / scenario.student_speed = 
    scenario.distance / (scenario.cart_speed_multiplier * scenario.student_speed) + scenario.head_start :=
sorry

end travel_equation_correct_l3348_334873


namespace solution_set_correct_l3348_334880

def solution_set : Set (ℚ × ℚ) :=
  {(-2/3, 1), (1, 1), (-1/3, -3), (-1/3, 2)}

def satisfies_equations (p : ℚ × ℚ) : Prop :=
  let x := p.1
  let y := p.2
  (3*x - y - 3*x*y = -1) ∧ (9*x^2*y^2 + 9*x^2 + y^2 - 6*x*y = 13)

theorem solution_set_correct :
  ∀ p : ℚ × ℚ, p ∈ solution_set ↔ satisfies_equations p :=
by sorry

end solution_set_correct_l3348_334880


namespace isosceles_triangle_base_length_l3348_334811

/-- The base length of an isosceles triangle with specific conditions -/
theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h_equilateral : equilateral_perimeter = 60) 
  (h_isosceles : isosceles_perimeter = 45) 
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) : 
  isosceles_base = 5 :=
sorry

end isosceles_triangle_base_length_l3348_334811


namespace multiple_of_8_in_second_column_thousand_in_second_column_l3348_334818

/-- Represents the column number in the arrangement -/
inductive Column
| First
| Second
| Third
| Fourth
| Fifth

/-- Represents the row type in the arrangement -/
inductive RowType
| Odd
| Even

/-- Function to determine the column of a given integer in the arrangement -/
def column_of_integer (n : ℕ) : Column :=
  sorry

/-- Function to determine the row type of a given integer in the arrangement -/
def row_type_of_integer (n : ℕ) : RowType :=
  sorry

/-- Theorem stating that any multiple of 8 appears in the second column -/
theorem multiple_of_8_in_second_column (n : ℕ) (h : 8 ∣ n) : column_of_integer n = Column.Second :=
  sorry

/-- Corollary: 1000 appears in the second column -/
theorem thousand_in_second_column : column_of_integer 1000 = Column.Second :=
  sorry

end multiple_of_8_in_second_column_thousand_in_second_column_l3348_334818


namespace only_setC_forms_right_triangle_l3348_334895

-- Define a function to check if three numbers can form a right triangle
def canFormRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def setA : List ℕ := [4, 5, 6]
def setB : List ℕ := [5, 7, 9]
def setC : List ℕ := [6, 8, 10]
def setD : List ℕ := [7, 8, 9]

-- Theorem stating that only set C can form a right triangle
theorem only_setC_forms_right_triangle :
  (¬ canFormRightTriangle setA[0] setA[1] setA[2]) ∧
  (¬ canFormRightTriangle setB[0] setB[1] setB[2]) ∧
  (canFormRightTriangle setC[0] setC[1] setC[2]) ∧
  (¬ canFormRightTriangle setD[0] setD[1] setD[2]) :=
by
  sorry

#check only_setC_forms_right_triangle

end only_setC_forms_right_triangle_l3348_334895


namespace odd_periodic_symmetry_ln_quotient_odd_main_theorem_l3348_334861

-- Definition of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- Theorem 1: Odd function with period 4 is symmetric about (2,0)
theorem odd_periodic_symmetry (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : IsPeriodic f 4) :
  ∀ x, f (2 + x) = f (2 - x) :=
sorry

-- Theorem 2: ln((1+x)/(1-x)) is an odd function on (-1,1)
theorem ln_quotient_odd :
  IsOdd (fun x => Real.log ((1 + x) / (1 - x))) :=
sorry

-- Main theorem combining both results
theorem main_theorem :
  (∃ f : ℝ → ℝ, IsOdd f ∧ IsPeriodic f 4 ∧ (∀ x, f (2 + x) = f (2 - x))) ∧
  IsOdd (fun x => Real.log ((1 + x) / (1 - x))) :=
sorry

end odd_periodic_symmetry_ln_quotient_odd_main_theorem_l3348_334861


namespace factorization_x4_minus_81_l3348_334857

theorem factorization_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorization_x4_minus_81_l3348_334857


namespace f_has_unique_minimum_l3348_334853

open Real

-- Define the function f(x) = 2x - ln x
noncomputable def f (x : ℝ) : ℝ := 2 * x - log x

-- Theorem statement
theorem f_has_unique_minimum :
  ∃! (x : ℝ), x > 0 ∧ IsLocalMin f x ∧ f x = 1 + log 2 := by
  sorry

end f_has_unique_minimum_l3348_334853


namespace arc_length_calculation_l3348_334844

theorem arc_length_calculation (r θ : Real) (h1 : r = 2) (h2 : θ = π/3) :
  r * θ = 2 * π / 3 := by
  sorry

end arc_length_calculation_l3348_334844


namespace f_neg_two_eq_neg_one_l3348_334814

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem f_neg_two_eq_neg_one : f (-2) = -1 := by
  sorry

end f_neg_two_eq_neg_one_l3348_334814


namespace tom_pennies_l3348_334896

/-- Represents the number of coins of each type --/
structure CoinCounts where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents given a CoinCounts --/
def totalValueInCents (coins : CoinCounts) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- The main theorem --/
theorem tom_pennies (coins : CoinCounts) 
    (h1 : coins.quarters = 10)
    (h2 : coins.dimes = 3)
    (h3 : coins.nickels = 4)
    (h4 : totalValueInCents coins = 500) :
    coins.pennies = 200 := by
  sorry


end tom_pennies_l3348_334896


namespace trigonometric_identity_l3348_334830

theorem trigonometric_identity (θ : Real) (h : Real.tan θ = 3) :
  Real.sin θ^2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ^2 = 1 := by
  sorry

end trigonometric_identity_l3348_334830


namespace angle_relationship_l3348_334831

theorem angle_relationship (larger_angle smaller_angle : ℝ) : 
  larger_angle = 99 ∧ smaller_angle = 81 → larger_angle - smaller_angle = 18 := by
  sorry

end angle_relationship_l3348_334831
