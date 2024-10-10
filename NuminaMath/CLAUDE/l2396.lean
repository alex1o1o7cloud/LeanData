import Mathlib

namespace trigonometric_equation_solutions_l2396_239607

open Real

theorem trigonometric_equation_solutions (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    sin (5 * a) * cos x₁ - cos (x₁ + 4 * a) = 0 ∧
    sin (5 * a) * cos x₂ - cos (x₂ + 4 * a) = 0 ∧
    ¬ ∃ k : ℤ, x₁ - x₂ = π * (k : ℝ)) ↔
  ∃ t : ℤ, a = π * ((4 * t + 1 : ℤ) : ℝ) / 2 :=
by sorry

end trigonometric_equation_solutions_l2396_239607


namespace cube_root_of_square_64_l2396_239694

theorem cube_root_of_square_64 (x : ℝ) (h : x^2 = 64) :
  x^(1/3) = 2 ∨ x^(1/3) = -2 := by sorry

end cube_root_of_square_64_l2396_239694


namespace male_response_rate_change_l2396_239683

/- Define the survey data structure -/
structure SurveyData where
  totalCustomers : ℕ
  malePercentage : ℚ
  femalePercentage : ℚ
  totalResponses : ℕ
  maleResponsePercentage : ℚ
  femaleResponsePercentage : ℚ

/- Define the surveys -/
def initialSurvey : SurveyData :=
  { totalCustomers := 100
  , malePercentage := 60 / 100
  , femalePercentage := 40 / 100
  , totalResponses := 10
  , maleResponsePercentage := 50 / 100
  , femaleResponsePercentage := 50 / 100 }

def finalSurvey : SurveyData :=
  { totalCustomers := 90
  , malePercentage := 50 / 100
  , femalePercentage := 50 / 100
  , totalResponses := 27
  , maleResponsePercentage := 30 / 100
  , femaleResponsePercentage := 70 / 100 }

/- Calculate male response rate -/
def maleResponseRate (survey : SurveyData) : ℚ :=
  (survey.maleResponsePercentage * survey.totalResponses) /
  (survey.malePercentage * survey.totalCustomers)

/- Calculate percentage change -/
def percentageChange (initial : ℚ) (final : ℚ) : ℚ :=
  ((final - initial) / initial) * 100

/- Theorem statement -/
theorem male_response_rate_change :
  percentageChange (maleResponseRate initialSurvey) (maleResponseRate finalSurvey) = 113.4 := by
  sorry

end male_response_rate_change_l2396_239683


namespace perpendicular_line_through_point_specific_perpendicular_line_l2396_239680

/-- A line passing through a point and perpendicular to another line -/
theorem perpendicular_line_through_point 
  (x₀ y₀ : ℝ) 
  (a b c : ℝ) 
  (h₁ : b ≠ 0) 
  (h₂ : a ≠ 0) :
  ∃ m k : ℝ, 
    (y₀ = m * x₀ + k) ∧ 
    (m = -a / b) ∧
    (k = y₀ - m * x₀) :=
sorry

/-- The specific line passing through (3, -5) and perpendicular to 2x - 6y + 15 = 0 -/
theorem specific_perpendicular_line : 
  ∃ m k : ℝ, 
    (-5 = m * 3 + k) ∧ 
    (m = -(2 : ℝ) / (-6 : ℝ)) ∧ 
    (k = -5 - m * 3) ∧
    (k = -4) ∧ 
    (m = 3) :=
sorry

end perpendicular_line_through_point_specific_perpendicular_line_l2396_239680


namespace unique_x_with_three_prime_factors_l2396_239679

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by sorry

end unique_x_with_three_prime_factors_l2396_239679


namespace perfect_square_mod_three_l2396_239666

theorem perfect_square_mod_three (n : ℤ) : 
  (∃ k : ℤ, n = k^2) → (n % 3 = 0 ∨ n % 3 = 1) := by
  sorry

end perfect_square_mod_three_l2396_239666


namespace min_max_m_l2396_239676

theorem min_max_m (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (eq1 : 3 * a + 2 * b + c = 5) (eq2 : 2 * a + b - 3 * c = 1) :
  let m := 3 * a + b - 7 * c
  ∃ (m_min m_max : ℝ), (∀ m', m' = m → m' ≥ m_min) ∧
                       (∀ m', m' = m → m' ≤ m_max) ∧
                       m_min = -5/7 ∧ m_max = -1/11 := by
  sorry

end min_max_m_l2396_239676


namespace not_always_possible_within_30_moves_l2396_239633

/-- Represents a move on the board -/
inductive Move
  | add_two : Fin 3 → Fin 3 → Move
  | subtract_all : Move

/-- The state of the board -/
def Board := Fin 3 → ℕ

/-- Apply a move to the board -/
def apply_move (b : Board) (m : Move) : Board :=
  match m with
  | Move.add_two i j => fun k => if k = i ∨ k = j then b k + 1 else b k
  | Move.subtract_all => fun k => if b k > 0 then b k - 1 else 0

/-- Check if all numbers on the board are zero -/
def all_zero (b : Board) : Prop := ∀ i, b i = 0

/-- The main theorem -/
theorem not_always_possible_within_30_moves :
  ∃ (initial : Board),
    (∀ i, 1 ≤ initial i ∧ initial i ≤ 9) ∧
    (∀ i j, i ≠ j → initial i ≠ initial j) ∧
    ¬∃ (moves : List Move),
      moves.length ≤ 30 ∧
      all_zero (moves.foldl apply_move initial) :=
by sorry

end not_always_possible_within_30_moves_l2396_239633


namespace brett_travel_distance_l2396_239608

/-- The distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Brett's travel distance in 12 hours at 75 miles per hour is 900 miles -/
theorem brett_travel_distance : distance_traveled 75 12 = 900 := by
  sorry

end brett_travel_distance_l2396_239608


namespace smallest_d_l2396_239658

/-- The smallest positive value of d that satisfies the equation √((4√3)² + (d+4)²) = 2d -/
theorem smallest_d : ∃ d : ℝ, d > 0 ∧ 
  (∀ d' : ℝ, d' > 0 → (4 * Real.sqrt 3)^2 + (d' + 4)^2 = (2 * d')^2 → d ≤ d') ∧
  (4 * Real.sqrt 3)^2 + (d + 4)^2 = (2 * d)^2 ∧
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
sorry

end smallest_d_l2396_239658


namespace expression_evaluation_l2396_239678

theorem expression_evaluation : -2^3 + (18 - (-3)^2) / (-3) = -11 := by
  sorry

end expression_evaluation_l2396_239678


namespace oranges_harvested_per_day_l2396_239649

theorem oranges_harvested_per_day :
  let total_sacks : ℕ := 56
  let total_days : ℕ := 14
  let sacks_per_day : ℕ := total_sacks / total_days
  sacks_per_day = 4 := by sorry

end oranges_harvested_per_day_l2396_239649


namespace power_of_power_of_three_l2396_239615

theorem power_of_power_of_three : (3^3)^3 = 19683 := by
  sorry

end power_of_power_of_three_l2396_239615


namespace pear_sales_ratio_l2396_239617

theorem pear_sales_ratio (total : ℕ) (afternoon : ℕ) 
  (h1 : total = 390)
  (h2 : afternoon = 260) :
  (afternoon : ℚ) / (total - afternoon : ℚ) = 2 := by
  sorry

end pear_sales_ratio_l2396_239617


namespace first_discount_percentage_l2396_239669

/-- Given an initial price and a final price after two discounts, 
    where the second discount is known, calculate the first discount percentage. -/
theorem first_discount_percentage 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : initial_price = 528)
  (h2 : final_price = 380.16)
  (h3 : second_discount = 0.1)
  : ∃ (first_discount : ℝ),
    first_discount = 0.2 ∧ 
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by
  sorry

end first_discount_percentage_l2396_239669


namespace greatest_length_segment_l2396_239613

theorem greatest_length_segment (AE CD CF AC FD CE : ℝ) : 
  AE = Real.sqrt 106 →
  CD = 5 →
  CF = Real.sqrt 20 →
  AC = 5 →
  FD = Real.sqrt 85 →
  CE = Real.sqrt 29 →
  AC + CE > AE ∧ AC + CE > CD + CF ∧ AC + CE > AC + CF ∧ AC + CE > FD :=
by sorry

end greatest_length_segment_l2396_239613


namespace frog_eggs_first_day_l2396_239619

/-- Represents the number of eggs laid by a frog over 4 days -/
def frog_eggs (x : ℕ) : ℕ :=
  let day1 := x
  let day2 := 2 * x
  let day3 := 2 * x + 20
  let day4 := 2 * (day1 + day2 + day3)
  day1 + day2 + day3 + day4

/-- Theorem stating that if the frog lays 810 eggs over 4 days following the given pattern,
    then it laid 50 eggs on the first day -/
theorem frog_eggs_first_day :
  ∃ (x : ℕ), frog_eggs x = 810 ∧ x = 50 :=
sorry

end frog_eggs_first_day_l2396_239619


namespace add_preserves_inequality_l2396_239692

theorem add_preserves_inequality (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end add_preserves_inequality_l2396_239692


namespace rectangle_circle_area_ratio_l2396_239667

theorem rectangle_circle_area_ratio :
  ∀ (b r : ℝ),
  b > 0 →
  r > 0 →
  6 * b = 2 * Real.pi * r →
  (2 * b^2) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end rectangle_circle_area_ratio_l2396_239667


namespace unique_prime_with_prime_sums_l2396_239603

theorem unique_prime_with_prime_sums : ∀ p : ℕ, 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) → p = 3 :=
sorry

end unique_prime_with_prime_sums_l2396_239603


namespace inequality_proof_l2396_239639

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l2396_239639


namespace provisions_duration_l2396_239677

theorem provisions_duration (initial_soldiers : ℕ) (initial_consumption : ℚ)
  (additional_soldiers : ℕ) (new_consumption : ℚ) (new_duration : ℕ) :
  initial_soldiers = 1200 →
  initial_consumption = 3 →
  additional_soldiers = 528 →
  new_consumption = 5/2 →
  new_duration = 25 →
  (↑initial_soldiers * initial_consumption * ↑new_duration =
   ↑(initial_soldiers + additional_soldiers) * new_consumption * ↑new_duration) →
  (↑initial_soldiers * initial_consumption * (1080000 / 3600 : ℚ) =
   ↑initial_soldiers * initial_consumption * 300) :=
by sorry

end provisions_duration_l2396_239677


namespace complex_cube_inequality_l2396_239674

theorem complex_cube_inequality (z : ℂ) (h : Complex.abs (z + 1) > 2) :
  Complex.abs (z^3 + 1) > 1 := by
  sorry

end complex_cube_inequality_l2396_239674


namespace square_49_using_50_l2396_239640

theorem square_49_using_50 : ∃ x : ℕ, 49^2 = 50^2 - x + 1 ∧ x = 100 := by
  sorry

end square_49_using_50_l2396_239640


namespace equation_proof_l2396_239622

theorem equation_proof : 289 + 2 * 17 * 4 + 16 = 441 := by
  sorry

end equation_proof_l2396_239622


namespace cube_root_abs_sqrt_equality_l2396_239628

theorem cube_root_abs_sqrt_equality : 
  (64 : ℝ)^(1/3) - |Real.sqrt 3 - 3| + Real.sqrt 36 = 7 + Real.sqrt 3 := by sorry

end cube_root_abs_sqrt_equality_l2396_239628


namespace linear_function_proof_l2396_239671

/-- A linear function passing through three given points -/
def linear_function (x : ℝ) : ℝ := 3 * x + 4

/-- Theorem stating that the linear function passes through the given points and f(40) = 124 -/
theorem linear_function_proof :
  (linear_function 2 = 10) ∧
  (linear_function 6 = 22) ∧
  (linear_function 10 = 34) ∧
  (linear_function 40 = 124) := by
  sorry

#check linear_function_proof

end linear_function_proof_l2396_239671


namespace total_trout_caught_l2396_239648

/-- The number of trout caught by Sara, Melanie, and John -/
def total_trout (sara melanie john : ℕ) : ℕ := sara + melanie + john

/-- Theorem stating the total number of trout caught -/
theorem total_trout_caught :
  ∃ (sara melanie john : ℕ),
    sara = 5 ∧
    melanie = 2 * sara ∧
    john = 3 * melanie ∧
    total_trout sara melanie john = 45 := by
  sorry

end total_trout_caught_l2396_239648


namespace modular_congruence_iff_divisibility_l2396_239653

theorem modular_congruence_iff_divisibility (a n k : ℕ) (ha : a ≥ 2) :
  a ^ k ≡ 1 [MOD a ^ n - 1] ↔ n ∣ k :=
sorry

end modular_congruence_iff_divisibility_l2396_239653


namespace britta_winning_strategy_l2396_239685

-- Define the game
def Game (n : ℕ) :=
  n ≥ 5 ∧ Odd n

-- Define Britta's winning condition
def BrittaWins (n x₁ x₂ y₁ y₂ : ℕ) : Prop :=
  (x₁ * x₂ * (x₁ - y₁) * (x₂ - y₂)) ^ ((n - 1) / 2) % n = 1

-- Define Britta's strategy
def BrittaStrategy (n : ℕ) (h : Game n) : Prop :=
  ∀ (x₁ x₂ : ℕ), x₁ < n ∧ x₂ < n ∧ x₁ ≠ x₂ →
  ∃ (y₁ y₂ : ℕ), y₁ < n ∧ y₂ < n ∧ y₁ ≠ y₂ ∧ BrittaWins n x₁ x₂ y₁ y₂

-- Theorem statement
theorem britta_winning_strategy (n : ℕ) (h : Game n) :
  BrittaStrategy n h ↔ Nat.Prime n :=
sorry

end britta_winning_strategy_l2396_239685


namespace calcium_bromide_weight_l2396_239638

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := 79.904

/-- The number of moles of calcium bromide -/
def moles : ℝ := 4

/-- The molecular weight of calcium bromide (CaBr2) in g/mol -/
def molecular_weight_CaBr2 : ℝ := calcium_weight + 2 * bromine_weight

/-- The total weight of the given number of moles of calcium bromide in grams -/
def total_weight : ℝ := moles * molecular_weight_CaBr2

theorem calcium_bromide_weight : total_weight = 799.552 := by
  sorry

end calcium_bromide_weight_l2396_239638


namespace total_money_l2396_239650

def cecil_money : ℕ := 600

def catherine_money : ℕ := 2 * cecil_money - 250

def carmela_money : ℕ := 2 * cecil_money + 50

theorem total_money : cecil_money + catherine_money + carmela_money = 2800 := by
  sorry

end total_money_l2396_239650


namespace wooden_toy_price_is_20_l2396_239652

/-- The original price of each wooden toy -/
def wooden_toy_price : ℝ := 20

/-- The number of paintings bought -/
def num_paintings : ℕ := 10

/-- The original price of each painting -/
def painting_price : ℝ := 40

/-- The number of wooden toys bought -/
def num_toys : ℕ := 8

/-- The discount rate for paintings -/
def painting_discount : ℝ := 0.1

/-- The discount rate for wooden toys -/
def toy_discount : ℝ := 0.15

/-- The total loss from the sale -/
def total_loss : ℝ := 64

theorem wooden_toy_price_is_20 :
  (num_paintings * painting_price + num_toys * wooden_toy_price) -
  (num_paintings * painting_price * (1 - painting_discount) +
   num_toys * wooden_toy_price * (1 - toy_discount)) = total_loss :=
by sorry

end wooden_toy_price_is_20_l2396_239652


namespace circle_area_increase_l2396_239621

theorem circle_area_increase (π : ℝ) (h : π > 0) : 
  π * 5^2 - π * 2^2 = 21 * π := by sorry

end circle_area_increase_l2396_239621


namespace tan_beta_minus_2alpha_l2396_239614

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3)
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (β - 2*α) = 4/3 := by sorry

end tan_beta_minus_2alpha_l2396_239614


namespace least_number_divisibility_l2396_239655

theorem least_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 234 → ¬((3072 + m) % 57 = 0 ∧ (3072 + m) % 29 = 0)) ∧ 
  ((3072 + 234) % 57 = 0 ∧ (3072 + 234) % 29 = 0) := by
  sorry

end least_number_divisibility_l2396_239655


namespace bernards_blue_notebooks_l2396_239657

/-- Represents the number of notebooks Bernard had -/
structure BernardsNotebooks where
  red : ℕ
  white : ℕ
  blue : ℕ
  given : ℕ
  left : ℕ

/-- Theorem stating the number of blue notebooks Bernard had -/
theorem bernards_blue_notebooks
  (notebooks : BernardsNotebooks)
  (h_red : notebooks.red = 15)
  (h_white : notebooks.white = 19)
  (h_given : notebooks.given = 46)
  (h_left : notebooks.left = 5)
  (h_total : notebooks.red + notebooks.white + notebooks.blue = notebooks.given + notebooks.left) :
  notebooks.blue = 17 := by
  sorry

end bernards_blue_notebooks_l2396_239657


namespace root_implies_h_value_l2396_239606

theorem root_implies_h_value (h : ℝ) :
  ((-3 : ℝ)^3 + h * (-3) - 10 = 0) → h = -37/3 := by
  sorry

end root_implies_h_value_l2396_239606


namespace total_puppies_l2396_239635

theorem total_puppies (female_puppies male_puppies : ℕ) 
  (h1 : female_puppies = 2)
  (h2 : male_puppies = 10)
  (h3 : (female_puppies : ℚ) / male_puppies = 0.2) :
  female_puppies + male_puppies = 12 := by
sorry

end total_puppies_l2396_239635


namespace power_equation_l2396_239689

theorem power_equation (a : ℝ) (m n k : ℤ) (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) :
  a^(3*m + 2*n - k) = 4 := by
sorry

end power_equation_l2396_239689


namespace min_abs_diff_sqrt_30_l2396_239659

theorem min_abs_diff_sqrt_30 (x : ℤ) : |x - Real.sqrt 30| ≥ |5 - Real.sqrt 30| := by
  sorry

end min_abs_diff_sqrt_30_l2396_239659


namespace equal_projections_l2396_239618

/-- A circle divided into 42 equal arcs -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (points : Fin 42 → ℝ × ℝ)

/-- Projection of a point onto a line segment -/
def project (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem equal_projections (c : Circle) :
  let A₀ := c.points 0
  let A₃ := c.points 3
  let A₆ := c.points 6
  let A₇ := c.points 7
  let A₉ := c.points 9
  let A₂₁ := c.points 21
  let A'₃ := project A₃ A₀ A₂₁
  let A'₆ := project A₆ A₀ A₂₁
  let A'₇ := project A₇ A₀ A₂₁
  let A'₉ := project A₉ A₀ A₂₁
  distance A'₃ A'₆ = distance A'₇ A'₉ := by
    sorry

end equal_projections_l2396_239618


namespace free_throws_stats_l2396_239697

def free_throws : List ℝ := [20, 12, 22, 25, 10, 16, 15, 12, 30, 10]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem free_throws_stats :
  median free_throws = 15.5 ∧ mean free_throws = 17.2 := by sorry

end free_throws_stats_l2396_239697


namespace existence_of_large_subset_l2396_239624

/-- A family of 3-element subsets with at most one common element between any two subsets -/
def ValidFamily (I : Finset Nat) (A : Set (Finset Nat)) : Prop :=
  ∀ a ∈ A, a.card = 3 ∧ a ⊆ I ∧ ∀ b ∈ A, a ≠ b → (a ∩ b).card ≤ 1

/-- The theorem statement -/
theorem existence_of_large_subset (n : Nat) (I : Finset Nat) (hI : I.card = n) 
    (A : Set (Finset Nat)) (hA : ValidFamily I A) :
  ∃ X : Finset Nat, X ⊆ I ∧ 
    (∀ a ∈ A, ¬(a ⊆ X)) ∧ 
    X.card ≥ Nat.floor (Real.sqrt (2 * n)) := by
  sorry

end existence_of_large_subset_l2396_239624


namespace cone_volume_over_pi_l2396_239656

/-- Given a cone formed from a 240-degree sector of a circle with radius 16,
    prove that the volume of the cone divided by π is equal to 8192√10 / 81. -/
theorem cone_volume_over_pi (r : ℝ) (h : ℝ) :
  r = 32 / 3 →
  h = 8 * Real.sqrt 10 / 3 →
  (1 / 3 * π * r^2 * h) / π = 8192 * Real.sqrt 10 / 81 := by sorry

end cone_volume_over_pi_l2396_239656


namespace complex_power_result_l2396_239698

theorem complex_power_result (n : ℕ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 3)^n = 256) : (1 + i : ℂ)^n = -4 := by
  sorry

end complex_power_result_l2396_239698


namespace adams_shopping_cost_l2396_239634

/-- Calculates the total cost of Adam's shopping, including discount and sales tax -/
def total_cost (sandwich_price : ℚ) (chips_price : ℚ) (water_price : ℚ) 
                (sandwich_count : ℕ) (chips_count : ℕ) (water_count : ℕ) 
                (tax_rate : ℚ) : ℚ :=
  let sandwich_cost := (sandwich_count - 1) * sandwich_price
  let chips_cost := chips_count * chips_price
  let water_cost := water_count * water_price
  let subtotal := sandwich_cost + chips_cost + water_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that Adam's total shopping cost is $29.15 -/
theorem adams_shopping_cost : 
  total_cost 4 3.5 2 4 3 2 0.1 = 29.15 := by
  sorry

end adams_shopping_cost_l2396_239634


namespace flour_weight_relation_l2396_239631

/-- Theorem: Given two equations representing the weight of flour bags, 
    prove that the new combined weight is equal to the original weight plus 33 pounds. -/
theorem flour_weight_relation (x y : ℝ) : 
  y = (16 - 4) + (30 - 6) + (x - 3) → 
  y = 12 + 24 + (x - 3) → 
  y = x + 33 := by
  sorry

end flour_weight_relation_l2396_239631


namespace number_problem_l2396_239644

theorem number_problem (A B : ℤ) 
  (h1 : A - B = 144) 
  (h2 : A = 3 * B - 14) : 
  A = 223 := by sorry

end number_problem_l2396_239644


namespace original_expenditure_l2396_239696

/-- Represents the hostel mess expenditure problem -/
structure HostelMess where
  initial_students : ℕ
  initial_expenditure : ℕ
  initial_avg_expenditure : ℕ

/-- Represents changes in the hostel mess -/
structure MessChange where
  day : ℕ
  student_change : ℤ
  expense_change : ℕ
  avg_expenditure_change : ℤ

/-- Theorem stating the original expenditure of the mess -/
theorem original_expenditure (mess : HostelMess) 
  (change1 : MessChange) (change2 : MessChange) (change3 : MessChange) : 
  mess.initial_students = 35 →
  change1.day = 10 → change1.student_change = 7 → change1.expense_change = 84 → change1.avg_expenditure_change = -1 →
  change2.day = 15 → change2.student_change = -5 → change2.expense_change = 40 → change2.avg_expenditure_change = 2 →
  change3.day = 25 → change3.student_change = 3 → change3.expense_change = 30 → change3.avg_expenditure_change = 0 →
  mess.initial_expenditure = 630 := by
  sorry

end original_expenditure_l2396_239696


namespace kg_to_tons_conversion_l2396_239668

theorem kg_to_tons_conversion (kg_per_ton : ℕ) (h : kg_per_ton = 1000) :
  (3600 - 600) / kg_per_ton = 3 := by
  sorry

end kg_to_tons_conversion_l2396_239668


namespace true_discount_calculation_l2396_239686

/-- Given a present worth and banker's gain, calculate the true discount. -/
theorem true_discount_calculation (PW BG : ℚ) (h1 : PW = 576) (h2 : BG = 16) :
  ∃ TD : ℚ, TD^2 = BG * PW ∧ TD = 96 := by
  sorry

end true_discount_calculation_l2396_239686


namespace smallest_positive_root_of_g_l2396_239642

open Real

theorem smallest_positive_root_of_g : ∃ s : ℝ,
  s > 0 ∧
  sin s + 3 * cos s + 4 * tan s = 0 ∧
  (∀ x, 0 < x → x < s → sin x + 3 * cos x + 4 * tan x ≠ 0) ∧
  ⌊s⌋ = 4 := by
  sorry

end smallest_positive_root_of_g_l2396_239642


namespace sqrt_product_equality_l2396_239637

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2396_239637


namespace geometric_sequence_problem_l2396_239620

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def SequenceConditions (a : ℕ → ℝ) : Prop :=
  GeometricSequence a ∧ 
  a 2 * a 3 = 2 * a 1 ∧ 
  (a 4 + 2 * a 7) / 2 = 5 / 4

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  SequenceConditions a → a 1 = 16 := by
  sorry

end geometric_sequence_problem_l2396_239620


namespace min_desks_for_arrangements_l2396_239695

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- The number of different seating arrangements -/
def num_arrangements : ℕ := 2

/-- The minimum number of empty desks between students -/
def min_empty_desks : ℕ := 1

/-- A function that calculates the number of valid seating arrangements
    given the number of desks -/
def valid_arrangements (num_desks : ℕ) : ℕ := sorry

/-- Theorem stating that 5 is the minimum number of desks required -/
theorem min_desks_for_arrangements :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ m : ℕ, m < n → valid_arrangements m < num_arrangements) ∧
  valid_arrangements n = num_arrangements :=
sorry

end min_desks_for_arrangements_l2396_239695


namespace number_squared_sum_equals_100_l2396_239647

theorem number_squared_sum_equals_100 : ∃ x : ℝ, (7.5 * 7.5) + 37.5 + (x * x) = 100 ∧ x = 2.5 := by
  sorry

end number_squared_sum_equals_100_l2396_239647


namespace closest_fraction_is_one_sixth_l2396_239665

def medals_won : ℚ := 17 / 100

def possible_fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction_is_one_sixth :
  ∀ x ∈ possible_fractions, x ≠ 1/6 → |medals_won - 1/6| < |medals_won - x| :=
sorry

end closest_fraction_is_one_sixth_l2396_239665


namespace cone_cylinder_volume_ratio_l2396_239643

/-- The ratio of the volume of a cone to the volume of a cylinder, given specific proportions -/
theorem cone_cylinder_volume_ratio 
  (r h : ℝ) 
  (h_pos : h > 0) 
  (r_pos : r > 0) : 
  (1 / 3 * π * (r / 2)^2 * (h / 3)) / (π * r^2 * h) = 1 / 36 := by
  sorry

end cone_cylinder_volume_ratio_l2396_239643


namespace polynomial_factorization_l2396_239693

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l2396_239693


namespace x_one_minus_f_equals_four_to_500_l2396_239682

theorem x_one_minus_f_equals_four_to_500 :
  let x : ℝ := (3 + Real.sqrt 5) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 4 ^ 500 := by
  sorry

end x_one_minus_f_equals_four_to_500_l2396_239682


namespace sufficient_not_necessary_l2396_239645

theorem sufficient_not_necessary (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) :=
sorry

end sufficient_not_necessary_l2396_239645


namespace juggling_balls_needed_l2396_239688

/-- The number of balls needed for a juggling spectacle -/
theorem juggling_balls_needed (num_jugglers : ℕ) (balls_per_juggler : ℕ) : 
  num_jugglers = 5000 → balls_per_juggler = 12 → num_jugglers * balls_per_juggler = 60000 := by
  sorry

end juggling_balls_needed_l2396_239688


namespace permutation_count_mod_1000_l2396_239690

/-- The number of permutations of a 15-character string with 4 A's, 5 B's, and 6 C's -/
def N : ℕ := sorry

/-- Condition: None of the first four letters is an A -/
axiom cond1 : sorry

/-- Condition: None of the next five letters is a B -/
axiom cond2 : sorry

/-- Condition: None of the last six letters is a C -/
axiom cond3 : sorry

/-- Theorem: The number of permutations N satisfying the conditions is congruent to 320 modulo 1000 -/
theorem permutation_count_mod_1000 : N ≡ 320 [MOD 1000] := by sorry

end permutation_count_mod_1000_l2396_239690


namespace train_length_l2396_239684

theorem train_length (platform1_length platform2_length : ℝ)
                     (platform1_time platform2_time : ℝ)
                     (h1 : platform1_length = 150)
                     (h2 : platform2_length = 250)
                     (h3 : platform1_time = 15)
                     (h4 : platform2_time = 20) :
  ∃ train_length : ℝ,
    train_length = 150 ∧
    (train_length + platform1_length) / platform1_time =
    (train_length + platform2_length) / platform2_time :=
by
  sorry


end train_length_l2396_239684


namespace quadratic_equations_solutions_l2396_239636

theorem quadratic_equations_solutions :
  (∀ x, 2 * x^2 - 4 * x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x, x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) := by
  sorry

end quadratic_equations_solutions_l2396_239636


namespace divisors_of_48n5_l2396_239632

/-- Given a positive integer n where 132n^3 has 132 positive integer divisors,
    48n^5 has 105 positive integer divisors -/
theorem divisors_of_48n5 (n : ℕ+) (h : (Nat.divisors (132 * n ^ 3)).card = 132) :
  (Nat.divisors (48 * n ^ 5)).card = 105 := by
  sorry

end divisors_of_48n5_l2396_239632


namespace inscribed_circle_radius_is_three_l2396_239623

/-- Right triangle PQR with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Angle R is a right angle -/
  angle_r_is_right : True

/-- Calculate the radius of the inscribed circle in a right triangle -/
def inscribedCircleRadius (t : RightTriangleWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed circle in the given right triangle is 3 -/
theorem inscribed_circle_radius_is_three :
  let t : RightTriangleWithInscribedCircle := ⟨15, 8, trivial⟩
  inscribedCircleRadius t = 3 := by
  sorry

end inscribed_circle_radius_is_three_l2396_239623


namespace no_odd_total_students_l2396_239626

theorem no_odd_total_students (B : ℕ) (T : ℕ) : 
  (T = B + (7.25 * B : ℚ).floor) → 
  (50 ≤ T ∧ T ≤ 150) → 
  ¬(T % 2 = 1) :=
by sorry

end no_odd_total_students_l2396_239626


namespace vector_magnitude_l2396_239662

theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = 20) →
  (b.1^2 + b.2^2 = 25) := by
    sorry

end vector_magnitude_l2396_239662


namespace shipwreck_year_conversion_l2396_239687

/-- Converts an octal number to its decimal equivalent -/
def octal_to_decimal (octal : Nat) : Nat :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal year of the shipwreck -/
def shipwreck_year_octal : Nat := 536

theorem shipwreck_year_conversion :
  octal_to_decimal shipwreck_year_octal = 350 := by
  sorry

end shipwreck_year_conversion_l2396_239687


namespace batsman_average_theorem_l2396_239673

def batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : ℕ :=
  let previous_average := (total_innings - 1) * (average_increase + (last_innings_score / total_innings))
  let new_total_score := previous_average + last_innings_score
  new_total_score / total_innings

theorem batsman_average_theorem :
  batsman_average 17 80 2 = 48 := by
  sorry

end batsman_average_theorem_l2396_239673


namespace complex_fraction_equality_l2396_239602

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x - y) / (2*x + 3*y) + (2*x + 3*y) / (x - y) = 2) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 34/15 := by
  sorry

end complex_fraction_equality_l2396_239602


namespace sqrt_24_times_sqrt_3_over_2_equals_6_l2396_239625

theorem sqrt_24_times_sqrt_3_over_2_equals_6 :
  Real.sqrt 24 * Real.sqrt (3/2) = 6 := by
  sorry

end sqrt_24_times_sqrt_3_over_2_equals_6_l2396_239625


namespace slope_product_negative_one_l2396_239670

/-- Two lines with slopes that differ by 45° and are negative reciprocals have a slope product of -1 -/
theorem slope_product_negative_one (m n : ℝ) : 
  (∃ θ : ℝ, m = Real.tan (θ + π/4) ∧ n = Real.tan θ) →  -- L₁ makes 45° larger angle than L₂
  m = -1/n →                                           -- slopes are negative reciprocals
  m * n = -1 :=                                        -- product of slopes is -1
by sorry

end slope_product_negative_one_l2396_239670


namespace angle_equality_l2396_239600

theorem angle_equality (θ : Real) (h1 : Real.sqrt 5 * Real.sin (15 * π / 180) = Real.cos θ + Real.sin θ) 
  (h2 : 0 < θ ∧ θ < π / 2) : θ = 30 * π / 180 := by
  sorry

end angle_equality_l2396_239600


namespace prime_sum_special_equation_l2396_239675

theorem prime_sum_special_equation (p q : ℕ) : 
  Prime p → Prime q → q^5 - 2*p^2 = 1 → p + q = 14 := by
  sorry

end prime_sum_special_equation_l2396_239675


namespace z_equals_four_when_x_is_five_l2396_239630

/-- The inverse relationship between 7z and x² -/
def inverse_relation (z x : ℝ) : Prop := ∃ k : ℝ, 7 * z = k / (x ^ 2)

/-- The theorem stating that given the inverse relationship and initial condition, z = 4 when x = 5 -/
theorem z_equals_four_when_x_is_five :
  ∀ z₀ : ℝ, inverse_relation z₀ 2 ∧ z₀ = 25 →
  ∃ z : ℝ, inverse_relation z 5 ∧ z = 4 :=
by
  sorry


end z_equals_four_when_x_is_five_l2396_239630


namespace perpendicular_construction_l2396_239663

-- Define the plane
structure Plane :=
  (Point : Type)
  (Line : Type)
  (on_line : Point → Line → Prop)
  (not_on_line : Point → Line → Prop)
  (draw_line : Point → Point → Line)
  (draw_perpendicular : Point → Line → Line)

-- Define the theorem
theorem perpendicular_construction 
  (P : Plane) (A : P.Point) (l : P.Line) (h : P.not_on_line A l) :
  ∃ (m : P.Line), P.on_line A m ∧ ∀ (X : P.Point), P.on_line X l → P.on_line X m → 
    ∃ (n : P.Line), P.on_line X n ∧ (∀ (Y : P.Point), P.on_line Y n → P.on_line Y m → Y = X) :=
sorry

end perpendicular_construction_l2396_239663


namespace estate_division_valid_l2396_239664

/-- Represents the estate division problem in Ancient Rome --/
structure EstateDivision where
  total_estate : ℕ
  son_share : ℕ
  daughter_share : ℕ
  wife_share : ℕ

/-- Checks if the given division is valid according to the problem constraints --/
def is_valid_division (d : EstateDivision) : Prop :=
  d.total_estate = 210 ∧
  d.son_share + d.daughter_share + d.wife_share = d.total_estate ∧
  d.son_share > d.daughter_share ∧
  d.son_share > d.wife_share ∧
  7 * d.son_share = 4 * d.total_estate ∧
  7 * d.daughter_share = d.total_estate ∧
  7 * d.wife_share = 2 * d.total_estate

/-- The proposed solution satisfies the constraints of the problem --/
theorem estate_division_valid : 
  is_valid_division ⟨210, 120, 30, 60⟩ := by
  sorry

#check estate_division_valid

end estate_division_valid_l2396_239664


namespace sin_monotone_interval_l2396_239601

theorem sin_monotone_interval (t : ℝ) : 
  (∀ x ∈ Set.Icc (-t) t, StrictMono (fun x ↦ Real.sin (2 * x + π / 6))) ↔ 
  t ∈ Set.Ioo 0 (π / 6) := by
  sorry

end sin_monotone_interval_l2396_239601


namespace f_unique_zero_and_inequality_l2396_239641

noncomputable section

variable (a : ℝ)

def f (x : ℝ) := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

def g (x : ℝ) := a * Real.exp x + x

theorem f_unique_zero_and_inequality (h : a ≥ 0) :
  (∃! x, f a x = 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ > -1 → x₂ > -1 → f a x₁ = g a x₁ - g a x₂ → x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2) :=
sorry

end

end f_unique_zero_and_inequality_l2396_239641


namespace selectStudents_eq_30_l2396_239660

/-- The number of ways to select 3 students from 4 boys and 3 girls, ensuring both genders are represented -/
def selectStudents : ℕ :=
  Nat.choose 4 2 * Nat.choose 3 1 + Nat.choose 4 1 * Nat.choose 3 2

/-- Theorem stating that the number of selections is 30 -/
theorem selectStudents_eq_30 : selectStudents = 30 := by
  sorry

end selectStudents_eq_30_l2396_239660


namespace correct_number_of_groups_l2396_239611

/-- The number of different groups of 3 marbles Tom can choose -/
def number_of_groups : ℕ := 16

/-- The number of red marbles Tom has -/
def red_marbles : ℕ := 1

/-- The number of blue marbles Tom has -/
def blue_marbles : ℕ := 1

/-- The number of black marbles Tom has -/
def black_marbles : ℕ := 1

/-- The number of white marbles Tom has -/
def white_marbles : ℕ := 4

/-- The total number of marbles Tom has -/
def total_marbles : ℕ := red_marbles + blue_marbles + black_marbles + white_marbles

/-- Theorem stating that the number of different groups of 3 marbles Tom can choose is correct -/
theorem correct_number_of_groups :
  number_of_groups = (Nat.choose white_marbles 3) + (Nat.choose 3 2 * Nat.choose white_marbles 1) :=
by sorry

end correct_number_of_groups_l2396_239611


namespace smaller_two_digit_factor_l2396_239609

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4761 → 
  min a b = 53 := by
sorry

end smaller_two_digit_factor_l2396_239609


namespace fraction_count_l2396_239699

theorem fraction_count : ∃ (S : Finset ℕ), 
  (∀ a ∈ S, a > 1 ∧ a < 7) ∧ 
  (∀ a ∉ S, a ≤ 1 ∨ a ≥ 7 ∨ (a - 1) / a ≥ 6 / 7) ∧
  S.card = 5 := by
  sorry

end fraction_count_l2396_239699


namespace correct_mark_calculation_l2396_239612

theorem correct_mark_calculation (n : ℕ) (initial_avg : ℚ) (wrong_mark : ℚ) (correct_avg : ℚ) :
  n = 10 →
  initial_avg = 100 →
  wrong_mark = 90 →
  correct_avg = 92 →
  ∃ x : ℚ, (n : ℚ) * initial_avg - wrong_mark + x = (n : ℚ) * correct_avg ∧ x = 10 :=
by sorry

end correct_mark_calculation_l2396_239612


namespace circle_condition_l2396_239651

/-- A circle in the xy-plane can be represented by an equation of the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. --/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The main theorem: if x^2 + y^2 - x + y + m = 0 represents a circle, then m < 1/2 --/
theorem circle_condition (m : ℝ) 
  (h : is_circle (fun x y => x^2 + y^2 - x + y + m)) : 
  m < (1/2 : ℝ) := by
  sorry

end circle_condition_l2396_239651


namespace toy_poodle_height_l2396_239661

/-- Heights of different poodle types -/
structure PoodleHeights where
  standard : ℝ
  miniature : ℝ
  toy : ℝ
  moyen : ℝ

/-- Conditions for poodle heights -/
def valid_poodle_heights (h : PoodleHeights) : Prop :=
  h.standard = h.miniature + 8.5 ∧
  h.miniature = h.toy + 6.25 ∧
  h.standard = h.moyen + 3.75 ∧
  h.moyen = h.toy + 4.75 ∧
  h.standard = 28

/-- Theorem: The toy poodle's height is 13.25 inches -/
theorem toy_poodle_height (h : PoodleHeights) 
  (hvalid : valid_poodle_heights h) : h.toy = 13.25 := by
  sorry

end toy_poodle_height_l2396_239661


namespace odd_function_theorem_l2396_239646

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_theorem (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_nonneg : ∀ x ≥ 0, f x = x^2 - 2*x) : 
  ∀ x, f x = x * (|x| - 2) := by
  sorry

end odd_function_theorem_l2396_239646


namespace restaurant_bill_example_l2396_239691

/-- Calculates the total cost for a group at a restaurant with specific pricing and discount rules. -/
def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (num_upgrades : ℕ) 
  (adult_meal_cost : ℚ) (upgrade_cost : ℚ) (adult_drink_cost : ℚ) (kid_drink_cost : ℚ) 
  (discount_rate : ℚ) : ℚ :=
  let num_adults := total_people - num_kids
  let meal_cost := num_adults * adult_meal_cost
  let upgrade_total := num_upgrades * upgrade_cost
  let drink_cost := num_adults * adult_drink_cost + num_kids * kid_drink_cost
  let subtotal := meal_cost + upgrade_total + drink_cost
  let discount := subtotal * discount_rate
  subtotal - discount

/-- Theorem stating that the total cost for the given group is $97.20 -/
theorem restaurant_bill_example : 
  restaurant_bill 11 2 4 8 4 2 1 (1/10) = 97.2 := by
  sorry

end restaurant_bill_example_l2396_239691


namespace top_book_cost_l2396_239610

/-- The cost of the "TOP" book -/
def top_cost : ℚ := 8

/-- The cost of the "ABC" book -/
def abc_cost : ℚ := 23

/-- The number of "TOP" books sold -/
def top_sold : ℕ := 13

/-- The number of "ABC" books sold -/
def abc_sold : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books -/
def earnings_difference : ℚ := 12

theorem top_book_cost :
  top_cost * top_sold - abc_cost * abc_sold = earnings_difference :=
sorry

end top_book_cost_l2396_239610


namespace train_crossing_time_l2396_239654

/-- The time taken for a train to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 700 ∧ train_speed_kmh = 125.99999999999999 →
  (train_length / (train_speed_kmh * (1000 / 3600))) = 20 := by
  sorry


end train_crossing_time_l2396_239654


namespace max_grandchildren_l2396_239672

/-- Calculates the number of grandchildren for a person with given conditions -/
def grandchildren_count (num_children : ℕ) (num_same_children : ℕ) (num_five_children : ℕ) (five_children : ℕ) : ℕ :=
  (num_same_children * num_children) + (num_five_children * five_children)

/-- Theorem stating that Max has 58 grandchildren -/
theorem max_grandchildren :
  let num_children := 8
  let num_same_children := 6
  let num_five_children := 2
  let five_children := 5
  grandchildren_count num_children num_same_children num_five_children five_children = 58 := by
  sorry

end max_grandchildren_l2396_239672


namespace hyperbola_eccentricity_l2396_239629

/-- Represents a hyperbola with equation y²/a² - x²/4 = 1 -/
structure Hyperbola where
  a : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  (1 / h.a^2 - 4 / 4 = 1) →  -- The hyperbola passes through (2, -1)
  eccentricity h = 3 := by
  sorry

end hyperbola_eccentricity_l2396_239629


namespace min_value_of_f_l2396_239604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then |x + a| + |x - 1| else x^2 - a*x + 2

theorem min_value_of_f (a : ℝ) : 
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) ↔ a ∈ ({-2 - 2*Real.sqrt 3, 2} : Set ℝ) :=
sorry

end min_value_of_f_l2396_239604


namespace knowledge_group_theorem_l2396_239681

/-- A group of people where some know each other -/
structure KnowledgeGroup (k : ℕ) where
  knows : Fin k → Fin k → Prop
  symm : ∀ i j, knows i j ↔ knows j i

/-- For any n people, there's an (n+1)-th person who knows them all -/
def HasKnowledgeable (n : ℕ) (g : KnowledgeGroup k) : Prop :=
  ∀ (s : Finset (Fin k)), s.card = n → 
    ∃ i, i ∉ s ∧ ∀ j ∈ s, g.knows i j

theorem knowledge_group_theorem (n : ℕ) :
  (∃ (g : KnowledgeGroup (2*n + 1)), HasKnowledgeable n g → 
    ∃ i, ∀ j, g.knows i j) ∧
  (∃ (g : KnowledgeGroup (2*n + 2)), HasKnowledgeable n g ∧ 
    ∀ i, ∃ j, ¬g.knows i j) := by
  sorry

end knowledge_group_theorem_l2396_239681


namespace min_value_expression_equality_condition_l2396_239605

theorem min_value_expression (x : ℝ) (h : x > 0) :
  2 + 3 * x + 4 / x ≥ 2 + 4 * Real.sqrt 3 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  2 + 3 * x + 4 / x = 2 + 4 * Real.sqrt 3 ↔ x = 2 * Real.sqrt 3 / 3 :=
by sorry

end min_value_expression_equality_condition_l2396_239605


namespace pigeonhole_disks_l2396_239616

/-- The number of distinct labels -/
def n : ℕ := 50

/-- The function that maps a label to the number of disks with that label -/
def f (i : ℕ) : ℕ := i

/-- The total number of disks -/
def total_disks : ℕ := n * (n + 1) / 2

/-- The minimum number of disks to guarantee at least 10 of the same label -/
def min_disks : ℕ := 415

theorem pigeonhole_disks :
  ∀ (S : Finset ℕ), S.card = min_disks →
  ∃ (i : ℕ), i ∈ Finset.range n ∧ (S.filter (λ x => x = i)).card ≥ 10 :=
by sorry

end pigeonhole_disks_l2396_239616


namespace train_meeting_theorem_l2396_239627

/-- Represents the meeting point of three trains given their speeds and departure times. -/
structure TrainMeeting where
  speed_a : ℝ
  speed_b : ℝ
  speed_c : ℝ
  time_b_after_a : ℝ
  time_c_after_b : ℝ

/-- Calculates the meeting point of three trains. -/
def calculate_meeting_point (tm : TrainMeeting) : ℝ × ℝ := sorry

/-- Theorem stating the correct speed of Train C and the meeting distance. -/
theorem train_meeting_theorem (tm : TrainMeeting) 
  (h1 : tm.speed_a = 30)
  (h2 : tm.speed_b = 36)
  (h3 : tm.time_b_after_a = 2)
  (h4 : tm.time_c_after_b = 1) :
  let (speed_c, distance) := calculate_meeting_point tm
  speed_c = 45 ∧ distance = 180 := by sorry

end train_meeting_theorem_l2396_239627
