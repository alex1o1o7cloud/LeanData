import Mathlib

namespace NUMINAMATH_CALUDE_password_probability_l3026_302689

def positive_single_digit_numbers : ℕ := 9
def alphabet_size : ℕ := 26
def vowels : ℕ := 5

def even_single_digit_numbers : ℕ := 4
def numbers_greater_than_five : ℕ := 4

theorem password_probability : 
  (even_single_digit_numbers : ℚ) / positive_single_digit_numbers *
  (vowels : ℚ) / alphabet_size *
  (numbers_greater_than_five : ℚ) / positive_single_digit_numbers = 40 / 1053 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l3026_302689


namespace NUMINAMATH_CALUDE_min_value_fraction_l3026_302655

theorem min_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + 2*c = 2) : 
  (a + b) / (a * b * c) ≥ 8 ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ + b₀ + 2*c₀ = 2 ∧ 
    (a₀ + b₀) / (a₀ * b₀ * c₀) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3026_302655


namespace NUMINAMATH_CALUDE_tenth_triangular_number_l3026_302668

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tenth_triangular_number : triangular_number 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_tenth_triangular_number_l3026_302668


namespace NUMINAMATH_CALUDE_power_five_2023_mod_11_l3026_302698

theorem power_five_2023_mod_11 : 5^2023 ≡ 4 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_five_2023_mod_11_l3026_302698


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3026_302616

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (1 : ℝ) / |x - 1| < 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 > 0}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3026_302616


namespace NUMINAMATH_CALUDE_first_player_wins_l3026_302654

/-- Represents the state of a Kayles game -/
structure KaylesGame where
  pins : List Bool
  turn : Nat

/-- Knocks over one pin or two adjacent pins -/
def makeMove (game : KaylesGame) (start : Nat) (count : Nat) : KaylesGame :=
  sorry

/-- Checks if the game is over (no pins left standing) -/
def isGameOver (game : KaylesGame) : Bool :=
  sorry

/-- Represents a strategy for playing Kayles -/
def Strategy := KaylesGame → Nat × Nat

/-- Checks if a strategy is winning for the current player -/
def isWinningStrategy (strat : Strategy) (game : KaylesGame) : Bool :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins :
  ∀ n : Nat, ∃ strat : Strategy, isWinningStrategy strat (KaylesGame.mk (List.replicate n true) 0) :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l3026_302654


namespace NUMINAMATH_CALUDE_pen_discount_problem_l3026_302624

theorem pen_discount_problem (original_price : ℝ) (h : original_price > 0) :
  let new_price := 0.75 * original_price
  let x := (25 : ℝ) / (1 / 0.75 - 1)
  x * original_price = (x + 25) * new_price → x = 75 :=
by sorry

end NUMINAMATH_CALUDE_pen_discount_problem_l3026_302624


namespace NUMINAMATH_CALUDE_total_rats_l3026_302611

/-- The number of rats each person has -/
structure RatCounts where
  hunter : ℕ
  elodie : ℕ
  kenia : ℕ
  teagan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (rc : RatCounts) : Prop :=
  rc.elodie = 30 ∧
  rc.elodie = rc.hunter + 10 ∧
  rc.kenia = 3 * (rc.hunter + rc.elodie) ∧
  ∃ p : ℚ, rc.teagan = rc.elodie + (p / 100) * rc.elodie ∧
           rc.teagan = rc.kenia - 5

/-- The theorem to be proved -/
theorem total_rats (rc : RatCounts) :
  satisfiesConditions rc → rc.hunter + rc.elodie + rc.kenia + rc.teagan = 345 :=
by
  sorry

end NUMINAMATH_CALUDE_total_rats_l3026_302611


namespace NUMINAMATH_CALUDE_stacys_farm_chickens_l3026_302696

theorem stacys_farm_chickens :
  ∀ (total_animals sick_animals piglets goats : ℕ),
    piglets = 40 →
    goats = 34 →
    sick_animals = 50 →
    2 * sick_animals = total_animals →
    total_animals = piglets + goats + (total_animals - piglets - goats) →
    total_animals - piglets - goats = 26 := by
  sorry

end NUMINAMATH_CALUDE_stacys_farm_chickens_l3026_302696


namespace NUMINAMATH_CALUDE_rain_probability_l3026_302672

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end NUMINAMATH_CALUDE_rain_probability_l3026_302672


namespace NUMINAMATH_CALUDE_same_gate_probability_proof_l3026_302600

/-- The number of ticket gates available -/
def num_gates : ℕ := 3

/-- The probability of two individuals selecting the same ticket gate -/
def same_gate_probability : ℚ := 1 / 3

/-- Theorem stating that the probability of two individuals selecting the same ticket gate
    out of three available gates is 1/3 -/
theorem same_gate_probability_proof :
  same_gate_probability = 1 / num_gates := by sorry

end NUMINAMATH_CALUDE_same_gate_probability_proof_l3026_302600


namespace NUMINAMATH_CALUDE_family_income_increase_l3026_302643

theorem family_income_increase (total_income : ℝ) 
  (masha_scholarship mother_salary father_salary grandfather_pension : ℝ) : 
  masha_scholarship + mother_salary + father_salary + grandfather_pension = total_income →
  masha_scholarship = 0.05 * total_income →
  mother_salary = 0.15 * total_income →
  father_salary = 0.25 * total_income →
  grandfather_pension = 0.55 * total_income :=
by
  sorry

#check family_income_increase

end NUMINAMATH_CALUDE_family_income_increase_l3026_302643


namespace NUMINAMATH_CALUDE_winnie_lollipops_theorem_l3026_302681

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total_lollipops friends : ℕ) : ℕ :=
  total_lollipops % friends

theorem winnie_lollipops_theorem (cherry wintergreen grape shrimp friends : ℕ) :
  let total_lollipops := cherry + wintergreen + grape + shrimp
  lollipops_kept total_lollipops friends = 
    total_lollipops - friends * (total_lollipops / friends) := by
  sorry

#eval lollipops_kept (67 + 154 + 23 + 312) 17

end NUMINAMATH_CALUDE_winnie_lollipops_theorem_l3026_302681


namespace NUMINAMATH_CALUDE_area_of_inner_triangle_l3026_302607

/-- Given a triangle and points dividing its sides in a 1:2 ratio, 
    the area of the new triangle formed by these points is 1/9 of the original triangle's area. -/
theorem area_of_inner_triangle (T : ℝ) (h : T > 0) :
  ∃ (A : ℝ), A = T / 9 ∧ A > 0 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inner_triangle_l3026_302607


namespace NUMINAMATH_CALUDE_tangent_line_inclination_angle_l3026_302679

/-- The curve y = x³ - 2x + 4 has a tangent line at (1, 3) with an inclination angle of 45° -/
theorem tangent_line_inclination_angle :
  let f (x : ℝ) := x^3 - 2*x + 4
  let f' (x : ℝ) := 3*x^2 - 2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let θ : ℝ := Real.pi / 4  -- 45° in radians
  (f x₀ = y₀) ∧ 
  (Real.tan θ = f' x₀) →
  θ = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_angle_l3026_302679


namespace NUMINAMATH_CALUDE_frustum_volume_l3026_302608

/-- The volume of a frustum with given ratio of radii and height, and slant height -/
theorem frustum_volume (r R h s : ℝ) (h1 : R = 4*r) (h2 : h = 4*r) (h3 : s = 10) 
  (h4 : s^2 = h^2 + (R - r)^2) : 
  (1/3 : ℝ) * Real.pi * h * (r^2 + R^2 + r*R) = 224 * Real.pi := by
  sorry

#check frustum_volume

end NUMINAMATH_CALUDE_frustum_volume_l3026_302608


namespace NUMINAMATH_CALUDE_rachel_chocolate_sales_l3026_302662

theorem rachel_chocolate_sales (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : 
  total_bars = 13 → price_per_bar = 2 → unsold_bars = 4 → 
  (total_bars - unsold_bars) * price_per_bar = 18 := by
sorry

end NUMINAMATH_CALUDE_rachel_chocolate_sales_l3026_302662


namespace NUMINAMATH_CALUDE_insect_count_l3026_302658

theorem insect_count (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 30) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 5 :=
by sorry

end NUMINAMATH_CALUDE_insect_count_l3026_302658


namespace NUMINAMATH_CALUDE_village_population_after_events_l3026_302623

theorem village_population_after_events (initial_population : ℕ) : 
  initial_population = 7800 → 
  (initial_population - initial_population / 10 - 
   (initial_population - initial_population / 10) / 4) = 5265 := by
sorry

end NUMINAMATH_CALUDE_village_population_after_events_l3026_302623


namespace NUMINAMATH_CALUDE_intersection_value_l3026_302692

-- Define the complex plane
variable (z : ℂ)

-- Define the first equation |z - 3| = 3|z + 3|
def equation1 (z : ℂ) : Prop := Complex.abs (z - 3) = 3 * Complex.abs (z + 3)

-- Define the second equation |z| = k
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the condition of intersection at exactly one point
def single_intersection (k : ℝ) : Prop :=
  ∃! z, equation1 z ∧ equation2 z k

-- The theorem to prove
theorem intersection_value :
  ∃! k, k > 0 ∧ single_intersection k ∧ k = 4.5 :=
sorry

end NUMINAMATH_CALUDE_intersection_value_l3026_302692


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l3026_302651

/-- Represents the engagement of a contractor --/
structure ContractorEngagement where
  daysWorked : ℕ
  daysAbsent : ℕ
  dailyWage : ℚ
  dailyFine : ℚ
  totalAmount : ℚ

/-- Theorem: Given the conditions, the contractor was engaged for 22 days --/
theorem contractor_engagement_days (c : ContractorEngagement) 
  (h1 : c.dailyWage = 25)
  (h2 : c.dailyFine = 7.5)
  (h3 : c.totalAmount = 490)
  (h4 : c.daysAbsent = 8) :
  c.daysWorked = 22 := by
  sorry


end NUMINAMATH_CALUDE_contractor_engagement_days_l3026_302651


namespace NUMINAMATH_CALUDE_max_value_xyz_l3026_302676

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024/14348907 :=
sorry

end NUMINAMATH_CALUDE_max_value_xyz_l3026_302676


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l3026_302694

theorem simple_interest_rate_for_doubling (principal : ℝ) (h : principal > 0) : 
  ∃ (rate : ℝ), 
    (rate > 0) ∧ 
    (rate < 100) ∧
    (principal * (1 + rate / 100 * 25) = 2 * principal) ∧
    (rate = 4) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l3026_302694


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3026_302649

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3026_302649


namespace NUMINAMATH_CALUDE_email_difference_l3026_302606

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 12

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := 44

/-- The difference between the number of emails Jack received in the morning and afternoon -/
theorem email_difference : morning_emails - afternoon_emails = 7 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l3026_302606


namespace NUMINAMATH_CALUDE_evaluate_g_l3026_302639

/-- The function g(x) = 3x^2 - 6x + 5 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- Theorem: 3g(2) + 2g(-4) = 169 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-4) = 169 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l3026_302639


namespace NUMINAMATH_CALUDE_worker_savings_fraction_l3026_302618

theorem worker_savings_fraction (P : ℝ) (S : ℝ) (h1 : P > 0) (h2 : 0 ≤ S ∧ S ≤ 1) 
  (h3 : 12 * S * P = 2 * (1 - S) * P) : S = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_fraction_l3026_302618


namespace NUMINAMATH_CALUDE_opposite_sides_range_l3026_302674

def line_equation (x y a : ℝ) : ℝ := x - y + a

theorem opposite_sides_range (a : ℝ) : 
  (line_equation 0 0 a) * (line_equation 1 (-1) a) < 0 ↔ -2 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l3026_302674


namespace NUMINAMATH_CALUDE_equality_holds_l3026_302693

-- Define the property P for the function f
def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f (x + y)| ≥ |f x + f y|

-- State the theorem
theorem equality_holds (f : ℝ → ℝ) (h : satisfies_inequality f) :
  ∀ x y : ℝ, |f (x + y)| = |f x + f y| := by
  sorry

end NUMINAMATH_CALUDE_equality_holds_l3026_302693


namespace NUMINAMATH_CALUDE_line_relationship_l3026_302637

-- Define a type for lines in 3D space
def Line : Type := ℝ × ℝ × ℝ → Prop

-- Define the relationships between lines
def Skew (l1 l2 : Line) : Prop := sorry

def Parallel (l1 l2 : Line) : Prop := sorry

def Intersecting (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : Skew a b) (h2 : Parallel c a) : 
  Intersecting c b ∨ Skew c b := by sorry

end NUMINAMATH_CALUDE_line_relationship_l3026_302637


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3026_302630

theorem fraction_zero_implies_x_one (x : ℝ) : (x - 1) / (x - 5) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3026_302630


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l3026_302631

-- Define the parallelogram properties
def base : ℝ := 20
def total_length : ℝ := 26
def slant_height : ℝ := 7

-- Define the area function for a parallelogram
def parallelogram_area (b h : ℝ) : ℝ := b * h

-- Theorem statement
theorem parallelogram_area_calculation :
  ∃ (height : ℝ), 
    height^2 + (total_length - base)^2 = slant_height^2 ∧
    parallelogram_area base height = 20 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l3026_302631


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3026_302635

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 9) :
  let perimeter := 2 * b + a
  perimeter = 22 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3026_302635


namespace NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_30_l3026_302687

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem largest_consecutive_nonprime_less_than_30 :
  ∃ (n : ℕ),
    isTwoDigit n ∧
    n < 30 ∧
    (∀ k : ℕ, k ∈ List.range 5 → ¬isPrime (n - k)) ∧
    (∀ m : ℕ, m > n → ¬(isTwoDigit m ∧ m < 30 ∧ (∀ k : ℕ, k ∈ List.range 5 → ¬isPrime (m - k)))) ∧
    n = 28 := by sorry

end NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_30_l3026_302687


namespace NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l3026_302695

/-- Calculates the number of seashells in Jar A after n weeks -/
def shellsInJarA (n : ℕ) : ℕ := sorry

/-- Calculates the number of seashells in Jar B after n weeks -/
def shellsInJarB (n : ℕ) : ℕ := sorry

/-- Calculates the total number of seashells in both jars after n weeks -/
def totalShells (n : ℕ) : ℕ := shellsInJarA n + shellsInJarB n

theorem seashell_count_after_six_weeks :
  shellsInJarA 0 = 50 →
  shellsInJarB 0 = 30 →
  (∀ k : ℕ, shellsInJarA (k + 1) = shellsInJarA k + 20) →
  (∀ k : ℕ, shellsInJarB (k + 1) = shellsInJarB k * 2) →
  (∀ k : ℕ, k % 3 = 0 → shellsInJarA (k + 1) = shellsInJarA k / 2) →
  (∀ k : ℕ, k % 3 = 0 → shellsInJarB (k + 1) = shellsInJarB k / 2) →
  totalShells 6 = 97 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l3026_302695


namespace NUMINAMATH_CALUDE_tangerines_remain_odd_last_fruit_is_tangerine_l3026_302683

/-- Represents the types of fruits in the vase -/
inductive Fruit
| Tangerine
| Apple

/-- Represents the state of the vase -/
structure VaseState where
  tangerines : Nat
  apples : Nat

/-- Represents the action of taking fruits -/
inductive TakeAction
| TwoTangerines
| TangerineAndApple
| TwoApples

/-- Function to update the vase state based on the take action -/
def updateVase (state : VaseState) (action : TakeAction) : VaseState :=
  match action with
  | TakeAction.TwoTangerines => 
      { tangerines := state.tangerines - 2, apples := state.apples + 1 }
  | TakeAction.TangerineAndApple => state
  | TakeAction.TwoApples => 
      { tangerines := state.tangerines, apples := state.apples - 1 }

/-- Theorem stating that the number of tangerines remains odd throughout the process -/
theorem tangerines_remain_odd (initial_tangerines : Nat) 
    (h_initial_odd : Odd initial_tangerines) 
    (actions : List TakeAction) :
    let final_state := actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }
    Odd final_state.tangerines ∧ final_state.tangerines > 0 := by
  sorry

/-- Theorem stating that the last fruit in the vase is a tangerine -/
theorem last_fruit_is_tangerine (initial_tangerines : Nat) 
    (h_initial_odd : Odd initial_tangerines) 
    (actions : List TakeAction) 
    (h_one_left : (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).tangerines + 
                  (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).apples = 1) :
    (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).tangerines = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_remain_odd_last_fruit_is_tangerine_l3026_302683


namespace NUMINAMATH_CALUDE_derivative_ln_x_over_x_l3026_302621

open Real

theorem derivative_ln_x_over_x (x : ℝ) (h : x > 0) :
  deriv (fun x => (log x) / x) x = (1 - log x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_x_over_x_l3026_302621


namespace NUMINAMATH_CALUDE_cat_dog_positions_after_365_moves_l3026_302680

/-- Represents the positions on the 3x3 grid --/
inductive GridPosition
  | TopLeft | TopCenter | TopRight
  | MiddleLeft | Center | MiddleRight
  | BottomLeft | BottomCenter | BottomRight

/-- Represents the edge positions on the 3x3 grid --/
inductive EdgePosition
  | LeftTop | LeftMiddle | LeftBottom
  | BottomLeft | BottomCenter | BottomRight
  | RightBottom | RightMiddle | RightTop
  | TopRight | TopCenter | TopLeft

/-- Calculates the cat's position after a given number of moves --/
def catPosition (moves : ℕ) : GridPosition :=
  match moves % 9 with
  | 0 => GridPosition.TopLeft
  | 1 => GridPosition.TopCenter
  | 2 => GridPosition.TopRight
  | 3 => GridPosition.MiddleRight
  | 4 => GridPosition.BottomRight
  | 5 => GridPosition.BottomCenter
  | 6 => GridPosition.BottomLeft
  | 7 => GridPosition.MiddleLeft
  | _ => GridPosition.Center

/-- Calculates the dog's position after a given number of moves --/
def dogPosition (moves : ℕ) : EdgePosition :=
  match moves % 16 with
  | 0 => EdgePosition.LeftMiddle
  | 1 => EdgePosition.LeftTop
  | 2 => EdgePosition.TopLeft
  | 3 => EdgePosition.TopCenter
  | 4 => EdgePosition.TopRight
  | 5 => EdgePosition.RightTop
  | 6 => EdgePosition.RightMiddle
  | 7 => EdgePosition.RightBottom
  | 8 => EdgePosition.BottomRight
  | 9 => EdgePosition.BottomCenter
  | 10 => EdgePosition.BottomLeft
  | 11 => EdgePosition.LeftBottom
  | 12 => EdgePosition.LeftMiddle
  | 13 => EdgePosition.LeftTop
  | 14 => EdgePosition.TopLeft
  | _ => EdgePosition.TopCenter

theorem cat_dog_positions_after_365_moves :
  catPosition 365 = GridPosition.Center ∧ dogPosition 365 = EdgePosition.LeftMiddle :=
sorry

end NUMINAMATH_CALUDE_cat_dog_positions_after_365_moves_l3026_302680


namespace NUMINAMATH_CALUDE_fraction_product_l3026_302619

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5 : ℚ) * (3 / 6 : ℚ) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3026_302619


namespace NUMINAMATH_CALUDE_smarties_remainder_l3026_302634

theorem smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l3026_302634


namespace NUMINAMATH_CALUDE_h_j_composition_l3026_302629

theorem h_j_composition (c d : ℝ) (h : ℝ → ℝ) (j : ℝ → ℝ)
  (h_def : ∀ x, h x = c * x + d)
  (j_def : ∀ x, j x = 3 * x - 4)
  (composition : ∀ x, j (h x) = 4 * x + 3) :
  c + d = 11 / 3 := by
sorry

end NUMINAMATH_CALUDE_h_j_composition_l3026_302629


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l3026_302642

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given values is approximately 18.58%. -/
theorem retailer_profit_percent :
  let ε := 0.01
  let result := profit_percent 225 28 300
  (result > 18.58 - ε) ∧ (result < 18.58 + ε) :=
sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l3026_302642


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3026_302601

/-- Represents the theater ticket sales problem --/
theorem theater_ticket_sales 
  (orchestra_price : ℕ) 
  (balcony_price : ℕ) 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (h1 : orchestra_price = 12)
  (h2 : balcony_price = 8)
  (h3 : total_tickets = 340)
  (h4 : total_revenue = 3320) :
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_revenue ∧
    balcony_tickets - orchestra_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3026_302601


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3026_302673

theorem inequality_solution_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3026_302673


namespace NUMINAMATH_CALUDE_max_circle_sum_is_15_l3026_302665

/-- Represents a configuration of numbers in the circle diagram -/
def CircleConfiguration := Fin 7 → Fin 7

/-- The sum of numbers in a given circle of the configuration -/
def circle_sum (config : CircleConfiguration) (circle : Fin 3) : ℕ :=
  sorry

/-- Checks if a configuration is valid (uses all numbers 0 to 6 exactly once) -/
def is_valid_configuration (config : CircleConfiguration) : Prop :=
  sorry

/-- Checks if all circles in a configuration have the same sum -/
def all_circles_equal_sum (config : CircleConfiguration) : Prop :=
  sorry

/-- The maximum possible sum for each circle -/
def max_circle_sum : ℕ := 15

theorem max_circle_sum_is_15 :
  ∃ (config : CircleConfiguration),
    is_valid_configuration config ∧
    all_circles_equal_sum config ∧
    ∀ (c : Fin 3), circle_sum config c = max_circle_sum ∧
    ∀ (config' : CircleConfiguration),
      is_valid_configuration config' →
      all_circles_equal_sum config' →
      ∀ (c : Fin 3), circle_sum config' c ≤ max_circle_sum :=
sorry

end NUMINAMATH_CALUDE_max_circle_sum_is_15_l3026_302665


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l3026_302666

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l3026_302666


namespace NUMINAMATH_CALUDE_prove_certain_number_l3026_302663

def w : ℕ := 468
def certain_number : ℕ := 2028

theorem prove_certain_number :
  (∃ (n : ℕ), (2^4 ∣ n * w) ∧ (3^3 ∣ n * w) ∧ (13^3 ∣ n * w)) ∧
  (∀ (x : ℕ), x < w → ¬(∃ (m : ℕ), (2^4 ∣ m * x) ∧ (3^3 ∣ m * x) ∧ (13^3 ∣ m * x))) →
  certain_number * w % 2^4 = 0 ∧
  certain_number * w % 3^3 = 0 ∧
  certain_number * w % 13^3 = 0 ∧
  (∀ (y : ℕ), y < certain_number →
    (y * w % 2^4 ≠ 0 ∨ y * w % 3^3 ≠ 0 ∨ y * w % 13^3 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_prove_certain_number_l3026_302663


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3026_302684

theorem polynomial_remainder (a b : ℤ) : 
  (∀ x : ℤ, ∃ q : ℤ, x^3 - 2*x^2 + a*x + b = (x-1)*(x-2)*q + (2*x + 1)) → 
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3026_302684


namespace NUMINAMATH_CALUDE_bottle_problem_l3026_302617

/-- Represents a bottle in the case -/
inductive Bottle
  | FirstPrize
  | SecondPrize
  | NoPrize

/-- Represents the case of bottles -/
def Case : Finset Bottle := sorry

/-- The number of bottles in the case -/
def caseSize : ℕ := 6

/-- The number of bottles with prizes -/
def prizeBottles : ℕ := 2

/-- The number of bottles without prizes -/
def noPrizeBottles : ℕ := 4

/-- Person A's selection of bottles -/
def Selection : Finset Bottle := sorry

/-- The number of bottles selected -/
def selectionSize : ℕ := 2

/-- Event A: A did not win a prize -/
def EventA : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ ∀ b ∈ s, b = Bottle.NoPrize}

/-- Event B: A won the first prize -/
def EventB : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ Bottle.FirstPrize ∈ s}

/-- Event C: A won a prize -/
def EventC : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ (Bottle.FirstPrize ∈ s ∨ Bottle.SecondPrize ∈ s)}

/-- The probability measure on the sample space -/
noncomputable def P : Set (Finset Bottle) → ℝ := sorry

theorem bottle_problem :
  (EventA ∩ EventC = ∅) ∧ (P (EventB ∪ EventC) = P EventC) := by sorry

end NUMINAMATH_CALUDE_bottle_problem_l3026_302617


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l3026_302604

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3360

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 330

theorem vegetable_ghee_weight : 
  weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
  weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume = total_weight := by
sorry

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l3026_302604


namespace NUMINAMATH_CALUDE_x_plus_one_is_linear_l3026_302609

/-- A linear equation is an equation with variables of only the first power -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

/-- The function representing x + 1 = 0 -/
def f (x : ℝ) : ℝ := x + 1

theorem x_plus_one_is_linear : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_is_linear_l3026_302609


namespace NUMINAMATH_CALUDE_smallest_number_l3026_302638

def number_set : Finset ℕ := {5, 9, 10, 2}

theorem smallest_number : 
  ∃ (x : ℕ), x ∈ number_set ∧ ∀ y ∈ number_set, x ≤ y ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3026_302638


namespace NUMINAMATH_CALUDE_units_digit_of_1583_pow_1246_l3026_302660

theorem units_digit_of_1583_pow_1246 : ∃ n : ℕ, 1583^1246 ≡ 9 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_1583_pow_1246_l3026_302660


namespace NUMINAMATH_CALUDE_train_crossing_bridge_l3026_302677

/-- A train crossing a bridge problem -/
theorem train_crossing_bridge (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  train_speed_kmph = 18 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_l3026_302677


namespace NUMINAMATH_CALUDE_only_satisfying_sets_l3026_302682

/-- A set of four real numbers satisfying the given condition -/
def SatisfyingSet (a b c d : ℝ) : Prop :=
  a + b*c*d = 2 ∧ b + a*c*d = 2 ∧ c + a*b*d = 2 ∧ d + a*b*c = 2

/-- The theorem stating the only satisfying sets -/
theorem only_satisfying_sets :
  ∀ a b c d : ℝ, SatisfyingSet a b c d ↔ 
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
    (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = 3) ∨
    (a = -1 ∧ b = -1 ∧ c = 3 ∧ d = -1) ∨
    (a = -1 ∧ b = 3 ∧ c = -1 ∧ d = -1) ∨
    (a = 3 ∧ b = -1 ∧ c = -1 ∧ d = -1) :=
by sorry

end NUMINAMATH_CALUDE_only_satisfying_sets_l3026_302682


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l3026_302602

theorem parabola_point_to_directrix_distance 
  (C : ℝ → ℝ → Prop) 
  (p : ℝ) 
  (A : ℝ × ℝ) :
  (∀ x y, C x y ↔ y^2 = 2*p*x) →  -- Definition of parabola C
  C A.1 A.2 →  -- A lies on C
  A = (1, Real.sqrt 5) →  -- Coordinates of A
  (A.1 + p/2) = 9/4 :=  -- Distance formula to directrix
by sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l3026_302602


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3026_302659

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin (a, 4) (-3, b) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3026_302659


namespace NUMINAMATH_CALUDE_unique_pythagorean_triple_l3026_302664

/-- A function to check if a triple of natural numbers is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The theorem stating that (5, 12, 13) is the only Pythagorean triple among the given options -/
theorem unique_pythagorean_triple :
  ¬ isPythagoreanTriple 3 4 5 ∧
  ¬ isPythagoreanTriple 1 1 2 ∧
  isPythagoreanTriple 5 12 13 ∧
  ¬ isPythagoreanTriple 1 3 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_pythagorean_triple_l3026_302664


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l3026_302632

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem perpendicular_parallel_implies_perpendicular 
  (l1 l2 : Line3D) (α : Plane3D) :
  perpendicular_line_plane l1 α → parallel_line_plane l2 α → 
  perpendicular_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l3026_302632


namespace NUMINAMATH_CALUDE_sum_of_three_odd_squares_l3026_302671

theorem sum_of_three_odd_squares (a b c : ℕ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- pairwise different
  (∃ k l m : ℕ, a = 2*k + 1 ∧ b = 2*l + 1 ∧ c = 2*m + 1) →  -- odd integers
  (∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℕ, a^2 + b^2 + c^2 = x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 + x₆^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_odd_squares_l3026_302671


namespace NUMINAMATH_CALUDE_mutual_fund_investment_l3026_302685

theorem mutual_fund_investment
  (total_investment : ℝ)
  (mutual_fund_ratio : ℝ)
  (h1 : total_investment = 250000)
  (h2 : mutual_fund_ratio = 3) :
  let commodity_investment := total_investment / (1 + mutual_fund_ratio)
  let mutual_fund_investment := mutual_fund_ratio * commodity_investment
  mutual_fund_investment = 187500 := by
sorry

end NUMINAMATH_CALUDE_mutual_fund_investment_l3026_302685


namespace NUMINAMATH_CALUDE_binomial_odd_even_difference_squares_l3026_302615

variable (x a : ℝ) (n : ℕ)

def A (x a : ℝ) (n : ℕ) : ℝ := sorry
def B (x a : ℝ) (n : ℕ) : ℝ := sorry

/-- For the binomial expansion (x+a)^n, where A is the sum of odd-position terms
    and B is the sum of even-position terms, A^2 - B^2 = (x^2 - a^2)^n -/
theorem binomial_odd_even_difference_squares :
  (A x a n)^2 - (B x a n)^2 = (x^2 - a^2)^n := by sorry

end NUMINAMATH_CALUDE_binomial_odd_even_difference_squares_l3026_302615


namespace NUMINAMATH_CALUDE_smallest_number_problem_l3026_302620

theorem smallest_number_problem (a b c : ℕ) 
  (h1 : 0 < a ∧ a < b ∧ b < c)
  (h2 : (a + b + c) / 3 = 30)
  (h3 : b = 29)
  (h4 : c = b + 6) :
  a = 26 := by sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l3026_302620


namespace NUMINAMATH_CALUDE_roots_of_quadratic_with_absolute_value_l3026_302626

theorem roots_of_quadratic_with_absolute_value
  (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (n : ℕ), n ≤ 4 ∧
  ∃ (roots : Finset ℂ), roots.card = n ∧
  ∀ z ∈ roots, a * z^2 + b * Complex.abs z + c = 0 :=
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_with_absolute_value_l3026_302626


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3026_302667

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4)^2 → area = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3026_302667


namespace NUMINAMATH_CALUDE_square_side_length_l3026_302650

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = d * d / 2 ∧ s = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3026_302650


namespace NUMINAMATH_CALUDE_percent_commutation_l3026_302690

theorem percent_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 45) : 0.4 * (0.3 * x) = 45 := by
  sorry

end NUMINAMATH_CALUDE_percent_commutation_l3026_302690


namespace NUMINAMATH_CALUDE_ninth_term_is_twelve_l3026_302686

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 5 + a 7 = 16
  third_term : a 3 = 4

/-- The 9th term of the arithmetic sequence is 12 -/
theorem ninth_term_is_twelve (seq : ArithmeticSequence) : seq.a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_twelve_l3026_302686


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l3026_302640

/-- The distance between two complex numbers 2+3i and -2+2i is √17 -/
theorem distance_between_complex_points : 
  Complex.abs ((2 : ℂ) + 3*I - ((-2 : ℂ) + 2*I)) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l3026_302640


namespace NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_problem_l3026_302669

/-- Calculates the interest rate of an additional investment given initial investment parameters and desired total return rate. -/
theorem additional_investment_rate 
  (initial_investment : ℝ) 
  (initial_rate : ℝ) 
  (additional_investment : ℝ) 
  (total_rate : ℝ) : ℝ :=
  let total_investment := initial_investment + additional_investment
  let initial_income := initial_investment * initial_rate
  let total_desired_income := total_investment * total_rate
  let additional_income_needed := total_desired_income - initial_income
  additional_income_needed / additional_investment

/-- Proves that the additional investment rate is 0.08 (8%) given the specific problem parameters. -/
theorem barbata_investment_problem : 
  additional_investment_rate 1400 0.05 700 0.06 = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_problem_l3026_302669


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3026_302697

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3026_302697


namespace NUMINAMATH_CALUDE_rex_cards_left_is_150_l3026_302652

/-- The number of Pokemon cards Rex has left after dividing his collection --/
def rexCardsLeft (nicolesCards : ℕ) : ℕ :=
  let cindysCards := nicolesCards * 2
  let totalCards := nicolesCards + cindysCards
  let rexCards := totalCards / 2
  rexCards / 4

/-- Theorem stating that Rex has 150 cards left --/
theorem rex_cards_left_is_150 : rexCardsLeft 400 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_is_150_l3026_302652


namespace NUMINAMATH_CALUDE_total_is_260_l3026_302699

/-- Represents the ratio of money shared among four people -/
structure MoneyRatio :=
  (a b c d : ℕ)

/-- Calculates the total amount of money shared given a ratio and the first person's share -/
def totalShared (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  firstShare * (ratio.a + ratio.b + ratio.c + ratio.d)

/-- Theorem stating that for the given ratio and first share, the total is 260 -/
theorem total_is_260 (ratio : MoneyRatio) (h1 : ratio.a = 1) (h2 : ratio.b = 2) 
    (h3 : ratio.c = 7) (h4 : ratio.d = 3) (h5 : firstShare = 20) : 
    totalShared ratio firstShare = 260 := by
  sorry


end NUMINAMATH_CALUDE_total_is_260_l3026_302699


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3026_302646

/-- 
Given a line L1 with equation 2x - y + 1 = 0, and a point P (1, 1),
prove that the line L2 passing through P and parallel to L1 has the equation 2x - y - 1 = 0.
-/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x - y + 1 = 0) →  -- L1 equation
  (∃ c : ℝ, 2 * x - y + c = 0 ∧ 2 * 1 - 1 + c = 0) →  -- L2 passes through (1, 1)
  (2 * x - y - 1 = 0)  -- L2 equation
:= by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3026_302646


namespace NUMINAMATH_CALUDE_doug_marbles_l3026_302612

theorem doug_marbles (ed_initial : ℕ) (doug_initial : ℕ) (ed_lost : ℕ) (ed_current : ℕ) 
  (h1 : ed_initial = doug_initial + 12)
  (h2 : ed_lost = 20)
  (h3 : ed_current = 17)
  (h4 : ed_initial = ed_current + ed_lost) : 
  doug_initial = 25 := by
sorry

end NUMINAMATH_CALUDE_doug_marbles_l3026_302612


namespace NUMINAMATH_CALUDE_mario_garden_blossoms_l3026_302647

/-- Calculates the total number of blossoms in Mario's garden after a given number of weeks. -/
def total_blossoms (weeks : ℕ) : ℕ :=
  let hibiscus1 := 2 + 3 * weeks
  let hibiscus2 := 4 + 4 * weeks
  let hibiscus3 := 16 + 5 * weeks
  let rose1 := 3 + 2 * weeks
  let rose2 := 5 + 3 * weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2

/-- Theorem stating that the total number of blossoms in Mario's garden after 2 weeks is 64. -/
theorem mario_garden_blossoms : total_blossoms 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_mario_garden_blossoms_l3026_302647


namespace NUMINAMATH_CALUDE_equation_solution_l3026_302653

theorem equation_solution : ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3026_302653


namespace NUMINAMATH_CALUDE_negation_equivalence_l3026_302645

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 0, x^2 ≤ |x|) ↔ (∀ x ∈ Set.Ioo (-1 : ℝ) 0, x^2 > |x|) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3026_302645


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l3026_302675

-- Define the number of each vegetable Conor can chop in a day
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8

-- Define the number of days Conor works per week
def work_days_per_week : ℕ := 4

-- Theorem to prove
theorem conor_weekly_vegetables :
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := by
  sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l3026_302675


namespace NUMINAMATH_CALUDE_probability_same_color_l3026_302633

/-- The number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- The number of colors -/
def num_colors : ℕ := 3

/-- The total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- The number of marbles Cheryl picks -/
def picked_marbles : ℕ := 3

/-- The probability of picking 3 marbles of the same color -/
theorem probability_same_color :
  (num_colors * Nat.choose marbles_per_color picked_marbles * 
   Nat.choose (total_marbles - picked_marbles) (total_marbles - 2 * picked_marbles)) /
  Nat.choose total_marbles picked_marbles = 1 / 28 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_l3026_302633


namespace NUMINAMATH_CALUDE_acute_triangle_tangent_difference_range_l3026_302691

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² - a² = ac, then 1 < 1/tan(A) - 1/tan(B) < 2√3/3 -/
theorem acute_triangle_tangent_difference_range 
  (A B C : ℝ) (a b c : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sides : b^2 - a^2 = a*c) :
  1 < 1 / Real.tan A - 1 / Real.tan B ∧ 
  1 / Real.tan A - 1 / Real.tan B < 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_tangent_difference_range_l3026_302691


namespace NUMINAMATH_CALUDE_bottle_refund_amount_l3026_302636

def debt : ℚ := 90
def twenty_bills : ℕ := 2
def ten_bills : ℕ := 4
def bottles : ℕ := 20

theorem bottle_refund_amount : 
  let cash : ℚ := (20 * twenty_bills + 10 * ten_bills)
  let shortage : ℚ := debt - cash
  shortage / bottles = 1/2 := by sorry

end NUMINAMATH_CALUDE_bottle_refund_amount_l3026_302636


namespace NUMINAMATH_CALUDE_negation_of_p_l3026_302670

def p : Prop := ∀ x : ℝ, Real.sqrt (2 - x) < 0

theorem negation_of_p : ¬p ↔ ∃ x₀ : ℝ, Real.sqrt (2 - x₀) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_p_l3026_302670


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3026_302625

/-- Given two points A(2, 1) and B(m, 4), prove that the range of the inclination angle α 
    of line AB is [π/6, 2π/3] when m ∈ [2-√3, 2+3√3] -/
theorem inclination_angle_range (m : ℝ) (h : m ∈ Set.Icc (2 - Real.sqrt 3) (2 + 3 * Real.sqrt 3)) :
  let α := if m = 2 then π/2 else Real.arctan ((4 - 1) / (m - 2))
  α ∈ Set.Icc (π/6) (2*π/3) :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3026_302625


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l3026_302622

theorem grocery_store_inventory (ordered : ℕ) (sold : ℕ) (storeroom : ℕ) 
  (h1 : ordered = 4458)
  (h2 : sold = 1561)
  (h3 : storeroom = 575) :
  ordered - sold + storeroom = 3472 :=
by sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l3026_302622


namespace NUMINAMATH_CALUDE_johns_initial_squat_weight_l3026_302614

/-- Calculates John's initial squat weight based on given conditions --/
theorem johns_initial_squat_weight :
  ∀ (initial_bench initial_deadlift new_total : ℝ),
  initial_bench = 400 →
  initial_deadlift = 800 →
  new_total = 1490 →
  ∃ (initial_squat : ℝ),
    initial_squat * 0.7 + initial_bench + (initial_deadlift - 200) = new_total ∧
    initial_squat = 700 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_initial_squat_weight_l3026_302614


namespace NUMINAMATH_CALUDE_tree_height_average_l3026_302688

def tree_heights (n : ℕ) : Type := Fin n → ℕ

def valid_heights (h : tree_heights 7) : Prop :=
  h 1 = 16 ∧
  ∀ i : Fin 6, (h i = 2 * h i.succ ∨ 2 * h i = h i.succ)

def average_height (h : tree_heights 7) : ℚ :=
  (h 0 + h 1 + h 2 + h 3 + h 4 + h 5 + h 6 : ℚ) / 7

theorem tree_height_average (h : tree_heights 7) 
  (hvalid : valid_heights h) : average_height h = 145.1 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_average_l3026_302688


namespace NUMINAMATH_CALUDE_half_dollar_percentage_l3026_302613

theorem half_dollar_percentage : 
  let nickel_count : ℕ := 80
  let half_dollar_count : ℕ := 40
  let nickel_value : ℕ := 5
  let half_dollar_value : ℕ := 50
  let total_value := nickel_count * nickel_value + half_dollar_count * half_dollar_value
  let half_dollar_total := half_dollar_count * half_dollar_value
  (half_dollar_total : ℚ) / total_value = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_half_dollar_percentage_l3026_302613


namespace NUMINAMATH_CALUDE_string_length_for_circular_token_l3026_302605

theorem string_length_for_circular_token : 
  let area : ℝ := 616
  let pi_approx : ℝ := 22 / 7
  let extra_length : ℝ := 5
  let radius : ℝ := Real.sqrt (area * 7 / 22)
  let circumference : ℝ := 2 * pi_approx * radius
  circumference + extra_length = 93 := by sorry

end NUMINAMATH_CALUDE_string_length_for_circular_token_l3026_302605


namespace NUMINAMATH_CALUDE_quadratic_x_axis_intersection_l3026_302641

theorem quadratic_x_axis_intersection (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_x_axis_intersection_l3026_302641


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l3026_302628

/-- Given two intersecting lines and a perpendicular line, prove the equations of a line through the intersection point and its symmetric line. -/
theorem intersection_and_perpendicular_lines
  (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) (perp_line : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x - 2*y + 4 = 0)
  (h2 : ∀ x y, line2 x y ↔ x + y - 2 = 0)
  (h3 : ∀ x y, perp_line x y ↔ 5*x + 3*y - 6 = 0)
  (P : ℝ × ℝ) (hP : line1 P.1 P.2 ∧ line2 P.1 P.2)
  (l : ℝ → ℝ → Prop) (hl : l P.1 P.2)
  (hperp : ∀ x y, l x y → (5 * (y - P.2) = -3 * (x - P.1))) :
  (∀ x y, l x y ↔ 3*x - 5*y + 10 = 0) ∧
  (∀ x y, (3*x - 5*y - 10 = 0) ↔ (l (-x) (-y))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l3026_302628


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l3026_302644

theorem floor_ceil_sum : ⌊(0.998 : ℝ)⌋ + ⌈(2.002 : ℝ)⌉ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l3026_302644


namespace NUMINAMATH_CALUDE_acute_angles_inequality_l3026_302661

theorem acute_angles_inequality (α β : Real) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) : 
  Real.sin α ^ 3 * Real.cos β ^ 3 + Real.sin α ^ 3 * Real.sin β ^ 3 + Real.cos α ^ 3 ≥ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_inequality_l3026_302661


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l3026_302610

theorem cubic_roots_determinant (p q : ℝ) (a b c : ℝ) : 
  a^3 + p*a + q = 0 → 
  b^3 + p*b + q = 0 → 
  c^3 + p*c + q = 0 → 
  Matrix.det !![1 + a, 1, 1; 1, 1 + b, 1; 1, 1, 1 + c] = p - q := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l3026_302610


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3026_302648

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_with_complement : A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3026_302648


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l3026_302656

theorem remainder_sum_powers_mod_5 :
  (Nat.pow 9 7 + Nat.pow 4 5 + Nat.pow 3 9) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l3026_302656


namespace NUMINAMATH_CALUDE_special_function_property_l3026_302657

def non_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem special_function_property :
  ∀ f : ℝ → ℝ,
  (∀ x ∈ Set.Icc 0 1, non_decreasing f 0 1) →
  f 0 = 0 →
  (∀ x ∈ Set.Icc 0 1, f (x / 3) = (1 / 2) * f x) →
  (∀ x ∈ Set.Icc 0 1, f (1 - x) = 1 - f x) →
  f (1 / 3) + f (1 / 8) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_special_function_property_l3026_302657


namespace NUMINAMATH_CALUDE_ronald_egg_sharing_l3026_302603

def total_eggs : ℕ := 16
def eggs_per_friend : ℕ := 2

theorem ronald_egg_sharing :
  total_eggs / eggs_per_friend = 8 := by sorry

end NUMINAMATH_CALUDE_ronald_egg_sharing_l3026_302603


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_seven_fifths_l3026_302627

theorem sqrt_fraction_equals_seven_fifths :
  (Real.sqrt 64 + Real.sqrt 36) / Real.sqrt (64 + 36) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_seven_fifths_l3026_302627


namespace NUMINAMATH_CALUDE_printer_ratio_l3026_302678

theorem printer_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hx_time : x = 16) (hy_time : y = 12) (hz_time : z = 8) :
  x / ((1 / y + 1 / z)⁻¹) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_printer_ratio_l3026_302678
