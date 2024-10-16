import Mathlib

namespace NUMINAMATH_CALUDE_factor_implies_b_value_l3259_325941

theorem factor_implies_b_value (b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, 9*x^2 + b*x + 44 = (3*x + 4) * k) → b = 45 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l3259_325941


namespace NUMINAMATH_CALUDE_eliminate_alpha_l3259_325971

theorem eliminate_alpha (x y : ℝ) (α : ℝ) 
  (hx : x = Real.tan α ^ 2) 
  (hy : y = Real.sin α ^ 2) : 
  x - y = x * y := by
  sorry

end NUMINAMATH_CALUDE_eliminate_alpha_l3259_325971


namespace NUMINAMATH_CALUDE_heximal_binary_equality_l3259_325950

/-- Converts a heximal (base-6) number to decimal --/
def heximal_to_decimal (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6^1 + d * 6^0

/-- Converts a binary number to decimal --/
def binary_to_decimal (a b c d e f g h : ℕ) : ℕ :=
  a * 2^7 + b * 2^6 + c * 2^5 + d * 2^4 + e * 2^3 + f * 2^2 + g * 2^1 + h * 2^0

/-- The theorem stating that k = 3 is the unique solution --/
theorem heximal_binary_equality :
  ∃! k : ℕ, k > 0 ∧ heximal_to_decimal 1 0 k 5 = binary_to_decimal 1 1 1 0 1 1 1 1 :=
by sorry

end NUMINAMATH_CALUDE_heximal_binary_equality_l3259_325950


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3259_325960

-- Define an isosceles triangle with side lengths 5, 5, and 2
def isoscelesTriangle (a b c : ℝ) : Prop :=
  a = 5 ∧ b = 5 ∧ c = 2

-- Define the perimeter of a triangle
def trianglePerimeter (a b c : ℝ) : ℝ :=
  a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, isoscelesTriangle a b c → trianglePerimeter a b c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3259_325960


namespace NUMINAMATH_CALUDE_transformed_is_ellipse_l3259_325938

-- Define the original circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scaling_transformation (x y : ℝ) : ℝ × ℝ := (5*x, 4*y)

-- Define the resulting equation after transformation
def transformed_equation (x' y' : ℝ) : Prop :=
  ∃ x y, circle_equation x y ∧ scaling_transformation x y = (x', y')

-- Statement to prove
theorem transformed_is_ellipse :
  ∃ a b, a > b ∧ a = 5 ∧
  ∀ x' y', transformed_equation x' y' ↔ (x'^2 / a^2) + (y'^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_transformed_is_ellipse_l3259_325938


namespace NUMINAMATH_CALUDE_fraction_under_21_l3259_325912

theorem fraction_under_21 (total : ℕ) (under_21 : ℕ) (over_65 : ℚ) :
  total > 50 →
  total < 100 →
  over_65 = 5/10 →
  under_21 = 30 →
  (under_21 : ℚ) / total = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_under_21_l3259_325912


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3259_325979

theorem no_solution_for_equation : 
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 2 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3259_325979


namespace NUMINAMATH_CALUDE_marching_band_members_l3259_325983

theorem marching_band_members : ∃ n : ℕ,
  150 < n ∧ n < 250 ∧
  n % 3 = 1 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  (∀ m : ℕ, 150 < m ∧ m < n →
    ¬(m % 3 = 1 ∧ m % 6 = 2 ∧ m % 8 = 3)) ∧
  n = 203 :=
by sorry

end NUMINAMATH_CALUDE_marching_band_members_l3259_325983


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3259_325948

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 →  -- Two sides are 5, one side is 2
  a = b →                  -- The triangle is isosceles
  a + b + c = 12 :=        -- The perimeter is 12
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3259_325948


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l3259_325982

theorem two_distinct_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + k = 0 ∧ x₂^2 - 2*x₂ + k = 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l3259_325982


namespace NUMINAMATH_CALUDE_even_operations_l3259_325986

theorem even_operations (a b : ℤ) (ha : Even a) (hb : Odd b) : 
  Even (a * b) ∧ Even (a * a) := by
  sorry

end NUMINAMATH_CALUDE_even_operations_l3259_325986


namespace NUMINAMATH_CALUDE_water_fraction_in_mixture_l3259_325947

/-- Given a cement mixture with total weight, sand fraction, and gravel weight,
    calculate the fraction of water in the mixture. -/
theorem water_fraction_in_mixture
  (total_weight : ℝ)
  (sand_fraction : ℝ)
  (gravel_weight : ℝ)
  (h1 : total_weight = 48)
  (h2 : sand_fraction = 1/3)
  (h3 : gravel_weight = 8) :
  (total_weight - (sand_fraction * total_weight + gravel_weight)) / total_weight = 1/2 := by
  sorry

#check water_fraction_in_mixture

end NUMINAMATH_CALUDE_water_fraction_in_mixture_l3259_325947


namespace NUMINAMATH_CALUDE_rhizobia_cultivation_comparison_l3259_325904

structure Rhizobia where
  nitrogen_fixing : Bool
  aerobic : Bool

structure CultureBox where
  sterile : Bool
  gas_introduced : String

structure CultivationResult where
  nitrogen_fixation : ℕ
  colony_size : ℕ

def cultivate (box : CultureBox) (bacteria : Rhizobia) : CultivationResult :=
  sorry

theorem rhizobia_cultivation_comparison 
  (box : CultureBox) 
  (rhizobia : Rhizobia) 
  (h1 : box.sterile = true) 
  (h2 : rhizobia.nitrogen_fixing = true) 
  (h3 : rhizobia.aerobic = true) :
  let n2_result := cultivate { box with gas_introduced := "N₂" } rhizobia
  let air_result := cultivate { box with gas_introduced := "sterile air" } rhizobia
  n2_result.nitrogen_fixation < air_result.nitrogen_fixation ∧ 
  n2_result.colony_size < air_result.colony_size :=
sorry

end NUMINAMATH_CALUDE_rhizobia_cultivation_comparison_l3259_325904


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3259_325942

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 + (1 - a) * x - 1 > 0

-- Define the solution set for a = 2
def solution_set_a2 : Set ℝ := {x | x < -1/2 ∨ x > 1}

-- Define the solution set for a > -1
def solution_set_a_gt_neg1 (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > 1}
  else if a > 0 then
    {x | x < -1/a ∨ x > 1}
  else
    {x | 1 < x ∧ x < -1/a}

theorem inequality_solution_sets :
  (∀ x, x ∈ solution_set_a2 ↔ inequality 2 x) ∧
  (∀ a, a > -1 → ∀ x, x ∈ solution_set_a_gt_neg1 a ↔ inequality a x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3259_325942


namespace NUMINAMATH_CALUDE_ronald_store_visits_l3259_325933

def store_visits (bananas_per_visit : ℕ) (total_bananas : ℕ) : ℕ :=
  total_bananas / bananas_per_visit

theorem ronald_store_visits :
  let bananas_per_visit := 10
  let total_bananas := 20
  store_visits bananas_per_visit total_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_ronald_store_visits_l3259_325933


namespace NUMINAMATH_CALUDE_problem_solution_l3259_325935

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : 
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y)^2 = 16) ∧ (x^2 + y^2 = 26) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3259_325935


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l3259_325931

theorem probability_at_least_one_female (total : ℕ) (male : ℕ) (female : ℕ) (selected : ℕ) :
  total = male + female →
  selected = 3 →
  male = 6 →
  female = 4 →
  (1 - (Nat.choose male selected / Nat.choose total selected : ℚ)) = 5/6 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l3259_325931


namespace NUMINAMATH_CALUDE_renata_lottery_winnings_l3259_325957

/-- Represents the financial transactions of Renata --/
structure RenataMoney where
  initial : ℕ
  donation : ℕ
  charityWin : ℕ
  waterCost : ℕ
  lotteryCost : ℕ
  final : ℕ

/-- Calculates the lottery winnings based on Renata's transactions --/
def lotteryWinnings (r : RenataMoney) : ℕ :=
  r.final + r.donation + r.waterCost + r.lotteryCost - r.initial - r.charityWin

/-- Theorem stating that Renata's lottery winnings were $2 --/
theorem renata_lottery_winnings :
  let r : RenataMoney := {
    initial := 10,
    donation := 4,
    charityWin := 90,
    waterCost := 1,
    lotteryCost := 1,
    final := 94
  }
  lotteryWinnings r = 2 := by sorry

end NUMINAMATH_CALUDE_renata_lottery_winnings_l3259_325957


namespace NUMINAMATH_CALUDE_alpha_value_l3259_325974

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (2 * Real.pi)) 
  (h2 : ∃ (x y : Real), x = Real.sin (Real.pi / 6) ∧ y = Real.cos (5 * Real.pi / 6) ∧ 
    x = Real.sin α ∧ y = Real.cos α) : 
  α = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3259_325974


namespace NUMINAMATH_CALUDE_basketball_club_boys_l3259_325968

theorem basketball_club_boys (total : ℕ) (attendance : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 30 →
  attendance = 18 →
  total = boys + girls →
  attendance = boys + (girls / 3) →
  boys = 12 := by
sorry

end NUMINAMATH_CALUDE_basketball_club_boys_l3259_325968


namespace NUMINAMATH_CALUDE_third_beats_seventh_l3259_325984

/-- Represents a chess tournament with 8 players -/
structure ChessTournament where
  /-- List of player scores in descending order -/
  scores : List ℕ
  /-- Ensure there are exactly 8 scores -/
  score_count : scores.length = 8
  /-- Ensure all scores are different -/
  distinct_scores : scores.Nodup
  /-- Second place score equals sum of last four scores -/
  second_place_condition : scores[1]! = scores[4]! + scores[5]! + scores[6]! + scores[7]!

/-- Represents the result of a game between two players -/
inductive GameResult
  | Win
  | Loss

/-- Function to determine the game result between two players based on their positions -/
def gameResult (t : ChessTournament) (player1 : Fin 8) (player2 : Fin 8) : GameResult :=
  if player1 < player2 then GameResult.Win else GameResult.Loss

theorem third_beats_seventh (t : ChessTournament) :
  gameResult t 2 6 = GameResult.Win :=
sorry

end NUMINAMATH_CALUDE_third_beats_seventh_l3259_325984


namespace NUMINAMATH_CALUDE_platform_length_l3259_325996

/-- Given a train of length 1200 m that crosses a tree in 120 sec and passes a platform in 230 sec,
    the length of the platform is 1100 m. -/
theorem platform_length (train_length : ℝ) (tree_crossing_time : ℝ) (platform_passing_time : ℝ) :
  train_length = 1200 →
  tree_crossing_time = 120 →
  platform_passing_time = 230 →
  let train_speed := train_length / tree_crossing_time
  let platform_length := train_speed * platform_passing_time - train_length
  platform_length = 1100 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3259_325996


namespace NUMINAMATH_CALUDE_bakers_friend_cakes_l3259_325991

/-- Given that Baker made 155 cakes initially and now has 15 cakes remaining,
    prove that Baker's friend bought 140 cakes. -/
theorem bakers_friend_cakes :
  let initial_cakes : ℕ := 155
  let remaining_cakes : ℕ := 15
  let friend_bought : ℕ := initial_cakes - remaining_cakes
  friend_bought = 140 := by sorry

end NUMINAMATH_CALUDE_bakers_friend_cakes_l3259_325991


namespace NUMINAMATH_CALUDE_cubic_relation_l3259_325905

theorem cubic_relation (x A B : ℝ) (h1 : x^3 + 1/x^3 = A) (h2 : x - 1/x = B) :
  A / B = B^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_relation_l3259_325905


namespace NUMINAMATH_CALUDE_ball_probabilities_l3259_325911

/-- The total number of balls in the box -/
def total_balls : ℕ := 12

/-- The number of red balls in the box -/
def red_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 4

/-- The number of white balls in the box -/
def white_balls : ℕ := 2

/-- The number of green balls in the box -/
def green_balls : ℕ := 1

/-- The probability of drawing a red or black ball -/
def prob_red_or_black : ℚ := (red_balls + black_balls) / total_balls

/-- The probability of drawing a red, black, or white ball -/
def prob_red_black_or_white : ℚ := (red_balls + black_balls + white_balls) / total_balls

theorem ball_probabilities :
  prob_red_or_black = 3/4 ∧ prob_red_black_or_white = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3259_325911


namespace NUMINAMATH_CALUDE_price_per_ring_is_correct_l3259_325910

/-- Calculates the price per pineapple ring given the following conditions:
  * Number of pineapples bought
  * Cost per pineapple
  * Number of rings per pineapple
  * Number of rings sold together
  * Total profit
-/
def price_per_ring (num_pineapples : ℕ) (cost_per_pineapple : ℚ) 
                   (rings_per_pineapple : ℕ) (rings_per_set : ℕ) 
                   (total_profit : ℚ) : ℚ :=
  let total_cost := num_pineapples * cost_per_pineapple
  let total_rings := num_pineapples * rings_per_pineapple
  let total_revenue := total_cost + total_profit
  let num_sets := total_rings / rings_per_set
  let price_per_set := total_revenue / num_sets
  price_per_set / rings_per_set

/-- Theorem stating that the price per pineapple ring is $1.25 under the given conditions -/
theorem price_per_ring_is_correct : 
  price_per_ring 6 3 12 4 72 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_price_per_ring_is_correct_l3259_325910


namespace NUMINAMATH_CALUDE_boat_breadth_l3259_325944

theorem boat_breadth (length : Real) (sink_depth : Real) (man_mass : Real) 
  (g : Real) (water_density : Real) : Real :=
by
  -- Define the given constants
  have h1 : length = 8 := by sorry
  have h2 : sink_depth = 0.01 := by sorry
  have h3 : man_mass = 160 := by sorry
  have h4 : g = 9.81 := by sorry
  have h5 : water_density = 1000 := by sorry

  -- Calculate the breadth
  let weight := man_mass * g
  let volume := weight / (water_density * g)
  let breadth := volume / (length * sink_depth)

  -- Prove that the breadth is equal to 2
  have h6 : breadth = 2 := by sorry

  exact breadth

/- Theorem statement: The breadth of a boat with length 8 m that sinks by 1 cm 
   when a 160 kg man gets on it is 2 m, given that the acceleration due to 
   gravity is 9.81 m/s² and the density of water is 1000 kg/m³. -/

end NUMINAMATH_CALUDE_boat_breadth_l3259_325944


namespace NUMINAMATH_CALUDE_exam_probabilities_l3259_325995

/-- Represents the probabilities of scoring in different ranges in a math exam --/
structure ExamProbabilities where
  above90 : ℝ
  between80and89 : ℝ
  between70and79 : ℝ
  between60and69 : ℝ

/-- Calculates the probability of scoring 80 or above --/
def prob_80_or_above (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89

/-- Calculates the probability of failing the exam (scoring below 60) --/
def prob_fail (p : ExamProbabilities) : ℝ :=
  1 - (p.above90 + p.between80and89 + p.between70and79 + p.between60and69)

/-- Theorem stating the probabilities of scoring 80 or above and failing the exam --/
theorem exam_probabilities 
  (p : ExamProbabilities) 
  (h1 : p.above90 = 0.18) 
  (h2 : p.between80and89 = 0.51) 
  (h3 : p.between70and79 = 0.15) 
  (h4 : p.between60and69 = 0.09) : 
  prob_80_or_above p = 0.69 ∧ prob_fail p = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_exam_probabilities_l3259_325995


namespace NUMINAMATH_CALUDE_solve_mushroom_problem_l3259_325909

def mushroom_pieces_problem (total_mushrooms : ℕ) 
                            (kenny_pieces : ℕ) 
                            (karla_pieces : ℕ) 
                            (remaining_pieces : ℕ) : Prop :=
  let total_pieces := kenny_pieces + karla_pieces + remaining_pieces
  total_pieces / total_mushrooms = 4

theorem solve_mushroom_problem : 
  mushroom_pieces_problem 22 38 42 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_mushroom_problem_l3259_325909


namespace NUMINAMATH_CALUDE_calculate_expression_l3259_325954

theorem calculate_expression : (30 / (10 - 2 * 3))^2 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3259_325954


namespace NUMINAMATH_CALUDE_train_passing_bridge_l3259_325999

/-- Time for a train to pass a bridge -/
theorem train_passing_bridge 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 500)
  (h2 : train_speed_kmh = 72)
  (h3 : bridge_length = 200) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_bridge_l3259_325999


namespace NUMINAMATH_CALUDE_u_equals_fib_l3259_325955

/-- Array I as defined in the problem -/
def array_I (n : ℕ) : Fin n → Fin 3 → ℕ :=
  λ i j => match j with
    | 0 => i + 1
    | 1 => i + 2
    | 2 => i + 3

/-- Number of SDRs for array I -/
def u (n : ℕ) : ℕ := sorry

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem stating that u_n is equal to the (n+1)th Fibonacci number for n ≥ 2 -/
theorem u_equals_fib (n : ℕ) (h : n ≥ 2) : u n = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_u_equals_fib_l3259_325955


namespace NUMINAMATH_CALUDE_triangle_angles_l3259_325906

theorem triangle_angles (x y z : ℝ) : 
  (y + 150 + 160 = 360) →
  (z + 150 + 160 = 360) →
  (x + y + z = 180) →
  (x = 80 ∧ y = 50 ∧ z = 50) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l3259_325906


namespace NUMINAMATH_CALUDE_f_sum_equals_six_l3259_325973

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 9
  else 4^(-x) + 3/2

-- Theorem statement
theorem f_sum_equals_six :
  f 27 + f (-Real.log 3 / Real.log 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_six_l3259_325973


namespace NUMINAMATH_CALUDE_amy_and_noah_total_l3259_325992

/-- The number of books each person has -/
structure BookCounts where
  maddie : ℕ
  luisa : ℕ
  amy : ℕ
  noah : ℕ

/-- The conditions of the problem -/
def book_problem (bc : BookCounts) : Prop :=
  bc.maddie = 2^4 - 1 ∧
  bc.luisa = 18 ∧
  bc.amy + bc.luisa = bc.maddie + 9 ∧
  bc.noah = Int.sqrt (bc.amy^2) + 2 ∧
  (Int.sqrt (bc.amy^2))^2 = bc.amy^2

/-- The theorem to prove -/
theorem amy_and_noah_total (bc : BookCounts) :
  book_problem bc → bc.amy + bc.noah = 14 := by
  sorry

end NUMINAMATH_CALUDE_amy_and_noah_total_l3259_325992


namespace NUMINAMATH_CALUDE_solution_set_equality_l3259_325976

-- Define the inequality function
def f (x : ℝ) := x^2 + 2*x - 3

-- State the theorem
theorem solution_set_equality :
  {x : ℝ | f x < 0} = Set.Ioo (-3 : ℝ) (1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3259_325976


namespace NUMINAMATH_CALUDE_area_of_right_triangle_PQR_l3259_325967

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTrianglePQR where
  -- P, Q, R are points in ℝ²
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- PQR is a right triangle with right angle at R
  is_right_triangle : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  -- Length of hypotenuse PQ is 50
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  -- Median through P lies along y = x + 2
  median_P : ∃ t : ℝ, (P.1 + R.1) / 2 = t ∧ (P.2 + R.2) / 2 = t + 2
  -- Median through Q lies along y = 2x + 3
  median_Q : ∃ t : ℝ, (Q.1 + R.1) / 2 = t ∧ (Q.2 + R.2) / 2 = 2*t + 3

/-- The area of the right triangle PQR is 500/3 -/
theorem area_of_right_triangle_PQR (t : RightTrianglePQR) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 500 / 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_PQR_l3259_325967


namespace NUMINAMATH_CALUDE_odd_decreasing_function_range_l3259_325940

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- State the theorem
theorem odd_decreasing_function_range (a : ℝ) 
  (h_odd : is_odd f) 
  (h_decreasing : is_decreasing f) 
  (h_condition : f (2 - a) + f (4 - a) < 0) : 
  a < 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_decreasing_function_range_l3259_325940


namespace NUMINAMATH_CALUDE_exists_n_not_perfect_square_l3259_325989

theorem exists_n_not_perfect_square : ∃ n : ℕ, n > 1 ∧ ¬ ∃ m : ℕ, 2^(2^n - 1) - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_not_perfect_square_l3259_325989


namespace NUMINAMATH_CALUDE_apartment_exchange_in_two_days_l3259_325936

universe u

theorem apartment_exchange_in_two_days {α : Type u} [Finite α] :
  ∀ (f : α → α), Function.Bijective f →
  ∃ (g h : α → α), Function.Involutive g ∧ Function.Involutive h ∧ f = g ∘ h :=
by sorry

end NUMINAMATH_CALUDE_apartment_exchange_in_two_days_l3259_325936


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_three_l3259_325934

theorem sqrt_equality_implies_one_three :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (1 + Real.sqrt (27 + 18 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_three_l3259_325934


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3259_325998

/-- An isosceles triangle with perimeter 18 and one side 4 has base length 7 -/
theorem isosceles_triangle_base_length : 
  ∀ (a b c : ℝ), 
    a + b + c = 18 →  -- perimeter is 18
    (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
    (a = 4 ∨ b = 4 ∨ c = 4) →  -- one side is 4
    (a + b > c ∧ b + c > a ∧ a + c > b) →  -- triangle inequality
    (if a = b then c else if b = c then a else b) = 7 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3259_325998


namespace NUMINAMATH_CALUDE_right_triangle_sin_y_l3259_325924

theorem right_triangle_sin_y (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 20) (h_b : b = 21) :
  let sin_y := a / c
  sin_y = 20 / 29 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_y_l3259_325924


namespace NUMINAMATH_CALUDE_f_increasing_interval_l3259_325903

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 9) / Real.log (1/3)

theorem f_increasing_interval :
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < -3 → f x₁ < f x₂ :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_interval_l3259_325903


namespace NUMINAMATH_CALUDE_max_value_of_f_l3259_325953

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3259_325953


namespace NUMINAMATH_CALUDE_polynomial_value_l3259_325902

theorem polynomial_value (a : ℝ) (h : a^3 - a = 4) : (-a)^3 - (-a) - 5 = -9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3259_325902


namespace NUMINAMATH_CALUDE_digit_1983_is_7_l3259_325919

/-- Represents the decimal formed by concatenating numbers from 1 to 999 -/
def x : ℚ := sorry

/-- Returns the nth digit after the decimal point in x -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The 1983rd digit after the decimal point in x is 7 -/
theorem digit_1983_is_7 : nthDigit 1983 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_1983_is_7_l3259_325919


namespace NUMINAMATH_CALUDE_problem_solution_l3259_325990

theorem problem_solution (a b c d e : ℝ) 
  (h1 : |2 + a| + |b - 3| = 0)
  (h2 : c ≠ 0)
  (h3 : 1 / c = -d)
  (h4 : e = -5) :
  -a^b + 1/c - e + d = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3259_325990


namespace NUMINAMATH_CALUDE_shaded_area_is_925_l3259_325969

-- Define the vertices of the square
def square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the shaded polygon
def shaded_vertices : List (ℝ × ℝ) := [(0, 0), (15, 0), (40, 30), (30, 40), (0, 20)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the shaded region is 925 square units
theorem shaded_area_is_925 :
  polygon_area shaded_vertices = 925 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_925_l3259_325969


namespace NUMINAMATH_CALUDE_jerry_first_table_trays_l3259_325966

/-- The number of trays Jerry can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Jerry made -/
def number_of_trips : ℕ := 2

/-- The number of trays Jerry picked up from the second table -/
def trays_from_second_table : ℕ := 7

/-- The number of trays Jerry picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * number_of_trips - trays_from_second_table

theorem jerry_first_table_trays :
  trays_from_first_table = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_first_table_trays_l3259_325966


namespace NUMINAMATH_CALUDE_correct_number_proof_l3259_325958

theorem correct_number_proof (n : ℕ) (initial_avg correct_avg wrong_num : ℝ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_avg = 16 →
  ∃ (correct_num : ℝ),
    correct_num = n * correct_avg - (n * initial_avg - wrong_num) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_proof_l3259_325958


namespace NUMINAMATH_CALUDE_solution_set_all_reals_solution_set_interval_l3259_325965

-- Part 1
theorem solution_set_all_reals (x : ℝ) : 8 * x - 1 ≤ 16 * x^2 := by sorry

-- Part 2
theorem solution_set_interval (a x : ℝ) (h : a < 0) :
  x^2 - 2*a*x - 3*a^2 < 0 ↔ 3*a < x ∧ x < -a := by sorry

end NUMINAMATH_CALUDE_solution_set_all_reals_solution_set_interval_l3259_325965


namespace NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l3259_325951

/-- An n-gon inscribed in a circle -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Rotation of a point around the origin by an angle -/
def rotate (p : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

/-- A polygon is irregular if not all sides have the same length -/
def irregular (P : Polygon n) : Prop := sorry

/-- A polygon coincides with itself after rotation -/
def coincides_after_rotation (P : Polygon n) (angle : ℝ) : Prop := sorry

/-- A number is composite if it's not prime and greater than 1 -/
def composite (n : ℕ) : Prop := ¬ Nat.Prime n ∧ n > 1

/-- Main theorem -/
theorem irregular_polygon_rotation_implies_composite
  (n : ℕ) (P : Polygon n) (α : ℝ) :
  irregular P →
  α ≠ 2 * Real.pi →
  coincides_after_rotation P α →
  composite n := by sorry

end NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l3259_325951


namespace NUMINAMATH_CALUDE_water_filling_canal_is_certain_l3259_325922

-- Define the type for events
inductive Event : Type
  | WaitingForRabbit : Event
  | ScoopingMoon : Event
  | WaterFillingCanal : Event
  | SeekingFishByTree : Event

-- Define what it means for an event to be certain
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.WaterFillingCanal => true
  | _ => false

-- State the theorem
theorem water_filling_canal_is_certain :
  isCertainEvent Event.WaterFillingCanal :=
sorry

end NUMINAMATH_CALUDE_water_filling_canal_is_certain_l3259_325922


namespace NUMINAMATH_CALUDE_prime_diff_cubes_sum_squares_l3259_325916

theorem prime_diff_cubes_sum_squares (p : ℕ) (a b : ℕ) :
  Prime p → p = a^3 - b^3 → ∃ (c d : ℤ), p = c^2 + 3 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_diff_cubes_sum_squares_l3259_325916


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3259_325980

/-- The minimum distance between two points on different curves -/
theorem min_distance_between_curves (m : ℝ) : 
  let A := {x : ℝ | ∃ y, y = m ∧ y = 2 * (x + 1)}
  let B := {x : ℝ | ∃ y, y = m ∧ y = x + Real.log x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ B ∧ 
    (∀ (a b : ℝ), a ∈ A → b ∈ B → |x₂ - x₁| ≤ |b - a|) ∧
    |x₂ - x₁| = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3259_325980


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3259_325975

theorem largest_solution_of_equation (x : ℚ) :
  (8 * (9 * x^2 + 10 * x + 15) = x * (9 * x - 45)) →
  x ≤ -5/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3259_325975


namespace NUMINAMATH_CALUDE_laptop_price_l3259_325929

theorem laptop_price (deposit : ℝ) (deposit_percentage : ℝ) (full_price : ℝ) 
  (h1 : deposit = 400)
  (h2 : deposit_percentage = 25)
  (h3 : deposit = (deposit_percentage / 100) * full_price) :
  full_price = 1600 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l3259_325929


namespace NUMINAMATH_CALUDE_simplify_fraction_l3259_325946

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (10 * a^2 * b) / (5 * a * b) = 2 * a :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3259_325946


namespace NUMINAMATH_CALUDE_factorization_equality_l3259_325927

theorem factorization_equality (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3259_325927


namespace NUMINAMATH_CALUDE_lemonade_water_calculation_l3259_325943

/-- The amount of water needed to make lemonade with a given ratio and total volume -/
def water_needed (water_ratio : ℚ) (juice_ratio : ℚ) (total_gallons : ℚ) (liters_per_gallon : ℚ) : ℚ :=
  (water_ratio / (water_ratio + juice_ratio)) * (total_gallons * liters_per_gallon)

/-- Theorem stating the amount of water needed for the lemonade recipe -/
theorem lemonade_water_calculation :
  let water_ratio : ℚ := 8
  let juice_ratio : ℚ := 2
  let total_gallons : ℚ := 2
  let liters_per_gallon : ℚ := 3785/1000
  water_needed water_ratio juice_ratio total_gallons liters_per_gallon = 6056/1000 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_calculation_l3259_325943


namespace NUMINAMATH_CALUDE_unique_bijective_function_satisfying_equation_l3259_325994

-- Define the property that a function f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = f x + y

-- State the theorem
theorem unique_bijective_function_satisfying_equation :
  ∃! f : ℝ → ℝ, Function.Bijective f ∧ SatisfiesEquation f ∧ (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_bijective_function_satisfying_equation_l3259_325994


namespace NUMINAMATH_CALUDE_tissue_paper_usage_l3259_325930

theorem tissue_paper_usage (initial : ℕ) (remaining : ℕ) (used : ℕ) : 
  initial = 97 → remaining = 93 → used = initial - remaining → used = 4 := by
  sorry

end NUMINAMATH_CALUDE_tissue_paper_usage_l3259_325930


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_not_three_l3259_325956

theorem negation_of_absolute_value_not_three :
  (¬ ∀ x : ℤ, abs x ≠ 3) ↔ (∃ x : ℤ, abs x = 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_not_three_l3259_325956


namespace NUMINAMATH_CALUDE_sequence_value_l3259_325978

theorem sequence_value (a : ℕ → ℕ) :
  a 1 = 0 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l3259_325978


namespace NUMINAMATH_CALUDE_max_availability_equal_all_days_l3259_325937

-- Define the days of the week
inductive Day
  | Mon
  | Tues
  | Wed
  | Thurs
  | Fri

-- Define the team members
inductive Member
  | Alice
  | Bob
  | Cindy
  | David

-- Define the availability function
def availability (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Mon => false
  | Member.Alice, Day.Tues => false
  | Member.Alice, Day.Wed => true
  | Member.Alice, Day.Thurs => true
  | Member.Alice, Day.Fri => false
  | Member.Bob, Day.Mon => true
  | Member.Bob, Day.Tues => false
  | Member.Bob, Day.Wed => false
  | Member.Bob, Day.Thurs => true
  | Member.Bob, Day.Fri => true
  | Member.Cindy, Day.Mon => false
  | Member.Cindy, Day.Tues => true
  | Member.Cindy, Day.Wed => false
  | Member.Cindy, Day.Thurs => false
  | Member.Cindy, Day.Fri => true
  | Member.David, Day.Mon => true
  | Member.David, Day.Tues => true
  | Member.David, Day.Wed => true
  | Member.David, Day.Thurs => false
  | Member.David, Day.Fri => false

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => availability m d) [Member.Alice, Member.Bob, Member.Cindy, Member.David]).length

-- Theorem: The maximum number of available members is equal for all days
theorem max_availability_equal_all_days :
  (List.map availableCount [Day.Mon, Day.Tues, Day.Wed, Day.Thurs, Day.Fri]).all (· = 2) := by
  sorry


end NUMINAMATH_CALUDE_max_availability_equal_all_days_l3259_325937


namespace NUMINAMATH_CALUDE_stock_price_change_l3259_325915

theorem stock_price_change (x : ℝ) : 
  (1 - x / 100) * 1.10 = 1.012 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l3259_325915


namespace NUMINAMATH_CALUDE_squirrel_acorn_division_l3259_325928

theorem squirrel_acorn_division (total_acorns : ℕ) (acorns_per_month : ℕ) (spring_acorns : ℕ) : 
  total_acorns = 210 → acorns_per_month = 60 → spring_acorns = 30 →
  (total_acorns - 3 * acorns_per_month) / (total_acorns / 3 - acorns_per_month) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorn_division_l3259_325928


namespace NUMINAMATH_CALUDE_diana_remaining_paint_l3259_325952

/-- The amount of paint required for one statue in gallons -/
def paint_per_statue : ℚ := 1/16

/-- The number of statues Diana can paint with the remaining paint -/
def statues_to_paint : ℕ := 7

/-- The amount of paint Diana has remaining in gallons -/
def remaining_paint : ℚ := paint_per_statue * statues_to_paint

theorem diana_remaining_paint :
  remaining_paint = 7/16 := by sorry

end NUMINAMATH_CALUDE_diana_remaining_paint_l3259_325952


namespace NUMINAMATH_CALUDE_interest_problem_l3259_325997

/-- Given a sum of money put at simple interest for 3 years, if increasing the
    interest rate by 2% results in Rs. 360 more interest, then the sum is Rs. 6000. -/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 2) * 3 / 100 - P * R * 3 / 100 = 360) → P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l3259_325997


namespace NUMINAMATH_CALUDE_siblings_average_age_l3259_325913

/-- Given 4 siblings where the youngest is 25.75 years old and the others are 3, 6, and 7 years older,
    the average age of all siblings is 29.75 years. -/
theorem siblings_average_age :
  let youngest_age : ℝ := 25.75
  let sibling_age_differences : List ℝ := [3, 6, 7]
  let all_ages : List ℝ := youngest_age :: (sibling_age_differences.map (λ x => youngest_age + x))
  (all_ages.sum / all_ages.length : ℝ) = 29.75 := by
  sorry

end NUMINAMATH_CALUDE_siblings_average_age_l3259_325913


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l3259_325993

def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2
def total_students : ℕ := number_of_boys + number_of_girls
def students_selected : ℕ := 2

theorem probability_at_least_one_girl :
  (Nat.choose total_students students_selected - Nat.choose number_of_boys students_selected) /
  Nat.choose total_students students_selected = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l3259_325993


namespace NUMINAMATH_CALUDE_slope_angle_is_150_degrees_l3259_325959

/-- A line passing through (2,0) intersecting y = √(2-x²) -/
structure IntersectingLine where
  slope : ℝ
  intersectsCircle : ∃ (a b : ℝ × ℝ), a.2 = Real.sqrt (2 - a.1^2) ∧ 
                                      b.2 = Real.sqrt (2 - b.1^2) ∧
                                      a.2 = slope * (a.1 - 2) ∧
                                      b.2 = slope * (b.1 - 2)

/-- The area of triangle AOB where A and B are intersection points -/
def triangleArea (l : IntersectingLine) : ℝ :=
  sorry

theorem slope_angle_is_150_degrees (l : IntersectingLine) 
  (h : triangleArea l = 1) : 
  Real.arctan l.slope = -5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_is_150_degrees_l3259_325959


namespace NUMINAMATH_CALUDE_number_exceeding_twenty_percent_l3259_325988

theorem number_exceeding_twenty_percent : ∃ x : ℝ, x = 0.20 * x + 40 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_twenty_percent_l3259_325988


namespace NUMINAMATH_CALUDE_men_science_majors_percentage_l3259_325907

/-- Represents the composition of a college class -/
structure ClassComposition where
  total_students : ℕ
  women_science_majors : ℕ
  non_science_majors : ℕ
  men : ℕ

/-- Calculates the percentage of men who are science majors -/
def percentage_men_science_majors (c : ClassComposition) : ℚ :=
  let total_science_majors := c.total_students - c.non_science_majors
  let men_science_majors := total_science_majors - c.women_science_majors
  (men_science_majors : ℚ) / (c.men : ℚ) * 100

/-- Theorem stating the percentage of men who are science majors -/
theorem men_science_majors_percentage (c : ClassComposition) 
  (h1 : c.women_science_majors = c.total_students * 30 / 100)
  (h2 : c.non_science_majors = c.total_students * 60 / 100)
  (h3 : c.men = c.total_students * 40 / 100) :
  percentage_men_science_majors c = 25 := by
  sorry

end NUMINAMATH_CALUDE_men_science_majors_percentage_l3259_325907


namespace NUMINAMATH_CALUDE_simplify_radical_product_l3259_325925

theorem simplify_radical_product (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 84 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l3259_325925


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_plus_five_l3259_325963

theorem sum_of_three_numbers_plus_five (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 48) 
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_plus_five_l3259_325963


namespace NUMINAMATH_CALUDE_kyler_wins_one_game_l3259_325914

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kyler : Player

/-- Represents the outcome of a chess game -/
inductive Outcome : Type
| Win : Outcome
| Loss : Outcome
| Draw : Outcome

/-- The number of games each player played -/
def games_per_player : ℕ := 6

/-- The total number of game outcomes recorded -/
def total_outcomes : ℕ := 18

/-- Function to get the number of wins for a player -/
def wins (p : Player) : ℕ :=
  match p with
  | Player.Peter => 3
  | Player.Emma => 2
  | Player.Kyler => 0  -- We'll prove this is actually 1

/-- Function to get the number of losses for a player -/
def losses (p : Player) : ℕ :=
  match p with
  | Player.Peter => 2
  | Player.Emma => 2
  | Player.Kyler => 3

/-- Function to get the number of draws for a player -/
def draws (p : Player) : ℕ :=
  match p with
  | Player.Peter => 1
  | Player.Emma => 2
  | Player.Kyler => 2

theorem kyler_wins_one_game :
  wins Player.Kyler = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_kyler_wins_one_game_l3259_325914


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3259_325964

theorem remainder_of_large_number : 2345678901 % 101 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3259_325964


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l3259_325962

def p (x : ℝ) : ℝ := x^2 - 3*x + 2

def q (x : ℝ) : ℝ := -x^2

def eval_points : List ℝ := [0, 1, 2, 3, 4]

theorem sum_of_composite_function :
  (eval_points.map (λ x => q (p x))).sum = -12 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l3259_325962


namespace NUMINAMATH_CALUDE_sequence_property_l3259_325921

def is_increasing_positive_integer_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : is_increasing_positive_integer_sequence a) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + 2 * a n) 
  (h3 : a 5 = 52) : 
  a 7 = 212 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3259_325921


namespace NUMINAMATH_CALUDE_prob_different_colors_l3259_325981

/-- Probability of drawing two different colored chips -/
theorem prob_different_colors (blue yellow red : ℕ) 
  (h_blue : blue = 6)
  (h_yellow : yellow = 4)
  (h_red : red = 2) :
  let total := blue + yellow + red
  (blue * yellow + blue * red + yellow * red) * 2 / (total * total) = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_l3259_325981


namespace NUMINAMATH_CALUDE_min_b_value_l3259_325949

/-- Given positive integers x, y, z in ratio 3:4:7 and y = 15b - 5, 
    prove the minimum positive integer b is 3 -/
theorem min_b_value (x y z b : ℕ+) : 
  (∃ k : ℕ+, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * b - 5 →
  (∀ b' : ℕ+, b' < b → 
    ¬∃ x' y' z' : ℕ+, (∃ k : ℕ+, x' = 3 * k ∧ y' = 4 * k ∧ z' = 7 * k) ∧ 
    y' = 15 * b' - 5) →
  b = 3 := by
  sorry

#check min_b_value

end NUMINAMATH_CALUDE_min_b_value_l3259_325949


namespace NUMINAMATH_CALUDE_range_of_a_l3259_325977

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The set A = {x | f(x) ≤ 0} -/
def A (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- The set B = {x | f(f(x)) ≤ 5/4} -/
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) ≤ 5/4}

/-- Theorem: Given f(x) = x^2 + ax + b, A = {x | f(x) ≤ 0}, B = {x | f(f(x)) ≤ 5/4},
    and A = B ≠ ∅, the range of a is [√5, 5] -/
theorem range_of_a (a b : ℝ) : 
  A a b = B a b ∧ A a b ≠ ∅ → a ∈ Set.Icc (Real.sqrt 5) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3259_325977


namespace NUMINAMATH_CALUDE_guessing_game_factor_l3259_325932

theorem guessing_game_factor (f : ℚ) : 33 * f = 2 * 51 - 3 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_guessing_game_factor_l3259_325932


namespace NUMINAMATH_CALUDE_bird_sale_theorem_l3259_325920

/-- Represents the purchase and sale of two birds with given conditions -/
def bird_sale (purchase_price1 purchase_price2 : ℝ) : Prop :=
  ∃ (selling_price : ℝ),
    selling_price = purchase_price1 * 0.8
    ∧ selling_price = purchase_price2 * 1.2
    ∧ (selling_price - purchase_price1) + (selling_price - purchase_price2) = -10

/-- Theorem stating the conditions and conclusion of the bird sale problem -/
theorem bird_sale_theorem :
  bird_sale 30 20 → ∃ (selling_price : ℝ), selling_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_bird_sale_theorem_l3259_325920


namespace NUMINAMATH_CALUDE_problem_solution_l3259_325926

def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

theorem problem_solution :
  (∀ x : ℝ, f x 3 > g 3 + 2 ↔ (x < -4 ∨ x > 2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-a) 1 → f x a ≤ g a) ↔ a ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3259_325926


namespace NUMINAMATH_CALUDE_binary_to_decimal_l3259_325918

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 : ℕ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_l3259_325918


namespace NUMINAMATH_CALUDE_scenario_one_scenario_two_scenario_three_scenario_four_l3259_325985

-- Define the probabilities for A and B
def prob_A : ℚ := 2/3
def prob_B : ℚ := 3/4

-- Define the complementary probabilities
def miss_A : ℚ := 1 - prob_A
def miss_B : ℚ := 1 - prob_B

-- Theorem 1
theorem scenario_one : 
  1 - prob_A^3 = 19/27 := by sorry

-- Theorem 2
theorem scenario_two :
  2 * prob_A^2 * miss_A * 2 * prob_B * miss_B = 1/6 := by sorry

-- Theorem 3
theorem scenario_three :
  miss_A^2 * prob_B^2 = 1/16 := by sorry

-- Theorem 4
theorem scenario_four :
  2 * prob_A * miss_A * 2 * prob_B * miss_B = 1/6 := by sorry

end NUMINAMATH_CALUDE_scenario_one_scenario_two_scenario_three_scenario_four_l3259_325985


namespace NUMINAMATH_CALUDE_malcolm_white_lights_l3259_325945

/-- Represents the brightness levels of lights --/
inductive Brightness
  | Low
  | Medium
  | High

/-- Represents different types of lights --/
inductive LightType
  | White
  | Red
  | Yellow
  | Blue
  | Green
  | Purple

/-- Returns the brightness value of a given brightness level --/
def brightnessValue (b : Brightness) : Rat :=
  match b with
  | Brightness.Low => 1/2
  | Brightness.Medium => 1
  | Brightness.High => 3/2

/-- Calculates the total brightness of a given number of lights with a specific brightness --/
def totalBrightness (count : Nat) (b : Brightness) : Rat :=
  count * brightnessValue b

/-- Represents Malcolm's initial light purchase --/
structure InitialPurchase where
  redCount : Nat
  yellowCount : Nat
  blueCount : Nat
  greenCount : Nat
  purpleCount : Nat
  redBrightness : Brightness
  yellowBrightness : Brightness
  blueBrightness : Brightness
  greenBrightness : Brightness
  purpleBrightness : Brightness

/-- Represents the additional lights Malcolm needs to buy --/
structure AdditionalPurchase where
  additionalBluePercentage : Rat
  additionalRedCount : Nat

/-- Theorem: Given Malcolm's initial and additional light purchases, prove that he had 38 white lights initially --/
theorem malcolm_white_lights (initial : InitialPurchase) (additional : AdditionalPurchase) :
  initial.redCount = 16 ∧
  initial.yellowCount = 4 ∧
  initial.blueCount = 2 * initial.yellowCount ∧
  initial.greenCount = 8 ∧
  initial.purpleCount = 3 ∧
  initial.redBrightness = Brightness.Low ∧
  initial.yellowBrightness = Brightness.High ∧
  initial.blueBrightness = Brightness.Medium ∧
  initial.greenBrightness = Brightness.Low ∧
  initial.purpleBrightness = Brightness.High ∧
  additional.additionalBluePercentage = 1/4 ∧
  additional.additionalRedCount = 10 →
  ∃ (whiteCount : Nat), whiteCount = 38 ∧
    totalBrightness whiteCount Brightness.Medium =
      totalBrightness initial.redCount initial.redBrightness +
      totalBrightness initial.yellowCount initial.yellowBrightness +
      totalBrightness initial.blueCount initial.blueBrightness +
      totalBrightness initial.greenCount initial.greenBrightness +
      totalBrightness initial.purpleCount initial.purpleBrightness +
      totalBrightness (Nat.ceil (additional.additionalBluePercentage * initial.blueCount)) Brightness.Medium +
      totalBrightness additional.additionalRedCount Brightness.Low :=
by
  sorry

end NUMINAMATH_CALUDE_malcolm_white_lights_l3259_325945


namespace NUMINAMATH_CALUDE_symmetric_circles_and_common_chord_l3259_325917

-- Define the symmetry relation with respect to line l
def symmetric_line (x y : ℝ) : Prop := ∃ (x' y' : ℝ), x' = y + 1 ∧ y' = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y = 0

-- Define circle C'
def circle_C' (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 10

-- Theorem statement
theorem symmetric_circles_and_common_chord :
  (∀ x y : ℝ, symmetric_line x y → (circle_C x y ↔ circle_C' y x)) ∧
  (∃ a b c d : ℝ, 
    circle_C a b ∧ circle_C c d ∧ 
    circle_C' a b ∧ circle_C' c d ∧
    (a - c)^2 + (b - d)^2 = 38) :=
sorry

end NUMINAMATH_CALUDE_symmetric_circles_and_common_chord_l3259_325917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3259_325923

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmeticSequence a →
  (a 2 = 0) →
  (S 3 + S 4 = 6) →
  (a 5 + a 6 = 21) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3259_325923


namespace NUMINAMATH_CALUDE_water_level_rise_rate_l3259_325908

/-- The water level function with respect to time -/
def water_level (t : ℝ) : ℝ := 0.3 * t + 3

/-- The time domain -/
def time_domain : Set ℝ := { t | 0 ≤ t ∧ t ≤ 5 }

/-- The rate of change of the water level -/
def water_level_rate : ℝ := 0.3

theorem water_level_rise_rate :
  ∀ t ∈ time_domain, 
    (water_level (t + 1) - water_level t) = water_level_rate := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_rate_l3259_325908


namespace NUMINAMATH_CALUDE_salt_solution_volume_l3259_325972

/-- Given a salt solution with a concentration of 15 grams per 1000 cubic centimeters,
    prove that 0.375 grams of salt corresponds to 25 cubic centimeters of solution. -/
theorem salt_solution_volume (concentration : ℝ) (volume : ℝ) (salt_amount : ℝ) :
  concentration = 15 / 1000 →
  salt_amount = 0.375 →
  volume * concentration = salt_amount →
  volume = 25 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l3259_325972


namespace NUMINAMATH_CALUDE_four_m₀_is_sum_of_three_or_four_primes_l3259_325939

-- Define the existence of a prime between n and 2n for any positive integer n
axiom exists_prime_between (n : ℕ) (hn : 0 < n) : ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n

-- Define the smallest even number greater than 2 that can't be expressed as sum of two primes
axiom exists_smallest_non_goldbach : ∃ m₀ : ℕ, 1 < m₀ ∧ 
  (∀ k < m₀, ∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * k = p + q) ∧
  (¬∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * m₀ = p + q)

-- Theorem statement
theorem four_m₀_is_sum_of_three_or_four_primes :
  ∃ m₀ : ℕ, 1 < m₀ ∧ 
  (∀ k < m₀, ∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * k = p + q) ∧
  (¬∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * m₀ = p + q) →
  ∃ p₁ p₂ p₃ p₄ : ℕ, (Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 4 * m₀ = p₁ + p₂ + p₃) ∨
                     (Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 4 * m₀ = p₁ + p₂ + p₃ + p₄) :=
by sorry

end NUMINAMATH_CALUDE_four_m₀_is_sum_of_three_or_four_primes_l3259_325939


namespace NUMINAMATH_CALUDE_max_areas_formula_l3259_325961

/-- Represents a circular disk with n equally spaced radii and one off-center chord -/
structure DividedDisk where
  n : ℕ  -- number of equally spaced radii
  has_off_center_chord : Bool

/-- Calculates the maximum number of non-overlapping areas in a divided disk -/
def max_areas (d : DividedDisk) : ℕ :=
  2 * d.n + 2

/-- Theorem: The maximum number of non-overlapping areas in a divided disk is 2n + 2 -/
theorem max_areas_formula (d : DividedDisk) (h : d.has_off_center_chord = true) :
  max_areas d = 2 * d.n + 2 := by
  sorry

#check max_areas_formula

end NUMINAMATH_CALUDE_max_areas_formula_l3259_325961


namespace NUMINAMATH_CALUDE_path_length_along_squares_l3259_325987

theorem path_length_along_squares (PQ : ℝ) (h : PQ = 73) : 
  3 * PQ = 219 := by
  sorry

end NUMINAMATH_CALUDE_path_length_along_squares_l3259_325987


namespace NUMINAMATH_CALUDE_x_axis_intercept_l3259_325901

/-- The x-axis intercept of the line y = 2x + 1 is -1/2 -/
theorem x_axis_intercept (x : ℝ) : 
  (2 * x + 1 = 0) → (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_x_axis_intercept_l3259_325901


namespace NUMINAMATH_CALUDE_bakers_pastry_problem_l3259_325900

/-- Baker's pastry problem -/
theorem bakers_pastry_problem (cakes_sold : ℕ) (difference : ℕ) (pastries_sold : ℕ) :
  cakes_sold = 97 →
  cakes_sold = pastries_sold + difference →
  difference = 89 →
  pastries_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_bakers_pastry_problem_l3259_325900


namespace NUMINAMATH_CALUDE_cubic_root_polynomial_l3259_325970

theorem cubic_root_polynomial (a b c : ℝ) (P : ℝ → ℝ) : 
  (a^3 + 4*a^2 + 7*a + 10 = 0) →
  (b^3 + 4*b^2 + 7*b + 10 = 0) →
  (c^3 + 4*c^2 + 7*c + 10 = 0) →
  (P a = b + c) →
  (P b = a + c) →
  (P c = a + b) →
  (P (a + b + c) = -22) →
  (∀ x, P x = 8/9*x^3 + 44/9*x^2 + 71/9*x + 2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_polynomial_l3259_325970
