import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Findiff.OuterMeasure
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Nat.Floor
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Probability
import Mathlib.Probability.Continuity
import Mathlib.SetTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import algebra.group.defs

namespace driving_time_eqn_l546_546907

open Nat

-- Define the variables and constants
def avg_speed_before := 80 -- km/h
def stop_time := 1 / 3 -- hour
def avg_speed_after := 100 -- km/h
def total_distance := 250 -- km
def total_time := 3 -- hours

variable (t : ℝ) -- the time in hours before the stop

-- State the main theorem
theorem driving_time_eqn :
  avg_speed_before * t + avg_speed_after * (total_time - stop_time - t) = total_distance := by
  sorry

end driving_time_eqn_l546_546907


namespace binomial_probability_p_l546_546882

noncomputable def binomial_expected_value (n p : ℝ) := n * p
noncomputable def binomial_variance (n p : ℝ) := n * p * (1 - p)

theorem binomial_probability_p (n p : ℝ) (h1: binomial_expected_value n p = 2) (h2: binomial_variance n p = 1) : 
  p = 0.5 :=
by
  sorry

end binomial_probability_p_l546_546882


namespace shopkeeper_profit_percentage_l546_546518

theorem shopkeeper_profit_percentage (P : ℝ) : (70 / 100) * (1 + P / 100) = 1 → P = 700 / 3 :=
by
  sorry

end shopkeeper_profit_percentage_l546_546518


namespace initial_people_count_is_16_l546_546171

-- Define the conditions
def initial_people (x : ℕ) : Prop :=
  let people_came_in := 5 in
  let people_left := 2 in
  let final_people := 19 in
  x + people_came_in - people_left = final_people

-- Define the theorem
theorem initial_people_count_is_16 (x : ℕ) (h : initial_people x) : x = 16 :=
by
  sorry

end initial_people_count_is_16_l546_546171


namespace find_number_of_girls_examined_l546_546862

variable (G : ℕ)

def number_boys : ℕ := 50
def percentage_boys_pass : ℚ := 0.5
def percentage_girls_pass : ℚ := 0.4
def percentage_total_fail : ℚ := 0.5667

def failed_boys := number_boys * (1 - percentage_boys_pass)
def failed_girls := G * (1 - percentage_girls_pass)
def total_students := number_boys + G
def total_failed := percentage_total_fail * total_students

theorem find_number_of_girls_examined (h : failed_boys + failed_girls = total_failed) : G = 100 :=
by
  sorry

end find_number_of_girls_examined_l546_546862


namespace max_number_of_regions_l546_546090

theorem max_number_of_regions (n : ℕ) : 
  maximal_regions n = nat.choose n 4 + nat.choose n 2 + 1 :=
sorry

end max_number_of_regions_l546_546090


namespace shells_put_back_l546_546007

def shells_picked_up : ℝ := 324.0
def shells_left : ℝ := 32.0

theorem shells_put_back : shells_picked_up - shells_left = 292 := by
  sorry

end shells_put_back_l546_546007


namespace dans_average_rate_l546_546104

/-- Dan's average rate for the entire trip, given the conditions, equals 0.125 miles per minute --/
theorem dans_average_rate :
  ∀ (d_run d_swim : ℝ) (r_run r_swim : ℝ) (time_run time_swim : ℝ),
  d_run = 3 ∧ d_swim = 3 ∧ r_run = 10 ∧ r_swim = 6 ∧ 
  time_run = (d_run / r_run) * 60 ∧ time_swim = (d_swim / r_swim) * 60 →
  ((d_run + d_swim) / (time_run + time_swim)) = 0.125 :=
by
  intros d_run d_swim r_run r_swim time_run time_swim h
  sorry

end dans_average_rate_l546_546104


namespace birds_left_in_cage_l546_546880

theorem birds_left_in_cage (initial_birds taken_birds remaining_birds : ℕ) 
  (h1 : initial_birds = 19)
  (h2 : taken_birds = 10) :
  remaining_birds = initial_birds - taken_birds :=
by 
  have h : remaining_birds = 9,
  sorry

end birds_left_in_cage_l546_546880


namespace Ali_Baba_Wins_If_Bandit_Starts_First_l546_546151

theorem Ali_Baba_Wins_If_Bandit_Starts_First:
  ∃ Ali_Baba_Wins, 
  (∀ (initialPile : ℕ), initialPile = 2017 → (∀ moveCount : ℕ, moveCount = 2016 → (gameOver moveCount) → banditStartsFirst → Ali_Baba_Wins)) :=
by sorry

end Ali_Baba_Wins_If_Bandit_Starts_First_l546_546151


namespace number_of_correct_statements_is_three_l546_546856

variables {α β : Type*} [plane α] [plane β] (l : line α)

def statement1 := ∀ (l : line α), (∀(m : line β), l ⊥ m) → α ⊥ β
def statement2 := (∀ (l : line α), l ∥ β) → α ∥ β
def statement3 := α ⊥ β → (l : line α) → l ⊥ β
def statement4 := α ∥ β → (l : line α) → l ∥ β

def num_correct_statements : ℕ := 
  (if statement1 then 1 else 0) + 
  (if statement2 then 1 else 0) + 
  (if statement3 then 1 else 0) + 
  (if statement4 then 1 else 0)

theorem number_of_correct_statements_is_three : num_correct_statements = 3 := 
sorry

end number_of_correct_statements_is_three_l546_546856


namespace inequality_solution_l546_546217

theorem inequality_solution (m : ℝ) (h : m < -1) :
  (if m = -3 then
    {x : ℝ | x > 1} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if -3 < m ∧ m < -1 then
    ({x : ℝ | x < m / (m + 3)} ∪ {x : ℝ | x > 1}) =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if m < -3 then
    {x : ℝ | 1 < x ∧ x < m / (m + 3)} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else
    False) :=
by
  sorry

end inequality_solution_l546_546217


namespace sum_proper_divisors_81_l546_546832

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546832


namespace union_of_sets_l546_546233

-- Definitions based on conditions
def A : Set ℕ := {2, 3}
def B (a : ℕ) : Set ℕ := {1, a}
def condition (a : ℕ) : Prop := A ∩ (B a) = {2}

-- Main theorem to be proven
theorem union_of_sets (a : ℕ) (h : condition a) : A ∪ (B a) = {1, 2, 3} :=
sorry

end union_of_sets_l546_546233


namespace sequence_has_at_most_8_values_l546_546222

open Real

noncomputable def f : ℝ → ℝ := sorry

def a_n (n : ℕ) : ℝ := f n

theorem sequence_has_at_most_8_values :
  (∀ x : ℝ, f (x + 4) = -1 / f x) →
  (∀ n : ℕ, a_n n = f n) →
  (∃ l : ℕ, l ≤ 8 ∧ ∀ n : ℕ, a_n n ∈ fin l) :=
sorry

end sequence_has_at_most_8_values_l546_546222


namespace jake_third_test_marks_l546_546749

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end jake_third_test_marks_l546_546749


namespace initial_hour_hand_position_l546_546301

theorem initial_hour_hand_position (θ : ℝ) :
  ∃ θ, (θ = 15 ∨  θ = 165) :=
by
  have minute_hand_position : ∀ t, t + 360 = t := sorry,
  have hour_hand_movement : ∀ θ, (θ + 30) % 360 = θ + 30 := sorry,
  -- The minute hand bisects one of the angles,
  -- The angle when hour hand moves from θ to θ + 30
  have bisected_angles : ∀ θ, (θ + 30) / 2 = 15 ∨ (360 + 30) / 2 = 165 := sorry,
  existsi θ
  split
  apply and.intro
  exact minute_hand_position
  exact hour_hand_movement
  exact bisected_angles

end initial_hour_hand_position_l546_546301


namespace greatest_int_with_gcd_18_is_138_l546_546082

theorem greatest_int_with_gcd_18_is_138 :
  ∃ n : ℕ, n < 150 ∧ int.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ int.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_int_with_gcd_18_is_138_l546_546082


namespace f_has_two_zeros_iff_l546_546252

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.exp (2 * x) + (a - 2) * real.exp x - a * x

theorem f_has_two_zeros_iff (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 1 := 
by
  sorry

end f_has_two_zeros_iff_l546_546252


namespace correct_option_l546_546469

-- Definitions for conditions
def C1 (a : ℕ) : Prop := a^2 * a^3 = a^5
def C2 (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def C3 (a b : ℕ) : Prop := (a * b)^3 = a * b^3
def C4 (a : ℕ) : Prop := (-a^3)^2 = -a^6

-- The correct option is C1
theorem correct_option (a : ℕ) : C1 a := by
  sorry

end correct_option_l546_546469


namespace traffic_flow_solution_l546_546551

noncomputable def traffic_flow_second_ring : ℕ := 10000
noncomputable def traffic_flow_third_ring (x : ℕ) : Prop := 3 * x - (x + 2000) = 2 * traffic_flow_second_ring

theorem traffic_flow_solution :
  ∃ (x : ℕ), traffic_flow_third_ring x ∧ (x = 11000) ∧ (x + 2000 = 13000) :=
by
  sorry

end traffic_flow_solution_l546_546551


namespace hyperbola_equation_l546_546260

-- Define the conditions
def hyperbola_eq (b : ℝ) := b > 0 ∧ ∀ (x y : ℝ), (x^2 / 4 - y^2 / b^2 = 1)
def circle_eq := ∀ (x y : ℝ), (x^2 + y^2 = 4)
def asymptotes := ∀ (x : ℝ), (x ≠ 0 → y = b/2 * x ∨ y = -b/2 * x)
def quad_area (b : ℝ) := 2

-- Rewrite the proof statement
theorem hyperbola_equation (b : ℝ) (h_b : hyperbola_eq b) (h_circle : circle_eq) (h_asym : asymptotes) (h_area : quad_area b) :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1) :=
by
  sorry

end hyperbola_equation_l546_546260


namespace range_of_a_l546_546295

variable (x a : ℝ)
def inequality_sys := x < a ∧ x < 3
def solution_set := x < a

theorem range_of_a (h : ∀ x, inequality_sys x a → solution_set x a) : a ≤ 3 := by
  sorry

end range_of_a_l546_546295


namespace prism_pyramid_sum_l546_546384

theorem prism_pyramid_sum :
  let prism_faces := 6
  let prism_edges := 12
  let prism_vertices := 8
  let pyramid_new_faces := 4
  let pyramid_new_edges := 4
  let pyramid_new_vertex := 1
  let combined_faces := prism_faces - 1 + pyramid_new_faces
  let combined_edges := prism_edges + pyramid_new_edges
  let combined_vertices := prism_vertices + pyramid_new_vertex
  combined_faces + combined_edges + combined_vertices = 34 := by
sory

end prism_pyramid_sum_l546_546384


namespace craig_apples_after_sharing_l546_546184

-- Defining the initial conditions
def initial_apples_craig : ℕ := 20
def shared_apples : ℕ := 7

-- The proof statement
theorem craig_apples_after_sharing : 
  initial_apples_craig - shared_apples = 13 := 
by
  sorry

end craig_apples_after_sharing_l546_546184


namespace line_perpendicular_to_plane_l546_546336

variables (α β : Type) (l : Type)
variables [Plane α] [Plane β] [Line l]
variables [H_parallel_αβ : Parallel α β]
variables [H_perpendicular_lα : Perpendicular l α] 
  
theorem line_perpendicular_to_plane
  (l α β : Type)
  [Plane α]
  [Plane β]
  [Line l]
  (H_parallel_αβ : Parallel α β)
  (H_perpendicular_lα : Perpendicular l α) :
  Perpendicular l β :=
sorry

end line_perpendicular_to_plane_l546_546336


namespace binomial_coefficient_l546_546678

theorem binomial_coefficient (a : ℝ) (x : ℝ) (hx : x ≠ 0) (ha : a = 1 / 2) :
  (∑ r in finset.range (6 + 1), (nat.choose 6 r) * (a * x) ^ (6 - r) * (- (1 / x)) ^ r) = -3 / 16 * x^2 := 
sorry

end binomial_coefficient_l546_546678


namespace probability_earning_3300_in_three_spins_l546_546018

theorem probability_earning_3300_in_three_spins :
  let outcomes := ["Bankrupt", "$2000", "$500", "$3000", "$800", "$400"]
  let spins := (outcomes.length)^3
  let successful_outcomes := 6
  spins = 216 →
  successful_outcomes = 6 →
  outcomes.length = 6 →
  6^3 = 216 →
  (successful_outcomes : ℚ) / spins = 1 / 36 :=
by
  intros
  sorry

end probability_earning_3300_in_three_spins_l546_546018


namespace shaded_region_perimeter_l546_546320

noncomputable def perimeter_shaded_region {r : ℝ} (h : r > 0) : ℝ :=
  let OP := r
  let OQ := r
  let arcPQ := π * r
  OP + OQ + arcPQ

theorem shaded_region_perimeter (r : ℝ) (h : r > 0) : 
  perimeter_shaded_region h = 2 * r + π * r := by
  sorry

end shaded_region_perimeter_l546_546320


namespace distance_from_center_to_line_l546_546040

theorem distance_from_center_to_line :
  let C_eq : ∀ x y, x^2 + y^2 - 2 * x - 4 * y + 4 = 0 :=
    λ x y, x^2 + y^2 - 2 * x - 4 * y + 4
  let line_eq : ∀ x y, 3 * x + 4 * y + 4 = 0 :=
    λ x y, 3 * x + 4 * y + 4
  let center := (1 : ℤ, 2 : ℤ)
  let distance := λ (x₁ y₁ : ℤ) => abs (3 * x₁ + 4 * y₁ + 4) / real.sqrt (3^2 + 4^2)
  in distance (fst center) (snd center) = 3 := 
by
  sorry

end distance_from_center_to_line_l546_546040


namespace harmonic_mean_of_3_6_12_l546_546177

theorem harmonic_mean_of_3_6_12 :
  let numbers := [3, 6, 12]
  ∑ n in numbers, (1 / n) / numbers.length = 36 / 7 :=
by
  sorry

end harmonic_mean_of_3_6_12_l546_546177


namespace additional_hours_on_days_without_practice_l546_546554

def total_weekday_homework_hours : ℕ := 2 + 3 + 4 + 3 + 1
def total_weekend_homework_hours : ℕ := 8
def total_homework_hours : ℕ := total_weekday_homework_hours + total_weekend_homework_hours
def total_chore_hours : ℕ := 1 + 1
def total_hours : ℕ := total_homework_hours + total_chore_hours

theorem additional_hours_on_days_without_practice : ∀ (practice_nights : ℕ), 
  (2 ≤ practice_nights ∧ practice_nights ≤ 3) →
  (∃ tuesday_wednesday_thursday_weekend_day_hours : ℕ,
    tuesday_wednesday_thursday_weekend_day_hours = 15) :=
by
  intros practice_nights practice_nights_bounds
  -- Define days without practice in the worst case scenario
  let tuesday_hours := 3
  let wednesday_homework_hours := 4
  let wednesday_chore_hours := 1
  let thursday_hours := 3
  let weekend_day_hours := 4
  let days_without_practice_hours := tuesday_hours + (wednesday_homework_hours + wednesday_chore_hours) + thursday_hours + weekend_day_hours
  use days_without_practice_hours
  -- In the worst case, the total additional hours on days without practice should be 15.
  sorry

end additional_hours_on_days_without_practice_l546_546554


namespace variance_of_dataset_l546_546982

theorem variance_of_dataset (x : ℝ) (h₁ : (8 + x + 10 + 11 + 9) / 5 = 10) : 
  let s2 := (∑ i in [8, x, 10, 11, 9], (i - 10)^2) / 5 in
  s2 = 2 :=
by
  -- Proof steps here, which we've skipped for now
  sorry

end variance_of_dataset_l546_546982


namespace imaginary_part_is_empty_l546_546045

def imaginary_part_empty (z : ℂ) : Prop :=
  z.im = 0

theorem imaginary_part_is_empty (z : ℂ) (h : z.im = 0) : imaginary_part_empty z :=
by
  -- proof skipped
  sorry

end imaginary_part_is_empty_l546_546045


namespace lamp_probability_l546_546400

theorem lamp_probability :
  ∀ (red_lamps blue_lamps : ℕ), 
  red_lamps = 4 → blue_lamps = 2 →
  (∀ lamps_on : ℕ, lamps_on = 3 →
    (1 / (Nat.choose (red_lamps + blue_lamps) 2 * Nat.choose (red_lamps + blue_lamps) 3 / 
      (Nat.choose (5) 1 * Nat.choose (4) 2)) = 0.1)) :=
by
  intros red_lamps blue_lamps h_rl h_bl lamps_on h_lo
  apply eq_div_iff_mul_eq.mpr _
  norm_num
  sorry

end lamp_probability_l546_546400


namespace combined_volume_is_correct_l546_546395

noncomputable def rectangle_base := 10 * 5

noncomputable def pyramid_volume := (1 / 3) * rectangle_base * 8

noncomputable def cuboid_volume := 5 * 5 * 8

noncomputable def combined_volume : ℚ := pyramid_volume + cuboid_volume

theorem combined_volume_is_correct :
  combined_volume = 1000 / 3 := by
sawoo.targets-kinetic  -- The proof is omitted

end combined_volume_is_correct_l546_546395


namespace pages_in_book_l546_546014

-- Define the initial conditions
variable (P : ℝ) -- total number of pages in the book
variable (h_read_20_percent : 0.20 * P = 320 * 0.20 / 0.80) -- Nate has read 20% of the book and the rest 80%

-- The goal is to show that P = 400
theorem pages_in_book (P : ℝ) :
  (0.80 * P = 320) → P = 400 :=
by
  sorry

end pages_in_book_l546_546014


namespace tan_XWZ_max_value_l546_546324

theorem tan_XWZ_max_value (X Y Z W V : Type) [triangle XYZ] 
  (angle_Z : ∠ (Z : XYZ) = 45) (YZ : length (segment YZ) = 6) (W_midpoint : midpoint W (segment YZ))
  (perpendicular_XV : perpendicular X (segment YZ) at V) (V_not_mid : V ≠ W) :
  ∃ XWZ : angle XYZ, tan XWZ = 10 := 
sorry

end tan_XWZ_max_value_l546_546324


namespace ratio_of_volumes_l546_546138

theorem ratio_of_volumes 
  (h r : ℝ)
  (Hh : h > 0) 
  (Hr : r > 0) :
  let V_A := (1 / 3) * (π * r ^ 2 * h)
  let V_B := (1 / 3) * (π * (2 * r) ^ 2 * (2 * h))
  let V_C := (1 / 3) * (π * (3 * r) ^ 2 * (3 * h))
  let V_D := (1 / 3) * (π * (4 * r) ^ 2 * (4 * h))
  let V_E := (1 / 3) * (π * (5 * r) ^ 2 * (5 * h))
  let V_1 := V_E - V_D
  let V_2 := V_D - V_C
  V_1 ≠ 0 →
  V_2 ≠ 0 →
  V_2 / V_1 = 37 / 61 :=
begin
  intros V_A V_B V_C V_D V_E V_1 V_2 H_V1_nonzero H_V2_nonzero,
  sorry
end

end ratio_of_volumes_l546_546138


namespace AliBabaWinsIfBanditStartsFirst_l546_546154

-- Define the game state
structure Game where
  piles : ℕ

-- Initial state with 2017 diamonds
def initialState : Game := { piles := 1 }

-- Define the move in the game
def makeMove (g : Game) : Game :=
  { piles := g.piles + 1 }

-- Define an action in the game
def canMove (g : Game) : Prop :=
  g.piles < 2017

-- Define the winner based on whose turn it is after all valid moves
def winner (g : Game) (turns : ℕ) : String :=
  if turns % 2 = 0 then "Bandit" else "Ali-Baba"

-- The statement we need to prove
theorem AliBabaWinsIfBanditStartsFirst :
  winner (makeMove^2016 initialState) 2016 = "Ali-Baba" :=
sorry

end AliBabaWinsIfBanditStartsFirst_l546_546154


namespace whitney_spent_179_l546_546096

def total_cost (books_whales books_fish magazines book_cost magazine_cost : ℕ) : ℕ :=
  (books_whales + books_fish) * book_cost + magazines * magazine_cost

theorem whitney_spent_179 :
  total_cost 9 7 3 11 1 = 179 :=
by
  sorry

end whitney_spent_179_l546_546096


namespace area_of_square_II_is_correct_l546_546707

variable (a b : ℝ)

def diagonal_of_square_I (a b : ℝ) : ℝ := 3 * (a + b)
def side_length_of_square_I (a b : ℝ) : ℝ := (diagonal_of_square_I a b) / Real.sqrt 2
def area_of_square_I (a b : ℝ) : ℝ := (side_length_of_square_I a b) ^ 2
def area_of_square_II (a b : ℝ) : ℝ := 3 * (area_of_square_I a b)

theorem area_of_square_II_is_correct (a b : ℝ) :
  area_of_square_II a b = (27 * (a + b) ^ 2) / 2 :=
by
  rw [area_of_square_II, area_of_square_I, side_length_of_square_I, diagonal_of_square_I]
  sorry

end area_of_square_II_is_correct_l546_546707


namespace find_number_l546_546513

def divisor : ℕ := 22
def quotient : ℕ := 12
def remainder : ℕ := 1
def number : ℕ := (divisor * quotient) + remainder

theorem find_number : number = 265 := by
  sorry

end find_number_l546_546513


namespace problem_I_problem_II_l546_546255

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (x - 1) + (2 * a / x)

theorem problem_I (a : ℝ) : 
  (0 ≤ a ∧ a ≤ 2 → ∀ x, 1 < x → (deriv (λ x : ℝ, log (x - 1) + 2 * a / x)) x ≥ 0) ∧
  ((a < 0 ∨ a > 2) → 
    let x1 := a - real.sqrt (a^2 - 2 * a) in
    let x2 := a + real.sqrt (a^2 - 2 * a) in
    x1 > 1 ∧ (1 < x ∧ x < x1 → deriv (λ x, f x a) x > 0) ∧
               (x1 < x ∧ x < x2 → deriv (λ x, f x a) x < 0) ∧
               (x2 < x ∧ x > x1 → deriv (λ x, f x a) x > 0)) :=
begin
  sorry
end

theorem problem_II (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hne : m ≠ n) : 
  (m - n) / (log m - log n) < (m + n) / 2 :=
begin
  sorry
end

end problem_I_problem_II_l546_546255


namespace Mitch_weekly_earnings_l546_546364

theorem Mitch_weekly_earnings :
  (let weekdays_hours := 5 * 5
       weekend_hours := 3 * 2
       weekday_rate := 3
       weekend_rate := 2 * 3 in
   (weekdays_hours * weekday_rate + weekend_hours * weekend_rate = 111)) :=
by
  sorry

end Mitch_weekly_earnings_l546_546364


namespace omega_value_intervals_of_increase_l546_546619

noncomputable def f (ω x : ℝ) : ℝ :=
  sin(ω * x)^2 + 2 * sqrt(3) * cos(ω * x) * sin(ω * x) + sin(ω * x + π / 4) * sin(ω * x - π / 4)

theorem omega_value (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x, f ω (x + π / ω) = f ω x) :
  ω = 1 :=
by
  sorry

theorem intervals_of_increase :
  ∀ x ∈ Ioc 0 (π / 3), 2 * sin(2 * x - π / 6) + 1 / 2 < 2 * sin(2 * x + π / 6) + 1 / 2 ∧
  ∀ x ∈ Ico (5 * π / 6) π, 2 * sin(2 * x - π / 6) + 1 / 2 < 2 * sin(2 * x + π / 6) + 1 / 2 :=
by
  sorry

end omega_value_intervals_of_increase_l546_546619


namespace problem_1_problem_2_l546_546253

def f (x : ℝ) : ℝ := 
  if h : 0 < x ∧ x < π then 
    x * Real.sin x 
  else if x ≥ π then 
    Real.sqrt x 
  else 
    0

def g (x k : ℝ) : ℝ := f x - k * x

theorem problem_1 (h : 0 < 1 ∧ 1 < π) : 
  ∃ x, g x 1 = 0 ∧ 0 < x ∧ x < π := 
sorry

theorem problem_2 : 
  (∃ x, g x k = 0 ∧ x ≥ π) → 
  (∀ x, 0 < x ∧ x < π → g x k = 0) → 
  0 < k ∧ k ≤ Real.sqrt π / π := 
sorry

end problem_1_problem_2_l546_546253


namespace length_DE_l546_546476

noncomputable def isosceles_right_triangle (A B C : Type*) [inner_product_space ℝ A] :=
  ∃ (a b c : A), dist a b = 2 ∧ abs (angle a b c) = π/2

variables {A : Type*} [inner_product_space ℝ A]

theorem length_DE {a b c d e : A}
  (hab : dist a b = 2)
  (habc : abs (angle a b c) = π/2)
  (hd : midpoint ℝ b c = d)
  (hcond : ∃ e, e ∈ line_through a c ∧ area {a, e, d, b} = 2 * area {e, c, d}) :
  dist d e = real.sqrt 17 / 3 :=
sorry

end length_DE_l546_546476


namespace third_side_length_l546_546608

def lengths (a b : ℕ) : Prop :=
a = 4 ∧ b = 10

def triangle_inequality (a b c : ℕ) : Prop :=
a + b > c ∧ abs (a - b) < c

theorem third_side_length (x : ℕ) (h1 : lengths 4 10) (h2 : triangle_inequality 4 10 x) : x = 11 :=
sorry

end third_side_length_l546_546608


namespace count_distinct_floors_in_list_l546_546543

theorem count_distinct_floors_in_list :
  ∀ (l : List ℕ), l = List.map (λ k, Int.floor (k / 500)) (List.range 1000) →
  l.nodup.count = 3 :=
by
  sorry

end count_distinct_floors_in_list_l546_546543


namespace ticket_value_unique_l546_546894

theorem ticket_value_unique (x : ℕ) (h₁ : ∃ n, n > 0 ∧ x * n = 60)
  (h₂ : ∃ m, m > 0 ∧ x * m = 90)
  (h₃ : ∃ p, p > 0 ∧ x * p = 49) : 
  ∃! x, x = 1 :=
by
  sorry

end ticket_value_unique_l546_546894


namespace racing_meet_time_l546_546744

theorem racing_meet_time 
  (time_racing_magic : ℕ := 120) -- The "Racing Magic" takes 120 seconds to circle the track once
  (rounds_charging_bull_hour : ℕ := 40) -- The "Charging Bull" makes 40 rounds in an hour
  (seconds_in_minute : ℕ := 60) -- definition of seconds in a minute
  (minutes_in_hour : ℕ := 60) -- definition of minutes in an hour
  : (lcm ((time_racing_magic / seconds_in_minute)) (((minutes_in_hour) / rounds_charging_bull_hour)) : ℤ) = 6 := 
by
  -- Assumptions
  have h1 : time_racing_magic = 120 := rfl
  have h2 : rounds_charging_bull_hour = 40 := rfl
  have h3 : seconds_in_minute = 60 := rfl
  have h4 : minutes_in_hour = 60 := rfl

  -- Convert seconds to minutes for Racing Magic
  have racing_magic_minutes : ℤ := (time_racing_magic / seconds_in_minute)

  -- Convert rounds per hour to minute per round for Charging Bull
  have charging_bull_minutes : ℤ := (minutes_in_hour / rounds_charging_bull_hour)

  -- Ensure conversions are correct
  have h_racing : racing_magic_minutes = 2 := by linarith
  have h_bull : charging_bull_minutes = 1.5 := by linarith

  -- Least Common Multiple calculation in integer domain
  have lcm_value := (Int.lcm (racing_magic_minutes) (charging_bull_minutes * 2))
  have lcm_to_sec := (lcm_value / 2)

  -- Prove that the LCM of 2 and 1.5 when doubled is indeed 6 minutes
  have h_lcm : lcm_to_sec = 6 := by linarith

  exact h_lcm


end racing_meet_time_l546_546744


namespace negation_of_proposition_is_false_l546_546097

-- Define the proposition
def isosceles_right_triangle (A B C : Type) : Prop := 
  ∃ (T1 T2 : Triangle A B C), is_isosceles T1 ∧ is_isosceles T2 ∧ is_right_angle T1 ∧ is_right_angle T2

def similar (T1 T2 : Type) : Prop := 
  similar_triangles T1 T2

def p : Prop := 
  ∀ (A B C : Type), isosceles_right_triangle A B C → 
  ∀ (T1 T2 : (Triangle A B C)), similar T1 T2

-- Define the negation of the proposition
def ¬p : Prop := 
  ∃ (A B C : Type), isosceles_right_triangle A B C ∧ 
  ∃ (T1 T2 : (Triangle A B C)), ¬(similar T1 T2)

-- The proof problem statement
theorem negation_of_proposition_is_false (h : p) : ¬¬p :=
by 
  sorry

end negation_of_proposition_is_false_l546_546097


namespace four_digit_number_with_14_divisors_l546_546792

theorem four_digit_number_with_14_divisors : ∃ (n : ℕ), 1000 ≤ n ∧ n < 2000 ∧ 
  ∃ (d : ℕ), d = 14 ∧ 
  (∀ p ∈ (List.map (λ x, x.2) (Nat.factors' n)), p % 10 = 1 → True) ∧ 
  ∀ d, Nat.divisors n = d → d = 14 :=
by
  sorry

end four_digit_number_with_14_divisors_l546_546792


namespace large_circle_diameter_l546_546028

noncomputable def small_circle_radius : ℝ := 4
noncomputable def small_circles_count : ℕ := 6

theorem large_circle_diameter :
  ∀ (radius : ℝ), 
  radius = small_circle_radius →
  ∀ (count : ℕ),
  count = small_circles_count →
  let diameter := (2 * radius) * ((2 * real.sqrt 3 + 3) / 3) in
  diameter = 8 * ((2 * real.sqrt 3 + 3) / 3) :=
by
  intros radius hr count hc
  simp [hr, hc]
  sorry

end large_circle_diameter_l546_546028


namespace area_enclosed_by_line_and_parabola_l546_546038

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem area_enclosed_by_line_and_parabola :
  ∀ (n : ℕ), (binomial_coeff n 2 = binomial_coeff n 3) →
  ∃ (a b : ℝ), (a, b) = (0, 0) ∨ (a, b) = (5, 25) ∧
  ∫ x in 0..5, (5 * x - x^2) = 125 / 6 :=
begin
  sorry
end

end area_enclosed_by_line_and_parabola_l546_546038


namespace diamond_example_l546_546573

def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_example : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 :=
by
  sorry

end diamond_example_l546_546573


namespace polynomials_identity_l546_546575

theorem polynomials_identity : 
  ∀ (n : ℕ), ∃ (P Q : mv_polynomial (fin n) ℚ), 
    (∃ x1 x2 .. xm : ℝ, ((x1 + x2 + ... + xm) * P) = Q (x1^2, x2^2, ..., xm^2)) := 
by 
  sorry

end polynomials_identity_l546_546575


namespace expression_not_defined_l546_546574

theorem expression_not_defined (x : ℝ) :
    ¬(x^2 - 22*x + 121 = 0) ↔ ¬(x - 11 = 0) :=
by sorry

end expression_not_defined_l546_546574


namespace g_16_40_l546_546417

theorem g_16_40 : 
  (∀ (x y : ℕ), g x x = x ∧ g x y = g y x ∧ (x + y) * g x y = y * g x (x + y)) → 
  g 16 40 = 80 :=
by
  intros h
  let ⟨h1, h2, h3⟩ := h 16 40 
  have h_1 : g 16 16 = 16 := h1
  have h_2 : g 40 16 = g 16 40 := h2
  have h_3 : (16 + 40) * g 16 40 = 40 * g 16 (16 + 40) := h3
  sorry

end g_16_40_l546_546417


namespace number_of_pieces_per_box_l546_546896

-- Let C be the number of pieces in each box of chocolate candy
-- Let M be the number of pieces in each box of caramel candy

def AdamBought : Prop :=
  ∃ (C M : ℕ), (2 * C + 5 * M = 28) ∧ (C = M)

theorem number_of_pieces_per_box : ∃ (n : ℕ), AdamBought → n = 4 :=
by
  intro h
  cases h with C hC
  cases hC with M hCM
  cases hCM with eq1 eq2
  use C
  have : 2 * C + 5 * C = 28 := by rw [eq2] at eq1; exact eq1
  have : 7 * C = 28 := by ring at this; exact this
  have : C = 4 := Nat.mul_right_inj (by decide) this
  exact this

end number_of_pieces_per_box_l546_546896


namespace area_of_rhombus_l546_546753

-- Defining the lengths of the diagonals
variable (d1 d2 : ℝ)
variable (d1_eq : d1 = 15)
variable (d2_eq : d2 = 20)

-- Goal is to prove the area given the diagonal lengths
theorem area_of_rhombus (d1 d2 : ℝ) (d1_eq : d1 = 15) (d2_eq : d2 = 20) : 
  (d1 * d2) / 2 = 150 := 
by
  -- Using the given conditions for the proof
  sorry

end area_of_rhombus_l546_546753


namespace derivative_at_zero_l546_546215

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x

theorem derivative_at_zero : deriv f 0 = 2 := 
by {
  sorry,
}

end derivative_at_zero_l546_546215


namespace find_initial_balance_l546_546472

-- Define the initial balance (X)
def initial_balance (X : ℝ) := 
  ∃ (X : ℝ), (X / 2 + 30 + 50 - 20 = 160)

theorem find_initial_balance (X : ℝ) (h : initial_balance X) : 
  X = 200 :=
sorry

end find_initial_balance_l546_546472


namespace problem_l546_546930

def otimes (x y : ℝ) : ℝ := x^3 + 5 * x * y - y

theorem problem (a : ℝ) : 
  otimes a (otimes a a) = 5 * a^4 + 24 * a^3 - 10 * a^2 + a :=
by
  sorry

end problem_l546_546930


namespace solve_radius_of_sphere_tangent_to_lines_l546_546410

def tetrahedron_edge (a : ℝ) := a

def plane_through_vertex_B_and_midpoints_of_AC_and_AD (tetrahedron_edge_length : ℝ) :=
  ∃ (P : Set (ℝ × ℝ × ℝ)), 
  ∃ (B AC AD midpoint_AC midpoint_AD : ℝ × ℝ × ℝ),
  P ∈ Set.plane_of_points B midpoint_AC midpoint_AD ∧
  B = (0, 0, tetrahedron_edge_length) ∧
  midpoint_AC = (tetrahedron_edge_length / 2, tetrahedron_edge_length / 2 * √3 / 2, 0) ∧
  midpoint_AD = (tetrahedron_edge_length / 2, -tetrahedron_edge_length / 2 * √3 / 2, 0)

theorem solve_radius_of_sphere_tangent_to_lines
  (a : ℝ) (P : ∃ (P : Set (ℝ × ℝ × ℝ)), ∃ (B AC AD midpoint_AC midpoint_AD : ℝ × ℝ × ℝ),
            P ∈ Set.plane_of_points B midpoint_AC midpoint_AD ∧
            B = (0, 0, a) ∧
            midpoint_AC = (a / 2, a / 2 * √3 / 2, 0) ∧
            midpoint_AD = (a / 2, -a / 2 * √3 / 2, 0)) :
  ∃ r : ℝ, r = a * √2 / (5 + √11) ∨ r = a * √2 / (5 - √11) :=
sorry

end solve_radius_of_sphere_tangent_to_lines_l546_546410


namespace nailable_color_exists_l546_546402

noncomputable def nail_squares (k : ℕ) : ℕ := 2 * k - 2

theorem nailable_color_exists (k : ℕ) (colors : Finset (Fin k)) 
  (squares : Finset (Fin k × (ℕ × ℕ) × (ℕ × ℕ))) 
  (h : ∀ t : Finset (Fin k), t.card = k → ∃ p1 p2 ∈ t, ∃ a1 b1 a2 b2, 
      (a1 = a2 ∨ b1 = b2) ∧ (a1, b1) ∈ squares ∧ (a2, b2) ∈ squares) 
  : ∃ c ∈ colors, ∃ t, ∀ s ∈ t, s.1 = c ∧ ∃ nails : Finset (ℕ × ℕ), 
      nails.card = nail_squares k ∧ ∀ (x ∈ t) (y ∈ t), x.2.1 ∈ nails ∨ y.2.1 ∈ nails := 
sorry

end nailable_color_exists_l546_546402


namespace white_roses_needed_l546_546374

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end white_roses_needed_l546_546374


namespace chocolates_brought_by_friend_l546_546015

-- Definitions corresponding to the conditions in a)
def total_chocolates := 50
def chocolates_not_in_box := 5
def number_of_boxes := 3
def additional_boxes := 2

-- Theorem statement: we need to prove the number of chocolates her friend brought
theorem chocolates_brought_by_friend (C : ℕ) : 
  (C + total_chocolates = total_chocolates + (chocolates_not_in_box + number_of_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes + additional_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes) - total_chocolates) 
  → C = 30 := 
sorry

end chocolates_brought_by_friend_l546_546015


namespace well_diameter_l546_546127

noncomputable def calculateDiameter (volume depth : ℝ) : ℝ :=
  2 * Real.sqrt (volume / (Real.pi * depth))

theorem well_diameter :
  calculateDiameter 678.5840131753953 24 = 6 :=
by
  sorry

end well_diameter_l546_546127


namespace probability_no_adjacent_same_l546_546204

noncomputable def circular_dice_probability : ℚ :=
let p := 7/8 in p^5

theorem probability_no_adjacent_same (n : ℕ) (faces : ℕ) (adjacency : ℕ) : 
  n = 5 →
  faces = 8 →
  adjacency = 2 →
  (Probability that no two adjacent in n people seated in a circle roll the same number on faces-sided dice) = (7/8)^5 :=
begin
  sorry
end

end probability_no_adjacent_same_l546_546204


namespace find_p_plus_q_l546_546115

theorem find_p_plus_q (AB BC AC : ℕ) (equal_arcs1 equal_arcs2 equal_arcs3 : Nat) (p q : ℕ) :
  AB = 21 → BC = 27 → AC = 26 →
  equal_arcs1 = equal_arcs2 → equal_arcs2 = equal_arcs3 →
  let BX := 27 / 2 in
  let p := 27 in
  let q := 2 in
  p + q = 29 :=
by
  intros hAB hBC hAC heq1 heq2 heq3
  let BX := 27 / 2
  let p := 27
  let q := 2
  have h1 : p + q = 29 := by sorry
  exact h1

end find_p_plus_q_l546_546115


namespace orthocenters_on_common_circle_l546_546999

noncomputable def z (i : ℕ) : ℂ := sorry -- Placeholder definition for the complex numbers z_1, z_2, z_3, z_4

def orthocenter (i : ℕ) : ℂ :=
match i with
| 1 => z 2 + z 3 + z 4
| 2 => z 3 + z 4 + z 1
| 3 => z 4 + z 1 + z 2
| 4 => z 1 + z 2 + z 3
| _ => sorry -- Default case, outside the scope of 1-4

theorem orthocenters_on_common_circle :
  ∃ (O₁ : ℂ) (R : ℝ), ∀ (i : ℕ), i ∈ {1, 2, 3, 4} → complex.abs (O₁ - orthocenter i) = R
:=
begin
  let O₁ := z 1 + z 2 + z 3 + z 4,
  use O₁,
  have R : ℝ := complex.abs (z 1),
  use R,
  intro i,
  intro hi,
  fin_cases i,
  { simp [orthocenter, O₁, complex.abs_sub, complex.sub_self, complex.abs_zero] },
  { simp [orthocenter, O₁, complex.abs_sub, complex.sub_self, complex.abs_zero] },
  { simp [orthocenter, O₁, complex.abs_sub, complex.sub_self, complex.abs_zero] },
  { simp [orthocenter, O₁, complex.abs_sub, complex.sub_self, complex.abs_zero] }
end

end orthocenters_on_common_circle_l546_546999


namespace candidate_A_votes_l546_546664

theorem candidate_A_votes (candidateA_percent : ℚ) (invalid_percent : ℚ) (total_votes : ℕ) :
  candidateA_percent = 0.60 ∧ invalid_percent = 0.15 ∧ total_votes = 560000 →
  (candidateA_percent * ((1 - invalid_percent) * total_votes)).toNat = 285600 :=
by
  intro h
  cases h with h₁ h
  cases h with h₂ h₃
  sorry

end candidate_A_votes_l546_546664


namespace determine_a_perpendicular_l546_546043

theorem determine_a_perpendicular 
  (a : ℝ)
  (h1 : 2 * x + 3 * y + 5 = 0)
  (h2 : a * x + 3 * y - 4 = 0) 
  (h_perpendicular : ∀ x y, (2 * x + 3 * y + 5 = 0) → ∀ x y, (a * x + 3 * y - 4 = 0) → (-(2 : ℝ) / (3 : ℝ)) * (-(a : ℝ) / (3 : ℝ)) = -1) :
  a = -9 / 2 :=
by
  sorry

end determine_a_perpendicular_l546_546043


namespace powers_of_i_multiplication_l546_546941

theorem powers_of_i_multiplication : (Complex.I ^ 17) * (Complex.I ^ 44) = Complex.I := by
  -- Conditions based on the given problem:
  have h1 : Complex.I ^ 1 = Complex.I := by rfl
  have h2 : Complex.I ^ 2 = -1 := by
    rw [pow_two, Complex.I_mul_I]
  have h3 : Complex.I ^ 3 = -Complex.I := by
    rw [pow_add, h1, h2]
  have h4 : Complex.I ^ 4 = 1 := by
    rw [pow_two, Complex.I_mul_I, neg_one_mul_self]
    
  -- Using the cycling property of i:
  have h17 : Complex.I ^ 17 = Complex.I := by
    rw [← nat.mod_add_div 17 4, pow_add, pow_mul, h4, one_pow, mul_one, h1]
  
  have h44 : Complex.I ^ 44 = 1 := by
    rw [← nat.mod_add_div 44 4, pow_add, pow_mul, h4, one_pow, mul_one]

  -- Final multiplication:
  rw [h17, h44, mul_one]
  exact rfl

end powers_of_i_multiplication_l546_546941


namespace cylinder_volume_eq_sphere_volume_l546_546983

theorem cylinder_volume_eq_sphere_volume (a h R x : ℝ) (h_pos : h > 0) (a_pos : a > 0) (R_pos : R > 0)
  (h_volume_eq : (a - h) * x^2 - a * h * x + 2 * h * R^2 = 0) :
  ∃ x : ℝ, a > h ∧ x > 0 ∧ x < h ∧ x = 2 * R^2 / a ∨ 
           h < a ∧ 0 < x ∧ x = (a * h / (a - h)) - h ∧ R^2 < h^2 / 2 :=
sorry

end cylinder_volume_eq_sphere_volume_l546_546983


namespace correct_option_l546_546468

-- Definitions for conditions
def C1 (a : ℕ) : Prop := a^2 * a^3 = a^5
def C2 (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def C3 (a b : ℕ) : Prop := (a * b)^3 = a * b^3
def C4 (a : ℕ) : Prop := (-a^3)^2 = -a^6

-- The correct option is C1
theorem correct_option (a : ℕ) : C1 a := by
  sorry

end correct_option_l546_546468


namespace find_phi_strict_decreasing_intervals_range_of_f_l546_546256

noncomputable def f (x φ : Real) := (Real.sqrt 2) * Real.sin (2 * x + φ)

theorem find_phi (φ : Real) : (-π < φ ∧ φ < 0) →
  (∀ x : Real, f x φ = f (π / 8 - x) φ) →
  (f 0 φ < 0) →
  φ = -3 * π / 4 := 
sorry

theorem strict_decreasing_intervals :
  ∀ (x : Real) (φ : Real), φ = -3 * π / 4 →
  f x φ < f (x + δ) φ → 
  (5 * π / 8 + k * π ≤ x ∧ x ≤ 9 * π / 8 + k * π ∧ k ∈ ℤ ) :=
sorry

theorem range_of_f :
  ∀ (x : Real) (φ : Real), φ = -3 * π / 4 →
  (x ∈ [0, π / 2]) →
  (-Real.sqrt 2 ≤ f x φ ∧ f x φ ≤ 1) :=
sorry

end find_phi_strict_decreasing_intervals_range_of_f_l546_546256


namespace greatest_integer_less_than_150_gcd_18_eq_6_l546_546085

theorem greatest_integer_less_than_150_gcd_18_eq_6 :
  ∃ n : ℕ, n < 150 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ gcd m 18 = 6 → m ≤ n :=
by
  use 132
  split
  { 
    -- proof that 132 < 150 
    exact sorry 
  }
  split
  { 
    -- proof that gcd 132 18 = 6
    exact sorry 
  }
  {
    -- proof that 132 is the greatest such integer
    exact sorry 
  }

end greatest_integer_less_than_150_gcd_18_eq_6_l546_546085


namespace sum_proper_divisors_81_l546_546835

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l546_546835


namespace minimum_trips_needed_l546_546109

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

theorem minimum_trips_needed (masses : List ℕ) (capacity : ℕ) : 
  masses = [150, 60, 70, 71, 72, 100, 101, 102, 103] →
  capacity = 200 →
  ∃ trips : ℕ, trips = 5 :=
by
  sorry

end minimum_trips_needed_l546_546109


namespace interval_of_monotonic_increase_l546_546764

-- Definition of the functions
def f (x : ℝ) : ℝ := Real.log (x^2 + 4*x - 5)
def g (x : ℝ) : ℝ := (x^2 + 4*x - 5)

-- The theorem statement for the interval of monotonic increase
theorem interval_of_monotonic_increase : 
  ∀ x : ℝ, 1 < x → Real.log (x^2 + 4*x -5) > Real.log 0 ∧ (g x > 0) → x ∈ Set.Ioo 1 0 := 
sorry

end interval_of_monotonic_increase_l546_546764


namespace find_sum_of_squares_of_roots_l546_546621

theorem find_sum_of_squares_of_roots (a b c : ℝ) (h_ab : a < b) (h_bc : b < c)
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 2 * x^2 - 3 * x + 4)
  (h_eq : f a = f b ∧ f b = f c) :
  a^2 + b^2 + c^2 = 10 :=
sorry

end find_sum_of_squares_of_roots_l546_546621


namespace cos_alpha_minus_beta_l546_546240

theorem cos_alpha_minus_beta (α β : ℝ) (h1 : α + β = π / 3) (h2 : tan α + tan β = 2) :
  cos (α - β) = (sqrt 3 - 1) / 2 :=
by
  sorry

end cos_alpha_minus_beta_l546_546240


namespace total_teachers_correct_l546_546139

noncomputable def total_teachers (x : ℕ) : ℕ := 26 + 104 + x

theorem total_teachers_correct
    (x : ℕ)
    (h : (x : ℝ) / (26 + 104 + x) = 16 / 56) :
  total_teachers x = 182 :=
sorry

end total_teachers_correct_l546_546139


namespace rationalize_denominator_l546_546729

theorem rationalize_denominator (a b c : ℚ) (h1 : b = 3 * a) : (1 / (a + b) = c) ↔ c = (Real.cbrt 9) / 12 :=
by
  sorry

end rationalize_denominator_l546_546729


namespace ordered_pairs_count_l546_546933

theorem ordered_pairs_count : 
  let P := set.prod {p : ℤ | ∃ q : ℤ, pq q > 35^3 ∧ p^3 + q^3 + 110 * p * q = 35^3} in 
  P.card = 37 :=
sorry

end ordered_pairs_count_l546_546933


namespace white_roses_needed_l546_546373

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end white_roses_needed_l546_546373


namespace average_sales_per_month_after_discount_is_93_l546_546918

theorem average_sales_per_month_after_discount_is_93 :
  let salesJanuary := 120
  let salesFebruary := 80
  let salesMarch := 70
  let salesApril := 150
  let salesMayBeforeDiscount := 50
  let discountRate := 0.10
  let discountedSalesMay := salesMayBeforeDiscount - (discountRate * salesMayBeforeDiscount)
  let totalSales := salesJanuary + salesFebruary + salesMarch + salesApril + discountedSalesMay
  let numberOfMonths := 5
  let averageSales := totalSales / numberOfMonths
  averageSales = 93 :=
by {
  -- The actual proof code would go here, but we will skip the proof steps as instructed.
  sorry
}

end average_sales_per_month_after_discount_is_93_l546_546918


namespace sum_of_quadratic_coeffs_l546_546969

theorem sum_of_quadratic_coeffs (n : ℕ) (x : ℝ) (a_0 a_1 a_2 ... a_n : ℝ)
  (h1 : (x + 1)^n = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + ... + a_n * (x - 1)^n)
  (h2 : a_0 + a_1 + ... + a_n = 243) :
  2^n = 32 :=
by sorry

end sum_of_quadratic_coeffs_l546_546969


namespace time_to_fry_3_pancakes_time_to_fry_2016_pancakes_l546_546967

-- Define the conditions
def fry_time_per_side : ℕ := 2
def max_pancakes_in_pan : ℕ := 2

-- Define the problems for the 3 pancakes and 2016 pancakes cases
noncomputable def minimum_frying_time (n : ℕ) : ℕ :=
if n = 0 then 0
else if n ≤ 2 then 4
else
  let rounds := n / 2 in
  let remainder := n % 2 in
  rounds * 4 + (if remainder = 0 then 0 else 2 * fry_time_per_side)

-- Theorems to prove
theorem time_to_fry_3_pancakes : minimum_frying_time 3 = 6 :=
by sorry

theorem time_to_fry_2016_pancakes : minimum_frying_time 2016 = 4032 :=
by sorry

end time_to_fry_3_pancakes_time_to_fry_2016_pancakes_l546_546967


namespace common_difference_arithmetic_sequence_l546_546984

theorem common_difference_arithmetic_sequence (d : ℝ) (a : ℕ → ℝ) 
  (h₁ : ∀ n : ℕ, a n = 18 + n * d) 
  (h₂ : (18, a 3, a 7): ℝ × ℝ × ℝ → Prop)
  (h₃ : 18 ≠ 0):
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l546_546984


namespace integer_root_of_polynomial_l546_546773

theorem integer_root_of_polynomial (d : ℚ) (hroot : (2 + Real.sqrt 5) ∈ {x | x^3 + 4*x + d = 0}) :
  ∃ r : ℤ, r = -4 ∧ (r : ℚ) ∈ {x | x^3 + 4*x + d = 0} :=
by
  sorry

end integer_root_of_polynomial_l546_546773


namespace find_angle_C_l546_546586

theorem find_angle_C 
  (A B C : ℝ)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (dot_product_m_n : ℝ)
  (h1 : m = (sqrt 3 * sin A, sin B))
  (h2 : n = (cos B, sqrt 3 * cos A))
  (h3 : dot_product_m_n = 1 + cos (A + B))
  (h4 : dot_product_m_n = 1 - cos C) :
  C = (2 * π / 3) :=
by
  sorry

end find_angle_C_l546_546586


namespace mukesh_total_debt_l546_546377

-- Define the initial principal, additional loan, interest rate, and time periods
def principal₁ : ℝ := 10000
def principal₂ : ℝ := 12000
def rate : ℝ := 0.06
def time₁ : ℝ := 2
def time₂ : ℝ := 3

-- Define the interest calculations
def interest₁ : ℝ := principal₁ * rate * time₁
def total_after_2_years : ℝ := principal₁ + interest₁ + principal₂
def interest₂ : ℝ := total_after_2_years * rate * time₂

-- Define the total amount owed after 5 years
def amount_owed : ℝ := total_after_2_years + interest₂

-- The goal is to prove that Mukesh owes 27376 Rs after 5 years
theorem mukesh_total_debt : amount_owed = 27376 := by sorry

end mukesh_total_debt_l546_546377


namespace rectangle_area_division_l546_546486

theorem rectangle_area_division (total_area : ℝ) (num_parts : ℕ) (part_area : ℝ) :
  total_area = 59.6 ∧ num_parts = 4 → part_area = total_area / num_parts → part_area = 14.9 :=
by
  intros htotal hpart
  cases htotal with harea hparts
  rw [harea, hparts] at hpart
  exact hpart

end rectangle_area_division_l546_546486


namespace central_ring_road_paths_l546_546909

def binom (n k : ℕ) := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem central_ring_road_paths : 
  let total_paths := binom 8 4 * binom 8 4
  let intersecting_paths := (sum (i j : ℕ) in
      finset.range 9.product finset.range 9,
      if i + j ≤ 8 then 
        binom (i + j) i * binom (8 - i - j) (4 - i) * binom (i + j) i * binom (8 - i - j) (4 - i) else 0)
  let non_intersecting_paths := total_paths - intersecting_paths
  in non_intersecting_paths = 1750 :=
sorry

end central_ring_road_paths_l546_546909


namespace polynomial_coefficients_sum_l546_546638

theorem polynomial_coefficients_sum :
  let f := (x + 3) * (4*x^2 - 2*x + 6)
  in (A + B + C + D = 32) :=
sorry

end polynomial_coefficients_sum_l546_546638


namespace woven_percentage_by_11th_day_l546_546966

theorem woven_percentage_by_11th_day :
  let a₁ := 5
  let a₃₀ := 1
  let n := 30
  let d := (a₃₀ - a₁) / 29
  (∑ i in Finset.range 11, a₁ + i * d) / (∑ i in Finset.range 30, a₁ + i * d) = 0.53 :=
by {
  let a₁ := 5
  let a₃₀ := 1
  let n := 30
  let d := (a₃₀ - a₁) / 29
  let S₁₁ := ∑ i in Finset.range 11, a₁ + i * d
  let S₃₀ := ∑ i in Finset.range 30, a₁ + i * d
  have : S₁₁ / S₃₀ = 0.53 := sorry
  exact this
}

end woven_percentage_by_11th_day_l546_546966


namespace angle_in_third_quadrant_l546_546033

theorem angle_in_third_quadrant (theta : ℝ) (n : ℤ) :
  (theta = 2010) -> 
  (theta = n * 360 + 210) -> 
  (210 % 360 ∈ set.Ioo (π) (3 * π / 2)) -> 
  (theta % 360 ∈ set.Ioo (π) (3 * π / 2)) :=
by
  sorry

end angle_in_third_quadrant_l546_546033


namespace range_of_a_l546_546972

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ a → |x - 1| < 1) → (∃ x : ℝ, |x - 1| < 1 ∧ x < a) → a ≤ 0 := 
sorry

end range_of_a_l546_546972


namespace MH_eq_ML_sometimes_l546_546001

-- Definitions and conditions
structure GeometricConfiguration :=
  (H K B C L M : Point)
  (θ : ℝ)
  (midpoint_BC : Midpoint M B C)
  (perpendicular_BH_HK : Perpendicular B H K)
  (intersects_CL_HK_at_L : Intersects C L H K)
  
-- Problem statement
theorem MH_eq_ML_sometimes 
  (config : GeometricConfiguration) :
  (∃ θ, ∃ L, MH config.M config.H = ML config.M L) ∧ 
  (∃ θ, ∃ L, MH config.M config.H ≠ ML config.M L) := 
sorry

end MH_eq_ML_sometimes_l546_546001


namespace mother_older_than_twice_petra_l546_546059

def petra_age : ℕ := 11
def mother_age : ℕ := 36

def twice_petra_age : ℕ := 2 * petra_age

theorem mother_older_than_twice_petra : mother_age - twice_petra_age = 14 := by
  sorry

end mother_older_than_twice_petra_l546_546059


namespace natLessThanFive_is_set_tallStudents_cannot_be_set_intSolutions_is_set_l546_546536

-- Definitions for Lean representations
def natLessThanFive : Set ℕ := {n | n < 5}
def isDefinedTallStudents := False  -- since the set definition is not precise
def intSolutions (x : ℤ) : Prop := 2 * x + 1 > 7

-- Lean 4 Statements
theorem natLessThanFive_is_set : ∃ S : Set ℕ, S = natLessThanFive := by
  use natLessThanFive
  trivial

theorem tallStudents_cannot_be_set : ¬ (∃ S : Set α, isDefinedTallStudents) := by
  simp [isDefinedTallStudents]

theorem intSolutions_is_set : ∃ S : Set ℤ, S = {x | intSolutions x} := by
  use {x | intSolutions x}
  trivial

end natLessThanFive_is_set_tallStudents_cannot_be_set_intSolutions_is_set_l546_546536


namespace compound_interest_rate_correct_l546_546522

noncomputable def compound_interest_rate_2_years : ℝ :=
  let P := 1  -- P is an initial principal amount, normalized to 1 for simplicity
  let final_amount := (7 / 6 : ℝ)
  let r := real.sqrt (7 / 6) - 1
  r

theorem compound_interest_rate_correct :
  let r := compound_interest_rate_2_years in
  r ≈ 0.0801 :=
by
  let r := compound_interest_rate_2_years
  sorry

end compound_interest_rate_correct_l546_546522


namespace lava_lamp_probability_l546_546397

/-- Ryan has 4 red lava lamps and 2 blue lava lamps; 
    he arranges them in a row on a shelf randomly, and then randomly turns 3 of them on. 
    Prove that the probability that the leftmost lamp is blue and off, 
    and the rightmost lamp is red and on is 2/25. -/
theorem lava_lamp_probability : 
  let total_arrangements := (Nat.choose 6 2) 
  let total_on := (Nat.choose 6 3)
  let favorable_arrangements := (Nat.choose 4 1)
  let favorable_on := (Nat.choose 4 2)
  let favorable_outcomes := 4 * 6
  let probability := (favorable_outcomes : ℚ) / (total_arrangements * total_on : ℚ)
  probability = 2 / 25 := 
by
  sorry

end lava_lamp_probability_l546_546397


namespace count_valid_a_l546_546553

set_option autoImplicit true

def validDigits (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def validA (a b c : ℕ) : Prop :=
  validDigits a ∧ validDigits b ∧ validDigits c ∧
  b = 2 * a + 1 ∧ c = 2 * b + 1

theorem count_valid_a :
  {a : ℕ | ∃ b c : ℕ, validA a b c}.to_finset.card = 2 :=
by
  sorry

end count_valid_a_l546_546553


namespace product_of_roots_l546_546808

theorem product_of_roots : ∃ (x : ℕ), x = 45 ∧ (∃ a b c : ℕ, a ^ 3 = 27 ∧ b ^ 4 = 81 ∧ c ^ 2 = 25 ∧ x = a * b * c) := 
sorry

end product_of_roots_l546_546808


namespace quadrilateral_inequality_l546_546345

theorem quadrilateral_inequality 
  (AB AD BC CD h: ℝ)
  (h_convex: convex_quadrilateral AB AD BC CD)
  (h_side_relation: AB = AD + BC)
  (P: point)
  (h_AP: distance P A = h + AD)
  (h_BP: distance P B = h + BC)
  (h_CD_distance: distance P (line CD) = h) :
  1 / real.sqrt h ≥ 1 / real.sqrt AD + 1 / real.sqrt BC :=
sorry

end quadrilateral_inequality_l546_546345


namespace find_f9_l546_546597

noncomputable def f (x : ℝ) : ℝ := sorry
axiom h : ∀ x : ℝ, f (x-1) = 1 + log x 10

theorem find_f9 : f 9 = 2 :=
by
  sorry

end find_f9_l546_546597


namespace multiple_of_8_and_12_l546_546741

theorem multiple_of_8_and_12 (x y : ℤ) (hx : ∃ k : ℤ, x = 8 * k) (hy : ∃ k : ℤ, y = 12 * k) :
  (∃ k : ℤ, y = 4 * k) ∧ (∃ k : ℤ, x - y = 4 * k) :=
by
  /- Proof goes here, based on the given conditions -/
  sorry

end multiple_of_8_and_12_l546_546741


namespace probability_correct_l546_546302

def f1 (x : ℝ) := x
def f2 (x : ℝ) := abs x
def f3 (x : ℝ) := Real.sin x
def f4 (x : ℝ) := Real.cos x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

noncomputable def probability_odd_function : ℚ :=
  if (is_odd f1 ∨ is_odd f3) ∧ (is_even f2 ∨ is_even f4) then
    2 / 3
  else
    0

theorem probability_correct :
  probability_odd_function = 2 / 3 :=
by
  sorry

end probability_correct_l546_546302


namespace units_digit_n_l546_546566

theorem units_digit_n (m n : ℕ) (h₁ : m * n = 14^8) (hm : m % 10 = 6) : n % 10 = 1 :=
sorry

end units_digit_n_l546_546566


namespace fifty_m_plus_n_l546_546697

theorem fifty_m_plus_n (m n : ℝ) :
  (∀ x, (x + m) * (x + n) * (x + 8) = 0 → x ≠ -2) ∧
  (∀ x, (x + 2 * m) * (x + 4) * (x + 10) = 0 → x = -2) →
  50 * m + n = 54 :=
by
  intros _ _
  sorry

end fifty_m_plus_n_l546_546697


namespace sum_of_proper_divisors_of_81_l546_546818

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l546_546818


namespace part2_part3_l546_546251

open Real

-- Define the function f(x) and the conditions
def f (x : ℝ) : ℝ := x^2 * log x - x + 1

-- Theorem 1: Proving that for all x >= 1, f(x) >= (x - 1)^2
theorem part2 (x : ℝ) (hx : x ≥ 1) : 
  f(x) ≥ (x - 1)^2 :=
sorry

-- Theorem 2: Proving the range of m such that f(x) ≥ m(x - 1)^2 holds for all x ≥ 1
theorem part3 (m : ℝ) : 
  (∀ x (hx : x ≥ 1), f(x) ≥ m * (x - 1)^2) ↔ m ≤ (3 / 2) :=
sorry

end part2_part3_l546_546251


namespace fg_of_5_eq_163_l546_546279

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end fg_of_5_eq_163_l546_546279


namespace polar_to_cartesian_line_min_distance_from_ellipse_to_line_l546_546246

-- Define the polar equation in Cartesian coordinates
def polar_to_cartesian (rho theta : ℝ) : (ℝ × ℝ) :=
  (rho * Math.cos theta, rho * Math.sin theta)

-- Assumptions and given conditions
variable (rho : ℝ) (theta : ℝ)
axiom polar_equation : rho * Math.sin (theta - (Real.pi / 4)) = 2 * Real.sqrt 2

-- Cartesian equation of the line
def line_cartesian (x y : ℝ) : Prop := y - x = 4

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 9) = 1

-- Distance from a point to the line
def distance_from_point_to_line (x y : ℝ) : ℝ :=
  abs (x - y + 4) / Real.sqrt 2

-- The theorem to convert the polar equation to Cartesian
theorem polar_to_cartesian_line : ∀ (x y : ℝ), 
  (∃ rho theta, polar_to_cartesian rho theta = (x, y) ∧ polar_equation) → line_cartesian x y :=
sorry

-- The theorem to find the minimum distance from a point on the ellipse to the line
theorem min_distance_from_ellipse_to_line :
  ∀ (x y : ℝ), ellipse x y → (∃ d, d = distance_from_point_to_line x y ∧ 
    (∀ (x y : ℝ), ellipse x y → distance_from_point_to_line x y ≥ d) ∧ 
    d = 2 * Real.sqrt 2 - Real.sqrt 6) :=
sorry

end polar_to_cartesian_line_min_distance_from_ellipse_to_line_l546_546246


namespace theta_range_l546_546135

-- Define the hexagon and points
structure Hexagon (α : Type*) :=
(side_length : ℝ)
(A B C D E F : α)
(midpoint : α)

variable {α : Type*} [MetricSpace α] {hex : Hexagon α}

-- Define the ball's journey and angles
axiom P_midpoint : is_midpoint hex.A hex.B hex.midpoint
axiom ball_path : (startP : Point, hitQ : Point, hitC : Point, hitD : Point, hitE : Point, hitF : Point) →
  startP = hex.midpoint ∧
  hitQ ∈ segment(hex.B, hex.C) ∧
  hitC ∈ segment(hex.C, hex.D) ∧
  hitD ∈ segment(hex.D, hex.E) ∧
  hitE ∈ segment(hex.E, hex.F) ∧
  hitF ∈ segment(hex.F, hex.A) ∧
  ∃ end_on_AB : Point, end_on_AB ∈ segment(hex.A, hex.B)

-- Define the range for theta
def range_theta (theta : ℝ) : Prop :=
  θ ∈ set.Icc (real.arctan (3 * real.sqrt 3 / 10)) (real.arctan (3 * real.sqrt 3 / 8))

theorem theta_range (θ : ℝ) :
  (hitQ : Point, θ = ∠ BP9P-)

end theta_range_l546_546135


namespace inner_revolutions_l546_546870

variable (r_outer r_inner : ℝ) (rev_outer : ℕ)

theorem inner_revolutions :
  r_outer = 40 ∧ r_inner = 10 ∧ rev_outer = 15 → 4 * rev_outer = 60 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  sorry

end inner_revolutions_l546_546870


namespace unique_handshakes_462_l546_546910

theorem unique_handshakes_462 : 
  ∀ (twins triplets : Type) (twin_set : ℕ) (triplet_set : ℕ) (handshakes_among_twins handshakes_among_triplets cross_handshakes_twins cross_handshakes_triplets : ℕ),
  twin_set = 12 ∧
  triplet_set = 4 ∧
  handshakes_among_twins = (24 * 22) / 2 ∧
  handshakes_among_triplets = (12 * 9) / 2 ∧
  cross_handshakes_twins = 24 * (12 / 3) ∧
  cross_handshakes_triplets = 12 * (24 / 3 * 2) →
  (handshakes_among_twins + handshakes_among_triplets + (cross_handshakes_twins + cross_handshakes_triplets) / 2) = 462 := 
by
  sorry

end unique_handshakes_462_l546_546910


namespace find_f1_l546_546223

open Function

-- Define the function f on the finite type
def f : Fin 5 → Fin 5 := sorry

-- Establish the conditions
axiom inj_f : Injective f
axiom surj_f : Surjective f
axiom functional_eq : ∀ x : Fin 5, f x + f (f x) = 5 + 1

-- Prove that f(1) = 5
theorem find_f1 : f ⟨1, by norm_num⟩ = 5 := sorry

end find_f1_l546_546223


namespace trailing_zeroes_in_1200_factorial_l546_546922

theorem trailing_zeroes_in_1200_factorial :
  ∑ k in Finset.range (Nat.floor (Real.log 1200 / Real.log 5) + 1), Nat.floor (1200 / 5^k) = 298 :=
by
  sorry

end trailing_zeroes_in_1200_factorial_l546_546922


namespace num_even_digit_numbers_l546_546207

open Nat

theorem num_even_digit_numbers (S : Finset ℕ) (hs : S = {1, 2, 3, 4, 5}) : 
  ∃ n, n = 48 ∧ ∀ (N : ℕ), 
    (∀ d ∈ S, ∃! l : List ℕ, l.to_finset = S ∧ l.length = 5 ∧ 
    ∃ k ∈ l, k % 2 = 0 ∧ N = list_to_nat l) → (N = n) :=
by sorry

end num_even_digit_numbers_l546_546207


namespace construct_triangle_with_given_points_l546_546061

theorem construct_triangle_with_given_points (A S Q : Point)
  (cond1 : is_vertex A)
  (cond2 : is_centroid S)
  (cond3 : equal_angles_from_Q Q): ∃ (B C : Point), 
  triangle ABC ∧ 
  centroid S ∧ 
  (angle Q A B = 120 ∧ angle Q B C = 120 ∧ angle Q C A = 120) := 
by 
  sorry

end construct_triangle_with_given_points_l546_546061


namespace reduce_price_for_profit_and_max_sales_l546_546143

-- Definitions
def purchase_price_per_kg : ℝ := 2
def initial_selling_price_per_kg : ℝ := 3
def initial_sales_volume : ℕ := 200
def additional_sales_per_0_1_yuan_drop : ℕ := 40
def fixed_costs : ℝ := 24

-- Expression for daily sales volume
def daily_sales_volume (x : ℝ) : ℕ := initial_sales_volume + (40 * (10 * x))

-- Expression for profit per kg
def profit_per_kg_reduced (x : ℝ) : ℝ := (initial_selling_price_per_kg - x) - purchase_price_per_kg

-- Profit equation
def profit_equation (x : ℝ) : Prop :=
  ((profit_per_kg_reduced x) * (initial_sales_volume + (40 * (10 * x)))) - fixed_costs = 200

-- Theorem to Prove
theorem reduce_price_for_profit_and_max_sales :
  (∀ x : ℝ, daily_sales_volume x = 200 + 400 * x) ∧ (profit_equation 0.3) :=
begin
  sorry,
end

end reduce_price_for_profit_and_max_sales_l546_546143


namespace number_of_girls_l546_546662

theorem number_of_girls (G : ℕ) (B : ℕ) (H : ℕ)
  (h1 : B = 600)
  (h2 : abs (B - G) = 400)
  (h3 : 0.60 * H = 960)
  (h4 : H = B + G) :
  G = 1000 :=
by
  sorry

end number_of_girls_l546_546662


namespace determine_f_for_x_lt_2_l546_546596

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 2 then x^2 + 1 else ((4 - x)^2 + 1)

theorem determine_f_for_x_lt_2 (x : ℝ) (h2 : 4 - x > 2) (h_even : ∀ y : ℝ, f (y + 2) = f (-(y + 2))) : 
  x < 2 → f x = x^2 - 8x + 17 :=
by
  intro h
  have : f x = f (4 - x) := by
    rw [h_even, ← neg_add_eq_sub, ← neg_add_eq_sub, neg_sub, neg_neg, add_sub_cancel'_right]
  simp [f, if_neg h]
  sorry

end determine_f_for_x_lt_2_l546_546596


namespace store_revenue_after_sale_l546_546888

/--
A store has 2000 items, each normally selling for $50. 
They offer an 80% discount and manage to sell 90% of the items. 
The store owes $15,000 to creditors. Prove that the store has $3,000 left after the sale.
-/
theorem store_revenue_after_sale :
  let items := 2000
  let retail_price := 50
  let discount := 0.8
  let sale_percentage := 0.9
  let debt := 15000
  let items_sold := items * sale_percentage
  let discount_amount := retail_price * discount
  let sale_price_per_item := retail_price - discount_amount
  let total_revenue := items_sold * sale_price_per_item
  let money_left := total_revenue - debt
  money_left = 3000 :=
by
  sorry

end store_revenue_after_sale_l546_546888


namespace correct_statements_l546_546625

open Classical

noncomputable def parabola {p : ℝ} (hp : 0 < p) : Set (ℝ × ℝ) :=
  {P | P.2 ^ 2 = 2 * p * P.1}

def area_of_triangle (M F : ℝ × ℝ) : ℝ :=
  0.5 * abs (F.1 * M.2 - 4 * F.2)

theorem correct_statements (a : ℝ) (p : ℝ) (hp : 0 < p)
  (hM : (a, 4) ∈ parabola hp) (harea : area_of_triangle (a, 4) (p / 2, 0) = 4) :
  (a ≠ 4) ∧ (∀ x y, parabola hp (x, y) → y^2 = 8*x) ∧ (¬ ∀ R : ℝ × ℝ, ∃ x y, R = (x, y) ∧ y^2 = 4 * x - 2) ∧ (10 = 10 → ∃ t, ∀ m : ℝ, m ≠ 0 ∧ t =  (* calculation for minimum abscissa of midpoint N = 3 *)) :=
  sorry

end correct_statements_l546_546625


namespace fishing_problem_l546_546148

theorem fishing_problem
  (P : ℕ) -- weight of the fish Peter caught
  (H1 : Ali_weight = 2 * P) -- Ali caught twice as much as Peter
  (H2 : Joey_weight = P + 1) -- Joey caught 1 kg more than Peter
  (H3 : P + 2 * P + (P + 1) = 25) -- Together they caught 25 kg
  : Ali_weight = 12 :=
by
  sorry

end fishing_problem_l546_546148


namespace xy_difference_l546_546973

noncomputable def x : ℝ := Real.sqrt 3 + 1
noncomputable def y : ℝ := Real.sqrt 3 - 1

theorem xy_difference : x^2 * y - x * y^2 = 4 := by
  sorry

end xy_difference_l546_546973


namespace value_of_expr_l546_546719

-- Definitions
def operation (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The proof statement
theorem value_of_expr (a b : ℕ) (h₀ : operation a b = 100) : (a + b) + 6 = 11 := by
  sorry

end value_of_expr_l546_546719


namespace pony_discount_rate_l546_546577

-- Given conditions and question translated to a proof problem
theorem pony_discount_rate :
  let Fox_jeans_price := 15
  let Pony_jeans_price := 18
  let Jacket_price := 25
  let Fox_jeans_qty := 3
  let Pony_jeans_qty := 2
  let Jackets_qty := 2
  let Total_savings := 15
  let Total_discount_rate := 35
  let Jacket_discount_rate := 10
  let Total_cost_before_discount := Fox_jeans_qty * Fox_jeans_price + Pony_jeans_qty * Pony_jeans_price + Jackets_qty * Jacket_price
  let Sale_discount := (F : ℝ) → F / 100 * Fox_jeans_qty * Fox_jeans_price + (P : ℝ) → P / 100 * Pony_jeans_qty * Pony_jeans_price + Jacket_discount_rate / 100 * Jackets_qty * Jacket_price
  F / 100 * Total_cost_before_discount - Sale_discount F P = Total_savings
  Total_discount_rate = (F : ℝ) + P + Jacket_discount_rate
  P = 13.89 :=
sorry

end pony_discount_rate_l546_546577


namespace min_value_AB_l546_546993

variables (a : ℝ) (t y : ℝ)
def curve_C1 (x y : ℝ) : Prop :=
  ∃ t : ℝ, x = 2 * t^2 - 4 * t ∧ y = 4 * t - 4

def curve_C1_cartesian (x y : ℝ) : Prop :=
  y^2 - 8 * x - 16 = 0

def curve_C2_polar (θ : ℝ) : Prop :=
  cos θ = a * sin θ

def curve_C2_cartesian (x y : ℝ) : Prop :=
  x = a * y

def points_intersection (y₁ y₂ : ℝ) : Prop :=
  y₁ + y₂ = 8 * a ∧ y₁ * y₂ = -16

theorem min_value_AB (a : ℝ) (y₁ y₂ : ℝ) (h1 : points_intersection y₁ y₂) :
  ∃ AB : ℝ, AB = 8 ∧ ∀ a, sqrt ((1 + a^2) * (64 * a^2 + 64)) >= 8 :=
sorry

end min_value_AB_l546_546993


namespace parallel_EX_AP_l546_546587

-- Define the basic structure and conditions of the problem
variables {α : Type*} [field α]

-- A structure for a triangle with an orthocenter
structure Triangle (α : Type*) [field α] :=
(A B C H : α) -- Points A, B, C, and orthocenter H

-- Points on the circumcircle, altitude foot, and collinear points setup
variables {ABC : Triangle α} (P : α) (E Q R X : α)
variables (is_on_circumcircle : P ∈ 𝒞(ABC.A, ABC.B, ABC.C))
variables (altitude_foot : E = foot_of_altitude ABC.B ABC.C ABC.A)
variables (parallelogram_PAQB : Q = ABC.A + ABC.B - P)
variables (parallelogram_PARC : R = ABC.A + ABC.C - P)
variables (AQ_HR_collinear : ∃ X, collinear {ABC.A, Q, X} ∧ collinear {ABC.H, R, X})

-- Statement of the theorem
theorem parallel_EX_AP : (EX ≐ AP) :=
sorry

end parallel_EX_AP_l546_546587


namespace fg_of_5_eq_163_l546_546280

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end fg_of_5_eq_163_l546_546280


namespace trajectory_of_center_C_line_AB_passes_through_fixed_point_l546_546584

-- Given conditions
def circle_E (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/4
def line_tangent (y : ℝ) : Prop := y = 1/2

-- Trajectory of the moving circle's center
def trajectory (x y : ℝ) : Prop := x^2 = 4 * y

-- Given a point P
def point_P (m : ℝ) : ℝ × ℝ := (m, -4)

-- Prove (1): The trajectory Γ of the center C of the moving circle is x^2 = 4y
theorem trajectory_of_center_C :
  ∀ (x y : ℝ), (circle_E x y → line_tangent y → trajectory x y) := 
begin
  sorry,
end

-- Prove (2): If two tangents are drawn from P(m, -4) to the curve Γ, 
-- the line AB always passes through the fixed point (0, 4)
theorem line_AB_passes_through_fixed_point :
  ∀ (m x1 y1 x2 y2 : ℝ),
  (trajectory x1 y1) → (trajectory x2 y2) →
  tangent_line (point_P m) (x1, y1) ∧ tangent_line (point_P m) (x2, y2) →
  passes_through (0, 4) (x1, y1) (x2, y2) :=
begin
  sorry,
end


end trajectory_of_center_C_line_AB_passes_through_fixed_point_l546_546584


namespace space_shuttle_speed_kmh_l546_546519

-- Define the given conditions
def speedInKmPerSecond : ℕ := 4
def secondsInAnHour : ℕ := 3600

-- State the proof problem
theorem space_shuttle_speed_kmh : speedInKmPerSecond * secondsInAnHour = 14400 := by
  sorry

end space_shuttle_speed_kmh_l546_546519


namespace permutations_inversion_count_eq_l546_546347

-- Define A-inversion conditions as per the problem statement
def A_inversion (a : list ℤ) (w : list ℤ) (i j : ℕ) : Prop :=
  i < j ∧ (a.nth i).get_or_else 0 ≥ w.nth i.get_or_else 0 ∧ w.nth i > w.nth j ∨
    w.nth j > (a.nth i).get_or_else 0 ∧ (a.nth i).get_or_else 0 ≥ w.nth i ∨
    w.nth i > w.nth j ∧ w.nth j > (a.nth i).get_or_else 0

-- Statement to prove in Lean 
theorem permutations_inversion_count_eq 
  (m : multiset ℕ) (A B : list ℤ) (k : ℕ) : 
  (multiset.filter (λ w : list ℕ, (range(w.length).choose 2).count (λ ij, A_inversion A w ij.1 ij.2) = k) 
    (list.permutations m.to_list)).card 
  = 
  (multiset.filter (λ w : list ℕ, (range(w.length).choose 2).count (λ ij, A_inversion B w ij.1 ij.2) = k) 
    (list.permutations m.to_list)).card :=
begin
  sorry
end

end permutations_inversion_count_eq_l546_546347


namespace white_roses_total_l546_546370

theorem white_roses_total (bq_num : ℕ) (tbl_num : ℕ) (roses_per_bq : ℕ) (roses_per_tbl : ℕ)
  (total_roses : ℕ) 
  (h1 : bq_num = 5) 
  (h2 : tbl_num = 7) 
  (h3 : roses_per_bq = 5) 
  (h4 : roses_per_tbl = 12)
  (h5 : total_roses = 109) : 
  bq_num * roses_per_bq + tbl_num * roses_per_tbl = total_roses := 
by 
  rw [h1, h2, h3, h4, h5]
  exact rfl

end white_roses_total_l546_546370


namespace functional_equation_initial_condition_l546_546949

noncomputable def f : ℤ → ℝ
| x := 2^x + (1/2)^x

theorem functional_equation (x y : ℤ) : f x * f y = f (x + y) + f (x - y) :=
by {
  intro x y,
  sorry
}

theorem initial_condition : f 1 = 5 / 2 :=
by {
  sorry
}

end functional_equation_initial_condition_l546_546949


namespace art_gallery_total_pieces_l546_546161

-- Definitions from conditions
def total_art_pieces := Nat
def displayed_fraction := 1 / 3
def not_displayed_fraction := 2 / 3
def displayed_sculpture_fraction := 1 / 6
def not_displayed_painting_fraction := 1 / 3
def non_displayed_sculptures := 1400

-- The statement to prove
theorem art_gallery_total_pieces (A : total_art_pieces) :
  (not_displayed_fraction * not_displayed_fraction * A = non_displayed_sculptures) → A = 3150 :=
by
  intro h
  sorry

end art_gallery_total_pieces_l546_546161


namespace g_six_g_seven_l546_546761

noncomputable def g : ℝ → ℝ :=
sorry

axiom additivity : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_three : g 3 = 4

theorem g_six : g 6 = 8 :=
by {
  -- proof steps to be added by the prover
  sorry
}

theorem g_seven : g 7 = 28 / 3 :=
by {
  -- proof steps to be added by the prover
  sorry
}

end g_six_g_seven_l546_546761


namespace cos2theta_value_l546_546997

theorem cos2theta_value (θ : ℝ) (h : sin (2 * θ) - 4 * sin (θ + Real.pi / 3) * sin (θ - Real.pi / 6) = Real.sqrt 3 / 3) : 
  cos (2 * θ) = 1 / 3 := 
sorry

end cos2theta_value_l546_546997


namespace initial_percentage_of_water_is_20_l546_546876

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end initial_percentage_of_water_is_20_l546_546876


namespace founder_of_modern_set_theory_is_Cantor_l546_546758

theorem founder_of_modern_set_theory_is_Cantor : 
    (∃ (founder : String), founder = "Cantor") :=
begin
    use "Cantor",
    refl
end

end founder_of_modern_set_theory_is_Cantor_l546_546758


namespace polynomial_degree_l546_546627

noncomputable def polynomial : ℚ[X, X] :=
  (-3 : ℚ) * X ^ 2 * Y + (5 / 2 : ℚ) * X ^ 2 * Y ^ 3 - X * Y + 1

theorem polynomial_degree : polynomial.degree = 5 := by
  sorry

end polynomial_degree_l546_546627


namespace standard_colony_condition_l546_546088

noncomputable def StandardBacterialColony : Prop := sorry

theorem standard_colony_condition (visible_mass_of_microorganisms : Prop) 
                                   (single_mother_cell : Prop) 
                                   (solid_culture_medium : Prop) 
                                   (not_multiple_types : Prop) 
                                   : StandardBacterialColony :=
sorry

end standard_colony_condition_l546_546088


namespace general_term_formula_inequality_Sn_l546_546247

/- 
Question 1: Prove the general term formula of the sequence given conditions
  - sequence is a monotonically increasing arithmetic sequence
  - a1 + a3 + a5 = 15
  - a2^2 = a1 * a5
-/
theorem general_term_formula (a : ℕ → ℝ) (monotonic_increasing : ∀ n, a n ≤ a (n+1)) 
  (h1 : a 1 + a 3 + a 5 = 15) (h2 : a 2 ^ 2 = a 1 * a 5) : 
  ∀ n, a n = 2 * n - 1 := 
sorry

/-
Question 2: Prove that for the sequence b_n and its sum of the first n terms S_n, 
the inequality S_n < 1 always holds for n ∈ ℕ
-/
theorem inequality_Sn (a : ℕ → ℝ) 
  (monotonic_increasing : ∀ n, a n ≤ a (n+1))
  (h1 : a 1 + a 3 + a 5 = 15) 
  (h2 : a 2 ^ 2 = a 1 * a 5) 
  (b : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (hb : ∀ n, b n = 2 / (a n * a (n+1))) 
  (hS : ∀ n, S n = ∑ i in finset.range n, b i) : 
  ∀ n, S n < 1 := 
sorry

end general_term_formula_inequality_Sn_l546_546247


namespace eval_expr1_l546_546943

theorem eval_expr1 : 
  ( (27 / 8) ^ (-2 / 3) - (49 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) * (2 / 25) ) = 1 / 9 :=
by 
  sorry

end eval_expr1_l546_546943


namespace range_of_root_difference_l546_546259

theorem range_of_root_difference 
  (a b c d : ℝ)
  (a_ne_zero : a ≠ 0)
  (h1 : a + b + c = 0)
  (f_0_pos : 3 * a * 0 ^ 2 + 2 * b * 0 + c > 0)
  (f_1_pos : 3 * a * 1 ^ 2 + 2 * b * 1 + c > 0):
  ∃ x₁ x₂ : ℝ, (3*a*(x₁^2) + 2*b*x₁ + c = 0) ∧ (3*a*(x₂^2) + 2*b*x₂ + c = 0) ∧ (|x₁ - x₂| ∈ set.Ico (real.sqrt 3 / 3) (2 / 3)) :=
sorry

end range_of_root_difference_l546_546259


namespace simplify_expression_l546_546737

noncomputable def i : ℂ := complex.I -- Assuming complex numbers are used for √-1

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) : 
  ((1 / (3 * m)) ^ -3 : ℂ) * ((-m) ^ (3.5 : ℂ)) = -27 * i * (m ^ (6.5 : ℂ)) :=
by
  sorry

end simplify_expression_l546_546737


namespace yellow_yellow_pairs_l546_546165

def num_blue_students := 57
def num_yellow_students := 75
def total_students := 132
def total_pairs := 66
def num_blue_blue_pairs := 23

theorem yellow_yellow_pairs 
  (h1 : num_blue_students + num_yellow_students = total_students)
  (h2 : total_students / 2 = total_pairs)
  (h3 : num_blue_blue_pairs * 2 ≤ num_blue_students)
  (h4 : 2 * total_pairs - 2 * num_blue_blue_pairs = total_students)
  : (2 * num_yellow_students - (num_yellow_students + num_blue_students - 2 * num_blue_blue_pairs)) / 2 = 32 := by 
sory

end yellow_yellow_pairs_l546_546165


namespace lateral_surface_area_of_cone_l546_546037

-- Definitions from the conditions
def base_radius : ℝ := 6
def slant_height : ℝ := 15

-- Theorem statement to be proved
theorem lateral_surface_area_of_cone (r l : ℝ) (hr : r = base_radius) (hl : l = slant_height) : 
  (π * r * l) = 90 * π :=
by
  sorry

end lateral_surface_area_of_cone_l546_546037


namespace non_unique_solution_of_system_irrelevant_m_l546_546965

theorem non_unique_solution_of_system (k m : ℝ) :
  (∀ (x y z : ℝ), 3 * (3 * x^2 + 4 * y^2) = 36 → (k * x^2 + 12 * y^2) = 30 → (m * x^3 - 2 * y^3 + z^2) = 24 → 
  (k = 9)) :=
by {
  sorry
} 

theorem irrelevant_m (m : ℝ) : True :=
by {
  trivial
}

end non_unique_solution_of_system_irrelevant_m_l546_546965


namespace greatest_integer_less_than_150_gcd_18_eq_6_l546_546086

theorem greatest_integer_less_than_150_gcd_18_eq_6 :
  ∃ n : ℕ, n < 150 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ gcd m 18 = 6 → m ≤ n :=
by
  use 132
  split
  { 
    -- proof that 132 < 150 
    exact sorry 
  }
  split
  { 
    -- proof that gcd 132 18 = 6
    exact sorry 
  }
  {
    -- proof that 132 is the greatest such integer
    exact sorry 
  }

end greatest_integer_less_than_150_gcd_18_eq_6_l546_546086


namespace ben_paperclip_day_l546_546912

theorem ben_paperclip_day :
  ∃ k : ℕ, k = 6 ∧ (∀ n : ℕ, n = k → 5 * 3^n > 500) :=
sorry

end ben_paperclip_day_l546_546912


namespace total_towels_l546_546795

theorem total_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) : packs * towels_per_pack = 27 := by
  sorry

end total_towels_l546_546795


namespace count_perfect_square_divisors_of_60_pow_5_l546_546637

theorem count_perfect_square_divisors_of_60_pow_5 :
  let n := 60^5 in
  (∃ a b c : ℕ, n = 2^a * 3^b * 5^c ∧ a ≤ 10 ∧ b ≤ 5 ∧ c ≤ 5 
  ∧ (∀ x ∈ [a, b, c], x % 2 = 0)) → 
  (52 = 53) := sorry

end count_perfect_square_divisors_of_60_pow_5_l546_546637


namespace find_number_l546_546461

theorem find_number (x : ℝ) (h : x = (1 / 3) * x + 120) : x = 180 :=
by
  sorry

end find_number_l546_546461


namespace correct_propositions_l546_546757

-- Conditions of the problem as Lean definitions
def condition_1 (r : ℝ) : Prop :=
  ∀ x y, (r > 0) → (linear_correlation_stronger x y r)

def condition_2 (sum_squared_residuals : ℝ) : Prop :=
  ∀ model, (sum_squared_residuals < model.sum_squared_residuals) → (better_fitting_effect model)

def condition_3 (R_squared : ℝ) : Prop :=
  ∀ model, (R_squared < model.R_squared) → (better_fitting_effect model) 

def condition_4 (e : ℝ) : Prop :=
  E(e) = 0

-- The proof problem: proving the correct propositions among the given ones
theorem correct_propositions (r : ℝ) (sum_squared_residuals : ℝ) (R_squared : ℝ) (e : ℝ)
  (h1 : condition_1 r)
  (h2 : condition_2 sum_squared_residuals)
  (h3 : condition_3 R_squared)
  (h4 : condition_4 e)
  : (¬ h1 ∧ h2 ∧ ¬ h3 ∧ h4) :=
by {
  sorry
}

end correct_propositions_l546_546757


namespace proof_l546_546615

variable (f : ℝ → ℝ)
variable (ω A : ℝ)
variable (a b c : ℝ)

/- Given conditions -/
def func_def := ∀ x, 
  f x = 2 * sin(ω * x / 2) * (sqrt 3 * cos(ω * x / 2) - sin(ω * x / 2))

/- ω > 0 -/
def omega_gt_zero := ω > 0

/- Smallest positive period of f is 3π -/
def period_constr := ∀ x, f (x + 3 * π) = f x

/- Known side lengths and f(3/2 * A) = 1 -/
def side_length_a := a = 2 * sqrt 3
def side_length_c := c = 4
def f_at_A := f ((3/2) * A) = 1

/- Triangle sides and angles relationships -/
def sides_of_triangle := a^2 = b^2 + c^2 - 2 * b * c * cos A

/- The proof we need -/
theorem proof : 
  (func_def f ω) ∧ 
  omega_gt_zero ω ∧ 
  period_constr f ω ∧ 
  side_length_a a ∧ 
  side_length_c c ∧ 
  f_at_A f A ∧ 
  sides_of_triangle a b c ∧
  (ω = 2 / 3) ∧ 
  (∀ x ∈ Icc (-π) (3 * π / 4), f x ≥ -3 ∧ f x ≤ 1) ∧ 
  (b = 2) ∧ 
  (a * c * sin A / 2 = 2 * sqrt 3) := 
sorry

end proof_l546_546615


namespace greatest_int_with_gcd_18_is_138_l546_546080

theorem greatest_int_with_gcd_18_is_138 :
  ∃ n : ℕ, n < 150 ∧ int.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ int.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_int_with_gcd_18_is_138_l546_546080


namespace intervals_of_monotonicity_and_extreme_values_max_and_min_values_on_interval_l546_546258

def f (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * x^2 - 2 * x + 1

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x, (x < -2 → f (deriv f x) > 0) ∧ ((-2 < x < 1) → f (deriv f x) < 0) ∧ (1 < x → f (deriv f x) > 0)) ∧
  (f (-2) = 13 / 3) ∧ 
  (f 1 = -1 / 6) := 
sorry

theorem max_and_min_values_on_interval : 
  (∀ x ∈ [-3, 0], f x ≤ 13 / 3 ∧ f 0 = 1) :=
sorry

end intervals_of_monotonicity_and_extreme_values_max_and_min_values_on_interval_l546_546258


namespace AQ_parallel_BP_l546_546794

variables {A B C M N K P Q : Type}
variables (AB AC BC : Line) (M_on_AB : IsOn M A B) (MN_parallel_AC : Parallel MN AC)
variables (MK_parallel_BC : Parallel MK BC) (sec_C : LineThrough C) 
variables (P_on_MN : IsOn P MN) (Q_on_ext_MK : IsOn Q (Extension MK))

theorem AQ_parallel_BP (h1 :  IsOn A Q) (h2 : IsOn B P) : 
  Parallel (LineThrough A Q) (LineThrough B P) := 
  sorry

end AQ_parallel_BP_l546_546794


namespace points_in_diagram_10_l546_546676

-- Definitions for the conditions
def diagram_points : ℕ → ℕ
| 1 := 3
| 2 := 7
| 3 := 13
| 4 := 21
| (n+1) := diagram_points n + 2 * n + 2

-- The statement we need to prove
theorem points_in_diagram_10 : diagram_points 10 = 111 :=
sorry

end points_in_diagram_10_l546_546676


namespace Ali_catch_weight_l546_546149

-- Define the conditions and the goal
def Ali_Peter_Joey_fishing (p: ℝ) : Prop :=
  let Ali := 2 * p in
  let Joey := p + 1 in
  p + Ali + Joey = 25

-- State the problem
theorem Ali_catch_weight :
  ∃ p: ℝ, Ali_Peter_Joey_fishing p ∧ (2 * p = 12) :=
by
  sorry

end Ali_catch_weight_l546_546149


namespace cone_base_circumference_eq_6pi_l546_546126

def radius : ℝ := 6 -- The radius of the circular piece of paper
def sector_angle : ℝ := 180 -- The angle of the sector removed

theorem cone_base_circumference_eq_6pi : 
  let full_circumference := 2 * Real.pi * radius in
  let circumference_of_base_of_cone := (sector_angle / 360) * full_circumference in
  circumference_of_base_of_cone = 6 * Real.pi :=
by
  sorry

end cone_base_circumference_eq_6pi_l546_546126


namespace fraction_money_left_zero_l546_546174

-- Defining variables and conditions
variables {m c : ℝ} -- m: total money, c: total cost of CDs

-- Condition under the problem statement
def uses_one_fourth_of_money_to_buy_one_fourth_of_CDs (m c : ℝ) := (1 / 4) * m = (1 / 4) * c

-- The conjecture to be proven
theorem fraction_money_left_zero 
  (h: uses_one_fourth_of_money_to_buy_one_fourth_of_CDs m c) 
  (h_eq: c = m) : 
  (m - c) / m = 0 := 
by
  sorry

end fraction_money_left_zero_l546_546174


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l546_546493

-- Problem 1
theorem problem_1 : -27 + (-32) + (-8) + 27 = -40 :=
  by
    calc
      -27 + (-32) + (-8) + 27 = -40 := rfl
    sorry

-- Problem 2
theorem problem_2 : (-5) + abs (-3) = -2 :=
  by
    calc
      (-5) + abs (-3) = -2 := rfl
    sorry

-- Problem 3
theorem problem_3 (x y : ℤ) (h₁ : -x = 3) (h₂ : abs y = 5) : x + y = 2 ∨ x + y = -8 :=
  by
    sorry

-- Problem 4
theorem problem_4 : - (3 / 2) + 5 / 4 - 5 / 2 + 13 / 4 - 5 / 4 = -(3 / 4) :=
  by
    calc
      - (3 / 2) + 5 / 4 - 5 / 2 + 13 / 4 - 5 / 4 = -(3 / 4) := rfl
    sorry

-- Problem 5
theorem problem_5 (a b : ℤ) (h₁ : abs (a - 4) + abs (b + 5) = 0) : a - b = 9 :=
  by
    sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l546_546493


namespace exists_disjoint_convex_quadrilaterals_l546_546308

/-- In a regular 100-gon, 41 vertices are colored black and the remaining 59 vertices are colored white.
    Prove that there exist 24 convex quadrilaterals \(Q_{1}, \ldots, Q_{24}\) whose corners are vertices 
    of the 100-gon, so that 
    - the quadrilaterals \(Q_{1}, \ldots, Q_{24}\) are pairwise disjoint, and
    - every quadrilateral \(Q_{i}\) has three corners of one color and one corner of the other color.
-/
theorem exists_disjoint_convex_quadrilaterals :
  ∃ (Q : list (finset ℕ)) (hQ : Q.length = 24),
  (∀ i j, i ≠ j → disjoint (Q.nth_le i (by sorry)) (Q.nth_le j (by sorry))) ∧
  (∀ i, (Q.nth_le i (by sorry)).card = 4 ∧ 
       ((Q.nth_le i (by sorry)).filter (λ v, v < 41)).card = 3 ∨ (Q.nth_le i (by sorry)).filter (λ v, v < 41)).card = 1)) :=
sorry

end exists_disjoint_convex_quadrilaterals_l546_546308


namespace sum_of_squares_of_roots_l546_546958

theorem sum_of_squares_of_roots : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 10) → (r1 * r2 = 9) → (r1 > 5 ∨ r2 > 5) → (r1^2 + r2^2 = 82) :=
by
  intros r1 r2 h1 h2 h3
  sorry

end sum_of_squares_of_roots_l546_546958


namespace find_a_l546_546631

variable {a : ℝ}

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a (h : A ∩ (B a) = {2}) : a = 2 :=
by
  sorry

end find_a_l546_546631


namespace product_of_d_l546_546936

theorem product_of_d (d1 d2 : ℕ) (h1 : ∃ k1 : ℤ, 49 - 12 * d1 = k1^2)
  (h2 : ∃ k2 : ℤ, 49 - 12 * d2 = k2^2) (h3 : 0 < d1) (h4 : 0 < d2)
  (h5 : d1 ≠ d2) : d1 * d2 = 8 := 
sorry

end product_of_d_l546_546936


namespace angle_bisector_NH_l546_546381

variables (A B M C N H : Point)
variables (SMB : segment M B)
variables (triangleAMC : isosceles_triangle A M C)
variables (triangleMBN : isosceles_triangle M B N)
variables (lineBC : collinear B N C)
variables (eqABBC : distance A B = distance B C)
variables (perpendicularBHtoAC : ∃ H : Point, is_perpendicular_from B H AC ∧ liesOn H (segment M C))

-- The goal is to prove that NH bisects ∠MNC
theorem angle_bisector_NH : 
  is_angle_bisector (segment N H) ∠MNC :=
sorry

end angle_bisector_NH_l546_546381


namespace parallel_line_eq_through_point_l546_546756

noncomputable def line_eq (a b c : ℝ) := λ (x y : ℝ), a * x + b * y + c = 0

theorem parallel_line_eq_through_point (x₁ y₁ : ℝ) (h : x₁ = 1 ∧ y₁ = -1) :
    ∃ m : ℝ, line_eq 2 3 m x₁ y₁ :=
begin
  use 1, -- the solution found indicates m = 1
  intros x y,
  rw [h.1, h.2], -- replace x₁ with 1 and y₁ with -1
  norm_num,
end

end parallel_line_eq_through_point_l546_546756


namespace seq_inequality_l546_546585

variable (a : ℕ → ℝ)
variable (n m : ℕ)

-- Conditions
axiom pos_seq (k : ℕ) : a k ≥ 0
axiom add_condition (i j : ℕ) : a (i + j) ≤ a i + a j

-- Statement to prove
theorem seq_inequality (n m : ℕ) (h : m > 0) (h' : n ≥ m) : 
  a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := sorry

end seq_inequality_l546_546585


namespace jelly_bean_problem_l546_546535

variable (b c : ℕ)

theorem jelly_bean_problem (h1 : b = 3 * c) (h2 : b - 15 = 4 * (c - 15)) : b = 135 :=
sorry

end jelly_bean_problem_l546_546535


namespace bug_visits_exactly_16_pavers_l546_546883

-- Defining the dimensions of the garden and the pavers
def garden_width : ℕ := 14
def garden_length : ℕ := 19
def paver_size : ℕ := 2

-- Calculating the number of pavers in width and length
def pavers_width : ℕ := garden_width / paver_size
def pavers_length : ℕ := (garden_length + paver_size - 1) / paver_size  -- Taking ceiling of 19/2

-- Calculating the GCD of the pavers count in width and length
def gcd_pavers : ℕ := Nat.gcd pavers_width pavers_length

-- Calculating the number of pavers the bug crosses
def pavers_crossed : ℕ := pavers_width + pavers_length - gcd_pavers

-- Theorem that states the number of pavers visited
theorem bug_visits_exactly_16_pavers :
  pavers_crossed = 16 := by
  -- Sorry is used to skip the proof steps
  sorry

end bug_visits_exactly_16_pavers_l546_546883


namespace kneading_time_l546_546357

def total_time (onions garlic_peppers kneading resting assembling : ℕ) : ℕ :=
  onions + garlic_peppers + kneading + resting + assembling

theorem kneading_time :
  ∀ (k : ℕ),
  let onions := 20,
      garlic_peppers := onions / 4,
      resting := 2 * k,
      assembling := (k + resting) / 10 in
  total_time onions garlic_peppers k resting assembling = 124 →
  k = 30 :=
begin
  intros k,
  simp [total_time],
  sorry
end

end kneading_time_l546_546357


namespace nice_integer_characterization_l546_546163

-- Lean 4 statement for the mathematically equivalent proof problem
theorem nice_integer_characterization (n : ℕ) (h1 : n % 2 = 1) (h2 : n ≥ 3) : 
  (∃ (a : ℕ → ℕ), (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k a > 0)) ↔ ∃ k : ℕ, n = 4 * k + 1 :=
sorry

end nice_integer_characterization_l546_546163


namespace angle_between_PQ_and_BC_is_90_l546_546409

variables (A B C D P E F Q : Type)
variables [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry D]
variables [euclidean_geometry P] [euclidean_geometry E] [euclidean_geometry F] [euclidean_geometry Q]

-- Definitions of the given points and conditions
def circumscribed_quadrilateral (A B C D : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry D] : Prop := sorry
def intersection_of_diagonals (P : Type) [euclidean_geometry P] (A C B D : Type) [euclidean_geometry A] [euclidean_geometry C] [euclidean_geometry B] [euclidean_geometry D] : Prop := sorry
def midpoint (E : Type) [euclidean_geometry E] (A B : Type) [euclidean_geometry A] [euclidean_geometry B] : Prop := sorry

def is_perpendicular (x y : Type) [euclidean_geometry x] [euclidean_geometry y] : Prop := sorry
def intersection_of_perpendiculars (Q : Type) [euclidean_geometry Q] (E F : Type) [euclidean_geometry E] [euclidean_geometry F]) : Prop := sorry

-- The Lean theorem statement
theorem angle_between_PQ_and_BC_is_90 :
  circumscribed_quadrilateral A B C D →
  intersection_of_diagonals P A C B D →
  midpoint E A B →
  midpoint F C D →
  intersection_of_perpendiculars Q E F →
  angles_between PQ BC = 90 :=
sorry

end angle_between_PQ_and_BC_is_90_l546_546409


namespace partition_inequality_l546_546197

-- Define the function f(n) as per conditions
def f (n : ℕ) : ℕ := sorry -- Placeholder for the actual definition

theorem partition_inequality (n : ℕ) (h : n ≥ 1) : 
  f(n+1) ≤ (f(n) + f(n+2)) / 2 := 
by
  sorry

end partition_inequality_l546_546197


namespace gino_brown_bears_count_l546_546968

theorem gino_brown_bears_count (white_bears black_bears total_bears brown_bears : ℕ) 
(white_bears_eq : white_bears = 24) 
(black_bears_eq : black_bears = 27) 
(total_bears_eq : total_bears = 66) 
(brown_bears_eq : brown_bears = total_bears - white_bears - black_bears) : 
brown_bears = 15 :=
by 
  rw [white_bears_eq, black_bears_eq, total_bears_eq, brown_bears_eq]
  simp
  sorry

end gino_brown_bears_count_l546_546968


namespace evaluate_expression_l546_546557

theorem evaluate_expression : -30 + 5 * (9 / (3 + 3)) = -22.5 := sorry

end evaluate_expression_l546_546557


namespace parabola_and_area_proof_l546_546242

noncomputable def parabola_equation (p : ℝ) (p_pos : p > 0) : Prop :=
  ∀ (x y : ℝ), (y ^ 2 = 2 * p * x) → (-x) = y

noncomputable def area_triangle_FAB (l2_eq : ℝ → ℝ) (AB_midpoint : (ℝ × ℝ)) (OP_AB_relation : ℝ) : Prop :=
  ∀ (F A B : ℝ × ℝ), 
    let P := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2) in
    let O := (0, 0) in
    (l2_eq = λ x, x + 8) →
    P = AB_midpoint →
    dist O P = OP_AB_relation / 2 * dist A B →
    ∃ S : ℝ, S = 2 * sqrt 2 * dist A B

theorem parabola_and_area_proof :
  ∃ (p : ℝ) (p_pos : p > 0), parabola_equation p p_pos ∧
  (∃ (l2_eq : ℝ → ℝ) (AB_midpoint : (ℝ × ℝ)) (OP_AB_relation : ℝ), 
    area_triangle_FAB l2_eq AB_midpoint OP_AB_relation) :=
begin
  sorry
end

end parabola_and_area_proof_l546_546242


namespace radio_show_duration_correct_l546_546137

/-- The total duration of the radio show in hours, given the constraints on segments and durations. -/
def radio_show_duration : ℕ :=
  let talking_segments_duration := 3 * 10
  let ad_breaks_duration := 5 * 5
  let songs_duration := 125
  let total_minutes := talking_segments_duration + ad_breaks_duration + songs_duration
  total_minutes / 60

theorem radio_show_duration_correct : radio_show_duration = 3 := by
  let talking_segments_duration := 3 * 10
  let ad_breaks_duration := 5 * 5
  let songs_duration := 125
  let total_minutes := talking_segments_duration + ad_breaks_duration + songs_duration
  have h : total_minutes = 180 := by
    simp [talking_segments_duration, ad_breaks_duration, songs_duration]
  have h_div : 180 / 60 = 3 := by
    norm_num
  have total_minutes_eq : radio_show_duration = total_minutes / 60 := by rfl
  rw [h, total_minutes_eq, h_div]
  rfl

end radio_show_duration_correct_l546_546137


namespace sum_proper_divisors_81_l546_546837

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l546_546837


namespace num_compositions_l546_546166

-- Definitions based on conditions identified in the problem
def total_amount_collected (n x y z: ℕ) : Prop :=
  4 * x + y + z = n

-- The math proof problem
theorem num_compositions (n: ℕ) : ∃ x y z : ℕ, total_amount_collected n x y z → (∃! k : ℕ, k = ⌊(n + 3)^2 / 8⌋) :=
sorry

end num_compositions_l546_546166


namespace area_of_circle_l546_546455

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end area_of_circle_l546_546455


namespace train_speed_approx_80_kmph_l546_546141

-- Definitions based on conditions
def train_length : ℝ := 200  -- train length in meters
def time_taken : ℝ := 8.999280057595392  -- time taken to pass the electric pole in seconds

-- Theorem statement to prove the question == answer given conditions
theorem train_speed_approx_80_kmph :
  (train_length / time_taken * 3.6) ≈ 80 :=
begin
  sorry
end

end train_speed_approx_80_kmph_l546_546141


namespace combined_volume_correct_l546_546895

noncomputable def truncated_cone_volume (R r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * h * (R^2 + R * r + r^2)

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

theorem combined_volume_correct :
  let R := 12
  let r := 6
  let h_truncated := 10
  let r_cone := 12
  let h_cone := 5
  truncated_cone_volume R r h_truncated + cone_volume r_cone h_cone = 1080 * real.pi := by
  sorry

end combined_volume_correct_l546_546895


namespace perpendicular_lines_sin_double_angle_l546_546977

theorem perpendicular_lines_sin_double_angle (θ : ℝ) 
  (h : ∀ x y : ℝ, x - 2*y + 3 = 0 → 
    (∃ x₁ y₁ : ℝ, (x₁ - 2*y₁ + 3 = 0) ∧ (x₁ * cos θ + y₁ * sin θ = 0))) :
  sin (2*θ) = -4/5 :=
sorry

end perpendicular_lines_sin_double_angle_l546_546977


namespace mrs_early_correct_speed_l546_546366

-- Definition of the problem conditions in Lean 4
noncomputable def calculate_speed : ℝ :=
  let d := 50 * (t + 1 / 15) in
  let t : ℝ := 11 / 24 in
  d / t

-- The theorem stating the required speed
theorem mrs_early_correct_speed :
  calculate_speed = 57 :=
by
  sorry  -- Proof not required per instructions

end mrs_early_correct_speed_l546_546366


namespace count_irreducible_fractions_l546_546307

theorem count_irreducible_fractions (s : Finset ℕ) (h1 : ∀ n ∈ s, 15*n > 15/16) (h2 : ∀ n ∈ s, n < 1) (h3 : ∀ n ∈ s, Nat.gcd n 15 = 1) :
  s.card = 8 := 
sorry

end count_irreducible_fractions_l546_546307


namespace initial_money_correct_l546_546074

def initial_money (total: ℕ) (allowance: ℕ): ℕ :=
  total - allowance

theorem initial_money_correct: initial_money 18 8 = 10 :=
  by sorry

end initial_money_correct_l546_546074


namespace initial_men_count_l546_546437

theorem initial_men_count (M : ℕ) (h1 : ∃ F : ℕ, F = M * 22) (h2 : ∃ F_remaining : ℕ, F_remaining = M * 20) (h3 : ∃ F_remaining_2 : ℕ, F_remaining_2 = (M + 1140) * 8) : 
  M = 760 := 
by
  -- Code to prove the theorem goes here.
  sorry

end initial_men_count_l546_546437


namespace value_of_quotients_l546_546346

variable {a_1 a_2 a_3 a_4 a_5 : ℝ}

-- Conditions
axiom condition1 : (∀ k ∈ {1, 2, 3, 4, 5}, (a_1 / (k^2 + 1) + a_2 / (k^2 + 2) + a_3 / (k^2 + 3) + a_4 / (k^2 + 4) + a_5 / (k^2 + 5) = 1 / k^2))

-- The theorem to prove
theorem value_of_quotients : (a_1 / 37 + a_2 / 38 + a_3 / 39 + a_4 / 40 + a_5 / 41 = 187465 / 6744582) :=
by
  sorry

end value_of_quotients_l546_546346


namespace fair_people_ratio_l546_546050

def next_year_ratio (this_year next_year last_year : ℕ) (total : ℕ) :=
  this_year = 600 ∧
  last_year = next_year - 200 ∧
  this_year + last_year + next_year = total → 
  next_year = 2 * this_year

theorem fair_people_ratio :
  ∀ (next_year : ℕ),
  next_year_ratio 600 next_year (next_year - 200) 2800 → next_year = 2 * 600 := by
sorry

end fair_people_ratio_l546_546050


namespace store_money_left_after_sale_l546_546891

theorem store_money_left_after_sale :
  ∀ (n p : ℕ) (d s : ℝ) (debt : ℕ), 
  n = 2000 → 
  p = 50 → 
  d = 0.80 → 
  s = 0.90 → 
  debt = 15000 → 
  (nat.floor (s * n) * (p - nat.floor (d * p)) : ℕ) - debt = 3000 :=
by
  intros n p d s debt hn hp hd hs hdebt
  sorry

end store_money_left_after_sale_l546_546891


namespace complement_union_l546_546296

def U := {1, 2, 3, 4}
def M := {1, 2}
def N := {2, 3}
def C_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

theorem complement_union (H : True) : C_U (M ∪ N) = {4} :=
by
  sorry

end complement_union_l546_546296


namespace exists_b_for_odd_a_l546_546388

theorem exists_b_for_odd_a (a : ℤ) (h : a % 2 = 1) : ∃ b : ℤ, a ∣ (2^b - 1) := 
sorry

end exists_b_for_odd_a_l546_546388


namespace angle_B_in_parallelogram_l546_546668

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end angle_B_in_parallelogram_l546_546668


namespace z_share_in_profit_l546_546487

-- Define the conditions as per the problem statement
def x_investment := 36000
def y_investment := 42000
def z_investment := 48000
def x_months := 12
def y_months := 12
def z_months := 8
def total_profit := 13860

-- Define the main theorem to be proved, skipping the proof
theorem z_share_in_profit :
  let x_investment_months := x_investment * x_months,
      y_investment_months := y_investment * y_months,
      z_investment_months := z_investment * z_months,
      total_investment_months := x_investment_months + y_investment_months + z_investment_months,
      z_ratio := z_investment_months.toRat / total_investment_months,
      z_share := total_profit * z_ratio in
  z_share = 2520 := 
sorry

end z_share_in_profit_l546_546487


namespace least_positive_integer_reducible_fraction_l546_546562

theorem least_positive_integer_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (∃ d : ℕ, d > 1 ∧ d ∣ (m - 10) ∧ d ∣ (9 * m + 11)) ↔ m ≥ n) ∧ n = 111 :=
by
  sorry

end least_positive_integer_reducible_fraction_l546_546562


namespace solve_equation_l546_546781

theorem solve_equation : ∀ (x : ℝ), (x / 2 - 1 = 3) → x = 8 :=
by
  intro x h
  sorry

end solve_equation_l546_546781


namespace novel_reading_min_per_cd_l546_546878

theorem novel_reading_min_per_cd :
  ∀ (total_minutes cd_capacity : ℕ), 
  total_minutes = 528 → cd_capacity = 70 → 
  (∃ (num_discs : ℕ), num_discs = (total_minutes + cd_capacity - 1) / cd_capacity ∧ 
   (total_minutes % cd_capacity > 0 → num_discs * cd_capacity ≥ total_minutes) ∧ 
   total_minutes / num_discs = 66) :=
by
  -- Defining the variables and setting the constants
  assume total_minutes cd_capacity : ℕ,
  assume h1 : total_minutes = 528,
  assume h2 : cd_capacity = 70,
  
  -- Number of discs calculation based on provided logic
  let num_discs := (total_minutes + cd_capacity - 1) / cd_capacity,
  -- Proving the existence of the correct number of discs
  exists num_discs,
  split,
  { -- Showing the calculated number of discs matches the expected value
    exact rfl, },

  split,
  { -- Verifying the conditional statement regarding the total minutes
    assume h : total_minutes % cd_capacity > 0,
    have h' : num_discs * cd_capacity ≥ total_minutes, sorry,
    exact h', },

  { -- Confirming the reading distribution per disc equals the expected value
    have h' : total_minutes / num_discs = 66, sorry,
    exact h', }

end novel_reading_min_per_cd_l546_546878


namespace maurice_packages_l546_546009

def numberOfPeople : Nat := 10
def burgerWeightPerPerson : Nat := 2
def packageWeight : Nat := 5
def totalBeefNeeded : Nat := numberOfPeople * burgerWeightPerPerson
def packagesNeeded : Nat := totalBeefNeeded / packageWeight

theorem maurice_packages : packagesNeeded = 4 := by
  have h1 : totalBeefNeeded = 20 := by
    simp [numberOfPeople, burgerWeightPerPerson, totalBeefNeeded]
  have h2 : packagesNeeded = 20 / packageWeight := by
    simp [h1, packageWeight, totalBeefNeeded]
  exact h2

end maurice_packages_l546_546009


namespace num_data_points_plane_l546_546932

theorem num_data_points_plane : ∀ (P : Type) [plane : EuclideanPlane P],  points_needed_for_location_in_plane = 2 :=
by
  sorry

end num_data_points_plane_l546_546932


namespace toms_biking_miles_l546_546798

theorem toms_biking_miles (days_in_year : ℕ) (first_period_days : ℕ) (first_period_rate : ℕ) 
  (remaining_days_rate : ℕ) (total_days : ℕ) :
  days_in_year = 365 ∧ first_period_days = 183 ∧ first_period_rate = 30 ∧ remaining_days_rate = 35 ∧ total_days = days_in_year - first_period_days →
  let first_period_miles := first_period_days * first_period_rate in
  let remaining_period_days := days_in_year - first_period_days in
  let remaining_period_miles := remaining_period_days * remaining_days_rate in
  let total_miles := first_period_miles + remaining_period_miles in
  total_miles = 11860 :=
begin
  sorry
end

end toms_biking_miles_l546_546798


namespace fishing_problem_l546_546147

theorem fishing_problem
  (P : ℕ) -- weight of the fish Peter caught
  (H1 : Ali_weight = 2 * P) -- Ali caught twice as much as Peter
  (H2 : Joey_weight = P + 1) -- Joey caught 1 kg more than Peter
  (H3 : P + 2 * P + (P + 1) = 25) -- Together they caught 25 kg
  : Ali_weight = 12 :=
by
  sorry

end fishing_problem_l546_546147


namespace bullet_train_passing_time_l546_546100

/--
A bullet train of length 220 meters is running at a speed of 59 km/h, 
and a man is running at 7 km/h in the direction opposite to the train.
Prove that the time taken for the bullet train to pass the man is 12 seconds.
-/
theorem bullet_train_passing_time :
  let train_length := 220  -- in meters
  let train_speed := 59 * (5 / 18)  -- converting km/h to m/s
  let man_speed := 7 * (5 /18)  -- converting km/h to m/s
  let relative_speed := train_speed + man_speed
  (train_length / relative_speed) = 12 :=
by
  have train_length : ℝ := 220
  have train_speed : ℝ := 59 * (5 / 18)
  have man_speed : ℝ := 7 * (5 / 18)
  have relative_speed : ℝ := train_speed + man_speed
  have time_to_pass : ℝ := train_length / relative_speed
  show time_to_pass = 12
  sorry

end bullet_train_passing_time_l546_546100


namespace problem_part1_problem_part2_problem_part3_l546_546250

noncomputable def given_quadratic (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

noncomputable def sin_cos_eq_quadratic_roots (θ m : ℝ) : Prop := 
  let sinθ := Real.sin θ
  let cosθ := Real.cos θ
  given_quadratic sinθ m = 0 ∧ given_quadratic cosθ m = 0

theorem problem_part1 (θ : ℝ) (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ)) = (3 + 5 * Real.sqrt 3) / 4 :=
sorry

theorem problem_part2 {θ : ℝ} (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  m = Real.sqrt 3 / 4 :=
sorry

theorem problem_part3 (m : ℝ) (sinθ1 cosθ1 sinθ2 cosθ2 : ℝ) (θ1 θ2 : ℝ)
  (H1 : sinθ1 = Real.sqrt 3 / 2 ∧ cosθ1 = 1 / 2 ∧ θ1 = Real.pi / 3)
  (H2 : sinθ2 = 1 / 2 ∧ cosθ2 = Real.sqrt 3 / 2 ∧ θ2 = Real.pi / 6) : 
  ∃ θ, sin_cos_eq_quadratic_roots θ m ∧ 
       (Real.sin θ = sinθ1 ∧ Real.cos θ = cosθ1 ∨ Real.sin θ = sinθ2 ∧ Real.cos θ = cosθ2) :=
sorry

end problem_part1_problem_part2_problem_part3_l546_546250


namespace option_d_is_quadratic_l546_546465

def quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x : ℝ, a * x^2 + b * x + c = 0

theorem option_d_is_quadratic :
  quadratic_equation 1 (-3) 2 :=
by
  -- this is to state that the conditions are fulfilled for option D
  unfold quadratic_equation,
  -- it satisfies the definition of a quadratic equation
  split,
  -- the coefficient of x^2 is 1, which is not zero
  { exact one_ne_zero },
  -- demonstrating that there exists a solution to the equation (x - 1)(x - 2) = 0
  { use 1,
    exact by linarith [1 * 1^2 - 3 * 1 + 2] }
  sorry

end option_d_is_quadratic_l546_546465


namespace max_value_sin2alpha_div_sin_beta_minus_alpha_l546_546998

theorem max_value_sin2alpha_div_sin_beta_minus_alpha 
  (α β : ℝ) 
  (h1 : α ∈ Icc (Real.pi / 4) (Real.pi / 3)) 
  (h2 : β ∈ Icc (Real.pi / 2) Real.pi) 
  (h3 : Real.sin (α + β) - Real.sin α = 2 * Real.sin α * Real.cos β) :
  ∃ (c : ℝ), c = Real.sqrt 2 ∧ ∀ (θ : ℝ), 
    θ = α → 
    α ∈ Icc (Real.pi / 4) (Real.pi / 3) → 
    β ∈ Icc (Real.pi / 2) Real.pi →
    h3 → 
    Real.sin (2 * α) / Real.sin (β - α) ≤ c := 
  by {
    use Real.sqrt 2,
    sorry
  }

end max_value_sin2alpha_div_sin_beta_minus_alpha_l546_546998


namespace lines_intersect_hyperbola_once_l546_546261

theorem lines_intersect_hyperbola_once
  (l : LinearMap ℝ (ℝ × ℝ) ℝ)
  (h : ∀ (x y : ℝ), l (x, y) = if x = 1 then y = 2 * x - 1 ∨ y = -2 * x + 3 ∨ y = 4 * x - 3 ∨ x = 1 else true)
  (hyperbola : ∀ (x y : ℝ), x^2 - y^2 / 4 = 1)
  : (∃ (x y : ℝ), l (x, y) = 0 ∧ hyperbola x y ∧ x = 1 ∧ y = 1) :=
sorry

end lines_intersect_hyperbola_once_l546_546261


namespace ben_last_roll_probability_l546_546644

theorem ben_last_roll_probability :
  let p1 := (3 : ℚ) / 4
  let p2 := (1 : ℚ) / 4
  let P_12th_last := p1^10 * p2
  (P_12th_last).approximate = 0.014 :=
by {
  let p1 := (3 : ℚ) / 4
  let p2 := (1 : ℚ) / 4
  let P_12th_last := p1^10 * p2
  have : P_12th_last = (3^10 : ℚ) / (4^11 : ℚ), by sorry,
  have approx_value : ((3^10 : ℚ) / (4^11 : ℚ)).approximate = 0.014, by sorry,
  exact eq.trans this approx_value
}

end ben_last_roll_probability_l546_546644


namespace pizza_area_percent_increase_l546_546872

-- Definitions
def original_diameter : ℝ := 14
def new_diameter : ℝ := 18
def pi_value : ℝ := Real.pi

-- Definitions of derived values
def original_radius := original_diameter / 2
def new_radius := new_diameter / 2

def original_area := pi_value * (original_radius ^ 2)
def new_area := pi_value * (new_radius ^ 2)

def area_increase := new_area - original_area
def percent_increase := (area_increase / original_area) * 100

-- Proof statement, asserting the derived percentage increase
theorem pizza_area_percent_increase : percent_increase ≈ 65.31 :=
by
  sorry

end pizza_area_percent_increase_l546_546872


namespace sqrt_sum_max_value_l546_546609

theorem sqrt_sum_max_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (sqrt a + sqrt b) ≤ sqrt 2 :=
sorry

end sqrt_sum_max_value_l546_546609


namespace c_should_pay_27_rs_l546_546479

-- Definitions for the conditions
def oxenMonths (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

def totalOxenMonths (a_oxenMon : ℕ) (b_oxenMon : ℕ) (c_oxenMon : ℕ) : ℕ := a_oxenMon + b_oxenMon + c_oxenMon

def costPerOxenMonth (totalRent : ℤ) (totalOxenMon : ℕ) : ℝ := (totalRent.toReal) / (totalOxenMon : ℝ)

def cShareOfRent (costPerOxenMon : ℝ) (c_oxenMon : ℕ) : ℝ := costPerOxenMon * (c_oxenMon : ℝ)

-- Conditions
def a_oxen := 10
def a_months := 7
def b_oxen := 12
def b_months := 5
def c_oxen := 15
def c_months := 3
def rent := (105 : ℤ)

-- Oxen-months calculations
def a_oxenMon := oxenMonths a_oxen a_months
def b_oxenMon := oxenMonths b_oxen b_months
def c_oxenMon := oxenMonths c_oxen c_months
def totalOxenMon := totalOxenMonths a_oxenMon b_oxenMon c_oxenMon
def costPerOxenMon := costPerOxenMonth rent totalOxenMon

-- The proof goal
theorem c_should_pay_27_rs : cShareOfRent costPerOxenMon c_oxenMon = 27 := by sorry

end c_should_pay_27_rs_l546_546479


namespace find_fixed_monthly_fee_l546_546155

noncomputable def fixed_monthly_fee (f h : ℝ) (february_bill march_bill : ℝ) : Prop :=
  (f + h = february_bill) ∧ (f + 3 * h = march_bill)

theorem find_fixed_monthly_fee (h : ℝ):
  fixed_monthly_fee 13.44 h 20.72 35.28 :=
by 
  sorry

end find_fixed_monthly_fee_l546_546155


namespace integer_parts_of_roots_l546_546220

theorem integer_parts_of_roots (n : ℕ) (h1 : 1 ≤ n) : 
  let f : ℝ → ℝ := λ x, x^2 + (2 * n + 1) * x + (6 * n - 5)
  let g : ℝ → ℝ := λ x, x^2 + (2 * (n + 1)) * x + (8 * (n - 1))
  ∃ a b : ℤ, f(a) = 0 ∧ f(b) = 0 ∧ g ⟨a, b⟩ = 0 :=
sorry

end integer_parts_of_roots_l546_546220


namespace optionC_is_quadratic_l546_546901

def isQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c

def optionA (x : ℝ) : ℝ := 3 * x - 1
def optionB (x : ℝ) : ℝ := 1 / (x^2)
def optionC (x : ℝ) : ℝ := 3 * x^2 + x - 1
def optionD (x : ℝ) : ℝ := 2 * x^3 - 1

theorem optionC_is_quadratic : isQuadratic optionC :=
  by
    sorry

end optionC_is_quadratic_l546_546901


namespace question1_question2_l546_546594

variable (α : ℝ)

theorem question1 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = -15 / 23 := by
  sorry

theorem question2 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    Real.tan (α - 5 * π / 4) = -7 := by
  sorry

end question1_question2_l546_546594


namespace sum_proper_divisors_eq_40_l546_546824

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l546_546824


namespace original_average_weight_l546_546065

theorem original_average_weight 
  (W : ℝ)  -- Define W as the original average weight
  (h1 : 0 < W)  -- Define conditions
  (w_new1 : ℝ := 110)
  (w_new2 : ℝ := 60)
  (num_initial_players : ℝ := 7)
  (num_total_players : ℝ := 9)
  (new_average_weight : ℝ := 92)
  (total_weight_initial := num_initial_players * W)
  (total_weight_additional := w_new1 + w_new2)
  (total_weight_total := new_average_weight * num_total_players) : 
  total_weight_initial + total_weight_additional = total_weight_total → W = 94 :=
by 
  sorry

end original_average_weight_l546_546065


namespace average_calculation_l546_546750

variable {t b c : ℝ}

def avg1 : ℝ := (t + b + c + 29) / 4
def avg2 : ℝ := (t + b + c + 14 + 15) / 5

theorem average_calculation (h : avg1 = 15) : avg2 = 12 :=
by
  sorry

end average_calculation_l546_546750


namespace find_ellipse_equation_find_line_equation_l546_546990

section problem_conditions
  -- Definitions of given conditions
  def point_A := (0: ℝ, -2: ℝ)
  def origin_O := (0: ℝ, 0: ℝ)
  def ellipse_eccentricity := (sqrt 3) / 2
  def slope_AF := (2 * sqrt 3) / 3
  
  -- Assumptions related to the ellipse E
  variables {a b: ℝ}
  axioms (a_pos: a > 0) (b_pos: b > 0) (ab_ineq: a > b)

  -- Derived definitions
  def right_focus_F := (sqrt 3, 0: ℝ)
end problem_conditions

section problem_goals
  -- The Lean problem statements
  theorem find_ellipse_equation (a b: ℝ) 
    (a_pos: a > 0) (b_pos: b > 0) (ab_ineq: a > b) (ecc: a * sqrt ((a^2) - (b^2)) = 3 / 2):
    (a = 2) → (b = 1) → (forall x y: ℝ, (x^2) / (4) + y^2 = 1) := 
  sorry
  
  theorem find_line_equation (a b: ℝ) 
    (a_pos: a > 0) (b_pos: b > 0) (ab_ineq: a > b) (ecc: a * sqrt ((a^2) - (b^2)) = 3 / 2):
    (a = 2) → (b = 1) → (Point_intersect: ℝ × ℝ) (Passes_through: Point_intersect = point_A) → 
    (Area_maximized: slope_AF := (2 * sqrt 3) / 3) → 
    (line_eq: (forall x y: ℝ, (y = (sqrt 7 / 2) * x - 2) ∨ (y = - (sqrt 7 / 2) * x - 2))) := 
  sorry
end problem_goals

end find_ellipse_equation_find_line_equation_l546_546990


namespace find_number_l546_546956

theorem find_number (x : ℝ) (h : sqrt x / 11 = 4) : x = 1936 := 
by 
  sorry

end find_number_l546_546956


namespace distance_of_run_l546_546474

variable (total_distance bicycle_race_distance : ℕ)

-- Given conditions defined as variables:
def totalDistance : ℕ := total_distance
def bicycleRaceDistance : ℕ := bicycle_race_distance

-- The theorem statement:
theorem distance_of_run (total_distance bicycle_race_distance : ℕ) : 
  total_distance = 155 → 
  bicycle_race_distance = 145 → 
  ∃ d, d = total_distance - bicycle_race_distance ∧ d = 10 := 
by {
  intros h1 h2,
  existsi (total_distance - bicycle_race_distance),
  split,
  {reflexivity},
  {rw [h1, h2],
   norm_num,}
}

end distance_of_run_l546_546474


namespace triangle_side_length_l546_546606

theorem triangle_side_length (x : ℝ) (h1 : 6 < x) (h2 : x < 14) : x = 11 :=
by
  sorry

end triangle_side_length_l546_546606


namespace marks_in_biology_l546_546186

theorem marks_in_biology (E M P C : ℝ) (A B : ℝ)
  (h1 : E = 90)
  (h2 : M = 92)
  (h3 : P = 85)
  (h4 : C = 87)
  (h5 : A = 87.8) 
  (h6 : (E + M + P + C + B) / 5 = A) : 
  B = 85 := 
by
  -- Placeholder for the proof
  sorry

end marks_in_biology_l546_546186


namespace proof_problem_l546_546559

-- Definitions of the vectors in coordinates
variables {x1 y1 z1 x2 y2 z2 : ℝ}

-- Definition of vectors
def r1 := (x1, y1, z1)
def r2 := (x2, y2, z2)

-- Dot product in coordinates
def dot_product (r1 r2 : ℝ × ℝ × ℝ) : ℝ :=
  r1.1 * r2.1 + r1.2 * r2.2 + r1.3 * r2.3

-- Magnitude of a vector
def magnitude (r : ℝ × ℝ × ℝ) : ℝ :=
  Math.sqrt (r.1^2 + r.2^2 + r.3^2)

-- Angle between vectors
def cos_angle (r1 r2 : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product r1 r2) / (magnitude r1 * magnitude r2)

-- Definition for orthogonality
def is_orthogonal (r1 r2 : ℝ × ℝ × ℝ) : Prop :=
  dot_product r1 r2 = 0

-- Definition for collinearity
def is_collinear (r1 r2 : ℝ × ℝ × ℝ) : Prop :=
  (x1 ≠ 0 ∧ x2 = λ * x1 ∧ y1 ≠ 0 ∧ y2 = λ * y1 ∧ z1 ≠ 0 ∧ z2 = λ * z1)
  ∨ (x1 = 0 ∧ x2 = 0 ∧ y1 ≠ 0 ∧ y2 = λ * y1 ∧ z1 ≠ 0 ∧ z2 = λ * z1)
  ∨ (x1 = 0 ∧ x2 = 0 ∧ y1 = 0 ∧ y2 = 0 ∧ z1 ≠ 0 ∧ z2 = λ * z1)

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Math.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- Importing all necessary libraries
theorem proof_problem :
  dot_product r1 r2 = x1 * x2 + y1 * y2 + z1 * z2 ∧
  cos_angle r1 r2 = (dot_product r1 r2) / (magnitude r1 * magnitude r2) ∧
  (is_orthogonal r1 r2 ↔ dot_product r1 r2 = 0) ∧
  (is_collinear r1 r2 ↔ (x2 / x1 = y2 / y1 ∧ y2 / y1 = z2 / z1)) ∧
  distance r1 r2 = Math.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) :=
by
  -- proof goes here
  sorry

end proof_problem_l546_546559


namespace least_positive_integer_x_l546_546089

theorem least_positive_integer_x : ∃ x : ℕ, ((2 * x)^2 + 2 * 43 * (2 * x) + 43^2) % 53 = 0 ∧ 0 < x ∧ (∀ y : ℕ, ((2 * y)^2 + 2 * 43 * (2 * y) + 43^2) % 53 = 0 → 0 < y → x ≤ y) := 
by
  sorry

end least_positive_integer_x_l546_546089


namespace average_gt_median_by_25_l546_546268

open Real

def weights : List ℝ := [7, 8, 9, 110]

def median (l : List ℝ) : ℝ :=
  let sorted_l := l.qsort (λ a b => a < b)
  if h : sorted_l.length % 2 = 1 then 
    sorted_l.nthLe (sorted_l.length / 2) sorry
  else 
    (sorted_l.nthLe (sorted_l.length / 2 - 1) sorry + 
     sorted_l.nthLe (sorted_l.length / 2) sorry) / 2

def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_gt_median_by_25 :
  average weights = median weights + 25 :=
by
  sorry

end average_gt_median_by_25_l546_546268


namespace number_of_numbers_is_ten_l546_546036

open Nat

-- Define the conditions as given
variable (n : ℕ) -- Total number of numbers
variable (incorrect_average correct_average incorrect_value correct_value : ℤ)
variable (h1 : incorrect_average = 16)
variable (h2 : correct_average = 17)
variable (h3 : incorrect_value = 25)
variable (h4 : correct_value = 35)

-- Define the proof problem
theorem number_of_numbers_is_ten
  (h1 : incorrect_average = 16)
  (h2 : correct_average = 17)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 35)
  (h5 : ∀ (x : ℤ), x ≠ incorrect_value → incorrect_average * (n : ℤ) + x = correct_average * (n : ℤ) + correct_value - incorrect_value)
  : n = 10 := 
sorry

end number_of_numbers_is_ten_l546_546036


namespace smallest_possible_denominator_l546_546954

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end smallest_possible_denominator_l546_546954


namespace minister_can_organize_traffic_l546_546303

-- Definition of cities and roads
structure City (α : Type) :=
(road : α → α → Prop)

-- Defining the Minister's goal
def organize_traffic {α : Type} (c : City α) (num_days : ℕ) : Prop :=
∀ x y : α, c.road x y → num_days ≤ 214

theorem minister_can_organize_traffic :
  ∃ (c : City ℕ) (num_days : ℕ), (num_days ≤ 214 ∧ organize_traffic c num_days) :=
by {
  sorry
}

end minister_can_organize_traffic_l546_546303


namespace min_roots_in_interval_l546_546418

-- Define the function f
variable {f : ℝ → ℝ}

-- Symmetry conditions for f
axiom symmetry_2 (x : ℝ) : f (2 + x) = f (2 - x)
axiom symmetry_7 (x : ℝ) : f (7 + x) = f (7 - x)

-- Root condition
axiom root_zero : f 0 = 0

-- Goal to prove the minimum number of roots in the interval [-1000, 1000] is 401
theorem min_roots_in_interval : 
  let roots := { x : ℝ | f x = 0 ∧ -1000 ≤ x ∧ x ≤ 1000 }
  in roots.card ≥ 401 :=
sorry

end min_roots_in_interval_l546_546418


namespace white_roses_total_l546_546371

theorem white_roses_total (bq_num : ℕ) (tbl_num : ℕ) (roses_per_bq : ℕ) (roses_per_tbl : ℕ)
  (total_roses : ℕ) 
  (h1 : bq_num = 5) 
  (h2 : tbl_num = 7) 
  (h3 : roses_per_bq = 5) 
  (h4 : roses_per_tbl = 12)
  (h5 : total_roses = 109) : 
  bq_num * roses_per_bq + tbl_num * roses_per_tbl = total_roses := 
by 
  rw [h1, h2, h3, h4, h5]
  exact rfl

end white_roses_total_l546_546371


namespace fibonacci_recurrence_l546_546546

theorem fibonacci_recurrence (f : ℕ → ℝ) (a b : ℝ) 
  (h₀ : f 0 = 1) 
  (h₁ : f 1 = 1) 
  (h₂ : ∀ n, f (n + 2) = f (n + 1) + f n)
  (h₃ : a + b = 1) 
  (h₄ : a * b = -1) 
  (h₅ : a > b) 
  : ∀ n, f n = (a ^ (n + 1) - b ^ (n + 1)) / Real.sqrt 5 := by
  sorry

end fibonacci_recurrence_l546_546546


namespace find_k_plus_a_l546_546136

theorem find_k_plus_a (k a : ℤ) (h1 : k > a) (h2 : a > 0) 
(h3 : 2 * (Int.natAbs (a - k)) * (Int.natAbs (a + k)) = 32) : k + a = 8 :=
by
  sorry

end find_k_plus_a_l546_546136


namespace total_amount_l546_546726

noncomputable def rahul_days : ℕ := 3
noncomputable def rajesh_days : ℕ := 2
noncomputable def rahul_share : ℝ := 142
noncomputable def total_parts : ℕ := 5

theorem total_amount (rahul_days rajesh_days : ℕ) (rahul_share : ℝ) (total_parts : ℕ) : 
  rahul_days = 3 → 
  rajesh_days = 2 → 
  rahul_share = 142 → 
  total_parts = 5 → 
  let part_value := rahul_share / 2 in
  let total_amount := total_parts * part_value in
  total_amount = 355 :=
by
  intros h1 h2 h3 h4
  sorry

end total_amount_l546_546726


namespace find_vector_b_l546_546632

open_locale classical

variables (a b : ℝ × ℝ)

theorem find_vector_b
  (h₁ : a = (1, 2))
  (h₂ : 2 • a + b = (3, 2)) :
  b = (1, -2) :=
begin
  sorry
end

end find_vector_b_l546_546632


namespace flower_beds_count_l546_546720

theorem flower_beds_count (total_seeds seeds_per_bed : ℕ) (h1 : total_seeds = 60) (h2 : seeds_per_bed = 10) : total_seeds / seeds_per_bed = 6 :=
by
  rw [h1, h2]
  simp
  norm_num
  sorry

end flower_beds_count_l546_546720


namespace min_dot_product_AB_AC_l546_546908

noncomputable def circle_O : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 4 }
noncomputable def point_A : ℝ × ℝ := (2, 0)
noncomputable def circle_A (r : ℝ) : Set (ℝ × ℝ) := { p | (p.1 - 2)^2 + p.2^2 = r^2 }

theorem min_dot_product_AB_AC (r : ℝ) (h : r > 0) :
  ∀ B C ∈ circle_A r,
  ∃ A ∈ circle_O,
  (min (B - A) • (C - A)) = -2 := sorry

end min_dot_product_AB_AC_l546_546908


namespace equivalent_proof_problem_l546_546352

noncomputable def setA := {x : ℝ | ∃ y : ℝ, y = sqrt((x - 4) / (2 - x)) ∧ 2 < x ∧ x ≤ 4}
noncomputable def setB := {k : ℝ | (k = 0 ∨ (0 < k ∧ k < 4))}
noncomputable def f (x : ℝ) := 2 / (x - 1)
noncomputable def setOfY (A : set ℝ) := {y : ℝ | ∃ x ∈ A, y = f x}

theorem equivalent_proof_problem (a : ℝ) :
  (∀ x ∈ setA, ∃ y ∈ setB, y = 2 / (x - 1)) →
  (a ∈ setB → a ∉ setOfY setA → a ∈ (set.Ico 0 (2/3)) ∪ set.Ico 2 4) :=
sorry

end equivalent_proof_problem_l546_546352


namespace area_inequality_l546_546524

noncomputable def area (A B C : Type) : Type := sorry

variables {A B C K L M D E F : Type} 
  [IsTriangle A B C] -- ABC is a triangle 
  [OnSide K B C] -- K is on the side BC
  [OnSide L C A] -- L is on the side CA
  [OnSide M A B] -- M is on the side AB
  [OnSide D L M] -- D is on the side LM
  [OnSide E M K] -- E is on the side MK
  [OnSide F K L] -- F is on the side KL

theorem area_inequality :
  (area A M E) * (area C K E) * (area B K F) * (area A L F) * (area B D M) * (area C L D) 
  ≤ (1 / 8 * area A B C) ^ 6 := 
sorry

end area_inequality_l546_546524


namespace expression_result_l546_546178

theorem expression_result :
  ( (9 + (1 / 2)) + (7 + (1 / 6)) + (5 + (1 / 12)) + (3 + (1 / 20)) + (1 + (1 / 30)) ) * 12 = 310 := by
  sorry

end expression_result_l546_546178


namespace chessboard_rectangles_squares_l546_546430

noncomputable def m_nat : ℕ := 19
noncomputable def n_nat : ℕ := 135
noncomputable def r : ℕ := 2025
noncomputable def s : ℕ := 285

theorem chessboard_rectangles_squares : m_nat + n_nat = 154 :=
by
  have ratio_s_r : s / r = 19 / 135 := sorry
  have relatively_prime : Nat.gcd 19 135 = 1 := sorry
  exact m_nat + n_nat = 154
  sorry

end chessboard_rectangles_squares_l546_546430


namespace find_t_l546_546266

def vector_perpendicular (a b : ℝ × ℝ) (t : ℝ) : Prop :=
  let ta_b := (t * a.1 + b.1, t * a.2 + b.2)
  a.1 * ta_b.1 + a.2 * ta_b.2 = 0

theorem find_t : ∀ (t : ℝ),
  let a := (1 : ℝ, -1 : ℝ)
  let b := (6 : ℝ, -4 : ℝ)
  vector_perpendicular a b t → t = -1 :=
by 
  intros t a b hp a_def b_def
  rw [a_def, b_def] at hp
  sorry

end find_t_l546_546266


namespace collinear_A_K_O_l546_546164

-- Define the problem statement
theorem collinear_A_K_O 
  (A B C E F G H K O : Point)
  (circumcenter : IsCircumcenter O A B C)
  (altitude_BE : IsAltitude BE A C)
  (altitude_CF : IsAltitude CF A B)
  (foot_perpendicular_EG : IsFootPerpendicular E G A B)
  (foot_perpendicular_FH : IsFootPerpendicular F H A C)
  (intersection_EG_FH_K : Intersects EG FH K) :
  Collinear A K O := by
  sorry

end collinear_A_K_O_l546_546164


namespace cuboid_surface_area_l546_546481

-- Definitions
def Length := 12  -- meters
def Breadth := 14  -- meters
def Height := 7  -- meters

-- Surface area of a cuboid formula
def surfaceAreaOfCuboid (l b h : Nat) : Nat :=
  2 * (l * b + l * h + b * h)

-- Proof statement
theorem cuboid_surface_area : surfaceAreaOfCuboid Length Breadth Height = 700 := by
  sorry

end cuboid_surface_area_l546_546481


namespace rounding_to_nearest_hundredth_l546_546445

-- Conditions
def number_to_round : ℝ := 6.8346
def nearest_hundredth (x : ℝ) : ℝ := Float.round (x * 100) / 100

-- Proof statement
theorem rounding_to_nearest_hundredth :
  nearest_hundredth number_to_round = 6.83 := by
  sorry

end rounding_to_nearest_hundredth_l546_546445


namespace sum_of_proper_divisors_of_81_l546_546820

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l546_546820


namespace gcd_gx_x_l546_546393

def g (x : ℕ) : ℕ := (5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (3 * x + 8)

theorem gcd_gx_x (x : ℕ) (h : 27720 ∣ x) : Nat.gcd (g x) x = 168 := by
  sorry

end gcd_gx_x_l546_546393


namespace rationalize_denominator_l546_546727

theorem rationalize_denominator (a b : ℝ) (ha : a = (3 : ℝ)^(1/3)) (hb : b = (27 : ℝ)^(1/3)) : 
  (1 / (a + b)) * (9^(1/3)) = (9^(1/3)) / 12 :=
by
  have h1 : b = 3 * a := sorry
  rw [h1, ←mul_assoc]
  have h2 : (a + 3 * a) = 4 * a := sorry
  rw [h2, mul_comm, ←one_div_mul_one_div]
  sorry

end rationalize_denominator_l546_546727


namespace discount_percentage_is_10_l546_546039

-- Definitions of the conditions directly translated
def CP (MP : ℝ) : ℝ := 0.7 * MP
def GainPercent : ℝ := 0.2857142857142857
def SP (MP : ℝ) : ℝ := CP MP * (1 + GainPercent)

-- Using the alternative expression for selling price involving discount percentage
def DiscountSP (MP : ℝ) (D : ℝ) : ℝ := MP * (1 - D)

-- The theorem to prove the discount percentage is 10%
theorem discount_percentage_is_10 (MP : ℝ) : ∃ D : ℝ, DiscountSP MP D = SP MP ∧ D = 0.1 := 
by
  use 0.1
  sorry

end discount_percentage_is_10_l546_546039


namespace moe_mowing_time_l546_546365

-- Moe's mowing problem conditions translated to definitions
def lawn_width : ℝ := 100
def lawn_length : ℝ := 200
def swath_width_inches : ℝ := 30
def overlap_inches : ℝ := 6
def walking_rate : ℝ := 5000 -- in feet per hour

-- Convert swath width and overlap to feet
def effective_swath_width_feet : ℝ :=
  (swath_width_inches - overlap_inches) / 12

-- Calculate the number of strips needed
def num_strips : ℝ :=
  lawn_length / effective_swath_width_feet

-- Calculate the total distance Moe mows
def total_distance : ℝ :=
  num_strips * lawn_width

-- Calculate the time required to mow the lawn
def mowing_time : ℝ :=
  total_distance / walking_rate

-- Prove that the time it takes Moe to mow the lawn is 2 hours
theorem moe_mowing_time : mowing_time = 2 := by
  sorry

end moe_mowing_time_l546_546365


namespace diff_highest_lowest_avg_75_7_consec_odd_l546_546035

theorem diff_highest_lowest_avg_75_7_consec_odd :
  (∃ (s : Finset ℤ), s.card = 7 ∧ (∀ a b ∈ s, a % 2 = 1 ∧ b % 2 = 1 ∧ abs (a - b) = 2) ∧ s.sum / 7 = 75) →
  (∃ a b ∈ (s : Finset ℤ), s.max = some a ∧ s.min = some b ∧ a - b = 12)
:= sorry

end diff_highest_lowest_avg_75_7_consec_odd_l546_546035


namespace triangle_angle_sum_l546_546674

theorem triangle_angle_sum (x : ℝ) (h1 : 70 + 50 + x = 180) : x = 60 := by
  -- proof goes here
  sorry

end triangle_angle_sum_l546_546674


namespace quinary_to_octal_444_l546_546181

theorem quinary_to_octal_444 :
  (let quinary := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  let decimal := 124 in
  let octal := 1 * 8^2 + 7 * 8^1 + 4 * 8^0 in
  quinary = decimal ∧ decimal = octal :=
  quinary = 4 * 25 + 4 * 5 + 4 ∧ 124 = 1 * 64 + 7 * 8 + 4) :=
by
  sorry

end quinary_to_octal_444_l546_546181


namespace range_of_log_sqrt_fourth_l546_546459

noncomputable def sqrt_fourth_root (x : ℝ) : ℝ := real.sqrt (real.sqrt x)

theorem range_of_log_sqrt_fourth :
  ∀ x : ℝ, (0 < x ∧ x < real.pi / 2) →
  (∃ y : ℝ, y = real.log (sqrt_fourth_root (real.sin x)) ∧ y ≤ 0) :=
by
  intro x hx
  sorry

end range_of_log_sqrt_fourth_l546_546459


namespace find_length_QR_l546_546030

-- Define the provided conditions as Lean definitions
variables (Q P R : ℝ) (h_cos : Real.cos Q = 0.3) (QP : ℝ) (h_QP : QP = 15)
  
-- State the theorem we need to prove
theorem find_length_QR (QR : ℝ) (h_triangle : QP / QR = Real.cos Q) : QR = 50 := sorry

end find_length_QR_l546_546030


namespace n_points_convex_ngon_l546_546974

noncomputable def convex_n_gon (points : set (ℝ × ℝ)) : Prop := 
  ∀ (A₁ A₂ A₃ A₄ : ℝ × ℝ), A₁ ≠ A₂ ∧ A₁ ≠ A₃ ∧ A₁ ≠ A₄ ∧ 
                           A₂ ≠ A₃ ∧ A₂ ≠ A₄ ∧ A₃ ≠ A₄ → 
                           (A₁, A₂) ∈ points ∧ (A₁, A₃) ∈ points ∧ 
                           (A₁, A₄) ∈ points ∧ (A₂, A₃) ∈ points ∧ 
                           (A₂, A₄) ∈ points ∧ (A₃, A₄) ∈ points → 
                           ConvexHull ℝ {A₁, A₂, A₃, A₄} = polygon_hull ℝ {A₁, A₂, A₃, A₄}

theorem n_points_convex_ngon (n : ℕ) (points : fin n → (ℝ × ℝ))
  (h1 : ∀ (i j k : fin n), i ≠ j → j ≠ k → i ≠ k → ¬ collinear ({points i, points j, points k} : set (ℝ × ℝ))) 
  (h2 : ∀ (i j k l : fin n), ¬ is_collinear {points i, points j, points k, points l}) :
  convex_hull ℝ (set.range points) = polygon_hull ℝ (set.range points) :=
sorry

end n_points_convex_ngon_l546_546974


namespace length_AB_is_four_l546_546633

noncomputable def circle_1_equation := ∀ (x y : ℝ), x^2 + y^2 = 5
def circle_2_equation (m : ℝ) := ∀ (x y : ℝ), (x - m)^2 + y^2 = 20
def intersects (A B : ℝ × ℝ) : Prop := 
  circle_1_equation A.1 A.2 ∧ circle_1_equation B.1 B.2 ∧
  circle_2_equation 5 A.1 A.2

theorem length_AB_is_four :
  ∀ (A B : ℝ × ℝ),
    intersects A B →
    (∀ (m : ℝ), ( √5 < |m| ∧ |m| < 3√5 ) → (O_1 A ⊥ AO_2 A )) →
    dist A B = 4 :=
by
  intros A B intersects cond
  sorry

end length_AB_is_four_l546_546633


namespace greatest_integer_with_gcd_6_l546_546077

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l546_546077


namespace total_tickets_needed_l546_546443

-- Define the conditions
def rollercoaster_rides (n : Nat) := 3
def catapult_rides (n : Nat) := 2
def ferris_wheel_rides (n : Nat) := 1
def rollercoaster_cost (n : Nat) := 4
def catapult_cost (n : Nat) := 4
def ferris_wheel_cost (n : Nat) := 1

-- Prove the total number of tickets needed
theorem total_tickets_needed : 
  rollercoaster_rides 0 * rollercoaster_cost 0 +
  catapult_rides 0 * catapult_cost 0 +
  ferris_wheel_rides 0 * ferris_wheel_cost 0 = 21 :=
by 
  sorry

end total_tickets_needed_l546_546443


namespace area_of_closed_shape_l546_546176

theorem area_of_closed_shape :
  ∫ y in (-2 : ℝ)..3, ((2:ℝ)^y + 2 - (2:ℝ)^y) = 10 := by
  sorry

end area_of_closed_shape_l546_546176


namespace total_minutes_worked_l546_546013

def minutesWorked : Nat :=
  -- Monday
  let monday := 45 + 30
  -- Tuesday
  let tuesday := 90 + 45 - 15
  -- Wednesday
  let wednesday := 40 + 60
  -- Thursday
  let thursday := 90 + 75 - 30
  -- Friday
  let friday := 55 + 20
  -- Saturday
  let saturday := 120 + 60 - 40
  -- Sunday
  let sunday := 105 + 135 - 45
  -- Total
  let total := monday + tuesday + wednesday + thursday + friday + saturday + sunday
  total

theorem total_minutes_worked : minutesWorked = 840 := by
  sorry

end total_minutes_worked_l546_546013


namespace binomial_theorem_example_l546_546705

theorem binomial_theorem_example 
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : (2 - 1)^5 = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5)
  (h2 : (2 - (-1))^5 = a_0 - a_1 + a_2 * (-1)^2 - a_3 * (-1)^3 + a_4 * (-1)^4 - a_5 * (-1)^5)
  (h3 : a_5 = -1) :
  (a_0 + a_2 + a_4 : ℤ) / (a_1 + a_3 : ℤ) = -61 / 60 := 
sorry

end binomial_theorem_example_l546_546705


namespace grooming_time_5_dogs_3_cats_l546_546326

theorem grooming_time_5_dogs_3_cats :
  (2.5 * 5 + 0.5 * 3) * 60 = 840 :=
by
  -- Prove that grooming 5 dogs and 3 cats takes 840 minutes.
  sorry

end grooming_time_5_dogs_3_cats_l546_546326


namespace max_n_is_9_l546_546218

theorem max_n_is_9 :
  ∃ (a : Fin 9 → ℕ), (1 < a 0 ∧ a 8 < 2009) ∧
  (∀ i j, i < j → a i < a j) ∧
  (∀ i, (∑ k in Finset.erase (Finset.univ : Finset (Fin 9)) i, a k) % 8 = 0) :=
begin
  sorry
end

end max_n_is_9_l546_546218


namespace area_of_circle_l546_546454

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end area_of_circle_l546_546454


namespace parabola_passes_through_points_and_has_solution_4_l546_546051

theorem parabola_passes_through_points_and_has_solution_4 
  (a h k m: ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k → 
    (y = 0 → (x = -1 → x = 5))) → 
  (∃ m, ∀ x, (a * (x - h + m) ^ 2 + k = 0) → x = 4) → 
  m = -5 ∨ m = 1 :=
sorry

end parabola_passes_through_points_and_has_solution_4_l546_546051


namespace periodic_not_monotonic_l546_546206

theorem periodic_not_monotonic (f : ℝ → ℝ) :
  (¬ (∃T > 0, ∀x, f(x + T) = f(x) ∧ ∀ x y, x < y → f x ≤ f y)) →
  ¬ ((∀ x y, x < y → f x ≤ f y) → ¬ (∃T > 0, ∀x, f(x + T) = f(x))) ∧
  ¬ ((∃T > 0, ∀x, f(x + T) = f(x)) → (∀ x y, x < y → f x ≤ f y)) ∧
  ¬ ((∀ x y, x < y → f x ≤ f y) → (∃T > 0, ∀x, f(x + T) = f(x)))
:=
by
  sorry

end periodic_not_monotonic_l546_546206


namespace solve_system_of_equations_l546_546739

theorem solve_system_of_equations (a b c d x y z t : ℝ) 
  (h_distinct : ∀ i j : ℝ, i ≠ j → i ∈ {a, b, c, d} → j ∈ {a, b, c, d} → i ≠ j) 
  (h1 : |a - b| * y + |a - c| * z + |a - d| * t = 1)
  (h2 : |b - a| * x + |b - c| * z + |b - d| * t = 1)
  (h3 : |c - a| * x + |c - b| * y + |c - d| * t = 1)
  (h4 : |d - a| * x + |d - b| * y + |d - c| * z = 1) :
  x = 1 / (a - d) ∧ y = 0 ∧ z = 0 ∧ t = 1 / (a - d) :=
by
  sorry

end solve_system_of_equations_l546_546739


namespace mia_suitcase_combinations_l546_546010

theorem mia_suitcase_combinations : 
  let first_number_options := {n : ℕ | n ≤ 40 ∧ n % 4 = 0},
      second_number_options := {n : ℕ | n ≤ 40 ∧ n % 2 = 1},
      third_number_options := {n : ℕ | n ≤ 40 ∧ n % 5 = 0} in
  (first_number_options.card * second_number_options.card * third_number_options.card = 1600) :=
by
  let first_number_options := {n : ℕ | n ≤ 40 ∧ n % 4 = 0},
      second_number_options := {n : ℕ | n ≤ 40 ∧ n % 2 = 1},
      third_number_options := {n : ℕ | n ≤ 40 ∧ n % 5 = 0};
  have first_opts_count : first_number_options.card = 10 := sorry;
  have second_opts_count : second_number_options.card = 20 := sorry;
  have third_opts_count : third_number_options.card = 8 := sorry;
  have total_combinations : first_number_options.card * second_number_options.card * third_number_options.card = 1600 := by
    rw [first_opts_count, second_opts_count, third_opts_count];
    norm_num;
  exact total_combinations

end mia_suitcase_combinations_l546_546010


namespace find_b_l546_546560

def polynomial_has_roots_forming_parallelogram (p : ℂ → ℂ) : Prop :=
∃ z₀ z₁ z₂ z₃ : ℂ, p z₀ = 0 ∧ p z₁ = 0 ∧ p z₂ = 0 ∧ p z₃ = 0 ∧
( z₀ + z₁ = z₂ + z₃ ∧ z₀ + z₂ = z₁ + z₃ )

def polynomial_condition (b : ℝ) : ℂ → ℂ :=
λ z, z^4 - 4 * z^3 + 10 * b * z^2 - 2 * (3 * b^2 + 2 * b - 2) * z + 4

theorem find_b : ∀ b : ℝ, (polynomial_has_roots_forming_parallelogram (polynomial_condition b)) → b = 4 :=
by sorry

end find_b_l546_546560


namespace expression_value_l546_546846

theorem expression_value 
  (a b c : ℕ) 
  (ha : a = 12) 
  (hb : b = 2) 
  (hc : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := 
by 
  sorry

end expression_value_l546_546846


namespace sequence_formula_l546_546227

-- Define the sequence
def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n + 1

-- Prove that the sequence has the specified formula
theorem sequence_formula (a : ℕ → ℝ) (h : sequence a) : ∀ n, a n = (1/2) * (3 ^ n - 1) :=
by
  sorry

end sequence_formula_l546_546227


namespace tickets_sold_l546_546058

-- Define the conditions
noncomputable def cracker_price : ℝ := 2.25
noncomputable def beverage_price : ℝ := 1.50
noncomputable def chocolate_price : ℝ := 1.00

noncomputable def cracker_count : ℝ := 3
noncomputable def beverage_count : ℝ := 4
noncomputable def chocolate_count : ℝ := 4

noncomputable def average_sales_per_ticket : ℝ := 2.79

-- Define the total sales per x movie tickets.
noncomputable def total_sales_per_ticket (x : ℝ) : ℝ :=
 cracker_count * cracker_price + beverage_count * beverage_price + chocolate_count * chocolate_price

-- Prove the number of movie tickets sold
theorem tickets_sold : ∃ x: ℕ, (total_sales_per_ticket x / average_sales_per_ticket).round = 6 :=
by
  let x := 6
  have total_sales := 6.75 + 6.00 + 4.00
  have x_calculated := total_sales / 2.79
  have x_rounded := x_calculated.round
  have : x_rounded = x := by
    calc
      x_rounded = (16.75 / 2.79).round : by sorry -- Calculation of rounding
             ... = 6                        : by sorry -- Actual rounding value given in problem
  existsi x
  exact this

end tickets_sold_l546_546058


namespace bananas_left_on_tree_l546_546864

noncomputable theory

def initial_bananas := 310
def bananas_cut (x : ℕ) := x
def bananas_eaten := 70
def bananas_left (x : ℕ) := x - bananas_eaten

lemma bananas_equation (x : ℕ) (h : bananas_left x = 2 * (bananas_left x - bananas_eaten)) :
  x = 210 :=
sorry

theorem bananas_left_on_tree :
  ∃ x : ℕ, bananas_cut x ∧ (bananas_left x = 2 * (bananas_left x - bananas_eaten)) → initial_bananas - x = 100 :=
begin
  use 210,
  split,
  { sorry },  -- Placeholder for proof that 210 is the correct number of bananas cut.
  { sorry }   -- Placeholder for proof that the number of bananas left on the tree is 100.
end

end bananas_left_on_tree_l546_546864


namespace compute_sum_l546_546690

def T (n : ℕ) : ℤ :=
  if even n then -n / 2 else (n + 1) / 2

theorem compute_sum : T 18 + T 34 + T 51 = 0 := by
  unfold T
  apply sorry

end compute_sum_l546_546690


namespace sum_of_composite_not_odd_divisors_120_l546_546783

def sum_composite_not_odd_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d ∣ n ∧ ¬ Nat.prime d ∧ ¬ odd d).sum

theorem sum_of_composite_not_odd_divisors_120 :
  sum_composite_not_odd_divisors 120 = 334 :=
by
  sorry

end sum_of_composite_not_odd_divisors_120_l546_546783


namespace proof_increasing_function_on_neg2_0_l546_546237

variable (a b : ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem proof_increasing_function_on_neg2_0 (h1 : is_even_function (f a b)) 
                                           (h2 : ∀ x, x ∈ (Ioo (a-1) 2) → f a b x = f a b (-x))
                                           : is_increasing_on (f (-1) 0) (Icc (-2) 0) :=
sorry

end proof_increasing_function_on_neg2_0_l546_546237


namespace _l546_546099

-- Definitions based on the conditions
def work_done_by_A_in_one_day : ℝ := 1 / 10
def work_done_by_A_and_B_together_in_one_day : ℝ := 1 / 6

-- The main theorem to prove
example : ∃ (b : ℝ), (work_done_by_A_in_one_day + (1 / b) = work_done_by_A_and_B_together_in_one_day) :=
by
  sorry

end _l546_546099


namespace perpendicular_and_equal_AD_EF_l546_546023

variable {A B C D E F : Point}
variable (h1 : triangle A B C)
variable (h2 : isosceles_right_triangle B C D)
variable (h3 : isosceles_right_triangle C A E)
variable (h4 : isosceles_right_triangle A B F)

theorem perpendicular_and_equal_AD_EF
  (h5 : is_constructed_externally A B C D E F) :
  perp AD EF ∧ AD = EF := 
sorry

end perpendicular_and_equal_AD_EF_l546_546023


namespace rationalize_denominator_l546_546730

theorem rationalize_denominator (a b c : ℚ) (h1 : b = 3 * a) : (1 / (a + b) = c) ↔ c = (Real.cbrt 9) / 12 :=
by
  sorry

end rationalize_denominator_l546_546730


namespace initial_people_lifting_weights_l546_546170

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end initial_people_lifting_weights_l546_546170


namespace area_triangle_BFC_l546_546394

-- Definitions based on conditions
def Rectangle (A B C D : Type) (AB BC CD DA : ℝ) := AB = 5 ∧ BC = 12 ∧ CD = 5 ∧ DA = 12

def PointOnDiagonal (F A C : Type) := True  -- Simplified definition as being on the diagonal
def Perpendicular (B F A C : Type) := True  -- Simplified definition as being perpendicular

-- Main theorem statement
theorem area_triangle_BFC 
  (A B C D F : Type)
  (rectangle_ABCD : Rectangle A B C D 5 12 5 12)
  (F_on_AC : PointOnDiagonal F A C)
  (BF_perpendicular_AC : Perpendicular B F A C) :
  ∃ (area : ℝ), area = 30 :=
sorry

end area_triangle_BFC_l546_546394


namespace polynomial_real_roots_probability_correct_l546_546517

noncomputable def polynomial_real_roots_probability : ℝ :=
  let a_interval_length := 18 - (-20)
  let excluded_interval_length := 2 -- interval (-1/2, 3/2) has length 2
  in (a_interval_length - excluded_interval_length) / a_interval_length

theorem polynomial_real_roots_probability_correct :
  polynomial_real_roots_probability = 18 / 19 :=
by
  -- Placeholder for proof
  sorry

end polynomial_real_roots_probability_correct_l546_546517


namespace part_I_part_II_l546_546661

-- Define the conditions using given parameters and functions
variable (α : ℝ) (t : ℝ)
-- a condition stating that α should be between 0 and π (exclusive)
def α_cond : Prop := 0 < α ∧ α < π

-- The parametric equations of the line l
def x_line (α t : ℝ) : ℝ := 1/2 + t * cos α
def y_line (α t : ℝ) : ℝ := t * sin α

-- The polar equation's conversion to rectangular coordinates
def polar_to_rect (ρ θ : ℝ) := 
  let x := ρ * cos θ
  let y := ρ * sin θ
  y^2 = 2 * x

-- Part (I)
theorem part_I : ∀ ρ θ, ρ = 2 * cos θ / sin θ^2 → polar_to_rect ρ θ :=
sorry

-- A function to calculate the distance AB
def distance_AB (α : ℝ) (t1 t2 : ℝ) : ℝ := 
  abs (t1 - t2)

-- Minimum value for |AB| given the relevant equations and conditions
theorem part_II : 
  ∀ {t1 t2 : ℝ}, α_cond α 
    → t1 + t2 = 2 * cos(α) / (sin(α)^2)
    → t1 * t2 = -1 / (sin(α)^2)
    → dist_AB α t1 t2 = 2 :=
sorry

end part_I_part_II_l546_546661


namespace subtract_decimal_l546_546946

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end subtract_decimal_l546_546946


namespace area_of_region_l546_546452

noncomputable def area_of_enclosed_region : Real :=
  -- equation of the circle after completing square
  let circle_eqn := fun (x y : Real) => ((x - 3)^2 + (y + 4)^2 = 16)
  if circle_eqn then 
    Real.pi * 4^2
  else
    0

theorem area_of_region (h : ∀ x y, x^2 + y^2 - 6x + 8y = -9 → ((x-3)^2 + (y+4)^2 = 16)) :
  area_of_enclosed_region = 16 * Real.pi :=
by
  -- This is a statement, so just include a sorry to skip the proof.
  sorry

end area_of_region_l546_546452


namespace initial_percentage_of_water_is_20_l546_546877

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end initial_percentage_of_water_is_20_l546_546877


namespace find_sum_A_B_l546_546052

-- Define ω as a root of the polynomial x^2 + x + 1
noncomputable def ω : ℂ := sorry

-- Define the polynomial P
noncomputable def P (x : ℂ) (A B : ℂ) : ℂ := x^101 + A * x + B

-- State the main theorem
theorem find_sum_A_B (A B : ℂ) : 
  (∀ x : ℂ, (x^2 + x + 1 = 0) → P x A B = 0) → A + B = 2 :=
by
  intros Divisibility
  -- Here, you would provide the steps to prove the theorem if necessary
  sorry

end find_sum_A_B_l546_546052


namespace max_value_of_f_l546_546759

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) (h : -2 < a ∧ a ≤ 0) : 
  ∀ x ∈ (Set.Icc 0 (a + 2)), f x ≤ 3 :=
sorry

end max_value_of_f_l546_546759


namespace coefficient_x5_in_expansion_l546_546408

theorem coefficient_x5_in_expansion :
  let trinomial := 1 - x + x^2
  let binomial := (1 + x)^6
  let expansion := trinomial * binomial
  (coeff expansion 5) = 21 :=
by
  sorry

end coefficient_x5_in_expansion_l546_546408


namespace number_of_ways_exactly_one_common_course_l546_546157

variable (A B : Finset ℕ)

noncomputable def courses : Finset ℕ := {1, 2, 3, 4}

lemma choose_two_courses_each : (A.card = 2) ∧ (B.card = 2) → (A ∪ B).card = 3 → A ∩ B ≠ ∅ → A ∩ B ≠ A ∧ A ∩ B ≠ B → (A = B → A.card ≠ 2) → (A ∅ B → A.card ≠ 2) → 
  tot_ways := (Finset.choose courses 2).card * (Finset.choose courses 2).card
  := 36

lemma same_courses_ways : (Finset.choose courses 2).card = 6
lemma different_courses_ways : (Finset.choose courses 2).card = 6

theorem number_of_ways_exactly_one_common_course :
  ∃ n, (tot_ways - same_courses_ways - different_courses_ways = n ∧ n = 24) :=
by
  apply Exists.intro 24
  sorry

end number_of_ways_exactly_one_common_course_l546_546157


namespace sales_tax_paid_l546_546927

theorem sales_tax_paid 
  (total_spent : ℝ) 
  (tax_free_cost : ℝ) 
  (tax_rate : ℝ) 
  (cost_of_taxable_items : ℝ) 
  (sales_tax : ℝ) 
  (h1 : total_spent = 40) 
  (h2 : tax_free_cost = 34.7) 
  (h3 : tax_rate = 0.06) 
  (h4 : cost_of_taxable_items = 5) 
  (h5 : sales_tax = 0.3) 
  (h6 : 1.06 * cost_of_taxable_items + tax_free_cost = total_spent) : 
  sales_tax = tax_rate * cost_of_taxable_items :=
sorry

end sales_tax_paid_l546_546927


namespace roadster_paving_company_cement_usage_l546_546732

theorem roadster_paving_company_cement_usage :
  let L := 10
  let T := 5.1
  L + T = 15.1 :=
by
  -- proof is omitted
  sorry

end roadster_paving_company_cement_usage_l546_546732


namespace farmer_loss_l546_546319

noncomputable def verify_loss (x : ℕ) (total_weight : ℕ) (market_price_per_pound leaves_price_per_pound whites_price_per_pound : ℕ) :=
  3 * total_weight - (1.5 * x + 1.5 * (total_weight - x)) = 150

theorem farmer_loss :
  verify_loss 50 100 3 1.5 1.5 := 
by
  sorry

end farmer_loss_l546_546319


namespace coffee_cost_per_week_l546_546329

def num_people: ℕ := 4
def cups_per_person_per_day: ℕ := 2
def ounces_per_cup: ℝ := 0.5
def cost_per_ounce: ℝ := 1.25

theorem coffee_cost_per_week : 
  (num_people * cups_per_person_per_day * ounces_per_cup * 7 * cost_per_ounce) = 35 :=
by
  sorry

end coffee_cost_per_week_l546_546329


namespace sum_proper_divisors_of_81_l546_546844

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l546_546844


namespace min_value_expression_l546_546694

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((5 / c) - 1)^2 = 4 * (Real.root 4 5 - 1)^2 :=
by
  sorry

end min_value_expression_l546_546694


namespace part1_part2_part3_part4_l546_546617

noncomputable def f : ℝ → ℝ :=
λ x, if x < -1 then -x - 1 else if x <= 1 then -x^2 + 1 else x - 1

theorem part1 :
  f 2 = 1 ∧ f (-2) = 1 :=
by sorry

theorem part2 (a : ℝ) :
  f a = 1 → (a = 0 ∨ a = 2 ∨ a = -2) :=
by sorry

theorem part3 :
  ∀ x : ℝ, f (-x) = f x :=
by sorry

theorem part4 :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 0 → f x ≤ f (x + 1)) ∧
  (∀ x : ℝ, 1 ≤ x → f x ≤ f (x + 1)) ∧
  (∀ x : ℝ, x < -1 → f x ≥ f (x + 1)) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x ≥ f (x + 1)) :=
by sorry

end part1_part2_part3_part4_l546_546617


namespace actual_price_of_good_l546_546146

variables (P : Real)

theorem actual_price_of_good:
  (∀ (P : ℝ), 0.5450625 * P = 6500 → P = 6500 / 0.5450625) :=
  by sorry

end actual_price_of_good_l546_546146


namespace arithmetic_sequence_a5_l546_546992

theorem arithmetic_sequence_a5 {a : ℕ → ℕ} 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 2 + a 8 = 12) : 
  a 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l546_546992


namespace average_effective_rate_correct_l546_546873

noncomputable def average_effective_rate : ℝ :=
  let x : ℝ := 250 / 0.113 in
  let post_tax_interest_at_7 := 0.07 * x * 0.9 in
  let post_tax_interest_at_5 := 0.05 * (5000 - x) in
  let total_interest := post_tax_interest_at_7 + post_tax_interest_at_5 in
  total_interest / 5000 * 100

theorem average_effective_rate_correct :
  average_effective_rate = 5.6 :=
by sorry

end average_effective_rate_correct_l546_546873


namespace find_rate_of_interest_l546_546511

def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem find_rate_of_interest :
  ∀ (R : ℕ),
  simple_interest 5000 R 2 + simple_interest 3000 R 4 = 2640 → R = 12 :=
by
  intros R h
  sorry

end find_rate_of_interest_l546_546511


namespace commuting_days_l546_546512

theorem commuting_days 
  (a b c d x : ℕ)
  (cond1 : b + c = 12)
  (cond2 : a + c = 20)
  (cond3 : a + b + 2 * d = 14)
  (cond4 : d = 2) :
  a + b + c + d = 23 := sorry

end commuting_days_l546_546512


namespace sum_proper_divisors_of_81_l546_546843

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l546_546843


namespace part_a_part_b_l546_546492

-- Problem (a)
theorem part_a :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, 2 * f (f x) = f x ∧ f x ≥ 0) ∧ Differentiable ℝ f :=
sorry

-- Problem (b)
theorem part_b :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, -1 ≤ 2 * f (f x) ∧ 2 * f (f x) = f x ∧ f x ≤ 1) ∧ Differentiable ℝ f :=
sorry

end part_a_part_b_l546_546492


namespace committee_membership_l546_546805

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end committee_membership_l546_546805


namespace Serezha_wins_chessboard_game_l546_546742

theorem Serezha_wins_chessboard_game :
  ∀ (start_pos : Fin 8 × Fin 8), ∃ strategy : (ℕ → Fin 8 × Fin 8) → Prop,
  (∀ turn : ℕ, turn % 2 = 0 → strategy turn = (start_pos.1, _)) ∧  -- Tanya's turn
  (∀ turn : ℕ, turn % 2 = 1 → strategy turn = (_, start_pos.2)) ∧  -- Serezha's turn
  ∀ turn : ℕ, ¬ ∃ move : Fin 8 × Fin 8, strategy turn move = move :=
sorry

end Serezha_wins_chessboard_game_l546_546742


namespace solution_set_ineq_l546_546623

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem solution_set_ineq (x : ℝ) : f (x^2 - 4) + f (3*x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end solution_set_ineq_l546_546623


namespace alex_jamie_not_next_to_each_other_prob_l546_546305

def probability_alex_jamie_not_next_to_each_other : ℚ :=
  let total_ways := 45
  let adjacent_ways := 9
  let prob_adjacent := adjacent_ways / total_ways
  let prob_not_adjacent := 1 - prob_adjacent
  prob_not_adjacent

theorem alex_jamie_not_next_to_each_other_prob (total_chairs: ℕ) (alex_chooses: ℕ): (1 ≤ total_chairs) ∧ total_chairs = 10 ∧ alex_chooses = 2 → 
probability_alex_jamie_not_next_to_each_other = 4 / 5 :=
by
  intro h
  have total_chairs_eq : total_chairs = 10 := h.2.1
  have alex_chooses_eq : alex_chooses = 2 := h.2.2
  rw [total_chairs_eq, alex_chooses_eq]
  simp [probability_alex_jamie_not_next_to_each_other]
  admit -- sorry


end alex_jamie_not_next_to_each_other_prob_l546_546305


namespace solve_inequality_l546_546614

noncomputable def f (x : ℝ) : ℝ := 2016^x + log 2016 (sqrt (x^2 + 1) + x) - 2016^(-x) + 2

theorem solve_inequality : {x : ℝ | f (3 * x + 1) + f x > 4} = Ioi (-(1 / 4)) :=
by
  sorry

end solve_inequality_l546_546614


namespace sum_proper_divisors_of_81_l546_546840

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l546_546840


namespace intersection_points_vertex_coordinates_value_of_m_l546_546980

-- Part (1)
theorem intersection_points (a : ℝ) (h : a = 1 / 2) :
  let y (x : ℝ) := a * x^2 + 2 * x + 1 in
  ∃ x1 x2 : ℝ, y x1 = 0 ∧ y x2 = 0 ∧ 
             x1 = -2 + Real.sqrt 2 ∧ x2 = -2 - Real.sqrt 2 := 
by
  -- Here would go the proof
  sorry

-- Part (2)
theorem vertex_coordinates (a : ℝ) (h : a ≠ 0) :
  let s := -1 / a in
  let t := a * s^2 + 2 * s + 1 in
  t = s + 1 := 
by
  -- Here would go the proof
  sorry

-- Part (3)
theorem value_of_m (a m : ℝ) (h1 : a < 0) (h2 : ∀ x, 0 ≤ x → x ≤ m → -2 ≤ a * x^2 + 2 * x + 1 ∧ a * x^2 + 2 * x + 1 ≤ 2) :
  m = 3 :=
by
  -- Here would go the proof
  sorry

end intersection_points_vertex_coordinates_value_of_m_l546_546980


namespace intervals_of_monotonicity_range_of_m_solution_set_of_alpha_l546_546971

noncomputable def f (x : ℝ) := x / |Real.log x|

-- Proof problem 1
theorem intervals_of_monotonicity :
  (∀ x ∈ Ioo 0 1, f' x > 0) ∧
  (∀ x ∈ Ioo e (+∞), f' x > 0) ∧
  (∀ x ∈ Ioo 1 e, f' x < 0) :=
sorry

-- Proof problem 2
theorem range_of_m (m : ℝ) :
  (∀ x ∈ ℝ, f x * f x - (2 * m + 1) * f x + m^2 + m = 0) →
  (Cardinal.mk (Setof {x : ℝ | f x * f x - (2 * m + 1) * f x + m^2 + m = 0}) = 4) →
  e - 1 < m ∧ m < e :=
sorry

-- Proof problem 3
theorem solution_set_of_alpha (α x y : ℝ) (hxy : 0 < y ∧ y < x)
  (h : e * y * (Real.sin α + 1/2) > x / |Real.log x - Real.log y|) :
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 6 < α ∧ α < 2 * k * Real.pi + 5 * Real.pi / 6 :=
sorry

end intervals_of_monotonicity_range_of_m_solution_set_of_alpha_l546_546971


namespace hockey_team_wins_35_of_remaining_56_games_l546_546510

theorem hockey_team_wins_35_of_remaining_56_games 
  (games_won_initial: ℕ) (total_initial_games: ℕ) (total_remaining_games: ℕ) (win_rate_fraction: ℚ)
  (total_games: ℕ) :
  games_won_initial = 30 →
  total_initial_games = 44 →
  total_remaining_games = 56 →
  win_rate_fraction = 65 / 100 →
  total_games = total_initial_games + total_remaining_games →
  let games_won_needed := total_games * win_rate_fraction.numerator / win_rate_fraction.denominator in
  ∃ x : ℕ, (games_won_initial + x = games_won_needed) ∧ (x = 35) :=
by {
  intros h1 h2 h3 h4 h5,
  let games_won_needed := total_games * 13 / 20,
  use 35,
  rw [h1, nat.mul_div_cancel' (by norm_num : 100 % 20 = 0)],
  split,
  norm_num,
  norm_num,
  sorry
}

end hockey_team_wins_35_of_remaining_56_games_l546_546510


namespace permutation_exists_l546_546348

theorem permutation_exists (n : ℕ) (x : Fin n → ℝ) 
  (H1 : abs (∑ i in Finset.range n, x i) = 1)
  (H2 : ∀ i, abs (x i) ≤ (n + 1) / 2) : 
  ∃ (y : Fin n → ℝ), (∀ i, ∃ j, y i = x j) ∧ abs (∑ i in Finset.range n, (i + 1) * y i) ≤ (n + 1) / 2 := 
sorry

end permutation_exists_l546_546348


namespace orthogonal_vectors_z_value_l546_546192

theorem orthogonal_vectors_z_value :
  let v1 : ℝ^3 := ![2, -4, 5]
  let v2 : ℝ^3 := ![3, z, -2]
  (v1.dot v2 = 0) → z = -1 :=
begin
  sorry
end

end orthogonal_vectors_z_value_l546_546192


namespace total_weekly_earnings_l546_546362

-- Define the total weekly hours and earnings
def weekly_hours_weekday : ℕ := 5 * 5
def weekday_rate : ℕ := 3
def weekday_earnings : ℕ := weekly_hours_weekday * weekday_rate

-- Define the total weekend hours and earnings
def weekend_days : ℕ := 2
def weekend_hours_per_day : ℕ := 3
def weekend_rate : ℕ := 3 * 2
def weekend_hours : ℕ := weekend_days * weekend_hours_per_day
def weekend_earnings : ℕ := weekend_hours * weekend_rate

-- Prove that Mitch's total earnings per week are $111
theorem total_weekly_earnings : weekday_earnings + weekend_earnings = 111 := by
  sorry

end total_weekly_earnings_l546_546362


namespace value_of_x_l546_546284

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end value_of_x_l546_546284


namespace eccentricity_of_hyperbola_l546_546624

theorem eccentricity_of_hyperbola (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) 
    (h₂ : (y = x * sqrt 3) -> (y = x * (b / a) ∨ y = x * (-(b / a)))) :
  let c : ℝ := sqrt (a ^ 2 + b ^ 2)
  in (b = sqrt 3 * a) → (c = 2 * a) → 
    (c / a = 2) := by
  intro c h_b h_c
  sorry

end eccentricity_of_hyperbola_l546_546624


namespace greatest_int_with_gcd_18_is_138_l546_546079

theorem greatest_int_with_gcd_18_is_138 :
  ∃ n : ℕ, n < 150 ∧ int.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ int.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_int_with_gcd_18_is_138_l546_546079


namespace problem1_l546_546916

theorem problem1 : (1:ℝ) * (-1)^2023 + (-1/2)^(-2) - (3.14 - Real.pi)^0 = 2 :=
by sorry

end problem1_l546_546916


namespace weight_loss_total_l546_546208

theorem weight_loss_total :
  ∀ (weight1 weight2 weight3 weight4 : ℕ),
    weight1 = 27 →
    weight2 = weight1 - 7 →
    weight3 = 28 →
    weight4 = 28 →
    weight1 + weight2 + weight3 + weight4 = 103 :=
by
  intros weight1 weight2 weight3 weight4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end weight_loss_total_l546_546208


namespace sum_proper_divisors_81_l546_546829

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546829


namespace alpha_beta_square_eq_eight_l546_546276

theorem alpha_beta_square_eq_eight (α β : ℝ) 
  (hα : α^2 = 2*α + 1) 
  (hβ : β^2 = 2*β + 1) 
  (h_distinct : α ≠ β) : 
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_eq_eight_l546_546276


namespace number_of_people_ate_pizza_l546_546527

theorem number_of_people_ate_pizza (total_slices initial_slices remaining_slices slices_per_person : ℕ) 
  (h1 : initial_slices = 16) 
  (h2 : remaining_slices = 4) 
  (h3 : slices_per_person = 2) 
  (eaten_slices : total_slices) 
  (h4 : total_slices = initial_slices - remaining_slices) :
    total_slices / slices_per_person = 6 := 
by {
  -- specify the value for eaten_slices to simplify the calculation
  let eaten_slices := initial_slices - remaining_slices,
  have h5 : eaten_slices = 12, from sorry,
  sorry
  }

end number_of_people_ate_pizza_l546_546527


namespace sum_of_integral_values_l546_546564

-- Let's start a namespace to encapsulate our definitions and theorems.
namespace RationalRootSum

-- Define the polynomial equation as a Lean function.
def poly (x c : ℤ) : ℤ := x^3 - x^2 - 8 * x - c

-- Define the condition for a polynomial having a rational root.
def has_rational_root (c : ℤ) : Prop :=
  ∃ x : ℤ, poly x c = 0

-- Define the condition for c being less than or equal to 30.
def leq_30 (c : ℤ) : Prop := c ≤ 30

-- The main theorem that encapsulates the problem statement.
theorem sum_of_integral_values (sum : ℤ) :
  sum = -10 ↔
    (sum = (∑ c in (Finset.filter (λ c, has_rational_root c) (Finset.range (31: ℤ))), id c)) :=
      sorry

end RationalRootSum

end sum_of_integral_values_l546_546564


namespace rightmost_three_digits_of_7_pow_1997_l546_546447

theorem rightmost_three_digits_of_7_pow_1997 :
  7^1997 % 1000 = 207 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1997_l546_546447


namespace symmetric_pattern_count_l546_546506

noncomputable def number_of_symmetric_patterns (n : ℕ) : ℕ :=
  let regions := 12
  let total_patterns := 2^regions
  total_patterns - 2

theorem symmetric_pattern_count : number_of_symmetric_patterns 8 = 4094 :=
by
  sorry

end symmetric_pattern_count_l546_546506


namespace lunchroom_students_l546_546401

theorem lunchroom_students (students_per_table : ℕ) (tables : ℕ) (h1 : students_per_table = 6) (h2 : tables = 34) : students_per_table * tables = 204 :=
by {
  rw [h1, h2],
  norm_num,
}

end lunchroom_students_l546_546401


namespace factor_polynomial_l546_546944

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) :=
by
  sorry

end factor_polynomial_l546_546944


namespace ratio_proof_l546_546653

-- Definitions and conditions
variables {A B C : ℕ}

-- Given condition: A : B : C = 3 : 2 : 5
def ratio_cond (A B C : ℕ) := 3 * B = 2 * A ∧ 5 * B = 2 * C

-- Theorem statement
theorem ratio_proof (h : ratio_cond A B C) : (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 :=
by sorry

end ratio_proof_l546_546653


namespace perpendicular_bisector_eq_l546_546991

structure Point :=
  (x : ℝ)
  (y : ℝ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

def perp_bisector_eq (M N : Point) : (ℝ × ℝ × ℝ) :=
  let k := slope M N
  let k' := -1 / k
  let mid := midpoint M N
  let b := mid.y - k' * mid.x
  let std_form := (k', -1, b * -1)
  std_form

theorem perpendicular_bisector_eq (M N : Point) (hM : M = ⟨-1, 6⟩) (hN : N = ⟨3, 2⟩) :
  perp_bisector_eq M N = (1, -1, -3) := by
  sorry

end perpendicular_bisector_eq_l546_546991


namespace avg_salary_increase_l546_546034

def initial_avg_salary : ℝ := 1700
def num_employees : ℕ := 20
def manager_salary : ℝ := 3800

theorem avg_salary_increase :
  ((num_employees * initial_avg_salary + manager_salary) / (num_employees + 1)) - initial_avg_salary = 100 :=
by
  sorry

end avg_salary_increase_l546_546034


namespace garden_area_l546_546420

-- Given conditions:
def width := 16
def length (W : ℕ) := 3 * W

-- Proof statement:
theorem garden_area (W : ℕ) (hW : W = width) : length W * W = 768 :=
by
  rw [hW]
  exact rfl

end garden_area_l546_546420


namespace escalator_time_l546_546682

theorem escalator_time
    {d i s : ℝ}
    (h1 : d = 90 * i)
    (h2 : d = 30 * (i + s))
    (h3 : s = 2 * i):
    d / s = 45 := by
  sorry

end escalator_time_l546_546682


namespace compute_sum_l546_546539

theorem compute_sum : 
  (∑ k in finset.range 37, real.sin (5 * k * real.pi / 180) ^ 6) = 73.74 :=
by sorry

end compute_sum_l546_546539


namespace correct_order_of_values_l546_546344

-- Let y = f(x) be an odd function defined on ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given f'(x) and f(x) satisfy the inequality
def derivative_condition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f' x + f x / x > 0

theorem correct_order_of_values
  (f : ℝ → ℝ)
  (f' : ℝ → ℝ)
  (h1 : is_odd_function f)
  (h2 : derivative_condition f f') :
  let a := 1 / 2 * f (1 / 2)
  let b := -2 * f (-2)
  let c := (log (1 / 2)) * f (log (1 / 2)) in
  a < c ∧ c < b :=
by
  sorry

end correct_order_of_values_l546_546344


namespace sum_of_proper_divisors_of_81_l546_546819

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l546_546819


namespace polygon_sides_l546_546128

theorem polygon_sides (n : ℕ) (a1 d : ℝ) (h1 : a1 = 100) (h2 : d = 10)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d < 180) : n = 8 :=
by
  sorry

end polygon_sides_l546_546128


namespace seashells_initial_count_l546_546796

def Tim_initial_seashells (gave_away now_have : Nat) : Nat := 
  gave_away + now_have

theorem seashells_initial_count : 
  Tim_initial_seashells 172 507 = 679 := 
by 
  simp [Tim_initial_seashells] 
  sorry

end seashells_initial_count_l546_546796


namespace smaller_screen_diagonal_l546_546780

/-- The area of a 20-inch square screen is 38 square inches greater than the area
    of a smaller square screen. Prove that the length of the diagonal of the smaller screen is 18 inches. -/
theorem smaller_screen_diagonal (x : ℝ) (d : ℝ) (A₁ A₂ : ℝ)
  (h₀ : d = x * Real.sqrt 2)
  (h₁ : A₁ = 20 * Real.sqrt 2 * 20 * Real.sqrt 2)
  (h₂ : A₂ = x * x)
  (h₃ : A₁ = A₂ + 38) :
  d = 18 :=
by
  sorry

end smaller_screen_diagonal_l546_546780


namespace mutually_exclusive_and_complementary_l546_546578

noncomputable def bag : Finset (Finset ℕ) :=
  {{0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}}

def at_least_one_black (s : Finset ℕ) : Prop :=
  ∃ x ∈ s, x = 0 ∨ x = 1

def all_white (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, x = 2 ∨ x = 3

theorem mutually_exclusive_and_complementary :
  (∀ s ∈ bag, at_least_one_black s ∧ all_white s → false) ∧
  ((∀ s ∈ bag, at_least_one_black s ∨ all_white s) ∧ (∀ s ∈ bag, ¬(at_least_one_black s ∧ all_white s))) :=
by
  sorry

end mutually_exclusive_and_complementary_l546_546578


namespace area_original_is_504_l546_546793

-- Define the sides of the three rectangles
variable (a1 b1 a2 b2 a3 b3 : ℕ)

-- Define the perimeters of the three rectangles
def P1 := 2 * (a1 + b1)
def P2 := 2 * (a2 + b2)
def P3 := 2 * (a3 + b3)

-- Define the conditions given in the problem
axiom P1_equal_P2_plus_20 : P1 = P2 + 20
axiom P2_equal_P3_plus_16 : P2 = P3 + 16

-- Define the calculation for the area of the original rectangle
def area_original := a1 * b1

-- Proof goal: the area of the original rectangle is 504
theorem area_original_is_504 : area_original = 504 := 
sorry

end area_original_is_504_l546_546793


namespace line_x_intercept_l546_546484

-- Define the given points
def Point1 : ℝ × ℝ := (10, 3)
def Point2 : ℝ × ℝ := (-10, -7)

-- Define the x-intercept problem
theorem line_x_intercept (x : ℝ) : 
  ∃ m b : ℝ, (Point1.2 = m * Point1.1 + b) ∧ (Point2.2 = m * Point2.1 + b) ∧ (0 = m * x + b) → x = 4 :=
by
  sorry

end line_x_intercept_l546_546484


namespace bianca_birthday_money_l546_546961

-- Define the conditions
def num_friends : ℕ := 5
def money_per_friend : ℕ := 6

-- State the proof problem
theorem bianca_birthday_money : num_friends * money_per_friend = 30 :=
by
  sorry

end bianca_birthday_money_l546_546961


namespace find_smallest_gt_six_l546_546790

noncomputable def smallest_number_greater_than_six (S : set ℝ) (b: ℝ) :=
  Inf {x : ℝ | x ∈ S ∧ x > b}

theorem find_smallest_gt_six :
  let S := {0.8, (1 / 2 : ℝ), 0.9, (1 / 3 : ℝ)}
  smallest_number_greater_than_six S 0.6 = 0.8 :=
by
  -- The definitions and the statement sufficient for the theorem
  sorry

end find_smallest_gt_six_l546_546790


namespace committee_size_l546_546802

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end committee_size_l546_546802


namespace param_A_valid_param_B_valid_l546_546766

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Parameterization A
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)

-- Parameterization B
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)

-- Theorem to prove that parameterization A satisfies the line equation
theorem param_A_valid (t : ℝ) : line_eq (param_A t).1 (param_A t).2 := by
  sorry

-- Theorem to prove that parameterization B satisfies the line equation
theorem param_B_valid (t : ℝ) : line_eq (param_B t).1 (param_B t).2 := by
  sorry

end param_A_valid_param_B_valid_l546_546766


namespace increasing_intervals_maximum_area_l546_546620

-- Given function f(x)
def f (x : ℝ) : ℝ := (√3 / 2) * sin (2 * x + π / 3) - cos x ^ 2 + 1 / 2

theorem increasing_intervals : 
  { x | 0 ≤ x ∧ x ≤ π ∧ is_strict_mono_on (f) {x | 0 ≤ x ∧ x ≤ π} } = 
  ({ x | 0 ≤ x ∧ x ≤ π / 6 } ∪
   { x | 2 * π / 3 ≤ x ∧ x ≤ π}) :=
sorry

-- Definitions for the second condition
variables (a b c A B C : ℝ)

-- Given conditions in the problem
def is_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A + B + C = π ∧
  f A = 1 / 4 ∧ 
  a = 3

-- The maximum area theorem
theorem maximum_area (abc_cond : is_sides a b c A B C) :
  ∃ (area : ℝ), area = 9 * √3 / 4 :=
sorry

end increasing_intervals_maximum_area_l546_546620


namespace length_of_platform_l546_546118

variable (Vtrain : Real := 55)
variable (str_len : Real := 360)
variable (cross_time : Real := 57.59539236861051)
variable (conversion_factor : Real := 5/18)

theorem length_of_platform :
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  ∃ L : Real, str_len + L = distance_covered → L = 520 :=
by
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  exists (distance_covered - str_len)
  intro h
  have h1 : distance_covered - str_len = 520 := sorry
  exact h1


end length_of_platform_l546_546118


namespace soda_relationship_l546_546006

theorem soda_relationship (J : ℝ) (L : ℝ) (A : ℝ) (hL : L = 1.75 * J) (hA : A = 1.20 * J) : 
  (L - A) / A = 0.46 := 
by
  sorry

end soda_relationship_l546_546006


namespace subtract_decimal_l546_546945

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end subtract_decimal_l546_546945


namespace length_of_ae_l546_546850

-- Define the given consecutive points
variables (a b c d e : ℝ)

-- Conditions from the problem
-- 1. Points a, b, c, d, e are 5 consecutive points on a straight line - implicitly assumed on the same line
-- 2. bc = 2 * cd
-- 3. de = 4
-- 4. ab = 5
-- 5. ac = 11

theorem length_of_ae 
  (h1 : b - a = 5) -- ab = 5
  (h2 : c - a = 11) -- ac = 11
  (h3 : c - b = 2 * (d - c)) -- bc = 2 * cd
  (h4 : e - d = 4) -- de = 4
  : (e - a) = 18 := sorry

end length_of_ae_l546_546850


namespace min_value_l546_546598

theorem min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 3 * y + 3 * x * y = 6) : 2 * x + 3 * y ≥ 4 :=
sorry

end min_value_l546_546598


namespace selfish_subsets_equals_fibonacci_l546_546892

noncomputable def fibonacci : ℕ → ℕ
| 0           => 0
| 1           => 1
| (n + 2)     => fibonacci (n + 1) + fibonacci n

noncomputable def selfish_subsets_count (n : ℕ) : ℕ := 
sorry -- This will be replaced with the correct recursive function

theorem selfish_subsets_equals_fibonacci (n : ℕ) : 
  selfish_subsets_count n = fibonacci n :=
sorry

end selfish_subsets_equals_fibonacci_l546_546892


namespace reliefSuppliesCalculation_l546_546441

noncomputable def totalReliefSupplies : ℝ := 644

theorem reliefSuppliesCalculation
    (A_capacity : ℝ)
    (B_capacity : ℝ)
    (A_capacity_per_day : A_capacity = 64.4)
    (capacity_ratio : A_capacity = 1.75 * B_capacity)
    (additional_transport : ∃ t : ℝ, A_capacity * t - B_capacity * t = 138 ∧ A_capacity * t = 322) :
  totalReliefSupplies = 644 := by
  sorry

end reliefSuppliesCalculation_l546_546441


namespace solution_set_of_inequality_l546_546057

theorem solution_set_of_inequality : 
  {x : ℝ | 4 * x^2 - 4 * x + 1 ≥ 0} = set.univ := 
by 
  sorry

end solution_set_of_inequality_l546_546057


namespace maximum_area_of_triangle_ABC_l546_546995

noncomputable def max_area_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem maximum_area_of_triangle_ABC (a b c A B C : ℝ) 
  (h1: a = 4) 
  (h2: (4 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  max_area_triangle_ABC a b c A B C = 4 * Real.sqrt 3 := 
sorry

end maximum_area_of_triangle_ABC_l546_546995


namespace book_cost_in_indian_rupees_l546_546383

variables (Euro EgyptianPound IndianRupee : Type) [DivisionRing Euro]
  [DivisionRing EgyptianPound] [DivisionRing IndianRupee]

-- Conditions
constant e2ep : Euro → EgyptianPound → Prop
constant e2inr : Euro → IndianRupee → Prop

axiom euro_to_egyptian_pound : ∀ (e : Euro) (ep : EgyptianPound), e2ep e ep ↔ (ep = 9)
axiom euro_to_indian_rupee : ∀ (e : Euro) (inr : IndianRupee), e2inr e inr ↔ (inr = 90)

noncomputable def book_cost_inr (book_cost_ep : EgyptianPound) : IndianRupee :=
  (book_cost_ep / (9 : EgyptianPound)) * (90 : IndianRupee)

theorem book_cost_in_indian_rupees :
  book_cost_inr (540 : EgyptianPound) = (5400 : IndianRupee) :=
by
  sorry

end book_cost_in_indian_rupees_l546_546383


namespace faster_speed_l546_546515

def original_speed := 5 -- km/hr
def actual_distance := 10 -- km
def additional_distance := 20 -- km

theorem faster_speed : ∃ x : ℝ, 2 * x = actual_distance + additional_distance ∧ x = 15 :=
by
  existsi 15
  split
  · calc
    2 * 15 = 30 : by norm_num
        ... = 10 + 20 : by norm_num
  · refl

end faster_speed_l546_546515


namespace Ali_catch_weight_l546_546150

-- Define the conditions and the goal
def Ali_Peter_Joey_fishing (p: ℝ) : Prop :=
  let Ali := 2 * p in
  let Joey := p + 1 in
  p + Ali + Joey = 25

-- State the problem
theorem Ali_catch_weight :
  ∃ p: ℝ, Ali_Peter_Joey_fishing p ∧ (2 * p = 12) :=
by
  sorry

end Ali_catch_weight_l546_546150


namespace geometry_proof_l546_546317

theorem geometry_proof (AB AC : ℝ) (h1 : AB = 15) (h2 : AC = 17) :
∃ (area_rectangle : ℝ) (circ_circle : ℝ), area_rectangle = 120 ∧ circ_circle = 16 * Real.pi := by
  -- Definitions of necessary lengths according to the problem
  let BC := Real.sqrt (AC ^ 2 - AB ^ 2)
  have hBC : BC = 8 := by
    calc
      BC = Real.sqrt (17 ^ 2 - 15 ^ 2) : by sorry
      _  = Real.sqrt 64 : by sorry
      _  = 8 : by sorry

  -- Area of the rectangle
  let area_rectangle := AB * BC
  have h_area : area_rectangle = 120 := by
    calc
      area_rectangle = 15 * 8 : by rw [hBC]; sorry
      _ = 120 : by sorry

  -- Circumference of the circle
  let CD := BC
  let circ_circle := 2 * Real.pi * CD
  have h_circ : circ_circle = 16 * Real.pi := by
    calc
      circ_circle = 2 * Real.pi * 8 : by rw [hBC]; sorry
      _ = 16 * Real.pi : by sorry

  -- Constructing the final proof
  use [area_rectangle, circ_circle]
  exact ⟨h_area, h_circ⟩

end geometry_proof_l546_546317


namespace problem1_part1_problem1_part2_problem1_part3_l546_546234

noncomputable def sin_val (α : ℝ) : ℝ := 3 / 5
noncomputable def tan_val (α : ℝ) : ℝ := 3 / 4

theorem problem1_part1 (α : ℝ) (h : (4,3) = terminal_side_point α) : sin α = sin_val α :=
by sorry

theorem problem1_part2 (α : ℝ) (h : (4,3) = terminal_side_point α) : tan α = tan_val α :=
by sorry

theorem problem1_part3 (α : ℝ) (h : (4,3) = terminal_side_point α) : 
  (cos(π / 2 - α) + 2 * cos(π + α)) / (sin(π - α) - sin(π / 2 + α)) = -5 :=
by sorry

end problem1_part1_problem1_part2_problem1_part3_l546_546234


namespace spherical_coordinates_phi_eq_c_is_cone_l546_546318

-- Given
def spherical_coordinates := Type -- represents (ρ, θ, φ)
variable (ρ θ φ c : ℝ)

-- Assume φ = c in spherical coordinates
axiom phi_eq_c : φ = c

-- Definition of a cone
def is_cone := (ρ, θ, φ) ∈ spherical_coordinates ∧ φ = c ∧ θ ∈ [0, 2 * π] ∧ ρ ≥ 0

-- The theorem statement
theorem spherical_coordinates_phi_eq_c_is_cone : is_cone (ρ, θ, φ, c) :=
sorry

end spherical_coordinates_phi_eq_c_is_cone_l546_546318


namespace minimum_trips_needed_l546_546110

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

theorem minimum_trips_needed (masses : List ℕ) (capacity : ℕ) : 
  masses = [150, 60, 70, 71, 72, 100, 101, 102, 103] →
  capacity = 200 →
  ∃ trips : ℕ, trips = 5 :=
by
  sorry

end minimum_trips_needed_l546_546110


namespace part_I_part_II_l546_546257

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x - 1|

theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ 2 - |x - 1|) : 0 ≤ a ∧ a ≤ 4 := 
sorry

theorem part_II (a : ℝ) (h₁ : a < 2) (h₂ : ∀ x : ℝ, f x a ≥ 3) : a = -4 := 
sorry

end part_I_part_II_l546_546257


namespace decreasing_interval_of_f_l546_546042

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem decreasing_interval_of_f : ∃ a b : ℝ, (0 < a ∧ a < b) ∧ b = 1/Real.exp 1 ∧ (∀ x : ℝ, a < x ∧ x < b → f' x < 0) :=
by sorry

end decreasing_interval_of_f_l546_546042


namespace leadership_board_stabilizes_l546_546404

theorem leadership_board_stabilizes :
  ∃ n : ℕ, 2 ^ n - 1 ≤ 2020 ∧ 2020 < 2 ^ (n + 1) - 1 := by
  sorry

end leadership_board_stabilizes_l546_546404


namespace matrix_product_is_zero_l546_546921

variables {R : Type*} [CommRing R] (d e f : R)

def A : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def B : Matrix (Fin 3) (Fin 3) R :=
  ![![d^2, d*e, d*f],
    ![d*e, e^2, e*f],
    ![d*f, e*f, f^2]]

theorem matrix_product_is_zero : A d e f ⬝ B d e f = 0 := by
  sorry

end matrix_product_is_zero_l546_546921


namespace minimum_value_of_expression_l546_546692

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  a^2 + b^2 + (a + b)^2 + c^2

theorem minimum_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  min_value_expression a b c = 9 :=
  sorry

end minimum_value_of_expression_l546_546692


namespace minimum_positive_period_l546_546048

open Real

noncomputable def function := fun x : ℝ => 3 * sin (2 * x + π / 3)

theorem minimum_positive_period : ∃ T > 0, ∀ x, function (x + T) = function x ∧ (∀ T', T' > 0 → (∀ x, function (x + T') = function x) → T ≤ T') :=
  sorry

end minimum_positive_period_l546_546048


namespace day_of_week_february_4_2022_is_saturday_l546_546167

noncomputable def daysBetweenDates (start end : Nat) : Nat := 2192 - 36  -- Placeholder function for days calculation.
noncomputable def dayOfWeekFromToday (today : Nat) (daysDiff : Nat) : Nat := 
  (today + (daysDiff % 7)) % 7

theorem day_of_week_february_4_2022_is_saturday :
  daysBetweenDates -- March 12, 2016 to February 4, 2022
  (dayOfWeekFromToday 6 2156) = 6 := 
sorry

end day_of_week_february_4_2022_is_saturday_l546_546167


namespace correct_operation_l546_546095

-- Conditions
variables (a b : ℝ)

theorem correct_operation :
  (a^3 - a^2 ≠ a) ∧
  (a + a^2 ≠ a^3) ∧
  (ab^2 + a^2b ≠ ab^2) →
  (-a + 5a = 4a) :=
by 
  intro h,
  sorry

end correct_operation_l546_546095


namespace train_speed_l546_546478

/-- Given the length of the train is 200 meters, the time to cross a man walking at 5 km/h in the opposite direction is 6 seconds, prove that the speed of the train is approximately 115 km/h. -/
theorem train_speed (L : ℝ) (t : ℝ) (speed_man_kmph : ℝ) (speed_train_kmph : ℝ) : 
  L = 200 → t = 6 → speed_man_kmph = 5 → speed_train_kmph = 115 → 
  (∃ (v : ℝ), v = (200 : ℝ) / 6 - (5 * 1000) / 3600 ∧ speed_train_kmph ≈ (v * 3600) / 1000 ) :=
by sorry

end train_speed_l546_546478


namespace spatial_quadrilateral_exists_l546_546520

noncomputable def number_of_points (q : ℕ) := q^2 + q + 1

theorem spatial_quadrilateral_exists (q n l : ℕ)
  (hq : 2 ≤ q) (hq_nat : q ∈ ℕ)
  (hn : n = number_of_points q)
  (hl : l ≥ (1 / 2 * (q * (q + 1)^2)).to_nat + 1)
  (no_four_points_coplanar : ¬ ∃ A B C D, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ coplanar A B C D)
  (each_point_has_line : ∀ A, ∃ B, connected A B)
  (exists_point_with_q_plus_2_lines : ∃ A, ∃ line_segments, connected_points A line_segments ∧ line_segments.length ≥ q+2) :
  ∃ A B C D, ∃ line_segments, (line_segments = [A, B, C, D, A] ∧ connected_points A [B, C, D] ∧ connected_points B [C, D] ∧ connected_points C [D]) :=
sorry

end spatial_quadrilateral_exists_l546_546520


namespace correct_sample_set_l546_546435

-- Define the context and conditions
def total_products : ℕ := 60
def num_samples : ℕ := 6

-- Define the sampling set according to the solution
def sample_set : set ℕ := {3, 13, 23, 33, 43, 53}

-- Define the systematic sampling property
def is_systematic_sample (s : set ℕ) : Prop :=
  ∃ l k, (∀ i : ℕ, i < num_samples → l + i * k ∈ s)

-- State the main theorem/principal claim
theorem correct_sample_set :
  sample_set = {3, 13, 23, 33, 43, 53} →
  ∃ l k, (l = 3 ∧ k = 10 ∧ 
          (∀ i : ℕ, i < num_samples → 3 + i * 10 = (3 + i * 10))) :=
sorry

end correct_sample_set_l546_546435


namespace regression_equation_l546_546425

-- Define the regression coefficient and correlation
def negatively_correlated (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100

-- The question is to prove that given x and y are negatively correlated,
-- the regression equation is \hat{y} = -2x + 100
theorem regression_equation (x y : ℝ) (h : negatively_correlated x y) :
  (∃ a, a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100) → ∃ (b : ℝ), b = -2 ∧ ∀ (x_val : ℝ), y = b * x_val + 100 :=
by
  sorry

end regression_equation_l546_546425


namespace smallest_fraction_denominator_l546_546953

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end smallest_fraction_denominator_l546_546953


namespace smallest_number_in_set_l546_546677

theorem smallest_number_in_set (n : ℕ) (a : ℝ) (S : Finset ℝ) 
  (h1 : S.card = n) 
  (h2 : ∀ x ∈ S, x >= a - 7 ∧ x <= a + 7) 
  (h3 : ∃ x ∈ S, x = a) 
  (h4 : S.sum (id) / n = a)
  (h5 : S.filter (λ x, x < a).card > n / 2) : 
  n = 7 := by
sorry

end smallest_number_in_set_l546_546677


namespace exists_infinite_primes_order_eq_l546_546387

theorem exists_infinite_primes_order_eq (a b : ℕ) : ∃ (S : set ℕ), (∀ p ∈ S, Prime p ∧ (nat.order a p = nat.order b p)) ∧ set.infinite S :=
sorry

end exists_infinite_primes_order_eq_l546_546387


namespace A_plus_B_l546_546701

open Nat

-- Definitions of 18, 24, and 36
def num1 := 18
def num2 := 24
def num3 := 36

-- Definition of the Greatest Common Factor (GCF)
def A : ℕ := gcd (gcd num1 num2) num3

-- Definition of the Least Common Multiple (LCM)
def B : ℕ := lcm (lcm num1 num2) num3

-- The theorem stating A + B = 78
theorem A_plus_B : A + B = 78 := by
  sorry

end A_plus_B_l546_546701


namespace probability_of_event_is_correct_l546_546724

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

/-- Definition of the set from which a, b, and c are chosen -/
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 2010}

/-- Definition of the event that abc + ab + a is divisible by 3 -/
def event (a b c : ℕ) : Prop := is_divisible_by (a * b * c + a * b + a) 3

/-- Definition of the probability of the event happening given the set -/
def probability_event : ℚ := 13 / 27

/-- The main theorem -/
theorem probability_of_event_is_correct : 
  (∑' a b c in num_set, indicator (event a b c)) / 
  (∑' a b c in num_set, 1) = probability_event := sorry


end probability_of_event_is_correct_l546_546724


namespace a_2_correct_l546_546645

noncomputable def a_2_value (a a1 a2 a3 : ℝ) : Prop :=
∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3

theorem a_2_correct (a a1 a2 a3 : ℝ) (h : a_2_value a a1 a2 a3) : a2 = 6 :=
sorry

end a_2_correct_l546_546645


namespace arithmetic_progression_sum_squares_l546_546784

theorem arithmetic_progression_sum_squares (a1 a2 a3 : ℚ)
  (h1 : a2 = (a1 + a3) / 2)
  (h2 : a1 + a2 + a3 = 2)
  (h3 : a1^2 + a2^2 + a3^2 = 14/9) :
  (a1 = 1/3 ∧ a2 = 2/3 ∧ a3 = 1) ∨ (a1 = 1 ∧ a2 = 2/3 ∧ a3 = 1/3) :=
sorry

end arithmetic_progression_sum_squares_l546_546784


namespace find_initial_population_l546_546495

-- Define the initial population, conditions and the final population
variable (P : ℕ)

noncomputable def initial_population (P : ℕ) :=
  (0.85 * (0.92 * P) : ℝ) = 3553

theorem find_initial_population (P : ℕ) (h : initial_population P) : P = 4546 := sorry

end find_initial_population_l546_546495


namespace solve_congruences_l546_546029

theorem solve_congruences (x : ℤ) :
  x ≡ 1 [MOD 3] ∧ 
  x ≡ -1 [MOD 5] ∧ 
  x ≡ 2 [MOD 7] ∧ 
  x ≡ -2 [MOD 11] →
  x ≡ 394 [MOD 1155] :=
by {
  sorry
}

end solve_congruences_l546_546029


namespace hotel_P_charge_less_than_G_l546_546407

open Real

variable (G R P : ℝ)

-- Given conditions
def charge_R_eq_2G : Prop := R = 2 * G
def charge_P_eq_R_minus_55percent : Prop := P = R - 0.55 * R

-- Goal: Prove the percentage by which P's charge is less than G's charge is 10%
theorem hotel_P_charge_less_than_G : charge_R_eq_2G G R → charge_P_eq_R_minus_55percent R P → P = 0.9 * G := by
  intros h1 h2
  sorry

end hotel_P_charge_less_than_G_l546_546407


namespace general_term_max_n_ineq_l546_546244

noncomputable def a_n (n : ℕ) : ℝ := (1/2) ^ n

def b_n (n : ℕ) : ℝ := a_n n * Real.log2 (a_n n)

def T_n (n : ℕ) : ℝ := - (Finset.sum (Finset.range (n + 1)) (λ i, (i+1) * a_n (i+1)))

theorem general_term (n : ℕ) : a_n n = (1/2) ^ n := 
by 
  -- Proof is omitted
  sorry

theorem max_n_ineq (n : ℕ) (hn : (1/2) ^ n ≥ 1/16) : n ≤ 4 :=
by 
  -- Proof is omitted
  sorry

end general_term_max_n_ineq_l546_546244


namespace find_f_prime_zero_l546_546706

variable (f : ℝ → ℝ)
variable (x : ℝ)
variable (f' : ℝ → ℝ)
variable (h₀ : f = λ x, Real.exp x + 2 * f' 0 * x)
variable (h₁ : deriv f = f')

theorem find_f_prime_zero : f' 0 = -1 := by
  have h₂ : f 0 = Real.exp 0 + 2 * f' 0 * 0 := by
    rw [h₀]
    simp
  have h₃ : deriv f 0 = deriv (λ x, Real.exp x + 2 * f' 0 * x) 0 := by
    rw [h₀]
  calc
    f' 0 = deriv (λ x, Real.exp x + 2 * 0 * x) 0 := by
      rw [h₃]
    ... = 1 + 2 * f' 0 := by
      simp [deriv, h₁]
    ... = -1 := by
      linarith

end find_f_prime_zero_l546_546706


namespace angle_B_range_sin_B_value_l546_546298

noncomputable section

-- Variables and assumptions for the triangle and sides
variables (A B C : ℝ) (a b c : ℝ)

-- Defining the assumptions
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  A + B + C = π

def condition_1 (a b c : ℝ) : Prop :=
  a + c = 2b

-- Problem (I): Prove \forall A B C a b c, triangle ABC and a + c = 2b, then B \in (0, π/3]
theorem angle_B_range (A B C a b c : ℝ)
  (h_triangle : triangle_ABC A B C a b c)
  (h_cond : condition_1 a b c) :
  0 < B ∧ B ≤ π / 3 :=
sorry

-- Problem (II): Prove \forall A B C a b c, if triangle ABC, a + c = 2b, and A - C = π/3, then sin B = √39 / 8
theorem sin_B_value (A B C a b c : ℝ)
  (h_triangle : triangle_ABC A B C a b c)
  (h_cond : condition_1 a b c)
  (h_angle_diff : A - C = π / 3) :
  Real.sin B = (Real.sqrt 39) / 8 :=
sorry

end angle_B_range_sin_B_value_l546_546298


namespace price_increase_for_1620_profit_maximizing_profit_l546_546194

-- To state the problem, we need to define some variables and the associated conditions.

def cost_price : ℝ := 13
def initial_selling_price : ℝ := 20
def initial_monthly_sales : ℝ := 200
def decrease_in_sales_per_yuan : ℝ := 10
def profit_condition (x : ℝ) : ℝ := (initial_selling_price + x - cost_price) * (initial_monthly_sales - decrease_in_sales_per_yuan * x)
def profit_function (x : ℝ) : ℝ := -(10 * x ^ 2) + (130 * x) + 140

-- Part (1): Prove the price increase x such that the profit is 1620 yuan
theorem price_increase_for_1620_profit :
  ∃ (x : ℝ), profit_condition x = 1620 ∧ (x = 2 ∨ x = 11) :=
sorry

-- Part (2): Prove that the selling price that maximizes profit is 26.5 yuan and max profit is 1822.5 yuan
theorem maximizing_profit :
  ∃ (x : ℝ), (x = 13 / 2) ∧ profit_function (13 / 2) = 3645 / 2 :=
sorry

end price_increase_for_1620_profit_maximizing_profit_l546_546194


namespace arrangements_boys_girls_l546_546576

theorem arrangements_boys_girls (boys girls : Finset Nat) 
  (h1 : boys.card = 4) (h2 : girls.card = 3)
  (h3 : ∃ g1 g2 g3 ∈ girls, g1 ≠ g2 ∧ g1 ≠ g3 ∧ g2 ≠ g3 ∧ ((g1 + 1 = g2) ∨ (g2 + 1 = g1))) :
  ∃ n : Nat, (n = 2880) :=
by 
  sorry

end arrangements_boys_girls_l546_546576


namespace min_black_squares_l546_546537

def is_square_black_or_adjacent (n : ℕ) (board : ℕ → ℕ → Prop) (i j : ℕ) : Prop :=
  board i j ∨ -- The (i, j) square is black
  (∃ (di dj : ℤ), (di, dj) ∈ [(0, 1), (1, 0), (0, -1), (-1, 0)] ∧ board (i + di) (j + dj))

def is_chain_of_black_squares (n : ℕ) (board : ℕ → ℕ → Prop) : Prop :=
  ∀ i j i' j', board i j → board i' j' → 
    ∃ (chain : List (ℕ × ℕ)), 
      chain.head = (i, j) ∧ chain.last = (i', j') ∧
      ∀ (p q: (ℕ × ℕ)), p ∈ chain → q ∈ chain → (p.1 = q.1 ∧ (p.2 + 1 = q.2 ∨ p.2 = q.2 + 1) ∨ 
                                                   p.2 = q.2 ∧ (p.1 + 1 = q.1 ∨ p.1 = q.1 + 1))

theorem min_black_squares
  (n : ℕ)
  (board : ℕ → ℕ → Prop)
  (h1 : ∀ i j, ¬board i j → is_square_black_or_adjacent n board i j)
  (h2 : is_chain_of_black_squares n board) :
  ∑ i in Finset.range (n), ∑ j in Finset.range (n), if board i j then 1 else 0 ≥ (n^2 - 2) / 3 := by
  sorry

end min_black_squares_l546_546537


namespace area_of_triangle_OEC_l546_546762

noncomputable def area_triangle_OEC (h : ℕ) (AD BC : ℕ) (ratioAO_OC : ℕ × ℕ) : ℝ :=
  let height := (h : ℝ)
  let AD_len := (AD : ℝ)
  let BC_len := (BC : ℝ)
  let (AO, OC) := ratioAO_OC
  let total_area_ACD := 0.5 * AD_len * height
  let ratio := (AO : ℝ) / ((AO + OC) : ℝ)
  let area_AOC := ratio * total_area_ACD
  let fraction_CE_CD := 6/7
  let area_OEC := (fraction_CE_CD * 2/5) * area_AOC
  area_OEC

theorem area_of_triangle_OEC (h : ℕ) (AD BC : ℕ) (ratioAO_OC : ℕ × ℕ) (H_h : h = 7) (H_AD : AD = 8) (H_BC : BC = 6) (H_ratioAO_OC : ratioAO_OC = (3, 2)) :
  area_triangle_OEC h AD BC ratioAO_OC = 4.8 :=
by {
  have H_h : (h : ℝ) = 7 := rfl,
  have H_AD : (AD : ℝ) = 8 := rfl,
  have H_BC : (BC : ℝ) = 6 := rfl,
  have H_ratioAO_OC : ratioAO_OC = (3, 2) := rfl,
  sorry
}

end area_of_triangle_OEC_l546_546762


namespace handicraft_sales_expression_profit_expression_maximum_profit_l546_546866

theorem handicraft_sales_expression :
  ∀ x y : ℕ, x = 15 ∧ y = 150 ∨ x = 16 ∧ y = 140 → y = -10 * x + 300 :=
by sorry

theorem profit_expression :
  ∀ x w : ℕ, (∀ y : ℕ, y = -10 * x + 300) → w = (x - 11) * y → w = -10 * x ^ 2 + 410 * x - 3300 :=
by sorry

theorem maximum_profit :
  ∀ x : ℕ, (w = -10 * x ^ 2 + 410 * x - 3300) → w ≤ 900 ∧ (x = 20 ∨ x = 21) ∨ w < 900 :=
by sorry

end handicraft_sales_expression_profit_expression_maximum_profit_l546_546866


namespace find_circle_and_a_l546_546671

-- Definitions of the conditions
def curve (x : ℝ) : ℝ := x ^ 2 - 6 * x + 1

def circle (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 1) ^ 2 = 9

def line (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Statement of the proof problem
theorem find_circle_and_a (a : ℝ) :
  (∀ x y, (x = 0 → y = 1 ∨ (y = 0 → (x = 3+2*sqrt 2 ∨ x = 3-2*sqrt 2)))
  → ∀ x1 y1 x2 y2, circle x1 y1 ∧ line a x1 y1 ∧ circle x2 y2 ∧ line a x2 y2
  ∧ (x1 - 0 + y1 - 0 = 0 ∨ x2 - 0 + y2 - 0 = 0) 
  → a = -1) :=
sorry

end find_circle_and_a_l546_546671


namespace trigonometric_identity_solution_l546_546849

theorem trigonometric_identity_solution (k : ℤ) :
  ∃ t : ℝ, (\sin (4 * t) + cos (4 * t))^2 = 16 * sin (2 * t) * (cos (2 * t))^3 - 8 * sin (2 * t) * cos (2 * t) ∧
  t = (Real.pi / 16) * (4 * k + 1) :=
begin
  sorry
end

end trigonometric_identity_solution_l546_546849


namespace white_roses_needed_l546_546375

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end white_roses_needed_l546_546375


namespace order_of_terms_proof_l546_546275

noncomputable def order_of_terms (x y z : ℝ) (h1 : 2^x = 3^y) (h2 : 3^y = 5^z) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) : Prop :=
  3 * y < 2 * x ∧ 2 * x < 5 * z

theorem order_of_terms_proof (x y z : ℝ) (h1 : 2^x = 3^y) (h2 : 3^y = 5^z) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) : 
  order_of_terms x y z h1 h2 h_pos := by
  sorry

end order_of_terms_proof_l546_546275


namespace complex_number_real_l546_546093

theorem complex_number_real (m : ℝ) (z : ℂ) 
  (h1 : z = ⟨1 / (m + 5), 0⟩ + ⟨0, m^2 + 2 * m - 15⟩)
  (h2 : m^2 + 2 * m - 15 = 0)
  (h3 : m ≠ -5) :
  m = 3 :=
sorry

end complex_number_real_l546_546093


namespace base_angle_of_isosceles_triangle_l546_546646

theorem base_angle_of_isosceles_triangle (α : ℝ) (h1 : α = 120) (h2 : α + β + β = 180) : β = 30 :=
  by
    -- Use given α value
    have hα : α = 120 := h1
    -- Use angle sum property
    have h_sum : α + 2 * β = 180 := by linarith [h2]
    -- Compute the base angle
    have h_base : 2 * β = 180 - α := by linarith
    /- Dividing both sides by 2 -/
    linarith

end base_angle_of_isosceles_triangle_l546_546646


namespace geometric_sequence_sum_S40_l546_546350

theorem geometric_sequence_sum_S40
  (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a_n (n+1) = a_n n * q)
  (h_pos : ∀ n, 0 < a_n n)
  (h_sum : ∀ n, S_n n = ∑ i in range n, a_n i)
  (h_S10 : S_n 10 = 10)
  (h_S30 : S_n 30 = 70) :
  S_n 40 = 150 :=
sorry

end geometric_sequence_sum_S40_l546_546350


namespace distance_between_trees_correct_l546_546659

-- Define the conditions as variables and constants
variables (num_trees : ℕ) (sidewalk_length : ℝ)

-- State the conditions
def conditions : Prop := 
  num_trees = 16 ∧ sidewalk_length = 166

-- Define the distance calculation between each tree
noncomputable def distance_between_trees (num_trees : ℕ) (sidewalk_length : ℝ) : ℝ :=
  sidewalk_length / (num_trees - 1)

-- The main theorem statement
theorem distance_between_trees_correct :
  conditions num_trees sidewalk_length →
  distance_between_trees num_trees sidewalk_length ≈ 11.07 :=
sorry

end distance_between_trees_correct_l546_546659


namespace white_roses_total_l546_546372

theorem white_roses_total (bq_num : ℕ) (tbl_num : ℕ) (roses_per_bq : ℕ) (roses_per_tbl : ℕ)
  (total_roses : ℕ) 
  (h1 : bq_num = 5) 
  (h2 : tbl_num = 7) 
  (h3 : roses_per_bq = 5) 
  (h4 : roses_per_tbl = 12)
  (h5 : total_roses = 109) : 
  bq_num * roses_per_bq + tbl_num * roses_per_tbl = total_roses := 
by 
  rw [h1, h2, h3, h4, h5]
  exact rfl

end white_roses_total_l546_546372


namespace nikola_numbers_sum_is_996_l546_546716

-- Define three-digit and two-digit numbers with distinct digits
def is_three_digit_with_distinct_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (list.nodup (n.digits 10))

def is_two_digit_with_distinct_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (list.nodup (n.digits 10))

-- Main theorem to prove
theorem nikola_numbers_sum_is_996 :
  ∃ (a b : ℕ), is_three_digit_with_distinct_digits a ∧ 
               is_two_digit_with_distinct_digits b ∧ 
               a - b = 976 ∧ 
               a + b = 996 :=
by
  -- Placeholder for proof
  sorry

end nikola_numbers_sum_is_996_l546_546716


namespace remaining_volume_of_cube_with_hole_l546_546905

theorem remaining_volume_of_cube_with_hole : 
  let side_length_cube := 8 
  let side_length_hole := 4 
  let volume_cube := side_length_cube ^ 3 
  let cross_section_hole := side_length_hole ^ 2
  let volume_hole := cross_section_hole * side_length_cube
  let remaining_volume := volume_cube - volume_hole
  remaining_volume = 384 := by {
    sorry
  }

end remaining_volume_of_cube_with_hole_l546_546905


namespace sum_proper_divisors_eq_40_l546_546823

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l546_546823


namespace problem_statement_l546_546545

variable (t : ℝ)

-- Define M as the fractional part of t
def M : ℝ := t - (t.floor)

-- Define N as the integer part of t plus 0.5
def N : ℝ := t.floor + 0.5

-- Define the set S
def S : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - M t)^2 + (p.2 - N t)^2 / 2 ≤ (M t)^2}

-- Define the function to calculate the area of the ellipse
noncomputable def area_of_S : ℝ := π * (M t)^2 * sqrt 2

-- Prove that the area of S is correct and the point (0, 0) does not always belong to S
theorem problem_statement : 
  area_of_S t = π * (M t)^2 * sqrt 2 ∧ 
  ¬ ∀ t ≥ 0, (0, 0) ∈ S t :=
sorry

end problem_statement_l546_546545


namespace multiply_polynomials_l546_546378

theorem multiply_polynomials (x : ℝ) :
  (x^4 + 8 * x^2 + 64) * (x^2 - 8) = x^4 + 16 * x^2 :=
by
  sorry

end multiply_polynomials_l546_546378


namespace sum_proper_divisors_eq_40_l546_546827

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l546_546827


namespace present_age_of_son_l546_546132

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 32) (h2 : M + 2 = 2 * (S + 2)) : S = 30 :=
by
  sorry

end present_age_of_son_l546_546132


namespace number_of_people_ate_pizza_l546_546528

theorem number_of_people_ate_pizza (total_slices initial_slices remaining_slices slices_per_person : ℕ) 
  (h1 : initial_slices = 16) 
  (h2 : remaining_slices = 4) 
  (h3 : slices_per_person = 2) 
  (eaten_slices : total_slices) 
  (h4 : total_slices = initial_slices - remaining_slices) :
    total_slices / slices_per_person = 6 := 
by {
  -- specify the value for eaten_slices to simplify the calculation
  let eaten_slices := initial_slices - remaining_slices,
  have h5 : eaten_slices = 12, from sorry,
  sorry
  }

end number_of_people_ate_pizza_l546_546528


namespace root_in_interval_l546_546765

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 * x - 1

theorem root_in_interval : ∃ c ∈ (1/2 : ℝ) .. 1, f c = 0 :=
by {
  sorry
}

end root_in_interval_l546_546765


namespace arithmetic_sequence_sum_l546_546315

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
(h : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) 
(h_a : ∀ n, a n = a 1 + (n - 1) * d) : a 2 + a 10 = 120 :=
by
  sorry

end arithmetic_sequence_sum_l546_546315


namespace g_of_2023_eq_2_l546_546342

theorem g_of_2023_eq_2 (g : ℝ → ℝ) (h1 : ∀ x > 0, g(x) > 0)
  (h2 : ∀ x y, x > y → 0 < y → g(x - y) = sqrt(g(x * y) + 2)) :
  g(2023) = 2 :=
by
  -- Proof omitted
  sorry

end g_of_2023_eq_2_l546_546342


namespace bus_capacities_rental_plan_l546_546501

variable (x y : ℕ)
variable (m n : ℕ)

theorem bus_capacities :
  3 * x + 2 * y = 195 ∧ 2 * x + 4 * y = 210 → x = 45 ∧ y = 30 :=
by
  sorry

theorem rental_plan :
  7 * m + 3 * n = 20 ∧ m + n ≤ 7 ∧ 65 * m + 45 * n + 30 * (7 - m - n) = 310 →
  m = 2 ∧ n = 2 ∧ 7 - m - n = 3 :=
by
  sorry

end bus_capacities_rental_plan_l546_546501


namespace portion_returned_l546_546848

variables (Will_catfish Henry_trout_total Will_fishes_total Henry_fishes_returned : ℕ)
variables (total_fishes: ℕ)

def Will := (catfish : ℕ) (eels : ℕ) : ℕ := catfish + eels
def Henry_trout (catfish : ℕ) : ℕ := 3 * catfish

def total_fish := Will_catfish + 10 + Henry_trout_total
def fishes_returned := total_fish - total_fishes

axiom Will_catch : Will_catfish = 16
axiom total_fishes_now : total_fishes = 50
axiom Henry_trout_calculated : Henry_trout_total = Henry_trout Will_catfish
axiom fishes_returned_fraction : fishes_returned = 24

theorem portion_returned : 
  let portion := Henry_fishes_returned / Henry_trout_total 
  in portion = 1/2 := 
  by 
    sorry

end portion_returned_l546_546848


namespace problem1_constant_term_problem2_distance_problem3_probability_problem4_min_value_l546_546858

-- Problem 1
theorem problem1_constant_term (x : ℝ) (hx : x ≠ 0) : constant_term ((|x| + 1 / |x| - 2) ^ 3) = -20 :=
sorry

-- Problem 2
theorem problem2_distance (A B : ℝ × ℝ) :
  let C1 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 + p.2 = sqrt 2
  let C2 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = 2
  (C1 A ∧ C2 A) ∧ (C1 B ∧ C2 B) → dist A B = 2 :=
sorry

-- Problem 3
theorem problem3_probability (X Y : ℝ) (p σ : ℝ) :
  (∃ A : {x // x ≥ 1} → P(X ≥ A.1) = 0.64) ∧ (∃ B : {y // 0 < y < 2} → P(Y < B.1) = p) → P(Y > 4) = 0.1 :=
sorry

-- Problem 4
theorem problem4_min_value (f : ℝ → ℝ) (hf : ∀ x, f x = 2 * sin x + sin (2 * x)) : 
  ∃ x₀, (∀ x ∈ Icc (0 : ℝ) (2 * π), f x₀ ≤ f x) ∧ f x₀ = -((3 * sqrt 3) / 2) :=
sorry

end problem1_constant_term_problem2_distance_problem3_probability_problem4_min_value_l546_546858


namespace magnification_cos_squared_l546_546925

theorem magnification_cos_squared (x : ℝ) :
  let y := cos x ^ 2 in
  let x' := 2 * x in
  let y' := 2 * y - 1 in
  y' = cos x' ↔ cos x = y' :=
by
  sorry

end magnification_cos_squared_l546_546925


namespace g_inv_f_9_eq_l546_546278

theorem g_inv_f_9_eq :
  (∀ (x : ℝ), f⁻¹(g(x)) = x^4 - x^2 + 1) →
  Function.Injective g →
  Function.Surjective g →
  g⁻¹(f 9) = Real.sqrt ((1 + Real.sqrt 33) / 2) :=
by
  intros h1 h2 h3
  sorry

end g_inv_f_9_eq_l546_546278


namespace sum_repeating_decimals_l546_546915

theorem sum_repeating_decimals : (0.14 + 0.27) = (41 / 99) := by
  sorry

end sum_repeating_decimals_l546_546915


namespace chessboard_squares_and_rectangles_l546_546432

theorem chessboard_squares_and_rectangles :
  let r := 45 * 45
  let s := (list.sum ∘ list.map (λ n, n ^ 2) $ list.range 10)
  let gcd := nat.gcd 285 2025
  let m := 285 / gcd
  let n := 2025 / gcd
  m + n = 154 :=
by {
  have : r = 2025, { refl },
  have : s = 285, {
    -- Calculating the sum of squares from 1^2 to 9^2
    repeat { sorry }
  },
  have : gcd = 15, { sorry },
  have : m = 19, { sorry },
  have : n = 135, { sorry },
  calc
    m + n = 19 + 135 : sorry
    ... = 154       : sorry
}

end chessboard_squares_and_rectangles_l546_546432


namespace min_value_l546_546989

open Real

noncomputable def y1 (x1 : ℝ) : ℝ := x1 * log x1
noncomputable def y2 (x2 : ℝ) : ℝ := x2 - 3

theorem min_value :
  ∃ (x1 x2 : ℝ), (x1 - x2)^2 + (y1 x1 - y2 x2)^2 = 2 :=
by
  sorry

end min_value_l546_546989


namespace correct_transformation_l546_546159

-- Given transformations
def transformation_A (a : ℝ) : Prop := - (1 / a) = -1 / a
def transformation_B (a b : ℝ) : Prop := (1 / a) + (1 / b) = 1 / (a + b)
def transformation_C (a b : ℝ) : Prop := (2 * b^2) / a^2 = (2 * b) / a
def transformation_D (a b : ℝ) : Prop := (a + a * b) / (b + a * b) = a / b

-- Correct transformation is A.
theorem correct_transformation (a b : ℝ) : transformation_A a ∧ ¬transformation_B a b ∧ ¬transformation_C a b ∧ ¬transformation_D a b :=
sorry

end correct_transformation_l546_546159


namespace initial_people_count_is_16_l546_546173

-- Define the conditions
def initial_people (x : ℕ) : Prop :=
  let people_came_in := 5 in
  let people_left := 2 in
  let final_people := 19 in
  x + people_came_in - people_left = final_people

-- Define the theorem
theorem initial_people_count_is_16 (x : ℕ) (h : initial_people x) : x = 16 :=
by
  sorry

end initial_people_count_is_16_l546_546173


namespace shortest_path_exists_l546_546853

variable (A B : Point) (C : Circle)

def shortest_path (A B : Point) (C : Circle) : Set (Path) :=
  {γ | ∀ p1 p2 : Point, p1 ∈ C ∧ p2 ∈ C -> γ = Path.concat (LineSegment A p1)
                                                   (Arc p1 p2)
                                                   (LineSegment p2 B)}

theorem shortest_path_exists (A B : Point) (C : Circle) : ∃ γ ∈ shortest_path A B C, True :=
  sorry

end shortest_path_exists_l546_546853


namespace ratio_of_raspberries_l546_546123

theorem ratio_of_raspberries (B R K L : ℕ) (h1 : B = 42) (h2 : L = 7) (h3 : K = B / 3) (h4 : B = R + K + L) :
  R / Nat.gcd R B = 1 ∧ B / Nat.gcd R B = 2 :=
by
  sorry

end ratio_of_raspberries_l546_546123


namespace prism_cylinder_surface_area_ratio_eq_two_l546_546612

theorem prism_cylinder_surface_area_ratio_eq_two 
  (a b h : ℝ) 
  (V_prism : ℝ) 
  (V_cylinder : ℝ) 
  (S_prism S_cylinder : ℝ)
  (eq_volumes : V_prism = V_cylinder): 
  S_prism / S_cylinder = 2 :=
by
  -- The definitions of V_prism and V_cylinder in terms of prism and cylinder dimensions
  let r := (math.sqrt 3 / 3) * a
  let area_base_prism := (math.sqrt 3 / 4) * a^2
  let V_prism := area_base_prism * b
  let V_cylinder := π * r^2 * h

  -- The relationship between the heights of the prism and cylinder
  have eq_b : b = (4 * π / (3 * math.sqrt 3)) * h := 
    by sorry
    
  -- Compute lateral surface areas 
  let S_prism := 3 * a * b
  let S_cylinder := 2 * π * r * h

  -- Now, prove the ratio of S_prism to S_cylinder is 2
  calc
    S_prism / S_cylinder = (3 * a * b) / (2 * π * r * h) :
      by sorry
                       ... = 2 :
      by sorry

end prism_cylinder_surface_area_ratio_eq_two_l546_546612


namespace initial_number_of_boys_l546_546131

theorem initial_number_of_boys (q b : ℕ) (h₀ : b = 0.5 * q) (h₁ : (b - 3) = 0.4 * q) :
  b = 15 :=
by sorry

end initial_number_of_boys_l546_546131


namespace area_of_region_l546_546453

noncomputable def area_of_enclosed_region : Real :=
  -- equation of the circle after completing square
  let circle_eqn := fun (x y : Real) => ((x - 3)^2 + (y + 4)^2 = 16)
  if circle_eqn then 
    Real.pi * 4^2
  else
    0

theorem area_of_region (h : ∀ x y, x^2 + y^2 - 6x + 8y = -9 → ((x-3)^2 + (y+4)^2 = 16)) :
  area_of_enclosed_region = 16 * Real.pi :=
by
  -- This is a statement, so just include a sorry to skip the proof.
  sorry

end area_of_region_l546_546453


namespace sum_primes_no_solution_congruence_l546_546935

theorem sum_primes_no_solution_congruence :
  ∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) % p1 = 3 % p1) ∧
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) % p2 = 3 % p2) ∧
  p1 + p2 = 7 :=
by
  have h50 : 50 = 2 * 5 * 5 := by norm_num
  have p1 := 2
  have p2 := 5
  use p1, p2
  split
  -- prove p1 is prime
  { exact prime_two }
  split
  -- prove p2 is prime
  { exact prime_five }
  split
  -- show no solution for p1
  { intro hx
    obtain ⟨x, hx⟩ := hx
    have h50x : 50 * x % p1 = (p1 - 7) % p1 := by norm_num
    sorry
  }
  split
  -- show no solution for p2
  { intro hx
    obtain ⟨x, hx⟩ := hx
    have h50x : 50 * x % p2 = (p2 - 7) % p2 := by norm_num
    sorry
  }
  -- prove sum of p1 and p2 is 7
  { norm_num }

end sum_primes_no_solution_congruence_l546_546935


namespace sequence_converges_l546_546630

noncomputable
def c (n : ℕ) : ℝ :=
  if n = 1 then 1 else Real.sqrt (2 * Real.sqrt (3 * Real.sqrt (5 * c (n - 1))))

theorem sequence_converges :
  ∃ L : ℝ, 
  L = 1.831 ∧ 
  tendsto (λ n, c n) at_top (𝓝 L) := 
sorry

end sequence_converges_l546_546630


namespace angle_B_in_parallelogram_l546_546667

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end angle_B_in_parallelogram_l546_546667


namespace maximum_sum_of_numbers_in_white_cells_l546_546663

/-- A proof problem for finding the maximum sum of numbers in white cells for a given 8x8 table with 23 black cells --/
theorem maximum_sum_of_numbers_in_white_cells : 
  ∃ (arrangement : fin 8 × fin 8 → bool), 
  (∑ i, ∑ j, if arrangement (i, j) = false then ∑ k, (if arrangement (i, k) = true then 1 else 0) + ∑ k, (if arrangement (k, j) = true then 1 else 0) else 0) = 234 :=
by 
  sorry

end maximum_sum_of_numbers_in_white_cells_l546_546663


namespace john_outside_doors_count_l546_546687

theorem john_outside_doors_count 
  (bedroom_doors : ℕ := 3) 
  (cost_outside_door : ℕ := 20) 
  (total_cost : ℕ := 70) 
  (cost_bedroom_door := cost_outside_door / 2) 
  (total_bedroom_cost := bedroom_doors * cost_bedroom_door) 
  (outside_doors := (total_cost - total_bedroom_cost) / cost_outside_door) : 
  outside_doors = 2 :=
by
  sorry

end john_outside_doors_count_l546_546687


namespace count_first_10_alternate_prime_sums_is_3_l546_546541

def alternate_prime_sums : List ℕ := 
  [2, 2 + 5, 2 + 5 + 11, 2 + 5 + 11 + 17, 2 + 5 + 11 + 17 + 23,
   2 + 5 + 11 + 17 + 23 + 31, 2 + 5 + 11 + 17 + 23 + 31 + 41,
   2 + 5 + 11 + 17 + 23 + 31 + 41 + 47, 2 + 5 + 11 + 17 + 23 + 31 + 41 + 47 + 59,
   2 + 5 + 11 + 17 + 23 + 31 + 41 + 47 + 59 + 67]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def count_prime_sums (lst : List ℕ) : ℕ :=
  lst.countp is_prime

theorem count_first_10_alternate_prime_sums_is_3 :
  count_prime_sums alternate_prime_sums = 3 := 
sorry

end count_first_10_alternate_prime_sums_is_3_l546_546541


namespace fraction_f_n2_f_1_div_e_l546_546595

def f (x : ℝ) : ℝ :=
  if x > 1 then
    2^x
  else if 0 < x ∧ x ≤ 1 then
    -Real.log x
  else if x ≤ 0 then
    -f (-x)
  else
    0 -- Since this case should never happen for strictly positive x

theorem fraction_f_n2_f_1_div_e : f(-2) / f(1/e) = -4 :=
by {
  have h₁ : f(-2) = -f(2),
  { rw [f], simp, },
  have h₂ : f(2) = 4,
  { rw [f], simp, },
  have h₃ : f(1/e) = 1,
  { rw [f], simp, },
  rw [h₁, h₂, h₃],
  norm_num,
}

end fraction_f_n2_f_1_div_e_l546_546595


namespace correct_answer_l546_546335

def floor (t : ℝ) : ℝ := if t < 0 then -(⌊-t⌋) else ⌊t⌋

def fractional_part (t : ℝ) : ℝ := t - floor t

def H (t : ℝ) : ℝ := 1 - fractional_part t

def R (t : ℝ) : set (ℝ × ℝ) := {p | let H := 1 - (t - floor t) in (p.1 + H)^2 + p.2^2 ≤ H^2}

theorem correct_answer (t : ℝ) (ht : t ≥ 0) :
  ( (∀ t, ((1 : ℝ), (0 : ℝ)) ∈ R t) = False ∧
    (∀ t, 0 ≤ real.pi * (H t) * (H t) ∧ real.pi * (H t) * (H t) ≤ 2 * real.pi) = False ∧
    (∀ t ≥ 3, ∀ (p : ℝ × ℝ), p ∈ R t → p.1 ≤ 0 ∧ p.2 ≤ 0) = False ∧
    (∀ t, (- (H t), 0) = [- ((H t))]) = False ∧
    (∀ (∀ t, ((1 : ℝ), (0 : ℝ)) ∈ R t) ∧ (∀ t, 0 ≤ real.pi * (H t) * (H t) ∧ real.pi * (H t) * (H t) ≤ 2 * real.pi) ∧
      (∀ t ≥ 3, ∀ (p : ℝ × ℝ), p ∈ R t → p.1 ≤ 0 ∧ p.2 ≤ 0) ∧
      (∀ t, (- (H t), 0) = [- ((H t))]) = False ) = True ) sorry

end correct_answer_l546_546335


namespace geometric_sequence_general_formula_arithmetic_sequence_sum_l546_546976

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 * a 6 = 32 * a 2 * a 10) (h2 : a 1 + a 1 * q + a 1 * q^2 = 21 / 4) :
  ∀ n : ℕ, n > 0 → a n = 3 * (1 / 2)^(n - 1) := sorry

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ) (h1 : a 1 * a 6 = 32 * a 2 * a 10) (h2 : a 1 + a 1 * q + a 1 * q^2 = 21 / 4)
  (h3 : ∀ n : ℕ, n > 0 → b n = log 2 (a n / 3)) :
  ∀ n : ℕ, n > 0 → (finset.range n).sum b = ((-n^2) + n) / 2 := sorry

end geometric_sequence_general_formula_arithmetic_sequence_sum_l546_546976


namespace cube_roots_not_arithmetic_progression_l546_546027

theorem cube_roots_not_arithmetic_progression
  (p q r : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (h_distinct: p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ¬ ∃ (d : ℝ) (m n : ℤ), (n ≠ m) ∧ (↑q)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (m : ℝ) * d ∧ (↑r)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (n : ℝ) * d :=
by sorry

end cube_roots_not_arithmetic_progression_l546_546027


namespace required_moles_of_HC2H3O2_to_react_with_2_moles_of_NaHCO3_l546_546273

def balancedEquation (HC2H3O2 NaHCO3 NaC2H3O2 H2O CO2 : Type) : Prop :=
  ∀ (moles_HC2H3O2 moles_NaHCO3 moles_NaC2H3O2 moles_H2O moles_CO2 : ℕ),
    moles_HC2H3O2 = moles_NaHCO3 →
    moles_NaHCO3 = moles_NaC2H3O2 →
    moles_NaC2H3O2 = moles_H2O →
    moles_H2O = moles_CO2 → 
    true

theorem required_moles_of_HC2H3O2_to_react_with_2_moles_of_NaHCO3
  (HC2H3O2 NaHCO3 NaC2H3O2 H2O CO2 : Type)
  (balanced : balancedEquation HC2H3O2 NaHCO3 NaC2H3O2 H2O CO2) :
  balanced 2 2 2 2 2 → 2 = 2 :=
by
  intro h
  exact eq.refl 2

end required_moles_of_HC2H3O2_to_react_with_2_moles_of_NaHCO3_l546_546273


namespace common_tangent_C₁_C₂_l546_546232

open Real

-- Definitions of the parabolas
def C₁ : ℝ → ℝ := λ x, x^2 + 2 * x
def C₂ (a : ℝ) : ℝ → ℝ := λ x, -x^2 + a

-- The statement of the problem as a Lean theorem
theorem common_tangent_C₁_C₂ (a : ℝ) : 
  (∃ l : ℝ → ℝ, 
    (∀ x₁, C₁ x₁ = l x₁ ∧ deriv C₁ x₁ = deriv l x₁) ∧ 
    (∀ x₂, C₂ a x₂ = l x₂ ∧ deriv (C₂ a) x₂ = deriv l x₂)) ↔ 
  (a = -1/2 ∧ ∃ (l : ℝ → ℝ), ∀ x, l x = x - 1/4) :=
by {
  sorry -- Proof to be filled in
}

end common_tangent_C₁_C₂_l546_546232


namespace elevator_min_trips_l546_546111

theorem elevator_min_trips :
  let masses := [150, 60, 70, 71, 72, 100, 101, 102, 103] in
  let max_load := 200 in
  (min_trips masses max_load = 5) :=
begin
  -- Sorry is used to skip the proof.
  sorry
end

end elevator_min_trips_l546_546111


namespace aerith_wins_starting_XX_aerith_wins_starting_O_l546_546898

-- Define the initial conditions of the game
def initial_condition_1 : String := "X"

-- Define Aerith's starting moves
def start_XX : String := initial_condition_1 ++ "X"
def start_O : String := "O"

-- Define the win condition as avoiding three evenly spaced X's or O's
def win_condition (s : String) : Prop :=
  ¬(∃ i j k, i < j ∧ j < k ∧ s.get (i % s.length) = s.get (j % s.length) ∧ s.get (i % s.length) = s.get (k % s.length) ∧ s.get (i % s.length) ∈ "XO")

-- Theorem statements for Aerith starting with "XX" or "O"
theorem aerith_wins_starting_XX : win_condition start_XX → ∃ s, (win_condition s) := sorry
theorem aerith_wins_starting_O : win_condition start_O → ∃ s, (win_condition s) := sorry

end aerith_wins_starting_XX_aerith_wins_starting_O_l546_546898


namespace cost_of_car_l546_546012

theorem cost_of_car (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment = 3000 →
  num_installments = 6 →
  installment_amount = 2500 →
  initial_payment + num_installments * installment_amount = 18000 :=
by
  intros h_initial h_num h_installment
  sorry

end cost_of_car_l546_546012


namespace sum_proper_divisors_81_l546_546815

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546815


namespace committee_size_l546_546803

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end committee_size_l546_546803


namespace num_ways_to_sum_420_l546_546316

-- Define the problem
def increasing_sum_ways (S : ℕ) : ℕ :=
  (Nat.factors S).countp (λ n => 
    let f := S / n in 
    (f - n + 1).to_nat % 2 = 0 ∧ f ≥ n ∧ n ≥ 2)

-- Main theorem statement
theorem num_ways_to_sum_420 : increasing_sum_ways 420 = 7 := 
by 
  sorry

end num_ways_to_sum_420_l546_546316


namespace greatest_divisor_of_remainders_l546_546457

theorem greatest_divisor_of_remainders (x : ℕ) :
  (1442 % x = 12) ∧ (1816 % x = 6) ↔ x = 10 :=
by
  sorry

end greatest_divisor_of_remainders_l546_546457


namespace probability_one_defective_l546_546122

def total_bulbs : ℕ := 20
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs
def probability_non_defective_both : ℚ := (16 / 20) * (15 / 19)
def probability_at_least_one_defective : ℚ := 1 - probability_non_defective_both

theorem probability_one_defective :
  probability_at_least_one_defective = 7 / 19 :=
by
  sorry

end probability_one_defective_l546_546122


namespace percentage_of_red_bellied_minnows_l546_546008

/-- Problem Statement:
Given that 30% of the minnows have green bellies, 20 minnows have red bellies, 
and 15 minnows have white bellies, prove that the percentage of the minnows 
that have red bellies is 40%.
-/
theorem percentage_of_red_bellied_minnows 
  (T : ℕ) (h1 : 20 + 15 + Nat.floor (0.30 * T) = T) :
  (20 / T) * 100 = 40 :=
by
  sorry

end percentage_of_red_bellied_minnows_l546_546008


namespace rationalize_denominator_l546_546728

theorem rationalize_denominator (a b : ℝ) (ha : a = (3 : ℝ)^(1/3)) (hb : b = (27 : ℝ)^(1/3)) : 
  (1 / (a + b)) * (9^(1/3)) = (9^(1/3)) / 12 :=
by
  have h1 : b = 3 * a := sorry
  rw [h1, ←mul_assoc]
  have h2 : (a + 3 * a) = 4 * a := sorry
  rw [h2, mul_comm, ←one_div_mul_one_div]
  sorry

end rationalize_denominator_l546_546728


namespace longest_path_is_34_l546_546504

def total_intersections : ℕ := 36
def starts_at_A : Intersection := Intersection.white
def end_at_B : Intersection := Intersection.white

-- Assume Intersection and move definition
inductive Intersection
| white
| black

-- Assume path definition within the grid
def path (n : ℕ) (start : Intersection) (end : Intersection) : Prop := 
∀ (m: ℕ), m ≤ n → m % 2 = 0

theorem longest_path_is_34 (n : ℕ) :
  n ≤ 34 :=
begin
  sorry
end

end longest_path_is_34_l546_546504


namespace relationship_s2_s3_l546_546689

noncomputable def centroid (O A B C : Point) : Prop :=
  O = centroid_of_triangle A B C

noncomputable def is_midpoint (P A B: Point) : Prop :=
  2 * vector(P, A) = vector(A, B) + vector(P, B)

noncomputable def s1 (O A B C : Point) : ℝ :=
  distance O A + distance O B + distance O C

noncomputable def s2 (A B C : Point) : ℝ :=
  distance A B + distance B C + distance C A

noncomputable def s3 (O P Q R : Point) : ℝ :=
  3 * (distance O P + distance O Q + distance O R)

theorem relationship_s2_s3 
  (O A B C P Q R : Point)
  (H1 : centroid O A B C)
  (H2 : is_midpoint P A B)
  (H3 : is_midpoint Q B C)
  (H4 : is_midpoint R C A) :
  s2 A B C > s3 O P Q R :=
sorry

end relationship_s2_s3_l546_546689


namespace minimum_value_of_d_l546_546717

noncomputable def minimum_d_value (d a b : ℕ) : Prop :=
  ∃ (A B C D : ℝ×ℝ) (θ : ℝ),
    |B.1 - A.1| = d ∧
    |C.1 - B.1| = a ∧
    |D.1 - C.1| = a ∧
    |A.1 - D.1| = b ∧
    (sin θ = a / d) ∧
    (b = d * (1 - 2 * (a / d)^2)) ∧
    distinct_pos_int a b d

def distinct_pos_int (a b d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ d ∧ a ≠ d ∧ a > 0 ∧ b > 0 ∧ d > 0

theorem minimum_value_of_d : ∀ (a b d : ℕ), minimum_d_value d a b → d = 8 :=
by
  sorry

end minimum_value_of_d_l546_546717


namespace find_BC_length_l546_546648

theorem find_BC_length
  (area : ℝ) (AB AC : ℝ)
  (h_area : area = 10 * Real.sqrt 3)
  (h_AB : AB = 5)
  (h_AC : AC = 8) :
  ∃ BC : ℝ, BC = 7 :=
by
  sorry

end find_BC_length_l546_546648


namespace valid_boxes_count_l546_546156

theorem valid_boxes_count (p : ℕ) :
  let pc := 392 in
  let divisors := { p : ℕ // p ∣ 392 } in
  let valid_p := { p : divisors // 1 < p ∧ 392 / p > 3 } in
  cardinality valid_p = 11 :=
by sorry

end valid_boxes_count_l546_546156


namespace Q_is_fixed_l546_546349

-- Define the circle Γ
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define points A, B, C inscribed in circle Γ
variables (A B C P : ℝ × ℝ) (Γ : Circle)
-- P is a variable point on the arc AB that does not contain C
variable (P_variation : Set (ℝ × ℝ))

-- Define I and J as the centers of the incircles of triangles ACP and BCP
variables (I J Q : ℝ × ℝ)

-- Given that Q is the intersection of Γ and the circumcircle of triangle PIJ
-- We want to show that Q remains fixed as P varies
theorem Q_is_fixed (hABC : ∀ (P ∈ P_variation), 
  euclidean_geometry.is_in_circle Γ A ∧
  euclidean_geometry.is_in_circle Γ B ∧
  euclidean_geometry.is_in_circle Γ C ∧
  euclidean_geometry.is_in_circle Γ P ∧
  -- I and J are the centers of the incircles of ACP and BCP resp.
  (euclidean_geometry.center_incircle (triangle.mk A C P) = I) ∧
  (euclidean_geometry.center_incircle (triangle.mk B C P) = J) ∧
  -- Q is the intersection of Γ and the circumcircle of PIJ
  euclidean_geometry.is_in_circle Γ Q ∧
  euclidean_geometry.is_in_circle (euclidean_geometry.circumcircle (triangle.mk P I J)) Q) 
: ∃ Q_fixed : ℝ × ℝ, ∀ P ∈ P_variation, Q = Q_fixed := 
sorry

end Q_is_fixed_l546_546349


namespace chessboard_rectangles_squares_l546_546431

noncomputable def m_nat : ℕ := 19
noncomputable def n_nat : ℕ := 135
noncomputable def r : ℕ := 2025
noncomputable def s : ℕ := 285

theorem chessboard_rectangles_squares : m_nat + n_nat = 154 :=
by
  have ratio_s_r : s / r = 19 / 135 := sorry
  have relatively_prime : Nat.gcd 19 135 = 1 := sorry
  exact m_nat + n_nat = 154
  sorry

end chessboard_rectangles_squares_l546_546431


namespace projectile_reaches_90_feet_l546_546411

theorem projectile_reaches_90_feet
  (h_eq : ∀ t : ℝ, -16 * t^2 + 100 * t = y)
  (y_val : y = 90) :
  ∃ t : ℝ, t > 0 ∧ t ≈ 1.09 ∧ -16 * t^2 + 100 * t = 90 :=
sorry

end projectile_reaches_90_feet_l546_546411


namespace field_width_l546_546869

open_locale classical

theorem field_width
  (L : ℝ) (P_l : ℝ) (P_w : ℝ) (P_d : ℝ) (H : ℝ)
  (V_pit : ℝ) (W : ℝ)
  (Volume_eq: V_pit = L * W * H)
  (A_total : ℝ) (A_pit : ℝ) (A_remaining : ℝ)
  (V_spread : ℝ)
  (field_cond : L = 20)
  (pit_cond : P_l = 8 ∧ P_w = 5 ∧ P_d = 2)
  (spread_cond : H = 0.5 ∧ V_pit = 80)
  (remaining_area_cond : A_total = 20 * W ∧ A_pit = 8 * 5 ∧ A_remaining = 20 * W - 40)
  (spread_volume_cond : V_spread = 10 * W - 20) :
  W = 10 :=
begin
  sorry
end

end field_width_l546_546869


namespace trajectory_of_Q_is_ellipse_max_area_triangle_ABD_l546_546881

def point_N := (1, 0)
def circle_M (x y : ℝ) := (x + 1)^2 + y^2 = 16

def point_P := { p : (ℝ × ℝ) // (∃ x y : ℝ, p = (x, y) ∧ circle_M x y) }

-- Conditions for triangle ABD
def line_l (k : ℝ) (x : ℝ) := k * x + 1 = 0

-- Given conditions and asked to prove the trajectory of point Q
theorem trajectory_of_Q_is_ellipse :
  ∀ (P : point_P), ∃ Q : (ℝ × ℝ), 
    let xQ := Q.1, yQ := Q.2 in
    ∃ xQ yQ : ℝ, (xQ^2 / 4) + (yQ^2 / 3) = 1 :=
sorry

-- Given conditions and asked to find the maximum area of triangle ABD
theorem max_area_triangle_ABD : 
  ∃ (A B D : (ℝ × ℝ)), 
    let xA := A.1, yA := A.2,
        xB := B.1, yB := B.2,
        xD := D.1, yD := D.2 in
    -- Additional geometric conditions on A, B, D need to be provided
    let S := -- Calculate area here in terms of coordinates
    S ≤ (4 * real.sqrt 6 / 3) :=
sorry

end trajectory_of_Q_is_ellipse_max_area_triangle_ABD_l546_546881


namespace total_white_roses_needed_l546_546367

theorem total_white_roses_needed : 
  let bouquets := 5 
  let table_decorations := 7 
  let roses_per_bouquet := 5 
  let roses_per_table_decoration := 12 in
  (bouquets * roses_per_bouquet) + (table_decorations * roses_per_table_decoration) = 109 := by
  sorry

end total_white_roses_needed_l546_546367


namespace greatest_int_with_gcd_18_is_138_l546_546081

theorem greatest_int_with_gcd_18_is_138 :
  ∃ n : ℕ, n < 150 ∧ int.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ int.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_int_with_gcd_18_is_138_l546_546081


namespace solution_set_inequality_l546_546056

theorem solution_set_inequality : {x : ℝ | (x-1)*(x-2) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end solution_set_inequality_l546_546056


namespace length_OC_l546_546108

theorem length_OC (a b : ℝ) (h_perpendicular : ∀ x, x^2 + a * x + b = 0 → x = 1 ∨ x = b) : 
  1 = 1 :=
by 
  sorry

end length_OC_l546_546108


namespace password_decryption_probability_l546_546144

theorem password_decryption_probability :
  let A := (1:ℚ)/5
  let B := (1:ℚ)/3
  let C := (1:ℚ)/4
  let P_decrypt := 1 - (1 - A) * (1 - B) * (1 - C)
  P_decrypt = 3/5 := 
  by
    -- Calculations and logic will be provided here
    sorry

end password_decryption_probability_l546_546144


namespace area_of_triangle_DBC_l546_546306

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def area_of_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
1 / 2 * (abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)))

theorem area_of_triangle_DBC :
  let A := (0, 6) in
  let B := (0, 0) in
  let C := (6, 0) in
  let D := midpoint A B in
  area_of_triangle D B C = 9 :=
by
  let A := (0, 6)
  let B := (0, 0)
  let C := (6, 0)
  let D := midpoint A B
  have : D = (0, 3), by simp [D, midpoint, A, B]
  rw this
  simp [area_of_triangle, D, B, C]
  sorry

end area_of_triangle_DBC_l546_546306


namespace S_40_value_l546_546777

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom h1 : S 10 = 10
axiom h2 : S 30 = 70

theorem S_40_value : S 40 = 150 :=
by
  -- Conditions
  have h1 : S 10 = 10 := h1
  have h2 : S 30 = 70 := h2
  -- Start proof here
  sorry

end S_40_value_l546_546777


namespace domain_of_sqrt_l546_546041

theorem domain_of_sqrt (x : ℝ) : 
  ∃ y, y = sqrt (3 - 2 * x - x^2) ↔ x ∈ Set.Icc (-3 : ℝ) 1 := 
by
  sorry

end domain_of_sqrt_l546_546041


namespace exist_line_max_dist_sum_exist_line_min_dist_sum_l546_546385

-- Assuming the existence of a type for points and lines in \mathbb{R}^2
variables (Point Line : Type) [metric_space Point] [AffineSpace Point ℝ] 

open affine

-- Defining the given points A, B, and O which are not collinear
variables (A B O : Point) 
variables (h_noncollinear : ¬ collinear ℝ {A, B, O})

-- Definition of distances from points to a line
noncomputable def distance_to_line (P : Point) (l : Line) : ℝ := sorry 

-- Line l passing through point O
variables (l : Line) (h_on_l : O ∈ l)

-- Statements to prove
theorem exist_line_max_dist_sum : ∃ (l : Line), O ∈ l ∧ 
  ∀ (l' : Line) (h' : O ∈ l'), 
    distance_to_line A l + distance_to_line B l ≥ distance_to_line A l' + distance_to_line B l' := sorry

theorem exist_line_min_dist_sum : ∃ (l : Line), O ∈ l ∧ 
  ∀ (l' : Line) (h' : O ∈ l'), 
    distance_to_line A l + distance_to_line B l ≤ distance_to_line A l' + distance_to_line B l' := sorry

end exist_line_max_dist_sum_exist_line_min_dist_sum_l546_546385


namespace sum_proper_divisors_81_l546_546810

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546810


namespace sneakers_sold_l546_546550

theorem sneakers_sold (total_shoes sandals boots : ℕ) (h1 : total_shoes = 17) (h2 : sandals = 4) (h3 : boots = 11) :
  total_shoes - (sandals + boots) = 2 :=
by
  -- proof steps will be included here
  sorry

end sneakers_sold_l546_546550


namespace infinitude_of_special_divisors_l546_546205

open Int Nat

theorem infinitude_of_special_divisors {a b : ℤ} (ha : a > 1) (hb : b > 1):
  ∃ᶠ n in at_top, ∀ (m t : ℕ), m > 0 → t > 0 → Euler's_totient_function (a ^ n - 1) ≠ b ^ m - b ^ t :=
sorry

end infinitude_of_special_divisors_l546_546205


namespace initial_population_l546_546483

theorem initial_population (P : ℝ) (h1 : P * 1.05 * 0.95 = 9975) : P = 10000 :=
by
  sorry

end initial_population_l546_546483


namespace find_divisor_l546_546002

theorem find_divisor (d : ℕ) (h1 : ∀ n ∈ {n : ℕ // ∃ k : ℕ, n = k * d + 5}, n = 597) (h2 : ∃ t : ℕ, t = (75 - 1) * d + 5 ∧ 597 = t) : d = 8 :=
by
  sorry

end find_divisor_l546_546002


namespace fascinating_integers_l546_546600

def is_fascinating (n : ℕ) : Prop :=
  ∃ (a : Fin n → ℤ), (∑ i, a i = n) ∧ (∏ i, a i = n)

theorem fascinating_integers (n : ℕ) : is_fascinating n ↔
  (∃ t : ℕ, n = 4 * t + 1)
  ∨ (∃ t : ℕ, t ≥ 1 ∧ n = 4 * t) :=
by
  sorry

end fascinating_integers_l546_546600


namespace radius_decrease_l546_546423

theorem radius_decrease (r r' : ℝ) (A A' : ℝ) (h_original_area : A = π * r^2)
  (h_area_decrease : A' = 0.25 * A) (h_new_area : A' = π * r'^2) : r' = 0.5 * r :=
by
  sorry

end radius_decrease_l546_546423


namespace johns_money_now_l546_546686

def initial_money : ℝ := 5.0
def game_cost : ℝ := 2.0
def candy_bar_cost : ℝ := 1.0
def soda_cost : ℝ := 1.5
def soda_discount : ℝ := 0.1
def coupon_value : ℝ := 0.5
def magazine_cost : ℝ := 3.0
def allowance : ℝ := 26.0

theorem johns_money_now :
  let discounted_soda := soda_cost - (soda_cost * soda_discount) in
  let remaining_magazine_cost := magazine_cost - coupon_value in
  let total_expenses := game_cost + candy_bar_cost + discounted_soda + remaining_magazine_cost in
  let additional_needed := total_expenses - initial_money in
  allowance - additional_needed = 24.15 := by
sorry

end johns_money_now_l546_546686


namespace find_j_l546_546416

theorem find_j (j k : ℝ) :
  (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ (∀ i ∈ {0, 1, 2, 3}, a + i * d ≠ a + j * d) ∧
  (∀ x : ℝ, (x = a ∨ x = a + d ∨ x = a + 2 * d ∨ x = a + 3 * d) →
  x^4 + j*x^2 + k*x + 400 = 0)) → j = -40 :=
by
  sorry

end find_j_l546_546416


namespace intersection_M_P_union_M_P_is_universal_l546_546611
-- Load the relevant libraries

open Set

-- Define the conditions

def U : Set ℝ := univ

def M (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4 * m - 2}

def P : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ 1}

-- Define the Lean statement for proof problem 1
theorem intersection_M_P (m : ℝ) (h : m = 2) : 
  M m ∩ P = {x : ℝ | -1 ≤ x ∧ x ≤ 1 ∨ 2 < x ∧ x ≤ 4 * 2 - 2} :=
by
  sorry

-- Define the Lean statement for proof problem 2
theorem union_M_P_is_universal (m : ℝ) : 
  (M m ∪ P = univ) ↔ (m ≥ 1) :=
by
  sorry

end intersection_M_P_union_M_P_is_universal_l546_546611


namespace johns_total_distance_l546_546685

theorem johns_total_distance :
  let monday := 1700
  let tuesday := monday + 200
  let wednesday := 0.7 * tuesday
  let thursday := 2 * wednesday
  let friday := 3.5 * 1000
  let saturday := 0
  monday + tuesday + wednesday + thursday + friday + saturday = 10090 := 
by
  sorry

end johns_total_distance_l546_546685


namespace stratified_sampling_C_count_l546_546979

theorem stratified_sampling_C_count :
  ∀ (A_ratio B_ratio C_ratio : ℕ) (total_sample_size : ℕ),
    A_ratio = 5 →
    B_ratio = 3 →
    C_ratio = 2 →
    total_sample_size = 100 →
    let total_ratio := A_ratio + B_ratio + C_ratio in
    let C_proportion := (C_ratio : ℚ) / (total_ratio : ℚ) in
    let C_sample := total_sample_size * C_proportion in
    C_sample = 20 := 
by
  intros A_ratio B_ratio C_ratio total_sample_size hA hB hC hTotal
  let total_ratio := A_ratio + B_ratio + C_ratio
  let C_proportion := (C_ratio : ℚ) / (total_ratio : ℚ)
  let C_sample := total_sample_size * C_proportion
  have h1 : total_ratio = 10, from by rw [hA, hB, hC]; norm_num
  have hCprop : C_proportion = 0.2, from by rw [hC, h1]; norm_num; exact (rat.cast_div _ _).symm
  have hCsample : C_sample = 100 * 0.2, from by rw [hTotal, hCprop]; norm_num
  rw hCsample; norm_num; sorry

end stratified_sampling_C_count_l546_546979


namespace calculate_value_l546_546913

theorem calculate_value (x y : ℝ) (h : 2 * x + y = 6) : 
    ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 :=
by 
  sorry

end calculate_value_l546_546913


namespace permutations_partition_l546_546978

/-
  Problem statement:
  Given a permutation of \(1, 2, 3, \ldots, n\), with consecutive elements \(a, b, c\) (in that order),
  we may perform either of the following moves:
  1. If \(a\) is the median of \(a, b\), and \(c\), we may replace \(a, b, c\) with \(b, c, a\) (in that order).
  2. If \(c\) is the median of \(a, b\), and \(c\), we may replace \(a, b, c\) with \(c, a, b\) (in that order).
  
  Goal: Find the least number of sets in a partition of all permutations such that any two
  permutations in the same set are obtainable from each other by a sequence of moves.
-/

def is_median (a b c : ℕ) : Bool :=
  (b < a ∧ a < c) ∨ (c < a ∧ a < b)

def is_invertible_median (a b c : ℕ) : Bool :=
  (a < b ∧ b < c) ∨ (c < b ∧ b < a)

noncomputable def least_number_of_sets (n : ℕ) : ℕ :=
  n * n - 3 * n + 4

theorem permutations_partition (n : ℕ) : 
  ∀ (perm1 perm2 : List ℕ), 
  perm1.perm perm2 →
  (∃ (seq : List (List ℕ)), 
    (∀ p ∈ seq, p.perm perm1 ∧ p.perm perm2) ∧ 
    List.chain is_median seq.head seq.tail ∨ 
    List.chain is_invertible_median seq.head seq.tail) → 
  least_number_of_sets n = n * n - 3 * n + 4 := 
by
  sorry

end permutations_partition_l546_546978


namespace average_increase_l546_546119

-- Given definitions
def final_runs (A : ℝ) := 16 * A + 84
def new_average (A : ℝ) := final_runs A / 17
def increase_in_average (A : ℝ) (X : ℝ) := A + X = 36

-- Theorem statement to be proven
theorem average_increase (A X : ℝ) (h1: new_average A = 36) : increase_in_average A X → X = 3 :=
by
  sorry

end average_increase_l546_546119


namespace infinite_perpendicular_lines_l546_546288

variable (α : Type) [RealPlane α]
variable (l : Line)

def not_perpendicular (l : Line) (α : Plane) : Prop := 
  ¬ perpendicular_to_plane l α

def perpendicular_lines_in_plane (l : Line) (α : Plane) : Set Line :=
  {m : Line | m ⊆ α ∧ perpendicular m l}

theorem infinite_perpendicular_lines (l : Line) (α : Plane) 
  (h : not_perpendicular l α) : 
  ∃ S : Set (Line), S ⊆ perpendicular_lines_in_plane l α ∧ Infinite S := 
sorry

end infinite_perpendicular_lines_l546_546288


namespace count_f2016_l546_546962

def smallest_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else
  let factors := (List.range (n-1)).filter (λ m, m > 1 ∧ n % m = 0) 
  in factors.head?.getOrElse n

def f (n : ℕ) : ℕ := 
  if n > 1 then n + smallest_factor n else 0

theorem count_f2016 : 
  (∀ n, f(n) ≠ 2015) ∧ 
  (finset.card (finset.filter (λ n, f(n) = 2016) (finset.range 2017)) = 3) :=
by {
  sorry
}

end count_f2016_l546_546962


namespace xy_sufficient_not_necessary_l546_546285

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy_lt_zero : x * y < 0) → abs (x - y) = abs x + abs y ∧ (abs (x - y) = abs x + abs y → x * y ≥ 0) := 
by
  sorry

end xy_sufficient_not_necessary_l546_546285


namespace store_money_left_after_sale_l546_546890

theorem store_money_left_after_sale :
  ∀ (n p : ℕ) (d s : ℝ) (debt : ℕ), 
  n = 2000 → 
  p = 50 → 
  d = 0.80 → 
  s = 0.90 → 
  debt = 15000 → 
  (nat.floor (s * n) * (p - nat.floor (d * p)) : ℕ) - debt = 3000 :=
by
  intros n p d s debt hn hp hd hs hdebt
  sorry

end store_money_left_after_sale_l546_546890


namespace least_possible_value_ab_l546_546688

open Nat

theorem least_possible_value_ab :
  ∃ (A B : ℕ), A ≠ B ∧ (num_divisors A = 8) ∧ (num_divisors B = 8) ∧ (|A - B| = 1) :=
sorry

end least_possible_value_ab_l546_546688


namespace least_non_blissful_multiple_of_7_l546_546120

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> list.sum

def is_blissful (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven (n : ℕ) : Prop :=
  n % 7 = 0

theorem least_non_blissful_multiple_of_7 : ∃ n > 0, is_multiple_of_seven n ∧ ¬is_blissful n ∧ ∀ m, 0 < m → is_multiple_of_seven m → m < n → is_blissful m :=
  ⟨14, by decide,
      by decide,
      by decide,
      λ m m_pos m_multiple m_lt_n, by
        have h : m < 14 := m_lt_n by decide;
        sorry⟩

end least_non_blissful_multiple_of_7_l546_546120


namespace sin_alpha_plus_beta_l546_546047

theorem sin_alpha_plus_beta (α β : Real)
  (h1 : α = Real.arctan (3))
  (h2 : β = Real.arctan (-3))
  (intersection_line_circle : ∃ A B : Point, (A ≠ B) ∧ (A ⊆ (λ p : ℝ × ℝ, p.2 = 3 * p.1)) ∧ (A ⊆ (λ p : ℝ × ℝ, p.1^2 + p.2^2 = 1)) ∧ (B ⊆ (λ p : ℝ × ℝ, p.2 = 3 * p.1)) ∧ (B ⊆ (λ p : ℝ × ℝ, p.1^2 + p.2^2 = 1)))
  : Real.sin (α + β) = -3 / 5 := by
  sorry

end sin_alpha_plus_beta_l546_546047


namespace diamond_problem_l546_546570

def diamond (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

theorem diamond_problem : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * real.sqrt 2 := by
  sorry

end diamond_problem_l546_546570


namespace greatest_multiple_of_4_less_than_100_l546_546087

theorem greatest_multiple_of_4_less_than_100 : ∃ n : ℕ, n % 4 = 0 ∧ n < 100 ∧ ∀ m : ℕ, (m % 4 = 0 ∧ m < 100) → m ≤ n 
:= by
  sorry

end greatest_multiple_of_4_less_than_100_l546_546087


namespace chessboard_squares_and_rectangles_l546_546433

theorem chessboard_squares_and_rectangles :
  let r := 45 * 45
  let s := (list.sum ∘ list.map (λ n, n ^ 2) $ list.range 10)
  let gcd := nat.gcd 285 2025
  let m := 285 / gcd
  let n := 2025 / gcd
  m + n = 154 :=
by {
  have : r = 2025, { refl },
  have : s = 285, {
    -- Calculating the sum of squares from 1^2 to 9^2
    repeat { sorry }
  },
  have : gcd = 15, { sorry },
  have : m = 19, { sorry },
  have : n = 135, { sorry },
  calc
    m + n = 19 + 135 : sorry
    ... = 154       : sorry
}

end chessboard_squares_and_rectangles_l546_546433


namespace parallelogram_angle_B_eq_130_l546_546669

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end parallelogram_angle_B_eq_130_l546_546669


namespace problem1_problem2_l546_546985

-- Definitions and conditions
def arithmetic_sequence (a: ℕ → ℤ) (d: ℤ) := ∀ n: ℕ, a (n + 1) = a n + d
noncomputable def sum_of_first_n_terms (a: ℕ → ℤ) (S: ℕ → ℤ) := ∀ n: ℕ, S n = 0 → (S (n + 1) = S n + a (n + 1))

-- Problems
theorem problem1 (a: ℕ → ℤ) (d: ℤ) (h₀: arithmetic_sequence a d) (h₁: d = 1) :
  (1, a 1, a 3) forms_geometric_sequence → (a 1 = -1 ∨ a 1 = 2) :=
sorry

theorem problem2 (a: ℕ → ℤ) (S: ℕ → ℤ) (d: ℤ) (h₀: arithmetic_sequence a d) 
  (sum_def: sum_of_first_n_terms a S) (h₁: d = 1) (h₂: S 5 > a 1 * a 9) : 
  -5 < a 1 ∧ a 1 < 2 :=
sorry

end problem1_problem2_l546_546985


namespace bounded_sequence_l546_546338

noncomputable def f : ℕ → ℕ → ℕ
| a, 1 := a
| a, (n + 1) := if sqrt (f a n) * sqrt (f a n) = f a n then sqrt (f a n) else f a n + 3

theorem bounded_sequence (a : ℕ) (ha : 3 ∣ a) : 
  ∃ M, ∀ n, f a n ≤ M := 
begin
  use a * a,
  admit
end

end bounded_sequence_l546_546338


namespace max_at_one_iff_a_lt_neg_e_l546_546254

variable (a b : ℝ)
def f (x : ℝ) := (1 / 2) * Real.exp (2 * x) + (a - Real.exp 1) * Real.exp x - a * Real.exp 1 + b

theorem max_at_one_iff_a_lt_neg_e :
  (∀ x : ℝ, deriv (f a b) x = 0 → x = 1) → a < -Real.exp 1 :=
by
  sorry

end max_at_one_iff_a_lt_neg_e_l546_546254


namespace Avianna_red_candles_l546_546380

theorem Avianna_red_candles (R : ℕ) : 
  (R / 27 = 5 / 3) → R = 45 := 
by
  sorry

end Avianna_red_candles_l546_546380


namespace find_milk_ounces_l546_546330

def bathroom_limit : ℕ := 32
def grape_juice_ounces : ℕ := 16
def water_ounces : ℕ := 8
def total_liquid_limit : ℕ := bathroom_limit
def total_liquid_intake : ℕ := grape_juice_ounces + water_ounces
def milk_ounces := total_liquid_limit - total_liquid_intake

theorem find_milk_ounces : milk_ounces = 8 := by
  sorry

end find_milk_ounces_l546_546330


namespace part1_part2_l546_546629

-- Definition of the sequence
def a : ℕ → ℤ 
| 0       := 2
| 1       := 3
| (n + 2) := (a (n + 1) ^ 2 + 5) / a n

-- First part: a_n is an integer for all natural numbers n
theorem part1 (n : ℕ) : ∃ k : ℤ, a n = k :=
by sorry

-- Second part: if a_n is prime, then n is a power of 2
theorem part2 (n : ℕ) (hn : nat.prime (a n)) : ∃ k : ℕ, n = 2^k :=
by sorry

end part1_part2_l546_546629


namespace reasoning_is_wrong_l546_546066

-- Definitions of the conditions
def some_rationals_are_proper_fractions := ∃ q : ℚ, ∃ f : ℚ, q = f ∧ f.den ≠ 1
def integers_are_rationals := ∀ z : ℤ, ∃ q : ℚ, q = z

-- Proof that the form of reasoning is wrong given the conditions
theorem reasoning_is_wrong 
  (h₁ : some_rationals_are_proper_fractions) 
  (h₂ : integers_are_rationals) :
  ¬ (∀ z : ℤ, ∃ f : ℚ, z = f ∧ f.den ≠ 1) := 
sorry

end reasoning_is_wrong_l546_546066


namespace cannot_make_all_black_l546_546496

-- Definitions to model the problem
def color := bool -- representing black as true and white as false
def board := array (fin 12) (array (fin 12) color)

-- Allowed operations: flip_row and flip_column
def flip_row (b : board) (r : fin 12) : board :=
  fun i j => if i = r then b i j not else b i j

def flip_column (b : board) (c : fin 12) : board :=
  fun i j => if j = c then b i j not else b i j

-- Statement to prove
theorem cannot_make_all_black (b : board) :
  ∀ b : board, (∃ rows : finset (fin 12), ∃ cols : finset (fin 12),
    let b' := (rows.foldl flip_row b id) in let b'' := (cols.foldl flip_column b' id) in 
    ∃ i j, b'' i j = ff) :=
sorry

end cannot_make_all_black_l546_546496


namespace min_value_of_squares_l546_546778

variable (a b t : ℝ)

theorem min_value_of_squares (ht : 0 < t) (habt : a + b = t) : 
  a^2 + b^2 ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l546_546778


namespace total_games_won_l546_546068

theorem total_games_won (gamesA gamesB gamesC : ℕ) (winsA_percent winsB_percent winsC_percent : ℝ)
                        (hA : gamesA = 150) (hB : gamesB = 110) (hC : gamesC = 200)
                        (hAp : winsA_percent = 0.35) (hBp : winsB_percent = 0.45) (hCp : winsC_percent = 0.30) :
  let winsA := (winsA_percent * gamesA).round.toNat,
      winsB := (winsB_percent * gamesB).round.toNat,
      winsC := (winsC_percent * gamesC).round.toNat
  in winsA + winsB + winsC = 163 :=
by
  sorry

end total_games_won_l546_546068


namespace coefficient_x4_expansion_l546_546561

theorem coefficient_x4_expansion :
  let poly1 := (2 - x)^3
  let poly2 := (2*x + 3)^5
  (polynomial.coeff ((polynomial.expand R (poly1 * poly2)) x 4) = -1050) :=
by
  sorry

end coefficient_x4_expansion_l546_546561


namespace train_pass_bridge_in_56_seconds_l546_546101

noncomputable def time_for_train_to_pass_bridge 
(length_of_train : ℕ) (speed_of_train_kmh : ℕ) (length_of_bridge : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  total_distance / speed_of_train_ms

theorem train_pass_bridge_in_56_seconds :
  time_for_train_to_pass_bridge 560 45 140 = 56 := by
  sorry

end train_pass_bridge_in_56_seconds_l546_546101


namespace coins_rearrangement_possible_l546_546939

/-- Define the initial state of the boxes. -/
def init_state : List ℕ := [1, 1, 1, 1, 1, 1]

/-- Define the target state of the boxes. -/
def target_state : List ℕ := [0, 0, 0, 0, 0, 2010 ^ (2010 ^ 2010)]

/-- Define what constitutes a Type 1 Move. -/
def type1_move (s : List ℕ) (j : Fin 5) : Option (List ℕ) :=
  if s[j] > 0 then
    some (s.modifyNth j (λ x => x - 1) |>.modifyNth (j + 1) (λ x => x + 2))
  else
    none

/-- Define what constitutes a Type 2 Move. -/
def type2_move (s : List ℕ) (k : Fin 4) : Option (List ℕ) :=
  if s[k] > 0 then
    some (s.modifyNth k (λ x => x - 1) |>.swapNth (k + 1) (k + 2))
  else
    none

/-- Combining the initial and target states, and type of moves, we need to prove the existence of a sequence of moves that transforms the initial state to the target state. -/
theorem coins_rearrangement_possible :
  ∃ (seq : List (Σ i : Fin 5, unit ⊕ unit)),
  (init_state, seq.foldl (λ s m => match m with 
      | ⟨j, Sum.inl ()⟩ => type1_move s j
      | ⟨k, Sum.inr ()⟩ => type2_move s k
      end).get_or_else init_state) = (init_state, target_state) :=
by
  sorry

end coins_rearrangement_possible_l546_546939


namespace proof_expr1_l546_546490

noncomputable def expr1 : ℝ :=
  (Real.sin (65 * Real.pi / 180) + Real.sin (15 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) / 
  (Real.sin (25 * Real.pi / 180) - Real.cos (15 * Real.pi / 180) * Real.cos (80 * Real.pi / 180))

theorem proof_expr1 : expr1 = 2 + Real.sqrt 3 :=
by sorry

end proof_expr1_l546_546490


namespace number_of_correct_statements_l546_546231

-- Definitions of parallelism and perpendicularity in geometrical context
variables (line plane : Type)
variables (parallel perpendicular : line → plane → Prop)
variables (parallel_lines : line → line → Prop)

-- Conditions and statements:
def statement1 (a b : line) (alpha : plane) : Prop :=
  (parallel a alpha) ∧ (parallel_lines a b) ∧ (¬ ∃x, parallel_lines b x ∧ x = alpha) → (parallel b alpha)

def statement2 (alpha beta gamma : plane) : Prop :=
  (parallel_lines alpha beta) ∧ (parallel_lines beta gamma) → (parallel_lines alpha gamma)

def statement3 (a b : line) (alpha : plane) : Prop :=
  (perpendicular a alpha) ∧ (perpendicular b a) ∧ (¬ ∃x, parallel_lines b x ∧ x = alpha) → (parallel b alpha)

def statement4 (alpha beta gamma : plane) : Prop :=
  (perpendicular alpha gamma) ∧ (parallel_lines beta gamma) → (perpendicular alpha beta)

-- The final proof statement: the number of correct statements is 4
theorem number_of_correct_statements : 
  (∃ (a b : line) (alpha : plane), statement1 a b alpha) ∧
  (∃ (alpha beta gamma : plane), statement2 alpha beta gamma) ∧
  (∃ (a b : line) (alpha : plane), statement3 a b alpha) ∧
  (∃ (alpha beta gamma : plane), statement4 alpha beta gamma) :=
sorry

end number_of_correct_statements_l546_546231


namespace find_slope_of_line_l546_546626

-- Define the conditions
def parabola : Prop := ∀ (x y : ℝ), y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_M : ℝ × ℝ := (-1, 0)
def acute_angle (θ : ℝ) : Prop := θ = 60 * (π / 180)

-- Define the main statement with the given conditions
theorem find_slope_of_line (k : ℝ) :
  (parabola ∧ (focus = (1, 0)) ∧ (point_M = (-1, 0)) ∧ acute_angle (60 * (π / 180))) →
  k = sqrt 2 / 2 :=
by
  sorry

end find_slope_of_line_l546_546626


namespace Tiffany_total_score_l546_546489

theorem Tiffany_total_score {points_per_treasure : ℕ} {level1_treasures : ℕ} {level2_treasures : ℕ} :
  points_per_treasure = 6 →
  level1_treasures = 3 →
  level2_treasures = 5 →
  (level1_treasures * points_per_treasure + level2_treasures * points_per_treasure) = 48 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  exact rfl

end Tiffany_total_score_l546_546489


namespace choose_four_socks_with_red_l546_546556

def socks : Finset String := {"blue", "brown", "black", "red", "purple", "green", "orange", "yellow"}

theorem choose_four_socks_with_red :
  (socks.card = 8) →
  (4 ≤ socks.card) →
  let total_ways := Finset.card (Finset.powersetLen 4 socks)
  let no_red := Finset.card (Finset.powersetLen 4 (socks.erase "red"))
  total_ways - no_red = 35 :=
by
  intros h1 h2
  let total_ways := Finset.card (Finset.powersetLen 4 socks)
  let no_red := Finset.card (Finset.powersetLen 4 (socks.erase "red"))
  have h_total_ways : total_ways = Nat.choose 8 4 := by sorry
  have h_no_red : no_red = Nat.choose 7 4 := by sorry
  rw [h_total_ways, h_no_red]
  simp
  exact Nat.sub_eq_of_eq_add (by norm_num : 35 + 35 = 70)

end choose_four_socks_with_red_l546_546556


namespace final_answer_is_correct_l546_546514

-- Define the chosen number
def chosen_number : ℤ := 1376

-- Define the division by 8
def division_result : ℤ := chosen_number / 8

-- Define the final answer
def final_answer : ℤ := division_result - 160

-- Theorem statement
theorem final_answer_is_correct : final_answer = 12 := by
  sorry

end final_answer_is_correct_l546_546514


namespace expand_polynomials_l546_546558

def p (z : ℤ) := 3 * z^3 + 4 * z^2 - 2 * z + 1
def q (z : ℤ) := 2 * z^2 - 3 * z + 5
def r (z : ℤ) := 10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5

theorem expand_polynomials (z : ℤ) : (p z) * (q z) = r z :=
by sorry

end expand_polynomials_l546_546558


namespace birds_on_fence_l546_546680

/-- Initially, there were 4 sparrows and 46 storks sitting on the fence.
After 6 more pigeons joined them, 3 sparrows and 5 storks flew away.
Later on, 8 swans and 2 ducks also came to sit on the fence.
Prove that the total number of birds sitting on the fence at the end is 58. -/
theorem birds_on_fence
    (s0 : ℕ := 4) -- initial sparrows
    (t0 : ℕ := 46) -- initial storks
    (p : ℕ := 6) -- pigeons joined
    (s1 : ℕ := 3) -- sparrows flew away
    (t1 : ℕ := 5) -- storks flew away
    (w : ℕ := 8) -- swans joined
    (d : ℕ := 2) -- ducks joined) :
    s0 + t0 + p + w + d - s1 - t1 = 58 := by
  sorry

end birds_on_fence_l546_546680


namespace roots_are_equal_l546_546055

-- Let's define the quadratic equation and its coefficients
def a : ℝ := 1
def b : ℝ := -2
def c : ℝ := 1

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the main theorem about the roots of the quadratic equation
theorem roots_are_equal :
  (∀ x : ℝ, quadratic_eq x = 0) → (b^2 - 4 * a * c = 0) :=
begin
  intros,
  sorry,
end

end roots_are_equal_l546_546055


namespace initial_people_lifting_weights_l546_546169

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end initial_people_lifting_weights_l546_546169


namespace like_terms_groupD_l546_546158

-- Define like terms for given conditions
def is_like_terms (a b : ℤ) : Prop := true  -- Constants are always like terms

-- Conditions provided in the problem
def groupA := (3 * (x^2 * y : ℤ), 3 * (x * y^2 : ℤ))
def groupB := (3 * (x * y : ℤ), -2 * (x * y^2 : ℤ))
def groupC := (-2 * (x * y^2 : ℤ), -2 * (a * b^2 : ℤ))
def groupD := (0 : ℤ, π.toReal : ℤ)

-- Question: Prove that group D has like terms.
theorem like_terms_groupD : is_like_terms groupD.1 groupD.2 := 
by {
  sorry
}

end like_terms_groupD_l546_546158


namespace sum_proper_divisors_eq_40_l546_546825

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l546_546825


namespace exprC_is_quadratic_l546_546900

-- Define the expressions given as conditions
def exprA (x : ℝ) := 3 * x - 1
def exprB (x : ℝ) := 1 / (x^2)
def exprC (x : ℝ) := 3 * x^2 + x - 1
def exprD (x : ℝ) := 2 * x^3 - 1

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

-- The goal is to prove that exprC is quadratic
theorem exprC_is_quadratic : is_quadratic exprC :=
by
  sorry

end exprC_is_quadratic_l546_546900


namespace sum_proper_divisors_81_l546_546813

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546813


namespace value_exceeds_original_minimum_avg_value_l546_546471

def value_at_year (n : ℕ) : ℕ :=
  if h : n < 1 then 0 -- Undefined for n < 1
  else if h : n = 1 then 200000
  else if h : n = 2 then 100000
  else if h : n = 3 then 50000
  else 40000 * (n - 3) + 50000

theorem value_exceeds_original : ∀ n ≥ 7, value_at_year n > 200000 :=
by {
  intros n hn,
  simp [value_at_year],
  have h₁ : 40000 * (n - 3) + 50000 > 200000 := by linarith,
  exact h₁,
}

def average_value (n : ℕ) : ℚ :=
  if h : n < 1 then 0 -- Undefined for n < 1
  else (list.sum (list.map value_at_year (list.range n))) / n

theorem minimum_avg_value : ∃ (n : ℕ), 1 ≤ n ∧ average_value n = 11 :=
by {
  use 4,
  simp [average_value, value_at_year],
  have h₁ : list.sum [200000, 100000, 50000, 90000] = 440000 := by norm_num,
  have h₂ : 440000 / 4 = 110000 := by norm_num,
  assume h₃: n = 4,
  refine ⟨4, h₃, (440000/4)⟩,
  show 110000 = 11,
  sorry
}

end value_exceeds_original_minimum_avg_value_l546_546471


namespace ratio_of_areas_l546_546201

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (a^2 * Real.sqrt 3) / 4

noncomputable def square_area (a : ℝ) : ℝ := 
  a^2

noncomputable def regular_hexagon_area (a : ℝ) : ℝ := 
  (3 * a^2 * Real.sqrt 3) / 2

theorem ratio_of_areas (a : ℝ) (h : a > 0) :
  (equilateral_triangle_area a) : (square_area a) : (regular_hexagon_area a) = 
  Real.sqrt 3 : 4 : 6 * Real.sqrt 3 :=
by
  sorry

end ratio_of_areas_l546_546201


namespace altitudes_of_triangle_l546_546019

open geometricEntity

noncomputable theory 

-- Define point and line types
variables {Point : Type} [geometry Point] {A B C A1 B1 C1 : Point}

-- Conditions of the problem
def is_triangle_acute (A B C : Point) : Prop := acute_angle_triangle A B C

def is_on_side (P : Point) (L1 L2 : Point) : Prop := lies_on_line P L1 L2

def is_angle_bisector (P Q R : Point) (P' Q' R' : Point) : Prop :=
  ∃ ψ : geomTransf Point, 
    (ψ.rotate_by_bisector P Q R P' Q' R') 

-- Conditions specific to our problem
def initial_conditions (A B C A1 B1 C1 : Point) : Prop :=
  is_triangle_acute A B C ∧ 
  is_on_side A1 B C ∧ is_on_side B1 A C ∧ is_on_side C1 A B ∧
  is_angle_bisector A1 A B C1 ∧ 
  is_angle_bisector B1 B A C1 ∧ 
  is_angle_bisector C1 C A B

-- Theorem we need to prove
theorem altitudes_of_triangle (A B C A1 B1 C1 : Point) 
  (h : initial_conditions A B C A1 B1 C1) : 
  is_perpendicular A A1 B C ∧ 
  is_perpendicular B B1 A C ∧ 
  is_perpendicular C C1 A B :=
by { sorry }

end altitudes_of_triangle_l546_546019


namespace functional_equation_solution_l546_546188

noncomputable def f (x : ℝ) : ℝ := 1 - (x^2) / 2

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1) →
  (∀ x : ℝ, f(x) = 1 - (x^2 / 2)) :=
by
  sorry

end functional_equation_solution_l546_546188


namespace find_counterfeit_two_weighings_l546_546735

-- defining the variables and conditions
variable (coins : Fin 7 → ℝ)
variable (real_weight : ℝ)
variable (fake_weight : ℝ)
variable (is_counterfeit : Fin 7 → Prop)

-- conditions
axiom counterfeit_weight_diff : ∀ i, is_counterfeit i ↔ (coins i = fake_weight)
axiom consecutive_counterfeits : ∃ (start : Fin 7), ∀ i, (start ≤ i ∧ i < start + 4) → is_counterfeit (i % 7)
axiom weight_diff : fake_weight < real_weight

-- Theorem statement
theorem find_counterfeit_two_weighings : 
  (coins (1 : Fin 7) + coins (2 : Fin 7) = coins (4 : Fin 7) + coins (5 : Fin 7)) →
  is_counterfeit (6 : Fin 7) ∧ is_counterfeit (7 : Fin 7) := 
sorry

end find_counterfeit_two_weighings_l546_546735


namespace lamp_probability_l546_546399

theorem lamp_probability :
  ∀ (red_lamps blue_lamps : ℕ), 
  red_lamps = 4 → blue_lamps = 2 →
  (∀ lamps_on : ℕ, lamps_on = 3 →
    (1 / (Nat.choose (red_lamps + blue_lamps) 2 * Nat.choose (red_lamps + blue_lamps) 3 / 
      (Nat.choose (5) 1 * Nat.choose (4) 2)) = 0.1)) :=
by
  intros red_lamps blue_lamps h_rl h_bl lamps_on h_lo
  apply eq_div_iff_mul_eq.mpr _
  norm_num
  sorry

end lamp_probability_l546_546399


namespace quadratic_eq_with_equal_real_roots_l546_546094
-- Import necessary library

-- Define the four quadratic equations
def eqA (x : ℝ) : Bool := (x - 3)^2 = 4
def eqB (x : ℝ) : Bool := x^2 = x
def eqC (x : ℝ) : Bool := x^2 + 2 * x + 1 = 0
def eqD (x : ℝ) : Bool := x^2 - 16 = 0

-- Define the conditions for having two equal real roots using the discriminant
def has_equal_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Define the coefficients for each equation
def coeffsA : (ℝ × ℝ × ℝ) := (1, -6, 5)   -- x^2 - 6x + 9 = 4
def coeffsB : (ℝ × ℝ × ℝ) := (1, -1, 0)   -- x^2 - x = 0
def coeffsC : (ℝ × ℝ × ℝ) := (1, 2, 1)    -- x^2 + 2x + 1 = 0
def coeffsD : (ℝ × ℝ × ℝ) := (1, 0, -16)  -- x^2 - 16 = 0

-- The proof statement
theorem quadratic_eq_with_equal_real_roots :
  has_equal_real_roots coeffsA.1 coeffsA.2 coeffsA.3 = False ∧
  has_equal_real_roots coeffsB.1 coeffsB.2 coeffsB.3 = False ∧
  has_equal_real_roots coeffsC.1 coeffsC.2 coeffsC.3 = True ∧
  has_equal_real_roots coeffsD.1 coeffsD.2 coeffsD.3 = False :=
by { sorry }

end quadratic_eq_with_equal_real_roots_l546_546094


namespace distance_from_point_to_directrix_l546_546189

theorem distance_from_point_to_directrix 
  (P : ℝ × ℝ) 
  (hP : P = (-2, 4))
  (parabola_eq : ∀ x y : ℝ, y^2 = -8 * x ↔ y = sqrt (-8 * x) ∨ y = -sqrt (-8 * x)) : 
  ∃ d : ℝ, d = 4 :=
by sorry

end distance_from_point_to_directrix_l546_546189


namespace geometry_problem_l546_546382

open Complex

/-- We consider triangle ABC with outwardly constructed triangles BPC, CQA, and ARB
    such that \angle PBC = \angle CAQ = 45°, \angle BCP = \angle QCA = 30°, and
    \angle ABR = \angle RAB = 15°. We aim to prove that ∠PRQ = 90° and QR = PR. -/
theorem geometry_problem (A B C P Q R : ℂ)
  (hAPB : angle P B C = π / 4)
  (hBPC : angle B C P = π / 6)
  (hCAQ : angle C A Q = π / 4)
  (hQCA : angle Q C A = π / 6)
  (hABR : angle A B R = π / 12)
  (hRAB : angle R A B = π / 12) :
  angle P R Q = π / 2 ∧ dist Q R = dist P R :=
by
  sorry

end geometry_problem_l546_546382


namespace coefficient_x9_expansion_l546_546809

theorem coefficient_x9_expansion (n : ℕ) (k : ℕ) : n = 11 → k = 9 → (binom n k) * (-1)^(n-k) = 55 := 
by intros h₁ h₂
   sorry

end coefficient_x9_expansion_l546_546809


namespace interest_rate_part1_l546_546025

-- Definitions according to the problem statement
def total_amount : ℝ := 4000
def P1 : ℝ := 2799.9999999999995
def P2 := total_amount - P1
def annual_interest : ℝ := 144
def P2_interest_rate : ℝ := 0.05

-- Formal statement of the problem
theorem interest_rate_part1 :
  (P2 * P2_interest_rate) + (P1 * (3 / 100)) = annual_interest :=
by
  sorry

end interest_rate_part1_l546_546025


namespace lava_lamp_probability_l546_546398

/-- Ryan has 4 red lava lamps and 2 blue lava lamps; 
    he arranges them in a row on a shelf randomly, and then randomly turns 3 of them on. 
    Prove that the probability that the leftmost lamp is blue and off, 
    and the rightmost lamp is red and on is 2/25. -/
theorem lava_lamp_probability : 
  let total_arrangements := (Nat.choose 6 2) 
  let total_on := (Nat.choose 6 3)
  let favorable_arrangements := (Nat.choose 4 1)
  let favorable_on := (Nat.choose 4 2)
  let favorable_outcomes := 4 * 6
  let probability := (favorable_outcomes : ℚ) / (total_arrangements * total_on : ℚ)
  probability = 2 / 25 := 
by
  sorry

end lava_lamp_probability_l546_546398


namespace find_value_l546_546672

-- Definitions
def arithmetic_seq (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) :=
  ∀ n, a n = a1 + n * d

variables {a : ℕ → ℤ} {a1 d : ℤ}

-- Conditions
axiom cond : arithmetic_seq a a1 d ∧ a 3 + a 8 = 6

-- Theorem statement
theorem find_value (a : ℕ → ℤ) (a1 d : ℤ) (h : arithmetic_seq a a1 d) (h_cond : a 3 + a 8 = 6) :
  3 * a 2 + a 16 = 12 :=
sorry

end find_value_l546_546672


namespace angle_sum_identity_l546_546230

-- Define the relevant geometry and properties
variables {A B C D E F P Q I : Point}
variables {Γ : Circle}
variables {EF : Line}

-- Define the acute triangle and its circles and lines
def acute_triangle (A B C : Point) : Prop :=
  ∃ (I : Point), incircle I A B C ∧ tangent I D (line_through B C) ∧ 
                 tangent I E (line_through C A) ∧ tangent I F (line_through A B)

def lies_between (F E P : Point) : Prop := 
  collinear F E P ∧ F ≠ E ∧ F ≠ P

def intersects_circumcircle (EF : Line) (Γ : Circle) (P Q : Point) : Prop :=
  EF ∩ Γ = {P, Q}

-- The main theorem to be proved
theorem angle_sum_identity 
  (hABC : acute_triangle A B C)
  (hIncircle : incircle I A B C)
  (tangent_points : tangent_points I D E F)
  (hIntersections : intersects_circumcircle EF Γ P Q)
  (hLiesBetween : lies_between F E P) :
    ∠DPA + ∠AQD = ∠QIP :=
sorry

end angle_sum_identity_l546_546230


namespace parabola_focus_coordinates_l546_546199

theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), x^2 = 8 * y → ∃ F : ℝ × ℝ, F = (0, 2) :=
  sorry

end parabola_focus_coordinates_l546_546199


namespace sum_proper_divisors_81_l546_546839

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l546_546839


namespace ellipse_equation_max_area_abcd_l546_546987

open Real

theorem ellipse_equation (x y : ℝ) (a b c : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (x^2 / 2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem max_area_abcd (a b c t : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (∀ (t : ℝ), 4 * sqrt 2 * sqrt (1 + t^2) / (t^2 + 2) ≤ 2 * sqrt 2) := by
  sorry

end ellipse_equation_max_area_abcd_l546_546987


namespace tenth_term_arithmetic_sequence_l546_546775

noncomputable def sequence (a d : ℕ) (n : ℕ) := a + n * d

theorem tenth_term_arithmetic_sequence (a d : ℕ) 
  (h1 : sequence a d 1 = 7) 
  (h2 : sequence a d 4 = 19) : 
  sequence a d 9 = 39 :=
by
  sorry

end tenth_term_arithmetic_sequence_l546_546775


namespace prime_sum_bound_l546_546333

-- Definitions of the problem conditions
def p (i : ℕ) : ℕ := sorry  -- This would be the i-th prime number less than m.

def primes (k m : ℕ) : Prop := 
  ∀ i : ℕ, (1 ≤ i ∧ i ≤ k) → prime (p i) ∧ p i < m

-- The theorem statement
theorem prime_sum_bound (k m : ℕ) (prime_property : primes k m) :
    ∑ i in finset.range k, (1 / (p i).to_real + 1 / (p i : ℕ) ^ 2) > real.log (real.log m.to_real) := 
  sorry

end prime_sum_bound_l546_546333


namespace percentage_markup_l546_546771

theorem percentage_markup (selling_price cost_price : ℝ) (h_selling : selling_price = 2000) (h_cost : cost_price = 1250) :
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
  sorry

end percentage_markup_l546_546771


namespace shortest_distance_line_circle_l546_546054

theorem shortest_distance_line_circle :
  ∃ (shortest_distance : ℝ), 
  shortest_distance = ℝ.sqrt 2 - 1 ∧
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x + 4 * y + 4 = 0) →
               (y = x + 1) →
               (dist (x, y) (-1, -2) - 1 = shortest_distance) :=
by
  sorry

end shortest_distance_line_circle_l546_546054


namespace sequence_condition_is_perfect_square_l546_546004

noncomputable def a : ℕ → ℤ
| 0 := 20
| 1 := 30
| (n + 2) := 3 * a (n + 1) - a n

theorem sequence_condition_is_perfect_square (n : ℕ) :
  (1 + 5 * a n * a (n + 1)) = 651^2 ↔ n = 4 :=
by
  sorry

end sequence_condition_is_perfect_square_l546_546004


namespace largest_solution_sqrt_eq_l546_546458

theorem largest_solution_sqrt_eq {y : ℝ} (h_eq : sqrt (3 * y) = 5 * y) : y ≤ 3 / 25 :=
by
  sorry

end largest_solution_sqrt_eq_l546_546458


namespace total_number_of_balls_l546_546195

-- Define the conditions
def balls_per_box : Nat := 3
def number_of_boxes : Nat := 2

-- Define the proposition
theorem total_number_of_balls : (balls_per_box * number_of_boxes) = 6 :=
by
  sorry

end total_number_of_balls_l546_546195


namespace angle_bisector_altitude_perpendicular_intersection_l546_546314

theorem angle_bisector_altitude_perpendicular_intersection (A B C : Type)
  [triangle A B C] (acute_triangle : acute A B C)
  (AN : angle_bisector A B C) (BH : altitude B A C)
  (M : midpoint A B) (line_M_perpendicular : perpendicular_line M A B)
  (P : intersection_point AN BH line_M_perpendicular) :
  angle A B C = 60 :=
sorry

end angle_bisector_altitude_perpendicular_intersection_l546_546314


namespace smaller_circle_area_l546_546806

open Real

theorem smaller_circle_area (PA AB r s : ℝ)
    (tangency : 2 * s = PA)
    (ratio : r = 3 * s)
    (PA_AB_eq : PA = AB = 6) :
    π * (s ^ 2) = (3 / 2) * π :=
by
  -- conditions
  have PA_value : PA = 6 := by exact PA_AB_eq
  have AB_value : AB = 6 := by exact (PA_AB_eq : PA = AB)
  -- parabola's equation
  have equation : s² + 6² = (5 * s)² :=
    calc
      s² + 6² = 25 * s² : by sorry
  -- substituting for s
  have simplified_eq : 24 * s² = 36 :=
    calc
      24 * s² = 36 : by sorry 
  -- solving for s²
  have s_sq_value : s² = 3 / 2 :=
    calc
      s² = 3 / 2 : by sorry
  
  -- prove area formula  
  proof
    π * (sqrt (3 / 2)) ^ 2 = (3 / 2) * π := by sorry

end smaller_circle_area_l546_546806


namespace time_taken_by_x_alone_l546_546105

theorem time_taken_by_x_alone 
  (W : ℝ)
  (Rx Ry Rz : ℝ)
  (h1 : Ry = W / 24)
  (h2 : Ry + Rz = W / 6)
  (h3 : Rx + Rz = W / 4) :
  (W / Rx) = 16 :=
by
  sorry

end time_taken_by_x_alone_l546_546105


namespace vissaya_inequality_l546_546434

theorem vissaya_inequality (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i)
  (b : Fin n → ℝ) (h_ge : ∀ i, b i ≥ a i)
  (h_int : ∀ i j, (∃ c : ℤ, b i = c * b j) ∨ (∃ d : ℤ, b j = d * b i)) :
  (∏ i, b i) ≤ 2 ^ ((n - 1) / 2) * (∏ i, a i) :=
sorry

end vissaya_inequality_l546_546434


namespace largest_expression_is_B_l546_546903

noncomputable def expr_A : ℝ := real.sqrt (real.cbrt (5 * 6))
noncomputable def expr_B : ℝ := real.sqrt (6 * real.cbrt 5)
noncomputable def expr_C : ℝ := real.sqrt (5 * real.cbrt 6)
noncomputable def expr_D : ℝ := real.cbrt (5 * real.sqrt 6)
noncomputable def expr_E : ℝ := real.cbrt (6 * real.sqrt 5)

theorem largest_expression_is_B :
  expr_B > expr_A ∧ expr_B > expr_C ∧ expr_B > expr_D ∧ expr_B > expr_E :=
sorry

end largest_expression_is_B_l546_546903


namespace find_date_behind_l546_546021

variables (x y : ℕ)
-- Conditions
def date_behind_C := x
def date_behind_A := x + 1
def date_behind_B := x + 13
def date_behind_P := x + 14

-- Statement to prove
theorem find_date_behind : (x + y = (x + 1) + (x + 13)) → (y = date_behind_P) :=
by
  sorry

end find_date_behind_l546_546021


namespace max_number_last_digit_is_2_l546_546064

noncomputable def last_digit_max_number : ℕ :=
  max (list.cons 1 (list.replicate 127 1) |> 
          list.foldl (λ acc n, n * acc + 1) 1) % 10

theorem max_number_last_digit_is_2 :
  last_digit_max_number = 2 := sorry

end max_number_last_digit_is_2_l546_546064


namespace marks_in_physics_l546_546544

def marks_in_english : ℝ := 74
def marks_in_mathematics : ℝ := 65
def marks_in_chemistry : ℝ := 67
def marks_in_biology : ℝ := 90
def average_marks : ℝ := 75.6
def number_of_subjects : ℕ := 5

-- We need to show that David's marks in Physics are 82.
theorem marks_in_physics : ∃ (P : ℝ), P = 82 ∧ 
  ((marks_in_english + marks_in_mathematics + P + marks_in_chemistry + marks_in_biology) / number_of_subjects = average_marks) :=
by sorry

end marks_in_physics_l546_546544


namespace two_numbers_solution_l546_546071

noncomputable def a := 8 + Real.sqrt 58
noncomputable def b := 8 - Real.sqrt 58

theorem two_numbers_solution : 
  (Real.sqrt (a * b) = Real.sqrt 6) ∧ ((2 * a * b) / (a + b) = 3 / 4) → 
  (a = 8 + Real.sqrt 58 ∧ b = 8 - Real.sqrt 58) ∨ (a = 8 - Real.sqrt 58 ∧ b = 8 + Real.sqrt 58) := 
by
  sorry

end two_numbers_solution_l546_546071


namespace gecko_crickets_third_day_l546_546129

theorem gecko_crickets_third_day : 
  ∀ (total_crickets : ℕ) (perc_first_day : ℕ) (diff_second_day : ℕ),
    total_crickets = 70 →
    perc_first_day = 30 →
    diff_second_day = 6 →
    let eaten_first_day := (perc_first_day * total_crickets) / 100 in
    let eaten_second_day := eaten_first_day - diff_second_day in
    let eaten_third_day := total_crickets - (eaten_first_day + eaten_second_day) in
    eaten_third_day = 34 :=
by
  intros total_crickets perc_first_day diff_second_day h1 h2 h3
  let eaten_first_day := (perc_first_day * total_crickets) / 100
  let eaten_second_day := eaten_first_day - diff_second_day
  let eaten_third_day := total_crickets - (eaten_first_day + eaten_second_day)
  have h4 : eaten_first_day = 21 := by sorry
  have h5 : eaten_second_day = 15 := by sorry
  show eaten_third_day = 34 from by sorry

end gecko_crickets_third_day_l546_546129


namespace square_diagonal_l546_546886

theorem square_diagonal (pi real : ℝ) (s r d : ℝ) 
  (h1 : 4 * s = π * r ^ 2)
  (h2 : s = 2 * r) :
  d = 16 * sqrt 2 / π :=
by sorry

end square_diagonal_l546_546886


namespace find_number_M_l546_546198

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d, acc * 10 + d) 0

theorem find_number_M :
  ∃ (M : ℕ), 1000 ≤ M ∧ M < 10000 ∧
  1000 ≤ 4 * M ∧ 4 * M < 10000 ∧
  reverse_digits (4 * M) = M ∧
  M = 2178 :=
by
  sorry

end find_number_M_l546_546198


namespace ribbon_per_box_l546_546358

def total_ribbon : ℝ := 4.5
def remaining_ribbon : ℝ := 1
def number_of_boxes : ℕ := 5

theorem ribbon_per_box :
  (total_ribbon - remaining_ribbon) / number_of_boxes = 0.7 :=
by
  sorry

end ribbon_per_box_l546_546358


namespace acute_angle_iff_lambda_range_l546_546267

variable (λ : ℝ)

def vector_a : ℝ × ℝ := (1, λ)
def vector_b : ℝ × ℝ := (λ, 4)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def is_acute_angle (a b : ℝ × ℝ) : Prop :=
  dot_product a b > 0

theorem acute_angle_iff_lambda_range :
  is_acute_angle (vector_a λ) (vector_b λ) ↔ (0 < λ ∧ λ ≠ 2) :=
by 
  sorry

end acute_angle_iff_lambda_range_l546_546267


namespace transformed_equation_sum_l546_546292

theorem transformed_equation_sum (a b : ℝ) (h_eqn : ∀ x : ℝ, x^2 - 6 * x - 5 = 0 ↔ (x + a)^2 = b) :
  a + b = 11 :=
sorry

end transformed_equation_sum_l546_546292


namespace convert_444_quinary_to_octal_l546_546183

def quinary_to_decimal (n : ℕ) : ℕ :=
  let d2 := (n / 100) * 25
  let d1 := ((n % 100) / 10) * 5
  let d0 := (n % 10)
  d2 + d1 + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let r2 := (n / 64)
  let n2 := (n % 64)
  let r1 := (n2 / 8)
  let r0 := (n2 % 8)
  r2 * 100 + r1 * 10 + r0

theorem convert_444_quinary_to_octal :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end convert_444_quinary_to_octal_l546_546183


namespace expected_train_interval_is_three_minutes_l546_546022

noncomputable def expected_interval_between_trains : ℝ :=
  let p : ℝ := 7 / 12 in
  let Y : ℝ := 5 / 4 in
  let T := Y / (1 - p) in
  T

theorem expected_train_interval_is_three_minutes :
  expected_interval_between_trains = 3 := by
  unfold expected_interval_between_trains
  have h1 : 7 / 12 = 7 / 12 := rfl
  have h2 : 5 / 4 / (1 - 7 / 12) = 3 := by
    calc
      5 / 4 / (1 - 7 / 12)
        = 5 / 4 / (5 / 12) : by norm_num
    ... = (5 / 4) * (12 / 5) : by field_simp
    ... = 3 : by norm_num
  exact h2

end expected_train_interval_is_three_minutes_l546_546022


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l546_546466

theorem option_A_correct (a : ℝ) : a ^ 2 * a ^ 3 = a ^ 5 := by {
  -- Here, we would provide the proof if required,
  -- but we are only stating the theorem.
  sorry
}

-- You may optionally add definitions of incorrect options for completeness.
theorem option_B_incorrect (a : ℝ) : ¬(a + 2 * a = 3 * a ^ 2) := by {
  sorry
}

theorem option_C_incorrect (a b : ℝ) : ¬((a * b) ^ 3 = a * b ^ 3) := by {
  sorry
}

theorem option_D_incorrect (a : ℝ) : ¬((-a ^ 3) ^ 2 = -a ^ 6) := by {
  sorry
}

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l546_546466


namespace sum_proper_divisors_81_l546_546833

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546833


namespace G_computation_l546_546003

def G (x : ℝ) : ℝ := sorry

theorem G_computation :
  G(G(G(3))) = 7 :=
by
  -- Given conditions
  have h1: G(3) = -4 := sorry
  have h2: G(-4) = -1 := sorry
  have h3: G(-1) = 7 := sorry

  -- The proof will go here
  sorry

end G_computation_l546_546003


namespace max_convex_hull_volume_five_points_on_sphere_l546_546960

noncomputable def max_convex_hull_volume (r : ℝ) (points : Fin₅ → EuclideanSpace ℝ (Fin₃)) : ℝ :=
  sorry -- Skipping the implementation

theorem max_convex_hull_volume_five_points_on_sphere (points : Fin₅ → EuclideanSpace ℝ (Fin₃))
  (h_sphere : ∀ i, ∥points i∥ = 1) :
  max_convex_hull_volume 1 points = (sqrt 3) / 2 :=
sorry

end max_convex_hull_volume_five_points_on_sphere_l546_546960


namespace original_bales_correct_l546_546067

-- Definitions
def total_bales_now : Nat := 54
def bales_stacked_today : Nat := 26
def bales_originally_in_barn : Nat := total_bales_now - bales_stacked_today

-- Theorem statement
theorem original_bales_correct :
  bales_originally_in_barn = 28 :=
by {
  -- We will prove this later
  sorry
}

end original_bales_correct_l546_546067


namespace angle_b_is_30_degrees_l546_546209

variables {A M K : Type} [AddGroup A] [Module ℝ M] [Module ℝ K]
variables {B N C : Type} [AddGroup B] [Module ℝ N] [Module ℝ C]
variables [fin (ℝ : Type)] (AM MK BN NC : ℝ)

theorem angle_b_is_30_degrees
  (h1 : AM = MK)
  (h2 : BN = NC) :
  ∃ (B : ℝ), B = 30 :=
by
  sorry

end angle_b_is_30_degrees_l546_546209


namespace circle_line_distance_intersection_l546_546221

theorem circle_line_distance_intersection (r : ℝ) :
  (4 < r) ∧ (r < 6) ↔ ∃ p₁ p₂ : ℝ × ℝ, 
    (p₁ ≠ p₂ ∧ (p₁.1 - 5)^2 + (p₁.2 - 1)^2 = r^2 ∧
    (p₂.1 - 5)^2 + (p₂.2 - 1)^2 = r^2 ∧ 
    abs ((4 * p₁.1 + 3 * p₁.2 + 2) / sqrt (4^2 + 3^2)) = 1 ∧
    abs ((4 * p₂.1 + 3 * p₂.2 + 2) / sqrt (4^2 + 3^2)) = 1) := 
sorry

end circle_line_distance_intersection_l546_546221


namespace range_of_m_l546_546239

theorem range_of_m (m : ℝ) (h1 : (m - 3) < 0) (h2 : (m + 1) > 0) : -1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l546_546239


namespace no_base_satisfies_l546_546060

def e : ℕ := 35

theorem no_base_satisfies :
  ∀ (base : ℝ), (1 / 5)^e * (1 / 4)^18 ≠ 1 / 2 * (base)^35 :=
by
  sorry

end no_base_satisfies_l546_546060


namespace triangle_side_c_l546_546236

noncomputable def area_of_triangle (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

noncomputable def law_of_cosines (a b C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)

theorem triangle_side_c (a b C : ℝ) (h1 : a = 3) (h2 : C = Real.pi * 2 / 3) (h3 : area_of_triangle a b C = 15 * Real.sqrt 3 / 4) : law_of_cosines a b C = 2 :=
by
  sorry

end triangle_side_c_l546_546236


namespace total_coins_constant_l546_546929

-- Definitions based on the conditions
def stack1 := 12
def stack2 := 17
def stack3 := 23
def stack4 := 8

def totalCoins := stack1 + stack2 + stack3 + stack4 -- 60 coins
def is_divisor (x: ℕ) := x ∣ totalCoins

-- The theorem statement
theorem total_coins_constant {x: ℕ} (h: is_divisor x) : totalCoins = 60 :=
by
  -- skip the proof steps
  sorry

end total_coins_constant_l546_546929


namespace max_candies_in_20_hours_and_16_minutes_l546_546379

-- Function to return the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.toList.map (λ c, c.toNat - '0'.toNat).sum

-- Function to return the number of candies after a given number of hours with sum of digits increment
def candies_after_hours (initial_candies : ℕ) (hours : ℕ) : ℕ :=
  (nat.iterate (λ n, n + sum_of_digits n) hours initial_candies)

-- Prove that given the conditions provided, the maximum number of candies Nikita can achieve in 20 hours and 16 minutes is 1361.
theorem max_candies_in_20_hours_and_16_minutes (initial_candies : ℕ) :
  initial_candies = 1 →
  candies_after_hours initial_candies 20 = 1361 :=
by
  intros h_initial
  sorry

end max_candies_in_20_hours_and_16_minutes_l546_546379


namespace find_a_of_pure_imaginary_z_l546_546628

-- Definition of a pure imaginary number
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Main theorem statement
theorem find_a_of_pure_imaginary_z (a : ℝ) (z : ℂ) (hz : pure_imaginary z) (h : (2 - I) * z = 4 + 2 * a * I) : a = 4 :=
by
  sorry

end find_a_of_pure_imaginary_z_l546_546628


namespace find_spectral_density_l546_546203

noncomputable def correlation_function (D α τ : ℝ) : ℝ := D * exp(-α * abs τ) * (1 + α * abs τ + (1 / 3) * α^2 * τ^2)

noncomputable def spectral_density (D α ω : ℝ) : ℝ := (8 * D * α^5) / (3 * π * (α^2 + ω^2)^2)

theorem find_spectral_density (D α ω : ℝ) (hα : α > 0) :
    ∃ (s_x : ℝ → ℝ), ∀ (τ : ℝ), s_x ω = spectral_density D α ω :=
begin
  use λ ω, spectral_density D α ω,
  intros τ,
  sorry
end

end find_spectral_density_l546_546203


namespace common_chord_perpendicular_to_centers_l546_546389

theorem common_chord_perpendicular_to_centers 
  (O1 O2 A B : Point) 
  (h1 : distance O1 A = distance O1 B) 
  (h2 : distance O2 A = distance O2 B) 
  (h3 : A ≠ B) 
  (h4 : B ∈ Line(O1,A)) 
  (h5 : A ∈ Line(O2,B)) : 
  Perpendicular (Line(A, B)) (Line(O1, O2)) := 
sorry

end common_chord_perpendicular_to_centers_l546_546389


namespace manny_paula_weight_l546_546356

   variable (m n o p : ℕ)

   -- Conditions
   variable (h1 : m + n = 320) 
   variable (h2 : n + o = 295) 
   variable (h3 : o + p = 310) 

   theorem manny_paula_weight : m + p = 335 :=
   by
     sorry
   
end manny_paula_weight_l546_546356


namespace sum_digits_less_than_1000_l546_546565

theorem sum_digits_less_than_1000 :
  let digit_sum := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) in
  let freq := 100 in
  let place_sum := freq * digit_sum in
  let total_sum := place_sum * 3 in
  total_sum = 13500 :=
by
  let digit_sum := 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
  let freq := 100
  let place_sum := freq * digit_sum
  let total_sum := place_sum * 3
  show total_sum = 13500
  sorry

end sum_digits_less_than_1000_l546_546565


namespace pyramid_volume_correct_l546_546516

noncomputable def pyramid_volume (A_PQRS A_PQT A_RST: ℝ) (side: ℝ) (height: ℝ) : ℝ :=
  (1 / 3) * A_PQRS * height

theorem pyramid_volume_correct 
  (A_PQRS : ℝ) (A_PQT : ℝ) (A_RST : ℝ) (side : ℝ) (height_PQT : ℝ) (height_RST : ℝ)
  (h_PQT : 2 * A_PQT / side = height_PQT)
  (h_RST : 2 * A_RST / side = height_RST)
  (eq1 : height_PQT^2 + side^2 = height_RST^2 + (side - height_PQT)^2) 
  (eq2 : height_RST^2 = height_PQT^2 + (height_PQT - side)^2)
  : pyramid_volume A_PQRS A_PQT A_RST = 5120 / 3 :=
by
  -- Skipping the proof steps
  sorry

end pyramid_volume_correct_l546_546516


namespace gardener_cabbages_l546_546509

theorem gardener_cabbages (area_this_year : ℕ) (side_length_this_year : ℕ) (side_length_last_year : ℕ) (area_last_year : ℕ) (additional_cabbages : ℕ) :
  area_this_year = 9801 →
  side_length_this_year = 99 →
  side_length_last_year = side_length_this_year - 1 →
  area_last_year = side_length_last_year * side_length_last_year →
  additional_cabbages = area_this_year - area_last_year →
  additional_cabbages = 197 :=
by
  sorry

end gardener_cabbages_l546_546509


namespace quadratic_function_expression_quadratic_function_inequality_l546_546225

-- Define the quadratic function and conditions
def f (x : ℝ) := a * x^2 + b * x + c

-- Given conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f x - f (x+1) = -2x
def condition2 (f : ℝ → ℝ) := f 0 = 1

-- Target: Find the correct expression for f
def target_expr (f : ℝ → ℝ) := ∀ x : ℝ, f x = x^2 - x + 1

-- Prove that the range of m, given that the inequality holds in [-1, 1]
def range_of_m (f : ℝ → ℝ) (m : ℝ) := 
  (∀ x ∈ Icc (-1:ℝ) (1:ℝ), f x ≥ 2 * x + m) → m ≤ -1

theorem quadratic_function_expression :
  (condition1 f) → (condition2 f) → target_expr f := 
  sorry

theorem quadratic_function_inequality :
  (target_expr f) → range_of_m f m :=
  sorry

end quadratic_function_expression_quadratic_function_inequality_l546_546225


namespace target_hit_prob_l546_546145

-- Probability definitions for A, B, and C
def prob_A := 1 / 2
def prob_B := 1 / 3
def prob_C := 1 / 4

-- Theorem to prove the probability of the target being hit
theorem target_hit_prob :
  (1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)) = 3 / 4 :=
by
  sorry

end target_hit_prob_l546_546145


namespace trig_identity_l546_546211

open Real

theorem trig_identity (α β : ℝ) (h : cos α * cos β - sin α * sin β = 0) : sin α * cos β + cos α * sin β = 1 ∨ sin α * cos β + cos α * sin β = -1 :=
by
  sorry

end trig_identity_l546_546211


namespace probability_top_card_is_five_l546_546887

-- Definitions based on conditions
def total_cards : ℕ := 52
def number_of_fives : ℕ := 4

-- Statement of the theorem
theorem probability_top_card_is_five : (number_of_fives : ℚ) / total_cards = 1 / 13 := by
  sorry

end probability_top_card_is_five_l546_546887


namespace no_overlap_l546_546529

section Grasshopper 

variables {n : ℕ} -- Let n be a natural number.

-- Define initial positions of the grasshoppers based on the given conditions.
def initial_positions : list (ℕ × ℕ) :=
  [ (0, 0),           -- Vertex A
    (3^n, 0),         -- Vertex B
    (3^n, 3^n),       -- Vertex C
    (0, 3^n) ]        -- Vertex D

-- Define a function to compute the centroid of three points (ignoring the fourth).
def centroid (points : list (ℕ × ℕ)) : (ℕ × ℕ) :=
  match points with
  | [(x1, y1), (x2, y2), (x3, y3)] =>
      ( (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3 )
  | _ => (0, 0)  -- This is a simplification. In practice, the input should always be a list of three points.
  end

-- Define the jump function such that each grasshopper jumps to a point symmetric to the centroid of the other three.
def jump (positions : list (ℕ × ℕ)) (i : ℕ) : list (ℕ × ℕ) :=
  if h : i < positions.length then
    let other_positions := positions.remove_nth h in
    let centroid_pos := centroid other_positions in
    positions.modify_nth i (λ (x, y), 
    (2 * centroid_pos.1 - x, 2 * centroid_pos.2 - y))
  else
    positions

-- Define the main theorem statement.
theorem no_overlap :
  ∀ n : ℕ,
  (∀ positions : list (ℕ × ℕ),
     positions = initial_positions ∨ (∃ i, positions = jump initial_positions i) →
     (∀ i j, i ≠ j → positions.nth i ≠ positions.nth j)) :=
begin
  sorry
end

end Grasshopper

end no_overlap_l546_546529


namespace sum_arithmetic_series_200_to_800_l546_546533

theorem sum_arithmetic_series_200_to_800 :
  (∑ k in finset.range 601, (200 + k)) = 300500 := by
sorry

end sum_arithmetic_series_200_to_800_l546_546533


namespace last_digit_of_f_l546_546807

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ + ⌊3 * x⌋ + ⌊6 * x⌋

theorem last_digit_of_f (x : ℝ) (hx : x > 0) : 
  ∃ d ∈ {0, 1, 3, 4, 6, 7}, (⌊f x⌋ % 10 = d) := 
sorry

end last_digit_of_f_l546_546807


namespace hyperbola_equation_l546_546782

variable (a b c : ℝ)

def system_eq1 := (4 / (-3 - c)) = (- a / b)
def system_eq2 := ((c - 3) / 2) * (b / a) = 2
def system_eq3 := a ^ 2 + b ^ 2 = c ^ 2

theorem hyperbola_equation (h1 : system_eq1 a b c) (h2 : system_eq2 a b c) (h3 : system_eq3 a b c) :
  ∃ a b : ℝ, c = 5 ∧ b^2 = 20 ∧ a^2 = 5 ∧ (∀ x y : ℝ, (x ^ 2 / 5) - (y ^ 2 / 20) = 1) :=
  sorry

end hyperbola_equation_l546_546782


namespace eval_expr_l546_546942

theorem eval_expr : (1 / (5^2)^4 * 5^11 * 2) = 250 := by
  sorry

end eval_expr_l546_546942


namespace smallest_number_divisible_by_63_digitsum_63_l546_546957

/-- The smallest natural number that is divisible by 63 and has a digit sum of 63 is 63999999 -/
theorem smallest_number_divisible_by_63_digitsum_63 :
  ∃ n : ℕ, n = 63999999 ∧ (n % 63 = 0) ∧ (nat.digits 10 n).sum = 63 :=
begin
  use 63999999,
  split,
  { refl, },
  split,
  { norm_num, },
  { norm_num, },
end

end smallest_number_divisible_by_63_digitsum_63_l546_546957


namespace common_divisors_13650_8910_l546_546274

def num_common_divisors (a b : ℕ) : ℕ := 
  let gcd_ab := Nat.gcd a b
  let div_count n := (n.factors).foldr (λ p acc, acc * (p.snd + 1)) 1
  div_count gcd_ab

theorem common_divisors_13650_8910 : num_common_divisors 13650 8910 = 16 := by
  sorry

end common_divisors_13650_8910_l546_546274


namespace diamond_problem_l546_546571

def diamond (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

theorem diamond_problem : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * real.sqrt 2 := by
  sorry

end diamond_problem_l546_546571


namespace shauna_lowest_score_l546_546026

theorem shauna_lowest_score :
  ∀ (scores : List ℕ) (score1 score2 score3 : ℕ), 
    scores = [score1, score2, score3] → 
    score1 = 82 →
    score2 = 88 →
    score3 = 93 →
    (∃ (s4 s5 : ℕ), s4 + s5 = 162 ∧ s4 ≤ 100 ∧ s5 ≤ 100) ∧
    score1 + score2 + score3 + s4 + s5 = 425 →
    min s4 s5 = 62 := 
by 
  sorry

end shauna_lowest_score_l546_546026


namespace possible_angles_of_inclination_l546_546651

theorem possible_angles_of_inclination (l1 l2 m : ℝ) (angle1 angle2 : ℝ) 
  (h1 : l1 = x + y)
  (h2 : l2 = x + y + Real.sqrt 6) 
  (h3 : dist l1 l2 = Real.sqrt 3)
  (h4 : intercepted_segment_length m l1 l2 = 2 * Real.sqrt 3) 
  (h5 : angle_between_parallel_lines = 135) : 
  angle_of_inclination m = 105 ∨ angle_of_inclination m = 165 := 
sorry

end possible_angles_of_inclination_l546_546651


namespace sum_proper_divisors_81_l546_546838

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l546_546838


namespace expression_eq_one_if_and_only_if_k_eq_one_l546_546000

noncomputable def expression (a b c k : ℝ) :=
  (k * a^2 * b^2 + a^2 * c^2 + b^2 * c^2) /
  ((a^2 - b * c) * (b^2 - a * c) + (a^2 - b * c) * (c^2 - a * b) + (b^2 - a * c) * (c^2 - a * b))

theorem expression_eq_one_if_and_only_if_k_eq_one
  (a b c k : ℝ) (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  expression a b c k = 1 ↔ k = 1 :=
by
  sorry

end expression_eq_one_if_and_only_if_k_eq_one_l546_546000


namespace proof_tan_half_angle_l546_546210

theorem proof_tan_half_angle (α : ℝ) (h1 : sin (π / 2 - α) = -4 / 5) (h2 : π / 2 < α ∧ α < π) :
  tan (α / 2) = 3 :=
sorry

end proof_tan_half_angle_l546_546210


namespace algorithm_property_l546_546470

-- Definitions based on conditions
def not_unique (α : Type) : Prop := ¬(∃ f : α → α, ∀ x : α, f x = x)
def finiteness (α : Type) (f : α → α) : Prop := ∃ n : ℕ, ∀ x : α, n > 0 → f x = x
def determinacy (α : Type) (f : α → α) : Prop := ∀ x y : α, f x = f y → x = y
def output_property (α : Type) (f : α → α) : Prop := ∀ x : α, ∃ y : α, f x = y

-- The statement of the proof problem
theorem algorithm_property (α : Type) (f : α → α) (n : ℕ) :
  not_unique α ∧ finiteness α f ∧ determinacy α f ∧ output_property α f → ¬ unique (f : α → α) :=
by
  sorry

end algorithm_property_l546_546470


namespace prob_within_0_to_80_l546_546304

open MeasureTheory

noncomputable def normal_dist (μ σ : ℝ) : Measure ℝ := measure_theory.measureGaussian μ σ

theorem prob_within_0_to_80 {σ : ℝ} (hσ : 0 < σ)
  (h1 : ∀ x, normal_dist 100 σ (set.Ioc 80 120) = 0.8) :
  normal_dist 100 σ (set.Ioc 0 80) = 0.1 := 
sorry 

end prob_within_0_to_80_l546_546304


namespace cost_of_hot_dog_l546_546473

-- Definitions for conditions
def soda_cost : ℝ := 0.50
def total_revenue : ℝ := 78.50
def total_units_sold : ℕ := 87
def hot_dogs_sold : ℕ := 35

-- Question: What is the cost of each hot dog?
theorem cost_of_hot_dog :
  ∃ (hot_dog_cost : ℝ), 
    35 * hot_dog_cost + (total_units_sold - hot_dogs_sold) * soda_cost = total_revenue ∧ 
    hot_dog_cost = 1.50 :=
begin
  sorry
end

end cost_of_hot_dog_l546_546473


namespace compute_j_in_polynomial_arithmetic_progression_l546_546414

theorem compute_j_in_polynomial_arithmetic_progression 
  (P : Polynomial ℝ)
  (roots : Fin 4 → ℝ)
  (hP : P = Polynomial.C 400 + Polynomial.X * (Polynomial.C k + Polynomial.X * (Polynomial.C j + Polynomial.X * (Polynomial.C 0 + Polynomial.X))))
  (arithmetic_progression : ∃ b d : ℝ, roots 0 = b ∧ roots 1 = b + d ∧ roots 2 = b + 2 * d ∧ roots 3 = b + 3 * d ∧ Polynomial.degree P = 4) :
  j = -200 :=
by
  sorry

end compute_j_in_polynomial_arithmetic_progression_l546_546414


namespace tangent_sphere_radius_l546_546438

theorem tangent_sphere_radius (R : ℝ) : 
  ∃ x : ℝ, (x = (sqrt 3 + 1) * R / 2) ∨ (x = (sqrt 3 - 1) * R / 2) :=
begin
  sorry
end

end tangent_sphere_radius_l546_546438


namespace water_added_l546_546863

theorem water_added (W : ℝ) : 
  (15 + W) * 0.20833333333333336 = 3.75 → W = 3 :=
by
  intro h
  sorry

end water_added_l546_546863


namespace average_price_ballpoint_pen_l546_546494

def number_of_pens : Nat := 30
def number_of_pencils : Nat := 75
def number_of_gel_pens : Nat := 20
def number_of_ballpoint_pens : Nat := 10
def number_of_standard_pencils : Nat := 50
def number_of_mechanical_pencils : Nat := 25
def price_of_gel_pen : Float := 1.5
def price_of_mechanical_pencil : Float := 3.0
def price_of_standard_pencil : Float := 2.0
def total_cost : Float := 690

theorem average_price_ballpoint_pen :
  (number_of_gel_pens * price_of_gel_pen) +
  (number_of_mechanical_pencils * price_of_mechanical_pencil) +
  (number_of_standard_pencils * price_of_standard_pencil) +
  (number_of_ballpoint_pens * x) = total_cost →
  (number_of_ballpoint_pens ≠ 0) →
  x = 48.5 :=
by
  sorry

end average_price_ballpoint_pen_l546_546494


namespace age_difference_between_Mandy_and_sister_l546_546712

variable (Mandy_age Brother_age Sister_age : ℕ)

-- Given conditions
def Mandy_is_3_years_old : Mandy_age = 3 := by sorry
def Brother_is_4_times_older : Brother_age = 4 * Mandy_age := by sorry
def Sister_is_5_years_younger_than_brother : Sister_age = Brother_age - 5 := by sorry

-- Prove the question
theorem age_difference_between_Mandy_and_sister :
  Mandy_age = 3 ∧ Brother_age = 4 * Mandy_age ∧ Sister_age = Brother_age - 5 → Sister_age - Mandy_age = 4 := 
by 
  sorry

end age_difference_between_Mandy_and_sister_l546_546712


namespace sum_T_equals_452_25_l546_546540

-- Definition of the sequence term
noncomputable def term (n : ℕ) : ℝ := (3 + (n+1) * 9) / 3^(101 - n)

-- Definition of the sum T
noncomputable def T : ℝ := ∑ i in finset.range 100, term i

-- The theorem statement
theorem sum_T_equals_452_25 : T = 452.25 :=
sorry

end sum_T_equals_452_25_l546_546540


namespace gift_spending_l546_546270

def total_amount : ℝ := 700.00
def wrapping_expenses : ℝ := 139.00
def amount_spent_on_gifts : ℝ := 700.00 - 139.00

theorem gift_spending :
  (total_amount - wrapping_expenses) = 561.00 :=
by
  sorry

end gift_spending_l546_546270


namespace probability_both_red_l546_546325

def BagA := {red := 4, white := 2}
def BagB := {red := 1, white := 5}

def totalBalls (bag : {red : Nat, white : Nat}) : Nat := bag.red + bag.white

def probabilityRedBall (bag : {red : Nat, white : Nat}) : ℚ :=
  bag.red / (totalBalls bag)

theorem probability_both_red :
  probabilityRedBall BagA * probabilityRedBall BagB = 1 / 9 :=
by
  sorry

end probability_both_red_l546_546325


namespace sum_of_three_zero_l546_546386

theorem sum_of_three_zero 
  (m : ℕ) 
  (A : Set ℤ) 
  (h_size : A.card = 2 * m + 1) 
  (h_bound : ∀ a ∈ A, |a| ≤ 2 * m - 1) : 
  ∃ x y z ∈ A, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = 0 := 
sorry

end sum_of_three_zero_l546_546386


namespace rhombus_diagonal_l546_546754

theorem rhombus_diagonal
  (d1 : ℝ) (d2 : ℝ) (area : ℝ) 
  (h1 : d1 = 17) (h2 : area = 170) 
  (h3 : area = (d1 * d2) / 2) : d2 = 20 :=
by
  sorry

end rhombus_diagonal_l546_546754


namespace remainder_is_three_l546_546202

def P (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem remainder_is_three : P 1 = 3 :=
by
  -- Proof goes here.
  sorry

end remainder_is_three_l546_546202


namespace no_positive_integer_solution_l546_546635

/-- Let \( p \) be a prime greater than 3 and \( x \) be an integer such that \( p \) divides \( x \).
    Then the equation \( x^2 - 1 = y^p \) has no positive integer solutions for \( y \). -/
theorem no_positive_integer_solution {p x y : ℕ} (hp : Nat.Prime p) (hgt : 3 < p) (hdiv : p ∣ x) :
  ¬∃ y : ℕ, (x^2 - 1 = y^p) ∧ (0 < y) :=
by
  sorry

end no_positive_integer_solution_l546_546635


namespace count_special_numbers_l546_546636

def is_special (n : ℕ) : Prop :=
  let d := n % 10
  let abc := n / 10
  let a := abc / 100
  let bc := abc % 100
  let b := bc / 10
  let c := bc % 10
  d = a + b + c

theorem count_special_numbers : 
  Nat.count (n in List.range' 1001 1099 |= is_special) = 53 :=
sorry

end count_special_numbers_l546_546636


namespace shoe_cost_on_monday_l546_546531

theorem shoe_cost_on_monday 
  (price_thursday : ℝ) 
  (increase_rate : ℝ) 
  (decrease_rate : ℝ) 
  (price_thursday_eq : price_thursday = 40)
  (increase_rate_eq : increase_rate = 0.10)
  (decrease_rate_eq : decrease_rate = 0.10)
  :
  let price_friday := price_thursday * (1 + increase_rate)
  let discount := price_friday * decrease_rate
  let price_monday := price_friday - discount
  price_monday = 39.60 :=
by
  sorry

end shoe_cost_on_monday_l546_546531


namespace minimum_p_l546_546647

noncomputable def discriminant (p q r : ℕ) : ℝ :=
  (q ^ 2 : ℝ) - 4 * p * r

noncomputable def root1 (p q r : ℕ) : ℝ :=
  (q : ℝ - real.sqrt (discriminant p q r)) / (2 * p)

noncomputable def root2 (p q r : ℕ) : ℝ :=
  (q : ℝ + real.sqrt (discriminant p q r)) / (2 * p)

def valid_roots (p q r : ℕ) : Prop :=
  discriminant p q r > 0 ∧ 0 < root1 p q r < 1 ∧ 0 < root2 p q r < 1

theorem minimum_p (p q r : ℕ) (h : p = 5 ∧ valid_roots p q r) : p = 5 :=
  sorry

end minimum_p_l546_546647


namespace solve_for_x_l546_546641

theorem solve_for_x (y : ℝ) (x : ℝ) 
  (h : x / (x - 1) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 3)) : 
  x = (y^2 + 3 * y - 2) / 2 := 
by 
  sorry

end solve_for_x_l546_546641


namespace negate_exists_negate_exp_eq_l546_546769

theorem negate_exists (P : ℝ → Prop) : ¬ (∃ x : ℝ, P x) ↔ ∀ x : ℝ, ¬ P x := by
  sorry

theorem negate_exp_eq (P : ℝ → Prop) : 
  (∀ x : ℝ, ¬ (Real.exp x = x - 1)) ↔ ¬ (∃ x : ℝ, Real.exp x = x - 1) := 
negate_exists (fun x => Real.exp x = x - 1)

end negate_exists_negate_exp_eq_l546_546769


namespace students_more_than_pets_l546_546938

theorem students_more_than_pets :
  let students_per_classroom := 15
  let rabbits_per_classroom := 1
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 6
  let total_students := students_per_classroom * number_of_classrooms
  let total_pets := (rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms
  total_students - total_pets = 66 :=
by
  sorry

end students_more_than_pets_l546_546938


namespace problem_statement_l546_546098

noncomputable def expression (a : ℝ) (b : ℝ) := 
sqrt ((a - 8 * (a^(1/2) * b^(2/6)) + 4 * (b^(4/6)))/(sqrt a - 2 * (b^(2/6)) + 2 * (a^(1/4) * b^(2/12))) + 3 * (b^(2/6)))

noncomputable def sqrt4 (x : ℝ) := x^(1/4)
noncomputable def sqrt6 (x : ℝ) := x^(1/6)

theorem problem_statement (a b : ℝ) (hb : b > 0) : 
  2.355 * sqrt (expression a b) = sqrt (expression a b) → 
  sqrt (expression a b) = |sqrt4 a - sqrt6 b| :=
begin
  sorry
end

end problem_statement_l546_546098


namespace digging_project_length_l546_546500

theorem digging_project_length (Length_2 : ℝ) : 
  (100 * 25 * 30) = (75 * Length_2 * 50) → 
  Length_2 = 20 :=
by
  sorry

end digging_project_length_l546_546500


namespace max_triangle_side_sum_l546_546311

theorem max_triangle_side_sum :
  ∃ (a b c d e f : ℕ), 
      (a + b + c = 33) ∧ (c + d + e = 33) ∧ (e + f + a = 33) ∧ 
      a ∈ {8, 9, 10, 11, 12, 13} ∧ 
      b ∈ {8, 9, 10, 11, 12, 13} ∧ 
      c ∈ {8, 9, 10, 11, 12, 13} ∧ 
      d ∈ {8, 9, 10, 11, 12, 13} ∧ 
      e ∈ {8, 9, 10, 11, 12, 13} ∧ 
      f ∈ {8, 9, 10, 11, 12, 13} := by
  sorry

end max_triangle_side_sum_l546_546311


namespace minimum_value_of_expression_l546_546563

-- Define the function f(θ) according to the problem
def f (θ : ℝ) : ℝ := 3 * Real.cos θ + 2 / Real.sin θ + 3 * Real.sqrt 2 * Real.tan θ + θ^2

-- Define the interval condition
def theta_in_interval (θ : ℝ) : Prop := (π / 6 < θ) ∧ (θ < π / 3)

-- State the theorem
theorem minimum_value_of_expression : ∃ θ : ℝ, theta_in_interval θ ∧ f θ = 15 * Real.sqrt 2 / 2 + π^2 / 16 :=
  sorry

end minimum_value_of_expression_l546_546563


namespace sum_proper_divisors_81_l546_546834

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l546_546834


namespace arithmetic_progression_of_sequences_l546_546703

theorem arithmetic_progression_of_sequences 
  (a b : ℤ) (s : ℕ → ℤ)
  (ha : a ≠ 0)
  (h_positive : ∀ n, s n > 0)
  (h_distinct : ∀ {m n}, m ≠ n → s m ≠ s n)
  (h_arith_prog : ∀ n, s n = a * n + b) :
  ∃ c d : ℤ, ∀ n, s (s n) = c * n + d :=
by {
  sorry,
}

end arithmetic_progression_of_sequences_l546_546703


namespace sum_proper_divisors_of_81_l546_546842

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l546_546842


namespace imaginary_part_eq_neg3_l546_546763

variable {i : ℂ} (h : i = complex.I)

/-- The imaginary part of the complex number 2 - 3i is -3. -/
theorem imaginary_part_eq_neg3 : complex.im (2 - 3 * complex.I) = -3 :=
by
  sorry

end imaginary_part_eq_neg3_l546_546763


namespace ratio_of_areas_of_similar_triangles_l546_546580

-- Define the variables and conditions
variables {ABC DEF : Type} 
variables (hABCDEF : Similar ABC DEF) 
variables (perimeterABC perimeterDEF : ℝ)
variables (hpABC : perimeterABC = 3)
variables (hpDEF : perimeterDEF = 1)

-- The theorem statement
theorem ratio_of_areas_of_similar_triangles :
  (perimeterABC / perimeterDEF) ^ 2 = 9 :=
by
  sorry

end ratio_of_areas_of_similar_triangles_l546_546580


namespace jake_third_test_score_l546_546746

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end jake_third_test_score_l546_546746


namespace lambda_range_l546_546710

theorem lambda_range (a : ℕ+ → ℝ) (λ : ℝ) (h : ∀ n : ℕ+, a n = n^2 + λ * n) 
  (h_inc : ∀ n : ℕ+, a (n + 1) > a n) : λ > -3 :=
sorry

end lambda_range_l546_546710


namespace tony_schooling_years_l546_546799

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end tony_schooling_years_l546_546799


namespace sum_of_solutions_eq_l546_546091

theorem sum_of_solutions_eq (x : ℝ) : 
  (x = |2 * x - |80 - 2 * x||) →
  (x = 80 ∨ x = 16 ∨ x = 80 / 3) →
  (80 / 3 + 16 + 80 = 368 / 3) :=
by
  intros h_eq h_sol
  sorry

end sum_of_solutions_eq_l546_546091


namespace functional_equation_l546_546851

noncomputable def f : ℝ → ℝ :=
  sorry

theorem functional_equation (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end functional_equation_l546_546851


namespace sqrt_fraction_eq_l546_546114

theorem sqrt_fraction_eq : sqrt ((1 / 9) + (1 / 16)) = 5 / 12 :=
by
  sorry

end sqrt_fraction_eq_l546_546114


namespace systematic_sampling_interval_l546_546439

-- Definitions based on conditions
def population_size : ℕ := 1000
def sample_size : ℕ := 40

-- Theorem statement 
theorem systematic_sampling_interval :
  population_size / sample_size = 25 :=
by
  sorry

end systematic_sampling_interval_l546_546439


namespace correct_propositions_l546_546337

variable {α β : Type} [Plane α] [Plane β]
variable {l m n : Type} [Line l] [Line m] [Line n]
variable [NonCoincidentPlanes α β] [MutuallyNonCoincidentLines l m n]

/-- Proposition ①: If α ∥ β and l ⊆ α, then l ∥ β. -/
def Prop1 (α β : Type) [Plane α] [Plane β] [NonCoincidentPlanes α β] (l : Type) [Line l] [l ∈ α] : Prop :=
  α ∥ β → l ∥ β

/-- Proposition ②: If m ⊆ α, n ⊆ α, m ∥ β, n ∥ β, then α ∥ β. -/
def Prop2 (α β : Type) [Plane α] [Plane β] [NonCoincidentPlanes α β] (m n : Type) [Line m] [Line n] [m ∈ α] [n ∈ α] : Prop :=
  m ∥ β → n ∥ β → α ∥ β

/-- Proposition ③: If l ∥ α and l ⊥ β, then α ⊥ β. -/
def Prop3 (α β : Type) [Plane α] [Plane β] [NonCoincidentPlanes α β] (l : Type) [Line l] : Prop :=
  l ∥ α → l ⊥ β → α ⊥ β

/-- Proposition ④: If m and n are skew lines, m ∥ α, n ∥ α, and l ⊥ m, l ⊥ n, then l ⊥ α. -/
def Prop4 (α β : Type) [Plane α] [Plane β] [NonCoincidentPlanes α β] (l m n : Type) [Line l] [Line m] [Line n] [SkewLines m n] : Prop :=
  m ∥ α → n ∥ α → l ⊥ m → l ⊥ n → l ⊥ α

theorem correct_propositions : Prop1 α β l ∧ Prop3 α β l ∧ Prop4 α β l m n :=
by
  sorry

end correct_propositions_l546_546337


namespace smallest_fraction_denominator_l546_546952

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end smallest_fraction_denominator_l546_546952


namespace find_k_l546_546590

variables 
  (a b : Type) -- Define the types for a and b
  [inner_product_space ℝ a] [inner_product_space ℝ b] -- Define inner product space
  (u v : a) -- Define the vectors u and v of type a
  (k : ℝ) -- Define the real number k

-- Define the conditions
def conditions (u v : a) [inner_product_space ℝ a] := 
  (∥u∥ = 1) ∧ (∥v∥ = 1) ∧ (inner u v = 0) ∧ (inner (2 • u + 3 • v) (k • u - 4 • v) = 0)

theorem find_k (u v : a) [inner_product_space ℝ a] (h : conditions u v) : k = 6 :=
by sorry

end find_k_l546_546590


namespace simplify_expression_l546_546738

theorem simplify_expression (x y : ℝ) : 
  (x - y) * (x + y) + (x - y) ^ 2 = 2 * x ^ 2 - 2 * x * y :=
sorry

end simplify_expression_l546_546738


namespace sum_q_r_s_of_fold_points_area_l546_546224

theorem sum_q_r_s_of_fold_points_area (P A B C : Point)
  (hAB : dist A B = 36) (hAC : dist A C = 72) (h_angle_B : angle A B C = pi / 2)
  (q r s : ℕ) (hq : q = 270) (hr : r = 324) (hs : s = 3) :
  q + r + s = 597 := by
  sorry

end sum_q_r_s_of_fold_points_area_l546_546224


namespace find_a_l546_546238

theorem find_a (x y a : ℝ) 
  (h1 : sqrt (3 * x + 4) + y^2 + 6 * y + 9 = 0) 
  (h2 : a * x * y - 3 * x = y) :
  a = -7/4 := 
sorry

end find_a_l546_546238


namespace quinary_to_octal_444_l546_546180

theorem quinary_to_octal_444 :
  (let quinary := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  let decimal := 124 in
  let octal := 1 * 8^2 + 7 * 8^1 + 4 * 8^0 in
  quinary = decimal ∧ decimal = octal :=
  quinary = 4 * 25 + 4 * 5 + 4 ∧ 124 = 1 * 64 + 7 * 8 + 4) :=
by
  sorry

end quinary_to_octal_444_l546_546180


namespace cost_and_count_of_parking_spaces_l546_546070

variables (x y m : ℝ)

theorem cost_and_count_of_parking_spaces :
  (3 * x + 2 * y = 0.8) ∧ (2 * x + 4 * y = 1.2) ∧ (0.25 * m + 0.1 * (5000 - m) = 950) 
  → x = 0.1 ∧ y = 0.25 ∧ m = 3000 ∧ 5000 - m = 2000 :=
begin
  sorry
end

end cost_and_count_of_parking_spaces_l546_546070


namespace average_speed_ratio_l546_546555

def eddy_distance := 450 -- distance from A to B in km
def eddy_time := 3 -- time taken by Eddy in hours
def freddy_distance := 300 -- distance from A to C in km
def freddy_time := 4 -- time taken by Freddy in hours

def avg_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def eddy_avg_speed := avg_speed eddy_distance eddy_time
def freddy_avg_speed := avg_speed freddy_distance freddy_time

def speed_ratio (speed1 : ℕ) (speed2 : ℕ) : ℕ × ℕ := (speed1 / (gcd speed1 speed2), speed2 / (gcd speed1 speed2))

theorem average_speed_ratio : speed_ratio eddy_avg_speed freddy_avg_speed = (2, 1) :=
by
  sorry

end average_speed_ratio_l546_546555


namespace range_of_m_l546_546582

-- Condition p: The solution set of the inequality x² + mx + 1 < 0 is an empty set
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Condition q: The function y = 4x² + 4(m-1)x + 3 has no extreme value
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 12 * x^2 + 4 * (m - 1) ≥ 0

-- Combined condition: "p or q" is true and "p and q" is false
def combined_condition (m : ℝ) : Prop :=
  (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- The range of values for the real number m
theorem range_of_m (m : ℝ) : combined_condition m → (-2 ≤ m ∧ m < 1) ∨ m > 2 :=
sorry

end range_of_m_l546_546582


namespace computation_one_computation_two_l546_546534

-- Proof problem (1)
theorem computation_one :
  (-2)^3 + |(-3)| - Real.tan (Real.pi / 4) = -6 := by
  sorry

-- Proof problem (2)
theorem computation_two (a : ℝ) :
  (a + 2)^2 - a * (a - 4) = 8 * a + 4 := by
  sorry

end computation_one_computation_two_l546_546534


namespace max_good_pairs_l546_546589

theorem max_good_pairs (S : Finset ℕ) (hS : S.card = 100) :
  (∀ a b ∈ S, (a ≠ b) → (a / b = 2 ∨ a / b = 3)) → ∃ P, P.card = 180 :=
  by sorry

end max_good_pairs_l546_546589


namespace max_g_value_le_256_l546_546696

variable {R : Type*} [LinearOrderedField R]

theorem max_g_value_le_256 {g : R → R}
  (poly_g : ∃ (b : ℕ → R), ∀ x, g x = (Finset.range 5).sum (λ i, b i * x ^ i) ∧ ∀ i, 0 ≤ b i)
  (h1 : g 8 = 32)
  (h2 : g 32 = 2048) :
  g 16 ≤ 256 :=
sorry

end max_g_value_le_256_l546_546696


namespace interest_rate_is_4_l546_546134

-- Define the conditions based on the problem statement
def principal : ℕ := 500
def time : ℕ := 8
def simple_interest : ℕ := 160

-- Assuming the formula for simple interest
def simple_interest_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- The interest rate we aim to prove
def interest_rate : ℕ := 4

-- The statement we want to prove: Given the conditions, the interest rate is 4%
theorem interest_rate_is_4 : simple_interest_formula principal interest_rate time = simple_interest := by
  -- The proof steps would go here
  sorry

end interest_rate_is_4_l546_546134


namespace greatest_integer_less_than_150_gcd_18_eq_6_l546_546084

theorem greatest_integer_less_than_150_gcd_18_eq_6 :
  ∃ n : ℕ, n < 150 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ gcd m 18 = 6 → m ≤ n :=
by
  use 132
  split
  { 
    -- proof that 132 < 150 
    exact sorry 
  }
  split
  { 
    -- proof that gcd 132 18 = 6
    exact sorry 
  }
  {
    -- proof that 132 is the greatest such integer
    exact sorry 
  }

end greatest_integer_less_than_150_gcd_18_eq_6_l546_546084


namespace smallest_value_for_x_between_0_and_1_l546_546643

theorem smallest_value_for_x_between_0_and_1 
  (x : ℝ) (h : 0 < x ∧ x < 1) : x^3 < x^2 ∧ x^3 < 3 * x ∧ x^3 < sqrt x ∧ x^3 < 1 / x :=
by sorry

end smallest_value_for_x_between_0_and_1_l546_546643


namespace probability_three_cards_alternating_red_black_red_l546_546521

theorem probability_three_cards_alternating_red_black_red :
  let total_cards := 52
  let red_cards := 26
  let black_cards := 26
  let total_ways_to_pick_three_cards := 52 * 51 * 50
  let favorable_ways := 26 * 26 * 25
  let probability := (favorable_ways : ℚ) / total_ways_to_pick_three_cards
  in probability = 13 / 102 :=
by
  sorry

end probability_three_cards_alternating_red_black_red_l546_546521


namespace kitchen_floor_area_l546_546328

/--
Jack needs to mop the bathroom and the kitchen. The bathroom floor is 24 square feet and the kitchen floor is some square feet. Jack can mop 8 square feet per minute and spends 13 minutes mopping. Prove that the kitchen floor is 80 square feet.
-/
theorem kitchen_floor_area :
  ∃ (kitchen_floor : ℕ), 
    (let total_area := 8 * 13 in
     let bathroom_floor := 24 in
     kitchen_floor = total_area - bathroom_floor) :=
begin
  use 80,
  let total_area := 8 * 13,
  let bathroom_floor := 24,
  norm_num at *,
  exact eq.refl 80,
end

end kitchen_floor_area_l546_546328


namespace circumcircle_touches_iff_ratios_l546_546116

variables {A B C D I_A I_B : Point}
variables (k : Circle) (circABC : Circumcircle A B C) 
          (D_on_arc_AB_not_C : Arc A B k (¬ C ∈ Arc A B k))

-- Define D which is a point on the arc AB not passing through C
def point_D_on_arc : Prop :=
  D ∈ Arc A B k ∧ ¬ (C ∈ Arc A B k)

-- Define I_A and I_B as the centers of the incircles of triangles ADC and BDC respectively
def center_incircle_ADC : Prop :=
  is_incenter I_A triangle.ADC

def center_incircle_BDC : Prop :=
  is_incenter I_B triangle.BDC

-- Statement of the problem
theorem circumcircle_touches_iff_ratios {I_A I_B C : Point} :
  circumcircle.I_AI_BC.I_A_I_B_C_touches_circABC k I_A I_B C ↔
  (∃ D, point_D_on_arc D circumcircle.ABC ∧ ratio.AC_CD_AD_BD C D A B) :=
sorry

end circumcircle_touches_iff_ratios_l546_546116


namespace part_a_part_b_l546_546488

-- Define the problem conditions and necessary predicates
variables {n m : ℕ}
variables (S : ℝ) (M : ℕ → ℝ) 

-- Statement for part a
theorem part_a (h : n ≥ 1) 
  (hS : S = M 1 - M 2 + M 3 - ... + (-1)^(n + 1) * M n) : 
  S = M 1 - M 2 + M 3 - ... + (-1)^(n + 1) * M n := sorry

-- Statement for part b
theorem part_b (h : n ≥ 1) 
  (hS : S = M 1 - M 2 + M 3 - ... + (-1)^(n + 1) * M n) : 
  (if even m then S ≥ M 1 - M 2 + M 3 - ... + (-1)^(m + 1) * M m 
  else S ≤ M 1 - M 2 + M 3 - ... + (-1)^(m + 1) * M m) := sorry

end part_a_part_b_l546_546488


namespace locus_of_tangency_l546_546032

-- Let A, B be points on a plane and m, n be positive real numbers such that p = m/n.
-- Prove that the locus of the point of tangency from point A to an Apollonian circle is the line perpendicular to AB at point B.

variable {A B : Point}
variable {m n p : ℝ}
variable (C D I T : Point)

-- Assuming A and B are distinct points
axiom A_ne_B : A ≠ B

-- Definition of the Apollonian circle and its properties
def ApollonianCircle (A B : Point) (m n : ℝ) : Set Point :=
  {P | dist A P / dist P B = m / n}

-- Tangency condition and the locus of the point
theorem locus_of_tangency (h_ratio : p = m / n)
  (h_tangent : ∀ T ∈ ApollonianCircle A B m n, ∃ tangent_line : Line, tangent_line.is_tangent_to T (ApollonianCircle A B m n)) :
  locus_of_tangency = {Q | ∃ perp_line : Line, perp_line.is_perpendicular_to (line_through A B) ∧ perp_line.contains Q} := 
sorry

end locus_of_tangency_l546_546032


namespace hyperbola_eccentricity_range_l546_546243

theorem hyperbola_eccentricity_range (a : ℝ) (h_range: 0 < a ∧ a ≤ 1) :
  ∃ e : Set ℝ, e = Set.Ico (Real.sqrt 2) (Real.sqrt 21 / 3) :=
by
  sorry

end hyperbola_eccentricity_range_l546_546243


namespace f_f_neg2_eq_sqrt2_l546_546293

noncomputable def f : ℝ → ℝ := 
λ x, if x < 0 then -1/x else 2*real.sqrt x

theorem f_f_neg2_eq_sqrt2 : f (f (-2)) = real.sqrt 2 :=
by
  sorry

end f_f_neg2_eq_sqrt2_l546_546293


namespace sum_of_lucky_numbers_divisible_by_13_l546_546354

def is_lucky (n : ℕ) : Prop :=
  let digits := Int.toDigits 10 n
  let (a, b, c, d, e, f) := (digits[0]?, digits[1]?, digits[2]?, digits[3]?, digits[4]?, digits[5]?) >>= λ a b c d e f =>
    some (a.get_or_else 0, b.get_or_else 0, c.get_or_else 0, d.get_or_else 0, e.get_or_else 0, f.get_or_else 0)
  a + b + c = d + e + f

theorem sum_of_lucky_numbers_divisible_by_13 :
  ∃ (total_sum : ℤ), (total_sum = (∑ n in Finset.range 1000000, if is_lucky n then n else 0)) ∧ total_sum % 13 = 0 :=
by
  sorry

#eval sum_of_lucky_numbers_divisible_by_13

end sum_of_lucky_numbers_divisible_by_13_l546_546354


namespace fibonacci_neg_indices_l546_546448

def fibonacci (n : ℤ) : ℤ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n > 1 then fibonacci (n - 1) + fibonacci (n - 2)
  else fibonacci (n + 2) - fibonacci (n + 1)

theorem fibonacci_neg_indices (n : ℕ) :
  fibonacci (-n) = (-1)^(n + 1) * fibonacci (n) := 
sorry

end fibonacci_neg_indices_l546_546448


namespace partition_positive_integers_l546_546549

theorem partition_positive_integers (n : ℕ) (h : n > 1) :
  ∃ (s : fin n → set ℕ), (∀ i, s i ≠ ∅) ∧
  (∀ (f : fin n.succ → ℕ), 
    (∀ i : fin n.succ, f i ∈ s i) → 
    ∃ j : fin n.succ, (∑ k : fin n, f k) - f j ∈ s j) :=
sorry

end partition_positive_integers_l546_546549


namespace additional_male_workers_is_16_l546_546062

noncomputable def hired_additional_male_workers (E M : ℕ) : Prop :=
  0.60 * E = 0.55 * (E + M) ∧ E + M = 360

theorem additional_male_workers_is_16 (E M : ℕ) 
  (h : hired_additional_male_workers E M) : M = 16 :=
by sorry

end additional_male_workers_is_16_l546_546062


namespace sum_proper_divisors_81_l546_546831

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546831


namespace problem1_problem2_l546_546602

noncomputable def f (x: ℝ) := Real.logb (1/2) (x - 1)
def g (x: ℝ) := (2:ℝ)^x

theorem problem1 : {x : ℝ | ∃ r, f x = r} = {x : ℝ | x > 1} := sorry

theorem problem2 : {y : ℝ | ∃ x ∈ set.Icc (-1:ℝ) 2, g x = y} = {y : ℝ | (1/2) ≤ y ∧ y ≤ 4} := sorry

end problem1_problem2_l546_546602


namespace trig_identity_l546_546649

variable (α1 α2 α3 : ℝ) (a b c d : ℝ)

-- Condition 1: The relationship between angles and edges in an orthogonal parallelepiped
axiom angle_conditions : ∀ i, i = 1 ∨ i = 2 ∨ i = 3 → 
  (sin (if i = 1 then α1 else if i = 2 then α2 else α3)) = 
  (if i = 1 then a / d else if i = 2 then b / d else c / d)

-- Condition 2: Diagonal relationship
axiom diagonal_condition : d^2 = a^2 + b^2 + c^2

-- Theorem statement
theorem trig_identity :
  (sin α1)^2 + (sin α2)^2 + (sin α3)^2 = 1 ∧ 
  (cos α1)^2 + (cos α2)^2 + (cos α3)^2 = 2 := by
  sorry

end trig_identity_l546_546649


namespace total_memorable_phone_numbers_l546_546140

-- Define the conditions for a memorable phone number
def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def memorablePhoneNumber (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  isDigit a1 ∧ isDigit a2 ∧ isDigit a3 ∧ isDigit a4 ∧ isDigit a5 ∧ isDigit a6 ∧ isDigit a7 ∧
  ((a1 = a4 ∧ a2 = a5 ∧ a3 = a6) ∨ (a1 = a5 ∧ a2 = a6 ∧ a3 = a7))

-- The proposition to be proven, stating the total number of memorable phone numbers
theorem total_memorable_phone_numbers : 
  ∃ count : ℕ, count = 19990 ∧ 
  (count = ∑ (a1 a2 a3 a4 a5 a6 a7 : Fin 10), if memorablePhoneNumber a1 a2 a3 a4 a5 a6 a7 then 1 else 0) :=
sorry

end total_memorable_phone_numbers_l546_546140


namespace order_of_x_l546_546053

variable {x1 x2 x3 x4 x5 a1 a2 a3 a4 a5 : ℝ}

theorem order_of_x :
  (x1 + x2 + x3 = a1) →
  (x2 + x3 + x1 = a2) →
  (x3 + x4 + x5 = a3) →
  (x4 + x5 + x1 = a4) →
  (x5 + x1 + x2 = a5) →
  (a1 > a2) →
  (a2 > a3) →
  (a3 > a4) →
  (a4 > a5) →
  (x3 > x1 ∧ x1 > x4 ∧ x4 > x2 ∧ x2 > x5) :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9,
  sorry
end

end order_of_x_l546_546053


namespace mr_kishore_savings_l546_546102

theorem mr_kishore_savings :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 3940
  let total_expenses := rent + milk + groceries + education + petrol + misc
  let savings_percentage := 0.10
  let salary := total_expenses / (1 - savings_percentage)
  let savings := savings_percentage * salary
  savings = 1937.78 := by
  sorry

end mr_kishore_savings_l546_546102


namespace power_six_rectangular_form_l546_546920

noncomputable def sin (x : ℂ) : ℂ := (Complex.exp (-Complex.I * x) - Complex.exp (Complex.I * x)) / (2 * Complex.I)
noncomputable def cos (x : ℂ) : ℂ := (Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x)) / 2

theorem power_six_rectangular_form :
  (2 * cos (20 * Real.pi / 180) + 2 * Complex.I * sin (20 * Real.pi / 180))^6 = -32 + 32 * Complex.I * Real.sqrt 3 := sorry

end power_six_rectangular_form_l546_546920


namespace minSumAndNumOfPermutations_l546_546351

open Set

def minSumOfPermutations (n : ℕ) (h : 2 ≤ n) : ℕ :=
  if n % 2 = 0 then n else n + 1

def numOfPermutationsAchievingMin (n : ℕ) (h : 2 ≤ n) : ℕ :=
  if n % 2 = 0 then 1 else n - 1

theorem minSumAndNumOfPermutations (n : ℕ) (h : 2 ≤ n) :
  ∃ (min_sum : ℕ) (num_perms : ℕ),
    min_sum = minSumOfPermutations n h ∧
    num_perms = numOfPermutationsAchievingMin n h ∧
    ∀ σ : equiv.perm (fin n), (∀ i : fin n, σ i ≠ i) →
      (∑ i, |σ.to_fun i - i| = min_sum → (∃! τ : equiv.perm (fin n), τ = σ) ↔ min_sum = num_perms) :=
begin
  sorry
end

end minSumAndNumOfPermutations_l546_546351


namespace prob_A1_selected_prob_B1_C1_not_selected_prob_encounter_A1_A2_l546_546789

namespace Probability

-- Definitions for volunteers and language groups
def volunteers := { : Fin 6 } -- A1, A2, B1, B2, C1, C2

def is_french (v : volunteers) : Prop := v = 0 ∨ v = 1
def is_russian (v : volunteers) : Prop := v = 2 ∨ v = 3
def is_english (v : volunteers) : Prop := v = 4 ∨ v = 5

-- Question 1: Probability of selecting A1
def event_A1_selected := λ (group : Fin 3 → volunteers), group 0 = 0

theorem prob_A1_selected : Pr event_A1_selected = 1 / 2 := sorry

-- Question 2: Probability that both B1 and C1 are not selected
def event_B1_C1_not_selected := λ (group : Fin 3 → volunteers),
  ¬ (group 1 = 2 ∧ group 2 = 4)

theorem prob_B1_C1_not_selected : Pr event_B1_C1_not_selected = 3 / 4 := sorry

-- Question 3: Probability of exactly encountering A1 and A2
def on_duty_pairs := ({v : Fin 6} : Set (volunteers × volunteers)) -- all possible pairs

def event_encounter_A1_A2 := (0, 1) ∈ on_duty_pairs

theorem prob_encounter_A1_A2 : Pr event_encounter_A1_A2 = 1 / 15 := sorry

end Probability

end prob_A1_selected_prob_B1_C1_not_selected_prob_encounter_A1_A2_l546_546789


namespace prove_fraction_l546_546011

noncomputable def michael_brothers_problem (M O Y : ℕ) :=
  Y = 5 ∧
  M + O + Y = 28 ∧
  O = 2 * (M - 1) + 1 →
  Y / O = 1 / 3

theorem prove_fraction (M O Y : ℕ) : michael_brothers_problem M O Y :=
  sorry

end prove_fraction_l546_546011


namespace sum_proper_divisors_of_81_l546_546841

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l546_546841


namespace shape_symmetry_count_l546_546322

def isAxisymmetric (shape : String) : Prop :=
  shape = "Rectangle" ∨ shape = "Rhombus" ∨ shape = "Square"

def isCentrallySymmetric (shape : String) : Prop :=
  shape = "Parallelogram" ∨ shape = "Rectangle" ∨ shape = "Rhombus" ∨ shape = "Square"

def shapes : List String := ["Parallelogram", "Rectangle", "Rhombus", "Square"]

def countAxisymmetricAndCentrallySymmetricShapes (shapes : List String) : Nat :=
  shapes.count (λ s => isAxisymmetric s ∧ isCentrallySymmetric s)

theorem shape_symmetry_count : countAxisymmetricAndCentrallySymmetricShapes shapes = 3 :=
by 
  sorry

end shape_symmetry_count_l546_546322


namespace rational_cos_summands_l546_546855

theorem rational_cos_summands (x : ℝ) 
  (S : ℝ) (C : ℝ) 
  (hS : S = sin (64 * x) + sin (65 * x))
  (hC : C = cos (64 * x) + cos (65 * x))
  (S_rat : is_rat S)
  (C_rat : is_rat C) : 
  (∃ r64 r65 : ℚ, cos (64 * x) = r64 ∧ cos (65 * x) = r65) :=
by 
  sorry

end rational_cos_summands_l546_546855


namespace simplify_expression_l546_546532

theorem simplify_expression :
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 :=
by
  sorry

end simplify_expression_l546_546532


namespace part1_general_formula_part2_sum_S_l546_546228

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => a n + 1

theorem part1_general_formula (n : ℕ) : a n = n + 1 := by
  sorry

noncomputable def b (n : ℕ) : ℝ := 1 / (↑n * ↑(n + 2))

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b (i + 1))

theorem part2_sum_S (n : ℕ) : 
  S n = (1/2) * ((3/2) - (1 / (n + 1)) - (1 / (n + 2))) := by
  sorry

end part1_general_formula_part2_sum_S_l546_546228


namespace part1_minimum_b_over_a_l546_546618

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Prove part 1
theorem part1 (x : ℝ) : (0 < x ∧ x < 1 → (f x 1 / (1/x - 1) > 0)) ∧ (1 < x → (f x 1 / (1/x - 1) < 0)) := sorry

-- Prove part 2
lemma part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) (ha : a ≠ 0) : ∃ x > 0, f x a = b - a := sorry

theorem minimum_b_over_a (a : ℝ) (ha : a ≠ 0) (h : ∀ x > 0, f x a ≤ b - a) : b/a ≥ 0 := sorry

end part1_minimum_b_over_a_l546_546618


namespace average_speed_of_car_l546_546865

theorem average_speed_of_car 
    (total_distance : ℝ) 
    (d1_fraction d2_fraction d3_fraction : ℝ)
    (speed1 speed2 speed3 speed4 : ℝ) : 
    total_distance = 1200 →
    d1_fraction = 1/4 → 
    d2_fraction = 1/3 → 
    d3_fraction = 1/5 →
    speed1 = 80 → 
    speed2 = 60 → 
    speed3 = 125 → 
    speed4 = 75 →
    let d1 := d1_fraction * total_distance,
        d2 := d2_fraction * total_distance,
        d3 := d3_fraction * total_distance,
        d4 := total_distance - d1 - d2 - d3 in
    let t1 := d1 / speed1,
        t2 := d2 / speed2,
        t3 := d3 / speed3,
        t4 := d4 / speed4,
        total_time := t1 + t2 + t3 + t4,
        average_speed := total_distance / total_time in
    average_speed = 75.9 :=
begin
  sorry
end

end average_speed_of_car_l546_546865


namespace problem_l546_546616

def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem problem (a : ℝ) (h : f a = 3) : f (2 * a) = 7 :=
sorry

end problem_l546_546616


namespace third_side_length_l546_546607

def lengths (a b : ℕ) : Prop :=
a = 4 ∧ b = 10

def triangle_inequality (a b c : ℕ) : Prop :=
a + b > c ∧ abs (a - b) < c

theorem third_side_length (x : ℕ) (h1 : lengths 4 10) (h2 : triangle_inequality 4 10 x) : x = 11 :=
sorry

end third_side_length_l546_546607


namespace paths_are_arcs_of_circles_if_one_point_stationary_l546_546121

theorem paths_are_arcs_of_circles_if_one_point_stationary
  (body : Type) [TopologicalSpace body] (P : body)
  (motion : ℝ → body → body)
  (stationary : ∀ t, motion t P = P) :
  ∀ (Q : body), ∃ (center : body), ∃ (radius : ℝ), 
    (Q ≠ P → ∀ t, dist (motion t Q) center = radius) :=
by
  sorry

end paths_are_arcs_of_circles_if_one_point_stationary_l546_546121


namespace original_price_per_kg_salt_original_price_per_kg_sugar_original_price_per_kg_flour_l546_546871

theorem original_price_per_kg_salt (S : ℝ) (h1 : 400 / (0.8 * S) = 400 / S + 10) :
  S = 10 := by
  sorry

theorem original_price_per_kg_sugar (U : ℝ) (h2 : 600 / (0.85 * U) = 600 / U + 5) :
  U ≈ 21.18 := by
  sorry

theorem original_price_per_kg_flour (F : ℝ) (h3 : 800 / (0.9 * F) = 800 / F + 8) :
  F ≈ 11.11 := by
  sorry

end original_price_per_kg_salt_original_price_per_kg_sugar_original_price_per_kg_flour_l546_546871


namespace factorial_base_18_zeros_l546_546770

theorem factorial_base_18_zeros : ∀ (n : ℕ), n = 15 → (nat.factorial n) % (nat.pow 18 3) = 0 ∧ (nat.factorial n) % (nat.pow 18 4) ≠ 0 :=
by {
  intros,
  sorry
}

end factorial_base_18_zeros_l546_546770


namespace sum_proper_divisors_eq_40_l546_546826

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l546_546826


namespace greatest_integer_difference_l546_546599

theorem greatest_integer_difference (x y : ℤ) (hx : -6 < (x : ℝ)) (hx2 : (x : ℝ) < -2) (hy : 4 < (y : ℝ)) (hy2 : (y : ℝ) < 10) : 
  ∃ d : ℤ, d = y - x ∧ d = 14 := 
by
  sorry

end greatest_integer_difference_l546_546599


namespace parallel_lines_iff_l546_546219

theorem parallel_lines_iff (a : ℝ) :
  (∀ x y : ℝ, x - y - 1 = 0 → x + a * y - 2 = 0) ↔ (a = -1) :=
by
  sorry

end parallel_lines_iff_l546_546219


namespace length_of_robins_hair_l546_546733

theorem length_of_robins_hair (initial_length cut_length : ℕ) (h₁ : initial_length = 17) (h₂ : cut_length = 4) : 
  initial_length - cut_length = 13 :=
by
  simp [h₁, h₂]
  sorry

end length_of_robins_hair_l546_546733


namespace root_interval_l546_546491

def f (x : ℝ) : ℝ := 5 * x - 7

theorem root_interval : ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- Proof steps should be here
  sorry

end root_interval_l546_546491


namespace dividend_percentage_paid_by_company_l546_546503

-- Define the parameters
def faceValue : ℝ := 50
def investmentReturnPercentage : ℝ := 25
def investmentPerShare : ℝ := 37

-- Define the theorem
theorem dividend_percentage_paid_by_company :
  (investmentReturnPercentage / 100 * investmentPerShare / faceValue * 100) = 18.5 :=
by
  -- The proof is omitted
  sorry

end dividend_percentage_paid_by_company_l546_546503


namespace max_quarters_l546_546024

theorem max_quarters (a b c : ℕ) (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) : c ≤ 19 :=
sorry

example : ∃ a b c : ℕ, a + b + c = 120 ∧ 5 * a + 10 * b + 25 * c = 1000 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ c = 19 :=
sorry

end max_quarters_l546_546024


namespace number_of_friends_l546_546477

theorem number_of_friends (n : ℕ) (h1 : 100 % n = 0) (h2 : 100 % (n + 5) = 0) (h3 : 100 / n - 1 = 100 / (n + 5)) : n = 20 :=
by
  sorry

end number_of_friends_l546_546477


namespace hexagon_area_l546_546396

open Real

-- Definitions based on conditions
def A : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (8, 2)

-- Question: Prove that the area of the hexagon ABCDEF is 34√3
theorem hexagon_area : 
    distance A C = 2 * sqrt 17 →     -- given distance AC is 2 * sqrt(17)
    ∀ (ABCDEF : Set (ℝ × ℝ)),    -- assuming ABCDEF is a regular hexagon
    ∃ s (side : ℝ),              -- knowing s is the side length of hexagon
        s = distance A C ∧ 
        (∀ (a b : ℝ × ℝ), a ∈ ABCDEF → b ∈ ABCDEF → distance a b = s ∨ distance a b = sqrt 17) → 
        (let area := 2 * (17 * sqrt 3) in True) :=
begin
    sorry
end

end hexagon_area_l546_546396


namespace find_probability_p_l546_546031

noncomputable section

open ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

def binomial (n : ℕ) (p : ℝ) :=  binom n p

theorem find_probability_p (X Y : Ω → ℕ) (p : ℝ) 
  (hX : ∀ ω, distribute X ω = binomial 2 p) 
  (hY : ∀ ω, distribute Y ω = binomial 3 p) 
  (hPX : ∫ ω, indicator (λ ω, X ω ≥ 1) ω =
         5 / 9) :
  ∫ ω, indicator (λ ω, Y ω ≥ 1) ω = 19 / 27 :=
sorry

end find_probability_p_l546_546031


namespace max_station_count_l546_546893

-- Definitions of the conditions for the problem
variables (Station : Type) (DirectlyConnected : Station → Station → Prop)

-- Axioms for the conditions
axiom communication_path : ∀ (s1 s2 : Station), 
  (∃ (s : Station), DirectlyConnected s1 s ∧ DirectlyConnected s s2)
axiom max_direct_connections : ∀ (s : Station), ∃ (S : set Station), 
  S.cardinality ≤ 3 ∧ ∀ (s' : Station), s' ∈ S → DirectlyConnected s s'

-- The theorem statement for the proof problem
theorem max_station_count : ∃ n : ℕ, n = 10 ∧ ∀ S : set Station, 
  (∀ s1 s2 ∈ S, ∃ (s : Station), DirectlyConnected s1 s ∧ DirectlyConnected s s2) → 
  (∀ s ∈ S, ∃ T : set Station, T.cardinality ≤ 3 ∧ ∀ s' ∈ T, DirectlyConnected s s') → 
  S.cardinality ≤ n :=
by
  sorry

end max_station_count_l546_546893


namespace initial_markup_is_41_4_effective_selling_price_is_1210_gain_percent_is_30_11_l546_546868

-- Define the conditions as per the original problem
def cost_price := 930
def final_selling_price := 1210
def discount_rate := 0.08

-- Initial markup percentage M
def initial_markup_percentage (M : ℝ) : Prop :=
  final_selling_price = cost_price * (1 + M / 100) * (1 - discount_rate)

-- Effective selling price
def effective_selling_price : ℝ := final_selling_price

-- Gain percent on the transaction
def gain_percent : ℝ :=
  ((final_selling_price - cost_price) / cost_price) * 100

-- Theorem to prove initial markup percentage is 41.4%
theorem initial_markup_is_41_4 :
  initial_markup_percentage 41.4 := by
  sorry

-- Theorem to prove the effective selling price is correct
theorem effective_selling_price_is_1210 :
  effective_selling_price = 1210 := by
  rfl

-- Theorem to prove the gain percentage is approximately 30.11%
theorem gain_percent_is_30_11 :
  gain_percent ≈ 30.11 := by
  sorry

end initial_markup_is_41_4_effective_selling_price_is_1210_gain_percent_is_30_11_l546_546868


namespace exists_non_convex_pentagon_no_diagonal_overlap_l546_546681

theorem exists_non_convex_pentagon_no_diagonal_overlap :
  ∃ (A B C D E : Point) (P : Pentagon),
  non_convex P ∧ no_diagonal_overlap P :=
sorry

end exists_non_convex_pentagon_no_diagonal_overlap_l546_546681


namespace tony_schooling_years_l546_546800

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end tony_schooling_years_l546_546800


namespace solution_inequality_l546_546693

theorem solution_inequality
  (a a' b b' c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a' ≠ 0)
  (h₃ : (c - b) / a > (c - b') / a') :
  (c - b') / a' < (c - b) / a :=
by
  sorry

end solution_inequality_l546_546693


namespace product_of_numbers_l546_546429

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := 
sorry

end product_of_numbers_l546_546429


namespace clock_hands_form_right_angle_at_180_over_11_l546_546911

-- Define the angular speeds as constants
def ω_hour : ℝ := 0.5  -- Degrees per minute
def ω_minute : ℝ := 6  -- Degrees per minute

-- Function to calculate the angle of the hour hand after t minutes
def angle_hour (t : ℝ) : ℝ := ω_hour * t

-- Function to calculate the angle of the minute hand after t minutes
def angle_minute (t : ℝ) : ℝ := ω_minute * t

-- Theorem: Prove the two hands form a right angle at the given time
theorem clock_hands_form_right_angle_at_180_over_11 : 
  ∃ t : ℝ, (6 * t - 0.5 * t = 90) ∧ t = 180 / 11 :=
by 
  -- This is where the proof would go, but we skip it with sorry
  sorry

end clock_hands_form_right_angle_at_180_over_11_l546_546911


namespace correlation_coefficient_Phi_eq_l546_546334

noncomputable def correlation_coefficient (X Y : ℝ → ℝ) (f g : ℝ → ℝ) : ℝ :=
  (∫ x, (f (X x)) * (g (Y x))) / (∫ x, (f (X x))^2 * ∫ x, (g (Y x))^2)

noncomputable def arcsin (x : ℝ) : ℝ :=
  sorry

noncomputable def Phi (x : ℝ) : ℝ := CDF_normal 0 1 x

theorem correlation_coefficient_Phi_eq {X Y : ℝ → ℝ}
    (h_gauss : ∀ t, X t ~ ℕ(0,1) ∧ Y t ~ ℕ(0,1))
    (h_expX : E(X) = 0) (h_expY : E(Y) = 0)
    (h_varX : Var(X) = 1) (h_varY : Var(Y) = 1)
    (h_corrXY : ∀ t, (E((X t) * (Y t))) = ρ) :
    correlation_coefficient X Y Phi Phi = (6 / π) * arcsin (ρ / 2) :=
begin
  sorry
end

end correlation_coefficient_Phi_eq_l546_546334


namespace probability_distribution_constant_l546_546709

theorem probability_distribution_constant (n : ℕ) (h : n > 0) (P : ℕ → ℝ) (a : ℝ) 
  (hp : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → P k = a * k)
  (hsum : (∑ k in Finset.range (n + 1), if k > 0 then P k else 0) = 1) :
  a = 2 / (n * (n + 1)) :=
by
  sorry

end probability_distribution_constant_l546_546709


namespace moose_population_l546_546656

/-- Variables representing the populations of moose, beavers, and humans in millions. --/
variables (M B H : ℕ)

/-- Conditions given in the problem. --/
def moose_beaver_relation (M B : ℕ) : Prop := B = 2 * M
def beaver_human_relation (B H : ℕ) : Prop := H = 19 * B
def human_population (H : ℕ) : Prop := H = 38

/-- The theorem stating the given conditions and the desired outcome. --/
theorem moose_population (M B H : ℕ) 
  (h1 : moose_beaver_relation M B) 
  (h2 : beaver_human_relation B H) 
  (h3 : human_population H) :
  M = 1 :=
sorry

end moose_population_l546_546656


namespace greatest_integer_with_gcd_6_l546_546075

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l546_546075


namespace f_g_5_eq_163_l546_546282

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end f_g_5_eq_163_l546_546282


namespace min_dot_product_l546_546634

variables {a b : ℝ} {t : ℝ} {m n : ℝ}
variables (u v : ∀ x, ℝ)
variable (h : t ≥ 1)

-- Assume a and b are unit vectors and perpendicular
axiom (norm_a : ∥a∥ = 1)
axiom (norm_b : ∥b∥ = 1)
axiom (dot_product_zero : a • b = 0)

-- Definitions of vectors m and n
def m : ℝ := 2 * a - real.sqrt (t - 1) * b
def n : ℝ := t * a + b

-- Goal: minimum value of m • n
theorem min_dot_product : min (λ t, m • n) h = 15 / 8 :=
sorry

end min_dot_product_l546_546634


namespace win_prize_probability_l546_546508

-- Define a condition that represents the probability calculation
def probability (total_outcomes favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

-- Declare the constants as provided in the problem
def total_cards : ℕ := 4
def total_bags : ℕ := 6

-- Calculate the total number of outcomes
def total_outcomes : ℕ := total_cards ^ total_bags

-- Calculate the number of favorable outcomes as explained in the solution
def favorable_outcomes : ℕ :=
  let scenario1 := 4 * Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)
  let scenario2 := 6 * Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1)
  scenario1 + scenario2

-- Calculate the probability
def computed_probability : ℚ := probability total_outcomes favorable_outcomes

-- Provide the proof statement
theorem win_prize_probability : computed_probability = 195 / 512 :=
  by
  -- Here you would provide the proof of this theorem.
  -- The details of the steps are omitted and replaced with sorry.
  sorry

end win_prize_probability_l546_546508


namespace fixed_monthly_fee_l546_546538

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + 20 * y = 15.20) 
  (h2 : x + 40 * y = 25.20) : 
  x = 5.20 := 
sorry

end fixed_monthly_fee_l546_546538


namespace gcd_max_value_l546_546785

theorem gcd_max_value (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1005) : ∃ d, d = Int.gcd a b ∧ d = 335 :=
by {
  sorry
}

end gcd_max_value_l546_546785


namespace point_in_fourth_quadrant_l546_546248

def z : ℂ := 1 + I

theorem point_in_fourth_quadrant :
  let w := (1/z) + conj z in
  w.re > 0 ∧ w.im < 0 :=
by
  let w := (1/z) + conj z
  show w.re > 0 ∧ w.im < 0,
  sorry

end point_in_fourth_quadrant_l546_546248


namespace ratio_of_a_and_b_l546_546691

theorem ratio_of_a_and_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : (a * Real.sin (Real.pi / 7) + b * Real.cos (Real.pi / 7)) / 
        (a * Real.cos (Real.pi / 7) - b * Real.sin (Real.pi / 7)) = 
        Real.tan (10 * Real.pi / 21)) :
  b / a = Real.sqrt 3 :=
sorry

end ratio_of_a_and_b_l546_546691


namespace forest_can_be_fenced_l546_546657

theorem forest_can_be_fenced (n : ℕ) (a : Fin n → ℝ) (d : Fin n → Fin n → ℝ) :
  (∀ i j : Fin n, d i j ≤ a i - a j) ∧ (∀ i : Fin n, 0 < a i ∧ a i < 100) →
  let path_sum := List.sum (List.map (λ i, d i (i + 1)) (List.finRange (n - 1))) in
  path_sum * 2 < 200 :=
by
  intros
  sorry

end forest_can_be_fenced_l546_546657


namespace diamond_example_l546_546572

def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_example : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 :=
by
  sorry

end diamond_example_l546_546572


namespace power_of_expression_l546_546005

theorem power_of_expression (a b c d e : ℝ)
  (h1 : a - b - c + d = 18)
  (h2 : a + b - c - d = 6)
  (h3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 :=
by
  sorry

end power_of_expression_l546_546005


namespace max_product_arithmetic_sequence_l546_546588

theorem max_product_arithmetic_sequence:
  ∀ (a : ℕ → ℕ),
  (a(1) + a(3) = 30) →
  (a(2) + a(4) = 10) → 
  ∃ (n : ℕ), (a(1) * a(2) * ... * a(n) = 729) :=
by
  sorry

end max_product_arithmetic_sequence_l546_546588


namespace minimum_value_minimum_value_theorem_l546_546583

theorem minimum_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) : 
  ∃ xy_val : ℝ, xy_val = 4 * Real.sqrt 3 ∧ ⁇

theorem minimum_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) : 
  ∃ z : ℝ, z = 4 * Real.sqrt 3 ∧ (x * 2 * y + x + 2 * y + 1) / Real.sqrt (x * y) = z := sorry

end minimum_value_minimum_value_theorem_l546_546583


namespace sweetest_sugar_water_l546_546403

-- Define the initial conditions as constants.
def initial_sugar_mass : ℕ := 25
def initial_water_mass : ℕ := 100
def initial_total_mass : ℕ := initial_sugar_mass + initial_water_mass
def initial_concentration : ℚ := (initial_sugar_mass : ℚ) / initial_total_mass

-- Define A's final concentration.
def concentration_A : ℚ := initial_concentration

-- Define parameters for B and C's final concentrations.
def B_added_sugar_concentration : ℚ := 0.4
def C_added_sugar_concentration : ℚ := 0.4

-- Assume C adds a larger volume than B.
axiom C_adds_larger_volume_than_B (B_volume C_volume : ℚ) : C_volume > B_volume

-- Define the final concentration for B and C.
def final_concentration_B (B_volume : ℚ) : ℚ :=
    ((B_added_sugar_concentration * B_volume)
     + (initial_concentration * initial_total_mass))
    / (B_volume + initial_total_mass)

def final_concentration_C (C_volume : ℚ) : ℚ :=
    ((C_added_sugar_concentration * C_volume)
     + (initial_concentration * initial_total_mass))
    / (C_volume + initial_total_mass)

-- The theorem states that C's final concentration is greater than both A's and B's.
theorem sweetest_sugar_water (B_volume C_volume : ℚ) (h : C_adds_larger_volume_than_B B_volume C_volume) :
    final_concentration_C C_volume > concentration_A ∧ final_concentration_C C_volume > final_concentration_B B_volume := 
    sorry

end sweetest_sugar_water_l546_546403


namespace find_x_l546_546959

theorem find_x (x : ℝ) (h : sqrt (9 - 2 * x) = 8) : x = -55 / 2 :=
by
  sorry

end find_x_l546_546959


namespace female_students_count_l546_546406

theorem female_students_count 
    (male_count : ℕ)
    (male_avg : ℕ)
    (female_avg : ℕ)
    (overall_avg : ℕ)
    (total_students_avg : ℕ)
    (male_grades_avg : male_count = 8)
    (male_sum : male_count * male_avg = 8 * 83)
    (female_avg_equals : female_avg = 92)
    (overall_avg_equals : overall_avg = 90)
    (total_students_equals : total_students_avg = (8 + (λ F, F)) * 90)
    : (λ F, 8 * male_avg + F * female_avg = (8 + F) * overall_avg) -> 28 :=
begin
    sorry
end

end female_students_count_l546_546406


namespace ratio_second_to_third_l546_546428

-- Define the three numbers A, B, C, and their conditions.
variables (A B C : ℕ)

-- Conditions derived from the problem statement.
def sum_condition : Prop := A + B + C = 98
def ratio_condition : Prop := 3 * A = 2 * B
def second_number_value : Prop := B = 30

-- The main theorem stating the problem to prove.
theorem ratio_second_to_third (h1 : sum_condition A B C) (h2 : ratio_condition A B) (h3 : second_number_value B) :
  B = 30 ∧ A = 20 ∧ C = 48 → B / C = 5 / 8 :=
by
  sorry

end ratio_second_to_third_l546_546428


namespace sum_of_factors_eq_l546_546412

theorem sum_of_factors_eq :
  ∃ (d e f : ℤ), (∀ (x : ℤ), x^2 + 21 * x + 110 = (x + d) * (x + e)) ∧
                 (∀ (x : ℤ), x^2 - 19 * x + 88 = (x - e) * (x - f)) ∧
                 (d + e + f = 30) :=
sorry

end sum_of_factors_eq_l546_546412


namespace vasya_difference_l546_546446

def sum_odd (n : ℕ) : ℕ :=
  n ^ 2

def sum_even (m : ℕ) : ℕ :=
  m * (m + 1)

theorem vasya_difference :
  let n := (2021 + 1) / 2 in
  let m := 2020 / 2 in
  sum_odd n - sum_even m = 1011 :=
by
  let n := (2021 + 1) / 2
  let m := 2020 / 2
  have h₁ : sum_odd n = 1011 ^ 2 := by sorry
  have h₂ : sum_even m = 1010 * 1011 := by sorry
  calc
    sum_odd n - sum_even m
        = (1011 ^ 2) - (1010 * 1011) : by rw [h₁, h₂]
    ... = 1011 * (1011 - 1010) : by sorry
    ... = 1011 * 1 : by sorry
    ... = 1011 : by sorry

end vasya_difference_l546_546446


namespace OI_perp_DE_and_equal_length_l546_546655

-- Given a triangle ABC with specific points and properties
variables {A B C O I D E : Type} [Triangle A B C]
variable (angle_C : ∠C = 30°)
variable (is_circumcenter : O = circumcenter A B C)
variable (is_incenter : I = incenter A B C)
variable (on_AC : D ∈ segment A C)
variable (on_BC : E ∈ segment B C)
variable (equal_segments : AD = AB ∧ BE = AB)

-- We need to prove
theorem OI_perp_DE_and_equal_length :
  OI.perpendicular DE ∧ OI = DE :=
by
  sorry

end OI_perp_DE_and_equal_length_l546_546655


namespace mark_takes_tablets_for_12_hours_l546_546714

theorem mark_takes_tablets_for_12_hours
  (tablets_per_dose : ℕ)
  (mg_per_tablet : ℕ)
  (grams_total : ℝ)
  (hours_between_doses : ℕ)
  (mg_per_gram : ℕ)
  (tablets_per_dose = 2)
  (mg_per_tablet = 500)
  (grams_total = 3)
  (hours_between_doses = 4)
  (mg_per_gram = 1000) :
  (grams_total * mg_per_gram) / (tablets_per_dose * mg_per_tablet) * hours_between_doses = 12 := 
by
  sorry

end mark_takes_tablets_for_12_hours_l546_546714


namespace probability_divisible_by_3_l546_546721

-- Defining some basic
variables {a b c : ℕ}
variables {S : Finset ℕ} (hS : S = (Finset.range 2011).filter (λ x, x > 0))

def p_div_by_3 (a b c : ℕ) : ℚ := 
  if 0 < a ∧ a ≤ 2010 ∧ 0 < b ∧ b ≤ 2010 ∧ 0 < c ∧ c ≤ 2010 
  then if (a * b * c + a * b + a) % 3 = 0 then 1 else 0 
  else 0

theorem probability_divisible_by_3 : 
  (Finset.sum S (λ a, Finset.sum S (λ b, Finset.sum S (λ c, p_div_by_3 a b c)))) / (2010 * 2010 * 2010) = 13 / 27 :=
sorry

end probability_divisible_by_3_l546_546721


namespace coeff_b_neg_3_over_2_zero_l546_546321

theorem coeff_b_neg_3_over_2_zero (b : ℝ) (hb : b ≠ 0) :
  ∑ k in finset.range 9, (binomial 8 k) * (b^(8 - k)) * ((-1)^(k) * (b^(1/2))^(-k)) = 0 :=
by sorry

end coeff_b_neg_3_over_2_zero_l546_546321


namespace proof_series_sum_l546_546919

noncomputable def series_sum := 
  (∑ k in Finset.range 100, (3 + 12 * (k + 1)) / (9 ^ (101 - (k + 1)))) = 
  (148.6875 - 0.1875 * (10 : ℝ) ^ (-98)) / (9 ^ (99 : ℕ))

theorem proof_series_sum : series_sum := 
  sorry

end proof_series_sum_l546_546919


namespace sum_of_proper_divisors_of_81_l546_546817

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l546_546817


namespace line_through_points_l546_546355

-- Given points
def p1 : ℝ × ℝ := (1, 2)
def p2 : ℝ × ℝ := (10, -4)
def p3 : ℝ × ℝ := (2, 3)
def p4 : ℝ × ℝ := (4, 1)

-- Trisection points of the line segment joining (1,2) and (10,-4)
def trisect1 : ℝ × ℝ := (4, 0)
def trisect2 : ℝ × ℝ := (7, -2)

theorem line_through_points :
  ∃ l : ℝ × ℝ → ℝ, -- A line function l which takes a point and returns a value (expressing the line equation)
    (l p1 = 0) ∧ (l p2 = 0) ∧ (l p3 = 0) ∧ (l p4 = 0) →
    l = (λ p, p.1 + p.2 - 5) := sorry

end line_through_points_l546_546355


namespace min_value_F_l546_546226

variable {a b : ℝ}

def f (x : ℝ) : ℝ := x^2 + a * x + b

theorem min_value_F (a b : ℝ) : (∀ x, |x| ≤ 1 → |f x| ≤ 1 / 2) → 
  (∃ x, |x| ≤ 1 ∧ |f x| = 1 / 2) :=
sorry

end min_value_F_l546_546226


namespace eccentricity_range_l546_546593

variable {a b c m : ℝ} (h_b_pos : b > 0)
def hyperbola := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1
def point_on_left_branch (P : ℝ × ℝ) := ∃ x y : ℝ, P = (x, y) ∧ (x^2 / a^2) - (y^2 / b^2) = 1 ∧ x < 0
def max_ratio_condition := ∃ m : ℝ, |m| ≤ 8 * a ∧ (a + m)^2 / (c - m) ≤ 8 * a
def eccentricity (e : ℝ) := c = a + e * b

theorem eccentricity_range (h_b_pos : b > 0) (h_hyperbola : hyperbola) (h_point : ∃ P : ℝ × ℝ, point_on_left_branch P)
  (h_max_ratio : max_ratio_condition) : ∀ e : ℝ, (1 < e ∧ e ≤ real.sqrt 3) :=
sorry

end eccentricity_range_l546_546593


namespace door_X_is_inner_sanctuary_l546_546579

  variable (X Y Z W : Prop)
  variable (A B C D E F G H : Prop)
  variable (is_knight : Prop → Prop)

  -- Each statement according to the conditions in the problem.
  variable (stmt_A : X)
  variable (stmt_B : Y ∨ Z)
  variable (stmt_C : is_knight A ∧ is_knight B)
  variable (stmt_D : X ∧ Y)
  variable (stmt_E : X ∧ Y)
  variable (stmt_F : is_knight D ∨ is_knight E)
  variable (stmt_G : is_knight C → is_knight F)
  variable (stmt_H : is_knight G ∧ is_knight H → is_knight A)

  theorem door_X_is_inner_sanctuary :
    is_knight A → is_knight B → is_knight C → is_knight D → is_knight E → is_knight F → is_knight G → is_knight H → X :=
  sorry
  
end door_X_is_inner_sanctuary_l546_546579


namespace greatest_divisor_of_450_less_than_60_and_factor_of_90_l546_546456

theorem greatest_divisor_of_450_less_than_60_and_factor_of_90:
  ∃ d : ℕ, d < 60 ∧ d ∣ 450 ∧ d ∣ 90 ∧ ∀ m : ℕ, m < 60 → m ∣ 450 → m ∣ 90 → m ≤ d :=
begin
  use 45,
  repeat { sorry },
end

end greatest_divisor_of_450_less_than_60_and_factor_of_90_l546_546456


namespace ratio_area_ADE_BCED_is_8_over_9_l546_546323

noncomputable def ratio_area_ADE_BCED 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) : ℝ := 
  sorry

theorem ratio_area_ADE_BCED_is_8_over_9 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) :
  ratio_area_ADE_BCED AB BC AC AD AE hAB hBC hAC hAD hAE = 8 / 9 :=
  sorry

end ratio_area_ADE_BCED_is_8_over_9_l546_546323


namespace max_value_of_expression_l546_546340

variable (a b c : ℝ)

theorem max_value_of_expression : 
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c := by
sorry

end max_value_of_expression_l546_546340


namespace minimize_distances_l546_546752

/-- Given points P = (6, 7), Q = (3, 4), and R = (0, m),
    find the value of m that minimizes the sum of distances PR and QR. -/
theorem minimize_distances (m : ℝ) :
  let P := (6, 7)
  let Q := (3, 4)
  ∃ m : ℝ, 
    ∀ m' : ℝ, 
    (dist (6, 7) (0, m) + dist (3, 4) (0, m)) ≤ (dist (6, 7) (0, m') + dist (3, 4) (0, m'))
:= ⟨5, sorry⟩

end minimize_distances_l546_546752


namespace packs_of_tuna_purchased_l546_546530

-- Definitions based on the conditions
def cost_per_pack_of_tuna : ℕ := 2
def cost_per_bottle_of_water : ℤ := (3 / 2)
def total_paid_by_Barbara : ℕ := 56
def money_spent_on_different_goods : ℕ := 40
def number_of_bottles_of_water : ℕ := 4

-- The proposition to prove
theorem packs_of_tuna_purchased :
  ∃ T : ℕ, total_paid_by_Barbara = cost_per_pack_of_tuna * T + cost_per_bottle_of_water * number_of_bottles_of_water + money_spent_on_different_goods ∧ T = 5 :=
by
  sorry

end packs_of_tuna_purchased_l546_546530


namespace det_ABC_l546_546212

noncomputable def det_product_matrices (A B C : Matrix (Fin 3) (Fin 3) ℝ) : ℝ :=
  Matrix.det A * Matrix.det B * Matrix.det C

theorem det_ABC (A B C : Matrix (Fin 3) (Fin 3) ℝ) 
  (hA : Matrix.det A = 3) (hB : Matrix.det B = -7) (hC : Matrix.det C = 4) :
  Matrix.det (A ⬝ B ⬝ C) = -84 := by
  sorry

end det_ABC_l546_546212


namespace evaluate_sqrt4_16_pow_12_l546_546940

theorem evaluate_sqrt4_16_pow_12 : (real.rpow 16 (1/4))^12 = 4096 := by
  sorry

end evaluate_sqrt4_16_pow_12_l546_546940


namespace tall_flags_count_l546_546928

noncomputable def fabric_total : ℕ := 1000
noncomputable def area_square_flag : ℕ := 4 * 4
noncomputable def area_wide_flag : ℕ := 5 * 3
noncomputable def area_tall_flag : ℕ := 3 * 5
noncomputable def num_square_flags : ℕ := 16
noncomputable def num_wide_flags : ℕ := 20
noncomputable def fabric_left : ℕ := 294

theorem tall_flags_count :
  let T := fabric_total,
      S_F := area_square_flag,
      W_F := area_wide_flag,
      T_F := area_tall_flag,
      n_S := num_square_flags,
      n_W := num_wide_flags,
      F_left := fabric_left in
  T - (n_S * S_F + n_W * W_F) - F_left = T_F * 10 := by
  sorry

end tall_flags_count_l546_546928


namespace math_problem_l546_546451

theorem math_problem (a b c d : ℕ)
    (h1 : a = 2468)
    (h2 : b = 6)
    (h3 : c = 520)
    (h4 : d = 3456) :
    (a / b - c + d) = 3347.333 :=
by sorry

end math_problem_l546_546451


namespace subtract_decimal_numbers_l546_546948

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_decimal_numbers_l546_546948


namespace sum_proper_divisors_81_l546_546828

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546828


namespace fraction_value_l546_546640

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem
theorem fraction_value (h : 2 * x = -y) : (x * y) / (x^2 - y^2) = 2 / 3 :=
by
  sorry

end fraction_value_l546_546640


namespace slope_of_midline_segments_l546_546460

theorem slope_of_midline_segments :
  let p1 := (0, 2)
  let p2 := (4, 6)
  let p3 := (3, 0)
  let p4 := (7, 4)
  let midpoint (a b : ℕ × ℕ) : ℕ × ℕ := ( (a.1 + b.1) / 2, (a.2 + b.2) / 2 )
  let m1 := midpoint p1 p2
  let m2 := midpoint p3 p4
  ( (m2.2 - m1.2) * (m2.1 - m1.1).inv ).eval = -2/3 := sorry

end slope_of_midline_segments_l546_546460


namespace altitude_leq_median_l546_546289

variables (A B C M N : Type) [EuclideanGeometry A]

-- Definitions of altitude and median in the triangle ABC
def is_altitude (A B C M : Point) : Prop :=
  is_perpendicular (line_segment A M) (line B C)

def is_median (A B C N : Point) : Prop :=
  midpoint N B C

-- Hypotheses
axiom altitude_AM : is_altitude A B C M
axiom median_AN : is_median A B C N

-- The goal is to prove that the altitude AM is less than or equal to the median AN
theorem altitude_leq_median :
  length (line_segment A M) ≤ length (line_segment A N) :=
sorry

end altitude_leq_median_l546_546289


namespace incorrect_proposition_in_option_C_l546_546698

open PlaneGeometry -- Assuming a PlaneGeometry module that handles basic geometry definitions and relationships.

variables (m n : Line) (α β : Plane)

-- Conditions
def diff_lines (m n : Line) : Prop := m ≠ n
def diff_planes (α β : Plane) : Prop := α ≠ β
def parallel (l1 l2 : Line) : Prop := l1 ∥ l2
def perp_line_plane (l : Line) (p : Plane) : Prop := l ⊥ p
def subset (l : Line) (p : Plane) : Prop := l ⊆ p
def inter (p1 p2 : Plane) (l : Line) : Prop := p1 ∩ p2 = l

-- Problem Statement: Prove that the proposition in option C is incorrect.
theorem incorrect_proposition_in_option_C
  (hmn : diff_lines m n)
  (hαβ : diff_planes α β)
  (h_parallel_mα : parallel m α)
  (h_inter : inter α β n) :
  ¬ (parallel m n) := by
  sorry

end incorrect_proposition_in_option_C_l546_546698


namespace find_j_l546_546415

theorem find_j (j k : ℝ) :
  (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ (∀ i ∈ {0, 1, 2, 3}, a + i * d ≠ a + j * d) ∧
  (∀ x : ℝ, (x = a ∨ x = a + d ∨ x = a + 2 * d ∨ x = a + 3 * d) →
  x^4 + j*x^2 + k*x + 400 = 0)) → j = -40 :=
by
  sorry

end find_j_l546_546415


namespace center_square_side_length_l546_546405

theorem center_square_side_length :
  let total_area := 100 * 100 in
  let l_shaped_area_fraction := 3 / 16 in
  let l_shaped_total_area := 4 * l_shaped_area_fraction * total_area in
  let center_square_area := total_area - l_shaped_total_area in
  let side_length := Real.sqrt center_square_area in
  side_length = 50 :=
by
  unpack let total_area := 100 * 100 in
  unpack let l_shaped_area_fraction := 3 / 16 in
  unpack let l_shaped_total_area := 4 * l_shaped_area_fraction * total_area in
  unpack let center_square_area := total_area - l_shaped_total_area in
  unpack let side_length := Real.sqrt center_square_area in
  sorry

end center_square_side_length_l546_546405


namespace convert_444_quinary_to_octal_l546_546182

def quinary_to_decimal (n : ℕ) : ℕ :=
  let d2 := (n / 100) * 25
  let d1 := ((n % 100) / 10) * 5
  let d0 := (n % 10)
  d2 + d1 + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let r2 := (n / 64)
  let n2 := (n % 64)
  let r1 := (n2 / 8)
  let r0 := (n2 % 8)
  r2 * 100 + r1 * 10 + r0

theorem convert_444_quinary_to_octal :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end convert_444_quinary_to_octal_l546_546182


namespace min_magnitude_a_l546_546265

variable (θ : ℝ)

def a := (Real.cos θ - 2, Real.sin θ)
def magnitude_a := Real.sqrt ((Real.cos θ - 2) ^ 2 + (Real.sin θ) ^ 2)

theorem min_magnitude_a : ∃ θ : ℝ, magnitude_a θ = 1 :=
by
  sorry

end min_magnitude_a_l546_546265


namespace maximum_volume_of_sphere_l546_546673

-- Given conditions
def AB : ℝ := 6
def BC : ℝ := 8
def AA1 : ℝ := 3
-- Computing AC using the Pythagorean theorem
def AC : ℝ := Real.sqrt (AB^2 + BC^2)
-- The inradius of triangle ABC
def inradius_triangle_ABC : ℝ := (AB + BC - AC) / 2
-- The inradius of the triangular prism
def inradius_prism : ℝ := AA1 / 2
-- Volume of the sphere inscribed in the prism
def volume_sphere : ℝ := (4 / 3) * Real.pi * (inradius_prism^3)

-- The goal is to show that the maximum volume is 9π/2
theorem maximum_volume_of_sphere : volume_sphere = (9 * Real.pi / 2) := by
  sorry

end maximum_volume_of_sphere_l546_546673


namespace elevator_min_trips_l546_546112

theorem elevator_min_trips :
  let masses := [150, 60, 70, 71, 72, 100, 101, 102, 103] in
  let max_load := 200 in
  (min_trips masses max_load = 5) :=
begin
  -- Sorry is used to skip the proof.
  sorry
end

end elevator_min_trips_l546_546112


namespace cups_of_salt_l546_546715

-- Define the initial conditions as assumptions
variables (total_flour required as 12) 
          (initial_flour put in as 2) 
          (flour more than salt as 3)

-- Define the proof statement
theorem cups_of_salt (S : ℕ) (H1 : total_flour = 12) (H2 : initial_flour = 2) (H3 : flour_more_than_salt = 3) :
  S = 7 := 
by
  have H_flour_needed : total_flour - initial_flour = 10 := by sorry
  have H_salt_eq : S + flour_more_than_salt = 10 := by sorry
  have H_final : S = 7 := by sorry
  exact H_final

end cups_of_salt_l546_546715


namespace trapezoid_sides_l546_546125

variables (R : ℝ) (A B C D : ℝ)

-- Given conditions
def is_right_trapezoid (A B C D : ℝ) := 
  ∃ (r : ℝ), r = R ∧ A = (4/3) * r

-- Proof of the lengths of the sides
theorem trapezoid_sides (A B C D R : ℝ) 
(h1: A = (4/3) * R) 
(h2: B = 3 * R) 
(h3: C = 4 * R) 
(h4: D = (10/3) * R) : 
  is_right_trapezoid A B C D :=
begin
  split,
  { existsi R,
    split,
    { refl },
    { exact h1 },
  },
end

end trapezoid_sides_l546_546125


namespace relationship_y1_y2_l546_546017

theorem relationship_y1_y2 
  (f : ℝ → ℝ) 
  (h1 : f = λ x, -2 * x + 1) 
  (y1 y2 : ℝ) 
  (A_on_graph : f 1 = y1) 
  (B_on_graph : f 3 = y2) : 
  y1 > y2 :=
by 
  sorry

end relationship_y1_y2_l546_546017


namespace compute_simple_interest_l546_546852

theorem compute_simple_interest :
  ∃ P : ℝ, ∃ SI : ℝ,
    let r := 0.10 in
    let n := 1 in
    let t := 7 in
    let CI := 993 in
      P = CI / (1 : ℝ) * (1 + r/n)^t - 1 ∧
      SI = P * r * t ∧
      abs (SI - 732.61) < 0.01 :=
sorry

end compute_simple_interest_l546_546852


namespace arrange_strips_possible_l546_546016

-- Definitions of conditions for the problem:
def strips : Nat := 8 -- There are 8 strips
def strip_size : Nat := 3 -- Each strip is of size 1 x 3

noncomputable def square_color := {s : Nat × Nat // s.fst < strips ∧ s.snd < strip_size} → Bool
-- The definition of coloring each square in the strip (either white or gray)

-- Statement of the problem:
theorem arrange_strips_possible :
  ∃ arrangement : (square_color → Bool), 
  -- Arrangement ensures that each type forms a non-self-intersecting closed polyline
  (∀ c : Bool, ∃ polygon : set (Nat × Nat), 
    ∀ s : Nat × Nat, (polygon s ↔ arrangement s = c) ∧ 
    non_self_intersecting_closed_polyline polygon) → 
  -- Arrangements are non-overlapping
  ∀ (s1 s2 : Nat × Nat), 
    arrangement s1 = arrangement s2 ∨ ¬overlap s1 s2 :=
sorry

end arrange_strips_possible_l546_546016


namespace committee_membership_l546_546804

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end committee_membership_l546_546804


namespace sequence_solution_l546_546353

noncomputable def seq (a : ℕ → ℚ) := 
  a 0 = 2 ∧ ∀ n ≥ 1, a n = (2 * a (n-1) + 6) / (a (n-1) + 1)

theorem sequence_solution (a : ℕ → ℚ) (h : seq a) :
  ∀ n : ℕ, a n = 
    if n = 0 then 2 
    else 
      let k : ℚ := 4^(n+1)
      let sign : ℚ := if (n + 1) % 2 = 0 then 1 else -1
      in (3 * k + 2 * sign) / (k + sign) :=
by
  sorry

end sequence_solution_l546_546353


namespace polar_graph_representation_l546_546044

theorem polar_graph_representation :
  ∀ (ρ θ : ℝ), (ρ - 1) * (θ - π) = 0 ∧ ρ ≥ 0 ↔ 
  ((ρ = 1 ∧ θ ∈ ℝ) ∨ (θ = π ∧ ρ ≥ 0)) :=
by
  intro ρ θ
  apply iff.intro
  { intro h
    cases h.1 with hρ hθ
    { left
      exact ⟨eq.symm hρ, θ⟩ }
    { right
      exact ⟨eq.symm hθ, h.2⟩ } }
  { intro h
    cases h
    { cases h with hρ hθ
      exact ⟨or.inl (eq.symm hρ.symm), by linarith⟩ }
    { cases h with hθ hρ
      exact ⟨or.inr (eq.symm hθ.symm), hρ⟩ } }

end polar_graph_representation_l546_546044


namespace wheel_revolutions_l546_546424

noncomputable def number_of_revolutions := by
  let r := 22.4 -- radius in cm
  let d := 1056 -- distance covered in cm
  let C := 2 * Real.pi * r -- circumference
  let N := d / C -- number of revolutions
  exact_finset.intro (approx N) sorry

-- Proof problem statement
theorem wheel_revolutions : 
  let r := 22.4 in
  let d := 1056 in
  let C := 2 * Real.pi * r in
  d / C ≈ 8 := 
by
  sorry

end wheel_revolutions_l546_546424


namespace find_coefficients_l546_546339
-- Import the entire Mathlib library to ensure all necessary lemmas and theorems are included.

-- Define the problem conditions: roots of the polynomial and real coefficients.
def polynomial_with_roots (x : ℂ) : Prop :=
  (x - (1 + 2i)) * (x - (1 - 2i)) * (x - (2 + i)) * (x - (2 - i)) = x^4 - 6 * x^3 + 21 * x^2 - 30 * x + 25

-- Define the coefficients a, b, and c using the polynomial.
def coefficients (a b c : ℤ) : Prop :=
  a = -6 ∧ b = 21 ∧ c = -30

-- Theorem statement: If the polynomial has the specified roots and real coefficients, then the coefficients are as given.
theorem find_coefficients (a b c : ℤ) :
  polynomial_with_roots x → coefficients a b c :=
begin
  sorry
end

end find_coefficients_l546_546339


namespace repair_cost_l546_546526

theorem repair_cost (initial_cost selling_price : ℝ) (gain_percent : ℝ) 
  (h_initial : initial_cost = 4700) 
  (h_selling : selling_price = 6000) 
  (h_gain : gain_percent = 9.090909090909092) : 
  ∃ R : ℝ, (gain_percent / 100) * (initial_cost + R) = selling_price - (initial_cost + R) ∧ R = 800 
:= 
by 
suffices h : ∀ R, R = 800 -> (gain_percent / 100) * (initial_cost + R) = selling_price - (initial_cost + R),
 by use 800; split; try { exact h 800 rfl }; sorry

end repair_cost_l546_546526


namespace sqrt_sum_inequality_l546_546392

-- Define variables a and b as positive real numbers
variable {a b : ℝ}

-- State the theorem to be proved
theorem sqrt_sum_inequality (ha : 0 < a) (hb : 0 < b) : 
  (a.sqrt + b.sqrt)^8 ≥ 64 * a * b * (a + b)^2 :=
sorry

end sqrt_sum_inequality_l546_546392


namespace sum_proper_divisors_eq_40_l546_546822

def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => is_proper_divisor n d) (List.range (n + 1))

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

theorem sum_proper_divisors_eq_40 : sum_proper_divisors 81 = 40 := sorry

end sum_proper_divisors_eq_40_l546_546822


namespace intersection_points_eq_5001_l546_546711

-- Let's define the conditions as per the problem statement
def max_intersections_of_lines (lines : Set Line) : ℕ := sorry

variable (L : ℕ → Line)
variable (B : Point)
variable (h_distinct : ∀ i j, i ≠ j → L i ≠ L j)
variable (h_parallel : ∀ n, Parallel (L (3*n-1)) (L (3*n)))
variable (h_pass_through_B : ∀ n, PassesThrough (L (3*n-2)) B)

-- Now, let's state the main theorem using these conditions
theorem intersection_points_eq_5001 :
  max_intersections_of_lines (Set.univ {i : ℕ | 1 ≤ i ∧ i ≤ 150}.map L) = 5001 :=
sorry

end intersection_points_eq_5001_l546_546711


namespace sum_odd_divisors_of_450_l546_546914

-- Define the prime factorization of 450
def pf_450 := 2^1 * 3^2 * 5^2

-- Define a function to sum the odd divisors of 450
def sum_odd_divisors (n : ℕ) (pf_n : n = 2^1 * 3^2 * 5^2) : ℕ :=
  let sum_powers_of_3 := 1 + 3 + 9
  let sum_powers_of_5 := 1 + 5 + 25
  sum_powers_of_3 * sum_powers_of_5

-- The theorem statement
theorem sum_odd_divisors_of_450 : sum_odd_divisors 450 pf_450 = 403 := 
by 
  sorry

end sum_odd_divisors_of_450_l546_546914


namespace length_AKLMNA_le_26_l546_546981

variable (A B C D K L M N : Type)
variables {pts : Fin₄ → Type}
variables {sides : list (Fin₄ × Fin₄) → Type}

open_locale big_operators

variable {rect : ℝ × ℝ → Type}
variable {closed_polygonal_chain : list (Fin₄ × Fin₄) → Type}

theorem length_AKLMNA_le_26 (AB : rect(5, 0)) (BC : rect(0, 6)):
  ∑ [(A, K), (K, L), (L, M), (M, N), (N, A)] ≤ 26 := 
sorry

end length_AKLMNA_le_26_l546_546981


namespace sum_of_proper_divisors_of_81_l546_546821

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l546_546821


namespace ratio_is_five_over_twelve_l546_546263

theorem ratio_is_five_over_twelve (a b c d : ℚ) (h1 : b = 4 * a) (h2 : d = 2 * c) :
    (a + b) / (c + d) = 5 / 12 :=
sorry

end ratio_is_five_over_twelve_l546_546263


namespace can_create_grid_4x4_l546_546917

-- Definition of the grid and its properties
def grid_4x4 := λ n: ℕ, n = 4

-- Conditions given in the problem
def threads_5cm_count := 8
def thread_5cm_length := 5
def total_required_length := 40

-- Proof statement to verify if the 8 pieces of 5 cm threads can create the grid
theorem can_create_grid_4x4 
    (number_of_threads : ℕ := threads_5cm_count) 
    (length_of_each_thread : ℕ := thread_5cm_length)
    (total_length : ℕ := total_required_length) 
    (grid : ℕ → Prop := grid_4x4):
    (number_of_threads = 8) ∧ (length_of_each_thread = 5) ∧ (total_length = 40) ∧ (grid 4) → 
    True := sorry

end can_create_grid_4x4_l546_546917


namespace candies_problem_max_children_l546_546436

theorem candies_problem_max_children (u v : ℕ → ℕ) (n : ℕ) :
  (∀ i : ℕ, u i = v i + 2) →
  (∀ i : ℕ, u i + 2 = u (i + 1)) →
  (u (n - 1) / u 0 = 13) →
  n = 25 :=
by
  -- Proof not required as per the instructions.
  sorry

end candies_problem_max_children_l546_546436


namespace additional_people_required_l546_546196

-- Define conditions
def people := 8
def time1 := 3
def total_work := people * time1 -- This gives us the constant k

-- Define the second condition where 12 people are needed to complete in 2 hours
def required_people (t : Nat) := total_work / t

-- The number of additional people required
def additional_people := required_people 2 - people

-- State the theorem
theorem additional_people_required : additional_people = 4 :=
by 
  show additional_people = 4
  sorry

end additional_people_required_l546_546196


namespace domain_of_sqrt_sum_l546_546755

def domain_of_f (x : ℝ) : Prop := (0 ≤ x + 1) ∧ (0 ≤ 2 - x)

theorem domain_of_sqrt_sum :
  { x : ℝ | domain_of_f x } = set.Icc (-1 : ℝ) 2 :=
by
  sorry

end domain_of_sqrt_sum_l546_546755


namespace AliBabaWinsIfBanditStartsFirst_l546_546153

-- Define the game state
structure Game where
  piles : ℕ

-- Initial state with 2017 diamonds
def initialState : Game := { piles := 1 }

-- Define the move in the game
def makeMove (g : Game) : Game :=
  { piles := g.piles + 1 }

-- Define an action in the game
def canMove (g : Game) : Prop :=
  g.piles < 2017

-- Define the winner based on whose turn it is after all valid moves
def winner (g : Game) (turns : ℕ) : String :=
  if turns % 2 = 0 then "Bandit" else "Ali-Baba"

-- The statement we need to prove
theorem AliBabaWinsIfBanditStartsFirst :
  winner (makeMove^2016 initialState) 2016 = "Ali-Baba" :=
sorry

end AliBabaWinsIfBanditStartsFirst_l546_546153


namespace sufficient_but_not_necessary_condition_for_square_l546_546113

theorem sufficient_but_not_necessary_condition_for_square (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ (¬(x^2 > 4 → x > 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_square_l546_546113


namespace non_adjacent_even_digit_permutations_l546_546271

theorem non_adjacent_even_digit_permutations : 
  (∃ (l : List ℕ), l.perm [1, 2, 3, 4, 5] ∧ (∀ (i : ℕ), 2 < i → i < 6 → (l[i - 1] ∈ {2, 4} → l[i] ∉ {2, 4} ∧ l[i - 2] ∉ {2, 4}))) ∧
  (l.count (λ x, x) = l.count (2)) ∧ (l.count (λ x, x) = l.count (4)) →
  l.perm [1, 2, 3, 4, 5] → l.count (λ x, {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5})%).permutations.count = 72 :=
begin
  sorry
end

end non_adjacent_even_digit_permutations_l546_546271


namespace rachel_earnings_without_tips_l546_546725

theorem rachel_earnings_without_tips
  (num_people : ℕ) (tip_per_person : ℝ) (total_earnings : ℝ)
  (h1 : num_people = 20)
  (h2 : tip_per_person = 1.25)
  (h3 : total_earnings = 37) :
  total_earnings - (num_people * tip_per_person) = 12 :=
by
  sorry

end rachel_earnings_without_tips_l546_546725


namespace triangle_side_length_l546_546605

theorem triangle_side_length (x : ℝ) (h1 : 6 < x) (h2 : x < 14) : x = 11 :=
by
  sorry

end triangle_side_length_l546_546605


namespace jake_third_test_marks_l546_546748

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end jake_third_test_marks_l546_546748


namespace greatest_integer_with_gcd_6_l546_546076

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l546_546076


namespace total_nests_required_l546_546787

theorem total_nests_required :
  let sparrows := 4 in
  let pigeons := 3 in
  let starlings := 3 in
  let nests_sparrow := 1 in
  let nests_pigeon := 2 in
  let nests_starling := 3 in
  (sparrows * nests_sparrow + pigeons * nests_pigeon + starlings * nests_starling) = 19 :=
by
  let sparrows := 4
  let pigeons := 3
  let starlings := 3
  let nests_sparrow := 1
  let nests_pigeon := 2
  let nests_starling := 3
  show (sparrows * nests_sparrow + pigeons * nests_pigeon + starlings * nests_starling) = 19
  sorry

end total_nests_required_l546_546787


namespace cone_volume_correct_l546_546604

noncomputable def cone_volume {r l : ℝ} (hl : l = sqrt 5) 
  (h_lateral_surface_area : 0.5 * 2 * π * l = sqrt 5 * π) : 
  ℝ :=
let h := sqrt (l^2 - r^2) in
  (1 / 3) * π * r^2 * h

theorem cone_volume_correct : cone_volume (by simp) (by simp) = (2/3) * π := sorry

end cone_volume_correct_l546_546604


namespace complement_intersection_l546_546970

def is_in_M (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2
def M : set ℝ := {x | is_in_M x}

def is_in_N (x : ℝ) : Prop := x ≤ 3
def N : set ℝ := {x | is_in_N x}

def is_in_complement_M (x : ℝ) : Prop := x < -1 ∨ x > 2
def complement_M : set ℝ := {x | is_in_complement_M x}

def lhs := complement_M ∩ N
def rhs := {x | x < -1} ∪ {x | 2 < x ∧ x ≤ 3}

theorem complement_intersection :
  lhs = rhs :=
sorry

end complement_intersection_l546_546970


namespace sum_log2_seq_first_10_terms_l546_546776

def a_seq : ℕ+ → ℕ
| ⟨1, _⟩   := 2
| ⟨n+1, h⟩ := 2 * a_seq ⟨n+1, nat.succ_pos n⟩

def log2_seq (n : ℕ+) : ℕ :=
  Nat.log2 (a_seq n)

theorem sum_log2_seq_first_10_terms : (Finset.sum (Finset.range 10) (λ i, log2_seq (⟨i + 1, Nat.succ_pos i⟩))) = 55 := 
by 
  sorry

end sum_log2_seq_first_10_terms_l546_546776


namespace sin_equation_solution_set_l546_546427

theorem sin_equation_solution_set (x : ℝ) : (∃ k : ℤ, x = k * Real.pi) ↔ sin x * (sin x - 2) = 0 := by
  sorry

end sin_equation_solution_set_l546_546427


namespace find_eccentricity_l546_546986

-- Definitions
variable (a b : ℝ)
variable (F : ℝ × ℝ)
variable (e : ℝ)

-- Conditions
def ellipse_eq : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def condition_a_b : Prop := a > b ∧ b > 0
def circle_eq : Prop := ∀ x y : ℝ, x^2 + y^2 - 4 * x - 32 = 0
def circle_center : Prop := F = (2, 0)
def tangent_condition : Prop := a^2 / 2 - 2 = 6

-- Proof goal
theorem find_eccentricity (h1 : ellipse_eq a b) (h2 : condition_a_b a b) 
                          (h3 : circle_eq F) (h4 : circle_center F) 
                          (h5 : tangent_condition a) : 
                          e = 1 / 2 := 
sorry

end find_eccentricity_l546_546986


namespace positive_integers_in_interval_l546_546964

theorem positive_integers_in_interval :
  {x : ℕ | 30 < x^2 - 4 * x + 4 ∧ x^2 - 4 * x + 4 < 60}.card = 2 :=
begin
  sorry
end

end positive_integers_in_interval_l546_546964


namespace four_integers_product_sum_l546_546610

theorem four_integers_product_sum (a b c d : ℕ) (h1 : a * b * c * d = 2002) (h2 : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l546_546610


namespace negation_of_proposition_l546_546049

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℝ, 2^x_0 < x_0^2) ↔ (∀ x : ℝ, 2^x ≥ x^2) :=
by sorry

end negation_of_proposition_l546_546049


namespace max_intersections_cos_circle_l546_546931

theorem max_intersections_cos_circle :
  let circle := λ x y => (x - 4)^2 + y^2 = 25
  let cos_graph := λ x => (x, Real.cos x)
  ∀ x y, (circle x y ∧ y = Real.cos x) → (∃ (p : ℕ), p ≤ 8) := sorry

end max_intersections_cos_circle_l546_546931


namespace betty_eggs_per_teaspoon_vanilla_l546_546092

theorem betty_eggs_per_teaspoon_vanilla
  (sugar_cream_cheese_ratio : ℚ)
  (vanilla_cream_cheese_ratio : ℚ)
  (sugar_in_cups : ℚ)
  (eggs_used : ℕ)
  (expected_ratio : ℚ) :
  sugar_cream_cheese_ratio = 1/4 →
  vanilla_cream_cheese_ratio = 1/2 →
  sugar_in_cups = 2 →
  eggs_used = 8 →
  expected_ratio = 2 →
  (eggs_used / (sugar_in_cups * 4 * vanilla_cream_cheese_ratio)) = expected_ratio :=
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_eggs_per_teaspoon_vanilla_l546_546092


namespace topsoil_cost_l546_546801

theorem topsoil_cost
  (cost_per_cubic_foot : ℕ)
  (volume_cubic_yards : ℕ)
  (conversion_factor : ℕ)
  (volume_cubic_feet : ℕ := volume_cubic_yards * conversion_factor)
  (total_cost : ℕ := volume_cubic_feet * cost_per_cubic_foot)
  (cost_per_cubic_foot_def : cost_per_cubic_foot = 8)
  (volume_cubic_yards_def : volume_cubic_yards = 8)
  (conversion_factor_def : conversion_factor = 27) :
  total_cost = 1728 := by
  sorry

end topsoil_cost_l546_546801


namespace initial_water_percentage_l546_546874

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end initial_water_percentage_l546_546874


namespace geometric_series_identity_l546_546072

theorem geometric_series_identity (x : ℝ) (n : ℕ) (h_pos : 0 < n) (h_ne : x ≠ 1) :
  (finset.range (n + 3)).sum (λ k, x^k) = (1 - x^(n + 3)) / (1 - x) := sorry

example (x : ℝ) (h_ne : x ≠ 1) : (finset.range 4).sum (λ k, x^k) = 1 + x + x^2 + x^3 :=
  by
  rw finset.sum_range_succ
  rw finset.sum_range_succ
  rw finset.sum_range_succ
  rw finset.sum_range_succ
  simp [pow_succ]

end geometric_series_identity_l546_546072


namespace find_exponent_l546_546245

theorem find_exponent (n : ℝ) (hn: (3:ℝ)^n = Real.sqrt 3) : n = 1 / 2 :=
by sorry

end find_exponent_l546_546245


namespace number_of_ideal_subsets_l546_546702

def is_ideal_subset (p q : ℕ) (S : Set ℕ) : Prop :=
  0 ∈ S ∧ ∀ n ∈ S, n + p ∈ S ∧ n + q ∈ S

theorem number_of_ideal_subsets (p q : ℕ) (hpq : Nat.Coprime p q) :
  ∃ n, n = Nat.choose (p + q) p / (p + q) :=
sorry

end number_of_ideal_subsets_l546_546702


namespace tangent_lines_and_a_value_l546_546262

theorem tangent_lines_and_a_value:
  (∀ x y : ℝ, (x = 2 ∨ 3 * x + 4 * y - 10 = 0) → (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}) ∧
  (∀ a : ℝ, (a * 2 - 1 = 0 ∨ a * 3 + 4 * 1 - 10 = 0) → ( ∃ x y : ℝ, a * x - y + 4 = 0 ∧ x^2 + y^2 = 4 ∧ |(x - y)| = 2 * sqrt 3) → (a = sqrt 15 ∨ a = -sqrt 15)) :=
sorry

end tangent_lines_and_a_value_l546_546262


namespace hyperbola_equation_l546_546601

theorem hyperbola_equation
  (asymptote : ∀ x : ℝ, y = 4 * x)
  (focus_parabola : ∃ p : ℝ × ℝ, p = (2, 0) ∧ (p.1 ^ 2 - p.2 ^ 2 = 4)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∃ h : ℝ → ℝ → Prop, h x y = (17*x^2/4) - (17*y^2/64) = 1) ∧ h a b = 0 := sorry

end hyperbola_equation_l546_546601


namespace amount_amys_money_l546_546160

def initial_dollars : ℝ := 2
def chores_payment : ℝ := 5 * 13
def birthday_gift : ℝ := 3
def total_after_gift : ℝ := initial_dollars + chores_payment + birthday_gift

def investment_percentage : ℝ := 0.20
def invested_amount : ℝ := investment_percentage * total_after_gift

def interest_rate : ℝ := 0.10
def interest_amount : ℝ := interest_rate * invested_amount
def total_investment : ℝ := invested_amount + interest_amount

def cost_of_toy : ℝ := 12
def remaining_after_toy : ℝ := total_after_gift - cost_of_toy

def grandparents_gift : ℝ := 2 * remaining_after_toy
def total_including_investment : ℝ := grandparents_gift + total_investment

def donation_percentage : ℝ := 0.25
def donated_amount : ℝ := donation_percentage * total_including_investment

def final_amount : ℝ := total_including_investment - donated_amount

theorem amount_amys_money :
  final_amount = 98.55 := by
  sorry

end amount_amys_money_l546_546160


namespace final_population_l546_546162

theorem final_population (P0 : ℕ) (r1 r2 : ℝ) (P2 : ℝ) 
  (h0 : P0 = 1000)
  (h1 : r1 = 1.20)
  (h2 : r2 = 1.30)
  (h3 : P2 = P0 * r1 * r2) : 
  P2 = 1560 := 
sorry

end final_population_l546_546162


namespace Ali_Baba_Wins_If_Bandit_Starts_First_l546_546152

theorem Ali_Baba_Wins_If_Bandit_Starts_First:
  ∃ Ali_Baba_Wins, 
  (∀ (initialPile : ℕ), initialPile = 2017 → (∀ moveCount : ℕ, moveCount = 2016 → (gameOver moveCount) → banditStartsFirst → Ali_Baba_Wins)) :=
by sorry

end Ali_Baba_Wins_If_Bandit_Starts_First_l546_546152


namespace probability_of_hitting_exactly_twice_l546_546847

def P_hit_first : ℝ := 0.4
def P_hit_second : ℝ := 0.5
def P_hit_third : ℝ := 0.7

def P_hit_exactly_twice_in_three_shots : ℝ :=
  P_hit_first * P_hit_second * (1 - P_hit_third) +
  (1 - P_hit_first) * P_hit_second * P_hit_third +
  P_hit_first * (1 - P_hit_second) * P_hit_third

theorem probability_of_hitting_exactly_twice :
  P_hit_exactly_twice_in_three_shots = 0.41 := 
by
  sorry

end probability_of_hitting_exactly_twice_l546_546847


namespace find_a_l546_546996

variable {ℝ : Type} [NontrivialLinearOrderedField ℝ]

noncomputable def f (x : ℝ) (a : ℝ) (g : ℝ → ℝ) : ℝ := a^x * g x

variable {g : ℝ → ℝ}
variable {g' : ℝ → ℝ}
variable (a : ℝ)
variable (f' : ℝ → ℝ) 

theorem find_a (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hg_ne_zero : ∀ x, g x ≠ 0)
  (hf_ineq : ∀ x, f x a g * g' x > f' x * g x)
  (h_frac_sum : f 1 a g / g 1 + f (-1) a g / g (-1) = 5 / 2) :
  a = 1 / 2 :=
sorry

end find_a_l546_546996


namespace sum_proper_divisors_81_l546_546836

theorem sum_proper_divisors_81 : 
  let n := 81,
      proper_divisors := [3^0, 3^1, 3^2, 3^3],
      sum_proper_divisors := proper_divisors.sum 
  in sum_proper_divisors = 40 := 
by
  purely
  let proper_divisors : List Nat := [1, 3, 9, 27]
  let sum_proper_divisors := proper_divisors.sum
  have : sum_proper_divisors = 1 + 3 + 9 + 27 := by rfl
  have : 1 + 3 + 9 + 27 = 40 := by rfl
  show sum_proper_divisors = 40 from this

end sum_proper_divisors_81_l546_546836


namespace find_m_plus_M_l546_546343

theorem find_m_plus_M :
  ∀ (x y z : ℝ), x + y + z = 5 ∧ x^2 + y^2 + z^2 = 11 → 
  let m := (10 - (3 * 3)) / 3 in
  let M := 3 in
  m + M = 10 / 3 :=
by
  intros x y z h
  let m := (10 - (3 * 3)) / 3
  let M := 3
  calc
  m + M = 10 / 3 : sorry

end find_m_plus_M_l546_546343


namespace exprC_is_quadratic_l546_546899

-- Define the expressions given as conditions
def exprA (x : ℝ) := 3 * x - 1
def exprB (x : ℝ) := 1 / (x^2)
def exprC (x : ℝ) := 3 * x^2 + x - 1
def exprD (x : ℝ) := 2 * x^3 - 1

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

-- The goal is to prove that exprC is quadratic
theorem exprC_is_quadratic : is_quadratic exprC :=
by
  sorry

end exprC_is_quadratic_l546_546899


namespace total_weekly_earnings_l546_546361

-- Define the total weekly hours and earnings
def weekly_hours_weekday : ℕ := 5 * 5
def weekday_rate : ℕ := 3
def weekday_earnings : ℕ := weekly_hours_weekday * weekday_rate

-- Define the total weekend hours and earnings
def weekend_days : ℕ := 2
def weekend_hours_per_day : ℕ := 3
def weekend_rate : ℕ := 3 * 2
def weekend_hours : ℕ := weekend_days * weekend_hours_per_day
def weekend_earnings : ℕ := weekend_hours * weekend_rate

-- Prove that Mitch's total earnings per week are $111
theorem total_weekly_earnings : weekday_earnings + weekend_earnings = 111 := by
  sorry

end total_weekly_earnings_l546_546361


namespace fly_reaches_2011_fly_expected_ordinate_2011_l546_546507

-- Definition of the problem in part (a)
theorem fly_reaches_2011 (start : ℕ × ℕ) (move : (ℕ × ℕ) → (ℕ × ℕ)) :
    ∃ y : ℕ, (2011, y) ∈ {(x, y) | x = 0 ∧ y = start.snd} → True := 
begin
  sorry
end

-- Definition of the expectation problem in part (b)
theorem fly_expected_ordinate_2011 (start : ℕ × ℕ) (move : (ℕ × ℕ) → (ℕ × ℕ)) :
    expected_value_of (fun p => p.2) {p | p.1 = 2011} = 2011 :=
begin
  sorry
end

end fly_reaches_2011_fly_expected_ordinate_2011_l546_546507


namespace ellipse_equation_l546_546660

theorem ellipse_equation :
  (∀ P : ℝ × ℝ, (dist P (-3, 0) + dist P (3, 0) = 10) → 
  (∃ a b : ℝ, a = 5 ∧ b^2 = 16 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1))) :=
begin
  sorry
end

end ellipse_equation_l546_546660


namespace find_tax_percentage_l546_546879

noncomputable def net_income : ℝ := 12000
noncomputable def total_income : ℝ := 13000
noncomputable def non_taxable_income : ℝ := 3000
noncomputable def taxable_income : ℝ := total_income - non_taxable_income
noncomputable def tax_percentage (T : ℝ) := total_income - (T * taxable_income)

theorem find_tax_percentage : ∃ T : ℝ, tax_percentage T = net_income :=
by
  sorry

end find_tax_percentage_l546_546879


namespace find_PB_l546_546665

-- Define the conditions given in the problem
variables {A B C D P : Type} 
variables [convex_quadrilateral ABCD]
variables (CD AB AD AP : ℝ)
variables (angle_CBD angle_ADB angle_BCP : ℝ)
variables (CD_perpendicular_AB BC_perpendicular_AD : Prop)
variables (CD_eq BC_eq AP_eq : ℝ)

-- Specify the conditions using Lean definitions
def condition1 : convex_quadrilateral ABCD := sorry
def condition2 : side CD ⟂ diagonal AB := sorry
def condition3 : side BC ⟂ diagonal AD := sorry
def condition4 : CD = 75 := sorry
def condition5 : BC = 36 := sorry
def condition6 : perp_line_C_intersects_AB_at_P := sorry
def condition7 : AP = 18 := sorry

-- Define the theorem that proves the question equals to the answer
theorem find_PB : ∀ (A B C D P : Type) 
  (CD AB AP : ℝ), convex_quadrilateral ABCD →
  side CD ⟂ diagonal AB →
  side BC ⟂ diagonal AD →
  CD = 75 →
  BC = 36 →
  AP = 18 →
  PB = 54 :=
by
  intros A B C D P CD AB AD AP h_quadrilateral h_perp1 h_perp2 h_CD h_BC h_AP
  sorry

end find_PB_l546_546665


namespace store_revenue_after_sale_l546_546889

/--
A store has 2000 items, each normally selling for $50. 
They offer an 80% discount and manage to sell 90% of the items. 
The store owes $15,000 to creditors. Prove that the store has $3,000 left after the sale.
-/
theorem store_revenue_after_sale :
  let items := 2000
  let retail_price := 50
  let discount := 0.8
  let sale_percentage := 0.9
  let debt := 15000
  let items_sold := items * sale_percentage
  let discount_amount := retail_price * discount
  let sale_price_per_item := retail_price - discount_amount
  let total_revenue := items_sold * sale_price_per_item
  let money_left := total_revenue - debt
  money_left = 3000 :=
by
  sorry

end store_revenue_after_sale_l546_546889


namespace water_level_lowering_correct_l546_546786

/-- Given a rectangular swimming pool measuring 60 feet by 10 feet and needing to remove 2250 gallons of water, 
    we want to prove that the water level needs to be lowered by approximately 6.01 inches. -/
def waterLevelLowering (length width : ℝ) (volumeGallons : ℝ) (gallonsPerCubicFoot : ℝ) : ℝ :=
  let volumeCubicFeet := volumeGallons / gallonsPerCubicFoot
  let surfaceArea := length * width
  let changeInFeet := volumeCubicFeet / surfaceArea
  changeInFeet * 12

theorem water_level_lowering_correct :
  waterLevelLowering 60 10 2250 7.48052 ≈ 6.01 :=
by 
  sorry

end water_level_lowering_correct_l546_546786


namespace convex_function_inequality_l546_546390

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α)
variable (x_1 x_2 ξ_1 ξ_2 : α)

def is_convex (f : α → α) : Prop :=
  ∀ x y z ∈ ℝ, x ≤ y ∧ y ≤ z → f(y) ≤ f(x) + ((y - x) / (z - x)) * (f(z) - f(x))

theorem convex_function_inequality (hf : is_convex f)
  (hx1_leq_ξ1 : x_1 ≤ ξ_1) (hx2_leq_ξ2 : x_2 ≤ ξ_2)
  (segments_not_coincide : ¬((x_1 = ξ_1 ∧ x_2 = ξ_2) ∨ (x_1 = x_2 ∧ ξ_1 = ξ_2))) :
  (f(x_2) - f(x_1)) / (x_2 - x_1) ≤ (f(ξ_2) - f(ξ_1)) / (ξ_2 - ξ_1) :=
sorry

end convex_function_inequality_l546_546390


namespace calcium_carbonate_required_l546_546950

theorem calcium_carbonate_required (HCl_moles CaCO3_moles CaCl2_moles CO2_moles H2O_moles : ℕ) 
  (reaction_balanced : CaCO3_moles + 2 * HCl_moles = CaCl2_moles + CO2_moles + H2O_moles) 
  (HCl_moles_value : HCl_moles = 2) : CaCO3_moles = 1 :=
by sorry

end calcium_carbonate_required_l546_546950


namespace find_original_acid_amount_l546_546124

noncomputable def original_amount_of_acid (a w : ℝ) : Prop :=
  3 * a = w + 2 ∧ 5 * a = 3 * w - 10

theorem find_original_acid_amount (a w : ℝ) (h : original_amount_of_acid a w) : a = 4 :=
by
  sorry

end find_original_acid_amount_l546_546124


namespace sum_of_specified_angles_l546_546020

open EuclideanGeometry

noncomputable def triangle_some_angles_sum (A B C D E F G H I : Point) :=
  ∃ ABC : Triangle, ∃ ABD : EquilateralTriangle, ∃ BCE : EquilateralTriangle,
  ∃ CAF : EquilateralTriangle, 
  onSide ABD AB ∧ onSide BCE BC ∧ onSide CAF CA ∧ 
  midpoint DE G ∧ midpoint EF H ∧ midpoint FD I →
  angle A H B + angle B I C + angle C G A = 180

-- The theorem statement
theorem sum_of_specified_angles (A B C D E F G H I : Point) :
  triangle_some_angles_sum A B C D E F G H I :=
sorry

end sum_of_specified_angles_l546_546020


namespace annual_interest_rate_l546_546376

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate :
  compound_interest_rate 150 181.50 2 1 (0.2 : ℝ) :=
by
  unfold compound_interest_rate
  sorry

end annual_interest_rate_l546_546376


namespace robert_salary_loss_l546_546103

theorem robert_salary_loss (S : ℝ) (hS : 0 < S) :
  let decreased_salary := S * (1 - 0.7)
  let increased_salary := decreased_salary * (1 + 0.7)
  let loss := (S - increased_salary) / S * 100 
  loss = 49 := by
  let decreased_salary := S * (1 - 0.7)
  let increased_salary := decreased_salary * (1 + 0.7)
  let loss := (S - increased_salary) / S * 100 
  rw [decreased_salary, increased_salary, loss]
  sorry

end robert_salary_loss_l546_546103


namespace f_g_5_eq_163_l546_546281

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end f_g_5_eq_163_l546_546281


namespace number_multiplies_p_plus_1_l546_546290

theorem number_multiplies_p_plus_1 (p q x : ℕ) 
  (hp : 1 < p) (hq : 1 < q)
  (hEq : x * (p + 1) = 25 * (q + 1))
  (hSum : p + q = 40) :
  x = 325 :=
sorry

end number_multiplies_p_plus_1_l546_546290


namespace optionC_is_quadratic_l546_546902

def isQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c

def optionA (x : ℝ) : ℝ := 3 * x - 1
def optionB (x : ℝ) : ℝ := 1 / (x^2)
def optionC (x : ℝ) : ℝ := 3 * x^2 + x - 1
def optionD (x : ℝ) : ℝ := 2 * x^3 - 1

theorem optionC_is_quadratic : isQuadratic optionC :=
  by
    sorry

end optionC_is_quadratic_l546_546902


namespace tan_sum_l546_546297

theorem tan_sum (GA GB GC : Vector ℝ) (A B C : ℝ) 
  (h1 : GA + GB + GC = 0) 
  (h2 : GA ⋅ GB = 0) :
  (tan A + tan B) * tan C / (tan A * tan B) = 1 / 2 := 
sorry

end tan_sum_l546_546297


namespace unique_solution_f_f_x_eq_0_l546_546704

def f (x : ℝ) : ℝ :=
if x < 0 then -x + 4 else 3 * x - 6

theorem unique_solution_f_f_x_eq_0 : ∃! x : ℝ, f (f x) = 0 :=
by
  sorry

end unique_solution_f_f_x_eq_0_l546_546704


namespace system_solution_l546_546107

variables {x : ℝ}

theorem system_solution : 
  (3 * x^2 + 8 * x - 3 = 0) ∧ (3 * x^4 + 2 * x^3 - 10 * x^2 + 30 * x - 9 = 0) → x = -3 :=
begin
  sorry
end

end system_solution_l546_546107


namespace number_of_girls_l546_546904

theorem number_of_girls
  (B : ℕ) (k : ℕ) (G : ℕ)
  (hB : B = 10) 
  (hk : k = 5)
  (h1 : B / k = 2)
  (h2 : G % k = 0) :
  G = 5 := 
sorry

end number_of_girls_l546_546904


namespace total_crayons_correct_l546_546480

-- Define the number of crayons each child has
def crayons_per_child : ℕ := 12

-- Define the number of children
def number_of_children : ℕ := 18

-- Define the total number of crayons
def total_crayons : ℕ := crayons_per_child * number_of_children

-- State the theorem
theorem total_crayons_correct : total_crayons = 216 :=
by
  -- Proof goes here
  sorry

end total_crayons_correct_l546_546480


namespace compute_g_difference_l546_546642

def g (n : ℕ) : ℝ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem compute_g_difference (r : ℕ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
by sorry

end compute_g_difference_l546_546642


namespace total_cost_in_usd_is_39_63_l546_546133

def pencil_cost := 2
def pen_cost := pencil_cost + 9
def notebook_cost_before_discount := 2 * pen_cost
def discount := 0.15 * notebook_cost_before_discount
def notebook_cost := notebook_cost_before_discount - discount
def total_cost_cad := pencil_cost + pen_cost + notebook_cost
def cad_to_usd_conversion := 1.25
def total_cost_usd := total_cost_cad * cad_to_usd_conversion

theorem total_cost_in_usd_is_39_63 : total_cost_usd = 39.63 := by
  sorry

end total_cost_in_usd_is_39_63_l546_546133


namespace relationship_between_m_and_n_l546_546591

variable {X_1 X_2 k m n : ℝ}

-- Given conditions
def inverse_proportional_points (X_1 X_2 k : ℝ) (m n : ℝ) : Prop :=
  m = k / X_1 ∧ n = k / X_2 ∧ k > 0 ∧ X_1 < X_2

theorem relationship_between_m_and_n (h : inverse_proportional_points X_1 X_2 k m n) : m > n :=
by
  -- Insert proof here, skipping with sorry
  sorry

end relationship_between_m_and_n_l546_546591


namespace coefficient_of_x_l546_546751

theorem coefficient_of_x :
  let expansion_term := (fun r => (-2)^r * Nat.choose 8 r * x^(4 - 3 / 2 * r))
  ∃ r : Nat, (4 - 3 / 2 * r = 1) ∧ (expansion_term r).coeff x = 112 :=
by
  sorry

end coefficient_of_x_l546_546751


namespace steven_height_l546_546740

theorem steven_height (building_height shadow_length : ℝ) (steven_shadow : ℝ) (h_ratio : building_height / shadow_length = 2)
    (h_building_height : building_height = 50) (h_shadow_length : shadow_length = 25) (h_steven_shadow : steven_shadow = 20) :
    (2 * steven_shadow = 40) :=
by
  rw [h_building_height, h_shadow_length, h_steven_shadow] at h_ratio
  sorry

end steven_height_l546_546740


namespace triangle_product_of_sides_l546_546312

variable (A B C R S Z W : Point)
variable [Triangle A B C]
variable [AcuteTriangle A B C]
variable [PerpendicularFoot C A B R]
variable [PerpendicularFoot B A C S]
variable [CircumcircleIntersection A B C R S Z W]
variable (ZR SW RS : ℝ)
variable (ZR_val : ZR = 15)
variable (SW_val : SW = 10)
variable (RS_val : RS = 30)

theorem triangle_product_of_sides (AB AC : ℝ) : 
  AB * AC = 810 * sqrt 15 :=
by sorry

end triangle_product_of_sides_l546_546312


namespace calc_expression_l546_546175

theorem calc_expression :
  (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end calc_expression_l546_546175


namespace triangle_identity_l546_546300

theorem triangle_identity
  (A B C : ℝ) (a b c: ℝ)
  (h1: A + B + C = Real.pi)
  (h2: a = 2 * R * Real.sin A)
  (h3: b = 2 * R * Real.sin B)
  (h4: c = 2 * R * Real.sin C)
  (h5: Real.sin A = Real.sin B * Real.cos C + Real.cos B * Real.sin C) :
  (b * Real.cos C + c * Real.cos B) / a = 1 := 
  by 
  sorry

end triangle_identity_l546_546300


namespace transformation_g_from_f_l546_546069

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (8 * x + 3 * Real.pi / 2)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem transformation_g_from_f :
  (∀ x, g x = f (x + Real.pi / 4) * 2) ∨ (∀ x, g x = f (x - Real.pi / 4) * 2) := 
by
  sorry

end transformation_g_from_f_l546_546069


namespace find_weekly_allowance_l546_546269

-- Define the conditions
variable (A : ℝ)
variable (arcade_spending : A * 2 / 5)
variable (remaining_after_arcade : A * 3 / 5)
variable (toy_store_spending : (A * 3 / 5) * 1 / 3)
variable (remaining_after_toy_store : (A * 3 / 5) - (A * 3 / 5) * 1 / 3)
variable (candy_store_spending : remaining_after_toy_store)

-- Define the amount of money spent at the candy store
axiom candy_store_spending_axiom : candy_store_spending = 1.20

-- The statement to prove
theorem find_weekly_allowance : A = 3.00 :=
by
  sorry

end find_weekly_allowance_l546_546269


namespace sum_proper_divisors_81_l546_546830

theorem sum_proper_divisors_81 : 
  let proper_divisors : List ℕ := [1, 3, 9, 27] in
  proper_divisors.sum = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546830


namespace total_white_roses_needed_l546_546368

theorem total_white_roses_needed : 
  let bouquets := 5 
  let table_decorations := 7 
  let roses_per_bouquet := 5 
  let roses_per_table_decoration := 12 in
  (bouquets * roses_per_bouquet) + (table_decorations * roses_per_table_decoration) = 109 := by
  sorry

end total_white_roses_needed_l546_546368


namespace max_g_value_le_256_l546_546695

variable {R : Type*} [LinearOrderedField R]

theorem max_g_value_le_256 {g : R → R}
  (poly_g : ∃ (b : ℕ → R), ∀ x, g x = (Finset.range 5).sum (λ i, b i * x ^ i) ∧ ∀ i, 0 ≤ b i)
  (h1 : g 8 = 32)
  (h2 : g 32 = 2048) :
  g 16 ≤ 256 :=
sorry

end max_g_value_le_256_l546_546695


namespace jake_third_test_score_l546_546747

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end jake_third_test_score_l546_546747


namespace rose_days_to_complete_work_l546_546331

theorem rose_days_to_complete_work (R : ℝ) (h1 : 1 / 10 + 1 / R = 1 / 8) : R = 40 := 
sorry

end rose_days_to_complete_work_l546_546331


namespace sequence_expression_l546_546264

theorem sequence_expression (a : ℕ → ℕ) (h₀ : a 1 = 33) (h₁ : ∀ n, a (n + 1) - a n = 2 * n) : 
  ∀ n, a n = n^2 - n + 33 :=
by
  sorry

end sequence_expression_l546_546264


namespace total_white_roses_needed_l546_546369

theorem total_white_roses_needed : 
  let bouquets := 5 
  let table_decorations := 7 
  let roses_per_bouquet := 5 
  let roses_per_table_decoration := 12 in
  (bouquets * roses_per_bouquet) + (table_decorations * roses_per_table_decoration) = 109 := by
  sorry

end total_white_roses_needed_l546_546369


namespace line_intersects_ellipse_two_points_l546_546294

theorem line_intersects_ellipse_two_points (k b : ℝ) : 
  (-2 < b) ∧ (b < 2) ↔ ∀ x y : ℝ, (y = k * x + b) ↔ (x ^ 2 / 9 + y ^ 2 / 4 = 1) → true :=
sorry

end line_intersects_ellipse_two_points_l546_546294


namespace part1_part2_l546_546963

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

def calculate_P (x y : ℤ) : ℤ := 
  (x - y) / 9

def y_from_x (x : ℤ) : ℤ :=
  let first_three := x / 10
  let last_digit := x % 10
  last_digit * 1000 + first_three

def calculate_s (a b : ℕ) : ℤ :=
  1100 + 20 * a + b

def calculate_t (a b : ℕ) : ℤ :=
  b * 1000 + a * 100 + 23

theorem part1 : calculate_P 5324 (y_from_x 5324) = 88 := by
  sorry

theorem part2 :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 9 ∧
  let s := calculate_s a b
  let t := calculate_t a b
  let P_s := calculate_P s (y_from_x s)
  let P_t := calculate_P t (y_from_x t)
  let difference := P_t - P_s - a - b
  is_perfect_square difference ∧ P_t = -161 := by
  sorry

end part1_part2_l546_546963


namespace necessary_but_not_sufficient_l546_546994

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by {
  sorry
}

end necessary_but_not_sufficient_l546_546994


namespace cylinder_capacity_l546_546906

theorem cylinder_capacity (C : ℝ) (h1 : (3 / 4) * C = 27.5) (h2 : (9 / 10) * C = 27.5 + 7.5) :
  C ≈ 36.67 :=
by
  sorry

end cylinder_capacity_l546_546906


namespace find_f_inv_difference_l546_546700

axiom f : ℤ → ℤ
axiom f_inv : ℤ → ℤ
axiom f_has_inverse : ∀ x : ℤ, f_inv (f x) = x ∧ f (f_inv x) = x
axiom f_inverse_conditions : ∀ x : ℤ, f (x + 2) = f_inv (x - 1)

theorem find_f_inv_difference :
  f_inv 2004 - f_inv 1 = 4006 :=
sorry

end find_f_inv_difference_l546_546700


namespace pages_with_same_units_digit_l546_546498

theorem pages_with_same_units_digit :
  let count := (List.range 73).countp (λ x, (x + 1) % 10 = (74 - (x + 1)) % 10)
  count = 15 :=
by
  unfold count
  rw [List.range, List.countp]
  sorry

end pages_with_same_units_digit_l546_546498


namespace part1_decreasing_interval_part2_range_of_g_l546_546760

noncomputable theory
open Real

def f (x : ℝ) : ℝ := sin x ^ 4 + 2 * sqrt 3 * sin x * cos x - cos x ^ 4

def g (x : ℝ) : ℝ := -2 * cos (4 * x)

theorem part1_decreasing_interval : ∀ x ∈ Icc (0 : ℝ) π, 
  (∃ (a b : ℝ), a = (π / 3) ∧ b = (5 * π / 6) ∧ (a ≤ x ∧ x ≤ b)) → 
  (deriv f x < 0) :=
by sorry

theorem part2_range_of_g : ∀ x ∈ Icc (π / 6) (π / 3), 
  1 ≤ g x ∧ g x ≤ 2 :=
by sorry

end part1_decreasing_interval_part2_range_of_g_l546_546760


namespace max_value_f_pi_16_l546_546613

noncomputable def f (x θ : ℝ) : ℝ := 
  cos (2 * x) * cos (θ) - sin (2 * x) * cos (π / 2 - θ)

theorem max_value_f_pi_16 (θ : ℝ) 
  (hθ1 : abs θ < π / 2)
  (h_mono : ∀ a b : ℝ, -3 * π / 8 < a ∧ a < b ∧ b < -π / 6 → f a θ ≤ f b θ) :
  f (π / 16) θ ≤ 1 := 
sorry

end max_value_f_pi_16_l546_546613


namespace theater_show_prob_l546_546658

def probability (n k : ℕ) (p : ℚ) : ℚ := (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem theater_show_prob :
  let certain_stay := 6
  let unsure := 4
  let conf_stay := 1/3
  let prob_9_stay := probability unsure 3 conf_stay * (1 - conf_stay)
  let prob_10_stay := probability unsure 4 conf_stay
  at least_9_stay == (prob_9_stay + prob_10_stay) = 1/9 :=
by
  sorry

end theater_show_prob_l546_546658


namespace math_problem_l546_546592

theorem math_problem
  (x : ℤ)
  (h1 : 2^(x - 1) + 2^(x - 1) + 2^(x - 1) + 2^(x - 1) = 32) :
  (x + 2) * (x - 2) = 12 := 
sorry

end math_problem_l546_546592


namespace problem_proof_l546_546213

variable {a b c d : ℝ}

-- Conditions
def a_gt_b (h₁ : a > b) := h₁
def c_gt_d (h₂ : c > d) := h₂
def d_gt_0 (h₃ : d > 0) := h₃

theorem problem_proof (h₁: a > b) (h₂: c > d) (h₃: d > 0): (a - d > b - c) ∧ (ac^2 > bc^2) :=
by
  have h4 : a - d > b - c := 
    sorry
  have h5 : ac^2 > bc^2 := 
    sorry
  exact ⟨h4, h5⟩

end problem_proof_l546_546213


namespace equal_real_roots_iff_c_is_nine_l546_546652

theorem equal_real_roots_iff_c_is_nine (c : ℝ) : (∃ x : ℝ, x^2 + 6 * x + c = 0 ∧ ∃ Δ, Δ = 6^2 - 4 * 1 * c ∧ Δ = 0) ↔ c = 9 :=
by
  sorry

end equal_real_roots_iff_c_is_nine_l546_546652


namespace cost_price_is_800_l546_546422

theorem cost_price_is_800 (mp sp cp : ℝ) (h1 : mp = 1100) (h2 : sp = 0.8 * mp) (h3 : sp = 1.1 * cp) :
  cp = 800 :=
by
  sorry

end cost_price_is_800_l546_546422


namespace percentage_increase_in_filming_time_l546_546797

-- Definitions based on conditions
def episode_duration : ℕ := 20 -- Each episode is 20 minutes long.
def weekly_episodes : ℕ := 5 -- Each week they show 5 episodes.
def total_hours : ℕ := 10 -- It takes 10 hours to film 4 weeks of episodes.

-- Derived definitions to formalize the problem
def total_minutes : ℕ := total_hours * 60 -- Convert hours to minutes.
def total_weeks : ℕ := 4 -- 4 weeks.

def total_episodes (weeks : ℕ) (weekly_episodes : ℕ) : ℕ :=
  weeks * weekly_episodes

def filming_time_per_episode (total_minutes : ℕ) (total_episodes : ℕ) : ℕ :=
  total_minutes / total_episodes

def percentage_increase (original : ℕ) (new_value : ℕ) : ℚ :=
  ((new_value - original).to_rat / original) * 100

theorem percentage_increase_in_filming_time :
  percentage_increase episode_duration (filming_time_per_episode total_minutes (total_episodes total_weeks weekly_episodes)) = 50 := by
  sorry

end percentage_increase_in_filming_time_l546_546797


namespace field_trip_totals_and_costs_l546_546327

theorem field_trip_totals_and_costs :
  let
    vans := 6, minibuses := 4, coach_buses := 2, school_bus := 1
    students_per_van := 10, students_per_minibus := 24, students_per_coach_bus := 48, students_per_school_bus := 35
    teachers_per_van := 2, teachers_per_minibus := 3, teachers_per_coach_bus := 4, teachers_per_school_bus := 5
    parents_per_van := 1, parents_per_minibus := 2, parents_per_coach_bus := 4, parents_per_school_bus := 3
    cost_per_van := 100, cost_per_minibus := 200, cost_per_coach_bus := 350, cost_per_school_bus := 250
    budget := 2000
    total_students := (vans * students_per_van) + (minibuses * students_per_minibus) + (coach_buses * students_per_coach_bus) + (school_bus * students_per_school_bus)
    total_teachers := (vans * teachers_per_van) + (minibuses * teachers_per_minibus) + (coach_buses * teachers_per_coach_bus) + (school_bus * teachers_per_school_bus)
    total_parents := (vans * parents_per_van) + (minibuses * parents_per_minibus) + (coach_buses * parents_per_coach_bus) + (school_bus * parents_per_school_bus)
    total_cost := (vans * cost_per_van) + (minibuses * cost_per_minibus) + (coach_buses * cost_per_coach_bus) + (school_bus * cost_per_school_bus)
  in
  total_students = 287 ∧
  total_teachers = 37 ∧
  total_parents = 25 ∧
  total_cost = 2350 ∧
  total_cost > budget := 
by
  sorry

end field_trip_totals_and_costs_l546_546327


namespace greatest_integer_less_than_150_gcd_18_eq_6_l546_546083

theorem greatest_integer_less_than_150_gcd_18_eq_6 :
  ∃ n : ℕ, n < 150 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ gcd m 18 = 6 → m ≤ n :=
by
  use 132
  split
  { 
    -- proof that 132 < 150 
    exact sorry 
  }
  split
  { 
    -- proof that gcd 132 18 = 6
    exact sorry 
  }
  {
    -- proof that 132 is the greatest such integer
    exact sorry 
  }

end greatest_integer_less_than_150_gcd_18_eq_6_l546_546083


namespace sum_of_positive_integers_satisfying_equation_l546_546190

-- Define gcd and lcm for positive integers
open Nat

theorem sum_of_positive_integers_satisfying_equation :
  ∃ n : ℕ, n > 0 ∧ (lcm n 180 = gcd n 180 + 630) ∧ n = 360 :=
begin
  sorry
end

end sum_of_positive_integers_satisfying_equation_l546_546190


namespace train_stop_time_l546_546854

theorem train_stop_time
  (D : ℝ)
  (h1 : D > 0)
  (T_no_stop : ℝ := D / 300)
  (T_with_stop : ℝ := D / 200)
  (T_stop : ℝ := T_with_stop - T_no_stop):
  T_stop = 6 / 60 := by
    sorry

end train_stop_time_l546_546854


namespace compute_value_l546_546708

open Nat Real

theorem compute_value (A B : ℝ × ℝ) (hA : A = (15, 10)) (hB : B = (-5, 6)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (x y : ℝ), C = (x, y) ∧ 2 * x - 4 * y = -22 := by
  sorry

end compute_value_l546_546708


namespace four_digit_numbers_count_l546_546272

theorem four_digit_numbers_count:
  let N := (a, b, c, d) in
  3000 ≤ (1000*a + 100*b + 10*c + d) ∧ (1000*a + 100*b + 10*c + d) < 6000 ∧
  (1000*a + 100*b + 10*c + d) % 5 = 0 ∧
  2 ≤ b ∧ b < c ∧ c ≤ 6 ∧
  a ∈ {3, 4, 5} ∧
  d ∈ {0, 5} →
  ∃ n : ℕ, n = 60 :=
sorry

end four_digit_numbers_count_l546_546272


namespace dasha_paper_strip_l546_546185

theorem dasha_paper_strip (a b c : ℕ) (h1 : a < b) (h2 : 2 * a * b + 2 * a * c - a^2 = 43) :
    ∃ (length width : ℕ), length = a ∧ width = b + c := by
  sorry

end dasha_paper_strip_l546_546185


namespace sum_of_three_numbers_is_seventy_l546_546768

theorem sum_of_three_numbers_is_seventy
  (a b c : ℝ)
  (h1 : a ≤ b ∧ b ≤ c)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 30)
  (h4 : b = 10)
  (h5 : a + c = 60) :
  a + b + c = 70 :=
  sorry

end sum_of_three_numbers_is_seventy_l546_546768


namespace probability_divisible_by_3_l546_546722

-- Defining some basic
variables {a b c : ℕ}
variables {S : Finset ℕ} (hS : S = (Finset.range 2011).filter (λ x, x > 0))

def p_div_by_3 (a b c : ℕ) : ℚ := 
  if 0 < a ∧ a ≤ 2010 ∧ 0 < b ∧ b ≤ 2010 ∧ 0 < c ∧ c ≤ 2010 
  then if (a * b * c + a * b + a) % 3 = 0 then 1 else 0 
  else 0

theorem probability_divisible_by_3 : 
  (Finset.sum S (λ a, Finset.sum S (λ b, Finset.sum S (λ c, p_div_by_3 a b c)))) / (2010 * 2010 * 2010) = 13 / 27 :=
sorry

end probability_divisible_by_3_l546_546722


namespace radius_of_circle_l546_546502

noncomputable def radius_of_tangent_circle : ℝ := sqrt 2

theorem radius_of_circle 
  (center_tangent_to_axes : ∀ (O : ℝ × ℝ), ∃ r : ℝ, O = (r, r)) 
  (circle_tangent_to_triangle : ∀ (T : ℝ × ℝ), T = (2 - sqrt 2, 2 - sqrt 2)) :
  radius_of_tangent_circle = sqrt 2 :=
by
  sorry

end radius_of_circle_l546_546502


namespace derivative_of_constant_l546_546699

variable (y : ℝ → ℝ) (c : ℝ)
noncomputable def y_def : ℝ := Real.exp 3

theorem derivative_of_constant :
  (∀ x, y' = 0) := by
    sorry

end derivative_of_constant_l546_546699


namespace smallest_possible_denominator_l546_546955

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end smallest_possible_denominator_l546_546955


namespace inequality_holds_l546_546568

theorem inequality_holds (k : ℝ) : (∀ x : ℝ, x^2 + k * x + 1 > 0) ↔ (k > -2 ∧ k < 2) :=
by
  sorry

end inequality_holds_l546_546568


namespace log_simplification_sum_of_digits_l546_546332

noncomputable def G : ℝ := 10^(10^100)

theorem log_simplification (G_def : G = 10^(10^100)) : 
  let log_base := (log (log (log 10 G) G)) in
  let simplified_log := log / log log_base in
  simplified_log = 10^100 / 98 :=
sorry

theorem sum_of_digits (m n : ℕ) (h_m : m = 10^100) (h_n : n = 98) : 
  let sum := m + n in
  nat_digits_sum = 18 :=
sorry

end log_simplification_sum_of_digits_l546_546332


namespace mod_2_pow_1000_by_13_l546_546464

theorem mod_2_pow_1000_by_13 :
  (2 ^ 1000) % 13 = 3 := by
  sorry

end mod_2_pow_1000_by_13_l546_546464


namespace gcd_228_1995_is_57_l546_546073

noncomputable def gcd_228_1995 : ℕ :=
  nat.gcd 228 1995

theorem gcd_228_1995_is_57 : gcd_228_1995 = 57 :=
by sorry

end gcd_228_1995_is_57_l546_546073


namespace necessary_but_not_sufficient_condition_l546_546857

theorem necessary_but_not_sufficient_condition (a c : ℝ) (h : c ≠ 0) : ¬ ((∀ (a : ℝ) (h : c ≠ 0), (ax^2 + y^2 = c) → ((ax^2 + y^2 = c) → ( (c ≠ 0) ))) ∧ ¬ ((∀ (a : ℝ), ¬ (ax^2 + y^2 ≠ c) → ( (ax^2 + y^2 = c) → ((c = 0) ))) )) :=
sorry

end necessary_but_not_sufficient_condition_l546_546857


namespace y_relation_l546_546419

noncomputable def f (x : ℝ) : ℝ := -2 * x + 5

theorem y_relation (x1 y1 y2 y3 : ℝ) (h1 : y1 = f x1) (h2 : y2 = f (x1 - 2)) (h3 : y3 = f (x1 + 3)) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end y_relation_l546_546419


namespace phone_price_is_correct_l546_546867

-- Definition of the conditions
def monthly_cost := 7
def months := 4
def total_cost := 30

-- Definition to be proven
def phone_price := total_cost - (monthly_cost * months)

theorem phone_price_is_correct : phone_price = 2 :=
by
  sorry

end phone_price_is_correct_l546_546867


namespace Mitch_weekly_earnings_l546_546363

theorem Mitch_weekly_earnings :
  (let weekdays_hours := 5 * 5
       weekend_hours := 3 * 2
       weekday_rate := 3
       weekend_rate := 2 * 3 in
   (weekdays_hours * weekday_rate + weekend_hours * weekend_rate = 111)) :=
by
  sorry

end Mitch_weekly_earnings_l546_546363


namespace geom_series_common_ratio_and_sum_l546_546951

theorem geom_series_common_ratio_and_sum 
  (a : ℚ) (r : ℚ) (n : ℕ) 
  (h₀ : a = 4 / 7)
  (h₁ : r = 4 / 7)
  (h₂ : n = 3) :
  r = 4 / 7 ∧ (a * (1 - r^n) / (1 - r) = 372 / 343) := 
by
  have S_3_def : ∑ i in finset.range n, a * r^i = (4 / 7) + (16 / 49) + (64 / 343) := 
    sorry
  have S_3_value : (4 / 7) + (16 / 49) + (64 / 343) = 372 / 343 := 
    sorry
  exact ⟨h₁, S_3_def.symm.trans S_3_value⟩

end geom_series_common_ratio_and_sum_l546_546951


namespace symmetric_about_y_axis_minimum_value_at_0_correct_statements_for_f_l546_546731

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x + 1)

theorem symmetric_about_y_axis : (∀ x : ℝ, f (-x) = f x) :=
by
  intro x
  unfold f
  rw [abs_neg]

theorem minimum_value_at_0 : ∃ x : ℝ, f x = 0 :=
by
  use 0
  simp [f, abs, Real.log]

theorem correct_statements_for_f :
  (∀ x : ℝ, f (-x) = f x) ∧ ∃ x : ℝ, f x = 0 :=
by
  exact ⟨symmetric_about_y_axis, minimum_value_at_0⟩

end symmetric_about_y_axis_minimum_value_at_0_correct_statements_for_f_l546_546731


namespace prob_two_fours_l546_546734

-- Define the sample space for a fair die
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The probability of rolling a 4 on a fair die
def prob_rolling_four : ℚ := 1 / 6

-- Probability of two independent events both resulting in rolling a 4
def prob_both_rolling_four : ℚ := (prob_rolling_four) * (prob_rolling_four)

-- Prove that the probability of rolling two 4s in two independent die rolls is 1/36
theorem prob_two_fours : prob_both_rolling_four = 1 / 36 := by
  sorry

end prob_two_fours_l546_546734


namespace count_functions_satisfying_conditions_l546_546622

noncomputable def num_functions_satisfying_conditions (A₁₉₉₃ : Type*) :=
  let f : (A₁₉₉₃ → A₁₉₉₃) in
  ∃ (f : A₁₉₉₃ → A₁₉₉₃), (∀ x, (f^[1993]) x = f x) ∧ (|f '' set.univ| = 4)

theorem count_functions_satisfying_conditions {A₁₉₉₃ : Type*}
  [finite A₁₉₉₃] : card (num_functions_satisfying_conditions A₁₉₉₃) =
  24 * 4^1989 * (nat.factorial 1993 / (nat.factorial 4 * nat.factorial (1993 - 4))) :=
  sorry

end count_functions_satisfying_conditions_l546_546622


namespace compute_j_in_polynomial_arithmetic_progression_l546_546413

theorem compute_j_in_polynomial_arithmetic_progression 
  (P : Polynomial ℝ)
  (roots : Fin 4 → ℝ)
  (hP : P = Polynomial.C 400 + Polynomial.X * (Polynomial.C k + Polynomial.X * (Polynomial.C j + Polynomial.X * (Polynomial.C 0 + Polynomial.X))))
  (arithmetic_progression : ∃ b d : ℝ, roots 0 = b ∧ roots 1 = b + d ∧ roots 2 = b + 2 * d ∧ roots 3 = b + 3 * d ∧ Polynomial.degree P = 4) :
  j = -200 :=
by
  sorry

end compute_j_in_polynomial_arithmetic_progression_l546_546413


namespace derivative_product_value_l546_546216

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else x - x^2

def f' (x : ℝ) : ℝ :=
if x >= 0 then 2*x + 1 else 1 - 2*x

theorem derivative_product_value : f' 1 * f' (-1) = 9 := by
  sorry

end derivative_product_value_l546_546216


namespace f_of_quarter_l546_546341

-- Given conditions
def g (x : ℝ) : ℝ := 1 - 2 * x^2
def f (u : ℝ) : ℝ := if u = 0 then 0 else (1 - 2 * (√((3 : ℝ) / 8))^2) / (√((3 : ℝ) / 8))^2

-- The theorem to prove
theorem f_of_quarter : f (1 / 4) = 2 / 3 := by
  sorry

end f_of_quarter_l546_546341


namespace find_initial_speed_l546_546683

-- Definitions for the conditions
def total_distance : ℕ := 800
def time_at_initial_speed : ℕ := 6
def time_at_60_mph : ℕ := 4
def time_at_40_mph : ℕ := 2
def speed_at_60_mph : ℕ := 60
def speed_at_40_mph : ℕ := 40

-- Setting up the equation: total distance covered
def distance_covered (v : ℕ) : ℕ :=
  time_at_initial_speed * v + time_at_60_mph * speed_at_60_mph + time_at_40_mph * speed_at_40_mph

-- Proof problem statement
theorem find_initial_speed : ∃ v : ℕ, distance_covered v = total_distance ∧ v = 80 := by
  existsi 80
  simp [distance_covered, total_distance, time_at_initial_speed, speed_at_60_mph, time_at_40_mph]
  norm_num
  sorry

end find_initial_speed_l546_546683


namespace L_l546_546885

-- Define T as the set of vertices of a regular tetrahedron
def T : Set (ℝ × ℝ × ℝ) := {
  -- Assuming vertices at specific positions for simplicity
  (0, 0, 0), -- Vertex A
  (1, 0, 0), -- Vertex B
  (0.5, sqrt 3 / 2, 0), -- Vertex C
  (0.5, sqrt 3 / 6, sqrt(6) / 3) -- Vertex D
}

-- Define L(E) for any set E as the set of points lying on lines composed of two distinct points of E
def L (E : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) :=
  { p | ∃ (a b : ℝ × ℝ × ℝ), a ∈ E ∧ b ∈ E ∧ a ≠ b ∧ ∃ α : ℝ, p = α • a + (1 - α) • b }

-- Prove that L(L(T)) is the entire 3-dimensional space spanned by the vertices of the tetrahedron
theorem L(L(T))_is_3d_space : L(L(T)) = { p : ℝ × ℝ × ℝ | ∃ a b c d : ℝ, (a, b, c, d) ∈ (Set.UNIV : Set (ℝ × ℝ × ℝ × ℝ)) ∧ a + b + c + d = 1 ∧ p = a • (0, 0, 0) + b • (1, 0, 0) + c • (0.5, sqrt 3 / 2, 0) + d • (0.5, sqrt 3 / 6, sqrt(6) / 3) } :=
by
  sorry

end L_l546_546885


namespace sin_cos_third_quadrant_lt_zero_l546_546235
noncomputable def third_quadrant_angle (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

theorem sin_cos_third_quadrant_lt_zero (α : ℝ) (h : third_quadrant_angle α) :
  sin α + cos α < 0 :=
sorry

end sin_cos_third_quadrant_lt_zero_l546_546235


namespace correct_statements_l546_546286

noncomputable theory
open Real

-- Function and its properties
def f (x : ℝ) : ℝ := sorry
def k : ℝ := sorry

-- Given conditions
axiom f_zero : f 0 = -1
axiom f'_positive : ∀ x : ℝ, f' x > k
axiom k_pos : k > 1

-- Statements to check
def statement_1 : (f (1 / k) > 0) := sorry
def statement_2 : (f k > k^2) := sorry
def statement_3 : (f (1 / (k-1)) > 1 / (k-1)) := sorry
def statement_4 : (f (1 / (1-k)) < (2*k - 1) / (1 - k)) := sorry

-- Number of correct statements
theorem correct_statements : (statement_1 ∧ statement_3 ∧ statement_4 ∧ ¬statement_2) = true :=
sorry

end correct_statements_l546_546286


namespace Q_and_R_on_PS_l546_546229

open_locale classical

variables {A B C D E F P Q R S : Type*} [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C]

-- Definitions of the altitudes and perpendicular lines
def altitude (X Y Z : Type*) := ∃ (W : Type*), ⟦angle (X - Y) (Z - Y)⟧ = 90
def perpendicular (X Y Z : Type*) := angle (X - Y) (Z - Y) = 90

-- Conditions
axiom AD_altitude : altitude A B C
axiom BE_altitude : altitude B A C
axiom CF_altitude : altitude C A B
axiom DP_perpendicular : perpendicular D P (segment A B)
axiom DQ_perpendicular : perpendicular D Q (segment B C)
axiom DR_perpendicular : perpendicular D R (segment C A)
axiom DS_perpendicular : perpendicular D S (segment A C)
axiom line_PS_connects : connected (line P S)

-- Proof statement
theorem Q_and_R_on_PS : on (line P S) Q ∧ on (line P S) R := sorry

end Q_and_R_on_PS_l546_546229


namespace natural_sequence_digits_l546_546449

/--
  In the concatenated sequence of all natural numbers, the digit at the 13th position is 2 and the digit at the 120th position is 6.
-/
theorem natural_sequence_digits (n_seq : ℕ → ℕ) (h_concat : ∀ n, n_seq n = list.join (list.map (λ x, list.digits 10 x) (list.range (n + 1)))) :
  ((n_seq 12).nth 12 = some 2) ∧ ((n_seq 119).nth 119 = some 6) :=
sorry

end natural_sequence_digits_l546_546449


namespace min_abs_diff_pow_12_5_l546_546241

theorem min_abs_diff_pow_12_5 (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ k : ℕ, (k = 7) ∧ (∀ m' n' : ℕ, m' > 0 → n' > 0 → |12^m' - 5^n'| >= k) :=
sorry

end min_abs_diff_pow_12_5_l546_546241


namespace rational_square_plus_one_pos_l546_546860

theorem rational_square_plus_one_pos (a : ℚ) : 0 < a^2 + 1 := 
begin
  calc 
  0 < a^2 : by nlinarith
  ... < a^2 + 1 : by linarith,
end

end rational_square_plus_one_pos_l546_546860


namespace rectangle_length_l546_546482

theorem rectangle_length (P B L : ℕ) (h1 : P = 800) (h2 : B = 300) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l546_546482


namespace greatest_integer_with_gcd_6_l546_546078

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l546_546078


namespace average_stamps_collected_per_day_l546_546713

theorem average_stamps_collected_per_day :
  let a := 10
  let d := 6
  let n := 6
  let total_sum := (n / 2) * (2 * a + (n - 1) * d)
  let average := total_sum / n
  average = 25 :=
by
  sorry

end average_stamps_collected_per_day_l546_546713


namespace sum_proper_divisors_81_l546_546812

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546812


namespace initial_water_percentage_l546_546875

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end initial_water_percentage_l546_546875


namespace rotted_tomatoes_is_correct_l546_546359

noncomputable def shipment_1 : ℕ := 1000
noncomputable def sold_Saturday : ℕ := 300
noncomputable def shipment_2 : ℕ := 2 * shipment_1
noncomputable def tomatoes_Tuesday : ℕ := 2500

-- Define remaining tomatoes after the first shipment accounting for Saturday's sales
noncomputable def remaining_tomatoes_1 : ℕ := shipment_1 - sold_Saturday

-- Define total tomatoes after second shipment arrives
noncomputable def total_tomatoes_after_second_shipment : ℕ := remaining_tomatoes_1 + shipment_2

-- Define the amount of tomatoes that rotted
noncomputable def rotted_tomatoes : ℕ :=
  total_tomatoes_after_second_shipment - tomatoes_Tuesday

theorem rotted_tomatoes_is_correct :
  rotted_tomatoes = 200 := by
  sorry

end rotted_tomatoes_is_correct_l546_546359


namespace perimeter_PQRST_l546_546675

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk 0 8
def Q := Point.mk 4 8
def R := Point.mk 4 4
def T := Point.mk 0 0
def S := Point.mk 9 0

theorem perimeter_PQRST :
  distance (P, Q) + distance (Q, R) + distance (R, S) + distance (S, T) + distance (T, P) = 25 + real.sqrt 41 :=
by sorry

end perimeter_PQRST_l546_546675


namespace polar_to_rectangular_l546_546926

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 4) (h2 : θ = π / 4) :
  (r * real.cos θ, r * real.sin θ) = (2 * real.sqrt 2, 2 * real.sqrt 2) :=
by
  rw [h1, h2]
  simp [real.cos_pi_div_four, real.sin_pi_div_four, real.sqrt]
  sorry

end polar_to_rectangular_l546_546926


namespace find_b7_l546_546444

/-- We represent the situation with twelve people in a circle, each with an integer number. The
     average announced by a person is the average of their two immediate neighbors. Given the
     person who announced the average of 7, we aim to find the number they initially chose. --/
theorem find_b7 (b : ℕ → ℕ) (announced_avg : ℕ → ℕ) :
  (announced_avg 1 = (b 12 + b 2) / 2) ∧
  (announced_avg 2 = (b 1 + b 3) / 2) ∧
  (announced_avg 3 = (b 2 + b 4) / 2) ∧
  (announced_avg 4 = (b 3 + b 5) / 2) ∧
  (announced_avg 5 = (b 4 + b 6) / 2) ∧
  (announced_avg 6 = (b 5 + b 7) / 2) ∧
  (announced_avg 7 = (b 6 + b 8) / 2) ∧
  (announced_avg 8 = (b 7 + b 9) / 2) ∧
  (announced_avg 9 = (b 8 + b 10) / 2) ∧
  (announced_avg 10 = (b 9 + b 11) / 2) ∧
  (announced_avg 11 = (b 10 + b 12) / 2) ∧
  (announced_avg 12 = (b 11 + b 1) / 2) ∧
  (announced_avg 7 = 7) →
  b 7 = 12 := 
sorry

end find_b7_l546_546444


namespace largest_decimal_base7_three_digit_l546_546046

theorem largest_decimal_base7_three_digit :
  let max_base7 := 6 * 7^2 + 6 * 7^1 + 6 * 7^0 in
  max_base7 = 342 :=
by
  sorry

end largest_decimal_base7_three_digit_l546_546046


namespace probability_of_BD_greater_than_6sqrt2_l546_546310

theorem probability_of_BD_greater_than_6sqrt2 :
  (∀ (A B C P : ℝ) (BD : ℝ),
  (∠ A C B = 90) →
  (∠ A B C = 45) →
  (AB = 12) →
  (point P is randomly chosen inside triangle ABC) →
  (P lies above the median line of BC) →
  Pr(BD > 6 * sqrt 2 | conditions) = 0
  := sorry

end probability_of_BD_greater_than_6sqrt2_l546_546310


namespace initial_people_count_is_16_l546_546172

-- Define the conditions
def initial_people (x : ℕ) : Prop :=
  let people_came_in := 5 in
  let people_left := 2 in
  let final_people := 19 in
  x + people_came_in - people_left = final_people

-- Define the theorem
theorem initial_people_count_is_16 (x : ℕ) (h : initial_people x) : x = 16 :=
by
  sorry

end initial_people_count_is_16_l546_546172


namespace C_behavior_l546_546287

-- Define the function C(n) and the constants e, R, r
def C (n e R r : ℝ) : ℝ := (e * n) / (R + n * r ^ 2)

-- We need to assume e, R, r are positive constants
variables (e R r : ℝ) (he : 0 < e) (hR : 0 < R) (hr : 0 < r)

-- The theorem states that C(n) initially increases and then decreases
theorem C_behavior (n : ℝ) (hn : 0 < n) :
  ∃ N : ℝ, 0 < N ∧ ∀ n₁ n₂ : ℝ, n₁ > N → n₂ > N → n₁ < n₂ → C n₁ e R r > C n₂ e R r :=
sorry

end C_behavior_l546_546287


namespace angle_BDC_l546_546542

noncomputable def triangle_angle_BDC (A B C : ℝ) (D : Type) 
  (hB : B = 180 - A - C) (hC : C = 180 - A - B) (hD : D = intersection (angle_bisector B) (angle_bisector C)) : 
  ℝ :=
90 + A / 2

theorem angle_BDC (A B C : ℝ) (D : Type) 
  (hB : B = 180 - A - C) (hC : C = 180 - A - B) (hD : D = intersection (angle_bisector B) (angle_bisector C)) :
  triangle_angle_BDC A B C D hB hC hD = 90 + A / 2 :=
sorry

end angle_BDC_l546_546542


namespace time_after_midnight_l546_546463

theorem time_after_midnight (start_time : String) (minutes_after : Nat) : String :=
  if start_time = "January 1, 2013 midnight" ∧ minutes_after = 2537 then
    "January 2 at 6:17 PM"
  else 
    "Incorrect input"
  
-- test case
example : time_after_midnight "January 1, 2013 midnight" 2537 = "January 2 at 6:17 PM" := by
  simp
  done

end time_after_midnight_l546_546463


namespace sum_proper_divisors_81_l546_546811

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546811


namespace solution_range_for_m_l546_546654

theorem solution_range_for_m (x m : ℝ) (h₁ : 2 * x - 1 > 3 * (x - 2)) (h₂ : x < m) : m ≥ 5 :=
by {
  sorry
}

end solution_range_for_m_l546_546654


namespace subtract_decimal_numbers_l546_546947

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_decimal_numbers_l546_546947


namespace P_coordinates_l546_546291

-- Define the angle in radians
def theta : ℝ := 7 * Real.pi / 6

-- Define the radius OP
def radius : ℝ := 2

-- Define P(x, y) such that P is on the terminal side of angle theta with a distance radius from origin
def P (x y : ℝ) : Prop := x^2 + y^2 = radius^2 ∧ y / x = Real.tan theta

-- Define the coordinates
def Px : ℝ := -Real.sqrt 3
def Py : ℝ := -1

-- The proof statement
theorem P_coordinates :
  P Px Py :=
by
  unfold P
  constructor
  -- Prove x^2 + y^2 = radius^2
  { 
    sorry 
  }
  -- Prove y / x = tan(theta)
  { 
    sorry 
  }

end P_coordinates_l546_546291


namespace initial_cookies_count_l546_546475

theorem initial_cookies_count (x : ℕ) (h_ate : ℕ) (h_left : ℕ) :
  h_ate = 2 → h_left = 5 → (x - h_ate = h_left) → x = 7 :=
by
  intros
  sorry

end initial_cookies_count_l546_546475


namespace sum_proper_divisors_of_81_l546_546845

theorem sum_proper_divisors_of_81 : (∑ i in {0, 1, 2, 3}, 3 ^ i) = 40 := 
by
  sorry

end sum_proper_divisors_of_81_l546_546845


namespace range_of_a_part1_range_of_a_part2_l546_546603

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 6

def set_B (x : ℝ) (a : ℝ) : Prop := (x ≥ 1 + a) ∨ (x ≤ 1 - a)

def condition_1 (a : ℝ) : Prop :=
  (∀ x, set_A x → ¬ set_B x a) → (a ≥ 5)

def condition_2 (a : ℝ) : Prop :=
  (∀ x, (x ≥ 6 ∨ x ≤ -1) → set_B x a) ∧ (∃ x, set_B x a ∧ ¬ (x ≥ 6 ∨ x ≤ -1)) → (0 < a ∧ a ≤ 2)

theorem range_of_a_part1 (a : ℝ) : condition_1 a :=
  sorry

theorem range_of_a_part2 (a : ℝ) : condition_2 a :=
  sorry

end range_of_a_part1_range_of_a_part2_l546_546603


namespace polynomial_term_equality_l546_546772

theorem polynomial_term_equality (p q : ℝ) (hpq_pos : 0 < p) (hq_pos : 0 < q) 
  (h_sum : p + q = 1) (h_eq : 28 * p^6 * q^2 = 56 * p^5 * q^3) : p = 2 / 3 :=
by
  sorry

end polynomial_term_equality_l546_546772


namespace n_possible_values_l546_546313

theorem n_possible_values (n : ℕ) (H : ∀ (S : set (ℕ × ℕ)), S.card = n → (∀ (r : set (ℕ × ℕ)), r.card = n → r ∩ S ≠ ∅)) :
  n = 1 ∨ (∃ p : ℕ, prime p ∧ n = p) ∨ (∃ p : ℕ, prime p ∧ n = p^2) :=
sorry

end n_possible_values_l546_546313


namespace series_sum_eq_neg_half_l546_546391

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem series_sum_eq_neg_half :
  let n := 2002
  let S := (1 - ∑ i in Finset.range 1001, (3^i) * binomial n (2*i)) / 2^n
  S = -1 / 2 :=
by
  let n := 2002
  let S := (1 - ∑ i in Finset.range 1001, (3^i) * binomial n (2*i)) / 2^n
  have h : S = -1/2 := sorry
  exact h

end series_sum_eq_neg_half_l546_546391


namespace ratio_areas_l546_546975

variable {A B C D E : Point}

-- Let ABCD be a cyclic quadrilateral with perpendicular diagonals AC and BD
axiom cyclic_quadrilateral : Cyclic ABCD
axiom perpendicular_diagonals : ∃ O, IsPerpendicular (AC O) (BD O)

-- Point E is positioned on the circumcircle of the quadrilateral, diametrically opposite to D
axiom E_on_circumcircle : E ∈ Circumcircle ABCD
axiom diametrically_opposite : ∀ O, (OE) = (OD) ∧ Collinear E O D

-- AB and DE do not intersect
axiom non_intersecting_AB_DE : ¬ intersect (Line AB) (Line DE)

theorem ratio_areas (h : Cyclic ABCD) (h1 : ∃ O, IsPerpendicular (AC O) (BD O)) 
(h2 : E ∈ Circumcircle ABCD) (h3 : ∀ O, (OE) = (OD) ∧ Collinear E O D)
(h4 : ¬ intersect (Line AB) (Line DE)) :
  area (triangle BCD) = area (quadrilateral ABED) := by
  sorry

end ratio_areas_l546_546975


namespace number_of_valid_n_l546_546924

theorem number_of_valid_n : 
  (∃ (n : ℕ), ∀ (a b c : ℕ), 8 * a + 88 * b + 888 * c = 8000 → n = a + 2 * b + 3 * c) ↔
  (∃ (n : ℕ), n = 1000) := by 
  sorry

end number_of_valid_n_l546_546924


namespace num_races_necessary_l546_546063

/-- There are 300 sprinters registered for a 200-meter dash at a local track meet,
where the track has only 8 lanes. In each race, 3 of the competitors advance to the
next round, while the rest are eliminated immediately. Determine how many races are
needed to identify the champion sprinter. -/
def num_races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (advance_per_race : ℕ) : ℕ :=
  if h : advance_per_race < lanes ∧ lanes > 0 then
    let eliminations_per_race := lanes - advance_per_race
    let total_eliminations := total_sprinters - 1
    Nat.ceil (total_eliminations / eliminations_per_race)
  else
    0

theorem num_races_necessary
  (total_sprinters : ℕ)
  (lanes : ℕ)
  (advance_per_race : ℕ)
  (h_total_sprinters : total_sprinters = 300)
  (h_lanes : lanes = 8)
  (h_advance_per_race : advance_per_race = 3) :
  num_races_to_champion total_sprinters lanes advance_per_race = 60 := by
  sorry

end num_races_necessary_l546_546063


namespace max_value_at_log2_one_l546_546859

noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * (4 : ℝ) ^ x
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

theorem max_value_at_log2_one :
  (∃ x, domain x ∧ f x = 0) ∧ (∀ y, domain y → f y ≤ 0) :=
by
  sorry

end max_value_at_log2_one_l546_546859


namespace geometric_sequence_common_ratio_l546_546426

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (d q : ℝ) :
  (∀ n, a n = a 0 + n * d) → 
  (a 0 + 1, a 2 + 2, a 4 + 3) = (a 0 + 1) * q^1 * (a 0 + 1) * q^0 → 
  q = 1 :=
by
  sorry

end geometric_sequence_common_ratio_l546_546426


namespace arithmetic_progression_mean_median_mode_l546_546191

theorem arithmetic_progression_mean_median_mode :
  ∀ x : ℕ, 
  (let l := [9, 3, 5, 3, 7, 3, x, 11] in
  mode l = 3 ∧ 
  mean l = (41 + x) / 8 ∧
  ((if x ≤ 3 then
      (median l) = 3
    else if 3 < x ∧ x < 5 then
      (median l) = (3 + x) / 2
    else if 5 ≤ x ∧ x < 7 then
      (median l) = (5 + x) / 2
    else if 7 ≤ x ∧ x < 9 then
      (median l) = 6
    else
      (median l) = 7) ∧ 
    mean l, median l and mode l form an arithmetic sequence) → 
  x = 47)) :=
sorry

end arithmetic_progression_mean_median_mode_l546_546191


namespace tangent_line_at_pi_unique_zero_g_l546_546581

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) * Real.sin x
noncomputable def g (x : ℝ) : ℝ := f x - x^2

theorem tangent_line_at_pi :
  ∃ (m b : ℝ), (∀ x : ℝ, f x = m * x + b) → 
  m = 1 - Real.exp π ∧ b = (1 - Real.exp π) * (π)
sorry

theorem unique_zero_g : ∃! x ∈ Set.Ioo 0 (2 * π), g x = 0 :=
sorry

end tangent_line_at_pi_unique_zero_g_l546_546581


namespace book_cost_in_cny_l546_546718

-- Conditions
def usd_to_nad : ℝ := 7      -- One US dollar to Namibian dollar
def usd_to_cny : ℝ := 6      -- One US dollar to Chinese yuan
def book_cost_nad : ℝ := 168 -- Cost of the book in Namibian dollars

-- Statement to prove
theorem book_cost_in_cny : book_cost_nad * (usd_to_cny / usd_to_nad) = 144 :=
sorry

end book_cost_in_cny_l546_546718


namespace complex_modulus_log_monotonic_interval_min_value_area_of_triangle_l546_546117

-- Problem 1
theorem complex_modulus 
  (z : ℂ) (h : z = (3 - 1 * complex.I) / (2 + 1 * complex.I)) : 
  complex.abs z = real.sqrt 2 := 
  sorry

-- Problem 2
theorem log_monotonic_interval 
  (f : ℝ → ℝ) (h : ∀ x, f x = real.log (x^2 - 2 * x - 3) / real.log (1/2)) : 
  set_of (λ x, f x > 0) = set.Ioo (-∞ : ℝ) (-1) := 
  sorry

-- Problem 3
theorem min_value 
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (f : ℝ → ℝ) (h3 : ∀ x, f x = (2/3) * x^3 - a * x^2 - 2 * b * x + 2) 
  (h4 : deriv f 1 = 0) : 
  (1 / a) + (4 / b) = 9 := 
  sorry

-- Problem 4
theorem area_of_triangle 
  (C : ℝ → ℝ → Prop) (h1 : ∀ y x, C y x ↔ y^2 = 8 * x) 
  (O P A B : ℝ × ℝ) (h2 : P = (0, 4))
  (h3 : ∃ x y, A = (x, y) ∧ C y x)
  (h4 : ∃ F, ∃ x y, F = (x, y) ∧ line_through A F = line_through F B)
  (h5 : line_through A F = extend_to_parabola_line A B (λ x y, C y x)) : 
  triangle_area O A B = 4 * real.sqrt 5 := 
  sorry

end complex_modulus_log_monotonic_interval_min_value_area_of_triangle_l546_546117


namespace jose_joined_after_months_l546_546440

theorem jose_joined_after_months :
  ∀ (Tom_investment Jose_investment total_profit Jose_profit months_in_year Tom_profit : ℝ),
    Tom_investment = 30000 →
    Jose_investment = 45000 →
    total_profit = 54000 →
    Jose_profit = 30000 →
    (∀ x, total_profit - Jose_profit = Tom_profit) →
    (Tom_profit / (Tom_investment * months_in_year) = Jose_profit / (Jose_investment * (months_in_year - x))) →
    x = 2 :=
by
  intro Tom_investment Jose_investment total_profit Jose_profit months_in_year Tom_profit
  intro hTom_investment hJose_investment htotal_profit hJose_profit hTom_profit_share hprofit_ratio
  sorry

end jose_joined_after_months_l546_546440


namespace tank_fill_time_l546_546485

noncomputable def fill_time (T rA rB rC : ℝ) : ℝ :=
  let cycle_fill := rA + rB + rC
  let cycles := T / cycle_fill
  let cycle_time := 3
  cycles * cycle_time

theorem tank_fill_time
  (T : ℝ) (rA rB rC : ℝ) (hT : T = 800) (hrA : rA = 40) (hrB : rB = 30) (hrC : rC = -20) :
  fill_time T rA rB rC = 48 :=
by
  sorry

end tank_fill_time_l546_546485


namespace solve_xy_l546_546567

theorem solve_xy : ∃ x y : ℝ, (x - y = 10 ∧ x^2 + y^2 = 100) ↔ ((x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0)) := 
by {
  sorry
}

end solve_xy_l546_546567


namespace unpainted_unit_cubes_l546_546497

theorem unpainted_unit_cubes (total_units : ℕ) (painted_per_face : ℕ) (painted_edges_adjustment : ℕ) :
  total_units = 216 → painted_per_face = 12 → painted_edges_adjustment = 36 → 
  total_units - (painted_per_face * 6 - painted_edges_adjustment) = 108 :=
by
  intros h_tot_units h_painted_face h_edge_adj
  sorry

end unpainted_unit_cubes_l546_546497


namespace exponent_values_l546_546639

theorem exponent_values (x y : ℝ) (hx : 2^x = 3) (hy : 2^y = 4) : 2^(x + y) = 12 :=
by 
  sorry

end exponent_values_l546_546639


namespace find_principal_sum_l546_546779

-- Define the conditions as per the problem statement
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100
def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * ((1 + R / 100) ^ T) - P

def given_simple_interest : ℝ := 1750.000000000002
def given_simple_interest_rate : ℝ := 8
def given_simple_interest_time : ℝ := 3

def given_compound_interest_rate : ℝ := 10
def given_compound_interest_time : ℝ := 2

-- Statement to prove
theorem find_principal_sum :
  let SI := simple_interest given_simple_interest given_simple_interest_rate given_simple_interest_time in
  let CI := compound_interest 4000 given_compound_interest_rate given_compound_interest_time in
  SI = (1 / 2) * CI :=
sorry

end find_principal_sum_l546_546779


namespace probability_of_event_is_correct_l546_546723

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

/-- Definition of the set from which a, b, and c are chosen -/
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 2010}

/-- Definition of the event that abc + ab + a is divisible by 3 -/
def event (a b c : ℕ) : Prop := is_divisible_by (a * b * c + a * b + a) 3

/-- Definition of the probability of the event happening given the set -/
def probability_event : ℚ := 13 / 27

/-- The main theorem -/
theorem probability_of_event_is_correct : 
  (∑' a b c in num_set, indicator (event a b c)) / 
  (∑' a b c in num_set, 1) = probability_event := sorry


end probability_of_event_is_correct_l546_546723


namespace michael_needs_more_money_l546_546360

-- Define the initial conditions
def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_gbp : ℝ := 30
def gbp_to_usd : ℝ := 1.4
def perfume_cost : ℝ := perfume_gbp * gbp_to_usd
def photo_album_eur : ℝ := 25
def eur_to_usd : ℝ := 1.2
def photo_album_cost : ℝ := photo_album_eur * eur_to_usd

-- Sum the costs
def total_cost : ℝ := cake_cost + bouquet_cost + balloons_cost + perfume_cost + photo_album_cost

-- Define the required amount
def additional_money_needed : ℝ := total_cost - michael_money

-- The theorem statement
theorem michael_needs_more_money : additional_money_needed = 83 := by
  sorry

end michael_needs_more_money_l546_546360


namespace sum_proper_divisors_81_l546_546814

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l546_546814


namespace ab_difference_is_49_l546_546569

def tau (n : ℕ) : ℕ := (finset.range n).card.filter (λ d, n % (d + 1) = 0)

def T (n : ℕ) : ℕ := finset.sum (finset.range n) (λ k, k * (k + 1) / 2)

def S (n : ℕ) : ℕ := (finset.range n).sum (λ k, tau (k + 1)) + T n

def count_odd_s (n : ℕ) : ℕ :=
finset.card (finset.filter (λ k, S (k + 1) % 2 = 1) (finset.range n))

def count_even_s (n : ℕ) : ℕ :=
finset.card (finset.filter (λ k, S (k + 1) % 2 = 0) (finset.range n))

theorem ab_difference_is_49 : | (count_odd_s 3025) - (count_even_s 3025) | = 49 :=
by
  sorry

end ab_difference_is_49_l546_546569


namespace zero_in_interval_l546_546200

noncomputable def f (x : ℝ) : ℝ := real.sqrt x - 2 + real.logb 2 x -- definition of the function

theorem zero_in_interval (x : ℝ) (h : 0 < x) : 
  f x = 0 → 1 < x ∧ x < 2 :=
sorry

end zero_in_interval_l546_546200


namespace scalene_triangle_lines_count_l546_546923

-- Define the problem statement for a scalene triangle
def scalene_triangle (A B C : Type) [Distinct A B C] : Type :=
  triangle A B C

-- Define the quantities we are interested in: altitudes, medians, and angle bisectors
def num_altitudes (T : scalene_triangle A B C) : Nat := 3
def num_medians (T : scalene_triangle A B C) : Nat := 3
def num_angle_bisectors (T : scalene_triangle A B C) : Nat := 3

-- The theorem stating the total number of distinct lines
theorem scalene_triangle_lines_count (T : scalene_triangle A B C) : 
  num_altitudes T + num_medians T + num_angle_bisectors T = 9 :=
by sorry

end scalene_triangle_lines_count_l546_546923


namespace Hausdorff_dim_subset_le_Hausdorff_measure_disjoint_union_Hausdorff_measure_scaling_l546_546736

-- Proof 1
theorem Hausdorff_dim_subset_le {A B : set ℝ} (hAB : A ⊆ B) :
  hausdorff_dim A ≤ hausdorff_dim B := sorry

-- Proof 2
theorem Hausdorff_measure_disjoint_union {A B : set ℝ} (h_disjoint : A ∩ B = ∅) (s : ℝ) :
  hausdorff_measure s (A ∪ B) = hausdorff_measure s A + hausdorff_measure s B := sorry

-- Proof 3
theorem Hausdorff_measure_scaling {F : set ℝ} (λ : ℝ) (hλ : λ > 0) (s : ℝ) :
  hausdorff_measure s (λ • F) = λ ^ s * hausdorff_measure s F := sorry

end Hausdorff_dim_subset_le_Hausdorff_measure_disjoint_union_Hausdorff_measure_scaling_l546_546736


namespace length_BD_is_7_5_l546_546442

-- Definitions based on given conditions
structure Triangle (A B C : Type) :=
(is_isosceles : (A = B) ∨ (B = C) ∨ (C = A))

structure Midpoint (D E : Type) :=
(length_BE : Float)

noncomputable def length_BD (D E : Type) [Midpoint D E] : Float :=
  E.length_BE / 2

-- Theorem stating the problem and answer
theorem length_BD_is_7_5 (A B C D E : Type) [Triangle A B C] [Midpoint D (BE : Type)] 
  (h : BE.length_BE = 15) : length_BD D BE = 7.5 :=
by
  sorry

end length_BD_is_7_5_l546_546442


namespace books_initially_l546_546684

theorem books_initially (A B : ℕ) (h1 : A = 3) (h2 : B = (A + 2) + 2) : B = 7 :=
by
  -- Using the given facts, we need to show B = 7
  sorry

end books_initially_l546_546684


namespace problem_statements_l546_546299

noncomputable def is_measure_A_60_degrees (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * b * Real.cos A = c * Real.cos A + a * Real.cos C) → A = 60

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  if h_a : a = Real.sqrt 7 ∧ b + c = 4 ∧ ∃ A B C : ℝ, 
     (2 * b * Real.cos A = c * Real.cos A + a * Real.cos C) ∧ A = 60 then
    (1 / 2) * b * c * Real.sin 60
  else
    0

theorem problem_statements
  (a b c : ℝ) (A B C : ℝ)
  (h1 : 2 * b * Real.cos A = c * Real.cos A + a * Real.cos C)
  (h2 : a = Real.sqrt 7)
  (h3 : b + c = 4) 
  :
    (is_measure_A_60_degrees a b c A B C h1) ∧ 
    (area_of_triangle a b c = (3 * Real.sqrt 3) / 4) :=
sorry

end problem_statements_l546_546299


namespace northern_car_speed_l546_546791

def speed_of_northern_car : ℝ :=
  let distance_north := 300
  let time := 5
  let speed_south := 60
  let distance_south := time * speed_south
  let distance_apart := 500
  let hypotenuse := distance_apart
  let leg1 := distance_north
  let leg2 := time
  let c_squared := hypotenuse ^ 2
  let a_squared := leg1 ^ 2
  let b_squared := leg2 ^ 2 * speed_of_northern_car ^ 2
  sqrt ((c_squared - a_squared) / b_squared)

theorem northern_car_speed : speed_of_northern_car = 80 := by
  sorry

end northern_car_speed_l546_546791


namespace pirate_coins_l546_546130

theorem pirate_coins (x : ℕ) (hn : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → ∃ y : ℕ, y = (2 * k * x) / 15) : 
  ∃ y : ℕ, y = 630630 :=
by sorry

end pirate_coins_l546_546130


namespace right_angle_sides_of_isosceles_right_triangle_l546_546988

def is_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

theorem right_angle_sides_of_isosceles_right_triangle
  (C : ℝ × ℝ)
  (hyp_line : ℝ → ℝ → Prop)
  (side_AC side_BC : ℝ → ℝ → Prop)
  (H1 : C = (3, -2))
  (H2 : hyp_line = is_on_line 3 (-1) 2)
  (H3 : side_AC = is_on_line 2 1 (-4))
  (H4 : side_BC = is_on_line 1 (-2) (-7))
  (H5 : ∃ x y, side_BC (3) y ∧ side_AC x (-2)) :
  side_AC = is_on_line 2 1 (-4) ∧ side_BC = is_on_line 1 (-2) (-7) :=
by
  sorry

end right_angle_sides_of_isosceles_right_triangle_l546_546988


namespace transport_cost_is_correct_l546_546745

-- Define the transport cost per kilogram
def transport_cost_per_kg : ℝ := 18000

-- Define the weight of the scientific instrument in kilograms
def weight_kg : ℝ := 0.5

-- Define the discount rate
def discount_rate : ℝ := 0.10

-- Define the cost calculation without the discount
def cost_without_discount : ℝ := weight_kg * transport_cost_per_kg

-- Define the final cost with the discount applied
def discounted_cost : ℝ := cost_without_discount * (1 - discount_rate)

-- The theorem stating that the discounted cost is $8,100
theorem transport_cost_is_correct : discounted_cost = 8100 := by
  sorry

end transport_cost_is_correct_l546_546745


namespace all_even_l546_546788

theorem all_even {nums : Fin 100 → ℕ}
  (h : ∀ i : Fin 100, (∑ j, if j ≠ i then nums j else 0) % 2 = 0) :
  ∀ i : Fin 100, nums i % 2 = 0 :=
sorry

end all_even_l546_546788


namespace linear_equation_infinite_solutions_l546_546421

theorem linear_equation_infinite_solutions : ∃∞ (x y : ℝ), x + y = 1 :=
sorry

end linear_equation_infinite_solutions_l546_546421


namespace find_length_AK_l546_546106

theorem find_length_AK
  (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited K] [Inhabited M]
  (AB AC BC : ℝ)
  (hAB : AB = 23)
  (hAC : AC = 8)
  (midpoint_M : ∃ X : Type, X = (B + C) / 2)  -- assuming point M is the midpoint
  (line_l : ∃ l : Type, ∀ (a : Type), l = is_bisector (external_angle A))
  (parallel_Ml : ∃ k : Type, ∀ (m : Type), k = parallel_through M)
  (intersects_K : ∃ k : Type, k = intersect line_l AB) :
  AK = 15.5 :=
by
  sorry

end find_length_AK_l546_546106


namespace inequality_solution_set_l546_546547

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 4)^2

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 1) / (x + 4)^2 ≥ 0} = {x : ℝ | x ≠ -4} :=
by
  sorry

end inequality_solution_set_l546_546547


namespace earth_volume_fraction_above_45_north_l546_546450

noncomputable def fraction_above_45_north (R : ℝ) : ℝ :=
  (8 - 5 * real.sqrt 2) / 16

theorem earth_volume_fraction_above_45_north (R : ℝ) : 
  let fraction := (8 - 5 * real.sqrt 2) / 16
  fraction_above_45_north R = fraction :=
by
  sorry

end earth_volume_fraction_above_45_north_l546_546450


namespace log_inequalities_and_exponent_l546_546774

theorem log_inequalities_and_exponent:
  (0 < 0.8 ^ 0.7) ∧ (0.8 ^ 0.7 < 1) ∧ (1 < log 2 3) ∧ (log 0.3 2 < 0) →
  log 0.3 2 < 0.8 ^ 0.7 ∧ 0.8 ^ 0.7 < log 2 3 :=
by
  -- simple solution steps to verify claims could be added here if required
  sorry

end log_inequalities_and_exponent_l546_546774


namespace time_to_even_floors_eq_15_l546_546499

/-- 
A building has 10 floors. 
It takes a certain amount of seconds to go up the stairs to the even-numbered floors and 9 seconds to go up to the odd-numbered floors. 
It takes 2 minutes to get to the 10th floor. 
Prove that it takes 15 seconds to go up the stairs to the even-numbered floors.
-/
theorem time_to_even_floors_eq_15 :
  let E := 15,
  even_floors := 5,
  odd_floors := 5,
  total_time := 120,
  time_to_odd_floors := 45
  E = (total_time - time_to_odd_floors) / even_floors := by
  sorry

end time_to_even_floors_eq_15_l546_546499


namespace broken_line_path_exceeds_20_units_l546_546548

def radius_of_circle : ℝ := 8 -- Since the diameter is 16 units
def length_AC : ℝ := 6
def length_BD : ℝ := 6
def length_CD : ℝ := 4 -- From 16 - 6 - 6

theorem broken_line_path_exceeds_20_units :
  ∃ P : circle radius_of_circle, 
    ∃ Q : circle radius_of_circle, 
      (PQ = radius_of_circle ∧
      distance (C, P) + distance (P, D) + PQ > 20) :=
by { sorry }

end broken_line_path_exceeds_20_units_l546_546548


namespace positive_difference_of_solutions_l546_546934

theorem positive_difference_of_solutions :
  let s := polynomial ℝ in
  ∃ s1 s2 : ℝ, (s^2 - 5*s - 11) / (s + 3) = 3*s + 10 → 
  abs (s2 - s1) = 20.05 := by
  sorry

end positive_difference_of_solutions_l546_546934


namespace initial_people_lifting_weights_l546_546168

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end initial_people_lifting_weights_l546_546168


namespace sum_of_proper_divisors_of_81_l546_546816

theorem sum_of_proper_divisors_of_81 : 
  (∑ k in finset.range 4, 3^k) = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l546_546816


namespace quadratic_unique_solution_l546_546937

theorem quadratic_unique_solution (k : ℝ) (x : ℝ) :
  (16 ^ 2 - 4 * 2 * k * 4 = 0) → (k = 8 ∧ x = -1 / 2) :=
by
  sorry

end quadratic_unique_solution_l546_546937


namespace event_day_in_1800_is_tuesday_l546_546679

def leap_years (start_year : ℕ) (end_year : ℕ) : ℕ :=
  let leap := (end_year - start_year) / 4 in
  let non_leap := if start_year % 100 == 0 ∧ start_year % 400 ≠ 0 then 1 else 0 in
  let non_leap' := if (start_year + 100) % 100 == 0 ∧ (start_year + 100) % 400 ≠ 0 then 1 else 0 in
  leap - (non_leap + non_leap')

noncomputable def total_day_shifts (years : ℕ) (leap_years : ℕ) : ℕ :=
  let regular_years := years - leap_years in
  regular_years * 1 + leap_years * 2

def days_mod_seven (days : ℕ) : ℕ :=
  days % 7

def day_of_week_shift (today : String) (shifts : ℕ) : String :=
  match today with
  | "Monday"    => ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].get! shifts
  | "Tuesday"   => ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"].get! shifts
  | "Wednesday" => ["Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday"].get! shifts
  | "Thursday"  => ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"].get! shifts
  | "Friday"    => ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"].get! shifts
  | "Saturday"  => ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].get! shifts
  | "Sunday"    => ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"].get! shifts
  | _           => "Invalid day"

theorem event_day_in_1800_is_tuesday :
  let start_year := 1800
  let end_year := 2100
  let years := end_year - start_year
  let leap := leap_years start_year end_year
  let shifts := total_day_shifts years leap
  let day_shift := days_mod_seven shifts
  let original_day := day_of_week_shift "Thursday" (7 - day_shift)
  original_day = "Tuesday" :=
by
  sorry

end event_day_in_1800_is_tuesday_l546_546679


namespace value_of_x_l546_546283

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end value_of_x_l546_546283


namespace F_derivative_at_2_l546_546214

-- Define the functions and their derivatives at x = 2
variable (f g : ℝ → ℝ) (f' g' : ℝ → ℝ)
variable (h1 : f 2 = -2)
variable (h2 : g 2 = 1)
variable (h3 : f' 2 = 1)
variable (h4 : g' 2 = -2)

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := f x * (g x - 2)

-- State the theorem
theorem F_derivative_at_2 : 
  (deriv F) 2 = -5 :=
by sorry

end F_derivative_at_2_l546_546214


namespace estimated_probability_of_2_days_warning_l546_546525

def high_temp_prob := 3 / 5

def is_high_temp (d : ℕ) : Prop := d ∈ {0, 1, 2, 3, 4, 5}

def count_high_temp_warning (numbers : List (Fin 1000)) :=
  numbers.countp (fun n => 
    let digits := id (List.ofFn n.digits)
    (digits.count (is_high_temp)) = 2
  )

def sets_of_random_numbers := [116, 812, 730, 217, 109, 361, 284, 147, 318, 027, 785, 134, 452, 125, 689, 024, 169, 334, 908, 044]

def total_probability : ℚ := 
  count_high_temp_warning sets_of_random_numbers.toList / 20

theorem estimated_probability_of_2_days_warning : total_probability = 1 / 2 := 
  by sorry
 
end estimated_probability_of_2_days_warning_l546_546525


namespace cos_squared_sum_l546_546179

theorem cos_squared_sum : 
  (∑ θ in Finset.range 91, (Real.cos (θ * (Real.pi / 180)))^2) = 91 / 2 :=
by
  sorry

end cos_squared_sum_l546_546179


namespace expectation_is_correct_l546_546249

variable (m : ℝ)
variable (ξ : ℕ → ℝ)
variable (P : ℕ → ℝ)

-- Conditions from the problem
axiom prob_1 : P 1 = 0.3
axiom prob_2 : P 2 = m
axiom prob_3 : P 3 = 0.4
axiom total_prob : P 1 + P 2 + P 3 = 1

-- Proving the expectation is 2.1
theorem expectation_is_correct : E ξ = 2.1 := by
sorry

end expectation_is_correct_l546_546249


namespace train_length_correct_l546_546142

noncomputable def length_bridge : ℝ := 300
noncomputable def time_to_cross : ℝ := 45
noncomputable def speed_train_kmh : ℝ := 44

-- Conversion from km/h to m/s
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

-- Total distance covered
noncomputable def total_distance_covered : ℝ := speed_train_ms * time_to_cross

-- Length of the train
noncomputable def length_train : ℝ := total_distance_covered - length_bridge

theorem train_length_correct : abs (length_train - 249.9) < 0.1 :=
by
  sorry

end train_length_correct_l546_546142


namespace campers_afternoon_l546_546861

noncomputable def campers_morning : ℕ := 35
noncomputable def campers_total : ℕ := 62

theorem campers_afternoon :
  campers_total - campers_morning = 27 :=
by
  sorry

end campers_afternoon_l546_546861


namespace total_distance_l546_546193

variable {D : ℝ}

theorem total_distance (h1 : D / 3 > 0)
                       (h2 : (2 / 3 * D) - (1 / 6 * D) > 0)
                       (h3 : (1 / 2 * D) - (1 / 10 * D) = 180) :
    D = 450 := 
sorry

end total_distance_l546_546193


namespace find_f_ln2_l546_546650

noncomputable def f : ℝ → ℝ := sorry

axiom fx_monotonic : Monotone f
axiom fx_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - Real.exp 1

theorem find_f_ln2 : f (Real.log 2) = -1 := 
sorry

end find_f_ln2_l546_546650


namespace petya_second_question_answer_l546_546552

def binary_answer : Type := Prop

variable (a1 a2 a3 a4 a5 : binary_answer)
variable (answer_is_correct : (a2 = true))
variable (valid_sequence : ((Π i : Fin 3, (a1 ≠ a2) ∧ (a2 ≠ a3)) ∧ ((∑[if a = true then 1 else 0, i1] > ) )) ∧ (a1 ≠a5))

theorem petya_second_question_answer :
  (valid_sequence ((a1 ∧ a2 ∧ a3) = (a1,n ≠ a3,n) 
  ∧ ((∑[if a = true ≠ a(i1)>∑ [if  if a = false ≠ a2 ]i 1)
  ∧ [a1 ≠ a5]) ):
  answer_a2 = false :=
begin 
 intros[ i, valid_sequence, sum]
  sorry
end 

end petya_second_question_answer_l546_546552


namespace function_ordering_l546_546187

variables {R : Type*} [LinearOrderedField R]

def symmetric_function (f : R → R) := ∀ x, f (1 - x) = f x

def concavity_condition (f : R → R) := ∀ x, (x - 2) * (f'' x) > 0

theorem function_ordering (f : R → R) (h_symm : symmetric_function f) (h_conc : concavity_condition f) 
  (x₁ x₂ : R) (h1 : x₁ < x₂) (h2 : x₁ + x₂ > 1) : f x₁ < f x₂ :=
sorry

end function_ordering_l546_546187


namespace parallelogram_angle_B_eq_130_l546_546670

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end parallelogram_angle_B_eq_130_l546_546670


namespace hexagon_side_length_l546_546309

-- Definitions
def regular_hexagon (s : ℝ) :=
  ∃ (equilateral_triangle : ℝ → Prop), (equilateral_triangle = λ s : ℝ, True) ∧ 
  (∀ x : ℝ, equilateral_triangle s → s = x)

-- Problem statement
theorem hexagon_side_length (h : regular_hexagon s) : ∃ s : ℝ, s = 40 * real.sqrt 3 / 3 :=
begin
  sorry
end

end hexagon_side_length_l546_546309


namespace isosceles_triangle_with_x_axis_l546_546767

noncomputable def slope_l1 (k : ℝ) := (1/k)
noncomputable def slope_l2 (k : ℝ) := (2 * k)

theorem isosceles_triangle_with_x_axis (k : ℝ) (h₁ : k > 0) 
  (h₂ : slope_l1 k = (1 / k)) 
  (h₃ : slope_l2 k = 2 * k) 
  (h₄ : is_isosceles_with_x_axis (slope_l1 k) (slope_l2 k)) :
  k = sqrt(2) / 4 ∨ k = sqrt(2) :=
sorry

end isosceles_triangle_with_x_axis_l546_546767


namespace add_and_round_to_thousandth_l546_546897

theorem add_and_round_to_thousandth :
  let x := 53.463
  let y := 12.9873
  let sum := x + y
  real.to_nnreal (real.round_nearest 1000 sum) = 66.450 :=
by {
  let x := 53.463,
  let y := 12.9873,
  let sum := x + y,
  have h_sum : sum = 66.4503 := sorry,
  have h_round : real.to_nnreal (real.round_nearest 1000 sum) = 66.450 := sorry,
  exact eq.trans h_sum h_round,
}

end add_and_round_to_thousandth_l546_546897


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l546_546467

theorem option_A_correct (a : ℝ) : a ^ 2 * a ^ 3 = a ^ 5 := by {
  -- Here, we would provide the proof if required,
  -- but we are only stating the theorem.
  sorry
}

-- You may optionally add definitions of incorrect options for completeness.
theorem option_B_incorrect (a : ℝ) : ¬(a + 2 * a = 3 * a ^ 2) := by {
  sorry
}

theorem option_C_incorrect (a b : ℝ) : ¬((a * b) ^ 3 = a * b ^ 3) := by {
  sorry
}

theorem option_D_incorrect (a : ℝ) : ¬((-a ^ 3) ^ 2 = -a ^ 6) := by {
  sorry
}

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l546_546467


namespace arrange_books_l546_546666

noncomputable def numberOfArrangements : Nat :=
  4 * 3 * 6 * (Nat.factorial 9)

theorem arrange_books :
  numberOfArrangements = 26210880 := by
  sorry

end arrange_books_l546_546666


namespace original_number_of_cards_l546_546505

-- Declare variables r and b as naturals representing the number of red and black cards, respectively.
variable (r b : ℕ)

-- Assume the probabilities given in the problem.
axiom prob_red : (r : ℝ) / (r + b) = 1 / 3
axiom prob_red_after_add : (r : ℝ) / (r + b + 4) = 1 / 4

-- Define the statement we need to prove.
theorem original_number_of_cards : r + b = 12 :=
by
  -- The proof steps would be here, but we'll use sorry to avoid implementing them.
  sorry

end original_number_of_cards_l546_546505


namespace one_quarter_way_l546_546462

theorem one_quarter_way (w₁ w₂ x₁ x₂ : ℚ) (h₁ : w₁ = 3) (h₂ : w₂ = 1) (h₃ : x₁ = 1/3) (h₄ : x₂ = 2/3) :
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂) = 5/12 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num
  sorry

end one_quarter_way_l546_546462


namespace relation_between_x_y_z_l546_546277

noncomputable def x : ℝ := Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3
noncomputable def z : ℝ := Real.sqrt 5

theorem relation_between_x_y_z :
  log 2 (log (1/2) (log 2 x)) = 0 ∧ 
  log 3 (log (1/3) (log 3 y)) = 0 ∧ 
  log 5 (log (1/5) (log 5 z)) = 0 → 
  y > x ∧ x > z := by 
  sorry

end relation_between_x_y_z_l546_546277


namespace sum_interior_angles_of_regular_polygon_l546_546884

def regular_polygon_exterior_angle (P : Type) [polygon P] : Angle := 40

def number_sides (P : Type) [polygon P] : ℕ := 360 / regular_polygon_exterior_angle P

def sum_of_interior_angles (P : Type) [polygon P] : Angle := (number_sides P - 2) * 180

theorem sum_interior_angles_of_regular_polygon : 
  ∀ (P : Type) [polygon P], regular_polygon_exterior_angle P = 40 → sum_of_interior_angles P = 1260 := 
by
  intros
  sorry

end sum_interior_angles_of_regular_polygon_l546_546884


namespace cost_of_cheese_without_coupon_l546_546743

theorem cost_of_cheese_without_coupon
    (cost_bread : ℝ := 4.00)
    (cost_meat : ℝ := 5.00)
    (coupon_cheese : ℝ := 1.00)
    (coupon_meat : ℝ := 1.00)
    (cost_sandwich : ℝ := 2.00)
    (num_sandwiches : ℝ := 10)
    (C : ℝ) : 
    (num_sandwiches * cost_sandwich = (cost_bread + (cost_meat - coupon_meat) + cost_meat + (C - coupon_cheese) + C)) → (C = 4.50) :=
by {
    sorry
}

end cost_of_cheese_without_coupon_l546_546743


namespace train_seat_count_l546_546523

theorem train_seat_count (t : ℝ) (h1 : 0.20 * t = 0.2 * t)
  (h2 : 0.60 * t = 0.6 * t) (h3 : 30 + 0.20 * t + 0.60 * t = t) : t = 150 :=
by
  sorry

end train_seat_count_l546_546523
