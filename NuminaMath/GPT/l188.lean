import Mathlib

namespace solve_remainder_l188_188825

theorem solve_remainder (y : ℤ) 
  (hc1 : y + 4 ≡ 9 [ZMOD 3^3])
  (hc2 : y + 4 ≡ 16 [ZMOD 5^3])
  (hc3 : y + 4 ≡ 36 [ZMOD 7^3]) : 
  y ≡ 32 [ZMOD 105] :=
by
  sorry

end solve_remainder_l188_188825


namespace john_weekly_earnings_l188_188191

theorem john_weekly_earnings :
  (4 * 4 * 10 = 160) :=
by
  -- Proposition: John makes $160 a week from streaming
  -- Condition 1: John streams for 4 days a week
  let days_of_streaming := 4
  -- Condition 2: He streams 4 hours each day.
  let hours_per_day := 4
  -- Condition 3: He makes $10 an hour.
  let earnings_per_hour := 10

  -- Now, calculate the weekly earnings
  -- Weekly earnings = 4 days/week * 4 hours/day * $10/hour
  have weekly_earnings : days_of_streaming * hours_per_day * earnings_per_hour = 160 := sorry
  exact weekly_earnings


end john_weekly_earnings_l188_188191


namespace binom_20_19_eq_20_l188_188730

theorem binom_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  -- use the property of binomial coefficients
  have h : Nat.choose 20 19 = Nat.choose 20 1 := Nat.choose_symm 20 19
  -- now, apply the fact that Nat.choose 20 1 = 20
  rw h
  exact Nat.choose_one 20

end binom_20_19_eq_20_l188_188730


namespace krystiana_earnings_l188_188193

def earning_building1_first_floor : ℝ := 5 * 15 * 0.8
def earning_building1_second_floor : ℝ := 6 * 25 * 0.75
def earning_building1_third_floor : ℝ := 9 * 30 * 0.5
def earning_building1_fourth_floor : ℝ := 4 * 60 * 0.85
def earnings_building1 : ℝ := earning_building1_first_floor + earning_building1_second_floor + earning_building1_third_floor + earning_building1_fourth_floor

def earning_building2_first_floor : ℝ := 7 * 20 * 0.9
def earning_building2_second_floor : ℝ := (25 + 30 + 35 + 40 + 45 + 50 + 55 + 60) * 0.7
def earning_building2_third_floor : ℝ := 6 * 60 * 0.6
def earnings_building2 : ℝ := earning_building2_first_floor + earning_building2_second_floor + earning_building2_third_floor

def total_earnings : ℝ := earnings_building1 + earnings_building2

theorem krystiana_earnings : total_earnings = 1091.5 := by
  sorry

end krystiana_earnings_l188_188193


namespace find_a_l188_188348

open ProbabilityTheory

noncomputable def normalDist (mean variance : ℝ) : Measure ℝ := {
  toMeasure := sorry, -- Definition of the normal distribution measure
}

theorem find_a (ξ : ℝ → probability_theory.Pmf ℝ)
  (h1 : ξ = normalDist 3 4)
  (h2 : ∀ a : ℝ, P (ξ < 2 * a - 2) = P (ξ > a + 2)) :
  a = 2 :=
begin
  sorry
end

end find_a_l188_188348


namespace arithmetic_sequence_properties_l188_188608

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 + a 3 = 21) 
  (h2 : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n, a n = -4 * n + 15 ∨ a n = 4 * n - 1) := 
by
  sorry

end arithmetic_sequence_properties_l188_188608


namespace exists_eleven_consecutive_numbers_sum_cube_l188_188551

theorem exists_eleven_consecutive_numbers_sum_cube :
  ∃ (n k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) + (n+9) + (n+10)) = k^3 :=
by
  sorry

end exists_eleven_consecutive_numbers_sum_cube_l188_188551


namespace dogs_not_eating_either_l188_188605

variable (U : Finset α) (A B : Finset α)
variable (hU : U.card = 75) (hA : A.card = 18) (hB : B.card = 55) (hAB : (A ∩ B).card = 10)

theorem dogs_not_eating_either (U A B : Finset α) (hU : U.card = 75) (hA : A.card = 18) (hB : B.card = 55) (hAB : (A ∩ B).card = 10) :
  (U.card - (A ∪ B).card) = 12 :=
by
  --Proof goes here
  sorry

end dogs_not_eating_either_l188_188605


namespace relationship_between_a_b_l188_188759

theorem relationship_between_a_b (a b c : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -2)
  (h3 : a * x + c * y = 1) (h4 : c * x - b * y = 2) : 9 * a + 4 * b = 1 :=
sorry

end relationship_between_a_b_l188_188759


namespace average_of_all_results_is_24_l188_188858

-- Definitions translated from conditions
def average_1 := 20
def average_2 := 30
def n1 := 30
def n2 := 20
def total_sum_1 := n1 * average_1
def total_sum_2 := n2 * average_2

-- Lean 4 statement
theorem average_of_all_results_is_24
  (h1 : total_sum_1 = n1 * average_1)
  (h2 : total_sum_2 = n2 * average_2) :
  ((total_sum_1 + total_sum_2) / (n1 + n2) = 24) :=
by
  sorry

end average_of_all_results_is_24_l188_188858


namespace least_clock_equiv_square_l188_188032

def is_clock_equiv (a b : ℕ) (m : ℕ) : Prop :=
  (a - b) % m = 0

def find_least_clock_equiv_greater_than_five : ℕ :=
  @WellFounded.fix _ _ ⟨5, sorry⟩ ⟨_, sorry⟩

theorem least_clock_equiv_square (n : ℕ) (h : n > 5) :
  n = find_least_clock_equiv_greater_than_five
→ is_clock_equiv ((find_least_clock_equiv_greater_than_five)^2) (find_least_clock_equiv_greater_than_five) 12 :=
by
  intro hn
  intro hclock
  sorry

example : find_least_clock_equiv_greater_than_five = 9 :=
by 
  sorry

end least_clock_equiv_square_l188_188032


namespace find_a_and_b_find_set_A_l188_188895

noncomputable def f (x a b : ℝ) := 4 ^ x - a * 2 ^ x + b

theorem find_a_and_b (a b : ℝ)
  (h₁ : f 1 a b = -1)
  (h₂ : ∀ x, ∃ t > 0, f x a b = t ^ 2 - a * t + b) :
  a = 4 ∧ b = 3 :=
sorry

theorem find_set_A (a b : ℝ)
  (ha : a = 4) (hb : b = 3) :
  {x : ℝ | f x a b ≤ 35} = {x : ℝ | x ≤ 3} :=
sorry

end find_a_and_b_find_set_A_l188_188895


namespace third_set_candies_l188_188662

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l188_188662


namespace number_of_2_face_painted_cubes_l188_188982

-- Condition definitions based on the problem statement
def painted_faces (n : ℕ) (type : String) : ℕ :=
  if type = "corner" then 8
  else if type = "edge" then 12
  else if type = "face" then 24
  else if type = "inner" then 9
  else 0

-- The mathematical proof statement
theorem number_of_2_face_painted_cubes : painted_faces 27 "edge" = 12 :=
by
  sorry

end number_of_2_face_painted_cubes_l188_188982


namespace expand_expression_l188_188886

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 :=
by 
  sorry

end expand_expression_l188_188886


namespace prob_selecting_green_ball_l188_188883

-- Definition of the number of red and green balls in each container
def containerI_red := 10
def containerI_green := 5
def containerII_red := 3
def containerII_green := 5
def containerIII_red := 2
def containerIII_green := 6
def containerIV_red := 4
def containerIV_green := 4

-- Total number of balls in each container
def total_balls_I := containerI_red + containerI_green
def total_balls_II := containerII_red + containerII_green
def total_balls_III := containerIII_red + containerIII_green
def total_balls_IV := containerIV_red + containerIV_green

-- Probability of selecting a green ball from each container
def prob_green_I := containerI_green / total_balls_I
def prob_green_II := containerII_green / total_balls_II
def prob_green_III := containerIII_green / total_balls_III
def prob_green_IV := containerIV_green / total_balls_IV

-- Probability of selecting any one container
def prob_select_container := (1:ℚ) / 4

-- Combined probability for a green ball from each container
def combined_prob_I := prob_select_container * prob_green_I 
def combined_prob_II := prob_select_container * prob_green_II 
def combined_prob_III := prob_select_container * prob_green_III 
def combined_prob_IV := prob_select_container * prob_green_IV 

-- Total probability of selecting a green ball
def total_prob_green := combined_prob_I + combined_prob_II + combined_prob_III + combined_prob_IV 

-- Theorem to prove
theorem prob_selecting_green_ball : total_prob_green = 53 / 96 :=
by sorry

end prob_selecting_green_ball_l188_188883


namespace maxwell_distance_l188_188351

-- Define the given conditions
def distance_between_homes : ℝ := 65
def maxwell_speed : ℝ := 2
def brad_speed : ℝ := 3

-- The statement we need to prove
theorem maxwell_distance :
  ∃ (x t : ℝ), 
    x = maxwell_speed * t ∧
    distance_between_homes - x = brad_speed * t ∧
    x = 26 := by sorry

end maxwell_distance_l188_188351


namespace cos_plus_sin_eq_sqrt_five_over_two_l188_188567

theorem cos_plus_sin_eq_sqrt_five_over_two (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
(h : sin α * cos α = 1 / 8) : 
cos α + sin α = real.sqrt (5) / 2 :=
by
  sorry

end cos_plus_sin_eq_sqrt_five_over_two_l188_188567


namespace abs_inequality_solution_l188_188060

theorem abs_inequality_solution (x : ℝ) : (|2 * x - 1| - |x - 2| < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end abs_inequality_solution_l188_188060


namespace volume_of_rectangular_prism_l188_188221

theorem volume_of_rectangular_prism (x y z : ℝ) 
  (h1 : x * y = 30) 
  (h2 : x * z = 45) 
  (h3 : y * z = 75) : 
  x * y * z = 150 :=
sorry

end volume_of_rectangular_prism_l188_188221


namespace gcd_1755_1242_l188_188381

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := 
by
  sorry

end gcd_1755_1242_l188_188381


namespace john_total_animals_is_114_l188_188795

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end john_total_animals_is_114_l188_188795


namespace celine_change_l188_188533

theorem celine_change :
  let laptop_price := 600
  let smartphone_price := 400
  let tablet_price := 250
  let headphone_price := 100
  let laptops_purchased := 2
  let smartphones_purchased := 4
  let tablets_purchased := 3
  let headphones_purchased := 5
  let discount_rate := 0.10
  let sales_tax_rate := 0.05
  let initial_amount := 5000
  let laptop_total := laptops_purchased * laptop_price
  let smartphone_total := smartphones_purchased * smartphone_price
  let tablet_total := tablets_purchased * tablet_price
  let headphone_total := headphones_purchased * headphone_price
  let discount := discount_rate * (laptop_total + tablet_total)
  let total_before_discount := laptop_total + smartphone_total + tablet_total + headphone_total
  let total_after_discount := total_before_discount - discount
  let sales_tax := sales_tax_rate * total_after_discount
  let final_price := total_after_discount + sales_tax
  let change := initial_amount - final_price
  change = 952.25 :=
  sorry

end celine_change_l188_188533


namespace first_course_cost_l188_188415

theorem first_course_cost (x : ℝ) (h1 : 60 - (x + (x + 5) + 0.25 * (x + 5)) = 20) : x = 15 :=
by sorry

end first_course_cost_l188_188415


namespace closely_related_interval_unique_l188_188583

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

theorem closely_related_interval_unique :
  closely_related f g 2 3 :=
sorry

end closely_related_interval_unique_l188_188583


namespace arithmetic_sequence_number_of_terms_l188_188012

theorem arithmetic_sequence_number_of_terms 
  (a d : ℝ) (n : ℕ) 
  (h1 : a + (a + d) + (a + 2 * d) = 34) 
  (h2 : (a + (n-3) * d) + (a + (n-2) * d) + (a + (n-1) * d) = 146) 
  (h3 : (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 390) : 
  n = 13 :=
by 
  sorry

end arithmetic_sequence_number_of_terms_l188_188012


namespace kayla_apples_correct_l188_188620

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end kayla_apples_correct_l188_188620


namespace car_distance_after_y_begins_l188_188717

theorem car_distance_after_y_begins (v_x v_y : ℝ) (t_y_start t_x_after_y : ℝ) (d_x_before_y : ℝ) :
  v_x = 35 → v_y = 50 → t_y_start = 1.2 → d_x_before_y = v_x * t_y_start → t_x_after_y = 2.8 →
  (d_x_before_y + v_x * t_x_after_y = 98) :=
by
  intros h_vx h_vy h_ty_start h_dxbefore h_txafter
  simp [h_vx, h_vy, h_ty_start, h_dxbefore, h_txafter]
  sorry

end car_distance_after_y_begins_l188_188717


namespace bob_equals_alice_l188_188471

-- Define conditions as constants
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25

-- Bob's total calculation
def bob_total : ℝ := (original_price * (1 + tax_rate)) * (1 - discount_rate)

-- Alice's total calculation
def alice_total : ℝ := (original_price * (1 - discount_rate)) * (1 + tax_rate)

-- Theorem statement to be proved
theorem bob_equals_alice : bob_total = alice_total := by sorry

end bob_equals_alice_l188_188471


namespace diagonals_in_convex_polygon_l188_188002

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end diagonals_in_convex_polygon_l188_188002


namespace problem1_problem2_l188_188543

-- Problem 1: Prove that (-11) + 8 + (-4) = -7
theorem problem1 : (-11) + 8 + (-4) = -7 := by
  sorry

-- Problem 2: Prove that -1^2023 - |1 - 1/3| * (-3/2)^2 = -(5/2)
theorem problem2 : (-1 : ℚ)^2023 - abs (1 - 1/3) * (-3/2)^2 = -(5/2) := by
  sorry

end problem1_problem2_l188_188543


namespace product_binary1101_ternary202_eq_260_l188_188286

-- Define the binary number 1101 in base 10
def binary1101 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 202 in base 10
def ternary202 := 2 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Prove that their product in base 10 is 260
theorem product_binary1101_ternary202_eq_260 : binary1101 * ternary202 = 260 := by
  -- Proof 
  sorry

end product_binary1101_ternary202_eq_260_l188_188286


namespace range_of_x_minus_2y_l188_188757

theorem range_of_x_minus_2y (x y : ℝ) (h₁ : -1 ≤ x) (h₂ : x < 2) (h₃ : 0 < y) (h₄ : y ≤ 1) :
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 :=
sorry

end range_of_x_minus_2y_l188_188757


namespace range_of_f_l188_188905

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x ^ 2 else Real.cos x

theorem range_of_f : Set.range f = Set.Ici (-1) := 
by
  sorry

end range_of_f_l188_188905


namespace integral_x_squared_l188_188968

theorem integral_x_squared:
  ∫ x in (0:ℝ)..(1:ℝ), x^2 = 1/3 :=
by
  sorry

end integral_x_squared_l188_188968


namespace biloca_path_proof_l188_188890

def diagonal_length := 5 -- Length of one diagonal as deduced from Pipoca's path
def tile_width := 3 -- Width of one tile as deduced from Tonica's path
def tile_length := 4 -- Length of one tile as deduced from Cotinha's path

def Biloca_path_length : ℝ :=
  3 * diagonal_length + 4 * tile_width + 2 * tile_length

theorem biloca_path_proof :
  Biloca_path_length = 43 :=
by
  sorry

end biloca_path_proof_l188_188890


namespace people_behind_yuna_l188_188981

theorem people_behind_yuna (total_people : ℕ) (people_in_front : ℕ) (yuna : ℕ)
  (h1 : total_people = 7) (h2 : people_in_front = 2) (h3 : yuna = 1) :
  total_people - people_in_front - yuna = 4 :=
by
  sorry

end people_behind_yuna_l188_188981


namespace jade_transactions_l188_188392

theorem jade_transactions :
  ∀ (transactions_mabel transactions_anthony transactions_cal transactions_jade : ℕ),
    transactions_mabel = 90 →
    transactions_anthony = transactions_mabel + transactions_mabel / 10 →
    transactions_cal = (transactions_anthony * 2) / 3 →
    transactions_jade = transactions_cal + 19 →
    transactions_jade = 85 :=
by
  intros transactions_mabel transactions_anthony transactions_cal transactions_jade
  intros h_mabel h_anthony h_cal h_jade
  sorry

end jade_transactions_l188_188392


namespace evan_books_l188_188740

theorem evan_books (B M : ℕ) (h1 : B = 200 - 40) (h2 : M * B + 60 = 860) : M = 5 :=
by {
  sorry  -- proof is omitted as per instructions
}

end evan_books_l188_188740


namespace min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l188_188455

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Prove the minimum value for a = 1 and x in [-1, 0]
theorem min_f_a_eq_1 : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f 1 x ≥ 5 :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when a ≤ -1
theorem min_f_a_le_neg1 (h : ∀ a : ℝ, a ≤ -1) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a (-1) ≤ f a x :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when -1 < a < 0
theorem min_f_neg1_lt_a_lt_0 (h : ∀ a : ℝ, -1 < a ∧ a < 0) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a a ≤ f a x :=
by
  sorry

end min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l188_188455


namespace oil_ratio_l188_188398

theorem oil_ratio (x : ℝ) (initial_small_tank : ℝ) (initial_large_tank : ℝ) (total_capacity_large : ℝ)
  (half_capacity_large : ℝ) (additional_needed : ℝ) :
  initial_small_tank = 4000 ∧ initial_large_tank = 3000 ∧ total_capacity_large = 20000 ∧
  half_capacity_large = total_capacity_large / 2 ∧ additional_needed = 4000 ∧
  (initial_large_tank + x + additional_needed = half_capacity_large) →
  x / initial_small_tank = 3 / 4 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end oil_ratio_l188_188398


namespace five_n_plus_3_composite_l188_188155

theorem five_n_plus_3_composite (n : ℕ)
  (h1 : ∃ k : ℤ, 2 * n + 1 = k^2)
  (h2 : ∃ m : ℤ, 3 * n + 1 = m^2) :
  ¬ Prime (5 * n + 3) :=
by
  sorry

end five_n_plus_3_composite_l188_188155


namespace letters_identity_l188_188210

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l188_188210


namespace obtuse_angle_at_515_l188_188242

-- Definitions derived from conditions
def minuteHandDegrees (minute: ℕ) : ℝ := minute * 6.0
def hourHandDegrees (hour: ℕ) (minute: ℕ) : ℝ := hour * 30.0 + (minute * 0.5)

-- Main statement to be proved
theorem obtuse_angle_at_515 : 
  let hour := 5
  let minute := 15
  let minute_pos := minuteHandDegrees minute
  let hour_pos := hourHandDegrees hour minute
  let angle := abs (minute_pos - hour_pos)
  angle = 67.5 :=
by
  sorry

end obtuse_angle_at_515_l188_188242


namespace wicket_keeper_age_l188_188364

/-- The cricket team consists of 11 members with an average age of 22 years.
    One member is 25 years old, and the wicket keeper is W years old.
    Excluding the 25-year-old and the wicket keeper, the average age of the remaining players is 21 years.
    Prove that the wicket keeper is 6 years older than the average age of the team. -/
theorem wicket_keeper_age (W : ℕ) (team_avg_age : ℕ := 22) (total_team_members : ℕ := 11) 
                          (other_member_age : ℕ := 25) (remaining_avg_age : ℕ := 21) :
    W = 28 → W - team_avg_age = 6 :=
by
  intros
  sorry

end wicket_keeper_age_l188_188364


namespace parabola_vertex_coordinates_l188_188637

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = -(x - 1) ^ 2 + 3 → (1, 3) = (1, 3) :=
by
  intros x y h
  sorry

end parabola_vertex_coordinates_l188_188637


namespace Hillary_reading_time_on_sunday_l188_188316

-- Define the assigned reading times for both books
def assigned_time_book_a : ℕ := 60 -- minutes
def assigned_time_book_b : ℕ := 45 -- minutes

-- Define the reading times already spent on each book
def time_spent_friday_book_a : ℕ := 16 -- minutes
def time_spent_saturday_book_a : ℕ := 28 -- minutes
def time_spent_saturday_book_b : ℕ := 15 -- minutes

-- Calculate the total time already read for each book
def total_time_read_book_a : ℕ := time_spent_friday_book_a + time_spent_saturday_book_a
def total_time_read_book_b : ℕ := time_spent_saturday_book_b

-- Calculate the remaining time needed for each book
def remaining_time_book_a : ℕ := assigned_time_book_a - total_time_read_book_a
def remaining_time_book_b : ℕ := assigned_time_book_b - total_time_read_book_b

-- Calculate the total remaining time and the equal time division
def total_remaining_time : ℕ := remaining_time_book_a + remaining_time_book_b
def equal_time_division : ℕ := total_remaining_time / 2

-- Theorem statement to prove Hillary's reading time for each book on Sunday
theorem Hillary_reading_time_on_sunday : equal_time_division = 23 := by
  sorry

end Hillary_reading_time_on_sunday_l188_188316


namespace elizabeth_needs_to_borrow_more_money_l188_188428

-- Define the costs of the items
def pencil_cost : ℝ := 6.00 
def notebook_cost : ℝ := 3.50 
def pen_cost : ℝ := 2.25 

-- Define the amount of money Elizabeth initially has and what she borrowed
def elizabeth_money : ℝ := 5.00 
def borrowed_money : ℝ := 0.53 

-- Define the total cost of the items
def total_cost : ℝ := pencil_cost + notebook_cost + pen_cost

-- Define the total amount of money Elizabeth has
def total_money : ℝ := elizabeth_money + borrowed_money

-- Define the additional amount Elizabeth needs to borrow
def amount_needed_to_borrow : ℝ := total_cost - total_money

-- The theorem to prove that Elizabeth needs to borrow an additional $6.22
theorem elizabeth_needs_to_borrow_more_money : 
  amount_needed_to_borrow = 6.22 := by 
    -- Proof goes here
    sorry

end elizabeth_needs_to_borrow_more_money_l188_188428


namespace jenny_questions_wrong_l188_188617

variable (j k l m : ℕ)

theorem jenny_questions_wrong
  (h1 : j + k = l + m)
  (h2 : j + m = k + l + 6)
  (h3 : l = 7) : j = 10 := by
  sorry

end jenny_questions_wrong_l188_188617


namespace number_of_valid_starting_lineups_l188_188958

-- Define the total number of players
def total_players : ℕ := 15

-- Define the specific players Leo, Max, and Neo
def Leo : ℕ := 1
def Max : ℕ := 2
def Neo : ℕ := 3

-- Define the number of players needed for the starting lineup
def starting_lineup_size : ℕ := 6

-- Define the combinations count function
def count_combinations (n k : ℕ) : ℕ := (Finset.range n).choose k

-- Calculating the number of valid lineups
def count_valid_lineups : ℕ :=
  count_combinations 12 5 +   -- Case 1: Leo starts, Max and Neo don't.
  count_combinations 12 5 +   -- Case 2: Max starts, Leo and Neo don't.
  count_combinations 12 5 +   -- Case 3: Neo starts, Leo and Max don't.
  count_combinations 12 6     -- Case 4: None of Leo, Max, or Neo start.

-- The theorem to prove the correct answer
theorem number_of_valid_starting_lineups : count_valid_lineups = 3300 :=
by
  -- This will eventually contain the detailed proof
  sorry

end number_of_valid_starting_lineups_l188_188958


namespace repeating_decimal_product_l188_188715

theorem repeating_decimal_product :
  (8 / 99) * (36 / 99) = 288 / 9801 :=
by
  sorry

end repeating_decimal_product_l188_188715


namespace jean_total_jail_time_l188_188185

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l188_188185


namespace turtle_speed_l188_188350

theorem turtle_speed
  (hare_speed : ℝ)
  (race_distance : ℝ)
  (head_start : ℝ) :
  hare_speed = 10 → race_distance = 20 → head_start = 18 → 
  (race_distance / (head_start + race_distance / hare_speed) = 1) :=
by
  intros
  sorry

end turtle_speed_l188_188350


namespace find_x_when_perpendicular_l188_188908

def a : ℝ × ℝ := (1, -2)
def b (x: ℝ) : ℝ × ℝ := (x, 1)
def are_perpendicular (a b: ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x_when_perpendicular (x: ℝ) (h: are_perpendicular a (b x)) : x = 2 :=
by
  sorry

end find_x_when_perpendicular_l188_188908


namespace necessary_and_sufficient_condition_holds_l188_188644

noncomputable def necessary_and_sufficient_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + m > 0

theorem necessary_and_sufficient_condition_holds (m : ℝ) :
  necessary_and_sufficient_condition m ↔ m > 1 :=
by
  sorry

end necessary_and_sufficient_condition_holds_l188_188644


namespace students_to_add_l188_188694

theorem students_to_add (students := 1049) (teachers := 9) : ∃ n, students + n ≡ 0 [MOD teachers] ∧ n = 4 :=
by
  use 4
  sorry

end students_to_add_l188_188694


namespace find_a_b_transform_line_l188_188149

theorem find_a_b_transform_line (a b : ℝ) (hA : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, a], ![b, 3]]) :
  (∀ x y : ℝ, (2 * (-(x) + a*y) - (b*x + 3*y) - 3 = 0) → (2*x - y - 3 = 0)) →
  a = 1 ∧ b = -4 :=
by {
  sorry
}

end find_a_b_transform_line_l188_188149


namespace linear_eq_rewrite_l188_188206

theorem linear_eq_rewrite (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end linear_eq_rewrite_l188_188206


namespace diagonals_in_15_sided_polygon_l188_188005

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end diagonals_in_15_sided_polygon_l188_188005


namespace probability_event_occurring_exactly_once_l188_188930

theorem probability_event_occurring_exactly_once
  (P : ℝ)
  (h1 : ∀ n : ℕ, P ≥ 0 ∧ P ≤ 1) -- Probabilities are valid for all trials
  (h2 : (1 - (1 - P)^3) = 63 / 64) : -- Given condition for at least once
  (3 * P * (1 - P)^2 = 9 / 64) := 
by
  -- Here you would provide the proof steps using the conditions given.
  sorry

end probability_event_occurring_exactly_once_l188_188930


namespace proof_problem_l188_188624

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as {y | y = 2^x, x ∈ ℝ}
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the set B as {x ∈ ℤ | x^2 - 4 ≤ 0}
def B : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}

-- Define the complement of A relative to U (universal set)
def CU_A : Set ℝ := {x | x ≤ 0}

-- Define the proposition to be proved
theorem proof_problem :
  (CU_A ∩ (Set.image (coe : ℤ → ℝ) B)) = {-2.0, 1.0, 0.0} :=
by 
  sorry

end proof_problem_l188_188624


namespace garden_area_increase_l188_188102

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l188_188102


namespace find_x1_l188_188758

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3)
  (h3 : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4 / 5 :=
sorry

end find_x1_l188_188758


namespace zoe_recycled_correctly_l188_188390

-- Let Z be the number of pounds recycled by Zoe
def pounds_by_zoe (total_points : ℕ) (friends_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_points * pounds_per_point - friends_pounds

-- Given conditions
def total_points : ℕ := 6
def friends_pounds : ℕ := 23
def pounds_per_point : ℕ := 8

-- Lean statement for the proof problem
theorem zoe_recycled_correctly : pounds_by_zoe total_points friends_pounds pounds_per_point = 25 :=
by
  -- proof to be provided here
  sorry

end zoe_recycled_correctly_l188_188390


namespace percent_profit_l188_188321

theorem percent_profit (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 :=
by
  sorry

end percent_profit_l188_188321


namespace vasya_number_l188_188239

theorem vasya_number (a b c d : ℕ) (h1 : a * b = 21) (h2 : b * c = 20) (h3 : ∃ x, x ∈ [4, 7] ∧ a ≠ c ∧ b = 7 ∧ c = 4 ∧ d = 5) : (1000 * a + 100 * b + 10 * c + d) = 3745 :=
sorry

end vasya_number_l188_188239


namespace c_plus_d_is_even_l188_188636

-- Define the conditions
variables {c d : ℕ}
variables (m n : ℕ) (hc : c = 6 * m) (hd : d = 9 * n)

-- State the theorem to be proven
theorem c_plus_d_is_even : 
  (c = 6 * m) → (d = 9 * n) → Even (c + d) :=
by
  -- Proof steps would go here
  sorry

end c_plus_d_is_even_l188_188636


namespace temple_shop_total_cost_l188_188811

theorem temple_shop_total_cost :
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  total_cost = 374 :=
by
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  show total_cost = 374
  sorry

end temple_shop_total_cost_l188_188811


namespace probability_of_forming_triangle_l188_188137

noncomputable def calculate_probability_of_triangle (s : finset (ℕ × ℕ)) : ℝ :=
  let total_segments := (15.choose 2) in
  let total_ways := finset.powersetLen 3 s in
  let triangle_conditions := total_ways.filter (λ t,
    let l := t.to_list in
    match l with
    | [a, b, c] := a + b > c ∧ a + c > b ∧ b + c > a
    | _ := false
    end) in
  (triangle_conditions.card : ℝ) / (total_ways.card : ℝ)

#eval calculate_probability_of_triangle (finset.univ : finset (ℕ × ℕ))

theorem probability_of_forming_triangle (prob : ℝ) :
  prob = calculate_probability_of_triangle (finset.univ : finset (ℕ × ℕ)) :=
sorry

end probability_of_forming_triangle_l188_188137


namespace planar_graph_edge_orientation_l188_188359

open SimpleGraph

theorem planar_graph_edge_orientation {G : SimpleGraph V} [DecidableRel G.adj] 
  (h1 : G.IsSimple) (h2 : G.IsPlanar) (h3 : Finite V) :
  ∃ (orient : ∀ (u v : V), G.adj u v → Prop), 
    (∀ (c : G.cycle), (∃ k, length c = k ∧ count (λ e, orient e.fst e.snd e.right) c.edges ≤ (3 * k) / 4)) ∧
  (∀ (H : SimpleGraph V') [DecidableRel H.adj] (p1 : H.IsSimple) (p2 : H.IsPlanar) (p3 : Finite V'), 
   ∃ (c : H.cycle), (∃ m, length c = m ∧ count (λ e, orient e.fst e.snd e.right) c.edges = (3 * m) / 4)). 

end planar_graph_edge_orientation_l188_188359


namespace range_of_m_l188_188299

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l188_188299


namespace total_workers_count_l188_188831

theorem total_workers_count 
  (W N : ℕ)
  (h1 : (W : ℝ) * 9000 = 7 * 12000 + N * 6000)
  (h2 : W = 7 + N) 
  : W = 14 :=
sorry

end total_workers_count_l188_188831


namespace geo_series_sum_l188_188438

theorem geo_series_sum (a r : ℚ) (n: ℕ) (ha : a = 1/3) (hr : r = 1/2) (hn : n = 8) : 
    (a * (1 - r^n) / (1 - r)) = 85 / 128 := 
by
  sorry

end geo_series_sum_l188_188438


namespace total_amount_received_l188_188276

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end total_amount_received_l188_188276


namespace total_arrangements_l188_188253

theorem total_arrangements :
  let students := 6
  let venueA := 1
  let venueB := 2
  let venueC := 3
  (students.choose venueA) * ((students - venueA).choose venueB) = 60 :=
by
  -- placeholder for the proof
  sorry

end total_arrangements_l188_188253


namespace combined_time_l188_188409

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l188_188409


namespace litter_collection_total_weight_l188_188295

/-- Gina collected 8 bags of litter: 5 bags of glass bottles weighing 7 pounds each and 3 bags of plastic waste weighing 4 pounds each. The 25 neighbors together collected 120 times as much glass as Gina and 80 times as much plastic as Gina. Prove that the total weight of all the collected litter is 5207 pounds. -/
theorem litter_collection_total_weight
  (glass_bags_gina : ℕ)
  (glass_weight_per_bag : ℕ)
  (plastic_bags_gina : ℕ)
  (plastic_weight_per_bag : ℕ)
  (neighbors_glass_multiplier : ℕ)
  (neighbors_plastic_multiplier : ℕ)
  (total_weight : ℕ)
  (h1 : glass_bags_gina = 5)
  (h2 : glass_weight_per_bag = 7)
  (h3 : plastic_bags_gina = 3)
  (h4 : plastic_weight_per_bag = 4)
  (h5 : neighbors_glass_multiplier = 120)
  (h6 : neighbors_plastic_multiplier = 80)
  (h_total_weight : total_weight = 5207) : total_weight = 
  glass_bags_gina * glass_weight_per_bag + 
  plastic_bags_gina * plastic_weight_per_bag + 
  neighbors_glass_multiplier * (glass_bags_gina * glass_weight_per_bag) + 
  neighbors_plastic_multiplier * (plastic_bags_gina * plastic_weight_per_bag) := 
by {
  /- Proof omitted -/
  sorry
}

end litter_collection_total_weight_l188_188295


namespace bill_left_with_22_l188_188130

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end bill_left_with_22_l188_188130


namespace prob_A_wins_all_three_rounds_prob_B_wins_within_five_rounds_l188_188852

-- Definitions of probabilities and initial conditions
noncomputable def P_A_first_wins : ℚ := 2/3
noncomputable def P_A_first_draw : ℚ := 1/6
noncomputable def P_B_first_wins : ℚ := 1/2
noncomputable def P_B_first_draw : ℚ := 1/4
noncomputable def P_B_first_loses : ℚ := 1 - P_B_first_wins - P_B_first_draw
noncomputable def round_A_wins_first : ℚ := P_A_first_wins * P_B_first_loses * P_B_first_loses

-- Problem Statement 1: Probability of the game ending within three rounds and A winning all rounds
theorem prob_A_wins_all_three_rounds : round_A_wins_first = 1/24 := 
by
  sorry

-- Problem Statement 2: Probability of the game ending within five rounds and B winning
noncomputable def prob_B_wins_in_three : ℚ := (1 - P_A_first_wins - P_A_first_draw)^3
noncomputable def prob_B_wins_in_four : ℚ := 3 * (1 - P_A_first_wins - P_A_first_draw)^2 * P_B_first_wins * 5/6
noncomputable def prob_B_wins_in_five : ℚ := 3 * (1 - P_A_first_wins - P_A_first_draw)^2 * P_B_first_wins * 5/6 * 1/2 
                                           + 3 * P_B_first_wins^2 * (1 - P_A_first_wins - P_A_first_draw) * (5/6)^2

theorem prob_B_wins_within_five_rounds : prob_B_wins_in_three + prob_B_wins_in_four + prob_B_wins_in_five = 31/216 := 
by
  sorry

end prob_A_wins_all_three_rounds_prob_B_wins_within_five_rounds_l188_188852


namespace total_candies_in_third_set_l188_188673

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l188_188673


namespace pascal_triangle_eighth_row_l188_188470

def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose (n-1) (k-1) 

theorem pascal_triangle_eighth_row:
  sum_interior_numbers 8 = 126 ∧ binomial_coefficient 8 3 = 21 :=
by
  sorry

end pascal_triangle_eighth_row_l188_188470


namespace binom_20_19_eq_20_l188_188722

theorem binom_20_19_eq_20 (n k : ℕ) (h₁ : n = 20) (h₂ : k = 19)
  (h₃ : ∀ (n k : ℕ), Nat.choose n k = Nat.choose n (n - k))
  (h₄ : ∀ (n : ℕ), Nat.choose n 1 = n) :
  Nat.choose 20 19 = 20 :=
by
  rw [h₁, h₂, h₃ 20 19, Nat.sub_self 19, h₄]
  apply h₄
  sorry

end binom_20_19_eq_20_l188_188722


namespace binom_20_19_equals_20_l188_188720

theorem binom_20_19_equals_20 : nat.choose 20 19 = 20 := 
by 
  sorry

end binom_20_19_equals_20_l188_188720


namespace mike_total_hours_l188_188631

-- Define the number of hours Mike worked each day.
def hours_per_day : ℕ := 3

-- Define the number of days Mike worked.
def days : ℕ := 5

-- Define the total number of hours Mike worked.
def total_hours : ℕ := hours_per_day * days

-- State and prove that the total hours Mike worked is 15.
theorem mike_total_hours : total_hours = 15 := by
  -- Proof goes here
  sorry

end mike_total_hours_l188_188631


namespace value_of_b_l188_188262

theorem value_of_b (b : ℝ) (f g : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2 - b * x + 3) ∧ 
  (∀ x, g x = 2 * x^2 + b * x + 3) ∧ 
  (∀ x, g x = f (x + 6)) →
  b = 12 :=
by
  sorry

end value_of_b_l188_188262


namespace ratio_of_boys_to_girls_l188_188404

variable {α β γ : ℝ}
variable (x y : ℕ)

theorem ratio_of_boys_to_girls (hα : α ≠ 1/2) (hprob : (x * β + y * γ) / (x + y) = 1/2) :
  (x : ℝ) / (y : ℝ) = (1/2 - γ) / (β - 1/2) :=
by
  sorry

end ratio_of_boys_to_girls_l188_188404


namespace replace_movie_cost_l188_188937

def num_popular_action_movies := 20
def num_moderate_comedy_movies := 30
def num_unpopular_drama_movies := 10
def num_popular_comedy_movies := 15
def num_moderate_action_movies := 25

def trade_in_rate_action := 3
def trade_in_rate_comedy := 2
def trade_in_rate_drama := 1

def dvd_cost_popular := 12
def dvd_cost_moderate := 8
def dvd_cost_unpopular := 5

def johns_movie_cost : Nat :=
  let total_trade_in := 
    (num_popular_action_movies + num_moderate_action_movies) * trade_in_rate_action +
    (num_moderate_comedy_movies + num_popular_comedy_movies) * trade_in_rate_comedy +
    num_unpopular_drama_movies * trade_in_rate_drama
  let total_dvd_cost :=
    (num_popular_action_movies + num_popular_comedy_movies) * dvd_cost_popular +
    (num_moderate_comedy_movies + num_moderate_action_movies) * dvd_cost_moderate +
    num_unpopular_drama_movies * dvd_cost_unpopular
  total_dvd_cost - total_trade_in

theorem replace_movie_cost : johns_movie_cost = 675 := 
by
  sorry

end replace_movie_cost_l188_188937


namespace squares_difference_l188_188006

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end squares_difference_l188_188006


namespace arrangement_6_people_l188_188990

theorem arrangement_6_people (A B : Type) : 
  (factorial 6) - 2 * (factorial 4) - 2 * (4 * (factorial 4)) = 504 :=
by
  sorry

end arrangement_6_people_l188_188990


namespace thomas_probability_of_two_pairs_l188_188971

def number_of_ways_to_choose_five_socks := Nat.choose 12 5
def number_of_ways_to_choose_two_pairs_of_colors := Nat.choose 4 2
def number_of_ways_to_choose_one_color_for_single_sock := Nat.choose 2 1
def number_of_ways_to_choose_two_socks_from_three := Nat.choose 3 2
def number_of_ways_to_choose_one_sock_from_three := Nat.choose 3 1

theorem thomas_probability_of_two_pairs : 
  number_of_ways_to_choose_five_socks = 792 →
  number_of_ways_to_choose_two_pairs_of_colors = 6 →
  number_of_ways_to_choose_one_color_for_single_sock = 2 →
  number_of_ways_to_choose_two_socks_from_three = 3 →
  number_of_ways_to_choose_one_sock_from_three = 3 →
  6 * 2 * 3 * 3 * 3 = 324 →
  (324 : ℚ) / 792 = 9 / 22 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end thomas_probability_of_two_pairs_l188_188971


namespace cards_received_while_in_hospital_l188_188630

theorem cards_received_while_in_hospital (T H C : ℕ) (hT : T = 690) (hC : C = 287) (hH : H = T - C) : H = 403 :=
by
  sorry

end cards_received_while_in_hospital_l188_188630


namespace angle_ABD_30_degrees_l188_188016

theorem angle_ABD_30_degrees (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB BD : ℝ) (angle_DBC : ℝ)
  (h1 : BD = AB * (Real.sqrt 3 / 2))
  (h2 : angle_DBC = 90) : 
  ∃ angle_ABD, angle_ABD = 30 :=
by
  sorry

end angle_ABD_30_degrees_l188_188016


namespace pool_filling_water_amount_l188_188237

theorem pool_filling_water_amount (Tina_pail Tommy_pail Timmy_pail Trudy_pail : ℕ) 
  (h1 : Tina_pail = 4)
  (h2 : Tommy_pail = Tina_pail + 2)
  (h3 : Timmy_pail = 2 * Tommy_pail)
  (h4 : Trudy_pail = (3 * Timmy_pail) / 2)
  (Timmy_trips Trudy_trips Tommy_trips Tina_trips: ℕ)
  (h5 : Timmy_trips = 4)
  (h6 : Trudy_trips = 4)
  (h7 : Tommy_trips = 6)
  (h8 : Tina_trips = 6) :
  Timmy_trips * Timmy_pail + Trudy_trips * Trudy_pail + Tommy_trips * Tommy_pail + Tina_trips * Tina_pail = 180 := by
  sorry

end pool_filling_water_amount_l188_188237


namespace y_range_l188_188748

theorem y_range (x y : ℝ) (h1 : 4 * x + y = 1) (h2 : -1 < x) (h3 : x ≤ 2) : -7 ≤ y ∧ y < -3 := 
by
  sorry

end y_range_l188_188748


namespace solve_equation_l188_188817

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l188_188817


namespace remaining_speed_l188_188397

theorem remaining_speed (D : ℝ) (V : ℝ) 
  (h1 : 0.35 * D / 35 + 0.65 * D / V = D / 50) : V = 32.5 :=
by sorry

end remaining_speed_l188_188397


namespace ariana_carnations_l188_188710

theorem ariana_carnations (total_flowers roses_fraction tulips : ℕ) (H1 : total_flowers = 40) (H2 : roses_fraction = 2 / 5) (H3 : tulips = 10) :
    (total_flowers - ((roses_fraction * total_flowers) + tulips)) = 14 :=
by
  -- Total number of roses
  have roses := (2 * 40) / 5
  -- Total number of roses and tulips
  have roses_and_tulips := roses + 10
  -- Total number of carnations
  have carnations := 40 - roses_and_tulips
  show carnations = 14
  sorry

end ariana_carnations_l188_188710


namespace time_spent_driving_l188_188827

def distance_home_to_work: ℕ := 60
def speed_mph: ℕ := 40

theorem time_spent_driving:
  (2 * distance_home_to_work) / speed_mph = 3 := by
  sorry

end time_spent_driving_l188_188827


namespace profit_june_correct_l188_188502

-- Define conditions
def profit_in_May : ℝ := 20000
def profit_in_July : ℝ := 28800

-- Define the monthly growth rate variable
variable (x : ℝ)

-- The growth factor per month
def growth_factor : ℝ := 1 + x

-- Given condition translated to an equation
def profit_relation (x : ℝ) : Prop :=
  profit_in_May * (growth_factor x) * (growth_factor x) = profit_in_July

-- The profit in June should be computed
def profit_in_June (x : ℝ) : ℝ :=
  profit_in_May * (growth_factor x)

-- The target profit in June we want to prove
def target_profit_in_June := 24000

-- Statement to prove
theorem profit_june_correct (h : profit_relation x) : profit_in_June x = target_profit_in_June :=
  sorry  -- proof to be completed

end profit_june_correct_l188_188502


namespace max_value_of_f_l188_188435

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * (Real.cos x)

theorem max_value_of_f : ∃ x : ℝ, f x ≤ 4 :=
sorry

end max_value_of_f_l188_188435


namespace problem_solution_l188_188562

theorem problem_solution (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 14) (h2 : a = b + c) : ab - bc + ac = 7 :=
  sorry

end problem_solution_l188_188562


namespace equation_represents_point_l188_188223

theorem equation_represents_point 
  (a b x y : ℝ) 
  (h : (x - a) ^ 2 + (y + b) ^ 2 = 0) : 
  x = a ∧ y = -b := 
by
  sorry

end equation_represents_point_l188_188223


namespace num_nat_numbers_l188_188837

theorem num_nat_numbers (n : ℕ) (h1 : n ≥ 1) (h2 : n ≤ 1992)
  (h3 : ∃ k3, n = 3 * k3)
  (h4 : ¬ (∃ k2, n = 2 * k2))
  (h5 : ¬ (∃ k5, n = 5 * k5)) : ∃ (m : ℕ), m = 266 :=
by
  sorry

end num_nat_numbers_l188_188837


namespace point_P_on_number_line_l188_188946

variable (A : ℝ) (B : ℝ) (P : ℝ)

theorem point_P_on_number_line (hA : A = -1) (hB : B = 5) (hDist : abs (P - A) = abs (B - P)) : P = 2 := 
sorry

end point_P_on_number_line_l188_188946


namespace f_2014_value_l188_188897

def f : ℝ → ℝ :=
sorry

lemma f_periodic (x : ℝ) : f (x + 2) = f (x - 2) :=
sorry

lemma f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 4) : f x = x^2 :=
sorry

theorem f_2014_value : f 2014 = 4 :=
by
  -- Insert proof here
  sorry

end f_2014_value_l188_188897


namespace min_shirts_to_save_money_l188_188266

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 75 + 8 * x < 12 * x ∧ x = 19 :=
sorry

end min_shirts_to_save_money_l188_188266


namespace four_p_plus_one_composite_l188_188056

theorem four_p_plus_one_composite (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) (h2p_plus1_prime : Nat.Prime (2 * p + 1)) : ¬ Nat.Prime (4 * p + 1) :=
sorry

end four_p_plus_one_composite_l188_188056


namespace infinitely_many_good_pairs_l188_188422

def is_triangular (t : ℕ) : Prop :=
  ∃ n : ℕ, t = n * (n + 1) / 2

theorem infinitely_many_good_pairs :
  ∃ (a b : ℕ), (0 < a) ∧ (0 < b) ∧ 
  ∀ t : ℕ, is_triangular t ↔ is_triangular (a * t + b) :=
sorry

end infinitely_many_good_pairs_l188_188422


namespace zoey_finishes_20th_book_on_wednesday_l188_188087

theorem zoey_finishes_20th_book_on_wednesday :
  let days_spent := (20 * 21) / 2
  (days_spent % 7) = 0 → 
  (start_day : ℕ) → start_day = 3 → ((start_day + days_spent) % 7) = 3 :=
by
  sorry

end zoey_finishes_20th_book_on_wednesday_l188_188087


namespace number_of_solutions_abs_eq_l188_188962

theorem number_of_solutions_abs_eq (f : ℝ → ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, f x = |3 * x| ∧ g x = |x - 2| ∧ (f x + g x = 4) → 
  ∃! x1 x2 : ℝ, 
    ((0 < x1 ∧ x1 < 2 ∧ f x1 + g x1 = 4 ) ∨ 
    (x2 < 0 ∧ f x2 + g x2 = 4) ∧ x1 ≠ x2)) :=
by
  sorry

end number_of_solutions_abs_eq_l188_188962


namespace minimum_value_expression_l188_188436

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ( (3*a*b - 6*b + a*(1-a))^2 + (9*b^2 + 2*a + 3*b*(1-a))^2 ) / (a^2 + 9*b^2) ≥ 4 :=
sorry

end minimum_value_expression_l188_188436


namespace maximum_n_for_dart_probability_l188_188865

theorem maximum_n_for_dart_probability (n : ℕ) (h : n ≥ 1) :
  (∃ r : ℝ, r = 1 ∧
  ∃ A_square A_circles : ℝ, A_square = n^2 ∧ A_circles = n * π * r^2 ∧
  (A_circles / A_square) ≥ 1 / 2) → n ≤ 6 := by
  sorry

end maximum_n_for_dart_probability_l188_188865


namespace garden_area_increase_l188_188106

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l188_188106


namespace integer_roots_of_quadratic_eq_are_neg3_and_neg7_l188_188433

theorem integer_roots_of_quadratic_eq_are_neg3_and_neg7 :
  {k : ℤ | ∃ x : ℤ, k * x^2 - 2 * (3 * k - 1) * x + 9 * k - 1 = 0} = {-3, -7} :=
by
  sorry

end integer_roots_of_quadratic_eq_are_neg3_and_neg7_l188_188433


namespace problem_A_problem_B_problem_C_problem_D_problem_E_l188_188882

-- Definitions and assumptions based on the problem statement
def eqI (x y z : ℕ) := x + y + z = 45
def eqII (x y z w : ℕ) := x + y + z + w = 50
def consecutive_odd_integers (x y z : ℕ) := y = x + 2 ∧ z = x + 4
def multiples_of_five (x y z w : ℕ) := (∃ a b c d : ℕ, x = 5 * a ∧ y = 5 * b ∧ z = 5 * c ∧ w = 5 * d)
def consecutive_integers (x y z w : ℕ) := y = x + 1 ∧ z = x + 2 ∧ w = x + 3
def prime_integers (x y z : ℕ) := Prime x ∧ Prime y ∧ Prime z

-- Lean theorem statements
theorem problem_A : ∃ x y z : ℕ, eqI x y z ∧ consecutive_odd_integers x y z := 
sorry

theorem problem_B : ¬ (∃ x y z : ℕ, eqI x y z ∧ prime_integers x y z) := 
sorry

theorem problem_C : ¬ (∃ x y z w : ℕ, eqII x y z w ∧ consecutive_odd_integers x y z) :=
sorry

theorem problem_D : ∃ x y z w : ℕ, eqII x y z w ∧ multiples_of_five x y z w := 
sorry

theorem problem_E : ∃ x y z w : ℕ, eqII x y z w ∧ consecutive_integers x y z w := 
sorry

end problem_A_problem_B_problem_C_problem_D_problem_E_l188_188882


namespace range_x_minus_2y_l188_188752

variable (x y : ℝ)

def cond1 : Prop := -1 ≤ x ∧ x < 2
def cond2 : Prop := 0 < y ∧ y ≤ 1

theorem range_x_minus_2y 
  (h1 : cond1 x) 
  (h2 : cond2 y) : 
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 := 
by
  sorry

end range_x_minus_2y_l188_188752


namespace cos_A_zero_l188_188588

theorem cos_A_zero (A : ℝ) (h : Real.tan A + (1 / Real.tan A) + 2 / (Real.cos A) = 4) : Real.cos A = 0 :=
sorry

end cos_A_zero_l188_188588


namespace remainder_of_3_pow_20_mod_7_l188_188247

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l188_188247


namespace money_difference_l188_188122

def share_ratio (w x y z : ℝ) (k : ℝ) : Prop :=
  w = k ∧ x = 6 * k ∧ y = 2 * k ∧ z = 4 * k

theorem money_difference (k : ℝ) (h : k = 375) : 
  ∀ w x y z : ℝ, share_ratio w x y z k → (x - y) = 1500 := 
by
  intros w x y z h_ratio
  rw [share_ratio] at h_ratio
  have h_w : w = k := h_ratio.1
  have h_x : x = 6 * k := h_ratio.2.1
  have h_y : y = 2 * k := h_ratio.2.2.1
  rw [h_x, h_y]
  rw [h] at h_x h_y
  sorry

end money_difference_l188_188122


namespace smaller_of_two_integers_l188_188830

theorem smaller_of_two_integers (m n : ℕ) (h1 : 100 ≤ m ∧ m < 1000) (h2 : 100 ≤ n ∧ n < 1000)
  (h3 : (m + n) / 2 = m + n / 1000) : min m n = 999 :=
by {
  sorry
}

end smaller_of_two_integers_l188_188830


namespace find_a_l188_188297

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 * Real.exp x

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 1 → (x - a) * (x - a + 2) ≤ 0) → a = 1 :=
by
  intro h
  sorry 

end find_a_l188_188297


namespace total_candies_third_set_l188_188653

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l188_188653


namespace determinant_of_matrix_A_l188_188731

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 4], ![3, x, -2], ![1, -3, 0]]

theorem determinant_of_matrix_A (x : ℝ) :
  Matrix.det (matrix_A x) = -46 - 4 * x :=
by
  sorry

end determinant_of_matrix_A_l188_188731


namespace sufficient_but_not_necessary_condition_l188_188389

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x^2 + y^2 ≤ 1) → ((x - 1)^2 + y^2 ≤ 4) ∧ ¬ ((x - 1)^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l188_188389


namespace cubic_foot_to_cubic_inches_l188_188912

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l188_188912


namespace math_problem_l188_188953

variable (a : ℝ)

theorem math_problem (h : a^2 + 3 * a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2 * a)) = 1 := 
sorry

end math_problem_l188_188953


namespace total_questions_on_test_l188_188123

/-- A teacher grades students' tests by subtracting twice the number of incorrect responses
    from the number of correct responses. Given that a student received a score of 64
    and answered 88 questions correctly, prove that the total number of questions on the test is 100. -/
theorem total_questions_on_test (score correct_responses : ℕ) (grading_system : ℕ → ℕ → ℕ)
  (h1 : score = grading_system correct_responses (88 - 2 * 12))
  (h2 : correct_responses = 88)
  (h3 : score = 64) : correct_responses + (88 - 2 * 12) = 100 :=
by
  sorry

end total_questions_on_test_l188_188123


namespace annie_crayons_l188_188417

def initial_crayons : ℕ := 4
def additional_crayons : ℕ := 36
def total_crayons : ℕ := initial_crayons + additional_crayons

theorem annie_crayons : total_crayons = 40 :=
by
  sorry

end annie_crayons_l188_188417


namespace verify_total_amount_spent_by_mary_l188_188809

def shirt_price : Float := 13.04
def shirt_sales_tax_rate : Float := 0.07

def jacket_original_price_gbp : Float := 15.34
def jacket_discount_rate : Float := 0.20
def jacket_sales_tax_rate : Float := 0.085
def conversion_rate_usd_per_gbp : Float := 1.28

def scarf_price : Float := 7.90
def hat_price : Float := 9.13
def hat_scarf_sales_tax_rate : Float := 0.065

def total_amount_spent_by_mary : Float :=
  let shirt_total := shirt_price * (1 + shirt_sales_tax_rate)
  let jacket_discounted := jacket_original_price_gbp * (1 - jacket_discount_rate)
  let jacket_total_gbp := jacket_discounted * (1 + jacket_sales_tax_rate)
  let jacket_total_usd := jacket_total_gbp * conversion_rate_usd_per_gbp
  let hat_scarf_combined_price := scarf_price + hat_price
  let hat_scarf_total := hat_scarf_combined_price * (1 + hat_scarf_sales_tax_rate)
  shirt_total + jacket_total_usd + hat_scarf_total

theorem verify_total_amount_spent_by_mary : total_amount_spent_by_mary = 49.13 :=
by sorry

end verify_total_amount_spent_by_mary_l188_188809


namespace set_intersection_complement_l188_188582

variable (U : Set ℕ)
variable (P Q : Set ℕ)

theorem set_intersection_complement {U : Set ℕ} {P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5, 6}) 
  (hP : P = {1, 2, 3, 4}) 
  (hQ : Q = {3, 4, 5, 6}) : 
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l188_188582


namespace remaining_money_l188_188427

-- Define the conditions
def num_pies : ℕ := 200
def price_per_pie : ℕ := 20
def fraction_for_ingredients : ℚ := 3 / 5

-- Define the total sales
def total_sales : ℕ := num_pies * price_per_pie

-- Define the cost for ingredients
def cost_for_ingredients : ℚ := fraction_for_ingredients * total_sales 

-- Prove the remaining money
theorem remaining_money : (total_sales : ℚ) - cost_for_ingredients = 1600 := 
by {
  -- This is where the proof would go
  sorry
}

end remaining_money_l188_188427


namespace solve_equation_l188_188813

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2/3 → (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2)) →
  (∀ x : ℝ, x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3) :=
by
  sorry

end solve_equation_l188_188813


namespace quadratic_inequality_solution_l188_188966

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
by
  sorry

end quadratic_inequality_solution_l188_188966


namespace race_distance_between_Sasha_and_Kolya_l188_188054

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l188_188054


namespace jose_speed_l188_188192

theorem jose_speed
  (distance : ℕ) (time : ℕ)
  (h_distance : distance = 4)
  (h_time : time = 2) :
  distance / time = 2 := by
  sorry

end jose_speed_l188_188192


namespace garden_area_increase_l188_188117

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l188_188117


namespace tens_digit_17_pow_1993_l188_188283

theorem tens_digit_17_pow_1993 :
  (17 ^ 1993) % 100 / 10 = 3 := by
  sorry

end tens_digit_17_pow_1993_l188_188283


namespace two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l188_188798

theorem two_divides_a_squared_minus_a (a : ℤ) : ∃ k₁ : ℤ, a^2 - a = 2 * k₁ :=
sorry

theorem three_divides_a_cubed_minus_a (a : ℤ) : ∃ k₂ : ℤ, a^3 - a = 3 * k₂ :=
sorry

end two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l188_188798


namespace number_of_divisors_10_factorial_greater_than_9_factorial_l188_188770

noncomputable def numDivisorsGreaterThan9Factorial : Nat :=
  let n := 10!
  let m := 9!
  let valid_divisors := (List.range 10).map (fun i => n / (i + 1))
  valid_divisors.count (fun d => d > m)

theorem number_of_divisors_10_factorial_greater_than_9_factorial :
  numDivisorsGreaterThan9Factorial = 9 := 
sorry

end number_of_divisors_10_factorial_greater_than_9_factorial_l188_188770


namespace other_equation_l188_188859

-- Define the variables for the length of the rope and the depth of the well
variables (x y : ℝ)

-- Given condition
def cond1 : Prop := (1/4) * x = y + 3

-- The proof goal
theorem other_equation (h : cond1 x y) : (1/5) * x = y + 2 :=
sorry

end other_equation_l188_188859


namespace range_of_a_l188_188171

-- Given conditions
def condition1 (x : ℝ) := (4 + x) / 3 > (x + 2) / 2
def condition2 (x : ℝ) (a : ℝ) := (x + a) / 2 < 0

-- The statement to prove
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, condition1 x → condition2 x a → x < 2) → a ≤ -2 :=
sorry

end range_of_a_l188_188171


namespace total_candies_in_third_set_l188_188679

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l188_188679


namespace bx_squared_l188_188483

open Isosceles
open Triangle

-- Definitions based on the problem conditions
structure Triangle :=
(A B C : Point)
(M : Point) (is_midpoint_M : midpoint A C M)
(N : Point) (is_bisector_CN : bisector C A B N)
(X : Point) (intersection_BM_CN : intersectsLineSegment B M X ∧ intersectsLineSegment C N X)

-- Defining the triangle with given conditions
def triangle_data : Triangle := {
  A := (0, 0),
  B := (0, sqrt 3 / 2),
  C := (4, 0),
  M := (2, 0),
  N := (-1 / 2, 0),
  X := (1 / 2, 0),
  is_midpoint_M := sorry,
  is_bisector_CN := sorry,
  intersection_BM_CN := sorry,
}

-- The main theorem to prove
theorem bx_squared (T : Triangle) (h_iso : isosceles T.B T.X T.N 1 1) : 
  length_squared T.B T.X = 1 := 
sorry

end bx_squared_l188_188483


namespace real_roots_determinant_l188_188024

variable (a b c k : ℝ)
variable (k_pos : k > 0)
variable (a_nonzero : a ≠ 0) 
variable (b_nonzero : b ≠ 0)
variable (c_nonzero : c ≠ 0)
variable (k_nonzero : k ≠ 0)

theorem real_roots_determinant : 
  ∃! x : ℝ, (Matrix.det ![![x, k * c, -k * b], ![-k * c, x, k * a], ![k * b, -k * a, x]] = 0) :=
sorry

end real_roots_determinant_l188_188024


namespace train_and_car_combined_time_l188_188406

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l188_188406


namespace find_point_symmetric_about_y_axis_l188_188326

def point := ℤ × ℤ

def symmetric_about_y_axis (A B : point) : Prop :=
  B.1 = -A.1 ∧ B.2 = A.2

theorem find_point_symmetric_about_y_axis (A B : point) 
  (hA : A = (-5, 2)) 
  (hSym : symmetric_about_y_axis A B) : 
  B = (5, 2) := 
by
  -- We declare the proof but omit the steps for this exercise.
  sorry

end find_point_symmetric_about_y_axis_l188_188326


namespace dave_total_rides_l188_188419

theorem dave_total_rides (rides_first_day rides_second_day : ℕ) (h1 : rides_first_day = 4) (h2 : rides_second_day = 3) :
  rides_first_day + rides_second_day = 7 :=
by
  sorry

end dave_total_rides_l188_188419


namespace cricket_target_run_l188_188015

theorem cricket_target_run (run_rate1 run_rate2 : ℝ) (overs1 overs2 : ℕ) (T : ℝ) 
  (h1 : run_rate1 = 3.2) (h2 : overs1 = 10) (h3 : run_rate2 = 25) (h4 : overs2 = 10) :
  T = (run_rate1 * overs1) + (run_rate2 * overs2) → T = 282 :=
by
  sorry

end cricket_target_run_l188_188015


namespace num_ordered_pairs_1806_l188_188840

theorem num_ordered_pairs_1806 :
  let n := 1806 in
  let pf := [(2, 1), (3, 2), (101, 1)] in
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1) in
  ∃ (c : ℕ), c = num_divisors ∧ c = 12 :=
by
  let n := 1806
  let pf := [(2, 1), (3, 2), (101, 1)]
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1)
  use num_divisors
  split
  . rfl
  . rfl
  sorry

end num_ordered_pairs_1806_l188_188840


namespace last_two_digits_l188_188505

theorem last_two_digits (a b : ℕ) (n : ℕ) (h : b ≡ 25 [MOD 100]) (h_pow : (25 : ℕ) ^ n ≡ 25 [MOD 100]) :
  (33 * b ^ n) % 100 = 25 :=
by
  sorry

end last_two_digits_l188_188505


namespace diagonals_in_15_sided_polygon_l188_188004

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end diagonals_in_15_sided_polygon_l188_188004


namespace total_candies_in_third_set_l188_188675

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l188_188675


namespace f_eq_32x5_l188_188800

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end f_eq_32x5_l188_188800


namespace P_eq_CU_M_union_CU_N_l188_188487

open Set

-- Definitions of U, M, N
def U : Set (ℝ × ℝ) := { p | True }
def M : Set (ℝ × ℝ) := { p | p.2 ≠ p.1 }
def N : Set (ℝ × ℝ) := { p | p.2 ≠ -p.1 }
def CU_M : Set (ℝ × ℝ) := { p | p.2 = p.1 }
def CU_N : Set (ℝ × ℝ) := { p | p.2 = -p.1 }

-- Theorem statement
theorem P_eq_CU_M_union_CU_N :
  { p : ℝ × ℝ | p.2^2 ≠ p.1^2 } = CU_M ∪ CU_N :=
sorry

end P_eq_CU_M_union_CU_N_l188_188487


namespace range_of_a_sufficient_but_not_necessary_condition_l188_188707

theorem range_of_a_sufficient_but_not_necessary_condition (a : ℝ) : 
  (-2 < x ∧ x < -1) → ((x + a) * (x + 1) < 0) → (a > 2) :=
sorry

end range_of_a_sufficient_but_not_necessary_condition_l188_188707


namespace simplify_and_evaluate_l188_188954

theorem simplify_and_evaluate (a : ℕ) (h : a = 2) : 
  (1 - (1 : ℚ) / (a + 1)) / (a / ((a * a) - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l188_188954


namespace d_divisibility_l188_188498

theorem d_divisibility (p d : ℕ) (h_p : 0 < p) (h_d : 0 < d)
  (h1 : Prime p) 
  (h2 : Prime (p + d)) 
  (h3 : Prime (p + 2 * d)) 
  (h4 : Prime (p + 3 * d)) 
  (h5 : Prime (p + 4 * d)) 
  (h6 : Prime (p + 5 * d)) : 
  (2 ∣ d) ∧ (3 ∣ d) ∧ (5 ∣ d) :=
by
  sorry

end d_divisibility_l188_188498


namespace garden_area_increase_l188_188101

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l188_188101


namespace length_of_AB_l188_188613

-- Define the problem variables
variables (AB CD : ℝ)
variables (h : ℝ)

-- Define the conditions
def ratio_condition (AB CD : ℝ) : Prop :=
  AB / CD = 7 / 3

def length_condition (AB CD : ℝ) : Prop :=
  AB + CD = 210

-- Lean statement combining the conditions and the final result
theorem length_of_AB (h : ℝ) (AB CD : ℝ) (h_ratio : ratio_condition AB CD) (h_length : length_condition AB CD) : 
  AB = 147 :=
by
  -- Definitions and proof would go here
  sorry

end length_of_AB_l188_188613


namespace one_cubic_foot_is_1728_cubic_inches_l188_188915

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l188_188915


namespace find_value_l188_188784

theorem find_value (x : ℝ) (f₁ f₂ : ℝ) (p : ℝ) (y₁ y₂ : ℝ) 
  (h1 : x * f₁ = (p * x) * y₁)
  (h2 : x * f₂ = (p * x) * y₂)
  (hf₁ : f₁ = 1 / 3)
  (hx : x = 4)
  (hy₁ : y₁ = 8)
  (hf₂ : f₂ = 1 / 8):
  y₂ = 3 := by
sorry

end find_value_l188_188784


namespace smallest_a1_l188_188484

theorem smallest_a1 (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - n) :
  a 1 ≥ 13 / 36 :=
by
  sorry

end smallest_a1_l188_188484


namespace compute_expression_l188_188544

theorem compute_expression (x : ℝ) (h : x = 7) : (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := 
by
  sorry

end compute_expression_l188_188544


namespace remainder_7623_div_11_l188_188385

theorem remainder_7623_div_11 : 7623 % 11 = 0 := 
by sorry

end remainder_7623_div_11_l188_188385


namespace james_trip_time_l188_188614

def speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

theorem james_trip_time:
  (distance / speed) + stop_time = 7 := 
by
  sorry

end james_trip_time_l188_188614


namespace intersection_M_N_l188_188806

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Prove the intersection of M and N equals (1, 2)
theorem intersection_M_N :
  ∀ x, x ∈ M ∩ N ↔ 1 < x ∧ x < 2 :=
by
  -- Skipping the proof here
  sorry

end intersection_M_N_l188_188806


namespace rectangle_width_l188_188368

-- Define the conditions
def length := 6
def area_triangle := 60
def area_ratio := 2/5

-- The theorem: proving that the width of the rectangle is 4 cm
theorem rectangle_width (w : ℝ) (A_triangle : ℝ) (len : ℝ) 
  (ratio : ℝ) (h1 : A_triangle = 60) (h2 : len = 6) (h3 : ratio = 2 / 5) 
  (h4 : (len * w) / A_triangle = ratio) : 
  w = 4 := 
by 
  sorry

end rectangle_width_l188_188368


namespace largest_fraction_l188_188081

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (6 : ℚ) / 13
  let C := (18 : ℚ) / 37
  let D := (101 : ℚ) / 202
  let E := (200 : ℚ) / 399
  E > A ∧ E > B ∧ E > C ∧ E > D := by
  sorry

end largest_fraction_l188_188081


namespace percent_answered_second_correctly_l188_188593

theorem percent_answered_second_correctly
  (nA : ℝ) (nAB : ℝ) (n_neither : ℝ) :
  nA = 0.80 → nAB = 0.60 → n_neither = 0.05 → 
  (nA + nB - nAB + n_neither = 1) → 
  ((1 - n_neither) = nA + nB - nAB) → 
  nB = 0.75 :=
by
  intros h1 h2 h3 hUnion hInclusion
  sorry

end percent_answered_second_correctly_l188_188593


namespace original_fraction_2_7_l188_188466

theorem original_fraction_2_7 (N D : ℚ) : 
  (1.40 * N) / (0.50 * D) = 4 / 5 → N / D = 2 / 7 :=
by
  intro h
  sorry

end original_fraction_2_7_l188_188466


namespace find_expression_for_f_l188_188923

noncomputable def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

-- Assuming a, b ∈ ℝ, f(x) is even, and range of f(x) is (-∞, 2]
theorem find_expression_for_f (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = f (-x) a b) (h2 : ∀ y : ℝ, ∃ x : ℝ, f x a b = y → y ≤ 2):
  f x a b = -x^2 + 2 :=
by 
  sorry

end find_expression_for_f_l188_188923


namespace range_of_a_l188_188763

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + a^2 > 0) ↔ (a < -1 ∨ a > (1 : ℝ) / 3) := 
sorry

end range_of_a_l188_188763


namespace unique_reconstruction_l188_188203

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end unique_reconstruction_l188_188203


namespace odd_powers_sum_divisible_by_p_l188_188940

theorem odd_powers_sum_divisible_by_p
  (p : ℕ)
  (hp_prime : Prime p)
  (hp_gt_3 : 3 < p)
  (a b c d : ℕ)
  (h_sum : (a + b + c + d) % p = 0)
  (h_cube_sum : (a^3 + b^3 + c^3 + d^3) % p = 0)
  (n : ℕ)
  (hn_odd : n % 2 = 1 ) :
  (a^n + b^n + c^n + d^n) % p = 0 :=
sorry

end odd_powers_sum_divisible_by_p_l188_188940


namespace wade_tips_l188_188975

/-- Wade has a hot dog food truck. 
     He makes $2.00 in tips per customer.
     On Friday he served 28 customers.
     He served three times that amount of customers on Saturday.
     On Sunday, he served 36 customers.
     Prove that Wade made $296 in tips between the 3 days. -/
theorem wade_tips : 
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let customers_sunday := 36
  let tips_friday := tips_per_customer * customers_friday
  let tips_saturday := tips_per_customer * customers_saturday
  let tips_sunday := tips_per_customer * customers_sunday
  let total_tips := tips_friday + tips_saturday + tips_sunday
  in total_tips = 296 := 
by
  sorry

end wade_tips_l188_188975


namespace cover_points_with_two_disks_l188_188749

theorem cover_points_with_two_disks :
  ∀ (points : Fin 2014 → ℝ × ℝ),
    (∀ (i j k : Fin 2014), i ≠ j → j ≠ k → i ≠ k → 
      dist (points i) (points j) ≤ 1 ∨ dist (points j) (points k) ≤ 1 ∨ dist (points i) (points k) ≤ 1) →
    ∃ (A B : ℝ × ℝ), ∀ (p : Fin 2014),
      dist (points p) A ≤ 1 ∨ dist (points p) B ≤ 1 :=
by
  sorry

end cover_points_with_two_disks_l188_188749


namespace total_candies_in_third_set_l188_188671

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l188_188671


namespace cubic_foot_to_cubic_inches_l188_188913

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l188_188913


namespace sequence_formulas_and_reciprocal_sum_l188_188787

-- Definitions based on problem's conditions
def arithmetic_sequence (a : ℕ → ℕ) := ∀ n, a n = 1 + (n-1) * d
def geometric_sequence (b : ℕ → ℕ) := ∀ n, b n = q^(n-1)
def Sn (a : ℕ → ℕ) := ∀ n, ∑ i in range (n+1), a i

-- Conditions from the problem
axiom a1_pos : ∀ n, a n > 0 -- Condition: all terms of the arithmetic sequence are positive
axiom a1_init : a 1 = 1 -- Initial term of the arithmetic sequence
axiom b1_init : b 1 = 1 -- Initial term of the geometric sequence
axiom condition1 : b 2 * Sn 2 = 6 -- Condition 1
axiom condition2 : b 2 + Sn 3 = 8 -- Condition 2

-- Theorem to prove the general formulas and the sum of reciprocals
theorem sequence_formulas_and_reciprocal_sum :
  (∀ n, a n = n) ∧
  (∀ n, b n = 2^(n-1)) ∧
  (∀ n, (∑ i in range (n+1), 1 / (Sn i)) = 2 * (1 - (1 / (n+1)))) := by
  sorry

end sequence_formulas_and_reciprocal_sum_l188_188787


namespace m_plus_n_eq_five_l188_188307

theorem m_plus_n_eq_five (m n : ℝ) (h1 : m - 2 = 0) (h2 : 1 + n - 2 * m = 0) : m + n = 5 := 
  by 
  sorry

end m_plus_n_eq_five_l188_188307


namespace part1_part2_l188_188233

def traditional_chinese_paintings : ℕ := 6
def oil_paintings : ℕ := 4
def watercolor_paintings : ℕ := 5

theorem part1 :
  traditional_chinese_paintings * oil_paintings * watercolor_paintings = 120 :=
by
  sorry

theorem part2 :
  (traditional_chinese_paintings * oil_paintings) + 
  (traditional_chinese_paintings * watercolor_paintings) + 
  (oil_paintings * watercolor_paintings) = 74 :=
by
  sorry

end part1_part2_l188_188233


namespace division_multiplication_order_l188_188716

theorem division_multiplication_order : 1100 / 25 * 4 / 11 = 16 := by
  sorry

end division_multiplication_order_l188_188716


namespace total_candies_third_set_l188_188652

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l188_188652


namespace child_ticket_price_correct_l188_188980

-- Definitions based on conditions
def total_collected := 104
def price_adult := 6
def total_tickets := 21
def children_tickets := 11

-- Derived conditions
def adult_tickets := total_tickets - children_tickets
def total_revenue_child (C : ℕ) := children_tickets * C
def total_revenue_adult := adult_tickets * price_adult

-- Main statement to prove
theorem child_ticket_price_correct (C : ℕ) 
  (h1 : total_revenue_child C + total_revenue_adult = total_collected) : 
  C = 4 :=
by
  sorry

end child_ticket_price_correct_l188_188980


namespace rectangle_area_l188_188261

namespace RectangleAreaProof

theorem rectangle_area (SqrArea : ℝ) (SqrSide : ℝ) (RectWidth : ℝ) (RectLength : ℝ) (RectArea : ℝ) :
  SqrArea = 36 →
  SqrSide = Real.sqrt SqrArea →
  RectWidth = SqrSide →
  RectLength = 3 * RectWidth →
  RectArea = RectWidth * RectLength →
  RectArea = 108 := by
  sorry

end RectangleAreaProof

end rectangle_area_l188_188261


namespace incenter_coordinates_l188_188931

-- Define lengths of the sides of the triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 6

-- Define the incenter formula components
def sum_of_sides : ℕ := a + b + c
def x : ℚ := a / (sum_of_sides : ℚ)
def y : ℚ := b / (sum_of_sides : ℚ)
def z : ℚ := c / (sum_of_sides : ℚ)

-- Prove the result
theorem incenter_coordinates :
  (x, y, z) = (1 / 3, 5 / 12, 1 / 4) :=
by 
  -- Proof skipped
  sorry

end incenter_coordinates_l188_188931


namespace find_original_number_l188_188088

theorem find_original_number
  (x : ℤ)
  (h : 3 * (2 * x + 5) = 123) :
  x = 18 := 
sorry

end find_original_number_l188_188088


namespace miller_rabin_probability_at_least_half_l188_188494

theorem miller_rabin_probability_at_least_half
  {n : ℕ} (hcomp : ¬Nat.Prime n) (s d : ℕ) (hd_odd : d % 2 = 1) (h_decomp : n - 1 = 2^s * d)
  (a : ℤ) (ha_range : 2 ≤ a ∧ a ≤ n - 2) :
  ∃ P : ℝ, P ≥ 1 / 2 ∧ ∀ a, (2 ≤ a ∧ a ≤ n - 2) → ¬(a^(d * 2^s) % n = 1)
  :=
sorry

end miller_rabin_probability_at_least_half_l188_188494


namespace jean_jail_time_l188_188182

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l188_188182


namespace boy_reaches_early_l188_188076

theorem boy_reaches_early (usual_rate new_rate : ℝ) (Usual_Time New_Time : ℕ) 
  (Hrate : new_rate = 9/8 * usual_rate) (Htime : Usual_Time = 36) :
  New_Time = 32 → Usual_Time - New_Time = 4 :=
by
  intros
  subst_vars
  sorry

end boy_reaches_early_l188_188076


namespace multiply_469111111_by_99999999_l188_188876

theorem multiply_469111111_by_99999999 :
  469111111 * 99999999 = 46911111053088889 :=
sorry

end multiply_469111111_by_99999999_l188_188876


namespace ratio_implies_sum_ratio_l188_188589

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_implies_sum_ratio_l188_188589


namespace garden_area_difference_l188_188112

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l188_188112


namespace find_x4_plus_y4_l188_188462

theorem find_x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^4 + y^4 = 135.5 :=
by
  sorry

end find_x4_plus_y4_l188_188462


namespace max_distance_from_ellipse_to_line_l188_188961

theorem max_distance_from_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 4) = 1
  let line (x y : ℝ) := x + 2 * y - Real.sqrt 2 = 0
  ∃ (d : ℝ), (∀ (x y : ℝ), ellipse x y → line x y → d = Real.sqrt 10) :=
sorry

end max_distance_from_ellipse_to_line_l188_188961


namespace solve_modified_system_l188_188230

theorem solve_modified_system (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : 4 * a1 + 6 * b1 = c1) 
  (h2 : 4 * a2 + 6 * b2 = c2) :
  (4 * a1 * 5 + 3 * b1 * 10 = 5 * c1) ∧ (4 * a2 * 5 + 3 * b2 * 10 = 5 * c2) :=
by
  sorry

end solve_modified_system_l188_188230


namespace inverse_value_l188_188163

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value :
  g (-3) = -103 :=
by
  sorry

end inverse_value_l188_188163


namespace total_candies_third_set_l188_188655

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l188_188655


namespace garden_perimeter_l188_188260

theorem garden_perimeter
  (a b : ℝ)
  (h1 : a^2 + b^2 = 1156)
  (h2 : a * b = 240) :
  2 * (a + b) = 80 :=
sorry

end garden_perimeter_l188_188260


namespace arithmetic_mean_18_27_45_l188_188684

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_18_27_45_l188_188684


namespace probability_distribution_l188_188924

namespace StandardParts

-- Conditions
def totalParts := 10
def standardParts := 8
def selectedParts := 2

-- Hypergeometric distribution calculations
def P_X_0 := (Nat.choose 8 0 * Nat.choose 2 2) / Nat.choose 10 2
def P_X_1 := (Nat.choose 8 1 * Nat.choose 2 1) / Nat.choose 10 2
def P_X_2 := (Nat.choose 8 2 * Nat.choose 2 0) / Nat.choose 10 2

-- Proof
theorem probability_distribution :
  P_X_0 = 1 / 45 ∧ P_X_1 = 16 / 45 ∧ P_X_2 = 28 / 45 := by
  sorry

end StandardParts

end probability_distribution_l188_188924


namespace find_fx_for_l188_188225

theorem find_fx_for {f : ℕ → ℤ} (h1 : f 0 = 1) (h2 : ∀ x, f (x + 1) = f x + 2 * x + 3) : f 2012 = 4052169 :=
by
  sorry

end find_fx_for_l188_188225


namespace train_and_car_combined_time_l188_188413

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l188_188413


namespace wendy_score_l188_188375

def score_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def treasures_second_level : ℕ := 3

theorem wendy_score :
  score_per_treasure * treasures_first_level + score_per_treasure * treasures_second_level = 35 :=
by
  sorry

end wendy_score_l188_188375


namespace circles_disjoint_l188_188560

theorem circles_disjoint (a : ℝ) : ((x - 1)^2 + (y - 1)^2 = 4) ∧ (x^2 + (y - a)^2 = 1) → (a < 1 - 2 * Real.sqrt 2 ∨ a > 1 + 2 * Real.sqrt 2) :=
by sorry

end circles_disjoint_l188_188560


namespace kevin_total_hops_l188_188938

/-- Define the hop function for Kevin -/
def hop (remaining_distance : ℚ) : ℚ :=
  remaining_distance / 4

/-- Summing the series for five hops -/
def total_hops (start_distance : ℚ) (hops : ℕ) : ℚ :=
  let h0 := hop start_distance
  let h1 := hop (start_distance - h0)
  let h2 := hop (start_distance - h0 - h1)
  let h3 := hop (start_distance - h0 - h1 - h2)
  let h4 := hop (start_distance - h0 - h1 - h2 - h3)
  h0 + h1 + h2 + h3 + h4

/-- Final proof statement: after five hops from starting distance of 2, total distance hopped should be 1031769/2359296 -/
theorem kevin_total_hops :
  total_hops 2 5 = 1031769 / 2359296 :=
sorry

end kevin_total_hops_l188_188938


namespace necessary_condition_for_positive_on_interval_l188_188804

theorem necessary_condition_for_positive_on_interval (a b : ℝ) (h : a + 2 * b > 0) :
  (∀ x, 0 ≤ x → x ≤ 1 → (a * x + b) > 0) ↔ ∃ c, 0 < c ∧ c ≤ 1 ∧ a + 2 * b > 0 ∧ ¬∀ d, 0 < d ∧ d ≤ 1 → a * d + b > 0 := 
by 
  sorry

end necessary_condition_for_positive_on_interval_l188_188804


namespace willie_gave_emily_7_stickers_l188_188521

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end willie_gave_emily_7_stickers_l188_188521


namespace sum_of_interior_angles_at_vertex_A_l188_188332

-- Definitions of the interior angles for a square and a regular octagon.
def square_interior_angle : ℝ := 90
def octagon_interior_angle : ℝ := 135

-- Theorem that states the sum of the interior angles at vertex A formed by the square and octagon.
theorem sum_of_interior_angles_at_vertex_A : square_interior_angle + octagon_interior_angle = 225 := by
  sorry

end sum_of_interior_angles_at_vertex_A_l188_188332


namespace minimum_value_f_x_l188_188696

theorem minimum_value_f_x (x : ℝ) (h : 1 < x) : 
  x + (1 / (x - 1)) ≥ 3 :=
sorry

end minimum_value_f_x_l188_188696


namespace eleven_percent_greater_than_seventy_l188_188695

theorem eleven_percent_greater_than_seventy : ∀ x : ℝ, (x = 70 * (1 + 11 / 100)) → (x = 77.7) :=
by
  intro x
  intro h
  sorry

end eleven_percent_greater_than_seventy_l188_188695


namespace karen_age_is_10_l188_188480

-- Definitions for the given conditions
def ages : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def to_park (a b : ℕ) : Prop := a + b = 20
def to_pool (a b : ℕ) : Prop := 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9
def stayed_home (karen_age : ℕ) : Prop := karen_age = 10

-- Theorem stating Karen's age is 10 given the conditions
theorem karen_age_is_10 :
  ∃ (a b c d e f g : ℕ),
  ages = [a, b, c, d, e, f, g] ∧
  ((to_park a b ∨ to_park a c ∨ to_park a d ∨ to_park a e ∨ to_park a f ∨ to_park a g ∨
  to_park b c ∨ to_park b d ∨ to_park b e ∨ to_park b f ∨ to_park b g ∨
  to_park c d ∨ to_park c e ∨ to_park c f ∨ to_park c g ∨
  to_park d e ∨ to_park d f ∨ to_park d g ∨
  to_park e f ∨ to_park e g ∨
  to_park f g)) ∧
  ((to_pool a b ∨ to_pool a c ∨ to_pool a d ∨ to_pool a e ∨ to_pool a f ∨ to_pool a g ∨
  to_pool b c ∨ to_pool b d ∨ to_pool b e ∨ to_pool b f ∨ to_pool b g ∨
  to_pool c d ∨ to_pool c e ∨ to_pool c f ∨
  to_pool d e ∨ to_pool d f ∨
  to_pool e f ∨
  to_pool f g)) ∧
  stayed_home 4 :=
sorry

end karen_age_is_10_l188_188480


namespace trip_duration_l188_188992

/--
Given:
1. The car averages 30 miles per hour for the first 5 hours of the trip.
2. The car averages 42 miles per hour for the rest of the trip.
3. The average speed for the entire trip is 34 miles per hour.

Prove: 
The total duration of the trip is 7.5 hours.
-/
theorem trip_duration (t T : ℝ) (h1 : 150 + 42 * t = 34 * T) (h2 : T = 5 + t) : T = 7.5 :=
by
  sorry

end trip_duration_l188_188992


namespace find_z_l188_188893

open Complex

theorem find_z (z : ℂ) (h : z * (2 - I) = 5 * I) : z = -1 + 2 * I :=
sorry

end find_z_l188_188893


namespace range_of_x_l188_188329

theorem range_of_x (x : ℝ) : (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_x_l188_188329


namespace apples_count_l188_188585

theorem apples_count (n : ℕ) (h₁ : n > 2)
  (h₂ : 144 / n - 144 / (n + 2) = 1) :
  n + 2 = 18 :=
by
  sorry

end apples_count_l188_188585


namespace find_m_of_pure_imaginary_l188_188629

theorem find_m_of_pure_imaginary (m : ℝ) (h1 : (m^2 + m - 2) = 0) (h2 : (m^2 - 1) ≠ 0) : m = -2 :=
by
  sorry

end find_m_of_pure_imaginary_l188_188629


namespace horses_for_camels_l188_188993

noncomputable def cost_of_one_elephant : ℕ := 11000
noncomputable def cost_of_one_ox : ℕ := 7333 -- approx.
noncomputable def cost_of_one_horse : ℕ := 1833 -- approx.
noncomputable def cost_of_one_camel : ℕ := 4400

theorem horses_for_camels (H : ℕ) :
  (H * cost_of_one_horse = cost_of_one_camel) → H = 2 :=
by
  -- skipping proof details
  sorry

end horses_for_camels_l188_188993


namespace john_cakes_bought_l188_188936

-- Conditions
def cake_price : ℕ := 12
def john_paid : ℕ := 18

-- Definition of the total cost
def total_cost : ℕ := 2 * john_paid

-- Calculate number of cakes
def num_cakes (total_cost cake_price : ℕ) : ℕ := total_cost / cake_price

-- Theorem to prove that the number of cakes John Smith bought is 3
theorem john_cakes_bought : num_cakes total_cost cake_price = 3 := by
  sorry

end john_cakes_bought_l188_188936


namespace continuous_at_1_l188_188205

theorem continuous_at_1 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → |(-4 * x^2 - 6) - (-10)| < ε :=
by
  sorry

end continuous_at_1_l188_188205


namespace max_x_value_l188_188346

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 7) (h2 : x * y + x * z + y * z = 12) : x ≤ 1 :=
by sorry

end max_x_value_l188_188346


namespace intersection_point_correct_l188_188180

-- Points in 3D coordinate space
def P : ℝ × ℝ × ℝ := (3, -9, 6)
def Q : ℝ × ℝ × ℝ := (13, -19, 11)
def R : ℝ × ℝ × ℝ := (1, 4, -7)
def S : ℝ × ℝ × ℝ := (3, -6, 9)

-- Vectors for parameterization
def pq_vector (t : ℝ) : ℝ × ℝ × ℝ := (3 + 10 * t, -9 - 10 * t, 6 + 5 * t)
def rs_vector (s : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - 10 * s, -7 + 16 * s)

-- The proof of the intersection point equals the correct answer
theorem intersection_point_correct : 
  ∃ t s : ℝ, pq_vector t = rs_vector s ∧ 
  pq_vector t = (-19 / 3, 10 / 3, 4 / 3) := 
by
  sorry

end intersection_point_correct_l188_188180


namespace determine_pairs_l188_188736

theorem determine_pairs (p q : ℕ) (h : (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q) : (p = 1 ∧ q = 1) ∨ (p = 2 ∧ q = 2) :=
by
  sorry

end determine_pairs_l188_188736


namespace problem_statement_l188_188168

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^2021 + a^2022 = 2 := 
by
  sorry

end problem_statement_l188_188168


namespace combined_time_l188_188410

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l188_188410


namespace unique_face_numbers_l188_188201

-- Define the problem statement and conditions
theorem unique_face_numbers (a b c d e f : ℤ) (sums : list ℤ) (h : sums = [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]) : 
  (∃ (n : ℕ → ℤ), (n 0 = a ∧ n 1 = b ∧ n 2 = c ∧ n 3 = d ∧ n 4 = e ∧ n 5 = f)) :=
by 
  rw h
  -- Additional detailed steps are omitted
  sorry

end unique_face_numbers_l188_188201


namespace total_candies_in_third_set_l188_188681

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l188_188681


namespace no_common_factor_l188_188952

open Polynomial

theorem no_common_factor (f g : ℤ[X]) : f = X^2 + X - 1 → g = X^2 + 2 * X → ∀ d : ℤ[X], d ∣ f ∧ d ∣ g → d = 1 :=
by
  intros h1 h2 d h_dv
  rw [h1, h2] at h_dv
  -- Proof steps would go here
  sorry

end no_common_factor_l188_188952


namespace sum_of_squares_l188_188069

theorem sum_of_squares (x : ℚ) (h : x + 2 * x + 3 * x = 14) : 
  (x^2 + (2 * x)^2 + (3 * x)^2) = 686 / 9 :=
by
  sorry

end sum_of_squares_l188_188069


namespace no_integer_solutions_l188_188026

theorem no_integer_solutions (a : ℕ) (h : a % 4 = 3) : ¬∃ (x y : ℤ), x^2 + y^2 = a := by
  sorry

end no_integer_solutions_l188_188026


namespace Tom_age_problem_l188_188972

theorem Tom_age_problem 
  (T : ℝ) 
  (h1 : T = T1 + T2 + T3 + T4) 
  (h2 : T - 3 = 3 * (T - 3 - 3 - 3 - 3)) : 
  T / 3 = 5.5 :=
by 
  -- sorry here to skip the proof
  sorry

end Tom_age_problem_l188_188972


namespace power_mod_remainder_l188_188246

theorem power_mod_remainder : (3^20) % 7 = 2 :=
by {
  -- condition: 3^6 ≡ 1 (mod 7)
  have h1 : (3^6) % 7 = 1 := by norm_num,
  -- we now use this to show 3^20 ≡ 2 (mod 7)
  calc
    (3^20) % 7 = ((3^6)^3 * 3^2) % 7 : by norm_num
          ... = (1^3 * 3^2) % 7       : by rw [←nat.modeq.modeq_iff_dvd, h1]
          ... =  (3^2) % 7            : by norm_num
          ... = 2                    : by norm_num
}

end power_mod_remainder_l188_188246


namespace one_leg_divisible_by_3_l188_188640

theorem one_leg_divisible_by_3 (a b c : ℕ) (h : a^2 + b^2 = c^2) : (3 ∣ a) ∨ (3 ∣ b) :=
by sorry

end one_leg_divisible_by_3_l188_188640


namespace equation_solution_l188_188821

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l188_188821


namespace convex_polygon_sides_l188_188845

theorem convex_polygon_sides (S : ℝ) (n : ℕ) (a₁ a₂ a₃ a₄ : ℝ) 
    (h₁ : S = 4320) 
    (h₂ : a₁ = 120) 
    (h₃ : a₂ = 120) 
    (h₄ : a₃ = 120) 
    (h₅ : a₄ = 120) 
    (h_sum : S = 180 * (n - 2)) :
    n = 26 :=
by
  sorry

end convex_polygon_sides_l188_188845


namespace log_a_properties_l188_188161

noncomputable def log_a (a x : ℝ) (h : 0 < a ∧ a < 1) : ℝ := Real.log x / Real.log a

theorem log_a_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 1 < x → log_a a x h < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → log_a a x h > 0) ∧
  (¬ ∀ x1 x2 : ℝ, log_a a x1 h > log_a a x2 h → x1 > x2) ∧
  (∀ x y : ℝ, log_a a (x * y) h = log_a a x h + log_a a y h) :=
by
  sorry

end log_a_properties_l188_188161


namespace num_ordered_pairs_l188_188290

theorem num_ordered_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x * y = 4410) : 
  ∃ (n : ℕ), n = 36 :=
sorry

end num_ordered_pairs_l188_188290


namespace percent_decrease_computer_price_l188_188172

theorem percent_decrease_computer_price (price_1990 price_2010 : ℝ) (h1 : price_1990 = 1200) (h2 : price_2010 = 600) :
  ((price_1990 - price_2010) / price_1990) * 100 = 50 := 
  sorry

end percent_decrease_computer_price_l188_188172


namespace problem_statement_l188_188563

variable (m n : ℝ)
noncomputable def sqrt_2_minus_1_inv := (Real.sqrt 2 - 1)⁻¹
noncomputable def sqrt_2_plus_1_inv := (Real.sqrt 2 + 1)⁻¹

theorem problem_statement 
  (hm : m = sqrt_2_minus_1_inv) 
  (hn : n = sqrt_2_plus_1_inv) : 
  m + n = 2 * Real.sqrt 2 := 
sorry

end problem_statement_l188_188563


namespace fresh_grapes_water_percentage_l188_188294

/--
Given:
- Fresh grapes contain a certain percentage (P%) of water by weight.
- Dried grapes contain 25% water by weight.
- The weight of dry grapes obtained from 200 kg of fresh grapes is 66.67 kg.

Prove:
- The percentage of water (P) in fresh grapes is 75%.
-/
theorem fresh_grapes_water_percentage
  (P : ℝ) (H1 : ∃ P, P / 100 * 200 = 0.75 * 66.67) :
  P = 75 :=
sorry

end fresh_grapes_water_percentage_l188_188294


namespace only_integer_solution_l188_188735

theorem only_integer_solution (n : ℕ) (h1 : n > 1) (h2 : (2 * n + 1) % n ^ 2 = 0) : n = 3 := 
sorry

end only_integer_solution_l188_188735


namespace marie_days_to_pay_cash_register_l188_188489

def daily_revenue_bread (loaves: Nat) (price_per_loaf: Nat) : Nat := loaves * price_per_loaf
def daily_revenue_cakes (cakes: Nat) (price_per_cake: Nat) : Nat := cakes * price_per_cake
def total_daily_revenue (loaves: Nat) (price_per_loaf: Nat) (cakes: Nat) (price_per_cake: Nat) : Nat :=
  daily_revenue_bread loaves price_per_loaf + daily_revenue_cakes cakes price_per_cake

def daily_expenses (rent: Nat) (electricity: Nat) : Nat := rent + electricity

def daily_profit (loaves: Nat) (price_per_loaf: Nat) (cakes: Nat) (price_per_cake: Nat) (rent: Nat) (electricity: Nat) : Nat :=
  total_daily_revenue loaves price_per_loaf cakes price_per_cake - daily_expenses rent electricity

def days_to_pay_cash_register (register_cost: Nat) (profit: Nat) : Nat :=
  register_cost / profit

theorem marie_days_to_pay_cash_register :
  days_to_pay_cash_register 1040 (daily_profit 40 2 6 12 20 2) = 8 :=
by
  calc
    days_to_pay_cash_register 1040 (daily_profit 40 2 6 12 20 2)
        = 1040 / daily_profit 40 2 6 12 20 2 : by rfl
    ... = 1040 / 130 : by rfl
    ... = 8 : by rfl

end marie_days_to_pay_cash_register_l188_188489


namespace pirate_finds_treasure_no_traps_l188_188705

noncomputable def pirate_prob : ℚ :=
  let probability_treasure : ℚ := 1 / 5
  let probability_traps : ℚ := 1 / 10
  let probability_neither : ℚ := 7 / 10
  let total_islands := 8
  let successful_islands := 4
  let comb := Nat.choose total_islands successful_islands
  comb * (probability_treasure ^ successful_islands) * (probability_neither ^ (total_islands - successful_islands))

theorem pirate_finds_treasure_no_traps :
  pirate_prob = 33614 / 1250000 :=
by
  sorry
 
end pirate_finds_treasure_no_traps_l188_188705


namespace range_of_y_div_x_l188_188782

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + (y-3)^2 = 1) : 
  (∃ k : ℝ, k = y / x ∧ (k ≤ -2 * Real.sqrt 2 ∨ k ≥ 2 * Real.sqrt 2)) :=
sorry

end range_of_y_div_x_l188_188782


namespace women_fraction_half_l188_188014

theorem women_fraction_half
  (total_people : ℕ)
  (married_fraction : ℝ)
  (max_unmarried_women : ℕ)
  (total_people_eq : total_people = 80)
  (married_fraction_eq : married_fraction = 1 / 2)
  (max_unmarried_women_eq : max_unmarried_women = 32) :
  (∃ (women_fraction : ℝ), women_fraction = 1 / 2) :=
by
  sorry

end women_fraction_half_l188_188014


namespace third_set_candies_l188_188658

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l188_188658


namespace remainder_of_3_pow_20_mod_7_l188_188248

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l188_188248


namespace letters_identity_l188_188215

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l188_188215


namespace one_cubic_foot_is_1728_cubic_inches_l188_188917

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l188_188917


namespace value_of_x_squared_plus_reciprocal_squared_l188_188448

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : x^4 + (1 / x^4) = 23) :
  x^2 + (1 / x^2) = 5 := by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l188_188448


namespace no_solution_for_x_l188_188780

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, (1 / (x - 4)) + (m / (x + 4)) ≠ ((m + 3) / (x^2 - 16))) ↔ (m = -1 ∨ m = 5 ∨ m = -1 / 3) :=
sorry

end no_solution_for_x_l188_188780


namespace number_of_participants_l188_188835

theorem number_of_participants (total_gloves : ℕ) (gloves_per_participant : ℕ)
  (h : total_gloves = 126) (h' : gloves_per_participant = 2) : 
  (total_gloves / gloves_per_participant = 63) :=
by
  sorry

end number_of_participants_l188_188835


namespace expression_equals_required_value_l188_188855

-- Define the expression as needed
def expression : ℚ := (((((4 + 2)⁻¹ + 2)⁻¹) + 2)⁻¹) + 2

-- Define the theorem stating that the expression equals the required value
theorem expression_equals_required_value : 
  expression = 77 / 32 := 
sorry

end expression_equals_required_value_l188_188855


namespace water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l188_188372

-- Define the tiered water pricing function
def tiered_water_cost (m : ℕ) : ℝ :=
  if m ≤ 20 then
    1.6 * m
  else if m ≤ 30 then
    1.6 * 20 + 2.4 * (m - 20)
  else
    1.6 * 20 + 2.4 * 10 + 4.8 * (m - 30)

-- Problem 1
theorem water_cost_10_tons : tiered_water_cost 10 = 16 := 
sorry

-- Problem 2
theorem water_cost_27_tons : tiered_water_cost 27 = 48.8 := 
sorry

-- Problem 3
theorem water_cost_between_20_30 (m : ℕ) (h : 20 < m ∧ m < 30) : tiered_water_cost m = 2.4 * m - 16 := 
sorry

-- Problem 4
theorem water_cost_above_30 (m : ℕ) (h : m > 30) : tiered_water_cost m = 4.8 * m - 88 := 
sorry

end water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l188_188372


namespace subcommittee_ways_l188_188829

theorem subcommittee_ways :
  ∃ (n : ℕ), n = Nat.choose 10 4 * Nat.choose 7 2 ∧ n = 4410 :=
by
  use 4410
  sorry

end subcommittee_ways_l188_188829


namespace remainder_3_pow_20_mod_7_l188_188243

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l188_188243


namespace area_increase_correct_l188_188097

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l188_188097


namespace equal_animals_per_aquarium_l188_188767

theorem equal_animals_per_aquarium (aquariums animals : ℕ) (h1 : aquariums = 26) (h2 : animals = 52) (h3 : ∀ a, a = animals / aquariums) : a = 2 := 
by
  sorry

end equal_animals_per_aquarium_l188_188767


namespace mixed_oil_rate_l188_188857

noncomputable def rate_of_mixed_oil
  (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℚ :=
(total_cost : ℚ) / (total_volume : ℚ)
where
  total_cost := volume1 * price1 + volume2 * price2
  total_volume := volume1 + volume2

theorem mixed_oil_rate :
  rate_of_mixed_oil 10 50 5 66 = 55.33 := 
by
  sorry

end mixed_oil_rate_l188_188857


namespace perfect_matching_exists_l188_188603

-- Define the set of boys and girls
def boys : Finset ℕ := Finset.range 10
def girls : Finset ℕ := Finset.range 10

-- Define the friendship relation as a set of pairs
variable (friends : ℕ → Finset ℕ)

-- Condition: For each 1 ≤ k ≤ 10 and for each group of k boys, the number of girls
-- who are friends with at least one boy in the group is not less than k.
def friendship_condition : Prop :=
  ∀ (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 10) (b : Finset ℕ) (hb : b.card = k ∧ b ⊆ boys),
    (b.bUnion friends).card ≥ k

-- The theorem to be proven
theorem perfect_matching_exists (friends : ℕ → Finset ℕ) (h : friendship_condition friends) :
  ∃ (matching : boys ↪ girls), ∀ b ∈ boys, (friends b).count (matching b) ≥ 1 := 
by
  sorry

end perfect_matching_exists_l188_188603


namespace no_30_cents_l188_188955

/-- Given six coins selected from nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total value of the six coins cannot be 30 cents or less. -/
theorem no_30_cents {n d q : ℕ} (h : n + d + q = 6) (hn : n * 5 + d * 10 + q * 25 <= 30) : false :=
by
  sorry

end no_30_cents_l188_188955


namespace find_rate_per_kg_mangoes_l188_188133

noncomputable def rate_per_kg_mangoes
  (cost_grapes_rate : ℕ)
  (quantity_grapes : ℕ)
  (quantity_mangoes : ℕ)
  (total_paid : ℕ)
  (rate_grapes : ℕ)
  (rate_mangoes : ℕ) :=
  total_paid = (rate_grapes * quantity_grapes) + (rate_mangoes * quantity_mangoes)

theorem find_rate_per_kg_mangoes :
  rate_per_kg_mangoes 70 8 11 1165 70 55 :=
by
  sorry

end find_rate_per_kg_mangoes_l188_188133


namespace find_values_l188_188638

theorem find_values (a b c : ℤ)
  (h1 : ∀ x, x^2 + 9 * x + 14 = (x + a) * (x + b))
  (h2 : ∀ x, x^2 + 4 * x - 21 = (x + b) * (x - c)) :
  a + b + c = 12 :=
sorry

end find_values_l188_188638


namespace sum_of_first_6n_integers_l188_188783

theorem sum_of_first_6n_integers (n : ℕ) (h1 : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_6n_integers_l188_188783


namespace find_x_for_g_inv_eq_3_l188_188312

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem find_x_for_g_inv_eq_3 : ∃ x : ℝ, g x = 113 :=
by
  exists 3
  unfold g
  norm_num

end find_x_for_g_inv_eq_3_l188_188312


namespace contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l188_188515

variable {A B : Prop}

def contrary (A : Prop) : Prop := A ∧ ¬A
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B)

theorem contrary_implies_mutually_exclusive (A : Prop) : contrary A → mutually_exclusive A (¬A) :=
by sorry

theorem contrary_sufficient_but_not_necessary (A B : Prop) :
  (∃ (A : Prop), contrary A) → mutually_exclusive A B →
  (∃ (A : Prop), contrary A ∧ mutually_exclusive A B) :=
by sorry

end contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l188_188515


namespace press_x_squared_three_times_to_exceed_10000_l188_188701

theorem press_x_squared_three_times_to_exceed_10000 :
  ∃ (n : ℕ), n = 3 ∧ (5^(2^n) > 10000) :=
by
  sorry

end press_x_squared_three_times_to_exceed_10000_l188_188701


namespace sequence_general_formula_l188_188601

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 12)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) :
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 12 :=
sorry

end sequence_general_formula_l188_188601


namespace tan_ratio_l188_188342

theorem tan_ratio (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5 / 8)
  (h2 : Real.sin (a - b) = 1 / 4) : 
  Real.tan a / Real.tan b = 7 / 3 := 
sorry

end tan_ratio_l188_188342


namespace unique_two_digit_u_l188_188511

theorem unique_two_digit_u:
  ∃! u : ℤ, 10 ≤ u ∧ u < 100 ∧ 
            (15 * u) % 100 = 45 ∧ 
            u % 17 = 7 :=
by
  -- To be completed in proof
  sorry

end unique_two_digit_u_l188_188511


namespace problem_projection_eq_l188_188315

variable (m n : ℝ × ℝ)
variable (m_val : m = (1, 2))
variable (n_val : n = (2, 3))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (dot_product u v) / (magnitude v)

theorem problem_projection_eq : projection m n = (8 * Real.sqrt 13) / 13 :=
by
  rw [m_val, n_val]
  sorry

end problem_projection_eq_l188_188315


namespace willie_gave_emily_7_stickers_l188_188522

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end willie_gave_emily_7_stickers_l188_188522


namespace tim_kittens_l188_188236

theorem tim_kittens (K : ℕ) (h1 : (3 / 5 : ℚ) * (2 / 3 : ℚ) * K = 12) : K = 30 :=
sorry

end tim_kittens_l188_188236


namespace solve_congruences_l188_188142

theorem solve_congruences :
  ∃ x : ℤ, 
  x ≡ 3 [ZMOD 7] ∧ 
  x^2 ≡ 44 [ZMOD 49] ∧ 
  x^3 ≡ 111 [ZMOD 343] ∧ 
  x ≡ 17 [ZMOD 343] :=
sorry

end solve_congruences_l188_188142


namespace solve_equation_l188_188819

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l188_188819


namespace system_solution_l188_188147

theorem system_solution (m n : ℝ) (h1 : -2 * m * 5 + 5 * 2 = 15) (h2 : 5 + 7 * n * 2 = 14) :
  ∃ (a b : ℝ), (-2 * m * (a + b) + 5 * (a - 2 * b) = 15) ∧ ((a + b) + 7 * n * (a - 2 * b) = 14) ∧ (a = 4) ∧ (b = 1) :=
by
  -- The proof is intentionally omitted
  sorry

end system_solution_l188_188147


namespace production_time_l188_188256

variable (a m : ℝ) -- Define a and m as real numbers

-- State the problem as a theorem in Lean
theorem production_time : (a / m) * 200 = 200 * (a / m) := by
  sorry

end production_time_l188_188256


namespace largest_prime_divisor_25_sq_plus_72_sq_l188_188143

theorem largest_prime_divisor_25_sq_plus_72_sq : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (25^2 + 72^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (25^2 + 72^2) → q ≤ p :=
sorry

end largest_prime_divisor_25_sq_plus_72_sq_l188_188143


namespace tangent_line_ln_l188_188146

theorem tangent_line_ln (b : ℝ) :
  (∀ x > 0, ∀ y, y = Real.log x → (deriv (λ x : ℝ, Real.log x) x = 1 / x)) →
  (∃ x > 0, ∃ y, y = Real.log x ∧ y = 1 / 2 * x + b) →
  b = Real.log 2 - 1 :=
by
  sorry

end tangent_line_ln_l188_188146


namespace identity_of_letters_l188_188214

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l188_188214


namespace total_animals_l188_188797

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end total_animals_l188_188797


namespace range_of_a_l188_188443

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a < 0 ∨ (1/4 < a ∧ a < 4) := 
sorry

end range_of_a_l188_188443


namespace piggy_bank_dimes_l188_188121

theorem piggy_bank_dimes (q d : ℕ) 
  (h1 : q + d = 100) 
  (h2 : 25 * q + 10 * d = 1975) : 
  d = 35 :=
by
  -- skipping the proof
  sorry

end piggy_bank_dimes_l188_188121


namespace total_amount_received_l188_188278

-- Define the initial prices and the increases
def initial_price_tv : ℝ := 500
def increase_ratio_tv : ℝ := 2/5
def initial_price_phone : ℝ := 400
def increase_ratio_phone : ℝ := 0.4

-- Calculate the total amount received
theorem total_amount_received : 
  initial_price_tv + increase_ratio_tv * initial_price_tv + initial_price_phone + increase_ratio_phone * initial_price_phone = 1260 :=
by {
  sorry
}

end total_amount_received_l188_188278


namespace school_election_votes_l188_188612

theorem school_election_votes (E S R L : ℕ)
  (h1 : E = 2 * S)
  (h2 : E = 4 * R)
  (h3 : S = 5 * R)
  (h4 : S = 3 * L)
  (h5 : R = 16) :
  E = 64 ∧ S = 80 ∧ R = 16 ∧ L = 27 := by
  sorry

end school_election_votes_l188_188612


namespace arithmetic_sequence_a9_l188_188568

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + d * (n - 1)

-- The sum of the first n terms of an arithmetic sequence.
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a9
  (a1 d : ℤ)
  (h1 : a1 + (a1 + d)^2 = -3)
  (h2 : S 5 a1 d = 10) :
  a 9 a1 d = 20 :=
begin
  sorry
end

end arithmetic_sequence_a9_l188_188568


namespace probability_of_johns_8th_roll_l188_188007

noncomputable def probability_johns_8th_roll_is_last : ℚ :=
  (7/8)^6 * (1/8)

theorem probability_of_johns_8th_roll :
  probability_johns_8th_roll_is_last = 117649 / 2097152 := by
  sorry

end probability_of_johns_8th_roll_l188_188007


namespace kayla_apples_l188_188622

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end kayla_apples_l188_188622


namespace product_squared_inequality_l188_188646

theorem product_squared_inequality (n : ℕ) (a : Fin n → ℝ) (h : (Finset.univ.prod (λ i => a i)) = 1) :
    (Finset.univ.prod (λ i => (1 + (a i)^2))) ≥ 2^n := 
sorry

end product_squared_inequality_l188_188646


namespace darnel_lap_difference_l188_188547

theorem darnel_lap_difference (sprint jog : ℝ) (h_sprint : sprint = 0.88) (h_jog : jog = 0.75) : sprint - jog = 0.13 := 
by 
  rw [h_sprint, h_jog] 
  norm_num

end darnel_lap_difference_l188_188547


namespace new_releases_fraction_is_2_over_5_l188_188984

def fraction_new_releases (total_books : ℕ) (frac_historical_fiction : ℚ) (frac_new_historical_fiction : ℚ) (frac_new_non_historical_fiction : ℚ) : ℚ :=
  let num_historical_fiction := frac_historical_fiction * total_books
  let num_new_historical_fiction := frac_new_historical_fiction * num_historical_fiction
  let num_non_historical_fiction := total_books - num_historical_fiction
  let num_new_non_historical_fiction := frac_new_non_historical_fiction * num_non_historical_fiction
  let total_new_releases := num_new_historical_fiction + num_new_non_historical_fiction
  num_new_historical_fiction / total_new_releases

theorem new_releases_fraction_is_2_over_5 :
  ∀ (total_books : ℕ), total_books > 0 →
    fraction_new_releases total_books (40 / 100) (40 / 100) (40 / 100) = 2 / 5 :=
by 
  intro total_books h
  sorry

end new_releases_fraction_is_2_over_5_l188_188984


namespace meters_examined_l188_188128

theorem meters_examined (x : ℝ) (h1 : 0.07 / 100 * x = 2) : x = 2857 :=
by
  -- using the given setup and simplification
  sorry

end meters_examined_l188_188128


namespace pencils_calculation_l188_188878

variable (C B D : ℕ)

theorem pencils_calculation : 
  (C = B + 5) ∧
  (B = 2 * D - 3) ∧
  (C = 20) →
  D = 9 :=
by sorry

end pencils_calculation_l188_188878


namespace problem1_l188_188287

variable (m : ℤ)

theorem problem1 : m * (m - 3) + 3 * (3 - m) = (m - 3) ^ 2 := by
  sorry

end problem1_l188_188287


namespace yellow_ball_count_l188_188322

def total_balls : ℕ := 500
def red_balls : ℕ := total_balls / 3
def remaining_after_red : ℕ := total_balls - red_balls
def blue_balls : ℕ := remaining_after_red / 5
def remaining_after_blue : ℕ := remaining_after_red - blue_balls
def green_balls : ℕ := remaining_after_blue / 4
def yellow_balls : ℕ := total_balls - (red_balls + blue_balls + green_balls)

theorem yellow_ball_count : yellow_balls = 201 := by
  sorry

end yellow_ball_count_l188_188322


namespace problem_statement_l188_188877

theorem problem_statement : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end problem_statement_l188_188877


namespace total_candies_in_third_set_l188_188680

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l188_188680


namespace union_of_A_and_B_l188_188623

/-- Given sets A and B defined as follows: A = {x | -1 <= x <= 3} and B = {x | 0 < x < 4}.
Prove that their union A ∪ B is the interval [-1, 4). -/
theorem union_of_A_and_B :
  let A := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
  let B := {x : ℝ | 0 < x ∧ x < 4}
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} :=
by
  sorry

end union_of_A_and_B_l188_188623


namespace product_of_1101_2_and_202_3_is_260_l188_188285

   /-- Convert a binary string to its decimal value -/
   def binary_to_decimal (b : String) : ℕ :=
     b.foldl (λ acc bit, acc * 2 + bit.toNat - '0'.toNat) 0

   /-- Convert a ternary string to its decimal value -/
   def ternary_to_decimal (t : String) : ℕ :=
     t.foldl (λ acc bit, acc * 3 + bit.toNat - '0'.toNat) 0

   theorem product_of_1101_2_and_202_3_is_260 :
     binary_to_decimal "1101" * ternary_to_decimal "202" = 260 :=
   by
     calc
       binary_to_decimal "1101" = 13 : by rfl
       ternary_to_decimal "202" = 20 : by rfl
       13 * 20 = 260 : by rfl
   
end product_of_1101_2_and_202_3_is_260_l188_188285


namespace triangle_side_range_l188_188231

theorem triangle_side_range (a : ℝ) :
  1 < a ∧ a < 4 ↔ 3 + (2 * a - 1) > 4 ∧ 3 + 4 > 2 * a - 1 ∧ 4 + (2 * a - 1) > 3 :=
by
  sorry

end triangle_side_range_l188_188231


namespace value_of_mathematics_l188_188836

def letter_value (n : ℕ) : ℤ :=
  -- The function to assign values based on position modulo 8
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 0 => 0
  | _ => 0 -- This case is practically unreachable

def letter_position (c : Char) : ℕ :=
  -- The function to find the position of a character in the alphabet
  c.toNat - 'a'.toNat + 1

def value_of_word (word : String) : ℤ :=
  -- The function to calculate the sum of values of letters in the word
  word.foldr (fun c acc => acc + letter_value (letter_position c)) 0

theorem value_of_mathematics : value_of_word "mathematics" = 6 := 
  by
    sorry -- Proof to be completed

end value_of_mathematics_l188_188836


namespace largest_power_of_three_in_s_l188_188539

noncomputable def q : ℝ :=
∑ k in Finset.range 10, (k + 1 : ℝ) * Real.log (k + 1)

noncomputable def s : ℝ := Real.exp q

theorem largest_power_of_three_in_s :
  ∃ n : ℕ, s = 3^27 * n ∧ ∀ m : ℕ, s = 3^m * n → m ≤ 27 :=
sorry

end largest_power_of_three_in_s_l188_188539


namespace no_real_solutions_for_equation_l188_188138

theorem no_real_solutions_for_equation (x : ℝ) :
  y = 3 * x ∧ y = (x^3 - 8) / (x - 2) → false :=
by {
  sorry
}

end no_real_solutions_for_equation_l188_188138


namespace probability_of_exactly_5_calls_probability_of_no_more_than_4_calls_probability_of_at_least_3_calls_l188_188405

noncomputable def number_of_subscribers : ℕ := 400
noncomputable def prob_of_call : ℝ := 0.01
noncomputable def mean_calls : ℝ := number_of_subscribers * prob_of_call

theorem probability_of_exactly_5_calls :
  Probability.Poisson.mean mean_calls (5) = 0.1563 :=
sorry

theorem probability_of_no_more_than_4_calls :
  (Probability.Poisson.mean mean_calls (0) +
   Probability.Poisson.mean mean_calls (1) +
   Probability.Poisson.mean mean_calls (2) +
   Probability.Poisson.mean mean_calls (3) +
   Probability.Poisson.mean mean_calls (4)) = 0.6289 :=
sorry

theorem probability_of_at_least_3_calls :
  (1 - (Probability.Poisson.mean mean_calls (0) +
       Probability.Poisson.mean mean_calls (1) +
       Probability.Poisson.mean mean_calls (2))) = 0.7619 :=
sorry

end probability_of_exactly_5_calls_probability_of_no_more_than_4_calls_probability_of_at_least_3_calls_l188_188405


namespace letters_identity_l188_188208

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l188_188208


namespace possible_final_state_l188_188790

-- Definitions of initial conditions and operations
def initial_urn : (ℕ × ℕ) := (100, 100)  -- (W, B)

-- Define operations that describe changes in (white, black) marbles
inductive Operation
| operation1 : Operation
| operation2 : Operation
| operation3 : Operation
| operation4 : Operation

def apply_operation (op : Operation) (state : ℕ × ℕ) : ℕ × ℕ :=
  match op with
  | Operation.operation1 => (state.1, state.2 - 2)
  | Operation.operation2 => (state.1, state.2 - 1)
  | Operation.operation3 => (state.1, state.2 - 1)
  | Operation.operation4 => (state.1 - 2, state.2 + 1)

-- The final state in the form of the specific condition to prove.
def final_state (state : ℕ × ℕ) : Prop :=
  state = (2, 0)  -- 2 white marbles are an expected outcome.

-- Statement of the problem in Lean
theorem possible_final_state : ∃ (sequence : List Operation), 
  (sequence.foldl (fun state op => apply_operation op state) initial_urn).1 = 2 :=
sorry

end possible_final_state_l188_188790


namespace new_ratio_of_partners_to_associates_l188_188399

theorem new_ratio_of_partners_to_associates
  (partners associates : ℕ)
  (rat_partners_associates : 2 * associates = 63 * partners)
  (partners_count : partners = 18)
  (add_assoc : associates + 45 = 612) :
  (partners:ℚ) / (associates + 45) = 1 / 34 :=
by
  -- Actual proof goes here
  sorry

end new_ratio_of_partners_to_associates_l188_188399


namespace baseball_team_groups_l188_188229

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) (h_new : new_players = 48) (h_return : returning_players = 6) (h_per_group : players_per_group = 6) : (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end baseball_team_groups_l188_188229


namespace ratio_of_newspapers_l188_188279

theorem ratio_of_newspapers (C L : ℕ) (h1 : C = 42) (h2 : L = C + 23) : C / (C + 23) = 42 / 65 := by
  sorry

end ratio_of_newspapers_l188_188279


namespace inverse_proportion_range_l188_188313

theorem inverse_proportion_range (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (y = (m + 5) / x) → ((x > 0 → y < 0) ∧ (x < 0 → y > 0))) →
  m < -5 :=
by
  intros h
  -- Skipping proof with sorry as specified
  sorry

end inverse_proportion_range_l188_188313


namespace binom_20_19_equals_20_l188_188719

theorem binom_20_19_equals_20 : nat.choose 20 19 = 20 := 
by 
  sorry

end binom_20_19_equals_20_l188_188719


namespace symmetry_condition_l188_188643

theorem symmetry_condition 
  (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : 
  (∀ a b : ℝ, b = 2 * a → (∃ y, y = (p * (b/2) + 2*q) / (r * (b/2) + 2*s) ∧  b = 2*(y/2) )) → 
  p + r = 0 :=
by
  sorry

end symmetry_condition_l188_188643


namespace sufficient_not_necessary_condition_l188_188625

variable {a : ℝ}

theorem sufficient_not_necessary_condition (ha : a > 1 / a^2) :
  a^2 > 1 / a ∧ ∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2) :=
by
  sorry

end sufficient_not_necessary_condition_l188_188625


namespace sons_ages_l188_188703

theorem sons_ages (m n : ℕ) (h : m * n + m + n = 34) : 
  (m = 4 ∧ n = 6) ∨ (m = 6 ∧ n = 4) :=
sorry

end sons_ages_l188_188703


namespace sqrt_operations_correctness_l188_188082

open Real

theorem sqrt_operations_correctness :
  (sqrt 2 + sqrt 3 ≠ sqrt 5) ∧
  (sqrt (2/3) * sqrt 6 = 2) ∧
  (sqrt 9 = 3) ∧
  (sqrt ((-6) ^ 2) = 6) :=
by
  sorry

end sqrt_operations_correctness_l188_188082


namespace frank_picked_apples_l188_188747

theorem frank_picked_apples (F : ℕ) 
  (susan_picked : ℕ := 3 * F) 
  (susan_left : ℕ := susan_picked / 2) 
  (frank_left : ℕ := 2 * F / 3) 
  (total_left : susan_left + frank_left = 78) : 
  F = 36 :=
sorry

end frank_picked_apples_l188_188747


namespace travel_time_l188_188850

-- Definitions of the conditions
variables (x : ℝ) (speed_elder speed_younger : ℝ)
variables (time_elder_total time_younger_total : ℝ)

def elder_speed_condition : Prop := speed_elder = x
def younger_speed_condition : Prop := speed_younger = x - 4
def elder_distance : Prop := 42 / speed_elder + 1 = time_elder_total
def younger_distance : Prop := 42 / speed_younger + 1 / 3 = time_younger_total

-- The main theorem we want to prove
theorem travel_time : ∀ (x : ℝ), 
  elder_speed_condition x speed_elder → 
  younger_speed_condition x speed_younger → 
  elder_distance speed_elder time_elder_total → 
  younger_distance speed_younger time_younger_total → 
  time_elder_total = time_younger_total ∧ time_elder_total = (10 / 3) :=
sorry

end travel_time_l188_188850


namespace total_candies_in_third_set_l188_188677

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l188_188677


namespace horizontal_distance_l188_188862

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Condition: y-coordinate of point P is 8
def P_y : ℝ := 8

-- Condition: y-coordinate of point Q is -8
def Q_y : ℝ := -8

-- x-coordinates of points P and Q solve these equations respectively
def P_satisfies (x : ℝ) : Prop := curve x = P_y
def Q_satisfies (x : ℝ) : Prop := curve x = Q_y

-- The horizontal distance between P and Q is 1
theorem horizontal_distance : ∃ (Px Qx : ℝ), P_satisfies Px ∧ Q_satisfies Qx ∧ |Px - Qx| = 1 :=
by
  sorry

end horizontal_distance_l188_188862


namespace finitely_many_negative_terms_l188_188282

theorem finitely_many_negative_terms (A : ℝ) :
  (∀ (x : ℕ → ℝ), (∀ n, x n ≠ 0) ∧ (∀ n, x (n+1) = A - 1 / x n) →
  (∃ N, ∀ n ≥ N, x n ≥ 0)) ↔ A ≥ 2 :=
sorry

end finitely_many_negative_terms_l188_188282


namespace arithmetic_mean_of_18_27_45_l188_188682

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_of_18_27_45_l188_188682


namespace mod_37_5_l188_188964

theorem mod_37_5 : 37 % 5 = 2 :=
by
  sorry

end mod_37_5_l188_188964


namespace geometric_sequence_a6_l188_188789

variable {α : Type} [LinearOrderedSemiring α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ a₁ q : α, ∀ n, a n = a₁ * q ^ n

theorem geometric_sequence_a6 
  (a : ℕ → α) 
  (h_seq : is_geometric_sequence a) 
  (h1 : a 2 + a 4 = 20) 
  (h2 : a 3 + a 5 = 40) : 
  a 6 = 64 :=
by
  sorry

end geometric_sequence_a6_l188_188789


namespace class_raised_initial_amount_l188_188810

/-- Miss Grayson's class raised some money for their field trip.
Each student contributed $5 each.
There are 20 students in her class.
The cost of the trip is $7 for each student.
After all the field trip costs were paid, there is $10 left in Miss Grayson's class fund.
Prove that the class initially raised $150 for the field trip. -/
theorem class_raised_initial_amount
  (students : ℕ)
  (contribution_per_student : ℕ)
  (cost_per_student : ℕ)
  (remaining_fund : ℕ)
  (total_students : students = 20)
  (per_student_contribution : contribution_per_student = 5)
  (per_student_cost : cost_per_student = 7)
  (remaining_amount : remaining_fund = 10) :
  (students * contribution_per_student + remaining_fund) = 150 := 
sorry

end class_raised_initial_amount_l188_188810


namespace determine_a_if_slope_angle_is_45_degrees_l188_188468

-- Define the condition that the slope angle of the given line is 45°
def is_slope_angle_45_degrees (a : ℝ) : Prop :=
  let m := -a / (2 * a - 3)
  m = 1

-- State the theorem we need to prove
theorem determine_a_if_slope_angle_is_45_degrees (a : ℝ) :
  is_slope_angle_45_degrees a → a = 1 :=
by
  intro h
  sorry

end determine_a_if_slope_angle_is_45_degrees_l188_188468


namespace net_income_after_tax_l188_188379

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end net_income_after_tax_l188_188379


namespace shaded_region_area_l188_188485

structure Point where
  x : ℝ
  y : ℝ

def W : Point := ⟨0, 0⟩
def X : Point := ⟨5, 0⟩
def Y : Point := ⟨5, 2⟩
def Z : Point := ⟨0, 2⟩
def Q : Point := ⟨1, 0⟩
def S : Point := ⟨5, 0.5⟩
def R : Point := ⟨0, 1⟩
def D : Point := ⟨1, 2⟩

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y) - (B.x * A.y + C.x * B.y + A.x * C.y)|

theorem shaded_region_area : triangle_area R D Y = 1 := by
  sorry

end shaded_region_area_l188_188485


namespace minuend_calculation_l188_188786

theorem minuend_calculation (subtrahend difference : ℕ) (h : subtrahend + difference + 300 = 600) :
  300 = 300 :=
sorry

end minuend_calculation_l188_188786


namespace net_income_after_tax_l188_188378

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end net_income_after_tax_l188_188378


namespace garden_area_increase_l188_188114

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l188_188114


namespace agatha_remaining_amount_l188_188709

theorem agatha_remaining_amount :
  let initial_amount := 60
  let frame_price := 15
  let frame_discount := 0.10 * frame_price
  let frame_final := frame_price - frame_discount
  let wheel_price := 25
  let wheel_discount := 0.05 * wheel_price
  let wheel_final := wheel_price - wheel_discount
  let seat_price := 8
  let seat_discount := 0.15 * seat_price
  let seat_final := seat_price - seat_discount
  let tape_price := 5
  let total_spent := frame_final + wheel_final + seat_final + tape_price
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 10.95 :=
by
  sorry

end agatha_remaining_amount_l188_188709


namespace material_for_one_pillowcase_l188_188073

def material_in_first_bale (x : ℝ) : Prop :=
  4 * x + 1100 = 5000

def material_in_third_bale : ℝ := 0.22 * 5000

def total_material_used_for_producing_items (x y : ℝ) : Prop :=
  150 * (y + 3.25) + 240 * y = x

theorem material_for_one_pillowcase :
  ∀ (x y : ℝ), 
    material_in_first_bale x → 
    material_in_third_bale = 1100 → 
    (x = 975) → 
    total_material_used_for_producing_items x y →
    y = 1.25 :=
by
  intro x y h1 h2 h3 h4
  rw [h3] at h4
  have : 150 * (y + 3.25) + 240 * y = 975 := h4
  sorry

end material_for_one_pillowcase_l188_188073


namespace remainder_7531_mod_11_is_5_l188_188079

theorem remainder_7531_mod_11_is_5 :
  let n := 7531
  let m := 7 + 5 + 3 + 1
  n % 11 = 5 ∧ m % 11 = 5 :=
by
  let n := 7531
  let m := 7 + 5 + 3 + 1
  have h : n % 11 = m % 11 := sorry  -- by property of digits sum mod
  have hm : m % 11 = 5 := sorry      -- calculation
  exact ⟨h, hm⟩

end remainder_7531_mod_11_is_5_l188_188079


namespace bowling_ball_weight_l188_188557

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end bowling_ball_weight_l188_188557


namespace race_distance_between_Sasha_and_Kolya_l188_188053

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l188_188053


namespace marie_needs_8_days_to_pay_for_cash_register_l188_188490

-- Definitions of the conditions
def cost_of_cash_register : ℕ := 1040
def price_per_loaf : ℕ := 2
def loaves_per_day : ℕ := 40
def price_per_cake : ℕ := 12
def cakes_per_day : ℕ := 6
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2

-- Derive daily income and expenses
def daily_income : ℕ := (price_per_loaf * loaves_per_day) + (price_per_cake * cakes_per_day)
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit : ℕ := daily_income - daily_expenses

-- Define days needed to pay for the cash register
def days_needed : ℕ := cost_of_cash_register / daily_profit

-- Proof goal
theorem marie_needs_8_days_to_pay_for_cash_register : days_needed = 8 := by
  sorry

end marie_needs_8_days_to_pay_for_cash_register_l188_188490


namespace oranges_in_bowl_l188_188174

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end oranges_in_bowl_l188_188174


namespace area_increase_correct_l188_188094

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l188_188094


namespace billy_ate_72_cherries_l188_188874

-- Definitions based on conditions:
def initial_cherries : Nat := 74
def remaining_cherries : Nat := 2

-- Problem: How many cherries did Billy eat?
def cherries_eaten := initial_cherries - remaining_cherries

theorem billy_ate_72_cherries : cherries_eaten = 72 :=
by
  -- proof here
  sorry

end billy_ate_72_cherries_l188_188874


namespace isosceles_triangle_angle_Q_l188_188513

theorem isosceles_triangle_angle_Q (x : ℝ) (PQR : Triangle)
  (h1 : PQR.angles Q = PQR.angles R)
  (h2 : PQR.angles R = 5 * PQR.angles P)
  (sum_angles : PQR.angles P + PQR.angles Q + PQR.angles R = 180) :
  PQR.angles Q = 900 / 11 :=
by
  sorry

end isosceles_triangle_angle_Q_l188_188513


namespace circles_coincide_S_touches_midlines_S_l188_188943

noncomputable def triangle_incircle (ABC : Triangle) : Circle := sorry
noncomputable def homothety (c : ℝ) (p : Point) (S : Circle) : Circle := sorry
noncomputable def nagel_point (ABC : Triangle) : Point := sorry
noncomputable def centroid (ABC : Triangle) : Point := sorry
noncomputable def midlines_touching (S : Circle) (ABC : Triangle) : Prop := sorry
noncomputable def midpoints_touching (S : Circle) (ABC : Triangle) (N : Point) : Prop := sorry

def S' (ABC : Triangle) := homothety (1/2) (nagel_point ABC) (triangle_incircle ABC)
def S (ABC : Triangle) := homothety (-1/2) (centroid ABC) (triangle_incircle ABC)

theorem circles_coincide (ABC : Triangle) : S ABC = S' ABC := sorry

theorem S_touches_midlines (ABC : Triangle) : midlines_touching (S ABC) ABC := sorry

theorem S'_touches_lines (ABC : Triangle) : midpoints_touching (S' ABC) ABC (nagel_point ABC) := sorry

end circles_coincide_S_touches_midlines_S_l188_188943


namespace flour_per_cake_l188_188849

theorem flour_per_cake (traci_flour harris_flour : ℕ) (cakes_each : ℕ)
  (h_traci_flour : traci_flour = 500)
  (h_harris_flour : harris_flour = 400)
  (h_cakes_each : cakes_each = 9) :
  (traci_flour + harris_flour) / (2 * cakes_each) = 50 := by
  sorry

end flour_per_cake_l188_188849


namespace estimate_height_of_student_l188_188650

theorem estimate_height_of_student
  (x_values : List ℝ)
  (y_values : List ℝ)
  (h_sum_x : x_values.sum = 225)
  (h_sum_y : y_values.sum = 1600)
  (h_length : x_values.length = 10 ∧ y_values.length = 10)
  (b : ℝ := 4) :
  ∃ a : ℝ, ∀ x : ℝ, x = 24 → (b * x + a = 166) :=
by
  have avg_x := (225 / 10 : ℝ)
  have avg_y := (1600 / 10 : ℝ)
  have a := avg_y - b * avg_x
  use a
  intro x h
  rw [h]
  sorry

end estimate_height_of_student_l188_188650


namespace part1_part2_l188_188904

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.sin (x + Real.pi / 3) - 1

theorem part1 : f (5 * Real.pi / 6) = -2 := by
  sorry

variables {A : ℝ} (hA1 : A > 0) (hA2 : A ≤ Real.pi / 3) (hFA : f A = 8 / 5)

theorem part2 (h : A > 0 ∧ A ≤ Real.pi / 3 ∧ f A = 8 / 5) : f (A + Real.pi / 4) = 6 / 5 :=
by
  sorry

end part1_part2_l188_188904


namespace largest_divisor_540_315_l188_188241

theorem largest_divisor_540_315 : ∃ d : ℕ, d ∣ 540 ∧ d ∣ 315 ∧ d = 45 := by
  sorry

end largest_divisor_540_315_l188_188241


namespace m_value_if_linear_l188_188317

theorem m_value_if_linear (m : ℝ) (x : ℝ) (h : (m + 2) * x^(|m| - 1) + 8 = 0) (linear : |m| - 1 = 1) : m = 2 :=
sorry

end m_value_if_linear_l188_188317


namespace derivative_at_zero_l188_188578
noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem derivative_at_zero : (deriv f 0) = -120 :=
by
  -- The proof is omitted
  sorry

end derivative_at_zero_l188_188578


namespace arithmetic_expression_evaluation_l188_188380

theorem arithmetic_expression_evaluation :
  12 / 4 - 3 - 6 + 3 * 5 = 9 :=
by
  sorry

end arithmetic_expression_evaluation_l188_188380


namespace quadratic_roots_product_sum_l188_188302

theorem quadratic_roots_product_sum :
  ∀ (f g : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 4 * x + 2 = 0 → x = f ∨ x = g) → 
  (f + g = 4 / 3) → 
  (f * g = 2 / 3) → 
  (f + 2) * (g + 2) = 22 / 3 :=
by
  intro f g roots_eq sum_eq product_eq
  sorry

end quadratic_roots_product_sum_l188_188302


namespace fixed_point_coordinates_l188_188503

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * a^(x + 1) - 3

theorem fixed_point_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  sorry

end fixed_point_coordinates_l188_188503


namespace right_triangle_other_angle_l188_188785

theorem right_triangle_other_angle (a b c : ℝ) 
  (h_triangle_sum : a + b + c = 180) 
  (h_right_angle : a = 90) 
  (h_acute_angle : b = 60) : 
  c = 30 :=
by
  sorry

end right_triangle_other_angle_l188_188785


namespace yoojeong_rabbits_l188_188199

theorem yoojeong_rabbits :
  ∀ (R C : ℕ), 
  let minyoung_dogs := 9
  let minyoung_cats := 3
  let minyoung_rabbits := 5
  let minyoung_total := minyoung_dogs + minyoung_cats + minyoung_rabbits
  let yoojeong_total := minyoung_total + 2
  let yoojeong_dogs := 7
  let yoojeong_cats := R - 2
  yoojeong_total = yoojeong_dogs + (R - 2) + R → 
  R = 7 :=
by
  intros R C minyoung_dogs minyoung_cats minyoung_rabbits minyoung_total yoojeong_total yoojeong_dogs yoojeong_cats
  have h1 : minyoung_total = 9 + 3 + 5 := rfl
  have h2 : yoojeong_total = minyoung_total + 2 := by sorry
  have h3 : yoojeong_dogs = 7 := rfl
  have h4 : yoojeong_cats = R - 2 := by sorry
  sorry

end yoojeong_rabbits_l188_188199


namespace least_integer_square_eq_12_more_than_three_times_l188_188519

theorem least_integer_square_eq_12_more_than_three_times (x : ℤ) (h : x^2 = 3 * x + 12) : x = -3 :=
sorry

end least_integer_square_eq_12_more_than_three_times_l188_188519


namespace beautiful_fold_probability_l188_188204

noncomputable def probability_beautiful_fold (a : ℝ) : ℝ := 1 / 2

theorem beautiful_fold_probability 
  (A B C D F : ℝ × ℝ) 
  (ABCD_square : (A.1 = 0) ∧ (A.2 = 0) ∧ 
                 (B.1 = a) ∧ (B.2 = 0) ∧ 
                 (C.1 = a) ∧ (C.2 = a) ∧ 
                 (D.1 = 0) ∧ (D.2 = a))
  (F_in_square : 0 ≤ F.1 ∧ F.1 ≤ a ∧ 0 ≤ F.2 ∧ F.2 ≤ a):
  probability_beautiful_fold a = 1 / 2 :=
sorry

end beautiful_fold_probability_l188_188204


namespace intervals_bound_l188_188628

variables {n : ℕ} (N : ℕ) (A : Finset (Fin n)) (A_i : Fin n → Finset (Fin n))

-- Condition: n ≥ 2
axiom h_n : n ≥ 2

-- Condition: Definition of interval
def interval (A : Finset (Fin n)) : Prop :=
  ∃ a b : Fin n, a.1 < b.1 ∧ A = Finset.Icc a b

-- Condition: A₁, ..., Aₙ are subsets such that A_i ∩ A_j is an interval
axiom h_intervals :
  ∀ {i j : Fin N}, i ≠ j → interval (A_i i ∩ A_i j)

-- Problem Statement: N ≤ ⌊ n^2 / 4 ⌋
theorem intervals_bound :
  N ≤ n * n / 4 :=
sorry

end intervals_bound_l188_188628


namespace jean_jail_time_l188_188189

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l188_188189


namespace coefficient_a5_l188_188920

theorem coefficient_a5 (a a1 a2 a3 a4 a5 a6 : ℝ) (h :  (∀ x : ℝ, x^6 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) :
  a5 = 6 :=
sorry

end coefficient_a5_l188_188920


namespace cos_neg_45_eq_one_over_sqrt_two_l188_188135

theorem cos_neg_45_eq_one_over_sqrt_two : Real.cos (-(45 : ℝ)) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_neg_45_eq_one_over_sqrt_two_l188_188135


namespace john_total_expenses_l188_188477

theorem john_total_expenses :
  (let epiPenCost := 500
   let yearlyMedicalExpenses := 2000
   let firstEpiPenInsuranceCoverage := 0.75
   let secondEpiPenInsuranceCoverage := 0.60
   let medicalExpensesCoverage := 0.80
   let firstEpiPenCost := epiPenCost * (1 - firstEpiPenInsuranceCoverage)
   let secondEpiPenCost := epiPenCost * (1 - secondEpiPenInsuranceCoverage)
   let totalEpiPenCost := firstEpiPenCost + secondEpiPenCost
   let yearlyMedicalExpensesCost := yearlyMedicalExpenses * (1 - medicalExpensesCoverage)
   let totalCost := totalEpiPenCost + yearlyMedicalExpensesCost
   totalCost) = 725 := sorry

end john_total_expenses_l188_188477


namespace matrix_problem_l188_188421

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![6, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 8], ![3, -5]]
def RHS : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![15, -3]]

theorem matrix_problem : 
  2 • A + B = RHS :=
by
  sorry

end matrix_problem_l188_188421


namespace aras_current_height_l188_188493

-- Define the variables and conditions
variables (x : ℝ) (sheas_original_height : ℝ := x) (ars_original_height : ℝ := x)
variables (sheas_growth_factor : ℝ := 0.30) (sheas_current_height : ℝ := 65)
variables (sheas_growth : ℝ := sheas_current_height - sheas_original_height)
variables (aras_growth : ℝ := sheas_growth / 3)

-- Define a theorem for Ara's current height
theorem aras_current_height (h1 : sheas_current_height = (1 + sheas_growth_factor) * sheas_original_height)
                           (h2 : sheas_original_height = ars_original_height) :
                           aras_growth + ars_original_height = 55 :=
by
  sorry

end aras_current_height_l188_188493


namespace valid_number_of_m_values_l188_188340

theorem valid_number_of_m_values : 
  (∃ m : ℕ, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m)) ∧ ∀ m, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m) → m > 1  → 
  ∃ n : ℕ, n = 22 :=
by
  sorry

end valid_number_of_m_values_l188_188340


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l188_188043

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l188_188043


namespace jean_total_jail_time_l188_188186

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l188_188186


namespace problem_statement_l188_188365

noncomputable def c := 3 + Real.sqrt 21
noncomputable def d := 3 - Real.sqrt 21

theorem problem_statement : 
  (c + 2 * d) = 9 - Real.sqrt 21 :=
by
  sorry

end problem_statement_l188_188365


namespace greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l188_188686

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem greatest_divisor_of_546_smaller_than_30_and_factor_of_126 :
  ∃ (d : ℕ), d < 30 ∧ is_factor d 546 ∧ is_factor d 126 ∧ ∀ e : ℕ, e < 30 ∧ is_factor e 546 ∧ is_factor e 126 → e ≤ d := 
sorry

end greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l188_188686


namespace vector_expression_evaluation_l188_188028

theorem vector_expression_evaluation (θ : ℝ) :
  let a := (2 * Real.cos θ, Real.sin θ)
  let b := (1, -6)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (2 * Real.cos θ + Real.sin θ) / (Real.cos θ + 3 * Real.sin θ) = 7 / 6 :=
by
  intros a b h
  sorry

end vector_expression_evaluation_l188_188028


namespace price_decrease_is_50_percent_l188_188506

-- Original price is 50 yuan
def original_price : ℝ := 50

-- Price after 100% increase
def increased_price : ℝ := original_price * (1 + 1)

-- Required percentage decrease to return to original price
def required_percentage_decrease (x : ℝ) : ℝ := increased_price * (1 - x)

theorem price_decrease_is_50_percent : required_percentage_decrease 0.5 = 50 :=
  by 
    sorry

end price_decrease_is_50_percent_l188_188506


namespace arithmetic_first_term_l188_188194

theorem arithmetic_first_term (a : ℕ) (d : ℕ) (T : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, T n = n * (2 * a + (n - 1) * d) / 2) →
  (∀ n : ℕ, T (4 * n) / T n = k) →
  d = 5 →
  k = 16 →
  a = 3 := 
by
  sorry

end arithmetic_first_term_l188_188194


namespace solve_equation_l188_188902

theorem solve_equation (Y : ℝ) : (3.242 * 10 * Y) / 100 = 0.3242 * Y := 
by 
  sorry

end solve_equation_l188_188902


namespace kayla_apples_l188_188621

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end kayla_apples_l188_188621


namespace expenditure_recording_l188_188008

def income : ℕ := 200
def recorded_income : ℤ := 200
def expenditure (e : ℕ) : ℤ := -(e : ℤ)

theorem expenditure_recording (e : ℕ) :
  expenditure 150 = -150 := by
  sorry

end expenditure_recording_l188_188008


namespace identify_letters_l188_188211

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l188_188211


namespace right_triangle_count_l188_188639

theorem right_triangle_count (A P B C Q D : ℝ × ℝ)
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : D ≠ A)
  (h5 : A ≠ C) (h6 : B ≠ D)
  (rectangle : A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2)
  (PQ_on_AC : ∃ t : ℝ , P = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
    ∧ Q = (A.1 + (1 - t) * (C.1 - A.1), A.2 + (1 - t) * (C.2 - A.2)))
  (PQ_perpendicular_BD : ∃ k : ℝ, P.2 - Q.2 = k * (B.2 - D.2) ∧ P.1 - Q.1 = -k * (B.1 - D.1)):
  (number_of_right_triangles A P B C Q D = 12) :=
by
  sorry

end right_triangle_count_l188_188639


namespace smallest_n_exists_l188_188331

theorem smallest_n_exists (G : Type) [Fintype G] [DecidableEq G] (connected : G → G → Prop)
  (distinct_naturals : G → ℕ) :
  (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 = 1) ∧
  (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 > 1) →
  (∀ n : ℕ, 
    (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) n = 1) ∧
    (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) n > 1) →
    15 ≤ n) :=
sorry

end smallest_n_exists_l188_188331


namespace divisors_of_10_factorial_greater_than_9_factorial_l188_188768

theorem divisors_of_10_factorial_greater_than_9_factorial :
  {d : ℕ | d ∣ nat.factorial 10 ∧ d > nat.factorial 9}.card = 9 := 
sorry

end divisors_of_10_factorial_greater_than_9_factorial_l188_188768


namespace correct_standardized_statement_l188_188083

-- Define and state the conditions as Lean 4 definitions and propositions
structure GeometricStatement :=
  (description : String)
  (is_standardized : Prop)

def optionA : GeometricStatement := {
  description := "Line a and b intersect at point m",
  is_standardized := False -- due to use of lowercase 'm'
}

def optionB : GeometricStatement := {
  description := "Extend line AB",
  is_standardized := False -- since a line cannot be further extended
}

def optionC : GeometricStatement := {
  description := "Extend ray AO (where O is the endpoint) in the opposite direction",
  is_standardized := False -- incorrect definition of ray extension
}

def optionD : GeometricStatement := {
  description := "Extend line segment AB to C such that BC=AB",
  is_standardized := True -- correct by geometric principles
}

-- The theorem stating that option D is the correct and standardized statement
theorem correct_standardized_statement : optionD.is_standardized = True ∧
                                         optionA.is_standardized = False ∧
                                         optionB.is_standardized = False ∧
                                         optionC.is_standardized = False :=
  by sorry

end correct_standardized_statement_l188_188083


namespace num_ordered_pairs_1806_l188_188839

theorem num_ordered_pairs_1806 :
  let n := 1806 in
  let pf := [(2, 1), (3, 2), (101, 1)] in
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1) in
  ∃ (c : ℕ), c = num_divisors ∧ c = 12 :=
by
  let n := 1806
  let pf := [(2, 1), (3, 2), (101, 1)]
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1)
  use num_divisors
  split
  . rfl
  . rfl
  sorry

end num_ordered_pairs_1806_l188_188839


namespace train_and_car_combined_time_l188_188408

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l188_188408


namespace Amy_balloons_l188_188615

-- Defining the conditions
def James_balloons : ℕ := 1222
def more_balloons : ℕ := 208

-- Defining Amy's balloons as a proof goal
theorem Amy_balloons : ∀ (Amy_balloons : ℕ), James_balloons - more_balloons = Amy_balloons → Amy_balloons = 1014 :=
by
  intros Amy_balloons h
  sorry

end Amy_balloons_l188_188615


namespace carnations_count_l188_188711

theorem carnations_count (total_flowers : ℕ) (fract_rose : ℚ) (num_tulips : ℕ) (h1 : total_flowers = 40) (h2 : fract_rose = 2 / 5) (h3 : num_tulips = 10) :
  total_flowers - ((fract_rose * total_flowers) + num_tulips) = 14 := 
by
  sorry

end carnations_count_l188_188711


namespace cos_neg_45_degree_l188_188136

theorem cos_neg_45_degree :
  real.cos (-π / 4) = real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_45_degree_l188_188136


namespace perpendicular_bisector_eqn_l188_188929

-- Definitions based on given conditions
def C₁ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem perpendicular_bisector_eqn {ρ θ : ℝ} :
  (∃ A B : ℝ × ℝ,
    A ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₁ ρ θ} ∧
    B ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₂ ρ θ}) →
  ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end perpendicular_bisector_eqn_l188_188929


namespace q_is_false_l188_188170

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q) = false) (h2 : ¬p = false) : q = false :=
by
  sorry

end q_is_false_l188_188170


namespace y_coordinate_of_A_l188_188306

theorem y_coordinate_of_A (a : ℝ) (y : ℝ) (h1 : y = a * 1) (h2 : y = (4 - a) / 1) : y = 2 :=
by
  sorry

end y_coordinate_of_A_l188_188306


namespace travel_time_total_l188_188265

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end travel_time_total_l188_188265


namespace minimum_degree_g_l188_188900

open Polynomial

theorem minimum_degree_g (f g h : Polynomial ℝ) 
  (h_eq : 5 • f + 2 • g = h)
  (deg_f : f.degree = 11)
  (deg_h : h.degree = 12) : 
  ∃ d : ℕ, g.degree = d ∧ d >= 12 := 
sorry

end minimum_degree_g_l188_188900


namespace prove_odd_function_definition_l188_188010

theorem prove_odd_function_definition (f : ℝ → ℝ) 
  (odd : ∀ x : ℝ, f (-x) = -f x)
  (pos_def : ∀ x : ℝ, 0 < x → f x = 2 * x ^ 2 - x + 1) :
  ∀ x : ℝ, x < 0 → f x = -2 * x ^ 2 - x - 1 :=
by
  intro x hx
  sorry

end prove_odd_function_definition_l188_188010


namespace initial_number_2008_l188_188120

theorem initial_number_2008 
  (numbers_on_blackboard : ℕ → Prop)
  (x : ℕ)
  (Ops : ∀ x, numbers_on_blackboard x → (numbers_on_blackboard (2 * x + 1) ∨ numbers_on_blackboard (x / (x + 2)))) 
  (initial_apearing : numbers_on_blackboard 2008) :
  numbers_on_blackboard 2008 = true :=
sorry

end initial_number_2008_l188_188120


namespace third_number_in_first_set_l188_188063

theorem third_number_in_first_set (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end third_number_in_first_set_l188_188063


namespace garden_area_increase_l188_188115

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l188_188115


namespace combination_20_choose_19_eq_20_l188_188727

theorem combination_20_choose_19_eq_20 : nat.choose 20 19 = 20 :=
sorry

end combination_20_choose_19_eq_20_l188_188727


namespace greatest_sum_of_consecutive_odd_integers_lt_500_l188_188853

-- Define the consecutive odd integers and their conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  n % 2 = 1 ∧ (n + 2) % 2 = 1

-- Define the condition that their product must be less than 500
def prod_less_500 (n : ℤ) : Prop :=
  n * (n + 2) < 500

-- The theorem statement
theorem greatest_sum_of_consecutive_odd_integers_lt_500 : 
  ∃ n : ℤ, consecutive_odd_integers n ∧ prod_less_500 n ∧ ∀ m : ℤ, consecutive_odd_integers m ∧ prod_less_500 m → n + (n + 2) ≥ m + (m + 2) :=
sorry

end greatest_sum_of_consecutive_odd_integers_lt_500_l188_188853


namespace binom_20_19_eq_20_l188_188729

theorem binom_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  -- use the property of binomial coefficients
  have h : Nat.choose 20 19 = Nat.choose 20 1 := Nat.choose_symm 20 19
  -- now, apply the fact that Nat.choose 20 1 = 20
  rw h
  exact Nat.choose_one 20

end binom_20_19_eq_20_l188_188729


namespace compute_expression_value_l188_188280

-- Define the expression
def expression : ℤ := 1013^2 - 1009^2 - 1011^2 + 997^2

-- State the theorem with the required conditions and conclusions
theorem compute_expression_value : expression = -19924 := 
by 
  -- The proof steps would go here.
  sorry

end compute_expression_value_l188_188280


namespace equal_probabilities_l188_188232

-- Definitions based on the conditions in the problem

def total_parts : ℕ := 160
def first_class_parts : ℕ := 48
def second_class_parts : ℕ := 64
def third_class_parts : ℕ := 32
def substandard_parts : ℕ := 16
def sample_size : ℕ := 20

-- Define the probabilities for each sampling method
def p1 : ℚ := sample_size / total_parts
def p2 : ℚ := (6 : ℚ) / first_class_parts  -- Given the conditions, this will hold for all classes
def p3 : ℚ := 1 / 8

theorem equal_probabilities :
  p1 = p2 ∧ p2 = p3 :=
by
  -- This is the end of the statement as no proof is required
  sorry

end equal_probabilities_l188_188232


namespace expression_value_l188_188520

theorem expression_value (a b : ℤ) (h₁ : a = -5) (h₂ : b = 3) :
  -a - b^4 + a * b = -91 := by
  sorry

end expression_value_l188_188520


namespace find_first_term_l188_188269

noncomputable def first_term_of_arithmetic_sequence : ℝ := -19.2

theorem find_first_term
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1050)
  (h2 : 50 * (2 * a + 199 * d) = 4050) :
  a = first_term_of_arithmetic_sequence :=
by
  -- Given conditions
  have h1' : 2 * a + 99 * d = 21 := by sorry
  have h2' : 2 * a + 199 * d = 81 := by sorry
  -- Solve for d
  have hd : d = 0.6 := by sorry
  -- Substitute d into h1'
  have h_subst : 2 * a + 99 * 0.6 = 21 := by sorry
  -- Solve for a
  have ha : a = -19.2 := by sorry
  exact ha

end find_first_term_l188_188269


namespace evaluate_expression_l188_188080

-- Define x as given in the condition
def x : ℤ := 5

-- State the theorem we need to prove
theorem evaluate_expression : x^3 - 3 * x = 110 :=
by
  -- Proof will be provided here
  sorry

end evaluate_expression_l188_188080


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l188_188048

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l188_188048


namespace no_upper_bound_l188_188535

-- Given Conditions
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {M : ℝ}

-- Condition: widths and lengths of plates are 1 and a1, a2, a3, ..., respectively
axiom width_1 : ∀ n, (S n > 0)

-- Condition: a1 ≠ 1
axiom a1_neq_1 : a 1 ≠ 1

-- Condition: plates are similar but not congruent starting from the second
axiom similar_not_congruent : ∀ n > 1, (a (n+1) > a n)

-- Condition: S_n denotes the length covered after placing n plates
axiom Sn_length : ∀ n, S (n+1) = S n + a (n+1)

-- Condition: a_{n+1} = 1 / S_n
axiom an_reciprocal : ∀ n, a (n+1) = 1 / S n

-- The final goal: no such real number exists that S_n does not exceed
theorem no_upper_bound : ∀ M : ℝ, ∃ n : ℕ, S n > M := 
sorry

end no_upper_bound_l188_188535


namespace problem_statement_l188_188774

theorem problem_statement (a b : ℝ) (h : a > b) : a - 1 > b - 1 :=
sorry

end problem_statement_l188_188774


namespace solve_simultaneous_eqns_l188_188743

theorem solve_simultaneous_eqns :
  ∀ (x y : ℝ), 
  (1/x - 1/(2*y) = 2*y^4 - 2*x^4 ∧ 1/x + 1/(2*y) = (3*x^2 + y^2) * (x^2 + 3*y^2)) 
  ↔ 
  (x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2) :=
by sorry

end solve_simultaneous_eqns_l188_188743


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l188_188044

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l188_188044


namespace ordered_pairs_1806_l188_188841

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l188_188841


namespace distance_from_two_eq_three_l188_188947

theorem distance_from_two_eq_three (x : ℝ) (h : |x - 2| = 3) : x = -1 ∨ x = 5 :=
sorry

end distance_from_two_eq_three_l188_188947


namespace find_f_2018_l188_188896

-- Define the function f, its periodicity and even property
variable (f : ℝ → ℝ)

-- Conditions
axiom f_periodicity : ∀ x : ℝ, f (x + 4) = -f x
axiom f_symmetric : ∀ x : ℝ, f x = f (-x)
axiom f_at_two : f 2 = 2

-- Theorem stating the desired property
theorem find_f_2018 : f 2018 = 2 :=
  sorry

end find_f_2018_l188_188896


namespace system_of_equations_solutions_l188_188370

theorem system_of_equations_solutions :
  ∃ (sol : Finset (ℝ × ℝ)), sol.card = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ sol ↔ (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1)) :=
by
  sorry

end system_of_equations_solutions_l188_188370


namespace quadratic_equation_conditions_l188_188157

theorem quadratic_equation_conditions :
  ∃ (a b c : ℝ), a = 3 ∧ c = 1 ∧ (a * x^2 + b * x + c = 0 ↔ 3 * x^2 + 1 = 0) :=
by
  use 3, 0, 1
  sorry

end quadratic_equation_conditions_l188_188157


namespace problem_statement_l188_188296

theorem problem_statement (a b : ℝ) (h : a^2 > b^2) : a > b → a > 0 :=
sorry

end problem_statement_l188_188296


namespace area_of_square_with_circles_l188_188284

theorem area_of_square_with_circles (r : ℝ) (h_r : r = 8) : 
  let d := 2 * r in 
  let side := 2 * d in 
  (side ^ 2) = 1024 :=
by
  intros
  have h_d : d = 16 := by linarith
  have h_side : side = 32 := by linarith
  rw [h_side]
  norm_num

end area_of_square_with_circles_l188_188284


namespace expression_defined_if_x_not_3_l188_188600

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end expression_defined_if_x_not_3_l188_188600


namespace no_perfect_square_integers_l188_188431

open Nat

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 10 * x^2 + 4 * x + 29

theorem no_perfect_square_integers : ∀ x : ℤ, ¬∃ a : ℤ, Q x = a^2 :=
by
  sorry

end no_perfect_square_integers_l188_188431


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l188_188046

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l188_188046


namespace num_audio_cassettes_in_second_set_l188_188823

-- Define the variables and constants
def costOfAudio (A : ℕ) : ℕ := A
def costOfVideo (V : ℕ) : ℕ := V
def totalCost (numOfAudio : ℕ) (numOfVideo : ℕ) (A : ℕ) (V : ℕ) : ℕ :=
  numOfAudio * (costOfAudio A) + numOfVideo * (costOfVideo V)

-- Given conditions
def condition1 (A V : ℕ) : Prop := ∃ X : ℕ, totalCost X 4 A V = 1350
def condition2 (A V : ℕ) : Prop := totalCost 7 3 A V = 1110
def condition3 : Prop := costOfVideo 300 = 300

-- Main theorem to prove: The number of audio cassettes in the second set is 7
theorem num_audio_cassettes_in_second_set :
  ∃ (A : ℕ), condition1 A 300 ∧ condition2 A 300 ∧ condition3 →
  7 = 7 :=
by
  sorry

end num_audio_cassettes_in_second_set_l188_188823


namespace proof_problem_l188_188750

noncomputable def arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  n * (a 1) + ((n * (n - 1)) / 2) * (a 2 - a 1)

theorem proof_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (d : ℕ)
  (h_d_gt_zero : d > 0)
  (h_a1 : a 1 = 1)
  (h_S : ∀ n, S n = arithmetic_sequence_sum n a)
  (h_S2_S3 : S 2 * S 3 = 36)
  (h_arith_seq : ∀ n, a (n + 1) = a 1 + n * d)
  (m k : ℕ)
  (h_mk_pos : m > 0 ∧ k > 0)
  (sum_condition : (k + 1) * (a m + a (m + k)) / 2 = 65) :
  d = 2 ∧ (∀ n, S n = n * n) ∧ m = 5 ∧ k = 4 :=
by 
  sorry

end proof_problem_l188_188750


namespace find_h_parallel_line_l188_188997

theorem find_h_parallel_line:
  ∃ h : ℚ, (3 * (h : ℚ) - 2 * (24 : ℚ) = 7) → (h = 47 / 3) :=
by
  sorry

end find_h_parallel_line_l188_188997


namespace jean_total_jail_time_l188_188187

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l188_188187


namespace fruit_bowl_oranges_l188_188176

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end fruit_bowl_oranges_l188_188176


namespace seven_thousand_twenty_two_is_7022_l188_188055

-- Define the translations of words to numbers
def seven_thousand : ℕ := 7000
def twenty_two : ℕ := 22

-- Define the full number by summing its parts
def seven_thousand_twenty_two : ℕ := seven_thousand + twenty_two

theorem seven_thousand_twenty_two_is_7022 : seven_thousand_twenty_two = 7022 := by
  sorry

end seven_thousand_twenty_two_is_7022_l188_188055


namespace garden_area_increase_l188_188116

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l188_188116


namespace garden_area_difference_l188_188111

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l188_188111


namespace arithmetic_mean_of_18_27_45_l188_188683

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_of_18_27_45_l188_188683


namespace find_x_l188_188291

theorem find_x (x : ℝ) (h_pos : x > 0) (h_eq : x * (⌊x⌋) = 132) : x = 12 := sorry

end find_x_l188_188291


namespace gcd_of_35_and_number_between_70_and_90_is_7_l188_188834

def number_between_70_and_90 (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 90

def gcd_is_7 (a b : ℕ) : Prop :=
  Nat.gcd a b = 7

theorem gcd_of_35_and_number_between_70_and_90_is_7 : 
  ∃ (n : ℕ), number_between_70_and_90 n ∧ gcd_is_7 35 n ∧ (n = 77 ∨ n = 84) :=
by
  sorry

end gcd_of_35_and_number_between_70_and_90_is_7_l188_188834


namespace unique_zero_iff_a_eq_half_l188_188451

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (1 - x))

theorem unique_zero_iff_a_eq_half :
  (∃! x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end unique_zero_iff_a_eq_half_l188_188451


namespace garden_area_increase_l188_188107

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l188_188107


namespace area_increase_correct_l188_188093

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l188_188093


namespace number_of_feet_on_branches_l188_188969

def number_of_birds : ℕ := 46
def feet_per_bird : ℕ := 2

theorem number_of_feet_on_branches : number_of_birds * feet_per_bird = 92 := 
by 
  sorry

end number_of_feet_on_branches_l188_188969


namespace eval_diff_squares_l188_188558

theorem eval_diff_squares : 81^2 - 49^2 = 4160 :=
by
  sorry

end eval_diff_squares_l188_188558


namespace evaluate_ratio_l188_188429

theorem evaluate_ratio : (2^3002 * 3^3005 / 6^3003 : ℚ) = 9 / 2 := 
sorry

end evaluate_ratio_l188_188429


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l188_188047

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l188_188047


namespace sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l188_188343

theorem sum_of_reciprocals_roots_transformed_eq_neg11_div_4 :
  (∃ a b c : ℝ, (a^3 - a - 2 = 0) ∧ (b^3 - b - 2 = 0) ∧ (c^3 - c - 2 = 0)) → 
  ( ∃ a b c : ℝ, a^3 - a - 2 = 0 ∧ b^3 - b - 2 = 0 ∧ c^3 - c - 2 = 0 ∧ 
  (1 / (a - 2) + 1 / (b - 2) + 1 / (c - 2) = - 11 / 4)) :=
by
  sorry

end sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l188_188343


namespace min_value_of_function_l188_188894

theorem min_value_of_function (x : ℝ) (h: x > 1) :
  ∃ t > 0, x = t + 1 ∧ (t + 3 / t + 3) = 3 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l188_188894


namespace rodney_probability_correct_guess_l188_188634

noncomputable def two_digit_integer (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

noncomputable def tens_digit (n : ℕ) : Prop :=
  (n / 10 = 7 ∨ n / 10 = 8 ∨ n / 10 = 9)

noncomputable def units_digit_even (n : ℕ) : Prop :=
  (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8)

noncomputable def greater_than_seventy_five (n : ℕ) : Prop := n > 75

theorem rodney_probability_correct_guess (n : ℕ) :
  two_digit_integer n →
  tens_digit n →
  units_digit_even n →
  greater_than_seventy_five n →
  (∃ m, m = 1 / 12) :=
sorry

end rodney_probability_correct_guess_l188_188634


namespace parabola_focus_coordinates_l188_188423

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), y = 4 * x^2 → (0, y / 16) = (0, 1 / 16) :=
by
  intros x y h
  sorry

end parabola_focus_coordinates_l188_188423


namespace solve_x_given_y_l188_188594

theorem solve_x_given_y (x : ℝ) (h : 2 = 2 / (5 * x + 3)) : x = -2 / 5 :=
sorry

end solve_x_given_y_l188_188594


namespace quadratic_real_roots_l188_188596

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x1 x2 : ℝ, k * x1^2 - 6 * x1 - 1 = 0 ∧ k * x2^2 - 6 * x2 - 1 = 0 ∧ x1 ≠ x2) ↔ k ≥ -9 := 
by
  sorry

end quadratic_real_roots_l188_188596


namespace solve_for_y_l188_188497

theorem solve_for_y (y : ℤ) (h : (y ≠ 2) → ((y^2 - 10*y + 24)/(y-2) + (4*y^2 + 8*y - 48)/(4*y - 8) = 0)) : y = 0 :=
by
  sorry

end solve_for_y_l188_188497


namespace tangent_line_circle_m_values_l188_188641

theorem tangent_line_circle_m_values {m : ℝ} :
  (∀ (x y: ℝ), 3 * x + 4 * y + m = 0 → (x - 1)^2 + (y + 2)^2 = 4) →
  (m = 15 ∨ m = -5) :=
by
  sorry

end tangent_line_circle_m_values_l188_188641


namespace lukas_avg_points_per_game_l188_188198

theorem lukas_avg_points_per_game (total_points games_played : ℕ) (h_total_points : total_points = 60) (h_games_played : games_played = 5) :
  (total_points / games_played = 12) :=
by
  sorry

end lukas_avg_points_per_game_l188_188198


namespace stream_speed_l188_188991

-- Definitions based on conditions
def speed_in_still_water : ℝ := 5
def distance_downstream : ℝ := 100
def time_downstream : ℝ := 10

-- The required speed of the stream
def speed_of_stream (v : ℝ) : Prop :=
  distance_downstream = (speed_in_still_water + v) * time_downstream

-- Proof statement: the speed of the stream is 5 km/hr
theorem stream_speed : ∃ v, speed_of_stream v ∧ v = 5 := 
by
  use 5
  unfold speed_of_stream
  sorry

end stream_speed_l188_188991


namespace sum_of_coordinates_l188_188220

variable (f : ℝ → ℝ)

/-- Given that the point (2, 3) is on the graph of y = f(x) / 3,
    show that (9, 2/3) must be on the graph of y = f⁻¹(x) / 3 and the
    sum of its coordinates is 29/3. -/
theorem sum_of_coordinates (h : 3 = f 2 / 3) : (9 : ℝ) + (2 / 3 : ℝ) = 29 / 3 :=
by
  have h₁ : f 2 = 9 := by
    linarith
    
  have h₂ : f⁻¹ 9 = 2 := by
    -- We assume that f has an inverse and it is well-defined
    sorry

  have point_on_graph : (9, (2 / 3)) ∈ { p : ℝ × ℝ | p.2 = f⁻¹ p.1 / 3 } := by
    sorry

  show 9 + 2 / 3 = 29 / 3
  norm_num

end sum_of_coordinates_l188_188220


namespace fraction_of_full_tank_used_l188_188275

-- Define the initial conditions as per the problem statement
def speed : ℝ := 50 -- miles per hour
def time : ℝ := 5   -- hours
def miles_per_gallon : ℝ := 30
def full_tank_capacity : ℝ := 15 -- gallons

-- We need to prove that the fraction of gasoline used is 5/9
theorem fraction_of_full_tank_used : 
  ((speed * time) / miles_per_gallon) / full_tank_capacity = 5 / 9 := by
sorry

end fraction_of_full_tank_used_l188_188275


namespace focus_of_parabola_tangent_to_circle_directrix_l188_188156

theorem focus_of_parabola_tangent_to_circle_directrix :
  ∃ p : ℝ, p > 0 ∧
  (∃ (x y : ℝ), x ^ 2 + y ^ 2 - 6 * x - 7 = 0 ∧
  ∀ x y : ℝ, y ^ 2 = 2 * p * x → x = -p) →
  (1, 0) = (p, 0) :=
by
  sorry

end focus_of_parabola_tangent_to_circle_directrix_l188_188156


namespace bill_left_with_money_l188_188131

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end bill_left_with_money_l188_188131


namespace continuity_at_2_l188_188035

theorem continuity_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-3 * x^2 - 5) + 17| < ε :=
by
  sorry

end continuity_at_2_l188_188035


namespace cyclic_sum_inequality_l188_188305

theorem cyclic_sum_inequality
  (a b c d e : ℝ)
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end cyclic_sum_inequality_l188_188305


namespace find_m_value_l188_188465

theorem find_m_value 
  (h : ∀ x y m : ℝ, 2*x + y + m = 0 → (1 : ℝ)*x + (-2 : ℝ)*y + 0 = 0)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ m : ℝ, m = 0 :=
sorry

end find_m_value_l188_188465


namespace exactly_one_wins_at_most_two_win_l188_188512

def prob_A : ℚ := 4 / 5 
def prob_B : ℚ := 3 / 5 
def prob_C : ℚ := 7 / 10

theorem exactly_one_wins :
  (prob_A * (1 - prob_B) * (1 - prob_C) + 
   (1 - prob_A) * prob_B * (1 - prob_C) + 
   (1 - prob_A) * (1 - prob_B) * prob_C) = 47 / 250 := 
by sorry

theorem at_most_two_win :
  (1 - (prob_A * prob_B * prob_C)) = 83 / 125 :=
by sorry

end exactly_one_wins_at_most_two_win_l188_188512


namespace total_plant_count_l188_188118

-- Definitions for conditions.
def total_rows : ℕ := 96
def columns_per_row : ℕ := 24
def divided_rows : ℕ := total_rows / 3
def undivided_rows : ℕ := total_rows - divided_rows
def beans_in_undivided_row : ℕ := columns_per_row
def corn_in_divided_row : ℕ := columns_per_row / 2
def tomatoes_in_divided_row : ℕ := columns_per_row / 2

-- Total number of plants calculation.
def total_bean_plants : ℕ := undivided_rows * beans_in_undivided_row
def total_corn_plants : ℕ := divided_rows * corn_in_divided_row
def total_tomato_plants : ℕ := divided_rows * tomatoes_in_divided_row

def total_plants : ℕ := total_bean_plants + total_corn_plants + total_tomato_plants

-- Proof statement.
theorem total_plant_count : total_plants = 2304 :=
by
  sorry

end total_plant_count_l188_188118


namespace simplify_expression_l188_188542

theorem simplify_expression (a b : ℝ) : 
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b :=
by
  sorry

end simplify_expression_l188_188542


namespace total_candies_in_third_set_l188_188676

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l188_188676


namespace starting_number_is_33_l188_188846

theorem starting_number_is_33 (n : ℕ)
  (h1 : ∀ k, (33 + k * 11 ≤ 79) → (k < 5))
  (h2 : ∀ k, (k < 5) → (33 + k * 11 ≤ 79)) :
  n = 33 :=
sorry

end starting_number_is_33_l188_188846


namespace final_length_of_movie_l188_188648

theorem final_length_of_movie :
  let original_length := 3600 -- original movie length in seconds
  let cut_1 := 3 * 60 -- first scene cut in seconds
  let cut_2 := (5 * 60) + 30 -- second scene cut in seconds
  let cut_3 := (2 * 60) + 15 -- third scene cut in seconds
  let total_cut := cut_1 + cut_2 + cut_3 -- total cut time in seconds
  let final_length_seconds := original_length - total_cut -- final length in seconds
  final_length_seconds = 2955 ∧ final_length_seconds / 60 = 49 ∧ final_length_seconds % 60 = 15
:= by
  sorry

end final_length_of_movie_l188_188648


namespace hyperbola_eccentricity_l188_188463

theorem hyperbola_eccentricity
    (a b e : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (h_hyperbola : ∀ x y, x ^ 2 / a^2 - y^2 / b^2 = 1)
    (h_circle : ∀ x y, (x - 2) ^ 2 + y ^ 2 = 4)
    (h_chord_length : ∀ x y, (x ^ 2 + y ^ 2)^(1/2) = 2) :
    e = 2 := 
sorry

end hyperbola_eccentricity_l188_188463


namespace inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l188_188566

theorem inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared 
  (a b c : ℝ)
  (h_sum : a + b + c = 0)
  (d : ℝ) 
  (h_d : d = max (abs a) (max (abs b) (abs c))) : 
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by 
  sorry

end inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l188_188566


namespace isosceles_triangle_angles_l188_188514

theorem isosceles_triangle_angles (y : ℝ) (h : y > 0) :
  let P := y
  let R := 5 * y
  let Q := R
  P + Q + R = 180 → Q = 81.82 :=
by
  sorry

end isosceles_triangle_angles_l188_188514


namespace liza_butter_amount_l188_188807

theorem liza_butter_amount (B : ℕ) (h1 : B / 2 + B / 5 + (1 / 3) * ((B - B / 2 - B / 5) / 1) = B - 2) : B = 10 :=
sorry

end liza_butter_amount_l188_188807


namespace garden_breadth_l188_188781

theorem garden_breadth (perimeter length breadth : ℕ) 
    (h₁ : perimeter = 680)
    (h₂ : length = 258)
    (h₃ : perimeter = 2 * (length + breadth)) : 
    breadth = 82 := 
sorry

end garden_breadth_l188_188781


namespace cubic_foot_to_cubic_inches_l188_188914

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l188_188914


namespace line_circle_intersect_l188_188453

theorem line_circle_intersect (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, (x - a)^2 + (y - 1)^2 = 2 ∧ x - a * y - 2 = 0 :=
sorry

end line_circle_intersect_l188_188453


namespace find_x_of_perpendicular_l188_188000

-- Definitions based on the conditions in a)
def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

-- The mathematical proof problem in Lean 4 statement: prove that the dot product is zero implies x = -2/3
theorem find_x_of_perpendicular (x : ℝ) (h : (a x).fst * b.fst + (a x).snd * b.snd = 0) : x = -2 / 3 := 
by
  sorry

end find_x_of_perpendicular_l188_188000


namespace remainder_problem_l188_188688

theorem remainder_problem : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end remainder_problem_l188_188688


namespace restaurant_bill_split_l188_188200

def original_bill : ℝ := 514.16
def tip_rate : ℝ := 0.18
def number_of_people : ℕ := 9
def final_amount_per_person : ℝ := 67.41

theorem restaurant_bill_split :
  final_amount_per_person = (1 + tip_rate) * original_bill / number_of_people :=
by
  sorry

end restaurant_bill_split_l188_188200


namespace third_set_candies_l188_188661

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l188_188661


namespace dentist_ratio_l188_188077

-- Conditions
def cost_cleaning : ℕ := 70
def cost_filling : ℕ := 120
def cost_extraction : ℕ := 290

-- Theorem statement
theorem dentist_ratio : (cost_cleaning + 2 * cost_filling + cost_extraction) / cost_filling = 5 := 
by
  -- To be proven
  sorry

end dentist_ratio_l188_188077


namespace lines_intersect_l188_188258

-- Define the parameterizations of the two lines
def line1 (t : ℚ) : ℚ × ℚ := ⟨2 + 3 * t, 3 - 4 * t⟩
def line2 (u : ℚ) : ℚ × ℚ := ⟨4 + 5 * u, 1 + 3 * u⟩

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = line2 u ∧ line1 t = ⟨26 / 11, 19 / 11⟩ :=
by
  sorry

end lines_intersect_l188_188258


namespace value_range_of_sum_difference_l188_188481

theorem value_range_of_sum_difference (a b c : ℝ) (h₁ : a < b)
  (h₂ : a + b = b / a) (h₃ : a * b = c / a) (h₄ : a + b > c)
  (h₅ : a + c > b) (h₆ : b + c > a) : 
  ∃ x y, x = 7 / 8 ∧ y = Real.sqrt 5 - 1 ∧ x < a + b - c ∧ a + b - c < y := sorry

end value_range_of_sum_difference_l188_188481


namespace candy_count_in_third_set_l188_188668

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l188_188668


namespace radius_of_larger_circle_l188_188963

theorem radius_of_larger_circle (r : ℝ) (r_pos : r > 0)
    (ratio_condition : ∀ (rs : ℝ), rs = 3 * r)
    (diameter_condition : ∀ (ac : ℝ), ac = 6 * r)
    (chord_tangent_condition : ∀ (ab : ℝ), ab = 12) :
     (radius : ℝ) = 3 * r :=
by
  sorry

end radius_of_larger_circle_l188_188963


namespace candy_count_in_third_set_l188_188669

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l188_188669


namespace greatest_x_l188_188687

theorem greatest_x (x : ℕ) : (x^6 / x^3 ≤ 27) → x ≤ 3 :=
by sorry

end greatest_x_l188_188687


namespace exists_divisor_between_l188_188706

theorem exists_divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) 
  (h_div1 : a ∣ n) (h_div2 : b ∣ n) (h_neq : a ≠ b) 
  (h_lt : a < b) (h_eq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end exists_divisor_between_l188_188706


namespace meaningful_expression_range_l188_188597

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end meaningful_expression_range_l188_188597


namespace min_races_top_3_l188_188354

theorem min_races_top_3 (max_horses_per_race : ℕ) (total_horses : ℕ) (no_timer : Prop) 
    (max_horses_per_race_condition : max_horses_per_race = 5) (total_horses_condition : total_horses = 25) 
    (no_timing_condition : no_timer) : 
    (∃ min_races : ℕ, min_races = 7) :=
by
  use 7
  sorry

end min_races_top_3_l188_188354


namespace interval_length_implies_difference_l188_188424

theorem interval_length_implies_difference (a b : ℝ) (h : (b - 5) / 3 - (a - 5) / 3 = 15) : b - a = 45 := by
  sorry

end interval_length_implies_difference_l188_188424


namespace number_of_candy_packages_l188_188038

theorem number_of_candy_packages (total_candies pieces_per_package : ℕ) 
  (h_total_candies : total_candies = 405)
  (h_pieces_per_package : pieces_per_package = 9) :
  total_candies / pieces_per_package = 45 := by
  sorry

end number_of_candy_packages_l188_188038


namespace product_closest_value_l188_188977

theorem product_closest_value (a b : ℝ) (ha : a = 0.000321) (hb : b = 7912000) :
  abs ((a * b) - 2523) < min (abs ((a * b) - 2500)) (min (abs ((a * b) - 2700)) (min (abs ((a * b) - 3100)) (abs ((a * b) - 2000)))) := by
  sorry

end product_closest_value_l188_188977


namespace fault_line_total_movement_l188_188418

theorem fault_line_total_movement (a b : ℝ) (h1 : a = 1.25) (h2 : b = 5.25) : a + b = 6.50 := by
  -- Definitions:
  rw [h1, h2]
  -- Proof:
  sorry

end fault_line_total_movement_l188_188418


namespace parabola_vertex_l188_188833

theorem parabola_vertex:
  ∃ x y: ℝ, y^2 + 8 * y + 2 * x + 1 = 0 ∧ (x, y) = (7.5, -4) := sorry

end parabola_vertex_l188_188833


namespace find_rate_of_stream_l188_188250

noncomputable def rate_of_stream (v : ℝ) : Prop :=
  let rowing_speed := 36
  let downstream_speed := rowing_speed + v
  let upstream_speed := rowing_speed - v
  (1 / upstream_speed) = 3 * (1 / downstream_speed)

theorem find_rate_of_stream : ∃ v : ℝ, rate_of_stream v ∧ v = 18 :=
by
  use 18
  unfold rate_of_stream
  sorry

end find_rate_of_stream_l188_188250


namespace smaller_number_of_product_l188_188507

theorem smaller_number_of_product :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5610 ∧ a = 34 :=
by
  -- Proof would go here
  sorry

end smaller_number_of_product_l188_188507


namespace solve_equation_l188_188739

def equation (x : ℝ) : Prop := (2 / x + 3 * (4 / x / (8 / x)) = 1.2)

theorem solve_equation : 
  ∃ x : ℝ, equation x ∧ x = - 20 / 3 :=
by
  sorry

end solve_equation_l188_188739


namespace root_of_function_is_four_l188_188444

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_function_is_four (a : ℝ) (h : f a = 0) : a = 4 :=
by
  sorry

end root_of_function_is_four_l188_188444


namespace quadratic_solution_l188_188058

theorem quadratic_solution :
  (∀ x : ℝ, 3 * x^2 - 13 * x + 5 = 0 → 
           x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6) 
  := by
  sorry

end quadratic_solution_l188_188058


namespace volume_of_pyramid_l188_188384

noncomputable def greatest_pyramid_volume (AB AC sin_α : ℝ) (max_angle : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_α = 4 / 5 ∧ max_angle ≤ 60 then
    5 * Real.sqrt 39 / 2
  else
    0

theorem volume_of_pyramid :
  greatest_pyramid_volume 3 5 (4 / 5) 60 = 5 * Real.sqrt 39 / 2 := by
  sorry -- Proof omitted as per instruction

end volume_of_pyramid_l188_188384


namespace bowling_ball_weight_l188_188556

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end bowling_ball_weight_l188_188556


namespace marbles_steve_now_l188_188951
-- Import necessary libraries

-- Define the initial conditions as given in a)
def initial_conditions (sam steve sally : ℕ) := sam = 2 * steve ∧ sally = sam - 5 ∧ sam - 6 = 8

-- Define the proof problem statement
theorem marbles_steve_now (sam steve sally : ℕ) (h : initial_conditions sam steve sally) : steve + 3 = 10 :=
sorry

end marbles_steve_now_l188_188951


namespace five_letter_words_start_end_same_l188_188335

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end five_letter_words_start_end_same_l188_188335


namespace num_solution_pairs_l188_188738

theorem num_solution_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  4 * x + 7 * y = 600 → ∃ n : ℕ, n = 21 :=
by
  sorry

end num_solution_pairs_l188_188738


namespace letters_identity_l188_188207

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l188_188207


namespace garden_area_increase_l188_188100

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l188_188100


namespace product_xyz_l188_188461

theorem product_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * y = 30 * (4:ℝ)^(1/3)) (h5 : x * z = 45 * (4:ℝ)^(1/3)) (h6 : y * z = 18 * (4:ℝ)^(1/3)) :
  x * y * z = 540 * Real.sqrt 3 :=
sorry

end product_xyz_l188_188461


namespace units_digit_of_power_ends_in_nine_l188_188386

theorem units_digit_of_power_ends_in_nine (n : ℕ) (h : (3^n) % 10 = 9) : n % 4 = 2 :=
sorry

end units_digit_of_power_ends_in_nine_l188_188386


namespace solution_exists_l188_188432

noncomputable def find_A_and_B : Prop :=
  ∃ A B : ℚ, 
    (A, B) = (75 / 16, 21 / 16) ∧ 
    ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 → 
    (6 * x + 3) / ((x - 12) * (x + 4)) = A / (x - 12) + B / (x + 4)

theorem solution_exists : find_A_and_B :=
sorry

end solution_exists_l188_188432


namespace vector_addition_correct_l188_188751

variables {A B C D : Type} [AddCommGroup A] [Module ℝ A]

def vector_addition (da cd cb ba : A) : Prop :=
  da + cd - cb = ba

theorem vector_addition_correct (da cd cb ba : A) :
  vector_addition da cd cb ba :=
  sorry

end vector_addition_correct_l188_188751


namespace coin_flip_probability_l188_188689

theorem coin_flip_probability (p : ℝ) 
  (h : p^2 + (1 - p)^2 = 4 * p * (1 - p)) : 
  p = (3 + Real.sqrt 3) / 6 :=
sorry

end coin_flip_probability_l188_188689


namespace cubic_inches_in_one_cubic_foot_l188_188911

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l188_188911


namespace remainder_987654_div_8_l188_188854

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end remainder_987654_div_8_l188_188854


namespace angle_in_first_quadrant_l188_188304

theorem angle_in_first_quadrant (x : ℝ) (h1 : Real.tan x > 0) (h2 : Real.sin x + Real.cos x > 0) : 
  0 < Real.sin x ∧ 0 < Real.cos x := 
by 
  sorry

end angle_in_first_quadrant_l188_188304


namespace find_xyz_l188_188742

theorem find_xyz (x y z : ℝ) :
  x - y + z = 2 ∧
  x^2 + y^2 + z^2 = 30 ∧
  x^3 - y^3 + z^3 = 116 →
  (x = -1 ∧ y = 2 ∧ z = 5) ∨
  (x = -1 ∧ y = -5 ∧ z = -2) ∨
  (x = -2 ∧ y = 1 ∧ z = 5) ∨
  (x = -2 ∧ y = -5 ∧ z = -1) ∨
  (x = 5 ∧ y = 1 ∧ z = -2) ∨
  (x = 5 ∧ y = 2 ∧ z = -1) := by
  sorry

end find_xyz_l188_188742


namespace meals_without_restrictions_l188_188352

theorem meals_without_restrictions (total_clients vegan kosher gluten_free halal dairy_free nut_free vegan_kosher vegan_gluten_free kosher_gluten_free halal_dairy_free gluten_free_nut_free vegan_halal_gluten_free kosher_dairy_free_nut_free : ℕ) 
  (h_tc : total_clients = 80)
  (h_vegan : vegan = 15)
  (h_kosher : kosher = 18)
  (h_gluten_free : gluten_free = 12)
  (h_halal : halal = 10)
  (h_dairy_free : dairy_free = 8)
  (h_nut_free : nut_free = 4)
  (h_vegan_kosher : vegan_kosher = 5)
  (h_vegan_gluten_free : vegan_gluten_free = 6)
  (h_kosher_gluten_free : kosher_gluten_free = 3)
  (h_halal_dairy_free : halal_dairy_free = 4)
  (h_gluten_free_nut_free : gluten_free_nut_free = 2)
  (h_vegan_halal_gluten_free : vegan_halal_gluten_free = 2)
  (h_kosher_dairy_free_nut_free : kosher_dairy_free_nut_free = 1) : 
  (total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free 
  - vegan_kosher - vegan_gluten_free - kosher_gluten_free - halal_dairy_free - gluten_free_nut_free 
  + vegan_halal_gluten_free + kosher_dairy_free_nut_free) = 30) :=
by {
  -- solution steps here
  sorry
}

end meals_without_restrictions_l188_188352


namespace lim_an_times_n_to_zero_l188_188027

-- Definitions of conditions
variable {a_n : ℕ → ℝ}

-- Assume a_n is positive
variable (h_pos : ∀ n, 0 < a_n n)

-- Assume a_n is monotonically decreasing
variable (h_decreasing : ∀ n, a_n (n + 1) ≤ a_n n)

-- Assume the sum of any finite number of terms is not greater than 1
variable (h_sum_le_one : ∀ n, ∑ i in Finset.range (n + 1), a_n i ≤ 1)

-- Prove that \(\lim_{n \rightarrow \infty} n a_{n}=0\)
theorem lim_an_times_n_to_zero :
  Tendsto (λ n : ℕ, n * a_n n) atTop (nhds 0) :=
begin
  sorry,
end

end lim_an_times_n_to_zero_l188_188027


namespace number_of_remaining_grandchildren_l188_188584

-- Defining the given values and conditions
def total_amount : ℕ := 124600
def half_amount : ℕ := total_amount / 2
def amount_per_remaining_grandchild : ℕ := 6230

-- Defining the goal to prove the number of remaining grandchildren
theorem number_of_remaining_grandchildren : (half_amount / amount_per_remaining_grandchild) = 10 := by
  sorry

end number_of_remaining_grandchildren_l188_188584


namespace right_triangle_cosine_l188_188469

theorem right_triangle_cosine (XY XZ YZ : ℝ) (hXY_pos : XY > 0) (hXZ_pos : XZ > 0) (hYZ_pos : YZ > 0)
  (angle_XYZ : angle_1 = 90) (tan_Z : XY / XZ = 5 / 12) : (XZ / YZ = 12 / 13) :=
by
  sorry

end right_triangle_cosine_l188_188469


namespace fraction_sum_l188_188591

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end fraction_sum_l188_188591


namespace correct_answers_max_l188_188604

def max_correct_answers (c w b : ℕ) : Prop :=
  c + w + b = 25 ∧ 4 * c - 3 * w = 40

theorem correct_answers_max : ∃ c w b : ℕ, max_correct_answers c w b ∧ ∀ c', max_correct_answers c' w b → c' ≤ 13 :=
by
  sorry

end correct_answers_max_l188_188604


namespace max_value_of_reciprocals_of_zeros_l188_188903

noncomputable def f (k : ℝ) : ℝ → ℝ :=
  λ x, if x ∈ Ioc 0 1 then k * x^2 + 2 * x - 1 else k * x + 1

theorem max_value_of_reciprocals_of_zeros (k : ℝ) (h : k < 0) (h_pos : k > -1) :
  let x1 := -1 / k in
  let x2 := 1 / (1 + real.sqrt (1 + k)) in
  1 / x1 + 1 / x2 ≤ 9 / 4 :=
begin
  sorry
end

end max_value_of_reciprocals_of_zeros_l188_188903


namespace career_preference_degrees_l188_188843

theorem career_preference_degrees (boys girls : ℕ) (ratio_boys_to_girls : boys / gcd boys girls = 2 ∧ girls / gcd boys girls = 3) 
  (boys_preference : ℕ) (girls_preference : ℕ) 
  (h1 : boys_preference = boys / 3)
  (h2 : girls_preference = 2 * girls / 3) : 
  (boys_preference + girls_preference) / (boys + girls) * 360 = 192 :=
by
  sorry

end career_preference_degrees_l188_188843


namespace grunters_win_4_out_of_6_l188_188957

/-- The Grunters have a probability of winning any given game as 60% --/
def p : ℚ := 3 / 5

/-- The Grunters have a probability of losing any given game as 40% --/
def q : ℚ := 1 - p

/-- The binomial coefficient for choosing exactly 4 wins out of 6 games --/
def binomial_6_4 : ℚ := Nat.choose 6 4

/-- The probability that the Grunters win exactly 4 out of the 6 games --/
def prob_4_wins : ℚ := binomial_6_4 * (p ^ 4) * (q ^ 2)

/--
The probability that the Grunters win exactly 4 out of the 6 games is
exactly $\frac{4860}{15625}$.
--/
theorem grunters_win_4_out_of_6 : prob_4_wins = 4860 / 15625 := by
  sorry

end grunters_win_4_out_of_6_l188_188957


namespace sum_of_edges_112_l188_188072

-- Define the problem parameters
def volume (a b c : ℝ) : ℝ := a * b * c
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)
def sum_of_edges (a b c : ℝ) : ℝ := 4 * (a + b + c)

-- The main theorem 
theorem sum_of_edges_112
  (b s : ℝ) (h1 : volume (b / s) b (b * s) = 512)
  (h2 : surface_area (b / s) b (b * s) = 448)
  (h3 : 0 < b ∧ 0 < s) : 
  sum_of_edges (b / s) b (b * s) = 112 :=
sorry

end sum_of_edges_112_l188_188072


namespace original_cost_of_plants_l188_188871

theorem original_cost_of_plants
  (discount : ℕ)
  (amount_spent : ℕ)
  (original_cost : ℕ)
  (h_discount : discount = 399)
  (h_amount_spent : amount_spent = 68)
  (h_original_cost : original_cost = discount + amount_spent) :
  original_cost = 467 :=
by
  rw [h_discount, h_amount_spent] at h_original_cost
  exact h_original_cost

end original_cost_of_plants_l188_188871


namespace train_speed_l188_188125

/--
Given:
  Length of the train = 500 m
  Length of the bridge = 350 m
  The train takes 60 seconds to completely cross the bridge.

Prove:
  The speed of the train is exactly 14.1667 m/s
-/
theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 500) (h_bridge : length_bridge = 350) (h_time : time = 60) :
  (length_train + length_bridge) / time = 14.1667 :=
by
  rw [h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_l188_188125


namespace garden_area_increase_l188_188105

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l188_188105


namespace modulus_of_z_l188_188575

-- Definitions of the problem conditions
def z := Complex.mk 1 (-1)

-- Statement of the math proof problem
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry -- Proof placeholder

end modulus_of_z_l188_188575


namespace simple_interest_amount_is_58_l188_188467

noncomputable def principal (CI : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  CI / ((1 + r / 100)^t - 1)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t / 100

theorem simple_interest_amount_is_58 (CI : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  CI = 59.45 -> r = 5 -> t = 2 -> P = principal CI r t ->
  simple_interest P r t = 58 :=
by
  sorry

end simple_interest_amount_is_58_l188_188467


namespace abs_abc_eq_one_l188_188942

theorem abs_abc_eq_one 
  (a b c : ℝ)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a)
  (h_eq : a + 1/b^2 = b + 1/c^2 ∧ b + 1/c^2 = c + 1/a^2) : 
  |a * b * c| = 1 := 
sorry

end abs_abc_eq_one_l188_188942


namespace train_and_car_combined_time_l188_188407

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l188_188407


namespace avg_weight_ab_l188_188501

theorem avg_weight_ab (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 30) 
  (h2 : (B + C) / 2 = 28) 
  (h3 : B = 16) : 
  (A + B) / 2 = 25 := 
by 
  sorry

end avg_weight_ab_l188_188501


namespace downstream_speed_l188_188998

variable (Vu Vs Vd Vc : ℝ)

theorem downstream_speed
  (h1 : Vu = 25)
  (h2 : Vs = 32)
  (h3 : Vu = Vs - Vc)
  (h4 : Vd = Vs + Vc) :
  Vd = 39 := by
  sorry

end downstream_speed_l188_188998


namespace math_competition_rankings_l188_188702

noncomputable def rankings (n : ℕ) : ℕ → Prop := sorry

theorem math_competition_rankings :
  (∀ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    
    -- A's guesses
    (rankings A 1 → rankings B 3 ∧ rankings C 5) →
    -- B's guesses
    (rankings B 2 → rankings E 4 ∧ rankings D 5) →
    -- C's guesses
    (rankings C 3 → rankings A 1 ∧ rankings E 4) →
    -- D's guesses
    (rankings D 4 → rankings C 1 ∧ rankings D 2) →
    -- E's guesses
    (rankings E 5 → rankings A 3 ∧ rankings D 4) →
    -- Condition that each position is guessed correctly by someone
    (∃ i, rankings A i) ∧
    (∃ i, rankings B i) ∧
    (∃ i, rankings C i) ∧
    (∃ i, rankings D i) ∧
    (∃ i, rankings E i) →
    
    -- The actual placing according to derived solution
    rankings A 1 ∧ 
    rankings D 2 ∧ 
    rankings B 3 ∧ 
    rankings E 4 ∧ 
    rankings C 5) :=
sorry

end math_competition_rankings_l188_188702


namespace sum_of_reciprocals_of_shifted_roots_l188_188545

noncomputable def cubic_poly (x : ℝ) := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) 
  (ha : cubic_poly a = 0) 
  (hb : cubic_poly b = 0) 
  (hc : cubic_poly c = 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_bounds_a : 0 < a ∧ a < 1)
  (h_bounds_b : 0 < b ∧ b < 1)
  (h_bounds_c : 0 < c ∧ c < 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := 
sorry

end sum_of_reciprocals_of_shifted_roots_l188_188545


namespace stones_in_courtyard_l188_188838

theorem stones_in_courtyard (S T B : ℕ) (h1 : T = S + 3 * S) (h2 : B = 2 * (T + S)) (h3 : B = 400) : S = 40 :=
by
  sorry

end stones_in_courtyard_l188_188838


namespace binom_20_19_eq_20_l188_188725

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l188_188725


namespace new_ratio_first_term_l188_188401

theorem new_ratio_first_term (x : ℕ) (r1 r2 : ℕ) (new_r1 : ℕ) :
  r1 = 4 → r2 = 15 → x = 29 → new_r1 = r1 + x → new_r1 = 33 :=
by
  intros h_r1 h_r2 h_x h_new_r1
  rw [h_r1, h_x] at h_new_r1
  exact h_new_r1

end new_ratio_first_term_l188_188401


namespace determine_y_minus_x_l188_188369

theorem determine_y_minus_x (x y : ℝ) (h1 : x + y = 360) (h2 : x / y = 3 / 5) : y - x = 90 := sorry

end determine_y_minus_x_l188_188369


namespace kayla_apples_correct_l188_188619

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end kayla_apples_correct_l188_188619


namespace parallelogram_area_l188_188985

theorem parallelogram_area (base height : ℕ) (h_base : base = 36) (h_height : height = 24) : base * height = 864 := by
  sorry

end parallelogram_area_l188_188985


namespace binom_20_19_eq_20_l188_188721

theorem binom_20_19_eq_20 (n k : ℕ) (h₁ : n = 20) (h₂ : k = 19)
  (h₃ : ∀ (n k : ℕ), Nat.choose n k = Nat.choose n (n - k))
  (h₄ : ∀ (n : ℕ), Nat.choose n 1 = n) :
  Nat.choose 20 19 = 20 :=
by
  rw [h₁, h₂, h₃ 20 19, Nat.sub_self 19, h₄]
  apply h₄
  sorry

end binom_20_19_eq_20_l188_188721


namespace intersection_A_and_B_l188_188458

-- Define the sets based on the conditions
def setA : Set ℤ := {x : ℤ | x^2 - 2 * x - 8 ≤ 0}
def setB : Set ℤ := {x : ℤ | 1 < Real.log x / Real.log 2}

-- State the theorem (Note: The logarithmic condition should translate the values to integers)
theorem intersection_A_and_B : setA ∩ setB = {3, 4} :=
sorry

end intersection_A_and_B_l188_188458


namespace find_first_term_of_geometric_series_l188_188273

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end find_first_term_of_geometric_series_l188_188273


namespace find_b_l188_188263

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 - b * x + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 + b * x + 3

theorem find_b : (b : ℝ) (h : ∀ x : ℝ, g x b = f (x + 6) b) → b = 12 :=
by
  intros b h
  sorry

end find_b_l188_188263


namespace gcd_459_357_l188_188366

/-- Prove that the greatest common divisor of 459 and 357 is 51. -/
theorem gcd_459_357 : gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l188_188366


namespace area_increase_correct_l188_188096

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l188_188096


namespace geom_sequence_sum_l188_188574

theorem geom_sequence_sum (n : ℕ) (a : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 4 ^ n + a) : 
  a = -1 := 
by
  sorry

end geom_sequence_sum_l188_188574


namespace combination_20_choose_19_eq_20_l188_188728

theorem combination_20_choose_19_eq_20 : nat.choose 20 19 = 20 :=
sorry

end combination_20_choose_19_eq_20_l188_188728


namespace distribute_teachers_l188_188235

theorem distribute_teachers :
  let math_teachers := 3
  let lang_teachers := 3 
  let schools := 2
  let teachers_each_school := 3
  let distribution_plans := 
    (math_teachers.choose 2) * (lang_teachers.choose 1) + 
    (math_teachers.choose 1) * (lang_teachers.choose 2)
  distribution_plans = 18 := 
by
  sorry

end distribute_teachers_l188_188235


namespace imaginary_unit_real_part_eq_l188_188298

theorem imaginary_unit_real_part_eq (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (∃ r : ℝ, ((3 + i) * (a + 2 * i) / (1 + i) = r)) → a = 4 :=
by
  sorry

end imaginary_unit_real_part_eq_l188_188298


namespace equation_solution_l188_188822

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l188_188822


namespace age_sum_proof_l188_188126

theorem age_sum_proof (a b c : ℕ) (h1 : a - (b + c) = 16) (h2 : a^2 - (b + c)^2 = 1632) : a + b + c = 102 :=
by
  sorry

end age_sum_proof_l188_188126


namespace min_omega_condition_l188_188464

theorem min_omega_condition :
  ∃ (ω: ℝ) (k: ℤ), (ω > 0) ∧ (ω = 6 * k + 1 / 2) ∧ (∀ (ω' : ℝ), (ω' > 0) ∧ (∃ (k': ℤ), ω' = 6 * k' + 1 / 2) → ω ≤ ω') := 
sorry

end min_omega_condition_l188_188464


namespace find_min_f_l188_188195

open BigOperators

variable (S : Finset ℕ) (A B : Finset ℕ) (f : ℕ) (C : Finset ℕ)

def symmetric_diff (X Y : Finset ℕ) : Finset ℕ := (X \ Y) ∪ (Y \ X)
def S := (Finset.range 2017).erase 0
def C := A.product B |>.image (λ p => p.1 + p.2)

theorem find_min_f (A_nonempty : A.nonempty) (B_nonempty : B.nonempty) :
  f = |symmetric_diff A S| + |symmetric_diff B S| + |symmetric_diff C S| →
  f = 2017 := sorry

end find_min_f_l188_188195


namespace smallest_natural_number_l188_188744

theorem smallest_natural_number (n : ℕ) (h : 2006 ^ 1003 < n ^ 2006) : n ≥ 45 := 
by {
    sorry
}

end smallest_natural_number_l188_188744


namespace number_of_sequences_with_at_least_two_reds_l188_188553

theorem number_of_sequences_with_at_least_two_reds (n : ℕ) (h : n ≥ 2) :
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2 :=
by
  intros
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  show T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2
  sorry

end number_of_sequences_with_at_least_two_reds_l188_188553


namespace compare_logs_l188_188891

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log 3 / Real.log 5
noncomputable def c := Real.log 5 / Real.log 8

theorem compare_logs : a < b ∧ b < c := by
  sorry

end compare_logs_l188_188891


namespace max_soap_boxes_in_carton_l188_188257

-- Define the measurements of the carton
def L_carton := 25
def W_carton := 42
def H_carton := 60

-- Define the measurements of the soap box
def L_soap_box := 7
def W_soap_box := 12
def H_soap_box := 5

-- Calculate the volume of the carton
def V_carton := L_carton * W_carton * H_carton

-- Calculate the volume of the soap box
def V_soap_box := L_soap_box * W_soap_box * H_soap_box

-- Define the number of soap boxes that can fit in the carton
def number_of_soap_boxes := V_carton / V_soap_box

-- Prove that the number of soap boxes that can fit in the carton is 150
theorem max_soap_boxes_in_carton : number_of_soap_boxes = 150 :=
by
  -- Placeholder for the proof
  sorry

end max_soap_boxes_in_carton_l188_188257


namespace total_food_each_day_l188_188645

-- Definitions as per conditions
def soldiers_first_side : Nat := 4000
def food_per_soldier_first_side : Nat := 10
def soldiers_difference : Nat := 500
def food_difference : Nat := 2

-- Proving the total amount of food
theorem total_food_each_day : 
  let soldiers_second_side := soldiers_first_side - soldiers_difference
  let food_per_soldier_second_side := food_per_soldier_first_side - food_difference
  let total_food_first_side := soldiers_first_side * food_per_soldier_first_side
  let total_food_second_side := soldiers_second_side * food_per_soldier_second_side
  total_food_first_side + total_food_second_side = 68000 := by
  -- Proof is omitted
  sorry

end total_food_each_day_l188_188645


namespace total_seashells_l188_188039

theorem total_seashells 
  (sally_seashells : ℕ)
  (tom_seashells : ℕ)
  (jessica_seashells : ℕ)
  (h1 : sally_seashells = 9)
  (h2 : tom_seashells = 7)
  (h3 : jessica_seashells = 5) : 
  sally_seashells + tom_seashells + jessica_seashells = 21 :=
by
  sorry

end total_seashells_l188_188039


namespace elimination_method_equation_y_l188_188973

theorem elimination_method_equation_y (x y : ℝ)
    (h1 : 5 * x - 3 * y = -5)
    (h2 : 5 * x + 4 * y = -1) :
    7 * y = 4 :=
by
  -- Adding the required conditions as hypotheses and skipping the proof.
  sorry

end elimination_method_equation_y_l188_188973


namespace necessarily_positive_l188_188360

theorem necessarily_positive (x y w : ℝ) (h1 : 0 < x ∧ x < 0.5) (h2 : -0.5 < y ∧ y < 0) (h3 : 0.5 < w ∧ w < 1) : 
  0 < w - y :=
sorry

end necessarily_positive_l188_188360


namespace third_set_candies_l188_188660

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l188_188660


namespace container_capacity_l188_188391

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 36 = 0.75 * C) : 
  C = 80 :=
sorry

end container_capacity_l188_188391


namespace small_cubes_with_painted_faces_l188_188994

-- Definitions based on conditions
def large_cube_edge : ℕ := 8
def small_cube_edge : ℕ := 2
def division_factor : ℕ := large_cube_edge / small_cube_edge
def total_small_cubes : ℕ := division_factor ^ 3

-- Proving the number of cubes with specific painted faces.
theorem small_cubes_with_painted_faces :
  (8 : ℤ) = 8 ∧ -- 8 smaller cubes with three painted faces
  (24 : ℤ) = 24 ∧ -- 24 smaller cubes with two painted faces
  (24 : ℤ) = 24 := -- 24 smaller cubes with one painted face
by
  sorry

end small_cubes_with_painted_faces_l188_188994


namespace cupcake_cookie_price_ratio_l188_188479

theorem cupcake_cookie_price_ratio
  (c k : ℚ)
  (h1 : 5 * c + 3 * k = 23)
  (h2 : 4 * c + 4 * k = 21) :
  k / c = 13 / 29 :=
  sorry

end cupcake_cookie_price_ratio_l188_188479


namespace Joan_bought_72_eggs_l188_188021

def dozen := 12
def dozens_Joan_bought := 6
def eggs_Joan_bought := dozens_Joan_bought * dozen

theorem Joan_bought_72_eggs : eggs_Joan_bought = 72 := by
  sorry

end Joan_bought_72_eggs_l188_188021


namespace john_total_animals_is_114_l188_188794

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end john_total_animals_is_114_l188_188794


namespace vic_max_marks_l188_188074

theorem vic_max_marks (M : ℝ) (h : 0.92 * M = 368) : M = 400 := 
sorry

end vic_max_marks_l188_188074


namespace arithmetic_sequence_S10_l188_188606

-- Definition of an arithmetic sequence and the corresponding sums S_n.
def is_arithmetic_sequence (S : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, S (n + 1) = S n + d

theorem arithmetic_sequence_S10 
  (S : ℕ → ℕ)
  (h1 : S 1 = 10)
  (h2 : S 2 = 20)
  (h_arith : is_arithmetic_sequence S) :
  S 10 = 100 :=
sorry

end arithmetic_sequence_S10_l188_188606


namespace solve_equation_l188_188814

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l188_188814


namespace dot_product_result_parallelism_condition_l188_188907

-- Definitions of the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

-- 1. Prove the dot product result
theorem dot_product_result :
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_2b := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  a_plus_b.1 * a_minus_2b.1 + a_plus_b.2 * a_minus_2b.2 = -14 :=
by
  sorry

-- 2. Prove parallelism condition
theorem parallelism_condition (k : ℝ) :
  let k_a_plus_b := (k * a.1 + b.1, k * a.2 + b.2)
  let a_minus_3b := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  k = -1/3 → k_a_plus_b.1 * a_minus_3b.2 = k_a_plus_b.2 * a_minus_3b.1 :=
by
  sorry

end dot_product_result_parallelism_condition_l188_188907


namespace ariana_carnations_l188_188712

theorem ariana_carnations 
  (total_flowers: ℕ) 
  (fraction_roses: ℚ) 
  (num_tulips: ℕ) 
  (num_roses := (fraction_roses * total_flowers)) 
  (num_roses_int: num_roses.natAbs = 16) 
  (num_flowers_roses_tulips := (num_roses + num_tulips)) 
  (num_carnations := total_flowers - num_flowers_roses_tulips) : 
  total_flowers = 40 → 
  fraction_roses = 2 / 5 → 
  num_tulips = 10 → 
  num_roses_int = 16 → 
  num_carnations = 14 :=
by
  intros ht hf htul hros
  sorry

end ariana_carnations_l188_188712


namespace root_implies_value_l188_188167

theorem root_implies_value (b c : ℝ) (h : 2 * b - c = 4) : 4 * b - 2 * c + 1 = 9 :=
by
  sorry

end root_implies_value_l188_188167


namespace sum_of_corners_of_9x9_grid_l188_188255

theorem sum_of_corners_of_9x9_grid : 
    let topLeft := 1
    let topRight := 9
    let bottomLeft := 73
    let bottomRight := 81
    topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  sorry
}

end sum_of_corners_of_9x9_grid_l188_188255


namespace probability_single_trial_l188_188607

theorem probability_single_trial (p : ℚ) (h₁ : (1 - p)^4 = 16 / 81) : p = 1 / 3 :=
sorry

end probability_single_trial_l188_188607


namespace total_distance_walked_l188_188808

theorem total_distance_walked (t1 t2 : ℝ) (r : ℝ) (total_distance : ℝ)
  (h1 : t1 = 15 / 60)  -- Convert 15 minutes to hours
  (h2 : t2 = 25 / 60)  -- Convert 25 minutes to hours
  (h3 : r = 3)         -- Average speed in miles per hour
  (h4 : total_distance = r * (t1 + t2))
  : total_distance = 2 :=
by
  -- here is where the proof would go
  sorry

end total_distance_walked_l188_188808


namespace ratio_of_shaded_to_non_shaded_l188_188651

open Real

-- Define the midpoints and the necessary variables
structure Point (x y : ℝ)

def midpoint (P Q : Point) : Point :=
  Point ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Let Triangle ABC has coordinates A(0, 0), B(6, 0), C(0, 8)
def A : Point := ⟨0, 0⟩
def B : Point := ⟨6, 0⟩
def C : Point := ⟨0, 8⟩
def D : Point := midpoint A B -- midpoint of AB
def F : Point := midpoint A C -- midpoint of AC
def E : Point := midpoint B C -- midpoint of BC
def G : Point := midpoint D F -- midpoint of DF
def H : Point := midpoint F E -- midpoint of FE

noncomputable def triangle_area (P Q R : Point) : ℝ :=
  abs ((P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)) / 2)

-- Calculate areas
noncomputable def area_ABC : ℝ := triangle_area A B C
noncomputable def area_DFG : ℝ := triangle_area D F G
noncomputable def area_FEH : ℝ := triangle_area F E H

noncomputable def shaded_area : ℝ := area_DFG + area_FEH
noncomputable def non_shaded_area : ℝ := area_ABC - shaded_area

noncomputable def ratio_shaded_to_non_shaded : ℚ :=
  (shaded_area / non_shaded_area).toRat

theorem ratio_of_shaded_to_non_shaded :
  ratio_shaded_to_non_shaded = (23 / 25 : ℚ) := by
    -- This is where the proof would go
    sorry

end ratio_of_shaded_to_non_shaded_l188_188651


namespace p3_mp_odd_iff_m_even_l188_188196

theorem p3_mp_odd_iff_m_even (p m : ℕ) (hp : p % 2 = 1) : (p^3 + m * p) % 2 = 1 ↔ m % 2 = 0 := sorry

end p3_mp_odd_iff_m_even_l188_188196


namespace garden_area_difference_l188_188109

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l188_188109


namespace part_a_part_b_l188_188524

theorem part_a (x y : ℕ) (h : x^3 + 5 * y = y^3 + 5 * x) : x = y :=
sorry

theorem part_b : ∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (x^3 + 5 * y = y^3 + 5 * x) :=
sorry

end part_a_part_b_l188_188524


namespace number_of_rabbits_l188_188925

-- Given conditions
variable (r c : ℕ)
variable (cond1 : r + c = 51)
variable (cond2 : 4 * r = 3 * (2 * c) + 4)

-- To prove
theorem number_of_rabbits : r = 31 :=
sorry

end number_of_rabbits_l188_188925


namespace total_votes_cast_correct_l188_188788

noncomputable def total_votes_cast : Nat :=
  let total_valid_votes : Nat := 1050
  let spoiled_votes : Nat := 325
  total_valid_votes + spoiled_votes

theorem total_votes_cast_correct :
  total_votes_cast = 1375 := by
  sorry

end total_votes_cast_correct_l188_188788


namespace find_LN_l188_188219

noncomputable def LM : ℝ := 9
noncomputable def sin_N : ℝ := 3 / 5
noncomputable def LN : ℝ := 15

theorem find_LN (h₁ : sin_N = 3 / 5) (h₂ : LM = 9) (h₃ : sin_N = LM / LN) : LN = 15 :=
by
  sorry

end find_LN_l188_188219


namespace minimum_value_expression_l188_188437

theorem minimum_value_expression : 
  ∀ (a b : ℝ), (a > 0) → (b > 0) → 
  ( ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (a_0 b_0 : ℝ) = 4
  (3 * a * b - 6 * b + a * (1 - a))^2 + (9 * b^2 + 2 * a + 3 * b * (1 - a))^2 / (a^2 + 9 * b^2) = 4 
sory

end minimum_value_expression_l188_188437


namespace sum_a3_a4_eq_14_l188_188303

open Nat

-- Define variables
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem sum_a3_a4_eq_14 : a 3 + a 4 = 14 := by
  sorry

end sum_a3_a4_eq_14_l188_188303


namespace a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l188_188068

-- Definitions for the point P and movements
def move (P : ℤ) (flip : Bool) : ℤ :=
  if flip then P + 1 else -P

-- Definitions for probabilities
def probability_of_event (events : ℕ) (successful : ℕ) : ℚ :=
  successful / events

def probability_a3_zero : ℚ :=
  probability_of_event 8 2  -- 2 out of 8 sequences lead to a3 = 0

def probability_a4_one : ℚ :=
  probability_of_event 16 2  -- 2 out of 16 sequences lead to a4 = 1

noncomputable def probability_an_n_minus_3 (n : ℕ) : ℚ :=
  if n < 3 then 0 else (n - 1) / (2 ^ n)

-- Statements to prove
theorem a3_probability_is_one_fourth : probability_a3_zero = 1/4 := by
  sorry

theorem a4_probability_is_one_eighth : probability_a4_one = 1/8 := by
  sorry

theorem an_n_minus_3_probability (n : ℕ) (hn : n ≥ 3) : probability_an_n_minus_3 n = (n - 1) / (2^n) := by
  sorry

end a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l188_188068


namespace simplify_fraction_l188_188361

theorem simplify_fraction (x : ℝ) (h : x ≠ 0) : 
  (x ^ (3 / 4) - 25 * x ^ (1 / 4)) / (x ^ (1 / 2) + 5 * x ^ (1 / 4)) = x ^ (1 / 4) - 5 :=
by
  sorry

end simplify_fraction_l188_188361


namespace textbook_weight_difference_l188_188618

variable (chemWeight : ℝ) (geomWeight : ℝ)

def chem_weight := chemWeight = 7.12
def geom_weight := geomWeight = 0.62

theorem textbook_weight_difference : chemWeight - geomWeight = 6.50 :=
by
  sorry

end textbook_weight_difference_l188_188618


namespace part_1_part_2_l188_188162

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x + a) + 2 * a

theorem part_1 (h : ∀ x : ℝ, f x a = f (3 - x) a) : a = -3 :=
by
  sorry

theorem part_2 (h : ∃ x : ℝ, f x a ≤ -abs (2 * x - 1) + a) : a ≤ -1 / 2 :=
by
  sorry

end part_1_part_2_l188_188162


namespace train_and_car_combined_time_l188_188412

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l188_188412


namespace prime_divisor_of_form_l188_188396

theorem prime_divisor_of_form (a p : ℕ) (hp1 : a > 0) (hp2 : Prime p) (hp3 : p ∣ (a^3 - 3 * a + 1)) (hp4 : p ≠ 3) :
  ∃ k : ℤ, p = 9 * k + 1 ∨ p = 9 * k - 1 :=
by
  sorry

end prime_divisor_of_form_l188_188396


namespace first_discount_percentage_l188_188228

theorem first_discount_percentage (x : ℕ) :
  let original_price := 175
  let discounted_price := original_price * (100 - x) / 100
  let final_price := discounted_price * 95 / 100
  final_price = 133 → x = 20 :=
by
  sorry

end first_discount_percentage_l188_188228


namespace total_candies_in_third_set_l188_188670

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l188_188670


namespace line_equation_slope_intercept_l188_188960

theorem line_equation_slope_intercept (m b : ℝ) (h1 : m = -1) (h2 : b = -1) :
  ∀ x y : ℝ, y = m * x + b → x + y + 1 = 0 :=
by
  intros x y h
  sorry

end line_equation_slope_intercept_l188_188960


namespace height_of_C_l188_188500

noncomputable def height_A_B_C (h_A h_B h_C : ℝ) : Prop := 
  (h_A + h_B + h_C) / 3 = 143 ∧ 
  h_A + 4.5 = (h_B + h_C) / 2 ∧ 
  h_B = h_C + 3

theorem height_of_C (h_A h_B h_C : ℝ) (h : height_A_B_C h_A h_B h_C) : h_C = 143 :=
  sorry

end height_of_C_l188_188500


namespace find_m_l188_188779

theorem find_m (m x : ℝ) 
  (h1 : (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) 
  (h2 : m^2 - 3 * m + 2 = 0)
  (h3 : m ≠ 1) : 
  m = 2 := 
sorry

end find_m_l188_188779


namespace jake_total_payment_l188_188935

-- Definitions based on conditions
def packages : ℕ := 3
def weight_per_package : ℕ := 2
def price_per_pound : ℕ := 4

-- Theorem to prove the total cost
theorem jake_total_payment : 
  let total_pounds := packages * weight_per_package in
  let total_cost := total_pounds * price_per_pound in
  total_cost = 24 :=
by
  sorry

end jake_total_payment_l188_188935


namespace matrix_operation_correct_l188_188540

open Matrix

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 5]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 4], ![0, -3]]
def matrix3 : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 0], ![-1, 8]]
def result : Matrix (Fin 2) (Fin 2) ℤ := ![![12, -7], ![1, 16]]

theorem matrix_operation_correct:
  matrix1 - matrix2 + matrix3 = result :=
by
  sorry

end matrix_operation_correct_l188_188540


namespace distance_between_Sasha_and_Kolya_l188_188041

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l188_188041


namespace partition_X_l188_188565

namespace ProofProblem

open Finset

-- Given integer n ≥ 3
variable (n : ℤ) (h_n : n ≥ 3)

-- Define the set X = {1, 2, ..., n^2 - n}
def X : Finset ℤ := (range (n^2 - n + 1)).filter (λ x, x > 0)

-- Define the condition that no subset with n elements satisfies the given inequality
def invalid_subset (s : Finset ℤ) : Prop :=
  ∃ (a : ℤ → ℤ), ∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ s ∧ a 1 < a 2 ∧ a 2 < a 3 ∧ a (n - 1) < a n ∧
    ∀ k, 2 ≤ k ∧ k < n → a k ≤ (a (k - 1) + a (k + 1)) / 2

-- Prove that X can be partitioned into two disjoint subsets such that neither contains the invalid subset
theorem partition_X : 
  ∃ (S T : Finset ℤ), S ≠ ∅ ∧ T ≠ ∅ ∧ S ∩ T = ∅ ∧ S ∪ T = X n ∧ ¬invalid_subset n S ∧ ¬invalid_subset n T :=
sorry

end ProofProblem

end partition_X_l188_188565


namespace additional_flowers_grew_l188_188139

-- Define the initial conditions
def initial_flowers : ℕ := 10  -- Dane’s two daughters planted 5 flowers each (5 + 5).
def flowers_died : ℕ := 10     -- 10 flowers died.
def baskets : ℕ := 5
def flowers_per_basket : ℕ := 4

-- Total flowers harvested (from the baskets)
def total_harvested : ℕ := baskets * flowers_per_basket  -- 5 * 4 = 20

-- The proof to show additional flowers grown
theorem additional_flowers_grew : (total_harvested - initial_flowers + flowers_died) = 10 :=
by
  -- The final number of flowers and the initial number of flowers are known
  have final_flowers : ℕ := total_harvested
  have initial_plus_grown : ℕ := initial_flowers + (total_harvested - initial_flowers)
  -- Show the equality that defines the additional flowers grown
  show (total_harvested - initial_flowers + flowers_died) = 10
  sorry

end additional_flowers_grew_l188_188139


namespace solution_set_of_inequality_l188_188070

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 4 * x - 3 > 0 } = { x : ℝ | 1 < x ∧ x < 3 } := sorry

end solution_set_of_inequality_l188_188070


namespace polynomial_inequality_holds_l188_188439

def polynomial (x : ℝ) : ℝ := x^6 + 4 * x^5 + 2 * x^4 - 6 * x^3 - 2 * x^2 + 4 * x - 1

theorem polynomial_inequality_holds (x : ℝ) :
  (x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2) →
  polynomial x ≥ 0 :=
by
  sorry

end polynomial_inequality_holds_l188_188439


namespace proof_problem_l188_188564

noncomputable theory
open Classical

def quadratic_symmetric (a b : ℝ) := ∀ x, a * x^2 + b * x = a * (-x-2)^2 + b * (-x-2)

def tangent_graph (a b : ℝ) := ∃ x, a * x^2 + b * x = x

def analytical_expression_correct := ∃ a b : ℝ, a ≠ 0 ∧ quadratic_symmetric a b ∧ tangent_graph a b ∧ (a = 1/2 ∧ b = 1)

def inequality_solution := ∀ t x : ℝ, abs t ≤ 2 → (π^(1/2 * x^2 + x) > (1/π)^(2 - t*x)) ↔ ((x < -3 - real.sqrt 5) ∨ (x > -3 + real.sqrt 5))

theorem proof_problem : analytical_expression_correct ∧ inequality_solution :=
sorry

end proof_problem_l188_188564


namespace NaCl_moles_formed_l188_188288

-- Definitions for the conditions
def NaOH_moles : ℕ := 2
def Cl2_moles : ℕ := 1

-- Chemical reaction of NaOH and Cl2 resulting in NaCl and H2O
def reaction (n_NaOH n_Cl2 : ℕ) : ℕ :=
  if n_NaOH = 2 ∧ n_Cl2 = 1 then 2 else 0

-- Statement to be proved
theorem NaCl_moles_formed : reaction NaOH_moles Cl2_moles = 2 :=
by
  sorry

end NaCl_moles_formed_l188_188288


namespace symmetric_polynomial_representation_l188_188344

noncomputable def f : ℝ × ℝ × ℝ → ℝ
noncomputable def g1 (x y z: ℝ) : ℝ := x * (x - y) * (x - z) + y * (y - z) * (y - x) + z * (z - x) * (z - y)
noncomputable def g2 (x y z: ℝ) : ℝ := (y + z) * (x - y) * (x - z) + (z + x) * (y - z) * (y - x) + (x + y) * (z - x) * (z - y)
noncomputable def g3 (x y z: ℝ) : ℝ := x * y * z

theorem symmetric_polynomial_representation 
  (f : ℝ × ℝ × ℝ → ℝ) 
  (h_sym : ∀ x y z : ℝ, f (x, y, z) = f (y, z, x)) 
  : ∃ (a b c : ℝ), (∀ x y z : ℝ, f (x, y, z) = a * g1 x y z + b * g2 x y z + c * g3 x y z) → 
    (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → f (x, y, z) ≥ 0 ↔ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :=
begin
  sorry
end

end symmetric_polynomial_representation_l188_188344


namespace three_solutions_no_solutions_2891_l188_188576

theorem three_solutions (n : ℤ) (hpos : n > 0) (hx : ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3 * x1 * y1^2 + y1^3 = n ∧ 
    x2^3 - 3 * x2 * y2^2 + y2^3 = n ∧ 
    x3^3 - 3 * x3 * y3^2 + y3^3 = n := 
sorry

theorem no_solutions_2891 : ¬ ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end three_solutions_no_solutions_2891_l188_188576


namespace speed_of_boat_in_still_water_l188_188509

theorem speed_of_boat_in_still_water :
  ∀ (v : ℚ), (33 = (v + 3) * (44 / 60)) → v = 42 := 
by
  sorry

end speed_of_boat_in_still_water_l188_188509


namespace probability_on_between_correct_l188_188989

noncomputable def probability_on_between (t : ℝ) : ℝ :=
  if 4 < t ∧ t < 5 then 7/20 else 0

theorem probability_on_between_correct :
    ∀ t, probability_on_between t = if 4 < t ∧ t < 5 then 7/20 else 0 := by
  intro t
  sorry

end probability_on_between_correct_l188_188989


namespace train_speed_l188_188867

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_conversion_factor : ℝ) (expected_speed : ℝ) (h_time_conversion : time_conversion_factor = 1 / 60) (h_time : time_minutes / 60 = 0.5) (h_distance : distance = 51) (h_expected_speed : expected_speed = 102) : distance / (time_minutes / 60) = expected_speed :=
by 
  sorry

end train_speed_l188_188867


namespace find_t_l188_188449

variable (a t : ℝ)

def f (x : ℝ) : ℝ := a * x + 19

theorem find_t (h1 : f a 3 = 7) (h2 : f a t = 15) : t = 1 :=
by
  sorry

end find_t_l188_188449


namespace diagonals_perpendicular_l188_188764

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 3 }
def B : Point := { x := 2, y := 6 }
def C : Point := { x := 6, y := -1 }
def D : Point := { x := -3, y := -4 }

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem diagonals_perpendicular :
  let AC := vector A C
  let BD := vector B D
  dot_product AC BD = 0 :=
by
  let AC := vector A C
  let BD := vector B D
  sorry

end diagonals_perpendicular_l188_188764


namespace letters_identity_l188_188209

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l188_188209


namespace cost_per_minute_of_each_call_l188_188873

theorem cost_per_minute_of_each_call :
  let calls_per_week := 50
  let hours_per_call := 1
  let weeks_per_month := 4
  let total_hours_in_month := calls_per_week * hours_per_call * weeks_per_month
  let total_cost := 600
  let cost_per_hour := total_cost / total_hours_in_month
  let minutes_per_hour := 60
  let cost_per_minute := cost_per_hour / minutes_per_hour
  cost_per_minute = 0.05 := 
by
  sorry

end cost_per_minute_of_each_call_l188_188873


namespace first_term_geometric_series_l188_188270

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end first_term_geometric_series_l188_188270


namespace sampling_interval_l188_188848

theorem sampling_interval (total_students sample_size k : ℕ) (h1 : total_students = 1200) (h2 : sample_size = 40) (h3 : k = total_students / sample_size) : k = 30 :=
by
  sorry

end sampling_interval_l188_188848


namespace lcm_48_147_l188_188144

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := sorry

end lcm_48_147_l188_188144


namespace weight_of_each_package_l188_188226

theorem weight_of_each_package (W : ℝ) 
  (h1: 10 * W + 7 * W + 8 * W = 100) : W = 4 :=
by
  sorry

end weight_of_each_package_l188_188226


namespace slope_ge_one_sum_pq_eq_17_l188_188482

noncomputable def Q_prob_satisfaction : ℚ := 1/16

theorem slope_ge_one_sum_pq_eq_17 :
  let p := 1
  let q := 16
  p + q = 17 := by
  sorry

end slope_ge_one_sum_pq_eq_17_l188_188482


namespace jean_jail_time_l188_188190

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l188_188190


namespace compound_interest_semiannual_l188_188526

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_semiannual :
  compound_interest 150 0.20 2 1 = 181.50 :=
by
  sorry

end compound_interest_semiannual_l188_188526


namespace bill_left_with_22_l188_188129

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end bill_left_with_22_l188_188129


namespace cow_manure_plant_height_l188_188792

theorem cow_manure_plant_height
  (control_plant_height : ℝ)
  (bone_meal_ratio : ℝ)
  (cow_manure_ratio : ℝ)
  (h1 : control_plant_height = 36)
  (h2 : bone_meal_ratio = 1.25)
  (h3 : cow_manure_ratio = 2) :
  (control_plant_height * bone_meal_ratio * cow_manure_ratio) = 90 :=
sorry

end cow_manure_plant_height_l188_188792


namespace creative_sum_l188_188066

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end creative_sum_l188_188066


namespace find_y_value_l188_188778

theorem find_y_value 
  (k : ℝ) 
  (y : ℝ) 
  (hx81 : y = 3 * Real.sqrt 2)
  (h_eq : ∀ (x : ℝ), y = k * x ^ (1 / 4)) 
  : (∃ y, y = 2 ∧ y = k * 4 ^ (1 / 4))
:= sorry

end find_y_value_l188_188778


namespace net_income_correct_l188_188377

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end net_income_correct_l188_188377


namespace tan_half_sum_l188_188025

theorem tan_half_sum (p q : ℝ)
  (h1 : Real.cos p + Real.cos q = (1:ℝ)/3)
  (h2 : Real.sin p + Real.sin q = (8:ℝ)/17) :
  Real.tan ((p + q) / 2) = (24:ℝ)/17 := 
sorry

end tan_half_sum_l188_188025


namespace candy_count_in_third_set_l188_188667

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l188_188667


namespace train_length_is_400_l188_188983

-- Conditions from a)
def train_speed_kmph : ℕ := 180
def crossing_time_sec : ℕ := 8

-- The corresponding length in meters
def length_of_train : ℕ := 400

-- The problem statement to prove
theorem train_length_is_400 :
  (train_speed_kmph * 1000 / 3600) * crossing_time_sec = length_of_train := by
  -- Proof is skipped as per the requirement
  sorry

end train_length_is_400_l188_188983


namespace additional_students_needed_l188_188692

theorem additional_students_needed 
  (n : ℕ) 
  (r : ℕ) 
  (t : ℕ) 
  (h_n : n = 82) 
  (h_r : r = 2) 
  (h_t : t = 49) : 
  (t - n / r) * r = 16 := 
by 
  sorry

end additional_students_needed_l188_188692


namespace total_candies_in_third_set_l188_188678

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l188_188678


namespace subtraction_of_decimals_l188_188714

theorem subtraction_of_decimals : 7.42 - 2.09 = 5.33 := 
by
  sorry

end subtraction_of_decimals_l188_188714


namespace geom_seq_frac_l188_188158

noncomputable def geom_seq_sum (a1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_frac (a1 q : ℕ) (hq : q > 1) (h_sum : a1 * (q ^ 3 + q ^ 6 + 1 + q + q ^ 2 + q ^ 5) = 20)
  (h_prod : a1 ^ 7 * q ^ (3 + 6) = 64) :
  geom_seq_sum a1 q 6 / geom_seq_sum a1 q 9 = 5 / 21 :=
by
  sorry

end geom_seq_frac_l188_188158


namespace trapezoid_area_l188_188611

theorem trapezoid_area (A B : ℝ) (n : ℕ) (hA : A = 36) (hB : B = 4) (hn : n = 6) :
    (A - B) / n = 5.33 := 
by 
  -- Given conditions and the goal
  sorry

end trapezoid_area_l188_188611


namespace third_set_candies_l188_188663

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l188_188663


namespace square_area_l188_188362

theorem square_area (XY ZQ : ℕ) (inscribed_square : Prop) : (XY = 35) → (ZQ = 65) → inscribed_square → ∃ (a : ℕ), a^2 = 2275 :=
by
  intros hXY hZQ hinscribed
  use 2275
  sorry

end square_area_l188_188362


namespace solution_set_of_inequality_l188_188308

theorem solution_set_of_inequality 
  {f : ℝ → ℝ}
  (hf : ∀ x y : ℝ, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  {x : ℝ | |f (x - 2)| > 2 } = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end solution_set_of_inequality_l188_188308


namespace total_bees_including_queen_at_end_of_14_days_l188_188860

-- Conditions definitions
def bees_hatched_per_day : ℕ := 5000
def bees_lost_per_day : ℕ := 1800
def duration_days : ℕ := 14
def initial_bees : ℕ := 20000
def queen_bees : ℕ := 1

-- Question statement as Lean theorem
theorem total_bees_including_queen_at_end_of_14_days :
  (initial_bees + (bees_hatched_per_day - bees_lost_per_day) * duration_days + queen_bees) = 64801 := 
by
  sorry

end total_bees_including_queen_at_end_of_14_days_l188_188860


namespace difference_of_distances_l188_188528

-- Definition of John's walking distance to school
def John_distance : ℝ := 0.7

-- Definition of Nina's walking distance to school
def Nina_distance : ℝ := 0.4

-- Assertion that the difference in walking distance is 0.3 miles
theorem difference_of_distances : (John_distance - Nina_distance) = 0.3 := 
by 
  sorry

end difference_of_distances_l188_188528


namespace percentage_relationship_l188_188319

theorem percentage_relationship (a b : ℝ) (h : a = 1.2 * b) : ¬ (b = 0.8 * a) :=
by
  -- assumption: a = 1.2 * b
  -- goal: ¬ (b = 0.8 * a)
  sorry

end percentage_relationship_l188_188319


namespace candy_count_in_third_set_l188_188665

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l188_188665


namespace exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l188_188932

theorem exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987 :
  ∃ n : ℕ, n ^ n + (n + 1) ^ n ≡ 0 [MOD 1987] := sorry

end exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l188_188932


namespace garden_area_increase_l188_188098

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l188_188098


namespace evaluate_expression_l188_188944

open Complex

theorem evaluate_expression (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + a * b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 18 :=
by
  sorry

end evaluate_expression_l188_188944


namespace count_five_letter_words_l188_188337

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end count_five_letter_words_l188_188337


namespace third_set_candies_l188_188659

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l188_188659


namespace jake_total_distance_l188_188018

noncomputable def jake_rate : ℝ := 4 -- Jake's walking rate in miles per hour
noncomputable def total_time : ℝ := 2 -- Jake's total walking time in hours
noncomputable def break_time : ℝ := 0.5 -- Jake's break time in hours

theorem jake_total_distance :
  jake_rate * (total_time - break_time) = 6 :=
by
  sorry

end jake_total_distance_l188_188018


namespace distance_between_foci_of_hyperbola_l188_188222

theorem distance_between_foci_of_hyperbola :
  (∀ x y : ℝ, (y = 2 * x + 3) ∨ (y = -2 * x + 1)) →
  ∀ p : ℝ × ℝ, (p = (2, 1)) →
  ∃ d : ℝ, d = 2 * Real.sqrt 30 :=
by
  sorry

end distance_between_foci_of_hyperbola_l188_188222


namespace tan_a3a5_equals_sqrt3_l188_188447

noncomputable def geometric_seq_property (a: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r -- for some common ratio r

theorem tan_a3a5_equals_sqrt3 (a : ℕ → ℝ) 
  (h_geom : geometric_seq_property a)
  (h_cond : a 2 * a 6 + 2 * (a 4)^2 = real.pi) :
  real.tan (a 3 * a 5) = real.sqrt 3 :=
sorry

end tan_a3a5_equals_sqrt3_l188_188447


namespace find_rate_of_interest_l188_188987

variable (P : ℝ) (R : ℝ) (T : ℕ := 2)

-- Condition for Simple Interest (SI = Rs. 660 for 2 years)
def simple_interest :=
  P * R * ↑T / 100 = 660

-- Condition for Compound Interest (CI = Rs. 696.30 for 2 years)
def compound_interest :=
  P * ((1 + R / 100) ^ T - 1) = 696.30

-- We need to prove that R = 11
theorem find_rate_of_interest (P : ℝ) (h1 : simple_interest P R) (h2 : compound_interest P R) : 
  R = 11 := by
  sorry

end find_rate_of_interest_l188_188987


namespace solve_equation_l188_188818

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l188_188818


namespace area_increase_correct_l188_188095

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l188_188095


namespace even_function_f_l188_188572

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem even_function_f (hx : ∀ x : ℝ, f (-x) = f x) 
  (hg : ∀ x : ℝ, g (-x) = -g x)
  (h_pass : g (-1) = 1)
  (hg_eq_f : ∀ x : ℝ, g x = f (x - 1)) 
  : f 7 + f 8 = -1 := 
by
  sorry

end even_function_f_l188_188572


namespace binom_20_19_eq_20_l188_188723

theorem binom_20_19_eq_20 : nat.choose 20 19 = 20 := sorry

end binom_20_19_eq_20_l188_188723


namespace find_first_number_l188_188959

theorem find_first_number (x : ℕ) : 
    (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end find_first_number_l188_188959


namespace no_valid_coloring_l188_188017

theorem no_valid_coloring (colors : Fin 4 → Prop) (board : Fin 5 → Fin 5 → Fin 4) :
  (∀ i j : Fin 5, ∃ c1 c2 c3 : Fin 4, 
    (c1 ≠ c2) ∧ (c2 ≠ c3) ∧ (c1 ≠ c3) ∧ 
    (board i j = c1 ∨ board i j = c2 ∨ board i j = c3)) → False :=
by
  sorry

end no_valid_coloring_l188_188017


namespace ordered_triples_2022_l188_188863

theorem ordered_triples_2022 :
  ∃ n : ℕ, n = 13 ∧ (∃ a c : ℕ, a ≤ c ∧ (a * c = 2022^2)) := by
  sorry

end ordered_triples_2022_l188_188863


namespace triangle_angle_contradiction_l188_188387

theorem triangle_angle_contradiction (α β γ : ℝ) (h : α + β + γ = 180) :
  (α > 60 ∧ β > 60 ∧ γ > 60) -> false :=
by
  sorry

end triangle_angle_contradiction_l188_188387


namespace truck_capacity_solution_l188_188970

variable (x y : ℝ)

theorem truck_capacity_solution (h1 : 3 * x + 4 * y = 22) (h2 : 2 * x + 6 * y = 23) :
  x + y = 6.5 := sorry

end truck_capacity_solution_l188_188970


namespace min_value_expression_l188_188347

open Real

theorem min_value_expression (x y z: ℝ) (h1: 0 < x) (h2: 0 < y) (h3: 0 < z)
    (h4: (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10):
    (x / y + y / z + z / x) * (y / x + z / y + x / z) = 25 :=
by
  sorry

end min_value_expression_l188_188347


namespace expression_defined_if_x_not_3_l188_188599

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end expression_defined_if_x_not_3_l188_188599


namespace total_males_below_50_is_2638_l188_188773

def branchA_total_employees := 4500
def branchA_percentage_males := 60 / 100
def branchA_percentage_males_at_least_50 := 40 / 100

def branchB_total_employees := 3500
def branchB_percentage_males := 50 / 100
def branchB_percentage_males_at_least_50 := 55 / 100

def branchC_total_employees := 2200
def branchC_percentage_males := 35 / 100
def branchC_percentage_males_at_least_50 := 70 / 100

def males_below_50_branchA := (1 - branchA_percentage_males_at_least_50) * (branchA_percentage_males * branchA_total_employees)
def males_below_50_branchB := (1 - branchB_percentage_males_at_least_50) * (branchB_percentage_males * branchB_total_employees)
def males_below_50_branchC := (1 - branchC_percentage_males_at_least_50) * (branchC_percentage_males * branchC_total_employees)

def total_males_below_50 := males_below_50_branchA + males_below_50_branchB + males_below_50_branchC

theorem total_males_below_50_is_2638 : total_males_below_50 = 2638 := 
by
  -- Numerical evaluation and equality verification here
  sorry

end total_males_below_50_is_2638_l188_188773


namespace jessica_remaining_time_after_penalties_l188_188793

-- Definitions for the given conditions
def questions_answered : ℕ := 16
def total_questions : ℕ := 80
def time_used_minutes : ℕ := 12
def exam_duration_minutes : ℕ := 60
def penalty_per_incorrect_answer_minutes : ℕ := 2

-- Define the rate of answering questions
def answering_rate : ℚ := questions_answered / time_used_minutes

-- Define the total time needed to answer all questions
def total_time_needed : ℚ := total_questions / answering_rate

-- Define the remaining time after penalties
def remaining_time_after_penalties (x : ℕ) : ℤ :=
  max 0 (0 - penalty_per_incorrect_answer_minutes * x)

-- The theorem to prove
theorem jessica_remaining_time_after_penalties (x : ℕ) : 
  remaining_time_after_penalties x = max 0 (0 - penalty_per_incorrect_answer_minutes * x) := 
by
  sorry

end jessica_remaining_time_after_penalties_l188_188793


namespace certain_event_C_union_D_l188_188922

variable {Ω : Type} -- Omega, the sample space
variable {P : Set Ω → Prop} -- P as the probability function predicates the events

-- Definitions of the events
variable {A B C D : Set Ω}

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ x, x ∈ A → x ∉ B
def complementary (A C : Set Ω) : Prop := ∀ x, x ∈ C ↔ x ∉ A

-- Given conditions
axiom A_and_B_mutually_exclusive : mutually_exclusive A B
axiom C_is_complementary_to_A : complementary A C
axiom D_is_complementary_to_B : complementary B D

-- Theorem statement
theorem certain_event_C_union_D : ∀ x, x ∈ C ∪ D := by
  sorry

end certain_event_C_union_D_l188_188922


namespace range_of_m_l188_188160

theorem range_of_m {m : ℝ} (h : ∀ x : ℝ, (3 * m - 1) ^ x = (3 * m - 1) ^ x ∧ (3 * m - 1) > 0 ∧ (3 * m - 1) < 1) :
  1 / 3 < m ∧ m < 2 / 3 :=
by
  sorry

end range_of_m_l188_188160


namespace no_int_solutions_a_b_l188_188933

theorem no_int_solutions_a_b :
  ¬ ∃ (a b : ℤ), a^2 + 1998 = b^2 :=
by
  sorry

end no_int_solutions_a_b_l188_188933


namespace find_sample_size_l188_188141

-- Define the frequencies
def frequencies (k : ℕ) : List ℕ := [2 * k, 3 * k, 4 * k, 6 * k, 4 * k, k]

-- Define the sum of the first three frequencies
def sum_first_three_frequencies (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k

-- Define the total number of data points
def total_data_points (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k + 6 * k + 4 * k + k

-- Define the main theorem
theorem find_sample_size (n k : ℕ) (h1 : sum_first_three_frequencies k = 27)
  (h2 : total_data_points k = n) : n = 60 := by
  sorry

end find_sample_size_l188_188141


namespace fraction_sum_l188_188592

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end fraction_sum_l188_188592


namespace f_odd_f_shift_f_in_range_find_f_7_5_l188_188442

def f : ℝ → ℝ := sorry  -- We define the function f (implementation is not needed here)

theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

theorem f_shift (x : ℝ) : f (x + 2) = -f x := sorry

theorem f_in_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = x := sorry

theorem find_f_7_5 : f 7.5 = 0.5 :=
by
  sorry

end f_odd_f_shift_f_in_range_find_f_7_5_l188_188442


namespace identify_letters_l188_188212

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l188_188212


namespace part1_part2_l188_188311

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem part1 : f (Real.pi / 4) = 2 := sorry

theorem part2 : ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 → 
  (2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) > 0) := sorry

end part1_part2_l188_188311


namespace boys_cannot_score_twice_l188_188812

-- Define the total number of points in the tournament
def total_points_in_tournament : ℕ := 15

-- Define the number of boys and girls
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 4

-- Define the points scored by boys and girls
axiom points_by_boys : ℕ
axiom points_by_girls : ℕ

-- The conditions
axiom total_points_condition : points_by_boys + points_by_girls = total_points_in_tournament
axiom boys_twice_girls_condition : points_by_boys = 2 * points_by_girls

-- The statement to prove
theorem boys_cannot_score_twice : False :=
  by {
    -- Note: provide a sketch to illustrate that under the given conditions the statement is false
    sorry
  }

end boys_cannot_score_twice_l188_188812


namespace oranges_in_bowl_l188_188175

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end oranges_in_bowl_l188_188175


namespace max_value_of_x_minus_y_l188_188627

theorem max_value_of_x_minus_y
  (x y : ℝ)
  (h : 2 * (x ^ 2 + y ^ 2 - x * y) = x + y) :
  x - y ≤ 1 / 2 := 
sorry

end max_value_of_x_minus_y_l188_188627


namespace molecular_weight_of_complex_compound_l188_188885

def molecular_weight (n : ℕ) (N_w : ℝ) (o : ℕ) (O_w : ℝ) (h : ℕ) (H_w : ℝ) (p : ℕ) (P_w : ℝ) : ℝ :=
  (n * N_w) + (o * O_w) + (h * H_w) + (p * P_w)

theorem molecular_weight_of_complex_compound :
  molecular_weight 2 14.01 5 16.00 3 1.01 1 30.97 = 142.02 :=
by
  sorry

end molecular_weight_of_complex_compound_l188_188885


namespace xy_relationship_l188_188776

theorem xy_relationship :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := 
by
  sorry

end xy_relationship_l188_188776


namespace tunnel_length_l188_188697

noncomputable def train_length : Real := 2 -- miles
noncomputable def time_to_exit_tunnel : Real := 4 -- minutes
noncomputable def train_speed : Real := 120 -- miles per hour

theorem tunnel_length : ∃ tunnel_length : Real, tunnel_length = 6 :=
  by
  -- We use the conditions given:
  let speed_in_miles_per_minute := train_speed / 60 -- converting speed from miles per hour to miles per minute
  let distance_travelled_by_front_in_4_min := speed_in_miles_per_minute * time_to_exit_tunnel
  let tunnel_length := distance_travelled_by_front_in_4_min - train_length
  have h : tunnel_length = 6 := by sorry
  exact ⟨tunnel_length, h⟩

end tunnel_length_l188_188697


namespace positive_integer_power_of_two_l188_188737

theorem positive_integer_power_of_two (n : ℕ) (hn : 0 < n) :
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) ↔ (∃ k : ℕ, n = 2^k) :=
by
  sorry

end positive_integer_power_of_two_l188_188737


namespace intersection_A_B_l188_188457

open Set

-- Define the sets A and B based on the conditions provided
def A : Set ℤ := { x | x^2 - 2 * x - 8 ≤ 0 }
def B : Set ℤ := { x | log 2 (x : ℝ) > 1 }

-- State the theorem that proves the intersection of A and B equals {3, 4}
theorem intersection_A_B : A ∩ B = { 3, 4 } := by
  sorry

end intersection_A_B_l188_188457


namespace distance_between_Sasha_and_Kolya_l188_188042

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l188_188042


namespace f_eq_32x5_l188_188799

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end f_eq_32x5_l188_188799


namespace g_of_neg3_l188_188504

def g (x : ℝ) : ℝ := x^2 + 2 * x

theorem g_of_neg3 : g (-3) = 3 :=
by
  sorry

end g_of_neg3_l188_188504


namespace find_number_l188_188402

theorem find_number
  (x a b c : ℕ)
  (h1 : x * a = 494)
  (h2 : x * b = 988)
  (h3 : x * c = 1729) :
  x = 247 :=
sorry

end find_number_l188_188402


namespace find_r_values_l188_188869

theorem find_r_values (r : ℝ) (h1 : r ≥ 8) (h2 : r ≤ 20) :
  16 ≤ (r - 4) ^ (3/2) ∧ (r - 4) ^ (3/2) ≤ 128 :=
by {
  sorry
}

end find_r_values_l188_188869


namespace play_children_count_l188_188708

theorem play_children_count (cost_adult_ticket cost_children_ticket total_receipts total_attendance adult_count children_count : ℕ) :
  cost_adult_ticket = 25 →
  cost_children_ticket = 15 →
  total_receipts = 7200 →
  total_attendance = 400 →
  adult_count = 280 →
  25 * adult_count + 15 * children_count = total_receipts →
  adult_count + children_count = total_attendance →
  children_count = 120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end play_children_count_l188_188708


namespace problem_solution_l188_188734

/-- Let f be an even function on ℝ such that f(x + 2) = f(x) and f(x) = x - 2 for x ∈ [3, 4]. 
    Then f(sin 1) < f(cos 1). -/
theorem problem_solution (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = f x)
  (h2 : ∀ x, f (x + 2) = f x)
  (h3 : ∀ x, 3 ≤ x ∧ x ≤ 4 → f x = x - 2) :
  f (Real.sin 1) < f (Real.cos 1) :=
sorry

end problem_solution_l188_188734


namespace different_languages_comb_same_language_comb_total_comb_l188_188371

namespace BookSelection

def numberOfJapaneseBooks : ℕ := 5
def numberOfEnglishBooks : ℕ := 7
def numberOfChineseBooks : ℕ := 10

def differentLanguagesCombinations : ℕ :=
  (numberOfJapaneseBooks * numberOfEnglishBooks) +
  (numberOfChineseBooks * numberOfJapaneseBooks) +
  (numberOfChineseBooks * numberOfEnglishBooks)

def sameLanguageCombinations : ℕ :=
  (numberOfChineseBooks * (numberOfChineseBooks - 1)) / 2 +
  (numberOfEnglishBooks * (numberOfEnglishBooks - 1)) / 2 +
  (numberOfJapaneseBooks * (numberOfJapaneseBooks - 1)) / 2

def totalCombinations : ℕ := (numberOfChineseBooks + numberOfJapaneseBooks + numberOfEnglishBooks) * ((numberOfChineseBooks + numberOfJapaneseBooks + numberOfEnglishBooks) - 1) / 2

theorem different_languages_comb : differentLanguagesCombinations = 155 := by
  sorry

theorem same_language_comb : sameLanguageCombinations = 76 := by
  sorry

theorem total_comb : totalCombinations = 231 := by
  sorry

end BookSelection

end different_languages_comb_same_language_comb_total_comb_l188_188371


namespace gcd_459_357_l188_188516

theorem gcd_459_357 :
  Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l188_188516


namespace part1_part2_l188_188450

noncomputable def f (m x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem part1 (t : ℝ) :
  (1 / 2 < t ∧ t < 1) →
  (∃! t : ℝ, f 1 t = 0) := sorry

theorem part2 :
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) := sorry

end part1_part2_l188_188450


namespace range_m_distinct_roots_l188_188009

theorem range_m_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (4^x₁ - m * 2^(x₁+1) + 2 - m = 0) ∧ (4^x₂ - m * 2^(x₂+1) + 2 - m = 0)) ↔ 1 < m ∧ m < 2 :=
by
  sorry

end range_m_distinct_roots_l188_188009


namespace solve_system_l188_188581

noncomputable def sqrt_cond (x y : ℝ) : Prop :=
  Real.sqrt ((3 * x - 2 * y) / (2 * x)) + Real.sqrt ((2 * x) / (3 * x - 2 * y)) = 2

noncomputable def quad_cond (x y : ℝ) : Prop :=
  x^2 - 18 = 2 * y * (4 * y - 9)

theorem solve_system (x y : ℝ) : sqrt_cond x y ∧ quad_cond x y ↔ (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 1.5) :=
by
  sorry

end solve_system_l188_188581


namespace range_of_a_l188_188777

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x = 1 → a * x^2 + 2 * x + 1 < 0) ↔ a < -3 :=
by
  sorry

end range_of_a_l188_188777


namespace center_digit_is_two_l188_188268

theorem center_digit_is_two :
  ∃ (a b : ℕ), (a^2 < 1000 ∧ b^2 < 1000 ∧ (a^2 ≠ b^2) ∧
  (∀ d, d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] → d ∈ [2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10] → d ∈ [2, 3, 4, 5, 6])) ∧
  (∀ d, (d ∈ [2, 3, 4, 5, 6]) → (d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] ∨ d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10])) ∧
  2 = (a^2 / 10) % 10 ∨ 2 = (b^2 / 10) % 10 :=
sorry -- no proof needed, just the statement

end center_digit_is_two_l188_188268


namespace cuboid_volume_l188_188746

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 14) (h_height : height = 13) : base_area * height = 182 := by
  sorry

end cuboid_volume_l188_188746


namespace factor_check_l188_188948

theorem factor_check :
  ∃ (f : ℕ → ℕ) (x : ℝ), f 1 = (x^2 - 2 * x + 3) ∧ f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 :=
by
  let f : ℕ → ℕ := sorry -- Define a sequence or function for the proof context
  let x : ℝ := sorry -- Define the variable x in our context
  have h₁ : f 1 = (x^2 - 2 * x + 3) := sorry -- Establish the first factor
  have h₂ : f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 := sorry -- Establish the polynomial expression
  exact ⟨f, x, h₁, h₂⟩ -- Use existential quantifier to capture the required form

end factor_check_l188_188948


namespace total_candies_in_third_set_l188_188672

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l188_188672


namespace least_integer_of_sum_in_ratio_l188_188292

theorem least_integer_of_sum_in_ratio (a b c : ℕ) (h1 : a + b + c = 90) (h2 : a * 3 = b * 2) (h3 : a * 5 = c * 2) : a = 18 :=
by
  sorry

end least_integer_of_sum_in_ratio_l188_188292


namespace horner_eval_hex_to_decimal_l188_188092

-- Problem 1: Evaluate the polynomial using Horner's method
theorem horner_eval (x : ℤ) (f : ℤ → ℤ) (v3 : ℤ) :
  (f x = 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12) →
  x = -4 →
  v3 = (((((3 * x + 5) * x + 6) * x + 79) * x - 8) * x + 35) * x + 12 →
  v3 = -57 :=
by
  intros hf hx hv
  sorry

-- Problem 2: Convert hexadecimal base-6 to decimal
theorem hex_to_decimal (hex : ℕ) (dec : ℕ) :
  hex = 210 →
  dec = 0 * 6^0 + 1 * 6^1 + 2 * 6^2 →
  dec = 78 :=
by
  intros hhex hdec
  sorry

end horner_eval_hex_to_decimal_l188_188092


namespace candy_count_in_third_set_l188_188666

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l188_188666


namespace find_a_l188_188760

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem find_a (a : ℝ) (h : binom_coeff 9 3 * (-a)^3 = -84) : a = 1 :=
by
  sorry

end find_a_l188_188760


namespace diagonals_in_convex_polygon_l188_188003

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end diagonals_in_convex_polygon_l188_188003


namespace tangent_line_at_one_l188_188434

noncomputable def equation_of_tangent_line (x : ℝ) : ℝ :=
  3 * (x^(1/3) - 2 * x^(1/2))

theorem tangent_line_at_one :
  let x := 1 in
  let y0 := equation_of_tangent_line x in
  let y' := deriv equation_of_tangent_line x in
  y' = -2 ∧ y0 = -3 ∧ (∀ x, equation_of_tangent_line x - y0 = y' * (x - 1) → equation_of_tangent_line x - y0 = -2 * (x - 1)) :=
by
  sorry

end tangent_line_at_one_l188_188434


namespace jean_jail_time_l188_188188

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l188_188188


namespace Kamal_biology_marks_l188_188022

theorem Kamal_biology_marks 
  (E : ℕ) (M : ℕ) (P : ℕ) (C : ℕ) (A : ℕ) (N : ℕ) (B : ℕ) 
  (hE : E = 66)
  (hM : M = 65)
  (hP : P = 77)
  (hC : C = 62)
  (hA : A = 69)
  (hN : N = 5)
  (h_total : N * A = E + M + P + C + B) 
  : B = 75 :=
by
  sorry

end Kamal_biology_marks_l188_188022


namespace letters_identity_l188_188216

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l188_188216


namespace prime_power_condition_l188_188887

open Nat

theorem prime_power_condition (u v : ℕ) :
  (∃ p n : ℕ, p.Prime ∧ p^n = (u * v^3) / (u^2 + v^2)) ↔ ∃ k : ℕ, k ≥ 1 ∧ u = 2^k ∧ v = 2^k := by {
  sorry
}

end prime_power_condition_l188_188887


namespace percent_formula_l188_188445

theorem percent_formula (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y :=
by
    sorry

end percent_formula_l188_188445


namespace point_B_not_on_curve_C_l188_188577

theorem point_B_not_on_curve_C {a : ℝ} : 
  ¬ ((2 * a) ^ 2 + (4 * a) ^ 2 + 6 * a * (2 * a) - 8 * a * (4 * a) = 0) :=
by 
  sorry

end point_B_not_on_curve_C_l188_188577


namespace range_of_x_minus_2y_l188_188756

theorem range_of_x_minus_2y (x y : ℝ) (h₁ : -1 ≤ x) (h₂ : x < 2) (h₃ : 0 < y) (h₄ : y ≤ 1) :
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 :=
sorry

end range_of_x_minus_2y_l188_188756


namespace unique_x_value_l188_188089

theorem unique_x_value (x : ℝ) (h : x ≠ 0) (h_sqrt : Real.sqrt (5 * x / 7) = x) : x = 5 / 7 :=
by
  sorry

end unique_x_value_l188_188089


namespace domain_of_f_l188_188330

noncomputable def f (x : ℝ) : ℝ := (sqrt (x + 2)) + (1 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (x ≥ -2 ∧ x ≠ 1) ↔ ∃ y : ℝ, y = f x :=
begin
  sorry
end

end domain_of_f_l188_188330


namespace new_person_weight_l188_188832

/-- Conditions: The average weight of 8 persons increases by 6 kg when a new person replaces one of them weighing 45 kg -/
theorem new_person_weight (W : ℝ) (new_person_wt : ℝ) (avg_increase : ℝ) (replaced_person_wt : ℝ) 
  (h1 : avg_increase = 6) (h2 : replaced_person_wt = 45) (weight_increase : 8 * avg_increase = new_person_wt - replaced_person_wt) :
  new_person_wt = 93 :=
by
  sorry

end new_person_weight_l188_188832


namespace range_of_x_minus_2y_l188_188755

theorem range_of_x_minus_2y (x y : ℝ) (h₁ : -1 ≤ x) (h₂ : x < 2) (h₃ : 0 < y) (h₄ : y ≤ 1) :
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 :=
sorry

end range_of_x_minus_2y_l188_188755


namespace total_candies_third_set_l188_188656

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l188_188656


namespace angle_A_is_pi_over_3_l188_188570

theorem angle_A_is_pi_over_3 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C)
  (h2 : a ^ 2 = b ^ 2 + c ^ 2 - bc * (2 * Real.cos A))
  (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A + B + C = π) :
  A = π / 3 :=
by
  sorry

end angle_A_is_pi_over_3_l188_188570


namespace solve_for_y_l188_188441

theorem solve_for_y (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x :=
by sorry

end solve_for_y_l188_188441


namespace total_shaded_area_l188_188328

theorem total_shaded_area (r R : ℝ) (h1 : π * R^2 = 100 * π) (h2 : r = R / 2) : 
    (1/4) * π * R^2 + (1/4) * π * r^2 = 31.25 * π :=
by
  sorry

end total_shaded_area_l188_188328


namespace determine_q_l188_188881

theorem determine_q (p q : ℝ) 
  (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q * x + 12) : 
  q = 7 :=
by
  sorry

end determine_q_l188_188881


namespace binom_20_19_eq_20_l188_188724

theorem binom_20_19_eq_20 : nat.choose 20 19 = 20 := sorry

end binom_20_19_eq_20_l188_188724


namespace train_and_car_combined_time_l188_188414

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l188_188414


namespace find_value_of_a_plus_b_l188_188152

variables (a b : ℝ)

theorem find_value_of_a_plus_b
  (h1 : a^3 - 3 * a^2 + 5 * a = 1)
  (h2 : b^3 - 3 * b^2 + 5 * b = 5) :
  a + b = 2 := 
sorry

end find_value_of_a_plus_b_l188_188152


namespace equivalent_conditions_l188_188803

theorem equivalent_conditions 
  (f : ℕ+ → ℕ+)
  (H1 : ∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m))
  (H2 : ∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :
  (∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m)) ↔ 
  (∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :=
sorry

end equivalent_conditions_l188_188803


namespace simplify_expression_l188_188218

theorem simplify_expression (x : ℝ) : 8 * x + 15 - 3 * x + 27 = 5 * x + 42 := 
by
  sorry

end simplify_expression_l188_188218


namespace not_always_possible_repaint_all_white_l188_188538

-- Define the conditions and the problem
def equilateral_triangle_division (n: ℕ) : Prop := 
  ∀ m, m > 1 → m = n^2

def line_parallel_repaint (triangles : List ℕ) : Prop :=
  -- Definition of how the repaint operation affects the triangle colors
  sorry

theorem not_always_possible_repaint_all_white (n : ℕ) (h: equilateral_triangle_division n) :
  ¬∀ triangles, line_parallel_repaint triangles → (∀ t ∈ triangles, t = 0) := 
sorry

end not_always_possible_repaint_all_white_l188_188538


namespace xiaodong_sister_age_correct_l188_188693

/-- Let's define the conditions as Lean definitions -/
def sister_age := 13
def xiaodong_age := sister_age - 8
def sister_age_in_3_years := sister_age + 3
def xiaodong_age_in_3_years := xiaodong_age + 3

/-- We need to prove that in 3 years, the sister's age will be twice Xiaodong's age -/
theorem xiaodong_sister_age_correct :
  (sister_age_in_3_years = 2 * xiaodong_age_in_3_years) → sister_age = 13 :=
by
  sorry

end xiaodong_sister_age_correct_l188_188693


namespace quadratic_polynomial_l188_188145

noncomputable def p (x : ℝ) : ℝ := (14 * x^2 + 4 * x + 12) / 15

theorem quadratic_polynomial :
  p (-2) = 4 ∧ p 1 = 2 ∧ p 3 = 10 :=
by
  have : p (-2) = (14 * (-2 : ℝ) ^ 2 + 4 * (-2 : ℝ) + 12) / 15 := rfl
  have : p 1 = (14 * (1 : ℝ) ^ 2 + 4 * (1 : ℝ) + 12) / 15 := rfl
  have : p 3 = (14 * (3 : ℝ) ^ 2 + 4 * (3 : ℝ) + 12) / 15 := rfl
  -- You can directly state the equalities or keep track of the computation steps.
  sorry

end quadratic_polynomial_l188_188145


namespace garden_area_increase_l188_188103

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l188_188103


namespace total_animals_l188_188796

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end total_animals_l188_188796


namespace phoenix_hike_length_l188_188033

theorem phoenix_hike_length (a b c d : ℕ)
  (h1 : a + b = 22)
  (h2 : b + c = 26)
  (h3 : c + d = 30)
  (h4 : a + c = 26) :
  a + b + c + d = 52 :=
sorry

end phoenix_hike_length_l188_188033


namespace multiplicative_inverse_l188_188023

def A : ℕ := 123456
def B : ℕ := 171428
def mod_val : ℕ := 1000000
def sum_A_B : ℕ := A + B
def N : ℕ := 863347

theorem multiplicative_inverse : (sum_A_B * N) % mod_val = 1 :=
by
  -- diverting proof with sorry since proof steps aren't the focus
  sorry

end multiplicative_inverse_l188_188023


namespace power_mod_remainder_l188_188245

theorem power_mod_remainder : (3^20) % 7 = 2 :=
by {
  -- condition: 3^6 ≡ 1 (mod 7)
  have h1 : (3^6) % 7 = 1 := by norm_num,
  -- we now use this to show 3^20 ≡ 2 (mod 7)
  calc
    (3^20) % 7 = ((3^6)^3 * 3^2) % 7 : by norm_num
          ... = (1^3 * 3^2) % 7       : by rw [←nat.modeq.modeq_iff_dvd, h1]
          ... =  (3^2) % 7            : by norm_num
          ... = 2                    : by norm_num
}

end power_mod_remainder_l188_188245


namespace speed_of_jogger_l188_188996

noncomputable def jogger_speed_problem (jogger_distance_ahead train_length train_speed_kmh time_to_pass : ℕ) :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := jogger_distance_ahead + train_length
  let relative_speed := total_distance / time_to_pass
  let jogger_speed_ms := train_speed_ms - relative_speed
  let jogger_speed_kmh := jogger_speed_ms * 3600 / 1000
  jogger_speed_kmh

theorem speed_of_jogger :
  jogger_speed_problem 240 210 45 45 = 9 :=
by
  sorry

end speed_of_jogger_l188_188996


namespace value_of_expression_l188_188587

theorem value_of_expression (x : ℝ) (hx : 23 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 5 :=
by
  sorry

end value_of_expression_l188_188587


namespace identity_of_letters_l188_188213

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l188_188213


namespace inequality_transform_l188_188919

theorem inequality_transform {a b : ℝ} (h : a < b) : -2 + 2 * a < -2 + 2 * b :=
sorry

end inequality_transform_l188_188919


namespace auction_sale_l188_188277

theorem auction_sale (TV_initial_price : ℝ) (TV_increase_fraction : ℝ) (Phone_initial_price : ℝ) (Phone_increase_percent : ℝ) :
  TV_initial_price = 500 → 
  TV_increase_fraction = 2 / 5 → 
  Phone_initial_price = 400 →
  Phone_increase_percent = 40 →
  let TV_final_price := TV_initial_price + TV_increase_fraction * TV_initial_price in
  let Phone_final_price := Phone_initial_price + (Phone_increase_percent / 100) * Phone_initial_price in
  TV_final_price + Phone_final_price = 1260 := by
  sorry

end auction_sale_l188_188277


namespace unique_reconstruction_possible_l188_188202

def can_reconstruct_faces (a b c d e f : ℤ) : 
  Prop :=
  let edges := [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]
  in ∀ (sums : list ℤ), sums = edges →
    ∃ (a' b' c' d' e' f' : ℤ), a = a' ∧ b = b' ∧ c = c' ∧ d = d' ∧ e = e' ∧ f = f'

theorem unique_reconstruction_possible :
  ∀ (a b c d e f : ℤ),
    can_reconstruct_faces a b c d e f :=
begin
  sorry
end

end unique_reconstruction_possible_l188_188202


namespace green_marble_probability_l188_188704

theorem green_marble_probability :
  let total_marbles := 21
  let total_green := 5
  let first_draw_prob := (total_green : ℚ) / total_marbles
  let second_draw_prob := (4 : ℚ) / (total_marbles - 1)
  (first_draw_prob * second_draw_prob) = (1 / 21) :=
by
  let total_marbles := 21
  let total_green := 5
  let first_draw_prob := (total_green : ℚ) / total_marbles
  let second_draw_prob := (4 : ℚ) / (total_marbles - 1)
  show first_draw_prob * second_draw_prob = (1 / 21)
  sorry

end green_marble_probability_l188_188704


namespace white_cannot_lose_l188_188552

-- Define a type to represent the game state
structure Game :=
  (state : Type)
  (white_move : state → state)
  (black_move : state → state)
  (initial : state)

-- Define a type to represent the double chess game conditions
structure DoubleChess extends Game :=
  (double_white_move : state → state)
  (double_black_move : state → state)

-- Define the hypothesis based on the conditions
noncomputable def white_has_no_losing_strategy (g : DoubleChess) : Prop :=
  ∃ s, g.double_white_move (g.double_white_move s) = g.initial

theorem white_cannot_lose (g : DoubleChess) :
  white_has_no_losing_strategy g :=
sorry

end white_cannot_lose_l188_188552


namespace maximum_triangle_area_le_8_l188_188085

def lengths : List ℝ := [2, 3, 4, 5, 6]

-- Function to determine if three lengths can form a valid triangle
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a 

-- Heron's formula to compute the area of a triangle given its sides
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement to prove that the maximum possible area with given stick lengths is less than or equal to 8 cm²
theorem maximum_triangle_area_le_8 :
  ∃ (a b c : ℝ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ 
  is_valid_triangle a b c ∧ heron_area a b c ≤ 8 :=
sorry

end maximum_triangle_area_le_8_l188_188085


namespace find_N_l188_188967

variable (a b c N : ℕ)

theorem find_N (h1 : a + b + c = 90) (h2 : a - 7 = N) (h3 : b + 7 = N) (h4 : 5 * c = N) : N = 41 := 
by
  sorry

end find_N_l188_188967


namespace prob_high_quality_correct_l188_188699

noncomputable def prob_high_quality_seeds :=
  let p_first := 0.955
  let p_second := 0.02
  let p_third := 0.015
  let p_fourth := 0.01
  let p_hq_first := 0.5
  let p_hq_second := 0.15
  let p_hq_third := 0.1
  let p_hq_fourth := 0.05
  let p_hq := p_first * p_hq_first + p_second * p_hq_second + p_third * p_hq_third + p_fourth * p_hq_fourth
  p_hq

theorem prob_high_quality_correct : prob_high_quality_seeds = 0.4825 :=
  by sorry

end prob_high_quality_correct_l188_188699


namespace num_four_digit_int_with_4_or_5_correct_l188_188164

def num_four_digit_int_with_4_or_5 : ℕ :=
  5416

theorem num_four_digit_int_with_4_or_5_correct (A B : ℕ) (hA : A = 9000) (hB : B = 3584) :
  num_four_digit_int_with_4_or_5 = A - B :=
by
  rw [hA, hB]
  sorry

end num_four_digit_int_with_4_or_5_correct_l188_188164


namespace chromium_percentage_bounds_l188_188486

noncomputable def new_alloy_chromium_bounds : Prop :=
  ∃ (x y z k : ℝ), 
    (x + y + z = 1) ∧ 
    (0.9 * x + 0.3 * z = 0.45) ∧ 
    (0.4 * x + 0.1 * y + 0.5 * z = k) ∧ 
    (0.25 ≤ k ∧ k ≤ 0.4)

theorem chromium_percentage_bounds : new_alloy_chromium_bounds :=
by
  sorry

end chromium_percentage_bounds_l188_188486


namespace intersection_sum_zero_l188_188609

-- Definitions from conditions:
def lineA (x : ℝ) : ℝ := -x
def lineB (x : ℝ) : ℝ := 5 * x - 10

-- Declaration of the theorem:
theorem intersection_sum_zero : ∃ a b : ℝ, lineA a = b ∧ lineB a = b ∧ a + b = 0 := sorry

end intersection_sum_zero_l188_188609


namespace window_area_properties_l188_188995

theorem window_area_properties
  (AB : ℝ) (AD : ℝ) (ratio : ℝ)
  (h1 : ratio = 3 / 1)
  (h2 : AB = 40)
  (h3 : AD = 3 * AB) :
  (AD * AB / (π * (AB / 2) ^ 2) = 12 / π) ∧
  (AD * AB + π * (AB / 2) ^ 2 = 4800 + 400 * π) :=
by
  -- Proof will go here
  sorry

end window_area_properties_l188_188995


namespace quadratic_completion_l188_188061

theorem quadratic_completion (x : ℝ) :
  2 * x^2 + 3 * x + 1 = 0 ↔ 2 * (x + 3 / 4)^2 - 1 / 8 = 0 :=
by
  sorry

end quadratic_completion_l188_188061


namespace f_is_32x5_l188_188802

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end f_is_32x5_l188_188802


namespace maria_candy_remaining_l188_188349

theorem maria_candy_remaining :
  let c := 520.75
  let e := c / 2
  let g := 234.56
  let r := e - g
  r = 25.815 := by
  sorry

end maria_candy_remaining_l188_188349


namespace combined_time_l188_188411

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l188_188411


namespace marcia_average_cost_l188_188030

theorem marcia_average_cost :
  let price_apples := 2
  let price_bananas := 1
  let price_oranges := 3
  let count_apples := 12
  let count_bananas := 4
  let count_oranges := 4
  let offer_apples_free := count_apples / 10 * 2
  let offer_oranges_free := count_oranges / 3
  let total_apples := count_apples + offer_apples_free
  let total_oranges := count_oranges + offer_oranges_free
  let total_fruits := total_apples + count_bananas + count_oranges
  let cost_apples := price_apples * (count_apples - offer_apples_free)
  let cost_bananas := price_bananas * count_bananas
  let cost_oranges := price_oranges * (count_oranges - offer_oranges_free)
  let total_cost := cost_apples + cost_bananas + cost_oranges
  let average_cost := total_cost / total_fruits
  average_cost = 1.85 :=
  sorry

end marcia_average_cost_l188_188030


namespace units_digit_of_n_l188_188550

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 11^4) (h2 : m % 10 = 9) : n % 10 = 9 := 
sorry

end units_digit_of_n_l188_188550


namespace bowling_ball_weight_l188_188554

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end bowling_ball_weight_l188_188554


namespace two_numbers_are_opposites_l188_188395

theorem two_numbers_are_opposites (x y z : ℝ) (h : (1 / x) + (1 / y) + (1 / z) = 1 / (x + y + z)) :
  (x + y = 0) ∨ (x + z = 0) ∨ (y + z = 0) :=
by
  sorry

end two_numbers_are_opposites_l188_188395


namespace cubic_diff_l188_188154

theorem cubic_diff (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : a^3 - b^3 = 208 :=
by
  sorry

end cubic_diff_l188_188154


namespace greatest_volume_of_pyramid_l188_188383

noncomputable def max_pyramid_volume (AB AC : ℝ) (sin_BAC : ℝ) (angle_limit : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_BAC = 4/5 ∧ angle_limit = π / 3 then 5 * Real.sqrt 39 / 2 else 0

theorem greatest_volume_of_pyramid :
  let AB := 3
  let AC := 5
  let sin_BAC := 4/5
  let angle_limit := π / 3
  max_pyramid_volume AB AC sin_BAC angle_limit = 5 * Real.sqrt 39 / 2 := by 
  sorry

end greatest_volume_of_pyramid_l188_188383


namespace geometric_sequence_n_value_l188_188926

theorem geometric_sequence_n_value
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 * a 2 * a 3 = 4)
  (h2 : a 4 * a 5 * a 6 = 12)
  (h3 : a (n-1) * a n * a (n+1) = 324)
  (h_geometric : ∃ r > 0, ∀ i, a (i+1) = a i * r) :
  n = 14 :=
sorry

end geometric_sequence_n_value_l188_188926


namespace garden_area_difference_l188_188108

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l188_188108


namespace value_of_a_l188_188300

theorem value_of_a (x : ℝ) (n : ℕ) (h : x > 0) (h_n : n > 0) :
  (∀ k : ℕ, 1 ≤ k → k ≤ n → x + k ≥ k + 1) → a = n^n :=
by
  sorry

end value_of_a_l188_188300


namespace race_distance_between_Sasha_and_Kolya_l188_188052

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l188_188052


namespace simplify_and_evaluate_l188_188496

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  (1 / (x - 1) - 2 / (x ^ 2 - 1)) = -1 := by
  sorry

end simplify_and_evaluate_l188_188496


namespace hyperbola_condition_l188_188460

theorem hyperbola_condition (k : ℝ) : (k > 1) -> ( ∀ x y : ℝ, (k - 1) * (k + 1) > 0 ↔ ( ∃ x y : ℝ, (k > 1) ∧ ((x * x) / (k - 1) - (y * y) / (k + 1)) = 1)) :=
sorry

end hyperbola_condition_l188_188460


namespace find_y_l188_188499

theorem find_y (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (hx : x = -4) : y = 41 / 2 :=
by
  sorry

end find_y_l188_188499


namespace verify_a_eq_x0_verify_p_squared_ge_4x0q_l188_188394

theorem verify_a_eq_x0 (p q x0 a b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + a * x + b)) : 
  a = x0 :=
by
  sorry

theorem verify_p_squared_ge_4x0q (p q x0 b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + x0 * x + b)) : 
  p^2 ≥ 4 * x0 * q :=
by
  sorry

end verify_a_eq_x0_verify_p_squared_ge_4x0q_l188_188394


namespace profit_percentage_correct_l188_188999

noncomputable def CP : ℝ := 460
noncomputable def SP : ℝ := 542.8
noncomputable def profit : ℝ := SP - CP
noncomputable def profit_percentage : ℝ := (profit / CP) * 100

theorem profit_percentage_correct :
  profit_percentage = 18 := by
  sorry

end profit_percentage_correct_l188_188999


namespace cubes_with_4_neighbors_l188_188561

theorem cubes_with_4_neighbors (a b c : ℕ) (h₁ : 3 < a) (h₂ : 3 < b) (h₃ : 3 < c)
  (h₄ : (a - 2) * (b - 2) * (c - 2) = 429) : 
  4 * ((a - 2) + (b - 2) + (c - 2)) = 108 := by
  sorry

end cubes_with_4_neighbors_l188_188561


namespace Lisa_types_correctly_l188_188065

-- Given conditions
def Rudy_wpm : ℕ := 64
def Joyce_wpm : ℕ := 76
def Gladys_wpm : ℕ := 91
def Mike_wpm : ℕ := 89
def avg_wpm : ℕ := 80
def num_employees : ℕ := 5

-- Define the hypothesis about Lisa's typing speaking
def Lisa_wpm : ℕ := (num_employees * avg_wpm) - Rudy_wpm - Joyce_wpm - Gladys_wpm - Mike_wpm

-- The statement to prove
theorem Lisa_types_correctly :
  Lisa_wpm = 140 := by
  sorry

end Lisa_types_correctly_l188_188065


namespace third_year_increment_l188_188824

-- Define the conditions
def total_payments : ℕ := 96
def first_year_cost : ℕ := 20
def second_year_cost : ℕ := first_year_cost + 2
def third_year_cost (x : ℕ) : ℕ := second_year_cost + x
def fourth_year_cost (x : ℕ) : ℕ := third_year_cost x + 4

-- The main proof statement
theorem third_year_increment (x : ℕ) 
  (H : first_year_cost + second_year_cost + third_year_cost x + fourth_year_cost x = total_payments) :
  x = 2 :=
sorry

end third_year_increment_l188_188824


namespace net_income_correct_l188_188376

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end net_income_correct_l188_188376


namespace logical_equivalence_l188_188549

variable (R S T : Prop)

theorem logical_equivalence :
  (R → ¬S ∧ ¬T) ↔ ((S ∨ T) → ¬R) :=
by
  sorry

end logical_equivalence_l188_188549


namespace jean_jail_time_l188_188184

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l188_188184


namespace steve_marbles_l188_188950

theorem steve_marbles {S : ℤ} (sam_initial : ℤ) (sally_initial : ℤ) :
  (sam_initial = 2 * S) →
  (sally_initial = 2 * S - 5) →
  (sam_initial - 3 - 3 = 8) →
  (S + 3 = 10) :=
begin
  intros h1 h2 h3,
  sorry
end

end steve_marbles_l188_188950


namespace number_of_divisors_of_10_factorial_greater_than_9_factorial_l188_188769

theorem number_of_divisors_of_10_factorial_greater_than_9_factorial :
  let divisors := {d : ℕ | d ∣ nat.factorial 10} in
  let bigger_divisors := {d : ℕ | d ∈ divisors ∧ d > nat.factorial 9} in
  set.card bigger_divisors = 9 := 
by {
  -- Let set.card be the cardinality function for sets
  sorry
}

end number_of_divisors_of_10_factorial_greater_than_9_factorial_l188_188769


namespace time_to_cover_escalator_l188_188416

-- Define the given conditions
def escalator_speed : ℝ := 20 -- feet per second
def escalator_length : ℝ := 360 -- feet
def delay_time : ℝ := 5 -- seconds
def person_speed : ℝ := 4 -- feet per second

-- Define the statement to be proven
theorem time_to_cover_escalator : (delay_time + (escalator_length - (escalator_speed * delay_time)) / (person_speed + escalator_speed)) = 15.83 := 
by {
  sorry
}

end time_to_cover_escalator_l188_188416


namespace total_rooms_booked_l188_188713

variable (S D : ℕ)

theorem total_rooms_booked (h1 : 35 * S + 60 * D = 14000) (h2 : D = 196) : S + D = 260 :=
by
  sorry

end total_rooms_booked_l188_188713


namespace fraction_simplify_l188_188034

theorem fraction_simplify (x : ℝ) (hx : x ≠ 1) (hx_ne_1 : x ≠ -1) :
  (x^2 - 1) / (x^2 - 2 * x + 1) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_simplify_l188_188034


namespace log_comparison_l188_188690

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := 
sorry

end log_comparison_l188_188690


namespace specific_gravity_is_0_6734_l188_188534

noncomputable def sphere_radius : ℝ := 8 -- Since diameter is 16 cm

noncomputable def dry_surface_area : ℝ := 307.2 -- cm²

noncomputable def buoyant_force_balance (x : ℝ) : Prop :=
  let volume_sphere := (4 / 3) * Real.pi * (sphere_radius ^ 3) in
  let height_dry_cap := dry_surface_area / (2 * Real.pi * sphere_radius) in
  let height_submerged_cap := 2 * sphere_radius - height_dry_cap in
  let volume_submerged_cap := (1 / 3) * Real.pi * (height_submerged_cap ^ 2) * (3 * sphere_radius - height_submerged_cap) in
  x = volume_submerged_cap / volume_sphere

theorem specific_gravity_is_0_6734 : ∃ x, buoyant_force_balance x ∧ x = 0.6734 :=
by 
  -- The proof would go here
  sorry

end specific_gravity_is_0_6734_l188_188534


namespace michael_total_payment_correct_l188_188945

variable (original_suit_price : ℕ := 430)
variable (suit_discount : ℕ := 100)
variable (suit_tax_rate : ℚ := 0.05)

variable (original_shoes_price : ℕ := 190)
variable (shoes_discount : ℕ := 30)
variable (shoes_tax_rate : ℚ := 0.07)

variable (original_dress_shirt_price : ℕ := 80)
variable (original_tie_price : ℕ := 50)
variable (combined_discount_rate : ℚ := 0.20)
variable (dress_shirt_tax_rate : ℚ := 0.06)
variable (tie_tax_rate : ℚ := 0.04)

def calculate_total_amount_paid : ℚ :=
  let discounted_suit_price := original_suit_price - suit_discount
  let suit_tax := discounted_suit_price * suit_tax_rate
  let discounted_shoes_price := original_shoes_price - shoes_discount
  let shoes_tax := discounted_shoes_price * shoes_tax_rate
  let combined_original_price := original_dress_shirt_price + original_tie_price
  let combined_discount := combined_discount_rate * combined_original_price
  let discounted_combined_price := combined_original_price - combined_discount
  let discounted_dress_shirt_price := (original_dress_shirt_price / combined_original_price) * discounted_combined_price
  let discounted_tie_price := (original_tie_price / combined_original_price) * discounted_combined_price
  let dress_shirt_tax := discounted_dress_shirt_price * dress_shirt_tax_rate
  let tie_tax := discounted_tie_price * tie_tax_rate
  discounted_suit_price + suit_tax + discounted_shoes_price + shoes_tax + discounted_dress_shirt_price + dress_shirt_tax + discounted_tie_price + tie_tax

theorem michael_total_payment_correct : calculate_total_amount_paid = 627.14 := by
  sorry

end michael_total_payment_correct_l188_188945


namespace molecular_weight_of_compound_l188_188976

def atomic_weight_Al : ℕ := 27
def atomic_weight_I : ℕ := 127
def atomic_weight_O : ℕ := 16

def num_Al : ℕ := 1
def num_I : ℕ := 3
def num_O : ℕ := 2

def molecular_weight (n_Al n_I n_O w_Al w_I w_O : ℕ) : ℕ :=
  (n_Al * w_Al) + (n_I * w_I) + (n_O * w_O)

theorem molecular_weight_of_compound :
  molecular_weight num_Al num_I num_O atomic_weight_Al atomic_weight_I atomic_weight_O = 440 := 
sorry

end molecular_weight_of_compound_l188_188976


namespace value_of_x_l188_188013

theorem value_of_x (x : ℝ) (h : x = 88 + 0.3 * 88) : x = 114.4 :=
by
  sorry

end value_of_x_l188_188013


namespace continuous_at_x0_l188_188036

variable (x : ℝ) (ε : ℝ)

def f (x : ℝ) : ℝ := -3 * x^2 - 5

def x0 : ℝ := 2

theorem continuous_at_x0 : 
  (0 < ε) → 
  ∃ δ > 0, ∀ x, abs (x - x0) < δ → abs (f x - f x0) < ε := by
sory

end continuous_at_x0_l188_188036


namespace jake_sausages_cost_l188_188934

theorem jake_sausages_cost :
  let package_weight := 2
  let num_packages := 3
  let cost_per_pound := 4
  let total_weight := package_weight * num_packages
  let total_cost := total_weight * cost_per_pound
  total_cost = 24 := by
  sorry

end jake_sausages_cost_l188_188934


namespace find_m_minus_n_l188_188899

-- Define line equations, parallelism, and perpendicularity
def line1 (x y : ℝ) : Prop := 3 * x - 6 * y + 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := x - m * y + 2 = 0
def line3 (x y : ℝ) (n : ℝ) : Prop := n * x + y + 3 = 0

def parallel (m1 m2 : ℝ) : Prop := m1 = m2
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_m_minus_n (m n : ℝ) (h_parallel : parallel (1/2) (1/m)) (h_perpendicular: perpendicular (1/2) (-1/n)) : m - n = 0 :=
sorry

end find_m_minus_n_l188_188899


namespace polygon_diagonals_eq_sum_sides_and_right_angles_l188_188978

-- Define the number of sides of the polygon
variables (n : ℕ)

-- Definition of the number of diagonals in a convex n-sided polygon
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Definition of the sum of interior angles of an n-sided polygon
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Definition of equivalent right angles for interior angles
def num_right_angles (n : ℕ) : ℕ := 2 * (n - 2)

-- The proof statement: prove that the equation holds for n
theorem polygon_diagonals_eq_sum_sides_and_right_angles (h : 3 ≤ n) :
  num_diagonals n = n + num_right_angles n :=
sorry

end polygon_diagonals_eq_sum_sides_and_right_angles_l188_188978


namespace original_salary_l188_188844

theorem original_salary (S : ℝ) (h : (1.12) * (0.93) * (1.09) * (0.94) * S = 1212) : 
  S = 1212 / ((1.12) * (0.93) * (1.09) * (0.94)) :=
by
  sorry

end original_salary_l188_188844


namespace percent_decrease_second_year_l188_188698

theorem percent_decrease_second_year
  (V_0 V_1 V_2 : ℝ)
  (p_2 : ℝ)
  (h1 : V_1 = V_0 * 0.7)
  (h2 : V_2 = V_1 * (1 - p_2 / 100))
  (h3 : V_2 = V_0 * 0.63) :
  p_2 = 10 :=
sorry

end percent_decrease_second_year_l188_188698


namespace solve_equation_l188_188815

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l188_188815


namespace find_b_l188_188642

theorem find_b (b : ℝ) (h1 : 0 < b) (h2 : b < 6)
  (h_ratio : ∃ (QRS QOP : ℝ), QRS / QOP = 4 / 25) : b = 6 :=
sorry

end find_b_l188_188642


namespace ordered_pairs_1806_l188_188842

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l188_188842


namespace rose_joined_after_six_months_l188_188478

noncomputable def profit_shares (m : ℕ) : ℕ :=
  12000 * (12 - m) - 9000 * 8

theorem rose_joined_after_six_months :
  ∃ (m : ℕ), profit_shares m = 370 :=
by
  use 6
  unfold profit_shares
  norm_num
  sorry

end rose_joined_after_six_months_l188_188478


namespace find_x0_l188_188571

noncomputable def f (x : ℝ) : ℝ := 13 - 8 * x + x^2

theorem find_x0 :
  (∃ x0 : ℝ, deriv f x0 = 4) → ∃ x0 : ℝ, x0 = 6 :=
by
  sorry

end find_x0_l188_188571


namespace unique_solution_l188_188892

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem unique_solution (x y : ℕ) :
  is_prime x →
  is_odd y →
  x^2 + y = 2007 →
  (x = 2 ∧ y = 2003) :=
by
  sorry

end unique_solution_l188_188892


namespace infinitely_many_not_representable_l188_188633

def can_be_represented_as_p_n_2k (c : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ c = p + n^(2 * k)

theorem infinitely_many_not_representable :
  ∃ᶠ m in at_top, ¬ can_be_represented_as_p_n_2k (2^m + 1) := 
sorry

end infinitely_many_not_representable_l188_188633


namespace first_term_geometric_series_l188_188271

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end first_term_geometric_series_l188_188271


namespace aisha_additional_miles_l188_188872

theorem aisha_additional_miles
  (D : ℕ) (d : ℕ) (v1 : ℕ) (v2 : ℕ) (v_avg : ℕ)
  (h1 : D = 18) (h2 : v1 = 36) (h3 : v2 = 60) (h4 : v_avg = 48)
  (h5 : d = 30) :
  (D + d) / ((D / v1) + (d / v2)) = v_avg :=
  sorry

end aisha_additional_miles_l188_188872


namespace train_speed_l188_188264

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : length = 55) 
    (h2 : time = 5.5) 
    (h3 : speed = (length / time) * (3600 / 1000)) : 
    speed = 36 :=
sorry

end train_speed_l188_188264


namespace one_add_i_cubed_eq_one_sub_i_l188_188459

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end one_add_i_cubed_eq_one_sub_i_l188_188459


namespace no_solution_k_eq_7_l188_188425

-- Define the condition that x should not be equal to 4 and 8
def condition (x : ℝ) : Prop := x ≠ 4 ∧ x ≠ 8

-- Define the equation
def equation (x k : ℝ) : Prop := (x - 3) / (x - 4) = (x - k) / (x - 8)

-- Prove that for the equation to have no solution, k must be 7
theorem no_solution_k_eq_7 : (∀ x, condition x → ¬ equation x 7) ↔ (∃ k, k = 7) :=
by
  sorry

end no_solution_k_eq_7_l188_188425


namespace harvest_rate_l188_188001

def days := 3
def total_sacks := 24
def sacks_per_day := total_sacks / days

theorem harvest_rate :
  sacks_per_day = 8 :=
by
  sorry

end harvest_rate_l188_188001


namespace sale_price_after_discounts_l188_188965

def calculate_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

theorem sale_price_after_discounts :
  calculate_sale_price 500 [0.10, 0.15, 0.20, 0.25, 0.30] = 160.65 :=
by
  sorry

end sale_price_after_discounts_l188_188965


namespace earnings_ratio_l188_188475

-- Definitions for conditions
def jerusha_earnings : ℕ := 68
def total_earnings : ℕ := 85
def lottie_earnings : ℕ := total_earnings - jerusha_earnings

-- Prove that the ratio of Jerusha's earnings to Lottie's earnings is 4:1
theorem earnings_ratio : 
  ∃ (k : ℕ), jerusha_earnings = k * lottie_earnings ∧ (jerusha_earnings + lottie_earnings = total_earnings) ∧ (jerusha_earnings = 68) ∧ (total_earnings = 85) →
  68 / (total_earnings - 68) = 4 := 
by
  sorry

end earnings_ratio_l188_188475


namespace largest_angle_is_120_l188_188333

variable (d e f : ℝ)
variable (h1 : d + 3 * e + 3 * f = d^2)
variable (h2 : d + 3 * e - 3 * f = -4)

theorem largest_angle_is_120 (h1 : d + 3 * e + 3 * f = d^2) (h2 : d + 3 * e - 3 * f = -4) : 
  ∃ (F : ℝ), F = 120 :=
by
  sorry

end largest_angle_is_120_l188_188333


namespace average_test_score_before_dropping_l188_188476

theorem average_test_score_before_dropping (A B C : ℝ) :
  (A + B + C) / 3 = 40 → (A + B + C + 20) / 4 = 35 :=
by
  intros h
  sorry

end average_test_score_before_dropping_l188_188476


namespace find_unknown_number_l188_188889

theorem find_unknown_number (x : ℝ) (h : (45 + 23 / x) * x = 4028) : x = 89 :=
sorry

end find_unknown_number_l188_188889


namespace line_eq_slope_form_l188_188491

theorem line_eq_slope_form (a b c : ℝ) (h : b ≠ 0) :
    ∃ k l : ℝ, ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (y = k * x + l) := 
sorry

end line_eq_slope_form_l188_188491


namespace arc_length_of_curve_l188_188393

open Real

noncomputable def curve_function (x : ℝ) : ℝ := (x^2 / 4) - (log x / 2)

theorem arc_length_of_curve :
  ∫ x in 1..2, sqrt (1 + ((deriv curve_function x)^2)) = 3 / 4 + 1 / 2 * log 2 :=
by
  sorry

end arc_length_of_curve_l188_188393


namespace garden_area_increase_l188_188099

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l188_188099


namespace jean_jail_time_l188_188183

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l188_188183


namespace non_degenerate_ellipse_l188_188548

theorem non_degenerate_ellipse (k : ℝ) : (∃ a, a = -21) ↔ (k > -21) := by
  sorry

end non_degenerate_ellipse_l188_188548


namespace production_difference_correct_l188_188140

variable (w t M T : ℕ)

-- Condition: w = 2t
def condition_w := w = 2 * t

-- Widgets produced on Monday
def widgets_monday := M = w * t

-- Widgets produced on Tuesday
def widgets_tuesday := T = (w + 5) * (t - 3)

-- Difference in production
def production_difference := M - T = t + 15

theorem production_difference_correct
  (h1 : condition_w w t)
  (h2 : widgets_monday M w t)
  (h3 : widgets_tuesday T w t) :
  production_difference M T t :=
sorry

end production_difference_correct_l188_188140


namespace platform_length_l188_188866

theorem platform_length (train_length : ℕ) (time_post : ℕ) (time_platform : ℕ) (speed : ℕ)
    (h1 : train_length = 150)
    (h2 : time_post = 15)
    (h3 : time_platform = 25)
    (h4 : speed = train_length / time_post)
    : (train_length + 100) / time_platform = speed :=
by
  sorry

end platform_length_l188_188866


namespace four_digit_integers_with_4_or_5_l188_188165

theorem four_digit_integers_with_4_or_5 : 
  (finset.range 10000).filter (λ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ d, d ∈ [4, 5] ∧ d ∈ n.digits 10)).card = 5416 :=
sorry

end four_digit_integers_with_4_or_5_l188_188165


namespace find_unknown_value_l188_188064

theorem find_unknown_value (x : ℝ) (h : (3 + 5 + 6 + 8 + x) / 5 = 7) : x = 13 :=
by
  sorry

end find_unknown_value_l188_188064


namespace countFibSequences_l188_188518

-- Define what it means for a sequence to be Fibonacci-type
def isFibType (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

-- Define a Fibonacci-type sequence condition with given constraints
def fibSeqCondition (a : ℤ → ℤ) (N : ℤ) : Prop :=
  isFibType a ∧ ∃ n : ℤ, 0 < a n ∧ a n ≤ N ∧ 0 < a (n + 1) ∧ a (n + 1) ≤ N

-- Main theorem
theorem countFibSequences (N : ℤ) :
  ∃ count : ℤ,
    (N % 2 = 0 → count = (N / 2) * (N / 2 + 1)) ∧
    (N % 2 = 1 → count = ((N + 1) / 2) ^ 2) ∧
    (∀ a : ℤ → ℤ, fibSeqCondition a N → (∃ n : ℤ, a n = count)) :=
by
  sorry

end countFibSequences_l188_188518


namespace distance_between_Sasha_and_Koyla_is_19m_l188_188049

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l188_188049


namespace average_visitors_per_day_l188_188119

theorem average_visitors_per_day:
  (∃ (Sundays OtherDays: ℕ) (visitors_per_sunday visitors_per_other_day: ℕ),
    Sundays = 4 ∧
    OtherDays = 26 ∧
    visitors_per_sunday = 600 ∧
    visitors_per_other_day = 240 ∧
    (Sundays + OtherDays = 30) ∧
    (Sundays * visitors_per_sunday + OtherDays * visitors_per_other_day) / 30 = 288) :=
sorry

end average_visitors_per_day_l188_188119


namespace james_new_fuel_cost_l188_188181

def original_cost : ℕ := 200
def price_increase_rate : ℕ := 20
def extra_tank_factor : ℕ := 2

theorem james_new_fuel_cost :
  let new_price := original_cost + (price_increase_rate * original_cost / 100)
  let total_cost := extra_tank_factor * new_price
  total_cost = 480 :=
by
  sorry

end james_new_fuel_cost_l188_188181


namespace count_digit_2_in_range_1_to_1000_l188_188745

theorem count_digit_2_in_range_1_to_1000 :
  let count_digit_occur (digit : ℕ) (range_end : ℕ) : ℕ :=
    (range_end + 1).digits 10
    |>.count digit
  count_digit_occur 2 1000 = 300 :=
by
  sorry

end count_digit_2_in_range_1_to_1000_l188_188745


namespace x_plus_2y_equals_5_l188_188921

theorem x_plus_2y_equals_5 (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : (x + y) / 3 = 1.222222222222222) : x + 2 * y = 5 := 
by sorry

end x_plus_2y_equals_5_l188_188921


namespace field_length_l188_188367

theorem field_length (w l : ℕ) (Pond_Area : ℕ) (Pond_Field_Ratio : ℚ) (Field_Length_Ratio : ℕ) 
  (h1 : Length = 2 * Width)
  (h2 : Pond_Area = 8 * 8)
  (h3 : Pond_Field_Ratio = 1 / 50)
  (h4 : Pond_Area = Pond_Field_Ratio * Field_Area)
  : l = 80 := 
by
  -- begin solution
  sorry

end field_length_l188_188367


namespace arithmetic_sequence_a10_l188_188301

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : a 1 + a 2 + a 3 = 15) 
  (h_geo : (a 1 + 2) * (a 3 + 13) = (a 2 + 5) * (a 2 + 5))  
  : a 10 = 21 := sorry

end arithmetic_sequence_a10_l188_188301


namespace collective_apples_l188_188358

theorem collective_apples :
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8 := by
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  show (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8
  sorry

end collective_apples_l188_188358


namespace find_resistance_x_l188_188324

theorem find_resistance_x (y r x : ℝ) (h₁ : y = 5) (h₂ : r = 1.875) (h₃ : 1/r = 1/x + 1/y) : x = 3 :=
by
  sorry

end find_resistance_x_l188_188324


namespace right_triangle_with_a_as_hypotenuse_l188_188595

theorem right_triangle_with_a_as_hypotenuse
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a = (b^2 + c^2 - a^2) / (2 * b * c))
  (h2 : b = (a^2 + c^2 - b^2) / (2 * a * c))
  (h3 : c = (a^2 + b^2 - c^2) / (2 * a * b))
  (h4 : a * ((b^2 + c^2 - a^2) / (2 * b * c)) + b * ((a^2 + c^2 - b^2) / (2 * a * c)) = c * ((a^2 + b^2 - c^2) / (2 * a * b))) :
  a^2 = b^2 + c^2 :=
by
  sorry

end right_triangle_with_a_as_hypotenuse_l188_188595


namespace coin_flip_probability_l188_188031

theorem coin_flip_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)
  (h_win : ∑' n, (1 - p) ^ n * p ^ (n + 1) = 1 / 2) :
  p = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end coin_flip_probability_l188_188031


namespace solve_quadratic_l188_188956

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 6 * x^2 + 9 * x - 24 = 0) : x = 4 / 3 :=
by
  sorry

end solve_quadratic_l188_188956


namespace area_identity_tg_cos_l188_188217

variable (a b c α β γ : Real)
variable (s t : Real) (area_of_triangle : Real)

-- Assume t is the area of the triangle and s is the semiperimeter
axiom area_of_triangle_eq_heron :
  t = Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Assume trigonometric identities for tangents and cosines of half-angles
axiom tg_half_angle_α : Real.tan (α / 2) = Real.sqrt ((s - b) * (s - c) / (s * (s - a)))
axiom tg_half_angle_β : Real.tan (β / 2) = Real.sqrt ((s - c) * (s - a) / (s * (s - b)))
axiom tg_half_angle_γ : Real.tan (γ / 2) = Real.sqrt ((s - a) * (s - b) / (s * (s - c)))

axiom cos_half_angle_α : Real.cos (α / 2) = Real.sqrt (s * (s - a) / (b * c))
axiom cos_half_angle_β : Real.cos (β / 2) = Real.sqrt (s * (s - b) / (c * a))
axiom cos_half_angle_γ : Real.cos (γ / 2) = Real.sqrt (s * (s - c) / (a * b))

theorem area_identity_tg_cos :
  t = s^2 * Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) ∧
  t = (a * b * c / s) * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) :=
by
  sorry

end area_identity_tg_cos_l188_188217


namespace range_of_a_l188_188766

noncomputable def A (a : ℝ) := {x : ℝ | a < x ∧ x < 2 * a + 1}
def B := {x : ℝ | abs (x - 1) > 2}

theorem range_of_a (a : ℝ) (h : A a ⊆ B) : a ≤ -1 ∨ a ≥ 3 := by
  sorry

end range_of_a_l188_188766


namespace geometric_sequence_value_l188_188928

variable {a_n : ℕ → ℝ}

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given: a_1 a_2 a_3 = -8
variable (a1 a2 a3 : ℝ) (h_seq : is_geometric_sequence a_n)
variable (h_cond : a1 * a2 * a3 = -8)

-- Prove: a2 = -2
theorem geometric_sequence_value : a2 = -2 :=
by
  -- Proof will be provided later
  sorry

end geometric_sequence_value_l188_188928


namespace intersection_point_exists_l188_188400

theorem intersection_point_exists :
  ∃ t u x y : ℚ,
    (x = 2 + 3 * t) ∧ (y = 3 - 4 * t) ∧
    (x = 4 + 5 * u) ∧ (y = -6 + u) ∧
    (x = 175 / 23) ∧ (y = 19 / 23) :=
by
  sorry

end intersection_point_exists_l188_188400


namespace vector_n_value_l188_188159

theorem vector_n_value {n : ℤ} (hAB : (2, 4) = (2, 4)) (hBC : (-2, n) = (-2, n)) (hAC : (0, 2) = (2 + -2, 4 + n)) : n = -2 :=
by
  sorry

end vector_n_value_l188_188159


namespace garden_area_increase_l188_188113

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l188_188113


namespace expected_scurried_home_mn_sum_l188_188234

theorem expected_scurried_home_mn_sum : 
  let expected_fraction : ℚ := (1/2 + 2/3 + 3/4 + 4/5 + 5/6 + 6/7 + 7/8)
  let m : ℕ := 37
  let n : ℕ := 7
  m + n = 44 := by
  sorry

end expected_scurried_home_mn_sum_l188_188234


namespace determine_c_l188_188884

noncomputable def fib (n : ℕ) : ℕ :=
match n with
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem determine_c (c d : ℤ) (h1 : ∃ s : ℂ, s^2 - s - 1 = 0 ∧ (c : ℂ) * s^19 + (d : ℂ) * s^18 + 1 = 0) : 
  c = 1597 :=
by
  sorry

end determine_c_l188_188884


namespace divisors_greater_than_9_factorial_l188_188771

theorem divisors_greater_than_9_factorial :
  let n := 10!
  let k := 9!
  (finset.filter (λ d, d > k) (finset.divisors n)).card = 9 :=
by
  sorry

end divisors_greater_than_9_factorial_l188_188771


namespace prime_sum_of_composites_l188_188084

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def can_be_expressed_as_sum_of_two_composites (p : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b

theorem prime_sum_of_composites :
  can_be_expressed_as_sum_of_two_composites 13 ∧ 
  ∀ p : ℕ, is_prime p ∧ p > 13 → can_be_expressed_as_sum_of_two_composites p :=
by 
  sorry

end prime_sum_of_composites_l188_188084


namespace find_a_range_find_value_x1_x2_l188_188454

noncomputable def quadratic_equation_roots_and_discriminant (a : ℝ) :=
  ∃ x1 x2 : ℝ, 
      (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
      (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧
      (x1 ≠ x2) ∧ 
      (∀ Δ > 0, Δ = 9 - 8 * a - 4)

theorem find_a_range (a : ℝ) : 
  (quadratic_equation_roots_and_discriminant a) → a < 5 / 8 :=
sorry

theorem find_value_x1_x2 (a : ℤ) (h : a = 0) (x1 x2 : ℝ) :
  (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
  (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧ 
  (x1 + x2 = 3) ∧ 
  (x1 * x2 = 1) → 
  (x1^2 * x2 + x1 * x2^2 = 3) :=
sorry

end find_a_range_find_value_x1_x2_l188_188454


namespace vacation_expenses_split_l188_188127

theorem vacation_expenses_split
  (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ)
  (hA : A = 180)
  (hB : B = 240)
  (hC : C = 120)
  (ha : a = 0)
  (hb : b = 0)
  : a - b = 0 := 
by
  sorry

end vacation_expenses_split_l188_188127


namespace percent_increase_calculation_l188_188197

variable (x y : ℝ) -- Declare x and y as real numbers representing the original salary and increment

-- The statement that the percent increase z follows from the given conditions
theorem percent_increase_calculation (h : y + x = x + y) : (y / x) * 100 = ((y / x) * 100) := by
  sorry

end percent_increase_calculation_l188_188197


namespace equation_solution_l188_188820

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l188_188820


namespace manicure_cost_per_person_l188_188531

-- Definitions based on given conditions
def fingers_per_person : ℕ := 10
def total_fingers : ℕ := 210
def total_revenue : ℕ := 200  -- in dollars
def non_clients : ℕ := 11

-- Statement we want to prove
theorem manicure_cost_per_person :
  (total_revenue : ℚ) / (total_fingers / fingers_per_person - non_clients) = 9.52 :=
by
  sorry

end manicure_cost_per_person_l188_188531


namespace product_of_invertible_function_labels_l188_188732

noncomputable def Function6 (x : ℝ) : ℝ := x^3 - 3 * x
def points7 : List (ℝ × ℝ) := [(-6, 3), (-5, 1), (-4, 2), (-3, -1), (-2, 0), (-1, -2), (0, 4), (1, 5)]
noncomputable def Function8 (x : ℝ) : ℝ := Real.sin x
noncomputable def Function9 (x : ℝ) : ℝ := 3 / x

def is_invertible6 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function6 x1 = y ∧ Function6 x2 = y ∧ (-2 ≤ x1 ∧ x1 ≤ 2) ∧ (-2 ≤ x2 ∧ x2 ≤ 2)
def is_invertible7 : Prop := ∀ (y : ℝ), ∃! x : ℝ, (x, y) ∈ points7
def is_invertible8 : Prop := ∀ (x1 x2 : ℝ), Function8 x1 = Function8 x2 → x1 = x2 ∧ (-Real.pi/2 ≤ x1 ∧ x1 ≤ Real.pi/2) ∧ (-Real.pi/2 ≤ x2 ∧ x2 ≤ Real.pi/2)
def is_invertible9 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function9 x1 = y ∧ Function9 x2 = y ∧ (-4 ≤ x1 ∧ x1 ≤ 4 ∧ x1 ≠ 0) ∧ (-4 ≤ x2 ∧ x2 ≤ 4 ∧ x2 ≠ 0)

theorem product_of_invertible_function_labels :
  (is_invertible6 = false) →
  (is_invertible7 = true) →
  (is_invertible8 = true) →
  (is_invertible9 = true) →
  7 * 8 * 9 = 504
:= by
  intros h6 h7 h8 h9
  sorry

end product_of_invertible_function_labels_l188_188732


namespace meaningful_expression_range_l188_188598

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end meaningful_expression_range_l188_188598


namespace rays_form_straight_lines_l188_188148

theorem rays_form_straight_lines
  (α β : ℝ)
  (h1 : 2 * α + 2 * β = 360) :
  α + β = 180 :=
by
  -- proof details are skipped using sorry
  sorry

end rays_form_straight_lines_l188_188148


namespace pears_picked_l188_188019

def Jason_pears : ℕ := 46
def Keith_pears : ℕ := 47
def Mike_pears : ℕ := 12
def total_pears : ℕ := 105

theorem pears_picked :
  Jason_pears + Keith_pears + Mike_pears = total_pears :=
by
  exact rfl

end pears_picked_l188_188019


namespace brad_trips_to_fill_barrel_l188_188420

noncomputable def bucket_volume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

noncomputable def barrel_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem brad_trips_to_fill_barrel :
  let r_bucket := 8  -- radius of the hemisphere bucket in inches
  let r_barrel := 8  -- radius of the cylindrical barrel in inches
  let h_barrel := 20 -- height of the cylindrical barrel in inches
  let V_bucket := bucket_volume r_bucket
  let V_barrel := barrel_volume r_barrel h_barrel
  (Nat.ceil (V_barrel / V_bucket) = 4) :=
by
  sorry

end brad_trips_to_fill_barrel_l188_188420


namespace area_of_triangle_AEC_l188_188610

theorem area_of_triangle_AEC (BE EC : ℝ) (h_ratio : BE / EC = 3 / 2) (area_abe : ℝ) (h_area_abe : area_abe = 27) : 
  ∃ area_aec, area_aec = 18 :=
by
  sorry

end area_of_triangle_AEC_l188_188610


namespace compute_u2_plus_v2_l188_188345

theorem compute_u2_plus_v2 (u v : ℝ) (hu : 1 < u) (hv : 1 < v)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^(Real.sqrt 5) + 7^(Real.sqrt 5) :=
by
  sorry

end compute_u2_plus_v2_l188_188345


namespace stripe_area_is_640pi_l188_188870

noncomputable def cylinder_stripe_area (diameter height stripe_width : ℝ) (revolutions : ℕ) : ℝ :=
  let circumference := Real.pi * diameter
  let length := circumference * (revolutions : ℝ)
  stripe_width * length

theorem stripe_area_is_640pi :
  cylinder_stripe_area 20 100 4 4 = 640 * Real.pi :=
by 
  sorry

end stripe_area_is_640pi_l188_188870


namespace total_paid_is_201_l188_188267

def adult_ticket_price : ℕ := 8
def child_ticket_price : ℕ := 5
def total_tickets : ℕ := 33
def child_tickets : ℕ := 21
def adult_tickets : ℕ := total_tickets - child_tickets
def total_paid : ℕ := (child_tickets * child_ticket_price) + (adult_tickets * adult_ticket_price)

theorem total_paid_is_201 : total_paid = 201 :=
by
  sorry

end total_paid_is_201_l188_188267


namespace one_cubic_foot_is_1728_cubic_inches_l188_188916

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l188_188916


namespace garden_area_increase_l188_188104

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l188_188104


namespace spider_socks_and_shoes_ordering_l188_188864

theorem spider_socks_and_shoes_ordering : 
  let legs := 10
  let items := 3 * legs
  let socks_per_leg_orderings := 2 -- number of ways to order socks on each leg
  let total_socks_and_shoes_orderings := (fact items) / ((fact legs) * (fact (2 * legs))) * socks_per_leg_orderings^legs
  total_socks_and_shoes_orderings = (fact 30) / (fact 10 * fact 20) * 1024
:= sorry

end spider_socks_and_shoes_ordering_l188_188864


namespace factorization_correct_l188_188249

theorem factorization_correct {m : ℝ} : 
  (m^2 - 4) = (m + 2) * (m - 2) := 
by
  sorry

end factorization_correct_l188_188249


namespace initial_money_l188_188733

-- Definitions based on conditions in the problem
def money_left_after_purchase : ℕ := 3
def cost_of_candy_bar : ℕ := 1

-- Theorem statement to prove the initial amount of money
theorem initial_money (initial_amount : ℕ) :
  initial_amount - cost_of_candy_bar = money_left_after_purchase → initial_amount = 4 :=
sorry

end initial_money_l188_188733


namespace find_missing_number_l188_188536

-- Define the given numbers as a list
def given_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

-- Define the arithmetic mean condition
def arithmetic_mean (xs : List ℕ) (mean : ℕ) : Prop :=
  (xs.sum + mean) / xs.length.succ = 12

-- Define the proof problem
theorem find_missing_number (x : ℕ) (h : arithmetic_mean given_numbers x) : x = 7 := 
sorry

end find_missing_number_l188_188536


namespace fraction_spent_on_furniture_l188_188488

theorem fraction_spent_on_furniture (original_savings : ℝ) (cost_of_tv : ℝ) (f : ℝ)
  (h1 : original_savings = 1800) 
  (h2 : cost_of_tv = 450) 
  (h3 : f * original_savings + cost_of_tv = original_savings) :
  f = 3 / 4 := 
by 
  sorry

end fraction_spent_on_furniture_l188_188488


namespace correct_propositions_l188_188898

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def symmetry_about_points (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x, f (x + k) = f (x - k)

theorem correct_propositions (h1: is_odd_function f) (h2 : ∀ x, f (x + 1) = f (x -1)) :
  period_2 f ∧ (∀ k : ℤ, symmetry_about_points f k) :=
by
  sorry

end correct_propositions_l188_188898


namespace sqrt_two_irrational_l188_188856

def irrational (x : ℝ) := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem sqrt_two_irrational : irrational (Real.sqrt 2) := 
by 
  sorry

end sqrt_two_irrational_l188_188856


namespace omega_value_l188_188762

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + Real.pi / 4)

theorem omega_value (ω : ℝ) (m n : ℝ) (h : 0 < ω)
  (range_condition : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f ω x ∈ Set.Icc m n)
  (difference_condition : n - m = 3) :
  ω = (5 * Real.pi) / 12 :=
sorry

end omega_value_l188_188762


namespace work_problem_l188_188861

/-- 
  Suppose A can complete a work in \( x \) days alone, 
  B can complete the work in 20 days,
  and together they work for 7 days, leaving a fraction of 0.18333333333333335 of the work unfinished.
  Prove that \( x = 15 \).
 -/
theorem work_problem (x : ℝ) : 
  (∀ (B : ℝ), B = 20 → (∀ (f : ℝ), f = 0.18333333333333335 → (7 * (1 / x + 1 / B) = 1 - f)) → x = 15) := 
sorry

end work_problem_l188_188861


namespace codys_grandmother_age_l188_188879

theorem codys_grandmother_age
  (cody_age : ℕ)
  (grandmother_multiplier : ℕ)
  (h_cody_age : cody_age = 14)
  (h_grandmother_multiplier : grandmother_multiplier = 6) :
  (cody_age * grandmother_multiplier = 84) :=
by
  sorry

end codys_grandmother_age_l188_188879


namespace sum_of_squares_of_roots_l188_188227

theorem sum_of_squares_of_roots (x_1 x_2 : ℚ) (h1 : 6 * x_1^2 - 13 * x_1 + 5 = 0)
                                (h2 : 6 * x_2^2 - 13 * x_2 + 5 = 0) 
                                (h3 : x_1 ≠ x_2) :
  x_1^2 + x_2^2 = 109 / 36 :=
sorry

end sum_of_squares_of_roots_l188_188227


namespace onlyD_is_PythagoreanTriple_l188_188388

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def validTripleA := ¬ isPythagoreanTriple 12 15 18
def validTripleB := isPythagoreanTriple 3 4 5 ∧ (¬ (3 = 3 ∧ 4 = 4 ∧ 5 = 5)) -- Since 0.3, 0.4, 0.5 not integers
def validTripleC := ¬ isPythagoreanTriple 15 25 30 -- Conversion of 1.5, 2.5, 3 to integers
def validTripleD := isPythagoreanTriple 12 16 20

theorem onlyD_is_PythagoreanTriple : validTripleA ∧ validTripleB ∧ validTripleC ∧ validTripleD :=
by {
  sorry
}

end onlyD_is_PythagoreanTriple_l188_188388


namespace prob_complement_union_l188_188178

-- Given conditions
variable {Ω : Type*} -- Universe of events
variable (A B : Ω → Prop) -- Events A and B
variable (P : ProbabilityMassFunction Ω) -- Probability mass function
variable (m n : ℝ) -- Probabilities of events A and B
variable (hA : P.event A = m) -- Given probability of event A
variable (hB : P.event B = n) -- Given probability of event B
variable (hAB : ∀ ω, ¬ (A ω ∧ B ω)) -- A and B are mutually exclusive events

-- Desired conclusion
theorem prob_complement_union :
  P.event (λ ω, ¬ (A ω ∨ B ω)) = 1 - m - n :=
sorry

end prob_complement_union_l188_188178


namespace trigonometric_identity_cos24_cos36_sub_sin24_cos54_l188_188071

theorem trigonometric_identity_cos24_cos36_sub_sin24_cos54  :
  (Real.cos (24 * Real.pi / 180) * Real.cos (36 * Real.pi / 180) - Real.sin (24 * Real.pi / 180) * Real.cos (54 * Real.pi / 180) = 1 / 2) := by
  sorry

end trigonometric_identity_cos24_cos36_sub_sin24_cos54_l188_188071


namespace set_intersection_and_subsets_l188_188314

open Set

variable (U S T : Set ℕ)
variable hU : U = {1, 2, 3, 4, 5, 6}
variable hS : S = {1, 2, 5}
variable hT : T = {2, 3, 6}
noncomputable def complement_U_T := U \ T
noncomputable def S_inter_complement_U_T := S ∩ complement_U_T

theorem set_intersection_and_subsets :
  (S ∩ (U \ T) = {1, 5}) ∧ (card S.to_finset.powerset = 8) := by
  sorry

end set_intersection_and_subsets_l188_188314


namespace zack_initial_marbles_l188_188086

noncomputable def total_initial_marbles (x : ℕ) : ℕ :=
  81 * x + 27

theorem zack_initial_marbles :
  ∃ x : ℕ, total_initial_marbles x = 270 :=
by
  use 3
  sorry

end zack_initial_marbles_l188_188086


namespace line_slope_intercept_sum_l188_188530

theorem line_slope_intercept_sum (m b : ℝ)
    (h1 : m = 4)
    (h2 : ∃ b, ∀ x y : ℝ, y = mx + b → y = 5 ∧ x = -2)
    : m + b = 17 := by
  sorry

end line_slope_intercept_sum_l188_188530


namespace planter_cost_l188_188240

-- Define costs
def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny : ℝ := 4.00
def cost_geranium : ℝ := 3.50

-- Define quantities
def num_creeping_jennies : ℝ := 4
def num_geraniums : ℝ := 4
def num_corners : ℝ := 4

-- Define the total cost
def total_cost : ℝ :=
  (cost_palm_fern
   + (cost_creeping_jenny * num_creeping_jennies)
   + (cost_geranium * num_geraniums))
  * num_corners

-- Prove the total cost is $180.00
theorem planter_cost : total_cost = 180.00 :=
by
  sorry

end planter_cost_l188_188240


namespace solve_equation_l188_188816

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l188_188816


namespace final_cost_cooking_gear_sets_l188_188430

-- Definitions based on conditions
def hand_mitts_cost : ℕ := 14
def apron_cost : ℕ := 16
def utensils_cost : ℕ := 10
def knife_cost : ℕ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def sales_tax_rate : ℚ := 0.08
def number_of_recipients : ℕ := 3 + 5

-- Proof statement: calculate the final cost
theorem final_cost_cooking_gear_sets :
  let total_cost_before_discount := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let discounted_cost_per_set := (total_cost_before_discount : ℚ) * (1 - discount_rate)
  let total_cost_for_recipients := (discounted_cost_per_set * number_of_recipients : ℚ)
  let final_cost := total_cost_for_recipients * (1 + sales_tax_rate)
  final_cost = 388.80 :=
by
  sorry

end final_cost_cooking_gear_sets_l188_188430


namespace minimum_berries_left_l188_188252

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

theorem minimum_berries_left {a r n S : ℕ} 
  (h_a : a = 1) 
  (h_r : r = 2) 
  (h_n : n = 100) 
  (h_S : S = geometric_sum a r n) 
  : S = 2^100 - 1 -> ∃ k, k = 100 :=
by
  sorry

end minimum_berries_left_l188_188252


namespace female_democrats_l188_188251

theorem female_democrats (F M D_f: ℕ) 
  (h1 : F + M = 780)
  (h2 : D_f = (1/2) * F)
  (h3 : (1/3) * 780 = 260)
  (h4 : 260 = (1/2) * F + (1/4) * M) : 
  D_f = 130 := 
by
  sorry

end female_democrats_l188_188251


namespace min_value_geometric_seq_l188_188473

theorem min_value_geometric_seq (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * r)
  (h3 : a 5 * a 4 * a 2 * a 1 = 16) :
  a 1 + a 5 = 4 :=
sorry

end min_value_geometric_seq_l188_188473


namespace value_of_a_l188_188011

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  sorry

end value_of_a_l188_188011


namespace box_2008_count_l188_188847

noncomputable def box_count (a : ℕ → ℕ) : Prop :=
  a 1 = 7 ∧ a 4 = 8 ∧ ∀ n : ℕ, 1 ≤ n ∧ n + 3 ≤ 2008 → a n + a (n + 1) + a (n + 2) + a (n + 3) = 30

theorem box_2008_count (a : ℕ → ℕ) (h : box_count a) : a 2008 = 8 :=
by
  sorry

end box_2008_count_l188_188847


namespace chameleons_all_red_l188_188632

theorem chameleons_all_red (Y G R : ℕ) (total : ℕ) (P : Y = 7) (Q : G = 10) (R_cond : R = 17) (total_cond : Y + G + R = total) (total_value : total = 34) :
  ∃ x, x = R ∧ x = total ∧ ∀ z : ℕ, z ≠ 0 → total % 3 = z % 3 → ((R : ℕ) % 3 = z) :=
by
  sorry

end chameleons_all_red_l188_188632


namespace arithmetic_mean_18_27_45_l188_188685

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_18_27_45_l188_188685


namespace towers_per_castle_jeff_is_5_l188_188647

-- Define the number of sandcastles on Mark's beach
def num_castles_mark : ℕ := 20

-- Define the number of towers per sandcastle on Mark's beach
def towers_per_castle_mark : ℕ := 10

-- Calculate the total number of towers on Mark's beach
def total_towers_mark : ℕ := num_castles_mark * towers_per_castle_mark

-- Define the number of sandcastles on Jeff's beach (3 times that of Mark's)
def num_castles_jeff : ℕ := 3 * num_castles_mark

-- Define the total number of sandcastles on both beaches
def total_sandcastles : ℕ := num_castles_mark + num_castles_jeff
  
-- Define the combined total number of sandcastles and towers on both beaches
def combined_total : ℕ := 580

-- Define the number of towers per sandcastle on Jeff's beach
def towers_per_castle_jeff : ℕ := sorry

-- Define the total number of towers on Jeff's beach
def total_towers_jeff (T : ℕ) : ℕ := num_castles_jeff * T

-- Prove that the number of towers per sandcastle on Jeff's beach is 5
theorem towers_per_castle_jeff_is_5 : 
    200 + total_sandcastles + total_towers_jeff towers_per_castle_jeff = combined_total → 
    towers_per_castle_jeff = 5
:= by
    sorry

end towers_per_castle_jeff_is_5_l188_188647


namespace problem1_problem2_problem3_l188_188134

-- Problem 1
theorem problem1 : 13 + (-7) - (-9) + 5 * (-2) = 5 :=
by 
  sorry

-- Problem 2
theorem problem2 : abs (-7 / 2) * (12 / 7) / (4 / 3) / (3 ^ 2) = 1 / 2 :=
by 
  sorry

-- Problem 3
theorem problem3 : -1^4 - (1 / 6) * (2 - (-3)^2) = 1 / 6 :=
by 
  sorry

end problem1_problem2_problem3_l188_188134


namespace ellipse_eccentricity_l188_188310

theorem ellipse_eccentricity :
  (∃ (e : ℝ), (∀ (x y : ℝ), ((x^2 / 9) + y^2 = 1) → (e = 2 * Real.sqrt 2 / 3))) :=
by
  sorry

end ellipse_eccentricity_l188_188310


namespace average_infections_l188_188403

theorem average_infections (x : ℝ) (h : 1 + x + x^2 = 121) : x = 10 :=
sorry

end average_infections_l188_188403


namespace tangent_line_eqn_unique_local_minimum_l188_188579

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + 2) / x

def tangent_line_at_1 (x y : ℝ) : Prop :=
  2 * x + y - Real.exp 1 - 4 = 0

theorem tangent_line_eqn :
  tangent_line_at_1 1 (f 1) :=
sorry

noncomputable def h (x : ℝ) : ℝ := Real.exp x * (x - 1) - 2

theorem unique_local_minimum :
  ∃! c : ℝ, 1 < c ∧ c < 2 ∧ (∀ x < c, f x > f c) ∧ (∀ x > c, f c < f x) :=
sorry

end tangent_line_eqn_unique_local_minimum_l188_188579


namespace probability_not_black_l188_188700

theorem probability_not_black (white_balls black_balls red_balls : ℕ) (total_balls : ℕ) (non_black_balls : ℕ) :
  white_balls = 7 → black_balls = 6 → red_balls = 4 →
  total_balls = white_balls + black_balls + red_balls →
  non_black_balls = white_balls + red_balls →
  (non_black_balls / total_balls : ℚ) = 11 / 17 :=
by
  sorry

end probability_not_black_l188_188700


namespace binom_20_19_eq_20_l188_188726

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l188_188726


namespace value_of_expression_l188_188941

theorem value_of_expression (a b : ℝ) (h1 : 3 * a^2 + 9 * a - 21 = 0) (h2 : 3 * b^2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (5 * b - 6) = -27 :=
by
  -- The proof is omitted, place 'sorry' to indicate it.
  sorry

end value_of_expression_l188_188941


namespace ab_necessary_not_sufficient_l188_188150

theorem ab_necessary_not_sufficient (a b : ℝ) : 
  (ab > 0) ↔ ((a ≠ 0) ∧ (b ≠ 0) ∧ ((b / a + a / b > 2) → (ab > 0))) := 
sorry

end ab_necessary_not_sufficient_l188_188150


namespace cubic_inches_in_one_cubic_foot_l188_188910

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l188_188910


namespace two_categorical_variables_l188_188474

-- Definitions based on the conditions
def smoking (x : String) : Prop := x = "Smoking" ∨ x = "Not smoking"
def sick (y : String) : Prop := y = "Sick" ∨ y = "Not sick"

def category1 (z : String) : Prop := z = "Whether smoking"
def category2 (w : String) : Prop := w = "Whether sick"

-- The main proof statement
theorem two_categorical_variables : 
  (category1 "Whether smoking" ∧ smoking "Smoking" ∧ smoking "Not smoking") ∧
  (category2 "Whether sick" ∧ sick "Sick" ∧ sick "Not sick") →
  "Whether smoking, Whether sick" = "Whether smoking, Whether sick" :=
by
  sorry

end two_categorical_variables_l188_188474


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l188_188045

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l188_188045


namespace total_candies_third_set_l188_188654

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l188_188654


namespace min_e1_plus_2e2_l188_188151

noncomputable def e₁ (r : ℝ) : ℝ := 2 / (4 - r)
noncomputable def e₂ (r : ℝ) : ℝ := 2 / (4 + r)

theorem min_e1_plus_2e2 (r : ℝ) (h₀ : 0 < r) (h₂ : r < 2) :
  e₁ r + 2 * e₂ r = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_e1_plus_2e2_l188_188151


namespace squares_difference_l188_188775

theorem squares_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) : a^2 - b^2 = 15 :=
by
  sorry

end squares_difference_l188_188775


namespace probability_journalist_A_to_group_A_l188_188224

open Nat

theorem probability_journalist_A_to_group_A :
  let group_A := 0
  let group_B := 1
  let group_C := 2
  let journalists := [0, 1, 2, 3]  -- four journalists

  -- total number of ways to distribute 4 journalists into 3 groups such that each group has at least one journalist
  let total_ways := 36

  -- number of ways to assign journalist 0 to group A specifically
  let favorable_ways := 12

  -- probability calculation
  ∃ (prob : ℚ), prob = favorable_ways / total_ways ∧ prob = 1 / 3 :=
sorry

end probability_journalist_A_to_group_A_l188_188224


namespace rational_number_div_eq_l188_188318

theorem rational_number_div_eq :
  ∃ x : ℚ, (-2 : ℚ) / x = 8 ∧ x = -1 / 4 :=
by
  existsi (-1 / 4 : ℚ)
  sorry

end rational_number_div_eq_l188_188318


namespace total_candies_in_third_set_l188_188674

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l188_188674


namespace count_five_letter_words_l188_188338

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end count_five_letter_words_l188_188338


namespace scout_hours_worked_l188_188492

variable (h : ℕ) -- number of hours worked on Saturday
variable (base_pay : ℕ) -- base pay per hour
variable (tip_per_customer : ℕ) -- tip per customer
variable (saturday_customers : ℕ) -- customers served on Saturday
variable (sunday_hours : ℕ) -- hours worked on Sunday
variable (sunday_customers : ℕ) -- customers served on Sunday
variable (total_earnings : ℕ) -- total earnings over the weekend

theorem scout_hours_worked {h : ℕ} (base_pay : ℕ) (tip_per_customer : ℕ) (saturday_customers : ℕ) (sunday_hours : ℕ) (sunday_customers : ℕ) (total_earnings : ℕ) :
  base_pay = 10 → 
  tip_per_customer = 5 → 
  saturday_customers = 5 → 
  sunday_hours = 5 → 
  sunday_customers = 8 → 
  total_earnings = 155 → 
  10 * h + 5 * 5 + 10 * 5 + 5 * 8 = 155 → 
  h = 4 :=
by
  intros
  sorry

end scout_hours_worked_l188_188492


namespace jelly_bean_matching_probability_l188_188537

-- Define the conditions
def abe_jelly_beans := [2/5, 3/5] -- probabilities for green and red respectively
def bob_jelly_beans := [2/7, 3/7] -- probabilities for green and red respectively

-- Define the event of matching colors
def prob_match_color : ℚ :=
  (abe_jelly_beans.head * bob_jelly_beans.head) + (abe_jelly_beans.tail.head * bob_jelly_beans.tail.head)

-- The proof goal
theorem jelly_bean_matching_probability :
  prob_match_color = 13 / 35 :=
by
  sorry

end jelly_bean_matching_probability_l188_188537


namespace inequality_chain_l188_188569

open Real

theorem inequality_chain (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_chain_l188_188569


namespace roots_of_quadratic_l188_188888

theorem roots_of_quadratic {a b c : ℝ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x, (x = a ∨ x = b ∨ x = c) ↔ x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 :=
by
  sorry

end roots_of_quadratic_l188_188888


namespace minimum_value_of_x_squared_l188_188067

theorem minimum_value_of_x_squared : ∃ x : ℝ, x = 0 ∧ ∀ y : ℝ, y = x^2 → y ≥ 0 :=
by
  sorry

end minimum_value_of_x_squared_l188_188067


namespace compute_expression_l188_188718

theorem compute_expression : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end compute_expression_l188_188718


namespace range_x_minus_2y_l188_188753

variable (x y : ℝ)

def cond1 : Prop := -1 ≤ x ∧ x < 2
def cond2 : Prop := 0 < y ∧ y ≤ 1

theorem range_x_minus_2y 
  (h1 : cond1 x) 
  (h2 : cond2 y) : 
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 := 
by
  sorry

end range_x_minus_2y_l188_188753


namespace luis_bought_6_pairs_of_blue_socks_l188_188029

open Nat

-- Conditions
def total_pairs_red := 4
def total_cost_red := 3
def total_cost := 42
def blue_socks_cost := 5

-- Deduce the spent amount on red socks, and from there calculate the number of blue socks bought.
theorem luis_bought_6_pairs_of_blue_socks :
  (yes : ℕ) -> yes * blue_socks_cost = total_cost - total_pairs_red * total_cost_red → yes = 6 :=
sorry

end luis_bought_6_pairs_of_blue_socks_l188_188029


namespace candy_count_in_third_set_l188_188664

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l188_188664


namespace jenny_run_distance_l188_188020

theorem jenny_run_distance (walk_distance : ℝ) (ran_walk_diff : ℝ) (h_walk : walk_distance = 0.4) (h_diff : ran_walk_diff = 0.2) :
  (walk_distance + ran_walk_diff) = 0.6 :=
sorry

end jenny_run_distance_l188_188020


namespace part1_part2_l188_188761

noncomputable def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

theorem part1 (a : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → f x ≤ a) ↔ a ≥ 4 :=
sorry

theorem part2 : {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ -4 < x ∧ x < 1} :=
sorry

end part1_part2_l188_188761


namespace five_letter_words_start_end_same_l188_188336

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end five_letter_words_start_end_same_l188_188336


namespace swimming_time_per_style_l188_188649

theorem swimming_time_per_style (d v1 v2 v3 v4 t: ℝ) 
    (h1: d = 600) 
    (h2: v1 = 45) 
    (h3: v2 = 35) 
    (h4: v3 = 40) 
    (h5: v4 = 30)
    (h6: t = 15) 
    (h7: d / 4 = 150) 
    : (t / 4 = 3.75) :=
by
  sorry

end swimming_time_per_style_l188_188649


namespace fractional_equation_no_solution_l188_188059

theorem fractional_equation_no_solution (x : ℝ) (h1 : x ≠ 3) : (2 - x) / (x - 3) ≠ 1 + 1 / (3 - x) :=
by
  sorry

end fractional_equation_no_solution_l188_188059


namespace find_q_l188_188374

theorem find_q (p q : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) (hp_congr : 5 * p ≡ 3 [MOD 4]) (hq_def : q = 13 * p + 2) : q = 41 := 
sorry

end find_q_l188_188374


namespace weekly_rental_fee_percentage_l188_188339

theorem weekly_rental_fee_percentage
  (camera_value : ℕ)
  (rental_period_weeks : ℕ)
  (friend_percentage : ℚ)
  (john_paid : ℕ)
  (percentage : ℚ)
  (total_rental_fee : ℚ)
  (weekly_rental_fee : ℚ)
  (P : ℚ)
  (camera_value_pos : camera_value = 5000)
  (rental_period_weeks_pos : rental_period_weeks = 4)
  (friend_percentage_pos : friend_percentage = 0.40)
  (john_paid_pos : john_paid = 1200)
  (percentage_pos : percentage = 1 - friend_percentage)
  (total_rental_fee_calc : total_rental_fee = john_paid / percentage)
  (weekly_rental_fee_calc : weekly_rental_fee = total_rental_fee / rental_period_weeks)
  (weekly_rental_fee_equation : weekly_rental_fee = P * camera_value)
  (P_calc : P = weekly_rental_fee / camera_value) :
  P * 100 = 10 := 
by 
  sorry

end weekly_rental_fee_percentage_l188_188339


namespace union_of_S_and_T_l188_188805

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := 
by
  sorry

end union_of_S_and_T_l188_188805


namespace increase_factor_is_46_8_l188_188517

-- Definitions for the conditions
def old_plates : ℕ := 26^3 * 10^3
def new_plates_type_A : ℕ := 26^2 * 10^4
def new_plates_type_B : ℕ := 26^4 * 10^2
def average_new_plates := (new_plates_type_A + new_plates_type_B) / 2

-- The Lean 4 statement to prove that the increase factor is 46.8
theorem increase_factor_is_46_8 :
  (average_new_plates : ℚ) / (old_plates : ℚ) = 46.8 := by
  sorry

end increase_factor_is_46_8_l188_188517


namespace dislike_both_tv_and_video_games_l188_188356

theorem dislike_both_tv_and_video_games (total_people : ℕ) (percent_dislike_tv : ℝ) (percent_dislike_tv_and_games : ℝ) :
  let people_dislike_tv := percent_dislike_tv * total_people
  let people_dislike_both := percent_dislike_tv_and_games * people_dislike_tv
  total_people = 1800 ∧ percent_dislike_tv = 0.4 ∧ percent_dislike_tv_and_games = 0.25 →
  people_dislike_both = 180 :=
by {
  sorry
}

end dislike_both_tv_and_video_games_l188_188356


namespace alpha_in_second_quadrant_l188_188166

theorem alpha_in_second_quadrant (α : ℝ) 
  (h1 : Real.sin α > Real.cos α)
  (h2 : Real.sin α * Real.cos α < 0) : 
  (Real.sin α > 0) ∧ (Real.cos α < 0) :=
by 
  -- Proof omitted
  sorry

end alpha_in_second_quadrant_l188_188166


namespace zero_in_M_l188_188765

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
by
  sorry

end zero_in_M_l188_188765


namespace profit_percentage_l188_188532

theorem profit_percentage (CP SP : ℝ) (h₁ : CP = 400) (h₂ : SP = 560) : 
  ((SP - CP) / CP) * 100 = 40 := by 
  sorry

end profit_percentage_l188_188532


namespace distance_between_Sasha_and_Koyla_is_19m_l188_188051

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l188_188051


namespace prob_one_exceeds_90_l188_188325

open Probability

-- Definitions of problem conditions
def normal_distribution (mean variance : ℝ) : ℝ → ℝ :=
λ x, 1 / (sqrt (2 * π * variance)) * exp (-((x - mean)^2) / (2 * variance))

def P_between (X : ℝ → ℝ) (a b : ℝ) : ℝ :=
stintegral ℝ (indicator (set.Icc a b) X)

def P_exceeds (X : ℝ → ℝ) (threshold : ℝ) : ℝ :=
1 - P_between X (threshold - real.pi) threshold

-- Problem conditions
noncomputable def X := normal_distribution 80 (sigma^2)
axiom P_X_between : P_between X 70 90 = 1 / 3
axiom number_of_students : ℕ := 3

-- Theorem statement
theorem prob_one_exceeds_90 :
  let P_X_exceeds_90 := P_exceeds X 90 in
  P_X_exceeds_90 = 1 / 3 →
  let P_event_A := (number_of_students.choose 1 * (2 / 3)^2 * (1 / 3)) in
  P_event_A = 4 / 9 :=
sorry

end prob_one_exceeds_90_l188_188325


namespace minimum_value_frac_sum_l188_188691

-- Define the statement problem C and proof outline skipping the steps
theorem minimum_value_frac_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
by
  -- Proof is to be constructed here
  sorry

end minimum_value_frac_sum_l188_188691


namespace range_x_minus_2y_l188_188754

variable (x y : ℝ)

def cond1 : Prop := -1 ≤ x ∧ x < 2
def cond2 : Prop := 0 < y ∧ y ≤ 1

theorem range_x_minus_2y 
  (h1 : cond1 x) 
  (h2 : cond2 y) : 
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 := 
by
  sorry

end range_x_minus_2y_l188_188754


namespace find_x_l188_188309

-- We define the given condition in Lean
theorem find_x (x : ℝ) (h : 6 * x - 12 = -(4 + 2 * x)) : x = 1 :=
sorry

end find_x_l188_188309


namespace find_other_solution_l188_188901

theorem find_other_solution (x₁ : ℚ) (x₂ : ℚ) 
  (h₁ : x₁ = 3 / 4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) 
  (eq : 72 * x₂^2 + 39 * x₂ - 18 = 0 ∧ x₂ ≠ x₁) : 
  x₂ = -31 / 6 := 
sorry

end find_other_solution_l188_188901


namespace frank_change_l188_188440

theorem frank_change (n_c n_b money_given c_c c_b : ℕ) 
  (h1 : n_c = 5) 
  (h2 : n_b = 2) 
  (h3 : money_given = 20) 
  (h4 : c_c = 2) 
  (h5 : c_b = 3) : 
  money_given - (n_c * c_c + n_b * c_b) = 4 := 
by
  sorry

end frank_change_l188_188440


namespace simplify_fraction_l188_188495

theorem simplify_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end simplify_fraction_l188_188495


namespace student_arrangements_l188_188254

theorem student_arrangements  :
  let num_students : ℕ := 6 in
  let venues : Finset ℕ := {1, 2, 3} in
  let venueA_students : ℕ := 1 in
  let venueB_students : ℕ := 2 in
  let venueC_students : ℕ := 3 in
  num_students = venueA_students + venueB_students + venueC_students →
  nat.choose num_students venueA_students *
  nat.choose (num_students - venueA_students) venueB_students *
  nat.choose (num_students - venueA_students - venueB_students) venueC_students = 60 :=
by
  intros num_students venues venueA_students venueB_students venueC_students h_sum_eq
  sorry

end student_arrangements_l188_188254


namespace squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l188_188523

theorem squares_have_consecutive_digits (n : ℕ) (h : ∃ j : ℕ, n = 33330 + j ∧ j < 10) :
    ∃ (a b : ℕ), n ^ 2 / 10 ^ a % 10 = n ^ 2 / 10 ^ (a + 1) % 10 :=
by
  sorry

theorem generalized_squares_have_many_consecutive_digits (k : ℕ) (n : ℕ)
  (h1 : k ≥ 4)
  (h2 : ∃ j : ℕ, n = 33333 * 10 ^ (k - 4) + j ∧ j < 10 ^ (k - 4)) :
    ∃ m, ∃ l : ℕ, ∀ i < m, n^2 / 10 ^ (l + i) % 10 = n^2 / 10 ^ l % 10 :=
by
  sorry

end squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l188_188523


namespace distance_between_Sasha_and_Koyla_is_19m_l188_188050

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l188_188050


namespace softball_players_l188_188323

theorem softball_players (cricket hockey football total : ℕ) (h1 : cricket = 12) (h2 : hockey = 17) (h3 : football = 11) (h4 : total = 50) : 
  total - (cricket + hockey + football) = 10 :=
by
  sorry

end softball_players_l188_188323


namespace exp_fn_max_min_diff_l188_188446

theorem exp_fn_max_min_diff (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (max (a^1) (a^0) - min (a^1) (a^0)) = 1 / 2 → (a = 1 / 2 ∨ a = 3 / 2) :=
by
  sorry

end exp_fn_max_min_diff_l188_188446


namespace arithmetic_sequence_solution_l188_188939

-- Definitions of a, b, c, and d in terms of d and sequence difference
def is_in_arithmetic_sequence (a b c d : ℝ) (diff : ℝ) : Prop :=
  a + diff = b ∧ b + diff = c ∧ c + diff = d

-- Conditions
def pos_real_sequence (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

def product_condition (a b c d : ℝ) (prod : ℝ) : Prop :=
  a * b * c * d = prod

-- The resulting value of d
def d_value_as_fraction (d : ℝ) : Prop :=
  d = (3 + Real.sqrt 95) / (Real.sqrt 2)

-- Proof statement
theorem arithmetic_sequence_solution :
  ∃ a b c d : ℝ, pos_real_sequence a b c d ∧ 
                 is_in_arithmetic_sequence a b c d (Real.sqrt 2) ∧ 
                 product_condition a b c d 2021 ∧ 
                 d_value_as_fraction d :=
sorry

end arithmetic_sequence_solution_l188_188939


namespace terry_total_driving_time_l188_188826

-- Define the conditions
def speed : ℝ := 40 -- miles per hour
def distance : ℝ := 60 -- miles

-- Define the time for one trip
def time_for_one_trip (d : ℝ) (s : ℝ) : ℝ := d / s

-- Define the total driving time for a round trip (forth and back)
def total_driving_time (d : ℝ) (s : ℝ) : ℝ := 2 * time_for_one_trip d s

-- State the theorem to be proven
theorem terry_total_driving_time : total_driving_time distance speed = 3 := 
by
  sorry

end terry_total_driving_time_l188_188826


namespace sum_of_interior_angles_l188_188580

def f (n : ℕ) : ℚ := (n - 2) * 180

theorem sum_of_interior_angles (n : ℕ) : f (n + 1) = f n + 180 :=
by
  unfold f
  sorry

end sum_of_interior_angles_l188_188580


namespace interest_rate_per_annum_l188_188363

-- Given conditions
variables (BG TD t : ℝ) (FV r : ℝ)
axiom bg_eq : BG = 6
axiom td_eq : TD = 50
axiom t_eq : t = 1
axiom bankers_gain_eq : BG = FV * r * t - (FV - TD) * r * t

-- Proof problem
theorem interest_rate_per_annum : r = 0.12 :=
by sorry

end interest_rate_per_annum_l188_188363


namespace readers_both_l188_188927

-- Definitions of the number of readers
def total_readers : ℕ := 150
def readers_science_fiction : ℕ := 120
def readers_literary_works : ℕ := 90

-- Statement of the proof problem
theorem readers_both :
  (readers_science_fiction + readers_literary_works - total_readers) = 60 :=
by
  -- Proof omitted
  sorry

end readers_both_l188_188927


namespace find_positive_real_number_solution_l188_188559

theorem find_positive_real_number_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) (hx : x > 0) : x = 15 :=
sorry

end find_positive_real_number_solution_l188_188559


namespace village_distance_l188_188508

theorem village_distance
  (d : ℝ)
  (uphill_speed : ℝ) (downhill_speed : ℝ)
  (total_time : ℝ)
  (h1 : uphill_speed = 15)
  (h2 : downhill_speed = 30)
  (h3 : total_time = 4) :
  d = 40 :=
by
  sorry

end village_distance_l188_188508


namespace divisors_of_factorial_gt_nine_factorial_l188_188772

theorem divisors_of_factorial_gt_nine_factorial :
  let ten_factorial := Nat.factorial 10
  let nine_factorial := Nat.factorial 9
  let divisors := {d // d > nine_factorial ∧ d ∣ ten_factorial}
  (divisors.card = 9) :=
by
  sorry

end divisors_of_factorial_gt_nine_factorial_l188_188772


namespace problems_per_page_l188_188091

theorem problems_per_page (total_problems finished_problems remaining_pages : Nat) (h1 : total_problems = 101) 
  (h2 : finished_problems = 47) (h3 : remaining_pages = 6) :
  (total_problems - finished_problems) / remaining_pages = 9 :=
by
  sorry

end problems_per_page_l188_188091


namespace apples_left_correct_l188_188062

noncomputable def apples_left (initial_apples : ℝ) (additional_apples : ℝ) (apples_for_pie : ℝ) : ℝ :=
  initial_apples + additional_apples - apples_for_pie

theorem apples_left_correct :
  apples_left 10.0 5.5 4.25 = 11.25 :=
by
  sorry

end apples_left_correct_l188_188062


namespace line_parabola_intersect_l188_188602

theorem line_parabola_intersect {k : ℝ} 
    (h1: ∀ x y : ℝ, y = k*x - 2 → y^2 = 8*x → x ≠ y)
    (h2: ∀ x1 x2 y1 y2 : ℝ, y1 = k*x1 - 2 → y2 = k*x2 - 2 → y1^2 = 8*x1 → y2^2 = 8*x2 → (x1 + x2) / 2 = 2) : 
    k = 2 := 
sorry

end line_parabola_intersect_l188_188602


namespace box_height_l188_188529

variables (length width : ℕ) (cube_volume cubes total_volume : ℕ)
variable (height : ℕ)

theorem box_height :
  length = 12 →
  width = 16 →
  cube_volume = 3 →
  cubes = 384 →
  total_volume = cubes * cube_volume →
  total_volume = length * width * height →
  height = 6 :=
by
  intros
  sorry

end box_height_l188_188529


namespace total_candies_third_set_l188_188657

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l188_188657


namespace remainder_3_pow_20_mod_7_l188_188244

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l188_188244


namespace marcella_matching_pairs_l188_188986

theorem marcella_matching_pairs (P : ℕ) (L : ℕ) (H : P = 20) (H1 : L = 9) : (P - L) / 2 = 11 :=
by
  -- definition of P and L are given by 20 and 9 respectively
  -- proof is omitted for the statement focus
  sorry

end marcella_matching_pairs_l188_188986


namespace david_marks_in_mathematics_l188_188281

-- Define marks in individual subjects and the average
def marks_in_english : ℝ := 70
def marks_in_physics : ℝ := 78
def marks_in_chemistry : ℝ := 60
def marks_in_biology : ℝ := 65
def average_marks : ℝ := 66.6
def number_of_subjects : ℕ := 5

-- Define a statement to be proven
theorem david_marks_in_mathematics : 
    average_marks * number_of_subjects 
    - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 60 := 
by simp [average_marks, number_of_subjects, marks_in_english, marks_in_physics, marks_in_chemistry, marks_in_biology]; sorry

end david_marks_in_mathematics_l188_188281


namespace boy_reaches_early_l188_188075

-- Given conditions
def usual_time : ℚ := 42
def rate_multiplier : ℚ := 7 / 6

-- Derived variables
def new_time : ℚ := (6 / 7) * usual_time
def early_time : ℚ := usual_time - new_time

-- The statement to prove
theorem boy_reaches_early : early_time = 6 := by
  sorry

end boy_reaches_early_l188_188075


namespace correct_equation_l188_188828

theorem correct_equation (x Planned : ℝ) (h1 : 6 * x = Planned + 7) (h2 : 5 * x = Planned - 13) :
  6 * x - 7 = 5 * x + 13 :=
by
  sorry

end correct_equation_l188_188828


namespace total_lines_to_write_l188_188979

theorem total_lines_to_write (lines_per_page pages_needed : ℕ) (h1 : lines_per_page = 30) (h2 : pages_needed = 5) : lines_per_page * pages_needed = 150 :=
by {
  sorry
}

end total_lines_to_write_l188_188979


namespace length_rest_of_body_l188_188353

theorem length_rest_of_body (height legs head arms rest_of_body : ℝ) 
  (hlegs : legs = (1/3) * height)
  (hhead : head = (1/4) * height)
  (harms : arms = (1/5) * height)
  (htotal : height = 180)
  (hr: rest_of_body = height - (legs + head + arms)) : 
  rest_of_body = 39 :=
by
  -- proof is not required
  sorry

end length_rest_of_body_l188_188353


namespace fred_has_9_dimes_l188_188293

-- Fred has 90 cents in his bank.
def freds_cents : ℕ := 90

-- A dime is worth 10 cents.
def value_of_dime : ℕ := 10

-- Prove that the number of dimes Fred has is 9.
theorem fred_has_9_dimes : (freds_cents / value_of_dime) = 9 := by
  sorry

end fred_has_9_dimes_l188_188293


namespace intersection_result_l188_188169

open Set

namespace ProofProblem

def A : Set ℝ := {x | |x| ≤ 4}
def B : Set ℝ := {x | 4 ≤ x ∧ x < 5}

theorem intersection_result : A ∩ B = {4} :=
  sorry

end ProofProblem

end intersection_result_l188_188169


namespace max_disjoint_regions_l188_188341

theorem max_disjoint_regions {p : ℕ} (hp : Nat.Prime p) (hp_ge3 : 3 ≤ p) : ∃ R, R = 3 * p^2 - 3 * p + 1 :=
by
  sorry

end max_disjoint_regions_l188_188341


namespace divisibility_theorem_l188_188037

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n + 1) ∣ (a + 1)^b - 1 :=
by 
sorry

end divisibility_theorem_l188_188037


namespace determine_x_l188_188426

noncomputable def x_candidates := { x : ℝ | x = (3 + Real.sqrt 105) / 24 ∨ x = (3 - Real.sqrt 105) / 24 }

theorem determine_x (x y : ℝ) (h_y : y = 3 * x) 
  (h_eq : 4 * y ^ 2 + 2 * y + 7 = 3 * (8 * x ^ 2 + y + 3)) :
  x ∈ x_candidates :=
by
  sorry

end determine_x_l188_188426


namespace minimum_races_to_find_top3_l188_188355

-- Define a constant to represent the number of horses and maximum horses per race
def total_horses : ℕ := 25
def max_horses_per_race : ℕ := 5

-- Define the problem statement as a theorem
theorem minimum_races_to_find_top3 (total_horses : ℕ) (max_horses_per_race : ℕ) : ℕ :=
  if total_horses = 25 ∧ max_horses_per_race = 5 then 7 else sorry

end minimum_races_to_find_top3_l188_188355


namespace total_tips_l188_188974

def tips_per_customer := 2
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

theorem total_tips : 
  (tips_per_customer * (customers_friday + customers_saturday + customers_sunday) = 296) :=
by
  sorry

end total_tips_l188_188974


namespace number_of_teachers_under_40_in_sample_l188_188179

def proportion_teachers_under_40 (total_teachers teachers_under_40 : ℕ) : ℚ :=
  teachers_under_40 / total_teachers

def sample_teachers_under_40 (sample_size : ℕ) (proportion : ℚ) : ℚ :=
  sample_size * proportion

theorem number_of_teachers_under_40_in_sample
(total_teachers teachers_under_40 teachers_40_and_above sample_size : ℕ)
(h_total : total_teachers = 400)
(h_under_40 : teachers_under_40 = 250)
(h_40_and_above : teachers_40_and_above = 150)
(h_sample_size : sample_size = 80)
: sample_teachers_under_40 sample_size 
  (proportion_teachers_under_40 total_teachers teachers_under_40) = 50 := by
sorry

end number_of_teachers_under_40_in_sample_l188_188179


namespace bowling_ball_weight_l188_188555

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end bowling_ball_weight_l188_188555


namespace greatest_positive_integer_x_l188_188382

theorem greatest_positive_integer_x (x : ℕ) (h₁ : x^2 < 12) (h₂ : ∀ y: ℕ, y^2 < 12 → y ≤ x) : 
  x = 3 := 
by
  sorry

end greatest_positive_integer_x_l188_188382


namespace number_of_rectangles_l188_188274

-- Definition of the problem: We have 12 equally spaced points on a circle.
def points_on_circle : ℕ := 12

-- The number of diameters is half the number of points, as each diameter involves two points.
def diameters (n : ℕ) : ℕ := n / 2

-- The number of ways to choose 2 diameters out of n/2 is given by the binomial coefficient.
noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Prove the number of rectangles that can be formed is 15.
theorem number_of_rectangles :
  binomial_coefficient (diameters points_on_circle) 2 = 15 := by
  sorry

end number_of_rectangles_l188_188274


namespace speed_of_current_l188_188259

theorem speed_of_current (m c : ℝ) (h1 : m + c = 18) (h2 : m - c = 11.2) : c = 3.4 :=
by
  sorry

end speed_of_current_l188_188259


namespace total_face_value_of_notes_l188_188173

theorem total_face_value_of_notes :
  let face_value := 5
  let number_of_notes := 440 * 10^6
  face_value * number_of_notes = 2200000000 := 
by
  sorry

end total_face_value_of_notes_l188_188173


namespace inequality_solution_l188_188626

theorem inequality_solution
  (f : ℝ → ℝ)
  (h_deriv : ∀ x : ℝ, deriv f x > 2 * f x)
  (h_value : f (1/2) = Real.exp 1)
  (x : ℝ)
  (h_pos : 0 < x) :
  f (Real.log x) < x^2 ↔ x < Real.exp (1/2) :=
sorry

end inequality_solution_l188_188626


namespace find_first_term_of_geometric_series_l188_188272

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end find_first_term_of_geometric_series_l188_188272


namespace unique_a_for_intersection_l188_188906

def A (a : ℝ) : Set ℝ := {-4, 2 * a - 1, a^2}
def B (a : ℝ) : Set ℝ := {a - 5, 1 - a, 9}

theorem unique_a_for_intersection (a : ℝ) :
  (9 ∈ A a ∩ B a ∧ ∀ x, x ∈ A a ∩ B a → x = 9) ↔ a = -3 := by
  sorry

end unique_a_for_intersection_l188_188906


namespace distance_between_Sasha_and_Kolya_l188_188040

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l188_188040


namespace simplify_expr_1_simplify_expr_2_l188_188057

theorem simplify_expr_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y :=
by
  sorry

theorem simplify_expr_2 (a b : ℝ) :
  (3 / 2) * (a^2 * b - 2 * (a * b^2)) - (1 / 2) * (a * b^2 - 4 * (a^2 * b)) + (a * b^2) / 2 = (7 / 2) * (a^2 * b) - 3 * (a * b^2) :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l188_188057


namespace number_of_boxes_ordered_l188_188741

-- Definitions based on the conditions
def boxes_contain_matchboxes : Nat := 20
def matchboxes_contain_sticks : Nat := 300
def total_match_sticks : Nat := 24000

-- Statement of the proof problem
theorem number_of_boxes_ordered :
  (total_match_sticks / matchboxes_contain_sticks) / boxes_contain_matchboxes = 4 := 
sorry

end number_of_boxes_ordered_l188_188741


namespace problem_statement_l188_188078

theorem problem_statement (h : 36 = 6^2) : 6^15 / 36^5 = 7776 := by
  sorry

end problem_statement_l188_188078


namespace nested_fraction_simplifies_l188_188541

theorem nested_fraction_simplifies : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := 
by 
  sorry

end nested_fraction_simplifies_l188_188541


namespace value_of_y_l188_188320

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 :=
by
  sorry

end value_of_y_l188_188320


namespace proof_equivalent_l188_188090

variables {α : Type*} [Field α]

theorem proof_equivalent (a b c d e f : α)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 :=
by sorry

end proof_equivalent_l188_188090


namespace runner_speed_ratio_l188_188851

noncomputable def speed_ratio (u1 u2 : ℝ) : ℝ := u1 / u2

theorem runner_speed_ratio (u1 u2 : ℝ) (h1 : u1 > u2) (h2 : u1 + u2 = 5) (h3 : u1 - u2 = 5/3) :
  speed_ratio u1 u2 = 2 :=
by
  sorry

end runner_speed_ratio_l188_188851


namespace garden_area_difference_l188_188110

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l188_188110


namespace cubic_inches_in_one_cubic_foot_l188_188909

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l188_188909


namespace conic_sections_hyperbola_and_ellipse_l188_188586

theorem conic_sections_hyperbola_and_ellipse
  (x y : ℝ) (h : y^4 - 9 * x^4 = 3 * y^2 - 3) :
  (∃ a b c : ℝ, a * y^2 - b * x^2 = c ∧ a = b ∧ c ≠ 0) ∨ (∃ a b c : ℝ, a * y^2 + b * x^2 = c ∧ a ≠ b ∧ c ≠ 0) :=
by
  sorry

end conic_sections_hyperbola_and_ellipse_l188_188586


namespace ratio_implies_sum_ratio_l188_188590

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_implies_sum_ratio_l188_188590


namespace lines_parallel_if_perpendicular_to_same_plane_l188_188373

-- Define a plane as a placeholder for other properties
axiom Plane : Type
-- Define Line as a placeholder for other properties
axiom Line : Type

-- Definition of what it means for a line to be perpendicular to a plane
axiom perpendicular_to_plane (l : Line) (π : Plane) : Prop

-- Definition of parallel lines
axiom parallel_lines (l1 l2 : Line) : Prop

-- Define the proof problem in Lean 4
theorem lines_parallel_if_perpendicular_to_same_plane
    (π : Plane) (l1 l2 : Line)
    (h1 : perpendicular_to_plane l1 π)
    (h2 : perpendicular_to_plane l2 π) :
    parallel_lines l1 l2 :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l188_188373


namespace region_area_proof_l188_188875

noncomputable def region_area := 
  let region := {p : ℝ × ℝ | abs (p.1 - p.2^2 / 2) + p.1 + p.2^2 / 2 ≤ 2 - p.2}
  2 * (0.5 * (3 * (2 + 0.5)))

theorem region_area_proof : region_area = 15 / 2 :=
by
  sorry

end region_area_proof_l188_188875


namespace find_expression_value_l188_188573

-- Given conditions
variables {a b : ℝ}

-- Perimeter condition
def perimeter_condition (a b : ℝ) : Prop := 2 * (a + b) = 10

-- Area condition
def area_condition (a b : ℝ) : Prop := a * b = 6

-- Goal statement
theorem find_expression_value (h1 : perimeter_condition a b) (h2 : area_condition a b) :
  a^3 * b + 2 * a^2 * b^2 + a * b^3 = 150 :=
sorry

end find_expression_value_l188_188573


namespace factors_180_count_l188_188918

theorem factors_180_count : 
  ∃ (n : ℕ), 180 = 2^2 * 3^2 * 5^1 ∧ n = 18 ∧ 
  ∀ p a b c, 
  180 = p^a * p^b * p^c →
  (a+1) * (b+1) * (c+1) = 18 :=
by {
  sorry
}

end factors_180_count_l188_188918


namespace angle_between_unit_vectors_l188_188153

open Real

variables {a b : ℝ^3} (θ : ℝ)

-- Define unit vectors
def is_unit_vector (v : ℝ^3) := ∥v∥ = 1

-- Given conditions
variables (ha : is_unit_vector a) (hb : is_unit_vector b)
          (ha_b: ∥a - 2 • b∥ = √3)

-- The goal: prove the angle θ between a and b is π/3
theorem angle_between_unit_vectors : θ = π / 3 :=
  sorry

end angle_between_unit_vectors_l188_188153


namespace cow_manure_plant_height_l188_188791

theorem cow_manure_plant_height (control_height bone_meal_percentage cow_manure_percentage : ℝ)
  (control_height_eq : control_height = 36)
  (bone_meal_eq : bone_meal_percentage = 125)
  (cow_manure_eq : cow_manure_percentage = 200) :
  let bone_meal_height := (bone_meal_percentage / 100) * control_height in
  let cow_manure_height := (cow_manure_percentage / 100) * bone_meal_height in
  cow_manure_height = 90 := by
  sorry

end cow_manure_plant_height_l188_188791


namespace smallest_interval_for_probability_of_both_events_l188_188289

theorem smallest_interval_for_probability_of_both_events {C D : Prop} (hC : prob_C = 5 / 6) (hD : prob_D = 7 / 9) :
  ∃ I : set ℝ, I = set.Icc (11 / 18) (7 / 9) ∧ (∃ p : ℝ, p ∈ I ∧ p = prob_C_and_D) :=
begin
  sorry
end

end smallest_interval_for_probability_of_both_events_l188_188289


namespace recurrence_sequence_a5_l188_188546

theorem recurrence_sequence_a5 :
  ∃ a : ℕ → ℚ, (a 1 = 5 ∧ (∀ n, a (n + 1) = 1 + 1 / a n) ∧ a 5 = 28 / 17) :=
  sorry

end recurrence_sequence_a5_l188_188546


namespace train_pass_bridge_time_l188_188868

-- Given conditions
def train_length : ℕ := 460  -- length in meters
def bridge_length : ℕ := 140  -- length in meters
def speed_kmh : ℝ := 45  -- speed in kilometers per hour

-- Prove that the time to pass the bridge is 48 seconds
theorem train_pass_bridge_time :
  let distance := train_length + bridge_length in
  let speed_ms := speed_kmh * (1000 / 3600) in
  (distance / speed_ms) = 48 :=
by
  sorry

end train_pass_bridge_time_l188_188868


namespace initial_time_for_train_l188_188124

theorem initial_time_for_train (S : ℝ)
  (length_initial : ℝ := 12 * 15)
  (length_detached : ℝ := 11 * 15)
  (time_detached : ℝ := 16.5)
  (speed_constant : S = length_detached / time_detached) :
  (length_initial / S = 18) :=
by
  sorry

end initial_time_for_train_l188_188124


namespace james_sells_boxes_l188_188616

theorem james_sells_boxes (profit_per_candy_bar : ℝ) (total_profit : ℝ) 
                          (candy_bars_per_box : ℕ) (x : ℕ)
                          (h1 : profit_per_candy_bar = 1.5 - 1)
                          (h2 : total_profit = 25)
                          (h3 : candy_bars_per_box = 10) 
                          (h4 : total_profit = (x * candy_bars_per_box) * profit_per_candy_bar) :
                          x = 5 :=
by
  sorry

end james_sells_boxes_l188_188616


namespace geometric_sequence_common_ratio_l188_188472

theorem geometric_sequence_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2)
  (h2 : a 5 = 4)
  (h3 : ∀ n, a n = a 1 * q^(n - 1)) : 
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l188_188472


namespace one_cow_one_bag_l188_188525

-- Definitions based on the conditions provided.
def cows : ℕ := 45
def bags : ℕ := 45
def days : ℕ := 45

-- Problem statement: Prove that one cow will eat one bag of husk in 45 days.
theorem one_cow_one_bag (h : cows * bags = bags * days) : days = 45 :=
by
  sorry

end one_cow_one_bag_l188_188525


namespace arithmetic_seq_formula_sum_first_n_terms_l188_188327

/-- Define the given arithmetic sequence an -/
def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0       => a1
| (n + 1) => arithmetic_seq a1 d n + d

variable {a3 a7 : ℤ}
variable (a3_eq : arithmetic_seq 1 2 2 = 5)
variable (a7_eq : arithmetic_seq 1 2 6 = 13)

/-- Define the sequence bn -/
def b_seq (n : ℕ) : ℚ :=
  1 / ((2 * n + 1) * (arithmetic_seq 1 2 n))

/-- Define the sum of the first n terms of the sequence bn -/
def sum_b_seq : ℕ → ℚ
| 0       => 0
| (n + 1) => sum_b_seq n + b_seq (n + 1)
          
theorem arithmetic_seq_formula:
  ∀ (n : ℕ), arithmetic_seq 1 2 n = 2 * n - 1 :=
by
  intros
  sorry

theorem sum_first_n_terms:
  ∀ (n : ℕ), sum_b_seq n = n / (2 * n + 1) :=
by
  intros
  sorry

end arithmetic_seq_formula_sum_first_n_terms_l188_188327


namespace bill_left_with_money_l188_188132

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end bill_left_with_money_l188_188132


namespace f_is_32x5_l188_188801

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end f_is_32x5_l188_188801


namespace min_rice_weight_l188_188635

theorem min_rice_weight (o r : ℝ) (h1 : o ≥ 4 + 2 * r) (h2 : o ≤ 3 * r) : r ≥ 4 :=
sorry

end min_rice_weight_l188_188635


namespace compare_neg_fractions_l188_188880

theorem compare_neg_fractions : (- (3 : ℚ) / 5) > (- (3 : ℚ) / 4) := sorry

end compare_neg_fractions_l188_188880


namespace fruit_bowl_oranges_l188_188177

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end fruit_bowl_oranges_l188_188177


namespace JaneTotalEarningsIs138_l188_188334

structure FarmData where
  chickens : ℕ
  ducks : ℕ
  quails : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenPricePerDozen : ℕ
  duckPricePerDozen : ℕ
  quailPricePerDozen : ℕ

def JaneFarmData : FarmData := {
  chickens := 10,
  ducks := 8,
  quails := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenPricePerDozen := 2,
  duckPricePerDozen := 3,
  quailPricePerDozen := 4
}

def eggsLaid (f : FarmData) : ℕ × ℕ × ℕ :=
((f.chickens * f.chickenEggsPerWeek), 
 (f.ducks * f.duckEggsPerWeek), 
 (f.quails * f.quailEggsPerWeek))

def earningsForWeek1 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := duckEggs / 12
let quailDozens := (quailEggs / 12) / 2
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek2 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := (3 * duckEggs / 4) / 12
let quailDozens := quailEggs / 12
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek3 (f : FarmData) : ℕ :=
let (_, duckEggs, quailEggs) := eggsLaid f
let duckDozens := duckEggs / 12
let quailDozens := quailEggs / 12
(duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def totalEarnings (f : FarmData) : ℕ :=
earningsForWeek1 f + earningsForWeek2 f + earningsForWeek3 f

theorem JaneTotalEarningsIs138 : totalEarnings JaneFarmData = 138 := by
  sorry

end JaneTotalEarningsIs138_l188_188334


namespace valerie_needs_72_stamps_l188_188238

noncomputable def total_stamps_needed : ℕ :=
  let thank_you_cards := 5
  let stamps_per_thank_you := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebates_more_than_bills := 3
  let rebate_stamps := 2
  let job_applications_factor := 2
  let job_application_stamps := 1

  let total_thank_you_stamps := thank_you_cards * stamps_per_thank_you
  let total_bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let total_rebates := total_bill_stamps + rebates_more_than_bills
  let total_rebate_stamps := total_rebates * rebate_stamps
  let total_job_applications := total_rebates * job_applications_factor
  let total_job_application_stamps := total_job_applications * job_application_stamps

  total_thank_you_stamps + total_bill_stamps + total_rebate_stamps + total_job_application_stamps

theorem valerie_needs_72_stamps : total_stamps_needed = 72 :=
  by
    sorry

end valerie_needs_72_stamps_l188_188238


namespace problem_1_solution_set_problem_2_minimum_value_a_l188_188452

-- Define the function f with given a value
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1: Prove the solution set for f(x) > 5 when a = -2 is {x | x < -4/3 ∨ x > 2}
theorem problem_1_solution_set (x : ℝ) : f x (-2) > 5 ↔ x < -4 / 3 ∨ x > 2 :=
by
  sorry

-- Problem 2: Prove the minimum value of a ensures f(x) ≤ a * |x + 3| is 1/2
theorem problem_2_minimum_value_a : (∀ x : ℝ, f x a ≤ a * |x + 3| ∨ a ≥ 1/2) :=
by
  sorry

end problem_1_solution_set_problem_2_minimum_value_a_l188_188452


namespace slope_of_line_through_origin_and_center_l188_188527

def Point := (ℝ × ℝ)

def is_center (p : Point) : Prop :=
  p = (3, 1)

def is_dividing_line (l : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, l x = y → y / x = 1 / 3

theorem slope_of_line_through_origin_and_center :
  ∃ l : ℝ → ℝ, (∀ p1 p2 : Point,
  p1 = (0, 0) →
  p2 = (3, 1) →
  is_center p2 →
  is_dividing_line l) :=
sorry

end slope_of_line_through_origin_and_center_l188_188527


namespace divides_x_by_5_l188_188949

theorem divides_x_by_5 (x y : ℤ) (hx1 : 1 < x) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end divides_x_by_5_l188_188949


namespace expressions_for_c_and_d_l188_188456

variables {a b c d r s : ℝ}

-- Conditions of the problem
def first_quadratic (x : ℝ) := x^2 + a * x + b = 0
def second_quadratic (x : ℝ) := x^2 + c * x + d = 0
def roots_r_s : Prop := first_quadratic r ∧ first_quadratic s
def roots_r2_s2 : Prop := second_quadratic (r^2) ∧ second_quadratic (s^2)
def rs_eq_2b : Prop := r * s = 2 * b

-- Target to prove
theorem expressions_for_c_and_d (h_a_b_c_d : rs_eq_2b ∧ roots_r_s ∧ roots_r2_s2) : 
  c = -a^2 + 2 * b ∧ d = b^2 :=
sorry

end expressions_for_c_and_d_l188_188456


namespace smallest_number_when_diminished_by_7_is_divisible_l188_188988

-- Variables for divisors
def divisor1 : Nat := 12
def divisor2 : Nat := 16
def divisor3 : Nat := 18
def divisor4 : Nat := 21
def divisor5 : Nat := 28

-- The smallest number x which, when diminished by 7, is divisible by the divisors.
theorem smallest_number_when_diminished_by_7_is_divisible (x : Nat) : 
  (x - 7) % divisor1 = 0 ∧ 
  (x - 7) % divisor2 = 0 ∧ 
  (x - 7) % divisor3 = 0 ∧ 
  (x - 7) % divisor4 = 0 ∧ 
  (x - 7) % divisor5 = 0 → 
  x = 1015 := 
sorry

end smallest_number_when_diminished_by_7_is_divisible_l188_188988


namespace car_time_interval_l188_188510

-- Define the conditions
def road_length := 3 -- in miles
def total_time := 10 -- in hours
def number_of_cars := 30

-- Define the conversion factor and the problem to prove
def hours_to_minutes (hours: ℕ) : ℕ := hours * 60
def time_interval_per_car (total_time_minutes: ℕ) (number_of_cars: ℕ) : ℕ := total_time_minutes / number_of_cars

-- The Lean 4 statement for the proof problem
theorem car_time_interval :
  time_interval_per_car (hours_to_minutes total_time) number_of_cars = 20 :=
by
  sorry

end car_time_interval_l188_188510


namespace Peter_finishes_all_tasks_at_5_30_PM_l188_188357

-- Definitions representing the initial conditions
def start_time : ℕ := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
def task_durations : List ℕ :=
  [30, 30, 60, 120, 240] -- Durations of the 5 tasks in minutes
  
-- Statement for the proof problem
theorem Peter_finishes_all_tasks_at_5_30_PM :
  let total_duration := task_durations.sum 
  let finish_time := start_time + total_duration
  finish_time = 17 * 60 + 30 := -- 5:30 PM in minutes
  sorry

end Peter_finishes_all_tasks_at_5_30_PM_l188_188357
