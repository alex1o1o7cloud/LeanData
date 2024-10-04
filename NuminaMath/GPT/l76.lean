import Mathlib

namespace ravi_prakash_finish_together_l76_76347

theorem ravi_prakash_finish_together (ravi_days prakash_days : ℕ) (h_ravi : ravi_days = 15) (h_prakash : prakash_days = 30) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 10 := 
by
  sorry

end ravi_prakash_finish_together_l76_76347


namespace part1_part2_l76_76407

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 9 → x ∈ Set.Icc (-2 : ℝ) 4 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) →
  (a ∈ Set.Icc (19 / 4) (7 : ℝ)) :=
sorry

end part1_part2_l76_76407


namespace no_infinite_set_exists_l76_76385

variable {S : Set ℕ} -- We assume S is a set of natural numbers

def satisfies_divisibility_condition (a b : ℕ) : Prop :=
  (a^2 + b^2 - a * b) ∣ (a * b)^2

theorem no_infinite_set_exists (h1 : Infinite S)
  (h2 : ∀ (a b : ℕ), a ∈ S → b ∈ S → satisfies_divisibility_condition a b) : false :=
  sorry

end no_infinite_set_exists_l76_76385


namespace total_digits_written_total_digit_1_appearances_digit_at_position_2016_l76_76632

-- Problem 1
theorem total_digits_written : 
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  digits_1_to_9 + digits_10_to_99 + digits_100_to_999 = 2889 := 
by
  sorry

-- Problem 2
theorem total_digit_1_appearances : 
  let digit_1_as_1_digit := 1
  let digit_1_as_2_digits := 10 + 9
  let digit_1_as_3_digits := 100 + 9 * 10 + 9 * 10
  digit_1_as_1_digit + digit_1_as_2_digits + digit_1_as_3_digits = 300 := 
by
  sorry

-- Problem 3
theorem digit_at_position_2016 : 
  let position_1_to_99 := 9 + 90 * 2
  let remaining_positions := 2016 - position_1_to_99
  let three_digit_positions := remaining_positions / 3
  let specific_number := 100 + three_digit_positions - 1
  specific_number % 10 = 8 := 
by
  sorry

end total_digits_written_total_digit_1_appearances_digit_at_position_2016_l76_76632


namespace lcm_first_ten_numbers_l76_76913

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76913


namespace range_of_a_l76_76538

noncomputable def f (a x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (f a x) > 0) ↔ a ≤ 2 := 
by
  sorry

end range_of_a_l76_76538


namespace price_of_water_margin_comics_l76_76366

-- Define the conditions
variables (x : ℕ) (y : ℕ)

-- Condition 1: Price relationship
def price_relationship : Prop := y = x + 60

-- Condition 2: Total expenditure on Romance of the Three Kingdoms comic books
def total_expenditure_romance_three_kingdoms : Prop := 60 * (y / 60) = 3600

-- Condition 3: Total expenditure on Water Margin comic books
def total_expenditure_water_margin : Prop := 120 * (x / 120) = 4800

-- Condition 4: Number of sets relationship
def number_of_sets_relationship : Prop := y = (4800 / x) / 2

-- The main statement to prove
theorem price_of_water_margin_comics (x : ℕ) (h1: price_relationship x (x + 60))
  (h2: total_expenditure_romance_three_kingdoms x)
  (h3: total_expenditure_water_margin x)
  (h4: number_of_sets_relationship x (x + 60)) : x = 120 :=
sorry

end price_of_water_margin_comics_l76_76366


namespace least_divisible_1_to_10_l76_76802

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76802


namespace current_price_after_adjustment_l76_76967

variable (x : ℝ) -- Define x, the original price per unit

theorem current_price_after_adjustment (x : ℝ) : (x + 10) * 0.75 = ((x + 10) * 0.75) :=
by
  sorry

end current_price_after_adjustment_l76_76967


namespace maxAdditionalTiles_l76_76599

-- Board definition
structure Board where
  width : Nat
  height : Nat
  cells : List (Nat × Nat) -- List of cells occupied by tiles

def initialBoard : Board := 
  ⟨10, 9, [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2),
            (6,1), (6,2), (7,1), (7,2)]⟩

-- Function to count cells occupied
def occupiedCells (b : Board) : Nat :=
  b.cells.length

-- Function to calculate total cells in a board
def totalCells (b : Board) : Nat :=
  b.width * b.height

-- Function to calculate additional 2x1 tiles that can be placed
def additionalTiles (board : Board) : Nat :=
  (totalCells board - occupiedCells board) / 2

theorem maxAdditionalTiles : additionalTiles initialBoard = 36 := by
  sorry

end maxAdditionalTiles_l76_76599


namespace compare_a_b_c_l76_76255

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l76_76255


namespace jasper_drinks_more_than_hot_dogs_l76_76292

-- Definition of conditions based on the problem
def bags_of_chips := 27
def fewer_hot_dogs_than_chips := 8
def drinks_sold := 31

-- Definition to compute the number of hot dogs
def hot_dogs_sold := bags_of_chips - fewer_hot_dogs_than_chips

-- Lean 4 statement to prove the final result
theorem jasper_drinks_more_than_hot_dogs : drinks_sold - hot_dogs_sold = 12 :=
by
  -- skipping the proof
  sorry

end jasper_drinks_more_than_hot_dogs_l76_76292


namespace not_possible_to_cut_rectangular_paper_l76_76626

theorem not_possible_to_cut_rectangular_paper 
  (a : ℝ) (b : ℝ) (s_area : ℝ) (r_area : ℝ) (ratio : ℝ)
  (h1 : s_area = 100) (h2 : r_area = 90) (h3 : ratio = 5 / 3) :
  ¬ (∃ l w, l / w = ratio ∧ l * w = r_area ∧ l ≤ sqrt s_area ∧ w ≤ sqrt s_area) := by
  sorry

end not_possible_to_cut_rectangular_paper_l76_76626


namespace find_ax5_by5_l76_76298

theorem find_ax5_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 :=
sorry

end find_ax5_by5_l76_76298


namespace unique_solution_iff_d_ne_4_l76_76237

theorem unique_solution_iff_d_ne_4 (c d : ℝ) : 
  (∃! (x : ℝ), 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := 
by 
  sorry

end unique_solution_iff_d_ne_4_l76_76237


namespace find_element_in_A_l76_76702

def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

def f (p : A) : B := (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem find_element_in_A : ∃ p : A, f p = (3, 1) ∧ p = (1, 1) := by
  sorry

end find_element_in_A_l76_76702


namespace possible_triangular_frames_B_l76_76335

-- Define the sides of the triangles and the similarity condition
def similar_triangles (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * b₃ = a₃ * b₁ ∧ a₂ * b₃ = a₃ * b₂

def sides_of_triangle_A := (50, 60, 80)

def is_a_possible_triangle (b₁ b₂ b₃ : ℕ) : Prop :=
  similar_triangles 50 60 80 b₁ b₂ b₃

-- Given conditions
def side_of_triangle_B := 20

-- Theorem to prove
theorem possible_triangular_frames_B :
  ∃ (b₂ b₃ : ℕ), (is_a_possible_triangle 20 b₂ b₃ ∨ is_a_possible_triangle b₂ 20 b₃ ∨ is_a_possible_triangle b₂ b₃ 20) :=
sorry

end possible_triangular_frames_B_l76_76335


namespace estimate_expr_l76_76386

theorem estimate_expr : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end estimate_expr_l76_76386


namespace problem_r_minus_s_l76_76144

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l76_76144


namespace least_divisible_1_to_10_l76_76799

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76799


namespace last_two_digits_of_1032_pow_1032_l76_76090

noncomputable def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_digits_of_1032_pow_1032 : last_two_digits (1032^1032) = 76 := by
  sorry

end last_two_digits_of_1032_pow_1032_l76_76090


namespace amount_of_solution_added_l76_76637

variable (x : ℝ)

-- Condition: The solution contains 90% alcohol
def solution_alcohol_amount (x : ℝ) : ℝ := 0.9 * x

-- Condition: Total volume of the new mixture after adding 16 liters of water
def total_volume (x : ℝ) : ℝ := x + 16

-- Condition: The percentage of alcohol in the new mixture is 54%
def new_mixture_alcohol_amount (x : ℝ) : ℝ := 0.54 * (total_volume x)

-- The proof goal: the amount of solution added is 24 liters
theorem amount_of_solution_added : new_mixture_alcohol_amount x = solution_alcohol_amount x → x = 24 :=
by
  sorry

end amount_of_solution_added_l76_76637


namespace f_eq_four_or_seven_l76_76437

noncomputable def f (a b : ℕ) : ℚ := (a^2 + a * b + b^2) / (a * b - 1)

theorem f_eq_four_or_seven (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : a * b ≠ 1) : 
  f a b = 4 ∨ f a b = 7 := 
sorry

end f_eq_four_or_seven_l76_76437


namespace complex_expression_l76_76327

theorem complex_expression (i : ℂ) (h : i^2 = -1) : ( (1 + i) / (1 - i) )^2006 = -1 :=
by {
  sorry
}

end complex_expression_l76_76327


namespace compute_a2004_l76_76114

def recurrence_sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 0
  else sorry -- We'll define recurrence operations in the proofs

theorem compute_a2004 : recurrence_sequence 2004 = -2^1002 := 
sorry -- Proof omitted

end compute_a2004_l76_76114


namespace workouts_difference_l76_76714

theorem workouts_difference
  (workouts_monday : ℕ := 8)
  (workouts_tuesday : ℕ := 5)
  (workouts_wednesday : ℕ := 12)
  (workouts_thursday : ℕ := 17)
  (workouts_friday : ℕ := 10) :
  workouts_thursday - workouts_tuesday = 12 := 
by
  sorry

end workouts_difference_l76_76714


namespace fair_total_revenue_l76_76216

noncomputable def price_per_ticket : ℝ := 8
noncomputable def total_ticket_revenue : ℝ := 8000
noncomputable def total_tickets_sold : ℝ := total_ticket_revenue / price_per_ticket

noncomputable def food_revenue : ℝ := (3/5) * total_tickets_sold * 10
noncomputable def rounded_ride_revenue : ℝ := (333 : ℝ) * 6
noncomputable def ride_revenue : ℝ := rounded_ride_revenue
noncomputable def rounded_souvenir_revenue : ℝ := (166 : ℝ) * 18
noncomputable def souvenir_revenue : ℝ := rounded_souvenir_revenue
noncomputable def game_revenue : ℝ := (1/10) * total_tickets_sold * 5

noncomputable def total_additional_revenue : ℝ := food_revenue + ride_revenue + souvenir_revenue + game_revenue
noncomputable def total_revenue : ℝ := total_ticket_revenue + total_additional_revenue

theorem fair_total_revenue : total_revenue = 19486 := by
  sorry

end fair_total_revenue_l76_76216


namespace fourth_vertex_exists_l76_76404

structure Point :=
  (x : ℚ)
  (y : ℚ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  let M_AC := Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2)
  let M_BD := Point.mk ((B.x + D.x) / 2) ((B.y + D.y) / 2)
  is_midpoint M_AC A C ∧ is_midpoint M_BD B D ∧ M_AC = M_BD

theorem fourth_vertex_exists (A B C : Point) (hA : A = ⟨-1, 0⟩) (hB : B = ⟨3, 0⟩) (hC : C = ⟨1, -5⟩) :
  ∃ D : Point, (D = ⟨1, 5⟩ ∨ D = ⟨-3, -5⟩) ∧ is_parallelogram A B C D :=
by
  sorry

end fourth_vertex_exists_l76_76404


namespace basketball_weight_l76_76253

theorem basketball_weight (b s : ℝ) (h1 : s = 20) (h2 : 5 * b = 4 * s) : b = 16 :=
by
  sorry

end basketball_weight_l76_76253


namespace least_divisible_1_to_10_l76_76797

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76797


namespace jumpy_implies_not_green_l76_76457

variables (Lizard : Type)
variables (IsJumpy IsGreen CanSing CanDance : Lizard → Prop)

-- Conditions given in the problem
axiom jumpy_implies_can_sing : ∀ l, IsJumpy l → CanSing l
axiom green_implies_cannot_dance : ∀ l, IsGreen l → ¬ CanDance l
axiom cannot_dance_implies_cannot_sing : ∀ l, ¬ CanDance l → ¬ CanSing l

theorem jumpy_implies_not_green (l : Lizard) : IsJumpy l → ¬ IsGreen l :=
by
  sorry

end jumpy_implies_not_green_l76_76457


namespace sequence_k_value_l76_76693

theorem sequence_k_value {k : ℕ} (h : 9 < (2 * k - 8) ∧ (2 * k - 8) < 12) 
  (Sn : ℕ → ℤ) (hSn : ∀ n, Sn n = n^2 - 7*n) 
  : k = 9 :=
by
  sorry

end sequence_k_value_l76_76693


namespace lcm_first_ten_numbers_l76_76908

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76908


namespace least_common_multiple_first_ten_integers_l76_76942

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76942


namespace lcm_first_ten_integers_l76_76809

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76809


namespace sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l76_76297

-- Define the real interval [0, π/2]
def interval_0_pi_over_2 (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 2

-- Define the proposition to be proven
theorem sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq (a b : ℝ) 
  (ha : interval_0_pi_over_2 a) (hb : interval_0_pi_over_2 b) :
  (Real.sin a)^6 + 3 * (Real.sin a)^2 * (Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b :=
by
  sorry

end sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l76_76297


namespace least_common_multiple_1_to_10_l76_76895

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76895


namespace tensor_calculation_jiaqi_statement_l76_76082

def my_tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem tensor_calculation :
  my_tensor (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := 
by
  sorry

theorem jiaqi_statement (a b : ℝ) (h : a + b = 0) :
  my_tensor a a + my_tensor b b = 2 * a * b := 
by
  sorry

end tensor_calculation_jiaqi_statement_l76_76082


namespace square_of_real_is_positive_or_zero_l76_76059

def p (x : ℝ) : Prop := x^2 > 0
def q (x : ℝ) : Prop := x^2 = 0

theorem square_of_real_is_positive_or_zero (x : ℝ) : (p x ∨ q x) :=
by
  sorry

end square_of_real_is_positive_or_zero_l76_76059


namespace real_number_unique_l76_76425

variable (a x : ℝ)

theorem real_number_unique (h1 : (a + 3) * (a + 3) = x)
  (h2 : (2 * a - 9) * (2 * a - 9) = x) : x = 25 := by
  sorry

end real_number_unique_l76_76425


namespace correct_propositions_l76_76548

noncomputable def f : ℝ → ℝ := sorry

def proposition1 : Prop :=
  ∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x) → ∀ x : ℝ, f (2 - x) = f x

def proposition2 : Prop :=
  ∀ x : ℝ, f (x - 2) = f (2 - x)

def proposition3 : Prop :=
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (2 + x) = -f x) → ∀ x : ℝ, f x = f (4 - x)

def proposition4 : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f x = f (-x - 2)) → ∀ x : ℝ, f (2 - x) = f x

theorem correct_propositions : proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l76_76548


namespace linear_equation_a_neg2_l76_76720

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l76_76720


namespace room_length_l76_76027

/-- Define the conditions -/
def width : ℝ := 3.75
def cost_paving : ℝ := 6187.5
def cost_per_sqm : ℝ := 300

/-- Prove that the length of the room is 5.5 meters -/
theorem room_length : 
  (cost_paving / cost_per_sqm) / width = 5.5 :=
by
  sorry

end room_length_l76_76027


namespace buses_needed_40_buses_needed_30_l76_76490

-- Define the number of students
def number_of_students : ℕ := 186

-- Define the function to calculate minimum buses needed
def min_buses_needed (n : ℕ) : ℕ := (number_of_students + n - 1) / n

-- Theorem statements for the specific cases
theorem buses_needed_40 : min_buses_needed 40 = 5 := 
by 
  sorry

theorem buses_needed_30 : min_buses_needed 30 = 7 := 
by 
  sorry

end buses_needed_40_buses_needed_30_l76_76490


namespace mod_equiv_1043_36_mod_equiv_1_10_l76_76380

open Int

-- Define the integers involved
def a : ℤ := -1043
def m1 : ℕ := 36
def m2 : ℕ := 10

-- Theorems to prove modulo equivalence
theorem mod_equiv_1043_36 : a % m1 = 1 := by
  sorry

theorem mod_equiv_1_10 : 1 % m2 = 1 := by
  sorry

end mod_equiv_1043_36_mod_equiv_1_10_l76_76380


namespace factorization_of_expression_l76_76469

open Polynomial

theorem factorization_of_expression (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^3 * b + a^3 * c + a^2 * b^2 + a^2 * b * c + a^2 * c^2 + a * b^3 + a * b * c^2 + a * c^3 + b^3 * c + b^2 * c^2 + b * c^3) := by
  sorry

end factorization_of_expression_l76_76469


namespace sum_of_x_values_proof_l76_76189

noncomputable def sum_of_x_values : ℝ := 
  (-(-4)) / 1 -- Sum of roots of x^2 - 4x - 7 = 0

theorem sum_of_x_values_proof (x : ℝ) (h : 7 = (x^3 - 2 * x^2 - 8 * x) / (x + 2)) : sum_of_x_values = 4 :=
sorry

end sum_of_x_values_proof_l76_76189


namespace find_larger_number_l76_76037

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l76_76037


namespace arithmetic_sum_l76_76424

variables {a d : ℝ}

theorem arithmetic_sum (h : 15 * a + 105 * d = 90) : 2 * a + 14 * d = 12 :=
sorry

end arithmetic_sum_l76_76424


namespace total_junk_mail_l76_76648

-- Definitions for conditions
def houses_per_block : Nat := 17
def pieces_per_house : Nat := 4
def blocks : Nat := 16

-- Theorem stating that the mailman gives out 1088 pieces of junk mail in total
theorem total_junk_mail : houses_per_block * pieces_per_house * blocks = 1088 := by
  sorry

end total_junk_mail_l76_76648


namespace geometric_sequence_and_general_formula_l76_76270

theorem geometric_sequence_and_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+1) = (2/3) * a n + 2) (ha1 : a 1 = 7) : 
  ∃ r : ℝ, ∀ n, a n = r ^ (n-1) + 6 :=
sorry

end geometric_sequence_and_general_formula_l76_76270


namespace carnations_count_l76_76652

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end carnations_count_l76_76652


namespace sum_of_possible_values_for_a_l76_76014

-- Define the conditions
variables (a b c d : ℤ)
variables (h1 : a > b) (h2 : b > c) (h3 : c > d)
variables (h4 : a + b + c + d = 52)
variables (differences : finset ℤ)

-- Hypotheses about the pairwise differences
variable (h_diff : differences = {2, 3, 5, 6, 8, 11})
variable (h_ad : a - d = 11)

-- The pairs of differences adding up to 11
variable (h_pairs1 : a - b + b - d = 11)
variable (h_pairs2 : a - c + c - d = 11)

-- The theorem to be proved
theorem sum_of_possible_values_for_a : a = 19 :=
by
-- Implemented variables and conditions correctly, and the proof is outlined.
sorry

end sum_of_possible_values_for_a_l76_76014


namespace solution_to_trig_equation_l76_76602

theorem solution_to_trig_equation (x : ℝ) (k : ℤ) :
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
  (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) →
  (Real.sin (x / 2) = Real.cos (x / 2)) →
  (∃ k : ℤ, x = (π / 2) + 2 * π * ↑k) :=
by sorry

end solution_to_trig_equation_l76_76602


namespace k_inverse_k_inv_is_inverse_l76_76444

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x - 4
def k (x : ℝ) : ℝ := f (g x)

def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem k_inverse {x : ℝ} : k_inv (k x) = x :=
by
  sorry

theorem k_inv_is_inverse {x y : ℝ} : k_inv (y) = x ↔ y = k(x) :=
by
  sorry

end k_inverse_k_inv_is_inverse_l76_76444


namespace max_truthful_students_l76_76165

def count_students (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem max_truthful_students : count_students 2015 = 2031120 :=
by sorry

end max_truthful_students_l76_76165


namespace meals_distinct_pairs_l76_76991

theorem meals_distinct_pairs :
  let entrees := 4
  let drinks := 3
  let desserts := 3
  let total_meals := entrees * drinks * desserts
  total_meals * (total_meals - 1) = 1260 :=
by 
  sorry

end meals_distinct_pairs_l76_76991


namespace average_marks_l76_76960

noncomputable def TatuyaScore (IvannaScore : ℝ) : ℝ :=
2 * IvannaScore

noncomputable def IvannaScore (DorothyScore : ℝ) : ℝ :=
(3/5) * DorothyScore

noncomputable def DorothyScore : ℝ := 90

noncomputable def XanderScore (TatuyaScore IvannaScore DorothyScore : ℝ) : ℝ :=
((TatuyaScore + IvannaScore + DorothyScore) / 3) + 10

noncomputable def SamScore (IvannaScore : ℝ) : ℝ :=
(3.8 * IvannaScore) + 5.5

noncomputable def OliviaScore (SamScore : ℝ) : ℝ :=
(3/2) * SamScore

theorem average_marks :
  let I := IvannaScore DorothyScore
  let T := TatuyaScore I
  let S := SamScore I
  let O := OliviaScore S
  let X := XanderScore T I DorothyScore
  let total_marks := T + I + DorothyScore + X + O + S
  (total_marks / 6) = 145.458333 := by sorry

end average_marks_l76_76960


namespace bowling_ball_weight_l76_76242

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end bowling_ball_weight_l76_76242


namespace amber_max_ounces_l76_76661

-- Define the problem parameters:
def cost_candy : ℝ := 1
def ounces_candy : ℝ := 12
def cost_chips : ℝ := 1.4
def ounces_chips : ℝ := 17
def total_money : ℝ := 7

-- Define the number of bags of each item Amber can buy:
noncomputable def bags_candy := (total_money / cost_candy).to_int
noncomputable def bags_chips  := (total_money / cost_chips).to_int

-- Define the total ounces of each item:
noncomputable def total_ounces_candy := bags_candy * ounces_candy
noncomputable def total_ounces_chips := bags_chips * ounces_chips

-- Problem statement asking to prove Amber gets the most ounces by buying chips:
theorem amber_max_ounces : max total_ounces_candy total_ounces_chips = total_ounces_chips :=
by sorry

end amber_max_ounces_l76_76661


namespace carnations_count_l76_76653

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end carnations_count_l76_76653


namespace total_signs_at_intersections_l76_76212

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end total_signs_at_intersections_l76_76212


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l76_76008

def is_sum_of_squares_of_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

def T (x : ℤ) : Prop :=
  ∃ n : ℤ, x = is_sum_of_squares_of_consecutive_integers n

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_5 :
  (∀ x, T x → ¬ (9 ∣ x)) ∧ (∃ y, T y ∧ (5 ∣ y)) :=
by
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l76_76008


namespace cylinder_surface_area_l76_76700

noncomputable def surface_area_of_cylinder (r l : ℝ) : ℝ :=
  2 * Real.pi * r * (r + l)

theorem cylinder_surface_area (r : ℝ) (h_radius : r = 1) (l : ℝ) (h_length : l = 2 * r) :
  surface_area_of_cylinder r l = 6 * Real.pi := by
  -- Using the given conditions and definition, we need to prove the surface area is 6π
  sorry

end cylinder_surface_area_l76_76700


namespace triangles_same_perimeter_l76_76058

theorem triangles_same_perimeter
    (S₁ S₂ : Circle)
    (P Q : Point)
    (h_intersect : intersects_disjoint S₁ S₂ P Q)
    (ℓ₁ ℓ₂ : Line)
    (h_parallel : parallel ℓ₁ ℓ₂)
    (A₁ A₂ B₁ B₂ : Point)
    (h_ℓ₁_through_P : passes_through ℓ₁ P)
    (h_ℓ₁_inter_S₁ : intersects_at ℓ₁ S₁ A₁)
    (h_ℓ₁_inter_S₂ : intersects_at ℓ₁ S₂ A₂)
    (h_A₁_ne_P : A₁ ≠ P)
    (h_A₂_ne_P : A₂ ≠ P)
    (h_ℓ₂_through_Q : passes_through ℓ₂ Q)
    (h_ℓ₂_inter_S₁ : intersects_at ℓ₂ S₁ B₁)
    (h_ℓ₂_inter_S₂ : intersects_at ℓ₂ S₂ B₂)
    (h_B₁_ne_Q : B₁ ≠ Q)
    (h_B₂_ne_Q : B₂ ≠ Q) :
    perimeter (Triangle.mk A₁ Q A₂) = perimeter (Triangle.mk B₁ P B₂) := 
sorry

end triangles_same_perimeter_l76_76058


namespace relation_between_A_and_B_l76_76541

-- Define the sets A and B
def A : Set ℤ := { x | ∃ k : ℕ, x = 7 * k + 3 }
def B : Set ℤ := { x | ∃ k : ℤ, x = 7 * k - 4 }

-- Prove the relationship between A and B
theorem relation_between_A_and_B : A ⊆ B :=
by
  sorry

end relation_between_A_and_B_l76_76541


namespace max_partial_sum_l76_76740

variable (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence and the conditions given
def arithmetic_sequence (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a_n n = a_1 + n * d

def condition1 (a_1 : ℤ) : Prop := a_1 > 0

def condition2 (a_n : ℕ → ℤ) (d : ℤ) : Prop := 3 * (a_n 8) = 5 * (a_n 13)

-- Define the partial sum of the arithmetic sequence
def partial_sum (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

-- Define the main problem: Prove that S_20 is the greatest
theorem max_partial_sum (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) (S : ℕ → ℤ) :
  arithmetic_sequence a_n a_1 d →
  condition1 a_1 →
  condition2 a_n d →
  partial_sum S a_n →
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → S 20 ≥ S n := by
  sorry

end max_partial_sum_l76_76740


namespace alloy_copper_percentage_l76_76515

theorem alloy_copper_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x)
  (h2 : (30 / 100) * x + (70 / 100) * 27 = 24.9) :
  x = 20 :=
sorry

end alloy_copper_percentage_l76_76515


namespace part_a_part_b_l76_76195

-- Part (a): Prove that \( 2^n - 1 \) is divisible by 7 if and only if \( 3 \mid n \).
theorem part_a (n : ℕ) : 7 ∣ (2^n - 1) ↔ 3 ∣ n := sorry

-- Part (b): Prove that \( 2^n + 1 \) is not divisible by 7 for all natural numbers \( n \).
theorem part_b (n : ℕ) : ¬ (7 ∣ (2^n + 1)) := sorry

end part_a_part_b_l76_76195


namespace least_divisible_1_to_10_l76_76800

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76800


namespace find_k_l76_76736

theorem find_k (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 ↔ |k * x - 4| ≤ 2) → k = 2 :=
by
  sorry

end find_k_l76_76736


namespace terms_before_one_l76_76715

-- Define the sequence parameters
def a : ℤ := 100
def d : ℤ := -7
def nth_term (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the target term we are interested in
def target_term : ℤ := 1

-- Define the main theorem
theorem terms_before_one : ∃ n : ℕ, nth_term n = target_term ∧ (n - 1) = 14 := by
  sorry

end terms_before_one_l76_76715


namespace least_common_multiple_of_first_ten_positive_integers_l76_76865

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76865


namespace pond_diameter_l76_76688

theorem pond_diameter 
  (h k r : ℝ)
  (H1 : (4 - h) ^ 2 + (11 - k) ^ 2 = r ^ 2)
  (H2 : (12 - h) ^ 2 + (9 - k) ^ 2 = r ^ 2)
  (H3 : (2 - h) ^ 2 + (7 - k) ^ 2 = (r - 1) ^ 2) :
  2 * r = 9.2 :=
sorry

end pond_diameter_l76_76688


namespace jasmine_pies_l76_76136

-- Definitions based on the given conditions
def total_pies : Nat := 30
def raspberry_part : Nat := 2
def peach_part : Nat := 5
def plum_part : Nat := 3
def total_parts : Nat := raspberry_part + peach_part + plum_part

-- Calculate pies per part
def pies_per_part : Nat := total_pies / total_parts

-- Prove the statement
theorem jasmine_pies :
  (plum_part * pies_per_part = 9) :=
by
  -- The statement and proof will go here, but we are skipping the proof part.
  sorry

end jasmine_pies_l76_76136


namespace Tony_change_l76_76614

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end Tony_change_l76_76614


namespace smallest_whole_number_greater_than_triangle_perimeter_l76_76188

theorem smallest_whole_number_greater_than_triangle_perimeter 
  (a b : ℝ) (h_a : a = 7) (h_b : b = 23) :
  ∀ c : ℝ, 16 < c ∧ c < 30 → ⌈a + b + c⌉ = 60 :=
by
  intros c h
  rw [h_a, h_b]
  sorry

end smallest_whole_number_greater_than_triangle_perimeter_l76_76188


namespace cats_kittentotal_l76_76293

def kittens_given_away : ℕ := 2
def kittens_now : ℕ := 6
def kittens_original : ℕ := 8

theorem cats_kittentotal : kittens_now + kittens_given_away = kittens_original := 
by 
  sorry

end cats_kittentotal_l76_76293


namespace least_common_multiple_of_first_ten_integers_l76_76931

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76931


namespace price_of_each_armchair_l76_76745

theorem price_of_each_armchair
  (sofa_price : ℕ)
  (coffee_table_price : ℕ)
  (total_invoice : ℕ)
  (num_armchairs : ℕ)
  (h_sofa : sofa_price = 1250)
  (h_coffee_table : coffee_table_price = 330)
  (h_invoice : total_invoice = 2430)
  (h_num_armchairs : num_armchairs = 2) :
  (total_invoice - (sofa_price + coffee_table_price)) / num_armchairs = 425 := 
by 
  sorry

end price_of_each_armchair_l76_76745


namespace solve_fraction_eq_zero_l76_76092

theorem solve_fraction_eq_zero (a : ℝ) (h : a ≠ -1) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by {
  sorry
}

end solve_fraction_eq_zero_l76_76092


namespace smallest_prime_with_composite_reverse_l76_76391

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m : Nat, m > 1 ∧ m < n ∧ n % m = 0

def reverse_digits (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem smallest_prime_with_composite_reverse :
  ∃ (n : Nat), 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (n / 10 = 3) ∧ is_composite (reverse_digits n) ∧
  (∀ m : Nat, 10 ≤ m ∧ m < n ∧ (m / 10 = 3) ∧ is_prime m → ¬is_composite (reverse_digits m)) :=
by
  sorry

end smallest_prime_with_composite_reverse_l76_76391


namespace painting_rate_l76_76782

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end painting_rate_l76_76782


namespace quadratic_binomial_form_l76_76322

theorem quadratic_binomial_form (y : ℝ) : ∃ (k : ℝ), y^2 + 14 * y + 40 = (y + 7)^2 + k :=
by
  use -9
  sorry

end quadratic_binomial_form_l76_76322


namespace circle_equation_through_points_l76_76785

-- Definitions of the points A, B, and C
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (-1, 1)

-- Prove that the equation of the circle passing through A, B, and C is (x - 1)^2 + y^2 = 5
theorem circle_equation_through_points :
  ∃ (D E F : ℝ), (∀ x y : ℝ, 
  x^2 + y^2 + D * x + E * y + F = 0 ↔
  x = -1 ∧ y = -1 ∨ 
  x = 2 ∧ y = 2 ∨ 
  x = -1 ∧ y = 1) ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x - 1)^2 + y^2 = 5 :=
by
  sorry

end circle_equation_through_points_l76_76785


namespace pedal_triangle_angle_pedal_triangle_angle_equality_l76_76134

variables {A B C T_A T_B T_C: Type*}
variables {α β γ : Real}
variables {triangle : ∀ (A B C : Type*) (α β γ : Real), α ≤ β ∧ β ≤ γ ∧ γ < 90}

theorem pedal_triangle_angle
  (h : α ≤ β ∧ β ≤ γ ∧ γ < 90)
  (angles : 180 - 2 * α ≥ γ) :
  true :=
sorry

theorem pedal_triangle_angle_equality
  (h : α = β)
  (angles : (45 < α ∧ α = β ∧ α ≤ 60) ∧ (60 ≤ γ ∧ γ < 90)) :
  true :=
sorry

end pedal_triangle_angle_pedal_triangle_angle_equality_l76_76134


namespace abc_value_l76_76698

variables (a b c d e f : ℝ)
variables (h1 : b * c * d = 65)
variables (h2 : c * d * e = 750)
variables (h3 : d * e * f = 250)
variables (h4 : (a * f) / (c * d) = 0.6666666666666666)

theorem abc_value : a * b * c = 130 :=
by { sorry }

end abc_value_l76_76698


namespace least_common_multiple_of_first_ten_l76_76920

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76920


namespace isabella_hair_length_l76_76135

-- Define the conditions and the question in Lean
def current_length : ℕ := 9
def length_cut_off : ℕ := 9

-- Main theorem statement
theorem isabella_hair_length 
  (current_length : ℕ) 
  (length_cut_off : ℕ) 
  (H1 : current_length = 9) 
  (H2 : length_cut_off = 9) : 
  current_length + length_cut_off = 18 :=
  sorry

end isabella_hair_length_l76_76135


namespace train_pass_man_time_l76_76508

/--
Prove that the train, moving at 120 kmph, passes a man running at 10 kmph in the opposite direction in approximately 13.85 seconds, given the train is 500 meters long.
-/
theorem train_pass_man_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) : 
  length_of_train = 500 →
  speed_of_train = 120 →
  speed_of_man = 10 →
  abs ((500 / ((speed_of_train + speed_of_man) * 1000 / 3600)) - 13.85) < 0.01 :=
by
  intro h1 h2 h3
  -- This is where the proof would go
  sorry

end train_pass_man_time_l76_76508


namespace intersection_correct_l76_76712

noncomputable def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def intersection_M_N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_correct : M ∩ N = intersection_M_N :=
by
  sorry

end intersection_correct_l76_76712


namespace find_integers_l76_76552

theorem find_integers (x y : ℕ) (d : ℕ) (x1 y1 : ℕ) 
  (hx1 : x = d * x1) (hy1 : y = d * y1)
  (hgcd : Nat.gcd x y = d)
  (hcoprime : Nat.gcd x1 y1 = 1)
  (h1 : x1 + y1 = 18)
  (h2 : d * x1 * y1 = 975) : 
  ∃ (x y : ℕ), (Nat.gcd x y > 0) ∧ (x / Nat.gcd x y + y / Nat.gcd x y = 18) ∧ (Nat.lcm x y = 975) :=
sorry

end find_integers_l76_76552


namespace least_common_multiple_1_to_10_l76_76905

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76905


namespace inequality_problem_l76_76481

theorem inequality_problem (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by 
  sorry

end inequality_problem_l76_76481


namespace least_common_multiple_of_first_ten_integers_l76_76926

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76926


namespace complementary_event_probability_l76_76100

-- Define A and B as events such that B is the complement of A.
section
variables (A B : Prop) -- A and B are propositions representing events.
variable (P : Prop → ℝ) -- P is a function that gives the probability of an event.

-- Define the conditions for the problem.
variable (h_complementary : ∀ A B, A ∧ B = false ∧ A ∨ B = true) 
variable (h_PA : P A = 1 / 5)

-- The statement to be proved.
theorem complementary_event_probability : P B = 4 / 5 :=
by
  -- Here we would provide the proof, but for now, we use 'sorry' to bypass it.
  sorry
end

end complementary_event_probability_l76_76100


namespace root_expr_calculation_l76_76671

theorem root_expr_calculation : (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := 
by 
  sorry

end root_expr_calculation_l76_76671


namespace survey_no_preference_students_l76_76500

theorem survey_no_preference_students (total_students pref_mac pref_both pref_windows : ℕ) 
    (h1 : total_students = 210) 
    (h2 : pref_mac = 60) 
    (h3 : pref_both = pref_mac / 3)
    (h4 : pref_windows = 40) : 
    total_students - (pref_mac + pref_both + pref_windows) = 90 :=
by
  sorry

end survey_no_preference_students_l76_76500


namespace least_common_multiple_of_first_ten_l76_76921

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76921


namespace find_x_l76_76392

-- Definition of logarithm in Lean
noncomputable def log (b a: ℝ) : ℝ := Real.log a / Real.log b

-- Problem statement in Lean
theorem find_x (x : ℝ) (h : log 64 4 = 1 / 3) : log x 8 = 1 / 3 → x = 512 :=
by sorry

end find_x_l76_76392


namespace smallest_n_l76_76238

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end smallest_n_l76_76238


namespace linear_equation_solution_l76_76718

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l76_76718


namespace least_common_multiple_first_ten_l76_76832

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76832


namespace geometric_sequence_l76_76198

theorem geometric_sequence (a b c r : ℤ) (h1 : b = a * r) (h2 : c = a * r^2) (h3 : c = a + 56) : b = 21 :=
by sorry

end geometric_sequence_l76_76198


namespace C_share_of_profit_l76_76494

variable (A B C P Rs_36000 k : ℝ)

-- Definitions as per the conditions given in the problem statement.
def investment_A := 24000
def investment_B := 32000
def investment_C := 36000
def total_profit := 92000
def C_Share := 36000

-- The Lean statement without the proof as requested.
theorem C_share_of_profit 
  (h_A : investment_A = 24000)
  (h_B : investment_B = 32000)
  (h_C : investment_C = 36000)
  (h_P : total_profit = 92000)
  (h_C_share : C_Share = 36000)
  : C_Share = (investment_C / k) / ((investment_A / k) + (investment_B / k) + (investment_C / k)) * total_profit := 
sorry

end C_share_of_profit_l76_76494


namespace race_outcome_permutations_l76_76686

theorem race_outcome_permutations : 
  let participants := 6 in
  let outcomes := Nat.factorial participants / Nat.factorial (participants - 4) in
  outcomes = 360 :=
by {
  let participants := 6,
  have h : outcomes = Nat.factorial participants / Nat.factorial (participants - 4),
  exact h,
  simp only [Nat.factorial],
  norm_num,
}

end race_outcome_permutations_l76_76686


namespace least_common_multiple_1_to_10_l76_76897

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76897


namespace prime_gt_three_square_mod_twelve_l76_76769

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end prime_gt_three_square_mod_twelve_l76_76769


namespace max_diagonals_in_grid_l76_76019

-- Define the dimensions of the grid
def grid_width := 8
def grid_height := 5

-- Define the number of 1x2 rectangles
def number_of_1x2_rectangles := grid_width / 2 * grid_height

-- State the theorem
theorem max_diagonals_in_grid : number_of_1x2_rectangles = 20 := 
by 
  -- Simplify the expression
  sorry

end max_diagonals_in_grid_l76_76019


namespace ratio_of_boxes_sold_l76_76138

-- Definitions for conditions
variables (T W Tu : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  W = 2 * T ∧
  Tu = 2 * W ∧
  T = 1200

-- The statement to prove the ratio Tu / W = 2
theorem ratio_of_boxes_sold (T W Tu : ℕ) (h : conditions T W Tu) :
  Tu / W = 2 :=
by
  sorry

end ratio_of_boxes_sold_l76_76138


namespace f_5times_8_eq_l76_76556

def f (x : ℚ) : ℚ := 1 / x ^ 2

theorem f_5times_8_eq :
  f (f (f (f (f (8 : ℚ))))) = 1 / 79228162514264337593543950336 := 
  by
    sorry

end f_5times_8_eq_l76_76556


namespace fishbowl_count_l76_76609

def number_of_fishbowls (total_fish : ℕ) (fish_per_bowl : ℕ) : ℕ :=
  total_fish / fish_per_bowl

theorem fishbowl_count (h1 : 23 > 0) (h2 : 6003 % 23 = 0) :
  number_of_fishbowls 6003 23 = 261 :=
by
  -- proof goes here
  sorry

end fishbowl_count_l76_76609


namespace mary_earns_per_home_l76_76158

noncomputable def earnings_per_home (T : ℕ) (n : ℕ) : ℕ := T / n

theorem mary_earns_per_home :
  ∀ (T n : ℕ), T = 276 → n = 6 → earnings_per_home T n = 46 := 
by
  intros T n h1 h2
  -- Placeholder proof step
  sorry

end mary_earns_per_home_l76_76158


namespace lcm_first_ten_integers_l76_76815

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76815


namespace sector_central_angle_l76_76403

-- Defining the problem as a theorem in Lean 4
theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 4) (h2 : (1 / 2) * r^2 * θ = 1) : θ = 2 :=
by
  sorry

end sector_central_angle_l76_76403


namespace triangle_shape_isosceles_or_right_l76_76133

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles

theorem triangle_shape_isosceles_or_right (h1 : a^2 + b^2 ≠ 0) (h2 : 
  (a^2 + b^2) * Real.sin (A - B) 
  = (a^2 - b^2) * Real.sin (A + B))
  (h3 : ∀ (A B C : ℝ), A + B + C = π) :
  ∃ (isosceles : Bool), (isosceles = true) ∨ (isosceles = false ∧ A + B = π / 2) :=
sorry

end triangle_shape_isosceles_or_right_l76_76133


namespace find_g_seven_l76_76277

variable {g : ℝ → ℝ}

theorem find_g_seven (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 :=
by
  sorry

end find_g_seven_l76_76277


namespace probability_hitting_target_is_0_point_7_l76_76401

noncomputable def probability_hitting_target : ℝ :=
  let P_A := 3 / 5
  let P_notA := 2 / 5
  let P_B_given_A := 0.9
  let P_B_given_notA := 0.4
  P_A * P_B_given_A + P_notA * P_B_given_notA

theorem probability_hitting_target_is_0_point_7 :
  probability_hitting_target = 0.7 := by
  sorry

end probability_hitting_target_is_0_point_7_l76_76401


namespace least_positive_integer_divisible_by_first_ten_l76_76872

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76872


namespace sqrt_five_eq_l76_76454

theorem sqrt_five_eq (m n a b c d : ℤ)
  (h : m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5)) :
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) := by
  sorry

end sqrt_five_eq_l76_76454


namespace gas_fee_calculation_l76_76567

theorem gas_fee_calculation (x : ℚ) (h_usage : x > 60) :
  60 * 0.8 + (x - 60) * 1.2 = 0.88 * x → x * 0.88 = 66 := by
  sorry

end gas_fee_calculation_l76_76567


namespace root_in_interval_imp_range_m_l76_76703

theorem root_in_interval_imp_range_m (m : ℝ) (f : ℝ → ℝ) (h : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0) : 2 < m ∧ m < 4 :=
by
  have exists_x : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0 := h
  sorry

end root_in_interval_imp_range_m_l76_76703


namespace front_crawl_speed_l76_76475
   
   def swim_condition := 
     ∃ F : ℝ, -- Speed of front crawl in yards per minute
     (∃ t₁ t₂ d₁ d₂ : ℝ, -- t₁ is time for front crawl, t₂ is time for breaststroke, d₁ and d₂ are distances
               t₁ = 8 ∧
               t₂ = 4 ∧
               d₁ = t₁ * F ∧
               d₂ = t₂ * 35 ∧
               d₁ + d₂ = 500 ∧
               t₁ + t₂ = 12) ∧
     F = 45
   
   theorem front_crawl_speed : swim_condition :=
     by
       sorry -- Proof goes here, with given conditions satisfying F = 45
   
end front_crawl_speed_l76_76475


namespace product_of_16_and_21_point_3_l76_76278

theorem product_of_16_and_21_point_3 (h1 : 213 * 16 = 3408) : 16 * 21.3 = 340.8 :=
by sorry

end product_of_16_and_21_point_3_l76_76278


namespace log_sequence_value_l76_76233

theorem log_sequence_value :
  ∃ x : ℝ, x = log 3 (81 + x) ∧ x > 0 ∧ x ≈ 5 :=
begin
  sorry
end

end log_sequence_value_l76_76233


namespace grape_juice_problem_l76_76065

noncomputable def grape_juice_amount (initial_mixture_volume : ℕ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  let initial_grape_juice := initial_mixture_volume * initial_concentration
  let total_volume := initial_mixture_volume + final_concentration * (final_concentration - initial_grape_juice) / (1 - final_concentration) -- Total volume after adding x gallons
  let added_grape_juice := total_volume - initial_mixture_volume -- x gallons added
  added_grape_juice

theorem grape_juice_problem :
  grape_juice_amount 40 0.20 0.36 = 10 := 
by
  sorry

end grape_juice_problem_l76_76065


namespace least_common_multiple_first_ten_l76_76818

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76818


namespace measure_of_two_equal_angles_l76_76221

noncomputable def measure_of_obtuse_angle (θ : ℝ) : ℝ := θ + (0.6 * θ)

-- Given conditions
def is_obtuse_isosceles_triangle (θ : ℝ) : Prop :=
  θ = 90 ∧ measure_of_obtuse_angle 90 = 144 ∧ 180 - 144 = 36

-- The main theorem
theorem measure_of_two_equal_angles :
  ∀ θ, is_obtuse_isosceles_triangle θ → 36 / 2 = 18 :=
by
  intros θ h
  sorry

end measure_of_two_equal_angles_l76_76221


namespace polynomial_factorization_l76_76731

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end polynomial_factorization_l76_76731


namespace least_positive_integer_divisible_by_first_ten_l76_76874

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76874


namespace compare_a_b_c_l76_76256

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l76_76256


namespace max_triangle_area_l76_76999

-- Definitions for the conditions
def Point := (ℝ × ℝ)

def point_A : Point := (0, 0)
def point_B : Point := (17, 0)
def point_C : Point := (23, 0)

def slope_ell_A : ℝ := 2
def slope_ell_C : ℝ := -2

axiom rotating_clockwise_with_same_angular_velocity (A B C : Point) : Prop

-- Question transcribed as proving a statement about the maximum area
theorem max_triangle_area (A B C : Point)
  (hA : A = point_A)
  (hB : B = point_B)
  (hC : C = point_C)
  (h_slopeA : ∀ p: Point, slope_ell_A = 2)
  (h_slopeC : ∀ p: Point, slope_ell_C = -2)
  (h_rotation : rotating_clockwise_with_same_angular_velocity A B C) :
  ∃ area_max : ℝ, area_max = 264.5 :=
sorry

end max_triangle_area_l76_76999


namespace least_common_multiple_of_first_ten_l76_76922

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76922


namespace simplify_fraction_l76_76774

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) : (x + 1) / (x^2 + 2 * x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l76_76774


namespace inverse_of_k_l76_76441

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l76_76441


namespace lcm_first_ten_integers_l76_76808

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76808


namespace least_common_multiple_first_ten_l76_76825

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76825


namespace roots_of_equation_l76_76173

theorem roots_of_equation :
  ∀ x : ℝ, x * (x - 1) + 3 * (x - 1) = 0 ↔ x = -3 ∨ x = 1 :=
by {
  sorry
}

end roots_of_equation_l76_76173


namespace least_positive_integer_solution_l76_76531

theorem least_positive_integer_solution : 
  ∃ x : ℕ, x + 3567 ≡ 1543 [MOD 14] ∧ x = 6 := 
by
  -- proof goes here
  sorry

end least_positive_integer_solution_l76_76531


namespace simplify_and_evaluate_l76_76459

theorem simplify_and_evaluate (a b : ℝ) (h : a - 2 * b = -1) :
  -3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -8 :=
by
  sorry

end simplify_and_evaluate_l76_76459


namespace vector_addition_l76_76107

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (6, 2)
def vector_b : ℝ × ℝ := (-2, 4)

-- Theorem statement to prove the sum of vector_a and vector_b equals (4, 6)
theorem vector_addition :
  vector_a + vector_b = (4, 6) :=
sorry

end vector_addition_l76_76107


namespace sum_q_p_evaluations_l76_76230

def p (x : ℝ) : ℝ := |x^2 - 4|
def q (x : ℝ) : ℝ := -|x|

theorem sum_q_p_evaluations : 
  q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3)) = -20 := 
by 
  sorry

end sum_q_p_evaluations_l76_76230


namespace least_common_multiple_of_first_10_integers_l76_76946

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76946


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76845

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76845


namespace least_common_multiple_of_first_ten_positive_integers_l76_76861

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76861


namespace least_common_multiple_of_first_ten_positive_integers_l76_76862

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76862


namespace andy_questions_wrong_l76_76427

variable (a b c d : ℕ)

theorem andy_questions_wrong
  (h1 : a + b = c + d)
  (h2 : a + d = b + c + 6)
  (h3 : c = 7)
  (h4 : d = 9) :
  a = 10 :=
by {
  sorry  -- Proof would go here
}

end andy_questions_wrong_l76_76427


namespace ratio_of_boys_to_girls_l76_76126

theorem ratio_of_boys_to_girls (total_students : ℕ) (girls : ℕ) (boys : ℕ)
  (h_total : total_students = 1040)
  (h_girls : girls = 400)
  (h_boys : boys = total_students - girls) :
  (boys / Nat.gcd boys girls = 8) ∧ (girls / Nat.gcd boys girls = 5) :=
sorry

end ratio_of_boys_to_girls_l76_76126


namespace solve_for_p_l76_76248

def cubic_eq_has_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  5*(a:ℝ)^3 - 5*(p + 1)*(a:ℝ)^2 + (71*p - 1)*(a:ℝ) + 1 = 66*p ∧
  5*(b:ℝ)^3 - 5*(p + 1)*(b:ℝ)^2 + (71*p - 1)*(b:ℝ) + 1 = 66*p ∧
  5*(c:ℝ)^3 - 5*(p + 1)*(c:ℝ)^2 + (71*p - 1)*(c:ℝ) + 1 = 66*p

theorem solve_for_p : ∀ (p : ℝ), cubic_eq_has_natural_roots p → p = 76 :=
by
  sorry

end solve_for_p_l76_76248


namespace malcolm_total_followers_l76_76015

variable (Instagram Facebook Twitter TikTok YouTube : ℕ)

def followers_conditions (Instagram Facebook Twitter TikTok YouTube : ℕ) : Prop :=
  Instagram = 240 ∧
  Facebook = 500 ∧
  Twitter = (Instagram + Facebook) / 2 ∧
  TikTok = 3 * Twitter ∧
  YouTube = TikTok + 510

theorem malcolm_total_followers : ∃ (tot_followers : ℕ), 
  followers_conditions Instagram Facebook Twitter TikTok YouTube →
  tot_followers = Instagram + Facebook + Twitter + TikTok + YouTube ∧
  tot_followers = 3840 :=
by
  intros h
  sorry

end malcolm_total_followers_l76_76015


namespace foci_of_ellipse_l76_76466

def ellipse_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ (y = 12 ∨ y = -12))

theorem foci_of_ellipse :
  ∀ (x y : ℝ), (x^2)/25 + (y^2)/169 = 1 → ellipse_focus x y :=
by
  intros x y h
  sorry

end foci_of_ellipse_l76_76466


namespace at_least_one_is_one_l76_76696

theorem at_least_one_is_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  (1/x + 1/y + 1/z = 1) → (1/(x + y + z) = 1) → (x = 1 ∨ y = 1 ∨ z = 1) :=
by
  sorry

end at_least_one_is_one_l76_76696


namespace malcolm_followers_l76_76016

theorem malcolm_followers :
  let instagram_followers := 240
  let facebook_followers := 500
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  let youtube_followers := tiktok_followers + 510
  instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 :=
by {
  sorry
}

end malcolm_followers_l76_76016


namespace common_ratio_l76_76608

theorem common_ratio (a S r : ℝ) (h1 : S = a / (1 - r))
  (h2 : ar^5 / (1 - r) = S / 81) : r = 1 / 3 :=
sorry

end common_ratio_l76_76608


namespace solutions_of_quadratic_eq_l76_76330

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end solutions_of_quadratic_eq_l76_76330


namespace find_larger_number_l76_76474

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 :=
  sorry

end find_larger_number_l76_76474


namespace least_divisible_1_to_10_l76_76805

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76805


namespace carnations_count_l76_76651

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end carnations_count_l76_76651


namespace tom_profit_l76_76477

-- Define the initial conditions
def initial_investment : ℕ := 20 * 3
def revenue_from_selling : ℕ := 10 * 4
def value_of_remaining_shares : ℕ := 10 * 6
def total_amount : ℕ := revenue_from_selling + value_of_remaining_shares

-- We claim that the profit Tom makes is 40 dollars
theorem tom_profit : (total_amount - initial_investment) = 40 := by
  sorry

end tom_profit_l76_76477


namespace find_x_l76_76004

def sum_sequence (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2  -- Sum of an arithmetic progression

theorem find_x (x : ℕ) (h1 : sum_sequence x 10 = 65) : x = 2 :=
by {
  -- the proof goes here
  sorry
}

end find_x_l76_76004


namespace Tony_temp_above_fever_threshold_l76_76616

def normal_temp : ℕ := 95
def illness_A : ℕ := 10
def illness_B : ℕ := 4
def illness_C : Int := -2
def fever_threshold : ℕ := 100

theorem Tony_temp_above_fever_threshold :
  let T := normal_temp + illness_A + illness_B + illness_C
  T = 107 ∧ (T - fever_threshold) = 7 := by
  -- conditions
  let t_0 := normal_temp
  let T_A := illness_A
  let T_B := illness_B
  let T_C := illness_C
  let F := fever_threshold
  -- calculations
  let T := t_0 + T_A + T_B + T_C
  show T = 107 ∧ (T - F) = 7
  sorry

end Tony_temp_above_fever_threshold_l76_76616


namespace soap_bars_problem_l76_76624

theorem soap_bars_problem :
  ∃ (N : ℤ), 200 < N ∧ N < 300 ∧ 2007 % N = 5 :=
sorry

end soap_bars_problem_l76_76624


namespace young_employees_l76_76172

theorem young_employees (ratio_young : ℕ)
                        (ratio_middle : ℕ)
                        (ratio_elderly : ℕ)
                        (sample_selected : ℕ)
                        (prob_selection : ℚ)
                        (h_ratio : ratio_young = 10 ∧ ratio_middle = 8 ∧ ratio_elderly = 7)
                        (h_sample : sample_selected = 200)
                        (h_prob : prob_selection = 0.2) :
                        10 * (sample_selected / prob_selection) / 25 = 400 :=
by {
  sorry
}

end young_employees_l76_76172


namespace sum_of_cubes_roots_poly_l76_76582

theorem sum_of_cubes_roots_poly :
  (∀ (a b c : ℂ), (a^3 - 2*a^2 + 2*a - 3 = 0) ∧ (b^3 - 2*b^2 + 2*b - 3 = 0) ∧ (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5) :=
by
  sorry

end sum_of_cubes_roots_poly_l76_76582


namespace max_ounces_amber_can_get_l76_76664

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l76_76664


namespace point_not_in_fourth_quadrant_l76_76739

theorem point_not_in_fourth_quadrant (m : ℝ) : ¬(m-2 > 0 ∧ m+1 < 0) := 
by
  -- Since (m+1) - (m-2) = 3, which is positive,
  -- m+1 > m-2, thus the statement ¬(m-2 > 0 ∧ m+1 < 0) holds.
  sorry

end point_not_in_fourth_quadrant_l76_76739


namespace simplify_expression_l76_76591

theorem simplify_expression :
  (625: ℝ)^(1/4) * (256: ℝ)^(1/3) = 20 := 
sorry

end simplify_expression_l76_76591


namespace hallie_number_of_paintings_sold_l76_76274

/-- 
Hallie is an artist. She wins an art contest, and she receives a $150 prize. 
She sells some of her paintings for $50 each. 
She makes a total of $300 from her art. 
How many paintings did she sell?
-/
theorem hallie_number_of_paintings_sold 
    (prize : ℕ)
    (price_per_painting : ℕ)
    (total_earnings : ℕ)
    (prize_eq : prize = 150)
    (price_eq : price_per_painting = 50)
    (total_eq : total_earnings = 300) :
    (total_earnings - prize) / price_per_painting = 3 :=
by
  sorry

end hallie_number_of_paintings_sold_l76_76274


namespace prime_factors_power_l76_76554

-- Given conditions
def a_b_c_factors (a b c : ℕ) : Prop :=
  (∀ x, x = a ∨ x = b ∨ x = c → Prime x) ∧
  a < b ∧ b < c ∧ a * b * c ∣ 1998

-- Proof problem
theorem prime_factors_power (a b c : ℕ) (h : a_b_c_factors a b c) : (b + c) ^ a = 1600 := 
sorry

end prime_factors_power_l76_76554


namespace same_volume_increase_rate_l76_76203

def initial_radius := 10
def initial_height := 5 

def volume_increase_rate_new_radius (x : ℝ) :=
  let r' := initial_radius + 2 * x
  (r' ^ 2) * initial_height  - (initial_radius ^ 2) * initial_height

def volume_increase_rate_new_height (x : ℝ) :=
  let h' := initial_height + 3 * x
  (initial_radius ^ 2) * h' - (initial_radius ^ 2) * initial_height

theorem same_volume_increase_rate (x : ℝ) : volume_increase_rate_new_radius x = volume_increase_rate_new_height x → x = 5 := 
  by sorry

end same_volume_increase_rate_l76_76203


namespace problem_r_minus_s_l76_76146

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l76_76146


namespace scientific_notation_280000_l76_76020

theorem scientific_notation_280000 : 
  ∃ n: ℝ, n * 10^5 = 280000 ∧ n = 2.8 :=
by
-- our focus is on the statement outline, thus we use sorry to skip the proof part
  sorry

end scientific_notation_280000_l76_76020


namespace price_difference_proof_l76_76605

theorem price_difference_proof (y : ℝ) (n : ℕ) :
  ∃ n : ℕ, (4.20 + 0.45 * n) = (6.30 + 0.01 * y * n + 0.65) → 
  n = (275 / (45 - y)) :=
by
  sorry

end price_difference_proof_l76_76605


namespace tyson_age_l76_76139

noncomputable def age_proof : Prop :=
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  t = 20           -- Statement that needs to be proved

theorem tyson_age : age_proof :=
by
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  show t = 20
  sorry

end tyson_age_l76_76139


namespace new_student_weight_l76_76057

theorem new_student_weight 
  (w_avg : ℝ)
  (w_new : ℝ)
  (condition : (5 * w_avg - 72 = 5 * (w_avg - 12) + w_new)) 
  : w_new = 12 := 
  by 
  sorry

end new_student_weight_l76_76057


namespace find_integer_pairs_l76_76527

theorem find_integer_pairs :
  ∃ (n : ℤ) (a : ℤ) (b : ℤ),
    (∀ a b : ℤ, (∃ m : ℤ, a^2 - 4*b = m^2) ∧ (∃ k : ℤ, b^2 - 4*a = k^2) ↔ 
    (a = 0 ∧ ∃ n : ℤ, b = n^2) ∨
    (b = 0 ∧ ∃ n : ℤ, a = n^2) ∨
    (b > 0 ∧ ∃ a : ℤ, a^2 > 0 ∧ b = -1 - a) ∨
    (a > 0 ∧ ∃ b : ℤ, b^2 > 0 ∧ a = -1 - b) ∨
    (a = 4 ∧ b = 4) ∨
    (a = 5 ∧ b = 6) ∨
    (a = 6 ∧ b = 5)) :=
sorry

end find_integer_pairs_l76_76527


namespace lcm_first_ten_numbers_l76_76915

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76915


namespace lcm_first_ten_l76_76849

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76849


namespace parabola_points_relation_l76_76453

theorem parabola_points_relation (c y1 y2 y3 : ℝ)
  (h1 : y1 = -(-2)^2 - 2*(-2) + c)
  (h2 : y2 = -(0)^2 - 2*(0) + c)
  (h3 : y3 = -(1)^2 - 2*(1) + c) :
  y1 = y2 ∧ y2 > y3 :=
by
  sorry

end parabola_points_relation_l76_76453


namespace least_common_multiple_first_ten_integers_l76_76940

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76940


namespace similar_triangle_perimeter_l76_76617

theorem similar_triangle_perimeter 
  (a b c : ℝ) (a_sim : ℝ)
  (h1 : a = b) (h2 : b = c)
  (h3 : a = 15) (h4 : a_sim = 45)
  (h5 : a_sim / a = 3) :
  a_sim + a_sim + a_sim = 135 :=
by
  sorry

end similar_triangle_perimeter_l76_76617


namespace zero_knights_l76_76197

noncomputable def knights_count (n : ℕ) : ℕ := sorry

theorem zero_knights (n : ℕ) (half_lairs : n ≥ 205) :
  knights_count 410 = 0 :=
sorry

end zero_knights_l76_76197


namespace max_value_of_quadratic_function_l76_76028

def quadratic_function (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 5 ∧ ∀ y : ℝ, quadratic_function y ≤ 5 :=
by
  sorry

end max_value_of_quadratic_function_l76_76028


namespace sample_size_is_15_l76_76074

-- Define the given conditions as constants and assumptions within the Lean environment.
def total_employees := 750
def young_workers := 350
def middle_aged_workers := 250
def elderly_workers := 150
def sample_young_workers := 7

-- Define the proposition that given these conditions, the sample size is 15.
theorem sample_size_is_15 : ∃ n : ℕ, (7 / n = 350 / 750) ∧ n = 15 := by
  sorry

end sample_size_is_15_l76_76074


namespace poly_square_of_binomial_l76_76488

theorem poly_square_of_binomial (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := 
by 
  sorry

end poly_square_of_binomial_l76_76488


namespace find_difference_l76_76754

variable (a b c d e f : ℝ)

-- Conditions
def cond1 : Prop := a - b = c + d + 9
def cond2 : Prop := a + b = c - d - 3
def cond3 : Prop := e = a^2 + b^2
def cond4 : Prop := f = c^2 + d^2
def cond5 : Prop := f - e = 5 * a + 2 * b + 3 * c + 4 * d

-- Problem Statement
theorem find_difference (h1 : cond1 a b c d) (h2 : cond2 a b c d) (h3 : cond3 a b e) (h4 : cond4 c d f) (h5 : cond5 a b c d e f) : a - c = 3 :=
sorry

end find_difference_l76_76754


namespace operation_8_to_cube_root_16_l76_76958

theorem operation_8_to_cube_root_16 : ∃ (x : ℕ), x = 8 ∧ (x * x = (Nat.sqrt 16)^3) :=
by
  sorry

end operation_8_to_cube_root_16_l76_76958


namespace tilling_time_in_minutes_l76_76225

-- Definitions
def plot_width : ℕ := 110
def plot_length : ℕ := 120
def tiller_width : ℕ := 2
def tilling_rate : ℕ := 2 -- 2 seconds per foot

-- Theorem: The time to till the entire plot in minutes
theorem tilling_time_in_minutes : (plot_width / tiller_width * plot_length * tilling_rate) / 60 = 220 := by
  sorry

end tilling_time_in_minutes_l76_76225


namespace least_common_multiple_of_first_ten_l76_76918

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76918


namespace cube_surface_area_ratio_l76_76054

variable (x : ℝ) (hx : x > 0)

theorem cube_surface_area_ratio (hx : x > 0):
  let side1 := 7 * x
  let side2 := x
  let SA1 := 6 * side1^2
  let SA2 := 6 * side2^2
  (SA1 / SA2) = 49 := 
by 
  sorry

end cube_surface_area_ratio_l76_76054


namespace polynomial_term_count_l76_76229

open Nat

theorem polynomial_term_count (N : ℕ) (h : (N.choose 5) = 2002) : N = 17 :=
by
  sorry

end polynomial_term_count_l76_76229


namespace relationship_between_y1_y2_l76_76563

theorem relationship_between_y1_y2 
  (y1 y2 : ℝ) 
  (hA : y1 = 6 / -3) 
  (hB : y2 = 6 / 2) : y1 < y2 :=
by 
  sorry

end relationship_between_y1_y2_l76_76563


namespace total_books_on_shelves_l76_76035

-- Definitions based on conditions
def num_shelves : Nat := 150
def books_per_shelf : Nat := 15

-- The statement to be proved
theorem total_books_on_shelves : num_shelves * books_per_shelf = 2250 := by
  sorry

end total_books_on_shelves_l76_76035


namespace number_of_students_from_second_department_is_17_l76_76132

noncomputable def students_selected_from_second_department 
  (total_students : ℕ)
  (num_departments : ℕ)
  (students_per_department : List (ℕ × ℕ))
  (sample_size : ℕ)
  (starting_number : ℕ) : ℕ :=
-- This function will compute the number of students selected from the second department.
sorry

theorem number_of_students_from_second_department_is_17 : 
  students_selected_from_second_department 600 3 
    [(1, 300), (301, 495), (496, 600)] 50 3 = 17 :=
-- Proof is left as an exercise.
sorry

end number_of_students_from_second_department_is_17_l76_76132


namespace StacyBoughtPacks_l76_76594

theorem StacyBoughtPacks (sheets_per_pack days daily_printed_sheets total_packs : ℕ) 
  (h1 : sheets_per_pack = 240)
  (h2 : days = 6)
  (h3 : daily_printed_sheets = 80) 
  (h4 : total_packs = (days * daily_printed_sheets) / sheets_per_pack) : total_packs = 2 :=
by 
  sorry

end StacyBoughtPacks_l76_76594


namespace find_digits_l76_76373

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l76_76373


namespace negation_of_p_l76_76550

theorem negation_of_p :
  (¬ (∀ x > 0, (x+1)*Real.exp x > 1)) ↔ 
  (∃ x ≤ 0, (x+1)*Real.exp x ≤ 1) :=
sorry

end negation_of_p_l76_76550


namespace smallest_positive_integer_l76_76487

theorem smallest_positive_integer (
  a : ℕ
) : 
  (a ≡ 5 [MOD 6]) ∧ (a ≡ 7 [MOD 8]) → a = 23 :=
by sorry

end smallest_positive_integer_l76_76487


namespace percentage_alcohol_second_vessel_l76_76510

theorem percentage_alcohol_second_vessel :
  (∀ (x : ℝ),
    (0.25 * 3 + (x / 100) * 5 = 0.275 * 10) -> x = 40) :=
by
  intro x h
  sorry

end percentage_alcohol_second_vessel_l76_76510


namespace average_age_correct_l76_76127

def ratio (m w : ℕ) : Prop := w * 8 = m * 9

def average_age_of_group (m w : ℕ) (avg_men avg_women : ℕ) : ℚ :=
  (avg_men * m + avg_women * w) / (m + w)

/-- The average age of the group is 32 14/17 given that the ratio of the number of women to the number of men is 9 to 8, 
    the average age of the women is 30 years, and the average age of the men is 36 years. -/
theorem average_age_correct
  (m w : ℕ)
  (h_ratio : ratio m w)
  (h_avg_women : avg_age_women = 30)
  (h_avg_men : avg_age_men = 36) :
  average_age_of_group m w avg_age_men avg_age_women = 32 + (14 / 17) := 
by
  sorry

end average_age_correct_l76_76127


namespace count_non_decreasing_maps_l76_76390

-- Define the set and the mapping condition
def is_non_decreasing_map (f : Fin 3 → Fin 5) : Prop :=
  ∀ (i j : Fin 3), i < j → f i ≤ f j

-- Define the problem as a theorem statement
theorem count_non_decreasing_maps :
  (Fin 3 → Fin 5) → Prop :=
  λ f, ∃! (f : Fin 3 → Fin 5), is_non_decreasing_map f :=
  35 :=
sorry

end count_non_decreasing_maps_l76_76390


namespace women_in_first_class_equals_22_l76_76526

def number_of_women (total_passengers : Nat) : Nat :=
  total_passengers * 50 / 100

def number_of_women_in_first_class (number_of_women : Nat) : Nat :=
  number_of_women * 15 / 100

theorem women_in_first_class_equals_22 (total_passengers : Nat) (h1 : total_passengers = 300) : 
  number_of_women_in_first_class (number_of_women total_passengers) = 22 :=
by
  sorry

end women_in_first_class_equals_22_l76_76526


namespace monotonic_decreasing_interval_of_f_l76_76029

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval_of_f :
  { x : ℝ | x > Real.exp 1 } = {y : ℝ | ∀ ε > 0, (x : ℝ) → (0 < x → (f (x + ε) < f x) ∧ (f x < f (x + ε)))}
:=
sorry

end monotonic_decreasing_interval_of_f_l76_76029


namespace min_pos_solution_eqn_l76_76685

theorem min_pos_solution_eqn (x : ℝ) (h : (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 25) : x = 7 * Real.sqrt 3 :=
sorry

end min_pos_solution_eqn_l76_76685


namespace least_common_multiple_first_ten_integers_l76_76939

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76939


namespace radius_of_sphere_inscribed_in_box_l76_76983

theorem radius_of_sphere_inscribed_in_box (a b c s : ℝ)
  (h1 : a + b + c = 42)
  (h2 : 2 * (a * b + b * c + c * a) = 576)
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) :
  s = 3 * Real.sqrt 33 :=
by sorry

end radius_of_sphere_inscribed_in_box_l76_76983


namespace least_positive_integer_divisible_by_first_ten_l76_76870

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76870


namespace shaded_fraction_in_fifth_diagram_l76_76118

-- Definitions for conditions
def geometric_sequence (a₀ r n : ℕ) : ℕ := a₀ * r^n

def total_triangles (n : ℕ) : ℕ := n^2

-- Lean theorem statement
theorem shaded_fraction_in_fifth_diagram 
  (a₀ r n : ℕ) 
  (h_geometric : a₀ = 1) 
  (h_ratio : r = 2)
  (h_step_number : n = 4):
  (geometric_sequence a₀ r n) / (total_triangles (n + 1)) = 16 / 25 :=
by
  sorry

end shaded_fraction_in_fifth_diagram_l76_76118


namespace parabola_line_non_intersect_l76_76438

def P (x : ℝ) : ℝ := x^2 + 3 * x + 1
def Q : ℝ × ℝ := (10, 50)

def line_through_Q_with_slope (m x : ℝ) : ℝ := m * (x - Q.1) + Q.2

theorem parabola_line_non_intersect (r s : ℝ) (h : ∀ m, (r < m ∧ m < s) ↔ (∀ x, 
  x^2 + (3 - m) * x + (10 * m - 49) ≠ 0)) : r + s = 46 := 
sorry

end parabola_line_non_intersect_l76_76438


namespace number_of_days_b_worked_l76_76962

variables (d_a : ℕ) (d_c : ℕ) (total_earnings : ℝ)
variables (wage_ratio : ℝ) (wage_c : ℝ) (d_b : ℕ) (wages : ℝ)
variables (total_wage_a : ℝ) (total_wage_c : ℝ) (total_wage_b : ℝ)

-- Given conditions
def given_conditions :=
  d_a = 6 ∧
  d_c = 4 ∧
  wage_c = 95 ∧
  wage_ratio = wage_c / 5 ∧
  wages = 3 * wage_ratio ∧
  total_earnings = 1406 ∧
  total_wage_a = d_a * wages ∧
  total_wage_c = d_c * wage_c ∧
  total_wage_b = d_b * (4 * wage_ratio) ∧
  total_wage_a + total_wage_b + total_wage_c = total_earnings

-- Theorem to prove
theorem number_of_days_b_worked :
  given_conditions d_a d_c total_earnings wage_ratio wage_c d_b wages total_wage_a total_wage_c total_wage_b →
  d_b = 9 :=
by
  intro h
  sorry

end number_of_days_b_worked_l76_76962


namespace g_h_2_equals_584_l76_76110

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end g_h_2_equals_584_l76_76110


namespace log_expr_solution_l76_76235

-- Define the function that represents the recursive logarithmic expression
def log_expr (x : ℝ) := Real.logb 3 (81 + x)

-- The main statement we want to prove
theorem log_expr_solution : ∃ x : ℝ, x = log_expr x ∧ 0 < x ∧ x = 8 :=
by
  sorry

end log_expr_solution_l76_76235


namespace trigonometric_identity_l76_76638

theorem trigonometric_identity
    (α φ : ℝ) :
    4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = Real.cos (2 * α) :=
by
  sorry

end trigonometric_identity_l76_76638


namespace circle_in_fourth_quadrant_l76_76730

theorem circle_in_fourth_quadrant (a : ℝ) :
  (∃ (x y: ℝ), x^2 + y^2 - 2 * a * x + 4 * a * y + 6 * a^2 - a = 0 ∧ (a > 0) ∧ (-2 * y < 0)) → (0 < a ∧ a < 1) :=
by
  sorry

end circle_in_fourth_quadrant_l76_76730


namespace john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l76_76578

theorem john_needs_to_sell_1200_pencils_to_make_120_dollars_profit :
  ∀ (buy_rate_pencils : ℕ) (buy_rate_dollars : ℕ) (sell_rate_pencils : ℕ) (sell_rate_dollars : ℕ),
    buy_rate_pencils = 5 →
    buy_rate_dollars = 7 →
    sell_rate_pencils = 4 →
    sell_rate_dollars = 6 →
    ∃ (n_pencils : ℕ), n_pencils = 1200 ∧ 
                        (sell_rate_dollars / sell_rate_pencils - buy_rate_dollars / buy_rate_pencils) * n_pencils = 120 :=
by
  sorry

end john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l76_76578


namespace frequency_of_heads_l76_76570

theorem frequency_of_heads (n h : ℕ) (h_n : n = 100) (h_h : h = 49) : (h : ℚ) / n = 0.49 :=
by
  rw [h_n, h_h]
  norm_num

end frequency_of_heads_l76_76570


namespace find_x_l76_76262

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : nat.factors x = [31, p1, p2]) : x = 32767 :=
by
  sorry

end find_x_l76_76262


namespace distinct_solutions_difference_l76_76151

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l76_76151


namespace least_common_multiple_of_first_10_integers_l76_76951

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76951


namespace range_of_x_l76_76522

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x :
  ∀ x : ℝ, (f x > f (2*x - 1)) ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end range_of_x_l76_76522


namespace pow_comparison_l76_76191

theorem pow_comparison : 2^700 > 5^300 :=
by sorry

end pow_comparison_l76_76191


namespace speed_ratio_l76_76361

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end speed_ratio_l76_76361


namespace relationship_abc_l76_76258

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l76_76258


namespace least_common_multiple_of_first_10_integers_l76_76954

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76954


namespace vertex_of_parabola_l76_76383

theorem vertex_of_parabola (x y : ℝ) : (y^2 - 4 * y + 3 * x + 7 = 0) → (x, y) = (-1, 2) :=
by
  sorry

end vertex_of_parabola_l76_76383


namespace difference_of_squares_l76_76282

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) : 
  x^2 - y^2 = 200 := 
sorry

end difference_of_squares_l76_76282


namespace find_x_l76_76261

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end find_x_l76_76261


namespace problem_remainder_l76_76051

theorem problem_remainder :
  ((12095 + 12097 + 12099 + 12101 + 12103 + 12105 + 12107) % 10) = 7 := by
  sorry

end problem_remainder_l76_76051


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76839

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76839


namespace inequality_of_cubic_powers_l76_76098

theorem inequality_of_cubic_powers 
  (a b: ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h_cond : a * |a| > b * |b|) : 
  a^3 > b^3 := by
  sorry

end inequality_of_cubic_powers_l76_76098


namespace linear_equation_solution_l76_76719

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l76_76719


namespace max_min_values_in_region_l76_76532

-- Define the function
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D (x y : ℝ) : Prop := (0 ≤ x) ∧ (x - 2 * y ≤ 0) ∧ (x + y - 6 ≤ 0)

-- Define the proof problem
theorem max_min_values_in_region :
  (∀ (x y : ℝ), D x y → z x y ≥ 0) ∧
  (∀ (x y : ℝ), D x y → z x y ≤ 32) :=
by 
  sorry -- Proof omitted

end max_min_values_in_region_l76_76532


namespace lcm_first_ten_numbers_l76_76914

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76914


namespace interest_rate_correct_l76_76186

namespace InterestProblem

variable (P : ℤ) (SI : ℤ) (T : ℤ)

def rate_of_interest (P : ℤ) (SI : ℤ) (T : ℤ) : ℚ :=
  (SI * 100) / (P * T)

theorem interest_rate_correct :
  rate_of_interest 400 140 2 = 17.5 := by
  sorry

end InterestProblem

end interest_rate_correct_l76_76186


namespace evaluate_g_h_2_l76_76108

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end evaluate_g_h_2_l76_76108


namespace A_sub_B_value_l76_76976

def A : ℕ := 1000 * 1 + 100 * 16 + 10 * 28
def B : ℕ := 355 + 245 * 3

theorem A_sub_B_value : A - B = 1790 := by
  sorry

end A_sub_B_value_l76_76976


namespace range_of_x_for_sqrt_meaningful_l76_76423

theorem range_of_x_for_sqrt_meaningful (x : ℝ) (h : x + 2 ≥ 0) : x ≥ -2 :=
by {
  sorry
}

end range_of_x_for_sqrt_meaningful_l76_76423


namespace selling_prices_l76_76219

theorem selling_prices {x y : ℝ} (h1 : y - x = 10) (h2 : (y - 5) - 1.10 * x = 1) :
  x = 40 ∧ y = 50 := by
  sorry

end selling_prices_l76_76219


namespace bd_ad_ratio_l76_76121

noncomputable def mass_point_geometry_bd_ad : ℚ := 
  let AT_OVER_ET := 5
  let DT_OVER_CT := 2
  let mass_A := 1
  let mass_D := 3 * mass_A
  let mass_B := mass_A + mass_D
  mass_B / mass_D

theorem bd_ad_ratio (h1 : AT/ET = 5) (h2 : DT/CT = 2) : BD/AD = 4 / 3 :=
by
  have mass_A := 1
  have mass_D := 3
  have mass_B := 4
  have h := mass_B / mass_D
  sorry

end bd_ad_ratio_l76_76121


namespace meaning_of_poverty_l76_76496

theorem meaning_of_poverty (s : String) : s = "poverty" ↔ s = "poverty" := sorry

end meaning_of_poverty_l76_76496


namespace ball_bounce_height_l76_76063

theorem ball_bounce_height :
  ∃ k : ℕ, 2000 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ j : ℕ, j < k → 2000 * (2 / 3 : ℝ) ^ j ≥ 2 :=
by {
  sorry
}

end ball_bounce_height_l76_76063


namespace polygon_sides_eq_four_l76_76565

theorem polygon_sides_eq_four (n : ℕ)
  (h_interior : (n - 2) * 180 = 360)
  (h_exterior : ∀ (m : ℕ), m = n -> 360 = 360) :
  n = 4 :=
sorry

end polygon_sides_eq_four_l76_76565


namespace butter_needed_for_original_recipe_l76_76087

-- Define the conditions
def butter_to_flour_ratio : ℚ := 12 / 56

def flour_for_original_recipe : ℚ := 14

def butter_for_original_recipe (ratio : ℚ) (flour : ℚ) : ℚ :=
  ratio * flour

-- State the theorem
theorem butter_needed_for_original_recipe :
  butter_for_original_recipe butter_to_flour_ratio flour_for_original_recipe = 3 := 
sorry

end butter_needed_for_original_recipe_l76_76087


namespace number_of_kids_per_day_l76_76294

theorem number_of_kids_per_day (K : ℕ) 
    (kids_charge : ℕ := 3) 
    (adults_charge : ℕ := kids_charge * 2) 
    (daily_earnings_from_adults : ℕ := 10 * adults_charge) 
    (weekly_earnings : ℕ := 588) 
    (daily_earnings : ℕ := weekly_earnings / 7) :
    (daily_earnings - daily_earnings_from_adults) / kids_charge = 8 :=
by
  sorry

end number_of_kids_per_day_l76_76294


namespace nuts_eaten_condition_not_all_nuts_eaten_l76_76223

/-- proof problem with conditions and questions --/

-- Let's define the initial setup and the conditions:

def anya_has_all_nuts (nuts : Nat) := nuts > 3

def distribution (a b c : ℕ → ℕ) (n : ℕ) := 
  ((a (n + 1) = b n + c n + (a n % 2)) ∧ 
   (b (n + 1) = a n / 2) ∧ 
   (c (n + 1) = a n / 2))

def nuts_eaten (a b c : ℕ → ℕ) (n : ℕ) := 
  (a n % 2 > 0 ∨ b n % 2 > 0 ∨ c n % 2 > 0)

-- Prove at least one nut will be eaten
theorem nuts_eaten_condition (a b c : ℕ → ℕ) (n : ℕ) :
  anya_has_all_nuts (a 0) → distribution a b c n → nuts_eaten a b c n :=
sorry

-- Prove not all nuts will be eaten
theorem not_all_nuts_eaten (a b c : ℕ → ℕ):
  anya_has_all_nuts (a 0) → distribution a b c n → 
  ¬∀ (n: ℕ), (a n = 0 ∧ b n = 0 ∧ c n = 0) :=
sorry

end nuts_eaten_condition_not_all_nuts_eaten_l76_76223


namespace local_minimum_at_2_l76_76707

noncomputable def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem local_minimum_at_2 (m : ℝ) (h : 2 * (2 - m)^2 + 2 * 4 * (2 - m) = 0) : m = 6 :=
by
  sorry

end local_minimum_at_2_l76_76707


namespace vector_magnitude_l76_76102

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude : 
  let AB := (-1, 2)
  let BC := (x, -5)
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  dot_product AB BC = -7 → magnitude AC = 5 :=
by sorry

end vector_magnitude_l76_76102


namespace daily_rate_problem_l76_76436

noncomputable def daily_rate : ℝ := 126.19 -- Correct answer

theorem daily_rate_problem
  (days : ℕ := 14)
  (pet_fee : ℝ := 100)
  (service_fee_rate : ℝ := 0.20)
  (security_deposit : ℝ := 1110)
  (deposit_rate : ℝ := 0.50)
  (x : ℝ) : x = daily_rate :=
by
  have total_cost := days * x + pet_fee + service_fee_rate * (days * x)
  have total_cost_with_fees := days * x * (1 + service_fee_rate) + pet_fee
  have security_deposit_cost := deposit_rate * total_cost_with_fees
  have eq_security : security_deposit_cost = security_deposit := sorry
  sorry

end daily_rate_problem_l76_76436


namespace inequality_proof_l76_76313

theorem inequality_proof (x y : ℝ) :
  abs ((x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2))) ≤ 1 / 2 := 
sorry

end inequality_proof_l76_76313


namespace polar_to_cartesian_and_distance_l76_76131

-- Define the main problem
theorem polar_to_cartesian_and_distance :
  (∀ θ : ℝ, (ρ θ = 6 * real.sin θ) → ∀ (x y : ℝ), (x = 1 ∧ y = 1) →
  ∀ (t1 t2 : ℝ), (|t1 - t2| = 3 * real.sqrt 2) →
  (t1 * t2 = -4 ∧ t1 = -2 * t2 ∨ t1 = 2 * t2) →
  |2 * t2 + t2| = 3 * real.sqrt 2) :=
begin
  sorry
end

end polar_to_cartesian_and_distance_l76_76131


namespace aunt_wang_bought_n_lilies_l76_76079

theorem aunt_wang_bought_n_lilies 
  (cost_rose : ℕ) 
  (cost_lily : ℕ) 
  (total_spent : ℕ) 
  (num_roses : ℕ) 
  (num_lilies : ℕ) 
  (roses_cost : num_roses * cost_rose = 10) 
  (total_spent_cond : total_spent = 55) 
  (cost_conditions : cost_rose = 5 ∧ cost_lily = 9) 
  (spending_eq : total_spent = num_roses * cost_rose + num_lilies * cost_lily) : 
  num_lilies = 5 :=
by 
  sorry

end aunt_wang_bought_n_lilies_l76_76079


namespace bowling_ball_weight_l76_76244

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end bowling_ball_weight_l76_76244


namespace distance_between_ports_l76_76504

theorem distance_between_ports (x : ℝ) (speed_ship : ℝ) (speed_water : ℝ) (time_difference : ℝ) 
  (speed_downstream := speed_ship + speed_water) 
  (speed_upstream := speed_ship - speed_water) 
  (time_downstream := x / speed_downstream) 
  (time_upstream := x / speed_upstream) 
  (h : time_downstream + time_difference = time_upstream) 
  (h_ship : speed_ship = 26)
  (h_water : speed_water = 2)
  (h_time : time_difference = 3) : x = 504 :=
by
  -- The proof is omitted 
  sorry

end distance_between_ports_l76_76504


namespace greatest_divisor_l76_76089

theorem greatest_divisor (n : ℕ) (h1 : 1657 % n = 6) (h2 : 2037 % n = 5) : n = 127 :=
by
  sorry

end greatest_divisor_l76_76089


namespace annalise_total_cost_correct_l76_76222

-- Define the constants from the problem
def boxes : ℕ := 25
def packs_per_box : ℕ := 18
def tissues_per_pack : ℕ := 150
def tissue_price : ℝ := 0.06
def discount_per_box : ℝ := 0.10
def volume_discount : ℝ := 0.08
def tax_rate : ℝ := 0.05

-- Calculate the total number of tissues
def total_tissues : ℕ := boxes * packs_per_box * tissues_per_pack

-- Calculate the total cost without any discounts
def initial_cost : ℝ := total_tissues * tissue_price

-- Apply the 10% discount on the price of the total packs in each box purchased
def cost_after_box_discount : ℝ := initial_cost * (1 - discount_per_box)

-- Apply the 8% volume discount for buying 10 or more boxes
def cost_after_volume_discount : ℝ := cost_after_box_discount * (1 - volume_discount)

-- Apply the 5% tax on the final price after all discounts
def final_cost : ℝ := cost_after_volume_discount * (1 + tax_rate)

-- Define the expected final cost
def expected_final_cost : ℝ := 3521.07

-- Proof statement
theorem annalise_total_cost_correct : final_cost = expected_final_cost := by
  -- Sorry is used as placeholder for the actual proof
  sorry

end annalise_total_cost_correct_l76_76222


namespace total_dinners_l76_76761

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end total_dinners_l76_76761


namespace lcm_first_ten_integers_l76_76813

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76813


namespace mod_equiv_m_in_range_l76_76724

theorem mod_equiv_m_in_range :
  ∃ m ∈ (set.range (λ n, 150 + n)).inter (set.Icc 150 201),
  (25 - 98) ≡ m [MOD 53] :=
by {
  let c := 25,
  let d := 98,
  let m := 192,
  have h_c : c ≡ 25 [MOD 53] := by refl,
  have h_d : d ≡ 98 [MOD 53] := by refl,
  have h_cd_mod : (c - d) % 53 = (-20) % 53 := by norm_num,
  have h_cd_equiv : (c - d) ≡ -20 [MOD 53] := int.modeq_of_dvd,
  have h_cd_33 : (-20) % 53 = 33 := by norm_num,
  have h_range : m ∈ set.range (λ n, 150 + n).inter (set.Icc 150 201),
  { use 42,
    split,
    { use 42,
      refl },
    { split,
      linarith,
      linarith } },
  use m,
  exact ⟨h_range, int.modeq.trans h_cd_equiv (int.modeq.of_eq h_cd_33)⟩
} 

end mod_equiv_m_in_range_l76_76724


namespace y_pow_x_eq_x_pow_y_l76_76346

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) :
    let x := (1 + 1 / (n : ℝ)) ^ n
    let y := (1 + 1 / (n : ℝ)) ^ (n + 1)
    y ^ x = x ^ y := 
    sorry

end y_pow_x_eq_x_pow_y_l76_76346


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76836

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76836


namespace find_smaller_number_l76_76162

theorem find_smaller_number (a b : ℕ) (h1 : b = 2 * a - 3) (h2 : a + b = 39) : a = 14 :=
by
  -- Sorry to skip the proof
  sorry

end find_smaller_number_l76_76162


namespace binomial_expansion_coefficient_and_sum_l76_76288

theorem binomial_expansion_coefficient_and_sum :
  (fincoeff ((x : ℚ) + (1/x))^5 1 = fincoeff (x + 1/x)^5 1 = 10) ∧
  (sum_all_coeff ((x : ℚ) + (1/x))^5 = (2 : ℚ)^5 = 32) :=
by
  sorry

end binomial_expansion_coefficient_and_sum_l76_76288


namespace least_common_multiple_of_first_ten_positive_integers_l76_76864

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76864


namespace least_common_multiple_first_ten_l76_76835

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76835


namespace two_numbers_max_product_l76_76338

theorem two_numbers_max_product :
  ∃ x y : ℝ, x - y = 4 ∧ x + y = 35 ∧ ∀ z w : ℝ, z - w = 4 → z + w = 35 → z * w ≤ x * y :=
by
  sorry

end two_numbers_max_product_l76_76338


namespace sin_cos_identity_tan_identity_l76_76690

open Real

namespace Trigonometry

variable (α : ℝ)

-- Given conditions
def given_conditions := (sin α + cos α = (1/5)) ∧ (0 < α) ∧ (α < π)

-- Prove that sin(α) * cos(α) = -12/25
theorem sin_cos_identity (h : given_conditions α) : sin α * cos α = -12/25 := 
sorry

-- Prove that tan(α) = -4/3
theorem tan_identity (h : given_conditions α) : tan α = -4/3 :=
sorry

end Trigonometry

end sin_cos_identity_tan_identity_l76_76690


namespace find_solution_l76_76422

-- Define the setup for the problem
variables (k x y : ℝ)

-- Conditions from the problem
def cond1 : Prop := x - y = 9 * k
def cond2 : Prop := x + y = 5 * k
def cond3 : Prop := 2 * x + 3 * y = 8

-- Proof statement combining all conditions to show the values of k, x, and y that satisfy them
theorem find_solution :
  cond1 k x y →
  cond2 k x y →
  cond3 x y →
  k = 1 ∧ x = 7 ∧ y = -2 := by
  sorry

end find_solution_l76_76422


namespace larger_number_l76_76042

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l76_76042


namespace find_alpha_beta_l76_76710

noncomputable def a : ℕ → ℕ
| n := let k := (√(8 * n + 1) - 1) / 2 in
       if n < (k * (k + 1)) / 2 then k else k + 1

noncomputable def S : ℕ → ℝ
| n := (∑ i in range (n + 1), a i : ℝ)

theorem find_alpha_beta :
  ∃ (α β : ℝ), (0 < α ∧ 0 < β) ∧ α = 3 / 2 ∧ β = (sqrt 2) / 3 ∧
  Tendsto (λ n, S n / (n : ℝ) ^ α) atTop (𝓝 β) := 
sorry

end find_alpha_beta_l76_76710


namespace probability_two_yellows_is_one_ninth_l76_76639

def total_marbles := 4 + 5 + 6
def probability_yellow : ℚ := 5 / total_marbles
def probability_two_yellows : ℚ := probability_yellow * probability_yellow

theorem probability_two_yellows_is_one_ninth :
  probability_two_yellows = 1 / 9 :=
sorry

end probability_two_yellows_is_one_ninth_l76_76639


namespace problem_I_solution_set_l76_76751

def f1 (x : ℝ) : ℝ := |2 * x| + |x - 1| -- since a = -1

theorem problem_I_solution_set :
  {x : ℝ | f1 x ≤ 4} = Set.Icc (-1 : ℝ) ((5 : ℝ) / 3) :=
sorry

end problem_I_solution_set_l76_76751


namespace andrena_has_more_dolls_than_debelyn_l76_76370

-- Definitions based on the given conditions
def initial_dolls_debelyn := 20
def initial_gift_debelyn_to_andrena := 2

def initial_dolls_christel := 24
def gift_christel_to_andrena := 5
def gift_christel_to_belissa := 3

def initial_dolls_belissa := 15
def gift_belissa_to_andrena := 4

-- Final number of dolls after exchanges
def final_dolls_debelyn := initial_dolls_debelyn - initial_gift_debelyn_to_andrena
def final_dolls_christel := initial_dolls_christel - gift_christel_to_andrena - gift_christel_to_belissa
def final_dolls_belissa := initial_dolls_belissa - gift_belissa_to_andrena + gift_christel_to_belissa
def final_dolls_andrena := initial_gift_debelyn_to_andrena + gift_christel_to_andrena + gift_belissa_to_andrena

-- Additional conditions
def andrena_more_than_christel := final_dolls_andrena = final_dolls_christel + 2
def belissa_equals_debelyn := final_dolls_belissa = final_dolls_debelyn

-- Proof Statement
theorem andrena_has_more_dolls_than_debelyn :
  andrena_more_than_christel →
  belissa_equals_debelyn →
  final_dolls_andrena - final_dolls_debelyn = 4 :=
by
  sorry

end andrena_has_more_dolls_than_debelyn_l76_76370


namespace slope_of_line_l76_76175

theorem slope_of_line (m : ℤ) (hm : (3 * m - 6) / (1 + m) = 12) : m = -2 := 
sorry

end slope_of_line_l76_76175


namespace total_birds_in_tree_l76_76458

theorem total_birds_in_tree (bluebirds cardinals swallows : ℕ) 
  (h1 : swallows = 2) 
  (h2 : swallows = bluebirds / 2) 
  (h3 : cardinals = 3 * bluebirds) : 
  swallows + bluebirds + cardinals = 18 := 
by 
  sorry

end total_birds_in_tree_l76_76458


namespace as_share_of_total_profit_l76_76426

-- Define variables and constants
def capital : ℝ := 1    -- Total capital
def total_profit : ℝ := 2300
def A_invest_ratio : ℝ := 1/6
def A_time_ratio : ℝ := 1/6
def B_invest_ratio : ℝ := 1/3
def B_time_ratio : ℝ := 1/3
def remaining_invest_ratio : ℝ := 1 - (A_invest_ratio + B_invest_ratio)
def total_time : ℝ := 1 -- Consider C for the whole time

-- Define the capital-time investments for A, B, and C
def A_capital_time := (A_invest_ratio * capital) * A_time_ratio
def B_capital_time := (B_invest_ratio * capital) * B_time_ratio
def C_capital_time := (remaining_invest_ratio * capital) * total_time

-- Sum the total capital-time investments
def total_capital_time := A_capital_time + B_capital_time + C_capital_time

-- Calculate A's share of the total profit
def A_share := (A_capital_time / total_capital_time) * total_profit

-- The formal problem statement to prove
theorem as_share_of_total_profit : A_share = 100 :=
by
  sorry

end as_share_of_total_profit_l76_76426


namespace probability_three_same_color_is_one_seventeenth_l76_76986

def standard_deck := {cards : Finset ℕ // cards.card = 52 ∧ ∃ reds blacks, reds.card = 26 ∧ blacks.card = 26 ∧ (reds ∪ blacks = cards)}

def num_ways_to_pick_3_same_color : ℕ :=
  (26 * 25 * 24) + (26 * 25 * 24)

def total_ways_to_pick_3 : ℕ :=
  52 * 51 * 50

def probability_top_three_same_color := (num_ways_to_pick_3_same_color / total_ways_to_pick_3 : ℚ)

theorem probability_three_same_color_is_one_seventeenth :
  probability_top_three_same_color = (1 / 17 : ℚ) := by sorry

end probability_three_same_color_is_one_seventeenth_l76_76986


namespace find_larger_number_l76_76045

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l76_76045


namespace ivar_total_water_needed_l76_76290

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end ivar_total_water_needed_l76_76290


namespace least_common_multiple_first_ten_l76_76826

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76826


namespace wire_length_l76_76204

theorem wire_length (r_sphere r_wire : ℝ) (h : ℝ) (V : ℝ)
  (h₁ : r_sphere = 24) (h₂ : r_wire = 16)
  (h₃ : V = 4 / 3 * Real.pi * r_sphere ^ 3)
  (h₄ : V = Real.pi * r_wire ^ 2 * h): 
  h = 72 := by
  -- we can use provided condition to show that h = 72, proof details omitted
  sorry

end wire_length_l76_76204


namespace lcm_first_ten_positive_integers_l76_76877

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76877


namespace intercepts_of_line_l76_76026

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := x / 4 - y / 3 = 1

-- Define the intercepts
def intercepts (x_intercept y_intercept : ℝ) : Prop :=
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept)

-- The problem statement: proving the values of intercepts
theorem intercepts_of_line :
  intercepts 4 (-3) :=
by
  sorry

end intercepts_of_line_l76_76026


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76843

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76843


namespace least_common_multiple_of_first_10_integers_l76_76952

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76952


namespace intersection_eq_l76_76409

noncomputable def A : Set ℕ := {1, 2, 3, 4}
noncomputable def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_eq : A ∩ B = {2, 3, 4} := 
by
  sorry

end intersection_eq_l76_76409


namespace integer_solutions_inequality_system_l76_76032

theorem integer_solutions_inequality_system :
  {x : ℤ | (x + 2 > 0) ∧ (2 * x - 1 ≤ 0)} = {-1, 0} := 
by
  -- proof goes here
  sorry

end integer_solutions_inequality_system_l76_76032


namespace fraction_books_sold_l76_76643

theorem fraction_books_sold :
  (∃ B F : ℝ, 3.50 * (B - 40) = 280.00000000000006 ∧ B ≠ 0 ∧ F = ((B - 40) / B) ∧ B = 120) → (F = 2 / 3) :=
by
  intro h
  obtain ⟨B, F, h1, h2, e⟩ := h
  sorry

end fraction_books_sold_l76_76643


namespace tony_will_have_4_dollars_in_change_l76_76613

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end tony_will_have_4_dollars_in_change_l76_76613


namespace lucky_ticket_N123456_l76_76965

def digits : List ℕ := [1, 2, 3, 4, 5, 6]

def is_lucky (digits : List ℕ) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, (f 1 (f (f 2 3) 4) * f 5 6) = 100

theorem lucky_ticket_N123456 : is_lucky digits :=
  sorry

end lucky_ticket_N123456_l76_76965


namespace number_of_points_in_star_polygon_l76_76368

theorem number_of_points_in_star_polygon :
  ∀ (n : ℕ) (D C : ℕ),
    (∀ i : ℕ, i < n → C = D - 15) →
    n * (D - (D - 15)) = 360 → n = 24 :=
by
  intros n D C h1 h2
  sorry

end number_of_points_in_star_polygon_l76_76368


namespace f_15_equals_227_l76_76418

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem f_15_equals_227 : f 15 = 227 := by
  sorry

end f_15_equals_227_l76_76418


namespace part1_equation_solution_part2_inequality_solution_l76_76964

theorem part1_equation_solution (x : ℝ) (h : x / (x - 1) = (x - 1) / (2 * (x - 1))) : 
  x = -1 :=
sorry

theorem part2_inequality_solution (x : ℝ) (h₁ : 5 * x - 1 > 3 * x - 4) (h₂ : - (1 / 3) * x ≤ 2 / 3 - x) : 
  -3 / 2 < x ∧ x ≤ 1 :=
sorry

end part1_equation_solution_part2_inequality_solution_l76_76964


namespace least_common_multiple_1_to_10_l76_76903

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76903


namespace correct_equation_l76_76320

noncomputable def team_a_initial := 96
noncomputable def team_b_initial := 72
noncomputable def team_b_final (x : ℕ) := team_b_initial - x
noncomputable def team_a_final (x : ℕ) := team_a_initial + x

theorem correct_equation (x : ℕ) : 
  (1 / 3 : ℚ) * (team_a_final x) = (team_b_final x) := 
  sorry

end correct_equation_l76_76320


namespace subtraction_of_bases_l76_76524

def base8_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 7^2 + ((n % 100) / 10) * 7^1 + (n % 10) * 7^0

theorem subtraction_of_bases :
  base8_to_base10 343 - base7_to_base10 265 = 82 :=
by
  sorry

end subtraction_of_bases_l76_76524


namespace subset_123_12_false_l76_76192

-- Definitions derived from conditions
def is_int (x : ℤ) := true
def subset_123_12 (A B : Set ℕ) := A = {1, 2, 3} ∧ B = {1, 2}
def intersection_empty {A B : Set ℕ} (hA : A = {1, 2}) (hB : B = ∅) := (A ∩ B = ∅)
def union_nat_real {A B : Set ℝ} (hA : Set.univ ⊆ A) (hB : Set.univ ⊆ B) := (A ∪ B)

-- The mathematically equivalent proof problem
theorem subset_123_12_false (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {1, 2}):
  ¬ (A ⊆ B) :=
by
  sorry

end subset_123_12_false_l76_76192


namespace f_2023_l76_76585

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_all : ∀ x : ℕ, f x ≠ 0 → (x ≥ 0)
axiom f_one : f 1 = 1
axiom f_functional_eq : ∀ a b : ℕ, f (a + b) = f a + f b - 3 * f (a * b)

theorem f_2023 : f 2023 = -(2^2022 - 1) := sorry

end f_2023_l76_76585


namespace projection_sum_of_squares_l76_76587

theorem projection_sum_of_squares (a : ℝ) (α β γ : ℝ) 
    (h1 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) 
    (h2 : (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = 2) :
    4 * a^2 * ((Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2) = 8 * a^2 := 
by
  sorry

end projection_sum_of_squares_l76_76587


namespace find_number_l76_76018

theorem find_number (x : ℤ) (h : ((x * 2) - 37 + 25) / 8 = 5) : x = 26 :=
sorry  -- Proof placeholder

end find_number_l76_76018


namespace processing_plant_growth_eq_l76_76066

-- Definition of the conditions given in the problem
def initial_amount : ℝ := 10
def november_amount : ℝ := 13
def growth_rate (x : ℝ) : ℝ := initial_amount * (1 + x)^2

-- Lean theorem statement to prove the equation
theorem processing_plant_growth_eq (x : ℝ) : 
  growth_rate x = november_amount ↔ initial_amount * (1 + x)^2 = 13 := 
by
  sorry

end processing_plant_growth_eq_l76_76066


namespace ratio_solves_for_x_l76_76727

theorem ratio_solves_for_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
by
  -- The formal proof would go here.
  sorry

end ratio_solves_for_x_l76_76727


namespace product_of_roots_l76_76684

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end product_of_roots_l76_76684


namespace speed_ratio_l76_76360

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end speed_ratio_l76_76360


namespace quadratic_does_not_pass_third_quadrant_l76_76709

-- Definitions of the functions
def linear_function (a b x : ℝ) : ℝ := -a * x + b
def quadratic_function (a b x : ℝ) : ℝ := -a * x^2 + b * x

-- Conditions
variables (a b : ℝ)
axiom a_nonzero : a ≠ 0
axiom passes_first_third_fourth : ∀ x, (linear_function a b x > 0 ∧ x > 0) ∨ (linear_function a b x < 0 ∧ x < 0) ∨ (linear_function a b x < 0 ∧ x > 0)

-- Theorem stating the problem
theorem quadratic_does_not_pass_third_quadrant :
  ¬ (∃ x, quadratic_function a b x < 0 ∧ x < 0) := 
sorry

end quadratic_does_not_pass_third_quadrant_l76_76709


namespace product_of_odd_integers_l76_76485

theorem product_of_odd_integers :
  let odd_factorial_product := ∏ i in finset.filter (λ x : ℕ, x % 2 = 1) (finset.range 10000), i
  in odd_factorial_product = 10000.factorial / (2^5000 * 5000.factorial) :=
by
  sorry 

end product_of_odd_integers_l76_76485


namespace eccentricity_of_ellipse_l76_76562

theorem eccentricity_of_ellipse (a b c e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 - b^2)) 
  (h4 : b = Real.sqrt 3 * c) : e = 1/2 :=
by
  sorry

end eccentricity_of_ellipse_l76_76562


namespace sum_of_numbers_l76_76030

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : (1 / x) = 3 * (1 / y)) :
  x + y = 8 :=
sorry

end sum_of_numbers_l76_76030


namespace collinear_iff_linear_combination_l76_76304

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V) (k : ℝ)

theorem collinear_iff_linear_combination (O A B C : V) (k : ℝ) :
  (C = k • A + (1 - k) • B) ↔ ∃ (k' : ℝ), C - B = k' • (A - B) :=
sorry

end collinear_iff_linear_combination_l76_76304


namespace sequence_count_21_l76_76413

-- Define the conditions and the problem
def valid_sequence (n : ℕ) : ℕ :=
  if n = 21 then 114 else sorry

theorem sequence_count_21 : valid_sequence 21 = 114 :=
  by sorry

end sequence_count_21_l76_76413


namespace farmer_shipped_pomelos_in_dozens_l76_76001

theorem farmer_shipped_pomelos_in_dozens :
  let pomelos_per_box := 240 / 10 in
  let dozens_per_box := pomelos_per_box / 12 in
  let total_boxes := 10 + 20 in
  total_boxes * dozens_per_box = 60 :=
by
  have pomelos_per_box_eq : pomelos_per_box = 24 := by norm_num
  have dozens_per_box_eq : dozens_per_box = 2 := by norm_num
  have total_boxes_eq : total_boxes = 30 := by norm_num
  rw [pomelos_per_box_eq, dozens_per_box_eq, total_boxes_eq]
  norm_num
  sorry

end farmer_shipped_pomelos_in_dozens_l76_76001


namespace least_common_multiple_first_ten_l76_76824

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76824


namespace train_speed_comparison_l76_76468

variables (V_A V_B : ℝ)

open Classical

theorem train_speed_comparison
  (distance_AB : ℝ)
  (h_distance : distance_AB = 360)
  (h_time_limit : V_A ≤ 72)
  (h_meeting_time : 3 * V_A + 2 * V_B > 360) :
  V_B > V_A :=
by {
  sorry
}

end train_speed_comparison_l76_76468


namespace fraction_is_irreducible_l76_76993

theorem fraction_is_irreducible :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16 : ℚ) / 
   (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by 
  sorry

end fraction_is_irreducible_l76_76993


namespace simplify_and_evaluate_expr_l76_76314

noncomputable def original_expr (x : ℝ) : ℝ := 
  ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))

noncomputable def x_val : ℝ := Real.sqrt 2 - 1

theorem simplify_and_evaluate_expr : original_expr x_val = 1 - (Real.sqrt 2) / 2 :=
  by
    sorry

end simplify_and_evaluate_expr_l76_76314


namespace least_common_multiple_first_ten_l76_76833

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76833


namespace perpendicular_line_equation_l76_76786

theorem perpendicular_line_equation (x y : ℝ) (h : 2 * x + y + 3 = 0) (hx : ∃ c : ℝ, x - 2 * y + c = 0) :
  (c = 7 ↔ ∀ p : ℝ × ℝ, p = (-1, 3) → (p.1 - 2 * p.2 + 7 = 0)) :=
sorry

end perpendicular_line_equation_l76_76786


namespace length_of_room_l76_76788

theorem length_of_room (width : ℝ) (cost_per_sq_meter : ℝ) (total_cost : ℝ) (L : ℝ) 
  (h_width : width = 2.75)
  (h_cost_per_sq_meter : cost_per_sq_meter = 600)
  (h_total_cost : total_cost = 10725)
  (h_area_cost_eq : total_cost = L * width * cost_per_sq_meter) : 
  L = 6.5 :=
by 
  simp [h_width, h_cost_per_sq_meter, h_total_cost, h_area_cost_eq] at *
  sorry

end length_of_room_l76_76788


namespace dice_sum_not_possible_l76_76046

   theorem dice_sum_not_possible (a b c d : ℕ) :
     (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → (1 ≤ d ∧ d ≤ 6) →
     (a * b * c * d = 360) → ¬ (a + b + c + d = 20) :=
   by
     intros ha hb hc hd prod eq_sum
     -- Proof skipped
     sorry
   
end dice_sum_not_possible_l76_76046


namespace distinct_solutions_diff_l76_76148

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l76_76148


namespace least_common_multiple_first_ten_l76_76828

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76828


namespace domain_of_f_l76_76620

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f : ∀ x : ℝ, x ≠ 8 ↔ ∃ y : ℝ, f x = y :=
  by admit

end domain_of_f_l76_76620


namespace arithmetic_sequence_example_l76_76573

theorem arithmetic_sequence_example (a : ℕ → ℝ) (h : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) (h₁ : a 1 + a 19 = 10) : a 10 = 5 :=
by
  sorry

end arithmetic_sequence_example_l76_76573


namespace evaluate_expression_l76_76088

theorem evaluate_expression :
  let x := 1.93
  let y := 51.3
  let z := 0.47
  Float.round (x * (y + z)) = 100 := by
sorry

end evaluate_expression_l76_76088


namespace lcm_first_ten_positive_integers_l76_76879

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76879


namespace main_diagonal_distinct_l76_76429

open Matrix

-- Define the problem in Lean 4
theorem main_diagonal_distinct
  (n : ℕ) 
  (a : Matrix (Fin (2 * n + 1)) (Fin (2 * n + 1)) ℕ)
  (sym : ∀ i j, a i j = a j i)
  (rows : ∀ i, ∃! perm : Fin (2 * n + 1) → Fin (2 * n + 1), ∀ j, a i j = perm j + 1 ∧ perm j + 1 ∈ Fin (2 * n + 1))
  (cols : ∀ j, ∃! perm : Fin (2 * n + 1) → Fin (2 * n + 1), ∀ i, a i j = perm i + 1 ∧ perm i + 1 ∈ Fin (2 * n + 1)):
  ∀ i j, i ≠ j → a i i ≠ a j j :=
by
  sorry

end main_diagonal_distinct_l76_76429


namespace subtracted_amount_l76_76353

theorem subtracted_amount (A N : ℝ) (h₁ : N = 200) (h₂ : 0.95 * N - A = 178) : A = 12 :=
by
  sorry

end subtracted_amount_l76_76353


namespace Benny_spent_95_dollars_l76_76667

theorem Benny_spent_95_dollars
    (amount_initial : ℕ)
    (amount_left : ℕ)
    (amount_spent : ℕ) :
    amount_initial = 120 →
    amount_left = 25 →
    amount_spent = amount_initial - amount_left →
    amount_spent = 95 :=
by
  intros h_initial h_left h_spent
  rw [h_initial, h_left] at h_spent
  exact h_spent

end Benny_spent_95_dollars_l76_76667


namespace ratio_of_volumes_l76_76486

theorem ratio_of_volumes (rC hC rD hD : ℝ) (h1 : rC = 10) (h2 : hC = 25) (h3 : rD = 25) (h4 : hD = 10) : 
  (1/3 * Real.pi * rC^2 * hC) / (1/3 * Real.pi * rD^2 * hD) = 2 / 5 :=
by
  sorry

end ratio_of_volumes_l76_76486


namespace next_podcast_duration_l76_76310

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end next_podcast_duration_l76_76310


namespace topsoil_cost_l76_76478

theorem topsoil_cost :
  let cubic_yard_to_cubic_foot := 27
  let cubic_feet_in_5_cubic_yards := 5 * cubic_yard_to_cubic_foot
  let cost_per_cubic_foot := 6
  let total_cost := cubic_feet_in_5_cubic_yards * cost_per_cubic_foot
  total_cost = 810 :=
by
  sorry

end topsoil_cost_l76_76478


namespace solve_system_l76_76461

theorem solve_system (x y z : ℝ) 
  (h1 : x^3 - y = 6)
  (h2 : y^3 - z = 6)
  (h3 : z^3 - x = 6) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_system_l76_76461


namespace percentage_students_on_trip_l76_76112

variable (total_students : ℕ)
variable (students_more_than_100 : ℕ)
variable (students_on_trip : ℕ)
variable (percentage_more_than_100 : ℝ)
variable (percentage_not_more_than_100 : ℝ)

-- Given conditions
def condition_1 := percentage_more_than_100 = 0.16
def condition_2 := percentage_not_more_than_100 = 0.75

-- The final proof statement
theorem percentage_students_on_trip :
  percentage_more_than_100 * (total_students : ℝ) /
  ((1 - percentage_not_more_than_100)) / (total_students : ℝ) * 100 = 64 :=
by
  sorry

end percentage_students_on_trip_l76_76112


namespace least_positive_integer_divisible_by_first_ten_l76_76868

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76868


namespace distinct_solutions_difference_l76_76152

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l76_76152


namespace least_common_multiple_first_ten_l76_76829

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76829


namespace find_side2_l76_76171

-- Define the given conditions
def perimeter : ℕ := 160
def side1 : ℕ := 40
def side3 : ℕ := 70

-- Define the second side as a variable
def side2 : ℕ := perimeter - side1 - side3

-- State the theorem to be proven
theorem find_side2 : side2 = 50 := by
  -- We skip the proof here with sorry
  sorry

end find_side2_l76_76171


namespace no_perfect_square_E_l76_76579

noncomputable def E (x : ℝ) : ℤ :=
  round x

theorem no_perfect_square_E (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, E (n + Real.sqrt n) = k * k) :=
  sorry

end no_perfect_square_E_l76_76579


namespace find_a_l76_76701

theorem find_a (a : ℝ) : 
  (∃ (a : ℝ), a * 15 + 6 = -9) → a = -1 :=
by
  intro h
  sorry

end find_a_l76_76701


namespace no_x_intersections_geometric_sequence_l76_76264

theorem no_x_intersections_geometric_sequence (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a * c > 0) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) = false :=
by
  sorry

end no_x_intersections_geometric_sequence_l76_76264


namespace total_cost_of_purchases_l76_76137

def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

theorem total_cost_of_purchases : cost_cat_toy + cost_cage = 21.95 := by
  -- skipping the proof
  sorry

end total_cost_of_purchases_l76_76137


namespace dog_food_weighs_more_l76_76758

def weight_in_ounces (weight_in_pounds: ℕ) := weight_in_pounds * 16
def total_food_weight (cat_food_bags dog_food_bags: ℕ) (cat_food_pounds dog_food_pounds: ℕ) :=
  (cat_food_bags * weight_in_ounces cat_food_pounds) + (dog_food_bags * weight_in_ounces dog_food_pounds)

theorem dog_food_weighs_more
  (cat_food_bags: ℕ) (cat_food_pounds: ℕ) (dog_food_bags: ℕ) (total_weight_ounces: ℕ) (ounces_in_pound: ℕ)
  (H1: cat_food_bags * weight_in_ounces cat_food_pounds = 96)
  (H2: total_food_weight cat_food_bags dog_food_bags cat_food_pounds dog_food_pounds = total_weight_ounces)
  (H3: ounces_in_pound = 16) :
  dog_food_pounds - cat_food_pounds = 2 := 
by sorry

end dog_food_weighs_more_l76_76758


namespace least_common_multiple_1_to_10_l76_76904

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76904


namespace sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l76_76122

theorem sin_C_eq_sqrt14_div_8 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  sinC = Real.sqrt 14 / 8 := 
by
  -- Proof is omitted
  sorry

theorem area_triangle_eq_sqrt7_div_4 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  let cosC := Real.sqrt (1 - sinC^2)
  let sinA := sinB * cosC + cosB * sinC
  let area := 1 / 2 * b * c * sinA
  area = Real.sqrt 7 / 4 := 
by
  -- Proof is omitted
  sorry

end sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l76_76122


namespace bowling_ball_weight_l76_76243

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end bowling_ball_weight_l76_76243


namespace harvest_weeks_l76_76446

/-- Lewis earns $403 every week during a certain number of weeks of harvest. 
If he has to pay $49 rent every week, and he earns $93,899 during the harvest season, 
we need to prove that the number of weeks in the harvest season is 265. --/
theorem harvest_weeks 
  (E : ℕ) (R : ℕ) (T : ℕ) (W : ℕ) 
  (hE : E = 403) (hR : R = 49) (hT : T = 93899) 
  (hW : W = 265) : 
  W = (T / (E - R)) := 
by sorry

end harvest_weeks_l76_76446


namespace carnations_in_last_three_bouquets_l76_76479

/--
Trevor buys six bouquets of carnations.
In the first bouquet, there are 9.5 carnations.
In the second bouquet, there are 14.25 carnations.
In the third bouquet, there are 18.75 carnations.
The average number of carnations in all six bouquets is 16.
Prove that the total number of carnations in the fourth, fifth, and sixth bouquets combined is 53.5.
-/
theorem carnations_in_last_three_bouquets:
  let bouquet1 := 9.5
  let bouquet2 := 14.25
  let bouquet3 := 18.75
  let total_bouquets := 6
  let average_per_bouquet := 16
  let total_carnations := average_per_bouquet * total_bouquets
  let remaining_carnations := total_carnations - (bouquet1 + bouquet2 + bouquet3)
  remaining_carnations = 53.5 :=
by
  sorry

end carnations_in_last_three_bouquets_l76_76479


namespace train_crossing_time_l76_76072

theorem train_crossing_time
    (length_of_train : ℕ)
    (speed_of_train_kmph : ℕ)
    (length_of_bridge : ℕ)
    (h_train_length : length_of_train = 160)
    (h_speed_kmph : speed_of_train_kmph = 45)
    (h_bridge_length : length_of_bridge = 215)
  : length_of_train + length_of_bridge / ((speed_of_train_kmph * 1000) / 3600) = 30 :=
by
  rw [h_train_length, h_speed_kmph, h_bridge_length]
  norm_num
  sorry

end train_crossing_time_l76_76072


namespace sea_creatures_lost_l76_76412

theorem sea_creatures_lost (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34) 
  (h2 : seashells = 21) 
  (h3 : snails = 29) 
  (h4 : items_left = 59) : 
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l76_76412


namespace interval_of_segmentation_l76_76476

-- Define the population size and sample size as constants.
def population_size : ℕ := 2000
def sample_size : ℕ := 40

-- State the theorem for the interval of segmentation.
theorem interval_of_segmentation :
  population_size / sample_size = 50 :=
sorry

end interval_of_segmentation_l76_76476


namespace evaluate_f_l76_76419

def f (x : ℝ) : ℝ := x^2 + 4*x - 3

theorem evaluate_f (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 :=
by 
  -- The proof is omitted
  sorry

end evaluate_f_l76_76419


namespace proportionality_problem_l76_76462

noncomputable def find_x (z w : ℝ) (k : ℝ) : ℝ :=
  k / (z^(3/2) * w^2)

theorem proportionality_problem :
  ∃ k : ℝ, 
    (find_x 16 2 k = 5) ∧
    (find_x 64 4 k = 5 / 32) :=
by
  sorry

end proportionality_problem_l76_76462


namespace lcm_first_ten_integers_l76_76810

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76810


namespace lcm_of_18_and_36_l76_76187

theorem lcm_of_18_and_36 : Nat.lcm 18 36 = 36 := 
by 
  sorry

end lcm_of_18_and_36_l76_76187


namespace min_moves_is_22_l76_76312

def casket_coins : List ℕ := [9, 17, 12, 5, 18, 10, 20]

def target_coins (total_caskets : ℕ) (total_coins : ℕ) : ℕ :=
  total_coins / total_caskets

def total_caskets : ℕ := 7

def total_coins (coins : List ℕ) : ℕ :=
  coins.foldr (· + ·) 0

noncomputable def min_moves_to_equalize (coins : List ℕ) (target : ℕ) : ℕ := sorry

theorem min_moves_is_22 :
  min_moves_to_equalize casket_coins (target_coins total_caskets (total_coins casket_coins)) = 22 :=
sorry

end min_moves_is_22_l76_76312


namespace floor_sqrt_27_squared_eq_25_l76_76675

theorem floor_sqrt_27_squared_eq_25 :
  (⌊Real.sqrt 27⌋)^2 = 25 :=
by
  have H1 : 5 < Real.sqrt 27 := sorry
  have H2 : Real.sqrt 27 < 6 := sorry
  have floor_sqrt_27_eq_5 : ⌊Real.sqrt 27⌋ = 5 := sorry
  rw floor_sqrt_27_eq_5
  norm_num

end floor_sqrt_27_squared_eq_25_l76_76675


namespace find_m_value_l76_76734

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end find_m_value_l76_76734


namespace P_neg2_eq_19_l76_76005

noncomputable def P : ℝ → ℝ := sorry  -- Definition of the polynomial P(x)

axiom degree_P : ∃ (d : ℕ), d = 4 ∧ ∀ (x : ℝ) (hx : P x ≠ 0), nat_degree (polynomial.of_real (P x)) = d
axiom P_0 : P 0 = 1
axiom P_1 : P 1 = 1
axiom P_2 : P 2 = 4
axiom P_3 : P 3 = 9
axiom P_4 : P 4 = 16

theorem P_neg2_eq_19 : P (-2) = 19 :=
by
  sorry

end P_neg2_eq_19_l76_76005


namespace max_value_expr_l76_76394

theorem max_value_expr (x : ℝ) : 
  ( x ^ 6 / (x ^ 12 + 3 * x ^ 8 - 6 * x ^ 6 + 12 * x ^ 4 + 36) <= 1/18 ) :=
by
  sorry

end max_value_expr_l76_76394


namespace negation_exists_l76_76169

theorem negation_exists (x : ℝ) (h : x ≥ 0) : (¬ (∀ x : ℝ, (x ≥ 0) → (2^x > x^2))) ↔ (∃ x₀ : ℝ, (x₀ ≥ 0) ∧ (2 ^ x₀ ≤ x₀^2)) := by
  sorry

end negation_exists_l76_76169


namespace adjust_collection_amount_l76_76224

/-- Define the error caused by mistaking half-dollars for dollars -/
def halfDollarError (x : ℕ) : ℤ := 50 * x

/-- Define the error caused by mistaking quarters for nickels -/
def quarterError (x : ℕ) : ℤ := 20 * x

/-- Define the total error based on the given conditions -/
def totalError (x : ℕ) : ℤ := halfDollarError x - quarterError x

theorem adjust_collection_amount (x : ℕ) : totalError x = 30 * x := by
  sorry

end adjust_collection_amount_l76_76224


namespace cut_rectangle_from_square_l76_76627

theorem cut_rectangle_from_square
  (side : ℝ) (length : ℝ) (width : ℝ) 
  (square_area_eq : side * side = 100)
  (rectangle_area_eq : length * width = 90)
  (ratio_length_width : 5 * width = 3 * length) : 
  ¬ (length ≤ side ∧ width ≤ side) :=
by 
  sorry

end cut_rectangle_from_square_l76_76627


namespace find_x0_range_l76_76501

variable {x y x0 : ℝ}

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def angle_condition (x0 : ℝ) : Prop :=
  let OM := Real.sqrt (x0^2 + 3)
  OM ≤ 2

theorem find_x0_range (h1 : circle_eq x y) (h2 : angle_condition x0) :
  -1 ≤ x0 ∧ x0 ≤ 1 := 
sorry

end find_x0_range_l76_76501


namespace smallest_value_of_n_l76_76241

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end smallest_value_of_n_l76_76241


namespace find_digits_l76_76372

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l76_76372


namespace age_problem_l76_76414

theorem age_problem (x y : ℕ) 
  (h1 : 3 * x = 4 * y) 
  (h2 : 3 * y - x = 140) : x = 112 ∧ y = 84 := 
by 
  sorry

end age_problem_l76_76414


namespace medals_awarded_correctly_l76_76129

def totalWaysToAwardMedals (totalAthletes europeans asians: ℕ) :=
  if totalAthletes = 10 ∧ europeans = 4 ∧ asians = 6 then
    let case1 := 6 * 5 * 4
    let case2 := 4 * 3 * (6 * 5)
    let case3 := (Nat.choose 4 2) * 3 * 6
    case1 + case2 + case3
  else 0

theorem medals_awarded_correctly :
  totalWaysToAwardMedals 10 4 6 = 588 := by
    simp [totalWaysToAwardMedals]
    sorry

end medals_awarded_correctly_l76_76129


namespace papers_left_after_giving_away_l76_76756

variable (x : ℕ)

-- Given conditions:
def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41
def total_initial_sheets := sheets_in_desk + sheets_in_backpack

-- Prove that Maria has 91 - x sheets left after giving away x sheets
theorem papers_left_after_giving_away (h : total_initial_sheets = 91) : 
  ∀ d b : ℕ, d = sheets_in_desk → b = sheets_in_backpack → 91 - x = total_initial_sheets - x :=
by
  sorry

end papers_left_after_giving_away_l76_76756


namespace find_second_divisor_l76_76333

theorem find_second_divisor :
  ∃ y : ℝ, (320 / 2) / y = 53.33 ∧ y = 160 / 53.33 :=
by
  sorry

end find_second_divisor_l76_76333


namespace maximum_ab_minimum_frac_minimum_exp_l76_76254

variable {a b : ℝ}

theorem maximum_ab (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  ab <= 1/8 :=
sorry

theorem minimum_frac (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2/a + 1/b >= 8 :=
sorry

theorem minimum_exp (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2^a + 4^b >= 2 * Real.sqrt 2 :=
sorry

end maximum_ab_minimum_frac_minimum_exp_l76_76254


namespace hydrogen_to_oxygen_ratio_l76_76480

theorem hydrogen_to_oxygen_ratio (total_mass_water mass_hydrogen mass_oxygen : ℝ) 
(h1 : total_mass_water = 117)
(h2 : mass_hydrogen = 13)
(h3 : mass_oxygen = total_mass_water - mass_hydrogen) :
(mass_hydrogen / mass_oxygen) = 1 / 8 := 
sorry

end hydrogen_to_oxygen_ratio_l76_76480


namespace ratio_of_areas_l76_76977

variable (A B : ℝ)

-- Conditions
def total_area := A + B = 700
def smaller_part_area := B = 315

-- Problem Statement
theorem ratio_of_areas (h_total : total_area A B) (h_small : smaller_part_area B) :
  (A - B) / ((A + B) / 2) = 1 / 5 := by
sorry

end ratio_of_areas_l76_76977


namespace least_common_multiple_of_first_ten_integers_l76_76928

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76928


namespace solidConstruction_l76_76590

-- Definitions
structure Solid where
  octagonal_faces : Nat
  hexagonal_faces : Nat
  square_faces : Nat

-- Conditions
def solidFromCube (S : Solid) : Prop :=
  S.octagonal_faces = 6 ∧ S.hexagonal_faces = 8 ∧ S.square_faces = 12

def circumscribedSphere (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

def solidFromOctahedron (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

-- Theorem statement
theorem solidConstruction {S : Solid} :
  solidFromCube S ∧ circumscribedSphere S → solidFromOctahedron S :=
by
  sorry

end solidConstruction_l76_76590


namespace f_increasing_on_Ioo_l76_76097

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_increasing_on_Ioo : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2 :=
by sorry

end f_increasing_on_Ioo_l76_76097


namespace find_a1_general_term_sum_of_terms_l76_76694

-- Given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom h_condition : ∀ n, S n = (3 / 2) * a n - (1 / 2)

-- Specific condition for finding a1
axiom h_S1_eq_1 : S 1 = 1

-- Prove statements
theorem find_a1 : a 1 = 1 :=
by
  sorry

theorem general_term (n : ℕ) : n ≥ 1 → a n = 3 ^ (n - 1) :=
by
  sorry

theorem sum_of_terms (n : ℕ) : n ≥ 1 → S n = (3 ^ n - 1) / 2 :=
by
  sorry

end find_a1_general_term_sum_of_terms_l76_76694


namespace douglas_vote_percentage_is_66_l76_76128

noncomputable def percentDouglasVotes (v : ℝ) : ℝ :=
  let votesX := 0.74 * (2 * v)
  let votesY := 0.5000000000000002 * v
  let totalVotes := 3 * v
  let totalDouglasVotes := votesX + votesY
  (totalDouglasVotes / totalVotes) * 100

theorem douglas_vote_percentage_is_66 :
  ∀ v : ℝ, percentDouglasVotes v = 66 := 
by
  intros v
  unfold percentDouglasVotes
  sorry

end douglas_vote_percentage_is_66_l76_76128


namespace find_digits_l76_76371

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l76_76371


namespace cube_vertex_plane_distance_l76_76202

theorem cube_vertex_plane_distance
  (d : ℝ)
  (h_dist : d = 9 - Real.sqrt 186)
  (h7 : ∀ (a b c  : ℝ), a^2 + b^2 + c^2 = 1 → 64 * (a^2 + b^2 + c^2) = 64)
  (h8 : ∀ (d : ℝ), 3 * d^2 - 54 * d + 181 = 0) :
  ∃ (p q r : ℕ), 
    p = 27 ∧ q = 186 ∧ r = 3 ∧ (p + q + r < 1000) ∧ (d = (p - Real.sqrt q) / r) := 
  by
    sorry

end cube_vertex_plane_distance_l76_76202


namespace eval_expression_l76_76534

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end eval_expression_l76_76534


namespace prob_two_girls_l76_76656

variable (Pboy Pgirl : ℝ)

-- Conditions
def prob_boy : Prop := Pboy = 1 / 2
def prob_girl : Prop := Pgirl = 1 / 2

-- The theorem to be proven
theorem prob_two_girls (h₁ : prob_boy Pboy) (h₂ : prob_girl Pgirl) : (Pgirl * Pgirl) = 1 / 4 :=
by
  sorry

end prob_two_girls_l76_76656


namespace lcm_first_ten_integers_l76_76811

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76811


namespace washing_machine_capacity_l76_76619

-- Define the conditions:
def shirts : ℕ := 39
def sweaters : ℕ := 33
def loads : ℕ := 9
def total_clothes : ℕ := shirts + sweaters -- which is 72

-- Define the statement to be proved:
theorem washing_machine_capacity : ∃ x : ℕ, loads * x = total_clothes ∧ x = 8 :=
by
  -- proof to be completed
  sorry

end washing_machine_capacity_l76_76619


namespace first_term_of_arithmetic_sequence_l76_76399

theorem first_term_of_arithmetic_sequence :
  ∃ (a_1 : ℤ), ∀ (d n : ℤ), d = 3 / 4 ∧ n = 30 ∧ a_n = 63 / 4 → a_1 = -6 := by
  sorry

end first_term_of_arithmetic_sequence_l76_76399


namespace units_digit_two_pow_2010_l76_76345

-- Conditions from part a)
def two_power_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case will not occur due to modulo operation

-- Question translated to a proof problem
theorem units_digit_two_pow_2010 : (two_power_units_digit 2010) = 4 :=
by 
  -- Proof would go here
  sorry

end units_digit_two_pow_2010_l76_76345


namespace no_three_partition_exists_l76_76081

/-- Define the partitioning property for three subsets -/
def partitions (A B C : Set ℤ) : Prop :=
  ∀ n : ℤ, (n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ (n ∈ A ↔ n-50 ∈ B ∧ n+1987 ∈ C) ∧ (n-50 ∈ A ∨ n-50 ∈ B ∨ n-50 ∈ C) ∧ (n-50 ∈ B ↔ n-50-50 ∈ A ∧ n-50+1987 ∈ C) ∧ (n+1987 ∈ A ∨ n+1987 ∈ B ∨ n+1987 ∈ C) ∧ (n+1987 ∈ C ↔ n+1987-50 ∈ A ∧ n+1987+1987 ∈ B)

/-- The main theorem stating that no such partition is possible -/
theorem no_three_partition_exists :
  ¬∃ A B C : Set ℤ, partitions A B C :=
sorry

end no_three_partition_exists_l76_76081


namespace least_positive_integer_divisible_by_first_ten_l76_76867

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76867


namespace find_c_value_l76_76281

theorem find_c_value 
  (a b c : ℝ)
  (h_a : a = 5 / 2)
  (h_b : b = 17)
  (roots : ∀ x : ℝ, x = (-b + Real.sqrt 23) / 5 ∨ x = (-b - Real.sqrt 23) / 5)
  (discrim_eq : ∀ c : ℝ, b ^ 2 - 4 * a * c = 23) :
  c = 26.6 := by
  sorry

end find_c_value_l76_76281


namespace g_h_2_equals_584_l76_76111

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end g_h_2_equals_584_l76_76111


namespace range_a_plus_2b_l76_76405

-- Define the function f(x) = |log x|
def f (x : ℝ) : ℝ := abs (Real.log x)
  
-- Main theorem statement
theorem range_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 3 < a + 2 * b :=
by
  -- Proof needs to be provided
  sorry

end range_a_plus_2b_l76_76405


namespace incorrect_operation_l76_76625

theorem incorrect_operation (a : ℝ) : ¬ (a^3 + a^3 = 2 * a^6) :=
by
  sorry

end incorrect_operation_l76_76625


namespace solutions_of_quadratic_eq_l76_76331

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end solutions_of_quadratic_eq_l76_76331


namespace hyperbola_distance_to_directrix_l76_76402

def hyperbola_has_distance_to_directrix_one (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  let c := 4 in
  let asymptote_ratio := b / a = Real.sqrt 3 in
  let focal_to_center_relation := c * c = a * a + b * b in
  let directrix_distance := (a * a) / c = 1 in
  asymptote_ratio ∧ focal_to_center_relation ∧ directrix_distance

theorem hyperbola_distance_to_directrix :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ hyperbola_has_distance_to_directrix_one a b :=
by
  use 2, 2 * Real.sqrt 3
  sorry

end hyperbola_distance_to_directrix_l76_76402


namespace acute_angle_inequality_l76_76305

theorem acute_angle_inequality (a b : ℝ) (α β : ℝ) (γ : ℝ) (h : γ < π / 2) :
  (a^2 + b^2) * Real.cos (α - β) ≤ 2 * a * b :=
sorry

end acute_angle_inequality_l76_76305


namespace tony_will_have_4_dollars_in_change_l76_76612

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end tony_will_have_4_dollars_in_change_l76_76612


namespace number_of_outfits_l76_76120

-- Define the number of shirts, pants, and jacket options.
def shirts : Nat := 8
def pants : Nat := 5
def jackets : Nat := 3

-- The theorem statement for the total number of outfits.
theorem number_of_outfits : shirts * pants * jackets = 120 := 
by
  sorry

end number_of_outfits_l76_76120


namespace minimum_value_of_m_plus_n_l76_76259

noncomputable def m (a b : ℝ) : ℝ := b + (1 / a)
noncomputable def n (a b : ℝ) : ℝ := a + (1 / b)

theorem minimum_value_of_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 1) :
  m a b + n a b = 4 :=
sorry

end minimum_value_of_m_plus_n_l76_76259


namespace abc_proof_l76_76416

noncomputable def abc_value (a b c : ℝ) : ℝ :=
  a * b * c

theorem abc_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b = 24 * (3 ^ (1 / 3)))
  (h5 : a * c = 40 * (3 ^ (1 / 3)))
  (h6 : b * c = 16 * (3 ^ (1 / 3))) : 
  abc_value a b c = 96 * (15 ^ (1 / 2)) :=
sorry

end abc_proof_l76_76416


namespace least_common_multiple_of_first_10_integers_l76_76955

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76955


namespace f_pi_over_4_l76_76103

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem f_pi_over_4 (ω φ : ℝ) (h : ω ≠ 0) 
  (symm : ∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) : 
  f ω φ (π / 4) = 2 ∨ f ω φ (π / 4) = -2 := 
by 
  sorry

end f_pi_over_4_l76_76103


namespace blue_paint_needed_l76_76205

theorem blue_paint_needed (total_cans : ℕ) (blue_ratio : ℕ) (yellow_ratio : ℕ)
  (h_ratio: blue_ratio = 5) (h_yellow_ratio: yellow_ratio = 3) (h_total: total_cans = 45) : 
  ⌊total_cans * (blue_ratio : ℝ) / (blue_ratio + yellow_ratio)⌋ = 28 :=
by
  sorry

end blue_paint_needed_l76_76205


namespace prime_gt_three_square_mod_twelve_l76_76768

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end prime_gt_three_square_mod_twelve_l76_76768


namespace speed_of_boat_is_correct_l76_76978

theorem speed_of_boat_is_correct (t : ℝ) (V_b : ℝ) (V_s : ℝ) 
  (h1 : V_s = 19) 
  (h2 : ∀ t, (V_b - V_s) * (2 * t) = (V_b + V_s) * t) :
  V_b = 57 :=
by
  -- Proof will go here
  sorry

end speed_of_boat_is_correct_l76_76978


namespace evaluate_decimal_expressions_l76_76523

theorem evaluate_decimal_expressions :
  let d1 := (2:ℚ) / 3
  let d2 := (2:ℚ) / 9
  let d3 := (4:ℚ) / 9
  let d4 := (1:ℚ) / 3
  d1 + d2 - d3 * d4 = (20:ℚ) / 27 :=
by
  sorry

end evaluate_decimal_expressions_l76_76523


namespace smallest_n_for_gn_gt_20_l76_76725

def g (n : ℕ) : ℕ := sorry -- definition of the sum of the digits to the right of the decimal of 1 / 3^n

theorem smallest_n_for_gn_gt_20 : ∃ n : ℕ, n > 0 ∧ g n > 20 ∧ ∀ m, 0 < m ∧ m < n -> g m ≤ 20 :=
by
  -- here should be the proof
  sorry

end smallest_n_for_gn_gt_20_l76_76725


namespace least_common_multiple_first_ten_integers_l76_76936

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76936


namespace ajay_distance_l76_76658

/- Definitions -/
def speed : ℝ := 50 -- Ajay's speed in km/hour
def time : ℝ := 30 -- Time taken in hours

/- Theorem statement -/
theorem ajay_distance : (speed * time = 1500) :=
by
  sorry

end ajay_distance_l76_76658


namespace factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l76_76525

-- Problem 1: Prove the factorization of x^4 - 3x^2 + 1
theorem factorize_x4_minus_3x2_plus_1 (x : ℝ) : 
  x^4 - 3 * x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := 
by
  sorry

-- Problem 2: Prove the factorization of a^5 + a^4 - 2a + 1
theorem factorize_a5_plus_a4_minus_2a_plus_1 (a : ℝ) : 
  a^5 + a^4 - 2 * a + 1 = (a^2 + a - 1) * (a^3 + a - 1) := 
by
  sorry

-- Problem 3: Prove the factorization of m^5 - 2m^3 - m - 1
theorem factorize_m5_minus_2m3_minus_m_minus_1 (m : ℝ) : 
  m^5 - 2 * m^3 - m - 1 = (m^3 + m^2 + 1) * (m^2 - m - 1) := 
by
  sorry

end factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l76_76525


namespace afternoon_emails_l76_76291

theorem afternoon_emails (A : ℕ) (five_morning_emails : ℕ) (two_more : five_morning_emails + 2 = A) : A = 7 :=
by
  sorry

end afternoon_emails_l76_76291


namespace thirteen_power_1997_tens_digit_l76_76382

def tens_digit (n : ℕ) := (n / 10) % 10

theorem thirteen_power_1997_tens_digit :
  tens_digit (13 ^ 1997 % 100) = 5 := by
  sorry

end thirteen_power_1997_tens_digit_l76_76382


namespace final_score_l76_76085

-- Definitions based on the conditions
def bullseye_points : ℕ := 50
def miss_points : ℕ := 0
def half_bullseye_points : ℕ := bullseye_points / 2

-- Statement to prove
theorem final_score : bullseye_points + miss_points + half_bullseye_points = 75 :=
by
  sorry

end final_score_l76_76085


namespace correct_blanks_l76_76060

def fill_in_blanks (category : String) (plural_noun : String) : String :=
  "For many, winning remains " ++ category ++ " dream, but they continue trying their luck as there're always " ++ plural_noun ++ " chances that they might succeed."

theorem correct_blanks :
  fill_in_blanks "a" "" = "For many, winning remains a dream, but they continue trying their luck as there're always chances that they might succeed." :=
sorry

end correct_blanks_l76_76060


namespace symmetric_point_correct_l76_76250

-- Define the point P in a three-dimensional Cartesian coordinate system.
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the function to find the symmetric point with respect to the x-axis.
def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point P(1, -2, 3).
def P : Point3D := { x := 1, y := -2, z := 3 }

-- The expected symmetric point
def symmetricP : Point3D := { x := 1, y := 2, z := -3 }

-- The proposition we need to prove
theorem symmetric_point_correct :
  symmetricWithRespectToXAxis P = symmetricP :=
by
  sorry

end symmetric_point_correct_l76_76250


namespace painter_completion_time_l76_76982

def hours_elapsed (start_time end_time : String) : ℕ :=
  match (start_time, end_time) with
  | ("9:00 AM", "12:00 PM") => 3
  | _ => 0

-- The initial conditions, the start time is 9:00 AM, and 3 hours later 1/4th is done
def start_time := "9:00 AM"
def partial_completion_time := "12:00 PM"
def partial_completion_fraction := 1 / 4
def partial_time_hours := hours_elapsed start_time partial_completion_time

-- The painter works consistently, so it would take 4 times the partial time to complete the job
def total_time_hours := 4 * partial_time_hours

-- Calculate the completion time by adding total_time_hours to the start_time
def completion_time : String :=
  match start_time with
  | "9:00 AM" => "9:00 PM"
  | _         => "unknown"

theorem painter_completion_time :
  completion_time = "9:00 PM" :=
by
  -- Definitions and calculations already included in the setup
  sorry

end painter_completion_time_l76_76982


namespace proof_combination_l76_76384

open Classical

theorem proof_combination :
  (∃ x : ℝ, x^3 < 1) ∧ (¬ ∃ x : ℚ, x^2 = 2) ∧ (¬ ∀ x : ℕ, x^3 > x^2) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by
  have h1 : ∃ x : ℝ, x^3 < 1 := sorry
  have h2 : ¬ ∃ x : ℚ, x^2 = 2 := sorry
  have h3 : ¬ ∀ x : ℕ, x^3 > x^2 := sorry
  have h4 : ∀ x : ℝ, x^2 + 1 > 0 := sorry
  exact ⟨h1, h2, h3, h4⟩

end proof_combination_l76_76384


namespace find_larger_number_l76_76038

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l76_76038


namespace lcm_first_ten_positive_integers_l76_76883

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76883


namespace remainder_mod_29_l76_76200

-- Definitions of the given conditions
def N (k : ℕ) := 899 * k + 63

-- The proof statement to be proved
theorem remainder_mod_29 (k : ℕ) : (N k) % 29 = 5 := 
by {
  sorry
}

end remainder_mod_29_l76_76200


namespace find_a5_a6_l76_76572

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

-- Given conditions
axiom h1 : a 1 + a 2 = 5
axiom h2 : a 3 + a 4 = 7

-- Arithmetic sequence property
axiom arith_seq : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

-- The statement we want to prove
theorem find_a5_a6 : a 5 + a 6 = 9 :=
sorry

end ArithmeticSequence

end find_a5_a6_l76_76572


namespace total_signs_at_intersections_l76_76211

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end total_signs_at_intersections_l76_76211


namespace least_common_multiple_first_ten_l76_76831

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76831


namespace find_greatest_consecutive_integer_l76_76544

theorem find_greatest_consecutive_integer (n : ℤ) 
  (h : n^2 + (n + 1)^2 = 452) : n + 1 = 15 :=
sorry

end find_greatest_consecutive_integer_l76_76544


namespace multiple_of_shirt_cost_l76_76489

theorem multiple_of_shirt_cost (S C M : ℕ) (h1 : S = 97) (h2 : C = 300 - S)
  (h3 : C = M * S + 9) : M = 2 :=
by
  -- The proof will be filled in here
  sorry

end multiple_of_shirt_cost_l76_76489


namespace tan_diff_l76_76722

theorem tan_diff (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) : Real.tan (α - β) = 1 / 7 := by
  sorry

end tan_diff_l76_76722


namespace lindsey_exercise_bands_l76_76447

theorem lindsey_exercise_bands (x : ℕ) 
  (h1 : ∀ n, n = 5 * x) 
  (h2 : ∀ m, m = 10 * x) 
  (h3 : ∀ d, d = m + 10) 
  (h4 : d = 30) : 
  x = 2 := 
by 
  sorry

end lindsey_exercise_bands_l76_76447


namespace original_price_of_dish_l76_76433

theorem original_price_of_dish (P : ℝ) (h1 : ∃ P, John's_payment = (0.9 * P) + (0.15 * P))
                               (h2 : ∃ P, Jane's_payment = (0.9 * P) + (0.135 * P))
                               (h3 : John's_payment = Jane's_payment + 0.51) : P = 34 := by
  -- John's Payment
  let John's_payment := (0.9 * P) + (0.15 * P)
  -- Jane's Payment
  let Jane's_payment := (0.9 * P) + (0.135 * P)
  -- Condition that John paid $0.51 more than Jane
  have h3 : John's_payment = Jane's_payment + 0.51 := sorry
  -- From the given conditions, we need to prove P = 34
  sorry

end original_price_of_dish_l76_76433


namespace purely_imaginary_complex_number_l76_76542

theorem purely_imaginary_complex_number (a : ℝ) 
  (h1 : (a^2 - 4 * a + 3 = 0))
  (h2 : a ≠ 1) 
  : a = 3 := 
sorry

end purely_imaginary_complex_number_l76_76542


namespace find_larger_number_l76_76467

-- Define the conditions
variables (L S : ℕ)
axiom condition1 : L - S = 1365
axiom condition2 : L = 6 * S + 35

-- State the theorem
theorem find_larger_number : L = 1631 :=
by
  sorry

end find_larger_number_l76_76467


namespace sum_of_eighth_powers_of_roots_l76_76775

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let root_disc := Real.sqrt discriminant
  ((-b + root_disc) / (2 * a), (-b - root_disc) / (2 * a))

theorem sum_of_eighth_powers_of_roots :
  let (p, q) := quadratic_roots 1 (-Real.sqrt 7) 1
  p^2 + q^2 = 5 ∧ p^4 + q^4 = 23 ∧ p^8 + q^8 = 527 :=
by
  sorry

end sum_of_eighth_powers_of_roots_l76_76775


namespace ratio_of_polynomials_eq_962_l76_76997

open Real

theorem ratio_of_polynomials_eq_962 :
  (10^4 + 400) * (26^4 + 400) * (42^4 + 400) * (58^4 + 400) /
  ((2^4 + 400) * (18^4 + 400) * (34^4 + 400) * (50^4 + 400)) = 962 := 
sorry

end ratio_of_polynomials_eq_962_l76_76997


namespace sum_of_all_four_numbers_is_zero_l76_76428

theorem sum_of_all_four_numbers_is_zero 
  {a b c d : ℝ}
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b = c + d)
  (h_prod : a * c = b * d) 
  : a + b + c + d = 0 := 
by
  sorry

end sum_of_all_four_numbers_is_zero_l76_76428


namespace lcm_first_ten_integers_l76_76806

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76806


namespace inequality_solution_l76_76593

theorem inequality_solution (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x > 1) ∧ 
  (0 < a ∧ a < 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x < 1) :=
by {
  sorry
}

end inequality_solution_l76_76593


namespace months_for_three_times_collection_l76_76296

def Kymbrea_collection (n : ℕ) : ℕ := 40 + 3 * n
def LaShawn_collection (n : ℕ) : ℕ := 20 + 5 * n

theorem months_for_three_times_collection : ∃ n : ℕ, LaShawn_collection n = 3 * Kymbrea_collection n ∧ n = 25 := 
by
  sorry

end months_for_three_times_collection_l76_76296


namespace prime_gt_five_condition_l76_76140

theorem prime_gt_five_condition (p : ℕ) [Fact (Nat.Prime p)] (h : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < p - a^2 ∧ p - a^2 < p - b^2 ∧ (p - a^2) ∣ (p - b)^2 := 
sorry

end prime_gt_five_condition_l76_76140


namespace running_speed_is_24_l76_76979

def walk_speed := 8 -- km/h
def walk_time := 3 -- hours
def run_time := 1 -- hour

def walk_distance := walk_speed * walk_time

def run_speed := walk_distance / run_time

theorem running_speed_is_24 : run_speed = 24 := 
by
  sorry

end running_speed_is_24_l76_76979


namespace transformed_parabola_is_correct_l76_76025

-- Definitions based on conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3
def shifted_left (x : ℝ) : ℝ := original_parabola (x - 2)
def shifted_up (y : ℝ) : ℝ := y + 2

-- Theorem statement
theorem transformed_parabola_is_correct :
  ∀ x : ℝ, shifted_up (shifted_left x) = 3 * x^2 + 6 * x - 1 :=
by 
  -- Proof will be filled in here
  sorry

end transformed_parabola_is_correct_l76_76025


namespace solve_logarithmic_equation_l76_76316

theorem solve_logarithmic_equation (x : ℝ) (h : x > 0 ∧ x ≠ 1 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → log (k/(k+1) : ℝ) ≠ 0)) :
  (1 / (log x / log (1/2 : ℝ)) + 
   1 / (log x / log (2/3 : ℝ)) + 
   1 / (log x / log (3/4 : ℝ)) + 
   1 / (log x / log (4/5 : ℝ)) + 
   1 / (log x / log (5/6 : ℝ)) + 
   1 / (log x / log (6/7 : ℝ)) + 
   1 / (log x / log (7/8 : ℝ)) + 
   1 / (log x / log (8/9 : ℝ)) + 
   1 / (log x / log (9/10 : ℝ)) = 1) → 
  x = 1/10 := sorry

end solve_logarithmic_equation_l76_76316


namespace product_of_digits_l76_76561

-- Define the conditions and state the theorem
theorem product_of_digits (A B : ℕ) (h1 : (10 * A + B) % 12 = 0) (h2 : A + B = 12) : A * B = 32 :=
  sorry

end product_of_digits_l76_76561


namespace least_positive_integer_divisible_by_first_ten_l76_76866

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76866


namespace power_function_decreasing_n_value_l76_76408

theorem power_function_decreasing_n_value (n : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < x → f x = (n^2 + 2 * n - 2) * x^(n^2 - 3 * n)) →
  (∀ x y : ℝ, 0 < x ∧ 0 < y → x < y → f y < f x) →
  n = 1 := 
by
  sorry

end power_function_decreasing_n_value_l76_76408


namespace polynomial_value_at_neg2_l76_76006

noncomputable def P : ℝ → ℝ
-- Define the polynomial P assuming the conditions

theorem polynomial_value_at_neg2 :
  (∀ P : ℝ → ℝ,
  ∃ (a b c d e : ℝ),      -- General form for polynomial degree 4
  P(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧
    P(0) = 1 ∧ P(1) = 1 ∧ P(2) = 4 ∧ P(3) = 9 ∧ P(4) = 16) →
  P(-2) = 19 :=
by 
  sorry

end polynomial_value_at_neg2_l76_76006


namespace infinite_n_exists_r_s_t_l76_76692

noncomputable def a (n : ℕ) : ℝ := n^(1/3 : ℝ)
noncomputable def b (n : ℕ) : ℝ := 1 / (a n - ⌊a n⌋)
noncomputable def c (n : ℕ) : ℝ := 1 / (b n - ⌊b n⌋)

theorem infinite_n_exists_r_s_t :
  ∃ (n : ℕ) (r s t : ℤ), (0 < n ∧ ¬∃ k : ℕ, n = k^3) ∧ (¬(r = 0 ∧ s = 0 ∧ t = 0)) ∧ (r * a n + s * b n + t * c n = 0) :=
sorry

end infinite_n_exists_r_s_t_l76_76692


namespace gcd_seq_consecutive_l76_76049

-- Define the sequence b_n
def seq (n : ℕ) : ℕ := n.factorial + 2 * n

-- Main theorem statement
theorem gcd_seq_consecutive (n : ℕ) : n ≥ 0 → Nat.gcd (seq n) (seq (n + 1)) = 2 :=
by
  intro h
  sorry

end gcd_seq_consecutive_l76_76049


namespace easter_egg_problem_l76_76610

-- Define the conditions as assumptions
def total_eggs : Nat := 63
def helen_eggs (H : Nat) := H
def hannah_eggs (H : Nat) := 2 * H
def harry_eggs (H : Nat) := 2 * H + 3

-- The theorem stating the proof problem
theorem easter_egg_problem (H : Nat) (hh : hannah_eggs H + helen_eggs H + harry_eggs H = total_eggs) : 
    helen_eggs H = 12 ∧ hannah_eggs H = 24 ∧ harry_eggs H = 27 :=
sorry -- Proof is omitted

end easter_egg_problem_l76_76610


namespace ivar_total_water_needed_l76_76289

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end ivar_total_water_needed_l76_76289


namespace problem_1_problem_2_l76_76537

def condition_p (x : ℝ) : Prop := 4 * x ^ 2 + 12 * x - 7 ≤ 0
def condition_q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Problem 1: When a=0, if p is true and q is false, the range of real numbers x
theorem problem_1 (x : ℝ) :
  condition_p x ∧ ¬ condition_q 0 x ↔ -7/2 ≤ x ∧ x < -3 := sorry

-- Problem 2: If p is a sufficient condition for q, the range of real numbers a
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, condition_p x → condition_q a x) ↔ -5/2 ≤ a ∧ a ≤ -1/2 := sorry

end problem_1_problem_2_l76_76537


namespace employee_salary_amount_l76_76157

theorem employee_salary_amount (total_revenue : ℝ) (ratio_salary : ℝ) (ratio_stock : ℝ) (total_parts : ℝ) (salary_ratio_fraction : ℝ) :
  total_revenue = 3000 →
  ratio_salary = 4 →
  ratio_stock = 11 →
  total_parts = ratio_salary + ratio_stock →
  salary_ratio_fraction = ratio_salary / total_parts →
  salary_ratio_fraction * total_revenue = 800 :=
by
  intros h_total_revenue h_ratio_salary h_ratio_stock h_total_parts h_salary_ratio_fraction
  rw [h_total_revenue, h_ratio_salary, h_ratio_stock, h_total_parts, h_salary_ratio_fraction]
  sorry

end employee_salary_amount_l76_76157


namespace least_common_multiple_1_to_10_l76_76902

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76902


namespace alyssa_puppies_l76_76362

-- Definitions from the problem conditions
def initial_puppies (P x : ℕ) : ℕ := P + x

-- Lean 4 Statement of the problem
theorem alyssa_puppies (P x : ℕ) (given_aw: 7 = 7) (remaining: 5 = 5) :
  initial_puppies P x = 12 :=
sorry

end alyssa_puppies_l76_76362


namespace lcm_first_ten_positive_integers_l76_76876

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76876


namespace distance_from_point_to_line_l76_76388

noncomputable def distance_point_to_line
(point : ℝ × ℝ × ℝ)
(line_point direction_vector : ℝ × ℝ × ℝ)
: ℝ :=
let \(\begin{pmatrix} x_0, y_0, z_0 \end{pmatrix}\) := point in
let \(\begin{pmatrix} x, y, z \end{pmatrix}\) := line_point in
let \(\begin{pmatrix} u, v, w \end{pmatrix}\) := direction_vector in
real.sqrt ((x - x_0 - 11/34 * u)^2 + (y - y_0 - 11/34 * v)^2 + (z - z_0 + 11/34 * w)^2)

theorem distance_from_point_to_line : 
  distance_point_to_line (2, 3, 4) (4, 5, 5) (4, 3, -3) = real.sqrt 5.44 :=
by
  sorry

end distance_from_point_to_line_l76_76388


namespace quarters_dimes_equivalence_l76_76595

theorem quarters_dimes_equivalence (m : ℕ) (h : 25 * 30 + 10 * 20 = 25 * 15 + 10 * m) : m = 58 :=
by
  sorry

end quarters_dimes_equivalence_l76_76595


namespace integer_not_natural_l76_76470

theorem integer_not_natural (n : ℕ) (a : ℝ) (b : ℝ) (x y z : ℝ) 
  (h₁ : x = (1 + a) ^ n) 
  (h₂ : y = (1 - a) ^ n) 
  (h₃ : z = a): 
  ∃ k : ℤ, (x - y) / z = ↑k ∧ (k < 0 ∨ k ≠ 0) :=
by 
  sorry

end integer_not_natural_l76_76470


namespace relation_between_m_and_n_l76_76013

variable {A x y z a b c d e n m : ℝ}
variable {p r : ℝ}
variable (s : finset ℝ) (hset : s = {x, y, z, a, b, c, d, e})
variable (hsorted : x < y ∧ y < z ∧ z < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e)
variable (hne : n ∉ s)
variable (hme : m ∉ s)

theorem relation_between_m_and_n 
  (h_avg_n : (s.sum + n) / 9 = (s.sum / 8) * (1 + p / 100)) 
  (h_avg_m : (s.sum + m) / 9 = (s.sum / 8) * (1 + r / 100)) 
  : m = n + 9 * (s.sum / 8) * (r / 100 - p / 100) :=
sorry

end relation_between_m_and_n_l76_76013


namespace concert_song_count_l76_76577

theorem concert_song_count (total_concert_time : ℕ)
  (intermission_time : ℕ)
  (song_duration_regular : ℕ)
  (song_duration_special : ℕ)
  (performance_time : ℕ)
  (total_songs : ℕ) :
  total_concert_time = 80 →
  intermission_time = 10 →
  song_duration_regular = 5 →
  song_duration_special = 10 →
  performance_time = total_concert_time - intermission_time →
  performance_time - song_duration_special = (total_songs - 1) * song_duration_regular →
  total_songs = 13 :=
begin
  intros h_total h_intermission h_regular h_special h_performance_time h_remaining_songs,
  sorry
end

end concert_song_count_l76_76577


namespace max_pawns_l76_76956

def chessboard : Type := ℕ × ℕ -- Define a chessboard as a grid of positions (1,1) to (8,8)
def e4 : chessboard := (5, 4) -- Define the position e4
def symmetric_wrt_e4 (p1 p2 : chessboard) : Prop :=
  p1.1 + p2.1 = 10 ∧ p1.2 + p2.2 = 8 -- Symmetry condition relative to e4

def placed_on (pos : chessboard) : Prop := sorry -- placeholder for placement condition

theorem max_pawns (no_e4 : ¬ placed_on e4)
  (no_symmetric_pairs : ∀ p1 p2, symmetric_wrt_e4 p1 p2 → ¬ (placed_on p1 ∧ placed_on p2)) :
  ∃ max_pawns : ℕ, max_pawns = 39 :=
sorry

end max_pawns_l76_76956


namespace linear_equation_a_neg2_l76_76721

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l76_76721


namespace cos_identity_proof_l76_76793

open Real

theorem cos_identity_proof : 
  2 * cos (16 * (π / 180)) * cos (29 * (π / 180)) - cos (13 * (π / 180)) = sqrt 2 / 2 :=
by
  sorry

end cos_identity_proof_l76_76793


namespace distance_traveled_l76_76560

theorem distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 12) : D = 100 := 
sorry

end distance_traveled_l76_76560


namespace train_speed_l76_76073

noncomputable def original_speed_of_train (v d : ℝ) : Prop :=
  (120 ≤ v / (5/7)) ∧
  (2 * d) / (5 * v) = 65 / 60 ∧
  (2 * (d - 42)) / (5 * v) = 45 / 60

theorem train_speed (v d : ℝ) (h : original_speed_of_train v d) : v = 50.4 :=
by sorry

end train_speed_l76_76073


namespace arithmetic_sequence_terms_l76_76473

theorem arithmetic_sequence_terms
  (a : ℕ → ℝ)
  (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 20)
  (h2 : a (n-2) + a (n-1) + a n = 130)
  (h3 : (n * (a 1 + a n)) / 2 = 200) :
  n = 8 := 
sorry

end arithmetic_sequence_terms_l76_76473


namespace digimon_pack_price_l76_76295

-- Defining the given conditions as Lean variables
variables (total_spent baseball_cost : ℝ)
variables (packs_of_digimon : ℕ)

-- Setting given values from the problem
def keith_total_spent : total_spent = 23.86 := sorry
def baseball_deck_cost : baseball_cost = 6.06 := sorry
def number_of_digimon_packs : packs_of_digimon = 4 := sorry

-- Stating the main theorem/problem to prove
theorem digimon_pack_price 
  (h1 : total_spent = 23.86)
  (h2 : baseball_cost = 6.06)
  (h3 : packs_of_digimon = 4) : 
  ∃ (price_per_pack : ℝ), price_per_pack = 4.45 :=
sorry

end digimon_pack_price_l76_76295


namespace least_divisible_1_to_10_l76_76804

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76804


namespace s_at_1_l76_76584

def t (x : ℚ) := 5 * x - 12
def s (y : ℚ) := (y + 12) / 5 ^ 2 + 5 * ((y + 12) / 5) - 4

theorem s_at_1 : s 1 = 394 / 25 := by
  sorry

end s_at_1_l76_76584


namespace trig_function_value_l76_76555

noncomputable def f : ℝ → ℝ := sorry

theorem trig_function_value:
  (∀ x, f (Real.cos x) = Real.cos (3 * x)) →
  f (Real.sin (Real.pi / 6)) = -1 :=
by
  intro h
  sorry

end trig_function_value_l76_76555


namespace solve_for_b_l76_76723

theorem solve_for_b (b : ℚ) (h : b + b / 4 - 1 = 3 / 2) : b = 2 :=
sorry

end solve_for_b_l76_76723


namespace rectangular_solid_width_l76_76208

theorem rectangular_solid_width 
  (l : ℝ) (w : ℝ) (h : ℝ) (S : ℝ)
  (hl : l = 5)
  (hh : h = 1)
  (hs : S = 58) :
  2 * l * w + 2 * l * h + 2 * w * h = S → w = 4 := 
by
  intros h_surface_area 
  sorry

end rectangular_solid_width_l76_76208


namespace range_of_m_l76_76395

theorem range_of_m (m : ℝ) (h : (m^2 + m) ^ (3 / 5) ≤ (3 - m) ^ (3 / 5)) : 
  -3 ≤ m ∧ m ≤ 1 :=
by { sorry }

end range_of_m_l76_76395


namespace least_common_multiple_of_first_ten_integers_l76_76929

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76929


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76842

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76842


namespace farmer_spending_l76_76644

theorem farmer_spending (X : ℝ) (hc : 0.80 * X + 0.60 * X = 49) : X = 35 := 
by
  sorry

end farmer_spending_l76_76644


namespace axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l76_76104

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt 3 * Real.sin (Real.pi - x) + 5 * Real.sin (Real.pi / 2 + x) + 5

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = f (Real.pi / 3 + k * Real.pi) :=
sorry

theorem center_of_symmetry :
  ∃ k : ℤ, f (k * Real.pi - Real.pi / 6) = 5 :=
sorry

noncomputable def g (x : ℝ) : ℝ := 10 * Real.sin (2 * x) - 8

theorem g_max_value :
  ∀ x : ℝ, g x ≤ 2 :=
sorry

theorem g_increasing_intervals :
  ∀ k : ℤ, -Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≤ g (x + 1) :=
sorry

theorem g_decreasing_intervals :
  ∀ k : ℤ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≥ g (x + 1) :=
sorry

end axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l76_76104


namespace lcm_first_ten_numbers_l76_76907

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76907


namespace emily_cell_phone_cost_l76_76640

noncomputable def base_cost : ℝ := 25
noncomputable def included_hours : ℝ := 25
noncomputable def cost_per_text : ℝ := 0.1
noncomputable def cost_per_extra_minute : ℝ := 0.15
noncomputable def cost_per_gigabyte : ℝ := 2

noncomputable def emily_texts : ℝ := 150
noncomputable def emily_hours : ℝ := 26
noncomputable def emily_data : ℝ := 3

theorem emily_cell_phone_cost : 
  let texts_cost := emily_texts * cost_per_text
  let extra_minutes_cost := (emily_hours - included_hours) * 60 * cost_per_extra_minute
  let data_cost := emily_data * cost_per_gigabyte
  base_cost + texts_cost + extra_minutes_cost + data_cost = 55 := by
  sorry

end emily_cell_phone_cost_l76_76640


namespace probability_of_green_ball_l76_76369

def container_X := (5, 7)  -- (red balls, green balls)
def container_Y := (7, 5)  -- (red balls, green balls)
def container_Z := (7, 5)  -- (red balls, green balls)

def total_balls (container : ℕ × ℕ) : ℕ := container.1 + container.2

def probability_green (container : ℕ × ℕ) : ℚ := 
  (container.2 : ℚ) / total_balls container

noncomputable def probability_green_from_random_selection : ℚ :=
  (1 / 3) * probability_green container_X +
  (1 / 3) * probability_green container_Y +
  (1 / 3) * probability_green container_Z

theorem probability_of_green_ball :
  probability_green_from_random_selection = 17 / 36 :=
sorry

end probability_of_green_ball_l76_76369


namespace Sharik_cannot_eat_all_meatballs_within_one_million_flies_l76_76286

theorem Sharik_cannot_eat_all_meatballs_within_one_million_flies:
  (∀ n: ℕ, ∃ i: ℕ, i > n ∧ ((∀ j < i, ∀ k: ℕ, ∃ m: ℕ, (m ≠ k) → (∃ f, f < 10^6) )) → f > 10^6 ) :=
sorry

end Sharik_cannot_eat_all_meatballs_within_one_million_flies_l76_76286


namespace lcm_first_ten_numbers_l76_76910

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76910


namespace solve_for_x_l76_76634

variables {A B C m n x : ℝ}

-- Existing conditions
def A_rate_condition : A = (B + C) / m := sorry
def B_rate_condition : B = (C + A) / n := sorry
def C_rate_condition : C = (A + B) / x := sorry

-- The theorem to be proven
theorem solve_for_x (A_rate_condition : A = (B + C) / m)
                    (B_rate_condition : B = (C + A) / n)
                    (C_rate_condition : C = (A + B) / x)
                    : x = (2 + m + n) / (m * n - 1) := by
  sorry

end solve_for_x_l76_76634


namespace least_common_multiple_of_first_ten_integers_l76_76935

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76935


namespace shaded_area_calculation_l76_76363

-- Define the grid and the side length conditions
def grid_size : ℕ := 5 * 4
def side_length : ℕ := 1
def total_squares : ℕ := 5 * 4

-- Define the area of one small square
def area_of_square (side: ℕ) : ℕ := side * side

-- Define the shaded region in terms of number of small squares fully or partially occupied
def shaded_squares : ℕ := 11

-- By analyzing the grid based on given conditions, prove that the area of the shaded region is 11
theorem shaded_area_calculation : (shaded_squares * side_length * side_length) = 11 := sorry

end shaded_area_calculation_l76_76363


namespace tomatoes_planted_each_kind_l76_76021

-- Definitions derived from Conditions
def total_rows : ℕ := 10
def spaces_per_row : ℕ := 15
def kinds_of_tomatoes : ℕ := 3
def kinds_of_cucumbers : ℕ := 5
def cucumbers_per_kind : ℕ := 4
def potatoes : ℕ := 30
def available_spaces : ℕ := 85

-- Theorem statement with the question and answer derived from the problem
theorem tomatoes_planted_each_kind : (kinds_of_tomatoes * (total_rows * spaces_per_row - Available_spaces - (kinds_of_cucumbers * cucumbers_per_kind + potatoes)) / kinds_of_tomatoes) = 5 :=
by 
  sorry

end tomatoes_planted_each_kind_l76_76021


namespace eight_digit_number_divisible_by_101_l76_76354

def repeat_twice (x : ℕ) : ℕ := 100 * x + x

theorem eight_digit_number_divisible_by_101 (ef gh ij kl : ℕ) 
  (hef : ef < 100) (hgh : gh < 100) (hij : ij < 100) (hkl : kl < 100) :
  (100010001 * repeat_twice ef + 1000010 * repeat_twice gh + 10010 * repeat_twice ij + 10 * repeat_twice kl) % 101 = 0 := sorry

end eight_digit_number_divisible_by_101_l76_76354


namespace least_common_multiple_first_ten_l76_76822

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76822


namespace smallest_n_l76_76239

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end smallest_n_l76_76239


namespace least_divisible_1_to_10_l76_76796

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76796


namespace least_common_multiple_1_to_10_l76_76901

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76901


namespace angle_EHG_65_l76_76306

/-- Quadrilateral $EFGH$ has $EF = FG = GH$, $\angle EFG = 80^\circ$, and $\angle FGH = 150^\circ$; and hence the degree measure of $\angle EHG$ is $65^\circ$. -/
theorem angle_EHG_65 {EF FG GH : ℝ} (h1 : EF = FG) (h2 : FG = GH) 
  (EFG : ℝ) (FGH : ℝ) (h3 : EFG = 80) (h4 : FGH = 150) : 
  ∃ EHG : ℝ, EHG = 65 :=
by
  sorry

end angle_EHG_65_l76_76306


namespace total_dinners_sold_203_l76_76763

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end total_dinners_sold_203_l76_76763


namespace cylinder_longest_segment_l76_76975

-- Define the radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 10

-- Definition for the longest segment inside the cylinder using Pythagorean theorem
def longest_segment (radius height : ℝ) : ℝ :=
  real.sqrt (radius * 2)^2 + height^2

-- Specify the expected answer for the proof
def expected_answer : ℝ := 10 * real.sqrt 2

-- The theorem stating the longest segment length inside the cylinder
theorem cylinder_longest_segment : longest_segment radius height = expected_answer :=
by {
  -- Lean code to set up and prove the equivalence
  sorry
}

end cylinder_longest_segment_l76_76975


namespace inequality_property_l76_76512

theorem inequality_property (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : (a / b) > (b / a) := 
sorry

end inequality_property_l76_76512


namespace least_common_multiple_of_first_10_integers_l76_76949

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76949


namespace quadratic_real_roots_range_find_k_l76_76706

theorem quadratic_real_roots_range (k : ℝ) (h : ∃ x1 x2 : ℝ, x^2 - 2 * (k - 1) * x + k^2 = 0):
  k ≤ 1/2 :=
  sorry

theorem find_k (k : ℝ) (x1 x2 : ℝ) (h₁ : x^2 - 2 * (k - 1) * x + k^2 = 0)
  (h₂ : x₁ * x₂ + x₁ + x₂ - 1 = 0) (h_range : k ≤ 1/2) :
    k = -3 :=
  sorry

end quadratic_real_roots_range_find_k_l76_76706


namespace smallest_n_power_mod_5_l76_76622

theorem smallest_n_power_mod_5 :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (2^N + 1) % 5 = 0 ∧ ∀ M : ℕ, 100 ≤ M ∧ M ≤ 999 ∧ (2^M + 1) % 5 = 0 → N ≤ M := 
sorry

end smallest_n_power_mod_5_l76_76622


namespace area_of_smaller_circle_l76_76337

noncomputable def radius_of_smaller_circle (r : ℝ) : ℝ := r

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 3 * r

noncomputable def length_PA := 5
noncomputable def length_AB := 5

theorem area_of_smaller_circle (r : ℝ) (h1 : radius_of_smaller_circle r = r)
  (h2 : radius_of_larger_circle r = 3 * r)
  (h3 : length_PA = 5) (h4 : length_AB = 5) :
  π * r^2 = (25 / 3) * π :=
  sorry

end area_of_smaller_circle_l76_76337


namespace merchant_mixture_solution_l76_76980

variable (P C : ℝ)

def P_price : ℝ := 2.40
def C_price : ℝ := 6.00
def total_weight : ℝ := 60
def total_price_per_pound : ℝ := 3.00
def total_price : ℝ := total_price_per_pound * total_weight

theorem merchant_mixture_solution (h1 : P + C = total_weight)
                                  (h2 : P_price * P + C_price * C = total_price) :
  C = 10 := 
sorry

end merchant_mixture_solution_l76_76980


namespace graph_of_equation_l76_76048

theorem graph_of_equation {x y : ℝ} (h : (x - 2 * y)^2 = x^2 - 4 * y^2) :
  (y = 0) ∨ (x = 2 * y) :=
by
  sorry

end graph_of_equation_l76_76048


namespace find_x_given_conditions_l76_76318

variables {x y z : ℝ}

theorem find_x_given_conditions (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (576 : ℝ)^(1/7) := 
sorry

end find_x_given_conditions_l76_76318


namespace gcd_power_minus_one_l76_76300

theorem gcd_power_minus_one (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) : gcd (2^a - 1) (2^b - 1) = 2^(gcd a b) - 1 :=
by
  sorry

end gcd_power_minus_one_l76_76300


namespace polynomial_roots_quartic_sum_l76_76583

noncomputable def roots_quartic_sum (a b c : ℂ) : ℂ :=
  a^4 + b^4 + c^4

theorem polynomial_roots_quartic_sum :
  ∀ (a b c : ℂ), (a^3 - 3 * a + 1 = 0) ∧ (b^3 - 3 * b + 1 = 0) ∧ (c^3 - 3 * c + 1 = 0) →
  (a + b + c = 0) ∧ (a * b + b * c + c * a = -3) ∧ (a * b * c = -1) →
  roots_quartic_sum a b c = 18 :=
by
  intros a b c hroot hsum
  sorry

end polynomial_roots_quartic_sum_l76_76583


namespace least_positive_integer_divisible_by_first_ten_l76_76875

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76875


namespace ratio_of_sides_product_of_areas_and_segments_l76_76549

variable (S S' S'' : ℝ) (a a' : ℝ)

-- Given condition
axiom proportion_condition : S / S'' = a / a'

-- Proofs that need to be verified
theorem ratio_of_sides (S S' : ℝ) (a a' : ℝ) (h : S / S'' = a / a') :
  S / a = S' / a' :=
sorry

theorem product_of_areas_and_segments (S S' : ℝ) (a a' : ℝ) (h: S / S'' = a / a') :
  S * a' = S' * a :=
sorry

end ratio_of_sides_product_of_areas_and_segments_l76_76549


namespace necessary_but_not_sufficient_condition_l76_76691

-- Define the condition p: x^2 - x < 0
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (x : ℝ) : Prop := -1 < x ∧ x < 1

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l76_76691


namespace find_a_10_l76_76540

theorem find_a_10 (a : ℕ → ℚ)
  (h0 : a 1 = 1)
  (h1 : ∀ n : ℕ, a (n + 1) = a n / (a n + 2)) :
  a 10 = 1 / 1023 :=
sorry

end find_a_10_l76_76540


namespace lcm_first_ten_l76_76852

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76852


namespace unique_triplet_nat_gt1_l76_76588

theorem unique_triplet_nat_gt1 :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
  c ∣ (a * b + 1) ∧ b ∣ (a * c + 1) ∧ a ∣ (b * c + 1) :=
  by
  sorry

end unique_triplet_nat_gt1_l76_76588


namespace least_common_multiple_of_first_ten_positive_integers_l76_76859

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76859


namespace maximal_sum_of_xy_l76_76179

theorem maximal_sum_of_xy (x y : ℤ) (h : x^2 + y^2 = 100) : ∃ (s : ℤ), s = 14 ∧ ∀ (u v : ℤ), u^2 + v^2 = 100 → u + v ≤ s :=
by sorry

end maximal_sum_of_xy_l76_76179


namespace polynomial_value_at_neg2_l76_76007

noncomputable def P (x : ℝ) : ℝ :=
  x^2 + (1/24) * (x-1) * (x-2) * (x-3) * (x-4)

theorem polynomial_value_at_neg2 :
  P(0) = 1 →
  P(1) = 1 →
  P(2) = 4 →
  P(3) = 9 →
  P(4) = 16 →
  P(-2) = 19 :=
by
  intros h0 h1 h2 h3 h4
  rw [P] at *
  -- rest of the proof would follow, but it's skipped here
  sorry

end polynomial_value_at_neg2_l76_76007


namespace sequence_elements_are_prime_l76_76498

variable {a : ℕ → ℕ} {p : ℕ → ℕ}

def increasing_seq (f : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → f i < f j

def divisible_by_prime (a p : ℕ → ℕ) : Prop :=
  ∀ n, Prime (p n) ∧ p n ∣ a n

def satisfies_condition (a p : ℕ → ℕ) : Prop :=
  ∀ n k, a n - a k = p n - p k

theorem sequence_elements_are_prime (h1 : increasing_seq a) 
    (h2 : divisible_by_prime a p) 
    (h3 : satisfies_condition a p) :
    ∀ n, Prime (a n) :=
by 
  sorry

end sequence_elements_are_prime_l76_76498


namespace factorize_expression_polygon_sides_l76_76350

-- Problem 1: Factorize 2x^3 - 8x
theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Find the number of sides of a polygon with interior angle sum 1080 degrees
theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end factorize_expression_polygon_sides_l76_76350


namespace right_triangle_inequality_equality_condition_l76_76163

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b ≤ 5 * c :=
by 
  sorry

theorem equality_condition (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b = 5 * c ↔ a / b = 3 / 4 :=
by
  sorry

end right_triangle_inequality_equality_condition_l76_76163


namespace problem_r_minus_s_l76_76145

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l76_76145


namespace f_neg2_minus_f_neg3_l76_76704

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end f_neg2_minus_f_neg3_l76_76704


namespace least_common_multiple_1_to_10_l76_76896

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76896


namespace division_quotient_remainder_l76_76031

theorem division_quotient_remainder :
  ∃ (q r : ℝ), 76.6 = 1.8 * q + r ∧ 0 ≤ r ∧ r < 1.8 ∧ q = 42 ∧ r = 1 := by
  sorry

end division_quotient_remainder_l76_76031


namespace problem1_problem2_l76_76499

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.tan α = -3/4) :
  (Real.cos ((π / 2) + α) * Real.sin (-π - α)) /
  (Real.cos ((11 * π) / 2 - α) * Real.sin ((9 * π) / 2 + α)) = -3 / 4 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ π) (h2 : Real.sin α + Real.cos α = 1 / 5) :
  Real.cos (2 * α - π / 4) = -31 * Real.sqrt 2 / 50 :=
by sorry

end problem1_problem2_l76_76499


namespace lateral_surface_area_of_prism_l76_76682

theorem lateral_surface_area_of_prism (h : ℝ) (angle : ℝ) (h_pos : 0 < h) (angle_eq : angle = 60) :
  ∃ S : ℝ, S = 6 * h^2 :=
by
  sorry

end lateral_surface_area_of_prism_l76_76682


namespace sum_of_roots_l76_76381

   theorem sum_of_roots : 
     let a := 2
     let b := 7
     let c := 3
     let roots := (-b / a : ℝ)
     roots = -3.5 :=
   by
     sorry
   
end sum_of_roots_l76_76381


namespace train_speed_l76_76357

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) (h_length : length = 975) (h_time : time = 48) (h_speed : speed = length / time * 3.6) : 
  speed = 73.125 := 
by 
  sorry

end train_speed_l76_76357


namespace total_dinners_l76_76760

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end total_dinners_l76_76760


namespace amber_max_ounces_l76_76660

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l76_76660


namespace binom_sum_l76_76518

theorem binom_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := 
by
  sorry

end binom_sum_l76_76518


namespace least_common_multiple_of_first_10_integers_l76_76950

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76950


namespace largest_possible_n_l76_76596

theorem largest_possible_n :
  ∃ (m n : ℕ), (0 < m) ∧ (0 < n) ∧ (m + n = 10) ∧ (n = 9) :=
by
  sorry

end largest_possible_n_l76_76596


namespace simplified_expression_num_terms_l76_76315

noncomputable def num_terms_polynomial (n: ℕ) : ℕ :=
  (n/2) * (1 + (n+1))

theorem simplified_expression_num_terms :
  num_terms_polynomial 2012 = 1012608 :=
by
  sorry

end simplified_expression_num_terms_l76_76315


namespace least_common_multiple_first_ten_l76_76834

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76834


namespace least_common_multiple_first_ten_integers_l76_76945

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76945


namespace lcm_first_ten_l76_76847

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76847


namespace sum_between_100_and_500_ending_in_3_l76_76252

-- Definition for the sum of all integers between 100 and 500 that end in 3
def sumOfIntegersBetween100And500EndingIn3 : ℕ :=
  let a := 103
  let d := 10
  let n := (493 - a) / d + 1
  (n * (a + 493)) / 2

-- Statement to prove that the sum is 11920
theorem sum_between_100_and_500_ending_in_3 : sumOfIntegersBetween100And500EndingIn3 = 11920 := by
  sorry

end sum_between_100_and_500_ending_in_3_l76_76252


namespace initial_weight_l76_76220

theorem initial_weight (lost_weight current_weight : ℕ) (h1 : lost_weight = 35) (h2 : current_weight = 34) :
  lost_weight + current_weight = 69 :=
sorry

end initial_weight_l76_76220


namespace sum_of_x_coordinates_of_intersections_l76_76597

def g : ℝ → ℝ := sorry  -- Definition of g is unspecified but it consists of five line segments.

theorem sum_of_x_coordinates_of_intersections 
  (h1 : ∃ x1, g x1 = x1 - 2 ∧ (x1 = -2 ∨ x1 = 1 ∨ x1 = 4))
  (h2 : ∃ x2, g x2 = x2 - 2 ∧ (x2 = -2 ∨ x2 = 1 ∨ x2 = 4))
  (h3 : ∃ x3, g x3 = x3 - 2 ∧ (x3 = -2 ∨ x3 = 1 ∨ x3 = 4)) 
  (hx1x2x3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = 3 := by
  -- Proof here
  sorry

end sum_of_x_coordinates_of_intersections_l76_76597


namespace max_area_rectangle_min_area_rectangle_l76_76397

theorem max_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k) 
  : (n - 1 + 2^(2*n)) * (4 * 2^(2*(n-1)) - 1/3) = 1/3 * (4^n - 1) * (4^n + n - 1) := sorry

theorem min_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k)
  : (2^n - 1)^2 = 4 * (2^n - 1)^2 := sorry

end max_area_rectangle_min_area_rectangle_l76_76397


namespace rational_number_div_l76_76276

theorem rational_number_div (x : ℚ) (h : -2 / x = 8) : x = -1 / 4 := 
by
  sorry

end rational_number_div_l76_76276


namespace surface_area_cube_l76_76729

theorem surface_area_cube (a : ℕ) (b : ℕ) (h : a = 2) : b = 54 :=
  by
  sorry

end surface_area_cube_l76_76729


namespace least_common_multiple_1_to_10_l76_76892

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76892


namespace min_overlap_l76_76160

variable (P : Set ℕ → ℝ)
variable (B M : Set ℕ)

-- Conditions
def P_B_def : P B = 0.95 := sorry
def P_M_def : P M = 0.85 := sorry

-- To Prove
theorem min_overlap : P (B ∩ M) = 0.80 := sorry

end min_overlap_l76_76160


namespace least_common_multiple_of_first_ten_l76_76916

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76916


namespace domain_f_log2_x_to_domain_f_x_l76_76543

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end domain_f_log2_x_to_domain_f_x_l76_76543


namespace josh_initial_money_l76_76743

/--
Josh spent $1.75 on a drink, and then spent another $1.25, and has $6.00 left. 
Prove that initially Josh had $9.00.
-/
theorem josh_initial_money : 
  ∃ (initial : ℝ), (initial - 1.75 - 1.25 = 6) ∧ initial = 9 := 
sorry

end josh_initial_money_l76_76743


namespace work_completion_days_l76_76352

theorem work_completion_days (A B : ℕ) (hB : B = 12) (work_together_days : ℕ) (work_together : work_together_days = 3) (work_alone_days : ℕ) (work_alone : work_alone_days = 3) : 
  (1 / A + 1 / B) * 3 + (1 / B) * 3 = 1 → A = 6 := 
by 
  intro h
  sorry

end work_completion_days_l76_76352


namespace sin_2012_equals_neg_sin_32_l76_76636

theorem sin_2012_equals_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = -Real.sin (32 * Real.pi / 180) := by
  sorry

end sin_2012_equals_neg_sin_32_l76_76636


namespace isosceles_triangle_perimeter_l76_76324

-- Define the conditions
def equilateral_triangle_side : ℕ := 15
def isosceles_triangle_side : ℕ := 15
def isosceles_triangle_base : ℕ := 10

-- Define the theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter : 
  (2 * isosceles_triangle_side + isosceles_triangle_base = 40) :=
by
  -- Placeholder for the actual proof
  sorry

end isosceles_triangle_perimeter_l76_76324


namespace least_common_multiple_of_first_ten_positive_integers_l76_76856

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76856


namespace point_not_on_graph_l76_76514

theorem point_not_on_graph (x y : ℝ) : 
  (x = 1 ∧ y = 5 → y ≠ 6 / x) :=
by
  intros h
  cases h with hx hy
  rw [hx, hy]
  norm_num
  exact ne_of_gt (by norm_num)
  sorry

end point_not_on_graph_l76_76514


namespace Carmela_difference_l76_76995

theorem Carmela_difference (Cecil Catherine Carmela : ℤ) (X : ℤ) (h1 : Cecil = 600) 
(h2 : Catherine = 2 * Cecil - 250) (h3 : Carmela = 2 * Cecil + X) 
(h4 : Cecil + Catherine + Carmela = 2800) : X = 50 :=
by { sorry }

end Carmela_difference_l76_76995


namespace problem_statement_l76_76750

open Real Polynomial

theorem problem_statement (a1 a2 a3 d1 d2 d3 : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
                 (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end problem_statement_l76_76750


namespace solution_set_f_l76_76267

noncomputable def f : ℝ → ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x = f x
axiom f_at_4 : f 4 = -3
axiom f_deriv_lt_3 : ∀ x : ℝ, deriv f x < 3

theorem solution_set_f (x : ℝ) : f x < 3 * x - 15 ↔ x > 4 :=
by 
  sorry

end solution_set_f_l76_76267


namespace base_amount_calculation_l76_76642

theorem base_amount_calculation (tax_amount : ℝ) (tax_rate : ℝ) (base_amount : ℝ) 
  (h1 : tax_amount = 82) (h2 : tax_rate = 82) : base_amount = 100 :=
by
  -- Proof will be provided here.
  sorry

end base_amount_calculation_l76_76642


namespace m_condition_sufficient_not_necessary_l76_76521

-- Define the function f(x) and its properties
def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

-- Define the condition for the function being increasing on (0, ∞)
def is_increasing_on_positives (m : ℝ) :=
  ∀ x y : ℝ, 0 < x → x < y → f m x < f m y

-- Prove that if m > 0, then the function is increasing on (0, ∞)
lemma m_gt_0_sufficient (m : ℝ) (h : 0 < m) : is_increasing_on_positives m :=
sorry

-- Show that the condition is indeed sufficient but not necessary
theorem m_condition_sufficient_not_necessary :
  ∀ m : ℝ, (0 < m → is_increasing_on_positives m) ∧ (is_increasing_on_positives m → 0 < m) :=
sorry

end m_condition_sufficient_not_necessary_l76_76521


namespace sum_of_arithmetic_sequence_l76_76415

variable {S : ℕ → ℕ}

def isArithmeticSum (S : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ n, S n = n * (2 * a + (n - 1) * d ) / 2

theorem sum_of_arithmetic_sequence :
  isArithmeticSum S →
  S 8 - S 4 = 12 →
  S 12 = 36 :=
by
  intros
  sorry

end sum_of_arithmetic_sequence_l76_76415


namespace least_common_multiple_first_ten_l76_76823

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76823


namespace variance_of_binomial_distribution_l76_76421

noncomputable def variance_binomial :=
  let n := 8
  let p := 0.7
  n * p * (1 - p)

theorem variance_of_binomial_distribution : variance_binomial = 1.68 :=
  by
  sorry

end variance_of_binomial_distribution_l76_76421


namespace movie_ticket_percentage_decrease_l76_76744

theorem movie_ticket_percentage_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100) 
  (h2 : new_price = 80) :
  ((old_price - new_price) / old_price) * 100 = 20 := 
by
  sorry

end movie_ticket_percentage_decrease_l76_76744


namespace volume_difference_l76_76393

theorem volume_difference (x1 x2 x3 Vmin Vmax : ℝ)
  (hx1 : 0.5 < x1 ∧ x1 < 1.5)
  (hx2 : 0.5 < x2 ∧ x2 < 1.5)
  (hx3 : 2016.5 < x3 ∧ x3 < 2017.5)
  (rV : 2017 = Nat.floor (x1 * x2 * x3))
  : abs (Vmax - Vmin) = 4035 := 
sorry

end volume_difference_l76_76393


namespace farmer_pomelos_dozen_l76_76000

theorem farmer_pomelos_dozen (pomelos_last_week : ℕ) (boxes_last_week : ℕ) (boxes_this_week : ℕ) :
  pomelos_last_week = 240 → boxes_last_week = 10 → boxes_this_week = 20 →
  (pomelos_last_week / boxes_last_week) * boxes_this_week / 12 = 40 := 
by
  intro h1 h2 h3
  sorry

end farmer_pomelos_dozen_l76_76000


namespace total_slices_l76_76635

def pizzas : ℕ := 2
def slices_per_pizza : ℕ := 8

theorem total_slices : pizzas * slices_per_pizza = 16 :=
by
  sorry

end total_slices_l76_76635


namespace inverse_of_composed_function_l76_76440

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l76_76440


namespace minimum_a_div_x_l76_76010

theorem minimum_a_div_x (a x y : ℕ) (h1 : 100 < a) (h2 : 100 < x) (h3 : 100 < y) (h4 : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
by sorry

end minimum_a_div_x_l76_76010


namespace Oshea_needs_50_small_planters_l76_76303

structure Planter :=
  (large : ℕ)     -- Number of large planters
  (medium : ℕ)    -- Number of medium planters
  (small : ℕ)     -- Number of small planters
  (capacity_large : ℕ := 20) -- Capacity of large planter
  (capacity_medium : ℕ := 10) -- Capacity of medium planter
  (capacity_small : ℕ := 4)  -- Capacity of small planter

structure Seeds :=
  (basil : ℕ)     -- Number of basil seeds
  (cilantro : ℕ)  -- Number of cilantro seeds
  (parsley : ℕ)   -- Number of parsley seeds

noncomputable def small_planters_needed (planters : Planter) (seeds : Seeds) : ℕ :=
  let basil_in_large := min seeds.basil (planters.large * planters.capacity_large)
  let basil_left := seeds.basil - basil_in_large
  let basil_in_medium := min basil_left (planters.medium * planters.capacity_medium)
  let basil_remaining := basil_left - basil_in_medium
  
  let cilantro_in_medium := min seeds.cilantro ((planters.medium * planters.capacity_medium) - basil_in_medium)
  let cilantro_remaining := seeds.cilantro - cilantro_in_medium
  
  let parsley_total := seeds.parsley + basil_remaining + cilantro_remaining
  parsley_total / planters.capacity_small

theorem Oshea_needs_50_small_planters :
  small_planters_needed 
    { large := 4, medium := 8, small := 0 }
    { basil := 200, cilantro := 160, parsley := 120 } = 50 := 
sorry

end Oshea_needs_50_small_planters_l76_76303


namespace determine_a_l76_76061

theorem determine_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x, y) = (-1, 2) → 3 * x + y + a = 0) → ∃ (a : ℝ), a = 1 :=
by
  sorry

end determine_a_l76_76061


namespace second_expression_l76_76780

theorem second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 79) (h₂ : a = 30) : x = 82 := by
  sorry

end second_expression_l76_76780


namespace determine_a_l76_76717

-- Define the conditions of the problem.
def linear_in_x_and_y (a : ℝ) : Prop :=
  (a - 2) * (x : ℝ)^(abs a - 1) + 3 * (y : ℝ) = 1

-- Prove that a = -2 under the condition defined.
theorem determine_a (a : ℝ) (h : ∀ x y : ℝ, linear_in_x_and_y a) : a = -2 :=
sorry

end determine_a_l76_76717


namespace simplest_square_root_l76_76075

theorem simplest_square_root : 
  let a1 := Real.sqrt 20
  let a2 := Real.sqrt 2
  let a3 := Real.sqrt (1 / 2)
  let a4 := Real.sqrt 0.2
  a2 = Real.sqrt 2 ∧
  (a1 ≠ Real.sqrt 2 ∧ a3 ≠ Real.sqrt 2 ∧ a4 ≠ Real.sqrt 2) :=
by {
  -- Here, we fill in the necessary proof steps, but it's omitted for now.
  sorry
}

end simplest_square_root_l76_76075


namespace initial_order_cogs_l76_76078

theorem initial_order_cogs (x : ℕ) (h : (x + 60 : ℚ) / (x / 36 + 1) = 45) : x = 60 := 
sorry

end initial_order_cogs_l76_76078


namespace smallest_n_for_divisibility_property_l76_76623

theorem smallest_n_for_divisibility_property :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n^2 + n % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n^2 + n % k ≠ 0) ∧ 
  ∀ m : ℕ, 0 < m ∧ m < n → ¬ ((∀ k : ℕ, 1 ≤ k ∧ k ≤ m → m^2 + m % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ m ∧ m^2 + m % k ≠ 0)) := sorry

end smallest_n_for_divisibility_property_l76_76623


namespace longest_segment_in_cylinder_l76_76972

-- Define the given conditions
def radius : ℝ := 5 -- Radius of the cylinder in cm
def height : ℝ := 10 -- Height of the cylinder in cm

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the longest segment L inside the cylinder using the Pythagorean theorem
noncomputable def longest_segment : ℝ := Real.sqrt ((diameter ^ 2) + (height ^ 2))

-- State the problem in Lean:
theorem longest_segment_in_cylinder :
  longest_segment = 10 * Real.sqrt 2 :=
sorry

end longest_segment_in_cylinder_l76_76972


namespace evaluate_expression_l76_76672

def binom (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem evaluate_expression : 
  (binom 2 5 * 3 ^ 5) / binom 10 5 = 0 := by
  -- Given conditions:
  have h1 : binom 2 5 = 0 := by sorry
  have h2 : binom 10 5 = 252 := by sorry
  -- Proof goal:
  sorry

end evaluate_expression_l76_76672


namespace find_y_for_orthogonality_l76_76679

theorem find_y_for_orthogonality (y : ℝ) : (3 * y + 7 * (-4) = 0) → y = 28 / 3 := by
  sorry

end find_y_for_orthogonality_l76_76679


namespace s_is_arithmetic_progression_l76_76024

variables (s : ℕ → ℕ) (ds1 ds2 : ℕ)

-- Conditions
axiom strictly_increasing : ∀ n, s n < s (n + 1)
axiom s_is_positive : ∀ n, 0 < s n
axiom s_s_is_arithmetic : ∃ d1, ∀ k, s (s k) = s (s 0) + k * d1
axiom s_s_plus1_is_arithmetic : ∃ d2, ∀ k, s (s k + 1) = s (s 0 + 1) + k * d2

-- Statement to prove
theorem s_is_arithmetic_progression : ∃ d, ∀ k, s (k + 1) = s 0 + k * d :=
sorry

end s_is_arithmetic_progression_l76_76024


namespace average_jump_difference_l76_76213

-- Define the total jumps and time
def total_jumps_liu_li : ℕ := 480
def total_jumps_zhang_hua : ℕ := 420
def time_minutes : ℕ := 5

-- Define the average jumps per minute
def average_jumps_per_minute (total_jumps : ℕ) (time : ℕ) : ℕ :=
  total_jumps / time

-- State the theorem
theorem average_jump_difference :
  average_jumps_per_minute total_jumps_liu_li time_minutes - 
  average_jumps_per_minute total_jumps_zhang_hua time_minutes = 12 := 
sorry


end average_jump_difference_l76_76213


namespace least_divisible_1_to_10_l76_76798

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76798


namespace distinct_solution_difference_l76_76153

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l76_76153


namespace klay_to_draymond_ratio_l76_76430

-- Let us define the points earned by each player
def draymond_points : ℕ := 12
def curry_points : ℕ := 2 * draymond_points
def kelly_points : ℕ := 9
def durant_points : ℕ := 2 * kelly_points

-- Total points of the Golden State Team
def total_points_team : ℕ := 69

theorem klay_to_draymond_ratio :
  ∃ klay_points : ℕ,
    klay_points = total_points_team - (draymond_points + curry_points + kelly_points + durant_points) ∧
    klay_points / draymond_points = 1 / 2 :=
by
  sorry

end klay_to_draymond_ratio_l76_76430


namespace greatest_number_remainder_l76_76681

theorem greatest_number_remainder (G R : ℕ) (h1 : 150 % G = 50) (h2 : 230 % G = 5) (h3 : 175 % G = R) (h4 : ∀ g, g ∣ 100 → g ∣ 225 → g ∣ (175 - R) → g ≤ G) : R = 0 :=
by {
  -- This is the statement only; the proof is omitted as per the instructions.
  sorry
}

end greatest_number_remainder_l76_76681


namespace marked_vertices_coincide_l76_76285

theorem marked_vertices_coincide :
  ∀ (P Q : Fin 16 → Prop),
  (∃ A B C D E F G : Fin 16, P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G) →
  (∃ A' B' C' D' E' F' G' : Fin 16, Q A' ∧ Q B' ∧ Q C' ∧ Q D' ∧ Q E' ∧ Q F' ∧ Q G') →
  ∃ (r : Fin 16), ∃ (A B C D : Fin 16), 
  (Q ((A + r) % 16) ∧ Q ((B + r) % 16) ∧ Q ((C + r) % 16) ∧ Q ((D + r) % 16)) :=
by
  sorry

end marked_vertices_coincide_l76_76285


namespace world_grain_supply_is_correct_l76_76536

def world_grain_demand : ℝ := 2400000
def supply_ratio : ℝ := 0.75
def world_grain_supply (demand : ℝ) (ratio : ℝ) : ℝ := ratio * demand

theorem world_grain_supply_is_correct :
  world_grain_supply world_grain_demand supply_ratio = 1800000 := 
by 
  sorry

end world_grain_supply_is_correct_l76_76536


namespace triangle_angles_l76_76033

noncomputable def angle_of_triangle (a b c : ℝ) : Prop :=
  let cosθ := (a^2 + b^2 - c^2) / (2 * a * b) in
  let θ := Real.arccos cosθ in
  θ = Real.arccos (7/18 + 2 * Real.sqrt 6 / 9)

-- The equivalent Lean statement for the math proof
theorem triangle_angles :
  let a := 3
  let b := 3
  let c := Real.sqrt 8 - Real.sqrt 3 in
  angle_of_triangle a b c ∧
  angle_of_triangle b c a ∧ 
  angle_of_triangle c a b :=
by
  sorry

end triangle_angles_l76_76033


namespace least_common_multiple_first_ten_integers_l76_76937

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76937


namespace a_minus_b_is_15_l76_76969

variables (a b c : ℝ)

-- Conditions from the problem statement
axiom cond1 : a = 1/3 * (b + c)
axiom cond2 : b = 2/7 * (a + c)
axiom cond3 : a + b + c = 540

-- The theorem we need to prove
theorem a_minus_b_is_15 : a - b = 15 :=
by
  sorry

end a_minus_b_is_15_l76_76969


namespace base5_product_is_correct_l76_76483

-- Definitions for the problem context
def base5_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 528 then [4, 1, 0, 0, 3] else []

-- Theorem to prove the base-5 multiplication result
theorem base5_product_is_correct :
  base10_to_base5 (base5_to_base10 2 3 1 * base5_to_base10 1 3 0) = [4, 1, 0, 0, 3] :=
by
  sorry

end base5_product_is_correct_l76_76483


namespace total_points_other_7_members_is_15_l76_76124

variable (x y : ℕ)
variable (h1 : y ≤ 21)
variable (h2 : y = x * 7 / 15 - 18)
variable (h3 : (1 / 3) * x + (1 / 5) * x + 18 + y = x)

theorem total_points_other_7_members_is_15 (h : x * 7 % 15 = 0) : y = 15 :=
by
  sorry

end total_points_other_7_members_is_15_l76_76124


namespace find_b_value_l76_76726

theorem find_b_value (b : ℕ) 
  (h1 : 5 ^ 5 * b = 3 * 15 ^ 5) 
  (h2 : b = 9 ^ 3) : b = 729 :=
by
  sorry

end find_b_value_l76_76726


namespace find_speed_of_stream_l76_76493

-- Definitions of the conditions:
def downstream_equation (b s : ℝ) : Prop := b + s = 60
def upstream_equation (b s : ℝ) : Prop := b - s = 30

-- Theorem stating the speed of the stream given the conditions:
theorem find_speed_of_stream (b s : ℝ) (h1 : downstream_equation b s) (h2 : upstream_equation b s) : s = 15 := 
sorry

end find_speed_of_stream_l76_76493


namespace determine_a_l76_76787

noncomputable def f (x a : ℝ) := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem determine_a (a : ℝ) 
  (h₁ : ∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f x a ≤ f 0 a)
  (h₂ : f 0 a = -3) :
  a = 2 + Real.sqrt 6 := 
sorry

end determine_a_l76_76787


namespace simplify_and_evaluate_expr_l76_76772

theorem simplify_and_evaluate_expr (x y : ℚ) (h1 : x = -3/8) (h2 : y = 4) :
  (x - 2 * y) ^ 2 + (x - 2 * y) * (x + 2 * y) - 2 * x * (x - y) = 3 :=
by
  sorry

end simplify_and_evaluate_expr_l76_76772


namespace simplification_of_expression_l76_76417

variable {a b : ℚ}

theorem simplification_of_expression (h1a : a ≠ 0) (h1b : b ≠ 0) (h2 : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ( (3 * a)⁻¹ - (b / 3)⁻¹ ) = -(a * b)⁻¹ := 
sorry

end simplification_of_expression_l76_76417


namespace min_triples_count_l76_76445

theorem min_triples_count (a : Fin 9 → ℝ) (m : ℝ)
  (h_avg : (∑ i, a i) / 9 = m) :
  ∃ A : ℕ, A = 28 ∧
  (∀ i j k, 1 ≤ i → i < j → j < k → k ≤ 9 → a i + a j + a k ≥ 3 * m) :=
  sorry

end min_triples_count_l76_76445


namespace units_digit_1_to_99_is_5_l76_76957

noncomputable def units_digit_of_product_of_odds : ℕ :=
  let seq := List.range' 1 99;
  (seq.filter (λ n => n % 2 = 1)).prod % 10

theorem units_digit_1_to_99_is_5 : units_digit_of_product_of_odds = 5 :=
by sorry

end units_digit_1_to_99_is_5_l76_76957


namespace sam_bought_9_cans_l76_76311

-- Definitions based on conditions
def spent_amount_dollars := 20 - 5.50
def spent_amount_cents := 1450 -- to avoid floating point precision issues we equate to given value in cents
def coupon_discount_cents := 5 * 25
def total_cost_no_discount := spent_amount_cents + coupon_discount_cents
def cost_per_can := 175

-- Main statement to prove
theorem sam_bought_9_cans : total_cost_no_discount / cost_per_can = 9 :=
by
  sorry -- Proof goes here

end sam_bought_9_cans_l76_76311


namespace rational_sum_p_q_l76_76265

noncomputable def x := (Real.sqrt 5 - 1) / 2

theorem rational_sum_p_q :
  ∃ (p q : ℚ), x^3 + p * x + q = 0 ∧ p + q = -1 := by
  sorry

end rational_sum_p_q_l76_76265


namespace triangle_area_is_9_l76_76669

-- Define the vertices of the triangle
def x1 : ℝ := 1
def y1 : ℝ := 2
def x2 : ℝ := 4
def y2 : ℝ := 5
def x3 : ℝ := 6
def y3 : ℝ := 1

-- Define the area calculation formula for the triangle
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The proof statement
theorem triangle_area_is_9 :
  triangle_area x1 y1 x2 y2 x3 y3 = 9 :=
by
  sorry

end triangle_area_is_9_l76_76669


namespace least_common_multiple_of_first_ten_integers_l76_76934

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76934


namespace painting_rate_l76_76781

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end painting_rate_l76_76781


namespace max_Cauchy_distribution_convergence_l76_76753

open ProbabilityTheory

-- define the conditions described in step a)
def cauchy_pdf (x : ℝ) : ℝ := 1 / (π * (1 + x^2))

-- iid random variables following Cauchy distribution
axiom X : ℕ → ℝ
axiom X_iid : ∀ i j, i ≠ j → Indep X i X j
axiom X_Cauchy : ∀ n, CDF_X (λ X, cauchy_pdf Xᵢ) = 1

-- M_n is the maximum of the first n variables
def M_n (n : ℕ) : ℝ := finset.max' (finset.range n) X

-- T is a random variable following exponential distribution with parameter 1/π
axiom T_exp : ∀ t : ℝ, distribution.exp_dist 1/π t → ∀ x, (∑ n, X^(-t)) = E[t]

-- the main statement to be proved
theorem max_Cauchy_distribution_convergence :
  tendsto (λ (n : ℕ), (M_n n) / (n : ℝ)) at_top (distribution.exp_dist 1/π) := sorry

end max_Cauchy_distribution_convergence_l76_76753


namespace compound_bar_chart_must_clearly_indicate_legend_l76_76971

-- Definitions of the conditions
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_bars_of_different_colors : Bool

-- The theorem stating that a compound bar chart must clearly indicate the legend
theorem compound_bar_chart_must_clearly_indicate_legend 
  (chart : CompoundBarChart)
  (distinguishes_quantities : chart.distinguishes_two_quantities = true)
  (uses_colors : chart.uses_bars_of_different_colors = true) :
  ∃ legend : String, legend ≠ "" := by
  sorry

end compound_bar_chart_must_clearly_indicate_legend_l76_76971


namespace lcm_first_ten_l76_76855

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76855


namespace ratio_roses_to_lilacs_l76_76689

theorem ratio_roses_to_lilacs
  (L: ℕ) -- number of lilacs sold
  (G: ℕ) -- number of gardenias sold
  (R: ℕ) -- number of roses sold
  (hL: L = 10) -- defining lilacs sold as 10
  (hG: G = L / 2) -- defining gardenias sold as half the lilacs
  (hTotal: R + L + G = 45) -- defining total flowers sold as 45
  : R / L = 3 :=
by {
  -- The actual proof would go here, but we skip it as per instructions
  sorry
}

end ratio_roses_to_lilacs_l76_76689


namespace yen_checking_account_l76_76655

theorem yen_checking_account (savings : ℕ) (total : ℕ) (checking : ℕ) (h1 : savings = 3485) (h2 : total = 9844) (h3 : checking = total - savings) :
  checking = 6359 :=
by
  rw [h1, h2] at h3
  exact h3

end yen_checking_account_l76_76655


namespace sum_of_first_3030_terms_l76_76792

-- Define geometric sequence sum for n terms
noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
axiom geom_sum_1010 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 1010 = 100
axiom geom_sum_2020 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 2020 = 190

-- Prove that the sum of the first 3030 terms is 271
theorem sum_of_first_3030_terms (a r : ℝ) (hr : r ≠ 1) :
  geom_sum a r 3030 = 271 :=
by
  sorry

end sum_of_first_3030_terms_l76_76792


namespace find_constants_monotonicity_l76_76547

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_constants (a b c : ℝ) 
  (h1 : f' (-2/3) a b = 0)
  (h2 : f' 1 a b = 0) :
  a = -1/2 ∧ b = -2 :=
by sorry

theorem monotonicity (a b c : ℝ)
  (h1 : a = -1/2) 
  (h2 : b = -2) : 
  (∀ x : ℝ, x < -2/3 → f' x a b > 0) ∧ 
  (∀ x : ℝ, -2/3 < x ∧ x < 1 → f' x a b < 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a b > 0) :=
by sorry

end find_constants_monotonicity_l76_76547


namespace least_common_multiple_of_first_10_integers_l76_76953

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76953


namespace common_ratio_of_gp_l76_76249

variable (r : ℝ)(n : ℕ)

theorem common_ratio_of_gp (h1 : 9 * r ^ (n - 1) = 1/3) 
                           (h2 : 9 * (1 - r ^ n) / (1 - r) = 40 / 3) : 
                           r = 1/3 := 
sorry

end common_ratio_of_gp_l76_76249


namespace total_dinners_sold_203_l76_76762

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end total_dinners_sold_203_l76_76762


namespace least_common_multiple_first_ten_l76_76827

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76827


namespace slope_tangent_line_at_one_l76_76176

open Real

theorem slope_tangent_line_at_one (f : ℝ → ℝ) (x : ℝ) (h : f = fun x => x * exp x) (hx : x = 1) :
  deriv f 1 = 2 * exp 1 :=
by 
  sorry

end slope_tangent_line_at_one_l76_76176


namespace pizza_dough_milk_needed_l76_76181

variable (milk_per_300 : ℕ) (flour_per_batch : ℕ) (total_flour : ℕ)

-- Definitions based on problem conditions
def milk_per_batch := milk_per_300
def batch_size := flour_per_batch
def used_flour := total_flour

-- The target proof statement
theorem pizza_dough_milk_needed (h1 : milk_per_batch = 60) (h2 : batch_size = 300) (h3 : used_flour = 1500) : 
  (used_flour / batch_size) * milk_per_batch = 300 :=
by
  rw [h1, h2, h3]
  sorry -- proof steps

end pizza_dough_milk_needed_l76_76181


namespace ratio_of_areas_two_adjacent_triangles_to_one_triangle_l76_76503

-- Definition of a regular hexagon divided into six equal triangles
def is_regular_hexagon_divided_into_six_equal_triangles (s : ℝ) : Prop :=
  s > 0 -- s is the area of one of the six triangles and must be positive

-- Definition of the area of a region formed by two adjacent triangles
def area_of_two_adjacent_triangles (s r : ℝ) : Prop :=
  r = 2 * s

-- The proof problem statement
theorem ratio_of_areas_two_adjacent_triangles_to_one_triangle (s r : ℝ)
  (hs : is_regular_hexagon_divided_into_six_equal_triangles s)
  (hr : area_of_two_adjacent_triangles s r) : 
  r / s = 2 :=
by
  sorry

end ratio_of_areas_two_adjacent_triangles_to_one_triangle_l76_76503


namespace cubic_root_of_determinant_l76_76747

open Complex 
open Matrix

noncomputable def matrix_d (a b c n : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![
    ![b + n^3 * c, n * (c - b), n^2 * (b - c)],
    ![n^2 * (c - a), c + n^3 * a, n * (a - c)],
    ![n * (b - a), n^2 * (a - b), a + n^3 * b]
  ]

theorem cubic_root_of_determinant (a b c n : ℂ) (h : a * b * c = 1) :
  (det (matrix_d a b c n))^(1/3 : ℂ) = n^3 + 1 :=
  sorry

end cubic_root_of_determinant_l76_76747


namespace black_car_overtakes_red_car_in_one_hour_l76_76182

-- Define the speeds of the cars
def red_car_speed := 30 -- in miles per hour
def black_car_speed := 50 -- in miles per hour

-- Define the initial distance between the cars
def initial_distance := 20 -- in miles

-- Calculate the time required for the black car to overtake the red car
theorem black_car_overtakes_red_car_in_one_hour : initial_distance / (black_car_speed - red_car_speed) = 1 := by
  sorry

end black_car_overtakes_red_car_in_one_hour_l76_76182


namespace combined_mpg_l76_76589

theorem combined_mpg :
  let mR := 150 -- miles Ray drives
  let mT := 300 -- miles Tom drives
  let mpgR := 50 -- miles per gallon for Ray's car
  let mpgT := 20 -- miles per gallon for Tom's car
  -- Total gasoline used by Ray and Tom
  let gR := mR / mpgR
  let gT := mT / mpgT
  -- Total distance driven
  let total_distance := mR + mT
  -- Total gasoline used
  let total_gasoline := gR + gT
  -- Combined miles per gallon
  let combined_mpg := total_distance / total_gasoline
  combined_mpg = 25 := by
    sorry

end combined_mpg_l76_76589


namespace probability_of_winning_l76_76326

theorem probability_of_winning (P_lose : ℚ) (h : P_lose = 3 / 7) : 
  let P_win := 1 - P_lose in P_win = 4 / 7 :=
by
  sorry

end probability_of_winning_l76_76326


namespace terminating_decimal_expansion_l76_76091

theorem terminating_decimal_expansion (a b : ℕ) (h : 1600 = 2^6 * 5^2) :
  (13 : ℚ) / 1600 = 65 / 1000 :=
by
  sorry

end terminating_decimal_expansion_l76_76091


namespace distinct_solution_difference_l76_76154

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l76_76154


namespace range_of_m_l76_76099

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (Set.Ioc 0 2) then 2^x - 1 else sorry

def g (x m : ℝ) : ℝ :=
x^2 - 2*x + m

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-2:ℝ) 2, f (-x) = -f x) ∧
  (∀ x ∈ Set.Ioc (0:ℝ) 2, f x = 2^x - 1) ∧
  (∀ x1 ∈ Set.Icc (-2:ℝ) 2, ∃ x2 ∈ Set.Icc (-2:ℝ) 2, g x2 m = f x1) 
  → -5 ≤ m ∧ m ≤ -2 :=
sorry

end range_of_m_l76_76099


namespace equivalent_math_problem_l76_76093

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := - (Real.sqrt 1011 + Real.sqrt 1012)
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem equivalent_math_problem :
  (P * Q)^2 * R * S = 8136957 :=
by
  sorry

end equivalent_math_problem_l76_76093


namespace least_common_multiple_first_ten_l76_76821

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76821


namespace total_road_signs_l76_76209

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end total_road_signs_l76_76209


namespace find_middle_number_l76_76180

theorem find_middle_number (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 12.5)
  (h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h3 : (a + b + c) / 3 = 11.6)
  (h4 : (c + d + e) / 3 = 13.5) : c = 12.8 :=
sorry

end find_middle_number_l76_76180


namespace goldfish_equal_after_8_months_l76_76992

noncomputable def B (n : ℕ) : ℝ := 3^(n + 1)
noncomputable def G (n : ℕ) : ℝ := 243 * 1.5^n

theorem goldfish_equal_after_8_months :
  ∃ n : ℕ, B n = G n ∧ n = 8 :=
by
  sorry

end goldfish_equal_after_8_months_l76_76992


namespace feet_per_inch_of_model_l76_76789

def height_of_statue := 75 -- in feet
def height_of_model := 5 -- in inches

theorem feet_per_inch_of_model : (height_of_statue / height_of_model) = 15 :=
by
  sorry

end feet_per_inch_of_model_l76_76789


namespace least_common_multiple_of_first_ten_l76_76925

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76925


namespace range_of_a_l76_76266

noncomputable def common_point_ellipse_parabola (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y

theorem range_of_a : ∀ a : ℝ, common_point_ellipse_parabola a → -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l76_76266


namespace farmer_apples_l76_76323

theorem farmer_apples (initial_apples : ℕ) (given_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 127) (h2 : given_apples = 88) 
  (h3 : final_apples = initial_apples - given_apples) : final_apples = 39 :=
by {
  -- proof steps would go here, but since only the statement is needed, we use 'sorry' to skip the proof
  sorry
}

end farmer_apples_l76_76323


namespace problem_solution_l76_76260

theorem problem_solution (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : {d ∈ (nat.prime_divisors x).to_finset | true}.card = 3) (h3 : 31 ∈ nat.prime_divisors x) : x = 32767 :=
sorry

end problem_solution_l76_76260


namespace lcm_first_ten_numbers_l76_76909

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76909


namespace claire_photos_l76_76755

theorem claire_photos (C : ℕ) (h1 : 3 * C = C + 20) : C = 10 :=
sorry

end claire_photos_l76_76755


namespace least_common_multiple_1_to_10_l76_76894

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76894


namespace sufficient_condition_ab_greater_than_1_l76_76052

theorem sufficient_condition_ab_greater_than_1 (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : ab > 1 := 
  sorry

end sufficient_condition_ab_greater_than_1_l76_76052


namespace smallest_value_of_n_l76_76240

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end smallest_value_of_n_l76_76240


namespace problem_inequality_l76_76695

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x - y + z) * (y - z + x) * (z - x + y) ≤ x * y * z := sorry

end problem_inequality_l76_76695


namespace probability_of_winning_l76_76325

def probability_of_losing : ℚ := 3 / 7

theorem probability_of_winning (h : probability_of_losing + p = 1) : p = 4 / 7 :=
by 
  sorry

end probability_of_winning_l76_76325


namespace lcm_first_ten_numbers_l76_76912

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76912


namespace smallest_k_condition_l76_76539

theorem smallest_k_condition (n k : ℕ) (h_n : n ≥ 2) (h_k : k = 2 * n) :
  ∀ (f : Fin n → Fin n → Fin k), (∀ i j, f i j < k) →
  (∃ a b c d : Fin n, a ≠ c ∧ b ≠ d ∧ f a b ≠ f a d ∧ f a b ≠ f c b ∧ f a b ≠ f c d ∧ f a d ≠ f c b ∧ f a d ≠ f c d ∧ f c b ≠ f c d) :=
sorry

end smallest_k_condition_l76_76539


namespace scouts_earnings_over_weekend_l76_76023

def base_pay_per_hour : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def hours_worked_saturday : ℝ := 4.0
def customers_served_saturday : ℝ := 5.0
def hours_worked_sunday : ℝ := 5.0
def customers_served_sunday : ℝ := 8.0

def earnings_saturday : ℝ := (hours_worked_saturday * base_pay_per_hour) + (customers_served_saturday * tip_per_customer)
def earnings_sunday : ℝ := (hours_worked_sunday * base_pay_per_hour) + (customers_served_sunday * tip_per_customer)

def total_earnings : ℝ := earnings_saturday + earnings_sunday

theorem scouts_earnings_over_weekend : total_earnings = 155.00 := by
  sorry

end scouts_earnings_over_weekend_l76_76023


namespace sales_tax_percentage_l76_76067

theorem sales_tax_percentage (total_amount : ℝ) (tip_percentage : ℝ) (food_price : ℝ) (tax_percentage : ℝ) : 
  total_amount = 158.40 ∧ tip_percentage = 0.20 ∧ food_price = 120 → tax_percentage = 0.10 :=
by
  intros h
  sorry

end sales_tax_percentage_l76_76067


namespace proof_method_characterization_l76_76959

-- Definitions of each method
def synthetic_method := "proceeds from cause to effect, in a forward manner"
def analytic_method := "seeks the cause from the effect, working backwards"
def proof_by_contradiction := "assumes the negation of the proposition to be true, and derives a contradiction"
def mathematical_induction := "base case and inductive step: which shows that P holds for all natural numbers"

-- Main theorem to prove
theorem proof_method_characterization :
  (analytic_method == "seeks the cause from the effect, working backwards") :=
by
  sorry

end proof_method_characterization_l76_76959


namespace Mina_digits_l76_76164

theorem Mina_digits (Carlos Sam Mina : ℕ) 
  (h1 : Sam = Carlos + 6) 
  (h2 : Mina = 6 * Carlos) 
  (h3 : Sam = 10) : 
  Mina = 24 := 
sorry

end Mina_digits_l76_76164


namespace least_common_multiple_first_ten_l76_76816

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76816


namespace stddev_newData_l76_76546

-- Definitions and conditions
def variance (data : List ℝ) : ℝ := sorry  -- Placeholder for variance definition
def stddev (data : List ℝ) : ℝ := sorry    -- Placeholder for standard deviation definition

-- Given data
def data : List ℝ := sorry                -- Placeholder for the data x_1, x_2, ..., x_8
def newData : List ℝ := data.map (λ x => 2 * x + 1)

-- Given condition
axiom variance_data : variance data = 16

-- Proof of the statement
theorem stddev_newData : stddev newData = 8 :=
by {
  sorry
}

end stddev_newData_l76_76546


namespace circle_intersection_l76_76741

theorem circle_intersection : 
  ∀ (O : ℝ × ℝ), ∃ (m n : ℤ), (dist (O.1, O.2) (m, n) ≤ 100 + 1/14) := 
sorry

end circle_intersection_l76_76741


namespace system_solution_equation_solution_l76_76317

-- Proof problem for the first system of equations
theorem system_solution (x y : ℝ) : 
  (2 * x + 3 * y = 8) ∧ (3 * x - 5 * y = -7) → (x = 1 ∧ y = 2) :=
by sorry

-- Proof problem for the second equation
theorem equation_solution (x : ℝ) : 
  ((x - 2) / (x + 2) - 12 / (x^2 - 4) = 1) → (x = -1) :=
by sorry

end system_solution_equation_solution_l76_76317


namespace amount_of_loan_l76_76456

theorem amount_of_loan (P R T SI : ℝ) (hR : R = 6) (hT : T = 6) (hSI : SI = 432) :
  SI = (P * R * T) / 100 → P = 1200 :=
by
  intro h
  sorry

end amount_of_loan_l76_76456


namespace maxwell_walking_speed_l76_76449

theorem maxwell_walking_speed :
  ∀ (distance_between_homes : ℕ)
    (brad_speed : ℕ)
    (middle_travel_maxwell : ℕ)
    (middle_distance : ℕ),
    distance_between_homes = 36 →
    brad_speed = 4 →
    middle_travel_maxwell = 12 →
    middle_distance = 18 →
    (middle_travel_maxwell : ℕ) / (8 : ℕ) = (middle_distance - middle_travel_maxwell) / brad_speed :=
  sorry

end maxwell_walking_speed_l76_76449


namespace lcm_first_ten_integers_l76_76812

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76812


namespace square_area_max_l76_76214

theorem square_area_max (perimeter : ℝ) (h_perimeter : perimeter = 32) : 
  ∃ (area : ℝ), area = 64 :=
by
  sorry

end square_area_max_l76_76214


namespace largest_number_is_56_l76_76196

-- Definitions based on the conditions
def ratio_three_five_seven (a b c : ℕ) : Prop :=
  3 * c = a ∧ 5 * c = b ∧ 7 * c = c

def difference_is_32 (a c : ℕ) : Prop :=
  c - a = 32

-- Statement of the proof
theorem largest_number_is_56 (a b c : ℕ) (h1 : ratio_three_five_seven a b c) (h2 : difference_is_32 a c) : c = 56 :=
by
  sorry

end largest_number_is_56_l76_76196


namespace larger_number_l76_76040

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l76_76040


namespace base8_subtraction_l76_76678

theorem base8_subtraction : 
  let a := 6 * 8^2 + 4 * 8^1 + 1 * 8^0
  let b := 3 * 8^2 + 2 * 8^1 + 4 * 8^0
  a - b = 3 * 8^2 + 1 * 8^1 + 7 * 8^0 := 
by 
  sorry

end base8_subtraction_l76_76678


namespace san_antonio_bus_passes_4_austin_buses_l76_76364

theorem san_antonio_bus_passes_4_austin_buses :
  ∀ (hourly_austin_buses : ℕ → ℕ) (every_50_minute_san_antonio_buses : ℕ → ℕ) (trip_time : ℕ),
    (∀ h : ℕ, hourly_austin_buses (h) = (h * 60)) →
    (∀ m : ℕ, every_50_minute_san_antonio_buses (m) = (m * 60 + 50)) →
    trip_time = 240 →
    ∃ num_buses_passed : ℕ, num_buses_passed = 4 :=
by
  sorry

end san_antonio_bus_passes_4_austin_buses_l76_76364


namespace num_complementary_sets_eq_117_l76_76387

structure Card :=
(shape : Type)
(color : Type)
(shade : Type)

def deck_condition: Prop := 
  ∃ (deck : List Card), 
  deck.length = 27 ∧
  ∀ c1 c2 c3, c1 ∈ deck ∧ c2 ∈ deck ∧ c3 ∈ deck →
  (c1.shape ≠ c2.shape ∨ c2.shape ≠ c3.shape ∨ c1.shape = c3.shape) ∧
  (c1.color ≠ c2.color ∨ c2.color ≠ c3.color ∨ c1.color = c3.color) ∧
  (c1.shade ≠ c2.shade ∨ c2.shade ≠ c3.shade ∨ c1.shade = c3.shade)

theorem num_complementary_sets_eq_117 :
  deck_condition → ∃ sets : List (List Card), sets.length = 117 := sorry

end num_complementary_sets_eq_117_l76_76387


namespace perpendicular_AD_BC_l76_76746

open Real

variables {A B C D E F P Q R M N : Point}
-- Assume basic geometric objects and their relationships
variables (triangle : Triangle A B C)
variables (on_bc : D ∈ Segment B C)
variables (on_ca : E ∈ Segment C A)
variables (on_ab : F ∈ Segment A B)
variables (concurrence : Concurrent (Line AD) (Line BE) (Line CF) P)

-- Line passing through A and its intersections with DE and DF
variables (line_through_A : ∃ l, Line l ∧ A ∈ l)
variables (intersect_DE : Q ∈ Ray DE ∧ Q ∈ Line_line_through_A)
variables (intersect_DF : R ∈ Ray DF ∧ R ∈ Line_line_through_A)

-- Points on rays DB and DC
variables (M_on_DB : M ∈ Ray DB)
variables (N_on_DC : N ∈ Ray DC)

-- Given geometric condition
variables (geo_condition : (QN^2 / DN) + (RM^2 / DM) = ((DQ + DR)^2 - 2 * RQ^2 + 2 * DM * DN) / MN)

-- Desired conclusion
theorem perpendicular_AD_BC : Perpendicular (Line AD) (Line BC) :=
sorry

end perpendicular_AD_BC_l76_76746


namespace max_blue_points_l76_76571

theorem max_blue_points (n : ℕ) (h_n : n = 2016) :
  ∃ r : ℕ, r * (2016 - r) = 1008 * 1008 :=
by {
  sorry
}

end max_blue_points_l76_76571


namespace nut_weights_l76_76641

noncomputable def part_weights (total_weight : ℝ) (total_parts : ℝ) : ℝ :=
  total_weight / total_parts

theorem nut_weights
  (total_weight : ℝ)
  (parts_almonds parts_walnuts parts_cashews ratio_pistachios_to_almonds : ℝ)
  (total_parts_without_pistachios total_parts_with_pistachios weight_per_part : ℝ)
  (weights_almonds weights_walnuts weights_cashews weights_pistachios : ℝ) :
  parts_almonds = 5 →
  parts_walnuts = 3 →
  parts_cashews = 2 →
  ratio_pistachios_to_almonds = 1 / 4 →
  total_parts_without_pistachios = parts_almonds + parts_walnuts + parts_cashews →
  total_parts_with_pistachios = total_parts_without_pistachios + (parts_almonds * ratio_pistachios_to_almonds) →
  weight_per_part = total_weight / total_parts_with_pistachios →
  weights_almonds = parts_almonds * weight_per_part →
  weights_walnuts = parts_walnuts * weight_per_part →
  weights_cashews = parts_cashews * weight_per_part →
  weights_pistachios = (parts_almonds * ratio_pistachios_to_almonds) * weight_per_part →
  total_weight = 300 →
  weights_almonds = 133.35 ∧
  weights_walnuts = 80.01 ∧
  weights_cashews = 53.34 ∧
  weights_pistachios = 33.34 :=
by
  intros
  sorry

end nut_weights_l76_76641


namespace sum_squares_reciprocal_l76_76332

variable (x y : ℝ)

theorem sum_squares_reciprocal (h₁ : x + y = 12) (h₂ : x * y = 32) :
  (1/x)^2 + (1/y)^2 = 5/64 := by
  sorry

end sum_squares_reciprocal_l76_76332


namespace fractional_equation_solution_l76_76177

noncomputable def problem_statement (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 = 2 / (x^2 - 1))

theorem fractional_equation_solution :
  ∀ x : ℝ, problem_statement x → x = -2 :=
by
  intro x hx
  sorry

end fractional_equation_solution_l76_76177


namespace tank_filling_l76_76628

theorem tank_filling (A_rate B_rate : ℚ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) :
  (1 / (A_rate - B_rate)) = 18 :=
by
  sorry

end tank_filling_l76_76628


namespace next_podcast_length_l76_76308

theorem next_podcast_length 
  (drive_hours : ℕ := 6)
  (podcast1_minutes : ℕ := 45)
  (podcast2_minutes : ℕ := 90) -- Since twice the first podcast (45 * 2)
  (podcast3_minutes : ℕ := 105) -- 1 hour 45 minutes (60 + 45)
  (podcast4_minutes : ℕ := 60) -- 1 hour 
  (minutes_per_hour : ℕ := 60)
  : (drive_hours * minutes_per_hour - (podcast1_minutes + podcast2_minutes + podcast3_minutes + podcast4_minutes)) / minutes_per_hour = 1 :=
by
  sorry

end next_podcast_length_l76_76308


namespace sum_abs_eq_l76_76559

theorem sum_abs_eq (a b : ℝ) (ha : |a| = 10) (hb : |b| = 7) (hab : a > b) : a + b = 17 ∨ a + b = 3 :=
sorry

end sum_abs_eq_l76_76559


namespace probability_y_greater_than_x_equals_3_4_l76_76996

noncomputable def probability_y_greater_than_x : Real :=
  let total_area : Real := 1000 * 4034
  let triangle_area : Real := 0.5 * 1000 * (4034 - 1000)
  let rectangle_area : Real := 3034 * 4034
  let area_y_greater_than_x : Real := triangle_area + rectangle_area
  area_y_greater_than_x / total_area

theorem probability_y_greater_than_x_equals_3_4 :
  probability_y_greater_than_x = 3 / 4 :=
sorry

end probability_y_greater_than_x_equals_3_4_l76_76996


namespace least_common_multiple_of_first_ten_positive_integers_l76_76860

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76860


namespace find_larger_number_l76_76044

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l76_76044


namespace expected_score_l76_76988

noncomputable def expected_score_problem : ℕ → ℚ := 
  λ n, ( (5 * n) + ((10 - n) * (-1)) )

theorem expected_score (n : ℕ) (p : ℚ) (X : ℕ → ProbabilityMassFunction ℕ) 
  (hx : X n = binomial 10 0.6) : 
  (expected_score_problem ((10:ℕ) * (0.6:ℚ))) = 26 := by
  sorry

end expected_score_l76_76988


namespace length_of_notebook_is_24_l76_76342

-- Definitions
def span_of_hand : ℕ := 12
def length_of_notebook (span : ℕ) : ℕ := 2 * span

-- Theorem statement that proves the question == answer given conditions
theorem length_of_notebook_is_24 :
  length_of_notebook span_of_hand = 24 :=
sorry

end length_of_notebook_is_24_l76_76342


namespace determine_a_l76_76716

-- Define the conditions of the problem.
def linear_in_x_and_y (a : ℝ) : Prop :=
  (a - 2) * (x : ℝ)^(abs a - 1) + 3 * (y : ℝ) = 1

-- Prove that a = -2 under the condition defined.
theorem determine_a (a : ℝ) (h : ∀ x y : ℝ, linear_in_x_and_y a) : a = -2 :=
sorry

end determine_a_l76_76716


namespace cost_price_article_l76_76963

theorem cost_price_article (x : ℝ) (h : 56 - x = x - 42) : x = 49 :=
by sorry

end cost_price_article_l76_76963


namespace suresh_wifes_speed_l76_76451

-- Define conditions
def circumference_of_track : ℝ := 0.726 -- track circumference in kilometers
def suresh_speed : ℝ := 4.5 -- Suresh's speed in km/hr
def meeting_time_in_hours : ℝ := 0.088 -- time till they meet in hours

-- Define the question and expected answer
theorem suresh_wifes_speed : ∃ (V : ℝ), V = 3.75 :=
  by
    -- Let Distance_covered_by_both = circumference_of_track
    let Distance_covered_by_suresh : ℝ := suresh_speed * meeting_time_in_hours
    let Distance_covered_by_suresh_wife : ℝ := circumference_of_track - Distance_covered_by_suresh
    let suresh_wifes_speed : ℝ := Distance_covered_by_suresh_wife / meeting_time_in_hours
    -- Expected answer
    existsi suresh_wifes_speed
    sorry

end suresh_wifes_speed_l76_76451


namespace percent_students_prefer_golf_l76_76604

theorem percent_students_prefer_golf (students_north : ℕ) (students_south : ℕ)
  (percent_golf_north : ℚ) (percent_golf_south : ℚ) :
  students_north = 1800 →
  students_south = 2200 →
  percent_golf_north = 15 →
  percent_golf_south = 25 →
  (820 / 4000 : ℚ) = 20.5 :=
by
  intros h_north h_south h_percent_north h_percent_south
  sorry

end percent_students_prefer_golf_l76_76604


namespace lcm_first_ten_numbers_l76_76911

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76911


namespace polynomial_addition_l76_76657

variable (x : ℝ)

def p := 3 * x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 2
def q := -3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 4

theorem polynomial_addition : p x + q x = -3 * x^3 + 2 * x^2 + 2 := by
  sorry

end polynomial_addition_l76_76657


namespace min_xy_l76_76557

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : xy = x + 4 * y + 5) : xy ≥ 25 :=
sorry

end min_xy_l76_76557


namespace solve_lambda_l76_76095

variable (a b : ℝ × ℝ)
variable (lambda : ℝ)

def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

axiom a_def : a = (-3, 2)
axiom b_def : b = (-1, 0)
axiom perp_def : perpendicular (a.1 + lambda * b.1, a.2 + lambda * b.2) b

theorem solve_lambda : lambda = -3 :=
by
  sorry

end solve_lambda_l76_76095


namespace Amy_bought_tomato_soup_l76_76990

-- Conditions
variables (chicken_soup_cans total_soups : ℕ)
variable (Amy_bought_soups : total_soups = 9)
variable (Amy_bought_chicken_soup : chicken_soup_cans = 6)

-- Question: How many cans of tomato soup did she buy?
def cans_of_tomato_soup (chicken_soup_cans total_soups : ℕ) : ℕ :=
  total_soups - chicken_soup_cans

-- Theorem: Prove that the number of cans of tomato soup Amy bought is 3
theorem Amy_bought_tomato_soup : 
  cans_of_tomato_soup chicken_soup_cans total_soups = 3 :=
by
  rw [Amy_bought_soups, Amy_bought_chicken_soup]
  -- The steps for the proof would follow here
  sorry

end Amy_bought_tomato_soup_l76_76990


namespace product_of_two_numbers_l76_76629

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 400) : x * y = 88 :=
by
  -- Proof goes here
  sorry

end product_of_two_numbers_l76_76629


namespace compound_difference_l76_76411

noncomputable def monthly_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let monthly_rate := annual_rate / 12
  let periods := 12 * years
  principal * (1 + monthly_rate) ^ periods

noncomputable def semi_annual_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let semi_annual_rate := annual_rate / 2
  let periods := 2 * years
  principal * (1 + semi_annual_rate) ^ periods

theorem compound_difference (principal : ℝ) (annual_rate : ℝ) (years : ℝ) :
  monthly_compound_amount principal annual_rate years - semi_annual_compound_amount principal annual_rate years = 23.36 :=
by
  let principal := 8000
  let annual_rate := 0.08
  let years := 3
  sorry

end compound_difference_l76_76411


namespace floor_sqrt_27_square_l76_76676

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end floor_sqrt_27_square_l76_76676


namespace find_set_A_l76_76711

-- Define the set A based on the condition that its elements satisfy a quadratic equation.
def A (a : ℝ) : Set ℝ := {x | x^2 + 2 * x + a = 0}

-- Assume 1 is an element of set A
axiom one_in_A (a : ℝ) (h : 1 ∈ A a) : a = -3

-- The final theorem to prove: Given 1 ∈ A a, A a should be {-3, 1}
theorem find_set_A (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} :=
by sorry

end find_set_A_l76_76711


namespace total_songs_sung_l76_76576

def total_minutes := 80
def intermission_minutes := 10
def long_song_minutes := 10
def short_song_minutes := 5

theorem total_songs_sung : 
  (total_minutes - intermission_minutes - long_song_minutes) / short_song_minutes + 1 = 13 := 
by 
  sorry

end total_songs_sung_l76_76576


namespace least_common_multiple_of_first_ten_l76_76919

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76919


namespace Tony_change_l76_76615

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end Tony_change_l76_76615


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76840

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76840


namespace brittany_average_correct_l76_76080

def brittany_first_score : ℤ :=
78

def brittany_second_score : ℤ :=
84

def brittany_average_after_second_test (score1 score2 : ℤ) : ℤ :=
(score1 + score2) / 2

theorem brittany_average_correct : 
  brittany_average_after_second_test brittany_first_score brittany_second_score = 81 := 
by
  sorry

end brittany_average_correct_l76_76080


namespace semicircle_radius_correct_l76_76630

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_correct (h :127 =113): semicircle_radius 113 = 113 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_correct_l76_76630


namespace next_podcast_length_l76_76307

theorem next_podcast_length 
  (drive_hours : ℕ := 6)
  (podcast1_minutes : ℕ := 45)
  (podcast2_minutes : ℕ := 90) -- Since twice the first podcast (45 * 2)
  (podcast3_minutes : ℕ := 105) -- 1 hour 45 minutes (60 + 45)
  (podcast4_minutes : ℕ := 60) -- 1 hour 
  (minutes_per_hour : ℕ := 60)
  : (drive_hours * minutes_per_hour - (podcast1_minutes + podcast2_minutes + podcast3_minutes + podcast4_minutes)) / minutes_per_hour = 1 :=
by
  sorry

end next_podcast_length_l76_76307


namespace negation_of_square_zero_l76_76472

variable {m : ℝ}

def is_positive (m : ℝ) : Prop := m > 0
def square_is_zero (m : ℝ) : Prop := m^2 = 0

theorem negation_of_square_zero (h : ∀ m, is_positive m → square_is_zero m) :
  ∀ m, ¬ is_positive m → ¬ square_is_zero m := 
sorry

end negation_of_square_zero_l76_76472


namespace even_function_a_value_l76_76117

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x - 1) = ((-x)^2 + a * (-x) - 1)) ↔ a = 0 :=
by
  sorry

end even_function_a_value_l76_76117


namespace calculate_length_of_floor_l76_76601

-- Define the conditions and the objective to prove
variable (breadth length : ℝ)
variable (cost rate : ℝ)
variable (area : ℝ)

-- Given conditions
def length_more_by_percentage : Prop := length = 2 * breadth
def painting_cost : Prop := cost = 529 ∧ rate = 3

-- Objective
def length_of_floor : ℝ := 2 * breadth

theorem calculate_length_of_floor : 
  (length_more_by_percentage breadth length) →
  (painting_cost cost rate) →
  length_of_floor breadth = 18.78 :=
by
  sorry

end calculate_length_of_floor_l76_76601


namespace system_of_equations_solution_l76_76528

theorem system_of_equations_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : x1 + x2 = x3^2) (h2 : x2 + x3 = x4^2)
  (h3 : x3 + x4 = x5^2) (h4 : x4 + x5 = x1^2) (h5 : x5 + x1 = x2^2) :
  x1 = 2 ∧ x2 = 2 ∧ x3 = 2 ∧ x4 = 2 ∧ x5 = 2 := 
sorry

end system_of_equations_solution_l76_76528


namespace sqrt_floor_square_l76_76674

theorem sqrt_floor_square {x : ℝ} (hx : 27 = x) (h1 : sqrt 25 < sqrt x) (h2 : sqrt x < sqrt 36) :
  (⌊sqrt x⌋.to_real)^2 = 25 :=
by {
  have hsqrt : 5 < sqrt x ∧ sqrt x < 6, by {
    split; linarith,
  },
  have h_floor_sqrt : ⌊sqrt x⌋ = 5, by {
    exact int.floor_eq_iff.mpr ⟨int.lt_floor_add_one.mpr hsqrt.2, hsqrt.1⟩,
  },
  rw h_floor_sqrt,
  norm_num,
  sorry  -- proof elided
}

end sqrt_floor_square_l76_76674


namespace least_common_multiple_first_ten_l76_76817

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76817


namespace fraction_div_subtract_l76_76795

theorem fraction_div_subtract : 
  (5 / 6 : ℚ) / (9 / 10) - (1 / 15) = 116 / 135 := 
by 
  sorry

end fraction_div_subtract_l76_76795


namespace variable_cost_per_book_fixed_cost_l76_76505

theorem variable_cost_per_book_fixed_cost (fixed_costs : ℝ) (selling_price_per_book : ℝ) 
(number_of_books : ℝ) (total_costs total_revenue : ℝ) (variable_cost_per_book : ℝ) 
(h1 : fixed_costs = 35630) (h2 : selling_price_per_book = 20.25) (h3 : number_of_books = 4072)
(h4 : total_costs = fixed_costs + variable_cost_per_book * number_of_books)
(h5 : total_revenue = selling_price_per_book * number_of_books)
(h6 : total_costs = total_revenue) : variable_cost_per_book = 11.50 := by
  sorry

end variable_cost_per_book_fixed_cost_l76_76505


namespace problem_statement_l76_76141

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l76_76141


namespace least_common_multiple_first_ten_l76_76830

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l76_76830


namespace deepak_wife_speed_l76_76161

-- Definitions and conditions
def track_circumference_km : ℝ := 0.66
def deepak_speed_kmh : ℝ := 4.5
def time_to_meet_hr : ℝ := 0.08

-- Theorem statement
theorem deepak_wife_speed
  (track_circumference_km : ℝ)
  (deepak_speed_kmh : ℝ)
  (time_to_meet_hr : ℝ)
  (deepak_distance : ℝ := deepak_speed_kmh * time_to_meet_hr)
  (wife_distance : ℝ := track_circumference_km - deepak_distance)
  (wife_speed_kmh : ℝ := wife_distance / time_to_meet_hr) : 
  wife_speed_kmh = 3.75 :=
sorry

end deepak_wife_speed_l76_76161


namespace least_common_multiple_first_ten_integers_l76_76938

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76938


namespace find_x7_plus_32x2_l76_76156

theorem find_x7_plus_32x2 (x : ℝ) (h : x^3 + 2 * x = 4) : x^7 + 32 * x^2 = 64 :=
sorry

end find_x7_plus_32x2_l76_76156


namespace total_canoes_built_by_april_l76_76668

theorem total_canoes_built_by_april
  (initial : ℕ)
  (production_increase : ℕ → ℕ) 
  (total_canoes : ℕ) :
  initial = 5 →
  (∀ n, production_increase n = 3 * n) →
  total_canoes = initial + production_increase initial + production_increase (production_increase initial) + production_increase (production_increase (production_increase initial)) →
  total_canoes = 200 :=
by
  intros h_initial h_production h_total
  sorry

end total_canoes_built_by_april_l76_76668


namespace lcm_first_ten_l76_76848

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76848


namespace least_common_multiple_of_first_ten_positive_integers_l76_76863

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76863


namespace complete_the_square_d_l76_76367

theorem complete_the_square_d (x : ℝ) (h : x^2 + 6 * x + 5 = 0) : ∃ d : ℝ, (x + 3)^2 = d ∧ d = 4 :=
by
  sorry

end complete_the_square_d_l76_76367


namespace intersection_nonempty_implies_m_eq_zero_l76_76272

theorem intersection_nonempty_implies_m_eq_zero (m : ℤ) (P Q : Set ℝ)
  (hP : P = { -1, ↑m } ) (hQ : Q = { x : ℝ | -1 < x ∧ x < 3/4 }) (h : (P ∩ Q).Nonempty) :
  m = 0 :=
by
  sorry

end intersection_nonempty_implies_m_eq_zero_l76_76272


namespace longest_segment_in_cylinder_l76_76974

theorem longest_segment_in_cylinder (radius height : ℝ) 
  (hr : radius = 5) (hh : height = 10) :
  ∃ segment_length, segment_length = 10 * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end longest_segment_in_cylinder_l76_76974


namespace budget_per_friend_l76_76432

-- Definitions for conditions
def total_budget : ℕ := 100
def parents_gift_cost : ℕ := 14
def number_of_parents : ℕ := 2
def number_of_friends : ℕ := 8

-- Statement to prove
theorem budget_per_friend :
  (total_budget - number_of_parents * parents_gift_cost) / number_of_friends = 9 :=
by
  sorry

end budget_per_friend_l76_76432


namespace determine_digit_z_l76_76378

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l76_76378


namespace nested_fraction_eval_l76_76246

theorem nested_fraction_eval : (1 / (1 + (1 / (2 + (1 / (1 + (1 / 4))))))) = (14 / 19) :=
by
  sorry

end nested_fraction_eval_l76_76246


namespace find_g_function_l76_76168

noncomputable def g : ℝ → ℝ :=
  sorry

theorem find_g_function (x y : ℝ) (h1 : g 1 = 2) (h2 : ∀ (x y : ℝ), g (x + y) = 5^y * g x + 3^x * g y) :
  g x = 5^x - 3^x :=
by
  sorry

end find_g_function_l76_76168


namespace vehicle_count_l76_76334

theorem vehicle_count (T B : ℕ) (h1 : T + B = 15) (h2 : 3 * T + 2 * B = 40) : T = 10 ∧ B = 5 :=
by
  sorry

end vehicle_count_l76_76334


namespace least_common_multiple_1_to_10_l76_76887

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76887


namespace lcm_first_ten_positive_integers_l76_76878

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76878


namespace missed_angle_l76_76611

theorem missed_angle (n : ℕ) (h1 : (n - 2) * 180 ≥ 3239) (h2 : n ≥ 3) : 3240 - 3239 = 1 :=
by
  sorry

end missed_angle_l76_76611


namespace ratio_of_adults_to_children_closest_to_one_l76_76778

theorem ratio_of_adults_to_children_closest_to_one (a c : ℕ) 
  (h₁ : 25 * a + 12 * c = 1950) 
  (h₂ : a ≥ 1) 
  (h₃ : c ≥ 1) : (a : ℚ) / (c : ℚ) = 27 / 25 := 
by 
  sorry

end ratio_of_adults_to_children_closest_to_one_l76_76778


namespace max_ounces_amber_can_get_l76_76663

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l76_76663


namespace assignment_plans_l76_76666

theorem assignment_plans {students towns : ℕ} (h_students : students = 5) (h_towns : towns = 3) :
  ∃ plans : ℕ, plans = 150 :=
by
  -- Given conditions
  have h1 : students = 5 := h_students
  have h2 : towns = 3 := h_towns
  
  -- The required number of assignment plans
  existsi 150
  -- Proof is not supplied
  sorry

end assignment_plans_l76_76666


namespace prob_x_lt_1_l76_76101

noncomputable def normalDist : Distribution := NormalDistribution.mk 2 1

axiom P_1_le_x_le_3: ℝ := 0.6826

theorem prob_x_lt_1 : ∀ (x : ℝ), ProbabilityDensityFunction normalDist x < 1 = 0.1587 :=
by
  intro x
  sorry

end prob_x_lt_1_l76_76101


namespace complex_fraction_equivalence_l76_76465

/-- The complex number 2 / (1 - i) is equal to 1 + i. -/
theorem complex_fraction_equivalence : (2 : ℂ) / (1 - (I : ℂ)) = 1 + (I : ℂ) := by
  sorry

end complex_fraction_equivalence_l76_76465


namespace range_of_a1_l76_76606

theorem range_of_a1 (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 1 / (2 - a n)) (h2 : ∀ n, a (n + 1) > a n) :
  a 1 < 1 :=
sorry

end range_of_a1_l76_76606


namespace complex_triple_sum_eq_sqrt3_l76_76749

noncomputable section

open Complex

theorem complex_triple_sum_eq_sqrt3 {a b c : ℂ} (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : a + b + c ≠ 0) (h5 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3) : abs (a + b + c) = Real.sqrt 3 :=
by
  sorry

end complex_triple_sum_eq_sqrt3_l76_76749


namespace log_expression_equals_four_l76_76234

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l76_76234


namespace total_road_signs_l76_76210

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end total_road_signs_l76_76210


namespace tan_alpha_eq_two_imp_inv_sin_double_angle_l76_76396

theorem tan_alpha_eq_two_imp_inv_sin_double_angle (α : ℝ) (h : Real.tan α = 2) : 
  (1 / Real.sin (2 * α)) = 5 / 4 :=
by
  sorry

end tan_alpha_eq_two_imp_inv_sin_double_angle_l76_76396


namespace maximize_volume_l76_76506

theorem maximize_volume
  (R H A : ℝ) (K : ℝ) (hA : 2 * π * R * H + 2 * π * R * (Real.sqrt (R ^ 2 + H ^ 2)) = A)
  (hK : K = A / (2 * π)) :
  R = (A / (π * Real.sqrt 5)) ^ (1 / 3) :=
sorry

end maximize_volume_l76_76506


namespace amber_max_ounces_l76_76659

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l76_76659


namespace N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l76_76452

open Nat -- Natural numbers framework

-- Definitions for game conditions would go here. We assume them to be defined as:
-- structure GameCondition (N : ℕ) :=
-- (players_take_turns_to_circle_numbers_from_1_to_N : Prop)
-- (any_two_circled_numbers_must_be_coprime : Prop)
-- (a_number_cannot_be_circled_twice : Prop)
-- (player_who_cannot_move_loses : Prop)

inductive Player
| first
| second

-- Definitions indicating which player wins for a given N
def first_player_wins (N : ℕ) : Prop := sorry
def second_player_wins (N : ℕ) : Prop := sorry

-- For N = 10
theorem N_10_first_player_wins : first_player_wins 10 := sorry

-- For N = 12
theorem N_12_first_player_wins : first_player_wins 12 := sorry

-- For N = 15
theorem N_15_second_player_wins : second_player_wins 15 := sorry

-- For N = 30
theorem N_30_first_player_wins : first_player_wins 30 := sorry

end N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l76_76452


namespace max_lcm_15_2_3_5_6_9_10_l76_76621

theorem max_lcm_15_2_3_5_6_9_10 : 
  max (max (max (max (max (Nat.lcm 15 2) (Nat.lcm 15 3)) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10) = 45 :=
by
  sorry

end max_lcm_15_2_3_5_6_9_10_l76_76621


namespace math_pattern_l76_76759

theorem math_pattern (n : ℕ) : (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by
  sorry

end math_pattern_l76_76759


namespace solve_for_T_l76_76336

theorem solve_for_T : ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (2 / 5) * (1 / 4) * 200 ∧ T = 80 :=
by
  use 80
  -- The proof part is omitted as instructed
  sorry

end solve_for_T_l76_76336


namespace number_of_registration_methods_l76_76113

theorem number_of_registration_methods
  (students : ℕ) (groups : ℕ) (registration_methods : ℕ)
  (h_students : students = 4) (h_groups : groups = 3) :
  registration_methods = groups ^ students :=
by
  rw [h_students, h_groups]
  exact sorry

end number_of_registration_methods_l76_76113


namespace least_common_multiple_first_ten_integers_l76_76943

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76943


namespace old_selling_price_l76_76492

theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) :
  C + 0.10 * C = 88 :=
by
  sorry

end old_selling_price_l76_76492


namespace lcm_first_ten_l76_76850

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76850


namespace algebra_expr_value_l76_76545

theorem algebra_expr_value (x y : ℝ) (h : x - 2 * y = 3) : 4 * y + 1 - 2 * x = -5 :=
sorry

end algebra_expr_value_l76_76545


namespace prime_square_mod_12_l76_76767

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end prime_square_mod_12_l76_76767


namespace range_xf_ge_0_l76_76280

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 2 else - (-x) - 2

theorem range_xf_ge_0 :
  { x : ℝ | x * f x ≥ 0 } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_xf_ge_0_l76_76280


namespace least_common_multiple_of_first_ten_l76_76924

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76924


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76841

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76841


namespace molecular_weight_of_1_mole_l76_76185

theorem molecular_weight_of_1_mole (m : ℝ) (w : ℝ) (h : 7 * m = 420) : m = 60 :=
by
  sorry

end molecular_weight_of_1_mole_l76_76185


namespace larger_number_l76_76041

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l76_76041


namespace part1_part2_l76_76406

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 :
  (∀ x, f x ≤ 9) → ∀ x, x ∈ (Icc (-2 : ℝ) (4 : ℝ)) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) ↔ a ∈ Icc (19 / 4 : ℝ) (7 : ℝ) :=
by
  sorry

end part1_part2_l76_76406


namespace sum_of_coordinates_of_D_l76_76022

def Point := (ℝ × ℝ)

def isMidpoint (M C D : Point) : Prop :=
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem sum_of_coordinates_of_D (M C : Point) (D : Point) (hM : isMidpoint M C D) (hC : C = (2, 10)) :
  D.1 + D.2 = 12 :=
sorry

end sum_of_coordinates_of_D_l76_76022


namespace distinct_solution_difference_l76_76155

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l76_76155


namespace rate_per_sqm_is_correct_l76_76784

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end rate_per_sqm_is_correct_l76_76784


namespace correct_quotient_l76_76055

theorem correct_quotient (D Q : ℕ) (h1 : 21 * Q = 12 * 56) : Q = 32 :=
by {
  -- Proof to be provided
  sorry
}

end correct_quotient_l76_76055


namespace least_common_multiple_of_first_ten_positive_integers_l76_76857

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76857


namespace A_speed_ratio_B_speed_l76_76358

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end A_speed_ratio_B_speed_l76_76358


namespace binomial_expansion_equality_l76_76351

theorem binomial_expansion_equality (x : ℝ) : 
  (x-1)^4 - 4*x*(x-1)^3 + 6*(x^2)*(x-1)^2 - 4*(x^3)*(x-1)*x^4 = 1 := 
by 
  sorry 

end binomial_expansion_equality_l76_76351


namespace least_common_multiple_of_first_ten_integers_l76_76930

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76930


namespace number_of_roots_eq_seven_l76_76232

noncomputable def problem_function (x : ℝ) : ℝ :=
  (21 * x - 11 + (Real.sin x) / 100) * Real.sin (6 * Real.arcsin x) * Real.sqrt ((Real.pi - 6 * x) * (Real.pi + x))

theorem number_of_roots_eq_seven :
  (∃ xs : List ℝ, (∀ x ∈ xs, problem_function x = 0) ∧ (∀ x ∈ xs, -1 ≤ x ∧ x ≤ 1) ∧ xs.length = 7) :=
sorry

end number_of_roots_eq_seven_l76_76232


namespace sluice_fill_time_l76_76077

noncomputable def sluice_open_equal_time (x y : ℝ) (m : ℝ) : ℝ :=
  -- Define time (t) required for both sluice gates to be open equally to fill the lake
  m / 11

theorem sluice_fill_time :
  ∀ (x y : ℝ),
    (10 * x + 14 * y = 9900) →
    (18 * x + 12 * y = 9900) →
    sluice_open_equal_time x y 9900 = 900 := sorry

end sluice_fill_time_l76_76077


namespace circle_ellipse_intersect_four_points_l76_76236

theorem circle_ellipse_intersect_four_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = a^2 → y = x^2 / 2 - a) →
  a > 1 :=
by
  sorry

end circle_ellipse_intersect_four_points_l76_76236


namespace lcm_first_ten_l76_76854

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76854


namespace least_common_multiple_1_to_10_l76_76890

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76890


namespace product_of_faces_and_vertices_of_cube_l76_76683

def number_of_faces := 6
def number_of_vertices := 8

theorem product_of_faces_and_vertices_of_cube : number_of_faces * number_of_vertices = 48 := 
by 
  sorry

end product_of_faces_and_vertices_of_cube_l76_76683


namespace find_larger_number_l76_76039

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l76_76039


namespace sum_of_ages_l76_76989

variable (a b c : ℕ)

theorem sum_of_ages (h1 : a = 20 + b + c) (h2 : a^2 = 2000 + (b + c)^2) : a + b + c = 80 := 
by
  sorry

end sum_of_ages_l76_76989


namespace mass_percentage_of_carbon_in_ccl4_l76_76251

-- Define the atomic masses
def atomic_mass_c : Float := 12.01
def atomic_mass_cl : Float := 35.45

-- Define the molecular composition of Carbon Tetrachloride (CCl4)
def mol_mass_ccl4 : Float := (1 * atomic_mass_c) + (4 * atomic_mass_cl)

-- Theorem to prove the mass percentage of carbon in Carbon Tetrachloride is 7.81%
theorem mass_percentage_of_carbon_in_ccl4 : 
  (atomic_mass_c / mol_mass_ccl4) * 100 = 7.81 := by
  sorry

end mass_percentage_of_carbon_in_ccl4_l76_76251


namespace units_produced_by_line_B_l76_76502

-- State the problem with the given conditions and prove the question equals the answer.
theorem units_produced_by_line_B (total_units : ℕ) (B : ℕ) (A C : ℕ) 
    (h1 : total_units = 13200)
    (h2 : A + B + C = total_units)
    (h3 : ∃ d : ℕ, A = B - d ∧ C = B + d) :
    B = 4400 :=
by
  sorry

end units_produced_by_line_B_l76_76502


namespace multiply_fractions_l76_76339

theorem multiply_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = (1 / 7) := by
  sorry

end multiply_fractions_l76_76339


namespace polynomial_factorization_l76_76732

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end polynomial_factorization_l76_76732


namespace point_not_on_graph_l76_76513

theorem point_not_on_graph :
  ¬ (1 * 5 = 6) :=
by 
  sorry

end point_not_on_graph_l76_76513


namespace perimeter_of_AF1B_l76_76268

noncomputable def ellipse_perimeter (a b x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (2 * a)

theorem perimeter_of_AF1B (h : (6:ℝ) = 6) :
  ellipse_perimeter 6 4 0 0 6 0 = 24 :=
by
  sorry

end perimeter_of_AF1B_l76_76268


namespace Seohyeon_l76_76431

-- Define the distances in their respective units
def d_Kunwoo_km : ℝ := 3.97
def d_Seohyeon_m : ℝ := 4028

-- Convert Kunwoo's distance to meters
def d_Kunwoo_m : ℝ := d_Kunwoo_km * 1000

-- The main theorem we need to prove
theorem Seohyeon's_distance_longer_than_Kunwoo's :
  d_Seohyeon_m > d_Kunwoo_m :=
by
  sorry

end Seohyeon_l76_76431


namespace min_distinct_values_l76_76646

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) 
  (h_mode : mode_count = 10) (h_total : total_count = 2018) 
  (h_distinct : ∀ k, k ≠ mode_count → k < 10) : 
  n ≥ 225 :=
by
  sorry

end min_distinct_values_l76_76646


namespace lcm_first_ten_positive_integers_l76_76880

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76880


namespace area_quadrilateral_l76_76737

theorem area_quadrilateral (EF GH: ℝ) (EHG: ℝ) 
  (h1 : EF = 9) (h2 : GH = 12) (h3 : GH = EH) (h4 : EHG = 75) 
  (a b c : ℕ)
  : 
  (∀ (a b c : ℕ), 
  a = 26 ∧ b = 18 ∧ c = 6 → 
  a + b + c = 50) := 
sorry

end area_quadrilateral_l76_76737


namespace next_equalities_from_conditions_l76_76348

-- Definitions of the equality conditions
def eq1 : Prop := 3^2 + 4^2 = 5^2
def eq2 : Prop := 10^2 + 11^2 + 12^2 = 13^2 + 14^2
def eq3 : Prop := 21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2
def eq4 : Prop := 36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2

-- The next equalities we want to prove
def eq5 : Prop := 55^2 + 56^2 + 57^2 + 58^2 + 59^2 + 60^2 = 61^2 + 62^2 + 63^2 + 64^2 + 65^2
def eq6 : Prop := 78^2 + 79^2 + 80^2 + 81^2 + 82^2 + 83^2 + 84^2 = 85^2 + 86^2 + 87^2 + 88^2 + 89^2 + 90^2

theorem next_equalities_from_conditions : eq1 → eq2 → eq3 → eq4 → (eq5 ∧ eq6) :=
by
  sorry

end next_equalities_from_conditions_l76_76348


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76838

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76838


namespace least_common_multiple_1_to_10_l76_76899

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76899


namespace distinct_solutions_difference_l76_76150

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l76_76150


namespace increasing_sequence_range_l76_76271

theorem increasing_sequence_range (a : ℝ) (a_seq : ℕ → ℝ)
  (h₁ : ∀ (n : ℕ), n ≤ 5 → a_seq n = (5 - a) * n - 11)
  (h₂ : ∀ (n : ℕ), n > 5 → a_seq n = a ^ (n - 4))
  (h₃ : ∀ (n : ℕ), a_seq n < a_seq (n + 1)) :
  2 < a ∧ a < 5 := 
sorry

end increasing_sequence_range_l76_76271


namespace apple_cost_price_l76_76984

theorem apple_cost_price (SP : ℝ) (loss_ratio : ℝ) (CP : ℝ) (h1 : SP = 18) (h2 : loss_ratio = 1/6) (h3 : SP = CP - loss_ratio * CP) : CP = 21.6 :=
by
  sorry

end apple_cost_price_l76_76984


namespace cone_volume_divided_by_pi_l76_76062

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def sector_to_cone_radius (arc_len : ℝ) : ℝ := arc_len / (2 * Real.pi)

noncomputable def cone_height (r_base : ℝ) (slant_height : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - r_base ^ 2)

noncomputable def cone_volume (r_base : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r_base ^ 2 * height

theorem cone_volume_divided_by_pi (r slant_height θ : ℝ) (h : slant_height = 15 ∧ θ = 270):
  cone_volume (sector_to_cone_radius (arc_length r θ)) (cone_height (sector_to_cone_radius (arc_length r θ)) slant_height) / Real.pi = (453.515625 * Real.sqrt 10.9375) :=
by
  sorry

end cone_volume_divided_by_pi_l76_76062


namespace relationship_abc_l76_76257

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l76_76257


namespace unique_element_a_values_set_l76_76115

open Set

theorem unique_element_a_values_set :
  {a : ℝ | ∃! x : ℝ, a * x^2 + 2 * x - a = 0} = {0} :=
by
  sorry

end unique_element_a_values_set_l76_76115


namespace maximize_x3y4_l76_76752

noncomputable def max_product (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 50) : ℝ :=
  x^3 * y^4

theorem maximize_x3y4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 50) :
  max_product x y hx hy h ≤ max_product (150/7) (200/7) (by norm_num) (by norm_num) (by norm_num) :=
  sorry

end maximize_x3y4_l76_76752


namespace solve_inequality_l76_76592

theorem solve_inequality :
  {x : ℝ | -3 * x^2 + 5 * x + 4 < 0} = {x : ℝ | x < 3 / 4} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solve_inequality_l76_76592


namespace marbles_lost_correct_l76_76435

-- Define the initial number of marbles
def initial_marbles : ℕ := 16

-- Define the current number of marbles
def current_marbles : ℕ := 9

-- Define the number of marbles lost
def marbles_lost (initial current : ℕ) : ℕ := initial - current

-- State the proof problem: Given the conditions, prove the number of marbles lost is 7
theorem marbles_lost_correct : marbles_lost initial_marbles current_marbles = 7 := by
  sorry

end marbles_lost_correct_l76_76435


namespace lcm_first_ten_numbers_l76_76906

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l76_76906


namespace D_is_necessary_but_not_sufficient_condition_for_A_l76_76012

variable (A B C D : Prop)

-- Conditions
axiom A_implies_B : A → B
axiom not_B_implies_A : ¬ (B → A)
axiom B_iff_C : B ↔ C
axiom C_implies_D : C → D
axiom not_D_implies_C : ¬ (D → C)

theorem D_is_necessary_but_not_sufficient_condition_for_A : (A → D) ∧ ¬ (D → A) :=
by sorry

end D_is_necessary_but_not_sufficient_condition_for_A_l76_76012


namespace evaluate_g_h_2_l76_76109

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end evaluate_g_h_2_l76_76109


namespace maurice_rides_before_visit_l76_76302

-- Defining all conditions in Lean
variables
  (M : ℕ) -- Number of times Maurice had been horseback riding before visiting Matt
  (Matt_rides_with_M : ℕ := 8 * 2) -- Number of times Matt rode with Maurice (8 times, 2 horses each time)
  (Matt_rides_alone : ℕ := 16) -- Number of times Matt rode solo
  (total_Matt_rides : ℕ := Matt_rides_with_M + Matt_rides_alone) -- Total rides by Matt
  (three_times_M : ℕ := 3 * M) -- Three times the number of times Maurice rode before visiting
  (unique_horses_M : ℕ := 8) -- Total number of unique horses Maurice rode during his visit

-- Main theorem
theorem maurice_rides_before_visit  
  (h1: total_Matt_rides = three_times_M) 
  (h2: unique_horses_M = M) 
  : M = 10 := sorry

end maurice_rides_before_visit_l76_76302


namespace prime_square_mod_12_l76_76766

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end prime_square_mod_12_l76_76766


namespace water_hydrogen_oxygen_ratio_l76_76047

/-- In a mixture of water with a total mass of 171 grams, 
    where 19 grams are hydrogen, the ratio of hydrogen to oxygen by mass is 1:8. -/
theorem water_hydrogen_oxygen_ratio 
  (h_total_mass : ℝ) 
  (h_mass : ℝ) 
  (o_mass : ℝ) 
  (h_condition : h_total_mass = 171) 
  (h_hydrogen_mass : h_mass = 19) 
  (h_oxygen_mass : o_mass = h_total_mass - h_mass) :
  h_mass / o_mass = 1 / 8 := 
by
  sorry

end water_hydrogen_oxygen_ratio_l76_76047


namespace work_rate_proof_l76_76053

def combined_rate (a b c : ℚ) : ℚ := a + b + c

def inv (x : ℚ) : ℚ := 1 / x

theorem work_rate_proof (A B C : ℚ) (h₁ : A + B = 1/15) (h₂ : C = 1/10) :
  inv (combined_rate A B C) = 6 :=
by
  sorry

end work_rate_proof_l76_76053


namespace least_common_multiple_1_to_10_l76_76891

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76891


namespace quadratic_roots_p_eq_l76_76167

theorem quadratic_roots_p_eq (b c p q r s : ℝ)
  (h1 : r + s = -b)
  (h2 : r * s = c)
  (h3 : r^2 + s^2 = -p)
  (h4 : r^2 * s^2 = q):
  p = 2 * c - b^2 :=
by sorry

end quadratic_roots_p_eq_l76_76167


namespace carnations_count_l76_76654

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end carnations_count_l76_76654


namespace cubic_trinomial_degree_l76_76735

theorem cubic_trinomial_degree (n : ℕ) (P : ℕ → ℕ →  ℕ → Prop) : 
  (P n 5 4) → n = 3 := 
  sorry

end cubic_trinomial_degree_l76_76735


namespace probability_diff_color_correct_l76_76566

noncomputable def probability_diff_color (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  (red_balls * yellow_balls) / ((total_balls * (total_balls - 1)) / 2)

theorem probability_diff_color_correct :
  probability_diff_color 5 3 2 = 3 / 5 :=
by
  sorry

end probability_diff_color_correct_l76_76566


namespace least_common_multiple_of_first_ten_integers_l76_76933

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76933


namespace Chloe_pairs_shoes_l76_76227

theorem Chloe_pairs_shoes (cost_per_shoe total_cost : ℤ) (h_cost: cost_per_shoe = 37) (h_total: total_cost = 1036) :
  (total_cost / cost_per_shoe) / 2 = 14 :=
by
  -- proof goes here
  sorry

end Chloe_pairs_shoes_l76_76227


namespace no_integers_satisfy_l76_76665

theorem no_integers_satisfy (a b c d : ℤ) : ¬ (a^4 + b^4 + c^4 + 2016 = 10 * d) :=
sorry

end no_integers_satisfy_l76_76665


namespace min_y_value_l76_76050

noncomputable def y (x : ℝ) : ℝ := x^2 + 16 * x + 20

theorem min_y_value : ∀ (x : ℝ), x ≥ -3 → y x ≥ -19 :=
by
  intro x hx
  sorry

end min_y_value_l76_76050


namespace max_non_managers_l76_76495

theorem max_non_managers (x : ℕ) (h : (7:ℚ) / 32 < 9 / x) : x = 41 := sorry

end max_non_managers_l76_76495


namespace remainder_property_l76_76341

theorem remainder_property (a : ℤ) (h : ∃ k : ℤ, a = 45 * k + 36) :
  ∃ n : ℤ, a = 45 * n + 36 :=
by {
  sorry
}

end remainder_property_l76_76341


namespace questions_left_blank_l76_76034

-- Definitions based on the conditions
def total_questions : Nat := 60
def word_problems : Nat := 20
def add_subtract_problems : Nat := 25
def algebra_problems : Nat := 10
def geometry_problems : Nat := 5
def total_time : Nat := 90

def time_per_word_problem : Nat := 2
def time_per_add_subtract_problem : Float := 1.5
def time_per_algebra_problem : Nat := 3
def time_per_geometry_problem : Nat := 4

def word_problems_answered : Nat := 15
def add_subtract_problems_answered : Nat := 22
def algebra_problems_answered : Nat := 8
def geometry_problems_answered : Nat := 3

-- The final goal is to prove that Steve left 12 questions blank
theorem questions_left_blank :
  total_questions - (word_problems_answered + add_subtract_problems_answered + algebra_problems_answered + geometry_problems_answered) = 12 :=
by
  sorry

end questions_left_blank_l76_76034


namespace supplement_of_supplement_of_58_l76_76184

theorem supplement_of_supplement_of_58 (α : ℝ) (h : α = 58) : 180 - (180 - α) = 58 :=
by
  sorry

end supplement_of_supplement_of_58_l76_76184


namespace least_divisible_1_to_10_l76_76801

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76801


namespace least_common_multiple_first_ten_l76_76819

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76819


namespace polygon_sides_arithmetic_sequence_l76_76279

theorem polygon_sides_arithmetic_sequence 
  (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : 2 * (180 * (n - 2)) = n * (100 + 140)) :
  n = 6 :=
  sorry

end polygon_sides_arithmetic_sequence_l76_76279


namespace angle_F_measure_l76_76574

theorem angle_F_measure (D E F : ℝ) (h₁ : D = 80) (h₂ : E = 2 * F + 24) (h₃ : D + E + F = 180) : F = 76 / 3 :=
by
  sorry

end angle_F_measure_l76_76574


namespace empty_can_mass_l76_76966

-- Define the mass of the full can
def full_can_mass : ℕ := 35

-- Define the mass of the can with half the milk
def half_can_mass : ℕ := 18

-- The theorem stating the mass of the empty can
theorem empty_can_mass : full_can_mass - (2 * (full_can_mass - half_can_mass)) = 1 := by
  sorry

end empty_can_mass_l76_76966


namespace lcm_first_ten_positive_integers_l76_76882

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76882


namespace least_common_multiple_of_first_10_integers_l76_76947

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76947


namespace jane_vases_per_day_l76_76575

theorem jane_vases_per_day : 
  ∀ (total_vases : ℝ) (days : ℝ), 
  total_vases = 248 → days = 16 → 
  (total_vases / days) = 15.5 :=
by
  intros total_vases days h_total_vases h_days
  rw [h_total_vases, h_days]
  norm_num

end jane_vases_per_day_l76_76575


namespace inequality_solution_l76_76607

def solutionSetInequality (x : ℝ) : Prop :=
  (x > 1 ∨ x < -2)

theorem inequality_solution (x : ℝ) : 
  (x+2)/(x-1) > 0 ↔ solutionSetInequality x := 
  sorry

end inequality_solution_l76_76607


namespace least_common_multiple_1_to_10_l76_76888

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76888


namespace lcm_first_ten_integers_l76_76807

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76807


namespace a_gt_b_neither_sufficient_nor_necessary_l76_76580

theorem a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) := 
sorry

end a_gt_b_neither_sufficient_nor_necessary_l76_76580


namespace distance_between_city_centers_l76_76321

def distance_on_map : ℝ := 45  -- Distance on the map in cm
def scale_factor : ℝ := 20     -- Scale factor (1 cm : 20 km)

theorem distance_between_city_centers : distance_on_map * scale_factor = 900 := by
  sorry

end distance_between_city_centers_l76_76321


namespace perpendicular_planes_parallel_l76_76410

-- Define the lines m and n, and planes alpha and beta
def Line := Unit
def Plane := Unit

-- Define perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

-- The main theorem statement: If m ⊥ α and m ⊥ β, then α ∥ β
theorem perpendicular_planes_parallel (m : Line) (α β : Plane)
  (h₁ : perpendicular m α) (h₂ : perpendicular m β) : parallel α β :=
sorry

end perpendicular_planes_parallel_l76_76410


namespace find_triples_l76_76529

-- Defining the conditions
def divides (x y : ℕ) : Prop := ∃ k, y = k * x

-- The main Lean statement
theorem find_triples (a b c : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  divides a (b * c - 1) → divides b (a * c - 1) → divides c (a * b - 1) →
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 3) ∨
  (a = 3 ∧ b = 2 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 2) ∨
  (a = 5 ∧ b = 2 ∧ c = 3) ∨ (a = 5 ∧ b = 3 ∧ c = 2) :=
sorry

end find_triples_l76_76529


namespace lcm_first_ten_integers_l76_76814

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l76_76814


namespace fraction_of_odd_products_is_0_25_l76_76284

noncomputable def fraction_of_odd_products : ℝ :=
  let odd_products := 8 * 8
  let total_products := 16 * 16
  (odd_products / total_products : ℝ)

theorem fraction_of_odd_products_is_0_25 :
  fraction_of_odd_products = 0.25 :=
by sorry

end fraction_of_odd_products_is_0_25_l76_76284


namespace determine_subtracted_number_l76_76217

theorem determine_subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 7 * x - y = 130) : y = 150 :=
by sorry

end determine_subtracted_number_l76_76217


namespace benny_final_comic_books_l76_76516

-- Define the initial number of comic books
def initial_comic_books : ℕ := 22

-- Define the comic books sold (half of the initial)
def comic_books_sold : ℕ := initial_comic_books / 2

-- Define the comic books left after selling half
def comic_books_left_after_sale : ℕ := initial_comic_books - comic_books_sold

-- Define the number of comic books bought
def comic_books_bought : ℕ := 6

-- Define the final number of comic books
def final_comic_books : ℕ := comic_books_left_after_sale + comic_books_bought

-- Statement to prove that Benny has 17 comic books at the end
theorem benny_final_comic_books : final_comic_books = 17 := by
  sorry

end benny_final_comic_books_l76_76516


namespace tan_ratio_of_triangle_sides_l76_76003

theorem tan_ratio_of_triangle_sides (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a^2 + b^2 = 2023 * c^2)
  (h2 : α + β + γ = π)
  (h3 : c ≠ 0):
  ( (Real.tan γ) / (Real.tan α + Real.tan β) ) = (a * b) / (1011 * c^2) := 
sorry

end tan_ratio_of_triangle_sides_l76_76003


namespace intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l76_76273

def l1 (x y : ℝ) : Prop := x + y = 2
def l2 (x y : ℝ) : Prop := x - 3 * y = -10
def l3 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

def M : (ℝ × ℝ) := (-1, 3)

-- Part (Ⅰ): Prove that M is the intersection point of l1 and l2
theorem intersection_l1_l2 : l1 M.1 M.2 ∧ l2 M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅱ): Prove the equation of the line passing through M and parallel to l3 is 3x - 4y + 15 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y + 15 = 0

theorem line_parallel_to_l3 : parallel_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅲ): Prove the equation of the line passing through M and perpendicular to l3 is 4x + 3y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

theorem line_perpendicular_to_l3 : perpendicular_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

end intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l76_76273


namespace squirrel_nuts_collection_l76_76985

theorem squirrel_nuts_collection (n : ℕ) (e u : ℕ → ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n → e k = u k + k) ∧
  (∀ k, 1 ≤ k ∧ k ≤ n → u k = e (k + 1) + u k / 100) ∧
  e n = n →
  n = 99 → 
  (∃ S : ℕ, (∀ k, 1 ≤ k ∧ k ≤ n → e k = S)) ∧ 
  S = 9801 :=
sorry

end squirrel_nuts_collection_l76_76985


namespace gcd_2835_9150_l76_76484

theorem gcd_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end gcd_2835_9150_l76_76484


namespace distinct_solutions_diff_l76_76147

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l76_76147


namespace sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l76_76400

theorem sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_β : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π / 4 := sorry

end sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l76_76400


namespace sara_spent_on_bought_movie_l76_76450

-- Define the costs involved
def cost_ticket : ℝ := 10.62
def cost_rent : ℝ := 1.59
def total_spent : ℝ := 36.78

-- Define the quantity of tickets
def number_of_tickets : ℝ := 2

-- Define the total cost on tickets
def cost_on_tickets : ℝ := cost_ticket * number_of_tickets

-- Define the total cost on tickets and rented movie
def cost_on_tickets_and_rent : ℝ := cost_on_tickets + cost_rent

-- Define the total amount spent on buying the movie
def cost_bought_movie : ℝ := total_spent - cost_on_tickets_and_rent

-- The statement we need to prove
theorem sara_spent_on_bought_movie : cost_bought_movie = 13.95 :=
by
  sorry

end sara_spent_on_bought_movie_l76_76450


namespace find_radius_l76_76779

-- Defining the conditions as given in the math problem
def sectorArea (r : ℝ) (L : ℝ) : ℝ := 0.5 * r * L

theorem find_radius (h1 : sectorArea r 5.5 = 13.75) : r = 5 :=
by sorry

end find_radius_l76_76779


namespace simplify_fraction_l76_76773

variable (x : ℝ)

theorem simplify_fraction : (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l76_76773


namespace registered_voters_democrats_l76_76125

variables (D R : ℝ)

theorem registered_voters_democrats :
  (D + R = 100) →
  (0.80 * D + 0.30 * R = 65) →
  D = 70 :=
by
  intros h1 h2
  sorry

end registered_voters_democrats_l76_76125


namespace man_swims_speed_l76_76649

theorem man_swims_speed (v_m v_s : ℝ) (h_downstream : 28 = (v_m + v_s) * 2) (h_upstream : 12 = (v_m - v_s) * 2) : v_m = 10 := 
by sorry

end man_swims_speed_l76_76649


namespace least_positive_integer_divisible_by_first_ten_l76_76871

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76871


namespace k_inverse_k_inv_is_inverse_l76_76443

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x - 4
def k (x : ℝ) : ℝ := f (g x)

def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem k_inverse {x : ℝ} : k_inv (k x) = x :=
by
  sorry

theorem k_inv_is_inverse {x y : ℝ} : k_inv (y) = x ↔ y = k(x) :=
by
  sorry

end k_inverse_k_inv_is_inverse_l76_76443


namespace least_common_multiple_1_to_10_l76_76893

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76893


namespace volume_of_water_in_prism_l76_76070

-- Define the given dimensions and conditions
def length_x := 20 -- cm
def length_y := 30 -- cm
def length_z := 40 -- cm
def angle := 30 -- degrees
def total_volume := 24 -- liters

-- The wet fraction of the upper surface
def wet_fraction := 1 / 4

-- Correct answer to be proven
def volume_water := 18.8 -- liters

theorem volume_of_water_in_prism :
  -- Given the conditions
  (length_x = 20) ∧ (length_y = 30) ∧ (length_z = 40) ∧ (angle = 30) ∧ (wet_fraction = 1 / 4) ∧ (total_volume = 24) →
  -- Prove that the volume of water is as calculated
  volume_water = 18.8 :=
sorry

end volume_of_water_in_prism_l76_76070


namespace number_of_rows_seating_exactly_9_students_l76_76970

theorem number_of_rows_seating_exactly_9_students (x : ℕ) : 
  ∀ y z, x * 9 + y * 5 + z * 8 = 55 → x % 5 = 1 ∧ x % 8 = 7 → x = 3 :=
by sorry

end number_of_rows_seating_exactly_9_students_l76_76970


namespace roots_of_polynomial_l76_76533

   -- We need to define the polynomial and then state that the roots are exactly {0, 3, -5}
   def polynomial (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x)

   theorem roots_of_polynomial :
     {x : ℝ | polynomial x = 0} = {0, 3, -5} :=
   by
     sorry
   
end roots_of_polynomial_l76_76533


namespace total_sums_attempted_l76_76987

-- Define the necessary conditions
def num_sums_right : ℕ := 8
def num_sums_wrong : ℕ := 2 * num_sums_right

-- Define the theorem to prove
theorem total_sums_attempted : num_sums_right + num_sums_wrong = 24 := by
  sorry

end total_sums_attempted_l76_76987


namespace sqrt_inequality_l76_76455

open Real

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z) 
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  sqrt (x + y + z) ≥ sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) :=
sorry

end sqrt_inequality_l76_76455


namespace goats_count_l76_76231

variable (h d c t g : Nat)
variable (l : Nat)

theorem goats_count 
  (h_eq : h = 2)
  (d_eq : d = 5)
  (c_eq : c = 7)
  (t_eq : t = 3)
  (l_eq : l = 72)
  (legs_eq : 4 * h + 4 * d + 4 * c + 4 * t + 4 * g = l) : 
  g = 1 := by
  sorry

end goats_count_l76_76231


namespace diet_soda_bottles_l76_76645

def total_bottles : ℕ := 17
def regular_soda_bottles : ℕ := 9

theorem diet_soda_bottles : total_bottles - regular_soda_bottles = 8 := by
  sorry

end diet_soda_bottles_l76_76645


namespace inverse_of_composed_function_l76_76439

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l76_76439


namespace length_of_train_l76_76968

variable (L : ℕ)

def speed_tree (L : ℕ) : ℚ := L / 120

def speed_platform (L : ℕ) : ℚ := (L + 500) / 160

theorem length_of_train
    (h1 : speed_tree L = speed_platform L)
    : L = 1500 :=
sorry

end length_of_train_l76_76968


namespace lcm_first_ten_positive_integers_l76_76885

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76885


namespace xiao_liang_correct_l76_76568

theorem xiao_liang_correct :
  ∀ (x : ℕ), (0 ≤ x ∧ x ≤ 26 ∧ 30 - x ≤ 24 ∧ 26 - x ≤ 20) →
  let boys_A := x
  let girls_A := 30 - x
  let boys_B := 26 - x
  let girls_B := 24 - girls_A
  ∃ k : ℤ, boys_A - girls_B = 6 := 
by 
  sorry

end xiao_liang_correct_l76_76568


namespace dennis_teaching_years_l76_76618

noncomputable def years_taught (V A D E N : ℕ) := V + A + D + E + N
noncomputable def sum_of_ages := 375
noncomputable def teaching_years : Prop :=
  ∃ (A V D E N : ℕ),
    V + A + D + E + N = 225 ∧
    V = A + 9 ∧
    V = D - 15 ∧
    E = A - 3 ∧
    E = 2 * N ∧
    D = 101

theorem dennis_teaching_years : teaching_years :=
by
  sorry

end dennis_teaching_years_l76_76618


namespace find_ab_l76_76094

variable (a b m n : ℝ)

theorem find_ab (h1 : (a + b)^2 = m) (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 :=
by
  sorry

end find_ab_l76_76094


namespace determine_digit_z_l76_76379

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l76_76379


namespace chord_ratio_l76_76183

theorem chord_ratio {FQ HQ : ℝ} (h : EQ * FQ = GQ * HQ) (h_eq : EQ = 5) (h_gq : GQ = 12) : 
  FQ / HQ = 12 / 5 :=
by
  rw [h_eq, h_gq] at h
  sorry

end chord_ratio_l76_76183


namespace no_solution_exists_l76_76535

theorem no_solution_exists (x y : ℝ) : ¬ ((2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9)) :=
sorry

end no_solution_exists_l76_76535


namespace simplest_square_root_l76_76076

theorem simplest_square_root : 
  let options := [Real.sqrt 20, Real.sqrt 2, Real.sqrt (1 / 2), Real.sqrt 0.2] in
  ∃ x ∈ options, x = Real.sqrt 2 ∧ (∀ y ∈ options, y ≠ Real.sqrt 2 → ¬(Real.sqrt y).simpler_than (Real.sqrt 2)) :=
by
  let options := [Real.sqrt 20, Real.sqrt 2, Real.sqrt (1 / 2), Real.sqrt 0.2]
  have h_sqrt_2_in_options : Real.sqrt 2 ∈ options := by simp [options]
  use Real.sqrt 2
  constructor
  . exact h_sqrt_2_in_options
  . intro y hy_ne_sqrt_2 hy_options
    sorry

end simplest_square_root_l76_76076


namespace rabbit_catches_up_at_time_l76_76794

-- Definitions of the conditions
def rabbit_acceleration := 2 * 60 -- in miles per hour squared (120 mph²)
def cat_speed := 20 -- in miles per hour
def head_start_time := 15 / 60 -- 0.25 hours (15 minutes converted to hours)
def cat_start_distance := cat_speed * head_start_time -- 5 miles

-- Theorem statement
theorem rabbit_catches_up_at_time : ∃ t : ℝ, t = (15 / 60) + Real.sqrt (5 / (0.5 * rabbit_acceleration)) :=
by
  sorry

end rabbit_catches_up_at_time_l76_76794


namespace maximum_pq_qr_rs_sp_l76_76170

open Finset

def pq_qr_rs_sp (p q r s : ℕ) : ℕ := p * q + q * r + r * s + s * p

theorem maximum_pq_qr_rs_sp :
  ∃ (p q r s : ℕ), {p, q, r, s} = {2, 4, 6, 8} ∧ pq_qr_rs_sp p q r s = 100 :=
by
  use 8, 4, 2, 6
  simp [pq_qr_rs_sp]
  split
  { norm_num }
  { sorry }

end maximum_pq_qr_rs_sp_l76_76170


namespace least_common_multiple_1_to_10_l76_76889

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76889


namespace books_loaned_out_during_month_l76_76650

-- Define the initial conditions
def initial_books : ℕ := 75
def remaining_books : ℕ := 65
def loaned_out_percentage : ℝ := 0.80
def returned_books_ratio : ℝ := loaned_out_percentage
def not_returned_ratio : ℝ := 1 - returned_books_ratio
def difference : ℕ := initial_books - remaining_books

-- Define the main theorem
theorem books_loaned_out_during_month : ∃ (x : ℕ), not_returned_ratio * (x : ℝ) = (difference : ℝ) ∧ x = 50 :=
by
  existsi 50
  simp [not_returned_ratio, difference]
  sorry

end books_loaned_out_during_month_l76_76650


namespace max_value_of_a_l76_76228

theorem max_value_of_a :
  ∃ b : ℤ, ∃ (a : ℝ), 
    (a = 30285) ∧
    (a * b^2 / (a + 2 * b) = 2019) :=
by
  sorry

end max_value_of_a_l76_76228


namespace train_average_speed_l76_76509

-- Define the variables used in the conditions
variables (D V : ℝ)
-- Condition: Distance D in 50 minutes at average speed V kmph
-- 50 minutes to hours conversion
def condition1 : D = V * (50 / 60) := sorry
-- Condition: Distance D in 40 minutes at speed 60 kmph
-- 40 minutes to hours conversion
def condition2 : D = 60 * (40 / 60) := sorry

-- Claim: Current average speed V
theorem train_average_speed : V = 48 :=
by
  -- Using the conditions to prove the claim
  sorry

end train_average_speed_l76_76509


namespace least_distinct_values_l76_76647

/-- 
Given a list of 2018 positive integers with a unique mode occurring exactly 10 times,
prove that the least number of distinct values in the list is 225.
-/
theorem least_distinct_values {α : Type*} (l : list α) (hl_len : l.length = 2018) (hm : ∃ m, ∀ x ∈ l, count l x ≤ count l m ∧ count l m = 10 ∧ ∀ y ≠ x, count l y < 10) :
  ∃ n, n = 225 ∧ (∀ x ∈ (l.erase_dup), count l x ≤ 10) :=
sorry

end least_distinct_values_l76_76647


namespace arithmetic_sequence_seventh_term_l76_76178

theorem arithmetic_sequence_seventh_term (a d : ℝ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 14) 
  (h2 : a + 4 * d = 9) : 
  a + 6 * d = 13.4 := 
sorry

end arithmetic_sequence_seventh_term_l76_76178


namespace solve_for_F_l76_76553

theorem solve_for_F (C F : ℝ) (h1 : C = 5 / 9 * (F - 32)) (h2 : C = 40) : F = 104 :=
by
  sorry

end solve_for_F_l76_76553


namespace coffee_consumption_l76_76742

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end coffee_consumption_l76_76742


namespace value_at_7_6_l76_76299

noncomputable def f : ℝ → ℝ := sorry

lemma periodic_f (x : ℝ) : f (x + 4) = f x := sorry

lemma f_on_interval (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) : f x = x := sorry

theorem value_at_7_6 : f 7.6 = -0.4 :=
by
  have p := periodic_f 7.6
  have q := periodic_f 3.6
  have r := f_on_interval (-0.4)
  sorry

end value_at_7_6_l76_76299


namespace simplify_div_expression_l76_76771

theorem simplify_div_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2 * x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 :=
sorry

end simplify_div_expression_l76_76771


namespace longest_side_of_rectangular_solid_l76_76471

theorem longest_side_of_rectangular_solid 
  (x y z : ℝ) 
  (h1 : x * y = 20) 
  (h2 : y * z = 15) 
  (h3 : x * z = 12) 
  (h4 : x * y * z = 60) : 
  max (max x y) z = 10 := 
by sorry

end longest_side_of_rectangular_solid_l76_76471


namespace digit_ends_with_l76_76374

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l76_76374


namespace probability_four_heads_l76_76343

-- Definitions for use in the conditions
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def biased_coin (h : ℚ) (n k : ℕ) : ℚ :=
  binomial_coefficient n k * (h ^ k) * ((1 - h) ^ (n - k))

-- Condition: probability of getting heads exactly twice is equal to getting heads exactly three times.
def condition (h : ℚ) : Prop :=
  biased_coin h 5 2 = biased_coin h 5 3

-- Theorem to be proven: probability of getting heads exactly four times out of five is 5/32.
theorem probability_four_heads (h : ℚ) (cond : condition h) : biased_coin h 5 4 = 5 / 32 :=
by
  sorry

end probability_four_heads_l76_76343


namespace card_probability_l76_76355

theorem card_probability (n m : ℕ)
  (h1 : (finset.range 44).card = 44)
  (h2 : 2 * 4 = 8)
  (total_ways : 44.choose 2 = 946)
  (pairs_remaining : 10 * nat.choose 4 2 = 60)
  (prob : (2 + 2) + 60 = 62)
  (simplified : nat.succ 30 * nat.succ 472 = 473 * 31)
  (rat_prod_eq : ((62 : ℚ) / 946) = (31 : ℚ) / 473)
  : (31 + 473 = 504) :=
begin
  sorry
end

end card_probability_l76_76355


namespace number_of_boys_l76_76497

variable (x y : ℕ)

theorem number_of_boys (h1 : x + y = 900) (h2 : y = (x / 100) * 900) : x = 90 :=
by
  sorry

end number_of_boys_l76_76497


namespace least_common_multiple_first_ten_integers_l76_76944

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76944


namespace mul_scientific_notation_l76_76226

theorem mul_scientific_notation (a b : ℝ) (c d : ℝ) (h1 : a = 7 * 10⁻¹) (h2 : b = 8 * 10⁻¹) :
  (a * b = 0.56) :=
by
  sorry

end mul_scientific_notation_l76_76226


namespace lcm_first_ten_l76_76853

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76853


namespace females_in_group_l76_76687

theorem females_in_group (n F M : ℕ) (Index_F Index_M : ℝ) 
  (h1 : n = 25) 
  (h2 : Index_F = (n - F) / n)
  (h3 : Index_M = (n - M) / n) 
  (h4 : Index_F - Index_M = 0.36) :
  F = 8 := 
by
  sorry

end females_in_group_l76_76687


namespace determine_suit_cost_l76_76631

def cost_of_suit (J B V : ℕ) : Prop :=
  (J + B + V = 150)

theorem determine_suit_cost
  (J B V : ℕ)
  (h1 : J = B + V)
  (h2 : J + 2 * B = 175)
  (h3 : B + 2 * V = 100) :
  cost_of_suit J B V :=
by
  sorry

end determine_suit_cost_l76_76631


namespace least_common_multiple_first_ten_l76_76820

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l76_76820


namespace find_expression_l76_76748

def B : ℂ := 3 + 2 * Complex.I
def Q : ℂ := -5 * Complex.I
def R : ℂ := 1 + Complex.I
def T : ℂ := 3 - 4 * Complex.I

theorem find_expression : B * R + Q + T = 4 + Complex.I := by
  sorry

end find_expression_l76_76748


namespace range_of_x_l76_76708

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem to prove the condition
theorem range_of_x (x : ℝ) : f (1 - x) + f (2 * x) > 2 ↔ x > -1 :=
by {
  sorry -- Proof placeholder
}

end range_of_x_l76_76708


namespace least_common_multiple_1_to_10_l76_76898

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76898


namespace lcm_first_ten_l76_76851

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76851


namespace circle_center_and_radius_l76_76598

theorem circle_center_and_radius:
  ∀ x y : ℝ, 
  (x + 1) ^ 2 + (y - 3) ^ 2 = 36 
  → ∃ C : (ℝ × ℝ), C = (-1, 3) ∧ ∃ r : ℝ, r = 6 := sorry

end circle_center_and_radius_l76_76598


namespace sqrt_sum_ineq_l76_76398

open Real

theorem sqrt_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 1) :
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) + a + b + c > 3 := by
  sorry

end sqrt_sum_ineq_l76_76398


namespace inequality_solution_l76_76083

-- Condition definitions in lean
def numerator (x : ℝ) : ℝ := (x^5 - 13 * x^3 + 36 * x) * (x^4 - 17 * x^2 + 16)
def denominator (y : ℝ) : ℝ := (y^5 - 13 * y^3 + 36 * y) * (y^4 - 17 * y^2 + 16)

-- Given the critical conditions
def is_zero_or_pm1_pm2_pm3_pm4 (y : ℝ) : Prop := 
  y = 0 ∨ y = 1 ∨ y = -1 ∨ y = 2 ∨ y = -2 ∨ y = 3 ∨ y = -3 ∨ y = 4 ∨ y = -4

-- The theorem statement
theorem inequality_solution (x y : ℝ) : 
  (numerator x / denominator y) ≥ 0 ↔ ¬ (is_zero_or_pm1_pm2_pm3_pm4 y) :=
sorry -- proof to be filled in later

end inequality_solution_l76_76083


namespace ratio_bananas_dates_l76_76586

theorem ratio_bananas_dates (s c b d a : ℕ)
  (h1 : s = 780)
  (h2 : c = 60)
  (h3 : b = 3 * c)
  (h4 : a = 2 * d)
  (h5 : s = a + b + c + d) :
  b / d = 1 :=
by sorry

end ratio_bananas_dates_l76_76586


namespace negate_universal_prop_l76_76328

theorem negate_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0 :=
sorry

end negate_universal_prop_l76_76328


namespace tan_square_of_cos_double_angle_l76_76263

theorem tan_square_of_cos_double_angle (α : ℝ) (h : Real.cos (2 * α) = -1/9) : Real.tan (α)^2 = 5/4 :=
by
  sorry

end tan_square_of_cos_double_angle_l76_76263


namespace unique_coprime_solution_l76_76002

theorem unique_coprime_solution 
  (p : ℕ) (a b m r : ℕ) 
  (hp : Nat.Prime p) 
  (hp_odd : p % 2 = 1)
  (hp_nmid_ab : ¬ (p ∣ a * b))
  (hab_gt_m2 : a * b > m^2) :
  ∃! (x y : ℕ), Nat.Coprime x y ∧ (a * x^2 + b * y^2 = m * p ^ r) := 
sorry

end unique_coprime_solution_l76_76002


namespace pyramid_total_surface_area_l76_76365

theorem pyramid_total_surface_area :
  ∀ (s h : ℝ), s = 8 → h = 10 →
  6 * (1/2 * s * (Real.sqrt (h^2 - (s/2)^2))) = 48 * Real.sqrt 21 :=
by
  intros s h s_eq h_eq
  rw [s_eq, h_eq]
  sorry

end pyramid_total_surface_area_l76_76365


namespace part1_part2_l76_76105

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x - 1| + |2 * x - a|

theorem part1 (x : ℝ) : (f x 2 < 2) ↔ (1/4 < x ∧ x < 5/4) := by
  sorry
  
theorem part2 (a : ℝ) (hx : ∀ x : ℝ, f x a ≥ 3 * a + 2) :
  (-3/2 ≤ a ∧ a ≤ -1/4) := by
  sorry

end part1_part2_l76_76105


namespace digit_makes_57A2_divisible_by_9_l76_76482

theorem digit_makes_57A2_divisible_by_9 (A : ℕ) (h : 0 ≤ A ∧ A ≤ 9) : 
  (5 + 7 + A + 2) % 9 = 0 ↔ A = 4 :=
by
  sorry

end digit_makes_57A2_divisible_by_9_l76_76482


namespace abs_eq_neg_of_le_zero_l76_76558

theorem abs_eq_neg_of_le_zero (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_of_le_zero_l76_76558


namespace binomial_expansion_coefficients_equal_l76_76464

theorem binomial_expansion_coefficients_equal (n : ℕ) (h : n ≥ 6)
  (h_eq : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : n = 7 := by
  sorry

end binomial_expansion_coefficients_equal_l76_76464


namespace find_a_plus_2b_l76_76009

open Real

theorem find_a_plus_2b 
  (a b : ℝ) 
  (ha : 0 < a ∧ a < π / 2) 
  (hb : 0 < b ∧ b < π / 2) 
  (h1 : 4 * (sin a)^2 + 3 * (sin b)^2 = 1) 
  (h2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0) :
  a + 2 * b = π / 2 :=
sorry

end find_a_plus_2b_l76_76009


namespace problem_solution_l76_76096

theorem problem_solution (x : ℝ) (h : ∃ (A B : Set ℝ), A = {0, 1, 2, 4, 5} ∧ B = {x-2, x, x+2} ∧ A ∩ B = {0, 2}) : x = 0 :=
sorry

end problem_solution_l76_76096


namespace problem1_problem2_l76_76269

section Problems

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x + 1

-- Problem 1: Tangent line problem for a = 1
def tangent_line_eqn (x : ℝ) : Prop :=
  let a := 1
  let f := f a
  (∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b)

-- Problem 2: Minimum value problem
def min_value_condition (a : ℝ) : Prop :=
  f a (1 / 4) = (11 / 12)

theorem problem1 : tangent_line_eqn 0 :=
  sorry

theorem problem2 : min_value_condition (1 / 4) :=
  sorry

end Problems

end problem1_problem2_l76_76269


namespace bowling_ball_weight_l76_76245

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end bowling_ball_weight_l76_76245


namespace find_d_value_l76_76123

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end find_d_value_l76_76123


namespace inequality_proof_l76_76011

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
by
  sorry

end inequality_proof_l76_76011


namespace smallest_positive_perfect_cube_l76_76581

theorem smallest_positive_perfect_cube (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ m : ℕ, m = (a * b * c^2)^3 ∧ (a^2 * b^3 * c^5 ∣ m)
:=
sorry

end smallest_positive_perfect_cube_l76_76581


namespace ratio_of_fifth_terms_l76_76106

theorem ratio_of_fifth_terms (a_n b_n : ℕ → ℕ) (S T : ℕ → ℕ)
  (hs : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (ht : ∀ n, T n = n * (b_n 1 + b_n n) / 2)
  (h : ∀ n, S n / T n = (7 * n + 2) / (n + 3)) :
  a_n 5 / b_n 5 = 65 / 12 :=
by
  sorry

end ratio_of_fifth_terms_l76_76106


namespace least_common_multiple_1_to_10_l76_76886

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l76_76886


namespace intersection_points_l76_76283

theorem intersection_points (a : ℝ) (h : 2 < a) :
  (∃ n : ℕ, (n = 1 ∨ n = 2) ∧ (∃ x1 x2 : ℝ, y = (a-3)*x^2 - x - 1/4 ∧ x1 ≠ x2)) :=
sorry

end intersection_points_l76_76283


namespace vector_opposite_direction_and_magnitude_l76_76564

theorem vector_opposite_direction_and_magnitude
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (-1, 2)) 
  (h2 : ∃ k : ℝ, k < 0 ∧ b = k • a) 
  (hb : ‖b‖ = Real.sqrt 5) :
  b = (1, -2) :=
sorry

end vector_opposite_direction_and_magnitude_l76_76564


namespace ellipse_standard_equation_l76_76699

theorem ellipse_standard_equation
  (F : ℝ × ℝ)
  (e : ℝ)
  (eq1 : F = (0, 1))
  (eq2 : e = 1 / 2) :
  ∃ (a b : ℝ), a = 2 ∧ b ^ 2 = 3 ∧ (∀ x y : ℝ, (y ^ 2 / 4) + (x ^ 2 / 3) = 1) :=
by
  sorry

end ellipse_standard_equation_l76_76699


namespace least_common_multiple_first_ten_integers_l76_76941

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l76_76941


namespace PaulineDressCost_l76_76764

-- Lets define the variables for each dress cost
variable (P Jean Ida Patty : ℝ)

-- Condition statements
def condition1 : Prop := Patty = Ida + 10
def condition2 : Prop := Ida = Jean + 30
def condition3 : Prop := Jean = P - 10
def condition4 : Prop := P + Jean + Ida + Patty = 160

-- The proof problem statement
theorem PaulineDressCost : 
  condition1 Patty Ida →
  condition2 Ida Jean →
  condition3 Jean P →
  condition4 P Jean Ida Patty →
  P = 30 := by
  sorry

end PaulineDressCost_l76_76764


namespace K_set_I_K_set_III_K_set_IV_K_set_V_l76_76670

-- Definitions for the problem conditions
def K (x y z : ℤ) : ℤ :=
  (x + 2 * y + 3 * z) * (2 * x - y - z) * (y + 2 * z + 3 * x) +
  (y + 2 * z + 3 * x) * (2 * y - z - x) * (z + 2 * x + 3 * y) +
  (z + 2 * x + 3 * y) * (2 * z - x - y) * (x + 2 * y + 3 * z)

-- The equivalent form as a product of terms
def K_equiv (x y z : ℤ) : ℤ :=
  (y + z - 2 * x) * (z + x - 2 * y) * (x + y - 2 * z)

-- Proof statements for each set of numbers
theorem K_set_I : K 1 4 9 = K_equiv 1 4 9 := by
  sorry

theorem K_set_III : K 4 9 1 = K_equiv 4 9 1 := by
  sorry

theorem K_set_IV : K 1 8 11 = K_equiv 1 8 11 := by
  sorry

theorem K_set_V : K 5 8 (-2) = K_equiv 5 8 (-2) := by
  sorry

end K_set_I_K_set_III_K_set_IV_K_set_V_l76_76670


namespace least_common_multiple_of_first_ten_positive_integers_l76_76858

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l76_76858


namespace least_common_multiple_of_first_ten_integers_l76_76927

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76927


namespace min_max_pieces_three_planes_l76_76673

theorem min_max_pieces_three_planes : 
  ∃ (min max : ℕ), (min = 4) ∧ (max = 8) := by
  sorry

end min_max_pieces_three_planes_l76_76673


namespace find_number_l76_76569

theorem find_number (N : ℝ) 
  (h1 : (5 / 6) * N = (5 / 16) * N + 200) : 
  N = 384 :=
sorry

end find_number_l76_76569


namespace certain_number_is_five_hundred_l76_76420

theorem certain_number_is_five_hundred (x : ℝ) (h : 0.60 * x = 0.50 * 600) : x = 500 := 
by sorry

end certain_number_is_five_hundred_l76_76420


namespace lcm_first_ten_positive_integers_l76_76884

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76884


namespace least_common_multiple_of_first_ten_l76_76923

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76923


namespace find_m_value_l76_76733

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end find_m_value_l76_76733


namespace digit_ends_with_l76_76375

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l76_76375


namespace nelly_earns_per_night_l76_76159

/-- 
  Nelly wants to buy pizza for herself and her 14 friends. Each pizza costs $12 and can feed 3 
  people. Nelly has to babysit for 15 nights to afford the pizza. We need to prove that Nelly earns 
  $4 per night babysitting.
--/
theorem nelly_earns_per_night 
  (total_people : ℕ) (people_per_pizza : ℕ) 
  (cost_per_pizza : ℕ) (total_nights : ℕ) (total_cost : ℕ) 
  (total_pizzas : ℕ) (cost_per_night : ℕ)
  (h1 : total_people = 15)
  (h2 : people_per_pizza = 3)
  (h3 : cost_per_pizza = 12)
  (h4 : total_nights = 15)
  (h5 : total_pizzas = total_people / people_per_pizza)
  (h6 : total_cost = total_pizzas * cost_per_pizza)
  (h7 : cost_per_night = total_cost / total_nights) :
  cost_per_night = 4 := sorry

end nelly_earns_per_night_l76_76159


namespace initial_salary_increase_l76_76790

theorem initial_salary_increase :
  ∃ x : ℝ, 5000 * (1 + x/100) * 0.95 = 5225 := by
  sorry

end initial_salary_increase_l76_76790


namespace enlarged_banner_height_l76_76770

-- Definitions and theorem statement
theorem enlarged_banner_height 
  (original_width : ℝ) 
  (original_height : ℝ) 
  (new_width : ℝ) 
  (scaling_factor : ℝ := new_width / original_width ) 
  (new_height : ℝ := original_height * scaling_factor) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 15): 
  new_height = 10 := 
by 
  -- The proof would go here
  sorry

end enlarged_banner_height_l76_76770


namespace tammy_speed_second_day_l76_76056

theorem tammy_speed_second_day :
  ∀ (v1 t1 v2 t2 : ℝ), 
    t1 + t2 = 14 →
    t2 = t1 - 2 →
    v2 = v1 + 0.5 →
    v1 * t1 + v2 * t2 = 52 →
    v2 = 4 :=
by
  intros v1 t1 v2 t2 h1 h2 h3 h4
  sorry

end tammy_speed_second_day_l76_76056


namespace polynomial_in_y_l76_76519

theorem polynomial_in_y {x y : ℝ} (h₁ : x^3 - 6 * x^2 + 11 * x - 6 = 0) (h₂ : y = x + 1/x) :
  x^2 * (y^2 + y - 6) = 0 :=
sorry

end polynomial_in_y_l76_76519


namespace least_positive_integer_divisible_by_first_ten_l76_76869

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76869


namespace find_x_l76_76301

def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem find_x 
  (x y : ℤ) 
  (h_star1 : star 5 4 2 2 = (7, 2)) 
  (h_eq : star x y 3 3 = (7, 2)) : 
  x = 4 := 
sorry

end find_x_l76_76301


namespace least_common_multiple_of_first_ten_l76_76917

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l76_76917


namespace hanoi_tower_l76_76349

noncomputable def move_all_disks (n : ℕ) : Prop := 
  ∀ (A B C : Type), 
  (∃ (move : A → B), move = sorry) ∧ -- Only one disk can be moved
  (∃ (can_place : A → A → Prop), can_place = sorry) -- A disk cannot be placed on top of a smaller disk 
  → ∃ (u_n : ℕ), u_n = 2^n - 1 -- Formula for minimum number of steps

theorem hanoi_tower : ∀ n : ℕ, move_all_disks n :=
by sorry

end hanoi_tower_l76_76349


namespace smallest_possible_b_l76_76777

theorem smallest_possible_b (a b : ℕ) (h1 : a - b = 6) 
  (h2 : Nat.gcd ((a ^ 3 + b ^ 3) / (a + b)) (a * b) = 9) : b = 3 :=
by
  sorry

end smallest_possible_b_l76_76777


namespace find_investment_sum_l76_76193

variable (P : ℝ)

def simple_interest (rate time : ℝ) (principal : ℝ) : ℝ :=
  principal * rate * time

theorem find_investment_sum (h : simple_interest 0.18 2 P - simple_interest 0.12 2 P = 240) :
  P = 2000 :=
by
  sorry

end find_investment_sum_l76_76193


namespace prob_of_entirely_black_l76_76199

noncomputable def prob_all_black_grid : ℚ :=
  if (is_center_black : Prop) ∧ 
     (are_edge_squares_black : Prop) ∧ 
     (are_corner_squares_black : Prop)
  then (1/2 : ℚ) * (7/16 : ℚ) * (7/16 : ℚ)
  else 0

theorem prob_of_entirely_black (h : prob_all_black_grid = 49 / 512) : true :=
by { sorry }

end prob_of_entirely_black_l76_76199


namespace find_subtracted_number_l76_76116

-- Given conditions
def t : ℕ := 50
def k : ℕ := 122
def eq_condition (n : ℤ) : Prop := t = (5 / 9 : ℚ) * (k - n)

-- The proof problem proving the number subtracted from k is 32
theorem find_subtracted_number : eq_condition 32 :=
by
  -- implementation here will demonstrate that t = 50 implies the number is 32
  sorry

end find_subtracted_number_l76_76116


namespace least_common_multiple_of_first_10_integers_l76_76948

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l76_76948


namespace gcd_three_numbers_l76_76389

def a : ℕ := 8650
def b : ℕ := 11570
def c : ℕ := 28980

theorem gcd_three_numbers : Nat.gcd (Nat.gcd a b) c = 10 :=
by 
  sorry

end gcd_three_numbers_l76_76389


namespace quadratic_two_distinct_real_roots_l76_76174

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ - k^2 = 0) ∧ (x₂^2 - x₂ - k^2 = 0) :=
by
  -- The proof is omitted as requested.
  sorry

end quadratic_two_distinct_real_roots_l76_76174


namespace determine_digit_z_l76_76377

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l76_76377


namespace seashells_total_correct_l76_76713

-- Define the initial counts for Henry, John, and Adam.
def initial_seashells_Henry : ℕ := 11
def initial_seashells_John : ℕ := 24
def initial_seashells_Adam : ℕ := 17

-- Define the total initial seashells collected by all.
def total_initial_seashells : ℕ := 83

-- Calculate Leo's initial seashells.
def initial_seashells_Leo : ℕ := total_initial_seashells - (initial_seashells_Henry + initial_seashells_John + initial_seashells_Adam)

-- Define the changes occurred when they returned home.
def extra_seashells_Henry : ℕ := 3
def given_away_seashells_John : ℕ := 5
def percentage_given_away_Leo : ℕ := 40
def extra_seashells_Leo : ℕ := 5

-- Define the final number of seashells each person has.
def final_seashells_Henry : ℕ := initial_seashells_Henry + extra_seashells_Henry
def final_seashells_John : ℕ := initial_seashells_John - given_away_seashells_John
def given_away_seashells_Leo : ℕ := (initial_seashells_Leo * percentage_given_away_Leo) / 100
def final_seashells_Leo : ℕ := initial_seashells_Leo - given_away_seashells_Leo + extra_seashells_Leo
def final_seashells_Adam : ℕ := initial_seashells_Adam

-- Define the total number of seashells they have now.
def total_final_seashells : ℕ := final_seashells_Henry + final_seashells_John + final_seashells_Leo + final_seashells_Adam

-- Proposition that asserts the total number of seashells is 74.
theorem seashells_total_correct :
  total_final_seashells = 74 :=
sorry

end seashells_total_correct_l76_76713


namespace expression_evaluation_l76_76517

def evaluate_expression : ℝ := (-1) ^ 51 + 3 ^ (2^3 + 5^2 - 7^2)

theorem expression_evaluation :
  evaluate_expression = -1 + (1 / 43046721) :=
by
  sorry

end expression_evaluation_l76_76517


namespace least_divisible_1_to_10_l76_76803

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l76_76803


namespace mary_received_more_l76_76017

theorem mary_received_more (investment_Mary investment_Harry profit : ℤ)
  (one_third_profit divided_equally remaining_profit : ℤ)
  (total_Mary total_Harry difference : ℤ)
  (investment_ratio_Mary investment_ratio_Harry : ℚ) :
  investment_Mary = 700 →
  investment_Harry = 300 →
  profit = 3000 →
  one_third_profit = profit / 3 →
  divided_equally = one_third_profit / 2 →
  remaining_profit = profit - one_third_profit →
  investment_ratio_Mary = 7/10 →
  investment_ratio_Harry = 3/10 →
  total_Mary = divided_equally + investment_ratio_Mary * remaining_profit →
  total_Harry = divided_equally + investment_ratio_Harry * remaining_profit →
  difference = total_Mary - total_Harry →
  difference = 800 := by
  sorry

end mary_received_more_l76_76017


namespace at_least_1991_red_points_l76_76319

theorem at_least_1991_red_points (P : Fin 997 → ℝ × ℝ) :
  ∃ (R : Finset (ℝ × ℝ)), 1991 ≤ R.card ∧ (∀ (i j : Fin 997), i ≠ j → ((P i + P j) / 2) ∈ R) :=
sorry

end at_least_1991_red_points_l76_76319


namespace find_actual_marks_l76_76207

theorem find_actual_marks (wrong_marks : ℕ) (avg_increase : ℕ) (num_pupils : ℕ) (h_wrong_marks: wrong_marks = 73) (h_avg_increase : avg_increase = 1/2) (h_num_pupils : num_pupils = 16) : 
  ∃ (actual_marks : ℕ), actual_marks = 65 :=
by
  have total_increase := num_pupils * avg_increase
  have eqn := wrong_marks - total_increase
  use eqn
  sorry

end find_actual_marks_l76_76207


namespace maximize_profit_l76_76738

noncomputable def production (m k : ℝ) : ℝ := 3 - k / (m + 1)

def profit (m : ℝ) : ℝ := 28 - m - 16 / (m + 1)

lemma k_value (m : ℝ) (h₁ : m = 0) (h₂ : production m k = 1) : k = 2 :=
by {
  simp [production, h₁] at h₂,
  calc
  1 = 3 - k : by simpa using h₂
  ... = k = 2 : by linarith 
}

lemma profit_maximizer : profit 3 = 21 :=
by {
  calc
  profit 3 = 28 - 3 - 16 / (3 + 1) : by simp [profit]
  ... = 21 : by norm_num
}

theorem maximize_profit : ∃ m, m = 3 ∧ profit m = 21 := 
⟨3, rfl, profit_maximizer⟩

#check k_value
#check profit
#check maximize_profit

end maximize_profit_l76_76738


namespace replaced_person_age_is_40_l76_76166

def average_age_decrease_replacement (T age_of_replaced: ℕ) : Prop :=
  let original_average := T / 10
  let new_total_age := T - age_of_replaced + 10
  let new_average := new_total_age / 10
  original_average - 3 = new_average

theorem replaced_person_age_is_40 (T : ℕ) (h : average_age_decrease_replacement T 40) : Prop :=
  ∀ age_of_replaced, age_of_replaced = 40 → average_age_decrease_replacement T age_of_replaced

-- To actually formalize the proof, you can use the following structure:
-- proof by calculation omitted
lemma replaced_person_age_is_40_proof (T : ℕ) (h : average_age_decrease_replacement T 40) : 
  replaced_person_age_is_40 T h :=
by
  sorry

end replaced_person_age_is_40_l76_76166


namespace songs_listened_l76_76961

theorem songs_listened (x y : ℕ) 
  (h1 : y = 9) 
  (h2 : y = 2 * (Nat.sqrt x) - 5) 
  : y + x = 58 := 
  sorry

end songs_listened_l76_76961


namespace least_common_multiple_of_first_ten_integers_l76_76932

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l76_76932


namespace fourth_throw_probability_l76_76190

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end fourth_throw_probability_l76_76190


namespace solve_equation_l76_76460

theorem solve_equation (x : ℝ) (h : (x - 3) / 2 - (2 * x) / 3 = 1) : x = -15 := 
by 
  sorry

end solve_equation_l76_76460


namespace three_digit_number_is_473_l76_76356

theorem three_digit_number_is_473 (x y z : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) (h5 : 0 ≤ z) (h6 : z ≤ 9)
  (h7 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 99)
  (h8 : x + y + z = 14)
  (h9 : x + z = y) : 100 * x + 10 * y + z = 473 :=
by
  sorry

end three_digit_number_is_473_l76_76356


namespace inverse_of_k_l76_76442

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l76_76442


namespace lcm_first_ten_positive_integers_l76_76881

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l76_76881


namespace least_positive_integer_divisible_by_first_ten_l76_76873

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l76_76873


namespace next_podcast_duration_l76_76309

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end next_podcast_duration_l76_76309


namespace integer_triples_condition_l76_76247

theorem integer_triples_condition (p q r : ℤ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) 
  (h4 : ((p - 1) * (q - 1) * (r - 1)) ∣ (p * q * r - 1)) : (p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15) :=
sorry

end integer_triples_condition_l76_76247


namespace rate_per_sqm_is_correct_l76_76783

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end rate_per_sqm_is_correct_l76_76783


namespace A_speed_ratio_B_speed_l76_76359

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end A_speed_ratio_B_speed_l76_76359


namespace sum_of_perimeters_correct_l76_76791

noncomputable def sum_of_perimeters (s w : ℝ) : ℝ :=
  let l := 2 * w
  let square_area := s^2
  let rectangle_area := l * w
  let sq_perimeter := 4 * s
  let rect_perimeter := 2 * l + 2 * w
  sq_perimeter + rect_perimeter

theorem sum_of_perimeters_correct (s w : ℝ) (h1 : s^2 + 2 * w^2 = 130) (h2 : s^2 - 2 * w^2 = 50) :
  sum_of_perimeters s w = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end sum_of_perimeters_correct_l76_76791


namespace least_common_multiple_1_to_10_l76_76900

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l76_76900


namespace sqrt_prod_plus_one_equals_341_l76_76998

noncomputable def sqrt_prod_plus_one : ℕ :=
  Nat.sqrt ((20 * 19 * 18 * 17) + 1)

theorem sqrt_prod_plus_one_equals_341 :
  sqrt_prod_plus_one = 341 := 
by
  sorry

end sqrt_prod_plus_one_equals_341_l76_76998


namespace dealer_is_cheating_l76_76071

variable (w a : ℝ)
noncomputable def measured_weight (w : ℝ) (a : ℝ) : ℝ :=
  (a * w + w / a) / 2

theorem dealer_is_cheating (h : a > 0) : measured_weight w a ≥ w :=
by
  sorry

end dealer_is_cheating_l76_76071


namespace ratio_initial_to_doubled_l76_76981

theorem ratio_initial_to_doubled (x : ℝ) (h : 3 * (2 * x + 8) = 84) : x / (2 * x) = 1 / 2 :=
by
  have h1 : 2 * x + 8 = 28 := by
    sorry
  have h2 : x = 10 := by
    sorry
  rw [h2]
  norm_num

end ratio_initial_to_doubled_l76_76981


namespace unique_solution_exists_l76_76084

theorem unique_solution_exists (n m k : ℕ) :
  n = m^3 ∧ n = 1000 * m + k ∧ 0 ≤ k ∧ k < 1000 ∧ (1000 * m ≤ m^3 ∧ m^3 < 1000 * (m + 1)) → n = 32768 :=
by
  sorry

end unique_solution_exists_l76_76084


namespace regression_total_sum_of_squares_l76_76463

variables (y : Fin 10 → ℝ) (y_hat : Fin 10 → ℝ)
variables (residual_sum_of_squares : ℝ) 

-- Given conditions
def R_squared := 0.95
def RSS := 120.53

-- The total sum of squares is what we need to prove
noncomputable def total_sum_of_squares := 2410.6

-- Statement to prove
theorem regression_total_sum_of_squares :
  1 - RSS / total_sum_of_squares = R_squared := by
sorry

end regression_total_sum_of_squares_l76_76463


namespace white_paint_amount_l76_76448

theorem white_paint_amount (total_blue_paint additional_blue_paint total_mix blue_parts red_parts white_parts green_parts : ℕ) 
    (h_ratio: blue_parts = 7 ∧ red_parts = 2 ∧ white_parts = 1 ∧ green_parts = 1)
    (total_blue_paint_eq: total_blue_paint = 140)
    (max_total_mix: additional_blue_paint ≤ 220 - total_blue_paint) 
    : (white_parts * (total_blue_paint / blue_parts)) = 20 := 
by 
  sorry

end white_paint_amount_l76_76448


namespace man_l76_76069

/-- A man can row downstream at the rate of 45 kmph.
    A man can row upstream at the rate of 23 kmph.
    The rate of current is 11 kmph.
    The man's rate in still water is 34 kmph. -/
theorem man's_rate_in_still_water
  (v c : ℕ)
  (h1 : v + c = 45)
  (h2 : v - c = 23)
  (h3 : c = 11) : v = 34 := by
  sorry

end man_l76_76069


namespace sequence_term_expression_l76_76705

theorem sequence_term_expression (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = 3^n + 1) :
  (a 1 = 4) ∧ (∀ n, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by
  sorry

end sequence_term_expression_l76_76705


namespace problem_statement_l76_76143

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l76_76143


namespace digit_ends_with_l76_76376

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l76_76376


namespace students_in_game_divisors_of_119_l76_76201

theorem students_in_game_divisors_of_119 (n : ℕ) (h1 : ∃ (k : ℕ), k * n = 119) :
  n = 7 ∨ n = 17 :=
sorry

end students_in_game_divisors_of_119_l76_76201


namespace guiding_normal_vector_l76_76530

noncomputable def ellipsoid (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 - 6

def point_M0 : ℝ × ℝ × ℝ := (1, -1, 1)

def normal_vector (x y z : ℝ) : ℝ × ℝ × ℝ := (
  2 * x,
  4 * y,
  6 * z
)

theorem guiding_normal_vector : normal_vector 1 (-1) 1 = (2, -4, 6) :=
by
  sorry

end guiding_normal_vector_l76_76530


namespace terry_mary_same_color_combination_l76_76068

theorem terry_mary_same_color_combination :
  let red_candies := 10
  let blue_candies := 10
  let total_candies := red_candies + blue_candies
  let terry_prob := (red_candies * (red_candies - 1) + blue_candies * (blue_candies - 1)) / (total_candies * (total_candies - 1))
  let mary_given_terry_red_prob := ((blue_candies * (blue_candies - 1)) / ((total_candies - 2) * (total_candies - 3))) + 
                                   ((red_candies - 2) * (red_candies - 3) / ((total_candies - 2) * (total_candies - 3)))
  let mary_given_terry_blue_prob := ((blue_candies - 2) * (blue_candies - 3) / ((total_candies - 2) * (total_candies - 3))) + 
                                    ((red_candies) * (red_candies - 1) / ((total_candies - 2) * (total_candies - 3)))
  let combined_prob := 2 * ((terry_prob * mary_given_terry_red_prob) + (terry_prob * mary_given_terry_blue_prob))
  combined_prob = 73 / 323 :=
sorry

end terry_mary_same_color_combination_l76_76068


namespace problem_statement_l76_76765

variable (a b c d : ℝ)

-- Definitions for the conditions
def condition1 := a + b + c + d = 100
def condition2 := (a / (b + c + d)) + (b / (a + c + d)) + (c / (a + b + d)) + (d / (a + b + c)) = 95

-- The theorem which needs to be proved
theorem problem_statement (h1 : condition1 a b c d) (h2 : condition2 a b c d) :
  (1 / (b + c + d)) + (1 / (a + c + d)) + (1 / (a + b + d)) + (1 / (a + b + c)) = 99 / 100 := by
  sorry

end problem_statement_l76_76765


namespace inequality_proof_l76_76776

theorem inequality_proof (a b c d : ℝ) (h : a > 0) (h : b > 0) (h : c > 0) (h : d > 0)
  (h₁ : (a * b) / (c * d) = (a + b) / (c + d)) : (a + b) * (c + d) ≥ (a + c) * (b + d) :=
sorry

end inequality_proof_l76_76776


namespace mod_product_l76_76994

theorem mod_product : (198 * 955) % 50 = 40 :=
by sorry

end mod_product_l76_76994


namespace total_logs_in_stack_l76_76215

-- Definitions for the conditions
def a : ℕ := 15
def l : ℕ := 4
def d : ℤ := -1
def n : ℕ := (l - a + d.abs) / d.abs + 1

-- Statement with the proof problem
theorem total_logs_in_stack :
  n = 12 ∧ ∑ k in finset.range n, (a - k) = 114 := by
  sorry

end total_logs_in_stack_l76_76215


namespace problem_statement_l76_76142

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l76_76142


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76844

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76844


namespace system_of_equations_solution_l76_76633

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end system_of_equations_solution_l76_76633


namespace least_number_to_add_l76_76340

theorem least_number_to_add (x : ℕ) : (1056 + x) % 28 = 0 ↔ x = 4 :=
by sorry

end least_number_to_add_l76_76340


namespace find_number_l76_76507

theorem find_number :
  ∃ (N : ℝ), (5 / 4) * N = (4 / 5) * N + 45 ∧ N = 100 :=
by
  sorry

end find_number_l76_76507


namespace range_of_k_has_extreme_values_on_interval_l76_76600

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - x^2 + 3 * x

theorem range_of_k_has_extreme_values_on_interval (k : ℝ) (h : k ≠ 0) :
  -9/8 < k ∧ k < 0 :=
sorry

end range_of_k_has_extreme_values_on_interval_l76_76600


namespace no_valid_pairs_l76_76275

theorem no_valid_pairs : ∀ (m n : ℕ), m ≥ n → m^2 - n^2 = 150 → false :=
by sorry

end no_valid_pairs_l76_76275


namespace coins_problem_l76_76757

theorem coins_problem (x y : ℕ) (h1 : x + y = 20) (h2 : x + 5 * y = 80) : x = 5 :=
by
  sorry

end coins_problem_l76_76757


namespace circle_through_points_and_intercepts_l76_76680

noncomputable def circle_eq (x y D E F : ℝ) : ℝ := x^2 + y^2 + D * x + E * y + F

theorem circle_through_points_and_intercepts :
  ∃ (D E F : ℝ), 
    circle_eq 4 2 D E F = 0 ∧
    circle_eq (-1) 3 D E F = 0 ∧ 
    D + E = -2 ∧
    circle_eq x y (-2) 0 (-12) = 0 :=
by
  unfold circle_eq
  sorry

end circle_through_points_and_intercepts_l76_76680


namespace time_b_used_l76_76491

noncomputable def time_b_used_for_proof : ℚ :=
  let C : ℚ := 1
  let C_a : ℚ := 1 / 4 * C
  let t_a : ℚ := 15
  let p_a : ℚ := 1 / 3
  let p_b : ℚ := 2 / 3
  let ratio : ℚ := (C_a * t_a) / ((C - C_a) * (t_a * p_a / p_b))
  t_a * p_a / p_b

theorem time_b_used : time_b_used_for_proof = 10 / 3 := by
  sorry

end time_b_used_l76_76491


namespace least_positive_integer_divisible_by_first_ten_integers_l76_76837

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l76_76837


namespace range_of_a_l76_76119

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l76_76119


namespace least_months_for_tripling_debt_l76_76434

theorem least_months_for_tripling_debt (P : ℝ) (r : ℝ) (t : ℕ) : P = 1500 → r = 0.06 → (3 * P < P * (1 + r) ^ t) → t ≥ 20 :=
by
  intros hP hr hI
  rw [hP, hr] at hI
  norm_num at hI
  sorry

end least_months_for_tripling_debt_l76_76434


namespace largest_root_is_sqrt6_l76_76036

theorem largest_root_is_sqrt6 (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -6) 
  (h3 : p * q * r = -18) : 
  max p (max q r) = Real.sqrt 6 := 
sorry

end largest_root_is_sqrt6_l76_76036


namespace lcm_first_ten_l76_76846

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l76_76846


namespace skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l76_76287

-- Define the variables for the number of vehicles each type of worker can install
variables {x y : ℝ}

-- Define the conditions for system of equations
def skilled_and_new_workers_system1 (x y : ℝ) : Prop :=
  2 * x + y = 10

def skilled_and_new_workers_system2 (x y : ℝ) : Prop :=
  x + 3 * y = 10

-- Prove the number of vehicles each skilled worker and new worker can install
theorem skilled_new_worker_installation (x y : ℝ) (h1 : skilled_and_new_workers_system1 x y) (h2 : skilled_and_new_workers_system2 x y) : x = 4 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

-- Define the average cost equation for electric and gasoline vehicles
def avg_cost (m : ℝ) : Prop :=
  1 = 4 * (m / (m + 0.6))

-- Prove the average cost per kilometer of the electric vehicle
theorem avg_cost_electric_vehicle (m : ℝ) (h : avg_cost m) : m = 0.2 :=
by {
  -- Proof skipped
  sorry
}

-- Define annual cost equations and the comparison condition
variables {a : ℝ}
def annual_cost_electric_vehicle (a : ℝ) : ℝ :=
  0.2 * a + 6400

def annual_cost_gasoline_vehicle (a : ℝ) : ℝ :=
  0.8 * a + 4000

-- Prove that when the annual mileage is greater than 6667 kilometers, the annual cost of buying an electric vehicle is lower
theorem cost_comparison (a : ℝ) (h : a > 6667) : annual_cost_electric_vehicle a < annual_cost_gasoline_vehicle a :=
by {
  -- Proof skipped
  sorry
}

end skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l76_76287


namespace garden_perimeter_l76_76603

-- Definitions for length and breadth
def length := 150
def breadth := 100

-- Theorem that states the perimeter of the rectangular garden
theorem garden_perimeter : (2 * (length + breadth)) = 500 :=
by sorry

end garden_perimeter_l76_76603


namespace comparison_among_three_numbers_l76_76329

theorem comparison_among_three_numbers (a b c : ℝ) (h1 : a = 7 ^ 0.3) (h2 : b = 0.3 ^ 7) (h3 : c = Real.log 0.3) 
  (h4 : a > 1) (h5 : 0 < b ∧ b < 1) (h6 : c < 0) : a > b ∧ b > c :=
by
  sorry

end comparison_among_three_numbers_l76_76329


namespace distinct_solutions_diff_l76_76149

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l76_76149


namespace largest_prime_divisor_S_l76_76086

-- Define the product of non-zero digits function
def p (n : ℕ) : ℕ :=
  (n.digits 10).filter (≠ 0).prod

-- Define S as the sum of p_i for i from 1 to 999
def S : ℕ :=
  (list.range 1 1000).map p).sum

-- Statement of the problem
theorem largest_prime_divisor_S : ∃ (p : ℕ), p.prime ∧ p.factors.max' = 103 :=
sorry

end largest_prime_divisor_S_l76_76086


namespace complement_correct_l76_76551

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}

theorem complement_correct : (U \ A) = {2, 4} := by
  sorry

end complement_correct_l76_76551


namespace longest_segment_in_cylinder_l76_76973

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l76_76973


namespace percentage_corresponding_to_120_l76_76728

variable (x p : ℝ)

def forty_percent_eq_160 := (0.4 * x = 160)
def p_times_x_eq_120 := (p * x = 120)

theorem percentage_corresponding_to_120 (h₁ : forty_percent_eq_160 x) (h₂ : p_times_x_eq_120 x p) :
  p = 0.30 :=
sorry

end percentage_corresponding_to_120_l76_76728


namespace find_larger_number_l76_76043

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l76_76043


namespace evaluate_expression_zero_l76_76677

-- Main proof statement
theorem evaluate_expression_zero :
  ∀ (a d c b : ℤ),
    d = c + 5 →
    c = b - 8 →
    b = a + 3 →
    a = 3 →
    a - 1 ≠ 0 →
    d - 6 ≠ 0 →
    c + 4 ≠ 0 →
    (a + 3) * (d - 3) * (c + 9) = 0 :=
by
  intros a d c b hd hc hb ha h1 h2 h3
  sorry -- The proof goes here

end evaluate_expression_zero_l76_76677


namespace wizard_answers_bal_l76_76194

-- Define the types for human and zombie as truth-tellers and liars respectively
inductive WizardType
| human : WizardType
| zombie : WizardType

-- Define the meaning of "bal"
inductive BalMeaning
| yes : BalMeaning
| no : BalMeaning

-- Question asked to the wizard
def question (w : WizardType) (b : BalMeaning) : Prop :=
  match w, b with
  | WizardType.human, BalMeaning.yes => true
  | WizardType.human, BalMeaning.no => false
  | WizardType.zombie, BalMeaning.yes => false
  | WizardType.zombie, BalMeaning.no => true

-- Theorem stating the wizard will answer "bal" to the given question
theorem wizard_answers_bal (w : WizardType) (b : BalMeaning) :
  question w b = true ↔ b = BalMeaning.yes :=
by
  sorry

end wizard_answers_bal_l76_76194


namespace valid_starting_day_count_l76_76206

-- Defining the structure of the 30-day month and conditions
def days_in_month : Nat := 30

-- A function to determine the number of each weekday in a month which also checks if the given day is valid as per conditions
def valid_starting_days : List Nat :=
  [1] -- '1' represents Tuesday being the valid starting day corresponding to equal number of Tuesdays and Thursdays

-- The theorem we want to prove
-- The goal is to prove that there is only 1 valid starting day for the 30-day month to have equal number of Tuesdays and Thursdays
theorem valid_starting_day_count (days : Nat) (valid_days : List Nat) : 
  days = days_in_month → valid_days = valid_starting_days :=
by
  -- Sorry to skip full proof implementation
  sorry

end valid_starting_day_count_l76_76206


namespace compound_interest_rate_l76_76218

theorem compound_interest_rate (
  P : ℝ) (r : ℝ)  (A : ℕ → ℝ) :
  A 2 = 2420 ∧ A 3 = 3025 ∧ 
  (∀ n : ℕ, A n = P * (1 + r / 100)^n) → r = 25 :=
by
  sorry

end compound_interest_rate_l76_76218


namespace roots_satisfy_conditions_l76_76520

variable (a x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * x2 + x1 + x2 - a = 0
def condition2 : Prop := x1 * x2 - a * (x1 + x2) + 1 = 0

-- Derived quadratic equation
def quadratic_eq : Prop := ∃ x : ℝ, x^2 - x + (a - 1) = 0

theorem roots_satisfy_conditions (h1: condition1 a x1 x2) (h2: condition2 a x1 x2) : quadratic_eq a :=
  sorry

end roots_satisfy_conditions_l76_76520


namespace swap_correct_l76_76130

variable (a b c : ℕ)

noncomputable def swap_and_verify (a : ℕ) (b : ℕ) : Prop :=
  let c := b
  let b := a
  let a := c
  a = 2012 ∧ b = 2011

theorem swap_correct :
  ∀ a b : ℕ, a = 2011 → b = 2012 → swap_and_verify a b :=
by
  intros a b ha hb
  sorry

end swap_correct_l76_76130


namespace amber_max_ounces_l76_76662

-- Define the problem parameters:
def cost_candy : ℝ := 1
def ounces_candy : ℝ := 12
def cost_chips : ℝ := 1.4
def ounces_chips : ℝ := 17
def total_money : ℝ := 7

-- Define the number of bags of each item Amber can buy:
noncomputable def bags_candy := (total_money / cost_candy).to_int
noncomputable def bags_chips  := (total_money / cost_chips).to_int

-- Define the total ounces of each item:
noncomputable def total_ounces_candy := bags_candy * ounces_candy
noncomputable def total_ounces_chips := bags_chips * ounces_chips

-- Problem statement asking to prove Amber gets the most ounces by buying chips:
theorem amber_max_ounces : max total_ounces_candy total_ounces_chips = total_ounces_chips :=
by sorry

end amber_max_ounces_l76_76662


namespace total_bathing_suits_l76_76064

theorem total_bathing_suits 
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ)
  (ha : a = 8500) (hb : b = 12750) (hc : c = 5900) (hd : d = 7250) (he : e = 1100) :
  a + b + c + d + e = 35500 :=
by
  sorry

end total_bathing_suits_l76_76064


namespace initial_kittens_l76_76511

theorem initial_kittens (kittens_given : ℕ) (kittens_left : ℕ) (initial_kittens : ℕ) :
  kittens_given = 4 → kittens_left = 4 → initial_kittens = kittens_given + kittens_left → initial_kittens = 8 :=
by
  intros hg hl hi
  rw [hg, hl] at hi
  -- Skipping proof detail
  sorry

end initial_kittens_l76_76511


namespace sum_of_remainders_mod_13_l76_76344

theorem sum_of_remainders_mod_13 :
  ∀ (a b c d e : ℤ),
    a ≡ 3 [ZMOD 13] →
    b ≡ 5 [ZMOD 13] →
    c ≡ 7 [ZMOD 13] →
    d ≡ 9 [ZMOD 13] →
    e ≡ 11 [ZMOD 13] →
    (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_of_remainders_mod_13_l76_76344


namespace find_a7_l76_76697

variable {a : ℕ → ℝ}

-- Conditions
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, a (n + 1) = a n * q

axiom a3_eq_4 : a 3 = 4
axiom harmonic_condition : (1 / a 1 + 1 / a 5 = 5 / 8)
axiom increasing_geometric : is_increasing_geometric_sequence a

-- The problem is to prove that a 7 = 16 given the above conditions.
theorem find_a7 : a 7 = 16 :=
by
  -- Proof goes here
  sorry

end find_a7_l76_76697
