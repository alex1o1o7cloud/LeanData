import Mathlib

namespace NUMINAMATH_GPT_rhombus_area_l1736_173696

theorem rhombus_area (s : ℝ) (d1 d2 : ℝ) (h1 : s = Real.sqrt 145) (h2 : abs (d1 - d2) = 10) : 
  (1/2) * d1 * d2 = 100 :=
sorry

end NUMINAMATH_GPT_rhombus_area_l1736_173696


namespace NUMINAMATH_GPT_product_of_three_numbers_l1736_173614

theorem product_of_three_numbers :
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ a = 2 * (b + c) ∧ b = 6 * c ∧ a * b * c = 12000 / 49 :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l1736_173614


namespace NUMINAMATH_GPT_total_cost_proof_l1736_173640

noncomputable def cost_proof : Prop :=
  let M := 158.4
  let R := 66
  let F := 22
  (10 * M = 24 * R) ∧ (6 * F = 2 * R) ∧ (F = 22) →
  (4 * M + 3 * R + 5 * F = 941.6)

theorem total_cost_proof : cost_proof :=
by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l1736_173640


namespace NUMINAMATH_GPT_Jorge_goals_total_l1736_173606

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end NUMINAMATH_GPT_Jorge_goals_total_l1736_173606


namespace NUMINAMATH_GPT_diamond_evaluation_l1736_173698

def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem diamond_evaluation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 :=
  by
  sorry

end NUMINAMATH_GPT_diamond_evaluation_l1736_173698


namespace NUMINAMATH_GPT_cost_per_sqft_is_6_l1736_173653

-- Define the dimensions of the room
def room_length : ℕ := 25
def room_width : ℕ := 15
def room_height : ℕ := 12

-- Define the dimensions of the door
def door_height : ℕ := 6
def door_width : ℕ := 3

-- Define the dimensions of the windows
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3

-- Define the total cost of whitewashing
def total_cost : ℕ := 5436

-- Calculate areas
def area_one_pair_of_walls : ℕ :=
  (room_length * room_height) * 2

def area_other_pair_of_walls : ℕ :=
  (room_width * room_height) * 2

def total_wall_area : ℕ :=
  area_one_pair_of_walls + area_other_pair_of_walls

def door_area : ℕ :=
  door_height * door_width

def window_area : ℕ :=
  window_height * window_width

def total_window_area : ℕ :=
  window_area * number_of_windows

def area_to_be_whitewashed : ℕ :=
  total_wall_area - (door_area + total_window_area)

def cost_per_sqft : ℕ :=
  total_cost / area_to_be_whitewashed

-- The theorem statement proving the cost per square foot is 6
theorem cost_per_sqft_is_6 : cost_per_sqft = 6 := 
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_per_sqft_is_6_l1736_173653


namespace NUMINAMATH_GPT_max_viewing_area_l1736_173682

theorem max_viewing_area (L W: ℝ) (h1: 2 * L + 2 * W = 420) (h2: L ≥ 100) (h3: W ≥ 60) : 
  (L = 105) ∧ (W = 105) ∧ (L * W = 11025) :=
by
  sorry

end NUMINAMATH_GPT_max_viewing_area_l1736_173682


namespace NUMINAMATH_GPT_quadratic_no_third_quadrant_l1736_173627

theorem quadratic_no_third_quadrant (x y : ℝ) : 
  (y = x^2 - 2 * x) → ¬(x < 0 ∧ y < 0) :=
by
  intro hy
  sorry

end NUMINAMATH_GPT_quadratic_no_third_quadrant_l1736_173627


namespace NUMINAMATH_GPT_max_collection_l1736_173603

theorem max_collection : 
  let Yoongi := 4 
  let Jungkook := 6 / 3 
  let Yuna := 5 
  max Yoongi (max Jungkook Yuna) = 5 :=
by 
  let Yoongi := 4
  let Jungkook := (6 / 3) 
  let Yuna := 5
  show max Yoongi (max Jungkook Yuna) = 5
  sorry

end NUMINAMATH_GPT_max_collection_l1736_173603


namespace NUMINAMATH_GPT_gumballs_remaining_l1736_173611

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end NUMINAMATH_GPT_gumballs_remaining_l1736_173611


namespace NUMINAMATH_GPT_parametric_to_ordinary_eq_l1736_173643

variable (t : ℝ)

theorem parametric_to_ordinary_eq (h1 : x = Real.sqrt t + 1) (h2 : y = 2 * Real.sqrt t - 1) (h3 : t ≥ 0) :
    y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end NUMINAMATH_GPT_parametric_to_ordinary_eq_l1736_173643


namespace NUMINAMATH_GPT_range_of_f_l1736_173665

def f (x : ℕ) : ℤ := 2 * x - 3

def domain := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

def range (f : ℕ → ℤ) (s : Set ℕ) : Set ℤ :=
  {y : ℤ | ∃ x ∈ s, f x = y}

theorem range_of_f :
  range f domain = {-1, 1, 3, 5, 7} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1736_173665


namespace NUMINAMATH_GPT_Sara_snow_volume_l1736_173664

theorem Sara_snow_volume :
  let length := 30
  let width := 3
  let first_half_length := length / 2
  let second_half_length := length / 2
  let depth1 := 0.5
  let depth2 := 1.0 / 3.0
  let volume1 := first_half_length * width * depth1
  let volume2 := second_half_length * width * depth2
  volume1 + volume2 = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_Sara_snow_volume_l1736_173664


namespace NUMINAMATH_GPT_egor_last_payment_l1736_173675

theorem egor_last_payment (a b c d : ℕ) (h_sum : a + b + c + d = 28)
  (h1 : b ≥ 2 * a) (h2 : c ≥ 2 * b) (h3 : d ≥ 2 * c) : d = 18 := by
  sorry

end NUMINAMATH_GPT_egor_last_payment_l1736_173675


namespace NUMINAMATH_GPT_mark_profit_l1736_173686

def initialPrice : ℝ := 100
def finalPrice : ℝ := 3 * initialPrice
def salesTax : ℝ := 0.05 * initialPrice
def totalInitialCost : ℝ := initialPrice + salesTax
def transactionFee : ℝ := 0.03 * finalPrice
def profitBeforeTax : ℝ := finalPrice - totalInitialCost
def capitalGainsTax : ℝ := 0.15 * profitBeforeTax
def totalProfit : ℝ := profitBeforeTax - transactionFee - capitalGainsTax

theorem mark_profit : totalProfit = 147.75 := sorry

end NUMINAMATH_GPT_mark_profit_l1736_173686


namespace NUMINAMATH_GPT_solve_for_b_l1736_173663

theorem solve_for_b (x y b : ℝ) (h1: 4 * x + y = b) (h2: 3 * x + 4 * y = 3 * b) (hx: x = 3) : b = 39 :=
sorry

end NUMINAMATH_GPT_solve_for_b_l1736_173663


namespace NUMINAMATH_GPT_second_player_can_form_palindrome_l1736_173670

def is_palindrome (s : List Char) : Prop :=
  s = s.reverse

theorem second_player_can_form_palindrome :
  ∀ (moves : List Char), moves.length = 1999 →
  ∃ (sequence : List Char), sequence.length = 1999 ∧ is_palindrome sequence :=
by
  sorry

end NUMINAMATH_GPT_second_player_can_form_palindrome_l1736_173670


namespace NUMINAMATH_GPT_price_of_each_sundae_l1736_173608

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ := 125) 
  (num_sundaes : ℕ := 125) 
  (total_price : ℝ := 225)
  (price_per_ice_cream_bar : ℝ := 0.60) :
  ∃ (price_per_sundae : ℝ), price_per_sundae = 1.20 := 
by
  -- Variables for costs of ice-cream bars and sundaes' total cost
  let cost_ice_cream_bars := num_ice_cream_bars * price_per_ice_cream_bar
  let total_cost_sundaes := total_price - cost_ice_cream_bars
  let price_per_sundae := total_cost_sundaes / num_sundaes
  use price_per_sundae
  sorry

end NUMINAMATH_GPT_price_of_each_sundae_l1736_173608


namespace NUMINAMATH_GPT_find_a_for_extraneous_roots_find_a_for_no_solution_l1736_173637

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end NUMINAMATH_GPT_find_a_for_extraneous_roots_find_a_for_no_solution_l1736_173637


namespace NUMINAMATH_GPT_probability_non_expired_bags_l1736_173676

theorem probability_non_expired_bags :
  let total_bags := 5
  let expired_bags := 2
  let selected_bags := 2
  let total_combinations := Nat.choose total_bags selected_bags
  let non_expired_bags := total_bags - expired_bags
  let favorable_outcomes := Nat.choose non_expired_bags selected_bags
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_non_expired_bags_l1736_173676


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1736_173657

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 11) (h3 : a = b ∨ b = b) :
  (5 + 11 + 11 = 27) := 
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1736_173657


namespace NUMINAMATH_GPT_min_pieces_pie_l1736_173620

theorem min_pieces_pie (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n : ℕ, n = p + q - 1 ∧ 
    (∀ m, m < n → ¬ (∀ k : ℕ, (k < p → n % p = 0) ∧ (k < q → n % q = 0))) :=
sorry

end NUMINAMATH_GPT_min_pieces_pie_l1736_173620


namespace NUMINAMATH_GPT_correct_transformation_l1736_173604

-- Definitions of the equations and their transformations
def optionA := (forall (x : ℝ), ((x / 5) + 1 = x / 2) -> (2 * x + 10 = 5 * x))
def optionB := (forall (x : ℝ), (5 - 2 * (x - 1) = x + 3) -> (5 - 2 * x + 2 = x + 3))
def optionC := (forall (x : ℝ), (5 * x + 3 = 8) -> (5 * x = 8 - 3))
def optionD := (forall (x : ℝ), (3 * x = -7) -> (x = -7 / 3))

-- Theorem stating that option D is the correct transformation
theorem correct_transformation : optionD := 
by 
  sorry

end NUMINAMATH_GPT_correct_transformation_l1736_173604


namespace NUMINAMATH_GPT_find_t_l1736_173678

variables {m n : ℝ}
variables (t : ℝ)
variables (mv nv : ℝ)
variables (dot_m_m dot_m_n dot_n_n : ℝ)
variables (cos_theta : ℝ)

-- Define the basic assumptions
axiom non_zero_vectors : m ≠ 0 ∧ n ≠ 0
axiom magnitude_condition : mv = 2 * nv
axiom cos_condition : cos_theta = 1 / 3
axiom perpendicular_condition : dot_m_n = (mv * nv * cos_theta) ∧ (t * dot_m_n + dot_m_m = 0)

-- Utilize the conditions and prove the target
theorem find_t : t = -6 :=
sorry

end NUMINAMATH_GPT_find_t_l1736_173678


namespace NUMINAMATH_GPT_equation_has_unique_integer_solution_l1736_173691

theorem equation_has_unique_integer_solution:
  ∀ m n : ℤ, (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n → m = 0 ∧ n = 0 := by
  intro m n
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_equation_has_unique_integer_solution_l1736_173691


namespace NUMINAMATH_GPT_cos_shifted_alpha_l1736_173674

theorem cos_shifted_alpha (α : ℝ) (h1 : Real.tan α = -3/4) (h2 : α ∈ Set.Ioc (3*Real.pi/2) (2*Real.pi)) :
  Real.cos (Real.pi/2 + α) = 3/5 :=
sorry

end NUMINAMATH_GPT_cos_shifted_alpha_l1736_173674


namespace NUMINAMATH_GPT_geometric_sequence_a3_l1736_173654

variable {a : ℕ → ℝ} (h1 : a 1 > 0) (h2 : a 2 * a 4 = 25)
def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (h_geom : geometric_sequence a) : 
  a 3 = 5 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l1736_173654


namespace NUMINAMATH_GPT_cost_of_book_first_sold_at_loss_l1736_173680

theorem cost_of_book_first_sold_at_loss (C1 C2 C3 : ℝ) (h1 : C1 + C2 + C3 = 810)
    (h2 : 0.88 * C1 = 1.18 * C2) (h3 : 0.88 * C1 = 1.27 * C3) : 
    C1 = 333.9 := 
by
  -- Conditions given
  have h4 : C2 = 0.88 * C1 / 1.18 := by sorry
  have h5 : C3 = 0.88 * C1 / 1.27 := by sorry

  -- Substituting back into the total cost equation
  have h6 : C1 + 0.88 * C1 / 1.18 + 0.88 * C1 / 1.27 = 810 := by sorry

  -- Simplifying and solving for C1
  have h7 : C1 = 333.9 := by sorry

  -- Conclusion
  exact h7

end NUMINAMATH_GPT_cost_of_book_first_sold_at_loss_l1736_173680


namespace NUMINAMATH_GPT_mary_initial_nickels_l1736_173690

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end NUMINAMATH_GPT_mary_initial_nickels_l1736_173690


namespace NUMINAMATH_GPT_sum_of_fractions_eq_13_5_l1736_173693

noncomputable def sumOfFractions : ℚ :=
  (1/10 + 2/10 + 3/10 + 4/10 + 5/10 + 6/10 + 7/10 + 8/10 + 9/10 + 90/10)

theorem sum_of_fractions_eq_13_5 :
  sumOfFractions = 13.5 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_eq_13_5_l1736_173693


namespace NUMINAMATH_GPT_negation_proposition_l1736_173695

theorem negation_proposition:
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
by sorry

end NUMINAMATH_GPT_negation_proposition_l1736_173695


namespace NUMINAMATH_GPT_find_n_satisfies_equation_l1736_173641

-- Definition of the problem:
def satisfies_equation (n : ℝ) : Prop := 
  (2 / (n + 1)) + (3 / (n + 1)) + (n / (n + 1)) = 4

-- The statement of the proof problem:
theorem find_n_satisfies_equation : 
  ∃ n : ℝ, satisfies_equation n ∧ n = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_satisfies_equation_l1736_173641


namespace NUMINAMATH_GPT_joan_friends_kittens_l1736_173630

theorem joan_friends_kittens (initial_kittens final_kittens friends_kittens : ℕ) 
  (h1 : initial_kittens = 8) 
  (h2 : final_kittens = 10) 
  (h3 : friends_kittens = 2) : 
  final_kittens - initial_kittens = friends_kittens := 
by 
  -- Sorry is used here as a placeholder to indicate where the proof would go.
  sorry

end NUMINAMATH_GPT_joan_friends_kittens_l1736_173630


namespace NUMINAMATH_GPT_min_value_of_reciprocals_l1736_173660

theorem min_value_of_reciprocals (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) : 
  ∃ (x : ℝ), x = 2 * a + b ∧ ∃ (y : ℝ), y = 2 * b + c ∧ ∃ (z : ℝ), z = 2 * c + a ∧ (1 / x + 1 / y + 1 / z = 27 / 8) :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocals_l1736_173660


namespace NUMINAMATH_GPT_select_two_integers_divisibility_l1736_173649

open Polynomial

theorem select_two_integers_divisibility
  (F : Polynomial ℤ)
  (m : ℕ)
  (a : Fin m → ℤ)
  (H : ∀ n : ℤ, ∃ i : Fin m, a i ∣ F.eval n) :
  ∃ i j : Fin m, i ≠ j ∧ ∀ n : ℤ, ∃ k : Fin m, k = i ∨ k = j ∧ a k ∣ F.eval n :=
by
  sorry

end NUMINAMATH_GPT_select_two_integers_divisibility_l1736_173649


namespace NUMINAMATH_GPT_number_of_subsets_of_set_l1736_173689

theorem number_of_subsets_of_set {n : ℕ} (h : n = 2016) :
  (2^2016) = 2^2016 :=
by
  sorry

end NUMINAMATH_GPT_number_of_subsets_of_set_l1736_173689


namespace NUMINAMATH_GPT_quilt_shaded_fraction_l1736_173612

theorem quilt_shaded_fraction :
  let original_squares := 9
  let shaded_column_squares := 3
  let fraction_shaded := shaded_column_squares / original_squares 
  fraction_shaded = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_quilt_shaded_fraction_l1736_173612


namespace NUMINAMATH_GPT_ak_divisibility_l1736_173638

theorem ak_divisibility {a k m n : ℕ} (h : a ^ k % (m ^ n) = 0) : a ^ (k * m) % (m ^ (n + 1)) = 0 :=
sorry

end NUMINAMATH_GPT_ak_divisibility_l1736_173638


namespace NUMINAMATH_GPT_inequality_abc_l1736_173636

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1736_173636


namespace NUMINAMATH_GPT_part1_part2_l1736_173621

-- Defining set A
def A : Set ℝ := {x | x^2 + 4 * x = 0}

-- Defining set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

-- Problem 1: Prove that if A ∩ B = A ∪ B, then a = 1
theorem part1 (a : ℝ) : (A ∩ (B a) = A ∪ (B a)) → a = 1 := by
  sorry

-- Problem 2: Prove the range of values for a if A ∩ B = B
theorem part2 (a : ℝ) : (A ∩ (B a) = B a) → a ∈ Set.Iic (-1) ∪ {1} := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1736_173621


namespace NUMINAMATH_GPT_car_value_proof_l1736_173688

-- Let's define the variables and the conditions.
def car_sold_value : ℝ := 20000
def sticker_price_new_car : ℝ := 30000
def percent_sold : ℝ := 0.80
def percent_paid : ℝ := 0.90
def out_of_pocket : ℝ := 11000

theorem car_value_proof :
  (percent_paid * sticker_price_new_car - percent_sold * car_sold_value = out_of_pocket) →
  car_sold_value = 20000 := 
by
  intros h
  -- Introduction of any intermediate steps if necessary should just invoke the sorry to indicate the need for proof later
  exact sorry

end NUMINAMATH_GPT_car_value_proof_l1736_173688


namespace NUMINAMATH_GPT_original_people_in_room_l1736_173626

theorem original_people_in_room (x : ℕ) 
  (h1 : 3 * x / 4 - 3 * x / 20 = 16) : x = 27 :=
sorry

end NUMINAMATH_GPT_original_people_in_room_l1736_173626


namespace NUMINAMATH_GPT_starting_player_can_ensure_integer_roots_l1736_173602

theorem starting_player_can_ensure_integer_roots :
  ∃ (a b c : ℤ), ∀ (x : ℤ), (x^3 + a * x^2 + b * x + c = 0) →
  (∃ r1 r2 r3 : ℤ, x = r1 ∨ x = r2 ∨ x = r3) :=
sorry

end NUMINAMATH_GPT_starting_player_can_ensure_integer_roots_l1736_173602


namespace NUMINAMATH_GPT_n_times_s_l1736_173694

noncomputable def f (x : ℝ) : ℝ := sorry

theorem n_times_s : (f 0 = 0 ∨ f 0 = 1) ∧
  (∀ (y : ℝ), f 0 = 0 → False) ∧
  (∀ (x y : ℝ), f x * f y - f (x * y) = x^2 + y^2) → 
  let n : ℕ := if f 0 = 0 then 1 else 1
  let s : ℝ := if f 0 = 0 then 0 else 1
  n * s = 1 :=
by
  sorry

end NUMINAMATH_GPT_n_times_s_l1736_173694


namespace NUMINAMATH_GPT_green_apples_ordered_l1736_173622

-- Definitions based on the conditions
variable (red_apples : Nat := 25)
variable (students : Nat := 10)
variable (extra_apples : Nat := 32)
variable (G : Nat)

-- The mathematical problem to prove
theorem green_apples_ordered :
  red_apples + G - students = extra_apples → G = 17 := by
  sorry

end NUMINAMATH_GPT_green_apples_ordered_l1736_173622


namespace NUMINAMATH_GPT_ceiling_is_multiple_of_3_l1736_173661

-- Given conditions:
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1
axiom exists_three_real_roots : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
  polynomial x1 = 0 ∧ polynomial x2 = 0 ∧ polynomial x3 = 0

-- Goal:
theorem ceiling_is_multiple_of_3 (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3)
  (hx1 : polynomial x1 = 0) (hx2 : polynomial x2 = 0) (hx3 : polynomial x3 = 0):
  ∀ n : ℕ, n > 0 → ∃ k : ℤ, k * 3 = ⌈x3^n⌉ := by
  sorry

end NUMINAMATH_GPT_ceiling_is_multiple_of_3_l1736_173661


namespace NUMINAMATH_GPT_abs_pos_of_ne_zero_l1736_173671

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end NUMINAMATH_GPT_abs_pos_of_ne_zero_l1736_173671


namespace NUMINAMATH_GPT_geometric_sum_thm_l1736_173605

variable (S : ℕ → ℝ)

theorem geometric_sum_thm (h1 : S n = 48) (h2 : S (2 * n) = 60) : S (3 * n) = 63 :=
sorry

end NUMINAMATH_GPT_geometric_sum_thm_l1736_173605


namespace NUMINAMATH_GPT_tank_capacity_l1736_173619

theorem tank_capacity (T : ℝ) (h : 0.4 * T = 0.9 * T - 36) : T = 72 := by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1736_173619


namespace NUMINAMATH_GPT_Sperner_theorem_example_l1736_173667

theorem Sperner_theorem_example :
  ∀ (S : Finset (Finset ℕ)), (S.card = 10) →
  (∀ (A B : Finset ℕ), A ∈ S → B ∈ S → A ⊆ B → A = B) → S.card = 252 :=
by sorry

end NUMINAMATH_GPT_Sperner_theorem_example_l1736_173667


namespace NUMINAMATH_GPT_walkway_area_correct_l1736_173633

-- Define the dimensions of one flower bed
def flower_bed_width := 8
def flower_bed_height := 3

-- Define the number of flower beds and the width of the walkways
def num_flowers_horizontal := 3
def num_flowers_vertical := 4
def walkway_width := 2

-- Calculate the total dimension of the garden including both flower beds and walkways
def total_garden_width := (num_flowers_horizontal * flower_bed_width) + ((num_flowers_horizontal + 1) * walkway_width)
def total_garden_height := (num_flowers_vertical * flower_bed_height) + ((num_flowers_vertical + 1) * walkway_width)

-- Calculate the total area of the garden and the total area of the flower beds
def total_garden_area := total_garden_width * total_garden_height
def total_flower_bed_area := (flower_bed_width * flower_bed_height) * (num_flowers_horizontal * num_flowers_vertical)

-- Calculate the total area of the walkways in the garden
def total_walkway_area := total_garden_area - total_flower_bed_area

-- The statement to be proven:
theorem walkway_area_correct : total_walkway_area = 416 := by
  sorry

end NUMINAMATH_GPT_walkway_area_correct_l1736_173633


namespace NUMINAMATH_GPT_hyperbola_center_l1736_173677

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x ^ 2 + 54 * x - 16 * y ^ 2 - 128 * y - 200 = 0) : 
  (x = -3) ∧ (y = -4) := 
sorry

end NUMINAMATH_GPT_hyperbola_center_l1736_173677


namespace NUMINAMATH_GPT_cycle_selling_price_l1736_173655

theorem cycle_selling_price
  (cost_price : ℝ)
  (gain_percentage : ℝ)
  (profit : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 930)
  (h2 : gain_percentage = 30.107526881720432)
  (h3 : profit = (gain_percentage / 100) * cost_price)
  (h4 : selling_price = cost_price + profit)
  : selling_price = 1210 := 
sorry

end NUMINAMATH_GPT_cycle_selling_price_l1736_173655


namespace NUMINAMATH_GPT_power_of_binomials_l1736_173672

theorem power_of_binomials :
  (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 :=
by
  -- This is a placeholder for the actual proof steps.
  -- We use 'sorry' to indicate that the proof is omitted here.
  sorry

end NUMINAMATH_GPT_power_of_binomials_l1736_173672


namespace NUMINAMATH_GPT_fourth_derivative_l1736_173628

noncomputable def f (x : ℝ) : ℝ := (5 * x - 8) * 2^(-x)

theorem fourth_derivative (x : ℝ) : 
  deriv (deriv (deriv (deriv f))) x = 2^(-x) * (Real.log 2)^4 * (5 * x - 9) :=
sorry

end NUMINAMATH_GPT_fourth_derivative_l1736_173628


namespace NUMINAMATH_GPT_circle_line_distance_condition_l1736_173683

theorem circle_line_distance_condition :
  ∀ (c : ℝ), 
    (∃ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 8 = 0 ∧ (x - y + c = 2 ∨ x - y + c = -2)) →
    -2*Real.sqrt 2 ≤ c ∧ c ≤ 2*Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_circle_line_distance_condition_l1736_173683


namespace NUMINAMATH_GPT_ball_returns_velocity_required_initial_velocity_to_stop_l1736_173687

-- Define the conditions.
def distance_A_to_wall : ℝ := 5
def distance_wall_to_B : ℝ := 2
def distance_AB : ℝ := 9
def initial_velocity_v0 : ℝ := 5
def acceleration_a : ℝ := -0.4

-- Hypothesize that the velocity when the ball returns to A is 3 m/s.
theorem ball_returns_velocity (t : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  initial_velocity_v0 * t + (1 / 2) * acceleration_a * t^2 = distance_AB + distance_A_to_wall →
  initial_velocity_v0 + acceleration_a * t = 3 := sorry

-- Hypothesize that to stop exactly at A, the initial speed should be 4 m/s.
theorem required_initial_velocity_to_stop (t' : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  (0.4 * t') * t' + (1 / 2) * acceleration_a * t'^2 = distance_AB + distance_A_to_wall →
  0.4 * t' = 4 := sorry

end NUMINAMATH_GPT_ball_returns_velocity_required_initial_velocity_to_stop_l1736_173687


namespace NUMINAMATH_GPT_manolo_rate_change_after_one_hour_l1736_173601

variable (masks_in_first_hour : ℕ)
variable (masks_in_remaining_time : ℕ)
variable (total_masks : ℕ)

-- Define conditions as Lean definitions
def first_hour_rate := 1 / 4  -- masks per minute
def remaining_time_rate := 1 / 6  -- masks per minute
def total_time := 4  -- hours
def masks_produced_in_first_hour (t : ℕ) := t * 15  -- t hours, 60 minutes/hour, at 15 masks/hour
def masks_produced_in_remaining_time (t : ℕ) := t * 10 -- (total_time - 1) hours, 60 minutes/hour, at 10 masks/hour

-- Main proof problem statement
theorem manolo_rate_change_after_one_hour :
  masks_in_first_hour = masks_produced_in_first_hour 1 →
  masks_in_remaining_time = masks_produced_in_remaining_time (total_time - 1) →
  total_masks = masks_in_first_hour + masks_in_remaining_time →
  (∃ t : ℕ, t = 1) :=
by
  -- Placeholder, proof not required
  sorry

end NUMINAMATH_GPT_manolo_rate_change_after_one_hour_l1736_173601


namespace NUMINAMATH_GPT_num_ways_distribute_balls_l1736_173668

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end NUMINAMATH_GPT_num_ways_distribute_balls_l1736_173668


namespace NUMINAMATH_GPT_find_number_l1736_173650

variable (x : ℝ)

theorem find_number (h : 20 * (x / 5) = 40) : x = 10 := by
  sorry

end NUMINAMATH_GPT_find_number_l1736_173650


namespace NUMINAMATH_GPT_intersection_of_sets_l1736_173652

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

def B : Set ℝ := Ico 0 4  -- Ico stands for interval [0, 4)

theorem intersection_of_sets : A ∩ B = Ico 2 4 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1736_173652


namespace NUMINAMATH_GPT_sum_of_ammeter_readings_l1736_173662

def I1 := 4 
def I2 := 4
def I3 := 2 * I2
def I5 := I3 + I2
def I4 := (5 / 3) * I5

theorem sum_of_ammeter_readings : I1 + I2 + I3 + I4 + I5 = 48 := by
  sorry

end NUMINAMATH_GPT_sum_of_ammeter_readings_l1736_173662


namespace NUMINAMATH_GPT_people_joined_l1736_173634

theorem people_joined (total_left : ℕ) (total_remaining : ℕ) (Molly_and_parents : ℕ)
  (h1 : total_left = 40) (h2 : total_remaining = 63) (h3 : Molly_and_parents = 3) :
  ∃ n, n = 100 := 
by
  sorry

end NUMINAMATH_GPT_people_joined_l1736_173634


namespace NUMINAMATH_GPT_scrap_metal_collected_l1736_173623

theorem scrap_metal_collected (a b : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9)
  (h₂ : 900 + 10 * a + b - (100 * a + 10 * b + 9) = 216) :
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by
  sorry

end NUMINAMATH_GPT_scrap_metal_collected_l1736_173623


namespace NUMINAMATH_GPT_min_num_stamps_is_17_l1736_173645

-- Definitions based on problem conditions
def initial_num_stamps : ℕ := 2 + 5 + 3 + 1
def initial_cost : ℝ := 2 * 0.10 + 5 * 0.20 + 3 * 0.50 + 1 * 2
def remaining_cost : ℝ := 10 - initial_cost
def additional_stamps : ℕ := 2 + 2 + 1 + 1
def total_stamps : ℕ := initial_num_stamps + additional_stamps

-- Proof that the minimum number of stamps bought is 17
theorem min_num_stamps_is_17 : total_stamps = 17 := by
  sorry

end NUMINAMATH_GPT_min_num_stamps_is_17_l1736_173645


namespace NUMINAMATH_GPT_solution1_solution2_l1736_173629

-- Definition for problem (1)
def problem1 : ℚ :=
  - (1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3 : ℚ) ^ 2) / (-7 : ℚ)

theorem solution1 : problem1 = -7 / 6 :=
by
  sorry

-- Definition for problem (2)
def problem2 : ℚ :=
  ((3 / 2 : ℚ) - (5 / 8) + (7 / 12)) / (-1 / 24) - 8 * ((-1 / 2 : ℚ) ^ 3)

theorem solution2 : problem2 = -34 :=
by
  sorry

end NUMINAMATH_GPT_solution1_solution2_l1736_173629


namespace NUMINAMATH_GPT_percentage_above_wholesale_cost_l1736_173631

def wholesale_cost : ℝ := 200
def paid_price : ℝ := 228
def discount_rate : ℝ := 0.05

theorem percentage_above_wholesale_cost :
  ∃ P : ℝ, P = 20 ∧ 
    paid_price = (1 - discount_rate) * (wholesale_cost + P/100 * wholesale_cost) :=
by
  sorry

end NUMINAMATH_GPT_percentage_above_wholesale_cost_l1736_173631


namespace NUMINAMATH_GPT_parabola_coordinates_and_area_l1736_173607

theorem parabola_coordinates_and_area
  (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (3, 0))
  (hC : C = (5 / 2, 1 / 4))
  (h_vertex : ∀ x y, y = -x^2 + 5 * x - 6 → 
                   ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C)) :
  A = (2, 0) ∧ B = (3, 0) ∧ C = (5 / 2, 1 / 4)
  ∧ (1 / 2 * (3 - 2) * (1 / 4) = 1 / 8) := 
by
  sorry

end NUMINAMATH_GPT_parabola_coordinates_and_area_l1736_173607


namespace NUMINAMATH_GPT_min_value_of_2a7_a11_l1736_173651

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the sequence terms

-- Conditions
axiom geometric_sequence (n m : ℕ) (r : ℝ) (h : ∀ k, a k > 0) : a n = a 0 * r^n
axiom geometric_mean_condition : a 4 * a 14 = 8

-- Theorem to Prove
theorem min_value_of_2a7_a11 : ∀ n : ℕ, (∀ k, a k > 0) → 2 * a 7 + a 11 ≥ 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_value_of_2a7_a11_l1736_173651


namespace NUMINAMATH_GPT_quadratic_has_root_in_interval_l1736_173617

theorem quadratic_has_root_in_interval (a b c : ℝ) (h : 2 * a + 3 * b + 6 * c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_has_root_in_interval_l1736_173617


namespace NUMINAMATH_GPT_minyoung_yoojung_flowers_l1736_173685

theorem minyoung_yoojung_flowers (m y : ℕ) 
(h1 : m = 4 * y) 
(h2 : m = 24) : 
m + y = 30 := 
by
  sorry

end NUMINAMATH_GPT_minyoung_yoojung_flowers_l1736_173685


namespace NUMINAMATH_GPT_complex_magnitude_problem_l1736_173646

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : abs z = 1)
  (hw : abs w = 2)
  (hzw : abs (z + w) = 3) :
  abs ((1 / z) + (1 / w)) = 3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_magnitude_problem_l1736_173646


namespace NUMINAMATH_GPT_harry_sandy_midpoint_l1736_173613

theorem harry_sandy_midpoint :
  ∃ (x y : ℤ), x = 9 ∧ y = -2 → ∃ (a b : ℤ), a = 1 ∧ b = 6 → ((9 + 1) / 2, (-2 + 6) / 2) = (5, 2) := 
by 
  sorry

end NUMINAMATH_GPT_harry_sandy_midpoint_l1736_173613


namespace NUMINAMATH_GPT_find_principal_amount_l1736_173618

noncomputable def principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (SI * 100) / (R * T)

theorem find_principal_amount :
  principal_amount 130 4.166666666666667 4 = 780 :=
by
  -- Sorry is used to denote that the proof is yet to be provided
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1736_173618


namespace NUMINAMATH_GPT_number_of_grade12_students_selected_l1736_173669

def total_students : ℕ := 1500
def grade10_students : ℕ := 550
def grade11_students : ℕ := 450
def total_sample_size : ℕ := 300
def grade12_students : ℕ := total_students - grade10_students - grade11_students

theorem number_of_grade12_students_selected :
    (total_sample_size * grade12_students / total_students) = 100 := by
  sorry

end NUMINAMATH_GPT_number_of_grade12_students_selected_l1736_173669


namespace NUMINAMATH_GPT_no_other_integer_solutions_l1736_173624

theorem no_other_integer_solutions :
  (∀ (x : ℤ), (x + 1) ^ 3 + (x + 2) ^ 3 + (x + 3) ^ 3 = (x + 4) ^ 3 → x = 2) := 
by sorry

end NUMINAMATH_GPT_no_other_integer_solutions_l1736_173624


namespace NUMINAMATH_GPT_acceptable_outfits_l1736_173648

-- Definitions based on the given conditions
def shirts : Nat := 8
def pants : Nat := 5
def hats : Nat := 7
def pant_colors : List String := ["red", "black", "blue", "gray", "green"]
def shirt_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]
def hat_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]

-- Axiom that ensures distinct colors for pants, shirts, and hats.
axiom distinct_colors : ∀ color ∈ pant_colors, color ∈ shirt_colors ∧ color ∈ hat_colors

-- Problem statement
theorem acceptable_outfits : 
  let total_outfits := shirts * pants * hats
  let monochrome_outfits := List.length pant_colors
  let acceptable_outfits := total_outfits - monochrome_outfits
  acceptable_outfits = 275 :=
by
  sorry

end NUMINAMATH_GPT_acceptable_outfits_l1736_173648


namespace NUMINAMATH_GPT_sum_of_integers_is_27_24_or_20_l1736_173656

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end NUMINAMATH_GPT_sum_of_integers_is_27_24_or_20_l1736_173656


namespace NUMINAMATH_GPT_speed_of_second_car_l1736_173673

theorem speed_of_second_car
  (t : ℝ)
  (distance_apart : ℝ)
  (speed_first_car : ℝ)
  (speed_second_car : ℝ)
  (h_total_distance : distance_apart = t * speed_first_car + t * speed_second_car)
  (h_time : t = 2.5)
  (h_distance_apart : distance_apart = 310)
  (h_speed_first_car : speed_first_car = 60) :
  speed_second_car = 64 := by
  sorry

end NUMINAMATH_GPT_speed_of_second_car_l1736_173673


namespace NUMINAMATH_GPT_solve_equation_l1736_173642

theorem solve_equation (x : ℝ) : x * (x-3)^2 * (5+x) = 0 ↔ (x = 0 ∨ x = 3 ∨ x = -5) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1736_173642


namespace NUMINAMATH_GPT_perpendicular_lines_l1736_173639

theorem perpendicular_lines (a : ℝ) :
  (∃ l₁ l₂ : ℝ, 2 * l₁ + l₂ + 1 = 0 ∧ l₁ + a * l₂ + 3 = 0 ∧ 2 * l₁ + 1 * l₂ + 1 * a = 0) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1736_173639


namespace NUMINAMATH_GPT_one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l1736_173692

-- Definitions from conditions
def cubic_eq (x p q : ℝ) := x^3 + p * x + q

-- Correct answers in mathematical proofs
theorem one_real_root (p q : ℝ) : 4 * p^3 + 27 * q^2 > 0 → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem multiple_coinciding_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem three_distinct_real_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 < 0 → ∃ x₁ x₂ x₃ : ℝ, 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ cubic_eq x₁ p q = 0 ∧ cubic_eq x₂ p q = 0 ∧ cubic_eq x₃ p q = 0 := sorry

theorem three_coinciding_roots_at_origin : ∃ x : ℝ, cubic_eq x 0 0 = 0 := sorry

end NUMINAMATH_GPT_one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l1736_173692


namespace NUMINAMATH_GPT_correct_average_weight_l1736_173615

noncomputable def initial_average_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def misread_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 66

theorem correct_average_weight : 
  (initial_average_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.9 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_weight_l1736_173615


namespace NUMINAMATH_GPT_sequence_formulas_range_of_k_l1736_173659

variable {a b : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {k : ℝ}

-- (1) Prove the general formulas for {a_n} and {b_n}
theorem sequence_formulas (h1 : ∀ n, a n + b n = 2 * n - 1)
  (h2 : ∀ n, S n = 2 * n^2 - n)
  (hS : ∀ n, a (n + 1) = S (n + 1) - S n)
  (hS1 : a 1 = S 1) :
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, b n = -2 * n + 2) :=
sorry

-- (2) Prove the range of k
theorem range_of_k (h3 : ∀ n, a n = k * 2^(n - 1))
  (h4 : ∀ n, b n = 2 * n - 1 - k * 2^(n - 1))
  (h5 : ∀ n, b (n + 1) < b n) :
  k > 2 :=
sorry

end NUMINAMATH_GPT_sequence_formulas_range_of_k_l1736_173659


namespace NUMINAMATH_GPT_pyramid_volume_l1736_173666

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l1736_173666


namespace NUMINAMATH_GPT_problem_statement_l1736_173658

theorem problem_statement (x y z : ℤ) (h1 : x = z - 2) (h2 : y = x + 1) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1736_173658


namespace NUMINAMATH_GPT_trapezoid_problem_l1736_173681

theorem trapezoid_problem (b h x : ℝ) 
  (h1 : x = (12500 / (x - 75)) - 75)
  (h_cond : (b + 75) / (b + 25) = 3 / 2)
  (b_solution : b = 75) :
  (⌊(x^2 / 100)⌋ : ℤ) = 181 :=
by
  -- The statement only requires us to assert the proof goal
  sorry

end NUMINAMATH_GPT_trapezoid_problem_l1736_173681


namespace NUMINAMATH_GPT_bolts_per_box_l1736_173610

def total_bolts_and_nuts_used : Nat := 113
def bolts_left_over : Nat := 3
def nuts_left_over : Nat := 6
def boxes_of_bolts : Nat := 7
def boxes_of_nuts : Nat := 3
def nuts_per_box : Nat := 15

theorem bolts_per_box :
  let total_bolts_and_nuts := total_bolts_and_nuts_used + bolts_left_over + nuts_left_over
  let total_nuts := boxes_of_nuts * nuts_per_box
  let total_bolts := total_bolts_and_nuts - total_nuts
  let bolts_per_box := total_bolts / boxes_of_bolts
  bolts_per_box = 11 := by
  sorry

end NUMINAMATH_GPT_bolts_per_box_l1736_173610


namespace NUMINAMATH_GPT_bisection_method_third_interval_l1736_173644

noncomputable def bisection_method_interval (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : (ℝ × ℝ) :=
  sorry  -- Definition of the interval using bisection method, but this is not necessary.

theorem bisection_method_third_interval (f : ℝ → ℝ) :
  (bisection_method_interval f (-2) 4 3) = (-1/2, 1) :=
sorry

end NUMINAMATH_GPT_bisection_method_third_interval_l1736_173644


namespace NUMINAMATH_GPT_shape_is_plane_l1736_173699

-- Define cylindrical coordinates
structure CylindricalCoord :=
  (r : ℝ) (theta : ℝ) (z : ℝ)

-- Define the condition
def condition (c : ℝ) (coord : CylindricalCoord) : Prop :=
  coord.z = c

-- The shape is described as a plane
def is_plane : Prop := ∀ (coord1 coord2 : CylindricalCoord), (coord1.z = coord2.z)

theorem shape_is_plane (c : ℝ) : 
  (∀ coord : CylindricalCoord, condition c coord) ↔ is_plane :=
by 
  sorry

end NUMINAMATH_GPT_shape_is_plane_l1736_173699


namespace NUMINAMATH_GPT_gridPolygon_side_longer_than_one_l1736_173635

-- Define the structure of a grid polygon
structure GridPolygon where
  area : ℕ  -- Area of the grid polygon
  perimeter : ℕ  -- Perimeter of the grid polygon
  no_holes : Prop  -- Polyon does not contain holes

-- Definition of a grid polygon with specific properties
def specificGridPolygon : GridPolygon :=
  { area := 300, perimeter := 300, no_holes := true }

-- The theorem we want to prove that ensures at least one side is longer than 1
theorem gridPolygon_side_longer_than_one (P : GridPolygon) (h_area : P.area = 300) (h_perimeter : P.perimeter = 300) (h_no_holes : P.no_holes) : ∃ side_length : ℝ, side_length > 1 :=
  by
  sorry

end NUMINAMATH_GPT_gridPolygon_side_longer_than_one_l1736_173635


namespace NUMINAMATH_GPT_c_share_correct_l1736_173600

def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def total_profit : ℕ := 5000

def total_investment : ℕ := investment_a + investment_b + investment_c
def c_ratio : ℚ := investment_c / total_investment
def c_share : ℚ := total_profit * c_ratio

theorem c_share_correct : c_share = 3000 := by
  sorry

end NUMINAMATH_GPT_c_share_correct_l1736_173600


namespace NUMINAMATH_GPT_additional_chicken_wings_l1736_173679

theorem additional_chicken_wings (friends : ℕ) (wings_per_friend : ℕ) (initial_wings : ℕ) (H1 : friends = 9) (H2 : wings_per_friend = 3) (H3 : initial_wings = 2) : 
  friends * wings_per_friend - initial_wings = 25 := by
  sorry

end NUMINAMATH_GPT_additional_chicken_wings_l1736_173679


namespace NUMINAMATH_GPT_taxi_fare_proof_l1736_173625

/-- Given equations representing the taxi fare conditions:
1. x + 7y = 16.5 (Person A's fare)
2. x + 11y = 22.5 (Person B's fare)

And using the value of the initial fare and additional charge per kilometer conditions,
prove the initial fare and additional charge and calculate the fare for a 7-kilometer ride. -/
theorem taxi_fare_proof (x y : ℝ) 
  (h1 : x + 7 * y = 16.5)
  (h2 : x + 11 * y = 22.5)
  (h3 : x = 6)
  (h4 : y = 1.5) :
  x = 6 ∧ y = 1.5 ∧ (x + y * (7 - 3)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_proof_l1736_173625


namespace NUMINAMATH_GPT_length_of_AB_l1736_173697

noncomputable def parabola_intersection (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
|x1 - x2|

theorem length_of_AB : 
  ∀ (x1 x2 y1 y2 : ℝ),
    (x1 + x2 = 6) →
    (A = (x1, y1)) →
    (B = (x2, y2)) →
    (y1^2 = 4 * x1) →
    (y2^2 = 4 * x2) →
    parabola_intersection x1 x2 y1 y2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l1736_173697


namespace NUMINAMATH_GPT_electric_energy_consumption_l1736_173609

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end NUMINAMATH_GPT_electric_energy_consumption_l1736_173609


namespace NUMINAMATH_GPT_expected_potato_yield_l1736_173632

-- Definitions based on the conditions
def steps_length : ℕ := 3
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def yield_rate : ℚ := 3 / 4

-- Calculate the dimensions in feet
def garden_length_feet : ℕ := garden_length_steps * steps_length
def garden_width_feet : ℕ := garden_width_steps * steps_length

-- Calculate the area in square feet
def garden_area_feet : ℕ := garden_length_feet * garden_width_feet

-- Calculate the expected yield in pounds
def expected_yield_pounds : ℚ := garden_area_feet * yield_rate

-- The theorem to prove the expected yield
theorem expected_potato_yield :
  expected_yield_pounds = 3037.5 := by
  sorry  -- Proof is omitted as per the instructions.

end NUMINAMATH_GPT_expected_potato_yield_l1736_173632


namespace NUMINAMATH_GPT_domain_width_p_l1736_173647

variable (f : ℝ → ℝ)
variable (h_dom_f : ∀ x, -12 ≤ x ∧ x ≤ 12 → f x = f x)

noncomputable def p (x : ℝ) : ℝ := f (x / 3)

theorem domain_width_p : (width : ℝ) = 72 :=
by
  let domain_p : Set ℝ := {x | -36 ≤ x ∧ x ≤ 36}
  have : width = 72 := sorry
  exact this

end NUMINAMATH_GPT_domain_width_p_l1736_173647


namespace NUMINAMATH_GPT_at_least_six_on_circle_l1736_173616

-- Defining the types for point and circle
variable (Point : Type)
variable (Circle : Type)

-- Assuming the existence of a well-defined predicate that checks whether points lie on the same circle
variable (lies_on_circle : Circle → Point → Prop)
variable (exists_circle : Point → Point → Point → Point → Circle)
variable (five_points_condition : ∀ (p1 p2 p3 p4 p5 : Point), 
  ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                   lies_on_circle c p3 ∧ lies_on_circle c p4)

-- Given 13 points on a plane
variables (P : List Point)
variable (length_P : P.length = 13)

-- The main theorem statement
theorem at_least_six_on_circle : 
  (∀ (P : List Point) (h : P.length = 13),
    (∀ p1 p2 p3 p4 p5 : Point, ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                               lies_on_circle c p3 ∧ lies_on_circle c p4)) →
    (∃ (c : Circle), ∃ (l : List Point), l.length ≥ 6 ∧ ∀ p ∈ l, lies_on_circle c p) :=
sorry

end NUMINAMATH_GPT_at_least_six_on_circle_l1736_173616


namespace NUMINAMATH_GPT_average_of_numbers_l1736_173684

theorem average_of_numbers (a b c d e : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) (h₄ : d = 11) (h₅ : e = 12) :
  (a + b + c + d + e) / 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_of_numbers_l1736_173684
