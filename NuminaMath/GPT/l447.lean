import Mathlib
import Mathlib.Algebra.Combinatorics.Factorial
import Mathlib.Algebra.GeomSum.Basic
import Mathlib.Algebra.Order.Rounding
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Limits.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialAlg
import Mathlib.Combinatorics.CombinatorialLibrary
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fin.FinTuple.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Polygon.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.NumberTheory.GCD
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.SimP

namespace complex_pow_sum_l447_447944

theorem complex_pow_sum (z : ℂ) (h : z + z⁻¹ = 2 * real.sqrt 2) : 
  z^8 + z^(-8) = 2 :=
by
  -- The proof is omitted
  sorry

end complex_pow_sum_l447_447944


namespace original_number_of_sides_l447_447164

theorem original_number_of_sides (sum_of_angles : ℕ) (H : (sum_of_angles = 2160)) : 
  ∃ x : ℕ, (2 * x - 2) * 180 = 2160 := 
by
  use 7
  have : (2 * 7 - 2) * 180 = 2160 := by sorry
  exact this

end original_number_of_sides_l447_447164


namespace vector_projection_of_BA_onto_BC_l447_447587

theorem vector_projection_of_BA_onto_BC
  (A B C : Type*)
  [euclidean_space A]
  [euclidean_space B]
  [euclidean_space C]
  (a b c : ℝ)
  (ha : a = 4 * real.sqrt 2)
  (hb : b = 5)
  (hcosA : real.cos A = -3/5)
  (proj_BA_BC : euclidean_space → euclidean_space → euclidean_space) :
  proj_BA_BC (BA) (BC) = real.sqrt 2 / 2 :=
sorry

end vector_projection_of_BA_onto_BC_l447_447587


namespace tim_books_l447_447733

theorem tim_books (mike_books total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) : ∃ (tim_books : ℕ), tim_books = 22 :=
by
  use total_books - mike_books
  rw [h1, h2]
  norm_num
  sorry

end tim_books_l447_447733


namespace surface_area_of_box_l447_447262

def original_cardboard_length : ℕ := 25
def original_cardboard_width : ℕ := 40
def square_side : ℕ := 8

noncomputable def original_area : ℕ := original_cardboard_length * original_cardboard_width
noncomputable def square_area : ℕ := square_side * square_side
noncomputable def total_removed_area : ℕ := 4 * square_area
noncomputable def remaining_area : ℕ := original_area - total_removed_area

theorem surface_area_of_box : remaining_area = 744 := by
  -- Definitions in the conditions
  let original_length := original_cardboard_length
  let original_width := original_cardboard_width
  let side_of_removed_square := square_side
  let area_original := original_area
  let area_one_square := square_area
  let area_removed := total_removed_area
  
  have h1 : area_original = 1000 := by
    unfold original_area
    simp [original_length, original_width]
  
  have h2 : area_one_square = 64 := by
    unfold square_area
    simp [side_of_removed_square]
  
  have h3 : area_removed = 256 := by
    unfold total_removed_area
    simp [area_one_square]

  show remaining_area = 744 from by
    unfold remaining_area
    simp [h1, h3]
    sorry


end surface_area_of_box_l447_447262


namespace four_students_three_choices_l447_447101

theorem four_students_three_choices (num_students events_per_student : ℕ) 
  (choices_per_student : ℕ) (number_of_ways : ℕ)
  (h1 : num_students = 4)
  (h2 : choices_per_student = 3) :
  number_of_ways = (choices_per_student ^ num_students) := 
by 
  have h : number_of_ways = 81 := sorry
  exact h

end four_students_three_choices_l447_447101


namespace simplify_tan_cot_60_l447_447683

theorem simplify_tan_cot_60 :
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7 / 3 :=
by
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  sorry

end simplify_tan_cot_60_l447_447683


namespace third_divisor_l447_447288

theorem third_divisor (x : ℕ) (h1 : x - 16 = 136) (h2 : ∃ y, y = x - 16) (h3 : 4 ∣ x) (h4 : 6 ∣ x) (h5 : 10 ∣ x) : 19 ∣ x := 
by
  sorry

end third_divisor_l447_447288


namespace length_AE_correct_l447_447191

-- Defining the geometry and area conditions
variables (ABCD : square) (E : point) (F : point)
  (side_length : ℝ) (area_AEF : ℝ)
  (right_angle_E : is_right_angle_at E A F)

-- Given conditions 
def ABCD_is_square : Prop := side_length = 8
def points_positions : Prop := on_line_segment E A B ∧ on_line_segment F D C
def area_AEF_condition : Prop := area_AEF = 0.3 * (side_length * side_length)
def EF_parallel_AD : Prop := F.y = D.y ∧ F.x - E.x = A.x - A.x

-- Length of AE calculation
noncomputable def length_AE : ℝ := 4.8

-- Final theorem
theorem length_AE_correct : 
  ABCD_is_square ∧ points_positions ∧ area_AEF_condition ∧ right_angle_E ∧ EF_parallel_AD → 
  AE = length_AE := 
by 
  sorry

end length_AE_correct_l447_447191


namespace probability_of_drawing_multiple_of_7_is_7_over_50_l447_447305

-- Define the set of cards numbered from 1 to 100
def cards := Finset.range 101

-- Define a predicate that checks if a number is a multiple of 7
def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0

-- Define the set of cards that are multiples of 7
def multiples_of_7 := cards.filter is_multiple_of_7

-- Define the probability of drawing a multiple of 7 card
def probability_multiple_of_7 : ℚ := multiples_of_7.card / cards.card

-- Theorem to prove the probability
theorem probability_of_drawing_multiple_of_7_is_7_over_50 :
  probability_multiple_of_7 = 7 / 50 := 
sorry

end probability_of_drawing_multiple_of_7_is_7_over_50_l447_447305


namespace seq_a_general_term_l447_447925

-- Define the sequence a_n with given initial conditions and recurrence relation
def seq_a : ℕ → ℕ
| 0     := 2
| 1     := 3
| 2     := 6
| (n+3) := (n + 3 + 1) * seq_a (n + 2) - 4 * (n + 3) * seq_a (n + 1) + (4 * (n + 3) - 8) * seq_a n

-- Theorem stating the sequence matches the given form
theorem seq_a_general_term (n : ℕ) : seq_a n = n! + 2^n :=
sorry

end seq_a_general_term_l447_447925


namespace floor_sqrt_30_squared_l447_447421

theorem floor_sqrt_30_squared : 
  ∀ (x : ℝ), (5 < x) ∧ (x < 6) ∧ (⌊x⌋ = 5) → ⌊x⌋^2 = 25 :=
by
  intro x
  intro h
  cases h with hx1 hx2
  cases hx2 with hx3 hx4
  sorry

end floor_sqrt_30_squared_l447_447421


namespace sum_sequence_l447_447040

theorem sum_sequence :
  (∑ i in finset.range 5, (100 - 20 * i) - (90 - 20 * i)) + 10 = 200 :=
by
  -- Claims will be filled here
  sorry

end sum_sequence_l447_447040


namespace seokjin_fewer_books_l447_447986

theorem seokjin_fewer_books (init_books : ℕ) (jungkook_initial : ℕ) (seokjin_initial : ℕ) (jungkook_bought : ℕ) (seokjin_bought : ℕ) :
  jungkook_initial = init_books → seokjin_initial = init_books → jungkook_bought = 18 → seokjin_bought = 11 →
  jungkook_initial + jungkook_bought - (seokjin_initial + seokjin_bought) = 7 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end seokjin_fewer_books_l447_447986


namespace tan_alpha_third_quadrant_l447_447120

variable (α : Real)
variable (h1 : 3 * sin α = -2) -- Using the condition that sin α = -2/3

-- In the third quadrant, both sine and cosine are negative.
def inThirdQuadrant (α : Real) : Prop :=
  π < α ∧ α < 3 * π / 2

theorem tan_alpha_third_quadrant (h0 : inThirdQuadrant α) (h1 : sin α = -2 / 3) : 
  tan α = 2 * Real.sqrt 5 / 5 :=
sorry

end tan_alpha_third_quadrant_l447_447120


namespace aarons_test_score_l447_447240

theorem aarons_test_score (S : list ℕ) (Aaron_score : ℕ) :
  S.length = 19 →
  (S.sum / S.length : ℕ) = 82 →
  ((S.sum + Aaron_score) / (S.length + 1) : ℕ) = 83 →
  Aaron_score = 102 :=
by
  intros h1 h2 h3
  sorry

end aarons_test_score_l447_447240


namespace select_k_plus_1_nums_divisible_by_n_l447_447249

theorem select_k_plus_1_nums_divisible_by_n (n k : ℕ) (hn : n > 0) (hk : k > 0) (nums : Fin (n + k) → ℕ) :
  ∃ (indices : Finset (Fin (n + k))), indices.card ≥ k + 1 ∧ (indices.sum (nums ∘ id)) % n = 0 :=
sorry

end select_k_plus_1_nums_divisible_by_n_l447_447249


namespace contrapositive_log_decreasing_l447_447701

theorem contrapositive_log_decreasing (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, log a x < 0 → x < 2) → (log a 2 ≥ 0 → ∃ x : ℝ, log a x ≥ log a (2 - x) ∧ x < 2) :=
  sorry

end contrapositive_log_decreasing_l447_447701


namespace largest_difference_l447_447754

theorem largest_difference : 
  ∃ (a b : ℕ), 
    (a ≥ 100) ∧ (a < 1000) ∧ 
    (b ≥ 10) ∧ (b < 100) ∧ 
    (∃ (digits_a digits_b : list ℕ), 
      digits_a.to_finset ∪ digits_b.to_finset = {0, 2, 5, 7, 9} ∧ 
      digits_a.to_finset ∩ digits_b.to_finset = ∅ ∧ 
      a = digits_a.foldl (λ acc d, acc * 10 + d) 0 ∧ 
      b = digits_b.foldl (λ acc d, acc * 10 + d) 0) ∧ 
    (a - b = 955) :=
sorry

end largest_difference_l447_447754


namespace students_in_both_competitions_l447_447355

theorem students_in_both_competitions (T A B X : ℕ) 
(hT : T = 55) (hA : A = 38) (hB : B = 42) 
(h_union : T = A + B - X) : X = 25 := 
by
  -- Using the inclusion-exclusion principle
  rw [hT, hA, hB] at h_union
  have h : 55 = 38 + 42 - X := h_union
  -- Simplifying the equation
  have h' : 55 = 80 - X := h
  rw [←nat.sub_eq_iff_eq_add] at h'
  -- Solving for X
  exact nat.sub_eq_zero_iff_eq.mpr (h_union.trans (eq.symm h'))
  sorry

end students_in_both_competitions_l447_447355


namespace MrKishoreSavings_l447_447385

noncomputable def TotalExpenses : ℕ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

noncomputable def MonthlySalary : ℕ :=
  (TotalExpenses * 10) / 9

noncomputable def Savings : ℕ :=
  (MonthlySalary * 1) / 10

theorem MrKishoreSavings :
  Savings = 2300 :=
by
  sorry

end MrKishoreSavings_l447_447385


namespace minimum_sum_l447_447285

theorem minimum_sum (a b c : ℕ) (h : a * b * c = 3006) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ 105 :=
sorry

end minimum_sum_l447_447285


namespace pyramid_volume_l447_447373

theorem pyramid_volume
  (s : ℝ) (h : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) (surface_area : ℝ)
  (h_base_area : base_area = s * s)
  (h_triangular_face_area : triangular_face_area = (1 / 3) * base_area)
  (h_surface_area : surface_area = base_area + 4 * triangular_face_area)
  (h_surface_area_value : surface_area = 768)
  (h_vol : h = 7.78) :
  (1 / 3) * base_area * h = 853.56 :=
by
  sorry

end pyramid_volume_l447_447373


namespace num_ways_to_tile_1x7_block_l447_447315

theorem num_ways_to_tile_1x7_block :
  let T := {1, 2, 3} in
  (∀ t ∈ T, t = 1 ∨ t = 2 ∨ t = 3) →
  (finset.sum (finset.powerset finset.finset (finset.filter (λ s, 
              finset.sum s = 7 ∧ ∀ x ∈ s, x ∈ T)
            finset.univ)).card) = 44 :=
by
  sorry

end num_ways_to_tile_1x7_block_l447_447315


namespace expected_points_in_E_correct_l447_447637

-- Definitions of areas and conditions
def rectangle_D : ℝ := 2 * 4 -- Area of the rectangle D
def function_y (x : ℝ) : ℝ := x^2 -- Function y = x^2

-- Integral calculation of area E under the curve y = x^2 from 0 to 2
def area_E : ℝ := ∫ x in (0 : ℝ)..2, function_y x

-- Proportion of area E to area D
def proportion_E_to_D : ℝ := area_E / rectangle_D

-- Number of points
def total_points : ℕ := 30
def expected_points_in_E : ℕ := total_points * proportion_E_to_D

theorem expected_points_in_E_correct :
  expected_points_in_E = 10 := by
  -- Detailed proofs would go here
  sorry

end expected_points_in_E_correct_l447_447637


namespace total_animals_seen_l447_447394

-- Definitions of the initial conditions
def beavers_morning : Nat := 50
def chipmunks_morning : Nat := 90
def beavers_afternoon : Nat := beavers_morning * 4
def chipmunks_afternoon : Nat := chipmunks_morning - 20

-- Statement to prove
theorem total_animals_seen : 
  let total_morning := beavers_morning + chipmunks_morning in
  let total_afternoon := beavers_afternoon + chipmunks_afternoon in
  let total_day := total_morning + total_afternoon in
  total_day = 410 :=
by
  sorry

end total_animals_seen_l447_447394


namespace cafeteria_earnings_l447_447719

def apples_initial : ℕ := 50
def oranges_initial : ℕ := 40
def apple_price : ℝ := 0.80
def orange_price : ℝ := 0.50
def apples_left : ℕ := 10
def oranges_left : ℕ := 6

theorem cafeteria_earnings : 
  let apples_sold := apples_initial - apples_left,
      oranges_sold := oranges_initial - oranges_left,
      earnings_apples := (apples_sold : ℝ) * apple_price,
      earnings_oranges := (oranges_sold : ℝ) * orange_price,
      total_earnings := earnings_apples + earnings_oranges
  in total_earnings = 49 :=
by
  sorry

end cafeteria_earnings_l447_447719


namespace log_base_property_l447_447870

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x / log a

theorem log_base_property
  (a : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (hf9 : f a 9 = 2) :
  f a (3^a) = 3 :=
by
  sorry

end log_base_property_l447_447870


namespace largest_positive_integer_n_lattice_points_l447_447084

theorem largest_positive_integer_n_lattice_points (n : ℕ) :
  (∃ (P : Fin n → ℤ × ℤ), 
   ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
   let C := (1/3 : ℚ) • (P i + P j + P k)
   in ¬ (C.1.den = 1 ∧ C.2.den = 1)) ↔ n ≤ 8 := 
sorry

end largest_positive_integer_n_lattice_points_l447_447084


namespace senior_ticket_cost_l447_447313

theorem senior_ticket_cost (adult_tickets senior_tickets total_tickets total_receipts adult_ticket_cost : ℕ)
  (htotal_tickets : total_tickets = 510)
  (htotal_receipts : total_receipts = 8748)
  (hadult_ticket_cost : adult_ticket_cost = 21)
  (hsenior_tickets : senior_tickets = 327)
  (hadult_tickets : adult_tickets = total_tickets - senior_tickets):
  ∃ (senior_ticket_cost : ℕ), senior_ticket_cost * senior_tickets + adult_ticket_cost * adult_tickets = total_receipts ∧ senior_ticket_cost = 15 := 
by {
  have h1 : adult_tickets = 183,
  { simp [hadult_tickets, hsenior_tickets, htotal_tickets], },
  have h2 : 21 * 183 = 3843,
  { norm_num, },
  have h3 : total_receipts - (21 * 183) = 4905,
  { rw [htotal_receipts, h2], norm_num, },
  have h4 : 4905 / 327 = 15,
  { norm_num, },
  use 15,
  exact ⟨by ria_norm [h4], h4⟩,
  sorry,
}

end senior_ticket_cost_l447_447313


namespace sequence_and_sum_correct_l447_447141

def x (n : ℕ) : ℕ → ℕ → ℕ
| 0 _ := 0
| 1 a := a
| 2 a b := b
| (n+1) a b := x n a b - x (n-1) a b

def S (n : ℕ) (x : ℕ → ℕ → ℕ) (a b : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, x i a b)

theorem sequence_and_sum_correct (a b : ℕ) :
  x 100 a b = -a ∧ S 100 (x 100) a b = 2 * b - a := 
by 
  sorry

end sequence_and_sum_correct_l447_447141


namespace mean_of_five_numbers_l447_447585

theorem mean_of_five_numbers (a b c d e : ℚ) (h : a + b + c + d + e = 2/3) : 
  (a + b + c + d + e) / 5 = 2 / 15 := 
by 
  -- This is where the proof would go, but we'll omit it as per instructions
  sorry

end mean_of_five_numbers_l447_447585


namespace cyclic_triples_l447_447377

theorem cyclic_triples (n : Nat) (wins losses : Fin n → Nat)
  (h1 : ∀ t, wins t = 6)
  (h2 : ∀ t, losses t = 9)
  (h3 : ∀ t u, t ≠ u → (wins t + losses t) = n - 1)
  (h4 : ∀ t u, t ≠ u → ((wins t + losses t)  + (wins u + losses u)) = n - 1)
: ∃ count, count = 320 := 
by
  use 320
  sorry

end cyclic_triples_l447_447377


namespace donuts_distribution_l447_447035

theorem donuts_distribution (kinds total min_each : ℕ) (h_kinds : kinds = 4) (h_total : total = 7) (h_min_each : min_each = 1) :
  ∃ n : ℕ, n = 20 := by
  sorry

end donuts_distribution_l447_447035


namespace exponential_function_range_l447_447132

noncomputable def exponential_function (a x : ℝ) : ℝ := a^x

theorem exponential_function_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : exponential_function a (-2) < exponential_function a (-3)) : 
  0 < a ∧ a < 1 :=
by
  sorry

end exponential_function_range_l447_447132


namespace rectangle_length_decrease_l447_447713

theorem rectangle_length_decrease (L W : ℝ) 
  (hW : W > 0) 
  (hL : L > 0) 
  (A : ℝ) 
  (hA : A = L * W)
  (new_W : ℝ)
  (hnew_W : new_W = 1.10 * W):
  ∃ new_L : ℝ, (new_L * new_W = A) ∧ (new_L = 0.909 * L) := 
by
  use 0.909 * L
  split
  -- here we would typically provide the necessary proof steps to show each part holds
  sorry

end rectangle_length_decrease_l447_447713


namespace percentage_of_girls_who_passed_l447_447354

theorem percentage_of_girls_who_passed (boys girls : ℕ) (boys_pass_percentage total_fail_percentage : ℚ) :
  boys = 50 →
  girls = 100 →
  boys_pass_percentage = 50 →
  total_fail_percentage = 56.67 →
  (let total_students := boys + girls in
   let boys_passed := boys_pass_percentage * boys / 100 in
   let total_failed := total_fail_percentage * total_students / 100 in
   let girls_failed := total_failed - (boys - boys_passed) in
   let girls_passed := girls - girls_failed in
   (girls_passed * 100 / girls) = 40) :=
begin
  intros h_boys h_girls h_boys_pass h_total_fail,
  rw [h_boys, h_girls, h_boys_pass, h_total_fail],
  let total_students := 50 + 100,
  let boys_passed := (50 * 50 : ℚ) / 100,
  let total_failed := (56.67 * 150 : ℚ) / 100,
  let girls_failed := total_failed - (50 - boys_passed),
  let girls_passed := 100 - girls_failed,
  have : (girls_passed * 100) / 100 = 40, sorry,
  exact this,
end

end percentage_of_girls_who_passed_l447_447354


namespace divisible_by_42_l447_447250

theorem divisible_by_42 (n : ℕ) : 42 ∣ (n^3 * (n^6 - 1)) :=
sorry

end divisible_by_42_l447_447250


namespace emily_vacation_duration_l447_447833

theorem emily_vacation_duration : 
  ∀ (dogs : ℕ) (food_per_dog_daily : ℕ) (total_food_kg : ℕ), 
  dogs = 4 → 
  food_per_dog_daily = 250 → 
  total_food_kg = 14 → 
  (total_food_kg * 1000) / (dogs * food_per_dog_daily) = 14 := 
by 
  intros dogs food_per_dog_daily total_food_kg h_dogs h_food_per_dog h_total_food
  rw [h_dogs, h_food_per_dog, h_total_food]
  -- Proceed to prove the statements using the given conditions
  -- This part will be filled in with the actual proof in practical use-case
  sorry

end emily_vacation_duration_l447_447833


namespace part1_distance_part2_slopes_l447_447460

-- Definitions of the ellipse and circle
def ellipse (x y : ℝ) : Prop := (x^2 / 4 + y^2 / 3 = 1)
def circle (x y : ℝ) : Prop := (x^2 + y^2 = 4)

-- Definitions of the line and slopes k, k1, k2
def line (x y : ℝ) (m : ℝ) : Prop := (y = (1 / 2) * x + m)

-- Points A and B for left and right vertices of the ellipse
def point_A := (0, -2) -- Using coordinates (0, -2) as left vertex of the ellipse
def point_B := (0, 2)  -- Using coordinates (0, 2) as right vertex of the ellipse

-- Distance from origin to line l
noncomputable def distance_from_origin_to_line (m : ℝ) : ℝ :=
  |m| / sqrt ((1 / 2)^2 + 1)

-- Slopes k1 and k2
def slope_k1 (x1 y1 : ℝ) : ℝ := y1 / (x1 + 2)
def slope_k2 (x2 y2 : ℝ) : ℝ := y2 / (x2 - 2)

theorem part1_distance (m : ℝ) : abs m = 2 → distance_from_origin_to_line m = 4 * sqrt 5 / 5 :=
sorry

theorem part2_slopes (M N : ℝ × ℝ) (x1 y1 x2 y2 m : ℝ) :
  ellipse x1 y1 ∧ ellipse x2 y2 →
  circle x1 y1 ∧ circle x2 y2 →
  line x1 y1 m ∧ line x2 y2 m →
  m^2 = 4 * (1 / 2)^2 + 3 →
  slope_k1 x1 y1 * slope_k2 x2 y2 = -3 :=
sorry

end part1_distance_part2_slopes_l447_447460


namespace find_parallel_slope_l447_447567

theorem find_parallel_slope (a : ℝ) :
  (∀ x y : ℝ, (a * x - y + 1 = 0) → (2 * x + 4 * y - 1 = 0) → (-4 * a = 1)) → a = -1 / 2 :=
by {
  intro h,
  have h_4a_eq_1 : 4 * a = -1, from sorry,
  exact eq_neg_of_eq_neg sorry,
}

end find_parallel_slope_l447_447567


namespace eq_neg2_multi_l447_447753

theorem eq_neg2_multi {m n : ℝ} (h : m = n) : -2 * m = -2 * n :=
by sorry

end eq_neg2_multi_l447_447753


namespace tan_angle_A_in_right_triangle_l447_447839

open Real

theorem tan_angle_A_in_right_triangle
  (A B C : Point) 
  (h_right : is_right_triangle A B C)
  (h_hypotenuse : segment_length B C = 41)
  (h_leg : segment_length A B = 40) :
  tan (angle_BAC A B C) = 9 / 40 :=
by
  sorry

end tan_angle_A_in_right_triangle_l447_447839


namespace integer_solution_count_l447_447281

theorem integer_solution_count (x : ℤ) : (12 * x - 1) * (6 * x - 1) * (4 * x - 1) * (3 * x - 1) = 330 ↔ x = 1 :=
by
  sorry

end integer_solution_count_l447_447281


namespace problem_statement_l447_447517

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447517


namespace union_A_B_intersection_A_CI_B_l447_447143

-- Define the sets
def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 5, 6, 7}

-- Define the complement of B in the universal set I
def C_I (I : Set ℕ) (B : Set ℕ) : Set ℕ := {x ∈ I | x ∉ B}

-- The theorem for the union of A and B
theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6, 7} := sorry

-- The theorem for the intersection of A and the complement of B in I
theorem intersection_A_CI_B : A ∩ (C_I I B) = {1, 2, 4} := sorry

end union_A_B_intersection_A_CI_B_l447_447143


namespace calculate_area_correct_l447_447438

-- Define the side length of the square
def side_length : ℝ := 5

-- Define the rotation angles in degrees
def rotation_angles : List ℝ := [0, 30, 45, 60]

-- Define the area calculation function (to be implemented)
def calculate_overlap_area (s : ℝ) (angles : List ℝ) : ℝ := sorry

-- Define the proof that the calculated area is equal to 123.475
theorem calculate_area_correct : calculate_overlap_area side_length rotation_angles = 123.475 :=
by
  sorry

end calculate_area_correct_l447_447438


namespace cartesian_eq_of_polar_eq_length_of_chord_AB_l447_447195

theorem cartesian_eq_of_polar_eq (ρ θ : ℝ) 
    (h : ρ * sin θ * sin θ = 4 * cos θ) :
    ∃ x y : ℝ, y^2 = 4 * x :=
sorry

theorem length_of_chord_AB (t₁ t₂ : ℝ) :
    ((2 + (1 / 2) * t₁, (sqrt 3 / 2) * t₁),
     (2 + (1 / 2) * t₂, (sqrt 3 / 2) * t₂)) ∈ 
    { p : ℝ × ℝ | p.2^2 = 4 * p.1 } →
    abs (t₁ - t₂) = (8 * sqrt 7) / 3 :=
sorry

end cartesian_eq_of_polar_eq_length_of_chord_AB_l447_447195


namespace min_value_x_plus_y_l447_447897

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : log 2 x + log 2 y = 1) : 
  x + y ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_y_l447_447897


namespace loss_percentage_initially_l447_447378

theorem loss_percentage_initially 
  (SP : ℝ) 
  (CP : ℝ := 400) 
  (h1 : SP + 100 = 1.05 * CP) : 
  (1 - SP / CP) * 100 = 20 := 
by 
  sorry

end loss_percentage_initially_l447_447378


namespace age_problem_l447_447588

open Nat

theorem age_problem :
  ∃ x : ℕ, (B_age = 37) ∧ (A_age = B_age + 7) ∧ (B_age - x = 1/2 * (A_age + 10)) ∧ (x = 10) :=
by
  let B_age : ℕ := 37
  let A_age : ℕ := 44 -- derived from A_age = B_age + 7
  let x : ℕ := 10
  have h1 : A_age = B_age + 7 := by linarith
  have h2 : B_age - x = (A_age + 10) / 2 := by linarith
  use x
  refine ⟨rfl, h1, h2, rfl⟩
  sorry

end age_problem_l447_447588


namespace gain_percentage_l447_447375

variables (C S : ℝ) (hC : C > 0)
variables (hS : S > 0)

def cost_price := 25 * C
def selling_price := 25 * S
def gain := 10 * S 

theorem gain_percentage (h_eq : 25 * S = 25 * C + 10 * S):
  (S = C) → 
  ((gain / cost_price) * 100 = 40) :=
by
  sorry

end gain_percentage_l447_447375


namespace infimum_of_function_l447_447436

open Real

-- Definitions given in the conditions:
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (1 - x) = f (1 + x)
def function_on_interval (f : ℝ → ℝ) := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = -3 * x ^ 2 + 2

-- Proof problem statement:
theorem infimum_of_function (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_periodic : periodic_function f) 
  (h_interval : function_on_interval f) : 
  ∃ M : ℝ, (∀ x : ℝ, f x ≥ M) ∧ M = -1 :=
by
  sorry

end infimum_of_function_l447_447436


namespace base_10_to_base_3_l447_447821

theorem base_10_to_base_3 : 
  ∀ (n : ℕ), n = 435 → (n : ℕ) = 1 * 3^5 + 2 * 3^4 + 1 * 3^3 + 1 * 3^1 + 0 * 3^0 :=
by 
  intros n h,
  sorry

end base_10_to_base_3_l447_447821


namespace sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l447_447881

-- (1)
theorem sqrt_S_n_arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d) (h3 : S n = (n * (2 * a 1 + (n - 1) * (2 : ℝ))) / 2) :
  ∃ d, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d :=
by sorry

-- (2)
theorem seq_sqrt_S_n_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ) :
  (∃ d, ∀ n, S n / 2 = n * (a1 + (n - 1) * d)) ↔ (∀ n, S n = a1 * n^2) :=
by sorry

end sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l447_447881


namespace final_value_l447_447550

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447550


namespace expression_value_l447_447536

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447536


namespace max_angle_position_l447_447302

-- Definitions for points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

-- Definitions for points A and B on the X-axis
def A (a : ℝ) : Point := { x := -a, y := 0 }
def B (a : ℝ) : Point := { x := a, y := 0 }

-- Definition for point C moving along the line y = 10 - x
def moves_along_line (C : Point) : Prop :=
  C.y = 10 - C.x

-- Definition for calculating the angle ACB (gamma)
def angle_ACB (A B C : Point) : ℝ := sorry -- The detailed function to calculate angle is omitted for brevity

-- Main statement to prove
theorem max_angle_position (a : ℝ) (C : Point) (ha : 0 ≤ a ∧ a ≤ 10) (hC : moves_along_line C) :
  (C = { x := 4, y := 6 } ∨ C = { x := 16, y := -6 }) ↔ (∀ C', moves_along_line C' → (angle_ACB (A a) (B a) C') ≤ angle_ACB (A a) (B a) C) :=
sorry

end max_angle_position_l447_447302


namespace exponent_calculation_l447_447806

theorem exponent_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end exponent_calculation_l447_447806


namespace find_bettys_balance_l447_447866

-- Define the conditions as hypotheses
def balance_in_bettys_account (B : ℕ) : Prop :=
  -- Gina has two accounts with a combined balance equal to $1,728
  (2 * (B / 4)) = 1728

-- State the theorem to be proven
theorem find_bettys_balance (B : ℕ) (h : balance_in_bettys_account B) : B = 3456 :=
by
  -- The proof is provided here as a "sorry"
  sorry

end find_bettys_balance_l447_447866


namespace quadratic_roots_equal_k_value_l447_447166

theorem quadratic_roots_equal_k_value (k : ℝ) : 
  let a := k;
      b := -3;
      c := 1 in 
  (b^2 - 4 * a * c = 0) → k = 9 / 4 :=
by {
  intro h,
  sorry
}

end quadratic_roots_equal_k_value_l447_447166


namespace sum_of_fractions_l447_447521

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447521


namespace range_of_m_l447_447449

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_m :
  ∃ m : ℝ, (A : ℝ × ℝ) = (1, m) ∧ 
           (∀ t : ℝ, t is_tangent_to_curve y = f(x)) ↔ 
           (-3 < m ∧ m < -2) := by
  sorry

end range_of_m_l447_447449


namespace final_value_l447_447544

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447544


namespace combined_resistance_parallel_l447_447603

theorem combined_resistance_parallel (R1 R2 R3 : ℝ) (r : ℝ) (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6) :
  (1 / r) = (1 / R1) + (1 / R2) + (1 / R3) → r = 15 / 13 :=
by
  sorry

end combined_resistance_parallel_l447_447603


namespace container_capacity_l447_447357

theorem container_capacity (C : ℝ) (h1 : 0.30 * C + 36 = 0.75 * C) : C = 80 :=
by
  sorry

end container_capacity_l447_447357


namespace integer_values_b_for_three_integer_solutions_l447_447724

theorem integer_values_b_for_three_integer_solutions (b : ℤ) :
  ¬ ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 + b * x1 + 5 ≤ 0) ∧
                     (x2^2 + b * x2 + 5 ≤ 0) ∧ (x3^2 + b * x3 + 5 ≤ 0) ∧
                     (∀ x : ℤ, x1 < x ∧ x < x3 → x^2 + b * x + 5 > 0) :=
by
  sorry

end integer_values_b_for_three_integer_solutions_l447_447724


namespace f_is_special_l447_447215

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ a b : ℝ, f(a + b) + f(a - b) = 3 * f(a) + f(b)
axiom initial_value : f(1) = 1

theorem f_is_special : ∀ x : ℝ, f(x) = if x = 1 then 1 else 0 :=
by
  sorry

end f_is_special_l447_447215


namespace spherical_to_rectangular_coords_l447_447410

noncomputable def sphericalToRectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * (Real.sin phi) * (Real.cos theta), 
   rho * (Real.sin phi) * (Real.sin theta), 
   rho * (Real.cos phi))

theorem spherical_to_rectangular_coords :
  sphericalToRectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coords_l447_447410


namespace cooking_time_l447_447348

theorem cooking_time (total_potatoes cooked_potatoes potato_time : ℕ) 
    (h1 : total_potatoes = 15) 
    (h2 : cooked_potatoes = 6) 
    (h3 : potato_time = 8) : 
    total_potatoes - cooked_potatoes * potato_time = 72 :=
by
    sorry

end cooking_time_l447_447348


namespace hyperbola_vertices_distance_l447_447074

/--
For the hyperbola given by the equation
(x^2 / 121) - (y^2 / 49) = 1,
the distance between its vertices is 22.
-/
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ,
  (x^2 / 121) - (y^2 / 49) = 1 →
  ∃ a : ℝ, a = 11 ∧ 2 * a = 22 :=
by
  intros x y h
  use 11
  split
  · refl
  · norm_num

end hyperbola_vertices_distance_l447_447074


namespace dot_product_ps_l447_447999

variable {V : Type*} [inner_product_space ℝ V]
variables (p q r s : V)
variables (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hr : ‖r‖ = 1) (hs : ‖s‖ = 1)
variables (h1 : inner p q = -1/7) (h2 : inner p r = -1/7) (h3 : inner q r = -1/7)
variables (h4 : inner q s = -1/7) (h5 : inner r s = -1/7)

theorem dot_product_ps : inner p s = -19/21 :=
  sorry

end dot_product_ps_l447_447999


namespace amar_distance_l447_447388

theorem amar_distance (d_car : ℝ) (d_amar_time : ℝ) (d_car_time : ℝ) : 
  (d_amar_time / d_car_time) = (15 / 40) → 
  d_amar_time = 712.5 → 
  d_car = 1.9 → 
  amar_covers d_car_distance = 712.5 :=
by
  sorry

end amar_distance_l447_447388


namespace floor_sum_A_l447_447862

def int_part (x : ℝ) : ℤ := int.floor x
def frac_part (x : ℝ) : ℝ := x - int_part x

def set_A : set ℝ := {x | frac_part x = (x + (int_part x : ℝ) + int_part (x + 1/2))/20}

def sum_A : ℝ := ∑ x in set.to_finset (set.univ.filter set_A), id x

theorem floor_sum_A : int.floor sum_A = 11 := sorry

end floor_sum_A_l447_447862


namespace geometry_proof_l447_447796

variables {A B C D E F M N H O : Type*}

/-- Given: 
  Triangle ABC with O as the circumcenter,
  Altitudes AD, BE, and CF intersecting at H,
  Line segments ED intersecting AB at M,
  and FD intersecting AC at N.

  Prove: 
  1. OB ⊥ DF and OC ⊥ DE
  2. OH ⊥ MN
-/

theorem geometry_proof (O A B C D E F M N H : Type*)
  [Circumcenter O A B C] 
  [AltitudeIntersect H A D]
  [AltitudeIntersect H B E]
  [AltitudeIntersect H C F]
  [LineIntersect ED AB M]
  [LineIntersect FD AC N] : 
  ⊥_line (ob DF) ∧ ⊥_line (oc DE) ∧ ⊥_line (oh MN) :=
sorry

end geometry_proof_l447_447796


namespace volume_of_tetrahedron_CDEA_l447_447600

open_locale real

-- Definitions based on the conditions in the problem
def square_side_length := 2
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (2, 0)
def point_D : ℝ × ℝ := (0, 2)
def point_C : ℝ × ℝ := (2, 2)
def midpoint_E : ℝ × ℝ := (1, 0)

-- The final statement to be proven
theorem volume_of_tetrahedron_CDEA :
  let side_length := square_side_length,
      A := point_A,
      B := point_B,
      C := point_C,
      D := point_D,
      E := midpoint_E in
  volume_of_tetrahedron A C D E = sqrt 3 / 3 :=
sorry

end volume_of_tetrahedron_CDEA_l447_447600


namespace expression_equals_l447_447044

theorem expression_equals :
  (Real.pi - 3.14)^0 + | - Real.sqrt 3 | - (1 / 2)^(-1 : ℤ) - Real.sin (Real.pi / 3) = -1 + Real.sqrt 3 / 2 :=
by 
  sorry

end expression_equals_l447_447044


namespace find_a_b_monotonicity_f_inequality_f_l447_447916

variable {a b x t : ℝ}

-- Define the function f(x):
def f (x : ℝ) : ℝ := (x + b) / (a * x^2 + 1)

-- Conditions:
-- 1. f is an odd function
axiom odd_f : ∀ x, f (-x) = -f (x)

-- 2. f (1 / 2) = 2 / 5
axiom f_half : f (1 / 2) = 2 / 5

-- Questions:
-- 1. Prove a = 1 and b = 0
theorem find_a_b : a = 1 ∧ b = 0 :=
  sorry

-- 2. Prove that f(x) = x / (1 + x^2) is monotonically increasing on (-1, 1)
def f_mono (x : ℝ) := x / (1 + x^2)

theorem monotonicity_f : ∀ x1 x2, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f_mono (x1) < f_mono (x2) :=
  sorry

-- 3. Inequality f(t + 1) + f(2t) < 0 for -1/2 < t < -1/3
theorem inequality_f : ∀ t, -1 / 2 < t ∧ t < -1 / 3 → f (t + 1) + f (2t) < 0 :=
  sorry

end find_a_b_monotonicity_f_inequality_f_l447_447916


namespace find_sum_s_u_l447_447307

theorem find_sum_s_u (p r s u : ℝ) (q t : ℝ) 
  (h_q : q = 5) 
  (h_t : t = -p - r) 
  (h_sum_imaginary : q + s + u = 4) :
  s + u = -1 := 
sorry

end find_sum_s_u_l447_447307


namespace butterfat_mixture_problem_l447_447149

/-- How many gallons of 10% butterfat milk must be added to 
8 gallons of 50% butterfat milk to obtain milk that is 20% butterfat? --/
theorem butterfat_mixture_problem (x : ℝ) :
  (4 + 0.1 * x) / (8 + x) = 0.20 ↔ x = 24 :=
begin
  sorry,
end

end butterfat_mixture_problem_l447_447149


namespace sum_of_two_numbers_is_147_l447_447849

theorem sum_of_two_numbers_is_147 (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) :
  A + B = 147 :=
by
  sorry

end sum_of_two_numbers_is_147_l447_447849


namespace s_one_eq_sixteen_l447_447646

def t (x : ℝ) : ℝ := 3 * x - 8
def s (y : ℝ) : ℝ := (λ x, x^2 + 3 * x - 2) ((3 * (y + 8 / 3)) / 3) -- Rearranged to match usage of t(x)

theorem s_one_eq_sixteen : s 1 = 16 :=
by 
-- This is where the proof will go
sorry

end s_one_eq_sixteen_l447_447646


namespace area_of_circle_l447_447686

theorem area_of_circle (C : ℝ) (hC : C = 30 * Real.pi) : ∃ k : ℝ, (Real.pi * (C / (2 * Real.pi))^2 = k * Real.pi) ∧ k = 225 :=
by
  sorry

end area_of_circle_l447_447686


namespace total_course_selection_schemes_l447_447363

theorem total_course_selection_schemes : 
    let total_courses := 8 in 
    let pe_courses := 4 in
    let art_courses := 4 in
    (∃ k, k = 2 ∨ k = 3) → 
    (∀ k, k = 2 → ∃ n1 n2, n1 = 1 ∧ n2 = 1 ∧ choose pe_courses n1 * choose art_courses n2 = 4 * 4) →
    (∀ k, k = 3 → 
        (∃ n1 n2, n1 = 2 ∧ n2 = 1 ∧ choose pe_courses n1 * choose art_courses n2 = 6 * 4) ∧ 
        (∃ n1 n2, n1 = 1 ∧ n2 = 2 ∧ choose pe_courses n1 * choose art_courses n2 = 4 * 6)) →
    (16 + 48 = 64) :=
begin
    sorry
end

end total_course_selection_schemes_l447_447363


namespace largest_sum_distinct_factors_l447_447593

theorem largest_sum_distinct_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (h4 : A * B * C = 2023) : A + B + C = 297 :=
sorry

end largest_sum_distinct_factors_l447_447593


namespace question1_question2_question3_l447_447254

-- Definition of degrees
def degree (p : Polynomial ℤ) : ℕ := p.natDegree

-- Definitions of proper and improper expressions
def is_proper_exp (num denom : Polynomial ℤ) : Prop :=
  degree num < degree denom

def is_improper_exp (num denom : Polynomial ℤ) : Prop :=
  degree num ≥ degree denom

-- Question 1: Prove that the expression (2/x) is a proper expression
theorem question1 : is_proper_exp (Polynomial.C 2) (Polynomial.X) :=
by sorry

-- Question 2: Convert the improper expression (x^2 - 1)/(x + 2) into its mixed form
theorem question2 : 
  (Polynomial.X ^ 2 - 1).divModBy (Polynomial.X + 2) 
  = (Polynomial.X - 2, 3) :=
by sorry

-- Question 3: Find all integer x such that (2x-1)/(x+1) is an integer
theorem question3 : 
  ∀ x : ℤ, (∃ k : ℤ, 2 * x - 1 = k * (x + 1)) ↔ x ∈ {0, -2, 2, -4} :=
by sorry

end question1_question2_question3_l447_447254


namespace giant_lollipop_calories_l447_447234

-- Definitions based on the conditions
def sugar_per_chocolate_bar := 10
def chocolate_bars_bought := 14
def sugar_in_giant_lollipop := 37
def total_sugar := 177
def calories_per_gram_of_sugar := 4

-- Prove that the number of calories in the giant lollipop is 148 given the conditions
theorem giant_lollipop_calories : (sugar_in_giant_lollipop * calories_per_gram_of_sugar) = 148 := by
  sorry

end giant_lollipop_calories_l447_447234


namespace slope_AB_l447_447710

-- Points A and B are defined
def A : ℝ × ℝ := (0, real.sqrt 3)
def B : ℝ × ℝ := (3, 0)

-- Definition of the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- Stating the theorem
theorem slope_AB : slope A B = -real.sqrt 3 / 3 := 
  sorry

end slope_AB_l447_447710


namespace sum_of_sequence_l447_447413

noncomputable theory

def complex_sum : ℂ := 
  (∑ k in finset.range 2015, (complex.I) ^ k)

theorem sum_of_sequence : complex_sum = complex.I :=
by
  sorry

end sum_of_sequence_l447_447413


namespace AM_GM_l447_447767

noncomputable def AM_GM_inequality (n : ℕ) (x : Fin n → ℝ) : Prop :=
∀ i, 0 < x i → 
  (∑ i, x i) / n ≥ (∏ i, x i) ^ (1 / n)

theorem AM_GM (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) : 
  AM_GM_inequality n x :=
by
sorry

end AM_GM_l447_447767


namespace correct_exp_identity_l447_447330

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end correct_exp_identity_l447_447330


namespace perpendicular_planes_l447_447997

/-- Two planes α and β are given, and two lines m and n lying outside those planes. 
    We have the following premises:
    1. m is perpendicular to n
    3. n is perpendicular to β
    4. m is perpendicular to α
    We need to show:
    2. α is perpendicular to β
-/
theorem perpendicular_planes (α β : Plane) (m n : Line) 
  (h₁ : Perpendicular m n) 
  (h₃ : Perpendicular n β) 
  (h₄ : Perpendicular m α) : 
  Perpendicular α β := 
sorry

end perpendicular_planes_l447_447997


namespace unique_triangle_with_consecutive_sides_and_angle_condition_l447_447675

theorem unique_triangle_with_consecutive_sides_and_angle_condition
    (a b c : ℕ) (A B C : ℝ) (h1 : a < b ∧ b < c)
    (h2 : b = a + 1 ∧ c = a + 2)
    (h3 : C = 2 * B)
    (h4 : ∀ x y z : ℕ, x < y ∧ y < z → y = x + 1 ∧ z = x + 2 → 2 * B = C)
    : ∃! (a b c : ℕ) (A B C : ℝ), (a < b ∧ b < c) ∧ (b = a + 1 ∧ c = a + 2) ∧ (C = 2 * B) :=
  sorry

end unique_triangle_with_consecutive_sides_and_angle_condition_l447_447675


namespace emma_in_middle_car_l447_447093

noncomputable def persons : Type :=
  { Allen, Brian, Chris, Diana, Emma }

variables {position : persons → fin 5}
  (h1 : position Allen = 1)
  (h2 : position Diana = 0)
  (h3 : ∃ k : fin 4, position Brian = k ∧ position Chris = k + 1)
  (h4 : 2 ≤ |position Emma - position Brian|)

theorem emma_in_middle_car :
  position Emma = 2 :=
sorry

end emma_in_middle_car_l447_447093


namespace star_value_l447_447869

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end star_value_l447_447869


namespace expected_value_of_shorter_gentlemen_correct_l447_447034
noncomputable def expected_value_of_shorter_gentlemen (n : ℕ) : ℝ :=
  ∑ j in Finset.range n, (j : ℝ) / n

theorem expected_value_of_shorter_gentlemen_correct (n : ℕ) :
  expected_value_of_shorter_gentlemen (n + 1) = n / 2 :=
by
  sorry

end expected_value_of_shorter_gentlemen_correct_l447_447034


namespace min_max_solution_l447_447052

noncomputable def f (x y : ℝ) : ℝ := abs (x^2 - 2*x*y)

noncomputable def max_f (y : ℝ) : ℝ := Sup { t | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ t = f x y }

noncomputable def min_max_f : ℝ := Inf { t | ∃ y : ℝ, t = max_f y }

theorem min_max_solution : min_max_f = 4 := 
sorry

end min_max_solution_l447_447052


namespace reach_2001_from_22_l447_447614

-- Define the allowed operation as a function.
def can_reach (n : ℕ) (target : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = n ∧ (can_reach a target ∨ can_reach b target)

-- Base case for when n equals the target.
theorem reach_2001_from_22 : can_reach 22 2001 :=
by {
  sorry
}

end reach_2001_from_22_l447_447614


namespace quadratic_distinct_roots_example_l447_447105

theorem quadratic_distinct_roots_example {b c : ℝ} (hb : b = 1) (hc : c = 0) :
    (b^2 - 4 * c) > 0 := by
  sorry

end quadratic_distinct_roots_example_l447_447105


namespace second_train_catches_first_l447_447006

-- Define the starting times and speeds
def t1_start_time := 14 -- 2:00 pm in 24-hour format
def t1_speed := 70 -- km/h
def t2_start_time := 15 -- 3:00 pm in 24-hour format
def t2_speed := 80 -- km/h

-- Define the time at which the second train catches the first train
def catch_time := 22 -- 10:00 pm in 24-hour format

theorem second_train_catches_first :
  ∃ t : ℕ, t = catch_time ∧
    t1_speed * ((t - t1_start_time) + 1) = t2_speed * (t - t2_start_time) := by
  sorry

end second_train_catches_first_l447_447006


namespace functional_equation_solution_l447_447826

noncomputable def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x) * f(y) + f(x + y) = x * y

theorem functional_equation_solution (f : ℝ → ℝ) :
  satisfies_functional_equation f ↔ (f = λ x, x - 1) ∨ (f = λ x, -x - 1) :=
by
  sorry

end functional_equation_solution_l447_447826


namespace parallel_vectors_l447_447930

variable (y : ℝ)

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

theorem parallel_vectors (h : (-1 * y - 3 * 2) = 0) : y = -6 :=
by
  sorry

end parallel_vectors_l447_447930


namespace zero_point_interval_l447_447714

def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_point_interval (h_cont : Continuous f) (h_decr : ∀ x y, x < y → f x > f y)
  (h_f3 : f 3 > 0) (h_f4 : f 4 < 0) : ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end zero_point_interval_l447_447714


namespace parabola_midpoint_locus_l447_447212

theorem parabola_midpoint_locus:
  let V1 := (0 : ℝ, 0 : ℝ)
  let F1 := (0 : ℝ, 1 / 8 : ℝ)
  (a b : ℝ) (A := (a, 2 * a^2) : ℝ × ℝ) (B := (b, 2 * b^2) : ℝ × ℝ)
  (M := ((a + b) / 2, 4 * ((a + b) / 2)^2 - 1 / 2) : ℝ × ℝ)
  (ab : ℝ := -1 / 2)
  (Q_eq : ∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.snd = 4 * p.fst^2 + 1 / 2})
  (V2 := (0 : ℝ, 1/2 : ℝ))
  (F2 := (0 : ℝ, 9/16 : ℝ))
  (F1F2 := real.sqrt ((0 - 0)^2 + ((9/16) - (1/8))^2))
  (V1V2 := real.sqrt ((0 - 0)^2 + (1 / 2)^2))
  (ratio := F1F2 / V1V2)
  -1 / (a * b) = -sqrt 3 ∧ (Q_eq M) ∧ real.sqrt ((0 - 0)^2 + ((9/16) - (1/8))^2) / real.sqrt ((0 - 0)^2 + (1 / 2)^2) = 5 / 8 :=
begin
  sorry
end

end parabola_midpoint_locus_l447_447212


namespace arrangement_count_l447_447023

def basil_tomato_arrangements : ℕ :=
  let basil_positions := 4
  let total_positions := 6 -- 4 basil plants & 2 positions for tomato groups
  let choose_2_slots := Nat.choose total_positions 2
  let permutations := 2! * 2! -- permutations of two groups of 2 tomato plants
  choose_2_slots * permutations

theorem arrangement_count (basil tomato : Type) [Fintype basil] [Fintype tomato] [DecidableEq basil] [DecidableEq tomato]
  (h_basil : Fintype.card basil = 4) (h_tomato : Fintype.card tomato = 4) :
  basil_tomato_arrangements = 40 :=
by
  have : Fintype.card (Finset.univ : Finset basil) = 4 := h_basil
  have : Fintype.card (Finset.univ : Finset tomato) = 4 := h_tomato
  exact sorry

end arrangement_count_l447_447023


namespace Lucy_speed_correct_l447_447835

variables (Eugene_speed : ℝ) (Carlos_ratio : ℝ) (Lucy_ratio : ℝ)

def EugeneCycles : Prop := Eugene_speed = 5
def CarlosCycles : Prop := Carlos_ratio = 4/5
def LucyCycles : Prop := Lucy_ratio = 6/7

theorem Lucy_speed_correct 
  (h1 : EugeneCycles Eugene_speed)
  (h2 : CarlosCycles Carlos_ratio)
  (h3 : LucyCycles Lucy_ratio) :
  let Carlos_speed := Carlos_ratio * Eugene_speed in
  let Lucy_speed := Lucy_ratio * Carlos_speed in
  Lucy_speed = 24 / 7 :=
by
  sorry

end Lucy_speed_correct_l447_447835


namespace trajectory_equation_l447_447453

theorem trajectory_equation (x y : ℝ) (h₁ : x^2 + y^2 = 1) (h₂ : 
  ∃ (P : ℝ × ℝ), P = ⟨x, y⟩ ∧ (∀ (M N : ℝ × ℝ), M ≠ N ∧ ∀ (θ : ℝ), θ ∈ Icc 0 2 * π → 
  angle P M N = π / 2)) : x^2 + y^2 = 2 := 
sorry

end trajectory_equation_l447_447453


namespace math_problem_l447_447527

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447527


namespace part_I_part_II_l447_447963

noncomputable def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
  (-3 / 5 * t + 2, 4 / 5 * t)

constant a : ℝ

def polar_eq_circle (θ : ℝ) : ℝ :=
  a * Real.sin θ

theorem part_I (a : ℝ) (h_a : a = 2) :
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 1)) ∧
  (∀ t : ℝ, parametric_eq_line t = (x, y) → 4 * x + 3 * y - 8 = 0) :=
  sorry

theorem part_II (h : ∀ t : ℝ, | 2 - (3 * -3 / 5 * t + 3 * 2 / 5) | / √(1 + (4 / 5)^2) = √3 * a / 2 → 5 * |a| = 2 * |3 * a - 16|) :
  a = 32 ∨ a = (32 / 11) :=
  sorry

end part_I_part_II_l447_447963


namespace length_QS_l447_447964

-- Definitions of all conditions
variables (P Q R S : Type) [real_vector_space P]
variables (PQ PR QR QS : ℝ)

-- Given conditions
axiom area_PQR : area_triangle P Q R = 30
axiom right_angle_at_Q : is_right_angle ∠ P Q R
axiom point_S_on_PR : on_line S P R
axiom QS_perpendicular_PR : is_perpendicular QS PR
axiom PQ_length : PQ = 5

-- Proof statement
theorem length_QS : QS = 60 / 13 :=
by sorry

end length_QS_l447_447964


namespace sqrt_meaningful_value_x_l447_447157

theorem sqrt_meaningful_value_x (x : ℝ) (h : x-1 ≥ 0) : x = 2 :=
by
  sorry

end sqrt_meaningful_value_x_l447_447157


namespace digit_B_value_l447_447611

theorem digit_B_value : ∃ B : ℕ, B < 10 ∧ (B * 10 + 4) * (8 * 10 + B) = 7008 ∧ B = 7 :=
by
  have cond : ∀ B : ℕ, B < 10 ∧ ((B * 10 + 4) * (8 * 10 + B) = 7008 → B = 7) :=
    λ B h, sorry
  exact ⟨7, by norm_num, by norm_num, cond 7⟩

end digit_B_value_l447_447611


namespace prob_triangle_includes_G_l447_447965

-- Definitions based on conditions in the problem
def total_triangles : ℕ := 6
def triangles_including_G : ℕ := 4

-- The theorem statement proving the probability
theorem prob_triangle_includes_G : (triangles_including_G : ℚ) / total_triangles = 2 / 3 :=
by
  sorry

end prob_triangle_includes_G_l447_447965


namespace cattle_population_reaches_450_in_5_years_l447_447572

theorem cattle_population_reaches_450_in_5_years :
  ∃ (n : ℕ), (C₀ : ℝ) → (C₀ = 200) → (∀ k, C (k + 1) = (3 / 2) * C k) → C n = 450 ∧ n = 5 :=
by sorry

end cattle_population_reaches_450_in_5_years_l447_447572


namespace initial_amount_simple_interest_l447_447799

theorem initial_amount_simple_interest 
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1125)
  (hR : R = 0.10)
  (hT : T = 5) :
  A = P * (1 + R * T) → P = 750 := 
by
  sorry

end initial_amount_simple_interest_l447_447799


namespace octahedron_net_folding_l447_447689

theorem octahedron_net_folding (net : OctahedronNet) (x : Segment) (E : Segment) :
  net.is_valid_net ∧ x ∈ net ∧ E ∈ net ∧ net.folded_segment_coincides x E → 
  net.segment_coincides x E :=
by
  sorry

end octahedron_net_folding_l447_447689


namespace expression_value_l447_447538

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447538


namespace common_ratio_arith_geom_seq_l447_447109

theorem common_ratio_arith_geom_seq (a : ℕ → ℝ) (d : ℝ) (h0 : d ≠ 0)
  (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h2 : (a 1), (a 5), (a 17) form a geometric sequence) :
  r = 3 :=
by
  -- Definitions of the terms in the arithmetic sequence
  let a1 := a 1
  let a5 := a 1 + 4 * d
  let a17 := a 1 + 16 * d

  -- given that (a1, a5, a17) is a geometric sequence, we have:
  have h_geometric : a 5 / a 1 = a 17 / a 5 := h2

  sorry -- proof to be filled in

end common_ratio_arith_geom_seq_l447_447109


namespace sum_of_fractions_l447_447522

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447522


namespace work_done_isothermal_l447_447669

variable (n : ℕ) (R T : ℝ) (P DeltaV : ℝ)

-- Definitions based on the conditions
def isobaric_work (P DeltaV : ℝ) := P * DeltaV

noncomputable def isobaric_heat (P DeltaV : ℝ) : ℝ :=
  (5 / 2) * P * DeltaV

noncomputable def isothermal_work (Q_iso : ℝ) : ℝ := Q_iso

theorem work_done_isothermal :
  ∃ (n R : ℝ) (P DeltaV : ℝ),
    isobaric_work P DeltaV = 20 ∧
    isothermal_work (isobaric_heat P DeltaV) = 50 :=
by 
  sorry

end work_done_isothermal_l447_447669


namespace geometric_sequence_common_ratio_l447_447096

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 2 * S 2 + 1) (h2 : a 4 = 2 * S 3 + 1) :
  ∃ q : ℝ, (q = 3) :=
by
  -- Proof will go here.
  sorry

end geometric_sequence_common_ratio_l447_447096


namespace number_of_incorrect_propositions_is_4_l447_447012

def proposition1 := ∀ {l m : ℕ}, (l ≠ m) ∧ (∃ (d : ℕ), ∀ (x : ℕ), dist x l = d ∧ dist x m = d) → (are_parallel l m ∨ intersect l m)
def proposition2 := ∀ (f : ℕ → ℕ), (is_inverse_proportion f) → (axis_symmetry f = 1)
def proposition3 := ∀ (b l : ℕ), (is_isosceles_triangle b l ∧ height_from_base b l = l/2) → (base_angle b l = 75)
def proposition4 := ∀ (circle : ℕ) (angle1 angle2 : ℕ), (is_congruent_circle circle) ∧ (inscribed_angle circle angle1 = inscribed_angle circle angle2) → (subtend_equal_arcs angle1 angle2)

theorem number_of_incorrect_propositions_is_4 :
  ¬ proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ ¬ proposition4 → 4 = 4 :=
by
  simp
  sorry

end number_of_incorrect_propositions_is_4_l447_447012


namespace order_of_a_ab_ab2_l447_447446

theorem order_of_a_ab_ab2 (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) : a < (a * b^2) ∧ (a * b^2) < (a * b) :=
by 
  have h1 : ab - ab^2 = a * b * (1 - b) := by sorry
  have h2 : ab^2 - a = a * (b^2 - 1) := by sorry
  have ab_gt_ab2 : a * b^2 < a * b := by sorry
  have a_lt_ab2 : a < a * b^2 := by sorry
  exact ⟨a_lt_ab2, ab_gt_ab2⟩

end order_of_a_ab_ab2_l447_447446


namespace twenty_fifth_number_in_base_three_l447_447955

-- Definitions and conditions
def base := 3
def decimal_number := 25

-- Proving the 25th number in base 3 is 221
theorem twenty_fifth_number_in_base_three :
  nat.toDigits base decimal_number = [2, 2, 1] :=
by sorry

end twenty_fifth_number_in_base_three_l447_447955


namespace avg_growth_rate_l447_447411

theorem avg_growth_rate :
  let jan_growth := 2.8 / 100 in
  let feb_growth := 2 / 100 in
  (1 + jan_growth) * (1 + feb_growth) = (1 + x)^2 :=
sorry

end avg_growth_rate_l447_447411


namespace infinite_geometric_series_sum_l447_447412

noncomputable def sum_infinite_geometric_series (a : ℚ) (r : ℚ) (|r| < 1) : ℚ :=
  a / (1 - r)

theorem infinite_geometric_series_sum
  (h₁ : 5 / (1 - (-1/2)) = 10/3) :
  sum_infinite_geometric_series 5 (-1/2) (by norm_num) = 10/3 :=
sorry

end infinite_geometric_series_sum_l447_447412


namespace exists_x_equality_l447_447427

open Real

theorem exists_x_equality :
  ∃ x : ℝ, 2^x + 3^x + 6^x = 7^x + 1 := by
  sorry

end exists_x_equality_l447_447427


namespace integer_values_b_l447_447726

theorem integer_values_b (h : ∃ b : ℤ, ∀ x : ℤ, (x^2 + b * x + 5 ≤ 0) → x ∈ {x | true}):
  {b : ℤ | ∃! x : ℤ, x^2 + b * x + 5 ≤ 0}.size = 2 :=
sorry

end integer_values_b_l447_447726


namespace least_value_in_S_l447_447638

def is_valid_set (S : Set ℕ) : Prop :=
  S ⊆ { n | n ≤ 12 } ∧ S.card = 6 ∧
  ∀ a b, a ∈ S → b ∈ S → a < b → ¬(b % a = 0)

theorem least_value_in_S (S : Set ℕ) (h : is_valid_set S) : ∃ x ∈ S, x = 4 :=
begin
  sorry
end

end least_value_in_S_l447_447638


namespace true_proposition_is_b_l447_447484

open Real

theorem true_proposition_is_b :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (¬ ∀ n : ℝ, n^2 ≥ n) ∧
  (¬ ∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧
  (¬ ∀ n : ℝ, n^2 < n) :=
  by
    sorry

end true_proposition_is_b_l447_447484


namespace six_times_eightx_plus_tenpi_eq_fourP_l447_447156

variable {x : ℝ} {π P : ℝ}

theorem six_times_eightx_plus_tenpi_eq_fourP (h : 3 * (4 * x + 5 * π) = P) : 
    6 * (8 * x + 10 * π) = 4 * P :=
sorry

end six_times_eightx_plus_tenpi_eq_fourP_l447_447156


namespace minimum_sqrt_value_l447_447679

theorem minimum_sqrt_value (x y : ℝ) (h : (x-3)^2 + (y-3)^2 = 1) : 
  ∃ m : ℝ, m = real.sqrt (15) ∧ ∀ z, z = real.sqrt (x^2 + y^2 + 2*y) → z ≥ m :=
sorry

end minimum_sqrt_value_l447_447679


namespace minimum_value_of_f_ge_7_l447_447851

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end minimum_value_of_f_ge_7_l447_447851


namespace usual_time_to_school_l447_447744

variables (R T : ℝ)

theorem usual_time_to_school (h₁ : T > 0) (h₂ : R > 0) (h₃ : R / T = (5 / 4 * R) / (T - 4)) :
  T = 20 :=
by
  sorry

end usual_time_to_school_l447_447744


namespace general_formula_range_of_values_l447_447996

-- Define the arithmetic sequence and sum.
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions:
variable (a1 d : ℤ)
variable (a1_pos : a1 > 0)
variable (an_seq : ℕ → ℤ)
variable S : ℕ → ℤ

-- Assume relationship between sum of first 9 terms and -a_5
axiom S9_eq_neg_a5 : sum_arithmetic_sequence a1 d 9 = -arithmetic_sequence a1 d 5
-- Assume a3 = 4
axiom a3_eq_4 : arithmetic_sequence a1 d 3 = 4

-- Prove the general formula for {a_n}
theorem general_formula (n : ℕ) : arithmetic_sequence a1 d n = -2 * n + 10 :=
  sorry

-- Prove the range of values of n for which S_n ≥ a_n
theorem range_of_values (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 10) : sum_arithmetic_sequence a1 d n ≥ arithmetic_sequence a1 d n :=
  sorry

end general_formula_range_of_values_l447_447996


namespace find_principal_amount_l447_447762

def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100
def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem find_principal_amount 
  (P R T : ℝ)
  (hR : R = 5)
  (hT : T = 2)
  (diff : compound_interest P R T - simple_interest P R T = 17) :
  P = 6800 :=
begin
  sorry
end

end find_principal_amount_l447_447762


namespace sin_double_angle_l447_447444

theorem sin_double_angle (α : ℝ) (h : Real.tan α + Real.cot α = 10 / 3) :
  Real.sin (2 * α) = 3 / 5 :=
sorry

end sin_double_angle_l447_447444


namespace polynomial_remainder_l447_447213

theorem polynomial_remainder (a : ℝ) (h : ∀ x : ℝ, x^3 + a * x^2 + 1 = (x^2 - 1) * (x + 2) + (x + 3)) : a = 2 :=
sorry

end polynomial_remainder_l447_447213


namespace range_of_a_l447_447645

-- Define the function f
def f (a x : ℝ) := (1 / 3) * x^3 - a * x^2 + 2 * x + 1

-- Condition for the function f to be monotonically increasing in [1,2]
def is_monotonically_increasing (a : ℝ) :=
  ∀ x ∈ Set.Icc (1:ℝ) (2:ℝ), deriv (f a) x ≥ 0

-- Define the equation for the hyperbola
def is_hyperbola (a : ℝ) :=
  2 * a^2 - 3 * a - 2 < 0

-- Main statement to prove
theorem range_of_a :
  (∀ a : ℝ, is_monotonically_increasing a ∧ is_hyperbola a → a ∈ Set.Ioc (-1 / 2) (Real.sqrt 2)) :=
sorry

end range_of_a_l447_447645


namespace log_problem_l447_447405

open Real

theorem log_problem : 2 * log 5 + log 4 = 2 := by
  sorry

end log_problem_l447_447405


namespace cabbage_price_l447_447627

theorem cabbage_price
  (earnings_wednesday : ℕ)
  (earnings_friday : ℕ)
  (earnings_today : ℕ)
  (total_weight : ℕ)
  (h1 : earnings_wednesday = 30)
  (h2 : earnings_friday = 24)
  (h3 : earnings_today = 42)
  (h4 : total_weight = 48) :
  (earnings_wednesday + earnings_friday + earnings_today) / total_weight = 2 := by
  sorry

end cabbage_price_l447_447627


namespace ones_digit_of_34_34_times_17_17_is_6_l447_447855

def cyclical_pattern_4 (n : ℕ) : ℕ :=
if n % 2 = 0 then 6 else 4

theorem ones_digit_of_34_34_times_17_17_is_6
  (h1 : 34 % 10 = 4)
  (h2 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4)
  (h3 : 17 % 2 = 1)
  (h4 : (34 * 17^17) % 2 = 0)
  (h5 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4) :
  (34^(34 * 17^17)) % 10 = 6 := 
by  
  sorry

end ones_digit_of_34_34_times_17_17_is_6_l447_447855


namespace base_area_functional_relationship_base_area_when_height_4_8_l447_447755

noncomputable def cylinder_base_area (h : ℝ) : ℝ := 24 / h

theorem base_area_functional_relationship (h : ℝ) (H : h ≠ 0) :
  cylinder_base_area h = 24 / h := by
  unfold cylinder_base_area
  rfl

theorem base_area_when_height_4_8 :
  cylinder_base_area 4.8 = 5 := by
  unfold cylinder_base_area
  norm_num

end base_area_functional_relationship_base_area_when_height_4_8_l447_447755


namespace asymptotes_of_C2_l447_447473

theorem asymptotes_of_C2 (a b : ℝ) (h_ab : a > b) (h_b : b > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 → False → True))
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 → True))
  (h_eccentricities : real.sqrt (1 - b^2 / a^2) * real.sqrt (1 + b^2 / a^2) = real.sqrt 3 / 2) :
  ∃ a' b' : ℝ, (a' = 1 / real.sqrt 2) ∧ (b' = 1 / real.sqrt 2) ∧
  ( ∀ x y : ℝ, x^2 / a'^2 - y^2 / b'^2 = 1 → (x + real.sqrt 2 * y = 0 ∨ x - real.sqrt 2 * y = 0) ) :=
by 
  -- Skipping the proof
  sorry

end asymptotes_of_C2_l447_447473


namespace expression_value_l447_447557

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447557


namespace problem_statement_l447_447561

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447561


namespace expected_value_of_X_l447_447029

noncomputable def expected_value_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

theorem expected_value_of_X (n : ℕ) (h : n > 0) :
  let X : ℕ → ℕ → ℚ := λ i n, (i - 1 : ℚ) / n
  let E_X : ℚ := ∑ i in Finset.range n, X (i + 1) n
  E_X = expected_value_shorter_gentlemen n := by
{
  sorry
}

end expected_value_of_X_l447_447029


namespace total_cost_of_long_distance_bill_l447_447435

theorem total_cost_of_long_distance_bill
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (minutes_billed : ℝ := 28.08) :
  monthly_fee + cost_per_minute * minutes_billed = 12.02 := by
  sorry

end total_cost_of_long_distance_bill_l447_447435


namespace peanuts_in_box_l447_447339

   theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (total_peanuts : ℕ) 
     (h1 : initial_peanuts = 4) (h2 : added_peanuts = 6) : total_peanuts = initial_peanuts + added_peanuts :=
   by
     sorry

   example : peanuts_in_box 4 6 10 rfl rfl = rfl :=
   by
     sorry
   
end peanuts_in_box_l447_447339


namespace intersection_A_B_l447_447466

-- Define set A
def A : Set ℝ := {x | x > 1}

-- Define set B
def B : Set ℝ := {y | y ≥ 2}

-- State the theorem
theorem intersection_A_B : A ∩ B = {z | z ∈ Icc 2 (Real.top)} :=
by 
  sorry

end intersection_A_B_l447_447466


namespace circumcircles_concurrent_l447_447635

variables (A B C D P Q : Type) [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q]
variables [convex_quadrilateral A B C D]
variables [intersection_point AD BC P]
variables [intersection_point AB CD Q]

theorem circumcircles_concurrent :
  concurrent (circumcircle CBQ) (circumcircle APB) (circumcircle DCP) (circumcircle ADQ) :=
sorry

end circumcircles_concurrent_l447_447635


namespace jake_time_to_plant_flowers_l447_447200

theorem jake_time_to_plant_flowers (hourly_rate charge_for_planting : ℝ)
  (h1 : hourly_rate = 20)
  (h2 : charge_for_planting = 45) :
  charge_for_planting / hourly_rate = 2.25 :=
by
  -- Given assumptions
  have hr: hourly_rate = 20 := h1
  have cp: charge_for_planting = 45 := h2
  
  -- Calculations derived from the problem
  calc 
    charge_for_planting / hourly_rate
        = 45 / 20 : by rw [cp, hr]
    ... = 2.25 : by norm_num

end jake_time_to_plant_flowers_l447_447200


namespace focal_distance_of_ellipse_l447_447912

theorem focal_distance_of_ellipse :
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 9) = 1 → (2 * Real.sqrt 7) = 2 * Real.sqrt 7 :=
by
  intros x y hxy
  sorry

end focal_distance_of_ellipse_l447_447912


namespace minimum_number_of_beams_l447_447020

def beam_problem : ℕ := 3030

theorem minimum_number_of_beams (n : ℕ) : 
  n = 2020 →
  (∀ (f : fin 3), ∃ (b : fin n × fin n → fin 2020), 
    (∀ x y, ∃ z, b ⟨x,y⟩ = ⟨y, z⟩) ∧ 
    (∀ (x1 x2 y1 y2 z1 z2 : fin n), b ⟨x1,y1⟩ = ⟨y2, z1⟩ → b ⟨x2,y2⟩ = ⟨y2, z2⟩ → x1 ≠ x2 ∨ z1 ≠ z2 ⇔ false)) →
  (∀ (x1 y1 x2 y2 : fin n), ¬ ∃ f, b ⟨x1,y1⟩ = f ∧ b ⟨x2,y2⟩ = f) →
  beam_problem = 3030 :=
by sorry

end minimum_number_of_beams_l447_447020


namespace janet_earnings_per_hour_l447_447619

def rate_per_post := 0.25  -- Janet’s rate per post in dollars
def time_per_post := 10    -- Time to check one post in seconds
def seconds_per_hour := 3600  -- Seconds in one hour

theorem janet_earnings_per_hour :
  let posts_per_hour := seconds_per_hour / time_per_post
  let earnings_per_hour := rate_per_post * posts_per_hour
  earnings_per_hour = 90 := sorry

end janet_earnings_per_hour_l447_447619


namespace max_value_m_l447_447472

theorem max_value_m (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∀ m, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : ∃ m, m = 12 :=
begin
  sorry
end

end max_value_m_l447_447472


namespace find_x_value_l447_447178

theorem find_x_value (X : ℕ) 
  (top_left : ℕ := 2)
  (top_second : ℕ := 3)
  (top_last : ℕ := 4)
  (bottom_left : ℕ := 3)
  (bottom_middle : ℕ := 5) 
  (top_sum_eq: 2 + 3 + X + 4 = 9 + X)
  (bottom_sum_eq: 3 + 5 + (X + 1) = 9 + X) : 
  X = 1 := by 
  sorry

end find_x_value_l447_447178


namespace selling_price_is_15_l447_447022

def cost_ingredients : ℕ := 12
def cakes : ℕ := 2
def cost_packaging_per_cake : ℕ := 1
def profit_per_cake : ℕ := 8
def selling_price_per_cake : ℕ := 15

theorem selling_price_is_15 
  (h1 : cost_ingredients = 12)
  (h2 : cakes = 2)
  (h3 : cost_packaging_per_cake = 1)
  (h4 : profit_per_cake = 8) :
  selling_price_per_cake = 15 :=
begin
  sorry
end

end selling_price_is_15_l447_447022


namespace option_C_correct_l447_447332

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end option_C_correct_l447_447332


namespace prove_union_l447_447111

variable (M N : Set ℕ)
variable (x : ℕ)

def M_definition := (0 ∈ M) ∧ (x ∈ M) ∧ (M = {0, x})
def N_definition := (N = {1, 2})
def intersection_condition := (M ∩ N = {2})
def union_result := (M ∪ N = {0, 1, 2})

theorem prove_union (M : Set ℕ) (N : Set ℕ) (x : ℕ) :
  M_definition M x → N_definition N → intersection_condition M N → union_result M N :=
by
  sorry

end prove_union_l447_447111


namespace logarithm_identity_l447_447940

theorem logarithm_identity (k x : ℝ) (hk : 0 < k ∧ k ≠ 1) (hx : 0 < x) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 3 → x = 343 :=
by
  intro h
  sorry

end logarithm_identity_l447_447940


namespace count_divisible_by_30_within_100_l447_447934

theorem count_divisible_by_30_within_100 : 
  (finset.range 101).filter (λ n, n % 30 = 0) = {30, 60, 90} :=
by
  sorry

end count_divisible_by_30_within_100_l447_447934


namespace tangent_line_at_1_range_of_m_l447_447135

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*x + a * Real.log x

theorem tangent_line_at_1 (a : ℝ) : 
  a = 2 → tangent of (λ x, f x a) at (1, f 1 a) = λ x, 2 * x - 3 :=
sorry

theorem range_of_m (a m x1 x2 : ℝ) (h_a_pos : 0 < a) (h_a_half : a < 1/2)
  (h_ext_pts : is_extr_pt (f x1 a) (f x2 a))
  (h_ineq : f x1 a ≥ m * x2) : 
  m ≤ -3/2 - Real.log 2 :=
sorry

end tangent_line_at_1_range_of_m_l447_447135


namespace minimum_employees_for_identical_training_l447_447379

def languages : Finset String := {"English", "French", "Spanish", "German"}

noncomputable def choose_pairings_count (n k : ℕ) : ℕ :=
Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem minimum_employees_for_identical_training 
  (num_languages : ℕ := 4) 
  (employees_per_pairing : ℕ := 4)
  (pairings : ℕ := choose_pairings_count num_languages 2) 
  (total_employees : ℕ := employees_per_pairing * pairings)
  (minimum_employees : ℕ := total_employees + 1):
  minimum_employees = 25 :=
by
  -- We skip the proof details as per the instructions
  sorry

end minimum_employees_for_identical_training_l447_447379


namespace monomial_completes_square_l447_447324

variable (x : ℝ)

theorem monomial_completes_square :
  ∃ (m : ℝ), ∀ (x : ℝ), ∃ (a b : ℝ), (16 * x^2 + 1 + m) = (a * x + b)^2 :=
sorry

end monomial_completes_square_l447_447324


namespace simplify_expression_l447_447115

variable (α : ℝ)
variable (hα : π < α ∧ α < 3 * π)

theorem simplify_expression :
  (sqrt ((1 + Real.cos (9 * π / 2 - α)) / (1 + Real.sin (α - 5 * π))) -
   sqrt ((1 - Real.cos (-3 * π / 2 - α)) / (1 - Real.sin (α - 9 * π)))) = -2 * Real.tan α :=
by sorry

end simplify_expression_l447_447115


namespace distance_between_hyperbola_vertices_l447_447072

theorem distance_between_hyperbola_vertices :
  (∀ x y : ℝ, (x^2 / 121) - (y^2 / 49) = 1) → 
  (∃ d : ℝ, d = 22) :=
by
  -- Assume the equation of the hyperbola
  intro hyp_eq,
  -- Use the provided information and conditions
  let a := Float.sqrt 121,
  -- The distance between the vertices is 2a
  have dist := 2 * a,
  -- Simplify a as sqrt(121) = 11
  have a_eq_11 : a = 11,
  -- Thus, distance is 2 * 11 = 22
  have dist_22 : dist = 22,
  use dist_22,
  sorry

end distance_between_hyperbola_vertices_l447_447072


namespace f_monotonicity_f_range_of_m_l447_447407

open Real

-- Define the function f(x) = e^(mx) + x^2 - mx
def f (m x : ℝ) : ℝ := exp (m * x) + x^2 - m * x

-- Prove that f(x) is monotonically decreasing on (-∞,0) and monotonically increasing on (0,∞)
theorem f_monotonicity (m : ℝ) :
  (∀ x : ℝ, x < 0 → deriv (f m) x < 0) ∧ (∀ x : ℝ, x > 0 → deriv (f m) x > 0) := 
sorry

-- Prove that if ∀ x1 x2 ∈ [-1, 1], |f(x1) - f(x2)| ≤ e - 1, then m ∈ [-1, 1]
theorem f_range_of_m (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ set.Icc (-1 : ℝ) 1 → x₂ ∈ set.Icc (-1 : ℝ) 1 → abs (f m x₁ - f m x₂) ≤ exp 1 - 1) → 
  m ∈ set.Icc (-1 : ℝ) 1 := 
sorry

end f_monotonicity_f_range_of_m_l447_447407


namespace four_digit_numbers_count_l447_447017

theorem four_digit_numbers_count :
  let valid_numbers_count : ℕ :=
    ( 
      let d_set : Finset ℕ := Finset.range 10 in
      let tens_even : Finset ℕ := {0, 2, 4, 6, 8} in
      d_set.subsetOf (λ d, (d != 0 ∧ d != 1 ∧ d != 8 ∧ d != 9)) in
      
      let case_08 := 2 * (tens_even.card - 3) * (d_set.card - (tens_even.card + 3)) in
      let case_19 := 2 * ((tens_even.card - 1) * (d_set.card - (tens_even.card + 3)) + 7) in
  
      case_08 + case_19
    )
  in
  valid_numbers_count = 104 :=
by 
-- Proof steps here
sorry

end four_digit_numbers_count_l447_447017


namespace arrange_numbers_l447_447502

theorem arrange_numbers (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  let y := x^x in
  let z := x^(x^x) in
  x < z ∧ z < y :=
by
  sorry

end arrange_numbers_l447_447502


namespace find_three_digit_number_l447_447426

noncomputable def three_digit_number (M Γ U : ℕ) : ℕ := 100 * M + 10 * Γ + U

theorem find_three_digit_number :
  ∃ (M Γ U : ℕ), M ≠ Γ ∧ Γ ≠ U ∧ U ≠ M ∧
    (three_digit_number M Γ U = (M + Γ + U) * (M + Γ + U - 2) ∧
     three_digit_number M Γ U = 195) :=
begin
  sorry
end

end find_three_digit_number_l447_447426


namespace cellphone_cost_l447_447235

-- Define the conditions
def cost_of_each_cellphone (C : ℝ) (total_paid : ℝ) (discount_rate : ℝ) : Prop :=
  total_paid = (2 * C) * (1 - discount_rate)

-- State the theorem
theorem cellphone_cost (C : ℝ) (total_paid : ℝ) (discount_rate : ℝ) (h : cost_of_each_cellphone C total_paid discount_rate) :
  C = 800 := by
  -- Assume the given values
  let total_paid := 1520
  let discount_rate := 0.05
  -- Simplify the conditions
  have h1 : total_paid = (2 * C) * (1 - discount_rate), from h
  -- Skip the detailed proof steps
  sorry

end cellphone_cost_l447_447235


namespace vasya_grades_l447_447194

theorem vasya_grades (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a3 = 4) 
  (h2 : (a1 + a2 + a3 + a4 + a5) / 5 = 19 / 5) 
  (h3 : (a1, a2, a3, a4, a5).to_list.most_freq = 5) 
  (h4 : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) :
  (a1, a2, a3, a4, a5) = (2, 3, 4, 5, 5) := 
by sorry

end vasya_grades_l447_447194


namespace repeating_decimals_difference_l447_447805

theorem repeating_decimals_difference :
  let x := 234 / 999
  let y := 567 / 999
  let z := 891 / 999
  x - y - z = -408 / 333 :=
by
  sorry

end repeating_decimals_difference_l447_447805


namespace distance_between_hyperbola_vertices_l447_447079

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l447_447079


namespace no_valid_solutions_l447_447087

theorem no_valid_solutions : ∀ (x : ℝ), sqrt (7 - x) ≠ x * sqrt (7 - x) - 1 :=
by
  intro x
  sorry

end no_valid_solutions_l447_447087


namespace bridge_length_is_correct_l447_447280

def train_length : ℝ := 145 -- Length of the train in meters
def train_speed : ℝ := 45 / 3.6 -- Convert the speed from km/hr to m/s (45 * (1000 / 3600))
def crossing_time : ℝ := 30 -- Time to cross in seconds
def total_distance : ℝ := train_speed * crossing_time -- Total distance covered in meters
def bridge_length : ℝ := total_distance - train_length -- Length of the bridge in meters

theorem bridge_length_is_correct : bridge_length = 230 := by
  sorry

end bridge_length_is_correct_l447_447280


namespace red_star_team_wins_l447_447607

theorem red_star_team_wins (x y : ℕ) (h1 : x + y = 9) (h2 : 3 * x + y = 23) : x = 7 := by
  sorry

end red_star_team_wins_l447_447607


namespace minimum_disks_needed_l447_447982

def files : ℕ := 30
def disk_space : ℝ := 1.44  -- Disk capacity in MB
def file_sizes : list ℝ := [0.8, 0.8, 0.8] ++ (repeat 0.7 12) ++ (repeat 0.4 15)  -- List of file sizes

def total_disks_needed : ℕ :=
  let num_large_files := countp (λ x, x = 0.8) file_sizes in
  let num_medium_files := countp (λ x, x = 0.7) file_sizes in
  let num_small_files := countp (λ x, x = 0.4) file_sizes in
  let paired_with_small := min num_large_files num_small_files in
  let remaining_large := num_large_files - paired_with_small in
  let remaining_small := num_small_files - paired_with_small in
  let total_remaining_files_size := (num_medium_files * 0.7) + (remaining_small * 0.4) in
  let additional_disks := ⌈ total_remaining_files_size / disk_space ⌉ in
  paired_with_small + remaining_large + additional_disks

theorem minimum_disks_needed (h : total_disks_needed = 13) : 
  ∀ (num_files : ℕ) (space_disk : ℝ) (sizes : list ℝ), num_files = 30 → space_disk = 1.44 → sizes = file_sizes → total_disks_needed = 13 :=
by
  intros
  sorry

end minimum_disks_needed_l447_447982


namespace _l447_447604

example : ∫ x in 0..π, sin x = 2 :=
by
  -- The antiderivative of sin x is -cos x
  -- Using the fundamental theorem of calculus
  sorry

end _l447_447604


namespace inequality_inverse_l447_447941

theorem inequality_inverse (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a) < (1 / b) :=
by
  sorry

end inequality_inverse_l447_447941


namespace general_formula_l447_447924

noncomputable def a : ℕ → ℤ
| 1     := 1
| (n+1) := 2 * (a n) + 4

theorem general_formula (n : ℕ) (hn : n > 0) : a n = 5 * 2^(n - 1) - 4 :=
by sorry

end general_formula_l447_447924


namespace angle_between_vectors_l447_447929

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Definitions given as conditions
def a_norm : ∥a∥ = 2 := sorry
def b_norm : ∥b∥ = sqrt 3 := sorry
def perpendicular : inner_product_space.inner b (a + b) = 0 := sorry

-- Theorem to prove
theorem angle_between_vectors :
  inner_product_space.angle ℝ a b = real.pi * 5 / 6 :=
sorry

end angle_between_vectors_l447_447929


namespace b_divisible_by_8_l447_447989

theorem b_divisible_by_8 (b : ℕ) (h_even: ∃ k : ℕ, b = 2 * k) (h_square: ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (b ^ n - 1) / (b - 1) = m ^ 2) : b % 8 = 0 := 
by
  sorry

end b_divisible_by_8_l447_447989


namespace next_palindromic_year_product_l447_447304

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10
  s = s.reverse

def digits_product (n : ℕ) : ℕ :=
  (n.to_digits 10).prod

theorem next_palindromic_year_product :
  ∀ n : ℕ, n > 1991 ∧ is_palindrome n ∧ (n % 10 = 3) → digits_product n = 0 :=
begin
  intros n h,
  sorry,
end

end next_palindromic_year_product_l447_447304


namespace tables_in_conference_hall_l447_447176

theorem tables_in_conference_hall (c t : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : 4 * c + 4 * t = 648) : 
  t = 18 :=
by sorry

end tables_in_conference_hall_l447_447176


namespace round_sum_to_nearest_tenth_l447_447008

theorem round_sum_to_nearest_tenth (a b : ℝ) (c : ℝ) (h₀ : a = 2.72) (h₁ : b = 0.76) (h₂ : c = 3.48) :
  Real.round (a + b) = 3.5 :=
by
  rw [h₀, h₁]
  have h_sum : a + b = c, by sorry
  rw [h_sum, ←h₂]
  rw [Real.round]
  sorry

end round_sum_to_nearest_tenth_l447_447008


namespace interval_length_and_a_b_sum_l447_447813

def lattice_grid (x y : ℕ) : Prop := 
  1 ≤ x ∧ x ≤ 50 ∧ 1 ≤ y ∧ y ≤ 50

def bounded_rectangle (x y : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 40 ∧ 10 ≤ y ∧ y ≤ 40

def num_points_below_line (m : ℚ) : ℕ := 
  finset.card {p : ℕ × ℕ | bounded_rectangle p.1 p.2 ∧ (p.2 : ℚ) ≤ m * (p.1 : ℚ)}

theorem interval_length_and_a_b_sum :
  ∃ (a b : ℕ), 
    a.gcd b = 1 ∧ 
    (∃ m : ℚ, bounded_rectangle (x : ℕ) (y : ℕ) → num_points_below_line m = 500 → m = a / b) ∧
    ((23 : ℚ) / 46) ∈ [(0 : ℚ), 1] ∧ 
    a + b = 69 := 
sorry

end interval_length_and_a_b_sum_l447_447813


namespace eccentricity_range_l447_447908

def ellipse (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def circle (b x y : ℝ) : Prop :=
  b > 0 ∧ (x^2 + y^2 = b^2)

def tangents_to_circle (m n b : ℝ) : Prop :=
  m^2 + n^2 = 4 * b^2

theorem eccentricity_range (a b m n e : ℝ) :
  ellipse a b m n →
  circle b m n →
  tangents_to_circle m n b →
  e = Real.sqrt (1 - (b / a)^2) →
  e ∈ Set.Ico (Real.sqrt 3 / 2) 1 :=
by
  intros H1 H2 H3 H4
  sorry

end eccentricity_range_l447_447908


namespace trigonometric_identity_proof_l447_447471

theorem trigonometric_identity_proof (x : ℝ) 
(h1 : sin (x + π / 6) = 1 / 4) : 
sin (5 * π / 6 - x) + sin^2 (π / 3 - x) + cos (2 * x + π / 3) = 33 / 16 :=
by
  sorry

end trigonometric_identity_proof_l447_447471


namespace power_sum_is_99_l447_447397

theorem power_sum_is_99 : 3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 :=
by sorry

end power_sum_is_99_l447_447397


namespace siblings_are_Emma_and_Olivia_l447_447956

structure Child where
  name : String
  eyeColor : String
  hairColor : String
  ageGroup : String

def Bella := Child.mk "Bella" "Green" "Red" "Older"
def Derek := Child.mk "Derek" "Gray" "Red" "Younger"
def Olivia := Child.mk "Olivia" "Green" "Brown" "Older"
def Lucas := Child.mk "Lucas" "Gray" "Brown" "Younger"
def Emma := Child.mk "Emma" "Green" "Red" "Older"
def Ryan := Child.mk "Ryan" "Gray" "Red" "Older"
def Sophia := Child.mk "Sophia" "Green" "Brown" "Younger"
def Ethan := Child.mk "Ethan" "Gray" "Brown" "Older"

def sharesCharacteristics (c1 c2 : Child) : Nat :=
  (if c1.eyeColor = c2.eyeColor then 1 else 0) +
  (if c1.hairColor = c2.hairColor then 1 else 0) +
  (if c1.ageGroup = c2.ageGroup then 1 else 0)

theorem siblings_are_Emma_and_Olivia :
  sharesCharacteristics Bella Emma ≥ 2 ∧
  sharesCharacteristics Bella Olivia ≥ 2 ∧
  (sharesCharacteristics Bella Derek < 2) ∧
  (sharesCharacteristics Bella Lucas < 2) ∧
  (sharesCharacteristics Bella Ryan < 2) ∧
  (sharesCharacteristics Bella Sophia < 2) ∧
  (sharesCharacteristics Bella Ethan < 2) :=
by
  sorry

end siblings_are_Emma_and_Olivia_l447_447956


namespace sequence_a6_value_l447_447971

theorem sequence_a6_value 
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n : ℕ, n ≥ 1 → (1 / a n) + (1 / a (n + 2)) = 2 / a (n + 1)) :
  a 6 = 1 / 3 :=
by
  sorry

end sequence_a6_value_l447_447971


namespace polynomial_evaluation_l447_447061

theorem polynomial_evaluation 
  (x : ℝ) (h : x^2 - 3*x - 10 = 0 ∧ x > 0) :
  x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 :=
sorry

end polynomial_evaluation_l447_447061


namespace relationship_among_a_b_c_l447_447877

-- Definitions from conditions
def f (x : ℝ) := (1/2)^x
def a := f (0.9^0.9)
def b := f (Real.log (Real.log10 9))
def c := f (1/Real.sin 1)

-- Final statement
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l447_447877


namespace find_angle_A_find_triangle_area_l447_447144

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * b * c * Math.sin A

theorem find_angle_A (a b c A B C : ℝ) :
  c * (1 + Math.cos A) = sqrt 3 * a * Math.sin C → A = π / 3 :=
by
  intro h
  sorry

theorem find_triangle_area (a b c A B C : ℝ) :
  a = sqrt 7 → b = 1 → A = π / 3 → S = 0.5 * b * c * Math.sin A := by
  intros ha hb hA
  let c := (7 - 1) -- Derived from calculations in the solution steps
  let S := 0.5 * b * c * Math.sin A
  have hS : S = 3 * Math.sqrt 3 / 4 := by sorry
  exact hS
  sorry

end find_angle_A_find_triangle_area_l447_447144


namespace trapezoid_area_inequality_l447_447886

variables (A B C D E F H G : Type) 
variables [Add A] [Mul A] [HasSubset A]

noncomputable def area (x y : A) : A := 
  x * y -- Define the area function

def is_trapezoid (ABCD : A) (AB_parallel_CD : Prop) (E_on_AB : Prop) (F_on_CD : Prop) : Prop :=
AB_parallel_CD ∧ E_on_AB ∧ F_on_CD

theorem trapezoid_area_inequality :
  ∀ (ABCD EHFG : A), ∀ (AB_parallel_CD E_on_AB F_on_CD : Prop), 
  is_trapezoid ABCD AB_parallel_CD E_on_AB F_on_CD → 
  area EHFG 1 ≤ (1/4) * area ABCD 1 := 
begin
  intros, 
  sorry
end

end trapezoid_area_inequality_l447_447886


namespace problem_proof_l447_447650

noncomputable def problem_statement (n : ℕ) (a b : ℕ → ℝ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i > 0 ∧ b i > 0) ∧
  (∑ i in finset.range n, a i = ∑ i in finset.range n, b i) →
  (∑ i in finset.range n, a i^2 / (a i + b i) ≥ (1 / 2) * ∑ i in finset.range n, a i)

-- The theorem that needs to be proved
theorem problem_proof (n : ℕ) (a b : ℕ → ℝ) : problem_statement n a b := sorry

end problem_proof_l447_447650


namespace cannot_use_diff_of_squares_l447_447014

/-- Among the following polynomial multiplications, the one that cannot be calculated using the difference of squares formula is (a - b)(-a + b) -/
theorem cannot_use_diff_of_squares : 
(∃ a b : ℤ, ¬(∃ x y : ℤ, (a - b) * (-a + b) = (x - y) * (x + y))) ∧
(∀ m n : ℤ, (∃ x y : ℤ, (m^3 - n^3) * (m^3 + n^3) = (x - y) * (x + y))) ∧
(∀ x : ℤ, (∃ x' y' : ℤ, (-7 - x) * (7 - x) = (x' - y') * (x' + y'))) ∧
(∀ x y : ℤ, (∃ x' y' : ℤ, (x^2 - y^2) * (y^2 + x^2) = (x' - y') * (x' + y'))) :=
begin
  sorry
end

end cannot_use_diff_of_squares_l447_447014


namespace sequence_has_correct_13th_term_l447_447818

def sequence (n : ℕ) : ℕ :=
  -- This function defines the nth term of the sequence
  if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 1
  else if n = 5 then 2
  else if n = 6 then 3
  else if n = 7 then 1
  else if n = 8 then 2
  else if n = 9 then 3
  else if n = 10 then 4
  else if n = 11 then 1
  else if n = 12 then 2
  else if n = 13 then 3
  else sorry -- leaving for further terms

theorem sequence_has_correct_13th_term : sequence 13 = 3 :=
by
  rfl

end sequence_has_correct_13th_term_l447_447818


namespace distance_between_hyperbola_vertices_l447_447073

theorem distance_between_hyperbola_vertices :
  (∀ x y : ℝ, (x^2 / 121) - (y^2 / 49) = 1) → 
  (∃ d : ℝ, d = 22) :=
by
  -- Assume the equation of the hyperbola
  intro hyp_eq,
  -- Use the provided information and conditions
  let a := Float.sqrt 121,
  -- The distance between the vertices is 2a
  have dist := 2 * a,
  -- Simplify a as sqrt(121) = 11
  have a_eq_11 : a = 11,
  -- Thus, distance is 2 * 11 = 22
  have dist_22 : dist = 22,
  use dist_22,
  sorry

end distance_between_hyperbola_vertices_l447_447073


namespace find_quotient_l447_447210

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![4, 5]]

noncomputable def matrix_b (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

theorem find_quotient (a b c d : ℝ) (H1 : matrix_a * (matrix_b a b c d) = (matrix_b a b c d) * matrix_a)
  (H2 : 2*b ≠ 3*c) : ((a - d) / (c - 2*b)) = 3 / 2 :=
  sorry

end find_quotient_l447_447210


namespace expression_equals_neg_eight_l447_447937

variable {a b : ℝ}

theorem expression_equals_neg_eight (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| ≠ |b|) :
  ( (b^2 / a^2 + a^2 / b^2 - 2) * 
    ((a + b) / (b - a) + (b - a) / (a + b)) * 
    (((1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2)) - ((1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)))
  ) = -8 :=
by
  sorry

end expression_equals_neg_eight_l447_447937


namespace final_value_l447_447548

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447548


namespace total_amount_paid_l447_447293

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l447_447293


namespace total_votes_in_election_l447_447182

theorem total_votes_in_election
    (A : ℝ)
    (valid_votes_percentage : ℝ)
    (invalid_votes_percentage : ℝ)
    (votes_candidate_A_percentage : ℝ)
    (total_valid_votes_A : ℝ)
    (H1 : total_valid_votes_A = 357000)
    (H2 : votes_candidate_A_percentage = 0.75)
    (H3 : invalid_votes_percentage = 0.15)
    (H4 : valid_votes_percentage = 1 - invalid_votes_percentage)
    : ∃ V : ℝ, V = 560000 :=
by
    have H5 : total_valid_votes_A = votes_candidate_A_percentage * (valid_votes_percentage * V) := (by sorry)
    have H6 : V = total_valid_votes_A / (votes_candidate_A_percentage * valid_votes_percentage) := (by sorry)
    use (357000 / (0.75 * 0.85))
    simp [H1, H2, H3, H4, H5, H6]
    sorry

end total_votes_in_election_l447_447182


namespace range_of_a_l447_447950

noncomputable def circle_one : Set (ℝ × ℝ) := {p | (p.1-2*a)^2 + (p.2-a-3)^2 = 4}
def circle_two : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

theorem range_of_a :
  (∀ p ∈ circle_one, ∃ q ∈ circle_two, dist p q = 1) ↔ (- (6:ℝ)/5 < a ∧ a < 0) := 
sorry

end range_of_a_l447_447950


namespace probability_B_given_A_l447_447707

variables {Ω : Type*} {P : MeasureTheory.Measure Ω} [ProbabilityMeasure P]
variables (A B : Set Ω)

theorem probability_B_given_A (hA : P(A) = 0.80) (hB : P(B) = 0.60) :
  (Probability.cond A B) = 0.75 :=
by
  sorry

end probability_B_given_A_l447_447707


namespace max_value_of_a_b_l447_447479

theorem max_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∃ A B : ℝ × ℝ, (A, B ∈ subtype.mk '' {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} ∧ (∃ k1 k2 : ℝ, 
  A = (k1, k2) ∧ B = (k2, -k1) ∧ a*k1 + b*k2 = 1 ∧ a*k2 + b*(-k1) = 1)) ∧ 
  1 / sqrt (a^2 + b^2) = sqrt 2 / 2) : 
  a + b ≤ 2 :=
sorry

end max_value_of_a_b_l447_447479


namespace caterer_cheapest_option_l447_447263

theorem caterer_cheapest_option :
  ∃ x : ℕ, x ≥ 42 ∧ (∀ y : ℕ, y ≥ x → (20 * y < 120 + 18 * y) ∧ (20 * y < 250 + 14 * y)) := 
by
  sorry

end caterer_cheapest_option_l447_447263


namespace red_apples_count_l447_447420

theorem red_apples_count
  (r y g : ℕ)
  (h1 : r = y)
  (h2 : g = 2 * r)
  (h3 : r + y + g = 28) : r = 7 :=
sorry

end red_apples_count_l447_447420


namespace true_propositions_count_l447_447016

theorem true_propositions_count :
  (∃ x₀ : ℤ, x₀^3 < 0) ∧
  ((∀ a : ℝ, (∃ x : ℝ, a*x^2 + 2*x + 1 = 0 ∧ x < 0) ↔ a ≤ 1) → false) ∧ 
  (¬ (∀ x : ℝ, x^2 = 1/4 * x^2 → y = 1 → false)) →
  true_prop_count = 1 := 
sorry

end true_propositions_count_l447_447016


namespace shaded_area_is_semicircle_area_l447_447065

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let semicircle_area := π * R^2 / 2
  in semicircle_area

theorem shaded_area_is_semicircle_area (R : ℝ) :
  area_shaded_figure R = π * R^2 / 2 :=
sorry

end shaded_area_is_semicircle_area_l447_447065


namespace smallest_sum_ab_l447_447113

theorem smallest_sum_ab (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2^10 * 3^6 = a^b) : a + b = 866 :=
sorry

end smallest_sum_ab_l447_447113


namespace sum_triples_l447_447048

theorem sum_triples (s : Finset (ℕ × ℕ × ℕ)) (cond : ∀ t ∈ s, (1 ≤ t.1 ∧ t.1 < t.2 ∧ t.2 < t.3)) :
  ∑ t in s, (1 : ℚ) / (2 ^ t.1 * 5 ^ t.2 * 7 ^ t.3) = 1 / 1836 := 
sorry

end sum_triples_l447_447048


namespace math_problem_l447_447530

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447530


namespace colored_cubes_count_l447_447404

open Set

-- Defining conditions for the cube
def colors := { faces : Fin 6 → Prop // ∀ i j k : Fin 6, i ≠ j → j ≠ k → k ≠ i → ¬(faces i ∧ faces j ∧ faces k) }

-- Hypothetical cube faces color assignment as per the problem
noncomputable def colored_faces : colors := 
  ⟨λ i, i < 3, by
    intros i j k h1 h2 h3
    simp only [and_false, false_and, not_and']
    intro h
    cases h
    repeat (exact h1 (h ▸ h_1))⟩

-- Defining the problem: Number of smaller cubes with at least one red face and one blue face
def count_colored_cubes (n : Nat) := 56

-- Theorem stating the number of such cubes
theorem colored_cubes_count : count_colored_cubes 512 = 56 :=
  by exact sorry

end colored_cubes_count_l447_447404


namespace range_of_a_l447_447169

theorem range_of_a (a x : ℝ) (h_eq : 2 * x - 1 = x + a) (h_pos : x > 0) : a > -1 :=
sorry

end range_of_a_l447_447169


namespace limit_sqrt_a_n_plus_n_l447_447926

noncomputable def a : ℕ → ℚ
| 0 := 2
| 1 := 6
| (n + 2) := (a (n + 1) ^ 2 - 2 * a (n + 1)) / a n

theorem limit_sqrt_a_n_plus_n :
  filter.tendsto (λ n : ℕ, real.sqrt (a n + n)) filter.at_top (nhds 1) :=
sorry

end limit_sqrt_a_n_plus_n_l447_447926


namespace range_of_k_l447_447168

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, ¬ (k * x^2 - 2 * |x - 1| + 6 * k < 0)) ↔ k ≥ (1 + real.sqrt 7) / 6 := by
  sorry

end range_of_k_l447_447168


namespace weight_of_new_student_l447_447267

theorem weight_of_new_student (W x y z : ℝ) (h : (W - x - y + z = W - 40)) : z = 40 - (x + y) :=
by
  sorry

end weight_of_new_student_l447_447267


namespace passes_through_0_2_l447_447334

def f (x : ℝ) : ℝ := (1 / 2) ^ x + 1

theorem passes_through_0_2 : f 0 = 2 :=
by 
  -- Proof goes here
  sorry

end passes_through_0_2_l447_447334


namespace length_of_DE_l447_447106

theorem length_of_DE (a : ℝ) :
  ∀ (A B C D E : Type*) (A_ne_B : A ≠ B) (A_ne_C : A ≠ C) (A_ne_D : A ≠ D) (A_ne_E : A ≠ E) 
    (AB AC BC : A ⟂ B) (BD_perp_BC : D ⟂ B ∧ D ⟂ C) (A_vertex : right_angle A B C = 90°)
    (angle_B_60 : angle (B A) = 60°) (angle_bisector_AE : bisects_angle A E B C) :
  ∃ (D E : Type*), length (D ↔ E) = a * (sqrt 3 - 3/2) :=
by
  sorry

end length_of_DE_l447_447106


namespace real_root_of_equation_l447_447858

theorem real_root_of_equation :
  ∃ x : ℝ, (sqrt (x + 4) + sqrt (x + 6) = 12) ∧ (x = 4465 / 144) :=
begin
  use (4465 / 144),
  split,
  {
    sorry,
  },
  {
    refl,
  }
end

end real_root_of_equation_l447_447858


namespace trapezium_area_correct_l447_447760

def trapezium_area (a b h : ℝ) : ℝ :=
  1 / 2 * (a + b) * h

theorem trapezium_area_correct :
  trapezium_area 26 18 15 = 330 := by
  sorry

end trapezium_area_correct_l447_447760


namespace prob_sum_multiple_of_ten_l447_447360

theorem prob_sum_multiple_of_ten :
  (∃ (cards : Finset ℕ), 
     cards = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ 
     let possible_pairs := (cards.product cards).filter (λ (p : ℕ × ℕ), (p.1 + p.2) % 10 = 0) in
     (possible_pairs.card / (cards.card * cards.card) : ℝ) = 1 / 10) :=
by
sorry

end prob_sum_multiple_of_ten_l447_447360


namespace expression_value_l447_447556

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447556


namespace g_of_f_roots_reciprocal_l447_447651

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c

theorem g_of_f_roots_reciprocal
  (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ∃ g : ℝ → ℝ, g 1 = (4 - a) / (4 * c) :=
sorry

end g_of_f_roots_reciprocal_l447_447651


namespace probability_no_3x3_green_square_l447_447417

theorem probability_no_3x3_green_square (p q : ℕ) (h_coprime: Nat.coprime p q) (h_fraction: p / gcd(p, q) = 2033 ∧ q / gcd(p, q) = 2048) :
  p + q = 4081 := by
  -- proof will go here
  sorry

end probability_no_3x3_green_square_l447_447417


namespace derivative_f_derivative_g_derivative_h_derivative_i_l447_447688

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := cos (2 * x)
noncomputable def h (x : ℝ) : ℝ := 3^x / (Real.log 3)
noncomputable def i (x : ℝ) : ℝ := Real.log10 x

theorem derivative_f : ∀ x, deriv f x = -1 / x^2 := by sorry
theorem derivative_g : ∀ x, deriv g x = -2 * sin (2 * x) := by sorry
theorem derivative_h : ∀ x, deriv h x = 3^x := by sorry
theorem derivative_i : ∀ x, deriv i x = 1 / (x * Real.log 10) := by sorry

example : (deriv f x ≠ 1 / x^2) ∧
          (deriv g x = -2 * sin (2 * x)) ∧
          (deriv h x = 3^x) ∧ 
          (deriv i x ≠ -1 / (x * Real.log 10)) :=
by 
  sorry

end derivative_f_derivative_g_derivative_h_derivative_i_l447_447688


namespace sum_integers_neg50_to_100_l447_447039

theorem sum_integers_neg50_to_100 : (List.range' (-50) 151).sum = 3775 := by
  sorry

end sum_integers_neg50_to_100_l447_447039


namespace prime_count_between_50_and_70_l447_447152

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : Nat) (p : Nat → Prop) : List Nat :=
  (List.range' a (b - a + 1)).filter p

theorem prime_count_between_50_and_70 : (primes_between 50 70 is_prime).length = 4 :=
by 
  sorry

end prime_count_between_50_and_70_l447_447152


namespace b_seq_general_term_sum_b_seq_3n_l447_447107

noncomputable def a_seq : ℕ → ℝ
| 0     := 2
| (n+1) := sorry -- recursive definition based on the given condition

-- Definition of b_seq using a_seq according to the problem
noncomputable def b_seq (n : ℕ) : ℝ := 1 / (a_seq n - 1)

-- Define the sum of first n terms of the sequence {b_n * 3^n}
def sum_b_n_times_3_to_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, b_seq i * 3^i)

-- Lean statement for the first part of the problem
theorem b_seq_general_term (n : ℕ) : b_seq n = (n + 2) / 3 :=
sorry

-- Lean statement for the second part of the problem
theorem sum_b_seq_3n (n : ℕ) : sum_b_n_times_3_to_n n = ((2 * n + 3) * 3^n - 3) / 4 :=
sorry

end b_seq_general_term_sum_b_seq_3n_l447_447107


namespace angle_is_90_degrees_l447_447844

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (dot_product u v / (magnitude u * magnitude v))

noncomputable def is_ninety_degrees : Prop :=
  angle_between_vectors (3, -2, 2) (2, 2, -1) = real.pi / 2

theorem angle_is_90_degrees : is_ninety_degrees :=
by
  sorry

end angle_is_90_degrees_l447_447844


namespace p_equality_l447_447258

-- Define the function p, which is 0 for n < 0
def p : ℤ → ℤ
| n := if n < 0 then 0 else sorry -- It's normally defined by other rules which should be filled in.

-- The sequence p_k is some predefined sequence, for example:
def p_k : ℤ → ℤ := sorry -- Fill in the appropriate sequence definition.

theorem p_equality (n : ℤ) :
  (p(n) = ∑ k in finset.filter (λ k => k ≠ 0) finset.univ, (-1)^(k+1) * p(n - p_k k)) :=
sorry -- Proof goes here.

end p_equality_l447_447258


namespace func_properties_l447_447490

-- Lean statement for the given problem conditions and questions
theorem func_properties (k b : ℝ) 
  (hx : ∀ x : ℝ, f x = k * x + b)
  (A B : ℝ × ℝ)
  (hA : A = (-b / k, 0))
  (hB : B = (0, b))
  (hAB : A.1 + A.2 = 2 ∧ B.1 + B.2 = 2)
  (g : ℝ → ℝ)
  (hg : ∀ x, g x = x^2 - x - 6) :

  (k = 1 ∧ b = 2) ∧ 
  (∀ x, (k * x + b > g x) → 
    (x + 2 > 0 → (let y := (g x + 1) / (k * x + b) in y) ≥ -3 ∧ y = -3 → x = -1)) :=
by
  sorry

end func_properties_l447_447490


namespace complement_intersection_l447_447867

-- Define the universal set U and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {4, 7, 8}

-- Compute the complements
def complement_U (s : Set ℕ) : Set ℕ := U \ s
def comp_A : Set ℕ := complement_U A
def comp_B : Set ℕ := complement_U B

-- Define the intersection of the complements
def intersection_complements : Set ℕ := comp_A ∩ comp_B

-- The theorem to prove
theorem complement_intersection :
  intersection_complements = {1, 2, 6} :=
by
  sorry

end complement_intersection_l447_447867


namespace complete_square_l447_447050

theorem complete_square :
  (∀ x: ℝ, 2 * x^2 - 4 * x + 1 = 2 * (x - 1)^2 - 1) := 
by
  intro x
  sorry

end complete_square_l447_447050


namespace largest_sphere_radius_l447_447005

theorem largest_sphere_radius (inner_radius outer_radius circle_center_z circle_center_x circle_radius : ℝ) 
  (inner_radius_pos : inner_radius = 4) (outer_radius_pos : outer_radius = 6)
  (circle_center_z_eq : circle_center_z = 1) (circle_center_x_eq : circle_center_x = 5) 
  (circle_radius_eq : circle_radius = 1) : 
  ∃ (r : ℝ), r = 13 / 2 :=
by
  have inner_radius := 4
  have outer_radius := 6
  have circle_center := (5, 0, 1)
  have circle_radius := 1
  use 13 / 2
  sorry

end largest_sphere_radius_l447_447005


namespace min_n_for_row_column_color_l447_447376

theorem min_n_for_row_column_color (n : ℕ) : (∀ (coloring : fin n → fin n → ℕ), 
  (∀ i : fin n, ∃ c : ℕ, (∃ j₁ j₂ j₃ : fin n, j₁ ≠ j₂ ∧ j₂ ≠ j₃ ∧ j₁ ≠ j₃ ∧ coloring i j₁ = c ∧ coloring i j₂ = c ∧ coloring i j₃ = c) ∨
  (∃ i₁ i₂ i₃ : fin n, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ ∧ coloring i₁ i = c ∧ coloring i₂ i = c ∧ coloring i₃ i = c))) ↔ n = 7 :=
begin
  sorry
end

end min_n_for_row_column_color_l447_447376


namespace tan_ratio_l447_447117

theorem tan_ratio (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 :=
sorry

end tan_ratio_l447_447117


namespace range_of_a_l447_447486

noncomputable def f : ℝ → ℝ := λ x, if x ≥ 0 then x^2 + 4 * x else 4 * x - x^2

theorem range_of_a (a : ℝ) : f (2 - 2 * a) > f a → a < 2 / 3 :=
by
  intro h
  sorry

end range_of_a_l447_447486


namespace no_regular_ngon_with_lattice_vertices_l447_447188

-- Define a point in the plane with integer coordinates
structure LatticePoint (x y : Int) : Type :=
  mk :: (x : Int) (y : Int)

-- Define a polygon with given number of vertices which are lattice points
def RegularNgon (n : Nat) (vertices : Fin n -> LatticePoint) : Prop := sorry

-- The main theorem stating the non-existence of such a polygon
theorem no_regular_ngon_with_lattice_vertices (n : Nat) (hn : n ≥ 7) :
  ¬ ∃ vertices : Fin n -> LatticePoint, RegularNgon n vertices :=
by sorry

end no_regular_ngon_with_lattice_vertices_l447_447188


namespace sum_of_digits_base5_1024_l447_447322

theorem sum_of_digits_base5_1024 :
  let base5_digits := [1, 3, 0, 4, 4] in
  base5_digits.sum = 12 :=
by
  sorry

end sum_of_digits_base5_1024_l447_447322


namespace geometric_sequence_sum_63_l447_447967

theorem geometric_sequence_sum_63
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_init : a 1 = 1)
  (h_recurrence : ∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 63 :=
by
  sorry

end geometric_sequence_sum_63_l447_447967


namespace joe_watches_all_episodes_in_10_days_l447_447203

variable (full_seasons : ℕ)
variable (episodes_per_season : ℕ)
variable (episodes_per_day : ℕ)

def total_episodes (full_seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  full_seasons * episodes_per_season

def days_to_watch_all_episodes (total_episodes : ℕ) (episodes_per_day : ℕ) : ℕ :=
  total_episodes / episodes_per_day

theorem joe_watches_all_episodes_in_10_days :
  full_seasons = 4 → 
  episodes_per_season = 15 →
  episodes_per_day = 6 → 
  days_to_watch_all_episodes (total_episodes 4 15) 6 = 10 :=
by
  intros h_seasons h_eps_per_season h_eps_per_day
  have total_eps : total_episodes 4 15 = 60 := by simp [total_episodes, h_seasons, h_eps_per_season]
  have days : days_to_watch_all_episodes 60 6 = 10 := by simp [days_to_watch_all_episodes]
  rw [total_eps] at days
  exact days

end joe_watches_all_episodes_in_10_days_l447_447203


namespace reciprocal_sum_proof_l447_447970

-- Define the original ellipse and its transformation to curve C
def original_ellipse (x y : ℝ) : Prop := x^2 + (y^2) / 4 = 1
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the polar and rectangular equations of line l
def polar_line_l (ρ θ : ℝ) : Prop := ρ * (Real.sin θ - Real.cos θ) = 1
def rect_line_l (x y : ℝ) : Prop := y - x = 1

-- Define point M and the requirement for intersection points A and B
def point_M := (1 : ℝ, 3 : ℝ)
def intersection_points (x1 y1 x2 y2 : ℝ) : Prop :=
  rect_line_l x1 y1 ∧ curve_C x1 y1 ∧
  rect_line_l x2 y2 ∧ curve_C x2 y2

-- Define the function to calculate 1/|MA| + 1/|MB|
def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def reciprocals_sum (x1 y1 x2 y2 xM yM : ℝ) : ℝ :=
  1 / dist x1 y1 xM yM + 1 / dist x2 y2 xM yM

-- Prove that the reciprocal sum for the intersection points is 2√2/5
theorem reciprocal_sum_proof :
  ∃ x1 y1 x2 y2 : ℝ,
    intersection_points x1 y1 x2 y2 ∧ reciprocals_sum x1 y1 x2 y2 1 3 = 2 * Real.sqrt 2 / 5 := by
  sorry

end reciprocal_sum_proof_l447_447970


namespace differentiable_f_at_0_l447_447633

open Real

variables {f g : ℝ → ℝ}
variable (h0 : ∃ U ∈ 𝓝 (0 : ℝ), ∀ x ∈ U, g x ≠ 0)
variable (h1 : ContinuousAt g 0)
variable (h2 : DifferentiableAt ℝ (λ x, f x * g x) 0)
variable (h3 : DifferentiableAt ℝ (λ x, f x / g x) 0)

theorem differentiable_f_at_0 :
  DifferentiableAt ℝ f 0 :=
by
  sorry

end differentiable_f_at_0_l447_447633


namespace expression_value_l447_447535

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447535


namespace eighth_term_l447_447962

noncomputable def sequence (n : ℕ) : ℚ :=
  (-1)^n * (n : ℚ) / 2^n

theorem eighth_term :
  sequence 8 = 1 / 32 :=
by
  sorry

end eighth_term_l447_447962


namespace a_value_l447_447654

-- Definition of the operation
def star (x y : ℝ) : ℝ := x + y - x * y

-- Main theorem to prove
theorem a_value :
  let a := star 1 (star 0 1)
  a = 1 :=
by
  sorry

end a_value_l447_447654


namespace difference_of_sums_1500_l447_447318

def sum_of_first_n_odd_numbers (n : ℕ) : ℕ :=
  n * n

def sum_of_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_of_sums_1500 :
  sum_of_first_n_even_numbers 1500 - sum_of_first_n_odd_numbers 1500 = 1500 :=
by
  sorry

end difference_of_sums_1500_l447_447318


namespace income_percentage_increase_l447_447337

theorem income_percentage_increase (b : ℝ) (a : ℝ) (h : a = b * 0.75) :
  (b - a) / a * 100 = 33.33 :=
by
  sorry

end income_percentage_increase_l447_447337


namespace value_of_x_l447_447170

theorem value_of_x (x : ℝ) (h1 : (x^2 - 4) / (x + 2) = 0) : x = 2 := by
  sorry

end value_of_x_l447_447170


namespace cube_surface_area_l447_447310

def P : ℝ × ℝ × ℝ := (6, 11, 11)
def Q : ℝ × ℝ × ℝ := (7, 7, 2)
def R : ℝ × ℝ × ℝ := (10, 2, 10)

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

noncomputable def is_equilateral_triangle (p q r: ℝ × ℝ × ℝ) : Prop :=
  distance p q = distance p r ∧ distance p r = distance q r

noncomputable def side_length (d : ℝ) : ℝ :=
  d / Real.sqrt 2

noncomputable def surface_area (a : ℝ) : ℝ :=
  6 * a^2

theorem cube_surface_area : is_equilateral_triangle P Q R → surface_area (side_length (distance P Q)) = 294 :=
by
  sorry

end cube_surface_area_l447_447310


namespace jason_seashells_remaining_l447_447983

-- Define the initial number of seashells Jason found
def initial_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given_to_tim : ℕ := 13

-- Define the number of seashells Jason now has
def seashells_now : ℕ := initial_seashells - seashells_given_to_tim

-- The theorem to prove: 
theorem jason_seashells_remaining : seashells_now = 36 := 
by
  -- Proof steps will go here
  sorry

end jason_seashells_remaining_l447_447983


namespace fx_expression_solve_inequality_l447_447876

-- Define the function condition
def fx_condition (f : ℝ → ℝ) := ∀ x : ℝ, 3 * f (2 - x) - 2 * f x = x^2 - 2 * x

-- Prove the analytical expression of the function
theorem fx_expression (f : ℝ → ℝ) (h : fx_condition f) : f = λ x, x^2 - 2 * x :=
sorry

-- Inequality solution as a theorem
theorem solve_inequality (a : ℝ) : 
  let f := λ x, x^2 - 2 * x in
  if a > 1 then ∀ x : ℝ, f x + a > 0
  else if a = 1 then ∀ x: ℝ, x ≠ 1 -> f x + a > 0
  else ∀ x: ℝ, x > 1 + real.sqrt (1 - a) ∨ x < 1 - real.sqrt (1 - a) -> f x + a > 0 :=
sorry

end fx_expression_solve_inequality_l447_447876


namespace vectors_equality_implies_parallel_l447_447015

variable {V : Type*} [NormedAddCommGroup V] (a b : V)

-- Axiom that abides if vectors are parallel
def vectors_parallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

-- The formal Lean statement to prove the correct answer
theorem vectors_equality_implies_parallel (h : a = b) : vectors_parallel a b := by
  sorry

end vectors_equality_implies_parallel_l447_447015


namespace ice_cream_ordering_l447_447171

-- Defining the conditions
def cone_choices := 3
def flavors := 6
def max_scoops := 3
def toppings := 6
def max_toppings := 3

-- Calculations based on the conditions to assert the total number of ways to order ice cream
def number_of_ways (cone_choices max_scoops flavors toppings max_toppings : ℕ) : ℕ :=
  let scoop_choices := (finset.range (max_scoops + 1)).sum (λ k, flavors ^ k)
  let topping_choices := (finset.range (max_toppings + 1)).sum (λ k, nat.choose (toppings + k - 1) k)
  cone_choices * scoop_choices * topping_choices

-- Main theorem statement with correct answer
theorem ice_cream_ordering :
  number_of_ways cone_choices max_scoops flavors toppings max_toppings = 65016 :=
by
  rw [number_of_ways]
  sorry

end ice_cream_ordering_l447_447171


namespace bob_victory_with_one_swap_l447_447387

-- Define the problem setup in Lean statements

-- Defining the grid dimension and numbers
def n : ℕ := 2011
def total_grids : ℕ := 2010

-- Grid definition and its properties
structure Grid :=
  (entries : Array (Array ℕ))
  (strictly_increasing_rows : ∀ i j (h1 : i < n) (h2 : j < n - 1), (entries[i][j] < entries[i][j+1]))
  (strictly_increasing_cols : ∀ i j (h1 : i < n - 1) (h2 : j < n), (entries[i][j] < entries[i+1][j]))

-- Example grid and manipulation
def A₀ : Grid := 
  { entries := Array.init n (λ i => Array.init n (λ j => n * i + j + 1)),
    strictly_increasing_rows := sorry,
    strictly_increasing_cols := sorry }

def swap_elements (grid : Grid) (r1 c1 r2 c2 : ℕ) : Grid := 
  sorry -- Swapping logic maintaining grid legality would go here

-- Proving the main theorem
theorem bob_victory_with_one_swap (bob_grid alice_grids : Array Grid) 
  (h_no_identical_alice_grids : ∀ i j (hi : i < total_grids) (hj : j < total_grids) (hneq : i ≠ j), alice_grids[i] ≠ alice_grids[j]) :
  ∃ (final_bob_grid : Grid), 
    final_bob_grid ∈ (bob_grid :: List.range total_grids).map (λ k, swap_elements bob_grid 0 k (k + 1) 0) ∧ 
    ∀ (a ∈ alice_grids), ∃ i j (hi : i < n) (hj : j < n) (hk : k < total_grids), final_bob_grid.entries[j][i] = a.entries[i][k] :=
sorry

-- The proof can be dismissed for now

end bob_victory_with_one_swap_l447_447387


namespace find_f3_l447_447102

def f : ℤ → ℤ
| x := if x ≥ 6 then x - 5 else f (x + 1)

theorem find_f3 : f 3 = 1 := by
  sorry

end find_f3_l447_447102


namespace total_amount_paid_l447_447296

noncomputable def cost_per_night_per_person : ℕ := 40
noncomputable def number_of_people : ℕ := 3
noncomputable def number_of_nights : ℕ := 3

theorem total_amount_paid (cost_per_night_per_person number_of_people number_of_nights : ℕ) :
  (cost_per_night_per_person * number_of_people * number_of_nights = 360) :=
by
  have h : cost_per_night_per_person * number_of_people * number_of_nights = 40 * 3 * 3 := by rfl
  rw h
  exact rfl

end total_amount_paid_l447_447296


namespace total_amount_paid_is_correct_l447_447301

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l447_447301


namespace final_value_l447_447543

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447543


namespace coefficient_x2_expansion_l447_447269

theorem coefficient_x2_expansion: 
    let f := (x + 1)^5 * (x - 2) in
    (∀ x : ℝ, polynomial.coeff f 2 = -15) := 
by
    sorry

end coefficient_x2_expansion_l447_447269


namespace range_of_a_l447_447919

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ 2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = h x) →
  1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1 :=
sorry

end range_of_a_l447_447919


namespace expression_value_l447_447551

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447551


namespace trajectory_length_l447_447429

theorem trajectory_length (F1 F2 : Real x Real) (P : Real x Real) (d : Real) 
  (h_F1 : F1 = (-1, 0)) (h_F2 : F2 = (1, 0)) (h : d = 2) :
  dist (P, F1) + dist (P, F2) = d → dist F1 F2 = 2 := sorry

end trajectory_length_l447_447429


namespace integer_values_b_l447_447725

theorem integer_values_b (h : ∃ b : ℤ, ∀ x : ℤ, (x^2 + b * x + 5 ≤ 0) → x ∈ {x | true}):
  {b : ℤ | ∃! x : ℤ, x^2 + b * x + 5 ≤ 0}.size = 2 :=
sorry

end integer_values_b_l447_447725


namespace mandy_more_than_three_friends_l447_447810

noncomputable def stickers_given_to_three_friends : ℕ := 4 * 3
noncomputable def total_initial_stickers : ℕ := 72
noncomputable def stickers_left : ℕ := 42
noncomputable def total_given_away : ℕ := total_initial_stickers - stickers_left
noncomputable def mandy_justin_total : ℕ := total_given_away - stickers_given_to_three_friends
noncomputable def mandy_stickers : ℕ := 14
noncomputable def three_friends_stickers : ℕ := stickers_given_to_three_friends

theorem mandy_more_than_three_friends : 
  mandy_stickers - three_friends_stickers = 2 :=
by
  sorry

end mandy_more_than_three_friends_l447_447810


namespace average_cards_collected_per_day_l447_447625

-- Definitions of the problem conditions
def first_term : ℕ := 12
def common_difference : ℕ := 10
def number_of_days : ℕ := 7

-- Statement of the problem translating the mathematically equivalent proof problem
theorem average_cards_collected_per_day :
  let last_term := first_term + (number_of_days - 1) * common_difference in
  (first_term + last_term) / 2 = 42 :=
by
  sorry

end average_cards_collected_per_day_l447_447625


namespace original_polygon_sides_l447_447162

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
(n - 2) * 180

theorem original_polygon_sides (x : ℕ) (h1 : sum_of_interior_angles (2 * x) = 2160) : x = 7 :=
by
  sorry

end original_polygon_sides_l447_447162


namespace min_disks_needed_l447_447980

/-- A theorem to determine the smallest number of disks required to store 30 files 
    given specific file sizes and disk capacity. 
-/
theorem min_disks_needed 
  (total_files : ℕ := 30)
  (disk_capacity : ℝ := 1.44)
  (file_size1_num : ℕ := 3) (file_size1 : ℝ := 0.8)
  (file_size2_num : ℕ := 12) (file_size2 : ℝ := 0.7)
  (file_size3_num : ℕ := 15) (file_size3 : ℝ := 0.4) :
  ∃ min_disks : ℕ, min_disks = 13 ∧ 
    ∀ n, n ≥ 13 → 
      (total_files = file_size1_num + file_size2_num + file_size3_num) ∧
      (file_size1_num * file_size1) + (file_size2_num * file_size2) + (file_size3_num * file_size3) ≤ n * disk_capacity :=
begin
  -- Proof would be here, but is replaced with sorry.
  sorry
end

end min_disks_needed_l447_447980


namespace probability_of_exactly_three_blue_marbles_l447_447419

-- Define the conditions
def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def total_selections : ℕ := 6
def blue_selections : ℕ := 3
def blue_probability : ℚ := 8 / 15
def red_probability : ℚ := 7 / 15
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability formula calculation
def binomial_probability : ℚ :=
  binomial_coefficient total_selections blue_selections * (blue_probability ^ blue_selections) * (red_probability ^ (total_selections - blue_selections))

-- The hypothesis (conditions) and conclusion (the solution)
theorem probability_of_exactly_three_blue_marbles :
  binomial_probability = (3512320 / 11390625) :=
by sorry

end probability_of_exactly_three_blue_marbles_l447_447419


namespace fuel_consumption_l447_447237

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end fuel_consumption_l447_447237


namespace difference_two_smallest_integers_l447_447309

theorem difference_two_smallest_integers :
  let lcm_val := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  lcm_val = 420 →
  ∀ (n₁ n₂ : ℕ),
    n₁ > 1 ∧ n₁ % 2 = 2 ∧ n₁ % 3 = 2 ∧ n₁ % 4 = 2 ∧ n₁ % 5 = 2 ∧ n₁ % 6 = 2 ∧ n₁ % 7 = 2 →
    n₂ > n₁ ∧ n₂ % 2 = 2 ∧ n₂ % 3 = 2 ∧ n₂ % 4 = 2 ∧ n₂ % 5 = 2 ∧ n₂ % 6 = 2 ∧ n₂ % 7 = 2 →
    (n₂ - n₁) = 420
| lcm_val, LCM, n₁, n₂, h₁, h₂ => sorry

end difference_two_smallest_integers_l447_447309


namespace distinct_placements_of_two_pieces_l447_447589

-- Definitions of the conditions
def grid_size : ℕ := 3
def cell_count : ℕ := grid_size * grid_size
def pieces_count : ℕ := 2

-- The theorem statement
theorem distinct_placements_of_two_pieces : 
  (number_of_distinct_placements : ℕ) = 10 := by
  -- Proof goes here with calculations and accounting for symmetry
  sorry

end distinct_placements_of_two_pieces_l447_447589


namespace ellipse_problem_l447_447907

theorem ellipse_problem
  (a b : ℝ)
  (h1 : 0 < b ∧ b < a)
  (x y : ℝ)
  (h2 : (x^2 / a^2) + (y^2 / b^2) = 1)
  (e : ℝ)
  (h3 : e = sqrt 3 / 2)
  (α β : ℝ)
  (h4 : tan α = y / (x + a))
  (h5 : tan β = y / (x - a)) :
  (cos (α - β)) / (cos (α + β)) = 3 / 5 := sorry

end ellipse_problem_l447_447907


namespace cos_2alpha_value_l447_447482

def cos_double_angle {α : Real} (cos_α : Real) : Real := 2 * cos_α^2 - 1

theorem cos_2alpha_value (α : Real) (x y : Real)
  (h_origin : (0, 0))
  (h_initial_side : x >= 0)
  (P : -1, 3)
  (cos_α : cos α = x / sqrt (x^2 + y^2)) :
  cos_double_angle (cos α) = -4 / 5 :=
by
  sorry

end cos_2alpha_value_l447_447482


namespace triangle_ratio_l447_447687

theorem triangle_ratio (a x y : ℝ) (h1 : 0 < a) (h2 : 0 < x) (h3 : 0 < y)
  (hAB : a = 64 * x)
  (hAY : 2 * y = a) :
  let p := 2, q := 1 in p + q = 3 :=
by
  let p := 2
  let q := 1
  have h := hAY
  sorry

end triangle_ratio_l447_447687


namespace total_volume_correct_l447_447774

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2812

-- Define the target volume
def total_volume_of_water : ℕ := 11248

-- The theorem to be proved
theorem total_volume_correct : volume_of_hemisphere * number_of_hemispheres = total_volume_of_water :=
by
  sorry

end total_volume_correct_l447_447774


namespace proof_equilateral_inscribed_circle_l447_447193

variables {A B C : Type*}
variables (r : ℝ) (D : ℝ)

def is_equilateral_triangle (A B C : Type*) : Prop := 
  -- Define the equilateral condition, where all sides are equal
  true

def is_inscribed_circle_radius (D r : ℝ) : Prop := 
  -- Define the property that D is the center and r is the radius 
  true

def distance_center_to_vertex (D r x : ℝ) : Prop := 
  x = 3 * r

theorem proof_equilateral_inscribed_circle 
  (A B C : Type*) 
  (r D : ℝ) 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_inscribed_circle_radius D r) : 
  distance_center_to_vertex D r (1 / 16) :=
by sorry

end proof_equilateral_inscribed_circle_l447_447193


namespace min_balls_to_ensure_20_l447_447359

theorem min_balls_to_ensure_20 (red green yellow blue purple white black : ℕ) (Hred : red = 30) (Hgreen : green = 25) (Hyellow : yellow = 18) (Hblue : blue = 15) (Hpurple : purple = 12) (Hwhite : white = 10) (Hblack : black = 7) :
  ∀ n, n ≥ 101 → (∃ r g y b p w bl, r + g + y + b + p + w + bl = n ∧ (r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ p ≥ 20 ∨ w ≥ 20 ∨ bl ≥ 20)) :=
by
  intro n hn
  sorry

end min_balls_to_ensure_20_l447_447359


namespace directrix_of_parabola_l447_447274

theorem directrix_of_parabola (y x : ℝ) : y^2 = 16 * x → ∃ a : ℝ, x = -a ∧ 4 * a = 16 :=
by
  intro h
  use 4
  rw [mul_comm, mul_div_cancel_left, add_eq_zero]
  sorry

end directrix_of_parabola_l447_447274


namespace strange_clock_time_l447_447057

/-- Given a peculiar clock with specific characteristics, the displayed time is calculated. -/
theorem strange_clock_time (A B C : ℕ) (hand_length_eq : A = B = C) 
(no_numbers : true) (unclear_top : true) 
(A_at_hour_mark : ∃ n : ℕ, A = n * (60 / 12)) 
(B_at_hour_mark : ∃ m : ℕ, B = m * (60 / 12)) 
(C_slightly_off : ∀ k : ℕ, ¬(C = k * (60 / 12))) :
  (time_displayed = "4:50") :=
sorry

end strange_clock_time_l447_447057


namespace part1_solution_set_part2_inequality_l447_447485

def f (x m : ℝ) := |x - m| + |x + 1/m|

theorem part1_solution_set (x : ℝ) : 
  (f x 2) > 3 ↔ x < -3/4 ∨ x > 9/4 := sorry

theorem part2_inequality (x m : ℝ) (h1 : 1 < m) :
  f x m + 1 / (m * (m - 1)) ≥ 3 := sorry

end part1_solution_set_part2_inequality_l447_447485


namespace conditional_probability_l447_447708

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end conditional_probability_l447_447708


namespace determine_identities_l447_447243

-- Definitions of knight and liar
def Knight (p : Prop) : Prop := p
def Liar (p : Prop) : Prop := ¬p

-- Definitions of A and B
variables {A_statement : Prop} {A_is_liar B_is_liar : Prop}

-- Assume A's statement is "At least one of us is a liar"
def A_statement := A_is_liar ∨ B_is_liar

-- Problem statement: Determine the identities of A and B
theorem determine_identities (A_is_knight : Knight A_statement) : (Knight A_is_liar) ∧ (Liar B_is_liar) :=
by
  sorry

end determine_identities_l447_447243


namespace percentage_not_working_on_either_l447_447181

theorem percentage_not_working_on_either (total : ℕ) (nA : ℕ) (nB : ℕ) (nA_and_nB : ℕ) :
  (total = 150) →
  (nA = 90) →
  (nB = 50) →
  (nA_and_nB = 30) →
  ((total - (nA + nB - nA_and_nB)) * 100 / total = 26.67) :=
by
  intros htotal hnA hnB hnA_and_nB
  rw [htotal, hnA, hnB, hnA_and_nB]
  norm_num
  sorry

end percentage_not_working_on_either_l447_447181


namespace product_approximation_l447_447399

theorem product_approximation :
    |3.57 * 9.052 * (6.18 + 3.821) - 315| < 100 :=
by
  sorry

end product_approximation_l447_447399


namespace chord_line_equation_l447_447921

/-- 
  Given the parabola y^2 = 4x and a chord AB 
  that exactly bisects at point P(1,1), prove 
  that the equation of the line on which chord AB lies is 2x - y - 1 = 0.
-/
theorem chord_line_equation (x y : ℝ) 
  (hx : y^2 = 4 * x)
  (bisect : ∃ A B : ℝ × ℝ, 
             (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2) ∧
             (A.1 + B.1 = 2 * 1) ∧ (A.2 + B.2 = 2 * 1)) :
  2 * x - y - 1 = 0 := sorry

end chord_line_equation_l447_447921


namespace terminal_side_is_second_quadrant_l447_447116

open Real
open Trigonometry

-- Define the conditions
def sin_positive (α : ℝ) : Prop := sin α > 0
def tan_negative (α : ℝ) : Prop := tan α < 0

-- The problem statement
theorem terminal_side_is_second_quadrant
  (α : ℝ) (h1 : sin_positive α) (h2 : tan_negative α) : 
  Quadrant (angle α) = Quadrant.two := sorry

end terminal_side_is_second_quadrant_l447_447116


namespace max_a_condition_range_a_condition_l447_447136

-- Definitions of the functions f and g
def f (x a : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Problem (I)
theorem max_a_condition (a : ℝ) :
  (∀ x, g x ≤ 5 → f x a ≤ 6) → a ≤ 1 :=
sorry

-- Problem (II)
theorem range_a_condition (a : ℝ) :
  (∀ x, f x a + g x ≥ 3) → a ≥ 2 :=
sorry

end max_a_condition_range_a_condition_l447_447136


namespace percentage_of_salt_correct_l447_447978

-- Define the conditions
def amount_of_seawater := 2000 -- in ml
def amount_of_salt := 400 -- in ml

-- Define the expected percentage of salt
def expected_percentage_salt := 20 -- in percentage

-- Prove that given the conditions, the percentage of salt is as expected
theorem percentage_of_salt_correct (amount_of_seawater amount_of_salt : ℤ) (h1 : amount_of_seawater = 2000) (h2 : amount_of_salt = 400) :
  (amount_of_salt * 100 / amount_of_seawater) = expected_percentage_salt := 
by
  rw [h1, h2]
  -- The final percentage calculation
  /- simplify fraction 400/2000 and multiply by 100 -/
  have h3 : 400 * 100 / 2000 = 20 := by norm_num
  exact h3

end percentage_of_salt_correct_l447_447978


namespace no_real_root_of_equation_l447_447060

theorem no_real_root_of_equation :
  ¬∃ x : ℝ, (real.sqrt (x + 9) - real.sqrt (x - 6) + 2 = 0) :=
by
  sorry

end no_real_root_of_equation_l447_447060


namespace min_val_of_f_l447_447852

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end min_val_of_f_l447_447852


namespace translate_and_min_distance_l447_447278

def f (x : ℝ) : ℝ := Real.sin (2*x)

def g (x : ℝ) : ℝ := Real.sin (2*x - Real.pi / 3)

theorem translate_and_min_distance :
  (∀ x, g(x) = Real.sin (2*x - Real.pi / 3)) ∧
  (∀ x₁ x₂, (|f(x₁) - g(x₂)| = 2) → |x₁ - x₂| = Real.pi / 2) :=
by
  sorry

end translate_and_min_distance_l447_447278


namespace expression_value_l447_447555

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447555


namespace arithmetic_sequence_general_term_find_n_given_sum_l447_447889

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : a 10 = 30)
  (h2 : a 15 = 40)
  : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d) ∧ a 10 = 30 ∧ a 15 = 40 :=
by {
  sorry
}

theorem find_n_given_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 d : ℕ)
  (h_gen : ∀ n, a n = a1 + (n - 1) * d)
  (h_sum : ∀ n, S n = n * a1 + (n * (n - 1) * d) / 2)
  (h_a1 : a1 = 12)
  (h_d : d = 2)
  (h_Sn : S 14 = 210)
  : ∃ n, S n = 210 ∧ n = 14 :=
by {
  sorry
}

end arithmetic_sequence_general_term_find_n_given_sum_l447_447889


namespace distance_between_hyperbola_vertices_l447_447080

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l447_447080


namespace projectile_highest_point_area_l447_447780

noncomputable def d (g u : ℝ) : ℝ := π / 8

theorem projectile_highest_point_area (u g : ℝ) (h_gu_pos : g > 0 ∧ u > 0) :
  ∃ d : ℝ, ∀ (φ : ℝ), 0 ≤ φ ∧ φ ≤ π →
    (∃ x y t : ℝ,
      let x := u * t * Real.cos φ in
      let y := u * t * Real.sin φ - 0.5 * g * t^2 in
      let t := u * Real.sin φ / g in
      let x := u^2 * (Real.sin φ * Real.cos φ) / g in
      let y := u^2 * (Real.sin φ)^2 / (2 * g) in
        (x / (u^2 / (2 * g)))^2 + ((y - (u^2 / (4 * g))) / (u^2 / (4 * g)))^2 = 1) →
      d = π / 8 := by
  use π / 8
  sorry

end projectile_highest_point_area_l447_447780


namespace hyperbola_vertices_distance_l447_447077

/--
For the hyperbola given by the equation
(x^2 / 121) - (y^2 / 49) = 1,
the distance between its vertices is 22.
-/
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ,
  (x^2 / 121) - (y^2 / 49) = 1 →
  ∃ a : ℝ, a = 11 ∧ 2 * a = 22 :=
by
  intros x y h
  use 11
  split
  · refl
  · norm_num

end hyperbola_vertices_distance_l447_447077


namespace min_students_with_blue_eyes_and_backpack_l447_447424

theorem min_students_with_blue_eyes_and_backpack (B P U : Finset ℕ)
  (h1 : B.card = 15) (h2 : P.card = 18) (h3 : U.card = 25) (h4 : B ⊆ U) (h5 : P ⊆ U) :
  ∃ b, b.card = B.card ∧ ∃ bp, bp.card >= 8 ∧ bp ⊆ B ∧ bp ⊆ P :=
by
  sorry

end min_students_with_blue_eyes_and_backpack_l447_447424


namespace janet_earnings_per_hour_l447_447624

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end janet_earnings_per_hour_l447_447624


namespace perpendicular_vectors_x_value_l447_447569

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (x, -1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : x = 2 :=
by
  sorry

end perpendicular_vectors_x_value_l447_447569


namespace movie_length_l447_447809

theorem movie_length (paused_midway : ∃ t : ℝ, t = t ∧ t / 2 = 30) : 
  ∃ total_length : ℝ, total_length = 60 :=
by {
  sorry
}

end movie_length_l447_447809


namespace smallest_difference_multiple_of_five_l447_447351

theorem smallest_difference_multiple_of_five : 
  ∀ (d1 d2 : ℕ), (d1 = 0 ∨ d1 = 5) ∧ (d2 = 0 ∨ d2 = 5) ∧ d1 ≠ d2 → abs (d1 - d2) = 5 :=
by
  sorry

end smallest_difference_multiple_of_five_l447_447351


namespace longest_side_of_triangle_l447_447696

theorem longest_side_of_triangle (y : ℝ) 
    (h1 : 10 + (y + 5) + (3y - 2) = 50) 
    : max 10 (max (y + 5) (3y - 2)) = 25.75 := 
by 
    sorry

end longest_side_of_triangle_l447_447696


namespace remainder_of_2n_div_11_l447_447948

theorem remainder_of_2n_div_11 (n k : ℤ) (h : n = 22 * k + 12) : (2 * n) % 11 = 2 :=
by
  sorry

end remainder_of_2n_div_11_l447_447948


namespace arithmetic_sequence_a101_eq_52_l447_447817

theorem arithmetic_sequence_a101_eq_52 (a : ℕ → ℝ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = 1 / 2) :
  a 101 = 52 :=
by
  sorry

end arithmetic_sequence_a101_eq_52_l447_447817


namespace thabo_books_l447_447761

theorem thabo_books :
  ∃ (H P F : ℕ), 
    P = H + 20 ∧ 
    F = 2 * P ∧ 
    H + P + F = 200 ∧ 
    H = 35 :=
by
  sorry

end thabo_books_l447_447761


namespace paint_left_is_zero_l447_447056

def initial_white_paint_gallons := 2
def initial_blue_paint_gallons := 1
def dexter_white_fraction := (3 : ℚ) / 8
def dexter_blue_fraction := (1 : ℚ) / 4
def jay_white_fraction := (5 : ℚ) / 8
def jay_blue_fraction := (3 : ℚ) / 4
def gallon_to_liter := 3.785

noncomputable def combined_paint_left_liters : ℝ :=
  let total_white_paint_used := (dexter_white_fraction * initial_white_paint_gallons + jay_white_fraction * initial_white_paint_gallons) * gallon_to_liter
  let total_blue_paint_used := (dexter_blue_fraction * initial_blue_paint_gallons + jay_blue_fraction * initial_blue_paint_gallons) * gallon_to_liter
  let total_white_paint_left := (initial_white_paint_gallons * gallon_to_liter) - total_white_paint_used
  let total_blue_paint_left := (initial_blue_paint_gallons * gallon_to_liter) - total_blue_paint_used
  total_white_paint_left + total_blue_paint_left

theorem paint_left_is_zero :
  combined_paint_left_liters = 0 := by
  sorry

end paint_left_is_zero_l447_447056


namespace Sarah_pool_depth_l447_447205

theorem Sarah_pool_depth (S J : ℝ) (h1 : J = 2 * S + 5) (h2 : J = 15) : S = 5 := by
  sorry

end Sarah_pool_depth_l447_447205


namespace math_problem_l447_447043

theorem math_problem :
  (π - 3.14)^0 + | -Real.sqrt 3 | - (1 / 2 : ℝ)^(-1) - Real.sin (Real.pi / 3) = -1 + Real.sqrt 3 / 2 :=
by
  sorry

end math_problem_l447_447043


namespace irrational_count_l447_447018

theorem irrational_count :
  let a := [real.sqrt 5, 0, -2.36, real.pi, real.sqrt 144, real.cbrt 6] in
  list.countp irrational a = 3 :=
by sorry

end irrational_count_l447_447018


namespace question_Ⅰ_question_Ⅱ_l447_447137

theorem question_Ⅰ (a b : ℝ) (h : ∀ x: ℝ, |x + a| < b ↔ (0 < x ∧ x < 8)) : a = -4 ∧ b = 4 :=
by 
  have h1 : -b - a = 0, from sorry,
  have h2 : b - a = 8, from sorry,
  -- We solve the system of equations:
  have := linear_system.solve h1 h2,
  cases this with ha hb,
  exact ⟨ha, hb⟩

theorem question_Ⅱ (at bt : ℝ → ℝ) (hAt : ∀ t : ℝ, at t = -4 * t + 16) (hBt : ∀ t : ℝ, bt t = 4 * t)
  : ∃ t : ℝ, t = 2 ∧ (∀ t' : ℝ, (sqrt (at t) + sqrt (bt t)) ≤ 8 ∧ (sqrt (at 2) + sqrt (bt 2) = 8)) :=
by
  let t := 2
  use t,
  split,
  -- State that t = 2
  { refl },
  -- Show max value and the inequality holds
  { split,
    { intro t',
      have h_sqrt : sqrt (at t') + sqrt (bt t') ≤ 8, from sorry,
      exact h_sqrt },
    { show sqrt (at 2) + sqrt (bt 2) = 8, from sorry } }

end question_Ⅰ_question_Ⅱ_l447_447137


namespace janet_earnings_per_hour_l447_447623

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end janet_earnings_per_hour_l447_447623


namespace find_possible_sets_C_l447_447467

open Set

def A : Set ℕ := {3, 4}
def B : Set ℕ := {0, 1, 2, 3, 4}
def possible_C_sets : Set (Set ℕ) :=
  { {3, 4}, {3, 4, 0}, {3, 4, 1}, {3, 4, 2}, {3, 4, 0, 1},
    {3, 4, 0, 2}, {3, 4, 1, 2}, {0, 1, 2, 3, 4} }

theorem find_possible_sets_C :
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B} = possible_C_sets :=
by
  sorry

end find_possible_sets_C_l447_447467


namespace michael_truck_meet_once_l447_447662

-- Define Michael's speed
def michael_speed : ℝ := 4

-- Define trash pail distance
def pail_distance : ℝ := 300

-- Define truck's speed
def truck_speed : ℝ := 12

-- Define truck's stop time at each pail
def truck_stop_time : ℝ := 40

-- Define the initial distance between Michael and the truck
def initial_distance : ℝ := 300

theorem michael_truck_meet_once :
  let cycle_time := pail_distance / truck_speed + truck_stop_time in
  let michael_travel_in_cycle := michael_speed * cycle_time in
  let truck_travel_in_cycle := pail_distance in
  (∀ t : ℝ, (t ≥ 0) →
    let michael_position := michael_speed * t in
    let truck_position := if (t % cycle_time) < (pail_distance / truck_speed)
                          then truck_speed * (t % cycle_time) + initial_distance 
                          else truck_travel_in_cycle + initial_distance in
    ∃ t1 : ℝ, ∃ t2 : ℝ, (t1 < t2 ∧ 
    michael_position = truck_position) →
    michael_position = truck_position) :=
by
  sorry

end michael_truck_meet_once_l447_447662


namespace vasya_mushrooms_l447_447742

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end vasya_mushrooms_l447_447742


namespace distance_between_hyperbola_vertices_l447_447071

theorem distance_between_hyperbola_vertices :
  (∀ x y : ℝ, (x^2 / 121) - (y^2 / 49) = 1) → 
  (∃ d : ℝ, d = 22) :=
by
  -- Assume the equation of the hyperbola
  intro hyp_eq,
  -- Use the provided information and conditions
  let a := Float.sqrt 121,
  -- The distance between the vertices is 2a
  have dist := 2 * a,
  -- Simplify a as sqrt(121) = 11
  have a_eq_11 : a = 11,
  -- Thus, distance is 2 * 11 = 22
  have dist_22 : dist = 22,
  use dist_22,
  sorry

end distance_between_hyperbola_vertices_l447_447071


namespace num_functions_with_range_1_4_l447_447493

theorem num_functions_with_range_1_4 :
  let f : ℝ → ℝ := λ x, x^2
  ∀ y ∈ {1, 4}, ∃ x ∈ {1, -1, 2, -2}, f x = y
  ∃ D ⊆ {1, -1, 2, -2}, set.range (λ x : D, x^2) = {1, 4} :=
  let domain_counts := finset.card {
    val := {1, -1, 2, -2},
    property := λ _, true
  } in
  domain_counts = 9 :=
sorry

end num_functions_with_range_1_4_l447_447493


namespace initial_value_subtract_perfect_square_l447_447369

theorem initial_value_subtract_perfect_square :
  ∃ n : ℕ, n^2 = 308 - 139 :=
by
  sorry

end initial_value_subtract_perfect_square_l447_447369


namespace quad_perimeter_l447_447777

variables {W X Y Z P : Type} [MetricSpace W] [MetricSpace X] [MetricSpace Y] [MetricSpace Z] [MetricSpace P]
          [AddCommGroup W] [AddCommGroup X] [AddCommGroup Y] [AddCommGroup Z] [AddCommGroup P]

theorem quad_perimeter (area_WXYZ : ℝ) (PW PX PY PZ : ℝ) : 
  (area_WXYZ = 3000) ∧ (PW = 30) ∧ (PX = 40) ∧ (PY = 35) ∧ (PZ = 50) → 
  perimeter_approx W X Y Z P ≈ 268.35 :=
sorry

end quad_perimeter_l447_447777


namespace proof_problem_l447_447129

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

noncomputable def isCorrect := 
  (∀ x, f (x) = 2 * Real.sin (2 * x - Real.pi / 3)) ∧
  (∃ x, x = Real.pi / 6 ∧ f (x) = 0) ∧ 
  (∀ k : ℤ, ∀ x, x ∈ [k * Real.pi + 5 * Real.pi / 12, k * Real.pi + 11 * Real.pi / 12] → 
    f.derivative x < 0)

theorem proof_problem : isCorrect := 
sorry

end proof_problem_l447_447129


namespace exists_two_numbers_sum_divisible_by_19_l447_447784

open Nat

theorem exists_two_numbers_sum_divisible_by_19 (A : Finset ℕ) (hA_card : A.card = 956) (hA_bounds : ∀ (a ∈ A), 1 ≤ a ∧ a ≤ 2014) :
  ∃ (a b ∈ A), (a + b) % 19 = 0 :=
sorry

end exists_two_numbers_sum_divisible_by_19_l447_447784


namespace min_value_of_expression_l447_447437

theorem min_value_of_expression : 
  ∃ x y : ℝ, (z = x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3) ∧ z = 1 ∧ x = 0 ∧ y = -1 :=
by
  sorry

end min_value_of_expression_l447_447437


namespace least_number_of_stamps_is_11_l447_447803

theorem least_number_of_stamps_is_11 (s t : ℕ) (h : 5 * s + 6 * t = 60) : s + t = 11 := 
  sorry

end least_number_of_stamps_is_11_l447_447803


namespace part_I_part_II_part_III_l447_447498

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ ∀ n, 2 * a (n + 1) = a n ^ 2 - 2 * a n + 4

def Sn (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range n, 1 / a (i + 1)

theorem part_I (a : ℕ → ℝ) (h : sequence a) : ∀ n, a (n + 1) > a n :=
sorry

theorem part_II (a : ℕ → ℝ) (h : sequence a) : ∀ n, a n ≥ 2 + (3 / 2) ^ (n - 1) :=
sorry

theorem part_III (a : ℕ → ℝ) (h : sequence a) (S : ℕ → ℝ) (hs : Sn a S) : 
  ∀ n, 1 - (2 / 3) ^ n ≤ S n ∧ S n < 1 :=
sorry

end part_I_part_II_part_III_l447_447498


namespace maria_needs_green_beans_l447_447231

theorem maria_needs_green_beans :
  ∀ (potatoes carrots onions green_beans : ℕ), 
  (carrots = 6 * potatoes) →
  (onions = 2 * carrots) →
  (green_beans = onions / 3) →
  (potatoes = 2) →
  green_beans = 8 :=
by
  intros potatoes carrots onions green_beans h1 h2 h3 h4
  rw [h4, Nat.mul_comm 6 2] at h1
  rw [h1, Nat.mul_comm 2 12] at h2
  rw [h2] at h3
  sorry

end maria_needs_green_beans_l447_447231


namespace integer_values_b_for_three_integer_solutions_l447_447723

theorem integer_values_b_for_three_integer_solutions (b : ℤ) :
  ¬ ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 + b * x1 + 5 ≤ 0) ∧
                     (x2^2 + b * x2 + 5 ≤ 0) ∧ (x3^2 + b * x3 + 5 ≤ 0) ∧
                     (∀ x : ℤ, x1 < x ∧ x < x3 → x^2 + b * x + 5 > 0) :=
by
  sorry

end integer_values_b_for_three_integer_solutions_l447_447723


namespace problem_statement_l447_447566

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447566


namespace find_s_l447_447218

noncomputable def s := 48

structure Polynomial (R : Type) [CommRing R] :=
(coeff : ℕ → R)
(degree : ℕ)

def monic_cubic_polynomial {R : Type} [CommRing R] (p : Polynomial R) : Prop :=
  p.degree = 3 ∧ p.coeff 3 = 1

axiom h_j_monic_cubic {R : Type} [CommRing R] (h j : Polynomial R) : monic_cubic_polynomial h ∧ monic_cubic_polynomial j

axiom h_roots (h : ℝ[X]) (s : ℝ) : 
  ∃ c : ℝ, (h = Polynomial.X - s - 2) * (Polynomial.X - s - 8) * (Polynomial.X - c)

axiom j_roots (j : ℝ[X]) (s : ℝ) : 
  ∃ d : ℝ, (j = Polynomial.X - s - 5) * (Polynomial.X - s - 11) * (Polynomial.X - d)

axiom h_minus_j_equals_2s (h j : ℝ[X]) (s : ℝ) : h - j = Polynomial.C (2 * s)

theorem find_s (h j : ℝ[X]) (s : ℝ) (hc : monic_cubic_polynomial h) (jc : monic_cubic_polynomial j) 
(h_roots : ∃ c, h = ((X - s - 2) * (X - s - 8) * (X - c))) 
(j_roots : ∃ d, j = ((X - s - 5) * (X - s - 11) * (X - d))) 
(hj_eq : h - j = Polynomial.C (2 * s)) : 
  s = 48 :=
begin
  sorry
end

end find_s_l447_447218


namespace brick_laying_days_l447_447579

theorem brick_laying_days (a m n d : ℕ) (hm : 0 < m) (hn : 0 < n) (hd : 0 < d) :
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  (a * rate_M * (d * total_days) + 2 * a * rate_N * (d * total_days)) = (a + 2 * a) :=
by
  -- Definitions from the problem conditions
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  have h0 : a * rate_M * (d * total_days) = a := sorry
  have h1 : 2 * a * rate_N * (d * total_days) = 2 * a := sorry
  exact sorry

end brick_laying_days_l447_447579


namespace cos_theta_value_l447_447146

noncomputable def vector_a : ℝ × ℝ := (3, -1)
noncomputable def vector_b : ℝ × ℝ := (2, 0)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := real.sqrt (u.1 * u.1 + u.2 * u.2)
noncomputable def cos_theta : ℝ :=
  dot_product vector_a vector_b / (magnitude vector_a * magnitude vector_b)

theorem cos_theta_value : cos_theta = 3 * real.sqrt 10 / 10 := 
sorry

end cos_theta_value_l447_447146


namespace partition_contains_ap_l447_447499

def is_arithmetic_progression (a b c : ℕ) : Prop := b - a = c - b

theorem partition_contains_ap (X : set ℕ) (P Q : set ℕ) (H : X = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (H_partition : X = P ∪ Q) (H_disjoint : P ∩ Q = ∅) :
  ∃ S ∈ {P, Q}, ∃ a ∈ S, ∃ b ∈ S, ∃ c ∈ S, is_arithmetic_progression a b c :=
sorry

end partition_contains_ap_l447_447499


namespace square_side_factor_l447_447583

theorem square_side_factor (k : ℝ) (h : k^2 = 1) : k = 1 :=
sorry

end square_side_factor_l447_447583


namespace bh_divides_fe_l447_447801

variable {α : Type*} [LinearOrderedField α]

-- Define points A, B, C, H, D, E, F, G, J
variables (A B C H D E F G J : Point α)

-- Assume the conditions from the problem
variables 
  (h1 : lies_on_line D BH)
  (h2 : intersects AD BC E)
  (h3 : intersects CD AB F)
  (h4 : is_projection_of G F AC)
  (h5 : is_projection_of J E AC)
  (h6 : area HEJ = 2 * area HFG)

theorem bh_divides_fe (BH FE : Line α) (height_ratio : α) : 
  divides_height BH FE height_ratio := 
begin
  -- The required proof that the ratio is sqrt(2) : 1
  sorry
end

end bh_divides_fe_l447_447801


namespace aq_ef_intersect_bc_l447_447994

variables {A B C H E F M Q : Type*}

-- Assume acute triangle ABC with orthocenter H
variables [is_acute_triangle A B C] [orthocenter A B C H]

-- Assume E and F are the feet of the altitudes from B and C respectively
variables [feet_of_altitudes B C E F]

-- Assume M is the midpoint of [BC]
variables [midpoint_of BC M]

-- Assume Q is the intersection of MH with circumcircle of ABC above BC
variables [circumcircle_intersection ABC M H Q]

-- The theorem we need to prove
theorem aq_ef_intersect_bc :
  ∃ P, is_intersection P (line_through A Q) (line_through E F) ∧ lies_on P (line_through B C) :=
sorry

end aq_ef_intersect_bc_l447_447994


namespace problem_statement_l447_447560

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447560


namespace range_of_d_l447_447658

variable (a b : ℕ → ℝ) (d : ℝ → ℝ)

def convex_seq (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, a (n + 1) + a (n - 1) ≤ 2 * a n

theorem range_of_d (h_arith_seq : ∀ n, b n = 2 + (n - 1) * ln d)
  (h_convex : convex_seq (λ n, (b n) / n)) :
  d ∈ set.Ici (Real.exp 2) :=
sorry

end range_of_d_l447_447658


namespace polynomial_coeff_sum_l447_447939

noncomputable def polynomial_expansion (x : ℝ) :=
  (2 * x + 3) * (4 * x^3 - 2 * x^2 + x - 7)

theorem polynomial_coeff_sum :
  let A := 8
  let B := 8
  let C := -4
  let D := -11
  let E := -21
  A + B + C + D + E = -20 :=
by
  -- The following proof steps are skipped
  sorry

end polynomial_coeff_sum_l447_447939


namespace shaded_logo_area_l447_447987

noncomputable def shaded_area (length width : ℝ) : ℝ :=
  let r := width / 4 in
  let circle_area := r^2 * Real.pi in
  let rectangle_area := length * width in
  rectangle_area - 4 * circle_area

theorem shaded_logo_area (length width : ℝ) (h_length : length = 30) (h_width : width = 15) :
  shaded_area length width = 450 - 56.25 * Real.pi :=
by
  rw [shaded_area, h_length, h_width]
  norm_num
  rw [←mul_assoc, ←mul_assoc]
  have : (15 / 4) ^ 2 * Real.pi = 14.0625 * Real.pi, by norm_num
  rw this
  norm_num
  rw sub_sub
  norm_num
  sorry

end shaded_logo_area_l447_447987


namespace find_y_l447_447873

theorem find_y :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧
    let x := -2272
    let y := 10^3 + 10^2 * c + 10 * b + a
    let z := 1
    (a * x + b * y + c * z = 1) ∧ y = 1987 :=
by
  sorry

end find_y_l447_447873


namespace sled_kinetic_friction_l447_447002

/-- Given the following conditions:
- A sled loaded with children starts from rest,
- Slides down a snowy 25° incline,
- Travels 85 meters in 17 seconds,
- Ignoring air resistance,

prove that the coefficient of kinetic friction between the sled and the slope is 0.40. -/
theorem sled_kinetic_friction
  (g : ℝ) (a : ℝ) (µ : ℝ)
  (h1 : g = 9.8)
  (h2 : a = 0.588)
  (h3 : 85 = 1 / 2 * a * 17^2)
  (h4 : ∀ m, m * g * sin (25 * π / 180) - µ * m * g * cos (25 * π / 180) = m * a) :
  µ = 0.40 :=
sorry

end sled_kinetic_friction_l447_447002


namespace sunset_time_range_l447_447009

theorem sunset_time_range (h : ℝ) :
  ¬(h ≥ 7) ∧ ¬(h ≤ 8) ∧ ¬(h ≤ 6) ↔ h ∈ Set.Ioi 8 :=
by
  sorry

end sunset_time_range_l447_447009


namespace total_hike_time_l447_447416

/-
We define the given conditions for the hike:
- flat1Distance: Distance of the first flat stretch in meters
- flat1Speed: Speed for the first flat stretch in km/h
- uphillDistance: Distance of the uphill climb in meters
- uphillSpeed: Speed for the uphill climb in km/h
- restTime: Time for the rest in minutes
- downhillDistance: Distance of the downhill trek in meters
- downhillSpeed: Speed for the downhill trek in km/h
- flat2Distance: Distance of the second flat stretch in meters
- flat2Speed: Speed for the second flat stretch in km/h 
-/

def flat1Distance := 800
def flat1Speed := 5
def uphillDistance := 400
def uphillSpeed := 3
def restTime := 10
def downhillDistance := 700
def downhillSpeed := 4
def flat2Distance := 600
def flat2Speed := 6

/-- Convert speed from km/h to m/min -/
def kmh_to_mmin (speed_kmh: ℝ) : ℝ :=
  speed_kmh * 1000 / 60

/-- Calculate time taken to cover a distance at a given speed -/
def time_taken (distance: ℝ) (speed_kmh: ℝ) : ℝ :=
  distance / (kmh_to_mmin speed_kmh)

theorem total_hike_time :
  time_taken flat1Distance flat1Speed +
  time_taken uphillDistance uphillSpeed +
  restTime +
  time_taken downhillDistance downhillSpeed +
  time_taken flat2Distance flat2Speed
  = 44.10 :=
by
  -- the theorem proof is omitted
  sorry

end total_hike_time_l447_447416


namespace sum_of_fractions_l447_447524

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447524


namespace intersection_curve_length_l447_447104

theorem intersection_curve_length (A B C D A₁ B₁ C₁ D₁ : Point) (r : ℝ) (s₁ s₂ s₃ s₄ : ℝ) (α : ℝ) :
  cube A B C D A₁ B₁ C₁ D₁ 1 →
  sphere_centered_at A r →
  r = (2 * Real.sqrt 3) / 3 →
  s₁ = (Real.sqrt 3) * π / 9 →
  s₂ = (Real.sqrt 3) * π / 6 →
  α = π / 2 →
  curve_length :=
  s₁ + s₂ = 3 * ((Real.sqrt 3 * π / 9)) + 3 * ((Real.sqrt 3 * π / 6)) →
  curve_length = (Real.sqrt 3 * π / 3) + (Real.sqrt 3 * π / 2) →
  curve_length =
  (2 * Real.sqrt 3 * π / 6) + (3 * Real.sqrt 3 * π / 6) →
  curve_length =
  (5 * Real.sqrt 3 * π / 6) :=
sorry

end intersection_curve_length_l447_447104


namespace angle_C_eq_pi_div_3_side_c_eq_7_l447_447975

theorem angle_C_eq_pi_div_3 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
  C = Real.pi / 3 :=
sorry

theorem side_c_eq_7 
  (a b c : ℝ) 
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h1a : a = 5) 
  (h1b : b = 8) 
  (h2 : C = Real.pi / 3) :
  c = 7 :=
sorry

end angle_C_eq_pi_div_3_side_c_eq_7_l447_447975


namespace houses_without_garage_nor_pool_l447_447241

def total_houses : ℕ := 85
def houses_with_garage : ℕ := 50
def houses_with_pool : ℕ := 40
def houses_with_both : ℕ := 35
def neither_garage_nor_pool : ℕ := 30

theorem houses_without_garage_nor_pool :
  total_houses - (houses_with_garage + houses_with_pool - houses_with_both) = neither_garage_nor_pool :=
by
  sorry

end houses_without_garage_nor_pool_l447_447241


namespace inscribed_triangle_min_longest_side_l447_447976

noncomputable def minimal_longest_side (ABC : Triangle) (C : Angle) (A : Angle) (BC : ℝ) : ℝ :=
  if (C = 90) ∧ (A = 30) ∧ (BC = 1) then sqrt(3 / 7) else 0

theorem inscribed_triangle_min_longest_side :
  ∀ (ABC : Triangle) (C A BC : ℝ),
  (C = 90) ∧ (A = 30) ∧ (BC = 1) →
  minimal_longest_side ABC C A BC = sqrt(3 / 7) :=
by
  intros
  sorry

end inscribed_triangle_min_longest_side_l447_447976


namespace total_yarn_length_is_1252_l447_447279

/-- Defining the lengths of the yarns according to the conditions --/
def green_yarn : ℕ := 156
def red_yarn : ℕ := 3 * green_yarn + 8
def blue_yarn : ℕ := (green_yarn + red_yarn) / 2
def average_yarn_length : ℕ := (green_yarn + red_yarn + blue_yarn) / 3
def yellow_yarn : ℕ := average_yarn_length - 12

/-- Proving the total length of the four pieces of yarn is 1252 cm --/
theorem total_yarn_length_is_1252 :
  green_yarn + red_yarn + blue_yarn + yellow_yarn = 1252 := by
  sorry

end total_yarn_length_is_1252_l447_447279


namespace max_tangent_length_within_triangle_l447_447790

-- Definitions based on given conditions
variables {ABC : Type} [triangle ABC] 
variables (p : ℝ) (perim_eq : ∃ a b c : ℝ, a + b + c = 2 * p)
variable (inscribed_circle : circle)

-- Lean statement for proving the maximum possible length of the tangent segment
theorem max_tangent_length_within_triangle : 
  ∃ (PQ : ℝ), (PQ ≤ p / 4) :=
sorry

end max_tangent_length_within_triangle_l447_447790


namespace integral_value_l447_447487

def f (x : ℝ) : ℝ := if x <= 0 then x^2 else x + 1

theorem integral_value : ∫ x in -2..2, f x = 20 / 3 := 
by
  sorry

end integral_value_l447_447487


namespace sequence_sum_from_3_to_12_l447_447804

def sequenceSum (start end : ℕ) : ℕ :=
  ∑ k in Finset.range (end - start + 1), (k + start)

theorem sequence_sum_from_3_to_12 : sequenceSum 3 12 = 65 := 
by 
  unfold sequenceSum 
  sorry

end sequence_sum_from_3_to_12_l447_447804


namespace roma_thought_l447_447255

def sum_of_digits (n : ℕ) : ℕ := 
  n.to_digits.foldl (· + ·) 0

theorem roma_thought (N : ℕ) (h1: sum_of_digits N % 8 = 0) (h2: sum_of_digits (N + 2) % 8 = 0) :
  N = 699 :=
sorry

end roma_thought_l447_447255


namespace problem_statement_l447_447518

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447518


namespace expr_is_101_l447_447766

-- Define all the relevant constants
def a := (25 / 9) ^ (1 / 2)
def b := (27 / 64) ^ (-2 / 3)
def c := (0.1) ^ (-2)
def d := (31 / 9) * (Real.pi) ^ 0
def e := Real.log 2 + Real.log 5

-- Define the expression
def expr := a + b + c - d + e

-- Prove that expr is equal to 101
theorem expr_is_101 : expr = 101 :=
by 
    sorry

end expr_is_101_l447_447766


namespace jessica_remaining_time_after_penalty_l447_447626

variable {x : ℕ}

def total_time := 90
def used_time := 15
def total_questions := 100
def answered_questions := 20
def time_per_question := 0.75
def time_penalty_per_wrong_answer := 2

theorem jessica_remaining_time_after_penalty :
  let remaining_questions := total_questions - answered_questions
  let remaining_time_without_penalty := total_time - used_time - remaining_questions * time_per_question
  let remaining_time := remaining_time_without_penalty - time_penalty_per_wrong_answer * x
  remaining_time = 15 - 2 * x :=
by
  sorry

end jessica_remaining_time_after_penalty_l447_447626


namespace sum_first_3000_terms_l447_447292

variable {α : Type*}

noncomputable def geometric_sum_1000 (a r : α) [Field α] : α := a * (r ^ 1000 - 1) / (r - 1)
noncomputable def geometric_sum_2000 (a r : α) [Field α] : α := a * (r ^ 2000 - 1) / (r - 1)
noncomputable def geometric_sum_3000 (a r : α) [Field α] : α := a * (r ^ 3000 - 1) / (r - 1)

theorem sum_first_3000_terms 
  {a r : ℝ}
  (h1 : geometric_sum_1000 a r = 1024)
  (h2 : geometric_sum_2000 a r = 2040) :
  geometric_sum_3000 a r = 3048 := 
  sorry

end sum_first_3000_terms_l447_447292


namespace total_amount_paid_is_correct_l447_447299

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l447_447299


namespace distance_origin_to_line_l447_447690

theorem distance_origin_to_line : 
    let a : ℝ := 1
    let b : ℝ := 2
    let c : ℝ := -5
    let origin_x : ℝ := 0
    let origin_y : ℝ := 0
in Real.abs( a * origin_x + b * origin_y + c ) / Real.sqrt (a * a + b * b) = Real.sqrt 5 := 
by 
    sorry

end distance_origin_to_line_l447_447690


namespace Tom_attended_games_total_l447_447100

theorem Tom_attended_games_total :
  let year1 := 4
  let year2 := 9
  let year3 := 5
  let year4 := 10
  let year5 := 6
  let year6 := 7
  year1 + year2 + year3 + year4 + year5 + year6 = 41 :=
by
  unfold year1 year2 year3 year4 year5 year6
  sorry

end Tom_attended_games_total_l447_447100


namespace expression_defined_domain_l447_447415

theorem expression_defined_domain (x : ℝ) : (x ≠ 1) ↔ (x ∈ Ioo (-∞) 1 ∪ Ioo 1 ∞) := by
  -- Proof should be here.
  sorry

end expression_defined_domain_l447_447415


namespace base_k_eq_26_l447_447582

theorem base_k_eq_26 (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 :=
by {
  -- The actual proof will go here.
  sorry
}

end base_k_eq_26_l447_447582


namespace sqrt_mul_example_complex_expression_example_l447_447401

theorem sqrt_mul_example : Real.sqrt 3 * Real.sqrt 27 = 9 :=
by sorry

theorem complex_expression_example : 
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6 :=
by sorry

end sqrt_mul_example_complex_expression_example_l447_447401


namespace bacteria_doubling_time_l447_447284

noncomputable def doubling_time_population 
    (initial final : ℝ) 
    (time : ℝ) 
    (growth_factor : ℕ) : ℝ :=
    time / (Real.log growth_factor / Real.log 2)

theorem bacteria_doubling_time :
  doubling_time_population 1000 500000 26.897352853986263 500 = 0.903 :=
by
  sorry

end bacteria_doubling_time_l447_447284


namespace sum_first_100_terms_l447_447783

def a_n (n : ℕ) : ℤ := (-1)^(n+1) * (2*n-1)

theorem sum_first_100_terms : (Finset.sum (Finset.range 100) (λ n, a_n n)) = 100 :=
by
  sorry

end sum_first_100_terms_l447_447783


namespace remainder_of_13_pow_a_mod_37_l447_447225

theorem remainder_of_13_pow_a_mod_37 (a : ℕ) (h_pos : a > 0) (h_mult : ∃ k : ℕ, a = 3 * k) : (13^a) % 37 = 1 := 
sorry

end remainder_of_13_pow_a_mod_37_l447_447225


namespace days_A_and_B_together_l447_447356

def work (W : ℝ) (days : ℝ) : ℝ := W / days

theorem days_A_and_B_together (W : ℝ) : 
  (work W 28) * 21 + (work W 40) * 10 = W → 
  (work W 28) * 21 + (work W 40) * x = W → 
  x = 10 :=
by
  intros h1 h2
  sorry

end days_A_and_B_together_l447_447356


namespace integer_solutions_l447_447746

theorem integer_solutions (x : ℤ) :
  x + 8 > 9 ∧ -3 * x > -15 → x = 2 ∨ x = 3 ∨ x = 4 :=
by
  intros h
  have h1 : x > 1 := by linarith [h.left]
  have h2 : x < 5 := by linarith [h.right]
  have hx : 1 < x ∧ x < 5 := ⟨h1, h2⟩
  interval_cases x
  all_goals { assumption }
  exfalso
  all_goals { linarith [hx.1] }
  sorry

end integer_solutions_l447_447746


namespace exists_N_consecutive_with_digit_5_l447_447229

-- Sequence definition
def sequence (n : ℕ) : ℕ := ⌊ n^((2018 : ℚ) / 2017) ⌋

-- Proof statement for the problem 
theorem exists_N_consecutive_with_digit_5 :
    ∃ N : ℕ, ∀ (k : ℕ), (k ≥ 2^2017) → ∃ i : ℕ, (i ≤ k + N) ∧ (i ≥ k) ∧ (5 ∈ (sequence i).digits 10) :=
sorry

end exists_N_consecutive_with_digit_5_l447_447229


namespace curveC_general_eq_proof_lineL_cartesian_eq_proof_alpha_value_proof_l447_447895

noncomputable def curveC_general_eq (φ : ℝ) : Prop :=
  let x := 2 * cos φ
  let y := sin φ
  (x / 2)^2 + y^2 = 1

theorem curveC_general_eq_proof : ∀ φ : ℝ, curveC_general_eq φ :=
sorry

noncomputable def lineL_cartesian_eq (α θ ρ : ℝ) : Prop :=
  ρ * sin (α - θ) = sin α ↔ let x := ρ * cos θ
                              let y := ρ * sin θ
                              y = x * tan α - tan α

theorem lineL_cartesian_eq_proof : ∀ α θ ρ : ℝ, lineL_cartesian_eq α θ ρ :=
sorry

noncomputable def alpha_value (M N : ℝ × ℝ) : Prop :=
  let PM := (M.1 - 1)^2 + M.2^2
  let PN := (N.1 - 1)^2 + N.2^2
  (abs (1 / PM - 1 / PN) = 1 / 3)

theorem alpha_value_proof : ∀ (M N : ℝ × ℝ), alpha_value M N →
  ∃ α : ℝ, α = π / 3 ∨ α = 2 * π / 3 :=
sorry

end curveC_general_eq_proof_lineL_cartesian_eq_proof_alpha_value_proof_l447_447895


namespace angle_between_vectors_45_degrees_l447_447898

open Real

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := sqrt (vec_dot v v)

noncomputable def vec_angle (v w : ℝ × ℝ) : ℝ := arccos (vec_dot v w / (vec_mag v * vec_mag w))

theorem angle_between_vectors_45_degrees 
  (e1 e2 : ℝ × ℝ)
  (h1 : vec_mag e1 = 1)
  (h2 : vec_mag e2 = 1)
  (h3 : vec_dot e1 e2 = 0)
  (a : ℝ × ℝ := (3, 0) - (0, 1))  -- (3 * e1 - e2) is represented in a direct vector form (3, -1)
  (b : ℝ × ℝ := (2, 0) + (0, 1)): -- (2 * e1 + e2) is represented in a direct vector form (2, 1)
  vec_angle a b = π / 4 :=  -- π / 4 radians is equivalent to 45 degrees
sorry

end angle_between_vectors_45_degrees_l447_447898


namespace arithmetic_sequence_common_difference_l447_447480

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℕ)
  (d : ℚ)
  (h_arith_seq : ∀ (n m : ℕ), (n > 0) → (m > 0) → (a n) / n - (a m) / m = (n - m) * d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1 / 9 ∧ a 12 = 20 :=
by 
  sorry

end arithmetic_sequence_common_difference_l447_447480


namespace point_on_line_l447_447328

theorem point_on_line (A a : Type) [IsPoint A] [IsLine a] (h : A ∈ a) : A ∈ a :=
by
  sorry

end point_on_line_l447_447328


namespace part1_part2_part3_l447_447108

-- Given conditions
variable (a p : ℕ)
variable (Sn : ℕ → ℕ)
variable (an : ℕ → ℕ)
variable (Pn : ℕ → ℕ)

-- Additional assumptions
hypothesis pos_p : p > 0
hypothesis seq_def : ∀ n : ℕ, Sn(n) = n * (an(n) - a) / 2
hypothesis S_def : ∀ n : ℕ, Sn(n) = n * (an(n) - a) / 2

-- Part (I)
theorem part1 : a = 0 := sorry

-- Part (II), general formula to prove arithmetic sequence
theorem part2 : ∀ n : ℕ, an(n) = (n - 1) * p := sorry

-- Part (III), Pn = (Sn(n+2) / Sn(n+1)) + (Sn(n+1) / Sn(n+2))
theorem part3 (n : ℕ) : (∀ n : ℕ, Pn(n) = Sn(n + 2) / Sn(n + 1) + Sn(n + 1) / Sn(n + 2) ) → ∀ n : ℕ, (Pn 1 + Pn 2 + ... + Pn n) < 2 * n + 3 := sorry

end part1_part2_part3_l447_447108


namespace three_pow_180_mod_five_eq_one_l447_447321

theorem three_pow_180_mod_five_eq_one : (3 ^ 180) % 5 = 1 := by
  -- establish the pattern
  have h1 : 3 % 5 = 3 := rfl
  have h2 : (3 ^ 2) % 5 = 4 := by
    norm_num
  have h3 : (3 ^ 3) % 5 = 2 := by
    norm_num
  have h4 : (3 ^ 4) % 5 = 1 := by
    norm_num
  -- use the pattern (3^k % 5) ≡ (3^(k mod 4) % 5)
  have h5 : (3 ^ 180) % 5 = (3 ^ (180 % 4)) % 5 := by
    exact pow_mod 3 180 5
  -- since 180 % 4 = 0
  have h6 : (180 % 4) = 0 := by
    norm_num
  rw [h6] at h5
  -- (3^0 % 5) = 1
  rw [pow_zero] at h5
  exact h5.symm

end three_pow_180_mod_five_eq_one_l447_447321


namespace lower_bound_of_square_l447_447945

theorem lower_bound_of_square (V : ℤ) (hV : V < 81) :
  ∃ k : ℤ, (k^2 > V) ∧ (k^2 < 225) ∧ ∀ n : ℤ, (n^2 > V) ∧ (n^2 < 225) → n ∈ ({-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6}) :=
sorry

end lower_bound_of_square_l447_447945


namespace lunch_break_duration_is_48_minutes_l447_447245

-- Define the conditions
def three_painters (rate_p : ℚ) (rate_h : ℚ) (lunch_break : ℚ) : Prop :=
  -- Monday's condition
  (8 - lunch_break) * (rate_p + rate_h) = 0.5 ∧
  -- Tuesday's condition
  (6.2 - lunch_break) * rate_h = 0.24 ∧
  -- Wednesday's condition
  (11.2 - lunch_break) * rate_p = 0.26

-- Define the target theorem
theorem lunch_break_duration_is_48_minutes :
  ∃ lunch_break : ℚ,
    (∃ rate_p rate_h : ℚ, three_painters rate_p rate_h lunch_break) ∧
    lunch_break * 60 = 48 :=
begin
  -- The proof will involve showing the precise calculations as in the given solution.
  sorry
end

end lunch_break_duration_is_48_minutes_l447_447245


namespace proof_of_propositions_l447_447463

namespace MathProof

def p : Prop :=
  ∀ x : ℝ, 2^x + 1 / 2^x > 2

def q : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ Real.sin x + Real.cos x = 1 / 2

def neg_p : Prop :=
  ¬p

def neg_q : Prop :=
  ¬q
  
theorem proof_of_propositions : ¬p ∧ ¬q :=
by
  -- Corresponding mathematical proof would go here
  exact sorry

end MathProof

end proof_of_propositions_l447_447463


namespace find_common_ratio_l447_447099

-- Definitions based on the conditions provided
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

def is_geometric_seq (a b c : ℝ) (q : ℝ) : Prop :=
  b * b = a * c

-- Main statement we need to prove
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  is_arithmetic_seq a →
  let S := sum_of_first_n_terms a in
  is_geometric_seq (S 1) (S 3) (S 2) q →
  q = (1 + Real.sqrt 37) / 6 ∨ q = (1 - Real.sqrt 37) / 6 :=
begin
  sorry
end

end find_common_ratio_l447_447099


namespace det_matrix_A_l447_447836

noncomputable def matrix_A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem det_matrix_A (x y z : ℝ) : 
  Matrix.det (matrix_A x y z) = x^3 + y^3 + z^3 - 3*x*y*z := by
  sorry

end det_matrix_A_l447_447836


namespace find_sum_invested_l447_447758

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end find_sum_invested_l447_447758


namespace intersection_area_l447_447732

theorem intersection_area :
  let rect := {(x, y) | (x = 2 ∧ y = 8) ∨ (x = 13 ∧ y = 8) ∨ (x = 13 ∧ y = -5) ∨ (x = 2 ∧ y = -5)},
      circle := {(x, y) | (x - 2)^2 + (y + 5)^2 = 16},
      intersection := {p | p ∈ rect ∩ circle} in
  ∃ area, area = 4 * Real.pi :=
sorry

end intersection_area_l447_447732


namespace coach_A_basketballs_count_l447_447257

-- Define the conditions
def cost_per_basketball := 29
def cost_per_baseball := 2.50
def cost_baseball_bat := 18
def coach_a_additional_cost := 237

def total_cost_coach_b : ℚ := 14 * cost_per_baseball + cost_baseball_bat
def total_cost_coach_a (x : ℕ) : ℚ := x * cost_per_basketball

-- State the theorem to prove
theorem coach_A_basketballs_count (x : ℕ) :
  total_cost_coach_a x = total_cost_coach_b + coach_a_additional_cost → x = 10 :=
begin
  sorry
end

end coach_A_basketballs_count_l447_447257


namespace range_of_a_l447_447914

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2) * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → 4 ≤ a ∧ a < 8 := 
sorry

end range_of_a_l447_447914


namespace mary_earns_per_home_l447_447232

theorem mary_earns_per_home :
  let total_earned := 12696
  let homes_cleaned := 276.0
  total_earned / homes_cleaned = 46 :=
by
  sorry

end mary_earns_per_home_l447_447232


namespace impossible_to_regenerate_consecutive_integers_l447_447715

def consecutive_integers (a : ℤ) (n : ℕ) : list ℤ :=
(list.range (2 * n)).map (λ i, a + i)

def operation (xs : list ℤ) : list ℤ :=
(xs.zip xs.tail).map (λ ⟨x, y⟩, [x - y, x + y]).join

theorem impossible_to_regenerate_consecutive_integers (n : ℕ) (a : ℤ) :
  ∀ xs, xs = consecutive_integers a n →
    ∀ xs', xs' = operation xs → 
      xs' ≠ consecutive_integers a n :=
by
  intros xs h1 xs' h2
  sorry

end impossible_to_regenerate_consecutive_integers_l447_447715


namespace find_s_t_u_sum_l447_447998

-- Defining mutually orthogonal unit vectors
variables {a b c d : ℝ^3}
variables (s t u : ℝ)

-- Conditions given in the problem
axiom h1 : a = s * (b × c) + t * (c × d) + u * (d × b)
axiom h2 : b ⬝ c × d = 1
axiom h3 : orthogonal ℝ b c
axiom h4 : orthogonal ℝ b d
axiom h5 : orthogonal ℝ c d
axiom h6 : ∥a∥ = 1
axiom h7 : ∥b∥ = 1
axiom h8 : ∥c∥ = 1
axiom h9 : ∥d∥ = 1

-- Proving the final statement
theorem find_s_t_u_sum : s + t + u = 0 := by sorry

end find_s_t_u_sum_l447_447998


namespace range_of_a_l447_447913

-- Define the piecewise function f
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 2 then -x^2 + 4*x
  else Real.log x / Real.log 2 - a

-- State the main theorem
theorem range_of_a :
  (∃ (a : ℝ), ∃! (x : ℝ), f x a = 0) ↔ 1 < a :=
sorry

end range_of_a_l447_447913


namespace find_product_of_slopes_l447_447910

-- Define the conditions of the ellipse and related parameters.
def ellipse (a b : ℝ) := a > b ∧ b > 0 ∧ a^2 = 2 * b^2

-- Define the divisor points Mₙ as described
def M_points (a : ℝ) (n : ℕ) := (2 * a * (n - 1008 : ℕ) / 2016, 0)

-- Define the slope and intersections.
def line_through (t : ℝ) (k : ℝ) (x : ℝ) := k * (x - t)
def x_coordinates (Mₙ : ℝ) (k : ℝ) (a : ℝ) (b : ℝ) :=
  let discriminant := 4 * Mₙ^2 * k^2 * (1 + 2 * k^2 - 2 * b^2)
  (Mₙ + a - discriminant / (8 * Mₙ * k^2 * (1 + 2 * k^2)), Mₙ + a + discriminant / (8 * Mₙ * k^2 * (1 + 2 * k^2)))

-- Define the product of slopes.
def product_of_slopes (slopes : List ℝ) : ℝ := slopes.foldl (*) 1

-- Main theorem statement.
theorem find_product_of_slopes (a b k : ℝ) (Mₙ_list : List ℕ) :
  ellipse a b →
  (M_points a).perm Mₙ_list →
  Mₙ_list.length = 4030 → 
  product_of_slopes
    (Mₙ_list.map (λ n, line_through (M_points a n).1 k ((x_coordinates (M_points a n).1 k a b).1) / 
                          line_through (M_points a n).1 k ((x_coordinates (M_points a n).1 k a b).2)))
    = -2 ^ -2015 :=
sorry

end find_product_of_slopes_l447_447910


namespace sum_of_fractions_l447_447520

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447520


namespace general_term_formula_l447_447891

-- Define the positive geometric sequence and given conditions
variables {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = q * a n 

noncomputable def q : ℝ :=
2

axiom pos_terms : ∀ n, 0 < a n
axiom first_term : a 1 = 1
axiom sum_first_three : a 1 + a 2 + a 3 = 7

-- Statement to prove:
theorem general_term_formula :
  (∀ n, a n = q ^ (n - 1)) :=
begin
  sorry
end

end general_term_formula_l447_447891


namespace combined_travel_time_l447_447230

noncomputable def total_travel_time (luke_bus_time : ℕ) (paula_factor : ℚ) (bike_factor : ℕ) : ℕ :=
  let paula_bus_time := paula_factor * luke_bus_time
  let luke_bike_time := bike_factor * luke_bus_time
  let luke_total_time := luke_bus_time + luke_bike_time
  let paula_total_time := paula_bus_time + paula_bus_time
  luke_total_time + paula_total_time

theorem combined_travel_time :
  total_travel_time 70 (3/5 : ℚ) 5 = 504 := 
by 
  -- conditions
  have h_luke_bus : 70 = 70 := rfl
  have h_paula_factor : (3/5 : ℚ) = 3/5 := rfl
  have h_bike_factor : 5 = 5 := rfl
  -- proof
  sorry

end combined_travel_time_l447_447230


namespace orthogonal_vectors_y_l447_447425

theorem orthogonal_vectors_y (y : ℝ) :
  let v1 := (⟨2, -4, 5⟩ : ℝ × ℝ × ℝ)
  let v2 := (⟨-3, y, 2⟩ : ℝ × ℝ × ℝ)
  (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0) → y = 1 := sorry

end orthogonal_vectors_y_l447_447425


namespace exist_n_consecutive_nonprime_l447_447222

theorem exist_n_consecutive_nonprime (n : ℕ) : ∃ k : ℕ, ∀ i : ℕ, (i < n) → ¬ nat.prime (k + i) := 
sorry

end exist_n_consecutive_nonprime_l447_447222


namespace twentieth_number_in_base_six_equals_32_l447_447183

/-- 
Convert the 20th number in base 10 to base 6 and prove it equals 32 in base 6.
--/
theorem twentieth_number_in_base_six_equals_32 :
  let n := 20 in
  let base := 6 in
  nat.to_digits base n = [3, 2] := 
by
  -- placeholder for actual proof
  sorry

end twentieth_number_in_base_six_equals_32_l447_447183


namespace Tracy_sold_paintings_l447_447738

theorem Tracy_sold_paintings (num_people : ℕ) (group1_customers : ℕ) (group1_paintings : ℕ)
    (group2_customers : ℕ) (group2_paintings : ℕ) (group3_customers : ℕ) (group3_paintings : ℕ) 
    (total_paintings : ℕ) :
    num_people = 20 →
    group1_customers = 4 →
    group1_paintings = 2 →
    group2_customers = 12 →
    group2_paintings = 1 →
    group3_customers = 4 →
    group3_paintings = 4 →
    total_paintings = (group1_customers * group1_paintings) + (group2_customers * group2_paintings) + 
                      (group3_customers * group3_paintings) →
    total_paintings = 36 :=
by
  intros 
  -- including this to ensure the lean code passes syntax checks
  sorry

end Tracy_sold_paintings_l447_447738


namespace angle_C_eq_pi_div_3_sin_A_plus_sin_B_range_l447_447928

variable (a b c A B C : ℝ)
variable (m n : ℝ × ℝ)

-- Conditions
def is_perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0
def side_lengths (a b c : ℝ) (A B C : ℝ) : Prop := A > 0 ∧ B > 0 ∧ C > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Problem 1: Prove that C = π/3
theorem angle_C_eq_pi_div_3 (h₁ : is_perpendicular (a + c, b) (a - c, b - a))
                            (h₂ : side_lengths a b c A B C) :
  C = π / 3 := 
sorry

-- Problem 2: Prove the range of values for sin A + sin B
theorem sin_A_plus_sin_B_range (h₁ : is_perpendicular (a + c, b) (a - c, b - a))
                               (h₂ : side_lengths a b c A B C)
                               (angle_C_pi_over_3 : C = π / 3) :
  (sqrt 3 / 2) < sin A + sin B ∧ sin A + sin B ≤ sqrt 3 :=
sorry

end angle_C_eq_pi_div_3_sin_A_plus_sin_B_range_l447_447928


namespace min_value_of_expression_l447_447432

theorem min_value_of_expression (n : ℕ) (h_pos : n > 0) : n = 8 → (n / 2 + 32 / n) = 8 :=
by sorry

end min_value_of_expression_l447_447432


namespace final_value_l447_447549

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447549


namespace arrange_polynomial_ascending_order_l447_447795

variable {R : Type} [Ring R] (x : R)

def p : R := 3 * x ^ 2 - x + x ^ 3 - 1

theorem arrange_polynomial_ascending_order : 
  p x = -1 - x + 3 * x ^ 2 + x ^ 3 :=
by
  sorry

end arrange_polynomial_ascending_order_l447_447795


namespace parametric_eq_correct_l447_447430

noncomputable theory
open_locale big_operators

-- Define the point M and the line's inclination angle
def M : ℝ × ℝ := (-1, 2)
def inclination_angle : ℝ := 3 * real.pi / 4

-- Define the parametric form of the line
def parametric_eq1 (t : ℝ) : ℝ × ℝ := (-1 - (t * real.sqrt 2 / 2), 2 + (t * real.sqrt 2 / 2))
def parametric_eq2 (t : ℝ) : ℝ × ℝ := (-1 - t, 2 + t)

-- The theorem we need to prove
theorem parametric_eq_correct :
  (∃ t : ℝ, parametric_eq1 t = (-1 - (t * real.sqrt 2 / 2), 2 + (t * real.sqrt 2 / 2))) ∧
  (∃ t : ℝ, parametric_eq2 t = (-1 - t, 2 + t)) :=
begin
  sorry
end

end parametric_eq_correct_l447_447430


namespace pancakes_total_l447_447037

theorem pancakes_total (bobby_pancakes : ℕ) (dog_pancakes : ℕ) (leftover_pancakes : ℕ) (total_pancakes : ℕ) :
  bobby_pancakes = 5 → dog_pancakes = 7 → leftover_pancakes = 9 → total_pancakes = bobby_pancakes + dog_pancakes + leftover_pancakes → total_pancakes = 21 :=
by
  intros
  simp [*]
  sorry

end pancakes_total_l447_447037


namespace least_element_of_set_six_non_multiple_l447_447640

theorem least_element_of_set_six_non_multiple : 
  ∀ (S : Set ℕ), 
    (∀ a b ∈ S, a < b → ¬ (b % a = 0)) ∧
    S ⊆ {n | 1 ≤ n ∧ n ≤ 12} ∧ 
    S.card = 6 → 
    ∃ k ∈ S, ∀ x ∈ S, k ≤ x ∧ k = 4 := 
by 
  sorry

end least_element_of_set_six_non_multiple_l447_447640


namespace ep_eq_x_pf_l447_447961

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C D E F M N P : V} {x y : ℝ}

-- Given conditions
def condition1 : Prop := ∃ (M : V), M ∈ open_segment ℝ A D ∧ M = (1 / (1 + x)) • A + (x / (1 + x)) • D
def condition2 : Prop := ∃ (N : V), N ∈ open_segment ℝ B C ∧ N = (1 / (1 + x)) • B + (x / (1 + x)) • C
def condition3 : Prop := ∃ (E : V), E ∈ open_segment ℝ A B ∧ E = (1 / (1 + y)) • A + (y / (1 + y)) • B
def condition4 : Prop := ∃ (F : V), F ∈ open_segment ℝ D C ∧ F = (1 / (1 + y)) • D + (y / (1 + y)) • C
def condition5 : Prop := ∃ (P : V), P ∈ open_segment ℝ M N ∧ P ∈ open_segment ℝ E F

-- Conclusion to be proved
theorem ep_eq_x_pf (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) :
  ∃ (P : V), P ∈ open_segment ℝ M N ∧ P ∈ open_segment ℝ E F ∧
  E + x • (P - E) = P := sorry

end ep_eq_x_pf_l447_447961


namespace basis_check_l447_447013

noncomputable def e1_A : ℝ × ℝ := (0, 0)
noncomputable def e2_A : ℝ × ℝ := (1, -2)
noncomputable def e1_B : ℝ × ℝ := (-1, 2)
noncomputable def e2_B : ℝ × ℝ := (5, 7)
noncomputable def e1_C : ℝ × ℝ := (2, -3)
noncomputable def e2_C : ℝ × ℝ := (1/2, -3/4)
noncomputable def e1_D : ℝ × ℝ := (3, 5)
noncomputable def e2_D : ℝ × ℝ := (6, 10)

theorem basis_check :
  ¬ collinear ℝ e1_A e2_A ∧
  ¬ collinear ℝ e1_C e2_C ∧
  ¬ collinear ℝ e1_D e2_D ∧
  (∀ (u v : ℝ × ℝ), u ≠ (0, 0) ∧ v ≠ (0, 0) → ¬ collinear ℝ u v → ∃ a b : ℝ, a • u + b • v = (0, 0) → a = 0 ∧ b = 0) :=
begin
  sorry
end

end basis_check_l447_447013


namespace problem_statement_l447_447559

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447559


namespace BK_eq_ND_l447_447972

-- Geometric definitions
variables {A B C D K M N : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space K] [metric_space M] [metric_space N]
variables (ABCD : trapezoid ABCD)
variables (AD BD CD BK ND AM : ℝ)
variables (K_point : K_point on BD)
variables (M_midpoint : M_midpoint CD)
variables (N_intersection : N_intersection AM BD)
variables (angle_BCD : angle BCD = 72)
variables (AD_eq_BD : AD = BD) (AD_eq_CD : AD = CD)
variables (AK_eq_AD : AK = AD)

-- The goal is to prove BK = ND
theorem BK_eq_ND : BK = ND :=
begin
  sorry,
end

end BK_eq_ND_l447_447972


namespace problem_statement_l447_447564

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447564


namespace perpendiculars_concurrent_l447_447180

-- Defining the right triangle ABC with right angle at C
variables {A B C D E : Type*}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E]
variables [metric_space A] [metric_space B] [metric_space C] 

-- Definition of being right-angled at C
def right_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
∃ (a b c : A), dist a b = dist b c ∧ dist b c ≠ 0 ∧ dist a c = (dist a b)^2 + (dist b c)^2

-- Triangle ABC
def triangle_ABC (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
right_triangle A B C

-- Points D and E on legs AC and CB respectively
def points_on_legs (D E : Type*) [metric_space D] [metric_space E] (A C E : Type*) : Prop :=
∃ (d e : D) (a c : A) (e : E), (dist a c ≠ 0 ∧ dist e c ≠ 0)

-- Concurrency of the feet of the perpendiculars
def concurrency_feet_perpendiculars (A B C D E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] : Prop :=
  ∃ (F G H I : Type*), dist F G = dist H I ∧ dist F H = dist G I

-- The main theorem statement
theorem perpendiculars_concurrent
  (A B C D E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (h_triangle : triangle_ABC A B C)
  (h_points_legs : points_on_legs D E A C E) :
  concurrency_feet_perpendiculars A B C D E :=
  sorry

end perpendiculars_concurrent_l447_447180


namespace num_primes_between_50_and_70_l447_447150

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71).filter (λ x, x ≥ 50)).card = 4 := by
  sorry

end num_primes_between_50_and_70_l447_447150


namespace math_problem_l447_447510

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447510


namespace find_all_f_l447_447841

noncomputable def f_solution (k : ℝ) (h : -1 / 2 ≤ k ∧ k ≤ 1) : (ℝ → ℝ) := λ x, (k / (x + 1)) + ((1 - k) / 3)

theorem find_all_f (f : ℝ → ℝ) : 
  (∀ x y z > 0, xyz = 1 → f(x + 1/y) + f(y + 1/z) + f(z + 1/x) = 1) ↔ 
  (∃ k, -1 / 2 ≤ k ∧ k ≤ 1 ∧ ∀ x, f x = (k / (x + 1)) + ((1 - k) / 3)) := sorry

end find_all_f_l447_447841


namespace work_duration_l447_447341

/-- Definition of the work problem, showing that the work lasts for 5 days. -/
theorem work_duration (work_rate_p work_rate_q : ℝ) (total_work time_p time_q : ℝ) 
  (p_work_days q_work_days : ℝ) 
  (H1 : p_work_days = 10)
  (H2 : q_work_days = 6)
  (H3 : work_rate_p = total_work / 10)
  (H4 : work_rate_q = total_work / 6)
  (H5 : time_p = 2)
  (H6 : time_q = 4 * total_work / 5 / (total_work / 2 / 3) )
  : (time_p + time_q = 5) := 
by 
  sorry

end work_duration_l447_447341


namespace problem1_problem2_problem3_problem4_problem5_problem6_l447_447402

-- Problem 1
theorem problem1 : (-20 + 3 - (-5) - 7 : Int) = -19 := sorry

-- Problem 2
theorem problem2 : (-2.4 - 3.7 - 4.6 + 5.7 : Real) = -5 := sorry

-- Problem 3
theorem problem3 : (-0.25 + ((-3 / 7) * (4 / 5)) : Real) = (-83 / 140) := sorry

-- Problem 4
theorem problem4 : ((-1 / 2) * (-8) + (-6)^2 : Real) = 40 := sorry

-- Problem 5
theorem problem5 : ((-1 / 12 - 1 / 36 + 1 / 6) * (-36) : Real) = -2 := sorry

-- Problem 6
theorem problem6 : (-1^4 + (-2) + (-1 / 3) - abs (-9) : Real) = -37 / 3 := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l447_447402


namespace calc_expr_is_45_l447_447807

-- Define the operations order following left to right rule for division and multiplication
def calc_expr : ℕ :=
  let step1 := 180 / 6 in  -- First division
  let step2 := step1 * 3 in -- Multiplication following division
  step2 / 2  -- Final division

-- State the theorem that the calculation equals to 45
theorem calc_expr_is_45 : calc_expr = 45 :=
by
  -- sorry is used to skip the actual proof
  sorry

end calc_expr_is_45_l447_447807


namespace g_at_5_l447_447695

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_5 :
  (∀ x y : ℝ, x * g y = y * g x) →
  g 10 = 25 →
  g 5 = 12.5 :=
begin
  intros h1 h2,
  sorry
end

end g_at_5_l447_447695


namespace coefficient_x2_expansion_l447_447827

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

theorem coefficient_x2_expansion : 
  binomial_coefficient 4 2 * binomial_coefficient 3 0 - binomial_coefficient 4 1 * binomial_coefficient 3 2 = -6 :=
by 
  sorry

end coefficient_x2_expansion_l447_447827


namespace part1_part2_l447_447488

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x / Real.log x + a * x

theorem part1 (h : ∀ x > 1, f x 2 ≤ f x (-2)) : a = 2 := 
sorry

theorem part2 (h : ∃ x1 x2 ∈ Set.Icc 1 Real.exp 1, x1 ≠ x2 ∧ (2 * x1 - m) * Real.log x1 + x1 = 0 ∧ (2 * x2 - m) * Real.log x2 + x2 = 0) :
  4 * Real.sqrt (Real.exp 1) < m ∧ m ≤ 3 * Real.exp 1 :=
sorry

end part1_part2_l447_447488


namespace circle_radius_condition_l447_447125

theorem circle_radius_condition (c: ℝ):
  (∃ x y : ℝ, (x^2 + y^2 + 4 * x - 2 * y - 5 * c = 0)) → c > -1 :=
by
  sorry

end circle_radius_condition_l447_447125


namespace max_integer_value_of_k_l447_447142

theorem max_integer_value_of_k :
  ∀ x y k : ℤ,
    x - 4 * y = k - 1 →
    2 * x + y = k →
    x - y ≤ 0 →
    k ≤ 0 :=
by
  intros x y k h1 h2 h3
  sorry

end max_integer_value_of_k_l447_447142


namespace hexagon_area_51_l447_447992

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (b, 2)
  let F := (-8 / Real.sqrt 3, 4)
  let AB := B - A
  let height := 8
  let base1 := 8 / Real.sqrt 3
  let base2 := 10 / Real.sqrt 3
  in ((height * base1) / 2) * 2 + (height * base2)

theorem hexagon_area_51 (b : ℝ) (h_conditions: 
  ∃ A B C D E F, 
    A = (0,0) ∧ 
    B = (b,2) ∧ 
    F = (-8 / Real.sqrt 3, 4) ∧ 
    B - A parallel D - E ∧ 
    B - C parallel E - F ∧ 
    D - C parallel F - A ∧ 
    distinct_y_coords: {0,2,4,6,8,10}
) : 
  hexagon_area b = 48 * Real.sqrt 3 :=
sorry

end hexagon_area_51_l447_447992


namespace janet_earnings_per_hour_l447_447620

def rate_per_post := 0.25  -- Janet’s rate per post in dollars
def time_per_post := 10    -- Time to check one post in seconds
def seconds_per_hour := 3600  -- Seconds in one hour

theorem janet_earnings_per_hour :
  let posts_per_hour := seconds_per_hour / time_per_post
  let earnings_per_hour := rate_per_post * posts_per_hour
  earnings_per_hour = 90 := sorry

end janet_earnings_per_hour_l447_447620


namespace distance_between_vertices_hyperbola_l447_447069

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l447_447069


namespace math_problem_l447_447507

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447507


namespace sequence_a_geometric_b_monotonically_increasing_iff_l447_447882

noncomputable def a : ℕ → ℝ
| 0     := 1  -- Note: indices in programming start from 0
| (n+1) := a n / (a n + 2)

noncomputable def b (λ : ℝ) : ℕ → ℝ
| 0     := -λ
| (n+1) := (n - λ : ℝ) * (1 / a n + 1)

theorem sequence_a_geometric : ∀ n : ℕ, 1 / a n + 1 = 2 ^ n :=
begin
  sorry -- Proof needed
end

theorem b_monotonically_increasing_iff (λ : ℝ) : 
  (∀ n : ℕ, b λ (n+1) > b λ n) ↔ (λ < 2) :=
begin
  sorry -- Proof needed
end

end sequence_a_geometric_b_monotonically_increasing_iff_l447_447882


namespace game_remaining_sprite_color_l447_447174

theorem game_remaining_sprite_color (m n : ℕ) : 
  (∀ m n : ℕ, ∃ sprite : String, sprite = if n % 2 = 0 then "Red" else "Blue") :=
by sorry

end game_remaining_sprite_color_l447_447174


namespace problem_statement_l447_447562

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447562


namespace trapezoid_concurrent_or_parallel_l447_447995

noncomputable def mid_point (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem trapezoid_concurrent_or_parallel
  (A B C D M N : ℝ × ℝ)
  (AB_parallel_CD : (B.2 - A.2) * (D.1 - C.1) = (D.2 - C.2) * (B.1 - A.1))
  (M_mid_AB : M = mid_point A B)
  (N_mid_CD : N = mid_point C D)
  (line_AD : set (ℝ × ℝ))
  (line_BC : set (ℝ × ℝ))
  (line_MN : set (ℝ × ℝ))
  (line_AD_def : line_AD = {P | ∃ (k : ℝ), P = (A.1 + k * (D.1 - A.1), A.2 + k * (D.2 - A.2))})
  (line_BC_def : line_BC = {P | ∃ (k : ℝ), P = (B.1 + k * (C.1 - B.1), B.2 + k * (C.2 - B.2))})
  (line_MN_def : line_MN = {P | ∃ (k : ℝ), P = (M.1 + k * (N.1 - M.1), M.2 + k * (N.2 - M.2))}) :
  (∃ S, (S ∈ line_AD ∧ S ∈ line_BC ∧ S ∈ line_MN)) ∨
  ∃ k, (A.1 + k * (D.1 - A.1) = B.1 + k * (C.1 - B.1) ∧ A.2 + k * (D.2 - A.2) = B.2 + k * (C.2 - B.2)) :=
sorry

end trapezoid_concurrent_or_parallel_l447_447995


namespace cement_percentage_of_second_concrete_l447_447776

theorem cement_percentage_of_second_concrete 
  (total_weight : ℝ) (final_percentage : ℝ) (partial_weight : ℝ) 
  (percentage_first_concrete : ℝ) :
  total_weight = 4500 →
  final_percentage = 0.108 →
  partial_weight = 1125 →
  percentage_first_concrete = 0.108 →
  ∃ percentage_second_concrete : ℝ, 
    percentage_second_concrete = 0.324 :=
by
  intros h1 h2 h3 h4
  let total_cement := total_weight * final_percentage
  let cement_first_concrete := partial_weight * percentage_first_concrete
  let cement_second_concrete := total_cement - cement_first_concrete
  let percentage_second_concrete := cement_second_concrete / partial_weight
  use percentage_second_concrete
  sorry

end cement_percentage_of_second_concrete_l447_447776


namespace find_ordered_triplets_l447_447765

theorem find_ordered_triplets (x y z : ℝ) :
  x^3 = z / y - 2 * y / z ∧
  y^3 = x / z - 2 * z / x ∧
  z^3 = y / x - 2 * x / y →
  (x = 1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) :=
sorry

end find_ordered_triplets_l447_447765


namespace george_painting_ways_l447_447442

def total_colors : ℕ := 10
def must_include (color : ℕ) : Prop := color = 1 -- Suppose 1 represents blue

theorem george_painting_ways (h : must_include 1) : nat.choose (total_colors - 1) 2 = 36 :=
by
  sorry

end george_painting_ways_l447_447442


namespace john_saves_money_l447_447985

theorem john_saves_money :
  let original_spending := 4 * 2
  let new_price_per_coffee := 2 + (2 * 0.5)
  let new_coffees := 4 / 2
  let new_spending := new_coffees * new_price_per_coffee
  original_spending - new_spending = 2 :=
by
  -- calculations omitted
  sorry

end john_saves_money_l447_447985


namespace inequality_solution_l447_447843

theorem inequality_solution (x : ℝ) :
  (3 * x - 8) * (x - 4) / (x + 1) > 0 ↔ 
  x ∈ Set.Ioo (Float.negInf) (-1) ∪ Set.Ioo 4 (Float.inf) := 
sorry

end inequality_solution_l447_447843


namespace first_neighborhood_barrels_l447_447381

theorem first_neighborhood_barrels :
  ∀ (x : ℕ), 
  (second_neighborhood := 2 * x) →
  (third_neighborhood := 2 * x + 100) →
  (total_used := x + second_neighborhood + third_neighborhood) →
  total_used = 1200 - 350 → 
  x = 150 :=
by
  -- The proof would go here
  sorry

end first_neighborhood_barrels_l447_447381


namespace max_value_of_z_l447_447481

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y ≤ 2) (h2 : x + y ≥ 0) (h3 : x ≤ 4) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≤ 11 :=
by
  sorry

end max_value_of_z_l447_447481


namespace part1_part2_part3_l447_447489

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 - (Real.cos x) ^ 2 - 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem part1 : f (2 * Real.pi / 3) = 2 := by
  sorry

theorem part2 : (∀ x, f x = f (x + Real.pi)) := by
  sorry

theorem part3 : ∀ k : ℤ, ∃ a b : ℝ, a = -5 * Real.pi / 6 + k * Real.pi ∧ b = -Real.pi / 3 + k * Real.pi ∧ 
  (∀ x : ℝ, a ≤ x ∧ x ≤ b → f' (x) > 0) := by
  sorry

end part1_part2_part3_l447_447489


namespace exists_infinitely_many_coprime_pairs_l447_447248

theorem exists_infinitely_many_coprime_pairs (m : ℤ) :
  ∃∞ (x y : ℤ), Int.gcd x y = 1 ∧ y ∣ (x^2 + m) ∧ x ∣ (y^2 + m) := 
sorry

end exists_infinitely_many_coprime_pairs_l447_447248


namespace cube_root_neg_eight_l447_447329

theorem cube_root_neg_eight : real.cbrt (-8) = -2 :=
by 
  -- Lean 4 proof would go here
  sorry

end cube_root_neg_eight_l447_447329


namespace prisoner_pardon_hamiltonian_l447_447372

-- Define the prison as a graph with cells as vertices and valid moves as edges
variable (prison : Type) [Fintype prison] [DecidableEq prison]

-- Define a starting cell A
variable (A : prison)

-- Two cells b and c where the maneuver occurs
variable (b c : prison)

-- Hamiltonian cycle property for the prison (exists a path visiting all vertices exactly once and returns to start)
variable (hamiltonian_cycle : ∃ p : List prison, p.nodup ∧ (p.head = some A) ∧ (p.last = some A) ∧ (∀ v ∈ p, ∃ w ∈ p, (v, w) ∈ p.zip (p.tail)) 

-- Prove Hamiltonian cycle given the specific maneuver at b -> c
theorem prisoner_pardon_hamiltonian (hb: b ∈ prison) (hc: c ∈ prison) (hbc_legal: (b,c) ∈ prison.zip prison.tail) 
: ∃ p : List prison, p.nodup ∧ (p.head = some A) ∧ (p.last = some A) := 
sorry

end prisoner_pardon_hamiltonian_l447_447372


namespace arithmetic_sequence_a7_l447_447186

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) (h3 : a 3 = 3) (h5 : a 5 = -3) : a 7 = -9 := 
sorry

end arithmetic_sequence_a7_l447_447186


namespace problem_statement_l447_447512

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447512


namespace evaluate_poly_at_2_l447_447227

def my_op (x y : ℕ) : ℕ := (x + 1) * (y + 1)
def star2 (x : ℕ) : ℕ := my_op x x

theorem evaluate_poly_at_2 :
  3 * (star2 2) - 2 * 2 + 1 = 24 :=
by
  sorry

end evaluate_poly_at_2_l447_447227


namespace divides_polynomial_l447_447652

theorem divides_polynomial (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∀ x : ℂ, (x^2 + x + 1) ∣ (x^(3 * m + 1) + x^(3 * n + 2) + 1) :=
by
  sorry

end divides_polynomial_l447_447652


namespace sandy_money_taken_l447_447256

-- Condition: Let T be the total money Sandy took for shopping, and it is known that 70% * T = $224
variable (T : ℝ)
axiom h : 0.70 * T = 224

-- Theorem to prove: T is 320
theorem sandy_money_taken : T = 320 :=
by 
  sorry

end sandy_money_taken_l447_447256


namespace remainder_of_x_div_9_l447_447326

theorem remainder_of_x_div_9 (x : ℕ) (hx_pos : 0 < x) (h : (6 * x) % 9 = 3) : x % 9 = 5 :=
by {
  sorry
}

end remainder_of_x_div_9_l447_447326


namespace percentage_of_money_spent_is_80_l447_447202

-- Define the cost of items
def cheeseburger_cost : ℕ := 3
def milkshake_cost : ℕ := 5
def cheese_fries_cost : ℕ := 8

-- Define the amount of money Jim and his cousin brought
def jim_money : ℕ := 20
def cousin_money : ℕ := 10

-- Define the total cost of the meal
def total_cost : ℕ :=
  2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money they brought
def combined_money : ℕ := jim_money + cousin_money

-- Define the percentage of combined money spent
def percentage_spent : ℕ :=
  (total_cost * 100) / combined_money

theorem percentage_of_money_spent_is_80 :
  percentage_spent = 80 :=
by
  -- proof goes here
  sorry

end percentage_of_money_spent_is_80_l447_447202


namespace find_beta_l447_447887

noncomputable def sin (x : ℝ) : ℝ := 
  sorry

noncomputable def cos (x : ℝ) : ℝ :=
  sorry

theorem find_beta (α β : ℝ) (h1 : 0 < α ∧ α < π/2)
                           (h2 : 0 < β ∧ β < π/2)
                           (h3 : sin α = √5 / 5)
                           (h4 : sin (α - β) = -√10 / 10) : 
  β = π / 4 :=
sorry

end find_beta_l447_447887


namespace straighten_river_channel_l447_447325

axiom principle_about_distances (P Q : Type) [metric_space P] [metric_space Q] (a b : P) : 
  ∃ principle : P → P → Prop, 
  (principle = λ x y, ∀ path, path(a, b) ≥ dist(a, b)) ∧ 
  (principle(a, b) = ∀ path, ∃ s t, s(a, b) = a * x.s(b) + b * x.t(a)) →
  (principle(a, b) ⟹ dist(a, b) = shortest_path(a, b)) 
 
theorem straighten_river_channel (P Q : Type) [metric_space Q] (a b : P) : 
  principle_about_distances (a, b) ↔ (shortest_path(a, b) = dist(a, b)) := 
by sorry

end straighten_river_channel_l447_447325


namespace expression_value_l447_447541

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447541


namespace find_m_prove_inequality_l447_447134

open Real

section Proof
  -- Definitions of the given conditions
  def f (x : ℝ) (m : ℝ) : ℝ := m - abs (x - 1)
  def g (x : ℝ) (m : ℝ) : ℝ := m - abs x

  -- Given conditions
  variable (m : ℝ)
  variable (a b c : ℝ)
  variable (ha : 0 < a)
  variable (hb :  0 < b )
  variable (hc : 0 < c)
  variable (h1 : g (-3) m >= 0 )
  variable (h2 : g (3) m >= 0 )
  variable (h3 : ∀ x, g x m >=0 -> x>=-3 /\ x<= 3->( g(-3) m >=0 /\ g(3) m <=0)) 
  variable (h4 : (1 : ℝ) /a + (1:ℝ)/(2*b) + (1:ℝ)/(3*c) = m)
  
  -- Prove 1: (Given that f(x + 1) >= 0 solution set being [-3, 3])
  theorem find_m :  m = 3 := by
      sorry

  
  -- Prove 2: (Given 1/a + 1/(2b) + 1/(3c) = 3 for positive a, b, c)
  theorem prove_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : (1 : ℝ) /a + (1:ℝ)/(2*b) + (1:ℝ)/(3*c) = 3 ):
     a + 2 * b + 3 * c >= 3 := by
      sorry

end Proof

end find_m_prove_inequality_l447_447134


namespace triangle_perimeter_theorem_l447_447138

-- Define the properties of the ellipse
def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

-- Define the line with the parameter λ
def line (λ : ℝ) (x y : ℝ) : Prop :=
  (1 + λ) * x + (λ - 1) * y + 2 + 2 * λ = 0

-- Define the condition λ ≠ ±1
def valid_lambda (λ : ℝ) : Prop :=
  λ ≠ 1 ∧ λ ≠ -1

-- Define the perimeter of triangle ABF
def perimeter_of_triangle (A B F : ℝ × ℝ) : ℝ :=
  (dist A B) + (dist B F) + (dist F A)

-- Define points A and B on the intersection with the ellipse
def on_ellipse (A B : ℝ × ℝ) : Prop :=
  is_ellipse A.1 A.2 ∧ is_ellipse B.1 B.2

-- Define points A and B on the line
def on_line (A B : ℝ × ℝ) (λ : ℝ) : Prop :=
  line λ A.1 A.2 ∧ line λ B.1 B.2

-- Right focus of the ellipse
def right_focus : ℝ × ℝ :=
  (2, 0)

-- The perimeter of the triangle ABF where A and B are the intersection points
-- and F is the right focus should be 16
theorem triangle_perimeter_theorem (λ : ℝ) (A B : ℝ × ℝ) (F : ℝ × ℝ) 
  (h1 : valid_lambda λ) 
  (h2 : on_ellipse A B) 
  (h3 : on_line A B λ) 
  (hF : F = right_focus) : 
  perimeter_of_triangle A B F = 16 :=
sorry

end triangle_perimeter_theorem_l447_447138


namespace proof_problem_correct_l447_447445

noncomputable def proof_problem (a b c : ℝ) : Prop :=
a > 1 ∧ b > 1 ∧ c > 1 ∧ 
real.exp(a) = 2 * a * real.exp(1 / 2) ∧
real.exp(b) = 3 * b * real.exp(1 / 3) ∧
real.exp(c) = 5 * c * real.exp(1 / 5) →
b * c * real.exp(a) < c * a * real.exp(b) 
∧ c * a * real.exp(b) < a * b * real.exp(c)

-- statement to be proved
theorem proof_problem_correct (a b c : ℝ) : proof_problem a b c :=
sorry

end proof_problem_correct_l447_447445


namespace total_paintable_area_correct_l447_447615

-- Bedroom dimensions and unoccupied wall space
def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 12
def bedroom1_height : ℕ := 9
def bedroom1_unoccupied : ℕ := 70

def bedroom2_length : ℕ := 12
def bedroom2_width : ℕ := 11
def bedroom2_height : ℕ := 9
def bedroom2_unoccupied : ℕ := 65

def bedroom3_length : ℕ := 13
def bedroom3_width : ℕ := 12
def bedroom3_height : ℕ := 9
def bedroom3_unoccupied : ℕ := 68

-- Total paintable area calculation
def calculate_paintable_area (length width height unoccupied : ℕ) : ℕ :=
  2 * (length * height + width * height) - unoccupied

-- Total paintable area of all bedrooms
def total_paintable_area : ℕ :=
  calculate_paintable_area bedroom1_length bedroom1_width bedroom1_height bedroom1_unoccupied +
  calculate_paintable_area bedroom2_length bedroom2_width bedroom2_height bedroom2_unoccupied +
  calculate_paintable_area bedroom3_length bedroom3_width bedroom3_height bedroom3_unoccupied

theorem total_paintable_area_correct : 
  total_paintable_area = 1129 :=
by
  unfold total_paintable_area
  unfold calculate_paintable_area
  norm_num
  sorry

end total_paintable_area_correct_l447_447615


namespace number_of_routes_from_A_to_B_using_each_road_exactly_once_l447_447403

-- Defining the cities and roads
inductive City
| A | B | C | D | E | F

open City

def road_connection : City → City → Prop :=
  λ x y,
    (x = A ∧ y = B) ∨ (x = A ∧ y = D) ∨ (x = A ∧ y = E) ∨ (x = B ∧ y = C) ∨
    (x = B ∧ y = D) ∨ (x = C ∧ y = D) ∨ (x = D ∧ y = E) ∨ (x = A ∧ y = F)

-- Define a route as a list of cities, ensuring all roads are used exactly once.
def valid_route (r : List City) : Prop :=
  (r.head = some A) ∧ (r.last = some B) ∧
  (∀ i, List.PrevNth r i ≠ none → road_connection (List.PrevNth r i).get! (List.at! r i)) ∧
  (List.length r = 9)  -- Since there are 8 roads, 9 cities must be visited in the route

-- Main proof statement
theorem number_of_routes_from_A_to_B_using_each_road_exactly_once : 
  ∃ r : List City, valid_route r ∧ r.perm.countp (λ p, valid_route p) = 16 := 
by
  sorry

end number_of_routes_from_A_to_B_using_each_road_exactly_once_l447_447403


namespace solution_set_inequality_system_l447_447290

theorem solution_set_inequality_system (x : ℝ) :
  (x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x) ↔ (-1 ≤ x ∧ x < 5) := by
  sorry

end solution_set_inequality_system_l447_447290


namespace decagon_diagonal_intersections_l447_447010

theorem decagon_diagonal_intersections : 
  ∃ (decagon : Type) [regular_polygon decagon 10],
  let num_points := (nat.choose 10 4) in
  num_points = 210 :=
begin
  sorry
end

end decagon_diagonal_intersections_l447_447010


namespace janet_earnings_per_hour_l447_447622

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end janet_earnings_per_hour_l447_447622


namespace solve_problem_l447_447209

-- Define the polynomial Q
def Q (x : ℝ) := x^2 - 5*x - 15

-- Define the interval [7, 17]
def interval := {x : ℝ | 7 ≤ x ∧ x ≤ 17}

-- Definitions for the values u, v, w, z, and y
def u := 1
def v := 0
def w := 0
def z := 0
def y := 10

/-- Prove that the sum of the values u, v, w, z, and y is 11 -/
theorem solve_problem :
  u + v + w + z + y = 11 :=
by simp [u, v, w, z, y]; sorry

end solve_problem_l447_447209


namespace convex_polygon_triangle_perimeter_l447_447875

theorem convex_polygon_triangle_perimeter (G : convex_polygon) :
  ∃ (v1 v2 v3 : G.vertices), perimeter (triangle v1 v2 v3) ≥ 0.7 * perimeter G :=
sorry

end convex_polygon_triangle_perimeter_l447_447875


namespace income_remaining_percentage_l447_447778

theorem income_remaining_percentage :
  let initial_income := 100
  let food_percentage := 42
  let education_percentage := 18
  let transportation_percentage := 12
  let house_rent_percentage := 55
  let total_spent := food_percentage + education_percentage + transportation_percentage
  let remaining_after_expenses := initial_income - total_spent
  let house_rent_amount := (house_rent_percentage * remaining_after_expenses) / 100
  let final_remaining_income := remaining_after_expenses - house_rent_amount
  final_remaining_income = 12.6 :=
by
  sorry

end income_remaining_percentage_l447_447778


namespace third_vertex_y_coordinate_l447_447794

theorem third_vertex_y_coordinate {x y : ℝ} 
  (h_eq : x = 1 ∨ x = 9 ∧ y = 10) 
  (h_side : dist (1, 10) (9, 10) = 8) 
  (h_equilateral : ∃ y₃ > 10, equilateral ((1, 10), (9, 10), (x, y₃)))
  (h_first_quadrant : x > 0 ∧ y > 0) :
  ∃ y₃, y₃ = 10 + 4 * (√3) := 
sorry

end third_vertex_y_coordinate_l447_447794


namespace distance_exists_l447_447834

noncomputable def equilateral_distance (side_length : ℝ) (dihedral_angle : ℝ) (distance : ℝ) : Prop :=
  let inradius := side_length * (Real.sqrt 3) / 6
  let circumradius := side_length * (Real.sqrt 3) / 3
  ∀ (A B C R S M : ℝ × ℝ × ℝ),
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2 = side_length^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 + (B.3 - C.3)^2 = side_length^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.3 - A.3)^2 = side_length^2 ∧
    (A.1 - R.1)^2 + (A.2 - R.2)^2 + (A.3 - R.3)^2 = (B.1 - R.1)^2 + 
    (B.2 - R.2)^2 + (B.3 - R.3)^2 = (C.1 - R.1)^2 + (C.2 - R.2)^2 + (C.3 - R.3)^2 ∧
    (A.1 - S.1)^2 + (A.2 - S.2)^2 + (A.3 - S.3)^2 = (B.1 - S.1)^2 + 
    (B.2 - S.2)^2 + (B.3 - S.3)^2 = (C.1 - S.1)^2 + (C.2 - S.2)^2 + (C.3 - S.3)^2 ∧
    -- Planar and geometric conditions for RAB and SAB forming a 60 degree dihedral
    -- Midpoint conditions for M and distances
    let mid_RS := (R.1 + S.1)/2, (R.2 + S.2)/2, (R.3 + S.3)/2 in
    (A.1 - mid_RS.1)^2 + (A.2 - mid_RS.2)^2 + (A.3 - mid_RS.3)^2 =
    (B.1 - mid_RS.1)^2 + (B.2 - mid_RS.2)^2 + (B.3 - mid_RS.3)^2 =
    (C.1 - mid_RS.1)^2 + (C.2 - mid_RS.2)^2 + (C.3 - mid_RS.3)^2 =
    (R.1 - mid_RS.1)^2 + (R.2 - mid_RS.2)^2 + (R.3 - mid_RS.3)^2 =
    (S.1 - mid_RS.1)^2 + (S.2 - mid_RS.2)^2 + (S.3 - mid_RS.3)^2 = distance^2

theorem distance_exists : equilateral_distance 500 60 375 := 
by {
    sorry
}

end distance_exists_l447_447834


namespace expression_equals_l447_447045

theorem expression_equals :
  (Real.pi - 3.14)^0 + | - Real.sqrt 3 | - (1 / 2)^(-1 : ℤ) - Real.sin (Real.pi / 3) = -1 + Real.sqrt 3 / 2 :=
by 
  sorry

end expression_equals_l447_447045


namespace OI_parallel_and_equal_DE_l447_447797

noncomputable theory

open_locale classical

variables {A B C D E O I : point}

-- Given conditions
variable (ABC : triangle)
variable (angle_A : ∠ A = 30) -- Triangle's angle at A
variable (DB_BC_CE : dist D B = dist B C ∧ dist B C = dist C E) -- DB = BC = CE
variable (circumcenter_O : O = circumcenter ABC) -- O is the circumcenter
variable (incenter_I : I = incenter ABC) -- I is the incenter

-- Proving the parallelism and equality of OI and DE
theorem OI_parallel_and_equal_DE (ABC : triangle) :
  OI_parallels_DE ∧ OI_equals_DE :=
sorry

end OI_parallel_and_equal_DE_l447_447797


namespace find_x_plus_y_l447_447573

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 3 * y = 27) (h2 : 3 * x + 5 * y = 1) : x + y = 31 / 17 :=
by
  sorry

end find_x_plus_y_l447_447573


namespace cafeteria_earnings_l447_447721

-- Definitions based on given conditions
def initial_apples : ℕ := 50
def initial_oranges : ℕ := 40
def apple_cost : ℝ := 0.80
def orange_cost : ℝ := 0.50
def final_apples : ℕ := 10
def final_oranges : ℕ := 6

-- Statement of the problem as a theorem
theorem cafeteria_earnings : 
  let apples_sold := initial_apples - final_apples in
  let oranges_sold := initial_oranges - final_oranges in
  let earnings_apples := apples_sold * apple_cost in
  let earnings_oranges := oranges_sold * orange_cost in
  let total_earnings := earnings_apples + earnings_oranges in
  total_earnings = 49 :=
by
  sorry

end cafeteria_earnings_l447_447721


namespace num_valid_sequences_21_l447_447935

namespace SequenceProblem

def base_values (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 0
  | 3 => 1
  | 4 => 1
  | 5 => 1
  | 6 => 2
  | 7 => 2
  | _ => 0

def f : ℕ → ℕ
| 1 => base_values 1
| 2 => base_values 2
| 3 => base_values 3
| 4 => base_values 4
| 5 => base_values 5
| 6 => base_values 6
| 7 => base_values 7
| n => if n ≥ 8 then f(n-4) + 2 * f(n-5) + f(n-6) else 0

theorem num_valid_sequences_21 : f 21 = 114 := 
  -- Here would be the proof showing that f(21) = 114
  sorry

end SequenceProblem

end num_valid_sequences_21_l447_447935


namespace max_det_bound_l447_447953

noncomputable def max_det_estimate : ℕ := 327680 * 2^16

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ)
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  abs (Matrix.det M) ≤ max_det_estimate :=
sorry

end max_det_bound_l447_447953


namespace ring_arrangement_count_l447_447468

theorem ring_arrangement_count (rings : Finset ℕ) (fingers : Finset ℕ) :
  rings.card = 10 ∧ fingers.card = 5 ∧ 
  ∃ arrangement: (Finset (Finset ℕ)), 
    (∀ a ∈ arrangement, a.card ≤ 3) ∧
    (∑ i in arrangement, i.card = 6) ∧
    (∑ i in arrangement, (i.card.factorial) * (arrangement.card.factorial) = 145152000) :=
begin
  sorry
end

end ring_arrangement_count_l447_447468


namespace area_of_circle_eq_l447_447277

-- Define the polar equation r = 3 cos θ - 4 sin θ
def polar_equation (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- Define the area of the circle represented by the polar equation
noncomputable def area_of_circle : ℝ := π * (5 / 2) ^ 2

-- Theorem stating the area of the given circle
theorem area_of_circle_eq : ∀ (r θ : ℝ), polar_equation r θ → area_of_circle = (25 / 4) * π :=
by
  intros r θ h
  unfold polar_equation at h
  sorry

end area_of_circle_eq_l447_447277


namespace sq_in_scientific_notation_l447_447475

theorem sq_in_scientific_notation (a : Real) (h : a = 25000) (h_scientific : a = 2.5 * 10^4) : a^2 = 6.25 * 10^8 :=
sorry

end sq_in_scientific_notation_l447_447475


namespace right_triangle_angle_B_l447_447631

theorem right_triangle_angle_B 
  (A B C M G I : Type) 
  [right_triangle A B C] 
  (hA : ∠A = 90)
  (hM : M ∈ segment A B)
  (h_ratio : AM / MB = 3 * real.sqrt 3 - 4)
  (h_symmetric : symmetric_point M (line G I) ∈ segment A C) : 
  ∠B = 30 :=
sorry

end right_triangle_angle_B_l447_447631


namespace number_of_tables_cost_price_l447_447272

theorem number_of_tables_cost_price
  (C S : ℝ)
  (N : ℝ)
  (h1 : N * C = 20 * S)
  (h2 : S = 0.75 * C) :
  N = 15 := by
  -- insert proof here
  sorry

end number_of_tables_cost_price_l447_447272


namespace painting_rate_is_correct_l447_447271

-- Definitions based on the conditions
def total_cost : ℝ := 343.98
def volume_of_cube : ℝ := 9261
def side_length_of_cube : ℝ := real.cbrt(volume_of_cube)
def surface_area_of_cube : ℝ := 6 * (side_length_of_cube ^ 2)
def rate_of_painting_per_sqcm : ℝ := total_cost / surface_area_of_cube

-- The statement to be proven
theorem painting_rate_is_correct : rate_of_painting_per_sqcm = 0.13 :=
sorry

end painting_rate_is_correct_l447_447271


namespace solution_set_f2x1_plus_fx_minus_b_pos_l447_447903

/-- Given function f and conditions, prove the solution set of f(2 * x - 1) + f(x - b) > 0 -/
theorem solution_set_f2x1_plus_fx_minus_b_pos (f : ℝ → ℝ)
  (b a : ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : f = (λ x, x^3 + b * x^2 + x))
  (h3 : a = -2)
  (h4 : b = 0) :
  { x : ℝ | f (2 * x - 1) + f (x - b) > 0 } = set.Ioc (1 / 3) 3 := 
by
  sorry

end solution_set_f2x1_plus_fx_minus_b_pos_l447_447903


namespace length_real_axis_l447_447920

theorem length_real_axis (x y : ℝ) : 
  (x^2 / 4 - y^2 / 12 = 1) → 4 = 4 :=
by
  intro h
  sorry

end length_real_axis_l447_447920


namespace min_log2_q_pos_geom_seq_l447_447969

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable (h_positive : ∀ n, 0 < a n)
variable (h_geom : ∃ a₀, ∀ n, a n = a₀ * q ^ n)
variable (h_a4 : a 4 = 4)

theorem min_log2_q_pos_geom_seq :
  ∀ q, (2 * a 2 + a 6) ≥ 8 * real.sqrt 2 →
  log 2 q = 1 / 4 :=
by
  sorry

end min_log2_q_pos_geom_seq_l447_447969


namespace right_triangle_construction_l447_447597

theorem right_triangle_construction (A B C P : Type)
  [right_triangle : RightAngledTriangle A B C]
  (hypotenuse : P ∈ bisector_angle A B C)
  (varphi psi : ℝ)
  (h₁ : 45 < varphi)
  (h₂ : varphi < 135)
  (h₃ : 45 < psi)
  (h₄ : ψ < 135) :
  construct_trianlge A B C P varphi ψ :=
begin
  sorry
end

end right_triangle_construction_l447_447597


namespace distance_between_hyperbola_vertices_l447_447070

theorem distance_between_hyperbola_vertices :
  (∀ x y : ℝ, (x^2 / 121) - (y^2 / 49) = 1) → 
  (∃ d : ℝ, d = 22) :=
by
  -- Assume the equation of the hyperbola
  intro hyp_eq,
  -- Use the provided information and conditions
  let a := Float.sqrt 121,
  -- The distance between the vertices is 2a
  have dist := 2 * a,
  -- Simplify a as sqrt(121) = 11
  have a_eq_11 : a = 11,
  -- Thus, distance is 2 * 11 = 22
  have dist_22 : dist = 22,
  use dist_22,
  sorry

end distance_between_hyperbola_vertices_l447_447070


namespace cafeteria_earnings_l447_447718

def apples_initial : ℕ := 50
def oranges_initial : ℕ := 40
def apple_price : ℝ := 0.80
def orange_price : ℝ := 0.50
def apples_left : ℕ := 10
def oranges_left : ℕ := 6

theorem cafeteria_earnings : 
  let apples_sold := apples_initial - apples_left,
      oranges_sold := oranges_initial - oranges_left,
      earnings_apples := (apples_sold : ℝ) * apple_price,
      earnings_oranges := (oranges_sold : ℝ) * orange_price,
      total_earnings := earnings_apples + earnings_oranges
  in total_earnings = 49 :=
by
  sorry

end cafeteria_earnings_l447_447718


namespace min_disks_needed_l447_447979

/-- A theorem to determine the smallest number of disks required to store 30 files 
    given specific file sizes and disk capacity. 
-/
theorem min_disks_needed 
  (total_files : ℕ := 30)
  (disk_capacity : ℝ := 1.44)
  (file_size1_num : ℕ := 3) (file_size1 : ℝ := 0.8)
  (file_size2_num : ℕ := 12) (file_size2 : ℝ := 0.7)
  (file_size3_num : ℕ := 15) (file_size3 : ℝ := 0.4) :
  ∃ min_disks : ℕ, min_disks = 13 ∧ 
    ∀ n, n ≥ 13 → 
      (total_files = file_size1_num + file_size2_num + file_size3_num) ∧
      (file_size1_num * file_size1) + (file_size2_num * file_size2) + (file_size3_num * file_size3) ≤ n * disk_capacity :=
begin
  -- Proof would be here, but is replaced with sorry.
  sorry
end

end min_disks_needed_l447_447979


namespace estimate_households_above_320_units_proved_l447_447590

noncomputable def estimate_households_above_320_units
    (households : ℕ)
    (μ σ : ℝ)
    (prob_interval : ℝ)
    (prob_above : ℝ)
    (h : households = 1000 ∧ μ = 300 ∧ σ = 10 ∧ prob_interval = 0.9544 ∧ prob_above = 0.0228)
    : Prop :=
  let expected_households := households * prob_above in
  expected_households = 23

theorem estimate_households_above_320_units_proved
  : estimate_households_above_320_units 1000 300 10 0.9544 0.0228 :=
by 
  sorry

end estimate_households_above_320_units_proved_l447_447590


namespace distance_between_vertices_hyperbola_l447_447068

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l447_447068


namespace probability_two_correct_packages_l447_447859

open Nat

noncomputable def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

theorem probability_two_correct_packages :
  let total_ways := factorial 5
  let choose_2 := Nat.choose 5 2
  let derange_3 := derangements 3
  (choose_2 * derange_3) / total_ways = 1 / 6 :=
by
  let total_ways := factorial 5
  let choose_2 := Nat.choose 5 2
  let derange_3 := derangements 3
  have h1 : total_ways = 120 := by simp [factorial]
  have h2 : choose_2 = 10 := by simp [Nat.choose]
  have h3 : derange_3 = 2 := by simp [derangements]
  rw [h1, h2, h3]
  norm_num
  rfl

end probability_two_correct_packages_l447_447859


namespace interest_rate_correct_l447_447856

def principal : ℝ := 958.9041095890411
def time_period : ℝ := 2.4
def amount_after_time : ℝ := 1120
def interest_earned : ℝ := 161.0958904109589
def interest_rate_per_annum : ℝ := 7 / 100

theorem interest_rate_correct :
  interest_earned = (amount_after_time - principal) ∧
  interest_earned = principal * interest_rate_per_annum * time_period :=
by
  sorry

end interest_rate_correct_l447_447856


namespace freight_train_distance_l447_447364

variable (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) 

def total_distance_traveled (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) : ℕ :=
  let traveled_distance := (time_minutes / travel_rate) 
  traveled_distance + initial_distance

theorem freight_train_distance :
  total_distance_traveled 2 5 90 = 50 :=
by
  sorry

end freight_train_distance_l447_447364


namespace sum_of_roots_l447_447580

theorem sum_of_roots (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : ∀ x : ℝ, x^2 + m * x + n = 0 → (x = m ∨ x = n)) :
  m + n = -1 :=
sorry

end sum_of_roots_l447_447580


namespace adam_clothing_ratio_l447_447383

-- Define the initial amount of clothing Adam took out
def initial_clothing_adam : ℕ := 4 + 4 + 8 + 20

-- Define the number of friends donating the same amount of clothing as Adam
def number_of_friends : ℕ := 3

-- Define the total number of clothes being donated
def total_donated_clothes : ℕ := 126

-- Define the ratio of the clothes Adam is keeping to the clothes he initially took out
def ratio_kept_to_initial (initial_clothing: ℕ) (total_donated: ℕ) (kept: ℕ) : Prop :=
  kept * initial_clothing = 0

-- Theorem statement
theorem adam_clothing_ratio :
  ratio_kept_to_initial initial_clothing_adam total_donated_clothes 0 :=
by 
  sorry

end adam_clothing_ratio_l447_447383


namespace perp_bisector_fixed_point_max_triangle_area_l447_447879

variable (p a t : ℝ) (A B : ℝ × ℝ)
variable (y1 y2 x1 x2 : ℝ)

-- Conditions
variable (hp : p > 0) (ha : a ≠ 0) (ha1 : a > 1)
variable (MidpointA_B : A = (x1, y1) ∧ B = (x2, y2) ∧ (2*a = x1 + x2) ∧ (2*t = y1 + y2))
variable (OnParabolaA : (y1)^2 = 2*p*x1)
variable (OnParabolaB : (y2)^2 = 2*p*x2)
variable (FocusF : (p/2, 0))
variable (DistanceFN : (p + a - (p/2), 0) = 2*a)

-- Part 1: Proof that the perpendicular bisector passes through a fixed point
theorem perp_bisector_fixed_point
  (h : A ≠ B) : ∃ N : ℝ × ℝ, N = (p + a, 0) ∧ ∃ l : ℝ → ℝ, ∀ P : ℝ × ℝ, l P = true → (P = A ∨ P = B) := 
sorry

-- Part 2: Proof of maximum area of triangle
theorem max_triangle_area 
  (h : |FN| = 2*a) : ∃ max_area : ℝ, max_area = (16 * sqrt 6 / 9 * a ^ 2) :=
sorry

end perp_bisector_fixed_point_max_triangle_area_l447_447879


namespace question1_question2_question3_l447_447127

-- Definitions for condition 1
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + m * x - 1

-- Problem statement for question 1
theorem question1 (m : ℝ) (h : ∀ x : ℝ, f x m = f (2 - x) m) : 
  Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) = 
  { y | ∃ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2), y = f x m } :=
sorry

-- Definitions for condition 2
def R_neg : Set ℝ := { x | x < 0 }

-- Problem statement for question 2
theorem question2 (m : ℝ) (h : R_neg ∩ { y | ∃ x : ℝ, y = f x m + 2 } = ∅) :
  -2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2 :=
sorry

-- Definitions for condition 3
def g (x a : ℝ) (m : ℝ) : ℝ := Real.abs (x - a) - x^2 - m * x

-- Problem statement for question 3
theorem question3 (a m : ℝ) : 
  (∃ x : ℝ, x^2 + Real.abs (x - a) - 1 = -a - (5 / 4)) :=
sorry

end question1_question2_question3_l447_447127


namespace factor_expression_l447_447838

theorem factor_expression (y : ℝ) : 
  5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) :=
by
  sorry

end factor_expression_l447_447838


namespace greatest_a_l447_447428

theorem greatest_a (a : ℝ) : a^2 - 14*a + 45 ≤ 0 → a ≤ 9 :=
by
  -- placeholder for the actual proof
  sorry

end greatest_a_l447_447428


namespace arithmetic_sequence_integers_l447_447602

theorem arithmetic_sequence_integers (a3 a18 : ℝ) (d : ℝ) (n : ℕ)
  (h3 : a3 = 14) (h18 : a18 = 23) (hd : d = 0.6)
  (hn : n = 2010) : 
  (∃ (k : ℕ), n = 5 * (k + 1) - 2) ∧ (k ≤ 401) :=
by
  sorry

end arithmetic_sequence_integers_l447_447602


namespace find_original_number_l447_447319

theorem find_original_number (x : ℤ) (h : (x + 5) % 23 = 0) : x = 18 :=
sorry

end find_original_number_l447_447319


namespace mass_percentage_Ba_of_mixture_l447_447367

noncomputable def molar_mass_BaOH2 : ℤ := 137327 + 16 * 2 + 1008 * 2
noncomputable def molar_mass_Ba : ℤ := 137327
noncomputable def mixture : ℤ := 25 + 15

theorem mass_percentage_Ba_of_mixture 
  (mass_BaOH2 : ℤ) 
  (mass_NaNO3 : ℤ)
  (molar_mass_BaOH2 molar_mass_Ba mixture : ℤ) : 
  (((25 / molar_mass_BaOH2) * molar_mass_Ba) / mixture) * 100 = 50.075 :=
by
  have mass_BaOH2 := 25
  have mass_NaNO3 := 15
  have molar_mass_BaOH2 := 171343
  have molar_mass_Ba := 137327
  have mixture := mass_BaOH2 + mass_NaNO3
  have percentage := ((25 / molar_mass_BaOH2) * molar_mass_Ba) / mixture * 100
  have percentage_val := 50.075
  sorry

end mass_percentage_Ba_of_mixture_l447_447367


namespace assignment_count_36_l447_447374

noncomputable def studentAssignments : Nat :=
  let choose_2_from_4 := Nat.choose 4 2
  let arrange_3_across_positions := Nat.factorial 3
  choose_2_from_4 * arrange_3_across_positions

theorem assignment_count_36 :
  ∃ n : Nat, n = studentAssignments :=
by
  let answer := 36
  sorry

end assignment_count_36_l447_447374


namespace solution_set_abs_inequality_l447_447350

theorem solution_set_abs_inequality (x : ℝ) : |3 - x| + |x - 7| ≤ 8 ↔ 1 ≤ x ∧ x ≤ 9 :=
sorry

end solution_set_abs_inequality_l447_447350


namespace infinite_set_of_pairwise_coprime_values_l447_447988

theorem infinite_set_of_pairwise_coprime_values
  (P : ℤ[X])
  (a b : ℤ)
  (h_distinct: a ≠ b)
  (h_coprime: Int.gcd (eval a P) (eval b P) = 1) :
  ∃ S : Set ℤ, S.infinite ∧ ∀ x ∈ S, ∀ y ∈ S, x ≠ y → Int.gcd (eval x P) (eval y P) = 1 :=
by
  sorry

end infinite_set_of_pairwise_coprime_values_l447_447988


namespace power_function_at_3_l447_447918

variable (a : ℝ) (α : ℝ)

-- Conditions: a > 0 and a ≠ 1
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom P_fixed : (2 : ℝ, 8 : ℝ) -- Fixed point (P(P, P))

-- Power function passing through point (2, 8) where α = 3
theorem power_function_at_3 : α = 3 → (2 : ℝ) ^ α = 8 → (3 : ℝ) ^ α = 27 := by
  sorry

end power_function_at_3_l447_447918


namespace cannot_cover_8x8_grid_l447_447251

theorem cannot_cover_8x8_grid :
  ∀ (grid : fin 8 × fin 8) (rect4x1 : fin 4 × fin 1 → Prop) (rect2x2 : fin 2 × fin 2 → Prop),
    (∃ (positions : list (fin 8 × fin 8)), 
      (positions.length = 15 ∧
        ∀ p ∈ positions, ∃ x y, rect4x1 (x,y) ∧ (grid = p + (x, y)))
      ∨
      (positions.length = 1 ∧
        ∀ p ∈ positions, ∃ x y, rect2x2 (x,y) ∧ (grid = p + (x, y)))) →
    False := 
sorry

end cannot_cover_8x8_grid_l447_447251


namespace gcd_binomial_coefficients_l447_447672

theorem gcd_binomial_coefficients (n : ℕ) : 
  (∃ (p : ℕ) (a : ℕ), nat.prime p ∧ a > 0 ∧ n = p^a ∧ nat.gcd n.choose 1 .. n.choose (n-1) = p) ∨
  (∃ (N p : ℕ) (a : ℕ), nat.prime p ∧ a > 0 ∧ n = N * p^a ∧ ¬ p ∣ N ∧ nat.gcd n.choose 1 .. n.choose (n-1) = 1) :=
sorry

end gcd_binomial_coefficients_l447_447672


namespace find_eccentricity_l447_447909

noncomputable def ellipse_gamma (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b) : Prop :=
∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def ellipse_focus (a b : ℝ) : Prop :=
∀ (x y : ℝ), x = 3 → y = 0

def vertex_A (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = b

def vertex_B (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = -b

def point_N : Prop :=
∀ (x y : ℝ), x = 12 → y = 0

theorem find_eccentricity : 
∀ (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b), 
  ellipse_gamma a b ha_gt hb_gt h → 
  ellipse_focus a b → 
  vertex_A b → 
  vertex_B b → 
  point_N → 
  ∃ e : ℝ, e = 1 / 2 := 
by 
  sorry

end find_eccentricity_l447_447909


namespace cover_square_with_three_unit_squares_l447_447199

-- Define the side length of the target square
def side_length : ℝ := 5 / 4

-- Calculate the area of the target square
def target_square_area : ℝ := side_length * side_length

-- Define the area of one unit square (side length 1)
def unit_square_area : ℝ := 1 * 1

-- Define the total area of three unit squares
def total_unit_squares_area : ℝ := 3 * unit_square_area

-- State the theorem asserting that three unit squares can cover the target square
theorem cover_square_with_three_unit_squares 
  (target_square_area : ℝ) (total_unit_squares_area : ℝ) 
  (h1 : target_square_area = side_length * side_length) 
  (h2 : total_unit_squares_area = 3 * unit_square_area) 
  (h3 : 3 * unit_square_area > target_square_area) : 
  ∃ (can_cover : Bool), can_cover = true :=
by
  -- Placeholder for the proof
  sorry

end cover_square_with_three_unit_squares_l447_447199


namespace smallest_n_l447_447001

def is_intermittent_periodic {α : Type*} (a : ℕ → α) (n : ℕ) : Prop :=
  ∃ i p : ℕ, i + 2 * p ≤ n ∧ ∀ j, 1 ≤ j ∧ j ≤ p → a (i + j) = a (i + p + j)

theorem smallest_n {k : ℕ} (hk : k > 0) :
  ∃ n : ℕ, (∀ a : ℕ → ℕ, (∀ i, i < n → a i ∈ fin k) → ¬is_intermittent_periodic a n) ∧
           (∀ a : ℕ → ℕ, (∀ i, i < n + 1 → a i ∈ fin k) → is_intermittent_periodic a (n + 1)) ∧
           n = 2 ^ k - 1 :=
by { sorry }

end smallest_n_l447_447001


namespace remainders_equal_l447_447226

theorem remainders_equal (P P' D R k s s' : ℕ) (h1 : P > P') 
  (h2 : P % D = 2 * R) (h3 : P' % D = R) (h4 : R < D) :
  (k * (P + P')) % D = s → (k * (2 * R + R)) % D = s' → s = s' :=
by
  sorry

end remainders_equal_l447_447226


namespace min_value_3x_plus_4y_l447_447454

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
(-1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem min_value_3x_plus_4y :
  ∀ (θ : ℝ), let P := curve θ in 3 * P.1 + 4 * P.2 ≥ -9 := 
by
  intros θ
  let P := curve θ
  sorry

end min_value_3x_plus_4y_l447_447454


namespace perturb_non_special_l447_447660

structure Point := (x : ℝ) (y : ℝ)

structure Heptagon := (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : Point)

def is_special_heptagon (H : Heptagon) (P : Point) : Prop :=
  let diag_1 := -- define the line from H.A₁ to H.A₄
  let diag_2 := -- define the line from H.A₂ to H.A₅
  let diag_3 := -- define the line from H.A₃ to H.A₆
  -- Check if these three diagonals intersect at P
  sorry

theorem perturb_non_special :
  ∀ (H : Heptagon) (P : Point), 
    is_special_heptagon H P →
    ∃ H' : Heptagon,
      (H ≠ H') ∧ ¬ is_special_heptagon H' P :=
by
  intros
  sorry

end perturb_non_special_l447_447660


namespace three_digit_2C4_not_multiple_of_5_l447_447098

theorem three_digit_2C4_not_multiple_of_5 : ∀ C : ℕ, C < 10 → ¬(∃ n : ℕ, 2 * 100 + C * 10 + 4 = 5 * n) :=
by
  sorry

end three_digit_2C4_not_multiple_of_5_l447_447098


namespace math_enthusiast_gender_relation_female_success_probability_l447_447361

-- Constants and probabilities
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 12
def d : ℕ := 28
def n : ℕ := 100
def P_male_success : ℚ := 3 / 4
def P_female_success : ℚ := 2 / 3
def K_threshold : ℚ := 6.635

-- Computation of K^2
def K_square : ℚ := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- The first part of the proof comparing K^2 with threshold
theorem math_enthusiast_gender_relation : K_square < K_threshold := sorry

-- The second part calculating given conditions for probability calculation
def P_A : ℚ := (P_male_success ^ 2 * (1 - P_female_success)) + (2 * (1 - P_male_success) * P_male_success * P_female_success)
def P_AB : ℚ := 2 * (1 - P_male_success) * P_male_success * P_female_success
def P_B_given_A : ℚ := P_AB / P_A

theorem female_success_probability : P_B_given_A = 4 / 7 := sorry

end math_enthusiast_gender_relation_female_success_probability_l447_447361


namespace non_monotonic_m_range_l447_447917

theorem non_monotonic_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, (3 * x^2 + 2 * x + m = 0)) →
  m ∈ Set.Ioo (-16 : ℝ) (1/3 : ℝ) :=
sorry

end non_monotonic_m_range_l447_447917


namespace cannot_divide_1980_into_four_groups_l447_447346

theorem cannot_divide_1980_into_four_groups :
  ¬∃ (S₁ S₂ S₃ S₄ : ℕ),
    S₂ = S₁ + 10 ∧
    S₃ = S₂ + 10 ∧
    S₄ = S₃ + 10 ∧
    (1 + 1980) * 1980 / 2 = S₁ + S₂ + S₃ + S₄ := 
sorry

end cannot_divide_1980_into_four_groups_l447_447346


namespace tan_x_plus_pi_over_6_parallel_interval_of_increase_perpendicular_l447_447148

-- Problem 1
theorem tan_x_plus_pi_over_6_parallel (x : ℝ) (H : Parallel (Vector.mk (2 * Real.cos x + 2 * Real.sqrt 3 * Real.sin x) 1) (Vector.mk (Real.cos x) (-1))) : 
  Real.tan (x + (π / 6)) = - Real.sqrt 3 / 9 := sorry

-- Problem 2
theorem interval_of_increase_perpendicular (x : ℝ) (y : ℝ) (H : Perpendicular (Vector.mk (2 * Real.cos x + 2 * Real.sqrt 3 * Real.sin x) 1) (Vector.mk (Real.cos x) (-y))) :
  let f : ℝ → ℝ := λ x, 2 * Real.sin (2 * x + π / 6) + 1;
  ∀ k : ℤ, [k * π - π / 3, k * π + π / 6] = {I | I ∈ Interval.Left x, f I < 0} := sorry

end tan_x_plus_pi_over_6_parallel_interval_of_increase_perpendicular_l447_447148


namespace Colin_speed_is_4_l447_447047

-- Define speeds for each individual
def Bruce_speed : ℝ := 1
def Tony_speed : ℝ := 2 * Bruce_speed
def Brandon_speed : ℝ := (1/3) * Tony_speed
def Colin_speed : ℝ := 6 * Brandon_speed

-- Proof statement that Colin's speed is 4 mph
theorem Colin_speed_is_4 : Colin_speed = 4 := 
by sorry

end Colin_speed_is_4_l447_447047


namespace eight_x_plus_y_l447_447160

theorem eight_x_plus_y (x y z : ℝ) (h1 : x + 2 * y - 3 * z = 7) (h2 : 2 * x - y + 2 * z = 6) : 
  8 * x + y = 32 :=
sorry

end eight_x_plus_y_l447_447160


namespace sean_total_apples_l447_447680

-- Define initial apples
def initial_apples : Nat := 9

-- Define the number of apples Susan gives each day
def apples_per_day : Nat := 8

-- Define the number of days Susan gives apples
def number_of_days : Nat := 5

-- Calculate total apples given by Susan
def total_apples_given : Nat := apples_per_day * number_of_days

-- Define the final total apples
def total_apples : Nat := initial_apples + total_apples_given

-- Prove the number of total apples is 49
theorem sean_total_apples : total_apples = 49 := by
  sorry

end sean_total_apples_l447_447680


namespace probability_at_least_one_boy_one_girl_l447_447266

noncomputable def combination (n k : ℕ) := nat.choose n k

def num_boys := 14
def num_girls := 10
def total_members := 24
def committee_size := 4

def total_committees := combination total_members committee_size
def all_boys_committees := combination num_boys committee_size
def all_girls_committees := combination num_girls committee_size
def combined_boys_girls_committees := all_boys_committees + all_girls_committees
def probability_at_least_one_boy_girl :=
  1 - (combined_boys_girls_committees / total_committees : ℚ)

theorem probability_at_least_one_boy_one_girl :
  probability_at_least_one_boy_girl = 11439 / 12650 := sorry

end probability_at_least_one_boy_one_girl_l447_447266


namespace sum_of_fractions_l447_447523

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447523


namespace work_done_in_isothermal_process_l447_447667

variable (n : ℕ) (R : ℝ) (P : ℝ) (ΔV : ℝ) (Q_iso : ℝ)

-- Conditions
def ideal_monoatomic_gas : Prop := (n = 1) ∧ (P * ΔV = 20)

-- Work done in isobaric process
def W_isobaric : ℝ := P * ΔV

-- Heat added in isobaric process
def Q_isobaric : ℝ := (5 / 2) * P * ΔV

-- Heat added in isothermal process (equal to isobaric heat)
def Q_isothermal : ℝ := Q_isobaric

-- Work done in isothermal process is equal to the heat added
def W_isothermal : ℝ := Q_isothermal

-- Proposition for the proof
theorem work_done_in_isothermal_process :
  ideal_monoatomic_gas n R P ΔV →
  Q_isothermal = 50 :=
by
  intro h,
  sorry

end work_done_in_isothermal_process_l447_447667


namespace hyperbola_focal_length_l447_447693

theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 4 = 1)) ∧ (∀ f : ℝ, f = 6) → m = 5 := 
  by 
    -- Using the condition that the focal length is 6
    sorry

end hyperbola_focal_length_l447_447693


namespace math_problem_l447_447533

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447533


namespace final_value_l447_447547

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447547


namespace john_moves_3594_pounds_l447_447984

def bench_press_weight := 15
def bench_press_reps := 10
def bench_press_sets := 3

def bicep_curls_weight := 12
def bicep_curls_reps := 8
def bicep_curls_sets := 4

def squats_weight := 50
def squats_reps := 12
def squats_sets := 3

def deadlift_weight := 80
def deadlift_reps := 6
def deadlift_sets := 2

def total_weight_moved : Nat :=
  (bench_press_weight * bench_press_reps * bench_press_sets) +
  (bicep_curls_weight * bicep_curls_reps * bicep_curls_sets) +
  (squats_weight * squats_reps * squats_sets) +
  (deadlift_weight * deadlift_reps * deadlift_sets)

theorem john_moves_3594_pounds :
  total_weight_moved = 3594 := by {
    sorry
}

end john_moves_3594_pounds_l447_447984


namespace total_weight_of_diamonds_and_jades_l447_447303

def weight_of_diamonds (n : ℕ) (weight_per_diamond : ℕ) : ℕ := n * weight_per_diamond
def weight_of_jades (n : ℕ) (weight_per_jade : ℕ) : ℕ := n * weight_per_jade

theorem total_weight_of_diamonds_and_jades
  (weight_of_5_diamonds : ℕ := 100)
  (jade_heavier_than_diamond : ℕ := 10) :
  (total_weight : ℕ := 140) :=
  let diamond_weight := weight_of_5_diamonds / 5 in
  let jade_weight := diamond_weight + jade_heavier_than_diamond in
  weight_of_diamonds 4 diamond_weight + weight_of_jades 2 jade_weight = total_weight :=
sorry

end total_weight_of_diamonds_and_jades_l447_447303


namespace sum_of_m_and_n_l447_447610

theorem sum_of_m_and_n :
  ∃ m n : ℝ, (∀ x : ℝ, (x = 2 → m = 6 / x) ∧ (x = -2 → n = 6 / x)) ∧ (m + n = 0) :=
by
  let m := 6 / 2
  let n := 6 / (-2)
  use m, n
  simp
  sorry -- Proof omitted

end sum_of_m_and_n_l447_447610


namespace roger_allowance_fraction_l447_447837

noncomputable def allowance_fraction (A m s p : ℝ) : ℝ :=
  m + s + p

theorem roger_allowance_fraction (A : ℝ) (m s p : ℝ) 
  (h_movie : m = 0.25 * (A - s - p))
  (h_soda : s = 0.10 * (A - m - p))
  (h_popcorn : p = 0.05 * (A - m - s)) :
  allowance_fraction A m s p = 0.32 * A :=
by
  sorry

end roger_allowance_fraction_l447_447837


namespace option_C_correct_l447_447333

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end option_C_correct_l447_447333


namespace final_statue_weight_l447_447759

def original_weight : ℝ := 300
def first_week_fraction_cut : ℝ := 0.3
def second_week_fraction_cut : ℝ := 0.3
def third_week_fraction_cut : ℝ := 0.15

def weight_after_first_week : ℝ := original_weight * (1 - first_week_fraction_cut)
def weight_after_second_week : ℝ := weight_after_first_week * (1 - second_week_fraction_cut)
def weight_after_third_week : ℝ := weight_after_second_week * (1 - third_week_fraction_cut)

theorem final_statue_weight : weight_after_third_week = 124.95 :=
by
  skip_proof
  sorry
  -- proof to be provided

end final_statue_weight_l447_447759


namespace robotic_cat_position_at_5_l447_447781

-- Definitions based on the conditions
def P : ℕ → ℤ
| 0     := 0
| (n+1) :=
  if (n + 1) % 5 < 3 then
    P n + 1
  else
    P n - 1

-- Statement for the math proof problem
theorem robotic_cat_position_at_5 : P 5 = 1 :=
sorry

end robotic_cat_position_at_5_l447_447781


namespace math_problem_l447_447534

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447534


namespace arrange_decimals_in_order_l447_447024

theorem arrange_decimals_in_order 
  (a b c d : ℚ) 
  (h₀ : a = 6 / 10) 
  (h₁ : b = 676 / 1000) 
  (h₂ : c = 677 / 1000) 
  (h₃ : d = 67 / 100) : 
  a < d ∧ d < b ∧ b < c := 
by
  sorry

end arrange_decimals_in_order_l447_447024


namespace find_denomination_of_other_notes_l447_447629

theorem find_denomination_of_other_notes :
  ∃ D : ℕ, ∀ (N₅₀ N_D : ℕ),
    (N₅₀ + N_D = 85) →
    (50 * N₅₀ + D * N_D = 5000) →
    (50 * N₅₀ = 3500) →
    D = 100 :=
by
  intros N₅₀ N_D h1 h2 h3
  have h4: N₅₀ = 70 := by sorry
  have h5: N_D = 15 := by sorry
  have h6: D * N_D = 1500 := by sorry
  have h7: D = 100 := by sorry
  exact ⟨100, h7⟩

end find_denomination_of_other_notes_l447_447629


namespace distance_knoxville_to_los_angeles_l447_447011

noncomputable def los_angeles := (0 : ℂ)
noncomputable def knoxville := (780 + 1040 * complex.I : ℂ)

theorem distance_knoxville_to_los_angeles : complex.abs (knoxville - los_angeles) = 1300 := by
  sorry

end distance_knoxville_to_los_angeles_l447_447011


namespace sum_of_g1_l447_447644

noncomputable def g : ℝ → ℝ := sorry

theorem sum_of_g1
  (h : ∀ x y : ℝ, g(g(x - y)) = g(x) * g(y) - g(x) + g(y) - 2 * x * y) :
  g 1 + g 1 = 0 :=
sorry

end sum_of_g1_l447_447644


namespace length_of_chord_EF_l447_447601

-- Defining the lengths of the segments
def length := 24

-- Definitions of the radii of circles
def radiusO := 12
def radiusN := 12
def radiusP := 12

-- Definition of the total length of AD
def AD := length * 3

-- Definition of midpoint positions and related points
def N_midpoint := length * 1.5
def G_tangent_to_N := true
def AG_is_tangent_to_circle_N_at_G : Prop := G_tangent_to_N

theorem length_of_chord_EF (AG_is_tangent_to_circle_N_at_G : Prop) :
    EF = 0 :=
sorry

end length_of_chord_EF_l447_447601


namespace sphere_radius_ratio_l447_447702

theorem sphere_radius_ratio (a : ℝ) (h1 : ∀ r₁, r₁ = a / 2) (h2 : ∀ r₂, r₂ = a * sqrt 3 / 2) :
  ratio (λ r₁ r₂, r₁ / r₂) 1 (sqrt 3) :=
by sorry

end sphere_radius_ratio_l447_447702


namespace minimum_crystals_to_kill_enemy_l447_447764

-- Define the structure of skills and their effects
structure Skill :=
  (crystal_cost : ℕ)
  (health_reduction : ℕ)
  (special_effect : (ℕ → ℕ) := id) -- Default to no special effect

-- Define the four skills
def water : Skill := ⟨4, 4⟩
def fire : Skill := ⟨10, 11⟩
def wind : Skill := ⟨10, 5, λ n, n / 2⟩
def earth : Skill := ⟨18, 0, λ n, if n % 2 = 1 then (n + 1) / 2 else n / 2⟩

-- Main statement to prove
theorem minimum_crystals_to_kill_enemy : ∃ (n : ℕ), n = 68 :=
sorry

end minimum_crystals_to_kill_enemy_l447_447764


namespace range_of_g_l447_447025

noncomputable def g (x : ℝ) : ℝ := (Real.sin x) ^ 6 + (Real.cos x) ^ 4

theorem range_of_g : set.range g = {1} :=
begin
  sorry
end

end range_of_g_l447_447025


namespace triangles_inequality_l447_447214

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangles_inequality
  (a b c a' b' c' S S' : ℝ)
  (h1 : S = area_triangle a b c)
  (h2 : S' = area_triangle a' b' c') :
  a^2 * (- a'^2 + b'^2 + c'^2) + b^2 * (a'^2 - b'^2 + c'^2) + c^2 * (a'^2 + b'^2 - c'^2) ≥ 16 * S * S' ↔ 
  ∃ k : ℝ, k > 0 ∧ a = k * a' ∧ b = k * b' ∧ c = k * c' := sorry

end triangles_inequality_l447_447214


namespace derivative_at_zero_l447_447395

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem derivative_at_zero : deriv f 0 = 2 := by
  sorry

end derivative_at_zero_l447_447395


namespace largest_of_six_consecutive_l447_447291

theorem largest_of_six_consecutive (n : ℕ) (h : n > 0 ∧ n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) = 2013) : n + 5 = 338 :=
by
  have sum_eq : 6 * n + 15 = 2013 := by
    simp [add_assoc, add_comm] at h
    exact add_assoc n n (n + 1 + (n + 2) + (n + 3) + (n + 4) + (n + 5))
  have h_n : 6 * n = 1998 := by
    linarith [sum_eq]
  have n_val : n = 333 := by
    linarith [nat.eq_of_mul_eq_mul_right (by norm_num) h_n]
  rw n_val
  norm_num
  sorry

end largest_of_six_consecutive_l447_447291


namespace expression_value_l447_447537

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447537


namespace solve_equation_l447_447097

noncomputable theory
open Real

-- Definitions for the problem
def problem_eq (n : ℕ) (x : ℝ) : ℝ :=
  ∏ i in range (n + 1), sin (x * i) + ∏ i in range (n + 1), cos (x * i) 

-- Statements of the equivalent Lean proof problem
theorem solve_equation (n : ℕ) (h : n > 0) :
  (∃ m : ℤ, x = 2 * m * π ∨ x = 2 * m * π + π / 2 → problem_eq n x = 1 ∧ (n = 1)) ∧
  (∃ m : ℤ, x = 2 * m * π → problem_eq n x = 1 ∧ ∃ l : ℕ, (n = 4 * l - 2 ∨ n = 4 * l + 1)) ∧
  (∃ m : ℤ, x = m * π → problem_eq n x = 1 ∧ ∃ l : ℕ, (n = 4 * l ∨ n = 4 * l - 1)) := sorry

end solve_equation_l447_447097


namespace new_ratio_after_2_years_l447_447286

-- Definitions based on conditions
variable (A : ℕ) -- Current age of a
variable (B : ℕ) -- Current age of b

-- Conditions
def ratio_a_b := A / B = 5 / 3
def current_age_b := B = 6

-- Theorem: New ratio after 2 years is 3:2
theorem new_ratio_after_2_years (h1 : ratio_a_b A B) (h2 : current_age_b B) : (A + 2) / (B + 2) = 3 / 2 := by
  sorry

end new_ratio_after_2_years_l447_447286


namespace expected_value_shorter_gentlemen_l447_447028

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_gentlemen_l447_447028


namespace inequalities_proof_l447_447664

theorem inequalities_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (a < (c / 2)) ∧ (b < a + c / 2) ∧ ¬(b < c / 2) :=
by
  constructor
  { sorry }
  { constructor
    { sorry }
    { sorry } }

end inequalities_proof_l447_447664


namespace novels_total_l447_447772

theorem novels_total (N : ℕ) (h1: 0.3 * N = 210) : N = 300 :=
by
  -- The proof goes here
  sorry

end novels_total_l447_447772


namespace valid_three_digit_numbers_l447_447154

theorem valid_three_digit_numbers : 
  let total_three_digit_numbers := 9 * 10 * 10,
      excluded_numbers := 9 * 9 in
  total_three_digit_numbers - excluded_numbers = 819 :=
by 
  let total_three_digit_numbers := 9 * 10 * 10
  let excluded_numbers := 9 * 9
  show total_three_digit_numbers - excluded_numbers = 819
  sorry

end valid_three_digit_numbers_l447_447154


namespace annual_population_decrease_due_to_migration_l447_447698

theorem annual_population_decrease_due_to_migration :
  ∃ x : ℝ, (1 + (9 - x) / 100)^3 = 1.259712 ∧ x ≈ 0.9007 :=
by
  -- This is where the proof would go.
  sorry

end annual_population_decrease_due_to_migration_l447_447698


namespace distance_between_hyperbola_vertices_l447_447078

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l447_447078


namespace trisector_length_l447_447465

/-- Given a right triangle DEF with DE = 5 and EF = 12, the length of the shorter angle trisector from F to the hypotenuse DF is given by the expression (1440 * real.sqrt 3 - 600) / 407. -/
theorem trisector_length (DE EF : ℝ) (h1 : DE = 5) (h2 : EF = 12) :
    let DF := real.sqrt (DE^2 + EF^2)
    DF = 13 →
    let x := (60 / (12 * real.sqrt 3 + 5)) * (12 * real.sqrt 3 - 5)
    (2 * x = (1440 * real.sqrt 3 - 600) / 407) :=
by
  intros
  sorry

end trisector_length_l447_447465


namespace tim_books_l447_447734

theorem tim_books (mike_books total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) : ∃ (tim_books : ℕ), tim_books = 22 :=
by
  use total_books - mike_books
  rw [h1, h2]
  norm_num
  sorry

end tim_books_l447_447734


namespace twenty_second_decreasing_number_l447_447769

def is_decreasing_number (n : ℕ) : Prop :=
  let digits := n.digits 10;
  (digits.length = 5 ∧ digits.pairwise (>) ∧ ∀ d ∈ digits, d ≤ 9)

def nth_decreasing_number (n : ℕ) : ℕ :=
  Finset.filter is_decreasing_number (Finset.range (100000:ℕ)).to_list.nth (n - 1)

theorem twenty_second_decreasing_number : nth_decreasing_number 22 = 73210 :=
by
  sorry

end twenty_second_decreasing_number_l447_447769


namespace find_e_l447_447283

def P (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

-- Conditions
variables (d e f : ℝ)
-- Mean of zeros, twice product of zeros, and sum of coefficients are equal
variables (mean_of_zeros equals twice_product_of_zeros equals sum_of_coefficients equals: ℝ)
-- y-intercept is 9
axiom intercept_eq_nine : f = 9

-- Vieta's formulas for cubic polynomial
axiom product_of_zeros : twice_product_of_zeros = 2 * (- (f / 3))
axiom mean_of_zeros_sum : mean_of_zeros = -18/3  -- 3 times the mean of the zeros
axiom sum_of_coef : 3 + d + e + f = sum_of_coefficients

-- All these quantities are equal to the same value
axiom triple_equality : mean_of_zeros = twice_product_of_zeros
axiom triple_equality_coefs : mean_of_zeros = sum_of_coefficients

-- Lean statement we need to prove
theorem find_e : e = -72 :=
by
  sorry

end find_e_l447_447283


namespace negation_even_product_l447_447816

theorem negation_even_product :
  (¬ (∀ (x y : ℤ), (even x → even (x * y)))) ↔
  (∃ (x y : ℤ), even x ∧ ¬ even (x * y)) :=
by
  sorry

end negation_even_product_l447_447816


namespace factor_expression_l447_447062

theorem factor_expression (y : ℝ) : 84 * y ^ 13 + 210 * y ^ 26 = 42 * y ^ 13 * (2 + 5 * y ^ 13) :=
by sorry

end factor_expression_l447_447062


namespace least_value_in_S_l447_447639

def is_valid_set (S : Set ℕ) : Prop :=
  S ⊆ { n | n ≤ 12 } ∧ S.card = 6 ∧
  ∀ a b, a ∈ S → b ∈ S → a < b → ¬(b % a = 0)

theorem least_value_in_S (S : Set ℕ) (h : is_valid_set S) : ∃ x ∈ S, x = 4 :=
begin
  sorry
end

end least_value_in_S_l447_447639


namespace correct_operation_l447_447390

theorem correct_operation (x a b : ℝ) :
  (x^2 * x^3 ≠ x^6) ∧ 
  ((-x^3)^2 = x^6) ∧ 
  (3 * a + 2 * a ≠ 5 * a^2) ∧ 
  ((a + b)^3 ≠ a^3 * b^3) :=
by
  split
  · calc x^2 * x^3 = x^(2+3) : by sorry  -- Uses the property x^m * x^n = x^(m+n)
                  ... ≠ x^6            : by sorry
  · calc (-x^3)^2 = (-1*x^3)^2 : by sorry
                  ... = 1 * x^6 : by sorry
                  ... = x^6          : by sorry
  · calc 3 * a + 2 * a = (3 + 2) * a : by sorry
                      ... = 5 * a     : by sorry
                      ... ≠ 5 * a^2   : by sorry
  · calc (a + b)^3 = (a + b) * (a + b) * (a + b) : by sorry
                   ... ≠ a^3 * b^3                    : by sorry

end correct_operation_l447_447390


namespace probability_B_given_A_l447_447706

variables {Ω : Type*} {P : MeasureTheory.Measure Ω} [ProbabilityMeasure P]
variables (A B : Set Ω)

theorem probability_B_given_A (hA : P(A) = 0.80) (hB : P(B) = 0.60) :
  (Probability.cond A B) = 0.75 :=
by
  sorry

end probability_B_given_A_l447_447706


namespace domain_of_function_monotonicity_of_function_l447_447494

theorem domain_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (if 1 < a then (0, ∞) else (−∞, 0)) = 
    {x : ℝ | (a > 1 → 0 < x) ∧ (a < 1 → x < 0)} := 
by sorry

theorem monotonicity_of_function (a : ℝ) (h : a = 2) :
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → 
  (f(x) = log 3 (2^x - 1)) →
  f(x2) > f(x1) := 
by sorry

end domain_of_function_monotonicity_of_function_l447_447494


namespace sphere_cross_section_sum_area_l447_447476

theorem sphere_cross_section_sum_area
  (A B : Point) 
  (r : ℝ)
  (radius_of_sphere : r = 5) 
  (distance_AB : dist A B = 8) 
  (O1 O2 : Plane) 
  (O1AB : Plane_passes_through O1 A B) 
  (O2AB : Plane_passes_through O2 A B) 
  (O1O2_perpendicular : planes_perpendicular O1 O2) 
  (S1 S2 : ℝ) 
  (area_of_section_O1 : area_of_cross_section O1 = S1) 
  (area_of_section_O2 : area_of_cross_section O2 = S2) :
  S1 + S2 = 41 * Real.pi := 
sorry

end sphere_cross_section_sum_area_l447_447476


namespace helium_has_lowest_temperature_l447_447697

def liquefaction_temp : Type :=
  ℝ

structure Gas :=
  (name : String)
  (temperature : liquefaction_temp)

def Oxygen : Gas :=
  ⟨"Oxygen", -183⟩

def Hydrogen : Gas :=
  ⟨"Hydrogen", -253⟩

def Nitrogen : Gas :=
  ⟨"Nitrogen", -195.8⟩

def Helium : Gas :=
  ⟨"Helium", -268⟩

theorem helium_has_lowest_temperature :
  ∀ (gases : List Gas), (Helium ∈ gases) →
  ∀ gas ∈ gases, Helium.temperature ≤ gas.temperature :=
by
  sorry

end helium_has_lowest_temperature_l447_447697


namespace range_of_a_l447_447121

variable (f : ℝ → ℝ)

-- f is an odd function
def odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 1: f is an odd function
axiom h_odd : odd_function f

-- Condition 2: f(x) + f(x + 3 / 2) = 0 for any real number x
axiom h_periodicity : ∀ x : ℝ, f x + f (x + 3 / 2) = 0

-- Condition 3: f(1) > 1
axiom h_f1 : f 1 > 1

-- Condition 4: f(2) = a for some real number a
variable (a : ℝ)
axiom h_f2 : f 2 = a

-- Goal: Prove that a < -1
theorem range_of_a : a < -1 :=
  sorry

end range_of_a_l447_447121


namespace arithmetic_sequence_problem_l447_447808

theorem arithmetic_sequence_problem :
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sum_first_sequence - sum_second_sequence - sum_third_sequence = 188725 :=
by
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sorry

end arithmetic_sequence_problem_l447_447808


namespace number_of_leading_zeros_l447_447829

theorem number_of_leading_zeros
  (a b c d : ℕ)
  (h : a = 2 ∧ b = 7 ∧ c = 5 ∧ d = 8)
  : ∀ (x : ℝ), x = (1 / (2^a * 5^c) * 3 / 5^2) → (to_decimal x).leading_zeros = 7 :=
by sorry

end number_of_leading_zeros_l447_447829


namespace triangle_proof_l447_447458

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions
axiom cos_rule_1 : a / cos A = c / (2 - cos C)
axiom b_value : b = 4
axiom c_value : c = 3
axiom area_equation : (1 / 2) * a * b * sin C = 3

-- The theorem statement
theorem triangle_proof : 3 * sin C + 4 * cos C = 5 := sorry

end triangle_proof_l447_447458


namespace sum_of_coefficients_eq_neg_two_l447_447112

-- Define the polynomial and its expansion
def polynomial1 (x : ℝ) : ℝ := (x^2 + 1)^2 * (2 * x + 1)^9

-- Define the expansion polynomial
noncomputable def polynomial2 (x : ℝ) : ℝ := 
  a_0 + a_1 * (x + 2) + a_2 * (x + 2)^2 + a_3 * (x + 2)^3 + 
  a_4 * (x + 2)^4 + a_5 * (x + 2)^5 + a_6 * (x + 2)^6 + 
  a_7 * (x + 2)^7 + a_8 * (x + 2)^8 + a_9 * (x + 2)^9 + 
  a_{10} * (x + 2)^{10} + a_{11} * (x + 2)^{11]

-- Statement to be proven
theorem sum_of_coefficients_eq_neg_two :
  (polynomial1 0) = polynomial2 0 → (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} = -2) :=
sorry

end sum_of_coefficients_eq_neg_two_l447_447112


namespace total_amount_paid_l447_447298

noncomputable def cost_per_night_per_person : ℕ := 40
noncomputable def number_of_people : ℕ := 3
noncomputable def number_of_nights : ℕ := 3

theorem total_amount_paid (cost_per_night_per_person number_of_people number_of_nights : ℕ) :
  (cost_per_night_per_person * number_of_people * number_of_nights = 360) :=
by
  have h : cost_per_night_per_person * number_of_people * number_of_nights = 40 * 3 * 3 := by rfl
  rw h
  exact rfl

end total_amount_paid_l447_447298


namespace min_rectangle_area_correct_l447_447815

noncomputable def min_rectangle_area : ℝ :=
  let distance_ab : ℝ := 1
  let distance_bc : ℝ := 3
  let distance_cd : ℝ := 1
  8

theorem min_rectangle_area_correct :
  let distance_ab : ℝ := 1
  let distance_bc : ℝ := 3
  let distance_cd : ℝ := 1 in
  (min_rectangle_area distance_ab distance_bc distance_cd) = 8 := 
by
  sorry

end min_rectangle_area_correct_l447_447815


namespace number_of_lines_through_focus_intersecting_hyperbola_l447_447496

open Set

noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 2) - y^2 = 1

-- The coordinates of the focuses of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Definition to express that a line passes through the right focus
def line_through_focus (l : ℝ → ℝ) : Prop := l 2 = 0

-- Definition for the length of segment AB being 4
def length_AB_is_4 (A B : ℝ × ℝ) : Prop := dist A B = 4

-- The statement asserting the number of lines satisfying the given condition
theorem number_of_lines_through_focus_intersecting_hyperbola:
  ∃ (n : ℕ), n = 3 ∧ ∀ (l : ℝ → ℝ),
  line_through_focus l →
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ length_AB_is_4 A B :=
sorry

end number_of_lines_through_focus_intersecting_hyperbola_l447_447496


namespace proof_problem_l447_447167

namespace ProofProblem

-- Definitions as required by the conditions
noncomputable def quadratic_radical_1 (a : ℝ) : ℝ := Real.sqrt (2 * a - 2)
noncomputable def quadratic_radical_2 (a : ℝ) : ℝ := Real.sqrt (-a + 16)
def new_operation (x y : ℝ) : ℝ := (Real.sqrt (x + y)) / (x - y)

-- Statement of the proof problem
theorem proof_problem (a : ℝ) (h : quadratic_radical_1 a = quadratic_radical_2 a) :
  (a = 6) ∧ 
  (Real.sqrt a = Real.sqrt 6 ∨ Real.sqrt a = -Real.sqrt 6) ∧
  (new_operation a (new_operation a (-2)) = 10 / 23) :=
  by sorry

end ProofProblem

end proof_problem_l447_447167


namespace work_done_in_isothermal_process_l447_447666

variable (n : ℕ) (R : ℝ) (P : ℝ) (ΔV : ℝ) (Q_iso : ℝ)

-- Conditions
def ideal_monoatomic_gas : Prop := (n = 1) ∧ (P * ΔV = 20)

-- Work done in isobaric process
def W_isobaric : ℝ := P * ΔV

-- Heat added in isobaric process
def Q_isobaric : ℝ := (5 / 2) * P * ΔV

-- Heat added in isothermal process (equal to isobaric heat)
def Q_isothermal : ℝ := Q_isobaric

-- Work done in isothermal process is equal to the heat added
def W_isothermal : ℝ := Q_isothermal

-- Proposition for the proof
theorem work_done_in_isothermal_process :
  ideal_monoatomic_gas n R P ΔV →
  Q_isothermal = 50 :=
by
  intro h,
  sorry

end work_done_in_isothermal_process_l447_447666


namespace vann_teeth_cleaning_l447_447316

def numDogsCleaned (D : Nat) : Prop :=
  let dogTeethCount := 42
  let catTeethCount := 30
  let pigTeethCount := 28
  let numCats := 10
  let numPigs := 7
  let totalTeeth := 706
  dogTeethCount * D + catTeethCount * numCats + pigTeethCount * numPigs = totalTeeth

theorem vann_teeth_cleaning : numDogsCleaned 5 :=
by
  sorry

end vann_teeth_cleaning_l447_447316


namespace total_ducks_l447_447717

-- Definitions based on the given conditions
def Muscovy : ℕ := 39
def Cayuga : ℕ := Muscovy - 4
def KhakiCampbell : ℕ := (Cayuga - 3) / 2

-- Proof statement
theorem total_ducks : Muscovy + Cayuga + KhakiCampbell = 90 := by
  sorry

end total_ducks_l447_447717


namespace math_problem_l447_447504

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447504


namespace sequence_sum_l447_447000

-- Definition of the sequence b_n
def b : ℕ → ℝ
| 0     := 0 -- We define b_0 to be 0 to make b_1 be the first element at index 1
| 1     := 2
| 2     := 3
| (n+3) := (1 / 2) * b (n + 2) + (1 / 3) * b (n + 1)

-- The statement to prove that the infinite sum of the sequence is 4.2
theorem sequence_sum : (∑' n, b n) = 4.2 := by
  sorry

end sequence_sum_l447_447000


namespace one_neither_prime_nor_composite_l447_447968

/-- Definition of a prime number in the natural numbers -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of a composite number in the natural numbers -/
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n 

/-- Theorem stating that the number 1 is neither prime nor composite -/
theorem one_neither_prime_nor_composite : ¬is_prime 1 ∧ ¬is_composite 1 :=
sorry

end one_neither_prime_nor_composite_l447_447968


namespace potions_needed_l447_447932

-- Definitions
def galleons_to_knuts (galleons : Int) : Int := galleons * 17 * 23
def sickles_to_knuts (sickles : Int) : Int := sickles * 23

-- Conditions from the problem
def cost_of_owl_in_knuts : Int := galleons_to_knuts 2 + sickles_to_knuts 1 + 5
def knuts_per_potion : Int := 9

-- Prove the number of potions needed is 90
theorem potions_needed : cost_of_owl_in_knuts / knuts_per_potion = 90 := by
  sorry

end potions_needed_l447_447932


namespace expression_value_l447_447553

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447553


namespace math_problem_l447_447508

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447508


namespace no_quadruples_sum_2013_l447_447064

theorem no_quadruples_sum_2013 :
  ¬ ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c + d = 2013 ∧
  2013 % a = 0 ∧ 2013 % b = 0 ∧ 2013 % c = 0 ∧ 2013 % d = 0 :=
by
  sorry

end no_quadruples_sum_2013_l447_447064


namespace concurrency_of_lines_l447_447648

open PlaneEuclideanGeometry

noncomputable def points_concurrent (A B C D X Y O M N : Point) (L : Line) (Γ1 Γ2 : Circle) : Prop :=
  are_concurrent (lineThrough A M) (lineThrough D N) (lineThrough X Y)

theorem concurrency_of_lines (A B C D X Y O M N : Point) 
  (Γ1 : Circle)
  (Γ2 : Circle)
  (h1 : collinear [A, B, C, D])
  (h2 : diameter Γ1 A C)
  (h3 : diameter Γ2 B D)
  (h4 : intersects Γ1 Γ2 X Y)
  (h5 : ¬ collinear [A, B, C, O])
  (h6 : onLine O (lineThrough X Y))
  (h7 : intersection_point (lineThrough C O) Γ1 M)
  (h8 : intersection_point (lineThrough B O) Γ2 N) : 
  points_concurrent A B C D X Y O M N := 
begin
  sorry
end

end concurrency_of_lines_l447_447648


namespace sphere_radius_l447_447003

theorem sphere_radius {r : ℝ} (h_cone_height : ℝ) (h_cone_shadow : ℝ) (h_sphere_shadow : ℝ)
  (H_cone : h_cone_height = 3) (H_cone_shadow : h_cone_shadow = 4) (H_sphere_shadow : h_sphere_shadow = 15) :
  r = 11.25 :=
by
  let theta := real.arctan (h_cone_height / h_cone_shadow)
  have H_tan_cone : real.tan theta = h_cone_height / h_cone_shadow,
  sorry  -- Proof step not needed as per instructions
  have H_tan_sphere : real.tan theta = r / h_sphere_shadow,
  sorry  -- Proof step not needed as per instructions
  have H_equal_tan : r / 15 = 3 / 4,
  sorry  -- Proof step not needed as per instructions
  calc
    r = 3 / 4 * 15 : by simp only [H_equal_tan, mul_div_cancel' _ (by norm_num)]
   ... = 11.25    : by norm_num

end sphere_radius_l447_447003


namespace limit_ratio_sectors_to_circle_l447_447049

open Classical
open Real

def equilateral_triangle_area (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

def circle_area (r : ℝ) : ℝ :=
  π * r^2

noncomputable def sectors_area (s r : ℝ) : ℝ :=
  equilateral_triangle_area s - circle_area r

noncomputable def sectors_to_circle_ratio (s r : ℝ) : ℝ :=
  sectors_area s r / circle_area r

theorem limit_ratio_sectors_to_circle (r : ℝ) (h : r > 0) :
  tendsto (λ s, sectors_to_circle_ratio s r) at_top (𝓝 ℝ ∞) :=
by
  sorry

end limit_ratio_sectors_to_circle_l447_447049


namespace exists_transform_to_square_exists_transform_to_rectangle_not_square_l447_447861

-- Define the point and transformations
structure Point :=
  (x : Int)
  (y : Int)

def alpha_transform (α : Int) (p : Point) : Point :=
  Point.mk p.x (α * p.x + p.y)

def beta_transform (α : Int) (p : Point) : Point :=
  Point.mk (p.x + α * p.y) p.y


-- Define quadrilateral type
inductive Quadrilateral
| rhombus (a b c d : Point) : Quadrilateral
| square (a b c d : Point) : Quadrilateral
| rectangle (a b c d : Point) : Quadrilateral

-- The main statements
theorem exists_transform_to_square (α : Int) :
  ¬ ∃ (rh : Quadrilateral), (is_rhombus rh) ∧ (∃ seq, transforms_to_square_seq seq α rh) :=
sorry

theorem exists_transform_to_rectangle_not_square (α : Int) :
  ∃ (rh : Quadrilateral), (is_rhombus rh) ∧ (∃ seq, transforms_to_rectangle_not_square_seq seq α rh) :=
sorry

-- necessary definitions (not needing proof)
def is_rhombus (q : Quadrilateral) : Prop := 
  -- check if all sides are equal
  sorry

def is_square (q : Quadrilateral) : Prop := 
  -- check if all sides and angles are equal
  sorry

def is_rectangle_not_square (q : Quadrilateral) : Prop :=
  -- check if both pairs of opposite sides are equal and one pair is not equal to the other
  sorry

def transforms_to_square_seq (seq : List (Point → Point)) (α : Int) (q : Quadrilateral)  : Prop := 
  -- A sequence of transformations that convert a rhombus into a square
  sorry

def transforms_to_rectangle_not_square_seq (seq : List (Point → Point)) (α : Int) (q : Quadrilateral) : Prop := 
  -- A sequence of transformations that convert a rhombus into a rectangle that is not a square
  sorry


end exists_transform_to_square_exists_transform_to_rectangle_not_square_l447_447861


namespace fuel_oil_used_l447_447239

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end fuel_oil_used_l447_447239


namespace arrangement_proof_l447_447059

def arrangement_problem : Prop :=
  ∃ (countries : Set ℕ) (hotels : Set ℕ) (f : ℕ → ℕ),
    countries = {(1, 2, 3, 4, 5)} ∧
    hotels = {(1, 2, 3)} ∧
    (∀ country ∈ countries, f country ∈ hotels) ∧
    (∀ hotel ∈ hotels, ∃ country ∈ countries, f country = hotel) ∧
    fintype.card {g : ℕ → ℕ // g = f} = 150

theorem arrangement_proof : arrangement_problem :=
begin
  -- proof would go here
  sorry,
end

end arrangement_proof_l447_447059


namespace tangent_line_at_1_monotonicity_F_l447_447871

noncomputable def f (a : ℝ) (x : ℝ) := a * x - 1 / x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := f a x - g x

theorem tangent_line_at_1 (a : ℝ) : g 1 = 0 ∧ (∃ m b, ∀ x > 0, g x - g 1 = m * (x - 1) + b) :=
  by
    have g_1 : g 1 = 0 := by simp [g]
    have g'_x : ∀ x > 0, deriv g x = 1 / x := by sorry
    have g'_1 : deriv g 1 = 1 := by sorry
    use [1, -1]
    intro x hx
    -- The proof continues
    sorry

theorem monotonicity_F (a : ℝ) : 
  (a ≥ 1 / 4 → ∀ x > 0, deriv (F a) x ≥ 0) ∧
  (a = 0 → (∀ x ∈ Ioo 0 1, deriv (F a) x > 0) ∧ (∀ x ∈ Ioo 1 (1/0), deriv (F a) x < 0)) ∧
  (0 < a ∧ a < 1 / 4 → ∃ x1 x2 > 0, x1 < x2 ∧
    (∀ x ∈ Ioo 0 x1, deriv (F a) x > 0) ∧
    (∀ x ∈ Ioo x1 x2, deriv (F a) x < 0) ∧
    (∀ x ∈ Ioi x2, deriv (F a) x > 0)) ∧
  (a < 0 → ∃ x1 > 0, ∀ x ∈ Ioo 0 x1, deriv (F a) x > 0 ∧ ∀ x ∈ Ioi x1, deriv (F a) x < 0) :=
  by
    -- The proof continues
    sorry

end tangent_line_at_1_monotonicity_F_l447_447871


namespace conjugate_coordinates_l447_447187

-- Define the given complex number
def complex_number := (2 - Complex.i) / (1 + Complex.i)

-- Define the simplified form of the given complex number
def simplified_complex_number := (1 / 2) - (3 / 2) * Complex.i

-- Define the conjugate of the simplified complex number
def conjugate_of_simplified := Complex.conj simplified_complex_number

-- Define the expected conjugate coordinates
def expected_conjugate := (1 / 2) + (3 / 2) * Complex.i

-- State that the conjugate of the given complex number equals the expected coordinates
theorem conjugate_coordinates :
  Complex.conj ((2 - Complex.i) / (1 + Complex.i)) = (1 / 2) + (3 / 2) * Complex.i :=
  by sorry

end conjugate_coordinates_l447_447187


namespace problem_statement_l447_447220

variables (a b c : ℝ)

def my_max (x y : ℝ) := max x y
def my_min (x y : ℝ) := min x y

theorem problem_statement :
  (my_max a b = my_max b a) ∧
  (my_max a (my_max b c) = my_max (my_max a b) c) ∧
  (my_min a (my_max b c) = my_max (my_min a b) (my_min a c)) :=
  by
  -- Proofs will follow here
  sorry

end problem_statement_l447_447220


namespace distance_from_start_l447_447036

-- Definitions based on conditions
def eastward_distance : ℝ := 5
def angle : ℝ := 45
def northward_distance : ℝ := 7

-- Hypotenuse calculation for the given conditions
def leg_length : ℝ := northward_distance / Real.sqrt 2

def total_eastward_distance : ℝ := eastward_distance + leg_length
def total_northward_distance : ℝ := leg_length

def final_distance : ℝ := Real.sqrt ((total_eastward_distance)^2 + (total_northward_distance)^2)

-- The proof problem
theorem distance_from_start : final_distance = Real.sqrt (74 + 35 * Real.sqrt 2) := sorry

end distance_from_start_l447_447036


namespace find_lambda_l447_447147

variables (a b : ℝ^3) (λ : ℝ)

-- Conditions
def mag_a : real := 2
def mag_b : real := 1
def angle_a_b : real := real.pi / 3 -- 60 degrees in radians
def perpendicular : Prop := a ∙ (a + λ • b) = 0

-- The proof problem statement
theorem find_lambda
  (h1: ∥a∥ = mag_a)
  (h2: ∥b∥ = mag_b)
  (h3: real.angle a b = angle_a_b)
  (h4: perpendicular a b λ) : λ = -4 :=
sorry

end find_lambda_l447_447147


namespace emancipation_proclamation_day_l447_447265
-- Import necessary Lean libraries

-- Define required components of the problem in Lean
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ ¬ (year % 100 = 0)) ∨ (year % 400 = 0)

def num_days_in_years (years : ℕ) : ℕ :=
  let leap_years := (List.range years).count (λ y, is_leap_year (1863 + y))
  let regular_years := years - leap_years
  regular_years * 365 + leap_years * 366

-- Define the initial condition of the problem
def initial_day_of_week_2013 : ℕ := 2  -- Tuesday, where 0=Sunday, 1=Monday, ..., 6=Saturday

def day_of_week_after_days (initial_day : ℕ) (days : ℕ) : ℕ :=
  (initial_day + days) % 7

-- The theorem to be proved
theorem emancipation_proclamation_day : 
  day_of_week_after_days initial_day_of_week_2013 (num_days_in_years 150) = 4 :=  -- 4 corresponds to Thursday
sorry

end emancipation_proclamation_day_l447_447265


namespace misread_weight_l447_447268

theorem misread_weight (avg_initial : ℝ) (avg_correct : ℝ) (n : ℕ) (actual_weight : ℝ) (x : ℝ) : 
  avg_initial = 58.4 → avg_correct = 58.7 → n = 20 → actual_weight = 62 → 
  (n * avg_correct - n * avg_initial = actual_weight - x) → x = 56 :=
by
  intros
  sorry

end misread_weight_l447_447268


namespace min_lines_to_separate_points_l447_447722

theorem min_lines_to_separate_points (m n : ℕ) (h_m : m = 8) (h_n : n = 8) : 
  (m - 1) + (n - 1) = 14 := by
  sorry

end min_lines_to_separate_points_l447_447722


namespace find_number_l447_447711

-- Given conditions
variables (x y : ℕ)

-- The conditions from the problem statement
def digit_sum : Prop := x + y = 12
def reverse_condition : Prop := (10 * x + y) + 36 = 10 * y + x

-- The final statement
theorem find_number (h1 : digit_sum x y) (h2 : reverse_condition x y) : 10 * x + y = 48 :=
sorry

end find_number_l447_447711


namespace mode_eighth_grade_is_81_median_seventh_grade_is_76_5_l447_447362

def seventh_grade_scores : List ℕ := [70, 85, 73, 80, 75, 76, 87, 74, 75, 94, 75, 79, 81, 71, 75, 81, 88, 59, 85, 77]
def eighth_grade_scores : List ℕ := [92, 74, 87, 82, 72, 81, 94, 83, 77, 83, 80, 81, 71, 81, 72, 77, 82, 80, 70, 41]

theorem mode_eighth_grade_is_81 : List.mode eighth_grade_scores = 81 := by
  sorry

theorem median_seventh_grade_is_76_5 : List.median seventh_grade_scores = 76.5 := by
  sorry

end mode_eighth_grade_is_81_median_seventh_grade_is_76_5_l447_447362


namespace rita_drive_distance_l447_447058

theorem rita_drive_distance :
  ∃ x : ℝ, 200 < x ∧ x ≠ 0 ∧
  let speed := x / 200,
      dist_before_rain := x / 4,
      dist_after_rain := 3 * x / 4,
      reduced_speed := speed - 15 / 60,
      time_before_rain := dist_before_rain / speed,
      time_after_rain := dist_after_rain / reduced_speed in
  time_before_rain + time_after_rain = 300 ∧
  x = 125 :=
begin
  use 125,
  split,
  { linarith },
  split,
  { norm_num },
  simp [200, 300, 15, 60],
  let speed := 125 / 200,
  let dist_before_rain := 125 / 4,
  let dist_after_rain := 3 * 125 / 4,
  let reduced_speed := speed - 15 / 60,
  let time_before_rain := dist_before_rain / speed,
  let time_after_rain := dist_after_rain / reduced_speed,
  have : speed = 0.625 := by norm_num [speed],
  have : dist_before_rain = 31.25 := by norm_num [dist_before_rain],
  have : dist_after_rain = 93.75 := by norm_num [dist_after_rain],
  have : reduced_speed = 0.375 := by norm_num [reduced_speed],
  have : time_before_rain = 50 := by norm_num [time_before_rain, this.1],
  have : time_after_rain = 250 := by norm_num [time_after_rain, this.2, this.3],
  norm_num [this.4, this.5],
end

end rita_drive_distance_l447_447058


namespace conditional_probability_l447_447709

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end conditional_probability_l447_447709


namespace S_40_eq_150_l447_447123

variable {R : Type*} [Field R]

-- Define the sum function for geometric sequences.
noncomputable def geom_sum (a q : R) (n : ℕ) : R :=
  a * (1 - q^n) / (1 - q)

-- Given conditions from the problem.
axiom S_10_eq : ∀ {a q : R}, geom_sum a q 10 = 10
axiom S_30_eq : ∀ {a q : R}, geom_sum a q 30 = 70

-- The main theorem stating S40 = 150 under the given conditions.
theorem S_40_eq_150 {a q : R} (h10 : geom_sum a q 10 = 10) (h30 : geom_sum a q 30 = 70) :
  geom_sum a q 40 = 150 :=
sorry

end S_40_eq_150_l447_447123


namespace largest_integer_n_exists_l447_447083

theorem largest_integer_n_exists :
  ∃ (x y z n : ℤ), (x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 10 = n^2) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_exists_l447_447083


namespace final_value_l447_447546

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447546


namespace triangle_ABC_perimeter_l447_447951

noncomputable def perimeter_triangle_ABC : Real :=
  let AB := 15
  let AX := 2 * AB
  let BY := AX
  let BC := AB / Real.sqrt 2
  AB + BC + BC

theorem triangle_ABC_perimeter :
  ∀ (A B C X Y Z W : Point) 
  (hC : ∠ C = 90)
  (hAB_len : length AB = 15)
  (h_rect : is_rectangle A B X Y)
  (h_square : is_square B C W Z)
  (h_AX : AX = 2 * AB)
  (h_points : all_points_on_circle [X, Y, Z, W]) : 
  perimeter_triangle_ABC = 15 + 15 * Real.sqrt 2 :=
sorry

end triangle_ABC_perimeter_l447_447951


namespace probability_all_zero_probability_product_is_4_l447_447728

-- Definitions for bag A and bag B
def bag_A : list ℕ := [0, 1, 1, 2, 2, 2]
def bag_B : list ℕ := [0, 0, 0, 0, 1, 2, 2]

-- Define the drawing process
def draw_card_A (idx : fin 6) : ℕ := bag_A.nth_le idx.1 idx.2
def draw_cards_B (idx1 : fin 7) (idx2 : fin 7) (h : idx1 ≠ idx2) : list ℕ :=
  [bag_B.nth_le idx1.1 idx1.2, bag_B.nth_le idx2.1 idx2.2]

-- Define probabilities as goals to prove
theorem probability_all_zero :
  (∑ (i : fin 6) (j k : fin 7) (h : j ≠ k),
    if (draw_card_A i = 0 ∧ draw_cards_B j k h = [0, 0])
    then (1 : ℚ)
    else 0) / (6 * ∑ (x y : fin 7), if x ≠ y then 1 else 0) = 1 / 21 :=
sorry

theorem probability_product_is_4 :
  (∑ (i : fin 6) (j k : fin 7) (h : j ≠ k),
    if ((draw_card_A i) * list.prod (draw_cards_B j k h) = 4)
    then (1 : ℚ)
    else 0) / (6 * ∑ (x y : fin 7), if x ≠ y then 1 else 0) = 4 / 63 :=
sorry

end probability_all_zero_probability_product_is_4_l447_447728


namespace largest_power_of_two_divides_17_to_4_minus_15_to_4_l447_447085

theorem largest_power_of_two_divides_17_to_4_minus_15_to_4 :
  ∃ k : ℕ, 17^4 - 15^4 = 2^k ∧ k = 7 :=
begin
  use 7,
  sorry
end

end largest_power_of_two_divides_17_to_4_minus_15_to_4_l447_447085


namespace students_in_band_l447_447800

theorem students_in_band (total_students : ℕ) (band_percentage : ℚ) (h_total_students : total_students = 840) (h_band_percentage : band_percentage = 0.2) : ∃ band_students : ℕ, band_students = 168 ∧ band_students = band_percentage * total_students := 
sorry

end students_in_band_l447_447800


namespace find_5_minus_a_l447_447581

-- Define the problem conditions as assumptions
variable (a b : ℤ)
variable (h1 : 5 + a = 6 - b)
variable (h2 : 3 + b = 8 + a)

-- State the theorem we want to prove
theorem find_5_minus_a : 5 - a = 7 :=
by
  sorry

end find_5_minus_a_l447_447581


namespace find_f8_f9_l447_447694

variable {ℝ : Type} [LinearOrderedField ℝ]

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

variable (f : ℝ → ℝ)
variable (h1 : odd_function f)
variable (h2 : periodic_function f 9)
variable (h3 : f 1 = 7)

theorem find_f8_f9 :
  f 8 + f 9 = -7 := 
sorry

end find_f8_f9_l447_447694


namespace limit_sequence_l447_447763

theorem limit_sequence :
  (Real.log (λ n : ℕ, let num := (n + 2)^4 - (n - 2)^4;
                    let denom := (n + 5)^2 + (n - 5)^2;
                    (num / denom))
  ) = +⟹
  sorry

end limit_sequence_l447_447763


namespace sphere_surface_area_l447_447900

theorem sphere_surface_area (R r : ℝ) (h1 : 2 * OM = R) (h2 : ∀ r, π * r^2 = 3 * π) : 4 * π * R^2 = 16 * π :=
by
  sorry

end sphere_surface_area_l447_447900


namespace find_PQ_l447_447190

open EuclideanGeometry

-- Definitions for the problem conditions.
variables {P Q R S Q' S' : Point}
variable {d : distance}

-- Definitions ensured to appear in conditions:
-- Triangle PQR is reflected over line PR to produce triangle PQ'S'
def is_reflection (P Q R S Q' S' : Point) : Prop :=
  reflection (line P R) P Q' Q ∧ reflection (line P R) P S' S

-- Given distances
axiom PQ_len : d P Q = 9
axiom QR_len : d Q R = 15
axiom PS_len : d P S = 20

-- The theorem to prove 
theorem find_PQ'_length (h : is_reflection P Q R S Q' S') : d P Q' = 9 := 
  sorry -- Proof to be filled in.

end find_PQ_l447_447190


namespace problem_statement_l447_447514

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447514


namespace proof_exists_a_b_A_l447_447347

open Set Classical BigOperators

noncomputable theory

variable (X : Set ℝ)
variable [Fintype X]

theorem proof_exists_a_b_A (hX : Fintype.card X = 2020) :
  ∃ (a b: ℝ) (A : Set ℝ), A ⊆ X ∧ 
    ∑ x in A, (x - a)^2 + ∑ x in (X \ A), (x - b)^2 ≤ (1009 / 1010) * ∑ x in X, x^2 := 
by
  sorry

end proof_exists_a_b_A_l447_447347


namespace station_master_is_correct_l447_447282

theorem station_master_is_correct : 
  ¬(∃ (A B C: Finset ℕ), 
      {1, 2, 3, 4, 5, 6, 7, 8, 9} = A ∪ B ∪ C ∧
      A.card = 3 ∧ B.card = 3 ∧ C.card = 3 ∧
      (A.max' (by dec_trivial) = (A.erase (A.max' (by dec_trivial))).sum) ∧
      (B.max' (by dec_trivial) = (B.erase (B.max' (by dec_trivial))).sum) ∧
      (C.max' (by dec_trivial) = (C.erase (C.max' (by dec_trivial))).sum)) :=
by sorry

end station_master_is_correct_l447_447282


namespace find_tim_books_l447_447735

theorem find_tim_books (Mike_books Total_books Tim_books : ℕ) :
  Mike_books = 20 → Total_books = 42 → Tim_books = Total_books - Mike_books → Tim_books = 22 :=
by
  intros hMike hTotal hTim
  rw [hMike, hTotal] at hTim
  calc
    Tim_books = 42 - 20 : by rw [hTim]
           ... = 22 : by norm_num

end find_tim_books_l447_447735


namespace plane_not_containing_any_l447_447344

variable (n : ℕ)
variable (M : Set (Fin n))

-- Define that points are in general position
def in_general_position (M : Set (Fin n)) : Prop :=
  ∀ (a b c d : Fin n), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  ¬ collinear {a, b, c} ∧ ¬ coplanar {a, b, c, d}

-- Define the planes formed by sets of three points
def planes (M : Set (Fin n)) : Set (Set (Fin n)) :=
  { t | t ⊆ M ∧ t.card = 3 }

-- Main theorem statement
theorem plane_not_containing_any (M : Set (Fin n)) [finite M] (h : in_general_position M) (A : Set (Fin n)) (hA : A.card = n - 3) :
  ∃ p ∈ planes M, disjoint p A := 
sorry

end plane_not_containing_any_l447_447344


namespace problem_statement_l447_447990

theorem problem_statement (A B C D E : ℤ) (hB : B ≠ 0) 
  (F : ℤ := A * D^2 - B * C * D + B^2 * E) (hF : F ≠ 0) :
  ∃ N : ℕ, (∀ x y : ℤ, A * x^2 + B * x * y + C * x + D * y + E = 0 → N ≤ 2 * (divisors (|F|)).card) :=
sorry

end problem_statement_l447_447990


namespace exists_integer_n_l447_447848

theorem exists_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1453 [MOD 10] :=
by
  use 7
  split
  · exact sorry -- Proof that 0 ≤ 7
  split
  · exact sorry -- Proof that 7 ≤ 9
  · exact sorry -- Proof that 7 ≡ -1453 [MOD 10]

end exists_integer_n_l447_447848


namespace find_value_of_2_times_x_minus_y_squared_minus_3_l447_447571

-- Define the conditions as noncomputable variables
variables (x y : ℝ)

-- State the main theorem
theorem find_value_of_2_times_x_minus_y_squared_minus_3 :
  (x^2 - x*y = 12) →
  (y^2 - y*x = 15) →
  2 * (x - y)^2 - 3 = 51 :=
by
  intros h1 h2
  sorry

end find_value_of_2_times_x_minus_y_squared_minus_3_l447_447571


namespace small_barrel_5_tons_l447_447789

def total_oil : ℕ := 95
def large_barrel_capacity : ℕ := 6
def small_barrel_capacity : ℕ := 5

theorem small_barrel_5_tons :
  ∃ (num_large_barrels num_small_barrels : ℕ),
  num_small_barrels = 1 ∧
  total_oil = (num_large_barrels * large_barrel_capacity) + (num_small_barrels * small_barrel_capacity) :=
by
  sorry

end small_barrel_5_tons_l447_447789


namespace math_problem_l447_447505

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447505


namespace domain_sqrt_tan_x_sub_sqrt3_l447_447082

open Real

noncomputable def domain := {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2}

theorem domain_sqrt_tan_x_sub_sqrt3 :
  {x | ∃ y : ℝ, y = sqrt (tan x - sqrt 3)} = domain :=
by
  sorry

end domain_sqrt_tan_x_sub_sqrt3_l447_447082


namespace find_quotient_l447_447954

theorem find_quotient : ∃ q : ℕ, 1375 = (66 * q) + 55 ∧ q = 20 :=
by
  exists 20
  split
  · norm_num
  · rfl

end find_quotient_l447_447954


namespace domain_of_f_correct_g_max_min_values_l447_447128

noncomputable theory

open Real

def domain_of_f : Set ℝ := {x | 4 * x - 1 > 0 ∧ log 3 (4 * x - 1) ≥ 0 ∧ 16 - 2^x ≥ 0 }

def function_g (x : ℝ) := (log2 x)^2 - 2 * (log2 x) - 1

theorem domain_of_f_correct :
  domain_of_f = {x | x ≥ 1 / 2 ∧ x ≤ 4} :=
sorry

theorem g_max_min_values :
  ∀ x ∈ {a | a ≥ 1 / 2 ∧ a ≤ 4},
  (∃ y, function_g x = y ∧ ( y = 2 ∧ x = 1 / 2 ∨ y = -2 ∧ x = 2)) :=
sorry

end domain_of_f_correct_g_max_min_values_l447_447128


namespace nine_exp_inverse_l447_447574

theorem nine_exp_inverse (y : ℝ) (h : 9^(2 * y) = 81) : 9^(-y) = 1 / 9 := by
  sorry

end nine_exp_inverse_l447_447574


namespace find_range_a_l447_447088

noncomputable def sincos_inequality (x a θ : ℝ) : Prop :=
  (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1 / 8

theorem find_range_a :
  (∀ (x : ℝ) (θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → sincos_inequality x a θ)
  ↔ a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end find_range_a_l447_447088


namespace incorrect_statement_D_l447_447158

noncomputable def incorrect_statement {Line Plane : Type} 
  (m n : Line) (α β : Plane) : Prop :=
(α ≠ β) ∧ (m ≠ n) ∧ (α ∩ β = m) ∧ (makes_equal_angles_with_planes n α β) ∧ (¬ (m ⊥ n))

-- Define makes_equal_angles_with_planes as a function that checks if the line makes equal angles with given planes.
axiom makes_equal_angles_with_planes (n : Line) (α β : Plane) : Prop

theorem incorrect_statement_D {Line Plane : Type}
  (m n : Line) (α β : Plane) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : α ∩ β = m) 
  (h4 : makes_equal_angles_with_planes n α β) : 
  ¬ (m ⊥ n) :=
sorry

end incorrect_statement_D_l447_447158


namespace sum_even_integers_202_to_1000_l447_447750

theorem sum_even_integers_202_to_1000 :
  (∑ k in finset.Icc 202 1000, if even k then k else 0) = 240400 :=
by
  -- skip the proof for now
  sorry

end sum_even_integers_202_to_1000_l447_447750


namespace cafeteria_earnings_l447_447720

-- Definitions based on given conditions
def initial_apples : ℕ := 50
def initial_oranges : ℕ := 40
def apple_cost : ℝ := 0.80
def orange_cost : ℝ := 0.50
def final_apples : ℕ := 10
def final_oranges : ℕ := 6

-- Statement of the problem as a theorem
theorem cafeteria_earnings : 
  let apples_sold := initial_apples - final_apples in
  let oranges_sold := initial_oranges - final_oranges in
  let earnings_apples := apples_sold * apple_cost in
  let earnings_oranges := oranges_sold * orange_cost in
  let total_earnings := earnings_apples + earnings_oranges in
  total_earnings = 49 :=
by
  sorry

end cafeteria_earnings_l447_447720


namespace new_alcohol_concentration_l447_447007

variables (capacity_A capacity_B capacity_C capacity_D : ℝ)
variables (alcohol_concentration_A alcohol_concentration_B alcohol_concentration_C : ℝ)

-- Conditions
def vessel_A := capacity_A = 5 ∧ alcohol_concentration_A = 0.25
def vessel_B := capacity_B = 12 ∧ alcohol_concentration_B = 0.45
def vessel_C := capacity_C = 7 ∧ alcohol_concentration_C = 0.35
def vessel_D := capacity_D = 26

-- Theorem
theorem new_alcohol_concentration
  (vA: vessel_A)
  (vB: vessel_B)
  (vC: vessel_C)
  (vD: vessel_D) : 
  ((0.25 * 5 + 0.45 * 12 + 0.35 * 7) / 26) * 100 = 35 :=
by
  sorry

end new_alcohol_concentration_l447_447007


namespace max_integer_value_of_f_l447_447423

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5)

theorem max_integer_value_of_f :
  ∃ n : ℤ, n = 17 ∧ ∀ x : ℝ, f x ≤ (n : ℝ) :=
by
  sorry

end max_integer_value_of_f_l447_447423


namespace expression_value_l447_447540

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447540


namespace unique_triangle_exists_l447_447677

-- Definition of a triangle with consecutive side lengths and angle constraints
structure TriangularProps (a b c : ℕ) (A B C : ℝ) where
  sides_consecutive : a + 1 = b ∧ b + 1 = c
  angles_relation : A = 2 * B

theorem unique_triangle_exists :
  ∃! (a b c : ℕ) (A B C : ℝ), TriangularProps a b c A B C := sorry

end unique_triangle_exists_l447_447677


namespace g_g_g_g_15_eq_3_l447_447656

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_g_g_g_15_eq_3 : g (g (g (g 15))) = 3 := 
by
  sorry

end g_g_g_g_15_eq_3_l447_447656


namespace solve_for_x_y_l447_447824

def ceil (m : ℝ) : ℤ := ⌈m⌉
def floor (m : ℝ) : ℤ := ⌊m⌋

theorem solve_for_x_y (x y : ℝ) (hx : 3 * floor x + 2 * ceil y = 2003) (hy : 2 * ceil x - floor y = 2001) :
  x + y = 572 :=
by
  sorry

end solve_for_x_y_l447_447824


namespace mark_all_points_l447_447665

theorem mark_all_points (d : ℕ) (h_coprime : Nat.gcd d 1001 = 1) : 
  ∃ (marks : finset ℕ), (∀ n ∈ marks, n ∈ finset.Icc 0 2002) ∧ 
  (∀ n ∈ finset.Icc 0 2002, n ∈ marks) :=
begin
  -- Yes, it is possible to mark all integer points.
  sorry
end

end mark_all_points_l447_447665


namespace quadratic_inequality_solution_set_l447_447949

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set (h : ∀ x, x > -1 ∧ x < 2 → ax^2 - bx + c > 0) :
  a + b + c = 0 :=
sorry

end quadratic_inequality_solution_set_l447_447949


namespace monkey_bananas_max_l447_447368

noncomputable def max_bananas_home : ℕ :=
  let total_bananas := 100
  let distance := 50
  let carry_capacity := 50
  let consumption_rate := 1
  let distance_each_way := distance / 2
  let bananas_eaten_each_way := distance_each_way * consumption_rate
  let bananas_left_midway := total_bananas / 2 - bananas_eaten_each_way
  let bananas_picked_midway := bananas_left_midway * 2
  let bananas_left_home := bananas_picked_midway - distance_each_way * consumption_rate
  bananas_left_home

theorem monkey_bananas_max : max_bananas_home = 25 :=
  sorry

end monkey_bananas_max_l447_447368


namespace proof_equivalent_l447_447863

-- Declare the real number variables and assumptions
variables (x y z w v : ℝ)
  (hxy : x > y) (hyz : y > z) (hzw : z > w) (hwv : w > v)
  (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ w ≠ v ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ w ∧ y ≠ v ∧ z ≠ v)

-- Define the min and max functions as used in the problem
def m (a b : ℝ) := min a b
def M (a b : ℝ) := max a b

-- State the theorem
theorem proof_equivalent :
  M (m y z) (M w (m x v)) = w :=
by sorry

end proof_equivalent_l447_447863


namespace nonneg_or_nonpos_l447_447252

theorem nonneg_or_nonpos (n : ℕ) (h : n ≥ 2) (c : Fin n → ℝ)
  (h_eq : (n - 1) * (Finset.univ.sum (fun i => c i ^ 2)) = (Finset.univ.sum c) ^ 2) :
  (∀ i, c i ≥ 0) ∨ (∀ i, c i ≤ 0) := 
  sorry

end nonneg_or_nonpos_l447_447252


namespace maximize_T_n_l447_447477

variable {a_n : ℕ → ℝ} (n : ℕ)
variable (T_n : ℕ → ℝ)
variable (q : ℝ) (a_2 a_6 : ℝ)
variable (product := λ n, (∏ i in Finset.range n, a_n i))

/-- Given that all terms a_n are positive in the geometric sequence,
    a_2 = 125,
    a_3 * a_6 * a_9 = 1/125,
    and the product of the first n terms is T_n,
    we want to prove that the value of n that maximizes T_n is 4 or 5. -/
theorem maximize_T_n :
  (∀ m, 0 < a_n m) →
  a_n 2 = 125 →
  a_n 3 * a_n 6 * a_n 9 = 1 / 125 →
  (∀ n, T_n n = product n) →
  let q := 1 / 5 in
  let a_6 := 1 / 5 in
  a_n 2 = 125 → -- condition a) appears once considering definitions
  (T_n 4 = T_n 5) ∧ 
  ((∀ k, k > 5 → T_n k < T_n 4) : Prop) ∧ 
  ((∀ k, k < 4 → T_n k < T_n 4) : Prop) :=
by
  sorry

end maximize_T_n_l447_447477


namespace impossible_four_teams_tie_possible_three_teams_tie_l447_447439

-- Definitions for the conditions
def num_teams : ℕ := 4
def num_matches : ℕ := (num_teams * (num_teams - 1)) / 2
def total_possible_outcomes : ℕ := 2^num_matches
def winning_rate : ℚ := 1 / 2

-- Problem 1: It is impossible for exactly four teams to tie for first place.
theorem impossible_four_teams_tie :
  ¬ ∃ (score : ℕ), (∀ (team : ℕ) (h : team < num_teams), team = score ∧
                     (num_teams * score = num_matches / 2 ∧
                      num_teams * score + num_matches / 2 = num_matches)) := sorry

-- Problem 2: It is possible for exactly three teams to tie for first place.
theorem possible_three_teams_tie :
  ∃ (score : ℕ), (∃ (teamA teamB teamC teamD : ℕ),
  (teamA < num_teams ∧ teamB < num_teams ∧ teamC < num_teams ∧ teamD <num_teams ∧ teamA ≠ teamB ∧ teamA ≠ teamC ∧ teamA ≠ teamD ∧ 
  teamB ≠ teamC ∧ teamB ≠ teamD ∧ teamC ≠ teamD)) ∧
  (teamA = score ∧ teamB = score ∧ teamC = score ∧ teamD = 0) := sorry

end impossible_four_teams_tie_possible_three_teams_tie_l447_447439


namespace find_CD_l447_447782

-- Geometry definitions and statements
variable {Point : Type} [MetricSpace Point]
variable (O A B C D : Point) (r : ℝ)

-- Conditions
def semidiameter (A B : Point) := dist A B = 100
def on_semicircle (C D O : Point) := dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r
def length_AC (A C : Point) := dist A C = 28
def length_BD (B D : Point) := dist B D = 60

-- The proof goal
theorem find_CD (h1 : semidiameter A B) (h2 : on_semicircle C D O) (h3 : length_AC A C) (h4 : length_BD B D) :
  dist C D = 60 := by
  sorry

end find_CD_l447_447782


namespace total_handshakes_l447_447393

-- Definition of the teams and handshakes
def num_teams : ℕ := 4

def num_handshakes_per_man (num_teams : ℕ) : ℕ := num_teams - 1

theorem total_handshakes (num_teams : ℕ) : 
  (num_teams > 0) → 
  num_teams * num_handshakes_per_man(num_teams) = 12 := 
by
  intros h
  -- skipped proof
  sorry

end total_handshakes_l447_447393


namespace max_weight_on_bar_l447_447628

theorem max_weight_on_bar (max_weight : ℕ) (safety_margin_two_people : ℚ) (john_weight mike_weight : ℕ) :
  max_weight = 1000 →
  safety_margin_two_people = 0.70 →
  john_weight = 250 →
  mike_weight = 180 →
  (max_weight * safety_margin_two_people - (john_weight + mike_weight) = 270) :=
  by
    intros h_max_weight h_safety_margin h_john_weight h_mike_weight
    have h1 : max_weight * safety_margin_two_people = 700 := by
      rw [h_max_weight, h_safety_margin]
      norm_num
    have h2 : john_weight + mike_weight = 430 := by
      rw [h_john_weight, h_mike_weight]
      norm_num
    have h3 : 700 - 430 = 270 := by norm_num
    rw [← h1, ← h2, h3]
    exact h3

end max_weight_on_bar_l447_447628


namespace invariant_lines_area_enclosed_infinite_sum_l447_447991

-- Define the linear transformation matrix A_n
def A (n : ℕ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![(1 - n : ℝ), 1; -((n : ℝ)*(n + 1)), (n + 2)]

-- Prove that there exist 2 invariant lines passing through the origin
theorem invariant_lines (n : ℕ):
  ∃ m₁ m₂ : ℝ, (m₁ = n) ∧ (m₂ = n + 1) :=
  sorry

-- Prove the area S_n of the figure enclosed by the lines and the curve y = x^2
theorem area_enclosed (n : ℕ):
  S_n = (1 / 6) * (3 * n * n + 3 * n + 1) :=
  sorry

-- Prove the infinite sum of 1 / (S_n - 1/6) equals 2
theorem infinite_sum:
  ∑' (n : ℕ) in (fun n => 1 / (S_n - (1 / 6))) = 2 := 
  sorry


end invariant_lines_area_enclosed_infinite_sum_l447_447991


namespace intersection_M_N_l447_447501

noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {x | abs x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l447_447501


namespace expected_value_of_X_l447_447031

noncomputable def expected_value_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

theorem expected_value_of_X (n : ℕ) (h : n > 0) :
  let X : ℕ → ℕ → ℚ := λ i n, (i - 1 : ℚ) / n
  let E_X : ℚ := ∑ i in Finset.range n, X (i + 1) n
  E_X = expected_value_shorter_gentlemen n := by
{
  sorry
}

end expected_value_of_X_l447_447031


namespace math_problem_l447_447630

noncomputable def A (x y : ℝ) := 0.5 * x * y
noncomputable def B (w z : ℝ) := 0.5 * w * z
noncomputable def P (x y z w : ℝ) := 0.25 * (x^2 + y^2 - z^2 - w^2)

theorem math_problem
(A B P θ₁ θ₂ : ℝ)
(h₀ : 0 < x) 
(h₁ : x ≤ y) 
(h₂ : y ≤ x + z + w) 
(h₃ : 0 < z) 
(h₄ : 0 < w) 
(h₅ : z^2 + w^2 < x^2 + y^2)
(h₆ : A (x y) = 0.5 * x * y)
(h₇ : B (w z) = 0.5 * w * z)
(h₈ : P (x y z w) = 0.25 * (x^2 + y^2 - z^2 - w^2))
(h₉ : P ≤ A + B)
(h₁₀ : A * cos θ₁ + B * cos θ₂ = P) : 
  (A * sin θ₁ + B * sin θ₂ ≤ sqrt ((A + B - P) * (A + B + P))) ∧
  ((θ₁ = θ₂ = arccos (P / (A + B))) → (A * sin θ₁ + B * sin θ₂ = sqrt ((A + B - P) * (A + B + P)))) ∧
  (∀ (x y z w : ℝ), (0 < x ∧ x ≤ y ∧ y ≤ x + z + w ∧ 0 < z ∧ 0 < w ∧ z^2 + w^2 < x^2 + y^2) →
     (cyclic_quadrilateral_max_area x y z w)) :=
sorry

end math_problem_l447_447630


namespace number_of_blind_students_l447_447004

variable (B D : ℕ)

-- Condition 1: The deaf-student population is 3 times the blind-student population.
axiom H1 : D = 3 * B

-- Condition 2: There are 180 students in total.
axiom H2 : B + D = 180

theorem number_of_blind_students : B = 45 :=
by
  -- Sorry is used to skip the proof steps. The theorem statement is correct and complete based on the conditions.
  sorry

end number_of_blind_students_l447_447004


namespace vacation_trip_l447_447661

theorem vacation_trip (airbnb_cost : ℕ) (car_rental_cost : ℕ) (share_per_person : ℕ) (total_people : ℕ) :
  airbnb_cost = 3200 → car_rental_cost = 800 → share_per_person = 500 → airbnb_cost + car_rental_cost / share_per_person = 8 :=
by
  intros h1 h2 h3
  sorry

end vacation_trip_l447_447661


namespace min_ones_binary_anti_pascal_triangle_l447_447456

theorem min_ones_binary_anti_pascal_triangle (n : ℕ) (h : 0 < n) :
  ∃ a : ℕ → ℕ → ℕ, 
    (∀ i j, i + j ≤ n + 1 → a i j = 0 ∨ a i j = 1) ∧
    (∀ i j, i + j ≤ n → (a i j + a i (j + 1) + a (i + 1) j ≡ 1 [MOD 2])) ∧
    ∑ i in finset.range n.succ, ∑ j in finset.range (n + 1 - i), a i j = nat.floor ((n * (n + 1) / 6 : ℚ)) :=
sorry

end min_ones_binary_anti_pascal_triangle_l447_447456


namespace compare_magnitude_l447_447868

theorem compare_magnitude (a b : ℝ) (h : a ≠ 1) : a^2 + b^2 > 2 * (a - b - 1) :=
by
  sorry

end compare_magnitude_l447_447868


namespace boris_climbs_4_times_l447_447936

theorem boris_climbs_4_times
  (hugo_elevation : ℕ) (boris_elevation_difference : ℕ) (hugo_climbs : ℕ)
  (hugo_elevation_is_10000 : hugo_elevation = 10000)
  (boris_elevation_is_shorter : boris_elevation_difference = 2500)
  (hugo_climbs_is_3 : hugo_climbs = 3):
  let boris_elevation := hugo_elevation - boris_elevation_difference in
  let hugo_total_climbed := hugo_elevation * hugo_climbs in
  let boris_climbs_needed := hugo_total_climbed / boris_elevation in
  boris_climbs_needed = 4 :=
by
  sorry

end boris_climbs_4_times_l447_447936


namespace exists_large_n_l447_447831

noncomputable def large_n_exists (m : ℕ) (h : m > 10^98) : ℕ := 100 * m + 3861

theorem exists_large_n :
  ∃ n : ℕ, (n > 10^100) ∧ (∀ d : ℕ, d ∈ digit_frequencies n^2 ↔ d ∈ digit_frequencies (n + 1)^2) :=
begin
  use large_n_exists (10^98 + 1) (nat.lt_add_one (10^98)),
  split,
  { simp [large_n_exists],
    exact nat.mul_lt_mul_of_pos_left (by norm_num) (by norm_num) },
  { sorry },  -- This part will prove the digit frequency condition
end

end exists_large_n_l447_447831


namespace distance_between_hyperbola_vertices_l447_447081

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l447_447081


namespace derivative_at_3pi_over_2_l447_447448

-- Define the function f(x) = x * sin x
def f (x : ℝ) : ℝ := x * Real.sin x

-- State the theorem we need to prove
theorem derivative_at_3pi_over_2 : deriv f (3 * Real.pi / 2) = -1 :=
by
  sorry

end derivative_at_3pi_over_2_l447_447448


namespace range_of_m_l447_447584

open Real

theorem range_of_m (m : ℝ) : (¬ ∃ x₀ : ℝ, m * x₀^2 + m * x₀ + 1 ≤ 0) ↔ (0 ≤ m ∧ m < 4) := by
  sorry

end range_of_m_l447_447584


namespace solve_for_a_l447_447657

def i : ℂ := complex.I
def complex_expr (a : ℝ) : ℂ := (a-i)/(2+i)
def real_part_opposite_imaginary_part (z : ℂ) : Prop := z.re = -z.im

theorem solve_for_a :
  ∃ a : ℝ, real_part_opposite_imaginary_part (complex_expr a) ∧ a = 3 :=
by
  sorry

end solve_for_a_l447_447657


namespace brownies_shared_l447_447737

theorem brownies_shared
  (total_brownies : ℕ)
  (tina_brownies : ℕ)
  (husband_brownies : ℕ)
  (remaining_brownies : ℕ)
  (shared_brownies : ℕ)
  (h1 : total_brownies = 24)
  (h2 : tina_brownies = 10)
  (h3 : husband_brownies = 5)
  (h4 : remaining_brownies = 5) :
  shared_brownies = total_brownies - (tina_brownies + husband_brownies + remaining_brownies) → shared_brownies = 4 :=
by
  sorry

end brownies_shared_l447_447737


namespace solve_for_a_l447_447408

def g(x : ℝ) := 3 * x - 4

theorem solve_for_a : ∃ a : ℝ, g(a) = 0 ∧ a = 4 / 3 :=
by
  use 4 / 3
  sorry

end solve_for_a_l447_447408


namespace maximum_value_l447_447323

-- Conditions
def f (x : ℝ) (b : ℝ) : ℝ := x^3 + b * x^2 - 12 * x
def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop := deriv f x = 0
def max_value_on_interval (f : ℝ → ℝ) (a b : ℝ) : ℝ := max (f a) (f b)

-- Statement
theorem maximum_value (b : ℝ) (h_extremum : has_extremum_at (λ x, f x b) 2)
  (h_interval : ∀ x : ℝ, -4 ≤ x → x ≤ 4) :
  max_value_on_interval (λ x, f x 0) (-4) 4 = 16 :=
by sorry

end maximum_value_l447_447323


namespace minimum_highways_l447_447663

open SimpleGraph

theorem minimum_highways (V : Finset ℕ) (hV : V.card = 10)
  (H : ∀ (u v w : V), u ≠ v → u ≠ w → v ≠ w →
   (G.adj u v ∧ G.adj v w ∧ G.adj w u) ∨ (¬G.adj u v ∧ ¬G.adj v w ∧ G.adj w u)): 
  ∃ (G : SimpleGraph V), G.edge_finset.card = 40 := by 
  sorry

end minimum_highways_l447_447663


namespace problem_1_problem_2_l447_447952

noncomputable def problem_conditions (a c : ℝ) (h1 : a + c = 3 * Real.sqrt 3)
  (b : ℝ) (h2 : b = 3) (dot_product : ℝ) (h3 : a * c * Real.cos B = 3) : Prop :=
let cos_B_min : ℝ := 1 / 3 in 
let angle_A : ℝ := if dot_product = 3 then math.pi / 2 else math.pi / 6 in
true

theorem problem_1 (a c : ℝ) (h1 : a + c = 3 * Real.sqrt 3) 
  (b : ℝ) (h2 : b = 3) (dot_product : ℝ) (h3 : a * c * Real.cos (acos (dot_product / (a * c))) = 3) : 
  ∃ (cos_B_min : ℝ), cos_B_min = 1 / 3 :=
sorry

theorem problem_2 (a c : ℝ) (h1 : a + c = 3 * Real.sqrt 3)
  (b : ℝ) (h2 : b = 3) (h3 : a * c * Real.cos (acos ((3/(a*c))) = 3)) :
  ∃ (angle_A : ℝ), angle_A = (if a = 2 * Real.sqrt 3 ∧ c = Real.sqrt 3 ∨ a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 
  then (math.pi / 2, math.pi / 6) :=
 sorry

end problem_1_problem_2_l447_447952


namespace complex_modulus_is_sqrt_5_l447_447124

noncomputable def complex_z : ℂ := (1 + 2 * complex.I)^2 / (-complex.I + 2)

theorem complex_modulus_is_sqrt_5 : complex.abs complex_z = real.sqrt 5 :=
by sorry

end complex_modulus_is_sqrt_5_l447_447124


namespace total_amount_paid_is_correct_l447_447300

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l447_447300


namespace hyperbola_vertices_distance_l447_447076

/--
For the hyperbola given by the equation
(x^2 / 121) - (y^2 / 49) = 1,
the distance between its vertices is 22.
-/
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ,
  (x^2 / 121) - (y^2 / 49) = 1 →
  ∃ a : ℝ, a = 11 ∧ 2 * a = 22 :=
by
  intros x y h
  use 11
  split
  · refl
  · norm_num

end hyperbola_vertices_distance_l447_447076


namespace problem_solution_l447_447046

noncomputable def given_problem : ℝ := (Real.pi - 3)^0 - Real.sqrt 8 + 2 * Real.sin (45 * Real.pi / 180) + (1 / 2)⁻¹

theorem problem_solution : given_problem = 3 - Real.sqrt 2 := by
  sorry

end problem_solution_l447_447046


namespace problem_statement_l447_447516

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447516


namespace perfect_square_probability_l447_447441

noncomputable def card_set : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem perfect_square_probability :
  let draws := {x | x ∈ (card_set.powerset.filter (λ s, s.card = 2))},
      is_perfect_square := (λ s, (s.to_list.prod) ∈ ({n : ℕ | ∃ m, m^2 = n} : set ℕ)) in
  ((({s | s ∈ draws ∧ is_perfect_square s}).card : ℚ) / (draws.card)) = (1 / 14) :=
by
  -- The proof goes here
  sorry

end perfect_square_probability_l447_447441


namespace jane_cycling_time_difference_l447_447201

theorem jane_cycling_time_difference :
  (3 * 5 / 6.5 - (5 / 10 + 5 / 5 + 5 / 8)) * 60 = 11 :=
by sorry

end jane_cycling_time_difference_l447_447201


namespace lunch_break_duration_is_48_l447_447247

noncomputable def solve_lunch_break_duration : ℕ :=
  let p := 1 / (9.0 / 5.0)
  let h := (16 / 9) * p
  let L := (11.2 - ((1 - (5 / 2)) / p)) * 60
  in 48

theorem lunch_break_duration_is_48 : solve_lunch_break_duration = 48 :=
by
  -- We skip the proof steps here as instructed
  sorry

end lunch_break_duration_is_48_l447_447247


namespace find_m_l447_447570

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (-1, -1)
noncomputable def a_minus_b : ℝ × ℝ := (2, 3)
noncomputable def m_a_plus_b (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m - 1)

theorem find_m (m : ℝ) : (a_minus_b.1 * (m_a_plus_b m).1 + a_minus_b.2 * (m_a_plus_b m).2) = 0 → m = 5 / 8 := 
by
  sorry

end find_m_l447_447570


namespace seventeenth_number_is_6834_l447_447727

theorem seventeenth_number_is_6834 :
  ∃ (num_list : List ℕ), 
  num_list = List.permutations [3, 4, 6, 8].bind (λ l, [1000*l.head + 100*l.tail.head + 10*l.tail.tail.head + l.tail.tail.tail.head]) ∧
  List.nth num_list 16 = some 6834 := by
  sorry

end seventeenth_number_is_6834_l447_447727


namespace min_value_of_k_l447_447598

-- Given definitions based on conditions
variable (students : Finset ℕ) (clubs : Finset (Finset ℕ))
variable (student_count : ℕ := 1200)
variable (club_membership_count : ℕ := 23)
variable (k : ℕ)

-- Condition: There are 1200 students
axiom h1 : students.card = student_count

-- Condition: Each student must join exactly k clubs
axiom h2 : ∀ s ∈ students, (Finset.filter (λ c, s ∈ c) clubs).card = k

-- Condition: Each club is joined by exactly 23 students
axiom h3 : ∀ c ∈ clubs, c.card = club_membership_count

-- Condition: No club is joined by all 1200 students
axiom h4 : ∀ c ∈ clubs, ¬(students ⊆ c)

-- Prove the smallest possible value of k is 23
theorem min_value_of_k : k = 23 :=
sorry

end min_value_of_k_l447_447598


namespace find_z_l447_447699

-- We define vectors a and b
def a (z : ℝ) : Vector3 := ⟨0, 4, z⟩
def b : Vector3 := ⟨-4, 6, -2⟩

-- Projection formula
def proj (u v : Vector3) : Vector3 :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let dot_vv := v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2
  (dot_uv / dot_vv) • v

-- Given condition of the projection being a specific vector
theorem find_z (z : ℝ) (h : proj (a z) b = (16 / 56) • b) : z = 4 :=
sorry

end find_z_l447_447699


namespace solve_system_l447_447261

theorem solve_system :
  ∃ (x y : ℝ), 
    (log x y2 (x - y) = 5 - log x y2 (x + y)) ∧ 
    ((log 10 (x) - log 10 4) / (log 10 (y) - log 10 3) = -1) ∧ 
    (x - y > 0) ∧ 
    (x + y > 0) ∧ 
    (x > 0) ∧ 
    (y > 0) ∧ 
    (y ≠ 3) ∧ 
    (x = 6) ∧ 
    (y = 2) 
    := sorry

end solve_system_l447_447261


namespace lunch_break_duration_is_48_l447_447246

noncomputable def solve_lunch_break_duration : ℕ :=
  let p := 1 / (9.0 / 5.0)
  let h := (16 / 9) * p
  let L := (11.2 - ((1 - (5 / 2)) / p)) * 60
  in 48

theorem lunch_break_duration_is_48 : solve_lunch_break_duration = 48 :=
by
  -- We skip the proof steps here as instructed
  sorry

end lunch_break_duration_is_48_l447_447246


namespace value_of_x_l447_447414

theorem value_of_x (x : ℝ) : abs (4 * x - 8) ≤ 0 ↔ x = 2 :=
by {
  sorry
}

end value_of_x_l447_447414


namespace cost_of_one_pie_l447_447233

theorem cost_of_one_pie (x c2 c5 : ℕ) 
  (h1: 4 * x = c2 + 60)
  (h2: 5 * x = c5 + 60) 
  (h3: 6 * x = c2 + c5 + 60) : 
  x = 20 :=
by
  sorry

end cost_of_one_pie_l447_447233


namespace value_of_b_l447_447338

theorem value_of_b (a c : ℝ) (b : ℝ) (h1 : a = 105) (h2 : c = 70) (h3 : a^4 = 21 * 25 * 15 * b * c^3) : b = 0.045 :=
by
  sorry

end value_of_b_l447_447338


namespace consecutive_integers_sum_256_l447_447854

theorem consecutive_integers_sum_256 :
  (∃ (n a : ℕ), n ≥ 2 ∧ a > 0 ∧ n * (2 * a + n - 1) = 512) → 2 :=
by sorry

end consecutive_integers_sum_256_l447_447854


namespace math_problem_l447_447042

theorem math_problem :
  (π - 3.14)^0 + | -Real.sqrt 3 | - (1 / 2 : ℝ)^(-1) - Real.sin (Real.pi / 3) = -1 + Real.sqrt 3 / 2 :=
by
  sorry

end math_problem_l447_447042


namespace cn_squared_eq_28_l447_447575

theorem cn_squared_eq_28 (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end cn_squared_eq_28_l447_447575


namespace edwards_total_number_of_good_games_l447_447832

def number_of_purchased_games_from_friend := 50
def number_of_purchased_games_from_garage_sale := 30
def number_of_purchased_games_from_online_store := 20

def defect_rate_friend := 0.60
def defect_rate_garage_sale := 0.25
def defect_rate_online_store := 0.10

def number_of_good_games_from_friend := (number_of_purchased_games_from_friend * (1 - defect_rate_friend)).toInt
def number_of_good_games_from_garage_sale := (number_of_purchased_games_from_garage_sale * (1 - defect_rate_garage_sale)).toInt
def number_of_good_games_from_online_store := (number_of_purchased_games_from_online_store * (1 - defect_rate_online_store)).toInt

def total_number_of_good_games := number_of_good_games_from_friend + number_of_good_games_from_garage_sale + number_of_good_games_from_online_store

theorem edwards_total_number_of_good_games : total_number_of_good_games = 61 := 
    by 
      simp [number_of_good_games_from_friend, number_of_good_games_from_garage_sale, number_of_good_games_from_online_store, total_number_of_good_games]
      sorry

end edwards_total_number_of_good_games_l447_447832


namespace john_age_l447_447433

-- Define the statement of the problem in Lean 4
theorem john_age : ∃ x : ℕ, (x - 5 = 1 / 2 * (x + 8)) ∧ x = 18 :=
by {
  use 18,
  split,
  { norm_num, },
  { refl, }
}

end john_age_l447_447433


namespace number_of_plants_l447_447716

--- The given problem conditions and respective proof setup
axiom green_leaves_per_plant : ℕ
axiom yellow_turn_fall_off : ℕ
axiom green_leaves_total : ℕ

def one_third (n : ℕ) : ℕ := n / 3

-- Specify the given conditions
axiom leaves_per_plant_cond : green_leaves_per_plant = 18
axiom fall_off_cond : yellow_turn_fall_off = one_third green_leaves_per_plant
axiom total_leaves_cond : green_leaves_total = 36

-- Proof statement for the number of tea leaf plants
theorem number_of_plants : 
  (green_leaves_per_plant - yellow_turn_fall_off) * 3 = green_leaves_total :=
by
  sorry

end number_of_plants_l447_447716


namespace tetrahedron_is_regular_l447_447788

theorem tetrahedron_is_regular
    (A B C D X : Type)
    (sphere_at_X_touches_edges : ∀ (E : Type), E ∈ {A, B, C, D} → touches (sphere X) (edge E))
    (vertex_spheres_touch_each_other : ∀ (P Q : Type), P ∈ {A, B, C, D} → Q ∈ {A, B, C, D} → P ≠ Q → touches (sphere P) (sphere Q))
    (sphere_at_X_touches_vertex_spheres : ∀ (P : Type), P ∈ {A, B, C, D} → touches (sphere X) (sphere P)):
    is_regular_tetrahedron (tetrahedron A B C D) :=
begin
  sorry
end

end tetrahedron_is_regular_l447_447788


namespace lucy_lovely_age_ratio_l447_447094

theorem lucy_lovely_age_ratio (L l : ℕ) (x : ℕ) (h1 : L = 50) (h2 : 45 = x * (l - 5)) (h3 : 60 = 2 * (l + 10)) :
  (45 / (l - 5)) = 3 :=
by
  sorry

end lucy_lovely_age_ratio_l447_447094


namespace sigma_has_prime_divisor_gt_two_pow_k_l447_447634

namespace divisor_sum

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def sigma (n : ℕ) : ℕ :=
  ∑ d in (finset.filter (λ x, x ∣ n) (finset.range (n+1))), d

theorem sigma_has_prime_divisor_gt_two_pow_k (k : ℕ) (h : 0 < k) :
  ∃ p : ℕ, is_prime p ∧ p ∣ sigma ((2^k)!) ∧ p > 2^k :=
sorry

end divisor_sum

end sigma_has_prime_divisor_gt_two_pow_k_l447_447634


namespace correct_net_is_C_l447_447752

-- Definitions of the conditions
def net_A : Prop := ∀ (cube : Cube), cube.has_two_holes_on_same_face
def net_B : Prop := ∀ (cube : Cube), cube.has_one_hole_on_edge_and_two_other_holes
def net_C : Prop := ∀ (cube : Cube), cube.has_holes_on_edges_of_four_faces
def net_D : Prop := ∀ (cube : Cube), cube.has_two_holes_on_same_face
def net_E : Prop := ∀ (cube : Cube), cube.has_hole_in_center_of_two_opposite_faces
def partial_cube : Prop := ∀ (cube : Cube), cube.has_holes_on_edges_of_four_faces

theorem correct_net_is_C : net_C → partial_cube := by
  sorry

end correct_net_is_C_l447_447752


namespace sum_of_solutions_l447_447090

theorem sum_of_solutions (a : ℝ) (h : 0 < a ∧ a < 1) :
  let x1 := 3 + a
  let x2 := 3 - a
  let x3 := 1 + a
  let x4 := 1 - a
  x1 + x2 + x3 + x4 = 8 :=
by
  intros
  sorry

end sum_of_solutions_l447_447090


namespace angle_A_is_60_degrees_l447_447172

theorem angle_A_is_60_degrees (a b c : ℝ) (h : (a + b + c) * (b + c - a) = 3 * b * c) :
  ∠A = 60 :=
by 
  sorry

end angle_A_is_60_degrees_l447_447172


namespace sum_of_c_n_l447_447890

variable {a_n : ℕ → ℕ}    -- Sequence {a_n}
variable {b_n : ℕ → ℕ}    -- Sequence {b_n}
variable {c_n : ℕ → ℕ}    -- Sequence {c_n}
variable {S_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {a_n}
variable {T_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {c_n}

axiom a3 : a_n 3 = 7
axiom S6 : S_n 6 = 48
axiom b_recur : ∀ n : ℕ, 2 * b_n (n + 1) = b_n n + 2
axiom b1 : b_n 1 = 3
axiom c_def : ∀ n : ℕ, c_n n = a_n n * (b_n n - 2)

theorem sum_of_c_n : ∀ n : ℕ, T_n n = 10 - (2*n + 5) * (1 / (2^(n-1))) :=
by
  -- Proof omitted
  sorry

end sum_of_c_n_l447_447890


namespace problem_proof_l447_447576

theorem problem_proof (n : ℕ) (h : n * (n - 1) = 42) : n.factorial / (3.factorial * (n - 3).factorial) = 35 :=
by
  sorry

end problem_proof_l447_447576


namespace sin_angle_APQ_is_correct_l447_447349

noncomputable def side_length : ℝ := 4
noncomputable def point_A := (0 : ℝ, 0 : ℝ)
noncomputable def point_B := (0 : ℝ, side_length)
noncomputable def point_C := (side_length, side_length)
noncomputable def point_D := (side_length, 0 : ℝ)

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def point_P := midpoint point_B point_C
noncomputable def point_Q := midpoint point_C point_D

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def angle_APQ := 
  let AP := dist point_A point_P
  let AQ := dist point_A point_Q
  let PQ := dist point_P point_Q
  Real.acos ((AP^2 + AQ^2 - PQ^2) / (2 * AP * AQ))

noncomputable def sin_phi := Real.sin angle_APQ

theorem sin_angle_APQ_is_correct :
  sin_phi = 3 / 5 := by
  sorry

end sin_angle_APQ_is_correct_l447_447349


namespace arithmetic_progression_sum_eq_l447_447643

variable {n : ℕ} (a : Fin n → ℝ)

def isArithmeticProgression (a : Fin n → ℝ) : Prop :=
  ∃ d : ℝ, ∀ k : Fin (n - 1), 0 < d ∧ a k.succ = a k + d

theorem arithmetic_progression_sum_eq (h : isArithmeticProgression a) :
  ∑ k in Finset.range (n - 1), 1 / (Real.sqrt (a ⟨k, Nat.lt_succ_iff.mpr (Nat.pred_lt_pred k.is_lt))⟩) + Real.sqrt (a ⟨k + 1, Fin.is_lt⟩)) =
    (n - 1) / (Real.sqrt (a 0) + Real.sqrt (a ⟨n - 1, Nat.pred_self_lt n⟩)) :=
sorry

end arithmetic_progression_sum_eq_l447_447643


namespace triathlete_average_speed_l447_447380

def harmonic_mean (a1 a2 a3 : ℝ) : ℝ :=
  3 / ((1 / a1) + (1 / a2) + (1 / a3))

def swimming_speed : ℝ := 3
def biking_speed : ℝ := 20
def running_speed : ℝ := 10

theorem triathlete_average_speed : harmonic_mean swimming_speed biking_speed running_speed = 6 := by
  sorry

end triathlete_average_speed_l447_447380


namespace cyclic_quadrilaterals_count_l447_447864

-- Define the types of quadrilaterals being considered
inductive Quadrilateral
  | RegularPentagon      -- Not actually a quadrilateral, keep for context
  | Kite
  | Rectangle
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a function to check if a quadrilateral is cyclic (can be inscribed in a circle)
def is_cyclic (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.RegularPentagon      => False  -- Regular pentagon is not a quadrilateral
  | Quadrilateral.Kite                 => True
  | Quadrilateral.Rectangle            => True
  | Quadrilateral.Parallelogram        => False  -- Assume generic parallelogram is not cyclic
  | Quadrilateral.IsoscelesTrapezoid   => False  -- Given no special conditions, generally not cyclic

-- Prove that there are exactly 2 quadrilaterals that are cyclic
theorem cyclic_quadrilaterals_count : (List.countp is_cyclic [
  Quadrilateral.RegularPentagon,
  Quadrilateral.Kite,
  Quadrilateral.Rectangle,
  Quadrilateral.Parallelogram,
  Quadrilateral.IsoscelesTrapezoid
]) = 2 := by
  sorry

end cyclic_quadrilaterals_count_l447_447864


namespace Q_7_value_l447_447224

theorem Q_7_value (g h i j k l : ℝ) (Q : ℝ → ℝ)
  (hQ_def : ∀ x, Q x = (3 * x^4 - 39 * x^3 + g * x^2 + h * x + i) * (4 * x^4 - 64 * x^3 + j * x^2 + k * x + l))
  (h_roots : ∀ x, Q x = 0 ↔ x ∈ ({2, 3, 4, 5, 6} : set ℝ)) :
  Q 7 = 23040 :=
by
  sorry

end Q_7_value_l447_447224


namespace math_problem_l447_447509

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447509


namespace log_inequality_solution_l447_447289

theorem log_inequality_solution :
  { x : ℝ | -1 < x ∧ x < 0 } = { x : ℝ | log 2 (1 - 1 / x) > 1 } :=
sorry

end log_inequality_solution_l447_447289


namespace dot_product_range_l447_447211

noncomputable section

variables {c d : Vector}
variables {θ : Real}

def norm_c : Real := 5
def norm_d : Real := 9

theorem dot_product_range (h1 : ∥c∥ = norm_c) (h2 : ∥d∥ = norm_d) :
  - norm_c * norm_d  ≤ c • d ∧ c • d ≤ norm_c * norm_d :=
by
  sorry

end dot_product_range_l447_447211


namespace monotonicity_of_f_range_of_a_l447_447915

def f (a x : ℝ) : ℝ := (1/3)*x^3 - (1+a)*x^2 + 4*a*x + 24*a

theorem monotonicity_of_f (a : ℝ) (h : a > 1) : 
  (∀ x : ℝ, 2 < x ∧ x < 2*a → deriv (f a) x < 0) ∧ 
  ((∀ x : ℝ, x < 2 → deriv (f a) x > 0) ∧ 
   (∀ x : ℝ, x > 2*a → deriv (f a) x > 0)) :=
sorry

theorem range_of_a (a : ℝ) (h : a > 1) : (∀ x : ℝ, x ≥ 0 → f a x > 0) ↔ 1 < a ∧ a < 6 :=
sorry

end monotonicity_of_f_range_of_a_l447_447915


namespace value_of_h_h_2_is_353_l447_447217

def h (x : ℕ) : ℕ := 3 * x^2 - x + 1

theorem value_of_h_h_2_is_353 : h (h 2) = 353 := 
by
  sorry

end value_of_h_h_2_is_353_l447_447217


namespace ratio_fraction_4A3B_5C2A_l447_447159

def ratio (a b c : ℝ) := a / b = 3 / 2 ∧ b / c = 2 / 6 ∧ a / c = 3 / 6

theorem ratio_fraction_4A3B_5C2A (A B C : ℝ) (h : ratio A B C) : (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := 
  sorry

end ratio_fraction_4A3B_5C2A_l447_447159


namespace prove_discount_rate_l447_447440

-- Define the given problem conditions
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def total_savings : ℝ := 8.55
def total_discount_rate : ℝ := 22

-- Define the discount rates for Fox and Pony jeans as variables
variables (F P : ℝ)

-- Define the conditions as Lean statements
def condition1 : Prop := F + P = total_discount_rate
def condition2 : Prop := 3 * (fox_price * (F / 100)) + 2 * (pony_price * (P / 100)) = total_savings

-- Define the main proposition
def discount_rate_on_pony_jeans (P : ℝ) : Prop := 
  F = 22 - P ∧ 
  3 * (15 * (F / 100)) + 2 * (18 * (P / 100)) = 8.55 ∧ 
  P = 15

-- Lean statement to prove the proposition under the given conditions
theorem prove_discount_rate : ∃ (P : ℝ), condition1 ∧ condition2 → discount_rate_on_pony_jeans P :=
begin
  sorry
end

end prove_discount_rate_l447_447440


namespace total_amount_paid_l447_447294

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l447_447294


namespace total_amount_paid_l447_447297

noncomputable def cost_per_night_per_person : ℕ := 40
noncomputable def number_of_people : ℕ := 3
noncomputable def number_of_nights : ℕ := 3

theorem total_amount_paid (cost_per_night_per_person number_of_people number_of_nights : ℕ) :
  (cost_per_night_per_person * number_of_people * number_of_nights = 360) :=
by
  have h : cost_per_night_per_person * number_of_people * number_of_nights = 40 * 3 * 3 := by rfl
  rw h
  exact rfl

end total_amount_paid_l447_447297


namespace order_of_magnitudes_l447_447874

theorem order_of_magnitudes (x y z : ℝ) (hx : x = 0.82 ^ 0.5) (hy : y = Real.sin 1) (hz : z = Real.log (3 ^ (1 / 2 * Real.log 7 / Real.log 3))) :
  y < z ∧ z < x := 
by {
  sorry
}

end order_of_magnitudes_l447_447874


namespace units_digit_of_m_squared_plus_two_to_the_m_is_seven_l447_447219

def m := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_the_m_is_seven :
  (m^2 + 2^m) % 10 = 7 := by
sorry

end units_digit_of_m_squared_plus_two_to_the_m_is_seven_l447_447219


namespace equal_angles_pac_qab_l447_447973

variables {A B C A' X B' C' P Q : Type}
variables [triangle_ABC : Triangle A B C]
variables [angle_bisector_AA' : IsAngleBisector A A']
variables [point_X_on_AA' : PointOnBisector X A A']
variables [B'_intersection_BX_AC : Intersection B' (Line B X) (Line A C)]
variables [C'_intersection_CX_AB : Intersection C' (Line C X) (Line A B)]
variables [P_intersection_A'B'_CC' : Intersection P (Line A' B') (Line C C')]
variables [Q_intersection_A'C'_BB' : Intersection Q (Line A' C') (Line B B')]

theorem equal_angles_pac_qab :
  angle PAC = angle QAB :=
sorry

end equal_angles_pac_qab_l447_447973


namespace unique_triangle_exists_l447_447676

-- Definition of a triangle with consecutive side lengths and angle constraints
structure TriangularProps (a b c : ℕ) (A B C : ℝ) where
  sides_consecutive : a + 1 = b ∧ b + 1 = c
  angles_relation : A = 2 * B

theorem unique_triangle_exists :
  ∃! (a b c : ℕ) (A B C : ℝ), TriangularProps a b c A B C := sorry

end unique_triangle_exists_l447_447676


namespace restore_triangle_ABC_l447_447196

-- Defining the basic elements
variables {A B C : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C]
variables (I L1 L2 L3 L4 L5 : A) (l_b l_c l_a : A → Prop)

-- Defining the conditions
def is_angle_bisector (l : A → Prop) (angle : A → Prop) : Prop := sorry
def foot_of_bisector (foot : A) (l : A → Prop) : Prop := sorry

-- Given conditions
axiom lb_is_bisector_B : is_angle_bisector l_b (angle B)
axiom lc_is_bisector_C : is_angle_bisector l_c (angle C)
axiom L1_is_foot_of_A_bisector : foot_of_bisector L1 l_a

-- Problem statement
theorem restore_triangle_ABC :
  ∃ (ABC : Type), 
    (∀ l₁ l₄ l₂ l₅ l₃ l₆ : A → Prop, 
      l₁ L1 → l₄ L4 → l₂ L2 → l₅ L5 → l₃ L3 → l₆ L6 → 
      (triangle_formed_by_lines ABC (L1L4) (L2L5) (L3L6))) :=
sorry

end restore_triangle_ABC_l447_447196


namespace find_smallest_positive_theta_l447_447431

noncomputable def sin_deg (x : ℝ) := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) := Real.cos (x * Real.pi / 180)
noncomputable def theta_solution := 45.0  -- degrees

theorem find_smallest_positive_theta :
  ∃ θ : ℝ, θ = theta_solution ∧ 0 < θ ∧ θ < 360 ∧
  cos_deg θ = sin_deg 45 + cos_deg 48 - sin_deg 18 - cos_deg 12 :=
by {
  existsi theta_solution,
  split,
  refl,
  split,
  norm_num,
  split,
  norm_num,
  sorry
}

end find_smallest_positive_theta_l447_447431


namespace number_of_arrangements_l447_447787

theorem number_of_arrangements :
  ∃ (n k : ℕ), n = 10 ∧ k = 5 ∧ Nat.choose n k = 252 := by
  sorry

end number_of_arrangements_l447_447787


namespace base_8_subtraction_l447_447063

theorem base_8_subtraction : 
  nat.sub 0o453 0o267 = 0o164 :=
by 
  sorry

end base_8_subtraction_l447_447063


namespace largest_perfect_square_factor_of_1764_l447_447749

theorem largest_perfect_square_factor_of_1764 : ∃ m, m * m = 1764 ∧ ∀ n, n * n ∣ 1764 → n * n ≤ 1764 :=
by
  sorry

end largest_perfect_square_factor_of_1764_l447_447749


namespace binom_22_5_l447_447469

theorem binom_22_5 : nat.choose 22 5 = 26334 :=
by
  have h1 : nat.choose 20 3 = 1140 := by sorry
  have h2 : nat.choose 20 4 = 4845 := by sorry
  have h3 : nat.choose 20 5 = 15504 := by sorry
  -- Use Pascal's identity
  have h4 : nat.choose 21 4 = nat.choose 20 3 + nat.choose 20 4 := by sorry
  rewrite [h1, h2] at h4
  have h5 : nat.choose 21 5 = nat.choose 20 4 + nat.choose 20 5 := by sorry
  rewrite [h2, h3] at h5
  have h6 : nat.choose 22 5 = nat.choose 21 4 + nat.choose 21 5 := by sorry
  rewrite [h4, h5] at h6
  -- Substitution
  rw [h4, h5]
  -- Verifying final computation
  have : 5985 + 20349 = 26334 := by sorry
  exact this

end binom_22_5_l447_447469


namespace batsman_average_after_17th_inning_l447_447336

-- Definition of the conditions
def score_in_17th_inning := 85
def increase_in_average := 3
def innings_before := 16
def total_innings := innings_before + 1

-- Formal statement of the proof problem
theorem batsman_average_after_17th_inning : 
  let initial_average := 34 in
  let final_average := initial_average + increase_in_average in
  final_average = 37 :=
by
  sorry

end batsman_average_after_17th_inning_l447_447336


namespace lunch_break_duration_is_48_minutes_l447_447244

-- Define the conditions
def three_painters (rate_p : ℚ) (rate_h : ℚ) (lunch_break : ℚ) : Prop :=
  -- Monday's condition
  (8 - lunch_break) * (rate_p + rate_h) = 0.5 ∧
  -- Tuesday's condition
  (6.2 - lunch_break) * rate_h = 0.24 ∧
  -- Wednesday's condition
  (11.2 - lunch_break) * rate_p = 0.26

-- Define the target theorem
theorem lunch_break_duration_is_48_minutes :
  ∃ lunch_break : ℚ,
    (∃ rate_p rate_h : ℚ, three_painters rate_p rate_h lunch_break) ∧
    lunch_break * 60 = 48 :=
begin
  -- The proof will involve showing the precise calculations as in the given solution.
  sorry
end

end lunch_break_duration_is_48_minutes_l447_447244


namespace perpendicular_pair_is_14_l447_447462

variable (x y : ℝ)

def equation1 := 4 * y - 3 * x = 16
def equation2 := -3 * x - 4 * y = 15
def equation3 := 4 * y + 3 * x = 16
def equation4 := 3 * y + 4 * x = 15

theorem perpendicular_pair_is_14 : (∃ y1 y2 x1 x2 : ℝ,
  4 * y1 - 3 * x1 = 16 ∧ 3 * y2 + 4 * x2 = 15 ∧ (3 / 4) * (-4 / 3) = -1) :=
sorry

end perpendicular_pair_is_14_l447_447462


namespace janet_earnings_per_hour_l447_447617

theorem janet_earnings_per_hour : 
  (∃ (rate_per_post : ℝ) (time_per_post : ℝ), 
    rate_per_post = 0.25 ∧ 
    time_per_post = 10 ∧ 
    (let posts_per_hour := 3600 / time_per_post in
     let earnings_per_hour := rate_per_post * posts_per_hour in
     earnings_per_hour = 90)) :=
by
  use 0.25
  use 10
  split
  rfl
  split
  rfl
  let posts_per_hour := 3600 / 10
  let earnings_per_hour := 0.25 * posts_per_hour
  have h : earnings_per_hour = 90, by sorry
  exact h

end janet_earnings_per_hour_l447_447617


namespace math_problem_l447_447793

open Real

theorem math_problem :
  (¬ (∀ x : ℝ, -2 < x → x < 1 → x > 1)) ∧
  (¬ (∃ x0 : ℝ, sin x0 > 1)) ∧
  (¬ (∀ x : ℝ, x = π / 4 → tan x = 1)) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = -f x) →
  ((f (log 3 2) + f (log 2 3) ≠ 0) → (∃! i, i ∈ {1, 2, 3, 4} ∧ true))) := 
sorry

end math_problem_l447_447793


namespace inv_seq_not_arith_seq_l447_447904

theorem inv_seq_not_arith_seq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_arith : ∃ d : ℝ, d ≠ 0 ∧ b = a + d ∧ c = a + 2 * d) :
  ¬ ∃ d' : ℝ, ∀ i j k : ℝ, i = 1 / a → j = 1 / b → k = 1 / c → j - i = d' ∧ k - j = d' :=
sorry

end inv_seq_not_arith_seq_l447_447904


namespace minimize_fraction_l447_447712

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) → (∀ m : ℕ, 0 < m → (n = m) → (3 * m + 27 / m ≥ 6)) := sorry

end minimize_fraction_l447_447712


namespace minimum_letters_for_grid_coloring_l447_447729

theorem minimum_letters_for_grid_coloring : 
  ∀ (grid_paper : Type) 
  (is_node : grid_paper → Prop) 
  (marked : grid_paper → Prop)
  (mark_with_letter : grid_paper → ℕ) 
  (connected : grid_paper → grid_paper → Prop), 
  (∀ n₁ n₂ : grid_paper, is_node n₁ → is_node n₂ → mark_with_letter n₁ = mark_with_letter n₂ → 
  (n₁ ≠ n₂ → ∃ n₃ : grid_paper, is_node n₃ ∧ connected n₁ n₃ ∧ connected n₃ n₂ ∧ mark_with_letter n₃ ≠ mark_with_letter n₁)) → 
  ∃ (k : ℕ), k = 2 :=
by
  sorry

end minimum_letters_for_grid_coloring_l447_447729


namespace Dorothy_found_57_pieces_l447_447802

def total_pieces_Dorothy_found 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) : ℕ := 
  let D_red := D_red_factor * (B_red + R_red)
  let D_blue := D_blue_factor * R_blue
  D_red + D_blue

theorem Dorothy_found_57_pieces 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) :
  total_pieces_Dorothy_found B_green B_red R_red R_blue D_red_factor D_blue_factor H1 H2 H3 H4 H5 H6 = 57 := by
    sorry

end Dorothy_found_57_pieces_l447_447802


namespace min_value_of_expression_l447_447464

theorem min_value_of_expression (x y : ℝ) (h : x^2 = 4 * y) :
  (∃ (y_min : ℝ), ∀ (x y : ℝ), x^2 = 4*y → y_min ≤ sqrt ((x - 3)^2 + (y - 1)^2) + y) ∧ 
  y_min = 2 :=
sorry

end min_value_of_expression_l447_447464


namespace total_length_of_shaded_border_l447_447287

-- Define the given conditions
def square_diagonal_length : ℝ := 1
def arc_count : ℕ := 8
def vertex_arc_count : ℕ := 4
def midpoint_arc_count : ℕ := 4
def radius_relation (s : ℝ) : ℝ := s / (2 * real.sqrt 2)

-- Define the lengths of the arcs based on the given conditions
def semicircle_arc_length (r : ℝ) : ℝ := vertex_arc_count * (real.pi * r)
def three_quarter_arc_length (r : ℝ) : ℝ := midpoint_arc_count * (3 / 2 * real.pi * r)

-- Define the total length of the arcs
def total_arc_length (s : ℝ) : ℝ :=
  let r := radius_relation s in
  semicircle_arc_length r + three_quarter_arc_length r

-- Theorem to be proven
theorem total_length_of_shaded_border (s : ℝ) (h : s = 1 / real.sqrt 2) : total_arc_length s = (5 / 2) * real.pi := by
  sorry

end total_length_of_shaded_border_l447_447287


namespace work_done_isothermal_l447_447668

variable (n : ℕ) (R T : ℝ) (P DeltaV : ℝ)

-- Definitions based on the conditions
def isobaric_work (P DeltaV : ℝ) := P * DeltaV

noncomputable def isobaric_heat (P DeltaV : ℝ) : ℝ :=
  (5 / 2) * P * DeltaV

noncomputable def isothermal_work (Q_iso : ℝ) : ℝ := Q_iso

theorem work_done_isothermal :
  ∃ (n R : ℝ) (P DeltaV : ℝ),
    isobaric_work P DeltaV = 20 ∧
    isothermal_work (isobaric_heat P DeltaV) = 50 :=
by 
  sorry

end work_done_isothermal_l447_447668


namespace math_problem_l447_447529

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447529


namespace min_val_of_f_l447_447853

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end min_val_of_f_l447_447853


namespace angle_opposite_c_is_zero_l447_447957

theorem angle_opposite_c_is_zero (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = 4 * a * b) : 
  ∠(a, b, c) = 0 :=
sorry

end angle_opposite_c_is_zero_l447_447957


namespace inequality_proof_l447_447103

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_abc : a * b * c = 1)

theorem inequality_proof :
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) 
  ≥ (3 / 2) + (1 / 4) * (a * (c - b) ^ 2 / (c + b) + b * (c - a) ^ 2 / (c + a) + c * (b - a) ^ 2 / (b + a)) :=
by
  sorry

end inequality_proof_l447_447103


namespace general_formula_sum_of_first_n_terms_l447_447905

variable (n : ℕ) (n_pos : n ≥ 1)
variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Define Sn
def S (n : ℕ) : ℕ := n^2 + n

-- Given Sn = n^2 + n
theorem general_formula
(S_n_def : ∀ (n : ℕ), S n = n^2 + n)
(n_pos : ∀ (n : ℕ), n ≥ 1) :
∀ (n : ℕ), a n = 2 * n := sorry

-- Given Sum of the first n terms of the sequence {1 / (an * a(n+1))} is Hn
-- where an = 2n
theorem sum_of_first_n_terms
(a_formula : ∀ (n : ℕ), a n = 2 * n)
(H : ℕ → ℕ)
(H_def : ∀ (n : ℕ), H n = ∑ i in range (n + 1), 1 / (a i * a (i + 1))) :
∀ (n : ℕ), H n = n / (4 * (n + 1)) := sorry

end general_formula_sum_of_first_n_terms_l447_447905


namespace find_sin_F_and_verify_triangle_dimensions_l447_447595

noncomputable def sin_F (DE EF : ℕ) : ℝ :=
  let DF := Real.sqrt (DE^2 + EF^2)
  DE / DF

theorem find_sin_F_and_verify_triangle_dimensions :
  ∀ (DE EF : ℕ), 
  (∀ (DE = 12) (EF = 5) (∠E = 90°) (is_right_triangle : RightTriangle ∠E), 
  sin_F DE EF = 12 / 13) :=
sorry

end find_sin_F_and_verify_triangle_dimensions_l447_447595


namespace probability_x_y_odd_l447_447820

def isFibonacci (n : ℕ) : Prop :=
  ∃ k1 k2 : ℕ, (Fibonacci k1 = n ∧ k1 ≤ 16 ∧ n ≤ 1000) ∨ (Fibonacci k2 = n ∧ k2 ≤ 16 ∧ n ≤ 1000) 

def isOddSquareInRange (y : ℕ) : Prop :=
  ∃ k : ℕ, y = (2 * k + 1)^2 ∧ y ≤ 1000

def isEven (n : ℕ) : Prop :=
  n % 2 = 0

theorem probability_x_y_odd : 
  let fib_numbers := {n : ℕ | isFibonacci n} in
  let odd_square_numbers := {y : ℕ | isOddSquareInRange y} in
  let even_fib_numbers := {n : ℕ | isFibonacci n ∧ isEven n} in
  let total_fib := finset.card finset.univ (filter isFibonacci finset.range 1001) in
  let even_fib := finset.card finset.univ (filter (λ n, isFibonacci n ∧ isEven n) finset.range 1001) in
  total_fib ≠ 0 → 
  (even_fib.toReal / total_fib.toReal) = 1 / 2 :=
begin
  sorry
end

end probability_x_y_odd_l447_447820


namespace math_problem_l447_447503

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447503


namespace cos_C_in_right_triangle_l447_447185

theorem cos_C_in_right_triangle 
  (A B C : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (is_right_triangle : ∀ (P Q R : A), angle P Q R = 90 → 
    (dist P Q) * (dist P Q) + (dist Q R) * (dist Q R) = (dist P R) * (dist P R))
  (AB BC : ℝ) (AB_eq : AB = 16) (BC_eq : BC = 20) : 
  ∃ (AC : ℝ), AC = 4 * Real.sqrt 41 ∧ cos (angle B C AC) = (5 * Real.sqrt 41) / 41 :=
by 
  sorry

end cos_C_in_right_triangle_l447_447185


namespace binom_11_9_l447_447811

theorem binom_11_9 : nat.choose 11 9 = 55 := by
  sorry

end binom_11_9_l447_447811


namespace num_parallel_diagonals_in_decagon_l447_447605

theorem num_parallel_diagonals_in_decagon :
  let n := 10
  let skip2_sets := 5
  let skip2_parallel_diags := skip2_sets * Nat.choose skip2_sets 2
  let skip3_sets := 5
  let skip3_parallel_diags := skip3_sets * Nat.choose skip3_sets 2
  let skip4_sets := 5
  let skip4_parallel_diags := skip4_sets * Nat.choose skip4_sets 2
  (skip2_parallel_diags + skip3_parallel_diags + skip4_parallel_diags) = 150 :=
by
  sorry

end num_parallel_diagonals_in_decagon_l447_447605


namespace sin_half_angle_of_second_quadrant_l447_447118

theorem sin_half_angle_of_second_quadrant (θ : ℝ) :
  (π/2 < θ ∧ θ < π) →
  (25 * sin θ ^ 2 + sin θ - 24 = 0) →
  (sin (θ / 2) = 4 / 5 ∨ sin (θ / 2) = -4 / 5) := by
  sorry

end sin_half_angle_of_second_quadrant_l447_447118


namespace probability_P_plus_S_mod_six_correct_l447_447740

noncomputable def probability_P_plus_S_mod_six (a b : ℕ) : ℚ :=
  if h : a ≠ b ∧ 1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 then
    let pairs := (finset.range 60).card.choose 2 in
    let valid_pairs := 
      (finset.range 60).filter (λ a, (finset.range 60).filter (λ b, a ≠ b ∧ 
       (a * b + a + b + 1) % 6 = 0)).card in
    (valid_pairs.to_nat / pairs.to_nat : ℚ)
  else 0

theorem probability_P_plus_S_mod_six_correct :
  probability_P_plus_S_mod_six = 109 / 354 :=
sorry

end probability_P_plus_S_mod_six_correct_l447_447740


namespace problem_l447_447396

noncomputable def v : ℝ → ℝ := -- the definition of v is assumed from the graph
  sorry

variables (f : ℝ → ℝ) (hx : ∀ x, v (-x) = -v x)

theorem problem :
  v (-3) + v (-1.5) + v (1.5) + v (3) = 0 :=
by 
  have h1 := hx 3
  have h2 := hx 1.5
  rw [← h1, ← h2]
  linarith

end problem_l447_447396


namespace derivative_f_area_of_triangle_l447_447131

def f (x : ℝ) : ℝ := sin (2 * x) + (sin x)^2

theorem derivative_f :
  ∀ x : ℝ, deriv f x = 2 * sin x * cos x + 2 * cos (2 * x) :=
by sorry

theorem area_of_triangle :
  let x0 := π / 4,
      f_x0 := f x0,
      f'_x0 := deriv f x0,
      tangent_line (x : ℝ) : ℝ := f_x0 + f'_x0 * (x - x0),
      y_intercept := tangent_line 0,
      x_intercept := x0 - f_x0 / f'_x0,
      area := (1 / 2) * (y_intercept * x_intercept)
  in area = (1 / 2) * (3 / 2 - π / 4) ^ 2 :=
by sorry

end derivative_f_area_of_triangle_l447_447131


namespace circle_diameter_mn_origin_l447_447911

-- Definitions based on conditions in (a)
def circle_equation (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line_equation (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def orthogonal (x1 x2 y1 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem to prove (based on conditions and correct answer in (b))
theorem circle_diameter_mn_origin 
  (m : ℝ) 
  (x1 y1 x2 y2 : ℝ)
  (h1: circle_equation m x1 y1) 
  (h2: circle_equation m x2 y2)
  (h3: line_equation x1 y1)
  (h4: line_equation x2 y2)
  (h5: orthogonal x1 x2 y1 y2) :
  m = 8 / 5 := 
sorry

end circle_diameter_mn_origin_l447_447911


namespace temperature_difference_l447_447122

def h : ℤ := 10
def l : ℤ := -5
def d : ℤ := 15

theorem temperature_difference : h - l = d :=
by
  rw [h, l, d]
  sorry

end temperature_difference_l447_447122


namespace coefficient_x4_in_expansion_l447_447053

theorem coefficient_x4_in_expansion :
  ∀ (x : ℂ), polynomial.coeff ((polynomial.C (1 : ℂ) * x - polynomial.C (1 : ℂ) * x⁻¹) ^ 10) 4 = -120 :=
by
  intro x
  sorry

end coefficient_x4_in_expansion_l447_447053


namespace boat_speed_in_still_water_l447_447771

theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (effective_speed : ℝ)
  (boat_speed : ℝ) :
  stream_speed = 5 ∧ distance = 108 ∧ time = 4 ∧ effective_speed = boat_speed + 5 →
  (distance / time = effective_speed) →
  boat_speed = 22 :=
by
  intros hconds heq
  cases hconds with hsrest hdistime
  cases hdistime with hdist htime
  cases htime with heff hboatstream
  sorry

end boat_speed_in_still_water_l447_447771


namespace trigonometric_simplification_l447_447899

noncomputable def trigonometric_expression_value (α : ℝ) (h : sin(3 * π - α) = 2 * sin(π / 2 + α)) : ℝ :=
  (sin(π - α) ^ 3 - sin(π / 2 - α)) / (3 * cos(π / 2 + α) + 2 * cos(π + α))

theorem trigonometric_simplification (α : ℝ) (h : sin(3 * π - α) = 2 * sin(π / 2 + α)) :
  trigonometric_expression_value α h = (8 - 5 * real.sqrt(5)) / (40 * real.sqrt(5)) :=
sorry

end trigonometric_simplification_l447_447899


namespace max_ab_value_l447_447642

-- Define a and b as positive real numbers
variables (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Define the given condition 3a + 8b = 72
axiom h_condition : 3 * a + 8 * b = 72

-- Problem: Prove that the maximum value of ab is 54 under the given conditions
theorem max_ab_value : ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ (3 * a + 8 * b = 72 ∧ a * b = 54) :=
begin
  sorry
end

end max_ab_value_l447_447642


namespace b_100_l447_447885

-- Define the sequence and the partial sum
def b : ℕ → ℚ
| 1 := 3
| (n + 1) := (λ T_n : ℚ, 3 * (T_n ^ 2) / (3 * T_n - 2)) (b (n + 1) + T n)

def T : ℕ → ℚ
| 1 := b 1
| (n + 1) := T n + b (n + 1)

-- The main statement to prove
theorem b_100 :
  b 100 = -4 / 89103 :=
sorry

end b_100_l447_447885


namespace distance_and_isosceles_check_l447_447846

/-- The distance formula between two points in 2D space, given their coordinates. -/
def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Points definitions -/
def P1 := (3 : ℝ, 5 : ℝ)
def P2 := (0 : ℝ, 1 : ℝ)
def P3 := (3 : ℝ, 1 : ℝ)

/-- Distances between points -/
noncomputable def d12 := dist P1.1 P1.2 P2.1 P2.2
noncomputable def d13 := dist P1.1 P1.2 P3.1 P3.2
noncomputable def d23 := dist P2.1 P2.2 P3.1 P3.2

/-- Main Theorem: Validate distance calculations and isosceles triangle condition -/
theorem distance_and_isosceles_check :
  d12 = 5 ∧ (d12 ≠ d13 ∧ d12 ≠ d23 ∧ d13 ≠ d23) :=
by
  -- Proof would go here
  sorry

end distance_and_isosceles_check_l447_447846


namespace total_weight_of_fish_l447_447386

-- Define the weights of fish caught by Peter, Ali, and Joey.
variables (P A J : ℕ)

-- Ali caught twice as much fish as Peter.
def condition1 := A = 2 * P

-- Joey caught 1 kg more fish than Peter.
def condition2 := J = P + 1

-- Ali caught 12 kg of fish.
def condition3 := A = 12

-- Prove the total weight of the fish caught by all three is 25 kg.
theorem total_weight_of_fish :
  condition1 P A → condition2 P J → condition3 A → P + A + J = 25 :=
by
  intros h1 h2 h3
  sorry

end total_weight_of_fish_l447_447386


namespace john_paint_problem_l447_447204

theorem john_paint_problem
  (initial_rooms : ℕ)
  (dropped_cans : ℕ)
  (remaining_rooms : ℕ)
  (room_diff : ℕ)
  (cans_needed : ℕ) :
  initial_rooms = 50 →
  dropped_cans = 5 →
  remaining_rooms = 40 →
  room_diff = initial_rooms - remaining_rooms →
  cans_needed = room_diff / dropped_cans →
  remaining_rooms / (room_diff / dropped_cans) = 20 :=
by
  intros h_init h_drop h_rem h_diff h_cans
  rw [h_init, h_drop, h_rem] at h_diff
  rw [h_diff, h_cans]
  sorry

end john_paint_problem_l447_447204


namespace bake_sale_earnings_eq_400_l447_447021

/-
  The problem statement derived from the given bake sale problem.
  We are to verify that the bake sale earned 400 dollars.
-/

def total_donation (bake_sale_earnings : ℕ) :=
  ((bake_sale_earnings - 100) / 2) + 10

theorem bake_sale_earnings_eq_400 (X : ℕ) (h : total_donation X = 160) : X = 400 :=
by
  sorry

end bake_sale_earnings_eq_400_l447_447021


namespace calculate_milk_and_oil_l447_447311

theorem calculate_milk_and_oil (q_f div_f milk_p oil_p : ℕ) (portions q_m q_o : ℕ) :
  q_f = 1050 ∧ div_f = 350 ∧ milk_p = 70 ∧ oil_p = 30 ∧
  portions = q_f / div_f ∧
  q_m = portions * milk_p ∧
  q_o = portions * oil_p →
  q_m = 210 ∧ q_o = 90 := by
  sorry

end calculate_milk_and_oil_l447_447311


namespace part1_AB_length_area_part2_AB_eqn_when_angle_ABC_90_max_AC_l447_447906

theorem part1_AB_length_area :
  (∀ (x y : ℝ), (x^2 + 3*y^2 = 4) → y = x → ∃ (A B C : ℝ × ℝ), 
    A = (1, 1) ∧ B = (-1, -1) ∧ C = (1, -1) ∧ 
    (let AB := (2 * real.sqrt 2) in 
     let area := 2 in 
     true)) := sorry

theorem part2_AB_eqn_when_angle_ABC_90_max_AC :
  (∃ (m : ℝ), (-1 ≤ m) ∧ (let AC_max_length := 11 - (m + 1) ^ 2 in
   true) → (∃ (A B : ℝ × ℝ), 
    A = (2, -1) ∧ B = (2, -3) ∧ 
    (let line_AB := y = x - 1 in 
    true))) := sorry

end part1_AB_length_area_part2_AB_eqn_when_angle_ABC_90_max_AC_l447_447906


namespace find_a2017_l447_447923

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧
  (∀ n : ℕ, n > 0 → (a (n + 1) - a n ≤ 3 ^ n) ∧ (a (n + 2) - a n ≥ 4 * 3 ^ n))

theorem find_a2017 (a : ℕ → ℤ) (h : sequence a) : a 2017 = ((3 ^ 2017) - 1) / 2 :=
by
  sorry

end find_a2017_l447_447923


namespace number_of_trapezoids_l447_447192

def reg_pent_midpoints := set (fin 5 → ℝ × ℝ)  -- Representation of the midpoints of a regular pentagon

theorem number_of_trapezoids (P : reg_pent_midpoints) : ∃ t : set (set (ℝ × ℝ)), t.card = 15 ∧ (∀ x ∈ t, is_trapezoid x) := 
sorry

end number_of_trapezoids_l447_447192


namespace curve_y_all_real_l447_447700

theorem curve_y_all_real (y : ℝ) : ∃ (x : ℝ), 2 * x * |x| + y^2 = 1 :=
sorry

end curve_y_all_real_l447_447700


namespace distance_A_F_l447_447114

def parametric_curve (θ : ℝ) : ℝ × ℝ := 
  (2 * (2:ℝ).sqrt * Real.cos θ, 1 + Real.cos (2 * θ))

def cartesian_eq (x y : ℝ) : Prop :=
  x^2 = 4 * y

def focus_of_parabola : ℝ × ℝ :=
  (0, 1)

def point_A : ℝ × ℝ :=
  (1, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_A_F : distance point_A focus_of_parabola = Real.sqrt 2 :=
by
  sorry

end distance_A_F_l447_447114


namespace problem_statement_l447_447515

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447515


namespace inequality_proof_l447_447223

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
by
  sorry

end inequality_proof_l447_447223


namespace quadratic_function_a_equals_one_l447_447457

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_function_a_equals_one
  (a b c : ℝ)
  (h1 : 1 < x)
  (h2 : x < c)
  (h_neg : ∀ x, 1 < x → x < c → quadratic_function a b c x < 0):
  a = 1 := by
  sorry

end quadratic_function_a_equals_one_l447_447457


namespace proficiency_test_pass_count_l447_447792

theorem proficiency_test_pass_count (total_students : ℕ) (pass_threshold : ℕ) (pass_count : ℕ) 
  (h1 : total_students = 1000) (h2 : pass_threshold = 70) (h3 : pass_count = 600) :
  ∃ (s : ℕ → ℕ), (∀ x, x ≥ pass_threshold → s x ≥ 0) ∧ (∑ x in (finset.range total_students), s x) = total_students ∧ 
  (∑ x in (finset.range total_students).filter (λ x, x ≥ pass_threshold), s x) = pass_count :=
by
  sorry

end proficiency_test_pass_count_l447_447792


namespace proof_equivalents_l447_447140

open Real

noncomputable def standard_form_curve_C : Prop :=
∀ (α : ℝ), sqrt 3 * cos α = sqrt (3 * (1 - sin α ^ 2))

noncomputable def length_AB (θ : ℝ) : ℝ :=
if θ = π / 4 then 3 / 2 * sqrt 2 else 0

noncomputable def range_PA_PB (θ : ℝ) : set ℝ :=
{ p | p = 2 / (1 + 2 * sin θ ^ 2) }

theorem proof_equivalents :
  (standard_form_curve_C ∧ (∀ θ : ℝ, length_AB θ) ∧ (∀ θ : ℝ, 2 / (1 + 2 * sin θ ^ 2) ∈ Icc (2/3) 2)) :=
by
  split
  sorry
  split
  sorry
  intro θ
  sorry

end proof_equivalents_l447_447140


namespace sum_base10_to_base4_l447_447398

theorem sum_base10_to_base4 : 
  (31 + 22 : ℕ) = 3 * 4^2 + 1 * 4^1 + 1 * 4^0 :=
by
  sorry

end sum_base10_to_base4_l447_447398


namespace conjugate_quadrant_l447_447947

theorem conjugate_quadrant (z : ℂ) (h : (1 + 2 * Complex.i) * z = 1 + Complex.i) : (Complex.re (Complex.conj z) > 0) ∧ (Complex.im (Complex.conj z) > 0) :=
by
  sorry

end conjugate_quadrant_l447_447947


namespace series_convergence_l447_447977

-- Define the series sum and the substitution q(x)
def series (x : ℝ) : ℝ := ∑' n : ℕ, (1 / (n^2 + 3)) * ((x + 1) / (x - 1))^n

-- State the main theorem: the convergence of the series
theorem series_convergence (x : ℝ) : 
  (series x).converges ↔ x ≤ 0 := 
sorry

end series_convergence_l447_447977


namespace sum_of_products_l447_447041

theorem sum_of_products : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end sum_of_products_l447_447041


namespace range_of_m_for_common_point_l447_447161

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ :=
  -x^2 - 2 * x + m

-- Define the condition for a common point with the x-axis (i.e., it has real roots)
def has_common_point_with_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_function x m = 0

-- The theorem statement
theorem range_of_m_for_common_point : ∀ m : ℝ, has_common_point_with_x_axis m ↔ m ≥ -1 := 
sorry

end range_of_m_for_common_point_l447_447161


namespace initial_investment_l447_447391

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment (A : ℝ) (r : ℝ) (n t : ℕ) (P : ℝ) :
  A = 3630.0000000000005 → r = 0.10 → n = 1 → t = 2 → P = 3000 →
  A = compound_interest P r n t :=
by
  intros hA hr hn ht hP
  rw [compound_interest, hA, hr, hP]
  sorry

end initial_investment_l447_447391


namespace negation_statement_l447_447684

variables {a b c : ℝ}

theorem negation_statement (h : a * b * c = 0) : ¬(a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end negation_statement_l447_447684


namespace num_primes_between_50_and_70_l447_447151

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71).filter (λ x, x ≥ 50)).card = 4 := by
  sorry

end num_primes_between_50_and_70_l447_447151


namespace meet_time_proof_l447_447757

noncomputable def meet_time {a_speed b_speed c_speed : ℝ} (track_length : ℝ) : ℝ :=
  Nat.lcm (Nat.ceil (track_length / a_speed)) (Nat.lcm (Nat.ceil (track_length / b_speed)) (Nat.ceil (track_length / c_speed)))

theorem meet_time_proof :
  let a_speed := 4000 / 60 -- Speed of A in m/min
  let b_speed := 6000 / 60 -- Speed of B in m/min
  let c_speed := 8000 / 60 -- Speed of C in m/min
  let track_length := 400
  meet_time track_length = 12 :=
by
  sorry

end meet_time_proof_l447_447757


namespace triangle_problems_l447_447974

noncomputable def triangle_area (a b : ℝ) (cosC : ℝ) : ℝ :=
  let sinC := Real.sqrt (1 - cosC^2) -- calculate sinC
  (1 / 2) * a * b * sinC

noncomputable def cosine_of_largest_angle (a b c : ℝ) : ℝ :=
  Real.max (Real.max (a^2 + b^2 - c^2) (a^2 + c^2 - b^2)) (b^2 + c^2 - a^2) /
           (2 * a * b)

theorem triangle_problems :
  ∀ (a b : ℝ) (cosC : ℝ),
  a = 5 → b = 8 → cosC = (1 / 2) →
    triangle_area a b cosC = 10 * Real.sqrt 3 ∧
    (cosine_of_largest_angle a b 7 = (1 / 7)) :=
by
  intros a b cosC h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end triangle_problems_l447_447974


namespace find_tim_books_l447_447736

theorem find_tim_books (Mike_books Total_books Tim_books : ℕ) :
  Mike_books = 20 → Total_books = 42 → Tim_books = Total_books - Mike_books → Tim_books = 22 :=
by
  intros hMike hTotal hTim
  rw [hMike, hTotal] at hTim
  calc
    Tim_books = 42 - 20 : by rw [hTim]
           ... = 22 : by norm_num

end find_tim_books_l447_447736


namespace problem_statement_l447_447568

-- Define proposition p
def prop_p : Prop := ∃ x : ℝ, Real.exp x ≥ x + 1

-- Define proposition q
def prop_q : Prop := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- The final statement we want to prove
theorem problem_statement : (prop_p ∧ ¬prop_q) :=
by
  sorry

end problem_statement_l447_447568


namespace moles_of_NaNO3_l447_447086

open Classical

theorem moles_of_NaNO3 (moles_NH4NO3 moles_NaOH : ℕ) 
    (h_eq : moles_NH4NO3 = 3 ∧ moles_NaOH = 3) 
    (balanced_eqn : ∀ n, NH4NO3 n + NaOH n → NaNO3 n + NH3 n + H2O n) 
    : ∃ moles_NaNO3, moles_NaNO3 = 3 :=
by
  use 3
  sorry

end moles_of_NaNO3_l447_447086


namespace pyramid_theorem_l447_447345

-- Given the conditions
variables (a b c d : ℝ)
variables (base_is_rectangle : Prop)
variables (side_edges_equal : Prop)
variables (sections_cut_by_plane : Prop)

-- Define the hypothesis based on given conditions
def pyramid_conditions  := base_is_rectangle ∧ side_edges_equal ∧ sections_cut_by_plane

-- State the theorem to be proved
theorem pyramid_theorem (h : pyramid_conditions):
  (1 / a + 1 / c) = (1 / b + 1 / d) :=
sorry

end pyramid_theorem_l447_447345


namespace haley_end_of_month_balance_l447_447931

noncomputable def initial_amount : ℝ := 2
noncomputable def received_chores : ℝ := 5.25
noncomputable def received_birthday : ℝ := 10
noncomputable def received_neighbor : ℝ := 7.5
noncomputable def found : ℝ := 0.50
noncomputable def received_aunt : ℝ := 3 * 1.3
noncomputable def spent_candy : ℝ := 3.75
noncomputable def lost : ℝ := 1.5

theorem haley_end_of_month_balance :
  let total_received := initial_amount + received_chores + received_birthday + received_neighbor + found + received_aunt in
  let total_spent_lost := spent_candy + lost in
  let final_amount := total_received - total_spent_lost in
  final_amount - initial_amount = 19.90 :=
by {
  let total_received := initial_amount + received_chores + received_birthday + received_neighbor + found + received_aunt,
  let total_spent_lost := spent_candy + lost,
  let final_amount := total_received - total_spent_lost,
  show final_amount - initial_amount = 19.90,
  sorry
}

end haley_end_of_month_balance_l447_447931


namespace minimizing_reciprocal_sum_l447_447606

theorem minimizing_reciprocal_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 30) :
  a = 10 ∧ b = 5 :=
by
  sorry

end minimizing_reciprocal_sum_l447_447606


namespace candidate_final_score_l447_447775

/- Given conditions -/
def interview_score : ℤ := 80
def written_test_score : ℤ := 90
def interview_weight : ℤ := 3
def written_test_weight : ℤ := 2

/- Final score computation -/
noncomputable def final_score : ℤ :=
  (interview_score * interview_weight + written_test_score * written_test_weight) / (interview_weight + written_test_weight)

theorem candidate_final_score : final_score = 84 := 
by
  sorry

end candidate_final_score_l447_447775


namespace proof_problem_l447_447860

def largest_odd_proper_divisor (n: ℕ) : ℕ :=
  if n % 2 = 1 then
    -- If n is odd, find the largest divisor less than n
    let divisors := (List.range (n // 2 + 1)).filter (λ d, d > 0 ∧ n % d = 0) in
    divisors.filter (λ d, d % 2 = 1) |>.maximum'
  else
    -- If n is even, strip factors of 2 and apply the function
    largest_odd_proper_divisor (n / 2)

noncomputable def problem_solution (N: ℕ) : ℕ :=
  (largest_odd_proper_divisor N) / (largest_odd_proper_divisor (largest_odd_proper_divisor (largest_odd_proper_divisor N)))

theorem proof_problem : problem_solution (20^23 * 23^20) = 25 :=
  by sorry

end proof_problem_l447_447860


namespace appeared_candidates_l447_447175

noncomputable def number_of_candidates_that_appeared_from_each_state (X : ℝ) : Prop :=
  (8 / 100) * X + 220 = (12 / 100) * X

theorem appeared_candidates (X : ℝ) (h : number_of_candidates_that_appeared_from_each_state X) : X = 5500 :=
  sorry

end appeared_candidates_l447_447175


namespace chocolates_in_large_box_l447_447768

noncomputable def large_box_chocolates : ℕ :=
  let chocolates := [20, 15, 18, 17, 20]
  chocolates.filter (λ x, x ≥ 18).sum

theorem chocolates_in_large_box :
  large_box_chocolates = 58 :=
by
  sorry

end chocolates_in_large_box_l447_447768


namespace bigger_wheel_roll_distance_l447_447966

/-- The circumference of the bigger wheel is 12 meters -/
def bigger_wheel_circumference : ℕ := 12

/-- The circumference of the smaller wheel is 8 meters -/
def smaller_wheel_circumference : ℕ := 8

/-- The distance the bigger wheel must roll for the points P1 and P2 to coincide again -/
theorem bigger_wheel_roll_distance : Nat.lcm bigger_wheel_circumference smaller_wheel_circumference = 24 :=
by
  -- Proof is omitted
  sorry

end bigger_wheel_roll_distance_l447_447966


namespace other_x_intercept_l447_447019

noncomputable def point1 : ℝ × ℝ := (1, 2)
noncomputable def point2 : ℝ × ℝ := (4, 0)
noncomputable def x_intercept := ∃ x : ℝ, (x, 0) ∈ set_of (λ P : ℝ × ℝ, real.sqrt ((P.1 - point1.1)^2 + (P.2 - point1.2)^2) + real.sqrt ((P.1 - point2.1)^2 + (P.2 - point2.2)^2) = 6)

theorem other_x_intercept : x_intercept ∧ (19 / 2, 0) ∈ set_of (λ P : ℝ × ℝ, (P.1, P.2) = ((19 / 2), 0)) :=
sorry

end other_x_intercept_l447_447019


namespace collinearity_of_points_l447_447655

theorem collinearity_of_points 
  (P Q O : Point)
  (R : ℝ)
  (ABC_circumcircle : Circumcircle)
  (sym_P_BC : Point)
  (sym_P_CA : Point)
  (sym_P_AB : Point)
  (Q_U_intersects_BC : Point)
  (Q_V_intersects_CA : Point)
  (Q_W_intersects_AB : Point)
  (line_OK : Line)
  (circumradius : ℝ)
  (circumcenter : Point)
  (same_line : Π (P : Point), P ∈ line_OK)
  (OP * OQ = R^2) : 
  collinear Q_U_intersects_BC Q_V_intersects_CA Q_W_intersects_AB := sorry

end collinearity_of_points_l447_447655


namespace trajectory_is_ellipse_exists_point_on_ellipse_with_area_condition_l447_447896

structure Point :=
(x : ℝ)
(y : ℝ)

def M : Point := { x := 4, y := 0 }
def N : Point := { x := 1, y := 0 }

def satisfies_condition (P : Point) : Prop :=
  let MN := (N.x - M.x, N.y - M.y)
  let MP := (P.x - M.x, P.y - M.y)
  let NP := (P.x - N.x, P.y - N.y)
  (MN.1 * MP.1 + MN.2 * MP.2) = 6 * real.sqrt (NP.1 ^ 2 + NP.2 ^ 2)

def ellipse_equation (P : Point) : Prop :=
  (P.x ^ 2) / 4 + (P.y ^ 2) / 3 = 1

theorem trajectory_is_ellipse :
  ∀ P: Point, satisfies_condition P → ellipse_equation P :=
sorry

def S_triangle_MNQ (Q : Point) : ℝ :=
  0.5 * 3 * |Q.y|

theorem exists_point_on_ellipse_with_area_condition :
  ∃ Q : Point, ellipse_equation Q ∧ S_triangle_MNQ Q = 3 / 2 :=
sorry

end trajectory_is_ellipse_exists_point_on_ellipse_with_area_condition_l447_447896


namespace alternating_sum_total_10_l447_447095

/-- Define the set from 1 to 10. -/
def set_10 : Set ℕ := {x | x ≥ 1 ∧ x ≤ 10}

/-- Define the alternating sum for a subset. -/
def alt_sum (s : Finset ℕ) : ℤ :=
  s.sort (· ≥ ·) |>.val.enum.map_with_index (λ i x, if i % 2 == 0 then x else -x) |>.sum

/-- The final claim: the sum of all alternating sums for subsets of {1, 2, 3, ..., 10} equals 5120. -/
theorem alternating_sum_total_10 : 
  (∑ s in (set_10.to_finset.powerset.erase ∅), alt_sum s) = 5120 :=
sorry

end alternating_sum_total_10_l447_447095


namespace min_buses_required_l447_447773

theorem min_buses_required (bus_capacity : ℕ) (total_students : ℕ) (total_drivers : ℕ) (min_buses : ℕ) :
  bus_capacity = 42 →
  total_students = 480 →
  total_drivers = 12 →
  min_buses = 12 →
  min_buses * bus_capacity >= total_students ∧ min_buses <= total_drivers :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  split
  · exact Nat.mul_le_mul_right 42 (by norm_num)
  · exact by norm_num

end min_buses_required_l447_447773


namespace total_amount_l447_447779

-- Conditions as given definitions
def ratio_a : Nat := 2
def ratio_b : Nat := 3
def ratio_c : Nat := 4
def share_b : Nat := 1500

-- The final statement
theorem total_amount (parts_b := 3) (one_part := share_b / parts_b) :
  (2 * one_part) + (3 * one_part) + (4 * one_part) = 4500 :=
by
  sorry

end total_amount_l447_447779


namespace problem_solution_l447_447703

-- Definitions for opposite number, reciprocal, and absolute value equivalence
def opposite (x : Int) : Int := -x

def reciprocal (x : Int) : Float := 1 / x

def absolute_value (x : Int) : Int := if x < 0 then -x else x

theorem problem_solution :
  reciprocal (opposite (-2)) = 1 / 2 ∧ 
  ({ y : Int | absolute_value y = 5 } = { 5, -5 }) := by 
  sorry

end problem_solution_l447_447703


namespace math_problem_l447_447506

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l447_447506


namespace tangent_line_f_at_0_l447_447133

-- Main problem context
variable (f g : ℝ → ℝ)

-- Given conditions
axiom g_tangent : tangent_line_eq g 0 (λ x, 2 * x + 1) 
axiom f_def : ∀ x, f(x) = Real.exp x + g(x)

-- Expected conclusion
theorem tangent_line_f_at_0 : tangent_line_eq f 0 (λ x, 3 * x + 2) :=
by
  sorry

-- Definitions to make the axioms clear
def tangent_line_eq (f : ℝ → ℝ) (a : ℝ) (line_eq : ℝ → ℝ) : Prop :=
  f(a) = line_eq a ∧ (∀ x, f x = f(a) + (deriv f a) * (x - a))


end tangent_line_f_at_0_l447_447133


namespace soccer_ball_white_hexagons_l447_447785

theorem soccer_ball_white_hexagons (black_pentagons : ℕ) (black_pentagon_sides : ℕ) (white_hexagon_sides : ℕ) (connected_sides_per_hexagon : ℕ) (total_connected_sides : ℕ) :
  black_pentagons = 12 →
  black_pentagon_sides = 5 →
  white_hexagon_sides = 6 →
  connected_sides_per_hexagon = 3 →
  total_connected_sides = black_pentagons * black_pentagon_sides →
  ∃ (x : ℕ), x * connected_sides_per_hexagon = total_connected_sides ∧ x = 20 :=
by
  intros h1 h2 h3 h4 h5
  use 20
  simp [h1, h2, h3, h4, h5]
  sorry -- Proof steps not required as per instructions

end soccer_ball_white_hexagons_l447_447785


namespace coefficient_x2_in_expansion_l447_447828

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the problem: Given (2x + 1)^5, find the coefficient of x^2 term
theorem coefficient_x2_in_expansion : 
  binomial 5 3 * (2 ^ 2) = 40 := by 
  sorry

end coefficient_x2_in_expansion_l447_447828


namespace sum_valid_n_l447_447221

theorem sum_valid_n (N : ℕ) (hN : N = 2014) : 
  ∑ n in Finset.filter 
      (λ n, ((x^2 + x + 1 : Polynomial ℚ).is_root ((Polynomial.X : Polynomial ℚ)^(2 * n) + (Polynomial.X : Polynomial ℚ)^n + 1)))
      (Finset.range (N + 1)) 
    id = 1354349 :=
sorry

end sum_valid_n_l447_447221


namespace partI_partII_l447_447500

-- Define the sets as intervals
def setA : Set ℝ := {x | 3 < x ∧ x < 7}
def setB : Set ℝ := {x | 2 < x ∧ x < 10}
def setC (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Complement of set A in ℝ
def C_R_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 7}

-- Calculate (C_R_A) ∩ B
def questionI : Set ℝ := {x | (2 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x < 10)}

-- Prove the results of question I
theorem partI : (C_R_A ∩ setB) = questionI := by
  sorry

-- State the conditions for question II
def range_a (a : ℝ) : Prop := ∀ x, x ∈ setC(a) → x ∈ (setA ∪ setB)

-- Prove the range of values for 'a'
theorem partII : (∀ a, range_a a → a ≤ 3) := by
  sorry

end partI_partII_l447_447500


namespace angle_ABC_equals_70_l447_447649

theorem angle_ABC_equals_70 
  (A B C D : Type) 
  (ABC : Triangle A B C)
  (angle_CAB : ∠ CAB = 20)
  (midpoint_D : Midpoint D A B)
  (angle_CDB : ∠ CDB = 40) :
  ∠ ABC = 70 := 
sorry

end angle_ABC_equals_70_l447_447649


namespace required_additional_amount_l447_447371

noncomputable def ryan_order_total : ℝ := 15.80 + 8.20 + 10.50 + 6.25 + 9.15
def minimum_free_delivery : ℝ := 50
def discount_threshold : ℝ := 30
def discount_rate : ℝ := 0.10

theorem required_additional_amount : 
  ∃ X : ℝ, ryan_order_total + X - discount_rate * (ryan_order_total + X) = minimum_free_delivery :=
sorry

end required_additional_amount_l447_447371


namespace sum_of_series_l447_447317

theorem sum_of_series (h1 : 2 + 4 + 6 + 8 + 10 = 30) (h2 : 1 + 3 + 5 + 7 + 9 = 25) : 
  ((2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9)) + ((1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10)) = 61 / 30 := by
  sorry

end sum_of_series_l447_447317


namespace first_day_of_the_month_is_sunday_l447_447264

-- Definition of the days of the week
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
  deriving DecidableEq, Repr

def daysInWeek : Fin 7 := ⟨7, by decide⟩

-- Given condition
def twenty_first_day_is_saturday (month : ℕ) : Prop := 
  True  -- Dummy implementation

-- Theorem to prove
theorem first_day_of_the_month_is_sunday (month : ℕ) (h : twenty_first_day_is_saturday month) :
  DayOfWeek :=
  sorry

end first_day_of_the_month_is_sunday_l447_447264


namespace expected_value_bound_l447_447110

open Probability Theory

noncomputable section

def trimOnce (l : List ℝ) : List ℝ := l.inits.filter (λ l => l.length = 3).map (λ l => l.sorted (λ a b => a ≤ b) !! 1)

def trimRepeatedly (l : List ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => l.head' |>.get_or_else 0
  | k + 1 => trimRepeatedly (trimOnce l) k

def finalTrimmedValue (l : List ℝ) : ℝ := trimRepeatedly l 2021

def E_X_abs_diff_half : ℝ :=
  ENNReal.toReal (expectation (λ (l : List ℝ) => |finalTrimmedValue l - (1/2)|))

theorem expected_value_bound :
  ∀ (l : List ℝ), l.length = 3^2021 ∧ (∀ x, x ∈ l → x ≥ 0 ∧ x ≤ 1) →
  E_X_abs_diff_half ≥ 1/4 * (2/3)^2021 :=
sorry

end expected_value_bound_l447_447110


namespace Ray_dog_walks_l447_447253

theorem Ray_dog_walks:
  let blocks_to_park := 4,
      blocks_to_high_school := 7,
      blocks_back_home := 11,
      blocks_per_route := blocks_to_park + blocks_to_high_school + blocks_back_home,
      total_blocks_per_day := 66 in
  total_blocks_per_day / blocks_per_route = 3 :=
by
  sorry

end Ray_dog_walks_l447_447253


namespace harmonic_series_induction_additional_terms_differential_l447_447314

theorem harmonic_series_induction (n : ℕ) (h : n ≥ 1) : 
  ∑ i in finset.range (2 * n + 1), (1 / (i + 1) : ℝ) ≥ 5 / 6 := sorry

theorem additional_terms_differential (k : ℕ) : 
  (1 / (3 * k + 1) : ℝ) + (1 / (3 * k + 2) : ℝ) + (1 / (3 * k + 3) : ℝ) - (1 / (k + 1) : ℝ) := sorry

end harmonic_series_induction_additional_terms_differential_l447_447314


namespace hyperbola_vertices_distance_l447_447075

/--
For the hyperbola given by the equation
(x^2 / 121) - (y^2 / 49) = 1,
the distance between its vertices is 22.
-/
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ,
  (x^2 / 121) - (y^2 / 49) = 1 →
  ∃ a : ℝ, a = 11 ∧ 2 * a = 22 :=
by
  intros x y h
  use 11
  split
  · refl
  · norm_num

end hyperbola_vertices_distance_l447_447075


namespace problem_1_problem_2_l447_447184

variables {x y : ℝ}
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vector_opposite (v : ℝ × ℝ) : ℝ × ℝ := (-v.1, -v.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def AB := (6, 1)
def BC := (x, y)
def CD := (-2, -3)
def DA := vector_opposite (vector_add (vector_add AB BC) CD)
def AC := vector_add AB BC
def BD := vector_add BC CD

theorem problem_1 (h_parallel : BC.1 * DA.2 - BC.2 * DA.1 = 0) : x + 2 * y = 0 :=
sorry

theorem problem_2 
  (h_parallel : BC.1 * DA.2 - BC.2 * DA.1 = 0)
  (h_perpendicular : dot_product AC BD = 0) 
  : (x = -6 ∧ y = 3 ∧ ((6:ℝ) * 3 - 6 * (-3) / 2 = 16)) ∨ 
    (x = 2 ∧ y = -1 ∧ ((6:ℝ) * (-1) - (-2) * (-3) / 2 = 16)) :=
sorry

end problem_1_problem_2_l447_447184


namespace non_right_triangle_option_l447_447613

-- Definitions based on conditions
def optionA (A B C : ℝ) : Prop := A + B = C
def optionB (A B C : ℝ) : Prop := A - B = C
def optionC (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def optionD (A B C : ℝ) : Prop := A = B ∧ A = 3 * C

-- Given conditions for a right triangle
def is_right_triangle (A B C : ℝ) : Prop := ∃(θ : ℝ), θ = 90 ∧ (A = θ ∨ B = θ ∨ C = θ)

-- The proof problem
theorem non_right_triangle_option (A B C : ℝ) :
  optionD A B C ∧ ¬(is_right_triangle A B C) := sorry

end non_right_triangle_option_l447_447613


namespace fuel_oil_used_l447_447238

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end fuel_oil_used_l447_447238


namespace column_product_independence_l447_447745

   noncomputable def table_entry (a b : ℕ → ℝ) (i j : ℕ) : ℝ :=
     a i + b j

   noncomputable def row_product (a b : ℕ → ℝ) (i n : ℕ) : ℝ :=
     ∏ j in finset.range n, table_entry a b i j

   theorem column_product_independence (a b : ℕ → ℝ) (n : ℕ)
       (h_a_distinct : function.injective a)
       (h_b_distinct : function.injective b)
       (h_row_product_independent : ∀ i, row_product a b i n = row_product a b 0 n) :
       ∀ j, ∏ i in finset.range n, table_entry a b i j = ∏ i in finset.range n, table_entry a b i 0 :=
   by 
     sorry
   
end column_product_independence_l447_447745


namespace person_c_is_lying_l447_447308

def valid_score (attempts : ℕ) (score : ℕ) : Prop :=
  ∃ (x : list ℕ), x.length = attempts ∧ 
  (∀ n, n ∈ x → n ∈ {1, 3, 5, 7, 9}) ∧
  list.sum x = score

theorem person_c_is_lying :
  ¬ valid_score 3 24 :=
by
  sorry

end person_c_is_lying_l447_447308


namespace expected_value_of_shorter_gentlemen_correct_l447_447033
noncomputable def expected_value_of_shorter_gentlemen (n : ℕ) : ℝ :=
  ∑ j in Finset.range n, (j : ℝ) / n

theorem expected_value_of_shorter_gentlemen_correct (n : ℕ) :
  expected_value_of_shorter_gentlemen (n + 1) = n / 2 :=
by
  sorry

end expected_value_of_shorter_gentlemen_correct_l447_447033


namespace domain_of_f_l447_447691

-- Definitions of the conditions and the domain
def condition1 (x : ℝ) : Prop := 1 - x ≥ 0
def condition2 (x : ℝ) : Prop := x > 0
def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

-- The proof goal
theorem domain_of_f (x : ℝ) : condition1 x → condition2 x → domain x :=
by
  intro h1 h2
  unfold condition1 at h1
  unfold condition2 at h2
  unfold domain
  apply And.intro
  exact h2
  exact h1

end domain_of_f_l447_447691


namespace cat_mouse_position_258_l447_447592

-- Define the cycle positions for the cat
def cat_position (n : ℕ) : String :=
  match n % 4 with
  | 0 => "top left"
  | 1 => "top right"
  | 2 => "bottom right"
  | _ => "bottom left"

-- Define the cycle positions for the mouse
def mouse_position (n : ℕ) : String :=
  match n % 8 with
  | 0 => "top middle"
  | 1 => "top right"
  | 2 => "right middle"
  | 3 => "bottom right"
  | 4 => "bottom middle"
  | 5 => "bottom left"
  | 6 => "left middle"
  | _ => "top left"

theorem cat_mouse_position_258 : 
  cat_position 258 = "top right" ∧ mouse_position 258 = "top right" := by
  sorry

end cat_mouse_position_258_l447_447592


namespace smallest_integer_in_set_l447_447599

theorem smallest_integer_in_set (n : ℤ) 
  (h : n + 6 < 3 * (n + (1 + 2 + 3 + 4 + 5 + 6) / 7)) : 
  n ≥ 0 :=
by
  simp [Real.ofInt_eq_coe_int, Int.cast_div, Int.cast_add, Int.cast_bit0, Int.cast_one, 
        Int.cast_succ, Int.cast_mul, Int.cast_add, Int.cast_neg_succ_of_nat, 
        Int.cast_coe_nat, Int.cast_add, Int.cast_bit1, Int.cast_of_nat, Int.cast_neg, 
        Int.cast_id, Int.cast_div] at h
  linarith

end smallest_integer_in_set_l447_447599


namespace trigonometric_identity_solution_exists_l447_447756

theorem trigonometric_identity_solution_exists
  (t : ℝ) (k : ℤ)
  (h : cos (2 * t - 18) * tan 50 + sin (2 * t - 18) = 1 / (2 * cos 130)) :
  (t = -31 + 180 * k) ∨ (t = 89 + 180 * k) :=
by
  sorry

end trigonometric_identity_solution_exists_l447_447756


namespace distance_between_vertices_hyperbola_l447_447067

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l447_447067


namespace prime_div_sum_of_squares_l447_447653

theorem prime_div_sum_of_squares 
  (p : ℕ) (h_prime : Nat.Prime p) (h_form : p % 4 = 3) (a b : ℤ) 
  (h_div : p ∣ (a^2 + b^2)): p ∣ a ∧ p ∣ b := 
  sorry

end prime_div_sum_of_squares_l447_447653


namespace sum_b_prove_l447_447198

noncomputable def q (n : ℕ) (hn : n ≥ 3) : ℝ := real.exp ((real.log 16) / (n - 1))

noncomputable def b (n : ℕ) (hn : n ≥ 3) : ℝ := (q n hn) ^ (n * (n - 1) / 2)

theorem sum_b_prove (n : ℕ) (hn : n ≥ 3) :
    ∑ k in finset.range (n - 2 + 1) \ {0, 1}, b (k + 2) hn = (4^(n+1) - 64) / 3 :=
begin
    sorry,
end

end sum_b_prove_l447_447198


namespace binomial_problems_l447_447673

-- (a)
def binomial_neg_eq (n k : ℕ) : Prop :=
  (if k ≤ n then (Nat.choose (-n) k) else 0) = (-1) ^ k * (Nat.choose (n + k - 1) k)

-- (b)
def binomial_series_neg (n : ℕ) (x : ℝ) : Prop :=
  (1 + x) ^ (-n : ℤ) = (List.range (n + 1)).sum (λ k, nat.choose (-n) k * x ^ k)

-- Statement for the problem
theorem binomial_problems (n : ℕ) (x : ℝ) (h : ∀ k, binomial_neg_eq n k) : binomial_series_neg n x :=
  sorry

end binomial_problems_l447_447673


namespace unique_n_log_eq_l447_447092

theorem unique_n_log_eq :
  ∃! (n : ℕ), 0 < n ∧ real.logb 2 (real.logb 32 n) = real.logb 5 (real.logb 5 n) ∧ n = 316 :=
sorry

end unique_n_log_eq_l447_447092


namespace line_tangent_circle_l447_447878

theorem line_tangent_circle (a : ℝ) :
  let line := λ (x y : ℝ), y = x + a in
  let circle := λ (x y : ℝ), x ^ 2 + y ^ 2 = 2 in
  (∀ x y, line x y ↔ y = x + a) ∧ (∀ x y, circle x y ↔ x^2 + y^2 = 2) →
  (∃ (dist : ℝ), dist = real.norm a / real.sqrt 2 ∧ dist = real.sqrt 2) →
  a = 2 ∨ a = -2 :=
by 
  sorry

end line_tangent_circle_l447_447878


namespace original_number_of_sides_l447_447165

theorem original_number_of_sides (sum_of_angles : ℕ) (H : (sum_of_angles = 2160)) : 
  ∃ x : ℕ, (2 * x - 2) * 180 = 2160 := 
by
  use 7
  have : (2 * 7 - 2) * 180 = 2160 := by sorry
  exact this

end original_number_of_sides_l447_447165


namespace geometric_sequence_derivative_l447_447608

noncomputable def f (x : ℝ) (a : Fin 8 → ℝ) : ℝ :=
  x * (Finset.prod (Finset.range 8) (λ n, x - a ⟨n, by linarith⟩))

theorem geometric_sequence_derivative :
  ∀ (a : Fin 8 → ℝ), a 0 = 2 → a 7 = 4 → 
  f 0 a = 2^12 :=
by
  intros a h1 h2
  sorry

end geometric_sequence_derivative_l447_447608


namespace distance_between_vertices_hyperbola_l447_447066

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l447_447066


namespace sum_divisible_by_prime_l447_447461

theorem sum_divisible_by_prime {p a b c d : ℕ} (hp: p.prime) (hp_odd : ¬even p)
  (ha_not_mult : ¬p ∣ a) (hb_not_mult : ¬p ∣ b) (hc_not_mult : ¬p ∣ c) (hd_not_mult : ¬p ∣ d)
  (h_sum_fractional : ∀ (n : ℕ), (¬p ∣ n) →
    (fract (n * a / p) + fract (n * b / p) + fract (n * c / p) + fract (n * d / p) = 2)) :
  ∃ (i j : ℕ), i ≠ j ∧ (i = a ∧ j = b ∨ i = a ∧ j = c ∨ i = a ∧ j = d ∨ i = b ∧ j = c ∨ i = b ∧ j = d ∨ i = c ∧ j = d) ∧
    (i + j) % p = 0 := sorry

end sum_divisible_by_prime_l447_447461


namespace correct_statements_about_series_l447_447819

def is_geometric_series (a r : ℝ) (s : ℕ → ℝ) := ∀ n, s n = a * r ^ n

noncomputable def series_sum (s : ℕ → ℝ) := ∑' n, s n

noncomputable def series := λ n : ℕ, 3 * (1 / 4) ^ n

theorem correct_statements_about_series : 
  (series_sum series = 4) ∧ 
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |series n - 0| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |series_sum series - 4| < ε) :=
by
  sorry

end correct_statements_about_series_l447_447819


namespace kiwoo_school_supplies_l447_447206

def use_for_school_supplies (W : ℝ) := W / 2
def remaining_after_school_supplies (W : ℝ) := W / 2
def saving_fraction (W : ℝ) := 3 / 16 * W
def remaining_after_saving (W : ℝ) := W / 2 - 3 / 16 * W

theorem kiwoo_school_supplies (W : ℝ) (h : remaining_after_saving W = 2500) :
  use_for_school_supplies W = 4000 :=
by
  sorry

end kiwoo_school_supplies_l447_447206


namespace area_of_trapezium_l447_447845

-- Definitions for the given conditions
def parallel_side_a : ℝ := 18  -- in cm
def parallel_side_b : ℝ := 20  -- in cm
def distance_between_sides : ℝ := 5  -- in cm

-- Statement to prove the area is 95 cm²
theorem area_of_trapezium : 
  let a := parallel_side_a
  let b := parallel_side_b
  let h := distance_between_sides
  (1 / 2 * (a + b) * h = 95) :=
by
  sorry  -- Proof is not required here

end area_of_trapezium_l447_447845


namespace num_subsets_of_A_l447_447586

theorem num_subsets_of_A (U : set ℕ) (A : set ℕ) (hU : U = {1, 2, 3, 4}) (hAU : U \ A = {2}) :
  (2 ^ A.card) = 8 :=
  sorry

end num_subsets_of_A_l447_447586


namespace periodic_sequence_result_l447_447883

def sequence (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, 2 ≤ n → a (n+1) = a n - a (n-1)

def initial_conditions (a1 b1 : ℕ → ℤ) : Prop :=
  a1 1 = a ∧ a1 2 = b

def sum_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 0 then 0 else a 1 + a 2 + sum_seq 2 n
    where
      sum_seq (i k : ℕ) : ℤ :=
        if i = k then 0
        else a (i+1) + sum_seq (i+1) k

theorem periodic_sequence_result (a b : ℕ → ℤ)
  (h_seq : sequence a)
  (h_initial : initial_conditions a) :
  a 100 = -a 1 ∧ sum_sequence a 100 = 2 * a 2 - a 1 :=
by {
  sorry
}

end periodic_sequence_result_l447_447883


namespace sin_cos_alpha_eq_fifth_l447_447443

variable {α : ℝ}
variable (h : Real.sin α = 2 * Real.cos α)

theorem sin_cos_alpha_eq_fifth : Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end sin_cos_alpha_eq_fifth_l447_447443


namespace polynomial_divisibility_l447_447455

noncomputable def exists_divisor_of_polynomial (F : ℤ[X]) (a : ℕ → ℤ) (m : ℕ) : Prop :=
  (∀ n : ℤ, ∃ k : ℕ, k < m ∧ a k ∣ F.eval n) → (∃ j : ℕ, j < m ∧ ∀ n : ℤ, a j ∣ F.eval n)

theorem polynomial_divisibility (F : ℤ[X]) (a : ℕ → ℤ) (m : ℕ) :
  exists_divisor_of_polynomial F a m :=
sorry

end polynomial_divisibility_l447_447455


namespace perimeter_of_PQRST_l447_447609

-- Define the coordinates of points P, Q, R, S, T
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 5⟩
def Q : Point := ⟨2, 5⟩
def R : Point := ⟨2, 2⟩
def S : Point := ⟨5, 0⟩
def T : Point := ⟨0, 0⟩

-- Calculate the distance between two points
def dist (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

-- Specific distances given in the problem
def PQ := dist P Q -- should be 2
def QR := dist Q R -- should be 3
def ST := dist S T -- should be 5
def PT := dist P T -- should be 5
def PS := dist P S -- should be calculated as part of the problem

-- Perimeter of polygon PQRST
def perimeter_PQRST : ℝ :=
  PQ + QR + dist R S + ST + PT

theorem perimeter_of_PQRST :
  perimeter_PQRST = 15 + 3 * Real.sqrt 2 :=
by
  -- Lean proof goes here
  sorry

end perimeter_of_PQRST_l447_447609


namespace correct_exp_identity_l447_447331

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end correct_exp_identity_l447_447331


namespace volume_of_intersection_of_cubes_l447_447273

-- Define the properties of the cubes and their configuration
variables (a : ℝ)

-- Given properties
def identical_cubes_diagonals_same_line : Prop := true -- Assuming this is true as per given info
def vertex_of_second_cube_coincides_with_center_of_first_cube : Prop := true -- Assuming this is true as per given info
def second_cube_rotated_by_60_degrees_around_diagonal : Prop := true -- Assuming this is true as per given info

-- The theorem to prove
theorem volume_of_intersection_of_cubes :
  identical_cubes_diagonals_same_line ∧
  vertex_of_second_cube_coincides_with_center_of_first_cube ∧
  second_cube_rotated_by_60_degrees_around_diagonal →
  (∃ V, V = (9 * a^3) / 64) :=
by 
  -- We skip the proof here with sorry, as instructed
  intros,
  sorry

end volume_of_intersection_of_cubes_l447_447273


namespace inequality_x_y_z_squares_l447_447450

theorem inequality_x_y_z_squares (x y z m : ℝ) (h : x + y + z = m) : x^2 + y^2 + z^2 ≥ (m^2) / 3 := by
  sorry

end inequality_x_y_z_squares_l447_447450


namespace point_M_hyperbola_condition_l447_447670

theorem point_M_hyperbola_condition :
  (∀ M : Point, 
    (5 * M.x + 12 * M.y = 0) → 
    ∀ F1 F2 : Point, 
      (F1 = Point.mk (-13) 0) → 
      (F2 = Point.mk 13 0) → 
      (|dist M F1 - dist M F2| ≠ 24)) :=
begin
  -- proof omitted
  sorry
end

end point_M_hyperbola_condition_l447_447670


namespace maria_bought_9_hardcover_volumes_l447_447418

def total_volumes (h p : ℕ) : Prop := h + p = 15
def total_cost (h p : ℕ) : Prop := 10 * p + 30 * h = 330

theorem maria_bought_9_hardcover_volumes (h p : ℕ) (h_vol : total_volumes h p) (h_cost : total_cost h p) : h = 9 :=
by
  sorry

end maria_bought_9_hardcover_volumes_l447_447418


namespace sum_of_fractions_l447_447519

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447519


namespace resolvent_kernel_l447_447089

noncomputable def resolvent_K (x t λ : ℝ) : ℝ :=
  (3 * x * t) / (3 - 2 * λ) + (5 * (x^2) * (t^2)) / (5 - 2 * λ)

theorem resolvent_kernel (K : ℝ → ℝ → ℝ)
  (hK : ∀ x t, K x t = x * t + x^2 * t^2)
  (λ : ℝ)
  (hλ : |λ| < 3 / 2) :
  ∀ x t, resolvent_K x t λ = (3 * x * t) / (3 - 2 * λ) + (5 * (x^2) * (t^2)) / (5 - 2 * λ) :=
by
  intros x t
  rw hK
  sorry

end resolvent_kernel_l447_447089


namespace unique_triangle_with_consecutive_sides_and_angle_condition_l447_447674

theorem unique_triangle_with_consecutive_sides_and_angle_condition
    (a b c : ℕ) (A B C : ℝ) (h1 : a < b ∧ b < c)
    (h2 : b = a + 1 ∧ c = a + 2)
    (h3 : C = 2 * B)
    (h4 : ∀ x y z : ℕ, x < y ∧ y < z → y = x + 1 ∧ z = x + 2 → 2 * B = C)
    : ∃! (a b c : ℕ) (A B C : ℝ), (a < b ∧ b < c) ∧ (b = a + 1 ∧ c = a + 2) ∧ (C = 2 * B) :=
  sorry

end unique_triangle_with_consecutive_sides_and_angle_condition_l447_447674


namespace division_remainder_l447_447400

theorem division_remainder : 1234567 % 112 = 0 := 
by 
  sorry

end division_remainder_l447_447400


namespace min_possible_value_of_N_l447_447685

noncomputable def P (z : ℂ) : ℂ := z^2 + 1
noncomputable def Q (z : ℂ) : ℂ := z^3 + 2
noncomputable def R (z : ℂ) : ℂ := (z+1)^6 + P z * Q z

theorem min_possible_value_of_N :
  ∃ N : ℤ, (∀ z : ℂ, P z * Q z = R z → z = -1) ∧ N = 1 :=
begin
  -- Definitions and degrees
  have deg_P : polynomial.degree (polynomial.C (1 : ℂ) + polynomial.X^2) = 2,
  { sorry },
  have deg_Q : polynomial.degree (polynomial.C (2 : ℂ) + polynomial.X^3) = 3,
  { sorry },
  have deg_R : polynomial.degree ((polynomial.C (1:ℂ) + polynomial.X)^6 + 
                                   (polynomial.C (1 : ℂ) + polynomial.X^2) * 
                                   (polynomial.C (2 : ℂ) + polynomial.X^3)) = 6,
  { sorry },
  -- Proving the minimum number N of distinct complex solutions
  use 1,
  split,
  { intros z hz,
    have hz' : (z+1)^6 = 0, by { sorry },
    linarith [polynomial.eq_zero_of_polynomial_eq_zero hz'],
    sorry
  },
  { refl },
end

end min_possible_value_of_N_l447_447685


namespace problem1_problem2_l447_447495

def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

theorem problem1 (x : ℝ) : f(x) ≤ 2 ↔ -4 ≤ x ∧ x ≤ 10 := sorry

theorem problem2 (m : ℝ) (h : ∃ x, f(x) - g(x) ≥ m - 3) : m ≤ 5 := sorry

end problem1_problem2_l447_447495


namespace no_55_rooms_l447_447960

theorem no_55_rooms 
  (count_roses count_carnations count_chrysanthemums : ℕ)
  (rooms_with_CC rooms_with_CR rooms_with_HR : ℕ)
  (at_least_one_bouquet_in_each_room: ∀ (room: ℕ), room > 0)
  (total_rooms : ℕ)
  (h_bouquets : count_roses = 30 ∧ count_carnations = 20 ∧ count_chrysanthemums = 10)
  (h_overlap_conditions: rooms_with_CC = 2 ∧ rooms_with_CR = 3 ∧ rooms_with_HR = 4):
  (total_rooms != 55) :=
sorry

end no_55_rooms_l447_447960


namespace construct_quadratic_l447_447409

-- Definitions from the problem's conditions
def quadratic_has_zeros (f : ℝ → ℝ) (r1 r2 : ℝ) : Prop :=
  f r1 = 0 ∧ f r2 = 0

def quadratic_value_at (f : ℝ → ℝ) (x_val value : ℝ) : Prop :=
  f x_val = value

-- Construct the Lean theorem statement
theorem construct_quadratic :
  ∃ f : ℝ → ℝ, quadratic_has_zeros f 1 5 ∧ quadratic_value_at f 3 10 ∧
  ∀ x, f x = (-5/2 : ℝ) * x^2 + 15 * x - 25 / 2 :=
sorry

end construct_quadratic_l447_447409


namespace particle_position_after_2023_minutes_l447_447370

theorem particle_position_after_2023_minutes
  (origin : ℕ × ℕ := (0,0))      
  (movement_pattern : ∀ n : ℕ, n ∈ {1, 3, 5, 7, ...})
  (time_to_complete_nth_square : ∀ n : ℕ, n > 0 → ℕ := λ n h, 4 * (2 * n - 1))
  (total_time_to_complete_n_squares : ∀ n : ℕ, ℕ := λ n, 4 * n^2)
  (first_minute_movement : ℕ × ℕ := (1,0)) :
  let n := nat.sqrt (2023 / 4) in
  let remaining_time := 2023 - total_time_to_complete_n_squares n in
  let final_position := (remaining_time, 0) in
  final_position = (87,0) :=
sorry

end particle_position_after_2023_minutes_l447_447370


namespace dave_trips_l447_447051

theorem dave_trips :
  let trays_at_a_time := 12
  let trays_table_1 := 26
  let trays_table_2 := 49
  let trays_table_3 := 65
  let trays_table_4 := 38
  let total_trays := trays_table_1 + trays_table_2 + trays_table_3 + trays_table_4
  let trips := (total_trays + trays_at_a_time - 1) / trays_at_a_time
  trips = 15 := by
    repeat { sorry }

end dave_trips_l447_447051


namespace parabola_tangent_line_l447_447055

theorem parabola_tangent_line (a : ℝ) :
  ∀ (x : ℝ), (y = ax^2 + 10) ∧ (y = 2x) → (x = 0 ∧ a = 1/10) := by
sorry

end parabola_tangent_line_l447_447055


namespace fill_tank_with_only_C_l447_447365

noncomputable def time_to_fill_with_only_C (x y z : ℝ) : ℝ := 
  let eq1 := (1 / z - 1 / x) * 2 = 1
  let eq2 := (1 / z - 1 / y) * 4 = 1
  let eq3 := 1 / z * 5 - (1 / x + 1 / y) * 8 = 0
  z

theorem fill_tank_with_only_C (x y z : ℝ) (h1 : (1 / z - 1 / x) * 2 = 1) 
  (h2 : (1 / z - 1 / y) * 4 = 1) (h3 : 1 / z * 5 - (1 / x + 1 / y) * 8 = 0) : 
  time_to_fill_with_only_C x y z = 11 / 6 :=
by
  sorry

end fill_tank_with_only_C_l447_447365


namespace rhombus_height_l447_447880

theorem rhombus_height (d1 d2 : ℝ) (h : ℝ) (side : ℝ) 
  (h_d1 : d1 = 6) (h_d2 : d2 = 8) 
  (h_side : side = real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) :
  side * h = d1 * d2 / 2 → h = 24 / 5 :=
by
  intro hyp_area
  simp [h_d1, h_d2, h_side] at *
  sorry

end rhombus_height_l447_447880


namespace interval_mono_increasing_l447_447447

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - Real.sqrt 3 * Real.cos x * Real.cos (x + π / 2)

theorem interval_mono_increasing :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), (0 ≤ x) ∧ (x ≤ π / 3) → monotone_on f (Icc 0 (π / 3)) := sorry

end interval_mono_increasing_l447_447447


namespace ellipse_h_k_a_b_sum_l447_447591

noncomputable def h : ℝ := 1
noncomputable def k : ℝ := 1
noncomputable def a : ℝ := abs (1 - 4)
noncomputable def c : ℝ := abs (1 - 8)
noncomputable def b : ℝ := Real.sqrt (c ^ 2 - a ^ 2)

theorem ellipse_h_k_a_b_sum : h + k + a + b = 5 + 2 * Real.sqrt 10 := by
  -- assert the definitions
  have h_eq : h = 1 := rfl
  have k_eq : k = 1 := rfl
  have a_eq : a = 3 := by simp [a, abs]
  have c_eq : c = 7 := by simp [c, abs]
  have b_eq : b = 2 * Real.sqrt 10 := by
    simp [b, c, a]
    rw [c_eq, a_eq]
    norm_num
  -- combine in the main result
  rw [h_eq, k_eq, a_eq, b_eq]
  norm_num
  sorry

end ellipse_h_k_a_b_sum_l447_447591


namespace ratio_is_four_to_one_l447_447741

-- Define the problem
noncomputable def ratio_of_liquid_rise (r1 r2 : ℝ) (h1 h2 : ℝ) (m_radius : ℝ) (v_eq : ℝ) : ℝ :=
  let marble_volume := (4 / 3) * Real.pi * (m_radius ^ 3)
  let cone_volume_1 := (1 / 3) * Real.pi * (r1 ^ 2) * h1
  let cone_volume_2 := (1 / 3) * Real.pi * (r2 ^ 2) * h2
  have h_volumes_eq : cone_volume_1 = cone_volume_2 := by
    rw [cone_volume_1, cone_volume_2, v_eq]
  let new_height_1 := h1 + (marble_volume * 3 / (Real.pi * (r1 ^ 2)))
  let new_height_2 := h2 + (marble_volume * 3 / (Real.pi * (r2 ^ 2)))
  let rise_1 := new_height_1 - h1
  let rise_2 := new_height_2 - h2
  rise_1 / rise_2

-- The statement we need to prove
theorem ratio_is_four_to_one (h1 h2 : ℝ) (r1 r2 : ℝ) (m_radius : ℝ) (v_eq : ℝ)
  (h_eq : r1 = 4) (h_eq2 : r2 = 8) (h_radius : m_radius = 2) (v_eq : v_eq = (16 / 3 * Real.pi) * h1 = (64 / 3 * Real.pi) * h2 ∧ (h1 / h2 = 4)): 
  ratio_of_liquid_rise r1 r2 h1 h2 2 v_eq = 4 := by
  -- Proof yet to be completed
  sorry

end ratio_is_four_to_one_l447_447741


namespace expected_value_of_X_l447_447030

noncomputable def expected_value_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

theorem expected_value_of_X (n : ℕ) (h : n > 0) :
  let X : ℕ → ℕ → ℚ := λ i n, (i - 1 : ℚ) / n
  let E_X : ℚ := ∑ i in Finset.range n, X (i + 1) n
  E_X = expected_value_shorter_gentlemen n := by
{
  sorry
}

end expected_value_of_X_l447_447030


namespace polynomial_roots_problem_l447_447470

theorem polynomial_roots_problem (γ δ : ℝ) (h₁ : γ^2 - 3*γ + 2 = 0) (h₂ : δ^2 - 3*δ + 2 = 0) :
  8*γ^3 - 6*δ^2 = 48 :=
by
  sorry

end polynomial_roots_problem_l447_447470


namespace expression_value_l447_447552

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447552


namespace vote_percentage_for_candidate_A_l447_447340

noncomputable def percent_democrats : ℝ := 0.60
noncomputable def percent_republicans : ℝ := 0.40
noncomputable def percent_voting_a_democrats : ℝ := 0.70
noncomputable def percent_voting_a_republicans : ℝ := 0.20

theorem vote_percentage_for_candidate_A :
    (percent_democrats * percent_voting_a_democrats + percent_republicans * percent_voting_a_republicans) * 100 = 50 := by
  sorry

end vote_percentage_for_candidate_A_l447_447340


namespace triangle_inequality_l447_447596

theorem triangle_inequality
  (a b c r s_a s_b : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_median_a : s_a = 1 / 2 * sqrt (2 * b^2 + 2 * c^2 - a^2))
  (h_median_b : s_b = 1 / 2 * sqrt (2 * a^2 + 2 * c^2 - b^2))
  (h_inradius : r = (a + b - c) / 2) :
  r^2 / (s_a^2 + s_b^2) ≤ (3 - 2 * sqrt 2) / 5 :=
by
  sorry

end triangle_inequality_l447_447596


namespace total_income_percentage_l447_447228

-- Define the base income of Juan
def juan_base_income (J : ℝ) := J

-- Define Tim's base income
def tim_base_income (J : ℝ) := 0.70 * J

-- Define Mary's total income
def mary_total_income (J : ℝ) := 1.232 * J

-- Define Lisa's total income
def lisa_total_income (J : ℝ) := 0.6489 * J

-- Define Nina's total income
def nina_total_income (J : ℝ) := 1.3375 * J

-- Define the sum of the total incomes of Mary, Lisa, and Nina
def sum_income (J : ℝ) := mary_total_income J + lisa_total_income J + nina_total_income J

-- Define the statement we need to prove: the percentage of Juan's total income
theorem total_income_percentage (J : ℝ) (hJ : J ≠ 0) :
  ((sum_income J / juan_base_income J) * 100) = 321.84 :=
by
  unfold juan_base_income sum_income mary_total_income lisa_total_income nina_total_income
  sorry

end total_income_percentage_l447_447228


namespace expression_value_l447_447554

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447554


namespace simplify_sqrt_expression_l447_447682

theorem simplify_sqrt_expression :
  ( ( ( ( ( sqrt 5 ) - 2 ) ^ ( sqrt 3 - 2 ) ) / ( ( sqrt 5 + 2 ) ^ ( sqrt 3 + 2 ) ) ) 
  = 41 + 20 * sqrt 5 ) :=
by
  sorry

end simplify_sqrt_expression_l447_447682


namespace convert_spherical_to_rectangular_l447_447822

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem convert_spherical_to_rectangular : spherical_to_rectangular 5 (Real.pi / 2) (Real.pi / 3) = 
  (0, 5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  sorry

end convert_spherical_to_rectangular_l447_447822


namespace janet_earnings_per_hour_l447_447618

theorem janet_earnings_per_hour : 
  (∃ (rate_per_post : ℝ) (time_per_post : ℝ), 
    rate_per_post = 0.25 ∧ 
    time_per_post = 10 ∧ 
    (let posts_per_hour := 3600 / time_per_post in
     let earnings_per_hour := rate_per_post * posts_per_hour in
     earnings_per_hour = 90)) :=
by
  use 0.25
  use 10
  split
  rfl
  split
  rfl
  let posts_per_hour := 3600 / 10
  let earnings_per_hour := 0.25 * posts_per_hour
  have h : earnings_per_hour = 90, by sorry
  exact h

end janet_earnings_per_hour_l447_447618


namespace prob_exactly_two_successes_l447_447382

/-!
# Problem: Probability of exactly two out of three people successfully decrypting a password

Given:
1. A, B, and C each independently attempt to decrypt a password.
2. The probability that each succeeds is 1/4.

Prove:
The probability that exactly two of them successfully decrypt the password is 9/64.
-/

-- Define the probability of success for each person
def prob_success (p : ℚ) := p

-- Define the probability of failure for each person
def prob_failure (p : ℚ) := 1 - p

-- Define the number of ways to choose 2 successes out of 3 attempts
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of exactly two successes out of three attempts
def prob_two_successes (p : ℚ) : ℚ :=
  (binom 3 2) * (p^2) * (prob_failure p)

theorem prob_exactly_two_successes :
  prob_two_successes (1/4) = 9/64 :=
by
  have h : prob_success (1/4) = 1/4 := rfl
  have h' : prob_failure (1/4) = 3/4 := rfl
  have c : binom 3 2 = 3 := rfl
  have calc_prob := by
    calc
    prob_two_successes (1/4)
        = (binom 3 2) * ((1/4)^2) * (3/4) : rfl
    ... = 3 * (1/16) * (3/4) : by rw [h', pow_two (1/4)]
    ... = 3 * (3/64) : rfl
    ... = 9/64 : rfl
  exact calc_prob

end prob_exactly_two_successes_l447_447382


namespace classes_after_drop_remaining_hours_of_classes_per_day_l447_447207

def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_classes : ℕ := 1

theorem classes_after_drop 
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ) :
  initial_classes - dropped_classes = 3 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

theorem remaining_hours_of_classes_per_day
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ)
  (h : initial_classes - dropped_classes = 3) :
  hours_per_class * (initial_classes - dropped_classes) = 6 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

end classes_after_drop_remaining_hours_of_classes_per_day_l447_447207


namespace div_poly_l447_447671

theorem div_poly (m n p : ℕ) : 
  (X^2 + X + 1) ∣ (X^(3*m) + X^(3*n + 1) + X^(3*p + 2)) := 
sorry

end div_poly_l447_447671


namespace vasya_mushrooms_l447_447743

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end vasya_mushrooms_l447_447743


namespace domain_of_f_l447_447748

def f (x : ℝ) : ℝ := real.sqrt (2 * x - 4) + real.cbrt (x - 5)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≥ 2} :=
by
  sorry

end domain_of_f_l447_447748


namespace log_ordering_l447_447578

theorem log_ordering (a b c : ℝ) (ha : a = Real.log 3 π) (hb : b = Real.log 7 6) (hc : c = Real.log 2 0.8) : 
  c < b ∧ b < a :=
by
  sorry

end log_ordering_l447_447578


namespace two_common_tangents_length_AB_max_distance_EF_l447_447145

-- Definitions for the circles and their properties
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 4 = 0

-- Statements equivalences in Lean 4
theorem two_common_tangents : (∃ A B, circle_O A.1 A.2 ∧ circle_M A.1 A.2 ∧ circle_O B.1 B.2 ∧ circle_M B.1 B.2)
  → (distance (0,0) (-2,1) = sqrt 5 ∧ 1 < sqrt 5 ∧ sqrt 5 < 3)
  → ∃ T1 T2, line T1 ∧ line T2 ∧ tangent T1 ∧ tangent T2
:= sorry

theorem length_AB (A B : ℝ × ℝ) 
  (hA : circle_O A.1 A.2 ∧ circle_M A.1 A.2)
  (hB : circle_O B.1 B.2 ∧ circle_M B.1 B.2) :
  distance A B = (4 * sqrt 5) / 5
:= sorry

theorem max_distance_EF (E F : ℝ × ℝ) 
  (hE : circle_O E.1 E.2)
  (hF : circle_M F.1 F.2) :
  max_distance E F = sqrt 5 + 3
:= sorry

end two_common_tangents_length_AB_max_distance_EF_l447_447145


namespace tripod_height_after_breaks_l447_447791

noncomputable def calculate_tripod_height (leg1 leg2 leg3 : ℝ) (original_height : ℝ) : ℝ :=
  let average_leg_length := (leg1 + leg2 + leg3) / 3
  in (average_leg_length / 6) * original_height

theorem tripod_height_after_breaks :
  let legA := 6
  let legB := 5
  let legC := 4
  let original_height := 5
  h = calculate_tripod_height legA legB legC original_height
  ⟶ h = 25/6 :=
by sorry

end tripod_height_after_breaks_l447_447791


namespace right_triangle_extension_length_l447_447179

theorem right_triangle_extension_length
  (A B C : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (dist : A → B → ℝ)
  (h1 : dist A C = 16)
  (h2 : dist B C = 12)
  (h_right : dist A C * dist A C + dist B C * dist B C = dist A B * dist A B)
  : ∃ K : Type*,
  ∃ dist_B_K : B → K → ℝ,
  dist_B_K B K = 15 :=
sorry

end right_triangle_extension_length_l447_447179


namespace quiz_competition_order_l447_447594

theorem quiz_competition_order
  (A B C : ℤ)
  (A_f B_f C_f : ℤ)
  (h1 : A + B = 2 * C)
  (h2 : A_f > B_f + C_f + 10)
  (h3 : B_f = C_f + 5)
  (hAf : A_f = 3 * A)
  (hBf : B_f = 3 * B)
  (hCf : C_f = 3 * C)
  (h_pos_A : A > 0)
  (h_pos_B : B > 0)
  (h_pos_C : C > 0):
  "Carol, Alan, Bob" := 
sorry

end quiz_competition_order_l447_447594


namespace problem_statement_l447_447565

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447565


namespace expression_value_l447_447542

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447542


namespace problem_statement_l447_447563

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l447_447563


namespace total_maple_trees_in_park_after_planting_l447_447306

def number_of_maple_trees_in_the_park (X_M : ℕ) (Y_M : ℕ) : ℕ := 
  X_M + Y_M

theorem total_maple_trees_in_park_after_planting : 
  number_of_maple_trees_in_the_park 2 9 = 11 := 
by 
  unfold number_of_maple_trees_in_the_park
  -- provide the mathematical proof here
  sorry

end total_maple_trees_in_park_after_planting_l447_447306


namespace euler_totient_sum_final_answer_l447_447353

-- Define the set S
def S : Set ℕ := {n | ∀ p, Nat.Prime p → p ∣ n → p ∈ {2, 3, 5, 7, 11}}

-- Define the Euler's totient function
def euler_totient (n : ℕ) : ℕ := Nat.totient n

-- Define the main theorem to be proven
theorem euler_totient_sum :
  (∑ q in S, (euler_totient q : ℚ) / (q ^ 2 : ℚ)) = 1152 / 385 := 
sorry

-- Final step to calculate a + b
theorem final_answer : 1152 + 385 = 1537 :=
by norm_num

end euler_totient_sum_final_answer_l447_447353


namespace arithmetic_sequence_S9_l447_447647

-- Definitions representing the conditions in the problem
variable {α : Type*} [OrderedRing α] {a_n : ℕ → α}

-- Condition from the problem: 4 + a5 = a6 + a4
axiom condition1 : 4 + a_n 5 = a_n 6 + a_n 4

-- Definition of the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) : α := (n * (a_n 1 + a_n n)) / 2

-- Prove the sum of the first 9 terms equals 36
theorem arithmetic_sequence_S9 : S_n 9 = 36 :=
by sorry

end arithmetic_sequence_S9_l447_447647


namespace final_value_l447_447545

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l447_447545


namespace solution_system_solution_rational_l447_447260

-- Definitions for the system of equations
def sys_eq_1 (x y : ℤ) : Prop := 2 * x - y = 3
def sys_eq_2 (x y : ℤ) : Prop := x + y = -12

-- Theorem to prove the solution of the system of equations
theorem solution_system (x y : ℤ) (h1 : sys_eq_1 x y) (h2 : sys_eq_2 x y) : x = -3 ∧ y = -9 :=
by {
  sorry
}

-- Definition for the rational equation
def rational_eq (x : ℤ) : Prop := (2 / (1 - x) : ℚ) + 1 = (x / (1 + x) : ℚ)

-- Theorem to prove the solution of the rational equation
theorem solution_rational (x : ℤ) (h : rational_eq x) : x = -3 :=
by {
  sorry
}

end solution_system_solution_rational_l447_447260


namespace participation_plans_count_l447_447681

theorem participation_plans_count :
  let students := {A, B, C, D, E}
  let competitions := {mathematics, physical, chemical}
  let consists := list.combinations 3 students
  (count (λ subset → 
    ∃ (assignment : list (student * competition)),
    (∀ (sc : student * competition ), sc ∈ assignment → prod.snd sc ∈ competitions ∧ prod.fst sc ∈ subset ∧ (prod.snd sc ≠ physical ∨ prod.fst sc ≠ A)) ∧
    ∃ (comp_assign : list competition), comp_assign ~≈ competitions ∧ list.any (map prod.snd assignment) = comp_assign), consists) = 48 :=
begin
  sorry
end

end participation_plans_count_l447_447681


namespace minimum_disks_needed_l447_447981

def files : ℕ := 30
def disk_space : ℝ := 1.44  -- Disk capacity in MB
def file_sizes : list ℝ := [0.8, 0.8, 0.8] ++ (repeat 0.7 12) ++ (repeat 0.4 15)  -- List of file sizes

def total_disks_needed : ℕ :=
  let num_large_files := countp (λ x, x = 0.8) file_sizes in
  let num_medium_files := countp (λ x, x = 0.7) file_sizes in
  let num_small_files := countp (λ x, x = 0.4) file_sizes in
  let paired_with_small := min num_large_files num_small_files in
  let remaining_large := num_large_files - paired_with_small in
  let remaining_small := num_small_files - paired_with_small in
  let total_remaining_files_size := (num_medium_files * 0.7) + (remaining_small * 0.4) in
  let additional_disks := ⌈ total_remaining_files_size / disk_space ⌉ in
  paired_with_small + remaining_large + additional_disks

theorem minimum_disks_needed (h : total_disks_needed = 13) : 
  ∀ (num_files : ℕ) (space_disk : ℝ) (sizes : list ℝ), num_files = 30 → space_disk = 1.44 → sizes = file_sizes → total_disks_needed = 13 :=
by
  intros
  sorry

end minimum_disks_needed_l447_447981


namespace prod_inequality_l447_447208

theorem prod_inequality (n : ℕ) (x : Fin (n+2) → ℝ) 
  (h_x_range : ∀ i, 0 ≤ x i ∧ x i ≤ 1)
  (h_x1_eq_xnplus1 : x ⟨0⟩ = x ⟨n + 1⟩) :
  (∏ i in Finset.range n, (1 - (x ⟨i+1⟩ * x ⟨i+2⟩) + (x ⟨i+1⟩^2))) ≥ 1 := sorry

end prod_inequality_l447_447208


namespace hyperbola_eccentricity_solution_l447_447692

def hyperbola_eccentricity_problem : Prop :=
  ∃ a b c e : ℝ,
  (a^2 = 1) ∧ (b^2 = 4) ∧ (c^2 = a^2 + b^2) ∧ (e = c / a) ∧ (e = sqrt 5)

theorem hyperbola_eccentricity_solution : hyperbola_eccentricity_problem :=
begin
  use 1,       -- a
  use 2,       -- b
  use sqrt 5,  -- c
  use sqrt 5,  -- e
  split,
  { norm_num },                    -- prove a^2 = 1 
  split,
  { norm_num },                    -- prove b^2 = 4 
  split,
  { norm_num },                    -- prove c^2 = a^2 + b^2 
  split,
  { norm_num, rw [mul_one (sqrt 5)] },  -- prove e = c / a
  { refl },                        -- prove e = sqrt 5
end

end hyperbola_eccentricity_solution_l447_447692


namespace sum_floor_eq_7400_l447_447632

def S := {p : ℕ × ℕ | (0 < p.1 ∧ 0 < p.2) ∧ Nat.gcd p.1 p.2 = 1}

theorem sum_floor_eq_7400 : 
  (∑ p in S, (Int.floor (300 / (2 * p.1 + 3 * p.2)))) = 7400 := 
  sorry

end sum_floor_eq_7400_l447_447632


namespace contradiction_assumption_l447_447270

variable (x y z : ℝ)

/-- The negation of "at least one is positive" for proof by contradiction is 
    "all three numbers are non-positive". -/
theorem contradiction_assumption (h : ¬ (x > 0 ∨ y > 0 ∨ z > 0)) : 
  (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) :=
by
  sorry

end contradiction_assumption_l447_447270


namespace evaluate_expression_l447_447422

def a := 3 + 6 + 9
def b := 2 + 5 + 8
def c := 3 + 6 + 9
def d := 2 + 5 + 8

theorem evaluate_expression : (a / b) - (d / c) = 11 / 30 :=
by
  sorry

end evaluate_expression_l447_447422


namespace vector_at_t1_is_correct_l447_447366

-- Define the parameterized line as a + t*d
structure Line (V : Type) [AddCommGroup V] [Module ℝ V] :=
  (a : V)
  (d : V)
  (param : ℝ → V)

-- Specific given vectors at t = 4 and t = 5
def vec_t4 : ℝ × ℝ := (2,5)
def vec_t5 : ℝ × ℝ := (4,-3)

-- The conditions
def condition1 (L : Line (ℝ × ℝ)) : Prop :=
  L.param 4 = vec_t4

def condition2 (L : Line (ℝ × ℝ)) : Prop :=
  L.param 5 = vec_t5

-- The vector at t = 1
def vec_t1 (L : Line (ℝ × ℝ)) : ℝ × ℝ :=
  L.a + (1 : ℝ) • L.d

-- The proof statement
theorem vector_at_t1_is_correct (L : Line (ℝ × ℝ)) 
  (h1 : condition1 L) 
  (h2 : condition2 L) : 
  vec_t1 L = (8, -19) := 
by 
  sorry

end vector_at_t1_is_correct_l447_447366


namespace highland_park_science_fair_l447_447798

noncomputable def juniors_and_seniors_participants (j s : ℕ) : ℕ :=
  (3 * j) / 4 + s / 2

theorem highland_park_science_fair 
  (j s : ℕ)
  (h1 : (3 * j) / 4 = s / 2)
  (h2 : j + s = 240) :
  juniors_and_seniors_participants j s = 144 := by
  sorry

end highland_park_science_fair_l447_447798


namespace nested_radicals_solution_l447_447840

noncomputable def g (x : ℝ) := Real.sqrt 23 + 101 / x

theorem nested_radicals_solution :
  let B := abs (Real.sqrt 23 + Real.sqrt 427) / 2 + abs (Real.sqrt 23 - Real.sqrt 427) / 2
  in B^2 = 427 :=
by
  let g : ℝ → ℝ := λ x, Real.sqrt 23 + 101 / x
  have h : g ∘ g ∘ g ∘ g ∘ g = id := sorry
  have roots := sorry -- Solve polynomial equation
  let B := abs (Real.sqrt 23 + Real.sqrt 427) / 2 + abs (Real.sqrt 23 - Real.sqrt 427) / 2
  show B^2 = 427, from sorry

end nested_radicals_solution_l447_447840


namespace intersection_of_A_B_l447_447993

variable (A : Set ℝ) (B : Set ℝ)

theorem intersection_of_A_B (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {x : ℝ | 0 < x ∧ x < 3}) :
  A ∩ B = {1, 2} :=
  sorry

end intersection_of_A_B_l447_447993


namespace f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l447_447492

open Real

noncomputable def f : ℝ → ℝ :=
sorry

axiom func_prop : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom pos_x_gt_1 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_1 : f 1 = 2

-- Prove that f(0) = 1
theorem f_0_eq_1 : f 0 = 1 :=
sorry

-- Prove that f(-1) ≠ 1 (and direct derivation showing f(-1) = 0)
theorem f_neg_1_ne_1 : f (-1) ≠ 1 ∧ f (-1) = 0 :=
sorry

-- Prove that f(x) is increasing
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ > f x₁ :=
sorry

-- Prove minimum value of f on [-3, 3] is -2
theorem min_f_neg3_3 : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -2 :=
sorry

end f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l447_447492


namespace fifteen_balls_three_boxes_l447_447678

theorem fifteen_balls_three_boxes :
  ∃ (ways : ℕ), ways = 91 ∧
    (∀ (x y z : ℕ), x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3 ∧ x + y + z = 15 → ways = 91) :=
begin
  existsi 91,
  split,
  { refl },
  { intros x y z,
    sorry }
end

end fifteen_balls_three_boxes_l447_447678


namespace circle_area_equality_l447_447155

-- Define the concept of a circle's radius and area
def radius_of_circle (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * (r * r)

-- Given Conditions
def radius1 : ℝ := 30
def circumference2 : ℝ := 60 * Real.pi
def radius2 : ℝ := radius_of_circle circumference2

-- Theorem to prove
theorem circle_area_equality : area_of_circle radius1 = area_of_circle radius2 :=
by
  sorry

end circle_area_equality_l447_447155


namespace fraction_value_l447_447747

theorem fraction_value : (3^2 + 3^2) / (3^(-2) + 3^(-2)) = 81 :=
by
  sorry

end fraction_value_l447_447747


namespace value_of_g_13_l447_447943

def g (n : ℕ) : ℕ := n^2 + 2 * n + 23

theorem value_of_g_13 : g 13 = 218 :=
by 
  sorry

end value_of_g_13_l447_447943


namespace shaded_area_of_rectangle_l447_447612

theorem shaded_area_of_rectangle :
  let length := 5   -- Length of the rectangle in cm
  let width := 12   -- Width of the rectangle in cm
  let base := 2     -- Base of each triangle in cm
  let height := 5   -- Height of each triangle in cm
  let rect_area := length * width
  let triangle_area := (1 / 2) * base * height
  let unshaded_area := 2 * triangle_area
  let shaded_area := rect_area - unshaded_area
  shaded_area = 50 :=
by
  -- Calculation follows solution steps.
  sorry

end shaded_area_of_rectangle_l447_447612


namespace derivative_f_at_pi_div_2_l447_447130

-- Define the function f
def f (x : ℝ) : ℝ := x / sin x

-- State the theorem to prove that the derivative of f at π/2 is 1
theorem derivative_f_at_pi_div_2 : deriv f (π / 2) = 1 := by
  sorry

end derivative_f_at_pi_div_2_l447_447130


namespace number_of_unique_four_digit_numbers_from_2004_l447_447933

-- Definitions representing the conditions
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def uses_digits_from_2004 (n : ℕ) : Prop := 
  ∀ d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10], d ∈ [0, 2, 4]

-- The proposition we need to prove
theorem number_of_unique_four_digit_numbers_from_2004 :
  ∃ n : ℕ, is_four_digit_number n ∧ uses_digits_from_2004 n ∧ n = 6 := 
sorry

end number_of_unique_four_digit_numbers_from_2004_l447_447933


namespace tangent_line_value_l447_447483

theorem tangent_line_value {k : ℝ} 
  (h1 : ∃ x y : ℝ, x^2 + y^2 - 6*y + 8 = 0) 
  (h2 : ∃ P Q : ℝ, x^2 + y^2 - 6*y + 8 = 0 ∧ Q = k * P)
  (h3 : P * k < 0 ∧ P < 0 ∧ Q > 0) : 
  k = -2 * Real.sqrt 2 :=
sorry

end tangent_line_value_l447_447483


namespace min_value_ab_l447_447497

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a / 2) + b = 1) :
  (1 / a) + (1 / b) = (3 / 2) + Real.sqrt 2 :=
by sorry

end min_value_ab_l447_447497


namespace find_line_equation_l447_447730

theorem find_line_equation (k : ℝ) (m b : ℝ) :
  (∃ k : ℝ, |(k^2 + 4*k + 4) - (m*k + b)| = 10) → 
  (m*1 + b = 6) → 
  (b ≠ 0) → 
  (y = 4*x + 2) := 
begin
  sorry
end

end find_line_equation_l447_447730


namespace expression_value_l447_447558

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l447_447558


namespace hyperbola_equation_is_correct_slope_k_is_valid_l447_447451

noncomputable def correct_hyperbola_equation : Prop :=
∀ (a b : ℝ), a^2 + b^2 = 3 → (a = sqrt 2) ∧ (b = 1)

theorem hyperbola_equation_is_correct :
  ∃ a b : ℝ, a^2 + b^2 = 3 ∧ a = sqrt 2 ∧ b = 1 → (∀ x y : ℝ, x^2 / 2 - y^2 = 1) :=
by
  sorry

noncomputable def valid_slope_k : Prop :=
  ( ∀ k : ℝ, (-3 + sqrt 11) / 2 ≤ k ∧ k ≤ (-3 - sqrt 11) / 2 ∨ k = 0 )

theorem slope_k_is_valid :
  ∃ k : ℝ, ( ∀ x y : ℝ, let A := (-sqrt 2, 0) and B := (sqrt 2, 0) in (y = k * (x + 2)) ∧ x = y ∧ x = -4k² / (1 - 2k²) ∧ y = 2k / (1 - 2k²) → k = (frac -3 ± sqrt 11) / 2 ∨ k = 0 ) :=
by
  sorry

end hyperbola_equation_is_correct_slope_k_is_valid_l447_447451


namespace intersection_point_coordinates_l447_447478

variables 
  {x1 y1 x2 y2 x3 y3 l m n : ℝ}
  (h₁ : m ≠ -l) 
  (h₂ : n ≠ -l) 
  (h₃ : l + m + n ≠ 0)

def P_x : ℝ := (l * x1 + m * x2 + n * x3) / (l + m + n)
def P_y : ℝ := (l * y1 + m * y2 + n * y3) / (l + m + n)

theorem intersection_point_coordinates : 
  P_x h₁ h₂ h₃ = (l * x1 + m * x2 + n * x3) / (l + m + n) ∧
  P_y h₁ h₂ h₃ = (l * y1 + m * y2 + n * y3) / (l + m + n) :=
by sorry

end intersection_point_coordinates_l447_447478


namespace geometric_shape_circle_l447_447865

variables (c φ_0 : ℝ)

-- Assuming positive constants for c and φ_0
axiom h_c_pos : c > 0
axiom h_φ0_pos : φ_0 > 0

-- The main statement: Given that ρ = c and φ = φ_0, the geometric shape is a circle.
theorem geometric_shape_circle (ρ θ φ : ℝ) (h_ρ : ρ = c) (h_φ : φ = φ_0) :
  ∃ (r : ℝ), r > 0 ∧ ∀ θ, (ρ = c ∧ φ = φ_0) → (∃ (x y : ℝ), x^2 + y^2 = r^2) :=
by 
  sorry

end geometric_shape_circle_l447_447865


namespace minimum_value_of_f_ge_7_l447_447850

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end minimum_value_of_f_ge_7_l447_447850


namespace distance_between_points_l447_447038

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points:
  distance (3, -4) (-5, 7) = real.sqrt 185 := 
sorry

end distance_between_points_l447_447038


namespace prob_both_balls_drawn_are_red_prob_balls_drawn_have_different_colors_l447_447770

-- Define the basic setup of the problem
inductive BallColor
| red
| black

def totalBalls := [BallColor.red, BallColor.red, BallColor.red, BallColor.red, BallColor.black, BallColor.black]

-- Draw two balls from the list
def drawTwo (l : List BallColor) : List (BallColor × BallColor) :=
  l.choose 2

-- Define event M: both balls drawn are red
def eventM (pair : BallColor × BallColor) : Prop :=
  pair.1 = BallColor.red ∧ pair.2 = BallColor.red

-- Define event N: the two balls drawn have different colors
def eventN (pair : BallColor × BallColor) : Prop :=
  pair.1 ≠ pair.2

-- Calculate the probability of event M
def probEventM : ℚ :=
  let draws := drawTwo totalBalls
  (draws.filter eventM).length / draws.length

-- Calculate the probability of event N
def probEventN : ℚ :=
  let draws := drawTwo totalBalls
  (draws.filter eventN).length / draws.length

-- Theorems to prove
theorem prob_both_balls_drawn_are_red :
  probEventM = 2 / 5 := by
  sorry

theorem prob_balls_drawn_have_different_colors :
  probEventN = 8 / 15 := by
  sorry

end prob_both_balls_drawn_are_red_prob_balls_drawn_have_different_colors_l447_447770


namespace calculate_gcd_correct_l447_447335

theorem calculate_gcd_correct (n : ℕ) (numbers : list ℕ) (h : numbers.length = n) 
  (h_pos : ∀ x ∈ numbers, 0 < x) : 
  let a := numbers.foldl gcd 0 in
  is_gcd_of_list numbers a :=
by
  assume numbers h h_pos
  -- proof goes here
  sorry

end calculate_gcd_correct_l447_447335


namespace number_of_girls_on_playground_l447_447731

theorem number_of_girls_on_playground (boys girls total : ℕ) 
  (h1 : boys = 44) (h2 : total = 97) (h3 : total = boys + girls) : 
  girls = 53 :=
by sorry

end number_of_girls_on_playground_l447_447731


namespace max_ab2_value_l447_447474

noncomputable def maximum_ab2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ℝ :=
  let f := λ b : ℝ, (2 - b) * b^2 in
  let critical_b := sqrt 6 / 3 in
  f critical_b

theorem max_ab2_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  maximum_ab2 a b h1 h2 h3 = (4 * sqrt 6) / 9 :=
by
  sorry

end max_ab2_value_l447_447474


namespace sum_of_valid_single_digit_z_l447_447054

theorem sum_of_valid_single_digit_z :
  let valid_z (z : ℕ) := z < 10 ∧ (16 + z) % 3 = 0
  let sum_z := (Finset.filter valid_z (Finset.range 10)).sum id
  sum_z = 15 :=
by
  -- Proof steps are omitted
  sorry

end sum_of_valid_single_digit_z_l447_447054


namespace math_problem_l447_447528

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447528


namespace exactly_one_tetrahedron_formed_l447_447927

def can_form_tetrahedron (triangles : List (List ℝ)) : ℕ :=
  if triangles = [[3, 4, 5], [3, 4, 5], [4, 5, Real.sqrt 41], [4, 5, Real.sqrt 41], 
                  [4, 5, Real.sqrt 41], [4, 5, Real.sqrt 41], 
                  [5/6 * Real.sqrt 2, 4, 5], [5/6 * Real.sqrt 2, 4, 5], 
                  [5/6 * Real.sqrt 2, 4, 5], [5/6 * Real.sqrt 2, 4, 5], 
                  [5/6 * Real.sqrt 2, 4, 5], [5/6 * Real.sqrt 2, 4, 5]] then 1 else 0

theorem exactly_one_tetrahedron_formed : can_form_tetrahedron [[3, 4, 5], [3, 4, 5], 
                               [4, 5, Real.sqrt 41], [4, 5, Real.sqrt 41], 
                               [4, 5, Real.sqrt 41], [4, 5, Real.sqrt 41], 
                               [5/6 * Real.sqrt 2, 4, 5], [5/6 * Real.sqrt 2, 4, 5], 
                               [5/6 * Real.sqrt 2, 4, 5], [5/6 * Real.sqrt 2, 4, 5], 
                               [5/6 * Real.sqrt 2, 4, 5], [5/6 * Real.sqrt 2, 4, 5]] = 1 := 
by sorry

end exactly_one_tetrahedron_formed_l447_447927


namespace sum_of_sequence_l447_447814

theorem sum_of_sequence (a : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : a 1 + a 2 = 5)
  (h3 : a 1 + a 2 + a 3 = 14) 
  (h4 : ∀ k, a (k + 1) - a k = k ^ 3) : 
  (∑ k in Finset.range n, a (k + 1)) = n * (n + 1) * (2 * n + 1) / 6 := 
by 
  sorry

end sum_of_sequence_l447_447814


namespace incorrect_statement_D_l447_447751

-- Definitions based on conditions
def length_of_spring (x : ℝ) : ℝ := 8 + 0.5 * x

-- Incorrect Statement (to be proved as incorrect)
def statement_D_incorrect : Prop :=
  ¬ (length_of_spring 30 = 23)

-- Main theorem statement
theorem incorrect_statement_D : statement_D_incorrect :=
by
  sorry

end incorrect_statement_D_l447_447751


namespace symmetric_line_eq_l447_447275

-- Definitions for the given lines
def l1 (x y : ℝ) := 3 * x - 2 * y - 6 = 0
def l2 (x y : ℝ) := x - y - 2 = 0

-- Prove the equation of the symmetric line
theorem symmetric_line_eq (x y : ℝ) :
  (∃ (x1 y1 : ℝ), l2 x1 y1) ∧ (∃ (x2 y2 : ℝ), l1 x2 y2) ∧
  (∃ (x_s y_s : ℝ), symmetric_with_respect_to_line x1 y1 x2 y2 x_s y_s) →
  (2 * x - 3 * y - 4 = 0) := sorry

end symmetric_line_eq_l447_447275


namespace sum_of_coeffs_is_minus_one_l447_447938

theorem sum_of_coeffs_is_minus_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) :
  (∀ x : ℤ, (1 - x^3)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9)
  → a = 1 
  → a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end sum_of_coeffs_is_minus_one_l447_447938


namespace total_amount_paid_l447_447295

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l447_447295


namespace geo_seq_theorem_l447_447892

noncomputable theory
open_locale big_operators

variables {α : Type*} [linear_ordered_field α]

-- Assume we have a geometric sequence a : ℕ → α, with a common ratio r : α
variables (a : ℕ → α) (r : α)

-- Define the condition that a_4 + a_8 = -2
def condition := a 4 + a 8 = -2

-- Define the property of a geometric sequence
def geo_seq := ∀ n k : ℕ, a (n + k) = a n * r ^ k

-- The main theorem stating the solution
theorem geo_seq_theorem (h_geo_seq : geo_seq a r) (h_condition : condition a) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geo_seq_theorem_l447_447892


namespace prime_count_between_50_and_70_l447_447153

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : Nat) (p : Nat → Prop) : List Nat :=
  (List.range' a (b - a + 1)).filter p

theorem prime_count_between_50_and_70 : (primes_between 50 70 is_prime).length = 4 :=
by 
  sorry

end prime_count_between_50_and_70_l447_447153


namespace abs_expression_eq_l447_447812

theorem abs_expression_eq (π : ℝ) (h1 : 2 * π < 12) (h2 : 5 * π > 12) :
  |3 * π - |2 * π - 12|| = 5 * π - 12 := by
  sorry

end abs_expression_eq_l447_447812


namespace expression_value_l447_447539

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l447_447539


namespace average_salary_is_8000_l447_447342

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

def average_salary : ℕ := total_salary / num_people

theorem average_salary_is_8000 : average_salary = 8000 := by
  sorry

end average_salary_is_8000_l447_447342


namespace problem_statement_l447_447513

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447513


namespace accidents_no_solution_l447_447177

noncomputable def accidents (H W : ℕ) (a b c : ℝ) : ℝ := a * (b ^ H) * (c ^ W)

theorem accidents_no_solution {a b c : ℝ} :
  (accidents 1000 120 a b c = 8 ∧ accidents 400 80 a b c = 5) → 
  ∃ A, A = accidents 0 150 a b c ∧ id (A)) := 
sorry

end accidents_no_solution_l447_447177


namespace divisibility_of_1234xy_l447_447825

theorem divisibility_of_1234xy (x y : ℕ) (h1 : x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  (∃ (x y : ℕ), x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (10 * (10 * (10 * (10 * 1 + 2) + 3) + 4) + 10 * x + y) % 9 = 0 ∧ (100 * x + 11 * y) % 8 = 0) → 
  (x = 8 ∧ y = 0) ∨ (x = 0 ∧ y = 8) ∨ (x = 9 ∧ y = 8) ∨ (x = 8 ∧ y = 9) := 
by {
  sorry
}

end divisibility_of_1234xy_l447_447825


namespace minimum_value_frac_sum_l447_447872

theorem minimum_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 / y = 3) :
  (2 / x + y) ≥ 8 / 3 :=
sorry

end minimum_value_frac_sum_l447_447872


namespace janet_earnings_per_hour_l447_447616

theorem janet_earnings_per_hour : 
  (∃ (rate_per_post : ℝ) (time_per_post : ℝ), 
    rate_per_post = 0.25 ∧ 
    time_per_post = 10 ∧ 
    (let posts_per_hour := 3600 / time_per_post in
     let earnings_per_hour := rate_per_post * posts_per_hour in
     earnings_per_hour = 90)) :=
by
  use 0.25
  use 10
  split
  rfl
  split
  rfl
  let posts_per_hour := 3600 / 10
  let earnings_per_hour := 0.25 * posts_per_hour
  have h : earnings_per_hour = 90, by sorry
  exact h

end janet_earnings_per_hour_l447_447616


namespace expected_value_shorter_gentlemen_l447_447026

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_gentlemen_l447_447026


namespace son_father_height_relationship_is_correlation_l447_447704

-- Define the possible relationships
inductive Relationship
| Deterministic : Relationship
| Correlation : Relationship
| Functional : Relationship
| None : Relationship

-- Define the relationship between a son's height and his father's height as correlation
def relationship : (height_son height_father : ℝ) → Relationship
| _, _ := Relationship.Correlation

-- The theorem to prove
theorem son_father_height_relationship_is_correlation :
    relationship height_son height_father = Relationship.Correlation :=
by
  sorry

end son_father_height_relationship_is_correlation_l447_447704


namespace angle_no_complement_greater_than_90_l447_447577

-- Definition of angle
def angle (A : ℝ) : Prop := 
  A = 100 + (15 / 60)

-- Definition of complement
def has_complement (A : ℝ) : Prop :=
  A < 90

-- Theorem: Angles greater than 90 degrees do not have complements
theorem angle_no_complement_greater_than_90 {A : ℝ} (h: angle A) : ¬ has_complement A :=
by sorry

end angle_no_complement_greater_than_90_l447_447577


namespace parabola_equation_l447_447847

theorem parabola_equation (a b c : ℝ) (h₁ : ∀ x, x ≠ 3 → ((a * (x - 3)^2) + 5 = -3 * (x - 3)^2 + 5))
  (h₂ : (a * (2 - 3)^2 + 5 = 2))
  : y = -3*x^2 + 18*x - 22 :=
by
  -- Conditions definitions
  have vertex_eq : ∀ x, y = a * (x - 3)^2 + 5 := h₁
  have through_point : 2 = a * (2 - 3)^2 + 5 := h₂
  
  -- Proof placeholder
  sorry

end parabola_equation_l447_447847


namespace fuel_consumption_l447_447236

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end fuel_consumption_l447_447236


namespace expected_value_shorter_gentlemen_l447_447027

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_gentlemen_l447_447027


namespace janet_earnings_per_hour_l447_447621

def rate_per_post := 0.25  -- Janet’s rate per post in dollars
def time_per_post := 10    -- Time to check one post in seconds
def seconds_per_hour := 3600  -- Seconds in one hour

theorem janet_earnings_per_hour :
  let posts_per_hour := seconds_per_hour / time_per_post
  let earnings_per_hour := rate_per_post * posts_per_hour
  earnings_per_hour = 90 := sorry

end janet_earnings_per_hour_l447_447621


namespace tangent_line_at_1_2_l447_447901

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then exp (-x - 1) - x else exp (x - 1) + x

theorem tangent_line_at_1_2 :
  f.2 (1 : ℝ) = 2 ∧ (y - 2) = 2 * (x - 1) := by
  sorry

end tangent_line_at_1_2_l447_447901


namespace integral_of_ratio_l447_447343

theorem integral_of_ratio :
  ∫ (x : ℝ) in Set.Ioo (-∞) ∞, 
  (2 * x^3 + 4 * x^2 + 2 * x + 2) / ((x^2 + x + 1) * (x^2 + x + 2)) 
  = -log (x^2 + x + 1) + (2 / sqrt 3) * arctan ((2 * x + 1) / sqrt 3) + 2 * log (x^2 + x + 2) + C := 
sorry

end integral_of_ratio_l447_447343


namespace geometric_progression_first_term_and_ratio_l447_447705

noncomputable def b1 := (224:ℚ) / 3
noncomputable def q := (1:ℚ) / 2

theorem geometric_progression_first_term_and_ratio :
  let b2 := (112:ℚ) / 3
  let b6 := (7:ℚ) / 3
  b2 = b1 * q ∧ b6 = b1 * q^5 :=
by {
  let b2 := (112:ℚ) / 3,
  let b6 := (7:ℚ) / 3,
  sorry
}

end geometric_progression_first_term_and_ratio_l447_447705


namespace prop_contrapositives_proposition_p_true_proposition_q_false_disjunction_impl_incorrect_conclusion_l447_447327

-- Conditions definitions
def p (x : ℝ) : Prop := x ∈ set.Icc 0 1 → exp x ≥ 1
def q : Prop := ∃ (x : ℝ), x^2 + x + 1 < 0
def converse (a b m : ℝ) : Prop := a * m^2 < b * m^2 → a < b

-- Given conditions
theorem prop_contrapositives (p q : Prop) : (p → q) ↔ (¬q → ¬p) := sorry

theorem proposition_p_true : ∀ x ∈ set.Icc (0 : ℝ) 1, exp x ≥ 1 := sorry

theorem proposition_q_false : ¬ ∃ x : ℝ, x^2 + x + 1 < 0 := sorry

theorem disjunction_impl (p q : Prop) : (¬(p ∨ q)) ↔ (¬p ∧ ¬q) := sorry

-- Mathematically equivalent proof problem
theorem incorrect_conclusion (a b m : ℝ) : 
  ¬ (∀ (a b m : ℝ), (a * m^2 < b * m^2) → (a < b) → (m = 0 → a * m^2 = b * m^2)) := 
sorry

end prop_contrapositives_proposition_p_true_proposition_q_false_disjunction_impl_incorrect_conclusion_l447_447327


namespace transformed_polynomial_roots_l447_447119

theorem transformed_polynomial_roots (p q r : ℝ) 
  (h₁ : p + q + r = 5) 
  (h₂ : p * q + q * r + r * p = 6) 
  (h₃ : p * q * r = 7) :
  ∃ (f : ℝ → ℝ), f(x) = x^3 - 10x^2 + 25x + 105 ∧ 
  (f(5 / (p - 1)) = 0) ∧ (f(5 / (q - 1)) = 0) ∧ (f(5 / (r - 1)) = 0) :=
sorry

end transformed_polynomial_roots_l447_447119


namespace two_patches_intersection_l447_447242

-- Definitions for conditions
axiom patches : Fin 5 → Set ℝ

-- Each patch has area at least 0.5
axiom patch_area : ∀ i, (∫ x in patches i, 1) ≥ 0.5

-- The total area of the coat is 1
noncomputable def coat_area : ℝ := 1

-- Overlapping area property
axiom overlapping_area : (Fin 5 → Prop) → Set ℝ

-- Statement of the proof problem
theorem two_patches_intersection :
  ∃ (i j : Fin 5), i ≠ j ∧ (∫ x in (patches i ∩ patches j), 1) ≥ 0.2 :=
sorry

end two_patches_intersection_l447_447242


namespace range_of_f_l447_447857

def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 4 - (Real.cos x) * (Real.sin x) + (Real.sin x) ^ 4 + Real.tan x

theorem range_of_f :
  ∀ y : ℝ, ∃ x : ℝ, f(x) = y :=
by
  sorry

end range_of_f_l447_447857


namespace find_g4_l447_447276

noncomputable def g : ℝ → ℝ := sorry

theorem find_g4 (h : ∀ x y : ℝ, x * g y = 2 * y * g x) (h₁ : g 10 = 5) : g 4 = 4 :=
sorry

end find_g4_l447_447276


namespace range_of_t_for_obtuse_triangle_l447_447893

def is_obtuse_triangle (a b c : ℝ) : Prop := ∃t : ℝ, a = t - 1 ∧ b = t + 1 ∧ c = t + 3

theorem range_of_t_for_obtuse_triangle :
  ∀ t : ℝ, is_obtuse_triangle (t-1) (t+1) (t+3) → (3 < t ∧ t < 7) :=
by
  intros t ht
  sorry

end range_of_t_for_obtuse_triangle_l447_447893


namespace domain_of_f_l447_447942

noncomputable def f (x : ℝ) := (3 * x) / (x - 4) + real.sqrt (x + 2)

theorem domain_of_f : 
  { x : ℝ | (x - 4 ≠ 0) ∧ (x + 2 ≥ 0) } = { x : ℝ | x ∈ [-2, 4) ∪ (4, ∞) } :=
by
  sorry

end domain_of_f_l447_447942


namespace rotated_vector_eq_l447_447922

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

def vector_transformed_by_rotation (v : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  let (x, y) := v
  let m := rotation_matrix θ
  (m 0 0 * x + m 0 1 * y, m 1 0 * x + m 1 1 * y)

theorem rotated_vector_eq :
  vector_transformed_by_rotation (1, 1) (Real.pi / 3) = ((1 - Real.sqrt 3) / 2, (1 + Real.sqrt 3) / 2) :=
by
  sorry

end rotated_vector_eq_l447_447922


namespace planning_committee_l447_447786

theorem planning_committee {x : ℕ} 
  (h1 : (x * (x - 1)) / 2 = 10) : (x.choose 3) = 10 :=
begin
  -- We solve the equation x * (x - 1) / 2 = 10
  -- and we get x = 5, then we compute 5.choose 3 = 10
  sorry
end

end planning_committee_l447_447786


namespace intersect_product_length_l447_447139

noncomputable def line_parametric := {x : ℝ, y : ℝ, t : ℝ // x = -1 + (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t}

noncomputable def curve_polar (ρ : ℝ) (θ : ℝ) := ρ * Real.cos θ * Real.cos θ - Real.sin θ = 0

noncomputable def point_M : ℝ × ℝ := (-1, 0)

theorem intersect_product_length :
  ∀ (x y t : ℝ),
  (x = -1 + (Real.sqrt 2 / 2) * t) ∧ (y = (Real.sqrt 2 / 2 * t)) → 
  ∀ ρ θ : ℝ, 
  ρ * Real.cos θ * Real.cos θ - Real.sin θ = 0 → 
  (∀ x y : ℝ, curve_polar (Real.sqrt (x^2 + y^2)) (Real.arctan2 y x) → 
  y = x^2) →
  |(-1 + (Real.sqrt 2 / 2) * t + 1) * (Real.arctan2 ((Real.sqrt 2 / 2 * t)) (-1 + (Real.sqrt 2 / 2) * t - (-1)))| = 2 :=
by
  sorry

end intersect_product_length_l447_447139


namespace original_polygon_sides_l447_447163

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
(n - 2) * 180

theorem original_polygon_sides (x : ℕ) (h1 : sum_of_interior_angles (2 * x) = 2160) : x = 7 :=
by
  sorry

end original_polygon_sides_l447_447163


namespace gift_packaging_combinations_l447_447358

def wrapping_paper_choices : ℕ := 10
def ribbon_choices : ℕ := 4
def gift_card_choices : ℕ := 5
def bow_choices : ℕ := 2

theorem gift_packaging_combinations :
  wrapping_paper_choices * ribbon_choices * gift_card_choices * bow_choices = 400 :=
by
  have h1 := wrapping_paper_choices
  have h2 := ribbon_choices
  have h3 := gift_card_choices
  have h4 := bow_choices
  calc
    h1 * h2 * h3 * h4 = 10 * 4 * 5 * 2 := by sorry
    ... = 400 := by sorry

end gift_packaging_combinations_l447_447358


namespace factory_workers_count_l447_447959

theorem factory_workers_count :
  ∃ (F S_f : ℝ), 
    (F * S_f = 30000) ∧ 
    (30 * (S_f + 500) = 75000) → 
    (F = 15) :=
by
  sorry

end factory_workers_count_l447_447959


namespace cube_vertex_shapes_l447_447352

theorem cube_vertex_shapes (vertices : Set (Fin 8)) (h : vertices.card = 4) :
  vertices = {1, 2, 3, 4} ∨
  vertices = {1, 2, 5, 6} ∨
  vertices = {1, 3, 5, 7} ∨
  vertices = {1, 4, 5, 8} :=
sorry

end cube_vertex_shapes_l447_447352


namespace find_n_from_lcm_gcf_l447_447659

open scoped Classical

noncomputable def LCM (a b : ℕ) : ℕ := sorry
noncomputable def GCF (a b : ℕ) : ℕ := sorry

theorem find_n_from_lcm_gcf (n m : ℕ) (h1 : LCM n m = 48) (h2 : GCF n m = 18) (h3 : m = 16) : n = 54 :=
by sorry

end find_n_from_lcm_gcf_l447_447659


namespace sum_of_fractions_l447_447526

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447526


namespace solution_set_f_x_minus_2_pos_l447_447216

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then 2 * x - 4 else 2 * (-x) - 4

theorem solution_set_f_x_minus_2_pos :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_x_minus_2_pos_l447_447216


namespace equilateral_triangle_l447_447894

noncomputable def angles_arithmetic_seq (A B C : ℝ) : Prop := B - A = C - B

noncomputable def sides_geometric_seq (a b c : ℝ) : Prop := b / a = c / b

theorem equilateral_triangle 
  (A B C a b c : ℝ) 
  (h_angles : angles_arithmetic_seq A B C) 
  (h_sides : sides_geometric_seq a b c) 
  (h_triangle : A + B + C = π) 
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (A = B ∧ B = C) ∧ (a = b ∧ b = c) :=
sorry

end equilateral_triangle_l447_447894


namespace math_problem_l447_447532

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447532


namespace least_element_of_set_six_non_multiple_l447_447641

theorem least_element_of_set_six_non_multiple : 
  ∀ (S : Set ℕ), 
    (∀ a b ∈ S, a < b → ¬ (b % a = 0)) ∧
    S ⊆ {n | 1 ≤ n ∧ n ≤ 12} ∧ 
    S.card = 6 → 
    ∃ k ∈ S, ∀ x ∈ S, k ≤ x ∧ k = 4 := 
by 
  sorry

end least_element_of_set_six_non_multiple_l447_447641


namespace checkerboard_square_count_l447_447406

def checkerboard_10_10 := matrix (fin 10) (fin 10) bool

noncomputable def count_squares_with_min_6_black (b : checkerboard_10_10) : ℕ :=
  let black : bool := ff in
  let is_black (r c : ℕ) : bool := b (fin.mk r (by linarith)) (fin.mk c (by linarith)) = black in
  let count_black (r c size : ℕ) : ℕ :=
    (finset.range size).sum (λ i, (finset.range size).sum (λ j, if is_black (r + i) (c + j) then 1 else 0)) in
  (finset.range (11 - size)).sum (λ r, (finset.range (11 - size)).sum (λ c, if count_black r c size ≥ 6 then 1 else 0))

theorem checkerboard_square_count :
  ∀ b : checkerboard_10_10, count_squares_with_min_6_black b = 155 := sorry

end checkerboard_square_count_l447_447406


namespace BDD1H_is_Spatial_in_Cube_l447_447189

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A B C D A1 B1 C1 D1 : Point3D)
(midpoint_B1C1 : Point3D)
(middle_B1C1 : midpoint_B1C1 = ⟨(B1.x + C1.x) / 2, (B1.y + C1.y) / 2, (B1.z + C1.z) / 2⟩)

def is_not_planar (a b c d : Point3D) : Prop :=
¬ ∃ α β γ δ : ℝ, α * a.x + β * a.y + γ * a.z + δ = 0 ∧ 
                α * b.x + β * b.y + γ * b.z + δ = 0 ∧ 
                α * c.x + β * c.y + γ * c.z + δ = 0 ∧ 
                α * d.x + β * d.y + γ * d.z + δ = 0

def BDD1H_is_spatial (cube : Cube) : Prop :=
is_not_planar cube.B cube.D cube.D1 cube.midpoint_B1C1

theorem BDD1H_is_Spatial_in_Cube (cube : Cube) : BDD1H_is_spatial cube :=
sorry

end BDD1H_is_Spatial_in_Cube_l447_447189


namespace translate_line_down_by_4_l447_447312

-- Define the original equation of the line
def original_line (x : ℝ) : ℝ := -2 * x + 3

-- Define the function that represents translating the y-intercept of a line downward by 4 units
def translate_down (f : ℝ → ℝ) (units : ℝ) (x : ℝ) : ℝ :=
  f(x) - units

-- Theorem to prove that translating down the line y = -2x + 3 by 4 units results in y = -2x - 1
theorem translate_line_down_by_4 : 
  ∀ (x : ℝ), translate_down original_line 4 x = -2 * x - 1 :=
by
  intro x
  sorry

end translate_line_down_by_4_l447_447312


namespace sum_of_fractions_l447_447525

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l447_447525


namespace math_problem_l447_447531

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l447_447531


namespace solve_equation_l447_447259

theorem solve_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) :
  (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 :=
sorry

end solve_equation_l447_447259


namespace range_of_log_cbrt_sin_l447_447320

noncomputable def function_range : Set ℝ :=
  {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ real.pi ∧ y = real.logb 4 (real.cbrt (real.sin x))}

theorem range_of_log_cbrt_sin : function_range = { y | y ≤ 0 } :=
  sorry

end range_of_log_cbrt_sin_l447_447320


namespace expected_value_of_shorter_gentlemen_correct_l447_447032
noncomputable def expected_value_of_shorter_gentlemen (n : ℕ) : ℝ :=
  ∑ j in Finset.range n, (j : ℝ) / n

theorem expected_value_of_shorter_gentlemen_correct (n : ℕ) :
  expected_value_of_shorter_gentlemen (n + 1) = n / 2 :=
by
  sorry

end expected_value_of_shorter_gentlemen_correct_l447_447032


namespace cost_of_each_candy_bar_is_3_l447_447823

def Dan_has_2_dollars : ℕ := 2
def total_cost_of_candy_bars : ℕ := 6
def number_of_candy_bars : ℕ := 2

theorem cost_of_each_candy_bar_is_3 :
  number_of_candy_bars = 2 → 
  total_cost_of_candy_bars = 6 → 
  total_cost_of_candy_bars / number_of_candy_bars = 3 :=
by
  intro h1 h2
  simp [h1, h2]
  sorry

end cost_of_each_candy_bar_is_3_l447_447823


namespace max_digits_l447_447392

-- Definition of terms
def count_zeros (a : ℕ → ℕ) (start len : ℕ) : ℕ :=
  (finset.range len).filter (λ i, a (start + i) = 0).card

def count_ones (a : ℕ → ℕ) (start len : ℕ) : ℕ :=
  (finset.range len).filter (λ i, a (start + i) = 1).card

-- Conditions as lean definitions
def condition1 (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, count_zeros a k 200 = count_ones a k 200

def condition2 (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, count_zeros a k 202 ≠ count_ones a k 202

-- Theorem statement
theorem max_digits (a : ℕ → ℕ) (n : ℕ) 
  (h1 : condition1 a) (h2 : condition2 a) : n ≤ 300 :=
  sorry

end max_digits_l447_447392


namespace binary_addition_l447_447384

theorem binary_addition :
  let b1 := 0b1101;
      b2 := 0b1010;
      b3 := 0b0111;
      b4 := 0b0101
  in b1 + b2 + b3 + b4 = 0b100011 :=
by
  sorry

end binary_addition_l447_447384


namespace infinite_quadruples_of_prime_divisors_l447_447434

def greatest_prime_divisor_of_square_plus_one (n : ℕ) : ℕ := 
  sorry -- This definition needs to handle the actual calculation of the greatest prime divisor of (n^2 + 1)

theorem infinite_quadruples_of_prime_divisors :
  ∃ (f : ℕ → ℕ), (∀ n, f n > 0) ∧ (∀ n, greatest_prime_divisor_of_square_plus_one (f n) = greatest_prime_divisor_of_square_plus_one (f (n + 1))) ∧
  ∀ k : ℕ, ∃ a b c d : ℕ, a < b < c < d ∧ greatest_prime_divisor_of_square_plus_one a = greatest_prime_divisor_of_square_plus_one b ∧ 
                                           greatest_prime_divisor_of_square_plus_one b = greatest_prime_divisor_of_square_plus_one c ∧ 
                                           greatest_prime_divisor_of_square_plus_one c = greatest_prime_divisor_of_square_plus_one d :=
sorry

end infinite_quadruples_of_prime_divisors_l447_447434


namespace problem_statement_l447_447511

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l447_447511


namespace exists_bisecting_line_l447_447452

theorem exists_bisecting_line {P : Point} {pentagon : Polygon} 
  (h_convex : pentagon.isConvex) (h_on_boundary : P ∈ pentagon.boundary)
  : ∃ (Q : Point), (Q ∈ pentagon.boundary) ∧ (line_through P Q).dividesAreaIntoTwoEqualParts :=
sorry

end exists_bisecting_line_l447_447452


namespace sin_A_value_l447_447173

variables {A B C a b c : ℝ}
variables {sin cos : ℝ → ℝ}

-- Conditions
axiom triangle_sides : ∀ (A B C: ℝ), ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0
axiom sin_cos_conditions : 3 * b * sin A = c * cos A + a * cos C

-- Proof statement
theorem sin_A_value (h : 3 * b * sin A = c * cos A + a * cos C) : sin A = 1 / 3 :=
by 
  sorry

end sin_A_value_l447_447173


namespace sequence_general_formula_sequence_sum_formula_l447_447884

section problem1

-- Given conditions
variables {a : ℕ → ℕ} {S : ℕ → ℕ}

def Sn_arithmetic (Sn : ℕ → ℕ) : Prop :=
  ∀ n, (Sn n) / n = (Sn 1) / 1 + (n - 1)

def general_formula : Prop :=
  ∀ n, a n = 2 * n - 1

-- Prove that given the conditions, the general formula holds
theorem sequence_general_formula (h_arith : Sn_arithmetic S)
  (h_a2 : a 2 = 3) (h_a3 : a 3 = 5) :
  general_formula :=
sorry

end problem1

section problem2

variables {b : ℕ → ℕ}

def bn_formula (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, a n * 3^n

def Tn_sum (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = (3 / 2).to_rat + (n - 1) * 3^n

theorem sequence_sum_formula (a_n : ℕ → ℕ) (h_gen : ∀ n, a n = 2 * n - 1) :
  ∀ T : ℕ → ℕ, (∀ n, T n = ∑ k in finset.range n, bn_formula a k) →
  Tn_sum T :=
sorry

end problem2

end sequence_general_formula_sequence_sum_formula_l447_447884


namespace third_angle_of_triangle_l447_447958

theorem third_angle_of_triangle (a b : ℝ) (ha : a = 50) (hb : b = 60) : 
  ∃ (c : ℝ), a + b + c = 180 ∧ c = 70 :=
by
  sorry

end third_angle_of_triangle_l447_447958


namespace arithmetic_general_formula_sum_T_10_l447_447459

-- Definitions based on conditions
def arithmetic_seq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + d * (↑n - 1)
def sum_seq (S : ℕ → ℤ) (a : ℕ → ℤ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2
def geometric_seq (b : ℕ → ℤ) := ∃ r : ℤ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given conditions
variable {a : ℕ → ℤ} {S : ℕ → ℤ} {b : ℕ → ℤ}
axiom h_arith_seq : arithmetic_seq a
axiom h_sum_3_5 : S 3 + S 5 = 58
axiom h_geo_a1_a3_a7 : (a 3)^2 = a 1 * (a 7)
axiom h_geom_seq : geometric_seq b
axiom h_b_cond : b 5 * b 6 + b 4 * b 7 = a 8
def log3 (x : ℤ) := log x / log 3
def T (n : ℕ) := ∑ i in (finset.range n).image (λ i, i + 1), log3 (b i)

-- Proof goals
theorem arithmetic_general_formula : ∃ a_1 d : ℤ, ∀ n : ℕ, a n = 2 * ↑n + 2 :=
by sorry

theorem sum_T_10 : T 10 = 10 :=
by sorry

end arithmetic_general_formula_sum_T_10_l447_447459


namespace solution_of_four_real_numbers_l447_447842

theorem solution_of_four_real_numbers (x y z t : ℝ)
  (h1 : x + y * z * t = 2)
  (h2 : y + x * z * t = 2)
  (h3 : z + x * y * t = 2)
  (h4 : t + x * y * z = 2) :
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = 3) ∨
  (x = -1 ∧ y = -1 ∧ z = 3 ∧ t = -1) ∨
  (x = -1 ∧ y = 3 ∧ z = -1 ∧ t = -1) ∨
  (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = -1) :=
sorry

end solution_of_four_real_numbers_l447_447842


namespace tenth_term_is_19_over_4_l447_447091

def nth_term_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

theorem tenth_term_is_19_over_4 :
  nth_term_arithmetic_sequence (1/4) (1/2) 10 = 19/4 :=
by
  sorry

end tenth_term_is_19_over_4_l447_447091


namespace incorrect_option_C_l447_447197

theorem incorrect_option_C (A B C : ℝ) (a b c : ℝ)
  (h_triangle : ∀ (x y : ℝ), x + y > 0 ∧ x = y -> False)
  (h1 : a^2 + c^2 - b^2 > 0)
  (h2 : A > B)
  (h3 : sin (2 * A) = sin (2 * B))
  (h4 : b = 3)
  (h5 : a = 4)
  (h6 : B = π / 6) : 
  ¬ (∃ (k : ℤ), A = B + k * π ∨ 2 * A + 2 * B = π + 2 * k * π) :=
sorry

end incorrect_option_C_l447_447197


namespace find_k_min_value_find_k_range_l447_447491

def f (x : ℝ) (k : ℝ) := (x^2 + k * x + 1) / (x^2 + 1)

theorem find_k_min_value {k : ℝ} (h : ∀ x : ℝ, x > 0 → f x k ≥ -1) : k = -4 := 
sorry

theorem find_k_range {k : ℝ} 
  (h : ∀ (x1 x2 x3 : ℝ), (0 < x1 ∧ 0 < x2 ∧ 0 < x3) → 
  (let a := f x1 k in let b := f x2 k in let c := f x3 k in 
   a + b > c ∧ a + c > b ∧ b + c > a)) : 
  -1 ≤ k ∧ k ≤ 2 := 
sorry

end find_k_min_value_find_k_range_l447_447491


namespace largest_of_four_numbers_l447_447389

theorem largest_of_four_numbers (a b c d : ℝ) (h1 : a = -3) (h2 : b = -1) (h3 : c = real.pi) (h4 : d = 4) :
  ∀ x ∈ {a, b, c, d}, x ≤ d := 
by 
  sorry

end largest_of_four_numbers_l447_447389


namespace solution_set_l447_447902

noncomputable def f (x : ℝ) : ℝ := sorry
def f'' (x : ℝ) : ℝ := sorry

theorem solution_set (h_domain : ∀ x, 0 < x → (0 < f(x) ∧ 0 < f''(x)))
                     (h_condition : ∀ x : ℝ, 0 < x → (f(x) + x * f''(x) > 0)) :
  {x : ℝ | 0 < x ∧ (x-1)*f(x^2-1) < f(x+1)} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_l447_447902


namespace square_of_chord_length_l447_447739

/--
Given two circles with radii 10 and 7, and centers 15 units apart, if they intersect at a point P such that the chords QP and PR are of equal lengths, then the square of the length of chord QP is 289.
-/
theorem square_of_chord_length :
  ∀ (r1 r2 d x : ℝ), r1 = 10 → r2 = 7 → d = 15 →
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  cos_theta1 = cos_theta2 →
  x^2 = 289 := 
by
  intros r1 r2 d x h1 h2 h3
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  intro h4
  sorry

end square_of_chord_length_l447_447739


namespace sin_angle_BAG_l447_447636

/-- Given the vertices of a cube, prove that the sine of the angle between
the vectors BA and GA is sqrt(2/3). -/
theorem sin_angle_BAG (A B G : ℝ × ℝ × ℝ) (hA : A = (0,0,0)) (hB : B = (1,0,0)) (hG : G = (1,1,1)) :
  Real.sin (Real.angle (B - A) (G - A)) = Real.sqrt (2/3) :=
by
  -- Explicitly provide the vectors
  let BA := (A.1 - B.1, A.2 - B.2, A.3 - B.3)
  let GA := (A.1 - G.1, A.2 - G.2, A.3 - G.3)

  -- Compute the dot product of BA and GA
  have dot_product : (BA.1 * GA.1 + BA.2 * GA.2 + BA.3 * GA.3) = 1 := sorry

  -- Compute the magnitude of BA and GA
  have mag_BA : Real.sqrt (BA.1^2 + BA.2^2 + BA.3^2) = 1 := sorry
  have mag_GA : Real.sqrt (GA.1^2 + GA.2^2 + GA.3^2) = Real.sqrt 3 := sorry

  -- Compute the cosine of the angle
  have cos_theta : Real.cos (Real.angle (B - A) (G - A)) = 1 / Real.sqrt 3 := sorry

  -- Use the identity sin^2(theta) + cos^2(theta) = 1 to find sin(theta)
  show Real.sin (Real.angle (B - A) (G - A)) = Real.sqrt (1 - (cos_theta)^2) := sorry

  -- Thus, prove the required result
  show Real.sin (Real.angle (B - A) (G - A)) = Real.sqrt (2 / 3) := sorry

end sin_angle_BAG_l447_447636


namespace part1_positive_root_part2_negative_solution_l447_447126

theorem part1_positive_root (x k : ℝ) (hx1 : x > 0)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k = 6 ∨ k = -8 := 
sorry

theorem part2_negative_solution (x k : ℝ) (hx2 : x < 0)
  (hx_ne1 : x ≠ 1) (hx_ne_neg1 : x ≠ -1)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k < -1 ∧ k ≠ -8 := 
sorry

end part1_positive_root_part2_negative_solution_l447_447126


namespace find_remainder_l447_447830

-- We state that p(x) is a polynomial and r(x) is the remainder when divided by (x-2)(x-5).
def polynomial_remainder (p r : ℚ[X]) : Prop :=
  ∃ q : ℚ[X], p = q * ((X - 2) * (X - 5)) + r

-- Given the conditions of the problem
def conditions (p : ℚ[X]) : Prop :=
  p.eval 2 = 7 ∧ p.eval 5 = 8 ∧ p.eval 0 = 6

-- Define r(x) as per the solution
noncomputable def r : ℚ[X] := (1/3:X) * X + (19/3:X)

-- The explicit statement proving the remainder is as specified
theorem find_remainder (p : ℚ[X]) (h : conditions p) : polynomial_remainder p r :=
  sorry

end find_remainder_l447_447830


namespace part1_part2_part3_l447_447888

variables {x y : ℝ}
def A : ℝ := 2 * x ^ 2 + 3 * x * y + 2 * y
def B : ℝ := x ^ 2 - x * y + x

theorem part1 : A - 2 * B = 5 * x * y - 2 * x + 2 * y := by
  sorry

theorem part2 (hx : x = -1) (hy : y = 3) : A - 2 * B = -7 := by
  sorry

theorem part3 (h : ∀ x, A - 2 * B = 5 * x * y - 2 * x + 2 * y) : y = 2 / 5 := by
  sorry

end part1_part2_part3_l447_447888


namespace max_value_fraction_sum_l447_447946

theorem max_value_fraction_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h_sum : a + b = 1) :
  (a / (a + 1) + b / (b + 1)) ≤ 2 / 3 :=
begin
  sorry
end

end max_value_fraction_sum_l447_447946
