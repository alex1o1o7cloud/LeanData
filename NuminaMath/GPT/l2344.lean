import Mathlib

namespace number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l2344_234402

theorem number_of_sixth_graders_who_bought_more_pens_than_seventh_graders 
  (p : ℕ) (h1 : 178 % p = 0) (h2 : 252 % p = 0) :
  (252 / p) - (178 / p) = 5 :=
sorry

end number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l2344_234402


namespace proof_problem_l2344_234486

variables (a b c : Line) (alpha beta gamma : Plane)

-- Define perpendicular relationship between line and plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relationship between lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Main theorem statement
theorem proof_problem 
  (h1 : perp_line_plane a alpha) 
  (h2 : perp_line_plane b beta) 
  (h3 : parallel_planes alpha beta) : 
  parallel_lines a b :=
sorry

end proof_problem_l2344_234486


namespace line_equation_l2344_234455

theorem line_equation (m : ℝ) (x1 y1 : ℝ) (b : ℝ) :
  m = -3 → x1 = -2 → y1 = 0 → 
  (∀ x y, y - y1 = m * (x - x1) ↔ 3 * x + y + 6 = 0) :=
sorry

end line_equation_l2344_234455


namespace move_point_A_l2344_234464

theorem move_point_A :
  let A := (-5, 6)
  let A_right := (A.1 + 5, A.2)
  let A_upwards := (A_right.1, A_right.2 + 6)
  A_upwards = (0, 12) := by
  sorry

end move_point_A_l2344_234464


namespace sum_of_geometric_sequence_l2344_234400

theorem sum_of_geometric_sequence :
  ∀ (a : ℕ → ℝ) (r : ℝ),
  (∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ,
   a 1 = a_1 ∧ a 2 = a_2 ∧ a 3 = a_3 ∧ a 4 = a_4 ∧ a 5 = a_5 ∧ a 6 = a_6 ∧ a 7 = a_7 ∧ a 8 = a_8 ∧ a 9 = a_9 ∧
   a_1 * r^1 = a_2 ∧ a_1 * r^2 = a_3 ∧ a_1 * r^3 = a_4 ∧ a_1 * r^4 = a_5 ∧ a_1 * r^5 = a_6 ∧ a_1 * r^6 = a_7 ∧ a_1 * r^7 = a_8 ∧ a_1 * r^8 = a_9 ∧
   a_1 + a_2 + a_3 = 8 ∧
   a_4 + a_5 + a_6 = -4) →
  a 7 + a 8 + a 9 = 2 :=
sorry

end sum_of_geometric_sequence_l2344_234400


namespace sufficient_but_not_necessary_condition_l2344_234452

def p (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def q (x a : ℝ) : Prop := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬ p x) ↔ a ≤ -1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l2344_234452


namespace find_m_l2344_234448

def vec (α : Type*) := (α × α)
def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) :
  let a : vec ℝ := (1, 3)
  let b : vec ℝ := (-2, m)
  let c : vec ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  dot_product a c = 0 → m = -1 :=
by
  sorry

end find_m_l2344_234448


namespace valid_three_digit_card_numbers_count_l2344_234430

def card_numbers : List (ℕ × ℕ) := [(0, 1), (2, 3), (4, 5), (7, 8)]

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 -- Ensures it's three digits

def three_digit_numbers : List ℕ := 
  [201, 210, 102, 120, 301, 310, 103, 130, 401, 410, 104, 140,
   501, 510, 105, 150, 601, 610, 106, 160, 701, 710, 107, 170,
   801, 810, 108, 180, 213, 231, 312, 321, 413, 431, 512, 521,
   613, 631, 714, 741, 813, 831, 214, 241, 315, 351, 415, 451,
   514, 541, 615, 651, 716, 761, 815, 851, 217, 271, 317, 371,
   417, 471, 517, 571, 617, 671, 717, 771, 817, 871, 217, 271,
   321, 371, 421, 471, 521, 571, 621, 671, 721, 771, 821, 871]

def count_valid_three_digit_numbers : ℕ :=
  three_digit_numbers.length

theorem valid_three_digit_card_numbers_count :
    count_valid_three_digit_numbers = 168 :=
by
  -- proof goes here
  sorry

end valid_three_digit_card_numbers_count_l2344_234430


namespace streamers_for_price_of_confetti_l2344_234414

variable (p q : ℝ) (x y : ℝ)

theorem streamers_for_price_of_confetti (h1 : x * (1 + p / 100) = y) 
                                   (h2 : y * (1 - q / 100) = x)
                                   (h3 : |p - q| = 90) :
  10 * (y * 0.4) = 4 * y :=
sorry

end streamers_for_price_of_confetti_l2344_234414


namespace arithmetic_seq_geom_seq_l2344_234468

theorem arithmetic_seq_geom_seq {a : ℕ → ℝ} 
  (h1 : ∀ n, 0 < a n)
  (h2 : a 2 + a 3 + a 4 = 15)
  (h3 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2) :
  a 10 = 19 :=
sorry

end arithmetic_seq_geom_seq_l2344_234468


namespace some_employees_not_managers_l2344_234431

-- Definitions of the conditions
def isEmployee : Type := sorry
def isManager : isEmployee → Prop := sorry
def isShareholder : isEmployee → Prop := sorry
def isPunctual : isEmployee → Prop := sorry

-- Given conditions
axiom some_employees_not_punctual : ∃ e : isEmployee, ¬isPunctual e
axiom all_managers_punctual : ∀ m : isEmployee, isManager m → isPunctual m
axiom some_managers_shareholders : ∃ m : isEmployee, isManager m ∧ isShareholder m

-- The statement to be proved
theorem some_employees_not_managers : ∃ e : isEmployee, ¬isManager e :=
by sorry

end some_employees_not_managers_l2344_234431


namespace tied_in_runs_l2344_234450

def aaron_runs : List ℕ := [4, 8, 15, 7, 4, 12, 11, 5]
def bonds_runs : List ℕ := [3, 5, 18, 9, 12, 14, 9, 0]

def total_runs (runs : List ℕ) : ℕ := runs.foldl (· + ·) 0

theorem tied_in_runs : total_runs aaron_runs = total_runs bonds_runs := by
  sorry

end tied_in_runs_l2344_234450


namespace correct_choice_l2344_234418

theorem correct_choice
  (options : List String)
  (correct : String)
  (is_correct : correct = "that") :
  "The English spoken in the United States is only slightly different from ____ spoken in England." = 
  "The English spoken in the United States is only slightly different from that spoken in England." :=
by
  sorry

end correct_choice_l2344_234418


namespace avg_price_of_returned_tshirts_l2344_234492

-- Define the conditions as Lean definitions
def avg_price_50_tshirts := 750
def num_tshirts := 50
def num_returned_tshirts := 7
def avg_price_remaining_43_tshirts := 720

-- The correct price of the 7 returned T-shirts
def correct_avg_price_returned := 6540 / 7

-- The proof statement
theorem avg_price_of_returned_tshirts :
  (num_tshirts * avg_price_50_tshirts - (num_tshirts - num_returned_tshirts) * avg_price_remaining_43_tshirts) / num_returned_tshirts = correct_avg_price_returned :=
by
  sorry

end avg_price_of_returned_tshirts_l2344_234492


namespace parameter_for_three_distinct_solutions_l2344_234410

open Polynomial

theorem parameter_for_three_distinct_solutions (a : ℝ) :
  (∀ x : ℝ, x^4 - 40 * x^2 + 144 = a * (x^2 + 4 * x - 12)) →
  (∀ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  (x1^4 - 40 * x1^2 + 144 = a * (x1^2 + 4 * x1 - 12) ∧ 
   x2^4 - 40 * x2^2 + 144 = a * (x2^2 + 4 * x2 - 12) ∧ 
   x3^4 - 40 * x3^2 + 144 = a * (x3^2 + 4 * x3 - 12) ∧
   x4^4 - 40 * x4^2 + 144 = a * (x4^2 + 4 * x4 - 12))) → a = 48 :=
by
  sorry

end parameter_for_three_distinct_solutions_l2344_234410


namespace systematic_sampling_draw_l2344_234440

theorem systematic_sampling_draw
  (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 8)
  (h2 : 160 ≥ 8 * 20)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 → 
    160 ≥ ((k - 1) * 8 + 1 + 7))
  (h4 : ∀ y : ℕ, y = 1 + (15 * 8) → y = 126)
: x = 6 := 
sorry

end systematic_sampling_draw_l2344_234440


namespace sin_150_eq_half_l2344_234413

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l2344_234413


namespace cost_per_acre_proof_l2344_234497

def cost_of_land (tac tl : ℕ) (hc hcc hcp heq : ℕ) (ttl : ℕ) : ℕ := ttl - (hc + hcc + hcp + heq)

def cost_per_acre (total_land : ℕ) (cost_land : ℕ) : ℕ := cost_land / total_land

theorem cost_per_acre_proof (tac tl hc hcc hcp heq ttl epl : ℕ) 
  (h1 : tac = 30)
  (h2 : hc = 120000)
  (h3 : hcc = 20 * 1000)
  (h4 : hcp = 100 * 5)
  (h5 : heq = 6 * 100 + 6000)
  (h6 : ttl = 147700) :
  cost_per_acre tac (cost_of_land tac tl hc hcc hcp heq ttl) = epl := by
  sorry

end cost_per_acre_proof_l2344_234497


namespace circle_equation_exists_l2344_234477

-- Define the necessary conditions
def tangent_to_x_axis (r b : ℝ) : Prop :=
  r^2 = b^2

def center_on_line (a b : ℝ) : Prop :=
  3 * a - b = 0

def intersects_formula (a b r : ℝ) : Prop :=
  2 * r^2 = (a - b)^2 + 14

-- Main theorem combining the conditions and proving the circles' equations
theorem circle_equation_exists (a b r : ℝ) :
  tangent_to_x_axis r b →
  center_on_line a b →
  intersects_formula a b r →
  ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x + 1)^2 + (y + 3)^2 = 9) :=
by
  intros h_tangent h_center h_intersects
  sorry

end circle_equation_exists_l2344_234477


namespace geom_seq_thm_l2344_234406

noncomputable def geom_seq (a : ℕ → ℝ) :=
  a 1 = 2 ∧ (a 2 * a 4 = a 6)

noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) :=
  1 / (Real.logb 2 (a (2 * n - 1)) * Real.logb 2 (a (2 * n + 1)))

noncomputable def sn_sum (b : ℕ → ℝ) (n : ℕ) :=
  (Finset.range (n + 1)).sum b

theorem geom_seq_thm (a : ℕ → ℝ) (n : ℕ) (b : ℕ → ℝ) :
  geom_seq a →
  ∀ n, a n = 2 ^ n ∧ sn_sum (b_seq a) n = n / (2 * n + 1) :=
by
  sorry

end geom_seq_thm_l2344_234406


namespace exponent_zero_value_of_neg_3_raised_to_zero_l2344_234470

theorem exponent_zero (x : ℤ) (hx : x ≠ 0) : x ^ 0 = 1 :=
by
  -- Proof goes here
  sorry

theorem value_of_neg_3_raised_to_zero : (-3 : ℤ) ^ 0 = 1 :=
by
  exact exponent_zero (-3) (by norm_num)

end exponent_zero_value_of_neg_3_raised_to_zero_l2344_234470


namespace average_speed_over_ride_l2344_234461

theorem average_speed_over_ride :
  let speed1 := 12 -- speed in km/h
  let time1 := 5 / 60 -- time in hours
  
  let speed2 := 15 -- speed in km/h
  let time2 := 10 / 60 -- time in hours
  
  let speed3 := 18 -- speed in km/h
  let time3 := 15 / 60 -- time in hours
  
  let distance1 := speed1 * time1 -- distance for the first segment
  let distance2 := speed2 * time2 -- distance for the second segment
  let distance3 := speed3 * time3 -- distance for the third segment
  
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  let avg_speed := total_distance / total_time
  
  avg_speed = 16 :=
by
  sorry

end average_speed_over_ride_l2344_234461


namespace gcd_polynomials_l2344_234436

theorem gcd_polynomials (b : ℤ) (h: ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 :=
sorry

end gcd_polynomials_l2344_234436


namespace geometric_sequence_problem_l2344_234460

-- Assume {a_n} is a geometric sequence with positive terms
variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Condition: all terms are positive numbers in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 0 * r ^ n

-- Condition: a_1 * a_9 = 16
def condition1 (a : ℕ → ℝ) : Prop :=
  a 1 * a 9 = 16

-- Question to prove: a_2 * a_5 * a_8 = 64
theorem geometric_sequence_problem
  (h_geom : is_geometric_sequence a r)
  (h_pos : ∀ n, 0 < a n)
  (h_cond1 : condition1 a) :
  a 2 * a 5 * a 8 = 64 :=
by
  sorry

end geometric_sequence_problem_l2344_234460


namespace find_values_of_p_l2344_234483

def geometric_progression (p : ℝ) : Prop :=
  (2 * p)^2 = (4 * p + 5) * |p - 3|

theorem find_values_of_p :
  {p : ℝ | geometric_progression p} = {-1, 15 / 8} :=
by
  sorry

end find_values_of_p_l2344_234483


namespace count_words_200_l2344_234487

theorem count_words_200 : 
  let single_word_numbers := 29
  let compound_words_21_to_99 := 144
  let compound_words_100_to_199 := 54 + 216
  single_word_numbers + compound_words_21_to_99 + compound_words_100_to_199 = 443 :=
by
  sorry

end count_words_200_l2344_234487


namespace find_b_l2344_234471

-- Define the function f(x)
def f (x : ℝ) : ℝ := 5 * x - 7

-- State the theorem
theorem find_b (b : ℝ) : f b = 0 ↔ b = 7 / 5 := by
  sorry

end find_b_l2344_234471


namespace extreme_values_of_f_range_of_a_for_intersection_l2344_234472

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 15 * x + a

theorem extreme_values_of_f :
  f (-1) = 5 ∧ f 3 = -27 :=
by {
  sorry
}

theorem range_of_a_for_intersection (a : ℝ) : 
  (-80 < a) ∧ (a < 28) ↔ ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ a ∧ f x₂ = g x₂ a ∧ f x₃ = g x₃ a :=
by {
  sorry
}

end extreme_values_of_f_range_of_a_for_intersection_l2344_234472


namespace mrs_mcpherson_percentage_l2344_234438

def total_rent : ℕ := 1200
def mr_mcpherson_amount : ℕ := 840
def mrs_mcpherson_amount : ℕ := total_rent - mr_mcpherson_amount

theorem mrs_mcpherson_percentage : (mrs_mcpherson_amount.toFloat / total_rent.toFloat) * 100 = 30 :=
by
  sorry

end mrs_mcpherson_percentage_l2344_234438


namespace probability_of_both_contracts_l2344_234434

open Classical

variable (P_A P_B' P_A_or_B P_A_and_B : ℚ)

noncomputable def probability_hardware_contract := P_A = 3 / 4
noncomputable def probability_not_software_contract := P_B' = 5 / 9
noncomputable def probability_either_contract := P_A_or_B = 4 / 5
noncomputable def probability_both_contracts := P_A_and_B = 71 / 180

theorem probability_of_both_contracts {P_A P_B' P_A_or_B P_A_and_B : ℚ} :
  probability_hardware_contract P_A →
  probability_not_software_contract P_B' →
  probability_either_contract P_A_or_B →
  probability_both_contracts P_A_and_B :=
by
  intros
  sorry

end probability_of_both_contracts_l2344_234434


namespace average_movers_l2344_234421

noncomputable def average_people_per_hour (total_people : ℕ) (total_hours : ℕ) : ℝ :=
  total_people / total_hours

theorem average_movers :
  average_people_per_hour 5000 168 = 29.76 :=
by
  sorry

end average_movers_l2344_234421


namespace num_solution_pairs_l2344_234408

theorem num_solution_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  4 * x + 7 * y = 600 → ∃ n : ℕ, n = 21 :=
by
  sorry

end num_solution_pairs_l2344_234408


namespace jason_home_distance_l2344_234432

theorem jason_home_distance :
  let v1 := 60 -- speed in miles per hour
  let t1 := 0.5 -- time in hours
  let d1 := v1 * t1 -- distance covered in first part of the journey
  let v2 := 90 -- speed in miles per hour for the second part
  let t2 := 1.0 -- remaining time in hours
  let d2 := v2 * t2 -- distance covered in second part of the journey
  let total_distance := d1 + d2 -- total distance to Jason's home
  total_distance = 120 := 
by
  simp only
  sorry

end jason_home_distance_l2344_234432


namespace range_of_m_l2344_234494

theorem range_of_m 
  (m : ℝ)
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = x^3 + (m / 2 + 2) * x^2 - 2 * x)
  (f_prime : ℝ → ℝ)
  (f_prime_def : ∀ x, f_prime x = 3 * x^2 + (m + 4) * x - 2)
  (f_prime_at_1 : f_prime 1 < 0)
  (f_prime_at_2 : f_prime 2 < 0)
  (f_prime_at_3 : f_prime 3 > 0) :
  -37 / 3 < m ∧ m < -9 := 
  sorry

end range_of_m_l2344_234494


namespace question1_perpendicular_question2_parallel_l2344_234403

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

noncomputable def vector_k_a_plus_2_b (k : ℝ) (a b : Vector2D) : Vector2D :=
  ⟨k * a.x + 2 * b.x, k * a.y + 2 * b.y⟩

noncomputable def vector_2_a_minus_4_b (a b : Vector2D) : Vector2D :=
  ⟨2 * a.x - 4 * b.x, 2 * a.y - 4 * b.y⟩

def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

def parallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

def opposite_direction (v1 v2 : Vector2D) : Prop :=
  parallel v1 v2 ∧ v1.x * v2.x + v1.y * v2.y < 0

noncomputable def vector_a : Vector2D := ⟨1, 1⟩
noncomputable def vector_b : Vector2D := ⟨2, 3⟩

theorem question1_perpendicular (k : ℝ) : 
  perpendicular (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ↔ 
  k = -21 / 4 :=
sorry

theorem question2_parallel (k : ℝ) :
  (parallel (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ∧
  opposite_direction (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b)) ↔ 
  k = -1 / 2 :=
sorry

end question1_perpendicular_question2_parallel_l2344_234403


namespace fraction_shaded_l2344_234412

theorem fraction_shaded (s r : ℝ) (h : s^2 = 3 * r^2) :
    (1/2 * π * r^2) / (1/4 * π * s^2) = 2/3 := 
  sorry

end fraction_shaded_l2344_234412


namespace range_of_x_l2344_234473

-- Define the even and increasing properties of the function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- The main theorem to be proven
theorem range_of_x (f : ℝ → ℝ) (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) 
  (h_cond : ∀ x : ℝ, f (x - 1) < f (2 - x)) :
  ∀ x : ℝ, x < 3 / 2 :=
by
  sorry

end range_of_x_l2344_234473


namespace storybooks_sciencebooks_correct_l2344_234442

-- Given conditions
def total_books : ℕ := 144
def ratio_storybooks_sciencebooks := (7, 5)
def fraction_storybooks := 7 / (7 + 5)
def fraction_sciencebooks := 5 / (7 + 5)

-- Prove the number of storybooks and science books
def number_of_storybooks : ℕ := 84
def number_of_sciencebooks : ℕ := 60

theorem storybooks_sciencebooks_correct :
  (fraction_storybooks * total_books = number_of_storybooks) ∧
  (fraction_sciencebooks * total_books = number_of_sciencebooks) :=
by
  sorry

end storybooks_sciencebooks_correct_l2344_234442


namespace solve_quadratic_l2344_234456

theorem solve_quadratic :
  (x = 0 ∨ x = 2/5) ↔ (5 * x^2 - 2 * x = 0) :=
by
  sorry

end solve_quadratic_l2344_234456


namespace number_of_subsets_l2344_234482

-- Defining the type of the elements
variable {α : Type*}

-- Statement of the problem in Lean 4
theorem number_of_subsets (s : Finset α) (h : s.card = n) : (Finset.powerset s).card = 2^n := 
sorry

end number_of_subsets_l2344_234482


namespace probability_factor_120_lt_8_l2344_234417

theorem probability_factor_120_lt_8 :
  let n := 120
  let total_factors := 16
  let favorable_factors := 6
  (6 / 16 : ℚ) = 3 / 8 :=
by 
  sorry

end probability_factor_120_lt_8_l2344_234417


namespace find_A_l2344_234453

/-- Given that the equation Ax + 10y = 100 has two distinct positive integer solutions, prove that A = 10. -/
theorem find_A (A x1 y1 x2 y2 : ℕ) (h1 : A > 0) (h2 : x1 > 0) (h3 : y1 > 0) 
  (h4 : x2 > 0) (h5 : y2 > 0) (distinct_solutions : x1 ≠ x2 ∧ y1 ≠ y2) 
  (eq1 : A * x1 + 10 * y1 = 100) (eq2 : A * x2 + 10 * y2 = 100) : 
  A = 10 := sorry

end find_A_l2344_234453


namespace elderly_teachers_in_sample_l2344_234475

-- Definitions based on the conditions
def numYoungTeachersSampled : ℕ := 320
def ratioYoungToElderly : ℚ := 16 / 9

-- The theorem that needs to be proved
theorem elderly_teachers_in_sample :
  ∃ numElderlyTeachersSampled : ℕ, 
    numYoungTeachersSampled * (9 / 16) = numElderlyTeachersSampled := 
by
  use 180
  sorry

end elderly_teachers_in_sample_l2344_234475


namespace iron_balls_molded_l2344_234404

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end iron_balls_molded_l2344_234404


namespace total_number_of_cards_l2344_234423

/-- There are 9 playing cards and 4 ID cards initially.
If you add 6 more playing cards and 3 more ID cards,
then the total number of playing cards and ID cards will be 22. -/
theorem total_number_of_cards :
  let initial_playing_cards := 9
  let initial_id_cards := 4
  let additional_playing_cards := 6
  let additional_id_cards := 3
  let total_playing_cards := initial_playing_cards + additional_playing_cards
  let total_id_cards := initial_id_cards + additional_id_cards
  let total_cards := total_playing_cards + total_id_cards
  total_cards = 22 :=
by
  sorry

end total_number_of_cards_l2344_234423


namespace problem_statement_l2344_234457

-- Definitions based on conditions
def position_of_3_in_8_063 := "thousandths"
def representation_of_3_in_8_063 : ℝ := 3 * 0.001
def unit_in_0_48 : ℝ := 0.01

theorem problem_statement :
  (position_of_3_in_8_063 = "thousandths") ∧
  (representation_of_3_in_8_063 = 3 * 0.001) ∧
  (unit_in_0_48 = 0.01) :=
sorry

end problem_statement_l2344_234457


namespace find_triplets_l2344_234424

theorem find_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧ (2 * y^3 + 1 = 3 * x * y) ∧ (2 * z^3 + 1 = 3 * y * z) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 / 2 ∧ y = -1 / 2 ∧ z = -1 / 2) :=
by 
  intro h
  sorry

end find_triplets_l2344_234424


namespace total_crayons_l2344_234465

-- Define relevant conditions
def crayons_per_child : ℕ := 8
def number_of_children : ℕ := 7

-- Define the Lean statement to prove the total number of crayons
theorem total_crayons : crayons_per_child * number_of_children = 56 :=
by
  sorry

end total_crayons_l2344_234465


namespace total_slides_used_l2344_234489

theorem total_slides_used (duration : ℕ) (initial_slides : ℕ) (initial_time : ℕ) (constant_rate : ℕ) (total_time: ℕ)
  (H1 : duration = 50)
  (H2 : initial_slides = 4)
  (H3 : initial_time = 2)
  (H4 : constant_rate = initial_slides / initial_time)
  (H5 : total_time = duration) 
  : (constant_rate * total_time) = 100 := 
by
  sorry

end total_slides_used_l2344_234489


namespace driving_time_equation_l2344_234447

theorem driving_time_equation :
  ∀ (t : ℝ), (60 * t + 90 * (3.5 - t) = 300) :=
by
  intro t
  sorry

end driving_time_equation_l2344_234447


namespace sequence_pattern_l2344_234445

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end sequence_pattern_l2344_234445


namespace bicycle_count_l2344_234451

theorem bicycle_count (B T : ℕ) (hT : T = 20) (h_wheels : 2 * B + 3 * T = 160) : B = 50 :=
by
  sorry

end bicycle_count_l2344_234451


namespace part1_part2_l2344_234454

-- Definitions as per the conditions
def A (a b : ℚ) := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) := - a^2 + (1/2) * a * b + 2 / 3

-- Part (1)
theorem part1 (a b : ℚ) (h1 : a = -1) (h2 : b = -2) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3 := 
by 
  sorry

-- Part (2)
theorem part2 (a : ℚ) : 
  (∀ a : ℚ, 4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3) → 
  b = 1/2 :=
by 
  sorry

end part1_part2_l2344_234454


namespace total_students_count_l2344_234405

-- Define the conditions
def num_rows : ℕ := 8
def students_per_row : ℕ := 6
def students_last_row : ℕ := 5
def rows_with_six_students : ℕ := 7

-- Define the total students
def total_students : ℕ :=
  (rows_with_six_students * students_per_row) + students_last_row

-- The theorem to prove
theorem total_students_count : total_students = 47 := by
  sorry

end total_students_count_l2344_234405


namespace fraction_of_sand_is_one_third_l2344_234459

noncomputable def total_weight : ℝ := 24
noncomputable def weight_of_water (total_weight : ℝ) : ℝ := total_weight / 4
noncomputable def weight_of_gravel : ℝ := 10
noncomputable def weight_of_sand (total_weight weight_of_water weight_of_gravel : ℝ) : ℝ :=
  total_weight - weight_of_water - weight_of_gravel
noncomputable def fraction_of_sand (weight_of_sand total_weight : ℝ) : ℝ :=
  weight_of_sand / total_weight

theorem fraction_of_sand_is_one_third :
  fraction_of_sand (weight_of_sand total_weight (weight_of_water total_weight) weight_of_gravel) total_weight
  = 1/3 := by
  sorry

end fraction_of_sand_is_one_third_l2344_234459


namespace axis_of_symmetry_l2344_234444

theorem axis_of_symmetry (a : ℝ) (h : a ≠ 0) : y = - 1 / (4 * a) :=
sorry

end axis_of_symmetry_l2344_234444


namespace weekly_deficit_is_2800_l2344_234458

def daily_intake (day : String) : ℕ :=
  if day = "Monday" then 2500 else 
  if day = "Tuesday" then 2600 else 
  if day = "Wednesday" then 2400 else 
  if day = "Thursday" then 2700 else 
  if day = "Friday" then 2300 else 
  if day = "Saturday" then 3500 else 
  if day = "Sunday" then 2400 else 0

def daily_expenditure (day : String) : ℕ :=
  if day = "Monday" then 3000 else 
  if day = "Tuesday" then 3200 else 
  if day = "Wednesday" then 2900 else 
  if day = "Thursday" then 3100 else 
  if day = "Friday" then 2800 else 
  if day = "Saturday" then 3000 else 
  if day = "Sunday" then 2700 else 0

def daily_deficit (day : String) : ℤ :=
  daily_expenditure day - daily_intake day

def weekly_caloric_deficit : ℤ :=
  daily_deficit "Monday" +
  daily_deficit "Tuesday" +
  daily_deficit "Wednesday" +
  daily_deficit "Thursday" +
  daily_deficit "Friday" +
  daily_deficit "Saturday" +
  daily_deficit "Sunday"

theorem weekly_deficit_is_2800 : weekly_caloric_deficit = 2800 := by
  sorry

end weekly_deficit_is_2800_l2344_234458


namespace topology_on_X_l2344_234429

-- Define the universal set X
def X : Set ℕ := {1, 2, 3}

-- Sequences of candidate sets v
def v1 : Set (Set ℕ) := {∅, {1}, {3}, {1, 2, 3}}
def v2 : Set (Set ℕ) := {∅, {2}, {3}, {2, 3}, {1, 2, 3}}
def v3 : Set (Set ℕ) := {∅, {1}, {1, 2}, {1, 3}}
def v4 : Set (Set ℕ) := {∅, {1, 3}, {2, 3}, {3}, {1, 2, 3}}

-- Define the conditions that determine a topology
def isTopology (X : Set ℕ) (v : Set (Set ℕ)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧ 
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋃₀ s ∈ v) ∧
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋂₀ s ∈ v)

-- The statement we want to prove
theorem topology_on_X : 
  isTopology X v2 ∧ isTopology X v4 :=
by
  sorry

end topology_on_X_l2344_234429


namespace abs_eq_neg_of_nonpos_l2344_234407

theorem abs_eq_neg_of_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
by
  have ha : |a| ≥ 0 := abs_nonneg a
  rw [h] at ha
  exact neg_nonneg.mp ha

end abs_eq_neg_of_nonpos_l2344_234407


namespace mapping_image_l2344_234441

theorem mapping_image (x y l m : ℤ) (h1 : x = 4) (h2 : y = 6) (h3 : l = x + y) (h4 : m = x - y) :
  (l, m) = (10, -2) := by
  sorry

end mapping_image_l2344_234441


namespace arithmetic_sequence_a4_l2344_234409

theorem arithmetic_sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) :
  S 8 = 30 → S 4 = 7 → 
      (∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) → 
      a 4 = a1 + 3 * d → 
      a 4 = 13 / 4 := by
  intros hS8 hS4 hS_formula ha4_formula
  -- Formal proof to be filled in
  sorry

end arithmetic_sequence_a4_l2344_234409


namespace sequence_increasing_or_decreasing_l2344_234422

theorem sequence_increasing_or_decreasing (x : ℕ → ℝ) (h1 : x 1 > 0) (h2 : x 1 ≠ 1) 
  (hrec : ∀ n, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∀ n, x n < x (n + 1) ∨ x n > x (n + 1) :=
by
  sorry

end sequence_increasing_or_decreasing_l2344_234422


namespace angle_quadrant_l2344_234480

theorem angle_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 90 < α + 180 ∧ α + 180 < 270 :=
by
  sorry

end angle_quadrant_l2344_234480


namespace max_students_l2344_234427

open Nat

theorem max_students (B G : ℕ) (h1 : 11 * B = 7 * G) (h2 : G = B + 72) (h3 : B + G ≤ 550) : B + G = 324 := by
  sorry

end max_students_l2344_234427


namespace fred_blue_marbles_l2344_234498

theorem fred_blue_marbles (tim_marbles : ℕ) (fred_marbles : ℕ) (h1 : tim_marbles = 5) (h2 : fred_marbles = 22 * tim_marbles) : fred_marbles = 110 :=
by
  sorry

end fred_blue_marbles_l2344_234498


namespace race_permutations_l2344_234435

-- Define the number of participants
def num_participants : ℕ := 4

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n + 1) * factorial n

-- Theorem: Given 4 participants, the number of different possible orders they can finish the race is 24.
theorem race_permutations : factorial num_participants = 24 := by
  -- sorry added to skip the proof
  sorry

end race_permutations_l2344_234435


namespace vertex_of_parabola_on_x_axis_l2344_234476

theorem vertex_of_parabola_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*x + c = 0)) ↔ c = 9 :=
by
  sorry

end vertex_of_parabola_on_x_axis_l2344_234476


namespace no_integer_triple_exists_for_10_l2344_234433

theorem no_integer_triple_exists_for_10 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 :=
sorry

end no_integer_triple_exists_for_10_l2344_234433


namespace function_increasing_interval_l2344_234463

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x ^ 2) / Real.log 2

def domain (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem function_increasing_interval : 
  ∀ x, domain x → 0 < x ∧ x < 1 → ∀ y, domain y → 0 < y ∧ y < 1 → x < y → f x < f y :=
by 
  intros x hx h0 y hy h1 hxy
  sorry

end function_increasing_interval_l2344_234463


namespace find_k_value_l2344_234462

-- Definitions based on conditions
variables {k b x y : ℝ} -- k, b, x, and y are real numbers

-- Conditions given in the problem
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Proposition: Given the conditions, prove that k = 2
theorem find_k_value (h₁ : ∀ x y, y = linear_function k b x → y + 6 = linear_function k b (x + 3)) : k = 2 :=
by
  sorry

end find_k_value_l2344_234462


namespace unique_symmetric_solutions_l2344_234426

theorem unique_symmetric_solutions (a b α β : ℝ) (h_mul : α * β = a) (h_add : α + β = b) :
  ∀ (x y : ℝ), x * y = a ∧ x + y = b → (x = α ∧ y = β) ∨ (x = β ∧ y = α) :=
by
  sorry

end unique_symmetric_solutions_l2344_234426


namespace tetrahedron_face_area_squared_l2344_234416

variables {S0 S1 S2 S3 α12 α13 α23 : ℝ}

-- State the theorem
theorem tetrahedron_face_area_squared :
  (S0)^2 = (S1)^2 + (S2)^2 + (S3)^2 - 2 * S1 * S2 * (Real.cos α12) - 2 * S1 * S3 * (Real.cos α13) - 2 * S2 * S3 * (Real.cos α23) :=
sorry

end tetrahedron_face_area_squared_l2344_234416


namespace ferris_wheel_seat_capacity_l2344_234481

-- Define the given conditions
def people := 16
def seats := 4

-- Define the problem and the proof goal
theorem ferris_wheel_seat_capacity : people / seats = 4 := by
  sorry

end ferris_wheel_seat_capacity_l2344_234481


namespace saree_original_price_l2344_234446

theorem saree_original_price
  (sale_price : ℝ)
  (P : ℝ)
  (h_discount : sale_price = 0.80 * P * 0.95)
  (h_sale_price : sale_price = 266) :
  P = 350 :=
by
  -- Proof to be completed later
  sorry

end saree_original_price_l2344_234446


namespace total_amount_divided_l2344_234420

theorem total_amount_divided (A B C : ℝ) (h1 : A = (2/3) * (B + C)) (h2 : B = (2/3) * (A + C)) (h3 : A = 200) :
  A + B + C = 500 :=
by
  sorry

end total_amount_divided_l2344_234420


namespace find_overhead_expenses_l2344_234485

noncomputable def overhead_expenses : ℝ := 35.29411764705882 / (1 + 0.1764705882352942)

theorem find_overhead_expenses (cost_price selling_price profit_percent : ℝ) (h_cp : cost_price = 225) (h_sp : selling_price = 300) (h_pp : profit_percent = 0.1764705882352942) :
  overhead_expenses = 30 :=
by
  sorry

end find_overhead_expenses_l2344_234485


namespace complex_expression_l2344_234401

theorem complex_expression (i : ℂ) (h : i^2 = -1) : ( (1 + i) / (1 - i) )^2006 = -1 :=
by {
  sorry
}

end complex_expression_l2344_234401


namespace tan_squared_sum_geq_three_over_eight_l2344_234419

theorem tan_squared_sum_geq_three_over_eight 
  (α β γ : ℝ) 
  (hα : 0 ≤ α ∧ α < π / 2) 
  (hβ : 0 ≤ β ∧ β < π / 2) 
  (hγ : 0 ≤ γ ∧ γ < π / 2) 
  (h_sum : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / 8 := 
sorry

end tan_squared_sum_geq_three_over_eight_l2344_234419


namespace cuts_for_20_pentagons_l2344_234411

theorem cuts_for_20_pentagons (K : ℕ) : 20 * 540 + (K - 19) * 180 ≤ 360 * K + 540 ↔ K ≥ 38 :=
by
  sorry

end cuts_for_20_pentagons_l2344_234411


namespace find_four_digit_numbers_l2344_234493

theorem find_four_digit_numbers (a b c d : ℕ) : 
  (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
  (1000 * a + 100 * b + 10 * c + d ≤ 9999) ∧ 
  (1000 ≤ 1000 * d + 100 * c + 10 * b + a) ∧ 
  (1000 * d + 100 * c + 10 * b + a ≤ 9999) ∧
  (a + d = 9) ∧ 
  (b + c = 13) ∧
  (1001 * (a + d) + 110 * (b + c) = 19448) → 
  (1000 * a + 100 * b + 10 * c + d = 9949 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9859 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9769 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9679 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9589 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9499) :=
sorry

end find_four_digit_numbers_l2344_234493


namespace smallest_n_satisfying_condition_l2344_234478

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (n > 1) ∧ (∀ i : ℕ, i ≥ 1 → i < n → (∃ k : ℕ, i + (i+1) = k^2)) ∧ n = 8 :=
sorry

end smallest_n_satisfying_condition_l2344_234478


namespace min_value_of_sum_of_sides_proof_l2344_234425

noncomputable def min_value_of_sum_of_sides (a b c : ℝ) (angleC : ℝ) : ℝ :=
  if (angleC = 60 * (Real.pi / 180)) ∧ ((a + b)^2 - c^2 = 4) then 4 * Real.sqrt 3 / 3 
  else 0

theorem min_value_of_sum_of_sides_proof (a b c : ℝ) (angleC : ℝ) 
  (h1 : angleC = 60 * (Real.pi / 180)) 
  (h2 : (a + b)^2 - c^2 = 4) 
  : min_value_of_sum_of_sides a b c angleC = 4 * Real.sqrt 3 / 3 := 
by
  sorry

end min_value_of_sum_of_sides_proof_l2344_234425


namespace angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l2344_234439

variable (a b c A B C : ℝ)

-- Condition 1
def cond1 : Prop := b / a = (Real.cos B + 1) / (Real.sqrt 3 * Real.sin A)

-- Condition 2
def cond2 : Prop := 2 * b * Real.sin A = a * Real.tan B

-- Condition 3
def cond3 : Prop := (c - a = b * Real.cos A - a * Real.cos B)

-- Angle B and area of the triangle for Condition 1
theorem angle_B_cond1 (h : cond1 a b A B) : B = π / 3 := sorry

theorem area_range_cond1 (h : cond1 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 2
theorem angle_B_cond2 (h : cond2 a b A B) : B = π / 3 := sorry

theorem area_range_cond2 (h : cond2 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 3
theorem angle_B_cond3 (h : cond3 a b c A B) : B = π / 3 := sorry

theorem area_range_cond3 (h : cond3 a b c A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

end angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l2344_234439


namespace furniture_cost_final_price_l2344_234443

theorem furniture_cost_final_price 
  (table_cost : ℤ := 140)
  (chair_ratio : ℚ := 1/7)
  (sofa_ratio : ℕ := 2)
  (discount : ℚ := 0.10)
  (tax : ℚ := 0.07)
  (exchange_rate : ℚ := 1.2) :
  let chair_cost := table_cost * chair_ratio
  let sofa_cost := table_cost * sofa_ratio
  let total_cost_before_discount := table_cost + 4 * chair_cost + sofa_cost
  let table_discount := discount * table_cost
  let discounted_table_cost := table_cost - table_discount
  let total_cost_after_discount := discounted_table_cost + 4 * chair_cost + sofa_cost
  let sales_tax := tax * total_cost_after_discount
  let final_cost := total_cost_after_discount + sales_tax
  final_cost = 520.02 
:= sorry

end furniture_cost_final_price_l2344_234443


namespace books_leftover_l2344_234415

-- Definitions of the conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def books_bought : ℕ := 26

-- The theorem stating the proof problem
theorem books_leftover : (initial_books + books_bought) - (shelves * books_per_shelf) = 2 := by
  sorry

end books_leftover_l2344_234415


namespace length_of_PB_l2344_234490

theorem length_of_PB 
  (AB BC : ℝ) 
  (PA PD PC PB : ℝ)
  (h1 : AB = 2 * BC) 
  (h2 : PA = 5) 
  (h3 : PD = 12) 
  (h4 : PC = 13) 
  (h5 : PA^2 + PB^2 = (AB^2 + BC^2) / 5) -- derived from question
  (h6 : PB^2 = ((2 * BC)^2) - PA^2) : 
  PB = 10.5 :=
by 
  -- We would insert proof steps here (not required as per instructions)
  sorry

end length_of_PB_l2344_234490


namespace blue_part_length_l2344_234466

variable (total_length : ℝ) (black_part white_part blue_part : ℝ)

-- Conditions
axiom h1 : black_part = 1 / 8 * total_length
axiom h2 : white_part = 1 / 2 * (total_length - black_part)
axiom h3 : total_length = 8

theorem blue_part_length : blue_part = total_length - black_part - white_part :=
by
  sorry

end blue_part_length_l2344_234466


namespace closest_integer_to_sqrt_11_l2344_234479

theorem closest_integer_to_sqrt_11 : 
  ∀ (x : ℝ), (3 : ℝ) ≤ x → x ≤ 3.5 → x = 3 :=
by
  intro x hx h3_5
  sorry

end closest_integer_to_sqrt_11_l2344_234479


namespace m_minus_n_eq_2_l2344_234495

theorem m_minus_n_eq_2 (m n : ℕ) (h1 : ∃ x : ℕ, m = 101 * x) (h2 : ∃ y : ℕ, n = 63 * y) (h3 : m + n = 2018) : m - n = 2 :=
sorry

end m_minus_n_eq_2_l2344_234495


namespace cone_base_circumference_l2344_234467

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (circ_res : ℝ) :
  r = 4 → θ = 270 → circ_res = 6 * Real.pi :=
by 
  sorry

end cone_base_circumference_l2344_234467


namespace draw_sequence_count_l2344_234449

noncomputable def total_sequences : ℕ :=
  (Nat.choose 4 3) * (Nat.factorial 4) * 5

theorem draw_sequence_count : total_sequences = 480 := by
  sorry

end draw_sequence_count_l2344_234449


namespace cos_alpha_value_l2344_234491

open Real

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = 3 / 5) (h2 : π / 6 < α ∧ α < 5 * π / 6) :
  cos α = (3 * sqrt 3 - 4) / 10 :=
by
  sorry

end cos_alpha_value_l2344_234491


namespace HVAC_cost_per_vent_l2344_234474

/-- 
The cost of Joe's new HVAC system is $20,000. It includes 2 conditioning zones, each with 5 vents.
Prove that the cost of the system per vent is $2,000.
-/
theorem HVAC_cost_per_vent
    (cost : ℕ := 20000)
    (zones : ℕ := 2)
    (vents_per_zone : ℕ := 5)
    (total_vents : ℕ := zones * vents_per_zone) :
    (cost / total_vents) = 2000 := by
  sorry

end HVAC_cost_per_vent_l2344_234474


namespace compare_y_l2344_234428

-- Define the points M and N lie on the graph of y = -5/x
def on_inverse_proportion_curve (x y : ℝ) : Prop :=
  y = -5 / x

-- Main theorem to be proven
theorem compare_y (x1 y1 x2 y2 : ℝ) (h1 : on_inverse_proportion_curve x1 y1) (h2 : on_inverse_proportion_curve x2 y2) (hx : x1 > 0 ∧ x2 < 0) : y1 < y2 :=
by
  sorry

end compare_y_l2344_234428


namespace integers_solution_l2344_234484

theorem integers_solution (a b : ℤ) (S D : ℤ) 
  (h1 : S = a + b) (h2 : D = a - b) (h3 : S / D = 3) (h4 : S * D = 300) : 
  ((a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10)) :=
by
  sorry

end integers_solution_l2344_234484


namespace ant_moves_probability_l2344_234469

theorem ant_moves_probability :
  let m := 73
  let n := 48
  m + n = 121 := by
  sorry

end ant_moves_probability_l2344_234469


namespace max_students_distribute_pens_pencils_l2344_234499

noncomputable def gcd_example : ℕ :=
  Nat.gcd 1340 1280

theorem max_students_distribute_pens_pencils : gcd_example = 20 :=
sorry

end max_students_distribute_pens_pencils_l2344_234499


namespace statement1_statement2_statement3_l2344_234496

variable (a b c m : ℝ)

-- Given condition
def quadratic_eq (a b c : ℝ) : Prop := a ≠ 0

-- Statement 1
theorem statement1 (h0 : quadratic_eq a b c) (h1 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = 2) : 2 * a - c = 0 :=
sorry

-- Statement 2
theorem statement2 (h0 : quadratic_eq a b c) (h2 : b = 2 * a + c) : (b^2 - 4 * a * c) > 0 :=
sorry

-- Statement 3
theorem statement3 (h0 : quadratic_eq a b c) (h3 : a * m^2 + b * m + c = 0) : b^2 - 4 * a * c = (2 * a * m + b)^2 :=
sorry

end statement1_statement2_statement3_l2344_234496


namespace william_wins_10_rounds_l2344_234437

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l2344_234437


namespace compute_fraction_power_l2344_234488

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l2344_234488
