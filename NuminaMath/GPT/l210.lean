import Mathlib

namespace trains_clear_time_l210_21050

noncomputable def length_train1 : ℝ := 150
noncomputable def length_train2 : ℝ := 165
noncomputable def speed_train1_kmh : ℝ := 80
noncomputable def speed_train2_kmh : ℝ := 65
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * (5/18)
noncomputable def speed_train1 : ℝ := kmh_to_mps speed_train1_kmh
noncomputable def speed_train2 : ℝ := kmh_to_mps speed_train2_kmh
noncomputable def total_distance : ℝ := length_train1 + length_train2
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_time : time_to_clear = 7.82 := 
sorry

end trains_clear_time_l210_21050


namespace benny_eggs_l210_21045

def dozen := 12

def total_eggs (n: Nat) := n * dozen

theorem benny_eggs:
  total_eggs 7 = 84 := 
by 
  sorry

end benny_eggs_l210_21045


namespace problem1_problem2_1_problem2_2_l210_21051

-- Define the quadratic function and conditions
def quadratic (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

-- Problem 1: Expression of the quadratic function given vertex
theorem problem1 (b c : ℝ) : (quadratic 2 b c = 0) ∧ (∀ x : ℝ, quadratic x b c = (x - 2)^2) ↔ (b = -4) ∧ (c = 4) := sorry

-- Problem 2.1: Given n < -5 and y1 = y2, range of b + c
theorem problem2_1 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : quadratic (3*n - 4) b c = y1)
  (h3 : quadratic (5*n + 6) b c = y2) (h4 : y1 = y2) : b + c < -38 := sorry

-- Problem 2.2: Given n < -5 and c > 0, compare values of y1 and y2
theorem problem2_2 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : c > 0) 
  (h3 : quadratic (3*n - 4) b c = y1) (h4 : quadratic (5*n + 6) b c = y2) : y1 < y2 := sorry

end problem1_problem2_1_problem2_2_l210_21051


namespace slope_of_line_l210_21012

theorem slope_of_line (a : ℝ) (h : a = (Real.tan (Real.pi / 3))) : a = Real.sqrt 3 := by
sorry

end slope_of_line_l210_21012


namespace percentage_neither_language_l210_21041

def total_diplomats : ℕ := 150
def french_speaking : ℕ := 17
def russian_speaking : ℕ := total_diplomats - 32
def both_languages : ℕ := 10 * total_diplomats / 100

theorem percentage_neither_language :
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  neither_language * 100 / total_diplomats = 20 :=
by
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  sorry

end percentage_neither_language_l210_21041


namespace fifteenth_term_l210_21011

noncomputable def seq : ℕ → ℝ
| 0       => 3
| 1       => 4
| (n + 2) => 12 / seq (n + 1)

theorem fifteenth_term :
  seq 14 = 3 :=
sorry

end fifteenth_term_l210_21011


namespace largest_integer_lt_100_with_rem_4_div_7_l210_21059

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l210_21059


namespace correct_number_is_650_l210_21078

theorem correct_number_is_650 
  (n : ℕ) 
  (h : n - 152 = 346): 
  n + 152 = 650 :=
by
  sorry

end correct_number_is_650_l210_21078


namespace range_of_x_range_of_a_l210_21004

variable (a x : ℝ)

-- Define proposition p: x^2 - 3ax + 2a^2 < 0
def p (a x : ℝ) : Prop := x^2 - 3 * a * x + 2 * a^2 < 0

-- Define proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- First theorem: Prove the range of x when a = 2 and p ∨ q is true
theorem range_of_x (h : p 2 x ∨ q x) : 2 < x ∧ x < 4 := 
by sorry

-- Second theorem: Prove the range of a when ¬p is necessary but not sufficient for ¬q
theorem range_of_a (h : ∀ x, q x → p a x) : 3/2 ≤ a ∧ a ≤ 2 := 
by sorry

end range_of_x_range_of_a_l210_21004


namespace range_of_t_l210_21067

theorem range_of_t (t : ℝ) (h : ∃ x : ℝ, x ∈ Set.Iic t ∧ (x^2 - 4*x + t ≤ 0)) : 0 ≤ t ∧ t ≤ 4 :=
sorry

end range_of_t_l210_21067


namespace triangle_third_side_lengths_l210_21095

theorem triangle_third_side_lengths : 
  ∃ (x : ℕ), (3 < x ∧ x < 11) ∧ (x ≠ 3) ∧ (x ≠ 11) ∧ 
    ((x = 4) ∨ (x = 5) ∨ (x = 6) ∨ (x = 7) ∨ (x = 8) ∨ (x = 9) ∨ (x = 10)) :=
by
  sorry

end triangle_third_side_lengths_l210_21095


namespace square_completion_form_l210_21016

theorem square_completion_form (x k m: ℝ) (h: 16*x^2 - 32*x - 512 = 0):
  (x + k)^2 = m ↔ m = 65 :=
by
  sorry

end square_completion_form_l210_21016


namespace trail_mix_total_weight_l210_21092

def peanuts : ℝ := 0.16666666666666666
def chocolate_chips : ℝ := 0.16666666666666666
def raisins : ℝ := 0.08333333333333333
def trail_mix_weight : ℝ := 0.41666666666666663

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = trail_mix_weight :=
sorry

end trail_mix_total_weight_l210_21092


namespace inclination_angle_of_line_l210_21066

-- Definitions drawn from the condition.
def line_equation (x y : ℝ) := x - y + 1 = 0

-- The statement of the theorem (equivalent proof problem).
theorem inclination_angle_of_line : ∀ x y : ℝ, line_equation x y → θ = π / 4 :=
sorry

end inclination_angle_of_line_l210_21066


namespace fraction_of_tea_in_final_cup2_is_5_over_8_l210_21085

-- Defining the initial conditions and the transfers
structure CupContents where
  tea : ℚ
  milk : ℚ

def initialCup1 : CupContents := { tea := 6, milk := 0 }
def initialCup2 : CupContents := { tea := 0, milk := 3 }

def transferOneThird (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let teaTransferred := (1 / 3) * cup1.tea
  ( { cup1 with tea := cup1.tea - teaTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk } )

def transferOneFourth (cup2 : CupContents) (cup1 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup2.tea + cup2.milk
  let amountTransferred := (1 / 4) * mixedTotal
  let teaTransferred := amountTransferred * (cup2.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup2.milk / mixedTotal)
  ( { tea := cup1.tea + teaTransferred, milk := cup1.milk + milkTransferred },
    { tea := cup2.tea - teaTransferred, milk := cup2.milk - milkTransferred } )

def transferOneHalf (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup1.tea + cup1.milk
  let amountTransferred := (1 / 2) * mixedTotal
  let teaTransferred := amountTransferred * (cup1.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup1.milk / mixedTotal)
  ( { tea := cup1.tea - teaTransferred, milk := cup1.milk - milkTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk + milkTransferred } )

def finalContents (cup1 cup2 : CupContents) : CupContents × CupContents :=
  let (cup1Transferred, cup2Transferred) := transferOneThird cup1 cup2
  let (cup1Mixed, cup2Mixed) := transferOneFourth cup2Transferred cup1Transferred
  transferOneHalf cup1Mixed cup2Mixed

-- Statement to be proved
theorem fraction_of_tea_in_final_cup2_is_5_over_8 :
  ((finalContents initialCup1 initialCup2).snd.tea / ((finalContents initialCup1 initialCup2).snd.tea + (finalContents initialCup1 initialCup2).snd.milk) = 5 / 8) :=
sorry

end fraction_of_tea_in_final_cup2_is_5_over_8_l210_21085


namespace borgnine_tarantulas_needed_l210_21055

def total_legs_goal : ℕ := 1100
def chimp_legs : ℕ := 12 * 4
def lion_legs : ℕ := 8 * 4
def lizard_legs : ℕ := 5 * 4
def tarantula_legs : ℕ := 8

theorem borgnine_tarantulas_needed : 
  let total_legs_seen := chimp_legs + lion_legs + lizard_legs
  let legs_needed := total_legs_goal - total_legs_seen
  let num_tarantulas := legs_needed / tarantula_legs
  num_tarantulas = 125 := 
by
  sorry

end borgnine_tarantulas_needed_l210_21055


namespace distance_travelled_downstream_in_12_minutes_l210_21024

noncomputable def speed_boat_still : ℝ := 15 -- in km/hr
noncomputable def rate_current : ℝ := 3 -- in km/hr
noncomputable def time_downstream : ℝ := 12 / 60 -- in hr (since 12 minutes is 12/60 hours)
noncomputable def effective_speed_downstream : ℝ := speed_boat_still + rate_current -- in km/hr
noncomputable def distance_downstream := effective_speed_downstream * time_downstream -- in km

theorem distance_travelled_downstream_in_12_minutes :
  distance_downstream = 3.6 := 
by
  sorry

end distance_travelled_downstream_in_12_minutes_l210_21024


namespace sum_of_three_different_squares_l210_21033

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def existing_list (ns : List Nat) : Prop :=
  ∀ n ∈ ns, is_perfect_square n

theorem sum_of_three_different_squares (a b c : Nat) :
  existing_list [a, b, c] →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 128 →
  false :=
by
  intros
  sorry

end sum_of_three_different_squares_l210_21033


namespace solve_x_l210_21072

theorem solve_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.84) : x = 72 := 
by
  sorry

end solve_x_l210_21072


namespace partition_sum_le_152_l210_21036

theorem partition_sum_le_152 {S : ℕ} (l : List ℕ) 
  (h1 : ∀ n ∈ l, 1 ≤ n ∧ n ≤ 10) 
  (h2 : l.sum = S) : 
  (∃ l1 l2 : List ℕ, l1.sum ≤ 80 ∧ l2.sum ≤ 80 ∧ l1 ++ l2 = l) ↔ S ≤ 152 := 
by
  sorry

end partition_sum_le_152_l210_21036


namespace Amanda_hiking_trip_l210_21093

-- Define the conditions
variable (x : ℝ) -- the total distance of Amanda's hiking trip
variable (forest_path : ℝ) (plain_path : ℝ)
variable (stream_path : ℝ) (mountain_path : ℝ)

-- Given conditions
axiom h1 : stream_path = (1/4) * x
axiom h2 : forest_path = 25
axiom h3 : mountain_path = (1/6) * x
axiom h4 : plain_path = 2 * forest_path
axiom h5 : stream_path + forest_path + mountain_path + plain_path = x

-- Proposition to prove
theorem Amanda_hiking_trip : x = 900 / 7 :=
by
  sorry

end Amanda_hiking_trip_l210_21093


namespace largest_divisor_of_expression_l210_21083

theorem largest_divisor_of_expression (x : ℤ) (h_even : x % 2 = 0) :
  ∃ k, (∀ x, x % 2 = 0 → k ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) ∧ 
       (∀ m, (∀ x, x % 2 = 0 → m ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) → m ≤ k) ∧ 
       k = 32 :=
sorry

end largest_divisor_of_expression_l210_21083


namespace sum_mod_9_l210_21064

theorem sum_mod_9 (x y z : ℕ) (h1 : x < 9) (h2 : y < 9) (h3 : z < 9) 
  (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : (x * y * z) % 9 = 1) (h8 : (7 * z) % 9 = 4) (h9 : (8 * y) % 9 = (5 + y) % 9) :
  (x + y + z) % 9 = 7 := 
by {
  sorry
}

end sum_mod_9_l210_21064


namespace total_number_of_balls_l210_21039

-- Define the conditions
def balls_per_box : Nat := 3
def number_of_boxes : Nat := 2

-- Define the proposition
theorem total_number_of_balls : (balls_per_box * number_of_boxes) = 6 :=
by
  sorry

end total_number_of_balls_l210_21039


namespace sequence_no_limit_l210_21087

noncomputable def sequence_limit (x : ℕ → ℝ) (a : ℝ) : Prop :=
    ∀ ε > 0, ∃ N, ∀ n > N, abs (x n - a) < ε

theorem sequence_no_limit (x : ℕ → ℝ) (a : ℝ) (ε : ℝ) (k : ℕ) :
    (ε > 0) ∧ (∀ n, n > k → abs (x n - a) ≥ ε) → ¬ sequence_limit x a :=
by
  sorry

end sequence_no_limit_l210_21087


namespace find_t_l210_21084

open Real

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (t : ℝ) :
  let m := (t + 1, 1)
  let n := (t + 2, 2)
  dot_product (vector_add m n) (vector_sub m n) = 0 → 
  t = -3 :=
by
  intro h
  sorry

end find_t_l210_21084


namespace avg_height_of_remaining_students_l210_21002

-- Define the given conditions
def avg_height_11_members : ℝ := 145.7
def number_of_members : ℝ := 11
def height_of_two_students : ℝ := 142.1

-- Define what we need to prove
theorem avg_height_of_remaining_students :
  (avg_height_11_members * number_of_members - 2 * height_of_two_students) / (number_of_members - 2) = 146.5 :=
by
  sorry

end avg_height_of_remaining_students_l210_21002


namespace not_54_after_one_hour_l210_21094

theorem not_54_after_one_hour (n : ℕ) (initial_number : ℕ) (initial_factors : ℕ × ℕ)
  (h₀ : initial_number = 12)
  (h₁ : initial_factors = (2, 1)) :
  (∀ k : ℕ, k < 60 →
    ∀ current_factors : ℕ × ℕ,
    current_factors = (initial_factors.1 + k, initial_factors.2 + k) ∨
    current_factors = (initial_factors.1 - k, initial_factors.2 - k) →
    initial_number * (2 ^ (initial_factors.1 + k)) * (3 ^ (initial_factors.2 + k)) ≠ 54) :=
by
  sorry

end not_54_after_one_hour_l210_21094


namespace arc_length_of_circle_l210_21037

theorem arc_length_of_circle (r θ : ℝ) (h1 : r = 2) (h2 : θ = 5 * Real.pi / 3) : (θ * r) = 10 * Real.pi / 3 :=
by
  rw [h1, h2]
  -- subsequent steps would go here 
  sorry

end arc_length_of_circle_l210_21037


namespace negation_of_proposition_l210_21098

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → (x + 1/x) ≥ 2

-- Define the negation of the original proposition
def negation_prop : Prop := ∃ x > 0, x + 1/x < 2

-- State that the negation of the original proposition is the stated negation
theorem negation_of_proposition : (¬ ∀ x, original_prop x) ↔ negation_prop := 
by sorry

end negation_of_proposition_l210_21098


namespace total_seashells_l210_21009

-- Define the conditions from part a)
def unbroken_seashells : ℕ := 2
def broken_seashells : ℕ := 4

-- Define the proof problem
theorem total_seashells :
  unbroken_seashells + broken_seashells = 6 :=
by
  sorry

end total_seashells_l210_21009


namespace math_proof_l210_21056

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end math_proof_l210_21056


namespace product_of_reciprocals_plus_one_geq_nine_l210_21017

theorem product_of_reciprocals_plus_one_geq_nine
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hab : a + b = 1) :
  (1 / a + 1) * (1 / b + 1) ≥ 9 :=
sorry

end product_of_reciprocals_plus_one_geq_nine_l210_21017


namespace valid_parameterizations_l210_21018

def point_on_line (x y : ℝ) : Prop := (y = 2 * x - 5)

def direction_vector_valid (vx vy : ℝ) : Prop := (∃ (k : ℝ), vx = k * 1 ∧ vy = k * 2)

def parametric_option_valid (px py vx vy : ℝ) : Prop := 
  point_on_line px py ∧ direction_vector_valid vx vy

theorem valid_parameterizations : 
  (parametric_option_valid 10 15 5 10) ∧ 
  (parametric_option_valid 3 1 0.5 1) ∧ 
  (parametric_option_valid 7 9 2 4) ∧ 
  (parametric_option_valid 0 (-5) 10 20) :=
  by sorry

end valid_parameterizations_l210_21018


namespace percent_flamingos_among_non_parrots_l210_21022

theorem percent_flamingos_among_non_parrots
  (total_birds : ℝ) (flamingos : ℝ) (parrots : ℝ) (eagles : ℝ) (owls : ℝ)
  (h_total : total_birds = 100)
  (h_flamingos : flamingos = 40)
  (h_parrots : parrots = 20)
  (h_eagles : eagles = 15)
  (h_owls : owls = 25) :
  ((flamingos / (total_birds - parrots)) * 100 = 50) :=
by sorry

end percent_flamingos_among_non_parrots_l210_21022


namespace no_real_roots_of_quadratic_l210_21089

theorem no_real_roots_of_quadratic (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b ≠ 0) ↔ ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry

end no_real_roots_of_quadratic_l210_21089


namespace BoatWorks_total_canoes_by_April_l210_21058

def BoatWorksCanoes : ℕ → ℕ
| 0 => 5
| (n+1) => 2 * BoatWorksCanoes n

theorem BoatWorks_total_canoes_by_April : (BoatWorksCanoes 0) + (BoatWorksCanoes 1) + (BoatWorksCanoes 2) + (BoatWorksCanoes 3) = 75 :=
by
  sorry

end BoatWorks_total_canoes_by_April_l210_21058


namespace find_angle_4_l210_21026

/-- Given angle conditions, prove that angle 4 is 22.5 degrees. -/
theorem find_angle_4 (angle : ℕ → ℝ) 
  (h1 : angle 1 + angle 2 = 180) 
  (h2 : angle 3 = angle 4) 
  (h3 : angle 1 = 85) 
  (h4 : angle 5 = 45) 
  (h5 : angle 1 + angle 5 + angle 6 = 180) : 
  angle 4 = 22.5 :=
sorry

end find_angle_4_l210_21026


namespace problem_proof_l210_21096

noncomputable def original_number_of_buses_and_total_passengers : Nat × Nat :=
  let k := 24
  let total_passengers := 529
  (k, total_passengers)

theorem problem_proof (k n : Nat) (h₁ : n = 22 + 23 / (k - 1)) (h₂ : 22 * k + 1 = n * (k - 1)) (h₃ : k ≥ 2) (h₄ : n ≤ 32) :
  (k, 22 * k + 1) = original_number_of_buses_and_total_passengers :=
by
  sorry

end problem_proof_l210_21096


namespace intersection_of_sets_l210_21021

noncomputable def universal_set (x : ℝ) := true

def set_A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

def set_B (x : ℝ) : Prop := ∃ y, y = Real.log (1 - x)

def complement_U_B (x : ℝ) : Prop := ¬ set_B x

theorem intersection_of_sets :
  { x : ℝ | set_A x } ∩ { x | complement_U_B x } = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_sets_l210_21021


namespace extreme_value_range_of_a_l210_21086

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) * (1 - a * x)

theorem extreme_value_range_of_a (a : ℝ) :
  a ∈ Set.Ioo (2 / 3 : ℝ) 2 ↔
    ∃ c ∈ Set.Ioo 0 1, ∀ x : ℝ, f a c = f a x :=
by
  sorry

end extreme_value_range_of_a_l210_21086


namespace range_of_a_l210_21047

theorem range_of_a 
  (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) 
  : -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l210_21047


namespace triangle_B_eq_2A_range_of_a_l210_21079

theorem triangle_B_eq_2A (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = c) : B = 2 * A := 
sorry

theorem range_of_a (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = 2) (h6 : 0 < (π - A - B)) (h7 : (π - A - B) < π/2) : 1 < a ∧ a < 2 := 
sorry

end triangle_B_eq_2A_range_of_a_l210_21079


namespace tim_initial_soda_l210_21001

-- Define the problem
def initial_cans (x : ℕ) : Prop :=
  let after_jeff_takes := x - 6
  let after_buying_more := after_jeff_takes + after_jeff_takes / 2
  after_buying_more = 24

-- Theorem stating the problem in Lean 4
theorem tim_initial_soda (x : ℕ) (h: initial_cans x) : x = 22 :=
by
  sorry

end tim_initial_soda_l210_21001


namespace rachel_budget_proof_l210_21042

-- Define the prices Sara paid for shoes and the dress
def shoes_price : ℕ := 50
def dress_price : ℕ := 200

-- Total amount Sara spent
def sara_total : ℕ := shoes_price + dress_price

-- Rachel's budget should be double of Sara's total spending
def rachels_budget : ℕ := 2 * sara_total

-- The theorem statement
theorem rachel_budget_proof : rachels_budget = 500 := by
  unfold rachels_budget sara_total shoes_price dress_price
  rfl

end rachel_budget_proof_l210_21042


namespace arithmetic_mean_is_one_l210_21029

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) : 
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 :=
by
  sorry

end arithmetic_mean_is_one_l210_21029


namespace multiplication_even_a_b_multiplication_even_a_a_l210_21060

def a : Int := 4
def b : Int := 3

theorem multiplication_even_a_b : a * b = 12 := by sorry
theorem multiplication_even_a_a : a * a = 16 := by sorry

end multiplication_even_a_b_multiplication_even_a_a_l210_21060


namespace simplify_expression_l210_21048

theorem simplify_expression :
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) := by
  sorry

end simplify_expression_l210_21048


namespace cost_of_pen_is_five_l210_21071

-- Define the given conditions
def pencils_per_box := 80
def num_boxes := 15
def total_pencils := num_boxes * pencils_per_box
def cost_per_pencil := 4
def total_cost_of_stationery := 18300
def additional_pens := 300
def num_pens := 2 * total_pencils + additional_pens

-- Calculate total cost of pencils
def total_cost_of_pencils := total_pencils * cost_per_pencil

-- Calculate total cost of pens
def total_cost_of_pens := total_cost_of_stationery - total_cost_of_pencils

-- The conjecture to prove
theorem cost_of_pen_is_five :
  (total_cost_of_pens / num_pens) = 5 :=
sorry

end cost_of_pen_is_five_l210_21071


namespace number_of_triangles_l210_21077

theorem number_of_triangles (points : List ℝ) (h₀ : points.length = 12)
  (h₁ : ∀ p ∈ points, p ≠ A ∧ p ≠ B ∧ p ≠ C ∧ p ≠ D): 
  (∃ triangles : ℕ, triangles = 216) :=
  sorry

end number_of_triangles_l210_21077


namespace smallest_number_l210_21027

theorem smallest_number (a b c d : ℝ) (h1 : a = -5) (h2 : b = 0) (h3 : c = 1/2) (h4 : d = Real.sqrt 2) : a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by
  sorry

end smallest_number_l210_21027


namespace product_of_coefficients_is_negative_integer_l210_21006

theorem product_of_coefficients_is_negative_integer
  (a b c : ℤ)
  (habc_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (discriminant_positive : (b * b - 4 * a * c) > 0)
  (product_cond : a * b * c = (c / a)) :
  ∃ k : ℤ, k < 0 ∧ k = a * b * c :=
by
  sorry

end product_of_coefficients_is_negative_integer_l210_21006


namespace anika_age_l210_21073

/-- Given:
 1. Anika is 10 years younger than Clara.
 2. Clara is 5 years older than Ben.
 3. Ben is 20 years old.
 Prove:
 Anika's age is 15 years.
 -/
theorem anika_age (Clara Anika Ben : ℕ) 
  (h1 : Anika = Clara - 10) 
  (h2 : Clara = Ben + 5) 
  (h3 : Ben = 20) : Anika = 15 := 
by
  sorry

end anika_age_l210_21073


namespace number_of_common_tangents_l210_21003

/-- Define the circle C1 with center (2, -1) and radius 2. -/
def C1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4}

/-- Define the symmetry line x + y - 3 = 0. -/
def symmetry_line := {p : ℝ × ℝ | p.1 + p.2 = 3}

/-- Circle C2 is symmetric to C1 about the line x + y = 3. -/
def C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 - 1)^2 = 4}

/-- Circle C3 with the given condition MA^2 + MO^2 = 10 for any point M on the circle. 
    A(0, 2) and O is the origin. -/
def C3 := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 4}

/-- The number of common tangents between circle C2 and circle C3 is 3. -/
theorem number_of_common_tangents
  (C1_sym_C2 : ∀ p : ℝ × ℝ, p ∈ C1 ↔ p ∈ C2)
  (M_on_C3 : ∀ M : ℝ × ℝ, M ∈ C3 → ((M.1)^2 + (M.2 - 2)^2) + ((M.1)^2 + (M.2)^2) = 10) :
  ∃ tangents : ℕ, tangents = 3 :=
sorry

end number_of_common_tangents_l210_21003


namespace sum_of_squares_positive_l210_21091

theorem sum_of_squares_positive (x_1 x_2 k : ℝ) (h : x_1 ≠ x_2) 
  (hx1 : x_1^2 + 2*x_1 - k = 0) (hx2 : x_2^2 + 2*x_2 - k = 0) :
  x_1^2 + x_2^2 > 0 :=
by
  sorry

end sum_of_squares_positive_l210_21091


namespace total_surface_area_of_cubes_aligned_side_by_side_is_900_l210_21019

theorem total_surface_area_of_cubes_aligned_side_by_side_is_900 :
  let volumes := [27, 64, 125, 216, 512]
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  (surface_areas.sum = 900) :=
by
  sorry

end total_surface_area_of_cubes_aligned_side_by_side_is_900_l210_21019


namespace common_ratio_geometric_series_l210_21028

theorem common_ratio_geometric_series {a r S : ℝ} (h₁ : S = (a / (1 - r))) (h₂ : (ar^4 / (1 - r)) = S / 64) (h₃ : S ≠ 0) : r = 1 / 2 :=
sorry

end common_ratio_geometric_series_l210_21028


namespace find_large_monkey_doll_cost_l210_21043

-- Define the conditions and the target property
def large_monkey_doll_cost (L : ℝ) (condition1 : 300 / (L - 2) = 300 / L + 25)
                           (condition2 : 300 / (L + 1) = 300 / L - 15) : Prop :=
  L = 6

-- The main theorem with the conditions
theorem find_large_monkey_doll_cost (L : ℝ)
  (h1 : 300 / (L - 2) = 300 / L + 25)
  (h2 : 300 / (L + 1) = 300 / L - 15) : large_monkey_doll_cost L h1 h2 :=
  sorry

end find_large_monkey_doll_cost_l210_21043


namespace find_y_l210_21008

variables (x y : ℝ)

theorem find_y (h1 : x = 103) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 515400) : y = 1 / 2 :=
sorry

end find_y_l210_21008


namespace second_quadrant_condition_l210_21075

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ -270 < α ∧ α < -180

theorem second_quadrant_condition (α : ℝ) : 
  (is_obtuse α → is_in_second_quadrant α) ∧ ¬(is_in_second_quadrant α → is_obtuse α) := 
by
  sorry

end second_quadrant_condition_l210_21075


namespace first_payment_amount_l210_21038

-- The number of total payments
def total_payments : Nat := 65

-- The number of the first payments
def first_payments : Nat := 20

-- The number of remaining payments
def remaining_payments : Nat := total_payments - first_payments

-- The extra amount added to the remaining payments
def extra_amount : Int := 65

-- The average payment
def average_payment : Int := 455

-- The total amount paid over the year
def total_amount_paid : Int := average_payment * total_payments

-- The variable we want to solve for: amount of each of the first 20 payments
variable (x : Int)

-- The equation for total amount paid
def total_payments_equation : Prop :=
  20 * x + 45 * (x + 65) = 455 * 65

-- The theorem stating the amount of each of the first 20 payments
theorem first_payment_amount : x = 410 :=
  sorry

end first_payment_amount_l210_21038


namespace disproving_equation_l210_21049

theorem disproving_equation 
  (a b c d : ℚ)
  (h : a / b = c / d)
  (ha : a ≠ 0)
  (hc : c ≠ 0) : 
  a + d ≠ (a / b) * (b + c) := 
by 
  sorry

end disproving_equation_l210_21049


namespace cuboid_can_form_square_projection_l210_21052

-- Definitions and conditions based directly on the problem
def length1 := 3
def length2 := 4
def length3 := 6

-- Statement to prove
theorem cuboid_can_form_square_projection (x y : ℝ) :
  (4 * x * x + y * y = 36) ∧ (x + y = 4) → True :=
by sorry

end cuboid_can_form_square_projection_l210_21052


namespace sufficient_not_necessary_condition_l210_21076

-- Definition of the conditions
def Q (x : ℝ) : Prop := x^2 - x - 2 > 0
def P (x a : ℝ) : Prop := |x| > a

-- Main statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, P x a → Q x) → a ≥ 2 :=
by
  sorry

end sufficient_not_necessary_condition_l210_21076


namespace new_profit_is_220_percent_l210_21069

noncomputable def cost_price (CP : ℝ) : ℝ := 100

def initial_profit_percentage : ℝ := 60

noncomputable def initial_selling_price (CP : ℝ) : ℝ :=
  CP + (initial_profit_percentage / 100) * CP

noncomputable def new_selling_price (SP : ℝ) : ℝ :=
  2 * SP

noncomputable def new_profit_percentage (CP SP2 : ℝ) : ℝ :=
  ((SP2 - CP) / CP) * 100

theorem new_profit_is_220_percent : 
  new_profit_percentage (cost_price 100) (new_selling_price (initial_selling_price (cost_price 100))) = 220 :=
by
  sorry

end new_profit_is_220_percent_l210_21069


namespace birthday_check_value_l210_21046

theorem birthday_check_value : 
  ∃ C : ℝ, (150 + C) / 4 = C ↔ C = 50 :=
by
  sorry

end birthday_check_value_l210_21046


namespace power_function_value_at_two_l210_21082

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_value_at_two (a : ℝ) (h : f (1/2) a = 8) : f 2 a = 1 / 8 := by
  sorry

end power_function_value_at_two_l210_21082


namespace parallel_lines_necessary_not_sufficient_l210_21081

theorem parallel_lines_necessary_not_sufficient {a : ℝ} 
  (h1 : ∀ x y : ℝ, a * x + (a + 2) * y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + a * y + 2 = 0) 
  (h3 : ∀ x y : ℝ, a * (1 * y + 2) = 1 * (a * y + 2)) : 
  (a = -1) -> (a = 2 ∨ a = -1 ∧ ¬(∀ b, a = b → a = -1)) :=
by
  -- proof goes here
  sorry

end parallel_lines_necessary_not_sufficient_l210_21081


namespace geometric_sequence_common_ratio_l210_21074

theorem geometric_sequence_common_ratio
  (q a_1 : ℝ)
  (h1: a_1 * q = 1)
  (h2: a_1 + a_1 * q^2 = -2) :
  q = -1 :=
by
  sorry

end geometric_sequence_common_ratio_l210_21074


namespace webinar_active_minutes_l210_21061

theorem webinar_active_minutes :
  let hours := 13
  let extra_minutes := 17
  let break_minutes := 22
  (hours * 60 + extra_minutes) - break_minutes = 775 := by
  sorry

end webinar_active_minutes_l210_21061


namespace irrational_roots_of_odd_quadratic_l210_21010

theorem irrational_roots_of_odd_quadratic (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ gcd p q = 1 ∧ p * p = a * (p / q) * (p / q) + b * (p / q) + c := sorry

end irrational_roots_of_odd_quadratic_l210_21010


namespace bangles_per_box_l210_21053

-- Define the total number of pairs of bangles
def totalPairs : Nat := 240

-- Define the number of boxes
def numberOfBoxes : Nat := 20

-- Define the proof that each box can hold 24 bangles
theorem bangles_per_box : (totalPairs * 2) / numberOfBoxes = 24 :=
by
  -- Here we're required to do the proof but we'll use 'sorry' to skip it
  sorry

end bangles_per_box_l210_21053


namespace intersection_of_A_and_B_l210_21031

def A := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := { x : ℝ | -1 < x ∧ x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

end intersection_of_A_and_B_l210_21031


namespace cucumbers_count_l210_21080

theorem cucumbers_count:
  ∀ (C T : ℕ), C + T = 420 ∧ T = 4 * C → C = 84 :=
by
  intros C T h
  sorry

end cucumbers_count_l210_21080


namespace dinner_cakes_today_6_l210_21070

-- Definitions based on conditions
def lunch_cakes_today : ℕ := 5
def dinner_cakes_today (x : ℕ) : ℕ := x
def yesterday_cakes : ℕ := 3
def total_cakes_served : ℕ := 14

-- Lean statement to prove the mathematical equivalence
theorem dinner_cakes_today_6 (x : ℕ) (h : lunch_cakes_today + dinner_cakes_today x + yesterday_cakes = total_cakes_served) : x = 6 :=
by {
  sorry -- Proof to be completed.
}

end dinner_cakes_today_6_l210_21070


namespace percentage_reduction_is_10_percent_l210_21062

-- Definitions based on the given conditions
def rooms_rented_for_40 : ℕ := sorry
def rooms_rented_for_60 : ℕ := sorry
def total_rent : ℕ := 2000
def rent_per_room_40 : ℕ := 40
def rent_per_room_60 : ℕ := 60
def rooms_switch_count : ℕ := 10

-- Define the hypothetical new total if the rooms were rented at different rates
def new_total_rent : ℕ := (rent_per_room_40 * (rooms_rented_for_40 + rooms_switch_count)) + (rent_per_room_60 * (rooms_rented_for_60 - rooms_switch_count))

-- Calculate the percentage reduction
noncomputable def percentage_reduction : ℝ := (((total_rent: ℝ) - (new_total_rent: ℝ)) / (total_rent: ℝ)) * 100

-- Statement to prove
theorem percentage_reduction_is_10_percent : percentage_reduction = 10 := by
  sorry

end percentage_reduction_is_10_percent_l210_21062


namespace find_divisor_exists_four_numbers_in_range_l210_21023

theorem find_divisor_exists_four_numbers_in_range :
  ∃ n : ℕ, (n > 1) ∧ (∀ k : ℕ, 39 ≤ k ∧ k ≤ 79 → ∃ a : ℕ, k = n * a) ∧ (∃! (k₁ k₂ k₃ k₄ : ℕ), 39 ≤ k₁ ∧ k₁ ≤ 79 ∧ 39 ≤ k₂ ∧ k₂ ≤ 79 ∧ 39 ≤ k₃ ∧ k₃ ≤ 79 ∧ 39 ≤ k₄ ∧ k₄ ≤ 79 ∧ k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧ k₁ % n = 0 ∧ k₂ % n = 0 ∧ k₃ % n = 0 ∧ k₄ % n = 0) → n = 19 :=
by sorry

end find_divisor_exists_four_numbers_in_range_l210_21023


namespace circus_tent_capacity_l210_21034

theorem circus_tent_capacity (num_sections : ℕ) (people_per_section : ℕ) 
  (h1 : num_sections = 4) (h2 : people_per_section = 246) :
  num_sections * people_per_section = 984 :=
by
  sorry

end circus_tent_capacity_l210_21034


namespace difference_of_squares_is_40_l210_21090

theorem difference_of_squares_is_40 {x y : ℕ} (h1 : x + y = 20) (h2 : x * y = 99) (hx : x > y) : x^2 - y^2 = 40 :=
sorry

end difference_of_squares_is_40_l210_21090


namespace find_m_l210_21020

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m + 1) * x + m = 0}

theorem find_m (m : ℝ) : B m ⊆ A → (m = 1 ∨ m = 2) :=
sorry

end find_m_l210_21020


namespace formation_enthalpy_benzene_l210_21000

/-- Define the enthalpy changes based on given conditions --/
def ΔH_acetylene : ℝ := 226.7 -- kJ/mol for C₂H₂
def ΔH_benzene_formation : ℝ := 631.1 -- kJ for reactions forming C₆H₆
def ΔH_benzene_phase_change : ℝ := -33.9 -- kJ for phase change of C₆H₆

/-- Define the enthalpy change of formation for benzene --/
def ΔH_formation_benzene : ℝ := 3 * ΔH_acetylene + ΔH_benzene_formation + ΔH_benzene_phase_change

/-- Theorem stating the heat change in the reaction equals the calculated value --/
theorem formation_enthalpy_benzene :
  ΔH_formation_benzene = -82.9 :=
by
  sorry

end formation_enthalpy_benzene_l210_21000


namespace ab_squared_ab_cubed_ab_power_n_l210_21057

-- Definitions of a and b as real numbers, and n as a natural number
variables (a b : ℝ) (n : ℕ)

theorem ab_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by 
  sorry

theorem ab_cubed (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by 
  sorry

theorem ab_power_n (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by 
  sorry

end ab_squared_ab_cubed_ab_power_n_l210_21057


namespace intersection_A_B_union_B_complement_A_l210_21054

open Set

variable (U A B : Set ℝ)

noncomputable def U_def : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
noncomputable def A_def : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def B_def : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_A_B : (A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
sorry

theorem union_B_complement_A : B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)} :=
sorry

attribute [irreducible] U_def A_def B_def

end intersection_A_B_union_B_complement_A_l210_21054


namespace bus_car_ratio_l210_21088

variable (R C Y : ℝ)

noncomputable def ratio_of_bus_to_car (R C Y : ℝ) : ℝ :=
  R / C

theorem bus_car_ratio 
  (h1 : R = 48) 
  (h2 : Y = 3.5 * C) 
  (h3 : Y = R - 6) : 
  ratio_of_bus_to_car R C Y = 4 :=
by sorry

end bus_car_ratio_l210_21088


namespace betty_needs_five_boxes_l210_21063

def betty_oranges (total_oranges first_box second_box max_per_box : ℕ) : ℕ :=
  let remaining_oranges := total_oranges - (first_box + second_box)
  let full_boxes := remaining_oranges / max_per_box
  let extra_box := if remaining_oranges % max_per_box == 0 then 0 else 1
  full_boxes + 2 + extra_box

theorem betty_needs_five_boxes :
  betty_oranges 120 30 25 30 = 5 := 
by
  sorry

end betty_needs_five_boxes_l210_21063


namespace abs_x_plus_2_l210_21068

theorem abs_x_plus_2 (x : ℤ) (h : x = -3) : |x + 2| = 1 :=
by sorry

end abs_x_plus_2_l210_21068


namespace total_metal_wasted_l210_21007

noncomputable def wasted_metal (a b : ℝ) (h : b ≤ 2 * a) : ℝ := 
  2 * a * b - (b ^ 2 / 2)

theorem total_metal_wasted (a b : ℝ) (h : b ≤ 2 * a) : 
  wasted_metal a b h = 2 * a * b - b ^ 2 / 2 :=
sorry

end total_metal_wasted_l210_21007


namespace minimum_sum_of_original_numbers_l210_21044

theorem minimum_sum_of_original_numbers 
  (m n : ℕ) 
  (h1 : m < n) 
  (h2 : 23 * m - 20 * n = 460) 
  (h3 : ∀ m n, 23 * m - 20 * n = 460 → m < n):
  m + n = 321 :=
sorry

end minimum_sum_of_original_numbers_l210_21044


namespace brainiacs_like_neither_l210_21032

variables 
  (total : ℕ) -- Total number of brainiacs.
  (R : ℕ) -- Number of brainiacs who like rebus teasers.
  (M : ℕ) -- Number of brainiacs who like math teasers.
  (both : ℕ) -- Number of brainiacs who like both rebus and math teasers.
  (math_only : ℕ) -- Number of brainiacs who like only math teasers.

-- Given conditions in the problem
def twice_as_many_rebus : Prop := R = 2 * M
def both_teasers : Prop := both = 18
def math_teasers_not_rebus : Prop := math_only = 20
def total_brainiacs : Prop := total = 100

noncomputable def exclusion_inclusion : ℕ := R + M - both

-- Proof statement: The number of brainiacs who like neither rebus nor math teasers totals to 4
theorem brainiacs_like_neither
  (h_total : total_brainiacs total)
  (h_twice : twice_as_many_rebus R M)
  (h_both : both_teasers both)
  (h_math_only : math_teasers_not_rebus math_only)
  (h_M : M = both + math_only) :
  total - exclusion_inclusion R M both = 4 :=
sorry

end brainiacs_like_neither_l210_21032


namespace ab_leq_one_fraction_inequality_l210_21065

-- Part 1: Prove that ab ≤ 1
theorem ab_leq_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) : a * b ≤ 1 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

-- Part 2: Prove that (1/a^3 - 1/b^3) > 3 * (1/a - 1/b) given b > a
theorem fraction_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) (h4 : b > a) :
  1/(a^3) - 1/(b^3) > 3 * (1/a - 1/b) :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end ab_leq_one_fraction_inequality_l210_21065


namespace storm_deposit_l210_21097

theorem storm_deposit (C : ℝ) (original_amount after_storm_rate before_storm_rate : ℝ) (after_storm full_capacity : ℝ) :
  before_storm_rate = 0.40 →
  after_storm_rate = 0.60 →
  original_amount = 220 * 10^9 →
  before_storm_rate * C = original_amount →
  C = full_capacity →
  after_storm = after_storm_rate * full_capacity →
  after_storm - original_amount = 110 * 10^9 :=
by
  sorry

end storm_deposit_l210_21097


namespace max_reflections_l210_21015

theorem max_reflections (n : ℕ) (angle_CDA : ℝ) (h_angle : angle_CDA = 12) : n ≤ 7 ↔ 12 * n ≤ 90 := by
    sorry

end max_reflections_l210_21015


namespace find_common_difference_l210_21014

-- Definitions of the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a_n (k + 1) = a_n k + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : Prop :=
  S_n = (n : ℝ) / 2 * (a_n 1 + a_n n)

variables {a_1 d : ℝ}
variables (a_n : ℕ → ℝ)
variables (S_3 S_9 : ℝ)

-- Conditions from the problem statement
axiom a2_eq_3 : a_n 2 = 3
axiom S9_eq_6S3 : S_9 = 6 * S_3

-- The proof we need to write
theorem find_common_difference 
  (h1 : arithmetic_sequence a_n d)
  (h2 : sum_of_first_n_terms a_n 3 S_3)
  (h3 : sum_of_first_n_terms a_n 9 S_9) :
  d = 1 :=
by
  sorry

end find_common_difference_l210_21014


namespace value_of_expression_l210_21005

theorem value_of_expression (x : ℕ) (h : x = 8) : 
  (x^3 + 3 * (x^2) * 2 + 3 * x * (2^2) + 2^3 = 1000) := by
{
  sorry
}

end value_of_expression_l210_21005


namespace car_speed_l210_21035

theorem car_speed
  (v : ℝ)       -- the unknown speed of the car in km/hr
  (time_80 : ℝ := 45)  -- the time in seconds to travel 1 km at 80 km/hr
  (time_plus_10 : ℝ := 55)  -- the time in seconds to travel 1 km at speed v

  (h1 : time_80 = 3600 / 80)
  (h2 : time_plus_10 = time_80 + 10) :
  v = 3600 / (55 / 3600) := sorry

end car_speed_l210_21035


namespace sin_identity_l210_21030

theorem sin_identity {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.sin (π / 6 - 2 * α) = -7 / 8 := 
by 
  sorry

end sin_identity_l210_21030


namespace find_minimum_value_max_value_when_g_half_l210_21040

noncomputable def f (a x : ℝ) : ℝ := 1 - 2 * a - 2 * a * (Real.cos x) - 2 * (Real.sin x) ^ 2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a <= 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem find_minimum_value (a : ℝ) :
  ∃ g_val, g_val = g a :=
  sorry

theorem max_value_when_g_half : 
  g (-1) = 1 / 2 →
  ∃ max_val, max_val = (max (f (-1) π) (f (-1) 0)) :=
  sorry

end find_minimum_value_max_value_when_g_half_l210_21040


namespace harvesting_days_l210_21099

theorem harvesting_days :
  (∀ (harvesters : ℕ) (days : ℕ) (mu : ℕ), 2 * 3 * (75 : ℕ) = 450) →
  (7 * 4 * (75 : ℕ) = 2100) :=
by
  sorry

end harvesting_days_l210_21099


namespace fifteen_percent_of_x_l210_21013

variables (x : ℝ)

-- Condition: Given x% of 60 is 12
def is_x_percent_of_60 : Prop := (x / 100) * 60 = 12

-- Prove: 15% of x is 3
theorem fifteen_percent_of_x (h : is_x_percent_of_60 x) : (15 / 100) * x = 3 :=
by
  sorry

end fifteen_percent_of_x_l210_21013


namespace lines_slope_angle_l210_21025

theorem lines_slope_angle (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : L1 = fun x => m * x)
  (h2 : L2 = fun x => n * x)
  (h3 : θ₁ = 3 * θ₂)
  (h4 : m = 3 * n)
  (h5 : θ₂ ≠ 0) :
  m * n = 9 / 4 :=
by
  sorry

end lines_slope_angle_l210_21025
