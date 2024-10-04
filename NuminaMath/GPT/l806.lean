import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.GeometricSeries
import Mathlib.Algebra.Group.Default
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Polynomial.Tactic
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecFunc.Exponential
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Logarithm
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Catalan
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.PrimeNormNum
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Logic.Basic
import Mathlib.Probability.Distributions
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import data.real.basic

namespace lg_inequality_solution_l806_806338

theorem lg_inequality_solution (a b : ℝ) (x : ℝ) (h_a_gt_1 : a > 1) (h_b_lt_1 : b < 1) (h_b_gt_0 : b > 0) (h_a_minus_b_eq_1 : a - b = 1) :
  (\lg (a ^ x - b ^ x) > 0) ↔ (x ∈ Set.Ioi 1) :=
sorry

end lg_inequality_solution_l806_806338


namespace sum_first_twelve_terms_of_arithmetic_sequence_l806_806269

theorem sum_first_twelve_terms_of_arithmetic_sequence :
    let a1 := -3
    let a12 := 48
    let n := 12
    let Sn := (n * (a1 + a12)) / 2
    Sn = 270 := 
by
  sorry

end sum_first_twelve_terms_of_arithmetic_sequence_l806_806269


namespace probability_question_l806_806310

-- Define the quadratic equation
def quadratic_eq_has_distinct_real_roots (k : ℕ) : Prop :=
  k^2 - 8 > 0

-- Define the set of possible die outcomes
def die_roll_outcomes : set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the event that a rolled number gives distinct real roots
def event_distinct_real_roots : set ℕ :=
  {k ∈ die_roll_outcomes | quadratic_eq_has_distinct_real_roots k}

-- Noncomputable definition to calculate probability
noncomputable def probability_event (E : set ℕ) (Ω : set ℕ) : ℚ :=
  (E.card : ℚ) / (Ω.card : ℚ)

-- Lean statement for the theorem
theorem probability_question :
probaility_event event_distinct_real_roots die_roll_outcomes = 2/3 := sorry

end probability_question_l806_806310


namespace sum_first_n_terms_b_l806_806334

def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := 
  if n = 1 then 3 else S n - S (n-1) 

def q : ℤ := 2

def b (n : ℕ) : ℤ := 
  (if q = 2 then 
     (3 / q) * q ^ (n-1) 
   else 
     (-3 / q) * q ^ (n-1))

def T (n : ℕ) : ℤ := 
  if q = 2 then 
    (3 / 2 : ℚ) * (2^n - 1)
  else
    (1 / 2 : ℚ) * ((-2)^n - 1)

-- Lean statement to be proved:
theorem sum_first_n_terms_b {n : ℕ} (hb2 : b 2 = 3) (hb4 : b 4 = 12) : T n = if q = 2 then (3 / 2 : ℚ) * (2^n - 1) else (1 / 2 : ℚ) * ((-2)^n - 1) :=
by 
  sorry

end sum_first_n_terms_b_l806_806334


namespace constants_correct_l806_806745

noncomputable def problem_constants (P Q R : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → 
    5 * x^2 / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2 

theorem constants_correct :
  problem_constants 20 (-15) (-10) :=
by
  intro x h,
  sorry

end constants_correct_l806_806745


namespace quadratic_distinct_real_roots_l806_806445

theorem quadratic_distinct_real_roots (k : ℝ) : k < 1 / 2 ∧ k ≠ 0 ↔ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (k * x1^2 - 2 * x1 + 2 = 0) ∧ (k * x2^2 - 2 * x2 + 2 = 0)) := 
by 
  sorry

end quadratic_distinct_real_roots_l806_806445


namespace proof_of_problem_l806_806971

def problem_statement (n : ℤ) : Prop :=
  (2 ^ (3 * n + 3) - 7 * n + 41) % 49 = 0

theorem proof_of_problem : ∀ n : ℤ, problem_statement n := 
by
  sorry

end proof_of_problem_l806_806971


namespace vector_equation_solution_not_collinear_l806_806784

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C O : V)
variables (OA OB OC : V)

theorem vector_equation_solution_not_collinear
  (h_not_collinear : ¬ collinear ℝ ({A, B, C} : Set V))
  (h_vector_eq : 16 • OA - 12 • OB - 3 • OC = (0 : V)) :
  OA = 12 • (OB - OA) + 3 • (OC - OA) :=
sorry

end vector_equation_solution_not_collinear_l806_806784


namespace number_of_boxes_on_pallet_l806_806687

-- Define the total weight of the pallet.
def total_weight_of_pallet : ℤ := 267

-- Define the weight of each box.
def weight_of_each_box : ℤ := 89

-- The theorem states that given the total weight of the pallet and the weight of each box,
-- the number of boxes on the pallet is 3.
theorem number_of_boxes_on_pallet : total_weight_of_pallet / weight_of_each_box = 3 :=
by sorry

end number_of_boxes_on_pallet_l806_806687


namespace find_sides_and_radius_l806_806661

variables (K L M N Q : Type)
variables (R : ℝ)
variables [InnerProductSpace ℝ (K L M N : Type)]

-- Definitions for the problem setup
def is_cyclic_quadrilateral (K L M N : Type) (R : ℝ) : Prop :=
  ∃c : ℝ, ∀ p ∈ {K, L, M, N}, dist p c = R

def intersection_of_diagonals (Q : Type) (K L M N : Type) : Prop :=
  ∃P : K → L → M → N, P K L = P M N

def height_from_L_to_KN (L : Type) (KN : ℝ) : ℝ := 6

def sides_sum (KN LM : ℝ) : Prop :=
  KN + LM = 24

def area_of_LMQ (LMQ : ℝ) : ℝ := 2

theorem find_sides_and_radius 
  (K L M N : Type) 
  (is_cyclic_quadrilateral KN) 
  (intersection_of_diagonals Q) 
  (h : height_from_L_to_KN L = 6) 
  (s : sides_sum KN LM) 
  (area : area_of_LMQ 2) :
  KN = 20 ∧ LM = 4 ∧ R = 5 * sqrt 5 :=
  sorry

end find_sides_and_radius_l806_806661


namespace line_passes_through_fixed_point_l806_806603

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧ y = k * (x - 1) + 2 :=
by
  intro k
  use (1, 2)
  simp
  split
  rfl
  split
  rfl
  ring
  sorry

end line_passes_through_fixed_point_l806_806603


namespace fruit_weights_l806_806936

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806936


namespace angle_between_vectors_orthogonal_l806_806831

theorem angle_between_vectors_orthogonal 
  (α β : ℝ^3)
  (hα : α ≠ 0)
  (hβ : β ≠ 0)
  (h : ∥α + β∥ = ∥α - β∥) 
  : angle α β = (π / 2) :=
by
  sorry

end angle_between_vectors_orthogonal_l806_806831


namespace powerjet_pumps_315_gallons_l806_806141

noncomputable def gallons_per_hour := 420
noncomputable def minutes_in_an_hour := 60
noncomputable def time_in_minutes := 45

theorem powerjet_pumps_315_gallons :
  let time_fraction := (time_in_minutes : ℚ) / minutes_in_an_hour,
      gallons_pumped := gallons_per_hour * time_fraction in
  gallons_pumped = 315 :=
by
  sorry

end powerjet_pumps_315_gallons_l806_806141


namespace exists_rectangle_with_same_color_l806_806287

theorem exists_rectangle_with_same_color (p : ℕ) (color : ℤ × ℤ → fin p) :
  ∃ (a b c d : ℤ × ℤ), 
    a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = c.2 ∧ b.2 = d.2 ∧ color a = color b ∧ color b = color c ∧ color c = color d :=
sorry

end exists_rectangle_with_same_color_l806_806287


namespace maximum_busses_l806_806858

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l806_806858


namespace translated_point_is_correct_l806_806369

-- Define the initial point P
def initial_point : ℝ × ℝ := (-5, 1)

-- Define the translation function for X and Y
def translate_x (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ := (p.1 + dx, p.2)
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ := (p.1, p.2 - dy)

-- Define the final translation
def translated_point : ℝ × ℝ :=
  let p' := translate_x initial_point 2 in
  translate_y p' 4

-- Prove that the translated point is (-3, -3)
theorem translated_point_is_correct :
  translated_point = (-3, -3) :=
sorry

end translated_point_is_correct_l806_806369


namespace fruit_weights_determined_l806_806947

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806947


namespace correct_calculation_l806_806641

theorem correct_calculation : ∀ (a : ℝ), a^2 * a^4 = a^6 :=
by {
  intro a,
  rw [←pow_add],
  rw [pow_two, pow_four],
  sorry
}

end correct_calculation_l806_806641


namespace least_four_digit_palindrome_divisible_by_5_l806_806114

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString in str = str.reverse

def is_divisible_by_5 (n: ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem least_four_digit_palindrome_divisible_by_5 : 
  ∃ n, is_palindrome n ∧ is_four_digit n ∧ is_divisible_by_5 n ∧ ∀ m, is_palindrome m ∧ is_four_digit m ∧ is_divisible_by_5 m → n ≤ m := 
sorry

end least_four_digit_palindrome_divisible_by_5_l806_806114


namespace balls_in_boxes_l806_806815

theorem balls_in_boxes :
  let balls := 6
  let boxes := 4
  (number_of_ways_to_distribute_balls (balls) (boxes) = 84) :=
by
  sorry

end balls_in_boxes_l806_806815


namespace garden_comparison_l806_806060

noncomputable def area (length : ℕ) (width : ℕ) : ℕ := length * width
noncomputable def perimeter (length : ℕ) (width : ℕ) : ℕ := 2 * (length + width)
noncomputable def square_area (side : ℕ) : ℕ := side * side
noncomputable def square_perimeter (side : ℕ) : ℕ := 4 * side

variables (karl_length : ℕ := 30) (karl_width : ℕ := 40) (makenna_side : ℕ := 35)

theorem garden_comparison :
  let karl_area := area karl_length karl_width,
      makenna_area := square_area makenna_side,
      karl_perimeter := perimeter karl_length karl_width,
      makenna_perimeter := square_perimeter makenna_side in
  karl_area + 25 = makenna_area ∧ karl_perimeter = makenna_perimeter :=
by
  sorry

end garden_comparison_l806_806060


namespace runway_trip_time_l806_806589

-- Define the conditions
def num_models := 6
def num_bathing_suit_outfits := 2
def num_evening_wear_outfits := 3
def total_time_minutes := 60

-- Calculate the total number of outfits per model
def total_outfits_per_model := num_bathing_suit_outfits + num_evening_wear_outfits

-- Calculate the total number of runway trips
def total_runway_trips := num_models * total_outfits_per_model

-- State the goal: Time per runway trip
def time_per_runway_trip := total_time_minutes / total_runway_trips

theorem runway_trip_time : time_per_runway_trip = 2 := by
  sorry

end runway_trip_time_l806_806589


namespace fruit_weights_l806_806938

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806938


namespace marble_problem_l806_806736

theorem marble_problem:
  (∀ (rx ry bx by x y : ℕ), 
    (rx + bx = x) ∧ 
    (ry + by = y) ∧ 
    (x + y = 34) ∧ 
    (rx * ry = 19 * x * y / 34) ∧ 
    (bx * by = 64) 
    → 64 + 289 = 353) :=
by
  intros rx ry bx by x y
  sorry

end marble_problem_l806_806736


namespace good_numbers_identification_l806_806837

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), 
    (∀ k : Fin n, ∃ m : ℕ, k.val + a k = m * m)

theorem good_numbers_identification : 
  { n : ℕ | ¬is_good_number n } = {1, 2, 4, 6, 7, 9, 11} :=
  sorry

end good_numbers_identification_l806_806837


namespace series_sum_eq_50_l806_806729

noncomputable def series_sum (x : ℝ) : ℝ :=
  2 + 6 * x + 10 * x^2 + 14 * x^3 -- This represents the series

theorem series_sum_eq_50 : 
  ∃ x : ℝ, series_sum x = 50 ∧ x = 0.59 :=
by
  sorry

end series_sum_eq_50_l806_806729


namespace smallest_z_value_l806_806040

theorem smallest_z_value :
  ∃ (x z : ℕ), (w = x - 2) ∧ (y = x + 2) ∧ (z = x + 4) ∧ ((x - 2)^3 + x^3 + (x + 2)^3 = (x + 4)^3) ∧ z = 2 := by
  sorry

end smallest_z_value_l806_806040


namespace revenue_and_empty_seats_l806_806160

-- Define seating and ticket prices
def seats_A : ℕ := 90
def seats_B : ℕ := 70
def seats_C : ℕ := 50
def VIP_seats : ℕ := 10

def ticket_A : ℕ := 15
def ticket_B : ℕ := 10
def ticket_C : ℕ := 5
def VIP_ticket : ℕ := 25

-- Define discounts
def discount : ℤ := 20

-- Define actual occupancy
def adults_A : ℕ := 35
def children_A : ℕ := 15
def adults_B : ℕ := 20
def seniors_B : ℕ := 5
def adults_C : ℕ := 10
def veterans_C : ℕ := 5
def VIP_occupied : ℕ := 10

-- Concession sales
def hot_dogs_sold : ℕ := 50
def hot_dog_price : ℕ := 4
def soft_drinks_sold : ℕ := 75
def soft_drink_price : ℕ := 2

-- Define the total revenue and empty seats calculation
theorem revenue_and_empty_seats :
  let revenue_from_tickets := (adults_A * ticket_A + children_A * ticket_A * (100 - discount) / 100 +
                               adults_B * ticket_B + seniors_B * ticket_B * (100 - discount) / 100 +
                               adults_C * ticket_C + veterans_C * ticket_C * (100 - discount) / 100 +
                               VIP_occupied * VIP_ticket)
  let revenue_from_concessions := (hot_dogs_sold * hot_dog_price + soft_drinks_sold * soft_drink_price)
  let total_revenue := revenue_from_tickets + revenue_from_concessions
  let empty_seats_A := seats_A - (adults_A + children_A)
  let empty_seats_B := seats_B - (adults_B + seniors_B)
  let empty_seats_C := seats_C - (adults_C + veterans_C)
  let empty_VIP_seats := VIP_seats - VIP_occupied
  total_revenue = 1615 ∧ empty_seats_A = 40 ∧ empty_seats_B = 45 ∧ empty_seats_C = 35 ∧ empty_VIP_seats = 0 := by
  sorry

end revenue_and_empty_seats_l806_806160


namespace apples_initial_total_l806_806180

theorem apples_initial_total :
  (∑ i in finset.range 11, (i + 1) * 10) + 340 = 1000 := by
sorry

end apples_initial_total_l806_806180


namespace correct_statements_BCD_l806_806200

theorem correct_statements_BCD :
  let r := linear_correlation_coefficient
  let data := [1, 3, 4, 5, 7, 9, 11, 16]
  let chi_squared := 3.937
  let critical_value := 3.841
  let total_students := 1500
  let sample_size := 100
  let male_in_sample := 55
  let female_students := total_students - (male_in_sample * (total_students / sample_size))
  (∃ r, (¬ (|r| → strong_linear_correlation))) ∧ -- condition A is (incorrectly) assumed false
  (percentile data 0.75 = 10) ∧ -- condition B correctly states the 75th percentile
  (chi_squared > critical_value ∧ error_probability ≤ 0.05) ∧ -- condition C correctly processes chi_squared and critical_value
  (female_students = 675) -- condition D correctly inferring number of females
  := by {
  sorry
}

end correct_statements_BCD_l806_806200


namespace max_buses_constraint_satisfied_l806_806841

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l806_806841


namespace fruit_weights_l806_806927

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806927


namespace count_perfect_squares_between_50_and_300_l806_806424

theorem count_perfect_squares_between_50_and_300 : 
  ∃ n, number_of_perfect_squares 50 300 = n ∧ n = 10 := 
sorry

end count_perfect_squares_between_50_and_300_l806_806424


namespace part_a_part_b_part_c_l806_806107

-- Part (a)
theorem part_a : 
  ∃ n : ℕ, n = 2023066 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (b)
theorem part_b : 
  ∃ n : ℕ, n = 1006 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x = y ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (c)
theorem part_c : 
  ∃ (x y z : ℕ), (x + y + z = 2013 ∧ (x * y * z = 671 * 671 * 671)) :=
sorry

end part_a_part_b_part_c_l806_806107


namespace length_GH_parallel_lines_l806_806635

theorem length_GH_parallel_lines (AB CD EF GH : ℝ) 
  (h1 : AB = 180) 
  (h2 : CD = 120) 
  (h3 : AB ∥ CD) 
  (h4 : CD ∥ EF) 
  (h5 : EF ∥ GH) : 
  GH = 72 :=
sorry

end length_GH_parallel_lines_l806_806635


namespace floor_width_is_120_l806_806973

def tile_length := 25 -- cm
def tile_width := 16 -- cm
def floor_length := 180 -- cm
def max_tiles := 54

theorem floor_width_is_120 :
  ∃ (W : ℝ), W = 120 ∧ (floor_length / tile_width) * W = max_tiles * (tile_length * tile_width) := 
sorry

end floor_width_is_120_l806_806973


namespace max_buses_l806_806861

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l806_806861


namespace altitude_contains_x_l806_806515

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806515


namespace intersecting_lines_l806_806158

theorem intersecting_lines (c d : ℝ) :
  (∀ x y : ℝ, (x = 1/3 * y + c ∧ y = 1/3 * x + d) → (x = 3 ∧ y = 3)) →
  c + d = 4 :=
by
  intros h
  -- We need to validate the condition holds at the intersection point
  have h₁ : 3 = 1/3 * 3 + c := by sorry
  have h₂ : 3 = 1/3 * 3 + d := by sorry
  -- Conclude that c = 2 and d = 2
  have hc : c = 2 := by sorry
  have hd : d = 2 := by sorry
  -- Thus the sum c + d = 4
  show 2 + 2 = 4 from rfl

end intersecting_lines_l806_806158


namespace Lincoln_County_houses_l806_806451

def original_houses : ℕ := 20817
def additional_houses : ℕ := 97741
def percentage_loss : ℝ := 0.18

theorem Lincoln_County_houses :
  original_houses + additional_houses - floor (percentage_loss * original_houses) = 114811 :=
by
  sorry

end Lincoln_County_houses_l806_806451


namespace count_four_digit_integers_with_5_or_7_l806_806415

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l806_806415


namespace largest_integer_odd_divides_expression_l806_806019

theorem largest_integer_odd_divides_expression (x : ℕ) (h_odd : x % 2 = 1) : 
    ∃ k, k = 384 ∧ ∀ m, m ∣ (8*x + 6) * (8*x + 10) * (4*x + 4) → m ≤ k :=
by {
  sorry
}

end largest_integer_odd_divides_expression_l806_806019


namespace four_digit_numbers_with_5_or_7_l806_806389

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l806_806389


namespace sphere_to_cube_ratio_cylinder_to_cube_ratio_l806_806248

variable (s : ℝ)

namespace InscribedVolumes

def sphere_volume := (4 / 3) * Real.pi * (s / 2) ^ 3
def cylinder_volume := Real.pi * (s / 2) ^ 2 * s
def cube_volume := s ^ 3

theorem sphere_to_cube_ratio : sphere_volume s / cube_volume s = Real.pi / 6 := by
  sorry

theorem cylinder_to_cube_ratio : cylinder_volume s / cube_volume s = Real.pi / 4 := by
  sorry

end InscribedVolumes

end sphere_to_cube_ratio_cylinder_to_cube_ratio_l806_806248


namespace num_terms_divisible_by_37_l806_806013

theorem num_terms_divisible_by_37 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1500) : 
  (card {k : ℕ | 1 ≤ k ∧ k ≤ 1500 ∧ (10 ^ k - 1) % 37 = 0}) = 499 :=
sorry

end num_terms_divisible_by_37_l806_806013


namespace number_of_perfect_squares_is_10_l806_806420

-- Define the number of integers n such that 50 ≤ n^2 ≤ 300
def count_perfect_squares_between_50_and_300 : ℕ :=
  (finset.Icc 8 17).card

-- Statement to prove
theorem number_of_perfect_squares_is_10 : count_perfect_squares_between_50_and_300 = 10 := by
  sorry

end number_of_perfect_squares_is_10_l806_806420


namespace baked_goods_not_eaten_l806_806889

theorem baked_goods_not_eaten : 
  let cookies_initial := 200
  let brownies_initial := 150
  let cupcakes_initial := 100
  
  let cookies_after_wife := cookies_initial - 0.30 * cookies_initial
  let brownies_after_wife := brownies_initial - 0.20 * brownies_initial
  let cupcakes_after_wife := cupcakes_initial / 2
  
  let cookies_after_daughter := cookies_after_wife - 40
  let brownies_after_daughter := brownies_after_wife - 0.15 * brownies_after_wife
  
  let cookies_after_friend := cookies_after_daughter - (cookies_after_daughter / 4)
  let brownies_after_friend := brownies_after_daughter - 0.10 * brownies_after_daughter
  let cupcakes_after_friend := cupcakes_after_wife - 10
  
  let cookies_after_other_friend := cookies_after_friend - 0.05 * cookies_after_friend
  let brownies_after_other_friend := brownies_after_friend - 0.05 * brownies_after_friend
  let cupcakes_after_other_friend := cupcakes_after_friend - 5
  
  let cookies_after_javier := cookies_after_other_friend / 2
  let brownies_after_javier := brownies_after_other_friend / 2
  let cupcakes_after_javier := cupcakes_after_other_friend / 2
  
  let total_remaining := cookies_after_javier + brownies_after_javier + cupcakes_after_javier
  total_remaining = 98 := by
{
  sorry
}

end baked_goods_not_eaten_l806_806889


namespace altitude_contains_x_l806_806516

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806516


namespace four_digit_integers_with_5_or_7_l806_806382

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l806_806382


namespace additional_discount_percentage_l806_806240

theorem additional_discount_percentage (P : ℝ) (lowest_price : ℝ) (d1_max : ℝ) (d2 : ℝ) 
  (hP : P = 30) (hl : lowest_price = 16.80) (h_max_d1 : d1_max = 0.30) : 
  d2 = 0.20 :=
by
  have d1 := 1 - d1_max
  have price_after_d1 := P * d1
  have price_after_d1_eq : price_after_d1 = 21 := by
    rw [hP, h_max_d1]
    norm_num
  have price_after_d2 := price_after_d1 * (1 - d2)
  have price_after_d2_eq : price_after_d2 = lowest_price := by
    rw [price_after_d1_eq, hl]
  have price_ratio := lowest_price / price_after_d1
  have ratio_eq : price_ratio = 0.8 := by
    rw [hl, price_after_d1_eq]
    norm_num
  have d2_eq : d2 = 1 - price_ratio := by
    rw [ratio_eq]
    norm_num
  exact d2_eq

end additional_discount_percentage_l806_806240


namespace correct_statement_l806_806342

-- Define the propositions p and q
def p (a b : ℝ) : Prop := a > |b| → a^2 > b^2
def q (x : ℝ) : Prop := x^2 = 4 → x = 2

-- The theorem that corresponds to the proof problem
theorem correct_statement (a b x : ℝ) (hp : p a b) (hq : q x) : (p a b ∨ q x) :=
by sorry

end correct_statement_l806_806342


namespace four_digit_numbers_with_5_or_7_l806_806387

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l806_806387


namespace slope_angle_acute_l806_806024

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem slope_angle_acute : 
  let f' := deriv f 
  in 0 < f' 1 := 
by
  sorry

end slope_angle_acute_l806_806024


namespace multiply_polynomials_l806_806548

def polynomial_multiplication (x : ℝ) : Prop :=
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824

theorem multiply_polynomials (x : ℝ) : polynomial_multiplication x :=
by
  sorry

end multiply_polynomials_l806_806548


namespace smallest_diff_of_YZ_XY_l806_806186

theorem smallest_diff_of_YZ_XY (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2509) (h4 : a + b > c) (h5 : b + c > a) (h6 : a + c > b) : b - a = 1 :=
by {
  sorry
}

end smallest_diff_of_YZ_XY_l806_806186


namespace equal_segments_l806_806074

-- Definitions of the problem's elements
variables {A B C I J : Point}
variables (triangle_ABC : Triangle A B C)
variables (incenter_I : Incenter triangle_ABC I)
variables (point_J : ∃ J, (Line_through A I ∈ Circumcircle triangle_ABC ∧ A ≠ J))

-- Theorem statement
theorem equal_segments :
  ∀ (J ∈ point_J), dist J B = dist J C ∧ dist J B = dist J I ∧ dist J C = dist J I :=
sorry

end equal_segments_l806_806074


namespace AM_half_BD_l806_806109

variable {α : Type*} [EuclideanGeometry α]

-- Definitions for points in Euclidean space
variables {A B C D E K M : α}

-- Conditions of the problem
axiom equilateral_triangle (A B C : α) : is_equilateral_triangle A B C
axiom point_on_side (D : α) (AC : Set α) : D ∈ AC
axiom point_on_side2 (E : α) (AB : Set α) : E ∈ AB
axiom ae_eq_cd (A E C D : α) : distance A E = distance C D
axiom midpoint_M (M D E : α) : is_midpoint M D E

-- Theorem to be proved
theorem AM_half_BD (A B C D E M : α)
  (h_eq_triangle : is_equilateral_triangle A B C)
  (h_point_D : D ∈ line_segment A C)
  (h_point_E : E ∈ line_segment A B)
  (h_ae_cd : distance A E = distance C D)
  (h_midpoint_M : is_midpoint M D E) : 
  distance A M = (1/2 : ℝ) * distance B D :=
sorry

end AM_half_BD_l806_806109


namespace students_language_difference_l806_806457

theorem students_language_difference :
  let total_students := 2500
  let S_min := 0.75 * total_students
  let S_max := 0.80 * total_students
  let F_min := 0.40 * total_students
  let F_max := 0.50 * total_students
  ∀ S F : ℝ,
    S_min ≤ S ∧ S ≤ S_max ∧ F_min ≤ F ∧ F ≤ F_max →
    ∃ m M : ℝ,
      m = total_students - S_min - F_min ∧
      M = total_students - S_max - F_max ∧
      M - m = 375
:= by
   intro total_students
   intro S_min S_max F_min F_max
   intros S F h
   use [total_students - S_min - F_min, total_students - S_max - F_max]
   cases h with hS hF
   split
   exact hS.left
   exact hS.right
   -- Proof steps omitted
   sorry

end students_language_difference_l806_806457


namespace sum_reciprocal_distances_l806_806328

theorem sum_reciprocal_distances (n : ℕ) (h : 2 ≤ n)
  (x : Fin n → ℝ) 
  (h_ordered : ∀ i j, i < j → x i < x j)
  (h_x_range : ∀ i, -1 ≤ x i ∧ x i ≤ 1)
  (t : Fin n → ℝ)
  (h_t_def : ∀ k, t k = ∏ i in (Finset.univ \ {k}), ((x k) - (x i))) 
  : (∑ k in Finset.univ, 1 / t k) ≥ 2^(n - 2) :=
sorry

end sum_reciprocal_distances_l806_806328


namespace like_terms_in_set_A_l806_806700

-- Define the monomials as given
def monomial_set_A1 := (1/3) * a^2 * b
def monomial_set_A2 := a^2 * b

def monomial_set_B1 := 3 * x^2 * y
def monomial_set_B2 := 3 * x * y^2

def monomial_set_C1 := a
def monomial_set_C2 := 1

def monomial_set_D1 := 2 * b * c
def monomial_set_D2 := 2 * a * b * c

-- Define the like_term condition: they are like terms if they have the same variables raised to the same powers.
def like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  (∀ v : ℕ, m1 v = m2 v)

-- Define the variables and their powers for each set
def vars_powers_A1 (v : ℕ) : ℕ := if v = 0 then 2 else if v = 1 then 1 else 0
def vars_powers_A2 (v : ℕ) : ℕ := if v = 0 then 2 else if v = 1 then 1 else 0

def vars_powers_B1 (v : ℕ) : ℕ := if v = 0 then 2 else if v = 1 then 1 else 0
def vars_powers_B2 (v : ℕ) : ℕ := if v = 0 then 1 else if v = 1 then 2 else 0

def vars_powers_C1 (v : ℕ) : ℕ := if v = 0 then 1 else 0
def vars_powers_C2 (v : ℕ) : ℕ := 0

def vars_powers_D1 (v : ℕ) : ℕ := if v = 1 then 1 else if v = 2 then 1 else 0
def vars_powers_D2 (v : ℕ) : ℕ := if v = 0 then 1 else if v = 1 then 1 else if v = 2 then 1 else 0

-- Define the proof statement
theorem like_terms_in_set_A :
  like_terms vars_powers_A1 vars_powers_A2 ∧
  ¬ like_terms vars_powers_B1 vars_powers_B2 ∧
  ¬ like_terms vars_powers_C1 vars_powers_C2 ∧
  ¬ like_terms vars_powers_D1 vars_powers_D2 :=
by sorry

end like_terms_in_set_A_l806_806700


namespace Desiree_age_correct_l806_806728

variable (Desiree_age : ℝ) (Cousin_age : ℝ) (years : ℝ)

axiom Desiree_current_age : Desiree_age = 2.99999835
axiom Cousin_current_age : Cousin_age = 1.499999175
axiom age_relation : Desiree_age = 2 * Cousin_age
axiom future_age_relation : Desiree_age + 30 = 0.6666666 * (Cousin_age + 30) + 14

theorem Desiree_age_correct : Desiree_age = 2.99999835 :=
by
  have h1 : Desiree_age = 2 * Cousin_age := age_relation
  have h2 : Desiree_age = 2.99999835 := Desiree_current_age
  sorry

end Desiree_age_correct_l806_806728


namespace weeks_saved_l806_806053

theorem weeks_saved (w : ℕ) :
  (10 * w / 2) - ((10 * w / 2) / 4) = 15 → 
  w = 4 := 
by
  sorry

end weeks_saved_l806_806053


namespace X_lies_on_altitude_BH_l806_806500

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806500


namespace ratio_of_volumes_is_correct_l806_806300

noncomputable def ratio_of_pyramid_volumes
  (a : ℝ) 
  (hex_base_area : ℝ := (3 / 2) * a^2 * Real.sqrt 3)
  (tri_base_area : ℝ := (a^2 * Real.sqrt 3) / 4)
  (slant_height : ℝ := 2 * a)
  (hex_height : ℝ := Real.sqrt((2 * a)^2 - (3 * a / (2 * Real.sqrt 3))^2))
  (tri_height : ℝ := Real.sqrt((2 * a)^2 - (a * Real.sqrt 3 / 6)^2))
  (V1 : ℝ := (1 / 3) * hex_base_area * hex_height)
  (V2 : ℝ := (1 / 3) * tri_base_area * tri_height) :
  ℝ :=
V1 / V2

theorem ratio_of_volumes_is_correct (a : ℝ) :
  ratio_of_pyramid_volumes a = (6 * Real.sqrt 1833) / 47 :=
by
  sorry

end ratio_of_volumes_is_correct_l806_806300


namespace trapezoid_area_ratio_l806_806254

theorem trapezoid_area_ratio (b h x : ℝ) 
  (base_relation : b + 150 = x)
  (area_ratio : (3 / 7) * h * (b + 75) = (1 / 2) * h * (b + x))
  (mid_segment : x = b + 150) 
  : ⌊x^3 / 1000⌋ = 142 :=
by
  sorry

end trapezoid_area_ratio_l806_806254


namespace intersection_of_sets_l806_806006

def M : Set ℝ := { x | x^2 - 4 ≤ 0 }
def N : Set ℝ := { x | log 2 x < 1 }
def M_inter_N : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_of_sets : M ∩ N = M_inter_N := by
  sorry

end intersection_of_sets_l806_806006


namespace sqrt_diff_inequality_l806_806761

theorem sqrt_diff_inequality (a : ℝ) (ha : a > 0) :
  sqrt (a + 5) - sqrt (a + 3) > sqrt (a + 6) - sqrt (a + 4) :=
sorry

end sqrt_diff_inequality_l806_806761


namespace sequence_sum_evaluation_l806_806210

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 0 = 3 ∧
  a 1 = 4 ∧
  ∀ n, a (n + 2) = a (n + 1) * a n + ⌈ (Int.sqrt ((a (n + 1))^2 - 1) * Int.sqrt ((a n)^2 - 1))⌉

theorem sequence_sum_evaluation (a : ℕ → ℤ) (h : sequence a) : 
    ∑' n, ((a (n + 3)) / (a (n + 2)) - (a (n + 2)) / (a n) + (a (n + 1)) / (a (n + 3)) - (a n) / (a (n + 1))) = 14 / 69 :=
sorry

end sequence_sum_evaluation_l806_806210


namespace all_propositions_correct_l806_806377

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem all_propositions_correct (m n : ℝ) (a b : α) (h1 : m ≠ 0) (h2 : a ≠ 0) : 
  (∀ (m : ℝ) (a b : α), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : α), (m - n) • a = m • a - n • a) ∧
  (∀ (m : ℝ) (a b : α), m • a = m • b → a = b) ∧
  (∀ (m n : ℝ) (a : α), m • a = n • a → m = n) :=
by {
  sorry
}

end all_propositions_correct_l806_806377


namespace minimum_value_g_monotonic_increasing_y_sum_log_n_inequality_l806_806325

variable (m : ℝ) (x n : ℝ)

def f (m x : ℝ) : ℝ := m * x^2 - (m-1)/x - log x
def g (x : ℝ) : ℝ := 1/x + log x
def y (m x : ℝ) : ℝ := f m x - g x

theorem minimum_value_g : g 1 = 1 := 
  sorry

theorem monotonic_increasing_y (hx : 1 ≤ x) : 
  monotone (λ x, y m x) ↔ 1 ≤ m := 
  sorry

theorem sum_log_n_inequality (hn : 0 < n) :
  (∑ i in range (n + 1), log i.succ / i.succ) < n^2 / (2 * (n + 1)) := 
  sorry

end minimum_value_g_monotonic_increasing_y_sum_log_n_inequality_l806_806325


namespace trig_inequality_intervals_l806_806579

theorem trig_inequality_intervals (x : ℝ) :
  (sin (4 * x) + cos (4 * x) + sin (5 * x) + cos (5 * x) < 0) ↔
  (30 * (π / 180) < x ∧ x < 70 * (π / 180)) ∨
  (110 * (π / 180) < x ∧ x < 150 * (π / 180)) ∨
  (180 * (π / 180) < x ∧ x < 190 * (π / 180)) ∨
  (230 * (π / 180) < x ∧ x < 270 * (π / 180)) ∨
  (310 * (π / 180) < x ∧ x < 350 * (π / 180)) :=
sorry

end trig_inequality_intervals_l806_806579


namespace cone_rotations_l806_806245

theorem cone_rotations (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r)
  (rotations : 17 * 2 * real.pi * r = 2 * real.pi * real.sqrt (r^2 + h^2)) :
  ∃ m n : ℕ, (m + n = 14) ∧ (∃ k : ℝ, k = m * real.sqrt n ∧ k = h / r) :=
by
  sorry

end cone_rotations_l806_806245


namespace count_four_digit_numbers_with_5_or_7_l806_806401

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l806_806401


namespace sum_greatest_least_second_row_l806_806104

-- Define the main theorem
theorem sum_greatest_least_second_row (n : ℕ) (grid : matrix (fin n) (fin n) ℕ) (middle : grid (fin.mk ((n + 1) / 2) sorry) (fin.mk ((n + 1) / 2) sorry) = 1) 
    (filled_clockwise : ∀ i j, ∃ (k : fin (n * n)), grid i j = k + 1) : 
    n = 17 → filled_grid_is_correctly_filled grid → 
    ∃ least greatest, 
        least = grid 1 (fin.mk 0 sorry) ∧ greatest = grid 1 (fin.mk (n - 2) sorry) ∧ least + greatest = 528 := 
by 
    sorry

end sum_greatest_least_second_row_l806_806104


namespace problem1_l806_806669

theorem problem1 (a b : ℝ) (i : ℝ) (h : (a-2*i)*i = b-i) : a^2 + b^2 = 5 := by
  sorry

end problem1_l806_806669


namespace fruit_weights_l806_806957

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806957


namespace meaningful_expression_range_l806_806443

theorem meaningful_expression_range (x : ℝ) (h : 1 - x > 0) : x < 1 := sorry

end meaningful_expression_range_l806_806443


namespace angle_bisectors_meet_l806_806092

-- Definitions extracted from the conditions
variables {A B C D P Q : Type*} [cyclic_convex_quadrilateral ABCD]
variables (AD BC AB ADC BCD : ℝ)

-- Condition stating AD + BC = AB
def condition1 := (AD + BC = AB)

-- Problem statement
theorem angle_bisectors_meet (h₁ : condition1) : 
  ∃ Q : Type*, 

-- Prove the bisectors of angles ADC and BCD intersect at Q
(Q lies_on AB ∧ bisector_of ADC Q ∧ bisector_of BCD Q) := sorry

end angle_bisectors_meet_l806_806092


namespace count_four_digit_integers_with_5_or_7_l806_806411

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l806_806411


namespace trapezoid_EFBA_area_l806_806214

noncomputable def rectangle_info := {
  AB: ℝ := 10,
  BC: ℝ := 2,
  area_ABCD: ℝ := 20,
  DE: ℝ := 2,
  FC: ℝ := 4
}

theorem trapezoid_EFBA_area : 
  let AB := rectangle_info.AB,
      BC := rectangle_info.BC,
      area_ABCD := rectangle_info.area_ABCD,
      DE := rectangle_info.DE,
      FC := rectangle_info.FC
  in
  AB * BC = area_ABCD ∧ DE + FC = AB  → 
  let height_EFBA := BC in
  let width_EF := AB - DE - FC in
  let area_EDA := (DE * height_EFBA) / 2 in
  let area_CFB := (FC * height_EFBA) / 2 in
  let area_EFAB := width_EF * height_EFBA in
  area_EDA + area_EFAB + area_CFB = 14 :=
by
  sorry

end trapezoid_EFBA_area_l806_806214


namespace find_angle_CED_l806_806877

noncomputable def angle_ABC : ℝ := 70
noncomputable def angle_BAC : ℝ := 50
noncomputable def angle_BCA : ℝ := 180 - angle_ABC - angle_BAC

theorem find_angle_CED :
  let angle_DCE := angle_BCA in
  let angle_CED := 90 - angle_DCE in
  (90 - 60) = (angle_CED) :=
by
  unfold angle_DCE angle_CED
  sorry

end find_angle_CED_l806_806877


namespace correct_conclusions_l806_806005

variables {p : ℝ} (hp : p > 0)
variables {x y : ℝ}
-- Parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Points A and B are on the parabola, and F is the focus
variables {A B F C D O : EuclideanSpace ℝ (Fin 2)}
variable H_F : F = ![p / 2, 0]
variable H_A : parabola A.val 0
variable H_B : parabola B.val 0

-- Points C and D on the directrix, perpendicular from A and B
variable H_C : C = ![-p / 2, (y 1)]
variable H_D : D = ![-p / 2, (y 2)]

-- Correct conclusions
theorem correct_conclusions :
  let AC := C - A
  let CD := D - C
  let BD := D - B
  let BA := A - B
  let AD := D - A
  let AO := O - A
  let FC := C - F
  let FD := D - F in
  (AC + CD = BD - BA ∧ (∃ λ : ℝ, AD = λ • AO) ∧ (FC ⬝ FD = 0) ∧ ¬(∀ M : EuclideanSpace ℝ (Fin 2), M.orthogonal_directrix → (A - M) ⬝ (B - M) > 0)) :=
sorry

end correct_conclusions_l806_806005


namespace prob_one_female_teacher_relationship_X_Y_expectation_X_l806_806680

theorem prob_one_female_teacher :
  let male_teachers := 5
  let female_teachers := 3
  let total_teachers := male_teachers + female_teachers
  let choices := Nat.choose total_teachers 2
  let prob_AB := (female_teachers * male_teachers) / choices.toReal
  let prob_A := ((Nat.choose female_teachers 1 * male_teachers) + (Nat.choose female_teachers 2)) / choices.toReal
  prob_A ≠ 0 → prob_AB / prob_A = 5 / 6 :=
by sorry

theorem relationship_X_Y :
  let Y := 3 / 4
  let X := 50 - 10 * Y
  ∀ n, X = 50 - 10 * n :=
by sorry

theorem expectation_X :
  let P_Y0 := (Nat.choose 5 2).toReal / (Nat.choose 8 2).toReal
  let P_Y1 := (3 * 5).toReal / (Nat.choose 8 2).toReal
  let P_Y2 := (Nat.choose 3 2).toReal / (Nat.choose 8 2).toReal
  let E_Y := 0 * P_Y0 + 1 * P_Y1 + 2 * P_Y2
  (50 - 10 * E_Y) = 42.5 :=
by sorry

end prob_one_female_teacher_relationship_X_Y_expectation_X_l806_806680


namespace correct_statement_5_l806_806375

variables (a b c : Vector)

theorem correct_statement_5 (a b : Vector) : 
  |a + b| ^ 2 = (a + b) • (a + b) :=
sorry

end correct_statement_5_l806_806375


namespace Euler_lines_intersect_at_single_point_l806_806076

theorem Euler_lines_intersect_at_single_point
  (A B C P : Point)
  (hP1 : ∠APB = 120°)
  (hP2 : ∠BPC = 120°)
  (hP3 : ∠CPA = 120°)
  (hA: ∠BAC < 120°)
  (hB: ∠ABC < 120°)
  (hC: ∠BCA < 120°):
  Euler_line (triangle A P B) ∩
  Euler_line (triangle B P C) ∩
  Euler_line (triangle C P A) ≠ ∅ := 
sorry

end Euler_lines_intersect_at_single_point_l806_806076


namespace correct_propositions_l806_806071

open ProbabilityTheory

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Let A and B be events
variable (A B : Event Ω)

-- Conditions for proposition 1
def prop1 : Prop :=
  (P A = 1 / 3) ∧ (P B = 1 / 2) ∧ (P (A ∪ B) = 1 / 6)

-- Conditions for proposition 2
def prop2 : Prop :=
  (A ∪ B = ({u | True} : Event Ω)) ∧ (P A + P B = 1)

-- Conditions for proposition 3
def prop3 : Prop :=
  (P A = 1 / 3) ∧ (P B = 2 / 3) ∧ (P (A ∩ (not B)) = 1 / 9)

-- Conditions for proposition 4
def prop4 : Prop :=
  (P (not A) = 1 / 3) ∧ (P (not B) = 1 / 4) ∧ (P ((not A) ∩ B) = 1 / 4)

-- The final proof statement
theorem correct_propositions:
  (prop1 → False) ∧ (prop2) ∧ (prop3) ∧ (prop4 → False) := by
  sorry

end correct_propositions_l806_806071


namespace derivative_hyperbolic_l806_806747

open Real

noncomputable def hyperbolic_derivative : ℝ → ℝ :=
  λ x, (1 / 2) * arctan (sinh x) - (sinh x / (2 * cosh x^2))

theorem derivative_hyperbolic :
  ∀ x,
    deriv hyperbolic_derivative x = (sinh x)^2 / (cosh x)^3 := by
  sorry

end derivative_hyperbolic_l806_806747


namespace fruit_weights_l806_806928

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806928


namespace four_digit_numbers_with_5_or_7_l806_806390

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l806_806390


namespace sequence_term_number_l806_806674

theorem sequence_term_number (a : ℕ) (h : a = 40) :
  ∃ n : ℕ, 3 * n + 1 = a ∧ n = 13 :=
by
  use 13
  simp [h]
  sorry

end sequence_term_number_l806_806674


namespace no_two_digit_prime_satisfies_conditions_l806_806593

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def has_integer_sqrt (n : ℕ) : Prop := ∃ (k : ℕ), k * k = n

theorem no_two_digit_prime_satisfies_conditions :
  ¬ ∃ (p : ℕ), is_prime p ∧ is_two_digit p ∧ has_integer_sqrt p ∧ (∃ (X Y : ℕ), 10 * X + Y = p ∧ 10 * Y + X - p = 90) :=
begin
  sorry
end

end no_two_digit_prime_satisfies_conditions_l806_806593


namespace integer_solutions_inequality_l806_806051

theorem integer_solutions_inequality
  (a : ℝ) (ha_pos : 0 < a) (h_sol_count : ∃ (x1 x2 x3 : ℤ), (1 / a < x1 ∧ x1 < 2 / a) 
    ∧ (1 / a < x2 ∧ x2 < 2 / a) ∧ (1 / a < x3 ∧ x3 < 2 / a) ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :
  ∃ (n : ℕ), n ∈ {2, 3, 4} ∧ ∃ (xs : fin n → ℤ), ∀ i, (2 / a < xs i) ∧ (xs i < 3 / a) := 
by 
  sorry

end integer_solutions_inequality_l806_806051


namespace fruit_weights_correct_l806_806945

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806945


namespace teacher_engineer_ratio_l806_806588

-- Define the context with the given conditions
variable (t e : ℕ)

-- Conditions
def avg_age (t e : ℕ) : Prop := (40 * t + 55 * e) / (t + e) = 45

-- The statement to be proved
theorem teacher_engineer_ratio
  (h : avg_age t e) :
  t / e = 2 := sorry

end teacher_engineer_ratio_l806_806588


namespace polynomial_value_at_2008_l806_806893

-- Let P be a polynomial with the given degree and coefficient constraints
theorem polynomial_value_at_2008 :
  ∃ (P : Polynomial ℝ), P.degree = 2008 ∧ P.leadingCoeff = 1 ∧ 
  (∀ i : ℕ, i ≤ 2007 → P.eval i = 2007 - i) → 
  P.eval 2008 = nat.factorial 2008 - 1 :=
by
  -- placeholder for the existence of such polynomial P and its properties
  sorry

end polynomial_value_at_2008_l806_806893


namespace determine_fruit_weights_l806_806923

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806923


namespace altitude_contains_x_l806_806517

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806517


namespace measure_angle_A44A45A43_l806_806346

structure Triangle (A B C : Type) :=
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (side_a : ℝ)
  (side_b : ℝ)
  (side_c : ℝ)
  (isosceles_right : angle_A = 90 ∧ side_b = side_c)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def construct_points (A1 A2 A3 : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  if n < 3 then [A1, A2, A3].nth n else
  let M := midpoint (A1 ((A2.1 + A3.1) / 2, (A2.2 + A3.2) / 2)) in
  midpoint ([A1, A2, A3].nth (n-3)) M

theorem measure_angle_A44A45A43 (A1 A2 A3 : ℝ × ℝ)
  (h : Triangle A1 A2 A3)
  (angle_A2_eq_45 : h.angle_B = 45)
  (angle_A3_eq_45 : h.angle_C = 45) :
  let A44 := construct_points A1 A2 A3 44 in
  let A45 := construct_points A1 A2 A3 45 in
  let A43 := construct_points A1 A2 A3 43 in
  (calculate_angle A44 A45 A43) = 45 :=
sorry

end measure_angle_A44A45A43_l806_806346


namespace fruit_weights_determined_l806_806952

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806952


namespace exists_n_int_coeffs_l806_806068

-- Definitions
def Q (P : ℚ[X]) (n : ℕ) : ℚ[X] := P.eval₂ (λ x, x + n) - P

-- Theorem statement
theorem exists_n_int_coeffs (P : ℚ[X]) :
  ∃ n : ℕ, ∀ x : ℤ, (Q P n).coeff x ∈ ℤ :=
sorry

end exists_n_int_coeffs_l806_806068


namespace distance_from_C_to_line_AB_is_sqrt2_l806_806007

-- Define the points in 3D space
structure Point3D (α : Type) := (x y z : α)

def A : Point3D ℝ := ⟨-1, 0, 0⟩
def B : Point3D ℝ := ⟨0, 1, -1⟩
def C : Point3D ℝ := ⟨-1, -1, 2⟩

-- Define the function for distance from a point to a line in space
noncomputable def distance_point_to_line (A B C : Point3D ℝ) : ℝ :=
  let AB := ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩ in
  let AC := ⟨C.x - A.x, C.y - A.y, C.z - A.z⟩ in
  let cross_prod := ⟨AB.y * AC.z - AB.z * AC.y, AB.z * AC.x - AB.x * AC.z, AB.x * AC.y - AB.y * AC.x⟩ in
  let cross_magnitude := real.sqrt (cross_prod.x^2 + cross_prod.y^2 + cross_prod.z^2) in
  let AB_magnitude := real.sqrt (AB.x^2 + AB.y^2 + AB.z^2) in
  cross_magnitude / AB_magnitude

-- State the theorem
theorem distance_from_C_to_line_AB_is_sqrt2 : distance_point_to_line A B C = real.sqrt 2 :=
by
  -- proof goes here
  sorry

end distance_from_C_to_line_AB_is_sqrt2_l806_806007


namespace second_horse_revolutions_l806_806234

theorem second_horse_revolutions (r1 r2 d1: ℝ) (n1 n2: ℕ) 
  (h1: r1 = 30) (h2: d1 = 36) (h3: r2 = 10) 
  (h4: 2 * Real.pi * r1 * d1 = 2 * Real.pi * r2 * n2) : 
  n2 = 108 := 
by
   sorry

end second_horse_revolutions_l806_806234


namespace reciprocal_relationship_l806_806016

theorem reciprocal_relationship (a b : ℚ)
  (h1 : a = (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12))
  (h2 : b = (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8)) :
  a = - 1 / b :=
by sorry

end reciprocal_relationship_l806_806016


namespace work_ratio_l806_806696

variables (a b : ℝ)

theorem work_ratio (h1 : b = 1 / 18) (h2 : a + b = 1 / 6) : a / b = 2 :=
by
  have h3 : a = 1 / 9 := by
    calc 
      a = 1 / 6 - b : by linarith [h2]
      _ = 1 / 6 - 1 / 18 : by rw [h1]
      _ = 1 / 9 : by norm_num
  calc 
    a / b = (1 / 9) / (1 / 18) : by rw [h1, h3]
    _ = 2 : by norm_num

end work_ratio_l806_806696


namespace nap_time_is_correct_l806_806267

-- Define the total trip time and the hours spent on each activity
def total_trip_time : ℝ := 15
def reading_time : ℝ := 2
def eating_time : ℝ := 1
def movies_time : ℝ := 3
def chatting_time : ℝ := 1
def browsing_time : ℝ := 0.75
def waiting_time : ℝ := 0.5
def working_time : ℝ := 2

-- Define the total activity time
def total_activity_time : ℝ := reading_time + eating_time + movies_time + chatting_time + browsing_time + waiting_time + working_time

-- Define the nap time as the difference between total trip time and total activity time
def nap_time : ℝ := total_trip_time - total_activity_time

-- Prove that the nap time is 4.75 hours
theorem nap_time_is_correct : nap_time = 4.75 :=
by
  -- Calculation hint, can be ignored
  -- nap_time = 15 - (2 + 1 + 3 + 1 + 0.75 + 0.5 + 2) = 15 - 10.25 = 4.75
  sorry

end nap_time_is_correct_l806_806267


namespace max_buses_in_city_l806_806853

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l806_806853


namespace largest_six_digit_number_l806_806692

/-- The largest six-digit number \( A \) that is divisible by 19, 
  the number obtained by removing its last digit is divisible by 17, 
  and the number obtained by removing the last two digits in \( A \) is divisible by 13 
  is \( 998412 \). -/
theorem largest_six_digit_number (A : ℕ) (h1 : A % 19 = 0) 
  (h2 : (A / 10) % 17 = 0) 
  (h3 : (A / 100) % 13 = 0) : 
  A = 998412 :=
sorry

end largest_six_digit_number_l806_806692


namespace crayons_given_to_Lea_minus_Mae_eq_seven_l806_806106

/-- Nori had 4 boxes of crayons with 8 crayons in each box.
    She gave 5 crayons to Mae and also gave some crayons to Lea.
    She has only 15 crayons left.
    Prove that the number of crayons Nori gave to Lea
    is 7 more than the number of crayons she gave to Mae. -/
theorem crayons_given_to_Lea_minus_Mae_eq_seven :
  (∃ (crayons_given_to_Lea crayons_given_to_Mae : ℕ), 
    let initial_crayons := 4 * 8 in
    let after_Mae_crayons := initial_crayons - crayons_given_to_Mae in
    let final_crayons := after_Mae_crayons - crayons_given_to_Lea in
    initial_crayons = 32 ∧ 
    crayons_given_to_Mae = 5 ∧ 
    final_crayons = 15 ∧ 
    crayons_given_to_Lea - crayons_given_to_Mae = 7) →
  T :=
by
  intros h,
  sorry

end crayons_given_to_Lea_minus_Mae_eq_seven_l806_806106


namespace balls_in_boxes_l806_806823

theorem balls_in_boxes :
  let num_balls := 6
  let num_boxes := 4
  (finset.card {x : fin (num_boxes + num_balls - 1) | finset.card (x.image x.pred) = num_balls - 1}) = 84 :=
by
  let num_balls := 6
  let num_boxes := 4
  have combination_formula : (∑ i in finset.range num_boxes, x i = num_balls) →
    (finset.card {x : fin (num_boxes + num_balls - 1) | finset.card (x.image x.pred) = num_balls - 1}) = 84 := sorry
  combination_formula sorry

end balls_in_boxes_l806_806823


namespace determine_fruit_weights_l806_806921

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806921


namespace count_four_digit_integers_with_5_or_7_l806_806414

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l806_806414


namespace store_profit_is_33_percent_l806_806239

noncomputable def store_profit (C : ℝ) : ℝ :=
  let initial_markup := 1.20 * C
  let new_year_markup := initial_markup + 0.25 * initial_markup
  let february_discount := new_year_markup * 0.92
  let shipping_cost := C * 1.05
  (february_discount - shipping_cost)

theorem store_profit_is_33_percent (C : ℝ) : store_profit C = 0.33 * C :=
by
  sorry

end store_profit_is_33_percent_l806_806239


namespace soccer_tournament_matches_l806_806585

theorem soccer_tournament_matches (x : ℕ) (h : 1 ≤ x) : (1 / 2 : ℝ) * x * (x - 1) = 45 := sorry

end soccer_tournament_matches_l806_806585


namespace sum_of_remainders_l806_806093

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5) = 5 :=
by
  sorry

end sum_of_remainders_l806_806093


namespace balls_in_boxes_l806_806821

theorem balls_in_boxes :
  let num_balls := 6
  let num_boxes := 4
  (finset.card {x : fin (num_boxes + num_balls - 1) | finset.card (x.image x.pred) = num_balls - 1}) = 84 :=
by
  let num_balls := 6
  let num_boxes := 4
  have combination_formula : (∑ i in finset.range num_boxes, x i = num_balls) →
    (finset.card {x : fin (num_boxes + num_balls - 1) | finset.card (x.image x.pred) = num_balls - 1}) = 84 := sorry
  combination_formula sorry

end balls_in_boxes_l806_806821


namespace range_of_a_l806_806347

theorem range_of_a (a : ℝ) (h1 : a ≤ 1) (h2 : ∃ S : set ℤ, S = {x : ℤ | ↑x ∈ set.Icc a (2 - a)} ∧ S.card = 3) : -1 < a ∧ a <= 0 := 
by
  sorry

end range_of_a_l806_806347


namespace combinations_of_three_toppings_l806_806139

def number_of_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_of_three_toppings : number_of_combinations 10 3 = 120 := by
  sorry

end combinations_of_three_toppings_l806_806139


namespace Bella_catch_correct_l806_806916

def Martha_catch : ℕ := 3 + 7
def Cara_catch : ℕ := 5 * Martha_catch - 3
def T : ℕ := Martha_catch + Cara_catch
def Andrew_catch : ℕ := T^2 + 2
def F : ℕ := Martha_catch + Cara_catch + Andrew_catch
def Bella_catch : ℕ := 2 ^ (F / 3)

theorem Bella_catch_correct : Bella_catch = 2 ^ 1102 := by
  sorry

end Bella_catch_correct_l806_806916


namespace ratio_of_areas_l806_806469

variables (AB CD h : ℝ)
variables (height_ratio : ℝ := 3)
variables (EAB_ratio : ℝ := 21 / 22)

-- Base lengths of trapezoid ABCD
def AB_len : ℝ := 7
def CD_len : ℝ := 15

-- The height of triangle EAB is three times the height of the trapezoid ABCD
def height_EAB : ℝ := height_ratio * h

-- Calculate the area of the trapezoid ABCD
def area_trapezoid : ℝ := (1 / 2) * (AB_len + CD_len) * h

-- Calculate the area of the triangle EAB
def area_triangle : ℝ := (1 / 2) * AB_len * height_EAB

-- Prove the ratio of the areas is 21/22
theorem ratio_of_areas
  (AB_CD_height : AB_len = 7 ∧ CD_len = 15 ∧ height_ratio = 3):
  (area_triangle AB_len h) / (area_trapezoid AB_len CD_len h) = EAB_ratio :=
by
  sorry

end ratio_of_areas_l806_806469


namespace inequality_problem_l806_806780

theorem inequality_problem
  (a b c d : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h_sum : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1 / 5 :=
by
  sorry

end inequality_problem_l806_806780


namespace solution_set_of_quadratic_inequality_l806_806732

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l806_806732


namespace balls_in_boxes_l806_806819

theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 62 ∧ 
    (∀ (b1 b2 b3 b4 : ℕ), b1 + b2 + b3 + b4 = 6) ∧ 
    (are_distinguishable b1 b2 b3 b4) :=
begin
  sorry
end

end balls_in_boxes_l806_806819


namespace range_f_x_sum_l806_806795

theorem range_f_x_sum {k m n : ℝ} (h1 : k > 0) (h2 : ∀ x, x ∈ set.Icc (-k) k → f x ∈ set.Icc m n) :
  m + n = 6 :=
by
  let f := λ x, 3 + Real.sin (2 * x)
  sorry

end range_f_x_sum_l806_806795


namespace min_value_expression_l806_806527

theorem min_value_expression (a b t : ℝ) (h : a + b = t) : 
  ∃ c : ℝ, c = ((a^2 + 1)^2 + (b^2 + 1)^2) → c = (t^4 + 8 * t^2 + 16) / 8 :=
by
  sorry

end min_value_expression_l806_806527


namespace exists_acute_triangle_side_lengths_l806_806480

-- Define the real numbers d_1, d_2, ..., d_12 in the interval (1, 12).
noncomputable def real_numbers_in_interval (d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → 1 < d n ∧ d n < 12

-- Define the condition for d_i, d_j, d_k to form an acute triangle
def forms_acuse_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- The main theorem statement
theorem exists_acute_triangle_side_lengths (d : ℕ → ℝ) (h : real_numbers_in_interval d) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ forms_acuse_triangle (d i) (d j) (d k) :=
sorry

end exists_acute_triangle_side_lengths_l806_806480


namespace euler_line_equation_circle_m_range_a_minimum_value_expression_l806_806450

-- Define triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (isosceles_AC_BC : (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 = (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2)

-- Define the Euler line equation verification
theorem euler_line_equation (T : Triangle)
  (A_vert : T.A = (-1, 0))
  (B_vert : T.B = (1, 2)):
  ∃ (l : ℝ × ℝ → Prop), (∀ p, l p ↔ p.1 + p.2 - 1 = 0) :=
  let C_vert : T.C.1 * T.C.2 = sorry in
  sorry

-- Define circle M and range condition for 'a'
theorem circle_m_range_a (r : ℝ) (h_r : r = 3 * Real.sqrt 2)
  (a : ℝ) (cond_a : ∀ (u v : ℝ), ((u + 5) ^ 2 + v ^ 2 = r ^ 2) 
    → (u ^ 2 + (v - a) ^ 2 = 2) 
    → a ∈ Set.Icc (- Real.sqrt 7) (Real.sqrt 7)) : 
  (a ∈ Set.Icc (- Real.sqrt 7) (Real.sqrt 7)) := 
sorry

-- Define minimum value for given expression on Euler line
theorem minimum_value_expression (x y : ℝ) 
  (h : x + y - 1 = 0) : 
  Real.sqrt (x ^ 2 + y ^ 2 - 2 * x - 2 * y + 2) 
  + Real.sqrt ((x - 2) ^ 2 + y ^ 2) = 2 := 
sorry

end euler_line_equation_circle_m_range_a_minimum_value_expression_l806_806450


namespace sum_digit_count_l806_806018

def digit_count (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem sum_digit_count (X Y : ℕ) (hX : 10 ≤ X ∧ X ≤ 99) (hY : 10 ≤ Y ∧ Y ≤ 99) :
  let sum := 1234 + (X * 100 + 65) + (Y * 10 + 2) in
  digit_count sum = 4 ∨ digit_count sum = 5 :=
sorry

end sum_digit_count_l806_806018


namespace find_locus_of_X_l806_806999

noncomputable def locus_of_X (sphere : Sphere) (l1 l2 : Line) (M N : Point) : Set Point :=
  {X | X ∈ sphere ∧ TangentToLineAt X l1 l2 ∧ X ∈ LineSegment M N ∧ TangentAt X sphere (LineSegment M N)}

theorem find_locus_of_X
  (sphere : Sphere)
  (O : Point)
  (R : ℝ)
  (l1 l2 : Line)
  (A B M N X : Point)
  (h1 : l1.TangentTo sphere A)
  (h2 : l2.TangentTo sphere B)
  (h3 : A ∈ l1 ∧ B ∈ l2)
  (h4 : M ∈ l1 ∧ N ∈ l2)
  (h5 : X ∈ LineSegment M N)
  (h6 : X ∈ sphere ∧ TangentAt X sphere (LineSegment M N)) :
  locus_of_X sphere l1 l2 M N = two_circles_on_sphere :
sorry

end find_locus_of_X_l806_806999


namespace unique_factorial_rep_l806_806570

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem unique_factorial_rep (A : ℕ) (hA_pos : A > 0) :
  ∃ (k : ℕ) (a : ℕ → ℕ),
      (∀ i, 1 ≤ i → i ≤ k → 0 ≤ a i ∧ a i ≤ i) ∧
      (A = ∑ i in (finset.range k).succ, (a i) * factorial i) := sorry

end unique_factorial_rep_l806_806570


namespace committee_probability_l806_806587

theorem committee_probability :
  let total_members := 24
  let boys := 12
  let girls := 12
  let committee_size := 5
  let total_committees := Nat.choose total_members committee_size
  let all_boys_girls_committees := 2 * Nat.choose boys committee_size
  let mixed_committees := total_committees - all_boys_girls_committees
  let probability := (mixed_committees : ℚ) / total_committees
  probability = 455 / 472 :=
by
  sorry

end committee_probability_l806_806587


namespace largest_digit_for_divisibility_by_3_l806_806880

theorem largest_digit_for_divisibility_by_3 :
  ∃ A : ℕ, (A ≤ 9 ∧ 3 + A + 6 + 7 + 9 + 2 ≡ 0 [MOD 3]) ∧ (∀ B : ℕ, (B ≤ 9 ∧ 3 + B + 6 + 7 + 9 + 2 ≡ 0 [MOD 3]) → B ≤ A) ∧ A = 9 :=
begin
  sorry
end

end largest_digit_for_divisibility_by_3_l806_806880


namespace X_on_altitude_BH_l806_806505

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806505


namespace dihedral_angle_of_isosceles_right_triangle_l806_806463

noncomputable def isosceles_right_triangle := 
{A B C : ℝ × ℝ // (A.1, A.2) = (0, 0) ∧ (B.1, B.2) = (1, 0) ∧ (C.1, C.2) = (0, 1) ∧ dist B A = 1 ∧ dist B C = 1 }

def midpoint (A C : ℝ × ℝ) : (ℝ × ℝ) :=
((A.1 + C.1) / 2, (A.2 + C.2) / 2)

theorem dihedral_angle_of_isosceles_right_triangle
  (A B C : ℝ × ℝ) (hABC : A ∈ isosceles_right_triangle ∧ B ∈ isosceles_right_triangle ∧ C ∈ isosceles_right_triangle)
  (M : ℝ × ℝ) (hM : M = midpoint A C)
  (h_fold : dist A C = 1) :
  ∠ C - B M - A = 90 :=
  sorry

end dihedral_angle_of_isosceles_right_triangle_l806_806463


namespace X_on_altitude_BH_l806_806489

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806489


namespace fruit_weights_assigned_l806_806967

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806967


namespace squared_distance_sums_l806_806067

variables {Point : Type} [Inhabited Point]

-- Definitions for i, n, A_i, B_i 
variables (n : ℕ) (A : Fin n.succ → Point) (B : Fin n.succ → Point)

-- Definition for translation vector t and its properties
variable (t : Point → Point → Point)
variable (translation : ∀ i : Fin n, t (A i) (A (i + 1)) = B i)

-- Defining the main proposition
def equal_squared_sums_of_distances : Prop :=
  ∑ i in Finset.range n, dist_v (A i) (B (i + 1)) = ∑ i in Finset.range n, dist_v (B i) (A (i + 1))

theorem squared_distance_sums : equal_squared_sums_of_distances n A B t translation :=
by
  sorry

end squared_distance_sums_l806_806067


namespace triangle_angle_ABC_l806_806049

theorem triangle_angle_ABC (A B C : Type) [EuclideanSpace A B C] 
  (AB AC BC : Real) (h1 : AB = 5) (h2 : AC = 3) (h3 : BC = 7) :
  ∠BAC = 2*π/3 := 
sorry

end triangle_angle_ABC_l806_806049


namespace smallest_number_among_bases_l806_806260

theorem smallest_number_among_bases : 
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2 + 1 in
  n4 < n3 ∧ n4 < n1 ∧ n4 < n2 :=
by
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  sorry

end smallest_number_among_bases_l806_806260


namespace part_a_roots_part_b_sum_l806_806668

theorem part_a_roots : ∀ x : ℝ, 2^x = x + 1 ↔ x = 0 ∨ x = 1 :=
by 
  intros x
  sorry

theorem part_b_sum (f : ℝ → ℝ) (h : ∀ x : ℝ, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 :=
by 
  sorry

end part_a_roots_part_b_sum_l806_806668


namespace problem_statement_l806_806080

theorem problem_statement :
  ∀ (b : ℝ), b ∈ Icc (-15 : ℝ) 15 →
  ∃ (m n : ℕ), Nat.coprime m n ∧ (m : ℝ)/n = 5/6 ∧ m + n = 11 :=
by
  sorry

end problem_statement_l806_806080


namespace negation_of_universal_proposition_l806_806371

open Set

theorem negation_of_universal_proposition :
  let M := {1, 2, 3, 4, 5, 6, 7} in
  (¬ (∀ n ∈ M, n > 1)) ↔ (∃ n ∈ M, n ≤ 1) :=
by
  sorry

end negation_of_universal_proposition_l806_806371


namespace two_digit_number_multiple_l806_806198

theorem two_digit_number_multiple (x : ℕ) (h1 : x ≥ 10) (h2 : x < 100) 
(h3 : ∃ k : ℕ, x + 1 = 3 * k) 
(h4 : ∃ k : ℕ, x + 1 = 4 * k) 
(h5 : ∃ k : ℕ, x + 1 = 5 * k) 
(h6 : ∃ k : ℕ, x + 1 = 7 * k) 
: x = 83 := 
sorry

end two_digit_number_multiple_l806_806198


namespace evaluate_expression_l806_806740

theorem evaluate_expression :
  let z1 := complex.mk 3 (-5)
  let z2 := complex.mk 3 5
  |z1| * |z2| + 2 * |z1| = 34 + 2 * real.sqrt 34 :=
by
  sorry

end evaluate_expression_l806_806740


namespace binomial_last_three_terms_sum_l806_806350

theorem binomial_last_three_terms_sum (n : ℕ) :
  (1 + n + (n * (n - 1)) / 2 = 79) → n = 12 :=
by
  sorry

end binomial_last_three_terms_sum_l806_806350


namespace num_four_digit_with_5_or_7_l806_806396

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l806_806396


namespace area_of_equilateral_figure_l806_806650

theorem area_of_equilateral_figure :
  ∀ (AF CD AB EF BC ED : ℝ) (angFAB angBCD : ℝ),
    AF = 1 ∧ CD = 1 ∧ AB = 1 ∧ EF = 1 ∧ BC = 1 ∧ ED = 1 ∧ 
    angFAB = 60 ∧ angBCD = 60 ∧ 
    (parallel AF CD) ∧ (parallel AB EF) ∧ (parallel BC ED) →
    area_of_figure AF CD AB EF BC ED angFAB angBCD = sqrt(3) :=
by
  intros AF CD AB EF BC ED angFAB angBCD
  intro h
  sorry -- Proof goes here

end area_of_equilateral_figure_l806_806650


namespace ratio_PQ_EF_l806_806972

noncomputable def point (x y : ℝ) : (ℝ × ℝ) := (x, y)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ratio_PQ_EF (A B C D E G F P Q : ℝ × ℝ)
  (hA : A = point 0 6) (hB : B = point 8 6) (hC : C = point 8 0) (hD : D = point 0 0)
  (hE : E = point 6 6) (hG : G = point 8 4) (hF : F = point 3 0)
  (hP : P = point (48 / 11) (60 / 11)) (hQ : Q = point (16 / 3) (14 / 3)) :
  (distance P Q / distance E F = 32 / (99 * real.sqrt 5)) :=
by
  sorry

end ratio_PQ_EF_l806_806972


namespace relationship_x_y_l806_806462

variables 
  (O : Point) -- center of the circle
  (r : ℝ) -- radius of the circle
  (circle : Circle O r) -- the circle with center O and radius r
  (A B C D : Point) -- points on the plane
  (x y : ℝ) -- angles x and y

-- Conditions/Definitions
def AB_chord_of_circle : Chord circle A B := sorry
def BC_equals_radius : dist B C = r := sorry
def CO_extends_to_D : Line C O ∧ collinear [C, O, D] := sorry
def AO_is_drawn : Line A O := sorry

-- Target statement to prove x = 3y
theorem relationship_x_y :
  x = 3 * y :=
sorry

end relationship_x_y_l806_806462


namespace fruit_weights_assigned_l806_806963

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806963


namespace smallest_two_digit_k_for_45k_l806_806196

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_two_digit_k_for_45k :
  ∃ k : ℕ, 10 ≤ k ∧ k < 100 ∧ is_perfect_square (45 * k) ∧ ∀ j : ℕ, 10 ≤ j ∧ j < k → ¬ is_perfect_square (45 * j) :=
begin
  use 20,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { use 30, norm_num },
  { intros j hj hjlt20,
    sorry
  },
end

end smallest_two_digit_k_for_45k_l806_806196


namespace jane_nail_polish_drying_time_l806_806054

theorem jane_nail_polish_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let index_finger_1 := 8
  let index_finger_2 := 10
  let middle_finger := 12
  let ring_finger := 11
  let pinky_finger := 14
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + index_finger_1 + index_finger_2 + middle_finger + ring_finger + pinky_finger + top_coat = 86 :=
by sorry

end jane_nail_polish_drying_time_l806_806054


namespace triangle_side_lengths_l806_806559

variable {A B C G D : Point} -- Points in the plane.
variable [EuclideanGeometry ℝ] -- Assuming plane Euclidean geometry over the reals.
variable (dist : Point → Point → ℝ) -- Distance function.

-- Assume point G is the centroid of triangle ABC.
def is_centroid (G A B C : Point) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), G = centroid A B C

-- Assume point D is the midpoint of side BC.
def is_midpoint (D B C : Point) : Prop :=
  dist B D = dist D C ∧ dist B D + dist D C = dist B C

-- Assume triangle BDG is equilateral with side length 1.
def equilateral_triangle (B D G : Point) : Prop :=
  dist B D = 1 ∧ dist D G = 1 ∧ dist B G = 1

theorem triangle_side_lengths
  (h1 : is_centroid G A B C)
  (h2 : is_midpoint D B C)
  (h3 : equilateral_triangle B D G) :
  dist A B = √7 ∧ dist B C = 2 ∧ dist C A = 2 * √3 :=
  sorry

end triangle_side_lengths_l806_806559


namespace total_payment_correct_l806_806572

namespace WorkPayment

-- Define the conditions
def rahul_days : ℕ := 3
def rajesh_days : ℕ := 2
def rahul_share : ℕ := 900

-- Define the total payment calculation using conditions
def total_payment_calculation : ℕ := 
  let rahul_rate := 1 / (rahul_days : ℚ)
  let rajesh_rate := 1 / (rajesh_days : ℚ)
  let total_rate := rahul_rate + rajesh_rate
  let rahul_ratio := rahul_rate / total_rate
  let rahul_work_share := rahul_ratio
  let total_payment := rahul_share / rahul_work_share
  total_payment.to_nat

-- Prove that the total payment is $2250
theorem total_payment_correct : total_payment_calculation = 2250 :=
by 
  sorry

end WorkPayment

end total_payment_correct_l806_806572


namespace fruit_weights_assigned_l806_806966

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806966


namespace only_one_statement_is_true_l806_806555

theorem only_one_statement_is_true (A B C D E: Prop)
  (hA : A ↔ B)
  (hB : B ↔ ¬ E)
  (hC : C ↔ (A ∧ B ∧ C ∧ D ∧ E))
  (hD : D ↔ ¬ (A ∨ B ∨ C ∨ D ∨ E))
  (hE : E ↔ ¬ A)
  (h_unique : ∃! x, x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∧ x = True) : E :=
by
  sorry

end only_one_statement_is_true_l806_806555


namespace cosine_angle_given_conditions_l806_806526

noncomputable def vector_norm {n : ℕ} (v : EuclideanSpace ℝ (Fin n)) : ℝ := Real.sqrt (∑ i, v i * v i)

def cosine_of_angle {n : ℕ} (a b : EuclideanSpace ℝ (Fin n)) : ℝ :=
  let dot_product := ∑ i, a i * b i
  let norm_a := vector_norm a
  let norm_b := vector_norm b
  dot_product / (norm_a * norm_b)

theorem cosine_angle_given_conditions {n : ℕ} (a b : EuclideanSpace ℝ (Fin n)) (ha : vector_norm a = 2 * vector_norm b) (hcond : vector_norm (λ i, 2 * a i + 3 * b i) = vector_norm a) :
  cosine_of_angle a b = -7 / 8 :=
by
  sorry

end cosine_angle_given_conditions_l806_806526


namespace evaluate_expression_l806_806271

theorem evaluate_expression :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 + 1/3) = -13 :=
by 
  sorry

end evaluate_expression_l806_806271


namespace find_f_neg_19_over_3_l806_806349

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x ∧ x < 1 then 8^x else 0 -- placeholder for other cases

theorem find_f_neg_19_over_3 (h₁ : ∀ x, f (x + 2) = f x)
                              (h₂ : ∀ x, f (-x) = -f x)
                              (h₃ : ∀ x, 0 < x ∧ x < 1 → f x = 8^x) :
  f (-19 / 3) = -2 :=
by
  sorry

end find_f_neg_19_over_3_l806_806349


namespace probability_white_ball_second_draw_l806_806029

noncomputable def probability_white_given_red (red_white_yellow_balls : Nat × Nat × Nat) : ℚ :=
  let (r, w, y) := red_white_yellow_balls
  let total_balls := r + w + y
  let p_A := (r : ℚ) / total_balls
  let p_AB := (r : ℚ) / total_balls * (w : ℚ) / (total_balls - 1)
  p_AB / p_A

theorem probability_white_ball_second_draw (r w y : Nat) (h_r : r = 2) (h_w : w = 3) (h_y : y = 1) :
  probability_white_given_red (r, w, y) = 3 / 5 :=
by
  rw [h_r, h_w, h_y]
  unfold probability_white_given_red
  simp
  sorry

end probability_white_ball_second_draw_l806_806029


namespace min_score_needed_to_increase_avg_l806_806211

theorem min_score_needed_to_increase_avg
    (scores : List ℕ) (next_increase : ℕ) (min_score : ℕ) (required_score : ℕ) :
    scores = [82, 76, 88, 94, 79, 85] →
    next_increase = 5 →
    min_score = 76 →
    required_score = 119 →
    let current_sum := scores.sum,
        num_tests := scores.length,
        current_avg := current_sum / num_tests,
        desired_avg := current_avg + next_increase,
        total_tests := num_tests + 1,
        total_score_needed := desired_avg * total_tests,
        next_test_score := total_score_needed - current_sum in
    next_test_score = required_score ∧ next_test_score ≥ min_score :=
by
  intros scores_eq ni_eq min_eq req_eq
  simp [scores_eq, ni_eq, min_eq, req_eq]
  sorry

end min_score_needed_to_increase_avg_l806_806211


namespace fruit_weights_correct_l806_806941

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806941


namespace main_inequality_l806_806337

-- Definitions for sequences x and y to fulfill conditions of the problem
variables (x y : ℕ → ℝ) (C : ℝ)
variables (h_x_increasing : ∀ i j : ℕ, i < j → x i < x j)
variables (h_y_increasing : ∀ i j : ℕ, i < j → y i < y j)
variables (h_C_range : -2 < C ∧ C < 2)
noncomputable def y_next (i n : ℕ) : ℝ := if i = n then y 0 else y (i + 1)

-- Function f as defined in the problem
noncomputable def f (x y : ℝ) (C : ℝ) : ℝ :=
sqrt (x^2 + C * x * y + y^2)

-- Theorem statement
theorem main_inequality (n : ℕ) (h_n : n ≥ 2) : 
  ∑ i in finRange n, f (x i) (y i) C < ∑ i in finRange n, f (x i) (y_next i n) C :=
sorry

end main_inequality_l806_806337


namespace num_four_digit_with_5_or_7_l806_806393

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l806_806393


namespace race_wheel_total_l806_806229

theorem race_wheel_total (total_racers : ℕ) (bicyclists_ratio : ℚ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) :
  total_racers = 40 ∧ bicyclists_ratio = 3 / 5 ∧ bicycle_wheels = 2 ∧ tricycle_wheels = 3 →
  let bicyclists := (bicyclists_ratio * total_racers : ℚ).toNat in
  let tricyclists := total_racers - bicyclists in
  let total_bicycle_wheels := bicyclists * bicycle_wheels in
  let total_tricycle_wheels := tricyclists * tricycle_wheels in
  total_bicycle_wheels + total_tricycle_wheels = 96 :=
by
  intros h
  let ⟨h_total_racers, h_ratio, h_bicycle_wheels, h_tricycle_wheels⟩ := h
  let bicyclists := (bicyclists_ratio * total_racers : ℚ).toNat
  let tricyclists := total_racers - bicyclists
  let total_bicycle_wheels := bicyclists * bicycle_wheels
  let total_tricycle_wheels := tricyclists * tricycle_wheels
  exact sorry

end race_wheel_total_l806_806229


namespace smallest_positive_value_l806_806303

-- Define the given expressions as functions
def exprA := 14 - 4 * Real.sqrt 15
def exprB := 4 * Real.sqrt 15 - 14
def exprC := 22 - 6 * Real.sqrt 17
def exprD := 66 - 16 * Real.sqrt 33
def exprE := 16 * Real.sqrt 33 - 66

-- Define the problem statement
theorem smallest_positive_value :
  ((4 * Real.sqrt 15 - 14) = 1.492) :=
sorry

end smallest_positive_value_l806_806303


namespace X_lies_on_altitude_BH_l806_806501

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806501


namespace valid_votes_B_l806_806035

theorem valid_votes_B (V : ℕ) (total_votes : V = 6720)
  (invalid_votes_percent : 0.20 * V = 0.20 * 6720)
  (A_exceeds_B_by : ∀ VA VB : ℕ, VA = VB + ⟨0.15 * V⟩)
  (total_valid_votes : ∀ VA VB : ℕ, VA + VB = 0.80 * 6720) :
  ∃ VB : ℕ, VB = 2184 := by
  sorry

end valid_votes_B_l806_806035


namespace count_four_digit_integers_with_5_or_7_l806_806412

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l806_806412


namespace simplify_expression_l806_806646

theorem simplify_expression :
  -3 - (+6) - (-5) + (-2) = -3 - 6 + 5 - 2 :=
by
  -- Here is where the proof would go, but we only need the statement
  sorry

end simplify_expression_l806_806646


namespace maximum_busses_l806_806855

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l806_806855


namespace swimming_pool_paint_area_l806_806992

theorem swimming_pool_paint_area :
  let length := 20 -- The pool is 20 meters long
  let width := 12  -- The pool is 12 meters wide
  let depth := 2   -- The pool is 2 meters deep
  let area_longer_walls := 2 * length * depth
  let area_shorter_walls := 2 * width * depth
  let total_side_wall_area := area_longer_walls + area_shorter_walls
  let floor_area := length * width
  let total_area_to_paint := total_side_wall_area + floor_area
  total_area_to_paint = 368 :=
by
  sorry

end swimming_pool_paint_area_l806_806992


namespace determine_superabundant_l806_806901

def is_divisor (a b : ℕ) : Prop := b % a = 0

def f (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ d, is_divisor d n).sum id

def is_superabundant (n : ℕ) : Prop := f (f n) = n + 3

theorem determine_superabundant : { n // is_superabundant n } = {3} :=
by
  sorry

end determine_superabundant_l806_806901


namespace imo1994_q36_l806_806294

theorem imo1994_q36 (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ k : ℤ, k * (m * n - 1) = n^3 + 1) ↔ 
  (m, n) ∈ {(2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (5, 2), (5, 3), (2, 5), (3, 5)} := 
by
  sorry

end imo1994_q36_l806_806294


namespace solve_for_n_l806_806981

theorem solve_for_n (n : ℕ) (h : (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 3) : n = 2 :=
by sorry

end solve_for_n_l806_806981


namespace figure_area_l806_806042

theorem figure_area :
  let area1 := 7 * 8
  let area2 := 4 * 3
  let area3 := 4 * 5
  let area4 := 4 * 2
  area1 + area2 + area3 + area4 = 96 :=
by
  let area1 := 7 * 8
  let area2 := 4 * 3
  let area3 := 4 * 5
  let area4 := 4 * 2
  show area1 + area2 + area3 + area4 = 96, from sorry

end figure_area_l806_806042


namespace ticket_sale_savings_l806_806707

theorem ticket_sale_savings (P : ℝ) (h : P > 0) :
  let original_price := 6 * P
  let sale_price := 3 * P
  let amount_saved := original_price - sale_price
  (amount_saved / original_price) * 100 = 50 :=
by
  -- Definitions based on conditions
  let original_price := 6 * P
  let sale_price := 3 * P
  let amount_saved := original_price - sale_price

  -- Prove the percentage saved
  have h1 : amount_saved = 3 * P := by sorry
  have h2 : (amount_saved / original_price) * 100 = (3 * P / (6 * P)) * 100 := by sorry
  have h3 : (3 * P / (6 * P)) * 100 = (1 / 2) * 100 := by sorry
  show (1 / 2) * 100 = 50 from sorry

end ticket_sale_savings_l806_806707


namespace polar_eq_line_l_cartesian_eq_curve_C_length_AB_l806_806047

-- Definitions of conditions
def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (t + 1, sqrt 3 * t + 1)

def polar_curve_C (ρ θ : ℝ) : Prop :=
  3 * ρ^2 * (cos θ)^2 + 4 * ρ^2 * (sin θ)^2 = 12

def cartesian_curve_C (x y : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / 3 = 1

-- Statements to prove
theorem polar_eq_line_l (ρ θ : ℝ) :
  (∃ t, ρ = sqrt((t + 1)^2 + (sqrt 3 * t + 1)^2) ∧ θ = atan2 (sqrt 3 * t + 1) (t + 1)) →
  (ρ * sin θ = sqrt 3 * ρ * cos θ - sqrt 3 + 1) := sorry

theorem cartesian_eq_curve_C (x y : ℝ) :
  (∃ ρ θ, ρ^2 * (3 * (cos θ)^2 + 4 * (sin θ)^2) = 12 ∧ x = ρ * cos θ ∧ y = ρ * sin θ) →
  (cartesian_curve_C x y) := sorry

theorem length_AB (t1 t2 l1 l2 : ℝ) :
  (∀ t, parametric_line_l t = (l1, l2)) →
  (∃ M, M = (1, 0)) →
  (∃ t1 t2, (5 * t1^2 + 4 * t1 - 12 = 0) ∧ (5 * t2^2 + 4 * t2 - 12 = 0) ∧ 
  t1 ≠ t2 ∧ t1 * t2 < 0) →
  abs (t1 - t2) = 16/5 := sorry

end polar_eq_line_l_cartesian_eq_curve_C_length_AB_l806_806047


namespace homothety_transformation_l806_806378

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V]

/-- Definition of a homothety transformation -/
def homothety (S A A' : V) (k : ℝ) : Prop :=
  A' = k • A + (1 - k) • S

theorem homothety_transformation (S A A' : V) (k : ℝ) :
  homothety S A A' k ↔ A' = k • A + (1 - k) • S := 
by
  sorry

end homothety_transformation_l806_806378


namespace incenter_circumcenter_distance_l806_806332

theorem incenter_circumcenter_distance (a b c r : ℝ)
  (h1 : a = 8) (h2 : b = 15) (h3 : c = 17)
  (right_triangle : a * a + b * b = c * c)
  (inradius_formula : r = (a + b - c) / 2) :
  let IO := real.sqrt ((r ^ 2) + ((c / 2 - r) ^ 2)) in
  IO = real.sqrt 157 / 2 :=
by
  sorry

end incenter_circumcenter_distance_l806_806332


namespace unique_element_set_l806_806446

theorem unique_element_set (a : ℝ) : (∀ x ∈ ({x : ℝ | a * x ^ 2 - 3 * x + 2 = 0} : set ℝ), True) → ({x : ℝ | a * x ^ 2 - 3 * x + 2 = 0}).card = 1 → a = 0 ∨ a = 9 / 8 := 
by 
  sorry

end unique_element_set_l806_806446


namespace totalPeoplePresent_is_630_l806_806111

def totalParents : ℕ := 105
def totalPupils : ℕ := 698

def groupA_fraction : ℚ := 30 / 100
def groupB_fraction : ℚ := 25 / 100
def groupC_fraction : ℚ := 20 / 100
def groupD_fraction : ℚ := 15 / 100
def groupE_fraction : ℚ := 10 / 100

def groupA_attendance : ℚ := 90 / 100
def groupB_attendance : ℚ := 80 / 100
def groupC_attendance : ℚ := 70 / 100
def groupD_attendance : ℚ := 60 / 100
def groupE_attendance : ℚ := 50 / 100

def junior_fraction : ℚ := 30 / 100
def intermediate_fraction : ℚ := 35 / 100
def senior_fraction : ℚ := 20 / 100
def advanced_fraction : ℚ := 15 / 100

def junior_attendance : ℚ := 85 / 100
def intermediate_attendance : ℚ := 80 / 100
def senior_attendance : ℚ := 75 / 100
def advanced_attendance : ℚ := 70 / 100

noncomputable def totalPeoplePresent : ℚ := 
  totalParents * groupA_fraction * groupA_attendance +
  totalParents * groupB_fraction * groupB_attendance +
  totalParents * groupC_fraction * groupC_attendance +
  totalParents * groupD_fraction * groupD_attendance +
  totalParents * groupE_fraction * groupE_attendance +
  totalPupils * junior_fraction * junior_attendance +
  totalPupils * intermediate_fraction * intermediate_attendance +
  totalPupils * senior_fraction * senior_attendance +
  totalPupils * advanced_fraction * advanced_attendance

theorem totalPeoplePresent_is_630 : totalPeoplePresent.floor = 630 := 
by 
  sorry -- no proof required as per the instructions

end totalPeoplePresent_is_630_l806_806111


namespace required_words_to_learn_l806_806824

def total_words : ℕ := 500
def required_percentage : ℕ := 85

theorem required_words_to_learn (x : ℕ) :
  (x : ℚ) / total_words ≥ (required_percentage : ℚ) / 100 ↔ x ≥ 425 := 
sorry

end required_words_to_learn_l806_806824


namespace fruit_weights_assigned_l806_806965

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806965


namespace students_juice_count_l806_806868

noncomputable def students_choosing_juice (total_students choosing_water : ℕ) : ℕ :=
  let ratio := (25 / 100 : ℚ) / (70 / 100 : ℚ)
  ratio * choosing_water

theorem students_juice_count (total_students choosing_water : ℕ) 
  (h1 : total_students * 70 / 100 = choosing_water) (h2 : choosing_water = 105) : 
  students_choosing_juice total_students choosing_water = 38 :=
by
  have ratio := (25 / 100 : ℚ) / (70 / 100 : ℚ)
  dsimp [students_choosing_juice]
  rw [h2]
  norm_num
  sorry

end students_juice_count_l806_806868


namespace centroid_traces_ellipse_l806_806374

noncomputable def fixed_base_triangle (A B : ℝ × ℝ) (d : ℝ) : Prop :=
(A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = d ∧ B.2 = 0)

noncomputable def vertex_moving_on_semicircle (A B C : ℝ × ℝ) : Prop :=
(C.1 - (A.1 + B.1) / 2)^2 + C.2^2 = ((B.1 - A.1) / 2)^2 ∧ C.2 ≥ 0

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem centroid_traces_ellipse
  (A B C G : ℝ × ℝ) (d : ℝ) 
  (h1 : fixed_base_triangle A B d) 
  (h2 : vertex_moving_on_semicircle A B C)
  (h3 : is_centroid A B C G) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (G.1^2 / a^2 + G.2^2 / b^2 = 1) := 
sorry

end centroid_traces_ellipse_l806_806374


namespace max_buses_constraint_satisfied_l806_806842

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l806_806842


namespace permutation_count_l806_806644

noncomputable def num_permutations : Nat :=
  1115600587

theorem permutation_count :
  (Multiset.permutations {c, e, i, i, i, n, o, s, t, t, u, v}.toMultiset).count "ut tensio sic vis" = num_permutations :=
sorry

end permutation_count_l806_806644


namespace general_term_of_sequence_smallest_n_for_sum_l806_806770

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, S n = 2 * a n - a 1) 
  (h2 : a 1 + a 3 = 2 * (a 2 + 1)) :
  ∀ n : ℕ, a n = 2 ^ n :=
sorry

theorem smallest_n_for_sum (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, a n = 2 ^ n) :
  ∃ n : ℕ, n ≥ 11 ∧ |(∑ i in finset.range n, (1 : ℚ) / a i) - 1| < 1 / 2016 :=
sorry

end general_term_of_sequence_smallest_n_for_sum_l806_806770


namespace max_temp_range_l806_806205

theorem max_temp_range (avg_temp : ℝ) (lowest_temp : ℝ) (days : ℕ) (total_temp : ℝ) (range : ℝ) : 
  avg_temp = 45 → 
  lowest_temp = 42 → 
  days = 5 → 
  total_temp = avg_temp * days → 
  range = 6 := 
by 
  sorry

end max_temp_range_l806_806205


namespace commutative_commutative2_l806_806078

variable (a b c : ℝ)

def op (a b : ℝ) := (a - b)^2

theorem commutative : op a b = op b a :=
by
  unfold op
  apply congr_arg
  apply pow_two_subtract_swap

theorem commutative2 : op a (b - c) = op (b - c) a :=
by
  unfold op
  apply congr_arg
  apply pow_two_subtract_swap

end commutative_commutative2_l806_806078


namespace sum_base_8_sequence_l806_806714

def sum_arithmetic_sequence_base_8 (a l : ℕ) : ℕ :=
let a_decimal := 3
let l_decimal := 64
let n := (l_decimal - a_decimal + 1)
let S_decimal := (n * (a_decimal + l_decimal)) / 2
in S_decimal -- Does not automatically convert to base 8 in this definition

theorem sum_base_8_sequence :
  sum_arithmetic_sequence_base_8 3 64 = 4035 :=
by sorry

end sum_base_8_sequence_l806_806714


namespace fruit_weights_l806_806930

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806930


namespace solve_right_triangle_correct_l806_806580

variables (C A B D: Type) [right_triangle : RightTriangle C A B]
variables (a l : ℝ)
-- L is the angle bisector of the right angle at C

noncomputable def solve_right_triangle (a l : ℝ) : ℝ × ℝ × ℝ :=
let delta := asin ((l + sqrt (l^2 + 2 * a^2)) / (2 * a)) in
let alpha := 90 in
let beta := delta * 0.5 in  -- since alpha/2 = 45 degrees
let gamma := 180 - (delta + beta) in
let b := a * (sin beta) / (sin alpha) in
let c := a * (sin gamma) / (sin alpha) in
(b, c, delta)

-- Theorem expressing our proof problem
theorem solve_right_triangle_correct (a l : ℝ) :
solve_right_triangle a l = ( 
    a * (sin ((asin ((l + sqrt (l^2 + 2 * a^2)) / (2 * a)) * 0.5)) / (sin 90)),
    a * (sin (180 - (asin ((l + sqrt (l^2 + 2 * a^2)) / (2 * a)) + asin ((l + sqrt (l^2 + 2 * a^2)) / (2 * a)) * 0.5)) / (sin 90)),
    asin ((l + sqrt (l^2 + 2 * a^2)) / (2 * a))
    ) :=
begin
  sorry
end

end solve_right_triangle_correct_l806_806580


namespace max_initial_segment_length_l806_806164

theorem max_initial_segment_length (sequence1 : ℕ → ℕ) (sequence2 : ℕ → ℕ)
  (period1 : ℕ) (period2 : ℕ)
  (h1 : ∀ n, sequence1 (n + period1) = sequence1 n)
  (h2 : ∀ n, sequence2 (n + period2) = sequence2 n)
  (p1 : period1 = 7) (p2 : period2 = 13) :
  ∃ max_length : ℕ, max_length = 18 :=
sorry

end max_initial_segment_length_l806_806164


namespace inequality_holds_for_a_in_interval_l806_806306

theorem inequality_holds_for_a_in_interval:
  (∀ x y : ℝ, 
     2 ≤ x ∧ x ≤ 3 ∧ 3 ≤ y ∧ y ≤ 4 → (3*x - 2*y - a) * (3*x - 2*y - a^2) ≤ 0) ↔ a ∈ Set.Iic (-4) :=
by
  sorry

end inequality_holds_for_a_in_interval_l806_806306


namespace number_of_perfect_squares_is_10_l806_806421

-- Define the number of integers n such that 50 ≤ n^2 ≤ 300
def count_perfect_squares_between_50_and_300 : ℕ :=
  (finset.Icc 8 17).card

-- Statement to prove
theorem number_of_perfect_squares_is_10 : count_perfect_squares_between_50_and_300 = 10 := by
  sorry

end number_of_perfect_squares_is_10_l806_806421


namespace lines_intersect_sum_c_d_l806_806156

theorem lines_intersect_sum_c_d (c d : ℝ) 
    (h1 : ∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) 
    (h2 : ∀ x y : ℝ, x = 3 ∧ y = 3) : 
    c + d = 4 :=
by sorry

end lines_intersect_sum_c_d_l806_806156


namespace altitude_contains_x_l806_806514

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806514


namespace compare_abc_l806_806781

noncomputable def a : ℝ := log 0.3 0.2
noncomputable def b : ℝ := Real.ln 0.2
noncomputable def c : ℝ := 0.3 ^ 0.2

theorem compare_abc : a > c ∧ c > b :=
by
  have h1 : a = log 0.3 0.2 := rfl
  have h2 : b = Real.ln 0.2 := rfl
  have h3 : c = 0.3 ^ 0.2 := rfl
  sorry

end compare_abc_l806_806781


namespace X_lies_on_altitude_BH_l806_806504

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806504


namespace total_initial_candles_l806_806697

-- Define the conditions
def used_candles : ℕ := 32
def leftover_candles : ℕ := 12

-- State the theorem
theorem total_initial_candles : used_candles + leftover_candles = 44 := by
  sorry

end total_initial_candles_l806_806697


namespace digits_with_five_or_seven_is_5416_l806_806406

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l806_806406


namespace total_population_combined_is_correct_l806_806883

theorem total_population_combined_is_correct : 
  ∃ (P_A P_B C_A C_B : ℕ), 
    0.90 * P_A = 23040 ∧ 
    0.80 * P_B = 17280 ∧ 
    C_A = 3 * C_B ∧ 
    P_A - C_A = P_B - C_B ∧ 
    (P_A + P_B = 47200) := by
  sorry

end total_population_combined_is_correct_l806_806883


namespace correct_statement_l806_806201

theorem correct_statement (a: Type) (b: Type) (c: Type) (N : Set Nat) (Q : Set Rational) :
  (a ⊆ {a, b, c}) → (∅ ∈ {0}) → ({0, 1} ⊊ N) → (√2 ∈ Q) → 
  (∃ A B C D, (A = (a ∈ {a, b, c})) ∧ (B = (∅ ⊆ {0})) ∧ 
  (C = ({0,1} ⊂ N)) ∧ (D = (¬(√2 ∈ Q))) ∧ C) := 
sorry

end correct_statement_l806_806201


namespace X_on_altitude_BH_l806_806511

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806511


namespace percentage_increase_in_surface_area_l806_806634

variable {R : ℝ} (hR : 0 < R)

def cap_radius : ℝ := R * (Real.sqrt 3) / 2

def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

def hemisphere_surface_area (rho : ℝ) : ℝ := 2 * Real.pi * rho^2

def spherical_cap_height : ℝ := R / 2

def spherical_cap_surface_area : ℝ := 2 * Real.pi * R * (R / 2)

theorem percentage_increase_in_surface_area :
  let 
    rho := cap_radius R
    A1 := sphere_surface_area R
    A2 := hemisphere_surface_area rho
    m := spherical_cap_height R
    A3 := spherical_cap_surface_area R
    increase_in_surface_area := A2 - A3
    percentage_increase := (increase_in_surface_area / A1) * 100
  in
    percentage_increase = 12.5 :=
by
  sorry

end percentage_increase_in_surface_area_l806_806634


namespace right_angled_triangle_l806_806256

theorem right_angled_triangle :
  (1 * 1 + √2 * √2 = √3 * √3) :=
by sorry

end right_angled_triangle_l806_806256


namespace value_of_y_l806_806435

theorem value_of_y (y: ℚ) (h: (2 / 5 - 1 / 7) = 14 / y): y = 490 / 9 :=
by
  sorry

end value_of_y_l806_806435


namespace sum_not_nat_l806_806069

noncomputable def sequence_of_integers (a : ℕ → ℕ) : Prop :=
∀ m n, m < n → a m < a n

noncomputable def contains_all_primes (a : ℕ → ℕ) : Prop :=
∀ p, prime p → ∃ i, a i = p

theorem sum_not_nat (a : ℕ → ℕ) (h1 : sequence_of_integers a) (h2 : contains_all_primes a) :
  ∀ n m, n < m → (∑ i in finset.range (m - n + 1), (1 / (a (n + i) : ℝ))) ∉ ℤ := sorry

end sum_not_nat_l806_806069


namespace mike_needs_to_pay_400_l806_806103

def xray_cost := 250
def mri_cost := 3 * xray_cost
def ct_scan_cost := 2 * mri_cost
def blood_tests_cost := 200
def ultrasound_cost := 0.5 * mri_cost
def deductible := 500

def total_cost := xray_cost + mri_cost + ct_scan_cost + blood_tests_cost + ultrasound_cost

def remaining_after_deductible := total_cost - deductible

def insurance_coverage_xray := 0.80 * xray_cost
def insurance_coverage_mri := 0.80 * mri_cost
def insurance_coverage_ct_scan := 0.70 * ct_scan_cost
def insurance_coverage_blood_tests := 0.50 * blood_tests_cost
def insurance_coverage_ultrasound := 0.60 * ultrasound_cost

def total_insurance_coverage := insurance_coverage_xray + insurance_coverage_mri + insurance_coverage_ct_scan + insurance_coverage_blood_tests + insurance_coverage_ultrasound

def remaining_amount_to_pay := remaining_after_deductible - total_insurance_coverage

theorem mike_needs_to_pay_400 : remaining_amount_to_pay = 400 :=
by
  sorry

end mike_needs_to_pay_400_l806_806103


namespace max_mn_l806_806023

theorem max_mn (m n : ℝ) (h1 : m ≥ 0) (h2 : n ≥ 0)
  (h3 : ∀ x ∈ set.Icc (1/2 : ℝ) 2, deriv (λ x, (1/2)*(m-2)*x^2 + (n-8)*x + 1) x ≤ 0) : 
  ∃ (m n : ℝ), mn = 18 :=
by
  sorry

end max_mn_l806_806023


namespace initial_oranges_in_bowl_l806_806221

theorem initial_oranges_in_bowl (A O : ℕ) (R : ℚ) (h1 : A = 14) (h2 : R = 0.7) 
    (h3 : R * (A + O - 15) = A) : O = 21 := 
by 
  sorry

end initial_oranges_in_bowl_l806_806221


namespace cos_x_plus_2y_is_one_l806_806327

theorem cos_x_plus_2y_is_one
    (x y : ℝ) (a : ℝ) 
    (hx : x ∈ Set.Icc (-Real.pi) Real.pi)
    (hy : y ∈ Set.Icc (-Real.pi) Real.pi)
    (h_eq : 2 * a = x ^ 3 + Real.sin x ∧ 2 * a = (-2 * y) ^ 3 - Real.sin (-2 * y)) :
    Real.cos (x + 2 * y) = 1 := 
sorry

end cos_x_plus_2y_is_one_l806_806327


namespace marbles_per_customer_l806_806546

theorem marbles_per_customer
  (initial_marbles remaining_marbles customers marbles_per_customer : ℕ)
  (h1 : initial_marbles = 400)
  (h2 : remaining_marbles = 100)
  (h3 : customers = 20)
  (h4 : initial_marbles - remaining_marbles = customers * marbles_per_customer) :
  marbles_per_customer = 15 :=
by
  sorry

end marbles_per_customer_l806_806546


namespace bin_to_oct_correct_l806_806278

def bin_to_dec (b : ℕ) : ℕ :=
  b / 100000 * 2^5 + (b / 10000 % 10) * 2^4 + (b / 1000 % 10) * 2^3 +
  (b / 100 % 10) * 2^2 + (b / 10 % 10) * 2^1 + (b % 10) * 2^0

def dec_to_oct (d : ℕ) : string :=
  let rec aux (n : ℕ) (acc : string) : string :=
    if n = 0 then acc else aux (n / 8) ((char.of_nat (48 + n % 8)).to_string ++ acc)
  aux d ""

theorem bin_to_oct_correct : 
  dec_to_oct (bin_to_dec 110101) = "66" :=
by 
  sorry

end bin_to_oct_correct_l806_806278


namespace intersection_point_of_parallelogram_l806_806172

theorem intersection_point_of_parallelogram
  (A B C D P Q R : Type)
  [has_zero A] [add_comm_group A]
  [module ℝ A] [affine_space A (affine_map ℝ A)]
  (n : ℕ)
  (AD_eq_parts : ∃ (f : fin (n+1) → A), function.injective f ∧ ∀ i, 0 < i → (f i -ᵥ f 0) = (f 1 -ᵥ f 0) * i)
  (P_is_first_point : P = f 1)
  (BP : affine_map ℝ A)
  (BP_lines : ∀ i, affine_map ℝ A ∧ affine_map ℝ A → (affine_map ℝ A).to_fun P (affine_map ℝ A) (affine_map ℝ A))
  (Q_is_intersection : ∃ t, (AC.fun t = BP.fun t) )
  : Q = (1 / (n + 1)) • AC :=
sorry

end intersection_point_of_parallelogram_l806_806172


namespace sum_coeff_abs_eq_l806_806309

open Polynomial

noncomputable def P (x : ℚ) : ℚ := 1 + (1/4) * x - (1/8) * x^2
noncomputable def Q (x : ℚ) : ℚ := P(x) * P(x^2) * P(x^4)

theorem sum_coeff_abs_eq : ∑ i in Finset.range 15, |(Q (Polynomial.X) - ∑ i in Finset.range 15, (Q (Polynomial.X))[i])| = 125 / 512 := by
  sorry

end sum_coeff_abs_eq_l806_806309


namespace most_likely_number_of_cars_l806_806629

theorem most_likely_number_of_cars 
    (cars_in_first_10_seconds : ℕ := 6) 
    (time_for_first_10_seconds : ℕ := 10) 
    (total_time_seconds : ℕ := 165) 
    (constant_speed : Prop := true) : 
    ∃ (num_cars : ℕ), num_cars = 100 :=
by
  sorry

end most_likely_number_of_cars_l806_806629


namespace obtuse_angle_is_B_l806_806043

-- Define the obtuse triangle with given conditions
variables {A B C : Type*} [IsTriangle A B C] 
variables {a b c : ℝ} -- side lengths opposite to angles A, B, and C
variables (angleA angleB angleC : ℝ) -- measures of angles A, B, and C

-- Assume the conditions given in the problem
axiom abc_obtuse (h1 : a < c) (h2 : c < b) (h3 : IsObtuseTriangle A B C) 
  (h_alloc_side1 : sideA = a) (h_alloc_side2 : sideB = b) (h_alloc_side3 : sideC = c)
  (h_alloc_angle1 : angleA = angleB) (h_alloc_angle2: angleB = angleA) (h_alloc_angle3: angleC = angleC): 
  angleB = IsObtuse angleB

theorem obtuse_angle_is_B (h1 : a < c) (h2 : c < b) (h3 : IsObtuseTriangle A B C) 
  (h_alloc_side1 : sideA = a) (h_alloc_side2 : sideB = b) (h_alloc_side3 : sideC = c)
  (h_alloc_angle1 : angleA = angleB) (h_alloc_angle2: angleB = angleA) (h_alloc_angle3: angleC = angleC): 
  ∃ B, B = angleB: sorry

end obtuse_angle_is_B_l806_806043


namespace remainder_of_sum_l806_806751

theorem remainder_of_sum :
  ((88134 + 88135 + 88136 + 88137 + 88138 + 88139) % 9) = 6 :=
by
  sorry

end remainder_of_sum_l806_806751


namespace matrix_power_B150_l806_806065

open Matrix

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

-- Prove that B^150 = I
theorem matrix_power_B150 : 
  (B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_power_B150_l806_806065


namespace olympic_ad_broadcasting_methods_l806_806284

theorem olympic_ad_broadcasting_methods
    (ads : Finset (Fin 5))  -- 5 advertisements represented by finite set
    (commercials : Finset (Fin 3))  -- 3 different commercial advertisements
    (olympic_ads : Finset (Fin 2))  -- 2 different Olympic promotional advertisements
    (last_ad_is_olympic : ∃ x ∈ olympic_ads, ads(4) = x)  -- Last advertisement must be an Olympic ad
    (no_consecutive_olympic_ads : ∀ x y ∈ olympic_ads, x ≠ y → ∃ i j, i < 4 ∧ j < 4 ∧ i ≠ j ∧ ads(i) = x ∧ ads(j) = y → ∀ k, k ≠ i → k ≠ j → ads(k) ∉ olympic_ads) 
    -- The 2 Olympic promotional advertisements cannot be broadcast consecutively
  : (ads.card = 5 ∧ commercials.card = 3 ∧ olympic_ads.card = 2) → 36 :=  -- proving there are 36 different broadcasting methods
begin
  sorry  -- proof omitted
end

end olympic_ad_broadcasting_methods_l806_806284


namespace rakesh_fixed_deposit_percentage_l806_806123

-- Definitions based on the problem statement
def salary : ℝ := 4000
def cash_in_hand : ℝ := 2380
def spent_on_groceries : ℝ := 0.30

-- The theorem to prove
theorem rakesh_fixed_deposit_percentage (x : ℝ) 
  (H1 : cash_in_hand = 0.70 * (salary - (x / 100) * salary)) : 
  x = 15 := 
sorry

end rakesh_fixed_deposit_percentage_l806_806123


namespace maximum_busses_l806_806856

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l806_806856


namespace g_even_function_l806_806473

noncomputable def g (x : ℝ) : ℝ := log (x^2)

theorem g_even_function : ∀ x : ℝ, g x = g (-x) :=
by
  intros
  simp [g, pow_two, log]
  sorry

end g_even_function_l806_806473


namespace owen_longer_flight_than_eloise_l806_806738

def eloise_glide_feet : ℝ := 6562 / (33 * 30)
def owen_flight_feet : ℝ := 6562 / (9 * 30)
def length_difference_feet := owen_flight_feet - eloise_glide_feet
theorem owen_longer_flight_than_eloise :
  abs (length_difference_feet - 17) < 1 :=
by
  unfold eloise_glide_feet owen_flight_feet length_difference_feet
  sorry

end owen_longer_flight_than_eloise_l806_806738


namespace bad_oranges_l806_806228

theorem bad_oranges (total_oranges : ℕ) (students : ℕ) (less_oranges_per_student : ℕ)
  (initial_oranges_per_student now_oranges_per_student shared_oranges now_total_oranges bad_oranges : ℕ) :
  total_oranges = 108 →
  students = 12 →
  less_oranges_per_student = 3 →
  initial_oranges_per_student = total_oranges / students →
  now_oranges_per_student = initial_oranges_per_student - less_oranges_per_student →
  shared_oranges = students * now_oranges_per_student →
  now_total_oranges = 72 →
  bad_oranges = total_oranges - now_total_oranges →
  bad_oranges = 36 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bad_oranges_l806_806228


namespace fruit_weights_assigned_l806_806962

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806962


namespace point_on_altitude_l806_806522

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806522


namespace train_length_l806_806653

theorem train_length (L : ℕ) :
  (L + 350) / 15 = (L + 500) / 20 → L = 100 := 
by
  intro h
  sorry

end train_length_l806_806653


namespace distance_traveled_is_6000_l806_806199

-- Define the conditions and the question in Lean 4
def footprints_per_meter_Pogo := 4
def footprints_per_meter_Grimzi := 3 / 6
def combined_total_footprints := 27000

theorem distance_traveled_is_6000 (D : ℕ) :
  footprints_per_meter_Pogo * D + footprints_per_meter_Grimzi * D = combined_total_footprints →
  D = 6000 :=
by
  sorry

end distance_traveled_is_6000_l806_806199


namespace fruit_weights_l806_806929

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806929


namespace max_buses_in_city_l806_806851

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l806_806851


namespace point_X_on_altitude_BH_l806_806495

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806495


namespace point_outside_circle_l806_806834

theorem point_outside_circle
  (a b : ℝ)
  (h : ∃ (P1 P2 : ℝ × ℝ), P1 ≠ P2 ∧ (P1.1^2 + P1.2^2 = 1) ∧ (P2.1^2 + P2.2^2 = 1) ∧ (a * P1.1 + b * P1.2 = 1) ∧ (a * P2.1 + b * P2.2 = 1)) :
  a^2 + b^2 > 1 :=
begin
  sorry
end

end point_outside_circle_l806_806834


namespace ab_sum_l806_806026

theorem ab_sum (a b : ℕ) (h1: (a + b) % 9 = 8) (h2: (a - b) % 11 = 7) : a + b = 8 :=
sorry

end ab_sum_l806_806026


namespace multiplication_problems_l806_806742

theorem multiplication_problems :
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) :=
by sorry

end multiplication_problems_l806_806742


namespace symmmedian_contains_line_l806_806619

open EuclideanGeometry

/-- Given that B and C are points on the circumcircle of triangle ABC, and P is the intersection 
point of the tangents to the circumcircle at B and C, prove that the line AP contains the symmedian AS. -/
theorem symmmedian_contains_line
  {A B C P : Point}
  (h1 : IsOnCircumcircle B A C)
  (h2 : IsOnCircumcircle C A B)
  (h3 : IsTangentIntersection A B C P) :
  ContainsSymmedian A P :=
sorry

end symmmedian_contains_line_l806_806619


namespace Riverdale_High_students_l806_806706

theorem Riverdale_High_students
  (f j : ℕ)
  (h1 : (3 / 7) * f + (3 / 4) * j = 234)
  (h2 : f + j = 420) :
  f = 64 ∧ j = 356 := by
  sorry

end Riverdale_High_students_l806_806706


namespace fruit_weights_l806_806935

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806935


namespace fruit_weights_l806_806932

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806932


namespace molly_bike_miles_l806_806545

def total_miles_ridden (daily_miles years_riding days_per_year : ℕ) : ℕ :=
  daily_miles * years_riding * days_per_year

theorem molly_bike_miles :
  total_miles_ridden 3 3 365 = 3285 :=
by
  -- The definition and theorem are provided; the implementation will be done by the prover.
  sorry

end molly_bike_miles_l806_806545


namespace solution_set_l806_806767

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain (x : ℝ) : true
axiom f_at_1 : f 1 = 1
axiom f_second_derivative (x : ℝ) : deriv (deriv f x) > -2

theorem solution_set :
  {x : ℝ | f (Real.log 2 (abs (3^x - 1))) < 3 - Real.log (Real.sqrt 2) (abs (3^x - 1))} = (-∞, 0) ∪ (0, 1) :=
sorry

end solution_set_l806_806767


namespace combined_mean_of_scores_l806_806547

theorem combined_mean_of_scores (f s : ℕ) (mean_1 mean_2 : ℕ) (ratio : f = (2 * s) / 3) 
  (hmean1 : mean_1 = 90) (hmean2 : mean_2 = 75) :
  (135 * s) / ((2 * s) / 3 + s) = 81 := 
by
  sorry

end combined_mean_of_scores_l806_806547


namespace find_dihedral_angle_cosine_l806_806264

-- Define the cube and coordinate points.
def point := ℝ × ℝ × ℝ
def D : point := (0, 0, 0)
def A : point := (1, 0, 0)
def C : point := (0, 1, 0)
def D1 : point := (0, 0, 1)
def M : point := (1, 1, 0.5)

-- Define the cosine value between the dihedral angles.
theorem find_dihedral_angle_cosine :
  let n1 := (1, 1, 1) / -- normal vector for ACD1 plane
      sqrt (1 * 1 + 1 * 1 + 1 * 1),
      n2 := (-1, 2, 2) / -- normal vector for MCD1 plane
      sqrt ((-1) * (-1) + 2 * 2 + 2 * 2)
  in n1 • n2 = 1 / sqrt 3 := 
sorry

end find_dihedral_angle_cosine_l806_806264


namespace Maryann_frees_all_friends_in_42_minutes_l806_806918

-- Definitions for the problem conditions
def time_to_pick_cheap_handcuffs := 6
def time_to_pick_expensive_handcuffs := 8
def number_of_friends := 3

-- Define the statement we need to prove
theorem Maryann_frees_all_friends_in_42_minutes :
  (time_to_pick_cheap_handcuffs + time_to_pick_expensive_handcuffs) * number_of_friends = 42 :=
by
  sorry

end Maryann_frees_all_friends_in_42_minutes_l806_806918


namespace second_negative_integer_l806_806436

theorem second_negative_integer (n : ℤ) (h : -11 * n + 5 = 93) : n = -8 :=
by
  sorry

end second_negative_integer_l806_806436


namespace time_to_fill_with_leak_l806_806558

theorem time_to_fill_with_leak (hA : ∀ t, t / 4) (hL : ∀ t, t / 8) : 8 := 
by
  sorry

end time_to_fill_with_leak_l806_806558


namespace range_of_y_for_x_gt_2_l806_806466

theorem range_of_y_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → 0 < 2 / x ∧ 2 / x < 1) :=
by 
  -- Proof is omitted
  sorry

end range_of_y_for_x_gt_2_l806_806466


namespace intersection_point_of_lines_l806_806192

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end intersection_point_of_lines_l806_806192


namespace percentage_defective_l806_806262

namespace DefectiveMeters

-- Conditions
def totalMetersExamined : ℕ := 8000
def defectiveMetersRejected : ℕ := 4

-- Proof problem: determine the percentage of defective meters
theorem percentage_defective : 
    (defectiveMetersRejected / totalMetersExamined.toRat) * 100 = 0.05 :=
by
  sorry

end DefectiveMeters

end percentage_defective_l806_806262


namespace length_XY_l806_806125

variable (P Q R S X Y : Type) [Field X] [Field Y] [EuclideanSpace X] [EuclideanSpace Y]

/-- Rectangle PQRS with sides PQ = 5 and QR = 12, and segment XY through Q perpendicular to diagonal PR. -/
def rectangle_PQRS (PQ QR : ℝ) := PQ = 5 ∧ QR = 12

/-- Line segment XY through Q perpendicular to PR with P on DX and R on DY. -/
def line_segment_XY (XY : ℝ) := ∃ P Q R, PQ = 5 ∧ QR = 12 ∧ XY = 26

theorem length_XY 
  (PQ QR PR XY : ℝ)
  (H1 : rectangle_PQRS PQ QR)
  (H2 : line_segment_XY XY) :
  XY = 26 :=
  sorry

end length_XY_l806_806125


namespace distinct_ab_not_perfect_square_l806_806534

theorem distinct_ab_not_perfect_square (d : ℕ) (h_d_ne : d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 13) :
  ∃ (a b ∈ ({2, 5, 13, d} : Finset ℕ)), a ≠ b ∧ ¬ ∃ (k : ℕ), k^2 = a * b - 1 := by
sorry

end distinct_ab_not_perfect_square_l806_806534


namespace triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l806_806772

theorem triangle_side_square_sum_eq_three_times_centroid_dist_square_sum
  {A B C O : EuclideanSpace ℝ (Fin 2)}
  (h_centroid : O = (1/3 : ℝ) • (A + B + C)) :
  (dist A B)^2 + (dist B C)^2 + (dist C A)^2 =
  3 * ((dist O A)^2 + (dist O B)^2 + (dist O C)^2) :=
sorry

end triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l806_806772


namespace avg_comm_avg_does_not_distribute_over_mul_mul_distributes_over_avg_l806_806722

noncomputable def avg (x y : ℝ) : ℝ := (x + y) / 2

theorem avg_comm (x y : ℝ) : avg x y = avg y x :=
by
  unfold avg
  rw [add_comm]

theorem avg_does_not_distribute_over_mul (x y z : ℝ) : avg x (y * z) ≠ avg x y * avg x z :=
by
  unfold avg
  intro h
  have : (x + y * z) / 2 = ((x + y) * (x + z)) / 4,
    from h
  sorry

theorem mul_distributes_over_avg (x y z : ℝ) : x * avg y z = avg (x * y) (x * z) :=
by
  unfold avg
  rw [mul_add, mul_div]
  congr
  rw [add_comm]

end avg_comm_avg_does_not_distribute_over_mul_mul_distributes_over_avg_l806_806722


namespace hugo_wins_given_first_roll_four_l806_806866

/-- Define the probability space and events of the dice game,
    and prove that the probability of Hugo's first roll being 4, given that he won the game,
    is equivalent to (1 / 6) * Ψ where Ψ is the calculated probability based on the tie-break rules. -/
theorem hugo_wins_given_first_roll_four 
  (Hugo_rolls : ℕ → ℕ)
  (other_rolls : ℕ → ℕ → ℕ)
  (P : set (ℕ → ℕ) → ℝ)
  (H1 : ℕ)
  (W : Prop)
  (Ψ : ℝ) :
  (H1 = 4 ∧ W) → 
  P {x | x Hugo_rolls 0 = 4 ∧ W} = (1 / 6) * Ψ :=
by 
  sorry

end hugo_wins_given_first_roll_four_l806_806866


namespace part1_part2_l806_806549

theorem part1 (n : ℕ) (h : n > 0) : 
  (sqrt (1 - (2 * n - 1) / n^2) = (n - 1) / n) :=
sorry

theorem part2 : 
  (sqrt (1 - 199 / 10000) = 99 / 100) :=
sorry

end part1_part2_l806_806549


namespace bacteria_population_at_2_15_l806_806684

noncomputable def bacteria_at_time (initial_pop : ℕ) (start_time end_time : ℕ) (interval : ℕ) : ℕ :=
  initial_pop * 2 ^ ((end_time - start_time) / interval)

theorem bacteria_population_at_2_15 :
  let initial_pop := 50
  let start_time := 0  -- 2:00 p.m.
  let end_time := 15   -- 2:15 p.m.
  let interval := 4
  bacteria_at_time initial_pop start_time end_time interval = 400 := sorry

end bacteria_population_at_2_15_l806_806684


namespace main_theorem_l806_806718

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 16 \ finset.singleton 0, (x - complex.exp ((2 * real.pi * complex.I) * ((k : ℂ) / 17) + (2 * real.pi * complex.I / 17)))

lemma exp_eq_roots_of_unity (x : ℂ) (n : ℕ) : (complex.exp (2 * real.pi * complex.I * (x / n)))^n = 1 := 
begin
  sorry
end

lemma polynomial_expression (x : ℂ) (h1: (P (complex.exp ((2 * real.pi * complex.I * (x / 20))))) = 1) : 
  ∏ k in finset.range 16 \ finset.singleton 0, (complex.exp (2 * real.pi * (x / 20) * complex.I) - complex.exp ((2 * real.pi * (k + 1) * complex.I) / 17)) = 1 :=
begin
  sorry
end

theorem main_theorem : ∏ j in finset.range 19 \ finset.singleton 0, ∏ k in finset.range 16 \ finset.singleton 0, (complex.exp (2 * real.pi * (j : ℂ) * complex.I / 20) - complex.exp (2 * real.pi * ((k : ℂ) * complex.I + complex.I) / 17)) = 1 :=
begin
  sorry
end

end main_theorem_l806_806718


namespace maximum_number_of_buses_l806_806846

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l806_806846


namespace greatest_power_of_two_divides_l806_806189

theorem greatest_power_of_two_divides (x y : ℕ) (h1 : x = 12 ^ 1002) (h2 : y = 6 ^ 501) :
  greatest_power_of_two (x - y) = 502 :=
sorry

end greatest_power_of_two_divides_l806_806189


namespace max_distance_rearrangement_l806_806551

theorem max_distance_rearrangement :
  ∃ n, (n = 670) ∧ ∀ P : Fin 2013 → Fin 2013,
  (∀ i j : Fin 2013, i ≠ j → abs (i.val - j.val) ≤ n → abs (P i.val - P j.val) > n) := 
begin
  -- Use tactics to eventually prove
  sorry
end

end max_distance_rearrangement_l806_806551


namespace dyck_paths_length_2n_l806_806220

-- Define Dyck Path of length 2n
structure DyckPath (n : Nat) := 
(start : Fin (2*n+1) × Fin n) 
(length : Nat)
(directions : List (Bool))
(never_below_x_axis : ∀ i < length, (steps i).snd ≥ 0)

def count_dyck_paths (n : Nat) : Nat := 
∑ k in (Finset.range n), (Catalan k) * (Catalan (n - k - 1))

theorem dyck_paths_length_2n (n : Nat) : 
∑ k in (Finset.range n), Catalan k * Catalan (n - k - 1) = Nat.choose (2 * n)  n - Nat.choose (2*n) (n+1) :=
by
  sorry

end dyck_paths_length_2n_l806_806220


namespace rate_of_increase_in_price_of_corn_l806_806165

variable (x : ℝ)

theorem rate_of_increase_in_price_of_corn (corn_initial wheat_initial : ℝ) (wheat_decrease_rate final_price : ℝ) :
  corn_initial = 3.20 →
  wheat_initial = 10.80 →
  wheat_decrease_rate = x * (Real.sqrt 2) - x →
  final_price = 10.218563187844456 →
  ∃ c : ℝ, c = x * (Real.sqrt 2) - x :=
by
  intros h_corn_initial h_wheat_initial h_wheat_decrease_rate h_final_price
  use x * (Real.sqrt 2) - x
  rw [h_wheat_decrease_rate, h_corn_initial, h_wheat_initial, h_final_price]
  sorry

end rate_of_increase_in_price_of_corn_l806_806165


namespace mary_hourly_rate_l806_806543

theorem mary_hourly_rate (R : ℝ) (h1 : 0 ≤ R)
  (h2 : ∀ hours : ℝ, 0 ≤ hours → hours ≤ 80 → 
    (hours ≤ 20 → earnings_from_hours hours R = hours * R) ∧ 
    (hours > 20 → earnings_from_hours hours R = 20 * R + (hours - 20) * (1.25 * R))) 
  (h3 : earnings_from_hours 80 R ≤ 760) : R = 8 := 
by sorry

end mary_hourly_rate_l806_806543


namespace four_digit_numbers_with_5_or_7_l806_806391

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l806_806391


namespace find_m_mul_t_l806_806902

-- Mathematical conditions as Lean definitions
def g : ℝ → ℝ := sorry

axiom main_axiom : ∀ x y : ℝ, g (g x + y) = g (x^2 + y) + 2 * g x * y

-- Lean statement for the problem
theorem find_m_mul_t :
  let m := 2 in
  let t := (0 + 4) in
  m * t = 8 :=
by
  sorry

end find_m_mul_t_l806_806902


namespace sets_are_equal_l806_806575

def setA : Set ℤ := {a | ∃ m n l : ℤ, a = 12 * m + 8 * n + 4 * l}
def setB : Set ℤ := {b | ∃ p q r : ℤ, b = 20 * p + 16 * q + 12 * r}

theorem sets_are_equal : setA = setB := sorry

end sets_are_equal_l806_806575


namespace solution_set_of_equation_l806_806085

theorem solution_set_of_equation (x : ℝ) : 
  (abs (2 * x - 1) = abs x + abs (x - 1)) ↔ (x ≤ 0 ∨ x ≥ 1) := 
by 
  sorry

end solution_set_of_equation_l806_806085


namespace train_length_is_770_meters_l806_806253

noncomputable def train_length (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) (cross_time_s : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (1000 / 3600)
  relative_speed_ms * cross_time_s

theorem train_length_is_770_meters :
  train_length 46.5 2.5 62.994960403167745 ≈ 770 :=
by
  sorry

end train_length_is_770_meters_l806_806253


namespace sum_of_k_values_max_area_triangle_l806_806609

theorem sum_of_k_values_max_area_triangle :
  ∃ k1 k2 : ℤ, 
  let A := (2, 8) in
  let B := (14, 17) in
  (9 / 12).to_rat * (14 - 2).to_rat ≠ (17 - 8).to_rat → 
  k1 ≠ k2 ∧ 
  let k_sum := abs (11 - k1) + abs (11 - k2) in
  k_sum = 22 :=
begin
  sorry
end

end sum_of_k_values_max_area_triangle_l806_806609


namespace spilled_wax_amount_l806_806052

-- Definitions based on conditions
def car_wax := 3
def suv_wax := 4
def total_wax := 11
def remaining_wax := 2

-- The theorem to be proved
theorem spilled_wax_amount : car_wax + suv_wax + (total_wax - remaining_wax - (car_wax + suv_wax)) = total_wax - remaining_wax :=
by
  sorry


end spilled_wax_amount_l806_806052


namespace general_formula_lambda_value_l806_806333

-- Non-computable section for defining sequences
noncomputable section

-- Define the sequence a_n
def a (n : ℕ) : ℝ :=
  if n = 0 then 1 else (1 / 2) ^ (n - 1)

-- Sum of the first n terms of the sequence a_n
def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

-- Sequence condition
lemma a_rec (n : ℕ) : a (n + 1) = 1 - S n / 2 :=
  sorry

-- General formula proof for a_n
theorem general_formula (n : ℕ) : a n = (1 / 2) ^ (n - 1) :=
  sorry

-- Arithmetic sequence condition
theorem lambda_value (S : ℕ → ℝ) (λ : ℝ) : 
  (∀ n, ((S (n + 1) + λ * (n + 1 + 1 / 2 ^ (n + 1))) - (S n + λ * (n + 1 / 2 ^ n))) = (S 1 + λ * (1 + 1 / 2) - (S 0 + λ * (0 + 1))))
  → λ = 2 :=
  sorry

end general_formula_lambda_value_l806_806333


namespace length_of_train_l806_806204

-- declare constants
variables (L S : ℝ)

-- state conditions
def condition1 : Prop := L = S * 50
def condition2 : Prop := L + 500 = S * 100

-- state the theorem to prove
theorem length_of_train (h1 : condition1 L S) (h2 : condition2 L S) : L = 500 :=
by sorry

end length_of_train_l806_806204


namespace max_buses_in_city_l806_806850

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l806_806850


namespace side_length_of_triangle_l806_806682

-- Define the conditions
variables (O A B C M : Point) (r s : ℝ)
variable [MetricSpace.Point Circle]

-- conditions given in the problem
def circle_with_area (O: Point) (r: ℝ) :=
  ∃ (A B: Point), circle O r = 100 * π

def triangle_equilateral (A: Point) (B: Point) (C: Point) :=
  ∃ (s : ℝ), is_equilateral_triangle(A, B, C, s)

def median_point (A: Point) (M: Point) (B: Point) (C: Point) :=
  M = midpoint B C

-- theorem to be proved
theorem side_length_of_triangle (h1 : circle_with_area O r) (h2 : triangle_equilateral A B C)
  (h3 : median_point A M B C) (h4: dist O A = 5) :  s = 5 :=
by
  sorry

end side_length_of_triangle_l806_806682


namespace row_even_col_odd_contradiction_row_odd_col_even_contradiction_l806_806250

theorem row_even_col_odd_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∃ i : Fin 15, M r i = 2) ∧ 
      (∀ c : Fin 15, ∀ j : Fin 20, M j c = 5)) := 
sorry

theorem row_odd_col_even_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∀ i : Fin 15, M r i = 5) ∧ 
      (∀ c : Fin 15, ∃ j : Fin 20, M j c = 2)) := 
sorry

end row_even_col_odd_contradiction_row_odd_col_even_contradiction_l806_806250


namespace volume_of_given_sphere_l806_806637

def volume_of_sphere {D : ℝ} (d : D) (V : ℝ → ℝ) : Prop :=
  d = 10 →
  V = (λ r, (4 / 3) * π * r^3) →
  V (d / 2) = 500 / 3 * π

theorem volume_of_given_sphere 
  : volume_of_sphere 10 (λ r, (4 / 3) * π * r^3) :=
by
  sorry

end volume_of_given_sphere_l806_806637


namespace perfect_squares_between_50_and_300_l806_806431

theorem perfect_squares_between_50_and_300 : 
  ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 50 < x ∧ x < 300 → ¬ is_square x ∨ (∃ k : ℕ, x = k^2 ∧ 8 ≤ k ∧ k ≤ 17)) :=
begin
  sorry
end

end perfect_squares_between_50_and_300_l806_806431


namespace incorrect_statements_l806_806442

variable {n : ℕ} (x : Fin n → ℝ)

-- Define the average of sample data x in group A to be 3
def average_group_A : Prop :=
  (∑ i, x i) / n = 3

-- Define the average of sample data in group B to be 5, with transformation 2x + a
def average_group_B (a : ℝ) : Prop :=
  (∑ i, 2 * x i + a) / n = 5

-- Main theorem to prove incorrect statements
theorem incorrect_statements (hA : average_group_A x) (hB : average_group_B x (-1)) : 
  (¬ (∀ a, average_group_B x a)) ∧
  (variance (λ i, 2 * x i + (-1)) ≠ 2 * variance x) ∧
  (range (λ i, 2 * x i + (-1)) ≠ range x) :=
by
  sorry


end incorrect_statements_l806_806442


namespace max_buses_l806_806860

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l806_806860


namespace stability_of_trivial_solution_l806_806633

open Real

-- Given System Definitions
def system1 (x y : ℝ) : ℝ := -x - 2*y + x^2 * y^2
def system2 (x y : ℝ) : ℝ := x - y/2 - (x^3 * y)/2

-- Lyapunov Function Definition
def V (a x y : ℝ) := a * x^2 + 2 * a * y^2

-- Time Derivative of V
def dV_dt (a x y : ℝ) := 2*a*x*(-x - 2*y + x^2*y^2) + 2*a*y*(x - y/2 - (x^3*y)/2)

theorem stability_of_trivial_solution (a : ℝ) (ha : 0 < a) : 
  ∀ (x y : ℝ), (dV_dt a x y) ≤ 0 := by
  sorry

end stability_of_trivial_solution_l806_806633


namespace polyhedron_volume_l806_806598

theorem polyhedron_volume (h a : ℝ) :
  volume_of_polyhedron_with_inscribed_triangles h a (60 / 180 * π) = (1 / 3) * a^2 * h * real.sqrt 3 :=
sorry

end polyhedron_volume_l806_806598


namespace fixed_point_on_line_l806_806097

theorem fixed_point_on_line (n : ℕ) (vertices : list (ℂ)) (fixed_points : list (ℂ))
  (Hvertices_len : vertices.length = 2 * n)
  (Hfixed_points_len : fixed_points.length = 2 * n - 1)
  (Hvertices_on_circle : ∃ (O : ℂ) (r : ℝ), ∀ v ∈ vertices, complex.abs (v - O) = r)
  (Hsides_through_fixed_points : ∀ i < 2 * n - 1, (∃ fp ∈ fixed_points, 
    ∃ P Q ∈ vertices, fp ∈ line_through P Q)) :
  ∃ fp ∈ fixed_points, ∃ P Q ∈ vertices, fp ∈ line_through P Q :=
sorry

end fixed_point_on_line_l806_806097


namespace maximum_number_of_buses_l806_806847

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l806_806847


namespace polyhedron_faces_same_number_of_edges_l806_806977

theorem polyhedron_faces_same_number_of_edges (n : ℕ) (h1 : 4 ≤ n) :
  ∃ (f1 f2 : ℕ) (h1f1 h1f2 : Prop), f1 ≠ f2 ∧ f1 ≤ n ∧ f2 ≤ n ∧ 
  (3 ≤ face_edges f1) ∧ (3 ≤ face_edges f2) ∧ 
  (face_edges f1 ≤ n - 1) ∧ (face_edges f2 ≤ n - 1) ∧ 
  (face_edges f1 = face_edges f2) := 
by
  sorry

end polyhedron_faces_same_number_of_edges_l806_806977


namespace cat_total_birds_caught_l806_806224

theorem cat_total_birds_caught (day_birds night_birds : ℕ) 
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) :
  day_birds + night_birds = 24 :=
sorry

end cat_total_birds_caught_l806_806224


namespace subtract_largest_unit_fraction_l806_806118

theorem subtract_largest_unit_fraction
  (a b n : ℕ) (ha : a > 0) (hb : b > a) (hn : 1 ≤ b * n ∧ b * n <= a * n + b): 
  (a * n - b < a) := by
  sorry

end subtract_largest_unit_fraction_l806_806118


namespace find_angle_B_find_a_and_c_l806_806050

-- Define the conditions and the target to prove \( B = 45^\circ \)

theorem find_angle_B
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : b^2 = a^2 + c^2 - a * c * sqrt 2)
  (law_of_sines : a * sin A + c * sin C = sqrt 2 * a * sin C + b * sin B) :
  B = π / 4 :=
sorry

-- Define the conditions and the target to prove \( a = 1 + \sqrt{3} \) and \( c = \sqrt{6} \)
theorem find_a_and_c
  (b : ℝ)
  (A : ℝ)
  (sin_A : sin A = (sqrt 2 + sqrt 6) / 4)
  (B : ℝ)
  (C : ℝ)
  (sin_B : sin B = sqrt 2 / 2)
  (sin_C : sin C = sqrt 3 / 2)
  (law_of_sines : b * sin C = c * sin B)
  (hA : A = 5 * π / 12) :
  ∃ a c : ℝ, a = 1 + sqrt 3 ∧ c = sqrt 6 :=
sorry

end find_angle_B_find_a_and_c_l806_806050


namespace last_two_digits_of_fraction_l806_806155

def a := 10^93
def b := 10^31 + 3
def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_fraction : last_two_digits (Int.floor (a / b)) = 8 :=
by
  sorry

end last_two_digits_of_fraction_l806_806155


namespace tangent_length_to_c2_from_c3_l806_806717

theorem tangent_length_to_c2_from_c3
  (r1 r2 : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 7)
  (O1 O2 O3 : Type*) (h_O1O2 : dist O1 O2 = r1 + r2)
  (C1 C2 C3 : Type*) (h_C1 : Center C1 = O1) (h_C2 : Center C2 = O2) (h_C3 : Center C3 = O3)
  (h_tangent1 : IsTangent C1 C3) (h_tangent2 : IsTangent C2 C3) (h_collinear: Collinear O1 O2 O3)
  : ∃ m n p : ℤ, m = 4 ∧ n = 1 ∧ p = 1 ∧ gcd m p = 1 ∧ length_of_tangent C3 C2 = (m * sqrt n) / p :=
begin
  sorry
end

end tangent_length_to_c2_from_c3_l806_806717


namespace firewood_sacks_l806_806716

theorem firewood_sacks (pieces_per_sack : ℕ) (total_pieces : ℕ) (h1 : pieces_per_sack = 20) (h2 : total_pieces = 80) :
  total_pieces / pieces_per_sack = 4 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left zero_lt_twentyofthepieces_per_sack  arith_met_nat.80_eq_20_times_eq.mp;
  clean

end firewood_sacks_l806_806716


namespace relationship_M_N_l806_806802

def M : Set Int := {-1, 0, 1}
def N : Set Int := {x | ∃ a b : Int, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem relationship_M_N : N ⊆ M ∧ N ≠ M := by
  sorry

end relationship_M_N_l806_806802


namespace no_partition_of_positive_integers_l806_806120

theorem no_partition_of_positive_integers :
  ∀ (A B C : Set ℕ), (∀ (x : ℕ), x ∈ A ∨ x ∈ B ∨ x ∈ C) →
  (∀ (x y : ℕ), x ∈ A ∧ y ∈ B → x^2 - x * y + y^2 ∈ C) →
  (∀ (x y : ℕ), x ∈ B ∧ y ∈ C → x^2 - x * y + y^2 ∈ A) →
  (∀ (x y : ℕ), x ∈ C ∧ y ∈ A → x^2 - x * y + y^2 ∈ B) →
  False := 
sorry

end no_partition_of_positive_integers_l806_806120


namespace determine_u_value_l806_806312

noncomputable def quadratic_root_condition (u : ℝ) : Prop := 
  let x := (-15 - real.sqrt 145) / 6
  3 * x^2 + 15 * x + u = 0

theorem determine_u_value : 
  quadratic_root_condition (20 / 3) := 
by
  let x := (-15 - real.sqrt 145) / 6
  -- sorry is used to skip the proof
  sorry

end determine_u_value_l806_806312


namespace solve_xyz_system_l806_806070

theorem solve_xyz_system :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
    (x * (6 - y) = 9) ∧ 
    (y * (6 - z) = 9) ∧ 
    (z * (6 - x) = 9) ∧ 
    x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

end solve_xyz_system_l806_806070


namespace max_norm_c_l806_806008

variables {R : Type*} [field R] [has_norm (euclidean_space R)] [normed_space R]

open_locale big_operators

variables (a b c : euclidean_space R)
variables (h_norm_a : ∥a∥ = 1)
variables (h_norm_b : ∥b∥ = 1)
variables (h_dot_ab : inner_product_space.inner a b = 1/2)
variables (h_inequality : ∥a - b + c∥ ≤ 1)

theorem max_norm_c : ∃ c : euclidean_space R, ∥c∥ = 2 :=
sorry

end max_norm_c_l806_806008


namespace alex_points_l806_806452

variable {x y : ℕ} -- x is the number of three-point shots, y is the number of two-point shots
variable (success_rate_3 success_rate_2 : ℚ) -- success rates for three-point and two-point shots
variable (total_shots : ℕ) -- total number of shots

def alex_total_points (x y : ℕ) (success_rate_3 success_rate_2 : ℚ) : ℚ :=
  3 * success_rate_3 * x + 2 * success_rate_2 * y

axiom condition_1 : success_rate_3 = 0.25
axiom condition_2 : success_rate_2 = 0.20
axiom condition_3 : total_shots = 40
axiom condition_4 : x + y = total_shots

theorem alex_points : alex_total_points x y 0.25 0.20 = 30 :=
by
  -- The proof would go here
  sorry

end alex_points_l806_806452


namespace altitude_contains_x_l806_806512

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806512


namespace geometric_sequence_sum_l806_806330

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q
def cond1 := a 0 + a 1 = 3
def cond2 := a 2 + a 3 = 12
def cond3 := is_geometric_sequence a

theorem geometric_sequence_sum :
  cond1 a →
  cond2 a →
  cond3 a q →
  a 4 + a 5 = 48 :=
by
  intro h1 h2 h3
  sorry

end geometric_sequence_sum_l806_806330


namespace habitable_fraction_of_earth_l806_806832

theorem habitable_fraction_of_earth (total_surface : ℝ) 
  (frac_not_water_covered : ℝ := 1 / 3) 
  (frac_habitable_of_land : ℝ := 1 / 3) 
  (frac_habitable : ℝ := frac_not_water_covered * frac_habitable_of_land) :
  frac_habitable = 1 / 9 :=
by {
  have h1 : frac_not_water_covered = 1 / 3 := rfl,
  have h2 : frac_habitable_of_land = 1 / 3 := rfl,
  have h3 : frac_habitable = frac_not_water_covered * frac_habitable_of_land := rfl,
  rw [h1, h2, h3],
  norm_num
}

end habitable_fraction_of_earth_l806_806832


namespace tangent_curve_line_a_eq_neg1_l806_806022

theorem tangent_curve_line_a_eq_neg1 (a : ℝ) (x : ℝ) : 
  (∀ (x : ℝ), (e^x + a = x) ∧ (e^x = 1) ) → a = -1 :=
by 
  intro h
  sorry

end tangent_curve_line_a_eq_neg1_l806_806022


namespace mean_temperature_is_correct_l806_806140

def temperatures : List ℤ := [-8, -6, -3, -3, 0, 4, -1]
def mean_temperature (temps : List ℤ) : ℚ := (temps.sum : ℚ) / temps.length

theorem mean_temperature_is_correct :
  mean_temperature temperatures = -17 / 7 :=
by
  sorry

end mean_temperature_is_correct_l806_806140


namespace sum_of_squares_first_15_l806_806176

def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_squares_first_15 : sum_of_squares 15 = 3720 :=
by
  sorry

end sum_of_squares_first_15_l806_806176


namespace count_600_to_1100_contains_2_and_5_l806_806812

def contains_digits (n : ℕ) (d1 d2 : ℕ) : Prop :=
  d1 ∈ repr n.to_list ∧ d2 ∈ repr n.to_list

def count_numbers_with_digits (lower upper : ℕ) (d1 d2 : ℕ) : ℕ :=
  ((list.range (upper - lower + 1)).map (λ x, x + lower)).count (λ n, contains_digits n d1 d2)

theorem count_600_to_1100_contains_2_and_5 : count_numbers_with_digits 600 1100 2 5 = 8 := 
sorry

end count_600_to_1100_contains_2_and_5_l806_806812


namespace isosceles_triangle_l806_806261

noncomputable def isosceles_triangle_angle_measure (s : ℝ) (h₀ : 0 < s) (h₁ : s < 1) : ℝ :=
  real.arccos (real.sqrt (2 * (1 - s)))

theorem isosceles_triangle (h₀ : 0 < s) (h₁ : s < 1) : 
  ∃ α : ℝ, α = isosceles_triangle_angle_measure s h₀ h₁ := 
sorry

end isosceles_triangle_l806_806261


namespace simplify_expression_l806_806645

theorem simplify_expression :
  -3 - (+6) - (-5) + (-2) = -3 - 6 + 5 - 2 :=
by
  -- Here is where the proof would go, but we only need the statement
  sorry

end simplify_expression_l806_806645


namespace basil_plants_count_l806_806710

-- Define the number of basil plants and the number of oregano plants
variables (B O : ℕ)

-- Define the conditions
def condition1 : Prop := O = 2 * B + 2
def condition2 : Prop := B + O = 17

-- The proof statement
theorem basil_plants_count (h1 : condition1 B O) (h2 : condition2 B O) : B = 5 := by
  sorry

end basil_plants_count_l806_806710


namespace find_a_plus_b_l806_806009

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

def parallel_condition (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 - a.2 * b.1 = 0)

theorem find_a_plus_b (m : ℝ) (h_parallel: 
  parallel_condition (⟨vector_a.1 + 2 * (vector_b m).1, vector_a.2 + 2 * (vector_b m).2⟩)
                     (⟨2 * vector_a.1 - (vector_b m).1, 2 * vector_a.2 - (vector_b m).2⟩)) :
  vector_a + vector_b (-1/2) = (-3/2, 3) := 
by
  sorry

end find_a_plus_b_l806_806009


namespace g_13_eq_27_l806_806151

noncomputable def g : ℝ → ℝ := sorry

axiom g_cond : ∀ x, g(x + g(x)) = 3 * g(x)
axiom g_one : g(1) = 3

theorem g_13_eq_27 : g(13) = 27 :=
sorry

end g_13_eq_27_l806_806151


namespace area_of_set_of_points_satisfying_condition_l806_806467

noncomputable def area_of_shape : ℝ :=
  2 * real.pi + 4

theorem area_of_set_of_points_satisfying_condition :
  let S := {p : ℝ × ℝ | (p.1^2 + p.2^2 + 2*p.1 + 2*p.2)*(4 - p.1^2 - p.2^2) >= 0} in
  ∃ S : set (ℝ × ℝ), real.area S = area_of_shape :=
begin
  sorry
end

end area_of_set_of_points_satisfying_condition_l806_806467


namespace find_a4_l806_806798

-- Define the sequence according to the given formula.
def sequence (n : ℕ) (hn : n > 0) : ℤ := n^2 - 3 * n - 4

-- State the theorem which we need to prove.
theorem find_a4 : sequence 4 (by decide) = 0 := by
  sorry

end find_a4_l806_806798


namespace sum_in_base_5_l806_806601

theorem sum_in_base_5 (a b : ℕ) (ha : a = 342) (hb : b = 78) : 
  nat.to_digits 5 (a + b) = [3, 1, 4, 0] := by
  sorry

end sum_in_base_5_l806_806601


namespace min_period_k_l806_806566

def has_period {α : Type*} [HasZero α] (r : α) (n : ℕ) : Prop :=
  -- A function definition to check if 'r' has a repeating decimal period of length 'n'
  sorry

theorem min_period_k (a b : ℚ) (h₁ : has_period a 30) (h₂ : has_period b 30) (h₃ : has_period (a - b) 15) :
  ∃ (k : ℕ), k = 6 ∧ has_period (a + k * b) 15 :=
begin
  sorry
end

end min_period_k_l806_806566


namespace part1_a1_union_b_is_correct_part1_a2_union_b_is_correct_part1_a3_union_b_is_correct_part2_range_of_m_l806_806715

-- Definitions of the sets A and B
def A₁ := { x : ℝ | Real.log (x + 1) / Real.log (1/2) ≥ -2 }
def A₂ := { x : ℝ | 1/8 ≤ (1/2)^x ∧ (1/2)^x < 2 }
def A₃ := { x : ℝ | (3 * x - 1) / (x + 1) ≤ 2 ∧ x ≠ -1 }

def B (m : ℝ) := { x : ℝ | 2 * m < x ∧ x < m^2 }

-- Proof Problem Part 1
theorem part1_a1_union_b_is_correct :
  (A₁ ∪ B (-1)) = { x : ℝ | -2 < x ∧ x ≤ 3 } :=
sorry

theorem part1_a2_union_b_is_correct :
  (A₂ ∪ B (-1)) = { x : ℝ | -2 < x ∧ x ≤ 3 } :=
sorry

theorem part1_a3_union_b_is_correct :
  (A₃ ∪ B (-1)) = { x : ℝ | -2 < x ∧ x ≤ 3 } :=
sorry

-- Proof Problem Part 2
theorem part2_range_of_m (A : Set ℝ := { x : ℝ | -1 < x ∧ x ≤ 3 }):
  (∀ m : ℝ, (A ⊆ B m) → (A ≠ B m) → -1/2 ≤ m ∧ m ≤ 2) :=
sorry

end part1_a1_union_b_is_correct_part1_a2_union_b_is_correct_part1_a3_union_b_is_correct_part2_range_of_m_l806_806715


namespace maximum_number_of_buses_l806_806844

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l806_806844


namespace camels_horses_oxen_elephants_l806_806670

theorem camels_horses_oxen_elephants :
  ∀ (C H O E : ℝ),
  10 * C = 24 * H →
  H = 4 * O →
  6 * O = 4 * E →
  10 * E = 170000 →
  C = 4184.615384615385 →
  (4 * O) / H = 1 :=
by
  intros C H O E h1 h2 h3 h4 h5
  sorry

end camels_horses_oxen_elephants_l806_806670


namespace tank_filling_time_l806_806652

-- We define the conditions given in the problem
def pipeA_fill_time := 60 -- Pipe A can fill the tank in 60 minutes
def pipeB_fill_time := 40 -- Pipe B can fill the tank in 40 minutes

def pipeA_rate := 1 / (pipeA_fill_time : ℝ) -- Rate of Pipe A
def pipeB_rate := 1 / (pipeB_fill_time : ℝ) -- Rate of Pipe B

def filled_by_pipeB_half_time (T: ℝ) := (T / 2) * pipeB_rate -- Amount filled by Pipe B in the first half
def filled_by_both_pipes_half_time (T: ℝ) := 
  (T / 2) * (pipeA_rate + pipeB_rate) -- Amount filled by both pipes in the second half

-- Formulate the proposition we want to prove as a Lean theorem statement
theorem tank_filling_time : 
  ∃ (T : ℝ), filled_by_pipeB_half_time T + filled_by_both_pipes_half_time T = 1 ∧ T = 30 :=
by
  sorry

end tank_filling_time_l806_806652


namespace time_to_pass_l806_806632

-- Definitions based on the conditions
def speed_faster_train_kmph : ℝ := 50
def speed_slower_train_kmph : ℝ := 32
def length_faster_train_m : ℝ := 75.006

-- Conversion rate from kmph to m/s
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (5 / 18)

-- Calculated relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph)

-- Theorem to prove
theorem time_to_pass : 
  let time := length_faster_train_m / relative_speed_mps
  time = 15.0012 :=
by
  sorry

end time_to_pass_l806_806632


namespace min_value_fraction_l806_806339

noncomputable def minimum_possible_value (a b c : EuclideanSpace ℝ) (λ μ : ℝ) : ℝ :=
  if (∥a∥ = ∥b∥ ∧ c = λ • a + μ • b ∧ ∥c∥ = 1 + inner a b ∧ inner (a + b) c = 1)
  then 2 - Real.sqrt 2
  else 0 -- or some other incorrect indicator value if conditions aren't met

theorem min_value_fraction (a b c : EuclideanSpace ℝ) (λ μ : ℝ)
  (h1 : ∥a∥ = ∥b∥) 
  (h2 : c = λ • a + μ • b) 
  (h3 : ∥c∥ = 1 + inner a b) 
  (h4 : inner (a + b) c = 1) : 
  minimum_possible_value a b c λ μ = 2 - Real.sqrt 2 := 
sorry

end min_value_fraction_l806_806339


namespace balls_in_boxes_l806_806816

theorem balls_in_boxes :
  let balls := 6
  let boxes := 4
  (number_of_ways_to_distribute_balls (balls) (boxes) = 84) :=
by
  sorry

end balls_in_boxes_l806_806816


namespace smallest_k_l806_806562

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l806_806562


namespace quadrilateral_area_correct_l806_806557

noncomputable def quadrilateral_area (A B C D : ℝ × ℝ) : ℝ :=
  let area_of_triangle := λ (P Q R : ℝ × ℝ),
    0.5 * abs ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))
  in area_of_triangle A B C + area_of_triangle A C D

theorem quadrilateral_area_correct :
  quadrilateral_area (0, 0) (0, 2) (3, 2) (3, 3) = 4.5 :=
by
  sorry

end quadrilateral_area_correct_l806_806557


namespace multiple_choice_question_count_l806_806459

-- Definitions based on the conditions
def true_false_count : ℕ := 5
def multiple_choice_options : ℕ := 4
def total_ways : ℕ := 480

-- The statement we need to prove
theorem multiple_choice_question_count :
  ∃ n : ℕ, (2^true_false_count - 2) * (multiple_choice_options^n) = total_ways ∧ n = 2 :=
begin
  sorry, -- proof placeholder
end

end multiple_choice_question_count_l806_806459


namespace problem_3_5_16_l806_806705

-- Defining the points and properties
variables (A B C D E F G H K L P : Point)
variables (s₁ s₂ s₃ : Square)
variables (h₁ : s₁.1 = A ∧ s₁.2 = B ∧ s₁.3 = C ∧ s₁.4 = D)
variables (h₂ : s₂.1 = D ∧ s₂.2 = E ∧ s₂.3 = F ∧ s₂.4 = G)
variables (h₃ : s₃.1 = F ∧ s₃.2 = H ∧ s₃.3 = L ∧ s₃.4 = K)
variables (midpoint_AK : P = midpoint A K)

-- The theorem to be proved
theorem problem_3_5_16 (h₁ : s₁.1 = A ∧ s₁.2 = B ∧ s₁.3 = C ∧ s₁.4 = D)
    (h₂ : s₂.1 = D ∧ s₂.2 = E ∧ s₂.3 = F ∧ s₂.4 = G)
    (h₃ : s₃.1 = F ∧ s₃.2 = H ∧ s₃.3 = L ∧ s₃.4 = K)
    (midpoint_AK : P = midpoint A K) : 
    is_perpendicular (line_through P E) (line_through C H) ∧ 
    distance P E = (1/2:ℝ) * distance C H :=
sorry

end problem_3_5_16_l806_806705


namespace original_price_of_new_system_l806_806058

variables {old_system_cost new_system_out_of_pocket discount_percentage new_system_partial_cost original_price : ℝ}

-- Given conditions
def old_system_cost := 250
def old_system_value_percentage := 0.80
def new_system_out_of_pocket := 250
def discount_percentage := 0.25
def new_system_partial_cost := old_system_cost * old_system_value_percentage + new_system_out_of_pocket
def discounted_fraction := 1 - discount_percentage

theorem original_price_of_new_system :
  original_price = 600 :=
by
  have h1 : new_system_partial_cost = old_system_cost * old_system_value_percentage + new_system_out_of_pocket := rfl
  -- Total amount John got for the old system
  have h2 : old_system_cost * old_system_value_percentage = 200
    := by norm_num
  -- Total amount John spent on the new system
  have h3 : new_system_partial_cost = 200 + 250
    := by norm_num [h1, h2]
  -- Calculate the original price
  have h4 : discounted_fraction * original_price = new_system_partial_cost
    := by norm_num [new_system_partial_cost, discounted_fraction]
  -- Rearrange to solve for original price
  have h5 : original_price = new_system_partial_cost / discounted_fraction
    := sorry
  exact sorry

end original_price_of_new_system_l806_806058


namespace zoo_animals_left_l806_806168

noncomputable def totalAnimalsLeft (x : ℕ) : ℕ := 
  let initialFoxes := 2 * x
  let initialRabbits := 3 * x
  let foxesAfterMove := initialFoxes - 10
  let rabbitsAfterMove := initialRabbits / 2
  foxesAfterMove + rabbitsAfterMove

theorem zoo_animals_left (x : ℕ) (h : 20 * x - 100 = 39 * x / 2) : totalAnimalsLeft x = 690 := by
  sorry

end zoo_animals_left_l806_806168


namespace problem_condition_holds_l806_806658

theorem problem_condition_holds (x y : ℝ) (h₁ : x + 0.35 * y - (x + y) = 200) : y = -307.69 :=
sorry

end problem_condition_holds_l806_806658


namespace inequality_proof_l806_806323

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c :=
sorry

end inequality_proof_l806_806323


namespace median_perp_segment_half_l806_806066

variables {O A B C D P M N Q R : Type*}
variables [HasDistance O]
variables [HasSegment ℝ O A B C D P M N Q R]

-- Given
axiom square_with_center (O A B C D : Type*) (center : O)
  (square : is_square O A B C D) : Prop

axiom point_on_minor_arc {P: Type*} {CD: ℝ} (point: P)
  (on_minor_arc : is_minor_arc P CD) : Prop

axiom tangents_to_incircle (P : Type*) (tangent1 tangent2 : ℝ)
  (incircle : is_incicle_of_square P tangent1 tangent2) : Prop

axiom intersects_segments (PM PN : ℝ) (BC AD : ℝ)
  (points : intersects PM BC Q) (points : intersects PN AD R) : Prop

-- Prove
theorem median_perp_segment_half {O M N Q R : Type*} :
  square_with_center O A B C D →
  point_on_minor_arc P CD →
  tangents_to_incircle P M N →
  intersects_segments PM PN BC AD →
  perp (median (O M N)) (QR) ∧ length (median (O M N)) = length (QR) / 2 :=
sorry

end median_perp_segment_half_l806_806066


namespace simplify_complex_squaring_l806_806979

theorem simplify_complex_squaring :
  (4 - 3 * Complex.i) ^ 2 = 7 - 24 * Complex.i :=
by
  intro
  sorry

end simplify_complex_squaring_l806_806979


namespace smallest_k_for_min_period_15_l806_806569

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l806_806569


namespace part_one_part_two_l806_806324

-- Part (1)
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

-- Part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : 
  2 * a + b = 8 :=
sorry

end part_one_part_two_l806_806324


namespace surplus_shortage_equation_l806_806212

theorem surplus_shortage_equation (x P : ℕ) (h1 : 9 * x - P = 11) (h2 : 6 * x - P = -16) : 9 * x - 11 = 6 * x + 16 :=
by sorry

end surplus_shortage_equation_l806_806212


namespace four_points_form_parallelogram_l806_806766

theorem four_points_form_parallelogram (n : ℕ) (points : fin n → ℝ × ℝ) :
  n ≥ 4 ∧ (∀ i j k : fin n, i ≠ j ∧ j ≠ k ∧ k ≠ i →
    ∃ l : fin n, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧
      parallelogram (points i) (points j) (points k) (points l))
  → n = 4 :=
by
  sorry

end four_points_form_parallelogram_l806_806766


namespace natural_numbers_satisfying_condition_l806_806764

open Nat

theorem natural_numbers_satisfying_condition (r : ℕ) :
  ∃ k : Set ℕ, k = { k | ∃ s t : ℕ, k = 2^(r + s) * t ∧ 2 ∣ t ∧ 2 ∣ s } :=
by
  sorry

end natural_numbers_satisfying_condition_l806_806764


namespace point_on_altitude_l806_806519

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806519


namespace dihedral_angle_cosine_l806_806631

theorem dihedral_angle_cosine (R1 R2 : ℝ) (h: R1 = 1.5 * R2) : 
  let d := R1 + R2,
  let θ := 2 * Real.arccos (cos (π / 8)),
  Real.cos θ ≈ 0.84 := 
by 
  sorry

end dihedral_angle_cosine_l806_806631


namespace point_X_on_altitude_BH_l806_806496

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806496


namespace complex_root_of_unity_prod_l806_806903

theorem complex_root_of_unity_prod (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 :=
by
  sorry

end complex_root_of_unity_prod_l806_806903


namespace fruit_weights_correct_l806_806942

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806942


namespace fraction_product_simplification_l806_806711

theorem fraction_product_simplification :
  (∏ k in Finset.range 60, (k + 1) / (k + 6)) = 1 / 43680 := 
by
  sorry

end fraction_product_simplification_l806_806711


namespace molecular_weight_CaOH₂_l806_806195

theorem molecular_weight_CaOH₂ :
  let Ca := 40.08    -- atomic weight of calcium (g/mol)
  let O := 16.00     -- atomic weight of oxygen (g/mol)
  let H := 1.01      -- atomic weight of hydrogen (g/mol)
  1 * Ca + 2 * O + 2 * H = 74.10 :=
by
  let Ca := 40.08
  let O := 16.00
  let H := 1.01
  calc
    1 * Ca + 2 * O + 2 * H = 1 * 40.08 + 2 * 16.00 + 2 * 1.01 : by congr; norm_num
                       ... = 40.08 + 32.00 + 2.02            : by norm_num
                       ... = 74.10                           : by norm_num

end molecular_weight_CaOH₂_l806_806195


namespace arrangement_count_l806_806621

def areAdjacent {α : Type} (x y : α) (l : list α) : Prop :=
  ∃ a b c : list α, l = a ++ x :: y :: b ∨ l = a ++ y :: x :: b

def notAdjacent {α : Type} (x y : α) (l : list α) : Prop :=
  ¬ areAdjacent x y l

def totalArrangements (n : ℕ) : ℕ :=
  Nat.fact n

noncomputable def specificArrangements : ℕ :=
  let people : list ℕ := [0, 1, 2, 3, 4, 5, 6] in
  let (A, B, C) : (ℕ, ℕ, ℕ) := (0, 1, 2) in
  let remainingPeople := people.erase_all [A, B, C] in
  let groupedAB := remainingPeople ++ [A :: [B]] in
  let possibleArrangements :=
    list.perm (A :: B :: C :: remainingPeople) groupedAB in
  totalArrangements 7

theorem arrangement_count :
  ∃ l : list ℕ, areAdjacent 0 1 l ∧ notAdjacent 0 2 l ∧ notAdjacent 1 2 l ∧ list.perm l [0,1,2,3,4,5,6] →
  specificArrangements = 960 := sorry

end arrangement_count_l806_806621


namespace part1_part2_l806_806796

-- Define the function f
def f (x m : ℝ) : ℝ := abs (x + m) + abs (2 * x - 1)

-- First part of the problem
theorem part1 (x : ℝ) : f x (-1) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 := 
by sorry

-- Second part of the problem
theorem part2 (m : ℝ) : 
  (∀ x, 3 / 4 ≤ x → x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 := 
by sorry

end part1_part2_l806_806796


namespace maximum_busses_l806_806857

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l806_806857


namespace min_period_k_l806_806565

def has_period {α : Type*} [HasZero α] (r : α) (n : ℕ) : Prop :=
  -- A function definition to check if 'r' has a repeating decimal period of length 'n'
  sorry

theorem min_period_k (a b : ℚ) (h₁ : has_period a 30) (h₂ : has_period b 30) (h₃ : has_period (a - b) 15) :
  ∃ (k : ℕ), k = 6 ∧ has_period (a + k * b) 15 :=
begin
  sorry
end

end min_period_k_l806_806565


namespace smallest_k_for_min_period_15_l806_806568

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l806_806568


namespace point_X_on_altitude_BH_l806_806492

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806492


namespace point_X_on_altitude_BH_l806_806493

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806493


namespace sum_first_60_digits_div_1234_l806_806638

theorem sum_first_60_digits_div_1234 : (∑ i in finset.range 60, (nat.of_digits 10 (list.take 60 (list.drop 1 (decimal_expansion 1 1234)))).digit_sum) = 180 :=
by sorry

end sum_first_60_digits_div_1234_l806_806638


namespace count_prime_10001_in_base_n_l806_806756

theorem count_prime_10001_in_base_n (H : ∀ n : ℕ, n ≥ 2 → Nat.Prime (n^4 + 1) ↔ n = 2) :
  Finset.card (Finset.filter (λ n : ℕ, Nat.Prime (n^4 + 1)) (Finset.range (2 + 1))) = 1 :=
by
  sorry

end count_prime_10001_in_base_n_l806_806756


namespace polynomial_sum_l806_806091

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l806_806091


namespace evaluate_expression_l806_806665

theorem evaluate_expression : 2009 * (2007 / 2008) + (1 / 2008) = 2008 := 
by 
  sorry

end evaluate_expression_l806_806665


namespace X_lies_on_altitude_BH_l806_806498

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806498


namespace log_sum_geometric_sequence_l806_806352

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a n ≠ 0 ∧ a (n + 1) / a n = a 1 / a 0

theorem log_sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geo : geometric_sequence a) 
  (h_eq : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) : 
  log (a 1) + log (a 2) + log (a 3) + log (a 4) + log (a 5) + 
  log (a 6) + log (a 7) + log (a 8) + log (a 9) + log (a 10) + 
  log (a 11) + log (a 12) + log (a 13) + log (a 14) + log (a 15) + 
  log (a 16) + log (a 17) + log (a 18) + log (a 19) + log (a 20) = 50 :=
sorry

end log_sum_geometric_sequence_l806_806352


namespace constant_function_no_decreasing_interval_l806_806995

theorem constant_function_no_decreasing_interval :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 2) → ¬(∃ (a b : ℝ), a < b ∧ ∀ x ∈ set.Ioo a b, f x > f (2 * x - a - b)) :=
 by 
  intros f hf
  sorry

end constant_function_no_decreasing_interval_l806_806995


namespace modulus_calculation_l806_806292

open Complex

def omega : ℂ := 7 + 3 * Complex.i

theorem modulus_calculation : 
  Complex.abs (omega^2 - 4 * omega + 13) = Real.sqrt 1525 := 
by
  sorry

end modulus_calculation_l806_806292


namespace mean_median_mode_equals_x_l806_806604

theorem mean_median_mode_equals_x (x : ℝ) (data : List ℝ) (h_data : data = [70, 110, x, 50, 60, 210, 100])
  (h_mean: (data.sum / data.length) = x)
  (h_median: data.nth (data.length / 2) = some x)
  (h_mode: ∀ y, data.count x ≥ data.count y) : x = 100 :=
  sorry

end mean_median_mode_equals_x_l806_806604


namespace distance_between_point1_point2_points_not_collinear_l806_806280

-- Definitions for points and distances
structure Point2D where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point2D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def slope (p1 p2 : Point2D) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def collinear (p1 p2 p3 : Point2D) : Prop :=
  slope p1 p2 = slope p2 p3

-- Example points
def point1 : Point2D := { x := -3, y := 4 }
def point2 : Point2D := { x := 0, y := -4 }
def point3 : Point2D := { x := 6, y := 0 }

-- Proof statements
theorem distance_between_point1_point2 :
  distance point1 point2 = real.sqrt 73 :=
by
  sorry

theorem points_not_collinear :
  ¬ collinear point1 point2 point3 :=
by
  sorry

end distance_between_point1_point2_points_not_collinear_l806_806280


namespace maximum_busses_l806_806854

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l806_806854


namespace sum_possible_x_if_gx_eq_5_l806_806907

   def g (x : ℝ) : ℝ :=
   if x < 3 then 7 * x + 20 else 3 * x - 18

   theorem sum_possible_x_if_gx_eq_5 :
     g x = 5 → x = -15/7 ∨ x = 23/3 →
     ((-15/7 : ℝ) + (23/3 : ℝ)) = (18/7 : ℝ) :=
   by
     intro h₁ h₂
     sorry
   
end sum_possible_x_if_gx_eq_5_l806_806907


namespace planes_parallel_l806_806348

noncomputable theory
open_locale classical

variables {m n : set Point} -- m and n are lines
variables {α β : set Plane} -- α and β are planes

-- Given conditions as hypothesis
variables (h1 : m ∥ n) (h2 : m ⊆ α) (h3 : n ⊆ β) (h4 : m ⊥ α) (h5 : n ⊥ β)

-- Proof statement
theorem planes_parallel (m n : set Point) (α β : set Plane) (h1 : m ∥ n) (h4 : m ⊥ α) (h5 : n ⊥ β) : α ∥ β :=
sorry

end planes_parallel_l806_806348


namespace election_total_votes_l806_806460

theorem election_total_votes
  (V : ℝ)
  (h1 : 0 ≤ V) 
  (h_majority : 0.70 * V - 0.30 * V = 182) :
  V = 455 := 
by 
  sorry

end election_total_votes_l806_806460


namespace sin_cos_theta_eq_two_fifths_l806_806591

-- Define the hypotheses
variables {θ : ℝ}

-- Define the conditions
def purely_imaginary_condition := (sin θ - 2 * cos θ = 0)

-- Conclude the result
theorem sin_cos_theta_eq_two_fifths (h : purely_imaginary_condition) : sin θ * cos θ = 2 / 5 :=
sorry

end sin_cos_theta_eq_two_fifths_l806_806591


namespace algebra_expression_evaluation_l806_806762

theorem algebra_expression_evaluation (a b : ℝ) (h : a + 3 * b = 4) : 2 * a + 6 * b - 1 = 7 := by
  sorry

end algebra_expression_evaluation_l806_806762


namespace find_b_for_hyperbola_l806_806364

def hyperbola_asymptote_b_value : Prop :=
  ∀ (b : ℝ), 
  (b > 0) → 
  (∃ a : ℝ, a > 0 ∧ (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) → 
    (∃ m n : ℝ, 3*x + 2*y = 0 ∧ y = m*x + n)))

theorem find_b_for_hyperbola : 
  hyperbola_asymptote_b_value → b = 3 :=
begin
  sorry
end

end find_b_for_hyperbola_l806_806364


namespace remainder_when_divided_by_x_minus_2_l806_806752

def polynomial (x : ℝ) := x^5 + 2 * x^3 - x + 4

theorem remainder_when_divided_by_x_minus_2 :
  polynomial 2 = 50 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l806_806752


namespace solve_ln_square_plus_ln_lt_zero_l806_806616

open Real

noncomputable def solution_set_inequality : Set ℝ :=
  {x : ℝ | exp (-1) < x ∧ x < 1}

theorem solve_ln_square_plus_ln_lt_zero :
  ∀ x : ℝ, 0 < x → (ln x)^2 + ln x < 0 ↔ x ∈ solution_set_inequality :=
by
  intro x hx
  sorry

end solve_ln_square_plus_ln_lt_zero_l806_806616


namespace perpendicular_line_theorem_l806_806188

-- Mathematical definitions used in the condition.
def Line := Type
def Plane := Type

variables {l m : Line} {π : Plane}

-- Given the predicate that a line is perpendicular to another line on the plane
def is_perpendicular (l m : Line) (π : Plane) : Prop :=
sorry -- Definition of perpendicularity in Lean (abstracted here)

-- Given condition: l is perpendicular to the projection of m on plane π
axiom projection_of_oblique (m : Line) (π : Plane) : Line

-- The Perpendicular Line Theorem
theorem perpendicular_line_theorem (h : is_perpendicular l (projection_of_oblique m π) π) : is_perpendicular l m π :=
sorry

end perpendicular_line_theorem_l806_806188


namespace tan_angle_PAB_correct_l806_806560

noncomputable def tan_angle_PAB (AB BC CA : ℝ) (P inside ABC : Prop) (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop) : ℝ :=
  180 / 329

theorem tan_angle_PAB_correct :
  ∀ (AB BC CA : ℝ)
    (P_inside_ABC : Prop)
    (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop),
    AB = 12 → BC = 15 → CA = 17 →
    (tan_angle_PAB AB BC CA P_inside_ABC PAB_angle_eq_PBC_angle_eq_PCA_angle) = 180 / 329 :=
by
  intros
  sorry

end tan_angle_PAB_correct_l806_806560


namespace find_width_of_floor_l806_806244

noncomputable def width_of_floor (W : ℝ) : Prop :=
  let length_of_floor := 12
  let strip_width := 3
  let length_of_rug := length_of_floor - 2 * strip_width
  let width_of_rug := W - 2 * strip_width
  let area_of_rug := 24
  length_of_rug * width_of_rug = area_of_rug ∧ W = 10

theorem find_width_of_floor : ∃ W : ℝ, width_of_floor W :=
by
  use 10
  unfold width_of_floor
  split
  { norm_num }
  { refl }

end find_width_of_floor_l806_806244


namespace max_buses_constraint_satisfied_l806_806839

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l806_806839


namespace concurrency_of_lines_l806_806989

noncomputable def point := Type

variables (A B C A1 B1 C1 A2 B2 C2 A3 B3 C3 : point)
variables (O I : point)
variables (R r : ℝ)

-- Define the triangle ABC, the points A1, B1, C1 on the incircle, 
-- A2, B2, C2 on the circumcircle, and A3, B3, C3 as tangent points intersections.
axiom triangle_ABC : ∀ x y z : point, x ≠ y ∧ y ≠ z ∧ z ≠ x
axiom circumcircle : ∀ x y z p q r : point, x ∈ circle y z ↔ is_on_circumcircle x y z ∧ p = y ∧ q = z ∧ r = x
axiom incircle : ∀ x y z p q r : point, x ∈ circle y z ↔ is_on_incircle x y z ∧ p = y ∧ q = z ∧ r = x
axiom tangent_points_intersection : ∀ x y z : point, is_tangent_intersection x y z

-- Statement using Lean 4
theorem concurrency_of_lines :
  ∀ (A B C A1 B1 C1 A2 B2 C2 A3 B3 C3 O I : point) (R r : ℝ),
  (triangle_ABC A B C) →
  (circumcircle A B C O R) →
  (incircle A B C I r) →
  intersects_angle_bisectors A2 B2 C2 A B C O →
  intersects_tangents_at_points A3 B3 C3 A2 B2 C2 O →
  incircle_tangents A1 B1 C1 A B C I →
  concurrent_lines A1 A2 B1 B2 C1 C2 A A3 B B3 C C3 :=
sorry

end concurrency_of_lines_l806_806989


namespace dilation_image_l806_806594

theorem dilation_image 
  (z z₀ : ℂ) (k : ℝ) 
  (hz : z = -2 + i) 
  (hz₀ : z₀ = 1 - 3 * I) 
  (hk : k = 3) : 
  (k * (z - z₀) + z₀) = (-8 + 9 * I) := 
by 
  rw [hz, hz₀, hk]
  -- Sorry means here we didn't write the complete proof, we assume it is correct.
  sorry

end dilation_image_l806_806594


namespace remainder_of_product_mod_seven_l806_806301

-- Definitions derived from the conditions
def seq : List ℕ := [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

-- The main statement to prove
theorem remainder_of_product_mod_seven : 
  (seq.foldl (λ acc x => acc * x) 1) % 7 = 0 := by
  sorry

end remainder_of_product_mod_seven_l806_806301


namespace proj_v_eq_v_l806_806307

-- Define the vectors v and w
def v : ℝ × ℝ := (8, -12)
def w : ℝ × ℝ := (-6, 9)

-- Define the projection function
def proj (w v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (dot * w.1, dot * w.2)

-- Statement of the theorem
theorem proj_v_eq_v : proj w v = v :=
by
  sorry

end proj_v_eq_v_l806_806307


namespace smallest_number_is_111111_2_l806_806258

def base9_to_decimal (n : Nat) : Nat :=
  (n / 10) * 9 + (n % 10)

def base6_to_decimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n % 100) / 10) * 6 + (n % 10)

def base4_to_decimal (n : Nat) : Nat :=
  (n / 1000) * 64

def base2_to_decimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n % 100000) / 10000) * 16 + ((n % 10000) / 1000) * 8 + ((n % 1000) / 100) * 4 + ((n % 100) / 10) * 2 + (n % 10)

theorem smallest_number_is_111111_2 :
  let n1 := base9_to_decimal 85
  let n2 := base6_to_decimal 210
  let n3 := base4_to_decimal 1000
  let n4 := base2_to_decimal 111111
  n4 < n1 ∧ n4 < n2 ∧ n4 < n3 := by
    sorry

end smallest_number_is_111111_2_l806_806258


namespace difference_q_r_share_l806_806654

theorem difference_q_r_share (x : ℝ) (h1 : 7 * x - 3 * x = 2800) :
  12 * x - 7 * x = 3500 :=
by
  sorry

end difference_q_r_share_l806_806654


namespace days_at_sister_l806_806476

def total_days_vacation : ℕ := 21
def days_plane : ℕ := 2
def days_grandparents : ℕ := 5
def days_train : ℕ := 1
def days_brother : ℕ := 5
def days_car_to_sister : ℕ := 1
def days_bus_to_sister : ℕ := 1
def extra_days_due_to_time_zones : ℕ := 1
def days_bus_back : ℕ := 1
def days_car_back : ℕ := 1

theorem days_at_sister : 
  total_days_vacation - (days_plane + days_grandparents + days_train + days_brother + days_car_to_sister + days_bus_to_sister + extra_days_due_to_time_zones + days_bus_back + days_car_back) = 3 :=
by
  sorry

end days_at_sister_l806_806476


namespace recipe_calls_for_certain_flour_l806_806101

/-- 
Mary is baking a cake. The recipe calls for a certain amount of flour and 14 cups of sugar. 
She already put in 10 cups of flour and 2 cups of sugar. She needs to add 12 more cups of sugar.
We need to prove how many cups of flour the recipe calls for.
-/
theorem recipe_calls_for_certain_flour
    (f : ℕ) -- a certain amount of flour
    (s : ℕ) -- amount of sugar recipe calls for
    (s_already : ℕ) -- amount of sugar already put in
    (s_add : ℕ) -- amount of sugar needed to add
    (f_already : ℕ) -- amount of flour already put in
    (h_sugar : s = 14)
    (h_s_already : s_already = 2)
    (h_s_add : s_add = 12)
    (h_flour_already : f_already = 10) :
    f = 10 :=
by
  have h : s_already + s_add = s,
  {
    sorry
  }
  exact h_flour_already

end recipe_calls_for_certain_flour_l806_806101


namespace cos_periodic_l806_806649

-- Definitions of premises
def is_trigonometric_function (f : ℝ → ℝ) : Prop := 
  ∃ (g : ℝ → ℝ), g = sin ∨ g = cos ∨ g = tan ∨ g = cot ∨ g = sec ∨ g = csc ∧ f = g

def is_periodic (f : ℝ → ℝ) : Prop := 
  ∃ (T : ℝ), T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Given premises
axiom all_trigonometric_functions_periodic : ∀ (f : ℝ → ℝ), is_trigonometric_function f → is_periodic f
axiom cos_is_trigonometric : is_trigonometric_function cos

-- Prove the conclusion
theorem cos_periodic : is_periodic cos :=
by {
  apply all_trigonometric_functions_periodic,
  exact cos_is_trigonometric,
}

end cos_periodic_l806_806649


namespace mary_animals_count_l806_806917

def initial_lambs := 18
def initial_alpacas := 5
def initial_baby_lambs := 7 * 4
def traded_lambs := 8
def traded_alpacas := 2
def received_goats := 3
def received_chickens := 10
def chickens_traded_for_alpacas := received_chickens / 2
def additional_lambs := 20
def additional_alpacas := 6

noncomputable def final_lambs := initial_lambs + initial_baby_lambs - traded_lambs + additional_lambs
noncomputable def final_alpacas := initial_alpacas - traded_alpacas + 2 + additional_alpacas
noncomputable def final_goats := received_goats
noncomputable def final_chickens := received_chickens - chickens_traded_for_alpacas

theorem mary_animals_count :
  final_lambs = 58 ∧ 
  final_alpacas = 11 ∧ 
  final_goats = 3 ∧ 
  final_chickens = 5 :=
by 
  sorry

end mary_animals_count_l806_806917


namespace multiple_of_first_number_is_eight_l806_806556

theorem multiple_of_first_number_is_eight 
  (a b c k : ℤ)
  (h1 : a = 7) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) 
  (h4 : 7 * k = 3 * c + (2 * b + 5)) : 
  k = 8 :=
by
  sorry

end multiple_of_first_number_is_eight_l806_806556


namespace inverse_proposition_l806_806997

theorem inverse_proposition (a b : ℝ) (h : ab = 0) : (a = 0 → ab = 0) :=
by
  sorry

end inverse_proposition_l806_806997


namespace distance_AB_bounds_l806_806803

noncomputable def distance_AC : ℕ := 10
noncomputable def distance_AD : ℕ := 10
noncomputable def distance_BE : ℕ := 10
noncomputable def distance_BF : ℕ := 10
noncomputable def distance_AE : ℕ := 12
noncomputable def distance_AF : ℕ := 12
noncomputable def distance_BC : ℕ := 12
noncomputable def distance_BD : ℕ := 12
noncomputable def distance_CD : ℕ := 11
noncomputable def distance_EF : ℕ := 11
noncomputable def distance_CE : ℕ := 5
noncomputable def distance_DF : ℕ := 5

theorem distance_AB_bounds (AB : ℝ) :
  8.8 < AB ∧ AB < 19.2 :=
sorry

end distance_AB_bounds_l806_806803


namespace price_on_wednesday_highest_and_lowest_prices_profit_calculation_l806_806581
-- Import the whole Mathlib to bring in all necessary libraries

-- Definitions representing conditions in the problem
def opening_price := 27
def shares_bought := 1000
noncomputable def daily_changes : List ℝ := [4, 4.5, -1, -2.5, -6, 2]

def handling_fee := 0.0015
def transaction_tax := 0.001

-- The theorem to prove the price per share at the close of trading on Wednesday
theorem price_on_wednesday : 
  let price := opening_price + daily_changes.take 3 |>.sum 
  price = 34.5 := 
by
  sorry

-- The theorem to prove the highest and lowest prices per share for the week
theorem highest_and_lowest_prices : 
  let prices := List.scanl (λ acc x => acc + x) opening_price daily_changes
  prices.maximum = some 35.5 ∧ prices.minimum = some 26 :=
by
  sorry

-- The theorem to prove Li Ming's profit if he sold all the stock before the close of trading on Saturday
theorem profit_calculation : 
  let buying_cost := shares_bought * opening_price * (1 + handling_fee)
  let selling_price := shares_bought * (opening_price + List.sum daily_changes)
  let selling_fees := selling_price * handling_fee
  let selling_taxes := selling_price * transaction_tax
  let net_selling_price := selling_price - selling_fees - selling_taxes
  let profit := net_selling_price - buying_cost
  profit = 889.5 :=
by
  sorry

end price_on_wednesday_highest_and_lowest_prices_profit_calculation_l806_806581


namespace combined_wages_duration_l806_806203

-- Define the daily wage of X and Y based on their total wage coverage days
def wage_x (S : ℝ) := S / 36
def wage_y (S : ℝ) := S / 45

-- Define the theorem to prove
theorem combined_wages_duration (S : ℝ) : 
  (S / (wage_x S + wage_y S)) = 20 := 
by
  sorry

end combined_wages_duration_l806_806203


namespace bus_driver_weekly_distance_total_l806_806227

theorem bus_driver_weekly_distance_total :
  let monday_hours := 3
  let monday_speed := 12
  let tuesday_hours := 3.5
  let tuesday_speed := 8
  let wednesday_hours := 2.5
  let wednesday_speed := 12
  let thursday_hours := 4
  let thursday_speed := 6
  let friday_hours := 2
  let friday_speed := 12
  let saturday_hours := 3
  let saturday_speed := 15
  let sunday_hours := 1.5
  let sunday_speed := 10
  let total_distance := 
      monday_hours * monday_speed +
      tuesday_hours * tuesday_speed +
      wednesday_hours * wednesday_speed +
      thursday_hours * thursday_speed +
      friday_hours * friday_speed +
      saturday_hours * saturday_speed +
      sunday_hours * sunday_speed
  in total_distance = 202 := by
  sorry

end bus_driver_weekly_distance_total_l806_806227


namespace monotonic_increase_interval_f_sum_of_roots_g_eq_4_l806_806808

def is_monotonic_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def interval_of_monotonic_increase (f : ℝ → ℝ) (I J : set ℝ) : Prop :=
J = { x | ∃ k : ℤ, k * real.pi - real.pi / 3 ≤ x ∧ x ≤ k * real.pi + real.pi / 6 }

noncomputable def f (x : ℝ) : ℝ :=
2 * real.sin (2 * x + real.pi / 6) + 3

theorem monotonic_increase_interval_f :
  interval_of_monotonic_increase f (set.Icc 0 (real.pi / 2)) (set_of (λ x, ∃ k : ℤ, k * real.pi - real.pi / 3 ≤ x ∧ x ≤ k * real.pi + real.pi / 6)) :=
sorry

noncomputable def g (x : ℝ) : ℝ :=
2 * real.sin (4 * x - real.pi / 6) + 3

theorem sum_of_roots_g_eq_4 :
  finset.sum (finset.image (λ k : ℤ, if (k * real.pi / 2 + real.pi / 12 ≤ real.pi / 2) ∧ (k * real.pi / 2 + real.pi / 12 ≥ 0)
  then (k * real.pi / 2 + real.pi / 12) else 0) (finset.range 10))
  = real.pi / 3 :=
sorry

end monotonic_increase_interval_f_sum_of_roots_g_eq_4_l806_806808


namespace remainder_when_500th_number_in_S_divided_by_500_is_364_l806_806898

def S := {n : ℕ // (n.bits.to_finset.card = 10) ∧ ∀ m : ℕ, (m.bits.to_finset.card = 10) → n > m → False }

def N := (Finset.card S < 500) → ∃ n, ∀ m : ℕ, (m ∈ S ∧ Finset.card { x : S | x.val < m } = 499) → n = m

theorem remainder_when_500th_number_in_S_divided_by_500_is_364 : (∃ n ∈ S, Finset.card { x : S // x < n } = 499) → (N % 500 = 364) :=
by sorry

end remainder_when_500th_number_in_S_divided_by_500_is_364_l806_806898


namespace smallest_k_for_min_period_15_l806_806567

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l806_806567


namespace count_valid_three_digit_numbers_l806_806014

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 720 ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → 
    (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ∉ [2, 5, 7, 9])) := 
sorry

end count_valid_three_digit_numbers_l806_806014


namespace garden_width_min_5_l806_806059

theorem garden_width_min_5 (width length : ℝ) (h_length : length = width + 20) (h_area : width * length ≥ 150) :
  width ≥ 5 :=
sorry

end garden_width_min_5_l806_806059


namespace log2_abs_even_and_increasing_l806_806730

theorem log2_abs_even_and_increasing : 
  (∀ x : ℝ, (real.log2 (abs x) = real.log2 (abs (-x))) ∧ (x > 0) → (real.log2 (abs x)) > real.log2 1) :=
sorry

end log2_abs_even_and_increasing_l806_806730


namespace min_candies_to_remove_l806_806182

theorem min_candies_to_remove {n : ℕ} (h : n = 31) : (∃ k, (n - k) % 5 = 0) → k = 1 :=
by
  sorry

end min_candies_to_remove_l806_806182


namespace measure_angle_EBF_20_l806_806876

-- Define points and angles
variables {A B C D E F : Type*}  -- abstract points
variables angle : Type*  -- abstract type for angles

-- Define conditions
-- E lies on line segment AB
axiom E_on_AB : E ∈ line segment A B

-- Triangles are isosceles
axiom isosceles_AED : isosceles_triangle A E D
axiom isosceles_BEC : isosceles_triangle B E C
axiom isosceles_BCF : isosceles_triangle B C F

-- Angle relation 
axiom angle_AED_80 : angle AED = 80
axiom angle_DEC_twice_ADE : angle DEC = 2 * angle ADE

-- Define the measure of angle EBF
axiom angle_measure : angle -> ℝ

-- Theorem to prove
theorem measure_angle_EBF_20 : angle_measure (angle EBF) = 20 :=
by sorry

end measure_angle_EBF_20_l806_806876


namespace rounding_estimation_l806_806573

-- Definitions for rounding actions
def round_up (x : ℕ) : ℕ := x + 1
def round_down (x : ℕ) : ℕ := x - 1

-- The proof problem statement
theorem rounding_estimation (a b c d : ℕ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_d : d > 0) :
  (round_up a : ℝ / round_down b + round_up c - round_up d) < (a / b + c - d) :=
by
  sorry

end rounding_estimation_l806_806573


namespace max_value_expression_l806_806041

theorem max_value_expression :
  ∃ (a b c d : ℕ), 
  (a = 17) ∧ (b = 17) ∧ (c = 17) ∧ (d = 17) ∧
  (∀ (op1 op2 op3 op4 : ℕ → ℕ → ℕ),
    (op1 = Nat.mul ∨ op1 = Nat.add ∨ op1 = Nat.sub ∨ op1 = Nat.div) ∧
    (op2 = Nat.mul ∨ op2 = Nat.add ∨ op2 = Nat.sub ∨ op2 = Nat.div) ∧
    (op3 = Nat.mul ∨ op3 = Nat.add ∨ op3 = Nat.sub ∨ op3 = Nat.div) ∧
    (op4 = Nat.mul ∨ op4 = Nat.add ∨ op4 = Nat.sub ∨ op4 = Nat.div) ∧
    (op1 ≠ op2) ∧ (op1 ≠ op3) ∧ (op1 ≠ op4) ∧
    (op2 ≠ op1) ∧ (op2 ≠ op3) ∧ (op2 ≠ op4) ∧
    (op3 ≠ op1) ∧ (op3 ≠ op2) ∧ (op3 ≠ op4) ∧
    (op4 ≠ op1) ∧ (op4 ≠ op2) ∧ (op4 ≠ op3) →
    let expr := op1 a b + op2 b c + op3 c d + op4 d a in
    expr = 305) :=
by
  sorry

end max_value_expression_l806_806041


namespace martha_total_cost_l806_806627

def weight_cheese : ℝ := 1.5
def weight_meat : ℝ := 0.55    -- converting grams to kg
def weight_pasta : ℝ := 0.28   -- converting grams to kg
def weight_tomatoes : ℝ := 2.2

def price_cheese_per_kg : ℝ := 6.30
def price_meat_per_kg : ℝ := 8.55
def price_pasta_per_kg : ℝ := 2.40
def price_tomatoes_per_kg : ℝ := 1.79

def tax_cheese : ℝ := 0.07
def tax_meat : ℝ := 0.06
def tax_pasta : ℝ := 0.08
def tax_tomatoes : ℝ := 0.05

def total_cost : ℝ :=
  let cost_cheese := weight_cheese * price_cheese_per_kg * (1 + tax_cheese)
  let cost_meat := weight_meat * price_meat_per_kg * (1 + tax_meat)
  let cost_pasta := weight_pasta * price_pasta_per_kg * (1 + tax_pasta)
  let cost_tomatoes := weight_tomatoes * price_tomatoes_per_kg * (1 + tax_tomatoes)
  cost_cheese + cost_meat + cost_pasta + cost_tomatoes

theorem martha_total_cost : total_cost = 19.9568 := by
  sorry

end martha_total_cost_l806_806627


namespace max_value_of_m_l806_806440

open Real

theorem max_value_of_m : (∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₂ < 0 → (x₂ * exp x₁ - x₁ * exp x₂) / (exp x₂ - exp x₁) > 1) → (∀ m, m ≥ 0 → ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₁ < m ∧ x₂ < m) :=
begin
  sorry
end

end max_value_of_m_l806_806440


namespace trajectory_is_ray_l806_806162

-- Definitions for points and the distance function.
structure Point where
  x : ℝ
  y : ℝ

def dist (A B : Point) : ℝ := Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- Points M and N
def M : Point := {x := 1, y := 0}
def N : Point := {x := 3, y := 0}

-- Definition of point P satisfying the given condition.
def validPointP (P : Point) : Prop := abs (dist P M - dist P N) = 2

-- Theorem stating the trajectory of P is a ray.
theorem trajectory_is_ray (P : Point) (h : validPointP P) : -- Fill in the appropriate Prop
 := sorry

end trajectory_is_ray_l806_806162


namespace asymptote_of_hyperbola_l806_806368

-- Define the parabola as y^2 = 8x
def parabola (y x : ℝ) : Prop := y^2 = 8 * x

-- Define the hyperbola as x^2/a^2 - y^2 = 1
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 = 1

-- Define the point M with distance constraint |MF| = 5
def distance_FM (M F : ℝ × ℝ) : ℝ := (real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2))

-- Assume the focus of the parabola is F(2, 0)
def F : ℝ × ℝ := (2, 0)

-- Assume point M (3, ±2sqrt(6))
def M : ℝ × ℝ := (3, 2 * real.sqrt 6)

-- The theorem to prove
theorem asymptote_of_hyperbola (a : ℝ) (h_para : parabola M.2 M.1) (h_hyper : hyperbola M.1 M.2 a) (h_dist : distance_FM M F = 5) : 
  ( ∃ m n : ℝ, (5 * m - 3 * n = 0) ∨ (5 * m + 3 * n = 0) ) := 
sorry

end asymptote_of_hyperbola_l806_806368


namespace odd_function_k_eq_1_range_of_g_no_such_lambda_l806_806082

variable {α : Type*}

-- Assuming the necessary context and definitions
def f (x : ℝ) (a : ℝ) (k : ℝ) := k * a^x - a^(-x)
def g (x : ℝ) (a : ℝ) := a^(2*x) + a^(-2*x) - 2 * f x a 1

theorem odd_function_k_eq_1 {a k : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x, f (-x) a k = -f x a k) :
  k = 1 := 
sorry

theorem range_of_g {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f 1 a 1 = 15 / 4) :
  ∀ x ∈ Icc (0 : ℝ) 1, 1 ≤ g x a ∧ g x a ≤ 137 / 16 :=
sorry

theorem no_such_lambda {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f 1 a 1 = 15 / 4) :
  ¬ ∃ (λ : ℕ), ∀ (x : ℝ), x ∈ Icc (-1 / 2) (1 / 2) → f (2 * x) a 1 ≥ λ * f x a 1 :=
sorry

end odd_function_k_eq_1_range_of_g_no_such_lambda_l806_806082


namespace fruit_weights_l806_806934

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806934


namespace probability_after_50_bell_rings_l806_806121

noncomputable def game_probability : ℝ :=
  let p_keep_money := (1 : ℝ) / 4
  let p_give_money := (3 : ℝ) / 4
  let p_same_distribution := p_keep_money^3 + 2 * p_give_money^3
  p_same_distribution^50

theorem probability_after_50_bell_rings : abs (game_probability - 0.002) < 0.01 :=
by
  sorry

end probability_after_50_bell_rings_l806_806121


namespace expression_eval_l806_806461

theorem expression_eval (a b c d : ℝ) :
  a * b + c - d = a * (b + c - d) :=
sorry

end expression_eval_l806_806461


namespace food_drive_ratio_l806_806237

/-- Mark brings in 4 times as many cans as Jaydon,
Jaydon brings in 5 more cans than a certain multiple of the amount of cans that Rachel brought in,
There are 135 cans total, and Mark brought in 100 cans.
Prove that the ratio of the number of cans Jaydon brought in to the number of cans Rachel brought in is 5:2. -/
theorem food_drive_ratio (J R : ℕ) (k : ℕ)
  (h1 : 4 * J = 100)
  (h2 : J = k * R + 5)
  (h3 : 100 + J + R = 135) :
  J / Nat.gcd J R = 5 ∧ R / Nat.gcd J R = 2 := by
  sorry

end food_drive_ratio_l806_806237


namespace number_of_duty_arrangements_l806_806986

noncomputable def dutyDays := {1, 2, 3, 4}

theorem number_of_duty_arrangements : 
  ∑ (A_day : dutyDays) in ({1, 4} : Finset ℕ), 
    ∑ (B_day : dutyDays) in ({1,2,3,4} \ ({A_day - 1, A_day + 1} : Finset ℕ)),
      2 = 8 := 
by
  sorry

end number_of_duty_arrangements_l806_806986


namespace hyperbola_eccentricity_l806_806799

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (a^2) / (b^2))

theorem hyperbola_eccentricity {b : ℝ} (hb_pos : b > 0)
  (h_area : b = 1) :
  eccentricity 1 b = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l806_806799


namespace mass_of_C_in_pure_CaCO3_l806_806683

def Ca_atomic_mass : ℝ := 40.08
def C_atomic_mass : ℝ := 12.01
def O_atomic_mass : ℝ := 16.00

def molar_mass_CaCO3 : ℝ := Ca_atomic_mass + C_atomic_mass + 3 * O_atomic_mass
def mass_percentage_C_in_CaCO3 : ℝ := (C_atomic_mass / molar_mass_CaCO3) * 100

def initial_sample_mass : ℝ := 100.0
def impurities_mass : ℝ := 30.0
def pure_CaCO3_mass : ℝ := initial_sample_mass - impurities_mass

theorem mass_of_C_in_pure_CaCO3 : pure_CaCO3_mass = 70 →
  (mass_percentage_C_in_CaCO3 / 100) * pure_CaCO3_mass = 8.4 :=
by
  sorry

end mass_of_C_in_pure_CaCO3_l806_806683


namespace find_y_l806_806984

theorem find_y (t : ℝ) (x y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 3) (h3 : x = -7) : y = 28 :=
by {
  sorry
}

end find_y_l806_806984


namespace min_value_g_l806_806358

-- Define the function f(x)
def f (x : ℝ) (phi : ℝ) : ℝ := 2 * sin (2 * x + phi)

-- Define the condition |phi| < π/2
variable (phi : ℝ)
variable (hphi : abs phi < π / 2)

-- Define the function g(x) as a left shift of f(x) by π/8
def g (x : ℝ) : ℝ := f (x + π / 8) phi

-- Assume that g(x) is symmetric about the y-axis, deducing phi = π/4
axiom g_symmetric : ∀ x, g (-x) = g x

-- Proving the minimum value of g(x) + g(x/2)
theorem min_value_g (x : ℝ) : ∃ m : ℝ, m = -9 / 4 ∧ ∀ y, g y + g(y / 2) ≥ m :=
by
  -- Placeholder for the proof
  sorry

end min_value_g_l806_806358


namespace part_a_part_b_part_c_l806_806663

-- Part (a)
theorem part_a :
  (423134 * 846267 - 423133) / (423133 * 846267 + 423134) = 1 := 
sorry

-- Part (b)
theorem part_b :
  ∃ x : ℕ, 
    (52367 - x) / (47633 + x) = 17 / 83 ∧ 
    x = 35367 := 
sorry

-- Part (c)
theorem part_c :
  (∑ k in (Finset.range 9), 7 / ((7 * k + 2) * (7 * k + 9))) = 245 / 72 := 
sorry

end part_a_part_b_part_c_l806_806663


namespace not_perfect_square_2023_l806_806643

theorem not_perfect_square_2023 : ¬ (∃ x : ℤ, x^2 = 5^2023) := 
sorry

end not_perfect_square_2023_l806_806643


namespace inequality_2_pow_n_gt_n_sq_for_n_5_l806_806628

theorem inequality_2_pow_n_gt_n_sq_for_n_5 : 2^5 > 5^2 := 
by {
    sorry -- Placeholder for the proof
}

end inequality_2_pow_n_gt_n_sq_for_n_5_l806_806628


namespace max_buses_l806_806863

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l806_806863


namespace digits_with_five_or_seven_is_5416_l806_806409

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l806_806409


namespace sin_C_equals_4_over_5_l806_806867

-- Define the right triangle with the given conditions
noncomputable def triangle_ABC := 
  ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧ 
    B = (3, 0) ∧ 
    C = (3, 4) ∧ 
    (B.1 - A.1)^2 + (C.2 - A.2)^2 = 5^2

-- State the goal to prove
theorem sin_C_equals_4_over_5 (A B C : ℝ × ℝ)
  (h_triangle : A = (0,0) ∧ B = (3,0) ∧ C = (3,4) ∧ (B.1 - A.1)^2 + (C.2 - A.2)^2 = 5^2):
  Real.sin (atan (C.2 / C.1)) = 4/5 := 
sorry

end sin_C_equals_4_over_5_l806_806867


namespace total_handshakes_l806_806552

def number_of_students : ℕ := 40

theorem total_handshakes (n : ℕ) (h : n = number_of_students) :
  ∑ i in finset.range n, i = 780 :=
by
  have h1 : n = 40 := h
  rw [h1]
  sorry

end total_handshakes_l806_806552


namespace fruit_weights_l806_806960

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806960


namespace B_pow_150_eq_I_l806_806063

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end B_pow_150_eq_I_l806_806063


namespace sum_of_coeffs_l806_806785

open BigOperators

-- Define a constant condition
axiom coeff_condition : 4 * nat.choose n 2 = 60

-- Define the statement that the sum of all coefficients in the expansion is 729
theorem sum_of_coeffs (n : ℕ) (h : 4 * nat.choose n 2 = 60) : (1 + 2)^n = 729 := by 
  sorry 

end sum_of_coeffs_l806_806785


namespace trajectory_of_M_on_line_segment_l806_806305

variable {M : Type} [metric_space M]
variable (F1 F2 : M)

def fixed_points : M × M := (F1, F2)

def distance_sum_constraint (M : M) (F1 F2 : M) : Prop :=
  dist M F1 + dist M F2 = dist F1 F2

theorem trajectory_of_M_on_line_segment
  (F1 : M) (F2 : M) (h_cond : distance_sum_constraint M F1 F2) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (F1 + t * (F2 - F1)) :=
sorry

end trajectory_of_M_on_line_segment_l806_806305


namespace fruit_weights_l806_806931

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806931


namespace minimum_a_probability_ge_half_l806_806032

theorem minimum_a_probability_ge_half :
  ∃ a : ℕ, (a >= 1 ∧ a <= 48) ∧ (p(a, a + 10) >= 1 / 2) ∧ (∀ b : ℕ, (b >= 1 ∧ b <= 48) ∧ (p(b, b + 10) >= 1 / 2) → a <= b) :=
sorry

noncomputable def p (a b : ℕ) : ℚ :=
((Nat.choose (48 - a) 2 + Nat.choose (a - 1) 2).toRat) / 1653

#eval minimum_a_probability_ge_half

end minimum_a_probability_ge_half_l806_806032


namespace fruit_weights_correct_l806_806940

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806940


namespace fruit_weights_l806_806955

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806955


namespace width_of_domain_g_l806_806828

theorem width_of_domain_g (h : ℝ → ℝ) (dom_h : ∀ x, -9 ≤ x ∧ x ≤ 9 → ∃ y, h y = x) :
  ∃ (width : ℝ), width = 54 ∧ ∀ x, g(x) = h (x / 3) → -27 ≤ x ∧ x ≤ 27 :=
by
  sorry

end width_of_domain_g_l806_806828


namespace complement_intersection_l806_806911

-- Define the universal set U
def U := {1, 2, 3, 4, 5}

-- Define the set A
def A := {1, 3, 5}

-- Define the set B
def B := {3, 4}

-- Define the complement of A with respect to U
def complement_U_A := U \ A

-- Define the intersection of complement_U_A and B
def intersect_complement_U_A_B := complement_U_A ∩ B

-- Statement to be proved
theorem complement_intersection (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 3, 5}) (hB : B = {3, 4}) :
  intersect_complement_U_A_B = {4} :=
by {
  sorry
}

end complement_intersection_l806_806911


namespace projection_length_l806_806615

def S : set (ℝ × ℝ) := 
  { p : ℝ × ℝ | 
    let x := p.1,
        y := p.2 in 
    log 2 (y^2 - y + 2) = 2 * (sin(x)^4) + 2 * (cos(x)^4) ∧ 
    -π / 8 ≤ x ∧ x ≤ π / 4 
  }

theorem projection_length (S : set (ℝ × ℝ)) : 
  ∃ a b : ℝ, a < b ∧ (∀ y ∈ { y | ∃ x, (x, y) ∈ S }, a ≤ y ∧ y ≤ b) ∧ b - a = 2 :=
sorry

end projection_length_l806_806615


namespace sin_beta_value_sin_2alpha_cos_relation_l806_806318

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : cos α = 3 / 5)
  (h6 : cos (β + α) = 5 / 13) :
  sin β = 16 / 65 := 
sorry

theorem sin_2alpha_cos_relation (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2)
  (h5 : cos α = 3 / 5) :
  (sin (2 * α)) / (cos α ^ 2 + cos (2 * α)) = 12 :=
sorry

end sin_beta_value_sin_2alpha_cos_relation_l806_806318


namespace find_S_l806_806905

noncomputable def P (z : ℂ) : ℂ → ℂ := sorry
noncomputable def S (z : ℂ) : ℂ → ℂ := sorry

theorem find_S :
  ∃ P S : ℂ → ℂ,
  (∀ z : ℂ, z^(2023 : ℂ) + 1 = (z^2 + z + 1) * P(z) + S(z)) ∧
  (deg (S : polynomial ℂ) < 2) ∧
  (S(z) = z + 1) :=
begin
  sorry
end

end find_S_l806_806905


namespace sally_balloons_l806_806126

theorem sally_balloons :
  (initial_orange_balloons : ℕ) → (lost_orange_balloons : ℕ) → 
  (remaining_orange_balloons : ℕ) → (doubled_orange_balloons : ℕ) → 
  initial_orange_balloons = 20 → 
  lost_orange_balloons = 5 →
  remaining_orange_balloons = initial_orange_balloons - lost_orange_balloons →
  doubled_orange_balloons = 2 * remaining_orange_balloons → 
  doubled_orange_balloons = 30 :=
by
  intro initial_orange_balloons lost_orange_balloons 
       remaining_orange_balloons doubled_orange_balloons
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h3] at h4
  sorry

end sally_balloons_l806_806126


namespace PR_length_in_square_l806_806039

theorem PR_length_in_square {x y : ℝ} (h : x^2 + y^2 = 300): sqrt (2 * 300) = 24.49 :=
by 
  sorry

end PR_length_in_square_l806_806039


namespace num_four_digit_with_5_or_7_l806_806397

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l806_806397


namespace f_of_5_eq_9_l806_806763

-- Define the function f(x)
def f : ℤ → ℤ
| x := if x >= 10 then x - 2 else f (x + 6)

-- The statement to prove f(5) = 9 given the function definition
theorem f_of_5_eq_9 : f 5 = 9 := 
by
  sorry

end f_of_5_eq_9_l806_806763


namespace find_m_l806_806434

variables (m x y : ℤ)

-- Conditions
def cond1 := x = 3 * m + 1
def cond2 := y = 2 * m - 2
def cond3 := 4 * x - 3 * y = 10

theorem find_m (h1 : cond1 m x) (h2 : cond2 m y) (h3 : cond3 x y) : m = 0 :=
by sorry

end find_m_l806_806434


namespace limit_of_sequence_sum_of_sequence_bounds_l806_806614

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 2 else sqrt (sequence (n - 1) + 8) - sqrt (sequence (n - 1) + 3)

theorem limit_of_sequence:
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (sequence n - 1) < ε := by sorry

theorem sum_of_sequence_bounds (n : ℕ) (hn : 1 ≤ n) :
  n ≤ ∑ k in finset.range n, sequence (k + 1) ∧ ∑ k in finset.range n, sequence (k + 1) ≤ n + 1 := by sorry

end limit_of_sequence_sum_of_sequence_bounds_l806_806614


namespace range_of_a_l806_806447

theorem range_of_a {a : ℝ} (h1 : ∀ x : ℝ, x - a ≥ 0 → 2 * x - 10 < 0) :
  3 < a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l806_806447


namespace cube_has_6_faces_l806_806230

theorem cube_has_6_faces : ¬ (A : ℕ = 8) :=
by
  have cube_faces : ℕ := 6
  have faces := 8
  sorry

end cube_has_6_faces_l806_806230


namespace johns_raise_percentage_increase_l806_806656

def initial_earnings : ℚ := 65
def new_earnings : ℚ := 70
def percentage_increase (initial new : ℚ) : ℚ := ((new - initial) / initial) * 100

theorem johns_raise_percentage_increase : percentage_increase initial_earnings new_earnings = 7.692307692 :=
by
  sorry

end johns_raise_percentage_increase_l806_806656


namespace functional_eq_linear_const_l806_806894

variable {α : Type*} [RealField α]

theorem functional_eq_linear_const (a b : α) (g : α → α)
    (ha : 0 < a ∧ a < 1/2)
    (hb : 0 < b ∧ b < 1/2)
    (hg : Continuous g)
    (hfunc : ∀ x : α, g(g(x)) = a * g(x) + b * x) :
    ∃ c : α, ∀ x : α, g(x) = c * x := by
  sorry

end functional_eq_linear_const_l806_806894


namespace M_cap_N_l806_806801
noncomputable theory

def M (x : ℝ) : Prop := 0 < log (x + 1) ∧ log (x + 1) < 3
def N (x y : ℝ) : Prop := y = sin x ∧ M x

theorem M_cap_N : (∀ y, (∃ x, M x ∧ y = sin x) ↔ y ∈ set.Ioo 0 1 ∪ {1}) := sorry

end M_cap_N_l806_806801


namespace count_four_digit_integers_with_5_or_7_l806_806413

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l806_806413


namespace number_of_perfect_squares_is_10_l806_806422

-- Define the number of integers n such that 50 ≤ n^2 ≤ 300
def count_perfect_squares_between_50_and_300 : ℕ :=
  (finset.Icc 8 17).card

-- Statement to prove
theorem number_of_perfect_squares_is_10 : count_perfect_squares_between_50_and_300 = 10 := by
  sorry

end number_of_perfect_squares_is_10_l806_806422


namespace probability_composite_first_50_l806_806659

noncomputable def is_composite (n : ℕ) : Prop :=
n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n

noncomputable def count_composites (n : ℕ) : ℕ :=
(nat.filter is_composite (list.range' 1 n).to_finset).card

noncomputable def probability_of_composite (n : ℕ) : ℚ :=
(count_composites n) / n

theorem probability_composite_first_50 : probability_of_composite 50 = 34 / 50 := 
by
  sorry

end probability_composite_first_50_l806_806659


namespace incenter_circumcircle_product_l806_806117

theorem incenter_circumcircle_product
  (A B C : Point)
  (O' : Point)  -- Incenter of triangle ABC
  (r R : ℝ)    -- Radii of the inscribed and circumscribed circles respectively
  (hO' : is_incenter O' A B C)  -- O' is the incenter of triangle ABC
  (hInradius : inradius O' A B C = r)  -- r is the radius of the inscribed circle
  (hCircumradius : circumradius A B C = R)  -- R is the radius of the circumscribed circle
  : dist O' A * dist O' B * dist O' C = 4 * R * r^2 := 
begin
  sorry
end

end incenter_circumcircle_product_l806_806117


namespace probability_person_A_three_consecutive_days_l806_806759

noncomputable def charity_event_probability : ℚ :=
  let total_scenarios :=
    Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 3)
  let favorable_scenarios := 4
  favorable_scenarios / total_scenarios

theorem probability_person_A_three_consecutive_days :
  charity_event_probability = 1/5 :=
by
  sorry

end probability_person_A_three_consecutive_days_l806_806759


namespace panthers_second_half_points_l806_806454

theorem panthers_second_half_points (C1 P1 C2 P2 : ℕ) 
  (h1 : C1 + P1 = 38) 
  (h2 : C1 = P1 + 16) 
  (h3 : C1 + C2 + P1 + P2 = 58) 
  (h4 : C1 + C2 = P1 + P2 + 22) : 
  P2 = 7 :=
by 
  -- Definitions and substitutions are skipped here
  sorry

end panthers_second_half_points_l806_806454


namespace distinguishable_colorings_tetrahedron_l806_806286

/-- Each face of a regular tetrahedron is painted either red, white, or blue. 
    Two colorings are considered indistinguishable if two congruent tetrahedra 
    with those colorings can be rotated so that their appearances are identical. 
    Prove that the number of distinguishable colorings is 15. -/
theorem distinguishable_colorings_tetrahedron : 
  let colors := {red, white, blue},
      tetrahedron_faces := 4,
      rotations := permutations (fin tetrahedron_faces) -- possible rotations of the tetrahedron faces
  in distinguishable_colorings tetrahedron_faces colors rotations = 15 := 
sorry

end distinguishable_colorings_tetrahedron_l806_806286


namespace find_A_find_a_l806_806882

variable (A B C a b c : ℝ)

-- Given the condition
axiom cos_A_def : cos A = (b * cos C + c * cos B) / (2 * a)

-- Additional conditions for part (2)
axiom b_eq_c_plus_2 : b = c + 2

axiom area_ABC : (1 / 2) * b * c * sin A = 15 * (Real.sqrt 3) / 4

-- Statements to prove
theorem find_A (hA : cos A = 1 / 2) : A = Real.pi / 3 :=
sorry

theorem find_a (A_eq : A = Real.pi / 3) (h_cos : cos A = 1 / 2) : a = Real.sqrt 19 :=
sorry

end find_A_find_a_l806_806882


namespace poly_sum_correct_l806_806089

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end poly_sum_correct_l806_806089


namespace greatest_x_for_4x_in_factorial_21_l806_806660

-- Definition and theorem to state the problem mathematically
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_x_for_4x_in_factorial_21 : ∃ x : ℕ, (4^x ∣ factorial 21) ∧ ∀ y : ℕ, (4^y ∣ factorial 21) → y ≤ 9 :=
by
  sorry

end greatest_x_for_4x_in_factorial_21_l806_806660


namespace train_speed_l806_806251

def train_length : ℝ := 110
def bridge_length : ℝ := 140
def time_to_cross : ℝ := 14.998800095992321

noncomputable def speed : ℝ := (train_length + bridge_length) / time_to_cross

theorem train_speed :
  speed ≈ 16.67 := 
by
  sorry

end train_speed_l806_806251


namespace faces_meet_at_vertex_of_icosahedron_l806_806379

def icosahedron (n_faces: ℕ) (is_equilateral: Π (i: ℕ), i < n_faces → Prop) : Prop :=
  n_faces = 20 ∧ (∀ i, i < n_faces → is_equilateral i)

theorem faces_meet_at_vertex_of_icosahedron 
  (n_faces: ℕ)
  (is_equilateral: Π (i: ℕ), i < n_faces → Prop)
  (H: icosahedron n_faces is_equilateral) :
  ∀ v, ∃ faces, faces = 5 := 
sorry

end faces_meet_at_vertex_of_icosahedron_l806_806379


namespace triangle_ABC_problem_l806_806470

-- Given definitions
def a_Def : ℝ := sqrt 7
def b_Def : ℝ := 2
def c_Def : ℝ := 3
def A_Def : ℝ := π / 3
def sin_A_Def : ℝ := sqrt 3 / 2

-- The problem statement
theorem triangle_ABC_problem 
  (a : ℝ) (b : ℝ) (c : ℝ) (A : ℝ) (sin_A: ℝ)
  (h1 : a = sqrt 7) 
  (h2 : b = 2)
  (h3 : c = 3)
  (h4 : A = π / 3)
  (h5 : sin_A = sqrt 3 / 2) :
  A = π / 3 ∧ (1 / 2 * b * c * sin_A = 3 * sqrt 3 / 2) :=
by { sorry }

end triangle_ABC_problem_l806_806470


namespace find_x_find_y_find_p_q_r_l806_806662

-- Condition: The number on the line connecting two circles is the sum of the two numbers in the circles.

-- For part (a):
theorem find_x (a b : ℝ) (x : ℝ) (h1 : a + 4 = 13) (h2 : a + b = 10) (h3 : b + 4 = x) : x = 5 :=
by {
  -- Proof can be filled in here to show x = 5 by solving the equations.
  sorry
}

-- For part (b):
theorem find_y (w y : ℝ) (h1 : 3 * w + w = y) (h2 : 6 * w = 48) : y = 32 := 
by {
  -- Proof can be filled in here to show y = 32 by solving the equations.
  sorry
}

-- For part (c):
theorem find_p_q_r (p q r : ℝ) (h1 : p + r = 3) (h2 : p + q = 18) (h3 : q + r = 13) : p = 4 ∧ q = 14 ∧ r = -1 :=
by {
  -- Proof can be filled in here to show p = 4, q = 14, r = -1 by solving the equations.
  sorry
}

end find_x_find_y_find_p_q_r_l806_806662


namespace frisbee_price_l806_806249

theorem frisbee_price 
  (total_frisbees : ℕ)
  (frisbees_at_3 : ℕ)
  (price_x_frisbees : ℕ)
  (total_revenue : ℕ) 
  (min_frisbees_at_x : ℕ)
  (price_at_3 : ℕ) 
  (n_min_at_x : ℕ)
  (h1 : total_frisbees = 60)
  (h2 : price_at_3 = 3)
  (h3 : total_revenue = 200)
  (h4 : n_min_at_x = 20)
  (h5 : min_frisbees_at_x >= n_min_at_x)
  : price_x_frisbees = 4 :=
by
  sorry

end frisbee_price_l806_806249


namespace volume_of_parallelepiped_l806_806755

theorem volume_of_parallelepiped (x y z : ℝ)
  (h1 : (x^2 + y^2) * z^2 = 13)
  (h2 : (y^2 + z^2) * x^2 = 40)
  (h3 : (x^2 + z^2) * y^2 = 45) :
  x * y * z = 6 :=
by 
  sorry

end volume_of_parallelepiped_l806_806755


namespace max_one_person_correct_seat_l806_806226

theorem max_one_person_correct_seat (n m : ℕ) (ticket_sold : Finset (Fin n × Fin m)) :
  (∀ (i : Fin n), ∃ j ∈ ticket_sold, (j.1 = i ∨ j.2 = i)) →
  (∀ (i : Fin m), ∃ j ∈ ticket_sold, (j.1 = i ∨ j.2 = i)) →
  (∃ k : ℕ, k = 1 ∧ (∀ arranging, ∃ k <= mn, people in correct seats)) :=
sorry

end max_one_person_correct_seat_l806_806226


namespace meaningful_expression_l806_806143

theorem meaningful_expression (x : ℝ) : (1 / Real.sqrt (x + 2) > 0) → (x > -2) := 
sorry

end meaningful_expression_l806_806143


namespace coloring_count_l806_806289

variable (colors : Finset ℕ) (A B C D : ℕ)

def valid_colorings (colors : Finset ℕ) : ℕ :=
  let count_a := 7 -- Number of choices for vertex A
  let count_b := 6 -- Number of choices for vertex B (different from A)
  let count_c := 6 -- Number of choices for vertex C (different from B)
  let count_d := 6 -- Number of choices for vertex D (different from A)
  count_a * count_b * count_c * count_d -- Total valid colorings

theorem coloring_count (hcolors : colors.card = 7) :
  valid_colorings colors = 1512 :=
  by
    intros
    unfold valid_colorings
    sorry

end coloring_count_l806_806289


namespace percentage_decrease_area_l806_806835

theorem percentage_decrease_area (r : ℝ) (h1 : 0 < r) :
  let A := Real.pi * r^2,
      r' := 0.5 * r,
      A' := Real.pi * (r'^2)
  in (A - A') / A * 100 = 75 :=
by
  sorry

end percentage_decrease_area_l806_806835


namespace increasing_interval_f_l806_806600

noncomputable def f : ℝ → ℝ := λ x, Real.log (x^2 - x - 2)

def domain_f (x : ℝ) : Prop := (x ∈ Set.Ioo 2 (Real.infinity)) ∨ (x ∈ Set.Ioo (-Real.infinity) (-1))

theorem increasing_interval_f :
  ∀ x, domain_f x → (∃ (a b : ℝ), a < x ∧ x < b) :=
begin
  sorry
end

end increasing_interval_f_l806_806600


namespace roses_in_february_l806_806149

-- Define initial counts of roses
def roses_oct : ℕ := 80
def roses_nov : ℕ := 98
def roses_dec : ℕ := 128
def roses_jan : ℕ := 170

-- Define the differences
def diff_on : ℕ := roses_nov - roses_oct -- 18
def diff_nd : ℕ := roses_dec - roses_nov -- 30
def diff_dj : ℕ := roses_jan - roses_dec -- 42

-- The increment in differences
def inc : ℕ := diff_nd - diff_on -- 12

-- Express the difference from January to February
def diff_jf : ℕ := diff_dj + inc -- 54

-- The number of roses in February
def roses_feb : ℕ := roses_jan + diff_jf -- 224

theorem roses_in_february : roses_feb = 224 := by
  -- Provide the expected value for Lean to verify
  sorry

end roses_in_february_l806_806149


namespace contacts_in_second_box_l806_806255

-- Define the conditions
def cost_per_contact_first_box : ℝ := 25 / 50
def total_cost_second_box : ℝ := 33
def cost_per_contact_chosen_box : ℝ := 1 / 3

-- Define the statement to prove
theorem contacts_in_second_box : (total_cost_second_box / cost_per_contact_chosen_box) = 99 := by
  sorry

end contacts_in_second_box_l806_806255


namespace find_x_l806_806693

theorem find_x (x : ℝ) (hx1 : x > 0) 
  (h1 : 0.20 * x + 14 = (1 / 3) * ((3 / 4) * x + 21)) : x = 140 :=
sorry

end find_x_l806_806693


namespace cannot_interchange_1_and_3_l806_806624

-- Define the initial condition of the tiles
def initial_tiles := [1, 2, 3]

-- Define adjoining poses
def adj (i j : ℕ) : Prop := (i = j + 1) ∨ (j = i + 1)

-- Define a move, slider move condition
def move (state : list ℕ) (i j : ℕ) : list ℕ := 
if adj i j ∧ j < state.length then
  let temp := state.get i in state.set i (state.get j).set j temp
else
  state

-- Define the invariant of the sequence 
def is_valid_perm (l1 l2 : list ℕ) : Prop :=
∃ perms, l2 = perms.foldl move l1

-- The state we want to achieve is [3, 2, 1] or [1, 3, 2]
def desired_1 : list ℕ := [3, 2, 1]
def desired_2 : list ℕ := [1, 3, 2]

theorem cannot_interchange_1_and_3 : 
  ¬ (is_valid_perm initial_tiles desired_1 ∨ is_valid_perm initial_tiles desired_2) :=
sorry

end cannot_interchange_1_and_3_l806_806624


namespace correct_option_l806_806789

theorem correct_option (
  (a_n : ℕ → ℕ) -- the sequence a_n
  (S_n : ℕ → ℕ) -- the sequence S_n is the sum of the first n terms of {a_n}
  (T_n : ℕ → ℕ) -- the sequence T_n is the sum of {a_n * S_n}
  (h1 : ∀ n, a_n n = 0 ∨ a_n n = 1)
  (h2 : ∀ n, S_n n = ∑ i in finset.range(n+1), a_n i)
  (h3 : ∀ n, T_n n = ∑ i in finset.range(n+1), a_n i * S_n i)
) :
  ∃ (a_n : ℕ → ℕ) (S_n : ℕ → ℕ),
  (∀ n, S_n n = 1) -- Example: a_n = 1 if n = 1 and 0 otherwise
  ∧ geometric_sequence (S_n) := sorry

end correct_option_l806_806789


namespace intersecting_lines_l806_806194

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end intersecting_lines_l806_806194


namespace box_digit_arrangement_l806_806437

-- Definitions corresponding to the conditions
def boxes : Fin 6 := sorry
def digits : Fin 4 := sorry
def empty_spot : Fin 1 := sorry

-- The theorem statement we want to prove
theorem box_digit_arrangement : 
  ∃ (f : Fin 6 → Option (Fin 4)), 
    (∀ i, f i ≠ some i) ∧ 
    (∃ j, f j = none) ∧ 
    (∃ (perms : Fin 5 → Fin 4), 
      (∀ i, f j = none ∧ ∀ i ≠ j, f i = some (perms i)) ∧ 
      Nat.card (Fin 6) * Nat.card (Fin 5) * Nat.card (Fin 4) * Nat.card (Fin 3) * Nat.card (Fin 2) = 720) := 
sorry

end box_digit_arrangement_l806_806437


namespace sequence_is_arithmetic_l806_806791

variable {α : Type*}

-- Define the progressions and other variables involved.
def geometric_sequence (f : ℕ → α) [group α] [has_pow α ℤ] := 
  ∃ r : α, ∀ n : ℕ, f(n+1) = r * f n

def arithmetic_sequence (f : ℕ → α) := 
  ∃ d : α, ∀ n : ℕ, f(n+1) = f n + d

variable {a : ℕ → ℝ}

-- Given conditions
axiom h_1 : geometric_sequence (λ n, 2^(a n))
axiom h_2 : ∀ n, ∑ i in finset.range n, a i = n^2 + 1
axiom h_3 : a 1 > 0
axiom h_4 : ∀ k ≥ 2, a k = (2/(k-1)) * (∑ i in finset.range (k-1), a (i+1))

-- Prove that {a_n} is an arithmetic sequence
theorem sequence_is_arithmetic : arithmetic_sequence a :=
sorry

end sequence_is_arithmetic_l806_806791


namespace percentage_raise_l806_806542

def hourly_wage_before_raise : ℕ := 40
def work_hours_per_day : ℕ := 8
def work_days_per_week : ℕ := 5
def old_weekly_bills : ℕ := 600
def weekly_personal_trainer : ℕ := 100
def weekly_leftover_after_raise : ℕ := 980

theorem percentage_raise :
  let weekly_hours := work_hours_per_day * work_days_per_week in
  let weekly_earnings_before_raise := weekly_hours * hourly_wage_before_raise in
  let old_leftover := weekly_earnings_before_raise - old_weekly_bills in
  let new_expenses := old_weekly_bills + weekly_personal_trainer in
  let new_weekly_earnings := weekly_leftover_after_raise + new_expenses in
  let earnings_difference := new_weekly_earnings - weekly_earnings_before_raise in
  let percentage_raise := (earnings_difference * 100) / weekly_earnings_before_raise in
  percentage_raise = 5 :=
by
  sorry

end percentage_raise_l806_806542


namespace find_f_7_5_l806_806336

noncomputable def f (x : ℝ) : ℝ := if h : x ∈ (Set.Ioo 2 3) then 3 - x else 0 -- placeholder

axiom ev_f : ∀ x : ℝ, f (-x) = f (x)
axiom anti_per_f : ∀ x : ℝ, f (x) = -f (x + 2)
axiom segment_f (x: ℝ) (h: x ∈ Set.Ioo 2 3) : f(x) = 3 - x

theorem find_f_7_5 : f 7.5 = -0.5 := 
by 
  sorry -- Proof goes here

end find_f_7_5_l806_806336


namespace dihedral_angle_cosine_l806_806630

theorem dihedral_angle_cosine (R1 R2 : ℝ) (h: R1 = 1.5 * R2) : 
  let d := R1 + R2,
  let θ := 2 * Real.arccos (cos (π / 8)),
  Real.cos θ ≈ 0.84 := 
by 
  sorry

end dihedral_angle_cosine_l806_806630


namespace ratio_of_areas_l806_806231

noncomputable theory

-- Definitions based on given conditions
def AB : ℝ := 36
def ratio_AD_AB : ℝ := 5 / 3
def AD : ℝ := ratio_AD_AB * AB

def radius_semi : ℝ := AB / 2
def area_circle : ℝ := Real.pi * radius_semi^2
def area_rectangle : ℝ := AD * AB
def side_square : ℝ := AB
def area_square : ℝ := side_square^2
def area_combined : ℝ := area_circle + area_square

-- Proof statement of the required ratio
theorem ratio_of_areas : area_rectangle / area_combined = 2160 / (324 * Real.pi + 1296) := by
  sorry

end ratio_of_areas_l806_806231


namespace stratified_sampling_l806_806695

-- Definition of conditions as hypothesis
def total_employees : ℕ := 100
def under_35 : ℕ := 45
def between_35_49 : ℕ := 25
def over_50 : ℕ := total_employees - under_35 - between_35_49
def sample_size : ℕ := 20
def sampling_ratio : ℚ := sample_size / total_employees

-- The target number of people from each group
def under_35_sample : ℚ := sampling_ratio * under_35
def between_35_49_sample : ℚ := sampling_ratio * between_35_49
def over_50_sample : ℚ := sampling_ratio * over_50

-- Problem statement
theorem stratified_sampling : 
  under_35_sample = 9 ∧ 
  between_35_49_sample = 5 ∧ 
  over_50_sample = 6 :=
  by
  sorry

end stratified_sampling_l806_806695


namespace unit_digit_of_15_pow_100_l806_806206

-- Define a function to extract the unit digit of a number
def unit_digit (n : ℕ) : ℕ := n % 10

-- Given conditions:
def base : ℕ := 15
def exponent : ℕ := 100

-- Define what 'unit_digit' of a number raised to an exponent means
def unit_digit_pow (base exponent : ℕ) : ℕ :=
  unit_digit (base ^ exponent)

-- Goal: Prove that the unit digit of 15^100 is 5.
theorem unit_digit_of_15_pow_100 : unit_digit_pow base exponent = 5 :=
by
  sorry

end unit_digit_of_15_pow_100_l806_806206


namespace triangle_length_DE_l806_806881

theorem triangle_length_DE 
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (BC : ℝ)
  (angleC : ℝ)
  (BC_length : BC = 40 * Real.sqrt 2)
  (angleC_value : angleC = 45)
  (midpoint_D : D)
  (midpoint_D_def : ∃ (M : Type), MetricSpace.is_midpoint M B C BC D)
  (perp_bisector_of_D : E)
  (perp_bisector_of_D_def : ∃ (P : Type), MetricSpace.on_perpendicular_bisector P A C D E) :
  ∃ (DE : ℝ), DE = 40 :=
sorry

end triangle_length_DE_l806_806881


namespace damaged_polynomial_correct_l806_806225

-- Definition of the damaged polynomial part
def damaged_polynomial (x y : ℝ) : ℝ := -3 * x + y ^ 2

-- The main theorem that encapsulates the problem statement
theorem damaged_polynomial_correct :
  (∀ x y : ℝ, -x + (1 / 3) * y ^ 2 - 2 * (x - (1 / 3) * y ^ 2) = damaged_polynomial x y)
  ∧ (damaged_polynomial (-3) (3 / 2) = 45 / 4) :=
begin
  split,
  { intros x y,
    calc
      -x + (1 / 3) * y ^ 2 - 2 * (x - (1 / 3) * y ^ 2)
          = -x + (1 / 3) * y ^ 2 - (2 * x - 2 * (1 / 3) * y ^ 2) : by ring
      ... = -x + (1 / 3) * y ^ 2 - 2 * x + (2 / 3) * y ^ 2          : by ring
      ... = -3 * x + y ^ 2                                         : by ring },
  { simp [damaged_polynomial],
    norm_num },
end

end damaged_polynomial_correct_l806_806225


namespace arithmetic_seq_contains_geometric_seq_l806_806703

theorem arithmetic_seq_contains_geometric_seq (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (ns : ℕ → ℕ) (k : ℝ), k ≠ 1 ∧ (∀ n, a + b * (ns (n + 1)) = k * (a + b * (ns n)))) ↔ (∃ (q : ℚ), a = q * b) :=
sorry

end arithmetic_seq_contains_geometric_seq_l806_806703


namespace geom_seq_expression_l806_806351

noncomputable def geometric_sequence : Nat → Rat
| n => if h: n > 0 then (2 ^ (n - 1)) / 3 else 0

theorem geom_seq_expression (a_n : Nat → Rat) (S₆ : Rat) (a₁ : Rat) (q : Rat) :
  (∑ i in Finset.range 6, geometric_sequence (i + 1)) = 21 →
  (4 * a₁) = 2 * (3 / 2 * a₁ * q) + a₁ * q →
  a₁ = 1 / 3 ∧ q = 2 →
  ∀ n ≥ 1, a_n n = geometric_sequence n :=
by
  intro h_sum h_arith h_a1_q
  sorry

end geom_seq_expression_l806_806351


namespace machine_p_vs_machine_q_l806_806541

variable (MachineA_rate MachineQ_rate MachineP_rate : ℝ)
variable (Total_sprockets : ℝ := 550)
variable (Production_rate_A : ℝ := 5)
variable (Production_rate_Q : ℝ := MachineA_rate + 0.1 * MachineA_rate)
variable (Time_Q : ℝ := Total_sprockets / Production_rate_Q)
variable (Time_P : ℝ)
variable (Difference : ℝ)

noncomputable def production_times_difference (MachineA_rate MachineQ_rate MachineP_rate : ℝ) : ℝ :=
  let Production_rate_Q := MachineA_rate + 0.1 * MachineA_rate
  let Time_Q := Total_sprockets / Production_rate_Q
  let Difference := Time_P - Time_Q
  Difference

theorem machine_p_vs_machine_q : 
  Production_rate_A = 5 → 
  Total_sprockets = 550 →
  Production_rate_Q = 5.5 →
  Time_Q = 100 →
  MachineP_rate = MachineP_rate →
  Time_P = Time_P →
  Difference = (Time_P - Time_Q) :=
by
  intros
  sorry

end machine_p_vs_machine_q_l806_806541


namespace Alex_can_make_100_dresses_l806_806698

theorem Alex_can_make_100_dresses 
  (num_friends : ℕ) (meters_per_friend : ℕ) (total_silk : ℕ)
  (silk_per_dress : ℕ) 
  (h1 : num_friends = 5) 
  (h2 : meters_per_friend = 20)
  (h3 : total_silk = 600)
  (h4 : silk_per_dress = 5) : 
  (total_silk - num_friends * meters_per_friend) / silk_per_dress = 100 := 
by 
  simp [h1, h2, h3, h4]
  sorry

end Alex_can_make_100_dresses_l806_806698


namespace derivative_f_l806_806216

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.exp (Real.sin x))

theorem derivative_f (x : ℝ) : deriv f x = ((Real.cos x)^2 - Real.sin x) * (Real.exp (Real.sin x)) :=
by
  sorry

end derivative_f_l806_806216


namespace hannah_time_difference_l806_806809

noncomputable def time_taken (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

def distance_monday := 9  -- kilometers
def pace_monday := 6 -- minutes per kilometer

def distance_wednesday := 4816 / 1000  -- converting meters to kilometers
def pace_wednesday := 5.5 -- minutes per kilometer

def distance_friday := 2095 / 1000  -- converting meters to kilometers
def pace_friday := 7 -- minutes per kilometer

def time_monday := time_taken distance_monday pace_monday
def time_wednesday := time_taken distance_wednesday pace_wednesday
def time_friday := time_taken distance_friday pace_friday
def combined_time_wed_fri := time_wednesday + time_friday

def time_difference := time_monday - combined_time_wed_fri

theorem hannah_time_difference : 
  time_difference = 12.847 :=
by
  sorry

end hannah_time_difference_l806_806809


namespace X_on_altitude_BH_l806_806490

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806490


namespace closest_whole_number_to_ratio_l806_806712

theorem closest_whole_number_to_ratio : 
  let ratio := (3^3000 + 3^3003) / (3^3001 + 3^3002) in 
  abs (ratio - 2) < 1 :=
by 
  let ratio := (3^3000 + 3^3003) / (3^3001 + 3^3002)
  sorry

end closest_whole_number_to_ratio_l806_806712


namespace large_monkey_doll_cost_proof_l806_806282

noncomputable def large_monkey_doll_cost (L : ℝ) : Prop :=
  let small_doll_cost := L - 2 in
  let num_large_dolls := 300 / L in
  let num_small_dolls := 300 / small_doll_cost in
  num_small_dolls = num_large_dolls + 25 ∧ 0 < L

theorem large_monkey_doll_cost_proof : large_monkey_doll_cost 6 := 
  by
    -- First condition verifies the difference equation
    simp only [large_monkey_doll_cost]
    let small_doll_cost := 6 - 2
    let num_large_dolls := 300 / 6
    let num_small_dolls := 300 / small_doll_cost
    have h1 : small_doll_cost = 4 := by norm_num
    have h2 : num_large_dolls = 50 := by norm_num
    have h3 : num_small_dolls = 75 := by norm_num
    have h4 : 75 = 50 + 25 := by norm_num
    simp only
    constructor
    exact h4
    exact by norm_num

end large_monkey_doll_cost_proof_l806_806282


namespace intersection_point_of_lines_l806_806191

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end intersection_point_of_lines_l806_806191


namespace fruit_weights_assigned_l806_806961

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806961


namespace additional_tanks_needed_l806_806056

theorem additional_tanks_needed 
    (initial_tanks : ℕ) 
    (initial_capacity_per_tank : ℕ) 
    (total_fish_needed : ℕ) 
    (new_capacity_per_tank : ℕ)
    (h_t1 : initial_tanks = 3)
    (h_t2 : initial_capacity_per_tank = 15)
    (h_t3 : total_fish_needed = 75)
    (h_t4 : new_capacity_per_tank = 10) : 
    (total_fish_needed - initial_tanks * initial_capacity_per_tank) / new_capacity_per_tank = 3 := 
by {
    sorry
}

end additional_tanks_needed_l806_806056


namespace point_on_altitude_l806_806520

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806520


namespace collective_land_area_l806_806987

theorem collective_land_area 
  (C W : ℕ) 
  (h1 : 42 * C + 35 * W = 165200)
  (h2 : W = 3400)
  : C + W = 4500 :=
sorry

end collective_land_area_l806_806987


namespace bottle_capacity_two_liters_l806_806315

noncomputable def bottle_capacity (V : ℝ) : ℝ :=
  let initial_salt := 0.12 * V in
  let after_first_dilution := (0.12 * (V - 1)) / V in
  let after_second_dilution := ((0.12 * (V - 1) - (0.12 - (0.12 / V))) / V) in
  after_second_dilution = 0.03

theorem bottle_capacity_two_liters : ∃ V : ℝ, bottle_capacity V = 0.03 ∧ V = 2 :=
by
  sorry

end bottle_capacity_two_liters_l806_806315


namespace average_income_correct_l806_806677

def incomes : List ℕ := [250, 400, 750, 400, 500]

noncomputable def average : ℕ := (incomes.sum) / incomes.length

theorem average_income_correct : average = 460 :=
by 
  sorry

end average_income_correct_l806_806677


namespace simplify_expression_l806_806647

theorem simplify_expression :
  -3 - (+6) - (-5) + (-2) = -3 - 6 + 5 - 2 :=
  sorry

end simplify_expression_l806_806647


namespace point_X_on_altitude_BH_l806_806491

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806491


namespace tedra_harvested_2000kg_l806_806138

noncomputable def totalTomatoesHarvested : ℕ :=
  let wednesday : ℕ := 400
  let thursday : ℕ := wednesday / 2
  let total_wednesday_thursday := wednesday + thursday
  let remaining_friday : ℕ := 700
  let given_away_friday : ℕ := 700
  let friday := remaining_friday + given_away_friday
  total_wednesday_thursday + friday

theorem tedra_harvested_2000kg :
  totalTomatoesHarvested = 2000 := by
  sorry

end tedra_harvested_2000kg_l806_806138


namespace sonny_received_45_boxes_l806_806137

def cookies_received (cookies_given_brother : ℕ) (cookies_given_sister : ℕ) (cookies_given_cousin : ℕ) (cookies_left : ℕ) : ℕ :=
  cookies_given_brother + cookies_given_sister + cookies_given_cousin + cookies_left

theorem sonny_received_45_boxes :
  cookies_received 12 9 7 17 = 45 :=
by
  sorry

end sonny_received_45_boxes_l806_806137


namespace dvds_rented_l806_806098

def total_cost : ℝ := 4.80
def cost_per_dvd : ℝ := 1.20

theorem dvds_rented : total_cost / cost_per_dvd = 4 := 
by
  sorry

end dvds_rented_l806_806098


namespace derivative_at_4_l806_806320

variable (f : ℝ → ℝ)

theorem derivative_at_4 (h : ∀ Δx : ℝ, Δx ≠ 0 → 
    (∃ L : ℝ, tendsto (λ Δx, (f 4 + Δx - f 4 - Δx) / Δx) (nhds 0) (nhds L) ∧ L = -10)) : 
    deriv f 4 = -5 :=
by sorry

end derivative_at_4_l806_806320


namespace largest_y_triangle_inequality_l806_806748

theorem largest_y (y: ℝ) (h_eq: |y - 8| = 15) : y = 23 :=
sorry

theorem triangle_inequality (y: ℝ) (h_eq: |y - 8| = 15) : 
  (20 + 9 > y) ∧ (20 + y > 9) ∧ (9 + y > 20) :=
begin
  have hy : y = 23, from largest_y y h_eq,
  rw hy,
  simp,
  exact ⟨29 > 23, 43 > 9, 32 > 20⟩,
end

end largest_y_triangle_inequality_l806_806748


namespace binomial_sum_alternating_zero_l806_806083

open BigOperators

theorem binomial_sum_alternating_zero (n : ℕ) (h : 0 < n) : 
  ∑ k in Finset.range (n + 1), (-1)^k * Nat.choose n k = 0 := 
sorry

end binomial_sum_alternating_zero_l806_806083


namespace count_perfect_squares_between_50_and_300_l806_806427

theorem count_perfect_squares_between_50_and_300 : 
  ∃ n, number_of_perfect_squares 50 300 = n ∧ n = 10 := 
sorry

end count_perfect_squares_between_50_and_300_l806_806427


namespace fruit_weights_determined_l806_806953

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806953


namespace num_values_b_l806_806308

noncomputable def satisfies_beq : ℕ :=
  let b_values := {b | (2 * 0 + b = 0^2 + b^2)} in
  b_values.to_finset.card

theorem num_values_b : satisfies_beq = 2 := by
  sorry

end num_values_b_l806_806308


namespace both_solve_correctly_l806_806830

-- Define the probabilities of making an error for individuals A and B
variables (a b : ℝ)

-- Assuming a and b are probabilities, they must lie in the interval [0, 1]
axiom a_prob : 0 ≤ a ∧ a ≤ 1
axiom b_prob : 0 ≤ b ∧ b ≤ 1

-- Define the event that both individuals solve the problem correctly
theorem both_solve_correctly : (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by
  sorry

end both_solve_correctly_l806_806830


namespace find_a1_l806_806247

theorem find_a1 (a : ℕ → ℝ) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) (h_init : a 3 = 1 / 5) : a 1 = 1 := by
  sorry

end find_a1_l806_806247


namespace parabola_p_through_point_line_intersection_parabola_l806_806004

theorem parabola_p_through_point (p : ℝ) (h₀ : 0 < p) (h₁ : 16 = 8 * p) : p = 2 :=
by {
  -- proof here
  sorry
}

theorem line_intersection_parabola (p : ℝ) (h₀ : 0 < p) (h₁ : 16 = 8 * p)
  (midpoint_condition : ∃ A B : ℝ × ℝ, (A.fst ≠ B.fst) ∧ parabola_eq p A ∧ parabola_eq p B ∧
    (A.fst + B.fst) / 2 = 2 ∧ (A.snd + B.snd) / 2 = 1/3):
    18 * x - 3 * y - 35 = 0 :=
by {
  -- proof here
  sorry
}

def parabola_eq (p : ℝ) (point : ℝ × ℝ) : Prop :=
  point.snd^2 = 2 * p * point.fst

end parabola_p_through_point_line_intersection_parabola_l806_806004


namespace find_m_l806_806993

theorem find_m {x1 x2 m : ℝ} 
  (h_eqn : ∀ x, x^2 - (m+3)*x + (m+2) = 0) 
  (h_cond : x1 / (x1 + 1) + x2 / (x2 + 1) = 13 / 10) : 
  m = 2 := 
sorry

end find_m_l806_806993


namespace correct_result_l806_806275

-- Given condition
def mistaken_calculation (x : ℤ) : Prop :=
  x / 3 = 45

-- Proposition to prove the correct result
theorem correct_result (x : ℤ) (h : mistaken_calculation x) : 3 * x = 405 := by
  -- Here we can solve the proof later
  sorry

end correct_result_l806_806275


namespace count_arithmetic_sequences_l806_806317

-- Defining the set S
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Defining a function to check if three elements form an arithmetic sequence
def is_arithmetic_seq (a b c : ℕ) : Prop := (b - a = c - b)

-- Counting the number of 3-element subsets of S that form arithmetic sequences
theorem count_arithmetic_sequences : 
  (S.Subsets.cardicity.filter (λ t, t.card = 3 ∧ is_arithmetic_seq t.elem)).count = 8 :=
sorry

end count_arithmetic_sequences_l806_806317


namespace max_t_for_sequence_2_pow_n_range_a_for_sequence_n_squared_minus_a_over_n_l806_806771

def sequence_property_P (a : ℕ → ℤ) (t : ℤ) : Prop := 
  ∀ m n : ℕ, m ≠ n → (a m - a n) / (m - n) ≥ t

theorem max_t_for_sequence_2_pow_n :
  sequence_property_P (λ n, 2^n) 2 :=
sorry

theorem range_a_for_sequence_n_squared_minus_a_over_n (a : ℤ) :
  sequence_property_P (λ n, n^2 - a / n) 10 → a ≥ 36 :=
sorry

end max_t_for_sequence_2_pow_n_range_a_for_sequence_n_squared_minus_a_over_n_l806_806771


namespace least_distance_fly_crawled_l806_806691

noncomputable def leastDistance (baseRadius height startDist endDist : ℝ) : ℝ :=
  let C := 2 * Real.pi * baseRadius
  let slantHeight := Real.sqrt (baseRadius ^ 2 + height ^ 2)
  let theta := C / slantHeight
  let x1 := startDist * Real.cos 0
  let y1 := startDist * Real.sin 0
  let x2 := endDist * Real.cos (theta / 2)
  let y2 := endDist * Real.sin (theta / 2)
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem least_distance_fly_crawled (baseRadius height startDist endDist : ℝ) (h1 : baseRadius = 500) (h2 : height = 150 * Real.sqrt 7) (h3 : startDist = 150) (h4 : endDist = 300 * Real.sqrt 2) :
  leastDistance baseRadius height startDist endDist = 150 * Real.sqrt 13 := by
  sorry

end least_distance_fly_crawled_l806_806691


namespace count_multiples_of_10_not_20_l806_806432

theorem count_multiples_of_10_not_20 :
  {n // n < 500 ∧ n % 10 = 0 ∧ n % 20 ≠ 0}.card = 25 :=
by
  sorry

end count_multiples_of_10_not_20_l806_806432


namespace number_of_perfect_squares_l806_806418

theorem number_of_perfect_squares (a b : ℕ) (ha : 50 < a^2) (hb : b^2 < 300) :
  ∃ (n : ℕ), a ≤ n ∧ n ≤ b ∧ ∑ i in (finset.range (b - a + 1)).filter (λ n, 50 < n^2 ∧ n^2 < 300), 1 = 10 :=
sorry

end number_of_perfect_squares_l806_806418


namespace max_buses_constraint_satisfied_l806_806843

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l806_806843


namespace train_cross_time_l806_806675

theorem train_cross_time 
  (length_of_train : ℝ)
  (speed_of_train_kmh : ℝ)
  (converted_speed : speed_of_train_kmh * 1000 / 3600 = 15)
  (length_condition : length_of_train = 180)
  (speed_condition : speed_of_train_kmh = 54) :
  length_of_train / 15 = 12 :=
by
  -- Let's solve this problem
  simp only [length_condition, speed_condition, converted_speed]
  -- This reduces to 180 / 15 = 12
  rw [converted_speed, length_condition]
  -- Now, the proof is simple arithmetic
  norm_num -- simplifies/completes the arithmetic
  -- Therefore, the time is 12 seconds.
  exact rfl

end train_cross_time_l806_806675


namespace fruit_weights_l806_806956

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806956


namespace allocation_schemes_l806_806170

theorem allocation_schemes (students factories: ℕ) (has_factory_a: Prop) (A_must_have_students: has_factory_a): students = 3 → factories = 4 → has_factory_a → (∃ n: ℕ, n = 4^3 - 3^3 ∧ n = 37) :=
by try { sorry }

end allocation_schemes_l806_806170


namespace solve_complex_eq_l806_806982

noncomputable def complex_solutions (x : ℂ) : Prop :=
  x = 0 ∨ x = 1 ∨ 
  x = 1 + (1/2 + (complex.sqrt 3 / 2) * complex.I) ∨ 
  x = 1 + (1/2 - (complex.sqrt 3 / 2) * complex.I)

theorem solve_complex_eq (x : ℂ) : 
  ((x - 1)^4 + (x - 1) = 0) ↔ complex_solutions x :=
by
  sorry

end solve_complex_eq_l806_806982


namespace even_function_of_shift_sine_l806_806321

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := (x - 6)^2 * Real.sin (ω * x)

theorem even_function_of_shift_sine :
  ∃ ω : ℝ, (∀ x : ℝ, f x ω = f (-x) ω) → ω = π / 4 :=
by
  sorry

end even_function_of_shift_sine_l806_806321


namespace ground_resistance_calculation_l806_806241

/-- Given the conditions of the pile driver problem:
 - mass m = 300 kg
 - falls from a height h = 3 m
 - after 30 blows
 - the pile is driven to a depth s = 35 cm
Prove that the ground resistance in kgf is 77182.67 kgf.
Assumptions:
- g = 9.81 m/s^2 (acceleration due to gravity)
- s must be converted to meters.
--/
theorem ground_resistance_calculation  
  (m : ℝ) (h : ℝ) (n : ℕ) (s : ℝ)
  (g : ℝ := 9.81)
  (m_pos : m = 300)
  (h_pos : h = 3)
  (n_pos : n = 30)
  (s_pos : s = 0.35) :
  let E_pot := m * g * h,
      E_total := n * E_pot,
      F_resistance := E_total / s,
      F_resistance_kgf := F_resistance / g in
  F_resistance_kgf = 77182.67 :=
by
  sorry

end ground_resistance_calculation_l806_806241


namespace number_of_smaller_semicircles_l806_806865

-- Definitions based on conditions
def large_semicircle_radius (D : ℝ) (N : ℝ) (r : ℝ) := r = D / (2 * N)
def smaller_semicircle_area (r : ℝ) := (π * r^2) / 2
def total_smaller_semicircle_area (N : ℝ) (r : ℝ) := N * (π * r^2) / 2
def large_semicircle_area (N : ℝ) (r : ℝ) := (π * (N * r)^2) / 2
def area_excluding_smaller_semicircle (N : ℝ) (r : ℝ) := (π * r^2 / 2) * (N^2 - N)
def area_ratio_condition (N : ℝ) (r : ℝ) : Prop :=
  let A := total_smaller_semicircle_area N r in
  let B := area_excluding_smaller_semicircle N r in
  A / B = 2 / 25

-- Theorem statement
theorem number_of_smaller_semicircles (D : ℝ) (N : ℝ) (r : ℝ) (h : large_semicircle_radius D N r)
    (h_ratio : area_ratio_condition N r) : N = 14 :=
by sorry

end number_of_smaller_semicircles_l806_806865


namespace smallest_k_l806_806561

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l806_806561


namespace num_four_digit_with_5_or_7_l806_806395

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l806_806395


namespace circle_center_l806_806295

theorem circle_center (x y : ℝ) (h : x^2 - 4 * x + y^2 - 6 * y - 12 = 0) : (x, y) = (2, 3) :=
sorry

end circle_center_l806_806295


namespace sum_of_roots_in_range_l806_806532

theorem sum_of_roots_in_range :
  let S := (finset.filter (λ x : ℝ, x > 0)
               (finset.univ.filter (λ x : ℝ, (x ^ (2 * real.sqrt 2)) = ((real.sqrt 2) ^ (2 ^ x))))
             ).sum in
  2 ≤ S ∧ S < 6 :=
by
  sorry

end sum_of_roots_in_range_l806_806532


namespace least_number_of_apples_l806_806544

theorem least_number_of_apples (b : ℕ) : (b % 3 = 2) → (b % 4 = 3) → (b % 5 = 1) → b = 11 :=
by
  intros h1 h2 h3
  sorry

end least_number_of_apples_l806_806544


namespace digits_with_five_or_seven_is_5416_l806_806405

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l806_806405


namespace sufficient_but_not_necessary_l806_806768

-- Define what it means for a line to be perpendicular to a plane
def line_perpendicular_to_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- Define what it means for a line to be perpendicular to countless lines in a plane
def line_perpendicular_to_countless_lines_in_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- The formal statement
theorem sufficient_but_not_necessary (l : Type) (alpha : Type) :
  (line_perpendicular_to_plane l alpha) → (line_perpendicular_to_countless_lines_in_plane l alpha) ∧ 
  ¬ ((line_perpendicular_to_countless_lines_in_plane l alpha) → (line_perpendicular_to_plane l alpha)) :=
by sorry

end sufficient_but_not_necessary_l806_806768


namespace point_X_on_altitude_BH_l806_806494

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806494


namespace four_digit_integers_with_5_or_7_l806_806381

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l806_806381


namespace max_handshakes_l806_806673

theorem max_handshakes {n : ℕ} (h : n = 30) : (n * (n - 1)) / 2 = 435 :=
by
  rw h
  norm_num
  sorry

end max_handshakes_l806_806673


namespace sequence_formulas_and_sum_l806_806779

theorem sequence_formulas_and_sum (a b c : ℕ → ℕ) (h1 : a 1 = 1) (h2 : b 1 = 1)
  (h3 : b 2 + b 3 = 2 * a 3) 
  (h4 : a 5 - 3 * b 2 = 7)
  (ha : ∀ n, a n = 2^(n-1))
  (hb : ∀ n, b n = 2*n - 1) :
  (∀ n, c n = a n * b n) ∧ 
  (∀ n, (∑ k in Finset.range n.succ, c k) = (2*n - 3) * 2^n + 3) :=
by {
  sorry
}

end sequence_formulas_and_sum_l806_806779


namespace decreasing_line_implies_m_half_l806_806994

theorem decreasing_line_implies_m_half (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * m - 1) * x₁ + b > (2 * m - 1) * x₂ + b) → m < 1 / 2 :=
by
  intro h
  sorry

end decreasing_line_implies_m_half_l806_806994


namespace arithmetic_sequence_solution_l806_806483

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + n * d
noncomputable def sum_first_n_terms (a₁ d n : ℕ) : ℕ :=
  n * a₁ + (n * (n - 1) * d) / 2

-- Main statement
theorem arithmetic_sequence_solution
  (a₁ d : ℕ) (S : ℕ → ℕ) (n m : ℕ)
  (h₀ : a₁ = 1)
  (h₁ : d = 2)
  (h₂ : S (n + 2) - S n = 36) :
  n = 8 := 
by
  -- Definitions based on arithmetic sequence
  let aₙ := λ (n : ℕ), a₁ + (n - 1) * d
  let S := λ (n : ℕ), (n * a₁) + ((n * (n - 1)) * d) / 2
  
  -- Simplify the given condition
  have h₃ : aₙ (n + 1) + aₙ (n + 2) = 36,
    { rw [←h₂], simp [aₙ, h₀, h₁] }
  
  -- Solving for n
  sorry

end arithmetic_sequence_solution_l806_806483


namespace union_complement_l806_806900

open Set

variable (U A B : Set ℕ)
variable (u_spec : U = {1, 2, 3, 4, 5})
variable (a_spec : A = {1, 2, 3})
variable (b_spec : B = {2, 4})

theorem union_complement (U A B : Set ℕ)
  (u_spec : U = {1, 2, 3, 4, 5})
  (a_spec : A = {1, 2, 3})
  (b_spec : B = {2, 4}) :
  A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end union_complement_l806_806900


namespace rhombus_incircle_parallels_l806_806265

theorem rhombus_incircle_parallels
  (ABCD: rhombus)
  (O: point)
  (E F G H: point)
  (M N P Q: point)
  (h_incirc: incircle ABCD O)
  (h_tangent_AB_E: is_tangent_point O E AB)
  (h_tangent_BC_F: is_tangent_point O F BC)
  (h_tangent_CD_G: is_tangent_point O G CD)
  (h_tangent_DA_H: is_tangent_point O H DA)
  (h_tangent_EF_AB_M: is_tangent_on_arc O (arc E F) AB M)
  (h_tangent_EF_BC_N: is_tangent_on_arc O (arc E F) BC N)
  (h_tangent_GH_CD_P: is_tangent_on_arc O (arc G H) CD P)
  (h_tangent_GH_DA_Q: is_tangent_on_arc O (arc G H) DA Q) :
  parallel MQ NP :=
sorry

end rhombus_incircle_parallels_l806_806265


namespace find_m_l806_806787

-- Define the points P and Q
structure Point where
  x : ℝ
  y : ℝ

def P (m : ℝ) := Point.mk (-2) m
def Q (m : ℝ) := Point.mk m 4

-- Define slope function
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Suppose the slope is 1, prove m = 1
theorem find_m (m : ℝ) (h : slope (P m) (Q m) = 1) : m = 1 :=
by 
  sorry

end find_m_l806_806787


namespace lucy_should_buy_correct_number_of_fish_l806_806915

-- Define the initial state and requirements in the aquarium
def initial_total_fish : ℕ := 212
def percent_neon_tetras : ℝ := 0.40
def percent_guppies : ℝ := 0.30
def percent_angelfish : ℝ := 0.30
def additional_fish : ℕ := 68

-- Translate conditions to Lean definitions
def initial_neon_tetras := (percent_neon_tetras * initial_total_fish).round
def initial_guppies := (percent_guppies * initial_total_fish).round
def initial_angelfish := (percent_angelfish * initial_total_fish).round

def required_new_neon_tetras := (percent_neon_tetras * additional_fish).round + 1 -- Adjusted for rounding
def required_new_guppies := (percent_guppies * additional_fish).round
def required_new_angelfish := (percent_angelfish * additional_fish).round

-- The theorem statement to prove
theorem lucy_should_buy_correct_number_of_fish :
  required_new_neon_tetras = 28 ∧
  required_new_guppies = 20 ∧
  required_new_angelfish = 20 :=
by sorry

end lucy_should_buy_correct_number_of_fish_l806_806915


namespace complement_union_example_l806_806667

open Set

universe u

variable (U : Set ℕ) (A B : Set ℕ)

def U_def : Set ℕ := {0, 1, 2, 3, 4}
def A_def : Set ℕ := {0, 1, 2}
def B_def : Set ℕ := {2, 3}

theorem complement_union_example :
  (U \ A) ∪ B = {2, 3, 4} := 
by
  -- Proving the theorem considering
  -- complement and union operations on sets
  sorry

end complement_union_example_l806_806667


namespace fruit_weights_l806_806937

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806937


namespace min_k_triangle_l806_806316

theorem min_k_triangle (k : ℕ) :
  (∀ (S : set ℕ), (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → a + b > c) → S ⊆ { n | n ≤ 2004 } → S.card ≥ k) ↔ k = 17 :=
by sorry

end min_k_triangle_l806_806316


namespace min_value_abs_diff_l806_806150

-- Definitions from the problem
def f (x : ℝ) := 2 * Real.sin x

-- Theorem to prove minimum value of |x1 - x2|
theorem min_value_abs_diff {x1 x2 : ℝ} (h1 : f x1 ≤ f x1) (h2 : f x2 ≥ f x2) :
  abs (x1 - x2) = Real.pi :=
sorry

end min_value_abs_diff_l806_806150


namespace min_area_PRQSQ_l806_806331

-- Define the setup
variables {m : ℝ} (h : m > 0)

-- Line passing through A(1,1) with slope -m
def line := {p : ℝ × ℝ // ∃ x y, p = (1 + x/m, 1 - m*x)}

-- Intersection points P and Q
def P := (1 + 1/m, 0)
def Q := (0, 1 + m)

-- Perpendicular feet R and S from P and Q on 2x + y = 0
def line₂ := {p : ℝ × ℝ // p.1 + 2*p.2 = 0}
def R := orthogonal_projection line₂ (1 + 1/m, 0)
def S := orthogonal_projection line₂ (0, 1 + m)

-- Lengths
def PR := distance P R
def QS := distance Q S
def RS := distance R S

-- Minimum area of quadrilateral PRSQ
theorem min_area_PRQSQ : 
  let area := (1 / 5) * (m + (1 / m) + (9 / 4)) ^ 2 - (1 / 80) in
  ∃ (area_min : ℝ), area ≥ 3.6 :=
sorry

end min_area_PRQSQ_l806_806331


namespace triangle_ab2_over_c2_eq_3_l806_806471

variable {α : Type*} [LinearOrderedField α]

def tan_of_sin_cos (sin_α cos_α : α) : α :=
  sin_α / cos_α

noncomputable def triangle_condition 
  (A B C : α)
  (a b c : α)
  (hA : 0 < A)
  (hB : 0 < B)
  (hC : 0 < C)
  (cosA sinA tanA : α)
  (cosB sinB tanB : α)
  (cosC sinC tanC : α)
  (hABC : A + B + C = Real.pi)
  (htanA : tanA = tan_of_sin_cos sinA cosA)
  (htanB : tanB = tan_of_sin_cos sinB cosB)
  (htanC : tanC = tan_of_sin_cos sinC cosC)
  (cos_rule : (cosC = (a^2 + b^2 - c^2) / (2 * a * b)))
  (triangle_eq : tanA * tanB = tanA * tanC + tanC * tanB) :
  a^2 + b^2 = 3 * c^2 :=
sorry

theorem triangle_ab2_over_c2_eq_3 
  (A B C : α) 
  (a b c : α) 
  (hA : 0 < A) 
  (hB : 0 < B) 
  (hC : 0 < C) 
  (cosA sinA tanA : α) 
  (cosB sinB tanB : α) 
  (cosC sinC tanC : α) 
  (hABC : A + B + C = Real.pi) 
  (htanA : tanA = tan_of_sin_cos sinA cosA)
  (htanB : tanB = tan_of_sin_cos sinB cosB)
  (htanC : tanC = tan_of_sin_cos sinC cosC)
  (cos_rule : (cosC = (a^2 + b^2 - c^2) / (2 * a * b)))
  (triangle_eq : tanA * tanB = tanA * tanC + tanC * tanB) : 
  (a^2 + b^2) / c^2 = 3 :=
begin
  have h_triv := triangle_condition A B C a b c hA hB hC cosA sinA tanA cosB sinB tanB cosC sinC tanC hABC htanA htanB htanC cos_rule triangle_eq,
  rw [h_triv],
  ring,
end

end triangle_ab2_over_c2_eq_3_l806_806471


namespace problem1_l806_806270

theorem problem1 :
  (-1 : ℤ)^2024 - (-1 : ℤ)^2023 = 2 := by
  sorry

end problem1_l806_806270


namespace tan_half_alpha_third_quadrant_l806_806826

theorem tan_half_alpha_third_quadrant
  (alpha beta : ℝ)
  (h1 : sin(alpha + beta) * cos(beta) - sin(beta) * cos(alpha + beta) = -12/13)
  (h2 : ∃ k : ℤ, k * π - π / 2 < alpha ∧ alpha < k * π) :
  tan (alpha / 2) = -3 / 2 :=
by
  sorry

end tan_half_alpha_third_quadrant_l806_806826


namespace no_tiling_possible_l806_806886

theorem no_tiling_possible : 
  ¬ ∃ (f : (Σ i j : Fin 8, (i, j) ≠ (0, 0) ∧ (i, j) ≠ (7, 7)) → Σ i j : Fin 8, i = j), 
      ∀ x y, (x ≠ y → (f x).1 ≠ (f x).1) :=
by
  sorry

end no_tiling_possible_l806_806886


namespace minimal_road_length_l806_806036

theorem minimal_road_length (N : ℕ) (hN : N ≥ 1) : 
  let num_cities := N^2,
      mst_edges := num_cities - 1,
      edge_length := 10
  in (10 * (num_cities - 1)) = 10 * (N^2 - 1) :=
by
  sorry

end minimal_road_length_l806_806036


namespace cos_product_triangle_l806_806472

theorem cos_product_triangle (A B C : ℝ) (h : A + B + C = π) (hA : A > 0) (hB : B > 0) (hC : C > 0) : 
  Real.cos A * Real.cos B * Real.cos C ≤ 1 / 8 := 
sorry

end cos_product_triangle_l806_806472


namespace price_per_pie_l806_806709

-- Define the relevant variables and conditions
def cost_pumpkin_pie : ℕ := 3
def num_pumpkin_pies : ℕ := 10
def cost_cherry_pie : ℕ := 5
def num_cherry_pies : ℕ := 12
def desired_profit : ℕ := 20

-- Total production and profit calculation
def total_cost : ℕ := (cost_pumpkin_pie * num_pumpkin_pies) + (cost_cherry_pie * num_cherry_pies)
def total_earnings_needed : ℕ := total_cost + desired_profit
def total_pies : ℕ := num_pumpkin_pies + num_cherry_pies

-- Proposition to prove that the price per pie should be $5
theorem price_per_pie : (total_earnings_needed / total_pies) = 5 := by
  sorry

end price_per_pie_l806_806709


namespace max_m_value_range_a_for_one_root_l806_806095

noncomputable def f (x a : ℝ) : ℝ := x^3 - (9/2) * x^2 + 6 * x - a

theorem max_m_value (a : ℝ) : ∀ x, deriv (f x a) x ≥ -3/4 := sorry

theorem range_a_for_one_root (a : ℝ) : (∀ x, f x a = 0 → (f (1:ℝ) a < 0) ∨ (f (2:ℝ) a > 0)) ↔ (a < 2 ∨ a > 5/2) := sorry

end max_m_value_range_a_for_one_root_l806_806095


namespace general_form_of_line_passing_points_l806_806152

theorem general_form_of_line_passing_points (A B : ℝ×ℝ) 
  (hA : A = (1, 1)) (hB : B = (-2, 4)) : 
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -2 ∧ 
  ∀ x y : ℝ, (x, y) ∈ line_eq A B ↔ a * x + b * y + c = 0 := 
sorry

def line_eq (A B : ℝ×ℝ) : set (ℝ×ℝ) := 
  { P | ∃ t : ℝ, P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) }


end general_form_of_line_passing_points_l806_806152


namespace seating_arrangements_possible_l806_806313

-- Define the problem statement
theorem seating_arrangements_possible
  (n : ℕ)
  (siblings : fin n → (ℕ × ℕ))
  (rows : list (list ℕ))
  (h_siblings_len : n = 4)
  (h_rows_len : rows.length = 2)
  (h_seats_per_row : ∀ row, row.length = 4)
  (h_no_next_to_each_other : ∀ row i, i < row.length - 1 → 
    (∀ k, k < n → (siblings k).1 ∈ row → ¬((siblings k).2) ∈ (list.tail (list.drop i row))))
  (h_no_behind_each_other : ∀ i j, i < rows.head.length → j < rows.tail.head.length →
    (∀ k, k < n → (siblings k).1 = rows.head.nth i → ¬((siblings k).2) = rows.tail.head.nth j → i ≠ j))
  :
  ∃ arrangements : ℕ, arrangements = 3456 :=
begin
  use (fact 4 * (derangements 4) * (2 ^ 4)),
  sorry
end

end seating_arrangements_possible_l806_806313


namespace least_four_digit_palindrome_divisible_by_5_l806_806115

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString in str = str.reverse

def is_divisible_by_5 (n: ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem least_four_digit_palindrome_divisible_by_5 : 
  ∃ n, is_palindrome n ∧ is_four_digit n ∧ is_divisible_by_5 n ∧ ∀ m, is_palindrome m ∧ is_four_digit m ∧ is_divisible_by_5 m → n ≤ m := 
sorry

end least_four_digit_palindrome_divisible_by_5_l806_806115


namespace least_palindrome_div_by_5_l806_806112

/-- A palindrome is a number that reads the same backward and forward. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

/-- The least possible positive four-digit palindrome that is divisible by 5 is 5005. -/
theorem least_palindrome_div_by_5 : ∃ n : ℕ, is_palindrome n ∧ (1000 ≤ n ∧ n < 10000) ∧ n % 5 = 0 ∧ ∀ m : ℕ, is_palindrome m ∧ (1000 ≤ m ∧ m < 10000) ∧ m % 5 = 0 → n ≤ m :=
begin
  use 5005,
  split,
  { unfold is_palindrome,
    dsimp,
    norm_num },
  split,
  { split, norm_num, norm_num },
  split,
  { norm_num },
  { intros m h,
    cases h with hm1 hm2,
    cases hm2 with hm3 hm4,
    cases hm3 with hm5 hm6,
    have h5 : m % 5 = 0 := hm4,
    norm_cast at h5,
    by_cases h : m = 5005,
    { rw h, exact le_refl _ },
    { have h7 : 5005 <= m, sorry } },
end

end least_palindrome_div_by_5_l806_806112


namespace dishonest_dealer_percentage_l806_806232

theorem dishonest_dealer_percentage :
  let standard_weight := 16
  let actual_weight := 14.8
  let difference := standard_weight - actual_weight
  (difference / standard_weight) * 100 = 7.5 :=
by
  let standard_weight := 16
  let actual_weight := 14.8
  let difference := standard_weight - actual_weight
  have h := (difference / standard_weight) * 100
  exact h

end dishonest_dealer_percentage_l806_806232


namespace common_tangent_of_A_and_B_l806_806273

-- Defining the geometrical configuration as given in the problem
variables {A B O : Type*} [circle A] [circle B] [circle O]
variables {C D M N P E F : Type*} 

-- Conditions
variable (h1 : (C ∈ A ∧ C ∈ B))
variable (h2 : (D ∈ A ∧ D ∈ B))
variable (h3 : (M ∈ A ∧ M ∈ O))
variable (h4 : (N ∈ B ∧ N ∈ O))
variable (h5 : (P ∉ C ∧ P ∈ O ∧ P ∈ ray CD))
variable (h6 : (E ∈ A ∧ E ∈ PM))
variable (h7 : (F ∈ B ∧ F ∈ PN))

-- Goal
theorem common_tangent_of_A_and_B : 
  is_common_tangent_of A B E F :=
sorry

end common_tangent_of_A_and_B_l806_806273


namespace integer_solutions_conditions_even_l806_806757

theorem integer_solutions_conditions_even (n : ℕ) (x : ℕ → ℤ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 
    x i ^ 2 + x ((i % n) + 1) ^ 2 + 50 = 16 * x i + 12 * x ((i % n) + 1) ) → 
  n % 2 = 0 :=
by 
sorry

end integer_solutions_conditions_even_l806_806757


namespace locus_of_TangentCircumcircles_is_circle_l806_806896

noncomputable def areCircumcirclesTangent (A B C D X : Point) : Prop :=
  -- Definition that checks if circumcircles of triangles XAB and XCD are tangent at X
  sorry

theorem locus_of_TangentCircumcircles_is_circle (A B C D : Point) (on_circle : ∀ P ∈ {A, B, C, D}, CircleContains P) :
  ∃ P : Point, ∃ r : Real,
    (∀ X : Point, areCircumcirclesTangent A B C D X ↔ Distance X P = r) :=
    sorry

end locus_of_TangentCircumcircles_is_circle_l806_806896


namespace rectangular_solid_surface_area_l806_806285

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem rectangular_solid_surface_area (l w h : ℕ) (hl : is_prime l) (hw : is_prime w) (hh : is_prime h) (volume_eq_437 : l * w * h = 437) :
  2 * (l * w + w * h + h * l) = 958 :=
sorry

end rectangular_solid_surface_area_l806_806285


namespace trig_identity_l806_806213

theorem trig_identity :
  2 * Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4)^2 + Real.cos (Real.pi / 3) = 1 :=
by
  sorry

end trig_identity_l806_806213


namespace sum_of_divisors_even_l806_806531

/-
  Let N(n) denote the number of distinct divisors of a positive integer n.
  For example, 24 has divisors 1, 2, 3, 4, 6, 8, 12, 24, so N(24) = 8.
  Determine whether the sum N(1) + N(2) + ... + N(1989) is odd or even.
 -/

noncomputable def N (n : ℕ) : ℕ :=
  -- Function to count the number of distinct divisors of n
  (Finset.range (n + 1)).filter (λ d => d > 0 ∧ n % d = 0).card

theorem sum_of_divisors_even :
  (Finset.range 1990).sum N % 2 = 0 :=
by
  sorry

end sum_of_divisors_even_l806_806531


namespace plates_difference_l806_806976

noncomputable def num_pots_angela : ℕ := 20
noncomputable def num_plates_angela (P : ℕ) := P
noncomputable def num_cutlery_angela (P : ℕ) := P / 2
noncomputable def num_pots_sharon : ℕ := 10
noncomputable def num_plates_sharon (P : ℕ) := 3 * P - 20
noncomputable def num_cutlery_sharon (P : ℕ) := P
noncomputable def total_kitchen_supplies_sharon (P : ℕ) := 
  num_pots_sharon + num_plates_sharon P + num_cutlery_sharon P

theorem plates_difference (P : ℕ) 
  (hP: num_plates_angela P > 3 * num_pots_angela) 
  (h_supplies: total_kitchen_supplies_sharon P = 254) :
  P - 3 * num_pots_angela = 6 := 
sorry

end plates_difference_l806_806976


namespace fruit_weights_correct_l806_806943

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806943


namespace upper_limit_of_x_l806_806448

theorem upper_limit_of_x 
  {x : ℤ} 
  (h1 : 0 < x) 
  (h2 : x < 15) 
  (h3 : -1 < x) 
  (h4 : x < 5) 
  (h5 : 0 < x) 
  (h6 : x < 3) 
  (h7 : x + 2 < 4) 
  (h8 : x = 1) : 
  0 < x ∧ x < 2 := 
by 
  sorry

end upper_limit_of_x_l806_806448


namespace find_f_2013_l806_806538

-- Given conditions in Lean definitions
variable (f : ℝ → ℝ)
variable hOdd : ∀ x, f (-x) = -f (x)
variable hFunctional : ∀ x, f (x + 3) = -f (1 - x)
variable hInitial : f 3 = 2

-- Proof goal in Lean
theorem find_f_2013 : f 2013 = -2 :=
by
  sorry

end find_f_2013_l806_806538


namespace car_speed_second_hour_l806_806174

theorem car_speed_second_hour
  (S : ℕ)
  (first_hour_speed : ℕ := 98)
  (avg_speed : ℕ := 79)
  (total_time : ℕ := 2)
  (h_avg_speed : avg_speed = (first_hour_speed + S) / total_time) :
  S = 60 :=
by
  -- Proof steps omitted
  sorry

end car_speed_second_hour_l806_806174


namespace aquarium_original_price_l806_806708

theorem aquarium_original_price (P : ℝ) :
  (0.5 * P + 0.05 * (0.5 * P) = 63) → P = 120 :=
by
  intro h,
  have h1 : 0.5 * P + 0.025 * P = 63, 
    -- The proof step simplifying the terms can be skipped with a direct sorry since it's explanatory.
    sorry,
  have h2 : 0.525 * P = 63 := sorry,
  have h3 : P = 63 / 0.525 := sorry,
  exact h3.symm sorry

end aquarium_original_price_l806_806708


namespace probability_of_two_red_balls_l806_806583

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_two_red_balls :
  let X := λ (selected : Finset ℕ), (∃ (subset : Finset ℕ), subset ⊆ (Finset.range 4) ∧ subset.card = 2) in
  P(X = 2) = 10 / 21 :=
by
  sorry

end probability_of_two_red_balls_l806_806583


namespace infinitely_many_non_prime_sums_l806_806664

def not_prime (n : ℕ) : Prop :=
  ¬ isPrime n

theorem infinitely_many_non_prime_sums : ∃ᶠ a in at_top, ∀ n : ℕ, not_prime (n^4 + a) :=
by
  sorry

end infinitely_many_non_prime_sums_l806_806664


namespace quad_area_AMBN_correct_l806_806003

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 3) ^ 2 = 4

-- Define the area of quadrilateral problem
def area_of_quadrilateral (area : ℝ) : Prop := area = 4 * Real.sqrt 2

-- The main theorem statement, proving the area of quadrilateral AMBN
theorem quad_area_AMBN_correct :
  ∃ A B : ℝ × ℝ,
    line_eq A.1 A.2 ∧ circle_eq A.1 A.2 ∧
    line_eq B.1 B.2 ∧ circle_eq B.1 B.2 ∧
    ∃ M N : ℝ × ℝ,
      midpoint A B M ∧ 
      ((N.1 = M.1 + 2 * 1 / Real.sqrt 2) ∧ (N.2 = M.2 + 2 * (-1 / Real.sqrt 2))) ∧ 
      area_of_quadrilateral (quad_area A B M N) := sorry

end quad_area_AMBN_correct_l806_806003


namespace equivalent_expression_l806_806690

-- Let a, b, c, d, e be real numbers
variables (a b c d e : ℝ)

-- Condition given in the problem
def condition : Prop := 81 * a - 27 * b + 9 * c - 3 * d + e = -5

-- Objective: Prove that 8 * a - 4 * b + 2 * c - d + e = -5 given the condition
theorem equivalent_expression (h : condition a b c d e) : 8 * a - 4 * b + 2 * c - d + e = -5 :=
sorry

end equivalent_expression_l806_806690


namespace poly_sum_correct_l806_806088

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end poly_sum_correct_l806_806088


namespace total_revenue_correct_l806_806099

def items : Type := ℕ × ℝ

def magazines : items := (425, 2.50)
def newspapers : items := (275, 1.50)
def books : items := (150, 5.00)
def pamphlets : items := (75, 0.50)

def revenue (item : items) : ℝ := item.1 * item.2

def total_revenue : ℝ :=
  revenue magazines +
  revenue newspapers +
  revenue books +
  revenue pamphlets

theorem total_revenue_correct : total_revenue = 2262.50 := by
  sorry

end total_revenue_correct_l806_806099


namespace cuberoot_sum_l806_806739

theorem cuberoot_sum : 
  real.cbrt (27 - 18 * real.sqrt 3) + real.cbrt (27 + 18 * real.sqrt 3) = 6 :=
by
  sorry

end cuberoot_sum_l806_806739


namespace maximum_number_of_buses_l806_806845

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l806_806845


namespace evaluate_nested_function_l806_806444

def f (x : ℝ) : ℝ :=
  if x > 0 then x - 2 else Real.exp (x + 1)

theorem evaluate_nested_function : f (f (-1)) = -1 := by
  sorry

end evaluate_nested_function_l806_806444


namespace find_room_dimension_l806_806595

def room_cost (length width height : ℕ) (door_w door_h : ℕ) (window_w window_h : ℕ) (num_windows : ℕ) (cost_per_sqft total_cost : ℕ) := 
  let wall_area := 2 * (length * height) + 2 * (width * height)
  let door_area := door_w * door_h
  let window_area := num_windows * (window_w * window_h)
  let net_area := wall_area - door_area - window_area
  let total_cost_computed := net_area * cost_per_sqft
  total_cost_computed = total_cost

theorem find_room_dimension : room_cost 25 x 12 6 3 4 3 3 7 6342 → x = 15 :=
begin
  sorry
end

end find_room_dimension_l806_806595


namespace binomial_expansion_constant_term_l806_806144

theorem binomial_expansion_constant_term :
  let x : ℂ := 1 -- assuming x is a nonzero complex number
  let constant_term (x) := ∑ k in finset.range 7, (nat.choose 6 k) * (x ^ (6 - k)) * ((2 / complex.sqrt x) ^ k) in 
  constant_term x = 240 :=
by
  sorry

end binomial_expansion_constant_term_l806_806144


namespace correct_function_description_l806_806642

-- Definitions for each condition and corresponding descriptions
def option_A : Prop := "Indicates the start and end of the algorithm"
def option_B : Prop := "Indicates the input and output information of the algorithm"
def option_C : Prop := "Assignment calculation"
def option_D : Prop := "Connects program boxes according to the order of the algorithm"

-- Assuming the program box in question is input_output
def function_box : Prop := "Indicates the input and output information of the algorithm"

-- Theorem stating that the correct description of the function represented by the program box " " is option_B
theorem correct_function_description : function_box = option_B := 
by
  sorry

end correct_function_description_l806_806642


namespace find_hyperbola_equation_l806_806590

noncomputable def hyperbola_equation (origin : ℝ × ℝ)
  (foci : ℝ × ℝ)
  (slope : ℝ)
  (distance_PQ : ℝ)
  (ortho_pq : (ℝ × ℝ) → (ℝ × ℝ) → Prop)
  : ℝ × ℝ → Prop := 
λ (x y : ℝ), (x^2) - (y^2 / 3) = 1

theorem find_hyperbola_equation 
  (center : ℝ × ℝ := (0,0)) 
  (focus_pos_x : ℝ)
  (foci_location : ∀ x : ℝ,  ( x , 0 ))
  (line_slope : ℝ := Real.sqrt (3/5)) 
  (line_through_focus : ℝ → Prop) 
  (P Q : ℝ × ℝ)
  (orthogonal : (P.1 * Q.1 + 3/5 * (P.1 - focus_pos_x) * (Q.1 - focus_pos_x) = 0))
  (PQ_distance: ∀ PQ : ℝ × ℝ, Real.sqrt (8/5) * abs (P.1 - Q.1) = 4) :

  hyperbola_equation (0,0) (2 * focus_pos_x, 0) _ _ orthogonal = 
  λ p, p.1^2 - (p.2^2 / 3) = 1 :=
by sorry

end find_hyperbola_equation_l806_806590


namespace who_has_largest_final_number_l806_806737

theorem who_has_largest_final_number :
  let ellen_final := (12 - 2) * 3 + 4,
      marco_final := (15 * 3 - 3) + 5,
      lucia_final := (13 - 3 + 5) * 3 in
  marco_final > ellen_final ∧ marco_final > lucia_final :=
by
  let ellen_final := (12 - 2) * 3 + 4
  let marco_final := (15 * 3 - 3) + 5
  let lucia_final := (13 - 3 + 5) * 3
  show marco_final > ellen_final ∧ marco_final > lucia_final
  sorry

end who_has_largest_final_number_l806_806737


namespace gcd_of_polynomials_l806_806758

open Int

theorem gcd_of_polynomials (a : ℕ) : gcd (a^4 + 3*a^2 + 1) (a^3 + 2*a) = 1 :=
by
  sorry

end gcd_of_polynomials_l806_806758


namespace EN_contains_X_l806_806453

variables (A B C D E M N P Q R X : Type)
variables (Mid : A → A → A → Prop)
variables (AP : A → A → Prop)
variables (BQ : A → A → Prop)
variables (CR : A → A → Prop)
variables (DM : A → A → Prop)
variables (EN : A → A → Prop)
variables (convex_pentagon : Prop)

-- Conditions
axiom convex_pentagon_ABCDE : convex_pentagon A B C D E
axiom midpoint_M : Mid A B M
axiom midpoint_N : Mid B C N
axiom midpoint_P : Mid C D P
axiom midpoint_Q : Mid D E Q
axiom midpoint_R : Mid E A R

axiom concur : ∃ X, AP A P X ∧ BQ B Q X ∧ CR C R X ∧ DM D M X

-- Goal
theorem EN_contains_X : convex_pentagon A B C D E →
                        Mid A B M → Mid B C N → Mid C D P → Mid D E Q → Mid E A R →
                        (∃ X, AP A P X ∧ BQ B Q X ∧ CR C R X ∧ DM D M X) →
                        ∃ X, EN E N X :=
begin
  intros,
  rcases concur with ⟨X, h1, h2, h3, h4⟩,
  use X,
  sorry -- Proof goes here
end

end EN_contains_X_l806_806453


namespace triangle_perimeter_l806_806177

theorem triangle_perimeter (L R B : ℕ) (hL : L = 12) (hR : R = L + 2) (hB : B = 24) : L + R + B = 50 :=
by
  -- proof steps go here
  sorry

end triangle_perimeter_l806_806177


namespace find_constants_find_c_range_l806_806094

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := 2*x^3 + 3*a*x^2 + 3*b*x + 8*c
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 6*x^2 + 6*a*x + 3*b

theorem find_constants (a b c : ℝ) :
  f'(1) a b = 0 ∧ f'(2) a b = 0 →
  a = -3 ∧ b = 4 := 
by
  intros
  sorry

theorem find_c_range (a b c : ℝ) :
  f' 1 (-3) 4 = 0 ∧ f' 2 (-3) 4 = 0 →
  (∀ x ∈ set.Icc 0 3, f x (-3) 4 c < c^2) ↔ 
  c > 9 ∨ c < -1 :=
by
  intros
  sorry

end find_constants_find_c_range_l806_806094


namespace part1_part2_l806_806539

noncomputable theory

-- Definitions of sets A, B, and C
def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {1, 2, a}
def B (a : ℝ) : Set ℝ := {2, a ^ 2}
def C : Set ℝ := {x : ℝ | 3^(2 * x - 1) > 27}

-- Complement of set C in U
def complementC : Set ℝ := {x : ℝ | x ≤ 2}

-- Part 1: Given A ∪ B = A, find possible values of a.
theorem part1 (a : ℝ) (h1 : A a ∪ B a = A a) : a = 0 ∨ a = -1 := by
  sorry

-- Part 2: Given A ⊆ complementC, find the range of a.
theorem part2 (a : ℝ) (h2 : A a ⊆ complementC) : a ∈ Set.Iio 1 ∨ a ∈ Set.Ioo 1 2 := by
  sorry

end part1_part2_l806_806539


namespace find_valid_k_l806_806743

def product_of_digits (k : ℕ) : ℕ :=
  k.digits 10 |> List.foldl (λ a b => a * b) 1

theorem find_valid_k : {k : ℕ // k > 0 ∧ product_of_digits k = (25 * k / 8) - 211} = {72, 88} :=
by
  sorry

end find_valid_k_l806_806743


namespace hyperbola_eccentricity_l806_806235

theorem hyperbola_eccentricity : 
  ∀ (a b : ℝ), 
  (b / a = sqrt 3) → 
  ∀ (e : ℝ), 
  (e = sqrt (1 + (b^2 / a^2))) →
  (e = 2) :=
by
  intros a b h_asym e h_ecc
  have hb_sq : b^2 = 3 * a^2 := 
    by rw [←sq b, ←sq a, ←mul_assoc, h_asym, pow_two (sqrt 3)]
  rw [h_ecc, hb_sq]
  sorry

end hyperbola_eccentricity_l806_806235


namespace among_given_inequalities_true_condition_l806_806319

theorem among_given_inequalities_true_condition 
  (a b : ℝ) (h₁ : 0 < a) (h₂ : a < b) (h₃ : a + b = 1) :
  (log 2 (b - a) < 0 ∧ log 2 (b / a + a / b) > 1) :=
sorry

end among_given_inequalities_true_condition_l806_806319


namespace coefficient_x3_l806_806279

noncomputable def polynomial : ℤ[X] := 4*(X^3 - 2*X^4) + 3*(X^2 - 3*X^3 + 4*X^6) - (5*X^4 - 2*X^3)

theorem coefficient_x3 : polynomial.coeff 3 = -3 := by
  sorry

end coefficient_x3_l806_806279


namespace hexagon_perimeter_eq_6_sqrt_2_l806_806276

theorem hexagon_perimeter_eq_6_sqrt_2 :
  let s_large := 2
  let area_large := (Real.sqrt 3 / 4) * s_large^2
  let area_small := area_large / 2
  let s_small := Real.sqrt (4 * area_small / Real.sqrt 3)
  let side_length_hexagon := s_small
  let perimeter_hexagon := 6 * side_length_hexagon
  in perimeter_hexagon = 6 * Real.sqrt 2 :=
by
  sorry

end hexagon_perimeter_eq_6_sqrt_2_l806_806276


namespace point_in_fourth_quadrant_l806_806874

def quadrant_of_point (x y : ℝ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "Origin or Axis"

theorem point_in_fourth_quadrant : quadrant_of_point 8 (-5) = "Fourth quadrant" :=
by
  sorry

end point_in_fourth_quadrant_l806_806874


namespace number_of_zeros_of_f_l806_806163

def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

theorem number_of_zeros_of_f (e : Emetric) : ∃! x : ℝ, f x = 0 := by
  sorry

end number_of_zeros_of_f_l806_806163


namespace count_four_digit_numbers_with_5_or_7_l806_806398

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l806_806398


namespace probability_correct_l806_806974

noncomputable def probability_all_players_have_5_after_2023_rings 
    (initial_money : ℕ)
    (num_rings : ℕ) 
    (target_money : ℕ)
    : ℝ := 
    if initial_money = 5 ∧ num_rings = 2023 ∧ target_money = 5 
    then 1 / 4 
    else 0

theorem probability_correct : 
        probability_all_players_have_5_after_2023_rings 5 2023 5 = 1 / 4 := 
by 
    sorry

end probability_correct_l806_806974


namespace find_angle_A_find_AB_l806_806449

theorem find_angle_A (A B C : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C)) (h2 : A + B + C = Real.pi) :
  A = Real.pi / 3 := by
  sorry

theorem find_AB (A B C : ℝ) (AB BC AC : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C))
  (h2 : BC = 2) (h3 : 1 / 2 * AB * AC * Real.sin (Real.pi / 3) = Real.sqrt 3)
  (h4 : A = Real.pi / 3) :
  AB = 2 := by
  sorry

end find_angle_A_find_AB_l806_806449


namespace fruit_weights_determined_l806_806948

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806948


namespace solve_for_q_l806_806131

theorem solve_for_q (k l q : ℝ) 
  (h1 : 3 / 4 = k / 48)
  (h2 : 3 / 4 = (k + l) / 56)
  (h3 : 3 / 4 = (q - l) / 160) :
  q = 126 :=
  sorry

end solve_for_q_l806_806131


namespace light_glow_ending_time_correct_l806_806998

def light_glow_period : ℝ := 15
def light_glowed_times : ℝ := 331.27
def starting_time_seconds : ℝ := (1 * 3600) + (57 * 60) + 58  -- 1:57:58 am in seconds

def ending_time_seconds : ℝ := starting_time_seconds + (light_glowed_times * light_glow_period)

noncomputable def ending_hours : ℕ := (ending_time_seconds / 3600).to_nat
noncomputable def remaining_seconds_after_hours : ℝ := ending_time_seconds - (ending_hours * 3600)
noncomputable def ending_minutes : ℕ := (remaining_seconds_after_hours / 60).to_nat
noncomputable def remaining_seconds_after_minutes : ℝ := remaining_seconds_after_hours - (ending_minutes * 60)
noncomputable def ending_seconds : ℝ := remaining_seconds_after_minutes

theorem light_glow_ending_time_correct :
  let ending_time := (5, 18, 45.05) in
  ending_hours = 5 ∧ ending_minutes = 18 ∧ abs (ending_seconds - 45.05) < 1e-2 :=
by
  sorry

end light_glow_ending_time_correct_l806_806998


namespace solution_of_equation_l806_806744

-- Condition: Define the equation
def equation (x : ℝ) : Prop := sqrt (x - 5 * sqrt (x - 9)) + 3 = sqrt (x + 5 * sqrt (x - 9)) - 3

-- Condition: x > 9
def condition (x : ℝ) : Prop := x > 9

-- Theorem: Solution
theorem solution_of_equation : ∃ x : ℝ, condition x ∧ equation x :=
  sorry

end solution_of_equation_l806_806744


namespace total_cement_used_l806_806574

-- Define the amounts of cement used for Lexi's street and Tess's street
def cement_used_lexis_street : ℝ := 10
def cement_used_tess_street : ℝ := 5.1

-- Prove that the total amount of cement used is 15.1 tons
theorem total_cement_used : cement_used_lexis_street + cement_used_tess_street = 15.1 := sorry

end total_cement_used_l806_806574


namespace fruit_weights_l806_806954

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806954


namespace ordered_pairs_count_l806_806750

theorem ordered_pairs_count : 
  (∃ pairs : List (ℤ × ℤ), 
    (∀ (p : ℤ × ℤ), p ∈ pairs → 
      let (m, n) := p in m * n ≥ 0 ∧ m^3 + n^3 + 81 * m * n = 27^3) ∧
    pairs.length = 29) :=
sorry

end ordered_pairs_count_l806_806750


namespace remainder_x_150_div_x_plus_1_pow_4_l806_806302

theorem remainder_x_150_div_x_plus_1_pow_4 :
  ∀ x : ℤ, Polynomial.x ^ 150 % (Polynomial.x + 1) ^ 4 = 551300 * Polynomial.x ^ 3 + 277161 * Polynomial.x ^ 2 + 736434 * Polynomial.x - 663863 :=
by
  intro x
  sorry

end remainder_x_150_div_x_plus_1_pow_4_l806_806302


namespace largest_n_2x2_subarray_sum_l806_806731

theorem largest_n_2x2_subarray_sum (A : Fin 5 → Fin 5 → ℕ) 
  (distinct_elements : ∀ i j, A i j ∈ Finset.range 1 26)
  (pairwise_distinct : ∀ (i1 j1 i2 j2 : Fin 5), (i1 ≠ i2 ∨ j1 ≠ j2) → A i1 j1 ≠ A i2 j2) :
  ∃ N, (∀ i j : Fin 4, A i j + A (i+1) j + A i (j+1) + A (i+1) (j+1) ≥ N) ∧ N = 45 :=
by {
  sorry
}

end largest_n_2x2_subarray_sum_l806_806731


namespace imaginary_part_of_one_over_one_plus_i_l806_806599

theorem imaginary_part_of_one_over_one_plus_i : 
  complex.im (1 / (1 + complex.i)) = -1/2 := by
  sorry

end imaginary_part_of_one_over_one_plus_i_l806_806599


namespace fruit_weights_l806_806926

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l806_806926


namespace sandy_remaining_puppies_l806_806975

-- Definitions from the problem
def initial_puppies : ℕ := 8
def given_away_puppies : ℕ := 4

-- Theorem statement
theorem sandy_remaining_puppies : initial_puppies - given_away_puppies = 4 := by
  sorry

end sandy_remaining_puppies_l806_806975


namespace max_pairs_of_corner_and_squares_l806_806184

def rectangle : ℕ := 3 * 100
def unit_squares_per_pair : ℕ := 4 + 3

-- Given conditions
def conditions := rectangle = 300 ∧ unit_squares_per_pair = 7

-- Proof statement
theorem max_pairs_of_corner_and_squares (h: conditions) : ∃ n, n = 33 ∧ n * unit_squares_per_pair ≤ rectangle := 
sorry

end max_pairs_of_corner_and_squares_l806_806184


namespace inequality_sum_ai_gt_three_halves_l806_806895

theorem inequality_sum_ai_gt_three_halves (n : ℕ) (a : Fin n → ℝ)
  (h1 : ∑ i, (a i)^3 = 3)
  (h2 : ∑ i, (a i)^5 = 5) :
  ∑ i, (a i) > 3 / 2 :=
by
  sorry

end inequality_sum_ai_gt_three_halves_l806_806895


namespace distinct_product_pairs_count_l806_806899

-- Given conditions
def T := {n | n ∣ 72000 ∧ n > 0}

-- Proof goal
theorem distinct_product_pairs_count : 
  (finset.card (finset.filter (λ p : ℕ × ℕ, p.1 ≠ p.2) 
                              ((finset.product T T).map (λ p, p.1 * p.2)))) = 381 :=
by
  sorry

end distinct_product_pairs_count_l806_806899


namespace probability_b_wins_l806_806864

theorem probability_b_wins (h_independent : ∀ (A B : Prop), A ∧ B → false)
  (h_outcomes : ∀ (A B : Prop), (A ∨ B) ∧ h_independent A B)
  (h_prob_A : ℝ) (h_A_wins : h_prob_A = 0.41) :
  let h_prob_B := 1 - h_prob_A in
  h_prob_B = 0.59 :=
by
  sorry

end probability_b_wins_l806_806864


namespace four_digit_integers_with_5_or_7_l806_806385

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l806_806385


namespace compute_remainder_3_pow_n_div_1000_l806_806481

def remainder_3_pow_49_mod_1000 : ℕ := 83

theorem compute_remainder_3_pow_n_div_1000 (n : ℕ) (h : n = 49) :
  (3 ^ n) % 1000 = remainder_3_pow_49_mod_1000 :=
by {
  rw h,
  sorry
}

end compute_remainder_3_pow_n_div_1000_l806_806481


namespace solve_quintic_equation_l806_806578

theorem solve_quintic_equation :
  {x : ℝ | x * (x - 3)^2 * (5 + x) * (x^2 - 1) = 0} = {0, 3, -5, 1, -1} :=
by
  sorry

end solve_quintic_equation_l806_806578


namespace least_palindrome_div_by_5_l806_806113

/-- A palindrome is a number that reads the same backward and forward. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

/-- The least possible positive four-digit palindrome that is divisible by 5 is 5005. -/
theorem least_palindrome_div_by_5 : ∃ n : ℕ, is_palindrome n ∧ (1000 ≤ n ∧ n < 10000) ∧ n % 5 = 0 ∧ ∀ m : ℕ, is_palindrome m ∧ (1000 ≤ m ∧ m < 10000) ∧ m % 5 = 0 → n ≤ m :=
begin
  use 5005,
  split,
  { unfold is_palindrome,
    dsimp,
    norm_num },
  split,
  { split, norm_num, norm_num },
  split,
  { norm_num },
  { intros m h,
    cases h with hm1 hm2,
    cases hm2 with hm3 hm4,
    cases hm3 with hm5 hm6,
    have h5 : m % 5 = 0 := hm4,
    norm_cast at h5,
    by_cases h : m = 5005,
    { rw h, exact le_refl _ },
    { have h7 : 5005 <= m, sorry } },
end

end least_palindrome_div_by_5_l806_806113


namespace difference_sixth_seventh_l806_806988

theorem difference_sixth_seventh
  (A1 A2 A3 A4 A5 A6 A7 A8 : ℕ)
  (h_avg_8 : (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8) / 8 = 25)
  (h_avg_2 : (A1 + A2) / 2 = 20)
  (h_avg_3 : (A3 + A4 + A5) / 3 = 26)
  (h_A8 : A8 = 30)
  (h_A6_A8 : A6 = A8 - 6) :
  A7 - A6 = 4 :=
by
  sorry

end difference_sixth_seventh_l806_806988


namespace sum_bk_geq_four_div_nplus1_l806_806533

open Real

theorem sum_bk_geq_four_div_nplus1 (n : ℕ) (b : ℕ → ℝ) (x : ℕ → ℝ) 
  (x_nonzero : ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ x i ≠ 0)
  (b_positive : ∀ k, 1 ≤ k ∧ k ≤ n → b k > 0)
  (sys_equations : ∀ k, 1 ≤ k ∧ k ≤ n → x (k - 1) - 2 * x k + x (k + 1) + b k * x k = 0)
  (x_zero : x 0 = 0 ∧ x (n + 1) = 0) :
  (Finset.range n).sum (λ k, b (k + 1)) ≥ 4 / (n + 1) :=
sorry

end sum_bk_geq_four_div_nplus1_l806_806533


namespace quadrilateral_CD_l806_806872

open Real

-- Definitions
variable (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variable (AB AD BD DE CD CE : ℝ)

-- Conditions
-- Quadrilateral ABCD s.t. ∠BAD ≅ ∠ADC, ∠CBD ≅ ∠BCA
variable (angle_BAD_congr_angle_ADC : ∀ {A B C D : Type}, ∠BAD = ∠ADC)
variable (angle_CBD_congr_angle_BCA : ∀ {A B C D : Type}, ∠CBD = ∠BCA)

-- Given lengths
variable (AB_length : AB = 10)
variable (BD_length : BD = 12)
variable (AD_length : AD = 7)

theorem quadrilateral_CD (h_angle_BAD_congr_angle_ADC : ∠BAD = ∠ADC)
  (h_angle_CBD_congr_angle_BCA : ∠CBD = ∠BCA)
  (h_AB_length : AB = 10)
  (h_BD_length : BD = 12)
  (h_AD_length : AD = 7) : 
  ∃ (m n : ℤ) (hmn_coprime : Int.gcd m n = 1), CD = m / n ∧ m + n = 247 :=
by 
  -- To prove that given the conditions, CD = 240/7, m = 240, n = 7, and m + n = 247
  sorry

end quadrilateral_CD_l806_806872


namespace max_buses_l806_806862

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l806_806862


namespace distinct_integer_counts_l806_806754

def f (x : ℝ) : ℝ := 
  ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_integer_counts : (∃ S : Finset ℤ, ∀ x : ℝ, (0 ≤ x ∧ x ≤ 100) → f x ∈ S ∧ S.card = 734) :=
  sorry

end distinct_integer_counts_l806_806754


namespace Calculate_Area_ABC_l806_806465

-- Definitions:
-- triangle ABC with right angle at A, B = C, and hypotenuse AC = 8√2

structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

def is_right_angle {α : Type} [LinearOrderedField α] (t : Triangle α) : Prop :=
∠ t.A t.B t.C = 90

def is_isosceles_right {α : Type} [LinearOrderedField α] (t : Triangle α) : Prop :=
  is_right_angle t ∧ ∠ t.B t.A t.C = ∠ t.C t.A t.B

noncomputable def hypotenuse {α : Type} [LinearOrderedField α] (t : Triangle α) : α :=
  dist t.A t.C

noncomputable def legs_length {α : Type} [LinearOrderedField α] (h : hypotenuse t = 8 * real.sqrt 2) : α :=
  8

noncomputable def area {α : Type} [LinearOrderedField α] (t : Triangle α) : α :=
  (1/2) * (legs_length h) * (legs_length h)

theorem Calculate_Area_ABC : 
  is_right_angle t ∧ ∠ t.B t.A t.C = ∠ t.C t.A t.B ∧ hypotenuse t = 8 * real.sqrt 2 →
  area t = 32 :=
by sorry

end Calculate_Area_ABC_l806_806465


namespace function_extremes_l806_806793

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 + 3 * (Real.log x / Real.log 2) + 2

theorem function_extremes :
  (∀ t : ℝ, t = Real.log x / Real.log 2 → -2 ≤ t ∧ t ≤ 2) ∧
  (∀ x : ℝ, f x ≥ -1/4 ∧ (f x = -1/4 → x = 2^(-3/2))) ∧
  (∀ x : ℝ, f x ≤ 12 ∧ (f x = 12 → x = 4)) := by
  -- Proof is omitted.
  sorry

end function_extremes_l806_806793


namespace intersection_lies_on_circumcircle_l806_806870

-- Define the given problem in Lean 4

variables {A B C I P D E F G : Point}

-- Assumptions and definitions
def isosceles_triangle (A B C : Point) :=
  dist A C = dist B C

def incenter_of_triangle (I A B C : Point) :=
  incenter I A B C

def lies_on_circumcircle (P A I B : Point) :=
  ∃ O : Point, circumcircle O A I B P -- P lies on the circumcircle of triangle AIB

def intersection_of_parallel_lines (P A B C D E F G : Point) :=
  parallel P D C A ∧
  parallel P E C B ∧
  parallel P F A B ∧
  parallel P G A B

def intersection_of_lines (D F E G : Point) :=
  ∃ M : Point, on_line DF M ∧ on_line EG M

-- The theorem statement
theorem intersection_lies_on_circumcircle
  (h1 : isosceles_triangle A B C)
  (h2 : incenter_of_triangle I A B C)
  (h3 : lies_on_circumcircle P A I B)
  (h4 : intersection_of_parallel_lines P A B C D E F G)
  (h5 : intersection_of_lines D F E G) :
  ∃ M : Point, lies_on_circumcircle M A B C :=
sorry

end intersection_lies_on_circumcircle_l806_806870


namespace sum_of_number_and_conjugate_l806_806720

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l806_806720


namespace rectangle_square_ratio_l806_806124

noncomputable def rectangle_square_area_ratio (a x y : ℝ) : ℝ :=
  let overlap_square := 0.2 * a^2
  let overlap_rectangle := 0.5 * x * y
  let condition1 := overlap_square
  let condition2 := overlap_rectangle

theorem rectangle_square_ratio (a x y : ℝ)
  (h1 : 0.2 * a^2 = 0.5 * x * y)
  (h2 : y = a / 5)
  (h3 : x = 2 * a) :
  x / y = 10 :=
by
  rw [h3, h2]
  field_simp
  norm_num

#check rectangle_square_ratio

end rectangle_square_ratio_l806_806124


namespace X_lies_on_altitude_BH_l806_806499

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806499


namespace apples_for_juice_is_correct_l806_806142

noncomputable def apples_per_year : ℝ := 8 -- 8 million tons
noncomputable def percentage_mixed : ℝ := 0.30 -- 30%
noncomputable def remaining_apples := apples_per_year * (1 - percentage_mixed) -- Apples after mixed
noncomputable def percentage_for_juice : ℝ := 0.60 -- 60%
noncomputable def apples_for_juice := remaining_apples * percentage_for_juice -- Apples for juice

theorem apples_for_juice_is_correct :
  apples_for_juice = 3.36 :=
by
  sorry

end apples_for_juice_is_correct_l806_806142


namespace height_on_fifth_bounce_l806_806625

-- Define initial conditions
def initial_height : ℝ := 96
def initial_efficiency : ℝ := 0.5
def efficiency_decrease : ℝ := 0.05
def air_resistance_loss : ℝ := 0.02

-- Recursive function to compute the height after each bounce
def bounce_height (height : ℝ) (efficiency : ℝ) : ℝ :=
  let height_after_bounce := height * efficiency
  height_after_bounce - (height_after_bounce * air_resistance_loss)

-- Function to compute the bounce efficiency after each bounce
def bounce_efficiency (initial_efficiency : ℝ) (n : ℕ) : ℝ :=
  initial_efficiency - n * efficiency_decrease

-- Function to calculate the height after n-th bounce
def height_after_n_bounces (n : ℕ) : ℝ :=
  match n with
  | 0     => initial_height
  | n + 1 => bounce_height (height_after_n_bounces n) (bounce_efficiency initial_efficiency n)

-- Lean statement to prove the problem
theorem height_on_fifth_bounce :
  height_after_n_bounces 5 = 0.82003694685696 := by
  sorry

end height_on_fifth_bounce_l806_806625


namespace min_students_using_both_l806_806031

theorem min_students_using_both (n: ℕ) : 
  (3 / 7 : ℝ) * n ∈ set_of m : ℝ | m.isNat → 
  (5 / 6 : ℝ) * n ∈ set_of m : ℝ | m.isNat → 
  n = 42 → 
  ∃ x : ℕ, x ≥ 11 ∧ x = 18 + 35 - n := 
sorry

end min_students_using_both_l806_806031


namespace count_four_digit_numbers_with_5_or_7_l806_806399

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l806_806399


namespace ellipse_eccentricity_l806_806146

theorem ellipse_eccentricity
  (a b : ℝ)
  (h_major : 2 * a = 4)
  (h_minor : 2 * b = 2)
  (h_condition : a > b)
  : (real.sqrt (a^2 - b^2)) / a = real.sqrt 3 / 2 :=
by
  sorry

end ellipse_eccentricity_l806_806146


namespace difference_of_digits_is_six_l806_806617

theorem difference_of_digits_is_six (a b : ℕ) (h_sum : a + b = 10) (h_number : 10 * a + b = 82) : a - b = 6 :=
sorry

end difference_of_digits_is_six_l806_806617


namespace circle_center_radius_l806_806354

-- Definition of the equation for the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 6 * y = 0

-- The center of the circle
def center : (ℝ × ℝ) := (1, -3)

-- The radius of the circle
def radius : ℝ := Real.sqrt 10

-- Statement to be proved
theorem circle_center_radius : 
  (∀ x y : ℝ, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by
  sorry

end circle_center_radius_l806_806354


namespace seventh_selected_bag_number_l806_806626

def total_bags : Nat := 800
def sample_size : Nat := 60
def starting_row : Nat := 7

-- Random number table excerpted from rows 7 to 10.
def random_numbers : List Nat := 
  [16, 22, 77, 94, 39,  49, 54, 43, 54, 82,  17, 37, 93, 23, 78,  87, 35, 20, 96, 43,  
  84, 26, 34, 91, 64, 84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 
  21, 76, 33, 50, 25, 83, 92, 12, 06, 76, 63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 
  98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79, 33, 21, 12, 34, 29,
  78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54,
  57, 60, 86, 32, 44,  09, 47, 27, 96, 54,  49, 17, 46, 09, 62,  90, 52, 84, 77, 27, 
  08, 02, 73, 43, 28]

-- Prove that the 7th valid bag number selected is 744.
theorem seventh_selected_bag_number :
  ∃ (seq : List Nat), 
    seq = [16, 22, 77, 94, 39,  49, 54, 43, 54, 82,  17, 37, 93, 23, 78,  87, 35, 20, 96, 43,
           84, 26, 34, 91, 64, 84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 
           21, 76, 33, 50, 25, 83, 92, 12, 06, 76, 63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 
           98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79, 33, 21, 12, 34, 29,
           78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54,
           57, 60, 86, 32, 44,  09, 47, 27, 96, 54,  49, 17, 46, 09, 62,  90, 52, 84, 77, 27, 
           08, 02, 73, 43, 28] → 
    random_numbers.length ≥ 60 
    ∧ random_numbers.nth 6 = some 744 := sorry

end seventh_selected_bag_number_l806_806626


namespace sum_of_first_20_terms_l806_806879

theorem sum_of_first_20_terms :
  ∃ (a : ℕ → ℝ), a 1 = -2 ∧ (∀ n : ℕ, 2 * a (n + 1) = 1 + 2 * a n) →
  (∑ i in Finset.range 20, a (i + 1)) = 55 :=
sorry

end sum_of_first_20_terms_l806_806879


namespace fruit_weights_determined_l806_806950

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806950


namespace equation_solution_simplify_7_52_simplify_7_53_l806_806132

theorem equation_solution (a : ℝ) (x : ℝ) (h1 : a ≠ 3) :
  (3 * 4^(x-2) + 27 = a + a * 4^(x-2)) ↔
  (x = 2 + real.logb 4 (a-27) - real.logb 4 (3-a) ∧ 3 < a < 27) :=
by 
  sorry

theorem simplify_7_52 (a b h : ℝ) :
  ∃ c : ℝ, c = (b^(real.logb 10 a / real.log a) * a^(real.logb 100 h / real.log h))^(2 * real.logb a (a + b)) :=
by 
  sorry

theorem simplify_7_53 (a b : ℝ) :
  ∃ d : ℝ, d = ((real.logb b (4 * a) + real.logb a (4 * b) + 2)^(1/2) + 2)^(1/2) - real.logb b a - real.logb a b :=
by 
  sorry

end equation_solution_simplify_7_52_simplify_7_53_l806_806132


namespace afternoon_to_morning_ratio_l806_806246

theorem afternoon_to_morning_ratio
  (A : ℕ) (M : ℕ)
  (h1 : A = 340)
  (h2 : A + M = 510) :
  A / M = 2 :=
by
  sorry

end afternoon_to_morning_ratio_l806_806246


namespace point_on_altitude_l806_806523

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806523


namespace two_triangles_with_given_heights_l806_806277

theorem two_triangles_with_given_heights (a mb mc : ℝ) (ha : a = 6) (hmb : mb = 1) (hmc : mc = 2) :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧
    (t1.side_a = a ∧ t1.height_b = mb ∧ t1.height_c = mc) ∧
    (t2.side_a = a ∧ t2.height_b = mb ∧ t2.height_c = mc) :=
sorry

end two_triangles_with_given_heights_l806_806277


namespace min_value_inequality_l806_806084

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ( (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) ) / (x * y * z) ≥ 336 := 
by
  sorry

end min_value_inequality_l806_806084


namespace lines_intersection_l806_806344

theorem lines_intersection (n : ℕ) (h : n > 1) (lines : fin n → Prop) 
  (non_parallel: ∀ i j : fin n, i ≠ j → ¬ (lines i = lines j)) :
  (∃ P : Prop, ∃ l1 l2 : fin n, l1 ≠ l2 ∧ P ∈ {lines l1 ∩ lines l2} ↔ exactly_two_lines_intersect P lines) ∨
  (∃ P : Prop, ∀ i : fin n, P ∈ {lines i} ↔ all_lines_intersect_at P lines) :=
by
  sorry

end lines_intersection_l806_806344


namespace minimize_dot_product_of_QA_QB_l806_806777

noncomputable def coordinates_of_Q : ℝ × ℝ × ℝ := 
  let OA := (1, 2, 3)
  let OB := (2, 1, 2)
  let OP := (1, 1, 2)
  let λ := 4 / 3
  (λ * OP.1, λ * OP.2, λ * OP.3)

theorem minimize_dot_product_of_QA_QB :
  let Q := coordinates_of_Q in
  Q = (4 / 3, 4 / 3, 8 / 3) := 
by
  sorry

end minimize_dot_product_of_QA_QB_l806_806777


namespace find_point_D_l806_806913

-- Define points A and B
structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := -3, y := 2}
def B : Point := {x := 5, y := 10}

-- Define the condition for the distance
def distance (P Q : Point) : ℚ := real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def is_twice_distance (D : Point) : Prop :=
  distance D A = 2 * distance D B

-- Define the slope condition
def slope (P Q : Point) : ℚ := (Q.y - P.y) / (Q.x - P.x)

def is_slope_one (A B : Point) : Prop := slope A B = 1

-- Define point D
def D : Point := {x := 7/3, y := 22/3}

-- Formulate the proof problem statement
theorem find_point_D :
  is_twice_distance D ∧ is_slope_one A B :=
by
  sorry

end find_point_D_l806_806913


namespace X_on_altitude_BH_l806_806486

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806486


namespace dave_walked_time_l806_806725

variable (total_time : ℕ) (jog_ratio walk_ratio : ℕ)
variable (time_jogged time_walked : ℕ)

-- Define the total time condition
def total_time_condition : Prop := total_time = 21

-- Define the ratio condition in terms of multiples of a variable x
def ratio_condition : Prop := ∃ x : ℕ, time_jogged = 4 * x ∧ time_walked = 3 * x

-- Combine the conditions to make the complete assumption for the problem
def all_conditions : Prop :=
  total_time_condition total_time ∧
  ratio_condition time_jogged time_walked ∧
  time_jogged + time_walked = total_time

-- State the theorem to prove
theorem dave_walked_time : all_conditions total_time time_jogged time_walked jog_ratio walk_ratio → time_walked = 9 :=
by 
  sorry

end dave_walked_time_l806_806725


namespace certain_number_is_11_l806_806672

theorem certain_number_is_11 (x : ℝ) (h : 15 * x = 165) : x = 11 :=
by {
  sorry
}

end certain_number_is_11_l806_806672


namespace max_buses_constraint_satisfied_l806_806840

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l806_806840


namespace lines_parallel_if_angles_satisfy_log_condition_l806_806458

-- Conditions translated to Lean definitions
variables {α β γ : ℝ}
variables {a c : ℝ}
variables {x y : ℝ}

-- The given condition in the problem
def condition : Prop := log (sin α) + log (sin γ) = 2 * log (sin β)

-- The problem statement
theorem lines_parallel_if_angles_satisfy_log_condition
  (h : condition) : 
  (∀ x y, ∀ a c, (x * (sin α)^2 + y * (sin α) = a) ∧ (x * (sin β)^2 + y * (sin γ) = c) → 
  - (sin α) = - (sin α)) :=
sorry

end lines_parallel_if_angles_satisfy_log_condition_l806_806458


namespace sum_of_coefficients_l806_806618

def polynomial (x y : ℕ) : ℕ := (x^2 - 3*x*y + y^2)^8

theorem sum_of_coefficients : polynomial 1 1 = 1 :=
sorry

end sum_of_coefficients_l806_806618


namespace number_of_perfect_squares_l806_806417

theorem number_of_perfect_squares (a b : ℕ) (ha : 50 < a^2) (hb : b^2 < 300) :
  ∃ (n : ℕ), a ≤ n ∧ n ≤ b ∧ ∑ i in (finset.range (b - a + 1)).filter (λ n, 50 < n^2 ∧ n^2 < 300), 1 = 10 :=
sorry

end number_of_perfect_squares_l806_806417


namespace john_owes_more_than_tripled_amount_l806_806475

theorem john_owes_more_than_tripled_amount:
  let P := 1500 in
  let r := 0.06 in
  ∃ (t : ℕ), (1 + r)^t > 3 ∧ (∀ k : ℕ, k < t → (1 + r)^k ≤ 3) := sorry

end john_owes_more_than_tripled_amount_l806_806475


namespace determine_b_constant_remainder_l806_806281

theorem determine_b_constant_remainder (b : ℚ) :
  (∀ r (x : ℚ), r = 12 * x^4 - 14 * x^3 + b * x^2 + 7 * x + 9 → r % (3 * x^2 - 4 * x + 2) = 0 -> r) → b = 16 / 3 :=
sorry

end determine_b_constant_remainder_l806_806281


namespace sum_of_integers_in_interval_l806_806135

-- Define the conditions
def condition1 (x : ℝ) : Prop := x^2 - x - 56 ≥ 0
def condition2 (x : ℝ) : Prop := x^2 - 25 * x + 136 ≥ 0
def condition3 (x : ℝ) : Prop := x > 8
def condition4 (x : ℝ) : Prop := x + 7 ≥ 0

-- Define the proof statement
theorem sum_of_integers_in_interval :
  ∑ x in (finset.Icc (-25 : ℤ) (25 : ℤ)).filter (λ x, 
    condition1 x ∧ condition2 x ∧ condition3 x ∧ condition4 x), 
    x = -285 :=
by sorry

end sum_of_integers_in_interval_l806_806135


namespace max_value_function_find_tan_alpha_l806_806217

-- Problem 1: Maximum value of the function
theorem max_value_function (x : ℝ) (h : x ∈ Icc (Real.pi / 6) (7 * Real.pi / 6)) :
  (3 - Real.sin x - 2 * (Real.cos x)^2) ≤ 2 :=
sorry

-- Problem 2: Finding tan alpha
variable (α β : ℝ)
hypothesis (h1 : 5 * Real.sin β = Real.sin (2 * α + β))
hypothesis (h2 : Real.tan (α + β) = 9 / 4)

theorem find_tan_alpha :
  Real.tan α = 3 / 2 :=
sorry

end max_value_function_find_tan_alpha_l806_806217


namespace intersecting_lines_l806_806159

theorem intersecting_lines (c d : ℝ) :
  (∀ x y : ℝ, (x = 1/3 * y + c ∧ y = 1/3 * x + d) → (x = 3 ∧ y = 3)) →
  c + d = 4 :=
by
  intros h
  -- We need to validate the condition holds at the intersection point
  have h₁ : 3 = 1/3 * 3 + c := by sorry
  have h₂ : 3 = 1/3 * 3 + d := by sorry
  -- Conclude that c = 2 and d = 2
  have hc : c = 2 := by sorry
  have hd : d = 2 := by sorry
  -- Thus the sum c + d = 4
  show 2 + 2 = 4 from rfl

end intersecting_lines_l806_806159


namespace fixed_point_of_line_l806_806002

theorem fixed_point_of_line (a : ℝ) : ∀ a : ℝ, (∃ (x y : ℝ), x = -1 ∧ y = -1 ∧ a * x + y + a + 1 = 0) :=
by
  intros a
  use [-1, -1]
  split
  { refl },
  split
  { refl },
  calc
    a * -1 + -1 + a + 1 = -a - 1 + a + 1 : by ring
                   ... = 0 : by ring

end fixed_point_of_line_l806_806002


namespace family_percentage_eaten_after_dinner_l806_806914

theorem family_percentage_eaten_after_dinner
  (total_brownies : ℕ)
  (children_percentage : ℚ)
  (left_over_brownies : ℕ)
  (lorraine_extra_brownie : ℕ)
  (remaining_percentage : ℚ) :
  total_brownies = 16 →
  children_percentage = 0.25 →
  lorraine_extra_brownie = 1 →
  left_over_brownies = 5 →
  remaining_percentage = 50 := by
  sorry

end family_percentage_eaten_after_dinner_l806_806914


namespace teacher_can_achieve_desired_pieces_of_candy_l806_806288

variables {Student : Type} [Fintype Student]

-- Condition: Each student has 0 to 6 pieces of candy.
def pieces_of_candy (student : Student) : ℕ := 0 -- Here, it's defined generally; initial conditions will bind appropriately.

-- Condition: The teacher can choose some students and give one candy to the chosen and their friends.
-- This models giving candy to selected students and their direct friends.
def give_candy (chosen : set Student) (friends : Student → set Student) : set Student :=
  chosen ∪ (chosen.mono_image friends)

-- Condition: A student who receives the seventh piece eats all 7 pieces.
def eat_all_seven_pieces (candy_count : ℕ) : ℕ :=
  if candy_count = 7 then 0 else candy_count

-- Condition: For every pair of students, there is at least one student who is friends with exactly one of them.
def friend_condition (students : set Student) (friends : Student → set Student) : Prop :=
  ∀ (a b : Student), a ≠ b → ∃ c : Student, c ≠ a ∧ c ≠ b ∧ (c ∈ friends a ∨ c ∈ friends b)

theorem teacher_can_achieve_desired_pieces_of_candy
  {friends : Student → set Student}
  (H : friend_condition univ friends)
  (initial : Student → ℕ)
  (desired : Student → ℕ) :
  ∃ (steps : ℕ) (final : Student → ℕ), ∀ (student : Student), final student = desired student :=
sorry

end teacher_can_achieve_desired_pieces_of_candy_l806_806288


namespace AH_perp_BP_l806_806760

/-- Isosceles Triangle ABC with AB = AC
    and M is the midpoint of AC. --/
variables {A B C M H P : Type} 
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space M]
variable [metric_space H]
variable [metric_space P]

open real

noncomputable def midpoint (x y : Type) [metric_space x] [metric_space y] 
: metric_space ((x + y) / 2) := sorry

noncomputable def perpendicular (x y : Type) [metric_space x] 
[metric_space y] : Prop := sorry

noncomputable def isosceles (x y z : Type) [metric_space x]
[metric_space y] [metric_space z] : Prop := distance x y = distance x z

variable (ABC_isosceles : isosceles A B C )
variable (M_midpoint_AC : midpoint A C M)
variable (MH_perpendicular_BC : perpendicular M H)
variable (P_midpoint_MH : midpoint M H P)

theorem AH_perp_BP : perpendicular H B P :=
sorry

end AH_perp_BP_l806_806760


namespace find_k_l806_806910

noncomputable def f (x k : ℝ) : ℝ := x * (x + k) * (x + 2 * k) * (x - 3 * k)

theorem find_k (k : ℝ) (h : deriv (f x k) 0 = 6) : k = -1 := sorry

end find_k_l806_806910


namespace P_at_one_seventh_l806_806904

noncomputable def P (x : ℝ) : ℝ := ∑' n, if even n then 4 * x^n else 8 * x^n

theorem P_at_one_seventh : P (1 / 7) = 5 / 3 := by
  sorry

end P_at_one_seventh_l806_806904


namespace maximum_number_of_buses_l806_806848

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l806_806848


namespace matrix_pow_99_l806_806478

open_locale matrix

def B : matrix (fin 3) (fin 3) ℤ :=
  ![![0, 0, 0], ![0, 0, 1], ![0, -1, 0]]

theorem matrix_pow_99 :
  B ^ 99 = ![![0, 0, 0], ![0, 0, -1], ![0, 1, 0]] :=
by {
  sorry
}

end matrix_pow_99_l806_806478


namespace f_of_f_neg2_eq_pi_l806_806666

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_of_f_neg2_eq_pi : f (f (-2)) = Real.pi :=
  sorry

end f_of_f_neg2_eq_pi_l806_806666


namespace smallest_t_l806_806001

noncomputable def D_n (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, (2 * (k + 1) + 1) / (2:ℝ)^((k + 1):ℝ))

theorem smallest_t (t : ℝ) : (∀ n : ℕ, D_n n < t) ↔ 5 := sorry

end smallest_t_l806_806001


namespace smallest_number_is_111111_2_l806_806257

def base9_to_decimal (n : Nat) : Nat :=
  (n / 10) * 9 + (n % 10)

def base6_to_decimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n % 100) / 10) * 6 + (n % 10)

def base4_to_decimal (n : Nat) : Nat :=
  (n / 1000) * 64

def base2_to_decimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n % 100000) / 10000) * 16 + ((n % 10000) / 1000) * 8 + ((n % 1000) / 100) * 4 + ((n % 100) / 10) * 2 + (n % 10)

theorem smallest_number_is_111111_2 :
  let n1 := base9_to_decimal 85
  let n2 := base6_to_decimal 210
  let n3 := base4_to_decimal 1000
  let n4 := base2_to_decimal 111111
  n4 < n1 ∧ n4 < n2 ∧ n4 < n3 := by
    sorry

end smallest_number_is_111111_2_l806_806257


namespace problem1_solution_set_problem2_inequality_l806_806000

theorem problem1_solution_set (x : ℝ) : (-1 < x) ∧ (x < 9) ↔ (|x| + |x - 3| < x + 6) :=
by sorry

theorem problem2_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hn : 9 * x + y = 1) : x + y ≥ 16 * x * y :=
by sorry

end problem1_solution_set_problem2_inequality_l806_806000


namespace light_flashes_count_in_three_quarters_hour_l806_806439

theorem light_flashes_count_in_three_quarters_hour:
  let flashes_every := 6 in
  let minutes_per_hour := 60 in
  let seconds_per_minute := 60 in
  let total_seconds := (3 / 4) * minutes_per_hour * seconds_per_minute in
  let expected_flashes := total_seconds / flashes_every in
  expected_flashes = 450 :=
by
  sorry

end light_flashes_count_in_three_quarters_hour_l806_806439


namespace length_of_chord_AB_l806_806873

section math_problem

-- Define the parametric equations of curve C
def parametric_curve_eq (α : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the polar equation of line l
def polar_eq_line_l (θ ρ : ℝ) : Prop :=
  θ = Real.pi / 4

-- The polar function for the given problem
def polar_eq_curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * ρ * Real.cos θ - 3 = 0

-- The Cartesian equation of line l
def cartesian_eq_line_l (x y : ℝ) : Prop :=
  y = x

-- Define the center and radius of the circle
def center_C : ℝ × ℝ :=
  (1, 0)

def radius_C : ℝ :=
  2

-- Distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Prove the length of chord AB is √14.
theorem length_of_chord_AB {A B : ℝ × ℝ} (hA : cartesian_eq_line_l A.1 A.2) (hB : cartesian_eq_line_l B.1 B.2) :
  distance A B = Real.sqrt 14 := sorry

end math_problem

end length_of_chord_AB_l806_806873


namespace polynomial_sum_l806_806090

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l806_806090


namespace geometric_sequence_sum_l806_806033

theorem geometric_sequence_sum (a : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (ha1 : a 1 = 2) 
  (hq_geom : ∀ n, a (n + 1) = 2 * q^n) 
  (ha_n_geom : ∀ n, a n + 1 = 2 * q^(n-1) + 1) 
  (hq1 : (2*q + 1)^2 = 3*(2*q^2 + 1) → q = 1) 
  : ∑ i in range n, a (i+1) = 2 * n := 
sorry

end geometric_sequence_sum_l806_806033


namespace distance_and_speed_l806_806623

-- Define the conditions given in the problem
def first_car_speed (y : ℕ) := y + 4
def second_car_speed (y : ℕ) := y
def third_car_speed (y : ℕ) := y - 6

def time_relation1 (x : ℕ) (y : ℕ) :=
  x / (first_car_speed y) = x / (second_car_speed y) - 3 / 60

def time_relation2 (x : ℕ) (y : ℕ) :=
  x / (second_car_speed y) = x / (third_car_speed y) - 5 / 60 

-- State the theorem to prove both the distance and the speed of the second car
theorem distance_and_speed : ∃ (x y : ℕ), 
  time_relation1 x y ∧ 
  time_relation2 x y ∧ 
  x = 120 ∧ 
  y = 96 :=
by
  sorry

end distance_and_speed_l806_806623


namespace num_integers_satisfying_inequalities_l806_806012

-- Define the inequalities that y must satisfy
def inequality1 (y : ℤ) : Prop := -3 * y ≥ y + 7
def inequality2 (y : ℤ) : Prop := -2 * y ≤ 12
def inequality3 (y : ℤ) : Prop := -4 * y ≥ 2 * y + 17

-- Define a predicate that y satisfies all three inequalities
def satisfies_all (y : ℤ) : Prop :=
  inequality1 y ∧ inequality2 y ∧ inequality3 y

-- Prove that the number of integers satisfying all three inequalities is 4
theorem num_integers_satisfying_inequalities : 
  card {y : ℕ | satisfies_all y} = 4 :=
sorry

end num_integers_satisfying_inequalities_l806_806012


namespace sum_of_percentages_l806_806187

theorem sum_of_percentages :
  let percent1 := 7.35 / 100
  let percent2 := 13.6 / 100
  let percent3 := 21.29 / 100
  let num1 := 12658
  let num2 := 18472
  let num3 := 29345
  let result := percent1 * num1 + percent2 * num2 + percent3 * num3
  result = 9689.9355 :=
by
  sorry

end sum_of_percentages_l806_806187


namespace stamps_per_ounce_l806_806474

def weight_per_piece_of_paper : ℚ := 1 / 5
def number_of_pieces_of_paper : ℕ := 8
def weight_of_envelope : ℚ := 2 / 5
def total_stamps : ℕ := 2

theorem stamps_per_ounce 
  (weight_per_piece_of_paper = 1 / 5)
  (number_of_pieces_of_paper = 8)
  (weight_of_envelope = 2 / 5)
  (total_stamps = 2) : 
  (total_stamps / ((number_of_pieces_of_paper * weight_per_piece_of_paper) + weight_of_envelope)) = 1 :=
by
  sorry

end stamps_per_ounce_l806_806474


namespace min_a2_b2_l806_806361

theorem min_a2_b2 :
  ∃ a b : ℝ, 
  (∀ x : ℝ, f(x) = log ((e*x) / (e-x)))
  → (∑ k in finset.range (2012), f((k + 1) * e / 2013) = 503 * (a + b))
  → a^2 + b^2 = 8 :=
begin
  -- The proof will go here
  sorry
end

end min_a2_b2_l806_806361


namespace total_people_waiting_in_line_l806_806597

-- Conditions
def people_fitting_in_ferris_wheel : ℕ := 56
def people_not_getting_on : ℕ := 36

-- Definition: Number of people waiting in line
def number_of_people_waiting_in_line : ℕ := people_fitting_in_ferris_wheel + people_not_getting_on

-- Theorem to prove
theorem total_people_waiting_in_line : number_of_people_waiting_in_line = 92 := by
  -- This is a placeholder for the actual proof
  sorry

end total_people_waiting_in_line_l806_806597


namespace inequality_solution_sum_l806_806133

noncomputable def sum_of_solutions : ℤ := 
  let interval := {-25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, 19}
  interval.sum

theorem inequality_solution_sum :
  (sum_of_solutions = -285) :=
begin
  -- Given Conditions
  have quad_cond1: ∀ x : ℝ, x ∈ interval → (x^2 - x - 56 ≥ 0) := sorry,
  have quad_cond2: ∀ x : ℝ, x ∈ interval → (x^2 - 25x + 136 ≥ 0) := sorry,
  have domain_cond1: ∀ x : ℝ, x ∈ interval → (x - 8 > 0) := sorry,
  have domain_cond2: ∀ x : ℝ, x ∈ interval → (x + 7 ≥ 0) := sorry,
  
  -- Translate the Inequality Condition
  have ineq_condition: ∀ x : ℝ, x ∈ interval → 
    √(x^2 - x - 56) - √(x^2 - 25x + 136) < 8 * √((x+7) / (x-8))
    := sorry,

  -- Proof of the sum of the valid integer solutions
  sorry
end

end inequality_solution_sum_l806_806133


namespace count_perfect_squares_between_50_and_300_l806_806425

theorem count_perfect_squares_between_50_and_300 : 
  ∃ n, number_of_perfect_squares 50 300 = n ∧ n = 10 := 
sorry

end count_perfect_squares_between_50_and_300_l806_806425


namespace range_f_x_in_A_range_x0_in_A_f_f_x_in_A_l806_806372

noncomputable def A : Set ℝ := Set.Ico 0 (1 / 2)
noncomputable def B : Set ℝ := Set.Icc (1 / 2) 1

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ A then x + (1 / 2)
else real.log (2 - x) / real.log 2

theorem range_f_x_in_A : ∀ x ∈ Set.Ioc (2 - real.sqrt 2) 1, f x ∈ A := 
by {
  intro x,
  intro hx,
  sorry
}

theorem range_x0_in_A_f_f_x_in_A : 
  ∀ x ∈ Set.Ioo ((3 / 2) - real.sqrt 2) (1 / 2), x ∈ A ∧ (f (f x)) ∈ A := 
by {
  intro x,
  intro hx,
  sorry
}

end range_f_x_in_A_range_x0_in_A_f_f_x_in_A_l806_806372


namespace vector_parallel_x_value_l806_806010

theorem vector_parallel_x_value (x : ℝ) :
  let a := (1 : ℝ, 2 : ℝ),
      b := (x, -4 : ℝ)
  in a.1 / b.1 = a.2 / b.2 → x = -2 :=
by
  intros a b h,
  sorry

end vector_parallel_x_value_l806_806010


namespace discount_equivalence_l806_806685

theorem discount_equivalence :
  ∀ (p d1 d2 : ℝ) (d : ℝ),
    p = 800 →
    d1 = 0.15 →
    d2 = 0.10 →
    p * (1 - d1) * (1 - d2) = p * (1 - d) →
    d = 0.235 := by
  intros p d1 d2 d hp hd1 hd2 heq
  sorry

end discount_equivalence_l806_806685


namespace no_integer_solutions_l806_806215

theorem no_integer_solutions (x y z : ℤ) (h : 2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) (hx : x ≠ 0) : false :=
sorry

end no_integer_solutions_l806_806215


namespace channel_top_width_l806_806991

theorem channel_top_width
  (bottom_width : ℝ)
  (area : ℝ)
  (height : ℝ)
  (trapezium_area : ∀ (a b h : ℝ), (a + b) / 2 * h = area)
  (h_conditions : bottom_width = 8 ∧ area = 770 ∧ height = 70) :
  ∃ w : ℝ, w = 14 := 
by
  obtain ⟨hb, ha, hh⟩ := h_conditions
  use 14
  sorry

end channel_top_width_l806_806991


namespace B_pow_150_eq_I_l806_806062

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end B_pow_150_eq_I_l806_806062


namespace exists_n_consecutive_numbers_l806_806283

theorem exists_n_consecutive_numbers:
  ∃ n : ℕ, n % 5 = 0 ∧ (n + 1) % 4 = 0 ∧ (n + 2) % 3 = 0 := sorry

end exists_n_consecutive_numbers_l806_806283


namespace find_a_l806_806596

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2 * x + a

theorem find_a (a : ℝ) (h : f'(a) 2 = 4) : a = 0 :=
by
  sorry

end find_a_l806_806596


namespace determine_fruit_weights_l806_806922

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806922


namespace four_digit_integers_with_5_or_7_l806_806383

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l806_806383


namespace perpendicular_bisector_of_intersection_points_l806_806806

theorem perpendicular_bisector_of_intersection_points 
  : let C₁ := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 13}
    let C₂ := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}
    let A, B : ℝ × ℝ := -- Assume A and B are points of intersection of C₁ and C₂
    ∃ A B : ℝ × ℝ, A ∈ C₁ ∧ A ∈ C₂ ∧ B ∈ C₁ ∧ B ∈ C₂ ∧ 
    ∀ x y : ℝ, 
        (((x, y) = A) ∨ ((x, y) = B) → 
        (x + 3 * y = 0)) → 
    (perpendicular_bisector_eqn : ℝ → ℝ → Prop) := 
      perpendicular_bisector_eqn := (fun x y => 3 * x - y - 9 = 0)

end perpendicular_bisector_of_intersection_points_l806_806806


namespace basketball_team_points_l806_806607

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end basketball_team_points_l806_806607


namespace probability_of_at_least_one_white_is_D_l806_806314

-- Define the bag and the events

noncomputable def bag := ['red, 'red, 'red, 'white, 'white]

-- Define the event of drawing 3 balls
def draw_3_balls (s : List (String × ℕ)) : Set (List String) := 
  {l | l.length = 3 ∧ ∀ x, x ∈ l → (s.count x) ≥ (l.count x)}

-- Define the total possible combinations
def total_combinations := @Set.univ (List String)

-- Define the event that we do not draw 3 red balls
def not_all_red : Set (List String) := {l | ¬(l.count 'red = 3)}

-- Probability that at least one of 3 drawn balls is white
def probability_at_least_one_white : ℝ := 
  1 - (draw_3_balls total_combinations ∩ not_all_red).size.val.fst / total_combinations.size.val.fst

theorem probability_of_at_least_one_white_is_D :
  probability_at_least_one_white = sorry := sorry

end probability_of_at_least_one_white_is_D_l806_806314


namespace sum_of_max_min_l806_806175

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log2 (x + 1)

theorem sum_of_max_min :
  let interval_min := 0
  let interval_max := 1
  let f_min := f interval_min
  let f_max := f interval_max
  f_min + f_max = 4 := by
  let interval_min := 0
  let interval_max := 1
  let f_min := f interval_min
  let f_max := f interval_max
  exact (show f_min + f_max = 4 from sorry)

end sum_of_max_min_l806_806175


namespace part_a_part_b_l806_806553
noncomputable theory

variables (A B C O A1 B1 C1 : Type) [triangle ABC : angle (A, B, C)]
variables (on_AB : B1 ∈ segment A B) (on_BC : A1 ∈ segment B C) (on_CA : C1 ∈ segment C A)
variables (intersection : point_of_intersection (CC1 : line C C1) (AA1 : line A A1) (BB1 : line B B1) = O)

-- Part (a)
theorem part_a : 
  (CO / O C1) = (CA1 / A1 B) + (CB1 / B1 A) :=
sorry

-- Part (b)
theorem part_b : 
  (AO / OA1) * (BO / OB1) * (CO / OC1) = (AO / OA1) + (BO / OB1) + (CO / OC1) + 2 ∧ (AO / OA1) * (BO / OB1) * (CO / OC1) ≥ 8 :=
sorry

end part_a_part_b_l806_806553


namespace calculate_fraction_of_time_l806_806223

theorem calculate_fraction_of_time :
  ∀ (distance time : ℝ) (speed_required : ℝ),
    time = 6 ∧ distance = 469 ∧ speed_required = 52.111111111111114 →
    distance / speed_required = (2 / 3) * time := 
by
  -- Define the variables for the given conditions
  intros distance time speed_required h,
  -- Extract the conditions from the hypothesis h
  cases h with h1 h_rest,
  cases h_rest with h2 h3,
  -- Use the provided conditions to form the statement
  rw h1 at *,
  rw h2 at *,
  rw h3,
  -- Apply the provided solutions
  sorry

end calculate_fraction_of_time_l806_806223


namespace problem1_problem2_l806_806363

-- Definitions for problem (1)
def f (x: ℝ) : ℝ := Real.exp x
def g (m x: ℝ) : ℝ := m * x
def h (m x: ℝ) : ℝ := f x - g m x

theorem problem1 (m : ℝ) : -Real.exp (-1) ≤ m ∧ m < Real.exp :=
  sorry

-- Definitions for problem (2)
def r (m x: ℝ) : ℝ := (m / Real.exp x) + (4 * x / (x + 4))

theorem problem2 (m : ℝ) (hm: m > 0) (x: ℝ) (hx: x ≥ 0) : r m x ≥ 1 :=
  sorry

end problem1_problem2_l806_806363


namespace cut_7x7_square_to_9_rectangles_l806_806724

-- Define the dimensions of the square and the constraint on rectangles
def square_side : ℕ := 7
def max_side : ℕ := 7
def rectangles_count : ℕ := 9

-- Prove that a 7 x 7 square can be cut into 9 rectangles to form any rectangle with sides not exceeding 7
theorem cut_7x7_square_to_9_rectangles :
  ∃ (rectangles : list (ℕ × ℕ)), 
  rectangles.length = rectangles_count ∧ 
  (∀ (h w : ℕ), h ≤ max_side → w ≤ max_side → 
   ∃ (subset : list (ℕ × ℕ)), 
   subset ⊆ rectangles ∧ 
   list.sum (subset.map prod.fst) = h ∧ 
   list.sum (subset.map prod.snd) = w) :=
sorry

end cut_7x7_square_to_9_rectangles_l806_806724


namespace derivative_of_y_l806_806296

noncomputable def y (a x : ℝ) : ℝ :=
  (1 / (2 * a * sqrt (1 + a^2))) * log ((a + sqrt (1 + a^2) * tanh x) / (a - sqrt (1 + a^2) * tanh x))

theorem derivative_of_y (a x : ℝ) :
  deriv (λ x, y a x) x = (1 / (a^2 * cosh x^2 + (1 + a^2) * sinh x^2)) :=
sorry

end derivative_of_y_l806_806296


namespace inequality_solution_sum_l806_806134

noncomputable def sum_of_solutions : ℤ := 
  let interval := {-25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, 19}
  interval.sum

theorem inequality_solution_sum :
  (sum_of_solutions = -285) :=
begin
  -- Given Conditions
  have quad_cond1: ∀ x : ℝ, x ∈ interval → (x^2 - x - 56 ≥ 0) := sorry,
  have quad_cond2: ∀ x : ℝ, x ∈ interval → (x^2 - 25x + 136 ≥ 0) := sorry,
  have domain_cond1: ∀ x : ℝ, x ∈ interval → (x - 8 > 0) := sorry,
  have domain_cond2: ∀ x : ℝ, x ∈ interval → (x + 7 ≥ 0) := sorry,
  
  -- Translate the Inequality Condition
  have ineq_condition: ∀ x : ℝ, x ∈ interval → 
    √(x^2 - x - 56) - √(x^2 - 25x + 136) < 8 * √((x+7) / (x-8))
    := sorry,

  -- Proof of the sum of the valid integer solutions
  sorry
end

end inequality_solution_sum_l806_806134


namespace circle_center_coordinates_l806_806329

theorem circle_center_coordinates (a b : ℝ) :
  (∃ a b : ℝ, (abs (a + 1) = abs (b - 4) ∧ sqrt ((a - 1)^2 + b^2) = abs (a + 1)) 
  ∧ (a + b - 3 = 0 ∧ b^2 = 4 * a)) 
  ∨ (∃ a b : ℝ, (abs (a + 1) = abs (b - 4) ∧ sqrt ((a - 1)^2 + b^2) = abs (a + 1)) 
  ∧ (a - b + 5 = 0 ∧ b^2 = 4 * a)) → 
  (a = 1 ∧ b = 2) ∨ (a = 9 ∧ b = -6) := 
sorry

end circle_center_coordinates_l806_806329


namespace increasing_interval_of_f_l806_806153

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (2 * x ^ 2 - 3 * x + 1)

theorem increasing_interval_of_f :
  ∃ a b : ℝ, a < b ∧ Ioo a b = (-∞, 3 / 4) ∧ (∀ x1 x2 : ℝ, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 < f x2) := sorry

end increasing_interval_of_f_l806_806153


namespace complex_number_z_l806_806020

noncomputable def solve_for_z (z : ℂ) : Prop :=
  (z̅ / (1 - I) = I) → (z = 1 - I)

theorem complex_number_z (z : ℂ) : solve_for_z z :=
by sorry

end complex_number_z_l806_806020


namespace sophist_statements_correct_l806_806968

-- Definitions based on conditions
def num_knights : ℕ := 40
def num_liars : ℕ := 25

-- Statements made by the sophist
def sophist_statement1 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_knights = 40
def sophist_statement2 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_liars + 1 = 26

-- Theorem to be proved
theorem sophist_statements_correct :
  sophist_statement1 ∧ sophist_statement2 :=
by
  -- Placeholder for the actual proof
  sorry

end sophist_statements_correct_l806_806968


namespace tangent_x_axis_extreme_values_local_minimum_l806_806797

-- Define the function g(x)
def g (x : ℝ) (a : ℝ) : ℝ := x^3 + 3*a*x - 2

-- Define the derivative of g(x)
def g' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 3*a

-- Part (1): Prove that if a = -1, the x-axis is tangent to the curve y = g(x)
theorem tangent_x_axis (a : ℝ) : (∃ m : ℝ, g' m a = 0 ∧ g m a = 0) → a = -1 := 
sorry

-- Part (2): Prove the conditions for extreme values and sum of maximum and minimum values
theorem extreme_values (a : ℝ) : 
  (a < 0 → ∃ m M : ℝ, g' m a = 0 ∧ g' M a = 0 ∧ g m a + g M a = -4) ∧ 
  (a ≥ 0 → ∀ x : ℝ, g' x a ≥ 0) := 
sorry

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := 
  (1 / 3 * (3*x^2 + 3*a) - a*x) * real.exp x - x^2

-- Define the derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 
  x * ((x + 2 - 2 / real.exp x) * real.exp x - 2)

-- Part (3): Prove the range of a
theorem local_minimum (a : ℝ) : 
  (∀ x : ℝ, x * ((x + 2 - 2 / real.exp x) * real.exp x - 2) = 0 → a < 0) := 
sorry

end tangent_x_axis_extreme_values_local_minimum_l806_806797


namespace digits_with_five_or_seven_is_5416_l806_806407

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l806_806407


namespace lines_intersect_at_single_point_l806_806734

theorem lines_intersect_at_single_point (m : ℚ)
    (h1 : ∃ x y : ℚ, y = 4 * x - 8 ∧ y = -3 * x + 9)
    (h2 : ∀ x y : ℚ, (y = 4 * x - 8 ∧ y = -3 * x + 9) → (y = 2 * x + m)) :
    m = -22/7 := by
  sorry

end lines_intersect_at_single_point_l806_806734


namespace count_primes_5p2p1_minus_1_perfect_square_l806_806814

-- Define the predicate for a prime number
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate for perfect square
def is_perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

-- The main theorem statement
theorem count_primes_5p2p1_minus_1_perfect_square :
  (∀ p : ℕ, is_prime p → is_perfect_square (5 * p * (2^(p + 1) - 1))) → ∃! p : ℕ, is_prime p ∧ is_perfect_square (5 * p * (2^(p + 1) - 1)) :=
sorry

end count_primes_5p2p1_minus_1_perfect_square_l806_806814


namespace halfway_between_fractions_l806_806749

theorem halfway_between_fractions : ( (1/8 : ℚ) + (1/3 : ℚ) ) / 2 = 11 / 48 :=
by
  sorry

end halfway_between_fractions_l806_806749


namespace four_digit_integers_with_5_or_7_l806_806380

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l806_806380


namespace emery_reading_days_l806_806291

theorem emery_reading_days (S E : ℕ) (h1 : E = S / 5) (h2 : (E + S) / 2 = 60) : E = 20 := by
  sorry

end emery_reading_days_l806_806291


namespace simplify_expression_l806_806648

theorem simplify_expression :
  -3 - (+6) - (-5) + (-2) = -3 - 6 + 5 - 2 :=
  sorry

end simplify_expression_l806_806648


namespace interval_contains_root_l806_806996

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 3 / x

theorem interval_contains_root :
  continuous_on f (Ioi 0) ∧ f e < 0 ∧ f 3 > 0 → ∃ x ∈ Ioo e 3, f x = 0 :=
by
  intros hc
  sorry

end interval_contains_root_l806_806996


namespace determine_fruit_weights_l806_806920

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806920


namespace area_enclosed_by_curve_is_correct_l806_806990

noncomputable def enclosed_area_of_curve 
  (n : ℕ) -- 9 arcs
  (arc_length : ℝ) -- arc_length is (5 * π / 6)
  (hexagon_side : ℝ) -- side length of hexagon is 3
  : ℝ :=
  let r := (2 * arc_length) / (2 * π) in -- calculate radius r
  let hexagon_area := (3 * Real.sqrt 3 * hexagon_side^2) / 2 in  -- hexagon area
  let sector_area := (arc_length * π * r^2) / (2 * π) in  -- area of one sector
  let total_sector_area := n * sector_area in
  let net_enclosed_area := hexagon_area + total_sector_area in
  net_enclosed_area

theorem area_enclosed_by_curve_is_correct :
  enclosed_area_of_curve 9 (5 * Real.pi / 6) 3 = 
  13.5 * Real.sqrt 3 + 375 * Real.pi / 8 :=
sorry

end area_enclosed_by_curve_is_correct_l806_806990


namespace sqrt3_minus_2_mul_sqrt3_minus_2_l806_806713

theorem sqrt3_minus_2_mul_sqrt3_minus_2 :
  (real.sqrt 3 - 2) * (real.sqrt 3 - 2) = 7 - 4 * real.sqrt 3 :=
by
  sorry

end sqrt3_minus_2_mul_sqrt3_minus_2_l806_806713


namespace susan_should_turn_over_a_5_l806_806980

noncomputable def vowel (c : char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

noncomputable def prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def card_identifiers : list char := ['A', 'B', 'S', '8', '5', '7']

-- Given Thomas's claim and Susan's options
theorem susan_should_turn_over_a_5
    (cards : card_identifiers) 
    (visible_side : char → char)
    (C : ∀ c : char, (vowel c → prime (visible_side c))) 
    : { visible_side 'A', visible_side '5' } 
=→ { A, 5 } :=
sorry

end susan_should_turn_over_a_5_l806_806980


namespace monotonic_increase_intervals_and_range_l806_806357

noncomputable def f (x : ℝ) (ω : ℝ) := 
  (2 * (Real.sqrt 3) * Real.cos (ω * x) + Real.sin (ω * x)) * (Real.sin (ω * x))
  - (Real.sin ((π / 2) + ω * x))^2

theorem monotonic_increase_intervals_and_range {ω : ℝ} (hω : ω > 0) :
  (
    ∀ k : ℤ, 
    ∃ a b : ℝ, 
    a = - (π / 6) + k * π 
    ∧ b = π / 3 + k * π 
    ∧ ∀ x : ℝ, (a ≤ x ∧ x ≤ b) → Function.monotone f x ω 
  ) 
  ∧ 
  (
    ∀ x : ℝ, 
    0 ≤ x ∧ x ≤ π / 2 → 
    -1 ≤ f x 1 ∧ f x 1 ≤ 2
  ) 
sorry

end monotonic_increase_intervals_and_range_l806_806357


namespace perpendicular_tangents_at_x0_l806_806355

noncomputable def curve (x: ℝ) : ℝ := abs x / Real.exp x

noncomputable def derivative (x: ℝ) : ℝ :=
if x > 0 then 
  (1 - x) / Real.exp x 
else 
  (x - 1) / Real.exp x

theorem perpendicular_tangents_at_x0 (x0 : ℝ) (h0 : x0 > 0) (hm : x0 ∈ set.Ioo (2 / 4) (3 / 4)) :
  let m := 2 in
  let slope_neg1 := derivative (-1) in
  let slope_x0 := derivative x0 in
  slope_neg1 * slope_x0 = -1 →
  m = 2 :=
by sorry

end perpendicular_tangents_at_x0_l806_806355


namespace arithmetic_sequence_a5_a7_l806_806773

variable {α : Type*} [AddGroup α] [LinearOrder α]
variable (a : ℕ → α)

def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a5_a7 :
  ∀ d a,
  arithmetic_sequence a d →
  (a 2 + a 4 = 4) →
  (a 3 + a 5 = 10) →
  (a 5 + a 7 = 22) :=
begin
  -- Proof goes here
  sorry,
end

end arithmetic_sequence_a5_a7_l806_806773


namespace number_2018_location_l806_806704

-- Define the odd square pattern as starting positions of rows
def odd_square (k : ℕ) : ℕ := (2 * k - 1) ^ 2

-- Define the conditions in terms of numbers in each row
def start_of_row (n : ℕ) : ℕ := (2 * n - 1) ^ 2 + 1

def number_at_row_column (n m : ℕ) :=
  start_of_row n + (m - 1)

theorem number_2018_location :
  number_at_row_column 44 82 = 2018 :=
by
  sorry

end number_2018_location_l806_806704


namespace range_of_a_l806_806027

theorem range_of_a (a : ℝ) : 
  (∀ P Q : ℝ × ℝ, P ≠ Q ∧ P.snd = a * P.fst ^ 2 - 1 ∧ Q.snd = a * Q.fst ^ 2 - 1 ∧ 
  P.fst + P.snd = -(Q.fst + Q.snd)) →
  a > 3 / 4 :=
by
  sorry

end range_of_a_l806_806027


namespace proof_problem_l806_806827

noncomputable def a : ℝ := Real.log 8
noncomputable def b : ℝ := Real.log 27

theorem proof_problem : 5^(a/b) + 2^(b/a) = 8 := by
  sorry

end proof_problem_l806_806827


namespace last_two_digits_of_sum_l806_806268

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum :
  last_two_digits (factorial 4 + factorial 5 + factorial 6 + factorial 7 + factorial 8 + factorial 9) = 4 :=
by
  sorry

end last_two_digits_of_sum_l806_806268


namespace smallest_possible_value_of_other_integer_l806_806639

theorem smallest_possible_value_of_other_integer 
  (n : ℕ) (hn_pos : 0 < n) (h_eq : (Nat.lcm 75 n) / (Nat.gcd 75 n) = 45) : n = 135 :=
by sorry

end smallest_possible_value_of_other_integer_l806_806639


namespace nonempty_disjoint_subsets_with_same_properties_l806_806774

theorem nonempty_disjoint_subsets_with_same_properties {n : ℕ} (h : n = 2000)
    (ns : Fin (n + 1) → ℕ)
    (h_ordered : ∀ i j : Fin (n + 1), i < j → ns i < ns j)
    (h_bound : ∀ i : Fin (n + 1), ns i < 10^100) :
  ∃ (A B : Finset (Fin (n + 1))), A.nonempty ∧ B.nonempty ∧ A.disjoint B ∧ 
  |A| = |B| ∧ A.sum (λ i, ns i) = B.sum (λ i, ns i) ∧
  A.sum (λ i, (ns i)^2) = B.sum (λ i, (ns i)^2) :=
by
  sorry

end nonempty_disjoint_subsets_with_same_properties_l806_806774


namespace monotonic_increasing_range_l806_806833

theorem monotonic_increasing_range (k : ℝ) :
  (∀ x : ℝ, x > 1/2 → deriv (λ x, k * x - log x) x ≥ 0) → k ≥ 2 :=
by {
  sorry
}

end monotonic_increasing_range_l806_806833


namespace find_even_increasing_l806_806699

theorem find_even_increasing (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 < x → x < y → 0 < y → f x < f y) ↔
  f = (fun x => 3 * x^2 - 1) ∨ f = (fun x => 2^|x|) :=
by
  sorry

end find_even_increasing_l806_806699


namespace parallelogram_APQR_l806_806110

noncomputable def isosceles_triangle (A B P C Q R : Point) := 
  ∃ (AP_eq_PB : dist A P = dist P B)
    (AQ_eq_QC : dist A Q = dist Q C)
    (BR_eq_RC : dist B R = dist R C), 
    true

theorem parallelogram_APQR 
  (A B C P Q R : Point) 
  (isosceles_ABC_PQ : isosceles_triangle A B P C Q R) 
  (AP_eq_BQ : dist A P = dist Q C) 
  (AQ_eq_RB : dist A Q = dist R B) 
  (BR_eq_PA : dist B R = dist P A) 
  (QC_eq_RP : dist Q C = dist R P)
:
  is_parallelogram A P Q R :=
by 
  sorry

end parallelogram_APQR_l806_806110


namespace domain_and_range_of_f_range_of_k_l806_806786

-- Define the function f
def f (x : ℝ) := (1 / 2) ^ (real.sqrt (-x^2 - 2 * x))

-- Define the function g
def g (k x : ℝ) := k + real.log x / real.log 2

-- Statement to prove the domain of f is A and range of f is B
theorem domain_and_range_of_f :
  let A := set.Icc (-2 : ℝ) 0,
      B := set.Ioc (1 / 2 : ℝ) 1
  in (∀ x, ¬ (-x^2 - 2 * x < 0) -> x ∈ A) ∧ (∀ x, x ∈ A -> f x ∈ B) :=
sorry

-- Statement to prove the range of the real number k
theorem range_of_k (k : ℝ) :
  let A := set.Icc (-2 : ℝ) 0,
      B := set.Ioc (1 / 2 : ℝ) 1
  in (∀ x, x ∈ B -> g k x ∈ A) -> k ≤ 0 :=
sorry

end domain_and_range_of_f_range_of_k_l806_806786


namespace greatest_triangle_area_l806_806038

def rectangle_six_squares_side_length := 4
def midpoint_A := true
def midpoint_B := true
def midpoint_C := true
def midpoint_D := true
def midpoint_E := true
def midpoint_F := true

theorem greatest_triangle_area :
  ∀ (s : ℕ) (P Q R S U V W X Y Z : Type) (triangle_PVU triangle_PXZ triangle_PVX triangle_PYS triangle_PQW : Type),
    (rectangle_six_squares_side_length = s) →
    (midpoint_A ∧ midpoint_B ∧ midpoint_C ∧ midpoint_D ∧ midpoint_E ∧ midpoint_F) →
    s = 4 →
    {triangle_PXZ.area > triangle_PVU.area} →
    {triangle_PXZ.area > triangle_PVX.area} →
    {triangle_PXZ.area > triangle_PYS.area} →
    {triangle_PXZ.area > triangle_PQW.area} →
    triangle_PXZ = argmax {triangle_PVU, triangle_PXZ, triangle_PVX, triangle_PYS, triangle_PQW} :=
by
  sorry

end greatest_triangle_area_l806_806038


namespace cube_4_edge_trips_l806_806602

theorem cube_4_edge_trips (P Q : ℕ) (h : P ≠ Q) 
    (edges : ∀ (X : ℕ), X ∈ [P, Q] → ℕ) : 
    (shortest_trip_length P Q 4 → 
    count_of_4_edge_trips P Q edges = 12):=
sorry

end cube_4_edge_trips_l806_806602


namespace ratio_children_to_women_l806_806185

theorem ratio_children_to_women 
  (total_spectators : ℕ) 
  (men_spectators : ℕ) 
  (children_spectators : ℕ)
  (h_total : total_spectators = 10000)
  (h_men : men_spectators = 7000)
  (h_children : children_spectators = 2500) :
  let women_spectators := total_spectators - men_spectators - children_spectators in
  children_spectators / women_spectators = 5 := 
by
  sorry

end ratio_children_to_women_l806_806185


namespace therapist_charge_difference_l806_806679

theorem therapist_charge_difference :
  ∃ F A : ℝ, F + 4 * A = 350 ∧ F + A = 161 ∧ F - A = 35 :=
by {
  -- Placeholder for the actual proof.
  sorry
}

end therapist_charge_difference_l806_806679


namespace determine_fruit_weights_l806_806924

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806924


namespace jerry_won_games_l806_806477

theorem jerry_won_games 
  (T : ℕ) (K D J : ℕ) 
  (h1 : T = 32) 
  (h2 : K = D + 5) 
  (h3 : D = J + 3) : 
  J = 7 := 
sorry

end jerry_won_games_l806_806477


namespace table_tennis_matches_l806_806671

theorem table_tennis_matches :
  (∃ (n : ℕ), n = 10) →
  (∀ i j : ℕ, i ≠ j → i ∈ Finset.range 10 → j ∈ Finset.range 10 →
  1) →
  (∃ (m : ℕ), m = (10 * 9) / 2 ∧ m = 45) :=
begin
  sorry
end

end table_tennis_matches_l806_806671


namespace find_degree_measure_l806_806746

theorem find_degree_measure :
  ∀ (sin cos : ℕ → ℝ), 
    δ = Real.arccos ((∑ k in Finset.range 6138 \ Finset.range 2537, Real.sin (k : ℝ) * Real.pi / 180) ^ 
                     (∑ j in Finset.range 6121 \ Finset.range 2520, Real.cos (j : ℝ) * Real.pi / 180)) → 
    δ = 73 :=
by
  sorry

end find_degree_measure_l806_806746


namespace balls_in_boxes_l806_806817

theorem balls_in_boxes :
  let balls := 6
  let boxes := 4
  (number_of_ways_to_distribute_balls (balls) (boxes) = 84) :=
by
  sorry

end balls_in_boxes_l806_806817


namespace like_terms_in_set_A_l806_806701

-- Define the monomials as given
def monomial_set_A1 := (1/3) * a^2 * b
def monomial_set_A2 := a^2 * b

def monomial_set_B1 := 3 * x^2 * y
def monomial_set_B2 := 3 * x * y^2

def monomial_set_C1 := a
def monomial_set_C2 := 1

def monomial_set_D1 := 2 * b * c
def monomial_set_D2 := 2 * a * b * c

-- Define the like_term condition: they are like terms if they have the same variables raised to the same powers.
def like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  (∀ v : ℕ, m1 v = m2 v)

-- Define the variables and their powers for each set
def vars_powers_A1 (v : ℕ) : ℕ := if v = 0 then 2 else if v = 1 then 1 else 0
def vars_powers_A2 (v : ℕ) : ℕ := if v = 0 then 2 else if v = 1 then 1 else 0

def vars_powers_B1 (v : ℕ) : ℕ := if v = 0 then 2 else if v = 1 then 1 else 0
def vars_powers_B2 (v : ℕ) : ℕ := if v = 0 then 1 else if v = 1 then 2 else 0

def vars_powers_C1 (v : ℕ) : ℕ := if v = 0 then 1 else 0
def vars_powers_C2 (v : ℕ) : ℕ := 0

def vars_powers_D1 (v : ℕ) : ℕ := if v = 1 then 1 else if v = 2 then 1 else 0
def vars_powers_D2 (v : ℕ) : ℕ := if v = 0 then 1 else if v = 1 then 1 else if v = 2 then 1 else 0

-- Define the proof statement
theorem like_terms_in_set_A :
  like_terms vars_powers_A1 vars_powers_A2 ∧
  ¬ like_terms vars_powers_B1 vars_powers_B2 ∧
  ¬ like_terms vars_powers_C1 vars_powers_C2 ∧
  ¬ like_terms vars_powers_D1 vars_powers_D2 :=
by sorry

end like_terms_in_set_A_l806_806701


namespace remainder_b33_div_35_l806_806081

-- Define the sequence bn
def b_n (n : ℕ) : ℕ :=
  (List.range (2 * n + 1)).filter (λ x, x % 2 = 0) |> (λ l, l.map (λ x, toString x).foldl (++) "") |> String.toNat

theorem remainder_b33_div_35 : (b_n 33) % 35 = 21 :=
  sorry

end remainder_b33_div_35_l806_806081


namespace sales_weekly_l806_806890

def john_sales_per_week
  (houses_per_day : Nat)
  (buy_percentage : ℚ)
  (ratio : ℚ)
  (price_set1 : Nat)
  (price_set2 : Nat)
  (working_days : Nat) : Nat := 
  let buyers_per_day := houses_per_day * buy_percentage
  let buyers_set1 := buyers_per_day * ratio
  let buyers_set2 :=  buyers_per_day * ratio
  let sales_per_day := buyers_set1 * price_set1 + buyers_set2 * price_set2
  sales_per_day * working_days

theorem sales_weekly
  (houses_per_day : Nat := 50)
  (buy_percentage : ℚ := 0.2)
  (ratio : ℚ := 0.5)
  (price_set1 : Nat := 50)
  (price_set2 : Nat := 150)
  (working_days : Nat := 5) : 
  john_sales_per_week houses_per_day buy_percentage ratio price_set1 price_set2 working_days = 5000 := 
  by
    sorry

end sales_weekly_l806_806890


namespace cards_per_pack_l806_806571

-- Definitions from the problem conditions
def packs := 60
def cards_per_page := 10
def pages_needed := 42

-- Theorem statement for the mathematically equivalent proof problem
theorem cards_per_pack : (pages_needed * cards_per_page) / packs = 7 :=
by sorry

end cards_per_pack_l806_806571


namespace fruit_weights_correct_l806_806946

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806946


namespace magnitude_z_plus_1_l806_806021

theorem magnitude_z_plus_1 (z : ℂ) (h : z / (2 - I) = 2 * I) : |z + 1| = Real.sqrt 17 :=
sorry

end magnitude_z_plus_1_l806_806021


namespace perimeter_PFQ_eq_48_l806_806345

open Real

def hyperbola_C (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def left_focus : ℝ × ℝ := (-5, 0)

def point_A : ℝ × ℝ := (5, 0)

def points_on_right_branch (P Q : ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, hyperbola_C x1 y1 ∧ x1 > 0 ∧ hyperbola_C x2 y2 ∧ x2 > 0 ∧ P = (x1, y1) ∧ Q = (x2, y2)

def length_PQ (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P;
  let (x2, y2) := Q;
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def imaginary_axis_length : ℝ := 8

def h1 (P Q : ℝ × ℝ) (hPQR: points_on_right_branch P Q) : length_PQ P Q = 8 :=
  by
    sorry

theorem perimeter_PFQ_eq_48 (P Q : ℝ × ℝ) (hPQR: points_on_right_branch P Q) (hPQ_length: length_PQ P Q = 8) :
  let (xp, yp) := P;
  let (xq, yq) := Q;
  let focus_distance := 5;
  sqrt ((xp + 5)^2 + yp^2) + sqrt ((xq + 5)^2 + yq^2) + length_PQ P Q = 48 :=
  by
    sorry

end perimeter_PFQ_eq_48_l806_806345


namespace sum_log_expr_value_l806_806719

noncomputable def sum_log_expr : ℝ :=
  ∑ k in (finset.range 49).map (λ i, i + 2),
    2 * log 3 (1 + 1 / k) * log k 3 * log (k + 1) 3

theorem sum_log_expr_value : sum_log_expr = 0.2608 :=
by
  sorry

end sum_log_expr_value_l806_806719


namespace seating_arrangements_for_students_l806_806183

noncomputable def permutation_count (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

theorem seating_arrangements_for_students :
  permutation_count 4 3 = 24 :=
by
  simp [permutation_count, Nat.factorial]
  norm_num
  sorry

end seating_arrangements_for_students_l806_806183


namespace perfect_squares_between_50_and_300_l806_806430

theorem perfect_squares_between_50_and_300 : 
  ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 50 < x ∧ x < 300 → ¬ is_square x ∨ (∃ k : ℕ, x = k^2 ∧ 8 ≤ k ∧ k ≤ 17)) :=
begin
  sorry
end

end perfect_squares_between_50_and_300_l806_806430


namespace sum_of_perpendiculars_constant_l806_806127

-- Definition of a regular pentagon centered at the origin with circumradius R
structure Pentagon :=
  (R : ℝ)

-- Definition of a point inside the pentagon
structure PointInsidePentagon (P : Pentagon) :=
  (x : ℝ)
  (y : ℝ)
  (inside : x^2 + y^2 < P.R^2)

-- Statement of the proof problem
theorem sum_of_perpendiculars_constant (P : Pentagon) (K : PointInsidePentagon P) : 
  ∃ C, ∀ (K : PointInsidePentagon P), 
    sum_of_perpendiculars_to_sides K = C :=
sorry

end sum_of_perpendiculars_constant_l806_806127


namespace smallest_positive_angle_solution_l806_806753

theorem smallest_positive_angle_solution (x : ℝ) (hx : 0 < x ∧ x < 90) (h : sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x)) : x = 10 :=
by
  sorry

end smallest_positive_angle_solution_l806_806753


namespace total_snake_owners_l806_806181

theorem total_snake_owners (total_people : ℕ)
  (only_dogs only_cats only_snakes both_dogs_cats both_cats_snakes both_dogs_snakes all_three : ℕ) :
  total_people = 120 →
  only_dogs = 30 →
  only_cats = 25 →
  only_snakes = 12 →
  both_dogs_cats = 15 →
  both_cats_snakes = 10 →
  both_dogs_snakes = 8 →
  all_three = 5 →
  only_snakes + both_cats_snakes + both_dogs_snakes + all_three = 35 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h4, h6, h7, h8]
  exact rfl

end total_snake_owners_l806_806181


namespace greatest_pq_plus_r_l806_806297

theorem greatest_pq_plus_r (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h : p * q + q * r + r * p = 2016) : 
  pq + r ≤ 1008 :=
sorry

end greatest_pq_plus_r_l806_806297


namespace least_positive_t_of_arcsin_sin_ap_l806_806274

theorem least_positive_t_of_arcsin_sin_ap (α : ℝ) (t : ℕ) : 
  0 < α ∧ α < (π / 2) → 
  let term1 := Real.arcsin (Real.sin α),
      term2 := Real.arcsin (Real.sin (3 * α)),
      term3 := Real.arcsin (Real.cos (5 * α)),
      term4 := Real.arcsin (Real.sin (t * α)) 
  in 
  (term1 + (term2 - term1) = term2) ∧ 
  (term2 + (term3 - term2) = term3) ∧ 
  (term3 + (term4 - term3) = term4) → 
  t = 7 :=
by
  sorry

end least_positive_t_of_arcsin_sin_ap_l806_806274


namespace X_lies_on_altitude_BH_l806_806502

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806502


namespace X_on_altitude_BH_l806_806484

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806484


namespace X_on_altitude_BH_l806_806506

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806506


namespace zoo_animal_arrangement_l806_806586

theorem zoo_animal_arrangement : 
  let lions : ℕ := 3
  let zebras : ℕ := 4
  let monkeys : ℕ := 6
  ∀ (n : ℕ),  n = lions + zebras + monkeys → lions! * zebras! * monkeys! * 3! = 622080 :=
by
  intros n h
  have h_lions : lions = 3 := rfl
  have h_zebras : zebras = 4 := rfl
  have h_monkeys : monkeys = 6 := rfl
  simp [h_lions, h_zebras, h_monkeys]
  sorry

end zoo_animal_arrangement_l806_806586


namespace num_four_digit_with_5_or_7_l806_806392

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l806_806392


namespace number_of_members_l806_806102

-- Define the conditions
def knee_pad_cost : ℕ := 6
def jersey_cost : ℕ := knee_pad_cost + 7
def wristband_cost : ℕ := jersey_cost + 3
def cost_per_member : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)
def total_expenditure : ℕ := 4080

-- Prove the number of members in the club
theorem number_of_members (h1 : knee_pad_cost = 6)
                          (h2 : jersey_cost = 13)
                          (h3 : wristband_cost = 16)
                          (h4 : cost_per_member = 70)
                          (h5 : total_expenditure = 4080) :
                          total_expenditure / cost_per_member = 58 := 
by 
  sorry

end number_of_members_l806_806102


namespace can_cut_square_l806_806885

theorem can_cut_square (S a b c : ℝ) : (a^2 + 3*b^2 + 5*c^2 = S^2) →
  (∃ a b c, a ≠ b ∧ a ≠ c ∧ b ≠ c) →
  (∀ x, x ∈ {a, b, c} → ∀ y, y ∈ {a, b, c} → (x = y ∨ x ≠ y)) →
  (∃ (a : ℝ) (b : ℝ) (c : ℝ), 
     a^2 + 3 * b^2 + 5 * c^2 = S^2) :=
by sorry

end can_cut_square_l806_806885


namespace fn_gt_sqrt_np1_l806_806529

noncomputable def f (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), 1 / Real.sqrt (i + 1)

theorem fn_gt_sqrt_np1 (n : ℕ) (hn : n ≥ 3) : f n > Real.sqrt (n + 1) := by
  sorry

end fn_gt_sqrt_np1_l806_806529


namespace paint_cost_contribution_l806_806888

theorem paint_cost_contribution :
  let price_per_gallon := 45 in
  let coverage_per_gallon := 400 in
  let total_area := 1600 in
  let num_coats := 2 in
  let total_paint_required := (total_area / coverage_per_gallon) * num_coats in
  let total_cost := total_paint_required * price_per_gallon in
  let contribution_per_person := total_cost / 2 in
  contribution_per_person = 180 :=
by
  let price_per_gallon := 45
  let coverage_per_gallon := 400
  let total_area := 1600
  let num_coats := 2
  let total_paint_required := (total_area / coverage_per_gallon) * num_coats
  let total_cost := total_paint_required * price_per_gallon
  let contribution_per_person := total_cost / 2
  sorry

end paint_cost_contribution_l806_806888


namespace three_digit_number_count_correct_l806_806011

def number_of_three_digit_numbers_with_repetition (digit_count : ℕ) (positions : ℕ) : ℕ :=
  let choices_for_repeated_digit := 5  -- 5 choices for repeated digit
  let ways_to_place_repeated_digit := 3 -- 3 ways to choose positions
  let choices_for_remaining_digit := 4 -- 4 choices for the remaining digit
  choices_for_repeated_digit * ways_to_place_repeated_digit * choices_for_remaining_digit

theorem three_digit_number_count_correct :
  number_of_three_digit_numbers_with_repetition 5 3 = 60 := 
sorry

end three_digit_number_count_correct_l806_806011


namespace maximum_sum_set_l806_806077

def no_two_disjoint_subsets_have_equal_sums (S : Finset ℕ) : Prop :=
  ∀ (A B : Finset ℕ), A ≠ B ∧ A ∩ B = ∅ → (A.sum id) ≠ (B.sum id)

theorem maximum_sum_set (S : Finset ℕ) (h : ∀ x ∈ S, x ≤ 15) (h_subset_sum : no_two_disjoint_subsets_have_equal_sums S) : S.sum id = 61 :=
sorry

end maximum_sum_set_l806_806077


namespace num_four_digit_with_5_or_7_l806_806394

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l806_806394


namespace number_of_perfect_squares_l806_806416

theorem number_of_perfect_squares (a b : ℕ) (ha : 50 < a^2) (hb : b^2 < 300) :
  ∃ (n : ℕ), a ≤ n ∧ n ≤ b ∧ ∑ i in (finset.range (b - a + 1)).filter (λ n, 50 < n^2 ∧ n^2 < 300), 1 = 10 :=
sorry

end number_of_perfect_squares_l806_806416


namespace speed_of_A_is_24_speed_of_A_is_18_l806_806878

-- Definitions for part 1
def speed_of_B (x : ℝ) := x
def speed_of_A_1 (x : ℝ) := 1.2 * x
def distance_AB := 30 -- kilometers
def distance_B_rides_first := 2 -- kilometers
def time_A_catches_up := 0.5 -- hours

theorem speed_of_A_is_24 (x : ℝ) (h1 : 0.6 * x = 2 + 0.5 * x) : speed_of_A_1 x = 24 := by
  sorry

-- Definitions for part 2
def speed_of_A_2 (y : ℝ) := 1.2 * y
def time_B_rides_first := 1/3 -- hours
def time_difference := 1/3 -- hours

theorem speed_of_A_is_18 (y : ℝ) (h2 : (30 / y) - (30 / (1.2 * y)) = 1/3) : speed_of_A_2 y = 18 := by
  sorry

end speed_of_A_is_24_speed_of_A_is_18_l806_806878


namespace henry_distance_l806_806810

noncomputable def distance_from_starting_point (north1 south east north2 : ℝ) : ℝ :=
  let net_south := (south - (north1 + north2)) in
  real.sqrt (east ^ 2 + net_south ^ 2)

theorem henry_distance :
  let meter_to_feet := 3.28084 in
  let north1 := 15 * meter_to_feet in
  let north2 := 10 * meter_to_feet in
  let south := 15 * meter_to_feet + 50 in
  let east := 40 in
  let expected_distance := 43.54 in
  abs (distance_from_starting_point north1 south east north2 - expected_distance) < 0.01 :=
by
  sorry

end henry_distance_l806_806810


namespace final_result_l806_806482

def N (x : ℝ) : ℝ := 2 * real.sqrt x
def O (x : ℝ) : ℝ := x^3

theorem final_result :
  N (O (N (O (N (O 2))))) = 724 * real.sqrt 2 :=
by
  sorry

end final_result_l806_806482


namespace perfect_squares_between_50_and_300_l806_806429

theorem perfect_squares_between_50_and_300 : 
  ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 50 < x ∧ x < 300 → ¬ is_square x ∨ (∃ k : ℕ, x = k^2 ∧ 8 ≤ k ∧ k ≤ 17)) :=
begin
  sorry
end

end perfect_squares_between_50_and_300_l806_806429


namespace X_on_altitude_BH_l806_806507

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806507


namespace perfect_squares_between_50_and_300_l806_806428

theorem perfect_squares_between_50_and_300 : 
  ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 50 < x ∧ x < 300 → ¬ is_square x ∨ (∃ k : ℕ, x = k^2 ∧ 8 ≤ k ∧ k ≤ 17)) :=
begin
  sorry
end

end perfect_squares_between_50_and_300_l806_806428


namespace X_on_altitude_BH_l806_806510

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806510


namespace sum_of_transformed_numbers_l806_806178

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
    3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l806_806178


namespace negation_of_existential_statement_l806_806605

variable (A : Set ℝ)

theorem negation_of_existential_statement :
  ¬(∃ x ∈ A, x^2 - 2 * x - 3 > 0) ↔ ∀ x ∈ A, x^2 - 2 * x - 3 ≤ 0 := by
  sorry

end negation_of_existential_statement_l806_806605


namespace basketball_team_points_l806_806608

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end basketball_team_points_l806_806608


namespace max_buses_in_city_l806_806852

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l806_806852


namespace ln_X_distribution_equality_l806_806536

noncomputable def gamma_const : ℝ := 0.5772 -- Euler-Mascheroni constant approximation

theorem ln_X_distribution_equality (α : ℝ) (X : ℝ) (Y : ℕ → ℝ) 
  (hX : X ~ Probability.distrib.gamma α 1) 
  (hY : ∀ n, Y n ~ Probability.distrib.exponential 1)
  (h_representation : ∀ n, 
    log X = (∑ i in Finset.range n, 
    log (X + (Finset.range (i - 1)).sum (λ k, Y k)) - 
    log (X + (Finset.range i).sum (λ k, Y k))) + 
    log (X + (Finset.range n).sum (λ i, Y i))) : 
  log X ~ -(gamma_const) + 
    ∑ n in Finset.range_natural, (1 / (n + 1) - Y n / (n + α)) :=
sorry

end ln_X_distribution_equality_l806_806536


namespace greatest_three_digit_divisible_by_8_ending_in_4_is_984_l806_806190

theorem greatest_three_digit_divisible_by_8_ending_in_4_is_984 :
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) → (n % 8 = 0) → (n % 10 = 4) → (n ≤ 984 ∧ (n = 984 → n = 984)) :=
by
  intros n hn h8 h4
  split
  sorry -- Proof that n ≤ 984
  intro
  exact a


end greatest_three_digit_divisible_by_8_ending_in_4_is_984_l806_806190


namespace lisa_quiz_goal_l806_806540

theorem lisa_quiz_goal (total_quizzes earned_A_on_first earned_A_goal remaining_quizzes additional_A_needed max_quizzes_below_A : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : earned_A_on_first = 30)
  (h3 : earned_A_goal = total_quizzes * 85 / 100)
  (h4 : remaining_quizzes = total_quizzes - 40)
  (h5 : additional_A_needed = earned_A_goal - earned_A_on_first)
  (h6 : max_quizzes_below_A = remaining_quizzes - additional_A_needed):
  max_quizzes_below_A = 0 :=
by sorry

end lisa_quiz_goal_l806_806540


namespace determine_fruit_weights_l806_806919

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806919


namespace fruit_weights_determined_l806_806951

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806951


namespace fraction_of_area_shaded_l806_806875

-- Define the equilateral triangle with side length of 6 cm
structure EquilateralTriangle (α : Type) :=
(side_length : α)
(is_equilateral : ∀ (a b c : α), a = b ∧ b = c ∧ c = side_length)

-- Define the condition that each shaded triangle is equilateral with 2 cm side length
def shaded_triangle : EquilateralTriangle ℝ := 
{ side_length := 2, 
  is_equilateral := by sorry }

-- Define the large triangle PQR
def large_triangle : EquilateralTriangle ℝ := 
{ side_length := 6, 
  is_equilateral := by sorry }

-- Define the fraction of the area that is shaded
def fraction_shaded_area (total_shaded : ℝ) (total_area : ℝ) : ℝ :=
  total_shaded / total_area

-- Define the main theorem: the fraction of the area of triangle PQR is 1/3
theorem fraction_of_area_shaded (n_shaded: ℕ) (n_total: ℕ) 
  (h1 : n_shaded = 3) (h2 : n_total = 9) : 
  fraction_shaded_area n_shaded n_total = 1 / 3 :=
by sorry

end fraction_of_area_shaded_l806_806875


namespace yangyang_helps_mom_for_5_days_l806_806612

-- Defining the conditions
def quantity_of_rice_in_warehouses_are_same : Prop := sorry
def dad_transports_all_rice_in : ℕ := 10
def mom_transports_all_rice_in : ℕ := 12
def yangyang_transports_all_rice_in : ℕ := 15
def dad_and_mom_start_at_same_time : Prop := sorry
def yangyang_helps_mom_then_dad : Prop := sorry
def finish_transporting_at_same_time : Prop := sorry

-- The theorem to prove
theorem yangyang_helps_mom_for_5_days (h1 : quantity_of_rice_in_warehouses_are_same) 
    (h2 : dad_and_mom_start_at_same_time) 
    (h3 : yangyang_helps_mom_then_dad) 
    (h4 : finish_transporting_at_same_time) : 
    yangyang_helps_mom_then_dad :=
sorry

end yangyang_helps_mom_for_5_days_l806_806612


namespace contradiction_proof_l806_806640

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem contradiction_proof (a b c : ℕ) :
  (¬ (exactly_one_odd a b c) ↔ (at_least_two_odd a b c ∨ all_even a b c)) :=
sorry

def exactly_one_odd (a b c : ℕ) : Prop :=
  (is_odd a ∨ is_odd b ∨ is_odd c) ∧ ¬(is_odd a ∧ is_odd b) ∧ ¬(is_odd a ∧ is_odd c) ∧ ¬(is_odd b ∧ is_odd c)

def at_least_two_odd (a b c : ℕ) : Prop :=
  (is_odd a ∧ is_odd b) ∨ (is_odd a ∧ is_odd c) ∨ (is_odd b ∧ is_odd c)

def all_even (a b c : ℕ) : Prop :=
  ¬(is_odd a) ∧ ¬(is_odd b) ∧ ¬(is_odd c)

end contradiction_proof_l806_806640


namespace four_digit_numbers_with_5_or_7_l806_806388

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l806_806388


namespace number_of_multiples_up_to_200_l806_806813

open Nat

theorem number_of_multiples_up_to_200 : 
  let multiples_3 := { n | n ≤ 200 ∧ n % 3 = 0 }
  let multiples_4 := { n | n ≤ 200 ∧ n % 4 = 0 }
  let multiples_6 := { n | n ≤ 200 ∧ n % 6 = 0 }
  let multiples_3_or_4 := (multiples_3 ∪ multiples_4)
  let multiples_3_or_4_not_6 := multiples_3_or_4 \ multiples_6
  in card multiples_3_or_4_not_6 = 50 :=
by 
  sorry

end number_of_multiples_up_to_200_l806_806813


namespace AlbertTookAwayCandies_l806_806969

-- Define the parameters and conditions given in the problem
def PatriciaStartCandies : ℕ := 76
def PatriciaEndCandies : ℕ := 71

-- Define the statement that proves the number of candies Albert took away
theorem AlbertTookAwayCandies :
  PatriciaStartCandies - PatriciaEndCandies = 5 := by
  sorry

end AlbertTookAwayCandies_l806_806969


namespace matrix_power_B150_l806_806064

open Matrix

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

-- Prove that B^150 = I
theorem matrix_power_B150 : 
  (B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_power_B150_l806_806064


namespace parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l806_806044

-- Curve C1 given by x^2 / 9 + y^2 = 1, prove its parametric form
theorem parametric_eq_C1 (α : ℝ) : 
  (∃ (x y : ℝ), x = 3 * Real.cos α ∧ y = Real.sin α ∧ (x ^ 2 / 9 + y ^ 2 = 1)) := 
sorry

-- Curve C2 given by ρ^2 - 8ρ sin θ + 15 = 0, prove its rectangular form
theorem rectangular_eq_C2 (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 
    (ρ ^ 2 - 8 * ρ * Real.sin θ + 15 = 0) ↔ (x ^ 2 + y ^ 2 - 8 * y + 15 = 0)) := 
sorry

-- Prove the maximum value of |PQ|
theorem max_dist_PQ : 
  (∃ (P Q : ℝ × ℝ), 
    (P = (3 * Real.cos α, Real.sin α)) ∧ 
    (Q = (0, 4)) ∧ 
    (∀ α : ℝ, Real.sqrt ((3 * Real.cos α) ^ 2 + (Real.sin α - 4) ^ 2) ≤ 8)) := 
sorry

end parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l806_806044


namespace num_solutions_fffx_eq_fx_l806_806528

theorem num_solutions_fffx_eq_fx : 
    let f : ℝ → ℝ := λ x, -2 * Real.sin (Real.pi * x) in
    ∀ (a b c : Set ℝ), a = { x | -2 ≤ x ∧ x ≤ 2 } ∧
                       b = {x | f(f(f(x))) = f(x)} ∧
                       c = {x | x ∈ a ∧ x ∈ b} →
    (∃ n, n = 61 ∧ Finset.card (Finset.filter (λ x, f(f(f(x))) = f(x)) (Finset.Icc (-2 : ℝ) 2)) = n)
:= by
  -- Conditions
  sorry

end num_solutions_fffx_eq_fx_l806_806528


namespace julia_height_is_172_7_cm_l806_806891

def julia_height_in_cm (height_in_inches : ℝ) (conversion_factor : ℝ) : ℝ :=
  height_in_inches * conversion_factor

theorem julia_height_is_172_7_cm :
  julia_height_in_cm 68 2.54 = 172.7 :=
by
  sorry

end julia_height_is_172_7_cm_l806_806891


namespace number_of_perfect_squares_l806_806419

theorem number_of_perfect_squares (a b : ℕ) (ha : 50 < a^2) (hb : b^2 < 300) :
  ∃ (n : ℕ), a ≤ n ∧ n ≤ b ∧ ∑ i in (finset.range (b - a + 1)).filter (λ n, 50 < n^2 ∧ n^2 < 300), 1 = 10 :=
sorry

end number_of_perfect_squares_l806_806419


namespace fruit_weights_correct_l806_806944

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l806_806944


namespace two_missing_faces_cube_valid_configs_l806_806686

-- Define the structure conditions of the "plus" shape and placement limitations.
def is_valid_addition (positions : list (ℕ × ℕ)) : Prop :=
  positions.length = 2 ∧
  ∀ p ∈ positions, p ∈ [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3)] ∧
  p.1 ≠ p.2

-- Define the number of configurations that can fold into a cube with two faces missing.
def valid_configurations : ℕ :=
  2

theorem two_missing_faces_cube_valid_configs :
  ∃ (n : ℕ), n = valid_configurations ∧ n = 2 :=
by {
  use 2,
  split,
  rfl,
  sorry
}

end two_missing_faces_cube_valid_configs_l806_806686


namespace point_on_altitude_l806_806524

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806524


namespace height_of_cone_l806_806613

def cone_height (V r : ℝ) : ℝ := (3 * V) / (Real.pi * r * r)

theorem height_of_cone :
  cone_height 12 3 = 4 / Real.pi :=
by 
  -- proof goes here
  sorry

end height_of_cone_l806_806613


namespace simplify_expression_l806_806130

theorem simplify_expression : 
  (1 : ℝ) / real.sqrt (1 + real.tan (real.of_real 160).toRadians ^ 2) = -real.cos (real.of_real 160).toRadians :=
by
  -- Proof is required here
  sorry

end simplify_expression_l806_806130


namespace tg_sum_of_angles_l806_806912

theorem tg_sum_of_angles (tetrahedron : Type) (face ABC : Set tetrahedron) (e : Line) (ϕ1 ϕ2 ϕ3 : ℝ) :
  (∀ (ϕ1 ϕ2 ϕ3), ∀ (angle_with_e : line → ℝ), angle_with_e(e) = ϕ1 ∧ angle_with_e(e) = ϕ2 ∧ angle_with_e(e) = ϕ3) →
  ∃ (a1 a2 a3 : ℝ), (a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0) →
  tan(ϕ1) * tan(ϕ1) + tan(ϕ2) * tan(ϕ2) + tan(ϕ3) * tan(ϕ3) = 12 
:= sorry

end tg_sum_of_angles_l806_806912


namespace area_parallelogram_equality_l806_806554

variables 
  {A B C D E F G H M L : ℝ^2} 
  (triangle_ABC : triangle A B C)
  (parallelogram_ACDE : parallelogram A C D E)
  (parallelogram_BCFG : parallelogram B C F G)
  (parallelogram_ABML : parallelogram A B M L)
  (point_H : ℝ^2)
  (extension_DE_FD : ∃ H, line DE ∧ line FD ∧ intersection DE FD H)
  (equal_parallel_AL_HC :  vector A L = vector H C ∧ parallel A L H C)
  (equal_parallel_BM_HC : vector B M = vector H C ∧ parallel B M H C)

theorem area_parallelogram_equality :
  (Area parallelogram_ABML) = 
  (Area parallelogram_ACDE + Area parallelogram_BCFG) :=
sorry

end area_parallelogram_equality_l806_806554


namespace regions_three_lines_l806_806108

theorem regions_three_lines (n : ℕ) (a2 a3 a4 a5 a6 ... an : ℕ) 
  (h1 : n > 2)
  (h2 : ∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → ¬ collinear i j k)
  (h3 : ∑ i in finset.range(n+1), i * ai = 2 * n^2)
  (h4 : ∑ i in finset.range(n+1), ai = (n^2 + n) / 2 + 1)
  (h5 : a3 + a4 + ... + an = (n^2 + n) / 2 + 1 - a2)
  (h6 : 3 * n = ∑ i in finset.range(n+1), i * ai):
  a3 ≥ a5 + 4 :=
sorry

end regions_three_lines_l806_806108


namespace trig_proof_l806_806788

variable {α a : ℝ}

theorem trig_proof (h₁ : (∃ a : ℝ, a < 0 ∧ (4 * a, -3 * a) = (4 * a, -3 * a)))
                    (h₂ : a < 0) :
  2 * Real.sin α + Real.cos α = 2 / 5 := 
sorry

end trig_proof_l806_806788


namespace part1_part2_l806_806360

noncomputable def f (x : Real) (ω : Real) : Real :=
  (Real.cos (ω * x))^2 - (Real.sin (ω * x))^2 + 2 * Real.sqrt 3 * (Real.cos (ω * x)) * (Real.sin (ω * x))

theorem part1 (ω : Real) (hω : ω > 0) (h : (Real.pi / ω) / 2 ≥ Real.pi / 2) :
  0 < ω ∧ ω ≤ 1 ∧ ∀ k : Int, 
    Real.smul k Real.pi / ω - Real.pi / (3 * ω) ≤
    x ∧ x ≤ Real.smul k Real.pi / ω + Real.pi / (6 * ω) :=
by
  sorry

theorem part2 (A B C : Real) (a b c : Real) 
  (ha : a = Real.sqrt 3) (hbc : b + c = 3) (hω : 1 ≤ 1)
  (hA : f A 1 = 1) :
  Real.sin B * Real.sin C = 1 / 2 :=
by 
  sorry

end part1_part2_l806_806360


namespace parabola_focus_coordinates_l806_806356

-- Define the given conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def passes_through (a : ℝ) (p : ℝ × ℝ) : Prop := p.snd = parabola a p.fst

-- Main theorem: Prove the coordinates of the focus
theorem parabola_focus_coordinates (a : ℝ) (h : passes_through a (1, 4)) (ha : a = 4) : (0, 1 / 16) = (0, 1 / (4 * a)) :=
by
  rw [ha] -- substitute the value of a
  simp -- simplify the expression
  sorry

end parabola_focus_coordinates_l806_806356


namespace find_b_values_l806_806776

open Real

theorem find_b_values
  (a b x : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : x > 1)
  (h_eq : 5 * (log a x)^2 + 9 * (log b x)^2 = (20 * (log x)^2) / (log a * log b)) :
  b = a ^ ((20 + sqrt 220) / 10) ∨ b = a ^ ((20 - sqrt 220) / 10) :=
by
  sorry

end find_b_values_l806_806776


namespace description_of_set_T_l806_806906

theorem description_of_set_T (x y : ℝ) :
  let T := { p : ℝ × ℝ | 
              (4 = p.1 + 1 ∧ p.2 - 5 ≤ 4) ∨ 
              (4 = p.2 - 5 ∧ p.1 + 1 ≤ 4) ∨ 
              (p.1 + 1 = p.2 - 5 ∧ 4 ≤ p.1 + 1) } in
  T = { p : ℝ × ℝ |
          (p.1 = 3 ∧ p.2 ≤ 9) ∨ 
          (p.1 ≤ 3 ∧ p.2 = 9) ∨ 
          (p.1 ≥ 3 ∧ p.2 = p.1 + 6) } :=
sorry

end description_of_set_T_l806_806906


namespace find_measure_angle_and_length_l806_806681

structure Triangle :=
(A B C : Point)
(AB : ℝ)
(∠ABC : ℝ)
(∠BAC : ℝ)

structure Circle :=
(center : Point)
(radius : ℝ)
(passes_through : Point -> Prop)

variables (A B C D : Point)
variables (triangle : Triangle)
variables (circle : Circle)

def right_triangle (t : Triangle) : Prop :=
t.∠ABC = 90 ∧ t.AB = 5

def tangent_to_circle (c : Circle) (l : Line) : Prop :=
∃ P, c.passes_through P ∧ l.tangent P c

def bisects_angle (ray : Ray) (angle : Angle) : Prop :=
ray.bisects angle

theorem find_measure_angle_and_length
  (h1 : right_triangle triangle)
  (h2 : circle.radius = 3)
  (h3 : circle.passes_through A)
  (h4 : circle.passes_through B)
  (h5 : tangent_to_circle circle (Line.mk C D))
  (h6 : bisects_angle (Ray.mk D A) (Angle.mk C D B)) :
  ∠ABD = Real.arcsin (5 / 6) ∧ AC = 225 / (16 * Real.sqrt 11) :=
sorry

end find_measure_angle_and_length_l806_806681


namespace simplify_expression_l806_806576

-- Define the expression
def expr := 7 * (2 - 3 * complex.I) + 4 * complex.I * (6 - complex.I) - 2 * (1 + 4 * complex.I)

-- Define the target value
def target := 16 - 5 * complex.I

theorem simplify_expression : expr = target :=
by
  -- The proof goes here, but it's omitted as per the instructions
  sorry

end simplify_expression_l806_806576


namespace X_lies_on_altitude_BH_l806_806503

variables (A B C H X : Point)
variables (w1 w2 : Circle)

-- Definitions and conditions from part a)
def X_intersects_w1_w2 : Prop := X ∈ w1 ∩ w2
def X_and_B_opposite_sides_AC : Prop := is_on_opposite_sides_of_line X B AC

-- Question to be proven
theorem X_lies_on_altitude_BH (h₁ : X_intersects_w1_w2 A B C H X w1 w2)
                             (h₂ : X_and_B_opposite_sides_AC A B C X BH) :
  lies_on_altitude X B H A C :=
sorry

end X_lies_on_altitude_BH_l806_806503


namespace sequence_general_term_l806_806171

open Nat

def sequence (n : ℕ) : ℝ :=
  if n = 1 then -sqrt 3
  else if n = 2 then 3
  else if n = 3 then -3 * sqrt 3
  else if n = 4 then 9
  else sorry -- Sequence continuation not required for proof

def general_term (n : ℕ) : ℝ :=
  (-1)^n * sqrt (3^n)

theorem sequence_general_term (n : ℕ) (h : n > 0) : sequence n = general_term n := by
  sorry

end sequence_general_term_l806_806171


namespace BE_computation_l806_806086

variables {A B C O F E : Type}
variables (AC BC BE FO E : ℝ)
variables (triangle_ABC : Prop)
variables (circumcenter_O : Prop)
variables (tangent_circumcircle_AOC_BC : Prop)
variables (intersection_line_AB_A_F : Prop)
variables (intersection_FO_BC_E : Prop)

noncomputable def compute_BE : ℝ :=
  let AC := 7 in
  AC / 2

theorem BE_computation
  (h_triangle_ABC : triangle_ABC)
  (h_circumcenter_O : circumcenter_O)
  (h_tangent_circumcircle_AOC_BC : tangent_circumcircle_AOC_BC)
  (h_intersection_line_AB_A_F : intersection_line_AB_A_F)
  (h_intersection_FO_BC_E : intersection_FO_BC_E)
  : BE = compute_BE :=
sorry

end BE_computation_l806_806086


namespace main_theorem_l806_806208

def scores_playerA : List ℕ := [7, 8, 8, 8, 9]
def scores_playerB : List ℕ := [7, 7, 7, 9, 10]
def new_scores_playerA : List ℕ := [7, 8, 8, 8, 9, 8]
def new_scores_playerB : List ℕ := [7, 7, 7, 9, 10, 8]

def average (scores : List ℕ) := (scores.sum) / scores.length

def variance (scores : List ℕ) := 
  let mean : ℚ := average scores
  (scores.map (fun x => (x - mean) ^ 2)).sum / scores.length

noncomputable def problem_statement : Prop := 
  average scores_playerA = 8 ∧ 
  average scores_playerB = 8 ∧ 
  variance scores_playerA < variance scores_playerB ∧ 
  variance new_scores_playerA < variance new_scores_playerB

theorem main_theorem : problem_statement := 
  by sorry

end main_theorem_l806_806208


namespace tangent_fixed_point_l806_806790

noncomputable def point_on_line (m : ℝ) : ℝ × ℝ := (9 - 2 * m, m)

def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

def line_equation (x y : ℝ) : Prop := x + 2 * y = 9

def fixed_point : ℝ × ℝ := (4/9, 8/9)

theorem tangent_fixed_point :
  ∀ m : ℝ, 
    (∃ (x y : ℝ), line_equation x y ∧ point_on_line m = (x, y)) → 
    (∃ (A B : ℝ × ℝ), 
       (∃ x y, circle_C x y ∧ (A = ⟨x, y⟩ ∨ B = ⟨x, y⟩)) ∧
       (∃ p1 p2 q1 q2 : ℝ, 
         A = (p1, p2) ∧ B = (q1, q2) ∧ (2 * m - 9) * p1 - m * p2 + 4 = 0 ∧ (2 * m - 9) * q1 - m * q2 + 4 = 0)) →
    ∃ (x y : ℝ), (x, y) = fixed_point :=
sorry

end tangent_fixed_point_l806_806790


namespace balls_in_boxes_l806_806820

theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 62 ∧ 
    (∀ (b1 b2 b3 b4 : ℕ), b1 + b2 + b3 + b4 = 6) ∧ 
    (are_distinguishable b1 b2 b3 b4) :=
begin
  sorry
end

end balls_in_boxes_l806_806820


namespace point_on_altitude_l806_806521

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806521


namespace det_N_cubed_l806_806017

variable {N : Matrix}

noncomputable def det_N_eq_3 : Prop := det N = 3

theorem det_N_cubed (h : det_N_eq_3) : det (N ^ 3) = 27 := by
  sorry

end det_N_cubed_l806_806017


namespace slopes_negative_reciprocals_minimum_area_triangle_ANB_conjecture_m_slopes_negative_reciprocals_conjecture_m_minimum_area_triangle_ANB_l806_806367

noncomputable def parabola : ℕ → ℝ → Prop := λ x y, y^2 = 4 * x
def point := (ℝ × ℝ)
def point_M := (1, 0)
def point_N := (-1, 0)
def on_parabola (p : point) : Prop := parabola p.1 p.2

def symmetric_wrt_y_axis (p : point) : point := (-p.1, p.2)
def min_triangle_area (m : ℝ) (m_gt_zero : m > 0) : ℝ := 4 * m * real.sqrt m

theorem slopes_negative_reciprocals :
  ∀ (l : ℝ → ℝ), ∃ (A B : point),
  on_parabola A ∧ on_parabola B ∧
  A ≠ B ∧ l = λ x, x - 1 →
  (A.2 / (A.1 + 1) + B.2 / (B.1 + 1) = 0) :=
sorry

theorem minimum_area_triangle_ANB :
  ∀ (M: point) (N : point) (A B : point),
  M = (1, 0) → N = symmetric_wrt_y_axis M → on_parabola A → on_parabola B →
  let area := real.abs (A.2 - B.2) in
  area = 4 :=
sorry

theorem conjecture_m_slopes_negative_reciprocals :
  ∀ (m : ℝ) (m_gt_zero : 0 < m), m ≠ 1 →
  ∀ (l : ℝ → ℝ), ∃ (A B : point),
  on_parabola A ∧ on_parabola B ∧
  A ≠ B ∧ l = λ x, x - m →
  (A.2 / (A.1 + m) + B.2 / (B.1 + m) = 0) :=
sorry

theorem conjecture_m_minimum_area_triangle_ANB :
  ∀ (m : ℝ) (m_gt_zero : 0 < m) (m_ne_one : m ≠ 1),
  ∀ (M: point), M = (m, 0) →
  ∀ (N : point), N = symmetric_wrt_y_axis M →
  ∀ (A B : point), on_parabola A → on_parabola B →
  let area := real.abs (A.2 - B.2) in
  area = min_triangle_area m m_gt_zero :=
sorry

end slopes_negative_reciprocals_minimum_area_triangle_ANB_conjecture_m_slopes_negative_reciprocals_conjecture_m_minimum_area_triangle_ANB_l806_806367


namespace probability_of_green_is_7_over_50_l806_806222

-- We have tiles numbered from 1 to 100
def total_tiles : ℕ := 100

-- A tile is green if it is congruent to 3 modulo 7
def is_green (n : ℕ) : Prop := n % 7 = 3

-- The number of green tiles
def green_tiles : ℕ := (List.range total_tiles).filter is_green).length

-- The probability of selecting a green tile
def probability_of_green : ℚ := green_tiles / total_tiles

-- The proof statement
theorem probability_of_green_is_7_over_50 : probability_of_green = 7 / 50 := by
  sorry

end probability_of_green_is_7_over_50_l806_806222


namespace money_spent_on_video_games_l806_806015

theorem money_spent_on_video_games (total : ℝ) (books_frac : ℝ) (toys_frac : ℝ) (snacks_frac : ℝ) :
  let money_spent_books := books_frac * total,
      money_spent_toys := toys_frac * total,
      money_spent_snacks := snacks_frac * total,
      total_spent := money_spent_books + money_spent_toys + money_spent_snacks,
      money_spent_video_games := total - total_spent
  in 
  total = 45 ∧
  books_frac = 1/4 ∧
  toys_frac = 1/3 ∧
  snacks_frac = 2/9 →
  money_spent_video_games = 8.75 := 
by
  intro h,
  obtain ⟨h1, h2, h3, h4⟩ := h,
  sorry

end money_spent_on_video_games_l806_806015


namespace max_teams_advancing_l806_806455

theorem max_teams_advancing:
  -- Definitions and conditions extracted:
  -- There are 7 teams
  -- Each team plays once against every other team
  -- Scored points: win = 3, draw = 1, loss = 0
  -- Teams advancing need 13 or more points
  -- Maximum points: 63
  let teams := 7,
      required_points := 13,
      total_games := 21,
      max_points := 63 in

  -- Question to prove:
  (∃ (n : ℕ), n ≤ teams ∧ n * required_points ≤ max_points) ∧ (¬∃ (n : ℕ), teams >= n ∧ n > 4) := 
by
  sorry

end max_teams_advancing_l806_806455


namespace hillside_hawks_loss_percentage_l806_806167

theorem hillside_hawks_loss_percentage (g_won g_lost : ℕ) (h_ratio : g_won.to_rat / g_lost.to_rat = 8 / 3) :
    (Float.ofRat (g_lost.to_rat / (g_won.to_rat + g_lost.to_rat)) * 100).toInt = 27 :=
by
  /- Definitions based on conditions -/
  let total_games := g_won + g_lost
  have h1 : (8 * g_lost) = (3 * g_won), from sorry
  let percent_lost := (g_lost.to_rat / total_games.to_rat) * 100
  /- Conclude the answer from the given conditions -/
  have percent_lost_nearest := Float.ofRat percent_lost
  exact Int.toInt percent_lost_nearest = 27

end hillside_hawks_loss_percentage_l806_806167


namespace length_of_train_l806_806252

theorem length_of_train (speed_kmph : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) 
  (h1 : speed_kmph = 45) (h2 : bridge_length_m = 220) (h3 : crossing_time_s = 30) :
  ∃ train_length_m : ℕ, train_length_m = 155 :=
by
  sorry

end length_of_train_l806_806252


namespace carolyn_removal_sum_l806_806983

-- Define the conditions
def n : ℕ := 7
def initial_removal : ℕ := 3

-- Define the sum of Carolyn's removals
def sum_removed_numbers := 9

-- State the theorem
theorem carolyn_removal_sum : 
  n = 7 →
  initial_removal = 3 →
  ∑ i in {3, 6}, i = sum_removed_numbers :=
by
  intros,
  sorry

end carolyn_removal_sum_l806_806983


namespace sequence_properties_l806_806096

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

def condition1 (n : ℕ) : Prop := 
  S n = a (n + 1) - 2 ^ (n + 1) + 1

def condition2 : Prop :=
  a 1 = 1

def is_arithmetic_sequence (f : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n : ℕ, f (n + 1) = f n + c

theorem sequence_properties (h1 : ∀ n ∈ ℕ, condition1 n) (h2 : condition2) :
  is_arithmetic_sequence (λ n, a n / 2 ^ (n - 1)) 1 ∧ (∀ n : ℕ, n > 0 → a n = n * 2 ^ (n - 1)) := 
sorry

end sequence_properties_l806_806096


namespace tangent_line_at_neg_one_l806_806147

noncomputable def f (x : ℝ) : ℝ := x^3 + (1 / x)

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_line_at_neg_one : 
  let x := -1 in
  let y := f x in
  2 * x - y = 0 :=
by
  sorry

end tangent_line_at_neg_one_l806_806147


namespace trigonometric_identity_simplification_l806_806577

theorem trigonometric_identity_simplification (x y : ℝ) :
  (cos x)^2 + (sin x)^2 + (cos (x + y))^2 - 2 * (cos x) * (cos y) * (cos (x + y)) - (sin x) * (sin y) = (sin (x - y))^2 := 
sorry

end trigonometric_identity_simplification_l806_806577


namespace bridge_length_is_correct_l806_806657

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * crossing_time_seconds
  total_distance - train_length

theorem bridge_length_is_correct :
  length_of_bridge 200 (60) 45 = 550.15 :=
by
  sorry

end bridge_length_is_correct_l806_806657


namespace four_digit_integers_with_5_or_7_l806_806384

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l806_806384


namespace ball_painting_probability_l806_806290

noncomputable def balls_probability : ℚ :=
  let p : ℚ := 1 / 2 in
  let combinations : ℕ := Nat.choose 8 4 in
  combinations * p^8

theorem ball_painting_probability :
  balls_probability = 35 / 128 :=
by
  sorry

end ball_painting_probability_l806_806290


namespace note_placement_l806_806376

/-- Hannah has thirteen notes with values: 1, 1, 2, 2, 2.5, 3, 3.5, 4, 4, 4.5, 5, 5.5, 6.
    She wants to distribute them into four boxes (X, Y, Z, W) such that:
    - Each box contains notes summing to an even or odd number.
    - Two boxes sum to even, and two boxes sum to odd.
    - A note with value 2 must go into box W.
    - A note with value 5 must go into box Y.
    We need to prove that given these conditions, the note with value 4.5 must go into box W. -/
theorem note_placement (X Y Z W : set ℝ) (hx : X.sum = 10) (hy : Y.sum = 9)
  (hz : Z.sum = 10) (hw : W.sum = 10) (h2 : 2 ∈ W) (h5 : 5 ∈ Y)
  (hvals : ∀ x ∈ X, x = 1 ∨ x = 2 ∨ x = 2.5 ∨ x = 3 ∨ x = 3.5 ∨ x = 4 ∨ x = 4.5 ∨ x = 5 ∨ x = 5.5 ∨ x = 6)
  (hvals' : ∀ y ∈ Y, y = 1 ∨ y = 2 ∨ y = 2.5 ∨ y = 3 ∨ y = 3.5 ∨ y = 4 ∨ y = 4.5 ∨ y = 5 ∨ y = 5.5 ∨ y = 6)
  (hvals'' : ∀ z ∈ Z, z = 1 ∨ z = 2 ∨ z = 2.5 ∨ z = 3 ∨ z = 3.5 ∨ z = 4 ∨ z = 4.5 ∨ z = 5 ∨ z = 5.5 ∨ z = 6)
  (hvals''' : ∀ w ∈ W, w = 1 ∨ w = 2 ∨ w = 2.5 ∨ w = 3 ∨ w = 3.5 ∨ w = 4 ∨ w = 4.5 ∨ w = 5 ∨ w = 5.5 ∨ w = 6) 
  : 4.5 ∈ W :=
sorry

end note_placement_l806_806376


namespace value_of_k_l806_806836

theorem value_of_k (x y k : ℝ) (h1 : 4 * x - 3 * y = k) (h2 : 2 * x + 3 * y = 5) (h3 : x = y) : k = 1 :=
sorry

end value_of_k_l806_806836


namespace midpoint_B_l806_806610

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 - dy)

def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_B'G'_is_correct:
  let B := (2, 3)
  let G := (6, 3)
  let B_translated := translate B 3 4
  let G_translated := translate G 3 4
  let B_reflected := reflect B_translated
  let G_reflected := reflect G_translated
  midpoint B_reflected G_reflected = (-1, 7) := by
  sorry

end midpoint_B_l806_806610


namespace contradiction_assumption_l806_806884

theorem contradiction_assumption (a : ℝ) (h : a < |a|) : ¬(a ≥ 0) :=
by 
  sorry

end contradiction_assumption_l806_806884


namespace simplify_complex_squaring_l806_806978

theorem simplify_complex_squaring :
  (4 - 3 * Complex.i) ^ 2 = 7 - 24 * Complex.i :=
by
  intro
  sorry

end simplify_complex_squaring_l806_806978


namespace interest_difference_l806_806207

def principal : ℝ := 3600
def rate : ℝ := 0.25
def time : ℕ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

theorem interest_difference :
  let SI := simple_interest principal rate time;
  let CI := compound_interest principal rate time;
  CI - SI = 225 :=
by
  sorry

end interest_difference_l806_806207


namespace increasing_on_interval_l806_806362

noncomputable def f (ω ϕ x : ℝ) : ℝ := sin (ω * x + ϕ) + cos (ω * x + ϕ)

theorem increasing_on_interval :
  ∀ (ω ϕ : ℝ), ω > 0 → 0 < ϕ < π →
  (∀ x : ℝ, f ω ϕ (-x) = -f ω ϕ x) →
  (∀ x1 x2 : ℝ, f ω ϕ x1 = sqrt 2 → f ω ϕ x2 = sqrt 2 → abs (x2 - x1) = π / 2) →
  ∀ x : ℝ, π / 8 ≤ x ∧ x ≤ 3 * π / 8 → (f 4 (3 * π / 4) x) < (f 4 (3 * π / 4) (x + π / 8)) :=
sorry

end increasing_on_interval_l806_806362


namespace max_number_of_elements_l806_806479

theorem max_number_of_elements (M : set ℕ) (hM : M ⊆ finset.range 2012 ∧ 
  (∀ a b c ∈ M, (a ∣ b ∨ b ∣ a) ∨ (a ∣ c ∨ c ∣ a) ∨ (b ∣ c ∨ c ∣ b))) : 
  finset.card M ≤ 18 := 
  sorry

end max_number_of_elements_l806_806479


namespace triangle_inequality_l806_806335

theorem triangle_inequality (A B C A1 A2 B1 B2 C1 C2 : Point)
  (h1 : Parallel (Line A1 A2) (Line B C))
  (h2 : Tangent (Incircle_triangle A B C) (Line A1 A2)) 
  (h3 : Parallel (Line B1 B2) (Line C A))
  (h4 : Tangent (Incircle_triangle A B C) (Line B1 B2)) 
  (h5 : Parallel (Line C1 C2) (Line A B))
  (h6 : Tangent (Incircle_triangle A B C) (Line C1 C2)) :
  (segment_length A A1 * segment_length A A2 +
   segment_length B B1 * segment_length B B2 +
   segment_length C C1 * segment_length C C2) ≥ 
  (1 / 9) * (segment_length A B ^ 2 +
             segment_length B C ^ 2 +
             segment_length C A ^ 2) :=
sorry

end triangle_inequality_l806_806335


namespace largest_even_whole_number_l806_806154

theorem largest_even_whole_number (x : ℕ) (h1 : 9 * x < 150) (h2 : x % 2 = 0) : x ≤ 16 :=
by
  sorry

end largest_even_whole_number_l806_806154


namespace hyperbola_equation_l806_806365

theorem hyperbola_equation (a b : ℝ)
  (hyp1 : a ≠ 0)
  (hyp2 : b ≠ 0)
  (hyperbola_condition : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → (a * a + 16) * (3 * 3 - 9) = 2)
  (short_distance_condition : (ℝ → ℝ) → (2 : ℝ))
  (point_on_asymptote : ∀ x, 4 = 3 * x)
  : ∃ (a b : ℝ), (a = 3) ∧ (b = 4) ∧ (a * a ≠ 0) ∧ (b * b ≠ 0) ∧ (x^2 / 9) - (y^2 / 16) = 1 := 
begin
  sorry
end

end hyperbola_equation_l806_806365


namespace chocolates_sold_eq_77_l806_806592

-- The given cost price and selling price relationships and gain percent
theorem chocolates_sold_eq_77 (C S : ℚ) (gain_percent : ℚ) (h_gain : gain_percent = 4 / 7) 
  (h_price_rel : S = C * (1 + gain_percent)) 
  (h_cost_selling : 121 * C = N * S) : 
  N = 77 :=
by
  -- We are given gain_percent = 4 / 7
  have h1 : gain_percent = 4 / 7 := h_gain
  -- substitute gain_percent into selling price S
  rw [h1] at h_price_rel

  -- S = C * (1 + 4 / 7)
  -- S = C * 11 / 7
  -- use this relationship in h_cost_selling
  rw [h_price_rel] at h_cost_selling
  
  -- Now our equation is 121 * C = N * (C * 11 / 7)
  -- solve for N
  have h2 : 121 * C = N * (C * (11 / 7)) := h_cost_selling
  -- divide both sides by C
  sorry

end chocolates_sold_eq_77_l806_806592


namespace point_X_on_altitude_BH_l806_806497

-- Definitions of points, lines, and circles
variables {A B C H X : Point} (w1 w2 : Circle)

-- Given conditions
-- 1. X is a point of intersection of circles w1 and w2
def intersection_condition : Prop := X ∈ w1 ∧ X ∈ w2

-- 2. X and B lie on opposite sides of line AC
def opposite_sides_condition : Prop := 
  (line_through A C).side_of_point X ≠ (line_through A C).side_of_point B

-- To prove: X lies on the altitude BH of triangle ABC
theorem point_X_on_altitude_BH
  (h1 : intersection_condition w1 w2) 
  (h2 : opposite_sides_condition A C X B H) :
  X ∈ line_through B H :=
sorry

end point_X_on_altitude_BH_l806_806497


namespace hexagon_fx_length_l806_806128

theorem hexagon_fx_length :
  ∀ (A B C D E F X : Type) (s : ℝ),
  regular_hexagon A B C D E F s ∧ side_length A B s ∧ s = 1 ∧ line_segment_extended A B X ∧
  length A X = 4 * length A B
  → length F X = 3 * Real.sqrt 7 := by
  intros A B C D E F X s h_hexagon h_side_len h_s_one h_extend h_ax_len
  sorry

end hexagon_fx_length_l806_806128


namespace find_b_in_expression_l806_806825

theorem find_b_in_expression
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^5 = a + b * Real.sqrt 3) :
  b = 44 :=
sorry

end find_b_in_expression_l806_806825


namespace derivative_at_0_l806_806782

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x * (deriv f 1)

theorem derivative_at_0 : deriv f 0 = -2 := sorry

end derivative_at_0_l806_806782


namespace discriminant_nonneg_l806_806438

open Real

theorem discriminant_nonneg (x : ℝ) :
  (∃ y : ℝ, 4 * y ^ 2 + 4 * x * y + x + 6 = 0) ↔ (x ≤ -2 ∨ x ≥ 3) :=
by
  split
  {
    intro h
    -- Proof will need to show x ≤ -2 ∨ x ≥ 3 from existence of y
    sorry
  }
  {
    intro h
    -- Proof will need to show existence of y if x ≤ -2 ∨ x ≥ 3
    sorry
  }

end discriminant_nonneg_l806_806438


namespace find_ratio_l806_806530

open Complex

theorem find_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h : (3 - 4 * Complex.i) * (p + q * Complex.i)).re = 0 : (p / q) = -4 / 3 :=
by sorry

end find_ratio_l806_806530


namespace determine_fruit_weights_l806_806925

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l806_806925


namespace sum_first_10_is_80_l806_806037

-- Define the sequence
def arith_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Given conditions
def cond1 (a d : ℤ) : Prop :=
  arith_seq a d 0 + arith_seq a d 2 + arith_seq a d 4 = 9

def cond2 (a d : ℤ) : Prop :=
  arith_seq a d 1 + arith_seq a d 3 + arith_seq a d 5 = 15

-- Sum of the first 10 terms
def sum_first_10_terms (a d : ℤ) : ℤ :=
  (0 to 9).sum (arith_seq a d)

-- The proof statement
theorem sum_first_10_is_80 (a d : ℤ)
  (h1 : cond1 a d) (h2 : cond2 a d) :
  sum_first_10_terms a d = 80 := 
  sorry

end sum_first_10_is_80_l806_806037


namespace prism_surface_area_l806_806179

theorem prism_surface_area
  (d : ℝ) (a : ℝ) (h : ℝ)
  (h_d : d = 2) (h_a : a = 1) (h_edge : h = real.sqrt (d^2 - 2 * a^2))
  : let S := 2 * a^2 + 4 * a * h in S = 4 * real.sqrt 2 + 2 := by
{
  sorry
}

end prism_surface_area_l806_806179


namespace min_period_k_l806_806564

def has_period {α : Type*} [HasZero α] (r : α) (n : ℕ) : Prop :=
  -- A function definition to check if 'r' has a repeating decimal period of length 'n'
  sorry

theorem min_period_k (a b : ℚ) (h₁ : has_period a 30) (h₂ : has_period b 30) (h₃ : has_period (a - b) 15) :
  ∃ (k : ℕ), k = 6 ∧ has_period (a + k * b) 15 :=
begin
  sorry
end

end min_period_k_l806_806564


namespace X_on_altitude_BH_l806_806485

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806485


namespace sufficient_but_not_necessary_condition_counter_example_x_ge_1_sufficient_not_necessary_l806_806326

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|) :=
by
  sorry

theorem counter_example (x : ℝ) : (x < -1) → (|x + 1| + |x - 1| = 2 * |x|) :=
by
  sorry

theorem x_ge_1_sufficient_not_necessary : (∀ x : ℝ, (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|)) 
  ∧ (∃ x : ℝ, (x < -1) ∧ (|x + 1| + |x - 1| = 2 * |x|)) :=
by
  split
  · intro x
    apply sufficient_but_not_necessary_condition
  · use -2
    split
    · linarith
    · apply counter_example
  sorry
  

end sufficient_but_not_necessary_condition_counter_example_x_ge_1_sufficient_not_necessary_l806_806326


namespace parallelogram_angle_A_l806_806464

theorem parallelogram_angle_A 
  (A B : ℝ) (h1 : A + B = 180) (h2 : A - B = 40) :
  A = 110 :=
by sorry

end parallelogram_angle_A_l806_806464


namespace fruit_weights_l806_806939

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806939


namespace X_on_altitude_BH_l806_806488

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806488


namespace digits_with_five_or_seven_is_5416_l806_806404

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l806_806404


namespace num_pairs_eq_factor_product_l806_806783

theorem num_pairs_eq_factor_product (n : ℕ) (hn : n > 0) :
  let factor_pairs := ∏ i in (factor_set (n^2)).to_list, (2 * i.snd + 1) in
  (∃ x y : ℕ, (x > 0 ∧ y > 0) ∧ (x * y = n * (x + y))) = factor_pairs :=
by sorry

end num_pairs_eq_factor_product_l806_806783


namespace different_colors_probability_l806_806892

-- Conditions
def shorts_colors : Finset ℕ := {1, 2, 3} -- Representing black, gold, red
def jersey_colors : Finset ℕ := {4, 5, 6} -- Representing black, white, green

def total_configurations : ℕ := shorts_colors.card * jersey_colors.card

def non_matching_configurations : ℕ :=
  (shorts_colors.product jersey_colors).filter (λ p, p.1 ≠ p.2).card

def probability (num favorable : ℕ) : ℚ := (favorable : ℚ) / num

-- Proof Statement
theorem different_colors_probability :
  probability total_configurations non_matching_configurations = 8 / 9 :=
by
  -- Proof goes here
  sorry

end different_colors_probability_l806_806892


namespace values_of_x_and_y_l806_806909

theorem values_of_x_and_y (x y : ℝ) (h : {x, y, x + y} = {0, x^2, x * y}) :
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) :=
by {
  sorry
}

end values_of_x_and_y_l806_806909


namespace percentage_less_than_y_l806_806838

variable (w x y z : ℝ)

-- Given conditions
variable (h1 : w = 0.60 * x)
variable (h2 : x = 0.60 * y)
variable (h3 : z = 1.50 * w)

theorem percentage_less_than_y : ( (y - z) / y) * 100 = 46 := by
  sorry

end percentage_less_than_y_l806_806838


namespace problem_l806_806970

theorem problem (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
sorry

end problem_l806_806970


namespace equilateral_hexagon_star_perimeter_l806_806897

theorem equilateral_hexagon_star_perimeter (ABCDEF : Hexagon)
  (h1 : equilateral ABCDEF)
  (h2 : convex ABCDEF)
  (h3 : perimeter ABCDEF = 1) :
  let s := star_perimeter (extend_sides ABCDEF) in
  max_s : s - min_s : s = 0 :=
sorry

end equilateral_hexagon_star_perimeter_l806_806897


namespace X_on_altitude_BH_l806_806508

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806508


namespace solution_set_l806_806769

-- Given problem conditions
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b
def k (x b : ℝ -> ℝ) : ℝ := -1 -- given that k should just be a placeholder since not going to use it directly here

-- Given constant and inequality condition
axiom constant_k (k b : ℝ) : k ≠ 0
axiom const_function_vals : ∀ (x : ℝ), linear_function k 3 (-2) = 3 ∧ 
                                          linear_function k 3 (-1) = 2 ∧ 
                                          linear_function k 3 (0) = 1 ∧ 
                                          linear_function k 3 (1) = 0 ∧ 
                                          linear_function k 3 (2) = -1 ∧ 
                                          linear_function k 3 (3) = -2

-- Required proof statement
theorem solution_set (k b : ℝ) : ∀ x, (linear_function k b x < 0) -> x > 1 :=
by
  intros x hx
  sorry

end solution_set_l806_806769


namespace find_seq_formula_l806_806046

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, 0 < n → 
  (∑ i in finset.range n, (a (i + 1) / (i + 1)^2)) = a n

theorem find_seq_formula (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, 0 < n → a n = (2 * n) / (n + 1) :=
by
  sorry

end find_seq_formula_l806_806046


namespace geometric_seq_fourth_term_l806_806148

theorem geometric_seq_fourth_term (a r : ℝ) (h_a : a = 1024) (h_r_pow : a * r ^ 5 = 125) :
  a * r ^ 3 = 2000 := by
  sorry

end geometric_seq_fourth_term_l806_806148


namespace part_one_part_two_l806_806775

-- Definitions based on the problem conditions
def p (m : ℝ) : Prop := ∃ x0 ∈ Set.Icc (0 : ℝ) 2, Real.log (x0 + 2) / Real.log 2 < 2 * m
def q (m : ℝ) : Prop := 1 - 3 * m^2 > 0 ∧ m ≠ 0

-- Part (I): Prove q(m) implies m is in the specified range
theorem part_one (m : ℝ) : q(m) → m ∈ Set.Ioo (-Real.sqrt 3 / 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3 / 3) :=
  sorry

-- Part (II): Prove (¬ p(m) ∧ q(m)) implies m is in the specified range 
theorem part_two (m : ℝ) : (¬ p(m)) ∧ q(m) → m ∈ (Set.Ioo (-Real.sqrt 3 / 3) 0 ∪ Set.Ioo 0 (1/2)) :=
  sorry

end part_one_part_two_l806_806775


namespace wake_up_time_l806_806582

-- Definition of the conversion ratio from normal minutes to metric minutes
def conversion_ratio := 36 / 25

-- Definition of normal minutes in a full day
def normal_minutes_in_day := 24 * 60

-- Definition of metric minutes in a full day
def metric_minutes_in_day := 10 * 100

-- Definition to convert normal time (6:36 AM) to normal minutes
def normal_minutes_from_midnight (h m : ℕ) := h * 60 + m

-- Converting normal minutes to metric minutes using the conversion ratio
def metric_minutes (normal_mins : ℕ) := (normal_mins / 36) * 25

-- Definition of the final metric time 2:75
def metric_time := (2 * 100 + 75)

-- Proving the final answer is 275
theorem wake_up_time : 100 * 2 + 10 * 7 + 5 = 275 := by
  sorry

end wake_up_time_l806_806582


namespace tan_theta_eq_neg_three_fourths_l806_806829

-- Definitions based on the given conditions
def z (θ : ℝ) : ℂ := (sin θ - (3 / 5 : ℝ)) + (cos θ - (4 / 5 : ℝ)) * complex.I

-- The main theorem statement
theorem tan_theta_eq_neg_three_fourths (θ : ℝ) (h : z θ.im = 0):
  tan θ = -3 / 4 :=
begin
  sorry
end

end tan_theta_eq_neg_three_fourths_l806_806829


namespace num_different_configurations_of_lights_l806_806622

-- Definition of initial conditions
def num_rows : Nat := 6
def num_columns : Nat := 6
def possible_switch_states (n : Nat) : Nat := 2^n

-- Problem statement to be verified
theorem num_different_configurations_of_lights :
  let num_configurations := (possible_switch_states num_rows - 1) * (possible_switch_states num_columns - 1) + 1
  num_configurations = 3970 :=
by
  sorry

end num_different_configurations_of_lights_l806_806622


namespace problem_1_problem_2_l806_806311

-- Define the sequence and the first-order difference sequence
def Δ (a : ℕ → ℕ) (n : ℕ) : ℕ := a (n + 1) - a n

-- Define the k-th order difference sequence
def Δk (a : ℕ → ℕ) (k n : ℕ) : ℕ :=
  if k = 0 then a n else Δ (Δk a (k - 1)) n

-- (1) Prove the value of a_2013
theorem problem_1 : ∀ (a : ℕ → ℕ), (Δ a n = 2) (a 1 = 1) → a 2013 = 4015 := 
  sorry

-- (2) Prove the general formula for the sequence
theorem problem_2 : ∀ (a : ℕ → ℕ), (Δk a 2 n - Δ a (n+1) + a n = -2^n) (a 1 = 1) → 
                     ∀ n, a n = n * 2^(n - 1) := 
  sorry

end problem_1_problem_2_l806_806311


namespace isosceles_triangle_count_l806_806811

-- Define the condition for the perimeter of the triangle
def perim_condition (a b : ℕ) : Prop := 2 * a + b = 11

-- Define the triangle inequality conditions
def triangle_inequality (a b : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ 2 * a > b ∧ a + b > a ∧ a + b > b

-- Count the number of valid isosceles triangles
def count_isosceles_triangles_with_perimeter_11 : ℕ :=
  (if perim_condition 3 5 ∧ triangle_inequality 3 5 then 1 else 0) + 
  (if perim_condition 4 3 ∧ triangle_inequality 4 3 then 1 else 0)

theorem isosceles_triangle_count : count_isosceles_triangles_with_perimeter_11 = 2 := sorry

end isosceles_triangle_count_l806_806811


namespace balls_in_boxes_l806_806822

theorem balls_in_boxes :
  let num_balls := 6
  let num_boxes := 4
  (finset.card {x : fin (num_boxes + num_balls - 1) | finset.card (x.image x.pred) = num_balls - 1}) = 84 :=
by
  let num_balls := 6
  let num_boxes := 4
  have combination_formula : (∑ i in finset.range num_boxes, x i = num_balls) →
    (finset.card {x : fin (num_boxes + num_balls - 1) | finset.card (x.image x.pred) = num_balls - 1}) = 84 := sorry
  combination_formula sorry

end balls_in_boxes_l806_806822


namespace monotonic_decrease_interval_find_c_l806_806794

-- Definitions for the function and given conditions
def f (x : ℝ) := sqrt 3 * sin (x - π / 3) + 2 * cos (x / 2) ^ 2

-- Part I: Monotonic Decrease Interval
theorem monotonic_decrease_interval (k : ℤ) : 
  monotonic_decr ( Icc (2 * k * π + 2 * π / 3) (2 * k * π + 5 * π / 3)) :=
sorry

-- Part II: Solving for c given specific conditions
theorem find_c (A B C a b c : ℝ) (h_a : a = sqrt 3) (h_sin_B : sin B = 2 * sin C) 
  (h_f_A : f A = 3 / 2) (h_sin_law : b / sin B = c / sin C) (cos_law: a^2 = b^2 + c^2 - 2 * b * c * cos A) : 
  c = 1 :=
sorry

end monotonic_decrease_interval_find_c_l806_806794


namespace mean_median_difference_l806_806034

def percentage_of_students (total : ℝ) (pct : ℝ) : ℝ := (pct / 100) * total

def scores (total : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ :=
  let s60 := percentage_of_students total 15
  let s75 := percentage_of_students total 20
  let s80 := percentage_of_students total 30
  let s85 := percentage_of_students total 25
  let s90 := percentage_of_students total (100 - (15 + 20 + 30 + 25))
  (s60, s75, s80, s85, s90)

def median_score (students : ℝ × ℝ × ℝ × ℝ × ℝ) : ℝ :=
  let (s60, s75, s80, s85, s90) := students
  s80 -- given the 20th and 21st student gets 80 points

def mean_score (students : ℝ × ℝ × ℝ × ℝ × ℝ) (total : ℝ) : ℝ :=
  let (s60, s75, s80, s85, s90) := students
  (60 * s60 + 75 * s75 + 80 * s80 + 85 * s85 + 90 * s90) / total

theorem mean_median_difference : 
  let total := 40
  let students := scores total
  let mean := mean_score students total
  let median := median_score students
  |mean - median| = 1.75 :=
by
  sorry

end mean_median_difference_l806_806034


namespace part1_part2_l806_806073

-- Definitions for the sets A and B
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Proof problem (1): A ∩ B = {2} implies a = -5 or a = 1
theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := 
sorry

-- Proof problem (2): A ∪ B = A implies a > 3
theorem part2 (a : ℝ) (h : A ∪ B a = A) : 3 < a :=
sorry

end part1_part2_l806_806073


namespace rectangle_area_l806_806243

variable (w l A P : ℝ)
variable (h1 : l = w + 6)
variable (h2 : A = w * l)
variable (h3 : P = 2 * (w + l))
variable (h4 : A = 2 * P)
variable (h5 : w = 3)

theorem rectangle_area
  (w l A P : ℝ)
  (h1 : l = w + 6)
  (h2 : A = w * l)
  (h3 : P = 2 * (w + l))
  (h4 : A = 2 * P)
  (h5 : w = 3) :
  A = 27 := 
sorry

end rectangle_area_l806_806243


namespace probability_class_4_drawn_first_second_l806_806233

noncomputable def P_1 : ℝ := 1 / 10
noncomputable def P_2 : ℝ := 9 / 100

theorem probability_class_4_drawn_first_second :
  P_1 = 1 / 10 ∧ P_2 = 9 / 100 := by
  sorry

end probability_class_4_drawn_first_second_l806_806233


namespace sum_four_digit_integers_ending_in_zero_l806_806636

def arithmetic_series_sum (a l d : ℕ) : ℕ := 
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_four_digit_integers_ending_in_zero : 
  arithmetic_series_sum 1000 9990 10 = 4945500 :=
by
  sorry

end sum_four_digit_integers_ending_in_zero_l806_806636


namespace sum_of_integers_in_interval_l806_806136

-- Define the conditions
def condition1 (x : ℝ) : Prop := x^2 - x - 56 ≥ 0
def condition2 (x : ℝ) : Prop := x^2 - 25 * x + 136 ≥ 0
def condition3 (x : ℝ) : Prop := x > 8
def condition4 (x : ℝ) : Prop := x + 7 ≥ 0

-- Define the proof statement
theorem sum_of_integers_in_interval :
  ∑ x in (finset.Icc (-25 : ℤ) (25 : ℤ)).filter (λ x, 
    condition1 x ∧ condition2 x ∧ condition3 x ∧ condition4 x), 
    x = -285 :=
by sorry

end sum_of_integers_in_interval_l806_806136


namespace X_on_altitude_BH_l806_806487

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- condition: X is a point of intersection of w1 and w2
def is_intersection (w1 w2 : Circle) (X : Point) : Prop :=
  w1.contains X ∧ w2.contains X

-- condition: X and B lie on opposite sides of line AC
def opposite_sides (A C B X : Point) : Prop :=
  ∃ l : Line, l.contains A ∧ l.contains C ∧ ((l.side B ≠ l.side X) ∨ (l.opposite_side B X))

-- Theorem: X lies on the altitude BH
theorem X_on_altitude_BH
  (intersect_w1w2 : is_intersection w1 w2 X)
  (opposite : opposite_sides A C B X)
  : lies_on_altitude X B H A C :=
sorry

end X_on_altitude_BH_l806_806487


namespace triangle_existence_condition_l806_806723

theorem triangle_existence_condition 
  (a b f_c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : f_c > 0) : 
  (2 * a * b / (a + b)) > f_c :=
sorry

end triangle_existence_condition_l806_806723


namespace triangle_BMN_equilateral_l806_806804

-- Define points A, B, C collinear on a line, with B between A and C
variables (A B C : Point)
variable (h_collinear : Collinear [A, B, C])
variable (h_between : Between A B C)

-- Define an equilateral triangle ABC1 on segment AB with C1 on the same side as A and B
variables (C1 : Point)
variable (h_equilateral_ABC1 : EquilateralTriangle A B C1)
variable (h_same_side_C1 : SameSide A B C1)

-- Define another equilateral triangle BCA1 on segment BC with A1 on the same side as B and C
variables (A1 : Point)
variable (h_equilateral_BCA1 : EquilateralTriangle B C A1)
variable (h_same_side_A1 : SameSide B C A1)

-- Define M as the midpoint of AA1 and N as the midpoint of CC1
variables (M N : Point)
variable (h_midpoint_M : Midpoint M A A1)
variable (h_midpoint_N : Midpoint N C C1)

-- Theorem to be proved: triangle BMN is equilateral
theorem triangle_BMN_equilateral :
  EquilateralTriangle B M N :=
by
  sorry

end triangle_BMN_equilateral_l806_806804


namespace circle_A_rotations_circle_A_rotations_l806_806266

-- Define the radii of the circles
variables (r : ℝ)

-- Define the problem statement
theorem circle_A_rotations (r : ℝ) : sorry := sorry

noncomputable def num_rotations (r : ℝ) : ℕ :=
  if r > 0 then 3 else sorry

theorem circle_A_rotations (r : ℝ) (hr : r > 0) :
  num_rotations r = 3 :=
sorry

end circle_A_rotations_circle_A_rotations_l806_806266


namespace number_of_perfect_squares_is_10_l806_806423

-- Define the number of integers n such that 50 ≤ n^2 ≤ 300
def count_perfect_squares_between_50_and_300 : ℕ :=
  (finset.Icc 8 17).card

-- Statement to prove
theorem number_of_perfect_squares_is_10 : count_perfect_squares_between_50_and_300 = 10 := by
  sorry

end number_of_perfect_squares_is_10_l806_806423


namespace digits_with_five_or_seven_is_5416_l806_806408

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l806_806408


namespace intersect_lines_l806_806871

theorem intersect_lines
  (A B C D : ℝ × ℝ × ℝ)
  (A_coord : A = (8, -9, 9))
  (B_coord : B = (18, -19, 15))
  (C_coord : C = (1, 2, -7))
  (D_coord : D = (3, -6, 13)) :
  ∃ t s : ℝ, 
    (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2), A.3 + t * (B.3 - A.3)) = 
    (C.1 + s * (D.1 - C.1), C.2 + s * (D.2 - C.2), C.3 + s * (D.3 - C.3)) ∧
    (C.1 + s * (D.1 - C.1), C.2 + s * (D.2 - C.2), C.3 + s * (D.3 - C.3)) = (4, -10, 23) :=
by
  sorry

end intersect_lines_l806_806871


namespace X_on_altitude_BH_l806_806509

-- Define the given geometric objects and their properties
variables {α : Type} [EuclideanAffineSpace α]

-- Assume ΔABC is a triangle with orthocenter H
variables (A B C H X : α)

-- Define the circles w1 and w2
variable (w1 w2 : Circle α)

-- Given conditions from the problem
axiom intersect_circles : X ∈ w1 ∧ X ∈ w2
axiom opposite_sides : ∀ l : Line α, A ∈ l → C ∈ l → (X ∈ l → ¬ (B ∈ l))

-- Define the altitude BH
def altitude (A  B : α) : Line α := 
  altitude (ℓ B C) B

-- Conclusion to prove
theorem X_on_altitude_BH :
  X ∈ altitude A B H
:= sorry

end X_on_altitude_BH_l806_806509


namespace value_of_f_at_quarter_l806_806025

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem value_of_f_at_quarter (α : ℝ) (hα : 4^α = 2) : f (1/4) α = 1/2 :=
by
  rw [f, ←Real.sqrt_eq_rpow_half]
  sorry

end value_of_f_at_quarter_l806_806025


namespace lines_intersect_sum_c_d_l806_806157

theorem lines_intersect_sum_c_d (c d : ℝ) 
    (h1 : ∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) 
    (h2 : ∀ x y : ℝ, x = 3 ∧ y = 3) : 
    c + d = 4 :=
by sorry

end lines_intersect_sum_c_d_l806_806157


namespace line_passes_through_fixed_point_l806_806105

theorem line_passes_through_fixed_point (x y k : ℝ) :
  (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0 → (x, y) = (2, 3) :=
by
  intro h
  have h1 : 2 * x - y - 1 = 0 := sorry
  have h2 : -x - 3 * y + 11 = 0 := sorry
  have hx : x = 2 := sorry
  have hy : y = 3 := sorry
  exact (prod.mk.inj_iff.2 ⟨hx, hy⟩)

end line_passes_through_fixed_point_l806_806105


namespace four_digit_numbers_with_5_or_7_l806_806386

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l806_806386


namespace orthogonal_circles_form_pencil_l806_806209

theorem orthogonal_circles_form_pencil (O : Point) (R : ℝ) (pencil : set Circle) 
  (h : ∀ S ∈ pencil, circle_center S = O ∧ circle_radius S = R) 
  (orthogonal : ∀ T, ∃ rad_axis, ∀ S ∈ pencil, T ⊥ S ↔ passes_through (circ_rad_axis S T) O) :
  ∃ new_pencil : set Circle, (∀ T ∈ new_pencil, ∀ S ∈ pencil, T ⊥ S) ∧
    (∀ T₁ T₂ ∈ new_pencil, circ_rad_axis T₁ T₂ passes_through O) :=
sorry

end orthogonal_circles_form_pencil_l806_806209


namespace altitude_contains_x_l806_806513

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806513


namespace days_vacuuming_l806_806263

theorem days_vacuuming (V : ℕ) (h1 : ∀ V, 130 = 30 * V + 40) : V = 3 :=
by
    have eq1 : 130 = 30 * V + 40 := h1 V
    sorry

end days_vacuuming_l806_806263


namespace smallest_number_among_bases_l806_806259

theorem smallest_number_among_bases : 
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2 + 1 in
  n4 < n3 ∧ n4 < n1 ∧ n4 < n2 :=
by
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  sorry

end smallest_number_among_bases_l806_806259


namespace part1_part2_l806_806359

-- Define the function f and its derivative
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x + 2 / x + a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := (deriv (λ x, 2 * x + 2 / x + a * Real.log x)) x

-- Define the function g
def g (x : ℝ) (a : ℝ) : ℝ := x^2 * (f' x a + 2 * x - 2)

-- Statement for part (1)
theorem part1 (a : ℝ) : (∀ x ∈ (Set.Ici 1), f' x a ≥ 0) → a ∈ Set.Ici 0 :=
sorry

-- Statement for part (2)
theorem part2 (a : ℝ) (h_min : (∃ x, x > 0 ∧ g x a = -6)) : a = -6 → ∀ x : ℝ, f x -6 = 2 * x + 2 / x - 6 * Real.log x :=
sorry

end part1_part2_l806_806359


namespace probability_both_in_picture_l806_806122

/-- Rachel completes a lap every 90 seconds, Robert completes a lap every 80 seconds,
and both start running from the same line at the same time. A picture is taken at a
random time between 600 and 660 seconds, showing one-fourth of the track centered
on the starting line. Prove that the probability of both Rachel and Robert being in
the picture is 3/16. -/
theorem probability_both_in_picture :
  let lap_rachel := 90
  let lap_robert := 80
  let t_start := 600
  let t_end := 660
  let picture_cover := 1/4
  let time_range := t_end - t_start
  let prob := (11.25 / 60 : ℚ)
  prob = 3 / 16 := sorry

end probability_both_in_picture_l806_806122


namespace girls_more_than_boys_by_155_l806_806869

def number_of_girls : Real := 542.0
def number_of_boys : Real := 387.0
def difference : Real := number_of_girls - number_of_boys

theorem girls_more_than_boys_by_155 :
  difference = 155.0 := 
by
  sorry

end girls_more_than_boys_by_155_l806_806869


namespace p_approximation_l806_806169

noncomputable def p_value : ℝ :=
  (-6 + 5 * Real.sqrt 2) / 14

def conditions (graph : Graph) (jorgosz_accommodation theater : Vertex) (p : ℝ) :=
  is_icosahedron graph ∧
  opposite_vertex graph jorgosz_accommodation theater ∧
  ∀ v ∈ vertices graph,
    (if v = jorgosz_accommodation ∨ v = theater then true
     else ∃ shortest_route : Prop, probability shortest_route p)

theorem p_approximation (graph : Graph) (jorgosz_accommodation theater : Vertex) :
  conditions graph jorgosz_accommodation theater p_value →
  probability_reach_accommodation_before_theater jorgosz_accommodation theater 0.5 :=
sorry

end p_approximation_l806_806169


namespace smallest_k_l806_806563

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l806_806563


namespace max_buses_l806_806859

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l806_806859


namespace speedster_convertibles_l806_806651

theorem speedster_convertibles 
  (T : ℕ)
  (h1 : T / 3 = 50)
  (h2 : 2 * T / 3 ∈ ℕ)
  (h3 : 4 * (2 * T / 3) / 5 ∈ ℕ) :
  (4 * (2 * T / 3) / 5 = 80) :=
sorry

end speedster_convertibles_l806_806651


namespace symmetric_point_cartesian_coordinates_l806_806370

def M_coordinates : ℝ × ℝ :=
  let r := 6
  let θ := (11 * Real.pi) / 6
  (r * Real.cos θ, r * Real.sin θ)

def symmetric_with_respect_to_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

theorem symmetric_point_cartesian_coordinates :
  symmetric_with_respect_to_y_axis M_coordinates = (-3 * Real.sqrt 3, -3) :=
by
  sorry

end symmetric_point_cartesian_coordinates_l806_806370


namespace max_buses_in_city_l806_806849

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l806_806849


namespace vertex_of_parabola_l806_806353

theorem vertex_of_parabola (a : ℝ) :
  (∃ (k : ℝ), ∀ x : ℝ, y = -4*x - 1 → x = 2 ∧ (a - 4) = -4 * 2 - 1) → 
  (2, -9) = (2, a - 4) → a = -5 :=
by
  sorry

end vertex_of_parabola_l806_806353


namespace count_four_digit_numbers_with_5_or_7_l806_806400

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l806_806400


namespace maria_towels_l806_806202

theorem maria_towels (green_towels white_towels given_towels : ℕ) (bought_green : green_towels = 40) 
(bought_white : white_towels = 44) (gave_mother : given_towels = 65) : 
  green_towels + white_towels - given_towels = 19 := by
sorry

end maria_towels_l806_806202


namespace length_of_A_l806_806072

-- Definitions of points A, B, C and their properties
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 14)
def C : ℝ × ℝ := (3, 7)

-- A' and B' are on the line y = x
def on_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

-- The lines AA' and BB' intersect at C
-- We assume that AA' and BB' are defined by the points given in the question

def AA'_intersects_C (A' : ℝ × ℝ) : Prop :=
  A'.1 = 5 ∧ A'.2 = 5

def BB'_intersects_C (B' : ℝ × ℝ) : Prop :=
  B'.1 = 4 + 0.2 ∧ B'.2 = 4 + 0.2

-- Define the distance between two points 
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

-- The final proof statement
theorem length_of_A'B' : 
  ∃ A' B', on_y_eq_x A' ∧ on_y_eq_x B' ∧ AA'_intersects_C A' ∧ BB'_intersects_C B' ∧ distance A' B' = real.sqrt 1.28 :=
by
  -- Definitions for A' and B'
  let A' := (5, 5)
  let B' := (4.2, 4.2)
  use A', B'
  split
  -- Proof A' is on the line y = x
  { 
    unfold on_y_eq_x,
    exact rfl,
  }
  split
  -- Proof B' is on the line y = x
  {
    unfold on_y_eq_x,
    exact rfl,
  }
  split
  -- Proof AA' intersects C
  { 
    unfold AA'_intersects_C,
    exact ⟨rfl, rfl⟩,
  }
  split
  -- Proof BB' intersects C
  { 
    unfold BB'_intersects_C,
    exact ⟨rfl, rfl⟩,
  }
  -- Proof the distance is sqrt(1.28)
  { 
    unfold distance,
    exact rfl,
  }

end length_of_A_l806_806072


namespace problem_solution_l806_806800

open Real

theorem problem_solution :
  (∃ x₀ : ℝ, log x₀ ≥ x₀ - 1) ∧ (¬ ∀ θ : ℝ, sin θ + cos θ < 1) :=
by
  sorry

end problem_solution_l806_806800


namespace required_blocks_l806_806676

def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

def volume_of_rect_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def number_of_blocks (volume_cyl volume_block : ℝ) :=
  (volume_cyl / volume_block).ceil

theorem required_blocks : 
  number_of_blocks (volume_of_cylinder 2.5 10) (volume_of_rect_prism 8 3 2) = 5 :=
by
  sorry

end required_blocks_l806_806676


namespace sin_double_angle_l806_806322

theorem sin_double_angle (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin (2 * x) = 3 / 5 := 
by 
  sorry

end sin_double_angle_l806_806322


namespace count_perfect_squares_between_50_and_300_l806_806426

theorem count_perfect_squares_between_50_and_300 : 
  ∃ n, number_of_perfect_squares 50 300 = n ∧ n = 10 := 
sorry

end count_perfect_squares_between_50_and_300_l806_806426


namespace set_intersection_l806_806373

open Set

def U := {x : ℝ | True}
def A := {x : ℝ | x^2 - 2 * x < 0}
def B := {x : ℝ | x - 1 ≥ 0}
def complement (U B : Set ℝ) := {x : ℝ | x ∉ B}
def intersection (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection :
  intersection A (complement U B) = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end set_intersection_l806_806373


namespace parallelogram_area_l806_806688

def base : ℕ := 20
def height : ℕ := 7

theorem parallelogram_area : base * height = 140 := by
  sorry

end parallelogram_area_l806_806688


namespace point_on_altitude_l806_806525

-- Definitions for the points and circle properties
variables {A B C H X : Point}
variables (w1 w2 : Circle)

-- Condition that X is in the intersection of w1 and w2
def X_in_intersection : Prop := X ∈ w1 ∧ X ∈ w2

-- Condition that X and B lie on opposite sides of line AC
def opposite_sides (X B : Point) (AC : Line) : Prop := 
  (X ∈ half_plane AC ∧ B ∉ half_plane AC) ∨ (X ∉ half_plane AC ∧ B ∈ half_plane AC)

-- Altitude BH in triangle ABC
def altitude_BH (B H : Point) (AC : Line) : Line := Line.mk B H

-- Main theorem statement to prove
theorem point_on_altitude
  (h1 : X_in_intersection w1 w2)
  (h2 : opposite_sides X B (Line.mk A C))
  : X ∈ (altitude_BH B H (Line.mk A C)) :=
sorry

end point_on_altitude_l806_806525


namespace trail_length_l806_806887

def miles_hiked {R : Type} [AddCommGroup R] [MulAction R ℝ] (x1 x2 x3 x4 x5 : R) : Prop :=
  x1 + x2 + x3 = 42 ∧
  x2 + x3 = 30 ∧
  x4 + x5 = 40 ∧
  x1 + x4 = 36

theorem trail_length 
  {R : Type} [AddCommGroup R] [MulAction R ℝ] 
  (x1 x2 x3 x4 x5 : R) 
  (h : miles_hiked x1 x2 x3 x4 x5) :
  x1 + x2 + x3 + x4 + x5 = 82 := 
sorry

end trail_length_l806_806887


namespace cube_root_count_l806_806433

theorem cube_root_count : ∃ n, n = {x : ℕ | x > 0 ∧ x ^ (1/3:ℝ) < 20}.card := 
by
  let S := {x : ℕ | x > 0 ∧ x < 8000}
  have h_eq: {x : ℕ | x > 0 ∧ (x:ℝ) ^ (1/3) < 20} = S := 
    sorry  -- a proof that sets {x : ℕ | x > 0 ∧ (x:ℝ) ^ (1/3) < 20} and {x : ℕ | x > 0 ∧ x < 8000} are equal
  let n := S.card
  have h_n: n = 7999 := 
    by 
      sorry -- a proof for counting elements in S
  existsi n
  exact h_n

end cube_root_count_l806_806433


namespace henry_time_proof_l806_806726

-- Define the time Dawson took to run the first leg of the course
def dawson_time : ℝ := 38

-- Define the average time they took to run a leg of the course
def average_time : ℝ := 22.5

-- Define the time Henry took to run the second leg of the course
def henry_time : ℝ := 7

-- Prove that Henry took 7 seconds to run the second leg
theorem henry_time_proof : 
  (dawson_time + henry_time) / 2 = average_time :=
by
  -- This is where the proof would go
  sorry

end henry_time_proof_l806_806726


namespace exists_coprime_less_than_100_l806_806116

theorem exists_coprime_less_than_100 (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ∃ d, d < 100 ∧ gcd d a = 1 ∧ gcd d b = 1 ∧ gcd d c = 1 :=
by sorry

end exists_coprime_less_than_100_l806_806116


namespace original_weight_of_apples_l806_806620

theorem original_weight_of_apples (x : ℕ) (h1 : 5 * (x - 30) = 2 * x) : x = 50 :=
by
  sorry

end original_weight_of_apples_l806_806620


namespace cost_of_laura_trip_l806_806061

noncomputable def cost_of_gas
  (a b : ℕ) (r : ℕ) (p : ℝ) : ℝ :=
  let distance := b - a
  let gallons := distance / r
  let cost := gallons * p
  Float.round (cost.toFloat)

theorem cost_of_laura_trip :
  cost_of_gas 85432 85470 25 3.85 = 5.85 :=
by
  sorry

end cost_of_laura_trip_l806_806061


namespace fruit_weights_l806_806959

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806959


namespace total_amount_spent_l806_806219

variable (n : ℕ) (t1 t2 : ℝ)

-- Definitions from conditions 
def people_count : ℕ := 7
def spent_by_six : ℝ := 66
def spent_by_one : ℝ := t1
def average_expenditure (t : ℝ) : ℝ := t / 7
def additional_expenditure : ℝ := 6

axiom total_spent_by_all_people (t : ℝ) : 
  t = spent_by_six + (average_expenditure t + additional_expenditure)

-- The statement to prove
theorem total_amount_spent : total_spent_by_all_people t2 → t2 = 84 := 
sorry

end total_amount_spent_l806_806219


namespace number_of_lattice_points_on_hyperbola_l806_806689

theorem number_of_lattice_points_on_hyperbola :
  {p : ℤ × ℤ | let x := p.1, y := p.2 in x^2 - y^2 = 1800^2}.to_finset.card = 250 :=
sorry

end number_of_lattice_points_on_hyperbola_l806_806689


namespace solution_of_system_l806_806197

theorem solution_of_system :
  ∃ x y z : ℚ,
    x + 2 * y = 12 ∧
    y + 3 * z = 15 ∧
    3 * x - z = 6 ∧
    x = 54 / 17 ∧
    y = 75 / 17 ∧
    z = 60 / 17 :=
by
  exists 54 / 17, 75 / 17, 60 / 17
  repeat { sorry }

end solution_of_system_l806_806197


namespace largest_possible_integer_l806_806236

def is_valid_list (l : List ℕ) : Prop :=
  (l.length = 5) ∧ 
  (List.count l 6 > 1) ∧ 
  (∀ x, x ≠ 6 → List.count l x ≤ 1) ∧
  (l.sorted.get! 2 = 7) ∧
  (l.sum / 5 = 11)

theorem largest_possible_integer : ∃ l : List ℕ, is_valid_list l ∧ l.maximum? = some 28 := sorry

end largest_possible_integer_l806_806236


namespace max_value_of_c_l806_806441

theorem max_value_of_c (a b c : ℝ) (h1 : 1 / 2^a + 1 / 2^b = 1) (h2 : 1 / 2^(a+b) + 1 / 2^(b+c) + 1 / 2^(a+c) = 1) :
  c ≤ 2 - Real.log2 3 := sorry

end max_value_of_c_l806_806441


namespace score_on_fourth_board_l806_806727

theorem score_on_fourth_board 
  (score1 score2 score3 score4 : ℕ)
  (h1 : score1 = 30)
  (h2 : score2 = 38)
  (h3 : score3 = 41)
  (total_score : score1 + score2 = 2 * score4) :
  score4 = 34 := by
  sorry

end score_on_fourth_board_l806_806727


namespace seating_arrangement_l806_806456

-- Definitions based on the conditions
def total_all_stars : ℕ := 9
def cubs_all_stars : ℕ := 4
def red_sox_all_stars : ℕ := 3
def yankees_all_stars : ℕ := 2
def cubs_coach : ℕ := 1

-- Proving the main statement
theorem seating_arrangement : 
    (cubs_all_stars + cubs_coach) + red_sox_all_stars + yankees_all_stars = total_all_stars →
    ∃ (ways : ℕ), ways = (3! * (5!) * (3!) * (2!)) ∧ ways = 8640 :=
begin
  intro h,
  use (3! * 5! * 3! * 2!),
  split,
  { refl },
  { norm_num }
end

end seating_arrangement_l806_806456


namespace possible_atomic_numbers_l806_806611

/-
Given the following conditions:
1. An element X is from Group IIA and exhibits a +2 charge.
2. An element Y is from Group VIIA and exhibits a -1 charge.
Prove that the possible atomic numbers for elements X and Y that can form an ionic compound with the formula XY₂ are 12 for X and 9 for Y.
-/

structure Element :=
  (atomic_number : Nat)
  (group : Nat)
  (charge : Int)

def GroupIIACharge := 2
def GroupVIIACharge := -1

axiom X : Element
axiom Y : Element

theorem possible_atomic_numbers (X_group_IIA : X.group = 2)
                                (X_charge : X.charge = GroupIIACharge)
                                (Y_group_VIIA : Y.group = 7)
                                (Y_charge : Y.charge = GroupVIIACharge) :
  (X.atomic_number = 12) ∧ (Y.atomic_number = 9) :=
sorry

end possible_atomic_numbers_l806_806611


namespace eval_floor_log2_sum_l806_806741

theorem eval_floor_log2_sum : (∑ n in Finset.range 6237.succ, Int.floor (Real.log n / Real.log 2)) = 66666 := 
sorry

end eval_floor_log2_sum_l806_806741


namespace right_triangle_sides_l806_806606

theorem right_triangle_sides (a b c : ℝ) (h : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c = 60 → h = 12 → a^2 + b^2 = c^2 → a * b = 12 * c → 
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end right_triangle_sides_l806_806606


namespace trajectory_of_P_is_line_segment_l806_806340

open Real

def point (x y : ℝ) : Type := (x, y)

def distance (p1 p2 : point) : ℝ :=
  sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

def is_line_segment (F1 F2 : point) (P : point → Prop) : Prop :=
  ∀ (P : point), distance P F1 + distance P F2 = 10 → P.1 ≥ min F1.1 F2.1 ∧ P.1 ≤ max F1.1 F2.1 ∧ P.2 = 0

theorem trajectory_of_P_is_line_segment (F1 F2 : point) (hF1 : F1 = (-5, 0)) (hF2 : F2 = (5, 0)) :
  ∃ (P : point → Prop), is_line_segment F1 F2 P :=
by
  sorry

end trajectory_of_P_is_line_segment_l806_806340


namespace find_a_from_inequality_l806_806366

theorem find_a_from_inequality (a : ℝ) :
  (∀ x : ℝ, (x - a) / (x + 1) > 0 ↔ x ∈ (-∞, -1) ∪ (4, ∞)) → a = 4 :=
by
  sorry

end find_a_from_inequality_l806_806366


namespace sum_of_odds_1_to_50_l806_806304

theorem sum_of_odds_1_to_50 : ∑ n in (Finset.range 50).filter (λ x, x % 2 = 1), n = 625 := by
  sorry

end sum_of_odds_1_to_50_l806_806304


namespace intersecting_lines_l806_806193

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end intersecting_lines_l806_806193


namespace determine_x_l806_806735

theorem determine_x (x : ℝ) (hx : 0 < x) (h : x * ⌊x⌋ = 72) : x = 9 :=
sorry

end determine_x_l806_806735


namespace perimeter_range_triangle_ABC_l806_806805

noncomputable def triangle_perimeter_range (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 1 ∧ a * (Real.cos C) + (1 / 2) * c = b →

-- Define the property that the perimeter l must lie in the range (2, 3]
  let l := a + b + c in 2 < l ∧ l ≤ 3

-- Assert the property for triangle ABC
theorem perimeter_range_triangle_ABC (a b c A B C : ℝ) 
  (h₀ : a = 1)
  (h₁ : a * (Real.cos C) + (1 / 2) * c = b) :
  2 < a + b + c ∧ a + b + c ≤ 3 :=
by
  sorry

end perimeter_range_triangle_ABC_l806_806805


namespace number_of_rotations_equals_twice_edges_l806_806119

noncomputable theory

-- Definitions based on conditions from the problem

def regular_polyhedron (P : Type) := 
  ∃ (V : set P) (E : set (P × P)) (F : set (set P)), 
    (∃ (v : P → Prop), ∃ (e : (P × P) → Prop), 
      ∃ a, (E.card = a) ∧
      (∀ x, v x ↔ x ∈ V) ∧ 
      (∀ y, e y ↔ y ∈ E) ∧ 
      -- Ensuring vertices, edges, and faces meet certain regular polyhedron properties
      -- including rotational symmetry.

def rotation (P : Type) := 
  ∀ θ : ℝ, ∃ (f : P → P), 
    (∀ p ∈ P, f p = p ∨ f p ≠ p) ∧ -- rotation by θ degrees
    (θ = 0 ∨ θ = 90 ∨ θ = 180 ∨ θ = 270) -- considering basic right angles including identity rotation.

-- Theorem to prove based on the mathematically equivalent statement
theorem number_of_rotations_equals_twice_edges {P : Type} 
  (hp: regular_polyhedron P) (a : ℕ) : 
  ∃ nr, nr = 2 * a :=
begin
  sorry
end

end number_of_rotations_equals_twice_edges_l806_806119


namespace sum_of_distinct_integers_l806_806079

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
(h_prod : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120) : 
a + b + c + d + e = 33 := 
sorry

end sum_of_distinct_integers_l806_806079


namespace fruit_weights_assigned_l806_806964

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l806_806964


namespace num_true_propositions_equals_two_l806_806792

noncomputable def condition1 (k : ℝ) (b : ℝ) (hb : k * b = 0) : Prop :=
  k = 0 ∨ b = 0

noncomputable def condition2 (a b : ℝ) (hab : a = b) : Prop :=
  a = 0 ∨ b = 0

noncomputable def condition3 (a b : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0) (h_not_parallel : a ≠ b) (h_eq_norm : abs a = abs b) : Prop :=
  (a + b) * (a - b) = 0

noncomputable def condition4 (a b : ℝ) (h_parallel : a = b) : Prop :=
  a * b = abs a * abs b

theorem num_true_propositions_equals_two :
  (condition1 0 0 rfl) ∧
  ¬(condition2 1 1 rfl) ∧
  (condition3 1 (-1) (by norm_num) (by norm_num) (by norm_num) (by norm_num)) ∧
  ¬(condition4 1 (-1) rfl) →
  (2 = 2) := by
  intro h
  sorry

end num_true_propositions_equals_two_l806_806792


namespace measure_angle_Q_l806_806129

theorem measure_angle_Q (BC DE : ℝ) (Q D E : ℝ) (angle_BC DE : ℝ) :
  ∀ ABCDEFGH : Octagon, is_regular_octagon ABCDEFGH →
  angle_BCD == 135 ∧ angle_DCE == 45 ∧ angle_EQD == 45 →
  angle QDE = 90 :=
begin
  sorry
end

end measure_angle_Q_l806_806129


namespace complex_point_coordinates_l806_806145

theorem complex_point_coordinates : 
  (let z := (3 : ℂ) + complex.I in 
  let w := (1 : ℂ) + complex.I in 
  z / w = 2 - complex.I) := 
by
  sorry

end complex_point_coordinates_l806_806145


namespace altitude_contains_x_l806_806518

open EuclideanGeometry

variables {A B C H X : Point}
variables {w1 w2 : Circle}

-- Define the given conditions
def intersecting_circles (w1 w2 : Circle) (X : Point) : Prop :=
  X ∈ w1 ∧ X ∈ w2
  
def opposite_sides (X B : Point) (l : Line) : Prop :=
  ¬same_side X B l
  
def altitude (B C : Point) (H : Point) (A : Point) : Line :=
  line_through B H

-- Prove the statement using given conditions
theorem altitude_contains_x (w1 w2 : Circle) (A B C H X : Point) :
  intersecting_circles w1 w2 X →
  opposite_sides X B (line_through A C) →
  X ∈ altitude B C H A :=
sorry

end altitude_contains_x_l806_806518


namespace find_angle_ABC_l806_806173

-- Define the given conditions for the problem.
variables (A B C : Type) 
          [triangle A B C]
          [tangentCircle C (triangle A B C)]
          (angleBAC : ℝ) (angleACB : ℝ)

-- Assume the given conditions: ∠BAC = 50° and ∠ACB = 40°.
axiom angleBAC_eq_50 : angleBAC = 50
axiom angleACB_eq_40 : angleACB = 40

-- Define the proof problem as: Prove that ∠ABC = 50°
theorem find_angle_ABC : ∠ABC = 50 :=
by
  sorry

end find_angle_ABC_l806_806173


namespace trajectory_of_moving_circle_center_maximum_triangle_area_l806_806238

-- Define the given conditions
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 9
def circle_F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory of the moving circle's center
def trajectory (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line equation
def line (m y : ℝ) : ℝ := m * y + 1

-- Define the area of the triangle
def area_triangle (y1 y2 : ℝ) : ℝ := Float.abs (y1 - y2)

-- Statement to prove the trajectory
theorem trajectory_of_moving_circle_center (x y : ℝ) (h1 : circle_F1 x y) (h2 : circle_F2 x y) :
  trajectory x y :=
sorry

-- Statement to prove the maximum area
theorem maximum_triangle_area (m x1 y1 x2 y2 : ℝ) (h_line1 : x1 = line m y1) (h_line2 : x2 = line m y2) 
  (h_traj1 : trajectory x1 y1) (h_traj2 : trajectory x2 y2) 
  (h_max : (area_triangle y1 y2) ≤ 3) : 
  (m = 0 ∧ (area_triangle y1 y2) = 3) :=
sorry

end trajectory_of_moving_circle_center_maximum_triangle_area_l806_806238


namespace find_A_and_cos_2C_l806_806028

variable {a b c A B C : Real}

-- Conditions:
-- In triangle ABC, sides opposite to angles A, B, C are a, b, c respectively,
-- and b * cos C + c * cos B = sqrt(2) * a * cos A
axiom h : b * cos C + c * cos B = sqrt(2) * a * cos A

-- Proof problem:
theorem find_A_and_cos_2C (h : b * cos C + c * cos B = sqrt(2) * a * cos A)
  (hA1 : a = 5) (hA2 : b = 3 * sqrt 2):
  (A = pi / 4) ∧ (cos (2 * C) = -24 / 25) :=
by
  sorry

end find_A_and_cos_2C_l806_806028


namespace dilation_rotation_matrix_l806_806299

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, Real.sin θ; -Real.sin θ, Real.cos θ]

def combined_transformation_matrix (k θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (rotation_matrix θ) ⬝ (dilation_matrix k)

theorem dilation_rotation_matrix :
  combined_transformation_matrix 2 (-(Float.pi / 2)) = !![0, 2; -2, 0] :=
by
  -- Expect the proof to be here
  sorry

end dilation_rotation_matrix_l806_806299


namespace neg_sin_leq_1_l806_806341

theorem neg_sin_leq_1 :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by
  sorry

end neg_sin_leq_1_l806_806341


namespace g_min_value_4sqrt578_l806_806584

noncomputable def g_min_value (X : ℝ^3) : ℝ :=
  let A := (0, 0, 0)
  let B := (48, 0, 0)
  let D := (24, √291, 0)
  let C := (24, -√291, 0)
  let dist (P Q : ℝ^3) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)
  min (dist A X + dist B X + dist C X + dist D X) -- This is a simplification assuming the X in 3D space
  
theorem g_min_value_4sqrt578 (X : ℝ^3) :
  g_min_value X = 4 * Real.sqrt 578 := 
sorry

end g_min_value_4sqrt578_l806_806584


namespace fruit_weights_l806_806933

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l806_806933


namespace dice_probability_l806_806218

theorem dice_probability :
  (let prob_one_digit := 9 / 20;
       prob_two_digit := 11 / 20;
       ways_to_choose := 20;
       probability := ways_to_choose * (prob_one_digit^3 * prob_two_digit^3))
  in probability = 970701 / 3200000 :=
by
  let prob_one_digit := 9 / 20
  let prob_two_digit := 11 / 20
  let ways_to_choose := 20
  let probability := ways_to_choose * (prob_one_digit^3 * prob_two_digit^3)
  show probability = 970701 / 3200000
  sorry

end dice_probability_l806_806218


namespace find_a12_l806_806468

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 1) = a n + 3) ∧ (a 2 + a 8 = 26)

theorem find_a12 (a : ℕ → ℤ) (h : sequence a) : a 12 = 34 :=
sorry

end find_a12_l806_806468


namespace scaling_transformation_coordinates_l806_806045

theorem scaling_transformation_coordinates :
  ∀ (A A' : ℝ × ℝ), A = (1/3, -2) →
  A' = (3 * A.1, (1 / 2) * A.2) →
  A' = (1, -1) :=
begin
  intros A A' hA hA',
  cases hA,
  cases hA',
  simp only [prod.mk.inj_iff] at *,
  exact ⟨hA.1, hA.2⟩,
end

end scaling_transformation_coordinates_l806_806045


namespace set_contains_all_nonnegative_integers_l806_806087

theorem set_contains_all_nonnegative_integers (S : Set ℕ) :
  (∃ a b, a ∈ S ∧ b ∈ S ∧ 1 < a ∧ 1 < b ∧ Nat.gcd a b = 1) →
  (∀ x y, x ∈ S → y ∈ S → y ≠ 0 → (x * y) ∈ S ∧ (x % y) ∈ S) →
  (∀ n, n ∈ S) :=
by
  intros h1 h2
  sorry

end set_contains_all_nonnegative_integers_l806_806087


namespace length_of_wall_l806_806678

def volume_of_brick(cm_length cm_width cm_height : ℝ) : ℝ :=
  (cm_length / 100) * (cm_width / 100) * (cm_height / 100)

def total_volume_needed(bricks : ℝ) (volume_of_one_brick : ℝ) : ℝ :=
  bricks * volume_of_one_brick

def volume_of_wall(length width height : ℝ) : ℝ :=
  length * width * height

theorem length_of_wall
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ)
  (num_bricks : ℝ) :
  num_bricks = 1366.6666666666667 →
  volume_of_brick brick_length brick_width brick_height = 0.0036 →
  volume_of_wall L wall_width wall_height = total_volume_needed num_bricks 0.0036
  → L = 0.06 :=
sorry

end length_of_wall_l806_806678


namespace balls_in_boxes_l806_806818

theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 62 ∧ 
    (∀ (b1 b2 b3 b4 : ℕ), b1 + b2 + b3 + b4 = 6) ∧ 
    (are_distinguishable b1 b2 b3 b4) :=
begin
  sorry
end

end balls_in_boxes_l806_806818


namespace no_rational_roots_for_quadratic_except_specific_primes_l806_806535

-- Definition of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

-- The main theorem to solve the given problem
theorem no_rational_roots_for_quadratic_except_specific_primes (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  (∃ x : ℚ, x^2 + p^2 * x + q^3 = 0) ↔ (p = 3 ∧ q = 2) :=
begin
  sorry
end

end no_rational_roots_for_quadratic_except_specific_primes_l806_806535


namespace KBrO3_bromine_mass_percentage_l806_806298

theorem KBrO3_bromine_mass_percentage :
  let molar_mass_K := 39.10
  let molar_mass_Br := 79.90
  let molar_mass_O := 16.00
  let total_molar_mass := molar_mass_K + molar_mass_Br + 3 * molar_mass_O
  shows (molar_mass_Br / total_molar_mass * 100) = 47.844 :=
by {
  let molar_mass_K := 39.10,
  let molar_mass_Br := 79.90,
  let molar_mass_O := 16.00,
  let total_molar_mass := molar_mass_K + molar_mass_Br + 3 * molar_mass_O,
  show (molar_mass_Br / total_molar_mass * 100) = 47.844, from sorry
}

end KBrO3_bromine_mass_percentage_l806_806298


namespace p_sufficient_for_q_q_not_necessary_for_p_l806_806765

variable (x : ℝ)

def p := |x - 2| < 1
def q := 1 < x ∧ x < 5

theorem p_sufficient_for_q : p x → q x :=
by sorry

theorem q_not_necessary_for_p : ¬ (q x → p x) :=
by sorry

end p_sufficient_for_q_q_not_necessary_for_p_l806_806765


namespace log_inner_evaluation_l806_806733

theorem log_inner_evaluation :
  log 6 (log 4 (log 3 81)) = 0 := 
by
  sorry

end log_inner_evaluation_l806_806733


namespace fruit_weights_determined_l806_806949

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l806_806949


namespace arc_length_l806_806030

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 10) (h_α : α = 2 * Real.pi / 3) : 
  r * α = 20 * Real.pi / 3 := 
by {
sorry
}

end arc_length_l806_806030


namespace number_of_diagonals_from_vertex_l806_806242

-- Define the internal angle as given in the condition
def internal_angle : ℝ := 140

-- Calculate the external angle
def external_angle : ℝ := 180 - internal_angle

-- Calculate the number of sides of the polygon
def num_sides : ℕ := 360 / external_angle

-- The statement to prove
theorem number_of_diagonals_from_vertex : num_sides - 3 = 6 := by
  sorry

end number_of_diagonals_from_vertex_l806_806242


namespace value_of_k_l806_806985

variables (f : ℝ → ℝ) (k : ℝ)

theorem value_of_k (h1 : ∀ x, f(x) + f(1 - x) = k)
                   (h2 : ∀ x, f(1 + x) = 3 + f(x))
                   (h3 : ∀ x, f(x) + f(-x) = 7) : 
                   k = 10 :=
by
  sorry

end value_of_k_l806_806985


namespace initial_toy_cars_l806_806550

-- Define the initial given conditions
variables U G Dad Mum Auntie Uncle Total : ℕ

-- Condition statements
hypothesis H1 : G = 2 * Uncle
hypothesis H2 : Dad = 10
hypothesis H3 : Mum = Dad + 5
hypothesis H4 : Auntie = Uncle + 1
hypothesis H5 : Total = 196

-- Given sum of toy cars from all family members
def sum_gifts : ℕ := G + Dad + Mum + Auntie + Uncle

-- Given total number of Olaf's toy cars after receiving gifts
def initial_cars : ℕ := Total - sum_gifts

-- Theorem to prove initial number of toy cars Olaf had
theorem initial_toy_cars :
  initial_cars = 150 :=
by
  -- Incorporate the above definitions and hypotheses
  sorry

end initial_toy_cars_l806_806550


namespace lowest_score_dropped_l806_806655

-- Conditions definitions
def total_sum_of_scores (A B C D : ℕ) := A + B + C + D = 240
def total_sum_after_dropping_lowest (A B C : ℕ) := A + B + C = 195

-- Theorem statement
theorem lowest_score_dropped (A B C D : ℕ) (h1 : total_sum_of_scores A B C D) (h2 : total_sum_after_dropping_lowest A B C) : D = 45 := 
sorry

end lowest_score_dropped_l806_806655


namespace juliet_supporter_probability_capulet_l806_806048

theorem juliet_supporter_probability_capulet :
  let total_population := 6
  let montague_population := total_population * 4 / 6
  let capulet_population := total_population * 1 / 6
  let verona_population := total_population * 1 / 6
  let juliet_supporters_montague := montague_population * 20 / 100
  let juliet_supporters_capulet := capulet_population * 70 / 100
  let juliet_supporters_verona := verona_population * 50 / 100
  let total_juliet_supporters := juliet_supporters_montague + juliet_supporters_capulet + juliet_supporters_verona
  in 
  (juliet_supporters_capulet / total_juliet_supporters) = 35 / 100
:= by
  sorry

end juliet_supporter_probability_capulet_l806_806048


namespace max_value_Q_b_l806_806537

open MeasureTheory ProbabilityTheory Set

/--
Given a real number \( b \) such that \( 0 \leq b \leq 1 \), 
let \( Q(b) \) represent the probability that 
\[ \cos^2(\pi x) + \cos^2(\pi y) < 1 \]
where \( x \) is chosen from the interval \( [0, b^2] \) 
and \( y \) from the interval \( [0, 1] \).

The maximum value of \( Q(b) \) is \( \frac{\pi}{4} \).
-/
theorem max_value_Q_b : ∀ (b : ℝ), 0 ≤ b ∧ b ≤ 1 → 
  let Q_b : ℝ → ℝ 
      := λ b, (⨍ x in interval 0 b^2, ⨍ y in interval 0 1, (χ ((λ (x y : ℝ), (cos (π * x))^2 + (cos (π * y))^2 < 1)) (x, y))) 
  (∀ (b : ℝ), Q_b(b) ≤ Q_b(1)) ∧ Q_b(1) = (π / 4) :=
by
  sorry

end max_value_Q_b_l806_806537


namespace num_males_selected_l806_806694

theorem num_males_selected (total_male total_female total_selected : ℕ)
                           (h_male : total_male = 56)
                           (h_female : total_female = 42)
                           (h_selected : total_selected = 28) :
  (total_male * total_selected) / (total_male + total_female) = 16 := 
by {
  sorry
}

end num_males_selected_l806_806694


namespace fruit_weights_l806_806958

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l806_806958


namespace sum_irr_not_rational_diff_irr_rational_prod_irr_rational_quot_irr_rational_l806_806272

-- Define irrational numbers
def irr1 := 7 + Real.sqrt 5
def irr2 := 3 + Real.sqrt 5
def irr3 := Real.sqrt 2
def irr4 := Real.sqrt 8
def irr5 := 5 + Real.sqrt 3
def irr6 := 2.5 + Real.sqrt 0.75

-- Conditions
def sum_irr := irr1 + irr2
def diff_irr := irr1 - irr2
def prod_irr := irr3 * irr4
def quot_irr := irr4 / irr3

-- Statements
theorem sum_irr_not_rational : ¬ Rational sum_irr :=
by sorry

theorem diff_irr_rational : Rational diff_irr :=
by sorry

theorem prod_irr_rational : Rational prod_irr :=
by sorry

theorem quot_irr_rational : Rational quot_irr :=
by sorry

end sum_irr_not_rational_diff_irr_rational_prod_irr_rational_quot_irr_rational_l806_806272


namespace valid_sequence_count_l806_806721

-- Define relevant properties of sequences
def is_valid_sequence (s : List Char) : Prop :=
  (∀ run in s.chunks, run.head = 'A' → run.length % 2 = 0) ∧ 
  (∀ run in s.chunks, run.head = 'B' → run.length % 2 = 1)

-- Define the sequence length
def sequence_length := 14

-- Main theorem to prove
theorem valid_sequence_count : 
  ∃ (count : ℕ), count = 144 ∧
  count = List.length (List.filter is_valid_sequence (List.replicateM sequence_length ['A', 'B'])) :=
sorry

end valid_sequence_count_l806_806721


namespace count_four_digit_numbers_with_5_or_7_l806_806402

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l806_806402


namespace algebraic_expression_value_l806_806343

theorem algebraic_expression_value (x y : ℝ) 
  (h : sqrt (x - 3) + y^2 - 4 * y + 4 = 0) :
  ( ( (x^2 - y^2) / (x * y) ) * (1 / (x^2 - 2 * x * y + y^2)) / (x / (x^2 * y - x * y^2)) - 1 ) = 2 / 3 :=
by
  -- Proof goes here
  sorry

end algebraic_expression_value_l806_806343


namespace probability_three_different_suits_l806_806057

noncomputable def pinochle_deck := 48
noncomputable def total_cards := 48
noncomputable def different_suits_probability := (36 / 47) * (23 / 46)

theorem probability_three_different_suits :
  different_suits_probability = 414 / 1081 :=
sorry

end probability_three_different_suits_l806_806057


namespace collinear_IOH_l806_806908

-- Definitions of the points and centers
variables {A B C A1 B1 C1 H I O : Type}
variables (in_touch_a1 : touches_incircle A1 A B C)
variables (in_touch_b1 : touches_incircle B1 B A C)
variables (in_touch_c1 : touches_incircle C1 C A B)
variables (orthocenter_h : is_orthocenter H A1 B1 C1)
variables (incenter_i : is_incenter I A B C)
variables (circumcenter_o : is_circumcenter O A B C)

theorem collinear_IOH :
  are_collinear I O H :=
sorry

end collinear_IOH_l806_806908


namespace ratio_w_y_l806_806166

open Real

theorem ratio_w_y (w x y z : ℝ) (h1 : w / x = 5 / 2) (h2 : y / z = 3 / 2) (h3 : z / x = 1 / 4) : w / y = 20 / 3 :=
by
  sorry

end ratio_w_y_l806_806166


namespace correct_cos_sum_l806_806293

noncomputable def cos_sum : ℂ :=
  ∑ k in Finset.range 46, (complex.I ^ k) * real.cos ((30 + 30 * k) * real.pi / 180)

theorem correct_cos_sum : cos_sum = complex.of_real (11 * real.sqrt 3 / 2) + (33 / 2) * complex.I :=
by
  sorry

end correct_cos_sum_l806_806293


namespace jen_problem_correct_answer_l806_806055

-- Definitions based on the conditions
def sum_178_269 : ℤ := 178 + 269
def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n - (n % 100) + 100 else n - (n % 100)

-- Prove the statement
theorem jen_problem_correct_answer :
  round_to_nearest_hundred sum_178_269 = 400 :=
by
  have h1 : sum_178_269 = 447 := rfl
  have h2 : round_to_nearest_hundred 447 = 400 := by sorry
  exact h2

end jen_problem_correct_answer_l806_806055


namespace trapezoid_area_l806_806075

theorem trapezoid_area
  (A B C D : Point)
  (h1 : isosceles_trapezoid A B C D)
  (h2 : distance A (line_through B C) = 15)
  (h3 : distance A (line_through C D) = 18)
  (h4 : distance A (line_through B D) = 10)
  (AD_eq_BC : segment_length A D = segment_length B C)
  (AB_lt_CD : segment_length A B < segment_length C D) :
  let K := trapezoid_area A B C D in
  √2 * K = 567 :=
sorry

end trapezoid_area_l806_806075


namespace count_four_digit_numbers_with_5_or_7_l806_806403

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l806_806403


namespace final_amount_after_5_years_l806_806100

-- Define conditions as hypotheses
def principal := 200
def final_amount_after_2_years := 260
def time_2_years := 2

-- Define our final question and answer as a Lean theorem
theorem final_amount_after_5_years : 
  (final_amount_after_2_years - principal) = principal * (rate * time_2_years) →
  (rate * 3) = 90 →
  final_amount_after_2_years + (principal * rate * 3) = 350 :=
by
  intros h1 h2
  -- Proof skipped using sorry
  sorry

end final_amount_after_5_years_l806_806100


namespace acai_berry_cost_correct_l806_806161

def cost_superfruit_per_litre : ℝ := 1399.45
def cost_mixed_fruit_per_litre : ℝ := 262.85
def litres_mixed_fruit : ℝ := 36
def litres_acai_berry : ℝ := 24
def total_litres : ℝ := litres_mixed_fruit + litres_acai_berry
def expected_cost_acai_per_litre : ℝ := 3104.77

theorem acai_berry_cost_correct :
  cost_superfruit_per_litre * total_litres -
  cost_mixed_fruit_per_litre * litres_mixed_fruit = 
  expected_cost_acai_per_litre * litres_acai_berry :=
by sorry

end acai_berry_cost_correct_l806_806161


namespace coordinates_P_on_curve_C_l806_806778

theorem coordinates_P_on_curve_C (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) :
    (3 * Real.cos θ, 4 * Real.sin θ) = (12 / 5, 12 / 5) ↔
    ∀ θ, Real.tan (π / 4) = 1 :=
by
  sorry

end coordinates_P_on_curve_C_l806_806778


namespace projection_magnitude_proof_l806_806807

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (3, 0)

def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

def projection_magnitude (a b : ℝ × ℝ) : ℝ :=
(dot_product a b) / (magnitude b)

theorem projection_magnitude_proof :
  projection_magnitude vector_a vector_b = -2 := 
by
  sorry

end projection_magnitude_proof_l806_806807


namespace count_four_digit_integers_with_5_or_7_l806_806410

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l806_806410


namespace measure_of_smaller_interior_angle_l806_806702

def is_congruent_isosceles_trapezoid (α β : ℝ) := α = β

theorem measure_of_smaller_interior_angle
  (n : ℕ) 
  (h1 : n = 8)
  (h2 : ∀ i, 1 ≤ i → i ≤ n → is_congruent_isosceles_trapezoid (interior_angle_adjacent_longer_base_of i) (interior_angle_adjacent_longer_base_of (i + 1)))
  (h3 : total_angle_sum_circle : ℝ := 360) :
  interior_angle_adjacent_longer_base_of (1) = 78.75 :=
by
  sorry

end measure_of_smaller_interior_angle_l806_806702
