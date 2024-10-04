import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.ProbTheory.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Vector.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace part1_M1_eq_part1_L1_eq_part2_Ma_subset_part2_exists_a_part3_even_iff_l630_630395

def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := x^3 - 3 * x^2

def M (f : ℝ → ℝ) (a : ℝ) : set ℝ := {t | ∃ x, t = f x - f a ∧ x ≥ a}
def L (f : ℝ → ℝ) (a : ℝ) : set ℝ := {t | ∃ x, t = f x - f a ∧ x ≤ a}

theorem part1_M1_eq : M f1 1 = set.Ici 0 := sorry

theorem part1_L1_eq : L f1 1 = set.Ici (-1) := sorry

theorem part2_Ma_subset : ∀ a : ℝ, M f2 a ⊆ set.Ici (-4) := sorry

theorem part2_exists_a : ∃ a : ℝ, -4 ∈ M f2 a := sorry

theorem part3_even_iff : 
  (∀ x : ℝ, f2 x = f2 (-x)) ↔ (∀ c : ℝ, 0 < c → M f2 (-c) = L f2 c) := sorry

end part1_M1_eq_part1_L1_eq_part2_Ma_subset_part2_exists_a_part3_even_iff_l630_630395


namespace limit_of_derivative_l630_630351

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

theorem limit_of_derivative (h : deriv f x₀ = 4) :
  filter.tendsto (λ Δx, (f (x₀ + 2 * Δx) - f x₀) / Δx) (nhds 0) (nhds 8) := by
  sorry

end limit_of_derivative_l630_630351


namespace arithmetic_series_sum_base8_l630_630710

-- Defining the series sum in base 8 (octal) format
def arith_sum_base8 : ℕ := 0o1 + 0o2 + 0o3 + ... + 0o30

-- Theorem to prove
theorem arithmetic_series_sum_base8 : arith_sum_base8 = 0o454 := by
  sorry

end arithmetic_series_sum_base8_l630_630710


namespace tip_travel_distance_l630_630337

def second_hand_distance (r : ℝ) (minutes : ℕ) : ℝ :=
  2 * real.pi * r * minutes

theorem tip_travel_distance (r : ℝ) (minutes : ℕ) (h₁ : r = 8) (h₂ : minutes = 45) :
  second_hand_distance r minutes = 720 * real.pi :=
by
  sorry

end tip_travel_distance_l630_630337


namespace total_oranges_l630_630429

def monday_oranges : ℕ := 100
def tuesday_oranges : ℕ := 3 * monday_oranges
def wednesday_oranges : ℕ := 70

theorem total_oranges : monday_oranges + tuesday_oranges + wednesday_oranges = 470 := by
  sorry

end total_oranges_l630_630429


namespace num_positive_int_values_l630_630719

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l630_630719


namespace max_intersection_points_of_two_rectangles_l630_630002

theorem max_intersection_points_of_two_rectangles 
  (rect1 rect2: Type)
  (A B: Type)
  [is_rectangle rect1]
  [is_rectangle rect2]
  [intersect_at_two_points rect1 rect2 A B]:
  ∃ n : ℕ, n = 8 :=
begin
  existsi 8,
  sorry
end

end max_intersection_points_of_two_rectangles_l630_630002


namespace find_my_age_l630_630568

noncomputable def age_my := I
noncomputable def age_great_grandfather := 4 * I
noncomputable def age_son_in_days := S * 7
noncomputable def age_grandson_in_days := age_son_in_days / 365
noncomputable def age_grandson_in_months := I * 12
noncomputable def age_granddaughter := (S / 2) / 52
noncomputable def age_son_in_years := S / 52

theorem find_my_age (I S : ℝ) (h1 : age_grandson_in_days = age_grandson_in_months)
  (h2 :  age_my + age_son_in_years + age_great_grandfather + age_granddaughter + age_my = 240) :
  I = 32 :=
by 
  -- sorry is a placeholder for proof
  sorry

end find_my_age_l630_630568


namespace car_speed_l630_630200

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_l630_630200


namespace number_of_functions_l630_630328

-- Set up the elements of the sets A and B
inductive A : Type
| a | b | c | d | e

inductive B : Type
| one | two | three

-- Define the function type and required properties
def f (x : A) : B

-- Define the conditions as functions
def is_range_B (f : A → B) : Prop :=
  ∀ y : B, ∃ x : A, f x = y

def f_not_equal (f : A → B) : Prop :=
  f A.a ≠ f A.b

-- State the theorem
theorem number_of_functions : 
  ∃ (n : ℕ), n = 114 ∧ ∃ f : (A → B), is_range_B f ∧ f_not_equal f := sorry

end number_of_functions_l630_630328


namespace amount_left_in_cookie_jar_l630_630498

def original_amount : ℕ := 21
def doris_spent : ℕ := 6
def martha_spent (doris: ℕ) : ℕ := doris / 2
def total_spent (doris martha: ℕ) : ℕ := doris + martha

theorem amount_left_in_cookie_jar :
  let doris := doris_spent in
  let martha := martha_spent doris in
  let total := total_spent doris martha in
  original_amount - total = 12 :=
by
  sorry

end amount_left_in_cookie_jar_l630_630498


namespace area_of_triangle_OPA_l630_630765

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)).abs

theorem area_of_triangle_OPA :
  ∀ (m : ℝ), (y = (-2) * x + 4) ∧ (1, m) lies on the line ∧ 
  (2, 0) lies on the line → 
  triangle_area (0, 0) (1, 2) (2, 0) = 2 :=
by
  sorry

end area_of_triangle_OPA_l630_630765


namespace toy_cost_price_l630_630963

theorem toy_cost_price (x : ℝ) (s : ℝ) (g : ℝ) : 
  (s = 16800) ∧ (g = x * 3) ∧ (s / 18 = x + g / 18) → x = 800 :=
by
  intros h
  obtain ⟨hs, hg, heq⟩ := h
  have := calc
    s / 18 = 16800 / 18 : by rw hs
           ... = x + 3 * x / 18 : by rw [hg, ← mul_assoc, mul_div_cancel']
  sorry

end toy_cost_price_l630_630963


namespace circular_garden_radius_l630_630962

theorem circular_garden_radius
  (r : ℝ) -- radius of the circular garden
  (h : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) :
  r = 12 := 
by {
  sorry
}

end circular_garden_radius_l630_630962


namespace total_volume_correct_l630_630465

-- Define the properties of the cone
def cone_radius : ℝ := 3
def cone_height : ℝ := 12

-- Define the properties of the hemisphere
def hemisphere_radius : ℝ := 2

-- Define the volume of the cone
def volume_cone : ℝ := (1 / 3) * Real.pi * cone_radius^2 * cone_height

-- Define the volume of the hemisphere
def volume_hemisphere : ℝ := (2 / 3) * Real.pi * hemisphere_radius^3

-- Total volume calculation
def total_volume : ℝ := volume_cone + volume_hemisphere

-- Prove that the total volume is equal to the given value
theorem total_volume_correct : total_volume = 41.33 * Real.pi := by
  -- This will contain the proof, which is currently omitted.
  sorry

end total_volume_correct_l630_630465


namespace circle_standard_equation_l630_630275

theorem circle_standard_equation (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y - 1)^2) ∧
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y + 2)^2) →
  (∃ x y : ℝ, (x - 2) ^ 2 + (y - 1) ^ 2 = 2) :=
sorry

end circle_standard_equation_l630_630275


namespace saroj_age_proof_l630_630585

def saroj_present_age (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : ℕ :=
  sorry    -- calculation logic would be here but is not needed per instruction

noncomputable def question_conditions (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : Prop :=
  vimal_age_6_years_ago / 6 = saroj_age_6_years_ago / 5 ∧
  (vimal_age_6_years_ago + 10) / 11 = (saroj_age_6_years_ago + 10) / 10 ∧
  saroj_present_age vimal_age_6_years_ago saroj_age_6_years_ago = 16

theorem saroj_age_proof (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) :
  question_conditions vimal_age_6_years_ago saroj_age_6_years_ago :=
  sorry

end saroj_age_proof_l630_630585


namespace op_two_four_l630_630859

def op (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem op_two_four : op 2 4 = 18 := by
  sorry

end op_two_four_l630_630859


namespace base_2_representation_of_123_is_1111011_l630_630527

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630527


namespace valid_moves_for_black_cells_l630_630586

theorem valid_moves_for_black_cells (n : ℕ) : 
  (∀ (table : Fin 4 × Fin 4 → bool), 
   (∀ (i j : Fin 4), table (i, j) = false) → 
   (∀ (move : Fin 4 × Fin 4), 
    let neighbors := [(move.1 + 1, move.2), (move.1 - 1, move.2), (move.1, move.2 + 1), (move.1, move.2 - 1)]
    ∀ (m : Fin 4 × Fin 4), m ∈ neighbors → 
    table m := !table m) → 
   (∀ (table' : Fin 4 × Fin 4 → bool), 
    (table' = fun (i, j) => true) → 
    ∃ (moves : ℕ), moves = n ∧ moves % 2 = 0 ∧ moves ≥ 6) 
→ ∃ (k : ℕ), k = 2 * n ∧ k ≥ 6.

end valid_moves_for_black_cells_l630_630586


namespace num_positive_int_values_l630_630717

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l630_630717


namespace range_of_a_l630_630811

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l630_630811


namespace complex_division_example_l630_630096

theorem complex_division_example : (2 - (1 : ℂ) * Complex.I) / (1 - (1 : ℂ) * Complex.I) = (3 / 2) + (1 / 2) * Complex.I :=
by
  sorry

end complex_division_example_l630_630096


namespace pow_mod_remainder_l630_630173

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l630_630173


namespace p_plus_q_eq_neg_half_plus_sqrt3_over_2_i_l630_630459

theorem p_plus_q_eq_neg_half_plus_sqrt3_over_2_i
  (p q : ℂ)
  (h_eq : ∀ x : ℂ, x^3 + p * x + q = 0)
  (h_roots : ∃ (z1 z2 z3 : ℂ), z1 ≠ z2 ∧ z2 ≠ z3 ∧ z3 ≠ z1 ∧
                                ∃ (a : ℂ), a = (z2 - z1) ∧ a * a = 3 ∧
                                ∃ (b : ℂ), b = (z3 - z2) ∧ b * b = 3 ∧
                                ∃ (c : ℂ), c = (z1 - z3) ∧ c * c = 3)
  (h_arg_q : q.arg = 2 * Real.pi / 3) :
  p + q = -1 / 2 + Complex.i * (Real.sqrt 3 / 2) := by
  sorry

end p_plus_q_eq_neg_half_plus_sqrt3_over_2_i_l630_630459


namespace isosceles_triangle_area_l630_630091

theorem isosceles_triangle_area (altitude perimeter : ℝ) (ratio : ℝ) 
  (h1 : altitude = 10) 
  (h2 : perimeter = 40) 
  (h3 : ratio = 5/3) : 
  (area : ℝ) := 
begin
  /- Definitions for the triangle dimensions based on the given ratio -/
  let b := 3/8 * perimeter,
  let s := 5/3 * b,
  have base : base = 2 * b,
  have perimeter_eq : 2 * s + 2 * b = perimeter := by linarith,
  have area : area = b * altitude := by linarith,
  exact area = 75 sorry
end

end isosceles_triangle_area_l630_630091


namespace trapezoid_perimeter_l630_630007

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h1 : AB = 5)
  (h2 : CD = 5)
  (h3 : AD = 16)
  (h4 : BC = 8) :
  AB + BC + CD + AD = 34 :=
by
  sorry

end trapezoid_perimeter_l630_630007


namespace num_positive_int_values_l630_630718

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l630_630718


namespace order_of_eq_1_order_of_eq_2_order_of_eq_3_order_of_eq_4_l630_630249

-- Prove that the order of the differential equation y'' - 3y' + 2y - 4 = 0 is 2
theorem order_of_eq_1 : ∀ (y : ℝ → ℝ), 
  (∀ x, deriv 2 (y x) - 3 * deriv 1 (y x) + 2 * y x - 4 = 0) → true :=
by sorry

-- Prove that the order of the differential equation x(1 + x)y' - (1 + 2x)y - (1 + 2x) = 0 is 1
theorem order_of_eq_2 : ∀ (y : ℝ → ℝ), 
  (∀ x, x * (1 + x) * deriv 1 (y x) - (1 + 2 * x) * y x - (1 + 2 * x) = 0) → true :=
by sorry

-- Prove that the order of the differential equation y^{IV} - 16y'' = 0 is 4
theorem order_of_eq_3 : ∀ (y : ℝ → ℝ), 
  (∀ x, deriv 4 (y x) - 16 * deriv 2 (y x) = 0) → true :=
by sorry

-- Prove that the order of the differential equation y''' - 6y'' + 11y' - 6y = 0 is 3
theorem order_of_eq_4 : ∀ (y : ℝ → ℝ), 
  (∀ x, deriv 3 (y x) - 6 * deriv 2 (y x) + 11 * deriv 1 (y x) - 6 * y x = 0) → true :=
by sorry

end order_of_eq_1_order_of_eq_2_order_of_eq_3_order_of_eq_4_l630_630249


namespace find_f_2010_l630_630103

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f x + f (3 - x)

theorem find_f_2010 : f 2010 = 0 := sorry

end find_f_2010_l630_630103


namespace min_period_cosine_l630_630106

theorem min_period_cosine :
  ∀ (x : ℝ), (∃ T > 0, (∀ y : ℝ, 3 * cos ((2 / 5) * (x + T) - π / 6) = 3 * cos ((2 / 5) * x - π / 6))) ∧
  (∀ T' > 0, (∀ y : ℝ, 3 * cos ((2 / 5) * (x + T') - π / 6) = 3 * cos ((2 / 5) * x - π / 6)) → T' ≥ 5 * π) :=
sorry

end min_period_cosine_l630_630106


namespace find_b_l630_630307

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 :=
sorry

end find_b_l630_630307


namespace remainder_of_3_pow_2023_mod_5_l630_630168

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l630_630168


namespace exists_consecutive_integers_not_sum_of_two_squares_l630_630685

open Nat

theorem exists_consecutive_integers_not_sum_of_two_squares : 
  ∃ (m : ℕ), ∀ k : ℕ, k < 2017 → ¬(∃ a b : ℤ, (m + k) = a^2 + b^2) := 
sorry

end exists_consecutive_integers_not_sum_of_two_squares_l630_630685


namespace tank_capacity_is_24_l630_630608

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end tank_capacity_is_24_l630_630608


namespace actual_percent_profit_is_35_l630_630999

-- Definitions for conditions
variables (CP LP SP : ℝ)

-- Condition 1: Label price for 50% profit on cost price
def labeled_price (CP : ℝ) : ℝ := CP + 0.5 * CP

-- Condition 2: Selling price after 10% discount on labeled price
def selling_price (LP : ℝ) : ℝ := LP - 0.1 * LP

-- Actual profit calculation
def profit (SP CP : ℝ) : ℝ := SP - CP

-- Actual percent profit calculation
def percent_profit (Profit CP : ℝ) : ℝ := (Profit / CP) * 100

-- Theorem to prove
theorem actual_percent_profit_is_35 (hCP : CP = 100) :
  percent_profit (profit (selling_price (labeled_price CP)) CP) CP = 35 :=
by
  sorry

end actual_percent_profit_is_35_l630_630999


namespace remainder_when_divided_by_x_minus_2_l630_630708

def p (x : ℤ) : ℤ := x^5 + x^3 + x + 3

theorem remainder_when_divided_by_x_minus_2 :
  p 2 = 45 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l630_630708


namespace map_point_to_result_l630_630472

def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem map_point_to_result :
  f 2 0 = (2, 2) :=
by
  unfold f
  simp

end map_point_to_result_l630_630472


namespace sum_of_odd_indices_up_to_1005_with_condition_l630_630401

noncomputable def sequence_a : ℕ → ℝ
| 1 := 0.3
| 2 := (0.301) ^ sequence_a 1
| n := if even n then (0: ℝ) + list.repeat 0 n |>.foldl (λ acc x => acc + x) (0.301).succ_pow (λ k => sequence_a (n - 1))
                       else (0: ℝ) + list.repeat 0 (n + 1) |>.foldl (λ acc x => acc + x) (0.301).succ_pow (λ k => sequence_a (n - 1))

theorem sum_of_odd_indices_up_to_1005_with_condition :
  ∑ k in finset.filter (λ k, sequence_a k = sequence_a k) (finset.range 1005).filter (odd) (finset.range 1005) = 253009 :=
sorry

end sum_of_odd_indices_up_to_1005_with_condition_l630_630401


namespace girl_from_grade_4_probability_l630_630822

-- Number of girls and boys in grade 3
def girls_grade_3 := 28
def boys_grade_3 := 35
def total_grade_3 := girls_grade_3 + boys_grade_3

-- Number of girls and boys in grade 4
def girls_grade_4 := 45
def boys_grade_4 := 42
def total_grade_4 := girls_grade_4 + boys_grade_4

-- Number of girls and boys in grade 5
def girls_grade_5 := 38
def boys_grade_5 := 51
def total_grade_5 := girls_grade_5 + boys_grade_5

-- Total number of children in playground
def total_children := total_grade_3 + total_grade_4 + total_grade_5

-- Probability that a randomly selected child is a girl from grade 4
def probability_girl_grade_4 := (girls_grade_4: ℚ) / total_children

theorem girl_from_grade_4_probability :
  probability_girl_grade_4 = 45 / 239 := by
  sorry

end girl_from_grade_4_probability_l630_630822


namespace area_of_triangle_in_polar_coordinates_l630_630005

theorem area_of_triangle_in_polar_coordinates :
  ∀ (r1 θ1 r2 θ2 : ℝ), 
  r1 = 6 → θ1 = π / 3 → r2 = 4 → θ2 = π / 6 →
  (1 / 2) * r1 * r2 * Real.sin (θ2 - θ1) = 6 := by {
  intros r1 θ1 r2 θ2 hr1 hθ1 hr2 hθ2,
  rw [hr1, hθ1, hr2, hθ2],
  norm_num,
  exact Real.sin_pi_div_two,
}


end area_of_triangle_in_polar_coordinates_l630_630005


namespace bodyweight_gain_is_8_l630_630664

-- Define the initial conditions
def initial_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def gain_percentage : ℝ := 0.15
def ratio : ℝ := 10

-- Define the new total based on the 15% gain
def new_total : ℝ := initial_total + (gain_percentage * initial_total)

-- Define the relationship between new total and new bodyweight
def new_bodyweight : ℝ := new_total / ratio

-- Compute the bodyweight gain
def bodyweight_gain : ℝ := new_bodyweight - initial_bodyweight

-- Theorem stating the bodyweight gain is 8 pounds
theorem bodyweight_gain_is_8 : bodyweight_gain = 8 := by sorry

end bodyweight_gain_is_8_l630_630664


namespace rectangular_prism_prob_l630_630625

-- Define the rectangular prism dimensions
def length : ℕ := 2
def width : ℕ := 3
def height : ℕ := 4

-- Define the total number of vertices in a rectangular prism
def vertices := length * width * height

-- Define a combinatorial function to calculate combinations
def choose (n k : ℕ) : ℕ :=
  if k > n then 0 else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_plane_contains_points_inside (num_vertices : ℕ) (correct_prob : ℚ) : Prop :=
  choose num_vertices 3 * (1 - (24 / 56)) = correct_prob

theorem rectangular_prism_prob :
  prob_plane_contains_points_inside 8 (4 / 7) := 
sorry

end rectangular_prism_prob_l630_630625


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630152

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630152


namespace vector_linear_combination_parallel_l630_630332

open vector

theorem vector_linear_combination_parallel
  (a₁ a₂ : ℝ) (b₁ b₂ : ℝ) (m : ℝ)
  (h₁ : (a₁, a₂) = (1, 2))
  (h₂ : (b₁, b₂) = (-2, m))
  (h₃ : a₁ * b₂ = a₂ * b₁) : 
  2 • (a₁, a₂) + 3 • (b₁, b₂) = (-4, -8) := 
by
  -- Defining the conditions
  rw [h₁, h₂] at h₃
  have m_eq : m = -4,
  { -- Deriving m from parallelism
    linarith },

  -- Simplifying the vectors using the determined value of m
  rw [h₁, h₂, m_eq],
  
  -- Calculating the vector addition
  simp,
  sorry


end vector_linear_combination_parallel_l630_630332


namespace right_triangle_area_l630_630944

theorem right_triangle_area (a c : ℝ) (h : a = 28) (h2 : c = 30) (h3 : c^2 = a^2 + (b : ℝ)^2):
  (1/2 * a * b = 28 * real.sqrt 29) :=
by
  sorry

end right_triangle_area_l630_630944


namespace altered_solution_contains_120_liters_of_detergent_l630_630477

theorem altered_solution_contains_120_liters_of_detergent
  (b d w : ℕ) (h₁ : 2 * d = 40 * b) (h₂ : 40 * w = 100 * d) (h₃ : w = 300) : d = 120 :=
by
  have h₄ : 6 * d = 240 * b,
  sorry

end altered_solution_contains_120_liters_of_detergent_l630_630477


namespace most_people_can_attend_each_day_l630_630687

def attendance_data : Type :=
  string × string

-- Defining the unavailability data for each day
def unavailability : list attendance_data :=
  [("Anna", "Mon"), ("Anna", "Wed"), ("Anna", "Fri"),
   ("Bill", "Tues"), ("Bill", "Thurs"),
   ("Carl", "Mon"), ("Carl", "Tues"), ("Carl", "Thurs"), ("Carl", "Fri"),
   ("Dana", "Wed")]

-- Define the days of the week
def days : list string := ["Mon", "Tues", "Wed", "Thurs", "Fri"]

-- Define the people
def people : list string := ["Anna", "Bill", "Carl", "Dana"]

-- A function to count the 'X's (unavailabilities) for a given day
def count_unavailabilities (day : string) : ℕ :=
  unavailability.filter (λ x, x.2 = day).length

-- Predicate for whether at least two people can attend a meeting on a given day
def at_least_two_can_attend (day : string) : Prop :=
  (people.length - count_unavailabilities day) ≥ 2

-- Prove that at least two people can attend on each day of the week
theorem most_people_can_attend_each_day :
  ∀ day ∈ days, at_least_two_can_attend day :=
by
  intros day hday
  have : ∀ day ∈ days, count_unavailabilities day ≤ 2 := sorry
  apply nat.sub_le_sub_right _ 2
  exact this day hday

end most_people_can_attend_each_day_l630_630687


namespace train_length_proof_l630_630616

-- Define the conditions as hypotheses
def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def jogger_head_start_m : ℝ := 240
def time_pass_jogger_s : ℝ := 45

-- Convert speeds to m/s
def kmh_to_ms (kmh : ℝ) : ℝ := (kmh * 1000) / (60 * 60)

def jogger_speed_ms : ℝ := kmh_to_ms jogger_speed_kmh
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh
def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms

-- Calculate the length of the train
def length_of_train : ℝ :=
  (relative_speed_ms * time_pass_jogger_s) - jogger_head_start_m

-- The theorem statement
theorem train_length_proof :
  length_of_train = 210 :=
by
  sorry

end train_length_proof_l630_630616


namespace power_mod_l630_630161

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l630_630161


namespace parallelepiped_inequality_l630_630391

noncomputable theory
open_locale classical

variable {a b c : ℝ}

theorem parallelepiped_inequality (a b c : ℝ) :
  real.sqrt (a^2 + b^2) + real.sqrt (b^2 + c^2) + real.sqrt (c^2 + a^2) 
  ≤ a + b + c + real.sqrt (a^2 + b^2 + c^2) :=
sorry

end parallelepiped_inequality_l630_630391


namespace discount_percentages_are_4_and_6_l630_630984

-- Definitions of the conditions
def original_price : ℕ := 21250
def final_price : ℕ := 19176

-- The discounts are single-digit percentages
def single_digit_discount (x : ℕ) : Prop := x < 10

-- The main statement we want to prove
theorem discount_percentages_are_4_and_6 :
  ∃ p q : ℕ, single_digit_discount p ∧ single_digit_discount q ∧
    original_price * (1 - p * 0.01) * (1 - q * 0.01) = final_price ∧
    (p = 4 ∧ q = 6 ∨ p = 6 ∧ q = 4) :=
by
  sorry

end discount_percentages_are_4_and_6_l630_630984


namespace balls_in_boxes_l630_630342

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end balls_in_boxes_l630_630342


namespace sum_first_10_terms_l630_630763

-- Define the conditions for the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_sequence (b c d : ℝ) : Prop :=
  2 * c = b + d

def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧
  geometric_sequence a q ∧
  arithmetic_sequence (4 * a 1) (2 * a 2) (a 3)

-- Define the sum of the first n terms of a geometric sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Prove the final result
theorem sum_first_10_terms (a : ℕ → ℝ) (q : ℝ) (h : conditions a q) :
  sum_first_n_terms a 10 = 1023 :=
sorry

end sum_first_10_terms_l630_630763


namespace fraction_of_sum_after_6_years_l630_630635

-- Define the principal amount, rate, and time period as given in the conditions
def P : ℝ := 1
def R : ℝ := 0.02777777777777779
def T : ℕ := 6

-- Definition of the Simple Interest calculation
def simple_interest (P R : ℝ) (T : ℕ) : ℝ :=
  P * R * T

-- Definition of the total amount after 6 years
def total_amount (P SI : ℝ) : ℝ :=
  P + SI

-- The main theorem to prove
theorem fraction_of_sum_after_6_years :
  total_amount P (simple_interest P R T) = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_after_6_years_l630_630635


namespace health_risk_probability_l630_630819

theorem health_risk_probability :
  let a := 0.08 * 500
  let b := 0.08 * 500
  let c := 0.08 * 500
  let d := 0.18 * 500
  let e := 0.18 * 500
  let f := 0.18 * 500
  let g := 0.05 * 500
  let h := 500 - (3 * 40 + 3 * 90 + 25)
  let q := 500 - (a + d + e + g)
  let p := 1
  let q := 3
  p + q = 4 := sorry

end health_risk_probability_l630_630819


namespace rest_duration_per_kilometer_l630_630994

theorem rest_duration_per_kilometer
  (speed : ℕ)
  (total_distance : ℕ)
  (total_time : ℕ)
  (walking_time : ℕ := total_distance / speed * 60)  -- walking_time in minutes
  (rest_time : ℕ := total_time - walking_time)  -- total resting time in minutes
  (number_of_rests : ℕ := total_distance - 1)  -- number of rests after each kilometer
  (duration_per_rest : ℕ := rest_time / number_of_rests)
  (h1 : speed = 10)
  (h2 : total_distance = 5)
  (h3 : total_time = 50) : 
  (duration_per_rest = 5) := 
sorry

end rest_duration_per_kilometer_l630_630994


namespace transportation_cost_l630_630098

-- Definitions for the conditions
def number_of_original_bags : ℕ := 80
def weight_of_original_bag : ℕ := 50
def total_cost_original : ℕ := 6000

def scale_factor_bags : ℕ := 3
def scale_factor_weight : ℚ := 3 / 5

-- Derived quantities
def number_of_new_bags : ℕ := scale_factor_bags * number_of_original_bags
def weight_of_new_bag : ℚ := scale_factor_weight * weight_of_original_bag
def cost_per_original_bag : ℚ := total_cost_original / number_of_original_bags
def cost_per_new_bag : ℚ := cost_per_original_bag * (weight_of_new_bag / weight_of_original_bag)

-- Final cost calculation
def total_cost_new : ℚ := number_of_new_bags * cost_per_new_bag

-- The statement that needs to be proved
theorem transportation_cost : total_cost_new = 10800 := sorry

end transportation_cost_l630_630098


namespace polynomial_simplification_l630_630507

noncomputable def given_polynomial (x : ℝ) : ℝ :=
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 + 2 * x^3

theorem polynomial_simplification (x : ℝ) :
  given_polynomial x = 2 * x^3 - x^2 - 11 * x + 27 :=
by
  -- The proof is skipped
  sorry

end polynomial_simplification_l630_630507


namespace rhombus_area_l630_630912

-- Declare the lengths of the diagonals
def diagonal1 := 6
def diagonal2 := 8

-- Define the area function for a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

-- State the theorem
theorem rhombus_area : area_of_rhombus diagonal1 diagonal2 = 24 := by sorry

end rhombus_area_l630_630912


namespace reciprocal_of_neg_2023_l630_630483

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l630_630483


namespace prove_n_eq_one_l630_630333

-- Definitions of the vectors a and b
def vector_a (n : ℝ) : ℝ × ℝ := (1, n)
def vector_b (n : ℝ) : ℝ × ℝ := (-1, n - 2)

-- Definition of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem to prove that if a and b are collinear, then n = 1
theorem prove_n_eq_one (n : ℝ) (h_collinear : collinear (vector_a n) (vector_b n)) : n = 1 :=
sorry

end prove_n_eq_one_l630_630333


namespace odd_sum_combinations_l630_630736

theorem odd_sum_combinations :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  (∃ (n : ℕ), n ∈ S ∧ ∃ (m : ℕ), m ∈ S ∧ (n + m) % 2 = 1) = 25 :=
sorry

end odd_sum_combinations_l630_630736


namespace bear_a_total_time_l630_630663

variables (S V_A V_B k : ℝ)

-- Conditions from the problem
axiom (h1 : V_A = k * V_B)
axiom (h2 : 1 < k)
axiom (h3 : ∀ V_B > 0, 2 * V_A + 2 * V_B = S - 1600)
axiom (h4 : ∀ V_B > 0, S = ?m1)
axiom (h5 : ∀ T t_meet : ℝ, T = (S - 1600) / (2 * (V_A + V_B)))
axiom (h6 : ∀ t_peak t_half t_return : ℝ, V_B <= T * 1600  ∧ 11200/6000 +11200/(6000 * 2) = 2.8 + 1.4)

theorem bear_a_total_time (V_A : ℝ): 
  (∃ k : ℝ, k * V_B = V_A ∧ 1 < k ) →
  (∃ T t_peak t_half T: ℝ, T = (S-1600)/(2*(V_A + V_B))) → 
  (V_A / V_B  > T from 2.8 travel total):
 sorry
 
end bear_a_total_time_l630_630663


namespace horner_evaluation_at_3_l630_630940

def f (x : ℕ) : ℕ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x
def horner_value (x : ℕ) : ℕ := ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x

theorem horner_evaluation_at_3 : f 3 = 21324 :=
by
  have h0 : f 3 = horner_value 3, by sorry
  have h1 : horner_value 3 = 21324, by sorry
  rw [h0, h1]

end horner_evaluation_at_3_l630_630940


namespace largest_c_for_range_l630_630270

noncomputable def g (x c : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_range (c : ℝ) : (∃ x : ℝ, g x c = 2) ↔ c ≤ 11 := 
sorry

end largest_c_for_range_l630_630270


namespace distance_A_to_B_l630_630620

theorem distance_A_to_B : 
  ∀ (D : ℕ),
    let boat_speed_with_wind := 21
    let boat_speed_against_wind := 17
    let time_for_round_trip := 7
    let stream_speed_ab := 3
    let stream_speed_ba := 2
    let effective_speed_ab := boat_speed_with_wind + stream_speed_ab
    let effective_speed_ba := boat_speed_against_wind - stream_speed_ba
    D / effective_speed_ab + D / effective_speed_ba = time_for_round_trip →
    D = 65 :=
by
  sorry

end distance_A_to_B_l630_630620


namespace counter_position_after_1234_moves_l630_630894

theorem counter_position_after_1234_moves :
  let move (n : ℕ) : ℕ := n^n
  let position (start : ℕ) (moves : ℕ → ℕ) (n : ℕ) : ℕ :=
    List.foldl (λ pos move => (pos + move) % 10) start (List.range n).map moves
  position 0 move 1234 = 7 := 
sorry

end counter_position_after_1234_moves_l630_630894


namespace verify_exercise4_inequality_l630_630141

variable {n : ℕ} {a : ℕ → ℝ}

theorem verify_exercise4_inequality (h : 0 < n) : 
  Real.sqrt (∑ i in Finset.range n, a i ^ 2 / n) ≥ (∑ i in Finset.range n, a i) / n :=
  sorry

end verify_exercise4_inequality_l630_630141


namespace grasshopper_approx_l630_630987

noncomputable def f : ℝ → ℝ := λ x, x / Real.sqrt 3
noncomputable def g : ℝ → ℝ := λ x, x / Real.sqrt 3 + (1 - 1 / Real.sqrt 3)

theorem grasshopper_approx (a : ℝ) (h_a : 0 ≤ a ∧ a ≤ 1) (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1) : 
  ∃ n : ℕ, ∃ h : Vector (ℝ → ℝ) n, 
    (∀ i, h.to_list.nth i = some f ∨ h.to_list.nth i = some g) ∧ 
    ∃ y, (foldl (λ b hf, hf b) x h.to_list = y) ∧ 
    abs (y - a) < 0.01 := 
sorry

end grasshopper_approx_l630_630987


namespace number_of_topless_cubical_box_figures_l630_630453

def valid_placements : Finset (Finset Nat) :=
Finset.fromList [{1, 2}, {5, 6}]

def number_of_valid_placements : Finset (Finset Nat) :=
valid_placements

theorem number_of_topless_cubical_box_figures :
  number_of_valid_placements.card = 2 := by
  sorry

end number_of_topless_cubical_box_figures_l630_630453


namespace decimal_to_binary_equivalent_123_l630_630533

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630533


namespace area_of_triangle_ABC_l630_630136

noncomputable def triangle_area : ℝ :=
  let A : RealAngle := RealAngle.ofDeg 60
  let B : RealAngle := RealAngle.ofDeg 30
  let r : ℝ := 3
  let area_incircle : ℝ := 9 * Real.pi
  let hypotenuse_length := 2 * r -- Corresponds to BC
  let base_length := r * Real.sqrt 3 -- Corresponds to AC
  (1/2) * hypotenuse_length * base_length

theorem area_of_triangle_ABC 
  (A ∠ = RealAngle.ofDeg 60)
  (B ∠ = RealAngle.ofDeg 30)
  (r = 3)
  (area_incircle = 9 * Real.pi) : triangle_area = 9 * Real.sqrt 3 :=
by sorry

end area_of_triangle_ABC_l630_630136


namespace socorro_training_hours_l630_630892

theorem socorro_training_hours :
  let daily_multiplication_time := 10  -- in minutes
  let daily_division_time := 20        -- in minutes
  let training_days := 10              -- in days
  let minutes_per_hour := 60           -- minutes in an hour
  let daily_total_time := daily_multiplication_time + daily_division_time
  let total_training_time := daily_total_time * training_days
  total_training_time / minutes_per_hour = 5 :=
by sorry

end socorro_training_hours_l630_630892


namespace min_distinct_sums_max_distinct_sums_l630_630287

theorem min_distinct_sums (n : ℕ) (h : 0 < n) : ∃ a b, (a + (n - 1) * b) = (n * (n + 1)) / 2 := sorry

theorem max_distinct_sums (n : ℕ) (h : 0 < n) : 
  ∃ m, m = 2^n - 1 := sorry

end min_distinct_sums_max_distinct_sums_l630_630287


namespace intersection_M_N_l630_630755

def M : Set ℝ := { x | x ≤ 4 }
def N : Set ℝ := { x | 0 < x }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x ≤ 4 } := 
by 
  sorry

end intersection_M_N_l630_630755


namespace joey_age_sum_of_digits_l630_630844

theorem joey_age_sum_of_digits
  (J M T : ℕ)
  (hJ : J = M + 2)
  (hT : T = 2)
  (h_mia_multiples : ∃ n, ∀ i ∈ (Finset.range 6).succ, (M + i) % (T + i) = 0) :
  let n := 12 in
  digit_sum (J + n) = 10 :=
by 
  have hM : M = 14 := sorry,
  have hJ' : J = 16 := by { rw hJ, exact (by norm_num : 14 + 2 = 16) },
  have hn : n = 12 := by { simp, exact (by norm_num : Finset.range 6).card },
  sorry

end joey_age_sum_of_digits_l630_630844


namespace not_possible_l630_630584

-- Define the types for knights and liars
inductive Person
  | knight : Person
  | liar : Person

-- Assume each person said, "Both of my neighbors are liars."
def said_neighbors_are_liars (p1 p2 p3 : Person) : Prop :=
  match p2 with
  | Person.knight => p3 = Person.liar ∧ p1 = Person.liar
  | Person.liar => ¬(p3 = Person.liar ∧ p1 = Person.liar)
  end

-- Theorem statement: Not possible for remaining to claim, "Both neighbors are knights" after one leaves
theorem not_possible (people : List Person) 
  (hlength : people.length = 10)
  (hstatements : ∀ (n : ℕ), n < 10 → said_neighbors_are_liars 
    (people.get! ((n + 9) % 10))
    (people.get! n)
    (people.get! ((n + 1) % 10))) :
  ¬(∃ p : List Person, 
    (p.length = 9) ∧ ∀ (m : ℕ), m < 9 → 
    (match p.get! m with
     | Person.knight => 
       (p.get! ((m + 8) % 9) = Person.knight) ∧ (p.get! ((m + 1) % 9) = Person.knight)
     | _ => false
     end)) :=
  sorry

end not_possible_l630_630584


namespace AM_AN_divide_BD_l630_630884

variable (A B C D M N P Q O : Type)
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables [AffineSpace ℝ M] [AffineSpace ℝ N] [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ O]
variables [Parallelogram A B C D]

-- Conditions
def is_midpoint (P Q R : Type) [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ R] : Prop :=
  dist P Q = dist Q R

noncomputable def M_midpoint_BC : Prop := is_midpoint B M C
noncomputable def N_midpoint_CD : Prop := is_midpoint C N D
noncomputable def O_intersection_AC_BD : Prop := ∃ O, segment (A ↔ C) ∩ segment (B ↔ D) = {O}

-- Question: Prove that the lines split the diagonal into three equal parts
theorem AM_AN_divide_BD (A B C D M N P Q O : Type)
  [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
  [AffineSpace ℝ M] [AffineSpace ℝ N] [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ O]
  [Parallelogram A B C D]
  (H1 : M_midpoint_BC A B C M)
  (H2 : N_midpoint_CD A C D N)
  (H3 : O_intersection_AC_BD A O B C D):
  divides_into_three_equal_parts AM AN BD := sorry

end AM_AN_divide_BD_l630_630884


namespace no_infinitely_many_polynomials_l630_630942

def is_interesting (P : Polynomial ℚ) : Prop :=
  ∃ (n : Finset ℕ), n.card = 1398 ∧ P = (∑ i in n, (Polynomial.C (1 : ℚ) * Polynomial.X ^ (i : ℕ))) + Polynomial.C 1

theorem no_infinitely_many_polynomials :
  ¬∃ (S : Set (Polynomial ℤ)), S.Infinite ∧
    ∀ (P Q : Polynomial ℤ), P ∈ S → Q ∈ S → P ≠ Q → is_interesting (P * Q) :=
by
  sorry

end no_infinitely_many_polynomials_l630_630942


namespace man_speed_correct_l630_630637

-- Definitions based on the problem conditions
def train_length : ℝ := 100 -- in meters
def train_speed_kmph : ℝ := 68 -- in kmph

-- Conversion factor from kmph to m/s
def mph_to_mps : ℝ := 1000 / 3600

-- Train speed in m/s
def train_speed_mps : ℝ := train_speed_kmph * mph_to_mps

-- Time taken to pass the man in seconds
def passing_time : ℝ := 5.999520038396929

-- Relative speed of train w.r.t man calculated as distance / time
def relative_speed_mps : ℝ := train_length / passing_time

-- The speed of the man in m/s
def man_speed_mps : ℝ := train_speed_mps - relative_speed_mps

-- Conversion factor from m/s to kmph
def mps_to_kmph : ℝ := 3600 / 1000

-- The speed of the man in km/h
def man_speed_kmph : ℝ := man_speed_mps * mps_to_kmph

theorem man_speed_correct :
  man_speed_kmph = 7.9988 :=
by
  -- all calculations have been set up as definitions, just need to assert equality
  sorry

end man_speed_correct_l630_630637


namespace find_min_value_of_fraction_l630_630777

noncomputable def f (m n x : ℝ) : ℝ := m * x^2 + (n - 1) * x + 2

theorem find_min_value_of_fraction (m n : ℝ) (h1 : m > 0) (h2 : n > 0) 
(h3 : ∀ x : ℝ, x ≥ 1/2 → (deriv (f m n x)) ≥ 0) :
  1/m + 1/n = 4 :=
sorry

end find_min_value_of_fraction_l630_630777


namespace stock_brokerage_contradiction_l630_630898

-- The proof is to show that the brokerage does not make sense in this context,
-- i.e., having an output greater than 100% leads to contradiction.

theorem stock_brokerage_contradiction
    (S : ℝ)
    (h1 : S + S / 400 = 104)
    (h2 : 104.25 = S + (S / 400)) :
    (104.25 / S) * 100 > 100 := by
  -- Proof is left as an exercise
  sorry

end stock_brokerage_contradiction_l630_630898


namespace quadratic_eq_integer_roots_iff_l630_630406

theorem quadratic_eq_integer_roots_iff (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x * y = n ∧ x + y = 4) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadratic_eq_integer_roots_iff_l630_630406


namespace find_BX_squared_l630_630872

-- Define the triangle and required points
variables {A B C M N X : Type}
variables (h_M_midpoint : M = midpoint A C)
variables (h_CN_bisector : is_angle_bisector C N A B)
variables (h_X_intersection : X = intersection (line_through B M) (line_through C N))
variables (h_BXC_equilateral : is_equilateral B X C)
variables (h_AC_length : distance A C = 4)

-- The main theorem
theorem find_BX_squared (h_BX_squared : BX^2 = 16) : BX^2 = 16 := sorry

end find_BX_squared_l630_630872


namespace determine_locus_l630_630302

noncomputable def locus_condition (A B C : EuclideanGeometry.Point) : Bool :=
  let AB := dist A B
  let height := (dist A C).abs * (dist B C).abs / AB
  height < AB
    
theorem determine_locus (A B : EuclideanGeometry.Point) (C : EuclideanGeometry.Point) :
  A = (-1, 0) ∧ B = (1, 0) → locus_condition A B C →
    ¬ dist C (1, 1) < 1 ∧ ¬ dist C (-1, 1) < 1 ∧ ¬ dist C (1, -1) < 1 ∧ ¬ dist C (-1, -1) < 1 :=
sorry

end determine_locus_l630_630302


namespace amount_subtracted_eq_l630_630428

/-- Definition of values based on problem conditions -/
def base_salary : ℝ := 3000
def total_earnings : ℝ := 8000
def commission_rate : ℝ := 0.02
def house_A_cost : ℝ := 60000

/-- Total commission earned -/
def total_commission : ℝ := total_earnings - base_salary

/-- Total sales price of the three houses -/
def total_sales_price : ℝ := total_commission / commission_rate

/-- Cost of House B -/
def house_B_cost : ℝ := 3 * house_A_cost

/-- Cost of House C (parameterized by amount subtracted) -/
noncomputable def house_C_cost (X : ℝ) : ℝ := 2 * house_A_cost - X

/-- The main theorem stating the amount subtracted from twice the cost of House A to get the cost of House C is $110,000. -/
theorem amount_subtracted_eq :
  ∃ X : ℝ, house_C_cost X + house_A_cost + house_B_cost = total_sales_price ∧ X = 110000 :=
by
  sorry

end amount_subtracted_eq_l630_630428


namespace charge_two_hours_l630_630591

def charge_first_hour (F A : ℝ) : Prop := F = A + 25
def total_charge_five_hours (F A : ℝ) : Prop := F + 4 * A = 250
def total_charge_two_hours (F A : ℝ) : Prop := F + A = 115

theorem charge_two_hours (F A : ℝ) 
  (h1 : charge_first_hour F A)
  (h2 : total_charge_five_hours F A) : 
  total_charge_two_hours F A :=
by
  sorry

end charge_two_hours_l630_630591


namespace train_length_l630_630220

theorem train_length
  (speed_train : ℕ) (speed_motorbike : ℕ) (overtake_time : ℕ)
  (h_train_speed : speed_train = 100)
  (h_motorbike_speed : speed_motorbike = 64)
  (h_overtake_time : overtake_time = 85) :
  (let relative_speed := 36 * (1000 / 3600) in
   let distance := relative_speed * overtake_time in
   distance = 850) :=
sorry

end train_length_l630_630220


namespace find_s_for_g_neg1_zero_l630_630405

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end find_s_for_g_neg1_zero_l630_630405


namespace symmetry_center_sum_l630_630046

def f (x : ℝ) : ℝ := x + Real.sin (π * x) - 3

theorem symmetry_center_sum :
  (∑ k in Finset.range 4032, f ((k + 1) / 2016) : ℝ) = -8062 :=
by
  sorry

end symmetry_center_sum_l630_630046


namespace sin_6phi_l630_630348

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * (Real.sqrt 8)) / 5) : 
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 :=
by
  sorry

end sin_6phi_l630_630348


namespace inequality_proof_l630_630747

open Real

noncomputable def sum_condition (x : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, 1 / (1 + x i) = n / 2

theorem inequality_proof (n : ℕ) (x : ℕ → ℝ) (h : ∀ i, x i > 0) (hc : sum_condition x n) :
    (∑ i in Finset.range n, ∑ j in Finset.range n, 1 / (x i + x j)) ≥ n^2 / 2 := by
  sorry

end inequality_proof_l630_630747


namespace sum_of_angles_divisible_by_360_l630_630603

theorem sum_of_angles_divisible_by_360 {n : ℕ} (h : n ≠ 0) :
  let sides := 2 * n in
  (sides - 2) * 180 = 360 * (n - 1) :=
by
  have sides_eq_2n : sides = 2 * n := rfl
  sorry

end sum_of_angles_divisible_by_360_l630_630603


namespace range_of_m_l630_630321

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (π / 2 < x) ∧ (x < π) ∧ (2 * (Real.sin x)^2 - sqrt 3 * (Real.sin (2 * x)) + m - 1 = 0))
  ↔ (-2 < m) ∧ (m < -1) :=
sorry

end range_of_m_l630_630321


namespace dice_probability_l630_630883

theorem dice_probability (n : ℕ) (h : n = 7) : 
  (∃ (x : (fin 6) → fin 7), (∃ (i j : fin 7), i ≠ j ∧ x i = x j)) :=
by
  sorry

end dice_probability_l630_630883


namespace right_triangle_pythagorean_l630_630567

theorem right_triangle_pythagorean (ABC : Type) [Triangle ABC] 
  (a b c : Real) (ha : a = BC) (hb : b = AC) (hc : c = AB)
  (angleC : ∠C = 90) : a^2 + b^2 = c^2 := 
by 
  sorry

end right_triangle_pythagorean_l630_630567


namespace cover_black_squares_impossible_for_small_n_l630_630277

theorem cover_black_squares (n : ℕ) (h : n % 2 = 1) (h1 : n ≥ 7):
  ∃ t : ℕ, t = (n + 1) / 2 ∧ 
  covers_black_squares_with_trominoes n t :=
sorry

theorem impossible_for_small_n (n : ℕ) (h : n % 2 = 1) (h2 : n < 7):
  ¬∃ t : ℕ, covers_black_squares_with_trominoes n t :=
sorry

def covers_black_squares_with_trominoes (n : ℕ) (t : ℕ) : Prop :=
-- This should define the property that t trominoes cover all black squares
sorry

end cover_black_squares_impossible_for_small_n_l630_630277


namespace jade_more_transactions_than_cal_l630_630881

theorem jade_more_transactions_than_cal :
  let Mabel_transactions := 90
  let Anthony_transactions := Mabel_transactions + (0.10 * Mabel_transactions)
  let Cal_transactions := (2 / 3) * Anthony_transactions
  let Jade_transactions := 84
  Jade_transactions - Cal_transactions = 18 :=
by
  let Mabel_transactions := 90
  let Anthony_transactions := Mabel_transactions + (0.10 * Mabel_transactions)
  let Cal_transactions := (2 / 3) * Anthony_transactions
  let Jade_transactions := 84
  sorry

end jade_more_transactions_than_cal_l630_630881


namespace k_is_perfect_square_l630_630041

theorem k_is_perfect_square (m n : ℤ) (hm : m > 0) (hn : n > 0) (k := ((m + n) ^ 2) / (4 * m * (m - n) ^ 2 + 4)) :
  ∃ (a : ℤ), k = a ^ 2 := by
  sorry

end k_is_perfect_square_l630_630041


namespace general_term_a_n_sum_b_n_l630_630771

-- First part: Prove the general term formula for a_n

def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop :=
  S_n 3 = 15 ∧ (2 * a_n 1 + 9 * (a_n 2 - a_n 1) = 24)

theorem general_term_a_n 
  (a_n : ℕ → ℕ)
  (h : arithmetic_sequence a_n id) :
  ∀ n, a_n n = 2 * n + 1 :=
sorry

-- Second part: Prove the sum of the first n terms for the sequence b_n

def b_n (a_n : ℕ → ℕ) (n : ℕ) : ℚ :=
1 / (a_n n ^ 2 - 1)

def T_n (a_n : ℕ → ℕ) (n : ℕ) : ℚ :=
∑ i in Finset.range n, b_n a_n (i + 1)

theorem sum_b_n 
  (a_n : ℕ → ℕ)
  (h : ∀ n, a_n n = 2 * n + 1)
  (n : ℕ) :
  T_n a_n n = n / (4 * (n + 1)) :=
sorry

end general_term_a_n_sum_b_n_l630_630771


namespace number_of_orange_ribbons_l630_630818

/-- Define the total number of ribbons -/
def total_ribbons (yellow purple orange black total : ℕ) : Prop :=
  yellow + purple + orange + black = total

/-- Define the fractions -/
def fractions (total_ribbons yellow purple orange black : ℕ) : Prop :=
  yellow = total_ribbons / 4 ∧ purple = total_ribbons / 3 ∧ orange = total_ribbons / 12 ∧ black = 40

/-- Define the black ribbons fraction -/
def black_fraction (total_ribbons : ℕ) : Prop :=
  40 = total_ribbons / 3

theorem number_of_orange_ribbons :
  ∃ (total : ℕ), total_ribbons (total / 4) (total / 3) (total / 12) 40 total ∧ black_fraction total ∧ (total / 12 = 10) :=
by
  sorry

end number_of_orange_ribbons_l630_630818


namespace reciprocal_of_neg_2023_l630_630481

theorem reciprocal_of_neg_2023 : ∃ x : ℝ, (-2023) * x = 1 ∧ x = -1 / 2023 := 
by {
  existsi (-1 / 2023),
  split,
  { -- Prove that the product of -2023 and -1/2023 is 1
    unfold has_mul.mul,
    norm_num,
  },
  { -- Prove that x is indeed -1/2023
    refl,
  }
}

end reciprocal_of_neg_2023_l630_630481


namespace decimal_to_binary_123_l630_630515

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630515


namespace range_of_a_l630_630326

noncomputable def f (a x : ℝ) : ℝ := a / x + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - 5

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ Icc (1/2 : ℝ) 2, f a x1 - g x2 ≥ 2) ↔ 1 ≤ a :=
by
  sorry

end range_of_a_l630_630326


namespace problem1_problem2_l630_630781

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a^2 / x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + g x

-- Problem 1: Prove that a = sqrt(3) given that x = 1 is an extremum point for h(x, a)
theorem problem1 (a : ℝ) (h_extremum : ∀ x : ℝ, x = 1 → 0 = (2 - a^2 / x^2 + 1 / x)) : a = Real.sqrt 3 := sorry

-- Problem 2: Prove the range of a is [ (e + 1) / 2, +∞ ) such that for any x1, x2 ∈ [1, e], f(x1, a) ≥ g(x2)
theorem problem2 (a : ℝ) :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f x1 a ≥ g x2) →
  (Real.exp 1 + 1) / 2 ≤ a :=
sorry

end problem1_problem2_l630_630781


namespace sin_6_phi_l630_630349

theorem sin_6_phi (φ : ℝ) (h : complex.exp (complex.I * φ) = (3 + complex.I * real.sqrt 8) / 5) :
  real.sin (6 * φ) = -396 * real.sqrt 2 / 15625 :=
by sorry

end sin_6_phi_l630_630349


namespace probability_at_least_four_same_l630_630711

/--
Theorem: The probability that at least four of the five dice show the same value is \( \frac{13}{648} \), given that five fair six-sided dice are rolled.

Conditions:
1. Five fair six-sided dice are rolled.
2. Each die roll is independent.
-/
theorem probability_at_least_four_same (rolled_dice : vector ℕ 5) (h : ∀ (die : ℕ), die ∈ rolled_dice → (1 ≤ die) ∧ (die ≤ 6)) :
  (∑ val in (finset.range 6).map nat.succ, ∑ pos in finset.powerset_len 4 (finset.range 5),
    if ∀ i ∈ pos, rolled_dice.nth i = some val then 1 else 0) / 6^5 = 13 / 648 :=
by sorry

end probability_at_least_four_same_l630_630711


namespace power_of_two_l630_630855

theorem power_of_two (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_prime : Prime (m^(4^n + 1) - 1)) : 
  ∃ t : ℕ, n = 2^t :=
sorry

end power_of_two_l630_630855


namespace sequence_property_exists_l630_630246

theorem sequence_property_exists :
  ∃ a₁ a₂ a₃ a₄ : ℝ, 
  a₂ - a₁ = a₃ - a₂ ∧ a₃ - a₂ = a₄ - a₃ ∧
  (a₃ / a₁ = a₄ / a₃) ∧ ∃ r : ℝ, r ≠ 0 ∧ a₁ = -4 * r ∧ a₂ = -3 * r ∧ a₃ = -2 * r ∧ a₄ = -r :=
by
  sorry

end sequence_property_exists_l630_630246


namespace find_avg_first_30_results_l630_630093

noncomputable def avg_first_30_results (A : ℤ) : Prop :=
  let sum_first_30 := 30 * A
  let sum_other_20 := 20 * 30
  let sum_all_50 := 50 * 24
  sum_first_30 + sum_other_20 = sum_all_50 → A = 20

-- We state the theorem that encapsulates the problem:
theorem find_avg_first_30_results (A : ℤ) (h : avg_first_30_results A) : A = 20 :=
by
  exact h

end find_avg_first_30_results_l630_630093


namespace Robi_contribution_eq_4000_l630_630440

variables (R : ℝ)

def Rudy_contribution (R : ℝ) : ℝ := (5 / 4) * R
def total_contribution (R : ℝ) : ℝ := R + Rudy_contribution R
def total_profit (R : ℝ) : ℝ := 0.20 * total_contribution R

theorem Robi_contribution_eq_4000 (R : ℝ) (h1 : total_profit R = 1800) : R = 4000 :=
by {
  have : total_profit R = (9 / 20) * R := by sorry,
  rw this at h1,
  linarith,
}

end Robi_contribution_eq_4000_l630_630440


namespace slope_of_chord_in_ellipse_l630_630003

noncomputable def slope_of_chord (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem slope_of_chord_in_ellipse :
  ∀ (x1 y1 x2 y2 : ℝ),
    (x1^2 / 16 + y1^2 / 9 = 1) →
    (x2^2 / 16 + y2^2 / 9 = 1) →
    ((x1 + x2) = -2) →
    ((y1 + y2) = 4) →
    slope_of_chord x1 y1 x2 y2 = 9 / 32 :=
by
  intro x1 y1 x2 y2 h1 h2 h3 h4
  sorry

end slope_of_chord_in_ellipse_l630_630003


namespace smallest_five_digit_divisible_by_53_l630_630553

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l630_630553


namespace Norbs_age_l630_630080

def guesses : List ℕ := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def two_off_by_one (n : ℕ) (guesses : List ℕ) : Prop := 
  (n - 1 ∈ guesses) ∧ (n + 1 ∈ guesses)

def at_least_half_too_low (n : ℕ) (guesses : List ℕ) : Prop := 
  (guesses.filter (· < n)).length ≥ guesses.length / 2

theorem Norbs_age : 
  ∃ x, is_prime x ∧ two_off_by_one x guesses ∧ at_least_half_too_low x guesses ∧ x = 37 := 
by 
  sorry

end Norbs_age_l630_630080


namespace line_intersects_circle_l630_630250

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  ∃ p : ℝ × ℝ, (p.1 ^ 2 + p.2 ^ 2 = 9) ∧ (a * p.1 - p.2 + 2 * a = 0) :=
by
  sorry

end line_intersects_circle_l630_630250


namespace travel_times_equal_solve_distances_l630_630976

-- Define the conditions as variables and constants
def school_to_county_distance : ℝ := 21
def walking_speed : ℝ := 4
def bus_speed : ℝ := 60
def rest_time : ℝ := 1 / 6

-- Define the unknowns
variables (x : ℝ) -- distance from school to place A
variables (distance_walked : ℝ) -- distance walked by the 7th-grade team

-- Condition for the travel times to be equal
theorem travel_times_equal :
  (21 - x) / walking_speed + rest_time = (21 + x) / bus_speed :=
sorry

-- Solve for the distance from school to place A and the distance walked
theorem solve_distances :
  x = 19 ∧ distance_walked = 2 :=
sorry

end travel_times_equal_solve_distances_l630_630976


namespace geometric_sequence_correct_l630_630360

theorem geometric_sequence_correct (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 8)
  (h2 : a 2 * a 3 = -8)
  (h_geom : ∀ (n : ℕ), a (n + 1) = a n * r) :
  a 4 = -1 :=
by {
  sorry
}

end geometric_sequence_correct_l630_630360


namespace fraction_multiplication_l630_630506

theorem fraction_multiplication : (1 / 3) * (1 / 4) * (1 / 5) * 60 = 1 := by
  sorry

end fraction_multiplication_l630_630506


namespace solve_equation_nat_numbers_l630_630574

theorem solve_equation_nat_numbers (a b : ℕ) (h : (a, b) = (11, 170) ∨ (a, b) = (22, 158) ∨ (a, b) = (33, 146) ∨
                                    (a, b) = (44, 134) ∨ (a, b) = (55, 122) ∨ (a, b) = (66, 110) ∨
                                    (a, b) = (77, 98) ∨ (a, b) = (88, 86) ∨ (a, b) = (99, 74) ∨
                                    (a, b) = (110, 62) ∨ (a, b) = (121, 50) ∨ (a, b) = (132, 38) ∨
                                    (a, b) = (143, 26) ∨ (a, b) = (154, 14) ∨ (a, b) = (165, 2)) :
  12 * a + 11 * b = 2002 :=
by
  sorry

end solve_equation_nat_numbers_l630_630574


namespace problem_statement_l630_630404

-- Define the function g and specify its properties
def g : ℕ → ℕ := sorry

axiom g_property (a b : ℕ) : g (a^2 + b^2) + g (a + b) = (g a)^2 + (g b)^2

-- Define the values of m and t that arise from the constraints on g(49)
def m : ℕ := 2
def t : ℕ := 106

-- Prove that the product m * t is 212
theorem problem_statement : m * t = 212 :=
by {
  -- Since g_property is an axiom, we use it to derive that
  -- g(49) can only take possible values 0 and 106,
  -- thus m = 2 and t = 106.
  sorry
}

end problem_statement_l630_630404


namespace total_lives_l630_630988

theorem total_lives (n_friends : ℕ) (n_lives_each : ℕ) (h_friends : n_friends = 8) (h_lives_each : n_lives_each = 8) : n_friends * n_lives_each = 64 :=
by
  rw [h_friends, h_lives_each]
  rfl

end total_lives_l630_630988


namespace remainder_of_division_l630_630174

def p (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
def d (x : ℝ) : ℝ := 4*x - 8

theorem remainder_of_division :
  (p 2) = 81 :=
by
  sorry

end remainder_of_division_l630_630174


namespace min_positive_sum_of_products_pair_l630_630289

theorem min_positive_sum_of_products_pair : 
  ∃ (a : Fin 95 → ℤ), (∀ i, a i = 1 ∨ a i = -1) ∧ 
                      let N := ∑ i in (Fin 95).cross (Fin 95), if h: i.1 < i.2 then a i.1 * a i.2 else 0
                      N = 13 :=
by 
  sorry

end min_positive_sum_of_products_pair_l630_630289


namespace tan_phi_median_angle_bisector_l630_630820

theorem tan_phi_median_angle_bisector (β φ : ℝ) (h : tan (β / 2) = 1 / real.sqrt (real.sqrt 2)) :
  tan φ = 1 / 2 :=
sorry

end tan_phi_median_angle_bisector_l630_630820


namespace cos_x_minus_6y_values_l630_630009

-- Definitions based on the given conditions
def condition1 (x y : ℝ) : Prop :=
  (sin (3 * x) / ((2 * cos (2 * x) + 1) * sin (2 * y)) = 1/5 + (cos (x - 2 * y))^2)

def condition2 (x y : ℝ) : Prop :=
  (cos (3 * x) / ((1 - 2 * cos (2 * x)) * cos (2 * y)) = 4 / 5 + (sin (x - 2 * y))^2)

-- The theorem we want to state
theorem cos_x_minus_6y_values (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  ∃ (k : ℤ), cos (x - 6 * y) = 1 ∨ cos (x - 6 * y) = -3/5 :=
by
  sorry

end cos_x_minus_6y_values_l630_630009


namespace intersection_correct_l630_630396

noncomputable def set_M : Set ℝ := { x | x^2 + x - 6 ≤ 0 }
noncomputable def set_N : Set ℝ := { x | abs (2 * x + 1) > 3 }
noncomputable def set_intersection : Set ℝ := { x | (x ∈ set_M) ∧ (x ∈ set_N) }

theorem intersection_correct : 
  set_intersection = { x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 2) } := 
by 
  sorry

end intersection_correct_l630_630396


namespace find_c_l630_630449

-- Introduce logarithm with arbitrary base
noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

lemma solve_equation (x : ℝ) : (9^(x + 6) = 5^x) → x = log_base (5 / 9) (9^6) := by
  intro h,
  sorry

-- Goal to extract the base c from the proven equation
theorem find_c : (∃ x : ℝ, 9^(x + 6) = 5^x) → ∃ c : ℝ, c = 5 / 9 := by
  intro hex,
  obtain ⟨x, h⟩ := hex,
  use 5 / 9,
  have sol := solve_equation x,
  apply sol,
  exact h

end find_c_l630_630449


namespace Amelia_wins_probability_correct_l630_630224

-- Define the probabilities
def probability_Amelia_heads := 1 / 3
def probability_Blaine_heads := 2 / 5

-- The infinite geometric series sum calculation for Amelia to win
def probability_Amelia_wins :=
  probability_Amelia_heads * (1 / (1 - (1 - probability_Amelia_heads) * (1 - probability_Blaine_heads)))

-- Given values p and q from the conditions
def p := 5
def q := 9

-- The correct answer $\frac{5}{9}$
def Amelia_wins_correct := 5 / 9

-- Prove that the probability calculation matches the given $\frac{5}{9}$, and find q - p
theorem Amelia_wins_probability_correct :
  probability_Amelia_wins = Amelia_wins_correct ∧ q - p = 4 := by
  sorry

end Amelia_wins_probability_correct_l630_630224


namespace calc_value_l630_630236

theorem calc_value (a b x : ℤ) (h₁ : a = 153) (h₂ : b = 147) (h₃ : x = 900) : x^2 / (a^2 - b^2) = 450 :=
by
  rw [h₁, h₂, h₃]
  -- Proof follows from the calculation in the provided steps
  sorry

end calc_value_l630_630236


namespace line_circle_intersections_l630_630782

theorem line_circle_intersections (a r : ℝ) (x y : ℝ) :
  (r > 0) → 
  let l := (fun x y => a * x - y + 2 - 2 * a = 0) in
  let C := (fun x y => (x - 4)^2 + (y - 1)^2 = r^2) in
  ( ∀ x y, ¬ l 2 (-2) ) ∧
  ( r > 2 → False ) ∧
  ( r = 3 → (∃ MN, MN ∈ [4, 6])) ∧
  ( r = 5 → (min_value (λ CM CN, CM • CN) = -25)) := 
sorry

end line_circle_intersections_l630_630782


namespace max_fraction_sum_l630_630392

theorem max_fraction_sum (n a b c d : ℕ) (h1 : n ≥ 2) (h2 : a + c ≤ n) (h3 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  ∃ (π : ℕ), π = a * c + 1 ∧ (a : ℚ) / (a + π) + (c : ℚ) / (c + 1) = ((a : ℚ) / b + (c : ℚ) / d).max  ⟨| sorry }.

end max_fraction_sum_l630_630392


namespace placement_of_pegs_problem_l630_630933

/-- The placement of pegs problem -/
theorem placement_of_pegs_problem : 
  ∀ (R G B Y O : ℕ), 
  R = 5 → G = 4 → B = 3 → Y = 2 → O = 1 → 
  ∀ (P : list ℕ), list.nodup P → ∀ (S : list (ℕ × ℕ)), 
  (∀ (i j : ℕ), i ≠ j → P i ≠ P j) →
  (∀ (i j k l : ℕ), (i, j) ≠ (k, l) → S (i, j) ≠ S (k, l)) → 
  ∃ unique_arrangement : fin (R + G + B + Y + O), unique_arrangement = 1 :=
by
  intros R G B Y O hR hG hB hY hO P hp S hs 
  sorry

end placement_of_pegs_problem_l630_630933


namespace sum_x_coordinates_Q3_l630_630587

theorem sum_x_coordinates_Q3 (x : Fin 36 → ℝ) 
  (h : ∑ i, x i = 144) : 
  let Q1 := x,
      Q2 := λ i : Fin 36, (x i + x ⟨(i + 1) % 36, sorry⟩) / 2,
      Q3 := λ i : Fin 36, (Q2 i + Q2 ⟨(i + 1) % 36, sorry⟩) / 2 in
  ∑ i, Q3 i = 144 := by
  sorry

end sum_x_coordinates_Q3_l630_630587


namespace problem1_problem2_l630_630581

-- Problem 1: Prove the value of the expression
theorem problem1 : (-2^0 + 4^(-1) * (-1)^(2009) * (-1/2)^(-2) = -2) :=
by
  sorry

-- Problem 2: Simplify the polynomial expression
theorem problem2 (x : ℝ) : (x + 1)^2 - (x - 1) * (x + 2) = x + 3 :=
by
  sorry

end problem1_problem2_l630_630581


namespace geometric_sequence_S8_l630_630770

theorem geometric_sequence_S8 (S : ℕ → ℝ) (hs2 : S 2 = 4) (hs4 : S 4 = 16) : 
  S 8 = 160 := by
  sorry

end geometric_sequence_S8_l630_630770


namespace nth_non_square_l630_630679

theorem nth_non_square (n : ℕ) : 
  ∃ m : ℕ, m = Int.toNat (Real.sqrt n).toInt ∧ (let m := Int.toNat (Real.sqrt n).toInt in deleted_squares !list.range! n = n + m) := 
sorry

end nth_non_square_l630_630679


namespace functional_equation_solution_l630_630264

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2 * x * y) + f (f (x + y)) = x * f y + y * f x + f (x + y)) →
  (f = 0 ∨ f = id ∨ f = λ x, 2 - x) :=
by 
  sorry

end functional_equation_solution_l630_630264


namespace second_number_in_set_l630_630094

theorem second_number_in_set (avg1 avg2 n1 n2 n3 : ℕ) (h1 : avg1 = (10 + 70 + 19) / 3) (h2 : avg2 = avg1 + 7) (h3 : n1 = 20) (h4 : n3 = 60) :
  n2 = n3 := 
  sorry

end second_number_in_set_l630_630094


namespace find_f_of_power_function_l630_630104

theorem find_f_of_power_function (a : ℝ) (alpha : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a ∧ a ≠ 1) 
  (h2 : ∀ x, f x = x^alpha) 
  (h3 : ∀ x, a^(x-2) + 3 = f (2)): 
  f 2 = 4 := 
  sorry

end find_f_of_power_function_l630_630104


namespace regular_octagon_l630_630931

-- Define the square vertices and centers of the circles
def A : ℝ × ℝ := (a, a)
def B : ℝ × ℝ := (-a, a)
def C : ℝ × ℝ := (-a, -a)
def D : ℝ × ℝ := (a, -a)
def center : ℝ × ℝ := (0, 0)

-- Define the radius of the circles
def radius := a * Real.sqrt 2

-- Equation of the circles centered at vertices
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2
def circle_A (x y : ℝ) : Prop := circle_eq a a radius x y
def circle_B (x y : ℝ) : Prop := circle_eq (-a) a radius x y
def circle_C (x y : ℝ) : Prop := circle_eq (-a) (-a) radius x y
def circle_D (x y : ℝ) : Prop := circle_eq a (-a) radius x y

-- Prove that the intersection points of the circles with the sides of the square form a regular octagon
theorem regular_octagon : 
  let intersection_points := {
    -- Intersection points calculated as per solution
    (a, -a + a * Real.sqrt 2), 
    (a - a * Real.sqrt 2, -a)
    -- include all other intersection points
  } in 
  is_regular_octagon intersection_points :=
sorry

end regular_octagon_l630_630931


namespace third_pasture_cows_l630_630496

theorem third_pasture_cows (x y : ℝ) (H1 : x + 27 * y = 18) (H2 : 2 * x + 84 * y = 51) : 
  10 * x + 10 * 3 * y = 60 -> 60 / 3 = 20 :=
by
  sorry

end third_pasture_cows_l630_630496


namespace find_area_of_triangle_BCD_l630_630393

noncomputable def area_of_triangle_BCD 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (is_perpendicular : ∀ (A B C D : ℝ × ℝ × ℝ), (B - A) ⬝ (C - A) = 0 ∧ (C - A) ⬝ (D - A) = 0 ∧ (B - A) ⬝ (D - A) = 0) 
: ℝ := (1 / 2) * real.sqrt (b^2 * c^2 + a^2 * c^2 + a^2 * b^2)

-- The theorem we want to prove
theorem find_area_of_triangle_BCD
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (is_perpendicular : ∀ (A B C D : ℝ × ℝ × ℝ), (B - A) ⬝ (C - A) = 0 ∧ (C - A) ⬝ (D - A) = 0 ∧ (B - A) ⬝ (D - A) = 0):
  area_of_triangle_BCD a b c ha hb hc is_perpendicular = (1 / 2) * real.sqrt (b^2 * c^2 + a^2 * c^2 + a^2 * b^2) :=
by sorry

end find_area_of_triangle_BCD_l630_630393


namespace tadpoles_to_frogs_ratio_l630_630131

def initial_fish : Nat := 50
def initial_tadpoles (F : Nat) : Nat := 3 * F
def remaining_fish (F C : Nat) : Nat := F - C
def remaining_tadpoles (T x : Nat) : Nat := T - x

theorem tadpoles_to_frogs_ratio :
  let F := initial_fish
  let T := initial_tadpoles F
  let C := 7
  let remaining_fishes := remaining_fish F C
  let remaining_tadpoles := remaining_tadpoles T
  let x := T - (remaining_fishes + 32) in
  (x : ℚ) / T = 1 / 2 :=
by
  sorry

end tadpoles_to_frogs_ratio_l630_630131


namespace find_ABC_l630_630266

theorem find_ABC :
    ∃ (A B C : ℚ), 
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5 → 
        (x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) 
    ∧ A = 5 / 3 ∧ B = -7 / 2 ∧ C = 8 / 3 := 
sorry

end find_ABC_l630_630266


namespace point_in_third_quadrant_l630_630808

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end point_in_third_quadrant_l630_630808


namespace find_p_and_probability_l630_630088

def prob_xiao_zhang_correct := 3 / 4

def prob_same_correct (p : ℚ) : ℚ :=
  ((1 - p)^2 / 16) + (3 * p * (1 - p) / 4) + (9 * p^2 / 16)

def prob_xiao_li_both_correct (p : ℚ) : ℚ :=
  (prob_xiao_zhang_correct)^2 * (p^2)

theorem find_p_and_probability (p : ℚ) :
  (prob_same_correct p = 61 / 144) →
  p = 2 / 3 ∧ prob_xiao_li_both_correct 2 / 3 = 1 / 4 :=
by {
  intro h,
  -- Add steps here (if necessary) after translation completion.
  sorry
}

end find_p_and_probability_l630_630088


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630151

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630151


namespace max_combined_subject_marks_l630_630649

theorem max_combined_subject_marks :
  let total_marks_math := (130 + 14) / 0.36,
      total_marks_physics := (120 + 20) / 0.40,
      total_marks_chemistry := (160 + 10) / 0.45,
      max_total_marks := total_marks_math + total_marks_physics + total_marks_chemistry in
  ⌊(total_marks_math + total_marks_physics + total_marks_chemistry)⌋ = 1127 :=
by
  -- The proof should be written here
  sorry

end max_combined_subject_marks_l630_630649


namespace pilot_fish_speed_theorem_l630_630850

noncomputable def pilot_fish_speed 
    (keanu_speed : ℕ)
    (shark_factor : ℕ) 
    (pilot_fish_factor : ℕ) 
    : ℕ :=
    let shark_speed_increase := keanu_speed * (shark_factor - 1) in
    let pilot_fish_speed_increase := shark_speed_increase / pilot_fish_factor in
    keanu_speed + pilot_fish_speed_increase

theorem pilot_fish_speed_theorem : 
    pilot_fish_speed 20 2 2 = 30 :=
by 
    simp [pilot_fish_speed]
    sorry  -- proof steps are omitted.

end pilot_fish_speed_theorem_l630_630850


namespace geometry_problem_l630_630657

variables {A O G E B D H F M K S I L T J : Type}
variables [square AOGE] [square BDHF] [square MKSI] [square MLTJ]
variables (M_is_midpoint_AB : midpoint M A B)
variables (collinear_E_I_J_F : collinear E I J F)
variables (collinear_C_K_L_D : collinear C K L D)

theorem geometry_problem 
  (H1 : square AOGE) 
  (H2 : square BDHF)
  (H3 : square MKSI)
  (H4 : square MLTJ)
  (H5 : midpoint M A B)
  (H6 : collinear E I J F)
  (H7 : collinear C K L D) :
  EI^2 + FJ^2 = CK^2 + DL^2 :=
sorry

end geometry_problem_l630_630657


namespace cube_side_length_l630_630812

def cube_volume (side : ℝ) : ℝ := side ^ 3

theorem cube_side_length (volume : ℝ) (h : volume = 729) : ∃ (side : ℝ), side = 9 ∧ cube_volume side = volume :=
by
  sorry

end cube_side_length_l630_630812


namespace sum_of_interior_angles_divisible_by_360_l630_630605

theorem sum_of_interior_angles_divisible_by_360
  (n : ℕ)
  (h : n > 0) :
  ∃ k : ℤ, ((2 * n - 2) * 180) = 360 * k :=
by
  sorry

end sum_of_interior_angles_divisible_by_360_l630_630605


namespace greatest_QPN_value_l630_630563

theorem greatest_QPN_value (N : ℕ) (Q P : ℕ) (QPN : ℕ) :
  (NN : ℕ) =
  10 * N + N ∧
  QPN = 100 * Q + 10 * P + N ∧
  N < 10 ∧ N ≥ 1 ∧
  NN * N = QPN ∧
  NN >= 10 ∧ NN < 100  -- Ensuring NN is a two-digit number
  → QPN <= 396 := sorry

end greatest_QPN_value_l630_630563


namespace combined_marble_remainder_l630_630955

theorem combined_marble_remainder (l j : ℕ) (h_l : l % 8 = 5) (h_j : j % 8 = 6) : (l + j) % 8 = 3 := by
  sorry

end combined_marble_remainder_l630_630955


namespace evaluate_expression_l630_630953

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end evaluate_expression_l630_630953


namespace fraction_of_dark_tiles_l630_630617

/--
 Given a repeating floor tiling pattern every 8 x 8 tiles, and the top-left 4 x 4 block of the 
 8 x 8 section contains 7 dark tiles, we want to prove that the fraction of the floor covered 
 by dark tiles is 7/16.
-/
theorem fraction_of_dark_tiles (repeating_pattern : Prop) 
    (top_left_block : ℕ → ℕ → Prop) :
    (∀ i j, top_left_block i j ↔ ((i < 4 ∧ j < 4) ∧ 7 dark tiles)) →
    ((4 quadrants ∧ 7 dark tiles/quadrant) → 
    (total_tiles = 64 ∧ dark_tiles = 28) → 
    (fraction_of_dark = 7 / 16)) :=
sorry

end fraction_of_dark_tiles_l630_630617


namespace at_least_one_not_lt_half_l630_630067

theorem at_least_one_not_lt_half (a b c : ℝ) (h : a + 2 * b + c = 2) : 
  ∃ x ∈ {a, b, c}, x ≥ 1 / 2 := 
sorry

end at_least_one_not_lt_half_l630_630067


namespace integer_count_n_l630_630794

theorem integer_count_n (n : ℤ) (H1 : n % 3 = 0) (H2 : 3 * n ≥ 1) (H3 : 3 * n ≤ 1000) : 
  ∃ k : ℕ, k = 111 := by
  sorry

end integer_count_n_l630_630794


namespace combined_speed_of_trains_l630_630905

-- Define the conditions
variables (distance_AB : ℕ) (relative_speed_difference : ℕ) (remaining_distance : ℕ) (time : ℕ)
variable (combined_speed : ℕ)

axiom initial_distance : distance_AB = 300
axiom speed_difference : relative_speed_difference = 10
axiom distance_after_time : remaining_distance = 40
axiom journey_time : time = 2

-- Define the main theorem
theorem combined_speed_of_trains : combined_speed = 130 :=
by
  have initial_distance := initial_distance,
  have speed_difference := speed_difference,
  have distance_after_time := distance_after_time,
  have journey_time := journey_time,
  sorry

end combined_speed_of_trains_l630_630905


namespace smallest_five_digit_divisible_by_53_l630_630554

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l630_630554


namespace next_chime_and_lightup_l630_630129

noncomputable def lcm_9_60 : Nat :=
  Nat.lcm 9 60

theorem next_chime_and_lightup (h_lcm : lcm_9_60 = 180) : 3:00 PM =
  let start_time := 12:00
  let interval := lcm_9_60
  (start_time + interval) :=
begin
  -- Proof would go here
  sorry
end

end next_chime_and_lightup_l630_630129


namespace angle_of_inclination_l630_630375

noncomputable def inclination_angle (A B : ℝ × ℝ) : ℝ :=
  let slope := (B.2 - A.2) / (B.1 - A.1)
  real.atan slope * (180 / real.pi)

theorem angle_of_inclination :
  inclination_angle (-1, 0) (1, 2) = 45 :=
by
  -- We declare the points A and B
  let A := (-1 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 2 : ℝ)
  -- Compute the slope of line passing through points A and B
  let slope := (B.2 - A.2) / (B.1 - A.1)
  -- Compute the angle of inclination using arctan
  have h : real.atan slope * (180 / real.pi) = 45 := 
    sorry -- Proof to be provided.
  exact h

end angle_of_inclination_l630_630375


namespace tetrahedron_probability_correct_l630_630626

noncomputable def tetrahedron_probability : ℚ :=
  let total_arrangements := 16
  let suitable_arrangements := 2
  suitable_arrangements / total_arrangements

theorem tetrahedron_probability_correct : tetrahedron_probability = 1 / 8 :=
by
  sorry

end tetrahedron_probability_correct_l630_630626


namespace num_positive_int_values_for_expression_is_7_l630_630720

theorem num_positive_int_values_for_expression_is_7 :
  {N : ℕ // 0 < N ∧ ∃ k : ℕ, 48 = k * (N + 3)}.card = 7 := 
sorry

end num_positive_int_values_for_expression_is_7_l630_630720


namespace bela_wins_l630_630665

noncomputable def bela_always_wins (n : ℕ) (h : n > 10) : Prop :=
  ∀ (strategy_bela strategy_jenn: ℕ → ℝ), 
  (∀ k, strategy_bela k ∈ set.Icc 0 n ∧ (k = 0 ∨ ∀ j < k, |strategy_bela j - strategy_bela k| > 2) ∧ strategy_jenn k ∈ set.Icc 0 n ∧ (∀ j < k, |strategy_jenn j - strategy_jenn k| > 2)) →
  (∃ m, strategy_jenn m = strategy_bela m) →
  (∃ i, ∀ j < i, strategy_bela j < 0)

theorem bela_wins (n : ℕ) (h : n > 10) : bela_always_wins n h :=
sorry

end bela_wins_l630_630665


namespace smallest_stable_triangle_side_length_l630_630853

/-- The smallest possible side length that can appear in any stable triangle with side lengths that 
are multiples of 5, 80, and 112, respectively, is 20. -/
theorem smallest_stable_triangle_side_length {a b c : ℕ} 
  (hab : ∃ k₁, a = 5 * k₁) 
  (hbc : ∃ k₂, b = 80 * k₂) 
  (hac : ∃ k₃, c = 112 * k₃) 
  (abc_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a = 20 ∨ b = 20 ∨ c = 20 :=
sorry

end smallest_stable_triangle_side_length_l630_630853


namespace max_marks_obtainable_l630_630651

theorem max_marks_obtainable 
  (math_pass_percentage : ℝ := 36 / 100)
  (phys_pass_percentage : ℝ := 40 / 100)
  (chem_pass_percentage : ℝ := 45 / 100)
  (math_marks : ℕ := 130)
  (math_fail_margin : ℕ := 14)
  (phys_marks : ℕ := 120)
  (phys_fail_margin : ℕ := 20)
  (chem_marks : ℕ := 160)
  (chem_fail_margin : ℕ := 10) : 
  ∃ max_total_marks : ℤ, max_total_marks = 1127 := 
by 
  sorry  -- Proof not required

end max_marks_obtainable_l630_630651


namespace coefficient_of_x6_in_expansion_l630_630004

theorem coefficient_of_x6_in_expansion : 
  let f := (2 * x^3 + 1 / x)^6 in 
  ∃ c : ℕ, c = 160 ∧ coefficient (x^6) f = c :=
by
  sorry

end coefficient_of_x6_in_expansion_l630_630004


namespace decimal_to_binary_equivalent_123_l630_630530

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630530


namespace find_equation_of_l_l630_630293

def l1 (x y : ℝ) : Prop := 3 * x - 5 * y - 10 = 0
def l2 (x y : ℝ) : Prop := x + y + 1 = 0
def l3 (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem find_equation_of_l (x y : ℝ) : 
  (∃ p : ℝ × ℝ, l1 p.1 p.2 ∧ l2 p.1 p.2) ∧ (∃ k, k = 2 ∧ l x y = true) → 
    16 * x - 8 * y - 23 = 0 :=
sorry

end find_equation_of_l_l630_630293


namespace flour_needed_correct_l630_630049

-- Define the total flour required and the flour already added
def total_flour : ℕ := 8
def flour_already_added : ℕ := 2

-- Define the equation to determine the remaining flour needed
def flour_needed : ℕ := total_flour - flour_already_added

-- Prove that the flour needed to be added is 6 cups
theorem flour_needed_correct : flour_needed = 6 := by
  sorry

end flour_needed_correct_l630_630049


namespace largest_percentage_increase_after_1990_l630_630114

noncomputable def sales : List (Nat × Nat) := [
  (1990, 200000),
  (1992, 300000),
  (1994, 450000),
  (1996, 475000),
  (1998, 700000)
]

def percentage_increase (p1 p2 : Nat × Nat) : ℚ :=
  let (year1, sales1) := p1
  let (year2, sales2) := p2
  if sales1 = 0 then 0 else ((sales2 - sales1) : ℚ) / (sales1 : ℚ) * 100

def max_percentage_increase_year (sales : List (Nat × Nat)) : Nat :=
  let percentages := List.map (λ p, percentage_increase p.1 p.2) (List.zip sales (sales.tail))
  let max_increase := List.maximum percentages
  let idx := List.indexOf percentages max_increase
  sales.nth idx + 2 |>.fst  -- since it is nth + 2 year

theorem largest_percentage_increase_after_1990 : max_percentage_increase_year sales = 1992 :=
by
  sorry

end largest_percentage_increase_after_1990_l630_630114


namespace sum_of_squares_l630_630081

theorem sum_of_squares (x y z : ℕ) (h1 : x + y + z = 30)
  (h2 : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12) :
  x^2 + y^2 + z^2 = 504 :=
by
  sorry

end sum_of_squares_l630_630081


namespace find_A_B_l630_630695

-- Define the condition that A and B have the same number of digits
def same_number_of_digits (A B : ℕ) : Prop :=
  (A > 0) ∧ (B > 0) ∧ (A / 10^(nat.log10 A) = 1) ∧ (B / 10^(nat.log10 B) = 1) ∧ (nat.log10 A = nat.log10 B)

-- Define the condition that 2 * A * B equals the concatenation of A and B
def concatenated_value (A B : ℕ) : ℕ :=
  A * 10^(nat.log10 B + 1) + B

def satisfies_equation (A B : ℕ) : Prop :=
  2 * A * B = concatenated_value A B

-- Statement asserting the only solutions to these conditions are (3, 6) and (13, 52)
theorem find_A_B :
  ∀ A B : ℕ,
  same_number_of_digits A B →
  satisfies_equation A B →
  (A = 3 ∧ B = 6) ∨ (A = 13 ∧ B = 52) :=
by sorry

end find_A_B_l630_630695


namespace systematic_sampling_example_l630_630504

theorem systematic_sampling_example
    (total_students : ℕ)
    (selected_students : ℕ)
    (start_number : ℕ)
    (group_number : ℕ)
    (range_low : ℕ)
    (range_high : ℕ)
    (interval : ℕ)
    (target_number : ℕ) :
    total_students = 900 →
    selected_students = 50 →
    start_number = 7 →
    group_number = 2 →
    range_low = 37 →
    range_high = 54 →
    interval = total_students / selected_students →
    target_number = start_number + group_number * interval →
    target_number = 43 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6] at h7
  have h8a : 900 / 50 = 18 := by norm_num
  rw [h8a] at h7
  have h8b : 7 + 2 * 18 = 43 := by norm_num
  rw [h8b] at h7
  exact h8.symm

end systematic_sampling_example_l630_630504


namespace tetrahedron_volume_l630_630237

variables {P Q R S : ℝ × ℝ × ℝ}

-- Given conditions
def PQ : ℝ := 3
def PR : ℝ := 4
def PS : ℝ := 5
def QR : ℝ := Real.sqrt 17
def QS : ℝ := 2 * Real.sqrt 10
def RS : ℝ := Real.sqrt 29

-- Volume calculation
theorem tetrahedron_volume :
  ∀ P Q R S : ℝ × ℝ × ℝ, 
  dist P Q = PQ →
  dist P R = PR →
  dist P S = PS →
  dist Q R = QR →
  dist Q S = QS →
  dist R S = RS →
  volume_tetrahedron P Q R S = 6 :=
by
  intros,
  -- Proof needs to be provided.
  sorry

noncomputable def volume_tetrahedron (P Q R S : ℝ × ℝ × ℝ) : ℝ := 
  (1 / 6) * abs ((vector_triple_product (to_vector P R) (to_vector P Q) (to_vector P S)).det)

def to_vector (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

def vector_triple_product (u v w : ℝ × ℝ × ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![u.1 * (v.2 * w.3 - v.3 * w.2) - u.2 * (v.1 * w.3 - v.3 * w.1) + u.3 * (v.1 * w.2 - v.2 * w.1)]]

end tetrahedron_volume_l630_630237


namespace values_of_N_l630_630726

theorem values_of_N (N : ℕ) : (∃ k, k ∈ ({4, 6, 8, 12, 16, 24, 48} : set ℕ) ∧ k = N + 3) ↔ (N ∈ {1, 3, 5, 9, 13, 21, 45} : set ℕ) :=
by 
  sorry

#eval values_of_N 4 -- Example usage: should give true if N = 1

end values_of_N_l630_630726


namespace no_tetrahedron_formed_l630_630288

noncomputable def can_form_tetrahedron (triangles : List (ℝ × ℝ × ℝ)) : Bool :=
  sorry -- This would include the actual implementation of checking tetrahedron formation

def triangles_set1 : List (ℝ × ℝ × ℝ) := [(3, 4, 5), (3, 4, 5)]
def triangles_set2 : List (ℝ × ℝ × ℝ) := [(4, 5, sqrt 41), (4, 5, sqrt 41), (4, 5, sqrt 41), (4, 5, sqrt 41)]
def triangles_set3 : List (ℝ × ℝ × ℝ) := [(5 / 6 * sqrt 2, 4, 5), (5 / 6 * sqrt 2, 4, 5), (5 / 6 * sqrt 2, 4, 5),
                                             (5 / 6 * sqrt 2, 4, 5), (5 / 6 * sqrt 2, 4, 5), (5 / 6 * sqrt 2, 4, 5)]

def all_triangles : List (ℝ × ℝ × ℝ) := triangles_set1 ++ triangles_set2 ++ triangles_set3

theorem no_tetrahedron_formed : can_form_tetrahedron all_triangles = false :=
  sorry

end no_tetrahedron_formed_l630_630288


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630153

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630153


namespace base_2_representation_of_123_l630_630516

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630516


namespace reciprocal_of_neg_2023_l630_630484

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l630_630484


namespace base_2_representation_of_123_is_1111011_l630_630524

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630524


namespace find_m_range_l630_630303

/--
Given:
1. Proposition \( p \) (p): The equation \(\frac{x^2}{2} + \frac{y^2}{m} = 1\) represents an ellipse with foci on the \( y \)-axis.
2. Proposition \( q \) (q): \( f(x) = \frac{4}{3}x^3 - 2mx^2 + (4m-3)x - m \) is monotonically increasing on \((-\infty, +\infty)\).

Prove:
If \( \neg p \land q \) is true, then the range of values for \( m \) is \( [1, 2] \).
-/

def p (m : ℝ) : Prop :=
  m > 2

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, (4 * x^2 - 4 * m * x + 4 * m - 3) >= 0

theorem find_m_range (m : ℝ) (hpq : ¬ p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end find_m_range_l630_630303


namespace common_volume_of_tetrahedra_l630_630836

open Real

noncomputable def volume_of_common_part (a b c : ℝ) : ℝ :=
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12))

theorem common_volume_of_tetrahedra (a b c : ℝ) :
  volume_of_common_part a b c =
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12)) :=
by sorry

end common_volume_of_tetrahedra_l630_630836


namespace proof_problem_l630_630324

open Classical

variable (x a : ℝ)

def f (x : ℝ) : ℝ := log (log (0.5 * x - 1))

def A : Set ℝ := {x | 2 < x ∧ x < 4}

def B : Set ℝ := {x | x < 1 ∨ x ≥ 3}

def RB : Set ℝ := {x | 1 ≤ x ∧ x < 3}

theorem proof_problem 
    (h1 : ∀ x, (2 < x ∧ x < 4) ↔ x ∈ A)
    (h2 : ∀ x, (x < 1 ∨ x ≥ 3) ↔ x ∈ B)
    (h3 : ∀ x, (1 ≤ x ∧ x < 3) ↔ x ∈ RB)
    (h4 : 2^a ∈ A)
    (h5 : log 2 (2 * a - 1) ∈ B) :
  (A ∪ B = {x | x < 1} ∪ {x | x > 2}) ∧ 
  (RB ∩ A = {x | 2 < x ∧ x < 3}) ∧ 
  (1 < a ∧ a < 1.5) :=
by
  sorry


end proof_problem_l630_630324


namespace dice_probability_pair_not_four_of_a_kind_l630_630686

/-- The probability that there is at least one pair but not a four-of-a-kind 
    when rolling six standard six-sided dice. -/
theorem dice_probability_pair_not_four_of_a_kind :
  let total_outcomes := 6^6,
      one_pair_four_diff := 6 * Nat.choose 6 2 * (5 * 4 * 3 * 2),
      two_pairs_two_diff := Nat.choose 6 2 * (Nat.choose 6 2 * Nat.choose 4 2) * (4 * 3) in
  one_pair_four_diff + two_pairs_two_diff = 27000 ∧
  (one_pair_four_diff + two_pairs_two_diff) / total_outcomes = 625 / 1089 := by
  sorry

end dice_probability_pair_not_four_of_a_kind_l630_630686


namespace specific_partition_line_exists_l630_630058

noncomputable def exists_partition_line (n : ℕ) (points : Finset (ℝ × ℝ)) (red_points : Finset (ℝ × ℝ)) (green_points : Finset (ℝ × ℝ)) : Prop :=
  (points.card = 4 * n + 2) ∧
  (red_points.card = 2 * n + 1) ∧
  (green_points.card = 2 * n + 1) ∧
  (∀ p1 p2 p3 ∈ points, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬(LineThrough p1 p2).through p3) ∧
  (red_points ∪ green_points = points) ∧
  ∃ (ℓ : Line), ∃ (r g : (ℝ × ℝ)), r ∈ red_points ∧ g ∈ green_points ∧ ℓ.through r ∧ ℓ.through g ∧
  (∃ (side : ℝ × ℝ → bool),
    (side r = side g) ∧
    (red_points.filter (λ p, side p = side r)).card = n ∧
    (green_points.filter (λ p, side p = side r)).card = n)

theorem specific_partition_line_exists (n : ℕ) (points : Finset (ℝ × ℝ)) (red_points : Finset (ℝ × ℝ)) (green_points : Finset (ℝ × ℝ)) :
  exists_partition_line n points red_points green_points :=
sorry

end specific_partition_line_exists_l630_630058


namespace hotel_towels_l630_630611

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l630_630611


namespace remainder_3_pow_2023_mod_5_l630_630156

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l630_630156


namespace triangle_area_opa_l630_630767

theorem triangle_area_opa (k m : ℝ)
  (h₁ : ∀ (x : ℝ), y = k * x + 4 ∧ P = (1, m))
  (h₂ : k = -2)
  (h₃ : ∀ (x : ℝ), y = 0 ∧ A = (x, 0)) :
  let O : (ℝ × ℝ) := (0, 0),
      P : (ℝ × ℝ) := (1, 2),
      A : (ℝ × ℝ) := (2, 0) in
  1 / 2 * (A.1 - O.1) * (P.2 - O.2) = 2 :=
sorry

end triangle_area_opa_l630_630767


namespace smallest_positive_period_monotonically_decreasing_l630_630225

theorem smallest_positive_period_monotonically_decreasing :
  ∃ f : ℝ → ℝ, (∀ x, f x = |sin x| ∧ (periodic f π))
               ∧ (∀ x ∈ Ioo (π / 2) π, monotone_decreasing_on f (Ioo (π / 2) π) x) := 
sorry

end smallest_positive_period_monotonically_decreasing_l630_630225


namespace am_gm_problem_l630_630109

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end am_gm_problem_l630_630109


namespace min_value_of_expr_l630_630063

theorem min_value_of_expr : ∃ (a b : ℕ), (0 < a ∧ a < 6) ∧ (0 < b ∧ b < 10) ∧ (2 * a - a * b = -35) := 
by {
    sorry
}

end min_value_of_expr_l630_630063


namespace probability_red_side_given_observed_l630_630982

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end probability_red_side_given_observed_l630_630982


namespace acute_triangle_inequality_l630_630830

variables {A B C : ℝ}

theorem acute_triangle_inequality
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : A + B + C = π) :
  (cos A ^ 2 / cos B ^ 2) + (cos B ^ 2 / cos C ^ 2) + (cos C ^ 2 / cos A ^ 2) 
    ≥ 4 * (cos A ^ 2 + cos B ^ 2 + cos C ^ 2) := sorry

end acute_triangle_inequality_l630_630830


namespace cricket_average_score_l630_630455

theorem cricket_average_score (avg1 avg2 matches1 matches2 : ℕ)
  (H1 : matches1 = 2)
  (H2 : avg1 = 27)
  (H3 : matches2 = 3)
  (H4 : avg2 = 32) :
  (matches1 + matches2 = 5 ∧ (avg1 * matches1 + avg2 * matches2) / (matches1 + matches2) = 30) :=
begin
  sorry
end

end cricket_average_score_l630_630455


namespace club_members_addition_l630_630592

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l630_630592


namespace part1_part2_l630_630190

-- Part (1)
theorem part1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : (x - x^2) < sin x ∧ sin x < x := sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, f = λ x, cos (a * x) - log (1 - x^2) ∧ ∀ x, f 0 = 0 ∧ f' x | x = 0 = 0 ∧ f'' x | x = 0 < 0) : a < -sqrt 2 ∨ a > sqrt 2 := sorry

end part1_part2_l630_630190


namespace find_divisor_l630_630562

theorem find_divisor (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
    (h1 : y % x = 5)
    (h2 : (3 * y) % x = 6) : x = 9 :=
begin
  sorry
end

end find_divisor_l630_630562


namespace find_constants_l630_630694

noncomputable def existence_of_polynomials (a b : ℝ) : Prop :=
∃ (p q : ℝ → ℝ), ∀ x : ℝ, p(x^2) * q(x+1) - p(x+1) * q(x^2) = x^2 + a * x + b

theorem find_constants : existence_of_polynomials (-1) (-1) :=
sorry

end find_constants_l630_630694


namespace cyclic_points_of_quadrilateral_l630_630874

variables {A B C D P Q R S : Type*}

-- Definitions for points and geometric conditions
variables [convex_quadrilateral A B C D]
variables [bisector B D (angle A B C)]
variables [circumcircle_intersect (triangle A B C) (line A D) P]
variables [circumcircle_intersect (triangle A B C) (line C D) Q]
variables [parallel_line_through D (line A C)]
variables [intersection (line_through D parallel_to (line A C)) (line B C) R]
variables [intersection (line_through D parallel_to (line A C)) (line B A) S]

theorem cyclic_points_of_quadrilateral :
  is_cyclic_quadrilateral P Q R S :=
sorry

end cyclic_points_of_quadrilateral_l630_630874


namespace sum_deficient_eq_2026_l630_630286

-- Define the sequence a_n
def a (n : ℕ) : ℝ := log (n + 2) / log (n + 1)

-- Define what it means to be a deficient number
def is_deficient (n : ℕ) : Prop := ∀ m ≤ n, ∃ k : ℤ, 
  log (m + 2) / log 2 = k

-- Define the sum of deficient numbers
noncomputable def sum_deficient_numbers : ℕ :=
  (∑ m in (Finset.range 2018).filter (λ n, is_deficient n), n)

-- The main theorem
theorem sum_deficient_eq_2026 : sum_deficient_numbers = 2026 :=
sorry

end sum_deficient_eq_2026_l630_630286


namespace parallelogram_angle_equality_l630_630575

variables {A B C D E F : Type}
variables [MetricSpace A] [InnerProductSpace ℝ A]

-- Definitions of points, conditions, and the statement
def is_parallelogram (A B C D : A) : Prop := parallel (line A B) (line C D) ∧ parallel (line A D) (line B C) ∧ dist A B = dist C D ∧ dist A D = dist B C

def point_on (p q : A) (r : A) : Prop := (∃ t : ℝ, t > 0 ∧ p + t • (q - p) = r)

def segment_extension (p q r : A) : Prop := q ∈ (line p r) ∧ dist q r = dist p q ∧ dist p q + dist q r = dist p r

def perpendicular (l1 l2 : Set A) : Prop := ∃ ⟨v1, _, _⟩ ∈ l1, ∃ ⟨v2, _, _⟩ ∈ l2, inner_product v1 v2 = 0

noncomputable def intersect (l1 l2 : Set A) : A := sorry

def angle_equal (α β : ℝ) : Prop := α = β

def angle_DAF_BAF (D A F B : A) : Prop := angle (line D A) (line D F) = angle (line B A) (line B F)

theorem parallelogram_angle_equality (A B C D E F : A) :
  is_parallelogram A B C D →
  segment_extension A B E →
  perpendicular (line C (intersect (line C (perpendicular_line (line B D))) (line E (perpendicular_line (line A B))))) (line B D) →
  perpendicular (line E (intersect (line C (perpendicular_line (line B D))) (line E (perpendicular_line (line A B))))) (line A B) →
  angle_DAF_BAF D A (intersect (line C (perpendicular_line (line B D))) (line E (perpendicular_line (line A B)))) B :=
sorry

end parallelogram_angle_equality_l630_630575


namespace tan_half_angle_product_l630_630028

theorem tan_half_angle_product {a b c e : ℝ} {α β : ℝ} (h_ellipse : e = c / a) (h_c : c = Real.sqrt (a^2 - b^2))
(h_angle1 : ∀ P : ℝ × ℝ, P ∈ {p | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1} → ∀ F1 F2 : ℝ × ℝ, True)  
(h_angle2 : ∀ P : ℝ × ℝ, P ∈ {p | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1} → ∀ F1 F2 : ℝ × ℝ, True) : 
  tan (α / 2) * tan (β / 2) = (1 - e) / (1 + e) :=
sorry

end tan_half_angle_product_l630_630028


namespace bisect_angle_AX_by_angle_bisector_l630_630189

-- Define the conditions and theorem
variables (A B C E F B' E' C' F' X A' : Type) 
          [acute_triangle ABC]
          [angle_eq A 60]
          [altitude BE F]
          [altitude CF]
          [reflection BE_perp_bisector_BC BE' E']
          [reflection CF_perp_bisector_BC CF' F']
          [intersection B'E' C'F' X]
          [reflection A_BC A']

-- Define the theorem we aim to prove
theorem bisect_angle_AX_by_angle_bisector :
  bisects (angle_bisector A) AX :=
sorry

end bisect_angle_AX_by_angle_bisector_l630_630189


namespace find_min_k_l630_630207

open Real

def essentially_increasing (f : ℝ → ℝ) : Prop :=
  ∀ s t : ℝ, s ≤ t → f s ≠ 0 → f t ≠ 0 → f s ≤ f t

theorem find_min_k :
  ∃ (k : ℕ), k = 11 ∧ ∀ (x : Fin 2022 → ℝ),
  ∃ (fs : Fin k → ℝ → ℝ), 
    (∀ i, essentially_increasing (fs i)) ∧
    (∀ n : Fin 2022, ∑ i in Finset.range k, fs i n = x n) :=
sorry

end find_min_k_l630_630207


namespace remainder_3_pow_2023_mod_5_l630_630155

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l630_630155


namespace pins_after_one_month_l630_630887

def avg_pins_per_day : ℕ := 10
def delete_pins_per_week_per_person : ℕ := 5
def group_size : ℕ := 20
def initial_pins : ℕ := 1000

theorem pins_after_one_month
  (avg_pins_per_day_pos : avg_pins_per_day = 10)
  (delete_pins_per_week_per_person_pos : delete_pins_per_week_per_person = 5)
  (group_size_pos : group_size = 20)
  (initial_pins_pos : initial_pins = 1000) : 
  1000 + (avg_pins_per_day * group_size * 30) - (delete_pins_per_week_per_person * group_size * 4) = 6600 :=
by
  sorry

end pins_after_one_month_l630_630887


namespace charles_whistles_l630_630443

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end charles_whistles_l630_630443


namespace number_of_groups_eq_five_l630_630124

-- Define conditions
def total_eggs : ℕ := 35
def eggs_per_group : ℕ := 7

-- Statement to prove the number of groups
theorem number_of_groups_eq_five : total_eggs / eggs_per_group = 5 := by
  sorry

end number_of_groups_eq_five_l630_630124


namespace proposition_p_l630_630739

def f (x : ℝ) : ℝ := (2 / 3) ^ x

theorem proposition_p : ∀ x : ℝ, 0 ≤ x → f x ≤ 1 :=
by
  intro x hx
  sorry

end proposition_p_l630_630739


namespace sum_of_first_8_terms_of_geometric_sequence_l630_630240

theorem sum_of_first_8_terms_of_geometric_sequence :
  ∀ (a : ℕ → ℝ) (q : ℝ), 
  (a 0 = 27) ∧ (a 8 = 1 / 243) ∧ (∀ n, a (n + 1) = a n * q) ∧ (q < 0)
  → ∑ i in Finset.range 8, a i = 1640 / 81 :=
by
  intros a q h
  sorry

end sum_of_first_8_terms_of_geometric_sequence_l630_630240


namespace base_2_representation_of_123_l630_630549

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630549


namespace total_calories_l630_630016

theorem total_calories (chips : ℕ) (cheezits : ℕ) (chip_calories_total : ℕ) (cheezit_ratio : ℚ) 
  (h1 : chips = 10) 
  (h2 : cheezits = 6) 
  (h3 : chip_calories_total = 60)
  (h4 : cheezit_ratio = 1/3) :
  ∃ (total_calories : ℕ), total_calories = 108 :=
by
  let chip_calories := chip_calories_total / chips
  let cheezit_calories := chip_calories + (cheezit_ratio * chip_calories).toNat
  let total_cheezit_calories := cheezit_calories * cheezits
  let total_chip_calories := chip_calories_total
  let total_calories := total_chip_calories + total_cheezit_calories
  use total_calories
  sorry

end total_calories_l630_630016


namespace domain_of_f_l630_630704

noncomputable def f (x : ℝ) : ℝ := real.sqrt (real.sin x) + real.sqrt (1/2 - real.cos x)

theorem domain_of_f {x : ℝ}:
  (∃ (k : ℤ), (2 * k * real.pi + real.pi / 3 ≤ x ∧ x ≤ 2 * k * real.pi + real.pi)) ↔ (f x = f x) :=
sorry

end domain_of_f_l630_630704


namespace triangle_inequality_convex_curves_l630_630870

variable (A B C : Type) [ConvexCurve A] [ConvexCurve B] [ConvexCurve C]

noncomputable def distance (A B : Type) [ConvexCurve A] [ConvexCurve B] : ℝ :=
  sorry  -- Assume a function that defines the distance between two convex curves

theorem triangle_inequality_convex_curves :
  distance A B + distance B C ≥ distance A C ∧
  distance A C + distance B C ≥ distance A B ∧
  distance A B + distance A C ≥ distance B C :=
sorry

end triangle_inequality_convex_curves_l630_630870


namespace smaller_angle_at_9_am_l630_630662

-- Define the angular positions of the minute and hour hands
def minute_hand_angle (minute : Nat) : ℕ := 0  -- At the 12 position
def hour_hand_angle (hour : Nat) : ℕ := hour * 30  -- 30 degrees per hour

-- Define the function to get the smaller angle between two angles on the clock from 0 to 360 degrees
def smaller_angle (angle1 angle2 : ℕ) : ℕ :=
  let angle_diff := Int.natAbs (angle1 - angle2)
  min angle_diff (360 - angle_diff)

-- The theorem to prove
theorem smaller_angle_at_9_am : smaller_angle (minute_hand_angle 0) (hour_hand_angle 9) = 90 := sorry

end smaller_angle_at_9_am_l630_630662


namespace total_oranges_picked_l630_630431

theorem total_oranges_picked : 
  let M := 100 in
  let T := 3 * M in
  let W := 70 in
  M + T + W = 470 :=
by
  let M := 100
  let T := 3 * M
  let W := 70
  show M + T + W = 470
  sorry

end total_oranges_picked_l630_630431


namespace range_of_x_sqrt_ineq_l630_630044

-- Define the real numbers x and y
variables (x y : ℝ)

-- Condition: x + y/4 = 1
def cond1 := x + y / 4 = 1

-- Condition: |7 - y| < 2 * x + 3
def cond2 := abs (7 - y) < 2 * x + 3

-- Statement 1: -1 < x < 0
theorem range_of_x (h1 : cond1 x y) (h2 : cond2 x y) : -1 < x ∧ x < 0 :=
  sorry

-- Condition: x > 0
def cond3 := x > 0

-- Condition: y > 0
def cond4 := y > 0

-- Statement 2: sqrt(x * y) ≥ x * y
theorem sqrt_ineq (h1 : cond1 x y) (h3 : cond3 x) (h4 : cond4 y) : sqrt (x * y) ≥ x * y :=
  sorry

end range_of_x_sqrt_ineq_l630_630044


namespace sum_cubes_induction_added_term_induction_l630_630140

theorem sum_cubes_induction (n : ℕ) (h : 0 < n) :
  ∑ i in Finset.range (n+1), i^3 = (n^6 + n^3) / 2 := by
sorry

theorem added_term_induction (k : ℕ) (h : 0 < k) :
  ∑ i in Finset.range (k+1) (k+1)^3 = ∑ i in Finset.range ((k+1)^3) := by
sorry

end sum_cubes_induction_added_term_induction_l630_630140


namespace point_in_third_quadrant_l630_630806

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end point_in_third_quadrant_l630_630806


namespace cost_per_page_correct_l630_630385

noncomputable def cost_notebooks : ℝ :=
  let price_notebook := 12 in
  let discount_sec_notebook := 0.5 * price_notebook in
  2 * price_notebook - discount_sec_notebook

noncomputable def cost_pens : ℝ :=
  let price_pens := 9 in
  let discount_pens := 0.2 * price_pens in
  price_pens - discount_pens

noncomputable def cost_folders : ℝ := 19

noncomputable def shipping_cost : ℝ := 5 + 10

noncomputable def subtotal : ℝ :=
  cost_notebooks + cost_pens + cost_folders + shipping_cost

noncomputable def sales_tax : ℝ :=
  0.05 * subtotal

noncomputable def total_cost_usd : ℝ :=
  subtotal + sales_tax

noncomputable def exchange_rate : ℝ := 2.3

noncomputable def total_cost_local : ℝ :=
  total_cost_usd * exchange_rate

noncomputable def total_pages : ℝ := 2 * 50

noncomputable def cost_per_page : ℝ :=
  total_cost_local / total_pages

theorem cost_per_page_correct :
  cost_per_page ≈ 1.43 :=
sorry

end cost_per_page_correct_l630_630385


namespace length_of_train_is_correct_l630_630964

-- Definitions based on conditions
def speed_kmh := 90
def time_sec := 10

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train := speed_ms * time_sec

-- Theorem to prove the length of the train
theorem length_of_train_is_correct : length_of_train = 250 := by
  sorry

end length_of_train_is_correct_l630_630964


namespace unique_lines_count_l630_630795

def point3D := { p : ℕ × ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 3 ∧ 1 ≤ p.3 ∧ p.3 ≤ 3 }

def is_valid_line (a b c : point3D) : Prop :=
  ∃ (u v : ℤ) (a_s b_s c_s : ℕ), 
  let a' := a.1
  let b' := b.1 
  let c' := c.1
  a_s = a'.1 + 2 * u ∧ b_s = a'.2 + 2 * v ∧ c_s = a'.3 + 2 * u + 2 * v ∧
  1 ≤ a_s ∧ a_s ≤ 3 ∧ 1 ≤ b_s ∧ b_s ≤ 3 ∧ 1 ≤ c_s ∧ c_s ≤ 3 ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem unique_lines_count : (finset.univ.filter (λ l : (point3D × point3D × point3D), is_valid_line l.1 l.2 l.3)).card = 40 := 
sorry

end unique_lines_count_l630_630795


namespace math_city_intersections_l630_630425

theorem math_city_intersections : 
  ∀ (n : ℕ), 
  (n = 10) → 
  (∀ (i j : ℕ), 
    (1 ≤ i ∧ i < j ∧ j ≤ n) → 
    straight_street i ∧ straight_street j ∧ ¬are_parallel i j ∧ ¬(three_meet_at_point i j)) → 
  (∃ (count : ℕ), count = 45) := 
by
  intro n hn condition
  have h_count : Σ (i j : ℕ), (i < j ∧ i ≤ n ∧ j ≤ n ∧ i ≠ j) = 45 
    sorry
  use 45
  exact h_count

end math_city_intersections_l630_630425


namespace bowling_ball_weight_l630_630712

def weight_of_canoe : ℕ := 32
def weight_of_canoes (n : ℕ) := n * weight_of_canoe
def weight_of_bowling_balls (n : ℕ) := 128

theorem bowling_ball_weight :
  (128 / 5 : ℚ) = (weight_of_bowling_balls 5 / 5 : ℚ) :=
by
  -- Theorems and calculations would typically be carried out here
  sorry

end bowling_ball_weight_l630_630712


namespace sufficient_condition_l630_630742

-- Define the types for lines and planes
variables {Line Plane : Type}

-- Define the conditions
variables {m n : Line} {α β : Plane}
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β
axiom parallel_planes : α ∥ β
axiom perpendicular_line_to_plane : n ⊥ β

-- Define the proposition to prove
theorem sufficient_condition (m n : Line) (α β : Plane) (h_mn : m ≠ n) (h_ab : α ≠ β)
  (h_parallel : α ∥ β) (h_perpendicular : n ⊥ β) : n ⊥ α :=
sorry

end sufficient_condition_l630_630742


namespace inequality_ac2_bc2_implies_a_b_l630_630715

theorem inequality_ac2_bc2_implies_a_b (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
sorry

end inequality_ac2_bc2_implies_a_b_l630_630715


namespace real_part_of_one_over_one_minus_z_l630_630415

open Complex

noncomputable def real_part_fraction {z : ℂ} (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) : ℝ :=
  re (1 / (1 - z))

theorem real_part_of_one_over_one_minus_z (z : ℂ) (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) :
  real_part_fraction hz1 hz2 = 1 / 2 :=
by
  sorry

end real_part_of_one_over_one_minus_z_l630_630415


namespace length_breadth_difference_l630_630932

theorem length_breadth_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 288) : L - W = 12 :=
by
  sorry

end length_breadth_difference_l630_630932


namespace parallel_implies_not_contained_l630_630056

variables {Line Plane : Type} (l : Line) (α : Plane)

-- Define the predicate for a line being parallel to a plane
def parallel (l : Line) (α : Plane) : Prop := sorry

-- Define the predicate for a line not being contained in a plane
def not_contained (l : Line) (α : Plane) : Prop := sorry

theorem parallel_implies_not_contained (l : Line) (α : Plane) (h : parallel l α) : not_contained l α :=
sorry

end parallel_implies_not_contained_l630_630056


namespace find_f_one_l630_630314

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) (hx : x > 0) : f x ∈ ℝ

axiom f_monotone : monotone f

axiom f_condition (x : ℝ) (hx : x > 0) : f (f x - real.log x) = 1 + real.exp 1

theorem find_f_one : f 1 = real.exp 1 := 
by 
  sorry

end find_f_one_l630_630314


namespace jo_page_an_hour_ago_l630_630843

variables (total_pages current_page hours_left : ℕ)
variables (steady_reading_rate : ℕ)
variables (page_an_hour_ago : ℕ)

-- Conditions
def conditions := 
  steady_reading_rate * hours_left = total_pages - current_page ∧
  total_pages = 210 ∧
  current_page = 90 ∧
  hours_left = 4 ∧
  page_an_hour_ago = current_page - steady_reading_rate

-- Theorem to prove that Jo was on page 60 an hour ago
theorem jo_page_an_hour_ago (h : conditions total_pages current_page hours_left steady_reading_rate page_an_hour_ago) : 
  page_an_hour_ago = 60 :=
sorry

end jo_page_an_hour_ago_l630_630843


namespace num_heavy_tailed_perms_l630_630623

open Finset

/-- 
  Define a heavy-tailed permutation as one in which the sum of 
  the first three elements is less than the sum of the last two elements.
  Calculate the total number of such permutations for the set {1, 2, 3, 4, 6}.
-/
def is_heavy_tailed (l : List ℕ) : Prop :=
  l.length = 5 ∧ l.nthLe 0 sorry + l.nthLe 1 sorry + l.nthLe 2 sorry < l.nthLe 3 sorry + l.nthLe 4 sorry

theorem num_heavy_tailed_perms :
  (univ.permutations.filter is_heavy_tailed).card = 30 :=
sorry

end num_heavy_tailed_perms_l630_630623


namespace no_real_roots_x2_plus_4_l630_630478

theorem no_real_roots_x2_plus_4 : ¬ ∃ x : ℝ, x^2 + 4 = 0 := by
  sorry

end no_real_roots_x2_plus_4_l630_630478


namespace square_traffic_sign_perimeter_l630_630634

-- Define the side length of the square
def side_length : ℕ := 4

-- Define the number of sides of the square
def number_of_sides : ℕ := 4

-- Define the perimeter of the square
def perimeter (l : ℕ) (n : ℕ) : ℕ := l * n

-- The theorem to be proved
theorem square_traffic_sign_perimeter : perimeter side_length number_of_sides = 16 :=
by
  sorry

end square_traffic_sign_perimeter_l630_630634


namespace car_speed_l630_630201

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_l630_630201


namespace alex_min_correct_answers_l630_630087

-- Define the number of problems and points system
def total_problems := 25
def points_per_correct := 7
def points_per_incorrect := 0
def points_per_unanswered := 2

-- Alex attempts the first 20 problems and leaves the last 5 unanswered
def attempted_problems := 20
def unanswered_problems := total_problems - attempted_problems
def minimum_total_points := 120

-- The condition to meet in order to score at least 120 points
theorem alex_min_correct_answers (x : ℕ) :
  (unanswered_problems * points_per_unanswered) + (x * points_per_correct) ≥ minimum_total_points →
  x ≥ 16 :=
begin
  -- Substituting numbers into the theorem simplifies the proof
  unfold total_problems,
  unfold points_per_correct,
  unfold points_per_incorrect,
  unfold points_per_unanswered,
  unfold attempted_problems,
  unfold unanswered_problems,
  unfold minimum_total_points,
  sorry
end

end alex_min_correct_answers_l630_630087


namespace regular_hexagon_area_quotient_l630_630460

theorem regular_hexagon_area_quotient (ABCDEF : Polygon) 
  (h_hex : is_regular_hexagon ABCDEF)
  (h_triangle : is_equilateral_triangle ACD) :
  (area ABCDEF) / (area ACD) = 3 :=
sorry

end regular_hexagon_area_quotient_l630_630460


namespace diagonal_length_not_possible_l630_630327

-- Define the side lengths of the parallelogram
def sides_of_parallelogram : ℕ × ℕ := (6, 8)

-- Define the length of a diagonal that cannot exist
def invalid_diagonal_length : ℕ := 15

-- Statement: Prove that a diagonal of length 15 cannot exist for such a parallelogram.
theorem diagonal_length_not_possible (a b d : ℕ) 
  (h₁ : sides_of_parallelogram = (a, b)) 
  (h₂ : d = invalid_diagonal_length) 
  : d ≥ a + b := 
sorry

end diagonal_length_not_possible_l630_630327


namespace spot_reachable_area_l630_630452

noncomputable def hexagonal_doghouse_area (side_length tether_length : ℝ) : ℝ :=
  let large_sector_area := π * tether_length ^ 2 * (240 / 360)
  let small_sector_area := π * (side_length ^ 2) * (60 / 360)
  large_sector_area + 2 * small_sector_area

theorem spot_reachable_area :
  hexagonal_doghouse_area 1.5 3 = 6.75 * π :=
by
  sorry

end spot_reachable_area_l630_630452


namespace g_10_plus_g_neg10_eq_6_l630_630864

variable (a b c : ℝ)
noncomputable def g : ℝ → ℝ := λ x => a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + 5

theorem g_10_plus_g_neg10_eq_6 (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 :=
by
  -- Proof goes here
  sorry

end g_10_plus_g_neg10_eq_6_l630_630864


namespace part1_part2_l630_630789

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α] [LinearOrderedField α]

def A : Set α := {x | x^2 - 5 * x - 6 < 0}
def B : Set α := {x | 6 * x^2 - 5 * x + 1 >= 0}
def C (m : α) : Set α := {x | (x - m) / (x - m - 9) < 0}

theorem part1 : A ∩ B = {x | -1 < x ∧ x ≤ (1 / 3) ∨ (1 / 2) ≤ x ∧ x < 6} := 
sorry

theorem part2 (m : α) : (A ⊆ C m) → -3 ≤ m ∧ m ≤ -1 := 
sorry

end part1_part2_l630_630789


namespace smallest_n_Sn_gt_zero_l630_630769

variables (a : ℕ → ℝ) (d : ℝ)

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def S (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ :=
n * a 1 + (n * (n - 1) / 2) * d

theorem smallest_n_Sn_gt_zero (a : ℕ → ℝ) (d : ℝ) (h_arith : arithmetic_sequence a d)
  (h1 : a 1 < 0) (h2 : a 8 + a 9 > 0) (h3 : a 8 * a 9 < 0) : ∃ n, S a d 16 > 0 ∧ ∀ m < 16, S a d m ≤ 0 :=
sorry

end smallest_n_Sn_gt_zero_l630_630769


namespace male_students_25_or_older_percentage_l630_630817

noncomputable def percentage_of_male_students_25_or_older 
  (T : ℝ) 
  (male_percentage : ℝ) 
  (female_percentage_25_or_older : ℝ) 
  (probability_under_25 : ℝ) : ℝ := 
    have prob_male_under_25 : ℝ := (1 - M_25) * male_percentage, from sorry,
    have prob_female_under_25 : ℝ := 0.7 * 0.6, from sorry, 
    have total_prob_under_25 : ℝ := prob_male_under_25 + prob_female_under_25, from sorry,
    if total_prob_under_25 = probability_under_25 then 1 - 0.6 else sorry

theorem male_students_25_or_older_percentage (T : ℝ) : 
  percentage_of_male_students_25_or_older T 0.40 0.30 0.66 = 0.40 :=
by sorry

end male_students_25_or_older_percentage_l630_630817


namespace sum_of_coeffs_l630_630918

-- Define the polynomial with real coefficients
def poly (p q r s : ℝ) : Polynomial ℝ := Polynomial.C 1 + Polynomial.C p * Polynomial.X + Polynomial.C q * Polynomial.X^2 + Polynomial.C r * Polynomial.X^3 + Polynomial.C s * Polynomial.X^4

-- Given conditions
def g (x : ℂ) : Polynomial ℂ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem sum_of_coeffs (p q r s : ℝ)
  (h1 : g (Complex.I * 3) = 0)
  (h2 : g (1 + 2 * Complex.I) = 0) :
  p + q + r + s = -41 :=
sorry

end sum_of_coeffs_l630_630918


namespace workers_cut_down_correct_l630_630126

def initial_oak_trees : ℕ := 9
def remaining_oak_trees : ℕ := 7
def cut_down_oak_trees : ℕ := initial_oak_trees - remaining_oak_trees

theorem workers_cut_down_correct : cut_down_oak_trees = 2 := by
  sorry

end workers_cut_down_correct_l630_630126


namespace points_on_same_sphere_l630_630006

structure Point (α : Type) := (x y z : α)
structure Tetrahedron (α : Type) := (S A B C : Point α)
structure EdgePoint (α : Type) := (A1 B1 C1 : Point α)

variables {α : Type} [NontrivialField α]

def is_on_edge (p q r : Point α) : Prop := sorry
def equidistant_ratios (S A A1 B B1 C C1 : Point α) : Prop :=
  dist S A * dist S A1 = dist S B * dist S B1 ∧ 
  dist S A * dist S A1 = dist S C * dist S C1

theorem points_on_same_sphere 
  (S A B C A1 B1 C1 : Point α)
  (h₁ : is_on_edge S A A1)
  (h₂ : is_on_edge S B B1)
  (h₃ : is_on_edge S C C1)
  (h₄ : equidistant_ratios S A A1 B B1 C C1) 
  : coplanar ({A, B, C, A1, B1, C1} : Set (Point α)) := 
sorry

end points_on_same_sphere_l630_630006


namespace max_value_of_f_l630_630775

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (2 * x^2 + 2 * x + 41) - Real.sqrt (2 * x^2 + 4 * x + 4)

theorem max_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(y) ≤ f(x) ∧ f(x) = 5 :=
by
  sorry

end max_value_of_f_l630_630775


namespace clock_strikes_twelve_l630_630335

def clock_strike_interval (strikes : Nat) (time : Nat) : Nat :=
  if strikes > 1 then time / (strikes - 1) else 0

def total_time_for_strikes (strikes : Nat) (interval : Nat) : Nat :=
  if strikes > 1 then (strikes - 1) * interval else 0

theorem clock_strikes_twelve (interval_six : Nat) (time_six : Nat) (time_twelve : Nat) :
  interval_six = clock_strike_interval 6 time_six →
  time_twelve = total_time_for_strikes 12 interval_six →
  time_six = 30 →
  time_twelve = 66 :=
by
  -- The proof will go here
  sorry

end clock_strikes_twelve_l630_630335


namespace repeated_six_product_l630_630272

theorem repeated_six_product : 
  (let x : ℚ := 2 / 3 in x * 8 = 16 / 3) :=
begin
  let x : ℚ := 2 / 3,
  have hx : x = 2 / 3, from rfl,
  suffices : x * 8 = 16 / 3,
  { exact this },
  calc
    x * 8 = (2 / 3) * 8 : by rw hx
        ... = 16 / 3     : by sorry
end

end repeated_six_product_l630_630272


namespace valid_parameterizations_l630_630467

theorem valid_parameterizations :
  (∀ t : ℝ, ∃ x y : ℝ, (x = 0 + 4 * t) ∧ (y = -4 + 8 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = 3 + 1 * t) ∧ (y = 2 + 2 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = -1 + 2 * t) ∧ (y = -6 + 4 * t) ∧ (y = 2 * x - 4)) :=
by
  -- Proof goes here
  sorry

end valid_parameterizations_l630_630467


namespace graph_even_edge_circuit_exists_graph_3n_edges_no_even_circuit_l630_630194

noncomputable theory

open Classical

def exists_even_edge_circuit (n : ℕ) : Prop :=
  ∀ (G : SimpleGraph (Fin (2 * n + 1))),
    G.edgeCount ≥ 3 * n + 1 → ∃ (circuit : G.Walk.Circuit),
      circuit.edges.length % 2 = 0

def nopeven_for_3n_edges (n : ℕ) : Prop :=
  ∃ (G : SimpleGraph (Fin (2 * n + 1))),
    G.edgeCount = 3 * n ∧ ∀ (circuit : G.Walk.Circuit), circuit.edges.length % 2 ≠ 0

theorem graph_even_edge_circuit_exists (n : ℕ) :
  exists_even_edge_circuit n := sorry

theorem graph_3n_edges_no_even_circuit (n : ℕ) :
  nopeven_for_3n_edges n := sorry

end graph_even_edge_circuit_exists_graph_3n_edges_no_even_circuit_l630_630194


namespace length_AB_of_right_triangle_l630_630828

-- Definitions based on conditions
def right_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∃ α : ℝ, α = real.pi / 2

def angle_A (A B C : Type) (α : ℝ) : Prop :=
  α = _ -- fixing angle definition needs context or better structure

def length_BC (A B C : Type) (a : ℝ) (BC : A → B → ℝ) : Prop :=
  BC A C = a

-- Goal to prove using the conditions
theorem length_AB_of_right_triangle {A B C : Type}
    [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (α : ℝ) (a : ℝ)
    (BC : A → B → ℝ) (AB : A → B → ℝ)
    (h1 : right_triangle A B C)
    (h2 : angle_A A B C α)
    (h3 : length_BC A B C a BC) :
    AB A C = a / (real.sin α) := 
sorry

end length_AB_of_right_triangle_l630_630828


namespace sum_exponential_to_polar_l630_630667

theorem sum_exponential_to_polar :
  5 * complex.exp (complex.i * 3 * real.pi / 4) + 5 * complex.exp (-complex.i * 3 * real.pi / 4) = -5 * real.sqrt 2 * complex.exp (complex.i * real.pi) := 
sorry

end sum_exponential_to_polar_l630_630667


namespace count_valid_pairs_l630_630282

open Nat

theorem count_valid_pairs : 
  (∑ y in Icc 1 147, (floor ((150 - y : ℤ) / (↑y * (y + 1) * (y + 2)))) : ℕ) = 33 :=
by
  sorry

end count_valid_pairs_l630_630282


namespace pentagon_area_l630_630508

noncomputable def compute_area (vertices : List (ℝ × ℝ)) : ℝ :=
  let rotated := vertices.concat (List.head! vertices)
  let paired := List.zip vertices rotated.tail
  let cross_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.2) - (a.2 * b.1)
  (1 / 2) * abs (List.sum (paired.map (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), cross_product p.1 p.2)))

-- Defining the vertices
def vertices : List (ℝ × ℝ) := [(3, 1), (1, 4), (6, 7), (8, 3), (5, 2)]

-- The theorem we want to prove
theorem pentagon_area : compute_area vertices = 22 := by
  sorry

end pentagon_area_l630_630508


namespace compare_median_mode_l630_630630

-- Define the dataset
def data : List ℝ := [15, 17, 14, 10, 15, 17, 17, 14, 16, 12]

-- Compute the average
def average (l : List ℝ) : ℝ :=
  (l.foldr ( + ) 0) / (l.length)

-- Compute the median
noncomputable def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· < ·)
  if h : sorted.length % 2 = 0 then
    (sorted.get ⟨(sorted.length / 2) - 1, sorry⟩ + 
     sorted.get ⟨sorted.length / 2, sorry⟩) / 2
  else
    sorted.get ⟨sorted.length / 2, sorry⟩

-- Compute the mode
noncomputable def mode (l : List ℝ) : ℝ :=
  l.groupBy id <| λa b => a = b
  |>.maxBy (·.length)
  |>.head!
  |>.head!


theorem compare_median_mode (m n p : ℝ) (data_averaged : average data = 14.7) 
  (data_med : median data = 15) (data_mod : mode data = 17) :
  m < n < p :=
by {
  rw [← data_averaged, ← data_med, ← data_mod];
  exact ⟨by norm_num, by norm_num⟩
}

end compare_median_mode_l630_630630


namespace is_opposite_if_differ_in_sign_l630_630363

-- Define opposite numbers based on the given condition in the problem:
def opposite_numbers (a b : ℝ) : Prop := a = -b

-- State the theorem based on the translation in c)
theorem is_opposite_if_differ_in_sign (a b : ℝ) (h : a = -b) : opposite_numbers a b := by
  sorry

end is_opposite_if_differ_in_sign_l630_630363


namespace travel_time_home_to_community_center_l630_630344

-- Definitions and assumptions based on the conditions
def time_to_library := 30 -- in minutes
def distance_to_library := 5 -- in miles
def time_spent_at_library := 15 -- in minutes
def distance_to_community_center := 3 -- in miles
noncomputable def cycling_speed := time_to_library / distance_to_library -- in minutes per mile

-- Time calculation to reach the community center from the library
noncomputable def time_from_library_to_community_center := distance_to_community_center * cycling_speed -- in minutes

-- Total time spent to travel from home to the community center
noncomputable def total_time_home_to_community_center :=
  time_to_library + time_spent_at_library + time_from_library_to_community_center

-- The proof statement verifying the total time
theorem travel_time_home_to_community_center : total_time_home_to_community_center = 63 := by
  sorry

end travel_time_home_to_community_center_l630_630344


namespace road_length_trees_l630_630135

theorem road_length_trees (total_trees : ℕ) (distance_between_trees : ℕ) (total_trees = 72) (distance_between_trees = 5) : 
  let trees_one_side := total_trees / 2
  let intervals := trees_one_side - 1
  let road_length := intervals * distance_between_trees
  road_length = 355 :=
by
  sorry

end road_length_trees_l630_630135


namespace find_ordered_pair_l630_630913

noncomputable def check_point_on_line (x y : ℝ) : Prop :=
  y = - (3 / 4) * x + 3

noncomputable def param_point (t s m : ℝ) : ℝ × ℝ :=
  (6 + t * m, s + t * 7)

theorem find_ordered_pair (s m : ℝ) :
  (∀ t : ℝ, check_point_on_line (fst (param_point t s m)) (snd (param_point t s m))) →
  s = -3 / 2 ∧ m = -7 / 3 :=
begin
  intro h,
  specialize h 0,
  specialize h 1,
  sorry -- the detailed proof would go here
end

end find_ordered_pair_l630_630913


namespace johnnys_hourly_wage_l630_630846

def totalEarnings : ℝ := 26
def totalHours : ℝ := 8
def hourlyWage : ℝ := 3.25

theorem johnnys_hourly_wage : totalEarnings / totalHours = hourlyWage :=
by
  sorry

end johnnys_hourly_wage_l630_630846


namespace renu_work_rate_l630_630888

theorem renu_work_rate (R : ℝ) :
  (∀ (renu_rate suma_rate combined_rate : ℝ),
    renu_rate = 1 / R ∧
    suma_rate = 1 / 6 ∧
    combined_rate = 1 / 3 ∧    
    combined_rate = renu_rate + suma_rate) → 
    R = 6 :=
by
  sorry

end renu_work_rate_l630_630888


namespace find_an_bn_Tn_l630_630750

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range n, a i

def seq_an_property (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) ^ 2 = 2 * (S a n) + n + 4

def seq_a_formula (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = n + 1

def seq_bn_formula (b : ℕ → ℕ) : Prop :=
  ∀ n, b n = 2 ^ n

def seq_cn (a b : ℕ → ℕ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = (-1) ^ n * (↑(a n) * ↑(b n))

def T_sum (c : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n, T n = -(3 * n + 2) / 9 * (-2) ^ (n + 1) - 2 / 9

theorem find_an_bn_Tn :
  ∃ a b c T,
    (seq_an_property a) ∧
    (a 2 = 1) ∧
    (seq_a_formula a) ∧
    (seq_bn_formula b) ∧
    (seq_cn a b c) ∧
    (T_sum c T) :=
by {
  sorry
}

end find_an_bn_Tn_l630_630750


namespace metro_school_count_l630_630832

theorem metro_school_count : 
  ∃ n : ℕ, 
    (∀ k ∈ {27, 46, 75}, k < 4 * n) ∧
    ((2 * n + 1 - 2) / 2 < 27) ∧
    (75 < 4 * n) ∧
    n = 19 :=
sorry

end metro_school_count_l630_630832


namespace sum_of_three_numbers_has_four_digits_l630_630802

theorem sum_of_three_numbers_has_four_digits
  (C D : ℕ)
  (hc : 1 ≤ C ∧ C ≤ 9)
  (hd : 1 ≤ D ∧ D ≤ 9) :
  let sum := 8765 + C * 100 + 43 + D * 10 + 2 in
  1000 ≤ sum ∧ sum < 10000 :=
by
  let sum := 8765 + C * 100 + 43 + D * 10 + 2
  split
  · sorry,
  · sorry

end sum_of_three_numbers_has_four_digits_l630_630802


namespace price_of_first_oil_is_54_l630_630583

/-- Let x be the price per litre of the first oil.
Given that 10 litres of the first oil are mixed with 5 litres of second oil priced at Rs. 66 per litre,
resulting in a 15-litre mixture costing Rs. 58 per litre, prove that x = 54. -/
theorem price_of_first_oil_is_54 :
  (∃ x : ℝ, x = 54) ↔
  (10 * x + 5 * 66 = 15 * 58) :=
by
  sorry

end price_of_first_oil_is_54_l630_630583


namespace smallest_c_value_l630_630352

theorem smallest_c_value (a b c : ℕ) (x y : ℤ) :
  (∃ x1 y1 : ℤ, a = x1^3 + y1^3) →
  (∃ x2 y2 : ℤ, b = x2^3 + y2^3) →
  c = a * b →
  (∀ x3 y3 : ℤ, c ≠ x3^3 + y3^3) →
  c = 4 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end smallest_c_value_l630_630352


namespace infinite_series_correct_l630_630668

noncomputable def infinite_series_sum : ℝ :=
  1 + 3 * (1 / 1001) + 5 * (1 / 1001)^2 + 7 * (1 / 1001)^3 + ∑' n, (2 * n + 1) * (1 / 1001)^(n + 1)

theorem infinite_series_correct : infinite_series_sum = 1.003002 :=
sorry

end infinite_series_correct_l630_630668


namespace club_members_addition_l630_630594

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l630_630594


namespace charles_whistles_l630_630444

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end charles_whistles_l630_630444


namespace donation_total_is_correct_l630_630880

-- Definitions and conditions
def Megan_inheritance : ℤ := 1000000
def Dan_inheritance : ℤ := 10000
def donation_percentage : ℚ := 0.1
def Megan_donation := Megan_inheritance * donation_percentage
def Dan_donation := Dan_inheritance * donation_percentage
def total_donation := Megan_donation + Dan_donation

-- Theorem statement
theorem donation_total_is_correct : total_donation = 101000 := by
  sorry

end donation_total_is_correct_l630_630880


namespace S_n_converges_l630_630655

noncomputable def S_n (n : ℕ) (x : ℝ) (f : ℝ → ℝ) : ℝ :=
  Real.exp (-n * x) * ∑ k in Finset.range (n + 1), f (k / n) * (n*x) ^ k / (Nat.factorial k)

theorem S_n_converges 
  {f : ℝ → ℝ} (hf_cont : ContinuousOn f Set.Ici 0) (hf_bounded : ∃ C, ∀ x ≥ 0, abs (f x) ≤ C)
  (a b : ℝ) (h_ab : 0 ≤ a ∧ a ≤ b) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, ∀ x ∈ Icc a b, abs (S_n n x f - f x) < ε :=
sorry

end S_n_converges_l630_630655


namespace decimal_to_binary_123_l630_630513

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630513


namespace unique_real_root_l630_630181

theorem unique_real_root : ∀ x : ℝ,
  (sqrt (4 * x^2 + 2 * x + 1) - sqrt (4 * x^2 + 14 * x + 5) = 6 * x + 2) ↔ x = -1 / 3 :=
sorry

end unique_real_root_l630_630181


namespace total_pencils_correct_l630_630495

def pencils_in_drawer : ℕ := 43
def pencils_on_desk_originally : ℕ := 19
def pencils_added_by_dan : ℕ := 16
def total_pencils : ℕ := pencils_in_drawer + pencils_on_desk_originally + pencils_added_by_dan

theorem total_pencils_correct : total_pencils = 78 := by
  sorry

end total_pencils_correct_l630_630495


namespace Charles_has_13_whistles_l630_630445

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end Charles_has_13_whistles_l630_630445


namespace quadratic_complete_square_l630_630142

theorem quadratic_complete_square (x p q : ℤ) 
  (h_eq : x^2 - 6 * x + 3 = 0) 
  (h_pq_form : x^2 - 6 * x + (p - x)^2 = q) 
  (h_int : ∀ t, t = p + q) : p + q = 3 := sorry

end quadratic_complete_square_l630_630142


namespace values_of_N_l630_630725

theorem values_of_N (N : ℕ) : (∃ k, k ∈ ({4, 6, 8, 12, 16, 24, 48} : set ℕ) ∧ k = N + 3) ↔ (N ∈ {1, 3, 5, 9, 13, 21, 45} : set ℕ) :=
by 
  sorry

#eval values_of_N 4 -- Example usage: should give true if N = 1

end values_of_N_l630_630725


namespace symmetric_point_line_l630_630312

theorem symmetric_point_line (m a : ℝ) (A B : ℝ × ℝ) (H_A : A = (1, -2)) (H_B : B = (m, 2))
  (H_symm : ∀ (x y : ℝ), ((x + a * y - 2 = 0) → (∃ (xm ym : ℝ), 
  xm = (fst A + fst B) / 2 ∧ ym = (snd A + snd B) / 2 ∧ xm + a * ym - 2 = 0))) :
  a = 2 :=
by
  sorry

end symmetric_point_line_l630_630312


namespace fold_length_square_l630_630997

theorem fold_length_square (A B C : ℝ) (a b : ℝ) (side_length : ℝ) (touch_point_distance : ℝ)
  (cond1 : A = sqrt((side_length)^2 + (side_length)^2 - 2 * side_length * side_length * cos (2 * π / 3)))
  (cond2 : touch_point_distance = 7)
  (cond3 : side_length = 10) :
  sqrt((3 * side_length ^ 2) - 4 * (side_length - touch_point_distance) * touch_point_distance) = sqrt(35557 / 484) :=
by
  sorry

end fold_length_square_l630_630997


namespace total_oranges_l630_630430

def monday_oranges : ℕ := 100
def tuesday_oranges : ℕ := 3 * monday_oranges
def wednesday_oranges : ℕ := 70

theorem total_oranges : monday_oranges + tuesday_oranges + wednesday_oranges = 470 := by
  sorry

end total_oranges_l630_630430


namespace base_2_representation_of_123_l630_630518

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630518


namespace base_2_representation_of_123_is_1111011_l630_630526

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630526


namespace rotated_line_eq_l630_630908

theorem rotated_line_eq :
  ∀ (x y : ℝ), -- Variables x and y in the real numbers
  (2 * x - y - 2 = 0) → -- Condition for the original line
  (∀ θ : ℝ, θ = Float.pi / 2) → -- Condition for rotation angle
  (x = 0 → y = -2) → -- Intersection point with the y-axis
  (x + 2 * y + 4 = 0) -- Resulting line equation after rotation
:=
begin
  sorry
end

end rotated_line_eq_l630_630908


namespace ballsInBoxes_theorem_l630_630341

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end ballsInBoxes_theorem_l630_630341


namespace volume_of_solid_revolution_l630_630669

noncomputable
def solid_revolution_volume : ℝ :=
  π * ∫ x in 0..1, (2*x - x^2)^2 - (-x + 2)^2

theorem volume_of_solid_revolution :
  solid_revolution_volume = (9/5) * π :=
by
  -- This is where the proof steps would go
  sorry

end volume_of_solid_revolution_l630_630669


namespace gcd_in_base3_l630_630316

def gcd_2134_1455_is_97 : ℕ :=
  gcd 2134 1455

def base3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) : List ℕ :=
      if n = 0 then [] else aux (n / 3) ++ [n % 3]
    aux n

theorem gcd_in_base3 :
  gcd_2134_1455_is_97 = 97 ∧ base3 97 = [1, 0, 1, 2, 1] :=
by
  sorry

end gcd_in_base3_l630_630316


namespace silver_dollars_l630_630053

variable (C : ℕ)
variable (H : ℕ)
variable (P : ℕ)

theorem silver_dollars (h1 : H = P + 5) (h2 : P = C + 16) (h3 : C + P + H = 205) : C = 56 :=
by
  sorry

end silver_dollars_l630_630053


namespace projection_of_b_onto_a_is_correct_l630_630738

open Real EuclideanGeometry

variables (a b : ℝ × ℝ × ℝ)

noncomputable def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2) + (v.3 * w.3)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scalar := (dot_product a b) / (magnitude a)^2 in
  (scalar * a.1, scalar * a.2, scalar * a.3)

theorem projection_of_b_onto_a_is_correct : projection (2, 0, 1) (3, 2, -5) = (2/5, 0, 1/5) :=
by 
  sorry

end projection_of_b_onto_a_is_correct_l630_630738


namespace green_ball_removal_l630_630588

variable (total_balls : ℕ)
variable (initial_green_balls : ℕ)
variable (initial_yellow_balls : ℕ)
variable (desired_green_percentage : ℚ)
variable (removals : ℕ)

theorem green_ball_removal :
  initial_green_balls = 420 → 
  total_balls = 600 → 
  desired_green_percentage = 3 / 5 →
  (420 - removals) / (600 - removals) = desired_green_percentage → 
  removals = 150 :=
sorry

end green_ball_removal_l630_630588


namespace Jiyeol_average_score_l630_630013

theorem Jiyeol_average_score (K M E : ℝ)
  (h1 : (K + M) / 2 = 26.5)
  (h2 : (M + E) / 2 = 34.5)
  (h3 : (K + E) / 2 = 29) :
  (K + M + E) / 3 = 30 := 
sorry

end Jiyeol_average_score_l630_630013


namespace tree_growth_l630_630128

theorem tree_growth (x : ℝ) : 4*x + 4*2*x + 4*2 + 4*3 = 32 → x = 1 :=
by
  intro h
  sorry

end tree_growth_l630_630128


namespace number_of_valid_N_count_valid_N_is_seven_l630_630728

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l630_630728


namespace expected_salary_correct_l630_630196

open_locale classical
noncomputable theory

def probability_distribution (n k : ℕ) : ℕ → ℚ
| 0 := 1 / (nat.choose n k : ℚ)
| 1 := (nat.choose k 1) * (nat.choose (n - k) (k - 1)) / (nat.choose n k : ℚ)
| 2 := (nat.choose k 2) * (nat.choose (n - k) (k - 2)) / (nat.choose n k : ℚ)
| 3 := (nat.choose k 3) * (nat.choose (n - k) (k - 3)) / (nat.choose n k : ℚ)
| 4 := 1 / (nat.choose n k : ℚ)
| _ := 0

def expected_salary (n k : ℕ) : ℚ :=
  let p := probability_distribution n k in
  3500 * p 4 + 2800 * p 3 + 2100 * (p 0 + p 1 + p 2)

theorem expected_salary_correct :
  expected_salary 8 4 = 2280 := 
by
  unfold expected_salary probability_distribution
  sorry

end expected_salary_correct_l630_630196


namespace desired_triangle_area_l630_630701

def area_of_triangle_in_figure : Real :=
  let area_of_rectangle := 6 * 8
  let area_of_triangle_1 := (1 / 2) * 6 * 4
  let area_of_triangle_2 := (1 / 2) * 6 * 2
  let area_of_triangle_3 := (1 / 2) * 4 * 4
  let total_area_of_triangles := area_of_triangle_1 + area_of_triangle_2 + area_of_triangle_3
  area_of_rectangle - total_area_of_triangles

theorem desired_triangle_area :
  area_of_triangle_in_figure = 22 := by
  sorry

end desired_triangle_area_l630_630701


namespace number_of_people_in_family_l630_630206

-- Define the conditions
def planned_spending : ℝ := 15
def savings_percentage : ℝ := 0.40
def cost_per_orange : ℝ := 1.5

-- Define the proof target: the number of people in the family
theorem number_of_people_in_family : 
  planned_spending * savings_percentage / cost_per_orange = 4 := 
by
  -- sorry to skip the proof; this is for statement only
  sorry

end number_of_people_in_family_l630_630206


namespace prove_radius_C1_l630_630503

variable (r : ℝ)

def radius_C1 := r  -- Define the radius of C1
def radius_C2 := 9  -- Given radius of C2
def radius_C3 := 4  -- Given radius of C3

-- Define center distances based on the conditions
def C1C2 := r + radius_C2 
def C1C3 := r + radius_C3
def C2C3 := radius_C2 + radius_C3

-- Define a theorem to prove radius of C1 equals 12
theorem prove_radius_C1 : r = 12 :=
by
  have H : 6 * sqrt r = 4 * sqrt r + sqrt (-4 * r^2 + 52 * r),
  {
    calc
      6 * sqrt r = _ : sorry -- This needs the full derivation but is skipped here
  },
  have H2 : (2 * sqrt r)^2 = (sqrt (-4 * r^2 + 52 * r))^2,
  {
    calc
      (2 * sqrt r)^2 = 4 * r        : by ring
      ... = -4 * r^2 + 52 * r        : sorry -- This needs the full derivation but is skipped here
  },
  have H3 : 4 * r = -4 * r^2 + 52 * r,
  {
    calc
      4 * r = _ : sorry -- Derivation step
  },
  have H4 : 4 * r * (1 - 12) = 0,
  {
    calc
      4 * r * (1 - 12) = 0 : by ring
  },
  show r = 12,
  {
    calc
      r = 12 : sorry -- We need proper full solution derivation here
  }

end prove_radius_C1_l630_630503


namespace sum_of_possible_p_l630_630108

theorem sum_of_possible_p (p q r t : ℚ) 
  (h1 : p < q) 
  (h2 : q < r) 
  (h3 : r < t) 
  (h4 : (p + q ≠ p + r) ∧ (p + q ≠ p + t) ∧ (p + q ≠ q + r) ∧ (p + q ≠ q + t) ∧ (p + q ≠ r + t) ∧
        (p + r ≠ p + t) ∧ (p + r ≠ q + r) ∧ (p + r ≠ q + t) ∧ (p + r ≠ r + t) ∧
        (p + t ≠ q + r) ∧ (p + t ≠ q + t) ∧ (p + t ≠ r + t) ∧
        (q + r ≠ q + t) ∧ (q + r ≠ r + t) ∧
        (q + t ≠ r + t) ) 
  (h5 : list.max [p + q, p + r, p + t, q + r, q + t, r + t] = 28)
  (h6 : list.erase (list.erase (list.erase (list.erase [p + q, p + r, p + t, q + r, q + t, r + t] 28) 25) 22) 19 ≠ p + q)
 : p = 7/2 ∨ p = 5 → ∑ x in {7/2, 5}, x = 17/2 :=
sorry

end sum_of_possible_p_l630_630108


namespace range_of_a_l630_630778

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - b * x

theorem range_of_a (a b x : ℝ) (h1 : ∀ x > 0, (1/x) - a * x - b ≠ 0) (h2 : ∀ x > 0, x = 1 → (1/x) - a * x - b = 0) : 
  (1 - a) = b ∧ a > -1 :=
by
  sorry

end range_of_a_l630_630778


namespace base_2_representation_of_123_l630_630517

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630517


namespace polynomial_sum_l630_630411

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l630_630411


namespace quadratic_roots_proof_l630_630910

noncomputable def quadratic_roots (a b c : ℝ) : set ℝ :=
  {x | a * x^2 + b * x + c = 0}

def polynomial_data : Type := 
  { 
    f : ℝ → ℝ // ∃ a b c : ℝ, (∀ x, f x = (2 / real.sqrt 3) * x^2 + b * x + c)
  }

def condition_1 (f : polynomial_data) (x₁ x₂ x₃ : ℝ) : Prop :=
  f.f x₁ = 0 ∧ f.f x₂ = 0 ∧ f.f x₃ = 0

def condition_2 (a b c x₁ x₂ x₃ : ℝ) : Prop :=
  let L := (0, x₁) in
  let K := (0, x₂) in
  let M := (0, x₃) in
  (L.1 - K.1)^2 + (L.2 - K.2)^2 = (L.1 - M.1)^2 + (L.2 - M.2)^2

def condition_3 (a b c x₁ x₂ x₃ : ℝ) : Prop :=
  let L := (0, x₁) in
  let K := (0, x₂) in
  let M := (0, x₃) in
  real.angle (K.1 - L.1, K.2 - L.2) (K.1 - M.1, K.2 - M.2) = real.pi / 3 -- 120 degrees

theorem quadratic_roots_proof :
  ∃ (p : polynomial_data) x₁ x₂ x₃,
    condition_1 p x₁ x₂ x₃ ∧
    condition_2 (2 / real.sqrt 3) p.1 p.2 x₁ x₂ x₃ ∧
    condition_3 (2 / real.sqrt 3) p.1 p.2 x₁ x₂ x₃ →
    quadratic_roots (2 / real.sqrt 3) p.1 p.2 = {0, 5, 1.5} :=
sorry

end quadratic_roots_proof_l630_630910


namespace find_eccentricity_l630_630299

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) := 
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_eq (p x y : ℝ) := 
  y^2 = 2 * p * x

noncomputable def cosine_angle (PF1 PF2 : ℝ) := 
  (PF2 / PF1) = (7 / 9)

theorem find_eccentricity (a b c p : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
  (P : ℝ × ℝ) (P_ellipse : ellipse_eq a b P.1 P.2) (P_parabola : parabola_eq p P.1 P.2)
  (cos_angle : cosine_angle (dist P (a, 0)) (dist P (c, 0))) :
  let e := c / a in 
  8 * e^2 - 7 * e + 1 = 0 := 
by sorry

end find_eccentricity_l630_630299


namespace sum_of_digits_inequality_l630_630036

def sum_of_digits (n : ℕ) : ℕ := -- Definition of the sum of digits function
  -- This should be defined, for demonstration we use a placeholder
  sorry

theorem sum_of_digits_inequality (n : ℕ) (h : n > 0) :
  sum_of_digits n ≤ 8 * sum_of_digits (8 * n) :=
sorry

end sum_of_digits_inequality_l630_630036


namespace number_of_valid_N_count_valid_N_is_seven_l630_630731

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l630_630731


namespace values_of_N_l630_630724

theorem values_of_N (N : ℕ) : (∃ k, k ∈ ({4, 6, 8, 12, 16, 24, 48} : set ℕ) ∧ k = N + 3) ↔ (N ∈ {1, 3, 5, 9, 13, 21, 45} : set ℕ) :=
by 
  sorry

#eval values_of_N 4 -- Example usage: should give true if N = 1

end values_of_N_l630_630724


namespace total_wheels_l630_630936

def num_wheels_in_garage : Nat :=
  let cars := 2 * 4
  let lawnmower := 4
  let bicycles := 3 * 2
  let tricycle := 3
  let unicycle := 1
  let skateboard := 4
  let wheelbarrow := 1
  let wagon := 4
  let dolly := 2
  let shopping_cart := 4
  let scooter := 2
  cars + lawnmower + bicycles + tricycle + unicycle + skateboard + wheelbarrow + wagon + dolly + shopping_cart + scooter

theorem total_wheels : num_wheels_in_garage = 39 := by
  sorry

end total_wheels_l630_630936


namespace equal_distances_l630_630380

noncomputable def Triangle :=
{A B C : Point}

noncomputable def midpoint (A B : Point) :=
{M : Point // dist M A = dist M B}

noncomputable def intersect_angle_45
  (A B C H K P : Point)
  (line1 : seg A B)
  (line2 : seg B H)
  (line3 : seg C K)
  (line4 : seg P H)
  (line5 : seg P K) :=
  angle line2 line4 = 45 ∧ 
  angle line3 line5 = 45

theorem equal_distances 
  (A B C M H K P : Point)
  (mid_M : midpoint B C = M)
  (intersect45 : intersect_angle_45 A B C H K P) :
  dist M P = dist M H ∧ dist M H = dist M K :=
by
  sorry

end equal_distances_l630_630380


namespace leading_coefficient_of_g_l630_630115

theorem leading_coefficient_of_g (g : ℕ → ℤ) (h : ∀ x : ℕ, g (x + 1) - g x = 4 * x + 6) : 
  leading_coeff g = 2 :=
sorry

end leading_coefficient_of_g_l630_630115


namespace reciprocal_of_neg_2023_l630_630485

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l630_630485


namespace correct_calculation_l630_630565

theorem correct_calculation (a b y m n : ℝ) : 
  (3 * a * b + 2 * a * b = 5 * a * b) ∧ 
  (5 * y^2 - 2 * y^2 ≠ 3) ∧ 
  (7 * a + a ≠ 7 * a^2) ∧ 
  (m^2 * n - 2 * m * n^2 ≠ - (m * n^2)) :=
by
  split
  · calc 3 * a * b + 2 * a * b = (3 + 2) * a * b : by ring
                        ... = 5 * a * b : by norm_num
  split
  · calc 5 * y^2 - 2 * y^2 = (5 - 2) * y^2 : by ring
                        ... = 3 * y^2 : by norm_num
  · rintro ⟨h⟩
    apply h
    calc (5 * y^2 - 2 * y^2) = (3 * y^2) : by ring
                        ... ≠ 3 : by norm_num
  split
  · calc 7 * a + a = (7 + 1) * a : by ring
                ... = 8 * a : by norm_num
  · rintro ⟨h⟩
    apply h
    calc 7 * a + a = 7 * a^2 : by ring
               ... = 8 * a : by norm_num
  split
  = sorry
  
lemma algebraic_incompatibility (m n : ℝ) :
  m^2 * n - 2 * m * n^2 ≠ -m * n^2 := 
by
  by_cases h1 : m = 0
  · simp [h1]
  by_cases h2 : n = 0
  · simp [h2]
  · sorry

end correct_calculation_l630_630565


namespace remainder_division_l630_630707

theorem remainder_division (x : ℂ) (β : ℂ) (hβ : β^7 = 1) :
  (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 0 ->
  (x^63 + x^49 + x^35 + x^14 + 1) % (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 5 :=
by
  intro h
  sorry

end remainder_division_l630_630707


namespace books_not_sold_l630_630203

theorem books_not_sold 
  (total_amount : ℝ)
  (price_per_book : ℝ)
  (fraction_sold : ℝ)
  (received_amount : ℝ)
  (h1 : total_amount / fraction_sold = received_amount / price_per_book) 
  (h2 : fraction_sold = 2 / 3)
  (h3 : price_per_book = 3.5)
  (h4 : received_amount = 280) : 
  total_amount * (1 - fraction_sold) = 40 :=
by
  sorry

end books_not_sold_l630_630203


namespace other_root_l630_630060

-- Define the condition that one root of the quadratic equation is -3
def is_root (a b c : ℤ) (x : ℚ) : Prop := a * x^2 + b * x + c = 0

-- Define the quadratic equation 7x^2 + mx - 6 = 0
def quadratic_eq (m : ℤ) (x : ℚ) : Prop := is_root 7 m (-6) x

-- Prove that the other root is 2/7 given that one root is -3
theorem other_root (m : ℤ) (h : quadratic_eq m (-3)) : quadratic_eq m (2 / 7) :=
by
  sorry

end other_root_l630_630060


namespace decimal_to_binary_123_l630_630514

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630514


namespace reciprocal_of_neg_2023_l630_630479

theorem reciprocal_of_neg_2023 : ∃ x : ℝ, (-2023) * x = 1 ∧ x = -1 / 2023 := 
by {
  existsi (-1 / 2023),
  split,
  { -- Prove that the product of -2023 and -1/2023 is 1
    unfold has_mul.mul,
    norm_num,
  },
  { -- Prove that x is indeed -1/2023
    refl,
  }
}

end reciprocal_of_neg_2023_l630_630479


namespace december_revenue_is_20_over_7_times_average_l630_630186

variable (D : ℚ) (N : ℚ) (J : ℚ)

-- Conditions: Revenue in November and January
def november_revenue_condition : Prop := N = (3/5) * D
def january_revenue_condition : Prop := J = (1/6) * N

-- Theorem
theorem december_revenue_is_20_over_7_times_average (h1 : november_revenue_condition D N) (h2 : january_revenue_condition J N) :
  D = (20 / 7) * ((N + J) / 2) :=
by 
  sorry

end december_revenue_is_20_over_7_times_average_l630_630186


namespace total_wage_calculation_l630_630619

def basic_pay_rate : ℝ := 20
def weekly_hours : ℝ := 40
def overtime_rate : ℝ := basic_pay_rate * 1.25
def total_hours_worked : ℝ := 48
def overtime_hours : ℝ := total_hours_worked - weekly_hours

theorem total_wage_calculation : 
  (weekly_hours * basic_pay_rate) + (overtime_hours * overtime_rate) = 1000 :=
by
  sorry

end total_wage_calculation_l630_630619


namespace area_of_ABC_l630_630877

noncomputable def area_triangle_ABC : Real :=
  let b : Real := sorry
  let c : Real := sorry
  let angle_BAC : Real := 45 * Real.pi / 180 -- angle in radians
  let OA := Real.sqrt 18
  let projection_area : Real := 12
  let b_c_product : Real := 24
  -- Using the conditions to define the area of the triangle
  let area := (1 / 2) * b * c * Real.sin angle_BAC
  area

theorem area_of_ABC
  (b c : Real)
  (OA : Real = Real.sqrt 18)
  (angle_BAC : Real = 45 * Real.pi / 180)
  (projection_area : Real = 12)
  (b_c_product : Real = 24) :
  area_triangle_ABC = 12 * Real.sqrt 2 := by
  sorry

end area_of_ABC_l630_630877


namespace range_of_a_l630_630753

variable {α : Type*} [LinearOrder α]

noncomputable def f (x : α) : α := sorry -- Definition of f(x)

theorem range_of_a (a : α) : 
  (∀ x : α, f(-x) = -f(x)) ∧ 
  (∀ x y : α, 0 ≤ x ∧ x ≤ y → f(x) ≤ f(y)) ∧ 
  f(2 - a^2) + f(a) > 0 → 
  (-1 : α) < a ∧ a < 2 :=
  sorry

end range_of_a_l630_630753


namespace range_of_t_l630_630476

theorem range_of_t (t : ℝ) (A : set ℝ) (h : A = {1, t}) : t ≠ 1 :=
by {
  rw h,
  intro h1,
  have : 1 ∈ A := by simp,
  have : t ∈ A := by simp,
  rw h1 at this,
  contradiction,
}

end range_of_t_l630_630476


namespace real_part_one_div_one_sub_eq_half_l630_630414

noncomputable def realPart {z : ℂ} (hz_nonreal : z ≠ conj z) (hz_norm : ∥z∥ = 1) : ℝ :=
  re (1 / (1 - z))

theorem real_part_one_div_one_sub_eq_half
  (z : ℂ) (hz_nonreal : z ≠ conj z) (hz_norm : ∥z∥ = 1) :
  realPart hz_nonreal hz_norm = 1 / 2 :=
sorry

end real_part_one_div_one_sub_eq_half_l630_630414


namespace number_of_correct_statements_l630_630331

-- Quadratic radicals problem setup
def A (x n : ℝ) := Real.sqrt (x^2 + x + n)
def B (x n : ℝ) := Real.sqrt (x^2 + x + n + 1)
def C (x n : ℕ) := A x n + B x n

-- Statements in the problem
def stmt1 (x : ℝ) := ¬ (∀ n, C x n = 0)
def stmt2 := (C 1 1 + C 1 3 + C 1 5 + ... + C 1 2021) - (C 1 2 + C 1 4 + ... + C 1 2022) = Real.sqrt 3 - 45
def stmt3 (x : ℝ) (n : ℕ) := (∑ i in Finset.range n, 1 / C x i) = Real.sqrt (x^2 + x + n + 1) - Real.sqrt (x^2 + x + 1)
def stmt4 (x : ℝ) := (C x 6)^2 + 1 / (C x 6)^2 = 4 * x^2 + 29 → x = 3 / 4

-- Main theorem
theorem number_of_correct_statements : 
    let correct_count := 
      (if stmt1 then 1 else 0) + 
      (if stmt2 then 1 else 0) + 
      (if stmt3 then 1 else 0) + 
      (if stmt4 then 1 else 0) 
    in correct_count = 3 :=
  sorry

end number_of_correct_statements_l630_630331


namespace mala_usha_speed_ratio_l630_630048

noncomputable def drinking_speed_ratio (M U : ℝ) (tM tU : ℝ) (fracU : ℝ) (total_bottle : ℝ) : ℝ :=
  let U_speed := fracU * total_bottle / tU
  let M_speed := (total_bottle - fracU * total_bottle) / tM
  M_speed / U_speed

theorem mala_usha_speed_ratio :
  drinking_speed_ratio (3/50) (1/50) 10 20 (4/10) 1 = 3 :=
by
  sorry

end mala_usha_speed_ratio_l630_630048


namespace probability_of_odd_product_is_zero_l630_630673

-- Define the spinners
def spinnerC : List ℕ := [1, 3, 5, 7]
def spinnerD : List ℕ := [2, 4, 6]

-- Define the condition that the odds and evens have a specific product property
axiom odd_times_even_is_even {a b : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 0) : (a * b) % 2 = 0

-- Define the probability of getting an odd product
noncomputable def probability_odd_product : ℕ :=
  if ∃ a ∈ spinnerC, ∃ b ∈ spinnerD, (a * b) % 2 = 1 then 1 else 0

-- Main theorem
theorem probability_of_odd_product_is_zero : probability_odd_product = 0 := by
  sorry

end probability_of_odd_product_is_zero_l630_630673


namespace ten_seq_l630_630026

-- Definitions of conditions based on problem statement.
variables {a : ℕ → ℕ} (h_increasing : ∀ n, a n < a (n+1))
variables {b : ℕ → ℕ} (h_b_def : ∀ n, b n = (Nat.divisors (a n)).filter (· < a n)).maximum!)
variables (h_decreasing : ∀ n, n < 9 → b n > b (n+1))

-- Main theorem to be proven.
theorem ten_seq (h_increasing : ∀ n < 9, a n < a (n+1))
  (h_b_def : ∀ n < 10, ∃ m, m < a n ∧ m ∣ a n ∧ b n = m)
  (h_decreasing : ∀ n < 9, b n > b (n+1)) :
  a 9 > 500 :=
sorry

end ten_seq_l630_630026


namespace isosceles_trapezoid_area_l630_630267

theorem isosceles_trapezoid_area (h : ℝ) :
  let angle := real.pi / 3 -- 60 degrees in radians
  let half_angle := angle / 2
  let AC := 2 * h
  let AK := h * real.sqrt 3
  let sum_parallel_sides := 2 * AK
  let area := (sum_parallel_sides * h) / 2
  area = h^2 * real.sqrt 3 :=
by
  sorry

end isosceles_trapezoid_area_l630_630267


namespace domain_of_function_l630_630907

noncomputable def is_domain (x : Real) : Prop :=
  ∃ (k : ℤ), (k * π + π / 3 ≤ x) ∧ (x < k * π + π / 2)

theorem domain_of_function :
  ∀ x : Real, (sqrt (sqrt 3 * tan x - 3) : Real) = sqrt (sqrt 3 * tan x - 3) ↔ is_domain x :=
sorry

end domain_of_function_l630_630907


namespace correct_statements_count_l630_630297

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (2 * a 0 + (n - 1) * d)) / 2

variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (a1_pos : a 0 > 0)
          (ratio_cond : -1 < (a 5 / a 4) ∧ (a 5 / a 4) < 0)
          (arith_seq : arithmetic_sequence a)
          (sum_terms : sum_of_first_n_terms a S)

theorem correct_statements_count : (arithmetic_sequence a ∧
                                   sum_of_first_n_terms a S ∧
                                   a1_pos ∧
                                   ratio_cond) →
                                   3 = (3:ℕ) :=
by
  -- here the proof steps would go
  sorry

end correct_statements_count_l630_630297


namespace does_not_divide_24_l630_630866

theorem does_not_divide_24 (n : ℕ) (h : 0 < n) 
  (h2 : (3 : ℚ)⁻¹ + (4 : ℚ)⁻¹ + (8 : ℚ)⁻¹ + (n : ℚ)⁻¹ ∈ ℤ) : ¬ (3 ∣ n) := 
by 
  sorry

end does_not_divide_24_l630_630866


namespace musicians_practice_together_l630_630229

open Nat

theorem musicians_practice_together :
  Nat.leastCommonMultiple [5, 6, 8, 9] = 360 :=
  by
    sorry

end musicians_practice_together_l630_630229


namespace average_first_300_terms_l630_630241

def seq (n : ℕ) : ℤ :=
  if n % 2 = 1 then (-1)^(n+1) * (2*n - 1) else (-1)^(n+1) * 2*n

theorem average_first_300_terms : 
  (∑ i in Finset.range 300, seq (i + 1)) / 300 = -76 := by
  sorry

end average_first_300_terms_l630_630241


namespace simplify_fraction_l630_630957

variables {m x : ℝ}
theorem simplify_fraction (m x : ℝ) (h : x ≠ 0) (hm : m ≠ 0) :
  (5 * m^2 * x^2) / (10 * m * x^2) = m / 2 :=
by 
  have hx : x^2 ≠ 0 := pow_ne_zero 2 h,
  have h1 : 5 * m^2 * x^2 = (5 * m * x^2) * m,
  { ring, },
  have h2 : 10 * m * x^2 = (5 * m * x^2) * 2,
  { ring, },
  rw [h1, h2],
  field_simp [hm, hx],
  ring,
  sorry

end simplify_fraction_l630_630957


namespace system_solution_l630_630809

theorem system_solution :
  ∃ x y : ℝ, (3 * x + y = 11 ∧ x - y = 1) ∧ (x = 3 ∧ y = 2) := 
by
  sorry

end system_solution_l630_630809


namespace grazing_months_b_l630_630969

theorem grazing_months_b (a_oxen a_months b_oxen c_oxen c_months total_rent c_share : ℕ) (x : ℕ) 
  (h_a : a_oxen = 10) (h_am : a_months = 7) (h_b : b_oxen = 12) 
  (h_c : c_oxen = 15) (h_cm : c_months = 3) (h_tr : total_rent = 105) 
  (h_cs : c_share = 27) : 
  45 * 105 = 27 * (70 + 12 * x + 45) → x = 5 :=
by
  sorry

end grazing_months_b_l630_630969


namespace compute_z_ninth_power_l630_630043

def z : ℂ := (-sqrt(3) + complex.i) / 2

theorem compute_z_ninth_power (h1 : z^2 = (1 - complex.i * sqrt(3)) / 2)
                             (h2 : z^3 = complex.i) :
                             z^9 = -complex.i :=
by
  sorry

end compute_z_ninth_power_l630_630043


namespace number_of_m_tuples_l630_630949

theorem number_of_m_tuples (n m : ℕ) (hm : 1 ≤ m) (hn : m ≤ n) :
  {x : (fin m → ℕ) // ∀ i, 0 < x i ∧ (finset.univ.sum x = n)}.card = (nat.choose (n-1) (m-1)) :=
sorry

end number_of_m_tuples_l630_630949


namespace base_2_representation_of_123_l630_630540

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630540


namespace problem_statement_l630_630862

variable (a b c : ℝ)

def g (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

theorem problem_statement (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 := by
  have h_even : ∀ x : ℝ, g a b c x = g a b c (-x) := by
    intro x
    simp [g]
  have h_neg10 : g a b c (-10) = g a b c 10 := h_even 10
  rw [h_neg10, h]
  norm_num
  sorry

end problem_statement_l630_630862


namespace game_probability_l630_630422

-- Define a function to represent the game
def game : ℕ → nat → nat → nat → nat → nat → bool :=
λ n initial_1 initial_2 initial_3 current_1 current_2 current_3,
if n = 2020 then (current_1 = 2 ∧ current_2 = 2 ∧ current_3 = 2) else
sorry -- game logic goes here

theorem game_probability :
  let initial_money := (2, 2, 2) in
  \(2020 : nat) (1, 2, 3, 4, 5, 6 : ℕ) *- (0.25) :=
begin yoth.initial_money (by)
  intro n,
  assumption
end eq_fight (by sorry)

end game_probability_l630_630422


namespace nutty_professor_mixture_weight_l630_630090

/-- The Nutty Professor's problem translated to Lean 4 -/
theorem nutty_professor_mixture_weight :
  let cashews_weight := 20
  let cashews_cost_per_pound := 6.75
  let brazil_nuts_cost_per_pound := 5.00
  let mixture_cost_per_pound := 5.70
  ∃ (brazil_nuts_weight : ℝ), cashews_weight * cashews_cost_per_pound + brazil_nuts_weight * brazil_nuts_cost_per_pound =
                             (cashews_weight + brazil_nuts_weight) * mixture_cost_per_pound ∧
                             (cashews_weight + brazil_nuts_weight = 50) := 
sorry

end nutty_professor_mixture_weight_l630_630090


namespace trigonometric_identity_l630_630580

theorem trigonometric_identity (k : ℤ) (x : ℝ) : 
  (2 * (Math.cos (4 * x) - Math.sin x * Math.cos (3 * x)) = 
    Math.sin (4 * x) + Math.sin (2 * x)) → 
  x = (Real.pi / 16) * (4 * k + 1) :=
  sorry

end trigonometric_identity_l630_630580


namespace more_cost_effective_to_buy_from_second_seller_l630_630059

theorem more_cost_effective_to_buy_from_second_seller
  (cost_per_kg_first : ℝ)
  (portion_edible_first : ℝ)
  (cost_per_kg_second : ℝ)
  (portion_edible_second : ℝ)
  (effective_cost_first : ℝ := cost_per_kg_first / portion_edible_first)
  (effective_cost_second : ℝ := cost_per_kg_second / portion_edible_second)
  (h_conditions : cost_per_kg_first = 150 ∧ portion_edible_first = (2/3) ∧ cost_per_kg_second = 100 ∧ portion_edible_second = (1/2))
  : effective_cost_second < effective_cost_first := by
begin
  -- skip the proof
  sorry
end

end more_cost_effective_to_buy_from_second_seller_l630_630059


namespace percentage_of_360_l630_630560

theorem percentage_of_360 (percentage : ℝ) : 
  (percentage / 100) * 360 = 93.6 → percentage = 26 := 
by
  intro h
  -- proof missing
  sorry

end percentage_of_360_l630_630560


namespace min_abs_diff_3xy_8x_3y_195_l630_630412

theorem min_abs_diff_3xy_8x_3y_195 :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 3 * x * y - 8 * x + 3 * y = 195 ∧
             ∀ z w : ℕ, 0 < z ∧ 0 < w ∧ 3 * z * w - 8 * z + 3 * w = 195 →
             |x - y| ≤ |z - w| ∧ |x - y| = 7 := 
sorry

end min_abs_diff_3xy_8x_3y_195_l630_630412


namespace sum_of_non_solutions_l630_630418

theorem sum_of_non_solutions (A B C x: ℝ) 
  (h1 : A = 2) 
  (h2 : B = C / 2) 
  (h3 : C = 28) 
  (eq_inf_solutions : ∀ x, (x ≠ -C ∧ x ≠ -14) → 
  (x + B) * (A * x + 56) = 2 * ((x + C) * (x + 14))) : 
  (-14 + -28) = -42 :=
by
  sorry

end sum_of_non_solutions_l630_630418


namespace max_marks_obtainable_l630_630652

theorem max_marks_obtainable 
  (math_pass_percentage : ℝ := 36 / 100)
  (phys_pass_percentage : ℝ := 40 / 100)
  (chem_pass_percentage : ℝ := 45 / 100)
  (math_marks : ℕ := 130)
  (math_fail_margin : ℕ := 14)
  (phys_marks : ℕ := 120)
  (phys_fail_margin : ℕ := 20)
  (chem_marks : ℕ := 160)
  (chem_fail_margin : ℕ := 10) : 
  ∃ max_total_marks : ℤ, max_total_marks = 1127 := 
by 
  sorry  -- Proof not required

end max_marks_obtainable_l630_630652


namespace symmetric_patterns_7x7_l630_630366

noncomputable def total_symmetric_patterns : ℕ := 1022

-- Problem: Prove that the total number of symmetric patterns in a 7x7 grid is 1022
theorem symmetric_patterns_7x7 :
  ∃ n : ℕ, n = 1022 ∧ 
  (∀ (grid : matrix (fin 7) (fin 7) bool), 
     (∃ b w : bool, b ≠ w ∧ ∃ r : fin 7, b = grid r r) ∧
     (∀ i j : fin 7, 
       grid i j = grid (fin.succ i) j ∧ 
       grid i j = grid i (fin.succ j) ∧ 
       grid i j = grid (6-i) (6-j)) 
     → n = total_symmetric_patterns) := 
begin
  use 1022,
  split,
  { reflexivity, },
  { intros grid h,
    have := sorry, -- Proof of grid symmetry and counting of patterns
    exact this, }
end

end symmetric_patterns_7x7_l630_630366


namespace correct_average_weight_l630_630367

theorem correct_average_weight 
  (n : ℕ) 
  (w_avg : ℝ) 
  (W_init : ℝ)
  (d1 : ℝ)
  (d2 : ℝ)
  (d3 : ℝ)
  (W_adj : ℝ)
  (w_corr : ℝ)
  (h1 : n = 30)
  (h2 : w_avg = 58.4)
  (h3 : W_init = n * w_avg)
  (h4 : d1 = 62 - 56)
  (h5 : d2 = 59 - 65)
  (h6 : d3 = 54 - 50)
  (h7 : W_adj = W_init + d1 + d2 + d3)
  (h8 : w_corr = W_adj / n) :
  w_corr = 58.5 := 
sorry

end correct_average_weight_l630_630367


namespace base_2_representation_of_123_l630_630546

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630546


namespace cost_of_parakeet_l630_630138

theorem cost_of_parakeet
  (P Py K : ℕ) -- defining the costs of parakeet, puppy, and kitten
  (h1 : Py = 3 * P) -- puppy is three times the cost of parakeet
  (h2 : P = K / 2) -- parakeet is half the cost of kitten
  (h3 : 2 * Py + 2 * K + 3 * P = 130) -- total cost equation
  : P = 10 := 
sorry

end cost_of_parakeet_l630_630138


namespace line_bisects_circle_slope_line_tangent_circle_slope_line_intersects_chord_slope_l630_630991

-- Given conditions
def P : ℝ × ℝ := (-3, -4)
def C : ℝ × ℝ := (1, -2)
def r : ℝ := 2  -- radius of the circle
def circle : set (ℝ × ℝ) := { p | (p.1 - C.1)^2 + (p.2 + C.2 + 2)^2 = r^2 }

-- Statement (1)
theorem line_bisects_circle_slope (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (line : ℝ → ℝ) : 
  (line bisects circle) → (slope line = (1/2)) :=
sorry

-- Statement (2)
theorem line_tangent_circle_slope (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (line : ℝ → ℝ) : 
  (line tangent to circle) → (slope line = 0 ∨ slope line = 4/3) :=
sorry

-- Statement (3)
theorem line_intersects_chord_slope (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (line : ℝ → ℝ) (chord_length : ℝ) : 
  (line intersects circle) ∧ (length chord = 2) → 
  (slope line = (8+sqrt(51))/13 ∨ slope line = (8-sqrt(51))/13) :=
sorry

end line_bisects_circle_slope_line_tangent_circle_slope_line_intersects_chord_slope_l630_630991


namespace six_points_concyclic_l630_630658

-- Define the points on a plane
variables {P Q R S M N : Type}

-- Assume the cyclic conditions given
variable (h1 : cyclic {M, N, P, Q})
variable (h2 : cyclic {P, Q, R, S})
variable (h3 : cyclic {R, S, M, N})

-- State the proof problem
theorem six_points_concyclic (h1 : cyclic {M, N, P, Q}) (h2 : cyclic {P, Q, R, S}) (h3 : cyclic {R, S, M, N}) :
  cyclic {P, Q, R, S, M, N} :=
sorry

end six_points_concyclic_l630_630658


namespace point_in_third_quadrant_l630_630807

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end point_in_third_quadrant_l630_630807


namespace iodine_atomic_weight_l630_630706

noncomputable def atomic_weight_of_iodine : ℝ :=
  127.01

theorem iodine_atomic_weight
  (mw_AlI3 : ℝ := 408)
  (aw_Al : ℝ := 26.98)
  (formula_mw_AlI3 : mw_AlI3 = aw_Al + 3 * atomic_weight_of_iodine) :
  atomic_weight_of_iodine = 127.01 :=
by sorry

end iodine_atomic_weight_l630_630706


namespace max_integer_value_l630_630355

theorem max_integer_value (x : ℝ) : 
  ∃ (n : ℤ), n = 15 ∧ ∀ x : ℝ, 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ n :=
by
  sorry

end max_integer_value_l630_630355


namespace find_y_l630_630804

theorem find_y (x y: ℤ) (h1: x^2 - 3 * x + 2 = y + 6) (h2: x = -4) : y = 24 :=
by
  sorry

end find_y_l630_630804


namespace seventy_five_inverse_mod_seventy_six_l630_630263

-- Lean 4 statement for the problem.
theorem seventy_five_inverse_mod_seventy_six : (75 : ℤ) * 75 % 76 = 1 :=
by
  sorry

end seventy_five_inverse_mod_seventy_six_l630_630263


namespace ballsInBoxes_theorem_l630_630340

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end ballsInBoxes_theorem_l630_630340


namespace f_monotonically_decreasing_on_interval_l630_630461

-- Define the function
def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * x^2 - 6 * x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := x^2 - x - 6

-- The interval we are focusing on
def interval := Set.Ioo (-2 : ℝ) (2 : ℝ)

-- The theorem to be proven: f is monotonically decreasing on the interval (-2, 2)
theorem f_monotonically_decreasing_on_interval : 
  ∀ x y ∈ interval, x < y → f'(x) < 0 ∧ f'(y) < 0 → f y ≤ f x := by
  sorry

end f_monotonically_decreasing_on_interval_l630_630461


namespace maximum_pentagon_distance_configuration_l630_630959

noncomputable def max_sum_of_distances := 
  let points_on_unit_circle (p: ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1
  let d (p1 p2: ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let sum_of_distances (points: list (ℝ × ℝ)) := 
    list.sum (list.map (λ (pair: (ℝ × ℝ) × (ℝ × ℝ)), d pair.1 pair.2) 
    (list.pairwise_with points))
  ∀ (p1 p2 p3 p4 p5 : ℝ × ℝ),
  points_on_unit_circle p1 ∧ points_on_unit_circle p2 ∧ points_on_unit_circle p3 ∧ 
  points_on_unit_circle p4 ∧ points_on_unit_circle p5 →
  sum_of_distances [p1, p2, p3, p4, p5] ≤ 10 * (real.sin (real.pi / 5) + real.sin (2 * real.pi / 5))

theorem maximum_pentagon_distance_configuration:
  ∀ (p1 p2 p3 p4 p5 : ℝ × ℝ),
  points_on_unit_circle p1 ∧ points_on_unit_circle p2 ∧ points_on_unit_circle p3 ∧ 
  points_on_unit_circle p4 ∧ points_on_unit_circle p5 →
  sum_of_distances [p1, p2, p3, p4, p5] = 10 * (real.sin (real.pi / 5) + real.sin (2 * real.pi / 5))
sorry

end maximum_pentagon_distance_configuration_l630_630959


namespace trigonometric_identity_l630_630757

open Real

theorem trigonometric_identity
  (x : ℝ)
  (h1 : sin x * cos x = 1 / 8)
  (h2 : π / 4 < x)
  (h3 : x < π / 2) :
  cos x - sin x = - (sqrt 3 / 2) :=
sorry

end trigonometric_identity_l630_630757


namespace geometric_sequence_first_term_l630_630315

noncomputable theory
open_locale big_operators

theorem geometric_sequence_first_term :
  ∃ b : ℕ → ℝ, 
    (∀ n, b (n + 1) = 2 * b n) ∧
    (∀ n, (Real.log2 (b n)) * (Real.log2 (b (n + 1)) - 2) = n^2 + 3 * n) ∧
    b 1 = 4 :=
by {
  sorry
}

end geometric_sequence_first_term_l630_630315


namespace io_perpendicular_bi_i_is_circumcenter_dek_l630_630821

/-- In a scalene triangle ABC, if the lengths of sides BC, CA, and AB form an arithmetic sequence, and 
I and O are the incenter and circumcenter of ABC, respectively, then IO is perpendicular to BI. -/
theorem io_perpendicular_bi
  (A B C I O : Point)
  (is_scalene : scalene_triangle ABC)
  (arithmetic_sequence : sides_form_arithmetic_sequence BC CA AB)
  (incenter : incenter_of_triangle I ABC)
  (circumcenter : circumcenter_of_triangle O ABC) 
  : perp IO BI := 
sorry

/-- In a scalene triangle ABC, if the lengths of sides BC, CA, and AB form an arithmetic sequence, 
if I and O are the incenter and circumcenter of ABC, respectively, 
if BI intersects AC at K, and D and E are the midpoints of sides BC and AB, respectively, 
then I is the circumcenter of DEK. -/
theorem i_is_circumcenter_dek
  (A B C I O K D E : Point)
  (is_scalene : scalene_triangle ABC)
  (arithmetic_sequence : sides_form_arithmetic_sequence BC CA AB)
  (incenter : incenter_of_triangle I ABC)
  (circumcenter : circumcenter_of_triangle O ABC)
  (intersect_K : intersects BI AC K)
  (midpoint_D : midpoint D BC)
  (midpoint_E : midpoint E AB)
  : circumcenter_of_triangle I DEK :=
sorry

end io_perpendicular_bi_i_is_circumcenter_dek_l630_630821


namespace minimum_possible_value_box_l630_630354

theorem minimum_possible_value_box (a b : ℤ) (h_distinct : a ≠ b ∧ a ≠ 34 ∧ b ≠ 34) (h_ab : a * b = 15) : ∃ (box : ℤ), box = 34 ∧ (∀ (c d : ℤ), c * d = 15 → c ≠ d ∧ c ≠ 34 ∧ d ≠ 34 → (c^2 + d^2) ≥ box) := 
by 
  use 34,
  split,
  sorry, -- Proof that for some a, b, a^2 + b^2 = 34
  intros c d hcd hdist,
  sorry -- Proof that for all c, d such that cd = 15 and c, d are distinct integers, c^2 + d^2 ≥ 34

end minimum_possible_value_box_l630_630354


namespace mowing_lawn_together_l630_630197

-- Definitions based on given conditions
def A_mowing_time := 130
def B_mowing_time := 100
def C_mowing_time := 150

-- Definition of their mowing rates
def A_mowing_rate := 1 / A_mowing_time
def B_mowing_rate := 1 / B_mowing_time
def C_mowing_rate := 1 / C_mowing_time

-- Combined mowing rate
def combined_mowing_rate := A_mowing_rate + B_mowing_rate + C_mowing_rate

-- Time taken for all three to mow the lawn together
def total_time := 1 / combined_mowing_rate

-- Stating the theorem
theorem mowing_lawn_together : 
  total_time ≈ 41.05 := by sorry

end mowing_lawn_together_l630_630197


namespace xiaoMing_xiaoHong_diff_university_l630_630735

-- Definitions based on problem conditions
inductive Student
| XiaoMing
| XiaoHong
| StudentC
| StudentD
deriving DecidableEq

inductive University
| A
| B
deriving DecidableEq

-- Definition for the problem
def num_ways_diff_university : Nat :=
  4 -- The correct answer based on the solution steps

-- Problem statement
theorem xiaoMing_xiaoHong_diff_university :
  let students := [Student.XiaoMing, Student.XiaoHong, Student.StudentC, Student.StudentD]
  let universities := [University.A, University.B]
  (∃ (assign : Student → University),
    assign Student.XiaoMing ≠ assign Student.XiaoHong ∧
    (assign Student.StudentC ≠ assign Student.StudentD ∨
     assign Student.XiaoMing ≠ assign Student.StudentD ∨
     assign Student.XiaoHong ≠ assign Student.StudentC ∨
     assign Student.XiaoMing ≠ assign Student.StudentC)) →
  num_ways_diff_university = 4 :=
by
  sorry

end xiaoMing_xiaoHong_diff_university_l630_630735


namespace intersection_of_sets_find_values_of_a_and_b_l630_630420

section

variable (x : ℝ)

def setA : Set ℝ := { x | x^2 - 3*x - 4 < 0 }
def setB : Set ℝ := { x | -3 < x ∧ x < 1 }

theorem intersection_of_sets : setA ∩ setB = { x | -1 < x ∧ x < 1 } := by
  simp [setA, setB]
  sorry

variable (a b : ℝ)

def inequality_set : Set ℝ := { x | 2*x^2 + a*x + b < 0 }

theorem find_values_of_a_and_b 
  (h : inequality_set = setB) :
  a = 3 ∧ b = 4 := by
  sorry

end

end intersection_of_sets_find_values_of_a_and_b_l630_630420


namespace find_sin_theta_l630_630857

variables {a b c : ℝ^3}
variables {θ : ℝ}

noncomputable def sin_theta :=
  let ab_cross_c := (a × b) × c
  let magnitude_b := ∥b∥
  let magnitude_c := ∥c∥
  let cos_theta := -1/5
  sqrt(1 - cos_theta^2)
   
theorem find_sin_theta
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : ¬(a ∥ b))
  (h5 : ¬(b ∥ c))
  (h6 : ¬(a ∥ c))
  (h7 : (a × b) × c = (1/5) * ∥b∥ * ∥c∥ * a) :
  sin_theta = 2 * sqrt 6 / 5 :=
begin
  sorry
end

end find_sin_theta_l630_630857


namespace find_b_c_l630_630402

theorem find_b_c (a b c : ℤ) (h : ∃ (k : ℤ), 2ax^3 + bx^2 + cx - 3 = (x^2 - 2x - 1) * (2ax + k)) :
  a = 1 → b = -1 ∧ c = -8 :=
by {
  intro ha,
  have hb : b = 3 - 4 * a,
  have hc : c = -6 - 2 * a,
  have hk : k = 3,
  subst ha,
  simp at *,
  sorry
}

end find_b_c_l630_630402


namespace reciprocal_of_neg_2023_l630_630480

theorem reciprocal_of_neg_2023 : ∃ x : ℝ, (-2023) * x = 1 ∧ x = -1 / 2023 := 
by {
  existsi (-1 / 2023),
  split,
  { -- Prove that the product of -2023 and -1/2023 is 1
    unfold has_mul.mul,
    norm_num,
  },
  { -- Prove that x is indeed -1/2023
    refl,
  }
}

end reciprocal_of_neg_2023_l630_630480


namespace units_digit_m_squared_plus_3_to_m_l630_630865

theorem units_digit_m_squared_plus_3_to_m (m : ℕ) (h : m = 2021^2 + 3^2021) : (m^2 + 3^m) % 10 = 7 :=
by
  sorry

end units_digit_m_squared_plus_3_to_m_l630_630865


namespace probability_exactly_4_replaced_expected_number_replaced_l630_630127

-- Definitions
def num_lamps := 9

def lamp_burns_out_independently (n : ℕ) : Prop := 
  ∀ (i : ℕ), i < n → burns_out (lamp i) is independent from all other lamps

def all_burned_out_lamps_replaced_if_adjacent : Prop := 
  ∀ (i j : ℕ), (burns_out (lamp i) ∧ burns_out (lamp j) ∧ adjacent i j) → all_burned_out_lamps replaced

def no_action_if_one_burns_out : Prop := 
  ∀ (i : ℕ), burns_out (lamp i) ∧ no_action_taken

-- Theorem Statements
theorem probability_exactly_4_replaced :
  lamp_burns_out_independently num_lamps ∧ 
  all_burned_out_lamps_replaced_if_adjacent ∧ 
  no_action_if_one_burns_out →
  probability_of_replacement 4 = 0.294 :=
  sorry

theorem expected_number_replaced :
  lamp_burns_out_independently num_lamps ∧ 
  all_burned_out_lamps_replaced_if_adjacent ∧ 
  no_action_if_one_burns_out →
  expected_number_of_replacement = 3.32 :=
  sorry

end probability_exactly_4_replaced_expected_number_replaced_l630_630127


namespace cans_display_rows_eq_ten_l630_630208

theorem cans_display_rows_eq_ten
  (n : ℕ)
  (sum_arith_series_eq : ∑ i in range n, (2 + 3 * i) = 169) :
  3 * n^2 + n - 338 = 0 := 
by 
  sorry

end cans_display_rows_eq_ten_l630_630208


namespace base_2_representation_of_123_l630_630548

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630548


namespace concurrency_of_reflections_l630_630132

noncomputable theory
open_locale classical

theorem concurrency_of_reflections 
  (A B C O : Point)
  (circ : Circle)
  (hA : A ∈ circ)
  (hB : B ∈ circ)
  (hC : C ∈ circ)
  (diam : Line)
  (hdiam : O ∈ diam)
  (A1 B1 C1 : Point)
  (hA1 : reflection_over_diameter A diam = A1)
  (hB1 : reflection_over_diameter B diam = B1)
  (hC1 : reflection_over_diameter C diam = C1)
  (hparallel1 : ∃ line1, line1 ∥ BC ∧ A1 ∈ line1)
  (hparallel2 : ∃ line2, line2 ∥ AC ∧ B1 ∈ line2)
  (hparallel3 : ∃ line3, line3 ∥ AB ∧ C1 ∈ line3):
  concurrent [line1, line2, line3] := 
sorry

end concurrency_of_reflections_l630_630132


namespace poly_eq_l630_630082

-- Definition of the polynomials f(x) and g(x)
def f (x : ℝ) := x^4 + 4*x^3 + 8*x
def g (x : ℝ) := 10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5

-- Define p(x) as a function that satisfies the given condition
def p (x : ℝ) := 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5

-- Prove that the function p(x) satisfies the equation
theorem poly_eq : ∀ x : ℝ, p x + f x = g x :=
by
  intro x
  -- Add a marker to indicate that this is where the proof would go
  sorry

end poly_eq_l630_630082


namespace line_passing_through_point_and_parallel_to_given_line_l630_630268

theorem line_passing_through_point_and_parallel_to_given_line :
  ∃ c : ℝ, ∀ x y : ℝ, (2 * x - y + c = 0) ∧ (P = (0,2)) ∧ (2 * 0 - 2 + c = 0) :=
begin
  -- Assume the point P is given
  let P := (0, 2),
  -- Assume the line equation is given
  let line_eq := λ x y c, 2 * x - y + c = 0,
  -- Prove there exists a c such that the line passes through P and is parallel to 2x - y = 0
  use 2,  -- we guess and use 2 for c as the derived solution
  intro x,
  intro y,
  split,
  { exact line_eq x y 2, },
  split,
  { exact P, },
  { sorry, }  -- You would complete the proof here, just a placeholder.
end

end line_passing_through_point_and_parallel_to_given_line_l630_630268


namespace complex_arithmetic_1_complex_arithmetic_2_l630_630974

-- Proof Problem 1
theorem complex_arithmetic_1 : 
  (1 : ℂ) * (-2 - 4 * I) - (7 - 5 * I) + (1 + 7 * I) = -8 + 8 * I := 
sorry

-- Proof Problem 2
theorem complex_arithmetic_2 : 
  (1 + I) * (2 + I) + (5 + I) / (1 - I) + (1 - I) ^ 2 = 3 + 4 * I := 
sorry

end complex_arithmetic_1_complex_arithmetic_2_l630_630974


namespace atomic_weight_of_nitrogen_l630_630469

-- Definitions and conditions
def molecular_weight : ℝ := 98
def hydrogen_atoms : ℕ := 4
def bromine_atoms : ℕ := 1
def hydrogen_weight : ℝ := 1.008
def bromine_weight : ℝ := 79.904

-- Statement to prove
theorem atomic_weight_of_nitrogen :
  let total_hydrogen_weight := hydrogen_atoms * hydrogen_weight in
  let total_bromine_weight := bromine_atoms * bromine_weight in
  let total_others_weight := total_hydrogen_weight + total_bromine_weight in
  let nitrogen_weight := molecular_weight - total_others_weight in
  nitrogen_weight = 14.064 :=
by 
  sorry

end atomic_weight_of_nitrogen_l630_630469


namespace max_slope_OQ_l630_630374

theorem max_slope_OQ (p : ℝ) (hp : 0 < p) (y1 : ℝ) (hy1 : 0 < y1) :
  let O : (ℝ × ℝ) := (0, 0)
  let F : (ℝ × ℝ) := (p / 2, 0)
  let P : (ℝ × ℝ) := (y1 ^ 2 / (2 * p), y1)
  let Q : (ℝ × ℝ) := (2 / 3 * (y1 ^ 2 / (2 * p)) + 1 / 3 * (p / 2), 2 / 3 * y1)
  let slope : ℝ := Q.2 / Q.1
  in slope ≤ sqrt 2 :=
sorry

end max_slope_OQ_l630_630374


namespace sin_6_phi_l630_630350

theorem sin_6_phi (φ : ℝ) (h : complex.exp (complex.I * φ) = (3 + complex.I * real.sqrt 8) / 5) :
  real.sin (6 * φ) = -396 * real.sqrt 2 / 15625 :=
by sorry

end sin_6_phi_l630_630350


namespace abs_T_n_le_one_l630_630064

noncomputable def T_n (n : ℕ) (x : ℝ) : ℝ := 
  if h : |x| ≤ 1 then 
    let φ := Real.arccos x in
    Real.cos (n * φ)
  else 
    0

theorem abs_T_n_le_one (n : ℕ) (x : ℝ) (h : x ≤ 1) : |T_n n x| ≤ 1 := 
by {
  sorry
}

end abs_T_n_le_one_l630_630064


namespace telescoping_series_problem_statement_l630_630398

noncomputable def sum_expression (n : ℕ) : ℝ := ∑ k in finset.range (n + 1) \ {0}, 1 / (real.sqrt (k + real.sqrt (k^2 - k)))

theorem telescoping_series :
  ∑ n in finset.range 5000, 1 / (real.sqrt (n + real.sqrt (n^2 - n))) = 50 * real.sqrt 2 :=
by
  sorry

theorem problem_statement :
  let T := sum_expression 5000 in
  ∃ (a b c : ℤ), T = a + b * real.sqrt c ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ ¬ ∃ p : ℤ, p > 1 ∧ p^2 ∣ c ∧ a + b + c = 53 :=
by
  use 50, 1, 2
  split
  apply telescoping_series
  repeat {split}
  dec_trivial
  dec_trivial
  dec_trivial
  intro p
  rintro ⟨hp1, hp2⟩
  exact int.not_true (p^2 ∣ 2) hp2
  dec_trivial

end telescoping_series_problem_statement_l630_630398


namespace decimal_to_binary_123_l630_630510

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630510


namespace answer_correct_l630_630185

variable {A B : Type}

def prob_A (p_A : ℝ) := p_A = 0.65
def prob_B (p_B : ℝ) := p_B = 0.55
def prob_neither (p_neither : ℝ) := p_neither = 0.20

def prob_both (p_both : ℝ) : Prop :=
  (∀ p_A p_B p_neither, prob_A p_A → prob_B p_B → prob_neither p_neither → p_both = p_A + p_B - (1 - p_neither))

theorem answer_correct : prob_both 0.40 :=
by
  intros p_A p_B p_neither hA hB hNe
  rw [hA, hB, hNe]
  norm_num
  sorry

end answer_correct_l630_630185


namespace jack_keeps_deers_weight_is_correct_l630_630383

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end jack_keeps_deers_weight_is_correct_l630_630383


namespace problem_condition_A_problem_condition_B_l630_630762

variable (z : ℂ)

def conjugate (z : ℂ) : ℂ := complex.conj z

theorem problem_condition_A : (z + conjugate z).im = 0 := sorry

theorem problem_condition_B : (z * conjugate z).im = 0 := sorry

end problem_condition_A_problem_condition_B_l630_630762


namespace like_terms_exponents_l630_630756

theorem like_terms_exponents (m n : ℤ) (h1 : m = 2) (h2 : n = 3) : (-n)^m = 9 :=
by {
  rw [h1, h2],
  norm_num,
  exact sorry,
}

end like_terms_exponents_l630_630756


namespace mean_temperature_correct_l630_630454

-- Define the list of temperatures
def temperatures : List ℤ := [-8, -5, -5, -6, 0, 4]

-- Define the mean temperature calculation
def mean_temperature (temps: List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

-- The theorem we want to prove
theorem mean_temperature_correct :
  mean_temperature temperatures = -10 / 3 :=
by
  sorry

end mean_temperature_correct_l630_630454


namespace square_perimeter_l630_630633

theorem square_perimeter (s : ℝ) (h1 : 5 * s = (n : ℝ) * l) (l = 48) : (2 * (s + (s / 5)) = 80) := 
by
  sorry

end square_perimeter_l630_630633


namespace problem_1_problem_2_problem_3_l630_630439

-- Problem 1
theorem problem_1 : 
  99 * (118 + 4 / 5) + 99 * (-1 / 5) - 99 * (18 + 3 / 5) = 9900 :=
sorry

-- Problem 2
theorem problem_2 : 
  24 * ((1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 4) * ... * (1 - 1 / 24)) = 1 :=
sorry

-- Problem 3
noncomputable def reciprocal_difference (a : ℚ) : ℚ := 1 / (1 - a)

def a_seq : ℕ → ℚ
| 0       := -2
| (n + 1) := reciprocal_difference (a_seq n)

theorem problem_3 :
  a_seq 0 + a_seq 1 + a_seq 2 - 2 * a_seq 3 - 2 * a_seq 4 - 2 * a_seq 5 + 3 * a_seq 6 + 3 * a_seq 7 + 3 * a_seq 8 = -(1 / 3) :=
sorry

end problem_1_problem_2_problem_3_l630_630439


namespace minimize_permutation_sum_l630_630471

theorem minimize_permutation_sum (x : ℝ) (x1 x2 x3 x4 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4) :
  let a := 1, b := 4, c := 2, d := 3 in
  (x1 - x4)^2 + (x4 - x2)^2 + (x2 - x3)^2 + (x3 - x1)^2 ≤ 
  (x - x)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x1)^2 := 
sorry

end minimize_permutation_sum_l630_630471


namespace overall_gain_percentage_l630_630212

def cost1 := 4000
def cost2 := 6000
def cost3 := 8000
def total_cost := cost1 + cost2 + cost3

def sale1 := 4500
def sale2 := 6500
def sale3 := 8500
def total_sale := sale1 + sale2 + sale3

def gain := total_sale - total_cost
def gain_percentage := (gain / total_cost) * 100

theorem overall_gain_percentage : gain_percentage = 8.33 :=
sorry

end overall_gain_percentage_l630_630212


namespace sum_special_primes_correct_l630_630709

open Nat

def is_special_prime (p : ℕ) : Prop :=
  p ∈ primes ∧ p % 42 = 5 ∧ p ≤ 100

def sum_special_primes : ℕ :=
  ∑ p in (filter is_special_prime (List.range 101)), p

theorem sum_special_primes_correct : sum_special_primes = 141 := by
  sorry

end sum_special_primes_correct_l630_630709


namespace angle_comparison_l630_630292

-- Definitions:
def cyclic_quadrilateral (A B C D P : Type) [MetricSpace A] : Prop :=
  ∃ (cyclic : A × B × C × D), ∀ W1 W2 W3: A, dist W1 W2 * dist W2 W3 * dist W3 W1 = dist A B * dist B C * dist C D

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Mathematical Statement:
theorem angle_comparison 
  (P : Type)
  (ABCD_same_sides : ∀ (a b c d : ℝ), quadrilateral_with_sides A B C D P a b c d)
  (cyclic_ABCD : cyclic_quadrilateral A B C D P)
  (angle_phi : ℝ) 
  (acute_angle_diag : ∀ (ψ : ℝ), psi ≤ φ) :
  ∀ (Q : Quadrilateral), (∀ (a b c d : ℝ), quadrilateral_with_sides Q4 a b c d) → acute_angle_between_diagonals Q ≤ φ := 
begin
  sorry
end

end angle_comparison_l630_630292


namespace find_root_interval_l630_630882

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem find_root_interval : ∃ k : ℕ, (f 1 < 0 ∧ f 2 > 0) → k = 1 :=
by
  sorry

end find_root_interval_l630_630882


namespace calculate_g_l630_630879

def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

theorem calculate_g : g 3 6 (-1) = 1 / 7 :=
by
    -- Proof is not included
    sorry

end calculate_g_l630_630879


namespace total_football_games_l630_630935

theorem total_football_games (months : ℕ) (games_per_month : ℕ) (season_length : months = 17 ∧ games_per_month = 19) :
  (months * games_per_month) = 323 :=
by
  sorry

end total_football_games_l630_630935


namespace reciprocal_of_neg_2023_l630_630487

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l630_630487


namespace total_cost_is_correct_l630_630851

def rabbit_toy_cost : ℝ := 6.51
def pet_food_cost : ℝ := 5.79
def cage_cost : ℝ := 12.51
def total_cost : ℝ := rabbit_toy_cost + pet_food_cost + cage_cost

theorem total_cost_is_correct : total_cost = 24.81 := 
by
  unfold total_cost
  unfold rabbit_toy_cost
  unfold pet_food_cost
  unfold cage_cost
  norm_num
  sorry

end total_cost_is_correct_l630_630851


namespace wheres_waldo_books_published_l630_630497

theorem wheres_waldo_books_published (total_minutes : ℕ) (minutes_per_puzzle : ℕ) (puzzles_per_book : ℕ)
  (h1 : total_minutes = 1350) (h2 : minutes_per_puzzle = 3) (h3 : puzzles_per_book = 30) :
  total_minutes / minutes_per_puzzle / puzzles_per_book = 15 :=
by
  sorry

end wheres_waldo_books_published_l630_630497


namespace jack_helping_toddlers_shoes_l630_630010

theorem jack_helping_toddlers_shoes (t_jack t_total : ℕ) (h_tjack : t_jack = 4) (h_ttotal : t_total = 18) :
  let t_help_total := t_total - t_jack in
  let t_help_each := t_help_total / 2 in
  t_help_each - t_jack = 3 :=
by {
  sorry
}

end jack_helping_toddlers_shoes_l630_630010


namespace david_marks_in_physics_l630_630244

theorem david_marks_in_physics
  (marks_english : ℕ)
  (marks_math : ℕ)
  (marks_chemistry : ℕ)
  (marks_biology : ℕ)
  (average_marks : ℚ)
  (num_subjects : ℕ)
  (h_marks_english : marks_english = 96)
  (h_marks_math : marks_math = 95)
  (h_marks_chemistry : marks_chemistry = 87)
  (h_marks_biology : marks_biology = 92)
  (h_average_marks : average_marks = 90.4)
  (h_num_subjects : num_subjects = 5) :
  (452 - (marks_english + marks_math + marks_chemistry + marks_biology) = 82) :=
by
  have total_marks := average_marks * num_subjects
  have total_known_marks := marks_english + marks_math + marks_chemistry + marks_biology
  have marks_physics := total_marks - total_known_marks
  calc
    marks_physics = 452 - 370 : sorry

end david_marks_in_physics_l630_630244


namespace scientific_notation_gdp_l630_630643

theorem scientific_notation_gdp :
  8837000000 = 8.837 * 10^9 := 
by
  sorry

end scientific_notation_gdp_l630_630643


namespace fraction_subtraction_identity_l630_630353

theorem fraction_subtraction_identity (x y : ℕ) (hx : x = 3) (hy : y = 4) : (1 / (x : ℚ) - 1 / (y : ℚ) = 1 / 12) :=
by
  sorry

end fraction_subtraction_identity_l630_630353


namespace solve_for_x_l630_630827

theorem solve_for_x 
  (a b c d x y z w : ℝ) 
  (H1 : x + y + z + w = 360)
  (H2 : a = x + y / 2) 
  (H3 : b = y + z / 2) 
  (H4 : c = z + w / 2) 
  (H5 : d = w + x / 2) : 
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) :=
sorry


end solve_for_x_l630_630827


namespace balanced_apple_trees_l630_630284

theorem balanced_apple_trees: 
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (x1 * y2 - x1 * y4 - x3 * y2 + x3 * y4 = 0) ∧
    (x2 * y1 - x2 * y3 - x4 * y1 + x4 * y3 = 0) ∧
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4) :=
  sorry

end balanced_apple_trees_l630_630284


namespace number_of_valid_N_count_valid_N_is_seven_l630_630729

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l630_630729


namespace quadratic_coeff_b_is_4_sqrt_15_l630_630424

theorem quadratic_coeff_b_is_4_sqrt_15 :
  ∃ m b : ℝ, (x^2 + bx + 72 = (x + m)^2 + 12) → (m = 2 * Real.sqrt 15) → (b = 4 * Real.sqrt 15) ∧ b > 0 :=
by
  -- Note: Proof not included as per the instruction.
  sorry

end quadratic_coeff_b_is_4_sqrt_15_l630_630424


namespace area_of_circle_l630_630943

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 + 8 * x + 10 * y = -9 → 
  ∃ a : ℝ, a = 32 * Real.pi :=
by
  sorry

end area_of_circle_l630_630943


namespace a_minus_b_value_l630_630378

/-
We start by defining the points involved in the problem.
- We have points (5, 5) and (9, 2).
- We have points (a, 13) and (15, b).
- We assume the shift from (5, 5) to (9, 2) should equal the shift from (a, 13) to (15, b).
-/

def point1 := (5, 5)
def point2 := (9, 2)
def point3 := (a, 13 : ℕ)
def point4 := (15, b : ℕ)

theorem a_minus_b_value (a b : ℕ) :
  let horizontal_shift := point2.1 - point1.1
  let vertical_shift := point2.2 - point1.2
  (point3.1 + horizontal_shift = point4.1) → (point3.2 + vertical_shift = point4.2) → a - b = 1 :=
by
  intros h_shift v_shift
  unfold point3 point4 at h_shift v_shift
  rw [h_shift, v_shift]
  sorry

end a_minus_b_value_l630_630378


namespace difference_of_squares_l630_630950

theorem difference_of_squares : 
  let a := 625
  let b := 575
  (a^2 - b^2) = 60000 :=
by 
  let a := 625
  let b := 575
  sorry

end difference_of_squares_l630_630950


namespace irrational_neg_sqrt_5_rational_sqrt_4_rational_frac_2_3_rational_zero_l630_630180

noncomputable def sqrt_4 : ℝ := Real.sqrt 4
noncomputable def neg_sqrt_5 : ℝ := - Real.sqrt 5
def frac_2_3 : ℝ := 2 / 3
def zero : ℝ := 0

theorem irrational_neg_sqrt_5 : ¬ (∃ (a b : ℤ), b ≠ 0 ∧ neg_sqrt_5 = a / b) :=
sorry

theorem rational_sqrt_4 : ∃ (a b : ℤ), b ≠ 0 ∧ sqrt_4 = a / b :=
sorry

theorem rational_frac_2_3 : ∃ (a b : ℤ), b ≠ 0 ∧ frac_2_3 = a / b :=
sorry

theorem rational_zero : ∃ (a b : ℤ), b ≠ 0 ∧ zero = a / b :=
sorry

end irrational_neg_sqrt_5_rational_sqrt_4_rational_frac_2_3_rational_zero_l630_630180


namespace total_fish_l630_630971

theorem total_fish {lilly_fish rosy_fish : ℕ} (h1 : lilly_fish = 10) (h2 : rosy_fish = 11) : 
lilly_fish + rosy_fish = 21 :=
by 
  sorry

end total_fish_l630_630971


namespace length_of_AB_l630_630992

theorem length_of_AB (A B : ℝ × ℝ)
  (hA : ∃ x₁, A = (x₁, 4 * x₁ ^ 2))
  (hB : ∃ x₂, B = (x₂, 4 * x₂ ^ 2))
  (h_line : ∃ k, ∀ (P : ℝ × ℝ), P ∈ {A, B} ↔ ∃ x, P = (x, k * x + 1 / 16))
  (h_sum_y : A.snd + B.snd = 2) : 
  dist A B = 17 / 8 :=
by sorry

end length_of_AB_l630_630992


namespace goods_train_length_proof_l630_630995

def train_speed_man : ℝ := 50 -- speed of man's train in km/h
def train_speed_goods : ℝ := 62 -- speed of goods train in km/h
def pass_time : ℝ := 9 -- time for goods train to pass the man in seconds

-- Conversion factor from km/h to m/s
def kmph_to_mps : ℝ := 1000 / 3600 

-- Calculate the relative speed in km/h
def relative_speed_kmph : ℝ := train_speed_man + train_speed_goods

-- Convert relative speed to m/s
def relative_speed_mps : ℝ := relative_speed_kmph * kmph_to_mps

-- Calculate the length of the goods train
def goods_train_length : ℝ := pass_time * relative_speed_mps

theorem goods_train_length_proof : 
    goods_train_length = 280 := 
by
    sorry

end goods_train_length_proof_l630_630995


namespace number_of_tigers_each_enclosure_is_4_l630_630642

/-- Given:
1. There are 4 tiger enclosures.
2. Behind each tiger enclosure, there are 2 zebra enclosures.
3. There are three times as many giraffe enclosures as zebra enclosures.
4. Each zebra enclosure holds 10 zebras.
5. Each giraffe enclosure holds 2 giraffes.
6. The total number of animals in the zoo is 144.

Prove: The number of tigers in each tiger enclosure is 4.
-/
theorem number_of_tigers_each_enclosure_is_4 : 
  (∃ T : ℕ, 
    let tiger_count := 4 * T,
        zebra_enclosures := 4 * 2,
        zebra_count := zebra_enclosures * 10,
        giraffe_enclosures := 3 * zebra_enclosures,
        giraffe_count := giraffe_enclosures * 2,
        total_animals := 144
    in 
    tiger_count + zebra_count + giraffe_count = total_animals ∧ T = 4) :=
by {
  sorry
}

end number_of_tigers_each_enclosure_is_4_l630_630642


namespace num_positive_int_values_for_expression_is_7_l630_630723

theorem num_positive_int_values_for_expression_is_7 :
  {N : ℕ // 0 < N ∧ ∃ k : ℕ, 48 = k * (N + 3)}.card = 7 := 
sorry

end num_positive_int_values_for_expression_is_7_l630_630723


namespace largest_hexagon_angle_l630_630468

-- We define the conditions first
def angle_ratios (x : ℝ) := [3*x, 3*x, 3*x, 4*x, 5*x, 6*x]
def sum_of_angles (angles : List ℝ) := angles.sum = 720

-- Now we state our proof goal
theorem largest_hexagon_angle :
  ∀ (x : ℝ), sum_of_angles (angle_ratios x) → 6 * x = 180 :=
by
  intro x
  intro h
  sorry

end largest_hexagon_angle_l630_630468


namespace drink_cost_l630_630388

theorem drink_cost (meal_cost : ℝ) (tip_rate : ℝ) (paid_amount : ℝ) (change : ℝ) (total_spent : ℝ)
    (h_paid : paid_amount - change = total_spent) : 
    meal_cost = 10 → tip_rate = 0.20 → paid_amount = 20 → change = 5 → total_spent = 15 → 
    let drink_cost := (total_spent - (meal_cost + 0.20 * meal_cost)) / 1.20 in 
    drink_cost = 2.50 :=
by
  intros h_meal h_tip h_paid_amount h_change h_total_spent
  have h_equation : 10 + (total_spent - (10 + 0.20 * 10)) / 1.20 + 0.20 * (10 + (total_spent - (10 + 0.20 * 10)) / 1.20) = 15,
  calc
    10 + (total_spent - (10 + 0.20 * 10)) / 1.20 + 0.20 * (10 + (total_spent - (10 + 0.20 * 10)) / 1.20)
        = 15 : by sorry,
  rw [h_total_spent] at h_equation,
  exact h_equation

end drink_cost_l630_630388


namespace equal_angles_of_pyramid_l630_630839

variables {P : Type*} [EuclideanGeometry P]
variables {A B C S D : P}

-- The lateral edge SA is perpendicular to the base ABC
def SA_perpendicular_to_ABC (h1 : S ≠ A) (h2 : isPerpendicular SA (plane A B C)) : Prop := true

-- The angle bisectors of <BAC and <BSC intersect
def angle_bisectors_intersect (h3 : lies_on D (segment B C)) (h4 : isAngleBisector A B C B S D S) : Prop := true

-- Prove <ABC = <ACB
theorem equal_angles_of_pyramid
  (h1 : S ≠ A)
  (h2 : isPerpendicular SA (plane A B C))
  (h3 : lies_on D (segment B C))
  (h4 : isAngleBisector A B C B S D S) :
  ∠ABC = ∠ACB :=
  sorry

end equal_angles_of_pyramid_l630_630839


namespace coordinates_of_C_l630_630474

theorem coordinates_of_C (a : ℝ) :
  ∃ Cx : ℝ, ∃ Cy : ℝ,
    Cx = 126 / (4 * a - 2) ∧ Cy = 4 * a - 2 ∧
    (let A : ℝ × ℝ := (0, 0) in
     let B : ℝ × ℝ := (0, 4 * a - 2) in
     let C : ℝ × ℝ := (Cx, Cy) in
     (0 - Cx) * (4 * a - 2)) / 2 = 63 :=
by
  sorry

end coordinates_of_C_l630_630474


namespace anna_final_stamp_count_l630_630654

theorem anna_final_stamp_count :
  let anna_initial_nature := 10
  let anna_initial_architecture := 15
  let anna_initial_animals := 12
  let alison_initial_nature := 8
  let alison_initial_architecture := 10
  let alison_initial_animals := 10
  let jeff_initial_nature := 12
  let jeff_initial_architecture := 9
  let jeff_initial_animals := 10
  let anna_after_alison_nature := anna_initial_nature + 4
  let anna_after_alison_architecture := anna_initial_architecture + 5
  let anna_after_alison_animals := anna_initial_animals + 5
  let anna_after_jeff1_nature := anna_after_alison_nature + 2
  let anna_after_jeff1_architecture := anna_after_alison_architecture
  let anna_after_jeff1_animals := anna_after_alison_animals - 1
  let anna_after_jeff2_nature := anna_after_jeff1_nature
  let anna_after_jeff2_architecture := anna_after_jeff1_architecture + 3
  let anna_after_jeff2_animals := anna_after_jeff1_animals - 5
  let anna_final_nature := anna_after_jeff2_nature + 7
  let anna_final_architecture := anna_after_jeff2_architecture
  let anna_final_animals := anna_after_jeff2_animals - 4
  anna_final_nature = 23 ∧ anna_final_architecture = 23 ∧ anna_final_animals = 7 :=
by
  simp [anna_final_nature, anna_final_architecture, anna_final_animals, 
        anna_after_alison_nature, anna_after_alison_architecture, anna_after_alison_animals, 
        anna_after_jeff1_nature, anna_after_jeff1_architecture, anna_after_jeff1_animals, 
        anna_after_jeff2_nature, anna_after_jeff2_architecture, anna_after_jeff2_animals]
  sorry

end anna_final_stamp_count_l630_630654


namespace exists_noncongruent_triangles_with_same_area_l630_630382

-- Defining the circle and inscribable triangles
noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  in sqrt (s * (s - a) * (s - b) * (s - c))

-- A given circle with radius r > 0
variables (r : ℝ) (h_r_pos : r > 0)

-- Proving the existence of two non-congruent triangles T1 and T2 with the same area that can be inscribed in the circle
theorem exists_noncongruent_triangles_with_same_area :
  ∃ (T1 T2 : ℝ × ℝ × ℝ), 
    (T1 ≠ T2) ∧
    (let ⟨a1, b1, c1⟩ := T1 in a1 < 2 * r ∧ b1 < 2 * r ∧ c1 < 2 * r ∧ a1 + b1 + c1 = 2 * r) ∧
    (let ⟨a2, b2, c2⟩ := T2 in a2 < 2 * r ∧ b2 < 2 * r ∧ c2 < 2 * r ∧ a2 + b2 + c2 = 2 * r) ∧
    (triangle_area (T1.1) (T1.2) (T1.3) = triangle_area (T2.1) (T2.2) (T2.3)) :=
sorry

end exists_noncongruent_triangles_with_same_area_l630_630382


namespace good_time_prevalent_l630_630458
noncomputable def isGoodTime (h m s : Nat) : Prop := 
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = (h % 12) * 30 + m / 2 + s / 120 ∧ 
                    θ₂ = (m % 60) * 6 + s / 10 ∧ 
                    θ₃ = (s % 60) * 6 ∧ 
                    ( ∀ d : ℝ, 0 ≤ d ∧ d < π → 
                        ( (θ₁ < θ₂ + d ∨ θ₂ < θ₁ + d) ∧
                          (θ₂ < θ₃ + d ∨ θ₃ < θ₂ + d) ∧ 
                          (θ₁ < θ₃ + d ∨ θ₃ < θ₁ + d)))

theorem good_time_prevalent : 
  ∑ i in range 24, (∑ j in range 60, (∑ k in range 60, if isGoodTime i j k then 1 else 0))
  > ∑ i in range 24, (∑ j in range 60, (∑ k in range 60, if ¬ isGoodTime i j k then 1 else 0)) :=
sorry

end good_time_prevalent_l630_630458


namespace proof_x_plus_y_l630_630895

variables (x y : ℝ)

-- Definitions for the given conditions
def cond1 (x y : ℝ) : Prop := 2 * |x| + x + y = 18
def cond2 (x y : ℝ) : Prop := x + 2 * |y| - y = 14

theorem proof_x_plus_y (x y : ℝ) (h1 : cond1 x y) (h2 : cond2 x y) : x + y = 14 := by
  sorry

end proof_x_plus_y_l630_630895


namespace vec_a_squared_minus_vec_b_squared_l630_630791

variable (a b : ℝ × ℝ)
variable (h1 : a + b = (-3, 6))
variable (h2 : a - b = (-3, 2))

theorem vec_a_squared_minus_vec_b_squared : (a.1 * a.1 + a.2 * a.2) - (b.1 * b.1 + b.2 * b.2) = 32 :=
sorry

end vec_a_squared_minus_vec_b_squared_l630_630791


namespace train_time_first_platform_correct_l630_630638

-- Definitions
variables (L_train L_first_plat L_second_plat : ℕ) (T_second : ℕ) (T_first : ℕ)

-- Given conditions
def length_train := 350
def length_first_platform := 100
def length_second_platform := 250
def time_second_platform := 20
def expected_time_first_platform := 15

-- Derived values
def total_distance_second_platform := length_train + length_second_platform
def speed := total_distance_second_platform / time_second_platform
def total_distance_first_platform := length_train + length_first_platform
def time_first_platform := total_distance_first_platform / speed

-- Proof Statement
theorem train_time_first_platform_correct : 
  time_first_platform = expected_time_first_platform :=
  by
  sorry

end train_time_first_platform_correct_l630_630638


namespace function_repeated_value_achieved_at_least_1993_times_l630_630462

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem function_repeated_value_achieved_at_least_1993_times :
  ∃ x₀, (∃ S : Finset ℝ, ∃ y : ℝ, y ∈ S ∧ S.card ≥ 1993 ∧ (∃ (n : ℕ), ∀ x ∈ S, (iterated f n x = x₀))) :=
by { sorry }

end function_repeated_value_achieved_at_least_1993_times_l630_630462


namespace base_2_representation_of_123_l630_630547

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630547


namespace problem_monotonic_intervals_and_max_ab_l630_630740

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := real.exp x - a * x - b

theorem problem_monotonic_intervals_and_max_ab (a b : ℝ) :
  (f 0 a b = 0) → (f'(0) = 0) → (∀ x : ℝ, (f x a b) ≥ 0) →
  ( ∀ x : ℝ, (e^x - a = 0 → ( a = 1)) → 
    ( ∀ x : ℝ, ( e^x  -  x - b = 1) → ( b = 1)) → 
      ( f'(x) > 0 → x > 0 ) ∧
      ( f'(x) < 0 → x < 0 ) ∧
      ( ∀ a > 0, ( f(x) = e^(ln a) -a (ln (a) - b) ≥ 0 ) → b ≤ a(1-ln a) →
        max (ab) = e/2 ) ) := sorry

end problem_monotonic_intervals_and_max_ab_l630_630740


namespace base_2_representation_of_123_l630_630537

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630537


namespace semicircle_center_of_mass_semi_disk_center_of_mass_l630_630702

theorem semicircle_center_of_mass (r : ℝ) (h : r > 0) : 
  let y := (2 * r) / π in 
  ∃ y, y = (2 * r) / π :=
begin
  use (2 * r) / π,
  sorry
end

theorem semi_disk_center_of_mass (r : ℝ) (h : r > 0) : 
  let z := (4 * r) / (3 * π) in 
  ∃ z, z = (4 * r) / (3 * π) :=
begin
  use (4 * r) / (3 * π),
  sorry
end

end semicircle_center_of_mass_semi_disk_center_of_mass_l630_630702


namespace cost_of_45_roses_l630_630232

theorem cost_of_45_roses (cost_15_roses : ℕ → ℝ) 
  (h1 : cost_15_roses 15 = 25) 
  (h2 : ∀ (n m : ℕ), cost_15_roses n / n = cost_15_roses m / m )
  (h3 : ∀ (n : ℕ), n > 30 → cost_15_roses n = (1 - 0.10) * cost_15_roses n) :
  cost_15_roses 45 = 67.5 :=
by
  sorry

end cost_of_45_roses_l630_630232


namespace cost_of_first_20_kgs_l630_630970

theorem cost_of_first_20_kgs 
  (l m n : ℕ) 
  (hl1 : 30 * l +  3 * m = 333) 
  (hl2 : 30 * l +  6 * m = 366) 
  (hl3 : 30 * l + 15 * m = 465) 
  (hl4 : 30 * l + 20 * m = 525) 
  : 20 * l = 200 :=
by
  sorry

end cost_of_first_20_kgs_l630_630970


namespace power_mod_l630_630159

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l630_630159


namespace math_problem_l630_630334

-- Definitions of vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vector_b : ℝ × ℝ := (3, -Real.sqrt 3)

-- Definitions of conditions
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Proof problem
theorem math_problem 
  (x : ℝ)
  (h₀ : 0 ≤ x ∧ x ≤ π)
  (h₁ : is_parallel (vector_a x) vector_b) :
  x = 5 * π / 6 ∧ 
  (f : ℝ → ℝ), 
  f = λ x, dot_product (vector_a x) vector_b ∧
  (∀ (x : ℝ), f x ≤ 3) ∧
  (∀ (x : ℝ), f x ≥ -2 * Real.sqrt 3) :=
by
  sorry

end math_problem_l630_630334


namespace range_of_a_l630_630714

-- Define the condition function
def inequality (a x : ℝ) : Prop := a^2 * x - 2 * (a - x - 4) < 0

-- Prove that given the inequality always holds for any real x, the range of a is (-2, 2]
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, inequality a x) : -2 < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l630_630714


namespace skateboarder_speed_l630_630077

theorem skateboarder_speed :
  let distance := 293.33
  let time := 20
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  let speed_ft_per_sec := distance / time
  let speed_mph := speed_ft_per_sec * (feet_per_mile / seconds_per_hour)
  speed_mph = 21.5 :=
by
  sorry

end skateboarder_speed_l630_630077


namespace inequality_proof_l630_630025

theorem inequality_proof (x : Fin 2011 → ℝ) (h1 : ∀ i, x i > 1) :
    (∑ i in Finset.range 2011, x i * x i / (x (⟨i + 1 % 2011, sorry⟩ ) - 1)) ≥ 8044 :=
sorry

end inequality_proof_l630_630025


namespace star_evaluation_l630_630280

def star (a b : ℝ) (h : a ≠ b) : ℝ :=
  (a^2 + b^2) / (a - b)

theorem star_evaluation : ((star 2 3 (by norm_num) star 4 (by norm_num)) = - (185 / 17)) :=
by
  have h23 : 2 ≠ 3 := by norm_num
  have hneg13 : star 2 3 h23 = -13 := by norm_num
  have hneg := star (-13) 4 (by norm_num)
  -- skipping detailed proof steps for brevity
  have h4 : star (-13) 4 (by norm_num) = - (185 / 17) := by norm_num
  exact h4

end star_evaluation_l630_630280


namespace solution_l630_630754

def p : Prop := ∀ x > 0, Real.log (x + 1) > 0
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

theorem solution : p ∧ ¬ q := by
  sorry

end solution_l630_630754


namespace polynomial_representation_polynomial_integer_values_iff_l630_630573

/-- Definition of x^(n) as given in the problem --/
def xFn (x : ℝ) (n : ℕ) : ℝ :=
  (List.prod (List.iota n).map (λ k, x - k)) / (Nat.factorial n)

/-- Part (a): Every polynomial of degree n can be represented as b_0 + b_1 x^(1) + ... + b_n x^(n) --/
theorem polynomial_representation (n : ℕ) (p : ℝ → ℝ) (h_deg : ∀ x, p x = ∑ i in Finset.range (n + 1), (p x - ∑ j in Finset.range i, xFn x j) * x ^ i) :
    ∃ (b : Fin (n + 1) → ℝ), ∀ x, p x = ∑ i in Finset.range (n + 1), b i * xFn x i :=
sorry

/-- Part (b): A polynomial takes integer values at all integer points if and only if all b_i are integers --/
theorem polynomial_integer_values_iff (n : ℕ) (p : ℝ → ℝ) :
  (∀ x : ℤ, (p x).isInt) ↔ ∀ (b : Fin (n + 1) → ℝ), (∀ (i : Fin (n + 1)), b i ∈ ℤ) :=
sorry

end polynomial_representation_polynomial_integer_values_iff_l630_630573


namespace tangent_through_M_unique_equal_intercepts_chord_length_l630_630783

-- Define the point M and circle C
def point_M (m : ℝ) := (1 : ℝ, m)
def circle_C (x y : ℝ) := x^2 + y^2 = 4

-- Define the first problem
theorem tangent_through_M_unique {m : ℝ} (h : circle_C 1 m) :
  (∀ k, (∀ x y, y - m = k * (x - 1) → ¬circle_C x y) → (k = -√3 / 3)) →
  (m = √3 ∨ m = -√3) ∧ ((x + √3 * y - 4 = 0) ∨ (x - √3 * y - 4 = 0)) :=
sorry

-- Define the second problem
theorem equal_intercepts_chord_length {m : ℝ} (h : ∃ a : ℝ, a ≠ 0 ∧ (∀ x y, x + y - a = 0 → ¬circle_C x y) ∧ (point_M m).fst + (point_M m).snd = a ∧ (2 * a = 2√3)) :
  m = (-1 + √2) ∨ m = (-1 - √2) :=
sorry

end tangent_through_M_unique_equal_intercepts_chord_length_l630_630783


namespace rearrange_table_l630_630145

theorem rearrange_table (n : ℕ) (T : Array (Array ℕ)) (h : ∀ i, (T[i].Nodup)) :
  ∃ T_star : Array (Array ℕ), (∀ i, T_star[i] ~ T[i]) ∧ (∀ j, (T_star.Column j).Nodup) :=
by
  sorry

end rearrange_table_l630_630145


namespace min_value_D_l630_630226

noncomputable def A (x : ℝ) : ℝ := x^2 + x
noncomputable def B (x : ℝ) : ℝ := |real.log x|
noncomputable def C (x : ℝ) : ℝ := x * real.sin x
noncomputable def D (x : ℝ) : ℝ := real.exp x + real.exp (-x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem min_value_D : is_even_function D ∧ ∀ x, D x ≥ 2 ∧ D 0 = 2 :=
by
  have even_D : is_even_function D :=
    fun x => by rw [D, D, real.exp_neg, add_comm]
  have deriv_D : ∀ x, deriv D x = real.exp x - real.exp (-x) :=
    fun x => by simp [D, deriv_exp, deriv_neg, deriv_id', deriv_add']
  have min_value_at_x0 : ∀ x, real.exp x + real.exp (-x) ≥ 2 :=
    fun x => by {
      let y := real.exp x
      let z := real.exp (-x)
      have H : y * z = 1 := by rw [real.exp_neg, ←mul_assoc, mul_comm y]
      show y + z ≥ 2
      exact real.add_le_add (real.exp_pos x).le (one_le_exp (-x))
      sorry
    }
  exact ⟨even_D, min_value_at_x0, rfl⟩

end min_value_D_l630_630226


namespace binom_expansion_coeff_l630_630800

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_expansion_coeff
  (n : ℕ)
  (h : (∫ x in 0..n, |x - 5|) = 25) :
  binom_coeff n 2 * 2^2 * (-1)^8 = 180 := 
by
  sorry

end binom_expansion_coeff_l630_630800


namespace surface_area_of_cuboid_correct_l630_630184

-- Define the cuboid dimensions.
variables (l b h : ℝ)
-- Set the cuboid dimensions to the given values.
axiom cuboid_dimensions : l = 8 ∧ b = 6 ∧ h = 9

-- Surface area of a cuboid.
def surface_area_cuboid (l b h : ℝ) : ℝ :=
  2 * (l * b) + 2 * (b * h) + 2 * (l * h)

-- Theorem statement: Prove the surface area of the cuboid is 348 cm²
theorem surface_area_of_cuboid_correct :
  surface_area_cuboid l b h = 348 :=
by
  -- Use the given dimensions.
  have dimensions := cuboid_dimensions,
  cases dimensions with l_eq b_h_eq,
  cases b_h_eq with b_eq h_eq,
  -- Substitute dimensions into the equation.
  rw [l_eq, b_eq, h_eq],
  -- Calculate the surface area using the formula.
  sorry

end surface_area_of_cuboid_correct_l630_630184


namespace base_2_representation_of_123_l630_630543

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630543


namespace parallel_iff_no_common_points_l630_630577

-- Definitions based on given conditions
variable (α : Type) [LinearOrder α] (line_parallel : Prop) (no_common_points : Prop)

-- Conditions
def line_is_parallel_to_plane (α : Type) : Prop :=
  line_parallel

def line_and_plane_have_no_common_points (α : Type) : Prop :=
  no_common_points

-- Problem statement: Prove that the line and plane having no common points 
-- is a necessary and sufficient condition for the line being parallel to the plane.
theorem parallel_iff_no_common_points (line_parallel : Prop) (no_common_points : Prop) :
  line_is_parallel_to_plane α ↔ line_and_plane_have_no_common_points α :=
sorry

end parallel_iff_no_common_points_l630_630577


namespace problem_l630_630111

open Real

theorem problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
sorry

end problem_l630_630111


namespace number_of_people_tasting_apple_pies_l630_630889

/-- Sedrach's apple pie problem -/
def apple_pies : ℕ := 13
def halves_per_apple_pie : ℕ := 2
def bite_size_samples_per_half : ℕ := 5

theorem number_of_people_tasting_apple_pies :
    (apple_pies * halves_per_apple_pie * bite_size_samples_per_half) = 130 :=
by
  sorry

end number_of_people_tasting_apple_pies_l630_630889


namespace all_n_k_equal_one_l630_630300

theorem all_n_k_equal_one (k : ℕ) (hk : k ≥ 2) (n : Fin k → ℕ)
    (h1 : n 1 % (2^(n 0) - 1) = 0)
    (h2 : n 2 % (2^(n 1) - 1) = 0)
    (h3 : n 3 % (2^(n 2) - 1) = 0)
    (h4 : n (k-1) % (2^(n (k-2)) - 1) = 0)
    (h5 : n 0 % (2^(n (k-1)) - 1) = 0) :
    ∀ i : Fin k, n i = 1 :=
by sorry

end all_n_k_equal_one_l630_630300


namespace cubic_real_root_l630_630434

theorem cubic_real_root (a b : ℝ) (h : a = 2) (h1 : b = 6) (root_condition : (a * (-2 - 3i)^3 + 3 * (-2 - 3i)^2 + b * (-2 - 3i) - 65 = 0)) :
  ∃ x : ℝ, (a * x^3 + 3 * x^2 + b * x - 65 = 0) ∧ (x = 5/2) :=
by
  use 5 / 2
  sorry

end cubic_real_root_l630_630434


namespace age_difference_l630_630570

variable (P M Mo N : ℚ)

-- Given conditions as per problem statement
axiom ratio_P_M : (P / M) = 3 / 5
axiom ratio_M_Mo : (M / Mo) = 3 / 4
axiom ratio_Mo_N : (Mo / N) = 5 / 7
axiom sum_ages : P + M + Mo + N = 228

-- Statement to prove
theorem age_difference (ratio_P_M : (P / M) = 3 / 5)
                        (ratio_M_Mo : (M / Mo) = 3 / 4)
                        (ratio_Mo_N : (Mo / N) = 5 / 7)
                        (sum_ages : P + M + Mo + N = 228) :
  N - P = 69.5 := 
sorry

end age_difference_l630_630570


namespace Evelyn_has_104_marbles_l630_630690

theorem Evelyn_has_104_marbles :
  ∀ (initial_marbles : ℕ) (extra_marbles : ℕ) (cards_bought : ℕ),
  initial_marbles = 95 →
  extra_marbles = 9 →
  (initial_marbles + extra_marbles) = 104 :=
by
  intros initial_marbles extra_marbles cards_bought
  intro h_initial h_extra
  rw [h_initial, h_extra]
  norm_num

end Evelyn_has_104_marbles_l630_630690


namespace Robie_boxes_with_him_l630_630441

-- Definition of the given conditions
def total_cards : Nat := 75
def cards_per_box : Nat := 10
def cards_not_placed : Nat := 5
def boxes_given_away : Nat := 2

-- Definition of the proof that Robie has 5 boxes with him
theorem Robie_boxes_with_him : ((total_cards - cards_not_placed) / cards_per_box) - boxes_given_away = 5 := by
  sorry

end Robie_boxes_with_him_l630_630441


namespace total_oranges_picked_l630_630432

theorem total_oranges_picked : 
  let M := 100 in
  let T := 3 * M in
  let W := 70 in
  M + T + W = 470 :=
by
  let M := 100
  let T := 3 * M
  let W := 70
  show M + T + W = 470
  sorry

end total_oranges_picked_l630_630432


namespace john_total_calories_l630_630014

def calories_from_chips (total_calories_per_chip : ℕ) (num_chips : ℕ) : ℕ := total_calories_per_chip * num_chips

def calories_per_cheezit (calories_per_chip : ℕ) (additional_fraction : ℚ) : ℕ := 
  calories_per_chip + (calories_per_chip * additional_fraction).to_int

def calories_from_cheezits (calories_per_cheezit : ℕ) (num_cheezits : ℕ) : ℕ := calories_per_cheezit * num_cheezits

def total_calories (calories_chips : ℕ) (calories_cheezits : ℕ) : ℕ := calories_chips + calories_cheezits

theorem john_total_calories :
  let total_calories_per_chip := 60
  let num_chips := 10
  let num_cheezits := 6
  let additional_fraction := 1 / 3
  let calories_per_chip := total_calories_per_chip / num_chips
  let calories_cheezit := calories_per_cheezit calories_per_chip additional_fraction
  calories_chips + calories_cheezits = 108 :=
  sorry

end john_total_calories_l630_630014


namespace evaluate_expression_l630_630954

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end evaluate_expression_l630_630954


namespace honey_left_in_jar_l630_630615

theorem honey_left_in_jar 
  (initial_honey : ℝ)
  (draw_percentage : ℝ)
  (iterations : ℕ)
  (final_honey : ℝ)
  (h_init : initial_honey = 1.2499999999999998)
  (h_percent : draw_percentage = 0.20)
  (h_iterations : iterations = 4)
  (h_final_honey : final_honey = 0.512) :
  let honey_remaining := 
    (λ init_honey percentage iters, 
      let rec loop (curr_honey : ℝ) (n : ℕ) : ℝ :=
        if n = 0 then curr_honey
        else loop ((1 - percentage) * curr_honey) (n - 1)
      in loop init_honey iters) in
  honey_remaining initial_honey draw_percentage iterations = final_honey :=
by
  sorry

end honey_left_in_jar_l630_630615


namespace set_intersection_l630_630330

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem set_intersection :
  A ∩ B = { x | -1 < x ∧ x < 1 } := 
sorry

end set_intersection_l630_630330


namespace club_members_addition_l630_630593

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l630_630593


namespace slower_train_speed_l630_630139

theorem slower_train_speed (v : ℝ) (faster_train_speed : ℝ) (time_pass : ℝ) (train_length : ℝ) :
  (faster_train_speed = 46) →
  (time_pass = 36) →
  (train_length = 50) →
  (v = 36) :=
by
  intro h1 h2 h3
  -- Formal proof goes here
  sorry

end slower_train_speed_l630_630139


namespace sin_triple_angle_eq_three_sin_iff_sin_zero_l630_630915

theorem sin_triple_angle_eq_three_sin_iff_sin_zero {x : ℝ} :
  (sin (3 * x) = 3 * sin x) ↔ (sin x = 0) :=
sorry

end sin_triple_angle_eq_three_sin_iff_sin_zero_l630_630915


namespace car_speed_travel_l630_630202

theorem car_speed_travel (v : ℝ) :
  600 = 3600 / 6 ∧
  (6 : ℝ) = (3600 / v) + 2 →
  v = 900 :=
by
  sorry

end car_speed_travel_l630_630202


namespace base_2_representation_of_123_l630_630519

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630519


namespace problem1_problem2_l630_630776

noncomputable def f (x : ℝ) := 2^x - 1
noncomputable def f_inv (y : ℝ) := Real.log (y + 1) / Real.log 2
noncomputable def g (x : ℝ) := Real.log (3 * x + 1) / Real.log 4

def condition1 (x : ℝ) : Prop := f_inv x ≤ g x

theorem problem1 :
  ∀ x : ℝ, condition1 x ↔ (0 ≤ x ∧ x ≤ 1) :=
by sorry

noncomputable def H (x : ℝ) := g x - (1 / 2) * f_inv x

theorem problem2 :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
  (0 ≤ H x ∧ H x ≤ 1 / 2) :=
by sorry

end problem1_problem2_l630_630776


namespace hotel_towels_l630_630614

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l630_630614


namespace perfect_square_trinomial_l630_630260

theorem perfect_square_trinomial (x : ℝ) : (x + 9)^2 = x^2 + 18 * x + 81 := by
  sorry

end perfect_square_trinomial_l630_630260


namespace treaty_of_paris_signed_on_thursday_l630_630089

def start_day := "Wednesday"
def days_advanced := 3059
def target_day := "Thursday"

theorem treaty_of_paris_signed_on_thursday : 
  (compute_weekday day_advanced start_day = target_day) := 
  sorry

end treaty_of_paris_signed_on_thursday_l630_630089


namespace Lisa_quiz_goal_l630_630234

theorem Lisa_quiz_goal (total_quizzes : ℕ) (required_percentage : ℝ) (a_scored : ℕ) (completed_quizzes : ℕ) : 
  total_quizzes = 60 → 
  required_percentage = 0.75 → 
  a_scored = 30 → 
  completed_quizzes = 40 → 
  ∃ lower_than_a_quizzes : ℕ, lower_than_a_quizzes = 5 :=
by
  intros total_quizzes_eq req_percent_eq a_scored_eq completed_quizzes_eq
  sorry

end Lisa_quiz_goal_l630_630234


namespace max_re_part_l630_630083

open Complex

theorem max_re_part (z w : ℂ) (hz : abs z = 1) (hw : abs w = 1) (hzw : z * conj w + conj z * w = 2) : re (z + w) ≤ 2 :=
sorry

end max_re_part_l630_630083


namespace same_units_digit_pages_equiv_13_l630_630621

theorem same_units_digit_pages_equiv_13 :
  (finset.filter (λ x : ℕ, (x % 10 = (65 - x) % 10)) (finset.range 65)).card = 13 :=
by
  sorry

end same_units_digit_pages_equiv_13_l630_630621


namespace binomial_expansion_l630_630322

theorem binomial_expansion (a b : ℕ) (n : ℕ) : (∑ k in finset.range (n + 1), nat.choose n k = 256) → n = 8 ∧ (5 = (n + 1) / 2 + 1) :=
by
  intro h
  sorry

end binomial_expansion_l630_630322


namespace power_mod_l630_630160

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l630_630160


namespace parallelogram_area_correct_l630_630488

noncomputable def parallelogram_area (a b : ℝ) (α : ℝ) (h : a < b) : ℝ :=
  (4 * a^2 - b^2) / 4 * (Real.tan α)

theorem parallelogram_area_correct (a b α : ℝ) (h : a < b) :
  parallelogram_area a b α h = (4 * a^2 - b^2) / 4 * (Real.tan α) :=
by
  sorry

end parallelogram_area_correct_l630_630488


namespace remainder_of_3_pow_2023_mod_5_l630_630167

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l630_630167


namespace quadratic_no_real_roots_l630_630501

theorem quadratic_no_real_roots (a b : ℝ) (h : ∃ x : ℝ, x^2 + b * x + a = 0) : false :=
sorry

end quadratic_no_real_roots_l630_630501


namespace resized_and_shifted_function_value_l630_630069

theorem resized_and_shifted_function_value :
  ∀ (ω φ : ℝ), (0 < ω) → (- (π / 2) < φ) → (φ < π / 2) →
  (f(x) = sin(ω * x + φ)) →
  (resized_and_shifted_f(x) = sin(2 * ω * (x - π / 4) + φ)) →
  resized_and_shifted_f(x) = sin(x) →
  f(π / 6) = sqrt(3) / 2 :=
by
  sorry

end resized_and_shifted_function_value_l630_630069


namespace point_A_coordinates_l630_630761

/-- Given points P and Q lie on the circle with equation x^2 + y^2 = 5. Point A lies on the line x + y - 4 = 0. 
    Prove that the coordinates of point A=(1, 3) if the maximum value of ∠PAQ is 90 degrees. -/
theorem point_A_coordinates (P Q A : ℝ × ℝ) (hP : P.1 ^ 2 + P.2 ^ 2 = 5) (hQ : Q.1 ^ 2 + Q.2 ^ 2 = 5)
    (hA : A.1 + A.2 = 4) (h_angle : max_angle (angle P A Q) = 90) :
    A = (1, 3) ∨ A = (3, 1) :=
sorry

end point_A_coordinates_l630_630761


namespace given_problem_true_conditions_l630_630909

theorem given_problem_true_conditions :
  (¬ (∀ m > 0, ∀ x y : ℝ, x^2 + m*y^2 = 1 → (x = 0 ∧ y = 0) → m ≠ 1)) ∧
  (∀ a : ℝ, (a = 1 → ∀ x y : ℝ, (a*x + y - 1 = 0) ∧ (x + a*y - 2 = 0) → x = y) ∧ (a ≠ 1 ∧ a ≠ -1)) ∧ 
  (¬ (∀ m : ℝ, (∀ x : ℝ, x^3 + m*x is_mono → m > 0) ∧ (m ≥ 0))) ∧
  (∀ (p q : Prop), (p ≠ q) → ((p ∨ q) → ¬ (p ∧ q)) ∧ ((p ∧ q) → (p ∨ q))) := 
by sorry

end given_problem_true_conditions_l630_630909


namespace part_a_l630_630966

theorem part_a (n : ℕ) (h : n > 100) (G : SimpleGraph (Fin n)) :
  (¬ ∀ (A B : Finset (Fin n)), A ∪ B = Finset.univ →
   ∀ (v : Fin n), v ∈ A → (∑ w in G.neighborFinset v, if w ∈ A then 1 else 0) ≥ 
                  (∑ w in G.neighborFinset v, if w ∈ B then 1 else 0)) :=
sorry

end part_a_l630_630966


namespace remainder_of_3_pow_2023_mod_5_l630_630165

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l630_630165


namespace exponent_and_fraction_evaluation_l630_630670

theorem exponent_and_fraction_evaluation : (3 - real.pi)^0 + (1/2)^(-1) = 3 :=
by 
  -- proof steps skipped
  sorry

end exponent_and_fraction_evaluation_l630_630670


namespace max_distance_is_correct_l630_630039

open Complex

noncomputable def max_distance (z : ℂ) (h : abs z = 3) : ℝ :=
  27 * Real.sqrt ((2 - Real.sqrt 0.9)^2 + (6 - 3 * Real.sqrt 0.9)^2)

theorem max_distance_is_correct (z : ℂ) (h : abs z = 3) :
  abs ((2 + 6*I) * z^3 - z^4) ≤ max_distance z h :=
sorry

end max_distance_is_correct_l630_630039


namespace find_PB_l630_630030

noncomputable def PA : ℝ := 4
def PT (AB : ℝ) : ℝ := 2 * (AB - PA)

theorem find_PB (AB PB x : ℝ) :
  PA = 4 ∧
  PT AB = 2 * (AB - PA) ∧
  PB = PA + AB ∧
  (4 * PB = (PT AB)^2) ∧
  PB = x :=
  x = (17 + real.sqrt 33) / 2
:= sorry

end find_PB_l630_630030


namespace determine_roles_l630_630061

/-
We have three inhabitants K, M, R.
One of them is a truth-teller (tt), one is a liar (l), 
and one is a trickster (tr).
K states: "I am a trickster."
M states: "That is true."
R states: "I am not a trickster."
A truth-teller always tells the truth.
A liar always lies.
A trickster sometimes lies and sometimes tells the truth.
-/

inductive Role
| truth_teller | liar | trickster

open Role

def inhabitant_role (K M R : Role) : Prop :=
  ((K = liar) ∧ (M = trickster) ∧ (R = truth_teller)) ∧
  (K = trickster → K ≠ K) ∧
  (M = truth_teller → M = truth_teller) ∧
  (R = trickster → R ≠ R)

theorem determine_roles (K M R : Role) : inhabitant_role K M R :=
sorry

end determine_roles_l630_630061


namespace bob_number_l630_630644

theorem bob_number (a b : ℕ) (h₁ : a = 36) (h₂ : ∀ p, prime p → p ∣ a → p ∣ b) (h₃ : ∃ q, prime q ∧ q ∣ b ∧ ¬(q ∣ a)) :
  b = 30 :=
by {
  sorry
}

end bob_number_l630_630644


namespace g_1990_equals_1990_l630_630101

noncomputable def g : ℝ → ℝ := sorry

axiom g_le_self (x : ℝ) : g(x) ≤ x

axiom g_subadditive (x y : ℝ) : g(x + y) ≤ g(x) + g(y)

theorem g_1990_equals_1990 : g 1990 = 1990 :=
sorry

end g_1990_equals_1990_l630_630101


namespace number_of_proper_subsets_of_A_l630_630248

def A : Set ℕ := {1, 2}

theorem number_of_proper_subsets_of_A (h : Fintype.card A = 2) : (Fintype.card (Set ℕ) - 1) = 3 :=
by
  sorry

end number_of_proper_subsets_of_A_l630_630248


namespace pilot_fish_speed_when_moved_away_l630_630848

/-- Conditions -/
def keanu_speed : ℕ := 20
def shark_new_speed (k : ℕ) : ℕ := 2 * k
def pilot_fish_increase_speed (k s_new : ℕ) : ℕ := k + (s_new - k) / 2

/-- The problem statement to prove -/
theorem pilot_fish_speed_when_moved_away (k : ℕ) (s_new : ℕ) (p_new : ℕ) 
  (h1 : k = 20) 
  (h2 : s_new = shark_new_speed k) 
  (h3 : p_new = pilot_fish_increase_speed k s_new) : 
  p_new = 30 :=
by
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3]
  sorry

end pilot_fish_speed_when_moved_away_l630_630848


namespace remainder_3_pow_2023_mod_5_l630_630157

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l630_630157


namespace common_elements_count_l630_630031

variable (n m : ℕ)

def S : Set ℕ := { x | ∃ k : ℕ, x = 5 * k ∧ k > 0 ∧ k ≤ 3000 }
def T : Set ℕ := { x | ∃ k : ℕ, x = 10 * k ∧ k > 0 ∧ k ≤ 2500 }

theorem common_elements_count : (S ∩ T).card = 1500 :=
sorry

end common_elements_count_l630_630031


namespace daily_increase_in_weaving_l630_630578

open Nat

theorem daily_increase_in_weaving :
  let d := 16 / 29
  (30 : ℝ) * 10 + (30 * 29 / 2 : ℝ) * d = 600 :=
by
  let d := 16 / 29
  have h1 : (30 : ℝ) * 10 + (30 * 29 / 2 : ℝ) * d = 600 := sorry
  exact h1

end daily_increase_in_weaving_l630_630578


namespace sum_of_digits_n_l630_630998

def digits_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_n {n : ℕ} 
  (h1 : (n + 2)! + (n + 3)! = n! * 1320) 
  (h2 : n > 0) : 
  digits_sum n = 1 :=
by
  sorry

end sum_of_digits_n_l630_630998


namespace final_price_is_correct_l630_630922

def initialPrice : ℝ := 250
def decreaseWeek1 : ℝ := 12.5 / 100
def increaseWeek1 : ℝ := 30 / 100
def decreaseWeek2 : ℝ := 20 / 100
def increaseWeek3 : ℝ := 50 / 100
def conversionRate : ℝ := 3
def salesTax : ℝ := 5 / 100

def finalPriceInCurrencyB : ℝ := 
  let priceAfterWeek1 := (initialPrice * (1 - decreaseWeek1)) * (1 + increaseWeek1)
  let priceAfterWeek2 := priceAfterWeek1 * (1 - decreaseWeek2)
  let priceAfterWeek3 := priceAfterWeek2 * (1 + increaseWeek3)
  let priceInCurrencyB := priceAfterWeek3 * conversionRate
  let priceWithTax := priceInCurrencyB * (1 + salesTax)
  priceWithTax

theorem final_price_is_correct :
  finalPriceInCurrencyB = 1074.94 :=
by
  sorry

end final_price_is_correct_l630_630922


namespace power_function_inequality_l630_630784

theorem power_function_inequality {m x : ℝ} (hm : m^3 - m + 1 = 1) (hx : x < 1 / 2 ∧ x ≠ -1):
  (m = -1 ∨ m = 0 ∨ m = 1) →
  (∀ x, f x = (m^3 - m + 1) * x ^ (1 / 2 * (1 - 8 * m - m^2)) → f x = x ^ (-4)) →
  f (x + 1) > f (x - 2) :=
by
  intros
  sorry

end power_function_inequality_l630_630784


namespace area_of_triangle_BCD_l630_630000

noncomputable def area_triangle_ABC : ℝ := 36
noncomputable def AC : ℝ := 8
noncomputable def CD : ℝ := 32

theorem area_of_triangle_BCD (A B C D : Type) [geometry A B C D]
  (hABC : area ABC = area_triangle_ABC)
  (hAC : seg A C = AC)
  (hCD : seg C D = CD)
  (collinear_ACD : collinear A C D)
  (non_collinear_B_AD : ¬ collinear B A D) :
  area BCD = 144 := by
  sorry

end area_of_triangle_BCD_l630_630000


namespace least_positive_value_tan_inv_k_l630_630875

theorem least_positive_value_tan_inv_k 
  (a b : ℝ) 
  (x : ℝ) 
  (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) 
  : x = Real.arctan 1 := 
sorry

end least_positive_value_tan_inv_k_l630_630875


namespace angle_BAD_l630_630502

noncomputable theory
open_locale real
open_locale real_inner_product_space

-- Definitions for points and angles in the triangle setup
variables {A B C D : Type}

structure triangle (α : Type) := 
(AB BC AD DC : α) 
(ABC : ℝ) 
(ADC : ℝ)
(point_in_triangle : Type)

def isosceles (t : triangle ℝ) : Prop := 
t.AB = t.BC ∧ t.AD = t.DC ∧ t.ABC = 60 ∧ t.ADC = 120 

def interior_angle (t : triangle ℝ) : ℝ := 
if isosceles t then 30 else 0

-- Statement to prove
theorem angle_BAD (t : triangle ℝ) (h : isosceles t) : interior_angle t = 30 :=
sorry

end angle_BAD_l630_630502


namespace fraction_of_bread_slices_eaten_l630_630054

theorem fraction_of_bread_slices_eaten
    (total_slices : ℕ)
    (slices_used_for_sandwich : ℕ)
    (remaining_slices : ℕ)
    (slices_eaten_for_breakfast : ℕ)
    (h1 : total_slices = 12)
    (h2 : slices_used_for_sandwich = 2)
    (h3 : remaining_slices = 6)
    (h4 : total_slices - slices_used_for_sandwich - remaining_slices = slices_eaten_for_breakfast) :
    slices_eaten_for_breakfast / total_slices = 1 / 3 :=
sorry

end fraction_of_bread_slices_eaten_l630_630054


namespace total_days_2010_to_2015_l630_630798

theorem total_days_2010_to_2015 :
  let common_years := [2010, 2011, 2013, 2014, 2015],
      leap_years := [2012] in
  (list.length common_years * 365) + (list.length leap_years * 366) = 2191 :=
by
  sorry

end total_days_2010_to_2015_l630_630798


namespace balls_per_bag_l630_630209

theorem balls_per_bag (total_balls bags_used: Nat) (h1: total_balls = 36) (h2: bags_used = 9) : total_balls / bags_used = 4 := by
  sorry

end balls_per_bag_l630_630209


namespace probability_div_and_even_l630_630242

theorem probability_div_and_even (r k : ℤ) (R : set ℤ) (K : set ℕ)
  (hR : R = {x | -5 < x ∧ x < 8})
  (hK : K = {x | 0 ≤ x ∧ x < 10})
  (h_cond : r ∈ R ∧ k ∈ K) :
  let valid_pairs := { (x, y) | x ∈ R ∧ y ∈ K ∧ y ≠ 0 ∧ x % y = 0 ∧ (x + y) % 2 = 0 } in
  (↑(finset.card valid_pairs) / ↑(finset.card (R.prod K)) : ℚ) = 1 / 20 :=
by
  sorry

end probability_div_and_even_l630_630242


namespace CK_in_right_triangle_l630_630433

theorem CK_in_right_triangle
  (A B C K : Type*)
  [AC_eq_two : AC = 2]
  [angle_A_30 : ∠A = 30°]
  [circle_property : circle_with_diameter_AC_intersects_AB_at_K]
  : CK = 1 := sorry

end CK_in_right_triangle_l630_630433


namespace lottery_distribution_l630_630020

theorem lottery_distribution :
  let x := 155250 in
  let y := x / 1000 in
  let z := y * 100 in
  z = 15525 :=
by
  sorry

end lottery_distribution_l630_630020


namespace tank_capacity_is_24_l630_630607

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end tank_capacity_is_24_l630_630607


namespace pow_mod_remainder_l630_630172

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l630_630172


namespace problem_expression_value_l630_630559

theorem problem_expression_value :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 : ℤ) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 : ℤ) = 6608 :=
by
  sorry

end problem_expression_value_l630_630559


namespace last_three_digits_of_7_pow_210_l630_630705

theorem last_three_digits_of_7_pow_210 : (7^210) % 1000 = 599 := by
  sorry

end last_three_digits_of_7_pow_210_l630_630705


namespace element_symbol_is_Si_l630_630928

-- Define the element with atomic number 14
def atomic_number_14 : Type := {Element // Element.atomic_number = 14}

-- Prove that the symbol for the element with atomic number 14 is "Si"
theorem element_symbol_is_Si (e : atomic_number_14) : e.symbol = "Si" := 
sorry

end element_symbol_is_Si_l630_630928


namespace intensity_of_added_paint_l630_630451

theorem intensity_of_added_paint (x : ℝ) (original_intensity : ℝ) (fraction_replaced : ℝ) (final_intensity : ℝ) 
    (h1 : original_intensity = 0.15) 
    (h2 : fraction_replaced = 1.5) 
    (h3 : final_intensity = 0.30) :
    x = 0.40 := 
by 
  have h : 0.15 * (1 / (1 + 1.5)) + x * (1.5 / (1 + 1.5)) = 0.30,
  from sorry,
  sorry

end intensity_of_added_paint_l630_630451


namespace net_profit_loss_percent_zero_l630_630182

variable (cost_price1 cost_price2 profit_percent1 loss_percent2 : ℝ)

theorem net_profit_loss_percent_zero
  (h1 : cost_price1 = 1000)
  (h2 : cost_price2 = 1000)
  (h3 : profit_percent1 = 10)
  (h4 : loss_percent2 = 10) :
  net_profit_loss_percent cost_price1 cost_price2 profit_percent1 loss_percent2 = 0 := by
  sorry

end net_profit_loss_percent_zero_l630_630182


namespace art_gallery_ratio_l630_630647

theorem art_gallery_ratio (A : ℕ) (D : ℕ) (S_not_displayed : ℕ) (P_not_displayed : ℕ)
  (h1 : A = 2700)
  (h2 : 1 / 6 * D = D / 6)
  (h3 : P_not_displayed = S_not_displayed / 3)
  (h4 : S_not_displayed = 1200) :
  D / A = 11 / 27 := by
  sorry

end art_gallery_ratio_l630_630647


namespace sum_of_digits_base_8_999_l630_630176

theorem sum_of_digits_base_8_999 :
  let n := 999 
  let d0 := 7
  let d1 := 4
  let d2 := 7
  let d3 := 1
  8^3 * d3 + 8^2 * d2 + 8^1 * d1 + 8^0 * d0 = n ∧ d3 + d2 + d1 + d0 = 19 :=
by
  let n := 999
  let d0 := 7
  let d1 := 4
  let d2 := 7
  let d3 := 1
  calc
    8^3 * d3 + 8^2 * d2 + 8^1 * d1 + 8^0 * d0 : sorry
    _ = n : sorry
    d3 + d2 + d1 + d0 : sorry
    _ = 19 : sorry

end sum_of_digits_base_8_999_l630_630176


namespace polynomial_sum_l630_630408

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l630_630408


namespace intersect_points_correct_l630_630939

noncomputable def intersect_points : set (ℝ × ℝ) :=
  {p | (p.2 = 3 * p.1 ^ 2 - 12 * p.1 - 18) ∧ (p.2 = 2 * p.1 ^ 2 - 8 * p.1 + 4)}

theorem intersect_points_correct :
  intersect_points = { (2 + Real.sqrt 26, 48), (2 - Real.sqrt 26, 48) } :=
by
  sorry

end intersect_points_correct_l630_630939


namespace magnitude_equation_l630_630037

variables (z w : ℂ)

-- Defining the conditions
def condition1 : Prop := complex.abs z = 2
def condition2 : Prop := complex.abs w = 4
def condition3 : Prop := complex.abs (z + w) = 5

-- The proof statement
theorem magnitude_equation (h₁ : condition1 z) (h₂ : condition2 w) (h₃ : condition3 z w) : 
  complex.abs (1/z + 1/w) = 5/8 :=
by
  sorry

end magnitude_equation_l630_630037


namespace tens_digit_8_pow_2023_l630_630252

theorem tens_digit_8_pow_2023 : (8 ^ 2023 % 100) / 10 % 10 = 1 := 
sorry

end tens_digit_8_pow_2023_l630_630252


namespace total_number_of_workers_l630_630571

theorem total_number_of_workers (W : ℕ)
  (avg_total_salary : W > 0 → 6750)
  (avg_tech_salary : 7 > 0 → 12000)
  (avg_non_tech_salary : (W - 7) > 0 → 6000) :
  W = 56 :=
by
  sorry

end total_number_of_workers_l630_630571


namespace price_per_cake_l630_630071

def number_of_cakes_per_day := 4
def number_of_working_days_per_week := 5
def total_amount_collected := 640
def number_of_weeks := 4

theorem price_per_cake :
  let total_cakes_per_week := number_of_cakes_per_day * number_of_working_days_per_week
  let total_cakes_in_four_weeks := total_cakes_per_week * number_of_weeks
  let price_per_cake := total_amount_collected / total_cakes_in_four_weeks
  price_per_cake = 8 := by
sorry

end price_per_cake_l630_630071


namespace real_part_one_div_one_sub_eq_half_l630_630413

noncomputable def realPart {z : ℂ} (hz_nonreal : z ≠ conj z) (hz_norm : ∥z∥ = 1) : ℝ :=
  re (1 / (1 - z))

theorem real_part_one_div_one_sub_eq_half
  (z : ℂ) (hz_nonreal : z ≠ conj z) (hz_norm : ∥z∥ = 1) :
  realPart hz_nonreal hz_norm = 1 / 2 :=
sorry

end real_part_one_div_one_sub_eq_half_l630_630413


namespace constant_term_of_expansion_eq_negative_160_l630_630760

noncomputable def constant_term_in_expansion (a : ℝ) (h_pos : 0 < a) (h_sum : (a - 1)^6 = 1) : ℝ :=
  let k := 3 in
  let binom := Nat.choose 6 k in
  let term := binom * 2^(6 - k) * (-1)^k in
  term

theorem constant_term_of_expansion_eq_negative_160 (a : ℝ) (h_pos : 0 < a) (h_sum : (a - 1)^6 = 1) :
  constant_term_in_expansion a h_pos h_sum = -160 :=
by
  have ha : a = 2 := by
    sorry
  rw [ha] at *
  unfold constant_term_in_expansion
  norm_num
  sorry

end constant_term_of_expansion_eq_negative_160_l630_630760


namespace find_lambda_for_eccentricity_l630_630773

theorem find_lambda_for_eccentricity (λ : ℝ) :
  (\dfrac{x^{2}}{9}+ \dfrac{y^{2}}{\λ + 9} = 1) ∧ (∃ e : ℝ, e = 1 / 2 ∧ 
    ((λ > 0 → (e = (\sqrt{λ}) / (\sqrt{λ + 9})))
    ∧ (λ < 0 → (e = (\sqrt{-λ}) / (\sqrt{9}))))) → 
  (λ = 3 ∨ λ = -9 / 4) :=
by
  sorry

end find_lambda_for_eccentricity_l630_630773


namespace problem_statement_l630_630678

noncomputable def g : ℝ → ℝ → ℝ :=
  λ x y, if x + y ≤ 5 then (x * y - x - 3) / (3 * x) else (x * y - y + 1) / (-3 * y)

theorem problem_statement : g 1 4 + g 3 3 = -7 / 9 := by
  sorry

end problem_statement_l630_630678


namespace problem_2007_l630_630214

theorem problem_2007 :
  let N := 2007 ^ 2007 in
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ N →
  let prob_states := { nnat : ℕ | 1 ≤ nnat ∧ nnat ≤ N ∧ Nat.gcd N (nnat^3 - 36 * nnat) = 1 } in
  let a := 440 in
  let b := 669 in
  a + b = 1109
  := by
    sorry

end problem_2007_l630_630214


namespace smallest_palindromic_integer_l630_630274

-- Define what it means to be a palindrome in a given base.
def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let str := n.digits base in
  str = str.reverse

-- Main statement to be proved.
theorem smallest_palindromic_integer : ∃ n, n > 30 ∧ is_palindrome n 2 ∧ is_palindrome n 6 ∧ (∀ m, m > 30 ∧ is_palindrome m 2 ∧ is_palindrome m 6 → n ≤ m) :=
  by
  sorry

end smallest_palindromic_integer_l630_630274


namespace median_of_free_throws_l630_630589

def list_of_free_throws : List ℕ := [6, 18, 15, 14, 19, 12, 19, 15]

theorem median_of_free_throws : list_of_free_throws.median = 15 := 
by 
  sorry

end median_of_free_throws_l630_630589


namespace remainder_of_3_pow_2023_mod_5_l630_630166

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l630_630166


namespace sum_smallest_largest_terms_l630_630900

theorem sum_smallest_largest_terms :
  let seq := {n | 1 ≤ n ∧ n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3},
      smallest_term := Nat.find (by
        apply Exists.intro 8 _;
        repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num }),
      largest_term := Nat.find (by
        apply Exists.intro 188 _;
        repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num })
  in smallest_term + largest_term = 196 :=
by
  let seq := {n | 1 ≤ n ∧ n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3}
  let smallest_term := Nat.find (by
    apply Exists.intro 8 _
    repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num })
  let largest_term := Nat.find (by
    apply Exists.intro 188 _
    repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num })
  have : smallest_term = 8 := by sorry
  have : largest_term = 188 := by sorry
  rw [this, this]
  norm_num

end sum_smallest_largest_terms_l630_630900


namespace total_amount_received_is_correct_l630_630018

def johns_winnings : ℝ := 155250
def donation_per_student : ℝ := johns_winnings / 1000
def number_of_students : ℕ := 100
def total_amount_received : ℝ := donation_per_student * number_of_students

theorem total_amount_received_is_correct :
  total_amount_received = 15525 :=
by
  sorry

end total_amount_received_is_correct_l630_630018


namespace quadratic_function_condition_l630_630745

theorem quadratic_function_condition (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
  sorry

end quadratic_function_condition_l630_630745


namespace total_amount_received_is_correct_l630_630019

def johns_winnings : ℝ := 155250
def donation_per_student : ℝ := johns_winnings / 1000
def number_of_students : ℕ := 100
def total_amount_received : ℝ := donation_per_student * number_of_students

theorem total_amount_received_is_correct :
  total_amount_received = 15525 :=
by
  sorry

end total_amount_received_is_correct_l630_630019


namespace probability_ab_not_selected_together_l630_630216

open Set

noncomputable def total_combinations : ℕ := (Finset.univ.subset_card_eq 4).choose 2
noncomputable def ab_combinations : ℕ := (Finset.univ.subset_card_eq 2).choose 2

theorem probability_ab_not_selected_together :
  (total_combinations : ℚ) = 6 → 
  (ab_combinations : ℚ) = 1 →
  (p : ℚ) = 1 - (ab_combinations / total_combinations) →
  p = 5 / 6 := by sorry

end probability_ab_not_selected_together_l630_630216


namespace quadrilateral_contains_points_l630_630841

structure ConvexPolygon (α : Type*) :=
(vertices : list (α × α))  -- list of vertices representing the polygon

def is_convex {α : Type*} [linear_order α] (p : ConvexPolygon α) : Prop := sorry
def inside {α : Type*} (p : ConvexPolygon α) (point : α × α) : Prop := sorry

theorem quadrilateral_contains_points {α : Type*} [linear_order α] 
  (P : ConvexPolygon α) (P1 P2 : α × α)
  (h_convex : is_convex P) (hP1_inside : inside P P1) (hP2_inside : inside P P2) :
  ∃ (A B C D : α × α), (A, B, C, D ∈ P.vertices) ∧ 
                        -- A quadrilateral exists with vertices A, B, C, D
                        inside_quadrilateral (A, B, C, D) P1 ∧ 
                        inside_quadrilateral (A, B, C, D) P2 := sorry

-- Here inside_quadrilateral is a auxiliary function that checks if a point is inside a quadrilateral
def inside_quadrilateral {α : Type*} : 
  (α × α) × (α × α) × (α × α) × (α × α) → (α × α) → Prop := sorry

end quadrilateral_contains_points_l630_630841


namespace constant_term_in_expansion_l630_630703

theorem constant_term_in_expansion:
  (finset.sum (finset.range 10)
   (λ r, (if 9 - (3 * r) / 2 = 0 then -1^r * (nat.choose 9 r) * (2 ^ (9 - r)) else 0)) = 672 := sorry

end constant_term_in_expansion_l630_630703


namespace tangent_line_y_intercept_l630_630929

noncomputable def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

theorem tangent_line_y_intercept : 
  let f'(x : ℝ) : ℝ := deriv f x in
  let tangent_line (x y : ℝ) : Prop := y - f 1 = f' 1 * (x - 1) in
  ∃ y : ℝ, tangent_line 0 y ∧ y = 3 :=
by
  sorry  -- Proof will go here.

end tangent_line_y_intercept_l630_630929


namespace tsar_expense_change_l630_630120

theorem tsar_expense_change (n : ℕ) (h : n > 0) : 
  let old_expense := 1000 in
  let new_expense := ((n / 2) * (1.5 * (1000 / n))) in
  ((new_expense - old_expense) / old_expense) * 100 = -25 :=
by
  sorry

end tsar_expense_change_l630_630120


namespace cost_of_ice_cream_carton_l630_630085

-- Define the conditions
variables (x : ℕ) -- cost of each carton of ice cream
variables (num_ice_cream num_yoghurt cost_yoghurt : ℕ)
variables (extra_spent : ℕ)

-- conditions 
def num_ice_cream := 19
def num_yoghurt := 4
def cost_yoghurt := 1
def extra_spent := 129

-- Define the statement to be proved
theorem cost_of_ice_cream_carton : x = 7 :=
by
  let total_cost_yoghurt := num_yoghurt * cost_yoghurt
  let total_cost_ice_cream := num_ice_cream * x
  have h1 : total_cost_ice_cream = total_cost_yoghurt + extra_spent := sorry
  have h2 : total_cost_ice_cream = 19 * x := sorry
  have h3 : total_cost_yoghurt = 4 * 1 := sorry
  have h4 : h1 → 19 * x = 4 + 129 := sorry
  have h5 : 19 * x = 133 := sorry
  have h6 : x = 133 / 19 := sorry
  exact sorry

end cost_of_ice_cream_carton_l630_630085


namespace ratio_of_x_to_y_l630_630682

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
sorry

end ratio_of_x_to_y_l630_630682


namespace number_of_valid_N_count_valid_N_is_seven_l630_630730

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l630_630730


namespace range_of_a_l630_630301

theorem range_of_a 
  (x y a : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy : x + y + 3 = x * y) 
  (h_a : ∀ x y : ℝ, (x + y)^2 - a * (x + y) + 1 ≥ 0) :
  a ≤ 37 / 6 := 
sorry

end range_of_a_l630_630301


namespace sum_series_eq_four_l630_630671

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n = 0 then 0 else (3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_series_eq_four :
  series_sum = 4 :=
by
  sorry

end sum_series_eq_four_l630_630671


namespace range_of_a_l630_630407

theorem range_of_a (a : ℝ) :
  (∃(p : Prop) (q : Prop), 
    (∀ x : ℝ, a^x > 1 ↔ x < 0) ∨ 
    (∀ x : ℝ, ∃ y : ℝ, y = sqrt (a*x^2 - x + a)) ∧ 
    ¬((∀ x : ℝ, a^x > 1 ↔ x < 0) ∧ 
      (∀ x : ℝ, ∃ y : ℝ, y = sqrt (a*x^2 - x + a)))) ↔ 
  a ∈ (Set.Ioo 0 (1/2) ∪ Set.Ici 1) :=
sorry

end range_of_a_l630_630407


namespace smallest_five_digit_multiple_of_53_l630_630555

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l630_630555


namespace trapezoid_median_l630_630640

-- Define the area of a triangle
def TriangleArea (b h : ℝ) : ℝ := (1 / 2) * b * h

-- Define the area of a trapezoid using its median
def TrapezoidArea (m h : ℝ) : ℝ := m * h

-- Lean statement to prove the problem
theorem trapezoid_median (b h m : ℝ) (hb : b = 18) (same_height : TriangleArea b h = TrapezoidArea m h) :
  m = 9 :=
by
  -- Simplify the given conditions
  rw [TriangleArea, TrapezoidArea, hb] at same_height
  -- Rewrite the equation to show the equality of areas
  have equality : (1 / 2) * 18 * h = m * h := same_height
  -- Assume the height is not zero to divide both sides by h
  have h_ne_zero : h ≠ 0 := sorry -- this should be justified properly outside of this theorem
  -- Divide both sides by h
  have median_equation : (1 / 2) * 18 = m := eq_of_mul_eq_mul_right h_ne_zero equality
  -- Simplify the equation to get m
  rw [← mul_assoc, div_mul_cancel _ (show (2 : ℝ) ≠ 0 by norm_num), mul_one] at median_equation
  exact median_equation

end trapezoid_median_l630_630640


namespace sum_equivalence_l630_630856

theorem sum_equivalence :
  ∑ n in Finset.range (5000 + 1) \ {0}, (1 / Real.sqrt (n + Real.sqrt (n^2 - 1))) =
  (Real.sqrt 5001 + Real.sqrt 5000 - 1) / Real.sqrt 2 :=
by
  -- Proof goes here.
  sorry

end sum_equivalence_l630_630856


namespace one_heads_one_tails_probability_l630_630958

def outcomes : List (String × String) := [("H", "H"), ("H", "T"), ("T", "H"), ("T", "T")]

def favorable_outcomes (outcome : String × String) : Bool :=
  (outcome = ("H", "T")) ∨ (outcome = ("T", "H"))

def probability_of_favorable_event : ℚ :=
  ⟨2, 4⟩  -- 2 favorable outcomes out of 4 possible outcomes (as a rational number simplified to 1/2)

theorem one_heads_one_tails_probability :
  ∃ (p : ℚ), p = probability_of_favorable_event :=
begin
  use ⟨1, 2⟩,
  sorry
end

end one_heads_one_tails_probability_l630_630958


namespace equation_of_chord_l630_630751

-- Define the ellipse equation and point P
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def P : ℝ × ℝ := (3, 2)
def is_midpoint (A B P : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2
def on_chord (A B : ℝ × ℝ) (x y : ℝ) : Prop := (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)

-- Lean Statement
theorem equation_of_chord :
  ∀ A B : ℝ × ℝ,
    ellipse_eq A.1 A.2 →
    ellipse_eq B.1 B.2 →
    is_midpoint A B P →
    ∀ x y : ℝ,
      on_chord A B x y →
      2 * x + 3 * y = 12 :=
by
  sorry

end equation_of_chord_l630_630751


namespace train_speed_is_45_kmph_l630_630219

def train_length : ℝ := 360 -- in meters
def platform_length : ℝ := 130 -- in meters
def passing_time : ℝ := 39.2 -- in seconds
def distance (train_length : ℝ) (platform_length : ℝ) : ℝ := train_length + platform_length -- in meters
def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time -- in m/s
def to_km_per_hr (speed_m_per_s : ℝ) : ℝ := speed_m_per_s * 3.6 -- conversion factor

theorem train_speed_is_45_kmph : 
  to_km_per_hr (speed (distance train_length platform_length) passing_time) = 45 :=
by
  sorry

end train_speed_is_45_kmph_l630_630219


namespace decimal_to_binary_123_l630_630512

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630512


namespace probability_red_side_l630_630980

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end probability_red_side_l630_630980


namespace greatest_possible_perimeter_l630_630824

theorem greatest_possible_perimeter :
  ∃ (x : ℤ), x ≥ 4 ∧ x ≤ 5 ∧ (x + 4 * x + 18 = 43 ∧
    ∀ (y : ℤ), y ≥ 4 ∧ y ≤ 5 → y + 4 * y + 18 ≤ 43) :=
by
  sorry

end greatest_possible_perimeter_l630_630824


namespace propositions_in_space_l630_630372

-- Definition: Perpendicular lines and planes
def perpendicular_to (line1 line2 : Line) : Prop :=
  ∃ (p : Point), p ∈ line1 ∧ p ∈ line2 ∧ 
    ∃(n : Vector), is_normal_vector_to n line1 ∧ is_normal_vector_to n line2

def perpendicular_to_plane (line : Line) (plane : Plane) : Prop :=
  ∃ (p : Point), p ∈ line ∧ p ∈ plane ∧ 
    ∃(n : Vector), is_normal_vector_to_plane n plane ∧ is_normal_vector_to_line n line

def perpendicular_planes (plane1 plane2 : Plane) : Prop :=
  ∃ (n1 : Vector), is_normal_vector_to_plane n1 plane1 ∧ 
    ∃ (n2 : Vector), is_normal_vector_to_plane n2 plane2 ∧
    n1 ≠ n2 ∧ n1 ∥ n2

def parallel (plane1 plane2 : Plane) : Prop :=
  ∀ (p : Point), p ∈ plane1 → p ∈ plane2

def parallel_lines (line1 line2 : Line) : Prop :=
  ∀ (p1 p2 : Point), p1 ∈ line1 → p1 ∈ line2 ∧ p2 ∈ line1 → p2 ∈ line2

axiom two_lines_perpendicular_to_same_line_parallel (line1 line2 commonLine : Line) 
  (hl1 : perpendicular_to line1 commonLine) (hl2 : perpendicular_to line2 commonLine) :
  parallel_lines line1 line2

theorem propositions_in_space (line1 line2 : Line) (plane1 plane2 : Plane) (commonLine : Line) (commonPlane : Plane) :
  (perpendicular_to_plane line1 commonPlane ∧ perpendicular_to_plane line2 commonPlane → parallel_lines line1 line2) ∧
  (perpendicular_planes plane1 commonLine ∧ perpendicular_planes plane2 commonLine → parallel plane1 plane2) ∧
  (perpendicular_planes plane1 commonPlane ∧ perpendicular_planes plane2 commonPlane → (parallel plane1 plane2 ∨ ∃ (p : Point), p ∈ plane1 ∧ p ∈ plane2)) ∧
  (perpendicular_to_plane line commonPlane ∧ perpendicular_planes plane1 commonLine → (parallel plane1 ⟨line⟩ ∨ ∃ (p : Point), p ∈ line ∧ p ∈ plane1)) ∧
  (perpendicular_planes plane commonLine ∧ perpendicular_to_plane line commonPlane → (parallel plane commonLine ∨ ∃ (p : Point), p ∈ line ∧ p ∈ commonLine)) :=
sorry

end propositions_in_space_l630_630372


namespace exists_integer_n_l630_630886

theorem exists_integer_n (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℤ, (n + 1981^k)^(1/2 : ℝ) + (n : ℝ)^(1/2 : ℝ) = (1982^(1/2 : ℝ) + 1) ^ k :=
sorry

end exists_integer_n_l630_630886


namespace num_multiples_of_15_l630_630339

theorem num_multiples_of_15 (a b n : ℕ) (h₁ : a < b) (h₂ : n = 15) (h₃ : 15 ∣ n) : 
  ∃ k : ℕ, k = 8 ∧ (∀ m : ℕ, (m * 15) ∈ set.Ico 8 124 ↔ m ∈ set.Icc 1 8) :=
by
  sorry

end num_multiples_of_15_l630_630339


namespace negation_of_universal_statement_l630_630916

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_of_universal_statement_l630_630916


namespace general_term_sequence_l630_630582

theorem general_term_sequence (a : ℕ → ℝ) (h₁ : a 1 = 1) (hn : ∀ (n : ℕ), a (n + 1) = (10 + 4 * a n) / (1 + a n)) :
  ∀ n : ℕ, a n = 5 - 7 / (1 + (3 / 4) * (-6)^(n - 1)) := 
sorry

end general_term_sequence_l630_630582


namespace sum_of_interior_angles_divisible_by_360_l630_630604

theorem sum_of_interior_angles_divisible_by_360
  (n : ℕ)
  (h : n > 0) :
  ∃ k : ℤ, ((2 * n - 2) * 180) = 360 * k :=
by
  sorry

end sum_of_interior_angles_divisible_by_360_l630_630604


namespace product_of_A_and_D_l630_630231

noncomputable theory

-- Define the set of numbers from 1 to 16.
def numbers : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

-- Define A, B, C, and D as choices from the number set, picked sequentially in a clockwise manner.
variables (A B C D : ℕ)

-- Conditions:
-- A and D picked even numbers.
axiom hA : A ∈ numbers ∧ even A
axiom hD : D ∈ numbers ∧ even D
-- The valid numbers chosen by A and D are 10 and 12.
axiom A_is_10 : A = 10
axiom D_is_12 : D = 12

-- The theorem we want to prove.
theorem product_of_A_and_D : A * D = 120 :=
by {
  rw [A_is_10, D_is_12],
  exact mul_eq_120 10 12
}

end product_of_A_and_D_l630_630231


namespace max_volume_of_right_triangular_prism_is_R_cubed_l630_630628

noncomputable def max_volume_of_right_triangular_prism (R : ℝ) : ℝ := R^3

theorem max_volume_of_right_triangular_prism_is_R_cubed (R : ℝ) :
  (∀ (a : ℝ), 
    let volume := (1 / 2) * sqrt ((3 * R^2 - a^2) * a^4) in
    volume ≤ R^3) := sorry

end max_volume_of_right_triangular_prism_is_R_cubed_l630_630628


namespace area_of_triangle_OPA_l630_630766

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)).abs

theorem area_of_triangle_OPA :
  ∀ (m : ℝ), (y = (-2) * x + 4) ∧ (1, m) lies on the line ∧ 
  (2, 0) lies on the line → 
  triangle_area (0, 0) (1, 2) (2, 0) = 2 :=
by
  sorry

end area_of_triangle_OPA_l630_630766


namespace rhombus_minimum_rotation_180_degrees_l630_630215

-- Define a rhombus with an angle and rotational properties
structure Rhombus (α : Type) [metric_space α] :=
  (side : α)
  (angle : ℝ)
  (rotational_symmetry : ∀ θ, ∃ k : ℤ, θ = k * 2 * π / 4)

-- Assume a particular rhombus with a given angle
def given_rhombus := Rhombus ℝ
axiom given_rhombus_angle : given_rhombus.angle = π / 3 -- 60 degrees

-- Define the minimum degree of rotation needed to coincide with its original position
def minimum_rotation_coincide (R : Rhombus ℝ) (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * π

-- The proof problem statement
theorem rhombus_minimum_rotation_180_degrees :
  ∀ R : Rhombus ℝ, R.angle = π / 3 → minimum_rotation_coincide R π :=
begin
  sorry
end

end rhombus_minimum_rotation_180_degrees_l630_630215


namespace height_of_building_l630_630639

theorem height_of_building 
  (height_tree : ℕ)
  (shadow_tree : ℕ)
  (shadow_building : ℕ)
  (ratio_proportional : (shadow_tree : ℚ) / height_tree = 6 / 5)
  (similar_angles : Prop) :
  let h := shadow_building * 6 / 5 in
  h.round = 86 :=
by
  -- problem statement, proof omitted
  sorry

end height_of_building_l630_630639


namespace integer_points_on_circle_l630_630281

noncomputable def circle_center : ℝ × ℝ := (3, -3)
noncomputable def circle_radius : ℝ := 8

def is_on_or_inside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in (x - 3) ^ 2 + (y + 3) ^ 2 ≤ 64

theorem integer_points_on_circle :
  { x : ℤ | is_on_or_inside_circle (x, x + 2) }.to_finset.card = 2 :=
sorry

end integer_points_on_circle_l630_630281


namespace angle_theta_of_rectangle_zigzag_l630_630941

theorem angle_theta_of_rectangle_zigzag (ACB : ℝ) (FEG : ℝ) (DCE : ℝ) (DEC : ℝ) (ACB_eq : ACB = 10) (FEG_eq : FEG = 26) (DCE_eq : DCE = 14) (DEC_eq : DEC = 33) : 
  let θ := 180 - 86 - 83 in θ = 11 := 
by sorry

end angle_theta_of_rectangle_zigzag_l630_630941


namespace brownies_left_l630_630500

noncomputable theory

-- Definitions of conditions
def total_initial_brownies : ℕ := 24
def brownies_Tina_eats : ℕ := 1 * 5 + 1 * 5 -- Tina's consumption
def brownies_husband_eats : ℕ := 1 * 5       -- Husband's consumption
def brownies_guests_share : ℕ := 4           -- Shared with guests

-- Total number of brownies eaten
def total_brownies_eaten : ℕ := brownies_Tina_eats + brownies_husband_eats + brownies_guests_share

-- Theorem to prove the number of brownies left
theorem brownies_left : (total_initial_brownies - total_brownies_eaten) = 5 := by
  sorry

end brownies_left_l630_630500


namespace problem_sin_cos_alpha_l630_630744

theorem problem_sin_cos_alpha (α : ℝ) (h1 : sin (π - α) - cos (π + α) = (sqrt 2) / 3) (h2 : π/2 < α ∧ α < π) : 
  (sin α - cos α = 2) ∧ (sin^3 (2 * π - α) + cos^3 (2 * π - α) = 0) := by
  sorry

end problem_sin_cos_alpha_l630_630744


namespace total_games_won_l630_630365

theorem total_games_won 
  (bulls_games : ℕ) (heat_games : ℕ) (knicks_games : ℕ)
  (bulls_condition : bulls_games = 70)
  (heat_condition : heat_games = bulls_games + 5)
  (knicks_condition : knicks_games = 2 * heat_games) :
  bulls_games + heat_games + knicks_games = 295 :=
by
  sorry

end total_games_won_l630_630365


namespace petting_zoo_animal_count_l630_630050

theorem petting_zoo_animal_count (double_counted_sheep goats chickens : Nat) 
  (missed_pigs ducks rabbits : Nat) (hiding_percentage : Real) 
  (mary_count : Nat) : 
  double_counted_sheep = 7 →
  goats = 4 →
  chickens = 5 →
  missed_pigs = 3 →
  ducks = 2 →
  rabbits = 6 →
  hiding_percentage = 0.10 →
  mary_count = 90 →
  ∃ (x : Nat), round (85 / 0.9) = x ∧ x = 94 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 
  sorry

end petting_zoo_animal_count_l630_630050


namespace base_2_representation_of_123_l630_630550

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630550


namespace ninth_term_correct_l630_630492

noncomputable def ninth_term_of_arithmetic_sequence
  (third_term fifteenth_term : ℚ) : ℚ :=
  (third_term + fifteenth_term) / 2

theorem ninth_term_correct :
  let third_term := (3 : ℚ) / 8
      fifteenth_term := (7 : ℚ) / 9
  in ninth_term_of_arithmetic_sequence third_term fifteenth_term = (83 : ℚ) / 144 :=
by
let third_term := (3 : ℚ) / 8
let fifteenth_term := (7 : ℚ) / 9
simp [ninth_term_of_arithmetic_sequence, third_term, fifteenth_term]
sorry

end ninth_term_correct_l630_630492


namespace num_positive_int_values_for_expression_is_7_l630_630722

theorem num_positive_int_values_for_expression_is_7 :
  {N : ℕ // 0 < N ∧ ∃ k : ℕ, 48 = k * (N + 3)}.card = 7 := 
sorry

end num_positive_int_values_for_expression_is_7_l630_630722


namespace probability_two_win_one_lose_l630_630223

noncomputable def p_A : ℚ := 1 / 5
noncomputable def p_B : ℚ := 3 / 8
noncomputable def p_C : ℚ := 2 / 7

noncomputable def P_two_win_one_lose : ℚ :=
  p_A * p_B * (1 - p_C) +
  p_A * p_C * (1 - p_B) +
  p_B * p_C * (1 - p_A)

theorem probability_two_win_one_lose :
  P_two_win_one_lose = 49 / 280 :=
by
  sorry

end probability_two_win_one_lose_l630_630223


namespace area_of_triangle_BCD_l630_630001

noncomputable def area_triangle_ABC : ℝ := 36
noncomputable def AC : ℝ := 8
noncomputable def CD : ℝ := 32

theorem area_of_triangle_BCD (A B C D : Type) [geometry A B C D]
  (hABC : area ABC = area_triangle_ABC)
  (hAC : seg A C = AC)
  (hCD : seg C D = CD)
  (collinear_ACD : collinear A C D)
  (non_collinear_B_AD : ¬ collinear B A D) :
  area BCD = 144 := by
  sorry

end area_of_triangle_BCD_l630_630001


namespace evaluate_expression_at_1_l630_630891

theorem evaluate_expression_at_1 :
  (let x := 1 in
   (2*x/(x+2) - x/(x-2) + 4*x/(x^2-4))) = 1/3 :=
by
  let x := 1
  sorry

end evaluate_expression_at_1_l630_630891


namespace power_zero_addition_l630_630146

theorem power_zero_addition : (5 ^ (-2)) ^ 0 + (5 ^ 0) ^ 3 = 2 := by
  have h1 : ∀ a : ℝ, a ≠ 0 → a ^ 0 = 1 := by sorry
  have h2 : 5 ^ 0 = 1 := by apply h1; norm_num
  calc
    (5 ^ (-2)) ^ 0 + (5 ^ 0) ^ 3 = 1 + (5 ^ 0) ^ 3     : by rw [h1 (5 ^ (-2))]; norm_num
                              ... = 1 + 1 ^ 3         : by rw [h2]
                              ... = 1 + 1             : by norm_num
                              ... = 2                 : by norm_num

end power_zero_addition_l630_630146


namespace remainderDegrees_l630_630956

def divisor : Polynomial ℝ := 4 * X^5 - 7 * X^3 + 3 * X - 8

theorem remainderDegrees :
  ∀ (p : Polynomial ℝ) (q r : Polynomial ℝ), p = q * divisor + r →
  r.degree < divisor.degree :=
sorry

end remainderDegrees_l630_630956


namespace no_swap_positions_l630_630230

-- 1. Definition of grid points in terms of complex numbers
def lattice_points : set ℂ :=
  { z | ∃ m n : ℤ, z = m + n * (1 + complex.i * real.sqrt 3 / 2) }

-- 2. Definition of adjacency between lattice points
def is_adjacent (a b : ℂ) : Prop :=
  complex.abs (a - b) = 1

-- 3. Frogs positions
variables (A B : ℂ)

-- 4. Game rules
def jump_A (A B : ℂ) : set (ℂ × ℂ) :=
  { (A', B) | is_adjacent A A' }

def jump_B (A B : ℂ) : set (ℂ × ℂ) :=
  { (A, B') | (∃ d, is_adjacent A B ∧ B' = B + d) ∨ 
             (∃ d, is_adjacent A B ∧ B' = B - 2 * d) }

def rule_I (A B A' B' : ℂ) : Prop :=
  (A', B') ∈ jump_A A B ∧ (A', B') ∈ jump_B A' B

def rule_II (A B A' B' : ℂ) : Prop :=
  is_adjacent A B ∧ 
  ((A', B') ∈ jump_A A B ∧ (A', B') ∈ jump_B A' B ∨
  ∃ A'', is_adjacent A B ∧ is_adjacent A'' B ∧ A' = A'' ∧ B' = B)

-- 5. Starting condition for frogs
axiom adjacent_start : is_adjacent A B

-- 6. Goal: Proving impossibility of reaching swapped positions
theorem no_swap_positions : 
  ¬ ∃ n : ℕ, ∃ f : fin n → ℂ × ℂ, 
  (∀ i, rule_I (f i).1 (f i).2 (f (i + 1)).1 (f (i + 1)).2 ∨ rule_II (f i).1 (f i).2 (f (i + 1)).1 (f (i + 1)).2) ∧
  (f 0).1 = A ∧ (f 0).2 = B ∧ (f n).1 = B ∧ (f n).2 = A := 
sorry

end no_swap_positions_l630_630230


namespace decimal_to_binary_123_l630_630509

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630509


namespace population_in_2060_l630_630256

noncomputable def population (year : ℕ) : ℕ :=
  if h : (year - 2000) % 20 = 0 then
    250 * 2 ^ ((year - 2000) / 20)
  else
    0 -- This handles non-multiples of 20 cases, which are irrelevant here

theorem population_in_2060 : population 2060 = 2000 := by
  sorry

end population_in_2060_l630_630256


namespace graph_fixed_point_l630_630463

theorem graph_fixed_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  (∃ x y, x = -3 ∧ y = -1 ∧ y = a^(x+3) - 2) :=
begin
  use [-3, -1],
  split,
  { refl },
  split,
  { refl },
  { rw [add_comm, add_neg_self, pow_zero, sub_self],
    refl },
end

end graph_fixed_point_l630_630463


namespace total_distance_biked_l630_630645

theorem total_distance_biked :
  let monday_distance := 12
  let tuesday_distance := 2 * monday_distance - 3
  let wednesday_distance := 2 * 11
  let thursday_distance := wednesday_distance + 2
  let friday_distance := thursday_distance + 2
  let saturday_distance := friday_distance + 2
  let sunday_distance := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance + saturday_distance + sunday_distance = 151 := 
by
  sorry

end total_distance_biked_l630_630645


namespace cos_XPY_l630_630825

noncomputable def point := ℝ × ℝ

def distance (p q : point) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def cos_angle (u v w : point) : ℝ :=
  let a := distance u v
  let b := distance v w
  let c := distance w u
  (a^2 + c^2 - b^2) / (2 * a * c)

def P : point := (5, 5 * real.sqrt 3)
def Q : point := (0, 0)
def R : point := (10, 0)

def X : point := (4, 0)
def Y : point := (8, 0)

theorem cos_XPY :
  cos_angle X P Y = real.sqrt 3 / 2 :=
sorry

end cos_XPY_l630_630825


namespace choose_k_from_n_l630_630826

noncomputable def factorial : ℕ → ℕ
| 0 := 1
| (n+1) := (n+1) * factorial n

def binom_coeff (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem choose_k_from_n (n k : ℕ) : binom_coeff n k = factorial n / (factorial k * factorial (n - k)) :=
by
  sorry

end choose_k_from_n_l630_630826


namespace inequality_implies_k_less_one_l630_630713

theorem inequality_implies_k_less_one (k : ℝ) (h : ∀ x ∈ Ioo 0 (π / 2), k * sin x * cos x < x) : k < 1 :=
sorry

end inequality_implies_k_less_one_l630_630713


namespace minimum_value_of_fraction_l630_630759

variable {a b : ℝ}

theorem minimum_value_of_fraction (h1 : a > b) (h2 : a * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ ∀ x > b, a * x = 1 -> 
  (x - b + 2 / (x - b) ≥ c) :=
by
  sorry

end minimum_value_of_fraction_l630_630759


namespace convert_108_kmph_to_mps_l630_630261

-- Definitions and assumptions
def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * (1000 / 3600)

-- Theorem statement
theorem convert_108_kmph_to_mps : kmph_to_mps 108 = 30 := 
by
  sorry

end convert_108_kmph_to_mps_l630_630261


namespace hotel_towels_l630_630612

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l630_630612


namespace union_A_B_complement_intersection_Ac_B_nonempty_intersection_A_C_l630_630306

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
sorry

theorem complement_intersection_Ac_B :
  (Aᶜ ∧ B) = ({x | 8 ≤ x ∧ x < 10} ∪ {x | 2 < x ∧ x < 4}) :=
sorry

theorem nonempty_intersection_A_C (a : ℝ) (h: (A ∧ C a) ≠ ∅) : 4 < a :=
sorry

end union_A_B_complement_intersection_Ac_B_nonempty_intersection_A_C_l630_630306


namespace num_positive_int_values_l630_630716

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l630_630716


namespace relation_between_gender_and_big_sciences_probability_at_least_one_female_l630_630470

variable (students : Nat)
variable (male_students female_students : Nat)
variable (big_sciences non_big_sciences : Nat)
variable (male_big_sciences male_non_big_sciences : Nat)
variable (female_big_sciences female_non_big_sciences : Nat)
variable (n : ℝ)

noncomputable def K_square (a b c d : ℕ) : ℝ :=
  let n := (a + b + c + d) in
  (n * ((a * d - b * c)^2)) / (↑a + ↑b) / (↑c + ↑d) / (↑a + ↑c) / (↑b + ↑d)

theorem relation_between_gender_and_big_sciences (
  h1 : students = 100
  h2 : male_students = 55
  h3 : female_students = 45
  h4 : big_sciences = 60
  h5 : non_big_sciences = 40
  h6 : male_big_sciences = 40
  h7 : male_non_big_sciences = 15
  h8 : female_big_sciences = 20
  h9 : female_non_big_sciences = 25
  h10: K_square 40 15 20 25 = 8.249
) : true :=
sorry

noncomputable def C (n k : ℕ) : ℕ :=
Nat.choose n k

theorem probability_at_least_one_female (
  h1: (C 6 2) = 15
  h2: (C 2 1 * C 4 1 + C 2 2) = 9
) : (9 / 15) = 3 / 5 :=
sorry

end relation_between_gender_and_big_sciences_probability_at_least_one_female_l630_630470


namespace power_mod_l630_630163

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l630_630163


namespace probability_no_shaded_square_l630_630977

theorem probability_no_shaded_square : 
  let n : ℕ := 502 * 1004
  let m : ℕ := 502^2
  let total_rectangles := 3 * n
  let rectangles_with_shaded := 3 * m
  let probability_includes_shaded := rectangles_with_shaded / total_rectangles
  1 - probability_includes_shaded = (1 : ℚ) / 2 := 
by 
  sorry

end probability_no_shaded_square_l630_630977


namespace students_taking_chorus_l630_630816

theorem students_taking_chorus
    (n : ℕ) (B : ℕ) (C_inter_B : ℕ) (not_either : ℕ)
    (total_students : n = 50)
    (B_students : B = 26)
    (both_chorus_band : C_inter_B = 2)
    (not_either_students : not_either = 8) :
    let C_union_B := n - not_either in
    let C := C_union_B + C_inter_B - B in
    C = 18 :=
by {
    sorry
}

end students_taking_chorus_l630_630816


namespace cube_sum_decomposition_l630_630100

theorem cube_sum_decomposition : 
  (∃ (a b c d e : ℤ), (1000 * x^3 + 27) = (a * x + b) * (c * x^2 + d * x + e) ∧ (a + b + c + d + e = 92)) :=
by
  sorry

end cube_sum_decomposition_l630_630100


namespace edge_BB1_division_ratio_l630_630499

-- Definitions of the points and midpoints in the parallelepiped
variables (A B C D A1 B1 C1 D1 : Point)
variables (M : Point) (N : Point)
variables (DB1 : Line)

-- Midpoints definitions
def midpoint_AD (A D M : Point) : Prop := M = (A + D) / 2
def midpoint_CC1 (C C1 N : Point) : Prop := N = (C + C1) / 2

-- The plane is parallel to the diagonal DB1 and passes through M and N
def plane_MN_parallel_DB1 (M N : Point) (DB1 : Line) (plane_MN : Plane) : Prop :=
  plane_MN.contains M ∧ plane_MN.contains N ∧ plane_MN.parallel DB1

-- Proof goal: the plane divides the edge BB1 in ratio 5:1
theorem edge_BB1_division_ratio :
  midpoint_AD A D M →
  midpoint_CC1 C C1 N →
  plane_MN_parallel_DB1 M N DB1 (some_plane : Plane) →
  divides_in_ratio BB B1 (some_plane.intersect BB1) (Point.ratio 5 1) :=
sorry

end edge_BB1_division_ratio_l630_630499


namespace largest_n_exists_unique_k_l630_630945

theorem largest_n_exists_unique_k (n k : ℕ) :
  (∃! k, (8 : ℚ) / 15 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 7 / 13) →
  n ≤ 112 :=
sorry

end largest_n_exists_unique_k_l630_630945


namespace maximize_profit_of_craft_store_l630_630178

-- Let's define the conditions
def cost_price : ℕ := 50
def profit_per_item : ℕ := 45
def marked_up_items : ℕ := 40
def sold_per_day : ℕ := 100
def additional_sale_rate : ℕ := 4

-- Define the proof problem
theorem maximize_profit_of_craft_store :
  ∃ (x y a w : ℕ), 
    x = 180 ∧ 
    y = 225 ∧ 
    a = 10 ∧ 
    w = 4900 ∧ 
    50 * x = 40 * (x + profit_per_item) ∧
    w = (profit_per_item - a) * (sold_per_day + additional_sale_rate * a) :=
begin
  sorry
end

end maximize_profit_of_craft_store_l630_630178


namespace reciprocal_of_neg_2023_l630_630482

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l630_630482


namespace mean_of_combined_sets_l630_630914

theorem mean_of_combined_sets (a b : Fin 8 → ℝ) (c d : Fin 10 → ℝ) (h1 : (∑ i, a i / 8) = 17) (h2 : (∑ i, d i / 10) = 23) :
  let combined_mean := (∑ i, a i + ∑ i, d i) / 18 in
  combined_mean = 20.33 :=
by
  sorry

end mean_of_combined_sets_l630_630914


namespace lottery_distribution_l630_630021

theorem lottery_distribution :
  let x := 155250 in
  let y := x / 1000 in
  let z := y * 100 in
  z = 15525 :=
by
  sorry

end lottery_distribution_l630_630021


namespace min_abs_alpha_gamma_l630_630610

noncomputable def f (α γ z : ℂ) : ℂ :=
  (3 - 2*complex.I) * z^3 + α * z + γ

theorem min_abs_alpha_gamma (α γ : ℂ) 
  (h1 : ∀ z, f α γ z = (3 - 2*complex.I) * z^3 + α * z + γ)
  (h2 : f α γ 1 ∈ ℝ)
  (h3 : f α γ (-1) ∈ ℝ) :
  complex.abs α + complex.abs γ = 2 :=
sorry

end min_abs_alpha_gamma_l630_630610


namespace decimal_to_binary_123_l630_630511

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l630_630511


namespace ratio_of_times_l630_630930

def speed_in_still_water : ℝ := 54
def speed_of_stream : ℝ := 18
def speed_upstream : ℝ := speed_in_still_water - speed_of_stream
def speed_downstream : ℝ := speed_in_still_water + speed_of_stream

def distance : ℝ := 1 -- Assuming unit distance to focus on ratio

def time_upstream : ℝ := distance / speed_upstream
def time_downstream : ℝ := distance / speed_downstream

theorem ratio_of_times :
  (time_upstream / time_downstream) = (2 : ℝ) :=
sorry

end ratio_of_times_l630_630930


namespace commute_times_abs_difference_l630_630213

theorem commute_times_abs_difference (x y : ℝ)
  (h_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (h_var : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) 
  : |x - y| = 4 :=
sorry

end commute_times_abs_difference_l630_630213


namespace ellipse_standard_equation_l630_630927

-- Defining the conditions of the problem
def minor_axis_length : ℝ := 8
def eccentricity : ℝ := 3 / 5

-- Definitions derived from the conditions
def b : ℝ := minor_axis_length / 2
def c : ℝ := eccentricity * sqrt (25)
def a : ℝ := sqrt (c^2 + b^2)

-- Lean theorem to state the math proof problem
theorem ellipse_standard_equation (minor_axis_length = 8) (eccentricity = 3 / 5) :
  (a = 5 ∧ b = 4 ∧ (c = 3 ∧ ((x:ℝ) * x) / 25 + ((y:ℝ) * y) / 16 = 1 ∨
  ((y:ℝ) * y) / 25 + ((x:ℝ) * x) / 16 = 1)) :=
sorry

end ellipse_standard_equation_l630_630927


namespace probability_of_same_label_l630_630978

-- Define the deck and properties
def deck : finset (fin 50) := finset.range 50

-- Define a function to label cards
def label (n : fin 50) : ℕ :=
if n.val < 48 then (n.val / 4) + 1 else 13

-- Total number of pairs
def total_pairs : ℕ := (deck.card * (deck.card - 1)) / 2

-- Number of favorable pairs
def favorable_pairs : ℕ :=
let labeled_1_to_12 := 12 * (4 * (4 - 1) / 2) in
let labeled_13 := 1 in
labeled_1_to_12 + labeled_13

-- Calculate the probability
def calculate_probability : ℚ :=
favorable_pairs.to_rat / total_pairs.to_rat

-- Statement of the problem
theorem probability_of_same_label : calculate_probability = 73 / 1225 := by
  sorry

end probability_of_same_label_l630_630978


namespace ratio_triangle_ABJ_to_ADE_l630_630068

-- Defining the regular octagon and conditions
variable [regular_octagon : Type] (ABCDEF : regular_octagon → Type) 

-- Defining the smaller triangles and areas
variable (small_triangle_area : ℕ)

-- Defining the area of the triangles involved
noncomputable def triangle_ABJ_area := 2 * small_triangle_area
noncomputable def triangle_ADE_area := 6 * small_triangle_area

-- Statement of the mathematical proof problem
theorem ratio_triangle_ABJ_to_ADE :
  triangle_ABJ_area / triangle_ADE_area = 1 / 3 := by
  sorry

end ratio_triangle_ABJ_to_ADE_l630_630068


namespace solve_abs_inequality_l630_630893

theorem solve_abs_inequality (x : ℝ) (h : x ≠ 1) : 
  abs ((3 * x - 2) / (x - 1)) > 3 ↔ (5 / 6 < x ∧ x < 1) ∨ (x > 1) := 
by 
  sorry

end solve_abs_inequality_l630_630893


namespace remainder_of_b_97_mod_36_l630_630040

/--
Let \( b_n = 5^n + 7^n \). Prove that the remainder of \( b_{97} \) when divided by 36 is 12.
-/
theorem remainder_of_b_97_mod_36 :
  let b_n (n : ℕ) := 5^n + 7^n in
  b_n 97 % 36 = 12 :=
by
  sorry

end remainder_of_b_97_mod_36_l630_630040


namespace sum_of_areas_of_tangent_circles_l630_630493

theorem sum_of_areas_of_tangent_circles :
  ∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧
    (r + s = 3) ∧
    (r + t = 4) ∧
    (s + t = 5) ∧
    π * (r^2 + s^2 + t^2) = 14 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l630_630493


namespace last_digit_fifth_power_l630_630074

theorem last_digit_fifth_power (R : ℤ) : (R^5 - R) % 10 = 0 := 
sorry

end last_digit_fifth_power_l630_630074


namespace percent_gain_is_5_333_l630_630609

noncomputable def calculate_percent_gain (total_sheep : ℕ) 
                                         (sold_sheep : ℕ) 
                                         (price_paid_sheep : ℕ) 
                                         (sold_remaining_sheep : ℕ)
                                         (remaining_sheep : ℕ) 
                                         (total_cost : ℝ) 
                                         (initial_revenue : ℝ) 
                                         (remaining_revenue : ℝ) : ℝ :=
  (remaining_revenue + initial_revenue - total_cost) / total_cost * 100

theorem percent_gain_is_5_333
  (x : ℝ)
  (total_sheep : ℕ := 800)
  (sold_sheep : ℕ := 750)
  (price_paid_sheep : ℕ := 790)
  (remaining_sheep : ℕ := 50)
  (total_cost : ℝ := (800 : ℝ) * x)
  (initial_revenue : ℝ := (790 : ℝ) * x)
  (remaining_revenue : ℝ := (50 : ℝ) * ((790 : ℝ) * x / 750)) :
  calculate_percent_gain total_sheep sold_sheep price_paid_sheep remaining_sheep 50 total_cost initial_revenue remaining_revenue = 5.333 := by
  sorry

end percent_gain_is_5_333_l630_630609


namespace max_of_sqrt_sum_value_of_a_b_c_l630_630310

def sqrt (x : ℝ) : ℝ := real.sqrt x

theorem max_of_sqrt_sum
  (a b c : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : a + b + 9*c^2 = 1)
  : sqrt a + sqrt b + sqrt 3 * c ≤ sqrt (7 / 3) :=
sorry

theorem value_of_a_b_c
  (a b c : ℝ)
  (h₀ : a = 3 / 7)
  (h₁ : b = 3 / 7)
  (h₂ : c = sqrt 7 / 21)
  : a + b + c = (18 + sqrt 7) / 21 :=
sorry

end max_of_sqrt_sum_value_of_a_b_c_l630_630310


namespace sufficient_not_necessary_a_equals_2_l630_630973

theorem sufficient_not_necessary_a_equals_2 {a : ℝ} :
  (∃ a : ℝ, (a = 2 ∧ 15 * a^2 = 60) → (15 * a^2 = 60) ∧ (15 * a^2 = 60 → a = 2)) → 
  (¬∀ a : ℝ, (15 * a^2 = 60) → a = 2) → 
  (a = 2 → 15 * a^2 = 60) ∧ ¬(15 * a^2 = 60 → a = 2) :=
by
  sorry

end sufficient_not_necessary_a_equals_2_l630_630973


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630149

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630149


namespace sequence_formula_sum_terms_maximum_sum_abs_sequence_l630_630045

def arithmetic_sequence (a : Nat -> Int) : Prop :=
  ∃ d : Int, ∀ n : Nat, a n = a 0 + n * d

def sequence (a : Nat -> Int) : Prop :=
  a 2 = 9 ∧ a 1 + a 5 = 16 ∧ arithmetic_sequence a

def general_formula (a : Nat -> Int) : Prop :=
  ∀ n : Nat, a n = 11 - n

def sum_of_first_n_terms (a: Nat -> Int) (S: Nat -> Int) : Prop := 
  ∀ n, S n = n * (21 - n) / 2

def sum_of_abs_sequence (a: Nat -> Int) (T: Nat -> Int) : Prop :=
  ∀ n, (n ≤ 11 → T n = n * (21 - n) / 2) ∧ (n ≥ 12 → T n = 110 - n * (21 - n) / 2)

theorem sequence_formula : 
  ∃ a: Nat -> Int, sequence a ∧ general_formula a :=
sorry

theorem sum_terms_maximum : 
  ∃ a S, sequence a ∧ sum_of_first_n_terms a S ∧ (S 10 = 55 ∧ S 11 = 55) :=
sorry

theorem sum_abs_sequence :
  ∃ a T, sequence a ∧ sum_of_abs_sequence a T :=
sorry

end sequence_formula_sum_terms_maximum_sum_abs_sequence_l630_630045


namespace part_a_l630_630965

theorem part_a (n : ℕ) (h : n > 100) (G : SimpleGraph (Fin n)) :
  (¬ ∀ (A B : Finset (Fin n)), A ∪ B = Finset.univ →
   ∀ (v : Fin n), v ∈ A → (∑ w in G.neighborFinset v, if w ∈ A then 1 else 0) ≥ 
                  (∑ w in G.neighborFinset v, if w ∈ B then 1 else 0)) :=
sorry

end part_a_l630_630965


namespace smaller_angle_at_9_00_am_l630_630660

-- Definitions based on conditions identified in Step A
def minute_hand_position : ℝ := 0
def hour_hand_position : ℝ := 270
def degrees_per_hour : ℝ := 30

-- The main theorem statement
theorem smaller_angle_at_9_00_am : 
  let smaller_angle := min (abs (minute_hand_position - hour_hand_position)) (360 - abs (minute_hand_position - hour_hand_position))
  in smaller_angle = 90 := 
by 
  -- Since we skip the proof, we just put 'sorry'
  sorry

end smaller_angle_at_9_00_am_l630_630660


namespace cyclist_speed_l630_630986

noncomputable def distance_m (d : ℕ) : ℝ := d / 1000.0
noncomputable def time_min_sec (min sec : ℕ) : ℝ := (min * 60 + sec) / 3600.0
noncomputable def speed (d t : ℝ) : ℝ := d / t

theorem cyclist_speed (d : ℕ) (min sec : ℕ) (h₁ : d = 750) (h₂ : min = 2) (h₃ : sec = 30) :
  speed (distance_m d) (time_min_sec min sec) = 18 :=
by {
  rw [distance_m, time_min_sec, speed],
  rw [h₁, h₂, h₃],
  norm_num,
  linarith,
  sorry
}

end cyclist_speed_l630_630986


namespace max_balanced_tournament_exists_l630_630144

def balanced_tournament (n : ℕ) : Prop :=
  n ≥ 4 ∧ (∀ (A B C D : Fin n), -- assuming Fin n represents the n teams
    distinct([A, B, C, D]) →
      (match_count([A, B, C, D]) = 3))

def largest_balanced_tournament (n : ℕ) : Prop :=
  balanced_tournament n ∧ (∀ m ≥ n, ¬balanced_tournament m)

theorem max_balanced_tournament_exists :
    ∃ n, largest_balanced_tournament n ∧ n = 5 := 
  sorry

end max_balanced_tournament_exists_l630_630144


namespace train_pass_jogger_time_l630_630989

/-- Given a jogger running at 7 km/hr, a train running at 60 km/hr, the jogger is 350 m ahead of the train
with the train being 150 m long, prove that the train will pass the jogger in approximately 33.96 seconds. -/
theorem train_pass_jogger_time :
  let jogger_speed := 7 * 1000 / 3600 in                      -- convert jogger speed to m/s
  let train_speed := 60 * 1000 / 3600 in                      -- convert train speed to m/s
  let relative_speed := train_speed - jogger_speed in
  let distance_ahead := 350 in
  let train_length := 150 in
  let total_distance := distance_ahead + train_length in
  let time := total_distance / relative_speed in
  time ≈ 33.96 :=
by
  sorry

end train_pass_jogger_time_l630_630989


namespace base_2_representation_of_123_l630_630521

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630521


namespace kristine_more_CDs_l630_630022

-- Conditions
variable (Dawn_CDs : ℕ) (total_CDs : ℕ)
variable (h1 : Dawn_CDs = 10) (h2 : total_CDs = 27)

-- Definition to express the CDs of Kristine based on the given conditions
def Kristine_CDs (Dawn_CDs total_CDs : ℕ) : ℕ := total_CDs - Dawn_CDs

-- Theorem to prove the difference in the number of CDs
theorem kristine_more_CDs : Kristine_CDs Dawn_CDs total_CDs - Dawn_CDs = 7 :=
by
  rw [Kristine_CDs, h1, h2]
  sorry

end kristine_more_CDs_l630_630022


namespace triangle_problems_l630_630490

-- Definitions for triangle sides and angles
variables {A B C : ℝ} {a b c : ℝ}

-- Condition: The area of triangle ABC is a^2 / (3 * sin A)
def area_condition (A B C a b c : ℝ) : Prop :=
  (1 / 2) * b * c * sin A = a^2 / (3 * sin A)

-- Given condition: 6 * cos B * cos C = 1
def cos_condition (B C : ℝ) : Prop :=
  6 * cos B * cos C = 1

-- Given side length a = 3
def side_a_condition (a : ℝ) : Prop :=
  a = 3

-- Main statement: proving \sin B \sin C and the perimeter ABC
theorem triangle_problems
  (A B C a b c : ℝ)
  (h_area : area_condition A B C a b c)
  (h_cos : cos_condition B C)
  (h_a : side_a_condition a) :
  sin B * sin C = 2 / 3 ∧ a + b + c = 3 + Real.sqrt 33 :=
sorry

end triangle_problems_l630_630490


namespace cyclic_quad_inscribed_angles_l630_630815

open EuclideanGeometry -- Assuming we have a geometry module.

noncomputable def cyclic_quadrilateral (a b c d : Point) : Prop :=
  ∃ o : Point, circline o (circle_radius_point o c) ∧
  on_circle a ∧ on_circle b ∧ on_circle c ∧ on_circle d

theorem cyclic_quad_inscribed_angles :
  ∀ (a b c d e f : Point),
    cyclic_quadrilateral a b c d ∧
    is_extended a b e ∧
    is_extended b c f ∧
    tangent_at_point f c ∧
    angle a b e = 110 ∧
    angle a d c = 60 →
    angle e b c = 60 := by
  sorry

end cyclic_quad_inscribed_angles_l630_630815


namespace simplification_and_evaluation_l630_630890

theorem simplification_and_evaluation (a : ℚ) (h : a = -1 / 2) :
  (3 * a + 2) * (a - 1) - 4 * a * (a + 1) = 1 / 4 := 
by
  sorry

end simplification_and_evaluation_l630_630890


namespace polynomial_sum_l630_630410

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l630_630410


namespace minimize_S_n_l630_630295

noncomputable def S_n (n : ℕ) : ℝ := 2 * (n : ℝ) ^ 2 - 30 * (n : ℝ)

theorem minimize_S_n :
  ∃ n : ℕ, S_n n = 2 * (7 : ℝ) ^ 2 - 30 * (7 : ℝ) ∨ S_n n = 2 * (8 : ℝ) ^ 2 - 30 * (8 : ℝ) := by
  sorry

end minimize_S_n_l630_630295


namespace find_10th_integer_l630_630359

-- Defining the conditions
def avg_20_consecutive_integers (avg : ℝ) : Prop :=
  avg = 23.65

def consecutive_integer_sequence (n : ℤ) (a : ℤ) : Prop :=
  a = n + 9

-- The main theorem statement
theorem find_10th_integer (n : ℤ) (avg : ℝ) (h_avg : avg_20_consecutive_integers avg) (h_seq : consecutive_integer_sequence n 23) :
  n = 14 :=
sorry

end find_10th_integer_l630_630359


namespace evaluate_f_at_2_l630_630323

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then x^2 - 4
  else if x > 2 then 2 * x
  else 0

theorem evaluate_f_at_2 : f 2 = 0 := by
  sorry

end evaluate_f_at_2_l630_630323


namespace find_b_l630_630400

open Real

def a : ℝ × ℝ × ℝ := (3, 2, 4)
def b : ℝ × ℝ × ℝ := (-3, 4, 0.5)

theorem find_b (dot_product_eq : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 14) 
  (cross_product_eq : (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1) = (-15, -10, 8)) :
  b = (-3, 4, 0.5) := 
sorry

end find_b_l630_630400


namespace square_d_perimeter_l630_630078

theorem square_d_perimeter (perimeter_C : ℝ) (half_area_of_C : ℝ) :
  perimeter_C = 32 → half_area_of_C = 32 → 
  (4 * (Real.sqrt (half_area_of_C))) * 4 = 16 * Real.sqrt 2 :=
by
  intros h1 h2
  rw [h1,h2]
  sorry

end square_d_perimeter_l630_630078


namespace sin_480_eq_sqrt3_div_2_l630_630121

theorem sin_480_eq_sqrt3_div_2 : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_eq_sqrt3_div_2_l630_630121


namespace collinear_C_M_K_l630_630191

-- Definitions of square, equilateral triangles, and collinearity using conditions from the given problem.
def square (A B C D : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  angle A B C = π/2 ∧ angle B C D = π/2 ∧ angle C D A = π/2 ∧ angle D A B = π/2

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Main theorem statement
theorem collinear_C_M_K (A B C D M K : Point)
  (h_square : square A B C D)
  (h_equilateral_AMD : equilateral_triangle A M D)
  (h_equilateral_AKB : equilateral_triangle A K B) :
  collinear {C, M, K} :=
sorry

end collinear_C_M_K_l630_630191


namespace base_2_representation_of_123_l630_630545

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630545


namespace smallest_x_for_equation_l630_630683

theorem smallest_x_for_equation :
  ∃ x : ℝ, x = -15 ∧ (∀ y : ℝ, 3*y^2 + 39*y - 75 = y*(y + 16) → x ≤ y) ∧ 
  3*(-15)^2 + 39*(-15) - 75 = -15*(-15 + 16) :=
sorry

end smallest_x_for_equation_l630_630683


namespace points_form_polygon_and_properties_l630_630871

noncomputable def parallelogram_polygon (A B C D E F O : Point) (X : Point ∈ triangle A B C) (Y : Point ∈ triangle D E F) : Prop := 
  ∃ Z : Point, is_parallelogram O X Z Y

theorem points_form_polygon_and_properties 
  (A B C D E F O : Point) 
  (X : Point ∈ triangle A B C) 
  (Y : Point ∈ triangle D E F) :
  (∃ (Z : Point), parallelogram_polygon A B C D E F O X Y)
  ∧ (polygon_sides (polygon_points Z) = 6)
  ∧ (polygon_perimeter (polygon_points Z) = (triangle_perimeter A B C) + (triangle_perimeter D E F)) :=
by 
  sorry

end points_form_polygon_and_properties_l630_630871


namespace perfect_square_product_exists_l630_630743

theorem perfect_square_product_exists (n : ℕ) (h : n ≥ 1)
  (p : ℕ → ℕ) (hp : ∀ i, 1 ≤ i ∧ i ≤ n → nat.prime (p i))
  (m : ℕ → ℕ) (hm : ∀ j, 1 ≤ j ∧ j ≤ n + 1 → ∃ α : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ n → α i ≥ 0) ∧ (m j = ∏ i in finset.range n, p i ^ α i)) :
  ∃ I : finset (fin (n + 1)), ↑I ≠ ∅ ∧ (∏ j in I, m j)^2 ∈ finset.range n :=
sorry

end perfect_square_product_exists_l630_630743


namespace polynomial_sum_l630_630409

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l630_630409


namespace base_2_representation_of_123_l630_630538

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630538


namespace minimum_valid_N_exists_l630_630697

theorem minimum_valid_N_exists (N : ℝ) (a : ℕ → ℕ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, a n < a (n+1)) →
  (∀ n : ℕ, (a (2*n - 1) + a (2*n)) / a n = N) →
  N ≥ 4 :=
by
  sorry

end minimum_valid_N_exists_l630_630697


namespace cut_frame_into_16_equal_parts_l630_630675

-- Define the frame as a rectangular structure with a hollow space inside it.
structure Frame where
  width : ℕ
  height : ℕ
  hollowWidth : ℕ
  hollowHeight : ℕ
  (hollowWidth < width) : Prop
  (hollowHeight < height) : Prop

-- Define the goal: prove the frame can be cut into 16 equal parts.
theorem cut_frame_into_16_equal_parts (f: Frame) : ∃ (parts : List (Frame × Frame)), parts.length = 16 := by
  sorry

end cut_frame_into_16_equal_parts_l630_630675


namespace number_of_results_in_second_group_l630_630092

-- Define the conditions as variables
variables {n : ℕ}

-- State the total sum of the first group of results
def sum_group1 := 30 * 20

-- State the total sum of the second group of results
def sum_group2 := n * 30

-- State the total number of results
def total_results := 30 + n

-- State the total sum of all the results using the overall average
def total_sum_all := (30 + n) * 24

-- Given the equation derived from the conditions:
def equation := sum_group1 + sum_group2 = total_sum_all

-- Prove that the number of results in the second group is 20
theorem number_of_results_in_second_group : n = 20 :=
by
  sorry

end number_of_results_in_second_group_l630_630092


namespace intersection_of_A_and_B_l630_630785

-- Definitions of the sets A and B
def A := {y | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ y = 2^x}
def B := {1, 2, 3, 4}

-- Statement to prove
theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l630_630785


namespace floor_subtraction_l630_630257

theorem floor_subtraction :
  Int.floor (-2.7) - Int.floor 4.5 = -7 := 
by
  sorry

end floor_subtraction_l630_630257


namespace MrsSmith_A_and_B_Students_l630_630814

-- Define the condition of Mr. Johnson's class
def MrJohnson_A_Students : ℕ := 12
def MrJohnson_Total_Students : ℕ := 20

-- Define the proportion of 'A' students
def Proportion_A_Johnson : ℚ := MrJohnson_A_Students / MrJohnson_Total_Students

-- Define Mrs. Smith's class
def MrsSmith_Total_Students : ℕ := 30

-- Define the proportion condition for Mrs. Smith's class
def Proportion_A_Smith : ℚ := 3 / 5

-- Define the number of students who received 'A' in Mrs. Smith's class using proportion
def MrsSmith_A_Students : ℕ :=
  MrsSmith_Total_Students * Proportion_A_Smith

-- Define the number of 'B' students in Mrs. Smith's class
def MrsSmith_B_Students : ℕ :=
  MrsSmith_Total_Students - MrsSmith_A_Students

-- Theorem to prove
theorem MrsSmith_A_and_B_Students :
  MrsSmith_A_Students = 18 ∧ MrsSmith_B_Students = 12 :=
by
  sorry

end MrsSmith_A_and_B_Students_l630_630814


namespace polar_coordinates_of_3_neg3_l630_630674

def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (r, θ)

theorem polar_coordinates_of_3_neg3 : point_rectangular_to_polar 3 (-3) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := by
  sorry

end polar_coordinates_of_3_neg3_l630_630674


namespace complement_of_A_in_R_l630_630308

open Set

variable (R : Set ℝ) (A : Set ℝ)

def real_numbers : Set ℝ := {x | true}

def set_A : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem complement_of_A_in_R : (real_numbers \ set_A) = {y | y < 0} := by
  sorry

end complement_of_A_in_R_l630_630308


namespace polynomial_division_l630_630417

theorem polynomial_division (P : Polynomial ℝ) (n : ℕ) (hn : 0 < n) :
  (P - Polynomial.X) ∣ (nat.iterate (λ Q, Q.comp P) n P - Polynomial.X) :=
by sorry

end polynomial_division_l630_630417


namespace part_a_part_b_l630_630968

-- For Part (a)
theorem part_a (n : ℕ) (h : n > 100) (exists_friends : ∃ (f : fin n → fin n → Prop), true) : 
  ¬(∀ (grp : fin n → bool),
    ∀ (i : fin n), 
    (∑ j in finset.univ, if grp i = grp j then if exists_friends.some i j then 1 else 0 else 0) ≥ 
    (∑ j in finset.univ, if grp i ≠ grp j then if exists_friends.some i j then 1 else 0 else 0)) :=
by
  sorry

-- For Part (b)
theorem part_b (n : ℕ) (h : n = 2022) (exists_friends : ∃ (friendship : fin n → fin n → Prop), true) :
  ∃ (grp : fin n → fin 15), ∀ (i : fin n),
  (∑ j in finset.univ, if grp i = grp j then if exists_friends.some i j then 1 else 0 else 0) ≤
  (1/15) * (∑ j in finset.univ, if exists_friends.some i j then 1 else 0) :=
by
  sorry

end part_a_part_b_l630_630968


namespace polynomial_root_properties_l630_630265

noncomputable theory
open Polynomial

theorem polynomial_root_properties :
  ∀ (P : Polynomial ℝ),
    ∀ a0 a1 a2 a3 a4 : ℝ,
      P = X^5 + C a4 * X^4 + C a3 * X^3 + C a2 * X^2 + C a1 * X + C a0 →
      (∀ α : ℂ, is_root P α → is_root P (1 - α) ∧ is_root P (1 / α)) →
      P = (X + 1) * (X - (1 / 2)) * (X - 2) * (X^2 - X + 1) 
      ∨ P = (X + 1)^3 * (X - (1 / 2)) * (X - 2) 
      ∨ P = (X + 1) * (X - (1 / 2))^3 * (X - 2) 
      ∨ P = (X + 1) * (X - (1 / 2)) * (X - 2)^3 
      ∨ P = (X + 1)^2 * (X - (1 / 2))^2 * (X - 2) 
      ∨ P = (X + 1) * (X - (1 / 2))^2 * (X - 2)^2 
      ∨ P = (X + 1)^2 * (X - (1 / 2)) * (X - 2)^2 :=
begin
  sorry
end

end polynomial_root_properties_l630_630265


namespace base_2_representation_of_123_is_1111011_l630_630523

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630523


namespace value_of_expression_at_three_l630_630952

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end value_of_expression_at_three_l630_630952


namespace product_of_eccentricities_l630_630772

theorem product_of_eccentricities (a b k : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a = sqrt 3 * k) (h₃ : b = k) (h₄ : k > 0) :
  (sqrt (a^2 - b^2) / a) * (sqrt (a^2 + b^2) / a) = (2 * sqrt 2) / 3 :=
by 
  sorry

end product_of_eccentricities_l630_630772


namespace solve_system_of_equations_l630_630450

theorem solve_system_of_equations :
  ∃ (x y z : ℤ), (x + y + z = 6) ∧ (x + y * z = 7) ∧ 
  ((x = 7 ∧ y = 0 ∧ z = -1) ∨ 
   (x = 7 ∧ y = -1 ∧ z = 0) ∨ 
   (x = 1 ∧ y = 3 ∧ z = 2) ∨ 
   (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end solve_system_of_equations_l630_630450


namespace part_a_part_b_l630_630967

-- For Part (a)
theorem part_a (n : ℕ) (h : n > 100) (exists_friends : ∃ (f : fin n → fin n → Prop), true) : 
  ¬(∀ (grp : fin n → bool),
    ∀ (i : fin n), 
    (∑ j in finset.univ, if grp i = grp j then if exists_friends.some i j then 1 else 0 else 0) ≥ 
    (∑ j in finset.univ, if grp i ≠ grp j then if exists_friends.some i j then 1 else 0 else 0)) :=
by
  sorry

-- For Part (b)
theorem part_b (n : ℕ) (h : n = 2022) (exists_friends : ∃ (friendship : fin n → fin n → Prop), true) :
  ∃ (grp : fin n → fin 15), ∀ (i : fin n),
  (∑ j in finset.univ, if grp i = grp j then if exists_friends.some i j then 1 else 0 else 0) ≤
  (1/15) * (∑ j in finset.univ, if exists_friends.some i j then 1 else 0) :=
by
  sorry

end part_a_part_b_l630_630967


namespace num_3digit_strictly_ordered_l630_630835

def is_strictly_increasing (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  (hundreds < tens) ∧ (tens < units)

def is_strictly_decreasing (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  (hundreds > tens) ∧ (tens > units)

def count_special_decreasing : ℕ :=
  List.sum $ List.map (λ x, x - 1) [2..9]

def count_numbers_with_special_property : ℕ :=
  let strictly_increasing_count := List.length (List.filter is_strictly_increasing [100..999])
  let strictly_decreasing_count := List.length (List.filter is_strictly_decreasing [100..999])
  strictly_increasing_count + strictly_decreasing_count + count_special_decreasing

theorem num_3digit_strictly_ordered : count_numbers_with_special_property = 204 :=
by sorry

end num_3digit_strictly_ordered_l630_630835


namespace points_on_same_circle_l630_630873

variables {α : Type*} [InnerProductSpace ℝ α] [Plane α]

-- Definitions of midpoint, foot of perpendicular, and orthogonal projections.
def midpoint (A B : α) : α := (A + B) / 2

def foot_of_perpendicular (B AC_line : α × α) : α := sorry

def orthogonal_projection (A bisector_line : α × α) : α := sorry

-- Theorem statement.
theorem points_on_same_circle 
  (A B C : α) 
  (AC_line : α × α := (A, C))
  (bisector_line : α × α := sorry) -- Define bisector of angle B
  (M : α := midpoint A C)
  (H : α := foot_of_perpendicular B AC_line)
  (P : α := orthogonal_projection A bisector_line) 
  (Q : α := orthogonal_projection C bisector_line) :
  ∃ (circle : α × ℝ), 
    ∀ (point : α), point ∈ {M, H, P, Q} → dist point (circle.1) = circle.2 :=
sorry

end points_on_same_circle_l630_630873


namespace smallest_sum_is_minus_half_l630_630271

def smallest_sum (x: ℝ) : ℝ := x^2 + x

theorem smallest_sum_is_minus_half : ∃ x : ℝ, ∀ y : ℝ, smallest_sum y ≥ smallest_sum (-1/2) :=
by
  use -1/2
  intros y
  sorry

end smallest_sum_is_minus_half_l630_630271


namespace exists_gcd_gt_one_l630_630024

noncomputable def Nat.gcd (a b : ℕ) : ℕ := sorry

theorem exists_gcd_gt_one (A : Finset ℕ) (h_card : A.card = 16) (h_prod : ∀ x y ∈ A, x ≠ y → x * y ≤ 1994) :
  ∃ a b ∈ A, a ≠ b ∧ Nat.gcd a b > 1 :=
begin
  -- proof will go here
  sorry
end

end exists_gcd_gt_one_l630_630024


namespace andres_possibilities_10_dollars_l630_630653

theorem andres_possibilities_10_dollars : 
  (∃ (num_1_coins num_2_coins num_5_bills : ℕ),
    num_1_coins + 2 * num_2_coins + 5 * num_5_bills = 10) → 
  ∃ (ways : ℕ), ways = 10 :=
by
  -- The proof can be provided here, but we'll use sorry to skip it in this template.
  sorry

end andres_possibilities_10_dollars_l630_630653


namespace S_p_plus_q_l630_630294

noncomputable def is_nonconstant_polynomial_with_real_coeffs (f : polynomial ℝ) : Prop :=
0 < f.degree

def S (f : polynomial ℝ) : ℝ := -(f.coeff (f.degree.to_nat - 1)) / f.coeff f.degree.to_nat

variables (p q : polynomial ℝ)
hypothesis (h_p : S p = 7)
hypothesis (h_q : S q = 9)
hypothesis (h_pq_diff : S (p - q) = 11)
hypothesis (h_nonconstant_p : is_nonconstant_polynomial_with_real_coeffs p)
hypothesis (h_nonconstant_q : is_nonconstant_polynomial_with_real_coeffs q)

theorem S_p_plus_q : S (p + q) = 5 :=
by sorry

end S_p_plus_q_l630_630294


namespace net_change_of_Toronto_Stock_Exchange_l630_630254

theorem net_change_of_Toronto_Stock_Exchange :
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  (monday + tuesday + wednesday + thursday + friday) = -119 :=
by
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  have h : (monday + tuesday + wednesday + thursday + friday) = -119 := sorry
  exact h

end net_change_of_Toronto_Stock_Exchange_l630_630254


namespace temperature_difference_l630_630057

theorem temperature_difference (t_low t_high : ℝ) (h_low : t_low = -2) (h_high : t_high = 5) :
  t_high - t_low = 7 :=
by
  rw [h_low, h_high]
  norm_num

end temperature_difference_l630_630057


namespace radius_of_circle_is_zero_l630_630273

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 2 * x^2 - 8 * x + 2 * y^2 - 4 * y + 10 = 0

-- Define the goal: To prove that given this equation, the radius of the circle is 0
theorem radius_of_circle_is_zero :
  ∀ x y : ℝ, circle_eq x y → (x - 2)^2 + (y - 1)^2 = 0 :=
sorry

end radius_of_circle_is_zero_l630_630273


namespace cyclic_sum_nonnegative_cyclic_sum_equality_cases_l630_630854

theorem cyclic_sum_nonnegative (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (\frac{a - b}{b + c} + \frac{b - c}{c + d} + \frac{c - d}{d + a} + \frac{d - a}{a + b} ≥ 0) :=
sorry

theorem cyclic_sum_equality_cases (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (\frac{a - b}{b + c} + \frac{b - c}{c + d} + \frac{c - d}{d + a} + \frac{d - a}{a + b} = 0) ↔ (a = b ∧ b = c ∧ c = d) :=
sorry

end cyclic_sum_nonnegative_cyclic_sum_equality_cases_l630_630854


namespace smallest_five_digit_multiple_of_53_l630_630556

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l630_630556


namespace smallest_integers_mod_13_diff_l630_630035

theorem smallest_integers_mod_13_diff :
  let m := Nat.find (λ x, x ≥ 100 ∧ x % 13 = 7) in
  let n := Nat.find (λ x, x ≥ 1000 ∧ x % 13 = 7) in
  n - m = 897 :=
by
  -- Definitions using the conditions
  let m := Nat.find (λ x, x ≥ 100 ∧ x % 13 = 7)
  let n := Nat.find (λ x, x ≥ 1000 ∧ x % 13 = 7)
  -- Goal statement
  have h1 : m = 111 := sorry
  have h2 : n = 1008 := sorry
  calc
  n - m = 1008 - 111 : by rw [h1, h2]
       ... = 897 : by norm_num

end smallest_integers_mod_13_diff_l630_630035


namespace ratio_of_volumes_l630_630122

theorem ratio_of_volumes (e : ℕ) : 
  let V1 := e^3 in
  let V2 := (2*e)^3 in
  V2 / V1 = 8 := by
  sorry

end ratio_of_volumes_l630_630122


namespace pow_mod_remainder_l630_630171

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l630_630171


namespace sum_of_first_five_integers_l630_630680

theorem sum_of_first_five_integers : (1 + 2 + 3 + 4 + 5) = 15 := 
by 
  sorry

end sum_of_first_five_integers_l630_630680


namespace jesse_room_width_l630_630387

theorem jesse_room_width
  (length : ℕ) (num_rooms : ℕ) (total_area : ℕ)
  (h_length : length = 19) (h_num_rooms : num_rooms = 20) (h_total_area : total_area = 6840) :
  (total_area / num_rooms) / length = 18 :=
by
  rw [h_length, h_num_rooms, h_total_area]
  simp
  sorry

end jesse_room_width_l630_630387


namespace real_part_of_one_div_two_minus_z_l630_630868

noncomputable def real_part_of_complex_inverse (z : ℂ) (hz : complex.abs z = 2) : ℝ :=
  (2 - z.re) / (4 * (2 - z.re))

theorem real_part_of_one_div_two_minus_z (z : ℂ) (hz : complex.abs z = 2) (hnz : z.im ≠ 0) :
  real_part_of_complex_inverse z hz = 1 / 4 := by
sorry

end real_part_of_one_div_two_minus_z_l630_630868


namespace Charles_has_13_whistles_l630_630446

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end Charles_has_13_whistles_l630_630446


namespace range_of_m_l630_630364

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y - x * y = 0) : 
  ∀ m : ℝ, (xy ≥ m^2 - 6 * m ↔ -2 ≤ m ∧ m ≤ 8) :=
sorry

end range_of_m_l630_630364


namespace solution_set_is_01_l630_630118

noncomputable def solution_set : set ℝ :=
  {x | x > 0 ∧ abs (x + real.logb 3 x) < abs x + abs (real.logb 3 x)}

theorem solution_set_is_01 : solution_set = {x | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_is_01_l630_630118


namespace pie_eaten_after_four_trips_l630_630961

theorem pie_eaten_after_four_trips : 
  let trip1 := (1 / 3 : ℝ)
  let trip2 := (1 / 3^2 : ℝ)
  let trip3 := (1 / 3^3 : ℝ)
  let trip4 := (1 / 3^4 : ℝ)
  trip1 + trip2 + trip3 + trip4 = (40 / 81 : ℝ) :=
by
  sorry

end pie_eaten_after_four_trips_l630_630961


namespace intersection_M_N_l630_630787

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℝ := {x | Real.log x / Real.log 4 ≥ 1}

theorem intersection_M_N :
  M ∩ N = {4, 5} := by
  sorry

end intersection_M_N_l630_630787


namespace pow_mod_remainder_l630_630170

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l630_630170


namespace base_2_representation_of_123_l630_630522

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630522


namespace pow_mod_remainder_l630_630169

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l630_630169


namespace combined_soldiers_correct_l630_630370

-- Define the parameters for the problem
def interval : ℕ := 5
def wall_length : ℕ := 7300
def soldiers_per_tower : ℕ := 2

-- Calculate the number of towers and the total number of soldiers
def num_towers : ℕ := wall_length / interval
def combined_soldiers : ℕ := num_towers * soldiers_per_tower

-- Prove that the combined number of soldiers is as expected
theorem combined_soldiers_correct : combined_soldiers = 2920 := 
by
  sorry

end combined_soldiers_correct_l630_630370


namespace polynomial_root_sum_l630_630867

theorem polynomial_root_sum {p q : ℝ} :
  (is_root (λ x : ℂ, x^3 + (p * x) + q) (2 + complex.I * real.sqrt 5)) → 
  p + q = 29 := 
sorry

end polynomial_root_sum_l630_630867


namespace find_x_positive_integers_l630_630696

theorem find_x_positive_integers (a b c x : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c = x * a * b * c) → (x = 1 ∧ a = 1 ∧ b = 2 ∧ c = 3) ∨
  (x = 2 ∧ a = 1 ∧ b = 1 ∧ c = 2) ∨
  (x = 3 ∧ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end find_x_positive_integers_l630_630696


namespace nonagon_shortest_diagonal_probability_l630_630148

theorem nonagon_shortest_diagonal_probability:
  let n := 9 in
  let total_diagonals := n * (n - 3) / 2 in
  let shortest_diagonals := n in
  (shortest_diagonals / total_diagonals) = (1 / 3) :=
by
  sorry

end nonagon_shortest_diagonal_probability_l630_630148


namespace intersection_eq_l630_630329

open Set

def P : Set ℤ := {-3, 0, 2, 4}
def Q : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_eq : P ∩ (coe '' Q) = {0, 2} :=
by
  sorry

end intersection_eq_l630_630329


namespace sequence_solution_l630_630837

theorem sequence_solution 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n / (2 + a n))
  (h2 : a 1 = 1) :
  ∀ n, a n = 1 / (2^n - 1) :=
sorry

end sequence_solution_l630_630837


namespace smallest_five_digit_divisible_by_53_l630_630552

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l630_630552


namespace area_of_polygon_ABHFGD_l630_630834

theorem area_of_polygon_ABHFGD
  (A B C D E F G H : Point)
  (h_sq_abcd : square A B C D 16)
  (h_sq_efgd : square E F G D 16)
  (h_midpoint_BC : midpoint H B C)
  (h_midpoint_EF : midpoint H E F) :
  area (polygon AB H FG D) = 24 := 
sorry

end area_of_polygon_ABHFGD_l630_630834


namespace poly_real_coeff_l630_630921

noncomputable def polynomial_g (p q r s : ℝ) : (ℝ[X]) :=
  X^4 + C p * X^3 + C q * X^2 + C r * X + C s

theorem poly_real_coeff (p q r s : ℝ) 
  (h1 : polynomial_g p q r s.eval (3 * Complex.i) = 0)
  (h2 : polynomial_g p q r s.eval (1 + 2 * Complex.i) = 0) :
  p + q + r + s = 39 := 
sorry

end poly_real_coeff_l630_630921


namespace molecular_weight_of_9_moles_CCl4_l630_630147

-- Define the atomic weight of Carbon (C) and Chlorine (Cl)
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular formula for carbon tetrachloride (CCl4)
def molecular_formula_CCl4 : ℝ := atomic_weight_C + (4 * atomic_weight_Cl)

-- Define the molecular weight of one mole of carbon tetrachloride (CCl4)
def molecular_weight_one_mole_CCl4 : ℝ := molecular_formula_CCl4

-- Define the number of moles
def moles_CCl4 : ℝ := 9

-- Define the result to check
def molecular_weight_nine_moles_CCl4 : ℝ := molecular_weight_one_mole_CCl4 * moles_CCl4

-- State the theorem to prove the molecular weight of 9 moles of carbon tetrachloride is 1384.29 grams
theorem molecular_weight_of_9_moles_CCl4 :
  molecular_weight_nine_moles_CCl4 = 1384.29 := by
  sorry

end molecular_weight_of_9_moles_CCl4_l630_630147


namespace part_I_part_II_l630_630325

open Real  -- Specify that we are working with real numbers

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- The first theorem: Prove the result for a = 1
theorem part_I (x : ℝ) : f x 1 + x > 0 ↔ (x > -3 ∧ x < 1 ∨ x > 3) :=
by
  sorry

-- The second theorem: Prove the range of a such that f(x) ≤ 3 for all x
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 3) ↔ (-5 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l630_630325


namespace simplest_sqrt_l630_630566

theorem simplest_sqrt 
  (A := Real.sqrt (1 / 2))
  (B := Real.sqrt 8)
  (C := Real.sqrt 1.5)
  (D := Real.sqrt 6) :
  (∀ x ∈ {A, B, C}, ∃ y, Real.sqrt y = x ∧ y = 2 ∨ y = 3 / 2)
  ∧ (∀ z ∈ {D}, ∃ w, Real.sqrt w = z ∧ w = 6) :=
by sorry

end simplest_sqrt_l630_630566


namespace base_2_representation_of_123_l630_630520

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l630_630520


namespace problem1_l630_630579

theorem problem1 : |-5| + (-1) ^ 2013 * (π - 3.14) ^ 0 - (-1/3) ^ -2 = -5 :=
by sorry

end problem1_l630_630579


namespace poly_real_coeff_l630_630920

noncomputable def polynomial_g (p q r s : ℝ) : (ℝ[X]) :=
  X^4 + C p * X^3 + C q * X^2 + C r * X + C s

theorem poly_real_coeff (p q r s : ℝ) 
  (h1 : polynomial_g p q r s.eval (3 * Complex.i) = 0)
  (h2 : polynomial_g p q r s.eval (1 + 2 * Complex.i) = 0) :
  p + q + r + s = 39 := 
sorry

end poly_real_coeff_l630_630920


namespace system_solution_iff_l630_630699

theorem system_solution_iff (a : ℝ) : (∃ (b x y : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8 * b^2 = 4 * b * (y - x) + 1) ↔ a ≥ -real.sqrt 2 - 1/4 :=
by sorry

end system_solution_iff_l630_630699


namespace acute_triangle_area_relation_l630_630073

open Real

variables (A B C R : ℝ)
variables (acute_triangle : Prop)
variables (S p_star : ℝ)

-- Conditions
axiom acute_triangle_condition : acute_triangle
axiom area_formula : S = (R^2 / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))
axiom semiperimeter_formula : p_star = (R / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))

-- Theorem to prove
theorem acute_triangle_area_relation (h : acute_triangle) : S = p_star * R := 
by {
  sorry 
}

end acute_triangle_area_relation_l630_630073


namespace smaller_angle_at_9_00_am_l630_630659

-- Definitions based on conditions identified in Step A
def minute_hand_position : ℝ := 0
def hour_hand_position : ℝ := 270
def degrees_per_hour : ℝ := 30

-- The main theorem statement
theorem smaller_angle_at_9_00_am : 
  let smaller_angle := min (abs (minute_hand_position - hour_hand_position)) (360 - abs (minute_hand_position - hour_hand_position))
  in smaller_angle = 90 := 
by 
  -- Since we skip the proof, we just put 'sorry'
  sorry

end smaller_angle_at_9_00_am_l630_630659


namespace problem_l630_630112

open Real

theorem problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
sorry

end problem_l630_630112


namespace odd_function_zeros_l630_630677

noncomputable def f (x : ℝ) : ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def zero_count (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry -- Imagine we have a way to count the zeros of f in the interval (a, b)

theorem odd_function_zeros (f : ℝ → ℝ) (h_odd : is_odd f) (h_zeros_pos : zero_count f 0 ∞ = 3) :
    zero_count f -∞ ∞ = 7 :=
begin
  sorry
end

end odd_function_zeros_l630_630677


namespace exists_perfect_square_intersection_l630_630238

theorem exists_perfect_square_intersection : ∃ n : ℕ, n > 1 ∧ ∃ k : ℕ, (2^n - n) = k^2 :=
by sorry

end exists_perfect_square_intersection_l630_630238


namespace point_in_third_quadrant_l630_630805

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end point_in_third_quadrant_l630_630805


namespace slope_of_line_joining_solutions_l630_630774

theorem slope_of_line_joining_solutions (x1 x2 y1 y2 : ℝ) :
  (4 / x1 + 5 / y1 = 1) → (4 / x2 + 5 / y2 = 1) →
  (x1 ≠ x2) → (y1 = 5 * x1 / (4 * x1 - 1)) → (y2 = 5 * x2 / (4 * x2 - 1)) →
  (x1 ≠ 1 / 4) → (x2 ≠ 1 / 4) →
  ((y2 - y1) / (x2 - x1) = - (5 / 21)) :=
by
  intros h_eq1 h_eq2 h_neq h_y1 h_y2 h_x1 h_x2
  -- Proof omitted for brevity
  sorry

end slope_of_line_joining_solutions_l630_630774


namespace quadrilateral_min_area_chord_length_equation_not_x_y_1_line_AB_passes_fixed_point_l630_630748

noncomputable def circle (x y : ℝ) := (x + 1)^2 + y^2 = 2
def line (x y : ℝ) := x - y - 3 = 0

theorem quadrilateral_min_area (x y : ℝ) (hx : circle x y) (hp : line x y) :
  ∃ P A B : ℝ × ℝ, (∃ p hx hy, p ∈ line ∧ p ≠ (x, y)) ∧
  (∀ P : ℝ × ℝ, P ∈ line → P ≠ (x, y) → let PA := tangent_line M P A,
    let PB := tangent_line M P B in
    (area_quadrilateral P A M B) = 2 * real.sqrt 3) := 
begin
  sorry
end

theorem chord_length (x y : ℝ) (hx : circle x y) (hp : line x y) :
  ∃ P A B : ℝ × ℝ, (∃ p hx hy, p ∈ line ∧ p ≠ (x, y)) ∧
  (∀ P : ℝ × ℝ, P ∈ line → P ≠ (x, y) → let PA := tangent_line M P A,
    let PB := tangent_line M P B in
    (length_chord_AB A B) = real.sqrt 6) :=
begin
  sorry
end

theorem equation_not_x_y_1 (x y : ℝ) (hx : circle x y) (hp : line x y) :
  ∃ P A B : ℝ × ℝ, (∃ p hx hy, p ∈ line ∧ p ≠ (x, y)) ∧
  (∀ P : ℝ × ℝ, P ∈ line → P ≠ (x, y) → let PA := tangent_line M P A,
    let PB := tangent_line M P B in
    ¬ equation_chord_AB_eq_x_y_1 A B) :=
begin
  sorry
end

theorem line_AB_passes_fixed_point (x y : ℝ) (hx : circle x y) (hp : line x y) :
  ∃ P A B : ℝ × ℝ, (∃ p hx hy, p ∈ line ∧ p ≠ (x, y)) ∧
  (∀ P : ℝ × ℝ, P ∈ line → P ≠ (x, y) → let PA := tangent_line M P A,
    let PB := tangent_line M P B in
    let fixed_point := (-1/2, -1/2) in
    (line_through_AB fixed_point)) :=
begin
  sorry
end

end quadrilateral_min_area_chord_length_equation_not_x_y_1_line_AB_passes_fixed_point_l630_630748


namespace cube_root_opposite_zero_l630_630904

theorem cube_root_opposite_zero (x : ℝ) (h : x^(1/3) = -x) : x = 0 :=
sorry

end cube_root_opposite_zero_l630_630904


namespace discount_policy_l630_630134

-- Define the prices of the fruits
def lemon_price := 2
def papaya_price := 1
def mango_price := 4

-- Define the quantities Tom buys
def lemons_bought := 6
def papayas_bought := 4
def mangos_bought := 2

-- Define the total amount paid by Tom
def amount_paid := 21

-- Define the total number of fruits bought
def total_fruits_bought := lemons_bought + papayas_bought + mangos_bought

-- Define the total cost without discount
def total_cost_without_discount := 
  (lemons_bought * lemon_price) + 
  (papayas_bought * papaya_price) + 
  (mangos_bought * mango_price)

-- Calculate the discount
def discount := total_cost_without_discount - amount_paid

-- The discount policy
theorem discount_policy : discount = 3 ∧ total_fruits_bought = 12 :=
by 
  sorry

end discount_policy_l630_630134


namespace integer_values_of_k_l630_630561

theorem integer_values_of_k (k : ℕ) :
  0 < k ∧ k ≤ 2007 ∧ ∃ m : ℕ, 1 ≤ m ∧ m ≤ 14 ∧ k = 9 * m^2 :=
begin
  sorry
end

end integer_values_of_k_l630_630561


namespace not_p_is_necessary_but_not_sufficient_l630_630311

-- Definitions based on the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = d

def not_p (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ n : ℕ, a (n + 2) - a (n + 1) ≠ d

def not_q (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ¬ is_arithmetic_sequence a d

-- Proof problem statement
theorem not_p_is_necessary_but_not_sufficient (d : ℝ) (a : ℕ → ℝ) :
  (not_p a d → not_q a d) ∧ (not_q a d → not_p a d) = False := 
sorry

end not_p_is_necessary_but_not_sufficient_l630_630311


namespace convex_pentagon_exists_l630_630746

theorem convex_pentagon_exists (A : fin 9 → ℝ × ℝ) (h : ∀ (i j k : fin 9), i ≠ j ∧ i ≠ k ∧ j ≠ k → (A i, A j, A k) are_not_collinear) : 
  ∃ (S : finset (fin 9)), S.card = 5 ∧ is_convex_hull S :=
sorry

end convex_pentagon_exists_l630_630746


namespace trigonometric_identity_l630_630876

noncomputable def π := Real.pi
noncomputable def tan (x : ℝ) := Real.sin x / Real.cos x

theorem trigonometric_identity (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : tan α = (1 + Real.sin β) / Real.cos β) :
  2 * α - β = π / 2 := 
sorry

end trigonometric_identity_l630_630876


namespace chess_group_players_l630_630934

noncomputable def number_of_players_in_chess_group (n : ℕ) : Prop := 
  n * (n - 1) / 2 = 21

theorem chess_group_players : ∃ n : ℕ, number_of_players_in_chess_group n ∧ n = 7 := 
by
  use 7
  split
  show number_of_players_in_chess_group 7
  unfold number_of_players_in_chess_group
  norm_num
  show 7 = 7
  rfl

end chess_group_players_l630_630934


namespace total_kids_in_lawrence_county_l630_630253

def kids_stayed_home : ℕ := 644997
def kids_went_to_camp : ℕ := 893835
def kids_from_outside : ℕ := 78

theorem total_kids_in_lawrence_county : kids_stayed_home + kids_went_to_camp = 1538832 := by
  sorry

end total_kids_in_lawrence_county_l630_630253


namespace count_m_tuples_l630_630947

open Nat

-- Define the strict positive natural numbers
def NatPos := { n : ℕ // n > 0 }

-- Define the number of m-tuples satisfying the given sum condition
noncomputable def num_m_tuples_satisfying_sum (m n : ℕ) : ℕ :=
  if m > n then 0 else Nat.choose (n - 1) (m - 1)

-- Statement of the theorem
theorem count_m_tuples (m n : ℕ) (h1 : m > 0) (h2 : n >= m) :
  (∃ (x : Fin m → NatPos), (∑ i, x i.val) = n) ↔ num_m_tuples_satisfying_sum m n = Nat.choose (n - 1) (m - 1) :=
sorry

end count_m_tuples_l630_630947


namespace lcm_condition_implies_all_two_l630_630698

theorem lcm_condition_implies_all_two (x : ℕ → ℕ)
  (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 20 → 
        x (i + 2) ^ 2 = Nat.lcm (x (i + 1)) (x i) + Nat.lcm (x i) (x (i - 1)))
  (h₂ : x 0 = x 20)
  (h₃ : x 21 = x 1)
  (h₄ : x 22 = x 2) :
  ∀ i, 1 ≤ i ∧ i ≤ 20 → x i = 2 := 
sorry

end lcm_condition_implies_all_two_l630_630698


namespace a_10_equals_neg_one_l630_630838

def seq (n : ℕ) : ℚ :=
  if n = 1 then -1
  else seq (n - 1) |> fun a => 1 / (1 - a)

theorem a_10_equals_neg_one : seq 10 = -1 := by
  sorry

end a_10_equals_neg_one_l630_630838


namespace angle_ABC_30_l630_630376

-- Variable declarations
variables {A B C D E F : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
          [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F]

-- Given conditions
axiom angle_ACB_90 : angle A C B = 90
axiom DE_perpendicular_BC : perpendicular D E B C
axiom BE_eq_AC : BE = AC
axiom BD_half_cm : distance B D = 1 / 2
axiom DE_plus_BC_eq_1_cm : distance D E + distance B C = 1

-- The theorem statement
theorem angle_ABC_30 :
  angle A B C = 30 :=
by
  -- Skipping the proof for now
  sorry

end angle_ABC_30_l630_630376


namespace average_speed_l630_630926

theorem average_speed (speed1 speed2: ℝ) (time1 time2: ℝ) (h1: speed1 = 90) (h2: speed2 = 40) (h3: time1 = 1) (h4: time2 = 1) :
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 65 := by
  sorry

end average_speed_l630_630926


namespace sum_A_B_C_l630_630923

noncomputable def number_B (A : ℕ) : ℕ := (A * 5) / 2
noncomputable def number_C (B : ℕ) : ℕ := (B * 7) / 4

theorem sum_A_B_C (A B C : ℕ) (h1 : A = 16) (h2 : A * 5 = B * 2) (h3 : B * 7 = C * 4) :
  A + B + C = 126 :=
by
  sorry

end sum_A_B_C_l630_630923


namespace sum_of_angles_divisible_by_360_l630_630602

theorem sum_of_angles_divisible_by_360 {n : ℕ} (h : n ≠ 0) :
  let sides := 2 * n in
  (sides - 2) * 180 = 360 * (n - 1) :=
by
  have sides_eq_2n : sides = 2 * n := rfl
  sorry

end sum_of_angles_divisible_by_360_l630_630602


namespace john_total_calories_l630_630015

def calories_from_chips (total_calories_per_chip : ℕ) (num_chips : ℕ) : ℕ := total_calories_per_chip * num_chips

def calories_per_cheezit (calories_per_chip : ℕ) (additional_fraction : ℚ) : ℕ := 
  calories_per_chip + (calories_per_chip * additional_fraction).to_int

def calories_from_cheezits (calories_per_cheezit : ℕ) (num_cheezits : ℕ) : ℕ := calories_per_cheezit * num_cheezits

def total_calories (calories_chips : ℕ) (calories_cheezits : ℕ) : ℕ := calories_chips + calories_cheezits

theorem john_total_calories :
  let total_calories_per_chip := 60
  let num_chips := 10
  let num_cheezits := 6
  let additional_fraction := 1 / 3
  let calories_per_chip := total_calories_per_chip / num_chips
  let calories_cheezit := calories_per_cheezit calories_per_chip additional_fraction
  calories_chips + calories_cheezits = 108 :=
  sorry

end john_total_calories_l630_630015


namespace sum_abs_c_eq_108_sum_possible_abs_c_l630_630896

variables {P : ℝ → ℝ}
variables {r s b c : ℝ}

-- Given conditions
def polynomial (x : ℝ) := x^3 + 4 * x^2 + b * x + c
def single_and_double_root (h_distinct : r ≠ s) :=
  ∃ (P : ℝ → ℝ), (∀ (x : ℝ), P x = (x - r) * (x - s)^2)
def specific_value (P : ℝ → ℝ) (s : ℝ) := P (-2 * s) = 324

theorem sum_abs_c_eq_108 (h_distinct: r ≠ s)
  (h1: single_and_double_root h_distinct) 
  (h2: specific_value polynomial s) : 
  abs c = 90 ∨ abs c = 18 := sorry

-- Sum of all possible values of |c|
theorem sum_possible_abs_c :
  ∑ x in {90, 18}, x = 108 := sorry

end sum_abs_c_eq_108_sum_possible_abs_c_l630_630896


namespace solve_using_factoring_method_l630_630075

theorem solve_using_factoring_method (x : ℝ) : (5 * x - 1)^2 = 3 * (5 * x - 1) → (5 * x - 1) * (5 * x - 4) = 0 :=
by
  intro h
  have h_eq : (5 * x - 1)^2 - 3 * (5 * x - 1) = 0 := by sorry
  exact h

end solve_using_factoring_method_l630_630075


namespace rearrange_ways_l630_630371

theorem rearrange_ways :
  (∃ f : Fin 12 → Nat,
    (∀ i, f i ∈ Finset.range 1 13) ∧
    (∀ i, (f i + f ((i + 1) % 12) + f ((i + 2) % 12)) % 3 = 0) ∧
    Function.bijective f) →
  Nat.card {f : Fin 12 → Nat // 
    (∀ i, f i ≥ 1 ∧ f i ≤ 12) ∧
    (∀ i, (f i + f ((i + 1) % 12) + f ((i + 2) % 12)) % 3 = 0) ∧
    Function.bijective f} = 82944 :=
sorry

end rearrange_ways_l630_630371


namespace decimal_to_binary_equivalent_123_l630_630536

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630536


namespace number_of_rectangles_l630_630276

theorem number_of_rectangles (H_lines V_lines : ℕ) (H_lines = 5) (V_lines = 6) :
  (Nat.choose H_lines 2) * (Nat.choose V_lines 2) = 150 :=
by
  sorry

end number_of_rectangles_l630_630276


namespace tangent_circle_properties_l630_630065

noncomputable section

variables {O1 O2 E1 E2 F1 F2 A1 A2 B1 B2 C1 C2 D1 D2 : Type} {r1 r2 a : ℝ}

/-- Prove that there exist circles, centered at the midpoint of O1 and O2, 
passing through the specified points with given radii. -/
theorem tangent_circle_properties 
    (midpoint : O1 × O2 → Type)
    (circle_radius_a : ∀ p : midpoint(O1, O2), ℝ)
    (circle_radius_ext : ∀ p : midpoint(O1, O2), ℝ)
    (circle_radius_int : ∀ p : midpoint(O1, O2), ℝ)
    (H1 : circle_radius_a(midpoint(O1,O2)) = a)
    (H2 : circle_radius_ext(midpoint(O1,O2)) = sqrt(a^2 + r1 * r2))
    (H3 : circle_radius_int(midpoint(O1,O2)) = sqrt(a^2 - r1 * r2))
    (condition1 : E1 = midpoint(O1, O2))
    (condition2 : E2 = midpoint(O1, O2))
    (condition3 : F1 = midpoint(O1, O2))
    (condition4 : F2 = midpoint(O1, O2)) :
  ∃ (radius_a radius_ext radius_int : ℝ), 
    radius_a = a ∧ radius_ext = sqrt(a^2 + r1 * r2) ∧ radius_int = sqrt(a^2 - r1 * r2) := 
  sorry

end tangent_circle_properties_l630_630065


namespace sin_pi_minus_2alpha_l630_630346

theorem sin_pi_minus_2alpha (α : ℝ) (h1 : Real.sin (π / 2 + α) = -3 / 5) (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (π - 2 * α) = -24 / 25 := by
  sorry

end sin_pi_minus_2alpha_l630_630346


namespace perpendicular_vectors_dot_product_zero_l630_630878

theorem perpendicular_vectors_dot_product_zero (m : ℝ) :
  let a := (1, 2)
  let b := (m + 1, -m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 1 :=
by
  intros a b h_eq
  sorry

end perpendicular_vectors_dot_product_zero_l630_630878


namespace casper_total_ways_l630_630632

theorem casper_total_ways (n : ℕ) (h : n = 8) : (n * (n - 1)) = 56 := 
by
  rw [h]
  simp
  exact show 8 * 7 = 56, from rfl

end casper_total_ways_l630_630632


namespace total_calories_l630_630017

theorem total_calories (chips : ℕ) (cheezits : ℕ) (chip_calories_total : ℕ) (cheezit_ratio : ℚ) 
  (h1 : chips = 10) 
  (h2 : cheezits = 6) 
  (h3 : chip_calories_total = 60)
  (h4 : cheezit_ratio = 1/3) :
  ∃ (total_calories : ℕ), total_calories = 108 :=
by
  let chip_calories := chip_calories_total / chips
  let cheezit_calories := chip_calories + (cheezit_ratio * chip_calories).toNat
  let total_cheezit_calories := cheezit_calories * cheezits
  let total_chip_calories := chip_calories_total
  let total_calories := total_chip_calories + total_cheezit_calories
  use total_calories
  sorry

end total_calories_l630_630017


namespace marching_band_total_weight_l630_630829

def weight_trumpets := 5
def weight_clarinets := 5
def weight_trombones := 10
def weight_tubas := 20
def weight_drums := 15

def count_trumpets := 6
def count_clarinets := 9
def count_trombones := 8
def count_tubas := 3
def count_drums := 2

theorem marching_band_total_weight :
  (count_trumpets * weight_trumpets) + (count_clarinets * weight_clarinets) + (count_trombones * weight_trombones) + 
  (count_tubas * weight_tubas) + (count_drums * weight_drums) = 245 :=
by
  sorry

end marching_band_total_weight_l630_630829


namespace mrs_hilt_apple_pies_l630_630426

def total_pies (rows : ℕ) (pies_per_row : ℕ) : ℕ :=
  rows * pies_per_row

theorem mrs_hilt_apple_pies
  (pecan_pies : ℕ) (rows : ℕ) (pies_per_row : ℕ) (apple_pies : ℕ)
  (h_baked_pecan : pecan_pies = 16)
  (h_wants_to_arrange : pies_per_row = 5)
  (h_total_rows : rows = 6) :
  apple_pies = 14 :=
by
  let total := total_pies rows pies_per_row
  have h_total_pies : total = 30 := by
    rw [h_total_rows, h_wants_to_arrange]
    exact rfl
  have h_apples : apple_pies = total - pecan_pies := by sorry
  rw [h_total_pies, h_baked_pecan] at h_apples
  exact h_apples

end mrs_hilt_apple_pies_l630_630426


namespace max_m_subset_triangle_property_l630_630247

theorem max_m_subset_triangle_property :
  ∃ m, (∀ s ⊆ (set.Icc 5 m), (set.card s = 11) → (∃ a b c ∈ s, a + b > c)) ∧ (m = 499) :=
sorry

end max_m_subset_triangle_property_l630_630247


namespace math_problem_conditions_l630_630283

variable (f : ℝ → ℝ)

theorem math_problem_conditions (h1 : ∀ x y : ℝ, f(x - y) - f(x + y) = f(x + 1) * f(y + 1))
                               (h2 : f(0) ≠ 0) :
  (f(1) = 0) ∧ (f(3) = f(-1)) ∧ (∑ k in (Finset.range 23).succ, f(k + 1) = -2) := 
by
  sorry

end math_problem_conditions_l630_630283


namespace ratio_of_radii_l630_630221

noncomputable def volume_ratio_radii : Prop :=
  let V_L : ℝ := 450 * Real.pi in
  let V_S : ℝ := 0.08 * V_L in
  let V_M : ℝ := 0.27 * V_L in
  let R : ℝ := (3 * V_L / (4 * Real.pi))^(1 / 3) in
  let r : ℝ := (3 * V_S / (4 * Real.pi))^(1 / 3) in
  let s : ℝ := (3 * V_M / (4 * Real.pi))^(1 / 3) in
  (r / R = 3 / 7) ∧ (s / R = 9 / 14)

theorem ratio_of_radii : volume_ratio_radii := 
  by 
  sorry

end ratio_of_radii_l630_630221


namespace least_positive_multiple_of_45_with_product_multiple_of_45_l630_630551

def is_multiple_of_45 (n : ℕ) : Prop :=
  n % 45 = 0

def product_of_digits (n : ℕ) : ℕ :=
  n.foldl (λ prod digit, prod * (digit.toNat % 10)) 1

def is_positive_multiple_of_45 (n : ℕ) : Prop :=
  product_of_digits n % 45 = 0

theorem least_positive_multiple_of_45_with_product_multiple_of_45 :
  ∃ n, is_multiple_of_45 n ∧ is_positive_multiple_of_45 n ∧ n = 945 :=
by
  sorry

end least_positive_multiple_of_45_with_product_multiple_of_45_l630_630551


namespace probability_females_not_less_than_males_l630_630211

noncomputable def prob_female_not_less_than_male : ℚ :=
  let total_students := 5
  let females := 2
  let males := 3
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose females 2 + females * males
  favorable_combinations / total_combinations

theorem probability_females_not_less_than_males (total_students females males : ℕ) :
  total_students = 5 → females = 2 → males = 3 →
  prob_female_not_less_than_male = 7 / 10 :=
by intros; sorry

end probability_females_not_less_than_males_l630_630211


namespace BD_passes_through_midpoint_CM_l630_630840

variables {A B C D M : Type*}
variables [geometry A B C D M] 
variable (trapezoid : Trapezoid A B C D)
variable (midpoint_AB : Midpoint M A B)
variable (AD_eq_2BC : AD = 2 * BC)
variable (midpoint_CM : Midpoint N C M)

theorem BD_passes_through_midpoint_CM :
  line BD passes through N :=
sorry

end BD_passes_through_midpoint_CM_l630_630840


namespace number_of_elements_in_N_l630_630788

def M : Set ℕ := {1, 2, 3, 4}
def N : Set (Set ℕ) := {P | P ⊆ M}
def number_of_elements_in_set (S : Set (Set ℕ)) : ℕ := S.count

theorem number_of_elements_in_N : number_of_elements_in_set N = 16 :=
by 
  sorry

end number_of_elements_in_N_l630_630788


namespace original_decimal_l630_630052

variable (x : ℝ)

theorem original_decimal (h : x - x / 100 = 1.485) : x = 1.5 :=
sorry

end original_decimal_l630_630052


namespace sqrt_defined_iff_nonneg_l630_630681

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 3)) ↔ x ≥ -3 :=
by
  sorry

end sqrt_defined_iff_nonneg_l630_630681


namespace storks_comparison_l630_630195

theorem storks_comparison (sparrows pigeons_initial crows storks_initial pigeons_additional storks_additional : ℕ)
  (hsparrows : sparrows = 12)
  (hpigeons_initial : pigeons_initial = 5)
  (hcrows : crows = 9)
  (hstorks_initial : storks_initial = 8)
  (hpigeons_additional : pigeons_additional = 4)
  (hstorks_additional : storks_additional = 15)
  : let pigeons_total := pigeons_initial + pigeons_additional
    let storks_total := storks_initial + storks_additional
    in storks_total - (sparrows + pigeons_total + crows) = -7 := by
  -- Declaring variables in a let statement
  let pigeons_total := pigeons_initial + pigeons_additional
  let storks_total := storks_initial + storks_additional
  -- Proving the statement with a placeholder for the proof
  sorry

end storks_comparison_l630_630195


namespace correct_diff_result_l630_630960

theorem correct_diff_result :
  deriv (λ x : ℝ, x / exp x) = λ x, (1 - x) / exp x :=
by
  sorry

end correct_diff_result_l630_630960


namespace sin_6phi_l630_630347

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * (Real.sqrt 8)) / 5) : 
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 :=
by
  sorry

end sin_6phi_l630_630347


namespace exist_triangle_no_point_on_side_l630_630296

theorem exist_triangle_no_point_on_side (S : Set (Point Plane)) (n : ℕ) 
  (h₀ : n ≥ 5) (h₁ : ∀ p ∈ S, p.color = Red ∨ p.color = Black)
  (h₂ : ∀ c, ∃ points : Finset (Point Plane), points.card ≥ 3 ∧ points ⊆ S ∧
    ∀ (p₁ p₂ p₃ ∈ points), ¬ (collinear p₁ p₂ p₃ ∧ p₁.color = p₂.color ∧ p₂.color = p₃.color)) :
  ∃ (t : Triangle (Point Plane)), (∀ p ∈ t.vertices, p.color = t.vertices.val.head.color) ∧
  ∃ s ∈ t.edges, ∀ p ∈ s, p ∉ S :=
by
  sorry

end exist_triangle_no_point_on_side_l630_630296


namespace water_needed_for_reaction_l630_630700

theorem water_needed_for_reaction (n : ℕ) (h1 : n = 1) : 
  let m := 2 * n in m = 2 := 
by
  sorry

end water_needed_for_reaction_l630_630700


namespace fifth_row_first_five_positions_l630_630262

noncomputable def grid_fifth_row : list (list (ℕ)) :=
  [[_, _, _, _, _, _],
   [_, _, _, _, _, _],
   [_, _, _, _, _, _],
   [_, _, _, _, _, _],
   [1, 5, 9, 9, 2, _], -- Here _ is a placeholder for the unknown sixth position.
   []]

theorem fifth_row_first_five_positions :
  (grid_fifth_row.nth 4).map (fun row => row.take 5) = some [1, 5, 9, 9, 2] :=
by
  sorry

end fifth_row_first_five_positions_l630_630262


namespace intersection_points_l630_630764

noncomputable def f (x : ℝ) : ℝ := 
if 0 ≤ x ∧ x ≤ 1 then x^2 else sorry  -- We use sorry to fill in the definition for x outside [0,1] while complying with the periodic and even nature conditions

lemma f_even : ∀ x : ℝ, f x = f (-x) :=
sorry  -- Proof of f being even

lemma f_periodic : ∀ x : ℝ, f (x + 2) = f x :=
sorry  -- Proof of f being periodic of period 2

theorem intersection_points (a : ℝ) :
    (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = x1 + a ∧ f x2 = x2 + a) ↔ (∃ k : ℤ, a = 2 * k ∨ a = 2 * k - 1) :=
begin
  sorry,  -- Proof that the line y = x + a intersects the curve y = f(x) at exactly 2 distinct points iff a = 2k or a = 2k-1
end

end intersection_points_l630_630764


namespace problem_statement_l630_630860

variables {a b c p q r : ℝ}

-- Given conditions
axiom h1 : 19 * p + b * q + c * r = 0
axiom h2 : a * p + 29 * q + c * r = 0
axiom h3 : a * p + b * q + 56 * r = 0
axiom h4 : a ≠ 19
axiom h5 : p ≠ 0

-- Statement to prove
theorem problem_statement : 
  (a / (a - 19)) + (b / (b - 29)) + (c / (c - 56)) = 1 :=
sorry

end problem_statement_l630_630860


namespace n_values_for_prime_n_n_plus_1_l630_630733

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∈ (Set.range (1 : ℕ)), m ∣ n → m = 1 ∨ m = n

def valid_n_values : Finset ℕ :=
  {n ∈ (Finset.range 16).filter (λ n, is_prime (n^n + 1))}

theorem n_values_for_prime_n_n_plus_1 :
  valid_n_values = {1, 2, 4} :=
by
  sorry

end n_values_for_prime_n_n_plus_1_l630_630733


namespace complement_intersection_l630_630799

open Set

theorem complement_intersection (U A B : Set ℕ) (h1 : U = {x | x < 9 ∧ x > 0}) (h2 : A = {1, 2, 3, 4})
  (h3 : B = {3, 4, 5, 6}) : (U \ A) ∩ (U \ B) = {7, 8} :=
by
  have hU : U = {1, 2, 3, 4, 5, 6, 7, 8} := sorry
  have hCA : (U \ A) = {5, 6, 7, 8} := sorry
  have hCB : (U \ B) = {1, 2, 7, 8} := sorry
  show (U \ A) ∩ (U \ B) = {7, 8} from sorry

end complement_intersection_l630_630799


namespace age_difference_l630_630897

theorem age_difference (P M Mo : ℕ) (h1 : P = (3 * M) / 5) (h2 : Mo = (4 * M) / 3) (h3 : P + M + Mo = 88) : Mo - P = 22 := 
by sorry

end age_difference_l630_630897


namespace angle_y_value_l630_630377

theorem angle_y_value 
  (h1 : ∃ A B C D : Type, ∀ (ABC_plane : linear ABC),
        (D_below_ABC : below ABC D) 
          (angle_ABC : angle A B C = 152) 
            (angle_ACD : angle A C D = 104)) :
  ∃ y : Type, angle BCD = y ∧ y = 48 :=
  sorry

end angle_y_value_l630_630377


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630150

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l630_630150


namespace trigonometric_identity_l630_630558

theorem trigonometric_identity : 
  sin (10 * real.pi / 180) * cos (50 * real.pi / 180) + cos (10 * real.pi / 180) * sin (50 * real.pi / 180)
  = sqrt 3 / 2 :=
by 
  sorry

end trigonometric_identity_l630_630558


namespace candy_from_sister_is_5_l630_630278

noncomputable def candy_received_from_sister (candy_from_neighbors : ℝ) (pieces_per_day : ℝ) (days : ℕ) : ℝ :=
  pieces_per_day * days - candy_from_neighbors

theorem candy_from_sister_is_5 :
  candy_received_from_sister 11.0 8.0 2 = 5.0 :=
by
  sorry

end candy_from_sister_is_5_l630_630278


namespace complex_pure_imaginary_a_l630_630810

theorem complex_pure_imaginary_a (a : ℝ) (z : ℂ) (h : z = (Complex.mk 1 1) / (Complex.mk 1 a)) : 
  Im (z) = z → a = -1 :=
by
  sorry

end complex_pure_imaginary_a_l630_630810


namespace matches_in_alignment_l630_630869

def binary_digit := ℕ -- use 0 and 1 for binary digits

def valid_binary_number (digits : List binary_digit) (length : ℕ) : Prop :=
length digits = length ∧ ∀ d ∈ digits, d = 0 ∨ d = 1

def binary_number_with_exactly_n_zeros (digits : List binary_digit) (length n : ℕ) : Prop :=
valid_binary_number digits length ∧ digits.count (λ d, d = 0) = n

theorem matches_in_alignment :
  ∀ (A B : List binary_digit), 
  binary_number_with_exactly_n_zeros A 20 10 → valid_binary_number B 20 → 
  ∃ (i : ℕ), 0 ≤ i ∧ i ≤ 20 ∧ (A.zip (B.drop i ++ B.take i)).count (λ p, p.1 = p.2) ≥ 10 :=
by sorry

end matches_in_alignment_l630_630869


namespace probability_red_side_given_observed_l630_630981

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end probability_red_side_given_observed_l630_630981


namespace number_of_possible_values_for_s_l630_630917

noncomputable def is_near_fraction (s : ℚ) (frac : ℚ) : Prop :=
  abs (s - frac) ≤ abs (s - 1/2) ∧ abs (s - frac) ≤ abs (s - 1/3)

noncomputable def possible_vals_count (lower upper : ℚ) : ℕ :=
  (upper * 10000).to_nat - (lower * 10000).to_nat + 1

theorem number_of_possible_values_for_s :
  ∃ (s : ℚ), (s = 2/5) ∧ (is_near_fraction s (2/5)) →
  (possible_vals_count 0.3667 0.45) = 834 :=
by
  sorry

end number_of_possible_values_for_s_l630_630917


namespace handshake_problem_l630_630233

-- Given the conditions that there are 8 married couples,
-- and each participant except one (who is allergic to physical contact) 
-- shakes hands with everyone else except their own spouse.
def handshake_count (num_couples : ℕ) (num_allergic : ℕ) : ℕ :=
  let num_people := 2 * num_couples in
  let num_handshakes_per_person := (num_people - 1 - num_allergic) in
  num_people * num_handshakes_per_person / 2

-- The theorem states that for 8 couples and 1 person allergic to physical contact,
-- there are 104 total handshakes
theorem handshake_problem : handshake_count 8 1 = 104 := by
  sorry

end handshake_problem_l630_630233


namespace f_value_at_3_l630_630102

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def periodic_shift (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (x + 2) = f x + 2

theorem f_value_at_3 (h_odd : odd_function f) (h_value : f (-1) = 1/2) (h_periodic : periodic_shift f) : 
  f 3 = 3 / 2 := 
sorry

end f_value_at_3_l630_630102


namespace no_common_root_of_quadratics_l630_630095

theorem no_common_root_of_quadratics (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, (x₀^2 + b * x₀ + c = 0 ∧ x₀^2 + a * x₀ + d = 0) := 
by
  sorry

end no_common_root_of_quadratics_l630_630095


namespace determine_r_as_m_approaches_0_l630_630027

noncomputable def L (m : ℝ) : ℝ := -real.sqrt (m + 8)
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem determine_r_as_m_approaches_0 :
  filter.tendsto r (filter.nhds_within 0 (set.Ioo (-8 : ℝ) 8)) (filter.nhds (1 / (2 * real.sqrt 2))) :=
sorry

end determine_r_as_m_approaches_0_l630_630027


namespace enjoyable_gameplay_total_l630_630011

/-
James gets bored with his game so decides to play a different one. 
The game promises 100 hours of gameplay but 80% of that is boring grinding. 
However, the expansion does add another 30 hours of enjoyable gameplay. 
How much enjoyable gameplay does James get? 
-/

def original_gameplay : ℕ := 100
def boring_percentage : ℕ := 80
def enjoyable_percentage : ℕ := 20
def expansion_gameplay : ℕ := 30

theorem enjoyable_gameplay_total : 
  let enjoyable_original := (enjoyable_percentage * original_gameplay) / 100 in
  enjoyable_original + expansion_gameplay = 50 :=
by
  let enjoyable_original := (enjoyable_percentage * original_gameplay) / 100
  sorry

end enjoyable_gameplay_total_l630_630011


namespace part_a_part_b_l630_630243

noncomputable def P1 : Polynomial ℤ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4
noncomputable def P1_prime : Polynomial ℤ := 6*x^5 - 24*x^3 - 12*x^2 + 18*x + 12
noncomputable def Q1 : Polynomial ℤ := x^4 + x^3 - 3*x^2 - 5*x - 2
noncomputable def R1 : Polynomial ℤ := x^2 - x - 2

theorem part_a : R1 = P1 / Q1 := sorry

noncomputable def P2 : Polynomial ℤ := x^5 + x^4 - 2*x^3 - 2*x^2 + x + 1
noncomputable def P2_prime : Polynomial ℤ := 5*x^4 + 4*x^3 - 6*x^2 - 4*x + 1
noncomputable def Q2 : Polynomial ℤ := x^3 + x^2 - x - 1
noncomputable def R2 : Polynomial ℤ := x^2 - 1

theorem part_b : R2 = P2 / Q2 := sorry

end part_a_part_b_l630_630243


namespace ratio_of_areas_l630_630435

noncomputable def angle (α β γ : Type) := sorry

-- Definitions based on conditions
def equilateral_triangle (A B C : Type) : Prop :=
  sorry -- Define equilateral triangle properties

def on_side (D C : Type) : Prop :=
  sorry -- Define a point D lying on side AC

def angle_measure (D B C : Type) (deg : ℝ) : Prop :=
  sorry -- Measure of angle DBC
  
-- The problem statement 
theorem ratio_of_areas 
  (A B C D : Type) 
  (h1 : equilateral_triangle A B C) 
  (h2 : on_side D C) 
  (h3 : angle_measure D B C 30) 
: ratio_of_areas_triangle_A B D C = 1 - real.sqrt 3 :=
sorry

end ratio_of_areas_l630_630435


namespace percentage_of_fruits_in_good_condition_l630_630631

theorem percentage_of_fruits_in_good_condition (total_oranges total_bananas rotten_percentage_oranges rotten_percentage_bananas : ℕ)
    (h1 : total_oranges = 600)
    (h2 : total_bananas = 400)
    (h3 : rotten_percentage_oranges = 15)
    (h4 : rotten_percentage_bananas = 8) :
    ((total_oranges - (rotten_percentage_oranges * total_oranges / 100)) + 
     (total_bananas - (rotten_percentage_bananas * total_bananas / 100))) 
     * 100 / (total_oranges + total_bananas) = 87.8 := 
begin
  sorry
end

end percentage_of_fruits_in_good_condition_l630_630631


namespace base_2_representation_of_123_l630_630542

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630542


namespace owner_overtakes_thief_l630_630636

theorem owner_overtakes_thief :
  ∀ (speed_thief speed_owner : ℕ) (time_theft_discovered : ℝ), 
    speed_thief = 45 →
    speed_owner = 50 →
    time_theft_discovered = 0.5 →
    (time_theft_discovered + (45 * 0.5) / (speed_owner - speed_thief)) = 5 := 
by
  intros speed_thief speed_owner time_theft_discovered h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end owner_overtakes_thief_l630_630636


namespace midpoint_scaled_rotated_l630_630833

def point1 : Complex := -5 + 7i
def point2 : Complex := 7 - 3i

theorem midpoint_scaled_rotated :
  2 * complex.i * ((point1 + point2) / 2) = -4 + 2 * complex.i := by
  sorry

end midpoint_scaled_rotated_l630_630833


namespace power_mod_l630_630162

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l630_630162


namespace correct_answer_l630_630684

-- Definitions of the groups
def group_1_well_defined : Prop := false -- Smaller numbers
def group_2_well_defined : Prop := true  -- Non-negative even numbers not greater than 10
def group_3_well_defined : Prop := true  -- All triangles
def group_4_well_defined : Prop := false -- Tall male students

-- Propositions representing the options
def option_A : Prop := group_1_well_defined ∧ group_4_well_defined
def option_B : Prop := group_2_well_defined ∧ group_3_well_defined
def option_C : Prop := group_2_well_defined
def option_D : Prop := group_3_well_defined

-- Theorem stating Option B is the correct answer
theorem correct_answer : option_B ∧ ¬option_A ∧ ¬option_C ∧ ¬option_D := by
  sorry

end correct_answer_l630_630684


namespace trailing_zeros_238_trailing_zeros_diff_238_236_l630_630107

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Function to count the number of trailing zeros in n!
def trailing_zeros (n : ℕ) : ℕ :=
  let count_factors x p := if x = 0 then 0 else x / p + count_factors (x / p) p 
  count_factors n 5

-- The statement of the proof problem
theorem trailing_zeros_238 :
  trailing_zeros 238 = 57 :=
sorry

theorem trailing_zeros_diff_238_236 :
  trailing_zeros 238 - trailing_zeros 236 = 0 :=
sorry

end trailing_zeros_238_trailing_zeros_diff_238_236_l630_630107


namespace base_2_representation_of_123_is_1111011_l630_630525

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630525


namespace question1_question2_l630_630319

-- Given conditions
variables {a_3 : ℝ} {S_3 : ℝ} {q m a_1 : ℝ}
def geometric_sequence (a_3 S_3 m : ℝ) := m^2 = a_3 * S_3
def conditions1 : a_3 = 3 / 2 := by sorry
def conditions2 : S_3 = 9 / 2 := by sorry

-- Proof problem translation
theorem question1 : geometric_sequence a_3 S_3 m → m = (3 * real.sqrt 3 / 2) ∨ m = -(3 * real.sqrt 3 / 2) :=
  by {intro hyp, sorry}

theorem question2 (q : ℝ) (h1 : a_3 = 3 / 2) (h2 : S_3 = 9 / 2) (h3 : geometric_sequence a_3 S_3 m) :
  a_1 = 3 / 2 ∨ a_1 = 6 :=
  by {sorry}

end question1_question2_l630_630319


namespace hexagon_internal_angle_A_l630_630369

theorem hexagon_internal_angle_A
  (B C D E F : ℝ) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) 
  (H : B + C + D + E + F + A = 720) : A = 120 := 
sorry

end hexagon_internal_angle_A_l630_630369


namespace count_distinct_products_l630_630397

def T := { x : ℕ | x ∣ 36000 ∧ x > 0 }

theorem count_distinct_products : 
  (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), ∃ a b, p.1 = a ∧ p.2 = b ∧ ¬(a = b) ∧ a ∈ T ∧ b ∈ T).card = 311 := sorry

end count_distinct_products_l630_630397


namespace club_additional_members_l630_630598

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l630_630598


namespace hypotenuse_length_l630_630047

theorem hypotenuse_length :
  ∃ (c : ℝ), 
  let D := (∃ (AD : ℝ), AD = Real.cos (Real.pi / 4)) in
  let E := (∃ (AE : ℝ), AE = Real.sin (Real.pi / 4)) in 
  ∃ (BC : ℝ), 
  (BC / 4 = D ∧ BC / 4 = E) ∧ 
  BC = c ∧ 
  c = 8/11 :=
sorry

end hypotenuse_length_l630_630047


namespace RX_perpendicular_BC_l630_630390

noncomputable def triangle (A B C : Point) : Prop := sorry -- Define a triangle given three noncollinear points

noncomputable def is_largest_angle (A B C : Point) (ABC : triangle A B C) : Prop := sorry -- Define that ∠ABC is the largest angle in the triangle

noncomputable def circumcenter (A B C R : Point) (ABC : triangle A B C) : Prop := sorry -- Define R as the circumcenter of the triangle

noncomputable def is_circumcircle (A B X R : Point) : Prop := sorry -- Define that points A, B, X, and R are concyclic (lie on the same circle)

noncomputable def perpendicular (X R : Point) (BC : Line) : Prop := sorry -- Define that RX is perpendicular to line BC

theorem RX_perpendicular_BC 
  (A B C R X : Point)
  (ABC : triangle A B C)
  (larg : is_largest_angle A B C ABC)
  (circum : circumcenter A B C R ABC)
  (circumcircle : is_circumcircle A R B X) :
  perpendicular R X (Line.mk B C) := 
  sorry

end RX_perpendicular_BC_l630_630390


namespace sandy_books_cost_l630_630442

theorem sandy_books_cost :
  ∀ (x : ℕ),
  (1280 + 880) / (x + 55) = 18 → 
  x = 65 :=
by
  intros x h
  sorry

end sandy_books_cost_l630_630442


namespace car_speed_l630_630198

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end car_speed_l630_630198


namespace count_valid_pairs_1651_l630_630421

def valid_pairs (m n : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 3000 ∧ 3^n < 2^m ∧ 2^(m+2) < 3^(n+1)

theorem count_valid_pairs_1651 :
  {p : ℕ × ℕ // valid_pairs p.1 p.2}.card = 1651 :=
sorry

end count_valid_pairs_1651_l630_630421


namespace hamburger_combinations_l630_630336

theorem hamburger_combinations :
  let condiments_choice := 2 ^ 10 in
  let patties_choice := 3 in
  let bun_choice := 2 in
  condiments_choice * patties_choice * bun_choice = 6144 :=
by
  sorry

end hamburger_combinations_l630_630336


namespace max_combined_subject_marks_l630_630650

theorem max_combined_subject_marks :
  let total_marks_math := (130 + 14) / 0.36,
      total_marks_physics := (120 + 20) / 0.40,
      total_marks_chemistry := (160 + 10) / 0.45,
      max_total_marks := total_marks_math + total_marks_physics + total_marks_chemistry in
  ⌊(total_marks_math + total_marks_physics + total_marks_chemistry)⌋ = 1127 :=
by
  -- The proof should be written here
  sorry

end max_combined_subject_marks_l630_630650


namespace weight_in_4th_minute_l630_630641

noncomputable def weightlifter_lift (initial_lh : ℕ) (initial_rh : ℕ) (add_lh : ℕ) (add_rh : ℕ) (decline : ℕ) (t : ℕ) : ℕ :=
  if t < 3 then
    initial_lh + initial_rh + t * (add_lh + add_rh)
  else
    initial_lh + initial_rh + 3 * (add_lh + add_rh) - (t - 3) * decline

theorem weight_in_4th_minute :
  weightlifter_lift 12 18 4 6 5 4 = 55 :=
by
  have h : weightlifter_lift 12 18 4 6 5 4 = (12 + 18) + 3 * (4 + 6) - (4 - 3) * 5 := rfl
  rw h
  norm_num
  sorry

end weight_in_4th_minute_l630_630641


namespace equilateral_triangle_distance_l630_630255

/-- Let \( \triangle ABC \) be an equilateral triangle with side length 400.
    Points \( P \) and \( Q \) lie outside the plane of \( \triangle ABC \) on opposite sides,
    with \( PA = PB = PC \) and \( QA = QB = QC \).
    The planes of \( \triangle PAB \) and \( \triangle QAB \) form a \(150^\circ\) dihedral angle.
    There is a point \( O \) whose distance from each of \( A, B, C, P, Q \) is \( d \).
    Prove that \( d = 200 \). -/
theorem equilateral_triangle_distance (A B C P Q O : ℝ × ℝ × ℝ)
    (d : ℝ)
    (hABC : dist A B = 400 ∧ dist B C = 400 ∧ dist A C = 400)
    (hPA : dist P A = dist P B ∧ dist P B = dist P C)
    (hQA : dist Q A = dist Q B ∧ dist Q B = dist Q C)
    (hABP : ∃ D, Plane D A B P)
    (hABQ : ∃ E, Plane E A B Q)
    (hangle : dihedral_angle hABP hABQ = 150)
    (hO : dist O A = d ∧ dist O B = d ∧ dist O C = d ∧ dist O P = d ∧ dist O Q = d) :
    d = 200 :=
sorry

end equilateral_triangle_distance_l630_630255


namespace hyperbola_right_branch_with_specific_properties_l630_630975

noncomputable def complex_hyperbola_shape (z : ℂ) : Prop :=
  abs (z - 3) = abs (z + 3) - 1

theorem hyperbola_right_branch_with_specific_properties (z : ℂ) :
  complex_hyperbola_shape z → (∃ a b: ℂ, abs z = abs (z - a) - b) :=
sorry

end hyperbola_right_branch_with_specific_properties_l630_630975


namespace monotone_increasing_interval_tan_cos_expr_l630_630193

-- Proof problem for Question 1: Monotonicity of the function on a specific interval
theorem monotone_increasing_interval (x : ℝ) 
  (h1 : -2 * Real.pi ≤ x ∧ x ≤ 2 * Real.pi) : 
  (-3 * Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) → 
  ∀ x₁ x₂, (x₁ < x₂ ∧ -3 * Real.pi / 2 ≤ x₁ ∧ x₂ ≤ Real.pi / 2) → 
  sin (1/2 * x₁ + Real.pi / 4) ≤ sin (1/2 * x₂ + Real.pi / 4) := by sorry

-- Proof problem for Question 2: Trigonometric expression evaluation
theorem tan_cos_expr : 
  ∑ \text{real_valued} 
  tan_real := λ x, Real.tan (Real.pi * x / 180)
  cos_real := λ x, Real.cos (Real.pi * x / 180)
  sin_real := λ x, Real.sin (Real.pi * x / 180)
  exponent := λ x, Real.sqrt (Real.pi * x / 180)
  angle_70 := Real.pi * 70 / 180
  angle_10 := Real.pi * 10 / 180
  angle_20 := Real.pi * 20 / 180
  tan_real angle_70 * cos_real angle_10 * (Real.sqrt 3 * tan_real angle_20 - 1) = -1 := by sorry

end monotone_increasing_interval_tan_cos_expr_l630_630193


namespace number_of_m_tuples_l630_630948

theorem number_of_m_tuples (n m : ℕ) (hm : 1 ≤ m) (hn : m ≤ n) :
  {x : (fin m → ℕ) // ∀ i, 0 < x i ∧ (finset.univ.sum x = n)}.card = (nat.choose (n-1) (m-1)) :=
sorry

end number_of_m_tuples_l630_630948


namespace count_m_tuples_l630_630946

open Nat

-- Define the strict positive natural numbers
def NatPos := { n : ℕ // n > 0 }

-- Define the number of m-tuples satisfying the given sum condition
noncomputable def num_m_tuples_satisfying_sum (m n : ℕ) : ℕ :=
  if m > n then 0 else Nat.choose (n - 1) (m - 1)

-- Statement of the theorem
theorem count_m_tuples (m n : ℕ) (h1 : m > 0) (h2 : n >= m) :
  (∃ (x : Fin m → NatPos), (∑ i, x i.val) = n) ↔ num_m_tuples_satisfying_sum m n = Nat.choose (n - 1) (m - 1) :=
sorry

end count_m_tuples_l630_630946


namespace quadratic_inequality_l630_630732

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 - a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end quadratic_inequality_l630_630732


namespace max_value_expr_l630_630032

theorem max_value_expr (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (θ : ℝ) (h_θ1 : 0 ≤ θ) (h_θ2 : θ ≤ (real.pi / 2)) :
  ∃ x : ℝ, 2 * (a - x) * (x + real.cos(θ) * real.sqrt(x^2 + b^2)) ≤ a^2 + real.cos(θ)^2 * b^2 :=
sorry

end max_value_expr_l630_630032


namespace ratio_Cheryl_C_to_Cyrus_Y_l630_630239

noncomputable def Cheryl_C : ℕ := 126
noncomputable def Madeline_M : ℕ := 63
noncomputable def Total_pencils : ℕ := 231
noncomputable def Cyrus_Y : ℕ := Total_pencils - Cheryl_C - Madeline_M

theorem ratio_Cheryl_C_to_Cyrus_Y : 
  Cheryl_C = 2 * Madeline_M → 
  Madeline_M + Cheryl_C + Cyrus_Y = Total_pencils → 
  Cheryl_C / Cyrus_Y = 3 :=
by
  intros h1 h2
  sorry

end ratio_Cheryl_C_to_Cyrus_Y_l630_630239


namespace bob_wins_even_n_l630_630143

def game_of_islands (n : ℕ) (even_n : n % 2 = 0) : Prop :=
  ∃ strategy : (ℕ → ℕ), -- strategy is a function representing each player's move
    ∀ A B : ℕ → ℕ, -- A and B represent the moves of Alice and Bob respectively
    (A 0 + B 1) = n → (A (A 0 + 1) ≠ B (A 0 + 1)) -- Bob can always mirror Alice’s move.

theorem bob_wins_even_n (n : ℕ) (h : n % 2 = 0) : game_of_islands n h :=
sorry

end bob_wins_even_n_l630_630143


namespace number_of_outfits_l630_630070

-- Definitions based on conditions
def trousers : ℕ := 4
def shirts : ℕ := 8
def jackets : ℕ := 3
def belts : ℕ := 2

-- The statement to prove
theorem number_of_outfits : trousers * shirts * jackets * belts = 192 := by
  sorry

end number_of_outfits_l630_630070


namespace tan_value_l630_630309

theorem tan_value (α : ℝ) (h1 : sin α - cos α = -√5 / 5) (h2 : π < α) (h3 : α < 3 * π / 2) : tan α = 2 := 
by 
  sorry

end tan_value_l630_630309


namespace remainder_3_pow_2023_mod_5_l630_630158

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l630_630158


namespace pentagon_area_l630_630797

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℕ := 
  let area_triangle := (1/2) * a * b
  let area_trapezoid := (1/2) * (c + e) * d
  area_triangle + area_trapezoid

theorem pentagon_area : area_of_pentagon 18 25 30 28 25 = 995 :=
by sorry

end pentagon_area_l630_630797


namespace rectangle_perimeter_l630_630911

-- Definitions based on the conditions
def length : ℕ := 15
def width : ℕ := 8

-- Definition of the perimeter function
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Statement of the theorem we need to prove
theorem rectangle_perimeter : perimeter length width = 46 := by
  sorry

end rectangle_perimeter_l630_630911


namespace monotonicity_of_f_range_of_a_l630_630672

noncomputable def f (a x : ℝ) : ℝ := 2 * a * Real.log x - x^2 + a

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → f a x ≤ f a (x - 1)) ∧ 
           (a > 0 → ((x < Real.sqrt a → f a x ≤ f a (x + 1)) ∨ 
                     (x > Real.sqrt a → f a x ≥ f a (x - 1))))) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 1) := sorry

end monotonicity_of_f_range_of_a_l630_630672


namespace determine_n_l630_630320

-- Definitions based on conditions
def a₁ := 1
def d := 3
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- Target proof statement
theorem determine_n : (a_n 100) = 298 :=
by
  let a₁ := 1
  let d := 3
  let a_n := λ n : ℕ, a₁ + (n - 1) * d
  show a_n 100 = 298
  sorry

end determine_n_l630_630320


namespace correct_ratio_l630_630606

def total_cans := 24
def cherry_soda_cans := 8

def orange_pop_cans := total_cans - cherry_soda_cans
def ratio_of_orange_to_cherry := orange_pop_cans / cherry_soda_cans

theorem correct_ratio : ratio_of_orange_to_cherry = 2 := by
  have eq1 : orange_pop_cans = 16 := rfl
  have eq2 : ratio_of_orange_to_cherry = 16 / 8 := rfl
  have eq3 : 16 / 8 = 2 := by norm_num
  rw [eq1, eq2, eq3]
  sorry

end correct_ratio_l630_630606


namespace bedroom_door_cost_ratio_l630_630845

noncomputable def cost_bedroom_door_ratio : Prop :=
  ∃ (B : ℝ),
  (3 * B + 2 * 20 = 70) ∧
  (B / 20 = 1 / 2)

theorem bedroom_door_cost_ratio :
  cost_bedroom_door_ratio :=
begin
  sorry
end

end bedroom_door_cost_ratio_l630_630845


namespace pentagon_area_l630_630903

noncomputable def area_of_pentagon (FG HIJ : ℝ) := 
  (9 / 2) * Real.sin (100 * Real.pi / 180) + 25

theorem pentagon_area : 
  area_of_pentagon 3 5 ≈ 29.42 := 
by
  sorry

end pentagon_area_l630_630903


namespace rectangular_plot_perimeter_l630_630466

theorem rectangular_plot_perimeter (w : ℝ) (P : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  (cost_per_meter = 6.5) →
  (total_cost = 1430) →
  (P = 2 * (w + (w + 10))) →
  (cost_per_meter * P = total_cost) →
  P = 220 :=
by
  sorry

end rectangular_plot_perimeter_l630_630466


namespace task_force_at_least_two_executives_l630_630099

open Nat

theorem task_force_at_least_two_executives :
  let total_members := 12
  let executives := 5
  let total_task_forces := Nat.choose total_members 5
  let zero_executive_forces := Nat.choose (total_members - executives) 5
  let one_executive_forces := Nat.choose executives 1 * Nat.choose (total_members - executives) 4
  total_task_forces - (zero_executive_forces + one_executive_forces) = 596 :=
by
  intros total_members executives total_task_forces zero_executive_forces one_executive_forces
  let total_members := 12
  let executives := 5
  let total_task_forces := Nat.choose total_members 5
  let zero_executive_forces := Nat.choose (total_members - executives) 5
  let one_executive_forces := Nat.choose executives 1 * Nat.choose (total_members - executives) 4
  sorry

end task_force_at_least_two_executives_l630_630099


namespace lashawn_twice_kymbrea_comics_l630_630389

theorem lashawn_twice_kymbrea_comics :
  ∀ (x : ℕ),
    (Kymbrea_books : ℕ → ℕ) (LaShawn_books : ℕ → ℕ),
    (Kymbrea_books = λ x, 50 + 3 * x) →
    (LaShawn_books = λ x, 20 + 7 * x) →
    LaShawn_books x = 2 * Kymbrea_books x → x = 80 :=
by
  intros x Kymbrea_books LaShawn_books hKymbrea hLaShawn hEq
  sorry

end lashawn_twice_kymbrea_comics_l630_630389


namespace fourth_term_geometric_progression_l630_630803

theorem fourth_term_geometric_progression (x : ℝ) (h : ∀ n : ℕ, 0 < n → 
  (x ≠ 0 ∧ (2 * (x) + 2 * (n - 1)) ≠ 0 ∧ (3 * (x) + 3 * (n - 1)) ≠ 0)
  → ((2 * x + 2) / x) = (3 * x + 3) / (2 * x + 2)) : 
  ∃ r : ℝ, r = -13.5 := 
by 
  sorry

end fourth_term_geometric_progression_l630_630803


namespace new_median_after_adding_eight_l630_630985

-- Definitions for the initial conditions
def mean (s : List ℝ) : ℝ := s.sum / s.length
def mode (s : List ℝ) : ℝ := 
  let grouped := s.groupBy id length
  (grouped.maximumBy (λ (xs : List ℝ) => xs.length)).head

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (≤)
  if sorted.length % 2 = 1 then
    sorted.get! (sorted.length / 2)
  else
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2

-- The formula for the mean of the original collection
def original_sum := 5 * 4.4

-- The Lean statement for the proof problem
theorem new_median_after_adding_eight
  (s : List ℝ)
  (h1 : s.length = 5)
  (h2 : mean s = 4.4)
  (h3 : mode s = 3)
  (h4 : median s = 4) :
  median (8 :: s) = 4.5 := 
sorry

end new_median_after_adding_eight_l630_630985


namespace high_quality_chip_prob_l630_630590

variable (chipsA chipsB chipsC : ℕ)
variable (qualityA qualityB qualityC : ℝ)
variable (totalChips : ℕ)

noncomputable def probability_of_high_quality_chip (chipsA chipsB chipsC : ℕ) (qualityA qualityB qualityC : ℝ) (totalChips : ℕ) : ℝ :=
  (chipsA / totalChips) * qualityA + (chipsB / totalChips) * qualityB + (chipsC / totalChips) * qualityC

theorem high_quality_chip_prob :
  let chipsA := 5
  let chipsB := 10
  let chipsC := 10
  let qualityA := 0.8
  let qualityB := 0.8
  let qualityC := 0.7
  let totalChips := 25
  probability_of_high_quality_chip chipsA chipsB chipsC qualityA qualityB qualityC totalChips = 0.76 :=
by
  sorry

end high_quality_chip_prob_l630_630590


namespace number_of_convex_polygons_l630_630338

def isConvex (polygon : List (ℕ × ℕ)) : Prop :=
  -- Definition for verifying the convexity of a given polygon

def validPolygonVertices : List (ℕ × ℕ) :=
  [(0, 5), (5, 5), (5, 0)] ++ -- additional vertices with non-negative integer coordinates
  [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (2, 2), (3, 3), (4, 4)]

def countConvexPolygons (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to count all the valid convex polygons forming with the vertices

theorem number_of_convex_polygons : countConvexPolygons validPolygonVertices = 77 := 
  sorry

end number_of_convex_polygons_l630_630338


namespace company_employees_count_l630_630368

theorem company_employees_count :
  (females : ℕ) ->
  (advanced_degrees : ℕ) ->
  (college_degree_only_males : ℕ) ->
  (advanced_degrees_females : ℕ) ->
  (110 = females) ->
  (90 = advanced_degrees) ->
  (35 = college_degree_only_males) ->
  (55 = advanced_degrees_females) ->
  (females - advanced_degrees_females + college_degree_only_males + advanced_degrees = 180) :=
by
  intros females advanced_degrees college_degree_only_males advanced_degrees_females
  intro h_females h_advanced_degrees h_college_degree_only_males h_advanced_degrees_females
  sorry

end company_employees_count_l630_630368


namespace intersect_at_one_point_l630_630464

/-- Given a triangle A1 A2 A3 with an inscribed circle touching the sides A2 A3, A3 A1,
and A1 A2 at S1, S2, and S3 respectively, and O1, O2, and O3 are the centers of the 
inscribed circles of triangles A1 S2 S3, A2 S3 S1, and A3 S1 S2 respectively,
prove that the lines O1 S1, O2 S2, and O3 S3 intersect at one point. -/
theorem intersect_at_one_point
  (A1 A2 A3 S1 S2 S3 O1 O2 O3 : ℝ^2)
  (h1 : inscribed_circle A1 A2 A3 S1 S2 S3)
  (h2 : is_center_of_inscribed_circle A1 S2 S3 O1)
  (h3 : is_center_of_inscribed_circle A2 S3 S1 O2)
  (h4 : is_center_of_inscribed_circle A3 S1 S2 O3) :
  ∃ P : ℝ^2, P ∈ Line O1 S1 ∧ P ∈ Line O2 S2 ∧ P ∈ Line O3 S3 :=
sorry

end intersect_at_one_point_l630_630464


namespace road_length_is_15_km_l630_630648

-- Define the constants and conditions
def initial_men : ℕ := 30
def extra_men : ℕ := 45
def total_men : ℕ := initial_men + extra_men
def total_days : ℕ := 300
def spent_days : ℕ := 100
def completed_work : ℝ := 2.5
def remaining_days : ℕ := total_days - spent_days

-- Define the work rate for initial men in spent days
def work_rate_initial : ℝ := completed_work / spent_days

-- Define the statement to prove the total length of the road
theorem road_length_is_15_km (L : ℝ) : 
  (L - completed_work) = work_rate_initial * (total_men / initial_men) * remaining_days 
  → L = 15 := 
by 
  sorry -- The proof steps are omitted as per instructions.

end road_length_is_15_km_l630_630648


namespace min_segments_for_octagon_perimeter_l630_630113

/-- Given an octagon formed by cutting a smaller rectangle from a larger rectangle,
the minimum number of distinct line segment lengths needed to calculate the perimeter 
of this octagon is 3. --/
theorem min_segments_for_octagon_perimeter (a b c d e f g h : ℝ)
  (cond : a = c ∧ b = d ∧ e = g ∧ f = h) :
  ∃ (u v w : ℝ), u ≠ v ∧ v ≠ w ∧ u ≠ w :=
by
  sorry

end min_segments_for_octagon_perimeter_l630_630113


namespace math_problem_l630_630858

open Real

theorem math_problem (n : ℕ) (θ : ℕ → ℝ) (h : ∀ i, i < n → 0 < θ i ∧ θ i < π / 2) :
  (∑ i in Finset.range n, tan (θ i)) * (∑ i in Finset.range n, cot (θ i)) ≥
  (∑ i in Finset.range n, sin (θ i))^2 + (∑ i in Finset.range n, cos (θ i))^2 :=
sorry

end math_problem_l630_630858


namespace not_monomial_expression_A_monomials_expression_B_C_D_l630_630564

def isMonomial (e : ℕ → ℕ → Prop) : Prop :=
∀ (a : ℕ), ∀ (n : ℕ), ∀ (x : ℕ), (e a n) = a * x ^ n

def expression_A (a : ℕ) :ℕ := 2 / a
def expression_B (a : ℕ) : ℕ:= a / 2
def expression_C : ℕ := 3
def expression_D : ℕ := 0

theorem not_monomial_expression_A (a : ℕ) : ¬ isMonomial expression_A := 
sorry

theorem monomials_expression_B_C_D (a : ℕ) : isMonomial expression_B ∧ isMonomial expression_C ∧ isMonomial expression_D :=
sorry

end not_monomial_expression_A_monomials_expression_B_C_D_l630_630564


namespace medals_distribution_l630_630123

/-- There are 10 sprinters in total and four of them are from Spain. 
    We are interested in determining the number of ways to award 
    gold, silver, and bronze medals to the top three finishers 
    such that at most two Spaniards receive medals. --/

theorem medals_distribution : 
    ∃ ways : ℕ, ways = 696 ∧  
    (∀ sprinter : ℕ, sprinter < 10) ∧ 
    (∀ s : ℕ, s < 4) :=
begin
    sorry
end

end medals_distribution_l630_630123


namespace ratio_of_sums_eq_19_over_17_l630_630258

theorem ratio_of_sums_eq_19_over_17 :
  let a₁ := 5
  let d₁ := 3
  let l₁ := 59
  let a₂ := 4
  let d₂ := 4
  let l₂ := 64
  let n₁ := 19  -- from solving l₁ = a₁ + (n₁ - 1) * d₁
  let n₂ := 16  -- from solving l₂ = a₂ + (n₂ - 1) * d₂
  let S₁ := n₁ * (a₁ + l₁) / 2
  let S₂ := n₂ * (a₂ + l₂) / 2
  S₁ / S₂ = 19 / 17 := by sorry

end ratio_of_sums_eq_19_over_17_l630_630258


namespace solve_inequality_l630_630076

theorem solve_inequality (x : ℝ) :
  |x - 2| + |x + 3| + |2x - 1| < 7 ↔ -1.5 ≤ x ∧ x < 2 :=
by
  sorry

end solve_inequality_l630_630076


namespace coords_of_point_C_l630_630062

noncomputable def point_on_segment (A B C : (ℝ × ℝ)) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

theorem coords_of_point_C (A B : (ℝ × ℝ)) (hA : A = (-2, -1)) (hB : B = (4, 7)) :
  ∃ C : (ℝ × ℝ), point_on_segment A B C ∧ (C.1 = 2) ∧ (C.2 = 13 / 3) :=
by
  use (2, 13 / 3)
  split
  · use 2 / 3
    constructor
    · linarith
    constructor
    · linarith
    · calc
        (2 / 3 * (-2) + (1 - 2 / 3) * 4, 2 / 3 * (-1) + (1 - 2 / 3) * 7)
          = (2, 13 / 3) : by sorry
  · sorry

end coords_of_point_C_l630_630062


namespace base_2_representation_of_123_l630_630541

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630541


namespace problem_statement_l630_630861

variable (a b c : ℝ)

def g (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

theorem problem_statement (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 := by
  have h_even : ∀ x : ℝ, g a b c x = g a b c (-x) := by
    intro x
    simp [g]
  have h_neg10 : g a b c (-10) = g a b c 10 := h_even 10
  rw [h_neg10, h]
  norm_num
  sorry

end problem_statement_l630_630861


namespace find_sum_of_abc_l630_630436

variable (a b c : ℝ)

-- Given conditions
axiom h1 : a^2 + a * b + b^2 = 1
axiom h2 : b^2 + b * c + c^2 = 3
axiom h3 : c^2 + c * a + a^2 = 4

-- Positivity constraints
axiom ha : a > 0
axiom hb : b > 0
axiom hc : c > 0

theorem find_sum_of_abc : a + b + c = Real.sqrt 7 := 
by
  sorry

end find_sum_of_abc_l630_630436


namespace dna_amounts_l630_630358

variable (a : ℝ)

theorem dna_amounts (a : ℝ) : 
  (primary_spermatocyte_DNA a = 4 * a) ∧ (secondary_spermatocyte_DNA a = 2 * a) := 
by
  sorry

end dna_amounts_l630_630358


namespace dot_product_CE_AD_l630_630373

open Classical

variables (A B C E D : Type)
variables [IsRightTriangle A B C] [AE : Midpoint E A B] 
variables [CD : PointsBetweenRatio C D B (2 / 3)]
variables (AB_AC_eq : dist A B = 1) (AC_eq_AB : dist A C = 1)

theorem dot_product_CE_AD :
  let CE := vector_from_to C E
  let AD := vector_from_to A D
  (CE ▸ AD) = (1 / 6) :=
begin
  sorry
end

end dot_product_CE_AD_l630_630373


namespace sum_of_tens_and_units_of_product_is_zero_l630_630475

-- Define the repeating patterns used to create the 999-digit numbers
def pattern1 : ℕ := 400
def pattern2 : ℕ := 606

-- Function to construct a 999-digit number by repeating a 3-digit pattern 333 times
def repeat_pattern (pat : ℕ) (times : ℕ) : ℕ := pat * (10 ^ (3 * times - 3))

-- Define the two 999-digit numbers
def num1 : ℕ := repeat_pattern pattern1 333
def num2 : ℕ := repeat_pattern pattern2 333

-- Function to compute the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Function to compute the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define the product of the two numbers
def product : ℕ := num1 * num2

-- Function to compute the sum of the tens and units digits of a number
def sum_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The statement to be proven
theorem sum_of_tens_and_units_of_product_is_zero :
  sum_digits product = 0 := 
sorry -- Proof steps are omitted

end sum_of_tens_and_units_of_product_is_zero_l630_630475


namespace mark_purchased_cans_l630_630012

theorem mark_purchased_cans : ∀ (J M : ℕ), 
    (J = 40) → 
    (100 - J = 6 * M / 5) → 
    M = 27 := by
  sorry

end mark_purchased_cans_l630_630012


namespace part_I_part_II_l630_630298

-- Definitions:
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def point_on_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop := P ∈ ellipse a b ⟨by linarith, by linarith⟩

def eccentricity (a c : ℝ) (h : a > 0) : ℝ := c / a

-- Given:
constants (a b : ℝ) (h : a > b ∧ b > 0)
constants (P : ℝ × ℝ) (P_cond : P = (1, sqrt 3 / 2)) 
constants (e : ℝ) (e_cond : e = sqrt 3 / 2)
constants (E : ℝ × ℝ) (E_cond : E = (0, -2))

-- Prove part (I)
theorem part_I (h1 : point_on_ellipse a b P) (h2 : eccentricity a (sqrt (a^2 - b^2)) h.1 = e) :
    a = 2 ∧ b = 1 :=
sorry

-- Prove part (II)
theorem part_II (h1 : point_on_ellipse 2 1 P)
                (h2 : line_passing_through E P ∧ meets_ellipse P Q 2)
                (h3 : k_condition):
    area_OPQ O P Q ≤ 1 :=
sorry

end part_I_part_II_l630_630298


namespace pilot_fish_speed_theorem_l630_630849

noncomputable def pilot_fish_speed 
    (keanu_speed : ℕ)
    (shark_factor : ℕ) 
    (pilot_fish_factor : ℕ) 
    : ℕ :=
    let shark_speed_increase := keanu_speed * (shark_factor - 1) in
    let pilot_fish_speed_increase := shark_speed_increase / pilot_fish_factor in
    keanu_speed + pilot_fish_speed_increase

theorem pilot_fish_speed_theorem : 
    pilot_fish_speed 20 2 2 = 30 :=
by 
    simp [pilot_fish_speed]
    sorry  -- proof steps are omitted.

end pilot_fish_speed_theorem_l630_630849


namespace ducks_remaining_after_four_nights_l630_630125

def initial_ducks : ℕ := 500

def first_night_remaining (initial : ℕ) : ℕ :=
  initial - (initial / 5)

def second_night_remaining (after_first : ℕ) : ℕ :=
  after_first - nat.sqrt after_first

def third_night_remaining (after_second : ℕ) : ℕ :=
  let stolen := after_second * 35 / 100 in
  let after_theft := after_second - stolen in
  let returning := after_theft / 5 in
  after_theft + returning

def fourth_night_remaining (after_third : ℕ) : ℕ :=
  let tripled := after_third * 3 in
  let reduced := tripled * 60 / 100 in
  reduced

theorem ducks_remaining_after_four_nights : fourth_night_remaining (third_night_remaining (second_night_remaining (first_night_remaining initial_ducks))) = 533 :=
by
  sorry

end ducks_remaining_after_four_nights_l630_630125


namespace initial_bottles_l630_630676

-- Define the conditions
def drank_bottles : ℕ := 144
def left_bottles : ℕ := 157

-- Define the total_bottles function
def total_bottles : ℕ := drank_bottles + left_bottles

-- State the theorem to be proven
theorem initial_bottles : total_bottles = 301 :=
by
  sorry

end initial_bottles_l630_630676


namespace min_value_f_max_value_f_l630_630290

def f (a b : ℝ) : ℝ := (a^2 + b^2 - 1) / (a * b)

theorem min_value_f (a b : ℝ) (ha : 1 ≤ a) (ha' : a ≤ sqrt 3) (hb : 1 ≤ b) (hb' : b ≤ sqrt 3) :
  1 ≤ f a b :=
by sorry

theorem max_value_f (a b : ℝ) (ha : 1 ≤ a) (ha' : a ≤ sqrt 3) (hb : 1 ≤ b) (hb' : b ≤ sqrt 3) :
  f a b ≤ sqrt 3 :=
by sorry

end min_value_f_max_value_f_l630_630290


namespace decimal_to_binary_equivalent_123_l630_630535

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630535


namespace inverse_of_A_cubed_l630_630801

open Matrix

theorem inverse_of_A_cubed 
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (h : A.inv = ![-5, 3; 1, 3]) :
  (A ^ 3).inv = ![-146, 66; 22, 30] := 
by
  sorry

end inverse_of_A_cubed_l630_630801


namespace non_convergence_uniform_integrable_seq_l630_630438

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [ProbMeasure P]

/-- Definitions from conditions --/
noncomputable def U : Ω → ℝ := sorry -- Uniform[0, 1]
noncomputable def V : Ω → ℝ := sorry -- Independent, Uniform[0, 1]
noncomputable def 𝔾 : MeasurableSpace Ω := sorry -- σ(V)

noncomputable def X_kn (k n : ℕ) : Ω → ℝ :=
  λ ω, n * (if n * U ω ≤ 1 then 1 else 0) * (if k-1 < n * V ω ∧ n * V ω ≤ k then 1 else 0)

noncomputable def xi_n (n : ℕ) : Ω → ℝ :=
  sorry -- List of X_kn like (X_11, X_12, X_22, X_13, X_23, X_33, ...)

-- Statement
theorem non_convergence_uniform_integrable_seq :
  (∀ (n : ℕ), UniformIntegrable (xi_n n) P) ∧ 
  (∃ (ω : Ω), (tsum (λ n, xi_n n ω) = 0 ∧ 
                 tsum (λ n, condexp 𝔾 (xi_n n) ω) ≠ 0)) :=
sorry

end non_convergence_uniform_integrable_seq_l630_630438


namespace intersection_P_Q_l630_630447

def P : Set ℝ := { x | x^2 - 9 < 0 }
def Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem intersection_P_Q : (P ∩ (coe '' Q)) = { x : ℝ | x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 } :=
by sorry

end intersection_P_Q_l630_630447


namespace determine_f1_g1_l630_630245

variables {ℝ : Type} [NonZero ℝ]

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

axiom f_g_conditions :
  ∀ (x : ℝ) (hx : x ≠ 0),
    f x + g (1/x) = x ∧ g x + f (1/x) = 1/x

theorem determine_f1_g1 : f 1 = 1/2 ∧ g 1 = 1/2 :=
by
  sorry

end determine_f1_g1_l630_630245


namespace find_PB_l630_630029

noncomputable def PA : ℝ := 4
def PT (AB : ℝ) : ℝ := 2 * (AB - PA)

theorem find_PB (AB PB x : ℝ) :
  PA = 4 ∧
  PT AB = 2 * (AB - PA) ∧
  PB = PA + AB ∧
  (4 * PB = (PT AB)^2) ∧
  PB = x :=
  x = (17 + real.sqrt 33) / 2
:= sorry

end find_PB_l630_630029


namespace quadratic_inequality_solution_range_l630_630251

theorem quadratic_inequality_solution_range (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0) ↔ (-3 / 2 < k ∧ k < 0) := sorry

end quadratic_inequality_solution_range_l630_630251


namespace overlap_length_l630_630924

noncomputable def total_length_red_segments : ℝ := 98
noncomputable def total_spanning_distance : ℝ := 83
noncomputable def number_of_overlaps : ℕ := 6

theorem overlap_length (x : ℝ) : number_of_overlaps * x = total_length_red_segments - total_spanning_distance → 
  x = (total_length_red_segments - total_spanning_distance) / number_of_overlaps := 
  by
    sorry

end overlap_length_l630_630924


namespace area_trajectory_correct_l630_630008

noncomputable def area_trajectory_of_point_P (AB AC OB : ℝ) (cosA sinA sin2A : ℝ) : ℝ :=
  let BC := real.sqrt (AB^2 + AC^2 - 2 * AB * AC * cosA) in
  if h : BC = 7 then
    let OB_square := OB * OB in
    let area := OB_square * sin2A in
    area
  else
    0

theorem area_trajectory_correct
  (AB : ℝ) (h1 : AB = 5)
  (AC : ℝ) (h2 : AC = 6)
  (cosA : ℝ) (h3 : cosA = 1 / 5)
  (sinA : ℝ) (h4 : sinA = 2 * real.sqrt 6 / 5)
  (sin2A : ℝ) (h5 : sin2A = 4 * real.sqrt 6 / 25)
  (OB : ℝ) (h6 : OB = 35 * real.sqrt 6 / 24) :
  area_trajectory_of_point_P AB AC OB cosA sinA sin2A = 49 * real.sqrt 6 / 24 :=
by {
  have hBC : real.sqrt (5^2 + 6^2 - 2 * 5 * 6 * (1/5)) = 7,
  { sorry },
  rw [h1, h2, h3, h4, h5, h6],
  dsimp [area_trajectory_of_point_P],
  rw if_pos hBC,
  sorry,
}

end area_trajectory_correct_l630_630008


namespace rectangular_solid_edges_sum_l630_630494

theorem rectangular_solid_edges_sum
  (b s : ℝ)
  (h_vol : (b / s) * b * (b * s) = 432)
  (h_sa : 2 * ((b ^ 2 / s) + b ^ 2 * s + b ^ 2) = 432)
  (h_gp : 0 < s ∧ s ≠ 1) :
  4 * (b / s + b + b * s) = 144 := 
by
  sorry

end rectangular_solid_edges_sum_l630_630494


namespace connected_subgraph_exists_l630_630084

open SimpleGraph

theorem connected_subgraph_exists (G : SimpleGraph V) [Fintype V] [∀ v, Fintype (G.neighborFinset v)]
  (n : ℕ) (h_n : ∀ v : V, G.degree v ≥ 3 * n / 4) (coloring : E → Prop) :
  ∃ (H : SimpleGraph V) (S : Finset V), (H = G.subgraphInduced S) ∧ (S.card ≥ 3 * n / 4 + 1) ∧ (∀ e ∈ H.edgeSet, coloring e) :=
sorry

end connected_subgraph_exists_l630_630084


namespace find_regression_line_equation_l630_630313

noncomputable def regression_line_equation : ℝ → ℝ → ℝ × ℝ → (ℝ → ℝ) :=
  sorry

-- Given conditions
def slope := 2.1
def center := (3.0, 4.0)

-- Required equation of regression line
def regression_line := λ x: ℝ, 2.1 * x - 2.3

theorem find_regression_line_equation :
  regression_line_equation slope center = regression_line :=
by
  sorry

end find_regression_line_equation_l630_630313


namespace values_of_N_l630_630727

theorem values_of_N (N : ℕ) : (∃ k, k ∈ ({4, 6, 8, 12, 16, 24, 48} : set ℕ) ∧ k = N + 3) ↔ (N ∈ {1, 3, 5, 9, 13, 21, 45} : set ℕ) :=
by 
  sorry

#eval values_of_N 4 -- Example usage: should give true if N = 1

end values_of_N_l630_630727


namespace number_of_valid_mappings_l630_630304

def A : Set := {a, b, c}
def B : Set := {0, 1}

def f (x : A) : B

axiom f_cond (f : A → B) : f a * f b = f c

theorem number_of_valid_mappings : ∃! (f : A → B), f a * f b = f c ∧ (set.count {f | f_cond f} = 4) :=
sorry

end number_of_valid_mappings_l630_630304


namespace club_additional_members_l630_630600

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l630_630600


namespace quadrilateral_area_lemma_l630_630624

-- Define the coordinates of the vertices
structure Point where
  x : ℤ
  y : ℤ

def A : Point := ⟨1, 3⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨2, 1⟩
def D : Point := ⟨2006, 2007⟩

-- Function to calculate the area of a quadrilateral given its vertices
def quadrilateral_area (A B C D : Point) : ℤ := 
  let triangle_area (P Q R : Point) : ℤ :=
    (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x) / 2
  triangle_area A B C + triangle_area A C D

-- The statement to be proved
theorem quadrilateral_area_lemma : quadrilateral_area A B C D = 3008 := 
  sorry

end quadrilateral_area_lemma_l630_630624


namespace jerry_daughters_games_l630_630386

theorem jerry_daughters_games (x y : ℕ) (h : 4 * x + 2 * x + 4 * y + 2 * y = 96) (hx : x = y) :
  x = 8 ∧ y = 8 :=
by
  have h1 : 6 * x + 6 * y = 96 := by linarith
  have h2 : x = y := hx
  sorry

end jerry_daughters_games_l630_630386


namespace f1_has_property_p_f2_does_not_have_property_p_f3_has_property_p_l630_630399

namespace Properties

def property_p (f : ℝ × ℝ → ℝ) : Prop :=
∀ (a b : ℝ × ℝ) (λ : ℝ), f (λ • a + (1 - λ) • b) = λ * f a + (1 - λ) * f b

def f_1 (m : ℝ × ℝ) : ℝ := m.1 - m.2
def f_2 (m : ℝ × ℝ) : ℝ := m.1 ^ 2 + m.2
def f_3 (m : ℝ × ℝ) : ℝ := m.1 + m.2 + 1

theorem f1_has_property_p : property_p f_1 := sorry
theorem f2_does_not_have_property_p : ¬ property_p f_2 := sorry
theorem f3_has_property_p : property_p f_3 := sorry

end Properties

end f1_has_property_p_f2_does_not_have_property_p_f3_has_property_p_l630_630399


namespace fraction_of_total_amount_l630_630569

-- Conditions
variable (p q r : ℕ)
variable (total_amount amount_r : ℕ)
variable (total_amount_eq : total_amount = 6000)
variable (amount_r_eq : amount_r = 2400)

-- Mathematical statement
theorem fraction_of_total_amount :
  amount_r / total_amount = 2 / 5 :=
by
  -- Sorry to skip the proof, as instructed
  sorry

end fraction_of_total_amount_l630_630569


namespace sum_of_sequence_l630_630235

theorem sum_of_sequence (n : ℕ) : 
  (finset.sum (finset.range (n + 1)) (λ k, 2 * k - 1)) = (n + 1) ^ 2 := 
sorry

end sum_of_sequence_l630_630235


namespace neznaika_made_mistake_l630_630055

theorem neznaika_made_mistake 
(a : Fin 11 → ℕ)
(d : Fin 11 → ℤ)
(h1 : ∑ i, d i = 0)
(h2 : (Finset.card ((Finset.univ.filter (λ i, d i = 1)) ∪ (Finset.univ.filter (λ i, d i = -1))) = 4))
(h3 : (Finset.card ((Finset.univ.filter (λ i, d i = 2)) ∪ (Finset.univ.filter (λ i, d i = -2))) = 4))
(h4 : (Finset.card ((Finset.univ.filter (λ i, d i = 3)) ∪ (Finset.univ.filter (λ i, d i = -3))) = 3))
: false :=
by
  sorry

end neznaika_made_mistake_l630_630055


namespace magnitude_equation_l630_630038

variables (z w : ℂ)

-- Defining the conditions
def condition1 : Prop := complex.abs z = 2
def condition2 : Prop := complex.abs w = 4
def condition3 : Prop := complex.abs (z + w) = 5

-- The proof statement
theorem magnitude_equation (h₁ : condition1 z) (h₂ : condition2 w) (h₃ : condition3 z w) : 
  complex.abs (1/z + 1/w) = 5/8 :=
by
  sorry

end magnitude_equation_l630_630038


namespace remainder_3_pow_2023_mod_5_l630_630154

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l630_630154


namespace value_of_expression_at_three_l630_630951

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end value_of_expression_at_three_l630_630951


namespace acceleration_constant_l630_630622

noncomputable def displacement (t : ℝ) : ℝ := t^2 - t + 6

theorem acceleration_constant (t : ℝ) (h : 1 ≤ t ∧ t ≤ 4) : 
  (derivative (derivative displacement t)) = 2 :=
by
  sorry

end acceleration_constant_l630_630622


namespace program_output_l630_630066

def iterate (S i : ℕ) : ℕ × ℕ :=
  if i ≤ 10 then (S + i, i^2 + 1) else (S, i)

noncomputable def final_S : ℕ :=
  let ini_S := 0
  let ini_i := 0
  Nat.iterate (λ ⟨S, i⟩, iterate S i) 4 (ini_S, ini_i) |>.1

theorem program_output :
  final_S = 8 :=
by
  sorry

end program_output_l630_630066


namespace average_rate_of_change_correct_l630_630779

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Define the start and end points within a range
def x_start : ℝ := 2
def x_end (Δx : ℝ) : ℝ := 2 + Δx

-- Define the average rate of change
def average_rate_of_change (Δx : ℝ) : ℝ :=
  (f (x_end Δx) - f x_start) / Δx

-- The statement to prove: the average rate of change equals Δx + 3
theorem average_rate_of_change_correct (Δx : ℝ) : average_rate_of_change Δx = Δx + 3 := by
  sorry

end average_rate_of_change_correct_l630_630779


namespace valid_paintings_of_faces_l630_630646

-- Define the sets of faces and pairs that add up to 9.
def faces : Finset ℕ := Finset.range 8

def pairs_summing_to_nine : Finset (ℕ × ℕ) :=
  {(1, 8), (8, 1), (2, 7), (7, 2), (3, 6), (6, 3), (4, 5), (5, 4)}

-- Define the main theorem to prove the valid combinations.
theorem valid_paintings_of_faces : ∃(s : Finset (Finset ℕ)), ∀ (t ∈ s), 
  t ⊆ faces ∧ t.card = 3 ∧ (∀ (a b ∈ t), (a, b) ∉ pairs_summing_to_nine) ∧
  s.card = 40 := 
sorry

end valid_paintings_of_faces_l630_630646


namespace sum_b_n_as_T_n_l630_630119

-- Considering the sequence \{a_n\} with the sum of first n terms Sn = n(2n - 1)
def Sn (n : ℕ) : ℕ := n * (2 * n - 1)

-- Define a_n using the conditions provided
def a_n : ℕ → ℕ
| 1 := Sn 1
| n := Sn n - Sn (n - 1)

-- Define the sequence \{b_n\} given by b_n = 2^n a_n
def b_n (n : ℕ) : ℕ := 2^n * a_n n

-- Define the sum of the first n terms of the sequence \{b_n\} denoted by T_n
def T_n (n : ℕ) : ℕ := (4 * n - 1) * 2^(n + 1) - 14

-- Proof statement: Proving the sum of the first n terms of b_n is T_n 
theorem sum_b_n_as_T_n (n : ℕ) : 
  (∑ k in Finset.range n, b_n k) = T_n n := sorry

end sum_b_n_as_T_n_l630_630119


namespace lego_set_cost_l630_630737

-- Definitions and conditions
def price_per_car := 5
def cars_sold := 3
def action_figures_sold := 2
def total_earnings := 120

-- Derived prices
def price_per_action_figure := 2 * price_per_car
def price_per_board_game := price_per_action_figure + price_per_car

-- Total cost of sold items (cars, action figures, and board game)
def total_cost_of_sold_items := 
  (cars_sold * price_per_car) + 
  (action_figures_sold * price_per_action_figure) + 
  price_per_board_game

-- Cost of Lego set
theorem lego_set_cost : 
  total_earnings - total_cost_of_sold_items = 70 :=
by
  -- Proof omitted
  sorry

end lego_set_cost_l630_630737


namespace tangent_line_tangent_point_unique_l630_630361

theorem tangent_line_tangent_point_unique (a b : ℝ) :
  (∃ x₁ : ℝ, 1 = exp x₁ ∧ x₁ + b = exp x₁) ∧
  (∃ x₂ : ℝ, 1 = (1 / (x₂ + a)) ∧ x₂ + b = log (x₂ + a)) →
  a = 2 ∧ b = 1 :=
by
  sorry

end tangent_line_tangent_point_unique_l630_630361


namespace cosine_of_angle_between_lines_l630_630993

noncomputable def cos_theta := (33 : ℝ) / Real.sqrt 1189

theorem cosine_of_angle_between_lines :
  let v1 := (4, 5)
  let v2 := (2, 5)
  Real.cos_angle (v1.1, v1.2) (v2.1, v2.2) = cos_theta :=
sorry

end cosine_of_angle_between_lines_l630_630993


namespace measure_angle_MBC_l630_630381

-- Given conditions
variable {α : Type}
variable [LinearOrder α] [LinearOrder α] [AddCommGroup α] 
variable (A B C M : euclidean_space α)
def ∠ (P Q R : euclidean_space α) : α := sorry  -- This would be the definition of the angle
def triangle (A B C : euclidean_space α) : Prop := sorry  -- This asserts A, B, and C form a triangle

-- The conditions (angles in degrees for simplicity)
axiom angle_ABC_50 : ∠ A B C = 50
axiom angle_ACB_30 : ∠ A C B = 30
axiom angle_MCB_20 : ∠ M C B = 20
axiom angle_MAC_40 : ∠ M A C = 40

-- We need to prove that the measure of ∠ MBC is 30 degrees
theorem measure_angle_MBC : ∠ M B C = 30 := sorry

end measure_angle_MBC_l630_630381


namespace value_of_expression_l630_630177

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x - 4)^2 = 100 :=
by
  -- Given the hypothesis h: x = -2
  -- Need to show: (3 * x - 4)^2 = 100
  sorry

end value_of_expression_l630_630177


namespace number_of_primes_in_factorization_l630_630666

theorem number_of_primes_in_factorization : 
  let n := 101 * 103 * 105 * 107 in
  (prime 101) ∧ (prime 103) ∧ (¬prime 105) ∧ (105 = 3 * 5 * 7) ∧ (prime 107) ∧
  (∀ p : ℕ, (p > 1) → (p ∣ n → p = 101 ∨ p = 103 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 107)) ∧
  (∃ primes : finset ℕ, primes = {101, 103, 3, 5, 7, 107} ∧ primes.card = 6) := 
sorry

end number_of_primes_in_factorization_l630_630666


namespace inf_solutions_integers_l630_630796

theorem inf_solutions_integers (x y z : ℕ) : ∃ (n : ℕ), ∀ n > 0, (x = 2^(32 + 72 * n)) ∧ (y = 2^(28 + 63 * n)) ∧ (z = 2^(25 + 56 * n)) → x^7 + y^8 = z^9 :=
by {
  sorry
}

end inf_solutions_integers_l630_630796


namespace total_music_school_cost_l630_630938

-- Define the tuition, discounts, and the children involved
def tuition : ℕ := 45
def first_sibling_discount : ℕ := 15
def additional_sibling_discount : ℕ := 10

-- Define the costs for each child
def ali_cost := tuition
def matt_cost := tuition - first_sibling_discount
def jane_cost := tuition - additional_sibling_discount
def sarah_cost := tuition - additional_sibling_discount

-- Define the total cost
def total_cost := ali_cost + matt_cost + jane_cost + sarah_cost

-- The theorem stating the total cost
theorem total_music_school_cost : total_cost = 145 :=
by
  -- Costs are calculated as follows
  have h_ali : ali_cost = 45 := rfl
  have h_matt : matt_cost = 45 - 15 := rfl
  have h_matt_cost : matt_cost = 30 := by rw [h_matt]; norm_num
  have h_jane : jane_cost = 45 - 10 := rfl
  have h_jane_cost : jane_cost = 35 := by rw [h_jane]; norm_num
  have h_sarah : sarah_cost = 45 - 10 := rfl
  have h_sarah_cost : sarah_cost = 35 := by rw [h_sarah]; norm_num

  -- Sum up all costs
  rw [h_ali, h_matt_cost, h_jane_cost, h_sarah_cost]
  norm_num


end total_music_school_cost_l630_630938


namespace make_fraction_meaningful_l630_630937

theorem make_fraction_meaningful (x : ℝ) : (x - 1) ≠ 0 ↔ x ≠ 1 :=
by
  sorry

end make_fraction_meaningful_l630_630937


namespace min_n_minus_m_l630_630780

noncomputable def f (x : ℝ) : ℝ := Real.log x - Real.log 2 + 1 / 2
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

theorem min_n_minus_m :
  ∃ n m : ℝ, g m = f n ∧ ∀ x y, g y = f x → x - y ≥ n - m := 
begin
  use [2 * Real.exp (1 / 2 - 1 / 2), 2 + Real.log (Real.exp 1)],
  split,
  { sorry },
  { intros x y hxy,
    sorry }
end

end min_n_minus_m_l630_630780


namespace probability_multiple_of_45_l630_630187

def single_digit_multiples_of_3 : List ℕ := [3, 6, 9]
def prime_numbers_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem probability_multiple_of_45 :
  let total_outcomes := (single_digit_multiples_of_3.length * prime_numbers_less_than_20.length : ℕ)
  let favorable_outcomes := (1 : ℕ)
  let probability := (favorable_outcomes.toReal / total_outcomes.toReal)
  in 
    probability = 1 / 24 := by
  sorry

end probability_multiple_of_45_l630_630187


namespace smallest_y_81_pow_y_gt_7_pow_42_l630_630175

theorem smallest_y_81_pow_y_gt_7_pow_42 : ∃ y : ℤ, 81^y > 7^42 ∧ ∀ z : ℤ, z < y → 81^z ≤ 7^42 :=
by
  sorry

end smallest_y_81_pow_y_gt_7_pow_42_l630_630175


namespace father_age_l630_630692

variable (F C1 C2 : ℕ)

theorem father_age (h1 : F = 3 * (C1 + C2))
  (h2 : F + 5 = 2 * (C1 + 5 + C2 + 5)) :
  F = 45 := by
  sorry

end father_age_l630_630692


namespace six_digit_number_l630_630693

/-- 
Find a six-digit number that starts with the digit 1 and such that if this digit is moved to the end, the resulting number is three times the original number.
-/
theorem six_digit_number (N : ℕ) (h₁ : 100000 ≤ N ∧ N < 1000000) (h₂ : ∃ x : ℕ, N = 1 * 10^5 + x ∧ 10 * x + 1 = 3 * N) : N = 142857 :=
by sorry

end six_digit_number_l630_630693


namespace sin_double_angle_l630_630758

theorem sin_double_angle (θ : ℝ) (h : Real.sin (π / 4 + θ) = 1 / 3) : Real.sin (2 * θ) = -7 / 9 :=
by
  sorry

end sin_double_angle_l630_630758


namespace base_2_representation_of_123_l630_630539

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l630_630539


namespace quadratic_equation_in_one_variable_l630_630179

def is_quadratic_in_one_variable (eq : String) : Prop :=
  match eq with
  | "2x^2 + 5y + 1 = 0" => False
  | "ax^2 + bx - c = 0" => ∃ (a b c : ℝ), a ≠ 0
  | "1/x^2 + x = 2" => False
  | "x^2 = 0" => True
  | _ => False

theorem quadratic_equation_in_one_variable :
  is_quadratic_in_one_variable "x^2 = 0" := by
  sorry

end quadratic_equation_in_one_variable_l630_630179


namespace sum_of_coeffs_l630_630919

-- Define the polynomial with real coefficients
def poly (p q r s : ℝ) : Polynomial ℝ := Polynomial.C 1 + Polynomial.C p * Polynomial.X + Polynomial.C q * Polynomial.X^2 + Polynomial.C r * Polynomial.X^3 + Polynomial.C s * Polynomial.X^4

-- Given conditions
def g (x : ℂ) : Polynomial ℂ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem sum_of_coeffs (p q r s : ℝ)
  (h1 : g (Complex.I * 3) = 0)
  (h2 : g (1 + 2 * Complex.I) = 0) :
  p + q + r + s = -41 :=
sorry

end sum_of_coeffs_l630_630919


namespace base_2_representation_of_123_is_1111011_l630_630529

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630529


namespace sum_of_extreme_terms_in_sequence_l630_630902

theorem sum_of_extreme_terms_in_sequence : 
  ∃ (a_min a_max : ℕ), 
  (∀ (n : ℕ), n ∈ set.range (λ k, 3 * k + 2) → n ∈ set.range (λ k, 5 * k + 3) → 1 ≤ n ∧ n ≤ 200 → 
  a_min ≤ n ∧ n ≤ a_max) ∧ 
  a_min == 8 ∧ 
  a_max == 188 ∧ 
  (a_min + a_max) = 196 := 
sorry

end sum_of_extreme_terms_in_sequence_l630_630902


namespace commission_percentage_l630_630279

theorem commission_percentage 
  (total_amount : ℝ) 
  (h1 : total_amount = 800) 
  (commission_first_500 : ℝ) 
  (h2 : commission_first_500 = 0.20 * 500) 
  (excess_amount : ℝ) 
  (h3 : excess_amount = (total_amount - 500)) 
  (commission_excess : ℝ) 
  (h4 : commission_excess = 0.25 * excess_amount) 
  (total_commission : ℝ) 
  (h5 : total_commission = commission_first_500 + commission_excess) 
  : (total_commission / total_amount) * 100 = 21.875 := 
by
  sorry

end commission_percentage_l630_630279


namespace max_carpets_in_room_l630_630130

theorem max_carpets_in_room : 
  let room_size := 13 * 13
  ∃ n, n = room_size ∧ (∀ (carpet : ℕ → set (ℕ × ℕ)), (∀ i, i < n → 
    carpet i ≠ ∅ ∧ 
    (∀ j, j < n → carpet i ≠ carpet j → carpet i ∩ carpet j = ∅ ∧ 
    ¬(carpet i ⊆ carpet j) ∧ ¬(carpet j ⊆ carpet i)))) :=
by sorry

end max_carpets_in_room_l630_630130


namespace constant_sum_of_squares_l630_630394

theorem constant_sum_of_squares {A B C H Q : Point} {a b c R : ℝ} 
  (h_orthocenter : is_orthocenter A B C H) 
  (h_euler_circle : on_euler_circle Q A B C) 
  (h_side_lengths : side_lengths A B C a b c) 
  (h_circumradius : circumradius A B C R) :
  (QA ^ 2 + QB ^ 2 + QC ^ 2 - QH ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 - 2 * R ^ 2) :=
sorry

end constant_sum_of_squares_l630_630394


namespace max_pairs_l630_630285

theorem max_pairs (n : ℕ) (S_max : ℕ) (k_max : ℕ) : 
  ∀ (k : ℕ), n = 2009 → 
            S_max = (4019 - k) * k / 2 → 
            k_max = 803 → 
            (∀ (a_i b_i : ℕ), 
              {i // i ∈ {1, 2, ..., 2009}} → 
              a_i < b_i → 
              a_i + b_i ≤ 2009 → 
              ∃! r, r = k_max) :=
sorry

end max_pairs_l630_630285


namespace player_A_prevents_B_winning_l630_630137

-- Define the conditions and elements of the problem
def game : Type := ℕ → Prop
def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def can_be_prefixed (d : ℕ) (seq : ℕ) : ℕ := d * 10 + seq
noncomputable def game_condition (f : game) (d : ℕ) : Prop := 
  ∀ (seq : ℕ), 
  (f (can_be_prefixed 0 seq) → f (can_be_prefixed d seq)) ∧
  (f (can_be_prefixed d seq) → perfect_square seq)

-- Stating the proof problem
theorem player_A_prevents_B_winning : ∀ (f : game), (∀ seq, f seq → ¬perfect_square seq) →
  ∃! (dA : ℕ) (seqA : ℕ), ∀ (dB : ℕ), ¬ game_condition f seqA :=
begin
  sorry
end

end player_A_prevents_B_winning_l630_630137


namespace probability_r_div_k_is_square_l630_630105

open Set

def r_set : Set ℤ := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def k_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem probability_r_div_k_is_square :
  let total_pairs := (r_set.card : ℚ) * (k_set.card : ℚ)
  let square_r_set := {r ∈ r_set | ∃ k ∈ k_set, (r : ℚ) = k^2}
in (square_r_set.card : ℚ) / total_pairs = 8/63 :=
by
  sorry

end probability_r_div_k_is_square_l630_630105


namespace factorize_expression_l630_630691

theorem factorize_expression (a b : ℝ) :
  4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) :=
by
  sorry

end factorize_expression_l630_630691


namespace weight_of_mixture_correct_l630_630983

-- Defining the fractions of each component in the mixture
def sand_fraction : ℚ := 2 / 9
def water_fraction : ℚ := 5 / 18
def gravel_fraction : ℚ := 1 / 6
def cement_fraction : ℚ := 7 / 36
def limestone_fraction : ℚ := 1 - sand_fraction - water_fraction - gravel_fraction - cement_fraction

-- Given weight of limestone
def limestone_weight : ℚ := 12

-- Total weight of the mixture that we need to prove
def total_mixture_weight : ℚ := 86.4

-- Proof problem statement
theorem weight_of_mixture_correct : (limestone_fraction * total_mixture_weight = limestone_weight) :=
by
  have h_sand := sand_fraction
  have h_water := water_fraction
  have h_gravel := gravel_fraction
  have h_cement := cement_fraction
  have h_limestone := limestone_fraction
  have h_limestone_weight := limestone_weight
  have h_total_weight := total_mixture_weight
  sorry

end weight_of_mixture_correct_l630_630983


namespace alex_charge_per_trip_l630_630222

theorem alex_charge_per_trip (x : ℝ)
  (savings_needed : ℝ) (n_trips : ℝ) (worth_groceries : ℝ) (charge_per_grocery_percent : ℝ) :
  savings_needed = 100 → 
  n_trips = 40 →
  worth_groceries = 800 →
  charge_per_grocery_percent = 0.05 →
  n_trips * x + charge_per_grocery_percent * worth_groceries = savings_needed →
  x = 1.5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end alex_charge_per_trip_l630_630222


namespace value_of_g_800_l630_630034

noncomputable def g : ℝ → ℝ :=
sorry

theorem value_of_g_800 (g_eq : ∀ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), g (x * y) = g x / (y^2))
  (g_at_1000 : g 1000 = 4) : g 800 = 625 / 2 :=
sorry

end value_of_g_800_l630_630034


namespace sum_evaluation_l630_630259

def b : ℕ := 5

theorem sum_evaluation : (∑ k in {1, 2, 3}, (b - k) * (b - (k + 1)) * (b - (k + 2))) = 30 := by
  sorry

end sum_evaluation_l630_630259


namespace base_2_representation_of_123_l630_630544

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l630_630544


namespace prime_consecutive_fraction_equivalence_l630_630356

theorem prime_consecutive_fraction_equivalence (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hq_p_consec : p + 1 ≤ q ∧ Nat.Prime (p + 1) -> p + 1 = q) (hpq : p < q) (frac_eq : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := sorry

end prime_consecutive_fraction_equivalence_l630_630356


namespace ratio_men_to_women_l630_630188

theorem ratio_men_to_women
  (W M : ℕ)      -- W is the number of women, M is the number of men
  (avg_height_all : ℕ) (avg_height_female : ℕ) (avg_height_male : ℕ)
  (h1 : avg_height_all = 180)
  (h2 : avg_height_female = 170)
  (h3 : avg_height_male = 182) 
  (h_avg : (170 * W + 182 * M) / (W + M) = 180) :
  M = 5 * W :=
by
  sorry

end ratio_men_to_women_l630_630188


namespace train_passes_jogger_in_time_l630_630990

def jogger_speed_kmph := 8
def train_speed_kmph := 50
def jogger_lead_m := 360
def train_length_m := 180

def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
def relative_speed_mps := train_speed_mps - jogger_speed_mps
def total_distance_m := jogger_lead_m + train_length_m

-- We state the theorem here:
theorem train_passes_jogger_in_time :
  (total_distance_m / relative_speed_mps) = 46.25 :=
by 
  sorry

end train_passes_jogger_in_time_l630_630990


namespace matrix_B_cannot_be_obtained_from_matrix_A_l630_630972

def A : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, -1, -1, 1],
  ![1, 1, -1, 1, -1]
]

def B : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, 1, -1, 1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, 1, -1, 1]
]

theorem matrix_B_cannot_be_obtained_from_matrix_A :
  A.det ≠ B.det := by
  sorry

end matrix_B_cannot_be_obtained_from_matrix_A_l630_630972


namespace problem_statement_l630_630042

noncomputable def a : ℝ := 6 * Real.sqrt 2
noncomputable def b : ℝ := 18 * Real.sqrt 2
noncomputable def c : ℝ := 6 * Real.sqrt 21
noncomputable def d : ℝ := 24 * Real.sqrt 2
noncomputable def e : ℝ := 48 * Real.sqrt 2
noncomputable def N : ℝ := 756 * Real.sqrt 10

axiom condition_a : a^2 + b^2 + c^2 + d^2 + e^2 = 504
axiom positive_numbers : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

theorem problem_statement : N + a + b + c + d + e = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by
  -- We'll insert the proof here later
  sorry

end problem_statement_l630_630042


namespace jeans_discount_rates_l630_630051

theorem jeans_discount_rates
    (M F P : ℝ) 
    (regular_price_moose jeans_regular_price_fox jeans_regular_price_pony : ℝ)
    (moose_count fox_count pony_count : ℕ)
    (total_discount : ℝ) :
    regular_price_moose = 20 →
    regular_price_fox = 15 →
    regular_price_pony = 18 →
    moose_count = 2 →
    fox_count = 3 →
    pony_count = 2 →
    total_discount = 12.48 →
    (M + F + P = 0.32) →
    (F + P = 0.20) →
    (moose_count * M * regular_price_moose + fox_count * F * regular_price_fox + pony_count * P * regular_price_pony = total_discount) →
    M = 0.12 ∧ F = 0.0533 ∧ P = 0.1467 :=
by
  intros
  sorry -- The proof is not required

end jeans_discount_rates_l630_630051


namespace hotel_towels_l630_630613

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l630_630613


namespace maximize_fraction_l630_630419

theorem maximize_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
sorry

end maximize_fraction_l630_630419


namespace min_q_ge_half_l630_630086

-- Define the parameters and conditions
def binom (n k : ℕ) : ℕ := n.choose k

def q (a : ℕ) : ℚ := (binom (48 - a) 2 + binom (a - 1) 2) / 1653

-- The main theorem statement
theorem min_q_ge_half (p q : ℕ) (hpq : p + q = 915) : 
  ∃ a, a ∈ {10, 38} ∧ q(a) ≥ 1/2 ∧ (∃ p q, p + q = 915 ∧ q(a) = (p / q)) :=
by
  sorry  -- Proof is not required for statement generation

end min_q_ge_half_l630_630086


namespace ratio_of_areas_l630_630852

def S3 (x y : ℝ) : Prop := log 10 (3 + 2 * x ^ 2 + 2 * y ^ 2) ≤ 2 + log 10 (x + 2 * y)
def S4 (x y : ℝ) : Prop := log 10 (5 + 3 * x ^ 2 + 3 * y ^ 2) ≤ 3 + log 10 (x + 3 * y)

noncomputable def area (S : ℝ × ℝ → Prop) : ℝ := sorry

theorem ratio_of_areas :
  area S4 / area S3 = 108 :=
sorry

end ratio_of_areas_l630_630852


namespace number_of_points_on_circle_at_distance_2_l630_630205

noncomputable def circle : set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 = 16 }
noncomputable def line : ℝ → ℝ → Prop := λ x y, 2 * x - y = 6
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

theorem number_of_points_on_circle_at_distance_2 :
  (|{ p : ℝ × ℝ | p ∈ circle ∧ distance p (0, 0) = 2 }| = 2) :=
by
  sorry

end number_of_points_on_circle_at_distance_2_l630_630205


namespace sum_of_extreme_terms_in_sequence_l630_630901

theorem sum_of_extreme_terms_in_sequence : 
  ∃ (a_min a_max : ℕ), 
  (∀ (n : ℕ), n ∈ set.range (λ k, 3 * k + 2) → n ∈ set.range (λ k, 5 * k + 3) → 1 ≤ n ∧ n ≤ 200 → 
  a_min ≤ n ∧ n ≤ a_max) ∧ 
  a_min == 8 ∧ 
  a_max == 188 ∧ 
  (a_min + a_max) = 196 := 
sorry

end sum_of_extreme_terms_in_sequence_l630_630901


namespace similar_inscribed_triangle_exists_l630_630437

variable {α : Type*} [LinearOrderedField α]

-- Representing points and triangles
structure Point (α : Type*) := (x : α) (y : α)
structure Triangle (α : Type*) := (A B C : Point α)

-- Definitions for inscribed triangles and similarity conditions
def isInscribed (inner outer : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

def areSimilar (Δ1 Δ2 : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

-- Main theorem
theorem similar_inscribed_triangle_exists (Δ₁ Δ₂ : Triangle α) (h_ins : isInscribed Δ₂ Δ₁) :
  ∃ Δ₃ : Triangle α, isInscribed Δ₃ Δ₂ ∧ areSimilar Δ₁ Δ₃ :=
sorry

end similar_inscribed_triangle_exists_l630_630437


namespace part1_monotonic_increasing_part2_BC_value_l630_630033

-- Definitions for part (1)
def f (x : ℝ) : ℝ := sin (2 * x + π / 6) - 2 * cos x ^ 2
def interval (k : ℤ) : Set ℝ := Set.Icc (-π / 6 + k * π) (π / 3 + k * π)

theorem part1_monotonic_increasing (k : ℤ) : Monotone (f ∘ interval k) :=
sorry

-- Definitions and conditions for part (2)
noncomputable def A := (2 * arcsin ((-5/4 + 1) / sqrt 3 + 1) + π/3) 
def CD : ℝ := 4
def DA : ℝ := 2
def BD : ℝ := sqrt 10
def cos_ABD : ℝ := sqrt 10 / 4

-- This represents a triangle with the vector relationship CD = 2 * DA
noncomputable def BC : triangle := {
  C := 4,
  D := 2,
  B := sqrt 10,
} 

theorem part2_BC_value : BC.dist 2 2 = 6 :=
sorry

end part1_monotonic_increasing_part2_BC_value_l630_630033


namespace number_of_subsets_of_A_l630_630786

open Set

theorem number_of_subsets_of_A :
  let A := {x : ℝ | x^2 - 3 = 0}
  ∃ (n : ℕ), n = 4 ∧ (finite A ∧ card (powerset A) = n) :=
by
  let A := {x : ℝ | x^2 - 3 = 0}
  have hA : A = {sqrt 3, -sqrt 3} := sorry
  have finite_A : finite A := sorry
  have card_powerset_A : card (powerset A) = 4 := sorry
  exact ⟨4, (finite_A, card_powerset_A)⟩

end number_of_subsets_of_A_l630_630786


namespace total_material_ordered_l630_630204

theorem total_material_ordered (c b s : ℝ) (hc : c = 0.17) (hb : b = 0.17) (hs : s = 0.5) :
  c + b + s = 0.84 :=
by sorry

end total_material_ordered_l630_630204


namespace club_additional_members_l630_630597

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l630_630597


namespace distinct_diff_positive_integers_count_l630_630793

set_option pp.all true  -- To ensure fully detailed output when necessary

def distinct_diff_integers (s : Set ℕ) : Set ℕ :=
  {d | ∃ x y ∈ s, x ≠ y ∧ d = abs (x - y)}

-- The set {2, 3, 4, ..., 18, 19}
def my_set : Set ℕ := {n | 2 ≤ n ∧ n ≤ 19}

theorem distinct_diff_positive_integers_count :
  (distinct_diff_integers my_set).card = 17 :=
by
  sorry

end distinct_diff_positive_integers_count_l630_630793


namespace checkerboard_swap_possible_iff_divisible_by_3_l630_630572

theorem checkerboard_swap_possible_iff_divisible_by_3 (n : ℕ) :
  (∃ grid : Array (Array char), 
    is_checkerboard_pattern grid n ∧ 
    corner_is_black grid ∧ 
    (∀ i j, (i < n - 1) → (j < n - 1) → 
    recolor_subgrid grid i j) ∧
    can_transform_to_swapped_checkerboard grid) ↔ 
  n % 3 = 0 := 
sorry

-- Definitions of the conditions
def is_checkerboard_pattern (grid : Array (Array char)) (n : ℕ) : Prop := sorry
def corner_is_black (grid : Array (Array char)) : Prop := sorry
def recolor_subgrid (grid : Array (Array char)) (i j : ℕ) : Prop := sorry
def can_transform_to_swapped_checkerboard (grid : Array (Array char)) : Prop := sorry

end checkerboard_swap_possible_iff_divisible_by_3_l630_630572


namespace sum_of_coordinates_of_other_endpoint_l630_630473

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (midpoint_cond : (x + 1) / 2 = 3)
  (midpoint_cond2 : (y - 3) / 2 = 5) :
  x + y = 18 :=
sorry

end sum_of_coordinates_of_other_endpoint_l630_630473


namespace smaller_angle_at_9_am_l630_630661

-- Define the angular positions of the minute and hour hands
def minute_hand_angle (minute : Nat) : ℕ := 0  -- At the 12 position
def hour_hand_angle (hour : Nat) : ℕ := hour * 30  -- 30 degrees per hour

-- Define the function to get the smaller angle between two angles on the clock from 0 to 360 degrees
def smaller_angle (angle1 angle2 : ℕ) : ℕ :=
  let angle_diff := Int.natAbs (angle1 - angle2)
  min angle_diff (360 - angle_diff)

-- The theorem to prove
theorem smaller_angle_at_9_am : smaller_angle (minute_hand_angle 0) (hour_hand_angle 9) = 90 := sorry

end smaller_angle_at_9_am_l630_630661


namespace similar_triangle_shortest_side_l630_630489

variable (x : ℝ) (a b c : ℝ)

theorem similar_triangle_shortest_side (h1 : a = 8) (h2 : b = 10) (h3 : c = 12) 
  (h4 : 8 * x + 10 * x + 12 * x = 150) : 8 * x = 40 :=
by
  rw [h1, h2, h3] at h4
  have hx : x = 150 / 30 := by linarith
  rw [hx]
  norm_num

end similar_triangle_shortest_side_l630_630489


namespace club_additional_members_l630_630596

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l630_630596


namespace remainder_by_19_l630_630996

theorem remainder_by_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
by sorry

end remainder_by_19_l630_630996


namespace rectangle_diagonal_length_l630_630317

-- Define the conditions
def is_rectangle (l w : ℝ) :=
  2 * (l + w) = 10 ∧ l * w = 6

-- Define the problem to be proved
theorem rectangle_diagonal_length :
  ∀ l w : ℝ, is_rectangle l w → (l^2 + w^2) = 13 :=
by
  intro l w h
  cases h with h1 h2
  sorry

end rectangle_diagonal_length_l630_630317


namespace sin_double_angle_l630_630345

-- Define the conditions and the goal
theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
sorry

end sin_double_angle_l630_630345


namespace g_10_plus_g_neg10_eq_6_l630_630863

variable (a b c : ℝ)
noncomputable def g : ℝ → ℝ := λ x => a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + 5

theorem g_10_plus_g_neg10_eq_6 (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 :=
by
  -- Proof goes here
  sorry

end g_10_plus_g_neg10_eq_6_l630_630863


namespace cost_per_place_setting_l630_630427

-- Definitions based on conditions
def number_of_tables : Nat := 20
def cost_per_tablecloth : Nat := 25
def roses_per_centerpiece : Nat := 10
def cost_per_rose : Nat := 5
def lilies_per_centerpiece : Nat := 15
def cost_per_lily : Nat := 4
def place_settings_per_table : Nat := 4
def total_cost : Nat := 3500

-- Prove that the cost per place setting is $10
theorem cost_per_place_setting : 
  let total_cost_tablecloths := number_of_tables * cost_per_tablecloth,
      cost_per_centerpiece := (roses_per_centerpiece * cost_per_rose) + (lilies_per_centerpiece * cost_per_lily),
      total_cost_centerpieces := number_of_tables * cost_per_centerpiece,
      total_other_costs := total_cost_tablecloths + total_cost_centerpieces,
      remaining_amount := total_cost - total_other_costs,
      total_place_settings := number_of_tables * place_settings_per_table in
  remaining_amount / total_place_settings = 10 := 
by sorry

end cost_per_place_setting_l630_630427


namespace prasanna_speed_l630_630023

variable (L_speed P_speed time apart : ℝ)
variable (h1 : L_speed = 40)
variable (h2 : time = 1)
variable (h3 : apart = 78)

theorem prasanna_speed :
  P_speed = apart - (L_speed * time) / time := 
by
  rw [h1, h2, h3]
  simp
  sorry

end prasanna_speed_l630_630023


namespace sequence_equality_l630_630505

noncomputable def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_equality (x : ℝ) (hx : x = 0 ∨ x = 1 ∨ x = -1) (n : ℕ) (hn : n ≥ 3) :
  (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end sequence_equality_l630_630505


namespace angle_BFC_135_l630_630813

open EuclideanGeometry

-- Define the basic setup of the problem
variables {A B C D E F : Point}
variables {a b c : ℝ} -- lengths of the sides
variables [ Nonempty (Triangle A B C)] -- non-empty triangle A B C

-- Define the conditions
axiom angle_BAC_eq_90 : ∠BAC = 90
axiom CD_eq_CA : dist C D = dist C A
axiom BE_eq_BA : dist B E = dist B A
axiom is_right_triangle_DEF : isIsoscelesRight DEF

-- Define the goal (conclusion)
theorem angle_BFC_135 :
  ∠BFC = 135 :=
sorry -- Proof to be filled in

end angle_BFC_135_l630_630813


namespace geometric_seq_sum_l630_630318

noncomputable def sum_sequence : ℕ → ℚ
| 0       := 0
| (n + 1) := sum_sequence n + 1 / ((n+1) * (n+3))

theorem geometric_seq_sum (n : ℕ) :
  (∑ k in Finset.range n, (1 : ℚ) / ((k + 1) * (k + 3))) = (3 / 4) - (2 * n + 3) / (2 * (n + 1) * (n + 2)) :=
sorry

end geometric_seq_sum_l630_630318


namespace income_growth_relation_l630_630448

-- Define all the conditions
def initial_income : ℝ := 1.3
def third_week_income : ℝ := 2
def growth_rate (x : ℝ) : ℝ := (1 + x)^2  -- Compound interest style growth over 2 weeks.

-- Theorem: proving the relationship given the conditions
theorem income_growth_relation (x : ℝ) : initial_income * growth_rate x = third_week_income :=
by
  unfold initial_income third_week_income growth_rate
  sorry  -- Proof not required.

end income_growth_relation_l630_630448


namespace triangle_area_opa_l630_630768

theorem triangle_area_opa (k m : ℝ)
  (h₁ : ∀ (x : ℝ), y = k * x + 4 ∧ P = (1, m))
  (h₂ : k = -2)
  (h₃ : ∀ (x : ℝ), y = 0 ∧ A = (x, 0)) :
  let O : (ℝ × ℝ) := (0, 0),
      P : (ℝ × ℝ) := (1, 2),
      A : (ℝ × ℝ) := (2, 0) in
  1 / 2 * (A.1 - O.1) * (P.2 - O.2) = 2 :=
sorry

end triangle_area_opa_l630_630768


namespace midpoint_circumcircle_extension_l630_630656

variables {A B C B1 C1 D1 D : Point}
variables [Triangle A B C]
variables (AB AC : ℝ) (AB1 AC1 AD AD1 : ℝ)
variables (circumcircle_ABC : Circumcircle A B C)

theorem midpoint_circumcircle_extension :
  (midpoint D1 B1 C1) ∧ (extension AB B1) ∧ (extension AC C1) ∧ (intersection AD1 circumcircle_ABC D) →
  (AB * AB1 + AC * AC1 = 2 * AD * AD1) :=
sorry

end midpoint_circumcircle_extension_l630_630656


namespace tangent_line_through_point_l630_630379

open Real

theorem tangent_line_through_point (x0 : ℝ) (h_pos : 0 < x0) (h_tangent : (λ x, log x).deriv.eval x0 * (-e - x0) = -1 - log x0) : 
  x0 = exp 1 ∧ log x0 = 1 := 
by 
  sorry

end tangent_line_through_point_l630_630379


namespace range_of_combined_set_l630_630072

-- Conditions as definitions in Lean 4

def x := {n : ℕ | nat.prime n ∧ 100 ≤ n ∧ n < 1000}
def y := {m : ℕ | m % 4 = 0 ∧ 0 < m ∧ m < 100}
def z := {p : ℕ | p % 5 = 0 ∧ 0 < p ∧ p < 200}

def combined_set := x ∪ y ∪ z

-- Prove the statement about the range of the combined set
theorem range_of_combined_set : set.range combined_set = 993 := by
  sorry

end range_of_combined_set_l630_630072


namespace parabola_focus_coordinates_l630_630097

theorem parabola_focus_coordinates :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 18) ∧ 
    ∃ (p : ℝ), y = 9 * x^2 → x^2 = 4 * p * y ∧ p = 1 / 18 :=
by
  sorry

end parabola_focus_coordinates_l630_630097


namespace trajectory_perimeter_range_area_ratio_range_l630_630749

noncomputable def trajectory_eq (x y : ℝ) := x^2 + y^2 = 4

theorem trajectory (x y : ℝ) :
  (abs (sqrt ((x+4)^2 + y^2)) = 2 * abs (sqrt ((x+1)^2 + y^2))) ↔ trajectory_eq x y := 
by
  sorry -- Proof omitted

theorem perimeter_range (x y : ℝ) (h : trajectory_eq x y) :
  6 < abs (sqrt ((x+4)^2 + y^2)) + abs (sqrt ((x+1)^2 + y^2)) < 12 := 
by
  sorry -- Proof omitted

theorem area_ratio_range (x y m : ℝ) (h : trajectory_eq x y) :
  let y2 := (-2 * m + m^2 + 1) / (m^2 + 1)
  let t := abs y / abs y2
  (1 / 3) < t ∧ t < 3 :=
by
  sorry -- Proof omitted

end trajectory_perimeter_range_area_ratio_range_l630_630749


namespace fraction_spent_on_furniture_l630_630423

variable (original_savings : ℕ)
variable (tv_cost : ℕ)
variable (f : ℚ)

-- Defining the conditions
def conditions := original_savings = 500 ∧ tv_cost = 100 ∧
  f = (original_savings - tv_cost) / original_savings

-- The theorem we want to prove
theorem fraction_spent_on_furniture : conditions original_savings tv_cost f → f = 4 / 5 := by
  sorry

end fraction_spent_on_furniture_l630_630423


namespace license_plates_count_expected_license_plates_l630_630210

theorem license_plates_count :
  ∃ (num_plates : ℕ), 
    num_plates = 47_320_000 ∧ 
    ∃ (two_letters : Fin 676) 
      (five_free_digits : Fin (10 ^ 5))
      (pos_two_letters : Fin 6)
      (pos_seven : Fin 7),
    True :=
by
  use 47_320_000
  have h1 : ∃ (two_letters : Fin 676), True, from by simp
  have h2 : ∃ (five_free_digits : Fin (10 ^ 5)), True, from by simp
  have h3 : ∃ (pos_two_letters : Fin 6), True, from by simp
  have h4 : ∃ (pos_seven : Fin 7), True, from by simp
  exact ⟨47_320_000, ⟨h1, h2, h3, h4⟩⟩

theorem expected_license_plates : 
  ∃ (num_plates : ℕ), num_plates = 47_320_000 :=
  ⟨47_320_000, by refl⟩

end license_plates_count_expected_license_plates_l630_630210


namespace jamies_father_days_to_lose_weight_l630_630842

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def calories_burned_per_day : ℕ := 2500
def calories_consumed_per_day : ℕ := 2000
def net_calories_burned_per_day : ℕ := calories_burned_per_day - calories_consumed_per_day
def total_calories_to_burn : ℕ := pounds_to_lose * calories_per_pound
def days_to_burn_calories := total_calories_to_burn / net_calories_burned_per_day

theorem jamies_father_days_to_lose_weight : days_to_burn_calories = 35 := by
  sorry

end jamies_father_days_to_lose_weight_l630_630842


namespace decimal_to_binary_equivalent_123_l630_630534

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630534


namespace sum_of_two_primes_l630_630491

theorem sum_of_two_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 93) : p * q = 178 := 
sorry

end sum_of_two_primes_l630_630491


namespace enclosed_area_is_correct_l630_630456

-- Define the given properties and results

def arc_length := (Real.pi / 2)
def side_length := 3
def num_arcs := 16
def radius := arc_length / (Real.pi / 2)
def sector_area := (1 / 4) * Real.pi
def total_sector_area := num_arcs * sector_area
def octagon_area := 54 + 38.4 * Real.sqrt 2
def enclosed_area := octagon_area + total_sector_area

-- Prove the enclosed area
theorem enclosed_area_is_correct: enclosed_area = 54 + 38.4 * Real.sqrt 2 + 4 * Real.pi :=
  by
    sorry

end enclosed_area_is_correct_l630_630456


namespace cylinder_radius_in_cone_l630_630627

-- Definitions of given conditions
def diameter_cone := 15
def height_cone := 15
def radius_cylinder := 15 / 4

-- The problem statement
theorem cylinder_radius_in_cone :
  ∃ r : ℚ, r = radius_cylinder ∧
            -- Definitions and conditions filled in as axioms or assumptions
            (∃ height_cylinder : ℚ, height_cylinder = 2 * r) ∧
            (let radius_cone := diameter_cone / 2 in radius_cone = 15 / 2) ∧
            (let altitude_cone := height_cone in altitude_cone = 15) ∧
            ((15 - 2 * r) / r = 2) :=
sorry

end cylinder_radius_in_cone_l630_630627


namespace real_part_of_one_over_one_minus_z_l630_630416

open Complex

noncomputable def real_part_fraction {z : ℂ} (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) : ℝ :=
  re (1 / (1 - z))

theorem real_part_of_one_over_one_minus_z (z : ℂ) (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) :
  real_part_fraction hz1 hz2 = 1 / 2 :=
by
  sorry

end real_part_of_one_over_one_minus_z_l630_630416


namespace remainder_of_3_pow_2023_mod_5_l630_630164

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l630_630164


namespace fewest_printers_l630_630601

theorem fewest_printers (x y : ℕ) (h : 8 * x = 7 * y) : x + y = 15 :=
sorry

end fewest_printers_l630_630601


namespace base_2_representation_of_123_is_1111011_l630_630528

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l630_630528


namespace decimal_to_binary_equivalent_123_l630_630531

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630531


namespace equilateral_triangle_CDE_l630_630228

variables {α : Type*} [EuclideanPlane α]
variables (A B C D E : α)

-- Define the conditions
axiom equilateral_triangle_ABC : equilateral_triangle A B C
axiom D_on_line_through_B_parallel_AC : ∃ l : line α, l ∥ line_through A C ∧ 
  point_on_line D l ∧ point_on_line B l
axiom D_C_same_side_AB : same_side C D (line_through A B)
axiom E_on_perpendicular_bisector_CD : ∃ m : line α, m = perpendicular_bisector C D ∧ 
  point_on_line E m ∧ point_on_line E (line_through A B)

-- The theorem we aim to prove
theorem equilateral_triangle_CDE : equilateral_triangle C D E :=
sorry

end equilateral_triangle_CDE_l630_630228


namespace roots_of_unity_in_cubic_polynomial_l630_630629

-- Definition of a root of unity
def is_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

-- Given polynomial
def polynomial (z : ℂ) (p q : ℤ) : Prop := z^3 + (p : ℂ) * z + (q : ℂ) = 0

-- Main statement
theorem roots_of_unity_in_cubic_polynomial :
  ∃ (p q : ℤ), ∀ (z : ℂ), (is_root_of_unity z 1 ∨ is_root_of_unity z 3) → polynomial z p q →
    count (λ z, polynomial z p q) (λ z, is_root_of_unity z 1 ∨ is_root_of_unity z 3) = 3 := sorry

end roots_of_unity_in_cubic_polynomial_l630_630629


namespace smallest_five_digit_multiple_of_53_l630_630557

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l630_630557


namespace am_gm_problem_l630_630110

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end am_gm_problem_l630_630110


namespace car_speed_l630_630199

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end car_speed_l630_630199


namespace reciprocal_of_neg_2023_l630_630486

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l630_630486


namespace balls_in_boxes_l630_630343

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end balls_in_boxes_l630_630343


namespace s_is_negation_of_t_l630_630362

variable (m n : Prop)
variable (p : Prop) := m → n
variable (r : Prop) := ¬m → ¬n
variable (s : Prop) := ¬n → ¬m
variable (t : Prop) := n → m

theorem s_is_negation_of_t :
  s = ¬t :=
sorry

end s_is_negation_of_t_l630_630362


namespace club_additional_members_l630_630599

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l630_630599


namespace sum_of_rationals_l630_630183

theorem sum_of_rationals (r1 r2 : ℚ) : ∃ r : ℚ, r = r1 + r2 :=
sorry

end sum_of_rationals_l630_630183


namespace greatest_value_divisible_by_3_l630_630357

theorem greatest_value_divisible_by_3 :
  ∃ (a : ℕ), (168026 + 1000 * a) % 3 = 0 ∧ a ≤ 9 ∧ ∀ b : ℕ, (168026 + 1000 * b) % 3 = 0 → b ≤ 9 → a ≥ b :=
sorry

end greatest_value_divisible_by_3_l630_630357


namespace sum_of_dog_cages_is_28_l630_630734

def num_cages := 12
def num_cats := 4
def num_dogs := 4
def num_mice := 4

def is_noisy (cage : ℕ) : Prop :=
  cage = 3 ∨ cage = 4 ∨ cage = 6 ∨ cage = 7 ∨ cage = 8 ∨ cage = 9

-- Define if a cat and a mouse are in the same column
def cat_mouse_in_same_column (cats mice : fin num_cages → ℕ) (c : fin num_cages) : Prop :=
  cats c = c ∧ mice c = c

-- Define if a mouse is between two cats, left and right
def mouse_between_cats (cats mice : fin num_cages → ℕ) (c : fin num_cages) : Prop :=
  (c.val > 0 ∧ c.val < num_cages - 1) ∧ (cats ⟨c.val - 1, by linarith⟩ = c.val - 1) ∧ 
  (cats ⟨c.val + 1, by linarith⟩ = c.val + 1) ∧ (mice c = c)

-- Define if a dog is between a cat and mouse
def dog_between_cat_mouse (cats dogs mice : fin num_cages → ℕ) (c : fin num_cages) : Prop :=
  (cats ⟨c.val - 1, by linarith⟩ = c.val - 1) ∧ (mice ⟨c.val + 1, by linarith⟩ = c.val + 1) ∧ (dogs c = c)

theorem sum_of_dog_cages_is_28 
  (cats dogs mice : fin num_cages → ℕ)
  (h_noisy : ∀ (c : fin num_cages), is_noisy c.val → 
            (cat_mouse_in_same_column cats mice c) ∨ 
            (mouse_between_cats cats mice c) ∨ 
            (dog_between_cat_mouse cats dogs mice c)) :
  ∑ i in finset.filter (λ j, dogs j.val = j.val) (finset.fin_range num_cages), i.val = 28 := 
sorry

end sum_of_dog_cages_is_28_l630_630734


namespace find_shifted_poly_l630_630403

/-- Let a, b, and c be the roots of the polynomial x^3 - 5x + 7 = 0.
    To prove that the monic polynomial whose roots are a + 3, b + 3, and c + 3 is x^3 - 9x^2 + 22x - 5.  --/
theorem find_shifted_poly (a b c : ℝ)
    (h_roots : ∀ x, (Polynomial.X ^ 3 - 5 * Polynomial.X + 7).eval x = 0 ↔ x = a ∨ x = b ∨ x = c) :
    (Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 22 * Polynomial.X - 5).roots = {a + 3, b + 3, c + 3} := 
sorry

end find_shifted_poly_l630_630403


namespace log_domain_l630_630906

def domain_of_log_function (f : ℝ → ℝ) : Set ℝ := {x | x > 1}

theorem log_domain :
  domain_of_log_function (λ x, Real.log x / Real.log 2) = {x : ℝ | x > 1} :=
sorry

end log_domain_l630_630906


namespace cleanup_event_results_l630_630457

/-
Question: 
1. Calculate the total number of children involved in the cleaning event.
2. Calculate the total number of teenagers involved in the cleaning event.
3. Calculate the total amount of recyclable materials collected by children.
4. Calculate the total amount of mixed trash collected by teenagers.
5. Calculate the total amount of various trash collected by seniors.
-/
/-
Conditions: 
1. Out of 2000 community members, 30% were adult men.
2. There were twice as many adult women as adult men.
3. Adult men collected an average of 4 pounds of heavy trash per person.
4. Adult women collected an average of 2.5 pounds of light trash per person.
5. Children collected an average of 1.5 pounds of recyclable materials per person.
6. Teenagers collected an average of 3 pounds of mixed trash per person.
7. Seniors contributed an average of 1 pound of various trash per person.
8. 5% of the participants were seniors.
9. The ratio of the number of children to teenagers was 3:2.
10. Each participant spent 4 hours on their respective tasks.
-/

-- Definitions from conditions
def total_members := 2000
def percent_adult_men := 0.3
def adult_men := percent_adult_men * total_members
def adult_women := 2 * adult_men
def percent_seniors := 0.05
def seniors := percent_seniors * total_members
def remaining_members := total_members - (adult_men + adult_women + seniors)
def ratio_children_to_teenagers := (3, 2)
def x := remaining_members / (ratio_children_to_teenagers.1 + ratio_children_to_teenagers.2)
def children := ratio_children_to_teenagers.1 * x
def teenagers := ratio_children_to_teenagers.2 * x
def child_recyclable_materials := 1.5
def teenager_mixed_trash := 3
def senior_various_trash := 1
def total_child_recyclable_materials := children * child_recyclable_materials
def total_teenager_mixed_trash := teenagers * teenager_mixed_trash
def total_senior_various_trash := seniors * senior_various_trash

-- State the theorem to be proved
theorem cleanup_event_results :
  children = 60 ∧
  teenagers = 40 ∧
  total_child_recyclable_materials = 90 ∧
  total_teenager_mixed_trash = 120 ∧
  total_senior_various_trash = 100 := sorry

end cleanup_event_results_l630_630457


namespace quadratic_trinomials_constant_l630_630116

open Real

noncomputable def quadratic_side_conditions (f g h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + g x > h x ∧ f x + h x > g x ∧ g x + h x > f x

noncomputable def quadratic_not_side_conditions (f g h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f x - 1) + (g x - 1) ≤ (h x - 1) ∧
           (f x - 1) + (h x - 1) ≤ (g x - 1) ∧
           (g x - 1) + (h x - 1) ≤ (f x - 1)

theorem quadratic_trinomials_constant
  (f g h : ℝ → ℝ)
  (hf : ∀ x, ∃ a b c, f x = a * x^2 + b * x + c)
  (hg : ∀ x, ∃ a b c, g x = a * x^2 + b * x + c)
  (hh : ∀ x, ∃ a b c, h x = a * x^2 + b * x + c)
  (h_side : quadratic_side_conditions f g h)
  (h_not_side : quadratic_not_side_conditions f g h) :
  ∃ k, (λ x, f x + g x - h x = k) ∨ (λ x, f x + h x - g x = k) ∨ (λ x, g x + h x - f x = k) := sorry

end quadratic_trinomials_constant_l630_630116


namespace pilot_fish_speed_when_moved_away_l630_630847

/-- Conditions -/
def keanu_speed : ℕ := 20
def shark_new_speed (k : ℕ) : ℕ := 2 * k
def pilot_fish_increase_speed (k s_new : ℕ) : ℕ := k + (s_new - k) / 2

/-- The problem statement to prove -/
theorem pilot_fish_speed_when_moved_away (k : ℕ) (s_new : ℕ) (p_new : ℕ) 
  (h1 : k = 20) 
  (h2 : s_new = shark_new_speed k) 
  (h3 : p_new = pilot_fish_increase_speed k s_new) : 
  p_new = 30 :=
by
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3]
  sorry

end pilot_fish_speed_when_moved_away_l630_630847


namespace find_k_l630_630192

theorem find_k (α β k : ℝ) (h₁ : α^2 - α + k - 1 = 0) (h₂ : β^2 - β + k - 1 = 0) (h₃ : α^2 - 2*α - β = 4) :
  k = -4 :=
sorry

end find_k_l630_630192


namespace Tim_score_30_l630_630133

theorem Tim_score_30 (n : ℕ) : 
  (∑ k in Finset.range n, 2 * (k + 1)) = 30 ↔ n = 5 :=
by
  sorry

end Tim_score_30_l630_630133


namespace log_base_change_l630_630741

theorem log_base_change (a b : ℝ) (h₁ : Real.log 5 / Real.log 3 = a) (h₂ : Real.log 7 / Real.log 3 = b) :
    Real.log 35 / Real.log 15 = (a + b) / (1 + a) :=
by
  sorry

end log_base_change_l630_630741


namespace decimal_to_binary_equivalent_123_l630_630532

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l630_630532


namespace club_additional_members_l630_630595

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l630_630595


namespace sum_powers_divisible_by_5_iff_l630_630885

theorem sum_powers_divisible_by_5_iff (n : ℕ) (h_pos : n > 0) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_divisible_by_5_iff_l630_630885


namespace sample_size_stratified_sampling_l630_630792

theorem sample_size_stratified_sampling (n : ℕ) 
  (total_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (middle_aged_sample : ℕ)
  (stratified_sampling : n * middle_aged_employees = middle_aged_sample * total_employees)
  (total_employees_pos : total_employees = 750)
  (middle_aged_employees_pos : middle_aged_employees = 250) :
  n = 15 := 
by
  rw [total_employees_pos, middle_aged_employees_pos] at stratified_sampling
  sorry

end sample_size_stratified_sampling_l630_630792


namespace probability_red_side_l630_630979

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end probability_red_side_l630_630979


namespace num_positive_int_values_for_expression_is_7_l630_630721

theorem num_positive_int_values_for_expression_is_7 :
  {N : ℕ // 0 < N ∧ ∃ k : ℕ, 48 = k * (N + 3)}.card = 7 := 
sorry

end num_positive_int_values_for_expression_is_7_l630_630721


namespace squares_on_sides_of_triangle_l630_630079

theorem squares_on_sides_of_triangle (A B C : ℕ) (hA : A = 3^2) (hB : B = 4^2) (hC : C = 5^2) : 
  A + B = C :=
by 
  rw [hA, hB, hC] 
  exact Nat.add_comm 9 16 ▸ rfl

end squares_on_sides_of_triangle_l630_630079


namespace ellipse_equation_and_fixed_line_l630_630752

theorem ellipse_equation_and_fixed_line (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_ineq : a > b)
  (P : ℝ × ℝ) (P_eq : P = (8 / 3, -b / 3)) (B : ℝ × ℝ) (B_eq : B = (0, -b))
  (BF2 : ℝ × ℝ) (F2 : ℝ × ℝ)
  (h_perpendicular : BF2.1 = (F2.1 - P.1) / 2 ∧ BF2.2 = -1 / ((F2.2 - P.2) / b)) :
  (∃ a² = 8 ∧ b² = 4) ∧ (∀ Q : ℝ × ℝ, (∀ M : ℝ × ℝ, M = (0, 1) → ∀ E D : ℝ × ℝ, 
  ((E.1 = D.1 ∧ E.1 = 0) ∨ 
  (E.2 = DX(D) + 1 → x² + 2y² = 8 )) ∧ 
  (ƛ (Q,E) (Q,D) ((Q,M)} / x) (Q,E) Q, M, Q.1 = 0) -->  (fixed line Q, some 4) :
  sorry

end ellipse_equation_and_fixed_line_l630_630752


namespace P_is_symmetric_l630_630217

-- Define the sequence of polynomials recursively
noncomputable def P : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, x, y, z => 1
| m+1, x, y, z => (x + z) * (y + z) * P m x y (z+1) - z^2 * P m x y z

-- Define the symmetry property of the polynomials
def is_symmetric (Q : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z, Q x y z = Q x z y ∧ Q x y z = Q y x z ∧ Q x y z = Q y z x ∧ Q x y z = Q z x y ∧ Q x y z = Q z y x

-- Prove the main theorem
theorem P_is_symmetric : ∀ m, is_symmetric (P m) :=
  by
    intro m
    induction m with
    | zero => 
        -- Base case: Show P_0 is symmetric
        simp [P, is_symmetric]; tauto
    | succ m ih =>
        -- Inductive case: Assume P_m is symmetric, show P_(m+1) is symmetric
        sorry

end P_is_symmetric_l630_630217


namespace mode_of_dataset_l630_630218

theorem mode_of_dataset (x : ℝ) (h₀ : (4 + 5 + x + 7 + 9) / 5 = 6) :
  let dataset := {4, 5, x, 7, 9}
  let updated_dataset := {4, 5, 5, 7, 9}
  multiset.mode (multiset.of_finset updated_dataset finite_univ) = 5 :=
by
  sorry

end mode_of_dataset_l630_630218


namespace proof_ineq_l630_630291

variable {a b c d k : ℝ}

theorem proof_ineq (h1 : abs k < 2)
    (h2 : a^2 + b^2 - k * a * b = 1)
    (h3 : c^2 + d^2 - k * c * d = 1) : 
    abs (a * c - b * d) ≤ 2 / real.sqrt (4 - k^2) :=
  sorry

end proof_ineq_l630_630291


namespace cycle_cost_price_l630_630618

theorem cycle_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1360) 
  (h2 : loss_percentage = 0.15) :
  SP = (1 - loss_percentage) * C → C = 1600 :=
by
  sorry

end cycle_cost_price_l630_630618


namespace negative_option_is_B_l630_630227

-- Define the options as constants
def optionA : ℤ := -( -2 )
def optionB : ℤ := (-1) ^ 2023
def optionC : ℤ := |(-1) ^ 2|
def optionD : ℤ := (-5) ^ 2

-- Prove that the negative number among the options is optionB
theorem negative_option_is_B : optionB = -1 := 
by
  rw [optionB]
  sorry

end negative_option_is_B_l630_630227


namespace angle_MTD_l630_630576

open_locale real

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def point_on_ratio (B C : ℝ × ℝ) (m n : ℝ) : ℝ × ℝ :=
  ((m * C.1 + n * B.1) / (m + n), (m * C.2 + n * B.2) / (m + n))

noncomputable def parallel_line_point (C T : ℝ × ℝ) : Prop := 
  C.x = T.x

theorem angle_MTD (A B C M D T : ℝ × ℝ) 
  (hM_mid : M = midpoint A B)
  (hD_ratio : D = point_on_ratio B C 3 1)
  (hT_parallel : parallel_line_point C T)
  (hT_angle : ∠ (C - T) (A - T) = 150) :
  ∠ (M - T) (D - T) = 120 :=
sorry

end angle_MTD_l630_630576


namespace emails_received_in_afternoon_l630_630384

def num_emails_morning : ℕ := 6
def more_emails_morning_than_afternoon : ℕ := 4

theorem emails_received_in_afternoon : ∃ A : ℕ, num_emails_morning = A + more_emails_morning_than_afternoon ∧ A = 2 :=
by
  let A := 2
  use A
  split
  · exact rfl
  · sorry

end emails_received_in_afternoon_l630_630384


namespace complex_modulus_squared_l630_630689

theorem complex_modulus_squared :
  (Complex.abs (3/4 + 3 * Complex.i))^2 = 153 / 16 := by
  sorry

end complex_modulus_squared_l630_630689


namespace clicks_time_l630_630117

theorem clicks_time (x : ℕ) : 
    (∃ t : ℕ, t * (1000 / 15) = x * (1000 / 60) * 60) :=
by
    -- Definitions based on conditions
    let rail_length := 15
    let click_distance := rail_length
    let speed_km_per_hr := x
    let speed_m_per_min := (1000 * speed_km_per_hr) / 60
    let clicks_per_min := speed_m_per_min / click_distance
    let clicks_per_sec := clicks_per_min / 60
    -- translating the question to finding t such that this equation holds
    have h : (54 : ℕ) * clicks_per_sec = x, from sorry
    use 54
    exact h

end clicks_time_l630_630117


namespace set_equality_l630_630790

open Set

namespace Proof

variables (U M N : Set ℕ) 
variables (U_univ : U = {1, 2, 3, 4, 5, 6})
variables (M_set : M = {2, 3})
variables (N_set : N = {1, 3})

theorem set_equality :
  {4, 5, 6} = (U \ M) ∩ (U \ N) :=
by
  rw [U_univ, M_set, N_set]
  sorry

end Proof

end set_equality_l630_630790


namespace integral_cos_sin_squared_l630_630269

-- Define the function to be integrated
def integrand (x : ℝ) : ℝ := cos x * (sin x)^2

-- State the main theorem
theorem integral_cos_sin_squared :
  ∫ x in set.Ici 0, integrand x = (sin x)^3 / 3 + C :=
begin
  sorry
end

end integral_cos_sin_squared_l630_630269


namespace no_such_arrangement_exists_l630_630831

theorem no_such_arrangement_exists :
  ¬ ∃ (f : ℕ → ℕ) (c : ℕ), 
    (∀ n, 1 ≤ f n ∧ f n ≤ 1331) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f ((x+1) + 11 * y + 121 * z) = c + 8) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f (x + 11 * (y+1) + 121 * z) = c + 9) :=
sorry

end no_such_arrangement_exists_l630_630831


namespace center_number_l630_630823

theorem center_number (n : ℕ) (a : ℕ) (M : ℤ) (C : ℤ)
    (h1 : n = 2015)
    (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∏ j in finset.range (n + 1), ((i, j) : ℤ) = a)
    (h3 : ∀ j : ℕ, 1 ≤ j ∧ j ≤ n → ∏ i in finset.range (n + 1), ((i, j) : ℤ) = a)
    (h4 : ∀ i j : ℕ, 2 ≤ i ∧ i ≤ n - 1 ∧ 2 ≤ j ∧ j ≤ n - 1 → ∏ p q in finset.range' 0 3, ((i + p, j + q) : ℤ) = M)
    (h5 : a = ʡ.to_int 2)
    (h6 : M = ʡ.to_int 1)
    (h7 : C = (2 : ℤ)^(-2017)) : true :=
by
  sorry

end center_number_l630_630823


namespace det_of_trig_matrix_l630_630688

open Real

-- Define the matrix and its determinant
def mat (α φ β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    [cos (α + φ) * cos (β + φ), cos (α + φ) * sin (β + φ), -sin (α + φ)],
    [-sin (β + φ), cos (β + φ), 0],
    [sin (α + φ) * cos (β + φ), sin (α + φ) * sin (β + φ), cos (α + φ)]
  ]

-- Define the problem statement
theorem det_of_trig_matrix (α φ β : ℝ) : Matrix.det (mat α φ β) = 1 := by
  sorry

end det_of_trig_matrix_l630_630688


namespace sum_smallest_largest_terms_l630_630899

theorem sum_smallest_largest_terms :
  let seq := {n | 1 ≤ n ∧ n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3},
      smallest_term := Nat.find (by
        apply Exists.intro 8 _;
        repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num }),
      largest_term := Nat.find (by
        apply Exists.intro 188 _;
        repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num })
  in smallest_term + largest_term = 196 :=
by
  let seq := {n | 1 ≤ n ∧ n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3}
  let smallest_term := Nat.find (by
    apply Exists.intro 8 _
    repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num })
  let largest_term := Nat.find (by
    apply Exists.intro 188 _
    repeat { try {split}; simp [Finset.mem_coe, seq]; norm_num })
  have : smallest_term = 8 := by sorry
  have : largest_term = 188 := by sorry
  rw [this, this]
  norm_num

end sum_smallest_largest_terms_l630_630899


namespace intersection_of_A_and_B_l630_630305

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x > 4}
  let B := {x : ℝ | x ∈ ℤ ∧ 3 < x ∧ x < 7}
  A ∩ B = {5, 6} :=
by
  -- definitions
  let A := {x : ℝ | x > 4}
  let B := {x : ℝ | x ∈ ℤ ∧ 3 < x ∧ x < 7}
  -- goal
  show A ∩ B = {5, 6}
  -- proof omitted
  sorry

end intersection_of_A_and_B_l630_630305


namespace area_quadrilateral_eq_sqrt_abcd_l630_630925

variable {a b c d : ℝ}

-- Given conditions
def is_incircle_circircle_quadrilateral (a b c d : ℝ) : Prop :=
  -- Conditions for the quadrilateral allowing circles to be inscribed and circumscribed
  (a + c = b + d)

-- The theorem to prove
theorem area_quadrilateral_eq_sqrt_abcd
  (h : is_incircle_circircle_quadrilateral a b c d) :
  let area := (√ (a * b * c * d)) in
  true := sorry

end area_quadrilateral_eq_sqrt_abcd_l630_630925
