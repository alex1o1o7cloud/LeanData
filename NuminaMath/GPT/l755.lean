import Mathlib
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Algebra.Probability
import Mathlib.Analysis.Probability.Basic
import Mathlib.Analysis.SpecialFunctions.Complex
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialLemmas
import Mathlib.Combinatorics.Composition
import Mathlib.Combinatorics.Partition
import Mathlib.Data.Angle
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Logic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Variables
import Mathlib.Tactic
import Mathlib.Topology.InfiniteSum
import Mathlib.Topology.MetricSpace.Basic

namespace probability_at_least_four_same_face_l755_755733

theorem probability_at_least_four_same_face :
  let total_outcomes := (2 : ℕ) ^ 5,
      favorable_outcomes := 1 + 1 + (Nat.choose 5 1) + (Nat.choose 5 1),
      probability := favorable_outcomes / total_outcomes in
  probability = (3 : ℚ) / 8 :=
by
  sorry

end probability_at_least_four_same_face_l755_755733


namespace age_of_new_person_l755_755132

theorem age_of_new_person (T A : ℕ) (h1 : (T / 10 : ℤ) - 3 = (T - 40 + A) / 10) : A = 10 := 
sorry

end age_of_new_person_l755_755132


namespace summation_is_500_l755_755752

def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

theorem summation_is_500 : 
  (finset.sum (finset.range 1000) (λ i, f (i+1) / 1001)) = 500 := 
sorry

end summation_is_500_l755_755752


namespace domain_of_func_l755_755562

-- Define the function
def func (x : ℝ) : ℝ := Real.tan ((π / 6) * x + π / 3)

-- Define the set of problematic points where the function is undefined
def problem_points : Set ℝ := { x | ∃ k : ℤ, x = 1 + 6 * k }

-- Define the domain of the function
def func_domain : Set ℝ := { x | ¬ (x ∈ problem_points) }

-- Formulate the statement
theorem domain_of_func :
  (SetOf (λ x, func x)).domain = func_domain := 
  sorry

end domain_of_func_l755_755562


namespace thomas_total_bill_l755_755205

def shipping_cost (purchase_price : ℝ) : ℝ :=
  if purchase_price < 50 then 5 else 0.20 * purchase_price

def total_purchase_cost : ℝ :=
  (3 * 12) + 5 + (2 * 15) + 14

def total_bill (purchase_price : ℝ) : ℝ :=
  purchase_price + shipping_cost purchase_price

theorem thomas_total_bill : total_bill total_purchase_cost = 102 := by
  sorry

end thomas_total_bill_l755_755205


namespace int_values_satisfying_x4_l755_755711

theorem int_values_satisfying_x4 :
  {x : ℤ | -100 < x^4 ∧ x^4 < 100}.finite.card = 7 := 
sorry

end int_values_satisfying_x4_l755_755711


namespace each_person_gets_9_wings_l755_755169

noncomputable def chicken_wings_per_person (initial_wings : ℕ) (additional_wings : ℕ) (friends : ℕ) : ℕ :=
  (initial_wings + additional_wings) / friends

theorem each_person_gets_9_wings :
  chicken_wings_per_person 20 25 5 = 9 :=
by
  sorry

end each_person_gets_9_wings_l755_755169


namespace initial_lives_l755_755857

theorem initial_lives (x : ℕ) 
  (h1 : 6 ∈ ℕ) 
  (h2 : 11 ∈ ℕ) 
  (h3 : x + 6 + 11 = 19) : 
  x = 2 :=
by
  sorry

end initial_lives_l755_755857


namespace trapezium_angle_equality_l755_755898

variables {Point : Type} [EuclideanGeometry Point]

/-- Let ABCD be a trapezium with AD parallel to BC. Suppose K and L are points on AB and CD 
respectively, such that ∠BAL = ∠CDK. Then ∠BLA = ∠CKD. -/
theorem trapezium_angle_equality (A B C D K L : Point) 
  (h_parallel : AD ∥ BC)
  (hK : K ∈ line_segment A B) 
  (hL : L ∈ line_segment C D) 
  (h_angle : ∠BAL = ∠CDK) : 
  ∠BLA = ∠CKD := 
sorry

end trapezium_angle_equality_l755_755898


namespace car_x_distance_from_y_start_l755_755619

def car_x_speed : ℝ := 35 -- miles per hour
def car_y_speed : ℝ := 41 -- miles per hour
def time_diff : ℝ := 72 / 60 -- hours

theorem car_x_distance_from_y_start : 
  (35 * 7 : ℝ) = 245 :=
by sorry

end car_x_distance_from_y_start_l755_755619


namespace smallest_n_term_decimal_contains_three_l755_755102

-- Definition of a terminating decimal number
def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

-- Definition of containing the digit '3'
def contains_digit_three (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ≠ 0 ∧ (string.contains ("3".to_list.to_set) ((n.to_nat % 10).digits_to_string.head?))

-- The main theorem statement
theorem smallest_n_term_decimal_contains_three :
  ∃ n : ℕ, is_terminating_decimal n ∧ contains_digit_three n ∧ (∀ m : ℕ, is_terminating_decimal m ∧ contains_digit_three m → n ≤ m) ∧ n = 3125 :=
  sorry

end smallest_n_term_decimal_contains_three_l755_755102


namespace x_squared_minus_y_squared_l755_755839

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l755_755839


namespace zero_point_in_interval_l755_755979

noncomputable def f (x : ℝ) : ℝ := real.log x + 2 * x - 6

theorem zero_point_in_interval : ∃ x ∈ set.Ioo 2 3, f x = 0 :=
begin
  sorry
end

end zero_point_in_interval_l755_755979


namespace shaded_areas_l755_755557

-- Let the isosceles triangle be denoted as △PQR with angle P = 120 degrees.
-- Let the equilateral triangle be △KLM with area 36 and K, M as midpoints of PQ and PR.

def isoscelesTriangle (P Q R : Point) (hPQR : angle Q P R = 120) : Triangle := sorry

def equilateralTriangle (K L M : Point) (area_KLM : area (triangle K L M) = 36)
  (midK : midpoint K PQ) (midM : midpoint M PR) : Triangle := sorry

theorem shaded_areas {P Q R K L M : Point} (isosceles : isoscelesTriangle P Q R)
  (equilateral : equilateralTriangle K L M)
  (midK : midpoint K PQ) (midM : midpoint M PR) :
  area (isoscelesTriangle P Q R) - area (equilateralTriangle K L M) = 28 := 
sorry

end shaded_areas_l755_755557


namespace sum_of_three_consecutive_numbers_as_product_l755_755775

theorem sum_of_three_consecutive_numbers_as_product (n : ℕ) (h : n > 100) :
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (a > 1) ∧ (b > 1) ∧ (c > 1)
    ∧ (∃ s ∈ ({n, n+1, n+2, n+3}.subsetsOfCard 3), s.sum = a * b * c) :=
by
  sorry

end sum_of_three_consecutive_numbers_as_product_l755_755775


namespace square_area_condition_l755_755193

theorem square_area_condition (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (hline : x1 ≠ x2) : 
  let side := abs (x1 - x2) in side^2 = 36 :=
by
  sorry

end square_area_condition_l755_755193


namespace recurring_decimal_to_fraction_l755_755317

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755317


namespace quadratic_real_equal_roots_l755_755717

theorem quadratic_real_equal_roots (m : ℝ) :
  (∃ x : ℝ, 3*x^2 + (2*m-5)*x + 12 = 0) ↔ (m = 8.5 ∨ m = -3.5) :=
sorry

end quadratic_real_equal_roots_l755_755717


namespace distance_difference_l755_755597

def distance_between_stations : ℕ := 444
def speed_first_train : ℕ := 16
def speed_second_train : ℕ := 21
def time_to_meet : ℕ := distance_between_stations / (speed_first_train + speed_second_train)

def distance_first_train : ℕ := speed_first_train * time_to_meet
def distance_second_train : ℕ := speed_second_train * time_to_meet

theorem distance_difference : distance_second_train - distance_first_train = 60 := by
  unfold time_to_meet
  unfold distance_first_train
  unfold distance_second_train
  calc
    distance_second_train - distance_first_train
        = (21 * (444 / 37)) - (16 * (444 / 37)) : by rfl
    ... = (21 * 12) - (16 * 12)              : by norm_num
    ... = 252 - 192                          : by norm_num
    ... = 60                                 : by norm_num

end distance_difference_l755_755597


namespace odd_function_expression_l755_755049

theorem odd_function_expression {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_neg : ∀ x < 0, f x = x^3 + 2^x - 1) :
  ∀ x > 0, f x = x^3 - 2^(-x) + 1 :=
by {
  intro x,
  assume h_pos : x > 0,
  have h_expr_neg : f (-x) = (-x)^3 + 2^(-x) - 1, from h_neg (-x) (neg_lt_zero.mpr h_pos),
  rw [neg_cube, h_odd, neg_eq_neg_one, neg_add_rev, inv_eq_one_div],
  exact sorry
}

end odd_function_expression_l755_755049


namespace sin_2theta_plus_pi_over_3_value_l755_755415

noncomputable def sin_2theta_plus_pi_over_3 (θ : ℝ) : ℝ :=
  sin (2 * θ + π / 3)

theorem sin_2theta_plus_pi_over_3_value (θ : ℝ) (h : tan θ = 3) :
  sin_2theta_plus_pi_over_3 θ = (3 - 4 * √3) / 10 :=
by sorry

end sin_2theta_plus_pi_over_3_value_l755_755415


namespace incident_ray_equation_l755_755180

open Real

noncomputable def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def line_equation (A B : ℝ × ℝ) : ℝ → ℝ → Prop :=
  λ x y, (B.2 - A.2) * (x - A.1) = (y - A.2) * (B.1 - A.1)

theorem incident_ray_equation :
  let A := (-2, 3)
  let B := (5, 7)
  let B' := reflect_across_x_axis B
  line_equation A B' 10 -1 := sorry

end incident_ray_equation_l755_755180


namespace repeating_decimal_equiv_fraction_l755_755290

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755290


namespace cake_and_milk_tea_cost_l755_755954

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end cake_and_milk_tea_cost_l755_755954


namespace consecutive_integer_cubes_sum_l755_755576

theorem consecutive_integer_cubes_sum : 
  ∀ (a : ℕ), 
  (a > 2) → 
  (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2)) →
  ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3) = 224 :=
by
  intro a ha h
  sorry

end consecutive_integer_cubes_sum_l755_755576


namespace pictures_left_after_deletion_l755_755633

variable (zoo museum deleted : ℕ)

def total_pictures_taken (zoo museum : ℕ) : ℕ := zoo + museum

def pictures_remaining (total deleted : ℕ) : ℕ := total - deleted

theorem pictures_left_after_deletion (h1 : zoo = 50) (h2 : museum = 8) (h3 : deleted = 38) :
  pictures_remaining (total_pictures_taken zoo museum) deleted = 20 :=
by
  sorry

end pictures_left_after_deletion_l755_755633


namespace max_area_of_triangle_ABC_l755_755851

noncomputable def triangle_area_maximum (a b c : ℝ) : ℝ :=
  if h : 7 * a^2 + b^2 + c^2 = 4 * real.sqrt 3 ∧ b = c then sqrt(5) / 5 else 0

theorem max_area_of_triangle_ABC
  (a b c : ℝ)
  (h₁ : b = c)
  (h₂ : 7 * a^2 + b^2 + c^2 = 4 * real.sqrt 3) :
  triangle_area_maximum a b c = sqrt(5) / 5 :=
by {
  have h : 7 * a^2 + b^2 + c^2 = 4 * real.sqrt 3 ∧ b = c,
  { split; assumption },
  simp [triangle_area_maximum, h],
  sorry
}

end max_area_of_triangle_ABC_l755_755851


namespace range_of_a_for_two_roots_l755_755422

theorem range_of_a_for_two_roots (a : ℝ) :
  (∃ x y : ℝ, f(x) = 0 ∧ f(y) = 0 ∧ x > 1 ∧ y < 1) ↔ -2 < a ∧ a < 1 :=
by
  let f (x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)
  sorry

end range_of_a_for_two_roots_l755_755422


namespace number_of_cases_in_1990_l755_755467

-- Define the initial and final cases
constants (initial_cases final_cases : ℕ) (year_1970 year_2000 year_1990 : ℕ)

-- Set the values for the given constants
axiom initial_cases_def : initial_cases = 600000
axiom final_cases_def : final_cases = 200
axiom year_1970_def : year_1970 = 1970
axiom year_2000_def : year_2000 = 2000
axiom year_1990_def : year_1990 = 1990

-- Define a linear decrease in cases over time
def linear_decrease (t : ℕ) : ℕ :=
  initial_cases - (initial_cases - final_cases) * (t - year_1970) / (year_2000 - year_1970)

-- The number of cases reported in 1990
def cases_1990 : ℕ := linear_decrease year_1990

-- The proof goal to be achieved
theorem number_of_cases_in_1990 : cases_1990 = 200133 := by
  -- Proof will be supplied later
  sorry

end number_of_cases_in_1990_l755_755467


namespace area_of_trapezoid_l755_755650

theorem area_of_trapezoid (A B C D L : Point) (ω : Circle)
  (h_ABCD : Trapezoid ABCD) (h_inscribed : InscribedCircle ω ABCD) 
  (h_tangency_L : TangentPoint L ω CD) (h_CL_LD_ratio : CL / LD = 1 / 4)
  (h_BC : BC = 9) (h_CD : CD = 30) :
  area ABCD = 972 := 
by 
  sorry

end area_of_trapezoid_l755_755650


namespace probability_A_starting_A_probability_A_starting_B_l755_755176

variables (n : ℕ)
variables (p11 p12 p21 p22 : ℝ)

-- Definition for part (a)
def probability_A_after_n_steps_from_A := 
  (p21 / (p12 + p21)) + (p12 / (p12 + p21)) * (p11 - p21)^n

-- Definition for part (b)
def probability_A_after_n_steps_from_B := 
  (p21 / (p21 + p12)) - (p21 / (p21 + p12)) * (p22 - p12)^n

theorem probability_A_starting_A (n : ℕ) (p11 p12 p21 p22 : ℝ) :
  probability_A_after_n_steps_from_A n p11 p12 p21 p22 =
  (p21 / (p12 + p21)) + (p12 / (p12 + p21)) * (p11 - p21)^n :=
sorry

theorem probability_A_starting_B (n : ℕ) (p11 p12 p21 p22 : ℝ) :
  probability_A_after_n_steps_from_B n p11 p12 p21 p22 = 
  (p21 / (p21 + p12)) - (p21 / (p21 + p12)) * (p22 - p12)^n :=
sorry

end probability_A_starting_A_probability_A_starting_B_l755_755176


namespace minimal_sum_of_numbers_l755_755589

theorem minimal_sum_of_numbers :
  ∀ (n : ℕ) (a : ℕ → ℤ), 
  n = 1001 ∧ 
  (∀ i, 1 ≤ i ∧ i < n → |a i - a (i + 1)| ≥ 4 ∧ a i + a (i + 1) ≥ 6) ∧
  |a n - a 1| ≥ 4 ∧ a n + a 1 ≥ 6 → 
  (∑ i in Finset.range n, a (i + 1)) ≥ 3009 :=
by
  sorry

end minimal_sum_of_numbers_l755_755589


namespace length_of_chord_AB_equation_of_perpendicular_bisector_l755_755816

-- Conditions
def line (x y : ℝ) := x + sqrt 3 * y - 2 = 0
def circle (x y : ℝ) := x^2 + y^2 = 2

-- Theorem 1: Length of chord AB
theorem length_of_chord_AB (A B : ℝ × ℝ) (hA : line A.1 A.2) (hB : line B.1 B.2) (hA_on_circle : circle A.1 A.2) (hB_on_circle : circle B.1 B.2) : 
  dist A B = 2 :=
sorry

-- Theorem 2: Equation of the perpendicular bisector of chord AB
theorem equation_of_perpendicular_bisector : ∃ c : ℝ, (c = 0 ∧ ∀ x y : ℝ, sqrt 3 * x - y + c = 0 → circle x y) :=
sorry

end length_of_chord_AB_equation_of_perpendicular_bisector_l755_755816


namespace possible_values_of_ABCD_l755_755632

noncomputable def discriminant (a b c : ℕ) : ℕ :=
  b^2 - 4*a*c

theorem possible_values_of_ABCD 
  (A B C D : ℕ)
  (AB BC CD : ℕ)
  (hAB : AB = 10*A + B)
  (hBC : BC = 10*B + C)
  (hCD : CD = 10*C + D)
  (h_no_9 : A ≠ 9 ∧ B ≠ 9 ∧ C ≠ 9 ∧ D ≠ 9)
  (h_leading_nonzero : A ≠ 0)
  (h_quad1 : discriminant A B CD ≥ 0)
  (h_quad2 : discriminant A BC D ≥ 0)
  (h_quad3 : discriminant AB C D ≥ 0) :
  ABCD = 1710 ∨ ABCD = 1810 :=
sorry

end possible_values_of_ABCD_l755_755632


namespace decimal_89_to_binary_l755_755228

def decimal_to_binary (n : ℕ) : list ℕ :=
  if h : n = 0 then [0]
  else
    let rec digits (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc else digits (n / 2) ((n % 2) :: acc)
    digits n []

theorem decimal_89_to_binary :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end decimal_89_to_binary_l755_755228


namespace friends_volunteering_with_james_l755_755496

-- Defining the conditions
def flowers_per_day_james : ℕ := 20
def total_flowers : ℕ := 200
def days : ℕ := 2
def total_flowers_james := flowers_per_day_james * days

-- Proving the main statement
theorem friends_volunteering_with_james :
  ∃ (friends : ℕ), friends * (total_flowers / (friends + 1)) = total_flowers - total_flowers_james :=
by
  use 4,
  sorry

end friends_volunteering_with_james_l755_755496


namespace solve_eq_l755_755702

theorem solve_eq {n : ℤ} (h : (5 : ℤ) * n / 4 + 5 / 4 = n) : 
  ∃ k : ℤ, n = -5 + 1024 * k :=
begin
  sorry
end

end solve_eq_l755_755702


namespace proof_problem_l755_755004

noncomputable def binom (n k : Nat) : ℕ := Nat.choose n k

noncomputable def f (x y : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), 
    (-1) ^ Nat.floor (k / 2) * binom n k * (x ^ (n - k)) * (y ^ k)

theorem proof_problem
  (x y : ℝ) (n : ℕ)
  (h_xy_cond : x^2 + y^2 ≤ 1) :
  |f x y n| ≤ sqrt 2 :=
sorry

end proof_problem_l755_755004


namespace company_pays_per_box_per_month_l755_755646

/-
  Given:
  - The dimensions of each box are 15 inches by 12 inches by 10 inches
  - The total volume occupied by all boxes is 1,080,000 cubic inches
  - The total cost for record storage per month is $480

  Prove:
  - The company pays $0.80 per box per month for record storage
-/

theorem company_pays_per_box_per_month :
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  cost_per_box_per_month = 0.80 :=
by
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  sorry

end company_pays_per_box_per_month_l755_755646


namespace continuity_at_4_l755_755628

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end continuity_at_4_l755_755628


namespace range_of_k_l755_755817

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 1) ∧ (1 < k ∧ k < sqrt 2) :=
begin
  sorry -- Proof is not required
end

end range_of_k_l755_755817


namespace recurring_decimal_to_fraction_l755_755315

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755315


namespace shaded_fraction_l755_755538

noncomputable def fraction_shaded (l w : ℝ) : ℝ :=
  1 - (1 / 8)

theorem shaded_fraction (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  fraction_shaded l w = 7 / 8 :=
by
  sorry

end shaded_fraction_l755_755538


namespace prob_points_l755_755158

-- Definition of the sequence p_n according to the given conditions
def p : ℕ → ℝ
| 0 => 0
| 1 => 1 / 2
| 2 => 3 / 4
| (n+3) => (1 / 2) * p (n+2) + (1 / 2) * p (n+1)

-- The target theorem to prove given the recurrence relation and initial conditions
theorem prob_points (n : ℕ) : p n = (2 / 3) + (1 / 3) * ((-1 / 2) ^ n) :=
sorry

end prob_points_l755_755158


namespace find_d_squared_l755_755047

noncomputable def ellipse_foci_coincide_with_hyperbola (d : ℝ) : Prop :=
  let a_ellipse := 5
  let a_hyperbola := 13 / 4
  let b_hyperbola := 2
  let c_hyperbola := Real.sqrt ((a_hyperbola)^2 + (b_hyperbola)^2)
  let d2_ellipse := a_ellipse^2 - (c_hyperbola)^2
  d^2 = d2_ellipse

theorem find_d_squared : ∃ d : ℝ, ellipse_foci_coincide_with_hyperbola d ∧ d^2 = 215 / 16 :=
by
  sorry  -- The proof steps are omitted, this is just the statement.

end find_d_squared_l755_755047


namespace marathon_remaining_yards_l755_755225

theorem marathon_remaining_yards:
  (miles_one_marathon : ℕ) (yards_one_marathon : ℕ) (yards_per_mile : ℤ) (num_marathons : ℕ) :
  miles_one_marathon = 26 ∧ yards_one_marathon = 385 ∧ yards_per_mile = 1760 ∧ num_marathons = 15 →
  (∃ m y, (0 ≤ y ∧ y < 1760) ∧ m * yards_per_mile + y = num_marathons * (miles_one_marathon * yards_per_mile + yards_one_marathon)) →
  y = 495 :=
by
  sorry

end marathon_remaining_yards_l755_755225


namespace measure_dihedral_angle_and_distance_l755_755209

/-- In dihedral angle D-AB-E, quadrilateral ABCD is a square with side length 2,
    AE = EB, F is a point on CE, and BF ⊥ plane ACE. We're proving that:
    1. The measure of the dihedral angle B-AC-E is arcsin (sqrt 6 / 3)
    2. The distance from point D to the plane ACE is 2 * sqrt 3 / 3 -/
theorem measure_dihedral_angle_and_distance
  (A B C D E F: Point)
  (plane_ACE: Plane)
  (CE: Line)
  (side_length: ℝ)
  (AE_BE: ℝ)
  (angle_AE_BF: angle)
  (distance_D_to_ACE: ℝ)
  (h_square: square ABCD)
  (h_side_length: side_length = 2)
  (h_AE_BE: AE = EB)
  (h_F_CE: F ∈ CE)
  (h_BF_perpendicular: BF ⊥ plane_ACE) :
  measure_dihedral_angle B-AC-E = arcsin (sqrt 6 / 3) ∧
  distance D to plane_ACE = 2 * sqrt 3 / 3 :=
by {
  sorry
}

end measure_dihedral_angle_and_distance_l755_755209


namespace acetone_weight_9_moles_correct_l755_755098

def molecular_weight_acetone (C_weight H_weight O_weight : ℝ) : ℝ :=
  3 * C_weight + 6 * H_weight + O_weight

def total_weight_of_9_moles (mol_weight : ℝ) : ℝ :=
  9 * mol_weight

theorem acetone_weight_9_moles_correct (C_weight : ℝ) (H_weight : ℝ) (O_weight : ℝ) :
  (C_weight = 12.01) → (H_weight = 1.008) → (O_weight = 16.00) →
  total_weight_of_9_moles (molecular_weight_acetone C_weight H_weight O_weight) = 522.702 :=
by
  intros hC hH hO
  rw [hC, hH, hO]
  dsimp [molecular_weight_acetone, total_weight_of_9_moles]
  norm_num
  exact Eq.refl 522.702

end acetone_weight_9_moles_correct_l755_755098


namespace max_gold_received_baba_l755_755679

def sumOfProducts (f : ℕ → ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n-1), f i (n - i)

noncomputable def f (a b : ℕ) : ℕ := a * b

theorem max_gold_received_baba (n : ℕ) (h1 : n > 0) :
  sumOfProducts f n = (n - 1) * n / 2 :=
by
  sorry

end max_gold_received_baba_l755_755679


namespace pyramid_lateral_surface_area_l755_755553

variables (H α β : ℝ)
-- Conditions of the problem
def is_rectangular_base : Prop := True
def is_perpendicular_faces : Prop := True
def is_inclined_faces (α β : ℝ) : Prop := True
def pyramid_height (H : ℝ) : Prop := H > 0

-- Definition of the surface area to prove
def lateral_surface_area (H α β : ℝ) : ℝ :=
  (H^2 / 2) * (cot α + cot β + (cot β / sin α) + (cot α / sin β))

-- The theorem statement
theorem pyramid_lateral_surface_area
  (H_pos : pyramid_height H)
  (rect_base : is_rectangular_base)
  (perp_faces : is_perpendicular_faces)
  (incl_faces : is_inclined_faces α β) :
  lateral_surface_area H α β = (H^2 / 2) * (cot α + cot β + (cot β / sin α) + (cot α / sin β)) :=
sorry

end pyramid_lateral_surface_area_l755_755553


namespace repeating_decimal_eq_fraction_l755_755341

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755341


namespace find_multiplier_l755_755118

theorem find_multiplier 
  (x : ℝ)
  (number : ℝ)
  (condition1 : 4 * number + x * number = 55)
  (condition2 : number = 5.0) :
  x = 7 :=
by
  sorry

end find_multiplier_l755_755118


namespace range_of_a_l755_755428

noncomputable def f (a x : ℝ) : ℝ := Real.exp x + x^2 + (3 * a + 2) * x

theorem range_of_a
  (h_f_continuous : ∀ a : ℝ, Continuous (f a)) 
  (h_deriv : ∀ a x : ℝ, deriv (f a) x = Real.exp x + 2 * x + 3 * a + 2) 
  (h_f_min_value : ∃ a : ℝ, ∀ x ∈ Ioo (-1 : ℝ) 0, IsLocalMin (f a) x → -1 < a ∧ a < - (1 / (3 * Real.exp 1))) 
  : ∀ a : ℝ, (∃ x : ℝ, x ∈ Ioo (-1 : ℝ) 0 ∧ IsLocalMin (f a) x) ↔ (-1 < a ∧ a < - (1 / (3 * Real.exp 1))) := 
by 
  sorry

end range_of_a_l755_755428


namespace birds_more_than_storks_l755_755144

theorem birds_more_than_storks :
  let initial_birds := 2
  let additional_birds := 5
  let total_birds := initial_birds + additional_birds
  let storks := 4
  total_birds - storks = 3 :=
by
  let initial_birds := 2
  let additional_birds := 5
  let total_birds := initial_birds + additional_birds
  let storks := 4
  show total_birds - storks = 3 from sorry

end birds_more_than_storks_l755_755144


namespace gcm_of_9_and_15_less_than_120_eq_90_l755_755994

theorem gcm_of_9_and_15_less_than_120_eq_90 
  (lcm_9_15 : Nat := Nat.lcm 9 15)
  (multiples : List Nat := List.range (120 / lcm_9_15) |> List.map (λ n => n * lcm_9_15)) : 
  lcm_9_15 = 45 ∧ multiples.max = some 90 := by
sorry

end gcm_of_9_and_15_less_than_120_eq_90_l755_755994


namespace larger_diagonal_opposite_larger_angle_l755_755916

-- Defining a parallelogram and the properties we need
variables {A B C D : Type} [has_lt Type] {AB CD AD BC : ℝ}

-- Conditions
def is_parallelogram (AB CD AD BC : ℝ) : Prop := 
  AB = CD ∧ AD = BC

noncomputable def angle_ABC_gt_angle_BAD (angle_ABC angle_BAD : ℝ) : Prop :=
  angle_ABC > angle_BAD

-- Mathematical proof statement
theorem larger_diagonal_opposite_larger_angle 
  (AB CD AD BC AC BD angle_ABC angle_BAD : ℝ)
  (h1 : is_parallelogram AB CD AD BC)
  (h2 : angle_ABC_gt_angle_BAD angle_ABC angle_BAD) :
  AC > BD := by
  sorry

end larger_diagonal_opposite_larger_angle_l755_755916


namespace continuity_at_x_0_l755_755630

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end continuity_at_x_0_l755_755630


namespace fraction_eq_repeating_decimal_l755_755279

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755279


namespace total_sum_of_money_is_71_l755_755077

noncomputable def totalCoins : ℕ := 334
noncomputable def coins20Paise : ℕ := 250
noncomputable def coins25Paise : ℕ := totalCoins - coins20Paise
noncomputable def value20Paise : ℕ := coins20Paise * 20
noncomputable def value25Paise : ℕ := coins25Paise * 25
noncomputable def totalValuePaise : ℕ := value20Paise + value25Paise
noncomputable def totalValueRupees : ℚ := totalValuePaise / 100

theorem total_sum_of_money_is_71 :
  totalValueRupees = 71 := by
  sorry

end total_sum_of_money_is_71_l755_755077


namespace sqrt3_5_sub_sqrt2_root_l755_755123

noncomputable def poly (x : ℝ) := x^6 - 6 * x^4 - 10 * x^3 - 60 * x + 7

theorem sqrt3_5_sub_sqrt2_root :
  poly (real.cbrt 5 - real.sqrt 2) = 0 :=
sorry

end sqrt3_5_sub_sqrt2_root_l755_755123


namespace exists_multiple_with_only_digits_zero_and_one_l755_755026

theorem exists_multiple_with_only_digits_zero_and_one (n : ℕ) :
  ∃ m : ℕ, m % n = 0 ∧ ∀ d ∈ (nat.digits 10 m), d = 0 ∨ d = 1 :=
by
  sorry

end exists_multiple_with_only_digits_zero_and_one_l755_755026


namespace union_of_A_and_B_l755_755403

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end union_of_A_and_B_l755_755403


namespace president_vice_president_combinations_l755_755602

theorem president_vice_president_combinations :
  let candidates := {Jungkook, Jimin, Yoongi, Yuna} in
  Fintype.card ({p : candidates // p ≠ president} → fin 2) = 12 :=
by
  sorry

end president_vice_president_combinations_l755_755602


namespace x_squared_minus_y_squared_l755_755837

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l755_755837


namespace sum_even_indices_l755_755516

theorem sum_even_indices (n : ℕ) (a : ℕ → ℕ)
  (h : (1 + x + x^2)^n = ∑ i in range (2*n+1), a i * x^i) :
  a 0 = 1 →
  (∑ i in range (n+1), a (2*i)) = (3^n - 1) / 2 :=
by sorry

end sum_even_indices_l755_755516


namespace simplify_and_evaluate_l755_755545

theorem simplify_and_evaluate (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6 * m + 9) / (m - 2)) = -1/2 :=
by
  sorry

end simplify_and_evaluate_l755_755545


namespace triangle_count_lower_bound_l755_755920

theorem triangle_count_lower_bound (n m : ℕ) (S : Finset (ℕ × ℕ))
  (hS : ∀ (a b : ℕ), (a, b) ∈ S → 1 ≤ a ∧ a < b ∧ b ≤ n) (hm : S.card = m) :
  ∃T, T ≥ 4 * m * (m - n^2 / 4) / (3 * n) := 
by 
  sorry

end triangle_count_lower_bound_l755_755920


namespace minSumAreas_l755_755758

-- Define the cube with edge length 1
structure Cube where
  A B C D A1 B1 C1 D1 : ℝ

-- Condition 1: Edge length of the cube is 1
def edge_length (c : Cube) : Prop := (dist c.A c.B = 1 ∧ dist c.B c.C = 1 ∧ dist c.C c.D = 1 ∧ dist c.D c.A = 1 ∧ 
                                      dist c.A c.A1 = 1 ∧ dist c.B c.B1 = 1 ∧ dist c.C c.C1 = 1 ∧ dist c.D c.D1 = 1)

-- Condition 2: F is midpoint of BC
def midpoint_F (c : Cube) : ℝ := (c.B + c.C) / 2

-- Condition 3: M is a moving point on A1F
def moving_point_M (c : Cube) (M : ℝ) : Prop := M ∈ segment c.A1 (midpoint_F c)

-- Find the minimum value of the sum of the areas of triangle MDD1 and MCC1
noncomputable def min_sum_of_areas (c: Cube) (M : ℝ) : ℝ :=
  triangle_area M c.D c.D1 + triangle_area M c.C c.C1

-- The sum of the areas of triangles MDD1 and MCC1 is minimized at sqrt(65) / 10
theorem minSumAreas (c : Cube) (M : ℝ)
  (h_cube : edge_length c) 
  (h_mid : midpoint_F c = F) 
  (h_M : moving_point_M c M) :
  min_sum_of_areas c M = sqrt 65 / 10 := 
sorry

end minSumAreas_l755_755758


namespace fraction_eq_repeating_decimal_l755_755276

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755276


namespace sequence_lemma_l755_755819

def a_n (n : ℕ) : ℕ :=
  if n > 0 then (-1 : ℤ) ^ n + 1 else 0

theorem sequence_lemma : a_n 4 = 2 := by
  sorry

end sequence_lemma_l755_755819


namespace magnitude_of_v_l755_755939

-- Definitions and conditions
variables {u v : ℂ}
variable (h1 : u * v = 20 - 15 * complex.I)
variable (h2 : complex.abs u = 5)

-- Theorem statement
theorem magnitude_of_v : complex.abs v = 5 :=
by 
  sorry

end magnitude_of_v_l755_755939


namespace triangle_equality_l755_755548

/-- 
Given two triangles ABC and PQR where:
1. A is the midpoint of QR.
2. P is the midpoint of BC.
3. QR is the bisector of angle BAC.
4. BC is the bisector of angle QPR.

We want to prove: AB + AC = PQ + PR.
-/
theorem triangle_equality
  (A B C P Q R : Point)
  (h1 : midpoint A Q R)
  (h2 : midpoint P B C)
  (h3 : is_angle_bisector Q R)
  (h4 : is_angle_bisector B C) 
  : length A B + length A C = length P Q + length P R := 
sorry

end triangle_equality_l755_755548


namespace total_students_last_year_l755_755063

-- Define the conditions
variable (X Y : ℕ)
variable (enrolled_YY_last_year : Y = 2400)
variable (growth_XX : 1.07 * X)
variable (growth_YY : 1.03 * Y)
variable (growth_difference : 0.07 * X - 0.03 * Y = 40)

-- Prove the total number of students last year
theorem total_students_last_year (hY : Y = 2400) (h_diff : 0.07 * (X : ℝ) - 0.03 * (Y : ℝ) = 40) :
  X + Y = 4000 :=
by
  sorry

end total_students_last_year_l755_755063


namespace total_membership_is_1600_l755_755683

def total_membership (total_votes_cast : ℕ) (votes_received_percent : ℚ) (membership_received_percent : ℚ) : ℕ :=
  (votes_received_percent * total_votes_cast : ℚ) / membership_received_percent

theorem total_membership_is_1600 :
  total_membership 525 0.60 0.196875 = 1600 :=
by 
  sorry

end total_membership_is_1600_l755_755683


namespace B_2_2_eq_16_l755_755708

def B : ℕ → ℕ → ℕ
| 0, n       => n + 2
| (m+1), 0   => B m 2
| (m+1), (n+1) => B m (B (m+1) n)

theorem B_2_2_eq_16 : B 2 2 = 16 := by
  sorry

end B_2_2_eq_16_l755_755708


namespace gen_formula_Sn_bounded_l755_755769

-- Defining the sequence {a_n}
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24

-- Conditions
variable (a : ℕ → ℝ)
variable (n : ℕ)
variable (S : ℕ → ℝ)

-- Roots of f(x)
axiom a1 : a 1 = 3
axiom a2 : a 2 = 8

-- Arithmetic sequence property
axiom a_n_arithmetic : ∀ n : ℕ, a (n + 1) - a n = 2 * n + 5

-- Increasing sequence
axiom a_increasing : ∀ n : ℕ, a n < a (n + 1)

-- Define S_n
def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, 1 / a (k + 1)

-- Theorems we need to prove
theorem gen_formula : a n = n^2 + 2 * n :=
sorry

theorem Sn_bounded : S n < 3 / 4 :=
sorry

end gen_formula_Sn_bounded_l755_755769


namespace inclination_angle_range_l755_755899

-- Define the sine curve
def sine_curve (x : ℝ) : ℝ := Real.sin x

-- Define the derivative of the sine curve
def tangent_slope (x : ℝ) : ℝ := Real.cos x

theorem inclination_angle_range :
  ∀ (x : ℝ), (tangent_slope x = Real.cos x) →
  (tangent_slope x ∈ set.Icc (-1 : ℝ) 1) →
  (Real.arctan (tangent_slope x) ∈ set.union (set.Icc 0 (Real.pi / 4)) (set.Icc (3 * Real.pi / 4) Real.pi)) :=
by
  intro x h_cos hx_range
  -- Proof omitted
  sorry

end inclination_angle_range_l755_755899


namespace repeating_decimal_equiv_fraction_l755_755283

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755283


namespace three_numbers_sum_to_fifty_l755_755745

theorem three_numbers_sum_to_fifty : ∃ a b c ∈ ({21, 19, 30, 25, 3, 12, 9, 15, 6, 27} : set ℤ), a + b + c = 50 :=
by {
  use [19, 25, 6],
  simp,
  refl,
}

end three_numbers_sum_to_fifty_l755_755745


namespace chicken_price_reaches_81_in_2_years_l755_755853

theorem chicken_price_reaches_81_in_2_years :
  ∃ t : ℝ, (t / 12 = 2) ∧ (∃ n : ℕ, (3:ℝ)^(n / 6) = 81 ∧ n = t) :=
by
  sorry

end chicken_price_reaches_81_in_2_years_l755_755853


namespace greatest_common_multiple_of_9_and_15_less_than_120_l755_755998

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end greatest_common_multiple_of_9_and_15_less_than_120_l755_755998


namespace solve_equation_l755_755583

theorem solve_equation :
  ∃ y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ↔ y = 6 / 25 :=
by
  sorry

end solve_equation_l755_755583


namespace triangle_semicircle_s_leq_8r_squared_l755_755675

theorem triangle_semicircle_s_leq_8r_squared
  {A B C : Type} [has_dist A] [has_dist B] [has_real_dist B]
  (r : ℝ) (h_diameter : is_diameter (segment A B) (semicircle r))
  (hC_ne : C ≠ A ∧ C ≠ B) : 
  let AC := dist A C;
      BC := dist B C;
      s := AC + BC in
  s^2 ≤ 8 * r^2 :=
by
  sorry

end triangle_semicircle_s_leq_8r_squared_l755_755675


namespace max_intersection_points_2010_l755_755668

noncomputable def Sphere_2010 := {x : ℝ^2011 // ∥x∥ = r}

def max_intersection_points (S : set (Sphere_2010 r)) : ℕ := 
  max (set.univ : set (fin 2 -> Sphere_2010 r))

theorem max_intersection_points_2010 {S : set (Sphere_2010 r) } :
  max_intersection_points S = 2 := 
sorry

end max_intersection_points_2010_l755_755668


namespace riya_speed_is_20_l755_755030

variable (R : ℝ) -- Let R be Riya's speed in kmph.

axiom riya_speed_condition_1 : Priya_speed = 30 -- Priya's speed is 30 kmph.
axiom riya_speed_condition_2 : Time = 0.5 -- 30 minutes is 0.5 hours.
axiom riya_speed_condition_3 : Distance = 25 -- They are 25 km apart.

theorem riya_speed_is_20 :
  (R + 30) * 0.5 = 25 → R = 20 := by
  sorry

end riya_speed_is_20_l755_755030


namespace perpendiculars_concurrent_iff_eqn_holds_l755_755917

-- Definitions of triangles and points
variables {A B C U V W : Point}

-- Define a condition that checks if three points are concurrent
def are_concurrent (A B C : Point) : Prop := sorry

-- Theorem statement: Perpendiculars from points U, V, W are concurrent if and only if the given equation holds
theorem perpendiculars_concurrent_iff_eqn_holds (A B C U V W : Point) :
  (are_concurrent U V W) ↔ (dist_squared (A, W) + dist_squared (B, U) + dist_squared (C, V) = dist_squared (A, V) + dist_squared (C, U) + dist_squared (B, W)) :=
begin
  sorry
end

end perpendiculars_concurrent_iff_eqn_holds_l755_755917


namespace smallest_mu_inequality_l755_755713

theorem smallest_mu_inequality (μ : ℝ) :
  (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) ↔ μ = 6 :=
begin
  sorry
end

end smallest_mu_inequality_l755_755713


namespace repeating_decimal_as_fraction_l755_755308

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755308


namespace ratio_of_hypotenuse_segments_l755_755846

theorem ratio_of_hypotenuse_segments (x : ℝ) (h_pos : x > 0) : 
  let AB := 3 * x in let BC := 2 * x in
  let AC := (AB^2 + BC^2)^0.5 in
  let BD := (BC / AB) * (AC / ((BC / AB) + 1)) in 
  let CD := AC - BD in
  BD / CD = 9 / 4 :=
by
  sorry

end ratio_of_hypotenuse_segments_l755_755846


namespace line_a_minus_b_l755_755459

theorem line_a_minus_b (a b : ℝ)
  (h1 : (2 : ℝ) = a * (3 : ℝ) + b)
  (h2 : (26 : ℝ) = a * (7 : ℝ) + b) :
  a - b = 22 :=
by
  sorry

end line_a_minus_b_l755_755459


namespace conjugate_of_z_l755_755757

-- Define the given complex number 
def z : ℂ := 5 / (2 + complex.I)

-- Define the result we want to prove
def z_conjugate : ℂ := 2 + complex.I
 
-- State the theorem
theorem conjugate_of_z : complex.conj z = z_conjugate :=
by
  sorry

end conjugate_of_z_l755_755757


namespace fraction_of_repeating_decimal_l755_755332

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755332


namespace at_least_six_consecutive_heads_l755_755166

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end at_least_six_consecutive_heads_l755_755166


namespace probability_product_divisible_by_8_l755_755600

theorem probability_product_divisible_by_8 :
  (let prob := 4 / 8;
       event := prob ^ 4 + prob ^ 3 * (3 / 8) + prob ^ 2 * (3 / 8)^2 * 3 + prob * (3 / 8)^3
   in 1 - event = 51 / 64) := by sorry

end probability_product_divisible_by_8_l755_755600


namespace solution_of_r_and_s_l755_755001

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l755_755001


namespace Tom_initial_investment_l755_755086

noncomputable def Jose_investment : ℝ := 45000
noncomputable def Jose_investment_time : ℕ := 10
noncomputable def total_profit : ℝ := 36000
noncomputable def Jose_share : ℝ := 20000
noncomputable def Tom_share : ℝ := total_profit - Jose_share
noncomputable def Tom_investment_time : ℕ := 12
noncomputable def proportion_Tom : ℝ := (4 : ℝ) / 5
noncomputable def Tom_expected_investment : ℝ := 6000

theorem Tom_initial_investment (T : ℝ) (h1 : Jose_investment = 45000)
                               (h2 : Jose_investment_time = 10)
                               (h3 : total_profit = 36000)
                               (h4 : Jose_share = 20000)
                               (h5 : Tom_investment_time = 12)
                               (h6 : Tom_share = 16000)
                               (h7 : proportion_Tom = (4 : ℝ) / 5)
                               : T = Tom_expected_investment :=
by
  sorry

end Tom_initial_investment_l755_755086


namespace notebooks_if_students_halved_l755_755743

-- Definitions based on the problem conditions
def totalNotebooks: ℕ := 512
def notebooksPerStudent (students: ℕ) : ℕ := students / 8
def notebooksWhenStudentsHalved (students notebooks: ℕ) : ℕ := notebooks / (students / 2)

-- Theorem statement
theorem notebooks_if_students_halved (S : ℕ) (h : S * (S / 8) = totalNotebooks) :
    notebooksWhenStudentsHalved S totalNotebooks = 16 :=
by
  sorry

end notebooks_if_students_halved_l755_755743


namespace total_profit_calculation_l755_755676

-- Definitions based on conditions
def initial_investment_A := 5000
def initial_investment_B := 8000
def initial_investment_C := 9000
def initial_investment_D := 7000

def investment_A_after_4_months := initial_investment_A + 2000
def investment_B_after_4_months := initial_investment_B - 1000

def investment_C_after_6_months := initial_investment_C + 3000
def investment_D_after_6_months := initial_investment_D + 5000

def profit_A_percentage := 20
def profit_B_percentage := 30
def profit_C_percentage := 25
def profit_D_percentage := 25

def profit_C := 60000

-- Total profit is what we need to determine
def total_profit := 240000

-- The proof statement
theorem total_profit_calculation :
  total_profit = (profit_C * 100) / profit_C_percentage := 
by 
  sorry

end total_profit_calculation_l755_755676


namespace factory_toy_production_l755_755616

/-
Given the following conditions:
1. A factory produces 4560 toys per week.
2. The workers at this factory work 4 days a week.
3. The workers make the same number of toys every day.
Prove that the number of toys produced each day is 1140.
-/
theorem factory_toy_production (total_toys_per_week : ℕ) (days_per_week : ℕ) (h1 : total_toys_per_week = 4560) (h2 : days_per_week = 4) :
  total_toys_per_week / days_per_week = 1140 :=
by
  rw [h1, h2]
  norm_num
  sorry

end factory_toy_production_l755_755616


namespace least_positive_angle_l755_755963

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

def is_rel_prime (a b : ℕ) : Prop := gcd a b = 1

noncomputable def alpha := (9 : ℝ) / 4

def trig_equation (alpha : ℝ) : Prop :=
  let term x := (3 / 4) - (Real.sin x)^2
  term alpha * term (3 * alpha) * term (9 * alpha) * term (27 * alpha) = 1 / 256

theorem least_positive_angle : trig_equation alpha ∧ is_rel_prime 9 4 ∧ (9 + 4 = 13) :=
by
  sorry

end least_positive_angle_l755_755963


namespace product_of_f_eq_1_l755_755233

def S : set (ℤ × ℤ) := 
  { (x, y) | x ∈ { -2, -1, 0, 1, 2 } ∧ y ∈ { -3, -2, -1, 0, 1, 2, 3 } } \ { (0, 2) }

structure triangle :=
(A B C : ℤ × ℤ)

def is_right_triangle (t : triangle) : Prop :=
  let (ax, ay) := t.A in
  let (bx, by) := t.B in
  let (cx, cy) := t.C in
  (bx - ax) * (cx - ax) + (by - ay) * (cy - ay) = 0

def T : set triangle :=
  { t | t.A ∈ S ∧ t.B ∈ S ∧ t.C ∈ S ∧ is_right_triangle t }

def f (t : triangle) : ℚ :=
  let (bx, by) := t.B in
  let (cx, cy) := t.C in
  if by - cy = 0 then 0 else (by - cy) / (bx - cx)

theorem product_of_f_eq_1 : ∏ t in T, f t = 1 := sorry

end product_of_f_eq_1_l755_755233


namespace num_ways_to_order_l755_755446

theorem num_ways_to_order : 
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 10} in
  (∃ l : List ℕ, l.perm (List.range' 1 10) ∧
    ∀ (i : ℕ) (h1: i < l.length) (h2: i > 0),
          let b := (l.nth_le i h1 < l.nth_le (i - 1) (Nat.pred_lt h2 h1)),
              a := (l.nth_le i h1 > l.nth_le (i - 1) (Nat.pred_lt h2 h1))
          in b ∨ a) → 
  (2^9 = 512) :=
by
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 10}
  let l := List.range' 1 10
  let b := 2^9
  exact 512
  sorry

end num_ways_to_order_l755_755446


namespace kim_earrings_l755_755884

-- Define the number of pairs of earrings on the first day E as a variable
variable (E : ℕ)

-- Define the total number of gumballs Kim receives based on the earrings she brings each day
def total_gumballs_received (E : ℕ) : ℕ :=
  9 * E + 9 * 2 * E + 9 * (2 * E - 1)

-- Define the total number of gumballs Kim eats in 42 days
def total_gumballs_eaten : ℕ :=
  3 * 42

-- Define the statement to be proved
theorem kim_earrings : 
  total_gumballs_received E = total_gumballs_eaten + 9 → E = 3 :=
by sorry

end kim_earrings_l755_755884


namespace find_CD_length_l755_755631

noncomputable def triangle_angle_bisector_length (a b : ℝ) (ha : a > b) : ℝ :=
  have h_division : ∀ (A B C : Type) (h : a / b = (A - B) / C), _
    Proof sorry
  := let y := a * b / (a - b) in y

theorem find_CD_length (a b : ℝ) (ha : a > b) :
  ∃ (CD : ℝ), CD = triangle_angle_bisector_length a b ha :=
  by use a * b / (a - b) ; sorry

end find_CD_length_l755_755631


namespace largest_expression_value_l755_755124

theorem largest_expression_value : max (max (max (3 + 1 + 4 + 6) (3 * 1 + 4 + 6)) (max (3 + 1 * 4 + 6) (3 + 1 + 4 * 6))) (3 * 1 * 4 * 6) = 72 :=
by
  have exprA := 3 + 1 + 4 + 6
  have exprB := 3 * 1 + 4 + 6
  have exprC := 3 + 1 * 4 + 6
  have exprD := 3 + 1 + 4 * 6
  have exprE := 3 * 1 * 4 * 6

  have exprA_val : exprA = 14 := by norm_num
  have exprB_val : exprB = 13 := by norm_num
  have exprC_val : exprC = 13 := by norm_num
  have exprD_val : exprD = 28 := by norm_num
  have exprE_val : exprE = 72 := by norm_num

  -- Combine the maximum values step-by-step
  have maxAB := max exprA exprB
  have max_ABC := max maxAB exprC
  have max_ABCD := max max_ABC exprD
  have result := max max_ABCD exprE

  show result = 72
  by {
    simp [exprA_val, exprB_val, exprC_val, exprD_val, exprE_val],
    norm_num
  }

end largest_expression_value_l755_755124


namespace arithmetic_mean_discarded_numbers_l755_755946

theorem arithmetic_mean_discarded_numbers :
  ∀ (nums : Fin 60 → ℕ), (finset.sum (finset.univ : Finset (Fin 60)) nums) = 2700 →
  nums 0 = 60 → nums 1 = 75 →
  (arithmetic_mean (nums \ {60, 75})) = 465 / 106 := 
sorry

end arithmetic_mean_discarded_numbers_l755_755946


namespace probability_at_least_four_same_face_l755_755730

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l755_755730


namespace maggie_candy_collected_l755_755437

theorem maggie_candy_collected : ∃ M : ℝ, (M > 0) ∧ (1.82 * M = 91) ∧ (M = 50) :=
by
  use 50
  split
  sorry
  split
  sorry

end maggie_candy_collected_l755_755437


namespace original_price_each_book_l755_755224

theorem original_price_each_book (p : ℝ) : 
  let book1_discounted_price := p * 0.75,
      book2_discounted_price := p * 0.60,
      total_spent := book1_discounted_price + book2_discounted_price
  in total_spent = 66 → p = 48.89 :=
by
  intro h
  sorry

end original_price_each_book_l755_755224


namespace gen_formula_Sn_bounded_l755_755771

-- Defining the sequence {a_n}
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24

-- Conditions
variable (a : ℕ → ℝ)
variable (n : ℕ)
variable (S : ℕ → ℝ)

-- Roots of f(x)
axiom a1 : a 1 = 3
axiom a2 : a 2 = 8

-- Arithmetic sequence property
axiom a_n_arithmetic : ∀ n : ℕ, a (n + 1) - a n = 2 * n + 5

-- Increasing sequence
axiom a_increasing : ∀ n : ℕ, a n < a (n + 1)

-- Define S_n
def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, 1 / a (k + 1)

-- Theorems we need to prove
theorem gen_formula : a n = n^2 + 2 * n :=
sorry

theorem Sn_bounded : S n < 3 / 4 :=
sorry

end gen_formula_Sn_bounded_l755_755771


namespace repeating_decimal_eq_fraction_l755_755337

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755337


namespace new_average_after_multiplication_l755_755623

theorem new_average_after_multiplication
  (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 7)
  (h2 : a = 25)
  (h3 : m = 5):
  (n * a * m / n) = 125 :=
by
  sorry


end new_average_after_multiplication_l755_755623


namespace square_area_l755_755191

open Real

-- Define the given conditions
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line := 8
def side_points (x : ℝ) := parabola x = line

-- Problem statement: Find the area of the square
theorem square_area :
  let x1 : ℝ := 1 in
  let x2 : ℝ := -5 in
  let side_length := abs (x1 - x2) in
  side_length * side_length = 36 :=
by
  -- Proof is replaced with sorry to satisfy the problem constraints
  sorry

end square_area_l755_755191


namespace find_z_find_a_range_l755_755785

-- Define the given conditions
variables {z : ℂ} (a b : ℝ)
def condition1 := (z + complex.I * 2).im = 0
def condition2 := (z / (2 - complex.I)).im = 0

-- Solution to the first part
theorem find_z (h1 : condition1) (h2 : condition2) : z = 4 - 2 * complex.I :=
by sorry

-- Define the conditions for the second part
def condition3 (a : ℝ) (h3 : (z + a * complex.I)^2) := 
  (complex.re ((z + a * complex.I)^2) > 0) ∧ (complex.im ((z + a * complex.I)^2) < 0)

-- Solution to the second part
theorem find_a_range (h3 : condition3 a (find_z h1 h2)) : -2 < a ∧ a < 2 :=
by sorry

end find_z_find_a_range_l755_755785


namespace range_of_m_l755_755754

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x - 3| ≤ 2 → 1 ≤ x ∧ x ≤ 5) → 
  (∀ x : ℝ, (x - m + 1) * (x - m - 1) ≤ 0 → m - 1 ≤ x ∧ x ≤ m + 1) → 
  (∀ x : ℝ, x < 1 ∨ x > 5 → x < m - 1 ∨ x > m + 1) → 
  2 ≤ m ∧ m ≤ 4 := 
by
  sorry

end range_of_m_l755_755754


namespace find_central_angle_l755_755393

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end find_central_angle_l755_755393


namespace no_universal_tico_number_l755_755217

def is_tico (n : ℕ) : Prop :=
  (nat.digits 10 n).sum % 2003 = 0

theorem no_universal_tico_number : ¬ ∃ N : ℕ, ∀ k : ℕ, is_tico (k * N) :=
by sorry

end no_universal_tico_number_l755_755217


namespace express_in_scientific_notation_l755_755038

theorem express_in_scientific_notation (billion : ℝ) (x : ℝ) (h_billion : billion = 10^9) (h_x_range : 1 ≤ x ∧ x < 10) :
  (1.097 * billion) = (1.097 * 10^9) :=
by
  rw h_billion
  -- Since x is not used further, we could have used 1.097 directly in range
  refine (1.097 : ℝ) ∈ set.Icc 1 10
  norm_num

-- Proof is omitted using sorry

end express_in_scientific_notation_l755_755038


namespace recurring_decimal_to_fraction_l755_755314

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755314


namespace intervals_of_monotonicity_range_of_a_l755_755806

noncomputable def f (x a : ℝ) := (2 - a) * (x - 1) - 2 * Real.log x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → (deriv (λ x, (2 - (1:ℝ)) * (x - 1) - 2 * Real.log x) x) < 0) ∧
  (∀ x : ℝ, x > 2 → (deriv (λ x, (2 - (1:ℝ)) * (x - 1) - 2 * Real.log x) x) > 0) :=
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (0 < x ∧ x < (1 / 3)) → ((2 - a) * (x - 1) - 2 * Real.log x) ≠ 0) :
  a ∈ Set.Ici (2 - 3 * Real.log 3) :=
  sorry

end intervals_of_monotonicity_range_of_a_l755_755806


namespace claire_initial_balloons_l755_755223

theorem claire_initial_balloons (B : ℕ) (h : B - 12 - 9 + 11 = 39) : B = 49 :=
by sorry

end claire_initial_balloons_l755_755223


namespace chris_birthday_after_45_days_l755_755698

theorem chris_birthday_after_45_days (k : ℕ) (h : k = 45) (tuesday : ℕ) (h_tuesday : tuesday = 2) : 
  (tuesday + k) % 7 = 5 := 
sorry

end chris_birthday_after_45_days_l755_755698


namespace sum_f_eq_38889_l755_755514

def f (n : ℕ) : ℕ :=
  n.digits 10 |> nat.digits.zero_count

def S : finset ℕ := finset.Icc 1 99999

theorem sum_f_eq_38889 : 
  ∑ n in S, f n = 38889 :=
by
  sorry

end sum_f_eq_38889_l755_755514


namespace problem_solution_l755_755865

variables {φ θ ρ : ℝ}
variables {x y : ℝ}
noncomputable theory

-- Conditions
def parametric_circle (φ : ℝ) : Prop := 
  x = 1 + cos φ ∧ y = sin φ

def polar_line (ρ θ : ℝ) : Prop :=
  2 * ρ * sin (θ + π / 3) = 6 * sqrt 3 

def ray_OM (θ : ℝ) : Prop :=
  θ = π / 6

-- Proof for the questions
theorem problem_solution :
  (∀ φ, parametric_circle φ) →
  (∃ x y : ℝ, (x - 1)^2 + y^2 = 1) ∧
  (∀ θ, θ ≠ π / 6 → ∃ ρ, ρ = 2 * cos θ) ∧
  (∃ θ, ray_OM θ ∧ ρ = sqrt 3 ∧ polar_line 3 sqrt 3 θ) ∧
  (∃ θ, ray_OM θ ∧ polar_line ρ θ → abs(sqrt 3 - 3 * sqrt 3) = 2 * sqrt 3) :=
begin
  sorry
end

end problem_solution_l755_755865


namespace miles_run_l755_755021

theorem miles_run (peter_additional: ℕ) (andrew_miles: ℕ) (days: ℕ) : 
  peter_additional = 3 ∧ andrew_miles = 2 ∧ days = 5 → 
  (days * (peter_additional + andrew_miles) + days * andrew_miles = 35) :=
by
  assume h,
  sorry

end miles_run_l755_755021


namespace sum_first_3n_terms_l755_755390

variable (S : ℕ → ℝ)  -- S(n) represents the sum of the first n terms of a geometric sequence

-- Given conditions
axiom Sum_first_n_terms : S n = 48
axiom Sum_first_2n_terms : S (2 * n) = 60

-- Statement to prove
theorem sum_first_3n_terms (n : ℕ) : S (3 * n) = 63 := by
  have h1 : S (2 * n) - S n = 12 := by
    rw [Sum_first_2n_terms, Sum_first_n_terms]
    norm_num
  
  have h2 : S n ≠ 0 := by
    rw [Sum_first_n_terms]
    norm_num
    
  -- Placeholder for the argument about geometric sequences and the second n terms.
  have h3 : S (3 * n) = S (2 * n) + (S (2 * n) - S n) ^ 2 / S n := by
    sorry
  
  -- Combine the results:
  rw [h1, Sum_first_2n_terms, Sum_first_n_terms] at h3
  norm_num at h3
  exact h3

end sum_first_3n_terms_l755_755390


namespace quadratic_func_condition_l755_755844

noncomputable def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_func_condition (b c : ℝ) (h : f (-3) b c = f 1 b c) :
  f 1 b c > c ∧ c > f (-1) b c :=
by
  sorry

end quadratic_func_condition_l755_755844


namespace work_completion_time_l755_755130

theorem work_completion_time (x : ℝ) (a_work_rate b_work_rate combined_work_rate : ℝ) :
  a_work_rate = 1 / 15 ∧
  b_work_rate = 1 / 20 ∧
  combined_work_rate = 1 / 7.2 ∧
  a_work_rate + b_work_rate + (1 / x) = combined_work_rate → 
  x = 45 := by
  sorry

end work_completion_time_l755_755130


namespace smallest_n_l755_755897

theorem smallest_n 
  (n : ℕ) 
  (x : ℕ → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_sum : ∑ i in finset.range n, x i = 1)
  (h_sum_squares : ∑ i in finset.range n, (x i)^2 ≤ 1 / 64) : 
  n ≥ 64 :=
sorry

end smallest_n_l755_755897


namespace average_net_sales_per_month_l755_755039

def sales_jan : ℕ := 120
def sales_feb : ℕ := 80
def sales_mar : ℕ := 50
def sales_apr : ℕ := 130
def sales_may : ℕ := 90
def sales_jun : ℕ := 160

def monthly_expense : ℕ := 30
def num_months : ℕ := 6

def total_sales := sales_jan + sales_feb + sales_mar + sales_apr + sales_may + sales_jun
def total_expenses := monthly_expense * num_months
def net_total_sales := total_sales - total_expenses

theorem average_net_sales_per_month : net_total_sales / num_months = 75 :=
by {
  -- Lean code for proof here
  sorry
}

end average_net_sales_per_month_l755_755039


namespace number_of_green_fish_l755_755904

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end number_of_green_fish_l755_755904


namespace solution_of_r_and_s_l755_755000

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l755_755000


namespace fraction_eq_repeating_decimal_l755_755269

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755269


namespace largest_integer_solution_l755_755053

theorem largest_integer_solution (x : ℤ) (h : (3 * x - 4) / 2 < x - 1) : x ≤ 1 :=
begin
  sorry
end

end largest_integer_solution_l755_755053


namespace largest_value_l755_755125

theorem largest_value :
  let A := 1/2
  let B := 1/3 + 1/4
  let C := 1/4 + 1/5 + 1/6
  let D := 1/5 + 1/6 + 1/7 + 1/8
  let E := 1/6 + 1/7 + 1/8 + 1/9 + 1/10
  E > A ∧ E > B ∧ E > C ∧ E > D := by
sorry

end largest_value_l755_755125


namespace repeating_decimal_eq_fraction_l755_755344

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755344


namespace funnel_height_l755_755159

noncomputable def radius : ℝ := 4
noncomputable def volume : ℝ := 150
noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

theorem funnel_height :
  ∃ h : ℝ, cone_volume radius h = volume ∧ h ≈ 9 :=
by
  sorry

end funnel_height_l755_755159


namespace base_of_right_angled_triangle_l755_755183

theorem base_of_right_angled_triangle 
  (height : ℕ) (area : ℕ) (hypotenuse : ℕ) (b : ℕ) 
  (h_height : height = 8)
  (h_area : area = 24)
  (h_hypotenuse : hypotenuse = 10) 
  (h_area_eq : area = (1 / 2 : ℕ) * b * height)
  (h_pythagorean : hypotenuse^2 = height^2 + b^2) : 
  b = 6 := 
sorry

end base_of_right_angled_triangle_l755_755183


namespace projection_theorem_l755_755862

-- Definitions of the geometric objects and conditions
variables (A B C D O : Type) 

-- Assuming basic properties of perpendicularity and projections
variable [plane ABC : Prop] 
variable [plane ABD : Prop] 
variable [plane BCD : Prop]

-- Assuming A, B, C, and D are points in the respective planes
variable [in_plane A ABD]
variable [in_plane B ABD]
variable [in_plane C ABD]
variable [in_plane D ABD]
variable [in_plane A BCD]
variable [in_plane B BCD]
variable [in_plane C BCD]
variable [in_plane D BCD]
variable [in_plane O BCD]

-- Geometric assumptions
variable [perpendicular CA ABD: Prop]
variable [projection A O BCD : Prop]

-- Areas of triangles involved
variables (S_ABC S_BOC S_BDC : ℝ)

-- Theorem statement in Lean
theorem projection_theorem 
  (h1 : CA ⊥ plane ABD) 
  (h2 : O is projection of A within plane BCD) :
  (S_ABC ^ 2 = S_BOC * S_BDC) :=
sorry -- proof is not required

end projection_theorem_l755_755862


namespace payment_amount_l755_755647

/-- 
A certain debt will be paid in 52 installments from January 1 to December 31 of a certain year.
Each of the first 25 payments is to be a certain amount; each of the remaining payments is to be $100 more than each of the first payments.
The average (arithmetic mean) payment that will be made on the debt for the year is $551.9230769230769.
Prove that the amount of each of the first 25 payments is $500.
-/
theorem payment_amount (X : ℝ) 
  (h1 : 25 * X + 27 * (X + 100) = 52 * 551.9230769230769) :
  X = 500 :=
sorry

end payment_amount_l755_755647


namespace range_of_a_l755_755807

noncomputable theory
open Real

def f (a x : ℝ) : ℝ := (2 * x - 1) * exp x - a * (x^2 + x)
def g (a x : ℝ) : ℝ := -a * x^2 - a

theorem range_of_a (a : ℝ) :
    (∀ x, f a x ≥ g a x) → (1 ≤ a ∧ a ≤ 4 * exp (3 / 2)) :=
by 
  sorry

end range_of_a_l755_755807


namespace final_inequality_l755_755401

noncomputable def problem (n : ℕ) (a : ℕ → ℝ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) →
  (∑ i in finset.range n, a i = 1) →
  (∏ i in finset.range n, (a i + (1 / a i)) ≥ (n + (1 / n))^n)

theorem final_inequality (n : ℕ) (a : ℕ → ℝ) : problem n a :=
begin
  intros,
  sorry,
end

end final_inequality_l755_755401


namespace balls_in_boxes_l755_755828

theorem balls_in_boxes : 
  ∃ (f : fin 2 → finset (fin 6)), 
    (∀ b : fin 2, 2 ≤ (f b).card) ∧
    (disjoint (f 0) (f 1)) ∧
    (2 + 2 ≤ 6) ∧
    (f 0 ∪ f 1 = {x | x < 6}) ∧
    (nat.card {f : fin 2 → finset (fin 6) // 
      (∀ b : fin 2, 2 ≤ (f b).card) ∧
      (disjoint (f 0) (f 1)) ∧
      (f 0 ∪ f 1 = {x | x < 6}) } = 50) := sorry

end balls_in_boxes_l755_755828


namespace recurring_decimal_to_fraction_l755_755320

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755320


namespace smallest_b_factors_l755_755376

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end smallest_b_factors_l755_755376


namespace no_perfect_square_factorials_l755_755122

theorem no_perfect_square_factorials :
  ¬ (is_square (100! * 101!) ∨
     is_square (100! * 102!) ∨
     is_square (101! * 102!) ∨
     is_square (101! * 103!) ∨
     is_square (102! * 103!)) :=
by sorry

end no_perfect_square_factorials_l755_755122


namespace vector_addition_l755_755435

-- Given vectors a and b
def a : ℝ × ℝ × ℝ := (3, -2, 1)
def b : ℝ × ℝ × ℝ := (-2, 4, 0)

-- The problem statement is to prove this vector equation
theorem vector_addition : a + (2 : ℝ) • b = (-1, 6, 1 : ℝ × ℝ × ℝ) :=
by
  -- This is where we would perform the proof.
  sorry

end vector_addition_l755_755435


namespace outer_edge_measurement_l755_755078

noncomputable def area_per_plant : ℝ := 4

noncomputable def number_of_plants : ℝ := 22.997889276778874

noncomputable def total_area_for_plants : ℝ := number_of_plants * area_per_plant

noncomputable def radius_of_bed : ℝ := real.sqrt (total_area_for_plants / real.pi)

noncomputable def circumference_of_bed : ℝ := 2 * real.pi * radius_of_bed

theorem outer_edge_measurement :
  abs (circumference_of_bed - 34.007194) < 0.000001 :=
begin
  sorry
end

end outer_edge_measurement_l755_755078


namespace perfect_square_iff_kth_power_iff_l755_755031

theorem perfect_square_iff (n : ℕ) :
  (∃ a : ℕ, n = a * a) ↔ ∀ p ∣ n, ∃ m : ℕ, nat.prime p ∧ n.factorization p = 2 * m := 
sorry

theorem kth_power_iff (n k : ℕ) :
  (∃ a : ℕ, n = a ^ k) ↔ ∀ p ∣ n, ∃ m : ℕ, nat.prime p ∧ n.factorization p = k * m := 
sorry

end perfect_square_iff_kth_power_iff_l755_755031


namespace min_value_integral_abs_diff_l755_755372

noncomputable def integral_abs_diff (a : ℝ) : ℝ :=
  ∫ x in 0..1, |real.exp (-x) - a|

theorem min_value_integral_abs_diff : (∀ a : ℝ, integral_abs_diff a) = 1 - 2 * real.exp (-1) :=
by
  sorry

end min_value_integral_abs_diff_l755_755372


namespace inequality_for_positive_reals_l755_755023

theorem inequality_for_positive_reals (k : ℝ) (n : ℕ) (a : Fin n → ℝ) 
  (hk : 0 < k ∧ k ≤ 1) (ha : ∀ i, 0 < (a i)) :
  (Finset.univ.sum (λ i, (a i / (Finset.univ.sum (λ j, if i = j then 0 else a j)))^k)) 
  ≥ n / (n - 1)^k :=
sorry

end inequality_for_positive_reals_l755_755023


namespace jordan_annual_income_l755_755975

theorem jordan_annual_income (q : ℝ) (I T : ℝ) 
  (h1 : T = q * 35000 + (q + 3) * (I - 35000))
  (h2 : T = (q + 0.4) * I) : 
  I = 40000 :=
by sorry

end jordan_annual_income_l755_755975


namespace trig_identity_solution_l755_755127

theorem trig_identity_solution (x : ℝ) (k : ℤ) :
  (1/2 * sin (4 * x) * sin x + sin (2 * x) * sin x = 2 * cos x ^ 2) →
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) := sorry

end trig_identity_solution_l755_755127


namespace total_votes_in_election_l755_755860

/-- 
In an election, candidate A got 85% of the total valid votes. 
15% of the total votes were declared invalid. 
The number of valid votes polled in favor of candidate A is 404600. 
We need to prove that the total number of votes in the election is 560000.
-/
theorem total_votes_in_election : 
  ∃ (V : ℝ), 
    0.15 * V + 0.85 * V * 0.85 = 404600 ∧ V = 560000 :=
begin
  use 560000,
  split,
  { sorry },
  { refl }
end

end total_votes_in_election_l755_755860


namespace james_weight_gain_l755_755497

def cheezits_calories (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) : ℕ :=
  bags * oz_per_bag * cal_per_oz

def chocolate_calories (bars : ℕ) (cal_per_bar : ℕ) : ℕ :=
  bars * cal_per_bar

def popcorn_calories (bags : ℕ) (cal_per_bag : ℕ) : ℕ :=
  bags * cal_per_bag

def run_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def swim_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def cycle_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def total_calories_consumed : ℕ :=
  cheezits_calories 3 2 150 + chocolate_calories 2 250 + popcorn_calories 1 500

def total_calories_burned : ℕ :=
  run_calories 40 12 + swim_calories 30 15 + cycle_calories 20 10

def excess_calories : ℕ :=
  total_calories_consumed - total_calories_burned

def weight_gain (excess_cal : ℕ) (cal_per_lb : ℕ) : ℚ :=
  excess_cal / cal_per_lb

theorem james_weight_gain :
  weight_gain excess_calories 3500 = 770 / 3500 :=
sorry

end james_weight_gain_l755_755497


namespace sin_value_of_angle_between_line_and_plane_l755_755787

def normal_vector : ℝ × ℝ × ℝ := (4, 1, 1)
def direction_vector : ℝ × ℝ × ℝ := (-2, -3, 3)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

noncomputable def sin_theta : ℝ :=
  abs (dot_product normal_vector direction_vector) / 
  (magnitude normal_vector * magnitude direction_vector)

theorem sin_value_of_angle_between_line_and_plane :
  sin_theta = 4 * Real.sqrt 11 / 33 :=
sorry

end sin_value_of_angle_between_line_and_plane_l755_755787


namespace pablo_total_blocks_l755_755911

/-- Pablo made 4 stacks of toy blocks where:
   - The first stack is 5 blocks tall.
   - The second stack is 2 blocks taller than the first.
   - The third stack is 5 blocks shorter than the second stack.
   - The last stack is 5 blocks taller than the third stack.

   We want to prove that the total number of toy blocks Pablo used is 21.
-/
theorem pablo_total_blocks : 
  let stack1 := 5
  let stack2 := stack1 + 2
  let stack3 := stack2 - 5
  let stack4 := stack3 + 5
  stack1 + stack2 + stack3 + stack4 = 21 :=
by
  let stack1 := 5
  let stack2 := stack1 + 2
  let stack3 := stack2 - 5
  let stack4 := stack3 + 5
  show stack1 + stack2 + stack3 + stack4 = 21 from sorry

end pablo_total_blocks_l755_755911


namespace frank_has_4_five_dollar_bills_l755_755741

theorem frank_has_4_five_dollar_bills
    (one_dollar_bills : ℕ := 7)
    (ten_dollar_bills : ℕ := 2)
    (twenty_dollar_bills : ℕ := 1)
    (change : ℕ := 4)
    (peanut_cost_per_pound : ℕ := 3)
    (days_in_week : ℕ := 7)
    (peanuts_per_day : ℕ := 3) :
    let initial_amount := (one_dollar_bills * 1) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)
    let total_peanuts_cost := (peanuts_per_day * days_in_week) * peanut_cost_per_pound
    let F := (total_peanuts_cost + change - initial_amount) / 5 
    F = 4 :=
by
  repeat { admit }


end frank_has_4_five_dollar_bills_l755_755741


namespace repeating_fraction_equality_l755_755354

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755354


namespace problem_statement_l755_755767

noncomputable def a (n : ℕ) : ℕ := n^2 + 2 * n
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24
def S (n : ℕ) : ℝ := (Finset.sum (Finset.range n) (λ k, 1 / (a (k + 1) : ℝ)))

theorem problem_statement 
  (h1 : ∃ r1 r2 : ℝ, f r1 = 0 ∧ f r2 = 0 ∧ r1 < r2 ∧ a 1 = r1 ∧ a 2 = r2)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n + 3) 
  : (∀ n : ℕ, a n = n^2 + 2 * n) ∧ (∀ n : ℕ, S n < 3 / 4) :=
by
  sorry

end problem_statement_l755_755767


namespace inequality_inequality_l755_755626

variable {n : ℕ}
variable {a : Fin n → ℝ} (s : ℝ)

theorem inequality_inequality (h1 : ∀ i, a i > 0) (h2 : n ≥ 2) (h3 : s = ∑ i, a i) :
  (∑ i, a i / (s - a i)) ≥ n / (n - 1) :=
sorry

end inequality_inequality_l755_755626


namespace expected_inflation_rate_l755_755068

/-- Defining the parameters given in the conditions --/
variables (i : ℝ) -- inflation rate
variables (p_sugar1994 p_sugar1996 : ℝ)
def rate_of_increase := (i + 0.03)

/-- The given conditions --/
def p_sugar1994 := 25 -- Price of sugar on January 1, 1994
def p_sugar1996 := 33.0625 -- Expected price of sugar on January 1, 1996

/-- Proof problem statement --/
theorem expected_inflation_rate (h: 25 * (1 + i + 0.03) * (1 + i + 0.03) = 33.0625) : i = 0.12 :=
sorry -- proof to be filled in


end expected_inflation_rate_l755_755068


namespace max_circle_radius_l755_755701

theorem max_circle_radius (AB BC CD DA : ℝ) (hAB : AB = 14) (hBC : BC = 9) (hCD : CD = 7) (hDA : DA = 12) :
  ∃ r, r = 2 * Real.sqrt 6 ∧ (exists_inside_or_on_boundary_circle AB BC CD DA r) :=
sorry

end max_circle_radius_l755_755701


namespace find_C_min_value_2a_b_l755_755493

-- Given the conditions in the problem.
variables {A B C : ℝ} {a b c : ℝ}

-- Assumptions
axiom triangle_sides {A B C : ℝ} (a b c : ℝ) 
  (h1 : angle_opposite A a)
  (h2 : angle_opposite B b)
  (h3 : angle_opposite C c)

-- Condition (b + 2a)cos(C) + ccos(B) = 0
axiom condition_1 : (b + 2 * a) * Real.cos(C) + c * Real.cos(B) = 0

-- Given condition, prove that C = 2π/3
theorem find_C (h1 : (b + 2 * a) * Real.cos C + c * Real.cos B = 0) : 
  C = 2 * Real.pi / 3 :=
sorry

-- Given C = 2π/3, the angle bisector of angle C intersects AB at D and CD = 2,
-- prove the minimum value of 2a + b is 6 + 4√2
theorem min_value_2a_b (h1 : C = 2 * Real.pi / 3) (h2 : CD = 2) : 
  ∃ (a b : ℝ), 2 * a + b = 6 + 4 * Real.sqrt 2 :=
sorry

end find_C_min_value_2a_b_l755_755493


namespace probability_at_least_four_same_face_l755_755735

theorem probability_at_least_four_same_face :
  let total_outcomes := (2 : ℕ) ^ 5,
      favorable_outcomes := 1 + 1 + (Nat.choose 5 1) + (Nat.choose 5 1),
      probability := favorable_outcomes / total_outcomes in
  probability = (3 : ℚ) / 8 :=
by
  sorry

end probability_at_least_four_same_face_l755_755735


namespace surface_area_of_reassembled_solid_l755_755181

noncomputable def total_surface_area : ℕ :=
let height_E := 1/4
let height_F := 1/6
let height_G := 1/9 
let height_H := 1 - (height_E + height_F + height_G)
let face_area := 2 * 1
(face_area * 2)     -- Top and bottom surfaces
+ 2                -- Side surfaces (1 foot each side * 2 sides)
+ (face_area * 2)   -- Front and back surfaces 

theorem surface_area_of_reassembled_solid :
  total_surface_area = 10 :=
by
  sorry

end surface_area_of_reassembled_solid_l755_755181


namespace restore_triangle_ABC_l755_755490

-- Definitions for the points and their properties
variables {A L M N : Point}
variable (triangle_ABC : Triangle A B C)

-- Conditions
def bisector_CL := is_angle_bisector (angle A C B) CL
def incircle_CAL_touches_AB_at_M := incircle_touches (Triangle A C L) AB M
def incircle_CBL_touches_AB_at_N := incircle_touches (Triangle C B L) AB N

-- Given the above conditions, state the problem in Lean 4
theorem restore_triangle_ABC (h1 : bisector_CL triangle_ABC CL) 
                             (h2 : incircle_CAL_touches_AB_at_M (Triangle A C L) M)
                             (h3 : incircle_CBL_touches_AB_at_N (Triangle C B L) N) :
  ∃ (C B : Point), restore_triangle A B C L M N :=
sorry

end restore_triangle_ABC_l755_755490


namespace avg_first_3_is_6_l755_755950

theorem avg_first_3_is_6 (A B C D : ℝ) (X : ℝ)
  (h1 : (A + B + C) / 3 = X)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11)
  (h4 : D = 4) :
  X = 6 := 
by
  sorry

end avg_first_3_is_6_l755_755950


namespace find_k_l755_755434

theorem find_k
  (a b : ℝ × ℝ)
  (ha : a = (1,2))
  (hb : b = (-3,2)) :
  ∃ k : ℝ, (let vector1 := (k * a.1 + b.1, k * a.2 + b.2),
                 vector2 := (a.1 - 3 * b.1, a.2 - 3 * b.2) in
             vector1.1 * vector2.1 + vector1.2 * vector2.2 = 0) ∧ k = 19 :=
by
  sorry

end find_k_l755_755434


namespace tangent_line_equation_l755_755783

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then log (-x) + 2 * x else log x - 2 * x

theorem tangent_line_equation (x := 1) (y := -2) :
  (∀ x, f (-x) = f x) ∧ (∀ x, x < 0 → f x = log (-x) + 2 * x) →
  (∃ m b, y = m * x + b ∧ (∀ (a b : ℝ), a * x + y + b = 0)) ↔
  (x + y + 1 = 0) :=
by {
  sorry
}

end tangent_line_equation_l755_755783


namespace tiles_count_l755_755662

variable (c r : ℕ)

-- given: r = 10
def initial_rows_eq : Prop := r = 10

-- assertion: number of tiles is conserved after rearrangement
def tiles_conserved : Prop := c * r = (c - 2) * (r + 4)

-- desired: total number of tiles is 70
def total_tiles : Prop := c * r = 70

theorem tiles_count (h1 : initial_rows_eq r) (h2 : tiles_conserved c r) : total_tiles c r :=
by
  subst h1
  sorry

end tiles_count_l755_755662


namespace general_formula_for_an_sum_sn_less_than_l755_755773

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24
def a₁ : ℝ := 3
def a₂ : ℝ := 8
def diff_seq (n : ℕ) : ℝ := 2 * n + 3
def a (n : ℕ) : ℝ := (n : ℝ)^2 + 2 * (n : ℝ)
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, 1 / a k)

-- We are to prove these statements
theorem general_formula_for_an : ∀ (n : ℕ), a n = n^2 + 2 * n := by
  sorry

theorem sum_sn_less_than : ∀ (n : ℕ), S n < 3 / 4 := by
  sorry

end general_formula_for_an_sum_sn_less_than_l755_755773


namespace geometric_sequence_of_a_maximum_b_l755_755396

variable (a : ℕ → ℝ)

-- The condition that the sum of the first n terms equals n minus the nth term.
def S (n : ℕ) : Prop :=
  ∑ i in Finset.range n, a (i + 1) = (n : ℝ) - a n

theorem geometric_sequence_of_a
  (hS : ∀ n, S n)
  (n : ℕ) :
  ∃ r, ∀ n, a n - 1 = r ^ n :=
by
  sorry

def b (n : ℕ) : ℝ :=
  (2 - n) * (a n - 1)

theorem maximum_b (hS : ∀ n, S n) :
  ∃ n, b n = (1 / 8) :=
by
  sorry

end geometric_sequence_of_a_maximum_b_l755_755396


namespace length_of_second_train_is_correct_l755_755091

noncomputable def length_of_first_train : ℝ := 120
noncomputable def speed_of_first_train : ℝ := 42 -- in kmph
noncomputable def speed_of_second_train : ℝ := 30 -- in kmph
noncomputable def time_to_clear_each_other : ℝ := 19.99840012798976 -- in seconds

theorem length_of_second_train_is_correct : 
    let rel_speed := (speed_of_first_train + speed_of_second_train) * (1000 / 3600),
        combined_length := rel_speed * time_to_clear_each_other
    in combined_length - length_of_first_train = 279.9680025597952 :=
by
  let rel_speed := (speed_of_first_train + speed_of_second_train) * (1000 / 3600)
  let combined_length := rel_speed * time_to_clear_each_other
  have h: combined_length - length_of_first_train = 279.9680025597952
  sorry

end length_of_second_train_is_correct_l755_755091


namespace math_marks_l755_755706

theorem math_marks (english physics chemistry biology total_marks math_marks : ℕ) 
  (h_eng : english = 73)
  (h_phy : physics = 92)
  (h_chem : chemistry = 64)
  (h_bio : biology = 82)
  (h_avg : total_marks = 76 * 5) :
  math_marks = 69 := 
by
  sorry

end math_marks_l755_755706


namespace problem_I_problem_II_l755_755809

open Real

def f (x a : ℝ) := log x + a / x

def h (x a : ℝ) := x * log x + a

def phi (x : ℝ) := x * exp (-x)

theorem problem_I (a : ℝ) (h₀ : 0 < a) : 
  (∃ x : ℝ, 0 < x ∧ f x a = 0) ↔ (0 < a ∧ a ≤ 1 / exp 1) := 
  sorry

theorem problem_II (a b : ℝ) (h₀ : a ≥ 2 / exp 1) (h₁ : b > 1) : 
  f (log b) a > 1 / b := 
  sorry

end problem_I_problem_II_l755_755809


namespace angle_is_10_l755_755798

theorem angle_is_10 (x : ℕ) (h1 : 180 - x = 2 * (90 - x) + 10) : x = 10 := 
by sorry

end angle_is_10_l755_755798


namespace gift_boxes_wrapped_l755_755126

/-- Xiao Ming's mother uses a 25-meter long ribbon to wrap gift boxes.
    Each gift box requires a 1.6-meter long ribbon. Prove that the ribbons can wrap 15 gift boxes. -/
theorem gift_boxes_wrapped (total_ribbon_length : ℝ) (ribbon_per_box : ℝ)
    (h1 : total_ribbon_length = 25) (h2 : ribbon_per_box = 1.6) : (total_ribbon_length / ribbon_per_box).toInt = 15 :=
by
  rw [h1, h2]
  have h : 25 / 1.6 = 15.625 := by sorry  -- Incorporate precise division if necessary
  norm_cast
  exact Int.ofNat 15 -- Casting to integer


end gift_boxes_wrapped_l755_755126


namespace part1_part2_part3_l755_755427

-- Define the function f(x)
def f (x : ℝ) := (Real.sin x)^2 - (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Prove that f(2π/3) = 2
theorem part1 : f (2 * Real.pi / 3) = 2 :=
by sorry

-- Prove that the smallest positive period of f(x) is π
theorem part2 : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by sorry

-- Define the intervals of monotonic increase for f(x)
def monotonic_intervals (k : ℤ) : Set ℝ := 
  {x | (k : ℝ) * Real.pi - (5 * Real.pi / 6) ≤ x ∧ x ≤ (k : ℝ) * Real.pi - (Real.pi / 3)}

-- Prove the intervals of monotonic increase for f(x)
theorem part3 : ∀ k : ℤ, ∀ x ∈ monotonic_intervals k, 
  ∃ I, (I = (k : ℝ) * Real.pi - (5 * Real.pi / 6), (k : ℝ) * Real.pi - (Real.pi / 3)) ∧ (∀ y ∈ I, f' y > 0) :=
by sorry

end part1_part2_part3_l755_755427


namespace fraction_for_repeating_56_l755_755259

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755259


namespace interest_for_1_rs_l755_755736

theorem interest_for_1_rs (I₅₀₀₀ : ℝ) (P : ℝ) (h : I₅₀₀₀ = 200) (hP : P = 5000) : I₅₀₀₀ / P = 0.04 :=
by
  rw [h, hP]
  norm_num

end interest_for_1_rs_l755_755736


namespace problem_solution_l755_755574

noncomputable def proof_problem : Prop :=
∀ x y : ℝ, y = (x + 1)^2 ∧ (x * y^2 + y = 1) → false

theorem problem_solution : proof_problem :=
by
  sorry

end problem_solution_l755_755574


namespace diff_of_squares_l755_755105

theorem diff_of_squares : (1001^2 - 999^2 = 4000) :=
by
  sorry

end diff_of_squares_l755_755105


namespace repeating_decimal_as_fraction_l755_755310

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755310


namespace addition_and_rounding_example_l755_755678

def add_and_round (x y : ℝ) : ℝ :=
  let sum := x + y
  let rounded := (Real.floor (sum * 10) / 10 : ℝ)
  rounded

theorem addition_and_rounding_example :
  add_and_round 76.893 34.2176 = 111.1 :=
by
  sorry

end addition_and_rounding_example_l755_755678


namespace position_after_90_moves_l755_755175

noncomputable def ω : ℂ := complex.exp (complex.I * (π / 3))

def move (z: ℂ) : ℂ := ω * z + 12

def particle_moves (n : ℕ) : ℂ :=
  nat.rec_on n 5 (λ n z, move z)

theorem position_after_90_moves : particle_moves 90 = 5 := 
sorry

end position_after_90_moves_l755_755175


namespace regular_price_of_mani_pedi_l755_755054

theorem regular_price_of_mani_pedi 
  (discount : ℝ) -- 0.25 (25%)
  (num_mani_pedis : ℕ) -- 5
  (total_discounted_price : ℝ) -- 150
  (P : ℝ) -- regular price of mani/pedi
  (discounted_price_per_mani_pedi : ℝ) -- 0.75 * P
  (individual_discounted_price : ℝ) -- total_discounted_price / num_mani_pedis
  (regular_price_proof : 0.75 * P = individual_discounted_price) :
  P = 40 := 
by
  let discounted_price := 1 - discount
  have individual_discounted_price_eq : individual_discounted_price = total_discounted_price / num_mani_pedis := sorry
  rw [individual_discounted_price_eq, regular_price_proof]
  have P_eq : P = individual_discounted_price / discounted_price := sorry
  rw [individual_discounted_price_eq]
  have final_price : P = 150 / 5 / 0.75 := by sorry
  exact final_price

end regular_price_of_mani_pedi_l755_755054


namespace gcm_of_9_and_15_less_than_120_eq_90_l755_755996

theorem gcm_of_9_and_15_less_than_120_eq_90 
  (lcm_9_15 : Nat := Nat.lcm 9 15)
  (multiples : List Nat := List.range (120 / lcm_9_15) |> List.map (λ n => n * lcm_9_15)) : 
  lcm_9_15 = 45 ∧ multiples.max = some 90 := by
sorry

end gcm_of_9_and_15_less_than_120_eq_90_l755_755996


namespace solve_for_y_l755_755928

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l755_755928


namespace problem_xy_pairs_l755_755896

theorem problem_xy_pairs (N : ℕ) :
  (∃ (pairs : Set (ℤ × ℤ)), (∀ p ∈ pairs, let ⟨x, y⟩ := p in x^2 + x*y + y^2 ≤ 2007) ∧ pairs.card = N) →
  (Odd N ∧ ¬ (N % 3 = 0)) :=
by
  sorry

end problem_xy_pairs_l755_755896


namespace pow_mod_3_225_l755_755100

theorem pow_mod_3_225 :
  (3 ^ 225) % 11 = 1 :=
by
  -- Given condition from problem:
  have h : 3 ^ 5 % 11 = 1 := by norm_num
  -- Proceed to prove based on this condition
  sorry

end pow_mod_3_225_l755_755100


namespace smallest_prime_after_five_consecutive_nonprimes_l755_755121

theorem smallest_prime_after_five_consecutive_nonprimes :
  ∃ p : ℕ, Nat.Prime p ∧ 
          (∀ n : ℕ, n < p → ¬ (n ≥ 24 ∧ n < 29 ∧ ¬ Nat.Prime n)) ∧
          p = 29 :=
by
  sorry

end smallest_prime_after_five_consecutive_nonprimes_l755_755121


namespace hiking_trip_distance_and_time_l755_755494

-- Conditions
def flat_terrain_distance : ℝ := 0.5  -- miles
def flat_terrain_speed : ℝ := 3.0     -- miles per hour
def uphill_distance : ℝ := 1.2        -- miles
def uphill_speed : ℝ := 2.5           -- miles per hour
def sandy_terrain_distance : ℝ := 0.8 -- miles
def sandy_terrain_speed : ℝ := 2.0    -- miles per hour
def downhill_distance : ℝ := 0.6      -- miles
def downhill_speed : ℝ := 2.8         -- miles per hour

-- Distance Calculation
def total_distance : ℝ := 
  flat_terrain_distance + uphill_distance + sandy_terrain_distance + downhill_distance

-- Time Calculation for each segment
def flat_terrain_time : ℝ := flat_terrain_distance / flat_terrain_speed
def uphill_time : ℝ := uphill_distance / uphill_speed
def sandy_terrain_time : ℝ := sandy_terrain_distance / sandy_terrain_speed
def downhill_time : ℝ := downhill_distance / downhill_speed

-- Total Time Calculation
def total_time_hours : ℝ := flat_terrain_time + uphill_time + sandy_terrain_time + downhill_time
def total_time_minutes : ℝ := total_time_hours * 60

-- Main Theorem
theorem hiking_trip_distance_and_time :
  total_distance = 3.1 ∧ total_time_minutes ≈ 75.66 := by
  sorry

end hiking_trip_distance_and_time_l755_755494


namespace parabola_chord_constant_l755_755080

theorem parabola_chord_constant :
  ∃ (d : ℝ), ∀ (A B : ℝ × ℝ),
      A.2 = 2 * A.1^2 ∧ B.2 = 2 * B.1^2 ∧ (A ≠ B) ∧ (A ≠ (0, d)) ∧ (B ≠ (0, d)) ∧
      (let C : ℝ × ℝ := (0, d) in
       let AC := real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) in
       let BC := real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) in
       let t := (1 / AC)^2 + (1 / BC)^2 in
       t = 16) :=
  sorry

end parabola_chord_constant_l755_755080


namespace angle_A_condition1_angle_A_condition2_angle_A_condition3_sides_bc_l755_755219

variables (a b c A B C : ℝ) (area : ℝ)
variables (cond1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
variables (cond2 : b^2 + c^2 - a^2 = b * c)
variables (cond3 : Real.sqrt 3 * Real.sin A - Real.cos A = 1)

-- Proving the angle A under different conditions
theorem angle_A_condition1 : cond1 → A = Real.pi / 3 := by
  intro h
  sorry

theorem angle_A_condition2 : cond2 → A = Real.pi / 3 := by
  intro h
  sorry

theorem angle_A_condition3 : cond3 → A = Real.pi / 3 := by
  intro h
  sorry

-- Given area S and side length a, proving b and c
theorem sides_bc (ha : a = 2) (harea : area = Real.sqrt 3) :
  ∃ b c, b * c = 4 ∧ b^2 + c^2 = 8 := by
  intro ha harea
  use 2, 2
  split
  · exact sorry  -- proof for bc = 4
  · exact sorry  -- proof for b^2 + c^2 = 8

end angle_A_condition1_angle_A_condition2_angle_A_condition3_sides_bc_l755_755219


namespace fraction_eq_repeating_decimal_l755_755273

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755273


namespace number_of_sets_without_perfect_squares_l755_755889

/-- Define the set T_i of all integers n such that 200i ≤ n < 200(i + 1). -/
def T (i : ℕ) : Set ℕ := {n | 200 * i ≤ n ∧ n < 200 * (i + 1)}

/-- The total number of sets T_i from T_0 to T_{499}. -/
def total_sets : ℕ := 500

/-- The number of sets from T_0 to T_{499} that contain at least one perfect square. -/
def sets_with_perfect_squares : ℕ := 317

/-- The number of sets from T_0 to T_{499} that do not contain any perfect squares. -/
def sets_without_perfect_squares : ℕ := total_sets - sets_with_perfect_squares

/-- Proof that the number of sets T_0, T_1, T_2, ..., T_{499} that do not contain a perfect square is 183. -/
theorem number_of_sets_without_perfect_squares : sets_without_perfect_squares = 183 :=
by
  sorry

end number_of_sets_without_perfect_squares_l755_755889


namespace julia_is_short_7_2_l755_755981

def cost (quantity : ℕ) (price : ℕ) : ℚ := price * quantity
def discount (amount : ℚ) (disc_perc : ℚ) := if amount >= 30 then amount * disc_perc else 0
def total_cost_no_discount (q_rock q_pop q_dance q_country : ℕ) : ℚ := 
  (cost q_rock 5) + (cost q_pop 10) + (cost q_dance 3) + (cost q_country 7)
def total_cost_with_discount (q_rock q_pop q_dance q_country : ℕ) : ℚ := 
  total_cost_no_discount q_rock q_pop q_dance q_country - (discount (cost q_pop 10) 0.1) - (discount (cost q_country 7) 0.1)
def julia_short : ℚ :=
  let q_rock := 3 in
  let q_pop := 4 in
  let q_dance := 2 in
  let q_country := 4 in
  let budget := 75 in
  total_cost_with_discount q_rock q_pop q_dance q_country - budget

theorem julia_is_short_7_2 {q_rock q_pop q_dance q_country : ℕ} : 
  (q_rock = 3) → (q_pop = 4) → (q_dance = 2) → (q_country = 4) →
  julia_short = 7.2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Calculation steps would go here
  sorry

end julia_is_short_7_2_l755_755981


namespace pentagon_area_calculation_l755_755177

theorem pentagon_area_calculation :
  ∀ (A B C D E : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
    [InnerProductSpace ℝ D] [InnerProductSpace ℝ E],
    -- Conditions:
    (AB DE AE AC CD DE : ℝ) (ABp : AB = 8) (ACp : AC = 12) (AEp : AE = 10)
    (AB_parallel_ED : AB ∥ DE) (equal_angles : ∀ θ : ℝ, θ = EAB ∧ θ = ABD ∧ θ = ACD ∧ θ = CDA) 
    -- Conclusion:
    (triangle_area_real : ∃ (a b c : ℕ), b.square_free ∧ a.gcd(c) = 1 ∧ 
      area_of_triangle C D E = (a * real.sqrt b) / c) → 
    (a + b + c = 264) := 
begin
  sorry
end

end pentagon_area_calculation_l755_755177


namespace slope_angle_of_line_l755_755581

theorem slope_angle_of_line : 
  let line_eq := λ x y : ℝ, x - sqrt 3 * y + 2 = 0 in
  let slope := sqrt 3 / 3 in 
  let α := real.arctan slope in 
  α = real.pi / 6 := by
sorry

end slope_angle_of_line_l755_755581


namespace square_area_condition_l755_755192

theorem square_area_condition (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (hline : x1 ≠ x2) : 
  let side := abs (x1 - x2) in side^2 = 36 :=
by
  sorry

end square_area_condition_l755_755192


namespace max_s_value_l755_755492

variables (X Y Z P X' Y' Z' : Type)
variables (p q r XX' YY' ZZ' s : ℝ)

-- Defining the conditions
def triangle_XYZ (p q r : ℝ) : Prop :=
p ≤ r ∧ r ≤ q ∧ p + q > r ∧ p + r > q ∧ q + r > p

def point_P_inside (X Y Z P : Type) : Prop :=
true -- Simplified assumption since point P is given to be inside

def segments_XX'_YY'_ZZ' (XX' YY' ZZ' : ℝ) : ℝ :=
XX' + YY' + ZZ'

def given_ratio (p q r : ℝ) : Prop :=
(p / (q + r)) = (r / (p + q))

-- The maximum value of s being 3p
def max_value_s_eq_3p (s p : ℝ) : Prop :=
s = 3 * p

-- The final theorem statement
theorem max_s_value 
  (p q r XX' YY' ZZ' s : ℝ)
  (h_triangle : triangle_XYZ p q r)
  (h_ratio : given_ratio p q r)
  (h_segments : s = segments_XX'_YY'_ZZ' XX' YY' ZZ') : 
  max_value_s_eq_3p s p :=
by
  sorry

end max_s_value_l755_755492


namespace repeating_fraction_equality_l755_755355

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755355


namespace expression_value_l755_755115

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l755_755115


namespace sufficient_prime_logarithms_l755_755056

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

-- Statement of the properties of logarithms
axiom log_mul (b x y : ℝ) : log_b b (x * y) = log_b b x + log_b b y
axiom log_div (b x y : ℝ) : log_b b (x / y) = log_b b x - log_b b y
axiom log_pow (b x : ℝ) (n : ℝ) : log_b b (x ^ n) = n * log_b b x

-- Main theorem
theorem sufficient_prime_logarithms (b : ℝ) (hb : 1 < b) :
  (∀ p : ℕ, is_prime p → ∃ Lp : ℝ, log_b b p = Lp) →
  ∀ n : ℕ, n > 0 → ∃ Ln : ℝ, log_b b n = Ln :=
by
  sorry

end sufficient_prime_logarithms_l755_755056


namespace general_formula_a_general_formula_b_sum_T_l755_755394

open Nat

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * n - 1

noncomputable def b (n : ℕ) : ℝ :=
  (1 / 2) ^ (n - 1)

noncomputable def c (n : ℕ) : ℝ :=
  a n * b n

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in range 1 (n + 1), c i

theorem general_formula_a (n : ℕ) :
  (∀ n, a 1 = 1 ∧ (a (n + 1) = a n + 2)) → a n = 2 * n - 1 := by sorry

theorem general_formula_b (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S n = 2 - b n) → b n = (1 / 2) ^ (n - 1) := by sorry

theorem sum_T (n : ℕ) :
  T n = 6 - (2 * n + 3) / (2 ^ (n - 1)) := by sorry

end general_formula_a_general_formula_b_sum_T_l755_755394


namespace min_ab_value_l755_755843

theorem min_ab_value (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : (a * b) ≥ 2 := by 
  sorry

end min_ab_value_l755_755843


namespace red_crayons_per_person_l755_755075

theorem red_crayons_per_person :
  ∃ red_crayons_per_person : ℕ,
    let total_red_crayons := 2 * 8 in
    let num_people := 3 in
    let result := total_red_crayons / num_people in
    result = 5 :=
by
  sorry

end red_crayons_per_person_l755_755075


namespace Jason_borrowed_420_dollars_l755_755877

theorem Jason_borrowed_420_dollars :
  (∑ i in range 6, (i + 1) * 5) * 4 = 420 :=
by
  sorry

end Jason_borrowed_420_dollars_l755_755877


namespace fraction_eq_repeating_decimal_l755_755271

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755271


namespace fifteenth_value_l755_755381

def r (a b : ℕ) := a % b 

theorem fifteenth_value (n : ℕ) : 
  (∀ n, 0 ≤ n ∧ ∃ k, n = k ∧ r (7 * n) 11 ≤ 5) → 
  (let lst := (List.range 100).filter (λ n, r (7 * n) 11 ≤ 5) in
  lst.nth 14 = some 29) :=
sorry

end fifteenth_value_l755_755381


namespace Antoine_fruit_supply_required_l755_755206

def total_fruit_sacks_per_week 
  (strawberries_by_bakery : List ℕ) 
  (blueberries_by_bakery : List ℕ) 
  (raspberries_by_bakery : List ℕ)
  : (ℕ × ℕ × ℕ) :=
  (strawberries_by_bakery.sum, blueberries_by_bakery.sum, raspberries_by_bakery.sum)

theorem Antoine_fruit_supply_required:
  let strawberries := [2, 4, 12, 8, 15, 5] in
  let blueberries := [3, 2, 10, 4, 6, 9] in
  let raspberries := [5, 8, 7, 3, 12, 11] in
  let (total_strawberries, total_blueberries, total_raspberries) := total_fruit_sacks_per_week strawberries blueberries raspberries in
  10 * total_strawberries = 460 ∧ 10 * total_blueberries = 340 ∧ 10 * total_raspberries = 460 := 
by 
  sorry

end Antoine_fruit_supply_required_l755_755206


namespace fraction_numerator_less_denominator_l755_755573

theorem fraction_numerator_less_denominator (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  (8 * x - 3 < 9 + 5 * x) ↔ (-3 ≤ x ∧ x < 3) :=
by sorry

end fraction_numerator_less_denominator_l755_755573


namespace square_area_l755_755188

theorem square_area :
  (∀ x : ℝ, (x^2 + 4*x + 3 = 8) → ((abs(1 - (-5)))^2 = 36)) :=
begin
  -- conditions
  intro x,
  intro h,
  -- proof steps (intentionally left out)
  sorry
end

end square_area_l755_755188


namespace problem_statement_l755_755760

variables {Point Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Conditions
def parallel (l : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def perpendicular_planes (α β : Plane) : Prop := sorry

-- The proof problem
theorem problem_statement (h1 : parallel l α) (h2 : perpendicular l β) : perpendicular_planes α β :=
sorry

end problem_statement_l755_755760


namespace cards_needed_l755_755506

theorem cards_needed (n : ℕ) (h : 0 < n) : 
∃ (c : ℕ), (∀ (t : ℕ), t ≤ n! → ∃ (cards : finset ℕ), 
    (∀ card ∈ cards, ∃ m : ℕ, card = m! ∧ m ≤ n) ∧ 
    (cards.sum id = t)) ∧ 
    c = (n * (n + 1)) / 2 + 1 :=
sorry

end cards_needed_l755_755506


namespace lives_per_player_l755_755082

theorem lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8) (h2 : additional_players = 2) (h3 : total_lives = 60) : 
  total_lives / (initial_players + additional_players) = 6 :=
by 
  sorry

end lives_per_player_l755_755082


namespace eval_expression_l755_755216

theorem eval_expression : (Real.pi + 2023)^0 + 2 * Real.sin (45 * Real.pi / 180) - (1 / 2)^(-1 : ℤ) + abs (Real.sqrt 2 - 2) = 1 :=
by
  sorry

end eval_expression_l755_755216


namespace log_base_1_over_5_decreasing_interval_l755_755962

theorem log_base_1_over_5_decreasing_interval (x : ℝ) (hx : x > 2) :
  ∀ y : ℝ, y = log (1/5) (abs (x-2)) → y < y :=
sorry

end log_base_1_over_5_decreasing_interval_l755_755962


namespace gcd_subtraction_l755_755368

theorem gcd_subtraction (a b : ℕ) (ha : a = 105) (hb : b = 90) : 
  let d := Nat.gcd a b in
  let smallest_prime_factor := if Nat.min_fac d = 0 then 0 else Nat.min_fac d in
  d = 15 ∧ 10 - smallest_prime_factor = 7 :=
by
  sorry

end gcd_subtraction_l755_755368


namespace arithmetic_sequence_sum_six_terms_l755_755587

noncomputable def sum_of_first_six_terms (a : ℤ) (d : ℤ) : ℤ :=
  let a1 := a
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let a5 := a4 + d
  let a6 := a5 + d
  a1 + a2 + a3 + a4 + a5 + a6

theorem arithmetic_sequence_sum_six_terms
  (a3 a4 a5 : ℤ)
  (h3 : a3 = 8)
  (h4 : a4 = 13)
  (h5 : a5 = 18)
  (d : ℤ) (a : ℤ)
  (h_d : d = a4 - a3)
  (h_a : a + 2 * d = 8) :
  sum_of_first_six_terms a d = 63 :=
by
  sorry

end arithmetic_sequence_sum_six_terms_l755_755587


namespace factorize_expression_l755_755250

variable (a x : ℝ)

theorem factorize_expression : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := 
by 
  sorry

end factorize_expression_l755_755250


namespace condensed_milk_greater_than_jam_l755_755138

def caloric_values (a b c : ℝ) : Prop :=
3 * a + 4 * b + 2 * c > 2 * a + 3 * b + 4 * c ∧ 
3 * a + 4 * b + 2 * c > 4 * a + 2 * b + 3 * c

theorem condensed_milk_greater_than_jam (a b c : ℝ)
(h₁ : caloric_values a b c) : b > c :=
begin
  sorry
end

end condensed_milk_greater_than_jam_l755_755138


namespace children_neither_happy_nor_sad_l755_755527

-- conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10

-- proof problem
theorem children_neither_happy_nor_sad :
  total_children - happy_children - sad_children = 20 := by
  sorry

end children_neither_happy_nor_sad_l755_755527


namespace train_speed_l755_755673

theorem train_speed (train_length bridge_length : ℕ) (time : ℝ)
  (h_train_length : train_length = 110)
  (h_bridge_length : bridge_length = 290)
  (h_time : time = 23.998080153587715) :
  (train_length + bridge_length) / time * 3.6 = 60 := 
by
  rw [h_train_length, h_bridge_length, h_time]
  sorry

end train_speed_l755_755673


namespace Haman_initial_trays_l755_755436

theorem Haman_initial_trays 
  (eggs_in_tray : ℕ)
  (total_eggs_sold : ℕ)
  (trays_dropped : ℕ)
  (additional_trays : ℕ)
  (trays_finally_sold : ℕ)
  (std_trays_sold : total_eggs_sold / eggs_in_tray = trays_finally_sold) 
  (eggs_in_tray_def : eggs_in_tray = 30) 
  (total_eggs_sold_def : total_eggs_sold = 540)
  (trays_dropped_def : trays_dropped = 2)
  (additional_trays_def : additional_trays = 7) :
  trays_finally_sold - additional_trays + trays_dropped = 13 := 
by 
  sorry

end Haman_initial_trays_l755_755436


namespace recurring_decimal_to_fraction_l755_755321

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755321


namespace question1_question2_l755_755524

namespace VectorSpace

open_locale classical

def a : ℝ × ℝ × ℝ := (3, 5, -4)
def b : ℝ × ℝ × ℝ := (2, 1, 8)

def dot_prod (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def lin_comb (λ μ : ℝ) : ℝ × ℝ × ℝ :=
  (λ * a.1 + μ * b.1, λ * a.2 + μ * b.2, λ * a.3 + μ * b.3)

theorem question1 :
  2 • a + 3 • b = (12, 13, 16) ∧
  3 • a - 2 • b = (5, 13, -28) ∧
  dot_prod a b = -21 := sorry

theorem question2 (λ μ : ℝ) :
  (lin_comb λ μ).3 = 0 → λ = 2 * μ := sorry

end VectorSpace

end question1_question2_l755_755524


namespace general_term_formula_l755_755580

theorem general_term_formula (n : ℕ) : n > 0 → 
  (∀ k : ℕ, (k = 1 → a_1 = real.sqrt 2) ∧ 
            (k = 2 → a_2 = real.sqrt 5) ∧ 
            (k = 3 → a_3 = real.sqrt 8) ∧ 
            (k = 4 → a_4 = real.sqrt 11) ∧ 
             k = n → a_n = real.sqrt (3 * n - 1)) :=
sorry

end general_term_formula_l755_755580


namespace helga_shoes_l755_755825

theorem helga_shoes (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := 
by
  sorry

end helga_shoes_l755_755825


namespace midpoint_product_zero_l755_755008

theorem midpoint_product_zero (x y : ℝ)
  (h_midpoint_x : (2 + x) / 2 = 4)
  (h_midpoint_y : (6 + y) / 2 = 3) :
  x * y = 0 :=
by
  sorry

end midpoint_product_zero_l755_755008


namespace geometric_sequence_property_l755_755858

noncomputable def geometric_sequence_sum : ℕ → (ℕ → ℝ) → Prop :=
  λ n a, ∀ k, a (k + n) = a k * a n / a 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_positive : ∀ n, 0 < a n)
  (h_geom : geometric_sequence_sum 1 a)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = 16) :
  a 3 ^ 2 + 2 * a 2 * a 6 + a 3 * a 7 = 400 :=
by
  sorry

end geometric_sequence_property_l755_755858


namespace sum_first_n_terms_of_c_n_l755_755870

-- Given conditions
variables {a : ℕ → ℕ} -- the arithmetic sequence
variables (h1 : a 2 = 4)
variables (h2 : a 4 + a 7 = 15)

-- Question 1: Find the general term of the sequence
def general_term_arithmetic_sequence (a : ℕ → ℕ) := 
  ∃ n : ℕ, ∀ n, a n = n + 2

-- Definition of b_n and c_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (∑ i in Finset.range n, a (i+1)) / n
def c_n (a : ℕ → ℕ) (n : ℕ) := a n + b_n a n

-- Sum of the first n terms of the sequence {c_n}
def T_n (c_n : ℕ → ℕ) (n : ℕ) := ∑ i in Finset.range n, c_n (i+1)

-- Final statement
theorem sum_first_n_terms_of_c_n (a : ℕ → ℕ) (h1 : a 2 = 4) (h2 : a 4 + a 7 = 15) : 
  general_term_arithmetic_sequence a → 
  ∃ T_n : ℕ → ℕ, T_n n = (3 * n^2 + 21 * n) / 4 :=
by
  sorry

end sum_first_n_terms_of_c_n_l755_755870


namespace range_of_m_l755_755782

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x < y → y < 0 → f y < f x) (h_cond : ∀ m : ℝ, f (1 - m) < f (m - 3)) : ∀ m, 1 < m ∧ m < 2 :=
by
  intros m
  sorry

end range_of_m_l755_755782


namespace quadratic_sum_unique_solution_l755_755714

theorem quadratic_sum_unique_solution :
  (∑ b in ({ b | ∃ x : ℝ, (4 * x^2 + 12 * x + b = 0) ∧ (144 - 16 * b = 0) }).to_finset, b) = 9 :=
by 
  sorry

end quadratic_sum_unique_solution_l755_755714


namespace distinct_pairs_count_l755_755572

noncomputable def num_distinct_pairs : ℕ :=
  let equations_satisfied (x y : ℝ) : Prop := x = x^2 + y^2 ∧ y = 2 * x * y
  let solutions_set : set (ℝ × ℝ) := {p : ℝ × ℝ | equations_satisfied p.1 p.2}
  set.to_finset solutions_set).card

theorem distinct_pairs_count :
  num_distinct_pairs = 4 :=
sorry

end distinct_pairs_count_l755_755572


namespace f_decreasing_a_range_l755_755425

/-- Define the function f(x) --/
def f (x : ℝ) : ℝ := 2 / (x - 1) - x

/-- Prove that f(x) is a decreasing function for x in [2, +∞) --/
theorem f_decreasing : ∀ x₁ x₂ : ℝ, 2 ≤ x₁ → 2 ≤ x₂ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

/-- Given (a + x)(x - 1) > 2 for all x in [2, +∞), prove the range of a --/
theorem a_range : (∀ x : ℝ, 2 ≤ x → (a + x) * (x - 1) > 2) → a > 0 :=
by
  sorry

end f_decreasing_a_range_l755_755425


namespace triangle_altitude_l755_755196

-- Define the base and altitudes of the triangle and parallelogram
variables (b h_triangle : ℝ)

-- Define the area of the parallelogram
def area_parallelogram (b : ℝ) (h_parallelogram : ℝ) : ℝ :=
  b * h_parallelogram

-- Define the area of the triangle
def area_triangle (b : ℝ) (h_triangle : ℝ) : ℝ :=
  1/2 * b * h_triangle

-- The main theorem statement
theorem triangle_altitude (h_parallelogram : ℝ) (eq_areas : area_parallelogram b h_parallelogram = area_triangle b h_triangle) : h_triangle = 2 * h_parallelogram :=
by
  sorry

end triangle_altitude_l755_755196


namespace max_regions_1002_1000_l755_755915

def regions_through_point (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 1

def max_regions (a b : ℕ) : ℕ := 
  let rB := regions_through_point b
  let first_line_through_A := rB + b + 1
  let remaining_lines_through_A := (a - 1) * (b + 2)
  first_line_through_A + remaining_lines_through_A

theorem max_regions_1002_1000 : max_regions 1002 1000 = 1504503 := by
  sorry

end max_regions_1002_1000_l755_755915


namespace smallest_x_inequality_l755_755726

theorem smallest_x_inequality : ∃ x : ℝ, (x^2 - 8 * x + 15 ≤ 0) ∧ (∀ y : ℝ, (y^2 - 8 * y + 15 ≤ 0) → (3 ≤ y)) ∧ x = 3 := 
sorry

end smallest_x_inequality_l755_755726


namespace calculate_expression_l755_755215

theorem calculate_expression : 5 * (-2) + Real.exp (0 * Real.log(π)) + Int.pow (-1) 2023 - Int.pow 2 3 = -18 := by
  sorry

end calculate_expression_l755_755215


namespace minimum_value_f_maximum_value_inequality_l755_755421

-- Define the given functions and the natural logarithm base e
def f (x : ℝ) : ℝ := Real.exp x - x + 1/2 * x^2
def g (x a b : ℝ) : ℝ := 1/2 * x^2 + a * x + b

-- Statement 1: Proof of minimum value of f(x) at x = 0 and equals to 1
theorem minimum_value_f :
    f 0 = 1 := sorry

-- Statement 2: Proof of maximum value of (b(a + 1))/2 equals to e/4 under the condition f(x) ≥ g(x)
theorem maximum_value_inequality (a b : ℝ) :
    (∀ x, f x ≥ g x a b) → (b * (a + 1)) / 2 = Real.exp(1) / 4 := sorry

end minimum_value_f_maximum_value_inequality_l755_755421


namespace fraction_for_repeating_56_l755_755265

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755265


namespace alien_energy_cells_base10_l755_755203

theorem alien_energy_cells_base10 : 
  let n := 3 * 7^2 + 2 * 7^1 + 1 * 7^0 
  in n = 162 :=
by
  let n := 3 * 7^2 + 2 * 7^1 + 1 * 7^0
  show n = 162
  sorry

end alien_energy_cells_base10_l755_755203


namespace find_expression_and_maximum_and_a_range_l755_755805

open Real

noncomputable def f (m n x : ℝ) : ℝ := (m * x) / (x^2 + n)
noncomputable def g (a x : ℝ) : ℝ := x^2 - 2 * a * x + a

theorem find_expression_and_maximum_and_a_range (m n : ℝ) (f_extremum : f m n 1 = 2) (f_deriv_zero : deriv (f m n) 1 = 0) :
  (f 4 1 = λ x, (4 * x) / (x^2 + 1)) ∧ (∀ x > 0, f 4 1 x ≤ 2) ∧ ((∀ x₁, ∃ x₂ ∈ Icc (-1) 0, g y = x₂ ≤ f 4 1 x₁) → a ∈ Iic (-1)) :=
by sorry

end find_expression_and_maximum_and_a_range_l755_755805


namespace f_neg_3_eq_neg_3_l755_755893

noncomputable def f (x : ℝ) : ℝ := 
 if x > 0 then log (x + 5) / log 2 else - log (-x + 5) / log 2

theorem f_neg_3_eq_neg_3 : f (-3) = -3 :=
by
  sorry

end f_neg_3_eq_neg_3_l755_755893


namespace exponent_expression_value_l755_755700

theorem exponent_expression_value :
  (0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.25 ^ (1 / 2)) = 10 := by
{
  -- Given conditions
  have h1 : 0.064 ^ (-1 / 3) = 5 / 2, by sorry,
  have h2 : (-7 / 8) ^ 0 = 1, by sorry,
  have h3 : 16 ^ 0.75 = 8, by sorry,
  have h4 : 0.25 ^ (1 / 2) = 1 / 2, by sorry,

  -- Proof
  calc
    0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.25 ^ (1 / 2)
        = (5 / 2) - 1 + 8 + 1 / 2 : by rw [h1, h2, h3, h4]
    ... = 10 : by norm_num
}

end exponent_expression_value_l755_755700


namespace number_of_sanferminera_colorings_l755_755723

def is_sanferminera_coloring (coloring : ℚ → bool) : Prop :=
  ∀ x y : ℚ, x ≠ y →
    (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) →
    coloring x ≠ coloring y

theorem number_of_sanferminera_colorings : (∃! c : ℚ → bool, is_sanferminera_coloring c) :=
sorry

end number_of_sanferminera_colorings_l755_755723


namespace point_set_M_max_angle_l755_755982

noncomputable def distance (A B : ℝ×ℝ) : ℝ :=
  ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt

theorem point_set_M (A B : ℝ × ℝ) :
  let AB := distance A B in
  let I := (A.1 + B.1) / 2, (A.2 + B.2) / 2 in
  let r := (\sqrt{3} / 2) * AB in
  { C : ℝ × ℝ | distance A C ^ 2 + distance B C ^ 2 = 2 * AB ^ 2 } =
  { C : ℝ × ℝ | distance I C = r } :=
by sorry

theorem max_angle (A B : ℝ × ℝ) :
  let AB := distance A B in
  let I := (A.1 + B.1) / 2, (A.2 + B.2) / 2 in
  let M := { C : ℝ × ℝ | distance A C ^ 2 + distance B C ^ 2 = 2 * AB ^ 2 } in
  ∀ C ∈ M, 
  angle C A B ≤ 60 :=
by sorry

end point_set_M_max_angle_l755_755982


namespace relationship_among_abc_l755_755890

noncomputable def a := (3 / 4)^(2 / 3)
noncomputable def b := (2 / 3)^(3 / 4)
noncomputable def c := Real.logBase (2 / 3) (4 / 3)

theorem relationship_among_abc : a > b ∧ b > c := sorry

end relationship_among_abc_l755_755890


namespace length_of_AB_l755_755536

theorem length_of_AB (A B C D E F G : Point)
  (h1 : midpoint A B C)
  (h2 : midpoint A C D)
  (h3 : midpoint A D E)
  (h4 : midpoint A E F)
  (h5 : midpoint A F G)
  (h6 : length A G = 5) : length A B = 160 :=
sorry

end length_of_AB_l755_755536


namespace recurring_decimal_to_fraction_l755_755323

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755323


namespace largest_int_divisible_condition_l755_755370

theorem largest_int_divisible_condition :
  ∃ N : ℕ, (∀ n : ℕ, n ≤ N → (⌊n / 3⌋ = ⌊n / 5⌋ + ⌊n / 7⌋ - ⌊n / 35⌋)) ∧ N = 65 :=
begin
  sorry
end

end largest_int_divisible_condition_l755_755370


namespace quadratic_root_square_condition_l755_755738

theorem quadratic_root_square_condition (p q r : ℝ) 
  (h1 : ∃ α β : ℝ, α + β = -q / p ∧ α * β = r / p ∧ β = α^2) : p - 4 * q ≥ 0 :=
sorry

end quadratic_root_square_condition_l755_755738


namespace general_formula_for_an_sum_sn_less_than_l755_755774

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24
def a₁ : ℝ := 3
def a₂ : ℝ := 8
def diff_seq (n : ℕ) : ℝ := 2 * n + 3
def a (n : ℕ) : ℝ := (n : ℝ)^2 + 2 * (n : ℝ)
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, 1 / a k)

-- We are to prove these statements
theorem general_formula_for_an : ∀ (n : ℕ), a n = n^2 + 2 * n := by
  sorry

theorem sum_sn_less_than : ∀ (n : ℕ), S n < 3 / 4 := by
  sorry

end general_formula_for_an_sum_sn_less_than_l755_755774


namespace gen_formula_Sn_bounded_l755_755770

-- Defining the sequence {a_n}
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24

-- Conditions
variable (a : ℕ → ℝ)
variable (n : ℕ)
variable (S : ℕ → ℝ)

-- Roots of f(x)
axiom a1 : a 1 = 3
axiom a2 : a 2 = 8

-- Arithmetic sequence property
axiom a_n_arithmetic : ∀ n : ℕ, a (n + 1) - a n = 2 * n + 5

-- Increasing sequence
axiom a_increasing : ∀ n : ℕ, a n < a (n + 1)

-- Define S_n
def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, 1 / a (k + 1)

-- Theorems we need to prove
theorem gen_formula : a n = n^2 + 2 * n :=
sorry

theorem Sn_bounded : S n < 3 / 4 :=
sorry

end gen_formula_Sn_bounded_l755_755770


namespace fenced_area_correct_l755_755959

-- Define the dimensions of the rectangle
def length := 20
def width := 18

-- Define the dimensions of the cutouts
def square_cutout1 := 4
def square_cutout2 := 2

-- Define the areas of the rectangle and the cutouts
def area_rectangle := length * width
def area_cutout1 := square_cutout1 * square_cutout1
def area_cutout2 := square_cutout2 * square_cutout2

-- Define the total area within the fence
def total_area_within_fence := area_rectangle - area_cutout1 - area_cutout2

-- The theorem that needs to be proven
theorem fenced_area_correct : total_area_within_fence = 340 := by
  sorry

end fenced_area_correct_l755_755959


namespace value_of_expression_l755_755778

theorem value_of_expression (x y : ℝ) (h1 : 3 * x + 2 * y = 7) (h2 : 2 * x + 3 * y = 8) :
  13 * x ^ 2 + 22 * x * y + 13 * y ^ 2 = 113 :=
sorry

end value_of_expression_l755_755778


namespace repeating_decimal_is_fraction_l755_755295

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755295


namespace A_days_l755_755640

-- Given conditions:
def B_days : ℕ := 15
def total_wages : ℕ := 3100
def A_wages : ℕ := 1860
def B_wages : ℕ := total_wages - A_wages

-- Definition of the problem in Lean 4:
theorem A_days (A : ℕ): B_days = 15 ∧ total_wages = 3100 ∧ A_wages = 1860 ∧ B_wages = (total_wages - A_wages) → 
  (A = 10) → (1 / A + 1 / B_days) * (A * B_wages / (A_wages + B_wages)) = (1 / B_days):
  by
    intros h
    sorry

end A_days_l755_755640


namespace repeating_decimal_is_fraction_l755_755297

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755297


namespace square_area_l755_755186

theorem square_area :
  (∀ x : ℝ, (x^2 + 4*x + 3 = 8) → ((abs(1 - (-5)))^2 = 36)) :=
begin
  -- conditions
  intro x,
  intro h,
  -- proof steps (intentionally left out)
  sorry
end

end square_area_l755_755186


namespace cost_of_green_pants_correct_l755_755692

noncomputable def cost_of_green_pants {P : ℝ} : ℝ :=
  let red_tshirt_price := 100 * 0.8
  let black_tshirt_price := 100 * 0.9
  let total_cost_red := 3 * red_tshirt_price
  let total_cost_black := 2 * black_tshirt_price
  let total_cost_blue := 2 * P
  let green_pant_price := 0.85 * P
  let total_cost_green := 2 * green_pant_price
  let total_cost := total_cost_red + total_cost_black + total_cost_blue + total_cost_green
  green_pant_price

theorem cost_of_green_pants_correct (P : ℝ) (h : 3.7 * P + 420 = 1500) : cost_of_green_pants = 0.85 * (1080 / 3.7) := by
  sorry

end cost_of_green_pants_correct_l755_755692


namespace solve_problem_l755_755793

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ a1 d, a 2 = 6 ∧ a 5 = 12 ∧ ∀ n, a n = a1 + (n - 1) * d

def bn_sn_rel (S : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, S 1 + (1 / 2) * b 1 = 1 ∧ (n ≥ 2 → S n + (1 / 2) * b n = 1)

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 2 / 3 ∧ ∀ n, n ≥ 2 → b n = (1 / 3) * b (n - 1)

def cn_sequence (c : ℕ → ℝ) (a : ℕ → ℕ) (b : ℕ → ℝ) : Prop :=
  ∀ n, c n = -2 / (a n * Real.log (b n / 2) / Real.log 3)

def Tn_sum (c : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n, T n = ∑ i in range n, c i

theorem solve_problem (a : ℕ → ℕ) (b : ℕ → ℝ) (S T : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence a →
  bn_sn_rel S b →
  geometric_sequence b →
  cn_sequence (-2 / (a n * log base (b n / 2))) →
  Tn_sum sum_range n c →
  (∀ n, T n < (m - 2013) / 2) →
  m ≥ 2015 :=
sorry

end solve_problem_l755_755793


namespace total_gas_cost_l755_755941

def gas_price_station_1 : ℝ := 3
def gas_price_station_2 : ℝ := 3.5
def gas_price_station_3 : ℝ := 4
def gas_price_station_4 : ℝ := 4.5
def tank_capacity : ℝ := 12

theorem total_gas_cost :
  let cost_station_1 := tank_capacity * gas_price_station_1
  let cost_station_2 := tank_capacity * gas_price_station_2
  let cost_station_3 := tank_capacity * gas_price_station_3
  let cost_station_4 := tank_capacity * gas_price_station_4
  cost_station_1 + cost_station_2 + cost_station_3 + cost_station_4 = 180 :=
by
  -- Proof is skipped
  sorry

end total_gas_cost_l755_755941


namespace eighth_term_of_arithmetic_sequence_l755_755603

theorem eighth_term_of_arithmetic_sequence
  (a l : ℕ) (n : ℕ) (h₁ : a = 4) (h₂ : l = 88) (h₃ : n = 30) :
  (a + 7 * (l - a) / (n - 1) = (676 : ℚ) / 29) :=
by
  sorry

end eighth_term_of_arithmetic_sequence_l755_755603


namespace amoeba_count_after_one_week_l755_755681

/-- An amoeba is placed in a puddle and splits into three amoebas on the same day. Each subsequent
    day, every amoeba in the puddle splits into three new amoebas. -/
theorem amoeba_count_after_one_week : 
  let initial_amoebas := 1
  let daily_split := 3
  let days := 7
  (initial_amoebas * (daily_split ^ days)) = 2187 :=
by
  sorry

end amoeba_count_after_one_week_l755_755681


namespace range_of_a_l755_755811

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l755_755811


namespace fraction_of_repeating_decimal_l755_755334

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755334


namespace sum_of_cubes_limit_l755_755010

variable (s r : ℝ)
variable (h : -1 < r ∧ r < 0)

theorem sum_of_cubes_limit (h : -1 < r ∧ r < 0) : 
  limit_at_infinity (sum $ λ n, (s * ((r + 1)^n))^3) = s^3 / (1 - (r + 1)^3) :=
sorry

end sum_of_cubes_limit_l755_755010


namespace repeating_decimal_eq_fraction_l755_755339

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755339


namespace greatest_common_multiple_of_9_and_15_less_than_120_l755_755997

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end greatest_common_multiple_of_9_and_15_less_than_120_l755_755997


namespace minimum_value_MP_MF_l755_755392

-- Definitions of the conditions
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def on_parabola (M : ℝ × ℝ) : Prop := parabola M.1 M.2
def focus (F : ℝ × ℝ) : Prop := F = (2, 0)  -- Focus of y^2 = 8x
def fixed_point (P : ℝ × ℝ) : Prop := P = (2, 1)

-- The statement we want to prove
theorem minimum_value_MP_MF (M F P : ℝ × ℝ) (hM : on_parabola M) (hF : focus F) (hP : fixed_point P) : 
  let MP := dist M P in
  let MF := dist M F in
  MP + MF = 4 :=
sorry

end minimum_value_MP_MF_l755_755392


namespace max_eccentricity_of_ellipse_l755_755433

theorem max_eccentricity_of_ellipse 
  (R_large : ℝ)
  (r_cylinder : ℝ)
  (R_small : ℝ)
  (D_centers : ℝ)
  (a : ℝ)
  (b : ℝ)
  (e : ℝ) :
  R_large = 1 → 
  r_cylinder = 1 → 
  R_small = 1/4 → 
  D_centers = 10/3 → 
  a = 5/3 → 
  b = 1 → 
  e = Real.sqrt (1 - (b / a) ^ 2) → 
  e = 4/5 := by 
  sorry

end max_eccentricity_of_ellipse_l755_755433


namespace prob_at_least_6_heads_in_8_flips_l755_755161

def fairCoinFlipProb : ℕ -> ℚ
| 8 := 17 / 256
| _ := 0

theorem prob_at_least_6_heads_in_8_flips : fairCoinFlipProb 8 = 17 / 256 :=
by
  sorry

end prob_at_least_6_heads_in_8_flips_l755_755161


namespace repeating_decimal_as_fraction_l755_755312

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755312


namespace find_a6_of_arithmetic_seq_l755_755976

noncomputable def arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_a6_of_arithmetic_seq 
  (a1 d : ℝ) 
  (S3 : ℝ) 
  (h_a1 : a1 = 2) 
  (h_S3 : S3 = 12) 
  (h_sum : S3 = sum_of_arithmetic_sequence 3 a1 d) :
  arithmetic_sequence 6 a1 d = 12 := 
sorry

end find_a6_of_arithmetic_seq_l755_755976


namespace EF_perpendicular_AI_l755_755872

open EuclideanGeometry

-- Define the properties and conditions
variables {A B C I D N M E F : Point}
variables [Triangle A B C]

noncomputable def incenter := I
def l_B := Perpendicular (CI B)
def l_C := Perpendicular (BI C)
def intersection_lB_lC := D
def intersection_lB_AC := N
def intersection_lC_AB := M
def midpoint_BN := E
def midpoint_CM := F

-- Define the theorem
theorem EF_perpendicular_AI (h1 : incenter I A B C)
  (h2 : IsPerpendicular l_B (CI B))
  (h3 : IsPerpendicular l_C (BI C))
  (h4 : intersection l_B l_C = D)
  (h5 : intersection l_B (AC A C) = N)
  (h6 : intersection l_C (AB A B) = M)
  (h7 : Midpoint E B N)
  (h8 : Midpoint F C M) :
  Perpendicular EF (AI A I) := sorry

end EF_perpendicular_AI_l755_755872


namespace max_blocks_fit_l755_755607

theorem max_blocks_fit :
  ∃ (blocks : ℕ), blocks = 12 ∧ 
  (∀ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 1 → 
  ∀ (x y z : ℕ), x = 5 ∧ y = 4 ∧ z = 4 → 
  blocks = (x * y * z) / (a * b * c) ∧
  blocks = (y * z / (b * c) * (5 / a))) :=
sorry

end max_blocks_fit_l755_755607


namespace log_diff_l755_755505

variable {a : ℕ → ℝ}

-- Conditions
axiom a_pos (n: ℕ) : 0 < a n
axiom a_mul (m n : ℕ) : a (m * n) = a m * a n
axiom a_bound (B : ℝ) (hB : 0 < B) (m n : ℕ) (h : m < n) : a m < B * a n

-- Goal
theorem log_diff : log 2015 (a 2015) - log 2014 (a 2014) = 0 :=
sorry

end log_diff_l755_755505


namespace trajectory_and_slope_l755_755397

-- We define the triangle and its vertices
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

-- We define the coordinate C and the condition on the slopes' product
def trajectory_eq (A B : ℝ × ℝ) (m : ℝ) (x y : ℝ) : Prop :=
  let k1 := (y + A.snd) / x
  let k2 := (y - B.snd) / x
  (k1 * k2 = m) → (x^2 * (-m/3) + y^2 * (1/3) = 1)

def slope_EF_is_constant (m : ℝ) (P : ℝ × ℝ) (EF : ℝ) : Prop :=
  m = -3/4 →
  P.1 = 1 →
  (∃ t > 0, P.2 = t ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    (x1^2 / 4 + y1^2 / 3 = 1) ∧ (x2^2 / 4 + y2^2 / 3 = 1) ∧
    (x1 = -1) ∧ (y1 = √3) ∧
    (x2 = 1) ∧ (y2 = √3) ∧
    EF = (y2 - y1) / (x2 - x1))) →
  EF = 1/2

-- Final statement combining both parts
theorem trajectory_and_slope (A B : ℝ × ℝ) (m : ℝ) (x y : ℝ) (P : ℝ × ℝ) (EF : ℝ) :
  trajectory_eq A B m x y →
  slope_EF_is_constant m P EF := by
  sorry

end trajectory_and_slope_l755_755397


namespace sand_removal_l755_755081

theorem sand_removal :
  let initial_weight := (8 / 3 : ℚ)
  let first_removal := (1 / 4 : ℚ)
  let second_removal := (5 / 6 : ℚ)
  initial_weight - (first_removal + second_removal) = (13 / 12 : ℚ) := by
  -- sorry is used here to skip the proof as instructed
  sorry

end sand_removal_l755_755081


namespace abs_nonneg_rational_l755_755635

theorem abs_nonneg_rational (a : ℚ) : |a| ≥ 0 :=
sorry

end abs_nonneg_rational_l755_755635


namespace bruce_bhishma_meet_again_l755_755618

theorem bruce_bhishma_meet_again (L S_B S_H : ℕ) (hL : L = 600) (hSB : S_B = 30) (hSH : S_H = 20) : 
  ∃ t : ℕ, t = 60 ∧ (t * S_B - t * S_H) % L = 0 :=
by
  sorry

end bruce_bhishma_meet_again_l755_755618


namespace max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l755_755420

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem max_values_of_f (k : ℤ) : 
  ∃ x, f x = 2 ∧ x = 4 * (k : ℝ) * Real.pi - (2 * Real.pi / 3) := 
sorry

theorem smallest_positive_period_of_f : 
  ∃ T, T = 4 * Real.pi := 
sorry

theorem intervals_where_f_is_monotonically_increasing (k : ℤ) : 
  ∀ x, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ x) ∧ (x ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  ∀ y, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ y) ∧ (y ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  (x ≤ y ↔ f x ≤ f y) :=
sorry

end max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l755_755420


namespace line_equation_l755_755257

noncomputable def center_of_circle : (ℝ × ℝ) := (-1, 2)

noncomputable def slope : ℝ := 1

theorem line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → slope = 1 →
  x - y + 3 = 0 :=
by sorry

end line_equation_l755_755257


namespace domain_of_tan_l755_755559

noncomputable def is_excluded_from_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 1 + 6 * k

theorem domain_of_tan {x : ℝ} :
  ∀ x, ¬ is_excluded_from_domain x ↔ ¬ ∃ k : ℤ, x = 1 + 6 * k := 
by 
  sorry

end domain_of_tan_l755_755559


namespace fraction_eq_repeating_decimal_l755_755366

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755366


namespace repeating_decimal_as_fraction_l755_755306

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755306


namespace percentage_of_athletes_born_in_july_l755_755051

theorem percentage_of_athletes_born_in_july : 
  let N := 150 
  let n_July := 22 
  \left(\frac{n_July}{N} \times 100 \right) = 14.67 :=
by
  -- Definitions and numerical verification
  sorry

end percentage_of_athletes_born_in_july_l755_755051


namespace repeating_fraction_equality_l755_755346

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755346


namespace point_inside_circle_l755_755489

def center : ℝ × ℝ := (-2, -3)
def radius : ℝ := 6
def candidate_point : ℝ × ℝ := (0, -3)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem point_inside_circle : distance center candidate_point < radius :=
by sorry

end point_inside_circle_l755_755489


namespace third_number_on_10th_row_l755_755974

/-- Consider the triangular arrangement of positive integers:
    1
    2 3
    4 5 6
    7 8 9 10
    11 12 13 14 15
    ...
    This theorem states that the 3rd number from the left on the 10th row of this triangular array is 48. -/
theorem third_number_on_10th_row : (triangular_number n) (row 10) = 48 := by
  sorry

end third_number_on_10th_row_l755_755974


namespace min_cuts_for_payment_7_days_l755_755617

theorem min_cuts_for_payment_7_days (n : ℕ) (h : n = 7) : ∃ k, k = 1 :=
by sorry

end min_cuts_for_payment_7_days_l755_755617


namespace minimize_sum_of_squares_of_roots_l755_755417

theorem minimize_sum_of_squares_of_roots (m : ℝ) (h : 100 - 20 * m ≥ 0) :
  (∀ a b : ℝ, (∀ x : ℝ, 5 * x^2 - 10 * x + m = 0 → x = a ∨ x = b) → (4 - 2 * m / 5) ≥ (4 - 2 * 5 / 5)) :=
by
  sorry

end minimize_sum_of_squares_of_roots_l755_755417


namespace count_integers_P_leq_zero_l755_755232

noncomputable def P (x : ℤ) : ℤ :=
  ∏ i in finset.range(100), (x - (i + 1)^2)

theorem count_integers_P_leq_zero :
  ∃ (count : ℕ), count = 5100 ∧ (∀ n : ℤ, P n <= 0 → n ∈ range count) := sorry

end count_integers_P_leq_zero_l755_755232


namespace members_who_play_both_l755_755131

theorem members_who_play_both (N B T Neither : ℕ) (hN : N = 30) (hB : B = 16) (hT : T = 19) (hNeither : Neither = 2) : 
  B + T - (N - Neither) = 7 :=
by
  sorry

end members_who_play_both_l755_755131


namespace length_of_EF_l755_755481

theorem length_of_EF {A B C D E F : Type*} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup F]
  (AB : ℝ) (BC : ℝ) (rectangle_ABCD : true) (point_B_coincides_with_D : true) 
  (pentagon_ABEFC : true) :
  AB = 4 ∧ BC = 8 → segment_length EF = 4 * sqrt 5 := 
by
  sorry

end length_of_EF_l755_755481


namespace instantaneous_velocity_at_1_l755_755174

def position_function (t : ℝ) : ℝ := 2 * t^2 + 3

def velocity_function (t : ℝ) : ℝ := (deriv position_function) t

theorem instantaneous_velocity_at_1 : velocity_function 1 = 4 :=
by sorry

end instantaneous_velocity_at_1_l755_755174


namespace fraction_for_repeating_56_l755_755267

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755267


namespace proof_equilateral_triangles_l755_755142

variables {A B C D E F E1 F1 : Point} -- Define all the points as variables

-- Define the conditions as hypotheses
hypothesis (H1 : IsRectangle A B C D)
hypothesis (H2 : OnLine E B C)
hypothesis (H3 : OnLine F D C)
hypothesis (H4 : Midpoint E1 A E)
hypothesis (H5 : Midpoint F1 A F)
hypothesis (H6 : EquilateralTriangle A E F)

-- Define what needs to be proved
theorem proof_equilateral_triangles :
  EquilateralTriangle D E1 C ∧ EquilateralTriangle B F1 C := 
sorry

end proof_equilateral_triangles_l755_755142


namespace work_completion_time_l755_755152

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 5) (hC : C = 1 / 20) :
  1 / (A + B + C) = 2 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l755_755152


namespace max_writing_instruments_in_fullest_case_l755_755076

def pencil_cases := 3
def total_pencils := 15
def total_pens := 14

def min_pencils_per_case := 4
def min_pens_per_case := 2

def condition_pencils_geq_pens (pencils_per_case : Nat) (pens_per_case : Nat) : Prop :=
  pencils_per_case >= pens_per_case

def total_items_case_max (cases : Fin pencil_cases → Nat × Nat) : Nat :=
  Fin.maximum (λ i => (cases i).1 + (cases i).2)

theorem max_writing_instruments_in_fullest_case :
  ∃ (cases : Fin pencil_cases → Nat × Nat),
    (∑ i, (cases i).1 = total_pencils) ∧
    (∑ i, (cases i).2 = total_pens) ∧
    (∀ i, (cases i).1 >= min_pencils_per_case) ∧
    (∀ i, (cases i).2 >= min_pens_per_case) ∧
    (∀ i, condition_pencils_geq_pens (cases i).1 (cases i).2) ∧
    total_items_case_max cases = 14 :=
sorry

end max_writing_instruments_in_fullest_case_l755_755076


namespace new_entrance_fee_after_increase_l755_755499

-- Definitions for the conditions
def initial_fee : ℕ := 5
def visits_first_year : ℕ := 12
def total_cost : ℕ := 116
def visits_per_year_after_increase : ℕ := 4
def years_after_increase : ℕ := 2

-- The question and answer as a Lean theorem
theorem new_entrance_fee_after_increase : 
  let cost_first_year := initial_fee * visits_first_year,
      cost_next_two_years := total_cost - cost_first_year,
      total_visits_next_two_years := visits_per_year_after_increase * years_after_increase,
      new_fee := cost_next_two_years / total_visits_next_two_years in
  new_fee = 7 :=
by
  sorry

end new_entrance_fee_after_increase_l755_755499


namespace pizza_diameter_increase_l755_755688

theorem pizza_diameter_increase :
  ∀ (d D : ℝ), 
    (D / d)^2 = 1.96 → D = 1.4 * d := by
  sorry

end pizza_diameter_increase_l755_755688


namespace recurring_decimal_to_fraction_l755_755319

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755319


namespace repeating_decimal_is_fraction_l755_755296

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755296


namespace focal_length_hyperbola_l755_755140

theorem focal_length_hyperbola : 
  (c : ℝ) (h : c = Real.sqrt(7 + 3)) → 2 * c = 2 * Real.sqrt 10 := 
by 
  intros c h 
  rw h 
  exact eq.refl _

end focal_length_hyperbola_l755_755140


namespace short_bingo_possibilities_l755_755476

theorem short_bingo_possibilities :
  let possibilities := {s : Finset ℕ // s.card = 5 ∧ ∑ i in s, i % 5 = 0}
  (∃ s : possibilities, s.val ⊆ Finset.range 16) ∧
  Finset.card {s : possibilities // s.val ⊆ Finset.range 16} = 72072 :=
  by sorry

end short_bingo_possibilities_l755_755476


namespace repeating_decimal_is_fraction_l755_755292

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755292


namespace family_together_arrangements_l755_755055

theorem family_together_arrangements (n m : ℕ) (h_family : m = 3) (h_total : n = 7) :
  let entities := n - m + 1 in 
  factorial entities * factorial m = 720 :=
by
  sorry

end family_together_arrangements_l755_755055


namespace two_lines_perpendicular_to_same_line_are_parallel_l755_755201

theorem two_lines_perpendicular_to_same_line_are_parallel
    {A B C : Type} [plane A] [line B] [line C]
    (h1 : perpendicular A B)
    (h2 : perpendicular C B) :
    parallel A C :=
by sorry

end two_lines_perpendicular_to_same_line_are_parallel_l755_755201


namespace turtle_count_l755_755985

theorem turtle_count 
  (T : ℕ)
  (H1 : 0.60 * T = (60/100) * T) -- 60% of T are female
  (H2 : 0.40 * T = (40/100) * T) -- 40% of T are male
  (H3 : ∀ t, t ∈ {(1/4) * (0.40 * T)} → t = 0.10 * T) -- 1 in 4 male turtles has stripes
  (H4 : 4 = 0.40 * 0.10 * T) -- 4 are baby striped turtles which form 40% of striped turtles
  : T = 100 :=
by
  sorry

end turtle_count_l755_755985


namespace unique_solution_x2_minus_2a_ln_x_minus_2ax_eq_0_l755_755414

theorem unique_solution_x2_minus_2a_ln_x_minus_2ax_eq_0 (a : ℝ) (h_a : a > 0) :
  (∃! x : ℝ, x > 0 ∧ x^2 - 2*a*log x - 2*a*x = 0) → a = 1/2 :=
by
  sorry

end unique_solution_x2_minus_2a_ln_x_minus_2ax_eq_0_l755_755414


namespace program_outputs_all_divisors_l755_755961

/--
  The function of the program is to output all divisors of \( n \), 
  given the initial conditions and operations in the program.
 -/
theorem program_outputs_all_divisors (n : ℕ) :
  ∀ I : ℕ, (1 ≤ I ∧ I ≤ n) → (∃ S : ℕ, (n % I = 0 ∧ S = I)) :=
by
  sorry

end program_outputs_all_divisors_l755_755961


namespace fraction_of_repeating_decimal_l755_755331

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755331


namespace domain_f_l755_755097

def f (x : ℝ) := Real.sqrt (x - 1) + Real.sqrt (Real.sqrt (Real.sqrt (x - 5)))

theorem domain_f :
  {x : ℝ | ∃ y : ℝ, y = f x} = {x : ℝ | x ≥ 5} :=
sorry

end domain_f_l755_755097


namespace sum_of_squares_is_128_l755_755967

-- Define the setup conditions of the problem
variables (a : ℝ)
def consecutive_even_integers := (a-2, a, a+2)
def sum_of_integers := (a-2) + a + (a+2)
def product_of_integers := (a-2) * a * (a+2)

-- Problem conditions
axiom product_equals_12_times_sum : product_of_integers a = 12 * sum_of_integers a

-- The proof problem: proving the sum of the squares is 128
theorem sum_of_squares_is_128 (a : ℝ) (h : product_of_integers a = 12 * sum_of_integers a) :
  (a-2)^2 + a^2 + (a+2)^2 = 128 :=
begin
  sorry
end

end sum_of_squares_is_128_l755_755967


namespace largest_last_digit_of_sequence_l755_755046

theorem largest_last_digit_of_sequence :
  ∃ (s : String), s.length = 2050 ∧ s.get 0 = '2' ∧ 
  (∀ i, 0 ≤ i ∧ i < 2049 → (let n := (String.toNat (s.get i) * 10 + String.toNat (s.get (i + 1))) in n % 17 = 0 ∨ n % 29 = 0)) ∧
  (String.toNat (s.get 2049) = 8) :=
sorry

end largest_last_digit_of_sequence_l755_755046


namespace toy_blocks_total_l755_755909

theorem toy_blocks_total :
  let stack1 := 5 in
  let stack2 := stack1 + 2 in
  let stack3 := stack2 - 5 in
  let stack4 := stack3 + 5 in
  stack1 + stack2 + stack3 + stack4 = 21 :=
by
  sorry

end toy_blocks_total_l755_755909


namespace coffee_mixture_l755_755157

theorem coffee_mixture (amount_cheaper amount_expensive : ℕ)
  (price_cheaper price_expensive price_mixture : ℝ) : 
  amount_cheaper = 75 → 
  amount_expensive = 75 → 
  price_cheaper = 9 → 
  price_expensive = 12 → 
  price_mixture = 11.25 → 
  (amount_cheaper + amount_expensive = 150) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2]
  exact rfl

end coffee_mixture_l755_755157


namespace ship_speed_in_still_water_l755_755665

theorem ship_speed_in_still_water
  (x y : ℝ)
  (h1: x + y = 32)
  (h2: x - y = 28)
  (h3: x > y) : 
  x = 30 := 
sorry

end ship_speed_in_still_water_l755_755665


namespace at_least_six_consecutive_heads_l755_755167

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end at_least_six_consecutive_heads_l755_755167


namespace repeating_decimal_is_fraction_l755_755298

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755298


namespace repeating_decimal_as_fraction_l755_755303

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755303


namespace polygon_sides_l755_755794

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l755_755794


namespace fraction_for_repeating_56_l755_755260

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755260


namespace sample_size_sufficiency_l755_755204

noncomputable def needed_sample_size (p q ε P : ℝ) : ℕ :=
  let z := 2.96 in -- Corresponds to Φ^(-1) (0.4985)
  let n := (z / ε) ^ 2 * p * q in
  Nat.ceil n

theorem sample_size_sufficiency (p q ε P : ℝ) (hp : p = 0.85) (hq : q = 1 - p) (hε : ε = 0.01) (hP : P = 0.997) :
  needed_sample_size p q ε P = 11171 :=
by
  sorry

end sample_size_sufficiency_l755_755204


namespace repeating_fraction_equality_l755_755356

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755356


namespace length_of_AB_eq_circles_l755_755412

theorem length_of_AB_eq_circles :
  let C1 : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 9}
  let C2 : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + 2 * p.2 - 3 = 0}
  ∃ A B : ℝ × ℝ,
    (A ∈ C1 ∧ A ∈ C2) ∧ (B ∈ C1 ∧ B ∈ C2) ∧
    dist A B = 12 * Real.sqrt 5 / 5 :=
sorry

end length_of_AB_eq_circles_l755_755412


namespace probability_at_least_four_same_face_l755_755731

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l755_755731


namespace length_EF_l755_755480

-- Define the rectangle ABCD with properties
structure Rectangle :=
  (A B C D : Point)
  (AB BC CD DA : ℝ) -- Sides of the rectangle

axioms
  (rect : Rectangle)
  (hAB : rect.AB = 4)
  (hBC : rect.BC = 8)
  (hCD : rect.CD = 4)
  (hDA : rect.DA = 8)
  (h_fold : coincides rect.B rect.D)

-- Define the pentagon ABEFC with points
structure Pentagon :=
  (A B E F C : Point)

axioms
  (pent : Pentagon)
  (h_eq : pent.A = rect.A ∧ pent.B = rect.B ∧ pent.C = rect.C)
  (h_BE_DC : distance pent.B pent.E = rect.CD)
  (h_AF_CB : distance pent.A pent.F = rect.BC)

-- The length of segment EF after folding, to be proven as √10
theorem length_EF : distance pent.E pent.F = √10 := by
  sorry

end length_EF_l755_755480


namespace translated_line_correct_l755_755965

theorem translated_line_correct : ∀ (x : ℝ), (let y_1 := -2 * x + 1 in y_1 - 4 = -2 * x - 3) := 
by
  intros x
  let y_1 := -2 * x + 1
  show y_1 - 4 = -2 * x - 3
  sorry

end translated_line_correct_l755_755965


namespace part_one_part_two_l755_755520

noncomputable def f (x : ℝ) : ℝ := log10 ((2 / (x + 1)) - 1)

def A : set ℝ := {x | (x - 1) / (x + 1) < 0}

def g (x : ℝ) (a : ℝ) : ℝ := sqrt (1 - a^2 - 2 * a * x - x^2)

def B (a : ℝ) : set ℝ := {x | 1 - a^2 - 2 * a * x - x^2 ≥ 0}

theorem part_one : f (1 / 2013) + f (-1 / 2013) = 0 := sorry

theorem part_two (a : ℝ) : (a ≥ 2 → A ∩ B a = ∅) ∧ (¬(∀ a, A ∩ B a = ∅ → a ≥ 2)) := sorry

end part_one_part_two_l755_755520


namespace inverse_matrix_of_eigenvalue_eigenvector_l755_755391

open Matrix

noncomputable def M (a c : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![a, 2],
  ![c, 1]
]

def eigenvector : Vector ℝ 2 := ![1, 1]

theorem inverse_matrix_of_eigenvalue_eigenvector {a c : ℝ}
  (h_eigen : M a c ⬝ eigenvector = 3 • eigenvector) :
  inverse (M a c) = ![
    ![-1/3, 2/3],
    ![2/3, -1/3]
  ] :=
sorry

end inverse_matrix_of_eigenvalue_eigenvector_l755_755391


namespace expression_for_an_l755_755398

noncomputable def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  2 + (n - 1) * d

theorem expression_for_an (d : ℕ) (n : ℕ) 
  (h1 : d > 0)
  (h2 : (arithmetic_sequence d 1) = 2)
  (h3 : (arithmetic_sequence d 1) < (arithmetic_sequence d 2))
  (h4 : (arithmetic_sequence d 2)^2 = 2 * (arithmetic_sequence d 4)) :
  arithmetic_sequence d n = 2 * n := sorry

end expression_for_an_l755_755398


namespace repeating_fraction_equality_l755_755348

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755348


namespace S6_eq_12_l755_755470

open Real

-- Define the geometric sequence and relevant sums.
variable (a : ℕ → ℝ) (r : ℝ) (S_n : ℕ → ℝ)
hypothesis geom_seq : ∀ (n : ℕ), a (n + 1) = r * a n
hypothesis positive_terms : ∀ (n : ℕ), a n > 0
hypothesis sum_def : ∀ (n : ℕ), S_n n = ∑ i in Finset.range n, a i

-- The given conditions
hypothesis S3_eq_3 : S_n 3 = 3
hypothesis S9_eq_39 : S_n 9 = 39

-- The statement to prove
theorem S6_eq_12 : S_n 6 = 12 :=
by
  sorry

end S6_eq_12_l755_755470


namespace average_first_15_even_numbers_l755_755095

theorem average_first_15_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30) / 15 = 16 :=
by 
  sorry

end average_first_15_even_numbers_l755_755095


namespace relationship_l755_755781

variable (α : ℝ) (hα1 : α ∈ Ioo (π / 4) (π / 2))
def a (hα1 : α ∈ Ioo (π / 4) (π / 2)) : ℝ := (Real.sin α)^(Real.sin α)
def b (hα1 : α ∈ Ioo (π / 4) (π / 2)) : ℝ := (Real.cos α)^(Real.sin α)
def c (hα1 : α ∈ Ioo (π / 4) (π / 2)) : ℝ := (Real.tan α)^(Real.sin α)

theorem relationship (hα1 : α ∈ Ioo (π / 4) (π / 2)) : let a' := a α hα1; let b' := b α hα1; let c' := c α hα1 in b' < a' ∧ a' < c' := 
by
  sorry

end relationship_l755_755781


namespace number_of_houses_built_l755_755083

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end number_of_houses_built_l755_755083


namespace product_of_x_and_y_l755_755861

variables (EF FG GH HE : ℕ) (x y : ℕ)

theorem product_of_x_and_y (h1: EF = 42) (h2: FG = 4 * y^3) (h3: GH = 2 * x + 10) (h4: HE = 32) (h5: EF = GH) (h6: FG = HE) :
  x * y = 32 :=
by
  sorry

end product_of_x_and_y_l755_755861


namespace one_sixths_in_fraction_l755_755445

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end one_sixths_in_fraction_l755_755445


namespace circles_tangent_intersection_distance_l755_755989

theorem circles_tangent_intersection_distance {A B E I : Type} 
  (rA rB : ℝ) (dAB : ℝ)
  (rA_eq : rA = 2) (rB_eq : rB = 3) (dAB_eq : dAB = 10)
  (E_eq : E = intersection_of_common_external_tangents circle_A circle_B)
  (I_eq : I = intersection_of_common_internal_tangents circle_A circle_B)
  : distance E I = 24 := 
sorry

end circles_tangent_intersection_distance_l755_755989


namespace smallest_b_factors_l755_755375

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end smallest_b_factors_l755_755375


namespace diagonals_bisect_in_rhombus_l755_755991

axiom Rhombus : Type
axiom Parallelogram : Type

axiom isParallelogram : Rhombus → Parallelogram
axiom diagonalsBisectEachOther : Parallelogram → Prop

theorem diagonals_bisect_in_rhombus (R : Rhombus) :
  ∀ (P : Parallelogram), isParallelogram R = P → diagonalsBisectEachOther P → diagonalsBisectEachOther (isParallelogram R) :=
by
  sorry

end diagonals_bisect_in_rhombus_l755_755991


namespace tom_first_part_speed_l755_755088

theorem tom_first_part_speed 
  (total_distance : ℕ)
  (distance_first_part : ℕ)
  (speed_second_part : ℕ)
  (average_speed : ℕ)
  (total_time : ℕ)
  (distance_remaining : ℕ)
  (T2 : ℕ)
  (v : ℕ) :
  total_distance = 80 →
  distance_first_part = 30 →
  speed_second_part = 50 →
  average_speed = 40 →
  total_time = 2 →
  distance_remaining = 50 →
  T2 = 1 →
  total_time = distance_first_part / v + T2 →
  v = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, we need to prove that v = 30 given the above conditions.
  sorry

end tom_first_part_speed_l755_755088


namespace tan_cos_proof_l755_755407

noncomputable def θ : ℝ := ... -- Assume θ is in the interval (0, π/4)
noncomputable def γ : ℝ := ... -- Assume γ is in the interval (π/4, π/2)

theorem tan_cos_proof
  (θ ∈ Ioo (0 : ℝ) (π / 4))
  (γ ∈ Ioo (π / 4) (π / 2))
  (H : ∀ (α β : ℝ), sin (α + γ) + sin (γ - β) = sin θ * (sin α - sin β) + cos θ * (cos α + cos β))
  : (tan θ * tan γ + cos (θ - γ)) / (sin (θ + (π / 4)))^2 = 2 := 
begin
  sorry
end

end tan_cos_proof_l755_755407


namespace time_to_fill_tank_l755_755195

def rate_X : ℝ := sorry
def rate_Y : ℝ := sorry
def rate_Z : ℝ := sorry
def T : ℝ := 1  -- Assuming T as 1 unit tank for simplification

# Define conditions based on problem statement
def condition1 : Prop := (rate_X + rate_Y = T / 3)
def condition2 : Prop := (rate_X + rate_Z = T / 4)
def condition3 : Prop := (rate_Y + rate_Z = T / 5)

# Combining all conditions
axiom assumptions : condition1 ∧ condition2 ∧ condition3

theorem time_to_fill_tank :
  assumptions →
  (rate_X + rate_Y + rate_Z = 47 / 120) :=
by
  intros
  sorry

end time_to_fill_tank_l755_755195


namespace repeating_decimal_as_fraction_l755_755311

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755311


namespace inverse_negation_l755_755567

theorem inverse_negation :
  (∀ x : ℝ, x ≥ 3 → x < 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ ¬ (x < 3)) :=
by
  sorry

end inverse_negation_l755_755567


namespace spelling_errors_better_l755_755234

theorem spelling_errors_better :
  let w := "better"
  let letters := ['b', 'e', 't', 't', 'e', 'r']
  (∃ perms, perms = List.permutations letters ∧
      ∀ p ∈ perms, p ≠ "better" ∧ length perms = 30 * 6) →
  (length perms - 1 = 179) :=
by
  sorry

end spelling_errors_better_l755_755234


namespace tangent_sum_simplified_l755_755541

theorem tangent_sum_simplified :
  tan (π / 12) + tan (5 * π / 12) = 2 * csc (π / 12) := 
sorry

end tangent_sum_simplified_l755_755541


namespace continuity_at_x_0_l755_755629

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end continuity_at_x_0_l755_755629


namespace cube_root_simplification_l755_755374

theorem cube_root_simplification (x : ℝ) :
  (∛(x^3 - 3 * x^2 + 3 * x - 1) + ∛(x^3 + 3 * x^2 + 3 * x + 1)) = 2 * x :=
by
  sorry

end cube_root_simplification_l755_755374


namespace four_digit_numbers_with_thousands_digit_3_l755_755441

open Nat

theorem four_digit_numbers_with_thousands_digit_3 : 
  ∃ (N : ℕ), N = 1000 ∧ ∀ n, 3000 ≤ n ∧ n < 4000 → 3000 ≤ n ∧ n < 4000 :=
begin
  sorry
end

end four_digit_numbers_with_thousands_digit_3_l755_755441


namespace possible_values_of_f_zero_l755_755048

theorem possible_values_of_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x * f y) :
  f 0 = 0 ∨ f 0 = 1 :=
by
  sorry

end possible_values_of_f_zero_l755_755048


namespace palm_trees_in_forest_l755_755651

variable (F D : ℕ)

theorem palm_trees_in_forest 
  (h1 : D = 2 * F / 5)
  (h2 : D + F = 7000) :
  F = 5000 := by
  sorry

end palm_trees_in_forest_l755_755651


namespace income_to_expenditure_ratio_l755_755052

theorem income_to_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 4000) (hSavings : S = I - E) : I / E = 5 / 3 := by
  -- To prove: I / E = 5 / 3 given hI, hS, and hSavings
  sorry

end income_to_expenditure_ratio_l755_755052


namespace problem_statement_l755_755768

noncomputable def a (n : ℕ) : ℕ := n^2 + 2 * n
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24
def S (n : ℕ) : ℝ := (Finset.sum (Finset.range n) (λ k, 1 / (a (k + 1) : ℝ)))

theorem problem_statement 
  (h1 : ∃ r1 r2 : ℝ, f r1 = 0 ∧ f r2 = 0 ∧ r1 < r2 ∧ a 1 = r1 ∧ a 2 = r2)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n + 3) 
  : (∀ n : ℕ, a n = n^2 + 2 * n) ∧ (∀ n : ℕ, S n < 3 / 4) :=
by
  sorry

end problem_statement_l755_755768


namespace average_of_remaining_two_numbers_l755_755948

theorem average_of_remaining_two_numbers (A B C D E : ℝ) 
  (h1 : A + B + C + D + E = 50) 
  (h2 : A + B + C = 12) : 
  (D + E) / 2 = 19 :=
by
  sorry

end average_of_remaining_two_numbers_l755_755948


namespace increasing_function_inequality_l755_755406

-- Defining f as an increasing function on the reals
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

-- Problem statement
theorem increasing_function_inequality {f : ℝ → ℝ} (h_inc : is_increasing f) (a b : ℝ)
  (h : a + b > 0) : f(a) + f(b) > f(-a) + f(-b) :=
by {
  sorry
}

end increasing_function_inequality_l755_755406


namespace distance_from_center_to_plane_l755_755066

theorem distance_from_center_to_plane (P Q R S : Point)
  (h1 : Sphere S 25 P)
  (h2 : Sphere S 25 Q)
  (h3 : Sphere S 25 R)
  (h8 : dist P Q = 15)
  (h9 : dist Q R = 20)
  (h10 : dist R P = 25) :
  let x := 25 in
  let y := 3 in
  let z := 2 in
  x + y + z = 30 := by
  sorry

end distance_from_center_to_plane_l755_755066


namespace fraction_for_repeating_56_l755_755261

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755261


namespace complement_intersection_l755_755822

open Set

variable {R : Type} [LinearOrder R]

def A : Set R := {x | 3 ≤ x ∧ x < 7}
def B : Set R := {x | 2 < x ∧ x < 10}

theorem complement_intersection (x : R) :
  x ∈ (Aᶜ ∩ B) ↔ (2 < x ∧ x < 3 ∨ 7 ≤ x ∧ x < 10) :=
by
  sorry

end complement_intersection_l755_755822


namespace functional_eq_l755_755252

noncomputable def c : ℝ := sorry -- "c" is a constant

def f (x : ℝ) : ℝ := (sin x + cos x) / 2
def g (x : ℝ) : ℝ := (sin x - cos x) / 2 + c

theorem functional_eq (x y : ℝ) : 
  f x + f y + g x - g y = sin x + cos y :=
by 
  sorry

end functional_eq_l755_755252


namespace cos_angle_relation_l755_755789

theorem cos_angle_relation (α β : ℝ) 
  (h_base_lateral : true) -- Dihedral angle between base and lateral face is α
  (h_adjacent_lateral : true) -- Dihedral angle between adjacent lateral faces is β
  : cos β = -cos(α)^2 :=
sorry

end cos_angle_relation_l755_755789


namespace circle_with_focus_of_parabola_tangent_to_directrix_l755_755727

-- Define the parabola and its properties
def parabola_focus (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def parabola_directrix (a : ℝ) : ℝ :=
  -a / 4

def radius_of_circle (focus_distance : ℝ) : ℝ :=
  focus_distance

-- The standard form of the equation of the circle
def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2

-- The main problem statement
theorem circle_with_focus_of_parabola_tangent_to_directrix :
  (circle_equation (1, 0) 2 ↔ (∀ x y: ℝ, (x - 1) ^ 2 + y ^ 2 = 4)) :=
sorry

end circle_with_focus_of_parabola_tangent_to_directrix_l755_755727


namespace mark_tip_percentage_l755_755526

-- Definitions
def check_amount : ℝ := 200
def friend_contribution : ℝ := 10
def mark_contribution : ℝ := 30

-- Theorem
theorem mark_tip_percentage : 
  ((friend_contribution + mark_contribution) / check_amount) * 100 = 20 := 
by 
  -- Adding the numbers
  let total_tip := friend_contribution + mark_contribution
  let percentage_tip := (total_tip / check_amount) * 100
  -- Proving the percentage
  have h1 : total_tip = 40 := rfl
  have h2 : total_tip / check_amount = 40 / 200 := by rw h1
  have h3 : 40 / 200 = 0.2 := by norm_num
  have h4 : (0.2 : ℝ) * 100 = 20 := by norm_num
  show percentage_tip = 20, by rw [h2, h3, h4]

end mark_tip_percentage_l755_755526


namespace solve_for_y_l755_755933

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l755_755933


namespace tan_squared_gamma_eq_tan_alpha_mul_tan_beta_l755_755779

theorem tan_squared_gamma_eq_tan_alpha_mul_tan_beta
  (α β γ : ℝ)
  (h : (sin γ)^2 / (sin α)^2 = 1 - (tan (α - β)) / (tan α)) :
  (tan γ)^2 = (tan α) * (tan β) :=
by
  sorry

end tan_squared_gamma_eq_tan_alpha_mul_tan_beta_l755_755779


namespace circle_standard_equation_l755_755488

/-
Given points A (1,3) and B (4,6) and a line equation x - 2y - 1 = 0,
prove that the standard equation of a circle passing through
points A and B with its center on the given line is (x - 5)^2 + (y - 2)^2 = 17.
-/

def Point := (ℝ × ℝ)

def circle_eq (h k r : ℝ) : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, (x - h)^2 + (y - k)^2 = r

def on_line (a b c : ℝ) : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, a * x + b * y + c = 0

theorem circle_standard_equation 
  (A B : Point)
  (L : ℝ × ℝ × ℝ)
  (center : Point)
  (r : ℝ)
  (h k : ℝ)
  (radius_squared : ℝ)
  (A_eq : A = (1, 3))
  (B_eq : B = (4, 6))
  (L_eq : L = (1, -2, -1))
  (center_eq : center = (5, 2))
  (radius_squared_eq : radius_squared = 17)
  (r_eq : r^2 = radius_squared) :
  on_line L.1 L.2 L.3 center ∧ circle_eq h k r A ∧ circle_eq h k r B →
  circle_eq 5 2 17 (λ (p : Point), Prop) := 
sorry

end circle_standard_equation_l755_755488


namespace fraction_of_repeating_decimal_l755_755324

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755324


namespace area_triangles_equal_l755_755135

-- Definitions for the points, triangles, and congruence
variables {α : Type} [LinearOrderedField α]
variables {ABC A'B'C' A_1B_1C_1 A_2B_2C_2 : α}

-- Conditions
def congruent_tris (ABC A'B'C' : α) : Prop := true -- placeholder for congruence definition
def points_on_extensions_ABC (A_1 B_1 C_1 ABC : α) : Prop := true -- placeholder
def points_on_extensions_A'B'C' (A_2 B_2 C_2 A'B'C' : α) : Prop := true -- placeholder

-- Mathematical statement
theorem area_triangles_equal
  (h1 : congruent_tris ABC A'B'C')
  (h2 : points_on_extensions_ABC A_1 B_1 C_1 ABC)
  (h3 : points_on_extensions_A'B'C' A_2 B_2 C_2 A'B'C') :
  area A_1B_1C_1 = area A_2B_2C_2 := 
sorry

end area_triangles_equal_l755_755135


namespace sufficient_not_necessary_phi_pi_l755_755139

theorem sufficient_not_necessary_phi_pi 
  : ∀ φ, (φ = π → (∃ x, (x = 0 ∧ sin (2 * x + φ) = 0))) 
        ∧ (∃ φ, (sin φ = 0 ∧ φ ≠ π) → (∃ x, (x = 0 ∧ sin (2 * x + φ) = 0))) :=
by sorry

end sufficient_not_necessary_phi_pi_l755_755139


namespace part_a_equal_areas_l755_755906

-- Define the triangle and points M, N, P
variables {A B C M N P M1 N1 P1 : Type*}
variables [triangle A B C] [on_side M A B] [on_side N B C] [on_side P A C]

-- Define that M1, N1, and P1 are symmetrical to M, N, and P with respect to midpoints of corresponding sides
axiom symmetrical_points_to_midpoints (M1 N1 P1 : Type*) : symmetrical_to_midpoints M N P A B C M1 N1 P1

-- Prove areas are equal for part (a)
theorem part_a_equal_areas : area (M N P) = area (M1 N1 P1) :=
sorry

end part_a_equal_areas_l755_755906


namespace rectangle_sides_l755_755800

theorem rectangle_sides (S d : ℝ) (a b : ℝ) : 
  a = Real.sqrt (S + d^2 / 4) + d / 2 ∧ 
  b = Real.sqrt (S + d^2 / 4) - d / 2 →
  S = a * b ∧ d = a - b :=
by
  -- definitions and conditions will be used here in the proofs
  sorry

end rectangle_sides_l755_755800


namespace result_of_subtraction_l755_755638

theorem result_of_subtraction (N : ℝ) (h1 : N = 100) : 0.80 * N - 20 = 60 :=
by
  sorry

end result_of_subtraction_l755_755638


namespace spring_stretch_150N_l755_755670

-- Definitions for the conditions
def spring_stretch (weight : ℕ) : ℕ :=
  if weight = 100 then 20 else sorry

-- The theorem to prove
theorem spring_stretch_150N : spring_stretch 150 = 30 := by
  sorry

end spring_stretch_150N_l755_755670


namespace hyperbola_1_equation_hyperbola_2_equation_l755_755728

section hyperbolas

noncomputable def hyperbola_1 : Prop :=
  let e := Ellipse.mk (0, 0) 4 5 0
  let h := Hyperbola.mk (0, 0) (Real.sqrt 5) 2 (0, 3)
  (Hyperbola.mirror_x h).passes_through ⟨-2, Real.sqrt 10⟩

-- Prove that the required hyperbola equation is true
theorem hyperbola_1_equation : hyperbola_1 :=
  sorry

noncomputable def hyperbola_2 : Prop :=
  let a := Pair.mk (Line.mk 1 2) 0
  let h := Hyperbola.mk (0, 0) 2 (Real.sqrt 3) (0, Real.sqrt 3)
  h.passes_through ⟨2, 2⟩

-- Prove that the required hyperbola equation is true
theorem hyperbola_2_equation : hyperbola_2 :=
  sorry

end hyperbolas

end hyperbola_1_equation_hyperbola_2_equation_l755_755728


namespace evaluate_polynomial_at_4_l755_755245

-- Define the polynomial function
def polynomial (x : ℕ) : ℕ := x^4 + x^3 + x^2 + x + 1

-- Statement that evaluates the polynomial at x = 4 to be 341
theorem evaluate_polynomial_at_4 : polynomial 4 = 341 := 
by
  simp [polynomial]
  norm_num
  triv
  sorry

end evaluate_polynomial_at_4_l755_755245


namespace grouping_equal_products_l755_755237

def group1 : List Nat := [12, 42, 95, 143]
def group2 : List Nat := [30, 44, 57, 91]

def product (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem grouping_equal_products :
  product group1 = product group2 := by
  sorry

end grouping_equal_products_l755_755237


namespace simplify_tan_sum_l755_755543

theorem simplify_tan_sum : tan (π / 12) + tan (5 * π / 12) = 8 := by
  sorry

end simplify_tan_sum_l755_755543


namespace original_price_of_article_l755_755656

theorem original_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (hSP : SP = 374) (hprofit : profit_percent = 0.10) : 
  CP = 340 ↔ SP = CP * (1 + profit_percent) :=
by 
  sorry

end original_price_of_article_l755_755656


namespace recurring_decimal_to_fraction_l755_755322

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755322


namespace fraction_of_repeating_decimal_l755_755330

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755330


namespace square_difference_l755_755106

theorem square_difference :
  let a := 1001
  let b := 999
  a^2 - b^2 = 4000 :=
by
  let a := 1001
  let b := 999
  have h1 : a^2 - b^2 = (a + b) * (a - b), from sorry
  have h2 : a + b = 2000, by sorry
  have h3 : a - b = 2, by sorry
  show a^2 - b^2 = 4000, by sorry

end square_difference_l755_755106


namespace smallest_abs_diff_l755_755832

theorem smallest_abs_diff (a b : ℕ) (h : a > 0 ∧ b > 0 ∧ ab - 5 * a + 2 * b = 102) : 
  ∃ a b, ab - 5 * a + 2 * b = 102 ∧ min (|a - b|) = 1 :=
sorry

end smallest_abs_diff_l755_755832


namespace no_positive_integer_n_l755_755093

def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  if b = 0 then 0 
  else (n % b) + sum_of_digits (n / b) b

theorem no_positive_integer_n (n : ℕ) (h : n > 0) :
  (sum_of_digits (2018 * n + 1337) 10) ≠ sum_of_digits (2018 * n + 1337) 4 + 2018 := 
  sorry

end no_positive_integer_n_l755_755093


namespace domain_of_func_l755_755561

-- Define the function
def func (x : ℝ) : ℝ := Real.tan ((π / 6) * x + π / 3)

-- Define the set of problematic points where the function is undefined
def problem_points : Set ℝ := { x | ∃ k : ℤ, x = 1 + 6 * k }

-- Define the domain of the function
def func_domain : Set ℝ := { x | ¬ (x ∈ problem_points) }

-- Formulate the statement
theorem domain_of_func :
  (SetOf (λ x, func x)).domain = func_domain := 
  sorry

end domain_of_func_l755_755561


namespace polynomial_expansion_l755_755248

theorem polynomial_expansion :
  (7 * X^2 + 5 * X - 3) * (3 * X^3 + 2 * X^2 + 1) = 
  21 * X^5 + 29 * X^4 + X^3 + X^2 + 5 * X - 3 :=
sorry

end polynomial_expansion_l755_755248


namespace alex_paired_with_jamie_probability_l755_755475

theorem alex_paired_with_jamie_probability (n : ℕ) (h : n = 32) :
  (1 : ℚ) / (n - 1) = 1 / 31 :=
by
  rw [h]
  norm_num
  sorry

end alex_paired_with_jamie_probability_l755_755475


namespace length_EF_l755_755479

-- Define the rectangle ABCD with properties
structure Rectangle :=
  (A B C D : Point)
  (AB BC CD DA : ℝ) -- Sides of the rectangle

axioms
  (rect : Rectangle)
  (hAB : rect.AB = 4)
  (hBC : rect.BC = 8)
  (hCD : rect.CD = 4)
  (hDA : rect.DA = 8)
  (h_fold : coincides rect.B rect.D)

-- Define the pentagon ABEFC with points
structure Pentagon :=
  (A B E F C : Point)

axioms
  (pent : Pentagon)
  (h_eq : pent.A = rect.A ∧ pent.B = rect.B ∧ pent.C = rect.C)
  (h_BE_DC : distance pent.B pent.E = rect.CD)
  (h_AF_CB : distance pent.A pent.F = rect.BC)

-- The length of segment EF after folding, to be proven as √10
theorem length_EF : distance pent.E pent.F = √10 := by
  sorry

end length_EF_l755_755479


namespace four_digit_numbers_with_thousands_digit_3_l755_755442

open Nat

theorem four_digit_numbers_with_thousands_digit_3 : 
  ∃ (N : ℕ), N = 1000 ∧ ∀ n, 3000 ≤ n ∧ n < 4000 → 3000 ≤ n ∧ n < 4000 :=
begin
  sorry
end

end four_digit_numbers_with_thousands_digit_3_l755_755442


namespace sum_of_distances_l755_755856

theorem sum_of_distances (A B D : ℝ × ℝ) (hA : A = (15, 0)) (hB : B = (0, 0)) (hD : D = (8, 6)) :
  19 < real.sqrt ((15 - 8)^2 + 6^2) + real.sqrt (8^2 + 6^2) < 20 :=
sorry

end sum_of_distances_l755_755856


namespace repeating_decimal_equiv_fraction_l755_755282

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755282


namespace minimum_a2_minus_b_l755_755418

theorem minimum_a2_minus_b (a b : ℝ) (f : ℝ → ℝ) (h : ∃ x, f x = 0)
  (hf : ∀ x, f x = x^4 + a * x^3 + b * x^2 + a * x + 1) :
  (∀ a b : ℝ, a^2 - b ≥ 1 ∧ (∃ t : ℝ, t = (minimize (a^2 - b))) → a^2 - b = 1) :=
by sorry

end minimum_a2_minus_b_l755_755418


namespace probability_digit3_in_fraction_l755_755533

def repeating_sequence_of_fraction (n d : ℕ) := (7 : ℕ) / 11 = 0.636363...

theorem probability_digit3_in_fraction : 
  ∀ (n d : ℕ),
  repeating_sequence_of_fraction n d →
  (∃ b, b = "63") →
  (repeating_sequence_of_fraction n d .index 1 = 3) →
  (1/2 : ℚ) := 
sorry

end probability_digit3_in_fraction_l755_755533


namespace probability_at_least_four_same_face_l755_755734

theorem probability_at_least_four_same_face :
  let total_outcomes := (2 : ℕ) ^ 5,
      favorable_outcomes := 1 + 1 + (Nat.choose 5 1) + (Nat.choose 5 1),
      probability := favorable_outcomes / total_outcomes in
  probability = (3 : ℚ) / 8 :=
by
  sorry

end probability_at_least_four_same_face_l755_755734


namespace sequence_diff_ge_m_minus_1_l755_755141

variable {α : Type*}

noncomputable def associated_sequence 
(a : ℕ → ℝ) (b : ℕ → ℕ) (m : ℕ) : Prop :=
  (∀ n, 1 ≤ n ∧ n ≤ m - 1 → a (n + 1) - a n > 0) ∧
  (∀ n, b n ∈ ℕ ∧ b (n + 1) ∈ ℕ ∧ b n > 0 ∧ b (n + 1) > 0)

theorem sequence_diff_ge_m_minus_1
  (a b : ℕ → ℝ) (m : ℕ)
  (h : associated_sequence a (λ x, (b x).toNat) m) :
  ∀ n, 1 ≤ n ∧ n ≤ m - 1 → a (n + 1) - a n ≥ m - 1 := sorry

end sequence_diff_ge_m_minus_1_l755_755141


namespace equivalence_of_conditions_l755_755007

-- Conditions definitions
variables {Ω : Type*} {ξ ξₙ : Ω → ℝ}
variable {μ : MeasureTheory.Measure Ω}
variables [MeasureTheory.ProbabilityMeasure μ]
variable (hn : (λ n, MeasureTheory.CondDistrib ξ ξₙ μ.toOuterMeasure μ).ToSeq)
variable (finite_integral : ∀ n, MeasureTheory.Integrable ξₙ μ)

-- Define expectations and convergence
variable h₁ : ∀ n, 0 ≤ ξₙ n
variable h_limit : MeasureTheory.WeakConvergenceInDistribution ξₙ ξ

-- Goal: To prove the equivalence of the conditions
theorem equivalence_of_conditions :
  (tendsto (λ n, MeasureTheory.integral μ (ξₙ n)) at_top (𝓝 (MeasureTheory.integral μ ξ)) ∧ (MeasureTheory.integral μ ξ < ∞)) ↔
  (limsup (λ n, MeasureTheory.integral μ (ξₙ n)) at_top ≤ MeasureTheory.integral μ ξ ∧ (MeasureTheory.integral μ ξ < ∞)) ↔
  (measure_theory.uniform_integrable μ (λ n, ξₙ n) at_top) :=
sorry

end equivalence_of_conditions_l755_755007


namespace length_of_garden_l755_755461

theorem length_of_garden (P B : ℕ) (hP : P = 1800) (hB : B = 400) : 
  ∃ L : ℕ, L = 500 ∧ P = 2 * (L + B) :=
by
  sorry

end length_of_garden_l755_755461


namespace Q_neither_necessary_nor_sufficient_l755_755900

-- Define the propositions P and Q
def PropositionP (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  ∀ x : ℝ, (a1*x^2 + b1*x + c1 > 0) ↔ (a2*x^2 + b2*x + c2 > 0)

def PropositionQ (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 / a2 = b1 / b2) ∧ (b1 / b2 = c1 / c2)

-- The final statement to prove that Q is neither necessary nor sufficient for P
theorem Q_neither_necessary_nor_sufficient (a1 b1 c1 a2 b2 c2 : ℝ) :
  ¬ ((PropositionQ a1 b1 c1 a2 b2 c2) ↔ (PropositionP a1 b1 c1 a2 b2 c2)) := sorry

end Q_neither_necessary_nor_sufficient_l755_755900


namespace empty_seats_after_second_stop_l755_755972

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end empty_seats_after_second_stop_l755_755972


namespace difference_max_min_change_l755_755213

theorem difference_max_min_change :
  ∃ (y_min y_max : ℕ), 
    40 ≤ y_min ∧ y_min ≤ 60 ∧
    40 ≤ y_max ∧ y_max ≤ 60 ∧
    y_max - y_min = 20 :=
begin
  sorry
end

end difference_max_min_change_l755_755213


namespace no_reappearance_except_all_elements_are_one_l755_755636

noncomputable def seq_transform (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (a * b, b * c, c * d, d * a)

theorem no_reappearance_except_all_elements_are_one
  (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ∀ (n : ℕ), seq_transform^n (a, b, c, d) = (a, b, c, d) → a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
by
  sorry

end no_reappearance_except_all_elements_are_one_l755_755636


namespace taras_total_gas_spent_is_180_l755_755944

def trip_duration := 2 -- in days
def gas_stations := 4 -- number of gas stations visited
def gas_prices := [3.0, 3.5, 4.0, 4.5] -- price per gallon at each gas station
def tank_capacity := 12.0 -- tank capacity in gallons

def total_gas_spent : ℝ :=
  (tank_capacity * gas_prices[0]) +
  (tank_capacity * gas_prices[1]) +
  (tank_capacity * gas_prices[2]) +
  (tank_capacity * gas_prices[3])

theorem taras_total_gas_spent_is_180 :
  total_gas_spent = 180 :=
by
  sorry

end taras_total_gas_spent_is_180_l755_755944


namespace one_sixths_in_fraction_l755_755444

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end one_sixths_in_fraction_l755_755444


namespace translated_midpoint_B_G_l755_755065

structure Point (α : Type) := (x : α) (y : α)

def translate (P : Point ℝ) (Δx Δy : ℝ) : Point ℝ :=
  ⟨P.x + Δx, P.y + Δy⟩

def midpoint (P Q : Point ℝ) : Point ℝ :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

theorem translated_midpoint_B_G :
  let B := Point.mk 2 3
      G := Point.mk 6 3
      translation_left := -7
      translation_down := -3
      mid_BG := midpoint B G
      mid_B'G' := translate mid_BG translation_left translation_down
  in mid_B'G' = Point.mk (-3) 0 :=
by {
  sorry
}

end translated_midpoint_B_G_l755_755065


namespace paintings_per_first_four_customers_l755_755089

theorem paintings_per_first_four_customers 
    (total_customers : ℕ)
    (next_customers : ℕ)
    (last_customers : ℕ)
    (total_paintings : ℕ)
    (next_paints_per_customer : ℕ) 
    (last_paints_per_customer : ℕ)
    (total_next_paints : ℕ)
    (total_last_paints : ℕ)
    (total_first_paints : ℕ)
    (first_customers : ℕ)
    (paints_per_first_customer : ℕ) :
    total_customers = 20 ∧
    (next_customers = 12 ∧ next_paints_per_customer = 1 ∧ total_next_paints = next_customers * next_paints_per_customer) ∧
    (last_customers = 4 ∧ last_paints_per_customer = 4 ∧ total_last_paints = last_customers * last_paints_per_customer) ∧
    total_paintings = 36 ∧
    total_first_paints = total_paintings - (total_next_paints + total_last_paints) ∧
    first_customers = 4 ∧
    paints_per_first_customer = total_first_paints / first_customers
    → paints_per_first_customer = 2 := 
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  sorry

end paintings_per_first_four_customers_l755_755089


namespace problem_statement_l755_755766

noncomputable def a (n : ℕ) : ℕ := n^2 + 2 * n
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24
def S (n : ℕ) : ℝ := (Finset.sum (Finset.range n) (λ k, 1 / (a (k + 1) : ℝ)))

theorem problem_statement 
  (h1 : ∃ r1 r2 : ℝ, f r1 = 0 ∧ f r2 = 0 ∧ r1 < r2 ∧ a 1 = r1 ∧ a 2 = r2)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n + 3) 
  : (∀ n : ℕ, a n = n^2 + 2 * n) ∧ (∀ n : ℕ, S n < 3 / 4) :=
by
  sorry

end problem_statement_l755_755766


namespace fraction_of_repeating_decimal_l755_755328

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755328


namespace log_equation_l755_755830

theorem log_equation (k x : ℝ) (h : log k x * log 7 k = 2) : x = 49 := 
by
  sorry

end log_equation_l755_755830


namespace ram_distance_from_base_l755_755530

theorem ram_distance_from_base (map_distance_mountains : ℝ) (actual_distance_mountains : ℝ) (map_distance_ram : ℝ) : 
  map_distance_mountains = 312 →
  actual_distance_mountains = 136 →
  map_distance_ram = 42 →
  actual_distance_ram ≈ 18.31 :=
by
  sorry

end ram_distance_from_base_l755_755530


namespace common_points_system_of_equations_l755_755373

theorem common_points_system_of_equations :
  let eq1 := (x y : ℝ) → (x - y + 3) * (4 * x + y - 5) = 0
  let eq2 := (x y : ℝ) → (x + y - 3) * (3 * x - 4 * y + 6) = 0
  (setOf x ∈ ℝ×ℝ | eq1 x.1 x.2 ∧ eq2 x.1 x.2).toFinset.card = 4 :=
sorry

end common_points_system_of_equations_l755_755373


namespace triangular_number_30_l755_755716

theorem triangular_number_30 : (30 * (30 + 1)) / 2 = 465 :=
by
  sorry

end triangular_number_30_l755_755716


namespace repeating_decimal_eq_fraction_l755_755338

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755338


namespace product_of_fractions_l755_755099

theorem product_of_fractions : (2 / 5) * (3 / 4) = 3 / 10 := 
  sorry

end product_of_fractions_l755_755099


namespace shortest_distance_bug_crawls_l755_755151

def point : Type := ℝ × ℝ

def line (a b c : ℝ) : Type := {p : point // a * p.1 + b * p.2 + c = 0}

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def symmetric_point (p : point) (l : line 1 (-1) 1) : point :=
  let x := p.1 in
  let y := p.2 in
  let a := 1 in
  let b := -1 in
  let c := 1 in
  let d := (x - (b ^ 2 - a ^ 2) * y + 2 * a * c) / (a ^ 2 + b ^ 2) in
  let e := (y - (a ^ 2 - b ^ 2) * x + 2 * b * c) / (a ^ 2 + b ^ 2) in
  (d, e)

theorem shortest_distance_bug_crawls :
  let P := symmetric_point (0, 0) ⟨⟨(0,0), sorry⟩ 
  distance P (1, 1) = 2 :=
by
  sorry

end shortest_distance_bug_crawls_l755_755151


namespace initial_number_of_people_in_group_l755_755951

-- Definitions based on conditions
def avg_weight_increase (n : ℕ) := 3.5 * n
def weight_difference := 68 - 47

-- Prove the initial number of people in the group
theorem initial_number_of_people_in_group : ∃ n : ℕ, avg_weight_increase n = weight_difference ∧ n = 6 :=
by
  sorry

end initial_number_of_people_in_group_l755_755951


namespace integer_roots_condition_l755_755894

noncomputable def has_integer_roots (n : ℕ) : Prop :=
  ∃ x : ℤ, x * x - 4 * x + n = 0

theorem integer_roots_condition (n : ℕ) (h : n > 0) :
  has_integer_roots n ↔ n = 3 ∨ n = 4 :=
by 
  sorry

end integer_roots_condition_l755_755894


namespace trailing_zeros_350_factorial_l755_755062

theorem trailing_zeros_350_factorial : 
  (nat.factorial 350).trailing_zero_count = 86 := 
by 
  sorry

end trailing_zeros_350_factorial_l755_755062


namespace quadratic_eq_pq_zero_l755_755456

-- Defining the context and given condition
def quadratic_real_roots (p q : ℝ) : Prop := 
  ∃ (a b : ℂ), a = 1 + complex.I ∧ b = 1 - complex.I ∧ 
  (∀ (x : ℂ), x^2 + (↑p : ℂ) * x + (↑q : ℂ) = 0 → (x = a ∨ x = b))

-- Stating the theorem to prove
theorem quadratic_eq_pq_zero (p q : ℝ) (h : quadratic_real_roots p q) : p + q = 0 := by
  sorry

end quadratic_eq_pq_zero_l755_755456


namespace second_largest_subtract_smallest_correct_l755_755592

-- Definition of the elements
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Conditions derived from the problem
def smallest_number : ℕ := 10
def second_largest_number : ℕ := 13

-- Lean theorem statement representing the problem
theorem second_largest_subtract_smallest_correct :
  (second_largest_number - smallest_number) = 3 := 
by
  sorry

end second_largest_subtract_smallest_correct_l755_755592


namespace balance_balls_l755_755018

variable (G B Y W P : ℝ)

-- Given conditions
def cond1 : 4 * G = 9 * B := sorry
def cond2 : 3 * Y = 8 * B := sorry
def cond3 : 7 * B = 5 * W := sorry
def cond4 : 4 * P = 10 * B := sorry

-- Theorem we need to prove
theorem balance_balls : 5 * G + 3 * Y + 3 * W + P = 26 * B :=
by
  -- skipping the proof
  sorry

end balance_balls_l755_755018


namespace surface_area_of_removed_columns_l755_755653

def large_cube := (5 : ℕ) -- Defining size of the large cube
def small_cube := (1 : ℕ) -- Defining size of the small cubes

-- Removing central columns from large_cube
structure columns_removed :=
  (x : ℕ := large_cube)
  (y : ℕ := large_cube)
  (z : ℕ := large_cube)

-- Define the computation of surface area after removal
def surface_area_of_resulting_solid : ℕ := 192

-- Proving the surface area of the resulting solid
theorem surface_area_of_removed_columns : 
  ∀ (c : columns_removed), c.x = 5 → c.y = 5 → c.z = 5 → surface_area_of_resulting_solid = 192 := by
  intros,
  sorry

end surface_area_of_removed_columns_l755_755653


namespace fraction_eq_repeating_decimal_l755_755275

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755275


namespace count_four_digit_integers_with_thousands_digit_3_l755_755439

theorem count_four_digit_integers_with_thousands_digit_3 : 
  (finset.card (finset.filter (λ n : ℕ, 3000 ≤ n ∧ n < 4000) (finset.range 10000))) = 1000 :=
by
  sorry

end count_four_digit_integers_with_thousands_digit_3_l755_755439


namespace fraction_for_repeating_56_l755_755258

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755258


namespace frank_columns_l755_755742

theorem frank_columns (people : ℕ) (brownies_per_person : ℕ) (rows : ℕ)
  (h1 : people = 6) (h2 : brownies_per_person = 3) (h3 : rows = 3) : 
  (people * brownies_per_person) / rows = 6 :=
by
  -- Proof goes here
  sorry

end frank_columns_l755_755742


namespace both_p_and_q_are_true_l755_755776

-- Define proposition p
def p : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ log (a + b) = log a + log b

-- Proposition q defined as stated
def q : Prop := ∀ (l₁ l₂ : set (ℝ × ℝ × ℝ)), 
  (¬ ∃ (α β γ : ℝ) (a1 b1 c1 a2 b2 c2 : ℝ),
      ∀ (t1 t2 : ℝ) (p1 : ℝ × ℝ × ℝ) (p2 : ℝ × ℝ × ℝ), 
        -- l1 is a line
        p1 = (α + a1 * t1, β + b1 * t1, γ + c1 * t1) → 
        -- l2 is a line
        p2 = (α + a2 * t2, β + b2 * t2, γ + c2 * t2) →         
        -- both points lies on same plane
        ((a1 * (β + b2 * t2 - β - b1 * t1) = (γ + c1 * t1 - γ - c2 * t2)) ∧ 
         (a1 ≠ 0 ∧ a2 ≠ 0 ∧ b1 ≠ 0 ∧ b2 ≠ 0 ∧ c1 ≠ 0 ∧ c2 ≠ 0))).

-- Combined proposition indicating both p and q are true
theorem both_p_and_q_are_true : p ∧ q := by
  sorry

end both_p_and_q_are_true_l755_755776


namespace rhombus_side_length_l755_755474

-- Define the structure of a Rhombus with properties
structure Rhombus :=
(short_diagonal : ℝ)
(long_diagonal : ℝ)
(side : ℝ)

-- Define the area condition
def area_rhombus (r : Rhombus) : ℝ := (1 / 2) * r.short_diagonal * r.long_diagonal

-- Define the side length condition using the Pythagorean theorem
def side_length_rhombus (r : Rhombus) : ℝ :=
  sqrt ((r.short_diagonal / 2) ^ 2 + (r.long_diagonal / 2) ^ 2)

-- Define a function to express short_diagonal in terms of area
def diagonal_from_area (A : ℝ) : ℝ := sqrt (2 * A / 3)

-- The main theorem statement
theorem rhombus_side_length (A : ℝ) (hA : A > 0):
  ∀ (r : Rhombus),
  r.short_diagonal = diagonal_from_area A →
  r.long_diagonal = 3 * diagonal_from_area A →
  r.side = (sqrt (20 * A / 3)) / 2 :=
begin
  -- provide the structure of proof but leave actual proof as sorry
  intros r hsd hld,
  sorry
end

end rhombus_side_length_l755_755474


namespace chocolate_chip_cookies_l755_755905

theorem chocolate_chip_cookies (baggies : ℕ) (cookies_per_bag : ℕ) (oatmeal_cookies : ℕ) (total_cookies : ℕ) :
  baggies = 6 → cookies_per_bag = 9 → oatmeal_cookies = 41 → total_cookies = baggies * cookies_per_bag → total_cookies - oatmeal_cookies = 13 :=
by
  intro hb hcpb hoc htc
  rw [hb, hcpb, hoc, htc]
  -- We assume all conditions hold and reached the final expression
  sorry

end chocolate_chip_cookies_l755_755905


namespace sin_x_theorem_l755_755453

noncomputable def sin_x (b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) : ℝ :=
  sin (real.atan (3 * b * c / (b ^ 2 - c ^ 2)))

theorem sin_x_theorem (b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (x : ℝ)
  (hx : 0 < x) (hx90 : x < 90) (htan : tan x = 3 * b * c / (b ^ 2 - c ^ 2)) :
  sin x = 3 * b * c / real.sqrt ((b ^ 2 + c ^ 2) ^ 2 + 5 * b ^ 2 * c ^ 2) :=
by
  sorry

end sin_x_theorem_l755_755453


namespace part_I_part_II_l755_755820

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end part_I_part_II_l755_755820


namespace vertex_of_parabola_l755_755718

def parabola_vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2 * a), c - b^2 / (4 * a))

theorem vertex_of_parabola :
  parabola_vertex 2 16 34 = (-4, 2) :=
sorry

end vertex_of_parabola_l755_755718


namespace fraction_eq_repeating_decimal_l755_755362

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755362


namespace compute_value_l755_755005

-- Definitions related to the problem
def circle (center : ℝ × ℝ) (radius : ℝ) := set_of (λ P : ℝ × ℝ, dist P center = radius)

variable (A B P : ℝ × ℝ)
variable (omega1_center omega2_center : ℝ × ℝ)
variable (radius1 radius2 : ℝ)

-- Hypotheses derived from conditions in part a
def omega1 : set (ℝ × ℝ) := circle omega1_center radius1
def omega2 : set (ℝ × ℝ) := circle omega2_center radius2

axiom center_omega2_on_omega1 : omega2_center ∈ omega1
axiom A_in_omega1_and_omega2 : A ∈ omega1 ∧ A ∈ omega2
axiom B_in_omega1_and_omega2 : B ∈ omega1 ∧ B ∈ omega2
axiom P_tangent_intersection : tangent_line omega2 A P ∧ tangent_line omega2 B P

-- The goal: calculate and prove the result
theorem compute_value :
  radius1 = 5 → radius2 = 2 → 100 * 192 + 10 * 6 + 25 = 19285 :=
by
  intros r1 r2,
  have h1 := rfl, -- Placeholder steps; the actual proof would replace this
  sorry

end compute_value_l755_755005


namespace coefficient_of_x_in_expansion_l755_755256

theorem coefficient_of_x_in_expansion : 
  let expr := (1 + x)^2 + (1 + sqrt(x))^3 + (1 + cbrt(x))^4 + (1 + x^(1/4))^5 + (1 + x^(1/5))^6 + 
              (1 + x^(1/6))^7 + (1 + x^(1/7))^8 + (1 + x^(1/8))^9 + (1 + x^(1/9))^10 in
  coefficient_of_x_in expr = 54 :=
by
  sorry

end coefficient_of_x_in_expansion_l755_755256


namespace repeating_fraction_equality_l755_755353

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755353


namespace projection_perpendicular_to_intersection_line_l755_755028

variable {Point : Type} 
noncomputable theory 

structure Plane : Type where
  contains : Point → Prop

structure Line : Type where
  contains : Point → Prop

def intersects (α β : Plane) (l : Line) : Prop :=
  ∀ p : Point, l.contains p → α.contains p ∧ β.contains p

def perpendicular (p : Line) (α : Plane) : Prop :=
  ∀ l' : Line, (∀ p : Point, α.contains p → l'.contains p) → ¬ ∃ q, p.contains q ∧ l'.contains q

def projection (p : Line) (α : Plane) : Line := sorry

def perpendicular_lines (p l : Line) : Prop :=
  ∀ v w : Point, p.contains v → l.contains w → ¬ (v = w)

theorem projection_perpendicular_to_intersection_line
  (α β : Plane) (l : Line) (p : Line) :
  intersects β α l →
  perpendicular p β →
  perpendicular_lines (projection p α) l :=
by {
  sorry,
}

end projection_perpendicular_to_intersection_line_l755_755028


namespace cindy_coins_l755_755220

theorem cindy_coins (n : ℕ) (h : ((factors n).length = 21)) : n = 576 :=
by sorry

end cindy_coins_l755_755220


namespace angle_conversion_l755_755229

theorem angle_conversion :
  (12 * (Real.pi / 180)) = (Real.pi / 15) := by
  sorry

end angle_conversion_l755_755229


namespace solve_for_y_l755_755927

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l755_755927


namespace initial_number_of_persons_l755_755552

theorem initial_number_of_persons (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ)
  (h1 : avg_increase = 2.5) 
  (h2 : old_weight = 75) 
  (h3 : new_weight = 95)
  (h4 : weight_diff = new_weight - old_weight)
  (h5 : weight_diff = avg_increase * n) : n = 8 := 
sorry

end initial_number_of_persons_l755_755552


namespace polynomial_evaluation_x_eq_4_l755_755241

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l755_755241


namespace candy_probability_l755_755148

theorem candy_probability :
  let total_ways := Nat.choose 12 4,
      red_ways := Nat.choose 5 3,
      non_red_ways := Nat.choose 7 1,
      favorable_ways := red_ways * non_red_ways,
      probability := favorable_ways / total_ways
  in probability = 14 / 99 :=
by
  let total_ways := Nat.choose 12 4
  let red_ways := Nat.choose 5 3
  let non_red_ways := 7 -- because the remaining candies need not involve another combination calculation
  let favorable_ways := red_ways * non_red_ways
  let probability := favorable_ways / total_ways
  have : favorable_ways = 70 := by
    rw [Nat.choose, Nat.choose, Nat.choose]
    sorry -- actual steps to equate
  have : total_ways = 495 := by
    rw [Nat.choose]
    sorry -- actual steps to equate
  have : probability = 70 / 495 := 
  by rw [this, this, favorable_ways, total_ways]

  have : 70 / 495 = 14 / 99 := 
  by sorry -- simplifying the fraction

  exact this

end candy_probability_l755_755148


namespace no_circle_intersects_each_side_of_convex_quad_l755_755382

theorem no_circle_intersects_each_side_of_convex_quad (Q : Type) [convex Q] : 
  ¬ ∃ (C : Type) [circle C], ∀ (side : Q → Q), intersects C side 2 :=
sorry

end no_circle_intersects_each_side_of_convex_quad_l755_755382


namespace customer_paid_amount_l755_755064

theorem customer_paid_amount (cost_price : ℝ) (markup_percent : ℝ) (total_amount_paid : ℝ) 
  (h1 : cost_price = 6625) (h2 : markup_percent = 24) : 
  total_amount_paid = cost_price + (markup_percent / 100) * cost_price :=
by
  rw [h1, h2]
  have markup : ℝ := (24 / 100) * 6625
  have total_amount_paid_target : ℝ := 6625 + markup
  have markup_value : markup = 1590 := by
    apply eq.mpr; simp [markup]; norm_num
  rw [markup_value]; simp [total_amount_paid_target]; norm_num;
  exact rfl

# Check to ensure the theorem verifies the expected total amount paid
noncomputable def expected_total_amount_paid : ℝ := 6625 + (24 / 100) * 6625
# expected_total_amount_paid should be 8215
# Eval ns_to_ews_8215 qed

end customer_paid_amount_l755_755064


namespace geometric_sequence_root_l755_755487

theorem geometric_sequence_root (a_n : ℕ → ℝ) (a_3 a_7 : ℝ) 
  (h1 : ∀ n, a_n = a_1 * r ^ (n - 1))
  (h2 : a_3 = (root1_of (x^2 - 4 * x + 3 = 0)))
  (h3 : a_7 = (root2_of (x^2 - 4 * x + 3 = 0))) :
  a_n 5 = ℝ.sqrt 3 := 
sorry

end geometric_sequence_root_l755_755487


namespace compute_sum_of_squares_l755_755517

noncomputable def polyRoots (p : Polynomial ℝ) (a b c : ℝ) :=
  p = Polynomial.C -8 + Polynomial.C 17 * Polynomial.X + Polynomial.C (-15) * Polynomial.X^2 + Polynomial.X^3

theorem compute_sum_of_squares {a b c : ℝ} 
  (h : polyRoots (Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 17 * Polynomial.X - 8) a b c) :
  (a + b) ^ 2 + (b + c) ^ 2 + (c + a) ^ 2 = 416 := 
sorry

end compute_sum_of_squares_l755_755517


namespace minimum_value_f_l755_755522

noncomputable def f (x : ℝ) (f1 f2 : ℝ) : ℝ :=
  f1 * x + f2 / x - 2

theorem minimum_value_f (f1 f2 : ℝ) (h1 : f2 = 2) (h2 : f1 = 3 / 2) :
  ∃ x > 0, f x f1 f2 = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end minimum_value_f_l755_755522


namespace negation_equiv_l755_755061

theorem negation_equiv (x : ℝ) : ¬ (x^2 - 1 < 0) ↔ (x^2 - 1 ≥ 0) :=
by
  sorry

end negation_equiv_l755_755061


namespace surface_area_ratio_l755_755882

-- Defining conditions
variable (V_E V_J : ℝ) (A_E A_J : ℝ)
variable (volume_ratio : V_J = 30 * (Real.sqrt 30) * V_E)

-- Statement to prove
theorem surface_area_ratio (h : V_J = 30 * (Real.sqrt 30) * V_E) :
  A_J = 30 * A_E :=
by
  sorry

end surface_area_ratio_l755_755882


namespace increase_average_l755_755949

variable (total_runs : ℕ) (innings : ℕ) (average : ℕ) (new_runs : ℕ) (x : ℕ)

theorem increase_average (h1 : innings = 10) 
                         (h2 : average = 30) 
                         (h3 : total_runs = average * innings) 
                         (h4 : new_runs = 74) 
                         (h5 : total_runs + new_runs = (average + x) * (innings + 1)) :
    x = 4 := 
sorry

end increase_average_l755_755949


namespace sin_scaled_shifted_increasing_l755_755988

theorem sin_scaled_shifted_increasing :
  ∀ x : ℝ, - π / 3 < x ∧ x < π / 6 → 
  monotone_on (λ x, sin (2 * x + π / 6) + 1) (set.Ioo (- π / 3) (π / 6)) :=
begin
  sorry
end

end sin_scaled_shifted_increasing_l755_755988


namespace find_number_l755_755145

-- Define the conditions
def condition (x : ℝ) : Prop := 0.65 * x = (4/5) * x - 21

-- Prove that given the condition, x is 140.
theorem find_number (x : ℝ) (h : condition x) : x = 140 := by
  sorry

end find_number_l755_755145


namespace cost_of_fencing_per_meter_l755_755568

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end cost_of_fencing_per_meter_l755_755568


namespace fraction_of_repeating_decimal_l755_755325

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755325


namespace problem_statement_l755_755749

theorem problem_statement (α : ℝ) (h : tan α + 1 / tan α = 5 / 2) :
  ∃ (val : ℝ), val = (2 * sin (3 * π - α) ^ 2 - 
  3 * cos (π / 2 + α) * sin (3 * π / 2 - α) + 2)
  ∧ (val = 6 / 5 ∨ val = 12 / 5) :=
sorry

end problem_statement_l755_755749


namespace problem1_problem2_l755_755424

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2 * x - 1
  else if 0 < x ∧ x ≤ 1 then -2 * x + 1
  else 0 -- considering the function is not defined outside the given range

-- Statement to prove that f(f(-1)) = -1
theorem problem1 : f (f (-1)) = -1 :=
by
  sorry

-- Statements to prove the solution set for |f(x)| < 1/2
theorem problem2 : { x : ℝ | |f x| < 1 / 2 } = { x : ℝ | -3/4 < x ∧ x < -1/4 } ∪ { x : ℝ | 1/4 < x ∧ x < 3/4 } :=
by
  sorry

end problem1_problem2_l755_755424


namespace negation_universal_exists_l755_755060

open Classical

theorem negation_universal_exists :
  (¬ ∀ x : ℝ, x > 0 → (x^2 - x + 3 > 0)) ↔ ∃ x : ℝ, x > 0 ∧ (x^2 - x + 3 ≤ 0) :=
by
  sorry

end negation_universal_exists_l755_755060


namespace calculate_revolutions_l755_755694

noncomputable def number_of_revolutions (diameter distance: ℝ) : ℝ :=
  distance / (Real.pi * diameter)

theorem calculate_revolutions :
  number_of_revolutions 10 5280 = 528 / Real.pi :=
by
  sorry

end calculate_revolutions_l755_755694


namespace jack_basket_capacity_is_12_l755_755498

-- Definitions for the conditions
def full_capacity_jack_basket : ℕ := -- J
def full_capacity_jill_basket (J : ℕ) : ℕ := 2 * J
def current_apples_in_jack_basket (J : ℕ) : ℕ := J - 4
def jill_basket_given_current_jack_apples (J : ℕ) : Prop :=
  3 * (J - 4) = 2 * J

-- The statement to be proven
theorem jack_basket_capacity_is_12 : ∃ J : ℕ, 
  jill_basket_given_current_jack_apples J ∧ 
  (full_capacity_jack_basket = J) ∧ 
  (full_capacity_jack_basket = 12) :=
by {
  -- Add the specific assumptions as hypotheses
  sorry
}

end jack_basket_capacity_is_12_l755_755498


namespace smallest_abs_sum_of_products_l755_755509

noncomputable def g (x : ℝ) : ℝ :=
  x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

def roots (f : ℝ → ℝ) : set ℝ := {x | f x = 0}

theorem smallest_abs_sum_of_products :
  let w : set ℝ := roots g in
  w = {-1, -3, 12, 4} → 
  ∃ (a b c d : ℝ), {a, b, c, d} = {-1, -3, 12, 4} ∧ (|a * b + c * d| = 24) :=
by
  sorry

end smallest_abs_sum_of_products_l755_755509


namespace minimize_rental_cost_l755_755674

def travel_agency (x y : ℕ) : ℕ := 1600 * x + 2400 * y

theorem minimize_rental_cost :
    ∃ (x y : ℕ), (x + y ≤ 21) ∧ (y ≤ x + 7) ∧ (36 * x + 60 * y = 900) ∧ 
    (∀ (a b : ℕ), (a + b ≤ 21) ∧ (b ≤ a + 7) ∧ (36 * a + 60 * b = 900) → travel_agency a b ≥ travel_agency x y) ∧
    travel_agency x y = 36800 :=
sorry

end minimize_rental_cost_l755_755674


namespace exercise_l755_755108

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l755_755108


namespace arithmetic_seq_common_difference_l755_755871

theorem arithmetic_seq_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 * a 11 = 6) (h2 : a 4 + a (14) = 5) : 
  d = 1 / 4 ∨ d = -1 / 4 :=
sorry

end arithmetic_seq_common_difference_l755_755871


namespace lily_feeds_puppy_l755_755014

/-- Lily feeds the puppy three times a day for the first two weeks (14 days), and then twice a day
at 1/2 cup per meal for the following two weeks. Including the 1/2 cup fed today, the total amount 
of food eaten by the puppy over the 4 weeks is 25 cups. Prove that the amount of food fed per meal 
three times a day in the first two weeks is 1/4 cup. -/
theorem lily_feeds_puppy :
  ∀ (x : ℚ), 
  (1 / 2) + (14 * 2 * (1 / 2)) + (14 * 3 * x) = 25 →
  x = 1 / 4 := 
begin
  intros x h,
  sorry
end

end lily_feeds_puppy_l755_755014


namespace find_a4_l755_755584

variable (a_1 d : ℝ)

def S (n : ℕ) : ℝ := (n / 2) * (2 * a_1 + (n - 1) * d)

theorem find_a4 (h : 6 * S 5 - 5 * S 3 = 5) : 
  (a_1 + 3 * d) = (1 : ℝ) / 3 := 
  sorry

end find_a4_l755_755584


namespace line_through_intersection_and_point_l755_755410

theorem line_through_intersection_and_point :
  let l : set (ℝ × ℝ) := {p | p.1 + 3 * p.2 - 8 = 0}
  in
  (∀ x y : ℝ, 7 * x + 5 * y - 24 = 0 ∧ x - y = 0 → (x, y) ∈ l) ∧
  (5, 1) ∈ l :=
by
  sorry

end line_through_intersection_and_point_l755_755410


namespace solution_of_r_and_s_l755_755002

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l755_755002


namespace find_line_l1_l755_755788

noncomputable def line_equation (x y : ℝ) : ℝ := x + y

theorem find_line_l1 (c : ℝ) :
  (line_equation x y + c = 0) ∧
  ∃ (d : ℝ), (line_equation x y - 1 = 0) ∧ (|c + 1| / d = sqrt 2 ∧ d = sqrt 2) ↔
  (c = 1 ∨ c = -3) :=
by
  sorry

end find_line_l1_l755_755788


namespace tangent_sum_simplified_l755_755542

theorem tangent_sum_simplified :
  tan (π / 12) + tan (5 * π / 12) = 2 * csc (π / 12) := 
sorry

end tangent_sum_simplified_l755_755542


namespace person_B_performance_less_volatile_than_A_l755_755534

theorem person_B_performance_less_volatile_than_A
  (shots : ℕ)
  (avg_score_A avg_score_B : ℝ)
  (variance_A variance_B : ℝ)
  (shots_eq : shots = 10)
  (avg_score_eq_A : avg_score_A = 9)
  (avg_score_eq_B : avg_score_B = 9)
  (variance_eq_A : variance_A = 1.4)
  (variance_eq_B : variance_B = 0.8) :
  variance_B < variance_A :=
by {
  rw [variance_eq_A, variance_eq_B],
  exact lt_of_eq_of_lt rfl (by norm_num),
}

end person_B_performance_less_volatile_than_A_l755_755534


namespace recurring_decimal_to_fraction_l755_755316

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755316


namespace range_f_l755_755460

def log_op (a b : ℝ) : ℝ :=
if a < b then b else a

def f (x : ℝ) : ℝ :=
log_op (Real.logBase 2 x) (Real.logBase (1/2) x)

theorem range_f : set.range f = set.Ici 0 := 
by
  sorry

end range_f_l755_755460


namespace exponential_expression_is_rational_l755_755503

noncomputable def B (n : ℕ) : ℕ :=
nat.popcount n

noncomputable def S : ℝ :=
∑' n, (B n : ℝ) / (n * (n + 1))

theorem exponential_expression_is_rational :
  Real.exp S = 4 := sorry

end exponential_expression_is_rational_l755_755503


namespace unique_true_proposition_is_B_l755_755680

-- Define the propositions
def prop_A := sqrt 4 = 2
def prop_B := ∀ {α β : Type} (P Q : α → β), ∀ (x1 y1 : α) (x2 y2 : β), (P x1 = x2) ∧ (P y1 = y2) → P x1 = P y2
def prop_C := ∀ (a b : ℝ), (a + b = 180) → (a = b)
def prop_D := ∀ (x : ℝ), x = 0 → (x * x * x = 0 → False)

-- Problem statement: The unique true proposition is Proposition B
theorem unique_true_proposition_is_B : 
  (prop_A = False) ∧ 
  (prop_B = True) ∧ 
  (prop_C = False) ∧ 
  (prop_D = False) := 
sorry

end unique_true_proposition_is_B_l755_755680


namespace fraction_of_repeating_decimal_l755_755327

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755327


namespace remaining_fuel_relationship_l755_755013

-- Define the initial amount of fuel and the consumption rate
def initial_fuel : ℕ := 60
def consumption_rate : ℕ := 8

-- Define the remaining fuel function
def remaining_fuel (t : ℕ) : ℕ := initial_fuel - consumption_rate * t

-- State the theorem
theorem remaining_fuel_relationship (t : ℕ) : 
  remaining_fuel(t) = initial_fuel - consumption_rate * t := by
  sorry

end remaining_fuel_relationship_l755_755013


namespace area_of_triangle_OAB_l755_755037

open Complex

theorem area_of_triangle_OAB
  (z1 z2 : ℂ)
  (h1 : abs z1 = 4)
  (h2 : 4 * z1 ^ 2 - 2 * z1 * z2 + z2 ^ 2 = 0) :
  let area := (1 / 2) * abs z1 * abs z2 * Real.sin (Real.pi / 3) in
  area = 8 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_OAB_l755_755037


namespace eccentricity_of_ellipse_maximum_area_of_triangle_l755_755764

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b) := 
  sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b > 0) (d : ℝ)
  (h1 : d = sqrt (1 - (b^2 / a^2))) (h2 : d = 1 - sqrt(3) / 3 * b) : d = 1/2 :=
by
  sorry

noncomputable def maximum_area_triangle (a b c : ℝ) (h : a = 2 * c) :=
  sqrt(3)

theorem maximum_area_of_triangle (a b c : ℝ) (h : a = 2 * c) (c1 : b^2 = 3 * c^2) (m_f_y : ℝ)
  (h1 : (sqrt 3)^2 / 4 + (sqrt 3 / 2)^2 / 3 = 1) (k : ℝ) 
  (h2 : k = -3/2) (m: ℝ) (h3: m ≠ 0) 
  (N_x : ℝ) (N_y : ℝ)
  (h4 : N_x = - (4 * 3 / (3 + 4 * k^2))) (h5 : N_y = 6 * 3 / (3 + 4*k^2)) :
  maximum_area_triangle a b c h = sqrt(3) :=
by
  sorry

end eccentricity_of_ellipse_maximum_area_of_triangle_l755_755764


namespace cubic_polynomials_l755_755079

theorem cubic_polynomials (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
    (h1 : a - 1/b = r₁ ∧ b - 1/c = r₂ ∧ c - 1/a = r₃)
    (h2 : r₁ + r₂ + r₃ = 5)
    (h3 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = -15)
    (h4 : r₁ * r₂ * r₃ = -3)
    (h5 : a₁ * b₁ * c₁ = 1 + Real.sqrt 2 ∨ a₁ * b₁ * c₁ = 1 - Real.sqrt 2)
    (h6 : a₂ * b₂ * c₂ = 1 + Real.sqrt 2 ∨ a₂ * b₂ * c₂ = 1 - Real.sqrt 2) :
    (-(a₁ * b₁ * c₁))^3 + (-(a₂ * b₂ * c₂))^3 = -14 := sorry

end cubic_polynomials_l755_755079


namespace perpendicular_probability_is_one_fourth_l755_755029

def is_perpendicular (a b : ℤ) : Prop := (-2 * a + b = 0)

def all_pairs : List (ℤ × ℤ) := 
  [(2, 4), (2, 6), (2, 8), 
   (3, 4), (3, 6), (3, 8), 
   (4, 4), (4, 6), (4, 8), 
   (5, 4), (5, 6), (5, 8)]

def perpendicular_pairs : List (ℤ × ℤ) := 
  List.filter (λ (p : ℤ × ℤ), is_perpendicular p.1 p.2) all_pairs

noncomputable def probability_perpendicular : ℚ := 
  (List.length perpendicular_pairs : ℚ) / (List.length all_pairs : ℚ)

theorem perpendicular_probability_is_one_fourth : 
  probability_perpendicular = (1 : ℚ) / 4 := by
  sorry

end perpendicular_probability_is_one_fourth_l755_755029


namespace count_valid_Q_polynomials_l755_755888

theorem count_valid_Q_polynomials :
  let P(x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4)
  ∃ (R : ℝ → ℝ) (Q : ℝ → ℝ), 
    (degree R = 4) ∧ 
    (degree Q = 2) ∧ 
    ∀ x, P (Q x) = P x * R x ∧ 
    (cardinal.mk (set_of (λ Q : (polynomial ℝ), (degree Q = 2 ∧ ∀ x, P (Q x) = P x * R x))) = 250) :=
begin
  sorry
end

end count_valid_Q_polynomials_l755_755888


namespace mean_of_remaining_two_l755_755925

theorem mean_of_remaining_two (a b c d e f : ℕ) (S₁ S₂ S₃ S₄ : list ℕ)
  (h₁ : S₁ = [a, b, c, d, e, f])
  (h₂ : S₂ ⊆ S₁)
  (h₃ : 4 = S₂.length)
  (h₄ : 2035 = (S₂.sum / 4) : ℕ) :
  2078.5 = ((S₁.sum - S₂.sum) / 2 : ℕ) := sorry

end mean_of_remaining_two_l755_755925


namespace fraction_eq_repeating_decimal_l755_755272

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755272


namespace part_I_part_II_l755_755137

noncomputable def a_seq (a : ℕ → ℝ) := a 0 = 1 ∧ ∀ n, a n ≤ a (n + 1)

noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (1 - (a k / a (k + 1))) / (real.sqrt (a (k + 1)))

theorem part_I (a : ℕ → ℝ) (h : a_seq a) (n : ℕ) : 0 ≤ b_seq a n ∧ b_seq a n < 2 := by
  sorry

theorem part_II (c : ℝ) (hc : 0 ≤ c ∧ c < 2) :
  ∃ a : ℕ → ℝ, a_seq a ∧ ∃ᶠ n in filter.at_top, b_seq a n > c := by
  sorry

end part_I_part_II_l755_755137


namespace cake_and_tea_cost_l755_755953

theorem cake_and_tea_cost (cost_of_milk_tea : ℝ) (cost_of_cake : ℝ)
    (h1 : cost_of_cake = (3 / 4) * cost_of_milk_tea)
    (h2 : cost_of_milk_tea = 2.40) :
    2 * cost_of_cake + cost_of_milk_tea = 6.00 := 
sorry

end cake_and_tea_cost_l755_755953


namespace problem1_problem2_l755_755869

-- Define the parametric equation of circle C
def circle_eq (theta : ℝ) : ℝ × ℝ :=
  let x := 1 + (sqrt 2) * cos theta
  let y := 1 + (sqrt 2) * sin theta
  (x, y)

-- Define the polar equation of line l in Cartesian form for general m
def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  x + y = m

-- Problem (I): Positional relationship between line l and circle C for m = 3
theorem problem1 : line_eq 3 (fst (circle_eq 0)) (snd (circle_eq 0)) -> 
  let center_dist := abs ((1 + 1 - 3) / sqrt 2)
  let radius := sqrt 2
  center_dist < radius :=
begin
  sorry
end

-- Problem (II): Coordinates of the points on circle C whose distance to line l is 2sqrt2
theorem problem2 (m : ℝ) (d_pts : d_pts_in_c : ℝ × ℝ) (r := sqrt 2) :
  ∃ p1 p2 : ℝ × ℝ, 
    (p1 ∈ circle_eq) ∧ (p2 ∈ circle_eq) ∧
    dist (fst p1, snd p1) (fst d_pts, snd d_pts) = 2 * sqrt 2 ∧
    dist (fst p2, snd p2) (fst d_pts, snd d_pts) = 2 * sqrt 2 :=
begin
  sorry
end

end problem1_problem2_l755_755869


namespace percent_games_lost_l755_755968

theorem percent_games_lost
  (w l t : ℕ)
  (h_ratio : 7 * l = 3 * w)
  (h_tied : t = 5) :
  (l : ℝ) / (w + l + t) * 100 = 20 :=
by
  sorry

end percent_games_lost_l755_755968


namespace equation_of_line_l_l755_755841

noncomputable def line_eq (a b c : ℚ) : ℚ → ℚ → Prop := λ x y => a * x + b * y + c = 0

theorem equation_of_line_l : 
  ∃ m : ℚ, 
  (∀ x y : ℚ, 
    (2 * x - 3 * y - 3 = 0 ∧ x + y + 2 = 0 → line_eq 3 1 m x y) ∧ 
    (3 * x + y - 1 = 0 → line_eq 3 1 0 x y)
  ) →
  line_eq 15 5 16 (-3/5) (-7/5) :=
by 
  sorry

end equation_of_line_l_l755_755841


namespace triangle_area_l755_755849

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin(c)

theorem triangle_area (a b : ℝ) (c : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : c = π/3) : area_of_triangle a b c = (3 * Real.sqrt 3) / 2 :=
sorry

end triangle_area_l755_755849


namespace problem_1_solution_problem_2_solution_l755_755812

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a

def has_two_distinct_real_roots (a b : ℝ) : Prop :=
  b > a ∧ a ≠ 0

def has_no_real_roots (a b : ℝ) : Prop :=
  a > b

theorem problem_1_solution :
  (finset.card
      (finset.filter
        (λ ab : ℝ × ℝ, has_two_distinct_real_roots ab.1 ab.2)
        (set.finite_to_finset
          {ab : ℝ × ℝ | ab.1 ∈ {0, 1, 2, 3} ∧ ab.2 ∈ {0, 1, 2, 3}})))
  / 16 = 3 / 16 := sorry

theorem problem_2_solution :
  let Ω := {ab : ℝ × ℝ | 0 ≤ ab.1 ∧ ab.1 ≤ 3 ∧ 0 ≤ ab.2 ∧ ab.2 ≤ 2},
  let M := {ab : ℝ × ℝ | 0 ≤ ab.1 ∧ ab.1 ≤ 3 ∧ 0 ≤ ab.2 ∧ ab.2 ≤ 2 ∧ has_no_real_roots ab.1 ab.2} in
  set.finite.card (set.to_finset M) / set.finite.card (set.to_finset Ω) = 2 / 3 := sorry

end problem_1_solution_problem_2_solution_l755_755812


namespace intersection_A_compB_l755_755821

def setA : Set ℤ := {x | (abs (x - 1) < 3)}
def setB : Set ℝ := {x | x^2 + 2 * x - 3 ≥ 0}
def setCompB : Set ℝ := {x | ¬(x^2 + 2 * x - 3 ≥ 0)}

theorem intersection_A_compB :
  { x : ℤ | x ∈ setA ∧ (x:ℝ) ∈ setCompB } = {-1, 0} :=
sorry

end intersection_A_compB_l755_755821


namespace perimeter_area_quadrilateral_l755_755672

-- Definitions:
variable (a : ℝ)

-- Problem Statement:
theorem perimeter_area_quadrilateral
  (a_pos : 0 < a) :
  let side_length := a * (2 - Real.sqrt 2) ^ (1/2)
  let perimeter := 4 * side_length
  let area := a^2 * (Real.sqrt 2 - 1)
  in perimeter = (4 * a * Real.sqrt(2 - Real.sqrt 2)) ∧
     area = (a^2 * (Real.sqrt 2 - 1)) :=
by
  sorry

end perimeter_area_quadrilateral_l755_755672


namespace jogged_on_Tuesday_l755_755529

-- Define the variables for the distances jogged
variables (M T W D : ℕ)

-- State the conditions
def condition1 : M = 2 := by sorry
def condition2 : W = 9 := by sorry
def condition3 : D = 16 := by sorry
def condition4 : M + T + W = D := by sorry

-- Prove that Debby jogged 5 km on Tuesday
theorem jogged_on_Tuesday : T = 5 := by
  rw [condition4, condition1, condition2, condition3]
  sorry

end jogged_on_Tuesday_l755_755529


namespace monthly_income_l755_755658

-- Define the conditions
variable (I : ℝ) -- Total monthly income
variable (remaining : ℝ) -- Remaining amount before donation
variable (remaining_after_donation : ℝ) -- Amount after donation

-- Conditions
def condition1 : Prop := remaining = I - 0.63 * I - 1500
def condition2 : Prop := remaining_after_donation = remaining - 0.05 * remaining
def condition3 : Prop := remaining_after_donation = 35000

-- Theorem to prove the total monthly income
theorem monthly_income (h1 : condition1 I remaining) (h2 : condition2 remaining remaining_after_donation) (h3 : condition3 remaining_after_donation) : I = 103600 := 
by sorry

end monthly_income_l755_755658


namespace fraction_eq_repeating_decimal_l755_755278

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755278


namespace jenna_works_30_hours_per_week_l755_755879

noncomputable def jenna_works_hours_per_week : ℕ :=
let concert_ticket_cost : ℝ := 181 in
let drink_ticket_cost : ℝ := 7 in
let number_of_drink_tickets : ℕ := 5 in
let total_drink_ticket_cost : ℝ := number_of_drink_tickets * drink_ticket_cost in
let total_outing_cost : ℝ := concert_ticket_cost + total_drink_ticket_cost in
let hourly_wage : ℝ := 18 in
let monthly_salary_percentage : ℝ := 0.10 in
let monthly_salary : ℝ := total_outing_cost / monthly_salary_percentage in
let weekly_hours : ℝ := monthly_salary / (4 * hourly_wage) in
weekly_hours.to_nat 

theorem jenna_works_30_hours_per_week :
  jenna_works_hours_per_week = 30 :=
by
  sorry

end jenna_works_30_hours_per_week_l755_755879


namespace parabola_bisects_rectangle_l755_755712
open Real

theorem parabola_bisects_rectangle (a : ℝ) (h_pos : a > 0) : 
  ((a^3 + a) / 2 = (a^3 / 3 + a)) → a = sqrt 3 := by
  sorry

end parabola_bisects_rectangle_l755_755712


namespace negation_of_proposition_l755_755058

variable (a : ℝ)

theorem negation_of_proposition :
  (¬ (∀ a ∈ set.Icc 0 1, a^4 + a^2 > 1)) ↔ (∃ a ∈ set.Icc 0 1, a^4 + a^2 ≤ 1) :=
sorry

end negation_of_proposition_l755_755058


namespace number_of_thrown_out_carrots_l755_755218

-- Definitions from the conditions
def initial_carrots : ℕ := 48
def picked_next_day : ℕ := 42
def total_carrots : ℕ := 45

-- Proposition stating the problem
theorem number_of_thrown_out_carrots (x : ℕ) : initial_carrots - x + picked_next_day = total_carrots → x = 45 :=
by
  sorry

end number_of_thrown_out_carrots_l755_755218


namespace solve_for_y_l755_755929

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l755_755929


namespace problem1_l755_755748

variable (θ : ℝ)

theorem problem1
  (h : sin θ + cos θ = -√10 / 5) :
  (1 / sin θ + 1 / cos θ = 2 * √10 / 3) ∧ (tan θ = -1/3 ∨ tan θ = -3) :=
by
  sorry

end problem1_l755_755748


namespace repeating_decimal_eq_fraction_l755_755335

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755335


namespace repeating_decimal_equiv_fraction_l755_755281

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755281


namespace evaluate_polynomial_at_4_l755_755244

-- Define the polynomial function
def polynomial (x : ℕ) : ℕ := x^4 + x^3 + x^2 + x + 1

-- Statement that evaluates the polynomial at x = 4 to be 341
theorem evaluate_polynomial_at_4 : polynomial 4 = 341 := 
by
  simp [polynomial]
  norm_num
  triv
  sorry

end evaluate_polynomial_at_4_l755_755244


namespace discriminant_value_l755_755715

theorem discriminant_value (a b c : ℤ) (m n p : ℤ) (h_eq : 3 * a * a + 4 * a + 7 = 0)
  (h_form : ∀ (x : ℤ), x = (m + (sqrt n)) / p ∨ x = (m - sqrt n) / p)
  (h_rel_prime : ∀ (x y : ℤ), gcd x y = 1 → (gcd x (gcd y m * gcd y p)) = 1) :
  n = 100 :=
by
  sorry

end discriminant_value_l755_755715


namespace find_some_number_l755_755609

theorem find_some_number :
  let some_number := (1/2) in
  (1/2 : ℚ) + ((2/3 : ℚ) * (3/8 : ℚ) + 4) - some_number = 4.25 →
  some_number = (1/2 : ℚ) :=
by
  sorry

end find_some_number_l755_755609


namespace perpendicular_OY_XY_l755_755555

variables {A B C D X Y O : Type}
variables [has_circumcircle ABCD O] [has_circumcircle ABX] [has_circumcircle CDX]
variables (cyclic_ABCD : cyclic ABCD) (intersect_AC_BD : intersect AC BD X)
variables (circ_ABX_meet_CDX : circumcircle ABX ∩ circumcircle CDX = {X, Y})
variables (center_O : circumcircle ABCD = O)
variables (distinct_OXY : O ≠ X ∧ X ≠ Y ∧ Y ≠ O)

theorem perpendicular_OY_XY 
    (cyclic_ABCD : cyclic ABCD) 
    (intersect_AC_BD : intersect AC BD X) 
    (circ_ABX_meet_CDX : circumcircle ABX ∩ circumcircle CDX = {X, Y}) 
    (center_O : circumcircle ABCD = O) 
    (distinct_OXY : O ≠ X ∧ X ≠ Y ∧ Y ≠ O) :
  is_perpendicular OY XY := 
sorry

end perpendicular_OY_XY_l755_755555


namespace general_formula_for_an_sum_sn_less_than_l755_755772

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 11 * x + 24
def a₁ : ℝ := 3
def a₂ : ℝ := 8
def diff_seq (n : ℕ) : ℝ := 2 * n + 3
def a (n : ℕ) : ℝ := (n : ℝ)^2 + 2 * (n : ℝ)
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, 1 / a k)

-- We are to prove these statements
theorem general_formula_for_an : ∀ (n : ℕ), a n = n^2 + 2 * n := by
  sorry

theorem sum_sn_less_than : ∀ (n : ℕ), S n < 3 / 4 := by
  sorry

end general_formula_for_an_sum_sn_less_than_l755_755772


namespace expected_remaining_matches_for_60_matches_l755_755938

noncomputable def expected_remaining_matches (n : ℕ) : ℝ :=
  if h : n = 60 then 7.795 else 0 -- We'll handle the specific case where n = 60

theorem expected_remaining_matches_for_60_matches :
  expected_remaining_matches 60 = 7.795 :=
by
  sorry

end expected_remaining_matches_for_60_matches_l755_755938


namespace dogs_bunnies_ratio_l755_755855

theorem dogs_bunnies_ratio (total : ℕ) (dogs : ℕ) (bunnies : ℕ) (h1 : total = 375) (h2 : dogs = 75) (h3 : bunnies = total - dogs) : (75 / 75 : ℚ) / (300 / 75 : ℚ) = 1 / 4 := by
  sorry

end dogs_bunnies_ratio_l755_755855


namespace cubics_sum_eq_l755_755547

-- Definitions from conditions
variables (a b c : ℝ)
axiom h1 : a + b + c = 1
axiom h2 : ab + ac + bc = 2
axiom h3 : abc = 5

-- The theorem to prove
theorem cubics_sum_eq : a^3 + b^3 + c^3 = 14 :=
by
  sorry

end cubics_sum_eq_l755_755547


namespace fraction_eq_repeating_decimal_l755_755274

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755274


namespace compound_interest_l755_755657

variables {a r : ℝ}

theorem compound_interest (a r : ℝ) :
  (a * (1 + r)^10) = a * (1 + r)^(2020 - 2010) :=
by
  sorry

end compound_interest_l755_755657


namespace smallest_ratio_is_three_l755_755182

theorem smallest_ratio_is_three (m n : ℕ) (a : ℕ) (h1 : 2^m + 1 = a * (2^n + 1)) (h2 : a > 1) : a = 3 :=
sorry

end smallest_ratio_is_three_l755_755182


namespace exercise_l755_755111

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l755_755111


namespace Mitzi_spent_on_ticket_l755_755017

theorem Mitzi_spent_on_ticket
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ)
  (ha : a = 75) (hb : b = 13) (hc : c = 23) (hd : d = 9)
  : a - (b + c + d) = 30 := by
  rw [ha, hb, hc, hd]
  dsimp
  norm_num

end Mitzi_spent_on_ticket_l755_755017


namespace triangle_right_angled_l755_755521

-- Definitions and theorems about triangles, circles, and tangency points.
noncomputable def triangle (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] := sorry
noncomputable def excircle_tangent_point (ABC : triangle) := sorry
noncomputable def circumcenter (A1 B1 C1 : Type) := sorry
noncomputable def circumcircle (ABC : triangle) := sorry

-- Problem statement in Lean 4
theorem triangle_right_angled
  (ABC : Type) [triangle ABC]
  (A1 B1 C1 : Type)
  (h1 : excircle_tangent_point ABC A1)
  (h2 : excircle_tangent_point ABC B1)
  (h3 : excircle_tangent_point ABC C1)
  (h4 : circumcenter (A1 B1 C1) ∈ circumcircle ABC) :
  is_right_triangle ABC := sorry

end triangle_right_angled_l755_755521


namespace main_theorem_l755_755891

variable {R : Type*} [LinearOrder R] [HasZero R] [HasOne R] [Neg R]

-- Definitions for the conditions
def is_even (f : R → R) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : R → R) : Prop :=
  ∀ x1 x2, (0 ≤ x1) → (0 ≤ x2) → (x1 ≠ x2) → (x1 < x2) → f x2 < f x1

-- The main theorem statement
theorem main_theorem (f : R → R)
  (h_even : is_even f)
  (h_dec : is_decreasing_on_nonneg f) :
  f 1 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end main_theorem_l755_755891


namespace fraction_eq_repeating_decimal_l755_755361

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755361


namespace asphalt_road_proof_l755_755986

-- We define the initial conditions given in the problem
def man_hours (men days hours_per_day : Nat) : Nat :=
  men * days * hours_per_day

-- Given the conditions for asphalting 1 km road
def conditions_1 (men1 days1 hours_per_day1 : Nat) : Prop :=
  man_hours men1 days1 hours_per_day1 = 2880

-- Given that the second road is 2 km long
def conditions_2 (man_hours1 : Nat) : Prop :=
  2 * man_hours1 = 5760

-- Given the working conditions for the second road
def conditions_3 (men2 days2 hours_per_day2 : Nat) : Prop :=
  men2 * days2 * hours_per_day2 = 5760

-- The theorem to prove
theorem asphalt_road_proof 
  (men1 days1 hours_per_day1 days2 hours_per_day2 men2 : Nat)
  (H1 : conditions_1 men1 days1 hours_per_day1)
  (H2 : conditions_2 (man_hours men1 days1 hours_per_day1))
  (H3 : men2 * days2 * hours_per_day2 = 5760)
  : men2 = 20 :=
by
  sorry

end asphalt_road_proof_l755_755986


namespace evaluate_polynomial_at_4_l755_755243

-- Define the polynomial function
def polynomial (x : ℕ) : ℕ := x^4 + x^3 + x^2 + x + 1

-- Statement that evaluates the polynomial at x = 4 to be 341
theorem evaluate_polynomial_at_4 : polynomial 4 = 341 := 
by
  simp [polynomial]
  norm_num
  triv
  sorry

end evaluate_polynomial_at_4_l755_755243


namespace repeating_decimal_is_fraction_l755_755300

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755300


namespace moviegoers_group_adults_l755_755690

theorem moviegoers_group_adults (A C : ℕ) (h1 : A + C = 7) (h2 : 9.50 * A + 6.50 * C = 54.50) : A = 3 :=
sorry

end moviegoers_group_adults_l755_755690


namespace math_problem_l755_755128

theorem math_problem 
  (X : ℝ)
  (num1 : ℝ := 1 + 28/63)
  (num2 : ℝ := 8 + 7/16)
  (frac_sub1 : ℝ := 19/24 - 21/40)
  (frac_sub2 : ℝ := 1 + 28/63 - 17/21)
  (denom_calc : ℝ := 0.675 * 2.4 - 0.02) :
  0.125 * X / (frac_sub1 * num2) = (frac_sub2 * 0.7) / denom_calc → X = 5 := 
sorry

end math_problem_l755_755128


namespace largest_e_l755_755833

variable (a b c d e : ℤ)

theorem largest_e 
  (h1 : a - 1 = b + 2) 
  (h2 : a - 1 = c - 3)
  (h3 : a - 1 = d + 4)
  (h4 : a - 1 = e - 6) 
  : e > a ∧ e > b ∧ e > c ∧ e > d := 
sorry

end largest_e_l755_755833


namespace part_I_part_II_l755_755518

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem part_I (k : ℝ) (hk : k = 1) :
  (∀ x, 0 < x ∧ x < 1 → 0 < f 1 x - f 1 1)
  ∧ (∀ x, 1 < x → f 1 1 > f 1 x)
  ∧ f 1 1 = 0 :=
by
  sorry

theorem part_II (k : ℝ) (h_no_zeros : ∀ x, f k x ≠ 0) :
  k > 1 / exp 1 :=
by
  sorry

end part_I_part_II_l755_755518


namespace kim_paid_with_amount_l755_755502

-- Define the conditions
def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_rate : ℝ := 0.20
def change_received : ℝ := 5

-- Define the total amount paid formula
def total_cost_before_tip := meal_cost + drink_cost
def tip_amount := tip_rate * total_cost_before_tip
def total_cost_after_tip := total_cost_before_tip + tip_amount
def amount_paid := total_cost_after_tip + change_received

-- Statement of the theorem
theorem kim_paid_with_amount : amount_paid = 20 := by
  sorry

end kim_paid_with_amount_l755_755502


namespace polygon_area_is_46_l755_755551

-- Let ABCDEF be a polygon with specific sides
variables (AB BC CD DE EF FA : ℝ)
-- Introduce point G as the intersection of extended lines DC and AF
variables (G : Type)

-- Given conditions
def conditions : Prop :=
  AB = 6 ∧ BC = 9 ∧ CD = 4 ∧ FA = 5

-- Define the areas of triangle and quadrilateral
def area_of_polygon : ℝ :=
  let area_ABCG := 54 -- AB * BC
  let area_GFED := 8  -- GF * GD
  area_ABCG - area_GFED

-- Proof statement
theorem polygon_area_is_46 : conditions → area_of_polygon = 46 :=
by sorry

end polygon_area_is_46_l755_755551


namespace count_valid_sequences_l755_755827

theorem count_valid_sequences : 
  ∃ n : ℕ, n = 390625 ∧ 
    (∀ (x : ℕ → ℕ), 
      (∀ i, i ∈ {0, 1, 2, 3, 4, 5, 6, 7} → x i < 10) ∧ 
      (∀ i, i < 7 → (x i % 2 ≠ x (i + 1) % 2)) ∧ 
      (x 0 % 2 = 1)
    → 
      (card {xs : (fin 8) → ℕ // (∀ i, i ∈ {0, 1, 2, 3, 4, 5, 6, 7} → xs i < 10) ∧
                   (∀ i, i < 7 → (xs i % 2 ≠ xs (i + 1) % 2)) ∧
                   (xs 0 % 2 = 1) }) = n) :=
by 
  use 390625
  split
  · refl
  · intro x h
    sorry

end count_valid_sequences_l755_755827


namespace negation_of_proposition_l755_755059

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x ≥ 0) ↔ (∃ x > 0, x^2 + x < 0) :=
by 
  sorry

end negation_of_proposition_l755_755059


namespace q_eq_zero_iff_arithmetic_sequence_l755_755847

noncomputable def arithmetic_seq_sum (a d : ℕ) : ℕ → ℕ
| 0     := 0
| (n+1) := (n + 1) * a + (n * (n + 1) / 2) * d

theorem q_eq_zero_iff_arithmetic_sequence (A B q : ℤ) (n : ℕ) (hA : A ≠ 0)
  (Sn : ℕ → ℤ := λ n, A * n^2 + B * n + q) :
  (q = 0) ↔ (∃ a d : ℤ, ∀ n : ℕ, Sn n = (n * a + n * (n - 1) / 2 * d)) :=
begin
  sorry
end

end q_eq_zero_iff_arithmetic_sequence_l755_755847


namespace largest_root_range_l755_755705

theorem largest_root_range (b_0 b_1 b_2 b_3 : ℝ)
  (hb_0 : |b_0| ≤ 3) (hb_1 : |b_1| ≤ 3) (hb_2 : |b_2| ≤ 3) (hb_3 : |b_3| ≤ 3) :
  ∃ s : ℝ, (∃ x : ℝ, x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 = 0 ∧ x > 0 ∧ s = x) ∧ 3 < s ∧ s < 4 := 
sorry

end largest_root_range_l755_755705


namespace total_trees_planted_total_trees_when_a_100_l755_755085

-- Define the number of trees planted by each team based on 'a'
def trees_first_team (a : ℕ) : ℕ := a
def trees_second_team (a : ℕ) : ℕ := 2 * a + 8
def trees_third_team (a : ℕ) : ℕ := (2 * a + 8) / 2 - 6

-- Define the total number of trees
def total_trees (a : ℕ) : ℕ := 
  trees_first_team a + trees_second_team a + trees_third_team a

-- The main theorem
theorem total_trees_planted (a : ℕ) : total_trees a = 4 * a + 6 :=
by
  sorry

-- The specific calculation when a = 100
theorem total_trees_when_a_100 : total_trees 100 = 406 :=
by
  sorry

end total_trees_planted_total_trees_when_a_100_l755_755085


namespace midpoint_of_OaOb_intersection_with_AB_l755_755466

/-
Two circles intersect at points \( A \) and \( B \). Common tangents of the circles intersect the circles 
at points \( C \) and \( D \). \( O_a \) is the circumcenter of \( \triangle ACD \). \( O_b \) is the 
circumcenter of \( \triangle BCD \). Prove that \( AB \) intersects \( O_aO_b \) at its midpoint.
-/

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def circumcenter (A B C : Point) : Point :=
sorry

variables {A B C D Oa Ob: Point}

/-- Given two circles intersect at points A and B, common tangents at C and D, and 
    Oa is the circumcenter of triangle ACD, Ob is the circumcenter of triangle BCD, 
    prove that AB intersects OaOb at its midpoint. -/
theorem midpoint_of_OaOb_intersection_with_AB :
  (circumcenter A C D) = Oa →
  (circumcenter B C D) = Ob →
  let M := Point.mk ((C.x + D.x) / 2) ((C.y + D.y) / 2) in
  ∃ M ∈ [A, B], M = midpoint Oa Ob :=
sorry

end midpoint_of_OaOb_intersection_with_AB_l755_755466


namespace g_negative_example1_g_negative_example2_g_negative_example3_l755_755515

noncomputable def g (a : ℚ) : ℚ := sorry

axiom g_mul (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : g (a * b) = g a + g b
axiom g_prime (p : ℕ) (hp : Nat.Prime p) : g (p * p) = p

theorem g_negative_example1 : g (8/81) < 0 := sorry
theorem g_negative_example2 : g (25/72) < 0 := sorry
theorem g_negative_example3 : g (49/18) < 0 := sorry

end g_negative_example1_g_negative_example2_g_negative_example3_l755_755515


namespace square_area_l755_755189

open Real

-- Define the given conditions
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line := 8
def side_points (x : ℝ) := parabola x = line

-- Problem statement: Find the area of the square
theorem square_area :
  let x1 : ℝ := 1 in
  let x2 : ℝ := -5 in
  let side_length := abs (x1 - x2) in
  side_length * side_length = 36 :=
by
  -- Proof is replaced with sorry to satisfy the problem constraints
  sorry

end square_area_l755_755189


namespace total_length_AD_is_40_l755_755914

-- Definitions for the problem
variables {A D B C E M : Type*}
variables [metric_space A] [metric_space D] [metric_space B] [metric_space C] [metric_space E] [metric_space M]
variables [dist : metric_space.dist]

-- The given conditions as hypostheses
-- 1. Points B, C, and E divide line segment AD into four equal parts.
-- 2. M is the midpoint of AD.
-- 3. The length of segment MC is 10 units.
variables (AB BC CD DE AM MD MC : ℝ)

-- Specific constraints for this problem
def B_divides_AD_into_4 (A D B C E : Type*) := AB = BC ∧ BC = CD ∧ CD = DE
def M_is_midpoint (A D M : Type*) := AM = MD
def length_MC_is_10 (M C : Type*) := MC = 10

theorem total_length_AD_is_40 
  (h1 : B_divides_AD_into_4 A D B C E) 
  (h2 : M_is_midpoint A D M)
  (h3 : length_MC_is_10 M C) :
  4 * BC = 40 :=
  sorry

end total_length_AD_is_40_l755_755914


namespace tire_price_l755_755178

theorem tire_price (x : ℝ) (h : 3 * x + 10 = 310) : x = 100 :=
sorry

end tire_price_l755_755178


namespace range_of_f_l755_755067

def f (x : ℝ) : ℝ := (1 - sin x) / (3 * sin x + 2)

theorem range_of_f :
  (set.range f) = set.Iic (-2) ∪ set.Ici 0 :=
sorry

end range_of_f_l755_755067


namespace length_of_EF_l755_755482

theorem length_of_EF {A B C D E F : Type*} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup F]
  (AB : ℝ) (BC : ℝ) (rectangle_ABCD : true) (point_B_coincides_with_D : true) 
  (pentagon_ABEFC : true) :
  AB = 4 ∧ BC = 8 → segment_length EF = 4 * sqrt 5 := 
by
  sorry

end length_of_EF_l755_755482


namespace bus_empty_seats_l755_755970

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end bus_empty_seats_l755_755970


namespace Cauchy_solution_on_X_l755_755092

section CauchyEquation

variable (f : ℝ → ℝ)

def is_morphism (f : ℝ → ℝ) := ∀ x y : ℝ, f (x + y) = f x + f y

theorem Cauchy_solution_on_X :
  (∀ a b : ℤ, ∀ c d : ℤ, a + b * Real.sqrt 2 = c + d * Real.sqrt 2 → a = c ∧ b = d) →
  is_morphism f →
  ∃ x y : ℝ, ∀ a b : ℤ,
    f (a + b * Real.sqrt 2) = a * x + b * y :=
by
  intros h1 h2
  let x := f 1
  let y := f (Real.sqrt 2)
  exists x, y
  intros a b
  sorry

end CauchyEquation

end Cauchy_solution_on_X_l755_755092


namespace exercise_l755_755110

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l755_755110


namespace sequence_500th_term_l755_755535

noncomputable def sequence_term (n : ℕ) : ℚ :=
  nat.rec_on n 2 (λ n' prev, 1 / (1 - prev))

theorem sequence_500th_term : sequence_term 499 = -1 :=
  sorry

end sequence_500th_term_l755_755535


namespace recurring_decimal_to_fraction_l755_755318

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755318


namespace max_touched_points_by_line_l755_755740

noncomputable section

open Function

-- Definitions of the conditions
def coplanar_circles (circles : Set (Set ℝ)) : Prop :=
  ∀ c₁ c₂ : Set ℝ, c₁ ∈ circles → c₂ ∈ circles → c₁ ≠ c₂ → ∃ p : ℝ, p ∈ c₁ ∧ p ∈ c₂

def max_touched_points (line_circle : ℝ → ℝ) : ℕ :=
  2

-- The theorem statement that needs to be proven
theorem max_touched_points_by_line {circles : Set (Set ℝ)} (h_coplanar : coplanar_circles circles) :
  ∀ line : ℝ → ℝ, (∃ (c₁ c₂ c₃ : Set ℝ), c₁ ∈ circles ∧ c₂ ∈ circles ∧ c₃ ∈ circles ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃) →
  ∃ (p : ℕ), p = 6 := 
sorry

end max_touched_points_by_line_l755_755740


namespace men_involved_with_women_undetermined_l755_755937

-- Definitions based on conditions
def work_done_by_man_per_day : ℝ := 1 / 100
def work_done_by_women (W : ℝ) := 15 * W
def total_work_done_per_day (M : ℝ) (W : ℝ) := M * work_done_by_man_per_day + work_done_by_women W
def total_work_required_per_day : ℝ := 1 / 5

-- Final equation
def work_equation (M : ℝ) (W : ℝ) : Prop :=
  total_work_done_per_day M W = total_work_required_per_day

theorem men_involved_with_women_undetermined (M : ℝ) (W : ℝ):
  W = (x : ℝ) * work_done_by_man_per_day →
  work_equation M W →
  M + 15 * x = 20 →
  ∃ (x : ℝ), true := 
sorry

end men_involved_with_women_undetermined_l755_755937


namespace range_of_m_on_line_l755_755866
noncomputable theory

open Real

theorem range_of_m_on_line 
  (A B : ℝ × ℝ)
  (hA : A = (1, 0))
  (hB : B = (4, 0))
  (m : ℝ)
  (x : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (x, x + m))
  (h_line : x - (x + m) + m = 0)
  (h_dist : dist P A = 1/2 * dist P B) :
  m ∈ Icc (-2 * sqrt 2) (2 * sqrt 2) :=
sorry

end range_of_m_on_line_l755_755866


namespace division_remainder_l755_755621

theorem division_remainder : 
  ∀ (Dividend Divisor Quotient Remainder : ℕ), 
  Dividend = 760 → 
  Divisor = 36 → 
  Quotient = 21 → 
  Dividend = (Divisor * Quotient) + Remainder → 
  Remainder = 4 := 
by 
  intros Dividend Divisor Quotient Remainder h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  have h5 : 760 = 36 * 21 + Remainder := h4
  linarith

end division_remainder_l755_755621


namespace fraction_for_repeating_56_l755_755268

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755268


namespace polynomial_evaluation_x_eq_4_l755_755240

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l755_755240


namespace distance_from_point_to_line_1_0_3x_4y_8_eq_1_l755_755045

noncomputable def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs(A * x0 + B * y0 + C) / sqrt(A ^ 2 + B ^ 2)

theorem distance_from_point_to_line_1_0_3x_4y_8_eq_1 : distance_from_point_to_line 1 0 3 4 (-8) = 1 := 
by
  sorry

end distance_from_point_to_line_1_0_3x_4y_8_eq_1_l755_755045


namespace sufficiently_large_integer_sum_of_elements_l755_755885

theorem sufficiently_large_integer_sum_of_elements (A : Set ℕ) (hA₁ : ∀ (d : ℕ), d > 1 → ¬ ∀ a ∈ A, d ∣ a) :
  ∃ N₀ : ℕ, ∀ n ≥ N₀, ∃ (m : ℕ → ℕ), (∑ i in A.to_finset, (m i) * i) = n := 
sorry

end sufficiently_large_integer_sum_of_elements_l755_755885


namespace problem_solution_l755_755006

def h (x : ℝ) : ℝ := (4 * x^2 + 2 * x + 7) / (x^2 - 2 * x + 5)
def k (x : ℝ) : ℝ := 2 * x + 3

theorem problem_solution : h (k (-1)) + k (h (-1)) = 8.5 := by
  sorry

end problem_solution_l755_755006


namespace pears_count_l755_755036

theorem pears_count (A F P : ℕ)
  (hA : A = 12)
  (hF : F = 4 * 12 + 3)
  (hP : P = F - A) :
  P = 39 := by
  sorry

end pears_count_l755_755036


namespace units_digit_m_sq_plus_2_m_l755_755510

def m := 2017^2 + 2^2017

theorem units_digit_m_sq_plus_2_m (m := 2017^2 + 2^2017) : (m^2 + 2^m) % 10 = 3 := 
by
  sorry

end units_digit_m_sq_plus_2_m_l755_755510


namespace cube_skew_lines_probability_l755_755235

noncomputable def probability_skew_lines (cube_edges : ℕ) (remaining_edges : ℕ) (skew_edges : ℕ) : ℚ :=
  skew_edges / remaining_edges

theorem cube_skew_lines_probability :
  let total_edges := 12 in
  let remaining_edges := 11 in
  let skew_edges := 4 in
  probability_skew_lines total_edges remaining_edges skew_edges = 4 / 11 :=
by
  sorry

end cube_skew_lines_probability_l755_755235


namespace ninth_term_arith_seq_l755_755586

-- Define the arithmetic sequence.
def arith_seq (a₁ d : ℚ) (n : ℕ) := a₁ + n * d

-- Define the third and fifteenth terms of the sequence.
def third_term := (5 : ℚ) / 11
def fifteenth_term := (7 : ℚ) / 8

-- Prove that the ninth term is 117/176 given the conditions.
theorem ninth_term_arith_seq :
    ∃ (a₁ d : ℚ), 
    arith_seq a₁ d 2 = third_term ∧ 
    arith_seq a₁ d 14 = fifteenth_term ∧
    arith_seq a₁ d 8 = 117 / 176 :=
by
  sorry

end ninth_term_arith_seq_l755_755586


namespace fraction_equality_l755_755027

theorem fraction_equality
  (a b c d : ℝ) 
  (h1 : b ≠ c)
  (h2 : (a * c - b^2) / (a - 2 * b + c) = (b * d - c^2) / (b - 2 * c + d)) : 
  (a * c - b^2) / (a - 2 * b + c) = (a * d - b * c) / (a - b - c + d) ∧
  (b * d - c^2) / (b - 2 * c + d) = (a * d - b * c) / (a - b - c + d) := 
by
  sorry

end fraction_equality_l755_755027


namespace calculate_permutation_sum_l755_755696

noncomputable def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem calculate_permutation_sum (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 3) :
  A (2 * n) (n + 3) + A 4 (n + 1) = 744 := by
  sorry

end calculate_permutation_sum_l755_755696


namespace fraction_a3_a5_l755_755762

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → a (n + 1) * a n = a n + (-1)^(n + 1)

theorem fraction_a3_a5 (a : ℕ → ℝ) (h : sequence a) : a 3 / a 5 = 3 / 4 :=
by
  sorry

end fraction_a3_a5_l755_755762


namespace find_number_l755_755246

-- Define the condition given in the problem
def condition (x : ℝ) : Prop := 100 / x = 400

-- Define the goal given the condition
theorem find_number : ∃ x : ℝ, condition x ∧ x = 0.25 :=
begin
  use 0.25,
  split,
  { sorry },
  { refl }
end

end find_number_l755_755246


namespace number_of_green_fish_l755_755903

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end number_of_green_fish_l755_755903


namespace cannot_determine_parallel_l755_755199

open_locale classical

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a, b and c
variables (a b c : V)

-- Define conditions for each option
def A_condition : Prop := (∀ k l : ℝ, a = k • c → b = l • c → a = k • b)
def B_condition : Prop := (∃ k1 k2 : ℝ, a = -k1 • c ∧ b = k2 • c)
def C_condition : Prop := (∃ k : ℝ, a = k • b)
def D_condition : Prop := (∃ k : ℝ, ∥a∥ = k * ∥b∥)

-- The statement we need to prove
theorem cannot_determine_parallel (hA : A_condition a b c) 
                                   (hB : B_condition a b c) 
                                   (hC : C_condition a b) 
                                   : ¬ (D_condition a b) :=
sorry

end cannot_determine_parallel_l755_755199


namespace find_remaining_sum_l755_755615

def series_sum (n : ℕ) (a : ℕ → ℚ) : ℚ := ∑ k in finset.range n, a k

def check_sum (S : finset ℚ) : Prop := S.sum id = 20.19

theorem find_remaining_sum :
  ∃ S : finset ℚ, 
  S = (finset.range 10).map (λ _, 1.11) ∪ (finset.range 11).filter (λ k, k ≠ 1 ∧ k ≠ 2).map (λ _, 1.01) ∧ check_sum S :=
by {
  sorry
}

end find_remaining_sum_l755_755615


namespace max_points_on_2010_dim_spheres_l755_755667

theorem max_points_on_2010_dim_spheres (S : Set (ℝ^2010)) (r : ℝ) (hS : ∀ s ∈ S, ∃ c : ℝ^2010, s = { p : ℝ^2010 | dist p c = r }) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end max_points_on_2010_dim_spheres_l755_755667


namespace conjugate_in_first_quadrant_l755_755416

noncomputable def i := Complex.I

def z : ℂ := (2 + i ^ 2016) / (1 + i)

def z_conj : ℂ := z.conj

def in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem conjugate_in_first_quadrant :
  in_first_quadrant z_conj :=
sorry

end conjugate_in_first_quadrant_l755_755416


namespace find_b_l755_755173

theorem find_b (b : ℝ) :
  (∀ x y : ℝ, (x = 1 ∧ y = 5) → (y = x^2 + b * x + 3)) →
  (∀ x y : ℝ, (x = 3 ∧ y = 5) → (y = x^2 + b * x + 3)) →
  (∀ x y : ℝ, (x = 0 ∧ y = 3) → (y = x^2 + b * x + 3)) →
  b = 1 :=
by
  intros h1 h2 h3,
  -- Assume the main Lean code logic
  sorry

end find_b_l755_755173


namespace ladder_problem_l755_755652

noncomputable def ladder_sliding_speed (x y l dx_dt : ℝ) (h1 : x^2 + y^2 = l^2) (h2 : dx_dt = 3) : ℝ :=
  let dydt := -(x * dx_dt) / y in
  |dydt|

theorem ladder_problem
  (length : ℝ)
  (dx_dt : ℝ)
  (x : ℝ)
  (y : ℝ)
  (dx : ℝ)
  (speed : ℝ)
  (h1 : length = 5)
  (h2 : dx_dt = 3)
  (h3 : x = 1.4)
  (h4 : x^2 + y^2 = length^2) : 
  ladder_sliding_speed x y length dx_dt = 0.875 :=
by
  sorry

end ladder_problem_l755_755652


namespace conic_equation_is_hyperbola_l755_755116

theorem conic_equation_is_hyperbola :
  (∃ (C : Type) [ConC : conic C],  (∀ x y : ℝ, (x-3)^2 = 3*(y+4)^2 + 27 → C = ConicSection.Hyperbola)) :=
sorry

end conic_equation_is_hyperbola_l755_755116


namespace find_other_root_l755_755179

theorem find_other_root (x : ℚ) (h: 63 * x^2 - 100 * x + 45 = 0) (hx: x = 5 / 7) : x = 1 ∨ x = 5 / 7 :=
by 
  -- Insert the proof steps here if needed.
  sorry

end find_other_root_l755_755179


namespace sequence_b_n_l755_755704

theorem sequence_b_n (b : ℕ → ℕ) (h₀ : b 1 = 3) (h₁ : ∀ n, b (n + 1) = b n + 3 * n + 1) :
  b 50 = 3727 :=
sorry

end sequence_b_n_l755_755704


namespace smallest_n_contains_digit9_and_terminating_decimal_l755_755371

-- Define the condition that a number contains the digit 9
def contains_digit_9 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

-- Define the condition that a number is of the form 2^a * 5^b
def is_form_of_2a_5b (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2 ^ a * 5 ^ b

-- Define the main theorem
theorem smallest_n_contains_digit9_and_terminating_decimal : 
  ∃ (n : ℕ), contains_digit_9 n ∧ is_form_of_2a_5b n ∧ (∀ m, (contains_digit_9 m ∧ is_form_of_2a_5b m) → n ≤ m) ∧ n = 12500 :=
  sorry

end smallest_n_contains_digit9_and_terminating_decimal_l755_755371


namespace graph_C_is_correct_l755_755050

-- Define the piecewise function f
def f : ℝ → ℝ
| x := if -3 ≤ x ∧ x ≤ 0 then -2 - x
       else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
       else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
       else 0

-- Define the transformed function f shifted upwards by 2 units
def f_shifted : ℝ → ℝ := λ x, f x + 2

-- State the theorem that graph C corresponds to the graph of y = f(x) + 2
theorem graph_C_is_correct :
  (∀ x, -(3 : ℝ) ≤ x ∧ x ≤ 3 → f_shifted x = 
    if -3 ≤ x ∧ x ≤ 0 then 0 - x 
    else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2)
    else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2) + 2
    else 0) 
  := sorry

end graph_C_is_correct_l755_755050


namespace range_of_a_sqrt_inequality_l755_755804

def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 1|

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x ≥ a ^ 2 - 2 * a - 1) : -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

theorem sqrt_inequality (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) (x : ℝ) :
  sqrt (2 * m + 1) + sqrt (2 * n + 1) ≤ 2 * sqrt (f x) :=
by
  sorry

end range_of_a_sqrt_inequality_l755_755804


namespace polynomial_has_integer_root_l755_755523

-- Define the polynomial with rational coefficients and conditions
noncomputable def polynomial (a b c : ℚ) : ℚ[X] :=
  polynomial.X^3 + polynomial.C a * polynomial.X^2 + polynomial.C b * polynomial.X + polynomial.C c

-- Define the root properties
def root1 : ℚ[X] := polynomial.C (2 : ℚ) - polynomial.C (√5)
def root2 : ℚ[X] := polynomial.C (2 : ℚ) + polynomial.C (√5)
def integer_root : ℚ[X] := polynomial.C -4

-- The main theorem to state, using the conditions and target result
theorem polynomial_has_integer_root (a b c : ℚ)
  (h1 : root1.is_root (polynomial a b c))
  (h2 : root2.is_root (polynomial a b c))
  (h3 : integer_root.is_root (polynomial a b c)) : 
  ∃ r : ℚ, r = -4 := 
begin
  sorry
end

end polynomial_has_integer_root_l755_755523


namespace number_incorrect_statements_is_two_l755_755200

-- Definitions of the propositions
def prop1 (l1 l2 l3 : Type) [Line l1] [Line l2] [Line l3] (perpendicular : l1 → l3 → Prop) (parallel : l2 → l2 → Prop) :=
  ∀ {a b : l1}, a ≠ b → perpendicular a b → parallel l2 l3

def prop2 (l1 : Type) (p1 : Type) [Line l1] [Plane p1] (perpendicular_to_plane : l1 → p1 → Prop) (parallel : l1 → l1 → Prop) :=
  ∀ {a b : l1}, a ≠ b → perpendicular_to_plane a p1 → perpendicular_to_plane b p1 → parallel a b

def prop3 (p1 p2 : Type) (l1 : Type) [Plane p1] [Plane p2] [Line l1] (perpendicular_to_line : p1 → l1 → Prop) (parallel : p1 → p1 → Prop) :=
  ∀ {a b : p1}, a ≠ b → perpendicular_to_line a l1 → perpendicular_to_line b l1 → parallel a b

def prop4 (p1 p2 p3 : Type) [Plane p1] [Plane p2] [Plane p3] (perpendicular : p1 → p2 → Prop) :=
  ∀ {a b : p1}, a ≠ b → perpendicular a p3 → perpendicular b p3 → ⊥

-- Proof problem.
theorem number_incorrect_statements_is_two
  (h1 : ¬ prop1) (h2 : prop2) (h3 : prop3) (h4 : ¬ prop4) : 
  2 = 2 := sorry

end number_incorrect_statements_is_two_l755_755200


namespace average_speed_remaining_l755_755153

theorem average_speed_remaining (D : ℝ) : 
    (0.4 * D / 40 + 0.6 * D / S) = D / 50 → S = 60 :=
by 
  sorry

end average_speed_remaining_l755_755153


namespace shortest_distance_PQ_l755_755226

def f (x : ℝ) : ℝ := Real.exp x + Real.sin x
def g (x : ℝ) : ℝ := x - 2

theorem shortest_distance_PQ (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 < x2) (h3 : f x1 = g x2) : 
  dist (x2 - x1) = 3 :=
sorry

end shortest_distance_PQ_l755_755226


namespace cylinder_views_not_same_shape_and_size_l755_755185

def Solid := {s: Type} -- A type representing possible solids
def sphere: Solid := sorry
def cube: Solid := sorry
def triangular_pyramid: Solid := sorry
def cylinder: Solid := sorry

-- Define a predicate that checks if all three views (top, front, side) of a solid are the same
def views_same_shape_and_size (s: Solid) := sorry

-- The theorem statement, assuming that one of the four solids cannot have the same views
theorem cylinder_views_not_same_shape_and_size :
  ¬(views_same_shape_and_size cylinder) :=
sorry

end cylinder_views_not_same_shape_and_size_l755_755185


namespace volleyball_team_probability_l755_755622

open Finset

def total_players : ℕ := 9
def chosen_players : ℕ := 6
def john_peter_count : ℕ := 2
def remaining_players : ℕ := total_players - john_peter_count
def remaining_chosen : ℕ := chosen_players - john_peter_count

theorem volleyball_team_probability : 
  (nat.choose remaining_players remaining_chosen) / (nat.choose total_players chosen_players) = 5 / 12 := 
by
  sorry

end volleyball_team_probability_l755_755622


namespace increasing_size_37_affects_mode_l755_755648

def sales_data : List ℕ := [2, 8, 10, 6, 2]

theorem increasing_size_37_affects_mode :
  ∀ (new_quantity : ℕ), mode ([2, 8, new_quantity, 6, 2]) = new_quantity ∨ mode ([2, 8, new_quantity, 6, 2]) = 10 :=
by
  sorry

end increasing_size_37_affects_mode_l755_755648


namespace units_digit_3_2014_4_2015_5_2016_l755_755978

-- Definitions from the conditions:
def power_units_digit_cycle (base exp : ℕ) : ℕ :=
  match base % 10 with
  | 3 => [3, 9, 7, 1].get! ((exp - 1) % 4)
  | 4 => [4, 6].get! ((exp - 1) % 2)
  | 5 => 5
  | _ => 0 -- This covers other bases which we don't need for this problem.
  end

-- Question rewritten as Lean problem statement:
theorem units_digit_3_2014_4_2015_5_2016 :
  let units_digit (base exp : ℕ) := power_units_digit_cycle base exp in
  (units_digit 3 2014 + units_digit 4 2015 + units_digit 5 2016) % 10 = 8 :=
by {
  sorry
}

end units_digit_3_2014_4_2015_5_2016_l755_755978


namespace weight_gain_difference_l755_755212

theorem weight_gain_difference :
  let orlando_gain := 5
  let jose_gain := 2 * orlando_gain + 2
  let total_gain := 20
  let fernando_gain := total_gain - (orlando_gain + jose_gain)
  let half_jose_gain := jose_gain / 2
  half_jose_gain - fernando_gain = 3 :=
by
  sorry

end weight_gain_difference_l755_755212


namespace expression_value_l755_755112

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l755_755112


namespace library_study_time_l755_755599

theorem library_study_time
  (prob_no_overlap : 0.75)
  (total_interval : ℝ := 120)
  (n : ℝ)
  (d e f : ℕ)
  (sqrt_f_not_perfect_square : ¬ (∃ m : ℕ, m * m = f))
  (prob_meet : ℝ := 0.25)
  (n_eq_expr : n = d - e * Real.sqrt f)
  (prob_eq : (total_interval - n)^2 / (total_interval * total_interval) = prob_no_overlap) :
  n = 120 - 60 * Real.sqrt 3 := 
sorry

end library_study_time_l755_755599


namespace empty_seats_after_second_stop_l755_755973

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end empty_seats_after_second_stop_l755_755973


namespace integer_values_of_n_l755_755710

theorem integer_values_of_n :
  (∃ n : ℕ, ∀ n ∈ (-2:ℤ)..2, is_integer (7200 * (3/5)^n)) ↔ (∀ n : ℕ, n = 5) := 
sorry

end integer_values_of_n_l755_755710


namespace circumcenter_of_equilateral_lateral_edges_l755_755565

theorem circumcenter_of_equilateral_lateral_edges
  {P A B C H : Point}
  (height_eq : height P A B C = dist P H)
  (angles_eq : ∀ (X Y : Point), angle (P - X) (P - base X Y) = angle (P - Y) (P - base X Y)) :
  is_circumcenter H A B C :=
sorry

end circumcenter_of_equilateral_lateral_edges_l755_755565


namespace find_number_l755_755613

theorem find_number (x : ℤ) (N : ℤ) (h1 : 3 * x = (N - x) + 18) (hx : x = 11) : N = 26 :=
by
  sorry

end find_number_l755_755613


namespace chord_length_l755_755554

theorem chord_length {x y : ℝ} (hx : (x + 2)^2 + (y - 2)^2 = 1) (hline : x - y + 3 = 0) :
  ∃ l : ℝ, l = sqrt 2 :=
begin
  sorry
end

end chord_length_l755_755554


namespace find_speed_of_stream_l755_755150

-- Define the given conditions
def boat_speed_still_water : ℝ := 14
def distance_downstream : ℝ := 72
def time_downstream : ℝ := 3.6

-- Define the speed of the stream (to be proven)
def speed_of_stream : ℝ := 6

-- The statement of the problem
theorem find_speed_of_stream 
  (h1 : boat_speed_still_water = 14)
  (h2 : distance_downstream = 72)
  (h3 : time_downstream = 3.6)
  (speed_of_stream_eq : boat_speed_still_water + speed_of_stream = distance_downstream / time_downstream) :
  speed_of_stream = 6 := 
by 
  sorry

end find_speed_of_stream_l755_755150


namespace all_fruits_fallen_l755_755984

-- Definition of conditions
def magical_tree (n : ℕ) : Prop :=
  ∃ d : ℕ, d = 10 ∧ d > 0 ∧ (∀ k : ℕ, k ≤ d → (sum (range (k + 1)) ≤ n)) ∧ 
  (∀ m : ℕ, sum (range (m + 1)) = n → m = d - 1)

-- Theorem to prove the magical tree will be empty of fruits after 10 days
theorem all_fruits_fallen : magical_tree 46 := 
by {
  sorry
}

end all_fruits_fallen_l755_755984


namespace repeating_decimal_eq_fraction_l755_755343

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755343


namespace total_cost_correct_l755_755691

def cost_first_day : Nat := 4 + 5 + 3 + 2
def cost_second_day : Nat := 5 + 6 + 4
def total_cost : Nat := cost_first_day + cost_second_day

theorem total_cost_correct : total_cost = 29 := by
  sorry

end total_cost_correct_l755_755691


namespace correct_conclusions_l755_755960

theorem correct_conclusions :
  (∀ n : ℤ, n < -1 -> n < -1) ∧
  (¬ ∀ a : ℤ, abs (a + 2022) > 0) ∧
  (∀ a b : ℤ, a + b = 0 -> a * b < 0) ∧
  (∀ n : ℤ, abs n = n -> n ≥ 0) :=
sorry

end correct_conclusions_l755_755960


namespace flood_monitoring_technology_l755_755033

def geographicInformationTechnologies : Type := String

def RemoteSensing : geographicInformationTechnologies := "Remote Sensing"
def GlobalPositioningSystem : geographicInformationTechnologies := "Global Positioning System"
def GeographicInformationSystem : geographicInformationTechnologies := "Geographic Information System"
def DigitalEarth : geographicInformationTechnologies := "Digital Earth"

def effectiveFloodMonitoring (tech1 tech2 : geographicInformationTechnologies) : Prop :=
  (tech1 = RemoteSensing ∧ tech2 = GeographicInformationSystem) ∨ 
  (tech1 = GeographicInformationSystem ∧ tech2 = RemoteSensing)

theorem flood_monitoring_technology :
  effectiveFloodMonitoring RemoteSensing GeographicInformationSystem :=
by
  sorry

end flood_monitoring_technology_l755_755033


namespace repeating_decimal_equiv_fraction_l755_755288

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755288


namespace expression_value_l755_755114

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l755_755114


namespace mode_is_48_median_is_48_range_is_38_quantile_5_percent_is_26_l755_755155

def data_set : List ℕ := [63, 38, 25, 42, 56, 48, 53, 39, 28, 47, 45, 52, 59, 48, 41, 62, 48, 50, 52, 27]

-- Prove the mode is 48
theorem mode_is_48 : (data_set.filter (fun x => x = 48)).length > (data_set.filter (fun x => x = 42)).length :=
by sorry

-- Prove the median is 48
theorem median_is_48 : List.median data_set = 48 :=
by sorry

-- Prove the range is 38
theorem range_is_38 : (List.maximum data_set - List.minimum data_set) = 38 :=
by sorry 

-- Prove the 5% quantile is 26
theorem quantile_5_percent_is_26 : (List.nth_le data_set 0 (by linarith) + List.nth_le data_set 1 (by linarith)) / 2 = 26 :=
by sorry

end mode_is_48_median_is_48_range_is_38_quantile_5_percent_is_26_l755_755155


namespace ratio_of_speeds_l755_755090

theorem ratio_of_speeds (k r t V1 V2 : ℝ) (hk : 0 < k) (hr : 0 < r) (ht : 0 < t)
    (h1 : r * (V1 - V2) = k) (h2 : t * (V1 + V2) = k) :
    |r + t| / |r - t| = V1 / V2 :=
by
  sorry

end ratio_of_speeds_l755_755090


namespace repeating_fraction_equality_l755_755349

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755349


namespace ways_to_choose_socks_l755_755540

open Finset

theorem ways_to_choose_socks :
  let socks := { "blue1", "blue2", "brown", "black", "red", "purple", "green" }
  let choices := socks.subsets.filter (λ s, s.card = 4 ∧ 2 ≤ s.count "blue1" + s.count "blue2")
  choices.card = 30 :=
by
  sorry

end ways_to_choose_socks_l755_755540


namespace probability_ratio_l755_755722

-- Definitions based on conditions
def numBalls : Nat := 24
def bins : Nat := 4
def binCount : Nat := 6

-- Theorem that needs to be proven
theorem probability_ratio :
  let A_num_ways := Nat.choose (numBalls) [binCount, binCount, binCount, binCount] in
  let B_num_ways := Nat.choose (numBalls) [binCount, binCount, binCount, binCount] in
  let p' := A_num_ways / (numBalls ^ bins) in
  let q' := B_num_ways / (numBalls ^ bins) in
  p' = q' :=
by
  -- Placeholder for proof
  sorry

end probability_ratio_l755_755722


namespace permutation_product_equals_permutation_l755_755143

-- Definitions used in Lean 4 statement
def permutation_product (n k : ℕ) : ℕ := ∏ i in range (n - k + 1), n - i

def permutation (n k : ℕ) : ℕ := n! / (n - k)!

-- Conditions translated to Lean definitions
def condition1 : ℕ := 18
def condition2 : ℕ := 10

-- Proof statement
theorem permutation_product_equals_permutation :
  permutation_product condition1 condition2 = permutation condition1 condition2 :=
sorry

end permutation_product_equals_permutation_l755_755143


namespace CombinedPointsFirstTwoRounds_l755_755238

-- Define the scores of players A and B
def PlayerAScores (a t s : ℕ) : list ℕ := [a, a + t + s, a + 2 * t + 4 * s, a + 3 * t + 9 * s]
def PlayerBScores (b d : ℕ) : list ℕ := [b, b + d, b + 2 * d, b + 3 * d]

-- Define the total scores
def TotalScore (scores : list ℕ) : ℕ := scores.sum

-- The conditions given in the problem
def Conditions (a t s b d : ℕ) : Prop :=
  let A := TotalScore (PlayerAScores a t s)
  let B := TotalScore (PlayerBScores b d)
  A = B ∧ A ≤ 25 ∧ B ≤ 25

-- The proposition to prove
theorem CombinedPointsFirstTwoRounds (a t s b d : ℕ) (h : Conditions a t s b d) :
  (PlayerAScores a t s).take 2.sum + (PlayerBScores b d).take 2.sum = 12 :=
by
  sorry

end CombinedPointsFirstTwoRounds_l755_755238


namespace gain_percentage_of_second_book_l755_755829

theorem gain_percentage_of_second_book
  (C1 C2 SP1 SP2 : ℝ)
  (h1 : C1 + C2 = 470)
  (h2 : C1 = 274.1666666666667)
  (h3 : SP1 = C1 * 0.85)
  (h4 : SP1 = SP2) :
  C2 = 470 - 274.1666666666667 →
  (G : ℝ) →
  SP2 = C2 * (1 + G / 100) →
  G ≈ 19 :=
by
  sorry

end gain_percentage_of_second_book_l755_755829


namespace main_theorem_l755_755895

-- Define polynomial of degree k
def polynomial (k : ℕ) (p : ℤ → ℤ) : Prop :=
  ∃ c : (fin (k+1) → ℤ), ∀ x : ℤ, p x = ∑ i in finset.range (k+1), c i * (binomial x i)

-- Define the condition that p_k(x) takes integer values at specific points
def takes_integer_values (p : ℤ → ℤ) (n : ℤ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ k → ∃ m : ℤ, p (n + i) = m

-- The main theorem
theorem main_theorem (k : ℕ) (p : ℤ → ℤ) (n : ℤ)
  (hk : polynomial k p) (h_values : takes_integer_values p n k) :
  ∃ c : (fin (k+1) → ℤ), ∀ x : ℤ, p x = ∑ i in finset.range (k+1), c i * (binomial x (k - i)) := 
sorry

end main_theorem_l755_755895


namespace green_fish_count_l755_755902

theorem green_fish_count (B O G : ℕ) (H1 : B = 40) (H2 : O = B - 15) (H3 : 80 = B + O + G) : G = 15 := 
by 
  sorry

end green_fish_count_l755_755902


namespace total_amount_distributed_l755_755682

def number_of_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem total_amount_distributed : (number_of_persons * amount_per_person) = 42900 := by
  sorry

end total_amount_distributed_l755_755682


namespace find_value_l755_755824

variable (x : ℝ)
variable (hx : x - sqrt (x^2 + 1) + 1 / (x + sqrt (x^2 + 1)) = 28)

theorem find_value :
  x^2 - sqrt (x^4 + 1) + 1 / (x^2 - sqrt (x^4 + 1)) = -2 * sqrt 38026 := 
sorry

end find_value_l755_755824


namespace length_of_BC_l755_755874

/-- In triangle ABC, points M and N are on segments AB and AC respectively, such that AM = MC and AN = NB. 
   Let P be the point such that PB and PC are tangent to the circumcircle of ABC. 
   Given that the perimeters of PMN and quadrilateral BCNM are 21 and 29 respectively, 
   and that PB = 5, prove that the length of BC is 200/21. --/
theorem length_of_BC {A B C P M N : Point} 
  (AM_eq_MC : dist A M = dist M C) 
  (AN_eq_NB : dist A N = dist N B) 
  (PB_tangent : tangent_to_circumcircle P B (circumcircle ⟨A, B, C⟩))
  (PC_tangent : tangent_to_circumcircle P C (circumcircle ⟨A, B, C⟩))
  (perimeter_PMN : perimeter ⟨P, M, N⟩ = 21)
  (perimeter_BCNM : perimeter ⟨B, C, N, M⟩ = 29)
  (PB_eq_5 : dist P B = 5) : 
  dist B C = 200 / 21 := sorry

end length_of_BC_l755_755874


namespace cylinder_cone_dimensions_l755_755664

theorem cylinder_cone_dimensions (r m : ℝ) 
    (h1 : 2 * π * r * m / (π * r * (sqrt (m^2 + r^2))) = 8 / 5)
    (h2 : r * m = 588) :
    r = 21 ∧ m = 28 := by
  sorry

end cylinder_cone_dimensions_l755_755664


namespace first_day_more_than_300_l755_755015

def paperclips (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_more_than_300 : ∃ n, paperclips n > 300 ∧ n = 4 := by
  sorry

end first_day_more_than_300_l755_755015


namespace infinite_planes_divide_regular_tetrahedron_l755_755231

-- Define a regular tetrahedron
structure RegularTetrahedron :=
(centroid : Point)
(vertices : Fin 4 → Point)

-- Define a plane
structure Plane :=
(point : Point)
(normal : Vector)

-- Define a property of a plane dividing a tetrahedron into two identical bodies
def divides_into_identical_bodies (T : RegularTetrahedron) (P : Plane) : Prop :=
  -- Dummy definition, replace with actual mathematical property
  sorry

theorem infinite_planes_divide_regular_tetrahedron (T : RegularTetrahedron) :
  ∃ (P : Plane) (count : ℵ₀), divides_into_identical_bodies(T, P) :=
by
  -- Dummy proof
  sorry

end infinite_planes_divide_regular_tetrahedron_l755_755231


namespace coords_of_A_l755_755465

theorem coords_of_A :
  ∃ (x y : ℝ), y = Real.exp x ∧ (Real.exp x = 1) ∧ y = 1 :=
by
  use 0, 1
  have hx : Real.exp 0 = 1 := Real.exp_zero
  have hy : 1 = Real.exp 0 := hx.symm
  exact ⟨hy, hx, rfl⟩

end coords_of_A_l755_755465


namespace toy_blocks_total_l755_755908

theorem toy_blocks_total :
  let stack1 := 5 in
  let stack2 := stack1 + 2 in
  let stack3 := stack2 - 5 in
  let stack4 := stack3 + 5 in
  stack1 + stack2 + stack3 + stack4 = 21 :=
by
  sorry

end toy_blocks_total_l755_755908


namespace find_numbers_l755_755956

-- Define the conditions
def condition_1 (L S : ℕ) : Prop := L - S = 8327
def condition_2 (L S : ℕ) : Prop := ∃ q r, L = q * S + r ∧ q = 21 ∧ r = 125

-- Define the math proof problem
theorem find_numbers (S L : ℕ) (h1 : condition_1 L S) (h2 : condition_2 L S) : S = 410 ∧ L = 8735 :=
by
  sorry

end find_numbers_l755_755956


namespace truncated_cone_volume_correct_l755_755720

-- Definitions based on conditions
def radius1 : ℝ := 2.0
def radius2 : ℝ := 4.0
def height1 : ℝ := (4.0 ^ 2 - 2.0 ^ 2).sqrt
def height2 : ℝ := (2.0 ^ 2 - 1.0 ^ 2).sqrt
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r ^ 2 * h

-- Volumes of the two cones
def volume_larger_cone : ℝ := volume_cone radius2 height1
def volume_smaller_cone : ℝ := volume_cone radius1 height2

-- Volume of the truncated cone
def truncated_cone_volume : ℝ := volume_larger_cone - volume_smaller_cone

theorem truncated_cone_volume_correct : 
  abs (truncated_cone_volume - 12.697) < 0.001 :=
by
  sorry

end truncated_cone_volume_correct_l755_755720


namespace average_of_multiples_of_10_from_10_to_500_l755_755094

theorem average_of_multiples_of_10_from_10_to_500 : 
  let multiples : List ℕ := List.range' 10 (50 * 10 + 1) 10 in
  multiples.sum / multiples.length = 255 :=
by
  let multiples : List ℕ := List.range' 10 (50 * 10 + 1) 10
  have h : multiples.length = 50 := by sorry
  have sum_eq : multiples.sum = 12750 := by sorry
  calc 
    multiples.sum / multiples.length = 12750 / 50 := by sorry
    ... = 255 := by norm_num

end average_of_multiples_of_10_from_10_to_500_l755_755094


namespace Katrina_sold_in_morning_l755_755500

theorem Katrina_sold_in_morning :
  ∃ M : ℕ, (120 - 57 - 16 - 11) = M := sorry

end Katrina_sold_in_morning_l755_755500


namespace find_initial_apples_l755_755969

def initial_apples (A : ℕ) : Prop :=
  ((A - 20) + 6 = 9) → (A = 23)

theorem find_initial_apples : ∃ A : ℕ, initial_apples A :=
begin
  use 23,
  simp [initial_apples],
  intro h,
  have h1 : (23 - 20) + 6 = 9, from h,
  exact h1,
end

end find_initial_apples_l755_755969


namespace num_divisors_of_square_product_l755_755655

theorem num_divisors_of_square_product (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_distinct : p ≠ q) :
  let n := p^2 * q^2 in
  let n_squared := n^2 in
  let num_divisors := (4 + 1) * (4 + 1) in
  number_of_divisors n_squared = num_divisors :=
sorry

end num_divisors_of_square_product_l755_755655


namespace function_domain_l755_755605

-- Define the functions required for the problem
def log_base (b x : ℝ) : ℝ := log x / log b

-- Function definition based on the problem statement
def f (x : ℝ) : ℝ := log_base 5 (log_base 7 (log_base 4 x - 1))

-- Formal statement of the problem
theorem function_domain :
  ∀ x : ℝ, 16 < x → ∃ y : ℝ, f x = y :=
by
  intro x
  intro h
  -- Rest of the proof skipped
  sorry

end function_domain_l755_755605


namespace equilateral_triangle_circumcircle_area_l755_755684

/-- Given an equilateral triangle DEF with DE = DF = EF = 8 units, 
    and a circle with radius 4 units tangent to DE at E and DF at F,
    prove that the area of the circumcircle that passes through D, E, and F is 64π/3 units². -/
theorem equilateral_triangle_circumcircle_area :
  ∀ (D E F : Point) (r : ℝ),
  is_equilateral_triangle D E F ∧
  distance D E = 8 ∧
  distance D F = 8 ∧
  distance E F = 8 ∧
  is_tangent_circle D E F r 4 → 
  circle_area (circumcircle D E F) = 64 * π / 3 :=
by {
  intros D E F r h_cond,
  sorry
}

end equilateral_triangle_circumcircle_area_l755_755684


namespace repeating_decimal_as_fraction_l755_755307

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755307


namespace repeating_decimal_is_fraction_l755_755291

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755291


namespace midpoints_slope_is_zero_l755_755610

def point := (ℚ × ℚ)

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def slope (p1 p2 : point) : ℚ :=
  if p1.1 = p2.1 then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

def segment1_start : point := (0, 0)
def segment1_end : point := (2, 3)
def segment2_start : point := (5, 0)
def segment2_end : point := (6, 3)

def midpoint1 : point := midpoint segment1_start segment1_end
def midpoint2 : point := midpoint segment2_start segment2_end

theorem midpoints_slope_is_zero : slope midpoint1 midpoint2 = 0 :=
  sorry

end midpoints_slope_is_zero_l755_755610


namespace distance_P_to_l_l755_755790

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (direction : Point3D)
  (point : Point3D)

noncomputable def distance_point_to_line (P A : Point3D) (direction : Point3D) : ℝ :=
  let PA : Point3D := ⟨A.x - P.x, A.y - P.y, A.z - P.z⟩ in
  let m_dot_PA : ℝ := direction.x * PA.x + direction.y * PA.y + direction.z * PA.z in
  let m_norm_square : ℝ := direction.x ^ 2 + direction.y ^ 2 + direction.z ^ 2 in
  let PA_norm_square : ℝ := PA.x ^ 2 + PA.y ^ 2 + PA.z ^ 2 in
  let cos_theta : ℝ := m_dot_PA / (Real.sqrt m_norm_square * Real.sqrt PA_norm_square) in
  let sin_theta_square : ℝ := 1 - cos_theta ^ 2 in
  Real.sqrt PA_norm_square * Real.sqrt sin_theta_square

def P : Point3D := ⟨-1, 1, -1⟩
def A : Point3D := ⟨4, 1, -2⟩
def m : Point3D := ⟨1, Real.sqrt 2, -1⟩

theorem distance_P_to_l : distance_point_to_line P A m = Real.sqrt 17 := by
  sorry

end distance_P_to_l_l755_755790


namespace floor_of_3_7_l755_755693

theorem floor_of_3_7 : Int.floor 3.7 = 3 := by
  sorry

end floor_of_3_7_l755_755693


namespace solution_to_inequality_l755_755253

theorem solution_to_inequality (x : ℝ) : 
  ((x^2 / ((x - 3) * (x + 2)) ≥ 0) ↔ (x ∈ Ioo (-∞) (-2) ∪ Icc (-2) (-2) ∪ Ioc 3 (∞))) := sorry

end solution_to_inequality_l755_755253


namespace taras_total_gas_spent_is_180_l755_755943

def trip_duration := 2 -- in days
def gas_stations := 4 -- number of gas stations visited
def gas_prices := [3.0, 3.5, 4.0, 4.5] -- price per gallon at each gas station
def tank_capacity := 12.0 -- tank capacity in gallons

def total_gas_spent : ℝ :=
  (tank_capacity * gas_prices[0]) +
  (tank_capacity * gas_prices[1]) +
  (tank_capacity * gas_prices[2]) +
  (tank_capacity * gas_prices[3])

theorem taras_total_gas_spent_is_180 :
  total_gas_spent = 180 :=
by
  sorry

end taras_total_gas_spent_is_180_l755_755943


namespace find_m_l755_755801

open Real

-- Define the circle equation
def circle_eq (x y m : ℝ) := x^2 + y^2 - 2 * x - 4 * y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) := x + 2 * y - 4 = 0

-- Define the midpoint of the circle
def is_center (x y : ℝ) := (x - 1)^2 + (y - 2)^2

-- Define the radius of the circle
def radius (m : ℝ) := sqrt (5 - m)

-- Define the point-to-line distance
def distance_to_line (x y : ℝ) := abs (x + 2 * y - 4) / sqrt (1^2 + 2^2)

-- The main theorem to prove:
theorem find_m (x y m : ℝ) (hC : circle_eq x y m) (hl : line_eq x y)
  (hMN : (radius m)^2 = distance_to_line 1 2^2 + (2 / sqrt 5)^2)
  : m = 4 :=
sorry

end find_m_l755_755801


namespace cost_of_fencing_per_meter_l755_755569

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end cost_of_fencing_per_meter_l755_755569


namespace log_base_change_l755_755747

theorem log_base_change 
  (b c : ℝ)
  (hb : log 10 3 = b) 
  (hc : log 10 5 = c) 
  : log 5 45 = (2 * b + c) / c := 
by 
  sorry

end log_base_change_l755_755747


namespace one_div_i_plus_i_pow_2015_eq_neg_two_i_l755_755753

def is_imaginary_unit (x : ℂ) : Prop := x * x = -1

theorem one_div_i_plus_i_pow_2015_eq_neg_two_i (i : ℂ) (h : is_imaginary_unit i) : 
  (1 / i + i ^ 2015) = -2 * i :=
sorry

end one_div_i_plus_i_pow_2015_eq_neg_two_i_l755_755753


namespace steve_speed_ratio_l755_755957

theorem steve_speed_ratio :
  let d := 35 -- The distance from Steve's house to work in km
  let t := 6 -- Total time spent on the roads in hours
  let v2 := 17.5 -- Speed on the way back in km/h
  ∃ v1 : ℝ, (d / v1) + (d / v2) = t ∧ (v2 / v1) = 2 := 
begin
  sorry
end

end steve_speed_ratio_l755_755957


namespace sum_arithmetic_sequence_mod_13_l755_755101

theorem sum_arithmetic_sequence_mod_13 : 
  (∑ k in Finset.range 12, (2357 + k) % 13) % 13 = 5 := by
  sorry

end sum_arithmetic_sequence_mod_13_l755_755101


namespace perimeter_of_shape_l755_755556

-- Define the shape constructed from 10 squares with 1cm side length.
def Square := {side_length : ℕ // side_length = 1}
def Shape := {squares : ℕ // squares = 10 ∧ squares *= side_length ∧ permutation equality}

-- State the theorem to prove the perimeter of the shape is 18 cm.
theorem perimeter_of_shape {s : Shape} :  (∃ perimeter < ℕ),
  perimeter = 18 := sorry

end perimeter_of_shape_l755_755556


namespace octagon_midpoints_area_l755_755663

theorem octagon_midpoints_area (side_length : ℝ) (h : side_length = 12) :
  ∃ (area : ℝ), area = 576 + 288 * Real.sqrt 2 := 
by
  exist sorry

end octagon_midpoints_area_l755_755663


namespace finite_S_k_iff_k_is_power_of_two_l755_755739

/-- Conditions for tuples in the set S_k -/
def S_k (k : ℕ) : set (ℕ × ℕ × ℕ) := 
  {t | let (n, a, b) := t in 
      n % 2 = 1 ∧ 
      Nat.gcd a b = 1 ∧ 
      a + b = k ∧ 
      n ∣ (a ^ n + b ^ n)}

/-- The main theorem stating S_k is finite if and only if k is a power of 2 -/
theorem finite_S_k_iff_k_is_power_of_two (k : ℕ) (h : k > 1) : 
  (set.finite (S_k k)) ↔ ∃ α : ℕ, k = 2 ^ α ∧ α > 0 :=
sorry

end finite_S_k_iff_k_is_power_of_two_l755_755739


namespace solve_for_x_l755_755236

theorem solve_for_x (x : ℝ) (h : sqrt (3 * sqrt (x - 3)) = nthRoot (10 - x) 4) : x = 37 / 10 := 
by
  sorry

end solve_for_x_l755_755236


namespace problem_A_B_l755_755887

open Set

theorem problem_A_B {A B : Set ℝ} (hA_nonempty : A ≠ ∅) (hB_nonempty: B ≠ ∅)
  (hA : A = {x : ℝ | 0 ≤ x ∧ x ≤ 3})
  (hB : B = {y : ℝ | y ≥ 1}) :
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B} = {x : ℝ | (0 ≤ x ∧ x < 1) ∨ (3 < x ∧ x ∈ Iio ∞)} :=
by
  sorry

end problem_A_B_l755_755887


namespace arithmetic_mean_of_successive_integers_l755_755214

theorem arithmetic_mean_of_successive_integers (S : ℕ → ℕ) (n m : ℕ) :
  S 1 = 5 → (∀ k, 1 ≤ k ∧ k ≤ 42 → S k = S 1 + (k - 1)) →
  (m = 42) →
  (real_arithmetic_mean : ℝ) := 
by 
  have sum := sum_of_sequence S S 1 S m 
  have mean := sum / 42
  sorry


end arithmetic_mean_of_successive_integers_l755_755214


namespace polygon_sides_l755_755796

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end polygon_sides_l755_755796


namespace square_area_condition_l755_755194

theorem square_area_condition (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (hline : x1 ≠ x2) : 
  let side := abs (x1 - x2) in side^2 = 36 :=
by
  sorry

end square_area_condition_l755_755194


namespace cubic_box_dimension_l755_755759

theorem cubic_box_dimension (a : ℤ) (h: 12 * a = 3 * (a^3)) : a = 2 :=
by
  sorry

end cubic_box_dimension_l755_755759


namespace shaded_fraction_correct_l755_755537

-- Define the regular octagon and its properties
structure RegularOctagon (O : Point) :=
(vertex : Fin 8 → Point)
(center_eqd : ∀ (i j : Fin 8), dist O (vertex i) = dist O (vertex j))
(vertex_angle : ∀ (i : Fin 8), angle O (vertex i) (vertex ((i + 1) % 8)) = π / 4)

-- Define the point Y dividing BC in the ratio 1:3
structure DividingPoint (B C Y : Point) :=
(ratio : dist B Y / dist B C = 1 / 4)

def shaded_area_fraction (O : Point) (oct : RegularOctagon O) (B C Y : Point) (div : DividingPoint B C Y) : ℚ :=
  17 / 32

theorem shaded_fraction_correct (O : Point) (oct : RegularOctagon O) (B C Y : Point) (div : DividingPoint B C Y) :
  shaded_area_fraction O oct B C Y div = 17 / 32 :=
sorry

end shaded_fraction_correct_l755_755537


namespace repeating_decimal_equiv_fraction_l755_755285

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755285


namespace p_or_q_sufficient_but_not_necessary_l755_755977

open Classical

theorem p_or_q_sufficient_but_not_necessary (p q : Prop) : 
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  split
  -- Proof of sufficiency (p ∧ q → p ∨ q)
  · intro h
    cases h
    left
    assumption
  · intro h
    apply not_imp_not.mpr
    intro h'
    exact false.elim (h.mpr h')
-- This statement is used to complete compilation without proof.
-- Proofs can be added as stated in the steps.
sorry

end p_or_q_sufficient_but_not_necessary_l755_755977


namespace problem1_problem2_l755_755207

open Real EuclideanGeometry Geometry

theorem problem1 (A B C I M N P T : Point)
  (hA_lt_B : angle A B C < angle B A C)
  (hGamma : Circumcircle (Triangle.mk A B C) = Γ)
  (hM : IsMidpoint M (Arc.mk B C Γ))
  (hN : IsMidpoint N (Arc.mk A C Γ))
  (hPC_parallel_MN : Parallel (Line.mk P C) (Line.mk M N))
  (hGamma_P : OnCircle P Γ)
  (hI : Incenter I (Triangle.mk A B C))
  (hT : OnCircle T Γ ∧ (Line.mk P I).meets T) :
  length (Segment.mk M P) * length (Segment.mk M T) = length (Segment.mk N P) * length (Segment.mk N T) :=
sorry

theorem problem2 (A B C Q I₁ I₂ T : Point)
  (h_A_lt_B : angle A B C < angle B A C)
  (hGamma : Circumcircle (Triangle.mk A B C) = Γ)
  (hQ : OnArc Q (Arc.mk A B Γ) ∧ Q ≠ B ∧ Q ≠ A ∧ Q ≠ C ∧ Q ≠ T)
  (hI₁ : Incenter I₁ (Triangle.mk A Q C))
  (hI₂ : Incenter I₂ (Triangle.mk Q C B))
  (hT : OnCircle T Γ) :
  Concyclic Q I₁ I₂ T :=
sorry

end problem1_problem2_l755_755207


namespace distance_P_to_l_l755_755791

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (direction : Point3D)
  (point : Point3D)

noncomputable def distance_point_to_line (P A : Point3D) (direction : Point3D) : ℝ :=
  let PA : Point3D := ⟨A.x - P.x, A.y - P.y, A.z - P.z⟩ in
  let m_dot_PA : ℝ := direction.x * PA.x + direction.y * PA.y + direction.z * PA.z in
  let m_norm_square : ℝ := direction.x ^ 2 + direction.y ^ 2 + direction.z ^ 2 in
  let PA_norm_square : ℝ := PA.x ^ 2 + PA.y ^ 2 + PA.z ^ 2 in
  let cos_theta : ℝ := m_dot_PA / (Real.sqrt m_norm_square * Real.sqrt PA_norm_square) in
  let sin_theta_square : ℝ := 1 - cos_theta ^ 2 in
  Real.sqrt PA_norm_square * Real.sqrt sin_theta_square

def P : Point3D := ⟨-1, 1, -1⟩
def A : Point3D := ⟨4, 1, -2⟩
def m : Point3D := ⟨1, Real.sqrt 2, -1⟩

theorem distance_P_to_l : distance_point_to_line P A m = Real.sqrt 17 := by
  sorry

end distance_P_to_l_l755_755791


namespace leo_current_weight_l755_755620

theorem leo_current_weight (L K : ℕ) 
  (h1 : L + 10 = 3 * K / 2) 
  (h2 : L + K = 160)
  : L = 92 :=
sorry

end leo_current_weight_l755_755620


namespace correct_statement_l755_755411

variable {α : Type} [LinearOrderedField α]

-- Conditions
variable (f : α → α)
variable (a b x : α)
variable (h_sym : ∀ x, f (-x) = f x)
variable (h_range : ∀ y ∈ set.Icc a b, min (f a) (f b) ≤ f y ∧ f y ≤ max (f a) (f b))

-- Definition using the conditions
def axis_of_symmetry := ∀ x, f (-x) = f x
def range_in_interval := ∀ y ∈ set.Icc a b, min (f a) (f b) ≤ f y ∧ f y ≤ max (f a) (f b)

-- Problem Statement
theorem correct_statement :
  axis_of_symmetry f → range_in_interval f a b → x ∉ set.Ioo a b :=
by
  intros h_sym h_range
  sorry 

end correct_statement_l755_755411


namespace not_divisible_l755_755923

theorem not_divisible {x y : ℕ} (hx : x > 0) (hy : y > 2) : ¬ (2^y - 1) ∣ (2^x + 1) := sorry

end not_divisible_l755_755923


namespace total_yen_l755_755677

-- Define the given conditions in Lean 4
def bal_bahamian_dollars : ℕ := 5000
def bal_us_dollars : ℕ := 2000
def bal_euros : ℕ := 3000

def exchange_rate_bahamian_to_yen : ℝ := 122.13
def exchange_rate_us_to_yen : ℝ := 110.25
def exchange_rate_euro_to_yen : ℝ := 128.50

def check_acc1 : ℕ := 15000
def check_acc2 : ℕ := 6359
def sav_acc1 : ℕ := 5500
def sav_acc2 : ℕ := 3102

def stocks : ℕ := 200000
def bonds : ℕ := 150000
def mutual_funds : ℕ := 120000

-- Prove the total amount of yen the family has
theorem total_yen : 
  bal_bahamian_dollars * exchange_rate_bahamian_to_yen + 
  bal_us_dollars * exchange_rate_us_to_yen + 
  bal_euros * exchange_rate_euro_to_yen
  + (check_acc1 + check_acc2 + sav_acc1 + sav_acc2 : ℝ)
  + (stocks + bonds + mutual_funds : ℝ) = 1716611 := 
by
  sorry

end total_yen_l755_755677


namespace partI_inequality_solution_partII_minimum_value_l755_755431

-- Part (I)
theorem partI_inequality_solution (x : ℝ) : 
  (abs (x + 1) + abs (2 * x - 1) ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 1) :=
sorry

-- Part (II)
theorem partII_minimum_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  (∀ a b c : ℝ, a + b + c = 2 ->  a > 0 -> b > 0 -> c > 0 -> 
    (1 / a + 1 / b + 1 / c) = (9 / 2)) :=
sorry

end partI_inequality_solution_partII_minimum_value_l755_755431


namespace smallest_positive_b_l755_755940

def p (g : ℝ → ℝ) := ∀ x, g (x - 30) = g x

theorem smallest_positive_b (g : ℝ → ℝ) (h : p g) : ∃ b > 0, (∀ x, g (x / 10 - b / 10) = g (x / 10)) ∧ b = 300 :=
by 
  use 300
  split
  · exact zero_lt_three_hundred
  · intro x
    specialize h (x / 10)
    rw [sub_sub_cancel] at h
    exact h
  sorry

end smallest_positive_b_l755_755940


namespace solve_for_y_l755_755934

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l755_755934


namespace ratio_of_numbers_l755_755464

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 33) (h2 : x = 22) : y / x = 1 / 2 :=
by
  sorry

end ratio_of_numbers_l755_755464


namespace part_a_part_b_part_c_l755_755643

noncomputable theory

namespace BirthProbability

open Classical

-- Define the various facts and probabilities.
def equally_probable := (1/2 : ℝ)

def probability_one_boy_one_girl : ℝ := 1/2
def probability_given_one_is_boy : ℝ := 2/3
def probability_given_boy_born_on_monday : ℝ := 14/27

-- The first problem
theorem part_a
  (B G : Type)
  [fintype (B × G)]
  (h : ∀ bg : B × G, equally_probable) :
  (probability_one_boy_one_girl = 1/2) :=
sorry

-- The second problem
theorem part_b
  (B G : Type)
  [fintype (B × G)]
  (h : ∀ bg : B × G, (bg.fst = “boy” ∨ bg.snd = “boy”) ∧ equally_probable) :
  (probability_given_one_is_boy = 2/3) :=
sorry

-- The third problem
theorem part_c
  (B G : Type)
  [fintype (B × G)]
  (h : ∀ bg : B × G, (bg.fst = “boy” ∧ bg.snd.born_on(monday) ∧ equally_probable) ∨ (bg.snd = “boy” ∧ bg.snd.born_on(monday) ∧ equally_probable)) :
  (probability_given_boy_born_on_monday = 14/27) :=
sorry -- Placeholder for the detailed complex proof
end BirthProbability

end part_a_part_b_part_c_l755_755643


namespace evaluate_g_at_3_l755_755834

def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 200 := by
  sorry

end evaluate_g_at_3_l755_755834


namespace actual_average_height_correct_l755_755947

noncomputable def actualAverageHeight : ℝ :=
  let numStudents := 50
  let initialAverageHeight := 175
  let totalInitialHeight := initialAverageHeight * numStudents
  let incorrectHeights := [151, 162, 185]
  let actualHeights := [136, 174, 169]
  let totalDifference := list.sum (list.zip_with (-) incorrectHeights actualHeights)
  let correctTotalHeight := totalInitialHeight - totalDifference
  correctTotalHeight / numStudents

theorem actual_average_height_correct :
  actualAverageHeight = 174.62 :=
by
  sorry

end actual_average_height_correct_l755_755947


namespace repeating_decimal_equiv_fraction_l755_755284

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755284


namespace correct_function_decreasing_on_positive_real_range_of_t_l755_755399

section
variable (f : ℝ → ℝ) (a b c t : ℝ)

-- Given conditions
def is_odd_function := ∀ x, f (-x) = -f x
def condition_A := f 1 = 1
def condition_B := f 2 = -1

-- Define the function based on the problem
def defined_function := ∀ x, f x = -x + 2 / x

-- Question 1: Check that f(x) is defined correctly
theorem correct_function : condition_A ∧ condition_B :=
by
  unfold condition_A condition_B
  simp [defined_function]
  sorry

-- Question 2: Prove f(x) is decreasing on (0, +∞)
theorem decreasing_on_positive_real : ∀ x : ℝ, 0 < x → (f' x < 0) :=
by
  unfold defined_function
  have h : ∀ x : ℝ, (f' x = -1 - (2 / x ^ 2)), by sorry
  rw h
  intros x hx
  linarith
  sorry

-- Question 3: Find the range of t given the conditions
theorem range_of_t : (∀ x ∈ [-2, -1] ∨ x ∈ [1, 2], |t - 1| ≤ f x + 2) → (t ∈ [0, 2]) :=
by
  unfold defined_function
  sorry

end

end correct_function_decreasing_on_positive_real_range_of_t_l755_755399


namespace range_of_a_l755_755808

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else -3*x

theorem range_of_a {a : ℝ} (h1 : a ≠ 0) (h2 : a * (f a - f (-a)) > 0) : a ∈ set.Ioc (-∞) (-2) ∪ set.Ioc 2 ∞ :=
sorry

end range_of_a_l755_755808


namespace min_b_for_quadratic_factorization_l755_755377

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end min_b_for_quadratic_factorization_l755_755377


namespace approximate_values_l755_755686

/-- Approximate 0.30105 to the nearest hundredth and to 3 significant figures -/
theorem approximate_values :
  (approximate_to_nearest_hundredth (0.30105) = 0.30) ∧
  (approximate_to_significant_figures (0.30105) (3) = 0.301) :=
by
  sorry

-- Ensure that the imported functions approximate_to_nearest_hundredth and
-- approximate_to_significant_figures adhere to the rules specified.
def approximate_to_nearest_hundredth (x : ℝ) : ℝ :=
  -- Placeholder for actual implementation
  if x = 0.30105 then 0.30 else 0.0

def approximate_to_significant_figures (x : ℝ) (n : ℕ) : ℝ :=
  -- Placeholder for actual implementation
  if x = 0.30105 ∧ n = 3 then 0.301 else 0.0

end approximate_values_l755_755686


namespace cone_max_volume_l755_755983

noncomputable def maximize_cone_height : ℝ :=
let slant_height := 18 in
let max_height := 6 * Real.sqrt 3 in
max_height

theorem cone_max_volume (h : ℝ) (V : ℝ) :
  (∀ r : ℝ, r^2 + h^2 = 18^2) →
  V = (1/3) * Real.pi * r^2 * h →
  h = 6 * Real.sqrt 3 :=
by
  sorry

end cone_max_volume_l755_755983


namespace count_four_digit_integers_with_thousands_digit_3_l755_755440

theorem count_four_digit_integers_with_thousands_digit_3 : 
  (finset.card (finset.filter (λ n : ℕ, 3000 ≤ n ∧ n < 4000) (finset.range 10000))) = 1000 :=
by
  sorry

end count_four_digit_integers_with_thousands_digit_3_l755_755440


namespace temperature_difference_l755_755563

-- Define variables for the highest and lowest temperatures.
def highest_temp : ℤ := 18
def lowest_temp : ℤ := -2

-- Define the statement for the maximum temperature difference.
theorem temperature_difference : 
  highest_temp - lowest_temp = 20 := 
by 
  sorry

end temperature_difference_l755_755563


namespace discriminant_of_trinomial_l755_755019

theorem discriminant_of_trinomial (x1 x2 : ℝ) (h : x2 - x1 = 2) : (x2 - x1)^2 = 4 :=
by
  sorry

end discriminant_of_trinomial_l755_755019


namespace standard_equation_of_gamma_sum_of_t1_t2_l755_755765

def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Definitions for given points
def M : ℝ × ℝ := (-2, 1)
def F : ℝ × ℝ := (sqrt 3, 0)
def N : ℝ × ℝ := (1, 0)

-- Given conditions
variables {a b : ℝ} (h₀ : a > b) (h₁ : b > 0)
def gamma (x y : ℝ) := ellipse x y a b

-- Conditions from the problem
def passes_through_M : gamma M.1 M.2 := 
by { simp [gamma, ellipse, M], sorry }

def focus_is_F := 
by { simp [F], sorry }

-- The standard equation needs to be proven
theorem standard_equation_of_gamma : 
    (a ^ 2 = 6) ∧ (b ^ 2 = 3) :=
by { sorry }

-- Definitions to capture vectors and dot products
def t (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ + 2) * (x₂ + 2) + (y₁ - 1) * (y₂ - 1)

-- Maximum and minimum values of t
def t1 : ℝ := sorry
def t2 : ℝ := sorry

-- Main theorem to prove
theorem sum_of_t1_t2 (t1 t2 : ℝ) :
    t1 + t2 = 13 / 2 :=
by { sorry }

end standard_equation_of_gamma_sum_of_t1_t2_l755_755765


namespace probability_closer_to_5_l755_755660

-- Define the segment [0, 6]
def segment : set ℝ := {x | 0 ≤ x ∧ x ≤ 6}

-- Define the equidistant midpoint between 1 and 5
def midpoint : ℝ := (1 + 5) / 2

-- Define the subset of the segment where the points are closer to 5 than to 1
def closer_to_5 : set ℝ := {x | 3 < x ∧ x ≤ 6}

-- Use measure_theory to define the length (Lebesgue measure) of the segments
open MeasureTheory

-- The probability that a randomly selected point from [0, 6] is closer to 5 than to 1
theorem probability_closer_to_5 : 
  (volume (closer_to_5 ∩ segment) / volume segment) = 0.5 := 
by
  -- We can assume the Lebesgue measure (volume) of the segments and use it in the proof
  sorry

end probability_closer_to_5_l755_755660


namespace sector_area_l755_755761

/-- Given a sector with a radius of 2 and a central angle of 90 degrees, the area of the sector is π. -/
theorem sector_area : 
  let r : ℝ := 2
  let alpha_degrees : ℝ := 90
  let alpha_radians : ℝ := (alpha_degrees * Real.pi) / 180
  let area : ℝ := (1 / 2) * alpha_radians * r^2
  area = Real.pi :=
by
  let r : ℝ := 2
  let alpha_degrees : ℝ := 90
  let alpha_radians : ℝ := (alpha_degrees * Real.pi) / 180
  let area : ℝ := (1 / 2) * alpha_radians * r^2
  have h_alpha : alpha_radians = Real.pi / 2 := by sorry
  have h_area : area = Real.pi := by sorry
  exact h_area

end sector_area_l755_755761


namespace sin_585_eq_neg_sqrt2_div_2_l755_755588

theorem sin_585_eq_neg_sqrt2_div_2 :
  sin 585 = - (Real.sqrt 2 / 2) :=
by
  have h₁ : sin 585 = sin (585 - 360), by sorry
  have h₂ : sin (585 - 360) = sin 225, by sorry
  have h₃ : sin 225 = sin (45 + 180), by sorry
  have h₄ : sin (45 + 180) = -(sin 45), by sorry
  have h₅ : sin 45 = Real.sqrt 2 / 2, by sorry
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end sin_585_eq_neg_sqrt2_div_2_l755_755588


namespace polynomial_evaluation_x_eq_4_l755_755242

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l755_755242


namespace pablo_total_blocks_l755_755910

/-- Pablo made 4 stacks of toy blocks where:
   - The first stack is 5 blocks tall.
   - The second stack is 2 blocks taller than the first.
   - The third stack is 5 blocks shorter than the second stack.
   - The last stack is 5 blocks taller than the third stack.

   We want to prove that the total number of toy blocks Pablo used is 21.
-/
theorem pablo_total_blocks : 
  let stack1 := 5
  let stack2 := stack1 + 2
  let stack3 := stack2 - 5
  let stack4 := stack3 + 5
  stack1 + stack2 + stack3 + stack4 = 21 :=
by
  let stack1 := 5
  let stack2 := stack1 + 2
  let stack3 := stack2 - 5
  let stack4 := stack3 + 5
  show stack1 + stack2 + stack3 + stack4 = 21 from sorry

end pablo_total_blocks_l755_755910


namespace repeating_decimal_eq_fraction_l755_755345

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755345


namespace fraction_of_repeating_decimal_l755_755329

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755329


namespace neces_not_suff_cond_l755_755614

theorem neces_not_suff_cond (a : ℝ) (h : a ≠ 0) : (1 / a < 1) → (a > 1) :=
sorry

end neces_not_suff_cond_l755_755614


namespace eval_expression_equals_4_l755_755247

noncomputable def eval_expression : ℝ :=
  (2 ^ (-3)) * (7 ^ 0) / (2 ^ (-5))

theorem eval_expression_equals_4 : eval_expression = 4 := 
by 
  sorry

end eval_expression_equals_4_l755_755247


namespace set_game_total_sets_l755_755134

def is_valid_set (x y z : ℕ) : Prop :=
  ∀ i : ℕ, i < 4 → (digit_at x i = digit_at y i ∧ digit_at y i = digit_at z i) ∨
                  (digit_at x i ≠ digit_at y i ∧ digit_at y i ≠ digit_at z i ∧ digit_at x i ≠ digit_at z i)

def digit_at (n i : ℕ) : ℕ := (n / 10 ^ i) % 10

def valid_number (n : ℕ) : Prop :=
  (digit_at n 0 ∈ {1, 2, 3}) ∧
  (digit_at n 1 ∈ {1, 2, 3}) ∧
  (digit_at n 2 ∈ {1, 2, 3}) ∧
  (digit_at n 3 ∈ {1, 2, 3}) ∧
  ((digit_at n 0) ≠ (digit_at n 1) ∧
   (digit_at n 0) ≠ (digit_at n 2) ∧
   (digit_at n 0) ≠ (digit_at n 3) ∧
   (digit_at n 1) ≠ (digit_at n 2) ∧
   (digit_at n 1) ≠ (digit_at n 3) ∧
   (digit_at n 2) ≠ (digit_at n 3))

def count_valid_sets : ℕ :=
  finset.card
    { n1 n2 n3 |
      valid_number n1 ∧
      valid_number n2 ∧
      valid_number n3 ∧
      n1 ≠ n2 ∧
      n2 ≠ n3 ∧
      n1 ≠ n3 ∧
      is_valid_set n1 n2 n3 }

theorem set_game_total_sets : count_valid_sets = 1080 :=
by {
  -- Proof goes here
  sorry
}

end set_game_total_sets_l755_755134


namespace quadratic_roots_l755_755729

theorem quadratic_roots (m n p : ℕ) (h : m.gcd p = 1) 
  (h1 : 3 * m^2 - 8 * m * p + p^2 = p^2 * n) : n = 13 :=
by sorry

end quadratic_roots_l755_755729


namespace find_b_l755_755462

theorem find_b (a b c y1 y2 : ℝ)
  (h1 : y1 = a * (2:^3) + b * 2 + c)
  (h2 : y2 = a * ((-2):^3) + b * (-2) + c)
  (h3 : y1 - y2 = 12) : b = 3 - 4 * a := by
sorry

end find_b_l755_755462


namespace reciprocal_sqrt_two_l755_755070

theorem reciprocal_sqrt_two : (√2) * (√2 / 2) = 1 :=
by
  sorry

end reciprocal_sqrt_two_l755_755070


namespace prob_at_least_6_heads_in_8_flips_l755_755162

def fairCoinFlipProb : ℕ -> ℚ
| 8 := 17 / 256
| _ := 0

theorem prob_at_least_6_heads_in_8_flips : fairCoinFlipProb 8 = 17 / 256 :=
by
  sorry

end prob_at_least_6_heads_in_8_flips_l755_755162


namespace fraction_of_repeating_decimal_l755_755333

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755333


namespace repeating_decimal_is_fraction_l755_755299

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755299


namespace tissue_actual_diameter_l755_755120

theorem tissue_actual_diameter (magnification_factor : ℝ) (magnified_diameter : ℝ) 
(h1 : magnification_factor = 1000)
(h2 : magnified_diameter = 0.3) : 
  magnified_diameter / magnification_factor = 0.0003 :=
by sorry

end tissue_actual_diameter_l755_755120


namespace max_digit_sum_l755_755854

-- Define the sum of digits of an integer as a helper function
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem max_digit_sum (h : 0 <= h ∧ h < 24) (m : 0 <= m ∧ m < 60) :
  (sum_of_digits h) % 2 = 0 → sum_of_digits h + sum_of_digits m ≤ 22 :=
  sorry

end max_digit_sum_l755_755854


namespace proof_problem_l755_755507

noncomputable def f : ℕ → ℕ := sorry -- We define f but leave implementation as 'sorry'

axiom k : ℕ -- Assume k is a natural number

axiom f_strictly_increasing : ∀ {a b : ℕ}, a < b → f(a) < f(b)

axiom f_property : ∀ n : ℕ, f(f(n)) = k * n

theorem proof_problem (n : ℕ) : (2 * k) / (k + 1) * n ≤ f(n) ∧ f(n) ≤ (k + 1) / 2 * n :=
by
  sorry

end proof_problem_l755_755507


namespace sum_of_angles_l755_755042

theorem sum_of_angles (C : Type*) [circle C] (n : ℕ) (A : point C) (P : point C) :
  (divided_into_arcs C 16) →
  (arc_span x 3 A) →
  (arc_span y 5 P) →
  sum_of_angles_eq x y (90) :=
by sorry

end sum_of_angles_l755_755042


namespace square_area_l755_755190

open Real

-- Define the given conditions
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line := 8
def side_points (x : ℝ) := parabola x = line

-- Problem statement: Find the area of the square
theorem square_area :
  let x1 : ℝ := 1 in
  let x2 : ℝ := -5 in
  let side_length := abs (x1 - x2) in
  side_length * side_length = 36 :=
by
  -- Proof is replaced with sorry to satisfy the problem constraints
  sorry

end square_area_l755_755190


namespace inequality_solution_l755_755546

noncomputable def solve_inequality : Set ℝ :=
  {x | (x - 5) / ((x - 3)^2) < 0}

theorem inequality_solution :
  solve_inequality = {x | x < 3} ∪ {x | 3 < x ∧ x < 5} :=
by
  sorry

end inequality_solution_l755_755546


namespace lcm_of_two_numbers_l755_755845

theorem lcm_of_two_numbers (A B : ℕ) (h1 : A * B = 62216) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2828 :=
by
  sorry

end lcm_of_two_numbers_l755_755845


namespace angle_AXB_identity_l755_755408

-- Define the given vectors
def vec_OP := (2 : ℝ, 1 : ℝ)
def vec_OA := (1 : ℝ, 7 : ℝ)
def vec_OB := (5 : ℝ, 1 : ℝ)

-- Define that X is a point on the line OP
def on_line_OP (x y : ℝ) : Prop := x = 2 * y

-- Define the vectors XA and XB based on a point X on line OP
def vec_XA (y : ℝ) : ℝ × ℝ := (1 - 2 * y, 7 - y)
def vec_XB (y : ℝ) : ℝ × ℝ := (5 - 2 * y, 1 - y)

-- Condition when the dot product XA . XB is minimized
def dot_product_XA_XB (y : ℝ) : ℝ := (1 - 2 * y) * (5 - 2 * y) + (7 - y) * (1 - y)

-- Given minimal dot product when y = 2.2
def minimized_y := 2.2

-- Prove that the angle A X B is equal to acos(-4√17/17)
theorem angle_AXB_identity :
  ∀ X : ℝ × ℝ, 
    (on_line_OP X.1 X.2) → 
    let cos_angle := - (4 * real.sqrt 17) / 17 in
    (X = (2 * minimized_y, minimized_y) → 
      ∃ θ : ℝ, θ = real.arccos cos_angle ∧ θ = real.arccos (-4 * real.sqrt 17 / 17)) :=
by 
  intro X -> sorry

end angle_AXB_identity_l755_755408


namespace polygon_sides_l755_755795

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l755_755795


namespace max_distance_between_A_B_l755_755484

noncomputable def A : set ℂ := {z : ℂ | z^3 - 8 = 0}
noncomputable def B : set ℂ := {z : ℂ | z^3 - 8*z^2 - 8*z + 64 = 0}
noncomputable def distance (z1 z2 : ℂ) : ℝ := complex.abs (z1 - z2)

theorem max_distance_between_A_B :
  ∀ z1 ∈ A, ∀ z2 ∈ B, distance z1 z2 ≤ 2 * real.sqrt 21 ∧
    ∃ w1 ∈ A, ∃ w2 ∈ B, distance w1 w2 = 2 * real.sqrt 21 :=
  sorry

end max_distance_between_A_B_l755_755484


namespace bankers_discount_l755_755624

theorem bankers_discount (T_D F_V : ℝ) (hT_D : T_D = 360) (hF_V : F_V = 2460) :
  let P_V := F_V - T_D in
  let B_D := (T_D * F_V) / P_V in
  B_D = 422 :=
by
  sorry

end bankers_discount_l755_755624


namespace cake_and_tea_cost_l755_755952

theorem cake_and_tea_cost (cost_of_milk_tea : ℝ) (cost_of_cake : ℝ)
    (h1 : cost_of_cake = (3 / 4) * cost_of_milk_tea)
    (h2 : cost_of_milk_tea = 2.40) :
    2 * cost_of_cake + cost_of_milk_tea = 6.00 := 
sorry

end cake_and_tea_cost_l755_755952


namespace game_24_set1_game_24_set2_l755_755012

-- Equivalent proof problem for set {3, 2, 6, 7}
theorem game_24_set1 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 2) (h₃ : c = 6) (h₄ : d = 7) :
  ((d / b) * c + a) = 24 := by
  subst_vars
  sorry

-- Equivalent proof problem for set {3, 4, -6, 10}
theorem game_24_set2 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = -6) (h₄ : d = 10) :
  ((b + c + d) * a) = 24 := by
  subst_vars
  sorry

end game_24_set1_game_24_set2_l755_755012


namespace circle_circumference_difference_l755_755859

theorem circle_circumference_difference (d_inner : ℝ) (h_inner : d_inner = 100) 
  (d_outer : ℝ) (h_outer : d_outer = d_inner + 30) :
  ((π * d_outer) - (π * d_inner)) = 30 * π :=
by 
  sorry

end circle_circumference_difference_l755_755859


namespace max_r_plus_s_is_3029_l755_755886

noncomputable def max_r_plus_s (P Q : Polynomial ℝ) (degree : ℕ) : ℕ :=
  if P.degree = degree ∧ Q.degree = degree ∧ ∃ (r s : ℕ),
    (∃ roots : multiset ℝ, roots.card = r ∧ (∀ x ∈ roots, P.eval x = 0 ∧ Q.eval x = 0)) ∧
    (∃ coeffs : finset ℕ, coeffs.card = s ∧ (∀ n ∈ coeffs, P.coeff n = Q.coeff n)) then 3029 else sorry

theorem max_r_plus_s_is_3029 {P Q : Polynomial ℝ} (degree : ℕ)
  (hP : P.degree = degree) (hQ : Q.degree = degree)
  (h : ∃ (r s : ℕ),
    (∃ roots : multiset ℝ, roots.card = r ∧ (∀ x ∈ roots, P.eval x = 0 ∧ Q.eval x = 0)) ∧
    (∃ coeffs : finset ℕ, coeffs.card = s ∧ (∀ n ∈ coeffs, P.coeff n = Q.coeff n))) :
  r + s = 3029 :=
sorry

end max_r_plus_s_is_3029_l755_755886


namespace fraction_eq_repeating_decimal_l755_755358

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755358


namespace minimum_value_x2_y2_z2_l755_755449

theorem minimum_value_x2_y2_z2 (x y z : ℝ) (h : x - 1 = 2 * (y + 1) ∧ x - 1 = 3 * (z + 2)) :
  x^2 + y^2 + z^2 = 293 / 49 :=
by
  let k := x - 1
  have hx : x = k + 1, by sorry
  have hy : y = k / 2 - 1, by sorry
  have hz : z = k / 3 - 2, by sorry
  have hxyz : x^2 + y^2 + z^2 = ((k + 1)^2 + (k / 2 - 1)^2 + (k / 3 - 2)^2), by sorry
  have h_min : (k = 6 / 49) -> (x^2 + y^2 + z^2 = 293 / 49), by sorry
  sorry

end minimum_value_x2_y2_z2_l755_755449


namespace probability_at_least_6_heads_l755_755163

theorem probability_at_least_6_heads (n : ℕ) (p : ℚ) : n = 8 ∧ p = (3 / 128) :=
  let total_outcomes := 2 ^ n in
  let successful_outcomes := 3 + 2 + 1 in
  n = 8 ∧ total_outcomes = 256 ∧ p = (successful_outcomes.to_rat / total_outcomes.to_rat) := sorry

end probability_at_least_6_heads_l755_755163


namespace part_one_part_two_part_three_l755_755426

variable (f : ℝ → ℝ) (a : ℝ)

-- Part I
theorem part_one (h : ∀ x, f x = -x^2 + 2*|x - a|) (hf : ∀ x, f x = f (-x)) : a = 0 :=
sorry

-- Part II
theorem part_two (h : ∀ x, f x = -x^2 + 2*|x - (1/2)|) : 
  (∀ x, f' x > 0 → x ∈ ℕ) → (-∞, -1] ∪ [1/2, 1] :=
sorry

-- Part III
theorem part_three (h : ∀ x, f x = -x^2 + 2*|x - a|)
  (hf : ∀ x ∈ set.Ici (0 : ℝ), f (x - 1) ≥ 2 * f x) (ha : 0 < a) : sqrt 6 - 2 ≤ a ∧ a ≤ 1/2 :=
sorry

end part_one_part_two_part_three_l755_755426


namespace size_of_third_jar_is_one_l755_755880

theorem size_of_third_jar_is_one
  (total_jars : ℕ)
  (gallons : ℝ)
  (quart : ℝ)
  (half_gallon : ℝ)
  (x : ℝ)
  (equal_jars : total_jars / 3)
  (total_volume : 16 * quart + 16 * half_gallon + 16 * x = 28)
  (quart_def : quart = 1/4)
  (half_gallon_def : half_gallon = 1/2)
  (total_jars_def : total_jars = 48)
  (gallons_def : gallons = 28) :
  x = 1 := by
  sorry

end size_of_third_jar_is_one_l755_755880


namespace range_of_a_l755_755873

theorem range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (x-a) / (2 - (x + 1 - a)) > 0)
  ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l755_755873


namespace expression_value_l755_755113

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l755_755113


namespace factorable_polynomial_conditions_l755_755388

theorem factorable_polynomial_conditions
  {n : ℕ} (n_ge_2 : n ≥ 2) 
  (a : Fin n → ℕ) (dist_a : ∀ i j, i ≠ j → a i ≠ a j) :
  (∃ (g h : ℤ[X]), 
    f = g * h ∧ degree g < n ∧ degree h < n) 
  ↔ (n = 2 ∧ a 0 = 2) ∨ 
     (n = 4 ∧ ∃ (a_perm : Fin 3 → ℕ), 
       ∀ i, a i = a_perm i ∧ {a_perm 0, a_perm 1, a_perm 2} = {1, 2, 3}) :=
sorry

end factorable_polynomial_conditions_l755_755388


namespace initial_ratio_of_milk_to_water_l755_755472

variable (M W : ℕ)
noncomputable def M_initial := 45 - W
noncomputable def W_new := W + 9

theorem initial_ratio_of_milk_to_water :
  M_initial = 36 ∧ W = 9 →
  M_initial / (W + 9) = 2 ↔ 4 = M_initial / W := 
sorry

end initial_ratio_of_milk_to_water_l755_755472


namespace fraction_eq_repeating_decimal_l755_755357

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755357


namespace barbier_theorem_Δ_curves_l755_755539

-- Define a Δ-curve
structure DeltaCurve :=
  (curve : Set ℝ)

-- Define rotations in degrees
def rotate (curve : DeltaCurve) (degrees : ℝ) : DeltaCurve := sorry

-- Define rotation by specific degrees
def rotate_120 (curve : DeltaCurve) : DeltaCurve := rotate curve 120
def rotate_240 (curve : DeltaCurve) : DeltaCurve := rotate curve 240

-- Define the summation of curves
def sum_curves (curve1 curve2 curve3 : DeltaCurve) : DeltaCurve := sorry

-- Barbier's theorem for Δ-curves
theorem barbier_theorem_Δ_curves (K : DeltaCurve) :
  let K' := rotate_120 K
  let K'' := rotate_240 K
  let M := sum_curves K K' K''
  is_circle M := sorry

end barbier_theorem_Δ_curves_l755_755539


namespace quadratic_inequality_solution_l755_755612

theorem quadratic_inequality_solution (m: ℝ) (h: m > 1) :
  { x : ℝ | x^2 + (m - 1) * x - m ≥ 0 } = { x | x ≤ -m ∨ x ≥ 1 } :=
sorry

end quadratic_inequality_solution_l755_755612


namespace area_ratio_l755_755746

variable {A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point}
variable {k : ℝ}
variable [Triangle A B C]
variable (parallel : ∀ t, Parallel (Line A A₁) t ∧ Parallel (Line B B₁) t ∧ Parallel (Line C C₁) t)
variable (divides : ∀ {A A₁} (k : ℝ), divides (Segment A A₁) A₂ k ∧
                                      divides (Segment B B₁) B₂ k ∧
                                      divides (Segment C C₁) C₂ k)

theorem area_ratio (h1 : divides k) (parallel_lines : parallel) :
  area_ratio (triangle A B C) (triangle A₂ B₂ C₂) = (k+1) / k :=
sorry

end area_ratio_l755_755746


namespace cake_and_milk_tea_cost_l755_755955

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end cake_and_milk_tea_cost_l755_755955


namespace fourth_term_of_expansion_constant_term_of_expansion_l755_755486

noncomputable def binomial_term (a b : ℕ) (x : ℕ) (n : ℕ) (r : ℕ) : ℕ :=
  nat.choose n r * (a ^ (n - r)) * (b ^ r)

noncomputable def fourth_term (x : ℕ) : ℤ :=
  -(7 * x ^ (2 / 3))

noncomputable def constant_term : ℚ :=
  35 / 8

theorem fourth_term_of_expansion (x : ℕ) :
  fourth_term x = -(7 * x ^ (2 / 3)) := 
sorry

theorem constant_term_of_expansion : 
  constant_term = 35 / 8 := 
sorry

end fourth_term_of_expansion_constant_term_of_expansion_l755_755486


namespace recurrence_calculations_recurrence_conjecture_sum_inequality_l755_755703

theorem recurrence_calculations (a : ℕ → ℝ) (h : ∀ n, a (n+1) = a n^2 - n * a n + 1) :
  a 1 = 2 → a 2 = 3 ∧ a 3 = 4 ∧ a 4 = 5 :=
sorry

theorem recurrence_conjecture (a : ℕ → ℝ) (h : ∀ n, a (n+1) = a n^2 - n * a n + 1) :
  a 1 ≥ 3 → ∀ n, a (n+1) ≥ n + 3 :=
sorry

theorem sum_inequality (a : ℕ → ℝ) (h : ∀ n, a (n+1) = a n^2 - n * a n + 1) :
  a 1 ≥ 3 → ∀ n, (∑ k in finset.range (n+1), 1 / (1 + a k)) ≤ 1/2 :=
sorry

end recurrence_calculations_recurrence_conjecture_sum_inequality_l755_755703


namespace area_triangle_AEH_l755_755863

theorem area_triangle_AEH :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (2 : ℝ, 0 : ℝ)
  let C := (2 : ℝ, 4 : ℝ)
  let D := (0 : ℝ, 4 : ℝ)
  let E := (2 : ℝ, (1 * 4 + 3 * 0) / 4)
  let F := ((1 * 0 + 3 * 2) / 4, 4 : ℝ)
  let G := (0 : ℝ, (1 * 4 + 3 * 0) / 4)
  let H := ((1.5 + 0) / 2, (4 + 1) / 2 : ℝ)
  let area := (|0 * (E.2 - H.2) + 2 * (H.2 - A.2) + 0.75 * (A.2 - E.2)| : ℝ) / 2
  area = 2.125 := 
  sorry

end area_triangle_AEH_l755_755863


namespace repeating_decimal_is_fraction_l755_755301

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755301


namespace max_mondays_in_59_days_l755_755606

theorem max_mondays_in_59_days (start_day : ℕ) : ∃ d : ℕ, d ≤ 6 ∧ 
  start_day = d → (d = 0 → ∃ m : ℕ, m = 9) :=
by 
  sorry

end max_mondays_in_59_days_l755_755606


namespace rectangles_cannot_cover_large_rectangle_l755_755389

theorem rectangles_cannot_cover_large_rectangle (n m : ℕ) (a b c d: ℕ) : 
  n = 14 → m = 9 → a = 2 → b = 3 → c = 3 → d = 2 → 
  (∀ (v_rects : ℕ) (h_rects : ℕ), v_rects = 10 → h_rects = 11 →
    (∀ (rect_area : ℕ), rect_area = n * m →
      (∀ (small_rect_area : ℕ), 
        small_rect_area = (v_rects * (a * b)) + (h_rects * (c * d)) →
        small_rect_area = rect_area → 
        false))) :=
by
  intros n_eq m_eq a_eq b_eq c_eq d_eq
       v_rects h_rects v_rects_eq h_rects_eq
       rect_area rect_area_eq small_rect_area small_rect_area_eq area_sum_eq
  sorry

end rectangles_cannot_cover_large_rectangle_l755_755389


namespace diagonal_ratio_l755_755069

variable (a b : ℝ)
variable (d1 : ℝ) -- diagonal length of the first square
variable (r : ℝ := 1.5) -- ratio between perimeters

theorem diagonal_ratio (h : 4 * a / (4 * b) = r) (hd1 : d1 = a * Real.sqrt 2) : 
  (b * Real.sqrt 2) = (2/3) * d1 := 
sorry

end diagonal_ratio_l755_755069


namespace num_aluminum_cans_l755_755945

def num_glass_bottles : ℕ := 10
def total_litter : ℕ := 18

theorem num_aluminum_cans : total_litter - num_glass_bottles = 8 :=
by
  sorry

end num_aluminum_cans_l755_755945


namespace triangle_problem_l755_755763

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : 0.5 * b * c * Real.sin A = 2 * Real.sqrt 2)
  : Real.cos A = 1 / 3 ∧
    (∀ (p : ℝ), (p = a + b + c) → (p ≥ 2 * Real.sqrt 6 + 2 * Real.sqrt 2)) :=
begin
  sorry
end

end triangle_problem_l755_755763


namespace average_multiples_of_10_from_10_to_200_l755_755604

/--
Prove that the average (arithmetic mean) of all multiples of 10 from 10 to 200 inclusive is 105.
-/
theorem average_multiples_of_10_from_10_to_200 : 
  let multiples := list.map (λ n, n * 10) (list.range' 1 20) in
  let a1 := 10 in
  let an := 200 in
  list.average multiples = 105 := 
by
  -- Defining the necessary values
  let multiples := list.map (λ n, n * 10) (list.range' 1 20)
  let a1 := 10
  let an := 200
  -- Sorry means we leave the proof incomplete since the steps aren't being considered.
  sorry

end average_multiples_of_10_from_10_to_200_l755_755604


namespace trigonometric_identity_l755_755385

theorem trigonometric_identity (α : ℝ) (h : (cos α + sin α) / (cos α - sin α) = 2) : 
  (1 + sin (4 * α) - cos (4 * α)) / (1 + sin (4 * α) + cos (4 * α)) = 3 / 4 :=
by
  sorry

end trigonometric_identity_l755_755385


namespace lines_parallel_iff_a_eq_1_l755_755784

theorem lines_parallel_iff_a_eq_1 (x y a : ℝ) :
    (a = 1 ↔ ∃ k : ℝ, ∀ x y : ℝ, a*x + y - 1 = k*(x + a*y + 1)) :=
sorry

end lines_parallel_iff_a_eq_1_l755_755784


namespace trioball_play_time_l755_755084

theorem trioball_play_time (total_duration : ℕ) (num_children : ℕ) (players_at_a_time : ℕ) 
  (equal_play_time : ℕ) (H1 : total_duration = 120) (H2 : num_children = 3) (H3 : players_at_a_time = 2)
  (H4 : equal_play_time = 240 / num_children)
  : equal_play_time = 80 := 
by 
  sorry

end trioball_play_time_l755_755084


namespace coffee_reduction_l755_755699

-- Define the conditions
def thermos_volume : ℕ := 20 -- Coffee only, in ounces
def cups_per_thermos_fill : ℕ := 1/2 -- Cups of milk in each fill
def cup_to_ounces : ℕ := 8 -- 1 cup = 8 ounces
def fills_per_day : ℕ := 2
def days_per_week : ℕ := 5
def reduction_fraction : ℚ := 1 / 4

-- Calculate the daily consumption of coffee and milk
def coffee_per_day : ℕ := thermos_volume * fills_per_day
def milk_per_fill : ℕ := (cups_per_thermos_fill * cup_to_ounces)
def milk_per_day : ℕ := milk_per_fill * fills_per_day

-- Calculate the weekly consumption before reduction
def total_liquid_per_week : ℕ := coffee_per_day * days_per_week
def total_milk_per_week : ℕ := milk_per_day * days_per_week
def total_coffee_per_week : ℕ := total_liquid_per_week - total_milk_per_week

-- Calculate the weekly consumption after reduction
def reduced_coffee_per_week : ℕ := ((total_coffee_per_week : ℕ) * (reduction_fraction)).natAbs

-- Prove the amount of coffee consumed per week after reduction is 40 ounces
theorem coffee_reduction : reduced_coffee_per_week = 40 := by
  sorry

end coffee_reduction_l755_755699


namespace problem_statement_l755_755386

theorem problem_statement :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
    (∀ x : ℝ, 1 + x^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + 
              a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) ∧
    (a_0 = 2) ∧
    (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 33)) →
  (∃ a_1 a_2 a_3 a_4 a_5 : ℝ, a_1 + a_2 + a_3 + a_4 + a_5 = 31) :=
by
  sorry

end problem_statement_l755_755386


namespace lambda_parallel_vectors_l755_755799

variables {R : Type*} [linear_ordered_field R] 
variables {V : Type*} [add_comm_group V] [module R V]

theorem lambda_parallel_vectors (a b : V) (λ : R) 
  (h1 : ¬ collinear a b)
  (h2 : ∃ k : R, λ • a + b = k • (2 • a + λ • b)) :
  λ = real.sqrt 2 ∨ λ = -real.sqrt 2 :=
sorry

end lambda_parallel_vectors_l755_755799


namespace solve_for_a_minus_b_l755_755448

theorem solve_for_a_minus_b (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := 
sorry

end solve_for_a_minus_b_l755_755448


namespace radius_of_circle_is_14_feet_l755_755156

-- Definition of the problem
def num_pencils : Nat := 56
def pencil_length_inch : Nat := 6

-- Condition: The total length of pencils forming the diameter of the circle
def diameter_inch : Nat := num_pencils * pencil_length_inch

-- Conversion from inches to feet
def inches_to_feet(inches : Nat) : Nat := inches / 12

-- The diameter in feet
def diameter_feet : Nat := inches_to_feet(diameter_inch)

-- Radius is half of the diameter
def radius_feet : Nat := diameter_feet / 2

-- Proof problem statement
theorem radius_of_circle_is_14_feet (h : diameter_feet = 28) : radius_feet = 14 :=
by
  -- Proof to be filled
  sorry

end radius_of_circle_is_14_feet_l755_755156


namespace sin_cos_sum_value_l755_755451

def sin_cos_sum (θ : ℝ) (b : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : cos (2 * θ) = b) : ℝ := 
  sin θ + cos θ

theorem sin_cos_sum_value (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < π / 2) (h2 : cos (2 * θ) = b) : 
  sin_cos_sum θ b h1 h2 = sqrt (2 - b) := 
  sorry

end sin_cos_sum_value_l755_755451


namespace length_lemma_l755_755452

def smallest_non_divisor (n : ℕ) : ℕ :=
  if h : n > 0 then Nat.find (Nat.exists_not_dvd h) else 1

def length (n : ℕ) : ℕ :=
  if hn : n >= 3 then Nat.part_rec_on (fun m => smallest_non_divisor m) (fun k => if k = 2 then 0 else 1) hn else 0

theorem length_lemma (n : ℕ) (h : n >= 3) :
  length n = if n % 2 = 0 then if (smallest_non_divisor (smallest_non_divisor n)) = 3 then 3 else 2 else 1 :=
sorry

end length_lemma_l755_755452


namespace coverage_of_each_bag_l755_755198

-- Define the dimensions of the lot
def lot_length : ℕ := 120
def lot_width : ℕ := 60

-- Define the dimensions of the concrete section
def concrete_length : ℕ := 40
def concrete_width : ℕ := 40

-- Define the number of bags required
def num_bags : ℕ := 100

-- Define the total area of the lot
def total_lot_area : ℕ := lot_length * lot_width

-- Define the total area of the concrete section
def concrete_area : ℕ := concrete_length * concrete_width

-- Define the area that needs grass seed
def grass_seed_area : ℕ := total_lot_area - concrete_area

-- Define the coverage per bag of grass seed
def coverage_per_bag : ℕ := grass_seed_area / num_bags

-- The theorem to prove
theorem coverage_of_each_bag :
  lot_length = 120 ∧ lot_width = 60 ∧
  concrete_length = 40 ∧ concrete_width = 40 ∧
  num_bags = 100 →
  coverage_per_bag = 56 :=
by {
  intros H,
  cases H with H_lot H_rest,
  cases H_lot with H_lot_length H_lot_width,
  cases H_rest with H_concrete H_num_bags,
  cases H_concrete with H_concrete_length H_concrete_width,
  cases H_num_bags with H_num_bags,
  sorry
}

end coverage_of_each_bag_l755_755198


namespace g_derivative_l755_755835

noncomputable def g (x : ℝ) : ℝ := log x / log 2 + 3 ^ x

theorem g_derivative (x : ℝ) : deriv g x = 1 / (x * log 2) + 3 ^ x * log 3 := by
sorrr

end g_derivative_l755_755835


namespace max_distance_from_curve_to_line_l755_755637

-- Definitions based on conditions
def polar_line := ∀ (ρ θ : ℝ), ρ * real.cos(θ - π / 4) = 3 * real.sqrt 2
def curve_c := ∀ (ρ θ : ℝ), ρ = 1

-- The theorem to prove
theorem max_distance_from_curve_to_line :
  let d_max := 3 * real.sqrt 2 + 1 in
    ∃ (d : ℝ), d = d_max ∧ ∀ (ρ θ : ℝ), curve_c ρ θ → polar_line ρ θ → d ≤ d_max :=
sorry

end max_distance_from_curve_to_line_l755_755637


namespace length_of_AC_l755_755221

theorem length_of_AC (r : ℝ) (Q : Type*) (Q_circle : Metric.Sphere 0 r = Set.univ) 
  (C_eq : 2 * π * r = 18 * π) (A B C : Q) (AB_diameter : dist A B = 2 * r) 
  (Q_at_center : dist 0 A = r) (QAC_angle : angle A 0 C = π / 4) :
  dist A C = 9 * Real.sqrt 2 :=
  sorry

end length_of_AC_l755_755221


namespace repeating_decimal_as_fraction_l755_755302

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755302


namespace repeating_fraction_equality_l755_755350

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755350


namespace find_expression_l755_755454

variable (a b E : ℝ)

-- Conditions
def condition1 := a / b = 4 / 3
def condition2 := E / (3 * a - 2 * b) = 3

-- Conclusion we want to prove
theorem find_expression : condition1 a b → condition2 a b E → E = 6 * b :=
by
  intro h1 h2
  sorry

end find_expression_l755_755454


namespace negation_of_proposition_p_l755_755402

def f : ℝ → ℝ := sorry

theorem negation_of_proposition_p :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔ (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := 
by
  sorry

end negation_of_proposition_p_l755_755402


namespace time_to_produce_one_item_l755_755172

-- Definitions based on the conditions
def itemsProduced : Nat := 300
def totalTimeHours : ℝ := 2.0
def minutesPerHour : ℝ := 60.0

-- The statement we need to prove
theorem time_to_produce_one_item : (totalTimeHours / itemsProduced * minutesPerHour) = 0.4 := by
  sorry

end time_to_produce_one_item_l755_755172


namespace equivalent_discount_l755_755170

theorem equivalent_discount : 
  ∀ (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ),
  original_price = 50 → first_discount = 0.30 → second_discount = 0.40 →
  let final_price := original_price * (1 - first_discount) * (1 - second_discount) in
  ∃ (single_discount : ℝ), 
  final_price = original_price * (1 - single_discount) ∧ 
  single_discount = 0.58 := 
by
  intros original_price first_discount second_discount h₁ h₂ h₃
  let final_price := original_price * (1 - first_discount) * (1 - second_discount)
  use 0.58
  split
  sorry -- Placeholder for actual proof
  sorry -- Placeholder for proof of equivalence


end equivalent_discount_l755_755170


namespace meters_conversion_equivalence_l755_755639

-- Define the conditions
def meters_to_decimeters (m : ℝ) : ℝ := m * 10
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- State the problem
theorem meters_conversion_equivalence :
  7.34 = 7 + (meters_to_decimeters 0.3) / 10 + (meters_to_centimeters 0.04) / 100 :=
sorry

end meters_conversion_equivalence_l755_755639


namespace prob_at_least_6_heads_in_8_flips_l755_755160

def fairCoinFlipProb : ℕ -> ℚ
| 8 := 17 / 256
| _ := 0

theorem prob_at_least_6_heads_in_8_flips : fairCoinFlipProb 8 = 17 / 256 :=
by
  sorry

end prob_at_least_6_heads_in_8_flips_l755_755160


namespace perimeter_of_square_l755_755550

theorem perimeter_of_square (area : ℝ) (h : area = 324) : ∃ P, P = 72 :=
by
  let side := Real.sqrt area
  have : side = 18, from sorry
  let P := 4 * side
  have : P = 72, from sorry
  exact ⟨P, ‹P = 72›⟩

end perimeter_of_square_l755_755550


namespace fraction_for_repeating_56_l755_755264

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755264


namespace third_member_income_l755_755590

noncomputable def income_of_third_member (total_members : ℕ) (average_income : ℝ) (income1 : ℝ) (income2 : ℝ) (income4 : ℝ) : ℝ :=
  let total_income := average_income * total_members
  let known_income := income1 + income2 + income4
  total_income - known_income

theorem third_member_income (total_members average_income income1 income2 income4 : ℝ) 
  (h1 : total_members = 4) 
  (h2 : average_income = 10000) 
  (h3 : income1 = 8000) 
  (h4 : income2 = 15000) 
  (h5 : income4 = 11000) : 
  income_of_third_member total_members average_income income1 income2 income4 = 5000 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end third_member_income_l755_755590


namespace domain_of_even_function_l755_755458

noncomputable def f (x : ℝ) (a : ℝ) := sqrt (8 - a * x - 2 * x ^ 2)

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

theorem domain_of_even_function :
  ∀ a, is_even (f · a) → ∀ x, -2 ≤ x ∧ x ≤ 2 :=
  sorry

end domain_of_even_function_l755_755458


namespace repeating_decimal_equiv_fraction_l755_755289

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755289


namespace x_squared_minus_y_squared_l755_755838

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l755_755838


namespace fraction_for_repeating_56_l755_755262

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755262


namespace volume_ratio_spheres_l755_755404

theorem volume_ratio_spheres (R : ℝ) (hR : R > 0) :
  let r := (sqrt 3 / 2) * R in
  let volume_sphere := λ r : ℝ, (4 / 3) * π * r^3 in
  volume_sphere r / volume_sphere R = (3 / 8) * sqrt 3 :=
by
  sorry

end volume_ratio_spheres_l755_755404


namespace minimum_value_l755_755831

theorem minimum_value (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 1) : 
  (∃ (x : ℝ), x = a + 2*b) → (∃ (y : ℝ), y = 2*a + b) → 
  (∀ (x y : ℝ), x + y = 3 → (1/x + 4/y) ≥ 3) :=
by
  sorry

end minimum_value_l755_755831


namespace fraction_eq_repeating_decimal_l755_755270

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755270


namespace sum_log_geometric_series_l755_755780

/-- Given that $\{a_n}$ is a geometric sequence and $a_1 a_{100} = 64$, 
prove that $\log_2 a_1 + \log_2 a_2 + ... + \log_2 a_{100} = 300$. -/
theorem sum_log_geometric_series (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_prod : a 0 * a 99 = 64) : finset.sum (finset.range 100) (λ n, real.logb 2 (a n)) = 300 :=
by {
  -- Proof not required, placeholder sorry
  sorry
}

end sum_log_geometric_series_l755_755780


namespace garden_perimeter_l755_755575

theorem garden_perimeter (length breadth : ℕ) (h_length : length = 140) (h_breadth : breadth = 100) : 
  2 * (length + breadth) = 480 := 
by 
  rw [h_length, h_breadth]; 
  norm_num

end garden_perimeter_l755_755575


namespace sum_of_m_and_n_l755_755755

theorem sum_of_m_and_n (m n : ℤ)
  (h1 : (1 : ℤ) + 1 * complex.i = (m - 1) + (n - 2) * complex.i) :
  m + n = 5 :=
by
  sorry

end sum_of_m_and_n_l755_755755


namespace solveGrid_l755_755146

-- Define the grid and its properties
def grid : Type := ℕ × ℕ → ℕ 

-- Define the conditions for the grid
def isValidGrid (g : grid) : Prop :=
  (∀ i, i ∈ [1, 2, 3, 4] → ∀ j, j ∈ [1, 2, 3, 4] →
    (g (i, j) ∈ [1, 2, 3, 4] ∧
    (∀ k, k ∈ [1, 2, 3, 4] → (k ≠ i → g (i, j) ≠ g (k, j)) ∧ (k ≠ j → g (i, j) ≠ g (i, k))))) ∧
  (∀ (i j : ℕ), i ≠ j →
    (g (1, 1) ≠ g (i, j) ∧ g (1, 2) ≠ g (i, j) ∧ g (2, 1) ≠ g (i, j) ∧ g (2, 2) ≠ g (i, j)) ∧
    (g (1, 3) ≠ g (i, j) ∧ g (1, 4) ≠ g (i, j) ∧ g (2, 3) ≠ g (i, j) ∧ g (2, 4) ≠ g (i, j)) ∧
    (g (3, 1) ≠ g (i, j) ∧ g (3, 2) ≠ g (i, j) ∧ g (4, 1) ≠ g (i, j) ∧ g (4, 2) ≠ g (i, j)) ∧
    (g (3, 3) ≠ g (i, j) ∧ g (3, 4) ≠ g (i, j) ∧ g (4, 3) ≠ g (i, j) ∧ g (4, 4) ≠ g (i, j)))

def partiallyFilledGrid : grid :=
  λ ij, match ij with
  | (1, 1) => 3
  | (1, 2) => 1
  | (1, 4) => 2
  | (2, 3) => 1
  | (2, 4) => 4
  | (3, 2) => 4
  | (4, 1) => 1
  | (4, 3) => 3
  | _ => 0 -- 0 denotes an empty cell
  end

-- Expected positions of the '4's
def positionsOfFour (g : grid) : Prop :=
  g (1, 3) = 4 ∧ g (2, 1) = 4 ∧ g (3, 4) = 4 ∧ g (4, 2) = 4

theorem solveGrid : ∃ g : grid, isValidGrid g ∧ positionsOfFour g :=
by
  -- solution steps to fill in the correct numbers
  sorry

end solveGrid_l755_755146


namespace isosceles_triangle_sum_x_l755_755477

noncomputable def sum_possible_values_of_x : ℝ :=
  let x1 : ℝ := 20
  let x2 : ℝ := 50
  let x3 : ℝ := 80
  x1 + x2 + x3

theorem isosceles_triangle_sum_x (x : ℝ) (h1 : x = 20 ∨ x = 50 ∨ x = 80) : sum_possible_values_of_x = 150 :=
  by
    sorry

end isosceles_triangle_sum_x_l755_755477


namespace frank_total_hours_worked_l755_755384

theorem frank_total_hours_worked : (∀ n : ℕ, (n = 8) → ∀ d : ℕ, (d = 4) → n * d = 32) :=
by
  intros n hn d hd
  rw [hn, hd]
  exact rfl

end frank_total_hours_worked_l755_755384


namespace find_total_mixture_weight_l755_755154

-- Define the weights of the components as variables
variable (total_mixture_weight : ℝ)

-- Define the conditions
def sand_weight (W : ℝ) := (1/5) * W
def water_weight (W : ℝ) := (3/4) * W
def gravel_weight := 6.0

-- State the theorem
theorem find_total_mixture_weight : (sand_weight total_mixture_weight + water_weight total_mixture_weight + gravel_weight = total_mixture_weight) ↔ (total_mixture_weight = 120.0) :=
by
  sorry

end find_total_mixture_weight_l755_755154


namespace fraction_for_repeating_56_l755_755266

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755266


namespace tom_lifting_capacity_factor_l755_755087

theorem tom_lifting_capacity_factor :
  ∃ F : ℝ, 
    (∀ (initial_capacity specialization_factor : ℝ), 
    initial_capacity = 160 ∧ specialization_factor = 1.10 ∧ (160 * F * specialization_factor = 352) 
    → F = 2) :=
begin
  use 2,
  intros initial_capacity specialization_factor h,
  rcases h with ⟨h1, h2, h3⟩,
  calc 
    2 = 352 / (160 * 1.10) : by sorry
    ... = F : by sorry
end

end tom_lifting_capacity_factor_l755_755087


namespace part_a_part_b_part_c_l755_755504

-- Part (a)
theorem part_a (G : SimpleGraph V) (f : SimpleGraph V → ℕ)
  (v : V → Bool) (n : ℕ)
  (Hle : ∀ (i : ℕ), i < n → f(G -i) ≤ f(G - v i)) :
  f(G) ≤ ∑ i in range n, f(G - v i) := 
sorry

-- Part (b)
theorem part_b (G : SimpleGraph V) (e : G.Edge) (f : SimpleGraph V → ℕ) :
  let G_prime := G.removeEdge(e).contractClique
  f(G) = f(G - e) + f(G_prime) := 
sorry

-- Part (c)
theorem part_c (alpha : ℝ) (h : alpha > 1) :
  ∃ (G : SimpleGraph V) (e : G.Edge), 
  f(G) / f(G - e) < alpha :=
sorry

end part_a_part_b_part_c_l755_755504


namespace shape_has_integer_edge_length_l755_755922

theorem shape_has_integer_edge_length 
  {R : Type} 
  (R_i : ℕ → Type) (n : ℕ)
  (mutually_exclusive : ∀ i j (hij : i ≠ j), (R_i i) ∩ (R_i j) = ∅)
  (edges_parallel : ∀ i, edges_parallel_to_R (R_i i) R)
  (has_integer_edge : ∀ i, ∃ e ∈ edges (R_i i), integer_length e) :
  ∃ e ∈ edges R, integer_length e :=
sorry

end shape_has_integer_edge_length_l755_755922


namespace find_d_l755_755850

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem find_d
  (A B C : Type)
  (AB BC AC : ℝ)
  (h_AB : AB = 300)
  (h_BC : BC = 320)
  (h_AC : AC = 400)
  (P : A)
  (d : ℝ)
  (h_parallel : ∀ (D D' E E' F F' : Type), 
    parallel D D' (line_through P A) ∧ parallel E E' (line_through P B) ∧ parallel F F' (line_through P C))
  (h_length : ∀ (D D' E E' F F' : Type), 
      length D D' = d ∧ length E E' = d ∧ length F F' = d)
  (area_ABC : ℝ := herons_formula AB BC AC)
  (area_DPD' : ℝ := area_ABC / 4)
  (similarity_ratio : (d / 300) = real.sqrt (area_DPD' / area_ABC)) :
  d = 150 :=
sorry

end find_d_l755_755850


namespace ordering_l755_755096

noncomputable def a := 2^29
noncomputable def b := 2^24
noncomputable def c := 2^23.7744

theorem ordering : c < b ∧ b < a := 
by
  have h₁ : c < b := sorry
  have h₂ : b < a := sorry
  exact ⟨h₁, h₂⟩

end ordering_l755_755096


namespace find_angle_C_proof_max_triangle_area_proof_l755_755491

open Real

noncomputable def find_angle_C (A B : ℝ) (C : ℝ) :=
  let m := (cos A, sin A)
  let n := (cos B, -sin B)
  let vector_distance := sqrt ((cos A - cos B)^2 + (sin A + sin B)^2)
  vector_distance = 1

noncomputable def max_triangle_area (A B C : ℝ) (a b c : ℝ) :=
  let cosC := -1/2
  let C := 2 * π / 3
  let triangle_area (a b C : ℝ) := 1/2 * a * b * sin C
  c = 3 ∧ a * b * sin C ≤ 3/2

theorem find_angle_C_proof (A B : ℝ) :
  (∃ C : ℝ, find_angle_C A B C) →
  C = 2 * π / 3 :=
by
  sorry

theorem max_triangle_area_proof (A B a b : ℝ) (h : ∃ C : ℝ, C = 2 * π / 3) :
  c = 3 → max_triangle_area A B h.some a b c → 
  (1/2 * a * b * sin (2 * π / 3) = 3 * sqrt 3 / 4) :=
by
  sorry

end find_angle_C_proof_max_triangle_area_proof_l755_755491


namespace area_ratio_triangle_quadrilateral_find_angle_CBL_l755_755532

-- Define the given conditions
def triangle (A B C P L M F : Type) : Prop :=
  BM : Point, MC : Point, α : ℕ, β : ℕ, α = 2, β = 5, 
  is_triangle A B C, Point_on_line_segment M B C, 
  BM: MC = 2 : 5,
  is_bisector BL, 
  is_perpendicular (BL, AM),

-- Part (a) proof statement
theorem area_ratio_triangle_quadrilateral 
  (A B C P L M : Type)
  [is_triangle A B C] 
  [Point_on_line_segment M B C]
  [BM_ratio_MC M B C 2 5]
  [angle_bisector BL A B C]
  [is_perpendicular (BL, AM)] :
  area_ratio (triangle_area (A B P), quadrilateral_area (L P M C)) = 9 / 40 :=
sorry

-- Part (b) proof statement
theorem find_angle_CBL 
  (A B C L M F : Type)
  [is_triangle A B C] 
  [Point_on_line_segment M B C]
  [BM_ratio_MC M B C 2 5]
  [Point_on_segment F M C]
  [MF_FC_ratio 1 4]
  [is_perpendicular LF BC] :
  angle_CBL (A B C L F) = arccos (3 * sqrt(3) / (2 * sqrt(7))) :=
sorry

end area_ratio_triangle_quadrilateral_find_angle_CBL_l755_755532


namespace existence_of_root_l755_755566

noncomputable def f (x : ℝ) : ℝ := 2^x - x - 2

theorem existence_of_root : ∃ x ∈ Ioo (-2 : ℝ) (-1), f x = 0 := by
  sorry

end existence_of_root_l755_755566


namespace max_k_value_l755_755777

theorem max_k_value (S : Finset ℕ) (hS : S.card = 10) 
  (A : Finset (Finset ℕ)) (hA : ∀ x ∈ A, x ≠ ∅) 
  (h_inter : ∀ (a b ∈ A), a ≠ b → (a ∩ b).card ≤ 2) : 
  A.card ≤ 175 :=
by sorry

end max_k_value_l755_755777


namespace repeating_decimal_as_fraction_l755_755309

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755309


namespace bills_needed_can_pay_groceries_l755_755525

theorem bills_needed_can_pay_groceries 
  (cans_of_soup : ℕ := 6) (price_per_can : ℕ := 2)
  (loaves_of_bread : ℕ := 3) (price_per_loaf : ℕ := 5)
  (boxes_of_cereal : ℕ := 4) (price_per_box : ℕ := 3)
  (gallons_of_milk : ℕ := 2) (price_per_gallon : ℕ := 4)
  (apples : ℕ := 7) (price_per_apple : ℕ := 1)
  (bags_of_cookies : ℕ := 5) (price_per_bag : ℕ := 3)
  (bottles_of_olive_oil : ℕ := 1) (price_per_bottle : ℕ := 8)
  : ∃ (bills_needed : ℕ), bills_needed = 4 :=
by
  let total_cost := (cans_of_soup * price_per_can) + 
                    (loaves_of_bread * price_per_loaf) +
                    (boxes_of_cereal * price_per_box) +
                    (gallons_of_milk * price_per_gallon) +
                    (apples * price_per_apple) +
                    (bags_of_cookies * price_per_bag) +
                    (bottles_of_olive_oil * price_per_bottle)
  let bills_needed := (total_cost + 19) / 20   -- Calculating ceiling of total_cost / 20
  sorry

end bills_needed_can_pay_groceries_l755_755525


namespace possible_values_of_k_l755_755432

theorem possible_values_of_k :
  ∃ (k_set : Set ℚ),
    (∀ x, x ∈ {y : ℚ | y^2 + y - 6 = 0} ↔ x ∈ ({2, -3} : Set ℚ))
    ∧ k_set = {k | ∀ x, x ∈ {y | k * y + 1 = 0} → x ∈ {2, -3}}
    ∧ k_set = {0, -1/2, 1/3} :=
by
  sorry

end possible_values_of_k_l755_755432


namespace correct_proposition_l755_755803

theorem correct_proposition (z : ℂ) (a b : ℝ) :
  (¬ (∀ z : ℂ, z^2 ≥ 0)) ∧
  (¬ (∀ a b : ℝ, a > b → a + complex.i > b + complex.i)) ∧
  ( ¬(∀ a : ℝ, (a + 1 : ℝ) * complex.i = 0 → true)) ∧
  (let w := (1 / complex.i) in (w^3 + 1).im > 0 ∧ (w^3 + 1).re > 0) :=
by {
  sorry
}

end correct_proposition_l755_755803


namespace green_fish_count_l755_755901

theorem green_fish_count (B O G : ℕ) (H1 : B = 40) (H2 : O = B - 15) (H3 : 80 = B + O + G) : G = 15 := 
by 
  sorry

end green_fish_count_l755_755901


namespace limit_of_n_bn_l755_755707

noncomputable def L (x : ℝ) : ℝ := x - x^2 / 2

def b_n (n : ℕ) : ℝ := 
  let f := λ x, L x
  (Nat.iterate f n) (25 / n.to_real)

theorem limit_of_n_bn :
  Filter.Tendsto (λ n : ℕ, n * b_n n) Filter.atTop (nhds (50 / 23)) := by
  sorry

end limit_of_n_bn_l755_755707


namespace age_ratio_l755_755579

variable (R D : ℕ)

theorem age_ratio (h1 : D = 24) (h2 : R + 6 = 38) : R / D = 4 / 3 := by
  sorry

end age_ratio_l755_755579


namespace jeff_ends_at_multiple_of_4_l755_755878

open Classical

noncomputable def prob_end_multiple_of_4 : ℚ :=
  let prob_picking_4_8_12 := (4 / 15: ℚ)
  let prob_picking_6_10_14 := (3 / 15: ℚ)
  let prob_picking_2_14 := (2 / 15: ℚ)
  let prob_spin_SS_RL_LR := 1 / 3 * 1 / 3 * 3
  let prob_spin_LL := 1 / 3 * 1 / 3
  let prob_spin_RR := 1 / 3 * 1 / 3
  prob_picking_4_8_12 * prob_spin_SS_RL_LR + prob_picking_6_10_14 * prob_spin_LL + prob_picking_2_14 * prob_spin_RR

theorem jeff_ends_at_multiple_of_4 : prob_end_multiple_of_4 = 17 / 135 := sorry

end jeff_ends_at_multiple_of_4_l755_755878


namespace domain_of_tan_l755_755560

noncomputable def is_excluded_from_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 1 + 6 * k

theorem domain_of_tan {x : ℝ} :
  ∀ x, ¬ is_excluded_from_domain x ↔ ¬ ∃ k : ℤ, x = 1 + 6 * k := 
by 
  sorry

end domain_of_tan_l755_755560


namespace fraction_eq_repeating_decimal_l755_755363

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755363


namespace speed_plane_east_l755_755659

-- Definitions of the conditions
def speed_west : ℕ := 275
def time_hours : ℝ := 3.5
def distance_apart : ℝ := 2100

-- Theorem statement to prove the speed of the plane traveling due East
theorem speed_plane_east (v: ℝ) 
  (h: (v + speed_west) * time_hours = distance_apart) : 
  v = 325 :=
  sorry

end speed_plane_east_l755_755659


namespace constant_sequence_general_term_l755_755815

/-- Define the sequence {a_n} satisfying the given conditions --/
def a : ℕ → ℝ
| 0       := x
| 1       := y
| (n + 2) := (a (n + 1) * a n + 1) / (a (n + 1) + a n)

/-- Prove that for given values of x and y, a_n becomes constant for n ≥ n_0 --/
theorem constant_sequence (x y : ℝ) (n : ℕ) :
  (x = 1 ∧ y ≠ -1) ∨ (x = -1 ∧ y ≠ 1) ∨ (y = 1 ∧ x ≠ -1) ∨ (y = -1 ∧ x ≠ 1) → 
  ∃ (n_0 : ℕ), ∀ m ≥ n_0, a m = 1 ∨ a m = -1 :=
sorry

/-- Prove the general term formula for the sequence {a_n} --/
theorem general_term (x y : ℝ) (n : ℕ) :
  n ≥ 2 →
  a n = (2 * ((y - 1) / (y + 1)) ^ fibonacci (n - 1) * ((x - 1) / (x + 1)) ^ fibonacci (n - 2)
    / (1 - ((y - 1) / (y + 1)) ^ fibonacci (n - 1) * ((x - 1) / (x + 1)) ^ fibonacci (n - 2))) - 1 :=
sorry

end constant_sequence_general_term_l755_755815


namespace tangent_normal_curve_l755_755133

noncomputable def tangent_line (t: ℝ) (x: ℝ -> ℝ) (y: ℝ -> ℝ) : ℝ -> ℝ :=
  λ x₀, (7/4) * x₀ + 1/16

noncomputable def normal_line (t: ℝ) (x: ℝ -> ℝ) (y: ℝ -> ℝ) : ℝ -> ℝ :=
  λ x₀, (-4/7) * x₀ + 101/56

theorem tangent_normal_curve (t₀ : ℝ) (h₀ : t₀ = 2)
  (x : ℝ -> ℝ) (hx : ∀ t, x t = (1 + t) / (t^2))
  (y : ℝ -> ℝ) (hy : ∀ t, y t = (3 / (2 * t^2)) + (2 / t)) :
  (∀ x₀, tangent_line t₀ x y x₀ = (7/4) * x₀ + 1/16) ∧
  (∀ x₀, normal_line t₀ x y x₀ = (-4/7) * x₀ + 101/56) :=
by
  sorry

end tangent_normal_curve_l755_755133


namespace trig_identity_l755_755634

theorem trig_identity (α : ℝ) : 
  (sin (α - π / 6)) ^ 2 + (sin (α + π / 6)) ^ 2 - (sin α) ^ 2 = 1 / 2 := 
by 
  sorry

end trig_identity_l755_755634


namespace decreasing_interval_l755_755571

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp (-x)

theorem decreasing_interval : {x : ℝ | x > 0} = (0, +∞) :=
by
  sorry

end decreasing_interval_l755_755571


namespace fraction_eq_repeating_decimal_l755_755365

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755365


namespace smart_charging_piles_equation_l755_755478

-- Define the problem's given conditions
variable (x : ℝ)
variable (first_month_piles third_month_piles : ℝ)

-- Define specific values according to the problem’s conditions
def first_month_piles := 302
def third_month_piles := 503

-- Statement of the problem in Lean
theorem smart_charging_piles_equation :
  302 * (1 + x)^2 = 503 :=
sorry

end smart_charging_piles_equation_l755_755478


namespace relationship_f_2011_2012_2013_l755_755792

noncomputable def f : ℝ → ℝ := sorry

theorem relationship_f_2011_2012_2013
  (H1 : ∀ x, f(x + 1) = f(-(x + 1)))
  (H2 : ∀ x, f(x + 2) = -f(x))
  (H3 : ∀ x1 x2, 1 ≤ x1 → x1 < x2 → x2 ≤ 3 → (f(x2) - f(x1)) * (x2 - x1) < 0) :
  f(2013) > f(2012) ∧ f(2012) > f(2011) :=
sorry

end relationship_f_2011_2012_2013_l755_755792


namespace parabola_standard_equation_l755_755802

variable (a : ℝ) (h : a < 0)

theorem parabola_standard_equation :
  (∃ p : ℝ, y^2 = -2 * p * x ∧ p = -2 * a) → y^2 = 4 * a * x :=
by
  sorry

end parabola_standard_equation_l755_755802


namespace solve_y_equation_l755_755930

noncomputable def solve_y : ℚ :=
  let y := (500 * 1 : ℚ) / 15 in
  y

theorem solve_y_equation (y : ℚ) :
  2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = solve_y := by
  intro h
  sorry

end solve_y_equation_l755_755930


namespace quadrilateral_area_l755_755964

theorem quadrilateral_area (a b c d e f : ℝ) : 
    (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 :=
    by sorry

noncomputable def quadrilateral_area_formula (a b c d e f : ℝ) : ℝ :=
    if H : (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 then 
    (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2)
    else 0

-- Ensure that the computed area matches the expected value
example (a b c d e f : ℝ) (H : (a^2 + c^2 - b^2 - d^2)^2 ≤ 4 * e^2 * f^2) : 
    quadrilateral_area_formula a b c d e f = 
        (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2) :=
by simp [quadrilateral_area_formula, H]

end quadrilateral_area_l755_755964


namespace basketball_league_l755_755468

theorem basketball_league (n : ℕ) :
  (let total_teams := n + 3 * n,
       total_matches := total_teams * (total_teams - 1) / 2,
       match_wins_women := 3 * (total_matches / 8),
       match_wins_men := 5 * (total_matches / 8) in
   total_matches = (match_wins_women + match_wins_men)
   ∧ match_wins_women / match_wins_men = 3 / 5)
   → n = 4 := 
begin
  sorry
end

end basketball_league_l755_755468


namespace sigma_algebra_convergence_l755_755009

-- Definitions based on conditions
variables {Ω : Type*} {P : ProbabilityMeasure Ω}
variables {ξ ξ₁ ξ₂ : ℕ → Ω → ℝ}

-- Condition: Convergence in probability
def converges_in_probability (ξn : ℕ → Ω → ℝ) (ξ : Ω → ℝ) (P : ProbabilityMeasure Ω) : Prop :=
∀ ε > 0, limsup_at_top (λ n, P {ω | abs (ξn n ω - ξ ω) > ε}) = 0

def is_complete_probability_space (P : ProbabilityMeasure Ω) : Prop :=
∀ E, P E = 0 → ∀ ε, ∃ F ⊆ E, P F = ε

-- Theorem statement
theorem sigma_algebra_convergence
  (h_complete : is_complete_probability_space P)
  (h_conv : converges_in_probability ξ₁ ξ P) :
  sigma_algebra ξ₁ = sigma_algebra (λ n, if n = 0 then ξ else ξ₁ (n - 1)) :=
sorry

end sigma_algebra_convergence_l755_755009


namespace problem1_solution_problem2_solution_l755_755814

noncomputable def g (x : ℝ) := 3^x
noncomputable def h (x : ℝ) := 9^x
noncomputable def p (x : ℝ) := g x / (g x + real.sqrt 3)

theorem problem1_solution :
  ∃ x : ℝ, h(x) - 8 * g(x) - h(1) = 0 ∧ x = 2 :=
by
  use 2
  have h1 : h(1) = 9 := by sorry
  have hx : h(2) = g (2) ^ 2 := by sorry
  have hx_reduced : g 2 ^ 2 - 8 * g 2 - 9 = 0 := by sorry
  have g2 : g 2 = 9 := by sorry
  exact ⟨hx, g2⟩

theorem problem2_solution :
  finset.sum (finset.range 2016) (λ i, p (i / 2016)) = (2015 / 2) :=
by
  have sum_result : finset.range 2016 = finset.range 1008 ∪ finset.range 1008 := by sorry
  have sum_half : finset.sum (finset.range 1008) (λ i, (p (i / 2016) + p ((2015 - i) / 2016))) = 1007 := by sorry
  have middle_term : p (1008 / 2016) = 1 / 2 := by sorry
  exact (sum_half + middle_term)

end problem1_solution_problem2_solution_l755_755814


namespace circle_tangency_problem_l755_755222

theorem circle_tangency_problem
  {Ω ω ω1 : Circle}
  (C : Ω.center ∈ ω)
  (AB : Chord Ω)
  (E : Midpoint AB)
  (tangent_at_E : TangentPoint ω AB E)
  (D Z F : Point)
  (tangents_DF : TangentPoints ω1 Ω ω AB D Z F)
  (P : ray_intersection_C ω1 Ω C D AB)
  (M : MidpointMajorArc Ω AB) :
  tan (angle ω1.radius₂ ω.radius₂ C D P E Z) = (P.distance E) / (C.distance M) := sorry

end circle_tangency_problem_l755_755222


namespace impossible_to_color_25_cells_l755_755875

theorem impossible_to_color_25_cells :
  ¬ ∃ (n : ℕ) (n_k : ℕ → ℕ), n = 25 ∧ (∀ k, k > 0 → k < 5 → (k % 2 = 1 → ∃ c : ℕ, n_k c = k)) :=
by
  sorry

end impossible_to_color_25_cells_l755_755875


namespace binary_to_base5_l755_755230

theorem binary_to_base5 (n : ℕ) (h : n = 0b1101011) : nat.to_digits 5 n = [4, 1, 2] :=
by
  sorry

end binary_to_base5_l755_755230


namespace range_of_m_l755_755892

-- Defining the conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_four_distinct_zeroes (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ a b : ℝ, 0 < b ∧ b < a ∧
  ∀ x, (f x - m = 0) ↔ (x = -a ∨ x = -b ∨ x = b ∨ x = a)

-- Statement of the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  is_even_function f →
  has_four_distinct_zeroes f m →
  m ∈ (set.univ \ {0}) :=
by
  sorry

end range_of_m_l755_755892


namespace domain_of_f_l755_755725

def domain_function (f : ℝ) : Prop :=
  ∀ x : ℝ, f = sqrt (4 - sqrt (6 - sqrt (7 - x))) → -93 ≤ x ∧ x ≤ 7

theorem domain_of_f:
  domain_function (λ x, sqrt (4 - sqrt (6 - sqrt (7 - x)))) :=
by
  sorry

end domain_of_f_l755_755725


namespace supremum_neg_frac_bound_l755_755512

noncomputable def supremum_neg_frac (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem supremum_neg_frac_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  supremum_neg_frac a b ≤ - 9 / 2 :=
sorry

end supremum_neg_frac_bound_l755_755512


namespace sum_four_digit_numbers_divisible_by_5_correct_l755_755611

noncomputable def sum_four_digit_numbers_divisible_by_5 : ℕ :=
  let a : ℕ := 1000
  let d : ℕ := 5
  let l : ℕ := 9995
  let n : ℕ := (l - a) / d + 1
  sum := n * (a + l) / 2
  sum

theorem sum_four_digit_numbers_divisible_by_5_correct :
  sum_four_digit_numbers_divisible_by_5 = 9895500 := 
by
  -- We will provide the proof here
  sorry

end sum_four_digit_numbers_divisible_by_5_correct_l755_755611


namespace maria_drank_bottles_l755_755016

theorem maria_drank_bottles (total : ℝ) (sister_drank : ℝ) (left : ℝ) (maria_drank : ℝ) :
  total = 45.0 ∧ sister_drank = 8.0 ∧ left = 23.0 ∧ total - sister_drank - maria_drank = left → maria_drank = 14.0 :=
by
  intros h,
  sorry

end maria_drank_bottles_l755_755016


namespace steiner_symmetrization_convex_l755_755918

variables {Point : Type} [OrderedRing Point] -- Assuming points are in some ordered ring space

structure ConvexPolygon (Point : Type) [OrderedRing Point] :=
(points : set Point)
(is_convex : ∀ A B ∈ points, ∀ t ∈ set.Icc (0 : Point) 1, t • A + (1 - t) • B ∈ points)

def SteinerSymmetrization (M : ConvexPolygon Point) (l : Point → Point) : ConvexPolygon Point := sorry

theorem steiner_symmetrization_convex (M : ConvexPolygon Point) (l : Point → Point) :
  let M' := SteinerSymmetrization M l in
  ∀ A B ∈ M'.points, ∀ t ∈ set.Icc (0 : Point) 1, t • A + (1 - t) • B ∈ M'.points := sorry

end steiner_symmetrization_convex_l755_755918


namespace repeating_decimal_is_fraction_l755_755293

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755293


namespace math_problem_l755_755697

-- Define the conditions
def a := -6
def b := 2
def c := 1 / 3
def d := 3 / 4
def e := 12
def f := -3

-- Statement of the problem
theorem math_problem : a / b + (c - d) * e + f^2 = 1 :=
by
  sorry

end math_problem_l755_755697


namespace edward_money_unknown_l755_755239

-- Definitions based on given conditions
def amount_spent : ℕ := 6
def book_cost : ℕ := 3
def books_bought : ℕ := 2
def initial_money : ℕ := sorry -- We cannot determine without knowing the initial amount

-- Question rephrased as a proof problem
theorem edward_money_unknown (initial_money: ℕ) :
  initial_money - amount_spent = initial_money - 6 := 
begin
  -- The condition states $6 was spent
  -- Proper solution unable to be determined without the initial money
  sorry
end

end edward_money_unknown_l755_755239


namespace sum_of_positive_real_solutions_l755_755380

theorem sum_of_positive_real_solutions :
  ∀ (x : ℝ), x > 0 ∧ (2 * sin (2 * x) * (sin (2 * x) - sin (1007 * π ^ 2 / x)) = sin (4 * x) - 1)
  → x = 1080 * π :=
sorry

end sum_of_positive_real_solutions_l755_755380


namespace supermarket_problem_l755_755649

-- Define that type A costs x yuan and type B costs y yuan
def cost_price_per_item (x y : ℕ) : Prop :=
  (10 * x + 8 * y = 880) ∧ (2 * x + 5 * y = 380)

-- Define purchasing plans with the conditions described
def purchasing_plans (a : ℕ) : Prop :=
  ∀ a : ℕ, 24 ≤ a ∧ a ≤ 26

theorem supermarket_problem : 
  (∃ x y, cost_price_per_item x y ∧ x = 40 ∧ y = 60) ∧ 
  (∃ n, purchasing_plans n ∧ n = 3) :=
by
  sorry

end supermarket_problem_l755_755649


namespace roots_equation_satisfied_l755_755511

theorem roots_equation_satisfied {p q : ℝ} (h_roots : ∀ x, x^2 - 3 * x * real.sqrt 3 + 3 = 0 → (x = p ∨ x = q)) :
  p^6 + q^6 = 99171 :=
by 
  sorry

end roots_equation_satisfied_l755_755511


namespace line_intersects_circle_l755_755913

variable {a x_0 y_0 : ℝ}

theorem line_intersects_circle (h1: x_0^2 + y_0^2 > a^2) (h2: a > 0) : 
  ∃ (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 = a ^ 2) ∧ (x_0 * p.1 + y_0 * p.2 = a ^ 2) :=
sorry

end line_intersects_circle_l755_755913


namespace steer_weight_conversion_l755_755591

theorem steer_weight_conversion (kilograms_per_pound : ℝ) (steer_weight_kg : ℝ) : 
  kilograms_per_pound = 0.4536 ∧ steer_weight_kg = 325 → 
  (steer_weight_kg / kilograms_per_pound).to_nearest_tenth = 716.8 :=
sorry

end steer_weight_conversion_l755_755591


namespace max_points_on_2010_dim_spheres_l755_755666

theorem max_points_on_2010_dim_spheres (S : Set (ℝ^2010)) (r : ℝ) (hS : ∀ s ∈ S, ∃ c : ℝ^2010, s = { p : ℝ^2010 | dist p c = r }) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end max_points_on_2010_dim_spheres_l755_755666


namespace urn_contains_four_red_three_blue_l755_755685

noncomputable def urn_problem : Prop :=
  let initial_red := 1
  let initial_blue := 1
  let total_operations := 5
  let final_total_balls := 7
  let desired_red := 4
  let desired_blue := 3
  let probability := 1 / 6
  (probability == sorry)  -- This is where we would compute and compare probabilities

theorem urn_contains_four_red_three_blue :
  urn_problem :=
by sorry

end urn_contains_four_red_three_blue_l755_755685


namespace total_cans_collected_l755_755921

variables (bags_saturday bags_sunday cans_per_bag : ℕ)
axiom bags_saturday_eq : bags_saturday = 4
axiom bags_sunday_eq : bags_sunday = 3
axiom cans_per_bag_eq : cans_per_bag = 6

theorem total_cans_collected (bags_saturday bags_sunday cans_per_bag : ℕ) : 
  bags_saturday + bags_sunday = 7 → (bags_saturday + bags_sunday) * cans_per_bag = 42 :=
  by intros h1; rw [bags_saturday_eq, bags_sunday_eq, cans_per_bag_eq];
     rw h1; exact absurd (by ring) (not_not_intro rfl)

end total_cans_collected_l755_755921


namespace chiming_time_is_5_l755_755549

-- Define the conditions for the clocks
def queen_strikes (h : ℕ) : Prop := (2 * h) % 3 = 0
def king_strikes (h : ℕ) : Prop := (3 * h) % 2 = 0

-- Define the chiming synchronization at the same time condition
def chiming_synchronization (h: ℕ) : Prop :=
  3 * h = 2 * ((2 * h) + 2)

-- The proof statement
theorem chiming_time_is_5 : ∃ h: ℕ, queen_strikes h ∧ king_strikes h ∧ chiming_synchronization h ∧ h = 5 :=
by
  sorry

end chiming_time_is_5_l755_755549


namespace find_three_digit_perfect_square_l755_755251

noncomputable def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n % 100) / 10) * (n % 10)

theorem find_three_digit_perfect_square :
  ∃ (n H : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (n = H * H) ∧ (digit_product n = H - 1) :=
by {
  sorry
}

end find_three_digit_perfect_square_l755_755251


namespace polygon_sides_l755_755797

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end polygon_sides_l755_755797


namespace yellow_green_block_weight_difference_l755_755501

theorem yellow_green_block_weight_difference :
  let yellow_weight := 0.6
  let green_weight := 0.4
  yellow_weight - green_weight = 0.2 := by
  sorry

end yellow_green_block_weight_difference_l755_755501


namespace min_b_for_quadratic_factorization_l755_755378

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end min_b_for_quadratic_factorization_l755_755378


namespace ratio_of_customers_third_week_l755_755881

def ratio_of_customers (c1 c3 : ℕ) (s k t : ℕ) : Prop := s = 500 ∧ k = 50 ∧ t = 760 ∧ c1 = 35 ∧ c3 = 105 ∧ (t - s - k) - (35 + 70) = c1 ∧ c3 = 105 ∧ (c3 / c1 = 3)

theorem ratio_of_customers_third_week (c1 c3 : ℕ) (s k t : ℕ)
  (h1 : s = 500)
  (h2 : k = 50)
  (h3 : t = 760)
  (h4 : c1 = 35)
  (h5 : c3 = 105)
  (h6 : (t - s - k) - (35 + 70) = c1)
  (h7 : c3 = 105) :
  (c3 / c1) = 3 :=
  sorry

end ratio_of_customers_third_week_l755_755881


namespace minimum_percentage_bad_work_l755_755071

-- Definitions based on conditions
def P_good : ℝ := 0.80
def P_bad : ℝ := 0.20
def P_error : ℝ := 0.10

-- Prove that the integer part of the minimum percentage of bad works among those rechecked by the experts is 66%
theorem minimum_percentage_bad_work :
  let good_misclassified_as_bad := P_error * P_good,
      actual_bad_correctly_identified := P_bad * (1 - P_error),
      total_rechecked := good_misclassified_as_bad + actual_bad_correctly_identified,
      percentage_bad_rechecked := (actual_bad_correctly_identified / total_rechecked) * 100 in
  percentage_bad_rechecked.to_int = 66 :=
by
  -- Skipping the proof steps
  sorry

end minimum_percentage_bad_work_l755_755071


namespace fraction_eq_repeating_decimal_l755_755364

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755364


namespace evaluate_expression_l755_755724

theorem evaluate_expression : 
  3 - (-3)^(3-(-3)) * 2 = -1455 := 
by 
  sorry

end evaluate_expression_l755_755724


namespace max_sum_is_103_sum_50_impossible_possible_values_of_T_59_l755_755136

variables {M A R D T E I K U : ℕ}

/-- Question 1: Maximum Sum --/
theorem max_sum_is_103 (hM : 1 ≤ M ∧ M ≤ 9) (hA : 1 ≤ A ∧ A ≤ 9) (hR : 1 ≤ R ∧ R ≤ 9) 
    (hD : 1 ≤ D ∧ D ≤ 9) (hT : 1 ≤ T ∧ T ≤ 9) (hE : 1 ≤ E ∧ E ≤ 9) (hI : 1 ≤ I ∧ I ≤ 9) 
    (hK : 1 ≤ K ∧ K ≤ 9) (hU : 1 ≤ U ∧ U ≤ 9)
    (h_unique : list.nodup [M, A, R, D, T, E, I, K, U]) : 
  4 * M + 4 * A + R + D + 2 * T + E + I + K + U ≤ 103 := 
sorry

/-- Question 2: Sum 50 is impossible --/
theorem sum_50_impossible (hM : 1 ≤ M ∧ M ≤ 9) (hA : 1 ≤ A ∧ A ≤ 9) (hR : 1 ≤ R ∧ R ≤ 9) 
    (hD : 1 ≤ D ∧ D ≤ 9) (hT : 1 ≤ T ∧ T ≤ 9) (hE : 1 ≤ E ∧ E ≤ 9) (hI : 1 ≤ I ∧ I ≤ 9) 
    (hK : 1 ≤ K ∧ K ≤ 9) (hU : 1 ≤ U ∧ U ≤ 9)
    (h_unique : list.nodup [M, A, R, D, T, E, I, K, U]) : 
  ¬ (4 * M + 4 * A + R + D + 2 * T + E + I + K + U = 50) := 
sorry

/-- Question 3: Possible values of T for sum 59 --/
theorem possible_values_of_T_59 (hM : 1 ≤ M ∧ M ≤ 9) (hA : 1 ≤ A ∧ A ≤ 9) (hR : 1 ≤ R ∧ R ≤ 9) 
    (hD : 1 ≤ D ∧ D ≤ 9) (hT : 1 ≤ T ∧ T ≤ 9) (hE : 1 ≤ E ∧ E ≤ 9) (hI : 1 ≤ I ∧ I ≤ 9) 
    (hK : 1 ≤ K ∧ K ≤ 9) (hU : 1 ≤ U ∧ U ≤ 9)
    (h_unique : list.nodup [M, A, R, D, T, E, I, K, U]) : 
  (4 * M + 4 * A + R + D + 2 * T + E + I + K + U = 59) → (T = 5 ∨ T = 2) := 
sorry

end max_sum_is_103_sum_50_impossible_possible_values_of_T_59_l755_755136


namespace total_production_first_four_days_max_min_production_difference_total_wage_l755_755645

def planned_production_per_day := 100
def actual_deviation : List Int := [5, -2, -4, 13, -10, 16, -9]
def planned_production_per_week := 700
def daily_deviations_first_four_days := [5, -2, -4, 13]

theorem total_production_first_four_days :
  (4 * planned_production_per_day + daily_deviations_first_four_days.sum) = 412 :=
by
  -- planned production for 4 days
  have planned_prod := 4 * planned_production_per_day
  -- total deviation for 4 days
  have total_dev := daily_deviations_first_four_days.sum
  -- total production
  have total_prod := planned_prod + total_dev
  show total_prod = 412
  sorry

theorem max_min_production_difference :
  (actual_deviation.max - actual_deviation.min) = 26 :=
by
  -- max production deviation
  have max_dev := actual_deviation.max
  -- min production deviation
  have min_dev := actual_deviation.min
  -- difference between max and min
  have diff := max_dev - min_dev
  show diff = 26
  sorry

theorem total_wage :
  let total_production := planned_production_per_week + actual_deviation.sum in
  let base_wage := planned_production_per_week * 60 in
  let additional_wage := (total_production - planned_production_per_week) * (60 + 15) in
  (base_wage + additional_wage) = 42675 :=
by
  -- total deviation for the week
  have total_dev := actual_deviation.sum
  -- total production
  have total_prod := planned_production_per_week + total_dev
  -- base wage
  have base_w := planned_production_per_week * 60
  -- additional wage for extra bicycles
  have add_w := (total_prod - planned_production_per_week) * (60 + 15)
  -- total wage
  have tot_wage := base_w + add_w
  show tot_wage = 42675
  sorry

end total_production_first_four_days_max_min_production_difference_total_wage_l755_755645


namespace order_of_trig_expressions_l755_755625

theorem order_of_trig_expressions (x : ℝ) (h : x ∈ Ioo (-1/2 : ℝ) 0) :
  let α1 := Real.cos (Real.sin (x * Real.pi))
  let α2 := Real.sin (Real.cos (x * Real.pi))
  let α3 := Real.cos ((x + 1) * Real.pi)
  in α3 < α2 ∧ α2 < α1 :=
by
  sorry

end order_of_trig_expressions_l755_755625


namespace maximum_value_of_f_l755_755057

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log (x + 2) / Real.log 2

theorem maximum_value_of_f : ∃ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x = 3 :=
by
  use (-1 : ℝ)
  split
  · norm_num
  · norm_num
  · sorry

end maximum_value_of_f_l755_755057


namespace mean_of_remaining_quiz_scores_l755_755578

theorem mean_of_remaining_quiz_scores (k : ℕ) (hk : k > 12) 
  (mean_k : ℝ) (mean_12 : ℝ) 
  (mean_class : mean_k = 8) 
  (mean_12_group : mean_12 = 14) 
  (mean_correct : mean_12 * 12 + mean_k * (k - 12) = 8 * k) :
  mean_k * (k - 12) = (8 * k - 168) := 
by {
  sorry
}

end mean_of_remaining_quiz_scores_l755_755578


namespace daily_sales_volume_eq_daily_profit_eq_6000_max_daily_profit_l755_755987

def cost_per_box := 40
def min_selling_price := 45

-- Condition: At a selling price of 45 yuan, 700 boxes are sold.
def initial_selling_price := 45
def initial_sales := 700

-- Condition: For each 1 yuan increase in price, 20 fewer boxes are sold.
def sales_decrease_per_yuan := 20

-- Part 1: Functional Relationship between daily sales volume y and selling price x
theorem daily_sales_volume_eq (x : ℝ) (h : x ≥ min_selling_price ∧ x < 80) :
  (initial_sales - sales_decrease_per_yuan * (x - initial_selling_price)) = (-20 * x + 1600) :=
sorry

-- Part 2: Selling price for a daily profit of 6000 yuan
theorem daily_profit_eq_6000 (x : ℝ) (profit_goal : ℝ := 6000) :
  let profit_per_box := x - cost_per_box in
  let total_profit := profit_per_box * (initial_sales - sales_decrease_per_yuan * (x - initial_selling_price)) in
  total_profit = profit_goal → x = 50 :=
sorry

-- Part 3: Maximizing Daily Profit
theorem max_daily_profit (x : ℝ) :
  let profit_per_box := x - cost_per_box in
  let profit_function := (-20 * x ^ 2 + 2400 * x - 64000) in
  profit_function = (-20 * (x - 60) ^ 2 + 8000) → x = 60 ∧ profit_function = 8000 :=
sorry

end daily_sales_volume_eq_daily_profit_eq_6000_max_daily_profit_l755_755987


namespace fraction_eq_repeating_decimal_l755_755360

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755360


namespace equation_of_tangent_line_at_1_maximum_value_of_f_l755_755430

-- Definitions based on conditions:
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 8

-- Statement for the equation of the tangent line at the point (1, f(1))
theorem equation_of_tangent_line_at_1 :
  let f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 8
  let tangent_at_1(x y : ℝ) : Prop := y = -3 * (x - 1) + 6
  ∀ x y, tangent_at_1 x y → 3 * x + y - 9 = 0 :=
by 
  sorry

-- Statement for the maximum value of function f
theorem maximum_value_of_f :
  let f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 8
  ∀ x, f 0 = 8 :=
by
  sorry

end equation_of_tangent_line_at_1_maximum_value_of_f_l755_755430


namespace probability_at_least_four_same_face_l755_755732

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l755_755732


namespace max_triangles_formed_l755_755593

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end max_triangles_formed_l755_755593


namespace part1_part2_l755_755823

variables {L1 L2 : ℝ → ℝ → Prop}
variables {P B C : Prod ℝ ℝ}
variables {L : ℝ → ℝ → ℝ → Prop}

-- Definitions
def L1 (x y : ℝ) : Prop := x + y - 1 = 0
def L2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def P : Prod ℝ ℝ := (-1, 2)
def L (a x y : ℝ) : Prop := a * x - y - 2 * a + 1 = 0

-- Statements to be proved
theorem part1 : (L a P.1 P.2) → a = -1/3 := by 
  sorry

theorem part2 (a:ℝ) (ha: (∀ x y, L a x y → L1 x y → false)) : 
  (area_of_triangle (L 1) L1 L2 = 12) := by
  sorry

end part1_part2_l755_755823


namespace count_liuhe_less_than_2012_l755_755455

def digit_sum (n : ℕ) : ℕ :=
  -- Assuming the definition of digit_sum where it computes the sum of the digits of n
  sorry

def is_liuhe (m : ℕ) : Prop :=
  m % 6 = 0 ∧ digit_sum m % 6 = 0

theorem count_liuhe_less_than_2012 : 
  (Finset.filter is_liuhe (Finset.range 2012)).card = 168 :=
by
  sorry

end count_liuhe_less_than_2012_l755_755455


namespace solution_set_l755_755011

variable {f g : ℝ → ℝ}
variable [Differentiable ℝ f] [Differentiable ℝ g]

-- Definitions of odd and even functions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Given conditions
axiom f_odd : odd_function f
axiom g_even : even_function g
axiom condition3 : ∀ x : ℝ, x < 0 → (deriv f x) * (g x) + (f x) * (deriv g x) > 0
axiom g_one : g 1 = 0

-- Proving the solution for the inequality
theorem solution_set : {x : ℝ | f x * g x < 0} = (Set.Ioo  (-∞) (-1) ∪ Set.Ioo (0) (1)) :=
sorry

end solution_set_l755_755011


namespace conic_section_foci_polar_coordinates_l755_755463

theorem conic_section_foci_polar_coordinates :
  ∀ (ρ θ : ℝ), (ρ = 16 / (5 - 3 * cos θ)) → ((ρ = 0 ∧ θ = 0) ∨ (ρ = 6 ∧ θ = 0)) :=
by
  sorry

end conic_section_foci_polar_coordinates_l755_755463


namespace binom_expansion_coeff_l755_755072

/-- Prove the coefficient of x^6 y^2 in the expansion of (8x + 3y)^3 * (x + 2y)^5 is 446 -/
theorem binom_expansion_coeff : 
  let a := ∑ k in Finset.range 4, Nat.descFactorial 3 k * (-2)^k
  let b := 2
  ∑ r in Finset.range 4, ∑ k in Finset.range 6, if r + k = 2 then 2^(3 + k - r) * nat.choose 3 r * nat.choose 5 k else 0 = 446 := 
by 
  have ha : a = 8 := 
    by simp [a] 
  have hb : b = 2 := 
    by simp [b]
  simp [ha, hb]
  sorry

end binom_expansion_coeff_l755_755072


namespace all_coins_same_face_or_equivalent_l755_755117

def coin_flip (coin : Type) : Type := coin  -- Definition of a coin flip result

-- Define the condition of 100 coins being tossed and all showing the same face
def all_show_same_face (coins : list coin) : Prop := 
  ∀ i j, i < coins.length ∧ j < coins.length → coins[i] = coins[j]

theorem all_coins_same_face_or_equivalent (coins : list coin) (h: all_show_same_face coins) : ∀ i j, i < coins.length → j < coins.length → coins[i] = coins [j] ↔ (coins[i] has_same_face := sorry

end all_coins_same_face_or_equivalent_l755_755117


namespace gcm_of_9_and_15_less_than_120_eq_90_l755_755995

theorem gcm_of_9_and_15_less_than_120_eq_90 
  (lcm_9_15 : Nat := Nat.lcm 9 15)
  (multiples : List Nat := List.range (120 / lcm_9_15) |> List.map (λ n => n * lcm_9_15)) : 
  lcm_9_15 = 45 ∧ multiples.max = some 90 := by
sorry

end gcm_of_9_and_15_less_than_120_eq_90_l755_755995


namespace neither_coffee_tea_juice_l755_755689

open Set

theorem neither_coffee_tea_juice (total : ℕ) (coffee : ℕ) (tea : ℕ) (both_coffee_tea : ℕ)
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) :
  total = 35 → 
  coffee = 18 → 
  tea = 15 → 
  both_coffee_tea = 7 → 
  juice = 6 → 
  juice_and_tea_not_coffee = 3 →
  (total - ((coffee + tea - both_coffee_tea) + (juice - juice_and_tea_not_coffee))) = 6 :=
sorry

end neither_coffee_tea_juice_l755_755689


namespace probability_line_circle_ne_intersect_l755_755119

noncomputable def die_probability (a b : ℕ) : ℚ :=
  if a < 1 ∨ a > 6 ∨ b < 1 ∨ b > 6 then 0 else
  if b < 2*a then 1 else 0

theorem probability_line_circle_ne_intersect :
  (∑ a in Finset.range 6, ∑ b in Finset.range 6, die_probability (a + 1) (b + 1)) / 36 = 2 / 3 := by
sorry

end probability_line_circle_ne_intersect_l755_755119


namespace pie_eating_contest_l755_755990

theorem pie_eating_contest :
  let pie1_first_student := 7 / 8
  let pie1_second_student := 5 / 6
  let pie2_first_student := 3 / 4
  let pie2_second_student := 2 / 3
  let total_first_student := pie1_first_student + pie2_first_student
  let total_second_student := pie1_second_student + pie2_second_student
  let difference := total_first_student - total_second_student
  difference = 1 / 8 :=
by
  sorry

end pie_eating_contest_l755_755990


namespace probability_at_least_6_heads_l755_755165

theorem probability_at_least_6_heads (n : ℕ) (p : ℚ) : n = 8 ∧ p = (3 / 128) :=
  let total_outcomes := 2 ^ n in
  let successful_outcomes := 3 + 2 + 1 in
  n = 8 ∧ total_outcomes = 256 ∧ p = (successful_outcomes.to_rat / total_outcomes.to_rat) := sorry

end probability_at_least_6_heads_l755_755165


namespace continuity_at_4_l755_755627

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end continuity_at_4_l755_755627


namespace repeating_decimal_eq_fraction_l755_755340

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755340


namespace repeating_decimal_equiv_fraction_l755_755287

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755287


namespace max_acute_angles_l755_755608

theorem max_acute_angles (n : ℕ) : 
  ∃ k : ℕ, k ≤ (2 * n / 3) + 1 :=
sorry

end max_acute_angles_l755_755608


namespace find_x_l755_755570

theorem find_x (x : ℝ) 
(h_mean : (70 + 110 + x + 60 + 50 + 220 + 100 + x + 90) / 9 = x) 
(h_median: list.median ([50, 60, 70, 90, 100, x, x, 110, 220]).sorted = x)
(h_mode : list.mode ([50, 60, 70, 90, 100, x, x, 110, 220]).head! = x) : 
  x = 100 :=
sorry

end find_x_l755_755570


namespace solve_y_equation_l755_755932

noncomputable def solve_y : ℚ :=
  let y := (500 * 1 : ℚ) / 15 in
  y

theorem solve_y_equation (y : ℚ) :
  2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = solve_y := by
  intro h
  sorry

end solve_y_equation_l755_755932


namespace fisherman_fish_distribution_l755_755924

theorem fisherman_fish_distribution :
  ∃(f : Fin 6 → ℕ), (∑ i, f i = 100) ∧ (∀ (i j : Fin 6), i ≠ j → f i ≠ f j)
  ∧ (∀ i : Fin 6, ∃ k : ℕ, (∑ j (H : j ≠ i), f j = 5 * k)  
  ∧ ∃ (m : Fin 6), ∃ k : ℕ, (∑ j, j ≠ m → f j = 5 * k)) :=
  sorry

end fisherman_fish_distribution_l755_755924


namespace fraction_of_repeating_decimal_l755_755326

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l755_755326


namespace repeating_fraction_equality_l755_755347

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755347


namespace minimum_boxes_to_eliminate_l755_755473

theorem minimum_boxes_to_eliminate (total_boxes remaining_boxes : ℕ) 
  (high_value_boxes : ℕ) (h1 : total_boxes = 30) (h2 : high_value_boxes = 10)
  (h3 : remaining_boxes = total_boxes - 20) :
  remaining_boxes ≥ high_value_boxes → remaining_boxes = 10 :=
by 
  sorry

end minimum_boxes_to_eliminate_l755_755473


namespace lines_parallel_l755_755818

theorem lines_parallel : ∀ (x y : ℝ), (x - y + 6 = 0) ∧ (2 * x - 2 * y + 3 = 0) → (∃ m b₁ b₂, (y = m * x + b₁) ∧ (y = m * x + b₂) ∧ b₁ ≠ b₂) :=
begin
  sorry
end

end lines_parallel_l755_755818


namespace blocks_needed_for_wall_l755_755149

theorem blocks_needed_for_wall (length height : ℕ) (block_heights block_lengths : List ℕ)
  (staggered : Bool) (even_ends : Bool)
  (h_length : length = 120)
  (h_height : height = 8)
  (h_block_heights : block_heights = [1])
  (h_block_lengths : block_lengths = [1, 2, 3])
  (h_staggered : staggered = true)
  (h_even_ends : even_ends = true) :
  ∃ (n : ℕ), n = 404 := 
sorry

end blocks_needed_for_wall_l755_755149


namespace complex_conjugate_power_l755_755695

noncomputable def imaginary_unit : ℂ :=
  complex.I -- Definition of the imaginary unit

theorem complex_conjugate_power (i : ℂ) (h : i = imaginary_unit) : 
  ((1 - i) ^ 2016 + (1 + i) ^ 2016 = 2 ^ 1009) := by
  sorry

end complex_conjugate_power_l755_755695


namespace number_of_students_l755_755041

theorem number_of_students 
  (n : ℕ) 
  (avg_decrease : 6)
  (total_weight_decrease : 60) 
  (weight_replaced : 120)
  (weight_new : 60) 
  (h : total_weight_decrease = weight_replaced - weight_new) 
  (h_decrease : total_weight_decrease = avg_decrease * n) : 
  n = 10 := 
by 
  sorry

end number_of_students_l755_755041


namespace isabella_final_hair_length_l755_755876

def initial_length : ℝ := 18
def extension_percentage : ℝ := 0.75
def cut_length_cm : ℝ := 6
def cm_to_inch : ℝ := 2.54

def length_after_extensions := initial_length * (1 + extension_percentage)
def cut_length_in := cut_length_cm / cm_to_inch
def final_length := length_after_extensions - cut_length_in

theorem isabella_final_hair_length : abs (final_length - 29.1378) < 0.0001 :=
by sorry

end isabella_final_hair_length_l755_755876


namespace original_number_of_workers_l755_755129

-- Definitions of the conditions given in the problem
def workers_days (W : ℕ) : ℕ := 35
def additional_workers : ℕ := 10
def reduced_days : ℕ := 10

-- The main theorem we need to prove
theorem original_number_of_workers (W : ℕ) (A : ℕ) 
  (h1 : W * workers_days W = (W + additional_workers) * (workers_days W - reduced_days)) :
  W = 25 :=
by
  sorry

end original_number_of_workers_l755_755129


namespace percent_spent_on_other_items_l755_755528

def total_amount_spent (T : ℝ) : ℝ := T
def clothing_percent (p : ℝ) : Prop := p = 0.45
def food_percent (p : ℝ) : Prop := p = 0.45
def clothing_tax (t : ℝ) (T : ℝ) : ℝ := 0.05 * (0.45 * T)
def food_tax (t : ℝ) (T : ℝ) : ℝ := 0.0 * (0.45 * T)
def other_items_tax (p : ℝ) (T : ℝ) : ℝ := 0.10 * (p * T)
def total_tax (T : ℝ) (tax : ℝ) : Prop := tax = 0.0325 * T

theorem percent_spent_on_other_items (T : ℝ) (p_clothing p_food x : ℝ) (tax : ℝ) 
  (h1 : clothing_percent p_clothing) (h2 : food_percent p_food)
  (h3 : clothing_tax tax T = 0.05 * (0.45 * T))
  (h4 : food_tax tax T = 0.0)
  (h5 : other_items_tax x T = 0.10 * (x * T))
  (h6 : total_tax T (clothing_tax tax T + food_tax tax T + other_items_tax x T)) : 
  x = 0.10 :=
by
  sorry

end percent_spent_on_other_items_l755_755528


namespace team_A_won_first_prize_l755_755721

theorem team_A_won_first_prize :
  ∃ (A B C : Prop),
    -- Conditions about who won
    (A ∨ B ∨ C) ∧ 
    ¬(A ∧ B) ∧ ¬(A ∧ C) ∧ ¬(B ∧ C) ∧ 
    -- Statements by representatives
    (C ↔ ¬A) ∧ 
    (B ↔ B) ∧ 
    (A ↔ (C ↔ ¬A)) ∧ 
    -- Condition that only one statement is false
    (¬((C ↔ ¬A) ∧ (B ↔ B) ∧ (A ↔ (C ↔ ¬A))) ∧
     (¬(C ↔ ¬A) ∧ (B ↔ B) ∧ (A ↔ (C ↔ ¬A))) ∧ 
     ((C ↔ ¬A) ∧ ¬(B ↔ B) ∧ (A ↔ (C ↔ ¬A))) ∧ 
     ((C ↔ ¬A) ∧ (B ↔ B) ∧ ¬(A ↔ (C ↔ ¬A)))) ->
    A :=
begin
  -- sorry keyword to indicate the proof is omitted
  sorry
end

end team_A_won_first_prize_l755_755721


namespace jake_peaches_is_seven_l755_755495

-- Definitions based on conditions
def steven_peaches : ℕ := 13
def jake_peaches (steven : ℕ) : ℕ := steven - 6

-- The theorem we want to prove
theorem jake_peaches_is_seven : jake_peaches steven_peaches = 7 := sorry

end jake_peaches_is_seven_l755_755495


namespace flour_per_large_tart_flour_required_for_each_large_tart_l755_755469

noncomputable def total_flour_used : ℚ := 36 * (1/12)

noncomputable def amount_of_flour_per_large_tart : ℚ := 3 / 18

theorem flour_per_large_tart :
  (total_flour_used = 3) →
  (amount_of_flour_per_large_tart = 1/6) :=
begin
  assume h : total_flour_used = 3,
  calc amount_of_flour_per_large_tart
      = 3 / 18       : rfl
  ... = 1 / 6       : by norm_num,
end

-- Theorem
theorem flour_required_for_each_large_tart :
  total_flour_used = 3 → amount_of_flour_per_large_tart = 1 / 6 :=
begin
  assume h : total_flour_used = 3,
  exact flour_per_large_tart h,
end

#eval flour_required_for_each_large_tart

end flour_per_large_tart_flour_required_for_each_large_tart_l755_755469


namespace recurring_decimal_to_fraction_l755_755313

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l755_755313


namespace enrolled_percentage_l755_755210

theorem enrolled_percentage (total_students : ℝ) (non_bio_students : ℝ)
    (h_total : total_students = 880)
    (h_non_bio : non_bio_students = 440.00000000000006) : 
    ((total_students - non_bio_students) / total_students) * 100 = 50 := 
by
  rw [h_total, h_non_bio]
  norm_num
  sorry

end enrolled_percentage_l755_755210


namespace solution_inequality_l755_755379

theorem solution_inequality (x : ℝ) : (3 - x) / (2 * x - 4) < 1 ↔ x ∈ set.Ioi (2) :=
by
  sorry

end solution_inequality_l755_755379


namespace golden_ratio_eqn_value_of_ab_value_of_pq_n_l755_755577

-- Part (1): Finding the golden ratio
theorem golden_ratio_eqn {x : ℝ} (h1 : x^2 + x - 1 = 0) : x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

-- Part (2): Finding the value of ab
theorem value_of_ab {a b m : ℝ} (h1 : a^2 + m * a = 1) (h2 : b^2 - 2 * m * b = 4) (h3 : b ≠ -2 * a) : a * b = 2 :=
sorry

-- Part (3): Finding the value of pq - n
theorem value_of_pq_n {p q n : ℝ} (h1 : p ≠ q) (eq1 : p^2 + n * p - 1 = q) (eq2 : q^2 + n * q - 1 = p) : p * q - n = 0 :=
sorry

end golden_ratio_eqn_value_of_ab_value_of_pq_n_l755_755577


namespace ways_to_distribute_problems_l755_755197

-- Definition of factorial
def fac : ℕ → ℕ
| 0       := 1
| (n+1) := (n+1) * fac n

-- Definition of binomial coefficient (n choose k)
def choose (n k : ℕ) : ℕ :=
  nat.div (fac n) ((fac k) * (fac (n - k)))

-- The main theorem statement
theorem ways_to_distribute_problems :
  let total_friends := 15
  let total_problems := 7 in
  let combinations := choose total_friends total_problems in
  let permutations := fac total_problems in
  combinations * permutations = 324324000 :=
by
  let total_friends := 15
  let total_problems := 7
  let combinations := choose total_friends total_problems
  let permutations := fac total_problems
  have : combinations * permutations = 324324000 := sorry
  exact this

end ways_to_distribute_problems_l755_755197


namespace graph_symmetry_about_point_l755_755564

def g (x : ℝ) := x^3 + Real.sin x

theorem graph_symmetry_about_point (f : ℝ → ℝ) (h : f = λ x, g x + 1) :
  (∀ x : ℝ, f (-x) - 1 = - (f x - 1)) ↔ (f = λ x, g x + 1) := by
  sorry

end graph_symmetry_about_point_l755_755564


namespace households_using_all_three_brands_correct_l755_755654

noncomputable def total_households : ℕ := 5000
noncomputable def non_users : ℕ := 1200
noncomputable def only_X : ℕ := 800
noncomputable def only_Y : ℕ := 600
noncomputable def only_Z : ℕ := 300

-- Let A be the number of households that used all three brands of soap
variable (A : ℕ)

-- For every household that used all three brands, 5 used only two brands and 10 used just one brand.
-- Number of households that used only two brands = 5 * A
-- Number of households that used only one brand = 10 * A

-- The equation for households that used just one brand:
def households_using_all_three_brands :=
10 * A = only_X + only_Y + only_Z

theorem households_using_all_three_brands_correct :
  (total_households - non_users = only_X + only_Y + only_Z + 5 * A + 10 * A) →
  (A = 170) := by
sorry

end households_using_all_three_brands_correct_l755_755654


namespace equilateral_triangle_l755_755750

variables {a b c A B C : ℝ} {S : ℝ}
variables {BA CA : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
axiom cond1 : a * Real.sin ((A + C) / 2) = b * Real.sin A
axiom cond2 : 2 * S = Real.sqrt 3 * (BA - CA)•(CA - BA)
axiom area_triangle : S = (1 / 2) * b * c * Real.sin A

-- Goal to prove
theorem equilateral_triangle : A = B ∧ B = C ∧ A = C :=
by
  sorry

end equilateral_triangle_l755_755750


namespace socks_headband_probability_l755_755883

/-- Keisha's basketball team uniform color probability -/
theorem socks_headband_probability :
  let socks_colors := {red, blue}
  let headband_colors := {red, blue, green}
  let total_combinations := (socks_colors.card * headband_colors.card)
  let matching_combinations := 2  -- (red with red, blue with blue)
  let non_matching_combinations := total_combinations - matching_combinations
  let probability := non_matching_combinations.to_rat / total_combinations.to_rat
  probability = (2 / 3) :=
by
  sorry

end socks_headband_probability_l755_755883


namespace angle_between_vectors_is_pi_div_3_l755_755756

open Real

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def norm_a : ‖a‖ = 2 := sorry
def norm_b : ‖b‖ = 2 := sorry
def dot_product_condition : (a + 2 • b) ⬝ (a - b) = -2 := sorry

-- The target theorem to be proved
theorem angle_between_vectors_is_pi_div_3
  (h1: norm_a)
  (h2: norm_b)
  (h3: dot_product_condition) :
  ∠ a b = π / 3 :=
sorry

end angle_between_vectors_is_pi_div_3_l755_755756


namespace hyperbola_eccentricity_l755_755958

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h2 : ∀ c : ℝ, c - a^2 / c = 2 * a) :
  e = 1 + Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l755_755958


namespace at_least_six_consecutive_heads_l755_755168

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end at_least_six_consecutive_heads_l755_755168


namespace rationalize_denominator_l755_755919

theorem rationalize_denominator :
  (7 / (Real.sqrt 175 - Real.sqrt 75)) = (7 * (Real.sqrt 7 + Real.sqrt 3) / 20) :=
by
  have h1 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 75 = 5 * Real.sqrt 3 := sorry
  sorry

end rationalize_denominator_l755_755919


namespace exercise_l755_755109

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l755_755109


namespace cost_of_article_l755_755450

-- Definitions for the given conditions
def sellingPriceUSD : ℝ := 780.50
def sellingPriceEUR : ℝ := 680.75
def exchangeRate : ℝ := 1 / 0.85 -- Convert EUR to USD
def gainPercent : ℝ := 0.15

-- Define the problem statement in Lean
theorem cost_of_article :
  ∃ (C G : ℝ),
  let S_EUR_to_USD := sellingPriceEUR * exchangeRate in
  (C + G = S_EUR_to_USD) ∧
  (C + (1 + gainPercent) * G = sellingPriceUSD) ∧
  (C = 936.75) :=
by
  sorry

end cost_of_article_l755_755450


namespace kitchen_cost_l755_755709

theorem kitchen_cost :
  ∀ 
  (total_sq_ft : ℕ) (kitchen_sq_ft : ℕ) (bathroom_sq_ft : ℕ) (num_bathrooms : ℕ) 
  (other_cost_per_sq_ft : ℕ) (cost_per_bathroom : ℕ) (total_cost : ℕ),
  total_sq_ft = 2000 → kitchen_sq_ft = 400 → bathroom_sq_ft = 150 → num_bathrooms = 2 
  → other_cost_per_sq_ft = 100 → cost_per_bathroom = 12000 → total_cost = 174000 
  → let other_sq_ft := total_sq_ft - kitchen_sq_ft - (num_bathrooms * bathroom_sq_ft) in
    let other_cost := other_sq_ft * other_cost_per_sq_ft in
    let bathroom_cost := num_bathrooms * cost_per_bathroom in
    let kitchen_cost := total_cost - other_cost - bathroom_cost in
    kitchen_cost = 20000 :=
by    
  cbv; -- simplifying the problem using call-by-value tactics
  intros;
  subst_vars; -- substituting all variable definitions into the problem
  have h1 : other_sq_ft = 1300, by sorry;
  have h2 : other_cost = 130000, by sorry;
  have h3 : bathroom_cost = 24000, by sorry;
  have h4 : kitchen_cost = 174000 - 130000 - 24000, by sorry;
  exact h4;

end kitchen_cost_l755_755709


namespace repeating_decimal_as_fraction_l755_755305

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755305


namespace range_f_l755_755413

def f (x : ℤ) : ℤ := x^2 - 1

theorem range_f :
  {f x | x ∈ ({-1, 0, 1} : set ℤ)} = {0, -1} :=
by
  sorry

end range_f_l755_755413


namespace initial_number_l755_755585

def initial_sum := 2013
def final_sum := 195
def transformation_factor := (2 : ℝ) / 3
def iterations := 7

theorem initial_number (a_1 a_2 a_3 : ℕ) (h1 : a_1 + a_2 + a_3 = initial_sum) :
  transformation_factor^iterations * initial_sum = final_sum → a_1 = 1841 := by
  sorry

end initial_number_l755_755585


namespace full_price_ticket_revenue_l755_755184

theorem full_price_ticket_revenue (f t : ℕ) (p : ℝ) 
  (h1 : f + t = 160) 
  (h2 : f * p + t * (p / 3) = 2500) 
  (h3 : p = 30) :
  f * p = 1350 := 
by sorry

end full_price_ticket_revenue_l755_755184


namespace exists_k_eq_A_k_value_l755_755043

-- Definitions
def f (A : ℕ) : ℕ :=
  let digits := A.digits 10
  List.sum (List.mapWithIndex (λ i a_i => (2^i) * a_i) digits)

def A_seq (A : ℕ) : ℕ → ℕ
| 0 => A
| (n + 1) => f (A_seq A n)

-- Statements to prove
theorem exists_k_eq (A : ℕ) : ∃ k : ℕ, A_seq A (k + 1) = A_seq A k :=
sorry

theorem A_k_value (A : ℕ) (h : A = 19^86) : ∃ k : ℕ, A_seq A k = 19 :=
sorry

end exists_k_eq_A_k_value_l755_755043


namespace function_is_monotonically_decreasing_l755_755813

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 3)

theorem function_is_monotonically_decreasing :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → deriv f x ≤ 0 :=
by
  sorry

end function_is_monotonically_decreasing_l755_755813


namespace find_lambda_l755_755508

-- Definitions of the vector components using Lean types
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (i j : V) (A B C D : V)
variables (λ : ℝ)

-- Conditions
def non_collinear_vectors : Prop := ¬(∃ c : ℝ, i = c • j)
def AB_eq : Prop := A - B = 3 • i + 2 • j
def CB_eq : Prop := C - B = i + λ • j
def CD_eq : Prop := C - D = -2 • i + j
def collinear_points : Prop := ∃ μ : ℝ, A - B = μ • (C - B + (D - C))

-- The theorem we want to prove
theorem find_lambda
  (h1 : non_collinear_vectors i j)
  (h2 : AB_eq i j A B)
  (h3 : CB_eq i j C B λ)
  (h4 : CD_eq i j C D)
  (h5 : collinear_points i j A B D) :
  λ = 3 :=
sorry

end find_lambda_l755_755508


namespace quadratic_roots_l755_755074

theorem quadratic_roots (k : ℝ) :
  (∃ x : ℝ, x = 2 ∧ 4 * x ^ 2 - k * x + 6 = 0) →
  k = 11 ∧ (∃ x : ℝ, x ≠ 2 ∧ 4 * x ^ 2 - 11 * x + 6 = 0 ∧ x = 3 / 4) := 
by
  sorry

end quadratic_roots_l755_755074


namespace number_of_students_l755_755040

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : (T - 100) / (N - 5) = 90) : N = 35 := 
by 
  sorry

end number_of_students_l755_755040


namespace intersecting_diagonals_in_golden_ratio_area_calculation_equal_area_pentagon_l755_755513

open Real

theorem intersecting_diagonals_in_golden_ratio 
  (ABCDE : ConvexPentagon)
  (h1 : area ABC = area BOD)
  (h2 : area BOD = area ODE)
  (h3 : area ODE = area DEA)
  (h4 : area DEA = area EAB) :
  ∀ d1 d2 : Diagonal,
  divides_in_golden_ratio (intersection d1 d2) :=
sorry

theorem area_calculation
  (ABCDE : ConvexPentagon)
  (h1 : area ABC = 1)
  (h2 : area BCD = 1)
  (h3 : area ODE = 1)
  (h4 : area DEA = 1)
  (h5 : area EAB = 1) :
  area ABCDE = (5 + sqrt 5) / 2 ∧
  area ACEBD = 3 * sqrt 5 - 5 ∧
  area FGHIJ = 5 - 2 * sqrt 5 :=
sorry

theorem equal_area_pentagon
  (ABCDE : ConvexPentagon)
  (h1 : area ABC = area BOD)
  (h2 : area BOD = area ODE)
  (h3 : area ODE = area DEA)
  (h4 : area DEA = area EAB)
  (h5 : area EAB = 1) :
  area ABC = 1 ∧
  area BOD = 1 ∧
  area ODE = 1 ∧
  area DEA = 1 ∧
  area EAB = 1 :=
sorry

end intersecting_diagonals_in_golden_ratio_area_calculation_equal_area_pentagon_l755_755513


namespace range_of_k_l755_755020

variable {x₁ x₂ k : ℝ}
variable (y₁ y₂ : ℝ)
variable (f : ℝ → ℝ := λ x, (4 - k) / x)

theorem range_of_k (h1 : x₁ < 0) (h2 : x₂ > 0) (h3 : y₁ = f x₁) (h4 : y₂ = f x₂) (h5 : y₁ < y₂) : k < 4 := 
  sorry

end range_of_k_l755_755020


namespace original_square_perimeter_l755_755671

theorem original_square_perimeter (x : ℕ) 
  (side_length := 5 * x)
  (letter_P_perimeter := 14 * x)
  (h : letter_P_perimeter = 56) :
  4 * side_length = 80 :=
by
  have x_val : x = 4, from sorry,
  have side_length_val : side_length = 20, from sorry,
  have perimeter_calculation : 4 * side_length = 80, from sorry,
  exact perimeter_calculation

end original_square_perimeter_l755_755671


namespace integer_root_of_polynomial_l755_755966

theorem integer_root_of_polynomial (b c : ℚ) :
  (5 - Real.sqrt 11) ∈ (polynomial.roots ((polynomial.C (-1 : ℤ)) * polynomial.C (5 - (Real.sqrt 11))) + (polynomial.C b + polynomial.C c)) →
  ∃ r : ℤ, r = -10 :=
by
  sorry

end integer_root_of_polynomial_l755_755966


namespace reading_club_coordinator_selection_l755_755211

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem reading_club_coordinator_selection :
  let total_ways := choose 18 4
  let no_former_ways := choose 10 4
  total_ways - no_former_ways = 2850 := by
  sorry

end reading_club_coordinator_selection_l755_755211


namespace digit_in_thousandths_place_of_7_div_54_l755_755993

theorem digit_in_thousandths_place_of_7_div_54 : 
  let decimal_repr := (7 : ℝ) / 54
  in (⇑(decimal_repr * 10^3).floor % 10) = 9 := 
by
  -- Proof is omitted
  sorry

end digit_in_thousandths_place_of_7_div_54_l755_755993


namespace value_of_m_l755_755836

theorem value_of_m (m : ℤ) (h : m + 1 = - (-2)) : m = 1 :=
sorry

end value_of_m_l755_755836


namespace cos_transform_equiv_l755_755596

-- Define the functions involved
def f (x : ℝ) : ℝ := Real.sin (2 * x)
def g (x : ℝ) : ℝ := Real.cos (2 * x - π / 3)

-- Define the transformation
def shifted_f (x : ℝ) : ℝ := f (x + π / 12)

-- Now state the theorem
theorem cos_transform_equiv :
  ∀ x : ℝ, g x = shifted_f x :=
  by sorry

end cos_transform_equiv_l755_755596


namespace max_constant_N_l755_755369

theorem max_constant_N (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0):
  (c^2 + d^2) ≠ 0 → ∃ N, N = 1 ∧ (a^2 + b^2) / (c^2 + d^2) ≤ 1 :=
by
  sorry

end max_constant_N_l755_755369


namespace probability_of_neither_red_nor_purple_l755_755644

theorem probability_of_neither_red_nor_purple :
  let total_balls := 100
  let white_balls := 20
  let green_balls := 30
  let yellow_balls := 10
  let red_balls := 37
  let purple_balls := 3
  let neither_red_nor_purple_balls := white_balls + green_balls + yellow_balls
  (neither_red_nor_purple_balls : ℝ) / (total_balls : ℝ) = 0.6 :=
by
  sorry

end probability_of_neither_red_nor_purple_l755_755644


namespace students_not_in_same_column_or_row_l755_755457

-- Define the positions of student A and student B as conditions
structure Position where
  row : Nat
  col : Nat

-- Student A's position is in the 3rd row and 6th column
def StudentA : Position := {row := 3, col := 6}

-- Student B's position is described in a relative manner in terms of columns and rows
def StudentB : Position := {row := 6, col := 3}

-- Formalize the proof statement
theorem students_not_in_same_column_or_row :
  StudentA.row ≠ StudentB.row ∧ StudentA.col ≠ StudentB.col :=
by {
  sorry
}

end students_not_in_same_column_or_row_l755_755457


namespace solution_of_r_and_s_l755_755003

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l755_755003


namespace sin_value_l755_755405

variables {α : ℝ}
hypothesis (h1 : cos α = 3/5)
hypothesis (h2 : tan α < 0)

theorem sin_value : sin α = -4/5 :=
  by
  sorry

end sin_value_l755_755405


namespace triangle_is_isosceles_right_l755_755840

theorem triangle_is_isosceles_right 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : sin B / b = cos A / a)
  (h2 : sin B / b = cos C / c) 
  (h3 : cos A / a = cos C / c) :
  A = 45 ∧ B = 90 ∧ C = 45 :=
sorry

end triangle_is_isosceles_right_l755_755840


namespace increasing_matrices_count_l755_755438

noncomputable def count_increasing_matrices : ℕ :=
  finset.univ.filter (λ M : matrix (fin 4) (fin 4) (fin 16), 
    (∀ i : fin 4, strict_anti (λ j : fin 4, M i j)) ∧
    (∀ j : fin 4, strict_anti (λ i : fin 4, M i j))).card

theorem increasing_matrices_count : count_increasing_matrices = 12 :=
sorry

end increasing_matrices_count_l755_755438


namespace domain_of_f_l755_755558

open Real

noncomputable def f (x : ℝ) : ℝ := ln (sin x) + sqrt (49 - x^2)

theorem domain_of_f :
  {x : ℝ | sin x > 0 ∧ 49 - x^2 ≥ 0} = {x | -2 * π < x ∧ x < -π} ∪ {x | 0 < x ∧ x < π} ∪ {x | 2 * π < x ∧ x ≤ 7} :=
by
  sorry

end domain_of_f_l755_755558


namespace eval_expr_at_values_l755_755032

variable (x y : ℝ)

def expr := 2 * (3 * x^2 + x * y^2)- 3 * (2 * x * y^2 - x^2) - 10 * x^2

theorem eval_expr_at_values : x = -1 → y = 0.5 → expr x y = 0 :=
by
  intros hx hy
  rw [hx, hy]
  sorry

end eval_expr_at_values_l755_755032


namespace sequence_identity_l755_755409

theorem sequence_identity
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_eq : ∀ n, ∑ k in Finset.range (n + 1), (a k) ^ 3 = (∑ k in Finset.range (n + 1), a k) ^ 2) :
  ∀ n, a n = n :=
by
  sorry

end sequence_identity_l755_755409


namespace four_thirds_of_twelve_fifths_l755_755254

theorem four_thirds_of_twelve_fifths : (4 / 3) * (12 / 5) = 16 / 5 := 
by sorry

end four_thirds_of_twelve_fifths_l755_755254


namespace total_gas_cost_l755_755942

def gas_price_station_1 : ℝ := 3
def gas_price_station_2 : ℝ := 3.5
def gas_price_station_3 : ℝ := 4
def gas_price_station_4 : ℝ := 4.5
def tank_capacity : ℝ := 12

theorem total_gas_cost :
  let cost_station_1 := tank_capacity * gas_price_station_1
  let cost_station_2 := tank_capacity * gas_price_station_2
  let cost_station_3 := tank_capacity * gas_price_station_3
  let cost_station_4 := tank_capacity * gas_price_station_4
  cost_station_1 + cost_station_2 + cost_station_3 + cost_station_4 = 180 :=
by
  -- Proof is skipped
  sorry

end total_gas_cost_l755_755942


namespace min_value_n_l755_755992

-- Definitions based on the conditions
def three_periodic : ℚ := 1 / 3

noncomputable def a_n (n : ℕ) : ℚ :=
3 * (1 - (10:ℚ)^(-n)) / 9

-- The mathematical problem rewritten as a Lean 4 statement
theorem min_value_n (n : ℕ) (h : |a_n n - three_periodic| < 1 / 2015) : n ≥ 4 :=
sorry

end min_value_n_l755_755992


namespace proposition_one_true_proposition_two_false_proposition_three_false_proposition_four_true_true_propositions_l755_755387

-- Given m and n represent different lines, and α, β, γ represent different planes.
variables (m n : Line) (α β γ : Plane)

-- Condition and Question 1
theorem proposition_one_true (h1 : α ∩ β = m) (h2 : n ⊆ α) (h3 : n ⊥ m) : α ⊥ β := sorry

-- Condition and Question 2
theorem proposition_two_false (h1 : α ⊥ β) (h2 : α ∩ γ = m) (h3 : β ∩ γ = n) : ¬(n ⊥ m) := sorry

-- Condition and Question 3
theorem proposition_three_false (h1 : α ⊥ β) (h2 : α ⊥ γ = m) (h3 : β ∩ γ = m) : ¬(m ⊥ α) := sorry

-- Condition and Question 4
theorem proposition_four_true (h1 : m ⊥ α) (h2 : n ⊥ β) (h3 : m ⊥ n) : α ⊥ β := sorry

-- Final Statement: Among the above four propositions, the true propositions are 1 and 4.
theorem true_propositions : 
  (proposition_one_true m n α β γ h1 h2 h3) ∧ (proposition_four_true m n α β γ h1 h2 h3) :=
  sorry

end proposition_one_true_proposition_two_false_proposition_three_false_proposition_four_true_true_propositions_l755_755387


namespace solve_for_y_l755_755935

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l755_755935


namespace constant_in_price_equation_l755_755073

theorem constant_in_price_equation (x y: ℕ) (h: y = 70 * x) : ∃ c, ∀ (x: ℕ), y = c * x ∧ c = 70 :=
  sorry

end constant_in_price_equation_l755_755073


namespace max_trig_cos_combination_l755_755400

theorem max_trig_cos_combination (a b c : ℝ) (x y z : ℝ) 
    (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
    (triangle_angles : x + y + z = π):
    (\max (λ (x y z : ℝ), a * Real.cos x + b * Real.cos y + c * Real.cos z) x y z) =
    (1 / 2) * ((a * b / c) + (a * c / b) + (b * c / a)) :=
sorry

end max_trig_cos_combination_l755_755400


namespace repeating_decimal_eq_fraction_l755_755336

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755336


namespace geometry_problem_l755_755485

theorem geometry_problem (BAC ABC ACB ADE CDE AED DEB : ℝ)
  (h1 : BAC = 180 - ABC - ACB)
  (h2 : ABC = 55)
  (h3 : ACB = 90)
  (h4 : ADE = 180 - CDE)
  (h5 : CDE = 50) :
  DEB = 165 :=
by 
  -- Initial triangle angles
  have hBAC : BAC = 35, by linarith,
  -- Angle ADE
  have hADE : ADE = 130, by linarith,
  -- Triangle AED angles
  have hAED : AED = 15, by linarith,
  -- Straight line angle DEB
  have hDEB : DEB = 180 - AED, by linarith,
  linarith,

end geometry_problem_l755_755485


namespace perfect_square_condition_least_k_solution_l755_755907

theorem perfect_square_condition (k : ℕ) (h : k ≥ 3) :
  (∀ p, (∃ steps, final_heap (1..k).toList steps = p) → (∃ n, p = n^2)) ↔ 
  (∃ x y, 2 * k + 2 = x^2 ∧ 3 * k + 1 = y^2) :=
by sorry

-- Finding the least k that satisfies both conditions
noncomputable def least_k : ℕ :=
  if h : ∃ k, k ≥ 3 ∧ (∃ x y, 2 * k + 2 = x^2 ∧ 3 * k + 1 = y^2)
  then Nat.find h
  else 0

theorem least_k_solution : least_k = 161 :=
by sorry

end perfect_square_condition_least_k_solution_l755_755907


namespace cos_a_eq_neg_quarter_l755_755852

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Conditions
axiom sin_eq : sin B - sin C = 1/4 * sin A
axiom ratio_b_c : 2 * b = 3 * c

theorem cos_a_eq_neg_quarter (sin_eq : sin B - sin C = 1/4 * sin A) (ratio_b_c : 2 * b = 3*c) :
  cos A = -1/4 :=
sorry

end cos_a_eq_neg_quarter_l755_755852


namespace solution_C_l755_755202

-- Define the conditions for each pair of numbers
def pair1 := (-(-2), -|-2|)
def pair2 := ((-1)^2, -1^2)
def pair3 := ((-2)^3, 6)
def pair4 := ((-2)^7, -2^7)

-- Function to check if the numbers in a pair are opposites
def are_opposites (a b : Int) : Prop := a = -b

-- Prove that the correct pairs are opposite numbers
theorem solution_C : (are_opposites (fst pair1) (snd pair1)) ∧ 
                     (are_opposites (fst pair2) (snd pair2)) ∧ 
                     ¬ (are_opposites (fst pair3) (snd pair3)) ∧ 
                     ¬ (are_opposites (fst pair4) (snd pair4)) := 
by
  sorry

end solution_C_l755_755202


namespace max_min_values_a_neg1_monotonic_interval_max_value_g_l755_755423

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 2

-- Statement 1: Find min and max values when a = -1
theorem max_min_values_a_neg1 : 
  let a := -1 in 
  (∀ x ∈ [-5, 5], f x a ≥ 1) ∧ (∀ x ∈ [-5, 5], f x a ≤ 37) := sorry

-- Statement 2: Determine the range of a such that y = f(x) is monotonic on [-5, 5]
theorem monotonic_interval :
  ∀ a : ℝ, (∀ x y ∈ [-5, 5], x < y → f x a ≤ f y a ∨ f x a ≥ f y a) ↔ a ∈ [5, +∞) ∪ (-∞, -5] := sorry

-- Define g(a) which is the minimum value of f(x)
def g (a : ℝ) := if a ∈ [-5, 5] then 2 - a^2 else if a < -5 then 27 + 10 * a else 27 - 10 * a

-- Statement 3: Find the maximum value of g(a)
theorem max_value_g : 
  ∀ a : ℝ, g(a) ≤ 2 ∧ (∃ a, g(a) = 2) := sorry

end max_min_values_a_neg1_monotonic_interval_max_value_g_l755_755423


namespace fraction_eq_repeating_decimal_l755_755359

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755359


namespace reflect_across_x_axis_l755_755867

theorem reflect_across_x_axis (x y : ℝ) : 
  let P := (x, y)
  let Q := (x, -y)
  in P = (-2, 3) → Q = (-2, -3) :=
by
  intro h
  have : P = (-2, 3) := h
  rw [this]
  unfold Q
  simp
  sorry

end reflect_across_x_axis_l755_755867


namespace greatest_common_multiple_of_9_and_15_less_than_120_l755_755999

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end greatest_common_multiple_of_9_and_15_less_than_120_l755_755999


namespace distance_from_origin_to_line_l755_755044

noncomputable def distance_to_line (a b c x0 y0 : ℝ) : ℝ := 
  abs (a * x0 + b * y0 + c) / (real.sqrt (a^2 + b^2))

theorem distance_from_origin_to_line : distance_to_line 4 3 (-1) 0 0 = 1 / 5 := 
by sorry

end distance_from_origin_to_line_l755_755044


namespace A_symmetry_l755_755383

-- Definition of A
def A (n k r : ℕ) : ℕ := 
  @Fintype.card
    ((Σ (x : Fin k → ℕ), List.sorted (x ≫ Fin.val) ∧ List.sum (x.toList) = n ∧ x 0 - x (k-1) ≤ r))
    (by sorry)

-- Main theorem statement
theorem A_symmetry (m s t : ℕ) : A m s t = A m t s := by
  sorry

end A_symmetry_l755_755383


namespace square_difference_l755_755107

theorem square_difference :
  let a := 1001
  let b := 999
  a^2 - b^2 = 4000 :=
by
  let a := 1001
  let b := 999
  have h1 : a^2 - b^2 = (a + b) * (a - b), from sorry
  have h2 : a + b = 2000, by sorry
  have h3 : a - b = 2, by sorry
  show a^2 - b^2 = 4000, by sorry

end square_difference_l755_755107


namespace product_513_12_l755_755208

theorem product_513_12 : 513 * 12 = 6156 := 
  by
    sorry

end product_513_12_l755_755208


namespace solve_y_equation_l755_755931

noncomputable def solve_y : ℚ :=
  let y := (500 * 1 : ℚ) / 15 in
  y

theorem solve_y_equation (y : ℚ) :
  2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = solve_y := by
  intro h
  sorry

end solve_y_equation_l755_755931


namespace binomial_coefficient_term_x8_expansion_l755_755255

theorem binomial_coefficient_term_x8_expansion : 
  binomial_coeff (λ n => (x^3 + (1 / (2 * sqrt x)))^n) 5 8 = 10 := 
sorry

end binomial_coefficient_term_x8_expansion_l755_755255


namespace series_sum_expression_l755_755249

theorem series_sum_expression :
  3 * (∑ n in Finset.range 14, (1 : ℚ) / (n + 1) / (n + 2)) = 14 / 5 :=
by sorry

end series_sum_expression_l755_755249


namespace eccentricities_correct_l755_755598

-- Define the arithmetic mean
def arithmetic_mean (x y : ℝ) : ℝ := (x + y) / 2

-- Define the geometric mean
def geometric_mean (x y : ℝ) : ℝ := Real.sqrt (x * y)

-- Define the eccentricity of an ellipse
def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt (a - b) / Real.sqrt a

-- Define the eccentricity of a hyperbola
def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt a

-- Given conditions
def a := arithmetic_mean 1 9
def b := geometric_mean 1 9

-- Problem statement to prove in Lean 4
theorem eccentricities_correct : 
  a = 5 ∧ b = 3 ∧ 
  (ellipse_eccentricity a b = Real.sqrt 10 / 5) ∧ 
  (hyperbola_eccentricity a b = 2 * Real.sqrt 10 / 5) :=
sorry

end eccentricities_correct_l755_755598


namespace rhinoceros_fold_transitions_l755_755147

/--
A Rhinoceros has folds on its skin in vertical and horizontal directions.
A state is described by four integers: (a, b, c, d) where
  - a is the vertical folds on the left side,
  - b is the horizontal folds on the left side,
  - c is the vertical folds on the right side,
  - d is the horizontal folds on the right side.
Initially, the Rhineoceros has state (0, 2, 2, 1).

Given the conditions:
1. If there are at least two vertical folds on a side, two can be smoothed out, and one vertical and one horizontal fold appear on the opposite side.
2. If there are at least two horizontal folds on a side, two can be smoothed out, and one vertical and one horizontal fold appear on the opposite side.

Prove that there exists a sequence of these transitions leading to the state (2, 0, 2, 1).
-/

theorem rhinoceros_fold_transitions :
  ∃ sequence_of_transitions,
    sequence_of_transitions (0, 2, 2, 1) = (2, 0, 2, 1) :=
sorry

end rhinoceros_fold_transitions_l755_755147


namespace square_area_l755_755187

theorem square_area :
  (∀ x : ℝ, (x^2 + 4*x + 3 = 8) → ((abs(1 - (-5)))^2 = 36)) :=
begin
  -- conditions
  intro x,
  intro h,
  -- proof steps (intentionally left out)
  sorry
end

end square_area_l755_755187


namespace repeating_fraction_equality_l755_755352

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755352


namespace trajectory_of_tangent_circle_is_ellipse_l755_755022

theorem trajectory_of_tangent_circle_is_ellipse
  (center_P : Point) (r_P : ℝ) (center_M : Point) (r_Q : ℝ)
  (hP : ∃ r_P, circle center_P r_P)
  (hM_inside_P : inside_circle center_P r_P center_M ∧ center_M ≠ center_P)
  (hQ : ∃ r_Q, circle center_M r_Q ∧ tangent circle center_P r_P (circle center_M r_Q)) :
  trajectory (centerQ : Point) (circle center_M r_Q) = ellipse := 
sorry

end trajectory_of_tangent_circle_is_ellipse_l755_755022


namespace fraction_eq_repeating_decimal_l755_755277

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l755_755277


namespace scientific_notation_10200000_l755_755595

theorem scientific_notation_10200000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 10.2 * 10^7 = a * 10^n := 
sorry

end scientific_notation_10200000_l755_755595


namespace reflection_x_axis_l755_755868

theorem reflection_x_axis (x y : ℝ) (P : (ℝ × ℝ)) (h1 : P = (-3, -5)) : 
  reflectionAcrossXAxis P = (-3, 5) := 
by
  -- Insert the Lean code for the definition of reflectionAcrossXAxis if needed.
  sorry

end reflection_x_axis_l755_755868


namespace candies_bought_l755_755926

theorem candies_bought :
  ∃ (S C : ℕ), S + C = 8 ∧ 300 * S + 500 * C = 3000 ∧ C = 3 :=
by
  sorry

end candies_bought_l755_755926


namespace repeating_decimal_is_fraction_l755_755294

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l755_755294


namespace bus_empty_seats_l755_755971

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end bus_empty_seats_l755_755971


namespace part_I_monotonicity_part_II_min_t_l755_755810

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1 / 2) * a * x^2 + x
def g (x : ℝ) : ℝ := -2 * x + 3

theorem part_I_monotonicity (a : ℝ) :
  (∀ x : ℝ, x > 0 → a ≤ 0 → (Real.log x - (1 / 2) * a * x^2 + (1 - a) * x + (3 / 2) * a)) → (∀ x : ℝ, x > 0 → a > 0 → x < 1 / a → 
    (Real.log x - (1 / 2) * a * x^2 + (1 - a) * x + (3 / 2) * a) ∧ ∀ x : ℝ, x > 0 → a > 0 → x > 1 / a → 
    ¬(Real.log x - (1 / 2) * a * x^2 + (1 - a) * x + (3 / 2) * a)) :=
sorry

theorem part_II_min_t (a : ℝ) (x1 x2 : ℝ) :
  (a ≥ -2 ∧ a ≤ -1) ∧ (1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2) →
  |Real.log x1 - (1 / 2) * a * x1^2 + x1 - (Real.log x2 - (1 / 2) * a * x2^2 + x2)|
  ≤ (11 / 4) * |-2 * x1 + 3 - (-2 * x2 + 3)| :=
sorry

end part_I_monotonicity_part_II_min_t_l755_755810


namespace nonnegative_integers_count_l755_755443

theorem nonnegative_integers_count : 
  ∃ (c : ℕ), c = 3281 ∧ ∀ n ∈ {k | ∃ (a : fin 8 → ℤ), ∀ i, a i ∈ {-1, 0, 1} ∧ set.sum (finset.univ) (λ i, a i * (3 ^ (i : ℕ))) = k}, 
  0 ≤ n → n < c :=
sorry

end nonnegative_integers_count_l755_755443


namespace odd_divisors_implies_perfect_square_l755_755024

theorem odd_divisors_implies_perfect_square (N : ℕ) (h : Nat.divisors N % 2 = 1) : ∃ d : ℕ, d * d = N := 
sorry

end odd_divisors_implies_perfect_square_l755_755024


namespace inequality_1_l755_755035

theorem inequality_1 (x : ℝ) : (x - 2) * (1 - 3 * x) > 2 → 1 < x ∧ x < 4 / 3 :=
by sorry

end inequality_1_l755_755035


namespace prove_P_A1_prove_P_B_prove_P_A1_given_B_l755_755980

-- Define constants for defect rates
def defect_rate_lathe1 : ℝ := 0.06
def defect_rate_lathe2 : ℝ := 0.05
def defect_rate_lathe3 : ℝ := 0.04

-- Define constants for the probability ratios
def parts_ratio_lathe1 : ℝ := 5
def parts_ratio_lathe2 : ℝ := 6
def parts_ratio_lathe3 : ℝ := 9

-- Calculate total parts ratio
def total_parts_ratio : ℝ := parts_ratio_lathe1 + parts_ratio_lathe2 + parts_ratio_lathe3

-- Calculate the probability of a part being processed by each lathe
def P_A1 : ℝ := parts_ratio_lathe1 / total_parts_ratio
def P_A2 : ℝ := parts_ratio_lathe2 / total_parts_ratio
def P_A3 : ℝ := parts_ratio_lathe3 / total_parts_ratio

-- Calculate the overall probability of selecting a defective part (P(B))
def P_B : ℝ := P_A1 * defect_rate_lathe1 + P_A2 * defect_rate_lathe2 + P_A3 * defect_rate_lathe3

-- Calculate the conditional probability P(A1|B) using Bayes' theorem
def P_A1_given_B : ℝ := (defect_rate_lathe1 * P_A1) / P_B

-- Prove the required statements

-- Prove that P(A1) = 0.25
theorem prove_P_A1 : P_A1 = 0.25 := by
  sorry

-- Prove that P(B) = 0.048
theorem prove_P_B : P_B = 0.048 := by
  sorry

-- Prove that P(A1|B) = 5 / 16
theorem prove_P_A1_given_B : P_A1_given_B = 5 / 16 := by
  sorry

end prove_P_A1_prove_P_B_prove_P_A1_given_B_l755_755980


namespace second_player_wins_l755_755531

/-- On the board the numbers 25 and 36 are written. In a move, it is allowed to add another natural 
number – the difference of any two numbers already on the board, as long as it hasn’t appeared 
previously. The player who cannot make a move loses. Prove that the second player wins this game. -/
theorem second_player_wins :
  let board : ℕ → Prop := λ n, n = 25 ∨ n = 36 ∨ ∃ a b, board a ∧ board b ∧ n = abs (a - b)
  let player1_losses := ∃ n, ¬board n → false
  ∀ first_move : ℕ, ¬board first_move →
    ∀ board_after_moves : ℕ → Prop, board_after_moves = board →
    ¬player1_losses →
    player1_losses ∧ ¬player1_losses :=
sorry

end second_player_wins_l755_755531


namespace problem_1_problem_2_l755_755395

noncomputable def a : ℕ → ℕ
| 0       := 2
| (n + 1) := 3 * a n + 2

def b (n : ℕ) : ℕ := n * a n

def T (n : ℕ) : ℕ := 
  (n / 2 - 1 / 4) * 3^(n + 1) + 3 / 4 - n * (n + 1) / 2

theorem problem_1 : ∃ r : ℕ, ∀ n : ℕ, (a (n + 1) + 1) = r * (a n + 1) :=
  sorry

theorem problem_2 : ∀ n : ℕ, (∑ i in Finset.range n, b i) = T n :=
  sorry

end problem_1_problem_2_l755_755395


namespace repeating_decimal_as_fraction_l755_755304

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l755_755304


namespace solution_m_in_interval_l755_755419

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x < 1 then -x^2 + 2 * m * x - 2 else 1 + Real.log x

theorem solution_m_in_interval :
  ∃ m : ℝ, (1 ≤ m ∧ m ≤ 2) ∧
  (∀ x < 1, ∀ y < 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x < 1, ∀ y ≥ 1, f x m ≤ f y m) :=
by
  sorry

end solution_m_in_interval_l755_755419


namespace WP_value_l755_755864

open Classical
open Real
open EuclideanGeometry

noncomputable def WP_solution (WXYZ : square) (P : WXYZ.diagonal) (P_condition : WP > PY)
  (O3 : circumcenter_triangle WXP) (O4 : circumcenter_triangle ZYP)
  (WZ : length 10) (angle_condition : angle O3 P O4 = 90°) : ℝ × ℝ :=
  let WP := 10 * sqrt 2
  let c := 200
  let d := 200
  (c, d)

theorem WP_value (WXYZ : square) (P : WXYZ.diagonal) (P_condition : WP > PY)
  (O3 : circumcenter_triangle WXP) (O4 : circumcenter_triangle ZYP)
  (WZ : length 10) (angle_condition : angle O3 P O4 = 90°) :
  let (c, d) := WP_solution WXYZ P P_condition O3 O4 WZ angle_condition in
  WP = sqrt c + sqrt d ∧ c + d = 40 :=
by {
  sorry
}

end WP_value_l755_755864


namespace relationship_of_y_values_l755_755842

theorem relationship_of_y_values (a : ℝ) (y1 y2 y3 : ℝ) :
  (3, y1) ∈ (λ x => (a^2 + 2) / x) ∧
  (-1, y2) ∈ (λ x => (a^2 + 2) / x) ∧
  (-6, y3) ∈ (λ x => (a^2 + 2) / x) →
  y1 > y3 ∧ y3 > y2 :=
by
  intro h
  sorry

end relationship_of_y_values_l755_755842


namespace solution_set_of_inequality_l755_755582

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 4*x - 5 < 0) ↔ (-5 < x ∧ x < 1) :=
by sorry

end solution_set_of_inequality_l755_755582


namespace exists_three_integers_divisible_by_seven_l755_755025

theorem exists_three_integers_divisible_by_seven (S : Finset ℤ) (hS : S.card = 7) :
  ∃ a b c ∈ S, (a^2 + b^2 + c^2 - a * b - b * c - c * a) % 7 = 0 :=
sorry

end exists_three_integers_divisible_by_seven_l755_755025


namespace triangle_inequality_l755_755687

theorem triangle_inequality
  (A B C O A' B' C' : Type*)
  [inner_point : O ∈ triangle A B C]
  (intersection_A : A' ∈ line(A, O))
  (intersection_B : B' ∈ line(B, O))
  (intersection_C : C' ∈ line(C, O))
  (intersect_opposite_A : A' ∈ line_segment(B, C))
  (intersect_opposite_B : B' ∈ line_segment(A, C))
  (intersect_opposite_C : C' ∈ line_segment(A, B)) :
  \frac{distance(A, O)}{distance(O, A')} + \frac{distance(B, O)}{distance(O, B')} + \frac{distance(C, O)}{distance(O, C')} \geq 6 := 
sorry

end triangle_inequality_l755_755687


namespace second_machine_completion_time_l755_755171

variable (time_first_machine : ℝ) (rate_first_machine : ℝ) (rate_combined : ℝ)
variable (rate_second_machine: ℝ) (y : ℝ)

def processing_rate_first_machine := rate_first_machine = 100
def processing_rate_combined := rate_combined = 1000 / 3
def processing_rate_second_machine := rate_second_machine = rate_combined - rate_first_machine
def completion_time_second_machine := y = 1000 / rate_second_machine

theorem second_machine_completion_time
  (h1: processing_rate_first_machine rate_first_machine)
  (h2: processing_rate_combined rate_combined)
  (h3: processing_rate_second_machine rate_combined rate_first_machine rate_second_machine)
  (h4: completion_time_second_machine rate_second_machine y) :
  y = 30 / 7 :=
sorry

end second_machine_completion_time_l755_755171


namespace minimum_breaks_for_weights_l755_755594

open Nat

/-- 
  Given a chain of 150 links, each link weighing 1 gram.
  Prove that the minimal number of links that need to be broken 
  in order to form every possible weight from 1 gram to 150 grams using the resulting pieces is 4.
-/
theorem minimum_breaks_for_weights (n : ℕ) (h : n = 150) : 
  ∃ k, (k = 4) ∧ (∀ w, 1 ≤ w ∧ w ≤ 150 → 
  ∃ s, (s ⊆ finset.range (n + 1)) ∧ (finset.card s = k) ∧ 
  ∃ L, (∀ i ∈ s, L i = 1) 
  ∧ (finset.sum (s \ {0}) L = w)) :=
sorry

end minimum_breaks_for_weights_l755_755594


namespace max_axbycz_value_l755_755786

theorem max_axbycz_value (a b c : ℝ) (x y z : ℝ) 
  (h_triangle: a + b > c ∧ b + c > a ∧ c + a > b)
  (h_positive: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) :=
  sorry

end max_axbycz_value_l755_755786


namespace fraction_for_repeating_56_l755_755263

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l755_755263


namespace max_intersection_points_2010_l755_755669

noncomputable def Sphere_2010 := {x : ℝ^2011 // ∥x∥ = r}

def max_intersection_points (S : set (Sphere_2010 r)) : ℕ := 
  max (set.univ : set (fin 2 -> Sphere_2010 r))

theorem max_intersection_points_2010 {S : set (Sphere_2010 r) } :
  max_intersection_points S = 2 := 
sorry

end max_intersection_points_2010_l755_755669


namespace sum_of_all_k_l755_755103

theorem sum_of_all_k :
  (k : ℤ) → 
  (∃ (p q : ℤ), pq = 5 ∧ 2 * (p + q) = k ∧ p ≠ q) → 
  ∑ (k : ℤ), (by { sorry }) = 0 :=
by sorry

end sum_of_all_k_l755_755103


namespace shaded_fractional_part_l755_755661

theorem shaded_fractional_part :
  let initial_division := λ(a b : ℕ), a / b,
      shaded_area := initial_division 4 6 in
  ∑' n : ℕ, shaded_area / (6 ^ n) = 4 / 5 :=
by 
  sorry

end shaded_fractional_part_l755_755661


namespace simplify_tan_sum_l755_755544

theorem simplify_tan_sum : tan (π / 12) + tan (5 * π / 12) = 8 := by
  sorry

end simplify_tan_sum_l755_755544


namespace solve_for_x_l755_755034

theorem solve_for_x (x : ℝ) : 5^x + 8 = 4 * 5^x - 45 → x = Real.log 5⁻¹ (53 / 3) :=
by
  sorry

end solve_for_x_l755_755034


namespace present_value_correct_l755_755447

noncomputable def PV (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

theorem present_value_correct : 
  PV (600000 : ℝ) 0.04 15 ≈ 333088.86 := 
by {
  sorry
}

end present_value_correct_l755_755447


namespace sum_of_7_terms_arithmetic_seq_l755_755483

variable {α : Type*} [LinearOrderedField α]

def arithmetic_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_7_terms_arithmetic_seq (a : ℕ → α) (h_arith : arithmetic_seq a)
  (h_a4 : a 4 = 2) :
  (7 * (a 1 + a 7)) / 2 = 14 :=
sorry

end sum_of_7_terms_arithmetic_seq_l755_755483


namespace fraction_eq_repeating_decimal_l755_755367

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l755_755367


namespace same_color_probability_l755_755642

variables (R W B : ℕ) (totalMarbles drawnMarbles redWays whiteWays blueWays totalWays : ℕ)
def marbles_conditions (R W B : ℕ) := R = 3 ∧ W = 6 ∧ B = 8 ∧ totalMarbles = R + W + B

def probability_same_color (R W B : ℕ) : ℚ :=
  let totalWays := (17.choose 4)
  let waysSameColor := (0 + (6.choose 4) + (8.choose 4))
  waysSameColor / totalWays 

theorem same_color_probability (h : marbles_conditions R W B) :
  probability_same_color R W B = 17 / 476 := sorry

end same_color_probability_l755_755642


namespace approximation_correct_l755_755601

-- Define the function as per problem's conditions
def f (x y z : ℝ) : ℝ :=
  x^2 / (y^(1/3) * (z^2)^(1/15))

-- Define the point of approximation and values from the problem statement
def x0 : ℝ := 2.04
def y0 : ℝ := 0.97
def z0 : ℝ := 1.02

-- Define the correct answer for approximation
def approx_a : ℝ := 4.16

-- State the theorem to be proved
theorem approximation_correct :
  f x0 y0 z0 ≈ approx_a :=
sorry

end approximation_correct_l755_755601


namespace fraction_sum_l755_755519

namespace GeometricSequence

-- Given conditions in the problem
def q : ℕ := 2

-- Definition of the sum of the first n terms (S_n) of a geometric sequence
def S_n (a₁ : ℤ) (n : ℕ) : ℤ := 
  a₁ * (1 - q ^ n) / (1 - q)

-- Specific sum for the first 4 terms (S₄)
def S₄ (a₁ : ℤ) : ℤ := S_n a₁ 4

-- Define the 2nd term of the geometric sequence
def a₂ (a₁ : ℤ) : ℤ := a₁ * q

-- The statement to prove: $\dfrac{S_4}{a_2} = \dfrac{15}{2}$
theorem fraction_sum (a₁ : ℤ) : (S₄ a₁) / (a₂ a₁) = Rat.ofInt 15 / Rat.ofInt 2 :=
  by
  -- Implementation of proof will go here
  sorry

end GeometricSequence

end fraction_sum_l755_755519


namespace sum_of_reciprocals_l755_755737

-- We state that for all non-zero real numbers x and y, if x + y = xy,
-- then the sum of their reciprocals equals 1.
theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1/x + 1/y = 1 :=
by
  sorry

end sum_of_reciprocals_l755_755737


namespace repeating_decimal_equiv_fraction_l755_755280

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755280


namespace four_digit_positive_integers_l755_755826

theorem four_digit_positive_integers :
  (∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧
            (∀ i ∈ [3, 4], ∃ d ∈ [2, 5, 7], nat.digit n i = d) ∧
            (∀ i ∈ [1, 2], ∃ d ∈ [1, 4, 5, 6], nat.digit n i = d) ∧
            nat.digit n 3 ≠ nat.digit n 4)
  → card {n | 1000 ≤ n ∧ n ≤ 9999 ∧
              (∀ i ∈ [1, 2], nat.digit n i ∈ [1, 4, 5, 6]) ∧
              (∀ i ∈ [3, 4], nat.digit n i ∈ [2, 5, 7]) ∧
              nat.digit n 3 ≠ nat.digit n 4} = 96 :=
by
  sorry

end four_digit_positive_integers_l755_755826


namespace selection_properties_from_1_to_11_l755_755744

theorem selection_properties_from_1_to_11 :
  ∀ (s : Finset ℕ), s.card = 6 ∧ ↑s ⊆ Finset.range 12 →
  (∃ x y ∈ s, Nat.coprime x y) ∧
  ¬ (∀ x y ∈ s, x ≠ y → Nat.dvd x y) ∧
  (∃ x y ∈ s, y ≠ x ∧ Nat.dvd (2 * x) y) :=
by
  -- Sorry is used here to skip the proof steps.
  sorry

end selection_properties_from_1_to_11_l755_755744


namespace equation1_equation2_equation3_equation4_l755_755936

-- 1. Solve: 2(2x-1)^2 = 8
theorem equation1 (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ (x = 3/2) ∨ (x = -1/2) :=
sorry

-- 2. Solve: 2x^2 + 3x - 2 = 0
theorem equation2 (x : ℝ) : 2 * x^2 + 3 * x - 2 = 0 ↔ (x = 1/2) ∨ (x = -2) :=
sorry

-- 3. Solve: x(2x-7) = 3(2x-7)
theorem equation3 (x : ℝ) : x * (2 * x - 7) = 3 * (2 * x - 7) ↔ (x = 7/2) ∨ (x = 3) :=
sorry

-- 4. Solve: 2y^2 + 8y - 1 = 0
theorem equation4 (y : ℝ) : 2 * y^2 + 8 * y - 1 = 0 ↔ (y = (-4 + 3 * Real.sqrt 2) / 2) ∨ (y = (-4 - 3 * Real.sqrt 2) / 2) :=
sorry

end equation1_equation2_equation3_equation4_l755_755936


namespace average_age_combined_l755_755471

theorem average_age_combined (num_people_X num_people_Y : ℕ) (avg_age_X avg_age_Y : ℕ) :
  num_people_X = 8 →
  avg_age_X = 35 →
  num_people_Y = 5 →
  avg_age_Y = 30 →
  let total_age_X := num_people_X * avg_age_X,
      total_age_Y := num_people_Y * avg_age_Y,
      combined_total_age := total_age_X + total_age_Y,
      total_number_people := num_people_X + num_people_Y,
      combined_avg_age := combined_total_age / total_number_people in
  combined_avg_age = 33 :=
by
  intros h1 h2 h3 h4
  let total_age_X := 8 * 35
  let total_age_Y := 5 * 30
  let combined_total_age := total_age_X + total_age_Y
  let total_number_people := 8 + 5
  let combined_avg_age := combined_total_age / total_number_people
  have h5 : combined_avg_age = 33 := sorry
  exact h5

end average_age_combined_l755_755471


namespace diff_of_squares_l755_755104

theorem diff_of_squares : (1001^2 - 999^2 = 4000) :=
by
  sorry

end diff_of_squares_l755_755104


namespace time_representation_l755_755848

theorem time_representation :
  (noon : Int = 0) → (afternoon : Int = 2) → (morning : Int = -1) →
  (11 : Int = -1) :=
by
  intro noon afternoon morning
  sorry

end time_representation_l755_755848


namespace repeating_decimal_eq_fraction_l755_755342

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l755_755342


namespace probability_at_least_6_heads_l755_755164

theorem probability_at_least_6_heads (n : ℕ) (p : ℚ) : n = 8 ∧ p = (3 / 128) :=
  let total_outcomes := 2 ^ n in
  let successful_outcomes := 3 + 2 + 1 in
  n = 8 ∧ total_outcomes = 256 ∧ p = (successful_outcomes.to_rat / total_outcomes.to_rat) := sorry

end probability_at_least_6_heads_l755_755164


namespace max_x_plus_y_max_x_mul_y_l755_755912

theorem max_x_plus_y (a b : ℝ) (x y : ℝ) (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  ∃ x_max : ℝ, x_max = a^2 / (a^2 + b^2).sqrt ∧ x + (b / a) * (a^2 - x^2).sqrt = (a^2 + b^2).sqrt := 
sorry

theorem max_x_mul_y (a b : ℝ) (x y : ℝ) (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  ∃ x_max : ℝ, x_max = a * real.sqrt 2 / 2 ∧ x * ((b / a) * (a^2 - x^2).sqrt) = ab / 2 := 
sorry

end max_x_plus_y_max_x_mul_y_l755_755912


namespace largest_value_l755_755719

def X := (2010 / 2009) + (2010 / 2011)
def Y := (2010 / 2011) + (2012 / 2011)
def Z := (2011 / 2010) + (2011 / 2012)

theorem largest_value : X > Y ∧ X > Z := 
by
  sorry

end largest_value_l755_755719


namespace repeating_fraction_equality_l755_755351

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l755_755351


namespace repeating_decimal_equiv_fraction_l755_755286

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l755_755286


namespace math_problem_l755_755751

-- Definitions based on conditions from part a)
def a_condition (a : ℝ) : Prop := a + a⁻¹ = 5 / 2 ∧ a > 1
def equation_1 (x y : ℝ) : Prop := 2 * log 10 (x - 2 * y) = log 10 x + log 10 y
def equation_1_conditions (x y : ℝ) : Prop := x - 2 * y > 0 ∧ (x - 2 * y)^2 = x * y

-- The proof problem statement in Lean4
theorem math_problem :
  ∀ (a x y : ℝ), 
  (a_condition a) →
  (equation_1 x y) →
  (equation_1_conditions x y) →
  (a = 2) ∧
  (a^{-\frac{1}{2}} + a^{\frac{1}{2}} = (3 / 2) * real.sqrt 2) ∧
  (a^{\frac{3}{2}} + a^{-\frac{3}{2}} = (9 / 4) * real.sqrt 2) ∧
  (log a (y / x) = -2) :=
by intros a x y ha heq1 heq1_cond; sorry

end math_problem_l755_755751


namespace total_profit_at_end_of_year_l755_755641

theorem total_profit_at_end_of_year:
  ∀ (A_initial B_initial A_withdraw B_advance share_A : ℝ) (months_first months_last : ℝ),
  A_initial = 3000 →
  B_initial = 4000 →
  A_withdraw = 1000 →
  B_advance = 1000 →
  share_A = 320 →
  months_first = 8 →
  months_last = 4 →
  let A_investment_months := (A_initial * months_first) + ((A_initial - A_withdraw) * months_last),
      B_investment_months := (B_initial * months_first) + ((B_initial + B_advance) * months_last),
      total_investment_months := A_investment_months + B_investment_months,
      ratio_A := A_investment_months / total_investment_months,
      P := share_A / ratio_A
  in P = 840 :=
  by {
    intros,
    sorry,
  }


end total_profit_at_end_of_year_l755_755641


namespace problem1_problem2_l755_755429

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 5 - a / (Real.exp x)

-- Problem (1): If f(x) is monotonically increasing on ℝ, then a ∈ [2e, +∞)
theorem problem1 (a : ℝ) (h : ∀ x : ℝ, 2*x - 4 + a / (Real.exp x) ≥ 0) : a ∈ Set.Ici (2 * Real.exp 1) := 
sorry

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4*x + 5) * (Real.exp x) - a

-- Problem (2): Given g(x1) + g(x2) = 2 * g(m) for x1 < m and x2 > m and m ≥ 1, prove x1 + x2 < 2m
theorem problem2 (a m x1 x2 : ℝ) (hm : m ≥ 1) 
    (hx1 : x1 < m) (hx2 : x2 > m) 
    (h : g a x1 + g a x2 = 2 * g a m) : 
    x1 + x2 < 2 * m := 
sorry

end problem1_problem2_l755_755429


namespace increase_circumference_l755_755227

theorem increase_circumference (d1 d2 : ℝ) (increase : ℝ) (P : ℝ) : 
  increase = 2 * Real.pi → 
  P = Real.pi * increase → 
  P = 2 * Real.pi ^ 2 := 
by 
  intros h_increase h_P
  rw [h_P, h_increase]
  sorry

end increase_circumference_l755_755227
