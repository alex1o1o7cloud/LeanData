import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Cubics
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Field
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Quot
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.BigOperators
import Mathlib.Combinatorics.Combinatorial
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Floor
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Nat
import Mathlib.Init.Data.Real.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Module.Basic
import Mathlib.Topology.Continuity

namespace inequality_for_all_pos_integers_l722_722646

theorem inequality_for_all_pos_integers (n : ℕ) (h : n > 0) : 
  (∑ k in finset.range(n), 1 / (n + k)) ≥ n * (real.rpow 2 (1 / n) - 1) :=
sorry

end inequality_for_all_pos_integers_l722_722646


namespace maximize_area_pentagon_90_degrees_l722_722344

theorem maximize_area_pentagon_90_degrees (A B C : Type) [triangle A B C]
  (h1 : angle A B C = 45)
  (h2 : angle B A C ≤ 90)
  (h3 : dist B C = 2)
  (h4 : dist A C ≥ dist A B)
  (h5 : dist A B ≥ 4) :
  (maximize_area_pentagon A B C) = 90 :=
sorry

end maximize_area_pentagon_90_degrees_l722_722344


namespace distance_to_school_l722_722610

variables (d : ℝ)
def jog_rate := 5
def bus_rate := 30
def total_time := 1 

theorem distance_to_school :
  (d / jog_rate) + (d / bus_rate) = total_time ↔ d = 30 / 7 :=
by
  sorry

end distance_to_school_l722_722610


namespace jack_mopping_time_l722_722606

-- Definitions for the conditions
def bathroom_area : ℝ := 24
def kitchen_area : ℝ := 80
def mopping_rate : ℝ := 8

-- The proof problem: Prove Jack will spend 13 minutes mopping
theorem jack_mopping_time : (bathroom_area + kitchen_area) / mopping_rate = 13 := by
  sorry

end jack_mopping_time_l722_722606


namespace pyramid_area_l722_722317

def edge_length (a : ℝ) : Prop := a > 0

def midpoint (a : ℝ) : ℝ := a * (Real.sqrt 2) / 2

def slant_height (a : ℝ) : ℝ := Real.sqrt (a^2 + (a * (Real.sqrt 2) / 4)^2)

def lateral_surface_area (a : ℝ) : ℝ := 4 * (1/2) * midpoint a * slant_height a

def base_area (a : ℝ) : ℝ := (midpoint a)^2

def total_surface_area (a : ℝ) : ℝ := base_area a + lateral_surface_area a

theorem pyramid_area (a : ℝ) (h : edge_length a) : total_surface_area a = 2 * a^2 := 
by
  sorry

end pyramid_area_l722_722317


namespace max_sum_disjoint_subsets_l722_722258

open Finset

theorem max_sum_disjoint_subsets :
  ∃ (M : Finset ℕ), M ⊆ range 1 26 ∧
  (∀ (A B : Finset ℕ), A ⊆ M → B ⊆ M → A ∩ B = ∅ → A.sum (+) ≠ B.sum (+)) ∧
  M.sum (+) = 123 := sorry

end max_sum_disjoint_subsets_l722_722258


namespace set_intersection_complement_equiv_l722_722640

open Set

variable {α : Type*}
variable {x : α}

def U : Set ℝ := univ
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {x | x^2 < 1}

theorem set_intersection_complement_equiv :
  M ∩ (U \ N) = {x | 1 ≤ x} :=
by
  sorry

end set_intersection_complement_equiv_l722_722640


namespace non_real_roots_interval_l722_722181

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722181


namespace cube_root_numbers_less_than_15_l722_722101

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l722_722101


namespace eccentricity_of_ellipse_max_area_triangle_OMN_l722_722517

namespace EllipseProof

def ellipse_center_origin (e : ℝ) := e = ℝ.sqrt 2 / 2

theorem eccentricity_of_ellipse (e : ℝ) (h₁ : e = ℝ.sqrt 2 / 2) : ellipse_center_origin e := by
  -- proof to be added
  sorry

def max_area_OMN (A : ℝ) := A = ℝ.sqrt 2

theorem max_area_triangle_OMN (A : ℝ) (h₂ : A = ℝ.sqrt 2) : max_area_OMN A := by
  -- proof to be added
  sorry

end EllipseProof

end eccentricity_of_ellipse_max_area_triangle_OMN_l722_722517


namespace pyramid_volume_l722_722783

open Real

theorem pyramid_volume (A B C D E F G : Point) 
  (area_hex : ℝ) (area_ABG : ℝ) (area_DEG : ℝ) 
  (hA : area_hex = 648)
  (hB : area_ABG = 180)
  (hC : area_DEG = 162) :
  volume (pyramid A B C D E F G) = 432 * sqrt 22 :=
by
  sorry

end pyramid_volume_l722_722783


namespace range_of_transformed_function_l722_722546

noncomputable def f (a x : ℝ) : ℝ := a^x / (1 + a^x)

theorem range_of_transformed_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : 
    set.range (λ x, ⌊f a x - 1/2⌋ + ⌊f a (-x) - 1/2⌋) = {-1, 0} :=
sorry

end range_of_transformed_function_l722_722546


namespace probability_intersection_l722_722752

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l722_722752


namespace circle_center_radius_sum_18_l722_722248

-- Conditions from the problem statement
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * y - 9 = -y^2 + 18 * x + 9

-- Goal is to prove a + b + r = 18
theorem circle_center_radius_sum_18 :
  (∃ a b r : ℝ, 
     (∀ x y : ℝ, circle_eq x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
     a + b + r = 18) :=
sorry

end circle_center_radius_sum_18_l722_722248


namespace proof_problem_l722_722941

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the actual even function

variables (α β : ℝ) (h1 : α ≠ β) (h2 : 0 < α) (h3 : α < π / 2)
  (h4 : 0 < β) (h5 : β < π / 2)
  (h6 : f(α) = f(-α)) -- f is even
  (h7 : ∀ x y, x < y → -1 ≤ x ∧ y ≤ 0 → f(x) > f(y)) -- f is decreasing on [-1, 0]

theorem proof_problem :
  f(cos α) < f(sin β) :=
by
  sorry

end proof_problem_l722_722941


namespace number_of_prime_factors_30_factorial_l722_722971

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722971


namespace find_parenthesis_value_l722_722555

theorem find_parenthesis_value (x : ℝ) (h : x * (-2/3) = 2) : x = -3 :=
by
  sorry

end find_parenthesis_value_l722_722555


namespace aunt_gemma_feeds_dogs_twice_per_day_l722_722805

theorem aunt_gemma_feeds_dogs_twice_per_day
  (num_dogs : ℕ) (food_per_meal : ℕ) (num_sacks : ℕ) (weight_per_sack : ℕ) (total_days : ℕ) : 
  num_dogs = 4 →
  food_per_meal = 250 →
  num_sacks = 2 →
  weight_per_sack = 50000 →  -- using grams instead of kg for consistency
  total_days = 50 →
  let total_food := num_sacks * weight_per_sack in
  let daily_food := total_food / total_days in
  let food_per_dog_per_day := daily_food / num_dogs in
  let meals_per_day := food_per_dog_per_day / food_per_meal in
  meals_per_day = 2 :=
begin
  intros h_dogs h_meal h_sacks h_weight h_days,
  dsimp at *,
  have total_food_calc : total_food = 100000, by {
    rw [h_sacks, h_weight],
    norm_num,
  },
  have daily_food_calc : daily_food = 2000, by {
    rw [total_food_calc, h_days],
    norm_num,
  },
  have food_per_dog_calc : food_per_dog_per_day = 500, by {
    rw [daily_food_calc, h_dogs],
    norm_num,
  },
  have meals_per_day_calc : meals_per_day = 2, by {
    rw [food_per_dog_calc, h_meal],
    norm_num,
  },
  exact meals_per_day_calc,
end

end aunt_gemma_feeds_dogs_twice_per_day_l722_722805


namespace find_inverse_value_l722_722087

noncomputable def g (x : ℕ) := 4 * x^3 + 3

theorem find_inverse_value (x : ℕ) : (g⁻¹ 5) = 503 :=
by
  sorry

end find_inverse_value_l722_722087


namespace relationship_l722_722935

noncomputable def a : ℝ := Real.log 2 (2 / 5)
noncomputable def b : ℝ := 0.4 ^ 8
noncomputable def c : ℝ := Real.log 2

theorem relationship (ha : a = Real.log 2 (2 / 5)) (hb : b = 0.4 ^ 8) (hc : c = Real.log 2) : a < b ∧ b < c := 
by sorry

end relationship_l722_722935


namespace all_three_reach_in_3_hours_l722_722409

-- Definitions
def distance_to_grandmothers_house : ℝ := 33
def speed_father_alone : ℝ := 25
def speed_father_with_passenger : ℝ := 20
def speed_son_walking : ℝ := 5
def total_time_allowed : ℝ := 3

-- Theorem Statement
theorem all_three_reach_in_3_hours
  (distance_to_grandmother : ℝ)
  (speed_father_alone : ℝ)
  (speed_father_with_passenger : ℝ)
  (speed_son_walking : ℝ)
  (total_time : ℝ) :
  distance_to_grandmother = 33 ∧
  speed_father_alone = 25 ∧
  speed_father_with_passenger = 20 ∧
  speed_son_walking = 5 ∧
  total_time = 3 →
  (∃ (t1 t2 t3 : ℝ),
    t1 + t2 + t3 ≤ total_time ∧
    (speed_father_alone * t1 + speed_father_with_passenger * t2 + speed_son_walking * t3) = distance_to_grandmother) :=
begin
  sorry
end

end all_three_reach_in_3_hours_l722_722409


namespace time_to_pay_back_l722_722621

def total_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

def monthly_profit := monthly_revenue - monthly_expenses

theorem time_to_pay_back : 
  (total_cost / monthly_profit) = 10 := 
by
  -- Definition of monthly_profit 
  have monthly_profit_def : monthly_profit = 4000 - 1500 := rfl
  rw [monthly_profit_def]
  
  -- Performing the division
  show (25000 / 2500) = 10
  apply div_eq_of_eq_mul
  norm_num
  sorry

end time_to_pay_back_l722_722621


namespace four_points_form_rhombus_with_60_angle_l722_722505

/-- Given 5 points on a plane, where any 3 points out of any 4 points form the vertices 
    of an equilateral triangle, prove that among the 5 points, there are 4 points 
    that form the vertices of a rhombus with an interior angle of 60°. -/

theorem four_points_form_rhombus_with_60_angle 
  (points : Fin 5 → ℝ × ℝ) 
  (h : ∀ (s : Finset (Fin 5)), s.card = 4 → ∃ (a b c : Fin 5), {a, b, c}.card = 3 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
        dist (points a) (points b) = dist (points b) (points c) ∧ 
        dist (points b) (points c) = dist (points c) (points a) ∧ 
        dist (points c) (points a) = dist (points a) (points b)) : 
  ∃ (x y z w : Fin 5), {x, y, z, w}.card = 4 ∧ 
      dist (points x) (points y) = dist (points y) (points z) ∧ 
      dist (points y) (points z) = dist (points z) (points w) ∧ 
      dist (points z) (points w) = dist (points w) (points x) ∧ 
      angle_eq (points x) (points y) (points z) (π / 3) := 
sorry

end four_points_form_rhombus_with_60_angle_l722_722505


namespace vector_magnitude_b_l722_722549

variable (a b : ℝ)

def vector_magnitude (v : ℝ) : ℝ := real.sqrt (v * v)  -- Definition of vector magnitude

variable (a b : ℝ)

# The conditions translated into Lean 4:
def vector_conditions (a b : ℝ) : Prop :=
  vector_magnitude a = 1 ∧
  ((a + b) * a = 0) ∧
  ((2 * a + b) * b = 0)

# The statement we want to prove:
theorem vector_magnitude_b (a b : ℝ) (h : vector_conditions a b) : vector_magnitude b = real.sqrt 2 :=
sorry

end vector_magnitude_b_l722_722549


namespace nested_sqrt_eq_l722_722871

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722871


namespace value_of_S_l722_722624

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def S := (1 / (4 - sqrt 15)) - (1 / (sqrt 15 - sqrt 14)) + (1 / (sqrt 14 - sqrt 13)) - (1 / (sqrt 13 - sqrt 12)) + (1 / (sqrt 12 - 3))

theorem value_of_S : S = 7 := 
by
  sorry

end value_of_S_l722_722624


namespace probability_intersection_l722_722750

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l722_722750


namespace volume_of_rectangular_box_l722_722360

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l722_722360


namespace non_real_roots_interval_l722_722177

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722177


namespace vector_magnitude_problem_l722_722080

variables (x : ℝ) (a b : ℝ × ℝ)
def is_perpendicular (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

def vector_sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
  (v₁.1 - v₂.1, v₁.2 - v₂.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_problem : 
  ∀ (x : ℝ), is_perpendicular (x + 1, 2) (1, -2) → magnitude (vector_sub (x + 1, 2) (1, -2)) = 5 := 
by
  intros x h1
  sorry

end vector_magnitude_problem_l722_722080


namespace smaller_square_area_proof_l722_722644

-- Definitions based on the given conditions
def larger_square_area : ℝ := 100
def side_length_larger_square : ℝ := Real.sqrt larger_square_area
def midpoint_side_length : ℝ := side_length_larger_square / 2
def smaller_square_side_length : ℝ := Real.sqrt (midpoint_side_length ^ 2 + midpoint_side_length ^ 2)
def smaller_square_area : ℝ := smaller_square_side_length ^ 2

-- The proof statement
theorem smaller_square_area_proof :
  ∀ (points_midpoints : Prop), 
  (larger_square_area = 100) ∧ 
  points_midpoints →
  smaller_square_area = 50 :=
by
  sorry

end smaller_square_area_proof_l722_722644


namespace kelly_password_combinations_l722_722611

theorem kelly_password_combinations : 
  let odd_digits := {1, 3, 5}
  let even_digits := {2, 4, 6}
  let total_combinations :=
    (3 * 3 * 6 * 6) + (3 * 3 * 6 * 6)
  in total_combinations = 648 :=
by
  let odd_digits := {1, 3, 5}
  let even_digits := {2, 4, 6}
  let total_combinations :=
    (3 * 3 * 6 * 6) + (3 * 3 * 6 * 6)
  show total_combinations = 648
  sorry

end kelly_password_combinations_l722_722611


namespace domain_of_f_period_of_f_increasing_interval_of_f_decreasing_interval_of_f_l722_722542

noncomputable def f (x : ℝ) := 4 * (Real.tan x) * (Real.sin (π/2 - x)) * (Real.cos (x - π/3)) - Real.sqrt 3

theorem domain_of_f :
  ∀ (x : ℝ), x ∉ {y | ∃ (k : ℤ), y = k * π + π / 2 } ↔ Function.is_domain f x := sorry

theorem period_of_f :
  Function.periodic f (π : ℝ) := sorry

theorem increasing_interval_of_f :
  ∀ (x : ℝ), 
  (-π / 12) ≤ x ∧ x ≤ π/4 -> 
  ∀ (y : ℝ), x ≤ y → y ≤ π/4 → f x ≤ f y := sorry

theorem decreasing_interval_of_f :
  ∀ (x : ℝ), 
  (-π / 4) ≤ x ∧ x ≤ -π/12 -> 
  ∀ (y : ℝ), x ≤ y → y ≤ -π/12 → f x ≥ f y := sorry

end domain_of_f_period_of_f_increasing_interval_of_f_decreasing_interval_of_f_l722_722542


namespace prob_A_inter_B_l722_722739

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l722_722739


namespace log_4_1_div_16_eq_neg_2_l722_722469

theorem log_4_1_div_16_eq_neg_2 : log 4 (1 / 16) = -2 :=
by
  sorry

end log_4_1_div_16_eq_neg_2_l722_722469


namespace distance_between_parallel_lines_l722_722901

theorem distance_between_parallel_lines 
    (A B : ℝ) 
    (C1 C2 : ℝ)
    (h1 : A = 2)
    (h2 : B = 3)
    (h3 : C1 = -5)
    (h4 : C2 = -2) :
    let d := (|C2 - C1| / Real.sqrt (A^2 + B^2))
    in d = 3 * Real.sqrt 13 / 13 :=
by
  sorry

end distance_between_parallel_lines_l722_722901


namespace sandy_total_money_received_l722_722651

def sandy_saturday_half_dollars := 17
def sandy_sunday_half_dollars := 6
def half_dollar_value : ℝ := 0.50

theorem sandy_total_money_received :
  (sandy_saturday_half_dollars * half_dollar_value) +
  (sandy_sunday_half_dollars * half_dollar_value) = 11.50 :=
by
  sorry

end sandy_total_money_received_l722_722651


namespace calculate_area_l722_722810

noncomputable def area_between_curves : ℝ :=
  let f := λ y : ℝ, 4 * y - 8
  let g := λ y : ℝ, (y - 2) ^ 3
  ∫ y in (2 : ℝ)..4, f y - g y

theorem calculate_area :
  area_between_curves = 8 :=
by
  sorry

end calculate_area_l722_722810


namespace friends_to_Hoseok_left_l722_722339

theorem friends_to_Hoseok_left (total_people friends_right : ℕ) (h_total : total_people = 16) (h_right : friends_right = 8) :
  (total_people - (friends_right + 1)) = 7 :=
by
  rw [h_total, h_right]
  norm_num

end friends_to_Hoseok_left_l722_722339


namespace number_of_distinct_prime_factors_30_fact_l722_722997

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722997


namespace man_twice_son_age_l722_722780

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 18) (h2 : M = S + 20) 
  (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  -- Proof steps can be added here later
  sorry

end man_twice_son_age_l722_722780


namespace volume_rectangular_box_l722_722368

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l722_722368


namespace number_of_distinct_prime_factors_30_fact_l722_722988

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722988


namespace Todd_spent_correct_amount_l722_722205

noncomputable def init_money : ℝ := 37.25
noncomputable def candy_bar : ℝ := 1.14
noncomputable def cookies : ℝ := 2.39
noncomputable def soda : ℝ := 1.75
noncomputable def chips : ℝ := 1.85
noncomputable def juice : ℝ := 2.69
noncomputable def hamburger : ℝ := 3.99

noncomputable def discount_candy_bar : ℝ := 0.12
noncomputable def discount_hamburger : ℝ := 0.12
noncomputable def discount_cookies : ℝ := 0.15

noncomputable def sales_tax : ℝ := 0.085

axiom final_amount_spent : ℝ := 13.93

theorem Todd_spent_correct_amount :
  let candy_bar_after_discount := candy_bar - candy_bar * discount_candy_bar,
      hamburger_after_discount := hamburger - hamburger * discount_hamburger,
      cookies_after_discount := cookies - cookies * discount_cookies,
      total_after_discounts := candy_bar_after_discount + cookies_after_discount + soda + chips + juice + hamburger_after_discount,
      tax_amount := total_after_discounts * sales_tax,
      total_amount := total_after_discounts + tax_amount
  in total_amount = final_amount_spent :=
by sorry

end Todd_spent_correct_amount_l722_722205


namespace intersection_point_distance_l722_722765

noncomputable def centerDistance : ℝ :=
  let r := 2
  let s := 2 * r
  let chordDistance := 2 * (Math.sqrt (2 * r ^ 2) - r)
  chordDistance

theorem intersection_point_distance (circle_radius bisects : ℝ) (h1 : circle_radius = 2) (h2 : bisects = 2) :
  centerDistance = 2 * √ 2 - 2 :=
by
  rw [h1, h2]
  sorry

end intersection_point_distance_l722_722765


namespace lattice_point_intersection_l722_722208

open Int

theorem lattice_point_intersection : ∃ (k_values : Finset ℤ), k_values.card = 4 ∧ ∀ k ∈ k_values, ∃ (x y : ℤ), y = 2 * x - 1 ∧ y = k * x + k :=
by
  sorry

end lattice_point_intersection_l722_722208


namespace sequence_inequality_l722_722639

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
  (h_condition : ∀ n m, a (n + m) ≤ a n + a m) :
  ∀ n m, n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := by
  sorry

end sequence_inequality_l722_722639


namespace generatrix_length_l722_722037

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722037


namespace range_of_a_for_less_than_three_zeros_l722_722530

def polynomial_has_less_than_three_zeros (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, (f = λ x, x^3 - a * x^2 + 4) ∧ (a ≤ 3 → (∀ x₁ x₂ x₃ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) → x₁ = x₂ ∨ x₁ = x₃ ∨ x₂ = x₃))

theorem range_of_a_for_less_than_three_zeros :
  ∀ (a : ℝ), polynomial_has_less_than_three_zeros (λ x : ℝ, x^3 - a * x^2 + 4) ↔ a ∈ Iic 3 :=
by sorry

end range_of_a_for_less_than_three_zeros_l722_722530


namespace james_total_earnings_l722_722607

-- Assume the necessary info for January, February, and March earnings
-- Definitions given as conditions in a)
def January_earnings : ℝ := 4000

def February_earnings : ℝ := January_earnings * 1.5 * 1.2

def March_earnings : ℝ := February_earnings * 0.8

-- The total earnings to be calculated
def Total_earnings : ℝ := January_earnings + February_earnings + March_earnings

-- Prove the total earnings is $16960
theorem james_total_earnings : Total_earnings = 16960 := by
  sorry

end james_total_earnings_l722_722607


namespace max_value_f_l722_722939

noncomputable def f (a b : ℝ) := abs (sqrt (a + 1/b) - sqrt (b + 1/a))

theorem max_value_f :
  ∀ (a b : ℝ), (a ∈ Icc 1 3) → (b ∈ Icc 1 3) → (a + b = 4)
  → f a b ≤ 2 - 2 / Real.sqrt 3 :=
by
  sorry

end max_value_f_l722_722939


namespace common_ratio_of_series_l722_722899

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l722_722899


namespace roses_cut_l722_722705

variable (initial final : ℕ) -- Declare variables for initial and final numbers of roses

-- Define the theorem stating the solution
theorem roses_cut (h1 : initial = 6) (h2 : final = 16) : final - initial = 10 :=
sorry -- Use sorry to skip the proof

end roses_cut_l722_722705


namespace find_y_div_x_max_sin2A_plus_2cosB_l722_722524

variables (x y : ℝ)
variables (A B C : ℝ)
variables (Δ : Triangle A B C)

-- Condition: x and y are non-zero
variable (non_zero_x : x ≠ 0)
variable (non_zero_y : y ≠ 0)

-- Condition: they satisfy the given equation
variable (h : (x * Real.sin (π / 5) + y * Real.cos (π / 5)) / (x * Real.cos (π / 5) - y * Real.sin (π / 5)) = Real.tan (9 * π / 20))

-- Proof requirement 1: Find the value of y/x
theorem find_y_div_x : y / x = 1 := sorry

-- Proof requirement 2: Given tan C = y / x in ΔABC, find the max value of sin 2A + 2 cos B
theorem max_sin2A_plus_2cosB (hC : Real.tan C = 1) : (sin (2 * A) + 2 * cos B) ≤ 3 / 2 := sorry

end find_y_div_x_max_sin2A_plus_2cosB_l722_722524


namespace function_periodicity_l722_722192

theorem function_periodicity (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : ∀ x, f (x + 1) = f (1 - x)) (h3 : f 1 = 5) : f 2015 = -5 :=
sorry

end function_periodicity_l722_722192


namespace sqrt_continued_fraction_l722_722862

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722862


namespace evaluate_nested_radical_l722_722860

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722860


namespace rounding_to_hundredth_l722_722650

theorem rounding_to_hundredth (x : ℝ) (h : x = 52.63847) : Real.round(100 * x) / 100 = 52.64 := 
by 
  sorry

end rounding_to_hundredth_l722_722650


namespace difference_one_third_0_333_l722_722352

theorem difference_one_third_0_333 :
  let one_third : ℚ := 1 / 3
  let three_hundred_thirty_three_thousandth : ℚ := 333 / 1000
  one_third - three_hundred_thirty_three_thousandth = 1 / 3000 :=
by
  sorry

end difference_one_third_0_333_l722_722352


namespace problem1_problem2_l722_722264

noncomputable def a_seq (n : ℕ) : ℝ := n

theorem problem1 (S_n : ℕ → ℝ) (a_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n > 0)
  (h2 : ∀ n, a_n n ^ 2 + a_n n = 2 * S_n n) :
  ∀ n, a_n n = n :=
sorry

theorem problem2 (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :
  (∀ n, b_n n = 2 / (a_seq n * a_seq (n + 2))) →
  (∀ n, T_n n = ∑ i in finset.range n, b_n i) →
  ∀ n, T_n n = 3 / 2 - 1 / (n + 1) - 1 / (n + 2) :=
sorry

end problem1_problem2_l722_722264


namespace trapezoid_area_is_correct_l722_722890

noncomputable def trapezoid_area
  (BC AD AB CD : ℝ)
  (h_base : BC = 4)
  (h_top_base : AD = 7)
  (h_leg1 : AB = 4)
  (h_leg2 : CD = 5) : ℝ :=
  let h := 4 in  -- Height derived from the right triangle formed in the solution.
  1 / 2 * (BC + AD) * h

theorem trapezoid_area_is_correct :
  trapezoid_area 4 7 4 5 4 7 4 5 = 22 :=
by
  simp [trapezoid_area]
  sorry

end trapezoid_area_is_correct_l722_722890


namespace tom_purchases_mangoes_l722_722343

theorem tom_purchases_mangoes (m : ℕ) (h1 : 8 * 70 + m * 65 = 1145) : m = 9 :=
by
  sorry

end tom_purchases_mangoes_l722_722343


namespace mode_of_data_is_9_l722_722204

def data : List ℕ := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ n m, if l.count n > l.count m then n else m) 0

theorem mode_of_data_is_9 : mode data = 9 := by
  sorry

end mode_of_data_is_9_l722_722204


namespace sum_of_digits_of_palindromes_l722_722788

def is_palindrome (n : ℕ) : Prop :=
  n / 100000 = n % 10
  ∧ (n / 10000) % 10 = (n % 100) / 10
  ∧ (n / 1000) % 10 = (n % 1000) / 100

def valid_digits (n : ℕ) : Prop :=
  let a := n / 100000 in
  1 ≤ a ∧ a ≤ 9

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_of_palindromes :
  let T := (List.range 1000000).filter (λ n, valid_digits n ∧ is_palindrome n) |>.sum in
  sum_of_digits T = 20 := by
  sorry

end sum_of_digits_of_palindromes_l722_722788


namespace chess_pieces_remaining_l722_722652

theorem chess_pieces_remaining (scarlett_lost : ℕ) (hannah_lost : ℕ) (initial_pieces : ℕ) : 
  scarlett_lost = 6 → hannah_lost = 8 → initial_pieces = 32 → 
  initial_pieces - (scarlett_lost + hannah_lost) = 18 :=
by
  intros hScarlett hHannah hInitial
  rw [hScarlett, hHannah, hInitial]
  norm_num
  sorry

end chess_pieces_remaining_l722_722652


namespace not_product_of_two_integers_l722_722628

theorem not_product_of_two_integers (n : ℕ) (hn : n > 0) :
  ∀ t k : ℕ, t * (t + k) = n^2 + n + 1 → k ≥ 2 * Nat.sqrt n :=
by
  sorry

end not_product_of_two_integers_l722_722628


namespace cone_generatrix_length_l722_722060

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722060


namespace find_PC_l722_722709

theorem find_PC (A B C P: Type) [EuclideanGeometry A B C] [RightTriangle B 90]
  (PA PB PC: ℝ) (h₁: PA = 12) (h₂: PB = 8)
  (angle_P: ℝ) (h₃: angle_P = 120):
  PC = 16 :=
by
  sorry

end find_PC_l722_722709


namespace common_tangent_line_l722_722544

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x + (m * x) / (x + 1)
def g (x : ℝ) : ℝ := x^2 + 1

theorem common_tangent_line (m : ℝ) (a : ℝ) (h_pos : a > 0) :
  (∃ x1 x2 : ℝ, 
    (a = 2 * x2) ∧ 
    (f x1 m = a * x1) ∧ 
    (f' x1 m = a) ∧ 
    (g x2 = a * x2) ∧ 
    (g' x2 = a)) → (m = 4) :=
begin
  sorry
end

end common_tangent_line_l722_722544


namespace mark_savings_l722_722271

-- Given conditions
def original_price : ℝ := 300
def discount_rate : ℝ := 0.20
def cheaper_lens_price : ℝ := 220

-- Definitions derived from conditions
def discount_amount : ℝ := original_price * discount_rate
def discounted_price : ℝ := original_price - discount_amount
def savings : ℝ := discounted_price - cheaper_lens_price

-- Statement to prove
theorem mark_savings : savings = 20 :=
by
  -- Definitions incorporated
  have h1 : discount_amount = 300 * 0.20 := rfl
  have h2 : discounted_price = 300 - discount_amount := rfl
  have h3 : cheaper_lens_price = 220 := rfl
  have h4 : savings = discounted_price - cheaper_lens_price := rfl
  sorry

end mark_savings_l722_722271


namespace find_x_l722_722485

noncomputable def t_negative (a : ℝ) (ha : a < 0) : ℝ := (1 - Real.sqrt (1 - 8 * a + 12 * a ^ 2)) / (2 * a)
noncomputable def x_negative (a : ℝ) (ha : a < 0) : ℝ := (t_negative a ha) ^ 2

noncomputable def t_positive (a : ℝ) (ha : 2 / 3 < a) : ℝ := (1 + Real.sqrt (1 - 8 * a + 12 * a ^ 2)) / (2 * a)
noncomputable def x_positive (a : ℝ) (ha : 2 / 3 < a) : ℝ := (t_positive a ha) ^ 2

noncomputable def x_a0 : ℝ := 4

def correct_x (a : ℝ) (b : ℝ) (hb : b = 0) : ℝ :=
  if ha : a < 0 then x_negative a ha
  else if ha : a = 0 then x_a0
  else if ha : 2 / 3 < a then x_positive a ha
  else 0 -- Default case

theorem find_x (a : ℝ) (b : ℝ) (hb : b = 0) :
  a < 0 ∨ a = 0 ∨ 2 / 3 < a → 
  (correct_x a b hb) = 
  if a < 0 then
    (1 - Real.sqrt (1 - 8 * a + 12 * a ^ 2)) ^ 2 / (4 * a ^ 2) 
  else if a = 0 then 4 
  else (1 + Real.sqrt (1 - 8 * a + 12 * a ^ 2)) ^ 2 / (4 * a ^ 2) := 
sorry

end find_x_l722_722485


namespace real_possible_b_values_quadratic_non_real_roots_l722_722147

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722147


namespace first_day_is_sunday_l722_722668

noncomputable def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_is_sunday :
  (day_of_week 18 = "Wednesday") → (day_of_week 1 = "Sunday") :=
by
  intro h
  -- proof would go here
  sorry

end first_day_is_sunday_l722_722668


namespace number_of_correct_propositions_is_zero_l722_722007

-- Definitions of the propositions
def proposition1 : Prop :=
  ∀ (prism : Type) (quad : Type), 
    (lateral_faces_congruent_quadrilaterals prism quad) → 
    (is_right_prism prism)

def proposition2 : Prop :=
  ∀ (hexahedron : Type) (rect : Type), 
    (opposite_faces_congruent_rectangles hexahedron rect) → 
    (is_rectangular_prism hexahedron)

def proposition3 : Prop :=
  ∀ (prism : Type) (face : Type),
    (two_lateral_faces_perpendicular prism face) → 
    (is_right_prism prism)

def proposition4 : Prop :=
  ∀ (rect_prism : Type) (quad : Type),
    (is_rectangular_prism rect_prism) → 
    (is_regular_quadrilateral_prism rect_prism quad)

-- Collecting the propositions into a list
def propositions : list Prop :=
  [proposition1, proposition2, proposition3, proposition4]

-- Main theorem stating that the number of correct propositions is 0
theorem number_of_correct_propositions_is_zero : 
  (count (λ p, p = true) propositions) = 0 := 
sorry

end number_of_correct_propositions_is_zero_l722_722007


namespace angle_B_value_sin_A_value_l722_722919

-- Define the properties of the given triangle
variables {A B C : ℝ} {a b c : ℝ}

-- First statement: proving B = π / 3
theorem angle_B_value (h : a^2 + c^2 - b^2 = ac) : B = π / 3 := sorry

-- Second statement: proving sin A when c = 3a
theorem sin_A_value (h1 : a^2 + (3 * a)^2 - b^2 = a * 3 * a) (h2 : c = 3 * a) : sin A = sqrt 21 / 14 := sorry

end angle_B_value_sin_A_value_l722_722919


namespace find_B_l722_722256

noncomputable def A : (ℝ × ℝ) := (2, 4)

def parabola (x : ℝ) : ℝ := 2 * x^2

def normal_line (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  let m : ℝ := -1 / 8
  m * x + (p.snd - m * p.fst)

theorem find_B :
  (∃ B : ℝ × ℝ,
    B ≠ A ∧
    B.snd = parabola B.fst ∧
    B.snd = normal_line A B.fst ∧
    B = ( ( -1 - real.sqrt 2177 ) / 32, parabola ( (-1 - real.sqrt 2177 ) / 32 ) ) ) :=
by
  sorry

end find_B_l722_722256


namespace triangle_segment_larger_part_l722_722694

theorem triangle_segment_larger_part (a b c : ℝ) (h_triangle : a = 30 ∧ b = 70 ∧ c = 80) 
    (h_altitude : ∃ x y : ℝ, (30^2 = x^2 + y^2) ∧ (70^2 = (80 - x)^2 + y^2)) : 
    (80 - (classical.some (classical.some_spec h_altitude).1)) = 65 :=
by 
  sorry

end triangle_segment_larger_part_l722_722694


namespace nested_radical_solution_l722_722838

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722838


namespace vertex_angle_measure_l722_722799

-- Definitions for Lean Proof
def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (α = γ) ∨ (β = γ)
def exterior_angle (interior exterior : ℝ) : Prop := interior + exterior = 180

-- Conditions from the problem
variables (α β γ : ℝ)
variable (ext_angle : ℝ := 110)

-- Lean 4 statement: The measure of the vertex angle is 70° or 40°
theorem vertex_angle_measure :
  isosceles_triangle α β γ ∧
  (exterior_angle γ ext_angle ∨ exterior_angle α ext_angle ∨ exterior_angle β ext_angle) →
  (γ = 70 ∨ γ = 40) :=
by
  sorry

end vertex_angle_measure_l722_722799


namespace pipes_fill_tank_in_one_hour_l722_722281

theorem pipes_fill_tank_in_one_hour (p q r s : ℝ) (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  1 / (p + q + r + s) = 1 :=
by
  sorry

end pipes_fill_tank_in_one_hour_l722_722281


namespace train_length_l722_722732

theorem train_length
  (S : ℝ)  -- speed of the train in meters per second
  (L : ℝ)  -- length of the train in meters
  (h1 : L = S * 20)
  (h2 : L + 500 = S * 40) :
  L = 500 := 
sorry

end train_length_l722_722732


namespace num_pos_nums_with_cube_root_lt_15_l722_722133

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l722_722133


namespace real_possible_b_values_quadratic_non_real_roots_l722_722146

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722146


namespace ratio_of_ages_l722_722772

theorem ratio_of_ages (father_age son_age : ℕ) (h1 : father_age = 40) (h2 : son_age = 10) : father_age / son_age = 4 :=
by
  rw [h1, h2]
  norm_num

end ratio_of_ages_l722_722772


namespace nested_sqrt_eq_l722_722848

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722848


namespace scientific_notation_1_2_billion_l722_722207

theorem scientific_notation_1_2_billion : (1_200_000_000 : ℝ) = 1.2 * 10^9 := by
  sorry

end scientific_notation_1_2_billion_l722_722207


namespace min_value_in_positive_reals_range_of_x_l722_722921

-- Proof Problem 1:
theorem min_value_in_positive_reals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / a + 4 / b) ≥ 9 / 2 :=
sorry

-- Proof Problem 2:
theorem range_of_x (x : ℝ) (A B : Type) [ordered_ring A] [ordered_ring B]) :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|)) →
  -5 / 2 ≤ x ∧ x ≤ 13 / 2 :=
sorry

end min_value_in_positive_reals_range_of_x_l722_722921


namespace circumcircle_triangle_tangent_l722_722450

variables {ω Ω : Circle} {A B M P Q : Point} {ℓ_P ℓ_Q: Line}

-- Conditions
axiom intersect (ω Ω: Circle) (A B : Point) : meet ω Ω A B

axiom midpoint (M : Point) (A B: Point) (ω: Circle) : arc_midpoint ω A B M

axiom chord_intersect (MP : Chord) (ω: Circle) (Ω: Circle) (M P Q : Point) :
  chord(M, P) → intersect MP Ω Q

axiom tangent_at (ℓ_p ℓ_q : Line) (P Q : Point) (ω Ω: Circle) :
  tangent_at ω P ℓ_p ∧ tangent_at Ω Q ℓ_q

-- Theorem to prove
theorem circumcircle_triangle_tangent
  (c1 : meet ω Ω A B)
  (c2 : arc_midpoint ω A B M)
  (c3 : chord_intersect (chord M P) ω Ω Q)
  (c4 : tangent_at ℓ_P ω P)
  (c5 : tangent_at ℓ_Q Ω Q) :
  tangent (circumcircle (triangle ℓ_P ℓ_Q (lineA_B))) Ω :=
sorry

end circumcircle_triangle_tangent_l722_722450


namespace find_height_of_door_l722_722222

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l722_722222


namespace hexagon_largest_angle_degrees_l722_722323

theorem hexagon_largest_angle_degrees :
  let x := (720 : ℝ) / 22 in
  6 * x = (2160 : ℝ) / 11 :=
by
  sorry

end hexagon_largest_angle_degrees_l722_722323


namespace cone_generatrix_length_l722_722050

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722050


namespace quadratic_non_real_roots_b_values_l722_722172

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722172


namespace nested_radical_solution_l722_722833

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722833


namespace domain_f_div_x_2_eq_l722_722538

noncomputable def domain_f_lg_x := set.Icc 0.1 100
noncomputable def domain_f_x := set.Icc (-1 : ℝ) (2 : ℝ)
noncomputable def domain_f_div_x_2 := set.Icc (-2 : ℝ) 4

theorem domain_f_div_x_2_eq :
  let f : ℝ → ℝ := sorry 
  (∀ x, x ∈ domain_f_lg_x → log x ∈ domain_f_x) →
  domain_f_div_x_2 = set.Icc (-2) 4 :=
by
  intros
  sorry

end domain_f_div_x_2_eq_l722_722538


namespace annie_blocks_walked_l722_722443

theorem annie_blocks_walked (x : ℕ) (h1 : 7 * 2 = 14) (h2 : 2 * x + 14 = 24) : x = 5 :=
by
  sorry

end annie_blocks_walked_l722_722443


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722108

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722108


namespace generatrix_length_of_cone_l722_722023

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722023


namespace sequence_100th_term_l722_722497

theorem sequence_100th_term (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, a (i + 1)) / n = n + 1) : 
  a 100 = 200 :=
sorry

end sequence_100th_term_l722_722497


namespace successive_product_4160_l722_722737

theorem successive_product_4160 (n : ℕ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_4160_l722_722737


namespace nested_sqrt_eq_l722_722847

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722847


namespace volume_rectangular_box_l722_722366

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l722_722366


namespace number_of_prime_factors_30_factorial_l722_722982

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722982


namespace real_possible_b_values_quadratic_non_real_roots_l722_722148

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722148


namespace solve_eq1_solve_eq2_l722_722301

open Real

noncomputable theory

-- Define the equation 1 as a Lean function
def eq1 (x: ℝ) : Prop := (5 / (2 * x)) - (1 / (x - 3)) = 0

-- Prove the solution for equation 1
theorem solve_eq1 : ∃ x : ℝ, eq1 x ∧ x = 5 := 
by {
  use 5,
  unfold eq1,
  norm_num,
  norm_num,
  sorry
}

-- Define the equation 2 as a Lean function
def eq2 (x: ℝ) : Prop := (1 / (x - 2)) = (4 / (x^2 - 4))

-- Prove there is no solution for equation 2
theorem solve_eq2 : ¬ (∃ x : ℝ, eq2 x) := 
by {
  intro h,
  cases h with x hx,
  unfold eq2 at hx,
  -- Handle the equation after unfolding, norm, and simplifying
  sorry
}

end solve_eq1_solve_eq2_l722_722301


namespace nested_sqrt_eq_l722_722850

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722850


namespace quadratic_non_real_roots_l722_722163

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722163


namespace necessary_sufficient_a_l722_722927

noncomputable def conditions_on_a (a : ℝ) (k : ℕ) : Prop :=
  match k with
  | 1 => 0 < a ∧ a < real.sqrt 3
  | 2 => 0 < a ∧ a < real.sqrt (2 + real.sqrt 3)
  | 3 => 0 < a
  | 4 => a > 1 / real.sqrt (2 + real.sqrt 3)
  | 5 => a > 1 / real.sqrt 3
  | _ => false

theorem necessary_sufficient_a (a : ℝ) (k : ℕ) :
  tetrahedron_exists_with_conditions k a ↔ conditions_on_a a k :=
sorry

end necessary_sufficient_a_l722_722927


namespace shorter_leg_of_right_triangle_l722_722585

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l722_722585


namespace smallest_possible_beta_l722_722631

noncomputable def unit_vector (v : ℝ × ℝ × ℝ) : Prop := ∥v∥ = 1

theorem smallest_possible_beta
  (u v w : ℝ × ℝ × ℝ)
  (beta : ℝ)
  (hu : unit_vector u)
  (hv : unit_vector v)
  (hw : unit_vector w)
  (angle_uv : real.angle u v = beta)
  (angle_w_uv : real.angle w (u × v) = beta / 2)
  (v_dot_wxu : v • (w × u) = sqrt 3 / 4) :
  beta = 30 :=
sorry

end smallest_possible_beta_l722_722631


namespace total_amount_of_check_l722_722404

def numParts : Nat := 59
def price50DollarPart : Nat := 50
def price20DollarPart : Nat := 20
def num50DollarParts : Nat := 40

theorem total_amount_of_check : (num50DollarParts * price50DollarPart + (numParts - num50DollarParts) * price20DollarPart) = 2380 := by
  sorry

end total_amount_of_check_l722_722404


namespace simplification_evaluates_to_1_l722_722293

-- Definitions based on conditions
variable (a : ℤ)
def valid_a (a : ℤ) : Prop := a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1 ∧ a ≠ 2

-- The Lean statement of the proof problem
theorem simplification_evaluates_to_1 (a : ℤ) (ha : valid_a a) :
  ( \left( \frac{a^2 - 2a}{a^2 - 4a + 4} + 1 \right) / \frac{a^2 - 1}{a^2 + a} ) = 1 := 
sorry

end simplification_evaluates_to_1_l722_722293


namespace time_to_pay_back_l722_722613

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l722_722613


namespace solution_set_of_inequality_l722_722489

theorem solution_set_of_inequality :
  { x : ℝ | 2 * x^2 - x - 3 > 0 } = { x : ℝ | x > 3 / 2 ∨ x < -1 } :=
sorry

end solution_set_of_inequality_l722_722489


namespace positive_whole_numbers_with_cube_roots_less_than_15_l722_722119

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l722_722119


namespace parallelogram_AD_length_l722_722625

noncomputable def compute_AD (x : ℝ) : ℝ := x + 1

theorem parallelogram_AD_length :
  ∀ (AB KD : ℝ) (angle_ABK angle_DBK : ℝ) (AD_correct: ℝ),
    AB = 1 → KD = 1 →
    angle_ABK = real.pi / 2 → angle_DBK = real.pi / 6 →
    AD_correct = compute_AD (real.cbrt 2) →
    AD_correct = 1 + real.cbrt 2 :=
by
  intros AB KD angle_ABK angle_DBK AD_correct hAB hKD hABK hDBK hADcorrect
  rw [compute_AD, hABcorrect]
  sorry

end parallelogram_AD_length_l722_722625


namespace fixed_point_of_f_l722_722320

theorem fixed_point_of_f (k a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (λ x, k * x - k - a ^ (x - 1)) 1 = -1 :=
by {
  sorry
}

end fixed_point_of_f_l722_722320


namespace number_of_prime_factors_30_factorial_l722_722976

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722976


namespace probability_of_exactly_two_correct_deliveries_l722_722909

theorem probability_of_exactly_two_correct_deliveries :
  let n := 5
  ∃ derangement_count_of_3 : ℕ,
  derangement_count_of_3 = 2 ∧
  (∀ ways_to_choose_2_exact_deliveries : ℕ, ways_to_choose_2_exact_deliveries = Nat.choose n 2) ∧
  let total_possible_arrangements := Nat.factorial n,
  let probability := (ways_to_choose_2_exact_deliveries * derangement_count_of_3 : ℚ) / total_possible_arrangements,
  probability = 1 / 6 :=
by {
  sorry
}

end probability_of_exactly_two_correct_deliveries_l722_722909


namespace compute_value_l722_722444

theorem compute_value
  (x y z : ℝ)
  (h1 : (xz / (x + y)) + (yx / (y + z)) + (zy / (z + x)) = -9)
  (h2 : (yz / (x + y)) + (zx / (y + z)) + (xy / (z + x)) = 15) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 13.5 :=
by
  sorry

end compute_value_l722_722444


namespace cone_generatrix_length_l722_722053

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722053


namespace sixteen_grams_on_left_pan_l722_722278

theorem sixteen_grams_on_left_pan :
  ∃ (weights : ℕ → ℕ) (pans : ℕ → ℕ) (n : ℕ),
    weights n = 16 ∧
    pans 0 = 11111 ∧
    ∃ k, (∀ i < k, weights i = 2 ^ i) ∧
    (∀ i < k, (pans 1 + weights i = 38) ∧ (pans 0 + 11111 = weights i + skeletal)) ∧
    k = 6 := by
  sorry

end sixteen_grams_on_left_pan_l722_722278


namespace part1_part2_l722_722922

-- Part 1
theorem part1 (x y : ℝ) (z : ℂ) (hx : x > 0) (hy : y < 0) (h1 : x + y = 7) (h2 : x^2 + y^2 = 169) :
  z = 12 - 5 * Complex.i :=
sorry

-- Part 2
theorem part2 (x y : ℝ) (z : ℂ) (hx1 : x > 0) (hy : y < 0) (hx2 : x^2 + y^2 = 6) (hre_nonzero : (Complex.re (z^2 + z)) ≠ 0) :
  (x > 3/2 → (Complex.re (z^2 + z) > 0 ∧ Complex.im (z^2 + z) < 0)) ∧
  (0 < x ∧ x < 3/2 → (Complex.re (z^2 + z) > 0 ∧ Complex.im (z^2 + z) < 0)) :=
sorry

end part1_part2_l722_722922


namespace max_value_two_pow_minus_four_pow_l722_722484

noncomputable def max_value_expression (x : ℝ) : ℝ := (2 : ℝ) ^ x - (4 :ℝ) ^ x 

theorem max_value_two_pow_minus_four_pow : 
  ∃ x : ℝ, (max_value_expression x = 1 / 4) ∧ 
           (∀ y : ℝ, max_value_expression y ≤ 1 / 4) :=
begin
  sorry
end

end max_value_two_pow_minus_four_pow_l722_722484


namespace arc_time_equals_l722_722604

-- Define the length of the minute hand
def minute_hand_length (R : ℝ) := R

-- Define the time taken for the tip of the minute hand to trace its full circumference
def full_circle_time : ℝ := 60

-- Define the circumference of the circle traced by the tip of the minute hand
def circumference (R : ℝ) := 2 * Real.pi * R

-- Define the arc length which is equal to the length of the minute hand
def arc_length (R : ℝ) := R

-- Define the time taken to trace the arc length R
def arc_time (R : ℝ) := (full_circle_time * arc_length R) / circumference R

-- The main theorem to prove the desired time
theorem arc_time_equals (R : ℝ) : arc_time R = 30 / Real.pi :=
by
  -- This is where the proof would go; using 'sorry' to denote omission of proof
  sorry

end arc_time_equals_l722_722604


namespace trig_identity_l722_722503

variable (θ : ℝ)

theorem trig_identity (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.sec θ ^ 2 = 29 / 5 := by
  sorry

end trig_identity_l722_722503


namespace max_min_values_of_x_volume_correctly_set_up_l722_722687

-- Define the parametric equations of the curve
def x (t : ℝ) : ℝ := 2 * (1 + Real.cos t) * Real.cos t
def y (t : ℝ) : ℝ := 2 * (1 + Real.cos t) * Real.sin t

-- Prove the maximum and minimum values of x
theorem max_min_values_of_x :
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi → x t ≤ 4) ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi ∧ x t = 4) ∧
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi → x t ≥ -1/2) ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi ∧ x t = -1/2) := 
  by 
    sorry

-- Define the volume integral
-- NOTE: This would typically involve some integration libraries or further setup in Lean, so we offer a placeholder below.
def volume_integral : ℝ :=
  let integrand (t : ℝ) := 
    let y_sq := (2 * (1 + Real.cos t) * Real.sin t) ^ 2
    let dxdt := -2 * Real.sin t * (1 + 2 * Real.cos t)
    y_sq * dxdt
  π * ∫ (t : ℝ) in 0..2 * Real.pi, integrand t

-- Placeholder to ensure formulation of integral is correct
-- Full proof may require numeric methods or simplifications
theorem volume_correctly_set_up : volume_integral = π * ∫ (t : ℝ) in 0..2 * Real.pi, sorry := 
  by 
    sorry

end max_min_values_of_x_volume_correctly_set_up_l722_722687


namespace prob_A_inter_B_l722_722743

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l722_722743


namespace water_purifier_problem_l722_722405

/-- Proof problem: Cost of Purifiers, Number of Purchasing Plans, Total Filters Given Away. -/
theorem water_purifier_problem :
  (∃ (x : ℕ), 36_000 / x = 2 * (27_000 / (x + 600)) ∧ x = 1_200 ∧ (x + 600) = 1_800) ∧
  let A_cost := 1_200, B_cost := 1_800 in
  ∃ plans : finset (ℕ × ℕ),
    (∀ a b, (a, b) ∈ plans ↔ 60_000 - (A_cost * a) - (B_cost * b) = 0 ∧ b ≤ 8) ∧
    plans.card = 4 ∧
  let profit := 5_250, A_price := 1_350, B_price := 2_100, A_filter := 400, B_filter := 500 in
  ∃ n : ℕ, (∀ (a b : ℕ), (a, b) ∈ finset.filter
    (λ p : ℕ × ℕ, fst p = 47 ∧ snd p = 2 ∨ fst p = 44 ∧ snd p = 4 ∨ fst p = 41 ∧ snd p = 6 ∨ fst p = 38 ∧ snd p = 8)
    plans.to_set →
    ((a * A_filter + b * B_filter) = 6)) :=
begin
  -- proof yet to be provided
  sorry,
end

end water_purifier_problem_l722_722405


namespace eval_sqrt_expression_l722_722877

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722877


namespace nested_radical_solution_l722_722837

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722837


namespace bn_geometric_l722_722001

-- Define the sequences a_n and b_n with the given conditions
def sequence_a : ℕ → ℤ
def sequence_S : ℕ → ℤ
def sequence_b : ℕ → ℤ

-- Conditions
axiom a1_condition : sequence_a 1 = 1
axiom Sn_condition : ∀ n, sequence_S (n + 1) = 4 * sequence_a n + 1
axiom bn_condition : ∀ n, sequence_b n = sequence_a (n + 1) - 2 * sequence_a n

-- Define these sequences according to the given conditions
def a1 := sequence_a 1
def Sn (n : ℕ) := sequence_S n
def bn (n : ℕ) := sequence_b n

-- Theorem statement: sequence b_n is a geometric sequence with the first term 2 and the common ratio 2
theorem bn_geometric : 
  bn 1 = 2 ∧ 
  (∀ n, bn (n + 1) = 2 * bn n) := sorry

end bn_geometric_l722_722001


namespace Raghu_investment_is_2500_l722_722350

noncomputable def RaghuInvestment : ℝ := 
  let R := 7225 / 2.89 in 
  R

theorem Raghu_investment_is_2500:
  RaghuInvestment = 2500 :=
by
  let R : ℝ := 7225 / 2.89
  have h1 : R = 2500 := by
    calc 
      R = 7225 / 2.89 : rfl
      ... = 2500      : by norm_num
  exact h1

end Raghu_investment_is_2500_l722_722350


namespace box_volume_l722_722361

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l722_722361


namespace common_tangents_and_angles_l722_722794

noncomputable def ellipse := {p : ℝ × ℝ | 16 * p.1^2 + 25 * p.2^2 = 400}
noncomputable def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 20}

theorem common_tangents_and_angles :
  (∃ t1 t2 : ℝ → ℝ, (∀ p ∈ ellipse, t1 p.1 = p.2) ∧ (∀ p ∈ circle, t2 p.1 = p.2))
  ∧ (∃ θ : ℝ, ∀ p ∈ (ellipse ∩ circle), 
              tan θ = abs ((∂/∂x 16 * p.1^2 + 25 * p.2^2) * (∂/∂x p.1^2 + p.2^2) +
                          (∂/∂y 16 * p.1^2 + 25 * p.2^2) * (∂/∂y p.1^2 + p.2^2)) /
                          (sqrt ((∂/∂x 16 * p.1^2 + 25 * p.2^2)^2 + (∂/∂y 16 * p.1^2 + 25 * p.2^2)^2) *
                           sqrt ((∂/∂x p.1^2 + p.2^2)^2 + (∂/∂y p.1^2 + p.2^2)^2))) := 
sorry

end common_tangents_and_angles_l722_722794


namespace min_balls_to_draw_l722_722199

theorem min_balls_to_draw (red blue green yellow white black : ℕ) (h_red : red = 35) (h_blue : blue = 25) (h_green : green = 22) (h_yellow : yellow = 18) (h_white : white = 14) (h_black : black = 12) : 
  ∃ n, n = 95 ∧ ∀ (r b g y w bl : ℕ), r ≤ red ∧ b ≤ blue ∧ g ≤ green ∧ y ≤ yellow ∧ w ≤ white ∧ bl ≤ black → (r + b + g + y + w + bl = 95 → r ≥ 18 ∨ b ≥ 18 ∨ g ≥ 18 ∨ y ≥ 18 ∨ w ≥ 18 ∨ bl ≥ 18) :=
by sorry

end min_balls_to_draw_l722_722199


namespace each_child_plays_40_minutes_l722_722201

variable (TotalMinutes : ℕ)
variable (NumChildren : ℕ)
variable (ChildPairs : ℕ)

theorem each_child_plays_40_minutes (h1 : TotalMinutes = 120) 
                                    (h2 : NumChildren = 6) 
                                    (h3 : ChildPairs = 2) :
  (ChildPairs * TotalMinutes) / NumChildren = 40 :=
by
  sorry

end each_child_plays_40_minutes_l722_722201


namespace only_one_function_satisfies_both_conditions_l722_722006

def f1(x : ℝ) : ℝ := x^3
def f2(x : ℝ) : ℝ := abs (x - 1)
def f3(x : ℝ) : ℝ := Real.cos (Real.pi * x)

def condition1(f : ℝ → ℝ) : Prop :=
  ∃ x0 : ℝ, f (-x0) = -f x0

def condition2(f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (1 + x)

theorem only_one_function_satisfies_both_conditions :
  ((condition1 f1 ∧ condition2 f1) ∨ (condition1 f2 ∧ condition2 f2) ∨ (condition1 f3 ∧ condition2 f3)) ∧
  ¬((condition1 f1 ∧ condition2 f1) ∧ (condition1 f2 ∧ condition2 f2)) ∧
  ¬((condition1 f1 ∧ condition2 f1) ∧ (condition1 f3 ∧ condition2 f3)) ∧
  ¬((condition1 f2 ∧ condition2 f2) ∧ (condition1 f3 ∧ condition2 f3)) ∧
  ((condition1 f3 ∧ condition2 f3)) :=
by
  sorry

end only_one_function_satisfies_both_conditions_l722_722006


namespace shorter_leg_of_right_triangle_l722_722586

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l722_722586


namespace Elsa_operations_finite_l722_722829

theorem Elsa_operations_finite :
  ∃ n : ℕ, ∀ (ns : List ℕ), (ns = [1, 2, 3, 4, 5, 6, 7, 8] →
    ∃ k ≤ n, (∀ m, (0 ≤ m ∧ m < k) → ∃ a b : ℕ, a + 2 ≤ b ∧ ns = ns.erase a ++ ns.erase b ++ [a+1, b-1]) ∧
    ¬ ∃ a b : ℕ, a + 2 ≤ b ∧ ns = ns.erase a ++ ns.erase b ++ [a+1, b-1])) :=
begin
  sorry
end

end Elsa_operations_finite_l722_722829


namespace num_ordered_pairs_l722_722325

theorem num_ordered_pairs : 
  {p : ℤ × ℤ // p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3}.to_finset.card = 35 := 
sorry

end num_ordered_pairs_l722_722325


namespace num_pos_nums_with_cube_root_lt_15_l722_722131

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l722_722131


namespace desargues_theorem_converse_desargues_theorem_l722_722383

-- Direct Desargues' Theorem
theorem desargues_theorem (ABC A'B'C' : Triangle) (X Y Z : Point) (δ : Plane) 
  (BC B'C' CA C'A' AB A'B' : Line) :
  BC ∩ B'C' = X →
  CA ∩ C'A' = Y →
  AB ∩ A'B' = Z →
  collinear δ X Y Z →
  (∃ S : Point, AA' ∩ BB' ∩ CC' = S ∨ 
   parallel AA' BB' ∧ parallel BB' CC' ∧ parallel CC' AA') :=
sorry

-- Converse Desargues' Theorem
theorem converse_desargues_theorem (ABC A'B'C' : Triangle) (S : Point) 
  (AA' BB' CC' : Line) :
  (AA' ∩ BB' ∩ CC' = S) →
  ∃ X Y Z : Point,
  (BC ∩ B'C' = X) ∧
  (CA ∩ C'A' = Y) ∧
  (AB ∩ A'B' = Z) ∧
  collinear δ X Y Z :=
sorry

end desargues_theorem_converse_desargues_theorem_l722_722383


namespace shorter_leg_of_right_triangle_l722_722584

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l722_722584


namespace proof_problem_l722_722193

variable (f : ℝ → ℝ) (x m : ℝ)

-- First part of the proof: finding f(x)
def f_satisfies_conditions := (∀ x, f(x + 1) - f(x) = 2 * x) ∧ (f 0 = 1)
def f_expression := ∀ x, f x = x^2 - x + 1

-- Second part of the proof: range of m
def f_inequality_holds := ∀ x, -1 ≤ x ∧ x ≤ 1 → f x > 2 * x + m
def range_of_m := m < -1

theorem proof_problem
  (h1 : f_satisfies_conditions f)
  (h2 : f_inequality_holds f m) :
  f_expression f ∧ range_of_m m :=
by
  sorry

end proof_problem_l722_722193


namespace improper_angle_measurement_l722_722785

-- Define a decagon (regular polygon with 10 sides)
def decagon := 10

-- Sum of interior angles in a regular decagon
def sum_interior_angles (n : ℕ) := 180 * (n - 2)

-- Given sum of interior angles with the construction error
def modified_sum_interior_angles : ℝ := 1470

-- Normal angle in a regular decagon
def normal_angle_in_decagon : ℝ := (sum_interior_angles decagon) / decagon

-- The difference due to the error
def angle_difference : ℝ := modified_sum_interior_angles - sum_interior_angles decagon

-- Measure of the improperly constructed angle
def improper_angle := normal_angle_in_decagon + angle_difference

theorem improper_angle_measurement : improper_angle = 174 := 
by
  sorry

end improper_angle_measurement_l722_722785


namespace non_real_roots_bounded_l722_722159

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722159


namespace magnitude_of_sum_l722_722964

def vector_a := (7 : ℝ, -1 : ℝ, 5 : ℝ)
def vector_b := (-3 : ℝ, 4 : ℝ, 7 : ℝ)
def vector_add (u v : ℝ × ℝ × ℝ) := (u.1 + v.1, u.2 + v.2, u.3 + v.3)
def magnitude (u : ℝ × ℝ × ℝ) := real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

theorem magnitude_of_sum : magnitude (vector_add vector_a vector_b) = 13 := by
  sorry

end magnitude_of_sum_l722_722964


namespace emily_days_off_per_month_l722_722830

theorem emily_days_off_per_month (total_holidays : ℕ) (months_in_year : ℕ) (total_holidays = 24) (months_in_year = 12) :
  (total_holidays / months_in_year) = 2 :=
by
  sorry

end emily_days_off_per_month_l722_722830


namespace frustum_volume_proof_l722_722078

noncomputable def volume_of_frustum
  (r₁ r₂ h : ℝ)
  (V_expected : ℝ) : Prop :=
  r₁ = sqrt 3 ∧ r₂ = 3 * sqrt 3 ∧ h = 6 ∧ V_expected = 78 * Real.pi → 
  (∃ V, V = (1 / 3) * (Real.pi * r₁^2 + Real.sqrt (Real.pi * r₁^2 * Real.pi * r₂^2) + Real.pi * r₂^2) * h ∧ V = V_expected)

theorem frustum_volume_proof 
  (r₁ r₂ h V_expected : ℝ) : 
  volume_of_frustum r₁ r₂ h V_expected :=
by
  sorry

end frustum_volume_proof_l722_722078


namespace simplify_polynomial_l722_722655

theorem simplify_polynomial (s : ℝ) :
  (2 * s ^ 2 + 5 * s - 3) - (2 * s ^ 2 + 9 * s - 6) = -4 * s + 3 :=
by 
  sorry

end simplify_polynomial_l722_722655


namespace bm_eq_cn_l722_722455

noncomputable def midpoint (A B : Point) : Point := sorry

noncomputable def parallel (A B : Line) : Prop := sorry

noncomputable def bisector_angle (A B C : Point) (D : Point) : Prop := sorry

variables {A B C D F M N : Point}

-- Assume triangle ABC
axiom triangle_abc : Triangle A B C

-- Assume D is such that AD is the angle bisector of ∠BAC
axiom bisector_AD : bisector_angle A D B C

-- Assume F is the midpoint of BC
axiom midpoint_F : F = midpoint B C

-- Assume a line MN through F parallel to AD
axiom parallel_MN_AD : parallel (line_through F M) (line_through A D) ∧ parallel (line_through F N) (line_through A D)

-- Intersection points
axiom M_on_AB : on_line M (line_through A B)
axiom N_on_AC : on_line N (line_through A C)

-- Goal to prove BM = CN
theorem bm_eq_cn :
  distance B M = distance C N :=
sorry

end bm_eq_cn_l722_722455


namespace cube_root_numbers_less_than_15_l722_722098

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l722_722098


namespace num_pairs_eq_19_l722_722685

theorem num_pairs_eq_19 :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (x : ℤ × ℤ), x ∈ S ↔ 
  let a := x.fst in
  let b := x.snd in
  a < b ∧ a + b < 100 ∧ (a : ℚ) / 4 + (b : ℚ) / 10 = 7 ) ∧ 
  S.card = 19 :=
sorry

end num_pairs_eq_19_l722_722685


namespace susie_pizza_sales_l722_722666

theorem susie_pizza_sales :
  ∃ x : ℕ, 
    (24 * 3 + 15 * x = 117) ∧ 
    x = 3 := 
by
  sorry

end susie_pizza_sales_l722_722666


namespace archibald_apples_l722_722800

theorem archibald_apples :
  ∀ (a b c d : ℕ), 
    (a = 1 ∧ b = 2 ∧ c = 3 ∧ (a * 7 * b) + (a * 7 * c) + d * 7 = 70 ∧ d * 7 = 14) →
    d = 1 :=
by intros a b c d h
   cases h with h1 h2
   cases h2 with h3 h4
   cases h4 with h5 h6
   cases h6 with h7 h8
   have h9 : d * 14 = 14 := by exact h8
   exact (Nat.eq_of_mul_eq_mul_right (by decide) h9)

end archibald_apples_l722_722800


namespace nested_radical_solution_l722_722839

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722839


namespace move_two_from_lower_to_upper_l722_722456

theorem move_two_from_lower_to_upper :
  let n := 8
  let k := 2
  finset.card (finset.image (λ s : finset ℕ, s) (finset.powerset_len k (finset.range n))) = 28 := by
sorry

end move_two_from_lower_to_upper_l722_722456


namespace length_generatrix_cone_l722_722065

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722065


namespace problem_statement_l722_722822

theorem problem_statement : (515 % 1000) = 515 :=
by
  sorry

end problem_statement_l722_722822


namespace value_of_x_plus_y_l722_722674

-- Define the interior angle of an equilateral triangle
def interior_angle_equilateral_triangle : ℝ := 60

-- Define the sum of angles in a triangle
def sum_of_angles_triangle : ℝ := 180

-- Define the sum of angles on a straight line
def sum_of_straight_line : ℝ := 180

-- Define a right angle
def right_angle : ℝ := 90

-- Define the relevant angles
variable (x y p q : ℝ)

-- Given conditions
axiom equilateral_triangle (x y : ℝ) : 
  x + (120 - x) + 60 = 180 ∧ y + (120 - y) + 60 = 180

axiom square_angles (x y : ℝ) :
  (120 - x) + (120 - y) + 90 = 180

-- The theorem to prove
theorem value_of_x_plus_y (x y : ℝ) (h1 : equilateral_triangle x y) (h2 : square_angles x y) :
  x + y = 150 := sorry

end value_of_x_plus_y_l722_722674


namespace proof_system_l722_722727

theorem proof_system (x y : ℝ) (hx : x = 1/2) (hy : y = 1/3) :
    (1 / 4^x + 1 / 27^y = 5 / 6) ∧ 
    (Real.log y / Real.log 27 - Real.log x / Real.log 4 >= 1 / 6) ∧
    (27^y - 4^x <= 1)  :=
by
  sorry

end proof_system_l722_722727


namespace expression_simplifies_to_49_l722_722137

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l722_722137


namespace integer_solutions_l722_722551

theorem integer_solutions (x : ℤ) :
  (x - 3) ^ (30 - 2 * x ^ 2) = 1 ↔
  (x = 4 ∨ x = 2) :=
by
  sorry

end integer_solutions_l722_722551


namespace hyperbola_eccentricity_l722_722198

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h_asymptote_slope : b / a = real.sqrt 3) :
  real.sqrt (1 + (b^2) / (a^2)) = 2 :=
by
  sorry

end hyperbola_eccentricity_l722_722198


namespace solution_l722_722659

noncomputable def problem : Prop :=
  ∃ (x y : ℝ), (8^x / 4^(x + y) = 64) ∧ (27^(x + y) / 9^(6 * y) = 81) ∧ (x * y = 644 / 9)

theorem solution : problem := sorry

end solution_l722_722659


namespace club_truncator_probability_l722_722816

theorem club_truncator_probability :
  let p_win := (1 : ℚ) / 3
  let p_loss := (1 : ℚ) / 3
  let p_tie := (1 : ℚ) / 3
  TotalOutcomes := 3^6
  FavorableOutcomes_W_eq_L :=
    20 + (6.choose 2 * (4.choose 2))  + (6.choose 1 * 5.choose 4) + 1
  Probability_W_eq_L := FavorableOutcomes_W_eq_L / TotalOutcomes
  Probability_W_neq_L := 1 - Probability_W_eq_L
  Probability_W_gt_L := Probability_W_neq_L / 2
  prob := Probability_W_gt_L
  fraction := prob.num / prob.denom,
  RelPrime := fraction.num.gcd fraction.denom = 1,

  (fraction.num + fraction.denom) = 341
:=
sorry

end club_truncator_probability_l722_722816


namespace ava_tiffany_knockout_tournament_probability_l722_722806

theorem ava_tiffany_knockout_tournament_probability :
  ∃ (a b : ℕ), Nat.coprime a b ∧ 100 * a + b = 3596 ∧ (a, b) = (31, 496) := 
by 
  have num_matches : ℕ := 32 - 1
  have total_pairs : ℕ := Nat.choose 32 2
  have prob_simplified : ℕ × ℕ := (31, 496)
  have h_coprime : Nat.coprime 31 496 := by sorry
  use (31, 496)
  finish

end ava_tiffany_knockout_tournament_probability_l722_722806


namespace find_two_points_distance_d_l722_722304

theorem find_two_points_distance_d (sets : Set (Set ℝ)) (h_disjoint : ∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅)
  (h_partition : ⋃₀ sets = Set.univ) (d : ℝ) (h_d_pos : d > 0) :
  ∃ s ∈ sets, ∃ x y ∈ s, x ≠ y ∧ dist x y = d :=
by
  sorry

end find_two_points_distance_d_l722_722304


namespace find_z2_l722_722945

open Complex

theorem find_z2
  (z1 z2 : ℂ)
  (h1 : (z1 - 2) * (1 + I) = 1 - I)
  (h2 : z2.im = 2)
  (h3 : isReal(z1 * z2)) :
  z2 = 4 + 2 * I := 
  sorry

end find_z2_l722_722945


namespace smallest_positive_x_satisfying_equation_l722_722716

theorem smallest_positive_x_satisfying_equation :
  ∃ x : ℝ, 0 < x ∧ sqrt (3 * x) = 5 * x - 2 ∧ x = 4 / 25 :=
by
  sorry

end smallest_positive_x_satisfying_equation_l722_722716


namespace first_day_exceeds_200_l722_722571

def bacteria_count (n : ℕ) : ℕ := 4 * 3^n

def exceeds_200 (n : ℕ) : Prop := bacteria_count n > 200

theorem first_day_exceeds_200 : ∃ n, exceeds_200 n ∧ ∀ m < n, ¬ exceeds_200 m :=
by sorry

end first_day_exceeds_200_l722_722571


namespace convex_polygon_sides_in_arithmetic_progression_l722_722682

theorem convex_polygon_sides_in_arithmetic_progression
  (n : ℕ)
  (angles : Fin n → ℝ)
  (h_arith_prog : ∀ (i : Fin (n - 1)), angles i.succ = angles i + 10)
  (h_smallest_angle : angles 0 = 120)
  (h_largest_angle : angles (Fin.last n) = 150)
  (h_convex : ∀ i, 0 < angles i ∧ angles i < 180)
  (h_sum_angles : ∑ i, angles i = 180 * (n - 2)) :
  n = 8 :=
  sorry

end convex_polygon_sides_in_arithmetic_progression_l722_722682


namespace oarsmen_count_l722_722673

theorem oarsmen_count 
  (n : ℕ) 
  (avg_weight_increase : ℝ) 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (avg_weight_increase = 1.8) 
  (weight_before = 53)
  (weight_after = 71)
  (weight_diff : ℝ := weight_after - weight_before) 
  (weight_diff = 18) 
  : n = 10 := 
sorry

end oarsmen_count_l722_722673


namespace math_problem_l722_722227

open Nat

-- Definitions of the sequences and conditions
variable (a : ℕ+ → ℝ) (q : ℝ) (b : ℕ+ → ℝ)
variables (h_geo : ∀ n : ℕ+, a (n + 1) = q * a n)
variable (h_a1 : a 1 > 1) (h_q_pos : q > 0)

-- Definitions of log-based sequence
variable (h_b : ∀ n : ℕ+, b n = Real.log 2 (a n))
variable (h_sum : b 1 + b 3 + b 5 = 6)
variable (h_prod : b 1 * b 3 * b 5 = 0)

-- Statement to be proved
theorem math_problem :
  (∀ n : ℕ+, b n = b 1 + (n - 1) * (log 2 q))
  ∧ ((∑ k in range (n : ℕ), b ⟨k+1, sorry⟩) = (9 * n - n ^ 2) / 2)
  ∧ (∀ n : ℕ+, a n = 2 ^ (5 - n)) :=
sorry

end math_problem_l722_722227


namespace area_of_tangent_triangle_l722_722313

noncomputable def tangentTriangleArea : ℝ :=
  let y := λ x : ℝ => x^3 + x
  let dy := λ x : ℝ => 3 * x^2 + 1
  let slope := dy 1
  let y_intercept := 2 - slope * 1
  let x_intercept := - y_intercept / slope
  let base := x_intercept
  let height := - y_intercept
  0.5 * base * height

theorem area_of_tangent_triangle :
  tangentTriangleArea = 1 / 2 :=
by
  sorry

end area_of_tangent_triangle_l722_722313


namespace problem_solution_l722_722458

def at (a b : ℕ) : ℤ := a * b - b ^ 3
def hash (a b : ℕ) : ℤ := a + b - a * b ^ 2

theorem problem_solution :
  (3@2) / (3#2) = 2 / 7 :=
by
  have at_def : at 3 2 = 3 * 2 - 2 ^ 3 := rfl
  have hash_def : hash 3 2 = 3 + 2 - 3 * 2 ^ 2 := rfl
  have at_result : at 3 2 = -2 := by
    rw at_def
    norm_num
  have hash_result : hash 3 2 = -7 := by
    rw hash_def
    norm_num
  rw [at_result, hash_result]
  norm_num
  sorry

end problem_solution_l722_722458


namespace find_value_of_expression_l722_722717

theorem find_value_of_expression :
  3 - (-3) ^ (-3 : ℤ) = 82 / 27 := by
sorry

end find_value_of_expression_l722_722717


namespace silk_diameter_scientific_notation_l722_722654

-- Definition of the given condition
def silk_diameter := 0.000014 

-- The goal to be proved
theorem silk_diameter_scientific_notation : silk_diameter = 1.4 * 10^(-5) := 
by 
  sorry

end silk_diameter_scientific_notation_l722_722654


namespace keiths_total_spending_l722_722244

theorem keiths_total_spending :
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  total_cost = 77.05 :=
by
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  have h : total_cost = 77.05 := sorry
  exact h

end keiths_total_spending_l722_722244


namespace log_eq_solution_l722_722299

theorem log_eq_solution (x : ℝ) (h : log 8 x + log 4 (x^3) = 9) : x = 2^(54/5) :=
by
  sorry

end log_eq_solution_l722_722299


namespace probability_third_draw_first_class_expected_value_first_class_in_10_draws_l722_722569

-- Define the problem with products
structure Products where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

-- Given products configuration
def products : Products := { total := 5, first_class := 3, second_class := 2 }

-- Probability calculation without replacement
-- Define the event of drawing
def draw_without_replacement (p : Products) (draws : ℕ) (desired_event : ℕ -> Bool) : ℚ := 
  if draws = 3 ∧ desired_event 3 ∧ ¬ desired_event 1 ∧ ¬ desired_event 2 then
    (2 / 5) * ((1 : ℚ) / 4) * (3 / 3)
  else 
    0

-- Define desired_event for the specific problem
def desired_event (n : ℕ) : Bool := 
  match n with
  | 3 => true
  | _ => false

-- The first problem's proof statement
theorem probability_third_draw_first_class : draw_without_replacement products 3 desired_event = 1 / 10 := sorry

-- Expected value calculation with replacement
-- Binomial distribution to find expected value
def expected_value_with_replacement (p : Products) (draws : ℕ) : ℚ :=
  draws * (p.first_class / p.total)

-- The second problem's proof statement
theorem expected_value_first_class_in_10_draws : expected_value_with_replacement products 10 = 6 := sorry

end probability_third_draw_first_class_expected_value_first_class_in_10_draws_l722_722569


namespace number_of_distinct_prime_factors_30_fact_l722_722992

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722992


namespace line_equation_l722_722779

theorem line_equation (a : ℝ) (P : ℝ × ℝ) (hx : P = (5, 6)) 
                      (cond : (a ≠ 0) ∧ (2 * a = 17)) : 
  ∃ (m b : ℝ), - (m * (0 : ℝ) + b) = a ∧ (- m * 17 / 2 + b) = 6 ∧ 
               (x + 2 * y - 17 =  0) := sorry

end line_equation_l722_722779


namespace sum_x_coordinates_g_eq_2_l722_722453

noncomputable def g (x: ℝ) : ℝ :=
  if x ∈ Set.Icc (-2 : ℝ) 0 then 3 * x - 3
  else if x ∈ Set.Icc 0 1 then -2 * x
  else if x ∈ Set.Icc 1 3 then 4 * x - 2
  else 0

theorem sum_x_coordinates_g_eq_2 : 
  (Finset.sum (Finset.filter (λ x => g x = 2) (Finset.Icc (-2 : ℝ) 3)) (λ x => x)) = 1 := 
sorry

end sum_x_coordinates_g_eq_2_l722_722453


namespace door_height_is_eight_l722_722212

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l722_722212


namespace ratio_truncated_pyramid_surface_area_l722_722681

theorem ratio_truncated_pyramid_surface_area (r α : ℝ) (r_pos : 0 < r)
  (α_pos : 0 < α ∧ α < π / 2) :
  let a := 2 * sqrt 3 * r * Real.cot (α / 2)
  let b := 2 * sqrt 3 * r * Real.tan (α / 2)
  let S_triangle_ABC := 3 * sqrt 3 * r^2 * (Real.cot (α / 2))^2
  let S_triangle_A1B1C1 := 3 * sqrt 3 * r^2 * (Real.tan (α / 2))^2
  let K1K := 2 * r / Real.sin α
  let S_lateral := 12 * sqrt 3 * r^2 / (Real.sin α)^2
  let S_total := 6 * sqrt 3 * r^2 * (4 - (Real.sin α)^2) / (Real.sin α)^2
  let S_sphere := 4 * Real.pi * r^2
  S_total / S_sphere = 3 * sqrt 3 / (2 * Real.pi) * (4 * (Real.cot α)^2 + 3) :=
sorry

end ratio_truncated_pyramid_surface_area_l722_722681


namespace total_books_proof_l722_722693

noncomputable def total_books (T : ℕ) :=
  let S := 0.3 * T in
  let Sb := 0.5 * T in
  S = Sb - 600

theorem total_books_proof : ∃ T : ℕ, total_books T = 6000 :=
by
  use 6000
  unfold total_books
  norm_num
  sorry

end total_books_proof_l722_722693


namespace carol_should_choose_optimal_interval_l722_722438

noncomputable def maximize_winning_probability : Prop :=
  ∀ (a b c : ℝ), 
    (a ∈ set.Icc 0 1) → 
    (b ∈ set.Icc (1/3) (2/3)) → 
    (c ∈ set.Icc (1/3) (2/3)) → 
    (c > a ∧ c < b) ∨ (c < a ∧ c > b) → 
    ∃ (x y : ℝ), 
      (x = 1/3) ∧ 
      (y = 2/3)

theorem carol_should_choose_optimal_interval : maximize_winning_probability := 
  sorry

end carol_should_choose_optimal_interval_l722_722438


namespace parametric_equation_of_line_passing_through_M_l722_722415

theorem parametric_equation_of_line_passing_through_M (
  t : ℝ
) : 
    ∃ x y : ℝ, 
      x = 1 + (t * (Real.cos (Real.pi / 3))) ∧ 
      y = 5 + (t * (Real.sin (Real.pi / 3))) ∧ 
      x = 1 + (1/2) * t ∧ 
      y = 5 + (Real.sqrt 3 / 2) * t := 
by
  sorry

end parametric_equation_of_line_passing_through_M_l722_722415


namespace find_f_neg1_l722_722565

theorem find_f_neg1 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 := 
by 
  -- skipping the proof: 
  sorry

end find_f_neg1_l722_722565


namespace value_of_x_l722_722134

theorem value_of_x (x : ℝ) (h : 1 ∈ {x, x^2}) : x = -1 :=
sorry

end value_of_x_l722_722134


namespace not_proportional_x2_y2_l722_722825

def directly_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x = k * y

def inversely_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x * y = k

theorem not_proportional_x2_y2 (x y : ℝ) :
  x^2 + y^2 = 16 → ¬directly_proportional x y ∧ ¬inversely_proportional x y :=
by
  sorry

end not_proportional_x2_y2_l722_722825


namespace power_combination_l722_722807

theorem power_combination :
  (-1)^43 + 2^(2^3 + 5^2 - 7^2) = -65535 / 65536 :=
by
  sorry

end power_combination_l722_722807


namespace other_root_of_quadratic_l722_722643

theorem other_root_of_quadratic (z z1 z2 : ℂ) (h_eq: z ^ 2 = -100 + 75 * complex.I) (h_root1 : z1 = 5 + 10 * complex.I) :
  z1 + z2 = 0 ∧ z2 = -5 - 10 * complex.I ∧ z = z1 ∨ z = z2 :=
by
  sorry

end other_root_of_quadratic_l722_722643


namespace max_elements_subset_l722_722251

open Function

def is_subset (S : Finset ℕ) (T : Finset ℕ) : Prop :=
  ∀ x ∈ S, x ∈ T

def sum_divisible_by_5 (x y : ℕ) : Prop :=
  (x + y) % 5 = 0

def no_pair_sum_div_by_5 (S : Finset ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → ¬ sum_divisible_by_5 x y

theorem max_elements_subset : 
  ∃ (S : Finset ℕ), 
    is_subset S (Finset.range 101) ∧ no_pair_sum_div_by_5 S ∧ S.card = 40 :=
sorry

end max_elements_subset_l722_722251


namespace shorter_leg_of_right_triangle_l722_722574

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l722_722574


namespace f_f_4_l722_722913

def f (x : ℝ) : ℝ :=
  if abs x ≤ 1 then sqrt x else 1 / x

theorem f_f_4 : f (f 4) = 1 / 2 := 
by
  sorry

end f_f_4_l722_722913


namespace binomial_sum_mod_prime_squared_l722_722391

theorem binomial_sum_mod_prime_squared (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  (∑ j in Finset.range (p + 1), Nat.choose p j * Nat.choose (p + j) j) ≡ 2^p + 1 [MOD p^2] := 
by
  sorry

end binomial_sum_mod_prime_squared_l722_722391


namespace solve_log_eq_l722_722297

theorem solve_log_eq (x : ℝ) (h : log 8 x + log 4 (x ^ 3) = 9) : 
  x = 2^(54 / 11) :=
by 
  sorry

end solve_log_eq_l722_722297


namespace door_height_eight_l722_722218

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l722_722218


namespace nested_sqrt_eq_l722_722873

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722873


namespace disk_areas_sum_eq_a_plus_b_plus_c_eq_l722_722294

noncomputable def sum_of_disk_areas : ℝ :=
  let r := Real.tan (11.25 * Real.pi / 180) in
  let area_one_disk := π * r^2 in
  16 * area_one_disk

theorem disk_areas_sum_eq (a b c : ℕ) (h : a = 6336) (hb : b = 0) (hc : c = 1) :
  sum_of_disk_areas = π * (a - b * Real.sqrt c) :=
by { sorry }

theorem a_plus_b_plus_c_eq :
  let a_b_c := 6336 + 0 + 1 in
  a_b_c = 6337 :=
by { sorry }

end disk_areas_sum_eq_a_plus_b_plus_c_eq_l722_722294


namespace simplify_expression_l722_722292

noncomputable def expr_simplify : Prop :=
  ( ( (\sqrt{5} - 2)^(2 - \sqrt{6}) ) / (\sqrt{5} + 2)^(2 + \sqrt{6}) ) = 1 / ( (9 - 4*\sqrt{5})^\sqrt{6}
  )

theorem simplify_expression : expr_simplify :=
  sorry

end simplify_expression_l722_722292


namespace exists_f_such_that_ff_eq_p_add_2_no_f_such_that_ff_eq_q_add_2_l722_722246

def p : ℕ → ℕ
| 1     := 2
| 2     := 3
| 3     := 4
| 4     := 1
| (n+5) := n+5

def q : ℕ → ℕ
| 1     := 3
| 2     := 4
| 3     := 2
| 4     := 1
| (n+5) := n+5

-- Part 1: Prove the existence of f such that f(f(n)) = p(n) + 2
theorem exists_f_such_that_ff_eq_p_add_2 : ∃ f : ℕ → ℕ, ∀ n : ℕ, f(f(n)) = p(n) + 2 := 
sorry

-- Part 2: Prove that no f exists such that f(f(n)) = q(n) + 2
theorem no_f_such_that_ff_eq_q_add_2 : ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f(f(n)) = q(n) + 2 :=
sorry

end exists_f_such_that_ff_eq_p_add_2_no_f_such_that_ff_eq_q_add_2_l722_722246


namespace sequence_decomposition_l722_722958

variable (a : ℕ)

def sequence (n : ℕ) : ℚ := n / (n + a)

theorem sequence_decomposition 
  (n : ℕ) 
  (h : 0 < a) : 
  ∃ u v : ℕ, sequence a n = sequence a u * sequence a v :=
sorry

end sequence_decomposition_l722_722958


namespace sqrt_continued_fraction_l722_722865

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722865


namespace number_of_prime_factors_30_factorial_l722_722977

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722977


namespace door_height_l722_722215

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l722_722215


namespace number_of_distinct_prime_factors_30_fact_l722_722989

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722989


namespace buttons_per_shirt_l722_722237

def shirts_per_kid : ℕ := 3
def kids : ℕ := 3
def total_buttons : ℕ := 63

theorem buttons_per_shirt : (shirts_per_kid * kids) * b = total_buttons → b = 7 :=
by
  assume h : (shirts_per_kid * kids) * b = total_buttons
  -- To be proven
  sorry

end buttons_per_shirt_l722_722237


namespace tiffany_cans_l722_722342

variable {M : ℕ}

theorem tiffany_cans : (M + 12 = 2 * M) → (M = 12) :=
by
  intro h
  sorry

end tiffany_cans_l722_722342


namespace part_a_part_b_part_c_l722_722802

open ProbabilityTheory

namespace Problem

-- Definitions of conditions
variable (B G : Event) (p : ℙ (B ∪ G) = 1 / 2) (h : ℙ B = ℙ G)

-- Part (a)
theorem part_a (hb : ℙ (B ∩ G)) : ℙ (B ∩ G) = 1 / 2 := by
  sorry 

-- Additional condition for part (b)
variable (OneBoy : Event) (B1 : ℙ OneBoy = ℙ (B ∩ OneBoy))

-- Part (b)
theorem part_b (hb1 : ℙ (B ∩ G ∩ OneBoy)) : ℙ (B ∩ G ∩ OneBoy) = 2 / 3 := by
  sorry

-- Additional condition for part (c)
variable (BoyMonday : Event) (Bm : ℙ BoyMonday = ℙ (B ∩ BoyMonday))

-- Part (c)
theorem part_c (hbm : ℙ (B ∩ G ∩ BoyMonday)) : ℙ (B ∩ G ∩ BoyMonday) = 14 / 27 := by
  sorry

end Problem

end part_a_part_b_part_c_l722_722802


namespace partition_arithmetic_sequence_l722_722234

theorem partition_arithmetic_sequence (S : Set ℕ) (A : Set ℕ) (B : Set ℕ) (h_partition : (A ∪ B = S) ∧ (∀ n ∈ A, n ∉ B) ∧ (∀ n ∈ B, n ∉ A))
  (h_infinite_arithmetic : ∀ (d > 0) (a : ℕ), ¬(∀ n : ℕ, a + n * d ∈ A) ∧ ¬(∀ n : ℕ, a + n * d ∈ B))
  (h_bounded_difference : ∀ (C : Set ℕ), (C = A ∨ C = B) → ∃ N, ∀ m ∈ C, ∃ n ∈ C, n > m ∧ n - m ≤ N) : False :=
by
  sorry

end partition_arithmetic_sequence_l722_722234


namespace eccentricity_of_ellipse_l722_722016

open Real

theorem eccentricity_of_ellipse 
  (O B F : ℝ × ℝ)
  (a b : ℝ) 
  (h_a_gt_b: a > b)
  (h_b_gt_0: b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_OB_eq_OF : dist O B = dist O F)
  (O_is_origin : O = (0,0))
  (B_is_upper_vertex : B = (0, b))
  (F_is_right_focus : F = (c, 0) ∧ c = Real.sqrt (a^2 - b^2)) :
 (c / a = sqrt 2 / 2)
:=
sorry

end eccentricity_of_ellipse_l722_722016


namespace probability_intersection_l722_722748

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l722_722748


namespace ellipse_equation_proof_triangle_area_proof_l722_722516

-- Definitions based on the given conditions
def is_ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1

def is_eccentricity (a c : ℝ) : Prop :=
  c = (2 * real.sqrt 2) / 3 * a

def is_perimeter (a c perimeter : ℝ) : Prop :=
  2 * a + 2 * c = perimeter

-- Main theorem statements
theorem ellipse_equation_proof (a b x y c : ℝ)
    (h1 : is_ellipse a b x y)
    (h2 : is_eccentricity a c)
    (h3 : is_perimeter a c (6 + 4 * real.sqrt 2)) :
    a = 3 ∧ c = 2 * real.sqrt 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 9) + (y^2 / 1) = 1) :=
  sorry

theorem triangle_area_proof (a b x y c k m : ℝ)
    (h1 : is_ellipse a b x y)
    (h2 : is_eccentricity a c)
    (h3 : is_perimeter a c (6 + 4 * real.sqrt 2))
    (h4 : ∀ A B : (ℝ × ℝ), (A = (k * B.2 + m, B.2)) ∧ (A = (x, y)) ∧ (B = (x, y)) ∧
    (intersection_with_circle A B c)) :
    ∃ t : ℝ, (0 < t ∧ t ≤ 1/9) ∧ (max_triangle_area t = 3/8) :=
  sorry

-- Definitions for intersection with the circle passing through the mentioned points
def intersection_with_circle (A B c : ℝ × ℝ) : Prop := sorry
def max_triangle_area (t : ℝ) : ℝ := 9 / 5 * (real.sqrt ((-144 / 25 * t^2) + t))

end ellipse_equation_proof_triangle_area_proof_l722_722516


namespace time_to_pay_back_l722_722619

def total_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

def monthly_profit := monthly_revenue - monthly_expenses

theorem time_to_pay_back : 
  (total_cost / monthly_profit) = 10 := 
by
  -- Definition of monthly_profit 
  have monthly_profit_def : monthly_profit = 4000 - 1500 := rfl
  rw [monthly_profit_def]
  
  -- Performing the division
  show (25000 / 2500) = 10
  apply div_eq_of_eq_mul
  norm_num
  sorry

end time_to_pay_back_l722_722619


namespace coordinates_A_B_l722_722284

theorem coordinates_A_B : 
  (∃ x, 7 * x + 2 * 3 = 41) ∧ (∃ y, 7 * (-5) + 2 * y = 41) → 
  ((∃ x, x = 5) ∧ (∃ y, y = 38)) :=
by
  sorry

end coordinates_A_B_l722_722284


namespace correct_calculation_l722_722372

theorem correct_calculation (a b x y : ℝ) :
  (7 * a^2 * b - 7 * b * a^2 = 0) ∧ 
  (¬ (6 * a + 4 * b = 10 * a * b)) ∧ 
  (¬ (7 * x^2 * y - 3 * x^2 * y = 4 * x^4 * y^2)) ∧ 
  (¬ (8 * x^2 + 8 * x^2 = 16 * x^4)) :=
sorry

end correct_calculation_l722_722372


namespace sum_of_solutions_l722_722697

theorem sum_of_solutions (x : ℕ) (h : 9 - 2 * x ≥ 0) : Σ' x, 9 - 2 * x ≥ 0 → Σ (i : ℕ) (h₁ : i ∈ [1, 2, 3, 4]),  i = 10 :=
by
  sorry

end sum_of_solutions_l722_722697


namespace quadratic_non_real_roots_b_values_l722_722175

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722175


namespace angle_between_vectors_l722_722965

variables {α : Type*} [inner_product_space ℝ α] 
variables (a b : α)

-- Given conditions
def condition_a := ∥a∥ = 1
def condition_b := ∥b∥ = 2
def condition_dot := inner a (a + b) = 0

-- The conclusion: the angle between a and b
theorem angle_between_vectors (h1 : condition_a a) (h2 : condition_b b) (h3 : condition_dot a b) : real.angle (inner a b / (∥a∥ * ∥b∥)) = π * (2 / 3) :=
sorry

end angle_between_vectors_l722_722965


namespace cone_generatrix_length_l722_722055

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722055


namespace baseball_cards_l722_722820

theorem baseball_cards (cards_per_page new_cards pages : ℕ) (h1 : cards_per_page = 8) (h2 : new_cards = 3) (h3 : pages = 2) : 
  (pages * cards_per_page - new_cards = 13) := by
  sorry

end baseball_cards_l722_722820


namespace exponential_satisfies_property_l722_722439

def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a^x

theorem exponential_satisfies_property (f : ℝ → ℝ) (h : is_exponential f) : 
  ∀ x y : ℝ, x > 0 → y > 0 → f(x + y) = f(x) * f(y) :=
by
  sorry

end exponential_satisfies_property_l722_722439


namespace find_2a_plus_b_l722_722947

theorem find_2a_plus_b (a b : ℝ) (h1 : 3 * a + 2 * b = 18) (h2 : 5 * a + 4 * b = 31) :
  2 * a + b = 11.5 :=
sorry

end find_2a_plus_b_l722_722947


namespace shorter_leg_in_right_triangle_l722_722576

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l722_722576


namespace incorrect_about_infinite_population_simple_random_sampling_l722_722374

-- Definitions based on the conditions
def randomly_selected (population : Type) := ∀ (x : population), ∃ (y : population), y ≠ x
def equally_likely (population : Type) := ∀ (x y : population), (x ≠ y → y ≠ x)
def sampling_with_or_without_replacement (population : Type) := ∀ (x : population), ∃ (sampled : population), sampled = x ∨ sampled ≠ x

-- Simple random sampling requires the population to be finite
theorem incorrect_about_infinite_population_simple_random_sampling (population : Type)
  (h_randomly_selected : randomly_selected population)
  (h_equally_likely : equally_likely population)
  (h_sampling_with_or_without_replacement : sampling_with_or_without_replacement population) :
  ¬ infinite population :=
sorry

end incorrect_about_infinite_population_simple_random_sampling_l722_722374


namespace monotonic_increasing_interval_l722_722683

def f (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_increasing_interval : ∀ x : ℝ, x > 0 → (0 < (Real.exp x - 1)) :=
by sorry

end monotonic_increasing_interval_l722_722683


namespace right_triangle_side_lengths_l722_722330

noncomputable def pythagorean_triple (m n : ℕ) : Prop :=
  coprime m n ∧ (m % 2 ≠ n % 2) ∧ m > n

theorem right_triangle_side_lengths :
  ∃ (a b c : ℕ), ∃ (m n : ℕ),
    pythagorean_triple m n ∧
    a = m^2 - n^2 ∧
    b = 2 * m * n ∧
    c = m^2 + n^2 ∧
    (a + b - c) / 2 = 420 ∧
    a = 399 ∧ b = 40 ∧ c = 401
:= by
  sorry

end right_triangle_side_lengths_l722_722330


namespace box_volume_l722_722362

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l722_722362


namespace nested_sqrt_eq_l722_722868

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722868


namespace shorter_leg_of_right_triangle_l722_722573

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l722_722573


namespace volume_of_tetrahedron_A_BCD_equiv_l722_722230

-- Define the tetrahedron and the points on its edges
variable {A B C D P Q : Type} 
variable (V_A_BPQ V_B_CPQ V_C_DPQ V_A_BCD : ℝ)
variable (point_on_AC : P ∈ segment A C)
variable (point_on_BD : Q ∈ segment B D)
variable (lines_connected : AQ ∧ CQ ∧ BP ∧ DP ∧ PQ)

-- Define the volumes given in the problem
def volume_A_BPQ : ℝ := 6
def volume_B_CPQ : ℝ := 2
def volume_C_DPQ : ℝ := 8

-- Define the Lean theorem statement
theorem volume_of_tetrahedron_A_BCD_equiv :
  volume_A_BPQ + volume_B_CPQ + volume_C_DPQ + (V_A_BCD - (volume_A_BPQ + volume_B_CPQ + volume_C_DPQ)) = 32 :=
sorry

end volume_of_tetrahedron_A_BCD_equiv_l722_722230


namespace shortest_elevator_path_l722_722677

def elevator_path (e : String) := e = "B" ∨ e = "J" ∨ e = "G"

theorem shortest_elevator_path : 
  (∃ path : List String, path = ["Entrance", "B", "J", "G", "Exit"] ∧ 
   ∀ e ∈ path.tail, elevator_path e) :=
begin
  use ["Entrance", "B", "J", "G", "Exit"],
  split,
  { refl },
  { intros e he,
    simp at he,
    cases he,
    { subst he, left, refl },
    cases he,
    { subst he, right, left, refl },
    cases he,
    { subst he, right, right, left, refl },
    { contradiction } }
end

end shortest_elevator_path_l722_722677


namespace probability_intersection_l722_722745

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l722_722745


namespace paint_needed_for_800_statues_l722_722558

-- Definition of conditions
def paint_needed_for_statue : ℕ := 2
def height_of_original_statue_ft : ℕ := 8
def number_of_statues : ℕ := 800
def height_of_small_statue_ft : ℕ := 2
def similarity_condition : Prop := true -- All statues are similar in shape
def thickness_condition : Prop := true -- All statues need to be painted to the same thickness

-- Statement to prove
theorem paint_needed_for_800_statues :
  similarity_condition → thickness_condition →
  (paint_needed_for_statue * number_of_statues * (height_of_small_statue_ft / height_of_original_statue_ft)^2) = 100 :=
by
  exact (paint_needed_for_statue * number_of_statues * (height_of_small_statue_ft / height_of_original_statue_ft)^2) = 100

end paint_needed_for_800_statues_l722_722558


namespace book_pages_and_average_l722_722399

theorem book_pages_and_average (pages_ch1 pages_ch2 pages_ch3 pages_ch4 total_pages : ℕ)
    (h1 : pages_ch1 = 60)
    (h2 : pages_ch2 = 75)
    (h3 : pages_ch3 = 56)
    (h4 : pages_ch4 = 42)
    (h_total : total_pages = 325) : 
    let pages_ch5 := total_pages - (pages_ch1 + pages_ch2 + pages_ch3 + pages_ch4) in
    pages_ch5 = 92 ∧ (total_pages / 5) = 65 :=
by
  sorry

end book_pages_and_average_l722_722399


namespace max_median_soda_l722_722672

theorem max_median_soda (cans customers: ℕ) (h1 : cans = 300) (h2 : customers = 120) (h3 : ∀ x, x < customers → 2 ≤ (soda_per_customer x)) : 
  ∃ m, m ≤ 3.0 ∧ (∀ y, y ≠ 60 ∧ y ≠ 61 → soda_per_customer y ≤ m) :=
by
  sorry

-- Define a function representing the number of sodas per customer
def soda_per_customer :ℕ → ℕ
| x := -- this part would be defined by the distribution logic of the problem but is not required for the theorem statement

end max_median_soda_l722_722672


namespace probability_intersection_l722_722751

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l722_722751


namespace man_speed_l722_722417

/-- Let a man crosses a bridge in 15 minutes and the length of the bridge is 2500 meters.
    Prove that the man's speed is 10 km/hr. -/
theorem man_speed (t_min : ℝ) (d_meters : ℝ) (h1 : t_min = 15) (h2 : d_meters = 2500) : 
  let t_hours := t_min / 60,
      d_km := d_meters / 1000 in
  (d_km / t_hours) = 10 :=
by
  -- Introduce the parameters and assumptions
  sorry

end man_speed_l722_722417


namespace number_of_distinct_prime_factors_30_fact_l722_722990

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722990


namespace non_real_roots_interval_l722_722189

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722189


namespace glucose_solution_volume_l722_722776

theorem glucose_solution_volume (gram_per_cc : ℚ) (volume_cc : ℚ) (grams_in_45cc : ℚ) : 
  grams_in_45cc / volume_cc = 10 / 100 :=
by
-- conditions
let gram_per_cc := 4.5
let volume_cc := 45
let grams_in_45cc := 4.5
-- setup proportion
have proportion : gram_per_cc / volume_cc = 10 / 100,
sorry
-- conclude solution
exact proportion

end glucose_solution_volume_l722_722776


namespace length_generatrix_cone_l722_722066

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722066


namespace average_speed_to_work_l722_722793

theorem average_speed_to_work
  (distance : ℝ)
  (speed_home : ℝ)
  (total_commute_time : ℝ)
  (distance_eq : distance = 18)
  (speed_home_eq : speed_home = 30)
  (total_commute_time_eq : total_commute_time = 1) :
  ∃ (v : ℝ), (v ≠ 0 ∧ (distance / v + distance / speed_home = total_commute_time)) ∧ (v = 45) :=
by {
  use 45,
  split, {
    split,
    { norm_num },
    linarith [distance_eq, speed_home_eq, total_commute_time_eq],
  },
  norm_num,
}

end average_speed_to_work_l722_722793


namespace product_expression_evaluates_to_32_l722_722813

theorem product_expression_evaluates_to_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  -- The proof itself is not required, hence we can put sorry here
  sorry

end product_expression_evaluates_to_32_l722_722813


namespace max_marked_cells_knight_moves_l722_722353

open Classical

theorem max_marked_cells_knight_moves :
  ∃ S : Finset (Fin 8 × Fin 8), 
    S.card = 8 ∧ 
    (∀ (c1 c2 : Fin 8 × Fin 8), c1 ∈ S → c2 ∈ S → c1 ≠ c2 → 
      ∃ (a b : Fin 8 × Fin 8), (a ∈ S ∧ b ∈ S ∧ knight_moves c1 a ∧ knight_moves a b ∧ knight_moves b c2)) :=
sorry

noncomputable def knight_moves (pos1 pos2 : Fin 8 × Fin 8) : Prop :=
  let dx := abs (pos1.1.val - pos2.1.val)
  let dy := abs (pos1.2.val - pos2.2.val)
  (dx = 1 ∧ dy = 2) ∨ (dx = 2 ∧ dy = 1)

end max_marked_cells_knight_moves_l722_722353


namespace only_one_prime_in_range_l722_722097

theorem only_one_prime_in_range : ∀ (n : ℕ), (200 < n ∧ n < 220) → (nat.prime n → n = 211) :=
by
  sorry

end only_one_prime_in_range_l722_722097


namespace max_b_no_lattice_points_max_possible_b_l722_722817

theorem max_b_no_lattice_points (m : ℚ) (H1 : 1 / 2 < m) (H2 : m < 76 / 151)
  (H3 : ∀ x : ℕ, 0 < x ∧ x ≤ 150 → ¬ is_lattice_point (m * x + 3)) : m < 76 / 151 :=
sorry

noncomputable def is_lattice_point (y : ℚ) : Prop :=
∃ n : ℤ, y = n

# The main theorem where the conditions are used.
theorem max_possible_b : 
  ∃ b : ℚ, b = 76 / 151 ∧ (∀ (m : ℚ), 1 / 2 < m ∧ m < b → ∀ x : ℕ, 0 < x ∧ x ≤ 150 → ¬ is_lattice_point (m * x + 3)) :=
sorry

end max_b_no_lattice_points_max_possible_b_l722_722817


namespace eval_sqrt_expression_l722_722881

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722881


namespace external_sides_of_shapes_l722_722819

theorem external_sides_of_shapes :
  let triangle_sides := 3,
      square_sides := 4,
      pentagon_sides := 5,
      hexagon_sides := 6,
      heptagon_sides := 7,
      octagon_sides := 8,
      nonagon_sides := 9,
      shared_sides := 1, -- each shape shares one side with the previous and next shape,
                          -- except the nonagon which only shares one side
      total_sides := (triangle_sides - shared_sides)
                    + (square_sides - 2 * shared_sides)
                    + (pentagon_sides - 2 * shared_sides)
                    + (hexagon_sides - 2 * shared_sides)
                    + (heptagon_sides - 2 * shared_sides)
                    + (octagon_sides - 2 * shared_sides)
                    + (nonagon_sides - shared_sides)
  in total_sides = 30 := 
  sorry

end external_sides_of_shapes_l722_722819


namespace shorter_leg_of_right_triangle_l722_722572

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l722_722572


namespace infinite_set_no_perfect_square_sum_l722_722826

theorem infinite_set_no_perfect_square_sum :
  ∃ (H : Set ℕ), (∀ n ∈ H, ∃ k : ℕ, n = 2 ^ (2 * k + 1)) ∧ 
  Set.Infinite H ∧ 
  (∀ K ⊆ H, K.Nonempty → (∑ n in K, n) ≠ k ^ 2)
  :=
by
  sorry

end infinite_set_no_perfect_square_sum_l722_722826


namespace shorter_leg_in_right_triangle_l722_722579

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l722_722579


namespace no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l722_722096

theorem no_even_integers_of_form_3k_plus_4_and_5m_plus_2 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, n = 3 * k + 4) (h3 : ∃ m : ℕ, n = 5 * m + 2) (h4 : n % 2 = 0) : false :=
sorry

end no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l722_722096


namespace coeff_of_z_in_eq2_l722_722948

-- Definitions of the conditions from part a)
def equation1 (x y z : ℤ) := 6 * x - 5 * y + 3 * z = 22
def equation2 (x y z : ℤ) := 4 * x + 8 * y - z = (7 : ℚ) / 11
def equation3 (x y z : ℤ) := 5 * x - 6 * y + 2 * z = 12
def sum_xyz (x y z : ℤ) := x + y + z = 10

-- Theorem stating that the coefficient of z in equation 2 is -1.
theorem coeff_of_z_in_eq2 {x y z : ℤ} (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) (h4 : sum_xyz x y z) :
    -1 = -1 :=
by
  -- This is a placeholder for the proof.
  sorry

end coeff_of_z_in_eq2_l722_722948


namespace domain_of_h_l722_722665

theorem domain_of_h {f : ℝ → ℝ} (hf : ∀ x, -9 ≤ x → x ≤ 6 → f x = f x) :
  ∀ x, h x = f (-3 * x) → (-2 ≤ x ∧ x ≤ 3) :=
by
  -- define h(x)
  let h := λ x, f (-3 * x)
  -- for all x, h(x) = f(-3x) implies the domain is within [-2, 3]
  assume x hx
  sorry

end domain_of_h_l722_722665


namespace average_of_k_with_pos_int_roots_l722_722073

theorem average_of_k_with_pos_int_roots :
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ pair => pair.1 + pair.2)
  (k_values.sum / k_values.length) = 15 :=
by
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ pair => pair.1 + pair.2)
  have : k_values = [25, 14, 11, 10], by sorry
  let avg_k := (k_values.sum) / (k_values.length : ℕ)
  have : k_values.sum = 60, by sorry
  have : k_values.length = 4, by sorry
  have : avg_k = 15, by sorry
  show avg_k = 15, from this

end average_of_k_with_pos_int_roots_l722_722073


namespace problem1_part1_problem1_part2_problem2_l722_722961

open Set

variable {U : Set ℝ}
variable {A B : Set ℝ}

-- Define Universal Set U, Set A, Set B
def U : Set ℝ := {x | true}
def A : Set ℝ := {x | 1 < x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | 6 - a < x ∧ x < 2a - 1}

-- Question 1: If a = 4
theorem problem1_part1 : A ∪ B 4 = {x | 1 < x ∧ x < 7} := by
  sorry

theorem problem1_part2 : B 4 ∩ (U \ A) = {x | 4 < x ∧ x < 7} := by
  sorry

-- Question 2: If A ⊆ B, find the range of values for a
theorem problem2 (a : ℝ) (h : A ⊆ B a) : a ≥ 5 := by
  sorry

end problem1_part1_problem1_part2_problem2_l722_722961


namespace quadratic_non_real_roots_iff_l722_722149

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722149


namespace range_of_b_l722_722532

noncomputable def curve (p q : ℝ) : Prop :=
  sqrt (p^2 + q^2) + abs (p - 4) = 5

def symmetric_points (b : ℝ) : Prop :=
  -- Define the symmetry condition as per the problem
  ∃ (p1 q1 p2 q2 p3 q3 : ℝ),
    curve p1 q1 ∧ curve p2 q2 ∧ curve p3 q3 ∧ 
    (p1 = 2*b - p2) ∧ (p2 = 2*b - p3)

theorem range_of_b :
  ∀ b : ℝ, (symmetric_points b) → 2 < b ∧ b < 4 :=
sorry

end range_of_b_l722_722532


namespace limit_of_function_l722_722812

open Real

theorem limit_of_function (a : ℝ) (h : a > 0 ) :
  (tendsto (λ x, (ln (cos (x / a) + 2) / 
    (a ^ ((a^2 * π^2) / (x^2) - (a * π) / x) - a ^ ((a * π) / x - 1))))
   (𝓝[<] (a * π)) (𝓝 (π^2 / (2 * ln a))))
:=
sorry

end limit_of_function_l722_722812


namespace vector_subtraction_l722_722451

open Matrix

def v1 : Vector ℤ 2 := ![2, -8]
def v2 : Vector ℤ 2 := ![1, -7]

theorem vector_subtraction : 3 • v1 - 2 • v2 = ![4, -10] :=
by
  sorry

end vector_subtraction_l722_722451


namespace number_of_cube_roots_lt_15_l722_722111

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l722_722111


namespace min_flips_to_all_down_l722_722702

-- Define the initial and final states of the cups
inductive CupState
| up : CupState
| down : CupState

-- Define the flip operation
def flip (c1 c2 c3 c4 : CupState) : CupState × CupState × CupState × CupState :=
  (if c1 = CupState.up then CupState.down else CupState.up,
   if c2 = CupState.up then CupState.down else CupState.up,
   if c3 = CupState.up then CupState.down else CupState.up,
   c4)

def flip_any_three (cups : CupState × CupState × CupState × CupState) : List (CupState × CupState × CupState × CupState) :=
  [flip cups.1 cups.2 cups.3 cups.4,
   flip cups.1 cups.2 cups.4 cups.3,
   flip cups.1 cups.3 cups.4 cups.2,
   flip cups.2 cups.3 cups.4 cups.1]

noncomputable def find_min_n (initial_state : CupState × CupState × CupState × CupState) (final_state : CupState × CupState × CupState × CupState) (n : ℕ) : Prop :=
  ∃ flips : List (CupState × CupState × CupState × CupState), flips.length = n ∧ 
   List.foldl (λ state flip, flip_any_three state) initial_state flips = final_state

-- The conjecture we want to prove
theorem min_flips_to_all_down : find_min_n (CupState.up, CupState.up, CupState.up, CupState.up) (CupState.down, CupState.down, CupState.down, CupState.down) 4 :=
by sorry

end min_flips_to_all_down_l722_722702


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722107

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722107


namespace range_x2y2z_range_a_inequality_l722_722761

theorem range_x2y2z {x y z : ℝ} (h : x^2 + y^2 + z^2 = 1) : 
  -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3 :=
by sorry

theorem range_a_inequality (a : ℝ) (h : ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) :
  (4 ≤ a) ∨ (a ≤ 0) :=
by sorry

end range_x2y2z_range_a_inequality_l722_722761


namespace sqrt_continued_fraction_l722_722861

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722861


namespace simplify_sqrt_l722_722657

theorem simplify_sqrt : sqrt 12 = 2 * sqrt 3 :=
by
  sorry

end simplify_sqrt_l722_722657


namespace find_smaller_number_l722_722696

-- Define the conditions
def sum_of_numbers (x y : ℕ) := x + y = 70
def second_number_relation (x y : ℕ) := y = 3 * x + 10

-- Define the problem statement
theorem find_smaller_number (x y : ℕ) (h1 : sum_of_numbers x y) (h2 : second_number_relation x y) : x = 15 :=
sorry

end find_smaller_number_l722_722696


namespace james_speed_on_second_trail_l722_722238

theorem james_speed_on_second_trail :
  ∀ (speed1 speed2 miles1 miles2 break_time : ℕ),
  let time1 := miles1 / speed1,
      effective_time2 := time1 - 1 - break_time,
      segment_distance := miles2 / 2
  in miles1 = 20 → speed1 = 5 →
     miles2 = 12 → break_time = 1 →
     effective_time2 = 2 →
     speed2 = segment_distance / 1 →
     speed2 = 6 :=
by
  intros speed1 speed2 miles1 miles2 break_time time1 effective_time2 segment_distance h_miles1 h_speed1 h_miles2 h_break_time h_effective_time2 h_speed2
  rw [h_miles1, h_speed1, h_miles2, h_break_time] at *
  simp at h_effective_time2 h_speed2
  exact h_speed2
  sorry

end james_speed_on_second_trail_l722_722238


namespace gum_distribution_l722_722240

theorem gum_distribution :
  let john_gum := 54
  let cole_gum := 45
  let aubrey_gum := 0
  let total_gum := john_gum + cole_gum + aubrey_gum
  let num_people := 3
  let each_gum := total_gum / num_people
  each_gum = 33 :=
by
  let john_gum := 54
  let cole_gum := 45
  let aubrey_gum := 0
  let total_gum := john_gum + cole_gum + aubrey_gum
  let num_people := 3
  let each_gum := total_gum / num_people
  have h1 : total_gum = 99 := by norm_num
  have h2 : each_gum = 99 / 3 := by norm_num
  exact eq.trans h2 (by norm_num : 99 / 3 = 33)

end gum_distribution_l722_722240


namespace exists_circle_nonoverlap_l722_722759

theorem exists_circle_nonoverlap (large_side small_side radius : ℝ) (n : ℕ) 
  (h_large : large_side = 15) 
  (h_small : small_side = 1) 
  (h_radius : radius = 1) 
  (h_n : n = 20) 
  (h_nonoverlap : ∀ i j, i ≠ j → disjoint (small_square i) (small_square j)) :
  ∃ center : ℝ × ℝ, ∀ i, ¬(circle_intersects_square center radius (small_square i)) :=
sorry

end exists_circle_nonoverlap_l722_722759


namespace sqrt_product_simplify_l722_722808

theorem sqrt_product_simplify (p : ℝ) : 
  sqrt (8 * p ^ 2) * sqrt (12 * p ^ 3) * sqrt (18 * p ^ 5) = 24 * p ^ 5 * sqrt 3 := 
by
  sorry

end sqrt_product_simplify_l722_722808


namespace simple_interest_proof_l722_722714

def simple_interest (P R T: ℝ) : ℝ :=
  P * R * T

theorem simple_interest_proof :
  simple_interest 810 (4.783950617283951 / 100) 4 = 154.80 :=
by
  sorry

end simple_interest_proof_l722_722714


namespace continuous_at_two_l722_722498

def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≥ 2 then x^2 + 1 else 3 * x + b

theorem continuous_at_two {b : ℝ} :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, abs x - 2 < δ → abs (f x b - f 2 b) < ε) ↔ b = -1 := 
sorry

end continuous_at_two_l722_722498


namespace dice_sum_10_l722_722719

/-- When 7 fair 6-sided dice are rolled, the number of successful outcomes 
where the sum of the top faces is 10 is 84 -/
theorem dice_sum_10 : ∃ n : ℕ, (n = 84) ∧ (n / 6^7 = probability_of_sum_10) := 
sorry
where 
  probability_of_sum_10 : ℚ := (number_of_successful_outcomes_in_sum_10 / (6 ^ 7))

def number_of_successful_outcomes_in_sum_10 : ℕ := 
  choose (9, 6) -- This represents the combination binom(9, 6)

def choose (n : ℕ) (k : ℕ) : ℕ :=
  if h : 0 ≤ k ∧ k ≤ n then 
    nat.choose n k
  else 
    0

attribute [irreducible] choose

example : choose 9 6 = 84 := by
  simp [choose, nat.choose]

end dice_sum_10_l722_722719


namespace evaluate_nested_radical_l722_722858

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722858


namespace solve_r_minus_s_l722_722255

noncomputable def r := 20
noncomputable def s := 4

theorem solve_r_minus_s
  (h1 : r^2 - 24 * r + 80 = 0)
  (h2 : s^2 - 24 * s + 80 = 0)
  (h3 : r > s) : r - s = 16 :=
by
  sorry

end solve_r_minus_s_l722_722255


namespace counting_true_statements_l722_722703

theorem counting_true_statements :
  let p1 := ∀ (f : α → β), f (dom f) ⊆ ran f
  let p2 := ∀ (y : ℕ → ℕ), (∀ x : ℕ, y x = 2 * x) → (∀ x : ℕ, int (y x) = y x)
  let p3 := ∀ (a : ℝ), (0 < a ∧ a ≠ 1) → (∀ x : ℝ, log a (a^x) = x)
  let p4 := ∀ (a : ℝ), (0 < a ∧ a ≠ 1) → (f(a, x := a^(x + 1) - 1)) ∈ {(x = -1 ∧ y = 0)}
  count_trues p1 p2 p3 p4 = 2 := 
by
  sorry

end counting_true_statements_l722_722703


namespace complex_number_in_fourth_quadrant_l722_722596

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 + complex.I) * (1 - 2 * complex.I)
  z.re = 3 ∧ z.im = -1 ∧ (z.re > 0 ∧ z.im < 0) :=
by {
  let z : ℂ := (1 + complex.I) * (1 - 2 * complex.I),
  sorry
}

end complex_number_in_fourth_quadrant_l722_722596


namespace cone_generatrix_length_l722_722059

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722059


namespace door_height_l722_722214

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l722_722214


namespace complex_coordinates_l722_722081

theorem complex_coordinates (i : ℂ) (z : ℂ) (h : i^2 = -1) (h_z : z = (1 + 2 * i^3) / (2 + i)) :
  z = -i := 
by {
  sorry
}

end complex_coordinates_l722_722081


namespace checkerboard_painting_l722_722514

statement:
theorem checkerboard_painting (n : ℕ) (initially_checkerboard : ∀ i j, (i + j) % 2 = 0 → cell_color i j = black) (one_black_corner : cell_color 0 0 = black)
  (repaint : ∀ i j, 0 ≤ i ∧ i < n - 1 ∧ 0 ≤ j ∧ j < n - 1 →
            let new_cell_color : ℕ → ℕ := λ c, if c = white then black else if c = black then green else white in
            cell_color i j = new_cell_color (cell_color i j) ∧
            cell_color (i+1) j = new_cell_color (cell_color (i+1) j) ∧
            cell_color i (j+1) = new_cell_color (cell_color i (j+1)) ∧
            cell_color (i+1) (j+1) = new_cell_color (cell_color (i+1) (j+1))
  ) :
  n % 3 = 0 := 
sorry

end checkerboard_painting_l722_722514


namespace geometric_sequence_sum_ratio_l722_722629

noncomputable def a (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * q^n

-- Sum of the first 'n' terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a1 q : ℝ) 
  (h : 8 * (a 11 a1 q) = (a 14 a1 q)) :
  (S 4 a1 q) / (S 2 a1 q) = 5 :=
by
  sorry

end geometric_sequence_sum_ratio_l722_722629


namespace range_of_f_value_of_a_in_triangle_l722_722548

open Real

-- Problem 1: Range of f(x)
noncomputable def f (x : ℝ) : ℝ := (sin x) * (sin x) + (sin x) * (cos x)

theorem range_of_f (x : ℝ) (hx : -π/4 ≤ x ∧ x ≤ π/4) : 
  f x ∈ set.Icc (- (sqrt 2) / 2 + 1 / 2) 1 :=
sorry

-- Problem 2: Value of a in triangle ABC
variables (A B C a b c : ℝ)

theorem value_of_a_in_triangle
  (hb : b = sqrt 2) (hc : c = sqrt 3) (hB : B = π / 4) (hA : A = π - B - (π/3)) :
  a = (sqrt 6 + sqrt 2) / 2 :=
sorry

end range_of_f_value_of_a_in_triangle_l722_722548


namespace eval_sqrt_expression_l722_722878

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722878


namespace sixth_circle_contains_6000_anglets_l722_722735

def anglet_per_degree : ℝ := 100
def degrees_in_circle : ℝ := 360
def sixth_of_circle : ℝ := degrees_in_circle / 6
def anglets_in_sixth_of_circle : ℝ := sixth_of_circle * anglet_per_degree

theorem sixth_circle_contains_6000_anglets :
  anglets_in_sixth_of_circle = 6000 := 
by
  -- proof to be provided
  sorry

end sixth_circle_contains_6000_anglets_l722_722735


namespace cone_generatrix_length_l722_722056

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722056


namespace non_real_roots_bounded_l722_722157

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722157


namespace part1_part2_l722_722952

section Part1
variables (x : ℝ) (a : ℝ)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + (2*a - 1) * x - 3

theorem part1 : 
  (∀ x ∈ set.Icc (-2 : ℝ) 3, f x 2 >= -21/4 ∧ f x 2 <= 15) :=
sorry
end Part1

section Part2
variables (x : ℝ) (a : ℝ)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + (2*a - 1) * x - 3

theorem part2 : 
  (∃ a : ℝ, ∀ x ∈ set.Icc (-1 : ℝ) 3, f x a ≤ 1) ↔ (a = -1 ∨ a = -1/3) :=
sorry
end Part2

end part1_part2_l722_722952


namespace siamese_cats_initially_l722_722422

-- Define the initial number of house cats
def house_cats : ℕ := 49

-- Define the number of cats sold
def cats_sold : ℕ := 19

-- Define the number of cats left after the sale
def cats_left : ℕ := 45

-- Statement: Prove the initial number of Siamese cats
theorem siamese_cats_initially (S : ℕ) (total_cats_initial : ℕ) : 
  total_cats_initial = cats_left + cats_sold → S + house_cats = total_cats_initial → S = 15 :=
by
  intros h1 h2
  have h3 : total_cats_initial = 64 := h1
  have h4 : S + house_cats = 64 := h2
  linarith [h4]


end siamese_cats_initially_l722_722422


namespace candy_pieces_per_pile_l722_722911

theorem candy_pieces_per_pile :
  ∀ (total_candies eaten_candies num_piles pieces_per_pile : ℕ),
    total_candies = 108 →
    eaten_candies = 36 →
    num_piles = 8 →
    pieces_per_pile = (total_candies - eaten_candies) / num_piles →
    pieces_per_pile = 9 :=
by
  intros total_candies eaten_candies num_piles pieces_per_pile
  sorry

end candy_pieces_per_pile_l722_722911


namespace intervals_of_monotonicity_and_extrema_l722_722086

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x - Real.log x

theorem intervals_of_monotonicity_and_extrema :
  (∀ x, 0 < x ∧ x < 1 → (f ' x > 0)) ∧
  (∀ x, 1 < x → (f ' x < 0)) ∧
  (∃ x, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧
    f x = 0 ∧
    ∀ y, y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f y ≤ f x) ∧
  (∃ x, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧
    f x = 2 - Real.exp 1 ∧
    ∀ y, y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f y ≥ f x) :=
by
  sorry

end intervals_of_monotonicity_and_extrema_l722_722086


namespace sum_squares_sin_is_45_l722_722463

noncomputable def sum_squares_sin_angles : ℝ :=
  (Finset.range 90).sum (λ n, Real.sin ((n + 1) * Real.pi / 180) ^ 2)

theorem sum_squares_sin_is_45 :
  sum_squares_sin_angles = 45 :=
sorry

end sum_squares_sin_is_45_l722_722463


namespace exist_plane_intersecting_lines_with_ratio_l722_722511

variables {P : Point} {a b c g : Line}

def point (P : Type) := sorry   -- placeholder definitions (to be replaced with actual type definitions)
def line (P : Type) := sorry    -- placeholder definitions (to be replaced with actual type definitions)
def plane (P : Type) := sorry   -- placeholder definitions (to be replaced with actual type definitions)

-- Definition for lines intersecting a plane at specified points
def intersects_at (S : plane) (l : line) (p : point) : Prop := sorry

-- Proof statement
theorem exist_plane_intersecting_lines_with_ratio
  (P : point) (a b c g : line) : 
  ∃ S : plane, 
    intersects_at S a A ∧ 
    intersects_at S b B ∧ 
    intersects_at S c C ∧ 
    AB_Over_BC A B C = 2 :=
sorry

end exist_plane_intersecting_lines_with_ratio_l722_722511


namespace complement_union_example_l722_722960

open Set

/-- Problem Statement: Given the universal set U, set S and set T,
    show that the complement of the union of S and T with respect to U is {2, 4}. -/

theorem complement_union_example :
  let U := {1, 2, 3, 4, 5, 6}
  let S := {1, 3, 5}
  let T := {3, 6}
  (U \ (S ∪ T)) = {2, 4} :=
by
  -- proof omitted
  sorry

end complement_union_example_l722_722960


namespace gas_cost_per_gallon_l722_722457

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) (gallons_used : ℝ) (cost_per_gallon : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 304 →
  total_cost = 38 →
  gallons_used = total_miles / miles_per_gallon →
  cost_per_gallon = total_cost / gallons_used →
  cost_per_gallon = 4 :=
by
  intros h1 h2 h3 h4 h5
  have h_gallons: gallons_used = total_miles / miles_per_gallon, from h4
  have h_cost: cost_per_gallon = total_cost / gallons_used, from h5
  rw [h1, h2, h3] at h_gallons
  rw [h_gallons] at h_cost
  sorry

end gas_cost_per_gallon_l722_722457


namespace cindy_gets_three_same_color_l722_722401

/-- A box contains 2 red marbles, 2 green marbles, and 4 yellow marbles. Alice takes 2 marbles from the box at random;
Bob then takes 3 of the remaining marbles at random; and Cindy takes the last 3 marbles.
This theorem proves that the probability Cindy gets 3 marbles of the same color is 13/140. -/
theorem cindy_gets_three_same_color :
  let total_marbles := 8,
      red_marbles := 2,
      green_marbles := 2,
      yellow_marbles := 4,
      combinations := Nat.choose in
  let total_ways := combinations total_marbles 2 * combinations (total_marbles - 2) 3 * combinations (total_marbles - 5) 3,
      favorable_ways := (combinations 4 2 * combinations (total_marbles - 4) 2 * 1) + (combinations 4 1 * combinations (total_marbles - 1) 3 * 1) in
  (favorable_ways : ℚ) / total_ways = 13 / 140 := 
sorry

end cindy_gets_three_same_color_l722_722401


namespace algebraic_expression_solution_l722_722720

theorem algebraic_expression_solution
  (a b : ℝ)
  (h : -2 * a + 3 * b = 10) :
  9 * b - 6 * a + 2 = 32 :=
by 
  -- We would normally provide the proof here
  sorry

end algebraic_expression_solution_l722_722720


namespace minimize_cost_l722_722767

theorem minimize_cost (x : ℝ) (h1 : 0 < x) (h2 : 400 / x * 40 ≤ 4 * x) : x = 20 :=
by
  sorry

end minimize_cost_l722_722767


namespace function_evaluation_l722_722556

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2 * x ^ 2 + 1) : 
  ∀ x : ℝ, f x = 2 * x ^ 2 - 4 * x + 3 :=
sorry

end function_evaluation_l722_722556


namespace find_m_l722_722486

theorem find_m (lg2: ℝ) (h: lg2 ≈ 0.3010) : ∃ m : ℕ, 10^(m-1) < 2^512 ∧ 2^512 < 10^m ∧ m = 155 := by
  sorry

end find_m_l722_722486


namespace part_I_part_II_l722_722084

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem part_I (x : ℝ) (h₁ : 0 < x) (h₂ : x < Real.pi) : f x > -1 := 
sorry

theorem part_II : ∃! x ∈ Set.Ioo 2 3, f x = 0 :=
sorry

end part_I_part_II_l722_722084


namespace generatrix_length_of_cone_l722_722044

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722044


namespace tour_routes_count_l722_722775

theorem tour_routes_count :
  let A n k := factorial n / factorial (n - k),
      C n k := factorial n / (factorial k * factorial (n - k)) in
  A 5 3 * C 4 2 = 600 :=
by
  sorry

end tour_routes_count_l722_722775


namespace door_height_eight_l722_722220

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l722_722220


namespace quadratic_non_real_roots_b_values_l722_722173

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722173


namespace expression_evaluation_l722_722138

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l722_722138


namespace collinear_ABD_l722_722962

section VectorProof

variables (e1 e2 : Type) [AddCommGroup e1] [AddCommGroup e2]

structure Vector (α : Type) :=
  (x : α)
  (y : α)
  (non_collinear : e1 ≠ e2)

variables (w1 w2 : Vector ℝ) -- Assume w1 and w2 are representations for e1 and e2

noncomputable def AB (k : ℝ) := (2 : ℝ) • w1 + k • w2
noncomputable def CB := (1 : ℝ) • w1 + (3 : ℝ) • w2
noncomputable def CD := (2 : ℝ) • w1 + (-1 : ℝ) • w2

theorem collinear_ABD (k : ℝ) (h : ∀ {α : Type} (a b c : α), a + b = c → a = c - b) : k = -8 := 
by
  sorry

end VectorProof

end collinear_ABD_l722_722962


namespace find_other_integer_l722_722464

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 140) (h2 : x = 20 ∨ y = 20) : x = 20 ∧ y = 20 :=
by
  sorry

end find_other_integer_l722_722464


namespace central_angle_of_sector_l722_722194

theorem central_angle_of_sector :
  ∃ R α : ℝ, (2 * R + α * R = 4) ∧ (1 / 2 * R ^ 2 * α = 1) ∧ α = 2 :=
by
  sorry

end central_angle_of_sector_l722_722194


namespace initial_blonde_girls_l722_722270

theorem initial_blonde_girls (total_initial_girls added_blonde_girls black_haired_girls : ℕ) 
  (h1 : total_initial_girls = 80) 
  (h2 : added_blonde_girls = 10) 
  (h3 : black_haired_girls = 50)
  (total_girls : ℕ := total_initial_girls + added_blonde_girls) 
  (h4 : total_girls = 90) : total_initial_girls - black_haired_girls = 30 := 
by
  have initial_blonde_girls := total_initial_girls - black_haired_girls
  show initial_blonde_girls = 30
  sorry

end initial_blonde_girls_l722_722270


namespace arithmetic_sequence_sum_l722_722594

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_seq : arithmetic_sequence a)
  (h1 : a 0 + a 1 = 1)
  (h2 : a 2 + a 3 = 9) :
  a 4 + a 5 = 17 :=
sorry

end arithmetic_sequence_sum_l722_722594


namespace generatrix_length_of_cone_l722_722041

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722041


namespace train_journey_time_l722_722731

theorem train_journey_time :
  ∃ T : ℝ, (30 : ℝ) / 60 = (7 / 6 * T) - T ∧ T = 3 :=
by
  sorry

end train_journey_time_l722_722731


namespace existence_of_large_independent_subset_l722_722018

theorem existence_of_large_independent_subset (X : Type) (S : set (set X)) (n : ℕ) (hX : fintype X) (hS : ∀ A B ∈ S, A ≠ B → (A ∩ B).card ≤ 1) (hX_card : (fintype.card X) = n) :
  ∃ A ⊆ X, A ∉ S ∧ fintype.card A ≥ nat.floor (real.sqrt (2 * n)) :=
begin
  sorry,
end

end existence_of_large_independent_subset_l722_722018


namespace angles_of_given_triangle_l722_722809

noncomputable def calculate_triangle_angles 
  (a b c : ℝ)
  (h1 : a = 3)
  (h2 : b = 3)
  (h3 : c = real.sqrt 8 - real.sqrt 3)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) 
  : ℝ × ℝ × ℝ := 
  sorry

theorem angles_of_given_triangle : 
  calculate_triangle_angles 3 3 (real.sqrt 8 - real.sqrt 3) 
  (by norm_num) 
  (by norm_num) 
  (by norm_num) 
  (by norm_num : 3 + 3 > real.sqrt 8 - real.sqrt 3 ∧ 3 + (real.sqrt 8 - real.sqrt 3) > 3 ∧ 3 + (real.sqrt 8 - real.sqrt 3) > 3) 
  = (15.0, 82.5, 82.5) := 
  sorry

end angles_of_given_triangle_l722_722809


namespace bridge_length_l722_722403

theorem bridge_length
  (bus_length : ℝ)
  (bus_speed_kmph : ℝ)
  (cross_time_seconds : ℝ)
  (bus_length = 100)
  (bus_speed_kmph = 50)
  (cross_time_seconds = 18) :
  let bus_speed_mps := bus_speed_kmph * 1000 / 3600 
  let total_distance := bus_speed_mps * cross_time_seconds in
  total_distance - bus_length = 150 :=
begin
  sorry
end

end bridge_length_l722_722403


namespace quadratic_non_real_roots_b_values_l722_722174

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722174


namespace product_area_perimeter_eq_68sqrt17_l722_722277

-- Definitions for vertices of the square
def E : ℝ × ℝ := (1, 5)
def F : ℝ × ℝ := (5, 6)
def G : ℝ × ℝ := (6, 2)
def H : ℝ × ℝ := (2, 1)

-- Define the side length calculation
def side_length (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.fst - P.fst)^2 + (Q.snd - P.snd)^2)

-- Define the area of the square
def area_square (side : ℝ) : ℝ :=
  side^2

-- Define the perimeter of the square
def perimeter_square (side : ℝ) : ℝ :=
  4 * side

-- Define the product of the area and the perimeter
def area_perimeter_product (side : ℝ) : ℝ :=
  area_square(side) * perimeter_square(side)

-- Assertion for the problem
theorem product_area_perimeter_eq_68sqrt17 :
  area_perimeter_product (side_length E F) = 68 * real.sqrt 17 := by
  sorry

end product_area_perimeter_eq_68sqrt17_l722_722277


namespace cone_generatrix_length_l722_722048

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722048


namespace generatrix_length_of_cone_l722_722046

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722046


namespace square_area_l722_722430

theorem square_area (P : ℝ) (h : P = 48) : ∃ (A : ℝ), A = 144 :=
by
  have s := P / 4
  have s_val : s = 12 := by
    rw [h]
    norm_num
  exists (s * s)
  rw [s_val]
  norm_num
  assumption

end square_area_l722_722430


namespace greatest_negative_value_x_minus_y_l722_722903

noncomputable theory

open Real

theorem greatest_negative_value_x_minus_y :
  ∀ x y : ℝ, (sin x + sin y) * (cos x - cos y) = 1 / 2 + sin(x - y) * cos(x + y) →
  x - y ≤ -π / 6 :=
by
  intros x y h
  -- Proof will go here
  sorry

end greatest_negative_value_x_minus_y_l722_722903


namespace smallest_n_ratio_l722_722908

theorem smallest_n_ratio (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (a + b * complex.I) ^ 4 = (a - b * complex.I) ^ 4 → b / a = 1 :=
by
  sorry

end smallest_n_ratio_l722_722908


namespace cone_generatrix_length_is_2sqrt2_l722_722033

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722033


namespace nested_radical_solution_l722_722836

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722836


namespace squirrel_walnuts_l722_722402

theorem squirrel_walnuts :
  let boy_gathered := 6
  let boy_dropped := 1
  let initial_in_burrow := 12
  let girl_brought := 5
  let girl_ate := 2
  initial_in_burrow + (boy_gathered - boy_dropped) + girl_brought - girl_ate = 20 :=
by
  sorry

end squirrel_walnuts_l722_722402


namespace non_real_roots_interval_l722_722184

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722184


namespace number_of_prime_factors_30_factorial_l722_722978

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722978


namespace function_relationship_value_of_x_when_y_is_1_l722_722015

variable (x y : ℝ) (k : ℝ)

-- Conditions
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x - 3)

axiom condition_1 : inverse_proportion x y
axiom condition_2 : y = 5 ∧ x = 4

-- Statements to be proved
theorem function_relationship :
  ∃ k : ℝ, (y = k / (x - 3)) ∧ (y = 5 ∧ x = 4 → k = 5) :=
by
  sorry

theorem value_of_x_when_y_is_1 (hy : y = 1) :
  ∃ x : ℝ, (y = 5 / (x - 3)) ∧ x = 8 :=
by
  sorry

end function_relationship_value_of_x_when_y_is_1_l722_722015


namespace sodium_nitrate_silver_chloride_formation_l722_722447

theorem sodium_nitrate_silver_chloride_formation:
  ∀ (m_AgNO3 m_NaOH m_HCl: ℝ),
  (m_AgNO3 = 2) →
  (m_NaOH = 2) →
  (m_HCl = 0.5) →
  let m_NaNO3 := min m_NaOH (m_AgNO3 - (m_HCl / 1)) in
  let m_AgCl := m_NaNO3 in
  let m_NaOH_unreacted := m_NaOH - m_NaNO3 in
  m_NaNO3 = 1.5 ∧ m_AgCl = 1.5 ∧ m_NaOH_unreacted = 0 := 
begin
  intros m_AgNO3 m_NaOH m_HCl h1 h2 h3,
  have h4 : m_HCl / 1 = 0.5, by rw div_one m_HCl,
  have h5 : m_AgNO3 - 0.5 = 1.5, by linarith,
  let m_NaNO3 := min m_NaOH 1.5,
  have h6 : m_NaNO3 = 1.5, by {
    unfold m_NaNO3,
    rw min_eq_right,
    exact le_of_lt (by linarith)
  },
  let m_AgCl := m_NaNO3,
  have h7 : m_AgCl = 1.5, by rw h6,
  let m_NaOH_unreacted := m_NaOH - m_NaNO3,
  have h8 : m_NaOH_unreacted = 0, by {
    unfold m_NaOH_unreacted,
    rw [h6, sub_self 1.5],
  },
  exact ⟨h6, h7, h8⟩,
end

end sodium_nitrate_silver_chloride_formation_l722_722447


namespace sqrt_of_product_of_factorials_eq_480_l722_722356

def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem sqrt_of_product_of_factorials_eq_480 : (Real.sqrt ((factorial 5 * factorial 4) / factorial 3))^2 = 480 := 
by 
  sorry

end sqrt_of_product_of_factorials_eq_480_l722_722356


namespace real_possible_b_values_quadratic_non_real_roots_l722_722145

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722145


namespace correct_quotient_is_48_l722_722385

theorem correct_quotient_is_48 (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 72 → 
  incorrect_quotient = 24 → 
  correct_divisor = 36 →
  dividend = incorrect_divisor * incorrect_quotient →
  correct_quotient = dividend / correct_divisor →
  correct_quotient = 48 :=
by
  sorry

end correct_quotient_is_48_l722_722385


namespace non_real_roots_bounded_l722_722158

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722158


namespace Melissa_commission_l722_722274

theorem Melissa_commission 
  (coupe_price : ℝ)
  (suv_multiplier : ℝ)
  (commission_rate : ℝ) :
  (coupe_price = 30000) →
  (suv_multiplier = 2) →
  (commission_rate = 0.02) →
  let suv_price := suv_multiplier * coupe_price in
  let total_sales := coupe_price + suv_price in
  let commission := commission_rate * total_sales in
  commission = 1800 :=
begin
  sorry
end

end Melissa_commission_l722_722274


namespace quadratic_non_real_roots_iff_l722_722151

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722151


namespace min_x2_y2_of_product_eq_zero_l722_722736

theorem min_x2_y2_of_product_eq_zero (x y : ℝ) (h : (x + 8) * (y - 8) = 0) : x^2 + y^2 = 64 :=
sorry

end min_x2_y2_of_product_eq_zero_l722_722736


namespace inscribed_circle_radius_l722_722327

noncomputable def calculate_r (a b c : ℝ) : ℝ :=
  let term1 := 1 / a
  let term2 := 1 / b
  let term3 := 1 / c
  let term4 := 3 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))
  1 / (term1 + term2 + term3 + term4)

theorem inscribed_circle_radius :
  calculate_r 6 10 15 = 30 / (10 * Real.sqrt 26 + 3) :=
by
  sorry

end inscribed_circle_radius_l722_722327


namespace number_of_prime_factors_30_factorial_l722_722969

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722969


namespace sqrt_continued_fraction_l722_722867

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722867


namespace door_height_eight_l722_722219

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l722_722219


namespace eval_sqrt_expression_l722_722880

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722880


namespace det_example_N_eq_correct_det_l722_722206

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- Given polynomials M and incorrect determinant calculation
def M (m n : ℤ) : ℤ := m^2 - 2 * m * n
def incorrect_det (m n : ℤ) : ℤ := 6 * m^2 - 7 * m * n

-- Proving the determinant | 2 -3 | = 22
                     -- | 4  5 |
theorem det_example : det 2 (-3) 4 5 = 22 :=
by { dsimp [det], norm_num, }

-- Find the polynomial N
def poly_N (m n : ℤ) : ℤ :=
let M_val := M m n in
6 * m^2 - 7 * m * n - 3 * M_val

theorem N_eq (m n : ℤ) : poly_N m n = 3 * m^2 - m * n :=
by { dsimp [poly_N, M], ring, }

-- Prove the correct determinant |M N| = -5mn
                             --  |1  3|
theorem correct_det (m n : ℤ) : det (M m n) (poly_N m n) 1 3 = -5 * m * n :=
by { dsimp [det, M, poly_N], ring, }

-- Provide unknown values so that Lean accepts the proof outline
def example_problem :=
(λ h : ∀ (m n : ℤ), (3 * (M m n) + poly_N m n = 6 * m^2 - 7 * m * n), det_example, N_eq, correct_det)

end det_example_N_eq_correct_det_l722_722206


namespace generatrix_length_of_cone_l722_722042

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722042


namespace polynomial_multiplication_l722_722475

theorem polynomial_multiplication :
  (5 * X^2 + 3 * X - 4) * (2 * X^3 + X^2 - X + 1) = 
  (10 * X^5 + 11 * X^4 - 10 * X^3 - 2 * X^2 + 7 * X - 4) := 
by {
  sorry
}

end polynomial_multiplication_l722_722475


namespace length_generatrix_cone_l722_722068

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722068


namespace dog_total_distance_l722_722762

-- Define the conditions
def distance_between_A_and_B : ℝ := 100
def speed_A : ℝ := 6
def speed_B : ℝ := 4
def speed_dog : ℝ := 10

-- Define the statement we want to prove
theorem dog_total_distance : ∀ t : ℝ, (speed_A + speed_B) * t = distance_between_A_and_B → speed_dog * t = 100 :=
by
  intro t
  intro h
  sorry

end dog_total_distance_l722_722762


namespace number_of_distinct_prime_factors_30_fact_l722_722998

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722998


namespace prob_2_lt_X_leq_4_l722_722090

-- Definitions based on the conditions
def P (X : ℕ) (a : ℕ) : ℝ := X / (2 * a)

def sum_of_probabilities_eq_one (a : ℕ) : Prop :=
  P 1 a + P 2 a + P 3 a + P 4 a = 1

-- The main statement to prove
theorem prob_2_lt_X_leq_4 (a : ℕ) (h : sum_of_probabilities_eq_one a) :
  P 3 a + P 4 a = 7 / 10 :=
sorry

end prob_2_lt_X_leq_4_l722_722090


namespace fill_cistern_time_l722_722781

theorem fill_cistern_time (fill_ratio : ℚ) (time_for_fill_ratio : ℚ) :
  fill_ratio = 1/11 ∧ time_for_fill_ratio = 4 → (11 * time_for_fill_ratio) = 44 :=
by
  sorry

end fill_cistern_time_l722_722781


namespace common_ratio_geometric_series_l722_722892

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l722_722892


namespace birgit_faster_than_group_l722_722661

/-- Given conditions: the average speed of a group during a hike, and Birgit's speed, 
prove that Birgit was 4 km/hour faster than the average speed of the group. -/
theorem birgit_faster_than_group
  (total_distance : ℝ)
  (total_time : ℝ)
  (birgit_distance : ℝ)
  (birgit_time_min : ℝ)
  (average_speed : ℝ := total_distance / total_time)
  (birgit_speed : ℝ := birgit_distance / (birgit_time_min / 60)) :
  total_distance = 21 →
  total_time = 3.5 →
  birgit_distance = 8 →
  birgit_time_min = 48 →
  birgit_speed - average_speed = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end birgit_faster_than_group_l722_722661


namespace track_width_eight_l722_722428

theorem track_width_eight (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 16 * Real.pi) : r1 - r2 = 8 := 
sorry

end track_width_eight_l722_722428


namespace curve_is_line_l722_722900

theorem curve_is_line (r θ : ℝ) (h : r = 2 / (Real.sin θ + Real.cos θ)) : 
  ∃ m b, ∀ θ, r * Real.cos θ = m * (r * Real.sin θ) + b :=
sorry

end curve_is_line_l722_722900


namespace cone_generatrix_length_is_2sqrt2_l722_722028

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722028


namespace generatrix_length_of_cone_l722_722043

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722043


namespace distance_from_edge_to_bottom_l722_722642

theorem distance_from_edge_to_bottom (d x : ℕ) 
  (h1 : 63 + d + 20 = 10 + d + x) : x = 73 := by
  -- This is where the proof would go
  sorry

end distance_from_edge_to_bottom_l722_722642


namespace proof_by_contradiction_m7_lt_n7_l722_722369

theorem proof_by_contradiction_m7_lt_n7 (m n : ℝ) (h : m < n) : m^7 < n^7 :=
begin
  assume h₁ : m^7 >= n^7,
  sorry
end

end proof_by_contradiction_m7_lt_n7_l722_722369


namespace common_ratio_geometric_series_l722_722891

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l722_722891


namespace evaluate_nested_radical_l722_722854

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722854


namespace Mr_Tom_invested_in_fund_X_l722_722276

theorem Mr_Tom_invested_in_fund_X (a b : ℝ) (h1 : a + b = 100000) (h2 : 0.17 * b = 0.23 * a + 200) : a = 42000 := 
by
  sorry

end Mr_Tom_invested_in_fund_X_l722_722276


namespace volume_of_rectangular_box_l722_722357

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l722_722357


namespace smallest_integer_satisfying_conditions_l722_722715

theorem smallest_integer_satisfying_conditions :
  ∃ M : ℕ, M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 ∧ M % 12 = 11 ∧ M = 27719 := by
  sorry

end smallest_integer_satisfying_conditions_l722_722715


namespace crocus_to_daffodil_ratio_is_3_l722_722239

-- Definitions based on conditions
def paid_per_bulb : ℝ := 0.50
def tulip_bulbs : ℕ := 20
def iris_bulbs : ℕ := tulip_bulbs / 2
def daffodil_bulbs : ℕ := 30
def total_earning : ℝ := 75

-- Calculation based on conditions and problem statement
def total_bulbs_excluding_crocus : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs
def earning_excluding_crocus : ℝ := total_bulbs_excluding_crocus * paid_per_bulb
def earning_crocus : ℝ := total_earning - earning_excluding_crocus
def crocus_bulbs : ℕ := earning_crocus / paid_per_bulb

-- Final ratio definition
def ratio_crocus_to_daffodil : ℝ := crocus_bulbs / daffodil_bulbs

-- The theorem to prove the ratio is 3:1
theorem crocus_to_daffodil_ratio_is_3 : ratio_crocus_to_daffodil = 3 := by
  -- Skipping the proof steps
  sorry

end crocus_to_daffodil_ratio_is_3_l722_722239


namespace ten_percent_of_fifty_percent_of_five_hundred_l722_722395

theorem ten_percent_of_fifty_percent_of_five_hundred :
  0.10 * (0.50 * 500) = 25 :=
by
  sorry

end ten_percent_of_fifty_percent_of_five_hundred_l722_722395


namespace negation_exists_l722_722324

theorem negation_exists :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ ∃ x : ℝ, x^2 + 1 < x :=
sorry

end negation_exists_l722_722324


namespace other_root_eq_l722_722938

theorem other_root_eq (b : ℝ) : (∀ x, x^2 + b * x - 2 = 0 → (x = 1 ∨ x = -2)) :=
by
  intro x hx
  have : x = 1 ∨ x = -2 := sorry
  exact this

end other_root_eq_l722_722938


namespace number_of_eight_letter_good_words_l722_722459

/-- Define a good word and calculate the number of eight-letter good words. --/
def good_word (seq : List Char) : Prop :=
  (∀ i, i < seq.length - 1 → (seq[i] = 'A' → seq[i+1] ≠ 'B') ∧
                     (seq[i] = 'B' → seq[i+1] ≠ 'C') ∧
                     (seq[i] = 'C' → seq[i+1] ≠ 'A' ∧ seq[i+1] ≠ 'D') ∧
                     (seq[i] = 'D' → seq[i+1] ≠ 'A' ∧ seq[i+1] ≠ 'C'))

theorem number_of_eight_letter_good_words : 
  let good_words := { seq : List Char | seq.length = 8 ∧ good_word seq } in
  Fintype.card good_words = 2441 :=
sorry

end number_of_eight_letter_good_words_l722_722459


namespace new_bottles_from_recycling_l722_722910

theorem new_bottles_from_recycling (initial_bottles : ℕ) (required_bottles : ℕ) (h : initial_bottles = 125) (r : required_bottles = 5) : 
∃ new_bottles : ℕ, new_bottles = (initial_bottles / required_bottles ^ 2 + initial_bottles / (required_bottles * required_bottles / required_bottles) + initial_bottles / (required_bottles * required_bottles * required_bottles / required_bottles * required_bottles * required_bottles)) :=
  sorry

end new_bottles_from_recycling_l722_722910


namespace time_to_pay_back_l722_722620

def total_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

def monthly_profit := monthly_revenue - monthly_expenses

theorem time_to_pay_back : 
  (total_cost / monthly_profit) = 10 := 
by
  -- Definition of monthly_profit 
  have monthly_profit_def : monthly_profit = 4000 - 1500 := rfl
  rw [monthly_profit_def]
  
  -- Performing the division
  show (25000 / 2500) = 10
  apply div_eq_of_eq_mul
  norm_num
  sorry

end time_to_pay_back_l722_722620


namespace polynomial_inequality_l722_722626

noncomputable def F (x a_3 a_2 a_1 k : ℝ) : ℝ :=
  x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + k^4

theorem polynomial_inequality 
  (p k : ℝ) 
  (a_3 a_2 a_1 : ℝ) 
  (h_p : 0 < p) 
  (h_k : 0 < k) 
  (h_roots : ∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧
    F (-x1) a_3 a_2 a_1 k = 0 ∧
    F (-x2) a_3 a_2 a_1 k = 0 ∧
    F (-x3) a_3 a_2 a_1 k = 0 ∧
    F (-x4) a_3 a_2 a_1 k = 0) :
  F p a_3 a_2 a_1 k ≥ (p + k)^4 := 
sorry

end polynomial_inequality_l722_722626


namespace cone_volume_l722_722196

noncomputable def cone_volume_lateral_surface_area (A : ℝ) (h : Type) 
  [has_pow h] [has_sqrt h] [has_mul h] [has_div h] [has_sub h] [has_add h] : ℝ := sorry

theorem cone_volume (A : ℝ) (h : Type) 
  [has_pow h] [has_sqrt h] [has_mul h] [has_div h] [has_sub h] [has_add h] 
  (h_area : A = 2 * real.pi) : cone_volume_lateral_surface_area A h = (sqrt 3 / 3) * real.pi := sorry

end cone_volume_l722_722196


namespace unit_vectors_dot_product_min_distance_l722_722528

variables {ℝ : Type*} [linear_ordered_field ℝ] [topological_space ℝ] [order_topology ℝ]

noncomputable def min_dist (e1 e2 : ℝ × ℝ) (λ : ℝ) : ℝ :=
real.sqrt ((λ + 1/2)^2 + 3/4)

theorem unit_vectors_dot_product
  (e1 e2 : ℝ × ℝ)
  (h1 : ∥e1∥ = 1)
  (h2 : ∥e2∥ = 1)
  (angle_e1_e2 : abs (real.angle.to_real (real.angle e1 e2)) = real.pi * (2/3)) :
  e1.1 * e2.1 + e1.2 * e2.2 = -1/2 :=
by {
  sorry
}

theorem min_distance
  (e1 e2 : ℝ × ℝ)
  (λ : ℝ)
  (h1 : ∥e1∥ = 1)
  (h2 : ∥e2∥ = 1)
  (angle_e1_e2 : abs (real.angle.to_real (real.angle e1 e2)) = real.pi * (2/3)) :
  ∃ λ_min : ℝ, min_dist e1 e2 λ = real.sqrt 3 / 2 :=
by {
  use -1/2,
  sorry
}

end unit_vectors_dot_product_min_distance_l722_722528


namespace arithmetic_seq_slope_l722_722209

theorem arithmetic_seq_slope {a : ℕ → ℤ} (h : a 2 - a 4 = 2) : ∃ a1 : ℤ, ∀ n : ℕ, a n = -n + (a 1) + 1 := 
by {
  sorry
}

end arithmetic_seq_slope_l722_722209


namespace cos_alpha_minus_pi_eq_l722_722010

theorem cos_alpha_minus_pi_eq :
  (∃ α : ℝ, (π / 2 < α ∧ α < π) ∧ (3 * sin (2 * α) = 2 * cos α)) →
  ∃ α : ℝ, cos (α - π) = (2 * sqrt 2) / 3 :=
by
  intro h
  cases h with α hα
  cases hα with H1 H2
  sorry

end cos_alpha_minus_pi_eq_l722_722010


namespace probability_of_four_digit_number_divisible_by_4_l722_722332

-- Define the set of possible outcomes for each spin
def outcomes := {1, 2, 3, 4}

-- Define a function to check if the given number is divisible by 4
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Define a function to calculate the four-digit number formed by the spins
def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

-- Define the probability of forming a four-digit number divisible by 4
noncomputable def probability_divisible_by_4 : ℚ :=
  let valid_numbers := {n ∈ {four_digit_number a b c d | a, b, c, d ∈ outcomes} | is_divisible_by_4 n}
  in (valid_numbers.to_finset.card : ℚ) / 256

-- The theorem statement
theorem probability_of_four_digit_number_divisible_by_4 : probability_divisible_by_4 = 1 / 4 :=
by sorry

end probability_of_four_digit_number_divisible_by_4_l722_722332


namespace initial_markup_percentage_l722_722421

theorem initial_markup_percentage (C : ℝ) (M : ℝ) 
  (h1 : ∀ S_1 : ℝ, S_1 = C * (1 + M))
  (h2 : ∀ S_2 : ℝ, S_2 = C * (1 + M) * 1.25)
  (h3 : ∀ S_3 : ℝ, S_3 = C * (1 + M) * 1.25 * 0.94)
  (h4 : ∀ S_3 : ℝ, S_3 = C * 1.41) : 
  M = 0.2 :=
by
  sorry

end initial_markup_percentage_l722_722421


namespace min_edge_weights_l722_722231

open Int

-- Given: Numbers from 1 to 8 are assigned to the vertices of a cube
-- Each edge weight is defined as the absolute difference between the numbers at its endpoints
constants (cube_vertices : Finset ℤ) (cube_edges : Finset (ℤ × ℤ))

-- Definition: The set of vertices of the cube
def vertices : Finset ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Definition: The edges of the cube where each edge connects two vertices
def edges : Finset (ℤ × ℤ) :=
  {(1, 2), (2, 3), (3, 4), (4, 1),
   (1, 5), (2, 6), (3, 7), (4, 8),
   (5, 6), (6, 7), (7, 8), (8, 5)}

-- Definition: Compute the weight of an edge as the absolute difference of the numbers at the endpoints
def weight (edge : ℤ × ℤ) : ℤ :=
  abs (edge.1 - edge.2)

-- Definition: The set of edge weights in the cube
def edge_weights : Finset ℤ :=
  edges.image weight

-- Theorem: The minimum number of distinct edge weights is 3
theorem min_edge_weights : edge_weights.card = 3 :=
  sorry

end min_edge_weights_l722_722231


namespace solve_price_of_uniform_l722_722416

-- Define the conditions
def total_salary_one_year := 500
def worked_time := 9 / 12  -- 9 months out of 12
def received_salary := 300

-- Definition of price of the uniform
def price_of_uniform : ℝ := 75

-- Theorem statement
theorem solve_price_of_uniform (total_salary_one_year : ℝ)
    (worked_time received_salary : ℝ) :
    let expected_salary := worked_time * total_salary_one_year in
    let price_of_uniform := expected_salary - received_salary in
    price_of_uniform = 75 := 
by 
  -- placeholder for proof steps
  sorry

end solve_price_of_uniform_l722_722416


namespace volume_of_rectangular_box_l722_722359

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l722_722359


namespace generatrix_length_of_cone_l722_722025

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722025


namespace sum_of_abscissas_l722_722884

open Real

theorem sum_of_abscissas :
  let f1 := λ x : ℝ, 8 * cos (π * x) * cos (2 * π * x)^2 * cos (4 * π * x)
  let f2 := λ x : ℝ, cos (5 * π * x)
  let points := {x | x ∈ Icc (-1) 0 ∧ f1 x = f2 x}
  (∑ x in points.to_finset, x) = -4.5 :=
by 
  let f1 := λ x : ℝ, 8 * cos (π * x) * cos (2 * π * x)^2 * cos (4 * π * x)
  let f2 := λ x : ℝ, cos (5 * π * x)
  let points := {x | x ∈ Icc (-1) 0 ∧ f1 x = f2 x}
  have h: (∑ x in points.to_finset, x) = -4.5,
  {
    sorry
  },
  exact h

end sum_of_abscissas_l722_722884


namespace area_of_triangle_XYZ_l722_722202

section
  variables {X Y Z W : Type*} [Real X] [Real Y] [Real Z] [Real W]
  variables {right_triangle_XYZ : Prop} {right_angle_Y : Prop} {foot_of_altitude : Prop}
  
  def area_of_right_triangle (XZ YW : ℝ) : ℝ := (1 / 2) * XZ * YW

  variables (XW WZ : ℝ)

  theorem area_of_triangle_XYZ (XW_eq : XW = 5) (WZ_eq : WZ = 7)
    (right_triangle_XYZ : right_triangle_XYZ) (right_angle_Y : right_angle_Y)
    (foot_of_altitude : foot_of_altitude) :
  area_of_right_triangle (XZ: 12) (YW: sqrt 35) = 6 * sqrt 35 :=
  sorry
end

end area_of_triangle_XYZ_l722_722202


namespace log_base_4_of_fraction_l722_722472

theorem log_base_4_of_fraction :
  log 4 (1 / 16) = -2 := by
  sorry

end log_base_4_of_fraction_l722_722472


namespace fish_length_l722_722500

theorem fish_length (x : ℝ) (h1 : ∀ t, t = 2 * t) (h2 : ∀ fish, fish = 48) 
                    (h3 : ∀ g, g = 1 * fish) (h4 : ∀ length, length = 3)
                    (h5 : ∀ length2, ∃ fish_count, fish_count = (48 / 3)) (h6 : ∀ length2, ∃ fish_count, fish_count - 1 = 15)
                    (h7 : ∀ tank2_count, tank2_count - 3 = 12) (h8 : ∀ tank2_gal, tank2_gal = 24) 
                    (h9 : ∀ fish_count2, fish_count2 = (24 / x)) : x = 2 :=
by
  sorry

end fish_length_l722_722500


namespace problem_l722_722019

-- Define the necessary conditions
variables {A B C P G : Type} [InnerProductSpace ℝ A]
variables (lambda mu : ℝ)
variables (vector_AB vector_AC vector_AG vector_AP : A)
variables {AB AC AG AP : A}

-- Assume G is the centroid of triangle ABC
def is_centroid (G : A) (A B C : A) : Prop :=
  (G = (A + B + C) / 3)

-- Assume P is an interior point of triangle GBC
def is_interior (P : A) (G B C : A) : Prop :=
  sorry -- Definition of being an interior point

-- Assume the vector equation given in the problem
def vector_eq (AP : A) (lambda : ℝ) (AB : A) (mu : ℝ) (AC : A) : Prop :=
  AP = λ • AB + μ • AC

-- The Lean 4 statement we need to prove
theorem problem (G P A B C : A) (λ μ : ℝ) 
  (hG : is_centroid G A B C) 
  (hP : is_interior P G B C) 
  (hAP : vector_eq (P - A) λ (B - A) μ (C - A)) :
  (2 / 3 : ℝ) < λ + μ ∧ λ + μ < 1 :=
sorry

end problem_l722_722019


namespace distance_between_locations_A_and_B_l722_722346

-- Define the conditions
variables {x y s t : ℝ}

-- Conditions specified in the problem
axiom bus_a_meets_bus_b_after_85_km : 85 / x = (s - 85) / y 
axiom buses_meet_again_after_turnaround : (s - 85 + 65) / x + 1 / 2 = (85 + (s - 65)) / y + 1 / 2

-- The theorem to be proved
theorem distance_between_locations_A_and_B : s = 190 :=
by
  sorry

end distance_between_locations_A_and_B_l722_722346


namespace number_of_workers_l722_722725

theorem number_of_workers (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 350000) : W = 1000 :=
sorry

end number_of_workers_l722_722725


namespace dunkers_starting_lineups_l722_722311

theorem dunkers_starting_lineups (players : Finset ℕ)
  (bob yogi zack : ℕ) (h : {bob, yogi, zack} ⊆ players) (h_card : players.card = 15) :
  (∑ p in (players \ {bob, yogi, zack}).powerset.filter (λs, 4 ≤ s.card ∧ s.card ≤ 5), 
    if 5 = s.card + 1 then 1 else 0) = 2277 := 
begin
  -- Sorry, proof is omitted
  sorry
end

end dunkers_starting_lineups_l722_722311


namespace sum_of_abscissas_l722_722886

theorem sum_of_abscissas
  (f g : ℝ → ℝ)
  (h_eq : ∀ x, f x = 8 * Cos (π * x) * (Cos (2 * π * x))^2 * Cos (4 * π * x))
  (h_eq' : ∀ x, g x = Cos (5 * π * x))
  (h_common_points : ∀ x, f x = g x)
  (h_abscissas : ∀ x, x ∈ Icc (-1 : ℝ) 0) :
  ∑ k in (Finset.filter (λ x : ℝ, h_common_points x) (Finset.Icc (-1) 0)), k = -4.5 :=
sorry

end sum_of_abscissas_l722_722886


namespace trader_cloth_sold_l722_722432

theorem trader_cloth_sold (total_price profit_per_meter cost_price_per_meter : ℕ) 
  (eq_total_price : total_price = 8925) 
  (eq_profit_per_meter : profit_per_meter = 10) 
  (eq_cost_price_per_meter : cost_price_per_meter = 95) : 
  let selling_price_per_meter := cost_price_per_meter + profit_per_meter in
  let x := total_price / selling_price_per_meter in
  x = 85 := 
by
  sorry

end trader_cloth_sold_l722_722432


namespace sum_possible_values_of_m_l722_722789

theorem sum_possible_values_of_m :
  let m_values := Finset.filter (λ m, 5 ≤ m ∧ m ≤ 17) (Finset.range 18)
  Finset.sum m_values id = 143 :=
by
  let m_values := Finset.filter (λ m, 5 ≤ m ∧ m ≤ 17) (Finset.range 18)
  have h : Finset.sum m_values id = 143 := sorry
  exact h

end sum_possible_values_of_m_l722_722789


namespace least_possible_pieces_l722_722768

-- Define the convex 2019-gon and the form of the diagonals
def is_convex_2019_gon (vertices : Fin 2019 → α) (polygon : Finset (Fin 2019)) : Prop :=
  polygon.card = 2019 ∧ ∀ i : Fin 2019, ∃ j : Fin 2019, j = (i + 3) % 2019

-- The least possible number of pieces when cutting the convex 2019-gon according to diagonals of the form A_i A_i+3
theorem least_possible_pieces (vertices : Fin 2019 → α) (polygon : Finset (Fin 2019))
  (h_convex : is_convex_2019_gon vertices polygon) : 
  Σ B, B.card = 5049 :=
sorry

end least_possible_pieces_l722_722768


namespace count_cube_roots_less_than_15_l722_722123

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l722_722123


namespace sqrt_recursive_value_l722_722842

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722842


namespace quadratic_non_real_roots_l722_722165

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722165


namespace vaishali_four_stripe_hats_l722_722349

def hat_stripes (num_three_stripe_hats : Nat) (num_four_stripe_hats : Nat) (num_no_stripe_hats : Nat) (num_five_stripe_hats : Nat) : Nat :=
  num_three_stripe_hats * 3 + num_four_stripe_hats * 4 + num_no_stripe_hats * 0 + num_five_stripe_hats * 5

theorem vaishali_four_stripe_hats :
  ∀ (num_four_stripe_hats : Nat),
  hat_stripes 4 num_four_stripe_hats 6 2 = 34 → num_four_stripe_hats = 3 :=
by
  assume n : Nat
  intro h : hat_stripes 4 n 6 2 = 34
  sorry

end vaishali_four_stripe_hats_l722_722349


namespace cone_generatrix_length_is_2sqrt2_l722_722030

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722030


namespace quadratic_non_real_roots_iff_l722_722154

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722154


namespace even_function_period_pi_div_2_l722_722440

noncomputable def f (x : ℝ) : ℝ := sin (4 * x)
noncomputable def g (x : ℝ) : ℝ := tan (2 * x)
noncomputable def h (x : ℝ) : ℝ := cos (2 * x) ^ 2 - sin (2 * x) ^ 2
noncomputable def k (x : ℝ) : ℝ := cos (2 * x)

theorem even_function_period_pi_div_2 :
  (∀ x : ℝ, h x = h (-x)) ∧ (∃ T > 0, ∀ x : ℝ, h (x + T) = h x ∧ T = π / 2) :=
by
  sorry

end even_function_period_pi_div_2_l722_722440


namespace find_a_plus_b_l722_722531

open Real

-- Definition and properties of functions f and g
def f (x : ℝ) (a : ℝ) : ℝ := (9^x - a) / (3^x)
def g (x : ℝ) (b : ℝ) : ℝ := log (10^x + 1) + b * x

-- Additional properties derived from the conditions
def f_odd (a : ℝ) : Prop := ∀ x : ℝ, f (-x) a = - f x a
def g_even (b : ℝ) : Prop := ∀ x : ℝ, g (-x) b = g x b

-- Main theorem statement
theorem find_a_plus_b (a b : ℝ) (h1 : f_odd a) (h2 : g_even b) : a + b = 1 / 2 := by
  sorry

end find_a_plus_b_l722_722531


namespace no_positive_integer_n_l722_722552

theorem no_positive_integer_n (n : ℕ) : 
  ∃ (x : ℕ), x = (sqrt (n + sqrt (n + sqrt (n + sqrt n)))) → x = 0 :=
by
  sorry

end no_positive_integer_n_l722_722552


namespace cone_generatrix_length_l722_722058

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722058


namespace max_profit_l722_722708

noncomputable def profit (x : ℝ) : ℝ :=
  -160 * x^2 + 560 * x + 3120

theorem max_profit :
  (profit 1.5 = 3600) ∧ (profit 2 = 3600) :=
by
  simp [profit]
  split
  {
    have : -160 * (1.5)^2 + 560 * 1.5 + 3120 = 3600, sorry
    exact this
  }
  {
    have : -160 * (2)^2 + 560 * 2 + 3120 = 3600, sorry
    exact this
  }

end max_profit_l722_722708


namespace parabola_translation_l722_722436

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := (x - 1) ^ 2 + 5
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 3

-- Statement of the translation problem in Lean 4
theorem parabola_translation :
  ∀ x : ℝ, g x = f (x + 2) - 3 := 
sorry

end parabola_translation_l722_722436


namespace cube_root_numbers_less_than_15_l722_722102

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l722_722102


namespace parallel_lines_l722_722252

noncomputable def circles (Γ1 Γ2 : Circle) : Prop :=
∃ A B P Q P' Q' : Point,
  (A ∈ Γ1) ∧ (A ∈ Γ2) ∧ 
  (B ∈ Γ1) ∧ (B ∈ Γ2) ∧
  (P ∈ Γ1) ∧ (Q ∈ Γ2) ∧
  (P' ∈ Γ1) ∧ (Q' ∈ Γ2) ∧
  collinear [A, P, Q] ∧
  collinear [B, P', Q'] ∧
  collinear [P', P, A] ∧
  collinear [Q', Q, A]

theorem parallel_lines (Γ1 Γ2 : Circle) (A B P Q P' Q' : Point)
  (hΓ1 : A ∈ Γ1)
  (hΓ2 : A ∈ Γ2)
  (hBΓ1 : B ∈ Γ1)
  (hBΓ2 : B ∈ Γ2)
  (hPΓ1 : P ∈ Γ1)
  (hQΓ2 : Q ∈ Γ2)
  (hP'Γ1 : P' ∈ Γ1)
  (hQ'Γ2 : Q' ∈ Γ2)
  (hCol1 : collinear [A, P, Q])
  (hCol2 : collinear [B, P', Q'])
  (hCol3 : collinear [P', P, A])
  (hCol4 : collinear [Q', Q, A]) :
  parallel (lineThrough P P') (lineThrough Q Q') := sorry

end parallel_lines_l722_722252


namespace number_of_distinct_prime_factors_30_fact_l722_722995

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722995


namespace variance_of_transformed_data_l722_722926

-- Define the variance function
def variance (n : ℕ) (x : Fin n → ℝ) : ℝ := 
  let μ := (∑ i, x i) / n
  (∑ i, (x i - μ) ^ 2) / n

-- Define the problem in Lean 4
theorem variance_of_transformed_data (n : ℕ) (x : Fin n → ℝ) (h : variance n (λ i, 2 * x i - 1) = 4) : variance n x = 1 :=
sorry

end variance_of_transformed_data_l722_722926


namespace minimum_value_of_quadratic_l722_722811

-- Definition of the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 6 * x + 13

-- Statement of the proof problem
theorem minimum_value_of_quadratic : ∃ (y : ℝ), ∀ x : ℝ, quadratic x >= y ∧ y = 4 := by
  sorry

end minimum_value_of_quadratic_l722_722811


namespace solution_set_of_inequality_l722_722638

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, x < y → x < 0 → f x < f y) ∧
  (f (-1) = 0) →
  {x : ℝ | x * (f x + f (-x)) < 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (1 < x)} :=
by
  intros,
  sorry

end solution_set_of_inequality_l722_722638


namespace valid_votes_B_l722_722590

theorem valid_votes_B (total_votes : ℕ) (invalid_percentage : ℕ) (a_exceeds_b_percentage : ℕ) (c_valid_percentage : ℕ) :
  total_votes = 10000 ->
  invalid_percentage = 20 ->
  a_exceeds_b_percentage = 10 ->
  c_valid_percentage = 5 ->
  ∃ (valid_votes_B : ℕ),
  let valid_votes := (100 - invalid_percentage) * total_votes / 100 in
  let V_C := c_valid_percentage * valid_votes / 100 in
  let V_A := valid_votes_B + a_exceeds_b_percentage * total_votes / 100 in
  V_A + valid_votes_B + V_C = valid_votes ∧ valid_votes_B = 3300 :=
begin
  sorry
end

end valid_votes_B_l722_722590


namespace regression_equation_negatively_correlated_l722_722535

theorem regression_equation_negatively_correlated (x y : ℝ) (neg_corr : negatively_correlated x y) 
(sample_means : (3, 3.5)) :
  let regression_eq1 : ℝ → ℝ := λ x, 0.4 * x + 2.3
  let regression_eq2 : ℝ → ℝ := λ x, 2 * x - 2.4
  let regression_eq3 : ℝ → ℝ := λ x, -2 * x + 9.5
  let regression_eq4 : ℝ → ℝ := λ x, -0.4 * x + 4.4
  regression_eq3 = -2 * x + 9.5 := sorry

end regression_equation_negatively_correlated_l722_722535


namespace non_real_roots_interval_l722_722179

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722179


namespace part_a_part_b_part_c_l722_722622

noncomputable def probability_of_meeting : ℚ :=
  let total_area := 60 * 60
  let meeting_area := 2 * (1/2 * 50 * 50)
  meeting_area / total_area

theorem part_a : probability_of_meeting = 11 / 36 := by
  sorry

noncomputable def probability_of_meeting_b : ℚ :=
  let total_area := 30 * 60
  let meeting_area := 2 * (1/2 * 20 * 30 - (1/2 * 10 * 10))
  meeting_area / total_area

theorem part_b : probability_of_meeting_b = 1 / 6 := by
  sorry

noncomputable def probability_of_meeting_c : ℚ :=
  let total_area := 50 * 60
  let meeting_area := (40 * 10 + 1/2 * 10 * 10 + 1/2 * 10 * 10)
  meeting_area / total_area

theorem part_c : probability_of_meeting_c = 3 / 200 := by
  sorry

end part_a_part_b_part_c_l722_722622


namespace generatrix_length_l722_722040

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722040


namespace volume_ratio_l722_722347

-- Given conditions of cones C and D
def r_C : ℝ := 10
def h_C : ℝ := 40
def r_D : ℝ := 2 * r_C
def h_D : ℝ := 2 * h_C

-- Volumes of cones C and D
def volume_C := (1 / 3) * π * r_C^2 * h_C
def volume_D := (1 / 3) * π * r_D^2 * h_D

-- Target theorem to prove
theorem volume_ratio : volume_C / volume_D = 1 / 8 := by sorry

end volume_ratio_l722_722347


namespace nested_sqrt_eq_l722_722849

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722849


namespace staff_member_pays_l722_722771

variable (d : ℝ)

def dress_price_discounted (d : ℝ) : ℝ := 1.60 * d
def staff_discount (d : ℝ) : ℝ := 0.20 * d
def price_after_staff_discount (d : ℝ) : ℝ := 1.40 * d
def sales_tax (price : ℝ) : ℝ := 0.10 * price
def total_price_with_tax (d : ℝ) : ℝ := price_after_staff_discount d + sales_tax (price_after_staff_discount d)
def environmental_surcharge (d : ℝ) : ℝ := if d > 100 then 0.15 * d else 0

noncomputable def final_price (d : ℝ) : ℝ := total_price_with_tax d + environmental_surcharge d

theorem staff_member_pays (d : ℝ) :
  (d ≤ 100 → final_price d = 1.54 * d) ∧
  (d > 100 → final_price d = 1.69 * d) :=
by {
  sorry
}

end staff_member_pays_l722_722771


namespace distance_point_to_line_l722_722676

theorem distance_point_to_line : 
  let d := |(3 * 1 - 4 * (-1) + 3 : ℝ)| / Real.sqrt (3^2 + (-4)^2) in
  d = 2 :=
by
  -- Define the point (1, -1) and the line 3x - 4y + 3 = 0
  let point := (1 : ℝ, -1 : ℝ)
  let A := (3 : ℝ)
  let B := (-4 : ℝ)
  let C := (3 : ℝ)

  -- Calculate the distance using the distance formula
  let d := |(A * point.1 + B * point.2 + C)| / Real.sqrt (A^2 + B^2)
  
  -- Show that the calculated distance is 2
  have : d = 2 := by sorry
  exact this

end distance_point_to_line_l722_722676


namespace line_equation_l722_722778

theorem line_equation (a b : ℝ) (h1 : (1, 2) ∈ line) (h2 : ∃ a b : ℝ, b = 2 * a ∧ line = {p : ℝ × ℝ | p.1 / a + p.2 / b = 1}) :
  line = {p : ℝ × ℝ | 2 * p.1 - p.2 = 0} ∨ line = {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0} :=
sorry

end line_equation_l722_722778


namespace apples_in_bowl_l722_722763

variable {A : ℕ}

theorem apples_in_bowl
  (initial_oranges : ℕ)
  (removed_oranges : ℕ)
  (final_oranges : ℕ)
  (total_fruit : ℕ)
  (fraction_apples : ℚ) :
  initial_oranges = 25 →
  removed_oranges = 19 →
  final_oranges = initial_oranges - removed_oranges →
  fraction_apples = (70 : ℚ) / (100 : ℚ) →
  final_oranges = total_fruit * (30 : ℚ) / (100 : ℚ) →
  A = total_fruit * fraction_apples →
  A = 14 :=
by
  sorry

end apples_in_bowl_l722_722763


namespace sqrt_recursive_value_l722_722845

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722845


namespace new_boat_distance_l722_722420

theorem new_boat_distance (old_boat_distance : ℝ) (increase_percentage : ℝ) :
  increase_percentage = 0.30 → old_boat_distance = 150 → 
  let new_boat_distance := old_boat_distance * (1 + increase_percentage) in
  new_boat_distance = 195 :=
by
  intros h1 h2
  let new_boat_distance := old_boat_distance * (1 + increase_percentage)
  sorry

end new_boat_distance_l722_722420


namespace passenger_decision_time_l722_722675

/-
Problem conditions
-/

def distance_between_A_B : ℝ := 90
def speed_A_to_B_initial : ℝ := 60
def interval_B_to_A : ℝ := 15 / 60 -- converting minutes to hours
def speed_B_to_A : ℝ := 80
def breakdown_distance : ℝ := 45
def speed_after_breakdown : ℝ := 20
def initial_time_A_to_breakdown : ℝ := breakdown_distance / speed_A_to_B_initial

/-
Expected answer
-/
def decision_time : ℝ := 3 / 60 -- converting minutes to hours

/-
Lean statement for the proof problem
-/

theorem passenger_decision_time :
  ∀ (distance_between_A_B = 90) 
    (speed_A_to_B_initial = 60)
    (interval_B_to_A = 15 / 60)
    (speed_B_to_A = 80)
    (breakdown_distance = 45)
    (speed_after_breakdown = 20),
    have initial_time_A_to_breakdown : initial_time_A_to_breakdown = (breakdown_distance / speed_A_to_B_initial),
    have distance_covered_due_speed : (speed_A_to_B_initial * initial_time_A_to_breakdown) = breakdown_distance,
    have remaining_distance : distance_between_A_B - breakdown_distance,
    have distance_traveled_9am_bus : speed_B_to_A * initial_time_A_to_breakdown = 0.75 * 80,
    have distance_covered_delay_bus : (speed_B_to_A * (0.75 - 0.25)),
    have distance_btwn_11am_bus : (90 - (45 + 40)) = 5,
    have rel_speed : ((speed_after_breakdown + speed_B_to_A)),
    decision_time = (distance_btwn_11am_bus / rel_speed).
  sorry

end passenger_decision_time_l722_722675


namespace sale_in_first_month_l722_722411

theorem sale_in_first_month 
  (sale_2 : ℝ) (sale_3 : ℝ) (sale_4 : ℝ) (sale_5 : ℝ) (sale_6 : ℝ) (avg_sale : ℝ)
  (h_sale_2 : sale_2 = 5366) (h_sale_3 : sale_3 = 5808) 
  (h_sale_4 : sale_4 = 5399) (h_sale_5 : sale_5 = 6124) 
  (h_sale_6 : sale_6 = 4579) (h_avg_sale : avg_sale = 5400) :
  ∃ (sale_1 : ℝ), sale_1 = 5124 :=
by
  let total_sales := avg_sale * 6
  let known_sales := sale_2 + sale_3 + sale_4 + sale_5 + sale_6
  have h_total_sales : total_sales = 32400 := by sorry
  have h_known_sales : known_sales = 27276 := by sorry
  let sale_1 := total_sales - known_sales
  use sale_1
  have h_sale_1 : sale_1 = 5124 := by sorry
  exact h_sale_1

end sale_in_first_month_l722_722411


namespace locus_of_tangents_l722_722905

/-
Given:
1. A circle with center O and radius r.
2. A segment length d.
The proof is to show that the locus of points from which tangents of length d to this circle can be drawn forms a circle concentric with the original one, with radius sqrt(r^2 + d^2).
-/

theorem locus_of_tangents (O : Point) (r d : ℝ) (h_r_pos : 0 < r) (h_d_pos : 0 < d) :
  ∃ (C : Circle), C.center = O ∧ C.radius = real.sqrt (r^2 + d^2) :=
sorry

end locus_of_tangents_l722_722905


namespace sequence_to_100_l722_722229

theorem sequence_to_100 : 
  ∃ s : List String, (s = ["123", "-", "45", "-", "67", "+", "89"]) ∧ (eval_expr s = 100) :=
sorry

-- Auxiliary function (not part of the proof statement) that evaluates the expression from a list of strings
def eval_expr (s : List String) : Int :=
  let join := s.foldl (fun acc x => acc ++ x) ""
  -- eval the expression from the concatenated string
  let parsed_expr := (Lean.parseExpr join).toOption.getD 0 -- we assume parsed_expr to be evaluated
  parsed_expr

end sequence_to_100_l722_722229


namespace largest_square_side_length_largest_rectangle_dimensions_l722_722512

variables {a b : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- Part (a)
theorem largest_square_side_length :
  let s := (a * b) / (a + b) in
  s = (a * b) / (a + b) :=
sorry

-- Part (b)
theorem largest_rectangle_dimensions :
  let x := a / 2
  let y := b / 2 in
  (x, y) = (a / 2, b / 2) :=
sorry

end largest_square_side_length_largest_rectangle_dimensions_l722_722512


namespace male_athletes_drawn_l722_722589

theorem male_athletes_drawn (total_males : ℕ) (total_females : ℕ) (total_sample : ℕ)
  (h_males : total_males = 20) (h_females : total_females = 10) (h_sample : total_sample = 6) :
  (total_sample * total_males) / (total_males + total_females) = 4 := 
  by
  sorry

end male_athletes_drawn_l722_722589


namespace area_three_arcs_correct_l722_722689

noncomputable def area_of_three_arcs (BC : ℝ) (X midpoint of AB : Prop) (Y midpoint of AC : Prop) (A midpoint of BC : Prop) : ℝ :=
  -- Mention that BC = 1
  let BC_length := (BC = 1) in
  -- Use the given and derived lengths and angles to calculate the area
  let angle_BAC := (36 : ℝ) * real.pi / 180 in
  let angle_BXC := (72 : ℝ) * real.pi / 180 in
  let AC := (sqrt 5 + 1) / 2 in
  let T_1 := real.pi / 20 * (3 + sqrt 5) in
  let T_2 := (3 * real.pi / 10) - (sqrt (10 + 2 * sqrt 5)) / 8 in
  let area := T_1 + 2 * T_2 in
  area

theorem area_three_arcs_correct :
  ∀ BC X midpoint of AB Y midpoint of AC A midpoint of BC,
    BC = 1 →
    area_of_three_arcs BC X Y A = 1.756 :=
by
  intros
  sorry

end area_three_arcs_correct_l722_722689


namespace product_of_tangents_is_constant_l722_722946

theorem product_of_tangents_is_constant (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ)
  (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (A1 A2 : ℝ × ℝ)
  (hA1 : A1 = (-a, 0))
  (hA2 : A2 = (a, 0)) :
  ∃ (Q1 Q2 : ℝ × ℝ),
  (A1.1 - Q1.1, A2.1 - Q2.1) = (b^2, b^2) :=
sorry

end product_of_tangents_is_constant_l722_722946


namespace quadratic_non_real_roots_l722_722164

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722164


namespace sum_of_inradii_correct_l722_722568

noncomputable def sum_of_inradii (AB AC BC : ℝ) (D_midpoint : BC / 2) : ℝ :=
  let s := (AB + AC + BC) / 2
  let area := Math.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r_ABD := area / ((AB + D_midpoint + Math.sqrt((AB^2 * D_midpoint + AC^2 * D_midpoint - D_midpoint^3) / BC)) / 2)
  let r_ADC := area / ((AC + D_midpoint + Math.sqrt((AB^2 * D_midpoint + AC^2 * D_midpoint - D_midpoint^3) / BC)) / 2)
  r_ABD + r_ADC

theorem sum_of_inradii_correct :
  ∀ (AB AC BC : ℝ), AB = 7 → AC = 9 → BC = 12 →
  sum_of_inradii AB AC BC (BC / 2) = 7 * Real.sqrt 20 / (13 + Real.sqrt 29) :=
by
  sorry

end sum_of_inradii_correct_l722_722568


namespace common_ratio_of_geometric_series_l722_722894

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l722_722894


namespace payback_time_l722_722618

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l722_722618


namespace find_f_24_25_26_l722_722679

-- Given conditions
def homogeneous (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (n a b c : ℤ), f (n * a) (n * b) (n * c) = n * f a b c

def shift_invariance (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c n : ℤ), f (a + n) (b + n) (c + n) = f a b c + n

def symmetry (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), f a b c = f c b a

-- Proving the required value under the conditions
theorem find_f_24_25_26 (f : ℤ → ℤ → ℤ → ℝ)
  (homo : homogeneous f) 
  (shift : shift_invariance f) 
  (symm : symmetry f) : 
  f 24 25 26 = 25 := 
sorry

end find_f_24_25_26_l722_722679


namespace nested_sqrt_eq_l722_722851

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722851


namespace sqrt_continued_fraction_l722_722864

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722864


namespace first_day_is_sunday_l722_722669

noncomputable def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_is_sunday :
  (day_of_week 18 = "Wednesday") → (day_of_week 1 = "Sunday") :=
by
  intro h
  -- proof would go here
  sorry

end first_day_is_sunday_l722_722669


namespace number_of_zeros_of_f_in_interval_l722_722686

theorem number_of_zeros_of_f_in_interval :
  let f (x : ℝ) := 2^x + x^3 - 2
   in (∃! x ∈ Ioo 0 1, f x = 0) :=
by 
  sorry

end number_of_zeros_of_f_in_interval_l722_722686


namespace placing_4_balls_into_4_boxes_with_2_empty_boxes_l722_722340

theorem placing_4_balls_into_4_boxes_with_2_empty_boxes :
  ∃ (n : ℕ), n = 84 ∧ 
  (∀ (balls boxes : Fin 4 → ℕ), 
   (∀ i, boxes i < 2) ∧ 
   (∃ (empty_boxes : Finset (Fin 4)), empty_boxes.card = 2) → 
   ∃ (placements : Fin 4 → Fin 4), 
     (∀ j, (Finset.univ.filter (λ i, placements i = j)).card < 2)
  ) :=
sorry

end placing_4_balls_into_4_boxes_with_2_empty_boxes_l722_722340


namespace evaluate_nested_radical_l722_722855

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722855


namespace systematic_sampling_l722_722791

variable (num_employees groups sampled_employees drawn_from_fifth drawn_from_tenth : ℕ)

-- Given conditions
def conditions : Prop :=
  num_employees = 200 ∧
  groups = 40 ∧
  sampled_employees = 40 ∧
  (∀ j, 1 ≤ j ∧ j ≤ groups → ∃ n, n = 5 * (j - 1) + 1 ∧ n + 4 = 5 * j) ∧
  drawn_from_fifth = 22

-- Theorem statement
theorem systematic_sampling :
  conditions num_employees groups sampled_employees drawn_from_fifth drawn_from_tenth →
  drawn_from_tenth = 47 :=
begin
  sorry
end

end systematic_sampling_l722_722791


namespace residue_of_neg_1237_mod_29_l722_722823

theorem residue_of_neg_1237_mod_29 :
  (-1237 : ℤ) % 29 = 10 :=
sorry

end residue_of_neg_1237_mod_29_l722_722823


namespace diameters_intersect_on_or_within_curve_of_constant_width_l722_722286

theorem diameters_intersect_on_or_within_curve_of_constant_width
    (C : Set Point) (hC : is_curve_of_constant_width C)
    (d₁ d₂ : Line) (hd₁ : is_diameter d₁ C) (hd₂ : is_diameter d₂ C) :
    ∃ A : Point, (A ∈ C ∧ A ∈ d₁ ∧ A ∈ d₂) ∨ 
    (A ∈ C ∧ A ∈ d₁ ∧ A ∈ d₂ ∧ is_corner_point A C ∧ ext_angle_A_C ≥ angle_between d₁ d₂) := sorry

end diameters_intersect_on_or_within_curve_of_constant_width_l722_722286


namespace wire_cut_problem_l722_722792

theorem wire_cut_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq_area : (a / 4) ^ 2 = π * (b / (2 * π)) ^ 2) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_problem_l722_722792


namespace right_triangle_short_leg_l722_722583

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l722_722583


namespace initial_water_amount_l722_722968

def water_loss_per_hour : ℕ := 2

def water_addition (hour : ℕ) : ℤ :=
  if hour = 3 then 1
  else if hour = 4 then 3
  else 0

def water_left_end_of_hour_4 (initial_amount : ℕ) : ℤ :=
  initial_amount - 4 * water_loss_per_hour + water_addition 3 + water_addition 4

theorem initial_water_amount (initial_amount : ℕ) (water_left : ℤ) :
  water_left_end_of_hour_4 initial_amount = water_left → initial_amount = 40 :=
by
  intro h
  have water_left_def : water_left_end_of_hour_4 initial_amount = initial_amount - 4 * 2 + 1 + 3 := by refl
  rw [water_left_def, h]
  sorry

end initial_water_amount_l722_722968


namespace math_problem_l722_722283

variables (P Q R S : Type) [linear_ordered_field P] 
variables (a b c k : P) 

-- Conditions
def collinear (P Q R S : Type) : Prop := sorry
def distinct_points (P Q R S : Type) : Prop := sorry
def positive_numbers (a b c k : P): Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0
def distances (a b c k : P): Prop := b = a + k ∧ b < 2 * a + c / 2 ∧ a < c / 2

-- Proof problem
theorem math_problem (P Q R S : Type) [linear_ordered_field P] (a b c k : P) 
  (h1 : collinear P Q R S) 
  (h2 : distinct_points P Q R S)
  (h3 : positive_numbers a b c k)
  (h4 : distances a b c k) : 
  (a < c / 2) ∧ (b < 2 * a + c / 2) :=
sorry

end math_problem_l722_722283


namespace log_four_one_div_sixteen_l722_722467

theorem log_four_one_div_sixteen : log 4 (1 / 16) = -2 := 
by 
  sorry

end log_four_one_div_sixteen_l722_722467


namespace order_of_M_N_P_Q_l722_722920

theorem order_of_M_N_P_Q (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  let M := |x|
      N := |y|
      P := (|x + y|) / 2
      Q := Real.sqrt (x * y)
  in N < Q ∧ Q < P ∧ P < M :=
by
  sorry

end order_of_M_N_P_Q_l722_722920


namespace find_number_of_z_l722_722774

open Complex

noncomputable def f (z : ℂ) : ℂ := I * conj z

theorem find_number_of_z : {z : ℂ // ↑(normSq z) = 64 ∧ f z = z}.finite.card = 2 := 
by
  sorry

end find_number_of_z_l722_722774


namespace quadratic_non_real_roots_l722_722168

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722168


namespace tanya_made_an_error_l722_722721

theorem tanya_made_an_error :
  let all_numbers := {n | 1 ≤ n ∧ n ≤ 333}
  let divisible_by_3 := {n // n % 3 = 0}
  let divisible_by_7 := {n // n % 7 = 0}
  let divisible_by_21 := {n // n % 21 = 0}
  let divisible_by_3_not_7 := divisible_by_3.diff divisible_by_21
  let divisible_by_7_not_3 := divisible_by_7.diff divisible_by_21
  let excluded_numbers := divisible_by_3_not_7 ∪ divisible_by_7_not_3
  let remaining_numbers := all_numbers.diff excluded_numbers
  in remaining_numbers.card = 205 :=
sorry

end tanya_made_an_error_l722_722721


namespace arrange_data_in_ascending_order_l722_722003

noncomputable def data_set : set ℕ := {1, 1, 3, 3}

theorem arrange_data_in_ascending_order (x1 x2 x3 x4 : ℕ)
  (h_pos : x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0)
  (h_mean : (x1 + x2 + x3 + x4) / 4 = 2)
  (h_median : list.median [x1, x2, x3, x4] = 2)
  (sigma : ℝ := stddev ({x1, x2, x3, x4} : set ℕ))
  (h_sigma : sigma = 1) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 3 ∧ x4 = 3) ∨ (x1 = 3 ∧ x2 = 3 ∧ x3 = 1 ∧ x4 = 1) := by
  sorry

end arrange_data_in_ascending_order_l722_722003


namespace nat_lemma_l722_722887

theorem nat_lemma (a b : ℕ) : (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) → (a = 1 ∧ b = 1) := by
  sorry

end nat_lemma_l722_722887


namespace non_real_roots_interval_l722_722183

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722183


namespace train_length_l722_722790

-- Define the conditions given in the problem
def speed_kmph : ℝ := 60
def time_seconds : ℝ := 4

-- Define conversion factor from km/hr to m/s
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Define the speed in m/s
def speed_mps := kmph_to_mps speed_kmph

-- Define the formula for distance
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The proof goal is to show that given these conditions, the length of the train is 66.68 meters
theorem train_length :
  distance speed_mps time_seconds = 66.68 :=
by
  -- The proof would go here
  sorry

end train_length_l722_722790


namespace reeya_weighted_average_is_correct_l722_722648

def scores : List ℚ := [55, 67, 76, 82, 85, 48, 150]
def total_possible : List ℚ := [100, 100, 100, 100, 100, 60, 200]

noncomputable def weighted_average (scores : List ℚ) (total_possible : List ℚ) : ℚ :=
let converted_scores := scores.zipWith (λ s t, (s / t) * 100) total_possible
let total_scores := converted_scores.sum
let total_points := total_possible.map (λ t, 100).sum
(total_scores / total_points) * 100

theorem reeya_weighted_average_is_correct :
  weighted_average scores total_possible = 74.29 :=
by
  sorry

end reeya_weighted_average_is_correct_l722_722648


namespace problem_statement_l722_722647

noncomputable def cyclic_quadrilateral (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
∃ circle : A × B × C × D → Prop, ∀ a b c d, circle a b c d

noncomputable def tangent_intersection (A B C D P : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] : Prop :=
∃ tA tC : P, ∀ p, is_tangent A p tA ∧ is_tangent C p tC ∧ collinear P B D

theorem problem_statement
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (h1 : cyclic_quadrilateral A B C D)
  (P : Type) [metric_space P]
  (h2 : tangent_intersection A B C D P) :
  AB * CD = BC * AD :=
by sorry

end problem_statement_l722_722647


namespace sum_of_x_values_l722_722306

theorem sum_of_x_values (x y : ℕ) (h : xy - 5 * x + 2 * y = 30) : 
  ∑ i in {x | ∃ y, xy - 5 * x + 2 * y = 30 ∧ 0 < x ∧ 0 < y}, i = 31 :=
sorry

end sum_of_x_values_l722_722306


namespace non_real_roots_interval_l722_722180

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722180


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722105

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722105


namespace minimum_words_for_score_80_l722_722494

-- Define constants and variables
def total_words : ℕ := 750
def relevance_percentage : ℝ := 0.30
def full_marks := 1.0
def partial_marks := 0.5

-- Minimum number of words needed to score at least 80%
def min_words_needed : ℕ := 574

-- Exam score calculation formula
def exam_score (x : ℕ) : ℝ :=
  (x + partial_marks * relevance_percentage * (total_words - x).to_float) / total_words.to_float

-- Proof statement
theorem minimum_words_for_score_80 :
  ∀ (x : ℕ), x >= min_words_needed → exam_score x ≥ 0.8 :=
by
  intros x hx
  unfold exam_score
  sorry

end minimum_words_for_score_80_l722_722494


namespace cone_generatrix_length_l722_722051

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722051


namespace positive_whole_numbers_with_cube_roots_less_than_15_l722_722120

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l722_722120


namespace IMO1989_30th_shortlisted_counterexample_l722_722250

structure LatticePoint (n m : ℤ) : Prop :=
(coord : ℤ × ℤ)

def points_on_line_segment (a b : LatticePoint) : set LatticePoint :=
  {p | ∃ t ∈ (0:ℚ)..1, p.coord.1 = round ((1 - t) * a.coord.1 + t * b.coord.1) ∧ p.coord.2 = round ((1 - t) * a.coord.2 + t * b.coord.2)}

theorem IMO1989_30th_shortlisted (L : set LatticePoint) (A B C : LatticePoint) (hA : A ∈ L) (hB : B ∈ L) (hC : C ∈ L) :
  ∃ (D : LatticePoint) (hD : D ∈ L) (hD_distinct : D ≠ A ∧ D ≠ B ∧ D ≠ C),
    ∀ (P : LatticePoint), P ∈ points_on_line_segment A D ∨ P ∈ points_on_line_segment B D ∨ P ∈ points_on_line_segment C D → P = A ∨ P = B ∨ P = C → P ∉ points_on_line_segment A D ∧ P ∉ points_on_line_segment B D ∧ P ∉ points_on_line_segment C D := sorry

theorem counterexample (A B C D : LatticePoint) :
  ∃ E : LatticePoint, E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D ∧ _root_.points_on_line_segment A E ∩ LatticePoint ∅ = ∅ :=
sorry

end IMO1989_30th_shortlisted_counterexample_l722_722250


namespace correct_calculation_l722_722373

variable (a : ℝ)

theorem correct_calculation : (-2 * a)^3 = -8 * a^3 :=
by
  sorry

end correct_calculation_l722_722373


namespace non_real_roots_bounded_l722_722162

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722162


namespace real_possible_b_values_quadratic_non_real_roots_l722_722142

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722142


namespace time_to_cross_lake_one_direction_l722_722707

-- Definitions for our conditions
def cost_per_hour := 10
def total_cost_round_trip := 80

-- Statement we want to prove
theorem time_to_cross_lake_one_direction : (total_cost_round_trip / cost_per_hour) / 2 = 4 :=
  by
  sorry

end time_to_cross_lake_one_direction_l722_722707


namespace max_value_of_function_l722_722522

theorem max_value_of_function (x : ℝ) (h : x < 1 / 2) : 
  ∃ y, y = 2 * x + 1 / (2 * x - 1) ∧ y ≤ -1 :=
by
  sorry

end max_value_of_function_l722_722522


namespace quadratic_non_real_roots_iff_l722_722150

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722150


namespace annuity_double_duration_approx_l722_722662

theorem annuity_double_duration_approx (r : ℝ) (p : ℝ) (x : ℝ) (e : ℝ) 
  (h_interest : p = 0.04) (h_duration : ∀ (t : ℝ), PV r p t = (r / e^t) * ((e^t) - 1) / (e - 1))
  (h_new_duration : ∀ (t : ℝ), PV (2 * r) p t = (2 * r / e^t) * ((e^t) - 1) / (e - 1))
  (h_equiv : ∀ (t1 t2 : ℝ), PV r p t1 = PV (2 * r) p t2) :
  x ≈ 8 :=
by sorry

-- Additional assumptions / definitions --
noncomputable def PV (r : ℝ) (p : ℝ) (t : ℝ) : ℝ := sorry -- placeholder for Present Value formula

end annuity_double_duration_approx_l722_722662


namespace find_candies_thursday_l722_722828

-- declaring a noncomputable theory
noncomputable theory

-- defining variables according to the given conditions
def num_candies_bought_tuesday : ℕ := 3
def num_candies_bought_friday : ℕ := 2
def num_candies_left : ℕ := 4
def num_candies_eaten : ℕ := 6

-- defining total number of candies
def total_candies : ℕ := num_candies_eaten + num_candies_left

-- defining known purchases
def total_known_purchases : ℕ := num_candies_bought_tuesday + num_candies_bought_friday

-- defining how many candies were bought on Thursday (this is what we're proving)
def num_candies_bought_thursday : ℕ := total_candies - total_known_purchases

-- theorem stating the desired proof
theorem find_candies_thursday : num_candies_bought_thursday = 5 := 
by
  simp [num_candies_bought_thursday, total_candies, total_known_purchases, num_candies_bought_tuesday, num_candies_bought_friday, num_candies_left, num_candies_eaten]
  sorry

end find_candies_thursday_l722_722828


namespace quadratic_non_real_roots_iff_l722_722155

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722155


namespace shortest_tangent_length_l722_722257

noncomputable def circle_center1 : (ℝ × ℝ) := (8, 5)
noncomputable def circle_center2 : (ℝ × ℝ) := (-12, 0)

noncomputable def radius1 : ℝ := 7
noncomputable def radius2 : ℝ := 8

noncomputable def distance_AB : ℝ := real.sqrt (20^2 + 5^2)

theorem shortest_tangent_length :
  let C1 := (x - circle_center1.1)^2 + (y - circle_center1.2)^2 = radius1^2,
      C2 := (x + circle_center2.1)^2 + y^2 = radius2^2 in
  (C1 → C2 → ∃ PQ, PQ = (2 * real.sqrt (radius1^2 - ( (radius1 * distance_AB) / (radius1 + radius2))^2 )) → 
     PQ * PQ = (2 * real.sqrt (1105) / 5)^2 :=
by {
  -- Proof placeholder
  sorry
}

end shortest_tangent_length_l722_722257


namespace log_base_4_of_fraction_l722_722473

theorem log_base_4_of_fraction :
  log 4 (1 / 16) = -2 := by
  sorry

end log_base_4_of_fraction_l722_722473


namespace non_real_roots_interval_l722_722187

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722187


namespace alice_vs_bob_payment_multiple_l722_722437

theorem alice_vs_bob_payment_multiple :
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  total_alice_payment / bob_payment = 9 := by
  -- define the variables as per the conditions
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  -- define the target statement
  show total_alice_payment / bob_payment = 9
  sorry

end alice_vs_bob_payment_multiple_l722_722437


namespace sin_double_angle_tan_sum_angle_l722_722501

variable (α : ℝ)

-- Conditions
def sin_alpha : Prop := sin α = 3 / 5
def alpha_second_quadrant : Prop := π / 2 < α ∧ α < π

-- Part 1: Proving sin 2α = -24/25
theorem sin_double_angle (h1 : sin_alpha α) (h2 : alpha_second_quadrant α) :
  sin (2 * α) = -24 / 25 := 
sorry

-- Part 2: Proving tan (α + π/4) = 1/7
theorem tan_sum_angle (h1 : sin_alpha α) (h2 : alpha_second_quadrant α) :
  tan (α + π / 4) = 1 / 7 := 
sorry

end sin_double_angle_tan_sum_angle_l722_722501


namespace expression_evaluation_l722_722448

theorem expression_evaluation :
  (1007 * (((7/4 : ℚ) / (3/4) + (3 / (9/4)) + (1/3)) /
    ((1 + 2 + 3 + 4 + 5) * 5 - 22)) / 19) = (4 : ℚ) :=
by
  sorry

end expression_evaluation_l722_722448


namespace pq_sum_no_x2_x3_terms_l722_722197

theorem pq_sum_no_x2_x3_terms (p q : ℝ) :
  (∀ x : ℝ, (x^2 + p) * (x^2 - q * x + 4) = x^4 + (0 * x^3) + (0 * x^2) + (-(p * q) * x) + 4 * p)
  → (p + q = -4) :=
begin
  intro h,
  sorry -- Proof steps are omitted as per instruction.
end

end pq_sum_no_x2_x3_terms_l722_722197


namespace non_real_roots_bounded_l722_722156

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722156


namespace ellipse_equation_fixed_point_y_l722_722930

-- Definitions and conditions
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ a > 0 ∧ b > 0

def foci (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (- sqrt 3, 0) ∧ F2 = (sqrt 3, 0)

def area_triangle (B F O : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - O.1) * (F.2 - O.2) - (B.2 - O.2) * (F.1 - O.1)).abs

theorem ellipse_equation (a b : ℝ) (B F O : ℝ × ℝ)
  (h_ellipse : ellipse B.1 B.2 a b)
  (h_foci : foci (- sqrt 3, 0) (sqrt 3, 0))
  (h_area : area_triangle B (sqrt 3, 0) (0,0) = sqrt 3 / 2) :
  ∃ c : ℝ, a^2 = b^2 + c^2 ∧ a^2 = 4 ∧ b = 1 := sorry

theorem fixed_point_y-axis (P M N E : ℝ × ℝ)
  (h_ellipse : ellipse P.1 P.2 2 1)
  (h_symmetry : M = (- N.1, N.2))
  (h_intersection : ∃ k : ℝ, ∃ x₁ x₂ y₁ y₂ : ℝ, P.2 = k * P.1 + 4 ∧ E.2 = k * N.1 + 4 ∧ y₁ = y₂ ∧ E = (x₂, y₂)) :
  ∃ y : ℝ, y = 1 / 4 := sorry

end ellipse_equation_fixed_point_y_l722_722930


namespace length_generatrix_cone_l722_722067

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722067


namespace shorter_leg_in_right_triangle_l722_722578

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l722_722578


namespace count_cube_roots_less_than_15_l722_722125

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l722_722125


namespace intersect_at_midpoint_l722_722423

-- Define triangle and points
variable (A B C D K L J P : Point)
variable (α β : ℝ)

-- Conditions of the problem
axiom angle_conditions : ∠C < ∠A ∧ ∠A < 90
axiom point_D_on_AC : D ∈ line_through A C
axiom BD_equals_BA : dist B D = dist B A
axiom tangents : InCircleTangentPoint K L triangle ABC
axiom J_incenter_BCD : IsIncenter J (triangle B C D)

-- Problem statement
theorem intersect_at_midpoint :
  line_through K L ∩ segment A J = Some (midpoint A J) :=
sorry

end intersect_at_midpoint_l722_722423


namespace expression_simplifies_to_49_l722_722135

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l722_722135


namespace smallest_integer_inverse_defined_mod_882_l722_722824

theorem smallest_integer_inverse_defined_mod_882 :
  ∃ n : ℕ, n > 1 ∧ coprime n 882 ∧ ∀ m : ℕ, m > 1 ∧ coprime m 882 → n ≤ m :=
begin
  use 5,
  split,
  { norm_num },
  split,
  { norm_num,
    exact dec_trivial },
  { intros m hm1 hm2,
    cases m,
    { exfalso, linarith },
    cases m,
    { exfalso, linarith },
    cases m,
    { norm_num at hm2,
      apply nat.succ_le_succ,
      apply nat.succ_le_succ,
      exact dec_trivial },
    { cases m,
      { norm_num at hm2,
        apply nat.succ_le_succ,
        apply nat.succ_le_succ,
        apply nat.succ_le_succ,
        exact dec_trivial },
      { cases m,
        { norm_num at hm1,
          exfalso,
          apply nat.not_succ_le_0,
          rwa ← nat.succ_le_succ_iff at hm1 },
        { norm_num,
          apply le_of_lt,
          exact nat.succ_lt_succ,
          exact nat.succ_lt_succ,
          exact nat.succ_lt_succ,
          exact nat.succ_lt_succ dec_trivial } } } }
end

end smallest_integer_inverse_defined_mod_882_l722_722824


namespace no_solution_of_abs_sum_l722_722393

theorem no_solution_of_abs_sum (a : ℝ) : (∀ x : ℝ, |x - 2| + |x + 3| < a → false) ↔ a ≤ 5 := sorry

end no_solution_of_abs_sum_l722_722393


namespace unique_triple_solution_l722_722888

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (y > 1) ∧ Prime y ∧
                  (¬(3 ∣ z ∧ y ∣ z)) ∧
                  (x^3 - y^3 = z^2) ∧
                  (x = 8 ∧ y = 7 ∧ z = 13) :=
by
  sorry

end unique_triple_solution_l722_722888


namespace find_number_l722_722726

theorem find_number (x : ℕ) (h : 8 * x = 64) : x = 8 :=
sorry

end find_number_l722_722726


namespace sufficient_but_not_necessary_l722_722017

variables {p q : Prop}

theorem sufficient_but_not_necessary :
  (p → q) ∧ (¬q → ¬p) ∧ ¬(q → p) → (¬q → ¬p) ∧ (¬(q → p)) :=
by
  sorry

end sufficient_but_not_necessary_l722_722017


namespace center_coordinates_l722_722316

noncomputable def center_of_circle (x y : ℝ) : Prop := 
  x^2 + y^2 + 2*x - 4*y = 0

theorem center_coordinates : center_of_circle (-1) 2 :=
by sorry

end center_coordinates_l722_722316


namespace cone_generatrix_length_l722_722049

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722049


namespace cos_angle_sum_eq_negative_sqrt_10_div_10_l722_722508

theorem cos_angle_sum_eq_negative_sqrt_10_div_10 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (α + π / 4) = - Real.sqrt 10 / 10 := by
  sorry

end cos_angle_sum_eq_negative_sqrt_10_div_10_l722_722508


namespace length_generatrix_cone_l722_722064

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722064


namespace log_eq_solution_l722_722298

theorem log_eq_solution (x : ℝ) (h : log 8 x + log 4 (x^3) = 9) : x = 2^(54/5) :=
by
  sorry

end log_eq_solution_l722_722298


namespace find_original_height_l722_722279

noncomputable def original_height (H : ℝ) : Prop :=
  let h1 := 1 / 2 * H
  let h2 := 1 / 4 * H
  H + 2 * h1 + 2 * h2 = 260

theorem find_original_height : ∃ H : ℝ, original_height H ∧ H = 104 := by
  exists 104
  unfold original_height
  simp
  norm_num
  split
  apply rfl
  sorry

end find_original_height_l722_722279


namespace eval_sqrt_expression_l722_722879

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722879


namespace number_of_distinct_prime_factors_30_fact_l722_722993

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722993


namespace ratio_of_a_over_b_l722_722951

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem ratio_of_a_over_b (a b : ℝ) (h_max : ∀ x : ℝ, f a b x ≤ 10)
  (h_cond1 : f a b 1 = 10) (h_cond2 : (deriv (f a b)) 1 = 0) :
  a / b = -2/3 :=
sorry

end ratio_of_a_over_b_l722_722951


namespace non_real_roots_interval_l722_722182

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722182


namespace nested_sqrt_eq_l722_722870

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722870


namespace probability_complement_l722_722487

variable (P : Set → ℝ)
variable (A B : Set)
variable (a b c : ℝ)

theorem probability_complement (ha : P A = a) (hb : P B = b) (hc : P (A ∪ B) = c) :
  P (Aᶜ ∩ Bᶜ) = 1 - c :=
sorry

end probability_complement_l722_722487


namespace number_of_prime_factors_30_factorial_l722_722985

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722985


namespace log_four_one_div_sixteen_l722_722465

theorem log_four_one_div_sixteen : log 4 (1 / 16) = -2 := 
by 
  sorry

end log_four_one_div_sixteen_l722_722465


namespace find_angle_A_range_of_b2_plus_c2_over_a2_l722_722602

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ} 

-- Part 1: Prove A = π / 3 under the given condition
theorem find_angle_A (h : (b + c) / a = 2 * sin (C + π / 6)) : 
  A = π / 3 :=
by sorry

-- Part 2: Prove the range of (b^2 + c^2) / a^2 is (1, 2]
theorem range_of_b2_plus_c2_over_a2 (h : (b + c) / a = 2 * sin (C + π / 6)) : 
  1 < (b^2 + c^2) / a^2 ∧ (b^2 + c^2) / a^2 ≤ 2 :=
by sorry

end find_angle_A_range_of_b2_plus_c2_over_a2_l722_722602


namespace common_ratio_of_series_l722_722898

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l722_722898


namespace measure_of_angle_B_l722_722013

variable (a b c A B C : ℝ)
variable (R : ℝ) -- Circumradius

-- Conditions
axiom triangle_sides : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
axiom triangle_conditions : 
  ∀ a A B b, (sqrt 3) * b * sin A - a * cos B - 2 * a = 0

-- The proof problem statement
theorem measure_of_angle_B :
  (sqrt 3) * b * sin A - a * cos B - 2 * a = 0 →
  B = 2 * pi / 3 :=
sorry

end measure_of_angle_B_l722_722013


namespace modulus_w_eq_one_l722_722623

open Complex

noncomputable def z : ℂ := ((5 - 2 * I) ^ 4 * (-3 + 9 * I) ^ 3) / (2 - 3 * I)
def w : ℂ := conj z / z

theorem modulus_w_eq_one : abs w = 1 :=
by sorry

end modulus_w_eq_one_l722_722623


namespace tangent_half_angle_sum_eq_product_l722_722285

variable {α β γ : ℝ}

theorem tangent_half_angle_sum_eq_product (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) =
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) :=
sorry

end tangent_half_angle_sum_eq_product_l722_722285


namespace sum_of_fifths_divisible_by_30_l722_722664

open BigOperators

theorem sum_of_fifths_divisible_by_30 {a : ℕ → ℕ} {n : ℕ} 
  (h : 30 ∣ ∑ i in Finset.range n, a i) : 
  30 ∣ ∑ i in Finset.range n, (a i) ^ 5 := 
by sorry

end sum_of_fifths_divisible_by_30_l722_722664


namespace cube_root_numbers_less_than_15_l722_722103

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l722_722103


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722109

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722109


namespace find_G16_l722_722249

variable (G : ℝ → ℝ)

def condition1 : Prop := G 8 = 28

def condition2 : Prop := ∀ x : ℝ, 
  (x^2 + 8*x + 16) ≠ 0 → 
  (G (4*x) / G (x + 4) = 16 - (64*x + 80) / (x^2 + 8*x + 16))

theorem find_G16 (h1 : condition1 G) (h2 : condition2 G) : G 16 = 120 :=
sorry

end find_G16_l722_722249


namespace triangle_PQR_QR_length_l722_722603

noncomputable def length_QR : ℝ :=
  let PQ : ℝ := 3 
  let angle_Q : ℝ := 90 
  let least_XZ : ℝ := 2.4 in
  least_XZ

theorem triangle_PQR_QR_length (PQ : ℝ) (angle_Q : ℝ) (least_XZ : ℝ) : PQ = 3 → angle_Q = 90 → least_XZ = 2.4 → length_QR = 2.4 :=
by
  intros
  sorry

end triangle_PQR_QR_length_l722_722603


namespace add_in_base_7_l722_722479

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end add_in_base_7_l722_722479


namespace product_of_a_l722_722322

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem product_of_a :
  (∀ a : ℝ, distance (3 * a, 2 * a - 5) (6, -2) = 3 * real.sqrt 17 →
    a ∈ ({120 / 13, -24 / 13} : set ℝ)) →
  (120 / 13) * (-24 / 13) = - (2880 / 169) :=
by
  sorry

end product_of_a_l722_722322


namespace nested_sqrt_eq_l722_722872

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722872


namespace range_of_PA_PB_PC_magnitude_l722_722527

noncomputable def point := (ℝ × ℝ)

def unit_circle (A : point) : Prop := A.1^2 + A.2^2 = 1

def perpendicular (A B C : point) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.1 + AB.2 * BC.2 = 0

def distance (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def P : point := (2, 0)

def PA (A : point) : point := (A.1 - P.1, A.2 - P.2)
def PB (B : point) : point := (B.1 - P.1, B.2 - P.2)
def PC (C : point) : point := (C.1 - P.1, C.2 - P.2)

def PA_PB_PC_magnitude (A B C : point) : ℝ :=
  real.abs (distance P A + distance P B + distance P C)

theorem range_of_PA_PB_PC_magnitude :
  ∀ (A B C : point),
  unit_circle A →
  unit_circle B →
  unit_circle C →
  perpendicular A B C →
  5 ≤ PA_PB_PC_magnitude A B C ∧ PA_PB_PC_magnitude A B C ≤ 7 :=
sorry

end range_of_PA_PB_PC_magnitude_l722_722527


namespace determine_number_of_terms_l722_722943

variables (a_1 a_n d : ℝ) (n : ℕ)

def arithmetic_sequence_sum_first_three (a_1 d : ℝ) : ℝ :=
  a_1 + (a_1 + d) + (a_1 + 2 * d)

def arithmetic_sequence_sum_last_three (a_n d : ℝ) : ℝ :=
  a_n + (a_n - d) + (a_n - 2 * d)

def arithmetic_sequence_sum_all (n a_1 a_n : ℝ) : ℝ :=
  n * (a_1 + a_n) / 2

theorem determine_number_of_terms (h1 : arithmetic_sequence_sum_first_three a_1 d = 94)
  (h2 : arithmetic_sequence_sum_last_three a_n d = 116)
  (h3 : arithmetic_sequence_sum_all n a_1 a_n = 280) : n = 8 :=
by
  sorry

end determine_number_of_terms_l722_722943


namespace number_of_prime_factors_30_factorial_l722_722975

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722975


namespace probability_intersection_l722_722756

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l722_722756


namespace largest_r_divides_l722_722452

-- Define the conditions for P
def satisfies_condition (P : ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∀ (a b c : ℝ), P a b c = 0 ↔ a = b ∧ b = c

-- Main statement
theorem largest_r_divides {P : ℤ → ℤ → ℤ → ℤ}
  (hP : satisfies_condition P) :
  ∀ (m n : ℤ), m ≠ 0 → m^2 ∣ P n (n + m) (n + 2 * m) :=
sorry

end largest_r_divides_l722_722452


namespace sequence_general_formula_l722_722957

open_locale classical

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (a n) / 2 + 1 / (a n) - 1

theorem sequence_general_formula (a : ℕ → ℝ) (H : ∀ n, S n a = (a n) / 2 + 1 / (a n) - 1 ∧ a n > 0) :
  ∀ n : ℕ, a n = real.sqrt (2 * n + 1) - real.sqrt (2 * n - 1) :=
by sorry

end sequence_general_formula_l722_722957


namespace fuel_consumption_le_9_min_fuel_consumption_100_km_l722_722764

variables (k x : ℝ) (h₁ : 60 ≤ x ∧ x ≤ 120) (h₂ : 60 ≤ k ∧ k ≤ 100)
def fuel_consumption (x : ℝ) (k : ℝ) : ℝ := (1/5 : ℝ) * (x - k + 4500 / x)

-- Question 1: Fuel consumption does not exceed 9 liters per hour.
theorem fuel_consumption_le_9 
  (h₃ : fuel_consumption 120 k = 11.5):
  60 ≤ x ∧ x ≤ 100 ↔ fuel_consumption x k ≤ 9 :=
sorry

-- Question 2: Minimum fuel consumption for 100 kilometers.
def fuel_consumption_100_km (x k : ℝ) : ℝ := (100/x) * fuel_consumption x k

theorem min_fuel_consumption_100_km 
  (h₃ : fuel_consumption 120 k = 11.5):
  (75 ≤ k ∧ k < 100 ∧ fuel_consumption_100_km (9000 / k) k = 20 - k^2 / 900) 
  ∨ (60 ≤ k ∧ k < 75 ∧ fuel_consumption_100_km 120 k = 105/4 - k/6) :=
sorry

end fuel_consumption_le_9_min_fuel_consumption_100_km_l722_722764


namespace floor_ineq_l722_722496

theorem floor_ineq (x y : ℝ) : 
  Int.floor (2 * x) + Int.floor (2 * y) ≥ Int.floor x + Int.floor y + Int.floor (x + y) := 
sorry

end floor_ineq_l722_722496


namespace limit_log_expr_is_one_l722_722560

noncomputable def limit_log_expr : ℝ := 
  λ x, log2 (8 * x - 6) - log2 (4 * x + 3)

theorem limit_log_expr_is_one (h₁ : ∀ x : ℝ, x > 0) : 
  tendsto (λ x, log2 (8 * x - 6) - log2 (4 * x + 3)) at_top (𝓝 1) := by
  sorry

end limit_log_expr_is_one_l722_722560


namespace isosceles_triangle_angle_bisector_length_l722_722314

noncomputable theory

open Real

variables {a : ℝ} {α : ℝ}

theorem isosceles_triangle_angle_bisector_length (hα : 0 < α ∧ α < π) :
  let cos_half_alpha := cos (α / 2)
      sin_sum := sin (π / 4 + 3 * α / 4)
  in (a * cos_half_alpha) / sin_sum = (a * cos (α / 2)) / sin (π / 4 + 3 * α / 4) := 
by
  sorry

end isosceles_triangle_angle_bisector_length_l722_722314


namespace probability_intersection_l722_722744

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l722_722744


namespace solution1_solution2_l722_722449

noncomputable def problem1 : ℝ :=
  -1 - (1 : ℝ)^2023 + (Real.pi - 3.14)^0 - (1 / 2)^(-2 : ℤ) - | -2 : ℝ |

noncomputable def problem2 : ℤ := 
  2017 * 2023 - 2020^2

theorem solution1 : problem1 = -6 := 
by 
  sorry

theorem solution2 : problem2 = -9 := 
by 
  sorry

end solution1_solution2_l722_722449


namespace measurable_abs_xi_implies_measurable_xi_l722_722235

namespace Measurability

def omega : Set ℕ := {0, 1}

def sigma_f : Set (Set ℕ) := {∅, omega}

def xi (ω : ℕ) : ℤ :=
  if ω = 0 then -1 else if ω = 1 then 1 else 0

def abs_xi (ω : ℕ) : ℤ :=
  |xi ω|

def measurable (f : ℕ → ℤ) (σ : Set (Set ℕ)) : Prop :=
  ∀ s, {ω | f ω ∈ s} ∈ σ

theorem measurable_abs_xi_implies_measurable_xi :
  measurable abs_xi sigma_f → ¬ measurable xi sigma_f :=
by 
  sorry

end Measurability

end measurable_abs_xi_implies_measurable_xi_l722_722235


namespace pyramid_volume_l722_722916

variable (a α : ℝ)

theorem pyramid_volume (h1 : 0 < α ∧ α < π / 2) : 
  ∃ V : ℝ, V = (sqrt 3 * a^3 * sqrt((1 + 4 * (tan α)^2)^3)) / (4 * (tan α)^2) :=
by
  sorry

end pyramid_volume_l722_722916


namespace minimum_value_2x_plus_y_l722_722937

theorem minimum_value_2x_plus_y (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : (1 / x) + (2 / (y + 1)) = 2) : 2 * x + y ≥ 3 := 
by
  sorry

end minimum_value_2x_plus_y_l722_722937


namespace problem_l722_722014

def f (x : ℚ) : ℚ :=
  x⁻¹ - (x⁻¹ / (1 - x⁻¹))

theorem problem : f (f (-3)) = 6 / 5 :=
by
  sorry

end problem_l722_722014


namespace cone_generatrix_length_l722_722057

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722057


namespace extreme_value_f_a1_k2_derivative_sum_positive_l722_722923

open Real

def f (x : ℝ) (k a : ℝ) : ℝ := -1/x - k * x + a * log x

-- Part (1): Prove the extreme value of the function f(x) when a = 1, k = 2
theorem extreme_value_f_a1_k2 : (∀ x : ℝ, x > 0 → f x 2 1 ≤ f 1 2 1) ∧ (f 1 2 1 = -3) :=
by sorry

-- Part (2): Given f(x₁) = f(x₂), prove f'(x₁) + f'(x₂) > 0
theorem derivative_sum_positive (x₁ x₂ k a : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h_diff : x₁ ≠ x₂) 
  (h_eq : f x₁ k a = f x₂ k a) : f' x₁ k a + f' x₂ k a > 0 :=
by sorry

-- Additional definition for derivative of the function (not explicitly given in problem but required for the theorem)
noncomputable def f' (x k a : ℝ) : ℝ := deriv (λ x, f x k a) x

end extreme_value_f_a1_k2_derivative_sum_positive_l722_722923


namespace train_speed_l722_722434

theorem train_speed (distance time : ℝ) (h₁ : distance = 300) (h₂ : time = 10) :
  distance / time = 30 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end train_speed_l722_722434


namespace generatrix_length_l722_722034

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722034


namespace generatrix_length_of_cone_l722_722024

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722024


namespace eval_sqrt_expression_l722_722876

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722876


namespace range_of_a_l722_722077

theorem range_of_a (a : ℝ) (h : ¬ ∃ x : ℝ, x^2 + (a + 1) * x + 1 ≤ 0) : -3 < a ∧ a < 1 :=
sorry

end range_of_a_l722_722077


namespace common_ratio_of_series_l722_722897

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l722_722897


namespace generatrix_length_of_cone_l722_722022

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722022


namespace chocolate_chips_per_family_member_l722_722612

def total_family_members : ℕ := 4
def batches_choco_chip : ℕ := 3
def batches_double_choco_chip : ℕ := 2
def batches_white_choco_chip : ℕ := 1
def cookies_per_batch_choco_chip : ℕ := 12
def cookies_per_batch_double_choco_chip : ℕ := 10
def cookies_per_batch_white_choco_chip : ℕ := 15
def choco_chips_per_cookie_choco_chip : ℕ := 2
def choco_chips_per_cookie_double_choco_chip : ℕ := 4
def choco_chips_per_cookie_white_choco_chip : ℕ := 3

theorem chocolate_chips_per_family_member :
  (batches_choco_chip * cookies_per_batch_choco_chip * choco_chips_per_cookie_choco_chip +
   batches_double_choco_chip * cookies_per_batch_double_choco_chip * choco_chips_per_cookie_double_choco_chip +
   batches_white_choco_chip * cookies_per_batch_white_choco_chip * choco_chips_per_cookie_white_choco_chip) / 
   total_family_members = 49 :=
by
  sorry

end chocolate_chips_per_family_member_l722_722612


namespace min_sum_of_distances_l722_722931

-- Define conditions and the problem statement
def line1 : Type := {p : ℝ × ℝ // 4 * p.1 - 3 * p.2 + 6 = 0}
def line2 : Type := {p : ℝ × ℝ // p.1 = -1}
def parabola : Type := {p : ℝ × ℝ // p.2 ^ 2 = 4 * p.1}

def distance (p l : ℝ × ℝ // l l1 ∨ l l2) : ℝ :=
  match l with
  | ⟨(x, y), Or.inl _⟩ => abs(4 * x - 3 * y + 6) / sqrt (4^2 + (-3)^2)
  | ⟨(x, _), Or.inr _⟩ => abs(x + 1)

noncomputable def sum_of_distances (a : ℝ) : ℝ :=
  let P := (a^2, 2*a)
  distance P (⟨P, line1⟩) + distance P (⟨P, line2⟩)

theorem min_sum_of_distances : ∃ a : ℝ, sum_of_distances a = 2 := 
sorry

end min_sum_of_distances_l722_722931


namespace measure_of_angle_ACB_l722_722598

theorem measure_of_angle_ACB 
  (angle_ABD : ℝ) (angle_BAC : ℝ) (angle_ABC : ℝ) :
  angle_ABD = 120 ∧ angle_BAC = 105 ∧ angle_ABC = 60 →
  180 - (angle_BAC + angle_ABC) = 15 :=
by
  intro h
  cases h with h_ABD h_rest
  cases h_rest with h_BAC h_ABC
  rw [h_ABD, h_BAC, h_ABC]
  norm_num
  sorry

end measure_of_angle_ACB_l722_722598


namespace average_speeds_equation_l722_722591

theorem average_speeds_equation (x : ℝ) (hx : 0 < x) : 
  10 / x - 7 / (1.4 * x) = 10 / 60 :=
by
  sorry

end average_speeds_equation_l722_722591


namespace find_smaller_number_l722_722335

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l722_722335


namespace number_of_distinct_prime_factors_30_fact_l722_722991

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722991


namespace quadratic_non_real_roots_b_values_l722_722170

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722170


namespace num_pos_nums_with_cube_root_lt_15_l722_722129

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l722_722129


namespace relationship_y1_y2_y3_l722_722563

-- Given function y = - (m^2 + 5) / x and points A, B, C on the graph of the function
def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := - (m^2 + 5) / x

-- Define the points A, B, and C
def A (m : ℝ) : ℝ × ℝ := (-2, inverse_proportion m (-2))
def B (m : ℝ) : ℝ × ℝ := (1, inverse_proportion m 1)
def C (m : ℝ) : ℝ × ℝ := (3, inverse_proportion m 3)

theorem relationship_y1_y2_y3 (m : ℝ) (y1 y2 y3 : ℝ) :
  y1 = inverse_proportion m (-2) →
  y2 = inverse_proportion m 1 →
  y3 = inverse_proportion m 3 →
  y2 < y3 ∧ y3 < y1 :=
by
  sorry

end relationship_y1_y2_y3_l722_722563


namespace equal_sums_even_moves_l722_722595

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ abs (p1.2 - p2.2) = 1) ∨ (p1.2 = p2.2 ∧ abs (p1.1 - p2.1) = 1)

def make_move (grid : ℕ → ℕ → ℤ) (p1 p2 : ℕ × ℕ) (n : ℤ) : ℕ → ℕ → ℤ :=
  λ i j, if (i, j) = p1 ∨ (i, j) = p2 then grid i j + n else grid i j

def sum_column (grid : ℕ → ℕ → ℤ) (col : ℕ) : ℤ :=
  ∑ i in Finset.range 5, grid i col

def sum_row (grid : ℕ → ℕ → ℤ) (row : ℕ) : ℤ :=
  ∑ j in Finset.range 5, grid row j

theorem equal_sums_even_moves
  (initial_grid : ℕ → ℕ → ℤ := λ _ _, 0)
  (moves : list ((ℕ × ℕ) × (ℕ × ℕ) × ℤ))
  (final_grid : ℕ → ℕ → ℤ := λ i j, (moves.foldl (λ g (p : (ℕ × ℕ) × (ℕ × ℕ) × ℤ), make_move g p.1 p.2) initial_grid) i j)
  (h_adj : ∀ (m : (ℕ × ℕ) × (ℕ × ℕ) × ℤ), is_adjacent m.1.1 m.1.2)
  (h_equal_sums : ∀ i j, sum_row final_grid i = sum_column final_grid j) :
  moves.length % 2 = 0 :=
by
  sorry

end equal_sums_even_moves_l722_722595


namespace non_real_roots_interval_l722_722186

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722186


namespace probability_intersection_l722_722757

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l722_722757


namespace negation_of_proposition_l722_722684

theorem negation_of_proposition (x y : ℝ) :
  (¬ (x + y = 1 → xy ≤ 1)) ↔ (x + y ≠ 1 → xy > 1) :=
by 
  sorry

end negation_of_proposition_l722_722684


namespace option_d_is_not_even_l722_722441

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def f1 (x : ℝ) : ℝ := x^2 + 4
def f2 (x : ℝ) : ℝ := abs (tan x)
def f3 (x : ℝ) : ℝ := cos (2 * x)
def f4 (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem option_d_is_not_even :
    ¬ is_even f4 ∧ is_even f1 ∧ is_even f2 ∧ is_even f3 :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end option_d_is_not_even_l722_722441


namespace half_percent_to_decimal_l722_722734

def percent_to_decimal (x : ℚ) : ℚ := x / 100

theorem half_percent_to_decimal : percent_to_decimal (1 / 2) = 0.005 :=
by
  sorry

end half_percent_to_decimal_l722_722734


namespace find_b_value_l722_722321

theorem find_b_value : 
  let midpoint := let x1 := 3; let y1 := 6; let x2 := 5; let y2 := 10 in 
                  ((x1 + x2) / 2, (y1 + y2) / 2) in
  let x := midpoint.1 in
  let y := midpoint.2 in
  (x + y = 12) :=
by
  let x1 := 3
  let y1 := 6
  let x2 := 5
  let y2 := 10 in
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2) in
  let x := midpoint.1 in
  let y := midpoint.2 in
  show x + y = 12 from sorry

end find_b_value_l722_722321


namespace pow_op_eq_op_pow_l722_722636

def op_⊕ (z w : ℝ∞) : ℝ∞ := min z w
def op_⊙ (z w : ℝ∞) : ℝ∞ := z + w
noncomputable def pow (z : ℝ∞) (n : ℕ) : ℝ∞ := n * z

theorem pow_op_eq_op_pow (x y : ℝ∞) (n : ℕ) (hn : 0 < n) : 
  pow (op_⊕ x y) n = op_⊕ (pow x n) (pow y n) :=
by
  sorry

end pow_op_eq_op_pow_l722_722636


namespace digit_sum_of_653xy_div_by_80_l722_722567

theorem digit_sum_of_653xy_div_by_80 (x y : ℕ) (h₀ : 6530 + 10 * x + y 80) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9) : x + y = 8 :=
sorry

end digit_sum_of_653xy_div_by_80_l722_722567


namespace billy_free_time_l722_722445

theorem billy_free_time
  (play_time_percentage : ℝ := 0.75)
  (read_pages_per_hour : ℝ := 60)
  (book_pages : ℝ := 80)
  (number_of_books : ℝ := 3)
  (read_percentage : ℝ := 1 - play_time_percentage)
  (total_pages : ℝ := number_of_books * book_pages)
  (read_time_hours : ℝ := total_pages / read_pages_per_hour)
  (free_time_hours : ℝ := read_time_hours / read_percentage) :
  free_time_hours = 16 := 
sorry

end billy_free_time_l722_722445


namespace max_quotient_l722_722554

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : (b / a) ≤ 15 :=
  sorry

end max_quotient_l722_722554


namespace quadratic_non_real_roots_l722_722169

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722169


namespace celebrity_matching_probability_l722_722410

theorem celebrity_matching_probability :
  let n := 4 in
  let total_possible_matches := Nat.factorial n in
  let correct_matches := 1 in
  (correct_matches / total_possible_matches : ℚ) = (1 / 24 : ℚ) := 
by 
  have h1 : total_possible_matches = Nat.factorial 4 := rfl
  have h2 : Nat.factorial 4 = 24 := by norm_num
  have h3 : total_possible_matches = 24 := h1.trans h2
  have h4 : (correct_matches / total_possible_matches : ℚ) = (1 / 24 : ℚ) := by rw [h3]; norm_num
  exact h4

end celebrity_matching_probability_l722_722410


namespace sector_angle_l722_722072

-- Defining the conditions
def perimeter (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1 / 2) * l * r = 4

-- Lean theorem statement
theorem sector_angle (r l θ : ℝ) :
  (perimeter r l) → (area r l) → (θ = l / r) → |θ| = 2 :=
by sorry

end sector_angle_l722_722072


namespace cups_remaining_l722_722609

-- Each definition only directly appears in the conditions problem
def required_cups : ℕ := 7
def added_cups : ℕ := 3

-- The proof problem capturing Joan needs to add 4 more cups of flour.
theorem cups_remaining : required_cups - added_cups = 4 := 
by
  -- The proof is skipped using sorry.
  sorry

end cups_remaining_l722_722609


namespace quadratic_solution_difference_l722_722690

theorem quadratic_solution_difference (x : ℝ) :
  ∀ x : ℝ, (x^2 - 5*x + 15 = x + 55) → (∃ a b : ℝ, a ≠ b ∧ x^2 - 6*x - 40 = 0 ∧ abs (a - b) = 14) :=
by
  sorry

end quadratic_solution_difference_l722_722690


namespace sum_of_abscissas_l722_722883

open Real

theorem sum_of_abscissas :
  let f1 := λ x : ℝ, 8 * cos (π * x) * cos (2 * π * x)^2 * cos (4 * π * x)
  let f2 := λ x : ℝ, cos (5 * π * x)
  let points := {x | x ∈ Icc (-1) 0 ∧ f1 x = f2 x}
  (∑ x in points.to_finset, x) = -4.5 :=
by 
  let f1 := λ x : ℝ, 8 * cos (π * x) * cos (2 * π * x)^2 * cos (4 * π * x)
  let f2 := λ x : ℝ, cos (5 * π * x)
  let points := {x | x ∈ Icc (-1) 0 ∧ f1 x = f2 x}
  have h: (∑ x in points.to_finset, x) = -4.5,
  {
    sorry
  },
  exact h

end sum_of_abscissas_l722_722883


namespace necessary_but_not_sufficient_l722_722070

def P (f : ℝ → ℝ) : Prop := 
  ∀ x1 x2 : ℝ, x1 ≠ x2 → (| (f x1 - f x2) / (x1 - x2) |) < 2017

def Q (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (deriv f x | < 2017

theorem necessary_but_not_sufficient (f : ℝ → ℝ) (h_diff : differentiable ℝ f) :
  (Q f) → (P f) ∧ ¬(P f → Q f) :=
sorry

end necessary_but_not_sufficient_l722_722070


namespace problem_M_value_and_floor_l722_722934

theorem problem_M_value_and_floor :
  (∑ k in finset.range 7, (1:ℝ) / (fact (3 + k) * fact (16 - k))) = (M:ℝ) / (fact 1 * fact 18) →
  M = 13787 ∧ ⌊M / 100⌋ = 137 :=
by
  sorry

end problem_M_value_and_floor_l722_722934


namespace volume_rectangular_box_l722_722367

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l722_722367


namespace cos_sum_arithmetic_seq_l722_722011

theorem cos_sum_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1 / 2 :=
  sorry

end cos_sum_arithmetic_seq_l722_722011


namespace probability_intersection_l722_722746

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l722_722746


namespace museum_ticket_cost_l722_722803

variable (x : ℝ) 

def child_ticket_price := x / 3
def total_cost_4_adult_3_child := 4 * x + 3 * child_ticket_price x
def total_cost_10_adult_8_child := 10 * x + 8 * child_ticket_price x

theorem museum_ticket_cost :
  (total_cost_4_adult_3_child x = 35) → (total_cost_10_adult_8_child x = 88.67) :=
sorry

end museum_ticket_cost_l722_722803


namespace find_volume_of_prism_l722_722331

noncomputable def volume_of_inscribed_prism 
  (a : ℝ) (α : ℝ) (φ : ℝ) : ℝ :=
  a^3 * (Real.sqrt 2) * (Real.cot φ)^2 / ((Real.cot φ) + 2 * (Real.cot α))^3

theorem find_volume_of_prism
  (a : ℝ) (α : ℝ) (φ : ℝ)
  (side_length_condition : a > 0)
  (upper_base_on_edges : True)
  (lower_base_in_plane : True)
  (diagonal_angle_condition : φ > 0 ∧ φ < Real.pi / 2)
  (lateral_edge_angle_condition : α > 0 ∧ α < Real.pi / 2) 
  : volume_of_inscribed_prism a α φ = 
    a^3 * (Real.sqrt 2) * (Real.cot φ)^2 / ((Real.cot φ) + 2 * (Real.cot α))^3 :=
  by sorry

end find_volume_of_prism_l722_722331


namespace time_to_cross_bridge_l722_722095

def train_length : ℝ := 110 -- Length of train in meters
def train_speed_kmh : ℝ := 72 -- Speed of train in kilometers per hour
def bridge_length : ℝ := 136 -- Length of bridge in meters

def speed_in_mps : ℝ := train_speed_kmh * (1000 / 3600) -- Convert speed to meters per second
def total_distance : ℝ := train_length + bridge_length -- Total distance to be covered by the train

theorem time_to_cross_bridge : 
  total_distance / speed_in_mps = 12.3 := 
by {
  sorry
}

end time_to_cross_bridge_l722_722095


namespace find_a_l722_722392

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 0 then Real.log (x^2 + a) else f a (x - 2)

theorem find_a (a : ℝ) (h : f a 2023 = 1) : a = Real.exp 1 - 1 :=
  sorry

end find_a_l722_722392


namespace even_perfect_square_factors_count_l722_722550

theorem even_perfect_square_factors_count : 
  let n := 2^6 * 7^10 * 3^2
  let count := (nat.factors n).to_finset.filter (λ x, 
    (x.factors.count 2 ∣ 2) ∧ (x.factors.count 2 ≥ 1) ∧ 
    (x.factors.count 7 ∣ 2) ∧ (x.factors.count 3 ∣ 2)).card
  count = 36 :=
by
  sorry

end even_perfect_square_factors_count_l722_722550


namespace magnitude_is_2_l722_722637

-- Definition for the condition
def condition (z : ℂ) : Prop :=
  2 + z = (2 - z) * complex.I

-- Theorem statement that proves the magnitude of z is 2 given the condition
theorem magnitude_is_2 (z : ℂ) (h : condition z) : complex.abs z = 2 :=
by sorry

end magnitude_is_2_l722_722637


namespace eval_sqrt_expression_l722_722875

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l722_722875


namespace quadratic_non_real_roots_l722_722167

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722167


namespace common_ratio_of_geometric_series_l722_722895

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l722_722895


namespace speed_of_stream_l722_722418

variable (b s : ℝ)

-- Conditions:
def downstream_eq : Prop := 90 = (b + s) * 3
def upstream_eq : Prop := 72 = (b - s) * 3

-- Goal:
theorem speed_of_stream (h1 : downstream_eq b s) (h2 : upstream_eq b s) : s = 3 :=
by
  sorry

end speed_of_stream_l722_722418


namespace neg_p_three_range_of_k_if_pq_false_l722_722520

variable {x : ℝ}
variable {k : ℝ}
-- Question 1: Prove the negation of proposition p when k = 3
theorem neg_p_three (h : ∀ x : ℝ, 3 * x^2 + 1 > 0) : True :=
sorry

-- Question 2: Prove the range of k if p ∨ q is false
theorem range_of_k_if_pq_false :
  (¬ (∃ x : ℝ, k * x^2 + 1 ≤ 0) ∧ (∃ x : ℝ, x^2 + 2 * k * x + 1 ≤ 0)) → (k ∈ Set.Icc (-∞) (-1) ∪ Set.Icc 1 (∞)) :=
sorry

end neg_p_three_range_of_k_if_pq_false_l722_722520


namespace sum_of_a_values_equidistant_from_axes_l722_722782

theorem sum_of_a_values_equidistant_from_axes :
  (let a₁ := 24 / 11 in
   let a₂ := -24 / 5 in
   a₁ + a₂ = -144 / 55) := 
by
  sorry

end sum_of_a_values_equidistant_from_axes_l722_722782


namespace sphere_to_hemisphere_volume_ratio_l722_722328

theorem sphere_to_hemisphere_volume_ratio (p : ℝ) (h1 : p > 0) : 
  let V_sphere := (4 / 3) * Real.pi * p^3,
  let V_hemisphere := (1 / 2) * (4 / 3) * Real.pi * (3 * p)^3
  in V_sphere / V_hemisphere = 2 / 27 :=
by
  sorry

end sphere_to_hemisphere_volume_ratio_l722_722328


namespace fuel_A_added_l722_722384

noncomputable def total_tank_capacity : ℝ := 218

noncomputable def ethanol_fraction_A : ℝ := 0.12
noncomputable def ethanol_fraction_B : ℝ := 0.16

noncomputable def total_ethanol : ℝ := 30

theorem fuel_A_added (x : ℝ) 
    (hA : 0 ≤ x) 
    (hA_le_capacity : x ≤ total_tank_capacity) 
    (h_eq : 0.12 * x + 0.16 * (total_tank_capacity - x) = total_ethanol) : 
    x = 122 := 
sorry

end fuel_A_added_l722_722384


namespace sqrt_recursive_value_l722_722840

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722840


namespace line_intersects_circle_two_points_slope_angle_given_ab_distance_line_equation_given_ap_pb_l722_722510

-- Given conditions
def circle_c : ℝ → ℝ → Prop := λ x y, x^2 + (y - 1)^2 = 5
def line_l : ℝ → (ℝ → Prop) := λ m, λ (x y : ℝ), mx - y + 1 - m = 0
def point_p : ℝ × ℝ := (1, 1)
def ab_distance := Real.sqrt 17

-- Prove statements
theorem line_intersects_circle_two_points (m : ℝ) :
  ∃ (A B : ℝ × ℝ), circle_c A.1 A.2 ∧ circle_c B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧ A ≠ B :=
sorry

theorem slope_angle_given_ab_distance (m : ℝ) (A B : ℝ × ℝ) 
  (hA : circle_c A.1 A.2) (hB : circle_c B.1 B.2) (hL_A : line_l m A.1 A.2) (hL_B : line_l m B.1 B.2) 
  (h_dist : A ≠ B ∧ Real.dist A B = ab_distance) :
  ∃ θ, θ = Real.pi / 3 ∨ θ = 2 * Real.pi / 3 :=
sorry

theorem line_equation_given_ap_pb (m : ℝ) (A B : ℝ × ℝ) 
  (hA : circle_c A.1 A.2) (hB : circle_c B.1 B.2) (hL_A : line_l m A.1 A.2) (hL_B : line_l m B.1 B.2) 
  (hP : 2 * ((1 : ℝ) - A.1, 1 - A.2) = (B.1 - 1, B.2 - 1)) :
  ∃ eq, (eq = λ x y, x - y = 0) ∨ (eq = λ x y, x + y - 2 = 0) :=
sorry

end line_intersects_circle_two_points_slope_angle_given_ab_distance_line_equation_given_ap_pb_l722_722510


namespace volume_rectangular_box_l722_722365

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l722_722365


namespace mean_median_difference_l722_722387

theorem mean_median_difference (s : Finset ℝ) (h1 : s.card = 105)
  (h2 : ∃ a : ℝ, ∀ n ∈ s, ∃ k ∈ (0:ℕ) .. 104, n = a + k * 11)
  (h3 : (∃ a : ℝ, (a + (a + 22)) / 2 = 346)) :
  let mean := (s.sum id) / 105
  let median := (s.to_list.sort.inth (s.card / 2))
  mean - median = 0 :=
by
  sorry

end mean_median_difference_l722_722387


namespace speed_of_A_l722_722397
-- Import necessary library

-- Define conditions
def initial_distance : ℝ := 25  -- initial distance between A and B
def speed_B : ℝ := 13  -- speed of B in kmph
def meeting_time : ℝ := 1  -- time duration in hours

-- The speed of A which is to be proven
def speed_A : ℝ := 12

-- The theorem to be proved
theorem speed_of_A (d : ℝ) (vB : ℝ) (t : ℝ) (vA : ℝ) : d = 25 → vB = 13 → t = 1 → 
  d = vA * t + vB * t → vA = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Enforcing the statement to be proved
  have := Eq.symm h4
  simp [speed_A, *] at *
  sorry

end speed_of_A_l722_722397


namespace aiguo_seashells_l722_722305

-- Define the number of seashells each child had
variables (A S V : ℕ)

-- Conditions given in the problem
def condition_1 : Prop := S = V + 16
def condition_2 : Prop := V = A - 5
def condition_3 : Prop := A + V + S = 66

-- Statement we want to prove
theorem aiguo_seashells (A S V : ℕ) (h1 : condition_1 S V) (h2 : condition_2 A V) (h3 : condition_3 A V S) : A = 20 :=
by {
    -- sorry allows us to skip the proof
    sorry,
}

end aiguo_seashells_l722_722305


namespace simplify_and_evaluate_l722_722656

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  ( (x + 3) / x - 1 ) / ( (x^2 - 1) / (x^2 + x) ) = Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l722_722656


namespace valid_c_values_count_l722_722912

def num_valid_c_values : Nat :=
  let count_11_multiples := (2000 / 11).toNat + 1
  let count_11k_plus_3_multiples := 
    (((2000 - 3) / 11).toNat + 1)
  count_11_multiples + count_11k_plus_3_multiples - 1

theorem valid_c_values_count :
  num_valid_c_values = 363 :=
sorry

end valid_c_values_count_l722_722912


namespace prob_A_inter_B_l722_722742

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l722_722742


namespace closest_point_to_given_l722_722907

open Real

-- Definition of the line
def line_eq (x : ℝ) : ℝ := 2 * x - 4

-- Definition of the Euclidean distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Given point
def given_point : ℝ × ℝ := (3, 1)

-- Define the closest point on the line to the given_point
def closest_point : ℝ × ℝ := (2.6, 1.2)

-- Define the closest point property: 
theorem closest_point_to_given : 
  ∀ p : ℝ × ℝ, p.snd = line_eq p.fst → dist closest_point given_point ≤ dist p given_point := sorry

end closest_point_to_given_l722_722907


namespace num_pos_nums_with_cube_root_lt_15_l722_722130

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l722_722130


namespace circle_center_coordinates_l722_722315

open Real

noncomputable def circle_center (x y : Real) : Prop := 
  x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center_coordinates :
  ∃ (a b : Real), circle_center a b ∧ a = 2 ∧ b = -3 :=
by
  use 2, -3
  sorry

end circle_center_coordinates_l722_722315


namespace triangle_BC_value_l722_722601

theorem triangle_BC_value
  (A B C : Type) [metric_space A B C]
  (AB BC AC: ℝ)
  (angle_B : ℝ)
  (h1 : AB = 100)
  (h2 : AC = 50 * real.sqrt 2)
  (h3 : angle_B = 45) :
  BC = 100 * real.sqrt 1.5 := 
sorry

end triangle_BC_value_l722_722601


namespace valid_rod_count_l722_722242

open Nat

theorem valid_rod_count :
  ∃ valid_rods : Finset ℕ,
    (∀ d ∈ valid_rods, 6 ≤ d ∧ d < 35 ∧ d ≠ 5 ∧ d ≠ 10 ∧ d ≠ 20) ∧ 
    valid_rods.card = 26 := sorry

end valid_rod_count_l722_722242


namespace sum_log_geometric_sequence_eq_five_l722_722914

-- Definitions of the geometric sequence and its properties
variables {a : ℕ → ℝ} (r : ℝ)

-- The conditions given in the problem
axiom condition1 : a 3 = 5
axiom condition2 : a 8 = 2
axiom geom_sequence : ∀ n, a (n + 1) = a n * r

-- The statement to be proved
theorem sum_log_geometric_sequence_eq_five : 
  (∑ i in Finset.range 10, Real.log (a i)) = 5 := 
by
  sorry

end sum_log_geometric_sequence_eq_five_l722_722914


namespace determine_relationship_l722_722956

noncomputable def polar_to_cartesian_eq_c : ℝ → ℝ → Prop := 
  λ x y, x^2 + y^2 - 2 * y = 0

noncomputable def parametric_to_cartesian_eq_l : ℝ → ℝ → Prop := 
  λ x y, 4 * x + 3 * y - 8 = 0

noncomputable def tangent_relation : Prop := 
  ∃ x y, polar_to_cartesian_eq_c x y ∧ parametric_to_cartesian_eq_l x y

theorem determine_relationship : 
  (∀ (ρ θ : ℝ), (ρ = 2 * sin θ) → ∀ t : ℝ, (x = -3 * t + 2) → (y = 4 * t) → 
    (polar_to_cartesian_eq_c x y) ∧ (parametric_to_cartesian_eq_l x y)) → tangent_relation :=
by
  intro h
  sorry

end determine_relationship_l722_722956


namespace graph_f1_entirely_above_graph_f2_l722_722454

def f1 (x : ℝ) : ℝ := abs (x^2 - (3/2) * x + 3)
def f2 (x : ℝ) : ℝ := x^2 + (3/2) * x + 3

theorem graph_f1_entirely_above_graph_f2 :
  ∀ x : ℝ, f1 x ≥ f2 x :=
by
  sorry

end graph_f1_entirely_above_graph_f2_l722_722454


namespace solve_dog_walking_minutes_l722_722967

-- Definitions based on the problem conditions
def cost_one_dog (x : ℕ) : ℕ := 20 + x
def cost_two_dogs : ℕ := 54
def cost_three_dogs : ℕ := 87
def total_earnings (x : ℕ) : ℕ := cost_one_dog x + cost_two_dogs + cost_three_dogs

-- Proving that the total earnings equal to 171 implies x = 10
theorem solve_dog_walking_minutes (x : ℕ) (h : total_earnings x = 171) : x = 10 :=
by
  -- The proof goes here
  sorry

end solve_dog_walking_minutes_l722_722967


namespace time_to_pay_back_l722_722615

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l722_722615


namespace positive_whole_numbers_with_cube_roots_less_than_15_l722_722121

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l722_722121


namespace expression_evaluation_l722_722139

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l722_722139


namespace count_cube_roots_less_than_15_l722_722126

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l722_722126


namespace glass_panels_in_neighborhood_l722_722419

def total_glass_panels_in_neighborhood := 
  let double_windows_downstairs : ℕ := 6
  let glass_panels_per_double_window_downstairs : ℕ := 4
  let single_windows_upstairs : ℕ := 8
  let glass_panels_per_single_window_upstairs : ℕ := 3
  let bay_windows : ℕ := 2
  let glass_panels_per_bay_window : ℕ := 6
  let houses : ℕ := 10

  let glass_panels_in_one_house : ℕ := 
    (double_windows_downstairs * glass_panels_per_double_window_downstairs) +
    (single_windows_upstairs * glass_panels_per_single_window_upstairs) +
    (bay_windows * glass_panels_per_bay_window)

  houses * glass_panels_in_one_house

theorem glass_panels_in_neighborhood : total_glass_panels_in_neighborhood = 600 := by
  -- Calculation steps skipped
  sorry

end glass_panels_in_neighborhood_l722_722419


namespace sqrt_expression_value_l722_722718

theorem sqrt_expression_value : sqrt (36 * sqrt (27 * sqrt 9)) = 18 := by
  sorry

end sqrt_expression_value_l722_722718


namespace sqrt_recursive_value_l722_722846

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722846


namespace ellipse_standard_equation_exists_fixed_point_l722_722518

-- Definitions based on given conditions
def ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Given conditions
def a : ℝ := sqrt 6
def b : ℝ := sqrt 2
def left_focus_dist : ℝ := sqrt 6 - 2
def major_axis_length : ℝ := 2 * sqrt 6

-- Goals to prove
theorem ellipse_standard_equation :
  ellipse a b 6 2 :=
sorry

theorem exists_fixed_point :
  ∃ (E : ℝ × ℝ), E = (7/3, 0) ∧
  ∀ (A B : ℝ × ℝ), ∃ (k : ℝ), k ≠ 0 ∧ (some additional property) ∧
  (let EA := ⟨fst A - fst E, snd A - 0⟩ in
   let EB := ⟨fst B - fst E, snd B - 0⟩ in
   EA.1 * EB.1 + EA.2 * EB.2 = -5/9) :=
sorry

end ellipse_standard_equation_exists_fixed_point_l722_722518


namespace count_cube_roots_less_than_15_l722_722127

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l722_722127


namespace move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l722_722282

-- Define the initial conditions
def pointA := (50 : ℝ)
def radius := (1 : ℝ)
def origin := (0 : ℝ)

-- Statement for part (a)
theorem move_point_inside_with_25_reflections :
  ∃ (n : ℕ) (r : ℝ), n = 25 ∧ r = radius + 50 ∧ pointA ≤ r :=
by
  sorry

-- Statement for part (b)
theorem cannot_move_point_inside_with_24_reflections :
  ∀ (n : ℕ) (r : ℝ), n = 24 → r = radius + 48 → pointA > r :=
by
  sorry

end move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l722_722282


namespace proof_problem_l722_722267

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry

lemma inequality (x : ℝ) (h : x ∈ Ioo (-π/2) 0) : f x < (f' x) * tan x := sorry

theorem proof_problem {π : ℝ} (h1 : odd_function f) (h2 : ∀ x ∈ Ioo (-π/2) 0, inequality x h2) :
  f (-π/3) > -√3 * f (π/6) := sorry

end proof_problem_l722_722267


namespace limit_f_derivative_at_1_l722_722539

def f (x : Real) : Real := 2 * Real.log (3 * x) + 8 * x

theorem limit_f_derivative_at_1 : 
  (Real.lim (fun Δx ↦ (f (1 + Δx) - f 1) / Δx) (0 : Real) = 10) :=
sorry

end limit_f_derivative_at_1_l722_722539


namespace physics_marks_l722_722381

theorem physics_marks (P C M : ℕ) 
  (h1 : (P + C + M) = 255)
  (h2 : (P + M) = 180)
  (h3 : (P + C) = 140) : 
  P = 65 :=
by
  sorry

end physics_marks_l722_722381


namespace number_of_distinct_prime_factors_30_fact_l722_722996

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722996


namespace circumscribing_sphere_radius_l722_722309

theorem circumscribing_sphere_radius (r : ℝ) (ρ : ℝ) (R : ℝ) 
  (h₁ : ρ = real.sqrt 6 - 1)
  (h₂ : ∀ (r_ : ℝ) (R_ : ℝ), R_ = (5 * (real.sqrt 2 + 1)) * r_)
  (h₃ : r_ = ρ) : 
  R = 5 * (real.sqrt 2 + 1) := 
by 
  sorry

end circumscribing_sphere_radius_l722_722309


namespace generatrix_length_l722_722035

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722035


namespace div_240_of_prime_diff_l722_722247

-- Definitions
def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p = n ∧ Prime p
def prime_with_two_digits (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- The theorem statement
theorem div_240_of_prime_diff (a b : ℕ) (ha : prime_with_two_digits a) (hb : prime_with_two_digits b) (h : a > b) :
  240 ∣ (a^4 - b^4) ∧ ∀ d : ℕ, (d ∣ (a^4 - b^4) → (∀ m n : ℕ, prime_with_two_digits m → prime_with_two_digits n → m > n → d ∣ (m^4 - n^4) ) → d ≤ 240) :=
by
  sorry

end div_240_of_prime_diff_l722_722247


namespace math_problem_l722_722254

theorem math_problem
  (m : ℕ) (h₁ : m = 8^126) :
  (m * 16) / 64 = 16^94 :=
by
  sorry

end math_problem_l722_722254


namespace cube_root_numbers_less_than_15_l722_722099

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l722_722099


namespace g_monotonically_increasing_l722_722953

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem g_monotonically_increasing (x : ℝ) :
  -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 3 →
  ∀ (y : ℝ), (-Real.pi / 12 ≤ y ∧ y ≤ x) → g y ≤ g x :=
sorry

end g_monotonically_increasing_l722_722953


namespace max_n_satisfies_condition_l722_722634

theorem max_n_satisfies_condition :
  ∃ n : ℕ, (∃ a : ℕ, 9 * n^2 + 5 * n + 26 = a * (a + 1)) ∧
  ∀ n₁ : ℕ, (∃ a₁ : ℕ, 9 * n₁^2 + 5 * n₁ + 26 = a₁ * (a₁ + 1)) → n₁ ≤ n :=
begin
  sorry
end

end max_n_satisfies_condition_l722_722634


namespace non_real_roots_interval_l722_722188

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722188


namespace sequence_equality_l722_722653

-- Noncomputable definitions as we are dealing with sequences not necessarily computable directly
noncomputable def a : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 1) := 2018 * a n / n + a (n - 1)

noncomputable def b : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 1) := 2020 * b n / n + b (n - 1)

theorem sequence_equality :
  a 1010 / 1010 = b 1009 / 1009 :=
sorry

end sequence_equality_l722_722653


namespace cone_generatrix_length_is_2sqrt2_l722_722027

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722027


namespace degree_of_d_l722_722562

-- Given definitions
def f := Polynomial ℝ -- A  polynomial f with degree 15.
def q := Polynomial ℝ -- A polynomial q with degree 8.
def r := Polynomial ℝ := (2:ℝ)*X^3 + (5:ℝ)*X^2 + X + (7:ℝ) -- The remainder r with degree 3.

-- Conditions as Lean definitions.
def deg_f := 15
def deg_q := 8
def deg_r := 3
def f_eq := ∀ d : Polynomial ℝ, f = d * q + r

-- The theorem to prove.
theorem degree_of_d (d : Polynomial ℝ): f_eq d → d.degree = 7 := 
by 
  sorry

end degree_of_d_l722_722562


namespace hyperbola_eccentricity_proof_l722_722093

-- Defining the hyperbola and its properties
variables {a b c : ℝ}
variables (ha : a > 0) (hb : b > 0)

-- Asymptote equations
def asymptote1 (x : ℝ) : ℝ := (b / a) * x
def asymptote2 (x : ℝ) : ℝ := (-b / a) * x

-- Given symmetric point of the right focus with respect to l1 lies on l2
def symmetric_point_lies_on_l2 (c : ℝ) : Prop :=
  let F : ℝ × ℝ := (c, 0) in
  let M : ℝ × ℝ := (-c / 2, (b * c) / (2 * a)) in
  asymptote2 M.1 = M.2

-- Eccentricity of the hyperbola
def eccentricity (a b c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity_proof :
  symmetric_point_lies_on_l2 a b c ha hb →
  b^2 = 3 * a^2 →
  c^2 = 4 * a^2 →
  eccentricity a b c = 2 :=
by
  sorry

end hyperbola_eccentricity_proof_l722_722093


namespace simplify_trig_expression_l722_722290

theorem simplify_trig_expression :
  (sin (40 * Real.pi / 180) + sin (80 * Real.pi / 180)) / 
  (cos (40 * Real.pi / 180) + cos (80 * Real.pi / 180)) = 
  Real.sqrt 3 := sorry

end simplify_trig_expression_l722_722290


namespace max_value_of_n_l722_722760

theorem max_value_of_n (n : ℕ) : n ≤ 5 :=
by
-- Define conditions
let three_digit_numbers (a : ℕ) := a / 100 + (a / 10 % 10) + (a % 10) = 9
let no_zero_digits (a : ℕ) := a / 100 ≠ 0 ∧ a / 10 % 10 ≠ 0 ∧ a % 10 ≠ 0
let no_shared_digits (list_of_nums : list ℕ) :=
  ∀ (i j : ℕ) (a b c d e f : ℕ),
    i ≠ j → 
    list_of_nums !! i = some a → list_of_nums !! j = some b → 
    a / 100 = c → a / 10 % 10 = d → a % 10 = e →
    b / 100 ≠ c ∧ b / 10 % 10 ≠ d ∧ b % 10 ≠ e

-- Set list of numbers
let nums := list.range n

-- All constraints combined
h1 : ∀ a ∈ nums, three_digit_numbers a := sorry
h2 : ∀ a ∈ nums, no_zero_digits a := sorry
h3 : no_shared_digits nums := sorry

-- Show final bound
show n ≤ 5, by sorry

end max_value_of_n_l722_722760


namespace probability_two_balls_one_white_one_red_l722_722701

/-- There are 15 balls in a bag: 10 white and 5 red. The probability of drawing two balls 
    from the bag such that one is white and one is red is 10/21. We formalize this statement 
    and provide the necessary conditions and definitions in Lean 4. -/
theorem probability_two_balls_one_white_one_red :
  let total_balls := 15 in
  let white_balls := 10 in
  let red_balls := 5 in
  let total_ways := Nat.choose total_balls 2 in
  let white_red_ways := Nat.choose white_balls 1 * Nat.choose red_balls 1 in
  (white_red_ways : ℚ) / total_ways = 10 / 21 := 
by
  sorry

end probability_two_balls_one_white_one_red_l722_722701


namespace M_is_real_l722_722260

open Complex

-- Define the condition that characterizes the set M
def M (Z : ℂ) : Prop := (Z - 1)^2 = abs (Z - 1)^2

-- Prove that M is exactly the set of real numbers
theorem M_is_real : ∀ (Z : ℂ), M Z ↔ Z.im = 0 :=
by
  sorry

end M_is_real_l722_722260


namespace generatrix_length_of_cone_l722_722020

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722020


namespace max_units_of_material_A_l722_722376

theorem max_units_of_material_A (x y z : ℕ) 
    (h1 : 3 * x + 5 * y + 7 * z = 62)
    (h2 : 2 * x + 4 * y + 6 * z = 50) : x ≤ 5 :=
by
    sorry 

end max_units_of_material_A_l722_722376


namespace non_real_roots_interval_l722_722178

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l722_722178


namespace spherical_to_rectangular_equiv_l722_722424

variables {ρ θ φ : ℝ}

def rect_coord (x y z ρ θ φ : ℝ) : Prop :=
  x = ρ * sin φ * cos θ ∧ y = ρ * sin φ * sin θ ∧ z = ρ * cos φ

theorem spherical_to_rectangular_equiv (h : rect_coord 3 (-4) 2 ρ θ φ) :
  rect_coord (-3) 4 2 ρ (θ + Real.pi) φ :=
by
  sorry

end spherical_to_rectangular_equiv_l722_722424


namespace box_made_by_Bellini_or_son_l722_722307

-- Definitions of the conditions
variable (B : Prop) -- Bellini made the box
variable (S : Prop) -- Bellini's son made the box
variable (inscription_true : Prop) -- The inscription "I made this box" is truthful

-- The problem statement in Lean: Prove that B or S given the inscription is true
theorem box_made_by_Bellini_or_son (B S inscription_true : Prop) (h1 : inscription_true → (B ∨ S)) : B ∨ S :=
by
  sorry

end box_made_by_Bellini_or_son_l722_722307


namespace joseph_decks_l722_722241

theorem joseph_decks (total_cards : ℕ) (cards_per_deck : ℕ) (expected_decks : ℕ) : 
  total_cards = 208 → cards_per_deck = 52 → expected_decks = 4 → total_cards / cards_per_deck = expected_decks := 
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end joseph_decks_l722_722241


namespace proof_problem_l722_722519

variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f x

-- Condition 2: The derivative f' exists
def has_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ f' : ℝ → ℝ, true

-- Condition 3: For x in (-∞, 0], xf'(x) < f(-x) holds
def condition3 (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x ∈ Iic 0, x * f' x < f (-x)

-- Define F(x) = x f(x)
def F (x : ℝ) : ℝ :=
  x * f x

theorem proof_problem
  (hf : is_odd f)
  (hf_deriv : has_derivative f)
  (hf_cond3 : ∀ f' : ℝ → ℝ, condition3 f f') :
  ∀ x, -1 < x ∧ x < 2 → F f 3 > F f (2 * x - 1) := 
sorry

end proof_problem_l722_722519


namespace cone_generatrix_length_l722_722052

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722052


namespace find_height_of_door_l722_722225

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l722_722225


namespace circle_radius_in_arc_triangle_l722_722688

theorem circle_radius_in_arc_triangle (d : ℝ) (A B C : Point)
  (h_dist : dist A B = d)
  (h_dist' : dist B C = d)
  (h_dist'' : dist C A = d) :
  ∃ (r : ℝ), (k_1 k_2 k_3 : Circle) (O_1 O_2 O_3 : Point),
  dist O_1 O_2 = r ∧ dist O_2 O_3 = r ∧ dist O_3 O_1 = r ∧
  dist O_1 B = d - r ∧
  dist O_2 A = d - r ∧
  dist O_3 C = d - r ∧
  r = d * ((3 * sqrt 2) - 4) :=
sorry

end circle_radius_in_arc_triangle_l722_722688


namespace bisection_method_root_interval_l722_722348

noncomputable def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 3 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  sorry

end bisection_method_root_interval_l722_722348


namespace ellipse_eccentricity_l722_722929

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (B F A C : ℝ × ℝ) 
    (h3 : (B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1))
    (h4 : (C.1 ^ 2 / a ^ 2 + C.2 ^ 2 / b ^ 2 = 1))
    (h5 : B.1 > 0 ∧ B.2 > 0)
    (h6 : C.1 > 0 ∧ C.2 > 0)
    (h7 : ∃ M : ℝ × ℝ, M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ (F = M)) :
    ∃ e : ℝ, e = (1 / 3) := 
  sorry

end ellipse_eccentricity_l722_722929


namespace number_of_prime_factors_30_factorial_l722_722986

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722986


namespace normal_curve_peak_position_l722_722942

-- Define the conditions
def normal_distribution {μ σ : ℝ} (x : ℝ) : Prop :=
  @MeasureTheory.ProbabilityDistribution.normal ℝ _ _ μ σ x

-- The problem statement
theorem normal_curve_peak_position :
  ∀ μ σ : ℝ,
  (∀ x : ℝ, normal_distribution μ σ x → MeasureTheory.MeasurableSet (Ioi 0.2) ∧ 
    MeasureTheory.MeasureSpace.measure (Ioi 0.2) = 0.5) →
  μ = 0.2 :=
by
  sorry

end normal_curve_peak_position_l722_722942


namespace find_z_l722_722191

theorem find_z (x : ℕ) (z : ℚ) (h1 : x = 103)
               (h2 : x^3 * z - 3 * x^2 * z + 2 * x * z = 208170) 
               : z = 5 / 265 := 
by 
  sorry

end find_z_l722_722191


namespace pages_with_same_units_digit_l722_722400

theorem pages_with_same_units_digit:
  (∃ k : ℕ, k = 7) ∧
  (∀ x : ℕ, x ∈ {1, 2, ..., 61} → 
  (x % 10 = (62 - x) % 10 → x ∈ {31, 36, 41, 46, 51, 56, 61})) ∧ 
  (∀ x ∈ {31, 36, 41, 46, 51, 56, 61}, x % 10 = (62 - x) % 10) :=
by {
  sorry
}

end pages_with_same_units_digit_l722_722400


namespace pi_is_irrational_l722_722796

theorem pi_is_irrational (π : ℝ) (h : π = Real.pi) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ π = a / b :=
by
  sorry

end pi_is_irrational_l722_722796


namespace real_possible_b_values_quadratic_non_real_roots_l722_722143

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722143


namespace projection_of_a_onto_b_is_three_l722_722963

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (1, 0)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem projection_of_a_onto_b_is_three : projection vec_a vec_b = 3 := by
  sorry

end projection_of_a_onto_b_is_three_l722_722963


namespace first_day_of_month_l722_722670

-- Define the days of the week as an enumeration
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Given condition
def eighteenth_day_is : Day := Wednesday

-- Theorem to prove
theorem first_day_of_month : 
  (eighteenth_day_is = Wednesday) → (eighteenth_day_is = Wednesday) := 
by
  intros h,
  -- We will need to establish the reverse order from 18th day is Wednesday back to 1st day is Sunday
  sorry

end first_day_of_month_l722_722670


namespace log_4_1_div_16_eq_neg_2_l722_722468

theorem log_4_1_div_16_eq_neg_2 : log 4 (1 / 16) = -2 :=
by
  sorry

end log_4_1_div_16_eq_neg_2_l722_722468


namespace generatrix_length_l722_722039

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722039


namespace analytical_expression_of_f_extreme_values_of_f_l722_722071

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^3 + b * x^2 + c

-- Define the derivative of f
def f' (x : ℝ) (a b c : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

-- Given conditions
variables (a b c : ℝ)
variable h1 : f 0 a b c = 1
variable h2 : f 1 a b c = 1
variable h3 : f' 1 a b c = 1

-- First part: Find the analytical expression of f(x)
theorem analytical_expression_of_f : f x 1 (-1) 1 = x^3 - x^2 + 1 :=
sorry

-- Second part: Find the extreme values of f(x)
theorem extreme_values_of_f :
  (∀ x, f' x 1 (-1) 1 = 0 → f x 1 (-1) 1 = 1 ∨ f x 1 (-1) 1 = 23 / 27) :=
sorry

end analytical_expression_of_f_extreme_values_of_f_l722_722071


namespace years_ago_twice_age_l722_722698

variables (H J x : ℕ)

def henry_age : ℕ := 20
def jill_age : ℕ := 13

axiom age_sum : H + J = 33
axiom age_difference : H - x = 2 * (J - x)

theorem years_ago_twice_age (H := henry_age) (J := jill_age) : x = 6 :=
by sorry

end years_ago_twice_age_l722_722698


namespace right_triangle_short_leg_l722_722581

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l722_722581


namespace max_tiles_fit_l722_722386

theorem max_tiles_fit (tile_length tile_width floor_length floor_width : ℕ) (h1 : tile_length = 50) (h2 : tile_width = 40) (h3 : floor_length = 120) (h4 : floor_width = 150) 
                      (tiles_in_orientation1 tiles_in_orientation2 : ℕ) (h5 : tiles_in_orientation1 = (2 * 3)) (h6 : tiles_in_orientation2 = (3 * 3)) :
  max tiles_in_orientation1 tiles_in_orientation2 = 9 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  simp
  sorry

end max_tiles_fit_l722_722386


namespace log_four_one_div_sixteen_l722_722466

theorem log_four_one_div_sixteen : log 4 (1 / 16) = -2 := 
by 
  sorry

end log_four_one_div_sixteen_l722_722466


namespace evaluate_nested_radical_l722_722856

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722856


namespace quadratic_non_real_roots_iff_l722_722153

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722153


namespace integral_value_l722_722814

noncomputable def f (x : ℝ) := x^2 / (x^2 + 2*x + 2)

theorem integral_value : ∫ x in -1..1, f x = 0.4 := sorry

end integral_value_l722_722814


namespace perimeter_of_rectangle_l722_722784

theorem perimeter_of_rectangle
  (L : ℝ) (B : ℝ) (hL : L = 260) (hB : B = 190) :
  2 * (L + B) = 900 :=
by
  -- Let's assume L and B satisfy the given equations
  rw [hL, hB]
  -- Simplify to compute the perimeter
  norm_num
  sorry

end perimeter_of_rectangle_l722_722784


namespace prob_A_inter_B_l722_722741

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l722_722741


namespace average_of_k_l722_722075

theorem average_of_k (k r1 r2: ℕ) (h1 : r1 * r2 = 24) (h2 : r1 + r2 = k) (h3 : r1 > 0) (h4 : r2 > 0) : 
  let ks := [25, 14, 11, 10] in
  (ks.sum / ks.length : ℝ) = 15 := by 
sorry

end average_of_k_l722_722075


namespace Melissa_commission_l722_722275

theorem Melissa_commission 
  (coupe_price : ℝ)
  (suv_multiplier : ℝ)
  (commission_rate : ℝ) :
  (coupe_price = 30000) →
  (suv_multiplier = 2) →
  (commission_rate = 0.02) →
  let suv_price := suv_multiplier * coupe_price in
  let total_sales := coupe_price + suv_price in
  let commission := commission_rate * total_sales in
  commission = 1800 :=
begin
  sorry
end

end Melissa_commission_l722_722275


namespace find_a_squared_plus_b_squared_l722_722504

theorem find_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a - b = 8)
  (h2 : a * b = 20) :
  a^2 + b^2 = 104 :=
begin
  sorry
end

end find_a_squared_plus_b_squared_l722_722504


namespace shorter_leg_of_right_triangle_l722_722575

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l722_722575


namespace melissa_earnings_from_sales_l722_722273

noncomputable def commission_earned (coupe_price suv_price commission_rate : ℕ) : ℕ :=
  (coupe_price + suv_price) * commission_rate / 100

theorem melissa_earnings_from_sales : 
  commission_earned 30000 60000 2 = 1800 :=
by
  sorry

end melissa_earnings_from_sales_l722_722273


namespace find_angle_BDC_l722_722226

theorem find_angle_BDC (A B C D : Type)
  [EquilateralTriangle : Triangle A B C]
  (h1 : Distance A B = Distance A D)
  (h2 : Angle A B C = 60) :
  Angle B D C = 30 :=
sorry

end find_angle_BDC_l722_722226


namespace histogram_height_representation_l722_722797

theorem histogram_height_representation (freq_ratio : ℝ) (frequency : ℝ) (class_interval : ℝ) 
  (H : freq_ratio = frequency / class_interval) : 
  freq_ratio = frequency / class_interval :=
by 
  sorry

end histogram_height_representation_l722_722797


namespace cone_generatrix_length_is_2sqrt2_l722_722031

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722031


namespace minimum_rooms_l722_722389

theorem minimum_rooms (students_A students_B : ℕ) (hA : students_A = 72) (hB : students_B = 5824) :
  let GCD := Nat.gcd students_A students_B
  let rooms_A := students_A / GCD
  let rooms_B := students_B / GCD
  rooms_A + rooms_B = 737 :=
by
  rw [hA, hB]
  have GCD_def : GCD = Nat.gcd 72 5824 := rfl
  rw GCD_def
  have : Nat.gcd 72 5824 = 8 := by sorry
  rw this at *
  have rooms_A_def : rooms_A = 72 / 8 := rfl
  have rooms_B_def : rooms_B = 5824 / 8 := rfl
  rw [rooms_A_def, rooms_B_def]
  norm_num

end minimum_rooms_l722_722389


namespace cube_root_numbers_less_than_15_l722_722100

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l722_722100


namespace triangle_BHD_equilateral_l722_722832

variables (A B C D E H K : Type)
variables [affine_space A] [affine_space B] [affine_space C] [affine_space D] [affine_space E] [affine_space H] [affine_space K]

-- Definitions of equilateral triangles
variable (is_equilateral_triangle : ∀ (X Y Z : Type), Prop)

def equilateral_triangle_ABC := is_equilateral_triangle A B C
def equilateral_triangle_CDE := is_equilateral_triangle C D E
def equilateral_triangle_EHK := is_equilateral_triangle E H K

-- Condition: Given vectors are equal
def equal_vectors_AD_DK := (vector A D = vector D K)

-- Goal: Prove triangle BHD is equilateral
theorem triangle_BHD_equilateral 
  (h1 : equilateral_triangle_ABC)
  (h2 : equilateral_triangle_CDE)
  (h3 : equilateral_triangle_EHK)
  (h4 : equal_vectors_AD_DK) :
  is_equilateral_triangle B H D := 
sorry

end triangle_BHD_equilateral_l722_722832


namespace paint_area_of_open_box_l722_722770

-- Define the dimensions of the box
def length := 18
def width := 10
def height := 2

-- Define the problem as a theorem
theorem paint_area_of_open_box : 
  let side1 := 2 * (length * height)
  let side2 := 2 * (width * height)
  let bottom := (length * width)
  side1 + side2 + bottom = 292 := by sorry

end paint_area_of_open_box_l722_722770


namespace find_smaller_number_l722_722334

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l722_722334


namespace common_ratio_of_geometric_series_l722_722896

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l722_722896


namespace midpoint_proof_l722_722722

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (A B : Point) : ℝ :=
(real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2))

def midpoint (A B C : Point) : Prop :=
distance A C = distance C B ∧ (distance A C + distance C B = distance A B)

theorem midpoint_proof (A B C : Point) :
  distance A B = 2 * distance A C ∧ distance A B = 2 * distance C B →
  midpoint A B C :=
by
  intros h
  sorry

end midpoint_proof_l722_722722


namespace find_g_l722_722319

noncomputable def g : ℝ → ℝ
| x => 2 * (4^x - 3^x)

theorem find_g :
  (g 1 = 2) ∧
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) →
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end find_g_l722_722319


namespace perpendicular_bisector_eq_l722_722094

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the midpoint M of the line segment AB
def M : ℝ × ℝ := ((fst A + fst B) / 2, (snd A + snd B) / 2)

-- Define the slope of AB
def slope_AB : ℝ := (snd B - snd A) / (fst B - fst A)

-- Define the slope of the perpendicular bisector of AB
def slope_perpendicular_bisector : ℝ := -(1 / slope_AB)

-- A formal statement to prove
theorem perpendicular_bisector_eq :
  (∀ x y : ℝ, y - 2 = slope_perpendicular_bisector * (x + 1) → x + 2 * y - 3 = 0) :=
  sorry

end perpendicular_bisector_eq_l722_722094


namespace problem_1_problem_2_l722_722326

theorem problem_1 :
  83 * 87 = 100 * 8 * (8 + 1) + 21 :=
by sorry

theorem problem_2 (n : ℕ) :
  (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 :=
by sorry

end problem_1_problem_2_l722_722326


namespace sqrt_diff_eq_neg_sixteen_l722_722446

theorem sqrt_diff_eq_neg_sixteen : 
  (Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2)) = -16 := 
  sorry

end sqrt_diff_eq_neg_sixteen_l722_722446


namespace proof_problem_l722_722004

noncomputable def circle (x y : ℝ) : Prop := (x-2)^2 + (y - (5/2))^2 = 25/4

def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (4, 4)
def on_x_axis (x : ℝ) : ℝ × ℝ := (x, 0)

def length_of_chord_intercepted_by_y_axis : ℝ := 3
def is_collinear (a b c : ℝ × ℝ) : Prop := (b.2 - a.2) * (c.1 - b.1) = (b.1 - a.1) * (c.2 - b.2)
def max_angle_AMB := Real.pi / 2

theorem proof_problem :
  (∃ x, circle (0) (1)) ∧
  (∃ x, circle (4) (4)) ∧
  ((∃ x, on_x_axis x) → length_of_chord_intercepted_by_y_axis = 3) ∧
  (is_collinear point_A point_B (2, 5/2)) ∧
  (∀ x, on_x_axis x → max_angle_AMB = Real.pi / 2) :=
sorry

end proof_problem_l722_722004


namespace complex_quadrant_l722_722521

theorem complex_quadrant :
  let z : ℂ := (1 + complex.I) / real.sqrt 2 in
  z ^ 2015.0.re > 0 ∧ z ^ 2015.0.im < 0 := 
sorry

end complex_quadrant_l722_722521


namespace face_value_is_100_l722_722431

-- Definitions based on conditions
def faceValue (F : ℝ) : Prop :=
  let discountedPrice := 0.92 * F
  let brokerageFee := 0.002 * discountedPrice
  let totalCostPrice := discountedPrice + brokerageFee
  totalCostPrice = 92.2

-- The proof statement in Lean
theorem face_value_is_100 : ∃ F : ℝ, faceValue F ∧ F = 100 :=
by
  use 100
  unfold faceValue
  simp
  norm_num
  sorry

end face_value_is_100_l722_722431


namespace real_possible_b_values_quadratic_non_real_roots_l722_722144

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l722_722144


namespace sqrt_continued_fraction_l722_722863

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722863


namespace probability_intersection_l722_722755

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l722_722755


namespace nested_radical_solution_l722_722835

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722835


namespace number_of_prime_factors_30_factorial_l722_722983

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722983


namespace line_circle_intersect_l722_722954

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end line_circle_intersect_l722_722954


namespace number_of_black_and_white_films_l722_722406

theorem number_of_black_and_white_films (B x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_fraction : (6 * y : ℚ) / ((y / (x : ℚ))/100 * (B : ℚ) + 6 * y) = 20 / 21) :
  B = 30 * x :=
sorry

end number_of_black_and_white_films_l722_722406


namespace quadratic_non_real_roots_iff_l722_722152

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l722_722152


namespace count_sets_with_six_l722_722341

-- Define the set S
def S : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the condition that a, b, c should be distinct elements from S, whose sum is 18
def valid_set (a b c : ℕ) := a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b + c = 18

-- Define the main problem statement
theorem count_sets_with_six : 
  (Finset.filter (λ s : Finset ℕ, 6 ∈ s ∧ ∃ a b c, {a, b, c} = s ∧ valid_set a b c) 
    (Finset.powersetLen 3 S.to_finset)).card = 4 :=
sorry

end count_sets_with_six_l722_722341


namespace number_of_elements_in_intersection_l722_722959

def A : Set ℝ := { x | |x| < 2 }
def B : Set ℝ := { -1, 0, 1, 2, 3 }

theorem number_of_elements_in_intersection : (A ∩ B).card = 3 := by
  sorry

end number_of_elements_in_intersection_l722_722959


namespace expression_simplifies_to_49_l722_722136

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l722_722136


namespace sum_of_oranges_l722_722371

theorem sum_of_oranges (N : ℕ) :
  ((∀ N < 100, N ≡ 5 [MOD 6] ∧ N ≡ 7 [MOD 8])
     → ∑ N in { N | N < 100 ∧ N % 6 = 5 ∧ N % 8 = 7 } = 236) :=
by
  sorry

end sum_of_oranges_l722_722371


namespace sponge_area_cleaned_l722_722787

/-- A semicircular sponge with a diameter of 20 cm is used to wipe a corner of a room's floor such
that the ends of the diameter continuously touch the two walls forming a right angle.
Prove that the area cleaned by the sponge is 100π cm². -/
theorem sponge_area_cleaned (d : ℝ) (r : ℝ) (A B : ℝ) (semicircle : set (ℝ × ℝ)) 
  (diameter_valid : d = 20)
  (radius_valid : r = d / 2) 
  (area_valid : A = π * r^2)
  (cleaned_area_valid : B = A / 4) :
  B = 100 * π := 
sorry

end sponge_area_cleaned_l722_722787


namespace car_speed_l722_722729

-- Define the given conditions 
def distance : ℝ := 624
def time : ℝ := 3 + 1 / 5

-- Define the expected speed calculation
def expected_speed : ℝ := distance / time

-- State the theorem to be proved
theorem car_speed :
  expected_speed = 195 := by
  -- The actual proof would go here
  sorry

end car_speed_l722_722729


namespace smaller_successive_number_l722_722388

noncomputable def solve_successive_numbers : ℕ :=
  let n := 51
  n

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 2652) : n = solve_successive_numbers :=
  sorry

end smaller_successive_number_l722_722388


namespace greatest_integer_whose_square_is_81_more_than_thrice_its_value_l722_722713

theorem greatest_integer_whose_square_is_81_more_than_thrice_its_value
  (x : ℤ) (h : x^2 = 3 * x + 81) : x ≤ 9 :=
by
  sorry

example : (∃ x : ℤ, x ^ 2 = 3 * x + 81 ∧ x = 9) :=
by
  use 9
  split
  · norm_num
  · norm_num

end greatest_integer_whose_square_is_81_more_than_thrice_its_value_l722_722713


namespace nested_sqrt_eq_l722_722869

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722869


namespace housewife_saving_percentage_approx_l722_722412

noncomputable def original_price (spent : ℝ) (saved : ℝ) : ℝ := spent + saved

noncomputable def percentage_saved (spent : ℝ) (saved : ℝ) : ℝ :=
  (saved / original_price spent saved) * 100

theorem housewife_saving_percentage_approx :
  percentage_saved 25 2.5 ≈ 9 :=
by
  sorry

end housewife_saving_percentage_approx_l722_722412


namespace positive_whole_numbers_with_cube_roots_less_than_15_l722_722117

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l722_722117


namespace rate_of_rainfall_is_one_l722_722605

variable (R : ℝ)
variable (h1 : 2 + 4 * R + 4 * 3 = 18)

theorem rate_of_rainfall_is_one : R = 1 :=
by
  sorry

end rate_of_rainfall_is_one_l722_722605


namespace log_4_1_div_16_eq_neg_2_l722_722470

theorem log_4_1_div_16_eq_neg_2 : log 4 (1 / 16) = -2 :=
by
  sorry

end log_4_1_div_16_eq_neg_2_l722_722470


namespace cone_generatrix_length_is_2sqrt2_l722_722029

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722029


namespace generatrix_length_l722_722036

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722036


namespace expected_number_of_remaining_bullets_l722_722429

-- Conditions given in the problem
def hit_rate : ℝ := 0.6
def total_bullets : ℕ := 4
def probability_of_miss : ℝ := 1 - hit_rate

-- Definition of possible values of remaining bullets and their probabilities
def prob_ξ (k : ℕ) : ℝ :=
  match k with
  | 0 => probability_of_miss ^ 3 * hit_rate
  | 1 => probability_of_miss ^ 2 * hit_rate * probability_of_miss
  | 2 => probability_of_miss * hit_rate * probability_of_miss
  | 3 => hit_rate
  | _ => 0

-- Expected value calculation
def expected_ξ : ℝ :=
  (0 * prob_ξ 0) + (1 * prob_ξ 1) + (2 * prob_ξ 2) + (3 * prob_ξ 3)

-- The proposition to prove
theorem expected_number_of_remaining_bullets : expected_ξ = 2.376 := by
  sorry

end expected_number_of_remaining_bullets_l722_722429


namespace height_of_parallelogram_l722_722483

theorem height_of_parallelogram
  (A B H : ℝ)
  (h1 : A = 480)
  (h2 : B = 32)
  (h3 : A = B * H) : 
  H = 15 := sorry

end height_of_parallelogram_l722_722483


namespace count_cube_roots_less_than_15_l722_722122

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l722_722122


namespace cone_generatrix_length_is_2sqrt2_l722_722032

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l722_722032


namespace sum_series_common_fraction_l722_722476

theorem sum_series_common_fraction :
  (\sum n in (Finset.Icc 1 14).filter (λ x, x ≠ 2), (1 : ℚ) / (n * (n + 1))) = 14 / 15 := 
sorry

end sum_series_common_fraction_l722_722476


namespace trig_func_2014_2017_l722_722083

open Real

def f (x : ℝ) : ℝ := sin (π / 5 * x - 7 * π / 10)

theorem trig_func_2014_2017 :
  f 2014 - f 2017 < 0 :=
sorry

end trig_func_2014_2017_l722_722083


namespace water_tank_full_capacity_l722_722407

-- Define the conditions
variable {C x : ℝ}
variable (h1 : x / C = 1 / 3)
variable (h2 : (x + 6) / C = 1 / 2)

-- Prove that C = 36
theorem water_tank_full_capacity : C = 36 :=
by
  sorry

end water_tank_full_capacity_l722_722407


namespace sin_alpha_beta_l722_722536

variables {α β : Real}

-- Conditions
def circle (x y : Real) : Prop := x^2 + y^2 = 1
def line (x y : Real) (m : Real) : Prop := y = 2 * x + m

-- Points of intersection
variables {x1 y1 x2 y2 m : Real}
variable h_int1 : circle x1 y1 ∧ line x1 y1 m
variable h_int2 : circle x2 y2 ∧ line x2 y2 m

-- Angles α and β are the angles made by OA and OB with the positive x-axis respectively
def angle_α (x y : Real) := Real.atan2 y x
def angle_β (x y : Real) := Real.atan2 y x

axiom α_def : α = angle_α x1 y1
axiom β_def : β = angle_β x2 y2

-- Theorem to prove
theorem sin_alpha_beta : Real.sin (α + β) = -4/5 :=
  by sorry

end sin_alpha_beta_l722_722536


namespace generatrix_length_of_cone_l722_722045

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722045


namespace find_height_of_door_l722_722223

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l722_722223


namespace number_of_prime_factors_30_factorial_l722_722974

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722974


namespace probability_intersection_l722_722747

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l722_722747


namespace part1_part2_l722_722950

-- Define f(x)
def f (x a : ℝ) := a * exp (2 * x - 1) - x^2 * (log x + 1 / 2)

-- Part (1): Prove f(x) >= x^2/2 - x^3 for a = 0
theorem part1 (a : ℝ) (x : ℝ) (hx : 0 < x) (h₁ : a = 0) : 
  f x a ≥ x^2 / 2 - x^3 := sorry

-- Define g(x)
def g (x a : ℝ) := x * f x a + x^2 / exp x

-- Part (2): Prove the range of a
theorem part2 (a x : ℝ) (hx : x > 1) 
  (h2 : ∀ x > 1, x * g (log x / (x - 1)) a < g (x * log x / (x - 1)) a) :
  a ≥ 1 / exp 1 := sorry

end part1_part2_l722_722950


namespace notebooks_left_over_l722_722203

theorem notebooks_left_over 
  (total_notebooks : ℕ := 1200)
  (initial_boxes : ℕ := 30)
  (new_notebooks_per_box : ℕ := 35) : 
  total_notebooks % new_notebooks_per_box = 10 := 
by
  have h1 : total_notebooks = 1200 := by rfl
  have h2 : new_notebooks_per_box = 35 := by rfl
  show 1200 % 35 = 10
  sorry

end notebooks_left_over_l722_722203


namespace S_n_expression_l722_722925

/-- 
  Given a sequence of positive terms {a_n} with sum of the first n terms represented as S_n,
  and given a_1 = 2, and given the relationship 
  S_{n+1}(S_{n+1} - 3^n) = S_n(S_n + 3^n), prove that S_{2023} = (3^2023 + 1) / 2.
-/
theorem S_n_expression
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hr : ∀ n, S (n + 1) * (S (n + 1) - 3^n) = S n * (S n + 3^n)) :
  S 2023 = (3^2023 + 1) / 2 :=
sorry

end S_n_expression_l722_722925


namespace problem_1_problem_2_l722_722474

noncomputable def poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 := sorry

theorem problem_1 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  a₁ + a₂ + a₃ + a₄ = -80 :=
sorry

theorem problem_2 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 625 :=
sorry

end problem_1_problem_2_l722_722474


namespace score_order_l722_722570

variable (A B C D : ℕ)

theorem score_order (h1 : A + B = C + D) (h2 : C + A > B + D) (h3 : C > A + B) :
  (C > A ∧ A > B ∧ B > D) :=
by
  sorry

end score_order_l722_722570


namespace number_of_cube_roots_lt_15_l722_722112

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l722_722112


namespace general_geometrical_propositions_not_justified_by_approximate_measurements_l722_722724

-- Definitions based on the problem conditions
def instrumental_measurements_approximate : Prop := 
  ∃ (instrument: Type) (measurement: instrument → ℝ), 
  ∀ x : instrument, ∃ ε > 0, measurement x ≠ x + ε

def general_geometrical_propositions_universal : Prop := 
  ∀ (P : Prop), (∀ (x : ℝ), P) → P

-- The main proof problem statement
theorem general_geometrical_propositions_not_justified_by_approximate_measurements:
  instrumental_measurements_approximate →
  general_geometrical_propositions_universal →
  ∀ (P : Prop), (P ∧ instrumental_measurements_approximate) → ¬ (general_geometrical_propositions_universal ∧ P) :=
by
  intros h_inst h_univ P h_imp
  sorry

end general_geometrical_propositions_not_justified_by_approximate_measurements_l722_722724


namespace nested_radical_solution_l722_722834

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l722_722834


namespace range_of_f_l722_722488

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 - x + 1)

theorem range_of_f :
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 4/3 ↔ ∃ x : ℝ, f(x) = y :=
by sorry

end range_of_f_l722_722488


namespace complex_square_sum_l722_722801

theorem complex_square_sum (n : ℕ) (x : Finₓ (n + 1) → ℝ) (hx : 2 ≤ n) 
    (h : ∑ k in Finset.range n, (x k)^2 ≥ (x 0)^2) :
    ∃ (y : Finₓ (n + 1) → ℝ), 
    (x 0 + complex.i * y 0)^2 = 
    ∑ k in Finset.range n, (x k + complex.i * y k)^2 := 
sorry

end complex_square_sum_l722_722801


namespace unit_digit_is_3_l722_722490

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_factorials (k : ℕ) : ℕ :=
  (List.range k).sum.map factorial

def unit_digit_of_power_sum (a b c : ℕ) : ℕ :=
  let sum_mod_10 (x : ℕ) : ℕ :=
    (sum_factorials x) % 10
  ((a ^ sum_mod_10 a) + (b ^ sum_mod_10 b) + (c ^ sum_mod_10 c)) % 10

theorem unit_digit_is_3 : unit_digit_of_power_sum 63 18 37 = 3 := by
  sorry

end unit_digit_is_3_l722_722490


namespace length_of_PS_l722_722232

variables {P Q R S : Type} -- representing points in trapezoid.
variables {PS RQ PQ SR : ℝ} -- representing the lengths of the segments.
variables (h : ℝ) -- representing the height of the trapezoid

-- Defining the area of the triangles using base and height.
def area_PQR (PQ h : ℝ) := 1 / 2 * PQ * h
def area_PSR (SR h : ℝ) := 1 / 2 * SR * h

-- Given Conditions as Lean Definitions
def condition1 : PS + RQ = 270 := sorry
def condition2 : area_PQR PQ h / area_PSR SR h = 5 / 4 := sorry

-- The statement to prove.
theorem length_of_PS (h: ℝ)  (PQ PS RQ SR : ℝ ) (condition1 : PS + RQ = 270) (condition2 : area_PQR PQ h / area_PSR SR h = 5 / 4) : 
  PS = 150 := sorry

end length_of_PS_l722_722232


namespace find_a_l722_722534

theorem find_a : 
  let θ := real.arcsin (2 * (real.sqrt 3) * real.sin (13 * real.pi / 12) * real.cos (real.pi / 12)) in
  let x := 2 * (real.sin (real.pi / 8))^2 - 1 in
  ∃ a, (x, a) lies_on_the_terminal_side_of θ → a = - (real.sqrt 6) / 2 :=
begin
  sorry
end

end find_a_l722_722534


namespace limit_an_bn_sqrt3_l722_722507

theorem limit_an_bn_sqrt3 (a_n b_n : ℕ → ℕ) (h : ∀ n, (1 + Real.sqrt 3) ^ n = a_n n + b_n n * Real.sqrt 3) : 
  Real.limit (λ n, (a_n n) / (b_n n)) (Real.sqrt 3) := 
sorry

end limit_an_bn_sqrt3_l722_722507


namespace part1_part2_l722_722000

open Nat

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 / 3 ∧
  a 2 = 2 ∧
  ∀ n > 2, 3 * (a (n + 1) - 2 * a n + a (n - 1)) = 2

theorem part1 (a : ℕ → ℚ)
  (h : sequence a) :
  ∃ d, ∀ n > 0, a (n + 1) - a n = d :=
sorry

theorem part2 (a : ℕ → ℚ)
  (h : sequence a) :
  ∃ n, frac_sum a n (∑ k in range n, (1 / a k)) > 5 / 2 ∧ n = 6 :=
sorry

end part1_part2_l722_722000


namespace find_black_balls_l722_722460

-- Define the conditions given in the problem.
def initial_balls : ℕ := 10
def all_red_balls (p_red : ℝ) : Prop := p_red = 1
def equal_red_black (p_red : ℝ) (p_black : ℝ) : Prop := p_red = 0.5 ∧ p_black = 0.5
def with_green_balls (p_red : ℝ) (green_balls : ℕ) : Prop := green_balls = 2 ∧ p_red = 0.7

-- Define the total probability condition
def total_probability (p_red : ℝ) (p_green : ℝ) (p_black : ℝ) : Prop :=
  p_red + p_green + p_black = 1

-- The final statement to prove
theorem find_black_balls :
  ∃ black_balls : ℕ,
    initial_balls = 10 ∧
    (∃ p_red : ℝ, all_red_balls p_red) ∧
    (∃ p_red p_black : ℝ, equal_red_black p_red p_black) ∧
    (∃ p_red : ℝ, ∃ green_balls : ℕ, with_green_balls p_red green_balls) ∧
    (∃ p_red p_green p_black : ℝ, total_probability p_red p_green p_black) ∧
    black_balls = 1 :=
sorry

end find_black_balls_l722_722460


namespace figure_with_conditions_is_square_l722_722773

def is_parallelogram (P : Type) [plane_geom : PlaneGeometry P] (f : figure P) : Prop :=
  is_convex_quadrilateral f ∧ opposite_sides_equal f

def all_sides_equal (f : figure) : Prop :=
  ∀ (side1 side2 : segment), side1 ∈ sides f → side2 ∈ sides f → length side1 = length side2

def all_angles_equal (f : figure) : Prop :=
  ∀ (angle1 angle2 : angle), angle1 ∈ angles f → angle2 ∈ angles f → measure angle1 = measure angle2

def is_square (f : figure) : Prop :=
  is_parallelogram f ∧ all_sides_equal f ∧ all_angles_equal f

theorem figure_with_conditions_is_square (f : figure) 
  (h1 : is_parallelogram f) 
  (h2 : all_sides_equal f) 
  (h3 : all_angles_equal f) : is_square f :=
begin
  sorry
end

end figure_with_conditions_is_square_l722_722773


namespace value_of_six_inch_cube_l722_722769

theorem value_of_six_inch_cube (value_of_four_inch_cube : ℝ) (volume_ratio : ℝ) :
  value_of_four_inch_cube = 800 ∧ volume_ratio = (6^3) / (4^3) → value_of_six_inch_cube = 2700 :=
by {
  sorry
}

end value_of_six_inch_cube_l722_722769


namespace equation_line_AB_equation_line_perpendicular_passing_C_l722_722933

-- Definition of points
def Point (x y : ℝ) := (x, y)
def A : Point := (2, -2)
def B : Point := (4, 6)
def C : Point := (-2, 0)

-- Equations of lines to prove
theorem equation_line_AB : ∃ (a b c : ℝ), a = 4 ∧ b = -1 ∧ c = -10 ∧ ∀ x y, y = (a * x + c) / b ↔ 4 * x - y - 10 = 0 := 
sorry

theorem equation_line_perpendicular_passing_C : ∃ (a b c : ℝ), a = 1 ∧ b = 4 ∧ c = 2 ∧ ∀ x y, y = (a * x + c) / b ↔ x + 4 * y + 2 = 0 := 
sorry

end equation_line_AB_equation_line_perpendicular_passing_C_l722_722933


namespace right_triangle_short_leg_l722_722582

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l722_722582


namespace no_square_in_sequence_l722_722630

def sequence (n : ℕ) : ℚ 
| 0       := 2016
| (n + 1) := sequence n + 2 / sequence n

theorem no_square_in_sequence : ∀ n : ℕ, ∀ q : ℚ, q^2 ≠ sequence n := by
  sorry

end no_square_in_sequence_l722_722630


namespace mountain_number_total_l722_722351

noncomputable def mountain_number_count : ℕ :=
  let total_case1 := (Finset.range 9).sum (λ y, Nat.choose (y - 1) 2)
  let total_case2 := (Finset.range 8).sum (λ x, 9 - x)
  let total_case3 := (Finset.range 6).sum (λ y, Nat.choose (y + 3) 3)
  total_case1 + total_case2 + total_case3

theorem mountain_number_total : mountain_number_count = 190 := by
  sorry

end mountain_number_total_l722_722351


namespace number_of_distinct_prime_factors_30_fact_l722_722987

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722987


namespace part1_part2_min_value_l722_722269

theorem part1 (A B C : ℝ) (a b c : ℝ) (h1 : C = 2 * Real.pi / 3)
(h2 : Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : 
B = Real.pi / 6 :=
sorry

theorem part2_min_value (A B C : ℝ) (a b c : ℝ) :
(∀ (h : a = c * Real.sin A / Real.sin C) (h : b = c * Real.sin B / Real.sin C) (h :∠A + ∠B + ∠C = Real.pi), 
(∃ δ > 0, ∀ x y : ℝ, (x - a)^2 + (y - b)^2 < δ → ∃ t, x = a^2 + b^2 / c^2 ≥ 4 * Real.sqrt 2 - 5)) :=
sorry

end part1_part2_min_value_l722_722269


namespace common_ratio_geometric_series_l722_722893

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l722_722893


namespace option_b_option_c_l722_722005

variable {x y θ : ℝ} (h1 : z1 = x + y * Complex.I) (h2 : z2 = Complex.cos θ + Complex.sin θ * Complex.I)

theorem option_b (h1 : z1 = x + y * Complex.I) (h2 : z2 = Complex.cos θ + Complex.sin θ * Complex.I) :
  ∀ (z2 : ℂ), (∃ θ : ℝ, z2 = Complex.cos θ + Complex.sin θ * Complex.I) → (∃ z : ℂ, z = z2^2 ∧ Complex.abs z = 1) :=
sorry

theorem option_c (h1 : z1 = x + y * Complex.I) (h2 : z2 = Complex.cos θ + Complex.sin θ * Complex.I) :
  Complex.conj (z1 + z2) = Complex.conj z1 + Complex.conj z2 :=
sorry

end option_b_option_c_l722_722005


namespace smallest_positive_period_of_f_monotonically_increasing_intervals_of_f_max_value_of_f_on_interval_l722_722540

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 6)

-- Part 1: Smallest positive period
theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
sorry

-- Part 2: Monotonically increasing intervals
theorem monotonically_increasing_intervals_of_f :
  ∃ (k : ℤ), ∀ x, k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 3 →
  (f' x > 0) ∧ k ∈ Set.univ :=
sorry

-- Part 3: Maximum value on [0, π/2]
theorem max_value_of_f_on_interval :
  ∃ y ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f y = 2 :=
sorry

end smallest_positive_period_of_f_monotonically_increasing_intervals_of_f_max_value_of_f_on_interval_l722_722540


namespace find_m_range_l722_722541

variable (a m : ℝ)
variable (f : ℝ → ℝ)

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem find_m_range (h_a : a ∈ set.Icc (-3) 0) 
  (h_ineq : ∀ (x1 x2 : ℝ), x1 ∈ set.Icc (0 : ℝ) 2 → x2 ∈ set.Icc (0 : ℝ) 2 → m - a * m^2 ≥ abs (f x1 - f x2)) 
  : m ∈ set.Ici (5 : ℝ) := 
sorry

end find_m_range_l722_722541


namespace max_sum_of_arithmetic_sequence_l722_722928

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (S_seq : ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)) 
  (S16_pos : S 16 > 0) (S17_neg : S 17 < 0) : 
  ∃ m, ∀ n, S n ≤ S m ∧ m = 8 := 
sorry

end max_sum_of_arithmetic_sequence_l722_722928


namespace find_perimeter_of_triangle_ABC_l722_722597

open EuclideanGeometry

noncomputable def perimeter_triangle_ABC
  (P Q R S A B C : Point)
  (HPQ : distance P Q = 4)
  (HQR : distance Q R = 4)
  (HRS : distance R S = 4)
  (HAS: distance A S = 4)
  (HAB : is_tangent_circle P 2 A B)
  (HBC : is_tangent_circle Q 2 B C)
  (HCA : is_tangent_circle R 2 C A) : ℝ :=
distance A B + distance B C + distance C A

theorem find_perimeter_of_triangle_ABC
  (P Q R S A B C : Point)
  (HPQ : distance P Q = 4)
  (HQR : distance Q R = 4)
  (HRS : distance R S = 4)
  (HAS: distance A S = 4)
  (HAB : is_tangent_circle P 2 A B)
  (HBC : is_tangent_circle Q 2 B C)
  (HCA : is_tangent_circle R 2 C A) :
  perimeter_triangle_ABC P Q R S A B C HPQ HQR HRS HAS HAB HBC HCA = 24 + 10 * Real.sqrt 3 :=
by
  sorry

end find_perimeter_of_triangle_ABC_l722_722597


namespace find_z2_l722_722944

open Complex

theorem find_z2
  (z1 z2 : ℂ)
  (h1 : (z1 - 2) * (1 + I) = 1 - I)
  (h2 : z2.im = 2)
  (h3 : isReal(z1 * z2)) :
  z2 = 4 + 2 * I := 
  sorry

end find_z2_l722_722944


namespace sqrt_recursive_value_l722_722841

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722841


namespace expression_evaluation_l722_722140

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l722_722140


namespace A_and_B_together_complete_work_in_6_days_l722_722378

-- Definitions for the problem
variable (A B : Type)
variable (complete_work : A → B → ℕ → Prop)
variable (work_done_alone : A → ℕ → Prop)
variable (days_to_complete : ℕ)

-- Conditions given in the problem
axiom work_completed_by_A_and_B_together : complete_work A B 6
axiom work_completed_by_A_alone : work_done_alone A 10

-- Question & Proof goal
theorem A_and_B_together_complete_work_in_6_days :
  complete_work A B 6 := sorry

end A_and_B_together_complete_work_in_6_days_l722_722378


namespace evaluate_nested_radical_l722_722859

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722859


namespace sum_of_divisors_of_form_2i_5j_eq_1000_l722_722333

-- Define the sum of positive divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, d

-- Define the specific positive integer form and the correct answer condition
theorem sum_of_divisors_of_form_2i_5j_eq_1000 (i j : ℕ) :
  sum_of_divisors (2^i * 5^j) = 1000 → i + j = 6 :=
by
  sorry

end sum_of_divisors_of_form_2i_5j_eq_1000_l722_722333


namespace volume_of_cube_in_pyramid_l722_722425

noncomputable def VolumeOfCubeInPyramid
    (base_side_length : ℝ)
    (lateral_face_height : ℝ)
    (cube_side_on_base: ℝ)
    (vertex_touch_midpoint : Bool) :
    ℝ :=
if base_side_length = 2
   ∧ lateral_face_height = 4
   ∧ cube_side_on_base = 2
   ∧ vertex_touch_midpoint = true
then 8
else 0

theorem volume_of_cube_in_pyramid : VolumeOfCubeInPyramid 2 4 2 true = 8 :=
by
  simp [VolumeOfCubeInPyramid]
  sorry

end volume_of_cube_in_pyramid_l722_722425


namespace number_of_prime_factors_30_factorial_l722_722970

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722970


namespace find_area_of_bounded_part_l722_722482

noncomputable def area_bounded_by_curves : ℝ :=
  ∫ x in 0..(7 - 3 * Real.sqrt 5) / 2, Real.sqrt x +
  ∫ x in (7 - 3 * Real.sqrt 5) / 2..1, (1 - 2 * Real.sqrt x + x)

theorem find_area_of_bounded_part :
  let a := (7 - 3 * Real.sqrt 5) / 2 in
  area_bounded_by_curves = (35 - 15 * Real.sqrt 5) / 12 := 
by
  sorry

end find_area_of_bounded_part_l722_722482


namespace positive_whole_numbers_with_cube_roots_less_than_15_l722_722118

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l722_722118


namespace percentage_increase_approx_l722_722692

-- Define the original salary S
variable (S : ℚ) -- using Rational numbers for S
-- Define the percentage increase P in percentage points
variable (P : ℚ) -- using Rational numbers for P

-- Express the conditions in Lean
def new_salary_after_increase := S + (P / 100) * S
def new_salary_after_decrease := new_salary_after_increase * 0.9
def final_salary := S * 1.01

-- Translate the proof problem into a Lean statement
theorem percentage_increase_approx (h : new_salary_after_decrease = final_salary) :
  P ≈ 12.22 := 
sorry

end percentage_increase_approx_l722_722692


namespace time_to_pay_back_l722_722614

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l722_722614


namespace subset_proof_l722_722266

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 1)}

-- The problem statement
theorem subset_proof : M ⊆ N ∧ ∃ y ∈ N, y ∉ M :=
by
  sorry

end subset_proof_l722_722266


namespace nested_sqrt_eq_l722_722874

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l722_722874


namespace find_m_times_t_l722_722253

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation (x y z : ℝ) : g(x^2 + y * g(z)) = x * g(x) + z * g(y)

theorem find_m_times_t : 
  let m := { g₃ | g₃ = g 3 }.to_finset.card in
  let t := { g3 | g3 = g 3 }.sum id in
  m * t = 6 :=
by
  sorry

end find_m_times_t_l722_722253


namespace non_real_roots_bounded_l722_722161

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722161


namespace solve_system_of_equations_l722_722303

theorem solve_system_of_equations {x y : ℝ} : 
  (x^2 - 5 * x * y + 6 * y^2 = 0) ∧ (x^2 + y^2 = 40) →
  (x = 4 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2) ∨
  (x = -4 * Real.sqrt 2 ∧ y = -2 * Real.sqrt 2) ∨
  (x = 6 ∧ y = 2) ∨
  (x = -6 ∧ y = -2) :=
by
  intro h,
  sorry

end solve_system_of_equations_l722_722303


namespace emily_art_supplies_l722_722831

theorem emily_art_supplies (total_spent skirts_cost skirt_quantity : ℕ) 
  (total_spent_eq : total_spent = 50) 
  (skirt_cost_eq : skirts_cost = 15) 
  (skirt_quantity_eq : skirt_quantity = 2) :
  total_spent - skirt_quantity * skirts_cost = 20 :=
by
  sorry

end emily_art_supplies_l722_722831


namespace c_geq_one_l722_722633

variable {α : Type*} [LinearOrderedField α]

theorem c_geq_one
  (a : ℕ → α)
  (c : α)
  (h1 : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, 0 < i → 0 < j → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 :=
sorry

end c_geq_one_l722_722633


namespace abs_diff_ge_2sqrt2_l722_722523

theorem abs_diff_ge_2sqrt2 (n : ℕ) (h : 0 < n) (a b : ℕ) (h_ab : a * b = n^2 + n + 1) : abs (a - b) ≥ 2 * Real.sqrt n := 
sorry

end abs_diff_ge_2sqrt2_l722_722523


namespace vector_magnitude_l722_722966

variable (x : ℝ)
def a := (1 : ℝ, -2 : ℝ)
def b := (x, 4)

theorem vector_magnitude :
  (1 / -2) = (x / 4) → ‖((1 + x), (-2 + 4))‖ = sqrt 5 :=
by
  sorry

end vector_magnitude_l722_722966


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722104

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722104


namespace average_of_k_l722_722076

theorem average_of_k (k r1 r2: ℕ) (h1 : r1 * r2 = 24) (h2 : r1 + r2 = k) (h3 : r1 > 0) (h4 : r2 > 0) : 
  let ks := [25, 14, 11, 10] in
  (ks.sum / ks.length : ℝ) = 15 := by 
sorry

end average_of_k_l722_722076


namespace sum_of_segments_eq_one_l722_722513

set_option maxHeartbeats 1000000

/-- Given a triangle ABC, construct two lines x and y such that for any point M on side AC,
    the sum of the lengths of the segments MX_M (parallel to x until intersecting AB) and
    MY_M (parallel to y until intersecting BC) is equal to 1. -/
theorem sum_of_segments_eq_one (A B C : Type) [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] :
  ∀ (M : A), M ∈ AC → let X_M := intersect M x AB,
                      let Y_M := intersect M y BC in
  dist M X_M + dist M Y_M = 1 :=
begin
  -- Variables "x", "y", "AC", "AB", "BC", "dist", and "intersect" should be defined properly 
  -- prior to use in the theorem or inferred appropriately by the context. Also, the AC, AB, BC
  -- are the sides of the triangle ABC in the problem definition.
  sorry
end

end sum_of_segments_eq_one_l722_722513


namespace solve_for_x_l722_722660

theorem solve_for_x (x y : ℕ) (h₁ : 9 ^ y = 3 ^ x) (h₂ : y = 6) : x = 12 :=
by
  sorry

end solve_for_x_l722_722660


namespace solve_log_eq_l722_722480

theorem solve_log_eq :
  (∃ y : ℝ, log y 16 = log 64 4) ↔ ∃ y : ℝ, y = 4096 :=
by
  -- Define the log identity
  have h1 : log 64 4 = 1 / 3 := by sorry
  -- Convert log equation to exponential equation
  have h2 : ∀ y : ℝ, log y 16 = 1 / 3 → (∃ y : ℝ, y = 16^3) := by sorry
  
  -- Combine the two parts to show equivalence
  split
  -- Forward direction
  intro h
  cases h with y hy
  rw [←h1, hy] at hy
  exact ⟨y, h2 y hy⟩
  
  -- Backward direction
  intro h
  cases h with y hy
  use y
  rw [h1, hy]

end solve_log_eq_l722_722480


namespace stratified_sampling_l722_722766

theorem stratified_sampling (
  (total_employees business_personnel management_personnel logistics_personnel sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : business_personnel = 120)
  (h3 : management_personnel = 16)
  (h4 : logistics_personnel = 24)
  (h5 : sample_size = 20)) :
  (nat.floor (business_personnel * sample_size / total_employees)) = 15 ∧ 
  (nat.floor (management_personnel * sample_size / total_employees)) = 2 ∧ 
  (nat.floor (logistics_personnel * sample_size / total_employees)) = 3 := 
  sorry

end stratified_sampling_l722_722766


namespace ellipse_standard_equation_l722_722798

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ c = 2 ∧ (3^2 / a^2 + (-2 * real.sqrt 6)^2 / b^2 = 1) ∧ (a^2 = 36 ∧ b^2 = 32) ∧ (∀ x y, x^2 / 36 + y^2 / 32 = 1)
:= sorry

end ellipse_standard_equation_l722_722798


namespace value_of_2a_plus_b_l722_722141

theorem value_of_2a_plus_b (a b : ℤ) (h1 : |a - 1| = 4) (h2 : |b| = 7) (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := sorry

end value_of_2a_plus_b_l722_722141


namespace train_length_l722_722711

noncomputable def length_of_train 
  (speed1 : ℝ)        -- Speed of the first train in km/hr
  (speed2 : ℝ)        -- Speed of the second train in km/hr
  (pass_time : ℝ)     -- Time taken for the slower train to pass the driver of the faster one in seconds
  : ℝ :=
let relative_speed := (speed1 + speed2) / 3600 in
let distance := relative_speed * pass_time in
(distance / 2) * 1000 -- Converting km to meters
  
theorem train_length
  (h₀ : speed1 = 55)      -- Speed of the first train is 55 km/hr
  (h₁ : speed2 = 40)      -- Speed of the second train is 40 km/hr
  (h₂ : pass_time = 36)   -- The time to pass is 36 seconds
  : length_of_train 55 40 36 = 475 := 
by 
  unfold length_of_train
  rw [h₀, h₁, h₂]
  simp
  norm_num
  sorry

end train_length_l722_722711


namespace rhombus_side_length_l722_722889

variable (s d1 d2 : ℝ)
variable (area : ℝ)

def is_rhombus (s d1 d2 : ℝ) : Prop :=
  d1 = 30 ∧ area = 600 ∧ s^2 = (d1/2)^2 + (d2/2)^2 ∧ area = (d1 * d2) / 2

theorem rhombus_side_length (h : is_rhombus s d1 d2) : s = 25 :=
by
  sorry

end rhombus_side_length_l722_722889


namespace marvin_took_six_bottle_caps_l722_722338

variable (initial_bottle_caps total_left_bottle_caps : ℕ)
variable (number_taken : ℕ)

-- defining the conditions
def initial_condition : Prop := initial_bottle_caps = 16
def take_condition : Prop := total_left_bottle_caps = 10
def number_taken_condition : Prop := number_taken = initial_bottle_caps - total_left_bottle_caps

-- stating the theorem
theorem marvin_took_six_bottle_caps 
    (h1: initial_condition) 
    (h2: take_condition) 
    (h3: number_taken_condition) : 
    number_taken = 6 := 
by sorry

end marvin_took_six_bottle_caps_l722_722338


namespace sum_log_product_l722_722355

theorem sum_log_product :
  (∑ k in Finset.range 15, Real.logb (7^k) (4^(k^2))) *
  (∑ k in Finset.range 50, Real.logb (16^k) (49^k)) = 12000 := by
  sorry

end sum_log_product_l722_722355


namespace door_height_is_eight_l722_722211

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l722_722211


namespace first_day_of_month_l722_722671

-- Define the days of the week as an enumeration
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Given condition
def eighteenth_day_is : Day := Wednesday

-- Theorem to prove
theorem first_day_of_month : 
  (eighteenth_day_is = Wednesday) → (eighteenth_day_is = Wednesday) := 
by
  intros h,
  -- We will need to establish the reverse order from 18th day is Wednesday back to 1st day is Sunday
  sorry

end first_day_of_month_l722_722671


namespace sum_of_arithmetic_sequence_l722_722268

theorem sum_of_arithmetic_sequence (d : ℚ) (h_d : d = 1 / 4) (a : ℕ → ℚ)
  (h_a1 : a 1 = 1)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geo : (a 1) * (a 6) = (a 3) ^ 2) :
  ∀ n, (∑ k in finset.range n, a k) = (n^2 / 8) + (7 * n / 8) :=
begin
  sorry
end

end sum_of_arithmetic_sequence_l722_722268


namespace num_pos_nums_with_cube_root_lt_15_l722_722128

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l722_722128


namespace probability_intersection_l722_722754

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l722_722754


namespace box_volume_l722_722363

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l722_722363


namespace distance_midpoint_B_to_directrix_l722_722547

theorem distance_midpoint_B_to_directrix
  (p : ℝ) (h : p > 0) (B : ℝ × ℝ) (hB : B = (p / 4, 1))
  (hB_on_parabola : 1^2 = 2 * p * (p / 4)) :
  ∃ d : ℝ, d = (3 * real.sqrt 2) / 4 :=
sorry

end distance_midpoint_B_to_directrix_l722_722547


namespace line_equation_l722_722515

noncomputable def arithmetic_sequence (n : ℕ) (a_1 d : ℝ) : ℝ :=
  a_1 + (n - 1) * d

theorem line_equation
  (a_2 a_4 a_5 : ℝ)
  (a_2_cond : a_2 = arithmetic_sequence 2 a_1 d)
  (a_4_cond : a_4 = arithmetic_sequence 4 a_1 d)
  (a_5_cond : a_5 = arithmetic_sequence 5 a_1 d)
  (sum_cond : a_2 + a_4 = 12)
  (a_5_val : a_5 = 10)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * y = 0 ↔ (x - 0)^2 + (y - 1)^2 = 1)
  : ∃ (line : ℝ → ℝ → Prop), line x y ↔ (6 * x - y + 1 = 0) :=
by
  sorry

end line_equation_l722_722515


namespace ratio_of_borders_l722_722200

theorem ratio_of_borders (s d : ℝ) (n : ℕ) (h_n : n = 30) (h_cover : (n : ℝ)^2 * s^2 / (n * s + 2 * n * d)^2 = 0.81) : d / s = 1 / 18 := 
by
  -- Given n is fixed as 30 in the conditions
  have h_n_real : (n : ℝ) = 30 := by
    exact (congr_arg coe h_n)
  -- sorry used to skip the proof steps
  sorry

end ratio_of_borders_l722_722200


namespace problem_statement_l722_722561

theorem problem_statement :
  let a : ℕ → ℤ := λ n, ite (n = 0) 1 (binom 2011 n * (-2) ^ n) in
  (a 1 / 2 + a 2 / 2^2 + a 3 / 2^3 + ... + a 2011 / 2^2011) = -1 :=
sorry

end problem_statement_l722_722561


namespace distance_from_edge_to_bottom_l722_722641

theorem distance_from_edge_to_bottom (d x : ℕ) 
  (h1 : 63 + d + 20 = 10 + d + x) : x = 73 := by
  -- This is where the proof would go
  sorry

end distance_from_edge_to_bottom_l722_722641


namespace otimes_calc_1_otimes_calc_2_otimes_calc_3_l722_722821

def otimes (a b : Int) : Int :=
  a^2 - Int.natAbs b

theorem otimes_calc_1 : otimes (-2) 3 = 1 :=
by
  sorry

theorem otimes_calc_2 : otimes 5 (-4) = 21 :=
by
  sorry

theorem otimes_calc_3 : otimes (-3) (-1) = 8 :=
by
  sorry

end otimes_calc_1_otimes_calc_2_otimes_calc_3_l722_722821


namespace number_of_cube_roots_lt_15_l722_722115

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l722_722115


namespace product_of_bases_in_decimal_l722_722477

theorem product_of_bases_in_decimal :
  let binary_to_decimal := 2^3 + 2 + 1
  let ternary_to_decimal := 2 * 3^2 + 3 + 2
  binary_to_decimal * ternary_to_decimal = 253 :=
by
  -- Definitions for binary and ternary values.
  let binary_to_decimal := 8 + 2 + 1  -- This is 11 in decimal
  let ternary_to_decimal := 18 + 3 + 2  -- This is 23 in decimal
  -- Desired equality
  have h : binary_to_decimal * ternary_to_decimal = 253
  -- Proof (to be filled in Lean)
  sorry

end product_of_bases_in_decimal_l722_722477


namespace number_of_prime_factors_30_factorial_l722_722984

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722984


namespace Juan_reads_pages_per_hour_l722_722243

theorem Juan_reads_pages_per_hour {S : ℝ} (S_pos : 0 < S) :
  let T := 4 in
  let pages_read_first_book := T * S in
  let pages_read_second_book := T * (0.75 * S) in
  let total_pages := pages_read_first_book + pages_read_second_book in
  let pages_per_hour := total_pages / T in
  pages_per_hour = 1.75 * S :=
by
  sorry

end Juan_reads_pages_per_hour_l722_722243


namespace soccer_team_starters_l722_722280

theorem soccer_team_starters (players : Finset ℕ) (quadruplets : Finset ℕ) :
  players.card = 16 ∧ quadruplets.card = 4 ∧ quadruplets ⊆ players →
  let starters := Finset.filter (λ p, p ∈ quadruplets) players in
  starters.card = 2 → 
  let non_quadruplets := players \ quadruplets in
  ∃ chosen_starters : Finset ℕ,
    chosen_starters.card = 6 ∧
    starters ⊆ chosen_starters ∧
    ∃ combination_count : ℕ,
      combination_count = ((quadruplets.card.choose 2) * (non_quadruplets.card.choose 4)) ∧
      combination_count = 2970 := 
sorry

end soccer_team_starters_l722_722280


namespace length_generatrix_cone_l722_722063

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722063


namespace solve_for_F_l722_722537

theorem solve_for_F (F C : ℝ) (h₁ : C = 4 / 7 * (F - 40)) (h₂ : C = 25) : F = 83.75 :=
sorry

end solve_for_F_l722_722537


namespace total_swordfish_caught_l722_722288

theorem total_swordfish_caught 
  (Shelly_catches_per_trip : ℕ)
  (Sam_catches_per_trip : ℕ)
  (total_catches_per_trip : ℕ)
  (total_trips : ℕ)
  (h1 : Shelly_catches_per_trip = 5 - 2)
  (h2 : Sam_catches_per_trip = Shelly_catches_per_trip - 1)
  (h3 : total_catches_per_trip = Shelly_catches_per_trip + Sam_catches_per_trip)
  (h4 : total_trips = 5)
  : (total_catches_per_trip * total_trips) = 25 :=
by simp [h1, h2, h3, h4]; exact sorry

end total_swordfish_caught_l722_722288


namespace tangent_line_eq_f_positive_find_a_l722_722085

noncomputable def f (x a : ℝ) : ℝ := 1 - (a * x^2) / (Real.exp x)
noncomputable def f' (x a : ℝ) : ℝ := (a * x * (x - 2)) / (Real.exp x)

-- Part 1: equation of tangent line
theorem tangent_line_eq (a : ℝ) (h1 : f' 1 a = 1) (hx : f 1 a = 2) : ∀ x, f 1 a + f' 1 a * (x - 1) = x + 1 :=
sorry

-- Part 2: f(x) > 0 for x > 0 when a = 1
theorem f_positive (x : ℝ) (h : x > 0) : f x 1 > 0 :=
sorry

-- Part 3: minimum value of f(x) is -3, find a
theorem find_a (a : ℝ) (h : ∀ x, f x a ≥ -3) : a = Real.exp 2 :=
sorry

end tangent_line_eq_f_positive_find_a_l722_722085


namespace exists_quadrilateral_containing_points_l722_722233

variable (A B C D E M N : Point)
variable (ABCDE : ConvexPentagon A B C D E)
variable (hM : InsidePentagon M ABCDE)
variable (hN : InsidePentagon N ABCDE)

theorem exists_quadrilateral_containing_points :
  ∃ P Q R S ∈ {A, B, C, D, E}, InsideQuadrilateral M P Q R S ∧ InsideQuadrilateral N P Q R S :=
by
  sorry

end exists_quadrilateral_containing_points_l722_722233


namespace volume_of_right_pyramid_l722_722786

-- Definition of a right pyramid with a regular hexagonal base
structure RightPyramid :=
  (base_area : ℝ)
  (tri_face_area : ℝ)
  (total_surface_area : ℝ)
  (volume : ℝ)

-- Given conditions
def hexagonal_pyramid_conditions := 
  { base_area := 324, 
    tri_face_area := 108, 
    total_surface_area := 648, 
    volume := 1728 * Real.sqrt 6 }

-- The main theorem statement
theorem volume_of_right_pyramid (P : RightPyramid)
  (h_base : P.base_area = 324)
  (h_tri : P.tri_face_area = 108)
  (h_surface : P.total_surface_area = 648) :
  P.volume = 1728 * Real.sqrt 6 :=
  sorry

end volume_of_right_pyramid_l722_722786


namespace find_nonneg_ints_l722_722481

/-- Find non-negative integers a, b, c, and d such that 5a + 6b + 7c + 11d = 1999. -/
theorem find_nonneg_ints (a b c d : ℕ) (h : 5 * a + 6 * b + 7 * c + 11 * d = 1999) :
  a = 389 ∧ b = 2 ∧ c = 1 ∧ d = 3 :=
begin
  sorry
end

end find_nonneg_ints_l722_722481


namespace generatrix_length_of_cone_l722_722026

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722026


namespace part_a_AK_eq_DC_l722_722382

-- Definitions corresponding to the conditions
variable (A B C D M K : Point)
variable (incircle : Circle)
variable (triangle : Triangle)
variable (DM_diameter : Circle) 
variable [Incircle incircle triangle]
variable [Touches incircle AC D]
variable [Diameter DM_diameter D M]
variable [IntersectsLine BM AC K]

-- Theorem statement
theorem part_a_AK_eq_DC (AK DC : Segment) : AK = DC :=
  sorry

end part_a_AK_eq_DC_l722_722382


namespace find_equation_of_line_l722_722902

-- Define the conditions
def line_passes_through_A (m b : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (1, 1) ∧ A.2 = -A.1 + b

def intercepts_equal (m b : ℝ) : Prop :=
  b = m

-- The goal to prove the equations of the line
theorem find_equation_of_line :
  ∃ (m b : ℝ), line_passes_through_A m b (1, 1) ∧ intercepts_equal m b ↔ 
  (∃ m b : ℝ, (m = -1 ∧ b = 2) ∨ (m = 1 ∧ b = 0)) :=
sorry

end find_equation_of_line_l722_722902


namespace average_and_variance_conditions_l722_722195

theorem average_and_variance_conditions
  (n : ℕ)
  (x : ℕ → ℕ)
  (m s : ℕ)
  (h1 : (∑ i in range n, (x i + m)) / n = 5)
  (h2 : (1 / n) * (∑ i in range n, ((x i + m - 5) ^ 2)) = 4)
  (h3 : (∑ i in range n, (3 * x i + 1)) / n = 10) :
  m = 2 ∧ s = 6 := by
  sorry

end average_and_variance_conditions_l722_722195


namespace find_abscissa_of_tangent_l722_722265

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

noncomputable def func (a x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

def slope_of_tangent (x : ℝ) := Real.exp x - Real.exp (-x)

theorem find_abscissa_of_tangent {a : ℝ} (h1 : is_odd_function (λ x => (Real.exp x - a * Real.exp (-x))))
  (h2 : slope_of_tangent x = 3 / 2) : x = Real.log 2 :=
sorry

end find_abscissa_of_tangent_l722_722265


namespace quadratic_non_real_roots_l722_722166

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l722_722166


namespace value_of_fraction_of_power_l722_722700

-- Define the values in the problem
def a : ℝ := 6
def b : ℝ := 30

-- The problem asks us to prove
theorem value_of_fraction_of_power : 
  (1 / 3) * (a ^ b) = 2 * (a ^ (b - 1)) :=
by
  -- Initial Setup
  let c := (1 / 3) * (a ^ b)
  let d := 2 * (a ^ (b - 1))
  -- The main claim
  show c = d
  sorry

end value_of_fraction_of_power_l722_722700


namespace proof_5x_plus_4_l722_722559

variable (x : ℝ)

-- Given condition
def condition := 5 * x - 8 = 15 * x + 12

-- Required proof
theorem proof_5x_plus_4 (h : condition x) : 5 * (x + 4) = 10 :=
by {
  sorry
}

end proof_5x_plus_4_l722_722559


namespace sqrt_recursive_value_l722_722844

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722844


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722106

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l722_722106


namespace simplify_expression_l722_722291

theorem simplify_expression :
  (256:ℝ)^(1/4) * (125:ℝ)^(1/2) = 20 * Real.sqrt 5 := 
by {
  have h1 : (256:ℝ) = 2^8 := by norm_num,
  have h2 : (125:ℝ) = 5^3 := by norm_num,
  rw [h1, h2], 
  simp,
  have h3 : (2 : ℝ) ^ 8 ^ (1 / 4) = 4 := by norm_num,
  have h4 : (5 : ℝ) ^ 3 ^ (1 / 2) = 5 * Real.sqrt 5 := by norm_num,
  rw [h3, h4],
  norm_num,
}

end simplify_expression_l722_722291


namespace range_of_x_l722_722526

variable (f : ℝ → ℝ) (h_inc : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f(x) ≤ f(y))

theorem range_of_x (x : ℝ) :
  (f (2 * x - 1) < f (1 / 3)) ↔ (1 / 2 ≤ x ∧ x < 2 / 3) :=
by
  sorry

end range_of_x_l722_722526


namespace transportation_time_savings_l722_722236

-- Define the data given in the problem
def time_to_walk : ℕ := 98

def time_saved_by_bicycle : ℕ := 64
def time_saved_by_car : ℕ := 85
def time_saved_by_bus : ℕ := 55

-- Define the correct answers
def expected_saving_by_bicycle := time_saved_by_bicycle
def expected_saving_by_car := time_saved_by_car
def expected_saving_by_bus := time_saved_by_bus

-- Define the theorem we need to prove
theorem transportation_time_savings :
  (time_saved_by_bicycle = 64) ∧
  (time_saved_by_car = 85) ∧
  (time_saved_by_bus = 55) :=
by
  split;
  {refl}

end transportation_time_savings_l722_722236


namespace circum_center_incenter_distance_eq_l722_722069

open EuclideanGeometry

variables {A B C O I D E F P : Point}

/-- Circumcenter and incenter definitions --/
axiom circum_center_exists (A B C : Point) : Point := O
axiom in_center_exists (A B C : Point) : Point := I

/-- Points D, E, F on sides BC, CA, AB respectively --/
axiom D_on_BC (D B C : Point) : Collinear B D C
axiom E_on_CA (E C A : Point) : Collinear C E A
axiom F_on_AB (F A B : Point) : Collinear A F B

/-- Conditions BD + BF = CA and CD + CE = AB --/
axiom BD_plus_BF_eq_CA : dist B D + dist B F = dist C A
axiom CD_plus_CE_eq_AB : dist C D + dist C E = dist A B

/-- The circumcircles of triangles BFD and CDE intersect at P --/
axiom circum1_intersects_circum2_at_P_diff_D : ∃ Γ1 Γ2 : Circle, (Circle₀ Γ1 A B (on_circle_₀) D (on_circle_D) F (on_circle_F)) ∧ (Circle₀ Γ2 C D (on_circle_D) E (on_circle_E)) ∧ Γ1 ∩ Γ2 = {D, P}
axiom P_neq_D : P ≠ D

/-- Prove OP = OI --/
theorem circum_center_incenter_distance_eq (A B C O I D E F P : Point) 
  (circum_center_exists : circum_center_exists A B C = O)
  (in_center_exists : in_center_exists A B C = I)
  (D_on_BC : D_on_BC D B C)
  (E_on_CA : E_on_CA E C A)
  (F_on_AB : F_on_AB F A B)
  (BD_plus_BF_eq_CA : BD_plus_BF_eq_CA)
  (CD_plus_CE_eq_AB : CD_plus_CE_eq_AB)
  (circum1_intersects_circum2_at_P_diff_D : circum1_intersects_circum2_at_P_diff_D)
  (P_neq_D : P_neq_D) : dist O P = dist O I := 
begin
  sorry
end

end circum_center_incenter_distance_eq_l722_722069


namespace solve_log_eq_l722_722295

theorem solve_log_eq (x : ℝ) (h : log 8 x + log 4 (x ^ 3) = 9) : 
  x = 2^(54 / 11) :=
by 
  sorry

end solve_log_eq_l722_722295


namespace range_of_a_l722_722940

theorem range_of_a (a : ℝ) (h1 : ∃ x : ℝ, x > 0 ∧ |x| = a * x - a) (h2 : ∀ x : ℝ, x < 0 → |x| ≠ a * x - a) : a > 1 :=
sorry

end range_of_a_l722_722940


namespace non_real_roots_bounded_l722_722160

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l722_722160


namespace kim_weekly_cost_l722_722245

noncomputable def watts_to_kwh (watts: ℕ) (hours: ℕ): ℝ :=
  (watts * hours) / 1000.0

noncomputable def weekly_energy_kwh(tv_hrs: ℕ, fridge_hrs: ℕ, ac_hrs: ℕ, washer_hrs: ℕ, washer_days: ℕ, laptop_hrs: ℕ): ℝ :=
  let tv_energy := watts_to_kwh 125 tv_hrs * 7
  let fridge_energy := watts_to_kwh 100 fridge_hrs * 7
  let ac_energy := watts_to_kwh 2000 ac_hrs * 7
  let washer_energy := watts_to_kwh 500 washer_hrs * washer_days
  let laptop_energy := watts_to_kwh 60 laptop_hrs * 7
  tv_energy + fridge_energy + ac_energy + washer_energy + laptop_energy

noncomputable def total_cost (energy_kwh: ℝ) (cost_per_kwh: ℝ): ℝ :=
  energy_kwh * cost_per_kwh

theorem kim_weekly_cost :
  let tv_hours_per_day := 4
  let fridge_hours_per_day := 24
  let ac_hours_per_day := 6
  let washer_hours_per_day := 2
  let washer_days_per_week := 3
  let laptop_hours_per_day := 5
  let weekly_energy := weekly_energy_kwh tv_hours_per_day fridge_hours_per_day ac_hours_per_day washer_hours_per_day washer_days_per_week laptop_hours_per_day
  let cost_per_kwh := 0.14
  total_cost weekly_energy cost_per_kwh = 15.316 :=
by
  sorry

end kim_weekly_cost_l722_722245


namespace regular_tetrahedron_proof_l722_722499

open EuclideanGeometry

/-- Define a Tetrahedron structure that will hold the points. -/
structure Tetrahedron :=
(A B C D : Point ℝ)

/-- Condition 1: Vertices C and D have perpendiculars to opposite faces with feet being incenters. -/
def condition1 (T : Tetrahedron) : Prop :=
  let {A, B, C, D} := T in
  is_incenter (project_onto_face A B C D) ∧ is_incenter (project_onto_face D A B C)

/-- Condition 2: Edge AB = BD. -/
def condition2 (T : Tetrahedron) (E : ℝ) : Prop :=
  let {A, B, _, D} := T in
  dist A B = E ∧ dist B D = E

/-- Theorem stating that tetrahedron ABCD is a regular tetrahedron given the conditions. -/
theorem regular_tetrahedron_proof (T : Tetrahedron) (E : ℝ)
  (h1 : condition1 T) (h2 : condition2 T E) : is_regular_tetrahedron T :=
sorry

end regular_tetrahedron_proof_l722_722499


namespace tan_alpha_eq_neg_half_simplifies_l722_722502

theorem tan_alpha_eq_neg_half_simplifies :
  ∀ α : ℝ, tan α = -1 / 2 → (2 * sin α * cos α) / (sin α ^ 2 - cos α ^ 2) = 4 / 3 :=
by
  intro α h
  sorry

end tan_alpha_eq_neg_half_simplifies_l722_722502


namespace number_of_distinct_prime_factors_30_fact_l722_722994

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722994


namespace prob_A_inter_B_l722_722740

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l722_722740


namespace find_y_l722_722491

theorem find_y (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 :=
by
  sorry

end find_y_l722_722491


namespace greatest_N_exists_l722_722904

def is_condition_satisfied (N : ℕ) (xs : Fin N → ℤ) : Prop :=
  ∀ i j : Fin N, i ≠ j → ¬ (1111 ∣ ((xs i) * (xs i) - (xs i) * (xs j)))

theorem greatest_N_exists : ∃ N : ℕ, (∀ M : ℕ, (∀ xs : Fin M → ℤ, is_condition_satisfied M xs → M ≤ N)) ∧ N = 1000 :=
by
  sorry

end greatest_N_exists_l722_722904


namespace ratio_is_9_l722_722818

-- Define the set of numbers
def set_of_numbers := { x : ℕ | ∃ n, n ≤ 8 ∧ x = 10^n }

-- Define the sum of the geometric series excluding the largest element
def sum_of_others : ℕ := (Finset.range 8).sum (λ n => 10^n)

-- Define the largest element
def largest_element := 10^8

-- Define the ratio of the largest element to the sum of the other elements
def ratio := largest_element / sum_of_others

-- Problem statement: The ratio is 9
theorem ratio_is_9 : ratio = 9 := by
  sorry

end ratio_is_9_l722_722818


namespace original_number_is_3199_l722_722370

theorem original_number_is_3199 (n : ℕ) (k : ℕ) (h1 : k = 3200) (h2 : (n + k) % 8 = 0) : n = 3199 :=
sorry

end original_number_is_3199_l722_722370


namespace num_rectangular_tables_l722_722414

theorem num_rectangular_tables (R : ℕ) 
  (rectangular_tables_seat : R * 10 = 70) :
  R = 7 := by
  sorry

end num_rectangular_tables_l722_722414


namespace removed_element_is_696_l722_722259

theorem removed_element_is_696 :
  let M := {1, 2, ..., 2017}
  let S := 2034172
  ∃ x: ℕ, x ∈ M ∧ (∃ k: ℕ, S - x = k * k) → x = 696 := by 
    sorry

end removed_element_is_696_l722_722259


namespace part1_proof_part2_proof_l722_722545

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - 1| - |x - m|

theorem part1_proof : ∀ x, f x 2 ≥ 1 ↔ x ≥ 2 :=
by 
  sorry

theorem part2_proof : (∀ x : ℝ, f x m ≤ 5) → (-4 ≤ m ∧ m ≤ 6) :=
by
  sorry

end part1_proof_part2_proof_l722_722545


namespace magic_money_box_l722_722649

theorem magic_money_box (initial_pennies : ℕ) (days : ℕ) 
  (triple_daily : ℕ → ℕ) : 
  (initial_pennies = 5) → 
  (∀ n, triple_daily n = n * 3) → 
  (days = 7) → 
  (triple_daily^[days] initial_pennies) = 10935 :=
by
  intros h_initial h_triple h_days
  have : triple_daily^[days + 1] initial_pennies = 10935
  { sorry }
  exact this

end magic_money_box_l722_722649


namespace standard_form_of_ellipse_and_point_D_l722_722394

-- Define the given conditions using Lean definitions
variable {a b c k  : ℝ}
variable {x y : ℝ}
variable {D : ℝ × ℝ}

noncomputable def M (x : ℝ) : ℝ := (2 * x * k - ((16 * k ^ 2 - 4) / (1 + 4 * k ^ 2))) / ((2 * x * k + (16 * k ^ 2 - 4) / (1 + 4 * k ^ 2)))

theorem standard_form_of_ellipse_and_point_D :
  (∀ a b c: ℝ, (c / a = sqrt 3 / 2 ∧ 1 / a^2 + 3 / (4 * b^2) = 1) → (∀ k : ℝ, (line_M = \frac{2k}{1 - 4k^{2}}x → M passes through D(0, 0)) ) )
  ∧ (standard_form_ellipse : ℝ) → ellipse equation = \( \frac{x^{2}}{4} + y^{2} = 1 \)
  sorry

end standard_form_of_ellipse_and_point_D_l722_722394


namespace difference_between_combined_length_and_width_l722_722588

noncomputable def floor_dimensions : ℝ × ℝ × ℝ × ℝ :=
  let L1 := real.sqrt (578 * 2)
  let W1 := ½ * L1
  let L2 := real.sqrt (450 * 3)
  let W2 := ⅓ * L2
  (L1, W1, L2, W2)

theorem difference_between_combined_length_and_width :
  let (L1, W1, L2, W2) := floor_dimensions in
  abs ((L1 + L2) - (W1 + W2) - 41.494) < 1e-4 :=
by
  sorry

end difference_between_combined_length_and_width_l722_722588


namespace laser_travel_distance_l722_722413

noncomputable def laser_path_distance (A B C D : ℝ × ℝ) : ℝ :=
  let D'' := (-D.1, -D.2)
  in Real.sqrt ((A.1 - D''.1)^2 + (A.2 - D''.2)^2)

theorem laser_travel_distance :
  let A := (2, 3 : ℝ) 
  let D := (8, 3 : ℝ)
  laser_path_distance A D = 2 * Real.sqrt 34 := 
by
  sorry

end laser_travel_distance_l722_722413


namespace garden_area_increase_l722_722426

noncomputable def original_garden_length : ℝ := 60
noncomputable def original_garden_width : ℝ := 8
noncomputable def original_garden_area : ℝ := original_garden_length * original_garden_width
noncomputable def original_garden_perimeter : ℝ := 2 * (original_garden_length + original_garden_width)

noncomputable def new_garden_side : ℝ := original_garden_perimeter / 3
noncomputable def new_garden_area : ℝ := (real.sqrt 3 / 4) * (new_garden_side ^ 2)

theorem garden_area_increase :
  new_garden_area - original_garden_area = 410.30 :=
begin
  sorry
end

end garden_area_increase_l722_722426


namespace max_val_f_on_1_to_e_min_val_f_on_1_to_e_f_less_than_g_on_1_to_inf_l722_722949

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3

-- Define the interval and properties to be checked.
def interval1 := Icc 1 Real.exp 1
def interval2 := {x : ℝ | 1 < x}

-- Maximum and minimum values on [1, e]
theorem max_val_f_on_1_to_e : ∀ x ∈ interval1, f x ≤ f Real.exp 1 := sorry
theorem min_val_f_on_1_to_e : ∀ x ∈ interval1, f Real.one ≤ f x := sorry

-- Inequality in the interval (1, +∞)
theorem f_less_than_g_on_1_to_inf : ∀ x ∈ interval2, f x < g x := sorry

end max_val_f_on_1_to_e_min_val_f_on_1_to_e_f_less_than_g_on_1_to_inf_l722_722949


namespace tan_theta_l722_722917

theorem tan_theta (θ : ℝ) (h1 : θ ∈ Ioo 0 real.pi) (h2 : sin θ + cos θ = 1 / 5) : tan θ = -4 / 3 :=
by
  sorry

end tan_theta_l722_722917


namespace positive_whole_numbers_with_cube_roots_less_than_15_l722_722116

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l722_722116


namespace number_of_prime_factors_30_factorial_l722_722972

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722972


namespace solve_inequality_2_star_x_l722_722495

theorem solve_inequality_2_star_x :
  ∀ x : ℝ, 
  6 < (2 * x - 2 - x + 3) ∧ (2 * x - 2 - x + 3) < 7 ↔ 5 < x ∧ x < 6 :=
by sorry

end solve_inequality_2_star_x_l722_722495


namespace oreo_shopping_ways_l722_722442

theorem oreo_shopping_ways :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_items := 4
  ∃ (ways : ℕ), ways = 2546 :=
begin
  sorry
end

end oreo_shopping_ways_l722_722442


namespace integer_solution_of_inequality_l722_722302

theorem integer_solution_of_inequality :
  ∀ (x : ℤ), 0 < (x - 1 : ℚ) * (x - 1) / (x + 1) ∧ (x - 1) * (x - 1) / (x + 1) < 1 →
  x > -1 ∧ x ≠ 1 ∧ x < 3 → 
  x = 2 :=
by
  sorry

end integer_solution_of_inequality_l722_722302


namespace work_completion_time_l722_722730

noncomputable def rate_b : ℝ := 1 / 24
noncomputable def rate_a : ℝ := 2 * rate_b
noncomputable def combined_rate : ℝ := rate_a + rate_b
noncomputable def completion_time : ℝ := 1 / combined_rate

theorem work_completion_time :
  completion_time = 8 :=
by
  sorry

end work_completion_time_l722_722730


namespace octagon_problem_l722_722336

noncomputable def area_of_octagon (side_length : ℚ) (ab_length : ℚ) : ℚ :=
  let area_one_triangle := (1 / 2) * ab_length * (side_length / 2)
  8 * area_one_triangle

theorem octagon_problem (m n : ℕ) (hmn_coprime : Nat.coprime m n) :
  let area := area_of_octagon 1 (73 / 144)
  (m : ℚ) / (n : ℚ) = area → m + n = 145 :=
by
  sorry

end octagon_problem_l722_722336


namespace analogical_reasoning_statements_incorrect_l722_722723

theorem analogical_reasoning_statements_incorrect :
  ¬ (∀ (s : ℕ), s = 1 → 
    ("Analogical reasoning is 'reasonable' reasoning, so its conjectured conclusion must be correct" ∨
     "Analogical reasoning is reasoning from general to specific" ∨
     "Analogical reasoning can be used to prove some mathematical propositions" ∨
     "Inductive reasoning is a form of analogical reasoning, so analogical reasoning is the same as inductive reasoning")) :=
by
  sorry

end analogical_reasoning_statements_incorrect_l722_722723


namespace good_quality_sufficient_condition_not_cheap_l722_722658

-- Define the statements
def good_quality : Prop := sorry
def not_cheap : Prop := sorry

-- Condition from Sister Qian's saying
axiom you_get_what_you_pay_for : good_quality ↔ not_cheap

-- Problem statement: Proving that "good quality" is a sufficient condition for "not cheap"
theorem good_quality_sufficient_condition_not_cheap : good_quality → not_cheap :=
by
  intro h,
  rw you_get_what_you_pay_for,
  exact h

end good_quality_sufficient_condition_not_cheap_l722_722658


namespace intersection_A_B_union_B_complement_A_l722_722092

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

end intersection_A_B_union_B_complement_A_l722_722092


namespace quadratic_non_real_roots_b_values_l722_722171

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722171


namespace find_PT_l722_722592

-- Declare the basic setup and conditions based on the problem description
variable {P Q R S T : Type}
variable {PQ QR : ℝ}

-- The geometric properties of rectangle and point T
variable Rectangle_PQRS : PQ = 15 ∧ QR = 5
variable angle_QRT_30 : ∠ Q R T = 30

-- Goal: Prove the length of PT
theorem find_PT (PQ QR : ℝ) (h_rectangle : PQ = 15 ∧ QR = 5) (h_angle : ∠ Q R T = 30) : 
  ∃ PT : ℝ, PT = 15 * real.sqrt 2 :=
by
  sorry

end find_PT_l722_722592


namespace number_of_prime_factors_30_factorial_l722_722980

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722980


namespace line_circle_intersect_l722_722955

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end line_circle_intersect_l722_722955


namespace twenty_five_percent_less_than_80_one_fourth_more_l722_722345

theorem twenty_five_percent_less_than_80_one_fourth_more (n : ℕ) (h : (5 / 4 : ℝ) * n = 60) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_one_fourth_more_l722_722345


namespace factorization_l722_722882

theorem factorization (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
sorry

end factorization_l722_722882


namespace RX_EQ_RY_l722_722390

variables (A B C P Q R X Y : Point)
variables [s : Triangle A B C]

-- Each condition is stated as follows:
-- AP = CQ (given)
-- RPBQ is cyclic (given)
-- tangents through A and C intersect RP and RQ at X and Y (given)
-- We need to prove RX = RY

noncomputable def AP_EQ_CQ (A P B C Q : Point) : Prop := dist A P = dist C Q
noncomputable def Cyclic_RPBQ (R P B Q : Point) : Prop := CyclicQuadrilateral R P B Q

noncomputable def TangentIntersection (A X C Y R P Q : Point) : Prop :=
  Tangent (circumcircle (TriangleCircumcircle A B C)) A X ∧ Tangent (circumcircle (TriangleCircumcircle A B C)) C Y ∧ 
  Collinear R P X ∧ Collinear R Q Y

theorem RX_EQ_RY
  (h1 : AP_EQ_CQ A P B C Q)
  (h2 : Cyclic_RPBQ R P B Q)
  (h3 : TangentIntersection A X C Y R P Q) :
  dist R X = dist R Y :=
sorry

end RX_EQ_RY_l722_722390


namespace determine_constants_l722_722461

structure Vector2D :=
(x : ℝ)
(y : ℝ)

def a := 11 / 20
def b := -7 / 20

def v1 : Vector2D := ⟨3, 2⟩
def v2 : Vector2D := ⟨-1, 6⟩
def v3 : Vector2D := ⟨2, -1⟩

def linear_combination (v1 v2 : Vector2D) (a b : ℝ) : Vector2D :=
  ⟨a * v1.x + b * v2.x, a * v1.y + b * v2.y⟩

theorem determine_constants (a b : ℝ) :
  ∃ (a b : ℝ), linear_combination v1 v2 a b = v3 :=
by
  use (11 / 20)
  use (-7 / 20)
  sorry

end determine_constants_l722_722461


namespace represent_380000_in_scientific_notation_l722_722667

theorem represent_380000_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 380000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.8 ∧ n = 5 :=
by
  sorry

end represent_380000_in_scientific_notation_l722_722667


namespace translate_sin_3x_add_pi_div4_l722_722680

theorem translate_sin_3x_add_pi_div4 : 
  ∀ (x : ℝ), sin(3 * (x + (π / 12))) = sin(3 * x + π / 4) :=
by sorry

end translate_sin_3x_add_pi_div4_l722_722680


namespace cookies_eaten_is_correct_l722_722377

-- Define initial and remaining cookies
def initial_cookies : ℕ := 7
def remaining_cookies : ℕ := 5
def cookies_eaten : ℕ := initial_cookies - remaining_cookies

-- The theorem we need to prove
theorem cookies_eaten_is_correct : cookies_eaten = 2 :=
by
  -- Here we would provide the proof
  sorry

end cookies_eaten_is_correct_l722_722377


namespace unique_not_in_range_l722_722318

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 23 = 23) (h₆ : f a b c d 101 = 101) (h₇ : ∀ x ≠ -d / c, f a b c d (f a b c d x) = x) :
  (a / c) = 62 := 
 sorry

end unique_not_in_range_l722_722318


namespace average_of_k_with_pos_int_roots_l722_722074

theorem average_of_k_with_pos_int_roots :
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ pair => pair.1 + pair.2)
  (k_values.sum / k_values.length) = 15 :=
by
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ pair => pair.1 + pair.2)
  have : k_values = [25, 14, 11, 10], by sorry
  let avg_k := (k_values.sum) / (k_values.length : ℕ)
  have : k_values.sum = 60, by sorry
  have : k_values.length = 4, by sorry
  have : avg_k = 15, by sorry
  show avg_k = 15, from this

end average_of_k_with_pos_int_roots_l722_722074


namespace history_class_test_l722_722795

theorem history_class_test (n : ℕ) (scores : fin n → ℕ) :
  (∀ i : fin 4, scores i = 100) →
  (∀ i : fin n, scores i ≥ 50) →
  (∃ i₁ i₂ : fin n, scores i₁ ≥ 70 ∧ scores i₂ ≥ 70) →
  (∑ i, scores i = 81 * n) →
  n ≥ 6 :=
by
  sorry

end history_class_test_l722_722795


namespace pears_in_basket_l722_722308

def TaniaFruits (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 18 ∧ b2 = 12 ∧ b3 = 9 ∧ b4 = b3 ∧ b5 + b1 + b2 + b3 + b4 = 58

theorem pears_in_basket {b1 b2 b3 b4 b5 : ℕ} (h : TaniaFruits b1 b2 b3 b4 b5) : b5 = 10 :=
by 
  sorry

end pears_in_basket_l722_722308


namespace radius_of_sphere_l722_722564

theorem radius_of_sphere (area : ℝ) (h : area = 4 * Real.pi) : 
  ∃ R, R = 2 ∧ Real.pi * R^2 = area :=
by
  use 2
  split
  · rfl
  · rw [h, Real.pi]
  sorry

end radius_of_sphere_l722_722564


namespace sum_of_primitive_roots_mod_11_l722_722493

def is_primitive_root (a p : ℕ) : Prop :=
  a.gcd(p) = 1 ∧ ∀ k : ℕ, (0 < k ∧ k < p) → a ^ k % p ≠ 1

def primitive_roots (p : ℕ) : List ℕ :=
  (List.range p).filter (λ a, is_primitive_root a p)

def sum_primitive_roots (p : ℕ) : ℕ :=
  (primitive_roots p).sum

theorem sum_of_primitive_roots_mod_11 : 
  sum_primitive_roots 11 = 8 := 
by 
  sorry

end sum_of_primitive_roots_mod_11_l722_722493


namespace probability_intersection_l722_722749

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l722_722749


namespace find_a_l722_722088

def function_y (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 + a * x - 5

def derivative_f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

theorem find_a (a : ℝ) (h : ∀ x ∈ Ioo (-3 : ℝ) 1, derivative_f a x < 0) : a = -3 := sorry

end find_a_l722_722088


namespace generatrix_length_of_cone_l722_722047

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l722_722047


namespace door_height_eight_l722_722221

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l722_722221


namespace cone_generatrix_length_l722_722061

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l722_722061


namespace cards_draw_proof_l722_722704

noncomputable def num_ways_to_draw_cards :=
  let color_count : ℕ := 3 in
  let label_count : ℕ := 5 in
  let subset_size : ℕ := 5 in
  let case1 := (color_count.choose 1) * (label_count.choose 3) * (2.choose 1) * (2.choose 1) / nat.factorial 2 in
  let case2 := (color_count.choose 1) * (label_count.choose 2) * (2.choose 1) * (3.choose 2) * (1.choose 1) / nat.factorial 2 in
  case1 + case2

theorem cards_draw_proof :
  num_ways_to_draw_cards = 150 := sorry

end cards_draw_proof_l722_722704


namespace exists_arithmetic_mean_not_greater_l722_722262

variable {α : Type*} [LinearOrderedField α]

theorem exists_arithmetic_mean_not_greater
  (n : ℕ) 
  (x : Fin (n+1) → α) 
  (X : α)
  (hx : (∑ i : Fin (n+1), x i) / (n + 1) = X) :
  ∃ (K : ℕ), 0 < K ∧ K ≤ n + 1 ∧ 
  ∀ j, 1 ≤ j → j ≤ K → (∑ i in Finset.range (K - j + 1), x ⟨j + i - 1, by linarith⟩) / (K - j + 1) ≤ X := 
sorry

end exists_arithmetic_mean_not_greater_l722_722262


namespace quadratic_non_real_roots_b_values_l722_722176

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l722_722176


namespace cos_periodic_l722_722678

theorem cos_periodic (x : ℝ) : ∃ p > 0, (∀ x : ℝ, cos (x + p) = cos x) ∧ p = 2 * π := 
sorry

end cos_periodic_l722_722678


namespace isosceles_triangle_l722_722645

variable (A B C C1 C2 : Type) [Point : Type]
hypotheses (triangleABC : Triangle A B C) 

def on (X Y Z : Type) : Prop := ConditionForLyingOn X Y Z

theorem isosceles_triangle 
  (h1 : on C1 A C)
  (h2 : on C2 B C)
  (h3 : congruent (triangle A B C1) (triangle B A C2)) :
  isosceles (triangle A B C) :=
sorry

end isosceles_triangle_l722_722645


namespace minimum_small_bottles_l722_722608

theorem minimum_small_bottles (small_bottle_cap large_bottle_cap vase_cap : ℕ)
  (h_small : small_bottle_cap = 45)
  (h_large : large_bottle_cap = 675)
  (h_vase : vase_cap = 95) :
  let bottles_for_large := large_bottle_cap / small_bottle_cap in
  let remainder_after_large := large_bottle_cap % small_bottle_cap in
  let bottles_for_vase := (vase_cap + remainder_after_large + small_bottle_cap - 1) / small_bottle_cap in
  bottles_for_large + bottles_for_vase = 18 :=
by
  -- The proof will go here
  sorry

end minimum_small_bottles_l722_722608


namespace chord_length_proof_l722_722228

noncomputable def chord_length {ρ θ : ℝ} (h1 : ρ = 4 * Real.sin θ) (h2 : ρ * Real.sin θ = 3) : ℝ :=
  2 * Real.sqrt(3)

theorem chord_length_proof : chord_length (by sorry) (by sorry) = 2 * Real.sqrt(3) :=
  by sorry

end chord_length_proof_l722_722228


namespace count_satisfying_ns_l722_722906

theorem count_satisfying_ns :
  let s := list.range' 2 24
  (∃ n : ℕ, n > 0 ∧ (n ∈ s → 
    (list.prod (s.map (λ x, n - x)) < 0)) = 12) :=
sorry

end count_satisfying_ns_l722_722906


namespace cos_90_l722_722593

def right_triangle (DE DF EF : ℝ) (D E F : Type*) := 
  is_right_angle D E F

theorem cos_90 (D E F : Type*) (DE EF : ℝ) (h : right_triangle D E F) (h_angle_D : ∠D = 90) : 
  cos (degToRad 90) = 0 :=
begin
  sorry
end

end cos_90_l722_722593


namespace trapezium_area_l722_722710

theorem trapezium_area (a b h : ℝ) (h_a : a = 4) (h_b : b = 5) (h_h : h = 6) :
  (1 / 2 * (a + b) * h) = 27 :=
by
  rw [h_a, h_b, h_h]
  norm_num

end trapezium_area_l722_722710


namespace door_height_is_eight_l722_722213

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l722_722213


namespace ratio_of_A_to_B_l722_722287

variable {R : Type} [LinearOrderedField R]

/-- Shares of A, B, and C --/
def A_share : R := 408
def B_share : R := 102
def C_share : R := 68

/-- Condition that B's share is 1/4 of C's share --/
def B_share_condition : Prop := B_share = (1 / 4) * C_share

/-- The ratio of A's share to B's share is 4:1 --/
theorem ratio_of_A_to_B : B_share_condition → A_share / B_share = 4 :=
by
  intro h
  sorry

end ratio_of_A_to_B_l722_722287


namespace number_of_clicks_approx_40_l722_722691

noncomputable def distance_per_click : ℝ := 50
noncomputable def initial_speed_mph : ℝ := 30
noncomputable def final_speed_mph : ℝ := 60
noncomputable def duration_seconds : ℝ := 30

def average_speed_mph : ℝ :=
  (initial_speed_mph + final_speed_mph) / 2

def average_speed_fpm : ℝ :=
  average_speed_mph * 5280 / 60

def clicks_per_minute : ℝ :=
  average_speed_fpm / distance_per_click

def clicks_in_30_seconds : ℝ :=
  clicks_per_minute / 2

theorem number_of_clicks_approx_40 :
  abs (clicks_in_30_seconds - 40) < 1 :=
sorry

end number_of_clicks_approx_40_l722_722691


namespace sample_size_is_50_l722_722706

theorem sample_size_is_50
  (population : ℕ)
  (individual : Type)
  (sample : set individual)
  (sample_size : ℕ) :
  population = 200 →
  (∀ student : individual, student ∈ population) →
  (∀ sample_student : individual, sample_student ∈ sample) →
  sample_size = 50 →
  sample_size = 50 :=
by sorry

end sample_size_is_50_l722_722706


namespace sqrt_continued_fraction_l722_722866

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l722_722866


namespace find_height_of_door_l722_722224

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l722_722224


namespace steve_total_money_l722_722663

theorem steve_total_money
    (nickels : ℕ)
    (dimes : ℕ)
    (nickel_value : ℕ := 5)
    (dime_value : ℕ := 10)
    (cond1 : nickels = 2)
    (cond2 : dimes = nickels + 4) 
    : (nickels * nickel_value + dimes * dime_value) = 70 := by
  sorry

end steve_total_money_l722_722663


namespace shortest_side_of_similar_triangle_l722_722427

theorem shortest_side_of_similar_triangle (h1 : ∀ (a b c : ℝ), a^2 + b^2 = c^2)
  (h2 : 15^2 + b^2 = 34^2) (h3 : ∃ (k : ℝ), k = 68 / 34) :
  ∃ s : ℝ, s = 2 * Real.sqrt 931 :=
by
  sorry

end shortest_side_of_similar_triangle_l722_722427


namespace right_triangle_short_leg_l722_722580

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l722_722580


namespace ellipse_eccentricity_half_l722_722932

noncomputable def ellipse_eccentricity (k : ℝ) (h : k > -1) : ℝ :=
  let a := 2 in
  let b := sqrt (k + 1) in
  let c := sqrt (a^2 - b^2) in
  c / a

theorem ellipse_eccentricity_half (k : ℝ) (h : k > -1) (h_perimeter : 4 * 2 = 8) :
  ellipse_eccentricity k h = 1 / 2 :=
by
  have ha : 2 = 2 := rfl
  have hk : k = 2 := by
    calc
      k + 2 = 4 : by rw [←ha, four_a]; norm_num
      k = 2 : by linarith
  rw [hk] at *
  calc_ellipse_eccentricity : ellipse_eccentricity 2 _ = 1 / 2 := sorry
  exact calc_ellipse_eccentricity

end ellipse_eccentricity_half_l722_722932


namespace generatrix_length_of_cone_l722_722021

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l722_722021


namespace unknown_rate_of_two_blankets_is_285_l722_722379

-- Question and Conditions:
def cost_known_blankets (n1 n2 : ℕ) (p1 p2 : ℕ) : ℕ := n1 * p1 + n2 * p2
def average_price (total_cost total_blankets : ℕ) : ℕ := total_cost / total_blankets
def total_cost_of_blankets (avg_price num_blankets : ℕ) : ℕ := avg_price * num_blankets

axiom blankets_condition 
  (n1 n2 n3 p1 p2 : ℕ) 
  (avg_price : ℕ) : 
  n1 = 3 →
  n2 = 5 →
  n3 = 2 →
  p1 = 100 →
  p2 = 150 →
  avg_price = 162 →
  let total_known_cost := cost_known_blankets n1 n2 p1 p2 in
  let total_cost := total_cost_of_blankets avg_price (n1 + n2 + n3) in
  let unknown_rate := (total_cost - total_known_cost) / n3 in
  unknown_rate = 285

-- Conditions as definitions:
def conditions : Prop :=
  blankets_condition 3 5 2 100 150 162

-- The proof problem:
theorem unknown_rate_of_two_blankets_is_285 : 
  conditions → 
  true :=
by
  intro h
  sorry

end unknown_rate_of_two_blankets_is_285_l722_722379


namespace intersection_A_B_union_B_complement_A_l722_722091

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

end intersection_A_B_union_B_complement_A_l722_722091


namespace number_of_prime_factors_30_factorial_l722_722979

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722979


namespace number_of_cube_roots_lt_15_l722_722110

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l722_722110


namespace log_eq_solution_l722_722300

theorem log_eq_solution (x : ℝ) (h : log 8 x + log 4 (x^3) = 9) : x = 2^(54/5) :=
by
  sorry

end log_eq_solution_l722_722300


namespace necessary_and_sufficient_condition_for_x2_ne_y2_l722_722695

theorem necessary_and_sufficient_condition_for_x2_ne_y2 (x y : ℤ) :
  (x ^ 2 ≠ y ^ 2) ↔ (x ≠ y ∧ x ≠ -y) :=
by
  sorry

end necessary_and_sufficient_condition_for_x2_ne_y2_l722_722695


namespace sum_of_cubes_of_rel_prime_numbers_is_multiple_l722_722261

open Nat

theorem sum_of_cubes_of_rel_prime_numbers_is_multiple (n : ℕ) (h : n ≥ 3) :
  n ∣ (∑ i in Finset.filter (λ a, Nat.coprime a n) (Finset.range (n + 1)), i^3) :=
by
  sorry

end sum_of_cubes_of_rel_prime_numbers_is_multiple_l722_722261


namespace sqrt_recursive_value_l722_722843

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l722_722843


namespace cone_volume_l722_722777

noncomputable def radius_of_sector : ℝ := 6
noncomputable def arc_length_of_sector : ℝ := (1 / 2) * (2 * Real.pi * radius_of_sector)
noncomputable def radius_of_base : ℝ := arc_length_of_sector / (2 * Real.pi)
noncomputable def slant_height : ℝ := radius_of_sector
noncomputable def height_of_cone : ℝ := Real.sqrt (slant_height^2 - radius_of_base^2)
noncomputable def volume_of_cone : ℝ := (1 / 3) * Real.pi * (radius_of_base^2) * height_of_cone

theorem cone_volume : volume_of_cone = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end cone_volume_l722_722777


namespace number_of_cube_roots_lt_15_l722_722113

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l722_722113


namespace algebraic_expression_value_l722_722012

variables (a b c d m : ℤ)

def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℤ) : Prop := c * d = 1
def abs_eq_2 (m : ℤ) : Prop := |m| = 2

theorem algebraic_expression_value {a b c d m : ℤ} 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : abs_eq_2 m) :
  (2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0) :=
by
  sorry

end algebraic_expression_value_l722_722012


namespace cone_generatrix_length_l722_722054

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l722_722054


namespace max_product_of_incongruent_rectangles_l722_722396

theorem max_product_of_incongruent_rectangles :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (length rectangles = 5 ∧ 
   (∀ r1 r2 ∈ rectangles, r1 ≠ r2 → (r1 ≠ r2 ∧ r2 ≠ r1)) ∧
   (∀ r ∈ rectangles, 1 ≤ r.fst ∧ r.fst ≤ 5 ∧ 1 ≤ r.snd ∧ r.snd ≤ 5) ∧ 
   (∑ r in rectangles, r.fst * r.snd = 25) ∧
   (∏ r in rectangles, r.fst * r.snd = 2304)) :=
sorry

end max_product_of_incongruent_rectangles_l722_722396


namespace fabian_total_cost_l722_722478

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem fabian_total_cost : total_cost = 111 := 
by 
  unfold total_cost mouse_cost keyboard_cost headphones_cost usb_hub_cost
  sorry

end fabian_total_cost_l722_722478


namespace p_necessary_but_not_sufficient_for_q_l722_722009

variable (x : ℝ)

def p := x + 1 ≥ 0
def q := |x - 1| < 2

theorem p_necessary_but_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) := by sorry

end p_necessary_but_not_sufficient_for_q_l722_722009


namespace richmond_tigers_revenue_l722_722312

theorem richmond_tigers_revenue
  (total_tickets : ℕ)
  (first_half_tickets : ℕ)
  (catA_first_half : ℕ)
  (catB_first_half : ℕ)
  (catC_first_half : ℕ)
  (priceA : ℕ)
  (priceB : ℕ)
  (priceC : ℕ)
  (catA_second_half : ℕ)
  (catB_second_half : ℕ)
  (catC_second_half : ℕ)
  (total_revenue_second_half : ℕ)
  (h_total_tickets : total_tickets = 9570)
  (h_first_half_tickets : first_half_tickets = 3867)
  (h_catA_first_half : catA_first_half = 1350)
  (h_catB_first_half : catB_first_half = 1150)
  (h_catC_first_half : catC_first_half = 1367)
  (h_priceA : priceA = 50)
  (h_priceB : priceB = 40)
  (h_priceC : priceC = 30)
  (h_catA_second_half : catA_second_half = 1350)
  (h_catB_second_half : catB_second_half = 1150)
  (h_catC_second_half : catC_second_half = 1367)
  (h_total_revenue_second_half : total_revenue_second_half = 154510)
  :
  catA_second_half * priceA + catB_second_half * priceB + catC_second_half * priceC = total_revenue_second_half :=
by
  sorry

end richmond_tigers_revenue_l722_722312


namespace non_real_roots_interval_l722_722190

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722190


namespace focus_coords_correct_l722_722924

noncomputable def focus_of_parabola {a : ℝ} (l : ℝ → ℝ) (parabola : ℝ → ℝ) : Prop :=
∀ (x y : ℝ), l x = y → parabola x = y^2 → (segment_length : ℝ) = 4 → (focus_coords : ℝ × ℝ) = (1, 0)

theorem focus_coords_correct :
  focus_of_parabola (λ x, 1) (λ x, 4 * x) :=
sorry

end focus_coords_correct_l722_722924


namespace smallest_n_terminating_contains_2_l722_722354

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_2 (n : ℕ) : Prop :=
  n.to_string.contains '2'

theorem smallest_n_terminating_contains_2 :
  ∃ n : ℕ, is_terminating_decimal n ∧ contains_digit_2 n ∧ (∀ m : ℕ, (is_terminating_decimal m ∧ contains_digit_2 m) → n ≤ m) :=
begin
  use 2,
  split,
  { existsi (1 : ℕ),
    existsi (0 : ℕ),
    exact rfl,
  },
  split,
  { sorry },  -- Proof that 2 contains the digit 2 in its string representation.
  { intros m hm,
    cases hm with H1 H2,
    cases H1 with a b,
    sorry  -- Proof that 2 is the smallest term satisfying the conditions.
  }
end

end smallest_n_terminating_contains_2_l722_722354


namespace classify_discuss_l722_722599

theorem classify_discuss (a b c : ℚ) (h : a * b * c > 0) : 
  (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) :=
sorry

end classify_discuss_l722_722599


namespace cubs_win_series_probability_l722_722310

theorem cubs_win_series_probability :
  ∑ k in Finset.range 5, (Nat.choose (4 + k) k * ((3:ℚ) / 5)^5 * ((2:ℚ) / 5)^k) = 243 / 625 :=
by
  sorry

end cubs_win_series_probability_l722_722310


namespace trig_maps_transform_l722_722435

def trigonometric_map := ℝ → ℝ

axiom sin : trigonometric_map
axiom cos : trigonometric_map
axiom tan : trigonometric_map
axiom arcsin : trigonometric_map
axiom arccos : trigonometric_map
axiom arctan : trigonometric_map

theorem trig_maps_transform (x : ℚ)
  (hx : 0 < x) :
  ∃ (t : list trigonometric_map), x = (t.foldr (λ step acc, step acc) 0) :=
sorry

end trig_maps_transform_l722_722435


namespace volume_of_rotation_l722_722815

noncomputable def volume_of_solid (f : ℝ → ℝ) (a b : ℝ) :=
  π * (∫ x in a..b, (f x)^2)

theorem volume_of_rotation :
  volume_of_solid (λ x, x * Real.exp x) 0 1 = (π * (Real.exp 2 - 1)) / 4 :=
by
  sorry

end volume_of_rotation_l722_722815


namespace infinite_folding_methods_l722_722712

theorem infinite_folding_methods (center_fold: ∀ l, is_line(l) ∧ passes_through_center(l) ∧ divides_ into_equal_parts(l)) :
  infinite (number_of_folding_methods(center_fold)) := sorry

end infinite_folding_methods_l722_722712


namespace non_real_roots_interval_l722_722185

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l722_722185


namespace toothbrushes_per_patient_l722_722408

theorem toothbrushes_per_patient :
  (hours_per_day : ℕ)
  (hours_per_visit : ℕ)
  (days_per_week : ℕ)
  (total_toothbrushes : ℕ) :
  hours_per_day = 8 →
  hours_per_visit = 1/2 →
  days_per_week = 5 →
  total_toothbrushes = 160 →
  total_toothbrushes / (hours_per_day / hours_per_visit * days_per_week) = 2 :=
by
  sorry

end toothbrushes_per_patient_l722_722408


namespace number_of_prime_factors_30_factorial_l722_722981

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722981


namespace probability_intersection_l722_722753

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l722_722753


namespace percentage_error_calc_l722_722380

theorem percentage_error_calc (a : ℚ) : 
  let incorrect_factor := (3 : ℚ) / 5
  let correct_factor := (5 : ℚ) / 3
  let ratio := incorrect_factor / correct_factor
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  -- Define the factors
  let incorrect_factor := (3 : ℚ) / 5
  let correct_factor := (5 : ℚ) / 3
  -- Compute the ratio
  let ratio := incorrect_factor / correct_factor
  -- Compute the percentage error
  let percentage_error := (1 - ratio) * 100
  -- Assert the percentage error is 64%
  have h1 : ratio = (9 : ℚ) / 25 := by sorry
  have h2 : (1 - (9 : ℚ) / 25) * 100 = 64 := by sorry
  exact h2

end percentage_error_calc_l722_722380


namespace given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l722_722632

theorem given_conditions_implies_a1d1_a2d2_a3d3_eq_zero
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, 
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - x + 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 0 :=
by
  sorry

end given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l722_722632


namespace quadrant_and_tan_l722_722533

-- Define the basic setup and assumptions
variables {m : ℝ} (h_nonzero : m ≠ 0)
-- Define the point P and the given sin(alpha)
def P := (-Real.sqrt 3, m)
def sin_alpha := (Real.sqrt 3 / 4) * m

-- Define the conditions as hypotheses
axiom h1 : sin_alpha = m / Real.sqrt (3 + m^2)

-- Lean proof statement for the equivalent problem
theorem quadrant_and_tan (h_nonzero : m ≠ 0) (h_sin : (Real.sqrt 3 / 4) * m = m / Real.sqrt (3 + m^2)) :
  (∃ q : ℤ, (q = 2 ∨ q = 3) ∧
  (if q = 2 then tan α = -Real.sqrt 7 / 3 else tan α = Real.sqrt 7 / 3)) :=
by
  sorry

end quadrant_and_tan_l722_722533


namespace sum_of_abscissas_l722_722885

theorem sum_of_abscissas
  (f g : ℝ → ℝ)
  (h_eq : ∀ x, f x = 8 * Cos (π * x) * (Cos (2 * π * x))^2 * Cos (4 * π * x))
  (h_eq' : ∀ x, g x = Cos (5 * π * x))
  (h_common_points : ∀ x, f x = g x)
  (h_abscissas : ∀ x, x ∈ Icc (-1 : ℝ) 0) :
  ∑ k in (Finset.filter (λ x : ℝ, h_common_points x) (Finset.Icc (-1) 0)), k = -4.5 :=
sorry

end sum_of_abscissas_l722_722885


namespace sqrt_sum_in_terms_of_n_l722_722918

theorem sqrt_sum_in_terms_of_n (n : ℝ) (h : sqrt 15 = n) : sqrt 0.15 + sqrt 1500 = (101 / 10) * n :=
by
  sorry

end sqrt_sum_in_terms_of_n_l722_722918


namespace door_height_is_eight_l722_722210

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l722_722210


namespace inequality_proof_l722_722525

variable {a b c : ℝ}

theorem inequality_proof (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2*Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := 
sorry

end inequality_proof_l722_722525


namespace num_pos_nums_with_cube_root_lt_15_l722_722132

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l722_722132


namespace growth_rate_correct_max_avg_visitors_correct_l722_722375

-- Define the conditions from part 1
def visitors_march : ℕ := 80000
def visitors_may : ℕ := 125000

-- Define the monthly average growth rate
def monthly_avg_growth_rate (x : ℝ) : Prop :=
(1 + x)^2 = (visitors_may / visitors_march : ℝ)

-- Define the condition for June
def visitors_june_1_10 : ℕ := 66250
def max_avg_visitors_per_day (y : ℝ) : Prop :=
6.625 + 20 * y ≤ 15.625

-- Prove the monthly growth rate
theorem growth_rate_correct : ∃ x : ℝ, monthly_avg_growth_rate x ∧ x = 0.25 := sorry

-- Prove the max average visitors per day in June
theorem max_avg_visitors_correct : ∃ y : ℝ, max_avg_visitors_per_day y ∧ y = 0.45 := sorry

end growth_rate_correct_max_avg_visitors_correct_l722_722375


namespace proof_problem_l722_722936

variable (m n : Type) (α β : Type)
variables [IsLine m] [IsLine n] [IsPlane α] [IsPlane β]
variables [AreDifferentLines m n] [AreDifferentPlanes α β]

-- Proposition for problem's condition B:
variable [are_parallel : ∀ m α, IsParallelLinePlane m α]
variable [are_perpendicular : ∀ n β, IsPerpendicularLinePlane n β ]
variable [are_parallel_lines : ∀ m n, IsParallelLines m n ]

theorem proof_problem :
  are_parallel m α → are_perpendicular n β → are_parallel_lines m n → ArePerpendicularPlanes α β :=
by sorry

end proof_problem_l722_722936


namespace no_separating_plane_l722_722627

theorem no_separating_plane (n : ℕ)
  (h_n : 2 ≤ n)
  (f : Fin n → Fin n → Fin n → ℝ)
  (h_nonzero_each : ∀ x y z, f x y z ≠ 0)
  (h_sum_1 : ∀ i : Fin n, ∑ y z, f i y z = 0)
  (h_sum_2 : ∀ j : Fin n, ∑ x z, f x j z = 0)
  (h_sum_3 : ∀ k : Fin n, ∑ x y, f x y k = 0) :
  ¬ (∃ a b c d : ℝ, ∀ x y z : Fin n,
      (a * (x : ℝ) + b * (y : ℝ) + c * (z : ℝ) + d) * f x y z > 0) :=
sorry

end no_separating_plane_l722_722627


namespace sum_of_reciprocals_l722_722699

variable (x y : ℝ)

theorem sum_of_reciprocals (h1 : x + y = 10) (h2 : x * y = 20) : 1 / x + 1 / y = 1 / 2 :=
by
  sorry

end sum_of_reciprocals_l722_722699


namespace shorter_leg_in_right_triangle_l722_722577

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l722_722577


namespace volume_of_rectangular_box_l722_722358

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l722_722358


namespace box_volume_l722_722364

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l722_722364


namespace evaluate_nested_radical_l722_722857

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l722_722857


namespace length_generatrix_cone_l722_722062

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l722_722062


namespace batsman_average_l722_722728

variable (x : ℝ)

theorem batsman_average (h1 : ∀ x, 11 * x + 55 = 12 * (x + 1)) : 
  x = 43 → (x + 1 = 44) :=
by
  sorry

end batsman_average_l722_722728


namespace payback_time_l722_722616

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l722_722616


namespace melissa_earnings_from_sales_l722_722272

noncomputable def commission_earned (coupe_price suv_price commission_rate : ℕ) : ℕ :=
  (coupe_price + suv_price) * commission_rate / 100

theorem melissa_earnings_from_sales : 
  commission_earned 30000 60000 2 = 1800 :=
by
  sorry

end melissa_earnings_from_sales_l722_722272


namespace number_of_cube_roots_lt_15_l722_722114

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l722_722114


namespace range_of_k_l722_722089

-- Given function f
def f (x : ℝ) : ℝ := sqrt 3 * sin x + cos x

-- Definition of g based on the transformations from f
def g (x : ℝ) : ℝ := f (2 * (x - π / 3))

-- Main theorem statement
theorem range_of_k :
  ∀ k : ℝ,
  1 ≤ k ∧ k < 2 →
  ∃ x1 x2 : ℝ,
    0 ≤ x1 ∧ x1 ≤ π / 2 ∧
    0 ≤ x2 ∧ x2 ≤ π / 2 ∧
    g x1 = k ∧
    g x2 = k ∧
    x1 ≠ x2 :=
begin
  sorry
end

end range_of_k_l722_722089


namespace height_of_water_sum_a_plus_b_l722_722337

-- Definitions of the given problem
def radius_of_tank := 16
def height_of_tank := 96
def water_percentage := 0.25

-- The volume formula of a cone
def volume_of_cone (r h : ℝ) : ℝ := (1/3) * Math.pi * r^2 * h

-- Let's define the volume of the full tank and the water in the tank
def volume_of_tank := volume_of_cone radius_of_tank height_of_tank
def volume_of_water := water_percentage * volume_of_tank

-- The height of the water in the tank, in the form a cbrt b
def height_of_water : ℝ := 48 * real.cbrt 2

-- The proof statement: the main goal
theorem height_of_water_sum_a_plus_b : 
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (b % (2^3) ≠ 0) ∧ height_of_water = a * real.cbrt b ∧ a + b = 50 :=
sorry

end height_of_water_sum_a_plus_b_l722_722337


namespace quadratic_continued_fraction_period_2_converse_not_universally_true_l722_722733

-- Define the problem conditions and statement for part (a)
theorem quadratic_continued_fraction_period_2 
  (a b : ℕ) (h : a ≠ b) : 
  ∃ α > 0, α^2 - (ab)/b * α - a/b = 0 ∧ α is a purely periodic continued fraction with period length 2 := 
sorry

-- Define the problem conditions and statement for part (b)
theorem converse_not_universally_true : 
  ∃ ( α > 0 ) (a b : ℕ), 
  α^2 - (ab)/b * α - a/b ≠ 0 is a purely periodic continued fraction with period length 2 := 
sorry

end quadratic_continued_fraction_period_2_converse_not_universally_true_l722_722733


namespace problem_integer_square_l722_722008

theorem problem_integer_square 
  (a b c d A : ℤ) 
  (H1 : a^2 + A = b^2) 
  (H2 : c^2 + A = d^2) : 
  ∃ (k : ℕ), 2 * (a + b) * (c + d) * (a * c + b * d - A) = k^2 :=
by
  sorry

end problem_integer_square_l722_722008


namespace officers_election_l722_722804

theorem officers_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ)
  (h_candidates : total_candidates = 18)
  (h_past_officers : past_officers = 8)
  (h_positions : positions = 6) :
  (Nat.choose total_candidates positions) - (Nat.choose (total_candidates - past_officers) positions) = 18354 :=
by
  have hc : Nat.choose 18 6 = 18564 := 
  sorry -- Calculation for total number of ways to choose 6 from 18
  have hn : Nat.choose 10 6 = 210 := 
  sorry -- Calculation for number of ways to choose 6 from 10 candidates
  calc
    Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions 
        = Nat.choose 18 6 - Nat.choose 10 6 : by
            rw [h_candidates, h_past_officers, h_positions]
        ... = 18564 - 210 : by
            rw [hc, hn]
        ... = 18354 : by
            norm_num

end officers_election_l722_722804


namespace equation_of_hyperbola_l722_722529

-- Definitions for conditions

def center_at_origin (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def focus_point (focus : ℝ × ℝ) : Prop :=
  focus = (Real.sqrt 2, 0)

def distance_to_asymptote (focus : ℝ × ℝ) (distance : ℝ) : Prop :=
  -- Placeholder for the actual distance calculation
  distance = 1 -- The given distance condition in the problem

-- The mathematical proof problem statement

theorem equation_of_hyperbola :
  center_at_origin (0,0) ∧
  focus_point (Real.sqrt 2, 0) ∧
  distance_to_asymptote (Real.sqrt 2, 0) 1 → 
    ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧
    (a^2 + b^2 = 2) ∧ (a^2 = 1) ∧ (b^2 = 1) ∧ 
    (∀ x y : ℝ, b^2*y^2 = x^2 - a^2*y^2 → (y = 0 ∧ x^2 = 1)) :=
sorry

end equation_of_hyperbola_l722_722529


namespace cotangent_inequality_l722_722600

variable {A B C : Type} [Triangle A B C]

def condition (BC2 AC2 AB2 : ℝ) : Prop := AC2 = (BC2 + AB2) / 2

theorem cotangent_inequality (BC AB AC : ℝ) (S : ℝ)
  (h : condition BC AC AB)
  (cotA cotB cotC : ℝ)
  (h1 : cotA = (AC^2 + AB^2 - BC^2) / (4 * S))
  (h2 : cotB = (BC^2 + AB^2 - AC^2) / (4 * S))
  (h3 : cotC = (BC^2 + AC^2 - AB^2) / (4 * S)) :
  cotB^2 ≥ cotA * cotC :=
sorry

end cotangent_inequality_l722_722600


namespace min_value_E_l722_722506

noncomputable def E (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  (∑ i in finset.range n, x i ^ 2) +
  (∑ i in finset.range (n-1), x i * x (i + 1)) +
  (∑ i in finset.range n, x i)

theorem min_value_E (n : ℕ) (x : ℕ → ℝ) : 
  E n x ≥ - (n + 2) / 8 := 
sorry

end min_value_E_l722_722506


namespace min_quotient_four_digits_l722_722462

theorem min_quotient_four_digits : 
  ∃ a b c d : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧
  (1000 * a + 100 * b + 10 * c + d) / (a + b + c + d) = 71.56 :=
sorry

end min_quotient_four_digits_l722_722462


namespace range_of_slope_l722_722509

theorem range_of_slope (k : ℝ) :
  (∃ P₁ P₂ P₃ : ℝ × ℝ, P₁ ≠ P₂ ∧ P₁ ≠ P₃ ∧ P₂ ≠ P₃ ∧
    (x^2 + y^2 - 4 * x - 4 * y = 0) ∧
    (sqrt(2)) = abs ((k * x - y) / sqrt (k ^ 2 + 1))) →
  2 - sqrt 3 ≤ k ∧ k ≤ 2 + sqrt 3 :=
by
  sorry

end range_of_slope_l722_722509


namespace solve_log_eq_l722_722296

theorem solve_log_eq (x : ℝ) (h : log 8 x + log 4 (x ^ 3) = 9) : 
  x = 2^(54 / 11) :=
by 
  sorry

end solve_log_eq_l722_722296


namespace payback_time_l722_722617

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l722_722617


namespace generatrix_length_l722_722038

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l722_722038


namespace part_one_part_two_l722_722002

noncomputable theory

-- Define the sequence a_n
def a : ℕ+ → ℝ
| 1 := 1
| (n+1) := a n / (2 * a n + 1)

-- Define the arithmetic sequence reciprocal of a_n
def reciprocal_seq_is_arithmetic : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ, 1 / a n = (2 * k - 1)

theorem part_one : reciprocal_seq_is_arithmetic := sorry

-- Define the sequence b_n
def b (n : ℕ+) : ℝ := a n * a (n+1)

-- Define the sum of the first n terms of b_n, T_n, and its expected value
def sum_seq_b_is_correct (n : ℕ+) : Prop :=
  let T : ℕ+ → ℝ := λ k, ∑ i in finset.range k, b i.succ
  in T n = n / (2 * n + 1)

theorem part_two (n : ℕ+) : sum_seq_b_is_correct n := sorry

end part_one_part_two_l722_722002


namespace simple_interest_sum_l722_722738

theorem simple_interest_sum (P_SI : ℕ) :
  let P_CI := 5000
  let r_CI := 12
  let t_CI := 2
  let r_SI := 10
  let t_SI := 5
  let CI := (P_CI * (1 + r_CI / 100)^t_CI - P_CI)
  let SI := CI / 2
  (P_SI * r_SI * t_SI / 100 = SI) -> 
  P_SI = 1272 := by {
  sorry
}

end simple_interest_sum_l722_722738


namespace log_base_4_of_fraction_l722_722471

theorem log_base_4_of_fraction :
  log 4 (1 / 16) = -2 := by
  sorry

end log_base_4_of_fraction_l722_722471


namespace a6_minus_b6_divisible_by_9_l722_722566

theorem a6_minus_b6_divisible_by_9 {a b : ℤ} (h₁ : a % 3 ≠ 0) (h₂ : b % 3 ≠ 0) : (a ^ 6 - b ^ 6) % 9 = 0 := 
sorry

end a6_minus_b6_divisible_by_9_l722_722566


namespace probability_participation_on_both_days_l722_722492

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end probability_participation_on_both_days_l722_722492


namespace nested_sqrt_eq_l722_722853

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722853


namespace nested_sqrt_eq_l722_722852

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l722_722852


namespace door_height_l722_722217

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l722_722217


namespace count_cube_roots_less_than_15_l722_722124

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l722_722124


namespace number_of_distinct_prime_factors_30_fact_l722_722999

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l722_722999


namespace combined_savings_value_after_one_year_l722_722827

def raise_don : ℝ := 800
def raise_percentage_don : ℝ := 0.06
def raise_wife : ℝ := 840
def raise_percentage_wife : ℝ := 0.07
def tax_rate_don : ℝ := 0.10
def tax_rate_wife : ℝ := 0.12
def savings_investment_rate : ℝ := 0.05
def savings_interest_rate : ℝ := 0.03

theorem combined_savings_value_after_one_year :
  let don_current_income := raise_don / raise_percentage_don in
  let wife_current_income := raise_wife / raise_percentage_wife in
  let don_new_income := don_current_income + raise_don in
  let wife_new_income := wife_current_income + raise_wife in
  let don_post_tax_income := don_new_income * (1 - tax_rate_don) in
  let wife_post_tax_income := wife_new_income * (1 - tax_rate_wife) in
  let combined_post_tax_income := don_post_tax_income + wife_post_tax_income in
  let investment := combined_post_tax_income * savings_investment_rate in
  let savings_after_one_year := investment * (1 + savings_interest_rate) in
  savings_after_one_year = 1237.99 :=
sorry

end combined_savings_value_after_one_year_l722_722827


namespace min_value_expression_l722_722635

theorem min_value_expression (y : Fin 50 → ℝ) (h_pos : ∀ i, y i > 0)
  (h_sum : (Finset.univ.sum (λ i, (y i) ^ 2)) = 1) :
  (Finset.univ.sum (λ i, (y i) / (1 - (y i) ^ 2))) = (3 * Real.sqrt 3) / 2 :=
sorry

end min_value_expression_l722_722635


namespace door_height_l722_722216

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l722_722216


namespace product_of_first_two_terms_l722_722329

theorem product_of_first_two_terms (a_7 : ℕ) (d : ℕ) (a_7_eq : a_7 = 17) (d_eq : d = 2) :
  let a_1 := a_7 - 6 * d
  let a_2 := a_1 + d
  a_1 * a_2 = 35 :=
by
  sorry

end product_of_first_two_terms_l722_722329


namespace continuity_piecewise_function_l722_722263

def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 3 * a * x + 1
  else if x < 0 then 3 * x - b
  else x^2 + 1

theorem continuity_piecewise_function (a b : ℝ) : 
  (3 * a * 3 + 1 = 3^2 + 1) ∧ (0^2 + 1 = 3 * 0 - b) → a + b = 0 := 
by 
  sorry

end continuity_piecewise_function_l722_722263


namespace swap_values_l722_722082

theorem swap_values (a b : ℕ) (h₁ : a = 3) (h₂ : b = 2) : 
  let t := a in
  let a := b in
  let b := t in
  a = 2 ∧ b = 3 :=
by
  intros
  simp_all
  exact And.intro rfl rfl

end swap_values_l722_722082


namespace number_of_prime_factors_30_factorial_l722_722973

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l722_722973


namespace extreme_points_product_l722_722543

noncomputable def f (x a : ℝ) := log x + 2 * x + a / x

noncomputable def g (x a : ℝ) := x * f x a - (a / 2 + 2) * x^2 - x

theorem extreme_points_product (a x1 x2 : ℝ) (h_a : a ∈ ℝ) 
  (h_extreme : ∃ x1 x2, x1 < x2 ∧ ∀ x, g x a = 0 → x = x1 ∨ x = x2) : 
  x1 * x2^2 > real.exp 3 := 
sorry

end extreme_points_product_l722_722543


namespace probability_of_selection_l722_722915

theorem probability_of_selection :
  ∀ (total_students eliminated_students selected_students remaining_students : ℕ),
  total_students = 2008 →
  eliminated_students = 8 →
  remaining_students = total_students - eliminated_students →
  selected_students = 50 →
  (remaining_students / total_students) * (selected_students / remaining_students) = (25 / 1004) :=
begin
  intros,
  sorry
end

end probability_of_selection_l722_722915


namespace find_n_l722_722557

theorem find_n (n : ℝ) (h1 : n > 0) (h2 : (5 - n) / (n - 2) = 2n) : n = 2.5 :=
by
  sorry

end find_n_l722_722557


namespace probability_intersection_l722_722758

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l722_722758


namespace minimum_possible_sum_l722_722553

theorem minimum_possible_sum (X : ℕ → ℕ) (h : ∀ n, 1 ≤ n → n ≤ 12 → X n > 0) : 
  (∑ n in finset.range 12, (-1 : ℤ)^(n+1) * (X (n+1) : ℤ)) = 1 :=
by sorry

end minimum_possible_sum_l722_722553


namespace shorter_leg_of_right_triangle_l722_722587

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l722_722587


namespace inequality_solution_sets_equivalence_l722_722079

theorem inequality_solution_sets_equivalence
  (a b : ℝ)
  (h1 : (∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0)) :
  (∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ bx^2 - 5 * x + a > 0) :=
  sorry

end inequality_solution_sets_equivalence_l722_722079


namespace train_speed_l722_722433

theorem train_speed (distance : ℝ) (time : ℝ) (distance_eq : distance = 270) (time_eq : time = 9)
  : (distance / time) * (3600 / 1000) = 108 :=
by 
  sorry

end train_speed_l722_722433


namespace estimate_turtles_in_pond_l722_722398

-- Definitions of conditions
def turtles_tagged_initial := 80
def turtles_sampled_october := 50
def turtles_tagged_october := 2
def predation_rate := 0.20
def birth_immigration_rate := 0.30

-- Calculation based on conditions
def turtles_present_in_october := turtles_sampled_october * (1 - birth_immigration_rate)

-- Statement of the theorem
theorem estimate_turtles_in_pond : 
  turtles_tagged_october / turtles_present_in_october =
  turtles_tagged_initial / (1400 : ℕ) := by
  sorry

end estimate_turtles_in_pond_l722_722398


namespace sum_geq_n_sqrt2_sub_1_l722_722289

theorem sum_geq_n_sqrt2_sub_1 (n : ℕ) (h : n ≥ 1) :
  (∑ k in Finset.range(n), (1 : ℝ) / (n + k)) ≥ n * (Real.sqrt (2 : ℝ) ^ (1 / n) - 1) :=
sorry

end sum_geq_n_sqrt2_sub_1_l722_722289
