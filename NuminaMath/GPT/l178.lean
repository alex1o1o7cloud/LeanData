import Mathlib
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle.Basic
import Mathlib.Noncomputable
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Independent
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace value_range_l178_178599

-- Define the function y = 1 - 2^x
def func (x : ℝ) : ℝ := 1 - 2^x

-- Define the domain : x ∈ [2, 3]
def domain (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Prove the value range of the function is [-7, -3]
theorem value_range : {y : ℝ | ∃ x : ℝ, domain x ∧ y = func x} = set.Icc (-7 : ℝ) (-3 : ℝ) :=
sorry

end value_range_l178_178599


namespace range_of_a_l178_178851

noncomputable def acute_angle_condition (a : ℝ) : Prop :=
  let M := (-2, 0)
  let N := (0, 2)
  let A := (-1, 1)
  (a > 0) ∧ (∀ P : ℝ × ℝ, (P.1 - a) ^ 2 + P.2 ^ 2 = 2 →
    (dist P A) > 2 * Real.sqrt 2)

theorem range_of_a (a : ℝ) : acute_angle_condition a ↔ a > Real.sqrt 7 - 1 :=
by sorry

end range_of_a_l178_178851


namespace complex_sum_correct_l178_178705

noncomputable def complex_sum := 
  ∑ n in finset.range 51, (complex.I ^ n) * (real.cos ((30 + 90 * n) * real.pi / 180))

theorem complex_sum_correct : complex_sum = (25 * real.sqrt 3 / 2 - 25 * complex.I / 2) := 
  sorry

end complex_sum_correct_l178_178705


namespace complement_set_M_l178_178045

-- Definitions of sets based on given conditions
def universal_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def set_M : Set ℝ := {x | x^2 - x ≤ 0}

-- The proof statement that we need to prove
theorem complement_set_M :
  {x | 1 < x ∧ x ≤ 2} = universal_set \ set_M := by
  sorry

end complement_set_M_l178_178045


namespace perpendicular_line_passing_through_point_l178_178685

-- Let's define the initial conditions
def point := (0, -3 : ℚ := by sorry)

def given_line := ∀ x y : ℚ, 3 * x + 4 * y + 5 = 0

-- Define the perpendicular condition:
def perpendicular_slope (m1 m2 : ℚ) : Prop :=
  m1 * m2 = -1

-- Let's define the perpendicular line equation passing through point (0, -3)
def line_equation (a b c : ℚ) := ∀ x y : ℚ, a * x + b * y + c = 0

theorem perpendicular_line_passing_through_point :
  ∃ a b c : ℚ, line_equation a b c ∧ a * (0 : ℚ) + b * (-3 : ℚ) + c = 0 ∧
                (∃ m, given_line → perpendicular_slope m (4 / 3)) :=
by
  sorry

end perpendicular_line_passing_through_point_l178_178685


namespace puppy_food_cost_l178_178137

theorem puppy_food_cost :
  let puppy_cost : ℕ := 10
  let days_in_week : ℕ := 7
  let total_number_of_weeks : ℕ := 3
  let cups_per_day : ℚ := 1 / 3
  let cups_per_bag : ℚ := 3.5
  let cost_per_bag : ℕ := 2
  let total_days := total_number_of_weeks * days_in_week
  let total_cups := total_days * cups_per_day
  let total_bags := total_cups / cups_per_bag
  let food_cost := total_bags * cost_per_bag
  let total_cost := puppy_cost + food_cost
  total_cost = 14 := by
  sorry

end puppy_food_cost_l178_178137


namespace arithmetic_sqrt_9_l178_178171

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l178_178171


namespace num_ak_divisible_by_11_l178_178106

def a (n : ℕ) : ℕ :=
  -- a helper function that generates the number a_n by concatenating the digits from 1 to n
  -- This definition is simplified since detailed implementation is nontrivial
  -- You would typically concatenate these numbers in a realistic implementation.
  sorry

def g (n : ℕ) : ℤ := 
  -- a helper function that calculates the alternating sum of the digits of a_n
  sorry

theorem num_ak_divisible_by_11 : ∀ k : ℕ, (1 ≤ k ∧ k ≤ 120) → (a k % 11 = 0) ↔ (exists m : ℕ, m ∈ (1..120) ∧ m % 22 = 0) :=
by {
  sorry
}

end num_ak_divisible_by_11_l178_178106


namespace sum_of_coefficients_l178_178059

theorem sum_of_coefficients (b : Fin 7 → ℤ) (x : ℤ) :
    (4 * x - 2)^6 = ∑ i, b i * x^i → ∑ i, b i = 64 :=
by
    intro h
    have h₁ : (4 * 1 - 2)^6 = 64 := by norm_num
    have h₂ : ∑ i, b i * (1 : ℤ)^i = ∑ i, b i := by simp
    rw [← h] at h₁
    rw [h₂] at h₁
    exact h₁

end sum_of_coefficients_l178_178059


namespace determine_x_l178_178722

theorem determine_x (x : ℝ) : (∃ x : ℝ, (x^(3/4) = 5)) → x = 5 * (5^(1/3)) :=
by
  intro h
  cases h with x hx
  sorry

end determine_x_l178_178722


namespace increasing_intervals_range_of_c_l178_178108

-- Given function definition
def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x ^ 2 - 3 * x + 1

-- Part 1: Proving intervals of increasing function
theorem increasing_intervals : 
  (∀ x, f' x > 0 → (x < -1 ∨ x > 3)) → 
  (∀ x, ∀ (I1 : ℝ × ℝ) (I2 : ℝ × ℝ), 
      I1 = (- (1 : ℝ), -∞) → I2 = (3, +∞) →
      ((∃ (y : ℝ), y ∈ I1 ∧ f'(y) > 0) ∧ (∃ (z : ℝ), z ∈ I2 ∧ f'(z) > 0))) :=
sorry

-- Part 2: Proving the range of values for c
theorem range_of_c : 
  (∀ x ∈ Icc 0 4, f x ≤ c) ↔ (c ≥ 1) :=
sorry

end increasing_intervals_range_of_c_l178_178108


namespace joe_error_percentage_l178_178499

-- Definitions corresponding to the conditions in a)
def convert_pounds_to_kg (x : ℝ) : ℝ := (x / 2) * 0.90 
def convert_kg_to_pounds (y : ℝ) : ℝ := y * 2 * 1.10

-- The theorem statement
theorem joe_error_percentage : 
  convert_kg_to_pounds (convert_pounds_to_kg 1) = 0.99 :=
sorry

end joe_error_percentage_l178_178499


namespace line_intersects_parabola_l178_178673

variable {a b y_A y_B y_C : ℝ}
variable (hA : y_A = a^2) (hB : y_B = b^2)
variable (hC : y_C = a * b)

theorem line_intersects_parabola :
  y_C = sqrt (y_A * y_B) :=
by
  rw [hC, hA, hB]
  sorry

end line_intersects_parabola_l178_178673


namespace jack_sugar_final_l178_178864

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l178_178864


namespace total_initial_seashells_l178_178444

-- Definitions for the conditions
def Henry_seashells := 11
def Paul_seashells := 24

noncomputable def Leo_initial_seashells (total_seashells : ℕ) :=
  (total_seashells - (Henry_seashells + Paul_seashells)) * 4 / 3

theorem total_initial_seashells 
  (total_seashells_now : ℕ)
  (leo_shared_fraction : ℕ → ℕ)
  (h : total_seashells_now = 53) : 
  Henry_seashells + Paul_seashells + leo_shared_fraction 53 = 59 :=
by
  let L := Leo_initial_seashells 53
  have L_initial : L = 24 := by sorry
  exact sorry

end total_initial_seashells_l178_178444


namespace trailer_cost_is_120000_l178_178496

-- Definitions of conditions
def house_cost : ℝ := 480000
def monthly_installments : ℕ := 240
def monthly_payment_difference : ℝ := 1500

-- Assume the monthly payment for the trailer is some amount T
variable (T : ℝ)

-- The monthly payment for the house
def house_monthly_payment := T + monthly_payment_difference

-- Total cost for the house loan over 240 months
def total_house_cost := house_monthly_payment * monthly_installments

-- Total cost for the trailer loan over 240 months
def total_trailer_cost := T * monthly_installments

-- Proof statement
theorem trailer_cost_is_120000
    (h : total_house_cost = house_cost) : total_trailer_cost = 120000 :=
by
  -- Given the known conditions state the proof
  sorry

end trailer_cost_is_120000_l178_178496


namespace smallest_difference_l178_178633

-- Define the predicate for 3-digit numbers using specific digits
def is_3_digit_number_using_digits (n : ℕ) (d1 d2 d3 : ℕ) : Prop :=
  n = 100 * d1 + 10 * d2 + d3

-- Define the specific digits we are going to use
def digit_set : Finset ℕ := {2, 3, 4, 6, 7, 8}

-- Define the problem conditions
def valid_numbers (n m : ℕ) : Prop :=
  is_3_digit_number_using_digits n (100) (10) (1) ∧ 
  is_3_digit_number_using_digits m (100) (10) (1) ∧
  {n % 10, (n / 10) % 10, (n / 100) % 10, 
  m % 10, (m / 10) % 10, (m / 100) % 10} = digit_set

-- The theorem statement
theorem smallest_difference (n m : ℕ) (h : valid_numbers n m) : n ≠ m → abs (n - m) = 638 :=
by
  sorry

end smallest_difference_l178_178633


namespace matt_sales_count_l178_178538

theorem matt_sales_count
  (regular_commission_rate : ℝ)
  (increased_commission_rate : ℝ)
  (sale_amount : ℝ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (commission_increase : ℝ)
  (new_avg_commission : ℝ) :
  regular_commission_rate = 0.1 →
  increased_commission_rate = 0.115 →
  sale_amount = 13000 →
  discount_rate = 0.05 →
  sales_tax_rate = 0.08 →
  commission_increase = 150 →
  new_avg_commission = 400 →
  ∃ (N : ℕ), N = 12 :=
begin
  intros hrc hic hsa hd hst hci hac,
  sorry
end

end matt_sales_count_l178_178538


namespace beanie_babies_total_l178_178910

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end beanie_babies_total_l178_178910


namespace nontrivial_solution_exists_l178_178123

theorem nontrivial_solution_exists 
  (a b : ℤ) 
  (h_square_a : ∀ k : ℤ, a ≠ k^2) 
  (h_square_b : ∀ k : ℤ, b ≠ k^2) 
  (h_nontrivial : ∃ (x y z w : ℤ), x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) : 
  ∃ (x y z : ℤ), x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) :=
by
  sorry

end nontrivial_solution_exists_l178_178123


namespace walking_time_proof_l178_178133

-- Define the conditions from the problem
def bus_ride : ℕ := 75
def train_ride : ℕ := 360
def total_trip_time : ℕ := 480

-- Define the walking time as variable
variable (W : ℕ)

-- State the theorem as a Lean statement
theorem walking_time_proof :
  bus_ride + W + 2 * W + train_ride = total_trip_time → W = 15 :=
by
  intros h
  sorry

end walking_time_proof_l178_178133


namespace janet_stuffies_l178_178497

theorem janet_stuffies (x : ℕ) : 
  let total_stuffies : ℕ := x,
      keep_fraction : ℚ := 3 / 7,
      give_away_fraction : ℚ := 4 / 7,
      distribution_ratio : ℕ := 3 + 4 + 2 + 1 + 5,
      janet_ratio : ℕ := 1 in
  janet_ratio * (give_away_fraction * total_stuffies) / distribution_ratio = 4 * x / 105 := 
by
  sorry

end janet_stuffies_l178_178497


namespace greatest_integer_with_gcd_30_eq_5_l178_178258

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l178_178258


namespace parallelepiped_intersection_polygon_has_parallel_sides_l178_178555

open Classical

-- Definitions of the geometric constructs
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
-- A definition for a parallelepiped
structure Parallelepiped (V : Type*) :=
(base : V)
(side_vectors : fin 3 → V)
(property_parallel : ∀ i j : fin 3, i ≠ j → side_vectors i ∥ side_vectors j)

-- Definitions for a plane intersecting a parallelepiped forming a polygon
structure Plane (V : Type*) :=
(normal : V)
(contains_point : V)
property_linear : ∃ p q : V, ∀ x, x ∈ linear_span ⟨p, q⟩

def is_an_intersection_polygon_with_more_than_three_sides
  (π : Plane V) (para : Parallelepiped V) : Prop :=
∃ poly, (∀ v ∈ poly.vertices, v ∈ intersection_of_plane_and_parallelepiped π para) ∧ fintype.card poly.vertices > 3

theorem parallelepiped_intersection_polygon_has_parallel_sides
  (π : Plane V) (para : Parallelepiped V)
  (h : is_an_intersection_polygon_with_more_than_three_sides π para) :
  ∃ sides1 sides2, sides1 ∥ sides2 := sorry

end parallelepiped_intersection_polygon_has_parallel_sides_l178_178555


namespace min_max_f_l178_178198

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f : 
  let min_val := - (3 * Real.pi) / 2 in
  let max_val := (Real.pi / 2) + 2 in
  ∃ x_min ∈ Set.Icc 0 (2 * Real.pi), f x_min = min_val ∧
  ∃ x_max ∈ Set.Icc 0 (2 * Real.pi), f x_max = max_val :=
sorry

end min_max_f_l178_178198


namespace distance_O_to_AB_is_half_CD_l178_178556

variable {O A B C D : Point}
variable {AB CD : Line}
variable (circle : Circle)
variable (O_on_circle : O ∈ circle)
variable (AB_chord : AB ⊆ circle)
variable (CD_chord : CD ⊆ circle)

-- Assume that distance_of_perpendicular is a function that calculates perpendicular distance from a point to a line
-- Assume that length_of is a function that calculates the length of a line

theorem distance_O_to_AB_is_half_CD : distance_of_perpendicular O AB = (length_of CD) / 2 :=
sorry

end distance_O_to_AB_is_half_CD_l178_178556


namespace one_team_beats_another_twice_l178_178700

theorem one_team_beats_another_twice (teams : Fin 99) (home_away : Fin 99 → Fin 98)
  (home_wins : ∀ (t : Fin 99), ∃ h1 h2: Fin 98, h1 ≠ h2 ∧ t.wins(h1) = 1/2 ∧ t.wins(h2) = 1/2)
  (away_wins : ∀ (t : Fin 99), ∃ h1 h2: Fin 98, h1 ≠ h2 ∧ t.wins(h1) = 1/2 ∧ t.wins(h2) = 1/2) : 
  ∃ (a b : Fin 99), a ≠ b ∧ (beats (a b) ∧ beats (a b)) :=
by {
  sorry
}

end one_team_beats_another_twice_l178_178700


namespace last_two_digits_sum_l178_178583

theorem last_two_digits_sum (n : ℕ) (h : n = 2005) :
  (2005 + ∑ k in finset.range (n - 1), (2005 ^ (k + 2))) % 100 = 5 :=
by
  sorry

end last_two_digits_sum_l178_178583


namespace inequality_satisfaction_l178_178582

theorem inequality_satisfaction (x y : ℝ) : 
  y - x < Real.sqrt (x^2) ↔ (y < 0 ∨ y < 2 * x) := by 
sorry

end inequality_satisfaction_l178_178582


namespace arithmetic_square_root_of_9_is_3_l178_178168

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l178_178168


namespace students_answered_both_correctly_l178_178844

theorem students_answered_both_correctly :
  ∀ (total_students set_problem function_problem both_incorrect x : ℕ),
    total_students = 50 → 
    set_problem = 40 →
    function_problem = 31 →
    both_incorrect = 4 →
    x = total_students - both_incorrect - (set_problem + function_problem - total_students) →
    x = 25 :=
by
  intros total_students set_problem function_problem both_incorrect x
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end students_answered_both_correctly_l178_178844


namespace base7_divisibility_rules_2_base7_divisibility_rules_3_l178_178230

def divisible_by_2 (d : Nat) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

def divisible_by_3 (d : Nat) : Prop :=
  d = 0 ∨ d = 3

def last_digit_base7 (n : Nat) : Nat :=
  n % 7

theorem base7_divisibility_rules_2 (n : Nat) :
  (∃ k, n = 2 * k) ↔ divisible_by_2 (last_digit_base7 n) :=
by
  sorry

theorem base7_divisibility_rules_3 (n : Nat) :
  (∃ k, n = 3 * k) ↔ divisible_by_3 (last_digit_base7 n) :=
by
  sorry

end base7_divisibility_rules_2_base7_divisibility_rules_3_l178_178230


namespace alice_profit_l178_178331

-- Define the variables and conditions
def total_bracelets : ℕ := 52
def material_cost : ℝ := 3.00
def bracelets_given_away : ℕ := 8
def sale_price : ℝ := 0.25

-- Calculate the number of bracelets sold
def bracelets_sold : ℕ := total_bracelets - bracelets_given_away

-- Calculate the revenue from selling the bracelets
def revenue : ℝ := bracelets_sold * sale_price

-- Define the profit as revenue minus material cost
def profit : ℝ := revenue - material_cost

-- The statement to prove
theorem alice_profit : profit = 8.00 := 
by
  sorry

end alice_profit_l178_178331


namespace intersection_of_lines_l178_178382

theorem intersection_of_lines :
  ∃ x y : ℚ, (12 * x - 5 * y = 40) ∧ (8 * x + 2 * y = 20) ∧ x = 45 / 16 ∧ y = -5 / 4 :=
by
  use (45 / 16), (-5 / 4)
  split
  -- verification for the first equation
  {
    have hx : 12 * (45 / 16) - 5 * (-5 / 4) = 40,
    {
      calc 12 * (45 / 16) - 5 * (-5 / 4)
          = 12 * 45 / 16 - 5 * (-5 / 4) : by norm_num
          ... = 40                    : by norm_num
    },
    exact hx,
  }
  split
  -- verification for the second equation
  {
    have hy : 8 * (45 / 16) + 2 * (-5 / 4) = 20,
    {
      calc 8 * (45 / 16) + 2 * (-5 / 4)
          = 8 * 45 / 16 + 2 * (-5 / 4) : by norm_num
          ... = 20                   : by norm_num
    },
    exact hy,
  }
  split
  -- verification for just x
  {
    exact rfl,
  }
  -- verification for just y
  {
    exact rfl,
  }
  sorry

end intersection_of_lines_l178_178382


namespace max_profit_price_l178_178621

/-
Conditions:
1. Purchase price is 80 yuan.
2. Initial selling price is 90 yuan with 400 units sold.
3. Sales decrease by 20 units for every 1 yuan increase in price.
-/ 

theorem max_profit_price :
  (∀ (purchase_price initial_selling_price : ℕ) (initial_units : ℕ) (decrease_rate : ℕ),
    purchase_price = 80 →
    initial_selling_price = 90 →
    initial_units = 400 →
    decrease_rate = 20 →
    let profit := (λ (selling_price : ℕ), let x := selling_price - initial_selling_price in 
                                           (10 + x) * (initial_units - decrease_rate * x)) in
    ∀ (optimal_selling_price : ℕ), (optimal_selling_price = 95) → 
    (∀ x, profit optimal_selling_price ≥ profit x)) :=
sorry

end max_profit_price_l178_178621


namespace necessary_not_sufficient_for_decreasing_l178_178206

-- Define the conditions
def func (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≥ f y 

-- Problem statement in Lean 4
theorem necessary_not_sufficient_for_decreasing (a : ℝ) :
  0 < a ∧ a < 2 → ∃ f, f = func a ∧ is_monotonically_decreasing f :=
by
  intros h,
  sorry

end necessary_not_sufficient_for_decreasing_l178_178206


namespace agm_inequality_l178_178552

theorem agm_inequality (a : ℕ → ℝ) (p : ℕ → ℕ) (n : ℕ) (hp : ∀ i, 1 ≤ p i) (ha : ∀ i, 0 < a i) :
  (∏ i in Finset.range n, (a i) ^ (p i)) ≤ 
    ((∑ i in Finset.range n, (p i * (a i))) / (∑ i in Finset.range n, p i)) ^ (∑ i in Finset.range n, p i) :=
by
  sorry

end agm_inequality_l178_178552


namespace savings_percentage_l178_178628

variables (I S : ℝ)
-- Condition 1: Man's income in the first year is I
-- Condition 2: Portion he saves in the first year is S
-- Condition 3: Income increases by 35% in the second year
-- Condition 4: Savings increase by 100% in the second year
-- Condition 5: Expenditure in the first year is I - S
-- Condition 6: Expenditure in the second year is 1.35I - 2S
-- Condition 7: Total expenditure in two years is double his expenditure in the first year

def first_year_expenditure (I S : ℝ) := I - S
def second_year_expenditure (I S : ℝ) := 1.35 * I - 2 * S

theorem savings_percentage :
  (first_year_expenditure I S + second_year_expenditure I S = 2 * first_year_expenditure I S) →
  0.35 * I = S →
  (S / I) * 100 = 35 :=
by
  intro h1 h2
  rw [h2]
  field_simp
  norm_num
  sorry

end savings_percentage_l178_178628


namespace number_and_sum_of_possible_values_of_f2_l178_178891

theorem number_and_sum_of_possible_values_of_f2 (
  f : ℝ → ℝ,
  h : ∀ x y : ℝ, f x * f y - f (x * y) = x * y + x + y
) : let n := 1 in let s := (5 : ℝ) / 2 in n * s = 5 / 2 :=
by
  sorry

end number_and_sum_of_possible_values_of_f2_l178_178891


namespace number_is_square_iff_b_gt_4_l178_178714

-- Definitions and conditions directly appearing from the problem
def is_square_base_b (b n : ℕ) : Prop :=
  let n_base_10 := b^2 + 4*b + 4 in
  ∃ k : ℕ, k^2 = n_base_10

-- The equivalence we aim to prove
theorem number_is_square_iff_b_gt_4 (b : ℕ) : is_square_base_b b (b^2 + 4*b + 4) ↔ b > 4 :=
by
  sorry

end number_is_square_iff_b_gt_4_l178_178714


namespace increasing_functions_l178_178336

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem increasing_functions :
  (∀ x ∈ Iio (0 : ℝ), has_deriv_at (λ x : ℝ, -x^2 / abs x) (1) x) ∧
  (∀ x ∈ Iio (0 : ℝ), has_deriv_at (λ x : ℝ, x + x / abs x) (1) x) :=
by
  sorry

end increasing_functions_l178_178336


namespace probability_of_rain_each_day_l178_178377

noncomputable def probability_of_rain := sorry 

theorem probability_of_rain_each_day:
  let x := probability_of_rain in
  (let f x p₀ := (1 - x) * (1 - x) in   -- this is the formula part from conditions analysis
    p₀ (1 - x) * x + p₀ 0.2 x - x * p₀ 0.2 x + sorry -- simplifying or setting up the conditions
    x + sorry) := 0.2 :
x = (1 / 9)
:= sorry

end probability_of_rain_each_day_l178_178377


namespace sum_mod_9753_to_9756_is_0_l178_178386

theorem sum_mod_9753_to_9756_is_0 :
  (9753 + 9754 + 9755 + 9756) % 13 = 0 :=
by
suffices : (9753 % 13 + 9754 % 13 + 9755 % 13 + 9756 % 13) % 13 = 0 by
  have h := Nat.add_mod 9753 9754 13
  have h' := Nat.add_mod (9753 + 9754) (9755 + 9756) 13
  sorry
sorry

end sum_mod_9753_to_9756_is_0_l178_178386


namespace modulus_conjugate_z_l178_178658

open Complex

theorem modulus_conjugate_z (z : ℂ) (h : (1 - I) / (I * z) = 1) : Complex.abs (conj z) = Real.sqrt 2 := by
  -- skipping the proof
  sorry

end modulus_conjugate_z_l178_178658


namespace f_at_1_f_at_4_f_inequality_range_l178_178164

variable {f : ℝ → ℝ}
variable {x : ℝ}

-- The function f is defined for all x > 0
-- f has property: f(xy) = f(x) + f(y)
axiom f_def : ∀ x y : ℝ, x > 0 → y > 0 → f(x * y) = f(x) + f(y)

-- Given: f(2) = 1
axiom f_at_2 : f 2 = 1

-- Given: f(x) > f(y) if and only if x > y > 0
axiom f_order : ∀ x y : ℝ, x > 0 → y > 0 → f(x) > f(y) ↔ x > y

-- Prove: f(1) = 0
theorem f_at_1 : f 1 = 0 := 
sorry

-- Prove: f(4) = 2
theorem f_at_4 : f 4 = 2 := 
sorry

-- Prove: For 3 < x ≤ 4, f(x) + f(x-3) ≤ 2
theorem f_inequality_range (hx : 3 < x ∧ x ≤ 4) : f(x) + f(x - 3) ≤ 2 :=
sorry

end f_at_1_f_at_4_f_inequality_range_l178_178164


namespace midpoint_trajectory_equation_l178_178021

theorem midpoint_trajectory_equation :
  (x y : ℝ) (hP : ∃ p q : ℝ, p^2 = 4*q ∧ 
    y = (q + 1) / 2 ∧ x = p / 2) (hF : F = (0, 1)) :
  x^2 = 2*y - 1 :=
begin
  sorry
end

end midpoint_trajectory_equation_l178_178021


namespace distance_between_x_intercepts_l178_178675

def line1 (x : ℝ) : ℝ := 4 * x - 12
def line2 (x : ℝ) : ℝ := -6 * x + 68

theorem distance_between_x_intercepts : 
  let intercept1 := 3 in
  let intercept2 := 34 / 3 in
  abs(intercept2 - intercept1) = 25 / 3 :=
by
  sorry

end distance_between_x_intercepts_l178_178675


namespace trig_eqn_solution_l178_178297

open Real

theorem trig_eqn_solution (x : ℝ) (n : ℤ) :
  sin x ≠ 0 →
  cos x ≠ 0 →
  sin x + cos x ≥ 0 →
  (sqrt (1 + tan x) = sin x + cos x) →
  ∃ k : ℤ, (x = k * π + π / 4) ∨ (x = k * π - π / 4) ∨ (x = (2 * k * π + 3 * π / 4)) :=
by
  sorry

end trig_eqn_solution_l178_178297


namespace find_wanderer_in_8th_bar_l178_178478

noncomputable def wanderer_probability : ℚ := 1 / 3

theorem find_wanderer_in_8th_bar
    (total_bars : ℕ)
    (initial_prob_in_any_bar : ℚ)
    (prob_not_in_specific_bar : ℚ)
    (prob_not_in_first_seven : ℚ)
    (posterior_prob : ℚ)
    (h1 : total_bars = 8)
    (h2 : initial_prob_in_any_bar = 4 / 5)
    (h3 : prob_not_in_specific_bar = 1 - (initial_prob_in_any_bar / total_bars))
    (h4 : prob_not_in_first_seven = prob_not_in_specific_bar ^ 7)
    (h5 : posterior_prob = initial_prob_in_any_bar / prob_not_in_first_seven) :
    posterior_prob = wanderer_probability := 
sorry

end find_wanderer_in_8th_bar_l178_178478


namespace sequence_general_term_l178_178006

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n / (a n + 3)) :
  ∀ n : ℕ, n > 0 → a n = 3 / (n + 2) := 
by 
  sorry

end sequence_general_term_l178_178006


namespace max_gcd_b_n_b_n_plus_1_l178_178904

noncomputable def b (n : ℕ) : ℚ := (2 ^ n - 1) / 3

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, Int.gcd (b n).num (b (n + 1)).num = 1 :=
by
  sorry

end max_gcd_b_n_b_n_plus_1_l178_178904


namespace find_area_of_triangle_DEF_l178_178858

noncomputable def area_of_triangle_DEF : Prop :=
  ∃ A B C D E F : ℝ,
  DE = 8 ∧ DF = 17 ∧
  DM = (1 / 2) * EF ∧
  let area := √(23.5 * 15.5 * 6.5 * 1.5) in
  area ≈ 59.905

theorem find_area_of_triangle_DEF (A B C D E F : ℝ) (DE : ℝ) (DF : ℝ) (DM : ℝ) (EF : ℝ) (area : ℝ) :
  DE = 8 ∧ DF = 17 ∧ DM = (1 / 2) * EF ∧ area = √(23.5 * 15.5 * 6.5 * 1.5) →
  area ≈ 59.905 :=
by
    sorry

end find_area_of_triangle_DEF_l178_178858


namespace count_multiples_of_4_l178_178447

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l178_178447


namespace sum_of_lowest_three_l178_178159

noncomputable def mean (scores : List ℕ) : ℕ := (scores.sum) / scores.length

theorem sum_of_lowest_three
  (scores : List ℕ)
  (h_length : scores.length = 7)
  (h_mean : mean scores = 85)
  (h_median : scores.nth (scores.length / 2) = some 88)
  (h_modes : ∀ x ∈ scores, x = 82 ∨ x = 90 → ∃! k, k ≥ 2 ∧ List.count scores x = k)
  :
  scores.take 3.sum = 237 :=
sorry

end sum_of_lowest_three_l178_178159


namespace factor_exists_l178_178235

def is_four_digit_num (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def formed_from_digits (n : ℕ) (digits : list ℕ) : Prop :=
  let digit_list := [((n % 10000) / 1000), ((n % 1000) / 100), ((n % 100) / 10), (n % 10)]
  in list.sort (≤) digit_list = list.sort (≤) digits

theorem factor_exists : ∃ m, formed_from_digits m [3, 4, 6, 8] ∧ 4386 * 2 = m :=
by
  use 8772
  split
  sorry
  rfl

#check factor_exists

end factor_exists_l178_178235


namespace fraction_subtraction_l178_178162

theorem fraction_subtraction (x : ℚ) : x - (1/5 : ℚ) = (3/5 : ℚ) → x = (4/5 : ℚ) :=
by
  sorry

end fraction_subtraction_l178_178162


namespace initial_apples_count_l178_178229

theorem initial_apples_count (a b : ℕ) (h₁ : b = 13) (h₂ : b = a + 5) : a = 8 :=
by
  sorry

end initial_apples_count_l178_178229


namespace log_sum_inequality_no_c_exists_l178_178128

noncomputable def geometric_sequence (a1 q: ℝ) (n: ℕ) : ℝ :=
  a1 * (q ^ n)

def sum_first_n_terms (a1 q: ℝ) (n: ℕ) : ℝ :=
  if q = 1 then (n + 1) * a1
  else a1 * (1 - q^(n+1)) / (1 - q)

theorem log_sum_inequality (a1 q: ℝ) (n: ℕ) (h_a1: a1 > 0) (h_q: q > 0) :
  (Real.log (sum_first_n_terms a1 q n) + Real.log (sum_first_n_terms a1 q (n+2))) / 2 < Real.log (sum_first_n_terms a1 q (n+1)) :=
sorry

theorem no_c_exists (a1 q: ℝ) (n: ℕ) (h_a1: a1 > 0) (h_q: q > 0) :
  ¬ ∃ (c: ℝ), c > 0 ∧ (Real.log (sum_first_n_terms a1 q n - c) + Real.log (sum_first_n_terms a1 q (n+2) - c)) / 2 = Real.log (sum_first_n_terms a1 q (n+1) - c) :=
  sorry

end log_sum_inequality_no_c_exists_l178_178128


namespace tabby_l178_178165

theorem tabby's_running_speed :
  ∃ R : ℝ, R = 8 ∧ (1 + R) / 2 = 4.5 :=
by
  use 8
  split
  · rfl
  · norm_num
  sorry

end tabby_l178_178165


namespace binomial_sum_fraction_l178_178770

theorem binomial_sum_fraction (a : ℕ → ℝ) :
  ( (a 0 + a 2 + a 4 + ... + a 2014) / (a 1 + a 3 + a 5 + ... + a 2015) = (1 + 3^2015) / (1 - 3^2015) ) :=
begin
  sorry
end

end binomial_sum_fraction_l178_178770


namespace greatest_int_less_than_200_gcd_30_is_5_l178_178244

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l178_178244


namespace power_function_value_l178_178786

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_value (α : ℝ) (h : 2 ^ α = (Real.sqrt 2) / 2) : f 4 α = 1 / 2 := 
by 
  sorry

end power_function_value_l178_178786


namespace monotonic_decreasing_interval_l178_178204

noncomputable def function_y (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def derivative_y (x : ℝ) : ℝ := x - (1 / x)

theorem monotonic_decreasing_interval :
  { x : ℝ | 0 < x ∧ derivative_y x ≤ 0 } = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  -- Proof goes here
  sorry

end monotonic_decreasing_interval_l178_178204


namespace ellipse_condition_l178_178373

theorem ellipse_condition (m : ℝ) :
  (\(-4 < m \wedge m < 16 \wedge m ≠ 6\)) → (∀ x y : ℝ, \( \frac{x^2}{16-m} + \frac{y^2}{m+4} = 1 \)) :=
sorry

end ellipse_condition_l178_178373


namespace arithmetic_sqrt_9_l178_178173

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l178_178173


namespace max_area_when_theta_eq_pi_l178_178424

variable {r θ : ℝ}

def circumference_eq (r θ : ℝ) : Prop := r * θ + 2 * r = 4

def area (r θ : ℝ) : ℝ := (1 / 2) * r^2 * θ

theorem max_area_when_theta_eq_pi (h : circumference_eq r θ) : θ = π := by
  sorry

end max_area_when_theta_eq_pi_l178_178424


namespace find_unique_function_l178_178381

theorem find_unique_function (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_unique_function_l178_178381


namespace min_shots_to_guarantee_hit_l178_178961

-- Define the grid size and cruiser size.
def grid_size : ℕ := 4
def cruiser_length : ℕ := 3

-- Define a predicate to state that a certain number of shots guarantee a hit.
def guarantees_hit (shots : ℕ) : Prop :=
  ∀ (cruiser_position : (ℕ × ℕ)), cruiser_position.1 < grid_size ∧ cruiser_position.2 < grid_size →
  ∃ (shot_positions : list (ℕ × ℕ)), shot_positions.length = shots ∧
  ∃ (cruiser_position : (ℕ × ℕ)), cruiser_position ∈ shot_positions

-- The theorem stating the minimum number of shots required.
theorem min_shots_to_guarantee_hit : ∃ (shots : ℕ), guarantees_hit shots ∧ shots = 4 := by
  sorry

end min_shots_to_guarantee_hit_l178_178961


namespace Fred_had_3_dollars_last_week_l178_178512

-- Definitions based on given conditions
def Fred_now : ℕ := 112
def Jason_now : ℕ := 63
def Jason_earned : ℕ := 60
def Jason_last_week : ℕ := 3

-- Goal: prove that Fred_last_week is 3
theorem Fred_had_3_dollars_last_week :
  ∃ (Fred_last_week : ℕ), Fred_last_week = 3 :=
begin
  -- Given Conditions
  let Jason_last_week := Jason_now - Jason_earned,
  have h1 : Jason_last_week = 3 := by simp [Jason_last_week, Jason_now, Jason_earned],

  let Fred_earned := Fred_now - Jason_last_week,
  have h2 : Fred_earned = 109 := by simp [Fred_earned, Fred_now, h1],

  let Fred_last_week := Fred_now - Fred_earned,
  have h3 : Fred_last_week = 3 := by simp [Fred_last_week, Fred_now, h2],

  -- Existential claim
  use Fred_last_week,
  exact h3,
end

end Fred_had_3_dollars_last_week_l178_178512


namespace lcm_of_2_3_5_l178_178276

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def lcm (a b : ℕ) : ℕ :=
  if a = 0 ∨ b = 0 then 0
  else a * (b / Nat.gcd a b)

def lcm_of_set (s : Set ℕ) : ℕ :=
  if h : s.Nonempty then
    Set.toFinset s h |>.fold lcm 1
  else
    1

theorem lcm_of_2_3_5 : lcm_of_set {2, 3, 5} = 30 :=
  sorry

end lcm_of_2_3_5_l178_178276


namespace count_even_digits_in_base_5_of_567_l178_178739

def is_even (n : ℕ) : Bool := n % 2 = 0

def base_5_representation (n : ℕ) : List ℕ :=
  if h : n > 0 then
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc else loop (n / 5) ((n % 5) :: acc)
    loop n []
  else [0]

def count_even_digits_in_base_5 (n : ℕ) : ℕ :=
  (base_5_representation n).filter is_even |>.length

theorem count_even_digits_in_base_5_of_567 :
  count_even_digits_in_base_5 567 = 2 := by
  sorry

end count_even_digits_in_base_5_of_567_l178_178739


namespace venus_speed_mph_l178_178236

theorem venus_speed_mph (speed_mps : ℝ) (seconds_per_hour : ℝ) (mph : ℝ) 
  (h1 : speed_mps = 21.9) 
  (h2 : seconds_per_hour = 3600)
  (h3 : mph = speed_mps * seconds_per_hour) : 
  mph = 78840 := 
  by 
  sorry

end venus_speed_mph_l178_178236


namespace jack_sugar_amount_l178_178869

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l178_178869


namespace common_factor_of_right_triangle_l178_178084

theorem common_factor_of_right_triangle (d : ℝ) 
  (h_triangle : (2*d)^2 + (4*d)^2 = (5*d)^2) 
  (h_side : 2*d = 45 ∨ 4*d = 45 ∨ 5*d = 45) : 
  d = 9 :=
sorry

end common_factor_of_right_triangle_l178_178084


namespace combination_of_6_choose_3_equals_20_l178_178141

theorem combination_of_6_choose_3_equals_20 :
  nat.choose 6 3 = 20 :=
sorry

end combination_of_6_choose_3_equals_20_l178_178141


namespace min_positive_period_of_f_f_monotonically_decreasing_intervals_min_value_in_interval_l178_178792

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x - real.sqrt 3 * cos x) + real.sqrt 3

theorem min_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = real.pi := by
  sorry

theorem f_monotonically_decreasing_intervals : 
  ∀ k : ℤ, ∀ x : ℝ, k * real.pi + 5 * real.pi / 12 ≤ x → x ≤ k * real.pi + 11 * real.pi / 12 → 
  f' x < 0 := by
  sorry

theorem min_value_in_interval : 
  ∃ x : ℝ, x ∈ set.Icc (real.pi / 2) real.pi ∧ f x = -2 ∧ x = 11 * real.pi / 12 := by
  sorry

end min_positive_period_of_f_f_monotonically_decreasing_intervals_min_value_in_interval_l178_178792


namespace graph_of_abs_g_l178_178581

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 4 then x - 2
  else 0

noncomputable def abs_g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then -(x + 3)
  else if -3 < x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 2 then -(x - 2)
  else if 2 < x ∧ x ≤ 4 then x - 2
  else 0

theorem graph_of_abs_g :
  ∀ x : ℝ, abs_g x = |g x| :=
by
  sorry

end graph_of_abs_g_l178_178581


namespace bonus_tasks_l178_178915

-- Definition for earnings without bonus
def earnings_without_bonus (tasks : ℕ) : ℕ := tasks * 2

-- Definition for calculating the total bonus received
def total_bonus (tasks : ℕ) (earnings : ℕ) : ℕ := earnings - earnings_without_bonus tasks

-- Definition for the number of bonuses received given the total bonus and a single bonus amount
def number_of_bonuses (total_bonus : ℕ) (bonus_amount : ℕ) : ℕ := total_bonus / bonus_amount

-- The theorem we want to prove
theorem bonus_tasks (tasks : ℕ) (earnings : ℕ) (bonus_amount : ℕ) (bonus_tasks : ℕ) :
  earnings = 78 →
  tasks = 30 →
  bonus_amount = 6 →
  bonus_tasks = tasks / (number_of_bonuses (total_bonus tasks earnings) bonus_amount) →
  bonus_tasks = 10 :=
by
  intros h_earnings h_tasks h_bonus_amount h_bonus_tasks
  sorry

end bonus_tasks_l178_178915


namespace area_quadrilateral_WXYZ_l178_178088

open Real

variables (P Q R S A B E F W X Y Z : ℝ)
variables (midpoint : ℝ → ℝ → ℝ)

-- Conditions
def rectanglePQRS (P Q R S : ℝ) : Prop := 
  true   -- Placeholder, assume setup of a rectangle PQRS with points P, Q, R, S.

def bisect (A B : ℝ) (P Q : ℝ) : Prop := 
  A = midpoint P Q ∧ B = midpoint P Q

def trisect (E F : ℝ) (S R : ℝ) : Prop :=
  E = (2 * S + R) / 3 ∧ F = (S + 2 * R) / 3

def midpoint_def (W X Y Z A B E F : ℝ) (P Q S R : ℝ) : Prop :=
  W = midpoint P A ∧
  X = midpoint Q B ∧
  Y = midpoint S F ∧
  Z = midpoint R E

def PE_PB_equal (PE PB : ℝ) : Prop := 
  PE = 8 ∧ PE = PB

-- Asserting midpoints formation for quadrilateral WXYZ
def quadrilateral_midpoints (W X Y Z : ℝ) : Prop := 
  true  -- Placeholder to ensure W, X, Y, Z are set correctly as midpoints.

-- Proof goal
theorem area_quadrilateral_WXYZ : 
  ∀ P Q R S A B E F W X Y Z : ℝ, 
  rectanglePQRS P Q R S → 
  bisect A B P Q → 
  trisect E F S R → 
  midpoint_def W X Y Z A B E F P Q S R → 
  PE_PB_equal 8 8 → 
  quadrilateral_midpoints W X Y Z → 
  area_quadrilateral W X Y Z = 32 / 9 :=
by sorry   -- Proof to be completed.

end area_quadrilateral_WXYZ_l178_178088


namespace intersection_A_complement_B_l178_178041

open Set

noncomputable def A : Set ℝ := {2, 3, 4, 5, 6}
noncomputable def B : Set ℝ := {x | x^2 - 8 * x + 12 >= 0}
noncomputable def complement_B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem intersection_A_complement_B :
  A ∩ complement_B = {3, 4, 5} :=
sorry

end intersection_A_complement_B_l178_178041


namespace jack_sugar_l178_178867

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l178_178867


namespace part1_part2_l178_178793

-- Given the conditions
variables (a b : ℝ) (f : ℝ → ℝ)
hypothesis h1 : 0 < a ∧ 0 < b
hypothesis h2 : ∀ x, f x = a * x - b * x^2
hypothesis h3 : ∀ x, f x ≤ 1

-- Part (1) theorem
theorem part1 : a ≤ 2 * Real.sqrt b :=
sorry

-- Additional hypothesis for part (2)
hypothesis h4 : 1 < b
hypothesis h5 : ∀ x ∈ set.Icc 0 1, abs (f x) ≤ 1

-- Part (2) theorem
theorem part2 : b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b :=
sorry

end part1_part2_l178_178793


namespace king_can_ensure_connectivity_l178_178853

-- Define the conditions as a directed graph with 100 vertices
def Graph : Type := Fin 100 → Fin 100 → Prop

-- State the initial conditions
variable (G : Graph)

-- G is a directed graph with 100 vertices where each pair is connected by one-way edge
axiom edge_unique : ∀ (u v : Fin 100), G u v ∨ G v u
axiom not_every_connects : ¬ ∀ (u v : Fin 100), ∃ p : List (Fin 100), List.Chain' G p ∧ p.head = u ∧ p.last = v 

-- The theorem that we want to prove
theorem king_can_ensure_connectivity :
  ∃ v : Fin 100, ∀ u w : Fin 100, ∃ p : List (Fin 100), List.Chain' (λ x y, G x y ∨ G y x) p ∧ p.head = u ∧ p.last = w :=
begin
  sorry
end

end king_can_ensure_connectivity_l178_178853


namespace traci_drive_distance_after_second_stop_l178_178925

theorem traci_drive_distance_after_second_stop (total_distance : ℕ) 
  (h1 : total_distance = 600)
  (distance_first_stop : ℕ) (h2 : distance_first_stop = total_distance / 3)
  (remaining_after_first_stop : ℕ) (h3 : remaining_after_first_stop = total_distance - distance_first_stop)
  (distance_second_stop : ℕ) (h4 : distance_second_stop = remaining_after_first_stop / 4)
  (remaining_after_second_stop : ℕ) (h5 : remaining_after_second_stop = remaining_after_first_stop - distance_second_stop) :
  remaining_after_second_stop = 300 := 
begin
  sorry
end

end traci_drive_distance_after_second_stop_l178_178925


namespace framed_painting_ratio_l178_178321

theorem framed_painting_ratio (x : ℝ) (h : (15 + 2 * x) * (30 + 4 * x) = 900) : (15 + 2 * x) / (30 + 4 * x) = 1 / 2 :=
by
  sorry

end framed_painting_ratio_l178_178321


namespace max_distance_to_line_l_l178_178972

noncomputable def parametric_C_x (φ : ℝ) : ℝ := 1 + sqrt 2 * cos φ
noncomputable def parametric_C_y (φ : ℝ) : ℝ := 1 - sqrt 2 * sin φ

def polar_line_l (ρ θ : ℝ) : Prop := 2 * ρ * cos θ + 2 * ρ * sin θ - 1 = 0

def ordinary_curve_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2
def ordinary_line_l (x y : ℝ) : Prop := 2 * x + 2 * y - 1 = 0

theorem max_distance_to_line_l (x y : ℝ) 
  (hC : ordinary_curve_C x y) (hl : ordinary_line_l x y) : 
  ∃ d : ℝ, d = (7 * sqrt 2) / 4 :=
sorry

end max_distance_to_line_l_l178_178972


namespace find_a_l178_178046

variable (U : Set ℝ) (A : Set ℝ) (a : ℝ)

theorem find_a (hU_def : U = {2, 3, a^2 - a - 1})
               (hA_def : A = {2, 3})
               (h_compl : U \ A = {1}) :
  a = -1 ∨ a = 2 := 
sorry

end find_a_l178_178046


namespace volleyball_team_lineup_l178_178119

theorem volleyball_team_lineup (n : ℕ) :
  (n! * n!) * 2^n = (n ! ^ 2) * 2^n := 
by
  sorry

end volleyball_team_lineup_l178_178119


namespace min_value_f_on_interval_l178_178989

def f (x : ℝ) : ℝ := x * (3 - x^2)

theorem min_value_f_on_interval : ∃ m, m = 0 ∧ ∀ x ∈ set.Icc (0 : ℝ) (real.sqrt 2), f x ≥ m := by
  sorry

end min_value_f_on_interval_l178_178989


namespace repeating_decimal_exceeds_decimal_representation_l178_178350

noncomputable def repeating_decimal : ℚ := 71 / 99
def decimal_representation : ℚ := 71 / 100

theorem repeating_decimal_exceeds_decimal_representation :
  repeating_decimal - decimal_representation = 71 / 9900 := by
  sorry

end repeating_decimal_exceeds_decimal_representation_l178_178350


namespace set_B_representation_l178_178752

def A := {-2, 2, 3, 4}

def B := {x | ∃ t ∈ A, x = t^2}

theorem set_B_representation : B = {4, 9, 16} :=
sorry

end set_B_representation_l178_178752


namespace remaining_sample_id_l178_178653

theorem remaining_sample_id (total_students : ℕ) (sample_size : ℕ) (known_sample_ids : set ℕ) (sampling_interval : ℕ)
  (h_total : total_students = 44)
  (h_sample_size : sample_size = 4)
  (h_known_ids : known_sample_ids = {6, 28, 39})
  (h_interval : sampling_interval = total_students / sample_size) :
  17 ∈ {6, 28, 39, 17} :=
by
  -- given conditions
  rw [h_total, h_sample_size, h_interval] at * 
  -- we should prove that 17 is in the set
  simp only [Finset.mem_cons] 
  -- here we use that 6 + 11 = 17 (this step should correspond to appropriate conditions and definitions)
  sorry

end remaining_sample_id_l178_178653


namespace sufficient_not_necessary_l178_178302

-- Definitions based on the conditions
def f1 (x y : ℝ) : Prop := x^2 + y^2 = 0
def f2 (x y : ℝ) : Prop := x * y = 0

-- The theorem we need to prove
theorem sufficient_not_necessary (x y : ℝ) : f1 x y → f2 x y ∧ ¬ (f2 x y → f1 x y) := 
by sorry

end sufficient_not_necessary_l178_178302


namespace collinear_PQS_l178_178934

variables (A B C D S P Q : Type*) [Circle A B C D] [Tangent SA S] [Tangent SD S]
variables [Line AB P] [Line CD P] [Line AC Q] [Line BD Q]

theorem collinear_PQS
  (h1 : OnCircle A B C D)
  (h2 : Tangent SA S)
  (h3 : Tangent SD S)
  (h4 : IntersectsLine AB CD P)
  (h5 : IntersectsLine AC BD Q) :
  Collinear P Q S :=
sorry

end collinear_PQS_l178_178934


namespace perimeter_of_trapezoid_l178_178856

theorem perimeter_of_trapezoid 
  (AD BC AB : ℝ) 
  (alpha : ℝ)
  (h1 : AD = 90)
  (h2 : BC = 40)
  (h3 : AB = 30)
  (h4 : alpha = Real.pi / 4) -- angle 45 degrees in radians
  (h5 : parallel AD BC) -- assume we have a definition for parallel lines
  : AD + BC + CD + AB = 210 :=
sorry

end perimeter_of_trapezoid_l178_178856


namespace one_div_a_add_one_div_b_eq_one_l178_178456

noncomputable def a : ℝ := Real.log 10 / Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 5

theorem one_div_a_add_one_div_b_eq_one (h₁ : 2 ^ a = 10) (h₂ : 5 ^ b = 10) :
  (1 / a) + (1 / b) = 1 :=
sorry

end one_div_a_add_one_div_b_eq_one_l178_178456


namespace solve_for_x_l178_178966

theorem solve_for_x : ∀ x : ℝ, (81 ^ (2 * x) = 27 ^ (3 * x - 4)) → x = 12 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l178_178966


namespace pizza_fraction_after_six_trips_l178_178624

noncomputable def total_pizza_eaten (n : ℕ) : ℚ :=
if n = 0 then 0 
else (1 / 3) + (1 / 3 * ((1 / 2) ^ n - 1))*(1 / 2)

theorem pizza_fraction_after_six_trips :
  total_pizza_eaten 6 = 21 / 32 :=
by sorry

end pizza_fraction_after_six_trips_l178_178624


namespace identify_incorrect_statement_l178_178085

-- Define each statement as a proposition in the conditions
def statement_A := "It is possible for a proof to be considered valid even if it utilizes statements that have not been formally proved within that proof."
def statement_B := "A proof can maintain its validity despite adapting different sequences in proving various interdependent propositions."
def statement_C := "Definitions in a proof must strictly precede their use in arguments to maintain logical consistency."
def statement_D := "Valid reasoning can lead to a true conclusion even if the premises include an untrue assertion, as long as it is not used in deriving the conclusion."
def statement_E := "Proof by contradiction is only applicable when attempting to disprove a single false assumption that leads directly to a logical contradiction."

-- Define the main theorem to prove (D) is the incorrect statement
theorem identify_incorrect_statement :
  ∀ (A B C D E : Prop),
    (A = statement_A) → 
    (B = statement_B) → 
    (C = statement_C) → 
    (D = statement_D) → 
    (E = statement_E) → 
    ¬ D := 
  sorry

end identify_incorrect_statement_l178_178085


namespace find_matrix_l178_178383

noncomputable def vec_cross (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.2 * b.2.1 - a.2.1 * b.2.2, 
 a.2.0 * b.2.2 - a.2.2 * b.2.0,
 a.2.1 * b.2.0 - a.2.0 * b.2.1)

noncomputable def mat_mult (M : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) 
  (v : ℝ × ℝ × ℝ) : ℝ := 
(M.1.1 * v.1 + M.1.2 * v.2.0 + M.1.3 * v.2.1,  
 M.2.0 * v.1 + M.2.1 * v.2.0 + M.2.2 * v.2.1,
 M.3.0 * v.1 + M.3.1 * v.2.0 + M.3.2 * v.2.1)

theorem find_matrix (M : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) 
  (v : ℝ × ℝ × ℝ) :
  mat_mult M v = 3 * vec_cross (3, 4, -2) v :=
  M = ((0, 6, 12), 
       (-6, 0, -9), 
       (12, -9, 0)) := sorry

end find_matrix_l178_178383


namespace distances_not_equal_l178_178921

-- Define the segment and coordinates
variable {A B : ℝ}
variable {P : Fin 45 → ℝ}

-- Distance function
def distance (x y : ℝ) := abs (x - y)

-- Segment AB (assuming length AB = 1)
axiom ab_length : distance A B = 1

-- 45 points outside the segment AB
axiom points_outside_segment : ∀ i : Fin 45, distance P i A > 1 ∨ distance P i B > 1

theorem distances_not_equal :
  (∑ i, distance (P i) A) ≠ (∑ i, distance (P i) B) :=
sorry

end distances_not_equal_l178_178921


namespace friends_popcorn_l178_178601

theorem friends_popcorn (pieces_per_serving : ℕ) (jared_count : ℕ) (total_servings : ℕ) (jared_friends : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_count = 90)
  (h3 : total_servings = 9)
  (h4 : jared_friends = 3) :
  (total_servings * pieces_per_serving - jared_count) / jared_friends = 60 := by
  sorry

end friends_popcorn_l178_178601


namespace triangle_angles_l178_178988

-- Define the main problem conditions and the proof statement.

theorem triangle_angles (A B C H M : Point) (angle_AMH : ℝ) (BC : ℝ) 
  (is_median_AM : IsMedian A B C M)
  (is_angle_AMH : angle_AMH = 40) 
  (median_length : distance A M = BC / 2) :
  (angle A B C = 90 ∧ angle B A C = 25 ∧ angle B C A = 65) :=
by
  sorry

end triangle_angles_l178_178988


namespace staircase_perimeter_l178_178092

-- Define the conditions and the region's properties
def condition_1 (region : Type) [topological_space region] : Prop :=
  ∀ (angle : region), is_right_angle angle

def condition_2 : ℕ := 12

def tick_mark_length : ℝ := 1

def region_area : ℝ := 89

def rectangle_length : ℝ := 12

-- Define the perimeter function based on those conditions
def perimeter (length height tick_mark_count mark_length : ℝ) : ℝ :=
  length + height + tick_mark_count * mark_length

-- Formalize the problem statement in Lean
theorem staircase_perimeter :
  ∀ (length height : ℝ) (tick_mark_count : ℕ),
    condition_2 = tick_mark_count →
    tick_mark_length = mark_length →
    (rectangle_length * height - tick_mark_count) = region_area →
    perimeter length height tick_mark_count tick_mark_length = 43 :=
by
  intros
  -- Use the known conditions and given equation to find perimeter
  sorry

end staircase_perimeter_l178_178092


namespace difference_of_roots_l178_178380

theorem difference_of_roots (r1 r2 : ℝ) 
    (h_eq : ∀ x : ℝ, x^2 - 9 * x + 4 = 0 ↔ x = r1 ∨ x = r2) : 
    abs (r1 - r2) = Real.sqrt 65 := 
sorry

end difference_of_roots_l178_178380


namespace surface_area_of_cube_l178_178783

theorem surface_area_of_cube (side_length : ℝ) (h1 : ∀ (face : ℕ), 1 ≤ face ∧ face ≤ 3 → square_face side_length) :
  side_length = 2 → surface_area_of_cube 2 = 24 := by
sorry

def square_face (side_length : ℝ) : Prop := ∃ x y : ℝ, x = side_length ∧ y = side_length
def surface_area_of_cube (side_length : ℝ) : ℝ := 6 * side_length ^ 2

end surface_area_of_cube_l178_178783


namespace exponent_addition_l178_178240

theorem exponent_addition : ((7:ℝ)⁻³)^0 + (7^0)^4 = 2 := by
  have h1 : (7:ℝ)⁻³ ^ 0 = 1 := by 
    rw [pow_zero, inv_pow, pow_neg, zero_pow]
    exact zero_lt_one
  have h2 : (7^0:ℝ)^4 = 1 := by 
    rw [pow_zero, one_pow, pow_one]
  rw [h1, h2]
  norm_num
  done

end exponent_addition_l178_178240


namespace profit_function_max_profit_l178_178559

-- Definitions
def m (x : ℝ) : ℝ := 12 - 18 / (2 * x + 1)
def n (x : ℝ) : ℝ := 9 + 9 / m x
def profit (x : ℝ) : ℝ := (m x) * (n x) - 8 * (m x) - x

-- Theorem statements
theorem profit_function (x : ℝ) : profit x = 21 - x - 18 / (2 * x + 1) := by
  sorry

theorem max_profit : ∃ x, x = 2.5 ∧ profit x = 15.5 := by
  sorry

end profit_function_max_profit_l178_178559


namespace coefficient_x2_in_f_prime_expansion_l178_178126

-- Defining the function f(x) and its derivative
noncomputable def f (x : ℝ) : ℝ := (1 - 2 * x)^10
noncomputable def f_prime (x : ℝ) : ℝ := -20 * (1 - 2 * x)^9

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The proof statement
theorem coefficient_x2_in_f_prime_expansion :
  ∃ c : ℝ, c = -2880 ∧
    let general_term (r : ℕ) := -20 * (binom 9 r : ℝ) * (-2)^r in
    general_term 2 = (-2880 : ℝ) * (x^2 : ℝ) :=
sorry


end coefficient_x2_in_f_prime_expansion_l178_178126


namespace number_of_five_digit_numbers_with_one_odd_digit_l178_178806

def odd_digits : List ℕ := [1, 3, 5, 7, 9]
def even_digits : List ℕ := [0, 2, 4, 6, 8]

def five_digit_numbers_with_one_odd_digit : ℕ :=
  let num_1st_position := odd_digits.length * even_digits.length ^ 4
  let num_other_positions := 4 * odd_digits.length * (even_digits.length - 1) * (even_digits.length ^ 3)
  num_1st_position + num_other_positions

theorem number_of_five_digit_numbers_with_one_odd_digit :
  five_digit_numbers_with_one_odd_digit = 10625 :=
by
  sorry

end number_of_five_digit_numbers_with_one_odd_digit_l178_178806


namespace exponent_rule_l178_178243

theorem exponent_rule :
  (7 ^ (-3)) ^ 0 + (7 ^ 0) ^ 4 = 2 :=
by
  sorry

end exponent_rule_l178_178243


namespace probability_interval_chebyshev_l178_178584

theorem probability_interval_chebyshev (X : ℝ → ℝ) (a : ℝ) (σ² : ℝ) :
  (∀ x, 49.5 ≤ X x ≤ 50.5) →
  (a = 50) →
  (σ² = 0.1) →
  (∀ x, P(λ y, 49.5 ≤ y ∧ y ≤ 50.5) ≥ 0.6) :=
begin
  sorry
end

end probability_interval_chebyshev_l178_178584


namespace fibonacci_divisibility_l178_178938

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

theorem fibonacci_divisibility (m n : ℕ) : (fib m ∣ fib n) ↔ (m ∣ n) :=
sorry

end fibonacci_divisibility_l178_178938


namespace parabola_x0_range_l178_178525

theorem parabola_x0_range {x_0 y_0 : ℝ} (h1 : y_0^2 = 8 * x_0) (h2 : abs(x_0 - 2) > 4) : x_0 > 2 :=
sorry

end parabola_x0_range_l178_178525


namespace ratio_sum_equals_four_l178_178409

namespace TriangleProblem

variables {A B C D E F D' E' F': Point}
variables [acute_triangle : AcuteTriangle A B C]
variables [isosceles_triangle_DAC : IsoscelesTriangle D A C (2 * angle A B C)]
variables [isosceles_triangle_EAB : IsoscelesTriangle E A B (2 * angle B A C)]
variables [isosceles_triangle_FBC : IsoscelesTriangle F B C (2 * angle C B A)]
variables [intersection_D' : Intersection (line D B) (line E F) D']
variables [intersection_E' : Intersection (line E C) (line D F) E']
variables [intersection_F' : Intersection (line F A) (line D E) F']

theorem ratio_sum_equals_four :
  (DB / DD') + (EC / EE') + (FA / FF') = 4 :=
sorry

end TriangleProblem

end ratio_sum_equals_four_l178_178409


namespace MP_eq_NQ_l178_178612

-- Define the circles and their intersection points
variable (C1 C2 : Type) [Circle C1] [Circle C2]
variable (A B : Point)
variable (intersect : intersect_points C1 C2 = [A, B])

-- Tangents at point A intersecting the other circle
variable (M N : Point)
variable (tangent_A_M : tangent_point C1 A = M)
variable (tangent_A_N : tangent_point C2 A = N)

-- Lines BM and BN and their intersections with circles
variable (P Q : Point)
variable (BM_line : line B M)
variable (BN_line : line B N)
variable (BM_intersect_P : intersect_again BM_line C1 = P)
variable (BN_intersect_Q : intersect_again BN_line C2 = Q)

-- The main proposition
theorem MP_eq_NQ : dist M P = dist N Q := by
  sorry

end MP_eq_NQ_l178_178612


namespace sequence_1006th_term_l178_178184

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d * d).sum

noncomputable def sequence (a : ℕ) : ℕ → ℕ
| 0 => a
| (n+1) => sum_of_squares_of_digits (sequence a n)

theorem sequence_1006th_term :
  sequence 2006 1006 = 145 :=
sorry

end sequence_1006th_term_l178_178184


namespace angle_MBC_60_l178_178861

variables (A B C M : Type) [affine_space A B C M]
variables (angle : A → A → ℝ)

-- Define the given angles
def angleMAB := 10
def angleMBA := 20
def angleMCA := 30
def angleMAC := 40

-- The theorem to prove
theorem angle_MBC_60 : angle B M C = 60 :=
sorry

end angle_MBC_60_l178_178861


namespace books_assigned_total_l178_178138

-- Definitions for the conditions.
def Mcgregor_books := 34
def Floyd_books := 32
def remaining_books := 23

-- The total number of books assigned.
def total_books := Mcgregor_books + Floyd_books + remaining_books

-- The theorem that needs to be proven.
theorem books_assigned_total : total_books = 89 :=
by
  sorry

end books_assigned_total_l178_178138


namespace no_solutions_sin_cos_sum_zero_l178_178387

theorem no_solutions_sin_cos_sum_zero :
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → (1 / Real.sin x + 1 / Real.cos x ≠ 4)) →
  let solutions := { x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ (1 / Real.sin x + 1 / Real.cos x = 4) } in
  (solutions.sum id = 0) :=
begin
  -- definition of the sum of solutions
  let solutions := { x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ (1 / Real.sin x + 1 / Real.cos x = 4) },
  -- proof to verify that the sum of solutions is 0 would go here
  sorry
end

end no_solutions_sin_cos_sum_zero_l178_178387


namespace polynomial_bound_implies_l178_178219

theorem polynomial_bound_implies :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by
  sorry

end polynomial_bound_implies_l178_178219


namespace point_P_on_extension_line_of_segment_AB_l178_178010

-- Define points on the plane
variables (O A B P : Type) [AddCommGroup O]
variables [Module ℝ O]

-- Define vectors between points
def vector (x y : O) : O := y - x

-- Conditions
variables (h1 : (O ≠ A) ∧ (O ≠ B) ∧ (A ≠ B))
variables (h2 : (2 • vector O P) = 2 • (vector O A) + 2 • (vector O B))

-- Theorem statement
theorem point_P_on_extension_line_of_segment_AB :
(∃ t : ℝ, P = A + t • (B - A) ∧ t > 1) :=
sorry

end point_P_on_extension_line_of_segment_AB_l178_178010


namespace intersection_of_A_and_B_l178_178466

variable {α : Type*}
variable (A B : Set α)

theorem intersection_of_A_and_B (hA : A = {x | x > 1}) (hB : B = {x | x < 3}) :
  A ∩ B = {x | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_of_A_and_B_l178_178466


namespace find_x_intercept_l178_178674

-- Define the conditions
def point := (3, 4)  -- The point through which the line passes
def slope := 2       -- The slope of the line

-- Define what we need to prove
theorem find_x_intercept : ∃ x : ℝ, 2 * x - 2 = 0 ∧ (0, x) = (0, 1) :=
by
  sorry

end find_x_intercept_l178_178674


namespace identify_infected_person_in_4_tests_l178_178058

theorem identify_infected_person_in_4_tests :
  (∀ (group : Fin 16 → Bool), ∃ infected : Fin 16, group infected = ff) →
  ∃ (tests_needed : ℕ), tests_needed = 4 :=
by sorry

end identify_infected_person_in_4_tests_l178_178058


namespace sufficient_condition_log_gt_l178_178334

theorem sufficient_condition_log_gt (a b : ℝ) (ha : a > 0) (hb : b > 0) : (log a > log b) → (a > b) :=
sorry

end sufficient_condition_log_gt_l178_178334


namespace range_of_values_for_a_l178_178066

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, (x + 2) / 3 - x / 2 > 1 → 2 * (x - a) ≤ 0) → a ≥ -2 :=
by
  intro h
  sorry

end range_of_values_for_a_l178_178066


namespace min_value_f_is_5sqrt5_l178_178345

def f (θ : Real) : Real := 1 / Real.sin θ + 8 / Real.cos θ

theorem min_value_f_is_5sqrt5 : ∃ (θ : Real), f θ = 5 * Real.sqrt 5 :=
by
  sorry

end min_value_f_is_5sqrt5_l178_178345


namespace part_one_part_two_l178_178032

def f (x : ℝ) : ℝ :=
  sin^2 x + sqrt 3 * cos x * cos (π / 2 - x)

theorem part_one : f (π / 6) = 1 := sorry

theorem part_two (k : ℝ) (h : 0 < k) : 
  monotone_on (fun x => f (k * x + π / 12)) (Icc (-π / 6) (π / 3)) ↔ k ≤ 3 / 4 := sorry

end part_one_part_two_l178_178032


namespace distance_between_A_and_B_l178_178636

theorem distance_between_A_and_B (D : ℝ) 
  (speed_faster_train : ℝ := 65)
  (speed_slower_train : ℝ := 29)
  (time_difference : ℝ := 5) :
  D / speed_slower_train = D / speed_faster_train + time_difference →
  D ≈ 261.81 := 
sorry

end distance_between_A_and_B_l178_178636


namespace sin_B_and_triangle_count_l178_178490

noncomputable def triangle_ABC := {A B C : ℝ}

axiom angle_A : ∀ (A B C : ℝ), A = 30 * ℝπ / 180
axiom side_a : ∀ (a b c : ℝ), a = 2
axiom side_b : ∀ (a b c : ℝ), b = 2 * real.sqrt 3

theorem sin_B_and_triangle_count (A B C a b : ℝ) : 
  (A = 30 * ℝπ / 180) → 
  (a = 2) → 
  (b = 2 * real.sqrt 3) → 
  (real.sin B = real.sqrt 3 / 2) ∧ 
  (∃ (num_triangles : ℕ), num_triangles = 2) :=
by 
  intros hA ha hb
  sorry

end sin_B_and_triangle_count_l178_178490


namespace circle_passes_through_fixed_point_l178_178389

theorem circle_passes_through_fixed_point (a : ℝ) (ha : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ (x y : ℝ), (x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0) → (x, y) = P :=
sorry

end circle_passes_through_fixed_point_l178_178389


namespace smallest_circle_radius_eq_l178_178090

open Real

-- Declaring the problem's conditions
def largestCircleRadius : ℝ := 10
def smallestCirclesCount : ℕ := 6
def congruentSmallerCirclesFitWithinLargerCircle (r : ℝ) : Prop :=
  3 * (2 * r) = 2 * largestCircleRadius

-- Stating the theorem to prove
theorem smallest_circle_radius_eq :
  ∃ r : ℝ, congruentSmallerCirclesFitWithinLargerCircle r ∧ r = 10 / 3 :=
by
  sorry

end smallest_circle_radius_eq_l178_178090


namespace total_travel_time_is_19_hours_l178_178710

-- Define the distances and speeds as constants
def distance_WA_ID := 640
def speed_WA_ID := 80
def distance_ID_NV := 550
def speed_ID_NV := 50

-- Define the times based on the given distances and speeds
def time_WA_ID := distance_WA_ID / speed_WA_ID
def time_ID_NV := distance_ID_NV / speed_ID_NV

-- Define the total time
def total_time := time_WA_ID + time_ID_NV

-- Prove that the total travel time is 19 hours
theorem total_travel_time_is_19_hours : total_time = 19 := by
  sorry

end total_travel_time_is_19_hours_l178_178710


namespace replace_stars_with_identity_l178_178958

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l178_178958


namespace distance_C_from_line_l178_178232

noncomputable def distance_of_point_C : ℝ :=
  let side_length := 2
  let rotation_angle := 30 * (Float.pi / 180) -- converting degrees to radians
  let diagonal_length := Float.sqrt (2 * side_length^2)
  let vertical_drop := diagonal_length * Float.sin rotation_angle
  0

theorem distance_C_from_line :
  ∀ (side_length : ℝ) (rotation_angle : ℝ),
  side_length = 2 →
  rotation_angle = 30 * (Float.pi / 180) → -- rotation 30 degrees in radians
  distance_of_point_C = 0 :=
by
  intros side_length rotation_angle h_side_length h_rotation_angle
  rw [distance_of_point_C, h_side_length, h_rotation_angle]
  -- assuming vertical_drop = sqrt(2)
  sorry

end distance_C_from_line_l178_178232


namespace zoo_rhinos_l178_178682

-- Define the conditions
def initial_animals : ℕ := 68
def gorillas_sent_away : ℕ := 6
def hippo_adopted : ℕ := 1
def lion_cubs_born : ℕ := 8
def meerkat_multiplier : ℕ := 2
def total_animals_at_end : ℕ := 90

-- Define the total animals before taking in rhinos
def animals_before_rhinos (initial animals gorillas hippo cubs meerkats_multiplier : ℕ) : ℕ :=
  initial - gorillas + hippo + cubs + (meerkats_multiplier * cubs)

-- Define the endangered rhinos taken in
def endangered_rhinos (end total before : ℕ) : ℕ :=
  end - before

-- State the proof problem
theorem zoo_rhinos :
  endangered_rhinos total_animals_at_end (animals_before_rhinos initial_animals gorillas_sent_away hippo_adopted lion_cubs_born meerkat_multiplier) = 3 :=
by
  sorry

end zoo_rhinos_l178_178682


namespace magnitude_sub_eq_5sqrt2_l178_178019

-- Define the vectors and the condition of orthogonality
variables (m : ℝ) (a b : ℝ × ℝ)
-- Define the vectors a and b
def a := (m, 1 : ℝ)
def b := (2, -6 : ℝ)

-- Define the orthogonality condition a ⊥ b
def orthogonal : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Prove the magnitude of a - b is 5√2
theorem magnitude_sub_eq_5sqrt2 (h : orthogonal a b) : 
  let diff := (a.1 - b.1, a.2 - b.2)
  in ∥diff∥ = 5 * Real.sqrt 2 := by
  sorry

end magnitude_sub_eq_5sqrt2_l178_178019


namespace total_fruit_weight_l178_178231

-- Definitions for the conditions
def mario_ounces : ℕ := 8
def lydia_ounces : ℕ := 24
def nicolai_pounds : ℕ := 6
def ounces_per_pound : ℕ := 16

-- Theorem statement
theorem total_fruit_weight : 
  ((mario_ounces / ounces_per_pound : ℚ) + 
   (lydia_ounces / ounces_per_pound : ℚ) + 
   (nicolai_pounds : ℚ)) = 8 := 
sorry

end total_fruit_weight_l178_178231


namespace replace_stars_with_identity_l178_178955

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l178_178955


namespace no_five_integer_solutions_divisibility_condition_l178_178358

variables (k : ℤ) 

-- Definition of equation
def equation (x y : ℤ) : Prop :=
  y^2 - k = x^3

-- Variables to capture the integer solutions
variables (x1 x2 x3 x4 x5 y1 : ℤ)

-- Prove that there do not exist five solutions satisfying the given forms
theorem no_five_integer_solutions :
  ¬(equation k x1 y1 ∧ 
    equation k x2 (y1 - 1) ∧ 
    equation k x3 (y1 - 2) ∧ 
    equation k x4 (y1 - 3) ∧ 
    equation k x5 (y1 - 4)) :=
sorry

-- Prove divisibility condition for the first four solutions
theorem divisibility_condition :
  (equation k x1 y1 ∧ 
   equation k x2 (y1 - 1) ∧ 
   equation k x3 (y1 - 2) ∧ 
   equation k x4 (y1 - 3)) → 
  63 ∣ (k - 17) :=
sorry

end no_five_integer_solutions_divisibility_condition_l178_178358


namespace min_positive_period_of_f_f_monotonically_decreasing_intervals_min_value_in_interval_l178_178791

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x - real.sqrt 3 * cos x) + real.sqrt 3

theorem min_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = real.pi := by
  sorry

theorem f_monotonically_decreasing_intervals : 
  ∀ k : ℤ, ∀ x : ℝ, k * real.pi + 5 * real.pi / 12 ≤ x → x ≤ k * real.pi + 11 * real.pi / 12 → 
  f' x < 0 := by
  sorry

theorem min_value_in_interval : 
  ∃ x : ℝ, x ∈ set.Icc (real.pi / 2) real.pi ∧ f x = -2 ∧ x = 11 * real.pi / 12 := by
  sorry

end min_positive_period_of_f_f_monotonically_decreasing_intervals_min_value_in_interval_l178_178791


namespace reflection_square_identity_l178_178889

open Matrix

noncomputable def reflectionMatrix (v : Vector (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ := sorry

theorem reflection_square_identity :
  let R := reflectionMatrix ![4, -2] in
  R * R = 1 :=
by
  sorry

end reflection_square_identity_l178_178889


namespace second_statue_weight_l178_178805

theorem second_statue_weight (S : ℕ) :
  ∃ S : ℕ,
    (80 = 10 + S + 15 + 15 + 22) → S = 18 :=
by
  sorry

end second_statue_weight_l178_178805


namespace cylindrical_log_distance_l178_178661

def cylinder_radius := 3
def R₁ := 104
def R₂ := 64
def R₃ := 84
def straight_segment := 100

theorem cylindrical_log_distance :
  let adjusted_radius₁ := R₁ - cylinder_radius
  let adjusted_radius₂ := R₂ + cylinder_radius
  let adjusted_radius₃ := R₃ - cylinder_radius
  let arc_distance₁ := π * adjusted_radius₁
  let arc_distance₂ := π * adjusted_radius₂
  let arc_distance₃ := π * adjusted_radius₃
  let total_distance := arc_distance₁ + arc_distance₂ + arc_distance₃ + straight_segment
  total_distance = 249 * π + 100 :=
sorry

end cylindrical_log_distance_l178_178661


namespace polygon_fraction_sum_lt_two_l178_178746

theorem polygon_fraction_sum_lt_two {n : ℕ} (a : Fin n → ℝ) (hpos: ∀ i, 0 < a i) 
  (htriangle : ∀ i, a i < ∑ j, if j = i then 0 else a j) :
  (∑ i, a i / (∑ j, if j = i then 0 else a j)) < 2 :=
by
  sorry

end polygon_fraction_sum_lt_two_l178_178746


namespace pyramid_volume_l178_178975

noncomputable def volume_of_pyramid (S α β : ℝ) : ℝ :=
  (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β)))

theorem pyramid_volume 
  (S α β : ℝ)
  (base_area : S > 0)
  (equal_lateral_edges : true)
  (dihedral_angles : α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2) :
  volume_of_pyramid S α β = (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β))) :=
by
  sorry

end pyramid_volume_l178_178975


namespace find_divisor_l178_178476

theorem find_divisor (D Q R Div : ℕ) (h1 : Q = 40) (h2 : R = 64) (h3 : Div = 2944) 
  (h4 : Div = (D * Q) + R) : D = 72 :=
by
  sorry

end find_divisor_l178_178476


namespace remaining_inventory_l178_178366

def initial_inventory : Int := 4500
def bottles_sold_mon : Int := 2445
def bottles_sold_tue : Int := 906
def bottles_sold_wed : Int := 215
def bottles_sold_thu : Int := 457
def bottles_sold_fri : Int := 312
def bottles_sold_sat : Int := 239
def bottles_sold_sun : Int := 188

def bottles_received_tue : Int := 350
def bottles_received_thu : Int := 750
def bottles_received_sat : Int := 981

def total_bottles_sold : Int := bottles_sold_mon + bottles_sold_tue + bottles_sold_wed + bottles_sold_thu + bottles_sold_fri + bottles_sold_sat + bottles_sold_sun
def total_bottles_received : Int := bottles_received_tue + bottles_received_thu + bottles_received_sat

theorem remaining_inventory (initial_inventory bottles_sold_mon bottles_sold_tue bottles_sold_wed bottles_sold_thu bottles_sold_fri bottles_sold_sat bottles_sold_sun bottles_received_tue bottles_received_thu bottles_received_sat total_bottles_sold total_bottles_received : Int) :
  initial_inventory - total_bottles_sold + total_bottles_received = 819 :=
by
  sorry

end remaining_inventory_l178_178366


namespace identity_of_polynomials_l178_178947

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l178_178947


namespace different_colors_probability_l178_178313

theorem different_colors_probability :
  let total_balls := 5
  let white_balls := 2
  let black_balls := 3
  let total_outcomes := total_balls * total_balls
  let favorable_outcomes := (white_balls * black_balls) + (black_balls * white_balls)
  let probability := favorable_outcomes / total_outcomes
  in probability = 12 / 25 := by
  sorry

end different_colors_probability_l178_178313


namespace ratio_of_speeds_l178_178681

theorem ratio_of_speeds (v_A v_B : ℝ) (t : ℝ) (hA : v_A = 120 / t) (hB : v_B = 60 / t) : v_A / v_B = 2 :=
by {
  sorry
}

end ratio_of_speeds_l178_178681


namespace cos_B_eq_sqrt_6_div_3_l178_178075

-- Let's set up the conditions and what we need to prove
theorem cos_B_eq_sqrt_6_div_3
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides opposite to the angles A, B, C respectively
  (hA : A = 60 * real.pi / 180) -- angle A in radians
  (ha : a = 15)
  (hb : b = 10)
  (hABC : A + B + C = real.pi) -- Sum of angles in a triangle in radians
  (hLawOfSines : a / real.sin A = b / real.sin B) -- Law of Sines
  : real.cos B = sqrt 6 / 3 :=
sorry

end cos_B_eq_sqrt_6_div_3_l178_178075


namespace multiples_of_15_between_21_205_l178_178053

theorem multiples_of_15_between_21_205 : 
  ∃ (n : ℕ), ∀ (k : ℕ), (21 < k * 15) ∧ (k * 15 < 205) ↔ k ∈ (finset.range (14)).filter (λ n, 2 ≤ n ∧ n ≤ 13) ∧ n = (13 - 2 + 1) :=
sorry

end multiples_of_15_between_21_205_l178_178053


namespace rain_is_random_event_l178_178227

def is_random_event (p : ℝ) : Prop := p > 0 ∧ p < 1

theorem rain_is_random_event (p : ℝ) (h : p = 0.75) : is_random_event p :=
by
  -- Here we will provide the necessary proof eventually.
  sorry

end rain_is_random_event_l178_178227


namespace specific_value_eq_l178_178308

def specific_value (x : ℕ) : ℕ := 25 * x

theorem specific_value_eq : specific_value 27 = 675 := by
  sorry

end specific_value_eq_l178_178308


namespace range_of_alpha_l178_178931

theorem range_of_alpha :
  ∀ x ∈ set.Icc (-1:ℝ) (real.sqrt 2),
    let α := real.arctan (x^2 - 1) in
    (α ∈ set.Icc 0 (π/4) ∨ α ∈ set.Icc (3 * π / 4) π) :=
begin
  sorry
end

end range_of_alpha_l178_178931


namespace factorial_base_9_zeroes_l178_178452

theorem factorial_base_9_zeroes (n : ℕ) (fact_n_eq : n = 12) :
  let factorial := n.factorial in
  let total_threes := 1   -- 3
                     + 1 -- 6
                     + 2 -- 9
                     + 1 -- 12 
  in (total_threes / 2).floor = 2 := 
by
  sorry

end factorial_base_9_zeroes_l178_178452


namespace students_met_goal_l178_178668

def money_needed_per_student : ℕ := 450
def number_of_students : ℕ := 6
def collective_expenses : ℕ := 3000
def amount_raised_day1 : ℕ := 600
def amount_raised_day2 : ℕ := 900
def amount_raised_day3 : ℕ := 400
def days_remaining : ℕ := 4
def half_of_first_three_days : ℕ :=
  (amount_raised_day1 + amount_raised_day2 + amount_raised_day3) / 2

def total_needed : ℕ :=
  money_needed_per_student * number_of_students + collective_expenses
def total_raised : ℕ :=
  amount_raised_day1 + amount_raised_day2 + amount_raised_day3 + (half_of_first_three_days * days_remaining)

theorem students_met_goal : total_raised >= total_needed := by
  sorry

end students_met_goal_l178_178668


namespace sin_double_angle_identity_l178_178015

variable (α : Real)

theorem sin_double_angle_identity (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 :=
by
  sorry

end sin_double_angle_identity_l178_178015


namespace lillian_total_candies_l178_178131

def initial_candies : ℕ := 88
def father_candies : ℕ := 5
def friend_multiplier : ℕ := 4

theorem lillian_total_candies :
  let total_candies := initial_candies + father_candies + friend_multiplier * father_candies in
  total_candies = 113 := by
  sorry

end lillian_total_candies_l178_178131


namespace count_positive_integers_solve_count_positive_integers_l178_178056

theorem count_positive_integers (n : ℕ) : 
  (1000 < n^3 ∧ n^3 < 4000) → (11 ≤ n ∧ n ≤ 15) :=
begin
  sorry
end

theorem solve_count_positive_integers : 
  (∃ n : ℕ, 1000 < n^3 ∧ n^3 < 4000) → 
  (finset.range 16).filter (λ n : ℕ, 1000 < n^3 ∧ n^3 < 4000).card = 5 :=
begin
  sorry
end

end count_positive_integers_solve_count_positive_integers_l178_178056


namespace rhombus_longest_diagonal_l178_178324

-- Definitions based on the conditions
def area : ℝ := 200
def ratio_d1_d2 : ℝ := 4 / 3

-- The statement we need to prove
theorem rhombus_longest_diagonal (d1 d2 : ℝ) (h1 : d1 = 4 * d2) (h2 : 0.5 * d1 * d2 = area) :
  d1 = 4 * real.sqrt (200 / 6) :=
by
  sorry

end rhombus_longest_diagonal_l178_178324


namespace perpendicular_plane_l178_178642

variables {α : Type*} {a b c : Set α} {A : Set α}

-- Assumptions
variable (h1 : ∀ x ∈ a, ∀ y ∈ b, orthogonal x y)
variable (h2 : ∀ x ∈ c, orthogonal x a)
variable (h3 : ∀ x ∈ c, orthogonal x b)
variable (h4 : a ⊆ α)
variable (h5 : b ⊆ α)
variable (h6 : intersect a b = A)

-- Goal
theorem perpendicular_plane (h1 : ∀ x ∈ a, ∀ y ∈ b, orthogonal x y)
                           (h2 : ∀ x ∈ c, orthogonal x a)
                           (h3 : ∀ x ∈ c, orthogonal x b)
                           (h4 : a ⊆ α)
                           (h5 : b ⊆ α)
                           (h6 : intersect a b = A) : 
    ∀ x ∈ c, orthogonal x α :=
sorry

end perpendicular_plane_l178_178642


namespace goldfish_sold_l178_178211

variables (buy_price sell_price tank_cost short_percentage : ℝ)

theorem goldfish_sold (h1 : buy_price = 0.25)
                      (h2 : sell_price = 0.75)
                      (h3 : tank_cost = 100)
                      (h4 : short_percentage = 0.45) :
  let profit_per_goldfish := sell_price - buy_price in
  let shortfall := tank_cost * short_percentage in
  let earnings := tank_cost - shortfall in
  let goldfish_count := earnings / profit_per_goldfish in
  goldfish_count = 110 :=
by {
  let profit_per_goldfish := sell_price - buy_price;
  let shortfall := tank_cost * short_percentage;
  let earnings := tank_cost - shortfall;
  let goldfish_count := earnings / profit_per_goldfish;
  calc goldfish_count
      = earnings / profit_per_goldfish : by exact rfl
  ... = 110 : by sorry
}

end goldfish_sold_l178_178211


namespace cos_2_beta_l178_178014

variable (α β : ℝ)

variables (h1 : π / 2 < β ∧ β < α ∧ α < 3 / 4 * π)
variables (h2 : cos (α + β) = -3 / 5)
variables (h3 : sin (α - β) = 5 / 13)

theorem cos_2_beta : cos (2 * β) = -56 / 65 :=
by
  -- The proof is omitted
  sorry

end cos_2_beta_l178_178014


namespace min_max_values_f_l178_178193

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l178_178193


namespace average_speed_trip_l178_178651

theorem average_speed_trip (d1 d2 s1 s2 : ℝ) 
  (h1 : d1 = 90) (h2 : s1 = 30) 
  (h3 : d2 = 75) (h4 : s2 = 60) : 
  let total_distance := d1 + d2 in
  let time1 := d1 / s1 in
  let time2 := d2 / s2 in
  let total_time := time1 + time2 in
  let average_speed := total_distance / total_time in
  average_speed ≈ 38.82 :=
by
  sorry

#check average_speed_trip

end average_speed_trip_l178_178651


namespace cannot_be_covered_by_dominoes_l178_178360

theorem cannot_be_covered_by_dominoes :
  let board_dimensions := [(4, 4, 1), (3, 7, 0), (5, 4, 0), (7, 2, 0)] in
  let total_squares (length : Nat) (width : Nat) (removed : Nat) := length * width - removed in
  let is_odd (n : Nat) := n % 2 = 1 in
  (∃(dim : Nat × Nat × Nat), dim ∈ board_dimensions ∧ is_odd (total_squares dim.1 dim.2 dim.3)) ↔
  {dim ∈ board_dimensions | is_odd (total_squares dim.1 dim.2 dim.3)} = { (4, 4, 1), (3, 7, 0) } :=
by
  sorry

end cannot_be_covered_by_dominoes_l178_178360


namespace maggie_remaining_goldfish_l178_178535

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end maggie_remaining_goldfish_l178_178535


namespace total_population_calculation_l178_178479

theorem total_population_calculation :
  ∀ (total_lions total_leopards adult_lions adult_leopards : ℕ)
  (female_lions male_lions female_leopards male_leopards : ℕ)
  (adult_elephants baby_elephants total_elephants total_zebras : ℕ),
  total_lions = 200 →
  total_lions = 2 * total_leopards →
  adult_lions = 3 * total_lions / 4 →
  adult_leopards = 3 * total_leopards / 5 →
  female_lions = 3 * total_lions / 5 →
  male_lions = 2 * total_lions / 5 →
  female_leopards = 2 * total_leopards / 3 →
  male_leopards = total_leopards / 3 →
  adult_elephants = (adult_lions + adult_leopards) / 2 →
  baby_elephants = 100 →
  total_elephants = adult_elephants + baby_elephants →
  total_zebras = adult_elephants + total_leopards →
  total_lions + total_leopards + total_elephants + total_zebras = 710 :=
by sorry

end total_population_calculation_l178_178479


namespace number_of_proper_subsets_l178_178585

def A : Set ℕ := {1, 2}

theorem number_of_proper_subsets : {s : Set ℕ | s ⊆ A ∧ s ≠ A}.toFinset.card = 3 := 
by
  sorry

end number_of_proper_subsets_l178_178585


namespace area_square_XY_l178_178609

theorem area_square_XY (XY YZ XZ : ℝ) (h : XY^2 + YZ^2 + XZ^2 = 500) (right_angled_Y : (XY^2 + YZ^2 = XZ^2)) : XY^2 = 125 :=
by
  -- Using the given right angle condition at Y
  have pythagoras := right_angled_Y
  -- Substituting the value of XZ^2 from Pythagoras theorem into the sum of areas
  have sum_of_areas := h
  rw [pythagoras] at sum_of_areas
  -- Simplifying the equation
  linarith

end area_square_XY_l178_178609


namespace no_eleven_in_sequence_p_l178_178592

def largest_prime_factor (n : ℕ) : ℕ :=
-- Definition for finding the largest prime factor, implementation skipped
sorry

def sequence_p : ℕ → ℕ
| 1 := 2
| (n+1) := largest_prime_factor (sequence_p 1 * sequence_p 2 * ... * sequence_p n + 1)
-- Implementation of the sequence_p function omitted for brevity
sorry

theorem no_eleven_in_sequence_p (n : ℕ) : sequence_p n ≠ 11 :=
sorry

end no_eleven_in_sequence_p_l178_178592


namespace antiderivative_correct_l178_178000

def f (x : ℝ) : ℝ := 2 * x
def F (x : ℝ) : ℝ := x^2 + 2

theorem antiderivative_correct :
  (∀ x, f x = deriv (F) x) ∧ (F 1 = 3) :=
by
  sorry

end antiderivative_correct_l178_178000


namespace right_triangle_one_right_angle_l178_178815

theorem right_triangle_one_right_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90 ∨ B = 90 ∨ C = 90) : (interp A B C).count (90) = 1 :=
by
  sorry

end right_triangle_one_right_angle_l178_178815


namespace cuckoo_clock_chimes_l178_178659

-- Define the conditions
def cuckoo_chimes_per_hour (h : ℕ) : ℕ :=
  match h with
  | h if h >= 1 ∧ h <= 12 => h
  | h if h > 12 => cuckoo_chimes_per_hour (h - 12)
  | _ => 0

-- State the problem in Lean 4
theorem cuckoo_clock_chimes :
  let start_hour := 9
  let end_hour := start_hour + 7
  let chimes := List.sum (List.map cuckoo_chimes_per_hour [10, 11, 12, 1, 2, 3, 4])
  chimes = 43 :=
by
  sorry

end cuckoo_clock_chimes_l178_178659


namespace arithmetic_sqrt_9_l178_178172

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l178_178172


namespace polynomial_is_correct_l178_178109

def f (x : ℝ) : ℝ := x^3 + (1 / 2) * x^2 + (11 / 2) * x + 6

theorem polynomial_is_correct :
  (∀ (x : ℝ), f(x) = x^3 + (1 / 2) * x^2 + (11 / 2) * x + 6) ∧
  f(0) = 6 ∧
  f(1) = 12 ∧
  f(-1) = 0 := 
by
  sorry

end polynomial_is_correct_l178_178109


namespace equal_area_of_inscribed_polygons_l178_178610

open EuclideanGeometry -- Assuming geometry is required

theorem equal_area_of_inscribed_polygons
  (n : ℕ)
  (circle : Circle ℝ)
  (polygon1 polygon2 : Finset ℝ)
  (h1 : polygon1.card = n)
  (h2 : polygon2.card = n)
  (h3 : polygon1 = polygon2) :
  (area_of_polygon (inscribed_polygon circle polygon1 h1) = 
   area_of_polygon (inscribed_polygon circle polygon2 h2)) := 
sorry

end equal_area_of_inscribed_polygons_l178_178610


namespace non_divisible_by_5_four_digit_count_l178_178339

theorem non_divisible_by_5_four_digit_count :
  let digits := {0, 1, 2, 3, 4, 5}
  let four_digit_nums_without_repetition := 
    {n : Nat | 
      ∃ a b c d : Nat, 
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
      a ≠ 0 ∧ (d ≠ 0 ∧ d ≠ 5) ∧ 
      n = a * 1000 + b * 100 + c * 10 + d}
  in four_digit_nums_without_repetition.size = 192 :=
sorry

end non_divisible_by_5_four_digit_count_l178_178339


namespace angle_eta_and_distance_l178_178927

noncomputable def square_side : ℝ := 20.0

def beta1 : ℝ := 40.0 * real.pi / 180.0
def delta1 : ℝ := 38.0 * real.pi / 180.0
def beta2 : ℝ := 32.0 * real.pi / 180.0
def delta2 : ℝ := 40.0 * real.pi / 180.0

theorem angle_eta_and_distance :
  let eta1 := 41.6 * real.pi / 180.0,
      distance := 13.62
  in 
  (
    arccos ((cot beta1)^2 + (cot delta1)^2 - 1 / (2 * cot beta1 * cot delta1)) = eta1 
    ∧ sqrt ((a * cot beta1 - a * cot delta1)^2 + (a / sin beta1 - a / sin delta1)^2) = distance
  ) :=
sorry

end angle_eta_and_distance_l178_178927


namespace reservation_charge_l178_178322

variable (F R : ℝ)

theorem reservation_charge :
  F + R = 216 ∧ (F + R) + (F / 2 + R) = 327 → R = 6 :=
by
  intros h
  cases h with h1 h2
  sorry

end reservation_charge_l178_178322


namespace part1_l178_178011

noncomputable def A (a : ℝ) : set ℝ := {x | (x - (a - 1)) * (x - (a + 1)) < 0}
noncomputable def B : set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem part1 (a : ℝ) (h : a = 2) : A a ∪ B = B := 
sorry

end part1_l178_178011


namespace cover_big_circle_with_smaller_ones_l178_178078

theorem cover_big_circle_with_smaller_ones :
  ∃ (five_circles : fin 5 → cylinder), 
  (∀ i : fin 5, (five_circles i).radius = 25) ∧
  (let big_circle : set (ℝ × ℝ) := {p | ↑(p.1 ^ 2 + p.2 ^ 2) < (40.95 ^ 2)} in
  ∀ p ∈ big_circle, ∃ (i : fin 5), p ∈ (five_circles i)) :=
sorry

end cover_big_circle_with_smaller_ones_l178_178078


namespace jack_sugar_final_l178_178865

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l178_178865


namespace power_series_expansion_l178_178096

theorem power_series_expansion (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, (∑ n in (finset.range 1000), ((log a) ^ n) / (n.factorial) * x ^ n) = a ^ x) :=
by sorry

end power_series_expansion_l178_178096


namespace problem_a_problem_b_l178_178146

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end problem_a_problem_b_l178_178146


namespace complex_power_2009_eq_I_l178_178598

def imaginary_unit_power (n : ℤ) : ℂ := complex.I ^ n

theorem complex_power_2009_eq_I : imaginary_unit_power 2009 = complex.I := 
sorry

end complex_power_2009_eq_I_l178_178598


namespace work_time_together_l178_178289

-- Define the basic constants and relationships in the problem
variables (B_rate : ℝ) (A_rate : ℝ)
variables (B_days : ℝ) (work : ℝ := 1)

-- Conditions from the problem
def condition1 : Prop := A_rate = 2 * B_rate
def condition2 : Prop := B_rate = work / B_days

-- Given B_days as 10
def B_days_val : Prop := B_days = 10

-- We need to prove this
theorem work_time_together : 
  (condition1 ∧ condition2 ∧ B_days_val) → 
  (∃ T : ℝ, T = 10 / 3) :=
  by
    intros
    sorry

end work_time_together_l178_178289


namespace star_five_seven_l178_178396

def star (a b : ℕ) : ℕ := (a + b + 3) ^ 2

theorem star_five_seven : star 5 7 = 225 := by
  sorry

end star_five_seven_l178_178396


namespace pyramid_volume_proof_l178_178560

variables (A B C D P : Type) [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C] [euclidean_space ℝ D] [euclidean_space ℝ P]

variables (AB : ℝ) (BC : ℝ) (PA : ℝ) (PB : ℝ) (PA_perp_AD : Prop) (PA_perp_AB : Prop)

noncomputable def volume_pyramid (AB BC PA : ℝ) : ℝ :=
  (1 / 3) * (AB * BC) * PA

theorem pyramid_volume_proof (h1 : AB = 8) (h2 : BC = 4) (h3 : PA_perp_AD) (h4 : PA_perp_AB) (h5 : PB = 17) :
  volume_pyramid AB BC PA = 160 :=
by
  sorry

end pyramid_volume_proof_l178_178560


namespace identity_of_polynomials_l178_178945

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l178_178945


namespace concurrent_lines_on_circle_l178_178690

theorem concurrent_lines_on_circle 
  {O : Type*} [metric_space O] [normed_group O] [normed_space ℝ O]  
  (A B X P Q : O) 
  (is_diameter : dist A B = 2 * (real.pi))
  (on_circle : dist A O = dist B O ∧ ∀ (X ≠ mid_pt(A,B)), dist X O = dist A O)
  (tangent_at_A : ∃ ℓ_A : set O, ∀ (x ∈ ℓ_A), dist x O = dist A O )
  (tangent_at_B : ∃ ℓ_B : set O, ∀ (b ∈ ℓ_B), dist b O = dist B O )
  (intersect_P : ∃ (ℓ_A : set O), (B ∈ ℓ_A) ∧ (X ∉ ℓ_A) ∧ (tangent_at_A = ℓ_A))
  (intersect_Q : ∃ (ℓ_B : set O), (A ∈ ℓ_B) ∧ (X ∉ ℓ_B) ∧ (tangent_at_B = ℓ_B)) :
  ∃ (K : O), collinear {P, Q, K} ∧ collinear {A, B, K} ∧ tangent_apex {X, K} :=
sorry

end concurrent_lines_on_circle_l178_178690


namespace identity_verification_l178_178952

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l178_178952


namespace right_triangle_right_angles_l178_178811

theorem right_triangle_right_angles (T : Triangle) (h1 : T.is_right_triangle) :
  T.num_right_angles = 1 :=
sorry

end right_triangle_right_angles_l178_178811


namespace parabola_y1_gt_y2_l178_178488

variable {x1 x2 y1 y2 : ℝ}

theorem parabola_y1_gt_y2 
  (hx1 : -4 < x1 ∧ x1 < -2) 
  (hx2 : 0 < x2 ∧ x2 < 2) 
  (hy1 : y1 = x1^2) 
  (hy2 : y2 = x2^2) : 
  y1 > y2 :=
by 
  sorry

end parabola_y1_gt_y2_l178_178488


namespace a_general_formula_S_sum_formula_l178_178855

-- Defining the sequence {a_n}
def a : ℕ → ℕ 
| 0       := 1
| (n + 1) := a n + 2^n

-- Proving the general formula for {a_n}
theorem a_general_formula (n : ℕ) : a (n + 1) = 2^(n + 1) - 1 := 
sorry

-- Defining the sequence {b_n}
def b (n : ℕ) := n * (1 + a n)

-- Defining the sum S_n of the first n terms of {b_n}
def S (n : ℕ) : ℕ :=
Finset.sum (Finset.range n) b

-- Proving the sum of the first n terms of {b_n}
theorem S_sum_formula (n : ℕ) : S n = 2 - 2^(n+1) + n * 2^(n+1) :=
sorry

end a_general_formula_S_sum_formula_l178_178855


namespace max_three_digit_divisible_by_25_in_sequence_l178_178663

def is_arith_progression (a : ℕ → ℕ) : Prop :=
∀ k : ℕ, (k + 2) < (nat.succ n) → a (k + 2) = 3 * a (k + 1) - 2 * a k - 1

def isDivisibleBy25 (n : ℕ) : Prop := n % 25 = 0

theorem max_three_digit_divisible_by_25_in_sequence (a : ℕ → ℕ) (n : ℕ) (h3 : 3 ≤ n) (h_sequence : is_arith_progression a) (h_contains2021 : ∃ k, k < n ∧ a k = 2021) :
  ∃ m, m = 36 ∧ (∀ i, 0 ≤ i ∧ i < n → (100 ≤ a i ∧ a i < 1000) → isDivisibleBy25 (a i)) :=
sorry

end max_three_digit_divisible_by_25_in_sequence_l178_178663


namespace sum_of_squares_induction_l178_178156

open Nat

theorem sum_of_squares_induction (n : ℕ) (h : 0 < n) :
  (∑ i in Finset.range n, if even i then -i^2 else i^2) = -n * (2 * n + 1) := sorry

end sum_of_squares_induction_l178_178156


namespace lunch_break_duration_l178_178547

theorem lunch_break_duration :
  ∃ L : ℝ, 
    ∀ (p h : ℝ),
      (9 - L) * (p + h) = 0.4 ∧
      (7 - L) * h = 0.3 ∧
      (12 - L) * p = 0.3 →
      L = 0.5 := by
  sorry

end lunch_break_duration_l178_178547


namespace intersection_with_plane_parallelogram_intersection_with_plane_rhombus_max_area_parallelogram_midpoints_l178_178630

structure Tetrahedron where
  A B C D : Point ℝ

theorem intersection_with_plane_parallelogram (T: Tetrahedron) :
  ∃ (P Q R S : Point ℝ), parallelogram P Q R S ∧ in_plane T P Q R S :=
sorry

theorem intersection_with_plane_rhombus (T: Tetrahedron) :
  ∃ (P Q R S : Point ℝ), rhombus P Q R S ∧ in_plane T P Q R S :=
sorry

theorem max_area_parallelogram_midpoints (T: Tetrahedron) :
  ∃ (P Q R S : Point ℝ), parallelogram P Q R S ∧
  in_plane T P Q R S ∧ max_area P Q R S :=
sorry

end intersection_with_plane_parallelogram_intersection_with_plane_rhombus_max_area_parallelogram_midpoints_l178_178630


namespace star_self_intersections_l178_178125

open Nat

theorem star_self_intersections (n k : ℕ) (h_coprime : coprime k n) (h_n_ge_5 : n ≥ 5) (h_k_lt_half_n : k < n / 2) : 
  (if n = 2018 ∧ k = 25 then (n * (k - 1) = 48432) else true) :=
sorry

end star_self_intersections_l178_178125


namespace arithmetic_square_root_of_9_is_3_l178_178169

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l178_178169


namespace sum_of_roots_l178_178459

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, (x^2 + 12 * x = 64) → ∃ s : ℝ, s = -12 ∧ (x = -6 + √100 ∨ x = -6 - √100)
  -- Note: We're adding the two possible roots representation for completeness

theorem sum_of_roots (h : ∀ x : ℝ, (x^2 + 12 * x = 64)) : problem_statement := 
by
  sorry

end sum_of_roots_l178_178459


namespace initial_percentage_alcohol_l178_178311

variables (P : ℝ) (initial_volume : ℝ) (added_volume : ℝ) (total_volume : ℝ) (final_percentage : ℝ) (init_percentage : ℝ)

theorem initial_percentage_alcohol (h1 : initial_volume = 6)
                                  (h2 : added_volume = 3)
                                  (h3 : total_volume = initial_volume + added_volume)
                                  (h4 : final_percentage = 50)
                                  (h5 : init_percentage = 100 * (initial_volume * P / 100 + added_volume) / total_volume)
                                  : P = 25 :=
by {
  sorry
}

end initial_percentage_alcohol_l178_178311


namespace problem_inequality_l178_178885

theorem problem_inequality (n a b : ℕ) (h₁ : n ≥ 2) 
  (h₂ : ∀ m, 2^m ∣ 5^n - 3^n → m ≤ a) 
  (h₃ : ∀ m, 2^m ≤ n → m ≤ b) : a ≤ b + 3 :=
sorry

end problem_inequality_l178_178885


namespace evaluate_expression_l178_178309

theorem evaluate_expression : 
  70 + (5 * 12) / (180 / 3) = 71 :=
  by
  sorry

end evaluate_expression_l178_178309


namespace proof_problem_l178_178849

variable {a : ℕ → ℝ} -- sequence a
variable {S : ℕ → ℝ} -- partial sums sequence S 
variable {n : ℕ} -- index

-- Define the conditions
def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n+1) = a n + d

def S_is_partial_sum (a S : ℕ → ℝ) : Prop := 
  ∀ n, S (n+1) = S n + a (n+1)

-- The properties given in the problem
def conditions (a S : ℕ → ℝ) : Prop :=
  is_arith_seq a ∧ 
  S_is_partial_sum a S ∧ 
  S 6 < S 7 ∧ 
  S 7 > S 8

-- The conclusions that need to be proved
theorem proof_problem (a S : ℕ → ℝ) (h : conditions a S) : 
  S 9 < S 6 ∧
  (∀ n, a 1 ≥ a (n+1)) ∧
  (∀ m, S 7 ≥ S m) := by 
  sorry

end proof_problem_l178_178849


namespace greatest_valid_number_l178_178260

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l178_178260


namespace jack_sugar_final_l178_178863

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l178_178863


namespace angle_ABC_is_90_degrees_l178_178304

theorem angle_ABC_is_90_degrees (A B C P Q : Type) 
  (h_iso : is_isosceles_triangle A B C)
  (h_equal : dist A C = dist A B)
  (h_P_on_AB : point_on_line P A B)
  (h_Q_on_BC : point_on_line Q B C)
  (h_AP_third_AC : dist A P = (1 / 3) * dist A C)
  (h_QB_third_AC : dist Q B = (1 / 3) * dist A C) :
  measure_angle A B C = 90 :=
sorry

end angle_ABC_is_90_degrees_l178_178304


namespace isosceles_triangle_l178_178086

theorem isosceles_triangle 
  (α β γ : ℝ) 
  (a b : ℝ) 
  (h_sum : a + b = (Real.tan (γ / 2)) * (a * (Real.tan α) + b * (Real.tan β)))
  (h_sum_angles : α + β + γ = π) 
  (zero_lt_γ : 0 < γ ∧ γ < π) 
  (zero_lt_α : 0 < α ∧ α < π / 2) 
  (zero_lt_β : 0 < β ∧ β < π / 2) : 
  α = β := 
sorry

end isosceles_triangle_l178_178086


namespace range_of_m_hyperbola_l178_178473

noncomputable def is_conic_hyperbola (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ f : ℝ, ∀ x y, expr x y = ((x - 2 * y + 3)^2 - f * (x^2 + y^2 + 2 * y + 1))

theorem range_of_m_hyperbola (m : ℝ) :
  is_conic_hyperbola (fun x y => m * (x^2 + y^2 + 2 * y + 1) - (x - 2 * y + 3)^2) → 5 < m :=
sorry

end range_of_m_hyperbola_l178_178473


namespace sum_of_sequence_l178_178065

variable {n : ℕ}
variable {x : ℕ → ℚ}

noncomputable def sequence_property := ∀ k, 1 ≤ k ∧ k ≤ n - 1 → x (k + 1) = x k + 1 / 3
noncomputable def initial_term := x 1 = 2

theorem sum_of_sequence (h1 : sequence_property) (h2 : initial_term) :
  ∑ i in finset.range n, x (i + 1) = n * (n + 11) / 6 := sorry

end sum_of_sequence_l178_178065


namespace mass_percentage_orange_juice_l178_178652
-- Assumptions as definitions
def initial_price_milk := 1
def initial_price_orange_juice := 6
def milk_price_decrease := 0.85 -- 85%
def orange_juice_price_increase := 1.1 -- 110%
def unchanged_drink_cost (x y : ℕ) : Prop :=
  x + 6 * y = 0.85 * x + 6 * 1.1 * y

-- Main theorem statement
theorem mass_percentage_orange_juice (x y : ℕ) (h : unchanged_drink_cost x y) : y = (1 / 5) * (x + y) :=
sorry

end mass_percentage_orange_juice_l178_178652


namespace amanda_reckonwith_graph_correct_l178_178694

open Real

noncomputable def circumference (r : ℝ) : ℝ := 2 * π * r
noncomputable def area (r : ℝ) : ℝ := π * r ^ 2

theorem amanda_reckonwith_graph_correct :
  (∀ (r : ℝ), r ∈ {1, 2, 3, 4, 5} → (circumference r, area r) belongs to the graph (A)) := sorry

end amanda_reckonwith_graph_correct_l178_178694


namespace sufficient_but_not_necessary_condition_l178_178039

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, |x - 3/4| ≤ 1/4 → (x - a) * (x - (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (x - a) * (x - (a + 1)) ≤ 0 → |x - 3/4| ≤ 1/4) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l178_178039


namespace initial_typists_count_l178_178069

theorem initial_typists_count 
  (typists_rate : ℕ → ℕ)
  (letters_in_20min : ℕ)
  (total_typists : ℕ)
  (letters_in_1hour : ℕ)
  (initial_typists : ℕ) 
  (h1 : letters_in_20min = 38)
  (h2 : letters_in_1hour = 171)
  (h3 : total_typists = 30)
  (h4 : ∀ t, 3 * (typists_rate t) = letters_in_1hour / total_typists)
  (h5 : ∀ t, typists_rate t = letters_in_20min / t) 
  : initial_typists = 20 := 
sorry

end initial_typists_count_l178_178069


namespace probability_A_not_losing_l178_178278

theorem probability_A_not_losing (P_draw : ℚ) (P_win_A : ℚ) (h1 : P_draw = 1/2) (h2 : P_win_A = 1/3) : 
  P_draw + P_win_A = 5/6 :=
by
  rw [h1, h2]
  norm_num

end probability_A_not_losing_l178_178278


namespace not_geometric_sequence_formula_sum_formula_min_k_l178_178797

variable {a : ℕ → ℝ}

-- Conditions
def sequence_conditions : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a n * a (n + 1) = 2 ^ n

-- Question 1: Prove that {a_n} is not a geometric sequence
theorem not_geometric (h : sequence_conditions) : ¬∃ r, ∀ n, a (n + 1) = r * a n :=
sorry

-- Question 2: Find the general formula for the sequence {a_n}
def general_formula (n : ℕ) : ℝ :=
  if odd n then 2 ^ ((n - 1) / 2) else 2 ^ (n / 2)

theorem sequence_formula (h : sequence_conditions) : ∀ n, a n = general_formula n :=
sorry

-- Question 3: Find the sum of the first n terms of the sequence {a_n}, denoted Sn
def sum_of_sequence (n : ℕ) : ℝ :=
  if odd n then 2 * 2 ^ ((n + 1) / 2) - 3 else 3 * (2 ^ (n / 2) - 1)

theorem sum_formula (h : sequence_conditions) : ∀ n, (finset.range n).sum a = sum_of_sequence n :=
sorry

-- Question 4: Find the minimum value of k such that 3(1 - k * a_{2n}) ≤ S_{2n} * a_{2n} holds for any n ∈ ℕ*
def S_2n (n : ℕ) : ℝ := 3 * (2 ^ n - 1)
def a_2n (n : ℕ) : ℝ := 2 ^ n

theorem min_k (h : sequence_conditions) : ∀ n k, (3 * (1 - k * a_2n n)) ≤ S_2n n * a_2n n → k ≥ -1/2 :=
sorry

end not_geometric_sequence_formula_sum_formula_min_k_l178_178797


namespace collinear_A_P_Q_l178_178760

variables {A B C D E F P Q : Type*} [convex_quad A B C D]
  (h1 : AB = BC ∧ AD = DC)
  (h2 : E ∈ segment A B)
  (h3 : F ∈ segment A D)
  (h4 : cyclic B E F D)
  (h5 : similar_triangle D P E A D C)
  (h6 : similar_triangle B Q F A B C)

theorem collinear_A_P_Q :
  collinear A P Q :=
sorry

end collinear_A_P_Q_l178_178760


namespace math_problem_l178_178453

variables {x y z a b c : ℝ}

theorem math_problem
  (h₁ : x / a + y / b + z / c = 4)
  (h₂ : a / x + b / y + c / z = 2) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end math_problem_l178_178453


namespace sumPossibleValuesOfG_l178_178715

-- Given definitions for the square entries and the common product P.
variables {b c d e g h P : ℕ}

-- Define the conditions for a multiplicative magic square.
def isMultiplicativeMagicSquare (P : ℕ) : Prop :=
  30 * b * c = P ∧
  d * e * 3 = P ∧
  g * h * 3 = P ∧
  30 * e * 3 = P ∧
  b * e * h = P ∧
  c * 3 * 3 = P ∧
  30 * e * g = P ∧
  c * e * 3 = P

-- Prove the sum of all possible values of g under the given conditions is 25.
theorem sumPossibleValuesOfG (hb hc hd he hg hh hP : ℕ) :
  isMultiplicativeMagicSquare hP →
  b > 0 → c > 0 → d > 0 → e > 0 → g > 0 → h > 0 →
  d = 30 → -- Given derived information from the solution steps.
  ∃ (gs : List ℕ), sum gs = 25 ∧ ∀ g ∈ gs, g > 0 := 
by
  sorry

end sumPossibleValuesOfG_l178_178715


namespace angle_relationship_l178_178182

/--
Given two planes α and β with a dihedral angle φ,
and a line m ⊂ α making an angle θ with plane β,
show that φ ≥ θ.
-/
theorem angle_relationship 
  (α β : Plane) (m : Line) (φ θ : ℝ) 
  (h1 : m ⊂ α) 
  (h2 : plane_angle α β = φ) 
  (h3 : line_to_plane_angle m β = θ) :
  φ ≥ θ :=
sorry

end angle_relationship_l178_178182


namespace isosceles_triangle_count_l178_178857

noncomputable def is_isosceles (a b c : Point) : Prop :=
  dist a b = dist a c ∨ dist b c = dist b a ∨ dist c a = dist c b

def seven_isosceles (A B C D E F : Point) (h1 : dist A B = dist A C)
                    (h2 : ∠ B A C = 72)
                    (h3 : is_angle_bisector B D A C)
                    (h4 : is_parallel D E A B)
                    (h5 : is_parallel E F B D) : Prop :=
  count_isosceles (list_of_triangles [A, B, C, D, E, F]) = 7

theorem isosceles_triangle_count (A B C D E F : Point) 
                                 (h1 : dist A B = dist A C)
                                 (h2 : ∠ B A C = 72)
                                 (h3 : is_angle_bisector B D A C)
                                 (h4 : is_parallel D E A B)
                                 (h5 : is_parallel E F B D) :
  seven_isosceles A B C D E F h1 h2 h3 h4 h5 := sorry

end isosceles_triangle_count_l178_178857


namespace area_of_given_trapezoid_l178_178181

-- Geometry definitions
def trapezoid_diagonal_lengths := (d1 d2 : ℝ) (h : ℝ)

-- Prove the area of the trapezoid given its diagonal lengths and height between parallel sides.
theorem area_of_given_trapezoid (d1 d2 h : ℝ) (hd1 : d1 = 200) (hd2 : d2 = 300) (hh : h = 160) :
  (1 / 2) * d1 * d2 * (1 / h) = 29901 :=
by
  sorry

end area_of_given_trapezoid_l178_178181


namespace solve_x_l178_178965

theorem solve_x (x : ℝ) (h : sqrt (9 + sqrt (18 + 9 * x)) + sqrt (3 + sqrt (3 + x)) = 3 + 3 * sqrt 3) : x = 34 :=
sorry

end solve_x_l178_178965


namespace sal_pizza_increase_l178_178959

noncomputable def percent_increase_area (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let area1 := Real.pi * r1^2
  let area2 := Real.pi * r2^2
  ((area2 - area1) / area1) * 100

theorem sal_pizza_increase :
  percent_increase_area 14 18 ≈ 65.31 := 
by 
  sorry

end sal_pizza_increase_l178_178959


namespace premium_nonfiction_fifth_day_fine_l178_178314

-- Define the initial conditions
def initial_fine : ℝ := 0.05

-- Define the function to calculate fine for premium members borrowing non-fiction books
def premium_nonfiction_fine (day_fine : ℝ) : ℝ :=
  min (day_fine * 2) (day_fine + 0.15)

-- Define the total fine calculation function recursive helper
def total_fine : ℕ → ℝ → ℝ
| 1, fine := fine
| (n + 1), fine := fine + (total_fine n (premium_nonfiction_fine fine))

-- Initialize the total fine calculation from day 1
def overdue_fine (days : ℕ) : ℝ := total_fine days initial_fine

-- Define the theorem to prove the fine on the fifth day for a non-fiction book borrowed by a premium member
theorem premium_nonfiction_fifth_day_fine : 
  overdue_fine 5 = 1.35 :=
by 
  dsimp [overdue_fine, total_fine, premium_nonfiction_fine, initial_fine];
  sorry

end premium_nonfiction_fifth_day_fine_l178_178314


namespace product_of_slopes_l178_178037

theorem product_of_slopes (p : ℝ) (hp : 0 < p) :
  let T := (p, 0)
  let parabola := fun x y => y^2 = 2*p*x
  let line := fun x y => y = x - p
  -- Define intersection points A and B on the parabola satisfying the line equation
  ∃ A B : ℝ × ℝ, 
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  -- O is the origin
  let O := (0, 0)
  -- define slope function
  let slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)
  -- slopes of OA and OB
  let k_OA := slope O A
  let k_OB := slope O B
  -- product of slopes
  k_OA * k_OB = -2 := sorry

end product_of_slopes_l178_178037


namespace solve_for_y_l178_178397

noncomputable def determinant3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

noncomputable def determinant2x2 (a b c d : ℝ) : ℝ := 
  a*d - b*c

theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
  determinant3x3 (y + 2 * b) y y y (y + 2 * b) y y y (y + 2 * b) = 0 → 
  y = -b / 2 :=
by
  sorry

end solve_for_y_l178_178397


namespace side_lengths_are_10_and_50_l178_178587

-- Define variables used in the problem
variables {s t : ℕ}

-- Define the conditions
def condition1 (s t : ℕ) : Prop := 4 * s = 20 * t
def condition2 (s t : ℕ) : Prop := s + t = 60

-- Prove that given the conditions, the side lengths of the squares are 10 and 50
theorem side_lengths_are_10_and_50 (s t : ℕ) (h1 : condition1 s t) (h2 : condition2 s t) : (s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50) :=
by sorry

end side_lengths_are_10_and_50_l178_178587


namespace polynomial_difference_square_l178_178124

theorem polynomial_difference_square (a : Fin 11 → ℝ) (x : ℝ) (sqrt2 : ℝ)
  (h_eq : (sqrt2 - x)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + 
          a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10) : 
  ((a 0 + a 2 + a 4 + a 6 + a 8 + a 10)^2 - (a 1 + a 3 + a 5 + a 7 + a 9)^2 = 1) :=
by
  sorry

end polynomial_difference_square_l178_178124


namespace sections_of_mountain_l178_178348

theorem sections_of_mountain (total_eagles types_per_section : ℕ) (h1 : total_eagles = 18) (h2 : types_per_section = 6) : total_eagles / types_per_section = 3 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num)
  sorry

end sections_of_mountain_l178_178348


namespace sequence_eventually_constant_l178_178898

noncomputable def eventually_constant_sequence (a : ℕ → ℕ) : Prop :=
  ∃ M, ∀ m, M ≤ m → a m = a (m+1)

theorem sequence_eventually_constant (a : ℕ → ℕ) (N : ℕ) (hN : N > 1) 
  (h : ∀ n, N ≤ n → 
    (∑ i in Finset.range n, ((a i.succ : ℚ) / a (i : ℕ + 1))) + (a n : ℚ) / a 1 ∈ ℤ) : 
  eventually_constant_sequence a :=
sorry

end sequence_eventually_constant_l178_178898


namespace exponent_rule_l178_178242

theorem exponent_rule :
  (7 ^ (-3)) ^ 0 + (7 ^ 0) ^ 4 = 2 :=
by
  sorry

end exponent_rule_l178_178242


namespace bracelet_bead_arrangements_l178_178845

theorem bracelet_bead_arrangements : 
  let n := 6 in
  let total_permutations := Nat.factorial n in
  let rotations := n in
  let reflections := 2 in
  (total_permutations / rotations) / reflections = 60 := 
by
  let n := 6
  let total_permutations := Nat.factorial n
  let rotations := n
  let reflections := 2
  have division1 : total_permutations / rotations = total_permutations / rotations := by sorry
  have division2 : (total_permutations / rotations) / reflections = 60 := by sorry
  exact division2

end bracelet_bead_arrangements_l178_178845


namespace range_of_k_for_empty_solution_set_l178_178720

theorem range_of_k_for_empty_solution_set :
  ∀ (k : ℝ), (∀ (x : ℝ), k * x^2 - 2 * |x - 1| + 3 * k < 0 → False) ↔ k ≥ 1 :=
by sorry

end range_of_k_for_empty_solution_set_l178_178720


namespace problem_equation_correct_l178_178142

theorem problem_equation_correct :
  ∑ n in finset.Ico 1006 3017 = 2011^2 :=
by
  sorry

end problem_equation_correct_l178_178142


namespace number_of_sequences_l178_178929

/- Conditions -/
variables (m n : ℕ)
variables (m_ge_one : 1 ≤ m) (n_ge_one : 1 ≤ n)

/- Question -/
theorem number_of_sequences (m n : ℕ) (m_ge_one : 1 ≤ m) (n_ge_one : 1 ≤ n) :
  let A := { s : list ℤ // s.length = m + 2n ∧ (s.sum = m) ∧ ∀ (k : ℕ), k < m + 2n → (s.take k).sum ≥ 0 } in 
    A.card = (m * nat.choose (m + 2n - 1) n) / (m + n) :=
by {
  sorry -- Proof omitted.
}

end number_of_sequences_l178_178929


namespace jane_brown_sheets_l178_178872

theorem jane_brown_sheets :
  ∀ (total_sheets yellow_sheets brown_sheets : ℕ),
    total_sheets = 55 →
    yellow_sheets = 27 →
    brown_sheets = total_sheets - yellow_sheets →
    brown_sheets = 28 := 
by
  intros total_sheets yellow_sheets brown_sheets ht hy hb
  rw [ht, hy] at hb
  simp at hb
  exact hb

end jane_brown_sheets_l178_178872


namespace probability_of_individual_selection_l178_178615

theorem probability_of_individual_selection (population_size sample_size : ℕ) (h_population : population_size = 1001) (h_sample : sample_size = 50) :
  let probability := (sample_size : ℚ) / population_size in 
  probability = 50 / 1001 := by
  sorry

end probability_of_individual_selection_l178_178615


namespace Terrence_earns_l178_178498

theorem Terrence_earns :
  ∀ (J T E : ℝ), J + T + E = 90 ∧ J = T + 5 ∧ E = 25 → T = 30 :=
by
  intro J T E
  intro h
  obtain ⟨h₁, h₂, h₃⟩ := h
  sorry -- proof steps go here

end Terrence_earns_l178_178498


namespace min_distance_l178_178768

theorem min_distance {
  y1 := λ x : ℝ, 2 * Real.exp x,
  y2 := λ x : ℝ, Real.log (x / 2),
  dist := λ (x1 y1 x2 y2 : ℝ), Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
} : ∃ (x1 x2 : ℝ), dist x1 (y1 x1) x2 (y2 x2) = Real.sqrt 2 * (1 + Real.log 2) := 
sorry

end min_distance_l178_178768


namespace min_value_ineq_l178_178528

theorem min_value_ineq (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := 
sorry

end min_value_ineq_l178_178528


namespace slope_of_tangent_at_neg1_l178_178779

def f (x : ℝ) : ℝ := if x > 0 then (2*x - 1) * Real.log x else (2*(-x) - 1) * Real.log (-x)

theorem slope_of_tangent_at_neg1 : 
  (∀ (x : ℝ), x > 0 → f x = (2*x - 1) * Real.log x) →
  (∀ (x : ℝ), f (-x) = f x) →
  HasDerivAt f (-1) (-1) := 
by
  intro h1 h2
  sorry

end slope_of_tangent_at_neg1_l178_178779


namespace multiple_of_set_2_3_5_l178_178274

theorem multiple_of_set_2_3_5 : ∀ n ∈ {2, 3, 5}, 30 % n = 0 :=
by
  sorry

end multiple_of_set_2_3_5_l178_178274


namespace difference_between_max_and_min_34_l178_178174

theorem difference_between_max_and_min_34 
  (A B C D E: ℕ) 
  (h_avg: (A + B + C + D + E) / 5 = 50) 
  (h_max: E ≤ 58) 
  (h_distinct: A < B ∧ B < C ∧ C < D ∧ D < E) 
: E - A = 34 := 
sorry

end difference_between_max_and_min_34_l178_178174


namespace geom_seq_min_value_l178_178091

open Real

/-- 
Theorem: For a geometric sequence {a_n} where a_n > 0 and a_7 = √2/2, 
the minimum value of 1/a_3 + 2/a_11 is 4.
-/
theorem geom_seq_min_value (a : ℕ → ℝ) (a_pos : ∀ n, 0 < a n) (h7 : a 7 = (sqrt 2) / 2) :
  (1 / (a 3) + 2 / (a 11) >= 4) :=
sorry

end geom_seq_min_value_l178_178091


namespace solution_set_inequality_l178_178222

theorem solution_set_inequality (x : ℝ) (h : 0 < x ∧ x ≤ 1) : 
  ∀ (x : ℝ), (0 < x ∧ x ≤ 1 ↔ ∀ a > 0, ∀ b ≤ 1, (2/x + (1-x) ^ (1/2) ≥ 1 + (1-x)^(1/2))) := sorry

end solution_set_inequality_l178_178222


namespace _l178_178854

noncomputable theorem solve_intersection_problem 
  (a : ℝ)
  (C1 : ∀ t : ℝ, a + sqrt 2 * t = x ∧ 1 + sqrt 2 * t = y)
  (C2 : ∀ θ : ℝ, (ρ * cos θ)^2 + 4 * ρ * cos θ - ρ = 0)
  (intersect_A_B : (x - y - a + 1 = 0) ∧ (y^2 = 4 * x))
  (P : ℝ × ℝ) (a_value : ℝ) : 
  (|P.1 - A| = 2 * |P.1 - B|) → (a = 1 / 36 ∨ a = 9 / 4) :=
begin
  sorry
end

end _l178_178854


namespace smallest_possible_value_of_AP_plus_BP_l178_178517

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem smallest_possible_value_of_AP_plus_BP :
  let A := (1, 0)
  let B := (-3, 4)
  ∃ P : ℝ × ℝ, (P.2 ^ 2 = 4 * P.1) ∧
  (distance A P + distance B P = 12) :=
by
  -- proof steps would go here
  sorry

end smallest_possible_value_of_AP_plus_BP_l178_178517


namespace ball_speed_after_collision_l178_178677

-- define the given conditions
def speed_plate : ℝ := 4 -- speed of the plate relative to Earth (m/s)
def speed_ball_initial : ℝ := 5 -- speed of the ball relative to Earth before collision (m/s)
def speed_ball_relative_plate : ℝ := speed_ball_initial + speed_plate -- relative speed of the ball w.r.t the plate before collision (m/s)
def speed_collision_elastic (v : ℝ) : ℝ := -v -- perfectly elastic collision, speed after collision in opposite direction

-- final speed of the ball relative to the Earth after the collision
def speed_ball_final := (speed_collision_elastic speed_ball_relative_plate) + speed_plate

-- statement to prove
theorem ball_speed_after_collision : speed_ball_final = 13 :=
by 
  -- using the definitions
  unfold speed_ball_final speed_collision_elastic speed_ball_relative_plate speed_collision_elastic
  -- calculate using the definitions
  sorry

end ball_speed_after_collision_l178_178677


namespace greatest_integer_gcf_l178_178266

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l178_178266


namespace karen_nuts_l178_178501

/-- Karen added 0.25 cup of walnuts to a batch of trail mix.
Later, she added 0.25 cup of almonds.
In all, Karen put 0.5 cups of nuts in the trail mix. -/
theorem karen_nuts (walnuts almonds : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_almonds : almonds = 0.25) : 
  walnuts + almonds = 0.5 := 
by
  sorry

end karen_nuts_l178_178501


namespace exponent_addition_l178_178238

theorem exponent_addition : ((7:ℝ)⁻³)^0 + (7^0)^4 = 2 := by
  have h1 : (7:ℝ)⁻³ ^ 0 = 1 := by 
    rw [pow_zero, inv_pow, pow_neg, zero_pow]
    exact zero_lt_one
  have h2 : (7^0:ℝ)^4 = 1 := by 
    rw [pow_zero, one_pow, pow_one]
  rw [h1, h2]
  norm_num
  done

end exponent_addition_l178_178238


namespace birds_find_more_than_sixty_percent_millet_on_wednesday_l178_178917

-- Definitions and conditions
def initial_amount_millet : ℝ := 2 / 5
def daily_addition_millet : ℝ := 2 / 5
def daily_consumption_rate : ℝ := 0.30

-- Helper to compute millet amount after nth day
noncomputable def millet_amount (n : ℕ) : ℝ :=
  (initial_amount_millet * (1 - daily_consumption_rate)^n + daily_addition_millet * ((1 - (1 - daily_consumption_rate)^n) / (daily_consumption_rate)))

-- The day threshold for more than 60% millet in the feeder
def more_than_sixty_percent_millet_day : ℕ :=
  if h : 1 ≤ n ∧ n < 7 then
    n where
    n: ℕ := Nat.find (λ n, millet_amount n > (3 / 5))
  else 0

-- Proof problem statement
theorem birds_find_more_than_sixty_percent_millet_on_wednesday :
  more_than_sixty_percent_millet_day = 3 := 
sorry

end birds_find_more_than_sixty_percent_millet_on_wednesday_l178_178917


namespace overall_average_of_all_boys_l178_178602

theorem overall_average_of_all_boys 
  (n : ℕ) (average_pass : ℕ) (average_fail : ℕ) (num_pass : ℕ)
  (h1 : n = 120)
  (h2 : average_pass = 39)
  (h3 : average_fail = 15)
  (h4 : num_pass = 110) : 
  (39 * 110 + 15 * (120 - 110)) / 120 = 37 := 
by 
  simp [h1, h2, h3, h4]
  simp_arith
  sorry

end overall_average_of_all_boys_l178_178602


namespace a_4_eq_7_div_8_b_n_geometric_l178_178764

-- Define the sequence {a_n} and sums {S_n}
noncomputable def a_n : ℕ → ℚ
| 1 := 1
| 2 := 3 / 2
| 3 := 5 / 4
| n := if n ≥ 4 then 
    (let S : ℕ → ℚ 
        | n := if n = 1 then 1 
               else if n = 2 then 5 / 2
               else if n = 3 then 19 / 4
               else a_n n + S (n-1)
    in (8 * S (n-1) + S (n-3) - 5 * S (n-2)) / 4 - S (n-2))
    else 0

-- Part 1: Prove that a_4 = 7 / 8
theorem a_4_eq_7_div_8 : a_n 4 = 7 / 8 := sorry

-- Part 2: Prove that {a_{n+1} - 1/2 * a_n} is geometric with ratio 1/2
noncomputable def b_n (n : ℕ) : ℚ := a_n (n+1) - 1/2 * a_n n

theorem b_n_geometric : ∃ r : ℚ, r = 1/2 ∧ ∀ n, b_n (n+1) = r * b_n n := sorry

end a_4_eq_7_div_8_b_n_geometric_l178_178764


namespace lily_milk_left_l178_178907

theorem lily_milk_left (h1 : ℝ) (h2 : ℝ) : h1 - h2 = 17 / 7 :=
by
  assume (h1 : 5)
  assume (h2 : 18/7)
  sorry

end lily_milk_left_l178_178907


namespace function_decreasing_on_R_l178_178433

-- Definition of the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

-- Conditions provided in the problem
variable (a : ℝ)
def decreasing_for_x_leq_1 : Prop := a - 3 < 0
def decreasing_for_x_gt_1 : Prop := a > 0
def boundary_condition : Prop := (a - 3) * 1 + 5 ≥ 2 * a / 1

-- Lean theorem statement
theorem function_decreasing_on_R (a : ℝ) :
  decreasing_for_x_leq_1 a → decreasing_for_x_gt_1 a → boundary_condition a → 0 < a ∧ a ≤ 2 := sorry

end function_decreasing_on_R_l178_178433


namespace sum_of_squares_of_roots_l178_178712

theorem sum_of_squares_of_roots (p q : ℝ) :
  let r1 r2 := (-p / 2 + Real.sqrt ((p / 2)^2 - q)), (-p / 2 - Real.sqrt ((p / 2)^2 - q))
  r1 ^ 2 + r2 ^ 2 = p ^ 2 - 2 * q :=
by 
  -- Definitions and conditions
  let r1 := -p / 2 + Real.sqrt ((p / 2)^2 - q)
  let r2 := -p / 2 - Real.sqrt ((p / 2)^2 - q)
  -- Calculation
  sorry

end sum_of_squares_of_roots_l178_178712


namespace intersection_complement_l178_178799

def U : set ℕ := {0, 1, 2, 3, 4}
def M : set ℕ := {0, 1, 2}
def N : set ℕ := {2, 3}

theorem intersection_complement (U M N : set ℕ)
  (hU : U = {0, 1, 2, 3, 4})
  (hM : M = {0, 1, 2})
  (hN : N = {2, 3}) :
  (U \ M) ∩ N = {3} :=
by 
  sorry

end intersection_complement_l178_178799


namespace most_numerous_difficulty_sets_l178_178487

-- Definitions based on the conditions
def is_set (a b c : ℤ) : Prop :=
  ∀i ∈ [0, 1, 2, 3], 
    ((a.digits (i) = b.digits (i)) ∧ (b.digits (i) = c.digits (i)))
    ∨ (a.digits (i) ≠ b.digits (i) ∧ b.digits (i) ≠ c.digits (i) ∧ c.digits (i) ≠ a.digits (i))

def difficulty (a b c : ℤ) : ℕ :=
  (List.range 4).count (λ i, a.digits (i) ≠ b.digits (i) ∧ b.digits (i) ≠ c.digits (i) ∧ c.digits (i) ≠ a.digits (i))

-- Main theorem to prove
theorem most_numerous_difficulty_sets : ∀ (n1 n2 n3 : ℤ),
  (∀i ∈ [0, 1, 2, 3], n1.digits (i) ∈ [1, 2, 3] ∧ n2.digits (i) ∈ [1, 2, 3] ∧ n3.digits (i) ∈ [1, 2, 3]) →
  difficulty n1 n2 n3 = 3 →
  sorry

end most_numerous_difficulty_sets_l178_178487


namespace p_sufficient_but_not_necessary_for_q_l178_178411

variables {x : ℝ}

def p : Prop := x > 2
def q : Prop := x > 0

theorem p_sufficient_but_not_necessary_for_q :
  (p → q) ∧ (¬ (q → p)) :=
by 
  sorry

end p_sufficient_but_not_necessary_for_q_l178_178411


namespace largest_defined_g_l178_178118

noncomputable def g1 (x : ℝ) : Option ℝ :=
if |x| < 1 then some (1 / (3 * (1 - x))) else none

noncomputable def gn : ℕ → ℝ → Option ℝ
| 1 := g1
| (n+1) := λ x, g1 (gn n x).get_or_else 0

theorem largest_defined_g (r : ℕ) : r = 5 :=
begin
  sorry
end

end largest_defined_g_l178_178118


namespace greatest_integer_gcf_l178_178267

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l178_178267


namespace maggie_remaining_goldfish_l178_178536

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end maggie_remaining_goldfish_l178_178536


namespace lines_intersect_iff_R_one_and_odd_number_on_sides_l178_178748

theorem lines_intersect_iff_R_one_and_odd_number_on_sides 
  (A B C A1 B1 C1 : Point) 
  (AA1 BB1 CC1 : Line) 
  (R : ℝ) 
  (odd_on_sides : Odd (number_of_points_on_sides [A1, B1, C1] [A, B, C])) :
  (lines_intersect AA1 BB1 CC1) ↔ (R = 1 ∧ odd_on_sides) := by
  sorry

end lines_intersect_iff_R_one_and_odd_number_on_sides_l178_178748


namespace integral_eval_l178_178726

open Real

theorem integral_eval : ∫ x in 1..Real.exp 1, (1/x + x) = (1/2) * Real.exp 2 + 1/2 :=
by
  -- Proof steps will go here.
  sorry

end integral_eval_l178_178726


namespace cost_of_siding_l178_178561

theorem cost_of_siding 
  (wall_width wall_height roof_face_base roof_face_height section_width section_height : ℝ)
  (section_cost : ℝ) : 
  wall_width = 10 → 
  wall_height = 7 → 
  roof_face_base = 10 → 
  roof_face_height = 6 → 
  section_width = 10 → 
  section_height = 15 → 
  section_cost = 35 → 
  (let wall_area := wall_width * wall_height in
   let roof_face_area := 0.5 * roof_face_base * roof_face_height in
   let total_area := wall_area + (roof_face_area * 2) in
   let sections_needed := ceil (total_area / (section_width * section_height)) in
   sections_needed * section_cost = 35) := 
by 
  intros _ _ _ _ _ _ _;
  sorry

end cost_of_siding_l178_178561


namespace inequality_proof_l178_178998

theorem inequality_proof (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 → x > 5 / 7 := 
by
  sorry

end inequality_proof_l178_178998


namespace greatest_valid_number_l178_178262

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l178_178262


namespace palindrome_word_count_l178_178237

theorem palindrome_word_count (X : List Char) (hX_length : X.length = 2014) (hX_letters : ∀ l, l ∈ X → l = 'A' ∨ l = 'B') : 
  ∃ (P : List (List Char)), (∀ p ∈ P, (∀ i j, i + j + 1 = p.length → p.nth i = p.nth (j))) ∧ (P.join = X) ∧ (P.length ≥ 806) := 
sorry

end palindrome_word_count_l178_178237


namespace alex_needs_more_coins_l178_178693

-- Define the conditions and problem statement 
def num_friends : ℕ := 15
def coins_alex_has : ℕ := 95 

-- The total number of coins required is
def total_coins_needed : ℕ := num_friends * (num_friends + 1) / 2

-- The minimum number of additional coins needed
def additional_coins_needed : ℕ := total_coins_needed - coins_alex_has

-- Formalize the theorem 
theorem alex_needs_more_coins : additional_coins_needed = 25 := by
  -- Here we would provide the actual proof steps
  sorry

end alex_needs_more_coins_l178_178693


namespace find_lesser_fraction_l178_178997

theorem find_lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 := 
by 
  sorry

end find_lesser_fraction_l178_178997


namespace pentagon_area_l178_178351

noncomputable def area_pentagon (AE EF BC CF : ℕ) : ℕ :=
  let area_AEF := (1/2 : ℚ) * AE * EF
  let average_bases := (BC + EF) / 2
  let area_BCEF := CF * average_bases
  area_AEF.to_nat + area_BCEF.to_nat

theorem pentagon_area
  (AE EF BC CF : ℕ)
  (hAE : AE = 17)
  (hEF : EF = 22)
  (hBC : BC = 26)
  (hCF : CF = 30) :
  area_pentagon AE EF BC CF = 907 :=
by
  rw [hAE, hEF, hBC, hCF]
  simp [area_pentagon]
  sorry

end pentagon_area_l178_178351


namespace price_of_regular_ticket_test_l178_178540

theorem price_of_regular_ticket (total_savings : ℝ) (price_per_vip : ℝ) (num_vip_tickets : ℕ)
                                (num_regular_tickets : ℕ) (amount_left : ℝ) 
                                (spent_on_tickets := total_savings - amount_left)
                                (total_vip_cost := num_vip_tickets * price_per_vip)
                                (remaining_for_regular := spent_on_tickets - total_vip_cost) :
  price_per_regular_ticket num_regular_tickets remaining_for_regular = remaining_for_regular / num_regular_tickets := 
by
  sorry

# Given values
def total_savings : ℝ := 500
def price_per_vip : ℝ := 100
def num_vip_tickets : ℕ := 2
def num_regular_tickets : ℕ := 3
def amount_left : ℝ := 150

# Calculated helper values that derive from the conditions
def spent_on_tickets : ℝ := total_savings - amount_left
def total_vip_cost : ℝ := num_vip_tickets * price_per_vip
def remaining_for_regular : ℝ := spent_on_tickets - total_vip_cost

# The price of each regular ticket should be $50
def price_per_regular_ticket : ℝ := remaining_for_regular / num_regular_tickets

theorem test : price_per_regular_ticket = 50 :=
by
  have h1 : spent_on_tickets = 350 := by simp [spent_on_tickets, total_savings, amount_left]
  have h2 : total_vip_cost = 200 := by simp [total_vip_cost, num_vip_tickets, price_per_vip]
  have h3 : remaining_for_regular = 150 := by simp [remaining_for_regular, spent_on_tickets, total_vip_cost, h1, h2]
  have h4 : price_per_regular_ticket = 50 := by simp [price_per_regular_ticket, remaining_for_regular, num_regular_tickets, h3]
  exact h4

end price_of_regular_ticket_test_l178_178540


namespace solve_system_eqns_l178_178571

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end solve_system_eqns_l178_178571


namespace coin_toss_is_random_l178_178280

theorem coin_toss_is_random : ∀ (coin : Type) (toss : coin → Prop), (∀ c : coin, c = heads ∨ c = tails) → (∀ c : coin, ¬ (toss c = heads ∧ toss c = tails)) → (toss heads ↔ ¬ toss tails) → random (toss heads) :=
by
  intro coin toss outcomes fair prob_independence
  sorry

end coin_toss_is_random_l178_178280


namespace cover_2x10_grid_l178_178363

def f : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 2
| (n + 1) := f n + f (n - 1)

theorem cover_2x10_grid : f 10 = 89 :=
by {
  -- Initial conditions
  have h1 : f 1 = 1 := rfl,
  have h2 : f 2 = 2 := rfl,
  -- Recursive relation
  assume h3 : ∀ n ≥ 3, f n = f (n - 1) + f (n - 2),
  
  -- Applying the recursive relation step by step:
  have h3_3 : f 3 = f 2 + f 1 := by rw [h3, nat.succ_sub_succ_eq_sub, nat.sub_zero]; rw [h2, h1]; refl,
  have h4 : f 4 = f 3 + f 2 := by rw [h3 4 (dec_trivial:4 ≥ 3), h3_3, h2]; refl,
  have h5 : f 5 = f 4 + f 3 := by rw [h3 5 (dec_trivial:5 ≥ 3), h4, h3_3]; refl,
  have h6 : f 6 = f 5 + f 4 := by rw [h3 6 (dec_trivial:6 ≥ 3), h5, h4]; refl,
  have h7 : f 7 = f 6 + f 5 := by rw [h3 7 (dec_trivial:7 ≥ 3), h6, h5]; refl,
  have h8 : f 8 = f 7 + f 6 := by rw [h3 8 (dec_trivial:8 ≥ 3), h7, h6]; refl,
  have h9 : f 9 = f 8 + f 7 := by rw [h3 9 (dec_trivial:9 ≥ 3), h8, h7]; refl,
  have h10 : f 10 = f 9 + f 8 := by rw [h3 10 (dec_trivial:10 ≥ 3), h9, h8]; refl,

  -- Show final result
  exact h10,
}

end cover_2x10_grid_l178_178363


namespace f_at_3_f_range_l178_178033

def g (x : ℝ) : ℝ := x^2 - 2

def f (x : ℝ) : ℝ :=
if x < g x then g x + x + 4 else g x - x

theorem f_at_3 : f 3 = 14 :=
by 
  sorry

theorem f_range : 
  set.range f = set.Icc (-9/4) 0 ∪ set.Ioi 2 :=
by
  sorry

end f_at_3_f_range_l178_178033


namespace contrapositive_of_inequality_l178_178180

variable {a b c : ℝ}

theorem contrapositive_of_inequality (h : a + c ≤ b + c) : a ≤ b :=
sorry

end contrapositive_of_inequality_l178_178180


namespace greatest_integer_with_gcd_30_eq_5_l178_178254

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l178_178254


namespace solve_inequality_l178_178740

theorem solve_inequality (x : ℝ) (h : |2 * x + 6| < 10) : -8 < x ∧ x < 2 :=
sorry

end solve_inequality_l178_178740


namespace solve_ticket_problem_l178_178604

def ticket_problem : Prop :=
  ∃ S N : ℕ, S + N = 2000 ∧ 9 * S + 11 * N = 20960 ∧ S = 520

theorem solve_ticket_problem : ticket_problem :=
sorry

end solve_ticket_problem_l178_178604


namespace field_day_difference_l178_178489

theorem field_day_difference :
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  total_boys - total_girls = 2 :=
by
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  have h1 : total_girls = 46 := rfl
  have h2 : total_boys = 48 := rfl
  have h3 : total_boys - total_girls = 2 := rfl
  exact h3

end field_day_difference_l178_178489


namespace unique_decomposition_in_base_b_l178_178520

theorem unique_decomposition_in_base_b (b : ℕ) (hb : b ≥ 2) :
  ∀ n : ℕ, ∃! (digits : ℕ → ℕ), (∀ i, digits i < b) ∧ n = ∑ i in Finset.range (n.bitsize), digits i * b^i :=
sorry

end unique_decomposition_in_base_b_l178_178520


namespace smallest_S_achievable_l178_178298

noncomputable def smallest_nonnegative_S : ℕ := 1

theorem smallest_S_achievable 
  (x : Fin 2022 → ℂ)
  (h1 : ∀ i, x i = 1 ∨ x i = -complex.I) : 
  let S := (∑ i, x i * x (i + 1) * x (i + 2)) 
            in S = smallest_nonnegative_S ∨ S ≥ smallest_nonnegative_S :=
sorry

end smallest_S_achievable_l178_178298


namespace increasing_sequence_and_bounded_l178_178393

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 23 * x) / (7 * (x - 5))

axiom initial_condition (x1 : ℝ) : 8 < x1 ∧ x1 < 12

def sequence (x : ℝ) (n : ℕ) : ℝ
  | 0     => x
  | n + 1 => f (sequence x n)

theorem increasing_sequence_and_bounded (x1 : ℝ) 
  (h1 : 8 < x1) (h2 : x1 < 12) :
  ∀ n : ℕ, (sequence x1 n) < (sequence x1 (n + 1)) ∧ (sequence x1 n) < 12 :=
sorry

end increasing_sequence_and_bounded_l178_178393


namespace road_length_l178_178924

theorem road_length (total_streetlights : ℕ) (streetlight_interval : ℕ) (total_streetlights = 120) (streetlight_interval = 10) : ∃ road_length : ℕ, road_length = 590 :=
by
  sorry

end road_length_l178_178924


namespace a_5_eq_14_l178_178406

def S (n : ℕ) : ℚ := (3 / 2) * n ^ 2 + (1 / 2) * n

def a (n : ℕ) : ℚ := S n - S (n - 1)

theorem a_5_eq_14 : a 5 = 14 := by {
  -- Proof steps go here
  sorry
}

end a_5_eq_14_l178_178406


namespace clothing_store_revenue_l178_178655

theorem clothing_store_revenue :
    let shirt_price := 10
    let jeans_price := 2 * shirt_price
    let jacket_price := 3 * jeans_price
    let socks_price := 2

    let shirts_sold := 20
    let jeans_sold := 10
    let jackets_sold := 15
    let socks_sold := 30

    let discount_jacket := 0.10
    let discount_bulk_socks := 0.20

    let revenue_shirts := (shirts_sold / 2) * shirt_price
    let revenue_jeans := jeans_sold * jeans_price
    let revenue_jackets := jackets_sold * (jacket_price * (1 - discount_jacket))
    let revenue_socks := socks_sold * (socks_price * (1 - discount_bulk_socks))

    let total_revenue := revenue_shirts + revenue_jeans + revenue_jackets + revenue_socks
in
    total_revenue = 1158 := 
begin
  sorry
end

end clothing_store_revenue_l178_178655


namespace decomposition_x_pqr_l178_178286

-- Definitions of vectors x, p, q, r
def x : ℝ := sorry
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- The linear combination we want to prove
theorem decomposition_x_pqr : 
  (x = -1 • p + 4 • q + 3 • r) :=
sorry

end decomposition_x_pqr_l178_178286


namespace rounding_theorem_l178_178704

def a : ℝ := 5739204.742
def b : ℝ := -176817.835

def s : ℝ := a + b
def r : ℤ := Int.nearest s

theorem rounding_theorem : r = 5562387 := 
by 
  -- proof steps would go here
  sorry

end rounding_theorem_l178_178704


namespace Mika_used_58_stickers_l178_178916

theorem Mika_used_58_stickers :
  let initial_stickers := 20
  let bought_stickers := 26
  let birthday_stickers := 20
  let given_away_stickers := 6
  let left_stickers := 2
  let total_stickers := initial_stickers + bought_stickers + birthday_stickers
  let after_given_away := total_stickers - given_away_stickers
  after_given_away - left_stickers = 58 :=
by {
  -- Definitions based on the problem conditions
  let initial_stickers := 20
  let bought_stickers := 26
  let birthday_stickers := 20
  let given_away_stickers := 6
  let left_stickers := 2
  let total_stickers := initial_stickers + bought_stickers + birthday_stickers
  let after_given_away := total_stickers - given_away_stickers

  -- The actual proof (not provided here, placeholder for the logical steps):
  show (after_given_away - left_stickers = 58), from sorry
}

end Mika_used_58_stickers_l178_178916


namespace complex_magnitude_product_l178_178725

theorem complex_magnitude_product : 
  | Complex.mk (Real.sqrt 8) (-2) * Complex.mk (2 * Real.sqrt 3) 6 | = 24 := by
sorry

end complex_magnitude_product_l178_178725


namespace flight_time_l178_178992

noncomputable def radius := 5000  -- radius of Earth at the equator in miles
noncomputable def speed1 := 600   -- speed for the first half in miles per hour
noncomputable def speed2 := 800   -- speed for the second half in miles per hour
noncomputable def pi_approx := 3.14  -- approximation of pi

theorem flight_time (radius speed1 speed2 : ℝ) (pi_approx : ℝ) :
  let circumference := 2 * pi * radius
      half_distance := circumference / 2
      T1 := half_distance / speed1
      T2 := half_distance / speed2
      total_time := T1 + T2
  in total_time ≈ 45.79 := 
by {
  sorry,
}

end flight_time_l178_178992


namespace juan_speed_l178_178875

-- Statement of given distances and time
def distance : ℕ := 80
def time : ℕ := 8

-- Desired speed in miles per hour
def expected_speed : ℕ := 10

-- Theorem statement: Speed is distance divided by time and should equal 10 miles per hour
theorem juan_speed : distance / time = expected_speed :=
  by
  sorry

end juan_speed_l178_178875


namespace min_max_f_l178_178200

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f : 
  let min_val := - (3 * Real.pi) / 2 in
  let max_val := (Real.pi / 2) + 2 in
  ∃ x_min ∈ Set.Icc 0 (2 * Real.pi), f x_min = min_val ∧
  ∃ x_max ∈ Set.Icc 0 (2 * Real.pi), f x_max = max_val :=
sorry

end min_max_f_l178_178200


namespace sum_of_squares_leq_l178_178564

theorem sum_of_squares_leq (n : ℕ) (hn : 0 < n) :
  (∑ i in Finset.range n, 1 / (i + 1) ^ 2) ≤ 2 - 1 / n :=
sorry

end sum_of_squares_leq_l178_178564


namespace greatest_valid_number_l178_178259

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l178_178259


namespace find_y_intercept_l178_178684

theorem find_y_intercept (m b x y : ℝ) (h1 : m = 2) (h2 : (x, y) = (239, 480)) (line_eq : y = m * x + b) : b = 2 :=
by
  sorry

end find_y_intercept_l178_178684


namespace votes_is_240_l178_178158

variable (votes : ℕ) (likes : ℕ) (dislikes : ℕ)

# Conditions
def score (likes dislikes : ℕ) : ℤ := likes - dislikes
def percentage_likes (votes : ℕ) : ℚ := (3 / 4 : ℚ) * votes

theorem votes_is_240 (h_score : score likes dislikes = 120) (h_percentage_likes : percentage_likes votes = likes) : votes = 240 :=
sorry

end votes_is_240_l178_178158


namespace product_of_fractions_l178_178271

theorem product_of_fractions :
  (1/3 : ℚ) * (2/5) * (3/7) * (4/8) = 1/35 := 
by 
  sorry

end product_of_fractions_l178_178271


namespace star_three_four_eq_zero_l178_178820

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - 2 * a * b

theorem star_three_four_eq_zero : star 3 4 = 0 := sorry

end star_three_four_eq_zero_l178_178820


namespace pyramid_surface_area_correct_l178_178995

noncomputable def pyramid_surface_area (a m n : ℝ) : ℝ :=
  (a^2 * real.sqrt 3 / 4) * (1 + real.sqrt (3 * (m + 2 * n) / m))

theorem pyramid_surface_area_correct (a m n : ℝ) (h_m : 0 < m) (h_n : 0 < n) :
  pyramid_surface_area a m n = (a^2 * real.sqrt 3 / 4) * (1 + real.sqrt (3 * (m + 2 * n) / m)) :=
by
  rw [pyramid_surface_area]
  norm_num
  sorry

end pyramid_surface_area_correct_l178_178995


namespace inequality_solution_set_range_of_t_l178_178713

def f (x : ℝ) := abs (x + 2) - abs (x - 2)
def g (x : ℝ) := x + 1/2

-- First part
theorem inequality_solution_set :
  {x : ℝ | f(x) ≥ g(x)} = {x : ℝ | x ≤ -9/2} ∪ {x : ℝ | 1/2 ≤ x ∧ x ≤ 7/2} :=
by
  sorry

-- Second part
theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f(x) ≥ t^2 - 5*t) ↔ 1 ≤ t ∧ t ≤ 4 :=
by
  sorry

end inequality_solution_set_range_of_t_l178_178713


namespace lambda_value_l178_178923

theorem lambda_value (s : ℝ) (λ : ℝ) (E_on_CD : E ∈ segment CD)
  (perimeter_triangle : (ABCD.perimeter) = 4 * s)
  (ratio_CE_CD : CE / CD = λ)
  (perimeter_BCE : BC + CE + BE = 3 * s) :
  960 * λ = 720 := sorry

end lambda_value_l178_178923


namespace find_a_l178_178064

theorem find_a (a x : ℝ) 
  (h : x^2 + 3 * x + a = (x + 1) * (x + 2)) : 
  a = 2 :=
sorry

end find_a_l178_178064


namespace mosaic_perimeter_is_correct_l178_178166

noncomputable def side_length := 20 -- The side length of the hexagon, squares, and triangles in cm

def perimeter_of_mosaic (hexagon_side_length : ℕ) : ℕ :=
  let square_perimeter := 6 * hexagon_side_length
  let triangle_perimeter := 6 * hexagon_side_length
  square_perimeter + triangle_perimeter

theorem mosaic_perimeter_is_correct : perimeter_of_mosaic side_length = 240 :=
by
  -- Each side of the hexagon, squares, and triangles is 20 cm
  let hex_side := side_length
  -- Calculate the contribution of squares and triangles to the perimeter
  let square_perimeter := 6 * hex_side
  let triangle_perimeter := 6 * hex_side
  -- Total perimeter calculation
  have calc_perimeter : ℕ := square_perimeter + triangle_perimeter
  show calc_perimeter = 240
  sorry

end mosaic_perimeter_is_correct_l178_178166


namespace find_formula_for_f_constant_triangle_area_l178_178530

variable (a b : ℝ)

def f (x : ℝ) := a * x - b / x
def tangent_line_equation (x : ℝ) (y : ℝ) := 7 * x - 4 * y - 12 = 0

theorem find_formula_for_f (h : tangent_line_equation 2 (f 2)) :
  f = λ x, x - 3 / x :=
sorry

theorem constant_triangle_area (x_0 : ℝ) (hx_0 : x_0 ≠ 0) :
  let y := x_0 - 3 / x_0,
      dy_dx := 1 + 3 / x_0^2,
      intersection_x := 0,
      intersection_y := -6 / x_0,
      intersection_xy := (2 * x_0, 2 * x_0)
  in 1/2 * abs (-6 / x_0) * abs (2 * x_0) = 6 :=
sorry

end find_formula_for_f_constant_triangle_area_l178_178530


namespace foldable_to_missing_faces_cube_l178_178356

/-
  Given a cross-shaped figure composed of five congruent squares, one in the center and one extending 
  from each side of the center square, and the possibility of attaching an additional congruent square 
  to one of twelve possible positions around the perimeter of the cross, prove that all twelve resulting 
  polygons can be folded to form a cube with one face missing.
-/

theorem foldable_to_missing_faces_cube : 
  ∀ (base_figure : set (ℤ × ℤ)) (attach_positions : set (ℤ × ℤ)),
  cross_shaped_figure base_figure ∧ (attach_positions ⊆ {pos | pos ∈ perimeter_positions base_figure}) →
  positions_count attach_positions = 12 →
  ∀ (pos ∈ attach_positions), foldable_with_attachment_to_cube base_figure pos :=
sorry

end foldable_to_missing_faces_cube_l178_178356


namespace factorial_trailing_zeros_remainder_l178_178884

def count_factors_of_five (n : ℕ) : ℕ :=
  if n < 5 then 0 else n / 5 + count_factors_of_five (n / 5)

def product_factors_of_five (n : ℕ) : ℕ :=
  (List.range n).sum (λ i, count_factors_of_five (i + 1))

theorem factorial_trailing_zeros_remainder :
  (product_factors_of_five 100) % 1000 = 54 :=
by
  sorry

end factorial_trailing_zeros_remainder_l178_178884


namespace no_solution_fraction_eq_l178_178967

theorem no_solution_fraction_eq (x : ℝ) : 
  (1 / (x - 2) = (1 - x) / (2 - x) - 3) → False := 
by 
  sorry

end no_solution_fraction_eq_l178_178967


namespace min_max_f_l178_178197

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f : 
  let min_val := - (3 * Real.pi) / 2 in
  let max_val := (Real.pi / 2) + 2 in
  ∃ x_min ∈ Set.Icc 0 (2 * Real.pi), f x_min = min_val ∧
  ∃ x_max ∈ Set.Icc 0 (2 * Real.pi), f x_max = max_val :=
sorry

end min_max_f_l178_178197


namespace cannot_be_sum_of_consecutive_nat_iff_power_of_two_l178_178281

theorem cannot_be_sum_of_consecutive_nat_iff_power_of_two (n : ℕ) : 
  (∀ a b : ℕ, n ≠ (b - a + 1) * (a + b) / 2) ↔ (∃ k : ℕ, n = 2 ^ k) := by
  sorry

end cannot_be_sum_of_consecutive_nat_iff_power_of_two_l178_178281


namespace largest_angle_degree_cos_B_value_l178_178765

theorem largest_angle_degree (a b c A B C : ℝ) (h1 : a, b, c form_arithmetic_sequence) (h2 : sin A / sin B = 3 / 5) :
  ∠ABC = 120° := sorry

theorem cos_B_value (a b c : ℝ) (h : b^2 - (a - c)^2 = vector dot (BA, BC)) :
  cos B = 2 / 3 := sorry

end largest_angle_degree_cos_B_value_l178_178765


namespace find_prime_factor_sum_l178_178895

noncomputable def calculate (x y : ℕ) : ℕ := 
  let m := prime_factors_count x
  let n := prime_factors_count y
  2 * m + 3 * n

theorem find_prime_factor_sum (x y : ℕ) (hx_pos: x > 0) (hy_pos: y > 0)
(h1 : 2 * (Real.log x / Real.log 10) + 3 * (Real.log (Nat.gcd x y) / Real.log 10) = 80)
(h2 : (Real.log y / Real.log 10) + 3 * (Real.log (Nat.lcm x y) / Real.log 10) = 582) :
  calculate x y = 880 :=
sorry

-- Auxiliary function to count total (not necessarily distinct) prime factors of n
noncomputable def prime_factors_count (n : ℕ) : ℕ :=
  (n.factorization.fold 0 (λ _ exp acc, acc + exp))

end find_prime_factor_sum_l178_178895


namespace position_after_steps_l178_178544

def equally_spaced_steps (total_distance num_steps distance_per_step steps_taken : ℕ) : Prop :=
  total_distance = num_steps * distance_per_step ∧ 
  ∀ k : ℕ, k ≤ num_steps → k * distance_per_step = distance_per_step * k

theorem position_after_steps (total_distance num_steps distance_per_step steps_taken : ℕ) 
  (h_eq : equally_spaced_steps total_distance num_steps distance_per_step steps_taken) 
  (h_total : total_distance = 32) (h_num : num_steps = 8) (h_steps : steps_taken = 6) : 
  steps_taken * (total_distance / num_steps) = 24 := 
by 
  sorry

end position_after_steps_l178_178544


namespace f_g_2_value_l178_178072

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2^(-x) - 2 else g(x)

-- g(x) is assumed to be a function from ℝ to ℝ.
noncomputable def g (x : ℝ) : ℝ := sorry

lemma f_is_odd (x : ℝ) : f(-x) = -f(x) := sorry

-- The main statement to prove 
theorem f_g_2_value : f(g(2)) = -(2^(g(2)) - 2) :=
by {
  have odd_f := f_is_odd,
  sorry
}

end f_g_2_value_l178_178072


namespace number_of_solutions_congruence_l178_178886

theorem number_of_solutions_congruence (p : ℕ) [Fact p.prime] : 
  let n := p \left(p + (-1)^( (p+1) / 2 )\right) 
  in (card { (x, y, z) : ℤ × ℤ × ℤ | (x^2 + y^2 + z^2 + 1) % p = 0 } = n) := 
sorry

end number_of_solutions_congruence_l178_178886


namespace simplify_expression_l178_178703

variable (x y : ℝ)

theorem simplify_expression : 
  [2 * x - (3 * y - (2 * x + 1))] - [(3 * y - (2 * x + 1)) - 2 * x] = 8 * x - 6 * y + 2 :=
by
  sorry

end simplify_expression_l178_178703


namespace probability_sum_of_two_4_sided_dice_equals_single_6_sided_die_roll_l178_178878

  theorem probability_sum_of_two_4_sided_dice_equals_single_6_sided_die_roll :
    (∃ (k : ℕ), k ∈ {1, 2, 3, 4, 5, 6} ∧ 
    (k = 2 ∧ (1, 1) ∈ {(a, b) | a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4}}) ∨
    (k = 3 ∧ (∃ (a b : ℕ), (a, b) ∈ {(1, 2), (2, 1)})) ∨
    (k = 4 ∧ (∃ (a b : ℕ), (a, b) ∈ {(1, 3), (2, 2), (3, 1)})) ∨
    (k = 5 ∧ (∃ (a b : ℕ), (a, b) ∈ {(1, 4), (2, 3), (3, 2), (4, 1)})) ∨
    (k = 6 ∧ (∃ (a b : ℕ), (a, b) ∈ {(2, 4), (3, 3), (4, 2)})) ∨
    (k = 7 ∧ (∃ (a b : ℕ), (a, b) ∈ {(3, 4), (4, 3)})) ∨
    (k = 8 ∧ (4, 4) ∈ {(4, 4)})) ∧
    (1 / 6) * (ways_to_sum k / 16) = 5 / 48) :=
  sorry
  
  def ways_to_sum (n : ℕ) : ℕ :=
    match n with
    | 2 => 1
    | 3 => 2
    | 4 => 3
    | 5 => 4
    | 6 => 3
    | 7 => 2
    | 8 => 1
    | _ => 0
  
end probability_sum_of_two_4_sided_dice_equals_single_6_sided_die_roll_l178_178878


namespace price_per_pound_of_steaks_l178_178495

theorem price_per_pound_of_steaks (buy_get_free : ∀ (x : ℕ), x / 2)
                                   (bought_pounds : ℕ) 
                                   (paid_amount : ℝ) : ℝ :=
by 
  let effective_pounds := bought_pounds / 2
  have price_per_pound := paid_amount / effective_pounds
  exact price_per_pound

example : price_per_pound_of_steaks (λ x, x / 2) 20 150 = 15 := sorry

end price_per_pound_of_steaks_l178_178495


namespace mike_finished_second_l178_178835

-- Definitions of participants and their relationships.
variable (positions : ℕ → ℕ)
variable h1 : positions 7 + 1 = positions 6 -- Mike finished 7 places ahead of Pete.
variable h2 : positions 5 + 1 = positions 2 -- Leah finished 2 places ahead of Nora.
variable h3 : positions 6 + 1 = positions 4 -- Pete finished 3 places behind Oliver.
variable h4 : positions 3 + 1 = positions 2 -- Oliver finished 2 places behind Nora.
variable h5 : positions 2 = 4 -- Nora finished in 4th place.

-- Prove that Mike finished in 2nd place.
theorem mike_finished_second : positions 7 = 2 := sorry

end mike_finished_second_l178_178835


namespace production_inequalities_minimum_production_cost_l178_178656

theorem production_inequalities (x : ℕ) (y : ℕ) :
  (9 * x + 4 * (50 - x) ≤ 360) ∧ (3 * x + 10 * (50 - x) ≤ 290) ∧ (30 ≤ x) ∧ (x ≤ 32) :=
sorry

theorem minimum_production_cost (x : ℕ) :
  (30 ≤ x)  ∧ (x ≤ 32) → -20 * x + 4500 = 3860 :=
sorry

end production_inequalities_minimum_production_cost_l178_178656


namespace start_cities_l178_178049

structure City := (name : String)

structure Ticket := (from : City) (to : City)

def cities : List City :=
  [ { name := "Saint Petersburg" }
  , { name := "Tver" }
  , { name := "Yaroslavl" }
  , { name := "Nizhny Novgorod" }
  , { name := "Moscow" }
  , { name := "Kazan" }
  ]

def tickets : List Ticket :=
  [ { from := { name := "Saint Petersburg" }, to := { name := "Tver" } }
  , { from := { name := "Yaroslavl" }, to := { name := "Nizhny Novgorod" } }
  , { from := { name := "Moscow" }, to := { name := "Kazan" } }
  , { from := { name := "Nizhny Novgorod" }, to := { name := "Kazan" } }
  , { from := { name := "Moscow" }, to := { name := "Tver" } }
  , { from := { name := "Moscow" }, to := { name := "Nizhny Novgorod" } }
  ]

def odd_degree_cities (tickets : List Ticket) : List City := sorry

noncomputable def start_options : List City :=
  odd_degree_cities tickets

theorem start_cities :
  start_options = [ { name := "Saint Petersburg" }, { name := "Yaroslavl" }] :=
sorry

end start_cities_l178_178049


namespace truncated_cone_volume_is_correct_l178_178707

noncomputable def truncated_cone_volume (dL hL dS hS : ℝ) : ℝ :=
  let rL := dL / 2    -- radius of larger cone
  let rS := dS / 2    -- radius of smaller cone
  let vL := (1/3) * π * rL^2 * hL    -- volume of larger cone
  let vS := (1/3) * π * rS^2 * hS    -- volume of smaller cone
  vL - vS

theorem truncated_cone_volume_is_correct :
  truncated_cone_volume 8 10 4 4 = 48 * π :=
by
  sorry

end truncated_cone_volume_is_correct_l178_178707


namespace tracy_initial_candies_l178_178607

variable (x y : ℕ) (h1 : 2 ≤ y) (h2 : y ≤ 6)

theorem tracy_initial_candies :
  (x - (1/5 : ℚ) * x = (4/5 : ℚ) * x) ∧
  ((4/5 : ℚ) * x - (1/3 : ℚ) * (4/5 : ℚ) * x = (8/15 : ℚ) * x) ∧
  y - 10 * 2 + ((8/15 : ℚ) * x - 20) = 5 →
  x = 60 :=
by
  sorry

end tracy_initial_candies_l178_178607


namespace cubes_sum_eighteen_l178_178906

theorem cubes_sum_eighteen (a b c d e : ℕ) 
  (h1 : a ∈ {0, 1, 2})
  (h2 : b ∈ {0, 1, 2})
  (h3 : c ∈ {0, 1, 2})
  (h4 : d ∈ {0, 1, 2})
  (h5 : e ∈ {0, 1, 2})
  (h_sum : a + b + c + d + e = 6)
  (h_squares_sum : a^2 + b^2 + c^2 + d^2 + e^2 = 10) :
  a^3 + b^3 + c^3 + d^3 + e^3 = 18 := 
sorry

end cubes_sum_eighteen_l178_178906


namespace identity_verification_l178_178951

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l178_178951


namespace range_of_f_gt_1_l178_178827

theorem range_of_f_gt_1 :
  ∀ (f g : ℝ → ℝ), (∀ x, g x = 2 ^ x) →
  (∀ x, f x = g (-x)) →
  {x : ℝ | f x > 1} = Iio 0 :=
by
  intros f g hg hf
  sorry

end range_of_f_gt_1_l178_178827


namespace base_n_representation_of_b_l178_178067

theorem base_n_representation_of_b (n m b : ℤ) :
  n > 9 ∧ ∃ (x y : ℤ), (2 * x - y = 6) ∧ (x^2 - (2 * n + 7) * x + b = 0) ∧
  (x = n ∧ y = m) → (b = n * m) ∧ (ltn_repr b = 10) :=
by 
  sorry

end base_n_representation_of_b_l178_178067


namespace jack_sugar_amount_l178_178871

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l178_178871


namespace problem_a_problem_b_l178_178147

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end problem_a_problem_b_l178_178147


namespace probability_of_rain_l178_178727

-- Define the conditions in Lean
variables (x : ℝ) -- probability of rain

-- Known condition: taking an umbrella 20% of the time
def takes_umbrella : Prop := 0.2 = x + ((1 - x) * x)

-- The desired problem statement
theorem probability_of_rain : takes_umbrella x → x = 1 / 9 :=
by
  -- placeholder for the proof
  intro h
  sorry

end probability_of_rain_l178_178727


namespace multiple_of_set_2_3_5_l178_178275

theorem multiple_of_set_2_3_5 : ∀ n ∈ {2, 3, 5}, 30 % n = 0 :=
by
  sorry

end multiple_of_set_2_3_5_l178_178275


namespace non_similar_triangles_count_l178_178807

/-- 
Proof that the number of non-similar triangles with angles that are 
distinct positive even integers in arithmetic progression is 29.
-/ 
theorem non_similar_triangles_count : 
  ∃ (a d : ℤ), (∀ (x y z : ℤ), x = 2*a - 2*d ∧ y = 2*a ∧ z = 2*a + 2*d → 
  (x + y + z = 180 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧ 
  ∑ i in finset.Icc 1 29, i = 29) := 
sorry

end non_similar_triangles_count_l178_178807


namespace arithmetic_sequence_conditions_geometric_sequence_and_sums_l178_178850

variable {a_n b_n S_n T_n : ℕ → ℤ}
variable {d : ℤ}

-- Define the sequences and sums based on the given problem
def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) := a_1 + d * (n - 1)
def sum_arithmetic_sequence (a_1 d : ℤ) (n : ℕ) := n * (a_1 + d * (n - 1) / 2)
def geometric_sequence (a r : ℤ) (n : ℕ) := a * r^(n - 1)
def sum_geometric_sequence (a r : ℤ) (n : ℕ) := a * (r^n - 1) / (r - 1)

-- Define the specific terms and sums based on the given answers
def a_n (n : ℕ) := 2 * n - 6
def S_n (n : ℕ) := n^2 - 5 * n
def b_n (n : ℕ) := 2 * n - 6 + 3^(n - 1)
def T_n (n : ℕ) := n^2 - 5 * n + (3^n - 1) / 2

-- Lean statements to verify the given conditions match the derived formulas
theorem arithmetic_sequence_conditions (a1 : ℤ) (d : ℤ) :
  2 * a1 + 3 * d = -2 ∧ 3 * a1 + 10 * d = 12 →
  a_n = a1 + d * (n - 1) ∧ S_n = n * (a1 + a1 + (n - 1) * d) / 2 := by sorry

theorem geometric_sequence_and_sums (a1 : ℤ) (d : ℤ) :
  ∀ n, b_n - a_n = 3^(n - 1) →
  b_n = a_n + 3^(n - 1) ∧ T_n = S_n + sum_geometric_sequence 1 3 n := by sorry

end arithmetic_sequence_conditions_geometric_sequence_and_sums_l178_178850


namespace eccentricity_of_ellipse_l178_178519

noncomputable def ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h_ellipse : a^2 = b^2 + c^2) : ℝ :=
c / a

theorem eccentricity_of_ellipse (a b c : ℝ) (e : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h_foci_dist : c = sqrt (a^2 - b^2))
  (h_P_F1_eq_F1_F2 : ∀ P, abs (P - (-c)) = 2 * c)
  (h_dist_origin_to_PF2 : ∀ P, abs P = b)
  (h_eccentricity : ellipse_eccentricity a b c h1 h2 (by sorry) = e) :
  e = 5 / 7 :=
sorry

end eccentricity_of_ellipse_l178_178519


namespace find_triplets_l178_178902

noncomputable def solution_triplets : set (ℕ × ℕ × ℕ) :=
  { (m, n, p) |
    ∃ (m n p : ℕ), m > 0 ∧ n > 0 ∧ prime p ∧ nat.choose m 3 - 4 = p ^ n }

theorem find_triplets : solution_triplets = {(6, 4, 2), (7, 1, 31)} :=
sorry

end find_triplets_l178_178902


namespace kathryn_remaining_money_l178_178505

variables (rent food_travel salary monthly_expenses remaining : ℝ)

-- Conditions
def rent_value := rent = 1200
def food_travel_expenses := food_travel = 2 * rent
def salary_value := salary = 5000
def shared_rent := monthly_expenses = rent / 2 + food_travel

-- Question and Answer
def money_remaining := remaining = salary - monthly_expenses

-- Theorem to prove
theorem kathryn_remaining_money (h1 : rent_value) (h2 : food_travel_expenses) (h3 : salary_value) (h4 : shared_rent) : money_remaining :=
sorry

end kathryn_remaining_money_l178_178505


namespace height_of_picture_frame_l178_178329

-- Define the given conditions
def width : ℕ := 6
def perimeter : ℕ := 30
def perimeter_formula (w h : ℕ) : ℕ := 2 * (w + h)

-- Prove that the height of the picture frame is 9 inches
theorem height_of_picture_frame : ∃ height : ℕ, height = 9 ∧ perimeter_formula width height = perimeter :=
by
  -- Proof goes here
  sorry

end height_of_picture_frame_l178_178329


namespace jack_sugar_l178_178866

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l178_178866


namespace identity_verification_l178_178949

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l178_178949


namespace difference_of_numbers_l178_178225

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 12390) (h2 : b = 2 * a + 18) : b - a = 4142 :=
by {
  sorry
}

end difference_of_numbers_l178_178225


namespace boys_and_girls_solution_l178_178672

theorem boys_and_girls_solution (x y : ℕ) 
  (h1 : 3 * x + y > 24) 
  (h2 : 7 * x + 3 * y < 60) : x = 8 ∧ y = 1 :=
by
  sorry

end boys_and_girls_solution_l178_178672


namespace inhabitable_fraction_of_mars_surface_l178_178471

theorem inhabitable_fraction_of_mars_surface :
  (3 / 5 : ℚ) * (2 / 3) = (2 / 5) :=
by
  sorry

end inhabitable_fraction_of_mars_surface_l178_178471


namespace min_value_rational_fn_l178_178399

theorem min_value_rational_fn {x : ℝ} (h : x > -1) : 
  ∃ y, y = (⨅ x : {x : ℝ // x > -1}, (x^2 + 7 * x + 10) / (x + 1)) ∧ y = 9 :=
sorry

end min_value_rational_fn_l178_178399


namespace find_ellipse_equation_find_max_area_AMBN_l178_178007

-- Define the constants and the ellipse equation conditions
def ellipse_equation (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the constraints for a, b, and c
def constraints (a b c : ℝ) := c = 2 ∧ (c / a) = (Real.sqrt 6 / 3) ∧ a^2 = b^2 + c^2 ∧ a > b ∧ b > 0

-- Define the focus and eccentricity conditions
def focus_condition := (2, 0)
def eccentricity_condition := (Real.sqrt 6 / 3)

-- Question 1: Prove the equation of the ellipse C
theorem find_ellipse_equation (a b c x y : ℝ) (h : constraints a b c) : ellipse_equation 6 2 := 
  sorry

-- Question 2: Prove the maximum area of the quadrilateral AMBN
theorem find_max_area_AMBN (a b c k : ℝ) (h : constraints a b c) : 
  (∃ (AMBN : ℝ), AMBN = 4 * Real.sqrt 3) :=
  sorry

end find_ellipse_equation_find_max_area_AMBN_l178_178007


namespace pipe_A_filling_time_l178_178548

theorem pipe_A_filling_time :
  ∃ (t : ℚ), 
  (∀ (t : ℚ), (t > 0) → (1 / t + 5 / t = 1 / 4.571428571428571) ↔ t = 27.42857142857143) := 
by
  -- definition of t and the corresponding conditions are directly derived from the problem
  sorry

end pipe_A_filling_time_l178_178548


namespace time_to_cross_is_fifteen_l178_178318

-- Define the conditions
def speed_km_hr : ℝ := 9  -- speed in kilometers per hour
def distance_m : ℝ := 2250  -- distance in meters
def meters_per_km : ℝ := 1000  -- meters in a kilometer
def minutes_per_hour : ℝ := 60  -- minutes in an hour

-- Define the speed in meters per minute
def speed_m_min : ℝ := (speed_km_hr * meters_per_km) / minutes_per_hour

-- Define the question and expected answer
def time_to_cross : ℝ := distance_m / speed_m_min

-- Prove that the time taken to cross the bridge is 15 minutes
theorem time_to_cross_is_fifteen : time_to_cross = 15 := by
  -- proof goes here; we use 'sorry' to leave it unproven
  sorry

end time_to_cross_is_fifteen_l178_178318


namespace proof_of_solution_set_l178_178718

noncomputable def solution_set (x y : ℝ) : Prop :=
  (y^2 - arccos (cos x)^2) * (y^2 - arccos (cos (x + π / 3))^2) * (y^2 - arccos (cos (x - π / 3))^2) < 0

theorem proof_of_solution_set (x y : ℝ) :
  solution_set x y := sorry

end proof_of_solution_set_l178_178718


namespace remaining_halves_cover_one_third_l178_178005

theorem remaining_halves_cover_one_third
  (A B : ℝ) -- A and B are the endpoints of segment AB
  (segments : set (ℝ × ℝ)) -- segments is a set of pairs representing the intervals of the shorter segments
  (covers_AB : ∀ x, x ∈ Icc A B → ∃ (s : ℝ × ℝ), s ∈ segments ∧ x ∈ Icc s.1 s.2) -- segments completely cover AB
  (discard_half : ∀ s ∈ segments, (s.2 - s.1) / 2): -- discarding half of each segment
  ∃ (remaining_half : set (ℝ × ℝ)), (∀ s ∈ segments, remaining_half = (λs, Icc ((s.1 + s.2) / 2) s.2)) ∧ (∀ x ∈ Icc A B, ∃ (s : ℝ × ℝ), s ∈ remaining_half ∧ x ∈ Icc s.1 s.2) sorry -- remaining halves cover at least 1/3 of the length of AB

end remaining_halves_cover_one_third_l178_178005


namespace arrange_in_ascending_order_ascending_order_l178_178755

noncomputable def a := 4 ^ 0.5
noncomputable def b := 0.5 ^ 4
noncomputable def c := Real.logBase 0.5 4

theorem arrange_in_ascending_order : c < b ∧ b < a := by
  have h1 : a = 4^(0.5) := rfl
  have h2 : a > 1 := by {
    apply Real.rpow_pos_of_pos; norm_num,
    apply Real.rpow_lt_rpow_of_exponent_gt_one; norm_num,
    norm_num
  }
  
  have h3 : b = 0.5^4 := rfl
  have h4 : 0 < b ∧ b < 1 := by {
    split;
    {
      apply Real.rpow_pos_of_pos; norm_num
    },
    {
      apply Real.rpow_lt_one_of_pos_of_lt_one; norm_num; norm_num
    }
  }

  have h5 : c = Real.logBase 0.5 4 := rfl
  have h6 : c < 0 := by {
    apply Real.logBase_lt_zero_of_gt_one _ _; norm_num
  }

  -- Now establish the final ordering using the inequalities
  exact ⟨h6, and.right h4, h2⟩

-- Assertion that ascending order is c < b < a
theorem ascending_order : c < b ∧ b < a := arrange_in_ascending_order

end arrange_in_ascending_order_ascending_order_l178_178755


namespace intersection_product_l178_178437

-- Define the curves C1 and C2.
def curve_C1 (t : ℝ) : ℝ × ℝ := (4 * t, 3 * t - 1)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Math.cos θ, ρ * Math.sin θ)

def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 8 * (Math.cos θ) / (1 - Math.cos (2 * θ))
  polar_to_cartesian ρ θ

-- The statement we need to prove.
theorem intersection_product :
  ∃ PA PB : ℝ, PA * PB = 25 / 9 :=
sorry

end intersection_product_l178_178437


namespace john_monthly_profit_l178_178500

theorem john_monthly_profit :
  let cost_per_component := 800
  let sell_multiplier := 1.4
  let computers_per_month := 60
  let rent_per_month := 5000
  let extra_expenses_per_month := 3000
  let selling_price_per_computer := cost_per_component * sell_multiplier
  let total_revenue := selling_price_per_computer * computers_per_month
  let total_cost_of_components := cost_per_component * computers_per_month
  let total_expenses := rent_per_month + extra_expenses_per_month
  let profit := total_revenue - (total_cost_of_components + total_expenses)
  profit = 11200 :=
by
  let cost_per_component := 800
  let sell_multiplier := 1.4
  let computers_per_month := 60
  let rent_per_month := 5000
  let extra_expenses_per_month := 3000
  let selling_price_per_computer := cost_per_component * sell_multiplier
  let total_revenue := selling_price_per_computer * computers_per_month
  let total_cost_of_components := cost_per_component * computers_per_month
  let total_expenses := rent_per_month + extra_expenses_per_month
  let profit := total_revenue - (total_cost_of_components + total_expenses)
  sorry

end john_monthly_profit_l178_178500


namespace problem_a_problem_b_l178_178148

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end problem_a_problem_b_l178_178148


namespace greatest_int_less_than_200_gcd_30_is_5_l178_178248

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l178_178248


namespace assign_integers_to_circles_l178_178307

-- Assume we have a type representing circles on the plane.
constant Circle : Type

-- Assume we have a relation indicating that two circles are connected by a line segment.
constant connected_by_segment : Circle → Circle → Prop

-- The main theorem which asserts that we can find an assignment of integers to circles
-- such that two circles are connected by a line segment if and only if the numbers in
-- them are relatively prime.
theorem assign_integers_to_circles (circles : List Circle) :
  ∃ (f : Circle → ℕ), 
    (∀ (c1 c2 : Circle), connected_by_segment c1 c2 ↔ Nat.coprime (f c1) (f c2)) :=
sorry

end assign_integers_to_circles_l178_178307


namespace sum_alternating_series_l178_178272

theorem sum_alternating_series : 
  ∑ i in (Finset.range 10002).filter (λ i, i % 2 = 0), (if i % 2 = 0 then (i / 2 + 1) else - (i / 2)) = 5001 :=
by
  sorry

end sum_alternating_series_l178_178272


namespace solve_system_eq_l178_178572

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end solve_system_eq_l178_178572


namespace length_perpendicular_half_side_ad_l178_178220

variables (A B C D O H : Type)
variable [metric_space O]
variable [circum_quadrilateral : cyclic_quadrilateral A B C D]
variable [center_of_circle : is_center_of_circumcircle O A B C D]
variable [perpendicular_ac_bd : is_perpendicular (line_through_points A C) (line_through_points B D)]
variable [foot_of_perpendicular : is_foot_of_perpendicular H (line_through_points O A) (line_through_points O D)]
variable [side_length_bc : length_of_side (line_through_points B C) = bc_length]
variable [length_oh : length_of_segment (perpendicular_segment O (line_through_points A D)) = oh_length]

theorem length_perpendicular_half_side_ad (A B C D O H : Type)
  [metric_space O]
  [circum_quadrilateral : cyclic_quadrilateral A B C D]
  [center_of_circle : is_center_of_circumcircle O A B C D]
  [perpendicular_ac_bd : is_perpendicular (line_through_points A C) (line_through_points B D)]
  [foot_of_perpendicular : is_foot_of_perpendicular H (line_through_points O A) (line_through_points O D)]
  [side_length_bc : length_of_side (line_through_points B C) = bc_length]
  [length_oh : length_of_segment (perpendicular_segment O (line_through_points A D)) = oh_length] :
  oh_length = bc_length / 2 :=
sorry

end length_perpendicular_half_side_ad_l178_178220


namespace spot_reachable_area_l178_178969

theorem spot_reachable_area (hex_side : ℝ) (rope_length : ℝ) (bush_distance : ℝ) (bush_sector_angle : ℝ) (reachable_sector_angle : ℝ) :
  hex_side = 2 ∧ rope_length = 4 ∧ bush_distance = 1 ∧ bush_sector_angle = 120 ∧ reachable_sector_angle = 240 → 
  let big_sector_area := (reachable_sector_angle / 360) * (pi * (rope_length ^ 2)),
      bush_area := (bush_sector_angle / 360) * (pi * (bush_distance ^ 2)),
      total_area := big_sector_area - bush_area in
      total_area = 21 * pi := 
by
  intro h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest',
  cases h_rest' with h3 h_rest'',
  cases h_rest'' with h4 h5,
  let big_sector_area := (h5 / 360) * (pi * (h2 ^ 2)),
  let bush_area := (h4 / 360) * (pi * (h3 ^ 2)),
  let total_area := big_sector_area - bush_area,
  have big_sector_area_eq : big_sector_area = (240 / 360) * (pi * (4^2)) := by simp [h2, h5],
  have bush_area_eq : bush_area = (120 / 360) * (pi * (1^2)) := by simp [h3, h4],
  rw [big_sector_area_eq, bush_area_eq],
  simp,
  apply eq_of_sub_eq_zero,
  ring,
  sorry

end spot_reachable_area_l178_178969


namespace area_of_triangle_AGE_l178_178545

noncomputable def square_side := 5
def pointA : (ℝ × ℝ) := (0, 0)
def pointB : (ℝ × ℝ) := (square_side, 0)
def pointC : (ℝ × ℝ) := (square_side, square_side)
def pointD : (ℝ × ℝ) := (0, square_side)
def pointE : (ℝ × ℝ) := (square_side, (2 * square_side / 5))

axiom circumcircle_intersects_bd_at_G (G : ℝ × ℝ) :
  (circumcircle (pointA, pointB, pointE)).intersects_bd G

theorem area_of_triangle_AGE : 
  ∃ (G : ℝ × ℝ), circumcircle_intersects_bd_at_G G →
  let area := (1 / 2) * |pointA.1 * (G.2 - pointE.2) + G.1 * (pointE.2 - pointA.2) + pointE.1 * (pointA.2 - G.2)| 
  in area = 58.25 := 
sorry

end area_of_triangle_AGE_l178_178545


namespace polyhedron_volume_l178_178483

noncomputable def is_square (s : ℝ) : ℝ := s ^ 2
noncomputable def is_right_triangle (a b : ℝ) : ℝ := (a * b) / 2
noncomputable def is_equilateral_triangle (s : ℝ) : ℝ := (s^2 * (sqrt 3)) / 4
noncomputable def volume_of_cube (s : ℝ) : ℝ := s^3

theorem polyhedron_volume (s : ℝ) (a b h : ℝ) :
  (is_square s = s^2) → 
  (is_right_triangle a b = (a * b) / 2) →
  (volume_of_cube s = s^3) →
  (s = 1) → 
  (a = s) → 
  (b = s) → 
  (h = s) →
  let tetrahedron_volume := (1/3) * (is_right_triangle a b) * h 
  in (volume_of_cube s) - tetrahedron_volume = 5 / 6 :=
by
  intros
  unfold is_square is_right_triangle volume_of_cube
  rw [h_1, h_2, h_3] 
  have : tetrahedron_volume = (1/3) * ((1 * 1) / 2) * 1 := by
    rw [h_5, h_6]
    simp
  rw [this]
  simp
  norm_num

#check polyhedron_volume

end polyhedron_volume_l178_178483


namespace right_triangle_one_right_angle_l178_178814

theorem right_triangle_one_right_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90 ∨ B = 90 ∨ C = 90) : (interp A B C).count (90) = 1 :=
by
  sorry

end right_triangle_one_right_angle_l178_178814


namespace cannot_achieve_1970_minuses_l178_178076

theorem cannot_achieve_1970_minuses :
  ∃ (x y : ℕ), x ≤ 100 ∧ y ≤ 100 ∧ (x - 50) * (y - 50) = 1515 → false :=
by
  sorry

end cannot_achieve_1970_minuses_l178_178076


namespace students_need_to_raise_each_l178_178670

def initial_amount_needed (num_students : ℕ) (amount_per_student : ℕ) (misc_expenses : ℕ) : ℕ :=
  (num_students * amount_per_student) + misc_expenses

def amount_raised_first_three_days (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  day1 + day2 + day3

def amount_raised_next_four_days (first_three_days_total : ℕ) : ℕ :=
  first_three_days_total / 2

def total_amount_raised_in_week (first_three_days_total : ℕ) (next_four_days_total : ℕ) : ℕ :=
  first_three_days_total + next_four_days_total

def amount_each_student_still_needs_to_raise 
  (total_needed : ℕ) (total_raised : ℕ) (num_students : ℕ) : ℕ :=
  if num_students > 0 then (total_needed - total_raised) / num_students else 0

theorem students_need_to_raise_each 
  (num_students : ℕ) (amount_per_student : ℕ) (misc_expenses : ℕ)
  (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) (next_half_factor : ℕ)
  (h_num_students : num_students = 6)
  (h_amount_per_student : amount_per_student = 450)
  (h_misc_expenses : misc_expenses = 3000)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 900)
  (h_day3 : day3 = 400)
  (h_next_half_factor : next_half_factor = 2) :
  amount_each_student_still_needs_to_raise
    (initial_amount_needed num_students amount_per_student misc_expenses)
    (total_amount_raised_in_week
      (amount_raised_first_three_days day1 day2 day3)
      (amount_raised_next_four_days (amount_raised_first_three_days day1 day2 day3 / h_next_half_factor)))
    num_students = 475 :=
by sorry

end students_need_to_raise_each_l178_178670


namespace pyramid_volume_correct_l178_178327

-- Define the side length of the equilateral triangle base
noncomputable def side_length : ℝ := 1 / Real.sqrt 2

-- Define the area of an equilateral triangle with the given side length
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := 
  (Real.sqrt 3 / 4) * s^2 

-- Define the base area of the pyramid
noncomputable def base_area : ℝ := equilateral_triangle_area side_length

-- Define the height (altitude) from the vertex to the base
noncomputable def height : ℝ := 1

-- Define the volume of the pyramid using the formula for pyramid volume
noncomputable def pyramid_volume (base_area height : ℝ) : ℝ := 
  (1 / 3) * base_area * height

-- The proof statement
theorem pyramid_volume_correct : 
  pyramid_volume base_area height = Real.sqrt 3 / 24 :=
by
  sorry

end pyramid_volume_correct_l178_178327


namespace problem_pure_imaginary_l178_178643

theorem problem_pure_imaginary (m i : ℂ) (h_i : i = complex.I)
  (h : ∃ k : ℂ, m * (m - 1) + m * i = k * complex.I) : m = 1 := 
sorry

end problem_pure_imaginary_l178_178643


namespace min_max_values_l178_178186

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_values : 
  ∃ (min_val max_val : ℝ), 
  min_val = -3 * Real.pi / 2 ∧ 
  max_val = Real.pi / 2 + 2 ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≥ min_val) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≤ max_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = min_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = max_val) := 
by
  sorry

end min_max_values_l178_178186


namespace greatest_integer_with_gcf_5_l178_178253

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l178_178253


namespace prove_right_triangle_l178_178832

-- Definitions corresponding to problem conditions
variables {A B C D E : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (x y z x1 y1 z1 x2 y2 z2 : ℝ)

-- Points in the triangle
def A := (x, y, z)
def B := (x1, y1, z1)
def C := (x2, y2, z2)

-- Definitions of the extension conditions
def CD := λ (x y z), x + 1 * (-1, 0, 2)
def AE := λ (x y z), x + 2 * (3, 0, -2)

-- Definition of the condition AD = BE
def AD_BE := λ (A B C D E), dist A D = dist B E

-- The main theorem stating the problem to solve
noncomputable def is_right_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
\(triangle : Type*)
(triangle ABC)

theorem prove_right_triangle (h1 : CD = CB) (h2 : AE = 2 * AC) (h3 : AD_BE A B C D E) : is_right_triangle A B C :=
begin
  sorry
end

end prove_right_triangle_l178_178832


namespace total_subjects_l178_178139

theorem total_subjects (subjects_monica subjects_marius subjects_millie : ℕ)
  (h1 : subjects_monica = 10)
  (h2 : subjects_marius = subjects_monica + 4)
  (h3 : subjects_millie = subjects_marius + 3) :
  subjects_monica + subjects_marius + subjects_millie = 41 :=
by
  sorry

end total_subjects_l178_178139


namespace collinear_points_l178_178801

variables (a b : ℝ × ℝ) (A B C D : ℝ × ℝ)

-- Define the vectors
noncomputable def vec_AB : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def vec_BC : ℝ × ℝ := (2 * a.1 + 8 * b.1, 2 * a.2 + 8 * b.2)
noncomputable def vec_CD : ℝ × ℝ := (3 * (a.1 - b.1), 3 * (a.2 - b.2))

-- Define the collinearity condition
def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Translate the problem statement into Lean
theorem collinear_points (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) (h₂ : ¬ (a.1 * b.2 - a.2 * b.1 = 0)):
  collinear (6 * (a.1 + b.1), 6 * (a.2 + b.2)) (5 * (a.1 + b.1, a.2 + b.2)) :=
sorry

end collinear_points_l178_178801


namespace place_mat_length_l178_178654

theorem place_mat_length :
  ∀ (R_table R_decoration : ℝ) (n : ℕ) (w y : ℝ),
  R_table = 5 →
  R_decoration = 1 →
  n = 7 →
  w = 1.5 →
  (∀ (i : ℕ), i < n → 
    ((∃ θ : ℝ, cos (θ / 2) = (w / (2 * R_table)) ∧ cos (θ / 2) = ((y - w) / 2R_decoration)) 
      ∧ ((π / n) = asin (w / (2 * R_table))) ∧ (θ = (2π / n)))) →
  y = 5.69 :=
by
  intros _ _ _ _ _ hR_table hR_decoration h_n h_w h_cos
  sorry

end place_mat_length_l178_178654


namespace probability_one_instrument_l178_178296

theorem probability_one_instrument (total_people : ℕ) (at_least_one_instrument_ratio : ℚ) (two_or_more_instruments : ℕ)
  (h1 : total_people = 800) (h2 : at_least_one_instrument_ratio = 1 / 5) (h3 : two_or_more_instruments = 128) :
  (160 - 128) / 800 = 1 / 25 :=
by
  sorry

end probability_one_instrument_l178_178296


namespace inequality_correct_solution_l178_178593

def inequality_solution_set := { x : ℝ | (x - 2) / (x + 3) > 0 } = set.Iio (-3) ∪ set.Ioi 2

theorem inequality_correct_solution :
  { x : ℝ | (x - 2) / (x + 3) > 0 } = set.Iio (-3) ∪ set.Ioi 2 :=
sorry

end inequality_correct_solution_l178_178593


namespace geometric_sequence_11th_term_l178_178183

theorem geometric_sequence_11th_term (a r : ℕ) :
    a * r^4 = 3 →
    a * r^7 = 24 →
    a * r^10 = 192 := by
    sorry

end geometric_sequence_11th_term_l178_178183


namespace expression_for_h_l178_178201

variables {α : Type*} (f h : α → α)

def is_reflection_across_y_axis (g : α → α) (f : α → α) : Prop :=
∀ x, g x = f (-x)

def is_shifted_right_by (g : α → α) (f : α → α) (a : α) : Prop :=
∀ x, g x = f (x - a)

theorem expression_for_h (f : ℝ → ℝ) (h : ℝ → ℝ)
  (H1 : ∀ x, h x = f (3 - x)) : 
  ∃ g₁ g₂, 
    (is_reflection_across_y_axis g₁ f) ∧ 
    (is_shifted_right_by g₂ g₁ 3) ∧ 
    (h = g₂) :=
by 
  sorry

end expression_for_h_l178_178201


namespace probability_of_product_is_29_over_36_l178_178130

open Classical

def probability_product_leq_36 :=
  let p_outcome := [1, 2, 3, 4, 5, 6]
  let m_outcome := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let pairs := [(p, m) | p ← p_outcome, m ← m_outcome]
  let valid_pairs := pairs.filter (λ pair => pair.1 * pair.2 ≤ 36)
  let total_prob := (valid_pairs.length : ℚ) / (pairs.length : ℚ)
  total_prob

theorem probability_of_product_is_29_over_36 :
  probability_product_leq_36 = 29 / 36 := 
by
  sorry

end probability_of_product_is_29_over_36_l178_178130


namespace range_of_a_relationship_a_sum_l178_178031

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

axiom distinct_zero_points {a : ℝ} (h : 0 < a ∧ a < 1 / Real.exp 1) :
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0

theorem range_of_a : ∀ {a : ℝ}, (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) →
 0 < a ∧ a < 1 / Real.exp 1 := sorry

theorem relationship_a_sum {a x₁ x₂ : ℝ} (hx : x₁ < x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) :
  2 / (x₁ + x₂) < a := sorry

end range_of_a_relationship_a_sum_l178_178031


namespace Kolya_is_CollectionAgency_l178_178876

-- Define the roles
inductive Role
| FinancialPyramid
| CollectionAgency
| Bank
| InsuranceCompany

-- Define the conditions parametrically
structure Scenario where
  lent_books : Bool
  promise_broken : Bool
  mediator_requested : Bool
  reward_requested : Bool

-- Define the theorem statement
theorem Kolya_is_CollectionAgency
  (scenario : Scenario)
  (h1 : scenario.lent_books = true)
  (h2 : scenario.promise_broken = true)
  (h3 : scenario.mediator_requested = true)
  (h4 : scenario.reward_requested = true) :
  Kolya_is_CollectionAgency :=
  begin
    -- Proof not required
    sorry
  end

end Kolya_is_CollectionAgency_l178_178876


namespace hyperbola_eccentricity_l178_178093

-- Lean 4 statement for the given math proof problem
theorem hyperbola_eccentricity (A B C H : Euclidean_space ℝ (fin 3))
  (cos_C : ℝ) 
  (h_cos_C : cos_C = (2 * sqrt 5) / 5)
  (h_AH_BC : (A - H) • (B - C) = 0)
  (h_AB_CA_CB : (A - B) • (C - A + C - B) = 0) :
  eccentricity_of_hyperbola_passing_through(A, H, C) = sqrt 5 + 2 :=
sorry

end hyperbola_eccentricity_l178_178093


namespace find_x4_y4_l178_178063

theorem find_x4_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end find_x4_y4_l178_178063


namespace distances_not_equal_l178_178922

-- Define the segment and coordinates
variable {A B : ℝ}
variable {P : Fin 45 → ℝ}

-- Distance function
def distance (x y : ℝ) := abs (x - y)

-- Segment AB (assuming length AB = 1)
axiom ab_length : distance A B = 1

-- 45 points outside the segment AB
axiom points_outside_segment : ∀ i : Fin 45, distance P i A > 1 ∨ distance P i B > 1

theorem distances_not_equal :
  (∑ i, distance (P i) A) ≠ (∑ i, distance (P i) B) :=
sorry

end distances_not_equal_l178_178922


namespace perfect_square_difference_l178_178901

theorem perfect_square_difference (m n : ℕ) (h : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, k^2 = m - n :=
sorry

end perfect_square_difference_l178_178901


namespace f_of_f_neg_two_l178_178107

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 else -x

theorem f_of_f_neg_two : f (f (-2)) = 4 :=
by
  sorry

end f_of_f_neg_two_l178_178107


namespace maximum_temperature_difference_cooling_period_l178_178596

-- Define the temperature function
def temperature (t : ℝ) : ℝ := 10 - sqrt 3 * cos (π / 12 * t) - sin (π / 12 * t)

-- Maximum temperature difference
theorem maximum_temperature_difference : 
  ∃ ΔT_max : ℝ, ΔT_max = 4 ∧ ∀ t ∈ Ico 0 24, ΔT_max = max (temperature t) - min (temperature t) := 
sorry

-- Time period for cooling requirement
theorem cooling_period (t : ℝ) (h : t ∈ Icc 10 18 → temperature t ≥ 11) : 
  ∀ t, t ∈ Ico 0 24 → (temperature t > 11 ↔ t ∈ Ico 10 18) := 
sorry

end maximum_temperature_difference_cooling_period_l178_178596


namespace min_log_sum_l178_178419

open Real

theorem min_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 4) :
  log 2 x + log 2 y = -1 :=
sorry

end min_log_sum_l178_178419


namespace culprit_is_Toropyzhka_l178_178641

variables (Syropchik Toropyzhka Ponchik : Prop)
variables (AteSyropchik AteToropyzhka AtePonchik : Prop)
variables (ToldTruthSyropchik ToldTruthToropyzhka ToldTruthPonchik : Prop)

-- Statements made by the suspects
def Statement_Syropchik := ¬ AteSyropchik
def Statement_Toropyzhka := AtePonchik ∨ AteSyropchik
def Statement_Ponchik := ¬ AteSyropchik

-- Conditions
axiom InnocentTellTruthLiarLies : ∀ {a}, 
    (ToldTruthSyropchik → (Statement_Syropchik ↔ ¬a)) ∧ 
    (ToldTruthToropyzhka → (Statement_Toropyzhka ↔ ¬a)) ∧ 
    (ToldTruthPonchik → (Statement_Ponchik ↔ ¬a))

axiom OnlyOneLiar : 
  (ToldTruthSyropchik ∧ ToldTruthToropyzhka ∧ ¬ToldTruthPonchik) ∨ 
  (ToldTruthSyropchik ∧ ¬ToldTruthToropyzhka ∧ ToldTruthPonchik) ∨ 
  (¬ToldTruthSyropchik ∧ ToldTruthToropyzhka ∧ ToldTruthPonchik)

-- Proof goal: Determine who ate the dog food
theorem culprit_is_Toropyzhka : 
  (ToldTruthSyropchik → Statement_Syropchik) ∧ 
  (ToldTruthToropyzhka → Statement_Toropyzhka) ∧ 
  (ToldTruthPonchik → Statement_Ponchik) →
  AteToropyzhka :=
by 
  sorry

end culprit_is_Toropyzhka_l178_178641


namespace binary_mul_shift_right_l178_178385

theorem binary_mul_shift_right : 
  let bin1 := 0b1101101 
  let bin2 := 0b1111
  let product := bin1 * bin2
  let shifted_product := (product / 4) in
  -- Convert the result to binary notation
  shifted_product = 0b1010011111.01 := 
by
  sorry

end binary_mul_shift_right_l178_178385


namespace increasing_f_in_I_1_solve_inequality_find_range_of_t_l178_178023

-- Given conditions
variable (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 1) 
variable (h_value_at_one : f 1 = 1)
variable (h_positive_quotient : ∀ m n, m ≠ -n → m ∈ Icc (-1) 1 → n ∈ Icc (-1) 1 → (f m + f n) / (m + n) > 0)

-- 1. Prove monotonicity
theorem increasing_f_in_I_1 : ∀ x y : ℝ, x ∈ Icc (-1) 1 → y ∈ Icc (-1) 1 → x < y → f x < f y := 
sorry

-- 2. Solve the inequality f(x + 1/2) < f(1 - x)
theorem solve_inequality : set_of (λ x : ℝ, 0 ≤ x ∧ x < 1/4) = 
            set_of (λ x, f(x + 1/2) < f(1 - x)) := 
sorry

-- 3. Prove range of t
theorem find_range_of_t : ∀ t : ℝ, 
            (∀ x a, x ∈ Icc (-1) 1 → a ∈ Icc (-1) 1 → f x ≤ -2 * a * t + 2) ↔ 
            t ∈ Icc (-1/2) (1/2) := 
sorry

end increasing_f_in_I_1_solve_inequality_find_range_of_t_l178_178023


namespace intersection_M_N_l178_178042

noncomputable def M : set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }
noncomputable def N : set ℝ := { y | ∃ x : ℝ, x^2 + y^2 = 1 }

theorem intersection_M_N : M ∩ N = { y | -1 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_M_N_l178_178042


namespace ellipse_focal_point_l178_178028

theorem ellipse_focal_point (m : ℝ) (m_pos : m > 0)
  (h : ∃ f : ℝ × ℝ, f = (1, 0) ∧ ∀ x y : ℝ, (x^2 / 4) + (y^2 / m^2) = 1 → 
    (x - 1)^2 + y^2 = (x^2 / 4) + (y^2 / m^2)) :
  m = Real.sqrt 3 := 
sorry

end ellipse_focal_point_l178_178028


namespace total_viewing_time_l178_178648

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end total_viewing_time_l178_178648


namespace sum_quotient_is_integer_l178_178747

theorem sum_quotient_is_integer 
  (n : ℕ) 
  (hn : n > 1) 
  (a : ℕ → ℕ)
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (k : ℕ)
  (hk : k > 0) : 
    ∃ m : ℤ, (m = ∑ i in finset.range n, (a i) ^ k / (∏ j in finset.range n, if i ≠ j then (a i - a j) else 1)) :=
sorry


end sum_quotient_is_integer_l178_178747


namespace students_met_goal_l178_178669

def money_needed_per_student : ℕ := 450
def number_of_students : ℕ := 6
def collective_expenses : ℕ := 3000
def amount_raised_day1 : ℕ := 600
def amount_raised_day2 : ℕ := 900
def amount_raised_day3 : ℕ := 400
def days_remaining : ℕ := 4
def half_of_first_three_days : ℕ :=
  (amount_raised_day1 + amount_raised_day2 + amount_raised_day3) / 2

def total_needed : ℕ :=
  money_needed_per_student * number_of_students + collective_expenses
def total_raised : ℕ :=
  amount_raised_day1 + amount_raised_day2 + amount_raised_day3 + (half_of_first_three_days * days_remaining)

theorem students_met_goal : total_raised >= total_needed := by
  sorry

end students_met_goal_l178_178669


namespace kathryn_financial_statement_l178_178502

def kathryn_remaining_money (rent : ℕ) (salary : ℕ) (share_rent : ℕ → ℕ) (total_expenses : ℕ → ℕ) (remaining_money : ℕ → ℕ) : Prop :=
  rent = 1200 ∧
  salary = 5000 ∧
  share_rent rent = rent / 2 ∧
  ∀ rent_total, total_expenses (share_rent rent_total) = (share_rent rent_total) + 2 * rent_total ∧
  remaining_money salary total_expenses = salary - total_expenses (share_rent rent)

theorem kathryn_financial_statement : kathryn_remaining_money 1200 5000 (λ rent, rent / 2) (λ rent, rent / 2 + 2 * rent) (λ salary expenses, salary - expenses (λ rent, rent / 2)) :=
by {
  sorry
}

end kathryn_financial_statement_l178_178502


namespace identity_of_polynomials_l178_178946

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l178_178946


namespace semi_circle_radius_l178_178993

theorem semi_circle_radius (P : ℝ) (π : ℝ) (r : ℝ) (hP : P = 10.797344572538567) (hπ : π = 3.14159) :
  (π + 2) * r = P → r = 2.1 :=
by
  intro h
  sorry

end semi_circle_radius_l178_178993


namespace new_average_salary_l178_178974

def average_salary_old_group : ℝ := 430
def old_supervisor_salary : ℝ := 870
def new_supervisor_salary : ℝ := 510

theorem new_average_salary : 
  let total_old_salary := average_salary_old_group * 9 in
  let total_workers_salary := total_old_salary - old_supervisor_salary in
  let total_new_salary := total_workers_salary + new_supervisor_salary in
  total_new_salary / 9 = 390 := 
by 
  sorry

end new_average_salary_l178_178974


namespace BC_gt_2AL_l178_178883

variable (A B C H M L : Type) [Geometry A B C H M L] -- Assume we have a geometry context
variable (AL BC : ℝ) -- AL and BC are real numbers representing lengths

-- Initial conditions
variable (acute_triangle : ∃ (α β γ : ℝ), α < 90 ∧ β < 90 ∧ γ < 90) -- Holds for an acute triangle
variable (is_median : is_median A B C M) -- AM is a median
variable (is_height : is_height A H C) -- AH is a height
variable (is_bisector : is_bisector A L C) -- AL is an internal angle bisector
variable (collinear_points : collinear [B, H, L, M, C]) -- Points B, H, L, M, C are collinear
variable (LH LM : ℝ) -- LH and LM are distances such that:
variable (LH_lt_LM : LH < LM) -- LH < LM

-- Question to prove
theorem BC_gt_2AL : BC > 2 * AL :=
by
  sorry

end BC_gt_2AL_l178_178883


namespace given_conditions_l178_178004

noncomputable def geometric_sum (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) / (2 ^ (i + 1))

theorem given_conditions 
  (∀ (n : ℕ), a_n = 4 * 4^n) 
  (a_1_eq : a_1 = 4) 
  (a5_cond : a_5^2 = 16 * a_2 * a_6) : 
  geometric_sum a_n = 2 - (n + 2) / 2^n := 
sorry

end given_conditions_l178_178004


namespace henry_added_water_l178_178051

theorem henry_added_water (initial_fraction full_capacity final_fraction : ℝ) (h_initial_fraction : initial_fraction = 3/4) (h_full_capacity : full_capacity = 56) (h_final_fraction : final_fraction = 7/8) :
  final_fraction * full_capacity - initial_fraction * full_capacity = 7 :=
by
  sorry

end henry_added_water_l178_178051


namespace max_abs_c_l178_178899

theorem max_abs_c (a b c d e : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -1 ≤ a * x^4 + b * x^3 + c * x^2 + d * x + e ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e ≤ 1) : abs c ≤ 8 :=
by {
  sorry
}

end max_abs_c_l178_178899


namespace triangle_values_l178_178833

-- Definitions and conditions
def angle_A := 120 -- in degrees
def b := 1
def area_ABC := Real.sqrt 3

-- Using the cosine rule, trigonometric properties and the conditions, prove the desired expression
theorem triangle_values : 
  let c := 4 in
  let a := Real.sqrt 21 in
  a - b + c = Real.sqrt 21 + 3 :=
by
  -- Proof steps go here
  sorry

end triangle_values_l178_178833


namespace identity_eq_l178_178939

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l178_178939


namespace intervals_of_increase_max_value_when_a_is_2_l178_178784

noncomputable def f (x a : ℝ) : ℝ :=
  3*x^4 - 4*(a+1)*x^3 + 6*a*x^2 - 12

noncomputable def f_prime (x a : ℝ) : ℝ :=
  12*x*(x-a)*(x-1)

theorem intervals_of_increase (a : ℝ) (ha : 0 < a) :
  (0 < a ∧ a < 1 → ∀ x, (0 ≤ x ∧ x ≤ a) ∨ (1 < x) → 0 < f_prime x a) ∧
  (a = 1 → ∀ x, 0 ≤ x ∧ x ≠ 1 → 0 < f_prime x a) ∧
  (1 < a → ∀ x, (0 ≤ x ∧ x ≤ 1) ∨ a < x → 0 < f_prime x a) :=
by sorry

theorem max_value_when_a_is_2 :
  let a := 2 in
  ∀ x, (x < 0 ∨ (0 ≤ x ∧ x ≤ 1) ∨ (1 < x ∧ x ≤ 2) ∨ 2 < x) ∧
       ∃ x_max, x_max ∈ {0, 1, 2} ∧ f x_max a = -9 :=
by sorry

end intervals_of_increase_max_value_when_a_is_2_l178_178784


namespace y_intercept_common_tangent_l178_178611

/-- 
Given two circles with centers at (1, 3) and (10, 6) with radii 3 and 6 respectively, 
prove that the y-intercept of their common external tangent line, expressed in the 
form y = mx + b with m > 0, is 9/4.
-/
theorem y_intercept_common_tangent
    (center1 : Real×Real := (1,3))
    (radius1 : Real := 3)
    (center2 : Real×Real := (10,6))
    (radius2 : Real := 6)
    (m : Real)
    (h_pos_m : 0 < m)
    (tangent_line : m * (fst center1) + (snd center1) - radius1 = 
                    m * (fst center2) + (snd center2) - radius2) :
    m = 3/4 → 
    ∃ b : Real, b = 9/4 :=
by
  sorry

end y_intercept_common_tangent_l178_178611


namespace mean_median_difference_l178_178600

def roller_coaster_drops := [140, 125, 152, 310, 205, 180]

def mean (lst : List ℚ) : ℚ := (list.sum lst) / lst.length

def median (lst : List ℚ) : ℚ :=
let sorted := lst.qsort (≤) in
let n := sorted.length in
if n % 2 = 0 then
  (list.nth_le sorted (n/2 - 1) (sorry) + list.nth_le sorted (n/2) (sorry)) / 2
else
  list.nth_le sorted (n/2) (sorry)

theorem mean_median_difference :
  (mean roller_coaster_drops - median roller_coaster_drops).abs = 19.67 := 
sorry

end mean_median_difference_l178_178600


namespace goldfish_sold_l178_178218

def buy_price : ℕ → ℝ := λ n, 0.25 * n
def sell_price : ℕ → ℝ := λ n, 0.75 * n
def tank_cost : ℝ := 100
def percentage_short : ℝ := 0.45
def percentage_achieved : ℝ := 1 - percentage_short

theorem goldfish_sold (n : ℕ) (h_buy : buy_price n = 0.25 * n)
  (h_sell : sell_price n = 0.75 * n)
  (h_tank : tank_cost = 100)
  (h_percentage : percentage_short = 0.45) :
  percentage_achieved * tank_cost / (sell_price 1 - buy_price 1) = 110 :=
by
  sorry

end goldfish_sold_l178_178218


namespace V_annual_decrease_l178_178834

-- Define initial budgets and annual budget changes
def budgets_in_1990 := (Q_budget_1990 : ℕ := 540000, V_budget_1990 : ℕ := 780000)
def Q_annual_increase := 30000

-- Define the condition when budgets are equal in 1994
def budgets_equal_1994 (D : ℕ) : Prop := 
  budgets_in_1990.Q_budget_1990 + 4 * Q_annual_increase = budgets_in_1990.V_budget_1990 - 4 * D

-- The theorem proving the annual decrease in project V's budget
theorem V_annual_decrease : ∃ (D : ℕ), budgets_equal_1994 D ∧ D = 30000 := 
  sorry

end V_annual_decrease_l178_178834


namespace gwen_points_per_bag_l178_178050

theorem gwen_points_per_bag : 
  ∀ (total_bags recycled_bags total_points_per_bag points_per_bag : ℕ),
  total_bags = 4 → 
  recycled_bags = total_bags - 2 →
  total_points_per_bag = 16 →
  points_per_bag = (total_points_per_bag / total_bags) →
  points_per_bag = 4 :=
by
  intros
  sorry

end gwen_points_per_bag_l178_178050


namespace contrapositive_statement_l178_178111

theorem contrapositive_statement (m : ℝ) (h : ¬ ∃ x : ℝ, x^2 = m) : m < 0 :=
sorry

end contrapositive_statement_l178_178111


namespace sue_initially_borrowed_six_movies_l178_178163

variable (M : ℕ)
variable (initial_books : ℕ := 15)
variable (returned_books : ℕ := 8)
variable (returned_movies_fraction : ℚ := 1/3)
variable (additional_books : ℕ := 9)
variable (total_items : ℕ := 20)

theorem sue_initially_borrowed_six_movies (hM : total_items = initial_books - returned_books + additional_books + (M - returned_movies_fraction * M)) : 
  M = 6 := by
  sorry

end sue_initially_borrowed_six_movies_l178_178163


namespace number_of_points_l178_178422

theorem number_of_points (m n : ℤ)
    (h1 : (m^2 - 4) * (m^2 + 12 * m + 32) + 4 = n^2) :
    (∃ A, ∀ P : ℤ × ℤ, P ∈ A ↔ (m, n^2) = P) ∧ M.card = 6 :=
sorry

end number_of_points_l178_178422


namespace trajectory_of_point_is_parabola_l178_178402

theorem trajectory_of_point_is_parabola (x y : ℝ) :
  5 * real.sqrt ((x - 1)^2 + (y - 2)^2) = abs (3 * x + 4 * y - 1) →
  ∃ a b c : ℝ, a * (x - b)^2 - y + c = 0 :=
by sorry

end trajectory_of_point_is_parabola_l178_178402


namespace jack_sugar_l178_178868

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l178_178868


namespace three_digit_numbers_with_two_even_digits_l178_178817

theorem three_digit_numbers_with_two_even_digits :
  let digits := {1, 2, 3, 4, 5, 6}
  let even_digits := {2, 4, 6}
  let odd_digits := {1, 3, 5}
  let count_even_choices := 3
  let count_odd_choices := 3
  let permutations_per_case := 6
  let arrangement_cases := 3
  count_even_choices * count_odd_choices * permutations_per_case * arrangement_cases = 162 := by
  sorry

end three_digit_numbers_with_two_even_digits_l178_178817


namespace correct_statement_l178_178982

-- Condition 1
def condition1 (x : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → tan x1 ≤ tan x2

-- Condition 2
def condition2 (x : ℝ) : Prop :=
  let f := λ x : ℝ, sin (|2 * x + (π / 6)|)
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x

-- Condition 3 and statement verification
def condition3 (λ : ℝ) : Prop :=
  let a := (2 : ℝ, λ)
  let b := (-3 : ℝ, 5)
  (a.1 * b.1 + a.2 * b.2) < 0

-- Statement 4 condition and verification
def condition4 (a : ℝ) : Prop :=
  ∀ x ∈ Set.Iic (1 : ℝ), a + 2 * 2^x + 4^x < 0

-- Correct statement
theorem correct_statement : 
  ¬ condition1 ∧
  ¬ condition2 ∧
  ¬ condition3 ∧
   condition4 a → a < -8 := 
by
  sorry

end correct_statement_l178_178982


namespace replace_stars_with_identity_l178_178956

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l178_178956


namespace range_of_a_l178_178757

variable (a : ℝ)

def A (a : ℝ) := { x : ℝ | -2 ≤ x ∧ x ≤ a }
def B (a : ℝ) := { y : ℝ | ∃ x ∈ A a, y = 2 * x + 3 }
def C (a : ℝ) := { t : ℝ | ∃ x ∈ A a, t = x^2 }

theorem range_of_a (h₁ : a ≥ -2)
                   (h₂ : C a ⊆ B a) :
  a ∈ set.Icc (1 / 2) 3 :=
sorry

end range_of_a_l178_178757


namespace equal_costs_at_60_minutes_l178_178637

-- Define the base rates and the per minute rates for each company
def base_rate_united : ℝ := 9.00
def rate_per_minute_united : ℝ := 0.25
def base_rate_atlantic : ℝ := 12.00
def rate_per_minute_atlantic : ℝ := 0.20

-- Define the total cost functions
def cost_united (m : ℝ) : ℝ := base_rate_united + rate_per_minute_united * m
def cost_atlantic (m : ℝ) : ℝ := base_rate_atlantic + rate_per_minute_atlantic * m

-- State the theorem to be proved
theorem equal_costs_at_60_minutes : 
  ∃ (m : ℝ), cost_united m = cost_atlantic m ∧ m = 60 :=
by
  -- Pending proof
  sorry

end equal_costs_at_60_minutes_l178_178637


namespace find_sum_of_c_and_d_l178_178464

variables {c d : ℝ}

-- Step 1: Define the function g(x)
def g (x : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

-- Step 2: Define the conditions for vertical asymptotes
def has_vertical_asymptotes (x : ℝ) : Prop := x = -1 ∨ x = 3

-- Step 3: Express the sum of c and d
def sum_of_coefficients : ℝ := c + d

-- Step 4: State the theorem
theorem find_sum_of_c_and_d 
  (h₁ : ∀ x : ℝ, has_vertical_asymptotes x → (x^2 + c * x + d) = 0) 
  : sum_of_coefficients = -5 :=
sorry

end find_sum_of_c_and_d_l178_178464


namespace fillets_per_fish_l178_178723

-- Definitions for the conditions
def fish_caught_per_day := 2
def days := 30
def total_fish_caught : Nat := fish_caught_per_day * days
def total_fish_fillets := 120

-- The proof problem statement
theorem fillets_per_fish (h1 : total_fish_caught = 60) (h2 : total_fish_fillets = 120) : 
  (total_fish_fillets / total_fish_caught) = 2 := sorry

end fillets_per_fish_l178_178723


namespace probability_neither_prime_nor_composite_l178_178295

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def is_neither_prime_nor_composite (n : ℕ) : Prop :=
  n = 1

theorem probability_neither_prime_nor_composite :
  (∑ n in Finset.range 100, if is_neither_prime_nor_composite n then 1 else 0) / 99 = 1 / 99 :=
  by
  sorry

end probability_neither_prime_nor_composite_l178_178295


namespace correct_propositions_count_l178_178413

/-- Given lines l and m, and planes a and b, with l ⊥ a, m intersects b, 
  the number of correct propositions out of the given four propositions is equal to one:
  1. If a ∥ b, then l ⊥ m.
  2. If l ⊥ m, then a ∥ b.
  3. If a ⊥ b, then l ∥ m.
  4. If l ∥ m, then a ⊥ b.
-/
theorem correct_propositions_count (l m : Type) (a b : Type)
  [Perp l a] [Intersects m b] : (number_of_correct_propositions l m a b = 1) :=
sorry

end correct_propositions_count_l178_178413


namespace determine_m_l178_178798

theorem determine_m (m : ℝ) : (5 ∈ ({1, m + 2, m^2 + 4} : set ℝ)) → (m = 3 ∨ m = 1) :=
by
  intro H
  cases H,
  sorry

end determine_m_l178_178798


namespace mean_height_of_basketball_team_is_correct_l178_178224

noncomputable def player_heights : List ℕ := 
  [48, 50, 51, 54, 56, 57, 57, 59, 60, 63, 64, 65, 67, 69, 69, 71, 72, 74]

def total_heights := player_heights.sum

def total_players := player_heights.length

def mean_height := (total_heights : ℚ) / total_players

theorem mean_height_of_basketball_team_is_correct : 
  mean_height = 61.444 :=
by 
  sorry

end mean_height_of_basketball_team_is_correct_l178_178224


namespace cards_per_layer_l178_178132

theorem cards_per_layer (total_cards : ℕ) (layers : ℕ) (h1 : total_cards = 36 * 104) (h2 : layers = 64) : total_cards / layers = 58.5 := 
by
  sorry

end cards_per_layer_l178_178132


namespace problem_solution_l178_178558

theorem problem_solution (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  let N := max (xy + xz + yz)
  let n := min (xy + xz + yz)
  N + 10 * n = 27 := 
by sorry

end problem_solution_l178_178558


namespace polynomial_remainder_l178_178121

noncomputable def Q : ℕ → ℤ → ℤ
| 0, x => 3
| 1, x => 9
| _, x => sorry  -- placeholder for higher degrees, not needed in this proof

theorem polynomial_remainder (Q : ℤ → ℤ)
  (h1 : Q 10 = 3)
  (h2 : Q 15 = 9) :
  ∃ a b, Q x = (x - 10) * (x - 15) * (λ y, (Q y - (a * y + b)) / ((y - 10) * (y - 15))) + (a * x + b) ∧
  a = 6 / 5 ∧ b = -9 :=
by
  sorry

end polynomial_remainder_l178_178121


namespace time_after_3031_minutes_l178_178620

-- Define the starting time
def start_time : String := "5:00 PM"

-- Define the date
def start_date := "December 31, 2020"

-- Define the total minutes
def total_minutes : ℕ := 3031

-- Define the hours and minutes for conversion
def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24

-- Define the expected result 
def expected_date := "January 2, 2021"
def expected_time : String := "7:31 PM"

-- The main theorem statement
theorem time_after_3031_minutes :
  (calculate_time total_minutes start_time start_date = (expected_time, expected_date)) := sorry

-- Helper function definitions would go here

noncomputable def calculate_time 
  (total_minutes : ℕ) 
  (start_time : String) 
  (start_date : String) : (String, String) := 
  sorry

end time_after_3031_minutes_l178_178620


namespace arithmetic_expression_l178_178843

theorem arithmetic_expression : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_expression_l178_178843


namespace P_T_S_collinear_P_K_L_collinear_l178_178151

-- Given conditions
variable (l : Line) (A B P Q R S T K L : Point)

-- Condition statements
axiom A_B_P_on_line_l : A ∈ l ∧ B ∈ l ∧ P ∈ l ∧ A ≠ B ∧ B ≠ P ∧ A ≠ P

axiom line_a : is_perpendicular (line_through A) l
axiom line_b : is_perpendicular (line_through B) l

axiom line_through_P : (line_through P) ≠ l
axiom Q_on_a_and_R_on_b : Q ∈ (line_through A) ∧ R ∈ (line_through B)

axiom line_perp_A_BQ : is_perpendicular (line_through A) (line_through B Q)
axiom L_on_BQ_and_T_on_BR : L ∈ (line_through B Q) ∧ T ∈ (line_through B R)

axiom line_perp_B_AR : is_perpendicular (line_through B) (line_through A R)
axiom K_on_AR_and_S_on_AQ : K ∈ (line_through A R) ∧ S ∈ (line_through A Q)

-- Prove (a): P, T, S collinear
theorem P_T_S_collinear : collinear P T S :=
sorry

-- Prove (b): P, K, L collinear
theorem P_K_L_collinear : collinear P K L :=
sorry

end P_T_S_collinear_P_K_L_collinear_l178_178151


namespace debra_two_tails_in_row_l178_178368

noncomputable def fair_coin_prob : ℚ :=
  let P_HTH := (1/2 : ℚ) * (1/2) * (1/2),
      Q := (1/4 : ℚ) / (1 - 1/4)
  in P_HTH * Q

theorem debra_two_tails_in_row (P_HTH Q : ℚ) (fair_coin : ∀ (n : ℕ), (1/2 : ℚ)) :
  P_HTH = (1/2) * (1/2) * (1/2) →
  Q = (1/4) / (1 - 1/4) →
  fair_coin_prob = 1/24 := by
    intro h_p h_q
    rw [fair_coin_prob, h_p, h_q]
    sorry

end debra_two_tails_in_row_l178_178368


namespace radius_wheel_l178_178589

noncomputable def pi : ℝ := 3.14159

theorem radius_wheel (D : ℝ) (N : ℕ) (r : ℝ) (h1 : D = 760.57) (h2 : N = 500) :
  r = (D / N) / (2 * pi) :=
sorry

end radius_wheel_l178_178589


namespace exists_n_for_all_c_l178_178515

def s (k : ℕ) : ℕ := 
  -- the number of ways to express k as the sum of distinct 2012th powers
  sorry

theorem exists_n_for_all_c (c : ℝ) : ∃ n : ℕ, s(n) > c * n :=
  sorry

end exists_n_for_all_c_l178_178515


namespace bounded_sequence_if_perfect_cube_l178_178454

def largest_cube_le (k : ℤ) : ℤ := (Int.sqrt (k : ℤ^(1/3))).nat_abs^3

theorem bounded_sequence_if_perfect_cube (p : ℕ) :
  (∃ a : ℕ → ℕ, 
    a 0 = p ∧ 
    ( ∀ n : ℕ, a (n + 1) = 3 * a n - 2 * largest_cube_le (a n)) ∧ 
    (bounded (set.range a))
  ) ↔ (∃ k : ℕ, p = k^3) :=
begin
  sorry
end

end bounded_sequence_if_perfect_cube_l178_178454


namespace maria_trip_distance_l178_178292

theorem maria_trip_distance
  (D : ℝ) -- D represents the total distance in miles
  (h1 : 0 < D) -- Total distance must be positive
  (h2 : let dist1 := 1 / 2 * D in true) -- First stop after traveling 1/2 of D
  (h3 : let dist_remain1 := D - 1 / 2 * D in true) -- Remaining distance after first stop, which is also 1/2 * D
  (h4 : let dist2 := 1 / 4 * dist_remain1 in true) -- Second stop after traveling 1/4 of the remaining distance, which is 1/8 * D
  (h5 : let dist_final := 150 in dist_final = dist_remain1 - dist2) -- The remaining distance after the second stop, which equals 150 miles
  : D = 1200 := -- To prove the total distance D equals 1200 miles
sorry -- The actual proof is omitted

end maria_trip_distance_l178_178292


namespace power_of_two_with_starting_digits_l178_178937

noncomputable def log_base_10_2 : ℝ := Real.log 2 / Real.log 10

theorem power_of_two_with_starting_digits (A : ℕ) (hA : 1 ≤ A) : 
  ∃ n m : ℕ, 10^m * A ≤ 2^n ∧ 2^n < 10^m * (A + 1) := 
by
  have h : log_base_10_2 ∉ set.rat := sorry
  sorry

end power_of_two_with_starting_digits_l178_178937


namespace kathryn_remaining_money_l178_178509

/-- Define the conditions --/
def rent := 1200
def salary := 5000
def food_and_travel_expenses := 2 * rent
def new_rent := rent / 2
def total_expenses := food_and_travel_expenses + new_rent
def remaining_money := salary - total_expenses

/-- Theorem to be proved --/
theorem kathryn_remaining_money : remaining_money = 2000 := by
  sorry

end kathryn_remaining_money_l178_178509


namespace sequence_general_term_l178_178416

noncomputable def a₁ : ℕ → ℚ := sorry

variable (S : ℕ → ℚ)

axiom h₀ : a₁ 1 = -1
axiom h₁ : ∀ n : ℕ, a₁ (n + 1) = S n * S (n + 1)

theorem sequence_general_term (n : ℕ) : S n = -1 / n := by
  sorry

end sequence_general_term_l178_178416


namespace optimal_play_result_l178_178002

-- Definitions to match the conditions
def is_sum_one {n : ℕ} (x : Fin (2 * n) → ℝ) : Prop :=
  (∑ i, x i) = 1

def all_non_negative {n : ℕ} (x : Fin (2 * n) → ℝ) : Prop :=
  ∀ i, 0 ≤ x i

-- The largest product from neighboring pairs in a circular arrangement
def largest_neighboring_product {n : ℕ} (x : Fin (2 * n) → ℝ) : ℝ :=
  Finset.sup (Finset.univ.image (λ i, x i * x ((i + 1) % (2 * n))))

-- Proof problem statement
theorem optimal_play_result (n : ℕ) (hn : n ≥ 2) (x : Fin (2 * n) → ℝ) 
    (hsum : is_sum_one x) (hpos : all_non_negative x) :
    largest_neighboring_product x = 1 / (8 * (n - 1)) := 
begin
  sorry
end

end optimal_play_result_l178_178002


namespace binomial_sum_divides_l178_178550

theorem binomial_sum_divides (n : ℕ) : 2^n ∣ ∑ i in Finset.range (n + 1), (Nat.choose (2 * n) (2 * i)) * (3 ^ i) := 
  sorry

end binomial_sum_divides_l178_178550


namespace circles_intersect_l178_178721

-- Define the equations of the circles
def circle1 (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + y ^ 2 = 1

def circle2 (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y + 2) ^ 2 = 9

-- Define the centers and radii for convenience
def center1 : ℝ × ℝ := (1, 0)
def radius1 : ℝ := 1

def center2 : ℝ × ℝ := (-1, -2)
def radius2 : ℝ := 3

-- Distance between centers
def distance_center : ℝ :=
  real.sqrt ((-1 - 1) ^ 2 + (-2 - 0) ^ 2)

theorem circles_intersect : (radius2 - radius1 < distance_center) ∧ (distance_center < radius2 + radius1) :=
by {
  sorry,
}

end circles_intersect_l178_178721


namespace a_increasing_a_sum_bound_l178_178129

variable {a : ℕ → ℝ}

noncomputable def a_seq (n : ℕ) : ℝ :=
if h : n > 0 then
  let a_1 := 1
  let a_n := λ n, (λ a_n, n * (a (n + 1))^2 / (n * a (n + 1) + 1))
  a_n n
else 0

-- Conditions
axiom a_1 (a) : a 1 = 1
axiom a_pos (n) : a n > 0
axiom a_recur (n) : a n = n * (a (n + 1))^2 / (n * a (n + 1) + 1)

-- Proof obligations
theorem a_increasing : ∀ n : ℕ+, a n < a (n + 1) :=
sorry

theorem a_sum_bound (n : ℕ+) : a n < 1 + ∑ k in finset.range (n + 1), 1 / k :=
sorry

end a_increasing_a_sum_bound_l178_178129


namespace parallel_vectors_m_value_l178_178441

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), (∀ k : ℝ, (1 : ℝ) = k * m ∧ (-2) = k * (-1)) -> m = (1 / 2) :=
by
  intros m h
  sorry

end parallel_vectors_m_value_l178_178441


namespace arithmetic_sequence_sum_l178_178484

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_condition : a 2 + a 10 = 16) :
  a 4 + a 8 = 16 :=
sorry

end arithmetic_sequence_sum_l178_178484


namespace identity_eq_l178_178941

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l178_178941


namespace center_of_symmetry_l178_178233

def f (x : ℝ) : ℝ := Real.sin(2 * x)

def g (x : ℝ) : ℝ := Real.sin(2 * x - Real.pi / 6)

theorem center_of_symmetry : g (Real.pi / 12) = 0 := by
  sorry

end center_of_symmetry_l178_178233


namespace shorter_piece_length_l178_178627

theorem shorter_piece_length (total_length : ℝ) (ratio : ℝ) (x : ℝ) :
  total_length = 80 ∧ ratio = 3/5 ∧ total_length = x + (x + ratio * x) →
  x ≈ 30.77 :=
by
  sorry

end shorter_piece_length_l178_178627


namespace distance_between_intersections_l178_178001

theorem distance_between_intersections (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = ax + b)
  (d1 : distance (set_of (λ x => x^2 = f x)) = sqrt 10)
  (d2 : distance (set_of (λ x => x^2 - 1 = f x + 1)) = sqrt 42) :
  distance (set_of (λ x => x^2 + 1 = f x + 2)) = sqrt 26 := 
sorry

end distance_between_intersections_l178_178001


namespace magic_trick_min_n_l178_178911

theorem magic_trick_min_n (colors : ℕ) (n : ℕ) (h_colors : colors = 2017) (n_gt_1 : n > 1) :
  (∃ strategy : (List ℕ → ℕ), ∀ cards : List ℕ, cards.length = n → 
  strategy cards ∈ cards ∧ ∀ helper_choice : ℕ, helper_choice ∈ cards → 
  strategy (turn_down_except_one cards helper_choice) = helper_choice) ↔ n = 2018 :=
sorry

-- Definitions and auxiliary functions for the theorem
def turn_down_except_one (cards : List ℕ) (helper_choice : ℕ) : List ℕ :=
(cards.erase helper_choice) ++ [helper_choice]


end magic_trick_min_n_l178_178911


namespace pyramid_volume_and_surface_area_l178_178994

open Real

variables (a : ℝ) (angle : ℝ) (h_1 : a > 0) (h_2 : angle = π / 4)

theorem pyramid_volume_and_surface_area (a : ℝ) (h_1 : a > 0) (h_2 : angle = π / 4) :
  let V := (a^3) / 24 in
  let S := (a^2 * sqrt(3) * (1 + sqrt(2))) / 4 in
  (volume_of_pyramid a angle = V) ∧ (surface_area_of_pyramid a angle = S) :=
begin
  -- Proof goes here
  sorry
end

end pyramid_volume_and_surface_area_l178_178994


namespace max_three_digit_numbers_divisible_by_25_l178_178665

theorem max_three_digit_numbers_divisible_by_25 
  (a : ℕ → ℕ) (n : ℕ)
  (h1 : 3 ≤ n) 
  (h2 : ∀ k, k ≤ n-2 → a (k + 2) = 3 * a (k + 1) - 2 * a k - 1)
  (h3 : ∃ k ≤ n, a k = 2021) 
  : ∃ m, m = 36 ∧ 
  ∀ x, 100 ≤ a x ∧ a x ≤ 999 → a x % 25 = 0 → 
    ∃ b, 1 ≤ b ∧ b ≤ m := 
begin
  sorry
end

end max_three_digit_numbers_divisible_by_25_l178_178665


namespace triangles_cover_base_l178_178766

theorem triangles_cover_base 
  (n : ℕ) (A : Fin n → Euclidean.Point) (S : Euclidean.Point) 
  (X : Fin n → Euclidean.Point)
  (convex_base : ConvexPolygon (A : Fin (n+1) → Euclidean.Point)) 
  (congruent_triangles : ∀ i : Fin n, 
    is_congruent (Triangle (X i) (A i) (A (i + 1))) (Triangle S (A i) (A (i + 1)))) :
  covers_base (Union (λ i : Fin n, Triangle (X i) (A i) (A (i + 1)))) 
  (ConvexPolygon (A : Fin (n+1) → Euclidean.Point)) :=
sorry

end triangles_cover_base_l178_178766


namespace classroom_students_count_l178_178837

-- Definitions from the conditions
def students (C S Sh : ℕ) : Prop :=
  S = 2 * C ∧
  S = Sh + 8 ∧
  Sh = C + 19

-- Proof statement
theorem classroom_students_count (C S Sh : ℕ) 
  (h : students C S Sh) : 3 * C = 81 :=
by
  sorry

end classroom_students_count_l178_178837


namespace proof_problem_l178_178787

noncomputable def f (x a : ℝ) : ℝ := (1 + x^2) * Real.exp x - a
noncomputable def f' (x a : ℝ) : ℝ := (1 + 2 * x + x^2) * Real.exp x
noncomputable def k_OP (a : ℝ) : ℝ := a - 2 / Real.exp 1
noncomputable def g (m : ℝ) : ℝ := Real.exp m - (m + 1)

theorem proof_problem (a m : ℝ) (h₁ : a > 0) (h₂ : f' (-1) a = 0) (h₃ : f' m a = k_OP a) 
  : m + 1 ≤ 3 * a - 2 / Real.exp 1 := by
  sorry

end proof_problem_l178_178787


namespace not_inequality_l178_178968

theorem not_inequality (x : ℝ) : ¬ (x^2 + 2*x - 3 < 0) :=
sorry

end not_inequality_l178_178968


namespace triangle_BEF_area_l178_178344

theorem triangle_BEF_area (area_ABCD : ℝ) (ratio_ADE_AEB : ℝ) (area_BEFEq : ℝ) :
  area_ABCD = 60 → ratio_ADE_AEB = 2 / 3 → ∃ area_BEF, area_BEF = 12 :=
by
  assume h1 : area_ABCD = 60
  assume h2 : ratio_ADE_AEB = 2 / 3
  sorry

end triangle_BEF_area_l178_178344


namespace part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l178_178623

-- Part (a)
theorem part_a_avg_area_difference : 
  let zahid_avg := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6
  let yana_avg := (21 / 6)^2
  zahid_avg - yana_avg = 35 / 12 := sorry

-- Part (b)
theorem part_b_prob_same_area :
  let prob_zahid_min n := (13 - 2 * n) / 36
  let prob_same_area := (1 / 36) * ((11 / 36) + (9 / 36) + (7 / 36) + (5 / 36) + (3 / 36) + (1 / 36))
  prob_same_area = 1 / 24 := sorry

-- Part (c)
theorem part_c_expected_value_difference :
  let yana_avg := 49 / 4
  let zahid_avg := (11 / 36 * 1^2 + 9 / 36 * 2^2 + 7 / 36 * 3^2 + 5 / 36 * 4^2 + 3 / 36 * 5^2 + 1 / 36 * 6^2)
  (yana_avg - zahid_avg) = 35 / 9 := sorry

end part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l178_178623


namespace parallelogram_height_l178_178734

theorem parallelogram_height (A b h : ℝ) (hA : A = 288) (hb : b = 18) : h = 16 :=
by
  sorry

end parallelogram_height_l178_178734


namespace circle_no_intersect_diagonal_prob_l178_178315

open Classical

noncomputable theory

def probability_no_intersection_of_circle_with_diagonal (rectangle_width rectangle_height circle_radius : ℕ) : ℚ :=
  let reduced_width := rectangle_width - 2 * circle_radius
  let reduced_height := rectangle_height - 2 * circle_radius
  let total_area := reduced_width * reduced_height
  let favorable_area := 375 -- This area corresponds to calculated region from solution part
  favorable_area / total_area

theorem circle_no_intersect_diagonal_prob {ABCD : Type} (circle_radius : ℕ) (rectangle_width : ℕ) (rectangle_height : ℕ) :
  rectangle_width = 15 → 
  rectangle_height = 36 → 
  circle_radius = 1 → 
  probability_no_intersection_of_circle_with_diagonal rectangle_width rectangle_height circle_radius = 375 / 442 :=
by 
  intros h_w h_h h_r
  rw [h_w, h_h, h_r]
  dsimp [probability_no_intersection_of_circle_with_diagonal]
  norm_num

end circle_no_intersect_diagonal_prob_l178_178315


namespace ratio_of_angles_l178_178591

def angle1 : ℝ := 32
def angle2 : ℝ := 96
def angle3 : ℝ := 52

theorem ratio_of_angles (h1 : angle2 = 3 * angle1) : angle2 / angle1 = 3 := by
  calc
    angle2 / angle1 = 96 / 32 : by rw [angle1, angle2]
             ... = 3         : by norm_num

end ratio_of_angles_l178_178591


namespace ratio_of_cost_to_marked_price_l178_178320

-- Define the conditions for the problem
variables {x : ℝ} -- marked price
def selling_price := (3 / 4) * x -- condition 2
def cost_price := (5 / 8) * selling_price -- condition 3

-- State the problem as a theorem
theorem ratio_of_cost_to_marked_price (x : ℝ) (h : x > 0) :
  cost_price / x = 15 / 32 := by
  sorry

end ratio_of_cost_to_marked_price_l178_178320


namespace common_ratio_l178_178936

variable (a_1 d : ℝ)
variable (k n p : ℕ)

def a_k := a_1 + (k - 1) * d
def a_n := a_1 + (n - 1) * d
def a_p := a_1 + (p - 1) * d

theorem common_ratio (q : ℝ) (h1 : a_n = a_k * q) (h2 : a_p = a_n * q) : 
  q = (n - p : ℝ) / (k - n : ℝ) :=
sorry

end common_ratio_l178_178936


namespace sum_of_cubes_identity_l178_178903

theorem sum_of_cubes_identity (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end sum_of_cubes_identity_l178_178903


namespace alice_1496_to_1499_digits_l178_178332

-- Define the sequence of numbers written by Alice
def alice_sequence : List Nat := 
    List.range (10) ++ List.range (20, 100) ++ List.range (200, 300) ++ List.range (2000, 3000)

-- Get the sequence of digits from Alice's sequence
def alice_digit_sequence : List Char :=
    alice_sequence.join.map (λ n => n.toString.data)

-- Define a helper function to extract a specific range of digits from the sequence
def digits_at_range (start n : Nat) (digits : List Char) : List Char :=
    digits.drop (start - 1).take n

-- The main theorem
theorem alice_1496_to_1499_digits :
  digits_at_range 1496 4 alice_digit_sequence = ['5', '8', '2', '2'] :=
  sorry

end alice_1496_to_1499_digits_l178_178332


namespace chips_draw_probability_l178_178312

-- Define all the conditions stated in the problem
def chips_info : Prop :=
  let total_chips := 14
  let tan_chips := 5
  let pink_chips := 3
  let violet_chips := 6
  total_chips = tan_chips + pink_chips + violet_chips &&
  tan_chips = 5 && pink_chips = 3 && violet_chips = 6

-- Define the final probability that needs to be proven
def desired_probability : ℚ := 1 / 168168

-- Prove that the desired probability is correct given the conditions
theorem chips_draw_probability : chips_info →
  (3! * 5! * 6! : ℚ) / 14! = desired_probability :=
by
  intros
  sorry

end chips_draw_probability_l178_178312


namespace kathryn_remaining_money_l178_178508

/-- Define the conditions --/
def rent := 1200
def salary := 5000
def food_and_travel_expenses := 2 * rent
def new_rent := rent / 2
def total_expenses := food_and_travel_expenses + new_rent
def remaining_money := salary - total_expenses

/-- Theorem to be proved --/
theorem kathryn_remaining_money : remaining_money = 2000 := by
  sorry

end kathryn_remaining_money_l178_178508


namespace time_for_trains_to_pass_l178_178613

noncomputable def time_to_pass : ℝ :=
  let length_train1 := 170.0 -- in meters
  let length_train2 := 170.0 -- in meters
  let speed_train1 := 55.0  -- in km/h
  let speed_train2 := 50.0  -- in km/h
  let relative_speed_km_per_h := speed_train1 + speed_train2
  let relative_speed_m_per_s := (relative_speed_km_per_h * 1000) / 3600
  let total_distance := length_train1 + length_train2
  total_distance / relative_speed_m_per_s

theorem time_for_trains_to_pass :
  time_to_pass ≈ 11.66 := by
  sorry

end time_for_trains_to_pass_l178_178613


namespace P_T_S_collinear_P_K_L_collinear_l178_178149

-- Given conditions
variable (l : Line) (A B P Q R S T K L : Point)

-- Condition statements
axiom A_B_P_on_line_l : A ∈ l ∧ B ∈ l ∧ P ∈ l ∧ A ≠ B ∧ B ≠ P ∧ A ≠ P

axiom line_a : is_perpendicular (line_through A) l
axiom line_b : is_perpendicular (line_through B) l

axiom line_through_P : (line_through P) ≠ l
axiom Q_on_a_and_R_on_b : Q ∈ (line_through A) ∧ R ∈ (line_through B)

axiom line_perp_A_BQ : is_perpendicular (line_through A) (line_through B Q)
axiom L_on_BQ_and_T_on_BR : L ∈ (line_through B Q) ∧ T ∈ (line_through B R)

axiom line_perp_B_AR : is_perpendicular (line_through B) (line_through A R)
axiom K_on_AR_and_S_on_AQ : K ∈ (line_through A R) ∧ S ∈ (line_through A Q)

-- Prove (a): P, T, S collinear
theorem P_T_S_collinear : collinear P T S :=
sorry

-- Prove (b): P, K, L collinear
theorem P_K_L_collinear : collinear P K L :=
sorry

end P_T_S_collinear_P_K_L_collinear_l178_178149


namespace find_constant_c_l178_178294

def f: ℝ → ℝ := sorry

noncomputable def constant_c := 8

theorem find_constant_c (h : ∀ x : ℝ, f x + 3 * f (constant_c - x) = x) (h2 : f 2 = 2) : 
  constant_c = 8 :=
sorry

end find_constant_c_l178_178294


namespace age_ratio_in_two_years_l178_178862

-- Definitions based on conditions
def lennon_age_current : ℕ := 8
def ophelia_age_current : ℕ := 38
def lennon_age_in_two_years := lennon_age_current + 2
def ophelia_age_in_two_years := ophelia_age_current + 2

-- Statement to prove
theorem age_ratio_in_two_years : 
  (ophelia_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 4 ∧
  (lennon_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 1 := 
by 
  sorry

end age_ratio_in_two_years_l178_178862


namespace unique_students_at_turing_high_school_l178_178346

theorem unique_students_at_turing_high_school
    (students_riemann : ℕ)
    (students_lovelace : ℕ)
    (students_euler : ℕ)
    (overlap_lovelace_euler : ℕ) :
    students_riemann = 12 →
    students_lovelace = 10 →
    students_euler = 8 →
    overlap_lovelace_euler = 3 →
    students_riemann + (students_lovelace + students_euler - overlap_lovelace_euler) = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end unique_students_at_turing_high_school_l178_178346


namespace intersection_eq_l178_178044

variable {R : Type} [linear_ordered_field R]

def setA (x : R) : Prop := x^2 - 2 * x > 0
def setB (x : R) : Prop := (x + 1) / (x - 1) ≤ 0

theorem intersection_eq : {x : R | setA x} ∩ {x : R | setB x} = {x : R | -1 ≤ x ∧ x < 0} := by
  sorry

end intersection_eq_l178_178044


namespace clock_angle_probability_l178_178325

theorem clock_angle_probability :
  ∃ (m n : ℕ), Nat.coprime m n ∧ (to_nnreal (m / n) = 4 / 11) ∧ (100 * m + n = 411) :=
begin
  use [4, 11],
  split,
  { norm_num },  -- Proves that 4 and 11 are coprime
  split,
  { norm_num },  -- Proves that (4 / 11) = 4/11
  { norm_num },  -- Proves that 100 * 4 + 11 = 411
end

end clock_angle_probability_l178_178325


namespace intersect_inverse_l178_178984

theorem intersect_inverse (c d : ℤ) (h1 : 2 * (-4) + c = d) (h2 : 2 * d + c = -4) : d = -4 := 
by
  sorry

end intersect_inverse_l178_178984


namespace maximal_C_l178_178379

theorem maximal_C (a : Fin 2019 → ℝ) (h : ∀ i j, 0 < a i ∧ a i ≠ a j) :
  ∃ C, C = 1010 ∧ ∀ (a : Fin 2019 → ℝ), 
  ∑ i, (a i) / (abs (a ((i + 1) % 2019) - a ((i + 2) % 2019))) > C :=
begin
  -- Proof omitted
  sorry
end

end maximal_C_l178_178379


namespace negation_exists_ge_zero_l178_178208

theorem negation_exists_ge_zero (h : ∀ x > 0, x^2 - 3 * x + 2 < 0) :
  ∃ x > 0, x^2 - 3 * x + 2 ≥ 0 :=
sorry

end negation_exists_ge_zero_l178_178208


namespace rectangles_in_grid_l178_178361

theorem rectangles_in_grid (rows cols : ℕ) : rows = 3 → cols = 4 → 
  (nat.choose (rows + 1) 2) * (nat.choose (cols + 1) 2) = 60 :=
by intros h_rows h_cols
   rw [h_rows, h_cols]
   rw [nat.choose_succ_succ, nat.choose_succ_succ]
   /- To verify the combinatorial selections and multiplication directly, using sorry for the proof completion -/
   sorry

end rectangles_in_grid_l178_178361


namespace tan_alpha_value_l178_178420

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : sin α ^ 2 + cos (2 * α) = 1 / 4) : tan α = sqrt 3 :=
sorry

end tan_alpha_value_l178_178420


namespace product_of_min_max_values_l178_178527

theorem product_of_min_max_values (x y : ℝ) (h : 3 * x^2 + 6 * x * y + 4 * y^2 = 1) : 
  let k := x^2 + 2 * x * y + y^2 in
  let m := (5 - Real.sqrt 7) / 6 in
  let M := (5 + Real.sqrt 7) / 6 in
  m * M = 8 / 9 :=
by
  sorry

end product_of_min_max_values_l178_178527


namespace proof_problem_l178_178043

open Set

variable {R : Set ℝ} (A B : Set ℝ) (complement_B : Set ℝ)

-- Defining set A
def setA : Set ℝ := { x | 1 < x ∧ x < 3 }

-- Defining set B based on the given functional relationship
def setB : Set ℝ := { x | 2 < x } 

-- Defining the complement of set B (in the universal set R)
def complementB : Set ℝ := { x | x ≤ 2 }

-- The intersection we need to prove is equivalent to the given answer
def intersection_result : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- The theorem statement (no proof)
theorem proof_problem : setA ∩ complementB = intersection_result := 
by
  sorry

end proof_problem_l178_178043


namespace distribution_of_X_median_and_contingency_table_l178_178697

/-- Given 40 mice randomly divided into two groups of 20 each, 
    X is the number of a specified pair of mice assigned to the control group. -/
theorem distribution_of_X :
  ∃ P : ℕ → ℚ, P 0 = 19 / 78 ∧ P 1 = 20 / 39 ∧ P 2 = 19 / 78 ∧
  (0 * P 0 + 1 * P 1 + 2 * P 2 = 1) :=
sorry

/-- Given the increase in body weight data for control and experimental groups,
    prove the median and contingency table. -/
theorem median_and_contingency_table :
  let control_data := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1,
                       32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2],
      exp_data := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8,
                   20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5],
      combined_data := control_data ++ exp_data,
      m := (combined_data.nth (20 - 1) + combined_data.nth (21 - 1)) / 2,
      a := 6, b := 14, c := 14, d := 6 in
  m = 23.4 ∧ 
  (∀ a b c d, (a + b = 20 ∧ c + d = 20 ∧ a + c = 20 ∧ b + d = 20) →
    (40 * (a*d - b*c)^2 / ((a + b)*(c + d)*(a + c)*(b + d)) ≈ 6.4)) ∧
  6.4 > 3.841 :=
sorry

end distribution_of_X_median_and_contingency_table_l178_178697


namespace product_of_distances_constant_l178_178847

noncomputable def point_on_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def distance_to_asymptote (a b : ℝ) (x y : ℝ) (sign : ℝ) : ℝ :=
  abs (a * y - sign * b * x) / sqrt (a^2 + b^2)

theorem product_of_distances_constant
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (x y : ℝ)
  (hP : point_on_hyperbola a b ha hb x y):
  (distance_to_asymptote a b x y 1) * (distance_to_asymptote a b x y (-1)) = (a^2 * b^2) / (a^2 + b^2) :=
begin
  sorry
end

end product_of_distances_constant_l178_178847


namespace problem1_l178_178708

theorem problem1 :
  (15 * (-3 / 4) + (-15) * (3 / 2) + 15 / 4) = -30 :=
by
  sorry

end problem1_l178_178708


namespace find_general_term_of_sequence_l178_178774

theorem find_general_term_of_sequence (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f(x * y) = x * f(y) + y * f(x))
  (a : ℕ → ℝ)
  (h2 : ∀ n : ℕ, a(n + 1) = f (2 ^ (n + 1)))
  (h3 : a 1 = 2) :
  (∀ n : ℕ, a(n + 2) = n * 2 ^ (n + 2)) :=
sorry

end find_general_term_of_sequence_l178_178774


namespace ratio_of_sum_of_evens_l178_178047

def h (n : ℕ) [Even n] : ℕ := ∑ i in Finset.range (n.div2 + 1), 2 * i

theorem ratio_of_sum_of_evens (m k n : ℕ) [Even n] :
  h (m * n) / h (k * n) = (m / k) * (m / k + 1) := 
sorry

end ratio_of_sum_of_evens_l178_178047


namespace B_pow_2023_eq_B_l178_178880

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![Real.cos (π / 4), 0, -Real.sin (π / 4)],
     ![0, 1, 0],
     ![Real.sin (π / 4), 0, Real.cos (π / 4)] ]

theorem B_pow_2023_eq_B : B ^ 2023 = B :=
  sorry

end B_pow_2023_eq_B_l178_178880


namespace total_cost_l178_178135

-- Definitions corresponding to the conditions
def puppy_cost : ℝ := 10
def daily_food_consumption : ℝ := 1 / 3
def food_bag_content : ℝ := 3.5
def food_bag_cost : ℝ := 2
def days_in_week : ℝ := 7
def weeks_of_food : ℝ := 3

-- Statement of the problem
theorem total_cost :
  let 
    days := weeks_of_food * days_in_week,
    total_food_needed := days * daily_food_consumption,
    bags_needed := total_food_needed / food_bag_content,
    food_cost := bags_needed * food_bag_cost
  in
  puppy_cost + food_cost = 14 := 
by
  sorry

end total_cost_l178_178135


namespace smallest_rho_l178_178514

noncomputable def rho (n : ℕ) : ℝ := 1 / (n - 1)

theorem smallest_rho (n : ℕ) (h : n ≥ 2) :
    ∀ x : Fin n → ℝ, (∀ i, 0 < x i) → (∏ i, x i = 1) →
    ∀ r : ℝ, r ≥ rho n →
    (∑ i, (1 / x i)) ≤ (∑ i, x i ^ r) :=
by
  sorry

end smallest_rho_l178_178514


namespace total_profit_proof_l178_178606

open Real

noncomputable theory

def Tom_Jerry_partnership := 
  let total_investment_Tom := 700
  let total_investment_Jerry := 300
  let total_profit := 3000
  let profit_per_effort_each := total_profit / 6
  let remaining_profit := (2 * total_profit) / 3
  let Tom_share := (7 / 10) * remaining_profit
  let Jerry_share := (3 / 10) * remaining_profit
  let total_Tom := Tom_share + profit_per_effort_each
  let total_Jerry := Jerry_share + profit_per_effort_each
  (total_Tom - total_Jerry) = 800

theorem total_profit_proof : 
  (Tom_Jerry_partnership) → (3000 : ℝ) = 3000 :=
by
  sorry

end total_profit_proof_l178_178606


namespace beanie_babies_total_l178_178909

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end beanie_babies_total_l178_178909


namespace volume_of_wedge_l178_178660

theorem volume_of_wedge 
  (r h : ℝ)
  (r_eq : r = 6)
  (h_eq : h = 6)
  (angle : ℝ)
  (angle_eq : angle = 90) :
  let V := π * (r ^ 2) * h in
  let wedge_volume := (angle / 360) * V in
  (wedge_volume ≈ 170) :=
by
  intros
  subst r_eq
  subst h_eq
  subst angle_eq
  sorry

end volume_of_wedge_l178_178660


namespace probability_of_multiple_3_or_8_or_both_ping_pong_ball_probability_l178_178563

theorem probability_of_multiple_3_or_8_or_both : real :=
  let total_balls := 75
  let multiples_of_3 := finset.filter (λ n, n % 3 = 0) (finset.range (total_balls + 1))
  let multiples_of_8 := finset.filter (λ n, n % 8 = 0) (finset.range (total_balls + 1))
  let multiples_of_24 := finset.filter (λ n, n % 24 = 0) (finset.range (total_balls + 1))
  let favorable_outcomes := multiples_of_3.card + multiples_of_8.card - multiples_of_24.card
  favorable_outcomes / total_balls

theorem ping_pong_ball_probability :
  probability_of_multiple_3_or_8_or_both = 31 / 75 :=
sorry

end probability_of_multiple_3_or_8_or_both_ping_pong_ball_probability_l178_178563


namespace min_max_values_f_l178_178196

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l178_178196


namespace increasing_function_range_l178_178435

theorem increasing_function_range (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Icc 2 (⊤ : ℝ), (f' x) ≥ 0) :
  0 ≤ a ∧ a ≤ 4 + 2 * Real.sqrt 5 :=
by {
  sorry
}

end increasing_function_range_l178_178435


namespace min_max_values_l178_178185

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_values : 
  ∃ (min_val max_val : ℝ), 
  min_val = -3 * Real.pi / 2 ∧ 
  max_val = Real.pi / 2 + 2 ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≥ min_val) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≤ max_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = min_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = max_val) := 
by
  sorry

end min_max_values_l178_178185


namespace part_a_part_b_l178_178144

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def collinear (A B C : Point ℝ) : Prop :=
∃ (a b c : ℝ), a * A.x + b * A.y = c ∧ a * B.x + b * B.y = c ∧ a * C.x + b * C.y = c

structure Point (ℝ : Type) [LinearOrderedField ℝ] :=
(x y : ℝ)

variables
  (A B P Q R L T K S : Point ℝ)
  (l a b : set (Point ℝ))
  (hA : A ∈ l)
  (hB : B ∈ l)
  (hP : P ∈ l)
  (hA_B : A ≠ B)
  (hB_P : B ≠ P)
  (hA_P : A ≠ P)
  (ha_per : ∀ x, x ∈ a ↔ ∃ y, x = ⟨0, y⟩)
  (hb_per : ∀ x, x ∈ b ↔ ∃ y, x = ⟨1, y⟩)
  (hPQ : Q ∈ a ∧ ¬Q ∈ l ∧ Q ∈ line_through P Q)
  (hPR : R ∈ b ∧ ¬R ∈ l ∧ R ∈ line_through P R)
  (hL : L ∈ line_through A T ∧ L ∈ line_through B Q ∧ line_through A Q = a ∧ line_through B R = b)
  (hT : T ∈ line_through A T ∧ T ∈ line_through A T)
  (hS : S ∈ line_through B S ∧ S ∈ line_through B Q ∧ line_through A R = a ∧ line_through B Q = b)
  (hK : K ∈ line_through A R ∧ K ∈ line_through B S ∧ K ∈ line_through A R)

theorem part_a : collinear ℝ P T S := sorry

theorem part_b : collinear ℝ P K L := sorry

end part_a_part_b_l178_178144


namespace elmer_saves_20_percent_l178_178724

theorem elmer_saves_20_percent (fuel_efficiency_old_car : ℝ) (fuel_cost_gasoline : ℝ) (h1 : fuel_efficiency_old_car > 0) (h2 : fuel_cost_gasoline > 0) : 
  let fuel_efficiency_new_car := 1.5 * fuel_efficiency_old_car,
      fuel_cost_diesel := 1.2 * fuel_cost_gasoline in
  ((fuel_cost_gasoline - (fuel_cost_diesel * 2/3)) / fuel_cost_gasoline) * 100 = 20 :=
by
  sorry

end elmer_saves_20_percent_l178_178724


namespace main_theorem_l178_178767

noncomputable def find_numbers (a b : ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ,
  ∃ (as : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → as i ≠ as j) ∧
    (∀ i j : Fin n, i ≠ j → Nat.coprime (as i) (as j)) ∧
    (∀ i : Fin (n - 1), f (as (i + 1)) ∣ f (as i))
    
theorem main_theorem (a b : ℕ) (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + a) ∣ f (Int.floor (Real.sqrt n) + b)) :
  find_numbers a b f :=
begin
  sorry
end

end main_theorem_l178_178767


namespace total_viewing_time_l178_178647

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end total_viewing_time_l178_178647


namespace negation_exists_cube_positive_l178_178207

theorem negation_exists_cube_positive :
  ¬ (∃ x : ℝ, x^3 > 0) ↔ ∀ x : ℝ, x^3 ≤ 0 := by
  sorry

end negation_exists_cube_positive_l178_178207


namespace kathryn_remaining_money_l178_178510

/-- Define the conditions --/
def rent := 1200
def salary := 5000
def food_and_travel_expenses := 2 * rent
def new_rent := rent / 2
def total_expenses := food_and_travel_expenses + new_rent
def remaining_money := salary - total_expenses

/-- Theorem to be proved --/
theorem kathryn_remaining_money : remaining_money = 2000 := by
  sorry

end kathryn_remaining_money_l178_178510


namespace second_person_percentage_of_Deshaun_l178_178367

variable (days : ℕ) (books_read_by_Deshaun : ℕ) (pages_per_book : ℕ) (pages_per_day_by_second_person : ℕ)

theorem second_person_percentage_of_Deshaun :
  days = 80 →
  books_read_by_Deshaun = 60 →
  pages_per_book = 320 →
  pages_per_day_by_second_person = 180 →
  ((pages_per_day_by_second_person * days) / (books_read_by_Deshaun * pages_per_book) * 100) = 75 := 
by
  intros days_eq books_eq pages_eq second_pages_eq
  rw [days_eq, books_eq, pages_eq, second_pages_eq]
  simp
  sorry

end second_person_percentage_of_Deshaun_l178_178367


namespace sold_on_saturday_l178_178920

-- Define all the conditions provided in the question
def amount_sold_thursday : ℕ := 210
def amount_sold_friday : ℕ := 2 * amount_sold_thursday
def amount_sold_sunday (S : ℕ) : ℕ := (S / 2)
def total_planned_sold : ℕ := 500
def excess_sold : ℕ := 325

-- Total sold is the sum of sold amounts from Thursday to Sunday
def total_sold (S : ℕ) : ℕ := amount_sold_thursday + amount_sold_friday + S + amount_sold_sunday S

-- The theorem to prove
theorem sold_on_saturday : ∃ S : ℕ, total_sold S = total_planned_sold + excess_sold ∧ S = 130 :=
by
  sorry

end sold_on_saturday_l178_178920


namespace math_books_count_l178_178638

theorem math_books_count (M H : ℤ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end math_books_count_l178_178638


namespace rotate_image_eq_A_l178_178283

def image_A : Type := sorry -- Image data for option (A)
def original_image : Type := sorry -- Original image data

def rotate_90_clockwise (img : Type) : Type := sorry -- Function to rotate image 90 degrees clockwise

theorem rotate_image_eq_A :
  rotate_90_clockwise original_image = image_A :=
sorry

end rotate_image_eq_A_l178_178283


namespace perpendicular_parallel_implies_perpendicular_l178_178018

-- Definitions of lines and plane
variable (l m : Line) (α : Plane)

-- Proof problem setup
theorem perpendicular_parallel_implies_perpendicular 
  (hne : l ≠ m) (hl_perp_alpha : l ⊥ α) (hl_parallel_m : l ∥ m) : m ⊥ α := 
sorry

end perpendicular_parallel_implies_perpendicular_l178_178018


namespace students_need_to_raise_each_l178_178671

def initial_amount_needed (num_students : ℕ) (amount_per_student : ℕ) (misc_expenses : ℕ) : ℕ :=
  (num_students * amount_per_student) + misc_expenses

def amount_raised_first_three_days (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  day1 + day2 + day3

def amount_raised_next_four_days (first_three_days_total : ℕ) : ℕ :=
  first_three_days_total / 2

def total_amount_raised_in_week (first_three_days_total : ℕ) (next_four_days_total : ℕ) : ℕ :=
  first_three_days_total + next_four_days_total

def amount_each_student_still_needs_to_raise 
  (total_needed : ℕ) (total_raised : ℕ) (num_students : ℕ) : ℕ :=
  if num_students > 0 then (total_needed - total_raised) / num_students else 0

theorem students_need_to_raise_each 
  (num_students : ℕ) (amount_per_student : ℕ) (misc_expenses : ℕ)
  (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) (next_half_factor : ℕ)
  (h_num_students : num_students = 6)
  (h_amount_per_student : amount_per_student = 450)
  (h_misc_expenses : misc_expenses = 3000)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 900)
  (h_day3 : day3 = 400)
  (h_next_half_factor : next_half_factor = 2) :
  amount_each_student_still_needs_to_raise
    (initial_amount_needed num_students amount_per_student misc_expenses)
    (total_amount_raised_in_week
      (amount_raised_first_three_days day1 day2 day3)
      (amount_raised_next_four_days (amount_raised_first_three_days day1 day2 day3 / h_next_half_factor)))
    num_students = 475 :=
by sorry

end students_need_to_raise_each_l178_178671


namespace largest_integer_satisfying_condition_l178_178897

-- Definition of the conditions
def has_four_digits_in_base_10 (n : ℕ) : Prop :=
  10^3 ≤ n^2 ∧ n^2 < 10^4

-- Proof statement: N is the largest integer satisfying the condition
theorem largest_integer_satisfying_condition : ∃ (N : ℕ), 
  has_four_digits_in_base_10 N ∧ (∀ (m : ℕ), has_four_digits_in_base_10 m → m ≤ N) ∧ N = 99 := 
sorry

end largest_integer_satisfying_condition_l178_178897


namespace polygon_sides_in_arithmetic_progression_l178_178203

theorem polygon_sides_in_arithmetic_progression 
  (n : ℕ) 
  (d : ℕ := 3)
  (max_angle : ℕ := 150)
  (sum_of_interior_angles : ℕ := 180 * (n - 2)) 
  (a_n : ℕ := max_angle) : 
  (max_angle - d * (n - 1) + max_angle) * n / 2 = sum_of_interior_angles → 
  n = 28 :=
by 
  sorry

end polygon_sides_in_arithmetic_progression_l178_178203


namespace min_max_f_l178_178192

noncomputable def f (x : ℝ) : ℝ := Math.cos x + (x + 1) * Math.sin x + 1

theorem min_max_f :
  ∃ a b : ℝ, a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a) ∧ 
  (∃ y ∈ Set.Icc 0 (2 * Real.pi), f y = a) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ b) ∧ 
  (∃ z ∈ Set.Icc 0 (2 * Real.pi), f z = b) :=
sorry

end min_max_f_l178_178192


namespace husk_consumption_l178_178475

theorem husk_consumption (number_of_cows : ℕ) (number_of_days : ℕ) (consumption_per_cow : ℕ) :
  number_of_cows = 26 ∧ number_of_days = 26 ∧ consumption_per_cow = 1 →
  number_of_cows * consumption_per_cow = 26 :=
by
  intro h
  obtain ⟨hcows, hd, hconsumption⟩ := h
  rw [hcows, hconsumption]
  exact rfl

end husk_consumption_l178_178475


namespace daniela_total_spending_l178_178365

def initial_cost (prices : List ℕ) (quantities : List ℕ) : ℕ :=
  prices.zip quantities |>.map (fun (p, q) => p * q) |>.sum

def category_discount (initial_cost : ℕ) (discount_rate : ℕ) : ℕ :=
  (discount_rate * initial_cost) / 100

def final_cost_each_category (prices : List ℕ) (quantities : List ℕ) (discounts : List ℕ) : ℕ :=
  prices.zip3 quantities discounts |>.map (fun (p, q, d) => p * q - category_discount (p * q) d) |>.sum

def additional_discount (total_cost : ℕ) (threshold : ℕ) (discount_rate : ℕ) : ℕ :=
  if total_cost > threshold then (total_cost * discount_rate) / 100 else 0

theorem daniela_total_spending :
  let prices := [60, 120, 25]
  let quantities := [3, 2, 3]
  let discounts := [30, 15, 50]
  let threshold := 200
  let additional_disc_rate := 10
  let initial := initial_cost prices quantities
  let after_category_discount := final_cost_each_category prices quantities discounts
  let final_discount := additional_discount after_category_discount threshold additional_disc_rate
  in initial = 495 ∧
     after_category_discount = 367.5 ∧
     (after_category_discount - final_discount) = 330.75 :=
begin
  sorry
end

end daniela_total_spending_l178_178365


namespace solve_system_eqns_l178_178570

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end solve_system_eqns_l178_178570


namespace greatest_int_less_than_200_gcd_30_is_5_l178_178246

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l178_178246


namespace train_speed_is_50_meters_per_second_l178_178310

-- Define the conditions
def train_length : ℝ := 300
def crossing_time : ℝ := 6

-- Define the speed calculation
def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Statement to be proven
theorem train_speed_is_50_meters_per_second : speed train_length crossing_time = 50 := 
by 
  sorry

end train_speed_is_50_meters_per_second_l178_178310


namespace intersection_of_A_and_B_l178_178467

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} :=
by
  sorry

end intersection_of_A_and_B_l178_178467


namespace necessary_but_not_sufficient_l178_178347

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x < 0 ∨ y < 0) → x + y < 0 :=
sorry

end necessary_but_not_sufficient_l178_178347


namespace kathryn_remaining_money_l178_178506

variables (rent food_travel salary monthly_expenses remaining : ℝ)

-- Conditions
def rent_value := rent = 1200
def food_travel_expenses := food_travel = 2 * rent
def salary_value := salary = 5000
def shared_rent := monthly_expenses = rent / 2 + food_travel

-- Question and Answer
def money_remaining := remaining = salary - monthly_expenses

-- Theorem to prove
theorem kathryn_remaining_money (h1 : rent_value) (h2 : food_travel_expenses) (h3 : salary_value) (h4 : shared_rent) : money_remaining :=
sorry

end kathryn_remaining_money_l178_178506


namespace exponent_addition_l178_178239

theorem exponent_addition : ((7:ℝ)⁻³)^0 + (7^0)^4 = 2 := by
  have h1 : (7:ℝ)⁻³ ^ 0 = 1 := by 
    rw [pow_zero, inv_pow, pow_neg, zero_pow]
    exact zero_lt_one
  have h2 : (7^0:ℝ)^4 = 1 := by 
    rw [pow_zero, one_pow, pow_one]
  rw [h1, h2]
  norm_num
  done

end exponent_addition_l178_178239


namespace root_close_to_zero_l178_178531

noncomputable def g (x : ℝ) : ℝ := 2^(2 * x - 1) + x - 1
noncomputable def f (x : ℝ) : ℝ := 2 * x - 1

theorem root_close_to_zero :
  ∃ x₁ x₂ : ℝ, (g x₁ = 0 ∧ f x₂ = 0) ∧ |x₁ - x₂| ≤ 1 / 4 :=
begin
  -- The proof for the theorem would go here.
  sorry
end

end root_close_to_zero_l178_178531


namespace right_triangle_one_right_angle_l178_178816

theorem right_triangle_one_right_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90 ∨ B = 90 ∨ C = 90) : (interp A B C).count (90) = 1 :=
by
  sorry

end right_triangle_one_right_angle_l178_178816


namespace number_of_sets_sum_150_l178_178990

noncomputable def sum_of_integers (a n : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

theorem number_of_sets_sum_150 : 
  (finset.filter (λ n, ∃ a, n ≥ 3 ∧ sum_of_integers a n = 150) (finset.range 301)).card = 4 :=
sorry

end number_of_sets_sum_150_l178_178990


namespace circle_eq_tangent_x_axis_l178_178371

theorem circle_eq_tangent_x_axis (h k r : ℝ) (x y : ℝ)
  (h_center : h = -5)
  (k_center : k = 4)
  (tangent_x_axis : r = 4) :
  (x + 5)^2 + (y - 4)^2 = 16 :=
sorry

end circle_eq_tangent_x_axis_l178_178371


namespace pyramid_volume_l178_178175

noncomputable def volume_of_pyramid (a b : ℝ) : ℝ :=
  (a^2 * b * Real.sqrt 3) / 12

theorem pyramid_volume (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_lap : ∀ {A B C : ℝ}, 
  is_triangle A B C ∧ A = a ∧ B = a ∧ C = b ∧ lateral_edges_inclined A B C 60) : 
  volume_of_pyramid a b = (a^2 * b * Real.sqrt 3) / 12 := 
sorry

end pyramid_volume_l178_178175


namespace prob_exactly_two_white_prob_at_least_two_white_l178_178698

-- Condition definitions
def urn : Type := finset (fin 8)
def is_white (ball : fin 8) : Prop := ball.val < 3  -- indices 0, 1, 2 are white balls
def is_black (ball : fin 8) : Prop := ball.val >= 3  -- indices 3, 4, 5, 6, 7 are black balls

-- Probability definitions
noncomputable def probability_white : ℚ := 3 / 8
noncomputable def probability_black : ℚ := 5 / 8

-- Question (a)
theorem prob_exactly_two_white :
  (probability_white ^ 2 * probability_black) * 3 = 135 / 512 :=
sorry

-- Question (b)
theorem prob_at_least_two_white :
  (probability_white ^ 2 * probability_black) * 3 + (probability_white ^ 3) = 81 / 256 :=
sorry

end prob_exactly_two_white_prob_at_least_two_white_l178_178698


namespace find_tan_of_sin_sin2_given_cond_l178_178421

variable {α : ℝ}

theorem find_tan_of_sin_sin2_given_cond
    (h1 : α ∈ Ioo 0 (π / 2))
    (h2 : sin α ^ 2 + sin (2 * α) = 1) : tan α = 1 / 2 := 
by
  sorry

end find_tan_of_sin_sin2_given_cond_l178_178421


namespace AI_perpendicular_PQ_and_AI_eq_PQ_l178_178881

-- Definitions for geometric notions and properties
variables {A B C D P Q I : Type*} 

-- Assumptions about the triangle and related geometric elements
variables [right_triangle : ∠A = 90°] [AB_lt_AC : AB < AC] [altitude_AD : is_altitude A D BC] 
variables [incircle_P : is_incenter P A B D] [incircle_Q : is_incenter Q A C D] [incircle_I : is_incenter I A B C]

-- The theorem to prove
theorem AI_perpendicular_PQ_and_AI_eq_PQ : is_perpendicular (line_through A I) (line_through P Q) ∧ dist A I = dist P Q :=
by sorry

end AI_perpendicular_PQ_and_AI_eq_PQ_l178_178881


namespace relationship_between_coefficients_l178_178374

theorem relationship_between_coefficients (a b c : ℚ) 
  (h : ∃ (α β : ℚ), β = 3 * α ∧ (α + β = -b / a) ∧ (α * β = c / a)) : 
  3 * b ^ 2 = 16 * a * c :=
sorry

end relationship_between_coefficients_l178_178374


namespace identity_verification_l178_178953

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l178_178953


namespace arithmetic_sequence_check_l178_178457

theorem arithmetic_sequence_check 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h : ∀ n : ℕ, a (n+1) = a n + d) 
  : (∀ n : ℕ, (a n + 1) - (a (n - 1) + 1) = d) 
    ∧ (∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 2 * d)
    ∧ (∀ n : ℕ, a (n + 1) - (a n + n) = d + 1) := 
by
  sorry

end arithmetic_sequence_check_l178_178457


namespace books_sold_on_thursday_l178_178874

theorem books_sold_on_thursday :
  ∀ (initial_stock : ℕ) (sold_mon sold_tue sold_wed sold_fri : ℕ) (percentage_not_sold : ℚ),
  initial_stock = 1300 →
  sold_mon = 75 →
  sold_tue = 50 →
  sold_wed = 64 →
  sold_fri = 135 →
  percentage_not_sold = 69.07692307692308 →
  let percentage_sold : ℚ := 100 - percentage_not_sold in
  let total_sold : ℕ := (percentage_sold / 100 * initial_stock).toNat in
  let sold_thr := total_sold - (sold_mon + sold_tue + sold_wed + sold_fri) in
  sold_thr = 78 := 
by
  intros initial_stock sold_mon sold_tue sold_wed sold_fri percentage_not_sold 
         stock_eq mon_eq tue_eq wed_eq fri_eq perc_not_sold_eq
  let percentage_sold := 100 - percentage_not_sold
  let total_sold : ℕ := (percentage_sold / 100 * initial_stock).toNat
  have sold_thr := total_sold - (sold_mon + sold_tue + sold_wed + sold_fri)
  rw [stock_eq, mon_eq, tue_eq, wed_eq, fri_eq, perc_not_sold_eq] at *
  -- Proof omitted
  sorry

end books_sold_on_thursday_l178_178874


namespace greatest_integer_with_gcd_30_eq_5_l178_178256

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l178_178256


namespace smallest_period_f4_l178_178337

-- Define the functions
def f1 (x : Real) : Real := 2018 * sin x
def f2 (x : Real) : Real := sin (2018 * x)
def f3 (x : Real) : Real := -cos (2 * x)
def f4 (x : Real) : Real := sin (4 * x + π / 4)

-- Define their respective periods
def period_f1 : Real := 2 * π
def period_f2 : Real := π / 1009
def period_f3 : Real := π
def period_f4 : Real := π / 2

theorem smallest_period_f4 :
  period_f4 = min (min (min period_f1 period_f2) period_f3) period_f4 :=
by
  sorry

end smallest_period_f4_l178_178337


namespace solve_for_x_l178_178567

variable x : ℝ

theorem solve_for_x (h : 4^x + 18 = 5 * 4^x - 50) : x = Real.log 17 / Real.log 4 :=
by
  sorry

end solve_for_x_l178_178567


namespace find_xy_value_l178_178776

theorem find_xy_value : 
  ∀ (a b c : ℝ),
  3 * a + 2 * b + c = 5 → 
  2 * a + b - 3 * c = 1 → 
  a ≥ 0 → 
  b ≥ 0 → 
  c ≥ 0 →
  let m := 3 * a + b - 7 * c,
      x := sup (set_of (λ (m: ℝ), ∃ (a b c : ℝ), 3 * a + 2 * b + c = 5 ∧ 2 * a + b - 3 * c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ m = 3 * a + b - 7 * c)),
      y := inf (set_of (λ (m: ℝ), ∃ (a b c : ℝ), 3 * a + 2 * b + c = 5 ∧ 2 * a + b - 3 * c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ m = 3 * a + b - 7 * c))
  in x * y = 5 / 77 :=
by sorry

end find_xy_value_l178_178776


namespace evaluate_fraction_l178_178458

variable (a b x : ℝ)
variable (h1 : a ≠ b)
variable (h2 : b ≠ 0)
variable (h3 : x = a / b)

theorem evaluate_fraction :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_fraction_l178_178458


namespace distance_inequality_of_centers_l178_178394

open EuclideanGeometry

variables {α β γ : ℝ} -- Angles of triangle ABC
variables {a b c : ℝ} -- Sides of triangle ABC
variables {A B C A_1 B_1 C_1 I H: Point} -- Points in the triangles

noncomputable def point_on_side (A B P : Point) : Prop :=
  ∃ r, 0 < r ∧ r < 1 ∧ A + r • (B - A) = P

noncomputable def angle_bisector (A B C P : Point) : Prop :=
  inner (B - A) (C - P) / ∥C - P∥ = inner (C - A) (B - P) / ∥B - P∥

noncomputable def incenter (ABC : Triangle) (I : Point) : Prop := 
  I = insphere_center ABC

noncomputable def orthocenter (A1 B1 C1 H : Triangle) (H : Point) : Prop :=
  H = orthocenter A1 B1 C1

theorem distance_inequality_of_centers 
  (h_acute : is_acute A B C)
  (h_pts sides: point_on_side B C A_1)
  (h_pt_B1: point_on_side C A B_1)
  (h_pt_C1: point_on_side A B C_1)
  (h_bisector_A1: angle_bisector A B C A_1)
  (h_bisector_B1: angle_bisector B C A B_1)
  (h_bisector_C1: angle_bisector C A B C_1)
  (h_incenter: incenter ⟨A, B, C⟩ I)
  (h_orthocenter: orthocenter ⟨A_1, B_1, C_1⟩ H):
  (distance A H) + (distance B H) + (distance C H) ≥
  (distance A I) + (distance B I) + (distance C I) := 
sorry

end distance_inequality_of_centers_l178_178394


namespace number_of_valid_pairs_l178_178323

-- Definition of the conditions according to step (a)
def perimeter (l w : ℕ) : Prop := 2 * (l + w) = 80
def integer_lengths (l w : ℕ) : Prop := true
def length_greater_than_width (l w : ℕ) : Prop := l > w

-- The mathematical proof problem according to step (c)
theorem number_of_valid_pairs : ∃ n : ℕ, 
  (∀ l w : ℕ, perimeter l w → integer_lengths l w → length_greater_than_width l w → ∃! pair : (ℕ × ℕ), pair = (l, w)) ∧
  n = 19 :=
by 
  sorry

end number_of_valid_pairs_l178_178323


namespace percentage_decrease_in_breadth_l178_178326

variable (L B : ℝ) (P : ℝ)

noncomputable def original_area : ℝ := L * B
noncomputable def new_length : ℝ := 0.80 * L
noncomputable def new_breadth : ℝ := B * (1 - P/100)
noncomputable def new_area : ℝ := new_length * new_breadth

theorem percentage_decrease_in_breadth :
  new_area L B P = (0.72 * original_area L B) → P = 10 := by
  sorry

end percentage_decrease_in_breadth_l178_178326


namespace locus_B_is_ellipse_tangents_parallel_l178_178024

variables (x y : ℝ)
def Q : Prop := (x + 4)^2 + y^2 = 72
def A : Prop := x = 4 ∧ y = 0
variables (P M N G H : ℝ × ℝ)
def B : Prop := ∃ (D : ℝ × ℝ), (Q D) ∧ perpendicular_bisector (A, D) (line D (-(x + 4), y)) (B, D)

theorem locus_B_is_ellipse :
  (B P) →
  ∃ a b : ℝ, a^2 = 18 ∧ b^2 = 2 ∧ ∀ {x y : ℝ}, ((x / a)^2 + (y / b)^2 = 1) ↔ (P = (x, y)) :=
sorry

theorem tangents_parallel :
  (B P) ∧ 
  (similar (curve C) (curve T)) ∧ 
  (curve T passes_through (-3, 0)) ∧ 
  (curve T passes_through (3, 0)) →
  tangents P C T (M, N) →
  intersects P C G H (B G) (B H) →
  parallel (line M N) (line G H) :=
sorry

end locus_B_is_ellipse_tangents_parallel_l178_178024


namespace teacher_wins_eventually_l178_178410

/-- 
  There is a game on an infinite grid between 30 students and 1 teacher.
  Players take turns (starting with the teacher) coloring unit segments that
  form boundaries between adjacent cells. A segment cannot be colored twice.
  The teacher wins if after any player's move, a $1 \times 2$ or $2 \times 1$ 
  rectangle exists with a fully colored boundary but uncolored interior segment.
  This statement proves that the students cannot prevent the teacher from winning eventually.
-/
theorem teacher_wins_eventually 
  (grid : Type) [infinite grid]
  (teacher_moves : ℕ)
  (student_moves : ℕ)
  (n_students : ℕ := 30)
  (valid_move : grid → Prop)
  (teacher_wins : grid → Prop) :
  ¬∀ (students_defense : grid → Prop) (max_moves : ℕ),
    (max_moves = student_moves * n_students 
     → ∀ t (h : teacher_moves ≤ t ≤ max_moves), ¬teacher_wins t) :=
sorry

end teacher_wins_eventually_l178_178410


namespace lines_parallel_iff_m_eq_1_l178_178439

-- Define the two lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y = 2 - m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 4 * y = -16

-- Parallel lines condition
def parallel_condition (m : ℝ) : Prop := (1 * 4 - 2 * m * (1 + m) = 0) ∧ (1 * 16 - 2 * m * (m - 2) ≠ 0)

-- The theorem to prove
theorem lines_parallel_iff_m_eq_1 (m : ℝ) : l1 m = l2 m → parallel_condition m → m = 1 :=
by 
  sorry

end lines_parallel_iff_m_eq_1_l178_178439


namespace interval_monotonic_increase_area_triangle_ABC_l178_178034

noncomputable def f (x : ℝ) : ℝ :=
  cos x * sin x + sqrt 3 * (cos x)^2 - sqrt 3 / 2

theorem interval_monotonic_increase {k : ℤ} :
  ∀ x, (k * π - 5 * π / 12) ≤ x → x ≤ (k * π + π / 12) → (∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

theorem area_triangle_ABC (A : ℝ) (a b : ℝ) (fA : f A = 1/2) (acute : A < π / 2) (a_eq : a = 3) (b_eq : b = 4) :
  ∃ c, (area : ℝ := 1/2 * b * c * sin A), area = 4 + sqrt 2 :=
sorry

end interval_monotonic_increase_area_triangle_ABC_l178_178034


namespace travel_time_difference_is_58_minutes_l178_178918

-- Define the distances and speeds for Minnie
def minnie_uphill_distance := 15
def minnie_uphill_speed := 10
def minnie_downhill_distance := 25
def minnie_downhill_speed := 40
def minnie_flat_distance := 30
def minnie_flat_speed := 25

-- Define the distances and speeds for Penny
def penny_flat_distance := 30
def penny_flat_speed := 35
def penny_downhill_distance := 25
def penny_downhill_speed := 50
def penny_uphill_distance := 15
def penny_uphill_speed := 15

-- Calculate Minnie's total travel time in hours
def minnie_time := (minnie_uphill_distance / minnie_uphill_speed) + 
                   (minnie_downhill_distance / minnie_downhill_speed) + 
                   (minnie_flat_distance / minnie_flat_speed)

-- Calculate Penny's total travel time in hours
def penny_time := (penny_flat_distance / penny_flat_speed) + 
                  (penny_downhill_distance / penny_downhill_speed) +
                  (penny_uphill_distance / penny_uphill_speed)

-- Calculate difference in minutes
def time_difference_minutes := (minnie_time - penny_time) * 60

-- The proof statement
theorem travel_time_difference_is_58_minutes :
  time_difference_minutes = 58 := by
  sorry

end travel_time_difference_is_58_minutes_l178_178918


namespace steps_to_return_to_start_l178_178082

-- Define the movement rule
def movement_rule (n : ℕ) : ℤ :=
  if Nat.Prime n then 2 else -3

-- Define the total movement after n moves
def total_movement (n : ℕ) : ℤ :=
  (List.range' 2 (n - 1)).sum movement_rule

-- Define the problem statement as a theorem
theorem steps_to_return_to_start : total_movement 30 = -37 :=
by
  sorry

end steps_to_return_to_start_l178_178082


namespace solve_for_x_l178_178105

def star (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem solve_for_x : ∃ x : ℝ, star 7 (star 3 (x - 1)) = 3 ∧ x = 3 / 4 :=
begin
  sorry
end

end solve_for_x_l178_178105


namespace apples_per_hour_l178_178140

def total_apples : ℕ := 15
def hours : ℕ := 3

theorem apples_per_hour : total_apples / hours = 5 := by
  sorry

end apples_per_hour_l178_178140


namespace min_max_values_l178_178188

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_values : 
  ∃ (min_val max_val : ℝ), 
  min_val = -3 * Real.pi / 2 ∧ 
  max_val = Real.pi / 2 + 2 ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≥ min_val) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≤ max_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = min_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = max_val) := 
by
  sorry

end min_max_values_l178_178188


namespace problem_equivalent_proof_l178_178706

theorem problem_equivalent_proof :
  (∑ k in Finset.range 10, Real.log2 (2^(k^2)) / Real.log2 (4^k)) *
  (∑ k in Finset.range 50, Real.log2 (16^k) / Real.log2 (8^k)) = 1833 + 1 / 3 := 
by
  sorry

end problem_equivalent_proof_l178_178706


namespace identify_substances_correct_l178_178970

open Classical

noncomputable def identify_substances (A B C : Type) : Prop :=
  (A = ethanol) ∧ (B = ethylene) ∧ (C = bromoethane)

theorem identify_substances_correct :
  ∀ (A B C : Type), 
    (∃ (A : Type), colorless A ∧ peculiar_smell A ∧ lighter_than A water ∧ soluble_in A water) →
    (∃ (B : Type), gas_produced A B ∧ lighter_than B air ∧ dehydration_reaction A B) →
    (∃ (C : Type), C = bromoethane ∧ reacts_with B HBr C)
    → identify_substances A B C :=
by
  intros A B C hA hB hC
  unfold identify_substances
  sorry

end identify_substances_correct_l178_178970


namespace goldfish_sold_l178_178213

/-- The pet store buys a goldfish for $0.25 and sells it for $0.75.
    The new tank costs $100 and the owner is 45% short of the price. 
    Prove that the owner sold 110 goldfish that week. -/
theorem goldfish_sold (buy_price sell_price tank_cost shortfall_percentage : ℝ)
  (h_buy : buy_price = 0.25)
  (h_sell : sell_price = 0.75)
  (h_tank : tank_cost = 100)
  (h_shortfall : shortfall_percentage = 0.45) :
  ∃ (n : ℕ), n = 110 :=
begin
  sorry
end

end goldfish_sold_l178_178213


namespace combinations_sol_eq_l178_178753

theorem combinations_sol_eq (x : ℕ) (h : Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) : x = 1 ∨ x = 3 := sorry

end combinations_sol_eq_l178_178753


namespace prove_triangle_center_is_centroid_l178_178841

noncomputable def triangle_center (A B C O : Point) (EF PQ HG BC CA AB : Line) : Point :=
  (IsAcuteTriangle A B C) ∧
  (IsInsideTriangle O A B C) ∧
  (ParallelLines EF BC) ∧
  (ParallelLines PQ CA) ∧
  (ParallelLines HG AB) ∧
  (Ratio EF BC = 2/3) ∧
  (Ratio PQ CA = 2/3) ∧
  (Ratio HG AB = 2/3) →
  (IsCentroid O A B C)
  
theorem prove_triangle_center_is_centroid (A B C O : Point) (EF PQ HG BC CA AB : Line) :
  triangle_center A B C O EF PQ HG BC CA AB :=
by
  sorry

end prove_triangle_center_is_centroid_l178_178841


namespace probability_interval_constant_term_l178_178778

noncomputable def expand_constant_term : ℝ :=
  (nat.choose 4 1) * (1 : ℝ / (2 : ℝ))^3 * (6 : ℝ)

theorem probability_interval_constant_term :
  let X := measure_theory.gaussian_measure (1 : ℝ) (1 : ℝ) in
  P(3 < X < 4) = 0.0214 :=
by
  sorry

end probability_interval_constant_term_l178_178778


namespace sum_of_digits_of_N_l178_178342

theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2016) : (N.digits.sum = 9) :=
sorry

end sum_of_digits_of_N_l178_178342


namespace find_sum_of_roots_l178_178431

noncomputable theory

def p (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5 * x

theorem find_sum_of_roots (h k : ℝ) (h_root : p h = 1) (k_root : p k = 5) : h + k = 2 := 
sorry

end find_sum_of_roots_l178_178431


namespace S_inter_T_eq_S_l178_178825

def S : Set ℝ := { y | ∃ x : ℝ, y = 3 * x }
def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

theorem S_inter_T_eq_S : S ∩ T = S :=
sorry

end S_inter_T_eq_S_l178_178825


namespace repeating_decimal_sum_l178_178279

theorem repeating_decimal_sum :
  let x := 0.1414141414 -- for the purposes of representing the repeating decimal pattern
  let frac := 14 / 99 in -- representing the repeating decimal as a fraction 
  (frac.num + frac.denom) = 113 := 
by
  sorry

end repeating_decimal_sum_l178_178279


namespace number_of_moles_of_methanol_formed_l178_178030

def ch4_to_co2 : ℚ := 1
def o2_to_co2 : ℚ := 2
def co2_prod_from_ch4 (ch4 : ℚ) : ℚ := ch4 * ch4_to_co2 / o2_to_co2

def co2_to_ch3oh : ℚ := 1
def h2_to_ch3oh : ℚ := 3
def ch3oh_prod_from_co2 (co2 h2 : ℚ) : ℚ :=
  min (co2 / co2_to_ch3oh) (h2 / h2_to_ch3oh)

theorem number_of_moles_of_methanol_formed :
  (ch3oh_prod_from_co2 (co2_prod_from_ch4 5) 10) = 10/3 :=
by
  sorry

end number_of_moles_of_methanol_formed_l178_178030


namespace clock_angles_7_23_and_7_53_l178_178597

open Real

noncomputable def hour_angle (h : ℝ) : ℝ := 30 * h
noncomputable def minute_angle (m : ℝ) : ℝ := 6 * m

theorem clock_angles_7_23_and_7_53 :
  (∃ (m1 m2 : ℝ), 
    minute_angle m1 - hour_angle (7 + m1 / 60) = 84 ∧  
    minute_angle m2 - hour_angle (7 + m2 / 60) = 84 ∧
    0 ≤ m1 ∧ m1 ≤ 60 ∧
    0 ≤ m2 ∧ m2 ≤ 60) → 
    ((m1 = 23 ∧ m2 = 53) ∨ (m1 = 53 ∧ m2 = 23)) :=
begin
  sorry
end

end clock_angles_7_23_and_7_53_l178_178597


namespace projection_of_c_onto_a_l178_178803

variable (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ)
variable h_coplanar : matrix.det ![[2, -1, 3], [-1, 4, -2], [7, 5, 65 / 9]] = 0

theorem projection_of_c_onto_a :
  let dot_product := (2 : ℝ) * 7 + (-1) * 5 + (3 : ℝ) * (65 / 9)
  let magnitude_a := real.sqrt ((2 : ℝ)^2 + (-1)^2 + (3: ℝ)^2)
  (dot_product / magnitude_a) = (107 * real.sqrt 14) / 6 :=
by
  sorry

end projection_of_c_onto_a_l178_178803


namespace find_extrema_of_f_l178_178785

noncomputable def f (x : ℝ) := x^2 - 4 * x - 2

theorem find_extrema_of_f : 
  (∀ x, (1 ≤ x ∧ x ≤ 4) → f x ≤ -2) ∧ 
  (∃ x, (1 ≤ x ∧ x ≤ 4 ∧ f x = -6)) :=
by sorry

end find_extrema_of_f_l178_178785


namespace area_of_ABE_l178_178167

noncomputable def square_area_side {α : Type*} [LinearOrderedField α] (area : α) : α :=
  real.sqrt area

noncomputable def triangle_area {α : Type*} [LinearOrderedField α] (base height : α) : α :=
  (base * height) / 2

theorem area_of_ABE (α : Type*) [LinearOrderedField α]
  (area_ABCD : α) (area_EFGH : α) (area_ACG : α)
  (side_BC lies_on_EH : Prop) (h_area_ABCD : area_ABCD = 9) (h_area_EFGH : area_EFGH = 64) (h_area_ACG : area_ACG = 6.75)
  : ∃ (area_ABE : α), area_ABE = 2.25 := 
by
  -- Definitions of side lengths based on the given areas
  let a := square_area_side area_ABCD
  let b := square_area_side area_EFGH
  -- Given the side BC lies on EH condition
  have side_BC : a = 3 := by sorry
  have side_EH : b = 8 := by sorry
  -- Using the area of triangle ACG
  have h1 : triangle_area a b = area_ACG := by sorry
  -- And the conditions and relationships ...
  let area_ABE := 2.25
  exact ⟨area_ABE, rfl⟩

end area_of_ABE_l178_178167


namespace correct_statements_l178_178425

variable (f : ℝ → ℝ)
variable (dom : Set ℝ) (h_dom : dom = Set.Ioi 0)
variable (h1 : ∀ x y ∈ dom, f (x * y) = f x + f y)
variable (h2 : ∀ x ∈ dom, x > 1 → f x < 0)
variable (h3 : f 2 = -1)

theorem correct_statements :
  f 1 = 0 ∧
  (∀ x y : ℝ, x ∈ dom ∧ y ∈ dom ∧ x < y → f y < f x) ∧
  (∀ x : ℝ, x ∈ dom ∧ x ≥ 4 → f(x⁻¹) - f(x - 3) ≥ 2) := by
  sorry

end correct_statements_l178_178425


namespace Eunji_score_equals_56_l178_178848

theorem Eunji_score_equals_56 (Minyoung_score Yuna_score : ℕ) (Eunji_score : ℕ) 
  (h1 : Minyoung_score = 55) (h2 : Yuna_score = 57)
  (h3 : Eunji_score > Minyoung_score) (h4 : Eunji_score < Yuna_score) : Eunji_score = 56 := by
  -- Given the hypothesis, it is a fact that Eunji's score is 56.
  sorry

end Eunji_score_equals_56_l178_178848


namespace minimize_sum_of_legs_l178_178404

noncomputable def area_of_right_angle_triangle (a b : ℝ) : Prop :=
  1/2 * a * b = 50

theorem minimize_sum_of_legs (a b : ℝ) (h : area_of_right_angle_triangle a b) :
  a + b = 20 ↔ a = 10 ∧ b = 10 :=
by
  sorry

end minimize_sum_of_legs_l178_178404


namespace area_of_square_l178_178574

variables (L M N P Q R S : Point)
variable (x : ℝ)

-- Conditions
def right_triangle (L M N : Point) : Prop := sorry

def square_inscribed (PQRS : quadrilateral) (LMN : triangle) : Prop := sorry

def LP (L P : Point) : ℝ := 15
def SN (S N : Point) : ℝ := 75

-- Problem: Prove that the area of the inscribed square PQRS is 1125 square units
theorem area_of_square
    (h_triangle : right_triangle L M N)
    (h_inscribed : square_inscribed PQRS LMN)
    (h_LP : dist L P = 15)
    (h_SN : dist S N = 75) :
    (x * x) = 1125 := 
by
  sorry

end area_of_square_l178_178574


namespace probability_product_power_of_6_l178_178935

theorem probability_product_power_of_6 :
  let eligible_numbers := {n ∈ finset.range 31 | n > 0 ∧ ∃ a b : ℕ, n = 2^a * 3^b}
  let combinations := eligible_numbers.powerset.filter (λ s, s.card = 5 ∧ ∀ n ∈ s, ∃ a b : ℕ, (∃ k : ℕ, n = 6^k) )
  let total_combinations := eligible_numbers.powerset.filter (λ s, s.card = 5 )
  in if total_combinations.card = 0 then false
  else combinations.card.to_rat / total_combinations.card.to_rat = 1 / 5 := sorry

end probability_product_power_of_6_l178_178935


namespace palindromic_times_24_hours_l178_178696

def is_palindromic_time (h m s : Nat) : Prop :=
  let hours := h % 24
  let mins := m % 60
  let secs := s % 60
  let hh := hours / 10 * 10 + hours % 10
  let mm := mins / 10 * 10 + mins % 10
  let ss := secs / 10 * 10 + secs % 10
  hh == ss && (hours / 10 == secs % 10) && (hours % 10 == secs / 10)

def count_palindromic_times_in_24_hours : Nat :=
  List.length [ (h, m, s) | h ← List.range 24, m ← List.range 60, s ← List.range 60, is_palindromic_time h m s ]

theorem palindromic_times_24_hours : count_palindromic_times_in_24_hours = 96 :=
  by sorry

end palindromic_times_24_hours_l178_178696


namespace books_found_in_wrong_place_l178_178494

-- Definitions given
def initial_books := 51
def books_left := 16
def books_shelved_history := 12
def books_shelved_fiction := 19
def books_shelved_children := 8

-- We need to prove that the number of books found in the wrong place is exactly 4
theorem books_found_in_wrong_place:
  let total_shelved := books_shelved_history + books_shelved_fiction + books_shelved_children in
  let should_shelved := initial_books - books_left in
  let extra_books := total_shelved - should_shelved in
  extra_books = 4 := by
  -- skip the proof
  sorry

end books_found_in_wrong_place_l178_178494


namespace distinct_positive_integers_count_l178_178445

-- Define the set of digits
def digits := {1, 2, 3, 4}

-- Define the statement as a theorem
theorem distinct_positive_integers_count :
  -- The number of distinct positive integers that can be formed from the digits without repeating any digit is 64
  (Finset.card (Finset.univ : Finset (Finset {1, 2, 3, 4}))) = 64 :=
sorry

end distinct_positive_integers_count_l178_178445


namespace parallel_vs_skew_parallel_vs_intersecting_l178_178095

-- Define the properties related to the lines
structure Line (P : Type) [EuclideanSpace P] :=
  (contains : ∀ x : P, Prop)

def coplanar {P : Type} [EuclideanSpace P] (l1 l2 : Line P) : Prop :=
  ∃ p : AffineSubspace ℝ P, p.contains l1 ∧ p.contains l2

def parallel {P : Type} [EuclideanSpace P] (l1 l2 : Line P) : Prop :=
  ∀ x y : P, l1.contains x → l2.contains y → (x -ᵥ y : P) ∈ (span ℝ {dir l1} ∩ span ℝ {dir l2})

def intersect {P : Type} [EuclideanSpace P] (l1 l2 : Line P) : Prop :=
  ∃ x : P, l1.contains x ∧ l2.contains x

def distance_preserving {P : Type} [EuclideanSpace P] (l1 l2 : Line P) : Prop :=
  ∀ x y : P, l1.contains x → l2.contains y → dist x y = const

def form_angle {P : Type} [EuclideanSpace P] (l1 l2 : Line P) : Prop :=
  ∃ θ : Real, 0 ≤ θ ∧ θ ≤ 180 ∧ ∀ x : P, l1.contains x → l2.contains x → angle x (dir l1) (dir l2) = θ

-- Problem 1: Parallel Lines vs. Skew Lines
theorem parallel_vs_skew (P : Type) [EuclideanSpace P] (l1 l2 : Line P) :
  non_intersect l1 l2 ∧ (¬ coplanar l1 l2 ∧ parallel l1 l2) ∧ (¬ parallel l1 l2 ∧ ¬ coplanar l1 l2 ∧ ¬ distance_preserving l1 l2) :=
sorry

-- Problem 2: Parallel Lines vs. Intersecting Lines
theorem parallel_vs_intersecting (P : Type) [EuclideanSpace P] (l1 l2 : Line P) :
  (parallel l1 l2 ∧ form_angle l1 l2) ∧ (non_intersect l1 l2 ∧ ¬ distance_preserving l1 l2) :=
sorry

end parallel_vs_skew_parallel_vs_intersecting_l178_178095


namespace magnitude_of_sum_of_vectors_l178_178523

open Real

-- Conditions
variables (x : ℝ)
def a := (x, 1 : ℝ)
def b := (4, -2 : ℝ)
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- Proof statement
theorem magnitude_of_sum_of_vectors (h : parallel (a x) b) :
  (sqrt ((a x).1^2 + (a x).2^2)) = sqrt 5 :=
sorry

end magnitude_of_sum_of_vectors_l178_178523


namespace other_root_is_minus_two_l178_178020

theorem other_root_is_minus_two (b : ℝ) (h : 1^2 + b * 1 - 2 = 0) : 
  ∃ (x : ℝ), x = -2 ∧ x^2 + b * x - 2 = 0 :=
by
  sorry

end other_root_is_minus_two_l178_178020


namespace missing_digit_l178_178578

def set_of_numbers : List ℕ := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

theorem missing_digit (mean : ℕ) : (mean = 109739368) → ¬(4 ∈ mean.digits 10) :=
by
  intro h
  rw h
  unfold digits
  -- Here, Lean's built-in arithmetic and list functionality can be used
  sorry

end missing_digit_l178_178578


namespace clara_cookies_l178_178709

theorem clara_cookies (n : ℕ) :
  (15 * n - 1) % 11 = 0 → n = 3 := 
sorry

end clara_cookies_l178_178709


namespace total_cost_l178_178099

theorem total_cost (skirt_width skirt_length : ℝ) (num_skirts : ℕ)
                   (shirt_area sleeve_area : ℝ) (num_sleeves : ℕ)
                   (bonnet_width bonnet_length : ℝ)
                   (shoe_cover_width shoe_cover_length : ℝ) (num_shoe_covers : ℕ)
                   (cost_skirt cost_bodice cost_bonnet cost_shoe_cover : ℝ) :
  skirt_width = 12 → skirt_length = 4 → num_skirts = 3 →
  shirt_area = 2 → sleeve_area = 5 → num_sleeves = 2 →
  bonnet_width = 2.5 → bonnet_length = 1.5 →
  shoe_cover_width = 1 → shoe_cover_length = 1.5 → num_shoe_covers = 2 →
  cost_skirt = 3 → cost_bodice = 2.5 → cost_bonnet = 1.5 → cost_shoe_cover = 4 →
  let total_area_skirts := num_skirts * (skirt_width * skirt_length)
  let total_cost_skirts := total_area_skirts * cost_skirt
  let total_area_bodice := shirt_area + num_sleeves * sleeve_area
  let total_cost_bodice := total_area_bodice * cost_bodice
  let total_area_bonnet := bonnet_width * bonnet_length
  let total_cost_bonnet := total_area_bonnet * cost_bonnet
  let total_area_shoe_covers := num_shoe_covers * (shoe_cover_width * shoe_cover_length)
  let total_cost_shoe_covers := total_area_shoe_covers * cost_shoe_cover
  let total_cost := total_cost_skirts + total_cost_bodice + total_cost_bonnet + total_cost_shoe_covers
  total_cost = 479.63 :=
by intros ; 
   simp only [total_area_skirts, total_cost_skirts, skirt_width,
              skirt_length, num_skirts, cost_skirt, total_cost]; 
   simp only [total_area_bodice, total_cost_bodice, shirt_area,
              sleeve_area, num_sleeves, cost_bodice, total_cost];
   simp only [total_area_bonnet, total_cost_bonnet, bonnet_width, 
              bonnet_length, cost_bonnet, total_cost];
   simp only [total_area_shoe_covers, total_cost_shoe_covers, shoe_cover_width, 
              shoe_cover_length, num_shoe_covers, cost_shoe_cover, total_cost];
   sorry

end total_cost_l178_178099


namespace find_ellipse_eq_find_line_eq_l178_178008

section
  -- Define the given conditions
  variables {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) (h1 : a = 2 * c) (h2 : b^2 = 3 * c^2)
  variable {P : ℝ × ℝ} (hP : P = (1, 3/2)) -- Point (1, 3/2) is on the ellipse
  variable {Q : ℝ × ℝ} (hQ : Q = (4, 0)) -- Line m passes through point (4, 0)
  
  -- Ellipse equation
  def ellipse_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

  -- Line equation and the intersection condition
  def line_eq (k x y : ℝ) : Prop := y = k * (x - 4)
  def intersects_ellipse (k : ℝ) : Prop :=
    let delta := (32 * k^2)^2 - 4 * (3 + 4 * k^2) * (64 * k^2 - 12) in
    - (1 / 2) < k ∧ k < 1 / 2 ∧ delta > 0

  -- Proving the answer for part (1)
  theorem find_ellipse_eq :
    ellipse_eq (1 : ℝ) (3/2 : ℝ) :=
  by
    -- Substitute the given conditions and solve
    sorry

  -- Proving the answer for part (2)
  theorem find_line_eq (k : ℝ) (h_intersect : intersects_ellipse k) :
    (sqrt 2 * 4 + 4 * k * (4 - 4)) = 4 * sqrt 2 ∨
    (sqrt 2 * 4 - 4 * k * (4 - 4)) = 4 * sqrt 2  :=
  by
    -- Substitute the given conditions and solve
    sorry
end

end find_ellipse_eq_find_line_eq_l178_178008


namespace original_selling_price_l178_178291

theorem original_selling_price:
  ∀ (P : ℝ), (1.17 * P - 1.10 * P = 56) → (P > 0) → 1.10 * P = 880 :=
by
  intro P h₁ h₂
  sorry

end original_selling_price_l178_178291


namespace matrix_computation_l178_178122

variable (N : Matrix (Fin 2) (Fin 2) ℤ)
variable (v1 v2 v3 v4 : Fin 2 → ℤ)

-- Define the vectors
def vec1 := ![3, -2]
def vec2 := ![-2, 4]
def vec3 := ![4, 1]
def vec4 := ![0, -2]
def vec5 := ![7, 0]
def result := ![14, 0]

-- Hypotheses based on the problem statement
axiom h1 : N.mul_vec vec1 = vec3
axiom h2 : N.mul_vec vec2 = vec4

-- Statement we need to prove
theorem matrix_computation : N.mul_vec vec5 = result :=
  sorry

end matrix_computation_l178_178122


namespace gasoline_price_increase_l178_178686

theorem gasoline_price_increase (high low : ℝ) (high_eq : high = 24) (low_eq : low = 18) : 
  ((high - low) / low) * 100 = 33.33 := 
  sorry

end gasoline_price_increase_l178_178686


namespace S_eq_T_l178_178802

open Nat

noncomputable def S (n : ℕ) : ℝ :=
  if n = 0 then 0 else ∑ i in range n, ((-1)^i) * (1 / (1 + i))

noncomputable def T (n : ℕ) : ℝ :=
  if n = 0 then 0 else ∑ i in Ico (succ (n-1)) (succ (2*n)), 1 / (i + 1)

theorem S_eq_T (n : ℕ) (h : n > 0) : S n = T n :=
  sorry

end S_eq_T_l178_178802


namespace goldfish_sold_l178_178215

/-- The pet store buys a goldfish for $0.25 and sells it for $0.75.
    The new tank costs $100 and the owner is 45% short of the price. 
    Prove that the owner sold 110 goldfish that week. -/
theorem goldfish_sold (buy_price sell_price tank_cost shortfall_percentage : ℝ)
  (h_buy : buy_price = 0.25)
  (h_sell : sell_price = 0.75)
  (h_tank : tank_cost = 100)
  (h_shortfall : shortfall_percentage = 0.45) :
  ∃ (n : ℕ), n = 110 :=
begin
  sorry
end

end goldfish_sold_l178_178215


namespace root_in_interval_l178_178986

def f (x : ℝ) : ℝ := log x + 3 * x - 7

theorem root_in_interval : ∃ a b : ℝ, f a < 0 ∧ f b > 0 ∧ (2 < a ∧ a < 3) ∧ (2 < b ∧ b < 3) :=
by {
    sorry 
}

end root_in_interval_l178_178986


namespace Kolya_is_CollectionAgency_l178_178877

-- Define the roles
inductive Role
| FinancialPyramid
| CollectionAgency
| Bank
| InsuranceCompany

-- Define the conditions parametrically
structure Scenario where
  lent_books : Bool
  promise_broken : Bool
  mediator_requested : Bool
  reward_requested : Bool

-- Define the theorem statement
theorem Kolya_is_CollectionAgency
  (scenario : Scenario)
  (h1 : scenario.lent_books = true)
  (h2 : scenario.promise_broken = true)
  (h3 : scenario.mediator_requested = true)
  (h4 : scenario.reward_requested = true) :
  Kolya_is_CollectionAgency :=
  begin
    -- Proof not required
    sorry
  end

end Kolya_is_CollectionAgency_l178_178877


namespace geometric_sequence_sum_l178_178080

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = r * a n)
  (h2 : 0 < r)
  (h3 : a 1 = 3)
  (h4 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l178_178080


namespace apple_running_rate_eq_l178_178699

-- Define Apple's running rate as a variable
variable (A : ℝ)

-- Define the conditions given in the problem
def mac_rate := 4 -- Mac's running rate in miles per hour
def race_distance := 24 -- The race distance in miles
def time_difference := 120 -- Time difference in minutes

-- Convert minutes to hours in the conditions
def time_diff_hours := (time_difference : ℝ) / 60

-- Mac's time converted to hours
def mac_time_hours := (race_distance : ℝ) / mac_rate

-- Apple's time in hours, knowing it is 120 minutes (2 hours) more than Mac's time
def apple_time_hours := mac_time_hours + time_diff_hours / 60

-- Apple’s running rate should satisfy this equation
theorem apple_running_rate_eq : A = race_distance / apple_time_hours := by
  sorry

end apple_running_rate_eq_l178_178699


namespace identity_of_polynomials_l178_178944

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l178_178944


namespace no_solution_iff_m_range_l178_178354

theorem no_solution_iff_m_range (m : ℝ) : 
  ¬ ∃ x : ℝ, |x-1| + |x-m| < 2*m ↔ (0 < m ∧ m < 1/3) := sorry

end no_solution_iff_m_range_l178_178354


namespace greatest_int_less_than_200_gcd_30_is_5_l178_178247

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l178_178247


namespace total_hours_watched_l178_178650

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end total_hours_watched_l178_178650


namespace condition_sufficient_but_not_necessary_l178_178178

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (|x| + |y| ≤ 1 → x^2 + y^2 ≤ 1) ∧ (x^2 + y^2 ≤ 1 → ¬ (|x| + |y| ≤ 1)) :=
sorry

end condition_sufficient_but_not_necessary_l178_178178


namespace minimum_value_of_exponential_sum_l178_178427

theorem minimum_value_of_exponential_sum :
  let eq := λ x y, y = - (1 / 2) * x + (3 / 2) in
  ∀ x y, eq x y → 2^x + 4^y ≥ 4 * Real.sqrt 2 :=
by sorry

end minimum_value_of_exponential_sum_l178_178427


namespace integer_point_in_convex_region_l178_178526

noncomputable def convex_region (K : set (ℝ × ℝ)) : Prop :=
convex ℝ K

noncomputable def symmetric_about_origin (K : set (ℝ × ℝ)) : Prop :=
∀ {x y : ℝ}, (x, y) ∈ K → (-x, -y) ∈ K

noncomputable def area_greater_than (K : set (ℝ × ℝ)) (a : ℝ) : Prop :=
∃ f : (ℝ × ℝ) → set (ℝ × ℝ), (∫ (x : ℝ × ℝ) in K, 1) > a

theorem integer_point_in_convex_region 
  (K : set (ℝ × ℝ))
  (h_convex : convex_region K)
  (h_symmetric : symmetric_about_origin K)
  (h_area : area_greater_than K 4) :
  ∃ (m n : ℤ), (m, n) ∈ K ∧ (m, n) ≠ (0, 0) := sorry

end integer_point_in_convex_region_l178_178526


namespace approximate_barometer_reading_l178_178978

theorem approximate_barometer_reading :
  ∀ x, (30.10 < x ∧ x < 30.30) → (x ≈ 30.20) :=
by
  -- proof omitted
  sorry

end approximate_barometer_reading_l178_178978


namespace range_m_condition_l178_178932

theorem range_m_condition {x y m : ℝ} (h1 : x^2 + (y - 1)^2 = 1) (h2 : x + y + m ≥ 0) : -1 < m :=
by
  sorry

end range_m_condition_l178_178932


namespace monotonicity_F_range_of_a_l178_178036

open Real

def f (a x : ℝ) : ℝ := ln x - a * x
def g (a x : ℝ) : ℝ := 1 / x + a
def F (a x : ℝ) : ℝ := ln x - a * x - (1 / x + a)

-- Monotonicity of F(x)
theorem monotonicity_F (a : ℝ) : 
  (∀ x > 0, if a ≤ 0 then (F a)' x > 0 else 
  ∀ x ∈ (0, (1 + sqrt (1 + 4 * a)) / (2 * a)), (F a)' x > 0 ∧ ∀ x ∈ ((1 + sqrt (1 + 4 * a)) / (2 * a), +∞), (F a)' x < 0) := sorry

-- Range of a if f(x) * g(x) ≤ 0
theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f a x * g a x ≤ 0 ↔ a ≥ 1/e ∨ a = - e) := sorry

end monotonicity_F_range_of_a_l178_178036


namespace sum_sin_squares_degrees_l178_178711

theorem sum_sin_squares_degrees : 
  (Finset.sum (Finset.range 29) (λ k, sin (Real.pi * (3 * (k + 1)) / 180) ^ 2)) = 15.5 := 
by
  sorry

end sum_sin_squares_degrees_l178_178711


namespace max_volume_at_x_5_l178_178717

noncomputable def volume (x : ℝ) : ℝ := (30 - 2 * x)^2 * x

theorem max_volume_at_x_5 : ∀ x : ℝ, x ∈ {4, 5, 6, 7} → volume 5 ≥ volume x := sorry

end max_volume_at_x_5_l178_178717


namespace new_eqn_a_new_eqn_b_new_eqn_c_new_eqn_d_l178_178980

variables {p q x : ℝ}
open Classical

-- For all questions: Assume given equations and Vieta's conditions
def vieta_conditions : Prop := ∃ x1 x2, x^2 + p*x + q = 0 ∧ x1 + x2 = -p ∧ x1*x2 = q

-- Part a
theorem new_eqn_a (x : ℝ) (p q : ℝ) (hc : vieta_conditions) : 
  ∃ x, x^2 + (p^3 - 3*p*q)*x + q^3 = 0 :=
sorry

-- Part b
theorem new_eqn_b (x : ℝ) (p q : ℝ) (hc : vieta_conditions) : 
  ∃ x, q^2*x^2 + (2*q - p^2)*x + 1 = 0 :=
sorry

-- Part c
theorem new_eqn_c (x : ℝ) (p q : ℝ) (hc : vieta_conditions) : 
  ∃ x, q*x^2 + p*(q + 1)*x + (q + 1)^2 = 0 :=
sorry

-- Part d
theorem new_eqn_d (x : ℝ) (p q : ℝ) (hc : vieta_conditions) : 
  ∃ x, q*x^2 + (2*q - p^2)*x + q = 0 :=
sorry

end new_eqn_a_new_eqn_b_new_eqn_c_new_eqn_d_l178_178980


namespace gcd_subtract_ten_l178_178619

theorem gcd_subtract_ten (a b : ℕ) (h₁ : a = 720) (h₂ : b = 90) : (Nat.gcd a b) - 10 = 80 := by
  sorry

end gcd_subtract_ten_l178_178619


namespace eugene_pants_l178_178477

theorem eugene_pants (number_of_pants : ℕ)
  (cost_tshirt cost_pants cost_shoes : ℕ)
  (qty_tshirt qty_pants qty_shoes : ℕ)
  (total_paid : ℕ)
  (discount : ℝ) :
  (cost_tshirt = 20 ∧ cost_pants = 80 ∧ cost_shoes = 150) ∧
  (qty_tshirt = 4 ∧ qty_shoes = 2) ∧
  discount = 0.10 ∧
  total_paid = 558 →
  0.90 * (qty_tshirt * cost_tshirt + qty_pants * cost_pants + qty_shoes * cost_shoes) = total_paid →
  qty_pants = 3 :=
by {
  intros h_cond h_eq,
  sorry
}

end eugene_pants_l178_178477


namespace greatest_integer_gcf_l178_178265

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l178_178265


namespace simplify_expression_l178_178625

theorem simplify_expression (x : ℝ) (hx_pos : x > 0) (hx_not_one : x ≠ 1) :
  (1 - x⁻²) / (x^(1/2) - x^(-1/2)) - 2 / x^(3/2) + (x⁻² - x) / (x^(1/2) - x^(-1/2)) = 
  -sqrt(x) * (1 + 2/x²) :=
sorry

end simplify_expression_l178_178625


namespace unique_consecutive_sum_to_20_l178_178057

theorem unique_consecutive_sum_to_20 :
  ∃! (a n : ℕ), n ≥ 2 ∧ (∑ i in range (n + 1), (a + i)) = 20 :=
sorry

end unique_consecutive_sum_to_20_l178_178057


namespace cost_price_per_meter_of_cloth_l178_178629

theorem cost_price_per_meter_of_cloth
  (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) (total_profit : ℕ) (cost_price : ℕ)
  (meters_eq : meters = 80)
  (selling_price_eq : selling_price = 10000)
  (profit_per_meter_eq : profit_per_meter = 7)
  (total_profit_eq : total_profit = profit_per_meter * meters)
  (selling_price_calc : selling_price = cost_price + total_profit)
  (cost_price_calc : cost_price = selling_price - total_profit)
  : (selling_price - total_profit) / meters = 118 :=
by
  -- here we would provide the proof, but we skip it with sorry
  sorry

end cost_price_per_meter_of_cloth_l178_178629


namespace eggs_used_in_afternoon_l178_178539

theorem eggs_used_in_afternoon (total_eggs used_in_morning : ℕ) (h1 : total_eggs = 1339) (h2 : used_in_morning = 816) : total_eggs - used_in_morning = 523 :=
by
  rw [h1, h2]
  sorry

end eggs_used_in_afternoon_l178_178539


namespace pet_store_cats_left_l178_178679

theorem pet_store_cats_left :
  let initial_siamese := 13.5
  let initial_house := 5.25
  let added_cats := 10.75
  let discount := 0.5
  let initial_total := initial_siamese + initial_house
  let new_total := initial_total + added_cats
  let final_total := new_total - discount
  final_total = 29 :=
by sorry

end pet_store_cats_left_l178_178679


namespace equilateral_triangle_on_ellipse_l178_178999

noncomputable def square_of_area_of_triangle : ℝ :=
  let triangle_area_squared := (27 / 4)
  in triangle_area_squared

theorem equilateral_triangle_on_ellipse
  (x y : ℝ)
  (condition1 : (0, 1) is_centroid_of_equilateral_triangle_on_ellipse x^2 + 3*y^2 = 3) :
  square_of_area_of_triangle = 27 / 4 :=
by
  sorry

end equilateral_triangle_on_ellipse_l178_178999


namespace fraction_spent_on_food_l178_178319

/-- The man spends 1/4 of his salary on rent. -/
def spend_rent (salary : ℝ) : ℝ := 1 / 4 * salary

/-- The man spends 1/5 of his salary on clothes. -/
def spend_clothes (salary : ℝ) : ℝ := 1 / 5 * salary

/-- The amount of salary left after all expenses is $1760. -/
def amount_left : ℝ := 1760

/-- The total salary of the man is $8123.08. -/
def total_salary : ℝ := 8123.08

/-- The fraction of the salary spent on food is approximately 1/3. -/
theorem fraction_spent_on_food : 
  (total_salary - (spend_rent total_salary + spend_clothes total_salary + amount_left)) / total_salary = 1 / 3 :=
sorry

end fraction_spent_on_food_l178_178319


namespace reflex_angle_at_T_l178_178392

def four_points_collinear (P Q R S : Point) : Prop := collinear ({P, Q, R, S} : set Point)
def angle_PQT := 100
def angle_TRS := 90
def reflex_angle_T (P Q R S T : Point) : ℕ := if ∠PQT = 100 ∧ ∠TRS = 90 then 360 - 10 else 0

theorem reflex_angle_at_T (P Q R S T : Point) 
  (h_collinear : four_points_collinear P Q R S)
  (h_angle_PQT : angle_PQT = 100)
  (h_angle_TRS : angle_TRS = 90) :
  reflex_angle_T P Q R S T = 350 :=
sorry

end reflex_angle_at_T_l178_178392


namespace find_values_conjecture_a_n_S_n_bound_l178_178796

open Nat

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), (n > 0) → (a n.succ + a n - 1) / (a n.succ - a n + 1) = n

def initial (a : ℕ → ℕ) : Prop :=
  a 2 = 6

noncomputable def a_1 (a : ℕ → ℕ) : ℕ := 1
noncomputable def a_3 (a : ℕ → ℕ) : ℕ := 15
noncomputable def a_4 (a : ℕ → ℕ) : ℕ := 28

theorem find_values (a : ℕ → ℕ) : seq a ∧ initial a → a 1 = 1 ∧ a 3 = 15 ∧ a 4 = 28 :=
  sorry

theorem conjecture_a_n (a : ℕ → ℕ) : seq a ∧ initial a → ∀ (n : ℕ), (n > 0) → a n = n * (2 * n - 1) :=
  sorry

noncomputable def S_n (a : ℕ → ℕ) (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ i, 1 / (a i : ℝ))

theorem S_n_bound (a : ℕ → ℕ) : seq a ∧ initial a → ∀ (n : ℕ), (n > 0) → S_n a n < 3/2 :=
  sorry

end find_values_conjecture_a_n_S_n_bound_l178_178796


namespace satisify_absolute_value_inequality_l178_178052

theorem satisify_absolute_value_inequality :
  ∃ (t : Finset ℤ), t.card = 2 ∧ ∀ y ∈ t, |7 * y + 4| ≤ 10 :=
by
  sorry

end satisify_absolute_value_inequality_l178_178052


namespace min_max_f_l178_178191

noncomputable def f (x : ℝ) : ℝ := Math.cos x + (x + 1) * Math.sin x + 1

theorem min_max_f :
  ∃ a b : ℝ, a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a) ∧ 
  (∃ y ∈ Set.Icc 0 (2 * Real.pi), f y = a) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ b) ∧ 
  (∃ z ∈ Set.Icc 0 (2 * Real.pi), f z = b) :=
sorry

end min_max_f_l178_178191


namespace greatest_integer_gcf_l178_178264

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l178_178264


namespace unique_solution_l178_178370

theorem unique_solution (n : ℕ) (h : n > 1) : Int ∃ k : ℤ, (2^n + 1) = k * (n^2) → n = 3 :=
by
  -- Define the properties of n as conditions here
  have int_n_square : ∃ k : ℤ, (2^n + 1) = k * (n^2) := sorry,
  -- Prove the unique value for n
  sorry

end unique_solution_l178_178370


namespace kathryn_financial_statement_l178_178503

def kathryn_remaining_money (rent : ℕ) (salary : ℕ) (share_rent : ℕ → ℕ) (total_expenses : ℕ → ℕ) (remaining_money : ℕ → ℕ) : Prop :=
  rent = 1200 ∧
  salary = 5000 ∧
  share_rent rent = rent / 2 ∧
  ∀ rent_total, total_expenses (share_rent rent_total) = (share_rent rent_total) + 2 * rent_total ∧
  remaining_money salary total_expenses = salary - total_expenses (share_rent rent)

theorem kathryn_financial_statement : kathryn_remaining_money 1200 5000 (λ rent, rent / 2) (λ rent, rent / 2 + 2 * rent) (λ salary expenses, salary - expenses (λ rent, rent / 2)) :=
by {
  sorry
}

end kathryn_financial_statement_l178_178503


namespace greatest_integer_with_gcf_5_l178_178250

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l178_178250


namespace average_weight_of_girls_l178_178631

theorem average_weight_of_girls :
  ∀ (total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight : ℝ),
  total_students = 25 →
  boys = 15 →
  girls = 10 →
  boys + girls = total_students →
  class_average_weight = 45 →
  boys_average_weight = 48 →
  total_weight = 1125 →
  girls_average_weight = (total_weight - (boys * boys_average_weight)) / girls →
  total_weight = class_average_weight * total_students →
  girls_average_weight = 40.5 :=
by
  intros total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight
  sorry

end average_weight_of_girls_l178_178631


namespace find_m_value_l178_178440

open Real

def vector_perpendicular {m : ℝ} : Prop :=
  let a := (1, 1)
  let b := (3, m)
  let diff := (1 - 3, 1 - m)
  a.1 * diff.1 + a.2 * diff.2 = 0 → m = -1

theorem find_m_value (m : ℝ) : vector_perpendicular m :=
by {
  let a := (1, 1) : ℝ × ℝ,
  let b := (3, m) : ℝ × ℝ,
  let diff := (1 - 3, 1 - m),
  have h1 : a.1 * diff.1 + a.2 * diff.2 = -2 + 1 - m,
  sorry,
}

end find_m_value_l178_178440


namespace cubic_polynomial_solution_l178_178522

theorem cubic_polynomial_solution :
  ∃ (q : ℝ[X]), q.monic ∧ q.degree = 3 ∧ q.eval (2 - 3 * complex.I) = 0 ∧ q.eval 0 = -36 ∧ 
  q = polynomial.C (-(468 / 13)) + polynomial.X * (polynomial.C (325 / 13)) + 
  polynomial.X * (polynomial.X * polynomial.C (-(88 / 13))) + polynomial.X ^ 3 :=
begin
  sorry
end

end cubic_polynomial_solution_l178_178522


namespace necessarily_positive_expression_l178_178157

theorem necessarily_positive_expression
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  0 < b + 3 * b^2 := 
sorry

end necessarily_positive_expression_l178_178157


namespace yellow_parrots_l178_178234

theorem yellow_parrots (total_parrots : ℕ) (fraction_red : ℚ) (fraction_yellow : ℚ) (red_parrots : ℕ) (yellow_parrots : ℕ) :
  total_parrots = 120 → fraction_red = 2 / 3 → fraction_yellow = 1 / 3 → red_parrots = total_parrots * fraction_red → yellow_parrots = total_parrots * fraction_yellow → yellow_parrots = 40 :=
by
  intros h_total h_fraction_red h_fraction_yellow h_red_parrots h_yellow_parrots
  rw [h_total, h_fraction_yellow, h_yellow_parrots]
  norm_num
  
#reduce yellow_parrots 120 (2/3) (1/3) (120 * (2/3)) (120 * (1/3)) rfl rfl rfl rfl rfl

end yellow_parrots_l178_178234


namespace day_50_of_year_N_minus_1_l178_178094

open Nat

namespace DayOfWeek

-- Definitions for weekdays for simplicity
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr, Inhabited

/-- The function to calculate offsets with respect to the day of the week -/
def add_days (d: Weekday) (n: ℕ) : Weekday :=
  (match d with
    | Weekday.Sunday   => 0
    | Weekday.Monday   => 1
    | Weekday.Tuesday  => 2
    | Weekday.Wednesday => 3
    | Weekday.Thursday => 4
    | Weekday.Friday   => 5
    | Weekday.Saturday => 6) + n % 7 match
    | 0 => Weekday.Sunday
    | 1 => Weekday.Monday
    | 2 => Weekday.Tuesday
    | 3 => Weekday.Wednesday
    | 4 => Weekday.Thursday
    | 5 => Weekday.Friday
    | _ => Weekday.Saturday

/-- The main theorem stating the problem -/
theorem day_50_of_year_N_minus_1 (N: ℕ):
  let day_250 := Weekday.Monday in
  let day_150_N_plus_1 := Weekday.Monday in
  (∀ d1 d2 : Weekday, d1 = add_days d2 365) →
  (let day_50 := add_days day_250 (365 + 365 - 200 - 50) in day_50 = Weekday.Thursday) :=
by intros;
   sorry

end DayOfWeek

end day_50_of_year_N_minus_1_l178_178094


namespace find_k1_k2_product_l178_178026

-- Define the conditions of the problem
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def parabola (k : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * k * x

def point_P : ℝ × ℝ := (-2, 0)

-- Define the problem statement
theorem find_k1_k2_product (a b c : ℝ) (e k : ℝ) (k1 k2 : ℝ) :
  a > b ∧ b > 0 ∧ c = sqrt 3 ∧ e = sqrt 3 / 2 ∧ a = 2 ∧ b = 1 ∧
  ellipse a b (-2) 0 ∧ parabola (sqrt 3) (focus.x) (focus.y) ∧ 
  (λ mn, mn y ≠ mn x) ∧ k1 ≠ k2 ∧ 
  (4 * k1 * k2 = 1) :=
  k1 * k2 = 1 / 4 :=
begin
  sorry -- Proof to be filled in
end

end find_k1_k2_product_l178_178026


namespace sum_of_first_100_terms_l178_178763

theorem sum_of_first_100_terms :
  let a : ℕ → ℕ := λ n, n in
  let seq := λ n, (1 : ℝ)/(a n * a (n + 1)) in
  (finset.range 100).sum (λ n, seq (n + 1)) = (100 : ℝ) / (101 : ℝ) :=
by
  sorry

end sum_of_first_100_terms_l178_178763


namespace log_eq_solution_l178_178730

theorem log_eq_solution (x : ℝ) (h : Real.log (x + 1) = (1 / 2) * Real.log 3 x) : x = 9 :=
sorry

end log_eq_solution_l178_178730


namespace arrange_abc_l178_178756

variable (a b c : ℝ)

theorem arrange_abc (h₁ : a = Real.sqrt 4) (h₂ : b = 0.5 ^ 4) (h₃ : c = Real.logBase 0.5 4) : c < b ∧ b < a := 
by 
  sorry

end arrange_abc_l178_178756


namespace range_of_k_l178_178521

theorem range_of_k :
  ∀ (a k : ℝ) (f : ℝ → ℝ),
    (∀ x, f x = if x ≥ 0 then k^2 * x + a^2 - k else x^2 + (a^2 + 4 * a) * x + (2 - a)^2) →
    (∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → x1 ≠ x2 → f x1 = f x2 → False) →
    -20 ≤ k ∧ k ≤ -4 :=
by
  sorry

end range_of_k_l178_178521


namespace triangle_PQR_PR_length_proof_l178_178859

noncomputable def triangle_PQR_PR_length (PQ QR PS : ℝ) (S_on_PR : Prop) (angle_bisector_PQ : Prop) : ℝ :=
2 * Real.sqrt 97

-- Given conditions as hypotheses
axiom PQ_value : PQ = 8
axiom QR_value : QR = 18
axiom PS_value : PS = 14
axiom S_on_PR_condition : S_on_PR
axiom angle_bisector_PQ_condition : angle_bisector_PQ

-- The proof problem
theorem triangle_PQR_PR_length_proof : triangle_PQR_PR_length 8 18 14 S_on_PR_condition angle_bisector_PQ_condition = 2 * Real.sqrt 97 := 
sorry

end triangle_PQR_PR_length_proof_l178_178859


namespace F_properties_l178_178893

variable (f : ℝ → ℝ)

-- Given that f is an increasing function on ℝ
def increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

theorem F_properties (hf : increasing f) :
  let F := fun x => f(x) - f(-x) in
  (∀ x, F(x) = -F(-x)) ∧ (increasing F) :=
by
  sorry

end F_properties_l178_178893


namespace bottle_T_cost_l178_178349

-- Define the given conditions
def cost_per_capsule_R : ℝ := 6.25 / 250
def num_capsules_T : ℕ := 100
def difference : ℝ := 0.005

-- Define what we want to prove
theorem bottle_T_cost :
  let bottle_T_cost_per_capsule := cost_per_capsule_R - difference in
  let bottle_T_cost := bottle_T_cost_per_capsule * (num_capsules_T:ℝ) in
  bottle_T_cost = 2.00 :=
by
  sorry

end bottle_T_cost_l178_178349


namespace initial_burgers_l178_178376

theorem initial_burgers (friends : ℕ) (slices_per_burger : ℕ) 
    (slices_f1 slices_f2 slices_f3 slices_f4 slices_era : ℕ) : 
    friends = 4 → 
    slices_per_burger = 2 → 
    slices_f1 = 1 → 
    slices_f2 = 2 → 
    slices_f3 = 3 → 
    slices_f4 = 3 → 
    slices_era = 1 → 
    (slices_f1 + slices_f2 + slices_f3 + slices_f4 + slices_era) / slices_per_burger = 5 :=
by {
    intros,
    sorry
}

end initial_burgers_l178_178376


namespace lcm_of_2_3_5_l178_178277

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def lcm (a b : ℕ) : ℕ :=
  if a = 0 ∨ b = 0 then 0
  else a * (b / Nat.gcd a b)

def lcm_of_set (s : Set ℕ) : ℕ :=
  if h : s.Nonempty then
    Set.toFinset s h |>.fold lcm 1
  else
    1

theorem lcm_of_2_3_5 : lcm_of_set {2, 3, 5} = 30 :=
  sorry

end lcm_of_2_3_5_l178_178277


namespace identity_of_polynomials_l178_178948

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l178_178948


namespace ellipse_property_l178_178340

noncomputable def a : ℝ := (Real.sqrt 20 + Real.sqrt 32) / 2
def k : ℝ := 0
def foci : List (ℝ × ℝ) := [(1, 0), (-1, 0)]
def point_on_ellipse : ℝ × ℝ := (3, 4)

theorem ellipse_property :
  let h := 0
  ∧ let b := 1
  ∧ (∀ x y, ((x - h) ^ 2 / a ^ 2) + ((y - k) ^ 2 / b ^ 2) = 1)
  ∧ (foci = [(1, 0), (-1, 0)])
  ∧ (point_on_ellipse = (3, 4)) in
  a + |k| = (Real.sqrt 20 + Real.sqrt 32) / 2 :=
by
  sorry

end ellipse_property_l178_178340


namespace identity_verification_l178_178950

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l178_178950


namespace units_digit_of_result_is_7_l178_178985

theorem units_digit_of_result_is_7 (a b c : ℕ) (h : a = c + 3) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  (original - reversed) % 10 = 7 :=
by
  sorry

end units_digit_of_result_is_7_l178_178985


namespace g_at_2002_l178_178418

noncomputable def f : ℝ → ℝ := sorry

def g (x : ℝ) : ℝ := f(x) + 1 - x

theorem g_at_2002 (h1 : f(1) = 1) 
                  (hf5 : ∀ x : ℝ, f(x + 5) ≥ f(x) + 5) 
                  (hf1 : ∀ x : ℝ, f(x + 1) ≤ f(x) + 1) 
                  : g 2002 = 1 := 
sorry

end g_at_2002_l178_178418


namespace intersection_point_unique_l178_178025

def f : ℝ → ℝ := sorry
def domain : set ℝ := set.Icc (-1) 5
def line_x_eq_one := { (x, y) : ℝ × ℝ | x = 1 }

theorem intersection_point_unique :
  (1 : ℝ) ∈ domain →
  set.subsingleton { (x, y) | (x, y) ∈ function.graph f ∧ (x, y) ∈ line_x_eq_one } :=
by
  intros h
  have h_domain : domain = set.Icc (-1) 5 := rfl
  have h_f_domain : ∀ x, x ∈ domain → ∃! y, (x, y) ∈ function.graph f := sorry
  sorry

end intersection_point_unique_l178_178025


namespace AM_GM_inequality_l178_178299

-- Let's first state the conditions.
variable {n : ℕ} (a : Fin n → ℝ)

-- Assume all a_i are positive.
def positive_numbers (a : Fin n → ℝ) :=
  ∀ i, 0 < a i

-- The main theorem statement.
theorem AM_GM_inequality (h : positive_numbers a) :
  (∑ i : Fin n, a i / a (i + 1) % n) ≥ n :=
sorry

end AM_GM_inequality_l178_178299


namespace num_divisors_f_2010_l178_178390

noncomputable def f (n : ℕ) : ℕ :=
3^n

theorem num_divisors_f_2010 : (nat.divisors (f 2010)).length = 2011 :=
by
  sorry

end num_divisors_f_2010_l178_178390


namespace min_x_prime_factorization_sum_eq_31_l178_178114

theorem min_x_prime_factorization_sum_eq_31
    (x y a b c d : ℕ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 7 * x^5 = 11 * y^13)
    (hx_prime_fact : ∃ a c b d : ℕ, x = a^c * b^d) :
    a + b + c + d = 31 :=
by
 sorry
 
end min_x_prime_factorization_sum_eq_31_l178_178114


namespace number_of_valid_x_l178_178209

def star (a b : ℕ) : ℕ := a^3 / b

theorem number_of_valid_x : 
  (∃ n : ℕ, (∀ x : ℕ, star 27 x = n → n > 0) → finset.card (finset.filter (λ x, 19683 % x = 0) (finset.range (19683 + 1))) = 10) :=
sorry

end number_of_valid_x_l178_178209


namespace min_max_f_l178_178189

noncomputable def f (x : ℝ) : ℝ := Math.cos x + (x + 1) * Math.sin x + 1

theorem min_max_f :
  ∃ a b : ℝ, a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a) ∧ 
  (∃ y ∈ Set.Icc 0 (2 * Real.pi), f y = a) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ b) ∧ 
  (∃ z ∈ Set.Icc 0 (2 * Real.pi), f z = b) :=
sorry

end min_max_f_l178_178189


namespace trigonometric_identity_l178_178395

variable (θ : ℝ) (h : Real.tan θ = 2)

theorem trigonometric_identity : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := 
sorry

end trigonometric_identity_l178_178395


namespace arithmetic_mean_of_primes_l178_178702

theorem arithmetic_mean_of_primes (l : List ℕ) (h : l = [22, 23, 26, 29, 31]) :
  (∑ x in l.filter Nat.prime, x : ℚ) / l.filter Nat.prime |>.length = 83 / 3 := by
  sorry

end arithmetic_mean_of_primes_l178_178702


namespace c1_c2_not_collinear_l178_178341

open Vector

noncomputable def a : ℝ × ℝ × ℝ := (0, 3, -2)
noncomputable def b : ℝ × ℝ × ℝ := (1, -2, 1)
noncomputable def c1 := (5 * a.1 - 2 * b.1, 5 * a.2 - 2 * b.2, 5 * a.3 - 2 * b.3)
noncomputable def c2 := (3 * a.1 + 5 * b.1, 3 * a.2 + 5 * b.2, 3 * a.3 + 5 * b.3)

theorem c1_c2_not_collinear : ¬ (∃ γ : ℝ, c1 = γ • c2) :=
by
  sorry

end c1_c2_not_collinear_l178_178341


namespace extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l178_178469

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x * exp x - (x + 1) ^ 2

-- Question 1: Extreme value when a = -1
theorem extreme_value_when_a_is_neg_one : 
  f (-1) (-1) = 1 / exp 1 := sorry

-- Question 2: Range of a such that ∀ x ∈ [-1, 1], f(x) ≤ 0
theorem range_of_a_for_f_non_positive :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 4 / exp 1 := sorry

end extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l178_178469


namespace correlation_of_derivative_variance_of_derivative_l178_178430

noncomputable def k_x (D α : ℝ) (tau : ℝ) : ℝ :=
  D * Real.exp (- α * Real.abs tau) * (1 + α * Real.abs tau)

noncomputable def k_dx (D α : ℝ) (tau : ℝ) : ℝ :=
  D * α^2 * Real.exp (- α * Real.abs tau) * (1 - α * Real.abs tau)

theorem correlation_of_derivative (D α : ℝ) (tau : ℝ) (hα : α > 0) :
  (λ tau, (k_x D α).second_derivative tau) = k_dx D α :=
by
  -- Proof of the correlation function of the derivative
  sorry

theorem variance_of_derivative (D α : ℝ) (hα : α > 0) :
  variance (λ t, (k_dx D α) 0) = D * α^2 :=
by
  -- Proof that the variance is proportional to D and α²
  sorry

end correlation_of_derivative_variance_of_derivative_l178_178430


namespace suff_but_not_nec_condition_for_q_l178_178038

noncomputable def p (a : ℝ) : Prop := (1 / a > 1 / 4)
noncomputable def q (a : ℝ) : Prop := ∀ x : ℝ, (a * x^2 + a * x + 1 > 0)

theorem suff_but_not_nec_condition_for_q (a : ℝ) : (p a → (0 < a ∧ a < 4)) ∧ (q a → (0 ≤ a ∧ a < 4)) → (p a → q a) ∧ ¬ (p a ↔ q a) :=
begin
  sorry
end

end suff_but_not_nec_condition_for_q_l178_178038


namespace rectangle_sides_l178_178403

theorem rectangle_sides (a b : ℝ) (h₁ : a < b) (h₂ : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 :=
sorry

end rectangle_sides_l178_178403


namespace number_of_integers_divisible_by_1_to_10_l178_178372

theorem number_of_integers_divisible_by_1_to_10 : 
  (finset.filter (λ n, ∀ k : ℕ, k ∈ (finset.range 11).erase 0 → k ∣ n) (finset.range 10^10 + 1)).card = 3968253 := 
sorry

end number_of_integers_divisible_by_1_to_10_l178_178372


namespace chocolates_sold_at_selling_price_l178_178468

theorem chocolates_sold_at_selling_price
  (C S : ℚ) (n : ℕ)
  (h1 : 44 * C = n * S)
  (h2 : S = 11/6 * C) :
  n = 24 :=
by
  -- Proof steps would be inserted here
  sorry

end chocolates_sold_at_selling_price_l178_178468


namespace product_of_x_y_l178_178846

theorem product_of_x_y (x y : ℝ) :
  (54 = 5 * y^2 + 20) →
  (8 * x^2 + 2 = 38) →
  x * y = Real.sqrt (30.6) :=
by
  intros h1 h2
  -- these would be the proof steps
  sorry

end product_of_x_y_l178_178846


namespace tangent_planes_of_surface_and_given_plane_l178_178732

-- Define the surface and the given plane
def surface (x y z : ℝ) := (x^2 + 4 * y^2 + 9 * z^2 = 1)
def given_plane (x y z : ℝ) := (x + y + 2 * z = 1)

-- Define the tangent plane equations to be proved
def tangent_plane_1 (x y z : ℝ) := (x + y + 2 * z - (109 / (6 * Real.sqrt 61)) = 0)
def tangent_plane_2 (x y z : ℝ) := (x + y + 2 * z + (109 / (6 * Real.sqrt 61)) = 0)

-- The statement to be proved
theorem tangent_planes_of_surface_and_given_plane :
  ∀ x y z, surface x y z ∧ given_plane x y z →
    tangent_plane_1 x y z ∨ tangent_plane_2 x y z :=
sorry

end tangent_planes_of_surface_and_given_plane_l178_178732


namespace cubic_polynomial_value_l178_178113

noncomputable def q (x : ℝ) : ℝ := sorry

theorem cubic_polynomial_value (h1 : [q 2]^3 - 2 = 0)
  (h2 : [q (-2)]^3 + 2 = 0)
  (h3 : [q 5]^3 - 5 = 0) :
  q 13 = sorry := 
sorry

end cubic_polynomial_value_l178_178113


namespace evaluate_expression_l178_178293

noncomputable def inner_sqrt_expr : ℝ :=
  -4 + Real.sqrt (6 * 4 * 3) + 2 * (3 + 5) - (7 + 11)

noncomputable def outer_sqrt_expr : ℝ :=
  Real.sqrt inner_sqrt_expr

noncomputable def final_expr : ℝ :=
  22 + outer_sqrt_expr

theorem evaluate_expression : final_expr ≈ 23.58 :=
by
  sorry

end evaluate_expression_l178_178293


namespace log_to_exp_l178_178305

theorem log_to_exp (b a c : ℝ) (hb : b > 0) (h : b ≠ 1) (h_log : log b a = c) : b ^ c = a :=
sorry

end log_to_exp_l178_178305


namespace area_of_triangle_l178_178754

noncomputable def cosine := real.cos (2/3 * real.pi)
noncomputable def sine := real.sin (2/3 * real.pi)
def a : ℝ × ℝ := (cosine, sine)
def OA (b : ℝ × ℝ) := (a.1 - b.1, a.2 - b.2)
def OB (b : ℝ × ℝ) := (a.1 + b.1, a.2 + b.2)
def orthogonal (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0
def magnitude (u : ℝ × ℝ) := real.sqrt (u.1 * u.1 + u.2 * u.2)
def isIsoscelesRightTriangle (b : ℝ × ℝ) := 
  orthogonal (OB b) (OA b) ∧ 
  magnitude (OB b) = magnitude (OA b)

/-- Given the conditions that a = (cos(2/3π), sin(2/3π)), 
OA = a - b, OB = a + b, and that ΔOAB is an isosceles right triangle 
with O as the right angle vertex, prove that the area of ΔOAB is 1. -/
theorem area_of_triangle {b : ℝ × ℝ} (h : isIsoscelesRightTriangle b) : 
  1 / 2 * magnitude (OB b) * magnitude (OA b) = 1 :=
sorry

end area_of_triangle_l178_178754


namespace geometric_sequence_sum_div_a2_l178_178777

variable {α : Type*} [Field α] [CharZero α]

/-- Given conditions -/
def common_ratio (q : α) : Prop := q = 2

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = a 1 * (1 - (2:α)^n) / (1 - 2)

/-- The main proof problem -/
theorem geometric_sequence_sum_div_a2
  (a : ℕ → α) (S : ℕ → α) (h1 : common_ratio 2) (h2 : sum_first_n_terms a S) :
  S 4 / a 2 = 15 / 2 :=
by
  sorry

end geometric_sequence_sum_div_a2_l178_178777


namespace increasing_G_on_pos_l178_178826

variable (f : ℝ → ℝ)
variable (h_diff : ∀ x > 0, differentiable_at ℝ f x)
variable (h_cond : ∀ x > 0, f x > -x * (deriv f x))

def G (x : ℝ) : ℝ := x * f x

theorem increasing_G_on_pos : ∀ x > 0, 0 < deriv G x :=
by
  intro x hx_pos
  -- Here we would prove that G'(x) = f(x) + x f'(x) > 0
  sorry

end increasing_G_on_pos_l178_178826


namespace greatest_value_x_plus_y_l178_178269

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 ∨ x + y = -6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_x_plus_y_l178_178269


namespace gasoline_price_increase_l178_178588

theorem gasoline_price_increase
  (P Q : ℝ) -- Prices and quantities
  (x : ℝ) -- The percentage increase in price
  (h1 : (P * (1 + x / 100)) * (Q * 0.95) = P * Q * 1.14) -- Given condition
  : x = 20 := 
sorry

end gasoline_price_increase_l178_178588


namespace simplify_expression_l178_178460

theorem simplify_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x / y) ^ (y - x) :=
by
  sorry

end simplify_expression_l178_178460


namespace goldfish_sold_l178_178210

variables (buy_price sell_price tank_cost short_percentage : ℝ)

theorem goldfish_sold (h1 : buy_price = 0.25)
                      (h2 : sell_price = 0.75)
                      (h3 : tank_cost = 100)
                      (h4 : short_percentage = 0.45) :
  let profit_per_goldfish := sell_price - buy_price in
  let shortfall := tank_cost * short_percentage in
  let earnings := tank_cost - shortfall in
  let goldfish_count := earnings / profit_per_goldfish in
  goldfish_count = 110 :=
by {
  let profit_per_goldfish := sell_price - buy_price;
  let shortfall := tank_cost * short_percentage;
  let earnings := tank_cost - shortfall;
  let goldfish_count := earnings / profit_per_goldfish;
  calc goldfish_count
      = earnings / profit_per_goldfish : by exact rfl
  ... = 110 : by sorry
}

end goldfish_sold_l178_178210


namespace limit_of_T_div_n_squared_l178_178110

-- Definition of the sequence T_n as the sum of binomial coefficients
def T (n : ℕ) : ℕ := ∑ k in finset.range (n + 1), nat.choose k 1

-- The statement to prove
theorem limit_of_T_div_n_squared :
  tendsto (λ n, (T n : ℝ) / (n^2 : ℝ)) at_top (nhds (1 / 2)) :=
sorry

end limit_of_T_div_n_squared_l178_178110


namespace greatest_integer_with_gcd_30_eq_5_l178_178257

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l178_178257


namespace integer_solutions_count_eq_11_l178_178446

theorem integer_solutions_count_eq_11 :
  ∃ (count : ℕ), (∀ n : ℤ, (n + 2) * (n - 5) + n ≤ 10 ↔ (n ≥ -4 ∧ n ≤ 6)) ∧ count = 11 :=
by
  sorry

end integer_solutions_count_eq_11_l178_178446


namespace determinant_projection_matrix_l178_178888

noncomputable def projection_matrix (v : ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := v in (1 / (x^2 + y^2)) • ![![x * x, x * y], ![x * y, y * y]]

theorem determinant_projection_matrix (v : ℝ × ℝ) (h : v = (3, -5)) : matrix.det (projection_matrix v) = 0 := 
by
  rw [h, projection_matrix]
  sorry

end determinant_projection_matrix_l178_178888


namespace cos_sum_nonneg_l178_178551

open Real

theorem cos_sum_nonneg (n : ℕ) (α : Fin n → ℝ) :
  (∑ i : Fin n, ∑ j : Fin n, (i + 1) * (j + 1) * cos (α i - α j)) ≥ 0 :=
sorry

end cos_sum_nonneg_l178_178551


namespace number_of_random_events_l178_178333

-- Definition of the conditions
def certain_event_1: Prop := True
def impossible_event_2: Prop := ∀ x : ℝ, x^2 + 2 * x + 8 ≠ 0
def random_event_3: Prop := True
def random_event_4: Prop := True

-- Main theorem
theorem number_of_random_events : (if random_event_3 then 1 else 0) + (if random_event_4 then 1 else 0) = 2 := by
  sorry

end number_of_random_events_l178_178333


namespace deers_distribution_l178_178976

theorem deers_distribution (a_1 d a_2 a_5 : ℚ) 
  (h1 : a_2 = a_1 + d)
  (h2 : 5 * a_1 + 10 * d = 5)
  (h3 : a_2 = 2 / 3) :
  a_5 = 1 / 3 :=
sorry

end deers_distribution_l178_178976


namespace monotonic_intervals_range_of_b_l178_178794

-- Define the given function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (a * x ^ 2 + x + a) * exp (-x)

-- Define the first part for monotonic intervals
theorem monotonic_intervals (a : ℝ) (x : ℝ) (h_a_nonneg : a ≥ 0) :
  (a = 0 → ∀ x, (f' a x > 0 → x < 1) ∧ (f' a x < 0 → x > 1)) ∧
  (a > 0 → ∀ x, ((1 - 1 / a) < x ∧ x < 1 → f' a x > 0) ∧
                 (x < 1 - 1 / a ∨ x > 1 → f' a x < 0)) :=
sorry

-- Define the second part for the range of b
theorem range_of_b (a b : ℝ) (x : ℝ) (h_a_nonpos : a ≤ 0) (h_x_nonneg : x ≥ 0) :
  (∀ a ≤ 0, f a x ≤ b * log (x + 1)) ↔ b ≥ 1 :=
sorry

end monotonic_intervals_range_of_b_l178_178794


namespace expected_area_K_n_l178_178513

def K : Type := { x : ℝ × ℝ // (y : ℝ × ℝ) ∈ equilateral_triangle x 1 } -- Assuming 'equilateral_triangle' is defined elsewhere

def K_n(area_K_n : ℝ) :=
  let n : ℕ
  let random_points : fin n → K
  let translations : finset (ℝ × ℝ) := translations_of_K_contain_points random_points
  let intersections : K := intersection_of_translations translations
  let expected_area : ℝ := E(area of intersections)

  expected_area

-- Now the theorem statement
theorem expected_area_K_n (n : ℕ) : 
  E(area(K_n)) = (2n-1) * (2n-2) / ((2n+1) * (2n+2)) := 
sorry

end expected_area_K_n_l178_178513


namespace f_neg3_equals_2_l178_178432

-- Define the function f as a piecewise function
noncomputable def f : ℤ → ℤ
| x => if h : 0 ≤ x then x + 1 else f (x + 2)

-- The theorem to prove
theorem f_neg3_equals_2 : f (-3) = 2 :=
sorry

end f_neg3_equals_2_l178_178432


namespace PB_value_l178_178087

variables {A B C D P : ℝ}
variables (CD BC AP : ℝ)
variables (hABCD_convex : true) (hCD_perp_AB : ⊥) (hBC_perp_AD : ⊥)
variables (hCD : CD = 50) (hBC : BC = 40) (hAP : AP = 13)

noncomputable def findPB (PB : ℝ) : Prop :=
  let AB := 52 in -- derived during the solution
  PB = AB - AP

theorem PB_value : ∃ PB, findPB PB := 
  sorry

end PB_value_l178_178087


namespace greatest_valid_number_l178_178263

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l178_178263


namespace goldfish_remaining_to_catch_l178_178534

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end goldfish_remaining_to_catch_l178_178534


namespace total_spent_on_concert_tickets_l178_178443

theorem total_spent_on_concert_tickets : 
  let price_per_ticket := 4
  let number_of_tickets := 3 + 5
  let discount_threshold := 5
  let discount_rate := 0.10
  let service_fee_per_ticket := 2
  let initial_cost := number_of_tickets * price_per_ticket
  let discount := if number_of_tickets > discount_threshold then discount_rate * initial_cost else 0
  let discounted_cost := initial_cost - discount
  let service_fee := number_of_tickets * service_fee_per_ticket
  let total_cost := discounted_cost + service_fee
  total_cost = 44.8 :=
by
  sorry

end total_spent_on_concert_tickets_l178_178443


namespace angle_between_vectors_l178_178775

variables (a b : ℝ^3)
variable (θ : ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

def inner_product (v1 v2 : ℝ^3) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def is_angle_between (a b : ℝ^3) (θ : ℝ) : Prop :=
  vector_magnitude a = 12 ∧
  vector_magnitude b = 9 ∧
  inner_product a b = -54 * real.sqrt 2 ∧
  real.cos θ = - (real.sqrt 2) / 2

theorem angle_between_vectors (a b : ℝ^3) (θ : ℝ) :
  is_angle_between a b θ → θ = 3 * real.pi / 4 :=
sorry

end angle_between_vectors_l178_178775


namespace constant_term_expansion_l178_178977

theorem constant_term_expansion : 
  let general_term (r : ℕ) := (Nat.choose 5 r) * ((1 : ℝ) / (Real.sqrt x)) ^ (5 - r) * (-2) ^ r in
  ∑ r in Finset.range 6, (x^2 + 1) * general_term r = -42 :=
sorry

end constant_term_expansion_l178_178977


namespace units_digit_7_pow_3_pow_4_l178_178744

theorem units_digit_7_pow_3_pow_4 :
  (7 ^ (3 ^ 4)) % 10 = 7 :=
by
  -- Here's the proof placeholder
  sorry

end units_digit_7_pow_3_pow_4_l178_178744


namespace count_multiples_of_4_l178_178449

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l178_178449


namespace find_A_l178_178971

noncomputable def A_value (A B C : ℝ) := (A = 1/4) 

theorem find_A : 
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / (x - 3)^2)) →
  A_value A B C :=
by 
  sorry

end find_A_l178_178971


namespace count_elements_as_difference_of_primes_l178_178054

def is_prime (n : ℕ) : Prop := sorry -- Prime number definition goes here.

def is_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 - p2

def is_in_set (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 + 10 * k

theorem count_elements_as_difference_of_primes : 
  {n : ℕ | is_in_set n ∧ is_difference_of_two_primes n}.to_finset.card = 1 :=
sorry

end count_elements_as_difference_of_primes_l178_178054


namespace min_value_expr_l178_178758

theorem min_value_expr (x y : ℝ) : 
  ∃ min_val, min_val = 2 ∧ min_val ≤ (x + y)^2 + (x - 1/y)^2 :=
sorry

end min_value_expr_l178_178758


namespace total_earnings_correct_l178_178626

-- Defining the conditions provided
variables (x y : ℝ)
def investment_ratio_A : ℝ := 3 * x
def investment_ratio_B : ℝ := 4 * x
def investment_ratio_C : ℝ := 5 * x

def return_ratio_A : ℝ := 6 * y / 100
def return_ratio_B : ℝ := 5 * y / 100
def return_ratio_C : ℝ := 4 * y / 100

def earnings_A : ℝ := investment_ratio_A * return_ratio_A
def earnings_B : ℝ := investment_ratio_B * return_ratio_B

-- Given condition that B earns Rs. 350 more than A
axiom earnings_difference (x y : ℝ) : earnings_B = earnings_A + 350

-- Defining the total earnings
def total_earnings : ℝ := earnings_A + earnings_B + (investment_ratio_C * return_ratio_C)

-- The proof statement
theorem total_earnings_correct : total_earnings = 10150 :=
by
  sorry

end total_earnings_correct_l178_178626


namespace distance_in_interval_l178_178205

open Set Real

def distance_to_town (d : ℝ) : Prop :=
d < 8 ∧ 7 < d ∧ 6 < d

theorem distance_in_interval (d : ℝ) : distance_to_town d → d ∈ Ioo 7 8 :=
by
  intro h
  have d_in_Ioo_8 := h.left
  have d_in_Ioo_7 := h.right.left
  have d_in_Ioo_6 := h.right.right
  /- The specific steps for combining inequalities aren't needed for the final proof. -/
  sorry

end distance_in_interval_l178_178205


namespace tan_alpha_eq_2_l178_178819

theorem tan_alpha_eq_2 
  (α : ℝ)
  (h : cos α + 2 * sin α = - real.sqrt 5) : 
  tan α = 2 :=
sorry

end tan_alpha_eq_2_l178_178819


namespace area_PQR_is_1_12_l178_178407

variable {A B C P Q R M : Type}
variable [Triangle A B C]
variable [Median AK BL CN]
variable [Centroid M AK BL CN]
variable (area_ABC : triangle_area A B C = 1) 
variable (AP_eq_PK : segment_length AP = segment_length PK)
variable (BQ_to_QL : segment_length BQ = (1/3) * segment_length BL)
variable (CR_to_RN : segment_length CR = (5/9) * segment_length CN)

-- Defining area of triangle PQR
def triangle_area_PQR : ℝ := triangle_area P Q R

-- Proposition: The area of triangle PQR is 1/12
theorem area_PQR_is_1_12 : triangle_area_PQR = 1 / 12 :=
by sorry

end area_PQR_is_1_12_l178_178407


namespace polynomial_real_roots_l178_178742

noncomputable def smallest_a : ℝ := 2 * (real.cbrt 2)

noncomputable def unique_b (a : ℝ) : ℝ := 6 * real.sqrt (real.cbrt 2)

theorem polynomial_real_roots :
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧
  (∀ x : ℝ, polynomial.eval x (polynomial.C 1 * x ^ 4 - polynomial.C (2 * a) * x ^ 3 + polynomial.C b * x ^ 2 + polynomial.C a * x - polynomial.C a) = 0 →
    ∃ (r s t u : ℝ), r + s + t + u = 2 * a ∧ r * s * t * u = a ∧ r = s = t = u) ∧
  a = smallest_a ∧ b = unique_b smallest_a := sorry

end polynomial_real_roots_l178_178742


namespace quadrilateral_is_trapezoid_l178_178429

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

noncomputable def is_trapezoid (AB BC CD : V) : Prop :=
∃ (k : ℝ), (CD = k • BC) ∧ ∄ (l : ℝ), (AB = l • CD)

theorem quadrilateral_is_trapezoid
  (a b : V) (hac : ∃ (k : ℝ), k ≠ 0 → ¬ (a = k • b))
  (AB BC CD : V)
  (hAB : AB = a + 2 • b)
  (hBC : BC = -4 • a - b)
  (hCD : CD = -5 • a - 3 • b) :
  is_trapezoid AB BC CD :=
sorry

end quadrilateral_is_trapezoid_l178_178429


namespace part_a_part_b_l178_178145

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def collinear (A B C : Point ℝ) : Prop :=
∃ (a b c : ℝ), a * A.x + b * A.y = c ∧ a * B.x + b * B.y = c ∧ a * C.x + b * C.y = c

structure Point (ℝ : Type) [LinearOrderedField ℝ] :=
(x y : ℝ)

variables
  (A B P Q R L T K S : Point ℝ)
  (l a b : set (Point ℝ))
  (hA : A ∈ l)
  (hB : B ∈ l)
  (hP : P ∈ l)
  (hA_B : A ≠ B)
  (hB_P : B ≠ P)
  (hA_P : A ≠ P)
  (ha_per : ∀ x, x ∈ a ↔ ∃ y, x = ⟨0, y⟩)
  (hb_per : ∀ x, x ∈ b ↔ ∃ y, x = ⟨1, y⟩)
  (hPQ : Q ∈ a ∧ ¬Q ∈ l ∧ Q ∈ line_through P Q)
  (hPR : R ∈ b ∧ ¬R ∈ l ∧ R ∈ line_through P R)
  (hL : L ∈ line_through A T ∧ L ∈ line_through B Q ∧ line_through A Q = a ∧ line_through B R = b)
  (hT : T ∈ line_through A T ∧ T ∈ line_through A T)
  (hS : S ∈ line_through B S ∧ S ∈ line_through B Q ∧ line_through A R = a ∧ line_through B Q = b)
  (hK : K ∈ line_through A R ∧ K ∈ line_through B S ∧ K ∈ line_through A R)

theorem part_a : collinear ℝ P T S := sorry

theorem part_b : collinear ℝ P K L := sorry

end part_a_part_b_l178_178145


namespace angle_C_side_c_area_of_triangle_l178_178480

open Real

variables (A B C a b c : Real)

noncomputable def acute_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
  (A < π / 2) ∧ (B < π / 2) ∧ (C < π / 2) ∧
  (a^2 - 2 * sqrt 3 * a + 2 = 0) ∧
  (b^2 - 2 * sqrt 3 * b + 2 = 0) ∧
  (2 * sin (A + B) - sqrt 3 = 0)

noncomputable def length_side_c (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2 - 2 * a * b * cos (π / 3))

noncomputable def area_triangle (a b : ℝ) : ℝ := 
  (1 / 2) * a * b * sin (π / 3)

theorem angle_C (h : acute_triangle A B C a b c) : C = π / 3 :=
  sorry

theorem side_c (h : acute_triangle A B C a b c) : c = sqrt 6 :=
  sorry

theorem area_of_triangle (h : acute_triangle A B C a b c) : area_triangle a b = sqrt 3 / 2 :=
  sorry

end angle_C_side_c_area_of_triangle_l178_178480


namespace ratio_AD_DC_l178_178831

-- Given: Triangle ABC with specific side lengths and point D on AC with specific properties
variable {A B C D : Type} [MetricSpace (A B C)]
variables {AB BC AC AD DC BD : ℝ}
variables {RealisOr := Real.IsOrding}
variables {A B C A1 B1 C1 : ℕ}

axiom distance_AB : dist A B = 6
axiom distance_BC : dist B C = 8.4
axiom distance_AC : dist A C := 10.8
axiom point_D : D ∈ segment A C
axiom distance_BD : dist B D = 6

-- To Prove: The ratio AD:DC is 227:97
theorem ratio_AD_DC : 
  (AD / DC = 227 / 97) :=
begin
  -- axioms and properties need to be proved
  sorry
end

end ratio_AD_DC_l178_178831


namespace ab_range_proof_l178_178759

noncomputable def circle_eq : String := "x^2 + y^2 + 2x - 4y + 1 = 0"
noncomputable def line_eq (a b : ℝ) : String := "ax - by + 1 = 0"
noncomputable def center_of_circle := (-1, 2)
noncomputable def distance_to_line (a b : ℝ) : ℝ := abs (-a - (2*b) + 1) / sqrt (a^2 + b^2)
noncomputable def ab_range := set.Icc 0 (2 - real.sqrt 3)

theorem ab_range_proof (a b : ℝ) : (abs (-a - (2*b) + 1) / sqrt (a^2 + b^2) ≥ 2) → a = 2*b - 1 → ab ∈ ab_range :=
sorry

end ab_range_proof_l178_178759


namespace conjugate_of_complex_number_l178_178179

open Complex

theorem conjugate_of_complex_number :
  let z := (3 - I) / (1 - I)
  ∃ w : ℂ, conj z = w ∧ w = 2 - I :=
by
  sorry

end conjugate_of_complex_number_l178_178179


namespace count_divisibles_l178_178451

theorem count_divisibles (n : ℕ) : ∀ (m : ℕ), m = 20 → count (λ k, k < 100 ∧ k % m = 0) (range 100) = 4 :=
by
  intros m hm
  rw hm
  sorry

end count_divisibles_l178_178451


namespace find_number_l178_178330

variable (x : ℝ)

theorem find_number 
  (h1 : 0.20 * x + 0.25 * 60 = 23) :
  x = 40 :=
sorry

end find_number_l178_178330


namespace sum_of_possible_values_l178_178115

open Real

theorem sum_of_possible_values (x y z : ℝ) (h : x * y * z - x / (y ^ 3) - y / (z ^ 3) - z / (x ^ 3) = 4) :
  let P := (x - 2) * (y - 2) * (z - 2)
  in ∃ (s : ℝ) (a b : ℝ), (P = a) ∧ (P = b) ∧ (s = a + b) ∧ (s = -23) :=
by
  sorry

end sum_of_possible_values_l178_178115


namespace dihedral_angles_ratio_l178_178027

theorem dihedral_angles_ratio (α β γ : ℝ) (h1 : ∀θ, 0 ≤ θ ∧ θ ≤ π) :
  let sin² := (λ x : ℝ, Real.sin x * Real.sin x)
  let cos² := (λ x : ℝ, Real.cos x * Real.cos x)
  sin² α + sin² β + sin² γ = 2 →
  cos² α + cos² β + cos² γ = 1 →
  (sin² α + sin² β + sin² γ) / (cos² α + cos² β + cos² γ) = 2 :=
by
  sorry

end dihedral_angles_ratio_l178_178027


namespace average_score_of_remaining_students_correct_l178_178836

noncomputable def average_score_remaining_students (n : ℕ) (h_n : n > 15) (avg_all : ℚ) (avg_subgroup : ℚ) : ℚ :=
if h_avg_all : avg_all = 10 ∧ avg_subgroup = 16 then
  (10 * n - 240) / (n - 15)
else
  0

theorem average_score_of_remaining_students_correct (n : ℕ) (h_n : n > 15) :
  (average_score_remaining_students n h_n 10 16) = (10 * n - 240) / (n - 15) :=
by
  dsimp [average_score_remaining_students]
  split_ifs with h_avg
  · sorry
  · sorry

end average_score_of_remaining_students_correct_l178_178836


namespace lindas_payment_l178_178914

theorem lindas_payment (pay_per_room : ℚ) (rooms_cleaned : ℚ) (total_payment : ℚ) :
  pay_per_room = 13 / 3 →
  rooms_cleaned = 8 / 5 →
  total_payment = (13 / 3) * (8 / 5) →
  total_payment = 104 / 15 := 
by intros; subst_vars; try rfl; apply sorry

end lindas_payment_l178_178914


namespace roots_real_irrational_for_pure_imaginary_k_l178_178359

theorem roots_real_irrational_for_pure_imaginary_k (k : ℂ) (h : ∃ d : ℝ, k = d * complex.I) :
  ∀ z : ℂ, (8 * z^2 + 6 * complex.I * z + k = 0) → z ∈ ℝ ∧ irrational z :=
by
  sorry

end roots_real_irrational_for_pure_imaginary_k_l178_178359


namespace product_of_distances_eq_b_squared_l178_178414

variables {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt_ab : a > b)
variables (θ : ℝ)

def points_distance_product :=
  let point1 := (real.sqrt (a^2 - b^2), 0 : ℝ)
  let point2 := (-real.sqrt (a^2 - b^2), 0 : ℝ)
  let line_eq := (x y : ℝ) ↦ x * real.cos θ / a + y * real.sin θ / b - 1
  let dist (p : ℝ × ℝ) := |line_eq p.1 p.2| / (real.sqrt (real.cos θ ^ 2 / (a ^ 2) + real.sin θ ^ 2 / (b ^ 2)))
  (dist point1) * (dist point2)

theorem product_of_distances_eq_b_squared : points_distance_product h_pos_a h_pos_b h_gt_ab θ = b^2 :=
sorry

end product_of_distances_eq_b_squared_l178_178414


namespace equal_copper_content_alloy_l178_178364

theorem equal_copper_content_alloy (a b : ℝ) :
  ∃ x : ℝ, 0 < x ∧ x < 10 ∧
  (10 - x) * a + x * b = (15 - x) * b + x * a → x = 6 :=
by
  sorry

end equal_copper_content_alloy_l178_178364


namespace max_value_a2b_l178_178751

theorem max_value_a2b (a b : ℝ) (h1 : 0 ≤ a - b ∧ a - b ≤ 1) (h2 : 1 ≤ a + b ∧ a + b ≤ 4) (h3 : a - 2b = 1) : 8 * a + 2002 * b = 8 :=
sorry

end max_value_a2b_l178_178751


namespace problem_area_W_l178_178991

def W : Set (ℝ × ℝ) :=
  {p | ∥2 - |p.1| - (|p.2| - 2)∥ ≤ 1}

noncomputable def area_W : ℝ := Geometry.area W

theorem problem_area_W : area_W = 30 := by
  sorry

end problem_area_W_l178_178991


namespace value_of_polynomial_l178_178062

variable {R : Type} [CommRing R]

theorem value_of_polynomial 
  (m : R) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2019 = 2022 := by
  sorry

end value_of_polynomial_l178_178062


namespace angle_a_90_angle_b_112_5_l178_178102

-- Definitions of the vertices and initial conditions of the square
variable {A B C D : Type} -- Placeholder for vertices
variable [linear_ordered_field A] -- Assuming coordinates can be represented in a linear ordered field
variable (a b : A)

-- Define the properties and conditions
def is_square (v₁ v₂ v₃ v₄ : A) : Prop :=
  (v₁ = 90 ∧ v₂ = 90 ∧ v₃ = 90 ∧ v₄ = 90)

def fold_brings_BD_to_diagonal (B D : A) : Prop :=
  true -- This is a conceptual placeholder

def fold_brings_C_to_A (C A : A) : Prop :=
  true -- This is a conceptual placeholder

-- The main theorems to be proven
theorem angle_a_90 (h_square : is_square A B C D)
    (h_fold1 : fold_brings_BD_to_diagonal B D)
    (h_fold2 : fold_brings_C_to_A C A) :
  a = 90 := 
sorry

theorem angle_b_112_5 (h_square : is_square A B C D)
    (h_fold1 : fold_brings_BD_to_diagonal B D)
    (h_fold2 : fold_brings_C_to_A C A) :
  b = 112.5 := 
sorry

end angle_a_90_angle_b_112_5_l178_178102


namespace min_max_values_f_l178_178195

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l178_178195


namespace factor_count_l178_178728

theorem factor_count (x : ℤ) : 
  (x^12 - x^3) = x^3 * (x - 1) * (x^2 + x + 1) * (x^6 + x^3 + 1) -> 4 = 4 :=
by
  sorry

end factor_count_l178_178728


namespace max_points_per_player_l178_178077

theorem max_points_per_player
  (num_players : ℕ)
  (total_points : ℕ)
  (min_points_per_player : ℕ)
  (extra_points : ℕ)
  (scores_by_two_or_three : Prop)
  (fouls : Prop) :
  num_players = 12 →
  total_points = 100 →
  min_points_per_player = 8 →
  scores_by_two_or_three →
  fouls →
  extra_points = (total_points - num_players * min_points_per_player) →
  q = min_points_per_player + extra_points →
  q = 12 :=
by
  intros
  sorry

end max_points_per_player_l178_178077


namespace length_AC_l178_178933

theorem length_AC (A B C : Type) (r : ℝ) (A_on_circle : dist A B = 8) (radius : r = 8) (AB : dist A B = 10) (C_midpoint : ∀ C, is_midpoint_of_arc A B C) : dist A C = Real.sqrt (128 - 16 * Real.sqrt 39) := 
  sorry

end length_AC_l178_178933


namespace right_triangle_right_angles_l178_178813

theorem right_triangle_right_angles (T : Triangle) (h1 : T.is_right_triangle) :
  T.num_right_angles = 1 :=
sorry

end right_triangle_right_angles_l178_178813


namespace solution_l178_178097

variable (x : ℝ)
variable (friend_contribution : ℝ) (james_payment : ℝ)

def adoption_fee_problem : Prop :=
  friend_contribution = 0.25 * x ∧
  james_payment = 0.75 * x ∧
  james_payment = 150 →
  x = 200

  theorem solution : adoption_fee_problem x friend_contribution james_payment :=
  by
  unfold adoption_fee_problem
  intros
  sorry

end solution_l178_178097


namespace min_air_pollution_at_4_a_half_range_of_a_for_max_index_l178_178964

def air_pollution_index (a x : ℝ) : ℝ :=
  abs (Real.log (x + 1) / Real.log 25 - a) + 2 * a + 1

theorem min_air_pollution_at_4_a_half :
  ∀ x ∈ set.Icc (0 : ℝ) 24, air_pollution_index 0.5 x ≥ air_pollution_index 0.5 4 := by
  sorry

theorem range_of_a_for_max_index :
  (∀ x ∈ set.Icc (0 : ℝ) 24, air_pollution_index a x ≤ 3) ↔ a ∈ set.Ioo (0 : ℝ) (2 / 3) ∪ {2/3} := by
  sorry

end min_air_pollution_at_4_a_half_range_of_a_for_max_index_l178_178964


namespace total_cost_l178_178134

-- Definitions corresponding to the conditions
def puppy_cost : ℝ := 10
def daily_food_consumption : ℝ := 1 / 3
def food_bag_content : ℝ := 3.5
def food_bag_cost : ℝ := 2
def days_in_week : ℝ := 7
def weeks_of_food : ℝ := 3

-- Statement of the problem
theorem total_cost :
  let 
    days := weeks_of_food * days_in_week,
    total_food_needed := days * daily_food_consumption,
    bags_needed := total_food_needed / food_bag_content,
    food_cost := bags_needed * food_bag_cost
  in
  puppy_cost + food_cost = 14 := 
by
  sorry

end total_cost_l178_178134


namespace functional_equation_solved_value_of_f_sqrt_2014_l178_178400

noncomputable def f (x : ℝ) : ℝ := 1 - (x^2) / 2

theorem functional_equation_solved (x y : ℝ) : 
  f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1 := 
by 
  sorry

theorem value_of_f_sqrt_2014 : f(Real.sqrt 2014) = -1006 :=
by 
  sorry

end functional_equation_solved_value_of_f_sqrt_2014_l178_178400


namespace option_D_correct_l178_178282

theorem option_D_correct (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 :=
by sorry

end option_D_correct_l178_178282


namespace john_spends_on_memory_cards_l178_178100

theorem john_spends_on_memory_cards :
  (10 * (3 * 365)) / 50 * 60 = 13140 :=
by
  sorry

end john_spends_on_memory_cards_l178_178100


namespace pq_value_l178_178594

open Nat

theorem pq_value (p q : ℕ) (h1 : prime p) (h2 : prime q) (h3 : p * 1 + q = 99) : p * q = 194 :=
sorry

end pq_value_l178_178594


namespace hyperbola_same_foci_as_ellipse_eccentricity_two_l178_178743

theorem hyperbola_same_foci_as_ellipse_eccentricity_two
  (a b c e : ℝ)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = 5 ∧ b = 3 ∧ c = 4))
  (eccentricity_eq : e = 2) :
  ∃ x y : ℝ, (x^2 / (c / e)^2 - y^2 / (c^2 - (c / e)^2) = 1) ↔ (x^2 / 4 - y^2 / 12 = 1) :=
by
  sorry

end hyperbola_same_foci_as_ellipse_eccentricity_two_l178_178743


namespace min_value_x_plus_2y_l178_178823

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2y_l178_178823


namespace center_sum_is_twelve_l178_178176

-- Define the given circle equation
def circle_eq : Prop := ∀ x y : ℝ, x^2 + y^2 = 6 * x + 18 * y - 63

-- Define the center coordinates (h, k) of the circle
def center_coords (h k : ℝ) : Prop := (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 18*y + 63)

-- Prove that the sum of the center coordinates is 12
theorem center_sum_is_twelve : ∃ h k : ℝ, center_coords h k ∧ (h + k = 12) :=
by 
  existsi 3
  existsi 9
  dsimp [center_coords]
  sorry

end center_sum_is_twelve_l178_178176


namespace unique_geometric_seq_a_no_arithmetic_diff_sequences_l178_178800

variable {α : Type*} [OrderedRing α]
variable {q1 q2 a b : α}

-- Condition Definitions
def geometric_seq {q : α} (a : α) (n : ℕ) : ℕ → α
| 0 => a
| (n+1) => a * q ^ n

def arithmetic_seq {d : α} (a : α) (n : ℕ) : α
| 0 => a
| (n+1) => a + d * n

-- Part 1 Proving a = 1/3
theorem unique_geometric_seq_a : ∀ {q : α} {a1 b1 a b : α}, 
  a1 = a > 0 → b1 - a1 = 1 → b1 * q^1 - a * q = 2 → b1 * q^2 - a * q^2 = 3 → a = (1 : α) / 3 :=
begin
  assume α a1 b1 a b h₁ h₂ h₃ h₄,
  sorry
end

-- Part 2 Non-existence of Sequences
theorem no_arithmetic_diff_sequences: ∀ {q1 q2 : α} {a1 b1 a : α},
  a1 = a > 0 → b1 * q1 ≠ a1 * q2 →
  b1 * q1 = 1 + a1 → b1 * q2 - a * q2 = 2 → 
  b1 * q2^2 - a * q1^2 = 3 → ¬ ∃ d, 
  ( ∀ n, b1 * q1^n - a * q2^n = arithmetic_seq d 0 n) :=
begin
  assume α a1 b1 a h₁ h₂ h₃ h₄,
  sorry
end

end unique_geometric_seq_a_no_arithmetic_diff_sequences_l178_178800


namespace kathryn_financial_statement_l178_178504

def kathryn_remaining_money (rent : ℕ) (salary : ℕ) (share_rent : ℕ → ℕ) (total_expenses : ℕ → ℕ) (remaining_money : ℕ → ℕ) : Prop :=
  rent = 1200 ∧
  salary = 5000 ∧
  share_rent rent = rent / 2 ∧
  ∀ rent_total, total_expenses (share_rent rent_total) = (share_rent rent_total) + 2 * rent_total ∧
  remaining_money salary total_expenses = salary - total_expenses (share_rent rent)

theorem kathryn_financial_statement : kathryn_remaining_money 1200 5000 (λ rent, rent / 2) (λ rent, rent / 2 + 2 * rent) (λ salary expenses, salary - expenses (λ rent, rent / 2)) :=
by {
  sorry
}

end kathryn_financial_statement_l178_178504


namespace eliza_received_12_almonds_l178_178442

theorem eliza_received_12_almonds (y : ℕ) (h1 : y - 8 = y / 3) : y = 12 :=
sorry

end eliza_received_12_almonds_l178_178442


namespace repeating_decimal_as_fraction_l178_178733

theorem repeating_decimal_as_fraction : 
  (0.\overline{36} : ℝ) = (4/11 : ℚ) := 
sorry

end repeating_decimal_as_fraction_l178_178733


namespace admission_exam_cutoff_score_l178_178842

theorem admission_exam_cutoff_score :
  ∃ (X : ℝ), 
    let total_participants := 1000
    let overall_avg_score := 55
    let admitted_candidates := 200
    let difference_avg_scores := 60
    let cutoff_difference := 4
    let non_admitted_candidates := total_participants - admitted_candidates
    let non_admitted_avg_score := X - difference_avg_scores
    let total_admitted_score := admitted_candidates * X
    let total_non_admitted_score := non_admitted_candidates * non_admitted_avg_score
    let combined_total_score := total_participants * overall_avg_score
    in ((total_admitted_score + total_non_admitted_score = combined_total_score) ∧
        (X - cutoff_difference = 99)) :=
sorry

end admission_exam_cutoff_score_l178_178842


namespace greatest_int_less_than_200_gcd_30_is_5_l178_178245

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l178_178245


namespace circumcenter_not_lattice_point_l178_178840

namespace CircumcenterProof

-- Define a triangle with integer coordinates
structure LatticeTriangle :=
  (A B C : ℤ × ℤ)
  -- for A = (0,0), which is irrelevant as long as A, B, C are distinct
  (non_degenerate : A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- Define the circumcenter of a triangle
noncomputable def circumcenter (T : LatticeTriangle) : ℚ × ℚ :=
  -- Circumcenter calculation logic placeholder
  sorry

-- Define the similarity between two triangles
def similar (T1 T2 : LatticeTriangle) : Prop :=
  -- Similarity relation logic placeholder
  sorry

-- Define the area of a triangle
noncomputable def area (T : LatticeTriangle) : ℚ :=
  -- Area calculation logic placeholder
  sorry

-- Define a minimal area triangle similar to a given triangle
noncomputable def smallest_similar_area_triangle (T : LatticeTriangle) : LatticeTriangle :=
  -- Similar minimal area triangle logic placeholder
  sorry

-- The theorem statement
theorem circumcenter_not_lattice_point (T : LatticeTriangle) :
  let T_min := smallest_similar_area_triangle T in
  let O := circumcenter T_min in
  O.1 ∉ ℤ \/ O.2 ∉ ℤ :=
sorry

end CircumcenterProof

end circumcenter_not_lattice_point_l178_178840


namespace number_of_proper_subsets_of_intersection_l178_178415

def sets_A (x y : ℝ) : Prop := y = |x|
def sets_B (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

theorem number_of_proper_subsets_of_intersection :
  let A := {p : ℝ × ℝ | sets_A p.1 p.2}
  let B := {p : ℝ × ℝ | sets_B p.1 p.2}
  let intersection := A ∩ B
  ∃ (n : ℕ), n = finset.filter (λ s, s ⊂ intersection).to_finset.powerset.card - 1 ∧ n = 3 :=
by
  sorry

end number_of_proper_subsets_of_intersection_l178_178415


namespace green_tea_cost_in_july_l178_178580

-- Definitions based on conditions
variable (C : ℝ) -- Cost per pound of green tea and coffee in June

-- Define conditions
def price_coffee_july (C : ℝ) : ℝ := 2 * C
def price_green_tea_july (C : ℝ) : ℝ := 0.1 * C
def mixture_cost := 3.15 -- Total cost of 3 lbs of mixture in July (equal quantities)

-- Cost equation based on the mixture containing equal quantities
theorem green_tea_cost_in_july :
  ∃ (C : ℝ), (1.5 * price_green_tea_july C + 1.5 * price_coffee_july C = mixture_cost) → price_green_tea_july C = 0.10 := 
by
  sorry -- Proof is omitted.

end green_tea_cost_in_july_l178_178580


namespace count_integers_reaching_one_l178_178369

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n * n - 1 else n / 2

def is_power_of_two (n : ℕ) : Prop :=
∃ k : ℕ, n = 2^k

theorem count_integers_reaching_one : 
  (finset.card {n ∈ finset.range 151 | 
    ∃ m : ℕ, (nat.iterate g m n) = 1} = 8) :=
sorry

end count_integers_reaching_one_l178_178369


namespace Q_subset_P_l178_178040

def P : Set ℝ := {x | x ≥ -1}
def Q : Set ℝ := {y | y ≥ 0}

theorem Q_subset_P : Q ⊆ P :=
by
  sorry

end Q_subset_P_l178_178040


namespace coeff_of_b_neg1_in_expansion_eq_zero_l178_178486

theorem coeff_of_b_neg1_in_expansion_eq_zero : 
  (coefficient b (-1) (expand (6 : ℕ) (λ x, b - (1/b)))).val = 0 := by
  sorry

end coeff_of_b_neg1_in_expansion_eq_zero_l178_178486


namespace bisection_method_zero_l178_178306

noncomputable def approxZeroBisection := 
  ∃ c : ℝ, (2.5 < c ∧ c < 2.75) ∧ (|c - 2.6| < 0.05)

theorem bisection_method_zero (h1: Real.log 2.5 ≈ 0.398)
  (h2: Real.log 2.75 ≈ 0.439)
  (h3: Real.log 2.5625 ≈ 0.409) :
  approxZeroBisection := 
by
  sorry

end bisection_method_zero_l178_178306


namespace greatest_integer_gcf_l178_178268

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l178_178268


namespace jack_sugar_amount_l178_178870

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l178_178870


namespace cos_sin_eq_l178_178013

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end cos_sin_eq_l178_178013


namespace puppy_food_cost_l178_178136

theorem puppy_food_cost :
  let puppy_cost : ℕ := 10
  let days_in_week : ℕ := 7
  let total_number_of_weeks : ℕ := 3
  let cups_per_day : ℚ := 1 / 3
  let cups_per_bag : ℚ := 3.5
  let cost_per_bag : ℕ := 2
  let total_days := total_number_of_weeks * days_in_week
  let total_cups := total_days * cups_per_day
  let total_bags := total_cups / cups_per_bag
  let food_cost := total_bags * cost_per_bag
  let total_cost := puppy_cost + food_cost
  total_cost = 14 := by
  sorry

end puppy_food_cost_l178_178136


namespace DEF_area_is_60_square_centimeters_l178_178608

-- Define the isosceles triangle and its side lengths
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b

-- Define the area of a triangle given its side lengths and height
def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

-- Problem conditions
def DEF_is_isosceles : Prop := is_isosceles_triangle 13 13 24

-- Given the height derived from the sides using Pythagorean theorem
def height_from_pythagorean_theorem : ℝ := 5

-- Statement to prove
theorem DEF_area_is_60_square_centimeters :
  DEF_is_isosceles → 
  triangle_area 24 height_from_pythagorean_theorem = 60 :=
by
  -- Proof goes here
  sorry

end DEF_area_is_60_square_centimeters_l178_178608


namespace three_squares_sum_l178_178155

theorem three_squares_sum (n : ℤ) (h : n > 5) : 
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 :=
by sorry

end three_squares_sum_l178_178155


namespace hyperbola_parabola_intersection_l178_178762

theorem hyperbola_parabola_intersection (x y m : ℝ)
  (hyp1 : x^2 - y^2/3 = 1)
  (hyp2 : ∃ M N : (ℝ × ℝ), -- M and N are points on the hyperbola
           ((y = x + m) → -- symmetric about the line
           ∃ P : (ℝ × ℝ), 
           (P = ((fst M + fst N) / 2, (snd M + snd N) / 2)) ∧
           P ∈ { p | (snd p)^2 = 18 * (fst p) })) :
  m = 0 ∨ m = -8 :=
sorry

end hyperbola_parabola_intersection_l178_178762


namespace equal_arcs_and_bisector_l178_178401

-- Definitions of points, angles, circles, and lines
variable (O C D A A1 : Point)
variable (r : ℝ) -- radius
variable (ϕ : ℝ) -- central angle in radians
variable (t : Line) -- tangent at D

-- Conditions
axiom circle_O : Circle O r
axiom diameter_CD : Diameter circle_O C D
axiom tangent_t : TangentAt t D
axiom central_angle_ϕ : CentralAngle (Arc D A) circle_O ϕ

axiom circle_C : Circle C r
axiom intersects_ray_CA : Intersects (CircleIntersection circle_C (Ray C A)) A1

-- Proofs to be obtained
theorem equal_arcs_and_bisector :
  length (Arc D A) = length (Arc D A1) ∧
  Bisects (Ray D A1) (AngleBetween (Ray DA) t) := by
  sorry

end equal_arcs_and_bisector_l178_178401


namespace initial_erasers_count_l178_178912

noncomputable def erasers_lost := 42
noncomputable def erasers_ended_up_with := 53

theorem initial_erasers_count (initial_erasers : ℕ) : 
  initial_erasers_ended_up_with = initial_erasers - erasers_lost → initial_erasers = 95 :=
by
  sorry

end initial_erasers_count_l178_178912


namespace cos_C_value_l178_178830

variables {A B C a b c : ℝ}
hypothesis (h1 : 8 * b = 5 * c)
hypothesis (h2 : C = 2 * B)
noncomputable def cos_C : ℝ := cos C

theorem cos_C_value : cos_C = 7 / 25 := sorry

end cos_C_value_l178_178830


namespace exists_divisor_pair_l178_178565

theorem exists_divisor_pair (n : ℕ) (s : Finset ℕ) (hs : s.card = n + 1) (hs_subset : s ⊆ Finset.range (2 * n + 1)) :
  ∃ a b ∈ s, a ∣ b ∨ b ∣ a := 
by
  sorry

end exists_divisor_pair_l178_178565


namespace total_profit_calculation_l178_178688

-- Defining the investments of A, B, and C
variable (x : ℝ)
def A_investment := 3 * x
def C_investment := (3 * x) * (3 / 2)

-- Defining the condition for C's share
def C_share := 18000.000000000004

-- The ratio of their investments is A : B : C = 3x : x : 9x/2
def total_parts : ℝ := 6 + 2 + 9

-- Prove that the total profit earned at the end of the year is Rs. 34000.00000000001
theorem total_profit_calculation :
  let one_part := C_share / 9 in
  let total_profit := one_part * total_parts in
  total_profit ≈ 34000.00000000001
:= by
  sorry

end total_profit_calculation_l178_178688


namespace original_price_of_shirts_is_397_66_l178_178590

variable (P : ℝ)
variable (h : 0.95 * 0.9 * P = 340)

theorem original_price_of_shirts_is_397_66 : P = 397.66 :=
by
  have h_main : P = 340 / (0.95 * 0.9) := by linarith
  have : P ≈ 397.66 := by linarith
  exact h_main

end original_price_of_shirts_is_397_66_l178_178590


namespace kathryn_remaining_money_l178_178507

variables (rent food_travel salary monthly_expenses remaining : ℝ)

-- Conditions
def rent_value := rent = 1200
def food_travel_expenses := food_travel = 2 * rent
def salary_value := salary = 5000
def shared_rent := monthly_expenses = rent / 2 + food_travel

-- Question and Answer
def money_remaining := remaining = salary - monthly_expenses

-- Theorem to prove
theorem kathryn_remaining_money (h1 : rent_value) (h2 : food_travel_expenses) (h3 : salary_value) (h4 : shared_rent) : money_remaining :=
sorry

end kathryn_remaining_money_l178_178507


namespace min_f_l178_178426

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ Icc 0 2 then
  x^2 - 2*x + 2
else if h : x ∈ Icc (-2 : ℝ) 0 then
  - (x^2 - 2*(-x) + 2)
else
  0

theorem min_f : ∀ x ∈ Icc (-2:ℝ) (2:ℝ), f(x) ≥ -2 :=
sorry

end min_f_l178_178426


namespace equivalent_problem_l178_178900

noncomputable def f : ℝ → ℝ := sorry

def M2r (x1 x2 : ℝ) (p1 p2 : ℝ) : ℝ := (p1 * x1^r + p2 * x2^r)^(1/r)

def Mn0 (f : ℝ → ℝ) (x : ℕ → ℝ) (p : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, f (x i)^p i

def Mnr (x : ℕ → ℝ) (p : ℕ → ℝ) (n : ℕ) : ℝ := (∑ i in finset.range n, p i * x i^r)^(1/r)

theorem equivalent_problem 
  (I : set ℝ)
  (f_cont : continuous_on f I)
  (f_pos : ∀ x ∈ I, 0 < f x)
  (x : ℕ → ℝ)
  (p : ℕ → ℝ)
  (hxi : ∀ i, x i ∈ I)
  (positive_p : ∀ i, 0 < p i)
  (p_sum : ∑ i in finset.range n, p i = 1)
  (h2 : ∀ (x1 x2 : ℝ) (p1 p2 : ℝ), 
    0 < p1 → 0 < p2 → p1 + p2 = 1 → 
    f x1^p1 * f x2^p2 ≥ f (M2r x1 x2 p1 p2))
  (n : ℕ)
  (n_ge_2 : 2 ≤ n) :
  Mn0 f x p n ≥ f (Mnr x p n) :=
sorry

end equivalent_problem_l178_178900


namespace four_digit_numbers_count_l178_178749

/-- 
  Prove that the number of valid four-digit numbers that can be formed using
  only the digits 1, 2, and 3, each used at least once, and with no identical digits
  being adjacent, is equal to 18.
-/
theorem four_digit_numbers_count : 
  (∃ numbers : Finset (Set (List Nat)), 
    (∀ number ∈ numbers, all_digits number ∧ no_adjacent_identical number ∧ has_all_digits number) 
    ∧ numbers.card = 18) :=
sorry

-- Definitions for conditions 

def all_digits (number : List Nat) : Prop := 
  ∀ d ∈ (number.toFinset), d = 1 ∨ d = 2 ∨ d = 3

def no_adjacent_identical (number : List Nat) : Prop :=
  ∀ (i : Nat), i < number.length - 1 → number.get i ≠ number.get (i+1)

def has_all_digits (number : List Nat) : Prop := 
  ∀ d ∈ {1, 2, 3}, d ∈ number.toFinset


end four_digit_numbers_count_l178_178749


namespace integer_solution_count_correct_l178_178569

noncomputable def integer_solution_count : ℕ := by 
  /- Define the parameters and the inequality -/
  let α : ℝ := 3
  let γ : ℝ := 1
  let a : ℝ := -15
  let b : ℝ := 54
  let inequality (x : ℝ) : Prop := 
    (sqrt ((x / γ) + (α + 2)) - (x / γ) - α) / (x^2 + a * x + b) ≥ 0

  /- Count integer solutions -/
  let integer_solutions := { x : ℤ | inequality (x : ℝ) }.to_finset
  exact integer_solutions.card

/- The count of integer solutions should match the solution's result. -/
theorem integer_solution_count_correct : integer_solution_count = 7 := by
  sorry

end integer_solution_count_correct_l178_178569


namespace bags_per_day_l178_178962

theorem bags_per_day (total_bags : ℕ) (days : ℕ) (h_total_bags : total_bags = 519) (h_days : days = 13) :
  total_bags / days = 40 :=
by {
  rw [h_total_bags, h_days],
  norm_num,
}

end bags_per_day_l178_178962


namespace personal_trainer_cost_proof_l178_178537

-- Define the conditions
def hourly_wage_before_raise : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 5
def old_bills_per_week : ℝ := 600
def leftover_money : ℝ := 980

-- Define the question
def new_hourly_wage : ℝ := hourly_wage_before_raise * (1 + raise_percentage)
def weekly_hours : ℕ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def total_weekly_expenses : ℝ := weekly_earnings - leftover_money
def personal_trainer_cost_per_week : ℝ := total_weekly_expenses - old_bills_per_week

-- Theorem statement
theorem personal_trainer_cost_proof : personal_trainer_cost_per_week = 100 := 
by
  -- Proof to be filled
  sorry

end personal_trainer_cost_proof_l178_178537


namespace exists_equilateral_triangle_diff_disks_equilateral_triangle_side_length_gt_96_l178_178378

-- Definition of the condition
def disk_radius : ℝ := 0.001

-- The lattice disk assumption
axiom lattice_disks :
  ∀ (x y : ℤ), ∃ (c : ℤ × ℤ), dist (x, y : ℝ × ℝ) ((c.fst, c.snd) : ℝ × ℝ) < disk_radius

-- Problem (a)
theorem exists_equilateral_triangle_diff_disks :
  ∃ (a b c : ℤ × ℤ), 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    is_equilateral ⟨a.1, a.2⟩ ⟨b.1, b.2⟩ ⟨c.1, c.2⟩ :=
sorry

-- Problem (b)
theorem equilateral_triangle_side_length_gt_96 :
  ∀ (a b c : ℤ × ℤ),
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    is_equilateral ⟨a.1, a.2⟩ ⟨b.1, b.2⟩ ⟨c.1, c.2⟩ →
    dist (a : ℝ × ℝ) (b : ℝ × ℝ) > 96 ∧ 
    dist (b : ℝ × ℝ) (c : ℝ × ℝ) > 96 ∧ 
    dist (c : ℝ × ℝ) (a : ℝ × ℝ) > 96 :=
sorry

end exists_equilateral_triangle_diff_disks_equilateral_triangle_side_length_gt_96_l178_178378


namespace inequality_proof_l178_178516

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x^2 + y * z) + 1 / (y^2 + z * x) + 1 / (z^2 + x * y)) ≤ 
  (1 / 2) * (1 / (x * y) + 1 / (y * z) + 1 / (z * x)) :=
by sorry

end inequality_proof_l178_178516


namespace definite_integral_arctan_x_sub_x_over_1_plus_x_sq_l178_178640

open Real

theorem definite_integral_arctan_x_sub_x_over_1_plus_x_sq :
  ∫ x in 0..1, (4 * arctan x - x) / (1 + x^2) = (π^2 - 4 * log 2) / 8 :=
by
  sorry

end definite_integral_arctan_x_sub_x_over_1_plus_x_sq_l178_178640


namespace solve_system_eq_l178_178573

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end solve_system_eq_l178_178573


namespace workshop_probability_l178_178676

noncomputable def probability_workshop_occurs : ℝ :=
  let total_volume : ℝ := 16 in
  let feasible_volume : ℝ := (4/3 - 1/4) in
  feasible_volume / total_volume

theorem workshop_probability :
  probability_workshop_occurs = 13 / 192 :=
by
  -- Provided conditions:
  -- Four individuals arrive randomly between 3:00 and 5:00 p.m.
  -- Each analyst waits for up to 30 minutes for the others before leaving.
  -- To prove:
  -- The probability that the workshop occurs is 13 / 192.
  sorry

end workshop_probability_l178_178676


namespace regression_difference_term_l178_178089

theorem regression_difference_term {data_point : ℝ} {regression_line_point : ℝ} :
  (difference_term data_point regression_line_point = sum_of_squares_of_residuals) :=
sorry

end regression_difference_term_l178_178089


namespace sqrt_expression_domain_l178_178070

theorem sqrt_expression_domain (x : ℝ) : sqrt(2 * x - 3).is_real ↔ x ≥ 3 / 2 := 
sorry

end sqrt_expression_domain_l178_178070


namespace find_natural_numbers_l178_178388

theorem find_natural_numbers :
  ∃ (x y : ℕ), 
    x * y - (x + y) = Nat.gcd x y + Nat.lcm x y ∧ 
    ((x = 6 ∧ y = 3) ∨ (x = 6 ∧ y = 4) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 6)) := 
by 
  sorry

end find_natural_numbers_l178_178388


namespace shooting_game_system_l178_178622

theorem shooting_game_system :
  ∃ x y : ℕ, (x + y = 20 ∧ 3 * x = y) :=
by
  sorry

end shooting_game_system_l178_178622


namespace determine_a_and_b_a_n_increasing_bounded_expression_for_a_n_l178_178614

variable {a b : ℝ}
variable {a_n b_n c_n : ℕ → ℝ}
variable {n : ℕ}

-- Conditions
axiom h1 : ∀ n, a_n = b_n + c_n
axiom h2 : ∀ n, b_{n+1} = a * a_n
axiom h3 : ∀ n, c_{n+1} = a_n + b * a_n^2
axiom h4 : a_n 1 = 1
axiom h5 : a_n 2 = 1.5
axiom h6 : a_n 3 = 1.875
axiom h7 : ∀ n, n > 0 -- n ∈ ℕ^*

-- Prove the values of a and b, the relationship between a_{n+1} and a_n
theorem determine_a_and_b : a = 1 ∧ b = -1/2 ∧ (∀ n, a_{n+1} = 2 * a_n - 1/2 * a_n^2) :=
begin
  sorry
end

-- Prove that a_n increases and is controlled within 2 ten thousand units
theorem a_n_increasing_bounded : ∀ n, a_n < 2 ∧ a_n < a_n n+1 :=
begin
  sorry
end

-- Prove the expression for a_n
theorem expression_for_a_n : ∀ n, a_n = 2 - 2 * (1/2)^(2^(n-1)) :=
begin
  sorry
end

end determine_a_and_b_a_n_increasing_bounded_expression_for_a_n_l178_178614


namespace a_n_geometric_sequence_b_n_general_term_l178_178782

theorem a_n_geometric_sequence (t : ℝ) (h : t ≠ 0 ∧ t ≠ 1) :
  (∀ n, ∃ r : ℝ, a_n = t^n) :=
sorry

theorem b_n_general_term (t : ℝ) (h1 : t ≠ 0 ∧ t ≠ 1) (h2 : ∀ n, a_n = t^n)
  (h3 : ∃ q : ℝ, q = (2 * t^2 + t) / 2) :
  (∀ n, b_n = (t^(n + 1) * (2 * t + 1)^(n - 1)) / 2^(n - 2)) :=
sorry

end a_n_geometric_sequence_b_n_general_term_l178_178782


namespace cos_sin_eq_l178_178012

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end cos_sin_eq_l178_178012


namespace extraneous_root_value_of_m_l178_178465

theorem extraneous_root_value_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (x + 7) / (x - 1) + 2 = (m + 5) / (x - 1)) → (3 = m) :=
by
  intro h
  have h1 : x = 1,
    sorry

end extraneous_root_value_of_m_l178_178465


namespace Q_condition_l178_178738

-- The statement of the problem in Lean 4
noncomputable def polynomial_Q : Polynomial ℝ :=
  -- Define Q(x) as the polynomial x^3 + x^2
  Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2

theorem Q_condition :
  ∃ Q : Polynomial ℝ, Q ≠ 0 ∧ Q(Q) = (Polynomial.X ^ 3 + Polynomial.X ^ 2 + 1) * Q ∧ Q = polynomial_Q :=
by
  sorry

end Q_condition_l178_178738


namespace min_omega_sin_two_max_l178_178737

theorem min_omega_sin_two_max (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ k : ℤ, (ω * x = (2 + 2 * k) * π)) →
  ∃ ω_min : ℝ, ω_min = 4 * π :=
by
  sorry

end min_omega_sin_two_max_l178_178737


namespace right_triangle_has_one_right_angle_l178_178808

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_l178_178808


namespace identity_eq_l178_178940

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l178_178940


namespace measure_angle_BDA_l178_178852

variable (ABCD : Type) (A B C D : Point ABCD)
variable (angle : Point ABCD → Point ABCD → Point ABCD → ℝ)

-- Definitions of angles
axiom angle_DAB : angle D A B = 90
axiom angle_BCD : angle B C D = 90

-- Definition of angle ratio
axiom angle_ratio : angle C D A = 2 * angle A B C

-- Definition of length ratio
axiom length_ratio : (dist A D) / (dist C B) = 1 / (sqrt 3)

-- Definition of convex quadrilateral
axiom is_convex : convex_quadrilateral A B C D

-- The theorem to prove
theorem measure_angle_BDA : angle B D A = 60 := by
  sorry

end measure_angle_BDA_l178_178852


namespace minimum_distance_P_to_O_l178_178357

theorem minimum_distance_P_to_O (x : ℝ) (hx : x ≠ 0) : 
  let y := 2 / x,
      PO := Real.sqrt (x^2 + y^2) in
  PO ≥ 2 :=
by
  let y := 2 / x
  let PO := Real.sqrt (x^2 + y^2)
  sorry

end minimum_distance_P_to_O_l178_178357


namespace coffee_prices_purchase_ways_l178_178639

-- Define the cost equations for coffee A and B
def cost_equation1 (x y : ℕ) : Prop := 10 * x + 15 * y = 230
def cost_equation2 (x y : ℕ) : Prop := 25 * x + 25 * y = 450

-- Define what we need to prove for task 1
theorem coffee_prices (x y : ℕ) (h1 : cost_equation1 x y) (h2 : cost_equation2 x y) : x = 8 ∧ y = 10 := 
sorry

-- Define the condition for valid purchases of coffee A and B
def valid_purchase (m n : ℕ) : Prop := 8 * m + 10 * n = 200

-- Prove that there are 4 ways to purchase coffee A and B with 200 yuan
theorem purchase_ways : ∃ several : ℕ, several = 4 ∧ (∃ m n : ℕ, valid_purchase m n) := 
sorry

end coffee_prices_purchase_ways_l178_178639


namespace sum_of_first_n_terms_of_cn_l178_178081

theorem sum_of_first_n_terms_of_cn (a_n b_n : ℕ → ℕ) (q common_ratio_ne_one : ℕ → ℕ → Prop)
  (three_pow : ∀ (n : ℕ), a_n n = 3^n)
  (arithmetic_seq : ∀ (n : ℕ), b_n n = 2 * n + 1)
  (cn : ℕ → ℕ := λ n, a_n n * b_n n) :
  ∀ (n : ℕ), (∑ i in finset.range n.succ, cn i) = n.succ * 3^(n.succ + 1) := sorry

end sum_of_first_n_terms_of_cn_l178_178081


namespace log_equation_solution_l178_178384

theorem log_equation_solution (x : ℝ) (h : log 2 (x + 2) + (log 2 (x^2 - 2) / log 2 4) + (log 2 (x + 2) / log 2 (1/2)) = 5) : x = Real.sqrt 1026 :=
sorry

end log_equation_solution_l178_178384


namespace smallest_nat_ending_in_two_double_shift_l178_178741

theorem smallest_nat_ending_in_two_double_shift :
  ∃ N : ℕ, (N % 10 = 2) ∧ (N = 105263157894736842) ∧ 
           (2 * N = 2 + 10^((nat.length_nat N) - 1 - 1) * (N / 10)) :=
by
  sorry

end smallest_nat_ending_in_two_double_shift_l178_178741


namespace chenny_candies_l178_178355

def friends_count : ℕ := 7
def candies_per_friend : ℕ := 2
def candies_have : ℕ := 10

theorem chenny_candies : 
    (friends_count * candies_per_friend - candies_have) = 4 := by
    sorry

end chenny_candies_l178_178355


namespace conjugate_of_z_l178_178398

variable (i : ℂ) (z : ℂ)

theorem conjugate_of_z 
  (h1 : i * i = -1) 
  (h2 : z = (i / (1 + i))) : 
   conj z = (1 / 2) - (1 / 2) * I := by
  -- proof placeholder
  sorry

end conjugate_of_z_l178_178398


namespace restore_numerical_record_l178_178662

-- Define base-4 system
def base_4_system (n : ℕ) : Prop :=
  ∀ d, d < 4 → d < n

-- Define the problem conditions
def conditions : Prop :=
  (3 + 1 = 4) ∧
  (∃ n : ℕ, n = 10203)

-- Define the numerical restoration property
def restored_record {n : ℕ} [base_4_system n] (record : ℕ) : Prop :=
  record = 10203

-- State the theorem
theorem restore_numerical_record (h : conditions) : ∃ record : ℕ, restored_record record :=
sorry

end restore_numerical_record_l178_178662


namespace graveling_cost_is_correct_l178_178290

noncomputable def graveling_cost (lawn_length lawn_breadth road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_breadth
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_area := road1_area + road2_area - intersection_area
  total_area * cost_per_sqm

theorem graveling_cost_is_correct :
  graveling_cost 80 60 10 2 = 2600 := by
  sorry

end graveling_cost_is_correct_l178_178290


namespace minimum_value_proof_l178_178073

noncomputable def minimum_value (a b : ℝ) (h : 0 < a ∧ 0 < b) : ℝ :=
  1 / (2 * a) + 1 / b

theorem minimum_value_proof (a b : ℝ) (h : 0 < a ∧ 0 < b)
  (line_bisects_circle : a + b = 1) : minimum_value a b h = (3 + 2 * Real.sqrt 2) / 2 := 
by
  sorry

end minimum_value_proof_l178_178073


namespace range_of_b_l178_178035

def f (x : ℝ) : ℝ := 3^x - 1

def g (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem range_of_b (a b : ℝ) (h : f(a) = g(b)) : b ∈ set.Ioo (-∞ : ℝ) 0 ∪ set.Ioo 2 ∞ :=
by
  sorry

end range_of_b_l178_178035


namespace balloon_rearrangements_l178_178716

-- Given conditions
def vowels := ['A', 'O', 'O']
def consonants := ['B', 'L', 'L', 'N']

-- Definitions for arranging the vowels and consonants
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def permutations_with_repetition (n : ℕ) (reps : List ℕ) : ℕ :=
  factorial n / reps.map factorial.prod

-- Problem statement
theorem balloon_rearrangements : permutations_with_repetition 4 [2] * permutations_with_repetition 3 [2] = 36 := by sorry

end balloon_rearrangements_l178_178716


namespace fifth_root_of_161051_l178_178719

theorem fifth_root_of_161051 :
  ∃ x : ℕ, (x ^ 5 = 161051 ∧ x = 11) :=
begin
  have h1 : 161051 = 1 * 10^5 + 5 * 10^4 + 10 * 10^3 + 10 * 10^2 + 5 * 10 + 1,
  { norm_num, },
  have h2 : (10 + 1)^5 = 161051,
  { norm_num, },
  use 11,
  split,
  { exact h2.symm, },
  { norm_num, }
end

end fifth_root_of_161051_l178_178719


namespace groups_of_complete_losers_exists_l178_178301

noncomputable def is_group_of_complete_losers (boys girls : Finset ℕ) (G : Finset (ℕ × ℕ)) (B : Finset ℕ) : Prop :=
  B ≠ ∅ ∧ (∀ g ∈ girls, ∃ b ∈ B, (b, g) ∈ G) ∧ (∃ h ∈ girls, ∀ b ∈ B, (b, h) ∉ G)

noncomputable def exists_k_groups_of_complete_losers (n : ℕ) (k : ℕ) : Prop :=
  ∃ (boys girls : Finset ℕ) (G : Finset (ℕ × ℕ)), 
    boys.card = n ∧ girls.card = n ∧ 
    (Finset.card (Finset.filter (λ B : Finset ℕ, is_group_of_complete_losers boys girls G B ∧ B.card ≤ n) (Finset.powerset boys)) = k)

theorem groups_of_complete_losers_exists (n k : ℕ) (h_bound : 0 ≤ k ∧ k < 2 * n) : exists_k_groups_of_complete_losers n k :=
  sorry -- Proof goes here.

end groups_of_complete_losers_exists_l178_178301


namespace empty_set_subsets_proof_problem_l178_178695

theorem empty_set_subsets_proof_problem :
  (∀ (A : Set), A = ∅ → A.subsets = {∅}) ∧
  (∃ (A : Set), A = ∅ → A.subsets.card = 1) ∧
  (∀ (A : Set), A ≠ ∅ → ∅ ⊂ A) ∧
  (∀ (A : Set), ∅ ⊂ A → A ≠ ∅) →
  (∃ B : Set, (∀ (B : Set), ∅.subsets.card = 1 ∧ ∅ ⊂ B) = 1) :=
sorry

end empty_set_subsets_proof_problem_l178_178695


namespace identity_eq_l178_178943

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l178_178943


namespace multiple_of_regular_rate_is_two_l178_178908

structure WorkDetails where
  regular_hours : ℝ
  regular_rate : ℝ
  total_hours : ℝ
  total_earnings : ℝ

def daily_multiple (w : WorkDetails) : ℝ :=
  let excess_hours := w.total_hours - w.regular_hours
  let regular_earnings := w.regular_hours * w.regular_rate
  let excess_earnings := w.total_earnings - regular_earnings
  excess_earnings / (excess_hours * w.regular_rate)

noncomputable def LloydDetails : WorkDetails := {
  regular_hours := 7.5,
  regular_rate := 4.5,
  total_hours := 10.5,
  total_earnings := 60.75
}

theorem multiple_of_regular_rate_is_two : 
  daily_multiple LloydDetails = 2 := 
by 
  sorry

end multiple_of_regular_rate_is_two_l178_178908


namespace tower_surface_area_l178_178541

def cube_volumes : List ℚ := [512, 343, 216, 125, 64, 27, 8, 1, 0.125]

def side_length (v : ℚ) : ℚ := v^(1/3)

def surface_area (s : ℚ) : ℚ := 6 * s * s

def total_surface_area (volumes : List ℚ) : ℚ :=
  let side_lengths := volumes.map side_length
  let individual_areas := side_lengths.map surface_area
  let adjusted_areas := List.zipWith (λ i area, if i > 0 then area - side_lengths.get! (i-1)^2 else area) (List.range individual_areas.length) individual_areas
  let top_adjusted := (adjusted_areas.get! 8) - (side_lengths.get! 7 / 2)^2
  (adjusted_areas.take (adjusted_areas.length - 1)).sum + top_adjusted

theorem tower_surface_area : total_surface_area cube_volumes = 1085.25 := by
  sorry

end tower_surface_area_l178_178541


namespace math_problem_l178_178773

noncomputable def proof_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  a^2 + 4 * b^2 + 1 / (a * b) ≥ 4

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : proof_problem a b ha hb :=
by
  sorry

end math_problem_l178_178773


namespace squared_expression_is_matching_string_l178_178701

theorem squared_expression_is_matching_string (n : ℕ) (h : n > 0) :
  let a := (10^n - 1) / 9
  let term1 := 4 * a * (9 * a + 2)
  let term2 := 10 * a + 1
  let term3 := 6 * a
  let exp := term1 + term2 - term3
  Nat.sqrt exp = 6 * a + 1 := by
  sorry

end squared_expression_is_matching_string_l178_178701


namespace consecutives_lcm_diff_2009_l178_178491

open Nat

def no_six_consecutive_numbers_satisfy_lcm_diff_eq_2009 : Prop :=
  ∀ n : ℕ,
    let a := n
    let b := n + 1
    let c := n + 2
    let d := n + 3
    let e := n + 4
    let f := n + 5
    lcm a (lcm b c) - lcm d (lcm e f) ≠ 2009

theorem consecutives_lcm_diff_2009 : no_six_consecutive_numbers_satisfy_lcm_diff_eq_2009 :=
  sorry

end consecutives_lcm_diff_2009_l178_178491


namespace problem_statement_l178_178127

noncomputable def f (x p q : ℝ) := (x + p) * (x + q) + 2

theorem problem_statement (p q : ℝ) (hp : 2^p + p + 2 = 0) (hq : log 2 q + q + 2 = 0) (hsum : p + q = -2) :
  f 2 p q = f 0 p q ∧ f 0 p q < f 3 p q :=
by
  sorry

end problem_statement_l178_178127


namespace min_period_of_f_f_monotonically_decreasing_interval_min_value_of_f_in_interval_l178_178789

def f (x : ℝ) := 2 * cos x * (sin x - sqrt 3 * cos x) + sqrt 3

-- Statement for minimum positive period T of f(x)
theorem min_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

-- Statement for interval where f(x) is monotonically decreasing
theorem f_monotonically_decreasing_interval (k : ℤ) :
  ∀ x, k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 → ∀ y, x ≤ y ∧ y ≤ x + π / 6 → f x ≥ f y := sorry

-- Statement for minimum value of f(x) when x ∈ [π/2, π]
theorem min_value_of_f_in_interval :
  ∃ x_min, x_min = 11 * π / 12 ∧ x_min ∈ set.Icc (π / 2) π ∧ f x_min = -2 := sorry

end min_period_of_f_f_monotonically_decreasing_interval_min_value_of_f_in_interval_l178_178789


namespace rectangle_perimeter_80_l178_178680

open Nat

theorem rectangle_perimeter_80 (w h : ℕ) :
  2 * (w + h) = 80 ∧ (w ≥ 2 * h ∨ h ≥ 2 * w) → 
  ∃ (n : ℕ), n = 14 ∧ (Finset.card (Finset.filter (λ p : ℕ × ℕ, 
    2 * (p.1 + p.2) = 80 ∧ (p.1 ≥ 2 * p.2 ∨ p.2 ≥ 2 * p.1))
    (Finset.product (Finset.range 41) (Finset.range 41))) = n) :=
by
  sorry

end rectangle_perimeter_80_l178_178680


namespace P_T_S_collinear_P_K_L_collinear_l178_178150

-- Given conditions
variable (l : Line) (A B P Q R S T K L : Point)

-- Condition statements
axiom A_B_P_on_line_l : A ∈ l ∧ B ∈ l ∧ P ∈ l ∧ A ≠ B ∧ B ≠ P ∧ A ≠ P

axiom line_a : is_perpendicular (line_through A) l
axiom line_b : is_perpendicular (line_through B) l

axiom line_through_P : (line_through P) ≠ l
axiom Q_on_a_and_R_on_b : Q ∈ (line_through A) ∧ R ∈ (line_through B)

axiom line_perp_A_BQ : is_perpendicular (line_through A) (line_through B Q)
axiom L_on_BQ_and_T_on_BR : L ∈ (line_through B Q) ∧ T ∈ (line_through B R)

axiom line_perp_B_AR : is_perpendicular (line_through B) (line_through A R)
axiom K_on_AR_and_S_on_AQ : K ∈ (line_through A R) ∧ S ∈ (line_through A Q)

-- Prove (a): P, T, S collinear
theorem P_T_S_collinear : collinear P T S :=
sorry

-- Prove (b): P, K, L collinear
theorem P_K_L_collinear : collinear P K L :=
sorry

end P_T_S_collinear_P_K_L_collinear_l178_178150


namespace range_of_f_l178_178434

/-- Define the piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else Real.cos x

/-- Prove that the range of f(x) is [-1, ∞) -/
theorem range_of_f : Set.range f = Set.Ici (-1) :=
by sorry

end range_of_f_l178_178434


namespace count_multiples_of_4_l178_178448

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l178_178448


namespace identity_eq_l178_178942

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l178_178942


namespace books_read_in_eight_hours_l178_178562

noncomputable def pages_per_hour : ℕ := 120
noncomputable def pages_per_book : ℕ := 360
noncomputable def total_reading_time : ℕ := 8

theorem books_read_in_eight_hours (h1 : pages_per_hour = 120) 
                                  (h2 : pages_per_book = 360) 
                                  (h3 : total_reading_time = 8) : 
                                  total_reading_time * pages_per_hour / pages_per_book = 2 := 
by sorry

end books_read_in_eight_hours_l178_178562


namespace John_spent_on_car_repairs_l178_178101

theorem John_spent_on_car_repairs
  (monthly_savings : ℕ)
  (months_in_year : ℕ)
  (num_years : ℕ)
  (remaining_amount : ℕ)
  (total_savings : ℕ := monthly_savings * months_in_year * num_years) :
  total_savings - remaining_amount = 400 :=
by
  have : total_savings = 25 * 12 * 2, from rfl,
  have : remaining_amount = 200, from rfl,
  have : 25 * 12 * 2 - 200 = 400, from rfl,
  rw [this, this, this],
  exact rfl

end John_spent_on_car_repairs_l178_178101


namespace polygons_homothetic_l178_178887

noncomputable def center_mass (n : ℕ) (vertices : Fin n → ℝ × ℝ) : ℝ × ℝ :=
  (1 / n : ℝ) • Finset.univ.sum (λ i, vertices i)

noncomputable def center_mass_removed (n : ℕ) (vertices : Fin n → ℝ × ℝ) (i : Fin n) : ℝ × ℝ :=
  (1 / (n - 1 : ℝ)) • (Finset.univ.sum (λ j, if j ≠ i then vertices j else (0,0)))

theorem polygons_homothetic {n : ℕ} (h₁ : 2 ≤ n) 
  (A : Fin n → ℝ × ℝ)
  (M := center_mass n A)
  (M_i := λ i, center_mass_removed n A i):
  ∃ r : ℝ, r = -1 / (n - 1) ∧ ∀ i, (M_i i) = M + r • (A i - M) :=
sorry

end polygons_homothetic_l178_178887


namespace PA_PB_product_l178_178482

noncomputable def point (γ : Type*) := (ℝ × ℝ)
noncomputable def inclination_angle := ℝ

variables (C : point ℝ → Prop) (P : point ℝ)
variables (θ : ℝ) (t : ℝ) (A B : point ℝ)

def circle_C (P : point ℝ) (r : ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = r ^ 2
def line_l (P : point ℝ) (inc_angle : inclination_angle) : point ℝ → ℝ → Prop
  | ⟨x, y⟩, t => x = P.1 + t * real.cos inc_angle ∧ y = P.2 + t * real.sin inc_angle

lemma intersect_length_product :
  circle_C (⟨x, y⟩) 5 →
  line_l (⟨3, 2⟩) (π / 3) (⟨x, y⟩) t →
  ∃ (t1 t2 : ℝ), t1 * t2 = -12 :=
begin
  intro hc, intro hl,
  -- Proof omitted
  sorry
end

theorem PA_PB_product :
  (circle_C (A) 5 ∧ circle_C (B) 5) →
  (line_l (⟨3, 2⟩) (π / 3) A t ∧ line_l (⟨3, 2⟩) (π / 3) B t) →
  ∃ t1 t2 : ℝ, |t1 * t2| = 12 :=
by
  intros hC hL,
  obtain ⟨t1, t2, ht⟩ := intersect_length_product hC hL,
  use [t1, t2],
  rw abs_eq_self.mpr,
  exact ht.symm,
  linarith,
  sorry

end PA_PB_product_l178_178482


namespace largest_prime_factor_of_9999_l178_178618

theorem largest_prime_factor_of_9999 :
  ∃ p, (prime p ∧ p ∣ 9999) ∧ (∀ q, prime q ∧ q ∣ 9999 → q ≤ p) :=
begin
  have factorization : 9999 = 3^2 * 11 * 101, by norm_num,
  have prime_3 : prime 3, by norm_num,
  have prime_11 : prime 11, by norm_num,
  have prime_101 : prime 101, by norm_num,
  use 101,
  split,
  { split,
    { exact prime_101, },
    { rw factorization, exact dvd_mul_left 101 (3^2 * 11), },
  },
  { intros q hq,
    cases hq with hq_prime hq_dvd,
    have hq_dvd_9999 := dvd_trans hq_dvd (dvd_refl 9999),
    rw factorization at hq_dvd_9999,
    repeat {rw gcd_mul_left at hq_dvd_9999 _ _},
    cases hq_dvd_9999 with hq_dvd_3 hq_dvd_11_101,
    { exfalso,
      exact prime.not_dvd_one hq_prime (dvd_of_mul_dvd_mul_right (dec_trivial) hq_dvd_3), },
    cases hq_dvd_11_101 with hq_dvd_11 hq_dvd_101,
    { exfalso,
      exact prime.not_dvd_one hq_prime (dvd_of_mul_dvd_mul_right (dec_trivial) hq_dvd_11), },
    exact le_of_eq (dvd_antisymm hq_dvd_101 (dvd_of_mul_dvd_mul_left (dec_trivial) hq_dvd)),
  },
end

end largest_prime_factor_of_9999_l178_178618


namespace pyramid_volume_l178_178838

def area_SAB : ℝ := 9
def area_SBC : ℝ := 9
def area_SCD : ℝ := 27
def area_SDA : ℝ := 27
def area_ABCD : ℝ := 36
def dihedral_angle_equal := ∀ (α β γ δ: ℝ), α = β ∧ β = γ ∧ γ = δ

theorem pyramid_volume (h_eq_dihedral : dihedral_angle_equal)
  (area_conditions : area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27)
  (area_quadrilateral : area_ABCD = 36) :
  (1 / 3 * area_ABCD * 4.5) = 54 :=
sorry

end pyramid_volume_l178_178838


namespace problem_l178_178769

-- Define the sets A and B
def setA (x : ℝ) : Prop := log 3 x < 1
def setB (x : ℝ) : Prop := (x + 1) / (x - 2) < 0

-- Translate the mathematically equivalent problem:
-- Prove that the intersection of A and B is {x | 0 < x < 2}
theorem problem (x : ℝ) : 
  (setA x ∧ setB x) ↔ (0 < x ∧ x < 2) := 
by {
  sorry
}

end problem_l178_178769


namespace min_max_values_l178_178187

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_values : 
  ∃ (min_val max_val : ℝ), 
  min_val = -3 * Real.pi / 2 ∧ 
  max_val = Real.pi / 2 + 2 ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≥ min_val) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≤ max_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = min_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = max_val) := 
by
  sorry

end min_max_values_l178_178187


namespace clinton_earnings_correct_l178_178575

-- Define the conditions as variables/constants
def num_students_Arlington : ℕ := 8
def days_Arlington : ℕ := 4

def num_students_Bradford : ℕ := 6
def days_Bradford : ℕ := 7

def num_students_Clinton : ℕ := 7
def days_Clinton : ℕ := 8

def total_compensation : ℝ := 1456

noncomputable def total_student_days : ℕ :=
  num_students_Arlington * days_Arlington + num_students_Bradford * days_Bradford + num_students_Clinton * days_Clinton

noncomputable def daily_wage : ℝ :=
  total_compensation / total_student_days

noncomputable def earnings_Clinton : ℝ :=
  daily_wage * (num_students_Clinton * days_Clinton)

theorem clinton_earnings_correct : earnings_Clinton = 627.2 := by 
  sorry

end clinton_earnings_correct_l178_178575


namespace exponent_rule_l178_178241

theorem exponent_rule :
  (7 ^ (-3)) ^ 0 + (7 ^ 0) ^ 4 = 2 :=
by
  sorry

end exponent_rule_l178_178241


namespace IncircleConcurrency_l178_178202

-- Given definitions
variable (A B C M N R S P Q : Point)
variable (t : Line)
hypothesis (h_incircle_touch : touches_incircle A B C M N R)
hypothesis (h_S_arc : arc_contains S M N)
hypothesis (h_tangent : tangent_to_arc t S M N)
hypothesis (h_tangent_meets_NC : meet t (line_through N C) P)
hypothesis (h_tangent_meets_MC : meet t (line_through M C) Q)

-- To prove the concurrency of lines AP, BQ, SR, and MN
theorem IncircleConcurrency :
  concurrent (line_through A P) (line_through B Q) (line_through S R) (line_through M N) :=
  sorry

end IncircleConcurrency_l178_178202


namespace min_max_f_l178_178190

noncomputable def f (x : ℝ) : ℝ := Math.cos x + (x + 1) * Math.sin x + 1

theorem min_max_f :
  ∃ a b : ℝ, a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a) ∧ 
  (∃ y ∈ Set.Icc 0 (2 * Real.pi), f y = a) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ b) ∧ 
  (∃ z ∈ Set.Icc 0 (2 * Real.pi), f z = b) :=
sorry

end min_max_f_l178_178190


namespace subset_inter_union_l178_178518

-- Define sets A, B, and C
variables {A B C : Set}

-- Define the condition to be proven
theorem subset_inter_union (A B C : Set) :
  (B ⊆ A ∧ C ⊆ A) ↔ ((A ∩ B) ∪ (A ∩ C) = B ∪ C) :=
sorry

end subset_inter_union_l178_178518


namespace derivative_f1_derivative_f2_l178_178731

-- Problem 1
def f1 (x : ℝ) : ℝ := (1 / 2) * x^2 - x - 1 / x
theorem derivative_f1 : ∀ x ≠ 0, deriv f1 x = (x^3 - x^2 + 1) / x^2 := by
  sorry

-- Problem 2
def f2 (x : ℝ) : ℝ := exp x + log x + sin x
theorem derivative_f2 : ∀ x > 0, deriv f2 x = exp x + 1 / x + cos x := by
  sorry

end derivative_f1_derivative_f2_l178_178731


namespace calculate_gf3_l178_178116

def f (x : ℕ) : ℕ := x^3 - 1
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem calculate_gf3 : g (f 3) = 2056 := by
  sorry

end calculate_gf3_l178_178116


namespace polynomial_value_at_n_plus_one_l178_178120

theorem polynomial_value_at_n_plus_one (n : ℕ) (P : Polynomial ℝ)
  (h1 : 1 ≤ n)
  (h2 : degree P ≤ n)
  (h3 : ∀ k : ℕ, k ≤ n → P.eval k = k / (k + 1)) :
  P.eval (n + 1) = ((-1 : ℝ) ^ n + n + 1) / (n + 2) :=
by sorry

end polynomial_value_at_n_plus_one_l178_178120


namespace max_value_problem_l178_178736

theorem max_value_problem :
  ∃ x in Set.Icc (-(3*Real.pi/4)) (-(Real.pi/2)), 
    ∀ y ∈ Set.Icc (-(3 * Real.pi / 4)) (-(Real.pi / 2)), 
      tan (y + 3 * Real.pi / 4) - tan y + sin (y + Real.pi / 4) ≤
      tan (x + 3 * Real.pi / 4) - tan x + sin (x + Real.pi / 4) ∧ 
      tan (x + 3 * Real.pi / 4) - tan x + sin (x + Real.pi / 4) = 0 := sorry

end max_value_problem_l178_178736


namespace tylenol_tablet_mg_l178_178098

/-- James takes 2 Tylenol tablets every 6 hours and consumes 3000 mg a day.
    Prove the mg of each Tylenol tablet. -/
theorem tylenol_tablet_mg (t : ℕ) (h1 : t = 2) (h2 : 24 / 6 = 4) (h3 : 3000 / (4 * t) = 375) : t * (4 * t) = 3000 :=
by
  sorry

end tylenol_tablet_mg_l178_178098


namespace max_three_digit_divisible_by_25_in_sequence_l178_178664

def is_arith_progression (a : ℕ → ℕ) : Prop :=
∀ k : ℕ, (k + 2) < (nat.succ n) → a (k + 2) = 3 * a (k + 1) - 2 * a k - 1

def isDivisibleBy25 (n : ℕ) : Prop := n % 25 = 0

theorem max_three_digit_divisible_by_25_in_sequence (a : ℕ → ℕ) (n : ℕ) (h3 : 3 ≤ n) (h_sequence : is_arith_progression a) (h_contains2021 : ∃ k, k < n ∧ a k = 2021) :
  ∃ m, m = 36 ∧ (∀ i, 0 ≤ i ∧ i < n → (100 ≤ a i ∧ a i < 1000) → isDivisibleBy25 (a i)) :=
sorry

end max_three_digit_divisible_by_25_in_sequence_l178_178664


namespace greatest_integer_with_gcf_5_l178_178252

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l178_178252


namespace number_told_to_sasha_l178_178616

-- Defining concepts
def two_digit_number (a b : ℕ) : Prop := a < 10 ∧ b < 10 ∧ a * b ≥ 1

def product_of_digits (a b : ℕ) (P : ℕ) : Prop := P = a * b

def sum_of_digits (a b : ℕ) (S : ℕ) : Prop := S = a + b

def petya_guesses_in_three_attempts (P : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), P = a * b ∧ P = c * d ∧ P = e * f ∧ 
  (a * b) ≠ (c * d) ∧ (a * b) ≠ (e * f) ∧ (c * d) ≠ (e * f)

def sasha_guesses_in_four_attempts (S : ℕ) : Prop :=
  ∃ (a b c d e f g h i j : ℕ), 
  S = a + b ∧ S = c + d ∧ S = e + f ∧ S = g + h ∧ S = i + j ∧
  (a + b) ≠ (c + d) ∧ (a + b) ≠ (e + f) ∧ (a + b) ≠ (g + h) ∧ (a + b) ≠ (i + j) ∧ 
  (c + d) ≠ (e + f) ∧ (c + d) ≠ (g + h) ∧ (c + d) ≠ (i + j) ∧ 
  (e + f) ≠ (g + h) ∧ (e + f) ≠ (i + j) ∧ 
  (g + h) ≠ (i + j)

theorem number_told_to_sasha : ∃ (S : ℕ), 
  ∀ (a b : ℕ), two_digit_number a b → 
  (product_of_digits a b (a * b) → petya_guesses_in_three_attempts (a * b)) → 
  (sum_of_digits a b S → sasha_guesses_in_four_attempts S) → S = 10 :=
by
  sorry

end number_told_to_sasha_l178_178616


namespace smallest_possible_t_l178_178221

noncomputable def smallest_t : ℤ → ℤ → Prop :=
λ a b, ∃ (t : ℤ), (t > a - b) ∧ (t < a + b) ∧ (t ∈ ℤ)

theorem smallest_possible_t : ∃ (t : ℤ), smallest_t 7.8 5.1 t ∧ t = 3 :=
by
  sorry

end smallest_possible_t_l178_178221


namespace part_a_part_b_part_c_l178_178317

noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else nat.choose n k

def fly_probability (right_steps up_steps total_steps : ℕ) : ℚ :=
  (binom total_steps right_steps) / (2 ^ total_steps : ℚ)

theorem part_a :
  fly_probability 8 10 18 = (binom 18 8) / (2 ^ 18 : ℚ) :=
sorry

theorem part_b :
  fly_probability 8 10 18 * fly_probability 5 6 11 * fly_probability 2 6 6
  = (binom 11 5 * binom 6 2) / (2 ^ 18 : ℚ) :=
sorry

theorem part_c :
  fly_probability 8 10 18 = (2 * binom 9 2 * binom 9 6 + 2 * binom 9 3 * binom 9 5 + binom 9 4 * binom 9 4) / (2 ^ 18 : ℚ) :=
sorry

end part_a_part_b_part_c_l178_178317


namespace goldfish_sold_l178_178214

/-- The pet store buys a goldfish for $0.25 and sells it for $0.75.
    The new tank costs $100 and the owner is 45% short of the price. 
    Prove that the owner sold 110 goldfish that week. -/
theorem goldfish_sold (buy_price sell_price tank_cost shortfall_percentage : ℝ)
  (h_buy : buy_price = 0.25)
  (h_sell : sell_price = 0.75)
  (h_tank : tank_cost = 100)
  (h_shortfall : shortfall_percentage = 0.45) :
  ∃ (n : ℕ), n = 110 :=
begin
  sorry
end

end goldfish_sold_l178_178214


namespace equation_of_line_given_conditions_l178_178071

-- Define conditions
def angle_of_inclination (θ : ℝ) : Prop := θ = 135
def line_passes_through (p : ℝ × ℝ) (A : ℝ × ℝ) : Prop := p = (1, 1)

-- Define the slope of the line when the angle of inclination is given
noncomputable def slope_of_line (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

-- Define the equation of the line
def equation_of_line (m : ℝ) (A : ℝ × ℝ) : ℝ → ℝ := λ x, m * (x - A.1) + A.2

-- Problem statement
theorem equation_of_line_given_conditions :
  angle_of_inclination 135 →
  line_passes_through (1, 1) →
  (equation_of_line (slope_of_line 135) (1, 1) = λ x, -x + 2) :=
begin
  intros h1 h2,
  sorry
end

end equation_of_line_given_conditions_l178_178071


namespace value_of_g_at_5_l178_178821

def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem value_of_g_at_5 : g 5 = 17 / 3 := by
  sorry

end value_of_g_at_5_l178_178821


namespace value_of_expression_l178_178226

theorem value_of_expression : 2 * 2015 - 2015 = 2015 :=
by
  sorry

end value_of_expression_l178_178226


namespace total_profit_proof_l178_178689

variable (A B C : ℕ) (x : ℕ) (total_profit : ℕ)

-- A's investment is 3 times B's investment
axiom a_investment : A = 3 * B
-- B's investment is two-thirds of C's investment
axiom b_investment : B = (2 * C) / 3
-- B's share of the profit is Rs. 800
axiom b_share : B's_share = 800
-- The total profit is calculated from the ratio A:B:C = 6:2:3
axiom ratio : A : B : C = 6 : 2 : 3

theorem total_profit_proof : total_profit = 4400 := by
  sorry

end total_profit_proof_l178_178689


namespace jake_present_weight_l178_178822

theorem jake_present_weight (J S B : ℝ) (h1 : J - 20 = 2 * S) (h2 : B = 0.5 * J) (h3 : J + S + B = 330) :
  J = 170 :=
by sorry

end jake_present_weight_l178_178822


namespace problem_l178_178553

theorem problem (z x : ℝ) (h1 : ∀ x, y = 2^(x^2)) (h2 : ∀ y, z = 2^(y^2)) :
  (x = ± √(0.5 * log 2 (log 2 z))) ∧ (z ≥ 2) :=
by
  sorry

end problem_l178_178553


namespace sarah_copies_total_pages_l178_178960

noncomputable def total_pages_copied (people : ℕ) (pages_first : ℕ) (copies_first : ℕ) (pages_second : ℕ) (copies_second : ℕ) : ℕ :=
  (pages_first * (copies_first * people)) + (pages_second * (copies_second * people))

theorem sarah_copies_total_pages :
  total_pages_copied 20 30 3 45 2 = 3600 := by
  sorry

end sarah_copies_total_pages_l178_178960


namespace total_hours_watched_l178_178649

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end total_hours_watched_l178_178649


namespace sum_a_b_l178_178818

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) : a + b = 176 / 57 :=
by
  sorry

end sum_a_b_l178_178818


namespace trigonometric_identity_l178_178771

open Real

theorem trigonometric_identity 
  (θ : ℝ)
  (h1 : θ ∈ Ioo (3 * π / 4) π)
  (h2 : sin θ * cos θ = -sqrt 3 / 2) : 
  cos θ - sin θ = -sqrt (1 + sqrt 3) := 
by
  sorry

end trigonometric_identity_l178_178771


namespace right_triangle_right_angles_l178_178812

theorem right_triangle_right_angles (T : Triangle) (h1 : T.is_right_triangle) :
  T.num_right_angles = 1 :=
sorry

end right_triangle_right_angles_l178_178812


namespace min_max_values_f_l178_178194

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l178_178194


namespace cone_lateral_surface_area_l178_178780

theorem cone_lateral_surface_area 
  (r : ℝ) (l : ℝ) (L : ℝ) 
  (h_r : r = 3) (h_l : l = 4) : L = 12 * Real.pi :=
by 
  sorry

end cone_lateral_surface_area_l178_178780


namespace inequality_for_n_l178_178542

theorem inequality_for_n (n : ℕ) (h : n ≥ 2) : 
  (1 + ∑ k in finset.range (n - 1), (1 / (k+2)^2 : ℝ)) < (2 * n - 1) / n := 
by
  sorry

end inequality_for_n_l178_178542


namespace event_probability_l178_178678

noncomputable def probability_event : ℝ :=
  let a : ℝ := (1 : ℝ) / 2
  let b : ℝ := (3 : ℝ) / 2
  let interval_length : ℝ := 2
  (b - a) / interval_length

theorem event_probability :
  probability_event = (3 : ℝ) / 4 :=
by
  -- Proof step will be supplied here
  sorry

end event_probability_l178_178678


namespace find_fourth_vertex_of_square_l178_178603

noncomputable def fourth_vertex_of_square (P Q R : ℂ) : ℂ :=
-(Q)

theorem find_fourth_vertex_of_square :
  ∃ S : ℂ, (P = 2 + 3 * Complex.i) ∧ (Q = 1 - 2 * Complex.i) ∧ (R = -2 - 3 * Complex.i) ∧ (S = -1 + 2 * Complex.i) :=
begin
  let P := 2 + 3 * Complex.i,
  let Q := 1 - 2 * Complex.i,
  let R := -2 - 3 * Complex.i,
  let S := fourth_vertex_of_square P Q R,
  use S,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  exact rfl,
end

end find_fourth_vertex_of_square_l178_178603


namespace stickers_distribution_l178_178928

-- Definitions for initial sticker quantities and stickers given to first four friends
def initial_space_stickers : ℕ := 120
def initial_cat_stickers : ℕ := 80
def initial_dinosaur_stickers : ℕ := 150
def initial_superhero_stickers : ℕ := 45

def given_space_stickers : ℕ := 25
def given_cat_stickers : ℕ := 13
def given_dinosaur_stickers : ℕ := 33
def given_superhero_stickers : ℕ := 29

-- Definitions for remaining stickers calculation
def remaining_space_stickers : ℕ := initial_space_stickers - given_space_stickers
def remaining_cat_stickers : ℕ := initial_cat_stickers - given_cat_stickers
def remaining_dinosaur_stickers : ℕ := initial_dinosaur_stickers - given_dinosaur_stickers
def remaining_superhero_stickers : ℕ := initial_superhero_stickers - given_superhero_stickers

def total_remaining_stickers : ℕ := remaining_space_stickers + remaining_cat_stickers + remaining_dinosaur_stickers + remaining_superhero_stickers

-- Definition for number of each type of new sticker
def each_new_type_stickers : ℕ := total_remaining_stickers / 4
def remainder_stickers : ℕ := total_remaining_stickers % 4

-- Statement to be proved
theorem stickers_distribution :
  ∃ X : ℕ, X = 3 ∧ each_new_type_stickers = 73 :=
by
  sorry

end stickers_distribution_l178_178928


namespace length_of_each_piece_l178_178055

theorem length_of_each_piece (total_length : ℝ) (num_pieces : ℕ) (conv_m_to_cm : ℝ) 
  (h1 : total_length = 25.5) (h2 : num_pieces = 30) (h3 : conv_m_to_cm = 100) :
  (total_length / num_pieces) * conv_m_to_cm = 85 := 
begin
  sorry
end

end length_of_each_piece_l178_178055


namespace cyclic_quadratic_parallels_l178_178761

-- Definitions based on conditions
variable (A B C D : Type) [cyclic_quad : Cyclic A B C D]

def point_in_triangle (p : Type) (occupied : A → B → C → p) := sorry

variable (L_a : point_in_triangle B C D) 
variable (L_b : point_in_triangle A C D)
variable (L_c : point_in_triangle A B D)
variable (L_d : point_in_triangle A B C)

noncomputable def distances_proportional_to_sides 
    (p : point_in_triangle A B C) : Prop := sorry

-- Proportional properties given in the conditions
axiom L_a_prop : distances_proportional_to_sides L_a B C D
axiom L_b_prop : distances_proportional_to_sides L_b A C D
axiom L_c_prop : distances_proportional_to_sides L_c A B D
axiom L_d_prop : distances_proportional_to_sides L_d A B C

-- Given that quadrilateral L_a L_b L_c L_d is cyclic
variable [cyclic_L : Cyclic L_a L_b L_c L_d]

-- Desired conclusion
theorem cyclic_quadratic_parallels : have_two_parallel_sides A B C D :=
by
  sorry

end cyclic_quadratic_parallels_l178_178761


namespace fraction_inequality_l178_178554

open Real

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) :=
  sorry

end fraction_inequality_l178_178554


namespace min_digits_fraction_l178_178270

def minDigitsToRightOfDecimal (n : ℕ) : ℕ :=
  -- This represents the minimum number of digits needed to express n / (2^15 * 5^7)
  -- as a decimal.
  -- The actual function body is hypothetical and not implemented here.
  15

theorem min_digits_fraction :
  minDigitsToRightOfDecimal 987654321 = 15 :=
by
  sorry

end min_digits_fraction_l178_178270


namespace odd_function_increasing_function_l178_178896

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 0.5

theorem odd_function (x : ℝ) : f (-x) = -f (x) :=
  by sorry

theorem increasing_function : ∀ x y : ℝ, x < y → f x < f y :=
  by sorry

end odd_function_increasing_function_l178_178896


namespace right_triangles_count_l178_178981

-- Definitions of points and rectangle
variables (A P R B C Q S D : Type) [Point A] [Point P] [Point R] [Point B] [Point C] [Point Q] [Point S] [Point D]

-- Conditions
def is_rectangle (A B C D : Type) : Prop := sorry -- Define the property of being a rectangle
def midpoint (X Y Z : Type) : Prop := sorry -- Define the property of Z being midpoint of X and Y
def cong_rectangles (ABC1 ABC2 ABC3 ABC4 : Type) : Prop := sorry -- Define congruent rectangles

-- Given conditions
axiom rect_ABCD : is_rectangle A B C D
axiom midP : midpoint A B P
axiom midR : midpoint A B R
axiom midQ : midpoint C D Q
axiom midS : midpoint C D S
axiom PQ_RS_cong : cong_rectangles (Segment PQ) (Segment RS) 

-- Prove that there are 24 right triangles formed
theorem right_triangles_count : 
  count_right_triangles {A, P, R, B, C, Q, S, D} = 24 := sorry

end right_triangles_count_l178_178981


namespace frac_eq_l178_178060

theorem frac_eq (x : ℝ) (h : 3 - 9 / x + 6 / x^2 = 0) : 2 / x = 1 ∨ 2 / x = 2 := 
by 
  sorry

end frac_eq_l178_178060


namespace range_of_f_cauchy_inequality_l178_178436

theorem range_of_f (x : ℝ) : 
  f(x) = |x - 1| - |2x + 4| → 
  ∃ y, y ∈ (f '' (Icc (-∞ : ℝ) 3)) := sorry

theorem cauchy_inequality (x y z : ℝ) (h : x + y + z = 3) : 
  x > 0 → y > 0 → z > 0 → 
  m = max (λ x, |x - 1| - |2x + 4|) ∧ m = 3 → 
  (x + y + z = 3) →
  (y^2 / x + z^2 / y + x^2 / z) ≥ 3 := sorry

end range_of_f_cauchy_inequality_l178_178436


namespace volume_ratio_correct_l178_178153

-- Define the variables related to the problem
variables (a b h : ℝ)
variables (AB BC DD1 KL MN : ℝ)
variables (E F : ℝ)

-- Define the conditions as hypotheses
def midpoint_relation_CF : Prop :=
  E = 1 / 2 * (BC * h) -- Midpoint E of CC1

def midpoint_relation_FD1 : Prop :=
  F = 1 / 2 * h -- Midpoint F of C1D1

def edge_ratios : Prop :=
  AB / BC = 4 / 3 ∧ KL / MN = 2 / 3

-- Define the volume calculations as definitions
def volume_prism (AB BC h : ℝ) : ℝ :=
  4 * a * 3 * a * h -- Volume of the rectangular prism

def volume_pyramid (b: ℝ) : ℝ :=
  (3 * b ^ 3 * Real.sqrt 3) / 4 -- Volume of the triangular pyramid

noncomputable def volume_ratio (a b h : ℝ) : ℝ :=
  volume_prism 4 3 h / volume_pyramid b

-- The main theorem to state the proof
theorem volume_ratio_correct :
  edge_ratios →
  volume_ratio a b h = 25 * Real.sqrt 3 / 16 :=
sorry -- Proof goes here

end volume_ratio_correct_l178_178153


namespace canteen_distance_is_468_l178_178667

noncomputable def girls_camp : ℝ := 450
noncomputable def boys_camp : ℝ := 750
noncomputable def desired_distance (d : ℝ) : Prop :=
  let a := (girls_camp ^ 2 + (boys_camp - d) ^ 2) in
  a = d ^ 2

theorem canteen_distance_is_468.75 : 
  desired_distance 468.75 :=
by
  sorry

end canteen_distance_is_468_l178_178667


namespace goteborg_to_stockholm_distance_l178_178979

/-- 
Given that the distance from Goteborg to Jonkoping on a map is 100 cm 
and the distance from Jonkoping to Stockholm is 150 cm, with a map scale of 1 cm: 20 km,
prove that the total distance from Goteborg to Stockholm passing through Jonkoping is 5000 km.
-/
theorem goteborg_to_stockholm_distance :
  let distance_G_to_J := 100 -- distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- distance from Jonkoping to Stockholm in cm
  let scale := 20 -- scale of the map, 1 cm : 20 km
  distance_G_to_J * scale + distance_J_to_S * scale = 5000 := 
by 
  let distance_G_to_J := 100 -- defining the distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- defining the distance from Jonkoping to Stockholm in cm
  let scale := 20 -- defining the scale of the map, 1 cm : 20 km
  sorry

end goteborg_to_stockholm_distance_l178_178979


namespace tan_A_tan_C_l178_178586

theorem tan_A_tan_C (A B C D H : Type) [triangle A B C] [alt Bd : altitude B D] [orthocenter H A B C]
  (HD : length H D = 8) (HB : length H B = 20) :
  (tan A) * (tan C) = 7 / 2 :=
sorry

end tan_A_tan_C_l178_178586


namespace optimal_meeting_point_is_first_home_l178_178300

open Real

structure GnomeHome where
  home : Point
  speed : ℝ

variable (homes : List GnomeHome)
variable (meetingPoint : Point)
variable (v1 v2 v3 : ℝ) -- speeds of the three gnomes
variable (d1 d2 d3 : ℝ) -- distances from the homes of the gnomes to the meeting point

-- Ensuring homes exist for three gnomes
axiom homes_exist : homes.length = 3

-- Speeds of the gnomes given
axiom speeds_given : 
  (homes.nth 0).get.speed = v1 ∧ 
  (homes.nth 1).get.speed = v2 ∧ 
  (homes.nth 2).get.speed = v3

-- Define the positions
noncomputable def meetAtFirstHome (homes : List GnomeHome) : Point := (homes.nth 0).get.home

-- Define the time spent by each gnome to reach the meeting point
def travelTime (home : Point) (speed : ℝ) (meeting : Point) : ℝ :=
  dist home meeting / speed

-- Total travel time
def totalTravelTime (homes : List GnomeHome) (meeting : Point) : ℝ :=
  (list.sum (homes.map (λ ghome, travelTime ghome.home ghome.speed meeting)))

-- Hypotheses relating total travel times at different points
axiom minimize_travel_time :
  ∀ (homes : List GnomeHome) (meeting : Point),
  totalTravelTime homes meeting ≥ totalTravelTime homes (meetAtFirstHome homes)

theorem optimal_meeting_point_is_first_home : meetingPoint = meetAtFirstHome homes :=
by
  apply minimize_travel_time
  sorry

end optimal_meeting_point_is_first_home_l178_178300


namespace replace_stars_with_identity_l178_178954

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l178_178954


namespace collinear_vectors_l178_178804

theorem collinear_vectors :
  ∀ (x : ℝ),
    let a := (4, -2)
    let b := (x, 1)
    4 * 1 - (-2) * x = 0 → x = -2 := 
by
  intro x
  let a := (4, -2)
  let b := (x, 1)
  assume h : 4 * 1 - (-2) * x = 0
  sorry

end collinear_vectors_l178_178804


namespace smallest_positive_period_sum_of_maximum_and_minimum_values_l178_178788

noncomputable def f (x : ℝ) : ℝ := 2 * (sqrt 2) * cos x * sin (x - (π / 4)) + 1

theorem smallest_positive_period :
  ∀ (x : ℝ), f (x + π) = f x := 
  sorry

theorem sum_of_maximum_and_minimum_values :
  ∀ (max_x min_x : ℝ), 
    (π / 12 ≤ max_x ∧ max_x ≤ π / 6) → 
    (π / 12 ≤ min_x ∧ min_x ≤ π / 6) → 
    ((f max_x + f min_x) = 0) := 
  sorry

end smallest_positive_period_sum_of_maximum_and_minimum_values_l178_178788


namespace problem_equivalence_l178_178795

noncomputable def line_param_eq : ℝ × ℝ → Prop :=
λ p, ∃ t : ℝ, p.1 = 1 + 3 * t ∧ p.2 = (3 / 2) * t

noncomputable def curve_c1_polar_eq : ℝ × ℝ → Prop :=
λ p, ∃ θ : ℝ, p.1 = 2 * (sin θ) * cos θ ∧ p.2 = 2 * (sin θ) * sin θ

theorem problem_equivalence :
  (∀ p : ℝ × ℝ, line_param_eq p → p.1 - 2 * p.2 - 1 = 0)
  ∧ (∀ p : ℝ × ℝ, curve_c1_polar_eq p → p.1^2 + p.2^2 - 2 * p.2 = 0)
  ∧ (∀ θ : ℝ, 
      let x := 3 * cos θ, y := 3 + 3 * sin θ in
      ∃ d_max : ℝ, d_max = 3 + (7 * √5) / 5 
      ∧ d_max = |(1 * x) + (-2 * y) + (-1)| / √(1^2 + (-2)^2)) :=
by {
  sorry
}

end problem_equivalence_l178_178795


namespace rational_values_imply_rational_coeffs_l178_178492

theorem rational_values_imply_rational_coeffs
  {P : ℚ[X]} :
  (∀ x : ℚ, P.eval x ∈ ℚ) → (∀ i : ℕ, P.coeff i ∈ ℚ) :=
  by
    sorry

end rational_values_imply_rational_coeffs_l178_178492


namespace number_of_victorious_sets_is_210_l178_178353

def is_victorious (V : Set ℤ) : Prop :=
  V.nonempty ∧ ∃ (P : ℤ[X]), P.eval 0 = 330 ∧ ∀ v ∈ V, P.eval v = 2 * v.natAbs

theorem number_of_victorious_sets_is_210 :
  (∃ (V : Finset ℤ), is_victorious V) →
  (∃ (n : ℕ), n = 210) := by
  sorry

end number_of_victorious_sets_is_210_l178_178353


namespace answered_both_correctly_l178_178461

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (P : Event Ω → ℝ) (μ : ProbabilityMeasure Ω)

def event_a := -- the event that a student answered the first question correctly 
def event_b := -- the event that a student answered the second question correctly 
def event_not_a := event_aᶜ -- the complement of event_a
def event_not_b := event_bᶜ -- the complement of event_b

-- Given conditions
axiom prob_a : P event_a = 0.75
axiom prob_b : P event_b = 0.35
axiom prob_not_ab : P (event_not_a ∩ event_not_b) = 0.20

theorem answered_both_correctly : P (event_a ∩ event_b) = 0.30 := by
  sorry

end answered_both_correctly_l178_178461


namespace sum_first_nine_terms_l178_178103

def sequence (n : ℕ) (x a : ℝ) : ℝ := x^n + n * a

theorem sum_first_nine_terms (x a : ℝ) : 
  ( ∑ n in Finset.range 9, sequence (n + 1) x a ) = (x * (x^9 - 1) / (x - 1)) + 45 * a :=
by
  sorry

end sum_first_nine_terms_l178_178103


namespace katie_songs_l178_178511

theorem katie_songs:
  ∀ (initial_songs deleted_songs added_songs final_songs : ℕ),
  initial_songs = 11 → deleted_songs = 7 → added_songs = 24 →
  final_songs = initial_songs - deleted_songs + added_songs →
  final_songs = 28 :=
by
  intros initial_songs deleted_songs added_songs final_songs
  assume h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4


end katie_songs_l178_178511


namespace min_max_f_l178_178199

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f : 
  let min_val := - (3 * Real.pi) / 2 in
  let max_val := (Real.pi / 2) + 2 in
  ∃ x_min ∈ Set.Icc 0 (2 * Real.pi), f x_min = min_val ∧
  ∃ x_max ∈ Set.Icc 0 (2 * Real.pi), f x_max = max_val :=
sorry

end min_max_f_l178_178199


namespace exists_positive_integer_solution_l178_178154

theorem exists_positive_integer_solution (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ n / m = ⌊(n^2 : ℝ)^(1/3)⌋ + ⌊(n : ℝ)^(1/2)⌋ + 1 := 
by
  sorry

end exists_positive_integer_solution_l178_178154


namespace volume_pyramid_correct_l178_178083

noncomputable def volume_pyramid (AB BC CG : ℝ) (M : ℝ × ℝ × ℝ) : ℝ :=
  if AB = 4 ∧ BC = 2 ∧ CG = 3 ∧ M = (2, 4, 1.5) then √20 else 0

theorem volume_pyramid_correct :
  volume_pyramid 4 2 3 (2, 4, 1.5) = √20 :=
by
  -- We skip proving the computations and intermediate steps as requested.
  sorry

end volume_pyramid_correct_l178_178083


namespace group_A_can_form_triangle_l178_178987

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem group_A_can_form_triangle : can_form_triangle 9 6 13 :=
by
  sorry

end group_A_can_form_triangle_l178_178987


namespace find_a1_l178_178890

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
a + (n - 1) * d

theorem find_a1
  (a : ℤ)
  (h1 : ∀ n : ℕ, S n = (10 * a + (n * (2 * n - 11) * d) / 2))
  (h2 : d = -2)
  (h3 : S 10 = S 11) :
  a = 20 :=
by
  sorry

end find_a1_l178_178890


namespace part_a_part_b_l178_178143

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def collinear (A B C : Point ℝ) : Prop :=
∃ (a b c : ℝ), a * A.x + b * A.y = c ∧ a * B.x + b * B.y = c ∧ a * C.x + b * C.y = c

structure Point (ℝ : Type) [LinearOrderedField ℝ] :=
(x y : ℝ)

variables
  (A B P Q R L T K S : Point ℝ)
  (l a b : set (Point ℝ))
  (hA : A ∈ l)
  (hB : B ∈ l)
  (hP : P ∈ l)
  (hA_B : A ≠ B)
  (hB_P : B ≠ P)
  (hA_P : A ≠ P)
  (ha_per : ∀ x, x ∈ a ↔ ∃ y, x = ⟨0, y⟩)
  (hb_per : ∀ x, x ∈ b ↔ ∃ y, x = ⟨1, y⟩)
  (hPQ : Q ∈ a ∧ ¬Q ∈ l ∧ Q ∈ line_through P Q)
  (hPR : R ∈ b ∧ ¬R ∈ l ∧ R ∈ line_through P R)
  (hL : L ∈ line_through A T ∧ L ∈ line_through B Q ∧ line_through A Q = a ∧ line_through B R = b)
  (hT : T ∈ line_through A T ∧ T ∈ line_through A T)
  (hS : S ∈ line_through B S ∧ S ∈ line_through B Q ∧ line_through A R = a ∧ line_through B Q = b)
  (hK : K ∈ line_through A R ∧ K ∈ line_through B S ∧ K ∈ line_through A R)

theorem part_a : collinear ℝ P T S := sorry

theorem part_b : collinear ℝ P K L := sorry

end part_a_part_b_l178_178143


namespace prob1_1_prob1_2_prob2_prob3_prob4_l178_178750

section
open Real

-- Problem 1.1
theorem prob1_1 (a b : ℝ) (hab : a * b = 1) : (1 / (1 + a^2)) + (1 / (1 + b^2)) = 1 :=
by sorry

-- Problem 1.2
theorem prob1_2 (a b : ℝ) (n : ℕ) (hab : a * b = 1) : (1 / (1 + a^n)) + (1 / (1 + b^n)) = 1 :=
by sorry

-- Problem 2
theorem prob2 (a b c : ℝ) (habc : a * b * c = 1) : (5 * a / (ab + a + 1)) + (5 * b / (bc + b + 1)) + (5 * c / (ca + c + 1)) = 5 :=
by sorry

-- Problem 3
theorem prob3 (a b : ℝ) (area : a * b = 9) : 2 * (a + b) ≥ 12 :=
by sorry

-- Problem 4
theorem prob4 (a b : ℝ) (hab : a * b = 1) : 
  (let M := (1 / (1 + a)) + (1 / (1 + 2 * b)) in inf ((λ x, (1 / (1 + x)) + (1 / (1 + 2 / x))) '' {x | x > 0})) = 2 * sqrt 2 - 2 :=
by sorry

end

end prob1_1_prob1_2_prob2_prob3_prob4_l178_178750


namespace bridge_length_approx_l178_178634

noncomputable def speed_in_m_per_s (speed_kmh : ℕ) : ℝ :=
  speed_kmh * 1000 / 3600

noncomputable def distance_covered (speed : ℝ) (time_s : ℕ) : ℝ :=
  speed * time_s

noncomputable def length_of_bridge (distance_covered : ℝ) (length_of_train : ℕ) : ℝ :=
  distance_covered - length_of_train

theorem bridge_length_approx :
  length_of_bridge (distance_covered (speed_in_m_per_s 60) 40) 150 ≈ 516.8 := 
begin
  sorry,
end

end bridge_length_approx_l178_178634


namespace necessary_but_not_sufficient_condition_parallel_lines_l178_178412

variables (a b : line) (M : plane)

-- Necessary condition: b forms equal angles with plane M
-- Not sufficient condition: a parallel b does not necessarily follow if b forms equal angles with M

theorem necessary_but_not_sufficient_condition_parallel_lines :
  (∀ x ∈ M, angle_between b x = angle_between b M) →
  ¬(a ∥ b → ∀ x ∈ M, angle_between b x = angle_between b M) :=
sorry

end necessary_but_not_sufficient_condition_parallel_lines_l178_178412


namespace sqrt_expression_integers_sum_l178_178472

theorem sqrt_expression_integers_sum :
  ∃ a b c : ℤ, 
  (sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c) ∧
  (c > 0) ∧
  (∀ c' : ℤ, (c' > 0 → (∃ a' b' : ℤ, (sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a' * sqrt 6 + b' * sqrt 8) / c') → c ≤ c')) ∧
  (a + b + c = 5) :=
sorry

end sqrt_expression_integers_sum_l178_178472


namespace coeff_x3_in_expansion_l178_178177

theorem coeff_x3_in_expansion :
  (coeff (expand (1 - x^2 + 2 / x)^7) 3) = -910 := 
sorry

end coeff_x3_in_expansion_l178_178177


namespace contrapositive_of_given_condition_l178_178919

-- Definitions
variable (P Q : Prop)

-- Given condition: If Jane answered all questions correctly, she will get a prize
axiom h : P → Q

-- Statement to be proven: If Jane did not get a prize, she answered at least one question incorrectly
theorem contrapositive_of_given_condition : ¬ Q → ¬ P := by
  sorry

end contrapositive_of_given_condition_l178_178919


namespace drum_filled_capacity_l178_178375

theorem drum_filled_capacity (C : ℝ) (h1 : 0 < C) :
    (4 / 5) * C + (1 / 2) * C = (13 / 10) * C :=
by
  sorry

end drum_filled_capacity_l178_178375


namespace goldfish_remaining_to_catch_l178_178533

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end goldfish_remaining_to_catch_l178_178533


namespace cross_product_zero_l178_178061

variables {V : Type*} [InnerProductSpace ℝ V]
variables (v u w : V)
variables h1 : v × w = ⟨3, -1, 2⟩
variables h2 : u × w = ⟨2, 3, -4⟩

theorem cross_product_zero :
  2 * (v + u) × 2 * (v + u) = ⟨0, 0, 0⟩ :=
by
  -- Proof part will go here.
  sorry

end cross_product_zero_l178_178061


namespace possible_values_of_x_l178_178772

theorem possible_values_of_x (a b: ℝ) (ha: a ≠ 0) (hb: b ≠ 0) :
    (let x := (a / |a|) + (b / |b|) + ((a * b) / |a * b|) in x = 3 ∨ x = -1) :=
sorry

end possible_values_of_x_l178_178772


namespace f_composition_l178_178894

def f (x : ℝ) : ℝ :=
if x >= 0 then x + 2 else 1

theorem f_composition : f (f (-1)) = 3 :=
by
  sorry

end f_composition_l178_178894


namespace solve_equation_nat_numbers_l178_178568

theorem solve_equation_nat_numbers (a b c d e f g : ℕ) 
  (h : a * b * c * d * e * f * g = a + b + c + d + e + f + g) : 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 2 ∧ g = 7) ∨ (f = 7 ∧ g = 2))) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 3 ∧ g = 4) ∨ (f = 4 ∧ g = 3))) :=
sorry

end solve_equation_nat_numbers_l178_178568


namespace shortest_distance_RS_l178_178529

-- Definitions of points R and S based on parameters u and v.
def point_R (u: ℝ) : ℝ × ℝ × ℝ := (u + 4, -u + 1, 2u + 3)
def point_S (v: ℝ) : ℝ × ℝ × ℝ := (2v - 1, 3v + 1, -2v + 5)

-- Function to calculate the distance squared between two points in 3D.
def distance_squared (p1 p2: ℝ × ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2

-- The minimum distance RS and its corresponding u and v values.
theorem shortest_distance_RS :
  ∃ u v : ℝ, distance_squared (point_R u) (point_S v) = 468 / 17 := by
  sorry

end shortest_distance_RS_l178_178529


namespace yellow_ball_kids_l178_178635

theorem yellow_ball_kids (total_kids white_ball_kids both_ball_kids : ℕ) :
  total_kids = 35 → white_ball_kids = 26 → both_ball_kids = 19 → 
  (total_kids = white_ball_kids + (total_kids - both_ball_kids)) → 
  (total_kids - (white_ball_kids - both_ball_kids)) = 28 :=
by
  sorry

end yellow_ball_kids_l178_178635


namespace solve_inequality_cos_tan_l178_178161

theorem solve_inequality_cos_tan (x n : ℝ) (k : ℤ) :
  (∀ x, \* (cos(2*x) ^ 2) / (cos(x) ^ 2) >= 3 * tan(x)) ↔ 
  (x ∈ (Icc (-7 * π / 12 + ↑k * π) (-π / 2 + ↑k * π) ∪ Icc (-π / 2 + ↑k * π) (π / 12 + ↑k * π)))
:= sorry

end solve_inequality_cos_tan_l178_178161


namespace solid_is_cuboid_if_views_are_rectangles_l178_178335

theorem solid_is_cuboid_if_views_are_rectangles
    (front_view : Π (s : Type), (s → Prop) → Prop)
    (top_view : Π (s : Type), (s → Prop) → Prop)
    (side_view : Π (s : Type), (s → Prop) → Prop)
    (is_rectangle : (s : Type) → Prop)
    (h1 : front_view s is_rectangle)
    (h2 : top_view s is_rectangle)
    (h3 : side_view s is_rectangle)
    : s = cuboid :=
by
  sorry

end solid_is_cuboid_if_views_are_rectangles_l178_178335


namespace circumscribed_circle_radius_l178_178481

theorem circumscribed_circle_radius 
  (ABC : Triangle)
  (acute_angled : ∀ A B C : Point, is_acute (angle A B C))
  (angle_ACB : angle C A B = 75)
  (height_CD : height C A B D = 1)
  (perimeter_ABC : perimeter A B C = 4 + √6 - √2) :
  circumscribed_radius A B C = √6 - √2 := 
sorry

end circumscribed_circle_radius_l178_178481


namespace right_triangle_has_one_right_angle_l178_178810

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_l178_178810


namespace slope_and_angle_l178_178683

-- Defining the points A and B
def point_A : ℝ × ℝ := (-2, 3)
def point_B : ℝ × ℝ := (-1, 2)

-- Defining the slope function
def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

-- The correct angle alpha for the given slope
def correct_angle (k : ℝ) : ℝ :=
  if k = -1 then 3 * Real.pi / 4 else 0 -- This function only checks for k=-1

-- Now we state the theorem
theorem slope_and_angle :
  slope point_A point_B = -1 ∧ correct_angle (slope point_A point_B) = 3 * Real.pi / 4 :=
by
  sorry

end slope_and_angle_l178_178683


namespace inscribed_circle_radius_l178_178408

theorem inscribed_circle_radius (r : ℝ) (α γ : ℝ)
  (h1 : Real.tan α = 1 / 3) 
  (h2 : Real.sin α * Real.sin γ = 1 / Real.sqrt 10) : 
  let ρ := (2 * Real.sqrt 10 - 5) * r / 5 in
  ρ = (2 * Real.sqrt 10 - 5) * r / 5 := by 
sorry   

end inscribed_circle_radius_l178_178408


namespace right_triangle_has_one_right_angle_l178_178809

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_l178_178809


namespace sum_of_diagonals_bound_l178_178557

-- Define the concept of a convex pentagon
structure ConvexPentagon (V : Type) :=
  (A B C D E : V)
  (is_triangle_inequality_ab_eb : ∀ (a b c : V), a ≠ b → b ≠ c → c ≠ a → a + b > c)

-- Define what the problem is asking for in terms of inequalities
theorem sum_of_diagonals_bound {V : Type} [OrderedAddCommGroup V] (p : ConvexPentagon V)
  (AB BC CD DE EA AC BD CE DA BE : V) :
  (AB + BC + CD + DE + EA) < (AC + BD + CE + DA + BE) ∧
  (AC + BD + CE + DA + BE) < 2 * (AB + BC + CD + DE + EA) :=
sorry

end sum_of_diagonals_bound_l178_178557


namespace goldfish_sold_l178_178216

def buy_price : ℕ → ℝ := λ n, 0.25 * n
def sell_price : ℕ → ℝ := λ n, 0.75 * n
def tank_cost : ℝ := 100
def percentage_short : ℝ := 0.45
def percentage_achieved : ℝ := 1 - percentage_short

theorem goldfish_sold (n : ℕ) (h_buy : buy_price n = 0.25 * n)
  (h_sell : sell_price n = 0.75 * n)
  (h_tank : tank_cost = 100)
  (h_percentage : percentage_short = 0.45) :
  percentage_achieved * tank_cost / (sell_price 1 - buy_price 1) = 110 :=
by
  sorry

end goldfish_sold_l178_178216


namespace number_of_elements_power_of_2_l178_178316

def is_valid_set (S : Finset ℕ) : Prop :=
  ∀ s ∈ S, ∀ d ∈ (Finset.filter (λ k => k ∣ s) (Finset.range (s+1))), 
  ∃! t ∈ S, Nat.gcd s t = d

theorem number_of_elements_power_of_2 (S : Finset ℕ) (h : is_valid_set S) : 
  ∃ k : ℕ, S.card = 2^k :=
sorry

end number_of_elements_power_of_2_l178_178316


namespace polygon_side_inequality_l178_178646

theorem polygon_side_inequality (n : ℕ) (hn : 0 < n) :
  ∃ s : ℝ, ∃ (polygon : ℕ → ℝ × ℝ),
    (∀ i, dist (polygon i) (polygon (i + 1)) = 1) ∧
    (polygon 0 = (0, 0)) ∧
    let regular_polygon := λ i, (real.cos (2 * real.pi * i / 2 / n), real.sin (2 * real.pi * i / 2 / n)) in
    (let side_length := dist (regular_polygon 0) (regular_polygon 1) in
      s ≥ side_length) := 
sorry

end polygon_side_inequality_l178_178646


namespace points_on_horizontal_line_l178_178391

theorem points_on_horizontal_line (u : ℝ) : 
  let x := 2 * (Real.cosh u - Real.sinh u),
      y := 4 * (Real.cosh u + Real.sinh u)
  in y^2 = 64 :=
by
  sorry

end points_on_horizontal_line_l178_178391


namespace smallest_n_for_4050_solutions_l178_178892

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

noncomputable def f (x : ℝ) : ℝ := |3 * (fractional_part x) - 1.5|

theorem smallest_n_for_4050_solutions :
  ∃ n : ℕ, (∀ x : ℝ, x ∈ set.Icc 0 (1.5 * (n : ℝ)) → nf(x * f(x)) = x) → n = 540 :=
sorry

end smallest_n_for_4050_solutions_l178_178892


namespace greatest_integer_with_gcf_5_l178_178251

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l178_178251


namespace range_of_a_l178_178828

variable (a x : ℝ)

theorem range_of_a (h : ax > 2) (h_transform: ax > 2 → x < 2/a) : a < 0 :=
sorry

end range_of_a_l178_178828


namespace largest_k_for_18k_dividing_30_factorial_l178_178735

theorem largest_k_for_18k_dividing_30_factorial :
  ∃ (k : ℤ), 18^k ∣ (30.factorial) ∧ ∀ (m : ℤ), (18^m ∣ (30.factorial)) → m ≤ 7 := by
sorry

end largest_k_for_18k_dividing_30_factorial_l178_178735


namespace floor_smallest_x_equals_13_l178_178524
noncomputable def smallest_x : ℝ :=
  let k := 1 in
  let x := (1 + Real.sqrt(1 + 720 * k)) / 2 in
  x

theorem floor_smallest_x_equals_13
  (hx : ∃ (k : ℤ), ∀ x > 2, x = (1 + Real.sqrt(1 + 720 * k)) / 2) :
  ⌊smallest_x⌋ = 13 :=
  sorry

end floor_smallest_x_equals_13_l178_178524


namespace arithmetic_square_root_of_9_is_3_l178_178170

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l178_178170


namespace greatest_integer_with_gcf_5_l178_178249

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l178_178249


namespace gcd_polynomial_multiple_l178_178017

theorem gcd_polynomial_multiple (b : ℤ) (h : b % 2373 = 0) : Int.gcd (b^2 + 13 * b + 40) (b + 5) = 5 :=
by
  sorry

end gcd_polynomial_multiple_l178_178017


namespace function_solution_l178_178729

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))

theorem function_solution (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x + 2 * f ((x - 1) / x) = 3 * x) →
  (∃ c : ℝ, ∀ x : ℝ, f x = if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))) :=
by
  intro h
  use (f 0)
  intro x
  split_ifs with h0 h1
  rotate_left -- to handle the cases x ≠ 0, 1 at first.
  sorry -- Additional proof steps required here.
  sorry -- Use the given conditions and functional equation to conclude f(0) = c.
  sorry -- Use the given conditions and functional equation to conclude f(1) = 3 - 2c.

end function_solution_l178_178729


namespace number_of_pairs_l178_178450

theorem number_of_pairs : ∃ n : ℕ, n = 9 ∧ ∀ (x y : ℕ), (x * y = 36) → 
  ∃ m : ℕ, m = 
  if (x, y) = (1, 36) ∨ (x, y) = (2, 18) ∨ (x, y) = (3, 12) ∨ (x, y) = (4, 9) ∨ 
    (x, y) = (6, 6) ∨ (x, y) = (9, 4) ∨ (x, y) = (12, 3) ∨ (x, y) = (18, 2) ∨ (x, y) = (36, 1)
  then n else 0 :=
begin
  sorry
end

end number_of_pairs_l178_178450


namespace square_divisor_probability_of_15_l178_178352

theorem square_divisor_probability_of_15! : 
  let n := 15!
  let prime_factorization := 
    (2, 11) :: (3, 6) :: (5, 3) :: (7, 1) :: (11, 1) :: (13, 1) :: []
  let total_divisors := (11+1) * (6+1) * (3+1) * (1+1) * (1+1) * (1+1)
  let square_divisors := 
    (6 * 4 * 2 * 1 * 1 * 1)
  in 
  (square_divisors / total_divisors) = (1 / 28) :=
by
  sorry

end square_divisor_probability_of_15_l178_178352


namespace reasoning_correct_statements_l178_178284

def InductiveReasoning : Prop := ∀ {p q : Prop}, (p → q) → p → q
def DeductiveReasoning : Prop := ∀ {p q : Prop}, (p → q) → q → p → q
def AnalogicalReasoning : Prop := ∀ {p q r : Prop}, (p → q) → (q → r) → p → r

theorem reasoning_correct_statements
  (h1 : InductiveReasoning)
  (h2 : DeductiveReasoning)
  (h3 : AnalogicalReasoning)
  (statement1 : InductiveReasoning)
  (statement2 : DeductiveReasoning)
  (statement3 : AnalogicalReasoning)
  : statement1 ∧ statement2 ∧ statement3 = true :=
sorry

end reasoning_correct_statements_l178_178284


namespace correct_reasoning_methods_l178_178285

-- Definitions based on conditions
def reasoning_1 : String := "Inductive reasoning"
def reasoning_2 : String := "Deductive reasoning"
def reasoning_3 : String := "Analogical reasoning"

-- Proposition stating that the correct answer is D
theorem correct_reasoning_methods :
  (reasoning_1 = "Inductive reasoning") ∧
  (reasoning_2 = "Deductive reasoning") ∧
  (reasoning_3 = "Analogical reasoning") ↔
  (choice = "D") :=
by sorry

end correct_reasoning_methods_l178_178285


namespace combined_bus_rides_length_l178_178617

theorem combined_bus_rides_length :
  let v := 0.62
  let z := 0.5
  let a := 0.72
  v + z + a = 1.84 :=
by
  let v := 0.62
  let z := 0.5
  let a := 0.72
  show v + z + a = 1.84
  sorry

end combined_bus_rides_length_l178_178617


namespace rectangle_perimeter_l178_178632

variables (L W : ℕ)

-- conditions
def conditions : Prop :=
  L - 4 = W + 3 ∧
  (L - 4) * (W + 3) = L * W

-- prove the solution
theorem rectangle_perimeter (h : conditions L W) : 2 * L + 2 * W = 50 := sorry

end rectangle_perimeter_l178_178632


namespace find_speed_first_hour_l178_178996

-- Defining the conditions
def speed_second_hour : ℕ := 60
def average_speed : ℕ := 90
def total_time : ℕ := 2

-- The theorem that needs to be proved
theorem find_speed_first_hour : ∃ (x : ℕ), (average_speed * total_time = x + speed_second_hour) ∧ x = 120 := 
by 
  have x := 120 
  use x 
  split
  {
    rw [←nat.mul_eq_mul_right_iff, nat.add_sub_assoc, nat.add_comm],
    exact rfl,
    exact nat.le_of_lt add_comm
  }
  {
    exact rfl
  }

end find_speed_first_hour_l178_178996


namespace map_line_total_length_l178_178543

theorem map_line_total_length (scale_A_cm: ℕ) (scale_A_km: ℕ) (scale_B_cm: ℕ) (scale_B_km: ℕ) (total_distance_km: ℕ) (region_A_cm: ℕ) :
(scale_A_cm = 7) → (scale_A_km = 35) → (scale_B_cm = 9) → (scale_B_km = 45) → (total_distance_km = 245) → (region_A_cm = 15) →
(region_A_cm * (scale_A_km / scale_A_cm) + (total_distance_km - region_A_cm * (scale_A_km / scale_A_cm)) / (scale_B_km / scale_B_cm) = 49) :=
by {
  intros h1 h2 h3 h4 h5 h6,
  sorry
}

end map_line_total_length_l178_178543


namespace find_angle_B_plus_D_l178_178417

variables (A B D F G : Type*)
variables [DecidableEq A] [DecidableEq B] [DecidableEq D] [DecidableEq F] [DecidableEq G]
variables (angle : A → ℝ)

-- Given conditions
def angle_A : Prop := angle A = 30
def is_isosceles_triangle_AFG : Prop := angle F = angle G
def angle_AFG_plus_BFD : Prop := angle F + angle D = 180
def F_midpoint_BD : Prop := true -- Assuming F is given as the midpoint

-- Goal
theorem find_angle_B_plus_D : angle_A A angle → is_isosceles_triangle_AFG F G angle 
→ angle_AFG_plus_BFD F D angle → F_midpoint_BD F B D → angle B + angle D = 75 :=
by
  sorry

end find_angle_B_plus_D_l178_178417


namespace domain_of_function_l178_178428

theorem domain_of_function {f : ℝ → ℝ} (h : ∀ y, 1 ≤ y ∧ y ≤ 2 ↔ ∃ x, f x = y) : 
  ∀ x, (2/3) ≤ x ∧ x ≤ 1 ↔ ∃ y, f x = y :=
by
  let f := λ x, (x + 1) / (3 - 2 * x)
  have domain_correct : (2/3) ≤ x ∧ x ≤ 1 ↔ (1 ≤ f x ∧ f x ≤ 2),
  sorry

end domain_of_function_l178_178428


namespace symmetric_point_condition_l178_178003

theorem symmetric_point_condition (a b : ℝ) (l : ℝ → ℝ → Prop) 
  (H_line: ∀ x y, l x y ↔ x + y + 1 = 0)
  (H_symmetric: l a b ∧ l (2*(-a-1) + a) (2*(-b-1) + b))
  : a + b = -1 :=
by 
  sorry

end symmetric_point_condition_l178_178003


namespace find_value_of_fraction_l178_178074

noncomputable def triangleABC (A B C O E F H M N : Type) :=
  ∃ (angleA : ℝ) (AB AC : ℝ), 
  angleA = 60 ∧ AB > AC ∧
  (circumcenter A B C O) ∧
  (altitude B A C E H) ∧
  (altitude C A B F H) ∧
  (on_segment M B H) ∧
  (on_segment N H F) ∧
  (BM = CN ∧ BM = BM)

theorem find_value_of_fraction
  (A B C O E F H M N : Type)
  (h : triangleABC A B C O E F H M N) :
  ∀ (MH NH OH : ℝ), MH + NH = sqrt 3 * OH := 
sorry

end find_value_of_fraction_l178_178074


namespace nylon_cord_length_approx_l178_178288

-- Defining the initial conditions
def tree := sorry  -- placeholder for the object tree
def dog := sorry   -- placeholder for the object dog
def cord (r : ℝ) := sorry  -- placeholder for the cord tied to the tree with length r

-- Path as a semicircle
def semicircle_arc_length (r : ℝ) : ℝ := π * r

-- Condition: the dog runs approximately 30 feet along the semicircle
def dog_runs_30_feet {r : ℝ} : Prop := semicircle_arc_length r ≈ 30

-- The approximate length of the cord is r
noncomputable def w_approx (r : ℝ) : ℝ := r

-- Using π approximation
noncomputable def pi_approx : ℝ := 3.14

theorem nylon_cord_length_approx (r : ℝ) (h1 : dog_runs_30_feet) : w_approx r ≈ 9.55 :=
by
  sorry

end nylon_cord_length_approx_l178_178288


namespace converse_proposition_true_l178_178644

-- Definitions based on the given conditions
def given_proposition : Prop :=
  ∀ (P : Type) [plane : Plane P] (l1 l2 : Line P),
  is_perpendicular (projection l1 P) l2 → is_perpendicular l1 l2

def converse_proposition : Prop :=
  ∀ (P : Type) [plane : Plane P] (l1 l2 : Line P),
  is_perpendicular l1 l2 → is_perpendicular (projection l1 P) l2

-- The proof problem
theorem converse_proposition_true :
  converse_proposition :=
by
  sorry

end converse_proposition_true_l178_178644


namespace volume_relation_l178_178839

variable {x y z V : ℝ}

theorem volume_relation
  (top_area : x * y = A)
  (side_area : y * z = B)
  (volume : x * y * z = V) :
  (y * z) * (x * y * z)^2 = z^3 * V := by
  sorry

end volume_relation_l178_178839


namespace num_correct_propositions_l178_178338

-- Definitions of propositions
def prop1 (a b : ℝ) : Prop := a < b → a^2 < b^2
def prop2 (a b : ℝ) : Prop := (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0
def prop3 (T1 T2 : Type) [triangle T1] [triangle T2] (area : triangle → ℝ) : Prop :=
  ¬ congruent T1 T2 → area T1 ≠ area T2

-- Main theorem statement
theorem num_correct_propositions : 
  let correct_prop_count := ([prop1, prop2, prop3].filter (λ p, p = true)).length
  correct_prop_count = 1 := sorry

end num_correct_propositions_l178_178338


namespace second_number_is_650_l178_178645

theorem second_number_is_650 (x : ℝ) (h1 : 0.20 * 1600 = 0.20 * x + 190) : x = 650 :=
by sorry

end second_number_is_650_l178_178645


namespace communication_difference_l178_178493

def total_morning_communications := 10 + 12 + 25 + 4
def total_afternoon_communications := 3 + 44 + 15 + 6

theorem communication_difference :
  abs (total_morning_communications - total_afternoon_communications) = 17 := by
  sorry

end communication_difference_l178_178493


namespace intersection_of_A_and_B_l178_178824

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_of_A_and_B : A ∩ B = {2, 6} :=
by
  sorry

end intersection_of_A_and_B_l178_178824


namespace triangle_equality_l178_178882

-- Definitions and variables as per the conditions
variables {A B C D M N P : Type} [EuclideanGeometry A B C D M N P]
variables (BC : line B C) (AB : segment A B) (AC : segment A C) (AD : segment A D) (AP : segment A P)
variables (bisects_AD_BAC : bisects AD (angle A B C))
variables (angle_MDA_eq_angle_ABC : congruent (angle D A M) (angle B A C))
variables (angle_NDA_eq_angle_ACB : congruent (angle D A N) (angle C A B))
variables (intersection_AD_MN_P : intersects at AD MN P)

theorem triangle_equality (h : ∃ (H1 : ∀ triangle ABC, ∀ point D on BC), ∀ (M : line segment, N : line segment, angle MDA = angle ABC, angle NDA = angle ACB), ∀ point P on intersection AD MN) :
  (AD^3 = AB * AC * AP) :=
sorry

end triangle_equality_l178_178882


namespace initial_erasers_count_l178_178913

noncomputable def erasers_lost := 42
noncomputable def erasers_ended_up_with := 53

theorem initial_erasers_count (initial_erasers : ℕ) : 
  initial_erasers_ended_up_with = initial_erasers - erasers_lost → initial_erasers = 95 :=
by
  sorry

end initial_erasers_count_l178_178913


namespace polynomial_value_at_zero_l178_178470

theorem polynomial_value_at_zero :
  ∀ (p q : ℝ), 
  (f : ℝ → ℝ), 
  (f = λ x, x^2 + p * x + q) → 
  (f 3 = 0) → 
  (f 2 = 0) → 
  f 0 = 6 :=
by
  intros p q f hf h3 h2
  sorry

end polynomial_value_at_zero_l178_178470


namespace intersects_negative_half_axis_range_l178_178438

noncomputable def f (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * m * x + 2 * m - 6

theorem intersects_negative_half_axis_range (m : ℝ) :
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) ↔ (∃ x : ℝ, f m x < 0) :=
sorry

end intersects_negative_half_axis_range_l178_178438


namespace evaluate_special_operation_l178_178905

-- Define the operation @
def special_operation (a b : ℕ) : ℚ := (a * b) / (a - b)

-- State the theorem
theorem evaluate_special_operation : special_operation 6 3 = 6 := by
  sorry

end evaluate_special_operation_l178_178905


namespace number_of_columns_is_approx_15_l178_178287

noncomputable def total_plants : ℕ := 52
noncomputable def plants_per_column : ℝ := 3.47
noncomputable def total_columns : ℝ := total_plants / plants_per_column

theorem number_of_columns_is_approx_15 :
  Real.floor (total_columns + 0.5) = 15 :=
sorry

end number_of_columns_is_approx_15_l178_178287


namespace harmonic_series_inequality_l178_178029

theorem harmonic_series_inequality (n : ℕ) : 
  1 + (∑ i in Finset.range (2^(n+1) - 1), (1:ℚ) / (i+2)) > (n + 1) / 2 :=
sorry

end harmonic_series_inequality_l178_178029


namespace profit_ratio_l178_178549

def praveen_initial_capital : ℝ := 3500
def hari_initial_capital : ℝ := 9000.000000000002
def total_months : ℕ := 12
def months_hari_invested : ℕ := total_months - 5

def effective_capital (initial_capital : ℝ) (months : ℕ) : ℝ :=
  initial_capital * months

theorem profit_ratio :
  effective_capital praveen_initial_capital total_months / effective_capital hari_initial_capital months_hari_invested 
  = 2 / 3 :=
by
  sorry

end profit_ratio_l178_178549


namespace equal_pipe_water_flow_l178_178462

theorem equal_pipe_water_flow (A : ℝ) (hA : A ≠ 0) : ∃ (B : ℝ), B = 36 / (A^2) :=
by
  use 36 / (A^2)
  sorry

end equal_pipe_water_flow_l178_178462


namespace Jackie_apples_count_l178_178328

variable (Adam_apples Jackie_apples : ℕ)

-- Conditions
axiom Adam_has_14_apples : Adam_apples = 14
axiom Adam_has_5_more_than_Jackie : Adam_apples = Jackie_apples + 5

-- Theorem to prove
theorem Jackie_apples_count : Jackie_apples = 9 := by
  -- Use the conditions to derive the answer
  sorry

end Jackie_apples_count_l178_178328


namespace car_speed_first_hour_l178_178595

-- Definitions based on the conditions in the problem
noncomputable def speed_second_hour := 30
noncomputable def average_speed := 45
noncomputable def total_time := 2

-- Assertion based on the problem's question and correct answer
theorem car_speed_first_hour: ∃ (x : ℕ), (average_speed * total_time) = (x + speed_second_hour) ∧ x = 60 :=
by
  sorry

end car_speed_first_hour_l178_178595


namespace even_g_g_g_l178_178117

-- Define the problem
variable {α : Type*} [AddGroup α] [HasSmul ℕ α] [HasSmul ℤ α]

-- Define an even function g
def even_function (g : α → α) : Prop := ∀ x, g (-x) = g x

-- Given g is an even function
variable (g : α → α)
hypothesis (h_even_g : even_function g)

-- Prove that g(g(g(x))) is even
theorem even_g_g_g (g_even : even_function g) : even_function (g ∘ g ∘ g) :=
by
  intro x
  calc
    g (g (g (-x))) = g (g (g x)) : by rw [g_even (-x), g_even, g_even]
  -- Simply states that g(g(g(-x))) equals g(g(g(x))) due to the even property
    
#sorries

end even_g_g_g_l178_178117


namespace each_friend_gave_bella_2_roses_l178_178692

-- Define the given conditions
def total_roses_from_parents : ℕ := 2 * 12
def total_roses_bella_received : ℕ := 44
def number_of_dancer_friends : ℕ := 10

-- Define the mathematical goal
def roses_from_each_friend (total_roses_from_parents total_roses_bella_received number_of_dancer_friends : ℕ) : ℕ :=
  (total_roses_bella_received - total_roses_from_parents) / number_of_dancer_friends

-- Prove that each dancer friend gave Bella 2 roses
theorem each_friend_gave_bella_2_roses :
  roses_from_each_friend total_roses_from_parents total_roses_bella_received number_of_dancer_friends = 2 :=
by
  sorry

end each_friend_gave_bella_2_roses_l178_178692


namespace find_y_value_l178_178485

-- Define the angles in Lean
def angle1 (y : ℕ) : ℕ := 6 * y
def angle2 (y : ℕ) : ℕ := 7 * y
def angle3 (y : ℕ) : ℕ := 3 * y
def angle4 (y : ℕ) : ℕ := 2 * y

-- The condition that the sum of the angles is 360
def angles_sum_to_360 (y : ℕ) : Prop :=
  angle1 y + angle2 y + angle3 y + angle4 y = 360

-- The proof problem statement
theorem find_y_value (y : ℕ) (h : angles_sum_to_360 y) : y = 20 :=
sorry

end find_y_value_l178_178485


namespace simplify_expression_l178_178566

theorem simplify_expression (x y : ℝ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 :=
by
  rw [hx, hy]
  -- here we would simplify but leave a hole
  sorry

end simplify_expression_l178_178566


namespace find_m_l178_178829

theorem find_m (x m : ℝ) :
  (2 * x + m) * (x - 3) = 2 * x^2 - 3 * m ∧ 
  (∀ c : ℝ, c * x = 0 → c = 0) → 
  m = 6 :=
by sorry

end find_m_l178_178829


namespace train_probability_theorem_l178_178687

noncomputable def train_problem : Prop :=
  let train_arrival := (0 : ℝ, 60 : ℝ)
  let train_wait := 15
  let john_arrival := (15 : ℝ, 75 : ℝ)
  let overlap_event :=
    ((0 ≤ y ∧ y ≤ 60) ∧
     (15 ≤ x ∧ x ≤ 75) ∧
     (y ≤ x ∧ x ≤ y + train_wait))
  in
  let total_event_area := (60 : ℝ) * (60 : ℝ)
  let overlap_area := 675
  let probability := overlap_area / total_event_area
  probability = (3 / 16)

theorem train_probability_theorem : train_problem :=
by sorry

end train_probability_theorem_l178_178687


namespace problem_given_ellipse_M_problem_max_area_triangle_l178_178009

noncomputable def ellipse_M : ellipse := 
{ center := (0, 0),
  semi_major := 2,
  semi_minor := sqrt 2,
  orientation := 0 }

theorem problem_given_ellipse_M :
  (∃ (e : ellipse), 
    e.center = (0, 0) ∧
    e.semi_major = 2 ∧
    e.semi_minor = sqrt 2 ∧
    point_on_ellipse (1, sqrt 2) e ∧
    e.equation = '" x^2/4 + y^2/2 = 1 "') := 
sorry

noncomputable def max_area_ABC : ℝ := 
sqrt 2

theorem problem_max_area_triangle :
  (∃ (l : line) (points : finite_set point), 
    line_direction_vector l = (1, sqrt 2) ∧
    intersects_ellipse l ellipse_M points ∧
    points_cardinality_eq points 2 ∧
    max_triangle_area any_point (l,-2, ABC) = sqrt 2) := 
sorry

end problem_given_ellipse_M_problem_max_area_triangle_l178_178009


namespace angle_A1_C1_O_eq_30_degrees_l178_178860

-- Assume the definitions of points and angles in a triangle
variables {A B C A1 B1 C1 O : Type} 

-- Assume that A is 120 degrees
axiom angle_A_eq_120_degrees : ∠ A = 120 

-- Given that AA1, BB1, and CC1 are angle bisectors and intersect at O
axiom AA1_is_bisector : bisector A A1
axiom BB1_is_bisector : bisector B B1
axiom CC1_is_bisector : bisector C C1
axiom bisectors_intersect_at_O : (AA1 ∩ BB1 ∩ CC1 = {O})

-- Prove that the angle ∠A1 C1 O = 30°
theorem angle_A1_C1_O_eq_30_degrees : ∠ A1 C1 O = 30 :=
sorry

end angle_A1_C1_O_eq_30_degrees_l178_178860


namespace simplify_expression_l178_178160

-- Defining the original expression
def original_expr (y : ℝ) : ℝ := 3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y)

-- Defining the simplified expression
def simplified_expr (y : ℝ) : ℝ := y^3 - 10 * y^2 + 21 * y + 9

-- The statement to prove
theorem simplify_expression (y : ℝ) : original_expr y = simplified_expr y :=
by sorry

end simplify_expression_l178_178160


namespace other_diagonal_length_l178_178079

-- Define the conditions for the quadrilateral problem.
def area_quad (a b c d : ℝ) (alpha beta gamma delta : ℝ) : ℝ :=
  (1/2) * a * b * sin alpha + (1/2) * c * d * sin beta

def diag_sum (a b c : ℝ) : ℝ := a + b + c

theorem other_diagonal_length (AB AC CD a b c : ℝ)
  (h1 : area_quad a b c AC = 32)
  (h2 : diag_sum AC AB CD = 16) :
  let BD := 8 * sqrt 2 in
  BD = 8 * sqrt 2 :=
sorry

end other_diagonal_length_l178_178079


namespace greatest_integer_with_gcd_30_eq_5_l178_178255

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l178_178255


namespace necessary_but_not_sufficient_l178_178223

-- Define the necessary conditions
variables {a b c d : ℝ}

-- State the main theorem
theorem necessary_but_not_sufficient (h₁ : a > b) (h₂ : c > d) : (a + c > b + d) :=
by
  -- Placeholder for the proof (insufficient as per the context problem statement)
  sorry

end necessary_but_not_sufficient_l178_178223


namespace limit_of_log_alpha_n_l178_178546

noncomputable def alpha_n (n : ℕ) : ℝ :=
  let f : ℝ → ℝ := sorry  -- polynomial of degree n with integer coefficients
  let integral_val := ∫ x in -1..1, (x^n) * f x
  2 / (Lcm (Finset.range (n + 1)).filter (λ k, 0 < k ∧ Even k))

theorem limit_of_log_alpha_n :
  tendsto (λ n : ℕ, (Real.log (alpha_n n) / n)) at_top (𝓝 (-5/3)) := sorry

end limit_of_log_alpha_n_l178_178546


namespace keith_total_cost_correct_l178_178879

noncomputable def total_cost_keith_purchases : Real :=
  let discount_toy := 6.51
  let price_toy := discount_toy / 0.90
  let pet_food := 5.79
  let cage_price := 12.51
  let tax_rate := 0.08
  let cage_tax := cage_price * tax_rate
  let price_with_tax := cage_price + cage_tax
  let water_bottle := 4.99
  let bedding := 7.65
  let discovered_money := 1.0
  let total_cost := discount_toy + pet_food + price_with_tax + water_bottle + bedding
  total_cost - discovered_money

theorem keith_total_cost_correct :
  total_cost_keith_purchases = 37.454 :=
by
  sorry -- Proof of the theorem will go here

end keith_total_cost_correct_l178_178879


namespace goldfish_sold_l178_178217

def buy_price : ℕ → ℝ := λ n, 0.25 * n
def sell_price : ℕ → ℝ := λ n, 0.75 * n
def tank_cost : ℝ := 100
def percentage_short : ℝ := 0.45
def percentage_achieved : ℝ := 1 - percentage_short

theorem goldfish_sold (n : ℕ) (h_buy : buy_price n = 0.25 * n)
  (h_sell : sell_price n = 0.75 * n)
  (h_tank : tank_cost = 100)
  (h_percentage : percentage_short = 0.45) :
  percentage_achieved * tank_cost / (sell_price 1 - buy_price 1) = 110 :=
by
  sorry

end goldfish_sold_l178_178217


namespace cross_pentomino_fitting_l178_178973

def cross_pentomino_area := 5
def number_of_crosses := 9
def chessboard_area := 64

theorem cross_pentomino_fitting:
  (number_of_crosses * cross_pentomino_area) ≤ chessboard_area :=
by
  have total_cross_area := number_of_crosses * cross_pentomino_area
  have chessboard_area := 64
  calc
    total_cross_area = 45 := by norm_num
    _ ≤ chessboard_area := by norm_num
  sorry

end cross_pentomino_fitting_l178_178973


namespace female_officers_on_police_force_l178_178926

theorem female_officers_on_police_force
  (percent_on_duty : ℝ)
  (total_on_duty : ℕ)
  (half_female_on_duty : ℕ)
  (h1 : percent_on_duty = 0.16)
  (h2 : total_on_duty = 160)
  (h3 : half_female_on_duty = total_on_duty / 2)
  (h4 : half_female_on_duty = 80)
  :
  ∃ (total_female_officers : ℕ), total_female_officers = 500 :=
by
  sorry

end female_officers_on_police_force_l178_178926


namespace translate_point_M_l178_178930

theorem translate_point_M :
  ∀ (x y: ℝ), (x, y) = (2, 5) →
    let newX := x - 2 in
    let newY := y - 3 in
    (newX, newY) = (0, 2) :=
by
  intros x y h_xy_eq
  let newX := x - 2
  let newY := y - 3
  sorry

end translate_point_M_l178_178930


namespace replace_stars_with_identity_l178_178957

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l178_178957


namespace parabola_focus_vertex_ratio_l178_178104

theorem parabola_focus_vertex_ratio :
  let P := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, (x - 1)^2 - 3)}
  let V1 := (1, -3)
  let F1 := (1, -2.75)  -- Derived based on standard parabola properties
  let Q := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 1)}
  let V2 := (0, -1)
  let F2 := (0, -0.75)  -- Derived based on standard parabola properties
  let A_B_condition := ∀ A B : ℝ × ℝ, A ∈ P ∧ B ∈ P ∧ (A.1 - 1) * (B.1 - 1) = -1
  let midpoint_AB := ∀ A B : ℝ × ℝ, (A.1 + B.1) / 2
  let locus_AB_midpoint := ∀ M : ℝ × ℝ, (∃ A B : ℝ × ℝ, A ∈ P ∧ B ∈ P ∧ (A.1 - 1) * (B.1 - 1) = -1 ∧ M = ⟨(A.1 + B.1) / 2, ((A.1 - 1)^2 + (B.1 - 1)^2 - 6) / 2⟩)
  in
  (A_B_condition) ∧ locus_AB_midpoint → 
  dist F1 F2 / dist V1 V2 = 1 :=
by 
  sorry

end parabola_focus_vertex_ratio_l178_178104


namespace hyperbola_proof_l178_178423

noncomputable def hyperbola_standard_equation (a b c : ℝ) : Prop :=
  ∃ (a b c : ℝ), a = 2 ∧ b = sqrt 3 ∧ c = sqrt 7 ∧
  (a^2 + b^2 = c^2) ∧
  (c = sqrt 7) ∧
  (a/b = sqrt 3 / 2) ∧
  (1 - x^2 / 4 / (y^2 / 3) = 1)

theorem hyperbola_proof :
  hyperbola_standard_equation (2 : ℝ) (sqrt 3 : ℝ) (sqrt 7 : ℝ) :=
begin
  sorry
end

end hyperbola_proof_l178_178423


namespace greatest_valid_number_l178_178261

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l178_178261


namespace normal_prob_neg1_to_1_l178_178781

noncomputable def normal_distribution_0_sigma (σ : ℝ) : ProbabilityTheory.ProbabilitySpace ℝ :=
ProbabilityTheory.ProbabilitySpace.normal 0 σ^2

theorem normal_prob_neg1_to_1 {σ : ℝ} (hσ : 0 < σ) :
  (ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, x < -1) = 0.2) →
  (ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, -1 < x ∧ x < 1) = 0.6) :=
begin
  intros h1,
  have h2 : ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, x > 1) = 0.2,
  { sorry },  -- skipped proof of symmetry, which relies on properties of the normal distribution
  have h_total : ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, true) = 1 := 
    ProbabilityTheory.ProbabilitySpace.probability_univ (normal_distribution_0_sigma σ),
  sorry  -- skipped proof,
end

end normal_prob_neg1_to_1_l178_178781


namespace sequence_a_formula_sequence_b_formula_sequence_c_sum_formula_l178_178405

-- Sequence a_n and S_n Condition
def sequence_a (n : ℕ) : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

axiom Sn_relation (n : ℕ) : S n = 2 - sequence_a n

-- Part 1: Prove the general formula for sequence a_n
theorem sequence_a_formula (n : ℕ) (h : S n = 2 - sequence_a n) :
  sequence_a n = (1 / 2)^(n - 1) := sorry

-- Sequence b_n condition and given terms
def sequence_b (n : ℕ) : ℕ → ℝ := sorry
axiom b_initial : sequence_b 1 = 1
axiom b_recursive (n : ℕ) : sequence_b (n + 1) = sequence_b n + sequence_a n

-- Part 2: Prove the general formula for sequence b_n
theorem sequence_b_formula (n : ℕ)
  (h1 : sequence_b 1 = 1)
  (h2 : ∀ n, sequence_b (n + 1) = sequence_b n + sequence_a n) :
  sequence_b n = 3 - 2 * (1 / 2)^(n - 1) := sorry

-- Sequence c_n and terms sum T_n
def sequence_c (n : ℕ) : ℕ → ℝ := λ n, n * (3 - sequence_b n)
def T (n : ℕ) : ℕ → ℝ := λ n, ∑ i in (range n), sequence_c (i + 1)

-- Part 3: Prove the sum of the first n terms of sequence c_n
theorem sequence_c_sum_formula (n : ℕ) 
  (hn : ∀ n, sequence_c n = n * (3 - sequence_b n)) :
  T n = 8 - (8 + 4 * n) * (1 / 2^n) := sorry

end sequence_a_formula_sequence_b_formula_sequence_c_sum_formula_l178_178405


namespace fibonacci_growth_l178_178576

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_growth (n a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
(h3 : min (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n) < a / b) 
(h4 : a / b < max (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n)) :
b ≥ fibonacci (n + 1) :=
sorry

end fibonacci_growth_l178_178576


namespace cost_of_coat_eq_l178_178605

-- Define the given conditions
def total_cost : ℕ := 110
def cost_of_shoes : ℕ := 30
def cost_per_jeans : ℕ := 20
def num_of_jeans : ℕ := 2

-- Define the cost calculation for the jeans
def cost_of_jeans : ℕ := num_of_jeans * cost_per_jeans

-- Define the known total cost (shoes and jeans)
def known_total_cost : ℕ := cost_of_shoes + cost_of_jeans

-- Prove James' coat cost
theorem cost_of_coat_eq :
  (total_cost - known_total_cost) = 40 :=
by
  sorry

end cost_of_coat_eq_l178_178605


namespace contrapositive_l178_178579

theorem contrapositive (a b : ℕ) (h : a = 0 → a * b = 0) : a * b ≠ 0 → a ≠ 0 :=
by
sory

end contrapositive_l178_178579


namespace ratio_EK_KF_l178_178152

-- Let's start by laying out the necessary foundations
variables (A B C H D E F G J : Type) [_inst : EuclideanGeometry A B C H D E F G J]

-- Conditions
axiom altitude_BH : ∃ (bh : Line), IsAltitude bh A B C
axiom point_D_on_BH : ∃ (d : Point), IsOnLine d bh
axiom intersect_AD_BC_at_E : ∃ (ad bc : Line) (e : Point), Intersect ad bc e
axiom intersect_CD_AB_at_F : ∃ (cd ab : Line) (f : Point), Intersect cd ab f
axiom projections_G_J_on_AC : ∃ (ac : Line) (g j : Point), Projection f ac g ∧ Projection e ac j
axiom area_HEJ_twice_HFG : ∃ (he hj hg : Triangle), EqualArea he hj 2 * Area hg

-- Ratio to prove
theorem ratio_EK_KF : ∃ (k : Point), SegmentRatio bh fe k = sqrt(2) / 1 := sorry

end ratio_EK_KF_l178_178152


namespace train_B_speed_l178_178532

variable Da Db D v : ℝ

-- Conditions
axiom H1 : Da / Db = 9 / 4
axiom H2 : Da + Db = D
axiom H3 : 60 = 60
axiom H4 : Da = 60 * 9
axiom H5 : Db = v * 4

theorem train_B_speed : v = 60 := by
  sorry

end train_B_speed_l178_178532


namespace max_three_digit_numbers_divisible_by_25_l178_178666

theorem max_three_digit_numbers_divisible_by_25 
  (a : ℕ → ℕ) (n : ℕ)
  (h1 : 3 ≤ n) 
  (h2 : ∀ k, k ≤ n-2 → a (k + 2) = 3 * a (k + 1) - 2 * a k - 1)
  (h3 : ∃ k ≤ n, a k = 2021) 
  : ∃ m, m = 36 ∧ 
  ∀ x, 100 ≤ a x ∧ a x ≤ 999 → a x % 25 = 0 → 
    ∃ b, 1 ≤ b ∧ b ≤ m := 
begin
  sorry
end

end max_three_digit_numbers_divisible_by_25_l178_178666


namespace largest_number_in_systematic_sample_l178_178657

theorem largest_number_in_systematic_sample (n interval first : ℕ) (h1 : n = 400) (h2 : interval = 25) (h3 : first = 8) : 
  ∃ largest, largest = first + (interval * (n / interval - 1)) :=
by 
  use (first + (interval * ((n / interval) - 1)))
  rw [h1, h2, h3]
  norm_num
  sorry

end largest_number_in_systematic_sample_l178_178657


namespace exists_trapezoid_l178_178362

variables (a b : ℝ) (α β : ℝ)
variables [ring ℝ]

theorem exists_trapezoid (a b α β : ℝ) 
  (h1 : α + β < real.pi) 
  (ha : a > 0) 
  (hb : b > 0) : 
  ∃ (A B C D : ℕ → ℝ), -- Vertex positions
    let AB := dist (A) (B), -- Distance AB
    let CD := dist (C) (D), -- Distance CD
    let angleACB := ∠ (A) (C) (B), -- Angle ACB
    let angleDBC := ∠ (D) (B) (C) in
    AB = a ∧ CD = b ∧ angleACB = α ∧ angleDBC = β := by
    sorry

end exists_trapezoid_l178_178362


namespace orthocenter_property_l178_178463

theorem orthocenter_property
  (A B C H : Point)
  (a b c : ℝ)
  (ha hb hc : ℝ)
  (HA HB HC : ℝ) :
  (orthocenter H A B C) ∧ 
  (side_length A B = c) ∧ (side_length B C = a) ∧ (side_length C A = b) ∧
  (altitude_length A B C = ha) ∧ (altitude_length B A C = hb) ∧ (altitude_length C A B = hc) ∧
  (point_distance H A = HA) ∧ (point_distance H B = HB) ∧ (point_distance H C = HC) →
  HA * ha + HB * hb + HC * hc = (a^2 + b^2 + c^2) / 2 := by
  sorry

end orthocenter_property_l178_178463


namespace one_div_a_add_one_div_b_eq_one_l178_178455

noncomputable def a : ℝ := Real.log 10 / Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 5

theorem one_div_a_add_one_div_b_eq_one (h₁ : 2 ^ a = 10) (h₂ : 5 ^ b = 10) :
  (1 / a) + (1 / b) = 1 :=
sorry

end one_div_a_add_one_div_b_eq_one_l178_178455


namespace smallest_n_for_sum_is_24_l178_178112

theorem smallest_n_for_sum_is_24 :
  ∃ (n : ℕ), (0 < n) ∧ 
    (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
    ∀ (m : ℕ), ((0 < m) ∧ 
                (∃ (k' : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = k') → n ≤ m) := sorry

end smallest_n_for_sum_is_24_l178_178112


namespace average_annual_growth_rate_l178_178343

-- Definitions related to the conditions
def initial_sales : ℕ := 50000
def final_sales : ℕ := 72000
def num_periods : ℕ := 2

-- Theorem statement proving the average annual growth rate
theorem average_annual_growth_rate :
  (real.sqrt ((final_sales : ℝ) / (initial_sales : ℝ))) - 1 = 0.2 :=
by
  sorry

end average_annual_growth_rate_l178_178343


namespace smallest_positive_period_of_f_extreme_values_of_f_on_interval_l178_178048

noncomputable def f (x : ℝ) : ℝ :=
  let a : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)
  let b : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)
  a.1 * b.1 + a.2 * b.2

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x ∧ p = Real.pi := sorry

theorem extreme_values_of_f_on_interval :
  ∃ max_val min_val, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
                     (∀ x ∈ Set.Icc 0 (Real.pi / 2), min_val ≤ f x) ∧
                     max_val = 3 ∧ min_val = 0 := sorry

end smallest_positive_period_of_f_extreme_values_of_f_on_interval_l178_178048


namespace product_of_sequence_eq_perm_formula_l178_178303

/-- Define the sequence and use it to prove the given equation. --/
theorem product_of_sequence_eq_perm_formula :
  (∏ i in (finset.range 11).map (nat.add 90), i) = nat.desc_fac 100 11 := 
sorry

end product_of_sequence_eq_perm_formula_l178_178303


namespace find_m_value_l178_178016

variable (α m : ℝ)

-- Conditions
def sin_cos_roots (α : ℝ) (m : ℝ) : Prop :=
  ((Real.sin α, Real.cos α) = (x : ℝ) | x^2 - (sqrt ⟨3⟩ + 1) * x + m / 3)

-- Statement
theorem find_m_value (h1 : α ∈ Ioo 0 (2 * Real.pi)) (h2 : sin_cos_roots α m) : 
  m = 3 * sqrt ⟨3⟩ / 2 :=
by
  -- Skip proof
  sorry

end find_m_value_l178_178016


namespace arrange_elderly_young_people_l178_178228

theorem arrange_elderly_young_people (n r : ℕ) (h : n > 2 * r) : 
  ∃ (num_ways : ℕ), 
  num_ways = (nat.factorial n * nat.factorial (n - r)) / nat.factorial (n - 2 * r) :=
by
  use (nat.factorial n * nat.factorial (n - r)) / nat.factorial (n - 2 * r)
  sorry

end arrange_elderly_young_people_l178_178228


namespace ice_cream_cost_l178_178691

open Real

theorem ice_cream_cost :
  ∀ (a c scoops money_per_person dinner_fraction change_per_person total_money dinner_cost remaining_money spent_on_ice_cream total_scoops price_per_scoop : ℝ),
    a = 40 →
    c = 40 →
    dinner_fraction = 3/4 →
    change_per_person = 1 →
    scoops = 6 →
    total_money = a + c →
    dinner_cost = dinner_fraction * total_money →
    remaining_money = total_money - dinner_cost →
    spent_on_ice_cream = remaining_money - 2 * change_per_person →
    total_scoops = 2 * scoops →
    price_per_scoop = spent_on_ice_cream / total_scoops →
    price_per_scoop = 1.50 :=
by
  intros a c scoops money_per_person dinner_fraction change_per_person total_money dinner_cost remaining_money spent_on_ice_cream total_scoops price_per_scoop
  assume h1 : a = 40,
  assume h2 : c = 40,
  assume h3 : dinner_fraction = 3/4,
  assume h4 : change_per_person = 1,
  assume h5 : scoops = 6,
  assume h6 : total_money = a + c,
  assume h7 : dinner_cost = dinner_fraction * total_money,
  assume h8 : remaining_money = total_money - dinner_cost,
  assume h9 : spent_on_ice_cream = remaining_money - 2 * change_per_person,
  assume h10 : total_scoops = 2 * scoops,
  assume h11 : price_per_scoop = spent_on_ice_cream / total_scoops,
  sorry

end ice_cream_cost_l178_178691


namespace fraction_halfway_between_l178_178983

theorem fraction_halfway_between (a b : ℚ) (h₁ : a = 1 / 6) (h₂ : b = 2 / 5) : (a + b) / 2 = 17 / 60 :=
by {
  sorry
}

end fraction_halfway_between_l178_178983


namespace length_chord_AB_equation_circle_C2_l178_178022

-- Part (1) : Length of chord AB
theorem length_chord_AB (hx hy hr : Real) (d : Real) :
  (x - hx)^2 + (y - hy)^2 + hr + hr = 1 → -- Circle equation transformed
  (hx = 1)∧(hy = 2)∧(hr = 1) →
  d = ∥x + 2*y - 4∥ / sqrt(5) →
  |2 * sqrt(hr^2 - d^2)| = 4*sqrt(5)/5 := sorry

-- Part (2) : Equation of circle C2
theorem equation_circle_C2 (E F : Point) :
  (∃ D E F, ∀ x y, x^2 + y^2 + D*x + E*y + F = 0) →
  (D = 6)∧(E = 0)∧(F = -16) :=
sorry

end length_chord_AB_equation_circle_C2_l178_178022


namespace min_period_of_f_f_monotonically_decreasing_interval_min_value_of_f_in_interval_l178_178790

def f (x : ℝ) := 2 * cos x * (sin x - sqrt 3 * cos x) + sqrt 3

-- Statement for minimum positive period T of f(x)
theorem min_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

-- Statement for interval where f(x) is monotonically decreasing
theorem f_monotonically_decreasing_interval (k : ℤ) :
  ∀ x, k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 → ∀ y, x ≤ y ∧ y ≤ x + π / 6 → f x ≥ f y := sorry

-- Statement for minimum value of f(x) when x ∈ [π/2, π]
theorem min_value_of_f_in_interval :
  ∃ x_min, x_min = 11 * π / 12 ∧ x_min ∈ set.Icc (π / 2) π ∧ f x_min = -2 := sorry

end min_period_of_f_f_monotonically_decreasing_interval_min_value_of_f_in_interval_l178_178790


namespace simplify_complex_fraction_l178_178963

theorem simplify_complex_fraction : let i : ℂ := complex.I in (1 + i) / (1 - i) = i :=
by
  let i : ℂ := complex.I
  sorry

end simplify_complex_fraction_l178_178963


namespace fran_speed_same_distance_as_joann_l178_178873

theorem fran_speed_same_distance_as_joann :
  ∀ (s : ℝ), (12 * 4.5) = (4 * s) → s = 13.5 :=
by
  intro s
  assume h : (12 * 4.5) = (4 * s)
  sorry

end fran_speed_same_distance_as_joann_l178_178873


namespace point_Q_in_third_quadrant_l178_178068

theorem point_Q_in_third_quadrant (m : ℝ) :
  (2 * m + 4 = 0 → (m - 3, m).fst < 0 ∧ (m - 3, m).snd < 0) :=
by
  sorry

end point_Q_in_third_quadrant_l178_178068


namespace find_integers_l178_178745

theorem find_integers (a b : ℤ) (h1 : a * b = a + b) (h2 : a * b = a - b) : a = 0 ∧ b = 0 :=
by 
  sorry

end find_integers_l178_178745


namespace percentage_hate_german_l178_178474

def percentage_hate_math : ℝ := 0.01
def percentage_hate_english : ℝ := 0.02
def percentage_hate_french : ℝ := 0.01
def percentage_hate_all_four : ℝ := 0.08

theorem percentage_hate_german : (0.08 - (0.01 + 0.02 + 0.01)) = 0.04 :=
by
  -- Proof goes here
  sorry

end percentage_hate_german_l178_178474


namespace angle_B_acute_l178_178577

variables {A B C K I : Type} [IsTriangle A B C]

theorem angle_B_acute [IsAngleBisector B K I (10 : 7)] : 
  IsAcute (θ B) := 
by sorry

end angle_B_acute_l178_178577


namespace goldfish_sold_l178_178212

variables (buy_price sell_price tank_cost short_percentage : ℝ)

theorem goldfish_sold (h1 : buy_price = 0.25)
                      (h2 : sell_price = 0.75)
                      (h3 : tank_cost = 100)
                      (h4 : short_percentage = 0.45) :
  let profit_per_goldfish := sell_price - buy_price in
  let shortfall := tank_cost * short_percentage in
  let earnings := tank_cost - shortfall in
  let goldfish_count := earnings / profit_per_goldfish in
  goldfish_count = 110 :=
by {
  let profit_per_goldfish := sell_price - buy_price;
  let shortfall := tank_cost * short_percentage;
  let earnings := tank_cost - shortfall;
  let goldfish_count := earnings / profit_per_goldfish;
  calc goldfish_count
      = earnings / profit_per_goldfish : by exact rfl
  ... = 110 : by sorry
}

end goldfish_sold_l178_178212


namespace sum_alternating_series_l178_178273

theorem sum_alternating_series : 
  ∑ i in (Finset.range 10002).filter (λ i, i % 2 = 0), (if i % 2 = 0 then (i / 2 + 1) else - (i / 2)) = 5001 :=
by
  sorry

end sum_alternating_series_l178_178273
